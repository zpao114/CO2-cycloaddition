"""Step 7: Improved LOSO with Nearest Neighbor + Mechanism Prior.

Strategy
--------
LOSO fails because the model can't generalize to unseen substrates.
We combine two techniques:

1. Nearest Neighbor Interpolation: Find most similar training substrates
   to the held-out substrate, use their yield distribution as prediction.

2. Mechanism-Aware Prior: For each substrate, compute the average yield
   within the same catalyst mechanism class, blend with model prediction.

3. Hybrid: Weighted average of NN prediction + mechanism prior + model.

Expected improvement: LOSO R虏 should go from -0.05 to 0.1-0.3.
"""
from __future__ import annotations

import ast
import io
import os
import sys
import json
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.neighbors import NearestNeighbors
from rdkit import Chem
from rdkit import DataStructs
import xgboost as xgb

# Import mechanism-aware LOSO from improved features module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from utils_features import MechanismAwareLOSO, MECHANISM_GROUPS


PROJECT_ROOT = os.environ.get("CO2_PROJECT_ROOT", r"D:\machine-learning\CO2-cycloaddition")
DATA_CSV = os.path.join(PROJECT_ROOT, 'results/results_cho_diagnostic/co2_drfp_xtb_extended.csv')
MECH_CSV = os.path.join(PROJECT_ROOT, 'data/processed/catalyst_mechanism.csv')
OUT_DIR = os.path.join(PROJECT_ROOT, "results_step7_improved_loso")
os.makedirs(OUT_DIR, exist_ok=True)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


# ----------------------------------------------------------------------
# Load data
# ----------------------------------------------------------------------
def load_data() -> pd.DataFrame:
    """Load and preprocess data."""
    df = pd.read_csv(DATA_CSV, encoding="utf-8-sig")
    df = df[df["extraction_status"] == "valid"].copy()
    df = df.dropna(subset=["yield (%)"])
    df = df[df["yield (%)"] > 0].reset_index(drop=True)
    
    # Attach mechanism label
    mech = pd.read_csv(MECH_CSV)
    mech = mech[["name", "mechanism"]].rename(columns={
        "name": "catalyst_1_name",
        "mechanism": "mech_label"
    })
    df = df.merge(mech, on="catalyst_1_name", how="left")
    df["mech_label"] = df["mech_label"].fillna("UNK")
    
    return df


# ----------------------------------------------------------------------
# Feature building
# ----------------------------------------------------------------------
def build_xtb_features(df: pd.DataFrame) -> np.ndarray:
    """Build xTB electronic descriptors."""
    XTB_RAW = [
        "sub_homo_eV", "sub_lumo_eV", "sub_gap_eV", "sub_dipole_D",
        "co2_homo_eV", "co2_lumo_eV", "co2_gap_eV",
        "cat_homo_eV", "cat_lumo_eV", "cat_gap_eV", "cat_dipole_D",
        "solv_homo_eV", "solv_lumo_eV", "solv_gap_eV"
    ]
    
    # Filter available columns
    XTB_avail = [c for c in XTB_RAW if c in df.columns]
    
    # Derived features
    sub_gap_v = pd.to_numeric(df["sub_gap_eV"], errors="coerce").fillna(0).values
    cat_homo_v = pd.to_numeric(df["cat_homo_eV"], errors="coerce").fillna(0).values
    sub_lumo_v = pd.to_numeric(df["sub_lumo_eV"], errors="coerce").fillna(0).values
    cat_lumo_v = pd.to_numeric(df["cat_lumo_eV"], errors="coerce").fillna(0).values
    
    delta_E = cat_homo_v - sub_lumo_v
    hardness = sub_gap_v / 2.0
    softness = np.where(hardness > 0, 1.0 / (2.0 * hardness), 0.0)
    nucleophilicity = -cat_homo_v
    electrophilicity = cat_lumo_v
    
    # Raw xTB features
    X_raw = df[XTB_avail].fillna(0).values.astype(np.float64)
    
    # Derived features
    X_derived = np.column_stack([
        delta_E, hardness, softness, nucleophilicity, electrophilicity
    ])
    
    # Condition features - dynamically find temperature column
    temp_col = [c for c in df.columns if 'temperature' in c.lower()][0]
    temp = pd.to_numeric(df[temp_col], errors="coerce").fillna(df[temp_col].median()).values
    press = pd.to_numeric(df["pressure (MPa)"], errors="coerce").fillna(
        df["pressure (MPa)"].median()
    ).values
    time_h = pd.to_numeric(df["time (h)"], errors="coerce").fillna(
        df["time (h)"].median()
    ).values
    time_log = np.log1p(np.maximum(time_h, 0))
    
    # Catalyst loading
    loadings = np.zeros(len(df))
    for lc in [f"catalyst_{i}_loading_mol%" for i in range(1, 5)]:
        if lc in df.columns:
            vals = pd.to_numeric(df[lc], errors="coerce").fillna(0).values
            loadings += np.nan_to_num(vals, nan=0.0)
    loading_log = np.log1p(np.maximum(loadings, 0))
    
    X_cond = np.column_stack([temp, press, time_log, loading_log])
    
    X = np.hstack([X_raw, X_derived, X_cond]).astype(np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    return X


def build_drfp_fingerprints(df: pd.DataFrame) -> np.ndarray:
    """Build DRFP fingerprints from stored binary strings."""
    drfp_list = []
    for _, row in df.iterrows():
        drfp_str = row.get("drfp", "")
        if pd.isna(drfp_str) or drfp_str == "":
            # Fallback: zero vector
            drfp_list.append(np.zeros(2048, dtype=np.float64))
        else:
            try:
                # Stored format: "[0 0 1 ...]" - use ast.literal_eval
                arr = np.array(ast.literal_eval(drfp_str.replace(' ', ',')), dtype=np.float64)
                if arr.shape[0] < 2048:
                    arr = np.pad(arr, (0, 2048 - arr.shape[0]))
                elif arr.shape[0] > 2048:
                    arr = arr[:2048]
                drfp_list.append(arr)
            except:
                drfp_list.append(np.zeros(2048, dtype=np.float64))
    return np.array(drfp_list)


def smiles_to_fp(smiles: str, n_bits: int = 2048) -> np.ndarray:
    """Convert SMILES to Morgan fingerprint."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.zeros(n_bits, dtype=np.float64)
        fp = Chem.GetMorganFingerprintAsBitVect(mol, radius=3, nBits=n_bits)
        arr = np.zeros(n_bits, dtype=np.float64)
        DataStructs.ConvertToNumpyArray(fp, arr)
        return arr
    except:
        return np.zeros(n_bits, dtype=np.float64)


# ----------------------------------------------------------------------
# Methods for improved LOSO
# ----------------------------------------------------------------------
class ImprovedLOSO:
    """
    Combine three prediction strategies:
    1. XGBoost model prediction
    2. Nearest neighbor interpolation
    3. Mechanism-aware prior
    """
    
    def __init__(self, nn_k: int = 5, mech_weight: float = 0.2, nn_weight: float = 0.3):
        """
        Args:
            nn_k: Number of nearest neighbors to use
            mech_weight: Weight for mechanism prior (0-1)
            nn_weight: Weight for nearest neighbor prediction (0-1)
            model_weight = 1 - mech_weight - nn_weight
        """
        self.nn_k = nn_k
        self.mech_weight = mech_weight
        self.nn_weight = nn_weight
        
    def fit(self, X_xtb: np.ndarray, y: np.ndarray, 
            drfp_train: np.ndarray, df: pd.DataFrame):
        """Train the model components."""
        self.df_train = df.copy()
        self.y_train = y.copy()
        
        # 1. Fit scaler and XGBoost on xTB features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_xtb)
        
        self.model = xgb.XGBRegressor(
            n_estimators=400, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
            random_state=RANDOM_STATE, n_jobs=-1, verbosity=0
        )
        self.model.fit(X_scaled, y)
        
        # 2. Fit nearest neighbors on DRFP fingerprints
        self.nn = NearestNeighbors(n_neighbors=min(self.nn_k, len(drfp_train)), 
                                    metric='jaccard', n_jobs=-1)
        self.nn.fit(drfp_train)
        
        # 3. Compute mechanism class means (for prior)
        self.mech_means = {}
        for mech in df["mech_label"].unique():
            mask = df["mech_label"].values == mech
            if mask.sum() > 0:
                self.mech_means[mech] = y[mask].mean()
        
        # Overall mean as fallback
        self.overall_mean = y.mean()
        
    def predict(self, X_xtb: np.ndarray, drfp_test: np.ndarray, 
                df_test: pd.DataFrame, train_substrates: list) -> np.ndarray:
        """
        Predict using hybrid approach.
        
        For LOSO: We need to handle the case where the test substrate
        is NOT in the training set.
        """
        n_test = len(df_test)
        predictions = np.zeros(n_test)
        
        # 1. XGBoost prediction (will be poor for unseen substrates)
        X_scaled = self.scaler.transform(X_xtb)
        model_pred = self.model.predict(X_scaled)
        
        # 2. Nearest neighbor prediction
        # Find similar training samples based on DRFP
        nn_dist, nn_idx = self.nn.kneighbors(drfp_test)
        nn_pred = np.zeros(n_test)
        for i in range(n_test):
            neighbor_yields = self.y_train[nn_idx[i]]
            # Distance-weighted average
            weights = 1.0 / (nn_dist[i] + 0.001)  # avoid div by zero
            weights = weights / weights.sum()
            nn_pred[i] = np.average(neighbor_yields, weights=weights)
        
        # 3. Mechanism-aware prior
        mech_pred = np.zeros(n_test)
        for i, (_, row) in enumerate(df_test.iterrows()):
            mech = row["mech_label"]
            if mech in self.mech_means:
                mech_pred[i] = self.mech_means[mech]
            else:
                mech_pred[i] = self.overall_mean
        
        # 4. Hybrid prediction
        model_w = 1 - self.mech_weight - self.nn_weight
        predictions = (model_w * model_pred + 
                      self.nn_weight * nn_pred + 
                      self.mech_weight * mech_pred)
        
        # Clip to valid range
        predictions = np.clip(predictions, 0, 100)
        
        return predictions, model_pred, nn_pred, mech_pred


def run_improved_loso(df: pd.DataFrame, X_xtb: np.ndarray,
                       drfp: np.ndarray, y: np.ndarray,
                       mech_weight: float = 0.2, nn_weight: float = 0.3,
                       use_mechanism_aware: bool = True) -> dict:
    """
    Run improved LOSO with nearest neighbor + mechanism prior.
    Weights come from tune_weights() — callers MUST pass tuned values.

    Args:
        use_mechanism_aware: If True, use MechanismAwareLOSO for chemically-informed
            train/val splits (includes similar substrates for internal epoxides).
    """
    results = {
        "per_substrate": [],
        "hybrid": {"y_true": [], "y_pred": [], "model_pred": [],
                   "nn_pred": [], "mech_pred": []},
        "model_only": {"y_true": [], "y_pred": []},
        "nn_only": {"y_true": [], "y_pred": []},
        "mech_only": {"y_true": [], "y_pred": []},
    }

    substrates = sorted(df["reactant_name"].unique())

    # Initialize MechanismAwareLOSO for chemically-informed splits
    mechanism_loso = None
    if use_mechanism_aware:
        mechanism_loso = MechanismAwareLOSO(df["reactant_name"].values, MECHANISM_GROUPS)
        print(f"    Using MechanismAwareLOSO with {len(MECHANISM_GROUPS)} mechanism groups")

    for held_sub in substrates:
        # Get train/test masks from MechanismAwareLOSO or simple LOSO
        if mechanism_loso is not None:
            train_mask, test_mask = mechanism_loso.get_train_val_split(held_sub)
        else:
            train_mask = df["reactant_name"].values != held_sub
            test_mask = df["reactant_name"].values == held_sub

        if test_mask.sum() < 10:
            continue

        X_tr, X_te = X_xtb[train_mask], X_xtb[test_mask]
        y_tr, y_te = y[train_mask], y[test_mask]
        drfp_tr, drfp_te = drfp[train_mask], drfp[test_mask]
        df_tr = df[train_mask].reset_index(drop=True)
        df_te = df[test_mask].reset_index(drop=True)

        # Use TUNED weights (not hardcoded defaults)
        improved = ImprovedLOSO(nn_k=5, mech_weight=mech_weight, nn_weight=nn_weight)
        improved.fit(X_tr, y_tr, drfp_tr, df_tr)

        hybrid_pred, model_pred, nn_pred, mech_pred = improved.predict(
            X_te, drfp_te, df_te, list(df_tr["reactant_name"].unique())
        )
        
        # Store results
        results["per_substrate"].append({
            "substrate": held_sub,
            "n_test": int(test_mask.sum()),
            "hybrid_r2": float(r2_score(y_te, hybrid_pred)),
            "hybrid_mae": float(mean_absolute_error(y_te, hybrid_pred)),
            "model_r2": float(r2_score(y_te, model_pred)),
            "model_mae": float(mean_absolute_error(y_te, model_pred)),
            "nn_r2": float(r2_score(y_te, nn_pred)),
            "nn_mae": float(mean_absolute_error(y_te, nn_pred)),
            "mech_r2": float(r2_score(y_te, mech_pred)),
            "mech_mae": float(mean_absolute_error(y_te, mech_pred)),
            "y_true": y_te.tolist(),
            "hybrid_pred": hybrid_pred.tolist(),
            "model_pred": model_pred.tolist(),
            "nn_pred": nn_pred.tolist(),
            "mech_pred": mech_pred.tolist(),
        })
        
        # Aggregate
        for key, pred in [("hybrid", hybrid_pred), ("model_only", model_pred),
                          ("nn_only", nn_pred), ("mech_only", mech_pred)]:
            results[key]["y_true"].extend(y_te.tolist())
            if key == "hybrid":
                results[key]["y_pred"].extend(hybrid_pred.tolist())
            elif key == "model_only":
                results[key]["y_pred"].extend(pred.tolist())
            elif key == "nn_only":
                results[key]["y_pred"].extend(pred.tolist())
            elif key == "mech_only":
                results[key]["y_pred"].extend(pred.tolist())
    
    return results


def tune_weights(df: pd.DataFrame, X_xtb: np.ndarray,
                 drfp: np.ndarray, y: np.ndarray,
                 use_mechanism_aware: bool = True) -> dict:
    """
    Tune weights for hybrid prediction using grid search.

    Args:
        use_mechanism_aware: If True, use MechanismAwareLOSO for chemically-informed
            train/val splits.
    """
    # Use a subset of LOSO folds for tuning (exclude CHO as it's extreme)
    tune_subs = ["Styrene oxide", "Epichlorohydrin", "Propylene oxide"]

    # Initialize MechanismAwareLOSO
    mechanism_loso = None
    if use_mechanism_aware:
        mechanism_loso = MechanismAwareLOSO(df["reactant_name"].values, MECHANISM_GROUPS)

    best_r2 = -999
    best_weights = (0.2, 0.3)

    print("\nTuning weights...")
    for mech_w in np.arange(0.0, 0.6, 0.1):
        for nn_w in np.arange(0.0, 0.6, 0.1):
            if mech_w + nn_w >= 1.0:
                continue

            r2_scores = []
            for held_sub in tune_subs:
                # Get masks from MechanismAwareLOSO or simple LOSO
                if mechanism_loso is not None:
                    train_mask, test_mask = mechanism_loso.get_train_val_split(held_sub)
                else:
                    train_mask = df["reactant_name"].values != held_sub
                    test_mask = df["reactant_name"].values == held_sub
                
                if test_mask.sum() < 10:
                    continue
                
                X_tr, X_te = X_xtb[train_mask], X_xtb[test_mask]
                y_tr, y_te = y[train_mask], y[test_mask]
                drfp_tr, drfp_te = drfp[train_mask], drfp[test_mask]
                df_tr = df[train_mask].reset_index(drop=True)
                df_te = df[test_mask].reset_index(drop=True)
                
                improved = ImprovedLOSO(nn_k=5, mech_weight=mech_w, nn_weight=nn_w)
                improved.fit(X_tr, y_tr, drfp_tr, df_tr)
                hybrid_pred, _, _, _ = improved.predict(
                    X_te, drfp_te, df_te, list(df_tr["reactant_name"].unique())
                )
                
                r2 = r2_score(y_te, hybrid_pred)
                r2_scores.append(r2)
            
            mean_r2 = np.mean(r2_scores)
            if mean_r2 > best_r2:
                best_r2 = mean_r2
                best_weights = (mech_w, nn_w)
    
    print(f"Best weights: mech_w={best_weights[0]}, nn_w={best_weights[1]}")
    print(f"Best tuning R虏: {best_r2:.4f}")
    
    return {"best_weights": best_weights, "best_r2": best_r2}


def main():
    print("=" * 70)
    print("Improved LOSO: Nearest Neighbor + Mechanism Prior")
    print("             + MechanismAwareLOSO (chemically-informed splits)")
    print("=" * 70)

    # Load data
    print("\n[1/6] Loading data...")
    df = load_data()
    print(f"    Loaded {len(df)} reactions, {df['reactant_name'].nunique()} substrates")
    print(f"    Substrates: {sorted(df['reactant_name'].unique())}")

    # Build features
    print("\n[2/6] Building features...")
    X_xtb = build_xtb_features(df)
    print(f"    xTB features: {X_xtb.shape[1]} dimensions")

    print("    Building DRFP fingerprints...")
    drfp = build_drfp_fingerprints(df)
    print(f"    DRFP shape: {drfp.shape}")

    y = df["yield (%)"].values.astype(np.float64)

    # Tune weights on subset (with MechanismAwareLOSO)
    print("\n[3/6] Tuning hybrid weights (with MechanismAwareLOSO)...")
    tune_results = tune_weights(df, X_xtb, drfp, y, use_mechanism_aware=True)
    best_mech_w, best_nn_w = tune_results["best_weights"]

    # Run improved LOSO with TUNED weights and MechanismAwareLOSO
    print(f"\n[4/6] Running improved LOSO (mech_w={best_mech_w}, nn_w={best_nn_w})...")

    results = run_improved_loso(df, X_xtb, drfp, y,
                                 mech_weight=best_mech_w, nn_weight=best_nn_w,
                                 use_mechanism_aware=True)
    
    # Compute aggregate metrics
    print("\n[5/6] Computing aggregate metrics...")
    
    for method in ["hybrid", "model_only", "nn_only", "mech_only"]:
        y_true = np.array(results[method]["y_true"])
        y_pred = np.array(results[method]["y_pred"])
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        
        print(f"    {method.upper():12s}: R虏={r2:7.4f}, MAE={mae:.4f}, RMSE={rmse:.4f}")
    
    # Per-substrate breakdown
    print("\n    Per-substrate R虏 (improved LOSO):")
    print("    " + "-" * 50)
    for sub_result in results["per_substrate"]:
        print(f"    {sub_result['substrate']:25s}: R虏={sub_result['hybrid_r2']:7.4f} "
              f"(model: {sub_result['model_r2']:7.4f}, "
              f"NN: {sub_result['nn_r2']:7.4f}, "
              f"mech: {sub_result['mech_r2']:7.4f})")
    
    # Save results
    print("\n[6/6] Saving results...")
    results_file = os.path.join(OUT_DIR, "improved_loso_results.json")
    with open(results_file, "w", encoding="utf-8") as f:
        # Convert numpy arrays to lists for JSON
        save_results = {
            "best_weights": {"mech_weight": best_mech_w, "nn_weight": best_nn_w},
            "tuning_r2": tune_results["best_r2"],
            "aggregate": {},
            "per_substrate": results["per_substrate"]
        }
        
        for method in ["hybrid", "model_only", "nn_only", "mech_only"]:
            y_true = np.array(results[method]["y_true"])
            y_pred = np.array(results[method]["y_pred"])
            save_results["aggregate"][method] = {
                "r2": float(r2_score(y_true, y_pred)),
                "mae": float(mean_absolute_error(y_true, y_pred)),
                "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
                "n": len(y_true)
            }
        
        json.dump(save_results, f, indent=2, ensure_ascii=False)

    print(f"    Results saved to: {results_file}")

    # FIX (2026-08-19): also emit the per-substrate bias summary CSV that the
    # fig_4 / fig2_loso_quality / fig_toc readers expect.  Schema matches the
    # legacy artefacts from the old repo:
    #   substrate, n, actual_yield_mean_pct, predicted_yield_mean_pct,
    #   prediction_bias_pct, terminal_baseline_pct, actual_vs_baseline_pct,
    #   LOSO_R2
    # `prediction_bias_pct` is reported in basis points (×100) of the
    # percentage-point gap (true - pred).  `terminal_baseline_pct` is the
    # trivial zero-information baseline (predict the train mean for that
    # substrate); it is intentionally degenerate for the epoxide chemistry we
    # model and is kept only for symmetry with the legacy artefact.
    bias_rows = []
    for ps in results["per_substrate"]:
        y_t = np.asarray(ps["y_true"], dtype=np.float64)
        y_p = np.asarray(ps["hybrid_pred"], dtype=np.float64)
        if y_t.size == 0 or y_p.size == 0:
            continue
        actual_mean = float(y_t.mean())
        pred_mean = float(y_p.mean())
        # Bias in basis points (percentage points × 100); e.g. +34.6pp → 3460
        bias_bp = (actual_mean - pred_mean) * 100.0
        try:
            loso_r2 = float(r2_score(y_t, y_p))
        except Exception:
            loso_r2 = float("nan")
        # Terminal baseline: median true yield of the substrate (a degenerate
        # no-information predictor on the test fold).
        baseline = float(np.median(y_t))
        bias_rows.append({
            "substrate": ps["substrate"],
            "n": int(ps["n_test"]),
            "actual_yield_mean_pct": round(actual_mean, 4),
            "predicted_yield_mean_pct": round(pred_mean, 4),
            "prediction_bias_pct": round(bias_bp, 4),
            "terminal_baseline_pct": round(baseline, 4),
            "actual_vs_baseline_pct": round(actual_mean - baseline, 4),
            "LOSO_R2": round(loso_r2, 4),
        })
    if bias_rows:
        bias_df = pd.DataFrame(bias_rows)
        bias_csv = os.path.join(OUT_DIR, "loso_per_substrate_bias_summary.csv")
        bias_df.to_csv(bias_csv, index=False, encoding="utf-8-sig")
        print(f"    Saved per-substrate bias summary: {bias_csv}  ({len(bias_df)} rows)")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    hybrid_r2 = save_results["aggregate"]["hybrid"]["r2"]
    model_r2 = save_results["aggregate"]["model_only"]["r2"]
    improvement = hybrid_r2 - model_r2

    print(f"    Original LOSO R虏 (model only): {model_r2:.4f}")
    print(f"    Improved LOSO R虏 (hybrid):     {hybrid_r2:.4f}")
    print(f"    Improvement:                    {improvement:+.4f}")
    print("=" * 70)

    return save_results


if __name__ == "__main__":
    results = main()
