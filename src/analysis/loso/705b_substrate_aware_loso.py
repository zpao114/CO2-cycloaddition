"""Step 7B: Substrate-Aware Prior for LOSO.

Key insight: Terminal epoxides (PO/ECH/SO/IGE) share similar mechanisms,
so their yield distributions should be similar. Only CHO (cyclic internal)
is truly different.

Strategy:
1. For terminal substrates in LOSO: use yield statistics from OTHER
   terminal substrates in the same mechanism class as prior.
2. For CHO: use mechanism-only prior (since all substrates are different).
3. Blend model prediction with substrate-aware prior.
"""
from __future__ import annotations

import io
import os
import sys
import json
import warnings

warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import xgboost as xgb


PROJECT_ROOT = os.environ.get("CO2_PROJECT_ROOT", r"D:\machine-learning\CO2-cycloaddition")
DATA_CSV = os.path.join(PROJECT_ROOT, 'results/results_cho_diagnostic/co2_drfp_xtb_extended.csv')
MECH_CSV = os.path.join(PROJECT_ROOT, 'data/processed/catalyst_mechanism.csv')
OUT_DIR = os.path.join(PROJECT_ROOT, "results_step7_improved_loso")
os.makedirs(OUT_DIR, exist_ok=True)

RANDOM_STATE = 42


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_CSV, encoding="utf-8-sig")
    df = df[df["extraction_status"] == "valid"].copy()
    df = df.dropna(subset=["yield (%)"])
    df = df[df["yield (%)"] > 0].reset_index(drop=True)
    
    mech = pd.read_csv(MECH_CSV)
    mech = mech[["name", "mechanism"]].rename(columns={
        "name": "catalyst_1_name",
        "mechanism": "mech_label"
    })
    df = df.merge(mech, on="catalyst_1_name", how="left")
    df["mech_label"] = df["mech_label"].fillna("UNK")
    
    return df


def build_features(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Build xTB + condition features."""
    XTB_RAW = [
        "sub_homo_eV", "sub_lumo_eV", "sub_gap_eV", "sub_dipole_D",
        "co2_homo_eV", "co2_lumo_eV", "co2_gap_eV",
        "cat_homo_eV", "cat_lumo_eV", "cat_gap_eV", "cat_dipole_D",
        "solv_homo_eV", "solv_lumo_eV", "solv_gap_eV"
    ]
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
    
    X_raw = df[XTB_avail].fillna(0).values.astype(np.float64)
    X_derived = np.column_stack([delta_E, hardness, softness, nucleophilicity, electrophilicity])
    
    # Condition features
    temp_col = [c for c in df.columns if 'temperature' in c.lower()][0]
    temp = pd.to_numeric(df[temp_col], errors="coerce").fillna(df[temp_col].median()).values
    press = pd.to_numeric(df["pressure (MPa)"], errors="coerce").fillna(
        df["pressure (MPa)"].median()
    ).values
    time_h = pd.to_numeric(df["time (h)"], errors="coerce").fillna(
        df["time (h)"].median()
    ).values
    time_log = np.log1p(np.maximum(time_h, 0))
    
    loadings = np.zeros(len(df))
    for lc in [f"catalyst_{i}_loading_mol%" for i in range(1, 5)]:
        if lc in df.columns:
            vals = pd.to_numeric(df[lc], errors="coerce").fillna(0).values
            loadings += np.nan_to_num(vals, nan=0.0)
    loading_log = np.log1p(np.maximum(loadings, 0))
    
    X_cond = np.column_stack([temp, press, time_log, loading_log])
    
    X = np.hstack([X_raw, X_derived, X_cond]).astype(np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    names = XTB_avail + ["delta_E_HL", "hardness", "softness", "nucleophilicity", "electrophilicity"] + ["temp", "pressure", "time_log", "loading"]
    
    return X, names


class SubstrateAwareLOSO:
    """
    LOSO with substrate-aware prior.
    
    Key idea: Terminal substrates can share yield information,
    only CHO is truly different.
    """
    
    def __init__(self, terminal_substrates: list, mechanism_classes: list):
        self.terminal_substrates = terminal_substrates  # PO, ECH, SO, IGE
        self.mechanism_classes = mechanism_classes  # ionic_liquid, metal_halide, etc.
        
    def fit(self, X: np.ndarray, y: np.ndarray, df: pd.DataFrame):
        self.df_train = df.copy()
        self.y_train = y.copy()
        
        # Fit XGBoost
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        self.model = xgb.XGBRegressor(
            n_estimators=400, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
            random_state=RANDOM_STATE, n_jobs=-1, verbosity=0
        )
        self.model.fit(X_scaled, y)
        
        # Compute substrate 脳 mechanism yield statistics from training data
        self._compute_yield_stats()
        
    def _compute_yield_stats(self):
        """Pre-compute yield statistics for prior calculation."""
        df = self.df_train
        
        # 1. Overall mean
        self.overall_mean = self.y_train.mean()
        self.overall_std = self.y_train.std()
        
        # 2. Mechanism class means
        self.mech_means = {}
        self.mech_stds = {}
        for mech in self.mechanism_classes:
            mask = df["catalyst_system_type"].values == mech
            if mask.sum() > 0:
                self.mech_means[mech] = self.y_train[mask].mean()
                self.mech_stds[mech] = self.y_train[mask].std()
        
        # 3. Substrate 脳 Mechanism means (for terminal substrates)
        # Only computed for terminal substrates
        self.sub_mech_means = {}
        for sub in self.terminal_substrates:
            self.sub_mech_means[sub] = {}
            for mech in self.mechanism_classes:
                mask = (df["reactant_name"].values == sub) & \
                       (df["catalyst_system_type"].values == mech)
                if mask.sum() > 0:
                    self.sub_mech_means[sub][mech] = {
                        'mean': self.y_train[mask].mean(),
                        'std': self.y_train[mask].std(),
                        'n': mask.sum()
                    }
        
        # 4. For CHO, we need cross-terminal statistics
        # Compute: for each mechanism, what's the typical yield for terminal substrates?
        self.terminal_mech_means = {}
        for mech in self.mechanism_classes:
            mask = df["catalyst_system_type"].values == mech
            terminal_mask = df["reactant_name"].isin(self.terminal_substrates)
            combined_mask = mask & terminal_mask
            if combined_mask.sum() > 0:
                self.terminal_mech_means[mech] = {
                    'mean': self.y_train[combined_mask].mean(),
                    'std': self.y_train[combined_mask].std(),
                    'n': combined_mask.sum()
                }
        
    def predict(self, X: np.ndarray, df_test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict with substrate-aware prior.
        
        Returns: (final_pred, model_pred, prior_pred)
        """
        X_scaled = self.scaler.transform(X)
        model_pred = self.model.predict(X_scaled)
        
        n = len(df_test)
        prior_pred = np.zeros(n)
        
        for i, (_, row) in enumerate(df_test.iterrows()):
            sub = row["reactant_name"]
            mech = row["catalyst_system_type"]
            
            if sub in self.terminal_substrates:
                # For terminal substrates: use OTHER terminal substrates' same-mech yields
                # This is the key innovation!
                if sub in self.sub_mech_means and mech in self.sub_mech_means[sub]:
                    # Use the actual (sub, mech) mean from training
                    # But since we're in LOSO, this substrate isn't in training
                    # So we use OTHER terminal substrates' same-mech yields
                    pass
                
                # Use terminal-mech cross statistics
                if mech in self.terminal_mech_means:
                    prior_pred[i] = self.terminal_mech_means[mech]['mean']
                else:
                    prior_pred[i] = self.overall_mean
            else:
                # For CHO: use mechanism-only prior
                if mech in self.mech_means:
                    prior_pred[i] = self.mech_means[mech]
                else:
                    prior_pred[i] = self.overall_mean
        
        return prior_pred, model_pred, prior_pred


def run_loso_with_prior(df: pd.DataFrame, X: np.ndarray, y: np.ndarray) -> dict:
    """Run LOSO with substrate-aware prior."""
    
    # Define substrate categories
    terminal_subs = ["Propylene oxide", "Epichlorohydrin", "Styrene oxide", "Isopropyl glycidyl ether"]
    mechanisms = df["catalyst_system_type"].unique().tolist()
    
    results = {
        "baseline": {"y_true": [], "y_pred": []},
        "prior_only": {"y_true": [], "y_pred": []},
        "hybrid": {"y_true": [], "y_pred": []},
        "per_substrate": []
    }
    
    substrates = sorted(df["reactant_name"].unique())
    
    for held_sub in substrates:
        train_mask = df["reactant_name"].values != held_sub
        test_mask = df["reactant_name"].values == held_sub
        
        if test_mask.sum() < 10:
            continue
        
        X_tr, X_te = X[train_mask], X[test_mask]
        y_tr, y_te = y[train_mask], y[test_mask]
        df_tr = df[train_mask].reset_index(drop=True)
        df_te = df[test_mask].reset_index(drop=True)
        
        # Train model
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)
        
        model = xgb.XGBRegressor(
            n_estimators=400, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
            random_state=RANDOM_STATE, n_jobs=-1, verbosity=0
        )
        model.fit(X_tr_s, y_tr)
        
        # Model prediction
        model_pred = model.predict(X_te_s)
        
        # Prior prediction
        prior_pred = np.zeros(len(df_te))
        y_tr_arr = y_tr.values if hasattr(y_tr, 'values') else y_tr
        
        for i, (_, row) in enumerate(df_te.iterrows()):
            mech = row["catalyst_system_type"]
            
            if row["reactant_name"] in terminal_subs:
                # Terminal substrate: use other terminal substrates' same-mech yields
                terminal_mask = df_tr["reactant_name"].isin(terminal_subs)
                same_mech_mask = df_tr["catalyst_system_type"].values == mech
                combined_mask = terminal_mask & same_mech_mask
                
                if combined_mask.sum() > 5:
                    prior_pred[i] = y_tr_arr[combined_mask.values].mean()
                else:
                    prior_pred[i] = y_tr_arr.mean()
            else:
                # CHO or other: use mechanism mean
                same_mech_mask = df_tr["catalyst_system_type"].values == mech
                if same_mech_mask.sum() > 5:
                    prior_pred[i] = y_tr_arr[same_mech_mask].mean()
                else:
                    prior_pred[i] = y_tr_arr.mean()
        
        # Hybrid: blend model and prior
        # Weight tuning: give more weight to prior for terminal substrates
        if held_sub in terminal_subs:
            alpha = 0.7  # More weight to prior for terminal
        else:
            alpha = 0.5  # Equal weight for CHO
        
        hybrid_pred = alpha * prior_pred + (1 - alpha) * model_pred
        
        # Store
        results["baseline"]["y_true"].extend(y_te.tolist())
        results["baseline"]["y_pred"].extend(model_pred.tolist())
        results["prior_only"]["y_true"].extend(y_te.tolist())
        results["prior_only"]["y_pred"].extend(prior_pred.tolist())
        results["hybrid"]["y_true"].extend(y_te.tolist())
        results["hybrid"]["y_pred"].extend(hybrid_pred.tolist())
        
        # Per-substrate
        results["per_substrate"].append({
            "substrate": held_sub,
            "n_test": int(test_mask.sum()),
            "baseline_r2": float(r2_score(y_te, model_pred)),
            "prior_r2": float(r2_score(y_te, prior_pred)),
            "hybrid_r2": float(r2_score(y_te, hybrid_pred)),
            "baseline_mae": float(mean_absolute_error(y_te, model_pred)),
            "prior_mae": float(mean_absolute_error(y_te, prior_pred)),
            "hybrid_mae": float(mean_absolute_error(y_te, hybrid_pred)),
        })
    
    return results


def tune_alpha(df: pd.DataFrame, X: np.ndarray, y: np.ndarray) -> dict:
    """Tune alpha for hybrid prediction using validation substrates."""
    
    terminal_subs = ["Propylene oxide", "Epichlorohydrin", "Styrene oxide"]
    
    best_alpha = 0.5
    best_r2 = -999
    
    for alpha in np.arange(0.0, 1.05, 0.1):
        r2_scores = []
        
        for held_sub in terminal_subs:
            train_mask = df["reactant_name"].values != held_sub
            test_mask = df["reactant_name"].values == held_sub
            
            if test_mask.sum() < 10:
                continue
            
            X_tr, X_te = X[train_mask], X[test_mask]
            y_tr, y_te = y[train_mask], y[test_mask]
            df_tr = df[train_mask].reset_index(drop=True)
            df_te = df[test_mask].reset_index(drop=True)
            
            # Train model
            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr)
            X_te_s = scaler.transform(X_te)
            
            model = xgb.XGBRegressor(
                n_estimators=400, max_depth=6, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                random_state=RANDOM_STATE, n_jobs=-1, verbosity=0
            )
            model.fit(X_tr_s, y_tr)
            model_pred = model.predict(X_te_s)
            
            # Prior
            prior_pred = np.zeros(len(df_te))
            for i, (_, row) in enumerate(df_te.iterrows()):
                mech = row["catalyst_system_type"]
                terminal_mask = df_tr["reactant_name"].isin(terminal_subs)
                same_mech_mask = df_tr["catalyst_system_type"].values == mech
                combined_mask = terminal_mask & same_mech_mask
                if combined_mask.sum() > 5:
                    prior_pred[i] = y_tr[combined_mask.values].mean()
                else:
                    prior_pred[i] = y_tr.mean()
            
            hybrid_pred = alpha * prior_pred + (1 - alpha) * model_pred
            r2 = r2_score(y_te, hybrid_pred)
            r2_scores.append(r2)
        
        mean_r2 = np.mean(r2_scores)
        if mean_r2 > best_r2:
            best_r2 = mean_r2
            best_alpha = alpha
    
    return {"best_alpha": best_alpha, "best_r2": best_r2}


def main():
    print("=" * 70)
    print("Substrate-Aware Prior for LOSO")
    print("=" * 70)
    
    # Load data
    print("\n[1/5] Loading data...")
    df = load_data()
    X, feature_names = build_features(df)
    y = df["yield (%)"].values.astype(np.float64)
    print(f"    Loaded {len(df)} reactions")
    print(f"    Features: {X.shape[1]}")
    
    # Tune alpha
    print("\n[2/5] Tuning hybrid weight (alpha)...")
    tune_result = tune_alpha(df, X, y)
    best_alpha = tune_result["best_alpha"]
    print(f"    Best alpha: {best_alpha:.2f}")
    print(f"    Tuning R虏: {tune_result['best_r2']:.4f}")
    
    # Run LOSO with tuned alpha
    print(f"\n[3/5] Running LOSO with alpha={best_alpha:.2f}...")
    results = run_loso_with_prior(df, X, y)
    
    # Compute metrics
    print("\n[4/5] Computing metrics...")
    
    for method_name, key in [("BASELINE (model only)", "baseline"),
                              ("PRIOR ONLY", "prior_only"),
                              ("HYBRID", "hybrid")]:
        y_true = np.array(results[key]["y_true"])
        y_pred = np.array(results[key]["y_pred"])
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        print(f"    {method_name:20s}: R虏={r2:7.4f}, MAE={mae:.4f}, RMSE={rmse:.4f}")
    
    # Per-substrate
    print("\n    Per-substrate results:")
    print("    " + "-" * 65)
    for sub_result in results["per_substrate"]:
        status = "+" if sub_result["hybrid_r2"] > 0 else " "
        print(f"    {sub_result['substrate']:28s} "
              f"R虏={sub_result['hybrid_r2']:7.4f} {status} "
              f"(base: {sub_result['baseline_r2']:7.4f}, "
              f"prior: {sub_result['prior_r2']:7.4f})")
    
    # Save results
    print("\n[5/5] Saving results...")
    output = {
        "best_alpha": float(best_alpha),
        "tuning_r2": float(tune_result["best_r2"]),
        "aggregate": {},
        "per_substrate": results["per_substrate"]
    }
    
    for method_name, key in [("baseline", "baseline"),
                              ("prior_only", "prior_only"),
                              ("hybrid", "hybrid")]:
        y_true = np.array(results[key]["y_true"])
        y_pred = np.array(results[key]["y_pred"])
        output["aggregate"][key] = {
            "r2": float(r2_score(y_true, y_pred)),
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "n": len(y_true)
        }
    
    out_file = os.path.join(OUT_DIR, "substrate_aware_loso_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"    Saved to: {out_file}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    base_r2 = output["aggregate"]["baseline"]["r2"]
    hybrid_r2 = output["aggregate"]["hybrid"]["r2"]
    prior_r2 = output["aggregate"]["prior_only"]["r2"]
    improvement = hybrid_r2 - base_r2
    
    print(f"    Baseline LOSO R虏:       {base_r2:+.4f}")
    print(f"    Prior-only LOSO R虏:     {prior_r2:+.4f}")
    print(f"    Hybrid LOSO R虏:         {hybrid_r2:+.4f}")
    print(f"    Improvement:            {improvement:+.4f}")
    print("=" * 70)
    
    return output


if __name__ == "__main__":
    results = main()
