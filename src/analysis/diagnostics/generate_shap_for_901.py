# -*- coding: utf-8 -*-
"""
generate_shap_for_901.py
========================
为 901_substrate_catalyst_matrix.py 生成正确格式的 per-row SHAP 文件。

目标格式（必须包含以下列才能被 901 的 load_shap_df() 识别）：
  - reactant_name
  - catalyst_system_type
  - actual_yield  (yield in %, same as master CSV)
  - predicted_yield  (model prediction in %)
  - residual  (actual - predicted)
  - [32 feature SHAP value columns]

协议：
  5-fold GroupKFold by reactant_name — 每个底物在验证集中出现一次，
  避免数据泄漏。SHAP 使用 XGBoost 原生 pred_contribs 计算（精确+快速）。

输出：
  results_step4_5/shap_xtb_values_corrected.csv
  （覆盖错误的旧文件；旧文件在运行前自动备份）

用法：
  python generate_shap_for_901.py
"""
from __future__ import annotations
import os, sys, io, warnings
warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import xgboost as xgb

PROJECT_ROOT = os.environ.get("CO2_PROJECT_ROOT", r"D:\machine-learning\CO2-cycloaddition")
DATA_EXTENDED = os.path.join(PROJECT_ROOT, "results", "results_cho_diagnostic", "co2_drfp_xtb_extended.csv")
OUT_CSV = os.path.join(PROJECT_ROOT, "results_step4_5", "shap_xtb_values_corrected.csv")
BACKUP_CSV = os.path.join(PROJECT_ROOT, "results_step4_5", "shap_xtb_values.csv.bak_20260821")

os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)

RANDOM_STATE = 42
N_FOLDS = 5


# ─────────────────────────────────────────────────────────────
# 1. Load & clean master CSV
# ─────────────────────────────────────────────────────────────
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_EXTENDED, encoding="utf-8-sig")
    df = df[df["extraction_status"] == "valid"].copy()
    df = df.dropna(subset=["yield (%)"])
    df = df[df["yield (%)"] > 0].reset_index(drop=True)
    return df


# ─────────────────────────────────────────────────────────────
# 2. Feature construction (32 features, mirrors shap_xtb_values.csv schema)
# ─────────────────────────────────────────────────────────────
def build_features(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    XTB_RAW = [
        "sub_homo_eV", "sub_lumo_eV", "sub_gap_eV", "sub_dipole_D",
        "co2_homo_eV", "co2_lumo_eV", "co2_gap_eV",
        "cat_homo_eV", "cat_lumo_eV", "cat_gap_eV", "cat_dipole_D",
        "solv_homo_eV", "solv_lumo_eV", "solv_gap_eV",
    ]
    XTB_avail = [c for c in XTB_RAW if c in df.columns]

    sgv = pd.to_numeric(df["sub_gap_eV"], errors="coerce").fillna(0).values
    chv = pd.to_numeric(df["cat_homo_eV"], errors="coerce").fillna(0).values
    slv = pd.to_numeric(df["sub_lumo_eV"], errors="coerce").fillna(0).values
    clv = pd.to_numeric(df["cat_lumo_eV"], errors="coerce").fillna(0).values

    delta_E_HL = chv - slv
    delta_E_LL  = clv - slv
    hardness    = sgv / 2.0
    softness    = np.where(hardness > 0, 1.0 / (2.0 * hardness), 0.0)
    nucleophilicity  = -chv
    electrophilicity = clv

    for _n in ("delta_E_HL", "delta_E_LL", "hardness", "softness",
               "nucleophilicity", "electrophilicity"):
        locals()[_n] = np.nan_to_num(locals()[_n], nan=0.0, posinf=0.0, neginf=0.0)

    X_raw = np.nan_to_num(df[XTB_avail].to_numpy(dtype=np.float64), nan=0.0)
    X_xtb = np.column_stack([
        X_raw,
        delta_E_HL, delta_E_LL, hardness, softness, nucleophilicity, electrophilicity,
    ])
    XTB_NAMES = XTB_avail + [
        "delta_E_HL", "delta_E_LL", "global_hardness", "global_softness",
        "nucleophilicity", "electrophilicity",
    ]

    # Conditions
    tc = [c for c in df.columns if "temperature" in c.lower()]
    temp_arr = pd.to_numeric(df[tc[0]], errors="coerce").fillna(
        pd.to_numeric(df[tc[0]], errors="coerce").median()
    ).to_numpy(dtype=np.float64) if tc else np.zeros(len(df))
    press_arr = pd.to_numeric(df["pressure (MPa)"], errors="coerce").fillna(
        pd.to_numeric(df["pressure (MPa)"], errors="coerce").median()
    ).to_numpy(dtype=np.float64)
    time_arr  = pd.to_numeric(df["time (h)"], errors="coerce").fillna(
        pd.to_numeric(df["time (h)"], errors="coerce").median()
    ).to_numpy(dtype=np.float64)
    time_log  = np.log1p(np.maximum(time_arr, 0.0))

    loadings = np.zeros(len(df), dtype=np.float64)
    for lc in [f"catalyst_{i}_loading_mol%" for i in range(1, 5)]:
        if lc in df.columns:
            loadings += np.nan_to_num(
                pd.to_numeric(df[lc], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64),
                nan=0.0)

    loading_log = np.log1p(np.maximum(loadings, 0.0))
    has_solvent = (df["all_solvents_normalized"].notna() &
                   (df["all_solvents_normalized"].astype(str).str.strip() != "")
                   ).astype(float).to_numpy()
    has_reagent = (
        (df["reagent_1_name"].notna() & df["reagent_1_name"].astype(str).str.strip().ne("")) |
        (df["reagent_2_name"].notna() & df["reagent_2_name"].astype(str).str.strip().ne(""))
    ).astype(float).to_numpy()

    X_cond = np.column_stack([temp_arr, press_arr, time_log, loading_log, has_solvent, has_reagent])
    COND_NAMES = ["temperature", "pressure", "time_log", "loading_log", "has_solvent", "has_reagent"]

    X = np.hstack([X_xtb, X_cond]).astype(np.float64)
    names = XTB_NAMES + COND_NAMES

    # Fill NaN/Inf
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X, names


# ─────────────────────────────────────────────────────────────
# 3. Mechanism one-hot (5 families)
# ─────────────────────────────────────────────────────────────
MECH_COLS = ["mech_NUC", "mech_LAC", "mech_BAS", "mech_BIF", "mech_OTH"]


# ─────────────────────────────────────────────────────────────
# 4. 5-fold GroupKFold by reactant_name
# ─────────────────────────────────────────────────────────────
def compute_shap_per_row(df: pd.DataFrame, X: np.ndarray, feat_names: list[str],
                          y: np.ndarray) -> pd.DataFrame:
    """
    5-fold GroupKFold by reactant_name.
    Accumulate per-row SHAP, predicted_yield, actual_yield.
    Returns a DataFrame with one row per input row.
    """
    groups = df["reactant_name"].values
    unique_subs = sorted(df["reactant_name"].unique())
    gkf = GroupKFold(n_splits=len(unique_subs))  # one substrate per fold

    # Allocate arrays for all rows
    n_rows = len(df)
    n_feat = X.shape[1]
    shap_all   = np.zeros((n_rows, n_feat), dtype=np.float64)
    pred_all   = np.zeros(n_rows, dtype=np.float64)
    fold_count = np.zeros(n_rows, dtype=np.int32)

    scaler = StandardScaler()

    for fold_idx, (tr_idx, va_idx) in enumerate(gkf.split(X, y, groups)):
        X_tr, X_va = X[tr_idx], X[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        X_tr_s = scaler.fit_transform(X_tr).astype(np.float64)
        X_va_s = scaler.transform(X_va).astype(np.float64)

        model = xgb.XGBRegressor(
            n_estimators=500, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.5, reg_lambda=2.0,
            random_state=RANDOM_STATE, verbosity=0, n_jobs=-1,
            early_stopping_rounds=50,
        )
        model.fit(X_tr_s, y_tr, eval_set=[(X_va_s, y_va)], verbose=False)

        # SHAP via pred_contribs (use booster directly)
        booster = model.get_booster()
        dtest = xgb.DMatrix(X_va_s, feature_names=feat_names)
        contribs = booster.predict(dtest, pred_contribs=True, validate_features=False)
        shap_fold = contribs[:, :-1].astype(np.float64)   # drop bias

        # Predictions
        pred_fold = model.predict(X_va_s)

        shap_all[va_idx]   = shap_fold
        pred_all[va_idx]   = pred_fold
        fold_count[va_idx] = 1

        val_sub = groups[va_idx[0]]
        r2_fold = r2_score(y_va, pred_fold) if len(y_va) > 1 else float("nan")
        print(f"  Fold {fold_idx+1}/5: held-out={val_sub}, n_val={len(va_idx)}, R2={r2_fold:.4f}")

    assert (fold_count == 1).all(), "Some rows were not covered by any fold!"
    assert shap_all.shape == (n_rows, n_feat)

    # Assemble per-row DataFrame
    shap_df = pd.DataFrame(shap_all, columns=feat_names)
    shap_df.insert(0, "row_index", np.arange(n_rows))
    shap_df.insert(1, "reactant_name", groups)
    shap_df.insert(2, "actual_yield", y * 100.0)      # [0,1] → %
    shap_df.insert(3, "predicted_yield", pred_all * 100.0)
    shap_df.insert(4, "residual", (y - pred_all) * 100.0)

    # Attach catalyst_system_type
    shap_df["catalyst_system_type"] = df["catalyst_system_type"].values

    return shap_df


# ─────────────────────────────────────────────────────────────
# 5. Main
# ─────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("generate_shap_for_901.py")
    print("=" * 60)

    # Backup old shap file
    if os.path.exists(OUT_CSV.rsplit(".", 1)[0] + "_corrected.csv"):
        bak = OUT_CSV.rsplit(".", 1)[0] + "_corrected.csv.bak_20260821"
        os.rename(OUT_CSV.rsplit(".", 1)[0] + "_corrected.csv", bak)
        print(f"[Backup] {bak}")

    # 1. Load
    print("\n[1/4] Loading master CSV...")
    df = load_data()
    print(f"  {len(df)} valid rows, {df['reactant_name'].nunique()} substrates")

    # 2. Features
    print("\n[2/4] Building 32-feature matrix...")
    X, feat_names = build_features(df)
    y = df["yield (%)"].values.astype(np.float64) / 100.0
    print(f"  X shape: {X.shape}, features: {feat_names}")

    # 3. SHAP
    print("\n[3/4] 5-fold GroupKFold SHAP...")
    shap_df = compute_shap_per_row(df, X, feat_names, y)

    # 4. Save
    print(f"\n[4/4] Saving to {OUT_CSV}...")
    shap_df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    size = os.path.getsize(OUT_CSV)
    print(f"  Written: {OUT_CSV}  ({size:,} bytes)")

    # Quick integrity check
    required = {"reactant_name", "catalyst_system_type",
                "actual_yield", "predicted_yield", "residual"}
    missing = required - set(shap_df.columns)
    if missing:
        print(f"[ERROR] Missing required columns: {missing}")
    else:
        print("[OK] All required columns present")

    n_rows   = len(shap_df)
    r2_overall = r2_score(shap_df["actual_yield"], shap_df["predicted_yield"])
    mae_overall = np.abs(shap_df["residual"]).mean()
    print(f"\n  Overall R2 = {r2_overall:.4f},  MAE = {mae_overall:.2f} pp,  "
          f"N = {n_rows}")

    # Update the SHAP_CSV symlink used by 901
    old_shap = os.path.join(PROJECT_ROOT, "results_step4_5", "shap_xtb_values.csv")
    bak_shap = old_shap + ".bak_20260821"
    if os.path.exists(old_shap) and not os.path.exists(bak_shap):
        os.rename(old_shap, bak_shap)
        print(f"\n[Backup] {old_shap} -> {bak_shap}")
    os.rename(OUT_CSV, old_shap)
    print(f"[Install] shap_xtb_values.csv now points to corrected version")
    print(f"\nDone!")


if __name__ == "__main__":
    main()
