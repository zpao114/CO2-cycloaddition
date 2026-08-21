# -*- coding: utf-8 -*-
"""
B2: Publication-Year OOD Holdout Benchmark
=========================================

Cross-temporal generalization: train on publications <= YEAR_CUTOFF,
test on publications > YEAR_CUTOFF.

This is a TRUE out-of-distribution (OOD) test because:
1. The test set contains reactions from publications the model has never seen
2. The test set spans a different time period (2021-2026)
3. Different publications may use different catalyst systems, conditions, etc.

Protocol: same as generate_si_s3_benchmark_full_v3_1.py but with
temporal split instead of GroupKFold/LOSO.

Outputs:
- results/results_si/year_ood_benchmark.csv
- results/results_si/year_ood_benchmark_details.csv
- Console summary table

Expected result: significantly lower R2 than LOMO, demonstrating the
year-based generalization gap.
"""
import os
import sys
import io
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import xgboost as xgb
import lightgbm as lgb
import torch
import torch.nn as nn

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
warnings.filterwarnings("ignore")

ROOT = Path(r"D:\machine-learning\CO2-cycloaddition")
PCL_DIR = ROOT / "results_pcl_ae"
SI_DIR = ROOT / "results" / "results_si"
SI_DIR.mkdir(parents=True, exist_ok=True)

N_REPEATS = 3
DEVICE = "cpu"


# =============================================================================
# 1. Extract Year from Raw Data
# =============================================================================
print("=" * 70)
print("[1/5] Loading data and extracting publication year")
print("=" * 70)

df_raw = pd.read_csv(ROOT / "data/raw/CO2_cycloaddition_merged.csv", encoding="gbk")


def extract_year(cond):
    """Extract 4-digit year from Conditions field like '80 C, ... (2015)'"""
    if pd.isna(cond):
        return None
    m = re.search(r'\((\d{4})\)', str(cond))
    return int(m.group(1)) if m else None


df_raw["year_extracted"] = df_raw["Conditions"].apply(extract_year)
df_raw["row_id"] = range(1, len(df_raw) + 1)

# Load cleaned and merge
df = pd.read_csv(ROOT / "data/processed/cleaned.csv", encoding="utf-8-sig")
df = df.dropna(subset=["yield (%)", "reactant_name"]).copy()
df["y_norm"] = df["yield (%)"].clip(0, 100) / 100.0

# Merge year info
df_yr = df_raw[["row_id", "year_extracted"]].copy()
df = df.merge(df_yr, on="row_id", how="left")

# Filter to rows that have features
feat_row_ids = set(
    pd.read_csv(ROOT / "results/results_cho_diagnostic/co2_drfp_xtb_extended.csv")["row_id"].tolist()
)
df = df[df["row_id"].isin(feat_row_ids)].copy()
df = df[df["year_extracted"].between(2000, 2026)].copy()
print(f"  Rows with valid year + features (2000-2026): {len(df)}")
print(f"  Year range: {df.year_extracted.min():.0f} - {df.year_extracted.max():.0f}")


# =============================================================================
# 2. Load Features (same as v3.1)
# =============================================================================
feat_df = pd.read_csv(ROOT / "results/results_cho_diagnostic/co2_drfp_xtb_extended.csv")
feat_df = feat_df.set_index("row_id").loc[df["row_id"].tolist()].reset_index()
assert len(feat_df) == len(df), f"{len(feat_df)} vs {len(df)}"

num_cols = [c for c in feat_df.columns
            if feat_df[c].dtype in [np.float64, np.float32, np.int64, np.int32]
            and c not in ("yield (%)", "yield", "row_id")]
X_num = feat_df[num_cols].fillna(0).to_numpy(dtype=np.float64)

# Align PCL latent with df by row_id
z_pcl_all = np.load(PCL_DIR / "pcl_ae_latent.npy").astype(np.float64)
pcl_row_ids = pd.read_csv(PCL_DIR / "row_id.csv")["row_id"].tolist()
pcl_df = pd.DataFrame({"row_id": pcl_row_ids, "z_idx": range(len(pcl_row_ids))})
df = df.merge(pcl_df, on="row_id", how="left")
assert df["z_idx"].notna().all(), "Some rows missing from pcl_ae_latent.npy"
z_pcl = z_pcl_all[df["z_idx"].values.astype(int)]
df = df.drop(columns=["z_idx"])

X_NUM_DIM = X_num.shape[1]
X_full = np.hstack([X_num, z_pcl])
y = df["y_norm"].to_numpy(dtype=np.float64)
print(f"  X_full={X_full.shape}, X_NUM_DIM={X_NUM_DIM}")


# =============================================================================
# 3. Model Definitions (same as v3.1)
# =============================================================================
class DualBranchANN:
    def __init__(self, n_drfp_dim, n_num_dim, seed=42):
        torch.manual_seed(seed)
        self.net = nn.Sequential(
            nn.Linear(n_num_dim + n_drfp_dim, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid(),
        )

    def fit(self, X_tr, y_tr):
        device = torch.device(DEVICE)
        Xt = torch.from_numpy(X_tr).float().to(device)
        yt = torch.from_numpy(y_tr.reshape(-1, 1)).float().to(device)
        opt = torch.optim.Adam(self.net.parameters(), lr=1e-3, weight_decay=1e-5)
        loss_fn = nn.MSELoss()
        self.net.train()
        n = len(yt)
        bs = 64
        for _ in range(60):
            perm = torch.randperm(n)
            for i in range(0, n, bs):
                idx = perm[i:i+bs]
                opt.zero_grad()
                pred = self.net(Xt[idx])
                loss = loss_fn(pred, yt[idx])
                loss.backward()
                opt.step()
        return self

    def predict(self, X_te):
        self.net.eval()
        with torch.no_grad():
            return self.net(torch.from_numpy(X_te).float()).numpy().flatten()


class ScaledModel:
    def __init__(self, model):
        self.model = model

    def fit(self, X_tr, y_tr):
        self.sc = StandardScaler()
        Xs = self.sc.fit_transform(X_tr).astype(np.float64)
        self.model.fit(Xs, y_tr)
        return self

    def predict(self, X_te):
        Xs = self.sc.transform(X_te).astype(np.float64)
        return self.model.predict(Xs)


def build_model(name, seed, n_features):
    n_drfp_dim = n_features - X_NUM_DIM
    if name == "XGB":
        return xgb.XGBRegressor(
            n_estimators=400, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0,
            min_child_weight=5, random_state=seed, n_jobs=4, verbosity=0,
        )
    if name == "LGBM":
        return lgb.LGBMRegressor(
            n_estimators=400, num_leaves=15, learning_rate=0.05,
            min_child_samples=20, reg_lambda=2.0,
            random_state=seed, n_jobs=4, verbosity=-1,
        )
    if name == "RF":
        return RandomForestRegressor(
            n_estimators=500, max_depth=12, min_samples_leaf=10,
            random_state=seed, n_jobs=4,
        )
    if name == "DualBranchANN":
        return DualBranchANN(n_drfp_dim=n_drfp_dim, n_num_dim=X_NUM_DIM, seed=seed)
    raise ValueError(name)


def evaluate(model, X_tr, y_tr, X_te, y_te):
    model.fit(X_tr, y_tr)
    pred = np.clip(model.predict(X_te), 0, 1)
    r2 = r2_score(y_te, pred)
    mae = mean_absolute_error(y_te, pred)
    rmse = np.sqrt(mean_squared_error(y_te, pred))
    return r2, mae, rmse


# =============================================================================
# 4. Temporal OOD Benchmark
# =============================================================================
MODELS = ["XGB", "LGBM", "RF", "DualBranchANN"]
YEAR_CUTOFF = 2021  # Train <= 2020, Test >= 2021

train_mask = df["year_extracted"] <= YEAR_CUTOFF
test_mask = df["year_extracted"] >= YEAR_CUTOFF + 1

X_tr = X_full[train_mask]
y_tr = y[train_mask]
X_te = X_full[test_mask]
y_te = y[test_mask]

print()
print("=" * 70)
print(f"[2/5] Temporal OOD split: Train <={YEAR_CUTOFF}, Test >={YEAR_CUTOFF+1}")
print("=" * 70)
print(f"  Train: {len(X_tr)} rows, years {df[train_mask].year_extracted.min():.0f}-{df[train_mask].year_extracted.max():.0f}")
print(f"  Test:  {len(X_te)} rows, years {df[test_mask].year_extracted.min():.0f}-{df[test_mask].year_extracted.max():.0f}")
print(f"  Train catalysts: {df[train_mask].catalyst_system_type.value_counts().to_dict()}")
print(f"  Test  catalysts: {df[test_mask].catalyst_system_type.value_counts().to_dict()}")
print()

results = []
all_details = []

print("=" * 70)
print(f"[3/5] Training {len(MODELS)} models x {N_REPEATS} seeds")
print("=" * 70)

for model_name in MODELS:
    print(f"\n--- {model_name} ---", flush=True)
    seed_r2_list = []
    for seed in range(N_REPEATS):
        r2, mae, rmse = evaluate(build_model(model_name, seed, X_full.shape[1]), X_tr, y_tr, X_te, y_te)
        seed_r2_list.append(r2)
        all_details.append({
            "model": model_name,
            "seed": seed,
            "train_n": len(X_tr),
            "test_n": len(X_te),
            "train_years": f"<={YEAR_CUTOFF}",
            "test_years": f">={YEAR_CUTOFF+1}",
            "r2": r2,
            "mae": mae,
            "rmse": rmse,
        })
        print(f"    seed {seed}: R2 = {r2:+.4f}  MAE = {mae:.4f}  RMSE = {rmse:.4f}", flush=True)

    mean_r2 = float(np.mean(seed_r2_list))
    std_r2 = float(np.std(seed_r2_list, ddof=1)) if len(seed_r2_list) > 1 else 0.0
    mean_mae = float(np.mean([d["mae"] for d in all_details if d["model"] == model_name]))
    mean_rmse = float(np.mean([d["rmse"] for d in all_details if d["model"] == model_name]))
    results.append({
        "model": model_name,
        "train_n": len(X_tr),
        "test_n": len(X_te),
        "train_years": f"<={YEAR_CUTOFF}",
        "test_years": f">={YEAR_CUTOFF+1}",
        "r2_mean": mean_r2,
        "r2_std": std_r2,
        "mae_mean": mean_mae,
        "rmse_mean": mean_rmse,
    })
    print(f"  {model_name}: R2 = {mean_r2:+.4f} +/- {std_r2:.4f}")


# =============================================================================
# 5. Save Results
# =============================================================================
print()
print("=" * 70)
print("[4/5] Saving results")
print("=" * 70)

detail_df = pd.DataFrame(all_details)
detail_df.to_csv(SI_DIR / "year_ood_benchmark_details.csv", index=False)
print(f"  Details: {SI_DIR / 'year_ood_benchmark_details.csv'} ({len(detail_df)} rows)")

results_df = pd.DataFrame(results)
results_df.to_csv(SI_DIR / "year_ood_benchmark.csv", index=False)
print(f"  Summary: {SI_DIR / 'year_ood_benchmark.csv'} ({len(results_df)} rows)")


# =============================================================================
# 6. Compare with LOMO
# =============================================================================
print()
print("=" * 70)
print("[5/5] Comparison: OOD (year split) vs LOMO (catalyst split)")
print("=" * 70)

try:
    lomo_df = pd.read_csv(SI_DIR / "lomo_v3_full_results.csv")
    print("\nLOMO results:")
    print(lomo_df[["model", "r2_mean"]].to_string(index=False))
    print("\nYear-OOD results:")
    print(results_df[["model", "r2_mean"]].to_string(index=False))

    cmp = results_df[["model", "r2_mean"]].rename(columns={"r2_mean": "ood_r2"})
    cmp = cmp.merge(
        lomo_df[["model", "r2_mean"]].rename(columns={"r2_mean": "lomo_r2"}),
        on="model"
    )
    cmp["gap"] = cmp["lomo_r2"] - cmp["ood_r2"]
    print("\nGap (LOMO - OOD):")
    print(cmp.to_string(index=False))
except Exception as e:
    print(f"  Could not load LOMO results for comparison: {e}")
    print("  (Run generate_si_s3_benchmark_full_v3_1.py first)")

print()
print("=" * 70)
print("Done")
print("=" * 70)
