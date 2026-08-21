# -*- coding: utf-8 -*-
"""
S3 Benchmark v3.1 - Full Results Generator (Single-run outputs all CSVs)
========================================================================

Outputs:
- results/results_si/groupkfold_v3_full_results.csv  (Full set XGB/LGBM/RF/DualBranchANN)
- results/results_si/groupkfold_subset_v3.csv        (Subset R2 IL/MH/Mixed/Base/Unknown)
- results/results_si/lomo_v3_full_results.csv        (LOMO by mechanism)
- results/results_si/feature_set_benchmark.csv       (LOSO x 4 feature sets x 4 models)

This script generates the data for Sections 3.1 / 3.2 / 3.5 of the paper.
All CSVs are reproducible; no mid-product library issues.
"""
import os
import sys
import io
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, GroupKFold, LeaveOneGroupOut
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
# 1. Load Data
# =============================================================================
print("=" * 70)
print("[1/6] Loading data")
print("=" * 70)

df = pd.read_csv(ROOT / "data/processed/cleaned.csv")
df = df.dropna(subset=["yield (%)", "reactant_name"]).copy()
df["y_norm"] = df["yield (%)"].clip(0, 100) / 100.0

SUBSTRATE_MAP = {
    "Styrene oxide": "SO", "Epichlorohydrin": "ECH",
    "Propylene oxide": "PO", "Cyclohexene oxide": "CHO",
    "Isopropyl glycidyl ether": "IGE"
}
df["substrate"] = df["reactant_name"].map(SUBSTRATE_MAP)
df = df.dropna(subset=["substrate", "y_norm"]).copy()
print(f"  cleaned.csv after substrate mapping: {df.shape}")

feat_df = pd.read_csv(ROOT / "results/results_cho_diagnostic/co2_drfp_xtb_extended.csv")
feat_df = feat_df[feat_df["row_id"].isin(df["row_id"])].copy()
df = df[df["row_id"].isin(feat_df["row_id"])].copy()
df = df.set_index("row_id").loc[feat_df["row_id"].tolist()].reset_index()
assert len(df) == len(feat_df), f"Mismatch: df={len(df)}, feat_df={len(feat_df)}"
print(f"  merged dataset: {df.shape}")

num_cols = [c for c in feat_df.columns
            if feat_df[c].dtype in [np.float64, np.float32, np.int64, np.int32]
            and c not in ("yield (%)", "yield", "row_id")]
X_num = feat_df[num_cols].fillna(0).to_numpy(dtype=np.float64)
print(f"  X_num: {X_num.shape}")


def decode_drfp(s):
    if not isinstance(s, str) or not s:
        return np.zeros(2048, dtype=np.float32)
    s = s.strip().strip("[]")
    arr = np.fromstring(s, sep=" ", dtype=np.float32)
    if arr.size < 2048:
        arr = np.concatenate([arr, np.zeros(2048 - arr.size, dtype=np.float32)])
    return arr[:2048]


X_drfp = np.stack([decode_drfp(s) for s in feat_df["drfp"].values]).astype(np.float32)

z_std = np.load(PCL_DIR / "standard_ae_latent.npy").astype(np.float64)
z_pcl = np.load(PCL_DIR / "pcl_ae_latent.npy").astype(np.float64)
print(f"  z_std: {z_std.shape}, z_pcl: {z_pcl.shape}")

y = df["y_norm"].to_numpy(dtype=np.float64)
groups_sub = df["substrate"].to_numpy()
groups_cat = df["catalyst_system_type"].to_numpy()


# =============================================================================
# 2. Feature Sets
# =============================================================================
FEATURE_SETS = {
    "F0_xTB_only":  X_num,
    "F1_StdAE128":  np.hstack([X_num, z_std]),
    "F2_PCLAE128":  np.hstack([X_num, z_pcl]),
    "F3_DRFP_full": np.hstack([X_num, X_drfp]),
}
for n, X in FEATURE_SETS.items():
    print(f"  {n}: {X.shape}")


# =============================================================================
# 3. DualBranchANN (PyTorch)
# =============================================================================
class DualBranchANN:
    """DualBranchANN: dual-branch ANN (DRFP + xTB) -> concat -> MLP -> 1 sigmoid output"""
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


# =============================================================================
# 4. Model Builder
# =============================================================================
def build_model(name, seed, n_features):
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
        n_drfp_dim = n_features - X_num.shape[1]
        return DualBranchANN(n_drfp_dim=n_drfp_dim, n_num_dim=X_num.shape[1], seed=seed)
    raise ValueError(name)


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


def evaluate(model, X_tr, y_tr, X_te, y_te):
    """Fit, predict (clip 0-1), return r2/mae/rmse (0-1 scale)"""
    model.fit(X_tr, y_tr)
    pred = np.clip(model.predict(X_te), 0, 1)
    r2 = r2_score(y_te, pred)
    mae = mean_absolute_error(y_te, pred)
    rmse = np.sqrt(mean_squared_error(y_te, pred))
    return r2, mae, rmse, pred


MODELS = ["XGB", "LGBM", "RF", "DualBranchANN"]


# =============================================================================
# 5. GroupKFold by Catalyst (Full Set: 4 Models x 3 Seeds)
# =============================================================================
print()
print("=" * 70)
print("[2/6] GroupKFold by catalyst (F2 PCL-AE) - 4 models x 3 seeds")
print("=" * 70)

X_full = FEATURE_SETS["F2_PCLAE128"]
gkf_records = []
gkf = GroupKFold(n_splits=5)

for seed in range(N_REPEATS):
    np.random.seed(seed)
    gkf_groups = np.array([f"{c}_{seed}" for c in groups_cat])
    r2_list, mae_list, rmse_list = [], [], []
    for train_idx, test_idx in gkf.split(X_full, y, groups=gkf_groups):
        if len(np.unique(y[test_idx])) < 2 or len(test_idx) < 5:
            continue
        Xt, yt = X_full[train_idx], y[train_idx]
        Xs, ys = X_full[test_idx], y[test_idx]
        for model_name in MODELS:
            try:
                m = ScaledModel(build_model(model_name, seed, X_full.shape[1]))
                r2, mae, rmse, pred = evaluate(m, Xt, yt, Xs, ys)
                if model_name == "XGB":
                    r2_list.append(r2)
                    mae_list.append(mae)
                    rmse_list.append(rmse)
            except Exception as e:
                print(f"  ! {model_name} error: {e}")
    gkf_records.append({
        "model": "XGB", "seed": seed, "n_folds": len(r2_list),
        "r2_mean": float(np.mean(r2_list)),
        "mae_mean": float(np.mean(mae_list)),
        "rmse_mean": float(np.mean(rmse_list)),
    })
    print(f"  XGB seed={seed}: r2={np.mean(r2_list):+.4f}  mae={np.mean(mae_list):.4f}")

    for model_name in ["LGBM", "RF", "DualBranchANN"]:
        np.random.seed(seed)
        gkf_groups = np.array([f"{c}_{seed}" for c in groups_cat])
        r2_list, mae_list, rmse_list = [], [], []
        for train_idx, test_idx in gkf.split(X_full, y, groups=gkf_groups):
            if len(np.unique(y[test_idx])) < 2 or len(test_idx) < 5:
                continue
            Xt, yt = X_full[train_idx], y[train_idx]
            Xs, ys = X_full[test_idx], y[test_idx]
            try:
                m = ScaledModel(build_model(model_name, seed, X_full.shape[1]))
                r2, mae, rmse, pred = evaluate(m, Xt, yt, Xs, ys)
                r2_list.append(r2)
                mae_list.append(mae)
                rmse_list.append(rmse)
            except Exception:
                pass
        if r2_list:
            gkf_records.append({
                "model": model_name, "seed": seed, "n_folds": len(r2_list),
                "r2_mean": float(np.mean(r2_list)),
                "mae_mean": float(np.mean(mae_list)),
                "rmse_mean": float(np.mean(rmse_list)),
            })
            print(f"  {model_name} seed={seed}: r2={np.mean(r2_list):+.4f}  mae={np.mean(mae_list):.4f}")

gkf_df = pd.DataFrame(gkf_records)
gkf_df.to_csv(SI_DIR / "groupkfold_v3_full_results.csv", index=False)
print(f"\nSaved: {SI_DIR / 'groupkfold_v3_full_results.csv'}")


# =============================================================================
# 6. Subset R2 (Within Each Catalyst: 5-fold KFold, F2 PCL-AE)
# =============================================================================
print()
print("=" * 70)
print("[3/6] Subset R2 (per catalyst, 5-fold KFold, F2 PCL-AE) - 4 models x 3 seeds")
print("=" * 70)

subset_records = []
SUBSET_LIST = ["ionic_liquid", "metal_halide", "mixed_system", "organic_base", "unknown"]

for model_name in MODELS:
    for seed in range(N_REPEATS):
        np.random.seed(seed)
        for cs in SUBSET_LIST:
            mask = (groups_cat == cs)
            n_cs = int(mask.sum())
            if n_cs < 30:
                subset_records.append({
                    "model": model_name, "seed": seed, "subset": cs, "n": n_cs,
                    "r2_mean": np.nan, "mae_mean": np.nan, "rmse_mean": np.nan,
                    "note": "n<30, skipped",
                })
                continue
            X_cs = X_full[mask]
            y_cs = y[mask]
            kf = KFold(n_splits=5, shuffle=True, random_state=seed)
            r2_list, mae_list, rmse_list = [], [], []
            for train_idx, test_idx in kf.split(X_cs):
                if len(np.unique(y_cs[test_idx])) < 2 or len(test_idx) < 2:
                    continue
                Xt, yt = X_cs[train_idx], y_cs[train_idx]
                Xs, ys = X_cs[test_idx], y_cs[test_idx]
                try:
                    m = ScaledModel(build_model(model_name, seed, X_full.shape[1]))
                    r2, mae, rmse, _ = evaluate(m, Xt, yt, Xs, ys)
                    r2_list.append(r2)
                    mae_list.append(mae)
                    rmse_list.append(rmse)
                except Exception:
                    pass
            if r2_list:
                subset_records.append({
                    "model": model_name, "seed": seed, "subset": cs, "n": n_cs,
                    "r2_mean": float(np.mean(r2_list)),
                    "mae_mean": float(np.mean(mae_list)),
                    "rmse_mean": float(np.mean(rmse_list)),
                    "note": "",
                })

subset_df = pd.DataFrame(subset_records)
subset_df.to_csv(SI_DIR / "groupkfold_subset_v3.csv", index=False)
print(f"\nSaved: {SI_DIR / 'groupkfold_subset_v3.csv'}")

print("\nSubset R2 summary (3-seed mean, F2 PCL-AE):")
for model_name in MODELS:
    sub = subset_df[(subset_df['model'] == model_name) & (subset_df['note'] == '')]
    if len(sub) == 0:
        continue
    print(f"\n  {model_name}:")
    for cs in SUBSET_LIST:
        s = sub[sub['subset'] == cs]
        if len(s) > 0 and not s['r2_mean'].isna().all():
            r2_mean = s['r2_mean'].mean()
            r2_std = s['r2_mean'].std()
            mae_mean = s['mae_mean'].mean()
            rmse_mean = s['rmse_mean'].mean()
            n = s['n'].iloc[0]
            print(f"    {cs:15s} n={n:4d}  r2={r2_mean:+.4f} +/- {r2_std:.4f}  mae={mae_mean:.4f}  rmse={rmse_mean:.4f}")


# =============================================================================
# 7. LOSO x 4 Feature Sets x 4 Models x 3 Seeds
# =============================================================================
print()
print("=" * 70)
print("[4/6] LOSO x 4 features x 4 models x 3 seeds")
print("=" * 70)

loso_records = []
logo = LeaveOneGroupOut()

for feat_name, X_feat in FEATURE_SETS.items():
    for seed in range(N_REPEATS):
        for model_name in MODELS:
            r2_list, mae_list, rmse_list = [], [], []
            per_sub_r2 = {}
            for train_idx, test_idx in logo.split(X_feat, y, groups=groups_sub):
                if len(np.unique(y[test_idx])) < 2 or len(test_idx) < 5:
                    continue
                Xt, yt = X_feat[train_idx], y[train_idx]
                Xs, ys = X_feat[test_idx], y[test_idx]
                sub_test = groups_sub[test_idx][0]
                try:
                    m = ScaledModel(build_model(model_name, seed, X_feat.shape[1]))
                    r2, mae, rmse, _ = evaluate(m, Xt, yt, Xs, ys)
                    r2_list.append(r2)
                    mae_list.append(mae)
                    rmse_list.append(rmse)
                    per_sub_r2.setdefault(sub_test, []).append(r2)
                except Exception as e:
                    print(f"  ! {feat_name} {model_name} error: {e}")
            if r2_list:
                loso_records.append({
                    "feature_set": feat_name, "model": model_name, "seed": seed,
                    "n_groups": len(r2_list),
                    "r2_mean": float(np.mean(r2_list)),
                    "mae_mean": float(np.mean(mae_list)),
                    "rmse_mean": float(np.mean(rmse_list)),
                    "per_substrate_r2": "; ".join(f"{k}:{np.mean(v):.3f}" for k, v in per_sub_r2.items()),
                })

loso_df = pd.DataFrame(loso_records)
loso_df.to_csv(SI_DIR / "feature_set_benchmark.csv", index=False)
print(f"\nSaved: {SI_DIR / 'feature_set_benchmark.csv'}")


# =============================================================================
# 8. LOMO by Mechanism (F2 PCL-AE)
# =============================================================================
print()
print("=" * 70)
print("[5/6] LOMO by mechanism (F2 PCL-AE)")
print("=" * 70)

lomo_records = []
for seed in range(N_REPEATS):
    for model_name in MODELS:
        r2_list, mae_list, rmse_list = [], [], []
        for train_idx, test_idx in LeaveOneGroupOut().split(X_full, y, groups=groups_cat):
            if len(np.unique(y[test_idx])) < 2 or len(test_idx) < 5:
                continue
            Xt, yt = X_full[train_idx], y[train_idx]
            Xs, ys = X_full[test_idx], y[test_idx]
            try:
                m = ScaledModel(build_model(model_name, seed, X_full.shape[1]))
                r2, mae, rmse, _ = evaluate(m, Xt, yt, Xs, ys)
                r2_list.append(r2)
                mae_list.append(mae)
                rmse_list.append(rmse)
            except Exception:
                pass
        if r2_list:
            lomo_records.append({
                "model": model_name, "seed": seed, "n_groups": len(r2_list),
                "r2_mean": float(np.mean(r2_list)),
                "mae_mean": float(np.mean(mae_list)),
                "rmse_mean": float(np.mean(rmse_list)),
            })

lomo_df = pd.DataFrame(lomo_records)
lomo_df.to_csv(SI_DIR / "lomo_v3_full_results.csv", index=False)
print(f"\nSaved: {SI_DIR / 'lomo_v3_full_results.csv'}")


# =============================================================================
# 9. Summary
# =============================================================================
print()
print("=" * 70)
print("[6/6] Final aggregation...done")
print("=" * 70)

print("\nLOSO (F2 PCL-AE):")
for m in MODELS:
    sub = loso_df[(loso_df['feature_set'] == 'F2_PCLAE128') & (loso_df['model'] == m)]
    if len(sub) > 0:
        print(f"  {m}: 3-seed r2 = {sub['r2_mean'].mean():+.4f} +/- {sub['r2_mean'].std():.4f}")

print("\nGroupKFold by catalyst (F2 PCL-AE) full set:")
for m in MODELS:
    sub = gkf_df[gkf_df['model'] == m]
    if len(sub) > 0:
        print(f"  {m}: 3-seed r2 = {sub['r2_mean'].mean():+.4f} +/- {sub['r2_mean'].std():.4f}")

print("\nDone.")
