# -*- coding: utf-8 -*-
"""
Y-Randomization v4 with 100 Permutations
=========================================

Standalone execution script (no import side effects).
All calculations in __main__ block.

Configuration (consistent with generate_si_s3_benchmark_full_v3_1.py):
- Feature set: F2 = X_num (53 dim) + PCL-AE 128-D latent
- Models: XGB max_depth=4; LGBM num_leaves=15; RF max_depth=12; DualBranchANN
- Protocol: 5-fold GroupKFold by catalyst_system_type
- Real R2: 3-seed average
- Permuted R2: 100 permutations

Inputs:
- data/processed/cleaned.csv
- results/results_cho_diagnostic/co2_drfp_xtb_extended.csv
- results_pcl_ae/pcl_ae_latent.npy

Outputs:
- results/results_y_randomization_v4_100perm/y_randomization_v4_100perm_results.csv
- results/results_y_randomization_v4_100perm/y_randomization_v4_100perm_summary.json
- results/results_y_randomization_v4_100perm/y_randomization_v4_100perm_report.txt

Runtime: ~5 minutes
"""
import os
import sys
import io
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import xgboost as xgb
import lightgbm as lgb
import torch
import torch.nn as nn

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
warnings.filterwarnings("ignore")

ROOT = Path(r"D:\machine-learning\CO2-cycloaddition")
PCL_DIR = ROOT / "results_pcl_ae"
OUT_DIR = ROOT / "results" / "results_y_randomization_v4_100perm"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_PERM = 100
N_REPEATS = 3
N_FOLDS = 5
DEVICE = "cpu"


# =============================================================================
# Model Definitions (consistent with v3.1)
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


def build_model(name, seed, n_features, x_num_dim):
    n_drfp_dim = n_features - x_num_dim
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
        return DualBranchANN(n_drfp_dim=n_drfp_dim, n_num_dim=x_num_dim, seed=seed)
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
    model.fit(X_tr, y_tr)
    pred = np.clip(model.predict(X_te), 0, 1)
    r2 = r2_score(y_te, pred)
    return r2


def cv_r2_mean(model_name, seed, X_full, y, groups_cat, x_num_dim):
    np.random.seed(seed)
    gkf_groups = np.array([f"{c}_{seed}" for c in groups_cat])
    gkf = GroupKFold(n_splits=N_FOLDS)
    r2_list = []
    for tr, te in gkf.split(X_full, y, groups=gkf_groups):
        if len(np.unique(y[te])) < 2 or len(te) < 5:
            continue
        try:
            m = ScaledModel(build_model(model_name, seed, X_full.shape[1], x_num_dim))
            r2 = evaluate(m, X_full[tr], y[tr], X_full[te], y[te])
            r2_list.append(r2)
        except Exception as e:
            print(f"  ! fold error: {e}", flush=True)
    if not r2_list:
        return float("nan")
    return float(np.mean(r2_list))


# =============================================================================
# Main
# =============================================================================
def main():
    t_start = time.time()
    print("=" * 70, flush=True)
    print("[1/4] Loading data (consistent with v3.1)", flush=True)
    print("=" * 70, flush=True)

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

    feat_df = pd.read_csv(ROOT / "results/results_cho_diagnostic/co2_drfp_xtb_extended.csv")
    feat_df = feat_df[feat_df["row_id"].isin(df["row_id"])].copy()
    df = df[df["row_id"].isin(feat_df["row_id"])].copy()
    df = df.set_index("row_id").loc[feat_df["row_id"].tolist()].reset_index()
    print(f"  dataset size: {len(df)} reactions")

    num_cols = [c for c in feat_df.columns
                if feat_df[c].dtype in [np.float64, np.float32, np.int64, np.int32]
                and c not in ("yield (%)", "yield", "row_id")]
    X_num = feat_df[num_cols].fillna(0).to_numpy(dtype=np.float64)

    z_pcl = np.load(PCL_DIR / "pcl_ae_latent.npy").astype(np.float64)
    X_NUM_DIM = X_num.shape[1]
    X_full = np.hstack([X_num, z_pcl])
    y = df["y_norm"].to_numpy(dtype=np.float64)
    groups_cat = df["catalyst_system_type"].to_numpy()
    print(f"  Loaded: X_full={X_full.shape}, X_NUM_DIM={X_NUM_DIM}, "
          f"catalyst types={len(set(groups_cat))}", flush=True)

    print(flush=True)
    print("=" * 70, flush=True)
    print(f"[2/4] Real R2 ({N_REPEATS} seeds) + {N_PERM} permutations", flush=True)
    print("=" * 70, flush=True)

    MODELS = ["XGB", "LGBM", "RF", "DualBranchANN"]
    results = []
    detail_rows = []

    for model_name in MODELS:
        print(f"\n--- {model_name} ---", flush=True)
        t0 = time.time()

        # Real R2: 3-seed average
        real_r2_seeds = []
        for seed in range(N_REPEATS):
            r = cv_r2_mean(model_name, seed, X_full, y, groups_cat, X_NUM_DIM)
            real_r2_seeds.append(r)
            print(f"    seed {seed}: real R2 = {r:+.4f}", flush=True)
        real_r2 = float(np.mean(real_r2_seeds))
        print(f"  real R2 ({N_REPEATS}-seed mean) = {real_r2:+.4f}", flush=True)

        # Permuted R2: N_PERM runs
        perm_r2_list = []
        for perm_id in range(N_PERM):
            rng = np.random.RandomState(2026 + perm_id)
            y_shuf = rng.permutation(y)
            np.random.seed(perm_id)
            gkf_groups = np.array([f"{c}_{perm_id}" for c in groups_cat])
            gkf = GroupKFold(n_splits=N_FOLDS)
            r2_list = []
            for tr, te in gkf.split(X_full, y_shuf, groups=gkf_groups):
                if len(np.unique(y_shuf[te])) < 2 or len(te) < 5:
                    continue
                try:
                    m = ScaledModel(build_model(model_name, perm_id, X_full.shape[1], X_NUM_DIM))
                    r2 = evaluate(m, X_full[tr], y_shuf[tr], X_full[te], y_shuf[te])
                    r2_list.append(r2)
                except Exception:
                    pass
            if r2_list:
                perm_r2 = float(np.mean(r2_list))
                perm_r2_list.append(perm_r2)
                detail_rows.append({"model": model_name, "perm_id": perm_id, "r2": perm_r2})
            if (perm_id + 1) % 10 == 0:
                elapsed = time.time() - t0
                print(f"    perm {perm_id+1}/{N_PERM} done  ({elapsed:.0f}s)", flush=True)

        perm_mean = float(np.mean(perm_r2_list))
        perm_std = float(np.std(perm_r2_list, ddof=1)) if len(perm_r2_list) > 1 else 0.0
        perm_max = float(np.max(perm_r2_list))
        p_value = float((np.sum(np.array(perm_r2_list) >= real_r2) + 1) / (len(perm_r2_list) + 1))
        delta = real_r2 - perm_mean
        print(f"  perm R2 mean={perm_mean:+.4f}  std={perm_std:.4f}  max={perm_max:+.4f}", flush=True)
        print(f"  delta = real - perm = {delta:+.4f}  p = {p_value:.4f}", flush=True)

        results.append({
            "model": model_name,
            "real_r2_seeds": real_r2_seeds,
            "real_r2": real_r2,
            "perm_mean": perm_mean,
            "perm_std": perm_std,
            "perm_max": perm_max,
            "delta_real_vs_perm": delta,
            "p_value": p_value,
            "n_permutations": len(perm_r2_list),
            "pass_threshold_2sigma": bool(delta > 2 * perm_std),
            "y_randomization_pass": bool(delta > 2 * perm_std),
        })

    # =============================================================================
    # 3. Save
    # =============================================================================
    print(flush=True)
    print("=" * 70, flush=True)
    print("[3/4] Saving results", flush=True)
    print("=" * 70, flush=True)

    detail_df = pd.DataFrame(detail_rows)
    detail_df.to_csv(OUT_DIR / "y_randomization_v4_100perm_results.csv", index=False)
    print(f"Saved: {OUT_DIR / 'y_randomization_v4_100perm_results.csv'}  ({len(detail_df)} rows)", flush=True)

    summary = {
        "n_permutations": N_PERM,
        "n_repeats_real": N_REPEATS,
        "n_folds": N_FOLDS,
        "feature_set": "F2 PCL-AE 128-D (X_num + z_pcl)",
        "random_seed_base": 2026,
        "optimization": "v3 tuned hyperparameters",
        "results": results,
    }
    (OUT_DIR / "y_randomization_v4_100perm_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Saved: {OUT_DIR / 'y_randomization_v4_100perm_summary.json'}", flush=True)

    # Text report
    elapsed_total = time.time() - t_start
    lines = []
    lines.append("=" * 70)
    lines.append("v3 Y-Randomization Report")
    lines.append("=" * 70)
    lines.append(f"Feature set: F2 PCL-AE 128-D (X_num + z_pcl)  Total samples: {len(df)}")
    lines.append(f"Protocol: 5-fold GroupKFold by catalyst_system_type")
    lines.append(f"Real R2: {N_REPEATS}-seed mean.  Permuted R2: {N_PERM} permutations")
    lines.append("")
    lines.append(f"{'Model':16s}  {'real R2':>9s}  {'perm mean':>9s}  {'perm std':>8s}  {'delta':>7s}  {'p':>7s}  pass")
    lines.append("-" * 70)
    for r in results:
        lines.append(f"{r['model']:16s}  {r['real_r2']:+.4f}    {r['perm_mean']:+.4f}    "
                     f"{r['perm_std']:.4f}   {r['delta_real_vs_perm']:+.4f}  "
                     f"{r['p_value']:.4f}  "
                     f"{'YES' if r['y_randomization_pass'] else 'NO'}")
    lines.append("")
    lines.append(f"Total runtime: {elapsed_total:.0f}s ({elapsed_total/60:.1f} min)")
    report_path = OUT_DIR / "y_randomization_v4_100perm_report.txt"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved: {report_path}", flush=True)

    print(flush=True)
    print("=" * 70, flush=True)
    print("[4/4] Done", flush=True)
    print("=" * 70, flush=True)
    for r in results:
        print(f"  {r['model']:16s}: real={r['real_r2']:+.4f}  "
              f"perm={r['perm_mean']:+.4f} +/- {r['perm_std']:.4f}  "
              f"delta={r['delta_real_vs_perm']:+.4f}  p={r['p_value']:.4f}  "
              f"pass={'YES' if r['y_randomization_pass'] else 'NO'}", flush=True)
    print(f"\nTotal runtime: {elapsed_total:.0f}s ({elapsed_total/60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
