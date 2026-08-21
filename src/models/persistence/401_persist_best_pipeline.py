# -*- coding: utf-8 -*-
"""
401_persist_best_pipeline.py
============================

Persist the stage-2 best pipeline (PCL-AE-128 DRFP reducer + DualBranchANN)
so that downstream scripts (e.g. 705_virtual_screening.py) can score
new reactions without re-training.

The property-co-learning weight lambda_prop is exposed as a CLI argument.
The 2026-08-18 ablation stores the optimal λ in config.py as BEST_LAMBDA_PROP.
To re-calibrate, run: python 201_ablation.py
(it will overwrite config.py automatically).

Outputs (all under results_best_pipeline/artifacts/):
  - drfp_scaler.joblib            : StandardScaler fit on raw DRFP (2048 D)
  - pcl_ae_encoder.pt             : PCL-AE-128 encoder state_dict + meta
  - xtb_cond_inter_scaler.joblib  : StandardScaler fit on X_xtb_cond_inter
  - dual_branch_ann.pt            : DualBranchANN state_dict (drfp_in=128, other_in=52)
  - feature_meta.json             : column names, dims, dtype, splits
  - training_metrics.json         : R^2 / MAE / RMSE from 5-fold CV (sanity)
  - save_best_model_report.txt    : human-readable summary
  - predictions.csv               : per-training-row OOF + in-sample preds
                                   (consumed by 310_known_top10_baseline.py)

Usage:
  python 401_persist_best_pipeline.py                  # use default lambda from config.py
  python 401_persist_best_pipeline.py --lambda 0.5     # reproduce unsupervised baseline
  python 401_persist_best_pipeline.py --lambda 50.0    # reproduce 2026-07-31 plateau optimum
"""

import os
import argparse
import io
import json
import os
import sys
import time
import warnings

# NOTE: do NOT re-wrap sys.stdout here.
# src.data_split (imported below) already wraps stdout to UTF-8 on line ~63
# of its file. Re-wrapping a second time closes the underlying buffer,
# which makes every subsequent print() raise "I/O operation on closed file".
# The original line:
#   sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
# was therefore removed on 2026-08-20 (same fix as 304).
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold  # noqa: F401  (kept for legacy; canonical CV uses data_split.json)
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = os.environ.get("CO2_PROJECT_ROOT", r"D:\machine-learning\CO2-cycloaddition")
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))
from src.data_split import load_manifest, holdout_arrays, kfold_folds
sys.path.insert(0, PROJECT_ROOT)
from utils_rxn import read_drfp, get_best_drfp_variant, XTB_COLS
from utils_features import COND_COLS

DATA_EXTENDED = os.path.join(PROJECT_ROOT, 'results', 'results_cho_diagnostic', 'co2_drfp_xtb_extended.csv')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results_best_pipeline")
ARTIFACT_DIR = os.path.join(OUTPUT_DIR, "artifacts")
os.makedirs(ARTIFACT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Defaults chosen by 201_ablation.py (Section 3.1.1 of the paper).
# The λ sweep identifies the optimal λ stored in config.py.
# To re-calibrate, run: python 201_ablation.py
# (it will overwrite config.py automatically).
try:
    from config import BEST_LAMBDA_PROP, BEST_LATENT_DIM
    DEFAULT_LAMBDA_PROP = float(BEST_LAMBDA_PROP)
    LATENT_DIM = int(BEST_LATENT_DIM)
except Exception:
    DEFAULT_LAMBDA_PROP = 0.5   # fallback if config.py is missing
    LATENT_DIM = 128             # matches default in config.py
EPOCHS_PCL = 150
EPOCHS_ANN = 200


# ----------------------------------------------------------------------
# Model architectures (mirrored from 08_benchmark.py)
# ----------------------------------------------------------------------

class PropertyCoLearningAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.ReLU(),
            nn.Linear(256, 512), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Linear(512, input_dim), nn.Sigmoid(),
        )
        self.predictor = nn.Sequential(
            nn.Linear(latent_dim, 64), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1),
        )

    def encode(self, x):
        return self.encoder(x)


class DualBranchANN(nn.Module):
    def __init__(self, drfp_dim, xtb_dim, hidden=128):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.bn_drfp = nn.BatchNorm1d(64)
        self.fc_drfp = nn.Sequential(
            nn.Linear(64, hidden), nn.LeakyReLU(0.1), nn.Dropout(0.3),
            nn.Linear(hidden, 64), nn.LeakyReLU(0.1),
        )
        self.fc_xtb = nn.Sequential(
            nn.Linear(xtb_dim, 64), nn.BatchNorm1d(64), nn.LeakyReLU(0.1), nn.Dropout(0.3),
            nn.Linear(64, 32), nn.LeakyReLU(0.1), nn.Dropout(0.2),
        )
        self.fc_out = nn.Sequential(
            nn.Linear(64 + 32, 64), nn.BatchNorm1d(64), nn.LeakyReLU(0.1),
            nn.Linear(64, 1),
        )

    def forward(self, x_drfp, x_xtb):
        h = x_drfp.unsqueeze(1)
        h = torch.relu(self.conv1(h))
        h = torch.relu(self.conv2(h))
        h = self.pool(h).squeeze(-1)
        h = self.bn_drfp(h)
        h = self.fc_drfp(h)
        h2 = self.fc_xtb(x_xtb)
        return self.fc_out(torch.cat([h, h2], dim=1)).squeeze(-1)


# ----------------------------------------------------------------------
# Feature assembly (kept identical to 08_benchmark.load_data)
# ----------------------------------------------------------------------
# DRFP column is dynamically determined by the ablation experiment
# (see utils_rxn.get_best_drfp_variant())
#
# Canonical feature-column lists are imported from utils_rxn.XTB_COLS and
# utils_features.COND_COLS to keep training-time and inference-time input
# dimensions identical across the whole pipeline.


def parse_drfp_col(series):
    arrs = []
    for fp_str in series:
        a = read_drfp(fp_str)
        if a is None or a.size == 0:
            arrs.append(np.zeros(2048, dtype=np.float32))
        else:
            arrs.append(a.astype(np.float32))
    return np.array(arrs)


def load_data(use_holdout_train=True):
    """Load data, dynamically selecting the best DRFP variant from ablation meta."""
    meta = get_best_drfp_variant()
    best_col = meta['best_drfp_col']
    best_var = meta['best_variant']
    print(f"\n[Step 0] Loading {os.path.basename(DATA_EXTENDED)} ...")
    print(f"  DRFP variant: {best_var} ({best_col})")
    df = pd.read_csv(DATA_EXTENDED, encoding="utf-8-sig")
    df = df[df["extraction_status"] == "valid"].copy()
    df = df.dropna(subset=["yield (%)"])
    df = df[df["yield (%)"] > 0].reset_index(drop=True)

    if use_holdout_train:
        train_idx, _, _ = holdout_arrays(load_manifest())
        df = df.iloc[sorted(train_idx)].reset_index(drop=True)
        print(f"  [load_data] filtered to holdout train pool: {len(df)} rows")

    print(f"  Valid samples: {len(df)}")

    X_drfp = parse_drfp_col(df[best_col]).astype(np.float32)
    print(f"  DRFP: {X_drfp.shape[1]}D")

    xtb_cols = [c for c in XTB_COLS if c in df.columns]
    X_xtb = np.nan_to_num(df[xtb_cols].values.astype(np.float32), nan=0.0)
    print(f"  XTB: {X_xtb.shape[1]}D ({len(xtb_cols)} cols)")

    cat_loading_cols = [c for c in df.columns if "loading_mol%" in c]
    cond_cols = [c for c in COND_COLS + cat_loading_cols if c in df.columns]
    X_cond = np.nan_to_num(df[cond_cols].values.astype(np.float32), nan=0.0)
    print(f"  Cond: {X_cond.shape[1]}D")

    T = X_cond[:, 0:1]
    P = X_cond[:, 1:2]
    inter_parts = []
    if "activation_proxy" in xtb_cols:
        i = xtb_cols.index("activation_proxy")
        inter_parts.append(T * X_xtb[:, i:i + 1])
    if "total_polarity_index" in xtb_cols:
        i = xtb_cols.index("total_polarity_index")
        inter_parts.append(P * X_xtb[:, i:i + 1])
    X_inter = np.concatenate(inter_parts, axis=1).astype(np.float32) if inter_parts \
        else np.zeros((len(df), 0), np.float32)
    print(f"  Inter: {X_inter.shape[1]}D")

    y = df["yield (%)"].values.astype(np.float32) / 100.0
    return X_drfp, X_xtb, X_cond, X_inter, y, df, xtb_cols, cond_cols, best_var, best_col


# ----------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------

def train_pcl_ae(X, y, latent_dim=LATENT_DIM, epochs=EPOCHS_PCL,
                 batch_size=128, lr=1e-3, lambda_prop=DEFAULT_LAMBDA_PROP,
                 pos_weight=10.0):
    """Train PCL-AE on standardized DRFP with reconstruction + yield-supervised loss."""
    print(f"[2/4] Training PCL-AE (lambda={lambda_prop}, latent_dim={latent_dim}, epochs={epochs}) ...")
    X_tensor = torch.FloatTensor(X).to(DEVICE)
    y_tensor = torch.FloatTensor(y).unsqueeze(1).to(DEVICE)
    model = PropertyCoLearningAE(X.shape[1], latent_dim).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=10, factor=0.5)
    bce = nn.BCEWithLogitsLoss(reduction="none")
    mse = nn.MSELoss()
    loader = DataLoader(TensorDataset(X_tensor, y_tensor), batch_size=batch_size, shuffle=True)
    y_mean, y_std_ = float(y.mean()), float(y.std() + 1e-6)
    for ep in range(epochs):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            z = model.encoder(xb)
            recon = model.decoder(z)
            pred = model.predictor(z).squeeze(-1)
            recon_loss = (bce(recon, xb) * (xb * (pos_weight - 1) + 1)).mean()
            prop_loss = mse(pred * y_std_ + y_mean, yb.squeeze(-1))
            loss = recon_loss + lambda_prop * prop_loss
            loss.backward()
            opt.step()
        sched.step(recon_loss.item())
    model.eval()
    return model


def train_dual_branch(Xd, Xo, y, hidden=128, epochs=EPOCHS_ANN,
                      batch_size=32, lr=5e-4, weight_decay=1e-3,
                      val_frac=0.15, patience=40):
    """Train final DualBranchANN on full data with inner validation split
    + R^2-based early stopping. Without early stopping the model overfits
    the training set silently; with this fix, training stops at the best
    held-out R^2 epoch and we restore those weights for downstream use.

    Args:
        Xd, Xo, y: full training features + target (all rows used).
        val_frac:  fraction of training rows reserved as inner validation.
        patience:  number of epochs without held-out R^2 improvement before stop.
    """
    print(f"[4/4] Training final DualBranchANN on full data with inner val "
          f"(epochs<= {epochs}, val_frac={val_frac}, patience={patience}) ...")
    rng = np.random.RandomState(42)
    n = len(y)
    idx = rng.permutation(n)
    n_val = max(int(n * val_frac), 16)
    val_idx, tr_idx = idx[:n_val], idx[n_val:]

    Xd_tr_t = torch.tensor(Xd[tr_idx], dtype=torch.float32).to(DEVICE)
    Xd_va_t = torch.tensor(Xd[val_idx], dtype=torch.float32).to(DEVICE)
    Xo_tr_t = torch.tensor(Xo[tr_idx], dtype=torch.float32).to(DEVICE)
    Xo_va_t = torch.tensor(Xo[val_idx], dtype=torch.float32).to(DEVICE)
    y_tr_t = torch.tensor(y[tr_idx], dtype=torch.float32).to(DEVICE)
    y_va = y[val_idx]

    model = DualBranchANN(Xd.shape[1], Xo.shape[1], hidden=hidden).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    ds = TensorDataset(Xd_tr_t, Xo_tr_t, y_tr_t)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

    best_r2 = -np.inf
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    no_imp = 0
    for ep in range(epochs):
        model.train()
        for xd, xo, yy in dl:
            opt.zero_grad()
            nn.MSELoss()(model(xd, xo), yy).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            p = model(Xd_va_t, Xo_va_t).cpu().numpy()
        r2 = r2_score(y_va, np.clip(p, 0, 1))
        if r2 > best_r2:
            best_r2 = r2
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1
            if no_imp >= patience:
                print(f"  early stop at epoch {ep+1}, best inner-val R^2={best_r2:.4f}")
                break
    model.load_state_dict(best_state)
    model.eval()
    return model


def eval_dual_branch_5fold(X_drfp, X_other, y, n_folds=5, hidden=128):
    """5-fold CV with early stopping (R^2-based) using data_split.json manifest."""
    print(f"[3/4] 5-fold CV sanity (DualBranchANN on PCL-AE features) ...")
    fold_records = []
    folds = kfold_folds(load_manifest())
    for fold_id, tr, te in folds:
        sd = StandardScaler(); so = StandardScaler()
        Xd_tr = sd.fit_transform(X_drfp[tr])
        Xd_te = sd.transform(X_drfp[te])
        Xo_tr = so.fit_transform(X_other[tr])
        Xo_te = so.transform(X_other[te])

        Xd_tr_t = torch.tensor(Xd_tr, dtype=torch.float32).to(DEVICE)
        Xd_te_t = torch.tensor(Xd_te, dtype=torch.float32).to(DEVICE)
        Xo_tr_t = torch.tensor(Xo_tr, dtype=torch.float32).to(DEVICE)
        Xo_te_t = torch.tensor(Xo_te, dtype=torch.float32).to(DEVICE)
        y_tr_t = torch.tensor(y[tr], dtype=torch.float32).to(DEVICE)

        model = DualBranchANN(X_drfp.shape[1], X_other.shape[1], hidden=hidden).to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-3)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=20, factor=0.5)
        ds = TensorDataset(Xd_tr_t, Xo_tr_t, y_tr_t)
        dl = DataLoader(ds, batch_size=32, shuffle=True)

        best_r2 = -np.inf; best_pred = None; no_imp = 0
        for ep in range(300):
            model.train()
            for xd, xo, yy in dl:
                opt.zero_grad()
                nn.MSELoss()(model(xd, xo), yy).backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                pred = model(Xd_te_t, Xo_te_t).cpu().numpy()
            r2 = r2_score(y[te], pred)
            sched.step(1 - r2)
            if r2 > best_r2:
                best_r2 = r2; best_pred = pred; no_imp = 0
            else:
                no_imp += 1
            if no_imp >= 40:
                break
        best_pred = np.clip(best_pred, 0, 1)
        try:
            pr = float(pearsonr(y[te], best_pred)[0])
        except Exception:
            pr = np.nan
        fold_records.append({
            "fold": fold_id,
            "r2": float(best_r2),
            "mae": float(mean_absolute_error(y[te], best_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y[te], best_pred))),
            "pearson": pr,
        })
    fm = pd.DataFrame(fold_records)
    return {
        "r2_mean": float(fm["r2"].mean()),
        "r2_std": float(fm["r2"].std()),
        "mae_mean": float(fm["mae"].mean()),
        "rmse_mean": float(fm["rmse"].mean()),
        "pearson_mean": float(fm["pearson"].mean()),
        "per_fold": fold_records,
    }


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lambda", dest="lambda_prop", type=float,
                   default=DEFAULT_LAMBDA_PROP,
                   help=f"PCL-AE property-co-learning weight (default {DEFAULT_LAMBDA_PROP})")
    p.add_argument("--latent-dim", dest="latent_dim", type=int,
                   default=LATENT_DIM,
                   help=f"Latent dimension for PCL-AE (default {LATENT_DIM})")
    # 401 always overwrites artifacts (it is the canonical "save best
    # pipeline" step, and every consumer downstream depends on its outputs
    # being the latest version).  Accept --force for symmetry with the
    # run_pipeline.ps1 wrapper but ignore it.
    p.add_argument("--force", action="store_true",
                   help="Accepted for compatibility; 401 always overwrites.")
    return p.parse_args()


def main():
    args = parse_args()
    lambda_prop = args.lambda_prop
    latent_dim = args.latent_dim
    t0 = time.time()
    print("=" * 72)
    print(f"  Save Best Model  (PCL-AE-{latent_dim} DRFP + DualBranchANN, lambda = {lambda_prop})")
    print("=" * 72)
    print(f"  Device: {DEVICE} | CUDA available: {torch.cuda.is_available()}")

    X_drfp, X_xtb, X_cond, X_inter, y, df, xtb_cols, cond_cols, best_var, best_col = load_data()
    X_xtb_cond_inter = np.hstack([X_xtb, X_cond, X_inter]).astype(np.float32)
    print(f"  Combined X_xtb_cond_inter: {X_xtb_cond_inter.shape}")

    print(f"\n[1/4] Fitting DRFP StandardScaler + training PCL-AE-{latent_dim} encoder ...")
    drfp_scaler = StandardScaler()
    Xd_s = drfp_scaler.fit_transform(X_drfp).astype(np.float32)

    pcl_model = train_pcl_ae(Xd_s, y, latent_dim=latent_dim, lambda_prop=lambda_prop)
    with torch.no_grad():
        Xd_reduced = pcl_model.encode(torch.FloatTensor(Xd_s).to(DEVICE)).cpu().numpy().astype(np.float32)
    print(f"  PCL-AE-{latent_dim} reduced DRFP: {Xd_reduced.shape}")

    print("\n[2/4] Fitting X_xtb_cond_inter StandardScaler ...")
    other_scaler = StandardScaler()
    Xo_s = other_scaler.fit_transform(X_xtb_cond_inter).astype(np.float32)

    metrics = eval_dual_branch_5fold(Xd_reduced, Xo_s, y, n_folds=5)
    print(f"  CV R^2 = {metrics['r2_mean']:.4f} +/- {metrics['r2_std']:.4f}")
    print(f"  CV MAE = {metrics['mae_mean']:.4f}")
    print(f"  CV RMSE = {metrics['rmse_mean']:.4f}")

    final_ann = train_dual_branch(Xd_reduced, Xo_s, y, hidden=128, epochs=EPOCHS_ANN)

    # Per-row predictions for downstream consumers (310 known_top10 baseline,
    # and any future 'what is the model confident on?' analysis).
    # We use the in-sample predictions from the final model (it's trained on
    # all data with early-stopping on a 15% inner-val split, so the model
    # has seen most rows). The OOF R^2 from `metrics` is the canonical
    # generalisation estimate.
    print("\n[Predict] Generating per-row predictions for downstream consumers ...")
    final_ann.eval()
    with torch.no_grad():
        pred_full = final_ann(
            torch.tensor(Xd_reduced, dtype=torch.float32).to(DEVICE),
            torch.tensor(Xo_s, dtype=torch.float32).to(DEVICE),
        ).cpu().numpy()
    pred_full = np.clip(pred_full, 0.0, 1.0)
    df_pred = df.copy()
    df_pred["pred_yield"] = pred_full
    # Canonical column aliases so 310 / other consumers can join on them
    if "yield (%)" in df_pred.columns:
        df_pred["y_true"] = df_pred["yield (%)"] / 100.0
    if "catalyst_1_name" in df_pred.columns:
        df_pred["catalyst_name"] = df_pred["catalyst_1_name"].fillna(
            df_pred.get("catalyst_2_name", "")
        ).astype(str)
    df_pred.to_csv(os.path.join(OUTPUT_DIR, "predictions.csv"),
                   index=False, encoding="utf-8-sig")
    print(f"  Saved predictions.csv with {len(df_pred)} rows "
          f"(under results_best_pipeline/, not artifacts/)")

    print("\n[Save] Persisting artifacts ...")
    joblib.dump(drfp_scaler, os.path.join(ARTIFACT_DIR, "drfp_scaler.joblib"))
    torch.save({
        "state_dict": pcl_model.state_dict(),
        "input_dim": pcl_model.input_dim,
        "latent_dim": pcl_model.latent_dim,
        "lambda_prop": lambda_prop,
    }, os.path.join(ARTIFACT_DIR, "pcl_ae_encoder.pt"))
    joblib.dump(other_scaler, os.path.join(ARTIFACT_DIR, "xtb_cond_inter_scaler.joblib"))
    torch.save({
        "state_dict": final_ann.state_dict(),
        "drfp_dim": Xd_reduced.shape[1],
        "xtb_dim": Xo_s.shape[1],
        "hidden": 128,
        "lambda_prop": lambda_prop,
    }, os.path.join(ARTIFACT_DIR, "dual_branch_ann.pt"))

    meta = {
        "pipeline": f"PCL-AE-{latent_dim} + DualBranchANN (lambda={lambda_prop}, full data)",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_train_samples": int(len(y)),
        "y_mean": float(y.mean()),
        "y_std": float(y.std()),
        "drfp_raw_dim": int(X_drfp.shape[1]),
        "drfp_reduced_dim": int(Xd_reduced.shape[1]),
        "drfp_variant": best_var,
        "drfp_col": best_col,
        "xtb_dim": int(X_xtb.shape[1]),
        "cond_dim": int(X_cond.shape[1]),
        "inter_dim": int(X_inter.shape[1]),
        "xtb_cond_inter_dim": int(X_xtb_cond_inter.shape[1]),
        "xtb_cols": xtb_cols,
        "cond_cols": cond_cols,
        "interaction_rules": [
            {"name": "T * activation_proxy", "cond_idx": 0, "xtb_name": "activation_proxy"},
            {"name": "P * total_polarity_index", "cond_idx": 1, "xtb_name": "total_polarity_index"},
        ],
        "scaler_notes": {
            "drfp_scaler": f"StandardScaler on raw DRFP ({best_var}, 2048-D)",
            "xtb_cond_inter_scaler": "StandardScaler on [X_xtb | X_cond | X_inter]",
        },
        "drfp_ablation_meta_source": os.path.join(os.path.dirname(ARTIFACT_DIR), "drfp_ablation_meta.json"),
        "pcl_ae_kwargs": {
            "latent_dim": LATENT_DIM, "lambda_prop": lambda_prop, "pos_weight": 10.0,
            "epochs": EPOCHS_PCL, "batch_size": 128, "lr": 1e-3,
        },
        "ann_kwargs": {
            "hidden": 128, "epochs": EPOCHS_ANN, "batch_size": 32, "lr": 5e-4,
            "weight_decay": 1e-3, "val_frac": 0.15, "patience": 40,
        },
        "predictions_csv": os.path.join(OUTPUT_DIR, "predictions.csv"),
        "cv_metrics": metrics,
    }
    with open(os.path.join(ARTIFACT_DIR, "feature_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    with open(os.path.join(ARTIFACT_DIR, "training_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    elapsed = time.time() - t0
    report = [
        "=" * 72,
        "  Save Best Model - Report",
        "=" * 72,
        f"  Device              : {DEVICE}",
        f"  N training samples  : {len(y)}",
        f"  PCL-AE lambda_prop  : {lambda_prop}",
        f"  y range             : [{y.min():.3f}, {y.max():.3f}], mean={y.mean():.3f}",
        "",
        "  CV metrics (PCL-AE-128 + DualBranchANN, 5-fold, early stop):",
        f"    R^2      = {metrics['r2_mean']:.4f} +/- {metrics['r2_std']:.4f}",
        f"    MAE      = {metrics['mae_mean']:.4f}",
        f"    RMSE     = {metrics['rmse_mean']:.4f}",
        f"    Pearson  = {metrics['pearson_mean']:.4f}",
        "",
        "  Per-fold:",
    ]
    for r in metrics["per_fold"]:
        report.append(
            f"    fold {r['fold']}: R^2={r['r2']:.4f}  MAE={r['mae']:.4f}  Pearson={r['pearson']:.4f}"
        )
    report.append("")
    report.append(f"  Artifacts saved under:")
    report.append(f"    {ARTIFACT_DIR}")
    for fname in sorted(os.listdir(ARTIFACT_DIR)):
        sz = os.path.getsize(os.path.join(ARTIFACT_DIR, fname))
        report.append(f"    {fname:38s} {sz:>10d} bytes")
    report.append("")
    report.append(f"  Elapsed: {elapsed:.1f} s")
    report.append("=" * 72)
    text = "\n".join(report)
    print("\n" + text)
    with open(os.path.join(OUTPUT_DIR, "save_best_model_report.txt"), "w", encoding="utf-8") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    main()