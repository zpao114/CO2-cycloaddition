# -*- coding: utf-8 -*-
"""201_ablation.py — DRFP variant ablation + λ (lambda) sweep + full benchmark.

Three-stage pipeline executed in a single run:

Stage 1 (DRFP variant ablation):
    - 4 DRFP variants (full / reactants / no_cats / no_sols) × XGBoost
    - → results_best_pipeline/drfp_ablation_meta.json
    - → results_best_pipeline/drfp_ablation_results.csv

Stage 2 (PCL-AE λ sweep on best DRFP variant):
    - λ ∈ {0, 0.05, 0.1, 0.2, 0.5, 1, 2, 3, 5, 7, 10, 20, 50,
            75, 100, 150, 200} × 3 AE seeds × 5 folds
    - → results_lambda_ablation/lambda_results.csv  (consumed by src/config.py)
    - → results_lambda_ablation/lambda_results_raw.csv
    - → results_lambda_ablation/figure_lambda_ablation.png
    - → results_lambda_ablation/lambda_results.txt

Stage 3 (full benchmark with best DRFP variant + best λ):
    - Part A: DRFP reduction (raw / pca128 / pca256 / ae128 / ae256 / pcl128 / pcl256)
              × feature suffix × multiple models
    - Part B: XTB/Cond baselines (no DRFP)
    - Part C: OOF ensemble of XGB+LGBM+DualANN on 3 DRFP variants
    - Part D: NOT IMPLEMENTED (was a "HeckLit-style" diagnostic, intentionally removed
              because the underlying methodology is unclear and it was never executed.)
    - → results_best_pipeline/full_benchmark_results.csv

Usage
-----
    python 201_ablation.py                      # full run (~2.7h on CPU)
    python 201_ablation.py --stage 1            # only DRFP variant ablation (~2 min)
    python 201_ablation.py --stage 2            # only λ sweep (~2h)
    python 201_ablation.py --stage 3            # only full benchmark (~30 min)
    python 201_ablation.py --force              # overwrite existing outputs
    python 201_ablation.py --quick              # smoke test (tiny grid, fast)

Notes
-----
- Splits come from data_split.json (yield-stratified, 5-fold, seed=2026).
- The dual-branch ANN, PCL-AE, and StandardAE definitions are the **single source
  of truth** used by train_pcl_ae.py and other downstream scripts.  Do not move
  them to a shared module without updating those imports.
- All logging goes to the console via the logging module; no print-to-stdout
  brittleness, no Unicode mojibake.
"""
from __future__ import annotations

import argparse
import io
import json
import logging
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

import lightgbm as lgb
import xgboost as xgb

# --- project setup ----------------------------------------------------------
PROJECT_ROOT = Path(os.environ.get(
    "CO2_PROJECT_ROOT", r"D:\machine-learning\CO2-cycloaddition"))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Force UTF-8 stdout (PowerShell default is GBK / cp936 on Windows)
if sys.stdout.encoding and sys.stdout.encoding.lower().replace("-", "") != "utf8":
    try:
        if hasattr(sys.stdout, "buffer"):
            sys.stdout = io.TextIOWrapper(
                sys.stdout.buffer,
                encoding="utf-8",
                errors="replace",
                line_buffering=not sys.stdout.isatty(),
            )
    except Exception:  # pragma: no cover
        pass
warnings.filterwarnings("ignore")

from utils_rxn import read_drfp                                  # noqa: E402
from src.config import BEST_LAMBDA_PROP as _CFG_BEST_LAMBDA     # noqa: E402
from src.data_split import (                                    # noqa: E402
    kfold_folds,
    load_manifest,
    split_iterator,
)
from src.paths import (                                         # noqa: E402
    DRFP_XTB_EXTENDED_CSV as DATA_EXTENDED,
)

# Output paths — kept consistent with orchestrator's -Output paths
# (run_pipeline_v2.ps1 L282: $ROOT\results_best_pipeline\full_benchmark_results.csv)
# NOT through src.paths because that uses RESULTS_DIR/ prefix.
ABLAT_OUT = PROJECT_ROOT / "results_best_pipeline"
LAMB_OUT  = PROJECT_ROOT / "results_lambda_ablation"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_FOLDS = 5
LATENT_DIM = 128

DRFP_VARIANTS: Dict[str, Tuple[str, str]] = {
    "full":      ("drfp",         "DRFP_full"),
    "reactants": ("drfp React",   "DRFP_reactants"),
    "no_cats":   ("drfp wo cats", "DRFP_no_cats"),
    "no_sols":   ("drfp wo sols", "DRFP_no_sols"),
}

LAMBDAS: Tuple[float, ...] = (
    0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0,
    10.0, 20.0, 50.0, 75.0, 100.0, 150.0, 200.0,
)
AE_SEEDS: Tuple[int, ...] = (42, 2026, 7)
EPOCHS_STD_AE = 100
EPOCHS_PCL_AE = 150
EPOCHS_DUAL_ANN = 300


# ============================================================================
# Logging configuration
# ============================================================================
def configure_logging(verbose: bool = False) -> None:
    """Configure root logger to write to stderr at INFO/DEBUG level."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
        force=True,
    )


log = logging.getLogger("ablation")


# ============================================================================
# Model definitions (single source of truth, used by train_pcl_ae.py)
# ============================================================================
class StandardAE(nn.Module):
    """Vanilla autoencoder (encoder + decoder, no property prediction head)."""
    def __init__(self, input_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.ReLU(),
            nn.Linear(256, 512), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Linear(512, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class PropertyCoLearningAE(nn.Module):
    """Property co-learning autoencoder (joint reconstruction + yield prediction)."""
    def __init__(self, input_dim: int, latent_dim: int) -> None:
        super().__init__()
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

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        return self.decoder(z), self.predictor(z).squeeze(-1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class DualBranchANN(nn.Module):
    """Dual-branch ANN: DRFP latent (1D conv) + XTB/Cond/Inter (MLP).

    This is the *evaluation* model used in Stage 2 / Stage 3; it does NOT
    participate in training the PCL-AE latent itself.
    """
    def __init__(self, drfp_dim: int, xtb_dim: int, hidden: int = 128) -> None:
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

    def forward(self, x_drfp: torch.Tensor, x_xtb: torch.Tensor) -> torch.Tensor:
        h = x_drfp.unsqueeze(1)
        h = torch.relu(self.conv1(h))
        h = torch.relu(self.conv2(h))
        h = self.pool(h).squeeze(-1)
        h = self.bn_drfp(h)
        h = self.fc_drfp(h)
        h2 = self.fc_xtb(x_xtb)
        return self.fc_out(torch.cat([h, h2], dim=1)).squeeze(-1)


# ============================================================================
# Data loading
# ============================================================================
def parse_drfp_col(series: pd.Series) -> np.ndarray:
    """Decode a DRFP string column to (n, 2048) float32 array."""
    arrs: List[np.ndarray] = []
    for fp_str in series:
        a = read_drfp(fp_str)
        if a is None or a.size == 0:
            arrs.append(np.zeros(2048, dtype=np.float32))
        else:
            arrs.append(a.astype(np.float32))
    return np.array(arrs)


def load_data(
    drfp_variant: str = "full",
    use_holdout_train: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """Load co2_drfp_xtb_extended.csv and split into DRFP / XTB / Cond / Inter / y.

    If ``use_holdout_train=True`` (default), restrict to the 85% holdout train
    pool from data_split.json so this script's 5-fold CV uses the SAME train
    pool as everywhere else.
    """
    log.info("[load_data] DRFP variant=%s  use_holdout_train=%s", drfp_variant, use_holdout_train)
    df = pd.read_csv(DATA_EXTENDED, encoding="utf-8-sig")
    df = df[df["extraction_status"] == "valid"].copy()
    df = df.dropna(subset=["yield (%)"])
    df = df[df["yield (%)"] > 0].reset_index(drop=True)

    if use_holdout_train:
        from src.data_split import holdout_arrays
        train_idx, _, _ = holdout_arrays(load_manifest())
        df = df.iloc[sorted(train_idx)].reset_index(drop=True)
        log.info("[load_data] filtered to holdout train pool: %d rows", len(df))

    col = DRFP_VARIANTS[drfp_variant][0]
    X_drfp = parse_drfp_col(df[col]).astype(np.float32)

    xtb_cols = [
        "sub_homo_eV", "sub_lumo_eV", "sub_gap_eV", "sub_dipole_D",
        "co2_homo_eV", "co2_lumo_eV", "co2_gap_eV",
        "cat_homo_eV", "cat_lumo_eV", "cat_gap_eV", "cat_dipole_D",
        "solv_homo_eV", "solv_lumo_eV", "solv_gap_eV",
        "delta_E_hl_cat_sub", "global_hardness", "nucleophilicity_index",
        "cat_homo_eV_min", "cat_lumo_eV_max", "cat_gap_eV_min",
        "cat_cation_homo_eV", "cat_cation_lumo_eV", "cat_cation_gap_eV",
        "cat_anion_homo_eV", "cat_anion_lumo_eV", "cat_anion_gap_eV",
        "cat_cation_dipole_D", "cat_anion_dipole_D",
        "activation_proxy", "charge_transfer_potential", "ion_pair_interaction",
        "electrophilicity_cat", "electrodonating_cat",
        "sub_cat_orbital_match", "gap_ratio", "hardness_ratio",
        "nucleophilicity_cat", "reaction_polarity", "co2_activation_proxy",
        "solv_cat_interaction", "solv_sub_interaction",
        "total_polarity_index", "dielectric_proxy",
    ]
    xtb_cols = [c for c in xtb_cols if c in df.columns]
    X_xtb = np.nan_to_num(df[xtb_cols].values.astype(np.float32), nan=0.0)

    cond_cols = ["temperature (°C)", "pressure (MPa)", "time (h)"]
    cat_loading_cols = [c for c in df.columns if "loading_mol%" in c]
    cond_cols = [c for c in cond_cols + cat_loading_cols if c in df.columns]
    X_cond = np.nan_to_num(df[cond_cols].values.astype(np.float32), nan=0.0)

    ai_idx = xtb_cols.index("activation_proxy") if "activation_proxy" in xtb_cols else None
    tpi_idx = xtb_cols.index("total_polarity_index") if "total_polarity_index" in xtb_cols else None
    inter_parts: List[np.ndarray] = []
    if ai_idx is not None:
        inter_parts.append(X_cond[:, 0:1] * X_xtb[:, ai_idx:ai_idx + 1])
    if tpi_idx is not None:
        inter_parts.append(X_cond[:, 1:2] * X_xtb[:, tpi_idx:tpi_idx + 1])
    X_inter = (
        np.concatenate(inter_parts, axis=1).astype(np.float32)
        if inter_parts else np.zeros((len(df), 0), dtype=np.float32)
    )

    y = df["yield (%)"].values.astype(np.float32) / 100.0
    return X_drfp, X_xtb, X_cond, X_inter, y, df


# ============================================================================
# Training primitives
# ============================================================================
def train_standard_ae(
    X: np.ndarray,
    latent_dim: int,
    epochs: int = EPOCHS_STD_AE,
    batch_size: int = 128,
    lr: float = 1e-3,
) -> np.ndarray:
    model = StandardAE(X.shape[1], latent_dim)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    crit = nn.MSELoss()
    loader = DataLoader(
        TensorDataset(torch.FloatTensor(X)), batch_size=batch_size, shuffle=True)
    model.train()
    for _ in range(epochs):
        for batch in loader:
            opt.zero_grad()
            loss = crit(model(batch[0]), batch[0])
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        return model.encode(torch.FloatTensor(X)).numpy()


def train_pcl_ae(
    X: np.ndarray,
    y: np.ndarray,
    latent_dim: int,
    lambda_prop: float,
    epochs: int = EPOCHS_PCL_AE,
    batch_size: int = 128,
    lr: float = 1e-3,
    pos_weight: float = 10.0,
    seed: int = 42,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    X_t = torch.FloatTensor(X)
    y_t = torch.FloatTensor(y).unsqueeze(1)
    model = PropertyCoLearningAE(X.shape[1], latent_dim)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=10, factor=0.5)
    bce = nn.BCEWithLogitsLoss(reduction="none")
    mse = nn.MSELoss()
    loader = DataLoader(
        TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    y_mean, y_std = float(y.mean()), float(y.std()) + 1e-6
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            z = model.encoder(xb)
            recon = model.decoder(z)
            pred = model.predictor(z).squeeze(-1)
            recon_loss = (bce(recon, xb) * (xb * (pos_weight - 1) + 1)).mean()
            prop_loss = mse(pred * y_std + y_mean, yb.squeeze(-1))
            (recon_loss + lambda_prop * prop_loss).backward()
            opt.step()
        sched.step(recon_loss.item())
    model.eval()
    with torch.no_grad():
        return model.encode(X_t).numpy()


def get_tree_model(name: str):
    """Build a tree regressor by short name."""
    if name == "RF":
        return RandomForestRegressor(
            n_estimators=200, max_depth=20, min_samples_leaf=2,
            n_jobs=-1, random_state=42,
        )
    if name == "XGB":
        return xgb.XGBRegressor(
            n_estimators=500, max_depth=8, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            tree_method="hist", random_state=42, verbosity=0,
        )
    if name == "LGBM":
        return lgb.LGBMRegressor(
            n_estimators=500, num_leaves=63, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_data_in_leaf=10,
            device="cpu", verbose=-1, random_state=42,
        )
    raise ValueError(f"unknown tree model: {name!r}")


# ============================================================================
# Cross-validated evaluators (use canonical 5-fold split from data_split.json)
# ============================================================================
def eval_sklearn_model(
    model_name: str,
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int = N_FOLDS,
    n_repeats: int = 3,
) -> Dict[str, float]:
    """5-fold CV evaluator (manifest-driven)."""
    records: List[Dict[str, float]] = []
    folds = list(kfold_folds(load_manifest()))
    for rep in range(n_repeats):
        for fold_id, tr, te in folds:
            sc = StandardScaler()
            X_tr = sc.fit_transform(X[tr])
            X_te = sc.transform(X[te])
            m = get_tree_model(model_name)
            m.fit(X_tr, y[tr])
            p = np.clip(m.predict(X_te), 0, 1)
            records.append({
                "rep": rep, "fold": fold_id,
                "r2": r2_score(y[te], p),
                "mae": mean_absolute_error(y[te], p),
                "rmse": float(np.sqrt(mean_squared_error(y[te], p))),
                "pearson": float(pearsonr(y[te], p)[0]),
            })
    df = pd.DataFrame(records)
    return {
        "r2_mean": float(df["r2"].mean()),
        "r2_std": float(df["r2"].std()),
        "mae_mean": float(df["mae"].mean()),
        "rmse_mean": float(df["rmse"].mean()),
        "pearson_mean": float(df["pearson"].mean()),
    }


def eval_dual_branch(
    Xd: np.ndarray,
    Xo: np.ndarray,
    y: np.ndarray,
    n_folds: int = N_FOLDS,
    n_repeats: int = 2,
    hidden: int = 128,
) -> Dict[str, float]:
    """Dual-branch ANN 5-fold CV using data_split.json manifest."""
    records: List[Dict[str, float]] = []
    folds = list(kfold_folds(load_manifest()))
    for rep in range(n_repeats):
        for fold_id, tr, te in folds:
            sd = StandardScaler(); so = StandardScaler()
            Xd_tr = sd.fit_transform(Xd[tr]); Xd_te = sd.transform(Xd[te])
            Xo_tr = so.fit_transform(Xo[tr]); Xo_te = so.transform(Xo[te])
            Xd_tr_t = torch.tensor(Xd_tr, dtype=torch.float32).to(DEVICE)
            Xd_te_t = torch.tensor(Xd_te, dtype=torch.float32).to(DEVICE)
            Xo_tr_t = torch.tensor(Xo_tr, dtype=torch.float32).to(DEVICE)
            Xo_te_t = torch.tensor(Xo_te, dtype=torch.float32).to(DEVICE)
            y_tr_t = torch.tensor(y[tr], dtype=torch.float32).to(DEVICE)
            model = DualBranchANN(Xd.shape[1], Xo.shape[1], hidden=hidden).to(DEVICE)
            opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
            sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=15, factor=0.5)
            ds = TensorDataset(Xd_tr_t, Xo_tr_t, y_tr_t)
            dl = DataLoader(ds, batch_size=64, shuffle=True)
            best_r2, best_p, no_imp = -np.inf, None, 0
            for _ in range(EPOCHS_DUAL_ANN):
                model.train()
                for xd, xo, yy in dl:
                    opt.zero_grad()
                    nn.MSELoss()(model(xd, xo), yy).backward()
                    opt.step()
                model.eval()
                with torch.no_grad():
                    p = model(Xd_te_t, Xo_te_t).cpu().numpy()
                r2 = r2_score(y[te], p)
                sched.step(1 - r2)
                if r2 > best_r2:
                    best_r2, best_p, no_imp = r2, p, 0
                else:
                    no_imp += 1
                if no_imp >= 40:
                    break
            p_clip = np.clip(best_p, 0, 1)
            records.append({
                "rep": rep, "fold": fold_id,
                "r2": float(best_r2),
                "mae": float(mean_absolute_error(y[te], p_clip)),
                "rmse": float(np.sqrt(mean_squared_error(y[te], p_clip))),
                "pearson": float(pearsonr(y[te], p_clip)[0]),
            })
    df = pd.DataFrame(records)
    return {
        "r2_mean": float(df["r2"].mean()),
        "r2_std": float(df["r2"].std()),
        "mae_mean": float(df["mae"].mean()),
        "rmse_mean": float(df["rmse"].mean()),
        "pearson_mean": float(df["pearson"].mean()),
    }


# ============================================================================
# Stage 1: DRFP variant ablation
# ============================================================================
def stage1_drfp_ablation(force: bool = False) -> str:
    """Compare 4 DRFP variants via XGB on the holdout train pool (5-fold CV).

    Returns the winning variant key.  Writes:
        results_best_pipeline/drfp_ablation_meta.json
        results_best_pipeline/drfp_ablation_results.csv
    """
    log.info("=" * 72)
    log.info("  Stage 1: DRFP variant ablation")
    log.info("=" * 72)

    meta_path = ABLAT_OUT / "drfp_ablation_meta.json"
    csv_path = ABLAT_OUT / "drfp_ablation_results.csv"
    if not force and meta_path.exists():
        log.info("[stage1] %s already exists; loading (use --force to overwrite)", meta_path)
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        return meta["best_variant"]

    folds = list(kfold_folds(load_manifest()))
    _, X_xtb, X_cond, X_inter, y, df = load_data("full")
    Xo = np.hstack([X_xtb, X_cond, X_inter]).astype(np.float32)
    log.info(
        "  shared features: XTB(%d) + Cond(%d) + Inter(%d) = %dD  |  n=%d",
        X_xtb.shape[1], X_cond.shape[1], X_inter.shape[1], Xo.shape[1], len(y),
    )

    best_r2, best_variant = -np.inf, None
    rows: List[Dict[str, Any]] = []

    for vk, (col, label) in DRFP_VARIANTS.items():
        log.info("  [%s] %s ...", vk, label)
        Xd = parse_drfp_col(df[col]).astype(np.float32)
        oof = np.zeros(len(y), dtype=np.float32)
        for fold_id, tr, te in folds:
            sd = StandardScaler(); so = StandardScaler()
            Xd_tr = sd.fit_transform(Xd[tr]); Xd_te = sd.transform(Xd[te])
            Xo_tr = so.fit_transform(Xo[tr]); Xo_te = so.transform(Xo[te])
            m = get_tree_model("XGB")
            m.fit(np.hstack([Xd_tr, Xo_tr]), y[tr])
            oof[te] = np.clip(m.predict(np.hstack([Xd_te, Xo_te])), 0, 1)
        r2 = r2_score(y, oof)
        mae = mean_absolute_error(y, oof)
        rmse = float(np.sqrt(mean_squared_error(y, oof)))
        pear = float(pearsonr(y, oof)[0])
        log.info("  [%s] R²=%.4f  RMSE=%.4f  MAE=%.4f  Pearson=%.4f", vk, r2, rmse, mae, pear)
        rows.append({
            "variant": vk, "variant_label": label, "drfp_col": col,
            "r2": r2, "rmse": rmse, "mae": mae, "pearson": pear,
        })
        if r2 > best_r2:
            best_r2, best_variant = r2, vk

    ABLAT_OUT.mkdir(parents=True, exist_ok=True)
    meta = {
        "best_variant": best_variant,
        "best_r2": float(best_r2),
        "best_drfp_col": DRFP_VARIANTS[best_variant][0],
        "best_label": DRFP_VARIANTS[best_variant][1],
        "all_variants": {
            v: {"r2": float(r["r2"]), "drfp_col": r["drfp_col"]}
            for v, r in zip(DRFP_VARIANTS, rows)
        },
    }
    meta_path.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    pd.DataFrame(rows).to_csv(csv_path, index=False, encoding="utf-8-sig")
    log.info("  ✔ best DRFP variant: %s  R²=%.4f", best_variant, best_r2)
    log.info("  → %s", meta_path)
    return best_variant


# ============================================================================
# Stage 2: PCL-AE λ sweep (on best DRFP variant)
# ============================================================================
def _plot_lambda_ablation(
    df_sum: pd.DataFrame,
    df_raw: pd.DataFrame,
    best_lam: float,
    fig_path: Path,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    lam_arr = df_sum["lambda"].values

    ax = axes[0]
    ann_r = df_sum["ann_r2_mean"].values
    ann_e = df_sum["ann_r2_seed_std"].values
    for si, seed in enumerate(AE_SEEDS):
        m = df_raw["ae_seed"] == seed
        ax.plot(
            df_raw.loc[m, "lambda"].values, df_raw.loc[m, "ann_r2"].values,
            "o--", alpha=0.3, color="steelblue", linewidth=1, markersize=4,
            label=f"seed={seed}" if si == 0 else "",
        )
    ax.fill_between(lam_arr, ann_r - ann_e, ann_r + ann_e, alpha=0.2, color="steelblue")
    ax.plot(lam_arr, ann_r, "o-", color="steelblue",
            linewidth=2.5, markersize=8, label="DualANN")
    ax2 = ax.twinx()
    rf_r = df_sum["rf_r2_mean"].values
    rf_e = df_sum["rf_r2_seed_std"].values
    ax2.fill_between(lam_arr, rf_r - rf_e, rf_r + rf_e, alpha=0.15, color="tomato")
    ax2.plot(lam_arr, rf_r, "s--", color="tomato",
             linewidth=1.5, markersize=6, label="RF")
    ax.axvline(x=best_lam, color="green", linestyle=":",
               linewidth=1.5, label=f"Best λ={best_lam}")
    ax.set_xlabel("Property co-learning weight λ", fontsize=11)
    ax.set_ylabel("DualANN R²", fontsize=11, color="steelblue")
    ax2.set_ylabel("RF R²", fontsize=11, color="tomato")
    ax.set_xscale("log")
    ax.set_ylim(0.15, 0.45); ax2.set_ylim(0.15, 0.45)
    ax.set_title(
        f"λ Ablation: R² vs λ\nBest λ={best_lam:.2f} "
        f"(R²={ann_r.max():.4f}±{ann_e.max():.4f})",
        fontsize=11,
    )
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="lower right", fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    ax3 = axes[1]
    for si, seed in enumerate(AE_SEEDS):
        m = df_raw["ae_seed"] == seed
        ax3.plot(
            df_raw.loc[m, "lambda"].values, df_raw.loc[m, "ann_pearson"].values,
            "o--", alpha=0.3, color="steelblue", linewidth=1, markersize=4,
        )
    ax3.plot(lam_arr, df_sum["ann_pearson_mean"].values, "o-", color="steelblue",
             linewidth=2.5, markersize=8, label="DualANN")
    ax3.plot(lam_arr, df_sum["rf_pearson_mean"].values, "s--", color="tomato",
             linewidth=1.5, markersize=6, label="RF")
    ax3.axvline(x=best_lam, color="green", linestyle=":", linewidth=1.5)
    ax3.set_xlabel("Property co-learning weight λ", fontsize=11)
    ax3.set_ylabel("Pearson r", fontsize=11)
    ax3.set_xscale("log")
    ax3.set_title("Pearson r vs λ", fontsize=11)
    ax3.legend(fontsize=9)
    ax3.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()


def stage2_lambda_sweep(
    best_variant: str,
    lambdas: Sequence[float] = LAMBDAS,
    ae_seeds: Sequence[int] = AE_SEEDS,
    force: bool = False,
) -> float:
    """λ sweep: train PCL-AE at each λ with multiple seeds, eval on 5-fold CV.

    Returns the best λ (max DualANN R²).
    Writes:
        results_lambda_ablation/lambda_results.csv
        results_lambda_ablation/lambda_results_raw.csv
        results_lambda_ablation/figure_lambda_ablation.png
        results_lambda_ablation/lambda_results.txt
    """
    log.info("=" * 72)
    log.info("  Stage 2: PCL-AE λ sweep  (best DRFP variant: %s)", best_variant)
    log.info("  λ range: %s  |  AE seeds: %s", list(lambdas), list(ae_seeds))
    log.info("=" * 72)

    csv_path = LAMB_OUT / "lambda_results.csv"
    if not force and csv_path.exists():
        log.info("[stage2] %s already exists; loading best λ (use --force to overwrite)", csv_path)
        df_existing = pd.read_csv(csv_path)
        best_lam = float(df_existing.loc[df_existing["ann_r2_mean"].idxmax(), "lambda"])
        log.info("  ✔ best λ (from existing CSV): %s", best_lam)
        return best_lam

    LAMB_OUT.mkdir(parents=True, exist_ok=True)

    X_drfp, X_xtb, X_cond, X_inter, y, _ = load_data(best_variant)
    X_other = np.hstack([X_xtb, X_cond, X_inter]).astype(np.float32)
    scaler = StandardScaler()
    Xd_s = scaler.fit_transform(X_drfp).astype(np.float32)

    raw_records: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []

    for lam in lambdas:
        log.info("  ─── λ = %s ───", lam)
        seed_ann_r2: List[float] = []
        seed_ann_pr: List[float] = []
        seed_rf_r2: List[float] = []
        seed_rf_pr: List[float] = []

        for ae_seed in ae_seeds:
            t1 = time.time()
            Xd_red = train_pcl_ae(Xd_s, y, LATENT_DIM, lam, seed=ae_seed)
            ae_t = time.time() - t1

            ann = eval_dual_branch(Xd_red, X_other, y, n_folds=N_FOLDS)
            rf = eval_sklearn_model("RF", Xd_red, y, n_folds=N_FOLDS)

            seed_ann_r2.append(ann["r2_mean"])
            seed_ann_pr.append(ann["pearson_mean"])
            seed_rf_r2.append(rf["r2_mean"])
            seed_rf_pr.append(rf["pearson_mean"])

            log.info(
                "    seed=%d (%ds)  DualANN R²=%.4f  RF R²=%.4f",
                ae_seed, int(ae_t), ann["r2_mean"], rf["r2_mean"],
            )
            raw_records.append({
                "lambda": lam, "ae_seed": ae_seed,
                "ann_r2": ann["r2_mean"], "ann_pearson": ann["pearson_mean"],
                "rf_r2": rf["r2_mean"], "rf_pearson": rf["pearson_mean"],
            })

        ann_r2_mean = float(np.mean(seed_ann_r2))
        ann_r2_std = float(np.std(seed_ann_r2))
        rf_r2_mean = float(np.mean(seed_rf_r2))
        rf_r2_std = float(np.std(seed_rf_r2))
        log.info(
            "  >>> λ=%5.2f  DualANN R²=%.4f±%.4f  RF R²=%.4f±%.4f",
            lam, ann_r2_mean, ann_r2_std, rf_r2_mean, rf_r2_std,
        )
        summary_rows.append({
            "lambda": lam,
            "ann_r2_mean": ann_r2_mean,
            "ann_r2_seed_std": ann_r2_std,
            "ann_pearson_mean": float(np.mean(seed_ann_pr)),
            "rf_r2_mean": rf_r2_mean,
            "rf_r2_seed_std": rf_r2_std,
            "rf_pearson_mean": float(np.mean(seed_rf_pr)),
            "best_model": "DualANN" if ann_r2_mean > rf_r2_mean else "RF",
        })

    df_sum = pd.DataFrame(summary_rows)
    df_sum.to_csv(csv_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(raw_records).to_csv(
        LAMB_OUT / "lambda_results_raw.csv", index=False, encoding="utf-8-sig",
    )

    best_row = df_sum.loc[df_sum["ann_r2_mean"].idxmax()]
    best_lam = float(best_row["lambda"])
    log.info(
        "  ✔ best λ = %s  (DualANN R² = %.4f ± %.4f)",
        best_lam, best_row["ann_r2_mean"], best_row["ann_r2_seed_std"],
    )

    # Note: src/config.py auto-loads BEST_LAMBDA_PROP from this CSV on next import.
    # No in-script writeback to config.py is needed (and no longer supported).
    csv_cfg = PROJECT_ROOT / "results_lambda_ablation" / "lambda_results.csv"
    if csv_cfg.exists():
        log.info("  [config] src/config.py auto-loads BEST_LAMBDA_PROP from %s", csv_cfg)
    else:
        log.warning("  [config] lambda_results.csv missing; config.py will fall back")

    # Plot
    try:
        _plot_lambda_ablation(
            df_sum, pd.DataFrame(raw_records), best_lam,
            LAMB_OUT / "figure_lambda_ablation.png",
        )
        log.info("  [saved] %s", LAMB_OUT / "figure_lambda_ablation.png")
    except Exception as ex:
        log.warning("  [plot] failed: %s", ex)

    # Text report
    lines = [
        "=" * 72, "  PCL-AE λ Ablation Study — Report", "=" * 72,
        f"Latent dim: {LATENT_DIM}  |  Folds: {N_FOLDS}  |  AE seeds: {list(ae_seeds)}",
        "  λ         DualANN R²                  RF R²                   Best model",
        "  " + "-" * 78,
    ]
    for _, r in df_sum.iterrows():
        w = "DualANN" if r["ann_r2_mean"] > r["rf_r2_mean"] else "RF"
        lines.append(
            f"  {r['lambda']:5.2f}    {r['ann_r2_mean']:.4f}±{r['ann_r2_seed_std']:.4f}    "
            f"{r['rf_r2_mean']:.4f}±{r['rf_r2_seed_std']:.4f}    {w}"
        )
    lines.append("  " + "-" * 78)
    lines.append(
        f"\n  Optimal λ (DualANN): {best_lam}  "
        f"R²={best_row['ann_r2_mean']:.4f}±{best_row['ann_r2_seed_std']:.4f}"
    )
    text = "\n".join(lines)
    log.info("\n%s", text)
    (LAMB_OUT / "lambda_results.txt").write_text(text, encoding="utf-8")

    return best_lam


# ============================================================================
# Stage 3: full benchmark
# ============================================================================
def stage3_benchmark(
    best_variant: str,
    best_lambda: Optional[float] = None,
    force: bool = False,
) -> pd.DataFrame:
    """Full benchmark with the best DRFP variant + best λ.

    Writes results_best_pipeline/full_benchmark_results.csv.
    """
    log.info("=" * 72)
    log.info("  Stage 3: full benchmark  (variant=%s, λ=%s)", best_variant, best_lambda)
    log.info("=" * 72)

    out_csv = ABLAT_OUT / "full_benchmark_results.csv"
    if not force and out_csv.exists():
        log.info("[stage3] %s already exists; loading (use --force to overwrite)", out_csv)
        return pd.read_csv(out_csv)

    # Always prefer config.py's auto-loaded value if present (post stage 2).
    lambda_for_pcl = (
        _CFG_BEST_LAMBDA if isinstance(_CFG_BEST_LAMBDA, (int, float))
        else (best_lambda if best_lambda is not None else 200.0)
    )
    log.info("  using λ = %s (from config.py or stage 2)", lambda_for_pcl)

    X_drfp, X_xtb, X_cond, X_inter, y, _ = load_data(best_variant)
    X_xtb_cond = np.hstack([X_xtb, X_cond]).astype(np.float32)
    X_xtb_cond_inter = np.hstack([X_xtb_cond, X_inter]).astype(np.float32)

    REDUCE_METHODS = [
        ("raw",    None, "DRFP-raw"),
        ("pca128", 128,  "PCA-128"),
        ("pca256", 256,  "PCA-256"),
        ("ae128",  128,  "AE-128"),
        ("ae256",  256,  "AE-256"),
        ("pcl128", 128,  f"PCL-AE-128(λ={lambda_for_pcl})"),
        ("pcl256", 256,  f"PCL-AE-256(λ={lambda_for_pcl})"),
    ]
    TREES = ["RF", "XGB", "LGBM"]

    all_results: List[Dict[str, Any]] = []
    best_r2, best_tag = -np.inf, ""

    # --- Part A: DRFP reduction × feature suffix × model --------------------
    log.info("  Part A: DRFP reduction × feature suffix × model")
    for method, dim, label in REDUCE_METHODS:
        if method == "raw":
            Xd_red = X_drfp
        else:
            scaler = StandardScaler()
            Xd_s = scaler.fit_transform(X_drfp).astype(np.float32)
            if method == "pca128":
                Xd_red = PCA(n_components=128, random_state=42).fit_transform(Xd_s).astype(np.float32)
            elif method == "pca256":
                Xd_red = PCA(n_components=256, random_state=42).fit_transform(Xd_s).astype(np.float32)
            elif method == "ae128":
                Xd_red = train_standard_ae(Xd_s, 128).astype(np.float32)
            elif method == "ae256":
                Xd_red = train_standard_ae(Xd_s, 256).astype(np.float32)
            elif method == "pcl128":
                Xd_red = train_pcl_ae(Xd_s, y, 128, lambda_for_pcl).astype(np.float32)
            elif method == "pcl256":
                Xd_red = train_pcl_ae(Xd_s, y, 256, lambda_for_pcl).astype(np.float32)
            else:  # pragma: no cover
                raise ValueError(f"unknown reduction method: {method!r}")

        suffix_map = {
            label:                  Xd_red,
            f"{label}_xtb":         np.hstack([Xd_red, X_xtb]).astype(np.float32),
            f"{label}_xtbc":        np.hstack([Xd_red, X_xtb_cond]).astype(np.float32),
            f"{label}_full":        np.hstack([Xd_red, X_xtb_cond_inter]).astype(np.float32),
        }

        for fsname, Xfull in suffix_map.items():
            log.info("    [%s] %s", fsname, Xfull.shape)
            for mname in TREES:
                t1 = time.time()
                m = eval_sklearn_model(mname, Xfull, y, n_folds=N_FOLDS, n_repeats=3)
                tag = f"{label} | {fsname} | {mname}"
                all_results.append({
                    "stage": "A_drfp_reduce", "drfp_method": label,
                    "feature_set": fsname, "model": mname,
                    **{k: m[k] for k in ["r2_mean", "r2_std", "mae_mean",
                                         "rmse_mean", "pearson_mean"]},
                    "feature_dim": Xfull.shape[1],
                    "time_s": time.time() - t1,
                })
                log.info(
                    "      %-5s R²=%.4f±%.4f  MAE=%.4f  (%ds)",
                    mname, m["r2_mean"], m["r2_std"], m["mae_mean"], int(time.time() - t1),
                )
                if m["r2_mean"] > best_r2:
                    best_r2, best_tag = m["r2_mean"], tag

            if Xfull.shape[1] >= 100 and method in ("raw", "pca128", "pca256", "pcl128", "pcl256"):
                n_drfp = Xd_red.shape[1]
                Xd_in = Xfull[:, :n_drfp]
                X_other = Xfull[:, n_drfp:]
                if X_other.shape[1] == 0:
                    continue
                t1 = time.time()
                m = eval_dual_branch(Xd_in, X_other, y, n_folds=N_FOLDS, n_repeats=2)
                tag = f"{label} | {fsname} | DualANN"
                all_results.append({
                    "stage": "A_drfp_reduce", "drfp_method": label,
                    "feature_set": fsname, "model": "DualANN",
                    **{k: m[k] for k in ["r2_mean", "r2_std", "mae_mean",
                                         "rmse_mean", "pearson_mean"]},
                    "feature_dim": Xfull.shape[1],
                    "time_s": time.time() - t1,
                })
                log.info(
                    "      DualANN R²=%.4f±%.4f  MAE=%.4f  (%ds)",
                    m["r2_mean"], m["r2_std"], m["mae_mean"], int(time.time() - t1),
                )
                if m["r2_mean"] > best_r2:
                    best_r2, best_tag = m["r2_mean"], tag

    # --- Part B: XTB / Cond / Inter baselines (no DRFP) ---------------------
    log.info("  Part B: XTB / Cond baselines (no DRFP)")
    baseline_sets = {
        "xtb": X_xtb,
        "cond": X_cond,
        "xtb_cond_inter": X_xtb_cond_inter,
    }
    for fsname, Xbase in baseline_sets.items():
        if Xbase.shape[1] == 0:
            continue
        for mname in TREES:
            t1 = time.time()
            m = eval_sklearn_model(mname, Xbase, y, n_folds=N_FOLDS, n_repeats=3)
            tag = f"{fsname} | {mname}"
            all_results.append({
                "stage": "B_no_drfp", "drfp_method": "NONE",
                "feature_set": fsname, "model": mname,
                **{k: m[k] for k in ["r2_mean", "r2_std", "mae_mean",
                                     "rmse_mean", "pearson_mean"]},
                "feature_dim": Xbase.shape[1],
                "time_s": time.time() - t1,
            })
            log.info(
                "      %-5s R²=%.4f±%.4f  (%ds)",
                mname, m["r2_mean"], m["r2_std"], int(time.time() - t1),
            )
            if m["r2_mean"] > best_r2:
                best_r2, best_tag = m["r2_mean"], tag

    # --- Part C: OOF ensemble (XGB+LGBM+DualANN on raw / pca128 / pca256) ---
    log.info("  Part C: OOF ensemble on raw / pca128 / pca256")
    scaler_p = StandardScaler()
    Xd_s = scaler_p.fit_transform(X_drfp).astype(np.float32)
    reduction_pool = [
        ("DRFP-raw", X_drfp.astype(np.float32)),
        ("PCA-128",  PCA(n_components=128, random_state=42).fit_transform(Xd_s).astype(np.float32)),
        ("PCA-256",  PCA(n_components=256, random_state=42).fit_transform(Xd_s).astype(np.float32)),
    ]
    folds_list = list(kfold_folds(load_manifest()))
    for feat_desc, Xd_red in reduction_pool:
        Xfull = np.hstack([Xd_red, X_xtb_cond_inter]).astype(np.float32)
        oof = np.zeros(len(y), dtype=np.float32)
        oof_cnt = np.zeros(len(y), dtype=np.float32)
        for fold_id, tr, te in folds_list:
            sd = StandardScaler(); so = StandardScaler()
            Xd_tr = sd.fit_transform(Xd_red[tr]); Xd_te = sd.transform(Xd_red[te])
            Xo_tr = so.fit_transform(X_xtb_cond_inter[tr]); Xo_te = so.transform(X_xtb_cond_inter[te])
            Xf_tr = np.hstack([Xd_tr, Xo_tr]); Xf_te = np.hstack([Xd_te, Xo_te])
            xg = get_tree_model("XGB"); xg.fit(Xf_tr, y[tr])
            lg = get_tree_model("LGBM"); lg.fit(Xf_tr, y[tr])
            # DualANN
            Xd_tr_t = torch.tensor(Xd_tr, dtype=torch.float32).to(DEVICE)
            Xd_te_t = torch.tensor(Xd_te, dtype=torch.float32).to(DEVICE)
            Xo_tr_t = torch.tensor(Xo_tr, dtype=torch.float32).to(DEVICE)
            Xo_te_t = torch.tensor(Xo_te, dtype=torch.float32).to(DEVICE)
            y_tr_t = torch.tensor(y[tr], dtype=torch.float32).to(DEVICE)
            ann = DualBranchANN(Xd_red.shape[1], X_xtb_cond_inter.shape[1]).to(DEVICE)
            opt = torch.optim.Adam(ann.parameters(), lr=1e-3, weight_decay=1e-4)
            sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=15, factor=0.5)
            ds = TensorDataset(Xd_tr_t, Xo_tr_t, y_tr_t)
            dl = DataLoader(ds, batch_size=64, shuffle=True)
            br2, bp, ni = -np.inf, None, 0
            for _ in range(EPOCHS_DUAL_ANN):
                ann.train()
                for xd, xo, yy in dl:
                    opt.zero_grad()
                    nn.MSELoss()(ann(xd, xo), yy).backward()
                    opt.step()
                ann.eval()
                with torch.no_grad():
                    p = ann(Xd_te_t, Xo_te_t).cpu().numpy()
                r2 = r2_score(y[te], p)
                sched.step(1 - r2)
                if r2 > br2:
                    br2, bp, ni = r2, p, 0
                else:
                    ni += 1
                if ni >= 30:
                    break
            oof[te] += (xg.predict(Xf_te) + lg.predict(Xf_te) + np.clip(bp, 0, 1)) / 3.0
            oof_cnt[te] += 1
        mask = oof_cnt > 0
        oof_avg = np.where(mask, oof / np.maximum(oof_cnt, 1), 0)
        pr2 = r2_score(y[mask], oof_avg[mask])
        log.info("    [%s] ensemble R² (pooled) = %.4f", feat_desc, pr2)
        all_results.append({
            "stage": "C_ensemble", "drfp_method": feat_desc,
            "feature_set": "ensemble", "model": "Ensemble",
            "r2_mean": float(pr2), "r2_std": np.nan,
            "mae_mean": float(mean_absolute_error(y[mask], oof_avg[mask])),
            "rmse_mean": float(np.sqrt(mean_squared_error(y[mask], oof_avg[mask]))),
            "pearson_mean": float(pearsonr(y[mask], oof_avg[mask])[0]),
            "feature_dim": Xfull.shape[1], "time_s": 0.0,
        })
        if pr2 > best_r2:
            best_r2, best_tag = pr2, f"Ensemble on {feat_desc}"

    # --- Save ---------------------------------------------------------------
    ABLAT_OUT.mkdir(parents=True, exist_ok=True)
    df_all = pd.DataFrame(all_results).sort_values("r2_mean", ascending=False)
    df_all.to_csv(out_csv, index=False, encoding="utf-8-sig")
    log.info("=" * 72)
    log.info("  Top 20 results:")
    cols = ["stage", "drfp_method", "feature_set", "model",
            "r2_mean", "r2_std", "mae_mean", "pearson_mean"]
    log.info("\n%s", df_all.head(20)[cols].to_string(index=False))
    log.info("  ✔ best: %s  R²=%.4f", best_tag, best_r2)
    log.info("  → %s", out_csv)
    return df_all


# ============================================================================
# CLI / main
# ============================================================================
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="DRFP variant ablation + PCL-AE λ sweep + full benchmark.",
    )
    p.add_argument(
        "--stage", type=int, choices=[1, 2, 3], default=None,
        help="Run only one stage (1=DRFP variants, 2=λ sweep, 3=full benchmark). "
             "Default: run all three in sequence.",
    )
    p.add_argument(
        "--force", action="store_true",
        help="Overwrite existing outputs (meta.json / lambda_results.csv / "
             "full_benchmark_results.csv).",
    )
    p.add_argument(
        "--quick", action="store_true",
        help="Smoke-test mode: tiny λ grid (5 values) and 1 AE seed.",
    )
    p.add_argument(
        "--lambdas", type=str, default=None,
        help="Comma-separated λ list to override the default (e.g. '0,1,10,50,200').",
    )
    p.add_argument(
        "--verbose", action="store_true", help="Enable DEBUG-level logging.",
    )
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    configure_logging(verbose=args.verbose)

    ABLAT_OUT.mkdir(parents=True, exist_ok=True)
    LAMB_OUT.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    log.info("=" * 72)
    log.info("  201_ablation.py  |  device: %s  |  CUDA: %s",
             DEVICE, torch.cuda.is_available())
    log.info("=" * 72)

    stages = [args.stage] if args.stage else [1, 2, 3]
    best_variant: Optional[str] = None
    best_lambda: Optional[float] = None

    if 1 in stages:
        best_variant = stage1_drfp_ablation(force=args.force)

    if 2 in stages:
        if best_variant is None:
            # Need best variant for stage 2 — try to load from existing meta.
            meta_path = ABLAT_OUT / "drfp_ablation_meta.json"
            if meta_path.exists():
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                best_variant = meta["best_variant"]
                log.info("[stage2] using best variant from existing meta: %s", best_variant)
            else:
                log.error("[stage2] no best variant available; run stage 1 first")
                return 1

        lambdas: Sequence[float] = LAMBDAS
        seeds: Sequence[int] = AE_SEEDS
        if args.quick:
            lambdas = (0.0, 1.0, 10.0, 100.0, 200.0)
            seeds = (42,)
            log.info("[--quick] using tiny λ grid: %s", list(lambdas))
        elif args.lambdas:
            lambdas = tuple(float(s) for s in args.lambdas.split(","))
            log.info("[--lambdas] using custom grid: %s", list(lambdas))

        best_lambda = stage2_lambda_sweep(
            best_variant, lambdas=lambdas, ae_seeds=seeds, force=args.force,
        )

    if 3 in stages:
        if best_variant is None:
            meta_path = ABLAT_OUT / "drfp_ablation_meta.json"
            if meta_path.exists():
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                best_variant = meta["best_variant"]
            else:
                log.error("[stage3] no best variant available; run stage 1 first")
                return 1
        stage3_benchmark(
            best_variant, best_lambda=best_lambda, force=args.force,
        )

    total = time.time() - t0
    log.info("=" * 72)
    log.info("  ALL DONE — total: %.0fs (%.1f min)", total, total / 60)
    log.info("  Stage 1 meta: %s", ABLAT_OUT / "drfp_ablation_meta.json")
    log.info("  Stage 2:      %s", LAMB_OUT / "lambda_results.csv")
    log.info("  Stage 3:      %s", ABLAT_OUT / "full_benchmark_results.csv")
    log.info("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())