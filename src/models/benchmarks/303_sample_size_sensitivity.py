# -*- coding: utf-8 -*-
"""
303_sample_size_sensitivity.py
================================

Sample-size / learning-curve analysis for the CO2-cycloaddition ML pipeline.

Question answered
-----------------
How does model R² change as the training set grows from a small sample up to
the full dataset (5-fold GroupKFold by ``catalyst_system_type`` with the test
fold held constant)?

This tells us:
  - Whether the model is data-hungry (R² still rising at large n).
  - The minimum viable training set for a given R² threshold.
  - Whether DualBranchANN benefits more or less from data than classical
    counterparts (XGB / LGBM / RF).

Protocol
--------
  - Feature set:  F2_PCLAE128 = X_num(~87 dim) + PCL-AE 128-D latent
                  (same as the SI §3 benchmark; consistent with tier_si / tier_main)
  - Models:       XGB, LGBM, RF, DualBranchANN  (same hyperparameters as v3)
  - CV protocol:  5-fold GroupKFold by ``catalyst_system_type``
                  (rows with ``catalyst_system_type == "unknown"`` are excluded
                   because they form a single group that defeats cross-validation)
  - Training sizes (n_train): a log-scaled grid from 50 up to the largest
                  fold-train size, clipped per fold.
  - Repeats:      3 seeds per size for stable mean / std.
  - Fixed test fold: the same 5 GroupKFold folds are reused for every
                  training size so the comparison is fair (test set stays constant).

Outputs (results_sample_size_sensitivity/)
------------------------------------------
  ML_ssts_v2_results.csv     — one row per (model, size, seed, fold)
  learning_curve_summary.csv — mean ± std across seeds per (model, size)
  fig_learning_curve.png     — R² / MAE vs n_train for all 4 models
  fig_learning_curve.pdf     — same as above (vector)

Inputs (required, from upstream tiers)
--------------------------------------
  data/processed/cleaned.csv
  results/results_cho_diagnostic/co2_drfp_xtb_extended.csv
  results_pcl_ae/pcl_ae_latent.npy
  results_pcl_ae/standard_ae_latent.npy
  results_pcl_ae/row_id.csv

Usage
-----
  python 303_sample_size_sensitivity.py
  python 303_sample_size_sensitivity.py --force            # overwrite outputs
  python 303_sample_size_sensitivity.py --include-unknown  # keep unknown catalyst rows
  python 303_sample_size_sensitivity.py --quick             # only n_train <= 800 (smoke-test)

Requires tier_pcl (train_pcl_ae.py) to have run first so that
results_pcl_ae/pcl_ae_latent.npy exists.

Typical runtime: ~25 min on the full grid (4 models × 12 sizes × 3 seeds × 5 folds).
"""

from __future__ import annotations

import argparse
import io
import logging
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

import lightgbm as lgb
import xgboost as xgb

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ────────────────────────────────────────────────────────────────────────────
# Encoding & warnings
# ────────────────────────────────────────────────────────────────────────────
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
if sys.stderr.encoding != "utf-8":
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")
warnings.filterwarnings("ignore")
os.environ.setdefault("PYTHONWARNINGS", "ignore")

# ────────────────────────────────────────────────────────────────────────────
# Paths
# ────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(
    os.environ.get("CO2_PROJECT_ROOT", r"D:\machine-learning\CO2-cycloaddition")
)
DATA_CSV = PROJECT_ROOT / "data/processed/cleaned.csv"
FEAT_CSV = PROJECT_ROOT / "results/results_cho_diagnostic/co2_drfp_xtb_extended.csv"
PCL_DIR = PROJECT_ROOT / "results_pcl_ae"
OUT_DIR = PROJECT_ROOT / "results_sample_size_sensitivity"

CSV_RESULTS = "ML_ssts_v2_results.csv"
CSV_SUMMARY = "learning_curve_summary.csv"
FIG_PNG = "fig_learning_curve.png"
FIG_PDF = "fig_learning_curve.pdf"

# ────────────────────────────────────────────────────────────────────────────
# Experiment config
# ────────────────────────────────────────────────────────────────────────────
N_FOLDS = 5
N_SEEDS = 3
DEVICE = "cpu"  # DualBranchANN is small enough for CPU

# Full log-scaled grid (will be clipped per fold automatically).
TRAIN_SIZES_FULL: Tuple[int, ...] = (
    50, 100, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000,
)
# Smoke-test grid for --quick mode.
TRAIN_SIZES_QUICK: Tuple[int, ...] = (50, 200, 800)

SUBSTRATE_MAP: Dict[str, str] = {
    "Styrene oxide": "SO",
    "Epichlorohydrin": "ECH",
    "Propylene oxide": "PO",
    "Cyclohexene oxide": "CHO",
    "Isopropyl glycidyl ether": "IGE",
}

# Columns that must NEVER enter X_num even if they are numeric.
NUMERIC_EXCLUDE_COLS: Tuple[str, ...] = ("yield (%)", "yield", "row_id")

# Logging
LOG_FMT = "%(asctime)s [%(levelname)s] %(message)s"
LOG_DATEFMT = "%H:%M:%S"
logger = logging.getLogger("303_sample_size")


# ────────────────────────────────────────────────────────────────────────────
# Model builders
# ────────────────────────────────────────────────────────────────────────────
def build_xgb(seed: int) -> xgb.XGBRegressor:
    return xgb.XGBRegressor(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=2.0,
        min_child_weight=5,
        random_state=seed,
        n_jobs=4,
        verbosity=0,
    )


def build_lgbm(seed: int) -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(
        n_estimators=400,
        num_leaves=15,
        learning_rate=0.05,
        min_child_samples=20,
        reg_lambda=2.0,
        random_state=seed,
        n_jobs=4,
        verbosity=-1,
    )


def build_rf(seed: int) -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=500,
        max_depth=12,
        min_samples_leaf=10,
        random_state=seed,
        n_jobs=4,
    )


# Registry — order is the order in which models are evaluated.
MODEL_REGISTRY: Dict[str, Callable[[int], object]] = {
    "XGB": build_xgb,
    "LGBM": build_lgbm,
    "RF": build_rf,
}


# ────────────────────────────────────────────────────────────────────────────
# DualBranchANN (PyTorch)
# ────────────────────────────────────────────────────────────────────────────
class DualBranchANN(nn.Module):
    """MLP head for F2_PCLAE128 = [PCL-AE latent 128D] + [X_num ~87D].

    Input dim = n_latent_dim + n_num_dim = 128 + ~87 ≈ 215.
    Architecture mirrors the v3 SI benchmark's F2 MLP:

        Linear(in, 64) → ReLU → Dropout(0.2)
        Linear(64, 32) → ReLU → Dropout(0.1)
        Linear(32, 1) → Sigmoid
    """

    def __init__(self, n_latent_dim: int, n_num_dim: int, seed: int = 42) -> None:
        super().__init__()
        torch.manual_seed(seed)
        if torch.cuda.is_available() and DEVICE.startswith("cuda"):
            torch.cuda.manual_seed_all(seed)
        self.in_dim = n_latent_dim + n_num_dim
        self.net = nn.Sequential(
            nn.Linear(self.in_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

    def fit(
        self,
        X_tr: np.ndarray,
        y_tr: np.ndarray,
        epochs: int = 60,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        batch_size: int = 64,
    ) -> "DualBranchANN":
        self.train()
        opt = torch.optim.Adam(
            self.parameters(), lr=lr, weight_decay=weight_decay
        )
        loss_fn = nn.MSELoss()
        Xt = torch.from_numpy(X_tr.astype(np.float32, copy=False))
        yt = torch.from_numpy(y_tr.astype(np.float32, copy=False).reshape(-1, 1))
        ds = TensorDataset(Xt, yt)
        dl = DataLoader(ds, batch_size=batch_size, shuffle=True)
        for _ in range(epochs):
            for xb, yb in dl:
                opt.zero_grad()
                loss_fn(self(xb), yb).backward()
                opt.step()
        return self

    @torch.no_grad()
    def predict(self, X_te: np.ndarray) -> np.ndarray:
        self.eval()
        out = self.net(torch.from_numpy(X_te.astype(np.float32, copy=False)))
        return out.cpu().numpy()


def build_ann(seed: int, n_latent: int, n_num: int) -> DualBranchANN:
    """Factory used by MODEL_REGISTRY (must be closed over n_latent/n_num)."""
    return DualBranchANN(n_latent_dim=n_latent, n_num_dim=n_num, seed=seed)


# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────
def stratified_sample(
    train_idx: np.ndarray,
    groups: np.ndarray,
    target_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Proportional stratified subsample of ``train_idx`` by catalyst group.

    Keeps each catalyst group's fractional share; truncates/pads to exactly
    ``target_size``; returns a shuffled array.
    """
    tr_groups = groups[train_idx]
    unique_groups, counts = np.unique(tr_groups, return_counts=True)
    fractions = counts / counts.sum()

    chosen: List[np.ndarray] = []
    for grp, frac in zip(unique_groups, fractions):
        grp_mask = train_idx[tr_groups == grp]
        n_grp = max(1, int(round(frac * target_size)))
        n_grp = min(n_grp, len(grp_mask))
        chosen.append(rng.choice(grp_mask, size=n_grp, replace=False))
    result = np.concatenate(chosen)

    # Correct rounding drift.
    if len(result) > target_size:
        result = rng.choice(result, size=target_size, replace=False)
    elif len(result) < target_size:
        extra = rng.choice(
            np.setdiff1d(train_idx, result, assume_unique=False),
            size=target_size - len(result),
            replace=False,
        )
        result = np.concatenate([result, extra])

    rng.shuffle(result)
    return result


def _evaluate_sklearn(model, X_tr_s: np.ndarray, y_tr: np.ndarray,
                      X_te_s: np.ndarray) -> np.ndarray:
    model.fit(X_tr_s, y_tr)
    pred = model.predict(X_te_s)
    return np.clip(pred, 0.0, 1.0)


def _evaluate_ann(model_fn, seed: int, X_tr_s: np.ndarray, y_tr: np.ndarray,
                  X_te_s: np.ndarray) -> np.ndarray:
    m = model_fn(seed=seed)
    m.fit(X_tr_s, y_tr)
    pred = m.predict(X_te_s)
    return np.clip(pred, 0.0, 1.0).reshape(-1)


def safe_pearson(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if np.std(y_true) < 1e-8 or np.std(y_pred) < 1e-8:
        return 0.0
    r = float(np.corrcoef(y_true, y_pred)[0, 1])
    return r if np.isfinite(r) else 0.0


def safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    r = float(r2_score(y_true, y_pred))
    return r if np.isfinite(r) else 0.0


# ────────────────────────────────────────────────────────────────────────────
# Data loading
# ────────────────────────────────────────────────────────────────────────────
class LoadedData:
    """Container for everything needed by the experiment loop."""

    def __init__(
        self,
        df: pd.DataFrame,
        X_num: np.ndarray,
        z_pcl: np.ndarray,
        X_full: np.ndarray,
        y: np.ndarray,
        groups: np.ndarray,
    ) -> None:
        self.df = df
        self.X_num = X_num
        self.z_pcl = z_pcl
        self.X_full = X_full
        self.y = y
        self.groups = groups
        self.n = len(df)
        self.n_num = X_num.shape[1]
        self.n_latent = z_pcl.shape[1]


def load_data(include_unknown: bool = True, exclude_unknown: bool = False) -> LoadedData:
    """Load + align + filter data; assemble X_num, z_pcl, X_full, y, groups."""
    logger.info("Loading data …")
    if not DATA_CSV.exists():
        raise FileNotFoundError(f"Missing cleaned CSV: {DATA_CSV}")
    if not FEAT_CSV.exists():
        raise FileNotFoundError(f"Missing features CSV: {FEAT_CSV}")
    if not (PCL_DIR / "pcl_ae_latent.npy").exists():
        raise FileNotFoundError(
            f"Missing PCL-AE latent (run train_pcl_ae.py first): {PCL_DIR / 'pcl_ae_latent.npy'}"
        )

    df = pd.read_csv(DATA_CSV, encoding="utf-8-sig")
    feat_df = pd.read_csv(FEAT_CSV, encoding="utf-8-sig")

    # ── Column presence guards (fail loud, not silent) ───────────────────────
    for col in ("row_id", "yield (%)", "reactant_name", "catalyst_system_type"):
        if col not in df.columns:
            raise KeyError(f"cleaned.csv is missing required column: {col!r}")
    if "row_id" not in feat_df.columns:
        raise KeyError("features CSV is missing required column: 'row_id'")

    # ── Substrate mapping ────────────────────────────────────────────────────
    df["substrate"] = df["reactant_name"].map(SUBSTRATE_MAP)
    unmapped = df.loc[df["substrate"].isna(), "reactant_name"].unique().tolist()
    if unmapped:
        raise ValueError(
            f"Found unmapped reactant_name(s): {unmapped}. "
            f"Update SUBSTRATE_MAP if a new substrate is intentional."
        )

    # ── Drop missing target + invalid yield ──────────────────────────────────
    n0 = len(df)
    df = df.dropna(subset=["yield (%)", "substrate"]).copy()
    df = df[df["yield (%)"].between(0, 100, inclusive="both")].copy()
    logger.info("  cleaned.csv: %d → %d rows after yield/NaN filter", n0, len(df))

    # ── Optionally exclude 'unknown' catalyst rows ─────────────────────────
    # By default we KEEP 'unknown' rows so GroupKFold(n=5) has enough groups
    # (4 known types + 1 'unknown'). Excluding them leaves only 4 groups and
    # makes n_splits=5 impossible.
    # Set --exclude-unknown to drop them (only safe if you also lower n_folds).
    if not include_unknown or exclude_unknown:
        n_unk = int((df["catalyst_system_type"] == "unknown").sum())
        df = df[df["catalyst_system_type"] != "unknown"].copy()
        logger.info(
            "  excluded %d 'unknown'-catalyst rows (--include-unknown to keep); "
            "remaining groups: %s",
            n_unk, sorted(df["catalyst_system_type"].unique().tolist()),
        )

    # ── Align features ↔ cleaned on row_id (intersection, preserve order) ──
    feat_df = feat_df[feat_df["row_id"].isin(df["row_id"])].copy()
    df = df[df["row_id"].isin(feat_df["row_id"])].copy()
    feat_df = feat_df.set_index("row_id").loc[df["row_id"].tolist()].reset_index()
    df = df.set_index("row_id").loc[feat_df["row_id"].tolist()].reset_index()
    assert len(df) == len(feat_df), f"row_id alignment mismatch: {len(df)} vs {len(feat_df)}"

    # ── y + groups ──────────────────────────────────────────────────────────
    y = (df["yield (%)"].to_numpy(dtype=np.float64) / 100.0).clip(0.0, 1.0)
    groups = df["catalyst_system_type"].to_numpy()
    assert not pd.isna(groups).any(), "catalyst_system_type contains NaN after filtering"

    # ── X_num ───────────────────────────────────────────────────────────────
    num_cols = [
        c for c in feat_df.columns
        if pd.api.types.is_numeric_dtype(feat_df[c])
        and c not in NUMERIC_EXCLUDE_COLS
    ]
    X_num = feat_df[num_cols].fillna(0).to_numpy(dtype=np.float64)

    # ── PCL-AE latent ───────────────────────────────────────────────────────
    z_pcl_raw = np.load(PCL_DIR / "pcl_ae_latent.npy").astype(np.float64)
    pcl_row_id = pd.read_csv(PCL_DIR / "row_id.csv", encoding="utf-8-sig")
    if z_pcl_raw.shape[0] != len(pcl_row_id):
        raise ValueError(
            f"PCL-AE latent/row_id mismatch: latent={z_pcl_raw.shape[0]} "
            f"vs row_id={len(pcl_row_id)}"
        )
    # Align latent rows to current df order via row_id.
    pcl_index = dict(zip(pcl_row_id["row_id"].tolist(), range(len(pcl_row_id))))
    try:
        latent_pos = np.array([pcl_index[rid] for rid in df["row_id"].tolist()])
    except KeyError as ex:
        missing = set(df["row_id"]) - set(pcl_row_id["row_id"])
        raise ValueError(
            f"{len(missing)} row_id(s) in df are missing from PCL-AE row_id.csv "
            f"(e.g. {list(missing)[:3]}). Re-run train_pcl_ae.py after tier-1."
        ) from ex
    z_pcl = z_pcl_raw[latent_pos]

    X_full = np.hstack([X_num, z_pcl]).astype(np.float64)

    logger.info("  df            : %d rows × %d cols", *df.shape)
    logger.info("  X_num         : %s", X_num.shape)
    logger.info("  z_pcl         : %s", z_pcl.shape)
    logger.info("  X_full (F2)   : %s", X_full.shape)
    logger.info("  catalyst grps : %s",
                pd.Series(groups).value_counts().to_dict())

    return LoadedData(df, X_num, z_pcl, X_full, y, groups)


# ────────────────────────────────────────────────────────────────────────────
# Experiment core
# ────────────────────────────────────────────────────────────────────────────
def build_folds(data: LoadedData, n_folds: int
                ) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Fixed GroupKFold splits shared across all training sizes."""
    gkf = GroupKFold(n_splits=n_folds)
    return list(gkf.split(data.X_full, data.y, groups=data.groups))


def run_one(
    data: LoadedData,
    folds: List[Tuple[np.ndarray, np.ndarray]],
    n_latent: int,
    n_num: int,
    train_sizes: Tuple[int, ...],
    n_seeds: int,
) -> pd.DataFrame:
    """Execute the full learning-curve experiment and return a tidy DataFrame."""
    records: List[Dict] = []

    # Build a per-run registry that closes over the latent/num dims for ANN.
    ann_factory: Callable[[int], DualBranchANN] = (
        lambda seed: build_ann(seed, n_latent=n_latent, n_num=n_num)
    )
    models: Dict[str, Callable[[int], object]] = {
        **MODEL_REGISTRY,
        "DualBranchANN": ann_factory,
    }

    # How big can any fold's train split be? (drives the upper bound on sizes)
    max_train_per_fold = max(len(tr) for tr, _ in folds)
    effective_sizes = [s for s in train_sizes if s <= max_train_per_fold]
    if len(effective_sizes) < len(train_sizes):
        logger.info(
            "Clipping TRAIN_SIZES: %s → %s (max fold-train=%d)",
            train_sizes, effective_sizes, max_train_per_fold,
        )

    t0 = time.time()
    for size in effective_sizes:
        size_records: List[Dict[str, float]] = []
        t_size = time.time()

        for seed_i in range(n_seeds):
            seed = seed_i * 111 + 42

            for fold_i, (tr_base, te_base) in enumerate(folds):
                rng = np.random.default_rng(seed * 1000 + fold_i)

                if size >= len(tr_base):
                    tr_idx = tr_base
                else:
                    tr_idx = stratified_sample(tr_base, data.groups, size, rng)

                X_tr, y_tr = data.X_full[tr_idx], data.y[tr_idx]
                X_te, y_te = data.X_full[te_base], data.y[te_base]

                sc = StandardScaler()
                X_tr_s = sc.fit_transform(X_tr).astype(np.float64)
                X_te_s = sc.transform(X_te).astype(np.float64)

                for model_name, model_fn in models.items():
                    try:
                        if model_name == "DualBranchANN":
                            pred = _evaluate_ann(
                                model_fn, seed + fold_i, X_tr_s, y_tr, X_te_s,
                            )
                        else:
                            m = model_fn(seed=seed + fold_i)
                            pred = _evaluate_sklearn(m, X_tr_s, y_tr, X_te_s)

                        rec = {
                            "model": model_name,
                            "n_train": int(len(tr_idx)),
                            "n_test": int(len(te_base)),
                            "seed": int(seed),
                            "fold": int(fold_i),
                            "r2": safe_r2(y_te, pred),
                            "mae": float(mean_absolute_error(y_te, pred)),
                            "rmse": float(np.sqrt(mean_squared_error(y_te, pred))),
                            "pearson_r": safe_pearson(y_te, pred),
                            "fit_seconds": 0.0,  # placeholder for future timing
                        }
                    except Exception as ex:  # noqa: BLE001 — keep going on model failure
                        logger.warning(
                            "skip %s fold=%d seed=%d n_train=%d: %s",
                            model_name, fold_i, seed, len(tr_idx), ex,
                        )
                        continue

                    records.append(rec)
                    size_records.append({"model": model_name, "r2": rec["r2"]})

        # Per-size summary line
        size_df = pd.DataFrame(size_records)
        elapsed_size = time.time() - t_size
        for m in models:
            r2s = size_df.loc[size_df["model"] == m, "r2"].to_numpy()
            if len(r2s):
                logger.info(
                    "  n_train=%5d  %-15s  R²=%.4f ± %.4f   (%.1fs)",
                    size, m, float(np.mean(r2s)), float(np.std(r2s)), elapsed_size,
                )

    elapsed_total = time.time() - t0
    logger.info("Total experiment elapsed: %.1f min", elapsed_total / 60.0)
    return pd.DataFrame(records)


# ────────────────────────────────────────────────────────────────────────────
# Aggregation + reporting
# ────────────────────────────────────────────────────────────────────────────
def make_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Mean / std per (model, n_train), with obs-count and best-seed stats."""
    if df.empty:
        return pd.DataFrame(columns=[
            "model", "n_train", "r2_mean", "r2_std",
            "mae_mean", "rmse_mean", "pearson_mean", "n_obs",
        ])
    return (
        df.groupby(["model", "n_train"], as_index=False)
        .agg(
            r2_mean=("r2", "mean"),
            r2_std=("r2", "std"),
            mae_mean=("mae", "mean"),
            rmse_mean=("rmse", "mean"),
            pearson_mean=("pearson_r", "mean"),
            n_obs=("r2", "count"),
        )
        .sort_values(["model", "n_train"])
        .reset_index(drop=True)
    )


def write_outputs(df_results: pd.DataFrame, df_summary: pd.DataFrame) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    p_results = OUT_DIR / CSV_RESULTS
    p_summary = OUT_DIR / CSV_SUMMARY
    df_results.to_csv(p_results, index=False, encoding="utf-8-sig")
    df_summary.to_csv(p_summary, index=False, encoding="utf-8-sig")
    logger.info("Wrote %s  (%d rows)", p_results, len(df_results))
    logger.info("Wrote %s  (%d rows)", p_summary, len(df_summary))


def print_console_table(df_summary: pd.DataFrame) -> None:
    if df_summary.empty:
        logger.info("(no rows to summarise)")
        return

    models = list(MODEL_REGISTRY) + ["DualBranchANN"]
    pivot = df_summary.pivot(index="n_train", columns="model", values="r2_mean")
    pivot_std = df_summary.pivot(index="n_train", columns="model", values="r2_std")

    logger.info("=" * 78)
    logger.info("  Sample-Size Sensitivity — Summary (R² mean ± std)")
    logger.info("=" * 78)
    header = f"{'n_train':>8}" + "".join(f"  {m:>16}" for m in models)
    logger.info(header)
    logger.info("-" * len(header))
    for n in sorted(pivot.index):
        row = f"{int(n):8d}"
        for m in models:
            if m in pivot.columns and n in pivot.index:
                v = pivot.loc[n, m]
                s = pivot_std.loc[n, m] if m in pivot_std.columns else 0.0
                row += f"  {v:>7.4f}±{s:>5.3f}"
            else:
                row += f"  {'—':>16}"
        logger.info(row)


# ────────────────────────────────────────────────────────────────────────────
# Plot
# ────────────────────────────────────────────────────────────────────────────
def plot_learning_curves(df_summary: pd.DataFrame) -> None:
    if df_summary.empty:
        logger.warning("summary is empty; skipping plot")
        return

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.font_manager as fm
        import matplotlib.pyplot as plt
    except Exception as ex:  # noqa: BLE001
        logger.warning("matplotlib unavailable, skipping plot: %s", ex)
        return

    # CJK font
    avail = [f.name for f in fm.fontManager.ttflist]
    cjk = [
        "simhei", "wenquanyi", "yahei", "noto sans cjk", "noto sans sc",
        "microsoft yahei", "source han", "arial unicode ms",
    ]
    chosen = next((f for f in avail if any(k in f.lower() for k in cjk)), None)
    plt.rcParams["font.sans-serif"] = (
        [chosen] if chosen else ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
    )
    plt.rcParams["axes.unicode_minus"] = False

    models = list(df_summary["model"].unique())
    colors = {"XGB": "#1f77b4", "LGBM": "#ff7f0e",
              "RF": "#2ca02c", "DualBranchANN": "#d62728"}
    markers = {"XGB": "o", "LGBM": "s", "RF": "^", "DualBranchANN": "D"}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    ax = axes[0]
    for m in models:
        sub = df_summary[df_summary["model"] == m].sort_values("n_train")
        ax.errorbar(
            sub["n_train"], sub["r2_mean"], yerr=sub["r2_std"].fillna(0),
            label=m, color=colors.get(m, "gray"), marker=markers.get(m, "x"),
            capsize=3, linewidth=1.5, markersize=5,
        )
    ax.set_xlabel("Training set size (n_train)", fontsize=11)
    ax.set_ylabel("$R^2$", fontsize=11)
    ax.set_title("Learning Curve: $R^2$ vs Training Size", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.1, 0.95)

    ax2 = axes[1]
    for m in models:
        sub = df_summary[df_summary["model"] == m].sort_values("n_train")
        ax2.plot(
            sub["n_train"], sub["mae_mean"],
            label=m, color=colors.get(m, "gray"), marker=markers.get(m, "x"),
            linewidth=1.5, markersize=5,
        )
    ax2.set_xlabel("Training set size (n_train)", fontsize=11)
    ax2.set_ylabel("MAE", fontsize=11)
    ax2.set_title("Learning Curve: MAE vs Training Size", fontsize=12)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    fig.suptitle(
        "Sample-Size Sensitivity — CO2 Cycloaddition ML Pipeline",
        fontsize=13, y=1.01,
    )
    fig.tight_layout()

    p_png = OUT_DIR / FIG_PNG
    p_pdf = OUT_DIR / FIG_PDF
    fig.savefig(p_png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(p_pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("Saved %s", p_png)
    logger.info("Saved %s", p_pdf)


# ────────────────────────────────────────────────────────────────────────────
# Orchestration
# ────────────────────────────────────────────────────────────────────────────
def outputs_already_present() -> bool:
    return (OUT_DIR / CSV_RESULTS).exists() and (OUT_DIR / CSV_SUMMARY).exists()


def parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Sample-size / learning-curve analysis (303)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--force", action="store_true",
                   help="Overwrite existing CSV/figure outputs.")
    p.add_argument("--include-unknown", action="store_true", default=True,
                   help="Keep rows with catalyst_system_type=='unknown'. "
                        "Default ON: needed for GroupKFold(n=5) which needs ≥5 groups.")
    p.add_argument("--exclude-unknown", action="store_true",
                   help="Drop unknown-catalyst rows (only safe if --n-folds <= 4).")
    p.add_argument("--quick", action="store_true",
                   help="Smoke-test: only n_train ∈ {50, 200, 800}.")
    p.add_argument("--seeds", type=int, default=N_SEEDS,
                   help="Number of repeats per (model, n_train, fold).")
    return p.parse_args(argv)


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format=LOG_FMT, datefmt=LOG_DATEFMT)
    # Quiet down chatty libs.
    for name in ("lightgbm", "xgboost", "matplotlib"):
        logging.getLogger(name).setLevel(logging.WARNING)


def main(argv: List[str] | None = None) -> int:
    configure_logging()
    args = parse_args(sys.argv[1:] if argv is None else argv)

    logger.info("=" * 70)
    logger.info("303 — Sample-Size Sensitivity (Learning Curve)")
    logger.info("=" * 70)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Skip-when-already-done guard ────────────────────────────────────────
    if not args.force and outputs_already_present():
        existing = pd.read_csv(OUT_DIR / CSV_RESULTS)
        logger.info("[SKIP] outputs already present (%d rows in %s). "
                    "Use --force to overwrite.", len(existing), CSV_RESULTS)
        return 0

    # ── Load data ────────────────────────────────────────────────────────────
    data = load_data(
        include_unknown=args.include_unknown,
        exclude_unknown=args.exclude_unknown,
    )

    # ── Build fixed folds ───────────────────────────────────────────────────
    folds = build_folds(data, n_folds=N_FOLDS)
    for i, (tr, te) in enumerate(folds):
        logger.info("Fold %d: train=%d  test=%d  (groups=%d)",
                    i, len(tr), len(te),
                    len(np.unique(data.groups[np.concatenate([tr, te])])))

    # ── Choose train-size grid ──────────────────────────────────────────────
    train_sizes = TRAIN_SIZES_QUICK if args.quick else TRAIN_SIZES_FULL
    if args.quick:
        logger.info("--quick mode: limited grid %s", train_sizes)

    # ── Run experiment ──────────────────────────────────────────────────────
    df_results = run_one(
        data=data,
        folds=folds,
        n_latent=data.n_latent,
        n_num=data.n_num,
        train_sizes=train_sizes,
        n_seeds=args.seeds,
    )

    # ── Aggregate + save ────────────────────────────────────────────────────
    df_summary = make_summary(df_results)
    write_outputs(df_results, df_summary)
    print_console_table(df_summary)
    plot_learning_curves(df_summary)

    logger.info("Outputs in %s/", OUT_DIR)
    logger.info("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())