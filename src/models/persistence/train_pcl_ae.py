# -*- coding: utf-8 -*-
"""train_pcl_ae.py — Train PCL-AE on the full DRFP and save 128-D latent to disk.

This script is the cache layer of the CO2 pipeline (tier_pcl in
run_pipeline_v2.ps1). It mirrors the architectures and training protocol
used inside 201_ablation.py (Stage 2, λ sweep), so the latent cached here
is identical (modulo seed=42 vs ablation seed scan) to the best-λ
configuration validated by the ablation.

Outputs:
    results_pcl_ae/pcl_ae_latent.npy           (n, 128)
    results_pcl_ae/standard_ae_latent.npy      (n, 128)
    results_pcl_ae/row_id.csv                  (alignment index)

Notes
-----
- n_rows is determined dynamically from co2_drfp_xtb_extended.csv at runtime
  (filter: extraction_status=='valid', yield>0).
- BEST_LAMBDA_PROP is auto-loaded by src/config.py from
  results_lambda_ablation/lambda_results.csv (max DualANN R²).
- The PCL-AE model and train loop are imported from 201_ablation.py to
  guarantee byte-identical training (apart from the seed).

Usage
-----
    python train_pcl_ae.py              # train both AE-128 and PCL-AE-128
    python train_pcl_ae.py --force      # overwrite existing latents
    python train_pcl_ae.py --standard-only
    python train_pcl_ae.py --pcl-only
"""
from __future__ import annotations

import argparse
import importlib.util
import io
import logging
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(os.environ.get(
    "CO2_PROJECT_ROOT", r"D:\machine-learning\CO2-cycloaddition"))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Force UTF-8 stdout on Windows PowerShell (cp936 default)
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

from src.config import BEST_LAMBDA_PROP, BEST_LATENT_DIM   # noqa: E402
from utils_rxn import read_drfp                            # noqa: E402

# ---------------------------------------------------------------------------
# Load the single source of truth for AE classes / train functions from
# 201_ablation.py via importlib. This is intentional: the project uses
# 201_ablation.py as a library for its model definitions, NOT a separate
# PCL_AE_modules package, because (a) src/data has no __init__.py package
# machinery, and (b) keeping both training scripts in lock-step guarantees
# the cached latent is identical to the λ-sweep latent.
# ---------------------------------------------------------------------------
_AB_PATH = PROJECT_ROOT / "src" / "data" / "201_ablation.py"
_spec = importlib.util.spec_from_file_location("_201_ablation_mod", _AB_PATH)
if _spec is None or _spec.loader is None:    # pragma: no cover
    raise ImportError(f"Could not load 201_ablation.py from {_AB_PATH}")
_mod_201 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod_201)

StandardAE = _mod_201.StandardAE
PropertyCoLearningAE = _mod_201.PropertyCoLearningAE
train_standard_ae = _mod_201.train_standard_ae
train_pcl_ae = _mod_201.train_pcl_ae

# Import improved PCL-AE from utils_benchmark
from utils_benchmark import train_pcl_ae_improved, ImprovedPCLAE

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
DATA_CSV = PROJECT_ROOT / "results" / "results_cho_diagnostic" / "co2_drfp_xtb_extended.csv"
OUT_DIR = PROJECT_ROOT / "results_pcl_ae"
ROW_ID_CSV = OUT_DIR / "row_id.csv"
STANDARD_LATENT_NPY = OUT_DIR / "standard_ae_latent.npy"
PCL_LATENT_NPY = OUT_DIR / "pcl_ae_latent.npy"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42

LATENT_DIM = int(BEST_LATENT_DIM)
LAMBDA_PROP = float(BEST_LAMBDA_PROP)


def configure_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
        force=True,
    )


log = logging.getLogger("train_pcl_ae")


# ---------------------------------------------------------------------------
# Data loading (matches 201_ablation.py's Stage 2 data path exactly)
# ---------------------------------------------------------------------------
def load_features() -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Load DRFP and yield from co2_drfp_xtb_extended.csv.

    Returns:
        X_drfp_s: standardized DRFP, (n, 2048) float32
        y:        normalized yield in [0, 1], (n,) float32
        df:       filtered dataframe (for row_id alignment)
    """
    log.info("Loading data from %s", DATA_CSV)
    df = pd.read_csv(DATA_CSV, encoding="utf-8-sig")
    df = df[df["extraction_status"] == "valid"].copy()
    df = df.dropna(subset=["yield (%)"])
    df = df[df["yield (%)"] > 0].reset_index(drop=True)
    y = df["yield (%)"].to_numpy(dtype=np.float32) / 100.0
    log.info("  %d valid rows", len(df))

    log.info("Decoding DRFP ...")
    arr = []
    for s in df["drfp"]:
        fp = read_drfp(s)
        if fp is None or fp.size == 0:
            arr.append(np.zeros(2048, dtype=np.float32))
        else:
            arr.append(fp.astype(np.float32))
    X_drfp = np.stack(arr).astype(np.float32)
    log.info("  DRFP shape: %s", X_drfp.shape)

    scaler = StandardScaler()
    X_drfp_s = scaler.fit_transform(X_drfp).astype(np.float32)
    return X_drfp_s, y, df


def save_row_id(df: pd.DataFrame) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"row_id": df["row_id"].values}).to_csv(ROW_ID_CSV, index=False)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_and_save_standard(X: np.ndarray, force: bool) -> np.ndarray:
    if STANDARD_LATENT_NPY.exists() and not force:
        log.info("[standard] %s already exists; loading (use --force to overwrite)",
                 STANDARD_LATENT_NPY)
        return np.load(STANDARD_LATENT_NPY)

    log.info("[1/2] Training Standard AE-128 ...")
    t0 = time.time()
    z = train_standard_ae(X, latent_dim=LATENT_DIM, epochs=100, batch_size=128, lr=1e-3)
    log.info("  done in %.1fs  latent: %s", time.time() - t0, z.shape)
    np.save(STANDARD_LATENT_NPY, z)
    return z


def train_and_save_pcl(X: np.ndarray, y: np.ndarray, force: bool) -> np.ndarray:
    if PCL_LATENT_NPY.exists() and not force:
        log.info("[pcl] %s already exists; loading (use --force to overwrite)",
                 PCL_LATENT_NPY)
        return np.load(PCL_LATENT_NPY)

    log.info("[2/2] Training PCL-AE-128 (lambda = %s) ...", LAMBDA_PROP)
    t0 = time.time()
    z = train_pcl_ae(
        X, y,
        latent_dim=LATENT_DIM, lambda_prop=LAMBDA_PROP,
        epochs=150, batch_size=128, lr=1e-3,
        pos_weight=10.0, seed=SEED,
    )
    log.info("  done in %.1fs  latent: %s", time.time() - t0, z.shape)
    np.save(PCL_LATENT_NPY, z)
    return z


def train_and_save_improved_pcl(X: np.ndarray, y: np.ndarray, force: bool) -> np.ndarray:
    """Train Improved PCL-AE with VAE-style architecture and Huber Loss.

    This uses the improved model from utils_benchmark which includes:
    - VAE-style latent space with KL divergence regularization
    - Huber Loss for property prediction (robust to outliers)
    - Gradient clipping and cosine annealing LR schedule

    Saves to: results_pcl_ae/improved_pcl_ae_latent.npy
    """
    improved_latent_path = OUT_DIR / "improved_pcl_ae_latent.npy"
    if improved_latent_path.exists() and not force:
        log.info("[improved_pcl] %s already exists; loading (use --force to overwrite)",
                 improved_latent_path)
        return np.load(improved_latent_path)

    log.info("[2/2] Training Improved PCL-AE-128 (VAE-style, Huber Loss) ...")
    t0 = time.time()
    z = train_pcl_ae_improved(
        X, y,
        latent_dim=LATENT_DIM, lambda_prop=LAMBDA_PROP,
        epochs=150, batch_size=128, lr=1e-3,
        beta=0.01,  # KL divergence weight
    )
    log.info("  done in %.1fs  latent: %s", time.time() - t0, z.shape)
    np.save(improved_latent_path, z)
    return z


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train PCL-AE (and/or Standard AE) on full DRFP and cache 128-D latent.",
    )
    p.add_argument(
        "--force", action="store_true",
        help="Overwrite existing latent .npy files.",
    )
    p.add_argument(
        "--standard-only", action="store_true",
        help="Train only the Standard AE; skip PCL-AE.",
    )
    p.add_argument(
        "--pcl-only", action="store_true",
        help="Train only the PCL-AE; skip Standard AE.",
    )
    p.add_argument(
        "--improved-only", action="store_true",
        help="Train only the Improved PCL-AE (VAE-style, Huber Loss); skip others.",
    )
    p.add_argument(
        "--all", action="store_true",
        help="Train Standard AE, PCL-AE, and Improved PCL-AE.",
    )
    p.add_argument(
        "--verbose", action="store_true", help="Enable DEBUG-level logging.",
    )
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    configure_logging(verbose=args.verbose)

    # Reproducibility: numpy + torch
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    log.info("=" * 72)
    log.info("  train_pcl_ae.py  |  device: %s  |  CUDA: %s",
             DEVICE, torch.cuda.is_available())
    log.info("  latent_dim=%d  |  lambda=%s  |  seed=%d",
             LATENT_DIM, LAMBDA_PROP, SEED)
    log.info("=" * 72)

    X, y, df = load_features()
    save_row_id(df)
    log.info("  saved row_id.csv (%d rows)", len(df))

    do_standard = not args.pcl_only
    do_pcl = not args.standard_only
    if args.standard_only and args.pcl_only:
        log.warning("--standard-only and --pcl-only both set; nothing to do")
        return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    if do_standard:
        z_std = train_and_save_standard(X, force=args.force)
    else:
        z_std = None

    if do_pcl:
        z_pcl = train_and_save_pcl(X, y, force=args.force)
    else:
        z_pcl = None

    # Train improved PCL-AE if requested
    z_improved = None
    if args.improved_only or args.all:
        z_improved = train_and_save_improved_pcl(X, y, force=args.force)

    log.info("=" * 72)
    log.info("  DONE — total %.1fs", time.time() - t0)
    log.info("  Latents saved to %s", OUT_DIR)
    if z_std is not None:
        log.info("    standard_ae_latent.npy : %s", z_std.shape)
    if z_pcl is not None:
        log.info("    pcl_ae_latent.npy       : %s", z_pcl.shape)
    if z_improved is not None:
        log.info("    improved_pcl_ae_latent.npy : %s", z_improved.shape)
    log.info("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())