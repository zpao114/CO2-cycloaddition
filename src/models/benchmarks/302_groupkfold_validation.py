# -*- coding: utf-8 -*-
"""
302_groupkfold_validation.py
============================

Stage 3 — SSTS (Subset Splitting Training Strategy) evaluation
plus a GroupKFold / LOCO-CV external-validity check on each
catalyst subset.

Scientific question
-------------------
How does model generalisation degrade when we test on a catalyst
the model has *never seen during training*? And does splitting the
data into homogeneous subsets (catalyst type, reactant, IL cation
subtype) buy us any LOCO-CV robustness compared to a single
all-data baseline?

Strategies compared
-------------------
  Baseline : single model trained on all data
  SSTS-A   : split by catalyst_system_type → one specialised model per type
  SSTS-B   : split by reactant_name (5 epoxides) → one model per reactant
  SSTS-C   : IL cation subtypes (ammonium / imidazolium / other) → one model per subtype

Evaluation protocol
-------------------
  - LOCO-CV: Leave-One-Catalyst-Out = GroupKFold(n_splits = n_catalysts).
            Each fold's test set is one catalyst the model has never seen.
            Up to 30 folds per subset (random sub-sample if more catalysts exist).
  - Random KFold: 5-fold leak control (random splits, not grouped).
            R² near 0 in random KFold means the model is no better than
            predicting the mean yield.
  - Yield is normalised to [0, 1] by default so that R² ∈ (−∞, 1].
    Use --raw-yield to keep the 0–100 scale (legacy behaviour).

Outputs (results_groupkfold_validation/)
----------------------------------------
  ML_groupkfold_results.csv — one row per (strategy, fold-protocol)

Inputs (from tier-1)
--------------------
  results/results_cho_diagnostic/co2_drfp_xtb_extended.csv
  results_pcl_ae/pcl_ae_latent.npy   (optional; used if available)
  results_pcl_ae/row_id.csv

Usage
-----
  python 302_groupkfold_validation.py
  python 302_groupkfold_validation.py --quick        # only Baseline + SSTS-A
  python 302_groupkfold_validation.py --models RF,LGBM
  python 302_groupkfold_validation.py --seeds 1       # fast smoke
  python 302_groupkfold_validation.py --raw-yield     # legacy 0-100 scale

Typical runtime: ~5 min for all 4 strategies × 2 models × 2 protocols on
a 2,500-row dataset.
"""

from __future__ import annotations

import argparse
import io
import logging
import os
import re
import sys
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold, LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler

try:
    from lightgbm import LGBMRegressor
    HAS_LGBM = True
except ImportError:  # pragma: no cover
    HAS_LGBM = False

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:  # pragma: no cover
    HAS_XGB = False

# ── Paths ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(
    os.environ.get("CO2_PROJECT_ROOT", r"D:\machine-learning\CO2-cycloaddition")
)
DATA_CSV = PROJECT_ROOT / "results" / "results_cho_diagnostic" / "co2_drfp_xtb_extended.csv"
PCL_DIR = PROJECT_ROOT / "results_pcl_ae"
OUT_DIR = PROJECT_ROOT / "results_groupkfold_validation"
CSV_OUT = "ML_groupkfold_results.csv"

# ── Constants ──────────────────────────────────────────────────────────────
CATALYST_SYSTEM_TYPES = ("ionic_liquid", "metal_halide", "mixed_system", "organic_base")
MIN_SAMPLES_FOR_LOCO = 50
MIN_SAMPLES_FOR_RAND = 20
MIN_SAMPLES_FOR_REACTANT = 30
MIN_SAMPLES_FOR_SUBTYPE = 50
MIN_TRAIN_FOLDS = 5
MAX_FOLDS_PER_SUBSET = 30
DEFAULT_SEEDS = (42, 123, 456)

LOG_FMT = "%(asctime)s [%(levelname)s] %(message)s"
LOG_DATEFMT = "%H:%M:%S"
logger = logging.getLogger("302_ssts")


# ── Data classes ───────────────────────────────────────────────────────────
@dataclass
class FoldResult:
    label: str
    model: str
    n_samples: int
    n_catalysts: int
    n_folds: int
    r2: float
    r2_std: float
    rmse: float
    mae: float
    extra: str = ""

    def to_row(self) -> Dict[str, object]:
        return asdict(self)


# ── Data loading ───────────────────────────────────────────────────────────
def _safe_read_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    return pd.read_csv(path, encoding="utf-8-sig")


def load_dataset(norm_yield: bool = True) -> Tuple[pd.DataFrame, str]:
    """Load + clean the master CSV. Return df and the yield column name."""
    if not DATA_CSV.exists():
        raise FileNotFoundError(
            f"Master CSV not found: {DATA_CSV}.  Run tier-1 first."
        )
    df = pd.read_csv(DATA_CSV, encoding="utf-8-sig")
    n0 = len(df)

    # Required columns
    required = ("yield (%)", "catalyst_system_type", "catalyst_1_name")
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Master CSV missing required columns: {missing}")

    df = df[df["extraction_status"] == "valid"].copy() if "extraction_status" in df.columns else df
    df = df.dropna(subset=["yield (%)"])
    df = df[df["yield (%)"] > 0].reset_index(drop=True)
    df = df[df["yield (%)"].notna()].copy()
    df["yield (%)"] = pd.to_numeric(df["yield (%)"], errors="coerce")
    df = df[df["yield (%)"].notna()].copy()
    logger.info("  master CSV: %d → %d rows after cleaning", n0, len(df))

    # Drop 'unknown' catalyst rows so GroupKFold has a finite group count.
    n_unk = int((df["catalyst_system_type"] == "unknown").sum())
    df = df[df["catalyst_system_type"] != "unknown"].copy()
    logger.info("  excluded %d 'unknown'-catalyst rows (--include-unknown to keep)",
                n_unk)

    # Optional yield normalisation (0–1).
    if norm_yield:
        df["yield (%)"] = df["yield (%)"].clip(0, 100) / 100.0

    return df, "yield (%)"


# ── Featurisation ──────────────────────────────────────────────────────────
def _drfp_to_array(s: str, n_bits: int = 2048) -> np.ndarray:
    """Parse a DRFP string column into a numpy array. Safe (no eval)."""
    if pd.isna(s) or not str(s).strip():
        return np.zeros(n_bits, dtype=np.float32)
    txt = str(s).strip().strip("[]")
    parts = [p.strip() for p in txt.split(",") if p.strip()]
    if not parts:
        return np.zeros(n_bits, dtype=np.float32)
    arr = np.zeros(n_bits, dtype=np.float32)
    try:
        idx = np.array([int(float(p)) for p in parts[:n_bits]], dtype=np.int64)
        idx = idx[(idx >= 0) & (idx < n_bits)]
        arr[idx] = 1.0
    except (ValueError, TypeError):
        # fall back: ignore this row
        return np.zeros(n_bits, dtype=np.float32)
    return arr


def _condition_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    cond_cols: List[str] = []
    for col in ("temperature", "pressure", "time", "loading"):
        for c in df.columns:
            if col in c.lower() and c != "yield (%)":
                cond_cols.append(c)
                break
    return cond_cols, [c for c in cond_cols]


def featurize_baseline(df_sub: pd.DataFrame, n_bits: int = 2048) -> Tuple[np.ndarray, np.ndarray]:
    """DRFP(2048) + 4 condition features."""
    drfp_col = next((c for c in df_sub.columns if c.lower() == "drfp"), "drfp")
    X_drfp = np.stack([_drfp_to_array(s, n_bits) for s in df_sub[drfp_col].values])
    cond_cols = _condition_columns(df_sub)[0]
    if cond_cols:
        X_cond = df_sub[cond_cols].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy(
            dtype=np.float32
        )
    else:
        X_cond = np.zeros((len(df_sub), 0), dtype=np.float32)
    X = np.hstack([X_drfp, X_cond]).astype(np.float32)
    y = df_sub["yield (%)"].to_numpy(dtype=np.float32)
    return X, y


# ── Model builders ─────────────────────────────────────────────────────────
def build_model(name: str, fold_i: int, seed: int) -> object:
    """Return a *fresh* estimator for the given fold."""
    s = seed * 1000 + fold_i
    if name == "LGBM":
        if not HAS_LGBM:
            raise ImportError("lightgbm is not installed")
        return LGBMRegressor(
            n_estimators=400, learning_rate=0.05, max_depth=8,
            num_leaves=64, random_state=s, n_jobs=-1, verbosity=-1,
        )
    if name == "XGB":
        if not HAS_XGB:
            raise ImportError("xgboost is not installed")
        return xgb.XGBRegressor(
            n_estimators=400, learning_rate=0.05, max_depth=8,
            random_state=s, n_jobs=-1, verbosity=0,
        )
    if name == "RF":
        return RandomForestRegressor(
            n_estimators=200, max_depth=15, random_state=s, n_jobs=-1,
        )
    raise ValueError(f"Unknown model: {name!r}")


MODEL_NAMES = ("LGBM", "RF") if HAS_LGBM else ("RF",)


# ── Fold evaluation core ───────────────────────────────────────────────────
def _make_loco_splits(
    cats: np.ndarray, max_folds: int, rng: np.random.Generator
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Return Leave-One-Catalyst-Out folds, capped at ``max_folds``."""
    unique_cats = np.unique(cats)
    if len(unique_cats) > max_folds:
        keep = rng.choice(unique_cats, size=max_folds, replace=False)
        keep_set = set(keep.tolist())
        logo = LeaveOneGroupOut()
        splits = [s for s in logo.split(np.zeros(len(cats)), groups=cats)
                  if cats[s[1][0]] in keep_set]
    else:
        gkf = GroupKFold(n_splits=len(unique_cats))
        splits = list(gkf.split(np.zeros(len(cats)), groups=cats))
    return splits


def _make_random_splits(n: int, k: int, seed: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    kf = KFold(n_splits=min(k, max(2, n // 10)), shuffle=True, random_state=seed)
    return list(kf.split(np.zeros(n)))


def _evaluate_fold(
    model_name: str,
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_va: np.ndarray, y_va: np.ndarray,
    fold_i: int, seed: int,
) -> Tuple[float, float, float]:
    sc = StandardScaler()
    X_tr = sc.fit_transform(X_tr)
    X_va = sc.transform(X_va)
    model = build_model(model_name, fold_i, seed)
    model.fit(X_tr, y_tr)
    pred = np.asarray(model.predict(X_va), dtype=np.float32)
    r2 = float(r2_score(y_va, pred))
    if not np.isfinite(r2):
        r2 = 0.0
    rmse = float(np.sqrt(mean_squared_error(y_va, pred)))
    mae = float(mean_absolute_error(y_va, pred))
    return r2, rmse, mae


def evaluate_protocol(
    df_subset: pd.DataFrame,
    label: str,
    model_name: str,
    splits: List[Tuple[np.ndarray, np.ndarray]],
    seed: int,
) -> Optional[FoldResult]:
    X, y = featurize_baseline(df_subset)
    cats = df_subset["catalyst_1_name"].to_numpy()
    r2s, rmses, maes = [], [], []
    for fold_i, (tr_idx, va_idx) in enumerate(splits):
        if len(va_idx) < 2 or len(tr_idx) < MIN_TRAIN_FOLDS:
            continue
        try:
            r2, rmse, mae = _evaluate_fold(
                model_name, X[tr_idx], y[tr_idx], X[va_idx], y[va_idx], fold_i, seed,
            )
        except Exception as ex:  # noqa: BLE001
            logger.warning("skip %s fold=%d: %s", label, fold_i, ex)
            continue
        r2s.append(r2); rmses.append(rmse); maes.append(mae)
    if not r2s:
        return None
    return FoldResult(
        label=label,
        model=model_name,
        n_samples=int(len(df_subset)),
        n_catalysts=int(len(np.unique(cats))),
        n_folds=int(len(r2s)),
        r2=float(np.mean(r2s)),
        r2_std=float(np.std(r2s)) if len(r2s) > 1 else 0.0,
        rmse=float(np.mean(rmses)),
        mae=float(np.mean(maes)),
        extra=f"protocol={('LOCO' if 'rand' not in label else 'random')}",
    )


# ── Public evaluators ──────────────────────────────────────────────────────
def evaluate_loco(
    df_subset: pd.DataFrame, label: str, model_name: str,
    seed: int, max_folds: int = MAX_FOLDS_PER_SUBSET,
) -> Optional[FoldResult]:
    cats = df_subset["catalyst_1_name"].to_numpy()
    if len(df_subset) < MIN_SAMPLES_FOR_LOCO:
        logger.info("  skip %s: n=%d < %d", label, len(df_subset), MIN_SAMPLES_FOR_LOCO)
        return None
    rng = np.random.default_rng(seed)
    splits = _make_loco_splits(cats, max_folds, rng)
    logger.info("  %-35s LOCO folds=%d", label, len(splits))
    return evaluate_protocol(df_subset, label, model_name, splits, seed)


def evaluate_random_kfold(
    df_subset: pd.DataFrame, label: str, model_name: str, seed: int, k: int = 5,
) -> Optional[FoldResult]:
    if len(df_subset) < MIN_SAMPLES_FOR_RAND:
        logger.info("  skip %s: n=%d < %d", label, len(df_subset), MIN_SAMPLES_FOR_RAND)
        return None
    splits = _make_random_splits(len(df_subset), k, seed)
    logger.info("  %-35s rand folds=%d", label, len(splits))
    return evaluate_protocol(df_subset, label, model_name, splits, seed)


# ── SSTS strategies ────────────────────────────────────────────────────────
def ssts_by_catalyst_type(
    df: pd.DataFrame, model_name: str, seed: int,
) -> List[FoldResult]:
    """SSTS-A — one specialised model per catalyst_system_type."""
    out: List[FoldResult] = []
    for cat in CATALYST_SYSTEM_TYPES:
        sub = df[df["catalyst_system_type"] == cat]
        for label, fn in (
            (f"SSTS-A_{cat}",       lambda s=sub: evaluate_loco(s, f"SSTS-A_{cat}", model_name, seed)),
            (f"SSTS-A_{cat}_rand",  lambda s=sub: evaluate_random_kfold(s, f"SSTS-A_{cat}_rand", model_name, seed)),
        ):
            r = fn()
            if r is not None:
                out.append(r)
    return out


def ssts_by_reactant(
    df: pd.DataFrame, model_name: str, seed: int,
) -> List[FoldResult]:
    """SSTS-B — one specialised model per reactant (epoxide)."""
    out: List[FoldResult] = []
    df_il = df[df["catalyst_system_type"] == "ionic_liquid"]
    for rname, grp in df_il.groupby("reactant_name"):
        if len(grp) < MIN_SAMPLES_FOR_REACTANT:
            continue
        safe = re.sub(r"[^a-zA-Z0-9]", "_", str(rname))[:30]
        label = f"SSTS-B_{safe}"
        for fn_label, fn in (
            (label,      lambda g=grp, l=label: evaluate_loco(g, l, model_name, seed)),
            (label + "_rand", lambda g=grp, l=label: evaluate_random_kfold(g, l + "_rand", model_name, seed)),
        ):
            r = fn()
            if r is not None:
                out.append(r)
    return out


def ssts_by_cation_subtype(
    df: pd.DataFrame, model_name: str, seed: int,
) -> List[FoldResult]:
    """SSTS-C — IL cation subtypes (ammonium / imidazolium / other)."""
    out: List[FoldResult] = []
    df_il = df[df["catalyst_system_type"] == "ionic_liquid"].copy()
    if df_il.empty:
        return out
    try:
        sys.path.insert(0, str(PROJECT_ROOT))
        sys.path.insert(0, str(PROJECT_ROOT / "src"))
        from utils_features import _detect_cation_subtype, _parse_catalyst_smiles  # type: ignore
    except Exception as ex:  # noqa: BLE001
        logger.warning("  utils_features not importable, SSTS-C skipped: %s", ex)
        return out

    subtypes: List[str] = []
    for _, row in df_il.iterrows():
        smi = str(row.get("catalyst_1_name", "")).strip()
        parsed = _parse_catalyst_smiles(smi) if smi else {}
        cation = (parsed.get("components") or {}).get("cation", "")
        subtypes.append(_detect_cation_subtype(cation))
    df_il = df_il.assign(_st=subtypes)

    for st, grp in df_il.groupby("_st"):
        if len(grp) < MIN_SAMPLES_FOR_SUBTYPE:
            continue
        label = f"SSTS-C_{st}"
        for lbl, fn in (
            (label,      lambda g=grp, l=label: evaluate_loco(g, l, model_name, seed)),
            (label + "_rand", lambda g=grp, l=label: evaluate_random_kfold(g, l + "_rand", model_name, seed)),
        ):
            r = fn()
            if r is not None:
                out.append(r)
    return out


# ── Summary helpers ────────────────────────────────────────────────────────
def baseline_pair(df: pd.DataFrame, model_name: str, seed: int) -> List[FoldResult]:
    """LOCO + rand on the whole dataset (single global model)."""
    out: List[FoldResult] = []
    r_loco = evaluate_loco(df, "Baseline_all_LOCO", model_name, seed)
    r_rand = evaluate_random_kfold(df, "Baseline_all_rand", model_name, seed)
    if r_loco is not None: out.append(r_loco)
    if r_rand is not None: out.append(r_rand)
    return out


def print_summary_table(df: pd.DataFrame) -> None:
    if df.empty:
        logger.info("(no rows)")
        return
    pivot = df.pivot_table(
        index="label", values=["r2", "r2_std"], aggfunc="first"
    ).sort_values("r2", ascending=False)
    logger.info("=" * 72)
    logger.info("  SSTS LOCO-CV R² ranking (best-first)")
    logger.info("=" * 72)
    logger.info("  %-30s %8s %8s %6s %6s %6s",
                "label", "r2", "±std", "n", "cats", "folds")
    for label, row in pivot.iterrows():
        full = df[df["label"] == label].iloc[0]
        logger.info("  %-30s %+8.4f %8.4f %6d %6d %6d",
                    label[:30],
                    float(row["r2"]), float(row["r2_std"]),
                    int(full["n_samples"]), int(full["n_catalysts"]),
                    int(full["n_folds"]))


def best_subsets(df: pd.DataFrame, k: int = 3) -> pd.DataFrame:
    return (
        df[~df["label"].str.contains("rand")]
          .sort_values("r2", ascending=False)
          .head(k)
          .reset_index(drop=True)
    )


# ── Orchestration ──────────────────────────────────────────────────────────
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stage 3 — SSTS / GroupKFold / LOCO-CV external validity",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--force", action="store_true",
                   help="Overwrite the output CSV even if it already exists.")
    p.add_argument("--quick", action="store_true",
                   help="Run only Baseline + SSTS-A (smoke test).")
    p.add_argument("--seeds", type=int, default=1,
                   help="Number of repeated seeds per (strategy, model).")
    p.add_argument("--models", default="LGBM,RF",
                   help="Comma-separated model list (LGBM, RF, XGB).")
    p.add_argument("--raw-yield", action="store_true",
                   help="Keep yield on the original 0-100 scale (default: 0-1).")
    p.add_argument("--include-unknown", action="store_true",
                   help="Keep 'unknown' catalyst rows (may break GroupKFold).")
    return p.parse_args(argv)


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format=LOG_FMT, datefmt=LOG_DATEFMT)
    for noisy in ("lightgbm", "xgboost"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _requested_models(spec: str) -> List[str]:
    out: List[str] = []
    for tok in spec.split(","):
        tok = tok.strip().upper()
        if not tok:
            continue
        if tok == "LGBM" and not HAS_LGBM:
            logger.warning("LGBM unavailable, skipping")
            continue
        if tok == "XGB" and not HAS_XGB:
            logger.warning("XGB unavailable, skipping")
            continue
        if tok not in ("LGBM", "RF", "XGB"):
            logger.warning("unknown model '%s', skipping", tok)
            continue
        out.append(tok)
    return out or ["RF"]


def main(argv: Optional[Sequence[str]] = None) -> int:
    configure_logging()
    args = parse_args(argv)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / CSV_OUT
    if not args.force and out_path.exists():
        existing = pd.read_csv(out_path)
        logger.info("[SKIP] %s already exists (%d rows). Use --force to re-run.",
                    out_path, len(existing))
        return 0

    # Encoding + warnings
    if sys.stdout.encoding != "utf-8":
        try:
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass
    warnings.filterwarnings("ignore")

    logger.info("=" * 72)
    logger.info("  302 — SSTS / GroupKFold / LOCO-CV external validity")
    logger.info("=" * 72)

    df, ycol = load_dataset(norm_yield=not args.raw_yield)
    if args.include_unknown:
        df_unknown = pd.read_csv(DATA_CSV, encoding="utf-8-sig")
        df = df_unknown[df_unknown["catalyst_system_type"] == "unknown"].copy()
        if args.raw_yield:
            df["yield (%)"] = df["yield (%)"].clip(0, 100)
        else:
            df["yield (%)"] = df["yield (%)"].clip(0, 100) / 100.0

    n_cats = df["catalyst_1_name"].nunique()
    logger.info("  data: %d rows, %d catalysts, %d substrate-reactants",
                len(df), n_cats, df["reactant_name"].nunique()
                if "reactant_name" in df.columns else 0)
    logger.info("  yield column: %s  (range %.3f … %.3f)",
                ycol, float(df[ycol].min()), float(df[ycol].max()))

    models = _requested_models(args.models)
    seeds = list(DEFAULT_SEEDS[: max(1, args.seeds)])
    logger.info("  models: %s   seeds: %s", models, seeds)

    all_rows: List[FoldResult] = []
    t0 = time.time()
    for model_name in models:
        for seed in seeds:
            logger.info("-" * 72)
            logger.info("  model=%s  seed=%d", model_name, seed)
            logger.info("-" * 72)

            logger.info("  [Baseline]")
            all_rows.extend(baseline_pair(df, model_name, seed))

            if not args.quick:
                logger.info("  [SSTS-A] by catalyst_system_type")
                all_rows.extend(ssts_by_catalyst_type(df, model_name, seed))

                logger.info("  [SSTS-B] by reactant_name")
                all_rows.extend(ssts_by_reactant(df, model_name, seed))

                logger.info("  [SSTS-C] by IL cation subtype")
                all_rows.extend(ssts_by_cation_subtype(df, model_name, seed))
            else:
                logger.info("  --quick: only Baseline + SSTS-A")
                all_rows.extend(ssts_by_catalyst_type(df, model_name, seed))

    elapsed = time.time() - t0
    logger.info("Total elapsed: %.1f min", elapsed / 60.0)

    df_res = pd.DataFrame([r.to_row() for r in all_rows])
    if df_res.empty:
        logger.error("No fold results produced — aborting.")
        return 1

    df_res.to_csv(out_path, index=False, encoding="utf-8-sig")
    logger.info("Wrote %s  (%d rows)", out_path, len(df_res))

    print_summary_table(df_res)
    best = best_subsets(df_res, k=3)
    if not best.empty:
        logger.info("=" * 72)
        logger.info("  Best SSTS subsets (LOCO):")
        for _, r in best.iterrows():
            logger.info("    %s  R²=%+.4f ± %.4f   (n=%d, cats=%d, folds=%d)",
                        r["label"], r["r2"], r["r2_std"],
                        int(r["n_samples"]), int(r["n_catalysts"]),
                        int(r["n_folds"]))
        logger.info("=" * 72)

    logger.info("Outputs in %s/", OUT_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())