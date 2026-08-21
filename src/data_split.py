# -*- coding: utf-8 -*-
"""data_split.py — canonical stratified train/test split + 5-fold CV manifest.

Inputs
------
co2_drfp_xtb_extended.csv          (master table after Tier 1)

Outputs
-------
results/results_data_split/data_split.json         — consumed by 901_substrate_catalyst_matrix.py + paths_audit.py
results/results_data_split/data_split_summary.txt  — human-readable report

NOTE: As of 2026-08-18, paths.DATA_SPLIT_JSON resolves to
    RESULTS_DIR / "results_data_split" = "results/results_data_split/".
    Earlier this docstring said "results/data_split/" — that was a typo,
    never the on-disk location.

Strategy
--------
Hold out a stratified 15% test set (seed=2026, yield-quartile strata) from
the master table; then run stratified 5-fold CV on the 85% remainder with
the same seed and strata. Stratifying by yield bins (not GroupKFold by
catalyst) keeps every fold well-populated because the dataset is
catalyst-rich.

Reproducibility note
--------------------
Manifest byte-equality depends on the installed scikit-learn RNG. Across
recent sklearn versions (>=1.3) the manifest is stable to within ~85%
overlap of fold assignments; seed=2026 is canonical.

Usage
-----
    python data_split.py             # regenerate manifest
    python data_split.py --force     # overwrite existing data_split.json
    python data_split.py --dry-run   # print splits without writing
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

PROJECT_ROOT = os.environ.get("CO2_PROJECT_ROOT", r"D:\machine-learning\CO2-cycloaddition")
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# Force UTF-8 stdout on Windows PowerShell (cp936 default)
if sys.stdout.encoding and sys.stdout.encoding.lower().replace("-", "") != "utf8":
    try:
        if hasattr(sys.stdout, "buffer"):
            sys.stdout = __import__("io").TextIOWrapper(
                sys.stdout.buffer,
                encoding="utf-8",
                errors="replace",
                line_buffering=not sys.stdout.isatty(),
            )
    except Exception:  # pragma: no cover
        pass

from src.paths import (  # noqa: E402
    DATA_SPLIT_JSON,
    DRFP_XTB_EXTENDED_CSV,
    RESULTS_DATA_SPLIT,
    ensure_dir,
)

RANDOM_SEED = 2026
TEST_SIZE = 0.15
N_SPLITS = 5

log = logging.getLogger("data_split")


def _make_yield_strata(y: np.ndarray) -> np.ndarray:
    """Bin continuous yields into 4 strata (Q1..Q4) for stratified splitting."""
    y = np.asarray(y, dtype=float)
    qs = np.quantile(y, [0.25, 0.50, 0.75])
    out = np.zeros(len(y), dtype=int)
    out[(y > qs[0]) & (y <= qs[1])] = 1
    out[(y > qs[1]) & (y <= qs[2])] = 2
    out[y > qs[2]] = 3
    return out


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build canonical stratified train/test manifest + 5-fold CV.",
    )
    p.add_argument("--force", action="store_true",
                   help="Overwrite existing data_split.json.")
    p.add_argument("--dry-run", action="store_true",
                   help="Compute splits but do not write any output.")
    p.add_argument("--verbose", action="store_true", help="Enable DEBUG logging.")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
        force=True,
    )

    log.info("Reading %s", DRFP_XTB_EXTENDED_CSV)
    if not DRFP_XTB_EXTENDED_CSV.exists():
        log.error("%s not found. Run 107_merge_substrate_xtb.py first.",
                  DRFP_XTB_EXTENDED_CSV)
        return 1

    df = pd.read_csv(DRFP_XTB_EXTENDED_CSV, encoding="utf-8-sig")
    df = df[df["extraction_status"] == "valid"].copy()
    df = df.dropna(subset=["yield (%)"])
    df = df[df["yield (%)"] > 0].reset_index(drop=True)

    n_total = len(df)
    log.info("%d valid rows after cleaning", n_total)

    strata = _make_yield_strata(df["yield (%)"].values)
    train_idx, test_idx = train_test_split(
        np.arange(n_total),
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=strata,
    )
    train_idx = sorted(int(i) for i in train_idx)
    test_idx = sorted(int(i) for i in test_idx)
    log.info("Holdout: n_train=%d, n_test=%d", len(train_idx), len(test_idx))

    train_y = df.iloc[train_idx]["yield (%)"].values
    train_strata = _make_yield_strata(train_y)

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
    splits: List[Dict[str, List[int]]] = []
    for fold_i, (tr, va) in enumerate(
        skf.split(np.arange(len(train_idx)), train_strata)
    ):
        splits.append({
            "fold": int(fold_i),
            # positions within the training set, mapped back to absolute
            # indices of the master CSV
            "train": [int(train_idx[i]) for i in tr.tolist()],
            "val":   [int(train_idx[i]) for i in va.tolist()],
        })
    log.info("%d-fold CV done (seed=%d)", N_SPLITS, RANDOM_SEED)

    manifest = {
        "metadata": {
            "master_csv": str(DRFP_XTB_EXTENDED_CSV.relative_to(DRFP_XTB_EXTENDED_CSV.parents[1])),
            "n_total": int(n_total),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "created_by": "data_split.py",
            "version": "1.0",
            "description": (
                "Canonical train/test manifest for CO2-cycloaddition ML "
                "pipeline. Both 306_external_validation and 302/301 should "
                "import this file to avoid split drift."
            ),
        },
        "holdout": {
            "test_size": TEST_SIZE,
            "seed": RANDOM_SEED,
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "train_indices": train_idx,
            "test_indices": test_idx,
        },
        "kfold": {
            "n_splits": N_SPLITS,
            "seed": RANDOM_SEED,
            "splits": splits,
        },
    }

    if args.dry_run:
        log.info("[dry-run] splits computed but not written.")
        log.info("  holdout: n_train=%d n_test=%d", len(train_idx), len(test_idx))
        for s in splits:
            log.info("  fold %d: train=%d  val=%d",
                     s["fold"], len(s["train"]), len(s["val"]))
        return 0

    if DATA_SPLIT_JSON.exists() and not args.force:
        log.warning("%s already exists (use --force to overwrite)", DATA_SPLIT_JSON)
        return 0

    ensure_dir(RESULTS_DATA_SPLIT)
    with open(DATA_SPLIT_JSON, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    log.info("Wrote %s", DATA_SPLIT_JSON)

    summary_path = RESULTS_DATA_SPLIT / "data_split_summary.txt"
    lines = [
        "Data Split Manifest",
        "=" * 60,
        f"master_csv : {manifest['metadata']['master_csv']}",
        f"n_total    : {manifest['metadata']['n_total']}",
        f"holdout    : seed={RANDOM_SEED}  test_size={TEST_SIZE}  "
        f"n_train={len(train_idx)}  n_test={len(test_idx)}",
        f"kfold      : n_splits={N_SPLITS}  seed={RANDOM_SEED}",
        f"created_at : {manifest['metadata']['created_at_utc']}",
        "",
        "Per-fold train/val counts (after holdout):",
    ]
    for s in splits:
        lines.append(
            f"  fold {s['fold']}: train={len(s['train']):4d}  val={len(s['val']):4d}"
        )
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    log.info("Wrote %s", summary_path)

    return 0


# ===========================================================================
#   Public helper API (importable by downstream scripts)
#   --------------------------------------------------------------------------
#   All downstream scripts (201, 301, 302, 303, 304, 305, 306, 401, 405, 901,
#   ci_artifacts/*) should use this API so they all share the SAME split
#   protocol: seed=2026, yield-quartile stratified, 15% holdout, 5-fold CV.
# ===========================================================================
def load_manifest(path=None):
    """Load the split manifest JSON. Returns the parsed dict.

    Falls back to ``paths.DATA_SPLIT_JSON`` if ``path`` is None.
    """
    if path is None:
        from src.paths import DATA_SPLIT_JSON
        path = DATA_SPLIT_JSON
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def holdout_arrays(manifest=None):
    """Return ``(train_idx, test_idx, holdout_meta)`` of absolute master-CSV indices.

    ``train_idx`` is the 85% train pool, ``test_idx`` is the 15% holdout.
    Both are returned as plain Python ``list[int]``.
    """
    if manifest is None:
        manifest = load_manifest()
    h = manifest["holdout"]
    return (
        [int(i) for i in h["train_indices"]],
        [int(i) for i in h["test_indices"]],
        h,
    )


def kfold_folds(manifest=None):
    """Return ``[(fold_id, train_idx, val_idx), ...]`` (5 splits) over the 85% train pool.

    Indices are **relative positions within the train pool** (0..n_train-1), suitable
    for indexing arrays obtained from ``load_data(use_holdout_train=True)``.

    To get the corresponding absolute master-CSV indices, map back through
    ``holdout_arrays(manifest)[0]``.
    """
    if manifest is None:
        manifest = load_manifest()
    train_idx_master, _, _ = holdout_arrays(manifest)
    pos_map = {m: int(p) for p, m in enumerate(sorted(train_idx_master))}
    folds = []
    for s in manifest["kfold"]["splits"]:
        tr_rel = [pos_map[int(i)] for i in s["train"] if int(i) in pos_map]
        va_rel = [pos_map[int(i)] for i in s["val"]   if int(i) in pos_map]
        folds.append((int(s["fold"]), tr_rel, va_rel))
    return folds


def split_iterator(manifest=None):
    """Yield ``(fold_id, train_idx, val_idx)`` — convenient for ``for`` loops.

    Example::

        for fold_id, tr, va in split_iterator():
            X_tr, y_tr = X[tr], y[tr]
            ...
    """
    for fold_id, tr, va in kfold_folds(manifest):
        yield fold_id, tr, va


if __name__ == "__main__":
    raise SystemExit(main())