# -*- coding: utf-8 -*-
"""
Project-level configuration. Single source of truth for tuned hyperparameters.

Workflow:
  201_ablation.py (DRFP variant ablation → λ sweep → full benchmark)
      ↓
  results_best_pipeline/drfp_ablation_meta.json    written
  results_best_pipeline/full_benchmark_results.csv written
  results_lambda_ablation/lambda_results.csv       written  ← BEST_LAMBDA_PROP source
  results_best_pipeline/drfp_ablation_meta.json    ← BEST_DRFP_VARIANT source

Hyperparameters below are AUTO-LOADED at import time from those artefacts,
NOT hard-coded. If 201_ablation.py has never been run, sensible fallbacks
are used (printed as warnings).
"""

import os
import warnings

# Project root resolution (matches utils_rxn.PROJECT_ROOT).
PROJECT_ROOT = os.environ.get(
    "CO2_PROJECT_ROOT", r"D:\machine-learning\CO2-cycloaddition"
)
_RESULTS = os.path.join(PROJECT_ROOT, "results_lambda_ablation",
                        "lambda_results.csv")
_DRFP_META = os.path.join(PROJECT_ROOT, "results_best_pipeline",
                          "drfp_ablation_meta.json")


def _load_best_lambda_prop():
    """
    Read BEST_LAMBDA_PROP from results_lambda_ablation/lambda_results.csv.

    Picks the λ whose DualANN R² is maximal (column ann_r2_mean).
    Falls back to 200.0 with a warning if the CSV is missing.
    """
    fallback = 200.0
    if not os.path.isfile(_RESULTS):
        warnings.warn(
            f"[config] {_RESULTS} not found.  Using fallback "
            f"BEST_LAMBDA_PROP={fallback}.  Run 201_ablation.py to refresh.",
            stacklevel=2,
        )
        return fallback

    try:
        import csv
        with open(_RESULTS, "r", encoding="utf-8-sig", newline="") as fh:
            reader = csv.DictReader(fh)
            rows = [
                r for r in reader
                if r.get("lambda") not in (None, "")
                and r.get("ann_r2_mean") not in (None, "")
            ]
        if not rows:
            warnings.warn(
                f"[config] {_RESULTS} has no parseable rows.  "
                f"Using fallback BEST_LAMBDA_PROP={fallback}.",
                stacklevel=2,
            )
            return fallback

        # Pick λ with the largest ann_r2_mean (DualANN is the primary model
        # in 201_ablation.py Stage 2; criterion matches line 530).
        best_row = max(rows, key=lambda r: float(r["ann_r2_mean"]))
        best_lam = float(best_row["lambda"])
        best_r2 = float(best_row["ann_r2_mean"])
        return best_lam, best_r2
    except Exception as e:
        warnings.warn(
            f"[config] failed to read {_RESULTS}: {e}.  "
            f"Using fallback BEST_LAMBDA_PROP={fallback}.",
            stacklevel=2,
        )
        return fallback


# Best λ for property-co-learning (auto-loaded from lambda_results.csv).
# Tuple: (best_lambda, best_ann_r2) — both available to downstream scripts.
_LAMBDA_RESULT = _load_best_lambda_prop()
if isinstance(_LAMBDA_RESULT, tuple):
    BEST_LAMBDA_PROP, BEST_LAMBDA_PROP_R2 = _LAMBDA_RESULT
else:
    BEST_LAMBDA_PROP = _LAMBDA_RESULT
    BEST_LAMBDA_PROP_R2 = None


def _load_best_drfp_variant():
    """
    Read BEST_DRFP_VARIANT from results_best_pipeline/drfp_ablation_meta.json.
    Falls back to 'full' with a warning if the JSON is missing.
    """
    fallback = "full"
    if not os.path.isfile(_DRFP_META):
        warnings.warn(
            f"[config] {_DRFP_META} not found.  Using fallback "
            f"BEST_DRFP_VARIANT='{fallback}'.  Run 201_ablation.py to refresh.",
            stacklevel=2,
        )
        return fallback
    try:
        import json as _json
        with open(_DRFP_META, "r", encoding="utf-8") as fh:
            meta = _json.load(fh)
        variant = meta.get("best_variant", fallback)
        if not isinstance(variant, str):
            raise ValueError(f"best_variant not a string: {variant!r}")
        return variant
    except Exception as e:
        warnings.warn(
            f"[config] failed to read {_DRFP_META}: {e}.  "
            f"Using fallback BEST_DRFP_VARIANT='{fallback}'.",
            stacklevel=2,
        )
        return fallback


# Latent dimension for PCL-AE (used by 05, 09, 10, 13).  This is a structural
# choice, not a tuned hyper-parameter, so it stays hard-coded.
BEST_LATENT_DIM = 128

# DRFP variant (auto-loaded from drfp_ablation_meta.json).
BEST_DRFP_VARIANT = _load_best_drfp_variant()


# ════════════════════════════════════════════════════════════════════════════════
# External I/O paths and rate limits (consumed by 102_smiles.py, 104b_run_xtb*.py)
# ════════════════════════════════════════════════════════════════════════════════

# Path to geckodriver.exe on Windows. Used by 102_smiles.py for Selenium-based
# PubChem fallback after Cactus. If file does not exist at runtime, 102 silently
# falls back to Cactus-only mode.
GECKODRIVER_PATH = r"C:\tools\geckodriver.exe"

# Cactus NCI-CADD name→SMILES resolver: minimum delay between HTTP requests.
# Official Cactus etiquette recommends ≥1 req/s to avoid temporary IP ban.
# Lower this only if you have a private API key / on-premise proxy.
CACTUS_REQUEST_DELAY_S = 1.0
