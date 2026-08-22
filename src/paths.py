# -*- coding: utf-8 -*-
"""
src/paths.py
============
Centralised path registry for the CO2-cycloaddition ML pipeline.

All scripts in this repository derive their input/output paths from the
constants defined here. Two usage patterns are supported:

1. **Default (recommended)** — just `from src import paths` (or `import paths`
   with `src/` on `sys.path`) and call `paths.DATA_PROCESSED / "cleaned.csv"`.
   Path constants resolve to the layout defined in `README.md`.

2. **Override via environment variable** — set `CO2_PROJECT_ROOT` to
   point at a different copy of the project (e.g. CI builds, Docker
   mounts, snapshots). All absolute paths are rebased automatically.

Examples
--------
>>> from src.paths import DATA_PROCESSED, RESULTS_DIR
>>> csv_path = DATA_PROCESSED / "cleaned.csv"
>>> plot_path = PROJECT_ROOT / "results_step4" / "summary_protocol.csv"

>>> import os
>>> os.environ["CO2_PROJECT_ROOT"] = r"D:\\snapshots\\co2_2026-08-17"
>>> # subsequent imports will pick up the new root
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Project root resolution
# ---------------------------------------------------------------------------
# Priority:
#   1. $CO2_PROJECT_ROOT env var (must be the absolute path to the repo root)
#   2. The parent of the directory containing this file (i.e. one level above
#      `src/paths.py`), which is the canonical repo root.
#   3. Fallback to the documented hard-coded path on Windows.
DEFAULT_PROJECT_ROOT = r"D:\machine-learning\CO2-cycloaddition"


def _resolve_project_root() -> Path:
    """Resolve the project root, honouring CO2_PROJECT_ROOT."""
    env_root = os.environ.get("CO2_PROJECT_ROOT")
    if env_root:
        return Path(env_root).resolve()

    here = Path(__file__).resolve()
    # this file is <repo>/src/paths.py -> parent.parent is <repo>
    inferred = here.parent.parent
    if (inferred / "src" / "paths.py").exists() and (inferred / "README.md").exists():
        return inferred

    return Path(DEFAULT_PROJECT_ROOT).resolve()


PROJECT_ROOT: Path = _resolve_project_root()


# ---------------------------------------------------------------------------
# Directory layout (matches the structure in README.md)
# ---------------------------------------------------------------------------
SRC_DIR          = PROJECT_ROOT / "src"
DATA_DIR         = PROJECT_ROOT / "data"
RESULTS_DIR      = PROJECT_ROOT / "results"
DOCS_DIR         = PROJECT_ROOT / "docs"
ASSETS_DIR       = PROJECT_ROOT / "assets"
LOGS_DIR         = PROJECT_ROOT / "logs"
SCRIPTS_DIR      = PROJECT_ROOT / "scripts"
TOOLS_DIR        = PROJECT_ROOT / "tools"
CONFIGS_DIR      = PROJECT_ROOT / "configs"

# Data subdirectories
DATA_RAW         = DATA_DIR / "raw"
DATA_PROCESSED   = DATA_DIR / "processed"
DATA_EXTERNAL    = DATA_DIR / "external"

# Documentation subdirectories
DOCS_PAPER            = DOCS_DIR / "paper"
DOCS_SUPPLEMENTARY    = DOCS_DIR / "supplementary"
DOCS_PIPELINE_NOTES   = DOCS_DIR / "pipeline_notes"

# Assets subdirectories
ASSETS_MOLECULAR_STRUCTURES = ASSETS_DIR / "molecular_structures"


# ---------------------------------------------------------------------------
# Results subdirectories (legacy path names preserved for compatibility)
# ---------------------------------------------------------------------------
# 2026-08-20 fix (final): paths.py splits into two groups:
#   (a) RESULTS_CHO_DIAGNOSTIC / RESULTS_DATA_SPLIT / RESULTS_SI /
#       RESULTS_Y_RANDOMIZATION_V4_100PERM -> under results/ because
#       scripts 104b/105b/107/901/902/si/yrand write there.
#   (b) RESULTS_STEP4 / RESULTS_STEP4_5 / RESULTS_STEP7_IMPROVED_LOSO /
#       RESULTS_SUBSTRATE_CATALYST_MATRIX / RESULTS_MECHANISM /
#       RESULTS_TRANSFERABILITY -> at PROJECT_ROOT because scripts 700/701/
#       705/901/602/603 write there.  See .bak_20260820_fix for history.
RESULTS_DATA_SPLIT              = RESULTS_DIR / "results_data_split"
RESULTS_BEST_PIPELINE           = PROJECT_ROOT / "results_best_pipeline"
RESULTS_LAMBDA_ABLATION         = PROJECT_ROOT / "results_lambda_ablation"
RESULTS_EXTERNAL_VALIDATION     = PROJECT_ROOT / "results_external_validation"
RESULTS_GROUP_KFOLD             = PROJECT_ROOT / "results_groupkfold_validation"
RESULTS_PCL_AE                  = PROJECT_ROOT / "results_pcl_ae"
RESULTS_PCL_AE_VIZ              = PROJECT_ROOT / "results_pcl_ae_viz"
# Improved PCL-AE results (VAE-style, Huber Loss) - new in 2026-08-21
IMPROVED_PCL_AE_LATENT_NPY      = RESULTS_PCL_AE / "improved_pcl_ae_latent.npy"
RESULTS_STEP4                   = PROJECT_ROOT / "results_step4"
RESULTS_STEP4_5                 = PROJECT_ROOT / "results_step4_5"
RESULTS_STEP5                   = PROJECT_ROOT / "results_step5"
RESULTS_STEP7_IMPROVED_LOSO     = PROJECT_ROOT / "results_step7_improved_loso"
RESULTS_SI                      = RESULTS_DIR / "results_si"
RESULTS_Y_RANDOMIZATION        = PROJECT_ROOT / "results_y_randomization"
RESULTS_Y_RANDOMIZATION_V3      = PROJECT_ROOT / "results_y_randomization_v3"
RESULTS_Y_RANDOMIZATION_V4_100PERM = RESULTS_DIR / "results_y_randomization_v4_100perm"
RESULTS_SUBSTRATE_CATALYST_MATRIX = PROJECT_ROOT / "results_substrate_catalyst_matrix"
RESULTS_CHO_DIAGNOSTIC          = RESULTS_DIR / "results_cho_diagnostic"
RESULTS_DRFP_ABLATION_DEEP      = PROJECT_ROOT / "results_drfp_ablation_deep"
RESULTS_MORDRED_ABLATION        = PROJECT_ROOT / "results_mordred_ablation"
RESULTS_HIERARCHICAL_MODEL      = PROJECT_ROOT / "results_hierarchical_model"
RESULTS_RANKING_METRICS         = PROJECT_ROOT / "results_ranking_metrics"
RESULTS_VIRTUAL_SCREENING       = PROJECT_ROOT / "results_virtual_screening"
RESULTS_SAMPLE_SIZE_SENSITIVITY = PROJECT_ROOT / "results_sample_size_sensitivity"
RESULTS_STATISTICAL_TEST        = PROJECT_ROOT / "results_statistical_test"
RESULTS_UNIFIED_V2              = PROJECT_ROOT / "results_unified_v2"
RESULTS_V2_EFFICIENT            = PROJECT_ROOT / "results_v2_efficient"
RESULTS_DFT                     = PROJECT_ROOT / "results_dft"
RESULTS_HYPOTHETICAL_SCREENING  = PROJECT_ROOT / "results_hypothetical_screening"
RESULTS_GREEN_METRICS           = PROJECT_ROOT / "results_green_metrics"
RESULTS_TRANSFORMER_COMPARISON  = PROJECT_ROOT / "results_transformer_comparison"
RESULTS_TANIMOTO_SENSITIVITY    = PROJECT_ROOT / "results_tanimoto_sensitivity"
RESULTS_MECHANISM              = PROJECT_ROOT / "results_mechanism"
RESULTS_TRANSFERABILITY        = PROJECT_ROOT / "results_transferability"
DFT_VALIDATION                 = PROJECT_ROOT / "dft_validation"
RESULTS_SHAP_COMPREHENSIVE     = PROJECT_ROOT / "results_shap_comprehensive"
RESULTS_PUBLICATION_BIAS      = PROJECT_ROOT / "results_publication_bias_sensitivity"

# ── WSL/ORCA path ─────────────────────────────────────────────────────────────
WSL_ORCA_ROOT = R"\\wsl.localhost\Ubuntu\home\zzj\orca"


# ---------------------------------------------------------------------------
# Frequently used file paths
# ---------------------------------------------------------------------------
CLEANED_CSV              = DATA_PROCESSED / "cleaned.csv"
SMILES_CSV               = DATA_PROCESSED / "co2_smiles.csv"
DRFP_CSV                 = DATA_PROCESSED / "co2_drfp.csv"
DRFP_XTB_CSV             = DATA_PROCESSED / "co2_drfp_xtb.csv"
DRFP_XTB_EXTENDED_CSV    = RESULTS_CHO_DIAGNOSTIC / "co2_drfp_xtb_extended.csv"
SUBSTRATE_XTB_BASELINE   = DATA_EXTERNAL  / "substrate_xtb_baseline.json"
SMILES_BASELINE          = DATA_EXTERNAL  / "smiles_baseline.json"
CLEANED_BASELINE         = DATA_EXTERNAL  / "cleaned_baseline.json"
EXTRACTION_REPORT        = DATA_EXTERNAL  / "extraction_report.csv"
DISCARD_REPORT           = DATA_EXTERNAL  / "discard_report.csv"
RAW_REAXYS_CSV           = DATA_RAW       / "CO2_cycloaddition_merged.csv"

DATA_SPLIT_JSON          = RESULTS_DATA_SPLIT / "data_split.json"


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------
def ensure_dir(path) -> Path:
    """Create *path* (and intermediates) if it does not exist; return *path*.

    Accepts either a ``pathlib.Path`` or a string. Strings are coerced
    via ``Path(path)`` so this is safe to call with ``os.path.join``
    results or argparse defaults.
    """
    p = path if isinstance(path, Path) else Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def ensure_results_subdir(name: str) -> Path:
    """Ensure `results/<name>/` exists; return it as a Path."""
    p = RESULTS_DIR / name
    return ensure_dir(p)


def report_layout() -> dict:
    """Return a dict of project layout for diagnostics / sanity checks."""
    return {
        "PROJECT_ROOT": str(PROJECT_ROOT),
        "SRC_DIR": str(SRC_DIR),
        "DATA_DIR": str(DATA_DIR),
        "DATA_RAW": str(DATA_RAW),
        "DATA_PROCESSED": str(DATA_PROCESSED),
        "DATA_EXTERNAL": str(DATA_EXTERNAL),
        "RESULTS_DIR": str(RESULTS_DIR),
        "DOCS_DIR": str(DOCS_DIR),
        "ASSETS_DIR": str(ASSETS_DIR),
        "CLEANED_CSV": str(CLEANED_CSV),
        "DRFP_XTB_EXTENDED_CSV": str(DRFP_XTB_EXTENDED_CSV),
        "DATA_SPLIT_JSON": str(DATA_SPLIT_JSON),
    }


if __name__ == "__main__":
    # When invoked as a script, print the resolved layout.
    import json
    print(json.dumps(report_layout(), indent=2))
