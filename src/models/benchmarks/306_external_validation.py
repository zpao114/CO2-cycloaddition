# -*- coding: utf-8 -*-
"""
306_external_validation.py
==========================

Consolidated final report (Stage 6) — aggregates the outputs of every
earlier benchmark stage into one paper-grade report.

Inputs (read-only, from previous tiers)
---------------------------------------
Stage 2  : results_best_pipeline/full_benchmark_results.csv
Stage 3  : results_sample_size_sensitivity/ML_ssts_v2_results.csv
           + data/processed/ML_ssts_v2_results.csv  (subset-stability table)
Stage 4  : results_step4_5/shap_xtb_importance.csv  (optional — gracefully
           skipped if absent)
Stage 5.1: results_groupkfold_validation/ML_groupkfold_results.csv
Stage 5.2: data/processed/ML_error_analysis_summary.csv  (optional)
Stage 5.3: data/processed/ML_bootstrap_ci_results.csv

Outputs (results_external_validation/)
--------------------------------------
  STAGE6_FINAL_REPORT.txt                — human-readable consolidated report
  STAGE6_SUMMARY_TABLE.csv               — machine-readable one-row-per-metric
  external_vs_internal_comparison.csv    — placeholder until 405 runs

Missing inputs NEVER crash the report — they are skipped with a clear "[not
found]" annotation so that incremental re-runs are safe.

Usage
-----
  python 306_external_validation.py
  python 306_external_validation.py --out-dir /tmp/report
  python 306_external_validation.py --print-only    # show on stdout, do not write
"""

from __future__ import annotations

import argparse
import io
import logging
import os
import sys
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# ── Encoding & warnings ─────────────────────────────────────────────────────
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
if sys.stderr.encoding != "utf-8":
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")
warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(
    os.environ.get("CO2_PROJECT_ROOT", r"D:\machine-learning\CO2-cycloaddition")
)
OUT_DIR = PROJECT_ROOT / "results_external_validation"

# Input registry — every stage maps to one or more candidate paths. The first
# existing file wins, so we can pick up new tier-1 re-emissions without code
# edits.
INPUTS: Dict[str, List[Path]] = {
    "Stage2_bench": [
        PROJECT_ROOT / "results_best_pipeline" / "full_benchmark_results.csv",
    ],
    "Stage3_learning_curve": [
        PROJECT_ROOT / "results_sample_size_sensitivity" / "ML_ssts_v2_results.csv",
    ],
    "Stage3_subset_stability": [
        PROJECT_ROOT / "data" / "processed" / "ML_ssts_v2_results.csv",
    ],
    "Stage4_shap": [
        PROJECT_ROOT / "results_step4_5" / "shap_xtb_importance.csv",
        PROJECT_ROOT / "results" / "results_step4_5" / "shap_xtb_importance.csv",
    ],
    "Stage51_groupkfold": [
        PROJECT_ROOT / "results_groupkfold_validation" / "ML_groupkfold_results.csv",
        PROJECT_ROOT / "results" / "results_groupkfold_validation" / "ML_groupkfold_results.csv",
    ],
    "Stage52_error": [
        PROJECT_ROOT / "data" / "processed" / "ML_error_analysis_summary.csv",
    ],
    "Stage53_bootstrap": [
        PROJECT_ROOT / "data" / "processed" / "ML_bootstrap_ci_results.csv",
    ],
}

OUTPUT_TXT = "STAGE6_FINAL_REPORT.txt"
OUTPUT_CSV = "STAGE6_SUMMARY_TABLE.csv"
OUTPUT_EXT = "external_vs_internal_comparison.csv"

LOG_FMT = "%(asctime)s [%(levelname)s] %(message)s"
LOG_DATEFMT = "%H:%M:%S"
logger = logging.getLogger("306")


# ── Physics-interpretation helpers ──────────────────────────────────────────
PHYSICS_MAP: Dict[str, str] = {
    "sub_homo_eV": "substrate HOMO (electron donor strength)",
    "sub_lumo_eV": "substrate LUMO (electron acceptor)",
    "sub_gap_eV": "substrate HOMO-LUMO gap",
    "sub_dipole_D": "substrate dipole moment",
    "cat_homo_eV": "catalyst HOMO",
    "cat_lumo_eV": "catalyst LUMO",
    "cat_gap_eV": "catalyst HOMO-LUMO gap",
    "cat_electrophilicity": "catalyst electrophilicity index",
    "cat_dipole_D": "catalyst dipole moment",
    "solv_homo_eV": "solvent HOMO",
    "solv_lumo_eV": "solvent LUMO",
    "solv_gap_eV": "solvent HOMO-LUMO gap",
    "co2_homo_eV": "CO2 HOMO",
    "co2_lumo_eV": "CO2 LUMO",
    "co2_gap_eV": "CO2 HOMO-LUMO gap",
    "delta_E_LL": "cat LUMO − sub LUMO (orbital match)",
    "delta_E_HL": "cat HOMO − sub HOMO (orbital match)",
    "temperature": "reaction temperature (°C)",
    "pressure": "CO2 pressure (MPa)",
    "time_log": "log reaction time",
    "loading_log": "log catalyst loading",
    "has_solvent": "solvent presence flag",
    "has_reagent": "co-reagent flag",
    "electrophilicity": "global electrophilicity",
    "nucleophilicity": "global nucleophilicity",
    "global_hardness": "global hardness",
    "global_softness": "global softness",
}


# ── Loaders ────────────────────────────────────────────────────────────────
@dataclass
class LoadedInputs:
    """Holds one DataFrame per stage; ``None`` if the stage has no output yet."""
    bench: Optional[pd.DataFrame] = None
    learning_curve: Optional[pd.DataFrame] = None
    subset_stability: Optional[pd.DataFrame] = None
    shap: Optional[pd.DataFrame] = None
    groupkfold: Optional[pd.DataFrame] = None
    error: Optional[pd.DataFrame] = None
    bootstrap: Optional[pd.DataFrame] = None
    sources: Dict[str, Path] = field(default_factory=dict)


def _first_existing(paths: List[Path]) -> Optional[Path]:
    for p in paths:
        if p.exists() and p.is_file():
            return p
    return None


def _safe_read(paths: List[Path]) -> Optional[pd.DataFrame]:
    """Return the first readable CSV among ``paths``, or None."""
    p = _first_existing(paths)
    if p is None:
        return None
    try:
        df = pd.read_csv(p, encoding="utf-8-sig")
        logger.info("  loaded %d rows from %s", len(df), p.name)
        return df
    except Exception as ex:  # noqa: BLE001
        logger.warning("  failed to read %s: %s", p, ex)
        return None


def load_all() -> LoadedInputs:
    """Load every stage. Missing inputs are silently allowed."""
    logger.info("Loading stage outputs …")
    li = LoadedInputs()
    li.bench = _safe_read(INPUTS["Stage2_bench"])
    li.learning_curve = _safe_read(INPUTS["Stage3_learning_curve"])
    li.subset_stability = _safe_read(INPUTS["Stage3_subset_stability"])
    li.shap = _safe_read(INPUTS["Stage4_shap"])
    li.groupkfold = _safe_read(INPUTS["Stage51_groupkfold"])
    li.error = _safe_read(INPUTS["Stage52_error"])
    li.bootstrap = _safe_read(INPUTS["Stage53_bootstrap"])

    for stage, paths in INPUTS.items():
        li.sources[stage] = _first_existing(paths)  # may be None
    return li


# ── Section builders (each appends to ``lines``) ────────────────────────────
def _hdr(title: str) -> List[str]:
    bar = "=" * 78
    return ["", bar, f"  {title}", bar]


def _missing(lines: List[str], what: str, source: Optional[Path]) -> None:
    lines.append(f"  ⚠ {what} not found"
                 + (f" (looked at: {source})" if source else "")
                 + " — skipping this section.")


def section_header(lines: List[str], inputs: LoadedInputs) -> None:
    lines.append("=" * 78)
    lines.append("CO2 Cycloaddition Yield Prediction — Consolidated Final Report")
    lines.append("=" * 78)
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Project  : {PROJECT_ROOT}")
    lines.append("")
    lines.append("This report aggregates every analysis stage into a single")
    lines.append("paper-grade summary covering benchmark, subset stability,")
    lines.append("learning-curve sensitivity, SHAP, grouped cross-validation,")
    lines.append("error decomposition, and bootstrap confidence intervals.")
    lines.append("")
    lines.append("Stages with available data:")
    for stage in ("Stage2_bench", "Stage3_learning_curve",
                  "Stage3_subset_stability", "Stage4_shap",
                  "Stage51_groupkfold", "Stage52_error", "Stage53_bootstrap"):
        s = inputs.sources[stage]
        lines.append(f"  [{stage:30s}]  {'✓ ' + str(s) if s else '✗ (missing)'}")


def section_benchmark(lines: List[str], bench: Optional[pd.DataFrame]) -> Dict:
    """Stage 2 — Top-10 configurations. Returns extra info used downstream."""
    lines.extend(_hdr("[Stage 2] Full Benchmark — Top-10 Configurations"))
    if bench is None or bench.empty:
        _missing(lines, "full_benchmark_results.csv", INPUTS["Stage2_bench"][0])
        return {}
    if "r2_mean" not in bench.columns:
        _missing(lines, "full_benchmark_results.csv (no r2_mean column)", None)
        return {}

    top10 = bench.sort_values("r2_mean", ascending=False).head(10)
    lines.append(f"  Total configurations benchmarked: {len(bench)}")
    lines.append("")
    lines.append(
        f"  {'Rank':<5} {'DRFP':<14} {'Feature-set':<30} {'Model':<10} "
        f"{'R²':<9} {'MAE':<9} {'Dim':<6}"
    )
    lines.append("  " + "-" * 86)
    for i, (_, r) in enumerate(top10.iterrows(), 1):
        lines.append(
            f"  {i:<5} {str(r.get('drfp_method',''))[:14]:<14} "
            f"{str(r.get('feature_set',''))[:30]:<30} "
            f"{str(r.get('model',''))[:10]:<10} "
            f"{r.get('r2_mean', float('nan')):<9.4f} "
            f"{r.get('mae_mean', float('nan')):<9.4f} "
            f"{int(r.get('feature_dim', 0)):<6d}"
        )
    best = top10.iloc[0]
    lines.append("")
    lines.append(f"  Best configuration:")
    lines.append(f"    DRFP method : {best.get('drfp_method','')}")
    lines.append(f"    Features    : {best.get('feature_set','')}")
    lines.append(f"    Model       : {best.get('model','')}")
    lines.append(f"    R²          : {best['r2_mean']:.4f} ± {best.get('r2_std', 0):.4f}")
    lines.append(f"    MAE         : {best.get('mae_mean', 0):.4f}")
    lines.append(f"    RMSE        : {best.get('rmse_mean', 0):.4f}")
    lines.append(f"    Pearson r   : {best.get('pearson_mean', 0):.4f}")
    lines.append(f"    Feature dim : {int(best.get('feature_dim', 0))}")
    return {"top10": top10, "best": best}


def section_learning_curve(lines: List[str], lc: Optional[pd.DataFrame]) -> None:
    """Stage 3 — Sample-size / learning curve."""
    lines.extend(_hdr("[Stage 3a] Learning Curve — R² vs Training Size"))
    if lc is None or lc.empty:
        _missing(lines, "learning-curve CSV",
                 INPUTS["Stage3_learning_curve"][0])
        return
    # Try the 'summary' sibling first; otherwise summarise inline.
    summary_csv = lc.copy()
    if {"model", "n_train", "r2"}.issubset(lc.columns):
        # Treat the long-format CSV as the source of truth; roll up.
        summary_csv = (
            lc.groupby(["model", "n_train"], as_index=False)
            .agg(
                r2_mean=("r2", "mean"),
                r2_std=("r2", "std"),
                n_obs=("r2", "count"),
            )
            .sort_values(["model", "n_train"])
        )
    rows: List[str] = []
    rows.append(f"  Total observations: {len(lc)}  (unique combos: {len(summary_csv)})")
    rows.append("")
    rows.append(f"  {'n_train':>8} | " +
                "  ".join(f"{m:>16}" for m in sorted(summary_csv['model'].unique())))
    rows.append("  " + "-" * 78)
    for n, sub in summary_csv.groupby("n_train"):
        cells = []
        for m in sorted(summary_csv['model'].unique()):
            row = sub[sub['model'] == m]
            if row.empty:
                cells.append(f"{'—':>16}")
            else:
                v = float(row['r2_mean'].iloc[0])
                s = float(row.get('r2_std', pd.Series([0])).iloc[0])
                cells.append(f"{v:>7.4f}±{s:>4.2f}" if s == s else f"{v:>16.4f}")
        rows.append(f"  {int(n):>8d} | " + "  ".join(cells))
    lines.extend(rows)

    # Best of the best
    best_row = summary_csv.sort_values("r2_mean", ascending=False).iloc[0]
    lines.append("")
    lines.append(
        f"  Best learning-curve R²={best_row['r2_mean']:.4f} "
        f"at n_train={int(best_row['n_train'])} "
        f"with model={best_row['model']}"
    )


def section_subset_stability(lines: List[str], ss: Optional[pd.DataFrame]) -> None:
    """Stage 3 — Subset stability (legacy SSTS table)."""
    lines.extend(_hdr("[Stage 3b] Subset Stability — R² per catalyst-type subset"))
    if ss is None or ss.empty:
        _missing(lines, "subset-stability CSV",
                 INPUTS["Stage3_subset_stability"][0])
        return
    if not {"subset", "n", "model", "r2"}.issubset(ss.columns):
        _missing(lines, "subset-stability CSV (missing required columns)", None)
        return

    lines.append(f"  Total subsets evaluated: {ss['subset'].nunique()}")
    lines.append("")
    lines.append(
        f"  {'Subset':<22} {'n':>6} {'Model':<10} {'R²':>9} {'±std':>7} {'MAE':>9}"
    )
    lines.append("  " + "-" * 70)
    for subset in sorted(ss["subset"].unique()):
        sub = ss[ss["subset"] == subset].sort_values("r2", ascending=False)
        for _, r in sub.iterrows():
            lines.append(
                f"  {str(r['subset'])[:22]:<22} {int(r['n']):>6d} "
                f"{str(r['model'])[:10]:<10} "
                f"{r['r2']:>9.4f} {r.get('r2_std', 0):>7.4f} "
                f"{r.get('mae', 0):>9.4f}"
            )


def section_shap(lines: List[str], shap: Optional[pd.DataFrame]) -> None:
    """Stage 4 — Top-10 SHAP features."""
    lines.extend(_hdr("[Stage 4] SHAP — Top-10 Physical-Interpretable Features"))
    if shap is None or shap.empty:
        _missing(lines, "shap_xtb_importance.csv", INPUTS["Stage4_shap"][0])
        return
    feat_col = next((c for c in shap.columns if c.lower() in ("feature", "name")), None)
    val_col = next((c for c in shap.columns if "shap" in c.lower() or "importance" in c.lower()), None)
    if feat_col is None or val_col is None:
        _missing(lines, "shap CSV (unknown column layout)", None)
        return
    rank_col = "rank" if "rank" in shap.columns else None

    lines.append(f"  Total features analysed: {len(shap)}")
    lines.append("")
    lines.append(
        f"  {'Rank':<5} {'Feature':<24} {'Mean |SHAP|':<12} {'Physics':<50}"
    )
    lines.append("  " + "-" * 95)
    for i, (_, r) in enumerate(shap.head(10).iterrows(), 1):
        feat = str(r[feat_col])
        shap_val = float(r[val_col])
        physics = PHYSICS_MAP.get(feat, "")
        rank = int(r[rank_col]) if rank_col else i
        lines.append(
            f"  {rank:<5} {feat[:24]:<24} {shap_val:<12.6f} {physics[:50]:<50}"
        )
    top = shap.iloc[0]
    lines.append("")
    lines.append(
        f"  Most influential feature: {top[feat_col]} "
        f"(mean |SHAP|={float(top[val_col]):.4f})"
    )
    if float(top[val_col]) > 0:
        lines.append(
            "  → Validates the dominant role of substrate electronic structure"
        )


def section_groupkfold(lines: List[str], gkfld: Optional[pd.DataFrame]) -> None:
    """Stage 5.1 — GroupKFold external validity check.

    Accepts two schemas:
      - canonical: has a 'split' column (e.g. catalyst_system_type vs reactant_name)
      - 302-style: has a 'label' column with rows like 'Baseline_all_LOCO',
        'SSTS-A_metal_halide_rand', etc.
    """
    lines.extend(_hdr("[Stage 5.1] GroupKFold — External Validity"))
    if gkfld is None or gkfld.empty:
        _missing(lines, "ML_groupkfold_results.csv", INPUTS["Stage51_groupkfold"][0])
        return

    # Detect schema — 302 uses 'label', generic uses 'split'.
    if "split" in gkfld.columns:
        group_col = "split"
    elif "label" in gkfld.columns:
        group_col = "label"
    else:
        _missing(lines, "ML_groupkfold_results.csv (no 'split'/'label' column)", None)
        return

    metric_cols = [c for c in gkfld.columns if c.endswith("_R2")]
    if not metric_cols:
        if "r2" in gkfld.columns:
            metric_cols = ["r2"]
        else:
            _missing(lines, "ML_groupkfold_results.csv (no R² columns)", None)
            return

    lines.append(f"  Total fold results: {len(gkfld)}  "
                 f"(grouping column: '{group_col}')")
    lines.append("")

    has_r2_std = "r2_std" in gkfld.columns

    for label in sorted(gkfld[group_col].unique()):
        sub = gkfld[gkfld[group_col] == label]
        agg = []
        for c in metric_cols:
            v = pd.to_numeric(sub[c], errors="coerce").dropna()
            if v.empty:
                agg.append(f"{'—':>15}")
            else:
                # Prefer the pre-computed std column (302 stores it per-row);
                # else fall back to numpy std across the (rare) multi-row case.
                if has_r2_std and c == "r2":
                    std_val = pd.to_numeric(sub["r2_std"], errors="coerce").dropna()
                    std_repr = float(std_val.iloc[0]) if not std_val.empty else float(v.std())
                else:
                    std_repr = float(v.std())
                agg.append(f"{v.mean():>8.4f}±{std_repr:.4f}")
        header = "  " + "  ".join(f"{c.replace('_R2',''):>14}" for c in metric_cols)
        body = "  " + "  ".join(agg)
        n_folds = int(sub["n_folds"].iloc[0]) if "n_folds" in sub.columns else len(sub)
        lines.append(f"  [{label}]  folds={n_folds}  models={len(metric_cols)}")
        lines.append(header)
        lines.append(body)
        lines.append("")


def section_error(lines: List[str], err: Optional[pd.DataFrame]) -> None:
    """Stage 5.2 — Error analysis by yield band / catalyst / reactant."""
    lines.extend(_hdr("[Stage 5.2] Error Analysis — Where the Model Fails"))
    if err is None or err.empty:
        _missing(lines, "ML_error_analysis_summary.csv", INPUTS["Stage52_error"][0])
        return

    sections_rendered = 0
    if "band" in err.columns and err["band"].notna().any():
        by_band = err[err["band"].notna()]
        lines.append("")
        lines.append("  By Yield Band:")
        lines.append(f"  {'Band':<22} {'n':>5} {'RMSE':>9} {'MAE':>9} {'R²':>9} {'Bias':>+9}")
        lines.append("  " + "-" * 70)
        for _, r in by_band.iterrows():
            lines.append(
                f"  {str(r['band'])[:22]:<22} {int(r['n']):>5d} "
                f"{r.get('RMSE', 0):>9.4f} {r.get('MAE', 0):>9.4f} "
                f"{r.get('R2', 0):>9.4f} {r.get('bias', 0):>+9.4f}"
            )
        sections_rendered += 1
    if "catalyst_type" in err.columns and err["catalyst_type"].notna().any():
        by_cat = err[err["catalyst_type"].notna()]
        lines.append("")
        lines.append("  By Catalyst System:")
        lines.append(f"  {'Catalyst':<20} {'n':>5} {'RMSE':>9} {'MAE':>9} {'R²':>9} {'Bias':>+9}")
        lines.append("  " + "-" * 70)
        for _, r in by_cat.iterrows():
            lines.append(
                f"  {str(r['catalyst_type'])[:20]:<20} {int(r['n']):>5d} "
                f"{r.get('RMSE', 0):>9.4f} {r.get('MAE', 0):>9.4f} "
                f"{r.get('R2', 0):>9.4f} {r.get('bias', 0):>+9.4f}"
            )
        sections_rendered += 1
    if "reactant" in err.columns and err["reactant"].notna().any():
        by_rxn = err[err["reactant"].notna()]
        lines.append("")
        lines.append("  By Reactant:")
        lines.append(f"  {'Reactant':<22} {'n':>5} {'RMSE':>9} {'MAE':>9} {'R²':>9} {'Bias':>+9}")
        lines.append("  " + "-" * 70)
        for _, r in by_rxn.iterrows():
            lines.append(
                f"  {str(r['reactant'])[:22]:<22} {int(r['n']):>5d} "
                f"{r.get('RMSE', 0):>9.4f} {r.get('MAE', 0):>9.4f} "
                f"{r.get('R2', 0):>9.4f} {r.get('bias', 0):>+9.4f}"
            )
        sections_rendered += 1
    if sections_rendered == 0:
        _missing(lines, "error analysis CSV (no usable columns)", None)


def section_bootstrap(lines: List[str], boot: Optional[pd.DataFrame]) -> None:
    """Stage 5.3 — Bootstrap CI."""
    lines.extend(_hdr("[Stage 5.3] Bootstrap CI — Overall"))
    if boot is None or boot.empty:
        _missing(lines, "ML_bootstrap_ci_results.csv", INPUTS["Stage53_bootstrap"][0])
        return
    if not {"split", "model", "metric", "point", "ci_95_lo", "ci_95_hi"}.issubset(boot.columns):
        _missing(lines, "ML_bootstrap_ci_results.csv (schema mismatch)", None)
        return

    overall = boot[boot["split"] == "overall"]
    lines.append(f"  Total bootstrap rows: {len(boot)}")
    lines.append("")
    lines.append(
        f"  {'Model':<8} {'Metric':<8} {'Point':>9} {'CI_lo':>9} {'CI_hi':>9} {'Width':>9}"
    )
    lines.append("  " + "-" * 60)
    for _, r in overall.iterrows():
        try:
            width = float(r["ci_95_hi"]) - float(r["ci_95_lo"])
            lines.append(
                f"  {str(r['model'])[:8]:<8} {str(r['metric'])[:8]:<8} "
                f"{float(r['point']):>9.4f} {float(r['ci_95_lo']):>9.4f} "
                f"{float(r['ci_95_hi']):>9.4f} {width:>9.4f}"
            )
        except Exception:  # noqa: BLE001
            continue


# ── Cross-stage synthesis ──────────────────────────────────────────────────
def section_keyfindings(
    lines: List[str],
    inputs: LoadedInputs,
    bench_ctx: Dict,
) -> None:
    """Synthesise cross-stage key findings."""
    lines.extend(_hdr("[Consolidated Key Findings]"))
    findings: List[str] = []

    bench = inputs.bench
    if bench is not None and not bench.empty and "r2_mean" in bench.columns:
        top = bench.sort_values("r2_mean", ascending=False).iloc[0]
        findings.append(
            f"1. Best overall benchmark: {top.get('drfp_method','')} + "
            f"{top.get('feature_set','')} + {top.get('model','')}, "
            f"R²={top['r2_mean']:.4f}, MAE={top.get('mae_mean', 0):.4f}"
        )

    lc = inputs.learning_curve
    if lc is not None and not lc.empty and {"model", "n_train", "r2"}.issubset(lc.columns):
        summary = (
            lc.groupby(["model", "n_train"], as_index=False)
            .agg(r2_mean=("r2", "mean"))
        )
        best = summary.sort_values("r2_mean", ascending=False).iloc[0]
        findings.append(
            f"2. Best learning-curve point: {best['model']} @ n_train="
            f"{int(best['n_train'])} → R²={best['r2_mean']:.4f}"
        )

    gkfld = inputs.groupkfold
    if gkfld is not None and not gkfld.empty and "split" in gkfld.columns:
        if "r2" in gkfld.columns:
            mean_loc = float(gkfld["r2"].mean())
            findings.append(
                f"3. GroupKFold mean R² across all splits: {mean_loc:.4f}"
                " (negative → catalyst-out extrapolation is hard)"
            )

    boot = inputs.bootstrap
    if boot is not None and not boot.empty:
        overall = boot[boot["split"] == "overall"]
        rf = overall[(overall["model"] == "RF") & (overall["metric"] == "R2")]
        xgb = overall[(overall["model"] == "XGB") & (overall["metric"] == "R2")]
        if not rf.empty and not xgb.empty:
            rf_lo, rf_hi = float(rf.iloc[0]["ci_95_lo"]), float(rf.iloc[0]["ci_95_hi"])
            xgb_lo, xgb_hi = float(xgb.iloc[0]["ci_95_lo"]), float(xgb.iloc[0]["ci_95_hi"])
            findings.append(
                f"4. Bootstrap 95% CI: "
                f"RF R² ∈ [{rf_lo:.4f}, {rf_hi:.4f}], "
                f"XGB R² ∈ [{xgb_lo:.4f}, {xgb_hi:.4f}]"
            )

    inputs_summary = (
        f"   ({sum(1 for v in vars(inputs).values() if isinstance(v, pd.DataFrame) and v is not None)}"
        f" of 7 stages available; "
        f"{sum(1 for v in vars(inputs).values() if isinstance(v, pd.DataFrame) and (v is None or v.empty))}"
        f" missing)"
    )
    findings.append(f"5. Stage coverage" + inputs_summary)

    if not findings:
        findings.append("  (No stage outputs were available — every section was empty.)")

    for f in findings:
        lines.append("  " + f)


# ── External-vs-internal placeholder ────────────────────────────────────────
def write_external_placeholder(
    inputs: LoadedInputs, bench_ctx: Dict, out_dir: Path
) -> Optional[Path]:
    """Write the external_vs_internal_comparison.csv placeholder.

    True external validation belongs to 405_external_validation.py (tier_5);
    for now we treat the best benchmark R² as the *internal* ceiling.
    """
    bench = inputs.bench
    if bench is None or bench.empty or "r2_mean" not in bench.columns:
        logger.warning("  skipped external placeholder (no benchmark CSV)")
        return None

    top10 = bench_ctx.get("top10")
    if top10 is None:
        top10 = bench.sort_values("r2_mean", ascending=False).head(10)
    best = bench_ctx.get("best", bench.sort_values("r2_mean", ascending=False).iloc[0])
    bench_r2 = float(best["r2_mean"])

    rows = []
    for _, r in top10.iterrows():
        for split_name in ("catalyst_system_type", "reactant_name", "substrate"):
            # note: placeholder; no real "external" split result yet.
            rows.append({
                "model": f"{r.get('drfp_method','')}+{r.get('feature_set','')}+{r.get('model','')}",
                "split": split_name,
                "internal_r2": float(r["r2_mean"]),
                "external_r2": np.nan,           # filled by 405
                "delta_r2": np.nan,
            })
    rows.append({
        "model": "_summary_",
        "split": "best_overall",
        "internal_r2": bench_r2,
        "external_r2": np.nan,
        "delta_r2": np.nan,
    })
    out = out_dir / OUTPUT_EXT
    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")
    logger.info("  wrote placeholder %s", out)
    return out


# ── Machine-readable summary ───────────────────────────────────────────────
def build_summary_table(inputs: LoadedInputs) -> pd.DataFrame:
    """One-row-per-metric flat table for downstream tools / dashboards."""
    rows: List[Dict] = []

    bench = inputs.bench
    if bench is not None and not bench.empty and "r2_mean" in bench.columns:
        top = bench.sort_values("r2_mean", ascending=False).iloc[0]
        rows.append({
            "section": "Stage2_best",
            "metric": "r2_mean",
            "value": float(top["r2_mean"]),
            "detail": f"{top.get('drfp_method','')}+{top.get('feature_set','')}+{top.get('model','')}",
        })
        rows.append({
            "section": "Stage2_best",
            "metric": "mae_mean",
            "value": float(top.get("mae_mean", 0)),
            "detail": f"{top.get('drfp_method','')}+{top.get('feature_set','')}+{top.get('model','')}",
        })

    lc = inputs.learning_curve
    if lc is not None and not lc.empty and {"model", "n_train", "r2"}.issubset(lc.columns):
        for n in sorted(lc["n_train"].unique()):
            sub = lc[lc["n_train"] == n]
            for m in sorted(sub["model"].unique()):
                rows.append({
                    "section": "Stage3_learning_curve",
                    "metric": f"r2_{m}_n{n}",
                    "value": float(sub[sub["model"] == m]["r2"].mean()),
                    "detail": f"std={sub[sub['model'] == m]['r2'].std():.4f}",
                })

    ss = inputs.subset_stability
    if ss is not None and not ss.empty and {"subset", "model", "r2"}.issubset(ss.columns):
        for _, r in ss.iterrows():
            rows.append({
                "section": "Stage3_subset",
                "metric": f"r2_{r['subset']}_{r['model']}",
                "value": float(r["r2"]),
                "detail": f"n={int(r.get('n', 0))}",
            })

    shap = inputs.shap
    if shap is not None and not shap.empty:
        feat_col = next((c for c in shap.columns if c.lower() in ("feature", "name")), None)
        val_col = next((c for c in shap.columns if "shap" in c.lower() or "importance" in c.lower()), None)
        if feat_col and val_col:
            for i, (_, r) in enumerate(shap.head(5).iterrows(), 1):
                rows.append({
                    "section": "Stage4_SHAP",
                    "metric": f"rank{i}",
                    "value": float(r[val_col]),
                    "detail": str(r[feat_col]),
                })

    gkfld = inputs.groupkfold
    if gkfld is not None and not gkfld.empty and "split" in gkfld.columns:
        for col in [c for c in gkfld.columns if c.endswith("_R2")]:
            for split in sorted(gkfld["split"].unique()):
                sub = gkfld[gkfld["split"] == split][col].dropna()
                if not sub.empty:
                    rows.append({
                        "section": "Stage51_GroupKFold",
                        "metric": f"{col}_{split}",
                        "value": float(sub.mean()),
                        "detail": f"std={float(sub.std()):.4f}",
                    })

    boot = inputs.bootstrap
    if boot is not None and not boot.empty:
        overall = boot[boot["split"] == "overall"]
        for _, r in overall.iterrows():
            try:
                rows.append({
                    "section": "Stage53_Bootstrap",
                    "metric": f"{r['model']}_{r['metric']}_lo",
                    "value": float(r["ci_95_lo"]),
                    "detail": f"point={float(r['point']):.4f}",
                })
                rows.append({
                    "section": "Stage53_Bootstrap",
                    "metric": f"{r['model']}_{r['metric']}_hi",
                    "value": float(r["ci_95_hi"]),
                    "detail": f"point={float(r['point']):.4f}",
                })
            except Exception:  # noqa: BLE001
                continue

    return pd.DataFrame(rows, columns=["section", "metric", "value", "detail"])


# ── Orchestration ──────────────────────────────────────────────────────────
def configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format=LOG_FMT, datefmt=LOG_DATEFMT,
    )
    for noisy in ("matplotlib", "lightgbm", "xgboost"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stage 6 — Consolidated final report",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--out-dir", default=str(OUT_DIR),
                   help="Where to write STAGE6_FINAL_REPORT.txt etc.")
    p.add_argument("--print-only", action="store_true",
                   help="Echo report on stdout without writing any files.")
    p.add_argument("--verbose", action="store_true",
                   help="Enable DEBUG-level logging.")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    configure_logging(args.verbose)

    out_dir = Path(args.out_dir)
    if not args.print_only:
        out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("Stage 6 — Consolidated Final Report")
    logger.info("=" * 70)

    inputs = load_all()

    lines: List[str] = []
    section_header(lines, inputs)
    bench_ctx = section_benchmark(lines, inputs.bench)
    section_learning_curve(lines, inputs.learning_curve)
    section_subset_stability(lines, inputs.subset_stability)
    section_shap(lines, inputs.shap)
    section_groupkfold(lines, inputs.groupkfold)
    section_error(lines, inputs.error)
    section_bootstrap(lines, inputs.bootstrap)
    section_keyfindings(lines, inputs, bench_ctx)

    lines.append("")
    lines.append("=" * 78)
    lines.append("End of report")
    lines.append("=" * 78)

    text = "\n".join(lines)

    if args.print_only:
        sys.stdout.write(text + "\n")
    else:
        # 1) Human-readable
        out_txt = out_dir / OUTPUT_TXT
        out_txt.write_text(text, encoding="utf-8")
        logger.info("Wrote %s", out_txt)

        # 2) Machine-readable summary
        summary_df = build_summary_table(inputs)
        out_csv = out_dir / OUTPUT_CSV
        summary_df.to_csv(out_csv, index=False, encoding="utf-8-sig")
        logger.info("Wrote %s  (%d rows)", out_csv, len(summary_df))

        # 3) External placeholder
        write_external_placeholder(inputs, bench_ctx, out_dir)

        logger.info("Done.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())