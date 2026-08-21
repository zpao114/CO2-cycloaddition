# -*- coding: utf-8 -*-
"""
808_catalyst_control.py
=======================
#3 — Catalyst-type control for the CHO sub_homo_eV sign reversal.

Scientific question
-------------------
Does the CHO sub_homo_eV sign reversal persist when we restrict analysis
to reactions catalysed by the SAME catalyst type (ionic_liquid, metal_halide,
organic_base, etc.)?  This controls for the confound that CHO may happen to
pair with a specific catalyst family that is itself negatively correlated
with sub_homo_eV.

Protocol
--------
For each catalyst_system_type C ∈ {ionic_liquid, metal_halide, organic_base, mixed}:
  1. Filter data to C only.
  2. Split into CHO-subset vs terminal-subset (LOCO on substrate within C).
  3. Train XGBoost; compute SHAP for sub_homo_eV within each group.
  4. Record: C, substrate, n_samples, shap_mean, shap_std.

Outputs
-------
results_shap_comprehensive/catalyst_control/
  catalyst_control_summary.csv    — shap_mean per (catalyst_type, substrate)
  fig_catalyst_control.png       — grouped bar chart

Tier placement: tier_si  (needs tier_main to complete first)
Typical runtime: ~5 min
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

_SCRIPT_DIR = Path(__file__).resolve().parents[1]
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
from paths import RESULTS_DIR

sys.path.insert(0, str(_SCRIPT_DIR / "benchmarks"))
from _shap_infra import (
    SUBSTRATE_ORDER, CHO_NAME, KEY_FEATURES,
    load_X_y_groups, compute_group_shap,
    logger,
)

OUT_DIR = RESULTS_DIR / "results_shap_comprehensive" / "catalyst_control"
os.makedirs(OUT_DIR, exist_ok=True)

CATALYST_TYPES = ["ionic_liquid", "metal_halide", "organic_base", "mixed_system"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    X, y, groups, feat_names, df = load_X_y_groups()
    feat_idx_sub_homo = feat_names.index("sub_homo_eV") if "sub_homo_eV" in feat_names else None

    results = []

    for cat_type in CATALYST_TYPES:
        logger.info("[Catalyst: %s]", cat_type)

        # Filter to this catalyst type
        cat_mask = (df["catalyst_system_type"] == cat_type).values
        if cat_mask.sum() < 20:
            logger.warning("  Skipping %s (only %d samples)", cat_type, cat_mask.sum())
            continue

        X_c = X[cat_mask]
        y_c = y[cat_mask]
        groups_c = np.array(groups)[cat_mask]

        # Count CHO vs terminal within this catalyst type
        cho_n = sum(1 for g in groups_c if g == CHO_NAME)
        term_n = len(groups_c) - cho_n
        logger.info("  CHO=%d  terminal=%d", cho_n, term_n)

        if cho_n < 5 or term_n < 5:
            logger.warning("  Skipping %s (one group too small: CHO=%d, term=%d)",
                          cat_type, cho_n, term_n)
            continue

        for sub_name in SUBSTRATE_ORDER:
            sub_mask = np.array([g == sub_name for g in groups_c])
            if sub_mask.sum() < 3:
                continue

            logger.info("  [substrate: %s, n=%d]", sub_name, sub_mask.sum())

            # Train on complement, evaluate on this substrate
            try:
                res = compute_group_shap(
                    X_c, y_c, sub_mask, feat_names,
                    random_state=42,
                )
            except Exception as e:
                logger.error("  SHAP failed for %s/%s: %s", cat_type, sub_name, e)
                continue

            shap_sub_homo = float(res["shap_mean"][feat_idx_sub_homo]) if feat_idx_sub_homo is not None else np.nan

            results.append(dict(
                catalyst_type=cat_type,
                substrate=sub_name,
                n_samples=int(sub_mask.sum()),
                shap_mean=float(res["shap_mean"].mean()),
                shap_mean_sub_homo=shap_sub_homo,
                r2=float(res["r2"]),
            ))

    # ── Save ────────────────────────────────────────────────────────────────
    results_df = pd.DataFrame(results)
    out_csv = OUT_DIR / "catalyst_control_summary.csv"
    results_df.to_csv(out_csv, index=False)
    logger.info("\nSaved: %s\n%s", out_csv, results_df.to_string())

    # ── Plot: faceted by catalyst type ─────────────────────────────────────
    if results_df.empty:
        logger.warning("No results — skipping plot")
        return

    _plot(results_df, OUT_DIR)


def _plot(df: pd.DataFrame, out_dir: Path):
    cat_types = df["catalyst_type"].unique()
    n_cats = len(cat_types)

    fig, axes = plt.subplots(1, n_cats, figsize=(4 * n_cats, 4), squeeze=False)

    for ax, cat in zip(axes[0], cat_types):
        sub_df = df[df["catalyst_type"] == cat]
        pivot = sub_df.pivot_table(index="substrate", values="shap_mean_sub_homo")
        pivot = pivot.reindex(SUBSTRATE_ORDER).dropna()
        x = np.arange(len(pivot))
        colors = ["tab:red" if v < 0 else "tab:blue" for v in pivot.values.flatten()]
        ax.bar(x, pivot.values.flatten(), color=colors, edgecolor="black", alpha=0.8)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [s.replace(" oxide", "").replace(" Isopropyl glycidyl ether", "IGE")
             for s in pivot.index],
            rotation=25, ha="right")
        ax.set_ylabel("sub_homo_eV SHAP")
        ax.set_title(cat.replace("_", " ").title())
        ax.grid(axis="y", alpha=0.3)

    blue_patch = mpatches.Patch(color="tab:blue", label="Positive")
    red_patch  = mpatches.Patch(color="tab:red",  label="Negative (CHO pattern)")
    fig.legend(handles=[blue_patch, red_patch], loc="lower right", ncol=2)
    fig.suptitle("Catalyst-type control: CHO sub_homo_eV SHAP reversal",
                 y=1.02, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_catalyst_control.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", out_dir / "fig_catalyst_control.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="808 catalyst-type control SHAP")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    main()
    sys.exit(0)
