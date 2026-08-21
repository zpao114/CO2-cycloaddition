# -*- coding: utf-8 -*-
"""
810_yield_distribution.py
========================
#6 — Publication bias assessment: yield distribution analysis.

Scientific question
------------------
Does the extracted dataset over-represent high-yield experiments, and if so,
does the CHO sub_homo_eV sign reversal persist when we restrict to
low-yield (< 70%) experiments?

Outputs
-------
results_shap_comprehensive/yield_distribution/
  yield_distribution_summary.csv     — yield stats per substrate
  yield_distribution_hist.png        — overlaid histograms per substrate
  yield_subset_shap_summary.csv      — sub_homo_eV SHAP on < 70% subset
  fig_yield_subset_comparison.png    — bar chart: all-data vs <70% subset

Tier placement: tier_data  (needs data processed, runs independently)
Typical runtime: ~2 min (plotting only)
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

OUT_DIR = RESULTS_DIR / "results_shap_comprehensive" / "yield_distribution"
os.makedirs(OUT_DIR, exist_ok=True)

YIELD_THRESHOLD = 70.0   # (%)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--threshold", type=float, default=YIELD_THRESHOLD,
                        help="Yield threshold for subset analysis (default: 70%%)")
    parser.add_argument("--shap-threshold", type=float, default=20,
                        help="Minimum samples per group for SHAP (default: 20)")
    args = parser.parse_args()

    # ── Load data ─────────────────────────────────────────────────────────────
    X, y, groups, feat_names, df = load_X_y_groups()
    y_pct = y * 100.0   # back to 0–100 scale

    feat_idx_sub_homo = feat_names.index("sub_homo_eV") if "sub_homo_eV" in feat_names else None

    # ── 1. Distribution summary per substrate ──────────────────────────────────
    rows = []
    for sub in SUBSTRATE_ORDER:
        mask = np.array([g == sub for g in groups])
        ys = y_pct[mask]
        rows.append(dict(
            substrate=sub,
            n=int(mask.sum()),
            yield_mean=float(ys.mean()),
            yield_median=float(np.median(ys)),
            yield_std=float(ys.std()),
            yield_min=float(ys.min()),
            yield_max=float(ys.max()),
            pct_lt_70=float((ys < args.threshold).mean() * 100),
            pct_gt_90=float((ys > 90).mean() * 100),
            pct_lt_30=float((ys < 30).mean() * 100),
        ))
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(OUT_DIR / "yield_distribution_summary.csv", index=False)
    logger.info("\n=== Yield distribution summary ===\n%s", summary_df.round(2).to_string())

    # ── 2. Histogram per substrate ───────────────────────────────────────────
    fig, axes = plt.subplots(1, 5, figsize=(16, 3.5), sharey=False)
    for ax, sub in zip(axes, SUBSTRATE_ORDER):
        mask = np.array([g == sub for g in groups])
        ys = y_pct[mask]
        ax.hist(ys, bins=20, color="steelblue", edgecolor="black", alpha=0.8)
        ax.axvline(args.threshold, color="red", linestyle="--", linewidth=1.5,
                   label=f"{args.threshold}%")
        ax.axvline(ys.mean(), color="orange", linestyle="-", linewidth=1.5,
                   label=f"μ={ys.mean():.1f}%")
        ax.set_title(sub.replace(" oxide", "\noxide").replace(" Isopropyl glycidyl ether", "IGE"),
                     fontsize=8)
        ax.set_xlabel("yield (%)")
        ax.set_xlim(0, 105)
        if ax == axes[0]:
            ax.set_ylabel("count")
        ax.legend(fontsize=7)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(f"Yield distribution per substrate (n={len(y_pct)} total)", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "yield_distribution_hist.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", OUT_DIR / "yield_distribution_hist.png")

    # ── 3. Overlaid density plot ──────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 4))
    for sub in SUBSTRATE_ORDER:
        mask = np.array([g == sub for g in groups])
        ys = y_pct[mask]
        ax.hist(ys, bins=20, alpha=0.4, label=sub.replace(" oxide", ""), density=True)
    ax.axvline(args.threshold, color="red", linestyle="--", linewidth=2,
               label=f"{args.threshold}% threshold")
    ax.set_xlabel("yield (%)")
    ax.set_ylabel("density")
    ax.set_title("Overlaid yield distributions by substrate")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "yield_distribution_overlaid.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", OUT_DIR / "yield_distribution_overlaid.png")

    # ── 4. SHAP on < threshold subset ────────────────────────────────────────
    logger.info("\n=== SHAP on < %d%% yield subset ===", args.threshold)
    low_yield_mask = y_pct < args.threshold
    X_low = X[low_yield_mask]
    y_low = y[low_yield_mask]
    groups_low = [g for i, g in enumerate(groups) if low_yield_mask[i]]

    shap_rows = []
    for sub_name in SUBSTRATE_ORDER:
        sub_mask = np.array([g == sub_name for g in groups_low])
        if sub_mask.sum() < args.shap_threshold:
            logger.warning("  Skipping %s (only %d < %d%% samples)",
                         sub_name, sub_mask.sum(), args.threshold)
            continue

        logger.info("  %s: n=%d", sub_name, sub_mask.sum())
        try:
            res = compute_group_shap(
                X_low, y_low, sub_mask, feat_names,
                random_state=42,
            )
        except Exception as e:
            logger.error("  SHAP failed: %s", e)
            continue

        shap_sub_homo = float(res["shap_mean"][feat_idx_sub_homo]) if feat_idx_sub_homo is not None else np.nan
        shap_rows.append(dict(
            substrate=sub_name,
            n=int(sub_mask.sum()),
            shap_mean=float(res["shap_mean"].mean()),
            shap_sub_homo=shap_sub_homo,
            r2=float(res["r2"]),
        ))

    low_df = pd.DataFrame(shap_rows)
    low_df.to_csv(OUT_DIR / "yield_subset_shap_summary.csv", index=False)
    logger.info("\n%s", low_df.to_string())

    # ── 5. Comparison bar chart: all vs <threshold ────────────────────────────
    # Load all-data SHAP from 807 (cross_model_shap_summary.csv has: substrate, model, feature, ...)
    all_data_path = RESULTS_DIR / "results_shap_comprehensive" / "cross_model" / "cross_model_shap_summary.csv"
    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(len(SUBSTRATE_ORDER))
    width = 0.35

    if all_data_path.exists():
        all_df = pd.read_csv(all_data_path)
        # cross_model_shap_summary.csv has cols: substrate, model, feature, shap_mean, ...
        if "feature" in all_df.columns and "model" in all_df.columns:
            all_df = all_df[(all_df["model"] == "XGB") & (all_df["feature"] == "sub_homo_eV")]
        elif "XGB" in all_df.columns:
            pass  # cross_model_sub_homo_eV.csv format: substrate,RF,XGB
        all_vals = all_df.set_index("substrate").reindex(SUBSTRATE_ORDER)["shap_mean"].values
    else:
        # Compute inline (fast: XGB only)
        all_vals = []
        for sub_name in SUBSTRATE_ORDER:
            sub_mask = np.array([g == sub_name for g in groups])
            if sub_mask.sum() < 10:
                all_vals.append(np.nan)
                continue
            res = compute_group_shap(X, y, sub_mask, feat_names, random_state=42)
            all_vals.append(float(res["shap_mean"][feat_idx_sub_homo]) if feat_idx_sub_homo is not None else np.nan)
        all_vals = np.array(all_vals)

    low_vals = low_df.set_index("substrate").reindex(SUBSTRATE_ORDER)["shap_sub_homo"].values

    b1 = ax.bar(x - width/2, all_vals, width, label=f"All data (≥0%)",
                color="steelblue", edgecolor="black", alpha=0.8)
    b2 = ax.bar(x + width/2, low_vals, width, label=f"<{args.threshold}% subset",
                color="coral", edgecolor="black", alpha=0.8)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [s.replace(" oxide", "").replace(" Isopropyl glycidyl ether", "IGE")
         for s in SUBSTRATE_ORDER], rotation=20, ha="right")
    ax.set_ylabel("sub_homo_eV SHAP (XGBoost)")
    ax.set_title(f"sub_homo_eV SHAP: all data vs <{args.threshold}% yield subset\n"
                 f"(blue=baseline, coral=<{args.threshold}% only)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_yield_subset_comparison.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", OUT_DIR / "fig_yield_subset_comparison.png")

    logger.info("\nAll done. Results in %s", OUT_DIR)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="810 yield distribution & subset SHAP")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--threshold", type=float, default=70.0)
    args = parser.parse_args()
    main()
    sys.exit(0)
