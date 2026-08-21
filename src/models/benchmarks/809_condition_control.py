# -*- coding: utf-8 -*-
"""
809_condition_control.py
========================
#4 — Reaction-condition control for the CHO sub_homo_eV sign reversal.

Scientific question
-------------------
Is the CHO sub_homo_eV sign reversal explained by systematic differences
in reaction conditions (temperature, pressure, time, catalyst loading)
between CHO reactions and terminal-epoxide reactions?

Protocol (matching / stratification approach)
---------------------------------------------
Two complementary strategies:

  Strategy A — Stratification:
    Bin the full dataset into T bins (e.g. 3 bins: <100°C, 100–120°C, >120°C).
    Within each bin, compute SHAP for sub_homo_eV separately for CHO and
    terminal substrates.  If the reversal disappears within matched bins,
    it was an artifact of T distribution; if it persists, the reversal is
    robust to condition differences.

  Strategy B — Residualisation:
    Regress y on [T, P, time, loading] within the terminal subset; take
    residuals as "condition-normalised yield".  Retrain XGBoost on residuals
    and check if the CHO sub_homo_eV reversal persists.

Outputs
-------
results_shap_comprehensive/condition_control/
  stratification_summary.csv       — SHAP per (T_bin, substrate)
  residualisation_summary.csv     — residualised SHAP results
  fig_stratification.png           — heatmap: T_bin × substrate SHAP
  fig_residualisation.png          — comparison bar chart

Tier placement: tier_si
Typical runtime: ~6 min
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
import matplotlib.colors as mcolors

_SCRIPT_DIR = Path(__file__).resolve().parents[1]
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
from paths import RESULTS_DIR

sys.path.insert(0, str(_SCRIPT_DIR / "benchmarks"))
from _shap_infra import (
    SUBSTRATE_ORDER, CHO_NAME, KEY_FEATURES,
    load_X_y_groups, compute_group_shap, stratified_group_shap,
    logger,
)

OUT_DIR = RESULTS_DIR / "results_shap_comprehensive" / "condition_control"
os.makedirs(OUT_DIR, exist_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--n-bins", type=int, default=3,
                        help="Number of temperature bins (default: 3)")
    args = parser.parse_args()

    X, y, groups, feat_names, df = load_X_y_groups()
    feat_idx_sub_homo = feat_names.index("sub_homo_eV") if "sub_homo_eV" in feat_names else None

    # ── Strategy A: Stratification on temperature ──────────────────────────
    logger.info("=== Strategy A: Stratified by temperature (%d bins) ===", args.n_bins)
    strat_results = []

    try:
        strat_idx = feat_names.index("temperature (°)")
    except ValueError:
        logger.error("temperature (°) not in features — cannot stratify")
        return

    T_vals = X[:, strat_idx]
    # Build roughly equal-size bins
    bins = np.linspace(T_vals.min(), T_vals.max(), args.n_bins + 1)
    bins[0] = -np.inf
    bins[-1] = np.inf
    bin_labels = [f"[{bins[i]:.0f}–{bins[i+1]:.0f})°C" for i in range(args.n_bins)]

    for bi, (lo, hi) in enumerate(zip(bins[:-1], bins[1:])):
        mask = (T_vals >= lo) & (T_vals < hi)
        X_b, y_b, groups_b = X[mask], y[mask], [g for i, g in enumerate(groups) if mask[i]]
        if len(groups_b) < 20:
            continue
        logger.info("  T bin %s: n=%d", bin_labels[bi], mask.sum())

        cho_mask = np.array([g == CHO_NAME for g in groups_b])
        term_mask = ~cho_mask

        if cho_mask.sum() < 5 or term_mask.sum() < 5:
            logger.warning("  Skipping %s (group too small)", bin_labels[bi])
            continue

        for sub_name in SUBSTRATE_ORDER:
            sub_mask = np.array([g == sub_name for g in groups_b])
            if sub_mask.sum() < 3:
                continue
            try:
                res = compute_group_shap(X_b, y_b, sub_mask, feat_names, random_state=42)
            except Exception as e:
                logger.error("  SHAP failed: %s", e)
                continue
            strat_results.append(dict(
                T_bin=bin_labels[bi],
                T_lo=float(lo), T_hi=float(hi),
                substrate=sub_name,
                n=int(sub_mask.sum()),
                shap_mean=float(res["shap_mean"].mean()),
                shap_sub_homo=float(res["shap_mean"][feat_idx_sub_homo]) if feat_idx_sub_homo is not None else np.nan,
                r2=float(res["r2"]),
            ))

    strat_df = pd.DataFrame(strat_results)
    strat_df.to_csv(OUT_DIR / "stratification_summary.csv", index=False)
    logger.info("\n%s", strat_df.to_string())

    # ── Strategy B: Residualisation ─────────────────────────────────────────
    logger.info("\n=== Strategy B: Residualisation on conditions ===")
    resid_results = []

    for sub_name in SUBSTRATE_ORDER:
        sub_mask = np.array([g == sub_name for g in groups])
        if sub_mask.sum() < 10:
            continue
        X_sub = X[sub_mask]
        y_sub = y[sub_mask]

        # Split terminal into train/complement
        term_mask = ~np.array([g == CHO_NAME for g in groups])
        term_mask = term_mask & np.array([g in SUBSTRATE_ORDER for g in groups])
        X_term = X[term_mask]
        y_term = y[term_mask]

        if X_term.shape[0] < 20 or sub_mask.sum() < 10:
            continue

        try:
            # Residualise terminal on conditions only
            from sklearn.linear_model import LinearRegression
            cond_idx = [i for i, fn in enumerate(feat_names)
                        if fn in ("temperature (°)", "pressure (MPa)", "time (h)",
                                  "catalyst_1_loading_mol%")]
            if len(cond_idx) < 2:
                logger.warning("Not enough condition features for residualisation")
                continue

            lr = LinearRegression()
            lr.fit(X_term[:, cond_idx], y_term)
            y_resid_term = y_term - lr.predict(X_term[:, cond_idx])

            # Also residualise the target substrate
            y_resid_sub = y_sub - lr.predict(X_sub[:, cond_idx])

            # Now train XGB on residuals
            # Use terminal residuals as train, target substrate residuals as test
            res = compute_group_shap(
                np.vstack([X_term, X_sub]),
                np.concatenate([y_resid_term, y_resid_sub]),
                np.arange(len(y_sub)) + len(y_resid_term),  # last n rows are test
                feat_names,
                random_state=42,
            )
            resid_results.append(dict(
                substrate=sub_name,
                n=int(sub_mask.sum()),
                shap_sub_homo=float(res["shap_mean"][feat_idx_sub_homo]) if feat_idx_sub_homo is not None else np.nan,
                shap_mean=float(res["shap_mean"].mean()),
                r2=float(res["r2"]),
            ))
            logger.info("  %s: sub_homo SHAP=%.4f  R^2=%.3f",
                       sub_name, resid_results[-1]["shap_sub_homo"], resid_results[-1]["r2"])
        except Exception as e:
            logger.error("  Residualisation failed for %s: %s", sub_name, e)
            continue

    resid_df = pd.DataFrame(resid_results)
    resid_df.to_csv(OUT_DIR / "residualisation_summary.csv", index=False)

    # ── Plots ────────────────────────────────────────────────────────────────
    _plot_stratification(strat_df, OUT_DIR)
    _plot_residualisation(resid_df, OUT_DIR)
    logger.info("All done. Results in %s", OUT_DIR)


def _plot_stratification(df: pd.DataFrame, out_dir: Path):
    if df.empty:
        return
    pivot = df.pivot_table(
        index="T_bin", columns="substrate", values="shap_sub_homo"
    )
    pivot = pivot.reindex(columns=SUBSTRATE_ORDER)
    pivot = pivot.sort_index()

    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdBu_r", vmin=-0.1, vmax=0.1)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(
        [c.replace(" oxide", "").replace(" Isopropyl glycidyl ether", "IGE")
         for c in pivot.columns], rotation=25, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title("Stratified SHAP (sub_homo_eV): T bin × substrate")
    plt.colorbar(im, ax=ax, label="mean SHAP")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_stratification.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", out_dir / "fig_stratification.png")


def _plot_residualisation(df: pd.DataFrame, out_dir: Path):
    if df.empty:
        return
    pivot = df.set_index("substrate").reindex(SUBSTRATE_ORDER).dropna()
    x = np.arange(len(pivot))
    colors = ["tab:red" if v < 0 else "tab:blue" for v in pivot["shap_sub_homo"]]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x, pivot["shap_sub_homo"].values, color=colors, edgecolor="black", alpha=0.8)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [s.replace(" oxide", "").replace(" Isopropyl glycidyl ether", "IGE")
         for s in pivot.index], rotation=25, ha="right")
    ax.set_ylabel("sub_homo_eV SHAP (condition-normalised)")
    ax.set_title("Condition-residualised SHAP: sub_homo_eV by substrate")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_residualisation.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", out_dir / "fig_residualisation.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="809 reaction-condition control SHAP")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--n-bins", type=int, default=3)
    args = parser.parse_args()
    main()
    sys.exit(0)
