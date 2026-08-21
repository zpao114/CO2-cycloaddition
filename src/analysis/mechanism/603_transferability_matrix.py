"""Step 3: Catalyst-mechanism × Substrate transferability matrix.

What it does
------------
Given the mechanism labels from Step 1 (NUC / LAC / BAS / BIF / OTH) and the 5
epoxide substrates, build a matrix where:

    row = catalyst mechanism class
    col = substrate
    cell = (n reactions, mean yield, std yield)

The cells reveal:
  * where experimental coverage is rich (n large + high yield)
  * where the literature is silent (n=0) -- the *transferability gaps*

Also produce a side-product heatmap: aggregated by substrate × T-bucket to show
which (substrate, T) regions are best characterised.

Output
------
results/cross_tab_mech_substrate.csv          -- the (n, mean, std) table
results/cross_tab_substrate_Tbucket.csv       -- substrate × T-bucket heatmap
results/transferability_matrix.csv            -- same mech-x-substrate but only
                                                 n (so it can be plotted as
                                                 empty-cell emphasis)
results/cross_tab_summary.json                -- high-level stats

Figures (printed, no on-disk PNGs here):
  fig-transferability.png                     -- double heatmap (n and mean)
  fig-substrate-T.png                         -- substrate × T
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# Make src/ importable for paths.py
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = Path(__file__).resolve().parents[2]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))
from paths import (  # noqa: E402
    RESULTS_TRANSFERABILITY,
    RESULTS_CHO_DIAGNOSTIC,
    DATA_PROCESSED,
)


# FIX (2026-08-19): Use dynamic paths. Previously used hardcoded old-repo paths.
#   CAT_MECH_CSV → data/processed/catalyst_mechanism.csv (produced by 601)
#   REACTION_CSV → results/results_cho_diagnostic/co2_drfp_xtb_extended.csv (canonical master)
#   OUT_DIR      → results_transferability/ (canonical results dir)
CAT_MECH_CSV   = str(DATA_PROCESSED / "catalyst_mechanism.csv")
REACTION_CSV   = str(RESULTS_CHO_DIAGNOSTIC / "co2_drfp_xtb_extended.csv")
OUT_DIR        = str(RESULTS_TRANSFERABILITY)
os.makedirs(OUT_DIR, exist_ok=True)


# -----------------------------------------------------------------------------
# Logger
# -----------------------------------------------------------------------------
logger = logging.getLogger("603_transferability_matrix")
if not logger.handlers:
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(h)
    logger.setLevel(logging.INFO)
logger.propagate = False

# Make a coherent temperature bucketing that respects the dataset's
# experimental spread (0-200 °C).
T_BINS = [-1, 25, 60, 90, 120, 150, 200]
T_LABELS = ["<25", "25-60", "60-90", "90-120", "120-150", "150-200"]


def _attach_mechanism(df: pd.DataFrame, cat_df: pd.DataFrame) -> pd.DataFrame:
    """Attach mechanism label to each reaction row by catalyst_1_name."""
    m = cat_df[["name", "mechanism"]].rename(columns={"name": "catalyst_1_name",
                                                       "mechanism": "mech_label"})
    return df.merge(m, on="catalyst_1_name", how="left")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    if not os.path.exists(CAT_MECH_CSV):
        logger.error("Missing input: %s (run 601_catalyst_mechanism_v2.py first)", CAT_MECH_CSV)
        sys.exit(2)
    if not os.path.exists(REACTION_CSV):
        logger.error("Missing input: %s (run tier_main first)", REACTION_CSV)
        sys.exit(2)

    cat_df = pd.read_csv(CAT_MECH_CSV)
    rx_full = pd.read_csv(REACTION_CSV)
    rx = _attach_mechanism(rx_full, cat_df)

    # Drop NOT_CAT and missing mech (data hygiene)
    rx = rx[rx["mech_label"].notna()]
    rx = rx[rx["mech_label"].isin(["NUC", "LAC", "BAS", "BIF", "OTH"])]
    logger.info("Reactions after mechanism attach + filter: %d / %d",
                len(rx), len(rx_full))

    # Substrate simplification (5 unique reactants)
    rx["substrate"] = rx["reactant_name"].str.strip()

    # Bucket T
    rx["T_bucket"] = pd.cut(rx["temperature (°)"], bins=T_BINS, labels=T_LABELS)

    # ---- mech x substrate ----
    grp = rx.groupby(["mech_label", "substrate"], observed=True)
    table = grp["yield (%)"].agg(["count", "mean", "std"]).reset_index()
    table.to_csv(os.path.join(OUT_DIR, "cross_tab_mech_substrate.csv"), index=False)

    # pivot for plotting
    n_mat = grp.size().unstack(fill_value=0)
    mean_mat = grp["yield (%)"].mean().unstack()
    # include all mech classes and all substrates (use reindex)
    all_mech = ["NUC", "LAC", "BAS", "BIF", "OTH"]
    all_sub = sorted(rx["substrate"].unique().tolist())
    n_mat = n_mat.reindex(index=all_mech, columns=all_sub, fill_value=0)
    mean_mat = mean_mat.reindex(index=all_mech, columns=all_sub)

    logger.info("\n=== n matrix ===\n%s", n_mat)
    logger.info("\n=== mean yield matrix ===\n%s", mean_mat.round(1))

    # save transferability (n)
    n_mat.to_csv(os.path.join(OUT_DIR, "transferability_matrix.csv"))

    # ---- substrate x T-bucket ----
    grp2 = rx.groupby(["substrate", "T_bucket"], observed=True)
    sub_T = grp2["yield (%)"].agg(["count", "mean"]).reset_index()
    sub_T.to_csv(os.path.join(OUT_DIR, "cross_tab_substrate_Tbucket.csv"), index=False)

    # ---- summary ----
    n_pairs = (n_mat > 0).values.sum()
    n_total = rx.shape[0]
    total_possible = n_mat.shape[0] * n_mat.shape[1]
    summary = {
        "n_reactions_used": int(n_total),
        "n_mech_classes": int(n_mat.shape[0]),
        "n_substrates": int(n_mat.shape[1]),
        "n_covered_pairs": int(n_pairs),
        "n_possible_pairs": int(total_possible),
        "coverage": float(n_pairs / total_possible),
        "n_empty_pairs": int(total_possible - n_pairs),
        "substrate_counts": rx["substrate"].value_counts().to_dict(),
        "T_bucket_counts": rx["T_bucket"].value_counts().sort_index().to_dict(),
        "mech_counts": rx["mech_label"].value_counts().to_dict(),
    }
    with open(os.path.join(OUT_DIR, "cross_tab_summary.json"), "w",
              encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info("\n=== Summary ===")
    for k, v in summary.items():
        logger.info("  %s: %s", k, v)

    # ---- figures ----
    fig_dir = os.path.join(OUT_DIR, "figs")
    os.makedirs(fig_dir, exist_ok=True)

    # Figure 1: double heatmap (n log, mean)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    # n (log)
    im0 = axes[0].imshow(np.where(n_mat == 0, np.nan, n_mat.values),
                         aspect="auto", cmap="Blues",
                         norm=LogNorm(vmin=1, vmax=max(10, n_mat.values.max())))
    axes[0].set_xticks(range(n_mat.shape[1]))
    axes[0].set_xticklabels(n_mat.columns, rotation=30, ha="right")
    axes[0].set_yticks(range(n_mat.shape[0]))
    axes[0].set_yticklabels(n_mat.index)
    axes[0].set_title(f"# reactions (log scale)\ncoverage = {n_pairs}/{total_possible}")
    plt.colorbar(im0, ax=axes[0], label="count")
    # overlay n in each cell
    for i in range(n_mat.shape[0]):
        for j in range(n_mat.shape[1]):
            v = n_mat.values[i, j]
            if v > 0:
                axes[0].text(j, i, str(int(v)), ha="center", va="center",
                             color="white" if v > 30 else "black", fontsize=8)

    # mean
    im1 = axes[1].imshow(mean_mat.values, aspect="auto", cmap="RdYlGn",
                         vmin=0, vmax=100)
    axes[1].set_xticks(range(mean_mat.shape[1]))
    axes[1].set_xticklabels(mean_mat.columns, rotation=30, ha="right")
    axes[1].set_yticks(range(mean_mat.shape[0]))
    axes[1].set_yticklabels(mean_mat.index)
    axes[1].set_title("mean yield (%)")
    plt.colorbar(im1, ax=axes[1], label="yield %")
    for i in range(mean_mat.shape[0]):
        for j in range(mean_mat.shape[1]):
            v = mean_mat.values[i, j]
            if not np.isnan(v):
                axes[1].text(j, i, f"{v:.0f}", ha="center", va="center",
                             color="black", fontsize=8)
    fig.suptitle("Transferability matrix: catalyst mechanism × substrate", y=1.02)
    fig.tight_layout()
    fig_path = os.path.join(fig_dir, "transferability_matrix.png")
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved fig: %s", fig_path)

    # Figure 2: substrate × T heatmap
    sub_T_mean = grp2["yield (%)"].mean().unstack()
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(sub_T_mean.values, aspect="auto", cmap="RdYlGn",
                   vmin=0, vmax=100)
    ax.set_xticks(range(sub_T_mean.shape[1]))
    ax.set_xticklabels(sub_T_mean.columns.astype(str))
    ax.set_yticks(range(sub_T_mean.shape[0]))
    ax.set_yticklabels(sub_T_mean.index)
    for i in range(sub_T_mean.shape[0]):
        for j in range(sub_T_mean.shape[1]):
            v = sub_T_mean.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=8)
    plt.colorbar(im, ax=ax, label="mean yield (%)")
    ax.set_title("Substrate × Temperature bucket")
    fig.tight_layout()
    fig_path2 = os.path.join(fig_dir, "substrate_T_matrix.png")
    fig.savefig(fig_path2, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved fig: %s", fig_path2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="603 transferability matrix (mech × substrate).")
    parser.add_argument("--force", action="store_true",
                        help="Pipeline-compatibility flag; this script always fully regenerates its outputs.")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging.")
    args = parser.parse_args()
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    main()
    sys.exit(0)