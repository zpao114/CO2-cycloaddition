"""Generate LOSO root cause analysis figure.

This figure visualizes why LOSO R虏 is negative:
- Per-substrate prediction bias (CHO dominates with +34.6% bias)
- Actual vs predicted yields
- Mechanism class breakdown

Output: results_step7_improved_loso/fig_loso_root_cause.png
"""
from __future__ import annotations

import io
import os
import sys
import json
import warnings

warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from sklearn.metrics import r2_score, mean_absolute_error


PROJECT_ROOT = os.environ.get("CO2_PROJECT_ROOT", r"D:\machine-learning\CO2-cycloaddition")
DATA_CSV = os.path.join(PROJECT_ROOT, 'results/results_cho_diagnostic/co2_drfp_xtb_extended.csv')
MECH_CSV = os.path.join(PROJECT_ROOT, 'data/processed/catalyst_mechanism.csv')
OUT_DIR = os.path.join(PROJECT_ROOT, "results_step7_improved_loso")
os.makedirs(OUT_DIR, exist_ok=True)


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_CSV, encoding="utf-8-sig")
    df = df[df["extraction_status"] == "valid"].copy()
    df = df.dropna(subset=["yield (%)"])
    df = df[df["yield (%)"] > 0].reset_index(drop=True)
    
    mech = pd.read_csv(MECH_CSV)
    mech = mech[["name", "mechanism"]].rename(columns={
        "name": "catalyst_1_name",
        "mechanism": "mech_label"
    })
    df = df.merge(mech, on="catalyst_1_name", how="left")
    df["mech_label"] = df["mech_label"].fillna("UNK")
    
    return df


def compute_loso_predictions(df: pd.DataFrame) -> dict:
    """
    Compute per-sample LOSO predictions using substrate-aware prior.
    Returns detailed predictions for visualization.
    """
    terminal_subs = ["Propylene oxide", "Epichlorohydrin", "Styrene oxide", "Isopropyl glycidyl ether"]
    CHO = "Cyclohexene oxide"
    
    all_data = []
    
    for held_sub in sorted(df["reactant_name"].unique()):
        train_mask = df["reactant_name"].values != held_sub
        test_mask = df["reactant_name"].values == held_sub
        
        df_train = df[train_mask].reset_index(drop=True)
        df_test = df[test_mask].reset_index(drop=True)
        
        overall_mean = df_train["yield (%)"].mean()
        mech_means = df_train.groupby("catalyst_system_type")["yield (%)"].mean().to_dict()
        
        # Terminal 脳 mechanism statistics (for terminal test substrates)
        terminal_mech_means = {}
        for mech in df["catalyst_system_type"].unique():
            data = df_train[(df_train["reactant_name"].isin(terminal_subs)) & 
                           (df_train["catalyst_system_type"] == mech)]
            if len(data) > 0:
                terminal_mech_means[mech] = data["yield (%)"].mean()
        
        # Compute predictions for each test sample
        for _, row in df_test.iterrows():
            sub = row["reactant_name"]
            mech = row["catalyst_system_type"]
            actual = row["yield (%)"]
            
            if sub in terminal_subs:
                pred = terminal_mech_means.get(mech, overall_mean)
            else:
                pred = mech_means.get(mech, overall_mean)
            
            all_data.append({
                "substrate": sub,
                "mechanism": mech,
                "actual": actual,
                "predicted": pred,
                "error": actual - pred,
            })
    
    return pd.DataFrame(all_data)


def generate_figure(df_loso: pd.DataFrame, out_path: str):
    """Generate a comprehensive 4-panel figure."""
    
    # Set style
    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
    })
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    
    # Color palette per substrate
    substrate_colors = {
        "Cyclohexene oxide": "#d62728",         # red (problem)
        "Propylene oxide": "#2ca02c",            # green
        "Epichlorohydrin": "#1f77b4",            # blue
        "Styrene oxide": "#ff7f0e",              # orange
        "Isopropyl glycidyl ether": "#9467bd",   # purple
    }
    
    substrate_short = {
        "Cyclohexene oxide": "CHO",
        "Propylene oxide": "PO",
        "Epichlorohydrin": "ECH",
        "Styrene oxide": "SO",
        "Isopropyl glycidyl ether": "IGE",
    }
    
    substrates = sorted(df_loso["substrate"].unique())
    
    # ====== Panel A: Actual vs Predicted scatter per substrate ======
    ax = axes[0, 0]
    for sub in substrates:
        data = df_loso[df_loso["substrate"] == sub]
        ax.scatter(data["predicted"], data["actual"], 
                  alpha=0.4, s=18, color=substrate_colors[sub],
                  label=f"{substrate_short[sub]} (n={len(data)})")
    
    ax.plot([0, 100], [0, 100], "k--", alpha=0.6, lw=1.5, label="y=x (perfect)")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Predicted Yield (%)", fontweight="bold")
    ax.set_ylabel("Actual Yield (%)", fontweight="bold")
    ax.set_title("(A) LOSO Predicted vs Actual Yield", fontweight="bold")
    ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    # Compute R虏 for whole dataset
    r2_all = r2_score(df_loso["actual"], df_loso["predicted"])
    ax.text(0.97, 0.03, f"R虏 = {r2_all:.4f}\nMAE = {mean_absolute_error(df_loso['actual'], df_loso['predicted']):.2f}",
            transform=ax.transAxes, ha="right", va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.9))
    
    # ====== Panel B: Per-substrate bias (bar chart) ======
    ax = axes[0, 1]
    
    bias_data = []
    for sub in substrates:
        data = df_loso[df_loso["substrate"] == sub]
        bias_data.append({
            "substrate": sub,
            "short": substrate_short[sub],
            "actual_mean": data["actual"].mean(),
            "pred_mean": data["predicted"].mean(),
            "bias": data["actual"].mean() - data["predicted"].mean(),
            "r2": r2_score(data["actual"], data["predicted"]),
            "n": len(data),
        })
    
    bias_df = pd.DataFrame(bias_data)
    
    x_pos = np.arange(len(bias_df))
    colors = [substrate_colors[s] for s in bias_df["substrate"]]
    bars = ax.bar(x_pos, bias_df["bias"], color=colors, alpha=0.8, edgecolor="black")
    
    ax.axhline(y=0, color="k", linestyle="-", lw=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"{s}\n(n={n})" for s, n in zip(bias_df["short"], bias_df["n"])])
    ax.set_xlabel("Substrate (held out)", fontweight="bold")
    ax.set_ylabel("Actual Mean - Predicted Mean (%)", fontweight="bold")
    ax.set_title("(B) Per-Substrate Prediction Bias", fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    
    # Annotate bars
    for i, (bar, row) in enumerate(zip(bars, bias_df.itertuples())):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., 
                height + (1 if height >= 0 else -3),
                f"{height:+.1f}",
                ha="center", va="bottom" if height >= 0 else "top",
                fontsize=9, fontweight="bold")
    
    # Add R虏 values as text
    for i, row in enumerate(bias_df.itertuples()):
        ax.text(i, -40, f"R虏={row.r2:.2f}", ha="center", va="top",
                fontsize=8, color="gray")
    
    ax.set_ylim(-45, 45)
    
    # ====== Panel C: Box plot of actual vs predicted by substrate ======
    ax = axes[1, 0]
    
    box_data = []
    positions_actual = []
    positions_pred = []
    labels = []
    colors_list = []
    
    pos = 0
    for i, sub in enumerate(substrates):
        data = df_loso[df_loso["substrate"] == sub]
        box_data.append(data["actual"].values)
        positions_actual.append(pos)
        pos += 1
        box_data.append(data["predicted"].values)
        positions_pred.append(pos)
        pos += 2.5  # gap between substrate groups
    
    bp = ax.boxplot(box_data, positions=positions_actual + positions_pred,
                    widths=0.7, patch_artist=True, showfliers=False)
    
    # Color boxes
    for i, patch in enumerate(bp["boxes"]):
        sub_idx = i // 2
        sub = substrates[sub_idx]
        if i % 2 == 0:  # actual
            patch.set_facecolor(substrate_colors[sub])
            patch.set_alpha(0.5)
        else:  # predicted
            patch.set_facecolor(substrate_colors[sub])
            patch.set_alpha(0.9)
            patch.set_hatch("//")
    
    # Custom x-tick labels
    sub_centers = [positions_actual[i] + 0.5 for i in range(len(substrates))]
    ax.set_xticks(sub_centers)
    ax.set_xticklabels([substrate_short[s] for s in substrates])
    ax.set_xlabel("Substrate (held out)", fontweight="bold")
    ax.set_ylabel("Yield (%)", fontweight="bold")
    ax.set_title("(C) Distribution: Actual (light) vs Predicted (hatched)", fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    
    # Custom legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="gray", alpha=0.5, label="Actual"),
        Patch(facecolor="gray", alpha=0.9, hatch="//", label="Predicted"),
    ]
    ax.legend(handles=legend_elements, loc="upper right")
    
    ax.set_ylim(-5, 105)
    
    # ====== Panel D: Mechanism 脳 Substrate heatmap ======
    ax = axes[1, 1]
    
    # Compute mean yield per (substrate, mechanism) cell
    pivot_data = df_loso.groupby(["substrate", "mechanism"]).agg({
        "actual": "mean",
        "predicted": "mean",
        "error": "mean"
    }).reset_index()
    
    mechanisms = sorted(pivot_data["mechanism"].unique())
    
    # Create matrix: error (actual - predicted)
    error_matrix = np.full((len(substrates), len(mechanisms)), np.nan)
    actual_matrix = np.full((len(substrates), len(mechanisms)), np.nan)
    pred_matrix = np.full((len(substrates), len(mechanisms)), np.nan)
    count_matrix = np.zeros((len(substrates), len(mechanisms)), dtype=int)
    
    for _, row in pivot_data.iterrows():
        i = substrates.index(row["substrate"])
        j = mechanisms.index(row["mechanism"])
        error_matrix[i, j] = row["error"]
        actual_matrix[i, j] = row["actual"]
        pred_matrix[i, j] = row["predicted"]
    
    # Count samples per cell
    for _, row in df_loso.iterrows():
        i = substrates.index(row["substrate"])
        j = mechanisms.index(row["mechanism"])
        count_matrix[i, j] += 1
    
    # Plot heatmap
    im = ax.imshow(error_matrix, cmap="RdBu_r", vmin=-50, vmax=50, aspect="auto")
    
    ax.set_xticks(range(len(mechanisms)))
    ax.set_xticklabels(mechanisms, rotation=45, ha="right")
    ax.set_yticks(range(len(substrates)))
    ax.set_yticklabels([substrate_short[s] for s in substrates])
    ax.set_xlabel("Mechanism Class", fontweight="bold")
    ax.set_ylabel("Substrate (held out)", fontweight="bold")
    ax.set_title("(D) Bias Heatmap: Actual - Predicted (%)", fontweight="bold")
    
    # Annotate cells
    for i in range(len(substrates)):
        for j in range(len(mechanisms)):
            err = error_matrix[i, j]
            if not np.isnan(err):
                color = "white" if abs(err) > 25 else "black"
                txt = f"{err:+.0f}\n(n={count_matrix[i, j]})"
                ax.text(j, i, txt, ha="center", va="center",
                       color=color, fontsize=8)
    
    plt.colorbar(im, ax=ax, label="Error (%)")
    
    # Overall title
    fig.suptitle("LOSO Root Cause Analysis: CHO's Structural Bias Dominates",
                 fontsize=14, fontweight="bold", y=0.995)
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.savefig(out_path.replace(".png", ".pdf"), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    
    print(f"Figure saved to: {out_path}")
    print(f"PDF version saved to: {out_path.replace('.png', '.pdf')}")


def main():
    print("=" * 70)
    print("Generating LOSO Root Cause Analysis Figure")
    print("=" * 70)
    
    # Load data
    print("\n[1/3] Loading data...")
    df = load_data()
    print(f"    Loaded {len(df)} reactions")
    
    # Compute LOSO predictions
    print("\n[2/3] Computing LOSO predictions...")
    df_loso = compute_loso_predictions(df)
    print(f"    Generated {len(df_loso)} predictions")
    
    # Print summary
    print("\n    Summary statistics:")
    for sub in sorted(df_loso["substrate"].unique()):
        data = df_loso[df_loso["substrate"] == sub]
        bias = data["actual"].mean() - data["predicted"].mean()
        r2 = r2_score(data["actual"], data["predicted"])
        print(f"    {sub:28s}: bias={bias:+6.2f}%, R虏={r2:+.4f}")
    
    # Generate figure
    print("\n[3/3] Generating figure...")
    out_path = os.path.join(OUT_DIR, "fig_loso_root_cause.png")
    generate_figure(df_loso, out_path)
    
    # Save CSV with predictions
    csv_path = os.path.join(OUT_DIR, "loso_predictions.csv")
    df_loso.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\nPredictions saved to: {csv_path}")
    
    print("\n" + "=" * 70)
    print("Figure generation complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()