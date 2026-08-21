#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
regenerate_paper_figures.py
=============================
Re-generate all 5 figures for paper_draft_JC_Chinese.md with correct
chemical notation (CO₂, R², GFN2-xTB, ΔE_HL, etc.) rendered via
matplotlib's built-in Unicode + font fallback so they look correct
when embedded in both .docx (via pandoc) and .tex (via xelatex).

Usage
-----
    python regenerate_paper_figures.py
    # Outputs: fig_graphical_abstract.png, fig5_loso_protocol.png,
    #          fig4_loso_root_cause.png, fig6_transferability_matrix.png,
    #          fig7_shap_direction.png
"""

import json
import os
import sys
import io

# ── UTF-8 stdout/stderr ─────────────────────────────────────────────────────
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LogNorm
from sklearn.metrics import r2_score, mean_absolute_error

PROJECT = r"D:\machine-learning\CO2-cycloaddition"
os.chdir(PROJECT)

# ── Global matplotlib settings ───────────────────────────────────────────────
# Use a font with good Unicode support; DejaVu Sans is bundled with matplotlib
# and supports most Unicode subscripts/superscripts.
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "axes.unicode_minus": False,
})

# ── Subscript / superscript helpers (pure Unicode) ──────────────────────────
# These work correctly in matplotlib text rendering.
SUB2  = "\u2082"   # subscript 2  → CO₂
SUP2  = "\u00b2"   # superscript 2 → R²
SUP1  = "\u00b9"   # superscript 1 → ¹
SUB42 = "\u2074\u00b2"  # superscript 42 → ⁴²

# ── Colour palette ───────────────────────────────────────────────────────────
C_CHO  = "#d62728"   # red  — problem substrate
C_TERM = "#2ca02c"   # green — terminal substrates
C_OTH  = "#1f77b4"   # blue  — other

SUB_COLORS = {
    "Cyclohexene oxide":          C_CHO,
    "Propylene oxide":            C_TERM,
    "Epichlorohydrin":            C_OTH,
    "Styrene oxide":              "#ff7f0e",
    "Isopropyl glycidyl ether":   "#9467bd",
}
SUB_SHORT = {
    "Cyclohexene oxide":          "CHO",
    "Propylene oxide":            "PO",
    "Epichlorohydrin":            "ECH",
    "Styrene oxide":              "SO",
    "Isopropyl glycidyl ether":   "IGE",
}

# ═══════════════════════════════════════════════════════════════════════════════
# FIG 1 — GRAPHICAL ABSTRACT
# ═══════════════════════════════════════════════════════════════════════════════
def make_fig1_graphical_abstract():
    """Three-column overview: funnel + pie + heatmap."""

    # ── Data ────────────────────────────────────────────────────────────────
    funnel = [5263, 3733, 2602, 2316]
    funnel_labels = ["5,263\n(raw)", "3,733", "2,602", "2,316\n(final)"]

    pie_labels   = ["IL 79.6%", "MH 7.6%", "Mixed 6.7%", "BAS 2.8%", "Unknown 3.2%"]
    pie_sizes    = [79.6, 7.6, 6.7, 2.8, 3.2]
    pie_colors   = ["#4CAF50", "#2196F3", "#9C27B0", "#FF9800", "#9E9E9E"]

    # Mechanism bifurcation data
    bif_labels = [
        "Terminal\n(top-1=sub_homo_eV)",
        "CHO\n(top-1=time_log)"
    ]
    bif_colors = [C_TERM, C_CHO]

    # 5×5 heatmap (from cross_tab_mech_substrate.csv)
    cross = pd.read_csv("results/cross_tab_mech_substrate.csv", encoding="utf-8-sig")
    mat = cross.pivot(index="mech_label", columns="substrate", values="count")
    mat_yield = cross.pivot(index="mech_label", columns="substrate", values="mean")

    substrates_order = ["Cyclohexene oxide", "Epichlorohydrin",
                        "Isopropyl glycidyl ether", "Propylene oxide", "Styrene oxide"]
    mechanisms_order = ["NUC", "LAC", "BAS", "BIF", "OTH"]
    mat = mat.reindex(index=mechanisms_order, columns=substrates_order, fill_value=0)
    mat_yield = mat_yield.reindex(index=mechanisms_order, columns=substrates_order)

    fig = plt.figure(figsize=(14, 5))
    gs  = GridSpec(1, 3, figure=fig, width_ratios=[1.1, 0.9, 1.3], wspace=0.3)

    # ── Panel A: Funnel ─────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[0])
    bars = ax.barh(range(len(funnel)), funnel, color=["#90A4AE", "#78909C", "#607D8B", "#455A64"],
                   edgecolor="black", linewidth=0.8)
    for i, (v, lbl) in enumerate(zip(funnel, funnel_labels)):
        ax.text(v + 50, i, lbl, va="center", ha="left", fontsize=9, color="black")
        if i > 0:
            ax.annotate("", xy=(funnel[i-1] - 80, i - 0.4),
                       xytext=(funnel[i] + 80, i - 0.4),
                       arrowprops=dict(arrowstyle="->", color="#78909C", lw=1.5))
    ax.set_yticks([])
    ax.set_xlim(0, 6200)
    ax.set_xlabel("Number of reactions")
    ax.set_title("(a) Data-cleaning funnel", fontweight="bold")
    ax.spines["left"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # R² annotations
    ax.text(0.5, 0.02,
            f"LOSO R${SUP2} = $-$0.051\nRandom CV R${SUP2} = 0.318",
            transform=ax.transAxes, ha="center", va="bottom",
            fontsize=9, color="#b71c1c",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#ffebee", edgecolor="#b71c1c", lw=1))

    # ── Panel B: Pie + bifurcation ──────────────────────────────────────────
    ax = fig.add_subplot(gs[1])
    wedges, _ = ax.pie(pie_sizes, colors=pie_colors, startangle=90,
                        wedgeprops=dict(edgecolor="white", linewidth=1.2))
    ax.legend(wedges, pie_labels, loc="center left", bbox_to_anchor=(-0.5, 0.5),
              fontsize=8)
    ax.set_title("(b) Catalyst system\ndistribution", fontweight="bold")

    # Bifurcation arrows below pie
    ax2 = ax.twinx()
    ax2.set_ylim(0, 1)
    ax2.axis("off")
    ax2.annotate("", xy=(0.5, 0.25), xytext=(0.1, 0.15),
                 arrowprops=dict(arrowstyle="-|>", color=C_TERM, lw=2))
    ax2.text(0.3, 0.12, bif_labels[0], ha="center", fontsize=7.5, color=C_TERM)
    ax2.annotate("", xy=(0.5, 0.25), xytext=(0.9, 0.15),
                 arrowprops=dict(arrowstyle="-|>", color=C_CHO, lw=2))
    ax2.text(0.7, 0.12, bif_labels[1], ha="center", fontsize=7.5, color=C_CHO)
    ax2.text(0.5, 0.30, "Mechanism\nbifurcation", ha="center", fontsize=7,
             color="gray", style="italic")

    # ── Panel C: 5×5 heatmap ────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[2])
    n_mat = mat.values.astype(float)
    im = ax3.imshow(n_mat, cmap="Blues", aspect="auto",
                    norm=LogNorm(vmin=1, vmax=n_mat.max()))
    ax3.set_xticks(range(len(substrates_order)))
    ax3.set_xticklabels([SUB_SHORT[s] for s in substrates_order], fontsize=9)
    ax3.set_yticks(range(len(mechanisms_order)))
    ax3.set_yticklabels(mechanisms_order, fontsize=9)
    for i in range(len(mechanisms_order)):
        for j in range(len(substrates_order)):
            v = int(n_mat[i, j])
            color = "white" if v > 100 else "black"
            ax3.text(j, i, str(v), ha="center", va="center",
                     color=color, fontsize=8)
    # Highlight CHO column (index 0)
    for i in range(len(mechanisms_order)):
        rect = mpatches.Rectangle((0 - 0.5, i - 0.5), 1, 1,
                                   fill=False, edgecolor=C_CHO, linewidth=2)
        ax3.add_patch(rect)
    plt.colorbar(im, ax=ax3, label="n reactions", shrink=0.8)
    ax3.set_title("(c) 5×5 substrate × catalyst\ncoverage (n) matrix", fontweight="bold")
    ax3.set_xlabel("Substrate")
    ax3.set_ylabel("Catalyst mechanism")

    fig.suptitle("Graphical Abstract — CO" + SUB2 + " Cycloaddition LOSO Diagnostic",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    out = "fig_graphical_abstract.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 2 — LOSO / LOMO PROTOCOL COMPARISON
# ═══════════════════════════════════════════════════════════════════════════════
def make_fig2_loso_protocol():
    """Grouped bar chart: R² for LOSO / LOMO / LOSO×LOMO × X0 / X1."""

    df = pd.read_csv('results/results_step4/summary_protocol.csv", encoding="utf-8-sig")
    df["r2"] = df["r2"].astype(float)
    df["mae"] = df["mae"].astype(float)

    protocols = ["5fold", "LOSO", "LOMO", "LOSO" + chr(215) + "LOMO"]
    # Map column names
    protocol_col = "protocol"
    x0 = df[df["feature_set"] == "X0_xTB_only"].set_index(protocol_col)
    x1 = df[df["feature_set"] == "X1_xTB+mech"].set_index(protocol_col)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # ── R² bar chart ────────────────────────────────────────────────────────
    ax = axes[0]
    x = np.arange(len(protocols))
    w = 0.35
    bars_x0 = ax.bar(x - w/2, [x0.loc[p, "r2"] if p in x0.index else 0 for p in protocols],
                     w, label="X0 = xTB only", color="#64B5F6", edgecolor="black")
    bars_x1 = ax.bar(x + w/2, [x1.loc[p, "r2"] if p in x1.index else 0 for p in protocols],
                     w, label="X1 = xTB + mech one-hot", color="#1976D2", edgecolor="black")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(protocols, fontsize=9)
    ax.set_ylabel("R" + SUP2)
    ax.set_title("LOSO / LOMO Protocol R" + SUP2 + " Comparison\n(XGBoost)", fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # Annotate key values
    for i, p in enumerate(protocols):
        for bars, xdf in [(bars_x0, x0), (bars_x1, x1)]:
            if p not in xdf.index:
                continue
            v = xdf.loc[p, "r2"]
            offset = -0.05 if v >= 0 else 0.05
            ax.text(i + ( -0.35 if bars == bars_x0 else 0.35), v + offset,
                    f"{v:.3f}", ha="center", fontsize=7.5, color="black")

    # ── MAE bar chart ───────────────────────────────────────────────────────
    ax2 = axes[1]
    ax2.bar(x - w/2, [x0.loc[p, "mae"] if p in x0.index else 0 for p in protocols],
            w, label="X0 = xTB only", color="#90CAF9", edgecolor="black")
    ax2.bar(x + w/2, [x1.loc[p, "mae"] if p in x1.index else 0 for p in protocols],
            w, label="X1 = xTB + mech one-hot", color="#1565C0", edgecolor="black")
    ax2.set_xticks(x)
    ax2.set_xticklabels(protocols, fontsize=9)
    ax2.set_ylabel("MAE (%)")
    ax2.set_title("LOSO / LOMO Protocol MAE Comparison\n(XGBoost)", fontweight="bold")
    ax2.legend(fontsize=8)
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out = "fig5_loso_protocol.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 3 — LOSO ROOT CAUSE ANALYSIS (4 panels)
# ═══════════════════════════════════════════════════════════════════════════════
def make_fig3_loso_root_cause():
    """Four-panel: scatter / bias bar / box / heatmap."""

    with open('results/results_step7_improved_loso/loso_root_cause_analysis.json", encoding="utf-8") as f:
        rca = json.load(f)

    preds = pd.read_csv('results/results_step7_improved_loso/loso_predictions.csv",
                        encoding="utf-8-sig")
    preds["actual"]    = preds["actual"].astype(float)
    preds["predicted"] = preds["predicted"].astype(float)
    preds["error"]    = preds["error"].astype(float)

    per_sub = pd.DataFrame(rca["per_substrate"])
    substrates = sorted(preds["substrate"].unique(),
                       key=lambda s: per_sub[per_sub["substrate"] == s]["bias"].values[0]
                       if len(per_sub[per_sub["substrate"] == s]) else 0,
                       reverse=True)

    CHO = "Cyclohexene oxide"
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    # ── Panel A: Scatter ────────────────────────────────────────────────────
    ax = axes[0, 0]
    for sub in substrates:
        d = preds[preds["substrate"] == sub]
        ax.scatter(d["predicted"], d["actual"],
                   alpha=0.4, s=18, color=SUB_COLORS.get(sub, "gray"),
                   label=f"{SUB_SHORT[sub]} (n={len(d)})")
    ax.plot([0, 100], [0, 100], "k--", alpha=0.6, lw=1.5, label="y = x (perfect)")
    ax.set_xlim(0, 105); ax.set_ylim(0, 105)
    ax.set_xlabel("Predicted yield (%)", fontweight="bold")
    ax.set_ylabel("Actual yield (%)", fontweight="bold")
    ax.set_title("(A) LOSO Predicted vs Actual", fontweight="bold")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    ax.grid(alpha=0.3)
    r2_all = r2_score(preds["actual"], preds["predicted"])
    ax.text(0.97, 0.03,
            f"R${SUP2} = {r2_all:.4f}\nMAE = {mean_absolute_error(preds['actual'], preds['predicted']):.2f}%",
            transform=ax.transAxes, ha="right", va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="gray", alpha=0.9), fontsize=9)

    # ── Panel B: Bias bar ───────────────────────────────────────────────────
    ax = axes[0, 1]
    bias_data = []
    for sub in substrates:
        d = preds[preds["substrate"] == sub]
        bias = d["actual"].mean() - d["predicted"].mean()
        r2   = r2_score(d["actual"], d["predicted"])
        bias_data.append({
            "sub": SUB_SHORT[sub], "bias": bias,
            "r2": r2, "n": len(d),
            "is_cho": (sub == CHO)
        })
    x_pos = np.arange(len(bias_data))
    colors_bar = [C_CHO if b["is_cho"] else SUB_COLORS.get(
        [k for k, v in SUB_SHORT.items() if v == b["sub"]][0], C_OTH)
                  for b in bias_data]
    bars = ax.bar(x_pos, [b["bias"] for b in bias_data],
                  color=colors_bar, alpha=0.8, edgecolor="black")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([b["sub"] for b in bias_data], fontsize=9)
    ax.set_ylabel("Actual mean $-$ Predicted mean (%)", fontweight="bold")
    ax.set_title("(B) Per-Substrate Prediction Bias", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    for i, b in enumerate(bias_data):
        h = b["bias"]
        va = "bottom" if h >= 0 else "top"
        ax.text(i, h + (1 if h >= 0 else -1.5),
                f"{h:+.1f}%\nR${SUP2}={b['r2']:.2f}",
                ha="center", va=va, fontsize=8, fontweight="bold")
    ax.set_ylim(-50, 50)

    # ── Panel C: Box plot ───────────────────────────────────────────────────
    ax = axes[1, 0]
    box_d, positions, colors_list = [], [], []
    pos = 0
    for sub in substrates:
        d_actual    = preds[(preds["substrate"] == sub)]["actual"].values
        d_predicted = preds[(preds["substrate"] == sub)]["predicted"].values
        box_d.append(d_actual);    positions.append(pos); pos += 1
        box_d.append(d_predicted); positions.append(pos); pos += 2
        colors_list.extend([SUB_COLORS.get(sub, C_OTH),
                            SUB_COLORS.get(sub, C_OTH)])
    bp = ax.boxplot(box_d, positions=positions, widths=0.7,
                    patch_artist=True, showfliers=False)
    for pi, (patch, col) in enumerate(zip(bp["boxes"], colors_list)):
        patch.set_facecolor(col)
        patch.set_alpha(0.5 if pi % 2 == 0 else 0.9)
        if pi % 2 == 1:
            patch.set_hatch("//")
    sub_centers = [positions[2*i] + 0.5 for i in range(len(substrates))]
    ax.set_xticks(sub_centers)
    ax.set_xticklabels([SUB_SHORT[s] for s in substrates], fontsize=9)
    ax.set_ylabel("Yield (%)", fontweight="bold")
    ax.set_title("(C) Distribution: Actual (light) vs Predicted (hatched)", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    legend_elements = [
        mpatches.Patch(facecolor="gray", alpha=0.5, label="Actual"),
        mpatches.Patch(facecolor="gray", alpha=0.9, hatch="//", label="Predicted"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=8)
    ax.set_ylim(-5, 105)

    # ── Panel D: Mechanism × Substrate bias heatmap ──────────────────────────
    ax = axes[1, 1]
    pivot_err = preds.pivot_table(index="mechanism", columns="substrate",
                                  values="error", aggfunc="mean")
    pivot_n   = preds.pivot_table(index="mechanism", columns="substrate",
                                  values="error", aggfunc="count")

    sub_order = substrates
    mech_order = ["ionic_liquid", "metal_halide", "mixed_system", "other", "UNK"]
    pivot_err = pivot_err.reindex(index=mech_order, columns=sub_order)
    pivot_n   = pivot_n.reindex(index=mech_order, columns=sub_order, fill_value=0)

    vmax = max(abs(pivot_err.values.max()), abs(pivot_err.values.min())) * 1.2
    im = ax.imshow(pivot_err.values, cmap="RdBu_r", aspect="auto",
                   vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(sub_order)))
    ax.set_xticklabels([SUB_SHORT[s] for s in sub_order], fontsize=9)
    ax.set_yticks(range(len(mech_order)))
    ax.set_yticklabels(mech_order, fontsize=9)
    for i in range(len(mech_order)):
        for j in range(len(sub_order)):
            v = pivot_err.values[i, j]
            if not np.isnan(v):
                c = "white" if abs(v) > vmax * 0.5 else "black"
                ax.text(j, i, f"{v:+.0f}", ha="center", va="center",
                        color=c, fontsize=8)
            n = pivot_n.values[i, j]
            if not np.isnan(n) and n > 0:
                ax.text(j, i + 0.3, f"(n={int(n)})", ha="center", va="center",
                        color=c, fontsize=6.5)
    # Highlight CHO column
    cho_idx = sub_order.index(CHO) if CHO in sub_order else -1
    for i in range(len(mech_order)):
        rect = mpatches.Rectangle((cho_idx - 0.5, i - 0.5), 1, 1,
                                   fill=False, edgecolor=C_CHO, linewidth=2)
        ax.add_patch(rect)
    plt.colorbar(im, ax=ax, label="Bias: actual $-$ predicted (%)", shrink=0.9)
    ax.set_title("(D) Bias heatmap: substrate × mechanism", fontweight="bold")
    ax.set_xlabel("Substrate (held out)", fontweight="bold")

    fig.suptitle("LOSO Failure Root Cause Analysis", fontsize=13, fontweight="bold", y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out = "fig4_loso_root_cause.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 4 — 5×5 SUBSTRATE × MECHANISM TRANSFERABILITY MATRIX
# ═══════════════════════════════════════════════════════════════════════════════
def make_fig4_transferability_matrix():
    """Dual heatmap: n reactions (log) + mean yield."""

    cross = pd.read_csv("results/cross_tab_mech_substrate.csv", encoding="utf-8-sig")
    cross["count"] = pd.to_numeric(cross["count"], errors="coerce").fillna(0)
    cross["mean"]  = pd.to_numeric(cross["mean"],  errors="coerce").fillna(0)
    cross["std"]   = pd.to_numeric(cross["std"],   errors="coerce").fillna(0)

    substrates_order = ["Cyclohexene oxide", "Epichlorohydrin",
                        "Isopropyl glycidyl ether", "Propylene oxide", "Styrene oxide"]
    mechanisms_order = ["NUC", "LAC", "BAS", "BIF", "OTH"]

    mat_n    = cross.pivot(index="mech_label", columns="substrate", values="count")
    mat_mean = cross.pivot(index="mech_label", columns="substrate", values="mean")
    mat_std  = cross.pivot(index="mech_label", columns="substrate", values="std")
    mat_n    = mat_n.reindex(index=mechanisms_order,    columns=substrates_order, fill_value=0)
    mat_mean = mat_mean.reindex(index=mechanisms_order, columns=substrates_order)
    mat_std  = mat_std.reindex(index=mechanisms_order,  columns=substrates_order, fill_value=np.nan)

    CHO = "Cyclohexene oxide"

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # ── Panel (a): n reactions ────────────────────────────────────────────────
    ax = axes[0]
    n_vals = mat_n.values.astype(float)
    n_vals_safe = np.where(n_vals == 0, np.nan, n_vals)
    im0 = ax.imshow(n_vals_safe, cmap="Blues", aspect="auto",
                    norm=LogNorm(vmin=1, vmax=n_vals.max()))
    ax.set_xticks(range(len(substrates_order)))
    ax.set_xticklabels([SUB_SHORT[s] for s in substrates_order], fontsize=9)
    ax.set_yticks(range(len(mechanisms_order)))
    ax.set_yticklabels(mechanisms_order, fontsize=9)
    for i in range(len(mechanisms_order)):
        for j in range(len(substrates_order)):
            v = int(n_vals[i, j])
            if v > 0:
                ax.text(j, i, str(v), ha="center", va="center",
                        color="white" if v > 200 else "black", fontsize=8)
    # Highlight CHO column
    cho_j = substrates_order.index(CHO)
    for i in range(len(mechanisms_order)):
        rect = mpatches.Rectangle((cho_j - 0.5, i - 0.5), 1, 1,
                                   fill=False, edgecolor=C_CHO, linewidth=2)
        ax.add_patch(rect)
    plt.colorbar(im0, ax=ax, label="n reactions (log scale)", shrink=0.8)
    ax.set_title("(a) Sample coverage: n reactions\n(CHO column highlighted)", fontweight="bold")
    ax.set_xlabel("Substrate", fontweight="bold")
    ax.set_ylabel("Catalyst mechanism", fontweight="bold")

    # ── Panel (b): mean yield ────────────────────────────────────────────────
    ax = axes[1]
    m_vals = mat_mean.values.astype(float)
    im1 = ax.imshow(m_vals, cmap="RdYlGn", aspect="auto", vmin=0, vmax=100)
    ax.set_xticks(range(len(substrates_order)))
    ax.set_xticklabels([SUB_SHORT[s] for s in substrates_order], fontsize=9)
    ax.set_yticks(range(len(mechanisms_order)))
    ax.set_yticklabels(mechanisms_order, fontsize=9)
    for i in range(len(mechanisms_order)):
        for j in range(len(substrates_order)):
            v = m_vals[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.0f}", ha="center", va="center",
                        color="black", fontsize=8)
            s = mat_std.values[i, j]
            if not np.isnan(s):
                ax.text(j, i + 0.28, f"\u00b1{s:.0f}", ha="center",
                        va="center", color="gray", fontsize=6.5)
    # Highlight CHO column
    for i in range(len(mechanisms_order)):
        rect = mpatches.Rectangle((cho_j - 0.5, i - 0.5), 1, 1,
                                   fill=False, edgecolor=C_CHO, linewidth=2)
        ax.add_patch(rect)
    plt.colorbar(im1, ax=ax, label="Mean yield (%)", shrink=0.8)
    ax.set_title("(b) Mean yield (%)\n(CHO column highlighted)", fontweight="bold")
    ax.set_xlabel("Substrate", fontweight="bold")
    ax.set_ylabel("Catalyst mechanism", fontweight="bold")

    fig.suptitle("5" + chr(215) + "5 Substrate " + chr(215) + " Catalyst Yield Matrix", fontsize=12, fontweight="bold")
    plt.tight_layout()
    out = "fig6_transferability_matrix.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 5 — SHAP DIRECTION DIAGNOSTIC
# ═══════════════════════════════════════════════════════════════════════════════
def make_fig5_shap_direction():
    """Per-substrate signed SHAP bar chart, showing direction flips."""

    top_df = pd.read_csv('results/results_step4_5/per_substrate_top_features.csv",
                         encoding="utf-8-sig")
    # Parse rank list
    import ast
    top_df["rank"] = top_df["rank"].apply(
        lambda x: ast.literal_eval(x)[0] if isinstance(x, str) and x.startswith("[") else int(x)
    )
    top_df = top_df[top_df["rank"] <= 5].copy()
    top_df["mean_abs_shap"]   = top_df["mean_abs_shap"].astype(float)
    top_df["mean_signed_shap"] = top_df["mean_signed_shap"].astype(float)

    substrates = ["Cyclohexene oxide", "Epichlorohydrin",
                  "Isopropyl glycidyl ether", "Propylene oxide", "Styrene oxide"]
    substrates = [s for s in substrates if s in top_df["substrate"].values]

    # Pick top-5 features globally by mean |SHAP|
    top_feats_global = (top_df.groupby("feature")["mean_abs_shap"]
                        .mean().sort_values(ascending=False).head(8).index.tolist())

    CHO = "Cyclohexene oxide"

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ── Panel (a): Grouped bar — signed SHAP per substrate ──────────────────
    ax = axes[0]
    n_feats = len(top_feats_global)
    width = 0.14
    x = np.arange(n_feats)
    for i, sub in enumerate(substrates):
        sub_data = top_df[top_df["substrate"] == sub].set_index("feature")
        vals = [sub_data.loc[f, "mean_signed_shap"] if f in sub_data.index else 0
                for f in top_feats_global]
        color = C_CHO if sub == CHO else SUB_COLORS.get(sub, C_OTH)
        ax.bar(x + (i - 2) * width, vals, width, label=SUB_SHORT.get(sub, sub),
               color=color, alpha=0.85, edgecolor="black", linewidth=0.5)
    ax.axhline(0, color="black", linewidth=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(top_feats_global, rotation=30, ha="right", fontsize=8.5)
    ax.set_ylabel("Mean signed SHAP value", fontweight="bold")
    ax.set_title("(a) Signed SHAP per substrate\n(top-8 features)", fontweight="bold")
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.grid(axis="y", alpha=0.3)

    # ── Panel (b): Heatmap — signed SHAP ───────────────────────────────────
    ax2 = axes[1]
    plot_data = top_df[top_df["feature"].isin(top_feats_global)]
    pivot = plot_data.pivot(index="feature", columns="substrate",
                             values="mean_signed_shap")
    pivot = pivot.reindex(columns=substrates, fill_value=0)
    pivot = pivot.loc[top_feats_global]

    vmax = max(abs(pivot.values.max()), abs(pivot.values.min())) * 1.2
    im = ax2.imshow(pivot.values, cmap="RdBu_r", aspect="auto",
                    vmin=-max(vmax, 3), vmax=max(vmax, 3))
    ax2.set_xticks(range(len(substrates)))
    ax2.set_xticklabels([SUB_SHORT.get(s, s) for s in substrates], fontsize=9)
    ax2.set_yticks(range(len(top_feats_global)))
    ax2.set_yticklabels(top_feats_global, fontsize=8.5)
    for i in range(len(top_feats_global)):
        for j in range(len(substrates)):
            v = pivot.values[i, j]
            color = "white" if abs(v) > max(vmax, 3) * 0.6 else "black"
            ax2.text(j, i, f"{v:+.2f}", ha="center", va="center",
                     color=color, fontsize=7.5)
    plt.colorbar(im, ax=ax2, label="Mean signed SHAP", shrink=0.85)
    ax2.set_title("(b) SHAP direction heatmap\n(red = pushes yield up, blue = down)", fontweight="bold")
    ax2.set_xlabel("Substrate", fontweight="bold")

    fig.suptitle("Per-substrate SHAP Direction Diagnostic", fontsize=12, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out = "fig7_shap_direction.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("Regenerating all 5 paper figures with correct notation")
    print("=" * 60)
    make_fig1_graphical_abstract()
    make_fig2_loso_protocol()
    make_fig3_loso_root_cause()
    make_fig4_transferability_matrix()
    make_fig5_shap_direction()
    print("=" * 60)
    print("All figures saved.")


if __name__ == "__main__":
    main()
