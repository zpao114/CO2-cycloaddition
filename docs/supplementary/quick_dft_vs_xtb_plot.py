#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
quick_dft_vs_xtb_plot.py
=========================
Quick (~30 sec) supplementary figure for §S6:
  - xTB vs DFT (B3LYP-D3BJ/def2-TZVP) for HOMO, LUMO, Gap, Dipole
  - Pearson R and Spearman rho computed and shown on each panel
  - Subset of 7 epoxide substrates highlighted in red
"""
import os
import csv
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = r"D:\machine-learning\CO2 cycloaddition"
DFT_CSV  = os.path.join(ROOT, "dft_validation", "dft_results_summary.csv")
XTB_CSV  = os.path.join(ROOT, "dft_validation", "xtb_on_dft_geometry_nosolv.csv")
REPORT_TXT = os.path.join(ROOT, "dft_validation", "514_dft_vs_xtb_report.txt")
OUT_PNG = os.path.join(ROOT, "fig_s6_dft_vs_xtb.png")
OUT_PDF = os.path.join(ROOT, "fig_s6_dft_vs_xtb.pdf")

# ---------- helpers ----------
def _safe_float(s):
    try:
        v = float(s)
        if math.isnan(v):
            return None
        return v
    except (ValueError, TypeError):
        return None

def _rank(xs):
    n = len(xs)
    idx = sorted(range(n), key=lambda i: xs[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and xs[idx[j + 1]] == xs[idx[i]]:
            j += 1
        r = (i + j + 2) / 2.0
        for k in range(i, j + 1):
            ranks[idx[k]] = r
        i = j + 1
    return ranks

def corr(x, y):
    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    num = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    denx = math.sqrt(sum((xi - mx) ** 2 for xi in x))
    deny = math.sqrt(sum((yi - my) ** 2 for yi in y))
    pearson = num / (denx * deny) if denx * deny != 0 else float("nan")
    rx, ry = _rank(x), _rank(y)
    mrx, mry = sum(rx) / n, sum(ry) / n
    num_r = sum((xi - mrx) * (yi - mry) for xi, yi in zip(rx, ry))
    denr_x = math.sqrt(sum((xi - mrx) ** 2 for xi in rx))
    denr_y = math.sqrt(sum((yi - mry) ** 2 for yi in ry))
    spearman = num_r / (denr_x * denr_y) if denr_x * denr_y != 0 else float("nan")
    return pearson, spearman

# ---------- load ----------
with open(DFT_CSV, "r", encoding="utf-8-sig", newline="") as f:
    dft_rows = list(csv.DictReader(f))
with open(XTB_CSV, "r", encoding="utf-8-sig", newline="") as f:
    xtb_rows = list(csv.DictReader(f))

SUBSTRATE_KEYWORDS = [
    "propylene_oxide", "styrene_oxide", "cyclohexene_oxide",
    "epichlorohydrin", "epoxybutane", "allyl_glycidyl_ether",
    "furfuryl_glycidyl_ether", "phenyl_glycidyl_ether",
    "isopropyl_glycidyl_ether",
]

def role_of(filename):
    n = filename.lower()
    for s in SUBSTRATE_KEYWORDS:
        if s in n:
            return "substrate"
    if "co2" in n:
        return "reactant"
    if "carbonate" in n:
        return "product"
    if "tbai" in n or "znbr" in n or "dbu" in n:
        return "catalyst"
    if any(x in n for x in ["dmf", "dmso", "ethanol", "methanol"]):
        return "solvent"
    return "other"

xtb_by_name = {r["name"].replace(".out", ""): r for r in xtb_rows}

# Collect pairs (only keep non-outlier TBAI_anion entries)
pairs = []
for dft in dft_rows:
    fn = dft["file"].replace(".out", "")
    if fn not in xtb_by_name:
        continue
    role = role_of(fn)
    pairs.append({
        "name": fn,
        "role": role,
        "xtb_homo": _safe_float(xtb_by_name[fn].get("homo_eV")),
        "dft_homo": _safe_float(dft.get("homo_eV")),
        "xtb_lumo": _safe_float(xtb_by_name[fn].get("lumo_eV")),
        "dft_lumo": _safe_float(dft.get("lumo_eV")),
        "xtb_gap":  _safe_float(xtb_by_name[fn].get("gap_eV")),
        "dft_gap":  _safe_float(dft.get("gap_eV")),
        "xtb_dip":  _safe_float(xtb_by_name[fn].get("dipole_D")),
        "dft_dip":  _safe_float(dft.get("dipole_debye")),
    })

print(f"Pairs loaded: {len(pairs)}")
for p in pairs:
    print(f"  {p['role']:10s}  {p['name']:40s}  "
          f"HOMO xTB={p['xtb_homo']}  DFT={p['dft_homo']}")

# ---------- plot ----------
fig, axes = plt.subplots(2, 2, figsize=(11, 10))
panels = [
    ("HOMO (eV)", "xtb_homo", "dft_homo", axes[0, 0]),
    ("LUMO (eV)", "xtb_lumo", "dft_lumo", axes[0, 1]),
    ("Gap (eV)",  "xtb_gap",  "dft_gap",  axes[1, 0]),
    ("Dipole (Debye)", "xtb_dip", "dft_dip", axes[1, 1]),
]

color_map = {"substrate": "#e74c3c", "catalyst": "#2980b9",
             "solvent": "#27ae60", "reactant": "#8e44ad",
             "product": "#f39c12", "other": "#7f8c8d"}

for title, xk, yk, ax in panels:
    xs, ys, names, roles = [], [], [], []
    for p in pairs:
        x, y = p[xk], p[yk]
        if x is None or y is None:
            continue
        # drop the obvious outlier (TBAI_anion.inp_atom53, gap 29.8 eV)
        if abs(y - x) > 12 and "tbai" in p["name"].lower():
            print(f"  [skip outlier] {p['name']}  |DFT-xTB|={abs(y-x):.1f} eV")
            continue
        xs.append(x); ys.append(y)
        names.append(p["name"]); roles.append(p["role"])
    if len(xs) < 2:
        ax.set_title(f"{title}: insufficient data", fontsize=11)
        continue
    pearson, spearman = corr(xs, ys)
    for x, y, n, r in zip(xs, ys, names, roles):
        ax.scatter(x, y, c=color_map.get(r, "#7f8c8d"),
                   s=70, edgecolors="black", linewidths=0.5, zorder=3, alpha=0.85)
        ax.annotate(n.replace("_", " "), (x, y),
                    xytext=(4, 4), textcoords="offset points", fontsize=6.5)
    # y=x reference
    lo = min(min(xs), min(ys)) - 0.5
    hi = max(max(xs), max(ys)) + 0.5
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=0.8, alpha=0.5, zorder=1)
    ax.set_xlabel("xTB (eV or D)", fontsize=10)
    ax.set_ylabel("DFT (eV or D)", fontsize=10)
    ax.set_title(f"{title}\nR = {pearson:+.3f}  Spearman ρ = {spearman:+.3f}  (N={len(xs)})",
                 fontsize=10.5)
    ax.grid(alpha=0.3)

# legend
from matplotlib.patches import Patch
legend_handles = [Patch(facecolor=c, edgecolor="black", label=r)
                  for r, c in color_map.items()]
fig.legend(handles=legend_handles, loc="lower center",
           ncol=6, fontsize=9, bbox_to_anchor=(0.5, -0.01))

fig.suptitle("GFN2-xTB vs B3LYP-D3BJ/def2-TZVP on 18-molecule calibration set\n"
             "(apples-to-apples: xTB evaluated on DFT-optimized geometries)",
             fontsize=12, y=1.00)
fig.tight_layout(rect=(0, 0.04, 1, 0.97))
fig.savefig(OUT_PNG, dpi=180, bbox_inches="tight")
fig.savefig(OUT_PDF, bbox_inches="tight")
print(f"\nSaved: {OUT_PNG}")
print(f"Saved: {OUT_PDF}")

# ---------- write summary table to console ----------
print("\n=== xTB vs DFT summary (N=18 minus 1 outlier) ===")
for title, xk, yk, _ in panels:
    xs, ys = [], []
    for p in pairs:
        x, y = p[xk], p[yk]
        if x is None or y is None:
            continue
        if abs(y - x) > 12 and "tbai" in p["name"].lower():
            continue
        xs.append(x); ys.append(y)
    if len(xs) < 2:
        continue
    pearson, spearman = corr(xs, ys)
    deltas = [y - x for x, y in zip(xs, ys)]
    mae = sum(abs(d) for d in deltas) / len(deltas)
    print(f"  {title:18s}  N={len(xs):2d}  MAE={mae:6.3f}  "
          f"R={pearson:+.3f}  Spearman={spearman:+.3f}")
