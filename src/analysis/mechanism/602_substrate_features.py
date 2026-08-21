"""Step 2: Substrate steric + electronic clustering + DFT anchor validation.

Goal
----
Build a *mechanistic coordinate system* for the 5 epoxide substrates:

    Steric axis   : %VBur on epoxide O atom  (computed via RDKit + a
                    proxy -- solvent-accessible sphere count)
    Electronic    : epoxide-O LUMO energy (from DFT) and Mulliken q(O)

This puts substrates on a 2-D plane. Cross-referencing with the catalyst-
mechanism matrix from Step 1 reveals which (catalyst-mechanism, substrate-
mechanism) pairings are best explained.

Outputs
-------
results/substrate_features.csv                -- per-substrate features
results/substrate_features_summary.json       -- top-level summary
results/figs/substrate_mechanism_plane.png    -- 2-D map of 5 substrates
results/figs/dft_vs_xtb_anchor.png            -- DFT/xTB consistency plot
results/figs/transferability_3panel.png       -- mechanism-axis projection

Dependency policy
-----------------
* RDKit-only features (vbur_pct, c_o_c_angle, sub_heavy_atoms, sasa, qO_gasteiger)
  ALWAYS run. Outputs `substrate_features.csv` (canonical).
* DFT/xTB features (homo_eV, lumo_eV, gap, dipole, mulliken_q_O) are read from
  `dft_validation/dft_results_summary.csv` and
  `dft_validation/xtb_on_dft_geometry_nosolv.csv`.
  Those files are produced by tier_dft (510 + 512) which may not be available
  in every environment (needs WSL + ORCA). When missing, this script DEGRADES
  gracefully: it skips the DFT-anchor sections (4.2, 4.5, 4.6) and emits only
  the RDKit-derived outputs. This is consistent with the tier_dft_post design
  introduced in run_pipeline_v2.ps1 (Phase 2, 2026-08-20).
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

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors, rdFreeSASA

# Make src/ importable for paths.py
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = Path(__file__).resolve().parents[2]  # src/
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))
from paths import (  # noqa: E402
    PROJECT_ROOT,
    RESULTS_MECHANISM,
    RESULTS_TRANSFERABILITY,
    RESULTS_CHO_DIAGNOSTIC,
    DATA_PROCESSED,
    DRFP_XTB_EXTENDED_CSV,
    DFT_VALIDATION,
)


# -----------------------------------------------------------------------------
# Paths (all derived from paths.py — no hard-coded absolutes)
# -----------------------------------------------------------------------------
OUT_DIR = str(RESULTS_MECHANISM)
FIG_DIR = os.path.join(OUT_DIR, "figs")
SUBSTRATE_FEATURES_CSV = os.path.join(OUT_DIR, "substrate_features.csv")
SUBSTRATE_FEATURES_WITH_YIELD_CSV = os.path.join(OUT_DIR, "substrate_features_with_yield.csv")
SUBSTRATE_ELECTRONIC_CSV = os.path.join(OUT_DIR, "substrate_electronic.csv")
SUBSTRATE_FEATURES_SUMMARY = os.path.join(OUT_DIR, "substrate_features_summary.json")
TRANSFERABILITY_MATRIX_CSV = os.path.join(RESULTS_TRANSFERABILITY, "transferability_matrix.csv")
CROSS_TAB_MECH_SUBSTRATE_CSV = os.path.join(RESULTS_TRANSFERABILITY, "cross_tab_mech_substrate.csv")

# DFT dependency files (may be absent in non-DFT environments)
# FIX 2026-08-20: files live in DFT_VALIDATION / "results/" subfolder
DFT_RESULTS_CSV = os.path.join(DFT_VALIDATION, "results", "dft_results_summary.csv")
XTB_ON_DFT_CSV  = os.path.join(DFT_VALIDATION, "results", "xtb_on_dft_geometry_nosolv.csv")

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)


# -----------------------------------------------------------------------------
# Logger (replaces bare print() so that pipeline runs are tee-logged)
# -----------------------------------------------------------------------------
logger = logging.getLogger("602_substrate_features")
if not logger.handlers:
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(h)
    logger.setLevel(logging.INFO)
logger.propagate = False


# -----------------------------------------------------------------------------
# 1.  Substrate SMILES + raw names
# -----------------------------------------------------------------------------
SUBSTRATES = {
    "Styrene oxide":              "C1OC1c1ccccc1",
    "Epichlorohydrin":            "ClCC1CO1",
    "Propylene oxide":            "CC1CO1",
    "Cyclohexene oxide":          "O1C2CCCCC12",   # 1,2-epoxycyclohexane (NOT cyclopentanol)
    "Isopropyl glycidyl ether":   "CC(C)OCC1CO1",
    # Extra substrates we have DFT for (from dft_validation) but not in cleaned.csv
    "Epoxybutane":                "CCC1CO1",
    "Allyl glycidyl ether":       "C=CCOCC1CO1",
    "Phenyl glycidyl ether":      "O(c1ccccc1)CC1CO1",
    "Furfuryl glycidyl ether":    "O(c1ccco1)CC1CO1",
}


# -----------------------------------------------------------------------------
# 2.  Steric feature: %VBur around the epoxide oxygen
# -----------------------------------------------------------------------------
# Method
# ------
# 1. Embed the molecule in 3-D with ETKDG + MMFF94.
# 2. Locate the epoxide-O atom.
# 3. Generate a sphere of probe-points around that O at multiple radii.
# 4. %VBur = fraction of probes buried inside the van-der-Waals spheres of
#    neighbouring atoms.
# 5. Average over a few radii (3.0, 3.5, 4.0 Å) -- a coarse but robust proxy.
# This mirrors the spirit of the original %VBur (Cavallo & Poater) but is
# implementable without external dependencies.

VDW_RADII = {  # Å, Bondi 1964
    "H": 1.20, "C": 1.70, "N": 1.55, "O": 1.52, "F": 1.47,
    "P": 1.80, "S": 1.80, "Cl": 1.75, "Br": 1.85, "I": 1.98,
    "Si": 2.10, "B": 1.92,
}

PROBE_RADII = [3.0, 3.5, 4.0, 4.5]
PROBE_N = 3000  # Fibonacci sphere count


def fibonacci_sphere(n: int) -> np.ndarray:
    """Return n (n,3) unit vectors evenly distributed on the sphere."""
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    theta = np.pi * (1 + 5**0.5) * i
    x = np.cos(theta) * np.sin(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(phi)
    return np.stack([x, y, z], axis=1)


def find_epoxide_O(mol) -> int:
    """Find the epoxide oxygen: an O atom with exactly 2 heavy-atom neighbours
    forming a 3-membered ring."""
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 8:
            continue
        if atom.GetTotalNumHs() > 1:
            continue  # hydroxyl, not epoxide
        heavy_nbrs = [n for n in atom.GetNeighbors()
                      if n.GetAtomicNum() != 1]
        if len(heavy_nbrs) != 2:
            continue
        # are the two heavy nbrs bonded? (epoxide = 3-membered ring)
        a, b = heavy_nbrs
        if mol.GetBondBetweenAtoms(a.GetIdx(), b.GetIdx()) is not None:
            return atom.GetIdx()
    return -1


def find_epoxide_carbons(mol, o_idx: int):
    return [n.GetIdx() for n in mol.GetAtomWithIdx(o_idx).GetNeighbors()
            if n.GetAtomicNum() != 1]


def vbur_proxy(mol, o_idx: int, radii=PROBE_RADII, n_probes=PROBE_N) -> float:
    """%VBur on epoxide O averaged over probe radii.

    For each radius, count fraction of Fibonacci probes that fall inside any
    neighbour atom's vdW sphere.
    """
    if o_idx < 0:
        return np.nan
    conf = mol.GetConformer()
    o_pos = np.array([
        conf.GetAtomPosition(o_idx).x,
        conf.GetAtomPosition(o_idx).y,
        conf.GetAtomPosition(o_idx).z,
    ])
    nbr_pos = []
    nbr_rad = []
    for nbr in mol.GetAtomWithIdx(o_idx).GetNeighbors():
        if nbr.GetAtomicNum() == 1:
            continue
        p = conf.GetAtomPosition(nbr.GetIdx())
        nbr_pos.append(np.array([p.x, p.y, p.z]))
        nbr_rad.append(VDW_RADII.get(nbr.GetSymbol(), 1.7))

    nbr_pos = np.array(nbr_pos)
    nbr_rad = np.array(nbr_rad)

    # Exclude the epoxide O itself from "neighbours" for burial — but keep
    # the two epoxide carbons (they define the ring opening face).
    probes = fibonacci_sphere(n_probes)
    buried_per_r = []
    for r in radii:
        sample_pts = o_pos + r * probes
        # distance to each neighbour
        d = np.linalg.norm(sample_pts[:, None, :] - nbr_pos[None, :, :], axis=2)
        # buried if within any neighbour's vdW
        inside = (d < nbr_rad[None, :]).any(axis=1)
        buried_per_r.append(inside.mean() * 100)
    return float(np.mean(buried_per_r))


def embed_mol(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 0xC0FFEE
    if AllChem.EmbedMolecule(mol, params) != 0:
        # try with random coords
        params.useRandomCoords = True
        AllChem.EmbedMolecule(mol, params)
    try:
        AllChem.MMFFOptimizeMolecule(mol, maxIters=400)
    except Exception:
        try:
            AllChem.UFFOptimizeMolecule(mol, maxIters=400)
        except Exception:
            pass
    return mol


def substrate_features(smiles: str, name: str) -> dict:
    mol = embed_mol(smiles)
    if mol is None:
        return {"name": name, "smiles": smiles, "ok": False}
    o_idx = find_epoxide_O(mol)
    if o_idx < 0:
        return {"name": name, "smiles": smiles, "ok": False,
                "note": "no epoxide O found"}

    # Steric
    vbur = vbur_proxy(mol, o_idx)
    c_idxs = find_epoxide_carbons(mol, o_idx)
    conf = mol.GetConformer()

    # Angles around the epoxide O (sum should be ~308-310° for an epoxide)
    o_pos = np.array([conf.GetAtomPosition(o_idx).x,
                       conf.GetAtomPosition(o_idx).y,
                       conf.GetAtomPosition(o_idx).z])
    c_pos = np.array([[conf.GetAtomPosition(c).x,
                        conf.GetAtomPosition(c).y,
                        conf.GetAtomPosition(c).z] for c in c_idxs])
    oc1 = c_pos[0] - o_pos
    oc2 = c_pos[1] - o_pos
    cos_angle = np.dot(oc1, oc2) / (np.linalg.norm(oc1) * np.linalg.norm(oc2))
    cos_angle = np.clip(cos_angle, -1, 1)
    c_o_c_angle = float(np.degrees(np.arccos(cos_angle)))

    # Volume of substituent on the alpha carbon (the carbon "far" from O)
    # = number of heavy atoms NOT in the epoxide ring
    epoxide_ring_atoms = {o_idx, c_idxs[0], c_idxs[1]}
    sub_heavy_atoms = sum(1 for a in mol.GetAtoms()
                           if a.GetAtomicNum() != 1
                           and a.GetIdx() not in epoxide_ring_atoms)

    # Approximate SASA on O (solvent-accessible)
    radii = rdFreeSASA.classifyAtoms(mol)
    sasa_o = rdFreeSASA.CalcSASA(mol, radii)

    # Gasteiger charge on O (electronic proxy for nucleophile-targetability)
    AllChem.ComputeGasteigerCharges(mol)
    o_atom = mol.GetAtomWithIdx(o_idx)
    qO_gasteiger = float(o_atom.GetDoubleProp("_GasteigerCharge"))

    # Atomic-number-weighted sum of substituents as crude Hammett-like proxy
    ring_substituent_composition = "".join(
        sorted({a.GetSymbol() for a in mol.GetAtoms()
                if a.GetAtomicNum() != 1
                and a.GetIdx() not in epoxide_ring_atoms})
    )

    return {
        "name": name,
        "smiles": smiles,
        "ok": True,
        "vbur_pct": round(vbur, 2),
        "c_o_c_angle_deg": round(c_o_c_angle, 1),
        "sub_heavy_atoms": int(sub_heavy_atoms),
        "substituent_composition": ring_substituent_composition,
        "sasa_A2": round(sasa_o, 3),
        "qO_gasteiger": round(qO_gasteiger, 4),
    }


# -----------------------------------------------------------------------------
# 3.  Electronic features from DFT (cross-check against RDKit)
# -----------------------------------------------------------------------------
def dft_features() -> pd.DataFrame | None:
    if not os.path.exists(DFT_RESULTS_CSV):
        logger.warning("[DFT] %s not found — skipping DFT anchor features", DFT_RESULTS_CSV)
        return None
    df = pd.read_csv(DFT_RESULTS_CSV)

    def role_of(name):
        n = name.lower()
        if "carbonate_product" in n: return "product"
        subs = ["propylene_oxide", "styrene_oxide", "cyclohexene_oxide",
                "epichlorohydrin", "epoxybutane", "allyl_glycidyl_ether",
                "furfuryl_glycidyl_ether", "phenyl_glycidyl_ether",
                "isopropyl_glycidyl_ether"]
        for s in subs:
            if s in n: return s
        if "co2" in n: return "CO2"
        return "other"

    df["substrate"] = df["file"].apply(role_of)
    sub_df = df[df["substrate"].str.contains("oxide") | df["substrate"].str.contains("glycidyl") | df["substrate"].isin(["epichlorohydrin", "epoxybutane"])].copy()
    sub_df["name"] = sub_df["file"].str.replace(r"\.out$", "", regex=True)
    sub_df = sub_df[["name", "substrate", "homo_eV", "lumo_eV", "gap_eV",
                     "dipole_debye"]].copy()
    return sub_df


def xtb_features() -> pd.DataFrame | None:
    if not os.path.exists(XTB_ON_DFT_CSV):
        logger.warning("[xTB] %s not found — skipping xTB-on-DFT-geometry features", XTB_ON_DFT_CSV)
        return None
    df = pd.read_csv(XTB_ON_DFT_CSV)
    sub_df = df[df["role"] == "substrate"].copy()
    return sub_df[["name", "homo_eV", "lumo_eV", "gap_eV",
                   "dipole_D", "mulliken_q_O"]].rename(columns={
        "homo_eV": "xtb_homo_eV",
        "lumo_eV": "xtb_lumo_eV",
        "gap_eV": "xtb_gap_eV",
        "dipole_D": "xtb_dipole",
        "mulliken_q_O": "xtb_qO",
    })


# -----------------------------------------------------------------------------
# 4.  Driver
# -----------------------------------------------------------------------------
def main():
    # 4.1  RDKit steric + electronic for the 5 main substrates (and 4 extras)
    #       ALWAYS runs — does not depend on tier_dft.
    rows = []
    for name, smi in SUBSTRATES.items():
        rows.append(substrate_features(smi, name))
    feat_df = pd.DataFrame(rows)
    feat_df.to_csv(SUBSTRATE_FEATURES_CSV, index=False)
    logger.info("=== RDKit-derived substrate features (%d substrates) ===", len(feat_df))
    logger.info("\n%s", feat_df.to_string())

    # 4.1b  Yield-joined substrate features (consumed by 702 integrated report)
    # Prefer the master 87-col merged table (DRFP_XTB_EXTENDED_CSV).
    # Fall back to the legacy co2_drfp_xtb.csv only if the master is absent
    # (legacy behaviour for environments that have not run 107_merge_substrate_xtb.py).
    master_csv = str(DRFP_XTB_EXTENDED_CSV)
    legacy_csv = str(DATA_PROCESSED / "co2_drfp_xtb.csv")
    cleaned_csv = master_csv if os.path.exists(master_csv) else legacy_csv
    if not os.path.exists(cleaned_csv):
        logger.warning("Neither %s nor %s exists; substrate_features_with_yield.csv will be skipped",
                       master_csv, legacy_csv)
        cleaned_csv = None
    try:
        if cleaned_csv is not None:
            cleaned = pd.read_csv(cleaned_csv)
            yield_per_sub = cleaned.groupby("reactant_name")["yield (%)"].agg(
                ["mean", "median", "std", "count"]
            ).reset_index().rename(columns={
                "mean": "yield_mean",
                "median": "yield_median",
                "std": "yield_std",
                "count": "yield_n",
            })
            with_yield = feat_df.merge(yield_per_sub, left_on="name", right_on="reactant_name", how="left")
            if "reactant_name" in with_yield.columns:
                with_yield = with_yield.drop(columns=["reactant_name"])
            with_yield.to_csv(SUBSTRATE_FEATURES_WITH_YIELD_CSV, index=False)
            logger.info("substrate_features_with_yield.csv saved (%d rows, source=%s)",
                        len(with_yield), cleaned_csv)
    except Exception as e:
        logger.warning("yield-join failed (%s); skipping substrate_features_with_yield.csv", e)

    # 4.2  DFT + xTB anchor (may be absent in non-DFT environments)
    dft_df = dft_features()
    xtb_df = xtb_features()
    has_dft = dft_df is not None or xtb_df is not None
    if has_dft:
        # Outer-merge so missing columns become NaN (we still want to plot whatever is present)
        if dft_df is None:
            dft_df = pd.DataFrame(columns=["name", "substrate", "homo_eV", "lumo_eV",
                                            "gap_eV", "dipole_debye", "mulliken_charge_mean_O"])
        if xtb_df is None:
            xtb_df = pd.DataFrame(columns=["name", "xtb_homo_eV", "xtb_lumo_eV",
                                            "xtb_gap_eV", "xtb_dipole", "xtb_qO"])
        merged = dft_df.merge(xtb_df, left_on="name", right_on="name", how="outer")
        merged.to_csv(SUBSTRATE_ELECTRONIC_CSV, index=False)
        logger.info("substrate_electronic.csv saved (%d rows)", len(merged))
    else:
        logger.warning("DFT + xTB inputs missing; substrate_electronic.csv will not be written")

    # 4.3  Summary
    summary = {
        "n_substrates": int(len(feat_df)),
        "vbur_range_pct": [float(feat_df["vbur_pct"].min()),
                            float(feat_df["vbur_pct"].max())],
        "sub_heavy_range": [int(feat_df["sub_heavy_atoms"].min()),
                             int(feat_df["sub_heavy_atoms"].max())],
    }
    if dft_df is not None and "lumo_eV" in dft_df.columns and not dft_df["lumo_eV"].isna().all():
        summary["dft_lumo_range_eV"] = [float(dft_df["lumo_eV"].min()),
                                          float(dft_df["lumo_eV"].max())]
    with open(SUBSTRATE_FEATURES_SUMMARY, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # 4.4  Figure: 2-D plane (steric vs electronic) — always emitted (RDKit only)
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    sub_main = feat_df[feat_df["name"].isin(SUBSTRATES.keys()) & feat_df["ok"]].copy()
    ax.scatter(sub_main["vbur_pct"], -sub_main["qO_gasteiger"],
               s=120, c="steelblue", edgecolors="black", zorder=3)
    for _, row in sub_main.iterrows():
        ax.annotate(row["name"],
                     (row["vbur_pct"], -row["qO_gasteiger"]),
                     xytext=(7, 7), textcoords="offset points", fontsize=9)
    ax.set_xlabel("%VBur on epoxide O (proxy, average over 3.0–4.5 Å)")
    ax.set_ylabel("-q(O) Gasteiger  (more negative = more nucleophilic O)")
    ax.set_title("Substrate mechanism coordinates\n(steric × electronic)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig_path = os.path.join(FIG_DIR, "substrate_mechanism_plane.png")
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved fig: %s", fig_path)

    # 4.5  DFT vs xTB consistency plot — SKIP if DFT inputs absent
    if not has_dft:
        logger.warning("[skip] dft_vs_xtb_anchor.png (no DFT inputs)")
    else:
        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        pairs = [("homo_eV", "xtb_homo_eV", "HOMO"),
                 ("lumo_eV", "xtb_lumo_eV", "LUMO"),
                 ("dipole_debye", "xtb_dipole", "dipole (Debye)")]
        for ax, (a, b, lbl) in zip(axes, pairs):
            if a not in merged.columns or b not in merged.columns:
                continue
            ax.scatter(merged[a], merged[b])
            for _, r in merged.iterrows():
                if pd.notna(r[a]) and pd.notna(r[b]):
                    ax.annotate(r["name"], (r[a], r[b]),
                                 xytext=(4, 4), textcoords="offset points",
                                 fontsize=7)
            ax.set_xlabel(f"DFT {lbl}")
            ax.set_ylabel(f"xTB {lbl}")
            ax.set_title(f"DFT vs xTB -- {lbl}")
            ax.grid(alpha=0.3)
        fig.suptitle("DFT / xTB consistency on the 9 epoxide geometries", y=1.02)
        fig.tight_layout()
        fig_path2 = os.path.join(FIG_DIR, "dft_vs_xtb_anchor.png")
        fig.savefig(fig_path2, dpi=200, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved fig: %s", fig_path2)

    # 4.6  Combined "transferability 3-panel" — SKIP if 603 outputs are not yet produced
    if not (os.path.exists(TRANSFERABILITY_MATRIX_CSV) and os.path.exists(CROSS_TAB_MECH_SUBSTRATE_CSV)):
        logger.warning("[skip] transferability_3panel.png (603 outputs not present yet)")
    else:
        n_mat = pd.read_csv(TRANSFERABILITY_MATRIX_CSV, index_col=0)
        mean_mat = pd.read_csv(CROSS_TAB_MECH_SUBSTRATE_CSV)
        mean_pivot = mean_mat.pivot(index="mech_label", columns="substrate", values="mean")

        sub_main = feat_df[feat_df["name"].isin(SUBSTRATES.keys()) & feat_df["ok"]].set_index("name")
        mech_colors = {"NUC": "tab:blue", "LAC": "tab:orange",
                       "BAS": "tab:green", "BIF": "tab:purple", "OTH": "tab:gray"}

        fig, ax = plt.subplots(figsize=(8, 6))
        for s in n_mat.columns:
            if s not in sub_main.index:
                continue
            x = sub_main.loc[s, "vbur_pct"]
            y = -sub_main.loc[s, "qO_gasteiger"]
            ax.scatter(x, y, s=400, c="lightgray", edgecolors="black", zorder=1)
            ax.annotate(s.replace(" oxide", "").replace(" glycidyl ether", "-GGE"),
                         (x, y), ha="center", va="center", fontsize=7, zorder=2)
        mech_centers = {"NUC": (-0.05, 0.95), "LAC": (1.05, 0.95),
                        "BAS": (-0.05, 0.05), "BIF": (1.05, 0.05)}
        for mech in ["NUC", "LAC", "BAS", "BIF"]:
            if mech not in n_mat.index:
                continue
            counts = n_mat.loc[mech]
            ax.text(0.02, mech_centers[mech][1],
                    f"{mech}: " + ", ".join(
                        [f"{s.replace(' oxide','').replace(' glycidyl ether','-GGE')}={int(c)}"
                         for s, c in counts.items()]),
                    transform=ax.transAxes,
                    fontsize=7, color=mech_colors[mech])
        ax.set_xlabel("%VBur on epoxide O")
        ax.set_ylabel("-q(O) Gasteiger")
        ax.set_title("Substrate plane + catalyst-mechanism coverage")
        fig.tight_layout()
        fig_path3 = os.path.join(FIG_DIR, "transferability_3panel.png")
        fig.savefig(fig_path3, dpi=200, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved fig: %s", fig_path3)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="602 substrate steric/electronic features.")
    parser.add_argument("--force", action="store_true",
                        help="Pipeline-compatibility flag; this script always fully regenerates its outputs.")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging.")
    args = parser.parse_args()
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    main()
    sys.exit(0)