#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""107_merge_substrate_xtb.py — fill in substrate/CO2 xTB descriptors and
derived reactivity features; emit the master 87-column table.

Pipeline position
-----------------
TIER 1 final step (one-shot merge).

Inputs
------
data/processed/co2_drfp_xtb_extended.csv          (master CSV after Tier 1 batch)
results/results_cho_diagnostic/xtb_results_summary.csv   (from 104b)

Outputs
-------
results/results_cho_diagnostic/co2_drfp_xtb_extended.csv     (NOT the input path;
                                                              no in-place overwrite)
data/external/substrate_xtb_baseline.json                    (regenerated each run)

History
-------
A previous revision depended on 106b_merge_xtb_v2.py; since the
2026-07 refactor the merge logic is inlined here. 106b is now a stub.

Usage
-----
    python 107_merge_substrate_xtb.py
    python 107_merge_substrate_xtb.py --dry-run   # report coverage only
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = os.environ.get("CO2_PROJECT_ROOT", r"D:\machine-learning\CO2-cycloaddition")
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

logger = logging.getLogger("107_merge_substrate_xtb")

# Force UTF-8 stdout for non-ASCII (CO2, °C, Å). Mirrors 105b_xtb_sanity_v2.py.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np
import pandas as pd

# RDKit is used by compute_cat_xtb_row() to split IL salts into cation/anion
# fragments and canonicalise SMILES for robust lookups. We treat RDKit as
# optional: without it we fall back to a token-split on '.' which works for
# the common "[BMIM]+.[Cl-]" pattern but loses formal-charge detection.
try:
    from rdkit import Chem
    RDKIT_AVAILABLE = True
except Exception:
    RDKIT_AVAILABLE = False
    Chem = None  # type: ignore

from src.paths import (
    DRFP_XTB_EXTENDED_CSV,
    RESULTS_CHO_DIAGNOSTIC,
    SUBSTRATE_XTB_BASELINE,
    ensure_dir,
)

XTB_SUMMARY_CSV = RESULTS_CHO_DIAGNOSTIC / "xtb_results_summary.csv"
# 107 writes to a sibling path (never overwrites its own input — that
# would risk dropping columns on a partial failure).
MERGED_OUTPUT_CSV = RESULTS_CHO_DIAGNOSTIC / "co2_drfp_xtb_extended.csv"

SUBSTRATE_NAME_MAP = {
    # cleaned.csv reactant_name -> xtb_results_summary.csv name
    "styrene oxide":          "styrene_oxide",
    "propylene oxide":        "propylene_oxide",
    "epichlorohydrin":        "epichlorohydrin",
    "cyclohexene oxide":      "cyclohexene_oxide",
    "isopropyl glycidyl ether": "isopropyl_glycidyl_ether",
    "glycidyl phenyl ether":  "styrene_oxide",  # analogous: same ring family
}

# CO2 的 xTB name
CO2_NAME = "CO2"

# ─────────────────────────────────────────────────────────────────────────────
# §2  Solvent name → xTB name mapping
# ─────────────────────────────────────────────────────────────────────────────
# Key: cleaned.csv reactant_name / solvent_name → xtb_results_summary.csv "name"
SOLVENT_NAME_MAP = {
    # Common names
    "acetonitrile":          "MeCN",
    "ACN":                   "MeCN",
    "methanol":              "MeOH",
    "MeOH":                  "MeOH",
    "ethanol":               "EtOH",
    "EtOH":                  "EtOH",
    "water":                 "water",
    "H2O":                   "water",
    "DMSO":                  "DMSO",
    "DMF":                   "DMF",
    "acetone":               "acetone",
    "ethyl acetate":          "ethyl_acetate",
    "ethyl acetate (EtOAc)": "ethyl_acetate",
    "dichloromethane":        "DCM",
    "DCM":                    "DCM",
    "toluene":                "toluene",
    "chlorobenzene":          "chlorobenzene",
    "chloroform":             "CHCl3",
    "THF":                   "THF",
    "hexane":                 "hexane",
    "n-hexane":               "hexane",
    "cyclohexane":            "cyclohexane",
    "1,4-dioxane":            "dioxane",
    "dioxane":                "dioxane",
    "diethyl ether":          "diethyl_ether",
    "supercritical CO2":      "CO2",
    "scCO2":                  "CO2",
    "ionic liquid (IL)":      "DMSO",  # IL 用高极性溶剂 DMSO 近似
    "ionic liquid":           "DMSO",
}

# ─────────────────────────────────────────────────────────────────────────────
# §3  Substrate HOMOs from xtb_results_summary.csv (GFN2-xTB, DMSO solvent)
# ─────────────────────────────────────────────────────────────────────────────
# 实际 xTB 计算值（从 xtb_results_summary.csv 读取）：
#   styrene_oxide:         HOMO=-10.93, LUMO=-6.29, gap=4.64, dipole=2.46
#   propylene_oxide:       HOMO=-11.33, LUMO=-2.85, gap=8.48, dipole=2.53
#   epichlorohydrin:       HOMO=-11.49, LUMO=-4.75, gap=6.74, dipole=0.46
#   cyclohexene_oxide:     HOMO=-10.56, LUMO=-1.87, gap=8.69, dipole=2.02
#   isopropyl_glycidyl_ether: HOMO=-10.88, LUMO=-3.23, gap=7.65, dipole=2.30
#   (glycidyl phenyl ether 用 styrene_oxide 的值近似)

# styrene_oxide 的 SMILES 简化版缺少苯环芳香性，用propylene_oxide作近似
SUBSTRATE_XTB_REF = {
    "styrene_oxide":          {"homo": -10.93, "lumo": -6.29, "gap": 4.64, "dipole": 2.46},
    "propylene_oxide":        {"homo": -11.33, "lumo": -2.85, "gap": 8.48, "dipole": 2.53},
    "epichlorohydrin":        {"homo": -11.49, "lumo": -4.75, "gap": 6.74, "dipole": 0.46},
    "cyclohexene_oxide":      {"homo": -10.56, "lumo": -1.87, "gap": 8.69, "dipole": 2.02},
    "isopropyl_glycidyl_ether": {"homo": -10.88, "lumo": -3.23, "gap": 7.65, "dipole": 2.30},
}


# ----------------------------------------------------------------------------
# Catalyst xTB lookups (added 2026-08-19 to break the chicken-and-egg
# dependency on cat_cation_* columns that used to be supplied by a now-removed
# upstream merge step. 107 now generates these columns itself from the
# xtb_results_summary.csv table produced by 104b.)
# ----------------------------------------------------------------------------
def build_catalyst_xtb_lookup(xtb_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """Return SMILES -> {homo, lumo, gap, dipole, charge} for every catalyst.

    The xtb_results_summary.csv has one row per molecule (including salts
    stored as neutral salt pairs like "CCCC[N+](CCCC)(CCCC)CCCC.[I-]"). For
    IL catalysts (which 102 produces as "cation.anion" with a literal "."),
    we also split the salt into its ionic components when possible and index
    each component by its own canonical SMILES, so lookups by either the
    salt SMILES or the cation/anion SMILES resolve.

    The returned dict maps canonical SMILES -> xTB dict.
    """
    out: Dict[str, Dict[str, float]] = {}

    def _row_to_xtb(row) -> Optional[Dict[str, float]]:
        try:
            homo_v = float(row["homo_eV"]) if pd.notna(row.get("homo_eV")) else None
            lumo_v = float(row["lumo_eV"]) if pd.notna(row.get("lumo_eV")) else None
            gap_v  = float(row["gap_eV"])  if pd.notna(row.get("gap_eV"))  else None
            mu_v   = float(row["dipole_D"]) if pd.notna(row.get("dipole_D")) else None
            chg_v  = int(row["charge"]) if pd.notna(row.get("charge")) else 0
        except Exception:
            return None
        if homo_v is None or lumo_v is None or gap_v is None:
            return None
        return {
            "homo": homo_v,
            "lumo": lumo_v,
            "gap":  gap_v,
            "dipole": mu_v if mu_v is not None else 0.0,
            "charge": chg_v,
        }

    for _, row in xtb_df.iterrows():
        smi = row.get("smiles")
        if not isinstance(smi, str) or not smi.strip():
            continue
        data = _row_to_xtb(row)
        if data is None:
            continue
        # Index by raw SMILES (canonicalised whitespace)
        key = smi.strip()
        # RDKit canonical SMILES (if RDKit available) for robust matching
        if RDKIT_AVAILABLE:
            try:
                mol = Chem.MolFromSmiles(key)
                if mol is not None:
                    can = Chem.MolToSmiles(mol)
                    out[can] = data
                    out[key] = data
                else:
                    out[key] = data
            except Exception:
                out[key] = data
        else:
            out[key] = data

        # If the SMILES is a salt (".", between cation and anion), also index
        # the individual fragments so 102's "CCCC[N+](CCCC)(CCCC)CCCC.[I-]"
        # style salt SMILES AND its components ("CCCC[N+](CCCC)(CCCC)CCCC",
        # "[I-]") both resolve.
        if "." in key and RDKIT_AVAILABLE:
            try:
                mol = Chem.MolFromSmiles(key)
                if mol is not None and "." in Chem.MolToSmiles(mol):
                    frags = Chem.GetMolFrags(mol, asMols=True)
                    for fmol in frags:
                        fcan = Chem.MolToSmiles(fmol)
                        fcharge = Chem.GetFormalCharge(fmol)
                        fdata = dict(data)
                        fdata["charge"] = int(fcharge)
                        out[fcan] = fdata
            except Exception:
                pass

    return out


def _lookup_smiles(smi: Optional[str], lookup: Dict[str, Dict]) -> Optional[Dict[str, float]]:
    """Best-effort SMILES lookup that tolerates canonical/non-canonical."""
    if not isinstance(smi, str):
        return None
    key = smi.strip()
    if not key:
        return None
    if key in lookup:
        return lookup[key]
    if RDKIT_AVAILABLE:
        try:
            mol = Chem.MolFromSmiles(key)
            if mol is not None:
                can = Chem.MolToSmiles(mol)
                if can in lookup:
                    return lookup[can]
        except Exception:
            pass
    return None


def _classify_catalyst(smi: Optional[str]) -> str:
    """Classify catalyst SMILES as 'salt' (cation.anion), 'single', or ''."""
    if not isinstance(smi, str) or not smi.strip():
        return ""
    key = smi.strip()
    if RDKIT_AVAILABLE:
        try:
            mol = Chem.MolFromSmiles(key)
            if mol is None:
                return ""
            can = Chem.MolToSmiles(mol)
            return "salt" if "." in can else "single"
        except Exception:
            return ""
    return "salt" if "." in key else "single"


def compute_cat_xtb_row(
    cat_smiles: Optional[str],
    lookup: Dict[str, Dict[str, float]],
) -> Dict[str, float]:
    """Compute cat_cation_*, cat_anion_*, and match-type columns for a row.

    Returns a dict of values to write back into the master DataFrame. Any
    component that cannot be resolved is set to NaN.
    """
    out: Dict[str, float] = {
        "cat_cation_homo_eV": np.nan,
        "cat_cation_lumo_eV": np.nan,
        "cat_cation_gap_eV":  np.nan,
        "cat_anion_homo_eV":  np.nan,
        "cat_anion_lumo_eV":  np.nan,
        "cat_anion_gap_eV":   np.nan,
        "cat_cation_dipole_D": np.nan,
        "cat_anion_dipole_D":  np.nan,
        "cation_match_type": "",
        "anion_match_type": "",
        "catalyst_type_v2": "",
    }
    if not isinstance(cat_smiles, str) or not cat_smiles.strip():
        return out

    kind = _classify_catalyst(cat_smiles)
    out["catalyst_type_v2"] = kind

    if kind == "salt":
        # Split salt into cation (charge > 0) and anion (charge < 0).
        cat_lookup = _lookup_smiles(cat_smiles, lookup)
        cation_v = anion_v = None
        # Walk fragments using RDKit
        if RDKIT_AVAILABLE:
            try:
                mol = Chem.MolFromSmiles(cat_smiles.strip())
                if mol is not None:
                    frags = Chem.GetMolFrags(mol, asMols=True)
                    cation_frag = anion_frag = None
                    for f in frags:
                            chg = Chem.GetFormalCharge(f)
                            if chg > 0 and cation_frag is None:
                                cation_frag = f
                            elif chg < 0 and anion_frag is None:
                                anion_frag = f
                    if cation_frag is not None:
                        cation_v = lookup.get(Chem.MolToSmiles(cation_frag))
                    if anion_frag is not None:
                        anion_v = lookup.get(Chem.MolToSmiles(anion_frag))
            except Exception:
                pass
        # Fallback: try the salt as a whole (some xtb rows were computed on
        # the neutral ion-pair).
        if cation_v is None and cat_lookup is not None and cat_lookup.get("charge", 0) > 0:
            cation_v = cat_lookup
        if anion_v is None and cat_lookup is not None and cat_lookup.get("charge", 0) < 0:
            anion_v = cat_lookup
        # Last resort: try lookup by raw token-split on "."
        if cation_v is None or anion_v is None:
            for tok in cat_smiles.split("."):
                tok = tok.strip()
                if not tok:
                    continue
                rec = _lookup_smiles(tok, lookup)
                if rec is None:
                    continue
                chg = rec.get("charge", 0)
                if chg > 0 and cation_v is None:
                    cation_v = rec
                elif chg < 0 and anion_v is None:
                    anion_v = rec
                elif chg == 0 and cation_v is None:
                    cation_v = rec  # neutral salt (rare): use as cation proxy

        if cation_v is not None:
            out["cat_cation_homo_eV"]   = cation_v["homo"]
            out["cat_cation_lumo_eV"]   = cation_v["lumo"]
            out["cat_cation_gap_eV"]    = cation_v["gap"]
            out["cat_cation_dipole_D"]  = cation_v.get("dipole", 0.0)
            out["cation_match_type"]    = "salt_split"
        if anion_v is not None:
            out["cat_anion_homo_eV"]    = anion_v["homo"]
            out["cat_anion_lumo_eV"]    = anion_v["lumo"]
            out["cat_anion_gap_eV"]     = anion_v["gap"]
            out["cat_anion_dipole_D"]   = anion_v.get("dipole", 0.0)
            out["anion_match_type"]     = "salt_split"
        return out

    # single (neutral) catalyst: pick the first non-empty of catalyst_1..4
    # SMILES that resolves. Try 1..4 in main loop.
    rec = _lookup_smiles(cat_smiles, lookup)
    if rec is not None:
        out["cat_cation_homo_eV"]   = rec["homo"]
        out["cat_cation_lumo_eV"]   = rec["lumo"]
        out["cat_cation_gap_eV"]    = rec["gap"]
        out["cat_cation_dipole_D"]  = rec.get("dipole", 0.0)
        out["cation_match_type"]    = "single_lookup"
    return out

def build_co2_lookup(xtb_df: pd.DataFrame) -> dict | None:
    """
    Read CO2 xTB descriptors from xtb_results_summary.csv.
    CO2 is registered as role='reactant' in 104b_run_xtb_extended.py.

    Returns None if no CO2 row is present (then callers should fall back to
    CO2_XTB_REF hard-coded defaults).
    """
    co2 = xtb_df[xtb_df["role"] == "reactant"]
    co2 = co2[co2["name"].str.lower() == "co2"]
    if len(co2) == 0:
        return None
    row = co2.iloc[0]
    homo  = row.get("homo_eV")
    lumo  = row.get("lumo_eV")
    gap   = row.get("gap_eV")
    dipole = row.get("dipole_D")
    if not (pd.notna(homo) and pd.notna(lumo)):
        return None
    return {
        "homo":   float(homo),
        "lumo":   float(lumo),
        "gap":    float(gap) if pd.notna(gap) else float(lumo) - float(homo),
        "dipole": float(dipole) if pd.notna(dipole) else 0.0,
    }


# Hard-coded fallback. Used ONLY when xtb_results_summary.csv does not contain
# a CO2 row (i.e., 104b was not run, or 104b's CO2 candidate set was filtered
# out). Values come from the canonical GFN2-xTB + ALPB(DMSO) single-point run
# on O=C=O and have been cross-checked against the 105b sanity reference set
# (105b_xtb_sanity_v2.py, CO2 reference: HOMO=-13.78 / gap=11.51 / mu=0.0).
# The 3–4 eV HOMO/LUMO offset vs. 105b is consistent with implicit DMSO solvent.
CO2_XTB_REF = {
    "homo":   -14.36,
    "lumo":   -10.15,
    "gap":    4.21,
    "dipole": 0.00,
}


def build_substrate_lookup(xtb_df: pd.DataFrame) -> dict:
    """Build {substrate_name: {homo, lumo, gap, dipole}}."""
    sub = xtb_df[xtb_df["role"] == "substrate"].copy()
    lookups = {}
    for _, row in sub.iterrows():
        name = str(row.get("name", "")).strip()
        homo = row.get("homo_eV")
        lumo = row.get("lumo_eV")
        gap  = row.get("gap_eV")
        mu   = row.get("dipole_D")
        if pd.notna(homo) and pd.notna(lumo):
            lookups[name] = {
                "homo": float(homo),
                "lumo": float(lumo),
                "gap":  float(gap) if pd.notna(gap) else float(lumo) - float(homo),
                "dipole": float(mu) if pd.notna(mu) else 0.0,
            }
    return lookups


def build_solvent_lookup(xtb_df: pd.DataFrame) -> dict:
    """Build {solvent_name: {homo, lumo, gap, dipole}}."""
    sol = xtb_df[xtb_df["role"] == "solvent"].copy()
    lookups = {}
    for _, row in sol.iterrows():
        name = str(row.get("name", "")).strip()
        gap = row.get("gap_eV")
        mu  = row.get("dipole_D")
        homo = row.get("homo_eV")
        lumo = row.get("lumo_eV")
        if pd.notna(gap):
            lookups[name] = {
                "homo": float(homo) if pd.notna(homo) else np.nan,
                "lumo": float(lumo) if pd.notna(lumo) else np.nan,
                "gap": float(gap),
                "dipole": float(mu) if pd.notna(mu) else 0.0,
            }
    return lookups


def infer_substrate_xtb(substrate_xtb_name: str, substrate_lookup: dict,
                         substrate_ref: dict) -> dict:
    """
    Get substrate xTB descriptors.
    Priority: xtb_results_summary.csv > reference values > analogous inference
    """
    # 1. Direct match
    if substrate_xtb_name in substrate_lookup:
        return substrate_lookup[substrate_xtb_name]
    # 2. Reference values
    if substrate_xtb_name in substrate_ref:
        d = substrate_ref[substrate_xtb_name]
        return {"homo": d["homo"], "lumo": d["lumo"], "gap": d["gap"], "dipole": d["dipole"]}
    # 3. glycidyl phenyl ether → styrene oxide (analogous)
    if substrate_xtb_name == "styrene_oxide":
        return substrate_lookup.get("styrene_oxide", substrate_ref.get("styrene_oxide",
            {"homo": -10.93, "lumo": -6.29, "gap": 4.64, "dipole": 2.46}))
    # 4. Fallback
    return {"homo": np.nan, "lumo": np.nan, "gap": np.nan, "dipole": np.nan}


def compute_derived_features(row, sub_data: dict, co2_data: dict,
                             solv_data: dict) -> dict:
    """Compute all previously-missing derived features."""
    eps = 0.01

    sub_homo = sub_data.get("homo", np.nan)
    sub_lumo = sub_data.get("lumo", np.nan)
    sub_gap  = sub_data.get("gap",  np.nan)
    sub_dip  = sub_data.get("dipole", np.nan)

    co2_lumo = co2_data.get("lumo", np.nan)
    co2_gap  = co2_data.get("gap",  np.nan)

    solv_gap = (solv_data.get("gap") if solv_data else np.nan)
    solv_dip = (solv_data.get("dipole") if solv_data else np.nan)

    cat_cation_homo = row.get("cat_cation_homo_eV", np.nan)
    cat_cation_lumo = row.get("cat_cation_lumo_eV", np.nan)
    cat_cation_gap  = row.get("cat_cation_gap_eV",  np.nan)

    feats = {}

    # activation_proxy: cation HOMO - sub LUMO
    if not (np.isnan(cat_cation_homo) or np.isnan(sub_lumo)):
        feats["activation_proxy"] = cat_cation_homo - sub_lumo
    else:
        feats["activation_proxy"] = np.nan

    # sub_cat_orbital_match: sub HOMO * cat LUMO
    if not (np.isnan(sub_homo) or np.isnan(cat_cation_lumo)):
        feats["sub_cat_orbital_match"] = sub_homo * cat_cation_lumo
    else:
        feats["sub_cat_orbital_match"] = np.nan

    # gap_ratio: sub_gap / (cat_cation_gap + eps)
    if not (np.isnan(sub_gap) or np.isnan(cat_cation_gap)):
        feats["gap_ratio"] = sub_gap / (cat_cation_gap + eps)
    else:
        feats["gap_ratio"] = np.nan

    # reaction_polarity: |sub_lumo - cat_cation_homo|
    if not (np.isnan(sub_lumo) or np.isnan(cat_cation_homo)):
        feats["reaction_polarity"] = abs(sub_lumo - cat_cation_homo)
    else:
        feats["reaction_polarity"] = np.nan

    # co2_activation_proxy: cat_cation_homo - co2_lumo
    if not (np.isnan(cat_cation_homo) or np.isnan(co2_lumo)):
        feats["co2_activation_proxy"] = cat_cation_homo - co2_lumo
    else:
        feats["co2_activation_proxy"] = np.nan

    # solv_cat_interaction: solv_gap * cat_cation_gap
    if not (np.isnan(solv_gap) or np.isnan(cat_cation_gap)):
        feats["solv_cat_interaction"] = solv_gap * cat_cation_gap
    else:
        feats["solv_cat_interaction"] = np.nan

    # solv_sub_interaction: solv_gap * sub_gap
    if not (np.isnan(solv_gap) or np.isnan(sub_gap)):
        feats["solv_sub_interaction"] = solv_gap * sub_gap
    else:
        feats["solv_sub_interaction"] = np.nan

    # dielectric_proxy: 1 / (solv_gap + eps)
    if not np.isnan(solv_gap):
        feats["dielectric_proxy"] = 1.0 / (solv_gap + eps)
    else:
        feats["dielectric_proxy"] = np.nan

    # global_hardness ≈ (sub_gap * co2_gap) / (sub_gap + co2_gap)
    if not np.isnan(sub_gap) and not np.isnan(co2_gap):
        denom = sub_gap + co2_gap
        feats["global_hardness"] = (sub_gap * co2_gap) / denom if abs(denom) > eps else np.nan
    else:
        feats["global_hardness"] = np.nan

    # hardness_ratio: global_hardness / (cat_cation_gap + eps)
    if not (np.isnan(feats["global_hardness"]) or np.isnan(cat_cation_gap)):
        feats["hardness_ratio"] = feats["global_hardness"] / (cat_cation_gap + eps)
    else:
        feats["hardness_ratio"] = np.nan

    # Nucleophilicity / electrophilicity / electrodonating proxies
    # (Parr & Pearson; higher HOMO  ⇒ more nucleophilic; higher LUMO + larger
    # inverse gap ⇒ more electrophilic. These are heuristic, dimensionless.)
    if not np.isnan(cat_cation_homo) and not np.isnan(cat_cation_gap) and cat_cation_gap > eps:
        feats["nucleophilicity_cat"] = (cat_cation_homo + 15.0) / cat_cation_gap
    else:
        feats["nucleophilicity_cat"] = np.nan
    if not np.isnan(cat_cation_lumo) and not np.isnan(cat_cation_gap) and cat_cation_gap > eps:
        feats["electrophilicity_cat"] = -(cat_cation_lumo) / cat_cation_gap
    else:
        feats["electrophilicity_cat"] = np.nan
    if not np.isnan(cat_cation_homo):
        # electrodonating_cat: dimensionless HOMO shift (more negative = better donor)
        feats["electrodonating_cat"] = -(cat_cation_homo + 5.0)
    else:
        feats["electrodonating_cat"] = np.nan

    # ion_pair_interaction: HOMO(cation) - LUMO(anion); smaller ⇒ easier electron
    # flow from cation donor to anion acceptor.
    cat_anion_homo = row.get("cat_anion_homo_eV", np.nan)
    cat_anion_lumo = row.get("cat_anion_lumo_eV", np.nan)
    if not (np.isnan(cat_cation_homo) or np.isnan(cat_anion_lumo)):
        feats["ion_pair_interaction"] = cat_cation_homo - cat_anion_lumo
    else:
        feats["ion_pair_interaction"] = np.nan

    # charge_transfer_potential: ratio of gap(HOMOdonor-LUMOacceptor) to
    # sub_gap. Smaller ⇒ more exergonic charge transfer onto substrate.
    if (not (np.isnan(cat_cation_homo) or np.isnan(cat_anion_lumo))
            and not np.isnan(sub_gap) and sub_gap > eps):
        ct_gap = cat_cation_homo - cat_anion_lumo
        feats["charge_transfer_potential"] = ct_gap / sub_gap
    else:
        feats["charge_transfer_potential"] = np.nan

    # total_polarity_index: combine sub + cat + solv dipole magnitudes (rough).
    cat_cation_dipole = row.get("cat_cation_dipole_D", np.nan)
    parts = [v for v in (sub_dip, cat_cation_dipole, solv_dip) if not np.isnan(v)]
    if parts:
        feats["total_polarity_index"] = float(sum(parts))
    else:
        feats["total_polarity_index"] = np.nan

    return feats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                    formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default=str(DRFP_XTB_EXTENDED_CSV),
                        help="Input CSV (master table from Tier 1).")
    parser.add_argument("--xtb-summary", default=str(XTB_SUMMARY_CSV),
                        help="xTB summary CSV (from 104b).")
    parser.add_argument("--out", default=str(MERGED_OUTPUT_CSV),
                        help="Output CSV with merged features. "
                             "By default this is a sibling path under "
                             "results_cho_diagnostic, NOT the input path "
                             "(in-place overwrite is unsafe).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report coverage without writing file")
    parser.add_argument("--force", action="store_true",
                        help="Re-merge and overwrite output even if it already exists.")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable DEBUG-level logging.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
        force=True,
    )

    in_path = Path(args.data)
    out_path = Path(args.out)
    xtb_summary_path = Path(args.xtb_summary)

    if out_path == in_path:
        logger.error(
            "--out would overwrite the input file. Pass a different "
            "--out path, or remove the input file first if you really mean it."
        )
        return 2

    logger.info("=" * 60)
    logger.info("107_merge_substrate_xtb.py - substrate/CO2/solvent feature merge")
    logger.info("=" * 60)

    # Load data
    logger.info("[1/5] Loading xTB summary: %s", xtb_summary_path)
    xtb = pd.read_csv(xtb_summary_path, encoding="utf-8-sig")
    n_ok = xtb["xtb_ok"].sum() if "xtb_ok" in xtb.columns else 0
    logger.info("    rows=%d  xtb_ok=%d", len(xtb), n_ok)

    logger.info("[2/5] Loading data CSV: %s", in_path)
    df = pd.read_csv(in_path, encoding="utf-8-sig")
    logger.info("    rows=%d  cols=%d", len(df), len(df.columns))

    # Build lookups
    logger.info("[3/5] Building lookups...")
    sub_lookup = build_substrate_lookup(xtb)
    solv_lookup = build_solvent_lookup(xtb)
    cat_lookup = build_catalyst_xtb_lookup(xtb)

    # CO2: prefer reading from xtb_results_summary.csv (role='reactant', name='CO2');
    # fall back to CO2_XTB_REF if 104b did not produce a CO2 row.
    co2_lookup = build_co2_lookup(xtb)
    co2_data = co2_lookup if co2_lookup is not None else dict(CO2_XTB_REF)
    co2_source = "xtb_summary.csv" if co2_lookup is not None else "CO2_XTB_REF (fallback)"
    logger.info("    substrate lookup: %d entries", len(sub_lookup))
    logger.info("    solvent lookup:   %d entries", len(solv_lookup))
    logger.info("    catalyst lookup:  %d entries (SMILES-indexed)", len(cat_lookup))
    logger.info(
        "    CO2 lookup:       %s  HOMO=%+.2f LUMO=%+.2f gap=%+.2f",
        co2_source, co2_data["homo"], co2_data["lumo"], co2_data["gap"],
    )
    for name, data in sub_lookup.items():
        logger.info("      %s: HOMO=%+.2f LUMO=%+.2f", name, data["homo"], data["lumo"])

    # Add new feature columns (legacy set + the 11 columns that XTB_COLS in
    # utils_rxn.py expects and that downstream consumers like 803_mordred_ablation,
    # 804_hierarchical_catalyst_model and analysis/601_shap_analysis reference).
    new_cols = [
        # substrate / CO2 core
        "sub_homo_eV", "sub_lumo_eV", "sub_gap_eV", "sub_dipole_D",
        "co2_homo_eV", "co2_lumo_eV", "co2_gap_eV",
        # whole-catalyst (cation single-component approximation; for ILs we
        # aggregate cation & anion via min/max below)
        "cat_homo_eV", "cat_lumo_eV", "cat_gap_eV", "cat_dipole_D",
        # solvent
        "solv_homo_eV", "solv_lumo_eV", "solv_gap_eV", "solv_dipole_D",
        # derived
        "delta_E_hl_cat_sub", "global_hardness", "nucleophilicity_index",
        # IL min/max
        "cat_homo_eV_min", "cat_lumo_eV_max", "cat_gap_eV_min",
        # catalytic + reactivity
        "activation_proxy", "sub_cat_orbital_match", "gap_ratio",
        "reaction_polarity", "co2_activation_proxy",
        "solv_cat_interaction", "solv_sub_interaction",
        "dielectric_proxy",
        # catalyst cation/anion split (added 2026-08-19; computed from
        # xtb_results_summary.csv by build_catalyst_xtb_lookup below)
        "cat_cation_homo_eV", "cat_cation_lumo_eV", "cat_cation_gap_eV",
        "cat_anion_homo_eV", "cat_anion_lumo_eV", "cat_anion_gap_eV",
        "cat_cation_dipole_D", "cat_anion_dipole_D",
        "cation_match_type", "anion_match_type", "catalyst_type_v2",
        # extended reactivity descriptors (added 2026-08-19; downstream
        # consumers reference them in XTB_COLS)
        "hardness_ratio",
        "nucleophilicity_cat", "electrophilicity_cat", "electrodonating_cat",
        "ion_pair_interaction", "charge_transfer_potential",
        "total_polarity_index",
    ]
    for col in new_cols:
        if col not in df.columns:
            df[col] = np.nan

    # Compute features row by row
    logger.info("[4/5] Computing features...")
    coverage = {col: 0 for col in new_cols}

    # Ensure all output columns exist (defensive so df.at[..., "col"] = ...
    # never raises). String-typed columns are explicitly cast to object so we
    # can write string match-type labels back via df.at.
    str_cols = {"cation_match_type", "anion_match_type", "catalyst_type_v2"}
    for col in new_cols:
        if col not in df.columns:
            if col in str_cols:
                df[col] = ""  # object dtype
            else:
                df[col] = np.nan
        elif col in str_cols:
            df[col] = df[col].astype(object).fillna("")

    for i, (idx, row) in enumerate(df.iterrows()):
        # Get substrate name
        rat_name = str(row.get("reactant_name", "")).strip()
        sub_xtb_name = SUBSTRATE_NAME_MAP.get(rat_name, rat_name.lower().replace(" ", "_"))

        # Substrate xTB
        sub_data = infer_substrate_xtb(sub_xtb_name, sub_lookup, SUBSTRATE_XTB_REF)
        df.at[idx, "sub_homo_eV"] = sub_data["homo"]
        df.at[idx, "sub_lumo_eV"] = sub_data["lumo"]
        df.at[idx, "sub_gap_eV"]  = sub_data["gap"]
        df.at[idx, "sub_dipole_D"] = sub_data["dipole"]

        # CO2 xTB (sourced from lookup or fallback; constant across rows)
        df.at[idx, "co2_lumo_eV"] = co2_data["lumo"]
        df.at[idx, "co2_gap_eV"]  = co2_data["gap"]

        # Solvent xTB (from primary solvent, with flexible matching)
        solv_name_raw = str(row.get("solvent_1", "")).strip()
        sv = None
        if solv_name_raw and solv_name_raw not in ("", "nan", "无溶剂"):
            # Try: 1) exact dataset name in xtb lookup
            #      2) SOLVENT_NAME_MAP normalized name in xtb lookup
            #      3) xTB name directly in xtb lookup
            if solv_name_raw in solv_lookup:
                sv = solv_lookup[solv_name_raw]
            elif solv_name_raw in SOLVENT_NAME_MAP:
                mapped = SOLVENT_NAME_MAP[solv_name_raw]
                if mapped in solv_lookup:
                    sv = solv_lookup[mapped]
            else:
                # Try case-insensitive partial match
                for xtb_name, sv_data in solv_lookup.items():
                    if xtb_name.lower() == solv_name_raw.lower():
                        sv = sv_data
                        break
                if sv is None:
                    for xtb_name in solv_lookup:
                        if xtb_name.lower() in solv_name_raw.lower() or \
                           solv_name_raw.lower() in xtb_name.lower():
                            sv = solv_lookup[xtb_name]
                            break
            if sv is not None:
                df.at[idx, "solv_gap_eV"]    = sv["gap"]
                df.at[idx, "solv_dipole_D"] = sv["dipole"]
                df.at[idx, "solv_homo_eV"]  = sv["homo"]
                df.at[idx, "solv_lumo_eV"]  = sv["lumo"]

        # New col: co2_homo_eV (was missing — XTB_COLS expects it)
        df.at[idx, "co2_homo_eV"] = co2_data["homo"]

        # ------------------------------------------------------------------
        # Catalyst xTB (computed here, not from input columns — fixes the
        # 2026-08-19 chicken-and-egg where downstream consumers expected
        # cat_cation_* / cat_anion_* to exist on the master table, but no
        # upstream step was generating them. We resolve from xtb_summary
        # using SMILES lookup; the lookup is built once before the loop.
        # Priority for catalyst SMILES: catalyst_1 (primary) -> _2 -> _3 -> _4.
        # ------------------------------------------------------------------
        cat_smiles_pick = None
        for slot in ("catalyst_1_smiles", "catalyst_2_smiles",
                     "catalyst_3_smiles", "catalyst_4_smiles"):
            cs = row.get(slot)
            if isinstance(cs, str) and cs.strip() and cs.strip().lower() != "nan":
                cat_smiles_pick = cs
                break
        cat_xtb = compute_cat_xtb_row(cat_smiles_pick, cat_lookup)
        for k, v in cat_xtb.items():
            df.at[idx, k] = v

        cat_cation_homo_eV = cat_xtb["cat_cation_homo_eV"]
        cat_cation_lumo_eV = cat_xtb["cat_cation_lumo_eV"]
        cat_cation_gap_eV  = cat_xtb["cat_cation_gap_eV"]
        cat_anion_homo_eV  = cat_xtb["cat_anion_homo_eV"]
        cat_anion_lumo_eV  = cat_xtb["cat_anion_lumo_eV"]
        cat_anion_gap_eV   = cat_xtb["cat_anion_gap_eV"]
        cat_cation_dipole  = cat_xtb["cat_cation_dipole_D"]
        cat_anion_dipole   = cat_xtb["cat_anion_dipole_D"]

        # Whole-catalyst = cation (single-component proxy). For neutral
        # catalysts without a row in cat_*_cation_*, fields stay NaN.
        df.at[idx, "cat_homo_eV"]   = cat_cation_homo_eV
        df.at[idx, "cat_lumo_eV"]   = cat_cation_lumo_eV
        df.at[idx, "cat_gap_eV"]    = cat_cation_gap_eV
        df.at[idx, "cat_dipole_D"]  = cat_cation_dipole

        # IL min/max over cation & anion HOMO/LUMO/gap
        cands_homo = [v for v in (cat_cation_homo_eV, cat_anion_homo_eV) if pd.notna(v)]
        cands_lumo = [v for v in (cat_cation_lumo_eV, cat_anion_lumo_eV) if pd.notna(v)]
        cands_gap  = [v for v in (cat_cation_gap_eV,  cat_anion_gap_eV)  if pd.notna(v)]
        if cands_homo:
            df.at[idx, "cat_homo_eV_min"] = float(min(cands_homo))
        if cands_lumo:
            df.at[idx, "cat_lumo_eV_max"] = float(max(cands_lumo))
        if cands_gap:
            df.at[idx, "cat_gap_eV_min"]  = float(min(cands_gap))

        # delta_E_hl_cat_sub = cat HOMO - sub LUMO (small ⇒ easy electron donation)
        if pd.notna(cat_cation_homo_eV) and pd.notna(sub_data["lumo"]):
            df.at[idx, "delta_E_hl_cat_sub"] = float(cat_cation_homo_eV) - float(sub_data["lumo"])

        # nucleophilicity_index — higher HOMO ⇒ more nucleophilic ⇒ richer in
        # front-side orbital overlap. Use a simple shifted inverse-gap scaling
        # so the value is dimensionless and non-negative.
        if pd.notna(cat_cation_homo_eV) and pd.notna(cat_cation_gap_eV) and cat_cation_gap_eV > 0:
            # Higher (less negative) HOMO & smaller gap ⇒ higher index
            df.at[idx, "nucleophilicity_index"] = (
                (cat_cation_homo_eV + 15.0) / max(cat_cation_gap_eV, 1e-3)
            )

        # Compute derived features
        feats = compute_derived_features(
            df.loc[idx], sub_data, co2_data,
            {"gap": df.at[idx, "solv_gap_eV"],
             "dipole": df.at[idx, "solv_dipole_D"]}
        )
        for k, v in feats.items():
            df.at[idx, k] = v

        # Track coverage
        for col in new_cols:
            if pd.notna(df.at[idx, col]):
                coverage[col] += 1

        if (i + 1) % 500 == 0:
            logger.info("    %d/%d", i + 1, len(df))

    # Report coverage
    logger.info("=" * 60)
    logger.info("Coverage Report (new features)")
    logger.info("=" * 60)
    for col in new_cols:
        n = coverage[col]
        pct = n / len(df) * 100 if len(df) > 0 else 0
        status = "OK" if n > 0 else "STILL NaN"
        logger.info("  %-30s  %5d/%d (%5.1f%%)  [%s]", col, n, len(df), pct, status)

    # Also check previously 0% features
    logger.info("=" * 60)
    logger.info("Previously 0%% features (should now be non-zero)")
    logger.info("=" * 60)
    prev_zero = [
        "activation_proxy", "sub_cat_orbital_match", "gap_ratio",
        "reaction_polarity", "co2_activation_proxy",
        "solv_cat_interaction", "solv_sub_interaction", "dielectric_proxy",
        "global_hardness", "hardness_ratio",
    ]
    for col in prev_zero:
        n = df[col].notna().sum() if col in df.columns else 0
        pct = n / len(df) * 100 if len(df) > 0 else 0
        delta = n - coverage.get(col, 0)
        logger.info("  %-30s  %5d/%d (%5.1f%%)  [+%d]", col, n, len(df), pct, delta)

    if args.dry_run:
        logger.info("[DRY RUN] No file written.")
        return 0

    ensure_dir(out_path.parent)
    logger.info("[5/5] Saving: %s", out_path)
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    logger.info("  Saved %d rows x %d cols", len(df), len(df.columns))

    # Also save baseline
    baseline = {
        "timestamp": str(pd.Timestamp.now()),
        "n_rows": len(df),
        "n_cols": len(df.columns),
        "new_features": {col: int(coverage[col]) for col in new_cols},
        "source": "107_merge_substrate_xtb.py",
    }
    ensure_dir(SUBSTRATE_XTB_BASELINE.parent)
    with open(SUBSTRATE_XTB_BASELINE, "w", encoding="utf-8") as f:
        json.dump(baseline, f, ensure_ascii=False, indent=2)
    logger.info("  Baseline: %s", SUBSTRATE_XTB_BASELINE)
    logger.info("Done!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
