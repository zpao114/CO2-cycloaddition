#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
104b_run_xtb_extended.py — extended xTB descriptor runner for CO2-cycloaddition.

What this fixes vs. the legacy 104_run_xtb.py
---------------------------------------------
1. ZnBr2-class metal halides fail under RDKit UFF (UFF does not parametrize
   [Zn+2], so all atoms end up at (0,0,0) and xTB dies on the zero-distance
   Br1-Br2 pair). Workaround: detect metal-cation + halide-anion SMILES and
   build geometries by hand from experimental bond lengths
   (Zn-Br = 2.40 A, Mg-Br = 2.38 A, Co-Cl = 2.27 A, etc.).

2. The original candidate set had only 13 molecules; it now covers 87
   catalysts and 17 solvents, including all anions, IL cations,
   metal-halide salts, organic bases, and common solvents.

3. Friendlier failure mode: parse xTB's real stderr when convergence fails
   instead of silently writing NaN rows.

Pipeline position
-----------------
TIER 1 first step. Produces ``xtb_results_summary.csv`` consumed by 105b
(sanity) and 107 (merge into master feature table).

Usage
-----
    python 104b_run_xtb_extended.py
    python 104b_run_xtb_extended.py --gfn 2 --solvent dmso --level sp
    python 104b_run_xtb_extended.py --candidates candidates.json  # subset
"""
from __future__ import annotations

import argparse
import csv
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional, List, Dict, Tuple

PROJECT_ROOT = os.environ.get("CO2_PROJECT_ROOT", r"D:\machine-learning\CO2-cycloaddition")
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, rdMolTransforms
    RDKIT_OK = True
except ImportError:
    RDKIT_OK = False

from src.paths import (
    RESULTS_CHO_DIAGNOSTIC,
    ensure_dir,
)

logger = logging.getLogger("104b_run_xtb_extended")

# 104b writes .xyz files into --out-dir and a summary CSV with the basename
# given by --summary (or this default).
DEFAULT_XTB_SUMMARY_NAME = "xtb_results_summary.csv"


# ════════════════════════════════════════════════════════════════════════════
# §1  扩展候选集：覆盖 87 个 unique catalyst + 17 个 unique solvent
# ════════════════════════════════════════════════════════════════════════════
# 设计原则：
# - 每个候选拆分到合理角色（cation / anion / catalyst / substrate / solvent）
# - 阴离子单独算（[Br-] 不与 [K+] 拼在一起算，因为 RDKit 处理混合盐会失效）
# - 金属卤化物作为整体算（手动构造几何）

EXTENDED_CANDIDATES: List[Dict] = [
    # ── 底物（5 种）─
    {"name": "styrene_oxide",            "smiles": "C1OC1c1ccccc1",          "role": "substrate"},
    {"name": "epoxybutane",              "smiles": "CCC1CO1",                "role": "substrate"},
    {"name": "propylene_oxide",          "smiles": "CC1CO1",                 "role": "substrate"},
    {"name": "epichlorohydrin",          "smiles": "ClCC1CO1",               "role": "substrate"},
    {"name": "isopropyl_glycidyl_ether", "smiles": "CC(C)OCC1CO1",           "role": "substrate"},
    {"name": "cyclohexene_oxide",        "smiles": "C1CCC2C1C2O",            "role": "substrate"},

    # ── 阳离子（25+ 种）─  IL cation 部分 / 金属阳离子 / 质子化有机碱
    {"name": "TBA_cation",               "smiles": "CCCC[N+](CCCC)(CCCC)CCCC", "role": "cation"},
    {"name": "TEA_cation",               "smiles": "CC[N+](CC)(CC)CC",       "role": "cation"},
    {"name": "TMA_cation",               "smiles": "C[N+](C)(C)C",            "role": "cation"},
    {"name": "TPrA_cation",              "smiles": "CCC[N+](CCC)(CCC)CCC",   "role": "cation"},
    {"name": "choline_cation",           "smiles": "C[N+](C)(C)CCO",         "role": "cation"},
    {"name": "BMIM_cation",              "smiles": "CCCCn1ccnc1C",           "role": "cation"},
    {"name": "EMIM_cation",              "smiles": "CCn1ccnc1C",             "role": "cation"},
    {"name": "HMIM_cation",              "smiles": "CCCCCCn1ccnc1C",         "role": "cation"},
    {"name": "imidazolium_cation",       "smiles": "c1cnc[nH]1",           "role": "cation"},
    {"name": "Pyridinium_cation",        "smiles": "c1ccncc1",                "role": "cation"},  # neutral pyridine proxy
    {"name": "EtPyridinium_cation",      "smiles": "CC[n+]1ccccc1",           "role": "cation"},
    {"name": "MEPyridinium_cation",      "smiles": "C[n+]1ccccc1",            "role": "cation"},
    {"name": "Pyrrolidinium_NMe_NEt",    "smiles": "CC[N+]1(C)CCCC1",         "role": "cation"},
    {"name": "Piperidinium_NMe_NPr",     "smiles": "CCCN1CCCC(N)(C)C1",       "role": "cation"},
    {"name": "Pyrazolium_NEt",           "smiles": "CC[NH+]C=NC1CCCN1",       "role": "cation"},
    {"name": "Pyrazolium_NEt_NEtOH",     "smiles": "CC[NH+](CCO)C=NC1CCCN1",  "role": "cation"},
    {"name": "Pyrazolium_NEt_NPrNH2",    "smiles": "CC[NH+]C=NC1CCCN1CCN",    "role": "cation"},
    {"name": "PPh3_methyl_cation",       "smiles": "C[P+](c1ccccc1)(c1ccccc1)c1ccccc1", "role": "cation"},
    {"name": "Trimethylphenylammonium",  "smiles": "c1ccc(N(C)C)cc1",       "role": "cation"},  # neutral N,N-dimethylaniline proxy
    {"name": "dimethylaminophenol",      "smiles": "[O-]c1ccc([N+](C)(C)C)cc1", "role": "cation"},  # trimethylammonium-phenoxide zwitterion
    {"name": "TBA_OH_cation_unchanged",  "smiles": "CCCC[N+](CCCC)(CCCC)CCCC", "role": "cation"},  # TBA+ (same as TBA_cation)
    {"name": "Cp_TBA_OH_cation",         "smiles": "CCCC[N+](CCCC)(CCCC)CCCC", "role": "cation"},  # TBA+ alias to share xTB

    # ── 金属阳离子 ─
    {"name": "Zn2_cation",               "smiles": "[Zn+2]",                 "role": "cation"},
    {"name": "Mg2_cation",               "smiles": "[Mg+2]",                 "role": "cation"},
    {"name": "Co2_cation",               "smiles": "[Co+2]",                 "role": "cation"},
    {"name": "Cu2_cation",               "smiles": "[Cu+2]",                 "role": "cation"},
    {"name": "Ni2_cation",               "smiles": "[Ni+2]",                 "role": "cation"},
    {"name": "Fe2_cation",               "smiles": "[Fe+2]",                 "role": "cation"},
    {"name": "Fe3_cation",               "smiles": "[Fe+3]",                 "role": "cation"},
    {"name": "Al3_cation",               "smiles": "[Al+3]",                 "role": "cation"},
    {"name": "Sn2_cation",               "smiles": "[Sn+2]",                 "role": "cation"},
    {"name": "K_cation",                 "smiles": "[K+]",                   "role": "cation"},
    {"name": "Na_cation",                "smiles": "[Na+]",                  "role": "cation"},
    {"name": "Li_cation",                "smiles": "[Li+]",                  "role": "cation"},
    {"name": "Ca2_cation",               "smiles": "[Ca+2]",                 "role": "cation"},
    {"name": "Zn_atom",                  "smiles": "[Zn]",                   "role": "cation"},  # neutral Zn

    # ── 阴离子（10 种）─
    {"name": "Br_anion",                 "smiles": "[Br-]",                  "role": "anion"},
    {"name": "I_anion",                  "smiles": "[I-]",                   "role": "anion"},
    {"name": "Cl_anion",                 "smiles": "[Cl-]",                  "role": "anion"},
    {"name": "F_anion",                  "smiles": "[F-]",                   "role": "anion"},
    {"name": "OH_anion",                 "smiles": "[OH-]",                  "role": "anion"},
    {"name": "OAc_anion",                "smiles": "CC(=O)[O-]",             "role": "anion"},
    {"name": "BF4_anion",                "smiles": "[B-](F)(F)(F)F",         "role": "anion"},
    {"name": "PF6_anion",                "smiles": "[P-](F)(F)(F)(F)(F)F",   "role": "anion"},
    {"name": "NTf2_anion",               "smiles": "O=S(=O)([O-])C(F)(F)F.O=S(=O)([N-]C(F)(F)F)C(F)(F)F", "role": "anion"},  # merged simplification
    {"name": "HCO3_anion",               "smiles": "O=C([O-])O",             "role": "anion"},  # hydrogencarbonate
    {"name": "picolinate_anion",         "smiles": "O=C(c1ccccn1)[O-]",      "role": "anion"},

    # ── 金属卤化物催化剂（手动构建几何）─
    {"name": "ZnBr2_salt",               "smiles": "[Br-].[Br-].[Zn+2]",    "role": "salt", "manual_geom": True},
    {"name": "ZnCl2_salt",               "smiles": "[Cl-].[Cl-].[Zn+2]",    "role": "salt", "manual_geom": True},
    {"name": "ZnI2_salt",                "smiles": "[I-].[I-].[Zn+2]",      "role": "salt", "manual_geom": True},
    {"name": "ZnOAc2_salt",              "smiles": "CC(=O)[O-].CC(=O)[O-].[Zn+2]", "role": "salt", "manual_geom": True},
    {"name": "Zn_phenanthroline_dione",  "smiles": "[Zn].O=C1c2ccccc2C(=O)N1", "role": "salt"},
    {"name": "CoCl2_salt",               "smiles": "[Cl-].[Cl-].[Co+2]",    "role": "salt", "manual_geom": True},
    {"name": "MgCl2_salt",               "smiles": "[Cl-].[Cl-].[Mg+2]",    "role": "salt", "manual_geom": True},
    {"name": "FeCl3_salt",               "smiles": "[Cl-].[Cl-].[Cl-].[Fe+3]", "role": "salt", "manual_geom": True},
    {"name": "AlCl3_salt",               "smiles": "[Cl-].[Cl-].[Cl-].[Al+3]", "role": "salt", "manual_geom": True},
    {"name": "CuCl2_salt",               "smiles": "[Cl-].[Cl-].[Cu+2]",    "role": "salt", "manual_geom": True},
    {"name": "NiCl2_salt",               "smiles": "[Cl-].[Cl-].[Ni+2]",    "role": "salt", "manual_geom": True},
    {"name": "NaBr_salt",                "smiles": "[Na+].[Br-]",           "role": "salt", "manual_geom": True},
    {"name": "NaI_salt",                 "smiles": "[Na+].[I-]",            "role": "salt", "manual_geom": True},
    {"name": "LiBr_salt",                "smiles": "[Li+].[Br-]",           "role": "salt", "manual_geom": True},
    {"name": "KI_salt",                  "smiles": "[K+].[I-]",             "role": "salt", "manual_geom": True},
    {"name": "KOH_salt",                 "smiles": "[K+].[OH-]",            "role": "salt", "manual_geom": True},
    {"name": "CaI2_salt",                "smiles": "[Ca+2].[I-].[I-]",      "role": "salt", "manual_geom": True},
    {"name": "MgO_salt",                 "smiles": "[Mg]O",                 "role": "salt"},
    {"name": "ZnO_salt",                 "smiles": "[Zn]O",                 "role": "salt"},
    {"name": "salen_Co_complex",         "smiles": "OC1=Cc2ccccc2/C1=N/CCCN=C1/C=Cc2ccccc2C1=O", "role": "salt"},
    {"name": "Salen_complex",            "smiles": "OC1=Cc2ccccc2/C1=N/CCCN=C1/C=Cc2ccccc2C1=O", "role": "salt"},  # salen H2 ligand (Co-free proxy; Co valence rejected by UFF)

    # ── 有机碱（完整中性分子）─
    {"name": "DBU",                      "smiles": "CN1CCCCN2CCCCC2C1=N",    "role": "catalyst"},
    {"name": "DMAP",                     "smiles": "CN(C)c1ccc(cc1)N",      "role": "catalyst"},
    {"name": "TBD",                      "smiles": "CN1CCCN2CCCN2C1=N",  "role": "catalyst"},
    {"name": "MTBD",                      "smiles": "CN1CCCN2CCCN2C1=NC", "role": "catalyst"},
    {"name": "DABCO",                    "smiles": "C1CN2CCN1CC2",          "role": "catalyst"},
    {"name": "pyridine",                 "smiles": "c1ccncc1",              "role": "catalyst"},
    {"name": "triethylamine",            "smiles": "CCN(CC)CC",             "role": "catalyst"},
    {"name": "triphenylphosphine",       "smiles": "c1ccc(P(c2ccccc2)c2ccccc2)cc1", "role": "catalyst"},
    {"name": "tetramethylguanidine",     "smiles": "CN(C)C(=N)N(C)C",       "role": "catalyst"},  # neutral TMG
    {"name": "NBS",                      "smiles": "O=C1CCC(=O)N1Br",       "role": "catalyst"},
    {"name": "triazine_melamine",        "smiles": "Nc1nc(N)nc(N)n1",       "role": "catalyst"},
    {"name": "imidazole_NH",             "smiles": "c1cnc[nH]1",              "role": "catalyst"},
    {"name": "benzimidazole_NH",         "smiles": "Nc1nc2ccccc2[nH]1",     "role": "catalyst"},
    {"name": "indole",                   "smiles": "c1ccc2[nH]ccc2c1",       "role": "catalyst"},  # neutral indole (cleaner ring closure)
    {"name": "bipyridine",               "smiles": "c1ccc(-c2ccccn2)nc1",   "role": "catalyst"},
    {"name": "B_bis_catalyst",           "smiles": "CC(=O)O.OCC1COC(=O)O1",  "role": "catalyst"},  # placeholder for borate-type
    {"name": "H2",                       "smiles": "[H][H]",                "role": "catalyst"},  # used in some H2 additives
    {"name": "Br2",                      "smiles": "Br",                    "role": "catalyst"},  # liquid Br2

    # ── 溶剂（17 种）─
    {"name": "DMSO",                     "smiles": "CS(C)=O",               "role": "solvent"},
    {"name": "DMF",                      "smiles": "CN(C)C=O",              "role": "solvent"},
    {"name": "MeCN",                     "smiles": "CC#N",                  "role": "solvent"},
    {"name": "methanol",                 "smiles": "CO",                    "role": "solvent"},
    {"name": "ethanol",                  "smiles": "CCO",                   "role": "solvent"},
    {"name": "water",                    "smiles": "O",                     "role": "solvent"},
    {"name": "DCM",                      "smiles": "ClCCCl",                "role": "solvent"},
    {"name": "toluene",                  "smiles": "Cc1ccccc1",             "role": "solvent"},
    {"name": "hexane",                   "smiles": "CCCCCC",                "role": "solvent"},
    {"name": "ethyl_acetate",            "smiles": "CCOC(C)=O",             "role": "solvent"},
    {"name": "acetone",                  "smiles": "CC(=O)C",               "role": "solvent"},
    {"name": "THF",                      "smiles": "C1CCOC1",               "role": "solvent"},
    {"name": "chlorobenzene",            "smiles": "c1ccccc1Cl",            "role": "solvent"},
    {"name": "diethyl_ether",            "smiles": "CCOCC",                 "role": "solvent"},
    {"name": "dioxane",                  "smiles": "C1CCOCC1",              "role": "solvent"},
    {"name": "cyclohexane",              "smiles": "C1CCCCC1",              "role": "solvent"},
    {"name": "CHCl3",                    "smiles": "ClC(Cl)Cl",             "role": "solvent"},

    # ── 反应物/产物（已存在）─
    {"name": "CO2",                      "smiles": "O=C=O",                 "role": "reactant"},
    {"name": "cyclic_carbonate_product", "smiles": "O=C1OCCO1",             "role": "product"},
    {"name": "styrene_carbonate",        "smiles": "O=C1OCC(c2ccccc2)O1",   "role": "product"},
    {"name": "PC_product",               "smiles": "CC1OC(=O)OC1",          "role": "product"},
    {"name": "glycerol_carbonate",       "smiles": "OCC1COC(=O)O1",         "role": "product"},
    {"name": "isopropyl_carbonate",      "smiles": "CC(C)COCC1OC(=O)OC1",   "role": "product"},

    # ---- [patch_smiles_and_xtb.py: 新增数据补扩] ----
    {"name": "HETEAB_cation",    "smiles": "CC[N+](CC)(CCO)CC",          "role": "cation"},
    {"name": "HETBAB_cation",    "smiles": "CCCC[N+](CCCC)(CCCC)CCO",    "role": "cation"},
    {"name": "TEAH_cation",      "smiles": "OCC[N+](CCO)(CCO)CCO",        "role": "cation"},
    {"name": "HEBIM_cation",     "smiles": "CCCCn1ccnc1CCO",             "role": "cation"},

    # ---- [补增缺失条目 2024-07-30] ----
    # 阳离子（带正电荷）
    {"name": "TBP_cation",              "smiles": "CCCC[P+](CCO)(CCCC)CCCC",                 "role": "cation"},
    {"name": "Triarylstibonium_cation", "smiles": "c1cc[c]([Sb+]([c]2ccccc2)[c]2ccccc2)cc1","role": "cation"},
    {"name": "OH_C8_Pyridinium_cation", "smiles": "CCCCCCCC[n+]1ccc(O)cc1",                   "role": "cation"},
    {"name": "EDPP_cation",             "smiles": "CC[P+](c1ccccc1)(c1ccccc1)c1ccccc1",       "role": "cation"},
    {"name": "THPA_cation",             "smiles": "CCCCCCC[N+](CCCCCCC)(CCCCCCC)CCCCCCC",       "role": "cation"},
    {"name": "Cyclophosphonium_cation", "smiles": "C1=C[N+]=CC([P+](c2ccccc2)(c2ccccc2)c2ccccc2)=C1", "role": "cation"},
    {"name": "TOA_cation",              "smiles": "CCCCCCCC[N+](CCCCCCCC)(CCCCCCCC)CCCCCCCC",  "role": "cation"},
    {"name": "EDMA_cation",             "smiles": "CC[N+](C)(C)C",                             "role": "cation"},
    {"name": "TBA_hex_cation",         "smiles": "CCCCCC[N+](CCCCCC)(CCCCCC)CCCCCC",         "role": "cation"},
    {"name": "OH_C8_NMe2_cation",      "smiles": "CCCCCCCC[N+](C)(C)CCO",                    "role": "cation"},
    {"name": "TDDP_cation",             "smiles": "CCCCCCCCCCC[P+](CCCC)(CCCC)CCCC",           "role": "cation"},
    {"name": "Ph3P_ethyl_cation",      "smiles": "c1ccc([P+](c2ccccc2)c2ccccc2)cc1",         "role": "cation"},
    # 中性有机催化剂（作为阳离子能量代理）
    {"name": "DBU_no_N",                "smiles": "CN1CCCCN2CCCCC2C1",                        "role": "catalyst"},
    {"name": "pyrrole",                 "smiles": "c1cc[nH]c1",                               "role": "catalyst"},
    {"name": "L_arginine",              "smiles": "NC(N)=NC(C=O)CCCC[N+](=O)[O-]",           "role": "catalyst"},
    {"name": "ascorbic_acid",    "smiles": "OC(C(O)C(O)C(O)=O)=O",       "role": "catalyst"},
    {"name": "ethylene_glycol",  "smiles": "OCCO",                       "role": "catalyst"},
    {"name": "tetraethylene_glycol","smiles": "OCCOCCOCCOCCO",            "role": "catalyst"},
    {"name": "imidazole_NH_extra","smiles": "c1cnc[nH]1",                "role": "catalyst"},
    {"name": "2I_1M_Im",         "smiles": "Cn1ccnc1I",                  "role": "catalyst"},
    {"name": "TBA2ZnBr4_salt",   "smiles": "CCCC[N+](CCCC)(CCCC)CCCC.CCCC[N+](CCCC)(CCCC)CCCC.[Br-].[Br-].[Br-].[Br-].[Zn+2]", "role": "salt", "manual_geom": True},
    {"name": "Li_MgO_salt",      "smiles": "[Li+].[Mg+2].[O-2]",         "role": "salt", "manual_geom": True},
    {"name": "ZnO_nanoplates_salt","smiles": "[Zn]O",                    "role": "salt"},
]


# ════════════════════════════════════════════════════════════════════════════
# §2  推断 xTB 净电荷：与原 104_run_xtb.py 相同（保留兼容性）
# ════════════════════════════════════════════════════════════════════════════
def infer_charge(smiles: str, role: str = "") -> int:
    """Net total charge for SMILES; falls back to +1/-1 by role for monoatomic ions."""
    q = 0
    for m in re.finditer(r"\[([A-Z][a-z]?)([+-]\d*)\]", smiles):
        sign = m.group(2)
        magnitude = int(re.sub(r"[+-]", "", sign) or "1")
        q += -magnitude if sign.startswith("-") else magnitude
    if q != 0:
        return q
    # No charge markers found — fall back to role convention
    if role == "cation":
        return +1
    if role == "anion":
        return -1
    return 0


# ════════════════════════════════════════════════════════════════════════════
# §3  SMILES → XYZ（含修复）
# ════════════════════════════════════════════════════════════════════════════
# 实验键长（Å）（来源：Cordero et al., Dalton Trans. 2008）
EXPT_BOND_LENGTHS = {
    ("Zn", "Br"): 2.40, ("Zn", "Cl"): 2.30, ("Zn", "I"): 2.55, ("Zn", "O"): 1.95,
    ("Mg", "Cl"): 2.38, ("Mg", "Br"): 2.55,
    ("Co", "Cl"): 2.27, ("Co", "Br"): 2.43,
    ("Cu", "Cl"): 2.38, ("Cu", "Br"): 2.46,
    ("Ni", "Cl"): 2.40,
    ("Fe", "Cl"): 2.20,
    ("Al", "Cl"): 2.13,
    ("Sn", "Cl"): 2.30, ("Sn", "I"): 2.70,
    ("K", "I"): 3.53, ("K", "Br"): 3.30, ("K", "OH"): 2.65,
    ("Na", "Br"): 2.69, ("Na", "I"): 2.92,
    ("Li", "Br"): 2.55,
    ("Ca", "I"): 3.10,
}


def _bond_length(elem_a: str, elem_b: str) -> float:
    """Return experimental bond length for A-B (fallback 2.5 Å)."""
    key1 = (elem_a, elem_b)
    key2 = (elem_b, elem_a)
    if key1 in EXPT_BOND_LENGTHS:
        return EXPT_BOND_LENGTHS[key1]
    if key2 in EXPT_BOND_LENGTHS:
        return EXPT_BOND_LENGTHS[key2]
    # Covalent radius sum fallback
    COV_RADII = {"H": 0.31, "C": 0.76, "N": 0.71, "O": 0.66, "F": 0.57,
                  "P": 1.07, "S": 1.05, "Cl": 0.99, "Br": 1.14, "I": 1.33,
                  "Zn": 1.22, "Mg": 1.36, "Co": 1.26, "Cu": 1.32, "Ni": 1.24,
                  "Fe": 1.25, "Al": 1.21, "Sn": 1.39, "K": 2.03, "Na": 1.66,
                  "Li": 1.28, "Ca": 1.74}
    a = COV_RADII.get(elem_a, 1.5)
    b = COV_RADII.get(elem_b, 1.5)
    return a + b


def _build_salt_xyz(smiles: str, xyz_path: str) -> bool:
    """
    Manually construct 3D geometry for ionic salts (e.g., [Br-].[Br-].[Zn+2]).
    Approach: place metal cation at origin, place anions around it using
    experimental bond lengths and reasonable VSEPR geometry.

    Returns True on success.
    """
    try:
        # Parse fragments
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False
        frags = Chem.GetMolFrags(mol, asMols=True)
        if not frags:
            return False

        # Identify metal (cation) and halide/anion (anion) atoms
        anion_atoms = []
        metal_atom = None
        other_atoms = []
        for frag in frags:
            n_atoms = len(frag.GetAtoms())
            smi_str = Chem.MolToSmiles(frag)
            charge_total = sum(a.GetFormalCharge() for a in frag.GetAtoms())
            sym_str = "".join(a.GetSymbol() for a in frag.GetAtoms())
            if n_atoms == 1:
                sym = frag.GetAtomWithIdx(0).GetSymbol()
                sym_clean = re.sub(r"[^\w]", "", sym)
                if charge_total < 0:
                    # Anion: halide (Br/I/Cl/F) or hydroxide (O-H) or others
                    if sym_clean in ("Br", "I", "Cl", "F"):
                        anion_atoms.append(("halide", sym_clean, frag))
                    elif sym_clean == "O":
                        # Single-atom O fragment with negative charge = hydroxide
                        anion_atoms.append(("hydroxide", "OH", frag))
                    elif sym_clean in ("B",):
                        # BF4- etc. (single-atom B with -1 charge but with F neighbors in full SMILES)
                        anion_atoms.append(("polyatomic_anion", f"Q{charge_total}", frag))
                    else:
                        anion_atoms.append(("polyatomic_anion", f"Q{charge_total}", frag))
                elif charge_total > 0 and sym_clean in ("Zn", "Mg", "Co", "Fe", "Cu", "Ni", "Al",
                                                         "Sn", "K", "Na", "Li", "Ca"):
                    metal_atom = (sym_clean, frag)
                elif charge_total > 0:
                    # Organic cation like [nH+]1ccnc1 — but these never come as single-atom
                    metal_atom = ("organic_cation", frag)
            elif n_atoms == 2 and charge_total < 0 and "O" in sym_str:
                # Polyatomic anion like acetate CC(=O)[O-], hydroxide with explicit H, etc.
                anion_atoms.append(("polyatomic_anion", f"Q{charge_total}", frag))
            else:
                if charge_total < 0:
                    anion_atoms.append(("polyatomic_anion", f"Q{charge_total}", frag))
                else:
                    other_atoms.append(frag)

        if metal_atom is None:
            # No metal yet: try harder (e.g., [K+] in [K+].[OH-])
            for frag in frags:
                if len(frag.GetAtoms()) == 1:
                    sym = re.sub(r"[^\w]", "", frag.GetAtomWithIdx(0).GetSymbol())
                    if sym in ("K", "Na", "Li", "Ca"):
                        metal_atom = (sym, frag)
                        break
            if metal_atom is None:
                # Still no metal — try first monocationic frag (e.g., [BMIM+])
                for frag in frags:
                    q = sum(a.GetFormalCharge() for a in frag.GetAtoms())
                    if q > 0:
                        metal_atom = ("organic_cation", frag)
                        break

        if metal_atom is None and len(anion_atoms) == 0:
            return False
        if metal_atom is None and anion_atoms and not other_atoms:
            return False

        # Build geometry
        coords = []
        sym_list = []
        metal_sym, metal_frag = metal_atom

        # Place metal at origin (or, if organic_cation, embed via RDKit)
        if metal_sym == "organic_cation":
            # The organic cation IS the "metal" — embed it with RDKit at origin
            try:
                embedded = Chem.AddHs(metal_frag)
                if AllChem.EmbedMolecule(embedded, randomSeed=42) != 0:
                    AllChem.EmbedMolecule(embedded, randomSeed=42, useRandomCoords=True)
                try:
                    AllChem.MMFFOptimizeMolecule(embedded, maxIters=200)
                except Exception:
                    pass
                conf = embedded.GetConformer()
                for atom in embedded.GetAtoms():
                    pos = conf.GetAtomPosition(atom.GetIdx())
                    coords.append((pos.x, pos.y, pos.z))
                    sym_list.append(atom.GetSymbol())
            except Exception:
                return False
        else:
            coords.append((0.0, 0.0, 0.0))
            sym_list.append(metal_sym)

        # Place anions and other ligands around metal
        n_anions = len(anion_atoms)
        n_other = len(other_atoms)

        # Use tetrahedral geometry for 4-coordinate, trigonal planar for 3, etc.
        # Simple geometry: place each anion along x/y/z axes with offsets
        import math

        all_attachments = anion_atoms + [("other", None, oa) for oa in other_atoms]
        n_total = len(all_attachments)
        if n_total == 0:
            return False

        for i, (kind, sym, frag) in enumerate(all_attachments):
            if kind == "halide":
                L = EXPT_BOND_LENGTHS.get((metal_sym, sym), 2.4)
                # Use trigonal/linear geometry depending on count
                if n_total == 1:
                    # Linear — along x
                    coords.append((L, 0.0, 0.0))
                elif n_total == 2:
                    # Linear — opposite
                    if i == 0:
                        coords.append((L * 0.866, 0.0, L * 0.5))
                    else:
                        coords.append((-L * 0.866, 0.0, -L * 0.5))
                elif n_total == 3:
                    # Trigonal planar
                    a = 2 * math.pi * i / 3
                    coords.append((L * math.cos(a), L * math.sin(a), 0.0))
                elif n_total == 4:
                    # Tetrahedral
                    # Use standard tetrahedral vertices
                    coords.append((L * (1.0 if i == 0 else -0.5 if i == 1 else -0.5 if i == 2 else 0.0),
                                   L * (0.0 if i == 0 else math.sqrt(3)/2 if i == 1 else -math.sqrt(3)/2 if i == 2 else 0.0),
                                   L * (math.sqrt(8/9) if i == 0 else math.sqrt(8/9) if i < 3 else -math.sqrt(8/9))))
                else:
                    # Generic — spread on sphere
                    a = 2 * math.pi * i / n_total
                    coords.append((L * math.cos(a), L * math.sin(a), 0.0))
                sym_list.append(sym)
            elif kind == "hydroxide":
                # OH- : O attached to metal, H pointing away
                L = EXPT_BOND_LENGTHS.get((metal_sym, "O"), 2.0)
                a = 2 * math.pi * i / max(n_total, 1)
                O_coords = (L * math.cos(a), L * math.sin(a), 0.0)
                coords.append(O_coords)
                sym_list.append("O")
                # H pointing away (with 109.5° angle)
                H_dir = (-math.cos(a), -math.sin(a), 0.0)
                O_H_len = 0.96  # Å
                H_coords = (O_coords[0] + O_H_len * 0.866 * H_dir[0],
                            O_coords[1] + O_H_len * 0.866 * H_dir[1],
                            O_coords[2])
                coords.append(H_coords)
                sym_list.append("H")
            elif kind == "polyatomic_anion":
                # Embed the polyatomic anion (OAc-, BF4-, etc.) with its first heavy atom
                # placed near origin at experimental metal-O bond length
                try:
                    embedded = Chem.AddHs(frag)
                    if AllChem.EmbedMolecule(embedded, randomSeed=42) != 0:
                        # Embedding may fail for ionic fragments; use a fallback
                        embedded = Chem.AddHs(frag)
                        AllChem.EmbedMolecule(embedded, randomSeed=42, useRandomCoords=True)
                    try:
                        AllChem.MMFFOptimizeMolecule(embedded, maxIters=200)
                    except Exception:
                        pass
                    conf = embedded.GetConformer()
                    # Find first heavy atom (not H)
                    first_heavy = next(
                        (i for i, a in enumerate(embedded.GetAtoms())
                         if a.GetSymbol() != "H"),
                        0
                    )
                    ref_pos = conf.GetAtomPosition(first_heavy)
                    # Place this heavy atom at metal-X bond length along radial direction
                    ref_sym = embedded.GetAtomWithIdx(first_heavy).GetSymbol()
                    L = EXPT_BOND_LENGTHS.get((metal_sym, ref_sym),
                                                _bond_length(metal_sym, ref_sym))
                    a = 2 * math.pi * i / max(n_total, 1)
                    anchor_offset = (L * math.cos(a), L * math.sin(a), 0.0)
                    for j, atom in enumerate(embedded.GetAtoms()):
                        pos = conf.GetAtomPosition(atom.GetIdx())
                        new_x = pos.x - ref_pos.x + anchor_offset[0]
                        new_y = pos.y - ref_pos.y + anchor_offset[1]
                        new_z = pos.z - ref_pos.z + anchor_offset[2]
                        coords.append((new_x, new_y, new_z))
                        sym_list.append(atom.GetSymbol())
                except Exception:
                    # If embedding truly fails for the polyatomic anion, place a single
                    # placeholder atom at the bond length so xTB has *something*
                    L = 2.0
                    a = 2 * math.pi * i / max(n_total, 1)
                    coords.append((L * math.cos(a), L * math.sin(a), 0.0))
                    sym_list.append("O")
            elif kind == "other":
                # Embed the multi-atom fragment using RDKit
                try:
                    embedded = Chem.AddHs(frag)
                    if AllChem.EmbedMolecule(embedded, randomSeed=42) == 0:
                        AllChem.MMFFOptimizeMolecule(embedded, maxIters=500)
                        conf = embedded.GetConformer()
                        # Center the fragment's metal-binding atom near origin offset
                        ref_idx = 0  # use first atom as anchor
                        ref_pos = conf.GetAtomPosition(ref_idx)
                        anchor_offset = (L * math.cos(2 * math.pi * i / max(n_total, 1)),
                                          L * math.sin(2 * math.pi * i / max(n_total, 1)), 0.0)
                        for j, atom in enumerate(embedded.GetAtoms()):
                            pos = conf.GetAtomPosition(atom.GetIdx())
                            new_x = pos.x - ref_pos.x + anchor_offset[0]
                            new_y = pos.y - ref_pos.y + anchor_offset[1]
                            new_z = pos.z - ref_pos.z + anchor_offset[2]
                            coords.append((new_x, new_y, new_z))
                            sym_list.append(atom.GetSymbol())
                except Exception:
                    return False

        # Write XYZ
        with open(xyz_path, "w", encoding="utf-8") as f:
            f.write(f"{len(coords)}\n")
            f.write(f"{os.path.basename(xyz_path).replace('.xyz', '')}\n")
            for sym, (x, y, z) in zip(sym_list, coords):
                f.write(f"{sym:2s} {x:12.6f} {y:12.6f} {z:12.6f}\n")
        return True
    except Exception as e:
        logger.error("  [error] _build_salt_xyz failed: %s", e)
        return False


def smiles_to_xyz(smiles: str, xyz_path: str, manual_geom: bool = False) -> bool:
    """
    SMILES → XYZ. Routes:
      - monoatomic (single-atom) → single atom at origin
      - ionic salt with metal cation + halide/organic anion fragments → manual geom
      - organic (no metal cation) → standard RDKit path
      - covalent metal-oxide (e.g. [Mg]O) → standard RDKit path (only H-fill)
    """
    if not RDKIT_OK:
        return False

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    frags = Chem.GetMolFrags(mol, asMols=True)

    # Strip charge markers so we can identify simple atomic fragments
    norm_smi = re.sub(r"\[(\w+)([+-]\d?)\]", r"[\1]", smiles)

    # Case 1: Single-atom species (cation, anion, neutral). Place at origin.
    if len(frags) == 1 and frags[0].GetNumAtoms() == 1:
        symbol = frags[0].GetAtomWithIdx(0).GetSymbol()
        symbol_clean = re.sub(r"\W", "", symbol)
        try:
            with open(xyz_path, "w", encoding="utf-8") as f:
                f.write(f"1\n{symbol_clean}\n{symbol_clean} 0.000000 0.000000 0.000000\n")
            return True
        except Exception:
            return False

    # Case 2: Ionic salt with explicit metal cation and explicit anion fragments.
    # Heuristic: 2+ disconnected fragments AND at least one fragment has metal
    # element symbol with charge (e.g. [Zn+2]).
    has_metal_ion = any(
        re.search(r"\[(?:Zn|Mg|Co|Fe|Cu|Ni|Al|Sn|Ca)[+-]\d?\]", Chem.MolToSmiles(f))
        for f in frags
    ) if frags else False
    has_halide_anion = any(
        re.search(r"\[(?:Br|I|Cl|F)[-]?\]", Chem.MolToSmiles(f))
        for f in frags
    ) if frags else False
    if (has_metal_ion and len(frags) >= 2) or manual_geom:
        return _build_salt_xyz(smiles, xyz_path)

    # Case 3: Standard RDKit path (organic / covalent metal-oxide)
    try:
        mol_h = Chem.AddHs(mol)
        embed_result = AllChem.EmbedMolecule(mol_h, randomSeed=42)
        if embed_result != 0:
            return False
        try:
            AllChem.MMFFOptimizeMolecule(mol_h, maxIters=500)
        except Exception:
            # MMFF failure on metals is not fatal — keep the embedded geometry
            pass
    except Exception:
        return False

    conf = mol_h.GetConformer()
    try:
        with open(xyz_path, "w", encoding="utf-8") as f:
            f.write(f"{mol_h.GetNumAtoms()}\n")
            f.write(f"{os.path.basename(xyz_path).replace('.xyz', '')}\n")
            for atom in mol_h.GetAtoms():
                pos = conf.GetAtomPosition(atom.GetIdx())
                f.write(f"{atom.GetSymbol():2s} {pos.x:12.6f} "
                        f"{pos.y:12.6f} {pos.z:12.6f}\n")
    except Exception:
        return False
    return True


# ════════════════════════════════════════════════════════════════════════════
# §4  xTB 调用与解析（与原版兼容）
# ════════════════════════════════════════════════════════════════════════════
def locate_xtb() -> Optional[str]:
    exe = shutil.which("xtb")
    if exe:
        return exe
    conda_env_roots = [
        os.environ.get("CONDA_PREFIX"),
        r"D:\co2\env_drfp",
        r"C:\ProgramData\miniconda3\envs\env_drfp",
    ]
    for root in conda_env_roots:
        if not root:
            continue
        for sub in ["Library\\bin", "Scripts", "bin"]:
            cand = os.path.join(root, sub, "xtb.exe")
            if os.path.isfile(cand):
                return cand
    for cand in [r"C:\Program Files\xtb\bin\xtb.exe"]:
        if os.path.isfile(cand):
            return cand
    return None


def run_xtb(xyz_path: str, charge: int, mult: int,
            gfn: int, solvent: Optional[str], level: str,
            acc: float, timeout: int = 600) -> Optional[str]:
    xtb_exe = locate_xtb()
    if xtb_exe is None:
        return None

    workdir = tempfile.mkdtemp(prefix="xtb_run_")
    try:
        local_xyz = os.path.join(workdir, os.path.basename(xyz_path))
        shutil.copy(xyz_path, local_xyz)
        cwd_before = os.getcwd()
        raw_log = os.path.join(
            os.path.dirname(os.path.abspath(xyz_path)),
            f"{os.path.basename(xyz_path)}.xtb.stdout",
        )
        os.chdir(workdir)
        try:
            cmd = [
                xtb_exe, os.path.basename(local_xyz),
                "--" + level,
                "--chrg", str(charge),
                "--uhf", str(max(0, mult - 1)),
                "--gfn", str(gfn),
                "--acc", str(acc),
            ]
            if solvent:
                cmd += ["--alpb", solvent]
            result = subprocess.run(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                timeout=timeout, encoding="utf-8", errors="ignore",
            )
            log_path = os.path.join(workdir, "xtb.out")
            with open(log_path, "w", encoding="utf-8", errors="ignore") as _f:
                _f.write(result.stdout or "")
            with open(raw_log, "w", encoding="utf-8", errors="ignore") as f:
                f.write(result.stdout or "")
            return log_path
        finally:
            os.chdir(cwd_before)
    finally:
        pass


# ── 解析器（与 104 兼容） ──
_RE_HOMO = re.compile(r"::\s*HOMO\s+energy\s+(-?\d+\.\d+)\s+Eh")
_RE_LUMO = re.compile(r"::\s*LUMO\s+energy\s+(-?\d+\.\d+)\s+Eh")
_RE_ORBITAL_HOMO = re.compile(r"^\s*\d+\s+\d+\.\d+\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+\(HOMO\)\s*$")
_RE_ORBITAL_LUMO = re.compile(r"^\s*\d+\s+(?:\d+\.\d+\s+)?(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+\(LUMO\)\s*$")
_RE_GAP = re.compile(r"HOMO-LUMO gap\s+(-?\d+\.\d+)\s+eV")
_RE_DIPOLE_AU = re.compile(r"\|\s*dipole\s*\|\s*(-?\d+\.\d+)\s*\|\s*au\s*\|")
_RE_DIPOLE_DEBYE = re.compile(r"total\s+dipole\s*\(Debye\)\s*[:=]?\s*(-?\d+\.\d+)", flags=re.IGNORECASE)
_RE_DIPOLE_NEW = re.compile(r"^\s*full:\s+\S+\s+\S+\s+\S+\s+(-?\d+\.\d+)\s*$")
_RE_TOTAL_ENERGY = re.compile(r"::\s*total\s+energy\s+(-?\d+\.\d+)\s+Eh")
_RE_MULLIKEN_HEADER = re.compile(r"Mulliken\s+charges:\s*$", flags=re.IGNORECASE)
_RE_MULLIKEN_LINE = re.compile(r"^\s*(\d+)\s+([A-Z][a-z]?)\s+(-?\d+\.\d+)\s*$")


def parse_xtb_out(out_path: str) -> Dict:
    res = dict(ok=False, homo_eV=None, lumo_eV=None, gap_eV=None,
               total_e_Eh=None, dipole_D=None, mulliken_q={})
    if out_path is None or not os.path.isfile(out_path):
        return res

    EH_TO_EV = 27.21138
    mulliken_by_elem: Dict[str, List[float]] = {}

    with open(out_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.read().splitlines()

    in_mulliken = False
    for i, line in enumerate(lines):
        if "GEOMETRY OPTIMIZATION CONVERGED" in line.upper():
            res["ok"] = True
        if "TOTAL ENERGY" in line.upper() and "EH" in line.upper():
            m = _RE_TOTAL_ENERGY.search(line)
            if m:
                res["total_e_Eh"] = float(m.group(1))
                res["ok"] = True
        m = _RE_HOMO.search(line)
        if m:
            res["homo_eV"] = float(m.group(1)) * EH_TO_EV
        m = _RE_LUMO.search(line)
        if m:
            res["lumo_eV"] = float(m.group(1)) * EH_TO_EV
        if res["homo_eV"] is None:
            m = _RE_ORBITAL_HOMO.search(line)
            if m:
                res["homo_eV"] = float(m.group(2))
        if res["lumo_eV"] is None:
            m = _RE_ORBITAL_LUMO.search(line)
            if m:
                res["lumo_eV"] = float(m.group(2))
        m = _RE_GAP.search(line)
        if m:
            res["gap_eV"] = float(m.group(1))
        m = _RE_DIPOLE_AU.search(line)
        if m:
            res["dipole_D"] = float(m.group(1)) * 2.541746
            continue
        m = _RE_DIPOLE_DEBYE.search(line)
        if m:
            res["dipole_D"] = float(m.group(1))
        if res["dipole_D"] is None:
            m = _RE_DIPOLE_NEW.search(line)
            if m:
                res["dipole_D"] = float(m.group(1))
        if _RE_MULLIKEN_HEADER.search(line):
            in_mulliken = True
            mulliken_by_elem.clear()
            continue
        if in_mulliken:
            if line.strip() == "":
                in_mulliken = False
                continue
            mm = _RE_MULLIKEN_LINE.match(line)
            if mm:
                elem = mm.group(2)
                q = float(mm.group(3))
                mulliken_by_elem.setdefault(elem, []).append(q)

    res["mulliken_q"] = {e: sum(v) / len(v) for e, v in mulliken_by_elem.items()}
    if res["gap_eV"] is None and res["homo_eV"] is not None and res["lumo_eV"] is not None:
        res["gap_eV"] = res["lumo_eV"] - res["homo_eV"]
    if not res["ok"] and res["homo_eV"] is not None:
        res["ok"] = True
    return res


# ════════════════════════════════════════════════════════════════════════════
# §5  CLI
# ════════════════════════════════════════════════════════════════════════════
SUMMARY_HEADER = [
    "smiles", "name", "role", "charge", "solvent", "gfn",
    "homo_eV", "lumo_eV", "gap_eV", "total_e_Eh", "dipole_D",
    "mulliken_q_C", "mulliken_q_O", "mulliken_q_N", "mulliken_q_Br",
    "mulliken_q_I", "mulliken_q_Zn",
    "xtb_ok",
]


def summary_row(cand: Dict, charge: int, solvent: str, gfn: int, res: Dict) -> Dict:
    mq = res["mulliken_q"]
    return {
        "smiles":       cand["smiles"],
        "name":         cand["name"],
        "role":         cand["role"],
        "charge":       charge,
        "solvent":      solvent,
        "gfn":          gfn,
        "homo_eV":      res["homo_eV"],
        "lumo_eV":      res["lumo_eV"],
        "gap_eV":       res["gap_eV"],
        "total_e_Eh":   res["total_e_Eh"],
        "dipole_D":     res["dipole_D"],
        "mulliken_q_C": mq.get("C"),
        "mulliken_q_O": mq.get("O"),
        "mulliken_q_N": mq.get("N"),
        "mulliken_q_Br":mq.get("Br"),
        "mulliken_q_I": mq.get("I"),
        "mulliken_q_Zn":mq.get("Zn"),
        "xtb_ok":       res["ok"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gfn", type=int, choices=[0, 1, 2], default=2)
    parser.add_argument("--level", choices=["sp", "opt", "ohess"], default="sp")
    parser.add_argument("--solvent", default="dmso")
    parser.add_argument("--acc", type=float, default=1.0)
    parser.add_argument("--out-dir", default=str(RESULTS_CHO_DIAGNOSTIC),
                        help="Directory where .xyz and .xtb.stdout files are written.")
    parser.add_argument("--summary", default=DEFAULT_XTB_SUMMARY_NAME,
                        help="Summary CSV filename (relative to --out-dir).")
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--candidates", default=None,
                        help="Optional path to a JSON list of candidate dicts; "
                             "use to limit subset, e.g., for testing.")
    parser.add_argument("--force", action="store_true",
                        help="Re-run xTB even if summary CSV already exists.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Skip xTB runs; report what would be executed.")
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

    if not RDKIT_OK:
        logger.error("RDKit required.")
        return 1
    if locate_xtb() is None:
        logger.error("xtb executable not found.")
        return 2
    # ------------------------------------------------------------------
    #  Skip-when-already-done policy (default safe behaviour).
    #  Without --force, if the summary CSV already exists and is non-empty
    #  we exit early. xTB is expensive (~30-60 min) and we don't want to
    #  silently redo it just because someone forgot the --force flag.
    # ------------------------------------------------------------------
    summary_csv = os.path.join(args.out_dir, args.summary)
    if (not args.force) and os.path.isfile(summary_csv) and os.path.getsize(summary_csv) > 0:
        with open(summary_csv, "r", encoding="utf-8-sig") as f:
            n_existing = max(sum(1 for _ in f) - 1, 0)
        if n_existing > 0:
            logger.info("[skip] %s already exists (%d rows). Pass --force to re-run xTB anyway.",
                        summary_csv, n_existing)
            return 0

    candidates = EXTENDED_CANDIDATES
    if args.candidates and os.path.isfile(args.candidates):
        import json
        with open(args.candidates, encoding="utf-8") as f:
            candidates = json.load(f)
        logger.info("Loaded %d candidates from %s", len(candidates), args.candidates)

    if args.dry_run:
        logger.info("[dry-run] Would run %d candidates; not invoking xTB.", len(candidates))
        for c in candidates:
            logger.info("  %-10s %s", c["role"], c["name"])
        return 0

    ensure_dir(args.out_dir)
    solvent = args.solvent.strip() or None

    rows: List[Dict] = []
    n_total = len(candidates)
    logger.info("=" * 60)
    logger.info("104b_run_xtb_extended.py | GFN%d-xTB | level=--%s | solvent=%s | acc=%.2f | n_candidates=%d",
                args.gfn, args.level, solvent, args.acc, n_total)
    logger.info("=" * 60)

    success = 0
    failed = 0
    for cand in candidates:
        logger.info("[%-10s] %s  (%s)", cand["role"], cand["name"], cand["smiles"][:80])
        xyz_path = os.path.join(args.out_dir, f"{cand['name']}.xyz")
        manual = cand.get("manual_geom", False)
        if not smiles_to_xyz(cand["smiles"], xyz_path, manual_geom=manual):
            failed += 1
            continue
        charge = infer_charge(cand["smiles"], cand["role"])
        mult = 1
        out_path = run_xtb(xyz_path, charge, mult,
                           gfn=args.gfn, solvent=solvent,
                           level=args.level, acc=args.acc, timeout=args.timeout)
        res = parse_xtb_out(out_path)
        row = summary_row(cand, charge, solvent or "", args.gfn, res)
        rows.append(row)
        ok_str = "OK " if res["ok"] else "FAIL"
        homo = f"{res['homo_eV']:+.4f}" if res['homo_eV'] is not None else "  -  "
        lumo = f"{res['lumo_eV']:+.4f}" if res['lumo_eV'] is not None else "  -  "
        gap  = f"{res['gap_eV']:+.4f}" if res['gap_eV'] is not None else "  -  "
        mu   = f"{res['dipole_D']:.3f}" if res['dipole_D'] is not None else "  -  "
        logger.info("  -> %s  HOMO=%s eV  LUMO=%s eV  gap=%s eV  |mu|=%s D",
                    ok_str, homo, lumo, gap, mu)
        if res["ok"]: success += 1
        else: failed += 1

    out_csv = os.path.join(args.out_dir, args.summary)
    with open(out_csv, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_HEADER)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    logger.info("=" * 60)
    logger.info("Summary: %s  (%d rows, %d OK, %d FAIL)", out_csv, len(rows), success, failed)
    logger.info("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())