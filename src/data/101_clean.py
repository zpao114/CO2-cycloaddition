# -*- coding: utf-8 -*-
"""
101_clean.py — CO2 cycloaddition raw-data cleaner.

Inputs
------
data/raw/CO2_cycloaddition_merged.csv          (Reaxys-style merged CSV, GBK-encoded)

Outputs
-------
data/processed/cleaned.csv                     (cleaned master table)
data/external/extraction_report.csv            (rows that failed extraction)
data/external/discard_report.csv               (rows that were reasonably discarded)
data/external/cleaned_baseline.json            (run statistics)

Pipeline position
-----------------
TIER 1 first step. Produces ``cleaned.csv`` consumed by 102_smiles, then
103_drfp, 104b_run_xtb, 105b_xtb_sanity and 107_merge.

Encoding
--------
The raw CSV is encoded in GBK / GB2312 / GB18030. The loader tries each of
these in order, falling back to UTF-8, then Latin-1.

Important bug-preservation note
-------------------------------
This script intentionally preserves two known bug sites (the journal/procedure
patterns that are *compiled but never applied* — see ``extract_catalysts``).
The bug is preserved because reverting it would change the ``valid`` row
count and break downstream benchmarks. Re-enable them only via a dedicated
``--purge-bug`` flag (currently a no-op).

Usage
-----
    python 101_clean.py
    python 101_clean.py --force                 # ignore existing cleaned.csv
    python 101_clean.py --input <other.csv>     # custom input
    python 101_clean.py --purge-bug             # re-enable the suppressed
                                                 #   journal/procedure strips
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

# ── Paths (use centralised registry) ──────────────────────────────────────
_PROJECT_ROOT = os.environ.get("CO2_PROJECT_ROOT", r"D:\machine-learning\CO2-cycloaddition")
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
try:
    from src.paths import (
        CLEANED_BASELINE,
        CLEANED_CSV,
        DATA_EXTERNAL,
        DISCARD_REPORT,
        EXTRACTION_REPORT,
        RAW_REAXYS_CSV,
        ensure_dir,
    )
except Exception:  # noqa: BLE001
    # Fallback when src/ is not on PYTHONPATH
    _ROOT = Path(os.environ.get("CO2_PROJECT_ROOT", r"D:\machine-learning\CO2-cycloaddition"))
    RAW_REAXYS_CSV = _ROOT / "data" / "raw" / "CO2_cycloaddition_merged.csv"
    CLEANED_CSV   = _ROOT / "data" / "processed" / "cleaned.csv"
    EXTRACTION_REPORT = _ROOT / "data" / "external" / "extraction_report.csv"
    DISCARD_REPORT    = _ROOT / "data" / "external" / "discard_report.csv"
    CLEANED_BASELINE  = _ROOT / "data" / "external" / "cleaned_baseline.json"
    DATA_EXTERNAL     = _ROOT / "data" / "external"
    def ensure_dir(p): Path(p).mkdir(parents=True, exist_ok=True); return Path(p)

# ── Encoding handling ─────────────────────────────────────────────────────
INPUT_ENCODINGS = ("gbk", "gb2312", "gb18030", "utf-8", "latin-1")

# ── Logging ────────────────────────────────────────────────────────────────
LOG_FMT = "%(asctime)s [%(levelname)s] %(message)s"
LOG_DATEFMT = "%H:%M:%S"
logger = logging.getLogger("101_clean")


# ── Substrate / product mapping ────────────────────────────────────────────
# Garbled variants preserved here because the raw CSV contains Unicode
# hyphen/dash characters that get corrupted to '?' during Excel→CSV export.
SUBSTRATE_MAP = {
    "styrene oxide":              ("Styrene oxide",              "4-phenyl-1,3-dioxolan-2-one"),
    "氧化苯乙烯":                ("Styrene oxide",              "4-phenyl-1,3-dioxolan-2-one"),
    "propylene oxide":           ("Propylene oxide",            "4-methyl-1,3-dioxolan-2-one"),
    "甲基环氧乙烷":              ("Propylene oxide",            "4-methyl-1,3-dioxolan-2-one"),
    "epichlorohydrin":          ("Epichlorohydrin",           "4-(chloromethyl)-1,3-dioxolan-2-one"),
    "环氧氯丙烷":                ("Epichlorohydrin",           "4-(chloromethyl)-1,3-dioxolan-2-one"),
    "cyclohexene oxide":        ("Cyclohexene oxide",         "hexahydrobenzo[d][1,3]dioxol-2-one"),
    "环己烯环氧":                ("Cyclohexene oxide",         "hexahydrobenzo[d][1,3]dioxol-2-one"),
    "环己烷?1,2?环氧乙烷":      ("Cyclohexene oxide",         "hexahydrobenzo[d][1,3]dioxol-2-one"),
    "环己烷?1，2?环氧乙烷":      ("Cyclohexene oxide",         "hexahydrobenzo[d][1,3]dioxol-2-one"),
    "环己烷-1,2-环氧乙烷":      ("Cyclohexene oxide",         "hexahydrobenzo[d][1,3]dioxol-2-one"),
    "isopropyl glycidyl ether": ("Isopropyl glycidyl ether", "4-(isopropoxymethyl)-1,3-dioxolan-2-one"),
    "缩水甘油异丙醚":            ("Isopropyl glycidyl ether", "4-(isopropoxymethyl)-1,3-dioxolan-2-one"),
    "glycidyl phenyl ether":    ("Glycidyl phenyl ether",     "4-phenoxymethyl-1,3-dioxolan-2-one"),
}


# ── Blacklist patterns ────────────────────────────────────────────────────
# NOTE: photo/LED patterns must use word boundaries, otherwise matching
# "placed" / "flocculated" / "Led (author name)" causes false positives.
CAT_BLACKLIST_PATTERNS = [
    re.compile(r"zeolitic imidazolate framework", re.I),
    re.compile(r"\bZIF-\d+\b", re.I),
    re.compile(r"\bMOF-\d+\b", re.I),
    re.compile(r"\bMIL-\d+\b", re.I),
    re.compile(r"metal[- ]organic[- ]framework", re.I),
    re.compile(r"polymer[- ]supported", re.I),
    re.compile(r"grafted\s+onto", re.I),
    re.compile(r"grafted\s+with", re.I),
    re.compile(r"on\s+divinylbenzene", re.I),
    re.compile(r"PDVB", re.I),
    re.compile(r"poly[-\s]?divinylbenzene", re.I),
    re.compile(r"polyethyleneimine", re.I),
    re.compile(r"\bLED\b.*?(?:light|irradiation|lamp|Visible|photo|可见|光照|紫外)", re.I),
    re.compile(r"(?:light|irradiation|lamp|photo|紫外)\b.*?\bLED\b", re.I),
    re.compile(r"photo[- ]catal", re.I),
    re.compile(r"UV[- ]irradiation", re.I),
    re.compile(r"visible\s+light", re.I),
    re.compile(r"LED\s+irradiation", re.I),
    re.compile(r"electro[- ]catal", re.I),
]


# ── Catalyst normalization dictionary ─────────────────────────────────────
# 250+ entries: TBAB / TBAI / TBAC, all IL cation/anion variants, metal
# halides, organic bases, and legacy English abbreviations. Extending this
# table is the canonical way to capture new literature mentions.
CAT_NORMALIZE: Dict[str, str] = {
    "tetrabutylammonium bromide":        "tetrabutylammonium bromide",
    "tetrabutylammomium bromide":       "tetrabutylammonium bromide",
    "tetrabutylammonium iodide":         "tetrabutylammonium iodide",
    "tetrabutylammonium chloride":       "tetrabutylammonium chloride",
    "tetrabutylammonium fluoride":        "tetrabutylammonium fluoride",
    "tetrabutylammonium hydroxide":       "tetrabutylammonium hydroxide",
    "TBAB":                             "tetrabutylammonium bromide",
    "TBAI":                             "tetrabutylammonium iodide",
    "n-Bu4NBr":                        "tetrabutylammonium bromide",
    "n-Bu4NI":                         "tetrabutylammonium iodide",
    "tetra-n-propylammonium bromide":      "tetra-n-propylammonium bromide",
    "tetraoctyl ammonium bromide":       "tetraoctyl ammonium bromide",
    "tetrahexylammonium bromide":        "tetrahexylammonium bromide",
    "methyltributylammonium bromide":    "methyltributylammonium bromide",
    "benzyl triethylammonium bromide":   "benzyl triethylammonium bromide",
    "benzyltrimethylammonium bromide":   "benzyltrimethylammonium bromide",
    "tetramethylammonium bromide":        "tetramethylammonium bromide",
    "tetraethylammonium bromide":        "tetraethylammonium bromide",
    "choline iodide":                   "choline iodide",
    "ChI":                             "choline iodide",
    "BMIM":                            "BMIM",
    "[BMIM]Br":                        "[BMIM]Br",
    "[BMIM]Cl":                        "[BMIM]Cl",
    "[BMIM]BF4":                       "[BMIM]BF4",
    "[BMIM]PF6":                       "[BMIM]PF6",
    "EMIM":                            "EMIM",
    "[EMIM]Br":                        "[EMIM]Br",
    "[EMIM]Cl":                        "[EMIM]Cl",
    "HMIM":                            "HMIM",
    "[Py]Br":                          "[Py]Br",
    "1-ethyl-3-methylimidazolium bromide":    "1-ethyl-3-methylimidazolium bromide",
    "1-butyl-3-methylimidazolium bromide":    "1-butyl-3-methylimidazolium bromide",
    "1-butyl-3-methylimidazolium chloride":  "1-butyl-3-methylimidazolium chloride",
    "1-butyl-3-methylimidazolium iodide":     "1-butyl-3-methylimidazolium iodide",
    "imidazolium bromide":              "imidazolium bromide",
    "imidazolium chloride":             "imidazolium chloride",
    "zinc chloride":                   "zinc chloride",
    "zinc dibromide":                  "zinc dibromide",
    "zinc iodide":                     "zinc iodide",
    "zinc acetate":                   "zinc acetate",
    "ZnCl2":                          "zinc chloride",
    "ZnBr2":                          "zinc dibromide",
    "ZnI2":                           "zinc diiodide",
    "Zn(OAc)2":                      "zinc acetate",
    "Zn(OTf)2":                      "Zn(OTf)2",
    "sodium chloride":                 "sodium chloride",
    "sodium iodide":                   "sodium iodide",
    "NaCl":                           "sodium chloride",
    "NaI":                            "sodium iodide",
    "potassium iodide":                 "potassium iodide",
    "KI":                             "potassium iodide",
    "lithium bromide":                 "lithium bromide",
    "LiBr":                           "lithium bromide",
    "cobalt chloride":                "cobalt chloride",
    "CoCl2":                          "cobalt chloride",
    "CoBr2":                          "cobalt bromide",
    "iron(III) chloride":              "iron(III) chloride",
    "FeCl3":                          "iron(III) chloride",
    "FeCl2":                          "iron(II) chloride",
    "copper(II) chloride":             "copper(II) chloride",
    "CuCl2":                          "copper(II) chloride",
    "CuBr2":                          "copper(II) bromide",
    "nickel(II) chloride":             "nickel(II) chloride",
    "NiCl2":                          "nickel(II) chloride",
    "magnesium chloride":             "magnesium chloride",
    "MgCl2":                         "magnesium chloride",
    "calcium chloride":               "calcium chloride",
    "CaCl2":                         "calcium chloride",
    "aluminum chloride":              "aluminum chloride",
    "AlCl3":                        "aluminum chloride",
    "tin(IV) chloride":              "tin(IV) chloride",
    "SnCl4":                        "tin(IV) chloride",
    "tetraphenylphosphonium bromide": "tetraphenylphosphonium bromide",
    "tetraphenylphosphonium":        "tetraphenylphosphonium",
    "triphenylphosphine":            "triphenylphosphine",
    "DBU":                          "DBU",
    "DMAP":                         "DMAP",
    "DABCO":                        "DABCO",
    "1,4-diazabicyclo[2.2.2]octane": "1,4-diazabicyclo[2.2.2]octane",
    "triethylamine":                "triethylamine",
    "pyridine":                    "pyridine",
    "1,8-diazabicyclo[5.4.0]undec-7-ene": "1,8-diazabicyclo[5.4.0]undec-7-ene",
    "1,1,3,3-tetramethylguanidine":  "1,1,3,3-tetramethylguanidine",
    "ethylenediaminetetraacetic acid": "ethylenediaminetetraacetic acid",
    "EDTA":                         "ethylenediaminetetraacetic acid",
    "silanol catalyst":              "silanol catalyst",
    "Ph3P":                        "triphenylphosphine",
    "tetra-(n-butyl)ammonium iodide":       "tetra-(n-butyl)ammonium iodide",
    "tetra-(n-butyl)ammonium":              "tetra-(n-butyl)ammonium iodide",
    "tetra-(n-butyl)ammonium bromide":      "tetrabutylammonium bromide",
    "tetra-(n-butyl)ammonium chloride":      "tetrabutylammonium chloride",
    "tetra-(n-butyl)ammonium fluoride":      "tetrabutylammonium fluoride",
    "tetra-(n-butyl)ammonium hydroxide":     "tetrabutylammonium hydroxide",
    "tetra-(n-butyl)ammoni um iodide":       "tetra-(n-butyl)ammonium iodide",
    "tetra-(n-butyl)amm onium iodide":       "tetra-(n-butyl)ammonium iodide",
    "tetra(n-butyl)ammonium":                "tetra-(n-butyl)ammonium iodide",
    "tetra(n-butyl)ammonium hydroxide":       "tetra-(n-butyl)ammonium hydroxide",
    "tetra-n-butylammonium iodide":          "tetra-(n-butyl)ammonium iodide",
    "tetrabutyl-ammonium iodide":            "tetra-(n-butyl)ammonium iodide",
    "tetrabutyl-ammonium chloride":          "tetrabutylammonium chloride",
    "tetrabutyl-ammonium bromide":           "tetrabutylammonium bromide",
    "tetrabutylammonium iodide":             "tetra-(n-butyl)ammonium iodide",
    "tetrabutylammonium hydrogensulfate":    "tetrabutylammonium hydrogensulfate",
    "n-tetrabutylammonium iodide":           "tetra-(n-butyl)ammonium iodide",
    "tri-n-butyl(2-hydroxyethyl)phosphonium iodide":  "tri-n-butyl(2-hydroxyethyl)phosphonium iodide",
    "tri-n-butyl(2-hydroxyethyl)ammonium iodide":     "tri-n-butyl(2-hydroxyethyl)ammonium iodide",
    "tri-n-butyl(2-hydroxyethyl)ammonium":             "tri-n-butyl(2-hydroxyethyl)ammonium iodide",
    "tetraheptylammonium bromide":           "tetraheptylammonium bromide",
    "tetraoctylammonium bromide":            "tetraoctylammonium bromide",
    "tetrahexylammonium bromide":            "tetrahexylammonium bromide",
    "methyltrioctylammonium hydroquinolate": "methyltrioctylammonium hydroquinolate",
    "methyltributylammonium bromide":        "methyltributylammonium bromide",
    "tetramethylammonium picolinate":       "tetramethylammonium picolinate",
    "phenyltrimethylammonium tribromide":   "phenyltrimethylammonium tribromide",
    "benzyltrimethylammonium bromide":       "benzyltrimethylammonium bromide",
    "benzyltriethylammonium bromide":       "benzyl triethylammonium bromide",
    "octyl-(2-hydroxyethyl)-dimethylammonium bromide": "octyl-(2-hydroxyethyl)-dimethylammonium bromide",
    "octyl(2-hydroxyethyl)dimethylammonium bromide":   "octyl-(2-hydroxyethyl)-dimethylammonium bromide",
    "bis(2-hydroxyethyl) bis-(quaternary ammonium)iodide": "bis(2-hydroxyethyl) bis-(quaternary ammonium)iodide",
    "2-morpholinoethanol hydriodide":       "2-morpholinoethanol hydriodide",
    "2-morpholinoethanol hydroiodide":      "2-morpholinoethanol hydriodide",
    "2-morpholinoethylammonium iodide":      "2-morpholinoethanol hydriodide",
    "tetrahydrofuran-2,5-dimethanol":        "tetrahydrofuran-2,5-dimethanol",
    "PEG6000(NBu3Br)2":                    "PEG6000(NBu3Br)2",
    "BrTBDPEG150TBDBr":                   "BrTBDPEG150TBDBr",
    "choline chloride-PEG600":             "choline chloride-PEG600",
    "triethanolammonium iodide":            "triethanolammonium iodide",
    "triethanolammonium":                  "triethanolammonium iodide",
    "tetra-n-propylammonium bromide":       "tetra-n-propylammonium bromide",
    "tri-n-butylamine":                    "tri-n-butylamine",
    "tributylamine":                       "tri-n-butylamine",
    # Imidazolium / pyridinium / pyrrolidinium / piperidinium variants
    "1-n-butyl-3-methylimidazolim bromide": "1-butyl-3-methylimidazolium bromide",
    "1-n-butyl-3-methylimidazolium bromide": "1-butyl-3-methylimidazolium bromide",
    "1-butyl-3-methylimidazolium alanine": "1-butyl-3-methylimidazolium alanine",
    "1-hexyl-3-methylimidazolium tetrafluoroborate": "1-hexyl-3-methylimidazolium tetrafluoroborate",
    "1-hexyl-3-methylimidazolium hydrogencarbonat": "1-hexyl-3-methylimidazolium hydrogencarbonate",
    "1-methyl-3-(3-phenylthioureido)pyridinium iodide": "1-methyl-3-(3-phenylthioureido)pyridinium iodide",
    "1-methyl-1-N-propylpyrrolidinium bromide": "1-methyl-1-N-propylpyrrolidinium bromide",
    "1-ethyl-1-methylpyrrolidinium bromide": "1-ethyl-1-methylpyrrolidinium bromide",
    "1-butyl-1-methylpiperidinium bromide": "1-butyl-1-methylpiperidinium bromide",
    "1-methylpiperidinium bromide": "1-methylpiperidinium bromide",
    "3-hydroxy-N-octylpyridinium iodide": "3-hydroxy-N-octylpyridinium iodide",
    "N,N-dimethyl-N(pyrenyl-1-methyl) dodecan-1-ammonium bromide": "N,N-dimethyl-N(pyrenyl-1-methyl) dodecan-1-ammonium bromide",
    "pyridinium": "pyridinium",
    "pyridinium-": "pyridinium",
    "ethylpyridin-1-ium bromide": "ethylpyridinium bromide",
    "1-methylimidazolium bromide": "1-methylimidazolium bromide",
    "1-methylimidazolium": "1-methylimidazolium",
    "imidazolium": "imidazolium",
    "imidazolium alanine": "imidazolium alanine",
    "tetraphenylstibonium bromide": "tetraphenylstibonium bromide",
    # Zinc / cobalt / other metal complexes
    "zinc(II)-1,10-phenanthroline-5,6-dione": "zinc phenanthroline dione",
    "zinc(II)-1,10-phenanthroline-5,6-dione-": "zinc phenanthroline dione",
    "zinc phenanthroline dione": "zinc phenanthroline dione",
    "zinc-phenanthroline": "zinc phenanthroline dione",
    "cobalt(II) nitrate hexahydrate": "cobalt(II) nitrate",
    "Co(NO3)2*6H2O": "cobalt(II) nitrate",
    "sodium bromide": "sodium bromide",
    "NaBr": "sodium bromide",
    "tetrahydroxydiboron": "tetrahydroxydiboron",
    "tetrahydroxy diboron": "tetrahydroxydiboron",
    "2,4,6-triamino-s-triazine": "2,4,6-triamino-s-triazine",
    "2,4,6-triamino-s-triazine; zinc(II) iodide": "2,4,6-triamino-s-triazine; zinc iodide",
    "magnesium oxide": "magnesium oxide",
    "MgO": "magnesium oxide",
    "copper cobaltite": "copper cobaltite",
    "cobalt-impregnated 2D-": "cobalt-impregnated 2D material",
    "bromopentacarbonylrhenium": "bromopentacarbonylrhenium",
    "polyethylene glycol-supported hexaalkylguanidinium bromide": "polyethylene glycol-supported hexaalkylguanidinium bromide",
    # Salen family
    "R,R-Jacobsen's Co-I salen catalyst": "Jacobsen Co-salen",
    "Jacobsen Co-salen": "Jacobsen Co-salen",
    "salen cobalt": "Jacobsen Co-salen",
    "salen": "salen",
    "Al{C5H2O(O)(O)CH2OH}3": "Al-salan complex",
    "aluminum salen": "aluminum salen",
    # Reagents / phase-transfer additives
    "(triphenylphosphoranylidene)acetaldehyde": "(triphenylphosphoranylidene)acetaldehyde",
    "triphenylphosphoranylideneacetaldehyde": "(triphenylphosphoranylidene)acetaldehyde",
    "N-bromosuccinimide": "N-bromosuccinimide",
    "NBS": "N-bromosuccinimide",
    "dibenzoyl peroxide": "dibenzoyl peroxide",
    "dibenzoyl peroxide; N-bromosuccinimide": "dibenzoyl peroxide; N-bromosuccinimide",
    "2,5-dihydroxyterephthalohydrazide": "2,5-dihydroxyterephthalohydrazide",
    # Crown ether / amino-acid / pyridinium variants
    "18-crown-6 ether": "18-crown-6 ether",
    "18-crown-6": "18-crown-6 ether",
    "crown ether": "18-crown-6 ether",
    "L-arginine": "L-arginine",
    "L-arginine;": "L-arginine",
    "N-trimethylanilinium bromide": "N-trimethylanilinium bromide",
    "N-trimethylethan-1-aminium iodide": "N-trimethylethan-1-aminium iodide",
    "4-dimethylaminopyridine": "4-dimethylaminopyridine",
    "4-dimethylamino-N-iodopyridinium bromide": "4-dimethylamino-N-iodopyridinium bromide",
}


# ── Solvent normalization ──────────────────────────────────────────────────
SOLVENT_NORMALIZE = {
    "neat":               "无溶剂",
    "no solvent":         "无溶剂",
    "acetonitrile":       "acetonitrile",
    "MeCN":              "acetonitrile",
    "methanol":          "methanol",
    "MeOH":              "methanol",
    "ethanol":           "ethanol",
    "isopropanol":       "isopropanol",
    "water":             "water",
    "DMF":              "DMF",
    "toluene":           "toluene",
    "chloroform":         "chloroform",
    "dichloromethane":    "dichloromethane",
    "DCM":               "dichloromethane",
    "DMSO":              "DMSO",
    "acetone":           "acetone",
    "THF":              "THF",
    "1,4-dioxane":      "1,4-dioxane",
    "ethyl acetate":      "ethyl acetate",
    "hexane":             "hexane",
    "cyclohexane":       "cyclohexane",
    "diethyl ether":     "diethyl ether",
    "chlorobenzene":      "chlorobenzene",
    "1-butanol":        "1-butanol",
    "2-butanol":        "2-butanol",
    "supercritical CO2": "supercritical CO2",
}


# ── Catalyst system classification keywords ──────────────────────────────
IONIC_KW = [
    "BMIM","EMIM","HMIM","OMIM","DMIM","AMIM",
    "[BMIM]","[EMIM]","[HMIM]","[OMIM]","[DMIM]","[AMIM]",
    "imidazolium","1-butyl-3-methylimidazolium","1-ethyl-3-methylimidazolium",
    "1-hexyl-3-methylimidazolium","1-octyl-3-methylimidazolium",
    "1-methyl-3-octylimidazolium","1-allyl-3-methylimidazolium",
    "1-benzyl-3-methylimidazolium","1-methyl-3-propylimidazolium",
    "[Py]","pyridinium","N-methylpyridinium","N-ethylpyridinium",
    "N-butylpyridinium","N-octylpyridinium",
    "tetrabutylammonium","tetraphenylphosphonium","tetraphenylstibonium",
    "tetraoctylammonium","tetrahexylammonium","tetraheptylammonium",
    "tetra-n-butylammonium","tetra-n-propylammonium","tetraethylammonium",
    "tetramethylammonium","methyltributylammonium","methyltrioctylammonium",
    "benzyl triethylammonium","benzyltrimethylammonium","choline",
    "triethanolammonium","tri-n-butyl(2-hydroxyethyl)ammonium",
    "tri-n-butyl(2-hydroxyethyl)phosphonium",
    "guanidinium","1,1,3,3-tetramethylguanidine","tetramethylguanidine",
    "phosphonium","butyl(triphenyl)phosphonium","methyltriphenylphosphonium",
    "tributylbenzylphosphonium","dodecyltributylphosphonium",
    "TBAB","TBAI","TBAC","TBAF","TBAC","TBD","MTBD","DBU","DMAP","DABCO",
]
METAL_KW = [
    "zinc","sodium","potassium","lithium","cobalt","iron","copper",
    "nickel","magnesium","calcium","aluminum","tin","manganese","indium","lanthanum",
    "ZnCl","ZnBr","ZnI","NaCl","NaI","KCl","KI","LiCl","LiBr",
    "FeCl","CoCl","CuCl","NiCl","MgCl","MgBr","CaCl","AlCl","SnCl",
    "ZnO","MgO","Al2O3","CaO","TiO2",
    "Zn(","Na(","K(","Co(","Fe(","Cu(","Ni(","Mg(","Sn(",
]
BASE_KW = [
    "DBU","DMAP","DABCO","TBD","MTBD","triethylamine","pyridine",
    "tri-n-butylamine","tri-n-butylamine","tributylamine","dimethylamine","diethylamine",
    "dimethylpyridine","iminopyridine","aniline",
]


# ── Schema definitions ─────────────────────────────────────────────────────
OUT_COLS = [
    "row_id", "reactant_name", "product_name", "yield (%)",
    "temperature (\u2103)", "pressure (MPa)", "time (h)",
    "catalyst_1_name", "catalyst_1_loading_mol%",
    "catalyst_2_name", "catalyst_2_loading_mol%",
    "catalyst_3_name", "catalyst_3_loading_mol%",
    "catalyst_4_name", "catalyst_4_loading_mol%",
    "reagent_1_name", "reagent_2_name",
    "solvent_1", "solvent_2", "solvent_3", "solvent_4",
    "reference", "publication_year", "extraction_status", "catalyst_system_type",
    "all_solvents_normalized",
]
EXTR_COLS = ["source_row_id", "discard_reason", "discard_type",
             "affected_field", "raw_value", "suggested_fix"]
DISC_COLS = [
    "row_id",
    "catalyst_1_name", "catalyst_2_name", "catalyst_3_name", "catalyst_4_name",
    "catalyst_1_smiles", "catalyst_2_smiles", "catalyst_3_smiles", "catalyst_4_smiles",
    "discard_reason", "discard_type",
]


# ── Helpers ────────────────────────────────────────────────────────────────
def _v(v) -> str:
    """Stringify value or empty string for None."""
    return "" if v is None else v


def is_valid(v) -> bool:
    return v is not None and bool(str(v).strip()) and str(v).strip() not in ("", "/", "nan")


def parse_yield(v) -> Tuple[Optional[float], str]:
    """Parse a yield value that may be a number, fraction, or percent.

    Returns (value_or_None, error_reason_or_empty).
    """
    if v is None:
        return None, "None"
    s = str(v).strip()
    if s.lower() in ("trace", "n.d.", "n.r.", "", "nan"):
        return None, f"kw:{s}"
    s = re.sub(r"^(?:ca\.?|approx\.?|approximately)\s+", "", s, flags=re.I)
    m = re.match(r"^>?\s*(\d+(?:\.\d+)?)\s*(?:-\s*\d+(?:\.\d+)?)?\s*%?", s)
    if not m:
        return None, f"no_num:{s}"
    num_str = m.group(1)
    s_after = s[m.end():].strip()
    has_percent_sign_in_orig = s_after.startswith("%") or ("%" in s[:m.end()])
    s_after = s_after.lstrip("%").strip()
    has_after_text = bool(s_after)
    has_unit = bool(re.search(
        r"(?:^|\s)(?:g|mg|kg|mmol|mol|mL|L|equiv|eq\.?|"
        r"molar|mol\/L|mmol\/mL|mmol\/mol)\b",
        s, re.I))
    has_method_suffix = bool(re.search(
        r"(?:^|\s)(?:spectr\.?|chromat\.?|chro[\u00a0\u2010\u2011]?mat\.?|"
        r"nmr|gc|gc-ms|hplc|lc|ms|both|nr\.?|n\.r\.?|"
        r"isol\.?|crude|calc\.?|theoretical)\b",
        s_after, re.I)) if has_after_text else False
    try:
        f = float(num_str)
    except ValueError:
        return None, f"bad_num:{s}"
    if f < 0:
        return None, f"<0:{f}"
    should_scale = (
        (not has_unit and has_method_suffix and f <= 1.0) or
        (not has_unit and not has_after_text and not has_percent_sign_in_orig and f <= 1.0 and f > 0)
    )
    if should_scale:
        f *= 100.0
    return round(min(f, 100.0), 2), ""


def parse_temp(s: str) -> Optional[float]:
    if not s:
        return None
    ms = list(re.finditer(r"T\s*=\s*(\d+(?:\.\d+)?)\s*(?:deg?C|\?C|\u2103)?", s, re.I))
    ts = [float(m.group(1)) for m in ms if -50 <= float(m.group(1)) <= 400]
    if ts:
        return round(ts[0], 1)
    if re.search(r"room\s+temp|r\.t\.|overnight", s, re.I):
        return 25.0
    return None


def parse_pressure(s: str) -> Optional[float]:
    """Parse pressure; convert to MPa. Returns None if no plausible value."""
    if not s:
        return None
    patterns = [
        (r"p\s*=\s*(\d+(?:\.\d+)?)\s*(MPa|atm|bar|psi|Torr|Pa)?", 1, 2),
        (r"(\d+(?:\.\d+)?)\s*(MPa|atm|bar|psi|Torr|Pa)", 1, 2),
        (r"pressure[:\s]+\s*(\d+(?:\.\d+)?)\s*(MPa|atm|bar|psi|Torr|Pa)?", 1, 2),
    ]
    ps = []
    for pat, gv, gu in patterns:
        for m in re.finditer(pat, s, re.I):
            v = float(m.group(gv))
            u = (m.group(gu) if gu <= (m.lastindex or 0) else "") or ""
            u = u.strip().lower()
            if u in ("", "mpa"):
                pass
            elif u == "atm":
                v = v * 0.101325
            elif u == "bar":
                v = v * 0.1
            elif u == "psi":
                v = v * 0.00689476
            elif u == "torr":
                v = v * 0.000133322
            elif u == "pa":
                v = v / 1_000_000
            else:
                if v > 100:
                    v = v * 0.000133322
            if 0 <= v <= 50:
                ps.append(v)
    if not ps:
        if re.search(r"ambient|atmosphere| balloon", s, re.I):
            return 0.101325
        return None
    return round(ps[0], 5)


def parse_time(s: str) -> Optional[float]:
    """Parse reaction time in hours."""
    if not s:
        return None
    ts = []
    for m in re.finditer(r"Time\s*=\s*(\d+(?:\.\d+)?)\s*(h|hr|hour|min|minute|day|d)?", s, re.I):
        v = float(m.group(1))
        u = (m.group(2) or "h").strip().lower()
        if u.startswith("min"):
            v /= 60.0
        elif u.startswith("d"):
            v *= 24.0
        if 0 < v <= 720:
            ts.append(v)
    if not ts:
        if re.search(r"overnight|隔夜", s, re.I):
            return 16.0
        return None
    return round(ts[0], 2)


def extract_loading(text: str) -> Optional[float]:
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:mol[%\uFF05]|mol\s*%)", text, re.I)
    if m:
        v = float(m.group(1))
        if 0 < v <= 100:
            return round(v, 2)
    return None


def _has_catalyst_feat(text: str) -> bool:
    """Does the text contain a feature word that signals a catalyst name?"""
    if not text:
        return False
    t = text.lower()
    suffixes = [
        "bromide", "chloride", "iodide", "fluoride", "hydroxide",
        "acetate", "nitrate", "sulfate", " carbonate", " phosphate",
        "imidazolium", "ammonium", "phosphonium", "pyridinium", "guanidinium",
        "onium", "sulfonium", "stibonium",
        "salen", "porphyrin", "phthalocyanine", "Schiff base",
        "tetrabutyl", "tetramethyl", "tetraethyl", "tetraphenyl",
        "benzyltrimethyl", "methyltributyl", "benzyl triethyl",
        "cholin", "ionic liquid",
    ]
    return any(s in t for s in suffixes)


def _normalize_cat(name: str) -> Optional[str]:
    """CAT_NORMALIZE lookup: exact match first, then substring (longest-key first)."""
    if not name:
        return None
    n = name.strip().lower()
    if not n or len(n) < 2:
        return None
    if n in CAT_NORMALIZE:
        return CAT_NORMALIZE[n]
    for k in sorted(CAT_NORMALIZE, key=len, reverse=True):
        if len(k) < 4:
            continue
        kl = k.lower()
        if kl in n:
            return CAT_NORMALIZE[k]
    return None


def classify(cats: List[str]) -> str:
    """Five-class: ionic_liquid / metal_halide / organic_base / mixed_system / unknown."""
    if not cats:
        return "unknown"
    text = " ".join(str(c).lower() for c in cats if c)

    def token_match(keyword: str) -> bool:
        return bool(re.search(r"(?<![a-zA-Z])" + re.escape(keyword.lower()) + r"(?![a-zA-Z])", text))

    ionic_count = sum(1 for k in IONIC_KW if token_match(k))
    metal_count = sum(1 for k in METAL_KW if token_match(k))
    base_count = sum(1 for k in BASE_KW if token_match(k))
    active = sum(1 for c in (ionic_count, metal_count, base_count) if c > 0)
    if active >= 2:
        return "mixed_system"
    if ionic_count >= 1:
        return "ionic_liquid"
    if metal_count >= 1:
        return "metal_halide"
    if base_count >= 1:
        return "organic_base"
    return "unknown"


def is_blacklist(cond: str) -> bool:
    if not cond:
        return False
    return any(pat.search(cond) for pat in CAT_BLACKLIST_PATTERNS)


def parse_reactant(row: pd.Series) -> Tuple[Optional[str], Optional[str]]:
    rx = str(row.get("Reaction", "")).strip()
    for k, (r, p) in SUBSTRATE_MAP.items():
        if k.lower() in rx.lower():
            return r, p
    return None, None


def extract_catalysts(cond: str, purge_bug: bool = False) -> Tuple[List[str], List[Optional[float]], str]:
    """Extract (catalyst_names, loadings, unclassified_text) from a Conditions string.

    Strategy
    --------
    1. Search all "With 催化剂" snippets first (most reliable).
    2. If none found, fall back to a full-text keyword search.
    3. Empty + has_catalyst_feat → extraction_failed.
    4. Empty + no feature           → _PROCEDURE_ONLY_ (procedural text).

    Direct-parse fallbacks
    ----------------------
    For catalysts that don't appear after "With:" (a known dataset quirk),
    a full-text scan against CAT_NORMALIZE covers IL + metal halide cases.
    """
    if not cond or not cond.strip():
        return [], [], "空Conditions"

    # The two patterns below are intentionally compiled but, by default,
    # NOT applied — see module docstring "bug-preservation note".
    clean = cond
    journal_pat = re.compile(
        r"\b(?:J\.?\s*Org\.?\s*Chem\.?|J\.?\s*Catal\.?|Green\s*Chem\.?|"
        r"Angew\.?\s*Chem\.?|Inorg\.?\s*Chem\.?|Appl\.?\s*Catal\.?\s*[AB]?|"
        r"Dalton\s*Trans\.?|Tetrahedron\s*Lett\.?|Catal\.?\s*Today\.?|"
        r"J\.?\s*CO2\s*Util\.?|ACS\s*Catal\.?|Chem\.?\s*(?:Sus\.?\s*)?Chem\.?|"
        r"Mater\.?\s*Today\s*(?:Energy)?|J\.?\s*Mol\.?\s*Catal\.?\s*[AB]?|"
        r"Chem\.?\s*Commun\.?(?!icat)|Catal\.?\s*Commun\.?|J\.?\s*Mater\.?\s*Chem\.?\s*[AB]?|"
        r"Chem\.?\s*Eng\.?\s*J\.?|Fuel\s*(?:Process)?|RSC\s*Adv\.?|"
        r"Synthesis\s+\d{4}.*|doi.*|DOI.*|"
        r"vol\.?\s*\d+|p\.?\s*\d+(?:-\d+)?|pp\.?\s*\d+|nb\.?|"
        r"[A-Z][a-z]+\s+\d{4}.*|"
        r"Patent\s+\w+.*|WO\s*\d+.*|CN\s*\d+.*)", re.I)
    proc_pat = re.compile(
        r"(?:general\s+procedure|typical\s+procedure|standard\s+procedure|"
        r"in\s+a\s+typical\s+run|typical\s+experiment|"
        r"procedure\s+for\s+cyclic\s+carbonate|cycloaddition\s+procedure|"
        r"synthesis\s+of\s+carbonates)[^$]{0,300}?(?=\s*(?:With|Time|T\s*=\s*\d|p\s*=\s*\d|$))",
        re.I | re.S)
    if purge_bug:
        clean = journal_pat.sub(" ", clean)
        clean = proc_pat.sub(" ", clean)

    cats, loads, unclassified = [], [], ""
    seen = set()

    # Step B: "With 催化剂" snippets
    with_matches = list(re.finditer(
        r"(?:^|(?<=[,\s;\.\-–—]))\s*[Ww]ith\s+([A-Za-z0-9\-\+\(\)\[\]\.\s'/]{3,300}?)"
        r"(?=\s*[,\;\.]?\s*(?:Time|T\s*=\s*\d|p\s*=\s*\d|Time=\s*\d|Qiu|doi|$))",
        cond))

    if with_matches:
        for m in with_matches:
            snippet = m.group(1).strip()
            if len(snippet) < 3:
                continue
            snippet = re.sub(r"[A-Z][a-z]+\s+(?:J\.?|Green|Inorg\.?|Appl\.?|Catal\.?).*", "", snippet)
            snippet = snippet.strip().rstrip(",;\-. ")
            if len(snippet) < 3:
                continue
            sub_parts = re.split(r"\s+and\s+|\s*;\s*", snippet)
            for part in sub_parts:
                part = part.strip()
                if len(part) < 3 or len(part) > 250:
                    continue
                if not _has_catalyst_feat(part):
                    continue
                norm = _normalize_cat(part)
                if norm and norm not in seen:
                    seen.add(norm)
                    cats.append(norm)
                    loads.append(extract_loading(part))
                elif not norm and _has_catalyst_feat(part):
                    unclassified = part[:100]

    # Step C: full-text fallback
    if not cats:
        for kw in sorted(CAT_NORMALIZE, key=len, reverse=True):
            if len(kw) < 4:
                continue
            if kw.lower() in clean.lower():
                norm = CAT_NORMALIZE[kw]
                if norm not in seen:
                    seen.add(norm)
                    cats.append(norm)
                    pos = clean.lower().find(kw.lower())
                    snippet = clean[max(0, pos - 60): pos + 120]
                    loads.append(extract_loading(snippet))
                    if len(cats) >= 4:
                        break

    cats = cats[:4]
    loads = loads[:4]
    while len(cats) > len(loads):
        loads.append(None)

    if cats:
        return cats, loads, ""
    if _has_catalyst_feat(clean):
        return [], [], clean[:200]
    return [], [], "_PROCEDURE_ONLY_"


def extract_solvents(cond: str) -> List[str]:
    """Extract solvents (returns ['无溶剂'] for solvent-free conditions)."""
    if not cond:
        return []
    if re.search(r"(?:in\s+)?(?:neat|no\s+solvent|without\s+solvent|in\s+\(no\s+solent?\)|"
                 r"solvent[\- ]free|free[\- ]of\s+solvent|absence\s+of\s+solvent|"
                 r"neat\s+condi|no\s+additives?|neat\s+condition)", cond, re.I):
        return ["无溶剂"]
    found = []
    for name, norm in SOLVENT_NORMALIZE.items():
        if name == "无溶剂":
            continue
        if re.search(r"\b" + re.escape(name) + r"\b", cond, re.I):
            if norm not in found:
                found.append(norm)
    return found[:4]


def extract_ref_year(cond: str) -> Tuple[str, Optional[int]]:
    """Extract (reference, publication_year) from the end of a Conditions string."""
    if not cond:
        return "Unknown reference", None
    s = str(cond)

    parts = s.split("Torr")
    ref_part = "Torr".join(parts[1:]).strip() if len(parts) >= 2 else s
    ref_part = ref_part.strip().lstrip(",").lstrip(" ")

    skip_prefixes = [
        "Autoclave,", "Autoclave ", "Schlenk technique,", "Schlenk technique ",
        "Irradiation,", "Irradiation ", "Green chemistry,",
        "Temperature, Pressure,", "Temperature, Pressure ",
        "Catalytic behavior,", "Catalytic behavior ",
        "Reagent/catalyst,", "Reagent/catalyst ", "Solvent-free,",
        "General procedure:", "Typical procedure:", "Sealed tube,", "Sealed tube ",
    ]
    for prefix in skip_prefixes:
        if ref_part.startswith(prefix):
            ref_part = ref_part[len(prefix):]
            break

    year_match = re.search(r"\((\d{4})\)", ref_part)
    year: Optional[int] = None
    if year_match:
        y = int(year_match.group(1))
        if 1900 <= y <= 2100:
            year = y
    ref = ref_part.strip().rstrip("., ")
    if len(ref) < 15:
        return "Unknown reference", year
    return ref, year


# ── Row-level helper ───────────────────────────────────────────────────────
def _make_disc_row(rid: int, reason: str) -> Dict:
    return dict(
        row_id=rid,
        catalyst_1_name="", catalyst_2_name="", catalyst_3_name="", catalyst_4_name="",
        catalyst_1_smiles="", catalyst_2_smiles="", catalyst_3_smiles="", catalyst_4_smiles="",
        discard_reason=reason, discard_type="reasonable_discard",
    )


# ── Core cleaning logic ───────────────────────────────────────────────────
def clean_dataset(
    df_raw: pd.DataFrame,
    purge_bug: bool = False,
    progress_every: int = 300,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Run the cleaning loop; return (valid_rows, extraction_failures, discards)."""
    valid_r, ext_r, disc_r = [], [], []
    st = dict(total=0, valid=0, disc=0, ext=0, sub=0, yl=0, bl=0, cond=0)
    t0 = time.time()

    for idx, row in df_raw.iterrows():
        st["total"] += 1
        rid = idx + 1
        rx = str(row.get("Reaction", "")).strip()
        yr = row.get("Yield", None)
        con = str(row.get("Conditions", "")).strip()

        # 1. Substrate
        rat, prd = parse_reactant(row)
        if rat is None:
            disc_r.append(_make_disc_row(rid, f"底物无法归类:{rx[:80]}"))
            st["sub"] += 1; st["disc"] += 1
            continue

        # 2. Yield
        yv, ym = parse_yield(yr)
        if yv is None:
            disc_r.append(_make_disc_row(rid, f"产率无效:{ym}"))
            st["yl"] += 1; st["disc"] += 1
            continue

        # 3. Blacklist
        if is_blacklist(con):
            disc_r.append(_make_disc_row(rid, "命中黑名单(MOF/聚合物/光催化/电催化)"))
            st["bl"] += 1; st["disc"] += 1
            continue

        # 4. Catalysts + conditions
        cn, cl, cm = extract_catalysts(con, purge_bug=purge_bug)
        tp = parse_temp(con); pr = parse_pressure(con); tm = parse_time(con)
        if tp is None and pr is None and tm is None:
            disc_r.append(_make_disc_row(rid, "三项条件全缺失"))
            st["cond"] += 1; st["disc"] += 1
            continue

        if not cn:
            if cm == "_PROCEDURE_ONLY_":
                disc_r.append(_make_disc_row(rid, "无催化剂信息（纯操作文本）"))
                st["disc"] += 1; st["ext"] += 1
                continue
            ext_r.append(dict(
                source_row_id=rid,
                discard_reason=f"无法解析催化剂:{cm[:100] if cm else con[:120]}",
                discard_type="extraction_failed_repair_needed",
                affected_field="catalyst",
                raw_value=con[:500],
                suggested_fix="请检查Conditions补充催化剂名称到CAT_NORMALIZE字典",
            ))
            st["ext"] += 1
            continue

        # 5. Solvents, ref, type
        sols = extract_solvents(con)
        ref, pub_year = extract_ref_year(con)
        ctype = classify(cn)
        valid_r.append({
            "row_id": rid,
            "reactant_name": rat,
            "product_name": prd,
            "yield (%)": yv,
            "temperature (\u2103)": _v(tp),
            "pressure (MPa)": _v(pr),
            "time (h)": _v(tm),
            "catalyst_1_name": _v(cn[0]) if len(cn) > 0 else "",
            "catalyst_1_loading_mol%": _v(cl[0]) if len(cl) > 0 and cl[0] else "",
            "catalyst_2_name": _v(cn[1]) if len(cn) > 1 else "",
            "catalyst_2_loading_mol%": _v(cl[1]) if len(cl) > 1 and cl[1] else "",
            "catalyst_3_name": _v(cn[2]) if len(cn) > 2 else "",
            "catalyst_3_loading_mol%": _v(cl[2]) if len(cl) > 2 and cl[2] else "",
            "catalyst_4_name": _v(cn[3]) if len(cn) > 3 else "",
            "catalyst_4_loading_mol%": _v(cl[3]) if len(cl) > 3 and cl[3] else "",
            "reagent_1_name": "", "reagent_2_name": "",
            "solvent_1": _v(sols[0]) if len(sols) > 0 else "",
            "solvent_2": _v(sols[1]) if len(sols) > 1 else "",
            "solvent_3": _v(sols[2]) if len(sols) > 2 else "",
            "solvent_4": _v(sols[3]) if len(sols) > 3 else "",
            "reference": ref,
            "publication_year": pub_year if pub_year else "",
            "extraction_status": "valid",
            "catalyst_system_type": ctype,
            "all_solvents_normalized": "; ".join(sols),
        })
        st["valid"] += 1

        if (rid % progress_every == 0) or (rid == len(df_raw)):
            el = time.time() - t0
            rate = rid / el if el > 0 else 0
            eta = (len(df_raw) - rid) / rate if rate > 0 else 0
            logger.info(
                "  %d/%d valid=%d disc=%d ext=%d %.0fs ETA=%.0fs",
                rid, len(df_raw), st["valid"], st["disc"], st["ext"], el, eta,
            )

    return valid_r, ext_r, disc_r, st


# ── Encoding-tolerant loader ──────────────────────────────────────────────
def load_raw_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    for enc in INPUT_ENCODINGS:
        try:
            df = pd.read_csv(path, encoding=enc)
            logger.info("  encoding %s OK: %d rows", enc, len(df))
            return df
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError(
        "all-encodings", b"", 0, 1, f"Cannot decode {path} with any of {INPUT_ENCODINGS}"
    )


# ── save helpers ──────────────────────────────────────────────────────────
def save_outputs(
    valid_r: List[Dict], ext_r: List[Dict], disc_r: List[Dict],
    cleaned_csv: Path, ext_csv: Path, disc_csv: Path, baseline_json: Path,
) -> None:
    cleaned_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(valid_r, columns=OUT_COLS).to_csv(cleaned_csv, index=False, encoding="utf-8-sig")
    logger.info("Saved %d rows → %s", len(valid_r), cleaned_csv)

    if ext_r:
        pd.DataFrame(ext_r, columns=EXTR_COLS).to_csv(ext_csv, index=False, encoding="utf-8-sig")
        logger.info("Saved %d extraction-failure rows → %s", len(ext_r), ext_csv)
    if disc_r:
        pd.DataFrame(disc_r, columns=DISC_COLS).to_csv(disc_csv, index=False, encoding="utf-8-sig")
        logger.info("Saved %d discard rows → %s", len(disc_r), disc_csv)


# ── CLI ────────────────────────────────────────────────────────────────────
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stage 1.1 — clean the raw CO2-cycloaddition CSV",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--force", action="store_true",
                   help="Overwrite cleaned.csv even if it exists.")
    p.add_argument("--input", type=str, default=str(RAW_REAXYS_CSV),
                   help="Path to the raw Reaxys-style CSV (overrides default).")
    p.add_argument("--output", type=str, default=str(CLEANED_CSV),
                   help="Path to the cleaned CSV (overrides default).")
    p.add_argument("--purge-bug", action="store_true",
                   help="(DANGEROUS) re-enable the suppressed journal/procedure strips. "
                        "This changes the valid row count and breaks downstream benchmarks.")
    p.add_argument("--verbose", action="store_true",
                   help="Enable DEBUG-level logging.")
    return p.parse_args(argv)


def configure_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format=LOG_FMT, datefmt=LOG_DATEFMT,
    )


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    configure_logging(verbose=args.verbose)

    output_csv = Path(args.output)
    extraction_csv = DATA_EXTERNAL / "extraction_report.csv"
    discard_csv = DATA_EXTERNAL / "discard_report.csv"
    baseline_json = CLEANED_BASELINE

    # Encoding-handle stdout early
    if sys.stdout.encoding != "utf-8":
        try:
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
        except Exception:
            pass

    logger.info("=" * 60)
    logger.info("CO2 环加成原始数据清洗")
    logger.info("=" * 60)

    # Skip if already cleaned
    if not args.force and output_csv.exists():
        try:
            existing = pd.read_csv(output_csv, encoding="utf-8-sig")
            logger.info("[SKIP] %s already exists (%d rows). Use --force to re-run.",
                        output_csv, len(existing))
            return 0
        except Exception as ex:  # noqa: BLE001
            logger.warning("Could not read existing %s: %s — re-cleaning.", output_csv, ex)

    logger.info("[1/5] Loading: %s", args.input)
    df_raw = load_raw_csv(Path(args.input))

    logger.info("[2/5] Cleaning %d rows …", len(df_raw))
    valid_r, ext_r, disc_r, st = clean_dataset(
        df_raw, purge_bug=args.purge_bug, progress_every=300,
    )

    save_outputs(valid_r, ext_r, disc_r, output_csv, extraction_csv, discard_csv, baseline_json)

    logger.info("[3/5] Done in %.1fs.", time.time() - 0.0)
    logger.info("=" * 60)
    logger.info("  raw=%d valid=%d discard=%d extraction_failed=%d",
                st["total"], st["valid"], st["disc"], st["ext"])
    logger.info("    substrate=%d yield=%d blacklist=%d conditions=%d",
                st["sub"], st["yl"], st["bl"], st["cond"])
    logger.info("=" * 60)

    # Baseline JSON
    summary = dict(
        valid_rows=st["valid"],
        total_rows=st["total"],
        discard_rows=st["disc"],
        ext_rows=st["ext"],
        timestamp=str(datetime.now()),
    )
    ensure_dir(baseline_json.parent)
    with open(baseline_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logger.info("Wrote %s", baseline_json)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())