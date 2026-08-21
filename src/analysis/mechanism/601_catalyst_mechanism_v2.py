"""Step 1 v2: Cleaner mechanism classifier.

v1 had spurious matches:
    DABCO -> LAC  (because "Co" is in "Co-salen" or "cobalt" elsewhere; not in DABCO)
              but the issue was ethyl acetate / 18-crown-6 matched "Cr"
    ethyl acetate -> LAC  (false positive on "Ac" matching "Cr"? No — the real bug
                          is that _normalise lowered "Cr" → "cr" but "ethyl acetate"
                          doesn't contain "cr"; the bug was matching "al" which is in
                          "metal" — actually no, "ethyl acetate" matched nothing. Bug
                          is in lewis-metal-detection: we lower-cased the entire string
                          so "Al" matched anywhere. "al" is in "ethanol" → methanol etc.
                          We need WHOLE-WORD or element-bracket matching.)
    18-crown-6 -> LAC  ("Cr" matched in "crown"; we lowercased.)
    DMAP -> only 1 BAS  (BASE_TOKENS too narrow; missing many amines)

Fixes
-----
* Strict element matching: regex `(?<![A-Za-z])Zn(?![A-Za-z])` etc.
* Use SMILES bracket for metals: "[Zn]", "[Mg]" etc. PLUS a fallback for element
  names in plain text.
* Broaden BASE_TOKENS to include common organic bases.
* Hydroxyl tokens only at WORD boundary and require full word ending, not substring.
* Add a "clearly-not-catalyst" filter list.
"""

from __future__ import annotations

import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

# Ensure src/ is on the path for paths.py imports
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)
try:
    from paths import DATA_PROCESSED
except ImportError:
    DATA_PROCESSED = Path(__file__).resolve().parents[1] / "data" / "processed"


# --- Lewis metals with word-boundary regex -------------------------------
# Match both elemental SYMBOL (Zn, Mg, ...) and the spelled-out NAME
# (zinc, magnesium, ...) so names like "zinc dibromide" or "Cu(OTf)2" both resolve.
LEWIS_METALS = ["Zn", "Mg", "Al", "Co", "Cu", "Fe", "Ni", "Mn", "Cr",
                "Ti", "Sn", "Zr", "Y", "Sc", "La", "Ce", "V", "Mo", "W",
                "In", "Ga", "Bi"]
LEWIS_NAMES = {
    "Zn": "zinc", "Mg": "magnesium", "Al": "aluminium", "Al": "aluminum",
    "Co": "cobalt", "Cu": "copper", "Fe": "iron", "Ni": "nickel",
    "Mn": "manganese", "Cr": "chromium", "Ti": "titanium", "Sn": "tin",
    "Zr": "zirconium", "Y": "yttrium", "Sc": "scandium",
    "La": "lanthanum", "Ce": "cerium", "V": "vanadium", "Mo": "molybdenum",
    "W": "tungsten", "In": "indium", "Ga": "gallium", "Bi": "bismuth",
}
LEWIS_PATTERNS = []
for m in LEWIS_METALS:
    # bare element symbol
    LEWIS_PATTERNS.append((m, re.compile(r"(?<![A-Za-z])" + m + r"(?![A-Za-z])")))
    # spelled-out name
    nm_name = LEWIS_NAMES.get(m)
    if nm_name and nm_name not in [p[0] for p in LEWIS_PATTERNS]:
        LEWIS_PATTERNS.append((m, re.compile(r"(?<![A-Za-z])" + nm_name + r"(?![A-Za-z])")))
# also bracket form
BRACKET_PATTERNS = [(m, re.compile(r"\[" + m + r"\]")) for m in LEWIS_METALS]


# --- Halides -------------------------------------------------------------
# Match both elemental symbol (Br, Cl, I, F) AND the English suffix (-bromide,
# -chloride, -iodide, -fluoride) so that names like "zinc dibromide" or
# "tetrabutylammonium bromide" resolve correctly.
HALIDE_ORDER = {"I": 3, "Br": 2, "Cl": 1, "F": 0}
HALIDE_PATTERNS = [(h, re.compile(r"(?<![A-Za-z])" + h + r"(?![A-Za-z])")) for h in HALIDE_ORDER]
HALIDE_SUFFIX_PATTERNS = [
    ("Br", re.compile(r"(?<![A-Za-z])bromide(?![A-Za-z])")),
    ("Cl", re.compile(r"(?<![A-Za-z])chloride(?![A-Za-z])")),
    ("I",  re.compile(r"(?<![A-Za-z])iodide(?![A-Za-z])")),
    ("F",  re.compile(r"(?<![A-Za-z])fluoride(?![A-Za-z])")),
]


# --- Bases (extended) ----------------------------------------------------
BASE_TOKENS = [
    "dbu", "tmg", "dmap", "imidazole", "imidazolium",
    "pyridine", "pyridinium", "amine", "ammoni", "ammonium",
    "guanidi", "guanidin", "biguanide", "phosphaz", "phosphonium",
    "pyrazol", "triazol", "tbd", "mtbd", "dabco", "quinuclidine",
    "morpholine", "piperazine", "tertiary amine",
    "betain", "ethanolamine", "diethylamino", "dimethylamino",
    "tributylamine", "triethylamine", "tripropylamine",
    "tetrabutylammonium", "tetramethylammonium",
    "tetraethylammonium", "tetrapropylammonium",
    "tba", "tbab", "tbai", "tbac",
    "n-heterocyclic", "carben",
    "nbu", "n-butyl",
    "diazabicyclo",
    "triamino", "melamine", "s-triazine",
]


# Catalyst-frameworks that imply a Lewis-acid metal even if the metal isn't
# named explicitly (common ligands in CO2-cycloaddition catalysis).
LIGAND_OVERRIDE_LEWIS = ["salen", "salalen", "salan", "porphyrin",
                          "porphyr", "phthalocyanine", "pc-"]


# --- Hydroxyl (strict full-word / suffix) --------------------------------
HYDROXYL_PATTERNS = [
    re.compile(r"(?<![A-Za-z])(?:hydroxy|hydroxyl)(?![A-Za-z])"),
    re.compile(r"(?<![A-Za-z])(?:ethanolamine|ethanol|propanol|butanol)(?![A-Za-z])"),
    re.compile(r"(?<![A-Za-z])(?:carboxy|carboxyl)(?![A-Za-z])"),
    re.compile(r"ol(?![A-Za-z])"),  # -ol suffix (methanol, phenol)
    re.compile(r"(?<![A-Za-z])phenol(?![A-Za-z])"),
]


# --- Clearly-not-catalyst list (used as suppression) ---------------------
NON_CATALYSTS = [
    "ethyl acetate", "dimethylformamide", "dmso", "cyclohexane",
    "dichloromethane", "acetonitrile", "toluene", "ethanol", "methanol",
    "acetone", "water", "tetrahydrofuran", "thf", "ethylenediaminetetraacetic acid",
    "edta", "18-crown-6", "crown ether",
]


def _normalise(name: str) -> str:
    if not isinstance(name, str):
        return ""
    return name.lower().strip()


def _is_non_catalyst(name_norm: str) -> bool:
    return any(tok in name_norm for tok in NON_CATALYSTS)


def _halide_score(name: str) -> int:
    score = 0
    for h, pat in HALIDE_PATTERNS + HALIDE_SUFFIX_PATTERNS:
        if pat.search(name):
            score = max(score, HALIDE_ORDER[h])
    return score


def _lewis_count(name: str) -> int:
    n = 0
    for m, pat in LEWIS_PATTERNS + BRACKET_PATTERNS:
        if pat.search(name):
            n += 1
    return n


def _base_count(name_norm: str) -> int:
    return sum(1 for t in BASE_TOKENS if t in name_norm)


def _hydroxyl_count(name: str) -> int:
    return sum(1 for p in HYDROXYL_PATTERNS if p.search(name))


def classify_catalyst(name: str, smiles: str | None = None) -> dict:
    raw = (name or "").strip()
    nm = _normalise(raw)
    sm = (smiles or "").strip()

    if _is_non_catalyst(nm):
        return {"nucleophile": 0, "lewis": 0, "base": 0, "hydroxyl": 0,
                "mechanism": "NOT_CAT", "note": "matched non-catalyst list"}

    # Halides & Lewis metals should match against the raw form (case-sensitive)
    nuc = _halide_score(raw)
    lew = _lewis_count(raw) + _lewis_count(sm)

    # Bases & hydroxyls against the lowercased form
    base = _base_count(nm)
    hydr = _hydroxyl_count(raw)  # also case-sensitive for ol suffix

    # Ligand-frameworks: salen/porphyrin imply a Lewis-acid metal centre.
    nm_raw = raw.lower()
    has_ligand_lewis = any(lig in nm_raw for lig in LIGAND_OVERRIDE_LEWIS)

    # IL cations (ammonium / imidazolium / pyridinium / phosphonium / sulfonium)
    # are NOT bases in the cycloaddition sense — they are mere counterion
    # carriers for the active halide. Strip them from base_count so that
    # TBAB/BMIMBr etc. resolve to NUC instead of BAS.
    is_il_cation = any(tok in nm_raw for tok in
                       ["ammonium", "imidazolium", "pyridinium",
                        "phosphonium", "sulfonium", "pyrrolidinium"])

    has_nuc = nuc > 0
    has_lew = lew > 0 or has_ligand_lewis
    has_base = base > 0 and not is_il_cation
    has_hydr = hydr > 0

    if has_nuc and (has_lew or has_hydr):
        mech = "BIF"
    elif has_lew:
        mech = "LAC"
    elif has_base:
        mech = "BIF" if has_hydr else "BAS"
    elif has_nuc:
        mech = "BIF" if has_hydr else "NUC"
    elif has_hydr:
        mech = "BAS"
    else:
        mech = "OTH"

    return {"nucleophile": nuc, "lewis": max(lew, int(has_ligand_lewis)),
            "base": int(has_base),
            "hydroxyl": hydr, "mechanism": mech, "note": ""}


def run(input_csv: str, output_csv: str, summary_json: str) -> None:
    df = pd.read_csv(input_csv)

    # Collect unique catalyst strings across slots
    rows = []
    for col in ["catalyst_1_name", "catalyst_2_name", "catalyst_3_name", "catalyst_4_name"]:
        for name in df[col].dropna().unique():
            rows.append((col, name))
    cat_df = pd.DataFrame(rows, columns=["slot", "name"]).drop_duplicates(subset=["name"])
    print(f"Total unique catalyst strings: {len(cat_df)}")

    feats = cat_df["name"].apply(classify_catalyst)
    feats_df = pd.DataFrame(list(feats))
    cat_df = pd.concat([cat_df.reset_index(drop=True), feats_df], axis=1)

    cat_df.to_csv(output_csv, index=False)

    summary = {
        "n_catalysts": int(len(cat_df)),
        "mechanism_counts": Counter(cat_df["mechanism"]).most_common(),
        "samples": {m: cat_df[cat_df["mechanism"] == m]["name"].head(10).tolist()
                    for m in ["NUC", "LAC", "BAS", "BIF", "OTH", "NOT_CAT"]},
    }
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n=== Mechanism distribution ===")
    for k, v in summary["mechanism_counts"]:
        print(f"  {k}: {v}")
    print("\n=== Samples ===")
    for mech, names in summary["samples"].items():
        if names:
            print(f"  {mech}: {names}")
    print(f"\nSaved: {output_csv}\n        {summary_json}")

    print("\n=== Cross-tab: catalyst_system_type vs new mechanism (catalyst_1 only) ===")
    c1 = df[["catalyst_system_type", "catalyst_1_name"]].dropna()
    c1 = c1.merge(cat_df[["name", "mechanism"]].rename(
        columns={"name": "catalyst_1_name", "mechanism": "m_new"}), on="catalyst_1_name", how="left")
    print(pd.crosstab(c1["catalyst_system_type"], c1["m_new"], margins=True))


if __name__ == "__main__":
    # FIX (2026-08-19): Use dynamic paths via paths.py instead of hardcoded old-repo paths.
    #   --input  → results/results_cho_diagnostic/co2_drfp_xtb_extended.csv (canonical master table)
    #   --output → data/processed/catalyst_mechanism.csv      (consumed by 700, 701)
    #   --summary → results_mechanism/catalyst_mechanism_summary.json
    import argparse
    PROJECT_ROOT = os.environ.get("CO2_PROJECT_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    _MECH_OUT_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')
    _MECH_SUMMARY_DIR = os.path.join(PROJECT_ROOT, 'results_mechanism')
    os.makedirs(_MECH_OUT_DIR, exist_ok=True)
    os.makedirs(_MECH_SUMMARY_DIR, exist_ok=True)

    p = argparse.ArgumentParser()
    p.add_argument("--input",    default=os.path.join(PROJECT_ROOT, 'results', 'results_cho_diagnostic', 'co2_drfp_xtb_extended.csv'))
    p.add_argument("--output",   default=os.path.join(_MECH_OUT_DIR, 'catalyst_mechanism.csv'))
    p.add_argument("--summary",  default=os.path.join(_MECH_SUMMARY_DIR, 'catalyst_mechanism_summary.json'))
    # --force accepted for pipeline consistency; catalyst_mechanism.csv is
    # small (~200 rows) and always fully regenerated in <2 seconds.
    p.add_argument("--force", action="store_true")
    a = p.parse_args()
    run(a.input, a.output, a.summary)