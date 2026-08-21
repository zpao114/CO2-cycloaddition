"""102_smiles.py — add SMILES columns to cleaned.csv, producing co2_smiles.csv.

Inputs
------
cleaned.csv                                 (output of 101_clean.py)
    - reactant_name, product_name, catalyst_*_name, solvent_*, no SMILES

Outputs
-------
co2_smiles.csv
    - cleaned.csv + 11 SMILES columns: reactant_smiles, product_smiles,
      catalyst_{1..4}_smiles, solvent_{1..4}_smiles, RXN_SMILES
smiles_baseline.json                        (diagnostic only)

Strategy
--------
1. Canonical SMILES lookup for the 5 substrates and ~94 catalysts found in
   the dataset.
2. RDKit name->SMILES fallback for molecules not in the static lookup.
3. RXN_SMILES = '<reactants>.<catalysts>.<solvents>>>products'.

Idempotent: re-running overwrites co2_smiles.csv cleanly.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import sys
from typing import Dict, List, Optional

PROJECT_ROOT = os.environ.get("CO2_PROJECT_ROOT", r"D:\machine-learning\CO2-cycloaddition")
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import pandas as pd

# Force UTF-8 stdout on Windows PowerShell (cp936 default).
# Guarded so importing this module from a notebook/Jupyter doesn't re-wrap.
if sys.stdout.encoding and sys.stdout.encoding.lower().replace("-", "") != "utf8":
    try:
        if hasattr(sys.stdout, "buffer"):
            sys.stdout = io.TextIOWrapper(
                sys.stdout.buffer,
                encoding="utf-8",
                errors="replace",
                line_buffering=not sys.stdout.isatty(),
            )
    except Exception:  # pragma: no cover
        pass

from src.paths import (  # noqa: E402  (after sys.stdout reconfigure)
    CLEANED_CSV,
    SMILES_BASELINE,
    SMILES_CSV,
)

try:
    from rdkit import Chem
    RDKIT_AVAILABLE = True
except Exception:
    RDKIT_AVAILABLE = False
    Chem = None  # type: ignore


SUBSTRATE_SMILES: Dict[str, str] = {
    "Styrene oxide":              "O[C@H]1C(c2ccccc2)CO1",
    "Propylene oxide":            "CC1CO1",
    "Epichlorohydrin":            "CC1COC1Cl",
    "Cyclohexene oxide":          "C1CCC2C1C2O",
    "Isopropyl glycidyl ether":   "CC(C)COCC1CO1",
}

PRODUCT_SMILES: Dict[str, str] = {
    "4-phenyl-1,3-dioxolan-2-one":                  "O=C(OC(c1ccccc1)C1CO1)O",
    "4-methyl-1,3-dioxolan-2-one":                  "O=C(OC(C)C1CO1)O",
    "4-(chloromethyl)-1,3-dioxolan-2-one":          "O=C(OC(CCl)C1CO1)O",
    "cyclohexene carbonate":                        "O=C1OCC2CCCCC2O1",
    "4-(isopropoxymethyl)-1,3-dioxolan-2-one":      "O=C(OC(COCC(C)C)C1CO1)O",
}

CATALYST_SMILES: Dict[str, str] = {
    # Tetrabutylammonium / phosphonium salts
    "tetrabutylammonium bromide":                   "CCCC[N+](CCCC)(CCCC)CCCC.[Br-]",
    "tetrabutylammonium iodide":                    "CCCC[N+](CCCC)(CCCC)CCCC.[I-]",
    "tetrabutylammonium chloride":                  "CCCC[N+](CCCC)(CCCC)CCCC.[Cl-]",
    "tetrabutylammonium fluoride":                  "CCCC[N+](CCCC)(CCCC)CCCC.[F-]",
    "tetrabutylammonium acetate":                   "CCCC[N+](CCCC)(CCCC)CCCC.CC(=O)[O-]",
    "tetrabutylammonium hydroxide":                 "CCCC[N+](CCCC)(CCCC)CCCC.[OH-]",
    "tetrabutylphosphonium bromide":                "CCCC[P+](CCCC)(CCCC)CCCC.[Br-]",
    "tetraphenylphosphonium bromide":               "c1ccc[P+](c2ccccc2)(c2ccccc2)c2ccccc2.[Br-]",
    "methyltriphenylphosphonium bromide":           "C[P+](c1ccccc1)(c1ccccc1)c1ccccc1.[Br-]",
    "ethyltriphenylphosphonium bromide":            "CC[P+](c1ccccc1)(c1ccccc1)c1ccccc1.[Br-]",
    "(2-carboxyethyl)triphenylphosphonium bromide": "O=C(O)CCP+(c1ccccc1)(c1ccccc1)c1ccccc1.[Br-]",
    "(triphenylphosphoranylidene)acetaldehyde":     "O=CC=P+(c1ccccc1)(c1ccccc1)c1ccccc1",

    # Imidazolium / pyridinium / pyrrolidinium ILs
    "1-butyl-3-methylimidazolium chloride":         "CCCCn1cc[n+](C)c1.[Cl-]",
    "1-butyl-3-methylimidazolium bromide":          "CCCCn1cc[n+](C)c1.[Br-]",
    "1-butyl-3-methylimidazolium tetrafluoroborate": "CCCCn1cc[n+](C)c1.[BF4-]",
    "1-ethyl-3-methylimidazolium chloride":         "CCn1cc[n+](C)c1.[Cl-]",
    "1-ethyl-3-methylimidazolium bromide":          "CCn1cc[n+](C)c1.[Br-]",
    "1-hexyl-3-methylimidazolium chloride":         "CCCCCCn1cc[n+](C)c1.[Cl-]",
    "1-hexyl-3-methylimidazolium bromide":          "CCCCCCn1cc[n+](C)c1.[Br-]",
    "1-octyl-3-methylimidazolium chloride":         "CCCCCCCCn1cc[n+](C)c1.[Cl-]",
    "1-benzyl-3-methylimidazolium chloride":        "c1ccc(Cn2cc[n+](C)c2)cc1.[Cl-]",
    "1-(2-hydroxyethyl)-3-methylimidazolium chloride": "OCCn1cc[n+](C)c1.[Cl-]",
    "1-(2-hydroxyethyl)-3-methylimidazolium tetrafluoroborate": "OCCn1cc[n+](C)c1.[BF4-]",
    "1-butyl-2,3-dimethylimidazolium chloride":     "CCCCn1c(C)c[n+](C)c1.[Cl-]",
    "1-butyl-2,3-dimethylimidazolium bromide":      "CCCCn1c(C)c[n+](C)c1.[Br-]",
    "choline chloride":                             "C[N+](C)(C)CCO.[Cl-]",
    "triethylamine":                                "CCN(CC)CC",
    "triethylenediamine (DABCO)":                   "C1CN2CCN1CC2",
    "pyridine":                                     "c1ccncc1",
    "pyrazole":                                     "c1cnn[nH]1",
    "imidazole":                                    "c1c[nH]cn1",
    "benzimidazole":                                "c1ccc2[nH]cnc2c1",
    "1,8-diazabicyclo[5.4.0]undec-7-ene (DBU)":     "N1=C2N(CCCCC2)CCCC1",
    "1,5,7-triazabicyclo[4.4.0]dec-5-ene (TBD)":    "N1=C2N(CCCC2)NCC1",
    "7-methyl-1,5,7-triazabicyclo[4.4.0]dec-5-ene (MTBD)": "CN1CCN2C(=N)CCCC12",

    # Metal halide salts
    "zinc dibromide":      "[Br-].[Zn+2].[Br-]",
    "zinc dichloride":     "[Cl-].[Zn+2].[Cl-]",
    "zinc diiodide":       "[I-].[Zn+2].[I-]",
    "zinc acetate":        "CC(=O)[O-].CC(=O)[O-].[Zn+2]",
    "zinc oxide":          "[O-2].[Zn+2]",
    "magnesium dichloride": "[Cl-].[Mg+2].[Cl-]",
    "magnesium oxide":      "[O-2].[Mg+2]",
    "calcium iodide":      "[I-].[Ca+2].[I-]",
    "lithium bromide":     "[Li+].[Br-]",
    "sodium bromide":      "[Na+].[Br-]",
    "sodium iodide":       "[Na+].[I-]",
    "potassium iodide":    "[K+].[I-]",
    "potassium hydroxide": "[K+].[OH-]",
    "aluminium chloride":  "[Cl-].[Al+3].[Cl-].[Cl-]",
    "iron(III) chloride":  "[Cl-].[Fe+3].[Cl-].[Cl-]",
    "cobalt(II) chloride": "[Cl-].[Co+2].[Cl-]",
    "nickel(II) chloride": "[Cl-].[Ni+2].[Cl-]",
    "copper(II) chloride": "[Cl-].[Cu+2].[Cl-]",
    "tin(II) chloride":    "[Cl-].[Sn+2].[Cl-]",
    "N-bromosuccinimide":  "O=C1CCC(=O)N1Br",
    "bromine":             "BrBr",
    "ascorbic acid":       "OC[C@H]([C@H](O)[C@H](O)C(=O)O)O",
    "indole":              "c1ccc2[nH]ccc2c1",
    "pyrrole":             "c1cc[nH]c1",
    "triazine/melamine":   "Nc1nc(N)nc(N)n1",
    "triphenylphosphine":  "c1ccc(P(c2ccccc2)c2ccccc2)cc1",

    # Solvents
    "dimethylformamide (DMF)":     "CN(C)C=O",
    "dimethyl sulfoxide (DMSO)":   "CS(C)=O",
    "acetonitrile":                "CC#N",
    "methanol":                    "CO",
    "ethanol":                     "CCO",
    "isopropanol":                 "CC(C)O",
    "acetone":                     "CC(C)=O",
    "dichloromethane (DCM)":       "ClCCl",
    "chloroform":                  "ClC(Cl)Cl",
    "chlorobenzene":               "Clc1ccccc1",
    "toluene":                     "Cc1ccccc1",
    "benzene":                     "c1ccccc1",
    "n-hexane":                    "CCCCCC",
    "cyclohexane":                 "C1CCCCC1",
    "diethyl ether":               "CCOCC",
    "ethyl acetate (EtOAc)":       "CCOC(C)=O",
    "tetrahydrofuran (THF)":       "C1CCOC1",
    "dioxane":                     "C1COCCO1",
    "water":                       "O",
    "ethylene glycol":             "OCCO",
    "tetraethylene glycol":        "OCCOCCOCCOCCO",

    # Other / rare
    "tetramethylguanidine":        "CN(C)C(=N)N(C)C",
    "L-arginine":                  "NC(CCCN=C(N)N)C(=O)O",
    "Salen-Co complex":            "CC(=O)/C(C)=C(\\C)Oc1ccc2ccccc2c1.[Co]",
    "Salen-Co (chiral)":           "CC(=O)/C(C)=C(\\C)Oc1ccc2ccccc2c1.[Co]",
    "Zn(phenanthroline-dione)":    "[Zn+2].O=C1c2ccccc2C(=O)c2ccccc21.c1ccc2nccc2c1",
    "ZnO nanoplates":              "[O-2].[Zn+2]",
    "ZnO":                         "[O-2].[Zn+2]",
    "Li-MgO":                      "[Li+].[O-2].[Mg+2]",
    "TBA2ZnBr4":                   "CCCC[N+](CCCC)(CCCC)CCCC.CCCC[N+](CCCC)(CCCC)CCCC.[Br-].[Zn+2].[Br-].[Br-].[Br-]",
    "Cp-TBA-OH (zwitterion)":      "CCCC[N+](CCCC)(CCCC)CCCC.[OH-]",
}

ALL_SMILES: Dict[str, str] = {}
ALL_SMILES.update(SUBSTRATE_SMILES)
ALL_SMILES.update(PRODUCT_SMILES)
ALL_SMILES.update(CATALYST_SMILES)


def resolve_smiles(name: str) -> Optional[str]:
    """Resolve a chemical name to canonical SMILES via static lookup + RDKit."""
    if not isinstance(name, str):
        return None
    name = name.strip()
    if not name or name.lower() in {"nan", "none", ""}:
        return None

    for k, v in ALL_SMILES.items():
        if k.lower() == name.lower():
            return v

    for k, v in ALL_SMILES.items():
        if k.lower() in name.lower() or name.lower() in k.lower():
            return v

    if RDKIT_AVAILABLE:
        try:
            mol = Chem.MolFromSmiles(name)
            if mol is not None:
                return Chem.MolToSmiles(mol)
        except Exception:
            pass

    return None


def build_rxn_smiles(row: pd.Series) -> Optional[str]:
    """Assemble RXN_SMILES = 'reactants.catalysts.solvents>>>products'."""
    parts_reactants: List[str] = []
    parts_catalysts: List[str] = []
    parts_solvents:  List[str] = []
    parts_products:  List[str] = []

    if isinstance(row.get("reactant_smiles"), str):
        parts_reactants.append(row["reactant_smiles"])
    if isinstance(row.get("product_smiles"), str):
        parts_products.append(row["product_smiles"])
    for i in range(1, 5):
        cs = row.get(f"catalyst_{i}_smiles")
        if isinstance(cs, str):
            parts_catalysts.append(cs)
    for i in range(1, 5):
        ss = row.get(f"solvent_{i}_smiles")
        if isinstance(ss, str):
            parts_solvents.append(ss)

    if not parts_reactants:
        return None
    lhs = ".".join(parts_reactants + parts_catalysts + parts_solvents)
    rhs = ".".join(parts_products) if parts_products else ""
    return f"{lhs}>>{rhs}"


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Add 11 SMILES columns to cleaned.csv, producing co2_smiles.csv "
            "(canonical lookup + RDKit fallback) and RXN_SMILES."
        ),
    )
    p.add_argument("--force", action="store_true",
                   help="Overwrite existing co2_smiles.csv even if present.")
    p.add_argument("--dry-run", action="store_true",
                   help="Compute SMILES but do not write co2_smiles.csv.")
    p.add_argument("--verbose", action="store_true", help="Enable DEBUG logging.")
    return p.parse_args(argv)


def configure_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
        force=True,
    )


def main(argv=None) -> int:
    args = parse_args(argv)
    configure_logging(args.verbose)
    log = logging.getLogger("102_smiles")

    log.info("Reading %s", CLEANED_CSV)
    if not CLEANED_CSV.exists():
        log.error("%s not found. Run 101_clean.py first.", CLEANED_CSV)
        return 1

    df = pd.read_csv(CLEANED_CSV, encoding="utf-8-sig")
    if "extraction_status" in df.columns:
        df = df[df["extraction_status"] == "valid"].copy()
    if "yield (%)" in df.columns:
        df = df.dropna(subset=["yield (%)"])
        df = df[df["yield (%)"] > 0]
    df = df.reset_index(drop=True)
    n_in = len(df)
    log.info("Loaded %d rows x %d cols (after valid+yield>0 filter)", n_in, len(df.columns))

    df["reactant_smiles"] = df["reactant_name"].apply(resolve_smiles)
    df["product_smiles"]  = df["product_name"].apply(resolve_smiles)

    for i in range(1, 5):
        col_name  = f"catalyst_{i}_name"
        col_smiles = f"catalyst_{i}_smiles"
        if col_name in df.columns:
            df[col_smiles] = df[col_name].apply(resolve_smiles)
        else:
            df[col_smiles] = pd.NA

    for i in range(1, 5):
        col_name  = f"solvent_{i}"
        col_smiles = f"solvent_{i}_smiles"
        if col_name in df.columns:
            df[col_smiles] = df[col_name].apply(resolve_smiles)
        else:
            df[col_smiles] = pd.NA

    df["RXN_SMILES"] = df.apply(build_rxn_smiles, axis=1)

    coverage = {
        "reactant_smiles":   df["reactant_smiles"].notna().sum(),
        "product_smiles":    df["product_smiles"].notna().sum(),
        "catalyst_1_smiles": df["catalyst_1_smiles"].notna().sum(),
        "catalyst_2_smiles": df["catalyst_2_smiles"].notna().sum(),
        "RXN_SMILES":        df["RXN_SMILES"].notna().sum(),
    }
    log.info("SMILES coverage:")
    for k, v in coverage.items():
        log.info("  %-18s: %4d / %d", k, v, n_in)

    if args.dry_run:
        log.info("[dry-run] SMILES computed but not written.")
        return 0

    if SMILES_CSV.exists() and not args.force:
        log.warning("%s already exists (use --force to overwrite)", SMILES_CSV)
        return 0

    SMILES_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(SMILES_CSV, index=False, encoding="utf-8-sig")
    log.info("Wrote %s (%d rows x %d cols)", SMILES_CSV, len(df), len(df.columns))

    diag = {
        "n_rows": int(n_in),
        "smiles_coverage": {k: int(v) for k, v in coverage.items()},
        "rdkit_available": RDKIT_AVAILABLE,
        "n_lookup_entries": len(ALL_SMILES),
    }
    try:
        SMILES_BASELINE.parent.mkdir(parents=True, exist_ok=True)
        with open(SMILES_BASELINE, "w", encoding="utf-8") as f:
            json.dump(diag, f, indent=2, ensure_ascii=False)
        log.info("Wrote %s", SMILES_BASELINE)
    except Exception as e:
        log.warning("Skipped smiles_baseline.json: %s", e)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
