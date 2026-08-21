#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
510_parse_dft_outputs.py
========================
Parse ORCA 6.1 output files (.out) from the DFT validation set and write:
  1. <name>.xyz.out               — final optimized geometry (XYZ frame)
  2. dft_results_summary.csv      — one row per molecule with HOMO/LUMO/gap/dipole

Used by:
  - 512_xtb_on_dft_geometry.py  (reads <name>.xyz.out for apples-to-apples xTB)
  - 514_dft_vs_xtb_report.py     (reads dft_results_summary.csv)

Default work directory: dft_validation/  (the dir 501_generate_dft_inputs.py
populates with <name>.inp / <name>.out / <name>.xyz / <name>.xyz.out).

Run:
    python 510_parse_dft_outputs.py                       # default work-dir
    python 510_parse_dft_outputs.py --work-dir <dir>     # explicit work-dir
    python 510_parse_dft_outputs.py --out   <csv_path>   # explicit output CSV

Reference: ORCA 6.1 Manual §7.3.4 (orbital energies), §9.4 (geometry
optimization), §10.2 (dipole moments). All regex patterns below are
version-tolerant for ORCA 5.x and 6.x.

The script is idempotent: re-running it overwrites <name>.xyz.out and
dft_results_summary.csv. Use --force from run_pipeline_v2.ps1 to bypass
any caching.
"""
import argparse
import csv
import os
import re
import sys
from typing import Dict, List, Optional, Tuple, Tuple


# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------
# 1 Hartree = 27.211386245988 eV (CODATA 2018, used by ORCA's own Eh->eV printout)
HARTREE_TO_EV = 27.211386245988

# Mirror CANDIDATES from 501_generate_dft_inputs.py so we can back-fill
# SMILES / role / category into the summary CSV.  TBAI is intentionally
# absent -- 501 splits it into TBAI_cation / TBAI_anion at write time.
NAME_METADATA: Dict[str, Dict[str, str]] = {
    # substrates
    "styrene_oxide":               {"smiles": "C1OC1c1ccccc1",     "role": "substrate",            "category": "Styrene-oxide family"},
    "epoxybutane":                 {"smiles": "CCC1CO1",            "role": "substrate",            "category": "Aliphatic epoxide"},
    "epichlorohydrin":             {"smiles": "ClCC1CO1",           "role": "substrate",            "category": "Epichlorohydrin"},
    "propylene_oxide":             {"smiles": "CC1CO1",             "role": "substrate",            "category": "Propylene-oxide"},
    "isopropyl_glycidyl_ether":    {"smiles": "CC(C)OCC1CO1",       "role": "substrate",            "category": "Glycidyl ether"},
    # catalysts
    "TBAI":                        {"smiles": "CCCC[N+](CCCC)(CCCC)CCCC.[I-]", "role": "catalyst_ionic_liquid", "category": "Ionic liquid"},
    "ZnBr2":                       {"smiles": "[Br-].[Zn+2].[Br-]", "role": "catalyst_metal_halide", "category": "Metal halide"},
    "DBU":                         {"smiles": "N1=C2N(CCCC2)CCCC1", "role": "catalyst_organic_base", "category": "Organic base"},
    # solvents
    "DMSO":                        {"smiles": "CS(C)=O",            "role": "solvent",              "category": "Polar aprotic"},
    "DMF":                         {"smiles": "CN(C)C=O",           "role": "solvent",              "category": "Polar aprotic"},
    # reactants / products
    "CO2":                         {"smiles": "O=C=O",              "role": "reactant",             "category": "Gas"},
    "cyclic_carbonate_product":    {"smiles": "O=C1OCCO1",           "role": "product",              "category": "Product"},
    # Split-salt additions that 501 generates separately.  SMILES for the
    # cation is the standard tetraalkylammonium skeleton.
    "TBAI_cation":                 {"smiles": "CCCC[N+](CCCC)(CCCC)CCCC", "role": "catalyst_ionic_liquid", "category": "Ionic liquid"},
    "TBAI_anion":                  {"smiles": "[I-]",               "role": "catalyst_ionic_liquid", "category": "Ionic liquid"},
}

# CSV columns emitted to dft_results_summary.csv.  The required keys are
# `name`, `homo_eV`, `lumo_eV`, `gap_eV`, `dipole_debye` (consumed by 514).
# Everything else is informational and helps cross-referencing.
SUMMARY_HEADER = [
    "name", "file", "smiles", "role", "category", "level",
    "converged", "final_e_Eh",
    "homo_eV", "lumo_eV", "gap_eV", "dipole_debye",
]


# ----------------------------------------------------------------------------
# Regex patterns (ORCA 5.x / 6.x tolerant)
# ----------------------------------------------------------------------------
# ORCA orbital-energy block ends with literal lines like:
#   HOMO:      -0.2345 Eh    ...
#   LUMO:      -0.0123 Eh    ...
# The block also prints "HOMO-LUMO GAP:    0.2222 Eh" but we compute it
# ourselves for safety (HOMO-LUMO in the source IS the gap, but we want
# a consistent unit conversion).
RE_HOMO = re.compile(
    r"^\s*HOMO\s*:\s*(-?\d+\.\d+)\s*Eh", re.IGNORECASE | re.MULTILINE
)
RE_LUMO = re.compile(
    r"^\s*LUMO\s*:\s*(-?\d+\.\d+)\s*Eh", re.IGNORECASE | re.MULTILINE
)
RE_DIPOLE = re.compile(
    r"Total Dipole Moment\s*:\s*(-?\d+\.\d+)", re.IGNORECASE
)
RE_FINAL_E = re.compile(
    r"FINAL SINGLE POINT ENERGY\s*[:=]?\s*(-?\d+\.\d+)", re.IGNORECASE
)
# ORCA 6.1 prints "*** OPTIMIZATION RUN DONE ***" on success
# and                                          "*** OPTIMIZATION FAILED ***" on failure.
RE_CONVERGED = re.compile(
    r"\*\*\*\s*OPTIMIZATION\s+(RUN\s+DONE|FAILED)\s*\*\*\*", re.IGNORECASE
)
# ORCA 5.0/4.x used "THE OPTIMIZATION HAS CONVERGED" instead.
RE_CONVERGED_OLD = re.compile(
    r"THE\s+OPTIMIZATION\s+HAS\s+CONVERGED", re.IGNORECASE
)
# Keyword line echoed in the output header: "! B3LYP D3BJ def2-TZVP Opt Freq"
RE_KEYWORDS = re.compile(
    r"^!\s*([A-Za-z0-9_\-]+(?:\s+[A-Za-z0-9_\-]+)*?)\s*$", re.MULTILINE
)
# Cartesian coordinates block (final frame).  ORCA 6.1 prints:
#   ==============================================================
#   CARTESIAN COORDINATES (ANGSTROEM)
#   ==============================================================
#     N   x   y   z   ...
#   <blank>
#   ==============================================================
# ORCA 5/4.x and some blocks within ORCA 6.x use plain "-" runs instead
# of "=" runs.  We accept both.
RE_CART_BLOCK = re.compile(
    r"CARTESIAN COORDINATES\s*\((ANGSTROEM|A\.U\.)\)\s*\n"
    r"[=\-]+\s*\n"
    r"(.*?)"
    r"\n\s*[=\-]+",
    re.DOTALL | re.IGNORECASE,
)
# One XYZ frame is "<count>\n<comment>\n<rows>\n"; we use the trailing
# of the LAST Cartesian block as the final geometry.
RE_XYZ_LINE = re.compile(r"^\s*([A-Z][a-z]?)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s*$")


# ----------------------------------------------------------------------------
# Parsing helpers
# ----------------------------------------------------------------------------
# ORCA 6.x orbital energy table format:
#   NO   OCC          E(Eh)            E(eV)
#    0   2.0000     -19.252111      -523.8766
#    1   2.0000     -10.442785      -284.1626
#   ...
#   25   0.0000       0.033295         0.9060    <- first unoccupied
# HOMO = highest-E occupied orbital (last row with OCC > 0)
# LUMO = lowest-E unoccupied orbital (first row with OCC == 0)
_RE_ORBITAL_TABLE = re.compile(
    r"ORBITAL ENERGIES\s*\n\s*-+\s*\n"
    r"\s*NO\s+OCC\s+E\(Eh\)\s+E\(eV\)\s*\n"
    r"(.*?)"
    r"\n\s*(?:[*-]|$)",
    re.DOTALL | re.IGNORECASE,
)


def _parse_homo_lumo_table(text: str) -> Tuple[Optional[float], Optional[float]]:
    """Parse ORCA 6.x orbital-energy table to extract HOMO and LUMO in Eh.

    Returns (homo_eh, lumo_eh).  Values are None if parsing fails.
    """
    block_match = _RE_ORBITAL_TABLE.search(text)
    if not block_match:
        return None, None

    homo_eh = None
    lumo_eh = None

    for line in block_match.group(1).splitlines():
        line = line.strip()
        if not line:
            continue
        # Each row: "NO  OCC  E(Eh)  E(eV)" — whitespace-separated
        # We expect at least 3 columns: NO, OCC, E(Eh)
        parts = line.split()
        if len(parts) < 3:
            continue
        try:
            occ = float(parts[1])
            e_eh = float(parts[2])
        except (ValueError, IndexError):
            continue

        if occ > 0 and (homo_eh is None or e_eh > homo_eh):
            homo_eh = e_eh
        if occ == 0 and lumo_eh is None:
            lumo_eh = e_eh

    return homo_eh, lumo_eh


def _safe_float(s: Optional[str]) -> Optional[float]:
    if s is None:
        return None
    try:
        v = float(s)
    except (ValueError, TypeError):
        return None
    return v


def parse_orca_out(path: str) -> Dict[str, Optional[object]]:
    """Return a dict of extracted quantities from a single ORCA .out file.

    All numeric fields may be None on failure -- the caller decides whether
    to skip the row or write a partial entry.
    """
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
    except OSError as e:
        return {
            "homo_eV": None, "lumo_eV": None, "gap_eV": None,
            "dipole_debye": None, "final_e_Eh": None,
            "converged": False, "keywords": None,
            "geometry_xyz": None, "error": f"read-fail: {e}",
        }

    # HOMO / LUMO  (always in Eh in ORCA output -> convert to eV)
    # Try two formats:
    #   ORCA 5.x  : "HOMO:      -0.2345 Eh    ..."  (one-shot regex)
    #   ORCA 6.x  : table "NO  OCC  E(Eh)  E(eV)"   (find highest OCC then lowest unocc)
    homo_eh, lumo_eh = _parse_homo_lumo_table(text)
    if homo_eh is None:
        homo_eh = _safe_float((RE_HOMO.search(text) or [None, None])[1])
    if lumo_eh is None:
        lumo_eh = _safe_float((RE_LUMO.search(text) or [None, None])[1])
    homo_eV = homo_eh * HARTREE_TO_EV if homo_eh is not None else None
    lumo_eV = lumo_eh * HARTREE_TO_EV if lumo_eh is not None else None
    gap_eV = (lumo_eV - homo_eV) if (homo_eV is not None and lumo_eV is not None) else None

    # Dipole moment  (already in Debye in ORCA output)
    dipole_d = _safe_float((RE_DIPOLE.search(text) or [None, None])[1])

    # Final electronic energy (single-point, Eh)
    final_e = _safe_float((RE_FINAL_E.search(text) or [None, None])[1])

    # Convergence
    converged = bool(RE_CONVERGED.search(text) or RE_CONVERGED_OLD.search(text))

    # Keywords line  (first matching line in the header)
    kw_match = RE_KEYWORDS.search(text)
    keywords = kw_match.group(1).strip() if kw_match else None

    # Final geometry -- last CARTESIAN COORDINATES (ANGSTROEM) block
    ang_geom = None
    cart_blocks = RE_CART_BLOCK.findall(text)
    if cart_blocks:
        unit, body = cart_blocks[-1]
        if unit.upper().replace(".", "") == "A" or unit.upper() == "ANGSTROEM":
            rows = []
            for line in body.splitlines():
                m = RE_XYZ_LINE.match(line)
                if m:
                    rows.append(f"{m.group(1):2s} {float(m.group(2)):14.6f} "
                                f"{float(m.group(3)):14.6f} {float(m.group(4)):14.6f}")
            if rows:
                ang_geom = f"{len(rows)}\n"
                ang_geom += f"extracted by 510_parse_dft_outputs.py\n"
                ang_geom += "\n".join(rows) + "\n"

    return {
        "homo_eV": homo_eV,
        "lumo_eV": lumo_eV,
        "gap_eV":  gap_eV,
        "dipole_debye": dipole_d,
        "final_e_Eh": final_e,
        "converged": converged,
        "keywords": keywords,
        "geometry_xyz": ang_geom,
        "error": None,
    }


def write_xyz_out(out_dir: str, basename: str, xyz_text: str) -> str:
    """Write <basename>.xyz.out and return its path."""
    path = os.path.join(out_dir, f"{basename}.xyz.out")
    with open(path, "w", encoding="utf-8") as f:
        f.write(xyz_text)
    return path


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------
def discover_out_files(work_dir: str) -> List[str]:
    """Return sorted basenames of .out files in work_dir (skip _b3lyp variants)."""
    if not os.path.isdir(work_dir):
        return []
    return sorted(
        fn for fn in os.listdir(work_dir)
        if fn.endswith(".out")
        and not fn.endswith(".xyz.out")
        and "_b3lyp" not in fn.lower()
        and os.path.isfile(os.path.join(work_dir, fn))
    )


def process_one(work_dir: str, basename: str) -> Dict[str, object]:
    """Parse one .out file and write its .xyz.out.  Return the summary row."""
    out_path = os.path.join(work_dir, basename)
    parsed = parse_orca_out(out_path)

    name = basename[:-len(".out")]  # strip ".out"
    meta = NAME_METADATA.get(name, {})

    # Write geometry sidecar for 512
    if parsed["geometry_xyz"]:
        try:
            write_xyz_out(work_dir, name, parsed["geometry_xyz"])
        except OSError as e:
            print(f"  [warn] failed to write {name}.xyz.out: {e}", file=sys.stderr)

    row = {
        "name":          name,
        "file":          basename,
        "smiles":        meta.get("smiles", ""),
        "role":          meta.get("role", ""),
        "category":      meta.get("category", ""),
        "level":         parsed["keywords"] or "",
        "converged":     "yes" if parsed["converged"] else "no",
        "final_e_Eh":    parsed["final_e_Eh"] if parsed["final_e_Eh"] is not None else "",
        "homo_eV":       parsed["homo_eV"]    if parsed["homo_eV"]    is not None else "",
        "lumo_eV":       parsed["lumo_eV"]    if parsed["lumo_eV"]    is not None else "",
        "gap_eV":        parsed["gap_eV"]     if parsed["gap_eV"]     is not None else "",
        "dipole_debye":  parsed["dipole_debye"] if parsed["dipole_debye"] is not None else "",
    }
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true",
                        help="Accept the global --force flag from run_pipeline_v2.ps1 "
                             "(this script always overwrites its outputs, so the "
                             "flag is a no-op here).")
    parser.add_argument("--work-dir", default=r"dft_validation",
                        help="Directory containing the ORCA <name>.out files.")
    parser.add_argument("--out", default=None,
                        help="Output CSV path.  Defaults to "
                             "<work-dir>/dft_results_summary.csv.")
    args = parser.parse_args()

    work_dir = os.path.abspath(args.work_dir)
    out_csv = args.out or os.path.join(work_dir, "dft_results_summary.csv")

    if not os.path.isdir(work_dir):
        print(f"[error] work-dir not found: {work_dir}", file=sys.stderr)
        sys.exit(1)

    out_files = discover_out_files(work_dir)
    if not out_files:
        print(f"[error] no .out files in {work_dir}", file=sys.stderr)
        sys.exit(2)

    print("=" * 60)
    print(f"510_parse_dft_outputs | work_dir={work_dir}")
    print(f"Found {len(out_files)} ORCA output files.")
    print("=" * 60)

    rows: List[Dict[str, object]] = []
    n_geo_written = 0
    n_with_homo_lumo = 0
    for basename in out_files:
        name = basename[:-len(".out")]
        print(f"\n[{name}]")
        try:
            row = process_one(work_dir, basename)
        except Exception as e:
            print(f"  [error] {e}", file=sys.stderr)
            row = {
                "name": name, "file": basename, "smiles": "", "role": "",
                "category": "", "level": "", "converged": "no",
                "final_e_Eh": "", "homo_eV": "", "lumo_eV": "",
                "gap_eV": "", "dipole_debye": "",
            }
        rows.append(row)
        # Human-readable summary
        if row["homo_eV"] != "" and row["lumo_eV"] != "":
            n_with_homo_lumo += 1
            print(f"  HOMO={float(row['homo_eV']):+.4f} eV  "
                  f"LUMO={float(row['lumo_eV']):+.4f} eV  "
                  f"gap={float(row['gap_eV']):+.4f} eV  "
                  f"|mu|={float(row['dipole_debye']) if row['dipole_debye'] != '' else 0.0:.3f} D")
        else:
            print(f"  HOMO/LUMO not found (job may not have converged)")
        if row["level"]:
            print(f"  level   : {row['level']}")
        if row["converged"] == "yes":
            print(f"  status  : converged")
        else:
            print(f"  status  : not converged (or no Opt keyword)")
        xyz_out = os.path.join(work_dir, f"{name}.xyz.out")
        if os.path.isfile(xyz_out):
            n_geo_written += 1
            print(f"  geometry: {xyz_out}")

    # Write the summary CSV
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    with open(out_csv, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_HEADER)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in SUMMARY_HEADER})

    print("\n" + "=" * 60)
    print(f"Wrote {len(rows)} rows to {out_csv}")
    print(f"Geometries written : {n_geo_written}/{len(rows)}")
    print(f"With HOMO/LUMO     : {n_with_homo_lumo}/{len(rows)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
