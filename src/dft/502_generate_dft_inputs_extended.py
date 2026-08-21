#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
502_generate_dft_inputs_extended.py
====================================
Re-generate ORCA 6.1 input files for the *extended* DFT validation set
(50 molecules: 6 substrates, 4 catalysts, 7 solvents, 5 IL ions, 5 complexes,
15 epoxide variants, 13 extra small molecules).

Reads the canonical manifest at
    dft_validation/extended/dft_validation_manifest.csv
and writes
    dft_validation/extended/inp_files/<name>.inp
    dft_validation/extended/xyz_files/<name>.xyz

The on-disk 50 inp + xyz pairs already exist from a prior 502 run; this
script re-generates them idempotently.  Use --force to overwrite.

Inputs (read-only):
    dft_validation/extended/dft_validation_manifest.csv

Outputs:
    dft_validation/extended/inp_files/<name>.inp  (50 ORCA inputs)
    dft_validation/extended/xyz_files/<name>.xyz  (50 XYZ geometries)
    dft_validation/extended/run_extended.sh       (WSL bash runner)

The generated inp files use the SAME template as 501_generate_dft_inputs.py
(inline XYZ block, B3LYP D3BJ def2-TZVP Opt Freq, %maxcore 4000, nprocs 8)
so that 510_parse_dft_outputs.py, 512_xtb_on_dft_geometry.py and
514_dft_vs_xtb_report.py work uniformly across the full DFT set.

Run:
    python src/dft/502_generate_dft_inputs_extended.py           # idempotent
    python src/dft/502_generate_dft_inputs_extended.py --force   # overwrite

History
-------
2026-08-20 rewrite: the previous 502_*.py was overwritten with the 902
content by accident during repo migration.  This rewrite reconstructs the
ext-input generator from scratch using the on-disk manifest as the single
source of truth, so the script can be re-run anytime without losing the
set of molecules.
"""
from __future__ import annotations

import argparse
import csv
import io
import os
import sys
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Make src/ importable for paths.py
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.dirname(_SCRIPT_DIR)  # src/
_PROJECT_ROOT = os.path.dirname(_SRC_DIR)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
from paths import DFT_VALIDATION  # noqa: E402

# ---------------------------------------------------------------------------
# RDKit import (degrades gracefully if missing)
# ---------------------------------------------------------------------------
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False


# ---------------------------------------------------------------------------
# Paths (all derived from paths.py; no hard-coded absolutes)
# ---------------------------------------------------------------------------
DEFAULT_MANIFEST = os.path.join(DFT_VALIDATION, "extended", "dft_validation_manifest.csv")
DEFAULT_INP_DIR = os.path.join(DFT_VALIDATION, "extended", "inp_files")
DEFAULT_XYZ_DIR = os.path.join(DFT_VALIDATION, "extended", "xyz_files")
DEFAULT_RUN_SH = os.path.join(DFT_VALIDATION, "extended", "run_extended.sh")

ORCA_EXE_WSL = "/home/zzj/orca/orca_6_1_1_linux_x86-64_shared_openmpi418_nodmrg/orca"

# Same keyword template as 501 — produces ORCA outputs that 510_parse_dft
# and 514_dft_vs_xtb_report can consume without modification.
DEFAULT_KEYWORDS = "B3LYP D3BJ def2-TZVP Opt Freq"
DEFAULT_NPROCS = 8
DEFAULT_MAXCORE_MB = 4000


# ---------------------------------------------------------------------------
# Hard-coded geometries (RDKit cannot 3D-embed these reliably)
# ---------------------------------------------------------------------------
_HARDCODED_XYZ: Dict[str, Tuple[int, str, List[str]]] = {
    "ZnBr2": (
        3,
        "ZnBr2  (linear, gas-phase Zn-Br = 2.27 A)",
        ["Br     0.000000     0.000000    -2.270000",
         "Zn     0.000000     0.000000     0.000000",
         "Br     0.000000     0.000000     2.270000"],
    ),
    "il_split_2_TBAI_anion": (
        1, "I-", ["I      0.000000     0.000000     0.000000"],
    ),
    "extra_12_carbonate_anion": (
        3,
        "CO3 2-  (planar, D3h)",
        ["O      0.000000     1.150000     0.000000",
         "C      0.000000     0.000000     0.000000",
         "O      0.996190    -0.575000     0.000000",
         "O     -0.996190    -0.575000     0.000000"],
    ),
}


# ---------------------------------------------------------------------------
# 3-D embedding
# ---------------------------------------------------------------------------
def smiles_to_xyz(smiles: str, name: str) -> Optional[str]:
    """RDKit ETKDG + MMFF94 → XYZ string; returns None on failure."""
    if name in _HARDCODED_XYZ:
        n, comment, rows = _HARDCODED_XYZ[name]
        body = "\n".join(rows) + "\n"
        return f"{n}\n{comment}\n{body}"

    if not RDKIT_AVAILABLE:
        print(f"  [error] RDKit not available; cannot embed {name}", file=sys.stderr)
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"  [warn] RDKit could not parse SMILES for {name}", file=sys.stderr)
            return None
        mol = Chem.AddHs(mol)
        if AllChem.EmbedMolecule(mol, randomSeed=42) != 0:
            print(f"  [warn] 3D embedding failed for {name}", file=sys.stderr)
            return None
        AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
        conf = mol.GetConformer()
        rows = [f"{mol.GetNumAtoms()}", name]
        for atom in mol.GetAtoms():
            pos = conf.GetAtomPosition(atom.GetIdx())
            rows.append(f"{atom.GetSymbol():2s} {pos.x:12.6f} {pos.y:12.6f} {pos.z:12.6f}")
        return "\n".join(rows) + "\n"
    except Exception as e:
        print(f"  [error] {name}: {e}", file=sys.stderr)
        return None


def make_inp_text(keywords: str, charge: int, mult: int, xyz_block: str) -> str:
    # Auto-downgrade monoatomic systems (e.g. il_split_2_TBAI_anion) from
    # Opt+Freq to SP -- a 1-atom Hessian is undefined and ORCA hangs.
    n_atoms = len([l for l in xyz_block.strip().splitlines() if l.strip()])
    if n_atoms <= 1 and ("Opt" in keywords or "Freq" in keywords):
        keywords = keywords.replace("Opt", "").replace("Freq", "").strip()
        keywords = f"{keywords} SP" if keywords else "SP"
        print(f"  [info] monoatomic -> {keywords}")
    return f"""! {keywords}

%maxcore {DEFAULT_MAXCORE_MB}

%pal
    nprocs {DEFAULT_NPROCS}
end

%output
    Print[ P_OrbEn ] 2        # full orbital energies -> HOMO/LUMO
    Print[ P_MOs ] 1          # MO coefficients
    Print[ P_Mayer ] 1        # Mayer bond orders
    Print[ P_Hirshfeld ] 1    # Hirshfeld charges
    Print[ P_homolumogap ] 1  # SCF-iteration HOMO/LUMO gap
end

* xyz {charge} {mult}
{xyz_block}*
"""


def write_file(path: str, content: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def make_xyz_block(xyz_path: str) -> str:
    with open(xyz_path, "r", encoding="utf-8") as f:
        lines = f.read().strip().splitlines()
    body = lines[2:]
    return "\n".join(body) + "\n"


# ---------------------------------------------------------------------------
# Manifest reader
# ---------------------------------------------------------------------------
def read_manifest(manifest_csv: str) -> List[Dict]:
    rows: List[Dict] = []
    # Manifest is sometimes saved with a UTF-8 BOM (notepad.exe, git on
    # Windows).  csv.DictReader would otherwise return "\ufeffidx" as the
    # first column name and every downstream key lookup would KeyError.
    with open(manifest_csv, "r", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            row["idx"] = int(row["idx"])
            row["charge"] = int(row["charge"])
            row["mult"] = int(row["mult"])
            rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Per-molecule generation
# ---------------------------------------------------------------------------
def generate_one(row: Dict, xyz_dir: str, inp_dir: str, keywords: str,
                  force: bool = False) -> Optional[str]:
    name = row["name"]
    xyz_path = os.path.join(xyz_dir, f"{name}.xyz")
    inp_path = os.path.join(inp_dir, f"{name}.inp")

    if not force and os.path.exists(inp_path) and os.path.exists(xyz_path):
        print(f"  [skip] {name} (inp+xyz already exist)")
        return inp_path

    xyz_text = smiles_to_xyz(row["smiles"], name)
    if xyz_text is None:
        print(f"  [fail] {name}: no geometry")
        return None
    write_file(xyz_path, xyz_text)

    xyz_block = make_xyz_block(xyz_path)
    inp_text = make_inp_text(keywords, row["charge"], row["mult"], xyz_block)
    write_file(inp_path, inp_text)
    print(f"  [ok]   {name}  (charge={row['charge']}, mult={row['mult']})")
    return inp_path


# ---------------------------------------------------------------------------
# WSL bash runner (mirrors 502_run_dft_wsl.ps1 but for the extended set)
# ---------------------------------------------------------------------------
def write_run_sh(out_path: str, inp_paths: List[str], orca_wsl: str) -> None:
    body = ["#!/usr/bin/env bash",
            "# Auto-generated by 502_generate_dft_inputs_extended.py",
            "set -u",
            "cd /home/zzj/co2_dft/dft_validation/extended",
            f'export ORCA_BIN="{orca_wsl}"',
            'export OMP_NUM_THREADS=$(nproc)',
            "",
            "mkdir -p logs",
            "",
            "run_one() {",
            "  local inp=\"$1\"",
            "  local name=\"${inp%.inp}\"",
            "  local out=\"${name}.out\"",
            "  if [ -f \"$out\" ] && grep -q 'TERMINATED NORMALLY' \"$out\"; then",
            '    echo "[SKIP] $name already converged"',
            "    return 0",
            "  fi",
            '  echo "[$(date +%H:%M:%S)] Running $name ..."',
            '  if "$ORCA_BIN" "$inp" > "$out" 2>&1 && grep -q "TERMINATED NORMALLY" "$out"; then',
            '    echo "  [OK]   $name"',
            "    return 0",
            "  fi",
            '  echo "  [FAIL] $name"',
            "  return 1",
            "}",
            "",
            "ok=0; fail=0",
            "for inp in *.inp; do",
            "  if run_one \"$inp\"; then ok=$((ok+1)); else fail=$((fail+1)); fi",
            "done",
            "",
            'echo ""',
            "total=$(ls -1 *.out 2>/dev/null | wc -l)",
            "conv=$(grep -l 'TERMINATED NORMALLY' *.out 2>/dev/null | wc -l)",
            'echo "=========================================="',
            'echo "  Converged: $conv / $total"',
            'echo "  OK  this run: $ok"',
            'echo "  FAIL this run: $fail"',
            'echo "=========================================="',
            ""]
    write_file(out_path, "\n".join(body))
    os.chmod(out_path, 0o755)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="502 extended DFT input generator.")
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST,
                        help="Path to dft_validation_manifest.csv")
    parser.add_argument("--inp-dir", default=DEFAULT_INP_DIR,
                        help="Output directory for *.inp")
    parser.add_argument("--xyz-dir", default=DEFAULT_XYZ_DIR,
                        help="Output directory for *.xyz")
    parser.add_argument("--run-sh", default=DEFAULT_RUN_SH,
                        help="Output path for the WSL bash runner")
    parser.add_argument("--keywords", default=DEFAULT_KEYWORDS,
                        help="ORCA !-line keywords (default: B3LYP D3BJ def2-TZVP Opt Freq)")
    parser.add_argument("--nproc", type=int, default=DEFAULT_NPROCS)
    parser.add_argument("--orca-wsl", default=ORCA_EXE_WSL,
                        help="Path to ORCA inside WSL")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing inp + xyz files")
    parser.add_argument("--skip-run-sh", action="store_true",
                        help="Don't regenerate the WSL runner")
    args = parser.parse_args()

    if not RDKIT_AVAILABLE:
        print("RDKit is required (pip install rdkit). Aborting.", file=sys.stderr)
        sys.exit(1)

    if not os.path.exists(args.manifest):
        print(f"Manifest not found: {args.manifest}", file=sys.stderr)
        sys.exit(2)

    os.makedirs(args.inp_dir, exist_ok=True)
    os.makedirs(args.xyz_dir, exist_ok=True)

    rows = read_manifest(args.manifest)
    print("=" * 60)
    print(f"502 extended DFT input generator")
    print(f"  manifest: {args.manifest}")
    print(f"  inp_dir : {args.inp_dir}")
    print(f"  xyz_dir : {args.xyz_dir}")
    print(f"  keywords: {args.keywords}")
    print(f"  total molecules: {len(rows)}")
    print("=" * 60)

    inp_paths: List[str] = []
    for row in rows:
        path = generate_one(row, args.xyz_dir, args.inp_dir, args.keywords, args.force)
        if path is not None:
            inp_paths.append(path)

    print(f"\nGenerated {len(inp_paths)}/{len(rows)} inputs.")

    if not args.skip_run_sh and inp_paths:
        write_run_sh(args.run_sh, inp_paths, args.orca_wsl)
        print(f"Wrote WSL runner: {args.run_sh}")

    # Family breakdown for log parity with old genlog.txt
    fam_counts: Dict[str, int] = {}
    for row in rows:
        fam_counts[row["family"]] = fam_counts.get(row["family"], 0) + 1
    print("\nFamily counts:")
    for fam, n in sorted(fam_counts.items()):
        print(f"  {fam:12s}: {n}")


if __name__ == "__main__":
    main()