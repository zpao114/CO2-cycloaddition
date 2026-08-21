#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
512_xtb_on_dft_geometry.py
==========================
Single-point GFN2-xTB on the **DFT-optimized** geometries.

Why this exists
---------------
The original xTB workflow (104_run_xtb.py) uses 3D coordinates produced by
RDKit + MMFF. The DFT workflow (501_generate_dft_inputs.py -> ORCA 6.1) opts
to a B3LYP D3BJ def2-TZVP minimum and 510_parse_dft_outputs.py writes
the final geometry to ``<name>.xyz.out``.

For an apples-to-apples xTB vs DFT validation we need xTB to be evaluated
on the **same geometry** as DFT. This script:

  1. Reads the 13 ``<name>.xyz.out`` files written by 715.
  2. Runs ``xtb input.xyz --sp --chrg N --alpb <solvent> --gfn 2`` on each,
     where the geometry is the DFT-optimized one.
  3. Writes ``xtb_on_dft_geometry.csv`` with the same schema as
     ``xtb_results_summary.csv`` so 716 can swap it in cleanly.

Inputs
------
  dft_validation/<name>.xyz.out          produced by 715

Outputs
-------
  dft_validation/xtb_on_dft_geometry.csv
  dft_validation/<name>.xtb.stdout       raw xTB log per molecule

Reference
---------
  xtb docs: https://xtb-docs.readthedocs.io/
  ORCA 6.1 Manual §7.3.4 (orbital energies), §3.5.2 (xTB flag mirror).
"""
import os
import argparse
import csv
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

# Reuse the parser and helpers from the existing 104 workflow so we keep
# one definition of xTB output parsing.
PROJECT_ROOT = os.environ.get("CO2_PROJECT_ROOT", r"D:\machine-learning\CO2-cycloaddition")
sys.path.insert(0, str(Path(PROJECT_ROOT) / 'src'))
from importlib import import_module
_mod_104 = import_module("data.104_run_xtb")

parse_xtb_out = _mod_104.parse_xtb_out
infer_charge  = _mod_104.infer_charge
locate_xtb    = _mod_104.locate_xtb


WORK_DIR_DEFAULT = r"D:\machine-learning\CO2-cycloaddition\dft_validation"
OUTPUT_CSV       = "results/xtb_on_dft_geometry.csv"


# ---------------------------------------------------------------------------
# 1. Pull SMILES / role from the original xTB summary.
#    715 only writes geometry, not SMILES, so we have to look elsewhere.
# ---------------------------------------------------------------------------
def load_prev_summary(path: str) -> Dict[str, Dict]:
    """Return name -> {smiles, role, charge, solvent}, keyed by basename."""
    out: Dict[str, Dict] = {}
    if not os.path.isfile(path):
        return out
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            out[row["name"]] = {
                "smiles":  row.get("smiles", ""),
                "role":    row.get("role", ""),
                "charge":  int(row.get("charge", "0") or 0),
                "solvent": row.get("solvent", ""),
            }
    return out


# ---------------------------------------------------------------------------
# 2. xyz.out -> clean xyz (drop the first 2 header lines).
# ---------------------------------------------------------------------------
def dft_xyzout_to_xyz(xyzout_path: str, xyz_path: str) -> bool:
    """510 writes a standard XYZ file whose ONLY marker is the literal
    comment line ``extracted by 510_parse_dft_outputs.py``. xTB (and most
    other tools) are happy with that, but we still rewrite the file to a
    ``.xyz`` extension so the file naming makes sense downstream.

    The file content is a single XYZ frame, so this is a trivial copy.
    """
    with open(xyzout_path, "r", encoding="utf-8", errors="ignore") as f:
        txt = f.read()
    if not txt.strip():
        return False
    # Sanity check: first non-empty line must be the atom count.
    first = next((ln for ln in txt.splitlines() if ln.strip()), "")
    if not first.strip().isdigit():
        return False
    with open(xyz_path, "w", encoding="utf-8") as f:
        f.write(txt)
    return True


# ---------------------------------------------------------------------------
# 3. Run xTB on one xyz file.
# ---------------------------------------------------------------------------
def run_xtb_one(xyz_path: str, charge: int, mult: int,
                gfn: int, solvent: Optional[str], acc: float,
                timeout: int, raw_log: str) -> Optional[str]:
    """Returns the path to the captured xtb.out log; None on hard failure."""
    xtb_exe = locate_xtb()
    if xtb_exe is None:
        print("[error] xtb executable not found on PATH.", file=sys.stderr)
        return None

    workdir = tempfile.mkdtemp(prefix="xtb_on_dft_")
    try:
        cwd_before = os.getcwd()
        os.chdir(workdir)
        try:
            # xTB needs the file in the CWD it runs from; use absolute path
            # to the file we already placed in workdir.
            cmd = [
                xtb_exe, xyz_path,
                "--sp",
                "--chrg", str(charge),
                "--uhf",  str(max(0, mult - 1)),
                "--gfn",  str(gfn),
                "--acc",  str(acc),
            ]
            if solvent:
                cmd += ["--alpb", solvent]
            print(f"    cmd: {' '.join(cmd)}")
            result = subprocess.run(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                timeout=timeout, encoding="utf-8", errors="ignore",
            )
            log_path = os.path.join(workdir, "xtb.out")
            with open(log_path, "w", encoding="utf-8", errors="ignore") as _f:
                _f.write(result.stdout or "")
            with open(raw_log, "w", encoding="utf-8", errors="ignore") as f:
                f.write(result.stdout or "")
            if result.returncode != 0:
                print(f"[warn] xtb returned {result.returncode}; see {raw_log}",
                      file=sys.stderr)
            return log_path
        finally:
            os.chdir(cwd_before)
    except subprocess.TimeoutExpired:
        print(f"[warn] xtb timed out after {timeout}s on {xyz_path}",
              file=sys.stderr)
        return None


# ---------------------------------------------------------------------------
# 4. Driver
# ---------------------------------------------------------------------------
SUMMARY_HEADER = [
    "smiles", "name", "role", "charge", "solvent", "gfn",
    "homo_eV", "lumo_eV", "gap_eV", "total_e_Eh", "dipole_D",
    "mulliken_q_C", "mulliken_q_O", "mulliken_q_N", "mulliken_q_Br",
    "mulliken_q_I", "mulliken_q_Zn", "xtb_ok",
]


def summary_row(name: str, smiles: str, role: str, charge: int,
                solvent: str, gfn: int, res: Dict) -> Dict:
    mq = res["mulliken_q"]
    return {
        "smiles":      smiles,
        "name":        name,
        "role":        role,
        "charge":      charge,
        "solvent":     solvent,
        "gfn":         gfn,
        "homo_eV":     res["homo_eV"],
        "lumo_eV":     res["lumo_eV"],
        "gap_eV":      res["gap_eV"],
        "total_e_Eh":  res["total_e_Eh"],
        "dipole_D":    res["dipole_D"],
        "mulliken_q_C":  mq.get("C"),
        "mulliken_q_O":  mq.get("O"),
        "mulliken_q_N":  mq.get("N"),
        "mulliken_q_Br": mq.get("Br"),
        "mulliken_q_I":  mq.get("I"),
        "mulliken_q_Zn": mq.get("Zn"),
        "xtb_ok":      res["ok"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true",
                        help="Overwrite output even if it exists (added so "
                             "run_pipeline.ps1's global --force flag is "
                             "accepted).")
    parser.add_argument("--dft-dir",   default=WORK_DIR_DEFAULT,
                        help="Directory containing 715's <name>.xyz.out")
    parser.add_argument("--prev-summary", default='dft_validation/results/xtb_results_summary.csv',
                        help="Original xtb_results_summary.csv (for SMILES)")
    parser.add_argument("--solvent", default="dmso",
                        help="Implicit solvent for ALPB; '' to disable. "
                             "Default: dmso (matches 104).")
    parser.add_argument("--gfn", type=int, choices=[0, 1, 2], default=2)
    parser.add_argument("--acc", type=float, default=1.0)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--output", default=OUTPUT_CSV)
    args = parser.parse_args()

    if locate_xtb() is None:
        print("[error] xtb executable not found on PATH.", file=sys.stderr)
        sys.exit(1)

    solvent = args.solvent.strip() or None

    prev = load_prev_summary(args.prev_summary)
    xyzouts = sorted(
        f for f in os.listdir(args.dft_dir)
        if f.endswith(".xyz.out")
        and os.path.isfile(os.path.join(args.dft_dir, f))
        # mirror 715's filter: skip duplicates/method variants
        and "_b3lyp" not in f.lower()
    )
    if not xyzouts:
        print(f"[error] no .xyz.out files in {args.dft_dir}", file=sys.stderr)
        sys.exit(2)

    print("=" * 60)
    print(f"512_xtb_on_dft_geometry | GFN{args.gfn}-xTB | solvent={solvent} "
          f"| acc={args.acc}")
    print(f"Found {len(xyzouts)} DFT geometries.")
    print("=" * 60)

    rows: List[Dict] = []
    for xyzout in xyzouts:
        name = xyzout[:-len(".xyz.out")]
        meta = prev.get(name, {})
        smiles = meta.get("smiles", "")
        role   = meta.get("role", "")
        charge = meta.get("charge")
        if charge is None:
            charge = infer_charge(smiles, role) if smiles else 0

        print(f"\n[{role or '?':<10s}] {name}  (charge={charge})")

        # Place the cleaned XYZ in a temp dir under a non-dot-prefixed name
        # so xTB 6.7.1 will see it (dotfiles were silently skipped on Windows).
        workdir = tempfile.mkdtemp(prefix="xtb_on_dft_")
        safe_stem = re.sub(r"[^A-Za-z0-9_]", "_", name)
        tmp_xyz = os.path.join(workdir, f"input_{safe_stem}.xyz")
        raw_log = os.path.join(args.dft_dir, f"{name}.xtb.stdout")
        try:
            if not dft_xyzout_to_xyz(os.path.join(args.dft_dir, xyzout),
                                     tmp_xyz):
                print(f"  [skip] failed to extract geometry from {xyzout}")
                continue
            log = run_xtb_one(tmp_xyz, charge, mult=1,
                              gfn=args.gfn, solvent=solvent,
                              acc=args.acc, timeout=args.timeout,
                              raw_log=raw_log)
        finally:
            shutil.rmtree(workdir, ignore_errors=True)
        if log is None:
            continue
        res = parse_xtb_out(log)
        rows.append(summary_row(name, smiles, role, charge, solvent or "",
                                args.gfn, res))
        ok_str = "OK " if res["ok"] else "FAIL"
        homo = f"{res['homo_eV']:+.4f}" if res["homo_eV"] is not None else "  -  "
        lumo = f"{res['lumo_eV']:+.4f}" if res["lumo_eV"] is not None else "  -  "
        gap  = f"{res['gap_eV']:+.4f}"  if res['gap_eV']  is not None else "  -  "
        mu   = f"{res['dipole_D']:.3f}"  if res['dipole_D'] is not None else "  -  "
        print(f"  -> {ok_str}  HOMO={homo} eV  LUMO={lumo} eV  gap={gap} eV  "
              f"|mu|={mu} D")

    out_csv = os.path.join(args.dft_dir, args.output)
    with open(out_csv, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_HEADER)
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {out_csv}")


if __name__ == "__main__":
    main()
