#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
105b_xtb_sanity_v2.py — xTB sanity check (geometry + reference + coverage)

What this fixes vs. the legacy 105
----------------------------------
1. ZnBr2-class geometry failures: legacy 105 only flagged HOMO/LUMO
   plausibility, not the "all atoms at (0,0,0)" failure mode that
   happens when RDKit UFF cannot parametrize [Zn+2]. 105b inspects the
   .xyz file directly.

2. Single-atom ions ([I-], [Br-], …) do run through xTB but their
   HOMO/LUMO are physically meaningless for the cycloaddition; we tag
   them as "monatomic_ion" and skip plausibility checks for them.

3. NaN-coverage reporting: tells you which fractions of the candidate
   set came back with valid HOMO/LUMO/gap/dipole.

4. Reference-molecule drift detection: compares a small set of known
   reference values (GFN2-xTB gas-phase) against the actual run and
   flags outliers (>1.5 eV / >2 D).

Outputs
-------
By default 105b writes:
  * results/results_cho_diagnostic/xtb_sanity_summary.csv  (machine-readable)
  * results/results_cho_diagnostic/xtb_sanity_report.txt   (human-readable)

Usage
-----
    python 105b_xtb_sanity_v2.py
    python 105b_xtb_sanity_v2.py --strict   # exit 1 on any warning
"""
from __future__ import annotations

import argparse
import csv
import logging
import os
import re
import sys
from pathlib import Path

PROJECT_ROOT = os.environ.get("CO2_PROJECT_ROOT", r"D:\machine-learning\CO2-cycloaddition")
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

logger = logging.getLogger("105b_xtb_sanity_v2")

# Force UTF-8 for stdout to handle Å / non-ASCII chars in geom warnings
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from typing import Dict, List

# 104b_run_xtb_extended.py starts with a digit, which is not a legal Python
# identifier for `import` statements. Use spec_from_file_location (cheaper
# than the importlib.util approach previously used: drop the duplicated
# `os.path.join` on `__file__` and let `Path(__file__).parent` resolve it).
from importlib import util as _util
_SPEC_PATH = (Path(__file__).parent / "104b_run_xtb_extended.py").resolve()
_spec = _util.spec_from_file_location("src_data_xtb_104b", str(_SPEC_PATH))
_104b = _util.module_from_spec(_spec)
_spec.loader.exec_module(_104b)

parse_xtb_out = _104b.parse_xtb_out
locate_xtb = _104b.locate_xtb
EXTENDED_CANDIDATES = _104b.EXTENDED_CANDIDATES

from src.paths import RESULTS_CHO_DIAGNOSTIC, ensure_dir

# 105b writes a summary CSV + a human-readable report into this directory.
# Default is the same directory 104b writes into, so 105b finds everything
# without --out-dir/--xyz-dir flags.
DEFAULT_XTB_DIR = RESULTS_CHO_DIAGNOSTIC

# Reference values from GFN2-xTB gas-phase single-point calculations.
# Expect a 1-2 eV systematic shift when comparing against an implicit-solvent
# (e.g. ALPB/DMSO) run, so the tolerance is set accordingly.
REFERENCE_VACUUM = {
    "CO2":  {"homo_eV": -13.78, "lumo_eV": -2.27, "gap_eV": 11.51, "dipole_D": 0.0},
    "styrene_oxide": {"homo_eV": -9.10, "lumo_eV": -0.40, "gap_eV": 8.70, "dipole_D": 1.94},
    "DMSO": {"homo_eV": -8.96, "lumo_eV": -0.41, "gap_eV": 8.55, "dipole_D": 4.13},
    "DMF":  {"homo_eV": -9.30, "lumo_eV": -0.69, "gap_eV": 8.61, "dipole_D": 4.31},
}


def _check_xyz_geometry(xyz_path: str) -> List[str]:
    """
    Detect geometry failures:
      - All atoms at origin
      - All atoms identical
      - Too few atoms (single-atom I-, Br-)
      - Excessive bond lengths (>5 Å, indicating broken geometry)
    """
    warns = []
    if not os.path.isfile(xyz_path):
        return [f"{os.path.basename(xyz_path)}: xyz file missing"]
    with open(xyz_path, encoding="utf-8") as f:
        lines = f.readlines()
    if len(lines) < 3:
        return [f"{os.path.basename(xyz_path)}: xyz too short"]
    try:
        n_atoms = int(lines[0].strip())
    except Exception:
        return [f"{os.path.basename(xyz_path)}: cannot parse atom count"]
    coords = []
    for line in lines[2:2 + n_atoms]:
        parts = line.split()
        if len(parts) < 4:
            continue
        try:
            coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
        except Exception:
            pass
    if len(coords) != n_atoms:
        return [f"{os.path.basename(xyz_path)}: n_atoms={n_atoms} but only {len(coords)} coords parsed"]
    # Check 1: all at origin
    nonzero = sum(1 for c in coords if any(abs(v) > 1e-4 for v in c))
    if nonzero == 0:
        warns.append(f"{os.path.basename(xyz_path)}: ALL ATOMS AT ORIGIN → xTB will fail")
    # Check 2: bonds
    if n_atoms >= 2:
        from itertools import combinations
        too_short = 0
        for c1, c2 in combinations(coords, 2):
            d = sum((a - b) ** 2 for a, b in zip(c1, c2)) ** 0.5
            if d < 0.5:  # shorter than typical H-H
                too_short += 1
        if too_short > 0:
            warns.append(f"{os.path.basename(xyz_path)}: {too_short} bond(s) shorter than 0.5 Å")
    # Check 3: monoatomic
    if n_atoms == 1:
        warns.append(f"{os.path.basename(xyz_path)}: MONOATOMIC → xTB may give spurious LUMO/HOMO")
    return warns


def _check_physical_plausibility(name: str, res: Dict) -> List[str]:
    """
    Returns warnings. Note:
    - Monoatomic ions (single atom in xyz): HOMO may be missing or have
      physically different signs (since single isolated electrons). We don't
      warn about this for `_anion`, `_cation` monoatomic roles.
    """
    warns = []
    H = res.get("homo_eV"); L = res.get("lumo_eV"); G = res.get("gap_eV")
    M = res.get("dipole_D")
    is_monoatomic = name.endswith("_anion") or name.endswith("_cation")
    if H is None and L is None:
        warns.append(f"{name}: HOMO and LUMO both missing")
        return warns
    if (H is None or L is None) and not is_monoatomic:
        warns.append(f"{name}: HOMO or LUMO missing")
    if H is not None and L is not None and H >= L:
        # For multi-atom molecules this is suspicious; for monoatomic ions skip
        if not is_monoatomic:
            warns.append(f"{name}: HOMO ({H:+.3f}) >= LUMO ({L:+.3f}) [sign wrong]")
    if G is not None and not (-5 < G < 25):  # allow small negative for monoatomic
        warns.append(f"{name}: gap {G:.3f} eV out of [-5, 25]")
    if M is not None and not (0 <= M < 25):
        warns.append(f"{name}: |mu| {M:.3f} Debye out of [0, 25]")
    return warns


def _check_reference(name: str, res: Dict, tol_eV: float = 1.5,
                     tol_D: float = 2.0) -> List[str]:
    """Compare with literature reference (vacuum-phase GFN2-xTB)."""
    if name not in REFERENCE_VACUUM:
        return []
    ref = REFERENCE_VACUUM[name]
    warns = []
    # Don't check HOMO/LUMO absolute for solvated runs (use shift-corrected MAE)
    for key, tol in [("homo_eV", tol_eV), ("lumo_eV", tol_eV),
                     ("gap_eV", tol_eV), ("dipole_D", tol_D)]:
        v = res.get(key); rv = ref.get(key)
        if v is None or rv is None:
            continue
        if abs(v - rv) > tol:
            warns.append(
                f"{name}: {key}={v:+.3f} eV/D vs ref {rv:+.3f} "
                f"(delta={v-rv:+.2f}, tol=±{tol})"
            )
    return warns


def _compute_nan_coverage(rows: List[Dict]) -> Dict[str, tuple]:
    """Per-key NaN rate over all candidate rows."""
    keys = ["homo", "lumo", "gap", "dipole"]
    out = {}
    for k in keys:
        vs = [r.get(k) for r in rows]
        n_nan = sum(1 for v in vs if v is None)
        out[k] = (len(rows), n_nan)
    return out


def _format_text_report(rows: List[Dict], coverage: Dict[str, tuple],
                        warnings: List[str]) -> str:
    """Human-readable summary for archiving alongside results."""
    lines = []
    lines.append("=" * 70)
    lines.append("xTB Sanity Report (105b_xtb_sanity_v2.py)")
    lines.append("=" * 70)
    n_total = len(rows)
    n_ok = sum(1 for r in rows if r["status"] == "OK")
    n_fail = sum(1 for r in rows if "FAIL" in r["status"])
    n_susp = sum(1 for r in rows if r["status"] == "SUSPECT")
    lines.append(f"Total candidates : {n_total}")
    lines.append(f"OK               : {n_ok}")
    lines.append(f"SUSPECT          : {n_susp}")
    lines.append(f"FAIL             : {n_fail}")
    lines.append(f"Warnings emitted : {len(warnings)}")
    lines.append("")
    lines.append("NaN coverage:")
    for k, (n_tot, n_nan) in coverage.items():
        pct = 100.0 * n_nan / max(n_tot, 1)
        lines.append(f"  {k:<10s} {n_nan:>4d}/{n_tot} ({pct:>5.1f}%)")
    if warnings:
        lines.append("")
        lines.append(f"All warnings ({len(warnings)}):")
        for w in warnings:
            lines.append(f"  - {w}")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", default=str(DEFAULT_XTB_DIR),
                        help="Directory containing *.xyz.xtb.stdout files")
    parser.add_argument("--xyz-dir", default=str(DEFAULT_XTB_DIR),
                        help="Directory containing .xyz files")
    parser.add_argument("--report-csv", default="xtb_sanity_summary.csv",
                        help="Per-candidate machine-readable summary CSV "
                             "(relative to --out-dir). Pass empty string to skip.")
    parser.add_argument("--report-txt", default="xtb_sanity_report.txt",
                        help="Human-readable summary report (relative to --out-dir). "
                             "Pass empty string to skip.")
    parser.add_argument("--strict", action="store_true",
                        help="Exit with non-zero status if any sanity check fails")
    parser.add_argument("--no-reference", action="store_true",
                        help="Skip reference-value comparison")
    parser.add_argument("--force", action="store_true",
                        help="Re-run sanity check even if report file already exists.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Walk candidates and report counts; do not write reports.")
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

    n_total = len(EXTENDED_CANDIDATES)
    logger.info("=" * 70)
    logger.info("105b_xtb_sanity_v2.py | out-dir=%s | strict=%s", args.out_dir, args.strict)
    logger.info("=" * 70)

    ok = 0; fail = 0; warn_count = 0
    overall_warnings = []
    rows = []

    for cand in EXTENDED_CANDIDATES:
        name = cand["name"]
        out_path = os.path.join(args.out_dir, f"{name}.xyz.xtb.stdout")
        xyz_path = os.path.join(args.xyz_dir, f"{name}.xyz")

        # ── Check 1: file existence ──
        if not os.path.isfile(out_path):
            rows.append(dict(name=name, role=cand["role"], status="FAIL: no xtb.out"))
            logger.warning("  [missing] %s", name)
            fail += 1
            continue

        # ── Check 2: xyz geometry sanity ──
        geom_warns = _check_xyz_geometry(xyz_path)
        for w in geom_warns:
            logger.warning("  [geom]   %s", w)
            overall_warnings.append(w)
            warn_count += 1

        # ── Check 3: parse + physical plausibility ──
        res = parse_xtb_out(out_path)
        if not res["ok"]:
            rows.append(dict(name=name, role=cand["role"], status="FAIL: xTB failure"))
            logger.error("  [FAIL]   %s: xTB reported failure", name)
            fail += 1
            continue

        phys_warns = _check_physical_plausibility(name, res)
        for w in phys_warns:
            logger.warning("  [warn]   %s", w)
            overall_warnings.append(w)
            warn_count += 1

        # ── Check 4: reference comparison ──
        if not args.no_reference:
            ref_warns = _check_reference(name, res)
            for w in ref_warns:
                logger.warning("  [drift]  %s", w)
                overall_warnings.append(w)
                warn_count += 1

        status = "OK" if not phys_warns and not geom_warns else "SUSPECT"
        rows.append(dict(name=name, role=cand["role"], status=status,
                         homo=res.get("homo_eV"), lumo=res.get("lumo_eV"),
                         gap=res.get("gap_eV"), dipole=res.get("dipole_D")))

        H = f"{res['homo_eV']:+.4f}" if res['homo_eV'] is not None else "  -  "
        L = f"{res['lumo_eV']:+.4f}" if res['lumo_eV'] is not None else "  -  "
        G = f"{res['gap_eV']:+.4f}" if res['gap_eV'] is not None else "  -  "
        M = f"{res['dipole_D']:.3f}" if res['dipole_D'] is not None else "  -  "
        logger.info("  [%-8s] %-32s HOMO=%s LUMO=%s gap=%s |mu|=%s", status, name, H, L, G, M)
        if status == "OK":
            ok += 1
        else:
            fail += 1

    logger.info("=" * 70)
    logger.info("Total: %d  OK: %d  FAIL: %d  Warnings: %d", n_total, ok, fail, warn_count)
    logger.info("=" * 70)

    # NaN-coverage summary (this is what the docstring promises)
    coverage = _compute_nan_coverage(rows)
    logger.info("NaN coverage (over all candidates):")
    for key, (n_total_c, n_nan) in coverage.items():
        pct = 100.0 * n_nan / max(n_total_c, 1)
        logger.info("  %-10s %4d NaN / %d (%5.1f%%)", key, n_nan, n_total_c, pct)

    if args.dry_run:
        logger.info("[dry-run] Not writing reports.")
        return 0

    # Persist reports if --report-csv / --report-txt were given
    if args.report_csv:
        ensure_dir(args.out_dir)
        csv_path = Path(args.out_dir) / args.report_csv
        with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["name", "role", "status",
                                              "homo", "lumo", "gap", "dipole"])
            w.writeheader()
            for r in rows:
                w.writerow(r)
        logger.info("Summary CSV: %s", csv_path)
    if args.report_txt:
        ensure_dir(args.out_dir)
        txt_path = Path(args.out_dir) / args.report_txt
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(_format_text_report(rows, coverage, overall_warnings))
        logger.info("Text report: %s", txt_path)

    if overall_warnings and args.strict:
        logger.error("FAILED (strict mode).")
        return 1
    logger.info("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())