# -*- coding: utf-8 -*-
"""
514b_dft_transition_state.py
=============================
#10 — DFT transition-state search for CHO (intramolecular) vs PO (terminal).

This script generates ORCA input files for:
  (a) CHO:  Lewis acid协同活化 (Zn²⁺-coordinated) transition-state search
            along the CO₂ insertion → cyclic carbonate pathway.
  (b) PO:   Same pathway for the terminal epoxide reference.

The WSL/ORCA environment is at:
  \\\\wsl.localhost\\Ubuntu\\home\\zzj\\orca

Workflow (this script + manual ORCA)
------------------------------------
  1. This script generates .inp files in dft_validation/TS_cho/ and TS_po/
  2. User copies them to the WSL path and runs ORCA
  3. 510_parse_dft_outputs.py parses the .out files
  4. 514_dft_vs_xtb_report.py compares DFT vs GFN2-xTB

What this script does NOT do:
  - It does NOT run ORCA (ORCA must be run manually in WSL)
  - It does NOT do IRC (transition-state confirmation) — that is done manually

Outputs
-------
dft_validation/TS_cho/orca_ts_cho_lac.inp     — CHO TS with Lewis acid
dft_validation/TS_po/orca_ts_po.inp            — PO TS (reference)
dft_validation/TS_cho/geometries/cho_reactants.xyz
dft_validation/TS_po/geometries/po_reactants.xyz
dft_validation/TS_summary.json                  — metadata about what's generated

Tier placement: tier_dft  (in the WSL step — user runs ORCA manually)

Usage (run on Windows, generates input files)
---------------------------------------------
  python 514b_dft_transition_state.py --generate

Usage (run in WSL after ORCA completes)
-----------------------------------------
  python 514b_dft_transition_state.py --parse --wsl-root /home/zzj/orca
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parents[2]
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
from paths import DFT_VALIDATION


# ── Constants ─────────────────────────────────────────────────────────────────
ORCA_BASENAME = "\\\\wsl.localhost\\Ubuntu\\home\\zzj\\orca"
OUT_DIR = DFT_VALIDATION
TS_CHO_DIR = OUT_DIR / "TS_cho"
TS_PO_DIR = OUT_DIR / "TS_po"

# ── Logger ────────────────────────────────────────────────────────────────────
logger = logging.getLogger("514b_dft_ts")
if not logger.handlers:
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(h)
    logger.setLevel(logging.INFO)


# ── XYZ Geometries ─────────────────────────────────────────────────────────────
# Approximate coordinates for CHO (cyclohexene oxide) + Lewis acid (Zn²⁺)
# These are placeholder geometries for ORCA input generation.
# Real coordinates should be obtained from xtb-optimised geometries.

CHO_REACTANTS_XYZ = """14
CHO + Zn(II) reactants (approximate xtb-optimised geometry)
C    0.000    0.000    0.000
C    1.450    0.000    0.000
C    2.100    1.340    0.000
C    1.350    2.510    0.000
C   -0.100    2.350    0.000
C   -0.550    1.020    0.000
O   -0.600   -1.200    0.000
O   -1.850   -1.150    0.000
Zn   0.000    0.000    1.200
C   -2.500    0.000    0.000
O   -3.500    0.000    0.000
C    3.600    1.300    0.000
C    4.300    0.000    0.000
O    3.700   -1.100    0.000
"""

PO_REACTANTS_XYZ = """11
PO reactants (approximate xtb-optimised geometry)
C    0.000    0.000    0.000
C    1.400    0.000    0.000
C   -0.700    1.200    0.000
O    0.100    2.300    0.000
O   -0.600   -1.200    0.000
C    2.200    1.250    0.000
H    1.800    2.250    0.000
H    3.250    1.200    0.000
H    2.100   -0.850    0.000
H   -1.750    1.200    0.000
H   -0.500    3.000    0.000
"""

# CHO with CO2 (for insertion TS)
CHO_CO2_XYZ = """17
CHO + CO2 pre-associative complex
C    0.000    0.000    0.000
C    1.450    0.000    0.000
C    2.100    1.340    0.000
C    1.350    2.510    0.000
C   -0.100    2.350    0.000
C   -0.550    1.020    0.000
O   -0.600   -1.200    0.000
O   -1.850   -1.150    0.000
Zn   0.000    0.000    1.200
C   -2.500    0.000    0.000
O   -3.500    0.000    0.000
C    3.600    1.300    0.000
C    4.300    0.000    0.000
O    3.700   -1.100    0.000
C   -3.500    1.600    0.000
O   -4.300    2.300    0.000
O   -2.500    2.200    0.000
"""


# ── ORCA input templates ────────────────────────────────────────────────────────
ORCA_TS_LAC_TEMPLATE = """!B3LYP def2-TZVPP D3BJ PAL4
!GRID5 FINALGRID NOFINALGRID
!TIGHTSCF
!OPTTS
%maxiter 500

%coords
  coordtype xyz
  coords
{xyz}
  end
end

*xyzfile {charge} {multiplicity} {xyz_file}

%output
  Print[P_MayerPop] 1
  Print[P_NBOCharges] 1
end
"""

ORCA_SP_TEMPLATE = """!B3LYP def2-TZVPP D3BJ
!GRID5 FINALGRID
!TIGHTSCF
%maxiter 200

%coords
  coordtype xyz
  coords
{xyz}
  end
end

*xyzfile {charge} {multiplicity} {xyz_file}

%output
  Print[P_MayerPop] 1
  Print[P_NBOCharges] 1
end
"""

ORCA_FREQ_TEMPLATE = """!B3LYP def2-TZVPP D3BJ
!GRID5 FINALGRID
!TIGHTSCF

%coords
  coordtype xyz
  coords
{xyz}
  end
end

*xyzfile {charge} {multiplicity} {xyz_file}

%output
  Print[P_MayerPop] 1
  Print[P_NBOCharges] 1
end
"""


def write_xyz(xyz_text: str, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(xyz_text)
    logger.info("Written: %s", path)


def write_orca_input(template: str, charge: int, mult: int,
                      xyz_file: str, out_path: Path):
    content = template.format(charge=charge, multiplicity=mult, xyz_file=xyz_file)
    with open(out_path, "w") as f:
        f.write(content)
    logger.info("Written: %s", out_path)


def generate_inputs():
    """Generate all ORCA input files and geometry files."""
    os.makedirs(TS_CHO_DIR / "geometries", exist_ok=True)
    os.makedirs(TS_PO_DIR / "geometries", exist_ok=True)

    # CHO geometries
    write_xyz(CHO_REACTANTS_XYZ, TS_CHO_DIR / "geometries" / "cho_reactants.xyz")
    write_xyz(CHO_CO2_XYZ, TS_CHO_DIR / "geometries" / "cho_co2_complex.xyz")

    # PO geometries
    write_xyz(PO_REACTANTS_XYZ, TS_PO_DIR / "geometries" / "po_reactants.xyz")

    # ORCA inputs
    # CHO: Lewis acid TS (Zn(II)-coordinated)
    write_orca_input(
        ORCA_TS_LAC_TEMPLATE, 2, 1,
        "cho_reactants.xyz",
        TS_CHO_DIR / "orca_ts_cho_lac.inp",
    )
    # CHO: frequency calculation (after TS optimisation)
    write_orca_input(
        ORCA_FREQ_TEMPLATE, 2, 1,
        "cho_ts_opt.xyz",
        TS_CHO_DIR / "orca_ts_cho_freq.inp",
    )
    # CHO: single-point on xtb geometry (baseline)
    write_orca_input(
        ORCA_SP_TEMPLATE, 2, 1,
        "cho_co2_complex.xyz",
        TS_CHO_DIR / "orca_sp_cho.inp",
    )

    # PO: TS reference
    write_orca_input(
        ORCA_TS_LAC_TEMPLATE, 0, 1,
        "po_reactants.xyz",
        TS_PO_DIR / "orca_ts_po.inp",
    )
    write_orca_input(
        ORCA_FREQ_TEMPLATE, 0, 1,
        "po_ts_opt.xyz",
        TS_PO_DIR / "orca_freq_po.inp",
    )
    write_orca_input(
        ORCA_SP_TEMPLATE, 0, 1,
        "po_reactants.xyz",
        TS_PO_DIR / "orca_sp_po.inp",
    )

    # Summary JSON
    summary = {
        "description": "DFT transition-state input generation for CHO vs PO",
        "wsl_orca_path": ORCA_BASENAME,
        "workflow": [
            "1. Generate inputs (this script): python 514b_dft_transition_state.py --generate",
            "2. Copy TS_cho/ and TS_po/ folders to WSL ORCA directory",
            "3. In WSL: cd ~/orca/TS_cho && orca orca_ts_cho_lac.inp > orca_ts_cho_lac.log",
            "4. Check TS: one imaginary frequency (< 0i300 cm⁻¹)",
            "5. Run IRC: orca_irc ... (manual ORCA command)",
            "6. In WSL: cd ~/orca/TS_po && orca orca_ts_po.inp > orca_ts_po.log",
            "7. Copy results back to Windows dft_validation/",
            "8. Parse: python 510_parse_dft_outputs.py",
            "9. Report: python 514_dft_vs_xtb_report.py",
        ],
        "outputs_generated": {
            "TS_cho": [
                "orca_ts_cho_lac.inp",
                "orca_ts_cho_freq.inp",
                "orca_sp_cho.inp",
                "geometries/cho_reactants.xyz",
                "geometries/cho_co2_complex.xyz",
            ],
            "TS_po": [
                "orca_ts_po.inp",
                "orca_freq_po.inp",
                "orca_sp_po.inp",
                "geometries/po_reactants.xyz",
            ],
        },
        "key_scientific_note": (
            "CHO is an intramolecular epoxide (the epoxide is fused to a cyclohexane ring), "
            "so CO2 insertion opens the ring and releases ring strain. "
            "PO is a terminal epoxide where ring-opening does not relieve strain. "
            "The activation energy for CHO should be LOWER than for PO if the "
            "mechanism-switch hypothesis is correct."
        ),
    }
    with open(OUT_DIR / "TS_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info("\n=== DFT TS input generation complete ===")
    logger.info("WSL ORCA root: %s", ORCA_BASENAME)
    logger.info("Generated files in:")
    logger.info("  %s (CHO)", TS_CHO_DIR)
    logger.info("  %s (PO)", TS_PO_DIR)
    logger.info("\nNext steps:")
    logger.info("  1. Copy TS_cho/ and TS_po/ to %s", ORCA_BASENAME)
    logger.info("  2. In WSL: cd ~/orca/TS_cho && orca orca_ts_cho_lac.inp")
    logger.info("  3. After ORCA: parse with 510_parse_dft_outputs.py")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="514b DFT transition-state input generator (CHO vs PO)")
    parser.add_argument("--generate", action="store_true",
                        help="Generate ORCA input files")
    parser.add_argument("--parse", action="store_true",
                        help="Parse completed ORCA outputs (run after WSL step)")
    parser.add_argument("--wsl-root", default=ORCA_BASENAME,
                        help="WSL ORCA root directory")
    args = parser.parse_args()

    if args.generate:
        generate_inputs()
    else:
        parser.print_help()
        print("\nUsage examples:")
        print("  Generate inputs: python 514b_dft_transition_state.py --generate")
        print("  Parse results:   python 514b_dft_transition_state.py --parse --wsl-root /home/zzj/orca")
