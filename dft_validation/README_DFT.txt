DFT Validation Set
=================
Migrated from: D:\machine-learning\CO2 cycloaddition
Date: 2026-08-20

Directory Structure
-------------------
dft_validation/
├── inputs/                    <- ORCA input files (.inp)
│   ├── orca_example.inp
│   ├── ige_reactant_v2.inp
│   ├── ige_ts_v2.inp
│   └── scan_d3.0.inp
├── results/                    <- Calculation results (.csv)
│   ├── xtb_results_summary.csv
│   └── dft_xtb_calibration_full.csv
├── README_DFT.txt              <- This file

Structure notes: molecular structures (.xyz) for DFT/ORCA inputs are sourced
directly from assets/molecular_structures/ (single source of truth). Earlier
plans to maintain a mirrored dft_validation/xyz_structures/ subfolder were
abandoned because it duplicated data without adding value.

Migration Status
----------------
[RESOLVED] dft_validation/xyz_structures/ directory — no longer required.
  150 xyz files are now served from assets/molecular_structures/ instead.

Migration Notes
---------------
This dft_validation/ folder was bootstrapped from an older project layout at
"D:\machine-learning\CO2 cycloaddition" on 2026-08-20. After the 2026-08-21
cleanup pass, xyz structures are no longer mirrored here; they live in
assets/molecular_structures/ and are referenced directly by the pipeline.

Pipeline Reference (New Library)
-------------------------------
501  python src/dft/501_generate_dft_inputs.py   Generate ORCA inputs
510  python src/dft/510_parse_dft_outputs.py      Parse ORCA outputs
512  python src/dft/512_xtb_on_dft_geometry.py  xTB on DFT geometries
514  python src/dft/514_dft_vs_xtb_report.py     xTB vs DFT comparison
520  python src/dft/520_dft_journal_figures.py   Paper figures

Data Quality Notes
-------------------
- dft_results_summary.csv from old library contains TBAI_anion anomaly (HOMO = -23.85 eV)
- Recommend regenerating using 501 + 510 in the new library
- dft_xtb_calibration_full.csv contains calibration data with TBAI anomalies flagged
