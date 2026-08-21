# CO2 Cycloaddition ML — Documentation Index

This document guides you to the right part of the documentation.

---

## For GitHub Visitors

Start here:
- **[README.md](../README.md)** — One-page overview: what this project is, key results, quick start, citation

---

## For Reproducing Results

Execution guide:
- **[PIPELINE.md](./PIPELINE.md)** — Complete pipeline execution guide: environment setup, 13 tiers, PowerShell runner, DFT/ORCA setup, troubleshooting

---

## For Understanding Results

Numerical results and ablation studies:
- **[RESULTS.md](./RESULTS.md)** — All tables: lambda scan, DRFP ablation, model benchmarks, LOSO/LOMO, external holdout, y-randomization, statistical tests, DFT-xTB calibration

---

## For Manuscript and Supplementary Information

Paper drafts and SI:
- **[paper/](./paper/)** — Manuscript drafts (English and Chinese)
- **[supplementary/](./supplementary/)** — Full supplementary information (SI)

### Supplementary Information Index

| Section | Topic | Key File |
|---------|-------|----------|
| **S1** | Dataset & funnel audit | [S1_data_audit.md](./supplementary/S1_data_audit.md) |
| **S2** | Lambda / DRFP / latent space ablation | [S2_ablation.md](./supplementary/S2_ablation.md) |
| **S3.1** | Per-substrate SHAP | [S3_1_per_substrate_shap.md](./supplementary/S3_1_per_substrate_shap.md) |
| **S3.2** | LOSO×LOMO anomaly explanation | [S3_2_loso_lomo_anomaly.md](./supplementary/S3_2_loso_lomo_anomaly.md) |
| **S3.3** | 8-protocol × 2-feature-set matrix | [S3_protocol_matrix.md](./supplementary/S3_protocol_matrix.md) |
| **S3.4** | LOSO/LOMO/GroupKFold × 4 models | [S3_4_model_benchmark_full.md](./supplementary/S3_4_model_benchmark_full.md) |
| **S4** | External holdout + temporal OOD | [S4_external_holdout.md](./supplementary/S4_external_holdout.md) |
| **S5** | Statistical tests (y-rand, Wilcoxon) | [S5_statistical_tests.md](./supplementary/S5_statistical_tests.md) |
| **S5.1** | Bootstrap CI for per-substrate SHAP | [S5_1_bootstrap_ci.md](./supplementary/S5_1_bootstrap_ci.md) |
| **S6** | DFT-xTB calibration | [S6_dft_xtb_calibration.md](./supplementary/S6_dft_xtb_calibration.md) |
| **S7** | Transition state calculations | [S7_ige_transition_state.md](./supplementary/S7_ige_transition_state.md) |

---

## Source Code Structure

| Directory | Description |
|-----------|-------------|
| `src/data/` | Data cleaning, DRFP encoding, GFN2-xTB descriptors |
| `src/models/benchmarks/` | Model benchmarks: GroupKFold, sampling, statistics, external holdout |
| `src/models/persistence/` | PCL-AE training, model persistence |
| `src/models/screening/` | Virtual screening, ranking metrics |
| `src/analysis/loso/` | LOSO/LOMO cross-validation, per-substrate SHAP |
| `src/analysis/mechanism/` | Catalyst mechanism classification, substrate features |
| `src/analysis/diagnostics/` | Paper abstract, substrate-catalyst matrix, CHO diagnostic |
| `src/dft/` | ORCA DFT input generation, output parsing, xTB-DFT comparison |
| `src/visualization/` | Paper figure scripts (bilingual EN/ZH) |
| `src/ci_artifacts/` | SI artifact generators |
| `scripts/` | PowerShell pipeline runner |

---

## Results Directory

| Directory | Contents |
|-----------|----------|
| `results_best_pipeline/` | Ablation results, saved models |
| `results_lambda_ablation/` | Lambda sweep results |
| `results_pcl_ae/` | PCL-AE encoder |
| `results_step4/` | LOSO/LOMO results |
| `results_step4_5/` | Per-substrate SHAP |
| `results_si/` | SI benchmarks |
| `results_shap_comprehensive/` | SHAP control experiments |
| `results_dft/` | DFT vs xTB results |
| `dft_validation/` | ORCA inputs/outputs |
