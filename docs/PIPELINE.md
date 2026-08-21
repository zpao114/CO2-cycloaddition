# CO2 Cycloaddition ML Pipeline — Execution Guide

This guide covers the complete pipeline execution for the manuscript *"Cross-substrate transferability in CO2 cycloaddition is bounded by substrate-mechanism orthogonality"* (target: *ACS Catalysis*).

For a quick overview and citation, see [README.md](../README.md).

---

## Table of Contents

1. [Environment Setup](#environment-setup)
2. [Pipeline Tiers](#pipeline-tiers)
3. [Tier Execution Order](#tier-execution-order)
4. [PowerShell Runner](#powershell-runner)
5. [Tier Details](#tier-details)
6. [DFT/ORCA Setup](#dftorca-setup)

---

## Environment Setup

### Requirements

```bash
conda create -n co2_ml python=3.10
conda activate co2_ml
pip install -r requirements.txt
conda install -c conda-forge xtb==22.1
```

Key packages: numpy 1.26.4, pandas >=2.0, scikit-learn 1.9.0, xgboost >=2.0, lightgbm >=4.0, torch >=2.0, rdkit >=2024.3.1, drfp 0.3.6, shap 0.44.0, matplotlib 3.7.1.

ORCA 6.x is required for DFT calculations (not pip-installable). Install from https://orcaforum.kofo.mpg.de/.

### Environment Variables

```powershell
# Default paths (override with env vars)
$env:CO2_PROJECT_ROOT = "D:\machine-learning\CO2-cycloaddition"
$env:CO2_PYTHON      = "D:\co2\env_drfp\python.exe"
```

---

## Pipeline Tiers

The pipeline is organized into 13 tiers. Run them in order unless using `-Resume` or `-Tier`.

| Tier | Name | Duration | Description |
|------|------|----------|-------------|
| 0 | `tier_data_split` | <1 min | Canonical 5-fold split (seed=2026) |
| 1 | `tier_data` | ~10 min (or ~6 hr with xTB) | Data cleaning, DRFP encoding, GFN2-xTB descriptors |
| 2 | `tier_ablation` | ~163 min | DRFP variant + lambda sweep + full benchmark |
| 3 | `tier_pcl` | ~5 min | PCL-AE training |
| 4 | `tier_main` | ~25 min | Core benchmarks: GroupKFold, sampling, statistics, external holdout |
| 5 | `tier_screening` | ~15 min | Virtual screening, ranking metrics |
| 6 | `tier_validation` | ~15 min | Secondary external validation |
| 7 | `tier_loso` | ~15 min | LOSO/LOMO cross-validation, per-substrate SHAP |
| 8 | `tier_si` | ~10-90 min | SI artifact generation: LOSO matrix, year-OOD, y-rand, bootstrap |
| 9 | `tier_abstract` | ~2 min | Paper abstract generation |
| 10 | `tier_regen` | ~5 min | Regenerate all v3 figures |
| 11 | `tier_figures` | ~10 min | Individual paper figures |
| 12 | `tier_dft` | ~45 min | DFT/ORCA validation (needs WSL) |
| S7 | `tier_s7` | varies | Extended ORCA inputs |

---

## Tier Execution Order

### Minimal Run (core results)

```powershell
# Tier 0-4: data -> features -> ablation -> benchmarks
.\scripts\run_pipeline_v2.ps1 -Tier tier_data_split
.\scripts\run_pipeline_v2.ps1 -Tier tier_data
.\scripts\run_pipeline_v2.ps1 -Tier tier_ablation
.\scripts\run_pipeline_v2.ps1 -Tier tier_pcl
.\scripts\run_pipeline_v2.ps1 -Tier tier_main

# Tier 7: LOSO analysis
.\scripts\run_pipeline_v2.ps1 -Tier tier_loso
```

### Full Run (excluding DFT)

```powershell
.\scripts\run_pipeline_v2.ps1 -Tier tier_data_split
.\scripts\run_pipeline_v2.ps1 -Tier tier_data
.\scripts\run_pipeline_v2.ps1 -Tier tier_ablation
.\scripts\run_pipeline_v2.ps1 -Tier tier_pcl
.\scripts\run_pipeline_v2.ps1 -Tier tier_main
.\scripts\run_pipeline_v2.ps1 -Tier tier_screening
.\scripts\run_pipeline_v2.ps1 -Tier tier_validation
.\scripts\run_pipeline_v2.ps1 -Tier tier_loso
.\scripts\run_pipeline_v2.ps1 -Tier tier_si
.\scripts\run_pipeline_v2.ps1 -Tier tier_abstract
.\scripts\run_pipeline_v2.ps1 -Tier tier_regen
.\scripts\run_pipeline_v2.ps1 -Tier tier_figures
```

### Full Run (including DFT)

```powershell
# After all above tiers complete, run DFT on WSL/Linux:
cd dft_validation
orca input.inp > output.out

# Then resume pipeline (mechanism analysis + LOSO improvement run automatically)
.\scripts\run_pipeline_v2.ps1 -Tier tier_dft -WaitDFT 120
```

---

## PowerShell Runner

`scripts/run_pipeline_v2.ps1` (Phase 2, 2026-08-19) manages all tier execution.

### Common Commands

```powershell
# List all tiers with durations
.\scripts\run_pipeline_v2.ps1 -List

# Show config and exit
.\scripts\run_pipeline_v2.ps1 -Diagnostic

# Dry run (preview commands)
.\scripts\run_pipeline_v2.ps1 -DryRun

# Resume: skip steps whose output is already up-to-date
.\scripts\run_pipeline_v2.ps1 -Resume

# Skip xTB (~6 hr saved)
.\scripts\run_pipeline_v2.ps1 -NoXTB

# Skip DFT tier
.\scripts\run_pipeline_v2.ps1 -SkipDFT

# Per-step timeout (minutes)
.\scripts\run_pipeline_v2.ps1 -StepTimeout 30
```

### Exit Codes

| Code | Meaning |
|------|---------|
| 0 | All required steps passed |
| 1 | At least one required step failed |
| 2 | Required steps passed, optional steps raised warnings |

---

## Tier Details

### Tier 0 — Canonical Split

Generates yield-stratified 5-fold cross-validation splits.

```
Input:  data/processed/co2_drfp_xtb_extended.csv (2,316 reactions)
Output: results_data_split/data_split.json (seed=2026)
Script: src/data_split.py
```

### Tier 1 — Data Preparation

Cleans raw data, generates DRFP fingerprints, runs GFN2-xTB calculations.

| Step | Script | Output | Duration |
|------|--------|--------|----------|
| 101_clean | `src/data/101_clean.py` | `data/processed/cleaned.csv` | <1 min |
| 102_smiles | `src/data/102_smiles.py` | `data/processed/co2_smiles.csv` | <5 min |
| 103_drfp | `src/data/103_drfp.py` | `data/processed/co2_drfp.csv` | <5 min |
| 104b_run_xtb | `src/data/104b_run_xtb_extended.py` | `results_cho_diagnostic/xtb_results_summary.csv` | ~6 hr |
| 105b_xtb_sanity | `src/data/105b_xtb_sanity_v2.py` | `results_cho_diagnostic/xtb_sanity_summary.csv` | <1 min |
| 107_merge | `src/data/107_merge_substrate_xtb.py` | `results_cho_diagnostic/co2_drfp_xtb_extended.csv` | <1 min |

### Tier 2 — Ablation

Performs DRFP variant ablation, lambda sweep, and full benchmark in three stages.

```
Input:  data/processed/co2_drfp_xtb_extended.csv
Output: results_lambda_ablation/lambda_results.csv
        results_best_pipeline/drfp_ablation_meta.json
        results_best_pipeline/full_benchmark_results.csv
Script: src/data/201_ablation.py
Duration: ~163 min
```

**Stage 1**: 4 DRFP variants x XGBoost -> best variant
**Stage 2**: 17-point lambda scan x 3 seeds x 5 folds -> BEST_LAMBDA_PROP
**Stage 3**: 7 feature configs x 4 models -> full benchmark results

**Outputs auto-loaded by `src/config.py`**:
- `BEST_LAMBDA_PROP = 200.0`
- `BEST_LAMBDA_PROP_R2 = 0.4149`
- `BEST_DRFP_VARIANT = 'full'`

### Tier 3 — PCL-AE Training

Trains Property-Co-Learning AutoEncoder with auto-tuned lambda.

```
Input:  data/processed/co2_drfp_xtb_extended.csv
Output: results_pcl_ae/pcl_ae_encoder.pt
Script: src/models/persistence/train_pcl_ae.py
```

### Tier 4 — Main Benchmarks

Core model evaluation suite.

| Step | Script | Output |
|------|--------|--------|
| 302_groupkfold | `src/models/benchmarks/302_groupkfold_validation.py` | `results_groupkfold_validation/ML_groupkfold_results.csv` |
| 303_sampling | `src/models/benchmarks/303_sample_size_sensitivity.py` | `results_sample_size_sensitivity/learning_curve_summary.csv` |
| 304_stat_sig | `src/models/benchmarks/304_statistical_significance.py` | `results_statistical_test/wilcoxon_results.csv` |
| 306_external | `src/models/benchmarks/306_external_validation.py` | `results_external_validation/STAGE6_FINAL_REPORT.txt` |
| 401_persist | `src/models/persistence/401_persist_best_pipeline.py` | `results_best_pipeline/artifacts/*.pt` |

### Tier 5 — Virtual Screening

Ranking performance evaluation.

| Step | Script | Output |
|------|--------|--------|
| 310_top10 | `src/models/benchmarks/310_known_top10_baseline.py` | `results_virtual_screening/top10_results.csv` |
| 403b_ranking | `src/models/screening/403b_ranking_metrics.py` | `results_ranking_metrics/ranking_metrics_summary.csv` |

### Tier 6 — Validation

Secondary external holdout validation.

```
Script: src/models/persistence/405_external_validation.py
Output: results_external_validation/external_validation_results.csv
```

### Tier 7 — LOSO/LOMO Analysis

Cross-substrate and cross-mechanism transferability analysis.

| Step | Script | Output |
|------|--------|--------|
| 700_loso_lomo | `src/analysis/loso/700_loso_lomo_cv.py` | `results_step4/summary_protocol.csv` |
| 701_per_sub_shap | `src/analysis/loso/701_per_substrate_shap.py` | `results_step4_5/per_substrate_shap.csv` |

Note: `702_integrated`, `705_improved`, `706_root_cause`, `901_substrate_cat`, `902_cho_diag` run inside `tier_dft` after ORCA completes.

### Tier 8 — SI Artifact Generation

Supplementary information benchmarks.

| Step | Script | Duration |
|------|--------|----------|
| si_s3 | `src/ci_artifacts/generate_si_s3_benchmark_full_v3_1.py` | ~10 min |
| year_ood | `src/ci_artifacts/generate_year_ood_benchmark.py` | ~10 min |
| yrand_100 | `src/ci_artifacts/generate_y_randomization_v4_100perm.py` | ~5 min |
| boot_substrate | `src/ci_artifacts/generate_bootstrap_substrate_ci.py` | <5 min |
| 803_mordred | `src/models/benchmarks/803_mordred_ablation.py` | optional |
| 804_hier | `src/models/benchmarks/804_hierarchical_catalyst_model.py` | optional |
| 807_cross_model | `src/models/benchmarks/807_cross_model_shap.py` | ~5 min |
| 808_catalyst_ctrl | `src/models/benchmarks/808_catalyst_control.py` | ~5 min |
| 809_condition_ctrl | `src/models/benchmarks/809_condition_control.py` | ~5 min |
| 810_yield_dist | `src/models/benchmarks/810_yield_distribution.py` | <5 min |

### Tier 9 — Diagnostics

```powershell
Script: src/analysis/diagnostics/900_paper_abstract.py
Output: paper_text/abstract_combined.md
```

### Tier 10 — Figure Generation

Individual paper figures (bilingual EN/ZH):

```powershell
# Regenerate all v3 figures
python src/visualization/regen_all_v3.py

# Or individual:
python src/visualization/fig_0_graphical_abstract.py
python src/visualization/fig1_protocol_comparison_en.py
python src/visualization/fig1_protocol_comparison_zh.py
# ... fig2-fig5 (en/zh pairs)
```

### Tier 11 — Extended ORCA Inputs

```powershell
# Generates additional ORCA inputs for extended DFT validation
Script: src/dft/713_orca_dft_generator.py (optional)
```

---

## DFT/ORCA Setup

DFT calculations require ORCA 6.x on WSL/Linux. Windows standalone ORCA is not supported.

### Step 1: Generate ORCA Inputs (Windows)

```powershell
.\scripts\run_pipeline_v2.ps1 -Tier tier_dft
```

This generates input files in `dft_validation/` directory. Copy the entire folder to your WSL/Linux machine.

### Step 2: Run ORCA (WSL/Linux)

```bash
cd dft_validation
# Edit orca_inputs.txt to list all .inp files, then:
while IFS= read -r inp; do
    orca "$inp" > "${inp%.inp}.out" 2>&1
done < orca_inputs.txt
```

ORCA typically runs 24-48 hours for the full set depending on hardware.

### Step 3: Resume Pipeline

```powershell
# After ORCA completes, copy results back to Windows
# Then resume:
.\scripts\run_pipeline_v2.ps1 -Tier tier_dft -WaitDFT 5
```

After ORCA results are detected, these steps run automatically:
1. Parse ORCA outputs (`510_parse_dft_outputs.py`)
2. xTB on DFT geometry (`512_xtb_on_dft_geometry.py`)
3. DFT vs xTB comparison (`514_dft_vs_xtb_report.py`)
4. Transition state analysis (`514b_dft_transition_state.py`)
5. Mechanism analysis (`601_catalyst_mechanism_v2.py`)
6. Substrate features (`602_substrate_features.py`)
7. Transferability matrix (`603_transferability_matrix.py`)
8. Integrated report (`702_integrated_report.py`)
9. Improved LOSO (`705_improved_loso.py`)
10. Root cause figure (`706_loso_root_cause_figure.py`)
11. Substrate-catalyst matrix (`901_substrate_catalyst_matrix.py`)
12. CHO mechanistic diagnostic (`902_cho_mechanistic_diagnostic.py`)

---

## Troubleshooting

### xTB fails to start
- Ensure `xtb` binary is in PATH or installed via `conda install -c conda-forge xtb==22.1`
- Check Windows Defender or antivirus is not blocking the binary

### ORCA fails on WSL
- Ensure ORCA license is activated (`oracrc` file)
- Check memory requirements (minimum 8 GB RAM per job recommended)

### Out of memory during PCL-AE training
- Reduce batch size in `train_pcl_ae.py`
- Use CPU mode by setting `$env:CUDA_VISIBLE_DEVICES=""`

### Script import errors
- Ensure `PYTHONPATH` includes both project root and `src/`:
  ```powershell
  $env:PYTHONPATH = "$env:CO2_PROJECT_ROOT;$env:CO2_PROJECT_ROOT\src"
  ```
- Or run from project root: `python src/...`

---

## Results Directory Structure

```
results/
|-- results_best_pipeline/        # ablation, model artifacts
|   |-- drfp_ablation_meta.json
|   |-- drfp_ablation_results.csv
|   |-- full_benchmark_results.csv
|   |-- save_best_model_report.txt
|   `-- artifacts/                 # saved models
|
|-- results_lambda_ablation/      # lambda sweep results
|   |-- lambda_results.csv
|   `-- lambda_results.txt
|
|-- results_pcl_ae/              # PCL-AE model
|   `-- pcl_ae_encoder.pt
|
|-- results_data_split/          # 5-fold splits
|   `-- data_split.json
|
|-- results_groupkfold_validation/
|-- results_sample_size_sensitivity/
|-- results_statistical_test/
|-- results_external_validation/
|-- results_virtual_screening/
|-- results_ranking_metrics/
|-- results_step4/               # LOSO/LOMO results
|-- results_step4_5/             # per-substrate SHAP
|-- results_step5/               # integrated report
|-- results_step7_improved_loso/
|-- results_si/                  # SI benchmarks
|-- results_shap_comprehensive/   # SHAP control experiments
|-- results_mechanism/
|-- results_transferability/
|-- results_substrate_catalyst_matrix/
|-- results_cho_diagnostic/
|-- results_dft/                 # DFT vs xTB results
|-- dft_validation/               # ORCA inputs/outputs
`-- paper_text/
    `-- abstract_combined.md
```
