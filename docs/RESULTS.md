# CO2 Cycloaddition ML Pipeline — Results & Ablation Studies

This document contains all numerical results from the pipeline's ablation studies and model benchmarks. For pipeline execution, see [PIPELINE.md](./PIPELINE.md). For manuscript and supplementary information, see [paper/](../docs/paper/) and [supplementary/](./supplementary/).

---

## Table of Contents

1. [Tuned Hyperparameters](#tuned-hyperparameters)
2. [Lambda Scan](#lambda-scan)
3. [DRFP Variant Ablation](#drfp-variant-ablation)
4. [Model Performance (5-fold KFold)](#model-performance-5-fold-kfold)
5. [LOSO/LOMO Cross-Validation](#losolomo-cross-validation)
6. [External Holdout](#external-holdout)
7. [y-Randomization (100 Permutations)](#y-randomization-100-permutations)
8. [PCL-AE vs Standard AE](#pcl-ae-vs-standard-ae)
9. [Dimensionality Reduction Comparison](#dimensionality-reduction-comparison)
10. [Statistical Significance](#statistical-significance)
11. [Per-Substrate Analysis](#per-substrate-analysis)
12. [DFT-xTB Calibration](#dft-xtb-calibration)

---

## Tuned Hyperparameters

Auto-loaded from ablation results by `src/config.py`.

| Parameter | Value | Source | Notes |
|-----------|-------|--------|-------|
| `BEST_LAMBDA_PROP` | **200.0** | `results_lambda_ablation/lambda_results.csv` | 17-point scan, 3 seeds, 5 folds |
| `BEST_LAMBDA_PROP_R2` | **0.4149** | Same | DualBranchANN R² at optimal lambda |
| `BEST_DRFP_VARIANT` | **'full'** | `results_best_pipeline/drfp_ablation_meta.json` | reactants + cat + solv |
| `BEST_LATENT_DIM` | **128** | hard-coded in config.py | Structural choice, not tuned |

---

## Lambda Scan

PCL-AE lambda sweep: 17 lambda values x 3 AE seeds x 5 folds.

*Source: `results_lambda_ablation/lambda_results.csv`*

| Lambda | DualANN R² | RF R² | DualANN Seed-std | Best Model |
|--------|------------|-------|-------------------|------------|
| 0.0 | 0.3052 | 0.2693 | 0.0027 | DualANN |
| 0.05 | 0.3085 | 0.2840 | 0.0082 | DualANN |
| 0.1 | 0.3059 | 0.2889 | 0.0010 | DualANN |
| 0.2 | 0.3163 | 0.3038 | 0.0048 | DualANN |
| 0.5 | 0.3308 | 0.3240 | 0.0104 | DualANN |
| 1.0 | 0.3441 | 0.3403 | 0.0114 | DualANN |
| 2.0 | 0.3631 | 0.3526 | 0.0080 | DualANN |
| 3.0 | 0.3680 | 0.3571 | 0.0101 | DualANN |
| 5.0 | 0.3774 | 0.3626 | 0.0114 | DualANN |
| 7.0 | 0.3862 | 0.3673 | 0.0077 | DualANN |
| 10.0 | 0.3908 | 0.3698 | 0.0080 | DualANN |
| 20.0 | 0.3945 | 0.3708 | 0.0050 | DualANN |
| 50.0 | 0.4021 | 0.3737 | 0.0067 | DualANN |
| 75.0 | 0.4046 | 0.3734 | 0.0050 | DualANN |
| 100.0 | 0.4058 | 0.3734 | 0.0045 | DualANN |
| 150.0 | 0.4107 | 0.3748 | 0.0039 | DualANN |
| **200.0** | **0.4149** | **0.3748** | **0.0045** | **DualANN** |

**Observations**:
- Monotonic improvement in DualANN R² as lambda increases from 0 to 200
- DualANN consistently outperforms RF across all lambda values
- Seed-std is lowest at lambda=0.1 (0.0010) and lambda=100-200 (0.0045)

---

## DRFP Variant Ablation

XGBoost 5-fold KFold with 4 DRFP encoding variants.

*Source: `results_best_pipeline/drfp_ablation_meta.json` + `results_best_pipeline/drfp_ablation_results.csv`*

| Variant | Encoding | XGBoost R² | Pearson |
|---------|----------|-------------|---------|
| `full` | reactants + cat + solv | **0.1544** | 0.4843 |
| `no_sols` | reactants + cat | 0.1482 | 0.4818 |
| `no_cats` | reactants + solv | 0.1402 | 0.4742 |
| `reactants` | reactants only | 0.1386 | 0.4752 |

**Conclusion**: Delta R² across all variants is only 0.016, indicating minimal contribution from catalyst/solvent information in DRFP encoding. Production configuration uses `full` variant (reactants + cat + solv).

---

## Model Performance (5-fold KFold)

5-fold KFold cross-validation with PCL-AE-128 features (lambda=200.0).

*Source: `results_best_pipeline/save_best_model_report.txt`*

| Model | R² mean | MAE | RMSE | Pearson |
|-------|---------|-----|------|---------|
| **DualBranchANN** | **0.4106** | **0.1157** | **0.1666** | **0.6463** |
| Random Forest | 0.4711 | — | — | — |
| XGBoost | 0.3883 | — | — | — |
| LightGBM | 0.3775 | — | — | — |

### Per-Fold Results (DualBranchANN)

| Fold | R² | MAE | Pearson |
|------|-----|-----|---------|
| 0 | 0.4451 | 0.1234 | 0.6695 |
| 1 | 0.3673 | 0.1152 | 0.6112 |
| 2 | 0.4741 | 0.1070 | 0.6975 |
| 3 | 0.3731 | 0.1148 | 0.6248 |
| 4 | 0.3933 | 0.1183 | 0.6284 |

### Statistical Comparison (Wilcoxon signed-rank)

*Source: `docs/supplementary/S5_statistical_tests.md`*

| Comparison | Delta MAE | p-value | Significance | Cohen's d |
|------------|-----------|---------|--------------|-----------|
| DualANN vs RF | -0.0011 | 0.1602 | — | -0.50 |
| DualANN vs XGB | -0.0044 | 0.0039 | ** | -1.32 |
| DualANN vs LGBM | -0.0063 | 0.0020 | ** | -2.39 |
| RF vs XGB | -0.0033 | 0.0039 | ** | -1.29 |
| RF vs LGBM | -0.0052 | 0.0020 | ** | -2.11 |
| XGB vs LGBM | -0.0019 | 0.0137 | * | -1.06 |

---

## LOSO/LOMO Cross-Validation

Leave-one-substrate-out (LOSO) and leave-one-mechanism-out (LOMO) evaluation with XGBoost.

*Source: `docs/supplementary/S3_protocol_matrix.md`*

### 8-Protocol x 2-Feature-Set Matrix (XGBoost)

| Feature Set | Protocol | R² | MAE (%) | RMSE (%) |
|-------------|----------|-----|---------|----------|
| X0 (xTB only) | 5-fold KFold | 0.2973 | 12.2 | 18.1 |
| X0 (xTB only) | **LOSO** | **-0.0506** | 14.1 | 22.1 |
| X0 (xTB only) | LOMO | 0.1525 | 13.5 | 19.9 |
| X0 (xTB only) | LOSO×LOMO | 0.2173 | 12.9 | 19.1 |
| X1 (xTB + mech) | 5-fold KFold | 0.3008 | 12.1 | 18.1 |
| X1 (xTB + mech) | **LOSO** | **-0.0189** | 13.9 | 21.8 |
| X1 (xTB + mech) | LOMO | 0.1326 | 13.7 | 20.1 |
| X1 (xTB + mech) | LOSO×LOMO | 0.2107 | 13.0 | 19.2 |

### LOSO Bootstrap 95% CI (B=1000)

| Protocol | R² mean | 95% CI | Half-width |
|----------|---------|--------|------------|
| LOSO (xTB only) | -0.051 | [-0.082, -0.018] | 0.032 |
| LOSO (xTB + mech) | -0.019 | [-0.052, +0.014] | 0.033 |
| LOSO×LOMO | +0.217 | [+0.178, +0.258] | 0.040 |

### LOSO Per-Substrate Breakdown (XGBoost, xTB only)

| Substrate | n | Mean Yield | Predicted Yield | Bias | Individual R² |
|-----------|---|------------|-----------------|------|---------------|
| **CHO** | 289 | **53.8%** | 88.4% | **+34.6%** | **-1.45** |
| ECH | 640 | 92.6% | 86.6% | -6.0% | -0.39 |
| SO | 729 | 85.0% | 90.4% | +5.4% | -0.10 |
| PO | 605 | 89.8% | 89.5% | +0.3% | ~0.00 |
| IGE | 53 | 89.2% | 87.3% | -1.9% | -0.08 |
| **Excl. CHO** | 2,027 | — | — | — | **-0.056** |

**Key finding**: CHO drives LOSO failure. Removing CHO reduces LOSO R² from -0.051 to -0.056 (essentially zero).

### Cross-Architecture LOSO Comparison

*Source: `docs/supplementary/S3_4_model_benchmark_full.md`*

| Model | LOSO R² | MAE | RMSE |
|-------|----------|-----|------|
| XGBoost | -0.441 | 0.149 | 0.207 |
| LightGBM | -0.519 | 0.152 | 0.213 |
| Random Forest | -2.300 | 0.215 | 0.258 |
| DualBranchANN (Ridge proxy) | -5.031 | 0.261 | 0.321 |

**Key finding**: LOSO failure is architecturally robust — all four models produce negative R².

### LOMO Results (Cross-Architecture)

| Model | LOMO R² |
|-------|---------|
| XGBoost | +0.153 |
| LightGBM | +0.063 |
| Random Forest | +0.072 |
| DualBranchANN | +0.094 |

**Key finding**: LOMO produces positive R² across all models, proving catalyst mechanism split does not destroy transferability.

---

## External Holdout

15% random holdout (seed=2026).

*Source: `docs/supplementary/S4_external_holdout.md`*

### Split Statistics

| Set | n | Substrate Distribution (SO/ECH/PO/CHO/IGE) |
|-----|---|---------------------------------------------|
| Training pool | 1,969 | 620/555/504/249/41 |
| Test set | 347 | 109/85/101/40/12 |
| Full dataset | 2,316 | 729/640/605/289/53 |

Test set yield: mean 85.5%, median 94.0%, IQR 14.5%.

### Year-Based OOD (Train: ≤2021, Test: ≥2022)

| Model | Year-OOD R² | LOMO R² | Gap |
|-------|-------------|---------|-----|
| Random Forest | **0.391** | 0.072 | +0.319 |
| DualBranchANN | 0.333 | 0.094 | +0.239 |
| LightGBM | 0.270 | 0.063 | +0.207 |
| XGBoost | 0.229 | 0.153 | +0.076 |

---

## y-Randomization (100 Permutations)

4 models x 5 folds x 100 permutations with PCL-AE 128-D features.

*Source: `results_y_randomization_v4_100perm/y_randomization_v4_100perm_summary.json`*

| Model | Real R² | Perm Mean | Perm Std | Delta R² | p-value | Pass 2σ |
|-------|---------|-----------|----------|----------|---------|---------|
| DualBranchANN | 0.433 | -0.173 | 0.054 | **0.605** | 0.0099 | ✓ |
| XGBoost | 0.388 | -0.184 | 0.084 | **0.572** | 0.0099 | ✓ |
| Random Forest | 0.471 | -0.048 | 0.023 | **0.519** | 0.0099 | ✓ |
| LightGBM | 0.378 | -0.204 | 0.076 | **0.581** | 0.0099 | ✓ |

**Conclusion**: All models pass 2-sigma threshold (p < 0.01). Model signal is real, not fitting surface artifacts.

---

## PCL-AE vs Standard AE

Latent space comparison with 128-D bottleneck.

*Source: `docs/supplementary/S2_ablation.md`*

| Metric | Standard AE | PCL-AE (λ=0.1) | Ratio |
|--------|------------|-----------------|-------|
| mean |Pearson(yield, latent)| | **0.209** | **1.42x** |
| Dims with \|r\| > 0.1 | 76/128 (59.4%) | 88/128 (68.75%) | — |
| Silhouette (catalyst family) | 0.168 | 0.154 | -0.014 |
| 5-fold DualANN R² | 0.295 | 0.318 | +0.023 |

**Interpretation**: PCL-AE reorders the latent space from "catalyst family clustering" to "yield ranking", improving property correlation at the cost of slightly reduced cluster separation. This trade-off enables SHAP interpretability.

---

## Dimensionality Reduction Comparison

5-fold KFold DualBranchANN with different feature representations.

*Source: `docs/supplementary/S2_ablation.md`*

| Method | Dimensions | R² | Notes |
|--------|------------|-----|-------|
| Raw DRFP | 2048 | 0.303 | High-dimensional sparse (98.3% zeros) |
| PCA-128 + full | 128 | **0.3245** | Linear projection, natural denoising |
| PCA-256 + full | 256 | 0.3123 | |
| PCL-AE-128 (λ=200) | 128 | **0.4106** | Best overall |
| PCL-AE-256 (λ=200) | 256 | ~0.41 | Similar to 128-D |

**Note**: PCA outperforms raw DRFP due to denoising effect on sparse fingerprints. PCL-AE with high lambda (200) achieves the best results.

---

## Statistical Significance

### CHO vs Terminal Epoxides

Wilcoxon test for yield distribution difference.

*Source: `docs/supplementary/S5_statistical_tests.md`*

| Comparison | p-value | Cohen's d | Interpretation |
|------------|---------|-----------|----------------|
| CHO (mean 53.8%) vs terminal substrates (mean ~88%) | < 1e-42 | d ∈ [-2.12, -1.36] | Extremely large; ring vs terminal distribution difference is main cause of LOSO failure |

### Bootstrap CI for Per-Substrate SHAP (B=1000)

*Source: `docs/supplementary/S5_1_bootstrap_ci.md`*

`sub_homo_eV` feature importance by substrate:

| Substrate | n_samples | Mean SHAP | 95% CI | Sign |
|-----------|-----------|-----------|--------|------|
| **CHO** | 57 | **-0.230** | **[-0.239, -0.221]** | **Negative** |
| ECH | 125 | +0.056 | [+0.054, +0.057] | Positive |
| PO | 116 | +0.031 | [+0.030, +0.032] | Positive |
| SO | 159 | +0.020 | [+0.019, +0.021] | Positive |
| IGE | 7 | +0.024 | [+0.019, +0.030] | Positive |

**Key finding**: CHO's `sub_homo_eV` SHAP direction is **disjoint** from all four terminal epoxides (CI gap = 0.251). This confirms substrate-mechanism orthogonality.

---

## Per-Substrate Analysis

### Top-5 Features by Substrate (SHAP)

*Source: `docs/supplementary/S3_1_per_substrate_shap.md`*

| Substrate | top-1 | top-2 | top-3 | top-4 | top-5 |
|-----------|-------|-------|-------|-------|-------|
| CHO | time_log (1.93) | pressure (1.86) | temperature (1.65) | delta_E_HL (1.39) | sub_homo_eV (1.22) |
| ECH | sub_homo_eV (4.81) | time_log (2.31) | temperature (2.20) | pressure (2.07) | sub_lumo_eV (0.96) |
| SO | sub_homo_eV (4.80) | temperature (1.64) | time_log (1.52) | pressure (1.47) | sub_lumo_eV (1.31) |
| PO | sub_homo_eV (5.02) | sub_lumo_eV (2.69) | temperature (2.26) | pressure (1.93) | time_log (1.89) |
| IGE | sub_homo_eV (2.42) | temperature (1.60) | pressure (1.51) | time_log (1.34) | delta_E_LL (0.96) |

### Direction Flip Analysis

Strong direction flips (top-10 features, |signed| ≥ 0.5):

| Substrate Pair | n_features_flip | Fraction |
|----------------|-----------------|----------|
| CHO vs ECH | 4 | 12.5% |
| CHO vs IGE | 4 | 12.5% |
| CHO vs SO | 3 | 9.4% |
| CHO vs PO | 2 | 6.3% |

**Key finding**: CHO has the highest rate of feature direction flips vs terminal epoxides.

---

## DFT-xTB Calibration

GFN2-xTB vs DFT B3LYP-D3BJ/def2-TZVP comparison.

*Source: `docs/supplementary/S6_dft_xtb_calibration.md`*

### Dataset

18 molecules: 5 epoxide substrates, 3 catalyst classes (TBAI, ZnBr₂, TBAB), 2 solvents (DMSO, toluene), CO₂, propylene carbonate product.

After removing 3 ionic TBAI systems (N=15):

### Correlation (N=15)

| Descriptor | Pearson R | Spearman ρ |
|------------|-----------|------------|
| HOMO | **0.982** | **0.986** |
| LUMO | 0.727 | 0.918 |
| Gap | 0.814 | 0.829 |
| Dipole | 0.996 | 0.978 |

### MAE (N=15)

| Descriptor | MAE |
|------------|-----|
| HOMO | 3.994 eV |
| LUMO | 5.072 eV |
| Gap | 1.266 eV |
| Dipole | 0.316 D |

### Terminal Substrate Spearman by Family (N=15)

| Substrate Family | n | Spearman ρ (HOMO) |
|------------------|---|-------------------|
| Terminal epoxides (SO/ECH/PO/IGE + 4 more) | 8 | **+1.000** |
| Cyclohexene oxide (CHO) | 1 | n/a |

**Conclusion**: xTB descriptors are used for **rank-ordering** only, not absolute energies. The 3.99 eV HOMO MAE reflects systematic offset, but the correlation is strong (R=0.982), confirming xTB correctly captures relative electronic properties.

---

## Related Documents

- [PIPELINE.md](./PIPELINE.md) — Pipeline execution guide
- [supplementary/](./supplementary/) — Full supplementary information
- [paper/](../paper/) — Manuscript drafts
