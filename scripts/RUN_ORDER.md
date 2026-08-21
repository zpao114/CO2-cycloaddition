# CO2 cycloaddition — single-script run order
# Date: 2026-08-18
# Convention:
#   [OK]    I run it directly (single python, <30 min)
#   [BIG]   I run it but it takes hours (acknowledge first)
#   [WSL]   You run it in Ubuntu / WSL (manual)
#   [SKIP]  Optional / no-op / data already exists

# ============================================================
# PHASE A — DATA FOUNDATION (raw → processed)
# Strict order: A1 → A2 → A3 → A4 (xTB) → A5 → A6 → A0
# A0 must run AFTER A6 because data_split.py reads the
# master table co2_drfp_xtb_extended.csv (A6's output).
# ============================================================

# A1. Cleaning + dedup
#   [OK]
python src/data/101_clean.py
#   in:    data/raw/CO2_cycloaddition_merged.csv  (5,263 rows)
#   out:   data/processed/cleaned.csv  (2,490 valid)
#   out:   data/external/discard_report.csv  (631 dropped)
#   side:   data/external/cleaned_baseline.json

# A2. SMILES canonicalisation
#   [OK]
python src/data/102_smiles.py
#   in:    data/processed/cleaned.csv
#   out:   data/processed/co2_smiles.csv  (2,490 rows)
#   side:   data/external/smiles_baseline.json

# A3. DRFP encoding (2048-bit reaction fingerprint, 4 variants)
#   [OK]
python src/data/103_drfp.py
#   in:    data/processed/co2_smiles.csv
#   out:   data/processed/co2_drfp.csv  (2,490 × 2048 × 4 variants)
#   side:   preserves legacy upstream DRFP for rows that already existed

# A4. xTB descriptors — BIG (≈6 h, GFN2-xTB per molecule)
#   [BIG]  --recommend run this overnight in batch
python src/data/104b_run_xtb_extended.py --timeout 90
#   in:    data/processed/co2_smiles.csv  (2,490 rows)
#   out:   results/results_cho_diagnostic/xtb_results_summary.csv
#   out:   results/results_cho_diagnostic/<smiles>.xyz  (one per molecule)
#   out:   results/results_cho_diagnostic/<smiles>.xtb.stdout
#   notes:  --timeout 90 = xTB walltime cap per molecule = 90 s
#           skip-if-done: if summary CSV exists, exits early unless --force
#           (104b does NOT write data/processed/co2_drfp_xtb.csv — that is
#            produced by an older 106_merge_xtb.py and is left in place as
#            the input to A6=107)

# A5. xTB sanity check (convergence rate, descriptor sanity)
#   [OK]
python src/data/105b_xtb_sanity_v2.py
#   in:    results/results_cho_diagnostic/xtb_results_summary.csv
#   out:   results/results_cho_diagnostic/xtb_sanity_report.txt
#   notes:  reads A4's summary; no xTB re-runs

# A6. Merge substrate xTB + add interaction features  ← KEY STEP
#   [OK]
python src/data/107_merge_substrate_xtb.py
#   in:    data/processed/co2_drfp_xtb.csv  (legacy 61-col; from old 106)
#           + results/results_cho_diagnostic/xtb_results_summary.csv  (A4)
#           + data/processed/cleaned.csv  (A1)
#   out:   results/results_cho_diagnostic/co2_drfp_xtb_extended.csv  (87-col MASTER)
#   side:   data/external/substrate_xtb_baseline.json
#   notes:  --out defaults to the CHO-diagnostic directory; to also mirror the
#           table under data/processed/ (for downstream paths.py), do:
#              cp results/results_cho_diagnostic/co2_drfp_xtb_extended.csv \
#                 data/processed/co2_drfp_xtb_extended.csv

# A0. Canonical 5-fold split + stratified 15% holdout
#   [OK]   ← MUST run AFTER A6 (A6's output is its input)
python src/data_split.py
#   in:    data/processed/co2_drfp_xtb_extended.csv  (87-col, n=2,316)
#   out:   results/results_data_split/data_split.json  (canonical manifest)
#   out:   results/results_data_split/data_split_summary.txt
#   notes:  seed=2026, test_size=0.15, yield-quartile strata
#           n_train=1,968, n_test=348
#   WARN:   if A6's output is in results/results_cho_diagnostic/ but
#           data/processed/co2_drfp_xtb_extended.csv is stale (older n_total),
#           you must mirror the new file first (see A6 cp note above) — otherwise
#           data_split will use stale row indices and break later steps.

# A7. (106b is a documented no-op stub — SKIP)

# ============================================================
# Phase A dependency summary
# ============================================================
# A1 → A2 → A3 → A4 → A5 → A6 → A0
#                              │
#                              └─► all later phases read this manifest
#                                  (305/405 generate their own splits internally;
#                                   901/902 read it for reproducibility)

# ============================================================
# PHASE B — ABLATION (DRFP variants + λ scan + benchmark)
# ============================================================

# 201_ablation.py — three stages in one run (~163 min)
#   [BIG]  ← Already done 2026-08-17~18; re-runnable for verification
# Reads:  results_best_pipeline/drfp_ablation_meta.json  (Stage 1 output)
# Writes: results_best_pipeline/drfp_ablation_meta.json  (Stage 1)
#         results_lambda_ablation/lambda_results.csv    (Stage 2)
#         config.py: BEST_LAMBDA_PROP auto-updated       (Stage 2)
#         results_best_pipeline/full_benchmark_results.csv  (Stage 3)
#
#   STAGE 1+2 take ~3 min, STAGE 3 takes ~160 min (XGB/LGBM/RF/DualANN × 5-fold × 3 seeds)
python src/data/201_ablation.py
#   Stage 1 (DRFP variants)        → results_best_pipeline/drfp_ablation_meta.json
#   Stage 2 (λ scan, 13 points)    → results_lambda_ablation/lambda_results.csv
#                                     + config.py: BEST_LAMBDA_PROP auto-updated
#   Stage 3 (4-model benchmark)    → results_best_pipeline/full_benchmark_results.csv
#
#   STAGE 1+2 take ~3 min, STAGE 3 takes ~160 min (XGB/LGBM/RF/DualANN × 5-fold × 3 seeds)

# ============================================================
# PHASE C — BENCHMARK + PERSISTENCE + EXTERNAL
# ============================================================

# C1. 301: 5-fold KFold baseline (4 models × 3 seeds)
#   [OK]
python src/models/benchmarks/301_benchmark.py
#   in:    config.py (BEST_LAMBDA_PROP, BEST_LATENT_DIM, BEST_DRFP_VARIANT)
#   out:   results_best_pipeline/301_results.csv
#           + subgroup CSVs by catalyst class

# C2. 401: persist the best pipeline (scaler + AE + DualANN weights)
#   [OK]
python src/models/persistence/401_persist_best_pipeline.py
#   in:    config.py + results from C1
#   out:   results_best_pipeline/artifacts/  (4 .pkl files)

# C3. 306: external holdout (15% unseen)
#   [OK]
python src/models/benchmarks/306_external_validation.py
#   in:    split_seed_2026.json + artifacts from C2
#   out:   results_external_validation/306_results.csv
#           (overall R²=0.382, MH R²=0.669)

# ============================================================
# PHASE D — DEEP ABLATIONS
# ============================================================

# D1. 801: deep DRFP ablation (per-bit, info-gain)
#   [OK]
python src/models/benchmarks/801_drfp_ablation_deep_analysis.py
#   in:    co2_drfp.csv + cleaned.csv
#   out:   results_drfp_ablation_deep/

# D2. 803: Mordred ablation (2-D physicochemical descriptors)
#   [OK]
python src/models/benchmarks/803_mordred_ablation.py
#   in:    cleaned.csv
#   out:   results_mordred_ablation/

# D3. 804: hierarchical catalyst-family model (mixed-effects)
#   [OK]
python src/models/benchmarks/804_hierarchical_catalyst_model.py
#   in:    cleaned.csv
#   out:   results_hierarchical_model/

# ============================================================
# PHASE E — STATISTICAL VALIDATION + SCREENING
# ============================================================

# E1. 302: GroupKFold (catalyst-grouped CV)
#   [OK]
python src/models/benchmarks/302_groupkfold_validation.py
#   in:    cleaned.csv + best pipeline config
#   out:   results_groupkfold_validation/

# E2. 303: sample-size sensitivity (subsample 25/50/75/100%)
#   [OK]
python src/models/benchmarks/303_sample_size_sensitivity.py
#   in:    cleaned.csv
#   out:   results_sample_size_sensitivity/

# E3. 304: statistical significance (5×2 paired t-test, Cohen's d)
#   [OK]
python src/models/benchmarks/304_statistical_significance.py
#   in:    results from C1
#   out:   results_statistical_test/

# E4. 305: y-randomization (30 permutations)
#   [OK]
python src/models/benchmarks/305_y_randomization.py
#   in:    cleaned.csv + best pipeline config
#   out:   results_y_randomization/

# E5. 310: known-top-10 baseline (chemistry-informed prior)
#   [OK]
python src/models/benchmarks/310_known_top10_baseline.py
#   in:    cleaned.csv
#   out:   results_known_top10/

# E6. 402: virtual screening (~3,000 candidates)
#   [OK]
python src/models/screening/402_virtual_screening.py
#   in:    artifacts from C2
#   out:   results_virtual_screening/candidates_full.csv

# E7. 403b: ranking metrics (NDCG@10 / MAP@10)
#   [OK]
python src/models/screening/403b_ranking_metrics.py
#   in:    results_virtual_screening/
#   out:   results_ranking_metrics/

# E8. 405: external validation (second check, 308-row holdout)
#   [OK]
python src/models/persistence/405_external_validation.py
#   in:    split_seed_2026.json + artifacts
#   out:   results_external_validation/405_results.csv

# ============================================================
# PHASE F — MECHANISM + LOSO + SHAP
# ============================================================

# F1. 601: 5-class catalyst mechanism clustering
#   [OK]
python src/analysis/mechanism/601_catalyst_mechanism_v2.py
#   in:    cleaned.csv
#   out:   results/catalyst_mechanism.csv
#           + catalyst_mechanism_summary.json

# F2. 602: substrate steric + electronic features
#   [OK]
python src/analysis/mechanism/602_substrate_features.py
#   in:    cleaned.csv
#   out:   results/substrate_features*.csv (3 files)

# F3. 603: 5×5 transferability matrix
#   [OK]
python src/analysis/mechanism/603_transferability_matrix.py
#   in:    results/catalyst_mechanism.csv + cleaned.csv
#   out:   results/transferability_matrix.csv

# F4. 700: LOSO / LOMO / LOSO×LOMO CV (4-protocol × 2-feature-set)
#   [OK]
python src/analysis/loso/700_loso_lomo_cv.py
#   in:    co2_drfp_xtb_extended.csv + catalyst_mechanism.csv
#   out:   results_step4/summary_protocol.csv  (8 combos)

# F5. 701: per-substrate / per-(substrate, mechanism) SHAP
#   [OK]
python src/analysis/loso/701_per_substrate_shap.py
#   in:    co2_drfp_xtb_extended.csv + catalyst_mechanism.csv
#   out:   results_step4_5/per_substrate_shap.csv
#           + bootstrap CI CSV
#           + shap_direction_flip.png

# F6. 702: integrated narrative (paper-ready synthesis)
#   [OK]
python src/analysis/loso/702_integrated_report.py
#   in:    F1-F5 outputs
#   out:   results_step5/integrated_narrative.md  (PAPER-READY)

# ============================================================
# PHASE G — DIAGNOSTIC + PAPER ARTEFACTS
# ============================================================

# G1. 900: bilingual abstract (EN+ZH)
#   [OK]
python src/analysis/diagnostics/900_paper_abstract.py
#   in:    results_step5/integrated_summary.json
#   out:   docs/paper/abstracts/{en,zh}.md

# G2. 901: substrate × catalyst matrix visualisation
#   [OK]
python src/analysis/diagnostics/901_substrate_catalyst_matrix.py
#   in:    results_step4_5/
#   out:   results_substrate_catalyst_matrix/

# G3. 902: CHO mechanistic diagnostic (Welch + Cohen's d)
#   [OK]
python src/analysis/diagnostics/902_cho_mechanistic_diagnostic.py
#   in:    results_step4_5/ + cleaned.csv
#   out:   results_cho_diagnostic/

# G4. regen_all_v3: one-pass regen (figs 1-8)
#   [OK]
python src/visualization/regen_all_v3.py
#   in:    all results above
#   out:   figures/paper/*.pdf  (14 figures)

# ============================================================
# PHASE H — VISUALISATION (paper figures)
# ============================================================

# Each fig script independent — run one at a time for inspection.
# H1. PCL-AE architecture diagram
#   [OK]
python src/visualization/fig_pcl_ae_architecture.py
#   out:   figures/fig_pcl_ae_architecture.pdf

# H2. y-randomization 100-perm histogram
#   [OK]
python src/visualization/fig_yrandomization_100perm.py
#   out:   figures/fig_yrandomization_100perm.pdf

# H3. Graphical abstract (ACS single-takeaway v4)
#   [OK]
python src/visualization/fig_0_graphical_abstract.py
#   out:   figures/fig0_graphical_abstract.pdf

# H4. TOC graphic v4
#   [OK]
python src/visualization/fig_toc.py
#   out:   figures/fig_toc.pdf

# H5. LOSO failure 4-panel (root cause)
#   [OK]
python src/visualization/fig_4_loso_root_cause.py
#   out:   figures/fig4_loso_root_cause.pdf

# H6. LOSO/LOMO 4-protocol bar
#   [OK]
python src/visualization/fig_5_loso_protocol.py
#   out:   figures/fig5_loso_protocol.pdf

# H7. 5×5 transferability matrix (coverage + yield)
#   [OK]
python src/visualization/fig_6_transferability_matrix.py
#   out:   figures/fig6_transferability_matrix.pdf

# H8. SHAP direction flip
#   [OK]
python src/visualization/fig_7_shap_direction.py
#   out:   figures/fig7_shap_direction.pdf

# H9. Bilingual main figures — run EN+ZH side by side
for fig in fig1_protocol_comparison fig2_loso_quality \
            fig3_coverage_5x5 fig4_shap_per_substrate \
            fig5_homo_vs_yield; do
  python src/visualization/${fig}_en.py
  python src/visualization/${fig}_zh.py
done

# ============================================================
# PHASE SI — SI-ONLY ENHANCEMENTS
# ============================================================

# SI1. SI S3.4 benchmark (full version)
#   [OK]
python src/ci_artifacts/generate_si_s3_benchmark_full_v3_1.py
#   out:   results_si/groupkfold_subset_v3.csv

# SI2. Year-OOD benchmark
#   [OK]
python src/ci_artifacts/generate_year_ood_benchmark.py
#   out:   results_si/year_ood.csv

# SI3. 100-permutation y-randomisation v4 (BIG ~40 min)
#   [BIG]
python src/ci_artifacts/generate_y_randomization_v4_100perm.py
#   out:   results_y_randomization_v4_100perm/

# SI4. Bootstrap substrate CI
#   [OK]
python src/ci_artifacts/generate_bootstrap_substrate_ci.py
#   out:   results_si/bootstrap_substrate_ci.csv

# ============================================================
# PHASE DFT — DFT/ORCA CALIBRATION
# ============================================================

# DFT1. Generate ORCA input deck
#   [OK]
python src/dft/501_generate_dft_inputs.py --level medium
#   in:    cleaned.csv
#   out:   dft_validation/orca_inputs/*.inp

# DFT2. Run ORCA — MANUAL STEP IN UBUNTU/WSL
#   [WSL]  ← YOU run this in Ubuntu
#   cmd:   cd /mnt/d/machine-learning/CO2\ cycloaddition/dft_validation/orca_inputs
#          for f in *.inp; do orca $f > ${f%.inp}.out; done
#   out:   dft_validation/orca_outputs/*.out  (one per molecule)

# DFT4. Parse ORCA outputs
#   [OK]
python src/dft/510_parse_dft_outputs.py
#   in:    dft_validation/orca_outputs/*.out
#   out:   dft_validation/dft_results_summary.csv

# DFT5. xTB on DFT-optimised geometries
#   [BIG]  ~30 min (re-run xTB on the 18 DFT geometries)
python src/dft/512_xtb_on_dft_geometry.py --solvent gas --output xtb_on_dft_geometry_nosolv.csv
#   in:    dft_validation/dft_results_summary.csv
#   out:   dft_validation/xtb_on_dft_geometry_nosolv.csv

# DFT6. xTB vs DFT comparison report
#   [OK]
python src/dft/514_dft_vs_xtb_report.py \
  --xtb-summary dft_validation/xtb_on_dft_geometry_nosolv.csv \
  --dft-summary dft_validation/dft_results_summary.csv \
  --output  dft_validation/514_dft_vs_xtb_report.csv \
  --report  dft_validation/514_dft_vs_xtb_report.txt

# DFT7. DFT journal figures
#   [OK]
python src/dft/520_dft_journal_figures.py
#   out:   figures/s6_dft_vs_xtb_grid.pdf

# ============================================================
# PHASE S7 — Partial TS verification
# ============================================================

# S71. Generate TS optimisation inputs (OptTS + NumFreq)
#   [OK]
python src/dft/713_orca_dft_generator.py
#   in:    results from DFT4
#   out:   dft_validation/ts_inputs/*.inp  (PO/ECH/IGE; CHO/SO may fail)

# S72. Run TS optimisation in ORCA — MANUAL IN UBUNTU
#   [WSL]
#   cmd:   cd /mnt/d/machine-learning/CO2\ cycloaddition/dft_validation/ts_inputs
#          for f in *.inp; do orca $f > ${f%.inp}.out; done

# ============================================================
# PHASE X — PCL-AE standalone training (optional)
# ============================================================

# X1. One-shot PCL-AE trainer (if you want a separate AE model)
#   [OK]
python src/models/persistence/train_pcl_ae.py
#   out:   results_pcl_ae/  (latent + weights)

# X2. PCL-AE latent visualisation (clustering, Pearson)
#   [OK]
python src/visualization/pcl_ae_visualization.py
#   out:   results_pcl_ae_viz/  (figures + viz_report.txt)