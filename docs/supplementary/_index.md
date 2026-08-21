# 补充材料索引（SI Index）

承接正文 `paper_draft_JC_Chinese.md`。本 SI 给出主文表格/数字背后的来源审计、统计检验、消融研究与边界讨论，按编号列出。

---

## §S1–§S7 完整索引

| 编号 | 主题 | 关联正文 | 主文件 | 关键数据源 |
|---|---|---|---|---|
| **S1** | 数据集 & 漏斗审计 | §2.1 | `SI/S1_data_audit.md` | `101_clean.py` / `discard_report.csv` / `results/data_audit/reference_traceability.csv` / `results/data_audit/coverage_summary.csv` |
| **S2** | λ / DRFP / 潜空间消融 | §2.3 (PCL-AE) / §2.4 (Config) | `SI/S2_ablation.md` | [201_ablation.py](201_ablation.py) / [results_lambda_ablation/](results_lambda_ablation/) / [results_best_pipeline/drfp_ablation_meta.json](results_best_pipeline/drfp_ablation_meta.json) |
| **S3.1** | per-substrate SHAP 方法 | §2.6 方法 / §3.4 | `SI/S3_1_per_substrate_shap.md` | [701_per_substrate_shap.py](701_per_substrate_shap.py) / [results_step4_5/per_substrate_top_features.csv](results_step4_5/per_substrate_top_features.csv) |
| **S3.2** | LOSO×LOMO 异常解释 | §3.2 | `SI/S3_2_loso_lomo_anomaly.md` | [results_step4/loso_kl_divergence.csv](results_step4/loso_kl_divergence.csv) / [results_step4/loso_variance_ratio.csv](results_step4/loso_variance_ratio.csv) |
| **S3.3** | 8 协议 × 2 特征集完整母表 | §3.2 | `SI/S3_protocol_matrix.md` | [results_step4/summary_protocol.csv](results_step4/summary_protocol.csv) / [results_step7_improved_loso/loso_per_substrate_bias_summary.csv](results_step7_improved_loso/loso_per_substrate_bias_summary.csv) |
| **S3.4** | LOSO/LOMO/GroupKFold × 4模型完整结果 | §3.2 | `SI/S3_4_model_benchmark_full.md` | [results_si/loso_full_results.csv](results_si/loso_full_results.csv) / [results_si/lomo_v3_full_results.csv](results_si/lomo_v3_full_results.csv) / [results_si/groupkfold_v3_full_results.csv](results_si/groupkfold_v3_full_results.csv) / [results_si/groupkfold_subset_v3.csv](results_si/groupkfold_subset_v3.csv) / [results_si/feature_set_benchmark.csv](results_si/feature_set_benchmark.csv) / [generate_si_s3_benchmark_full_v3_1.py](generate_si_s3_benchmark_full_v3_1.py) |
| **S4** | 外部 holdout 细节 + 时间 OOD | §2.5 / §3.5 | `SI/S4_external_holdout.md` | [306_external_validation.py](306_external_validation.py) / [generate_year_ood_benchmark.py](generate_year_ood_benchmark.py) / [results_external_validation/](results_external_validation/) / [results_data_split/data_split.json](results_data_split/data_split.json) / [results_si/year_ood_benchmark.csv](results_si/year_ood_benchmark.csv) |
| **S5** | 统计检验 | §2.4 / §3.1 / §3.5 | `SI/S5_statistical_tests.md` | [304_statistical_significance.py](304_statistical_significance.py) / [generate_y_randomization_v4_100perm.py](generate_y_randomization_v4_100perm.py) / [results_statistical_test/wilcoxon_results.csv](results_statistical_test/wilcoxon_results.csv) / [results_y_randomization_v4_100perm/y_randomization_v4_100perm_summary.json](results_y_randomization_v4_100perm/y_randomization_v4_100perm_summary.json) |
| **S5.1** | 100次Bootstrap CI（per-substrate SHAP）+ 100-perm y-randomization | §3.4 / §3.1 | `SI/S5_statistical_tests.md` / `SI/S5_1_bootstrap_ci.md` | [generate_bootstrap_substrate_ci.py](generate_bootstrap_substrate_ci.py) / [generate_y_randomization_v4_100perm.py](generate_y_randomization_v4_100perm.py) / [bootstrap_substrate_shap_ci.csv](bootstrap_substrate_shap_ci.csv) / [results_y_randomization_v4_100perm/y_randomization_v4_100perm_summary.json](results_y_randomization_v4_100perm/y_randomization_v4_100perm_summary.json) |
| **S6** | DFT-xTB 校准 | §2.3 / §4.3 | `SI/S6_dft_xtb_calibration.md` | [dft_validation/514_dft_vs_xtb_report.csv](dft_validation/514_dft_vs_xtb_report.csv) / [dft_validation/README_DFT.txt](dft_validation/README_DFT.txt) / [SI/S6_dft_vs_xtb.png](S6_dft_vs_xtb.png) / [SI/S6_dft_vs_xtb.pdf](S6_dft_vs_xtb.pdf) |
| **S7** | 五底物 TS（PO/ECH/IGE 已收敛；CHO/SO 未收敛） | §3.4 / §4.4 | `SI/S7_ige_transition_state.md` | [dft_validation/ts_5_substrates/PO/ts_PO.{out,property.txt}](dft_validation/ts_5_substrates/PO/) / [dft_validation/ts_5_substrates/ECH/ts_ECH.{out,property.txt}](dft_validation/ts_5_substrates/ECH/) / [dft_validation/ts_5_substrates/IGE/ts_IGE_v2.{out,property.txt,hess}](dft_validation/ts_5_substrates/IGE/) / [dft_validation/ts_5_substrates/CHO/ts_CHO.{out,xyz}](dft_validation/ts_5_substrates/CHO/) / [dft_validation/ts_5_substrates/SO/ts_SO.{out,xyz}](dft_validation/ts_5_substrates/SO/) |

---

## SI 文件清单（按子目录布局）

```
SI/
├── _index.md                       # 本文件
├── S1_data_audit.md                # §S1 数据集审计
├── S2_ablation.md                  # §S2 λ/DRFP 消融
├── S3_1_per_substrate_shap.md      # §S3.1 SHAP 方法
├── S3_2_loso_lomo_anomaly.md      # §S3.2 LOSO×LOMO 异常
├── S3_protocol_matrix.md            # §S3.3 完整协议母表
├── S3_4_model_benchmark_full.md    # §S3.4 4模型完整benchmark
├── S4_external_holdout.md          # §S4 外部验证
├── S5_statistical_tests.md         # §S5 统计检验
├── S5_1_bootstrap_ci.md            # §S5.1 100次Bootstrap CI
├── S6_dft_vs_xtb.png            # §S6 xTB vs DFT 散点图（4面板：HOMO/LUMO/Gap/Dipole）
├── S6_dft_vs_xtb.pdf            # §S6 xTB vs DFT 散点图（矢量版）
├── S6_dft_xtb_calibration.md    # §S6 DFT 校准详细说明
├── quick_dft_vs_xtb_plot.py     # §S6 图生成脚本
└── S7_ige_transition_state.md    # §S7 五底物 TS（PO/ECH/IGE 收敛；CHO/SO 未收敛）
```

---

## 快速导航

### 数据与审计
- [S1_data_audit.md](S1_data_audit.md) - 4 阶段数据清洗漏斗、近重复审计、底物/催化剂分布

### 模型配置与消融
- [S2_ablation.md](S2_ablation.md) - PCL-AE λ 扫描、DRFP 变体对比、潜空间对比
- [S3_protocol_matrix.md](S3_protocol_matrix.md) - 8 协议 × 2 特征集完整结果

### SHAP 与机制诊断
- [S3_1_per_substrate_shap.md](S3_1_per_substrate_shap.md) - per-substrate SHAP 方法与方向反转统计
- [S3_2_loso_lomo_anomaly.md](S3_2_loso_lomo_anomaly.md) - LOSO×LOMO 非单调现象解释
- [S3_4_model_benchmark_full.md](S3_4_model_benchmark_full.md) - LOSO/LOMO/GroupKFold在4模型架构下的完整结果

### 验证与统计
- [S4_external_holdout.md](S4_external_holdout.md) - 外部 holdout 切分 + 时间 OOD holdout（按年份）
- [S5_statistical_tests.md](S5_statistical_tests.md) - y-randomization（100-perm）、5×2 CV、Cohen's d
- [S5_1_bootstrap_ci.md](S5_1_bootstrap_ci.md) - 100次bootstrap CI验证SHAP方向反转

### DFT 计算
- [S6_dft_xtb_calibration.md](S6_dft_xtb_calibration.md) - GFN2-xTB vs B3LYP-D3BJ/def2-TZVP 校准（含分层分析、KL散度bootstrap）
- [S6_dft_vs_xtb.png](S6_dft_vs_xtb.png) / [S6_dft_vs_xtb.pdf](S6_dft_vs_xtb.pdf) - 4面板散点图
- [S7_ige_transition_state.md](S7_ige_transition_state.md) - 五底物 TS 计算（PO/ECH/IGE 收敛；CHO/SO 待重启）

---

## 关联正文章节

| 正文章节 | 主要关联 SI |
|---|---|
| §2.1 数据集 | S1 |
| §2.3 描述符系统 | S2, S6 |
| §2.4 模型架构 | S2 |
| §2.5 评估协议 | S2 |
| §3.1 内插基线 | S5 |
| §3.2 LOSO/LOMO | S3.2, S3.3, S3.4 |
| §3.4 SHAP 诊断 | S3.1, S5.1, S7 |
| §3.5 外部验证 | S4 |
| §4.3 xTB 精度 | S6 |
| §4.5 局限 | S7 |
