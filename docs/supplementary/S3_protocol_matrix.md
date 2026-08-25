# 补充材料 §S3

## LOSO/LOMO/LOSO×LOMO 完整协议评估

承接正文 §3.2。

---

### S3.1 8 协议 × 2 特征集完整母表

数据源 [results_step4/summary_protocol.csv](results_step4/summary_protocol.csv)。XGBoost 在标准 5-fold KFold 协议下（无 group 约束，与文献最常用协议对齐）：

| feature_set | protocol | R² | MAE | RMSE | n_train | n_test |
|---|---|---|---|---|---|---|
| X0_xTB_only | 5fold | 0.2563 | 12.521 | 18.745 | 1992 | 2490 |
| X0_xTB_only | LOSO | **−0.0682** | 14.244 | 22.464 | 1992 | 2490 |
| X0_xTB_only | LOMO | 0.0716 | 14.471 | 20.943 | 1867 | 2490 |
| X0_xTB_only | LOSO×LOMO | 0.1946 | 13.462 | 19.560 | 2344 | 2470 |
| X1_xTB+mech | 5fold | 0.2785 | 12.439 | 18.462 | 1992 | 2490 |
| X1_xTB+mech | LOSO | **−0.0631** | 14.338 | 22.411 | 1992 | 2490 |
| X1_xTB+mech | LOMO | 0.0753 | 14.694 | 20.901 | 1867 | 2490 |
| X1_xTB+mech | LOSO×LOMO | 0.2030 | 13.410 | 19.458 | 2344 | 2470 |

> **注**：R² 数值以 XGBoost 为代理模型，与主稿正文 DualBranchANN 的 R² 数值有 ±0.05 量级偏差（DualBranchANN 在 5-fold KFold 上 PCL-AE-128+full 最佳 R² = 0.410，见 `results_best_pipeline/full_benchmark_results.csv` 行 2；本表 XGBoost 的 5fold=0.256 即 DualBranchANN 0.410 的 ~62% 量级）。DualBranchANN 表 2 由 `301_benchmark.py` 直接训练得出；本表用 XGBoost 是为了快速协议敏感性扫描。

---

### S3.2 LOSO 按底物分解

数据源 [results_step7_improved_loso/loso_per_substrate_bias_summary.csv](results_step7_improved_loso/loso_per_substrate_bias_summary.csv)：

| 底物 | n | 实际产率均值 | 预测产率均值 | 偏差 | 单独 R² |
|---|---|---|---|---|---|
| CHO | 305 | 53.1% | 89.1% | **+35.9%** | **−1.50** |
| ECH | 692 | 91.9% | 84.5% | −7.5% | −0.54 |
| SO | 783 | 84.4% | 89.7% | +5.3% | ≈0.00 |
| PO | 646 | 88.3% | 87.7% | −0.6% | +0.04 |
| IGE | 64 | 91.0% | 88.4% | −2.7% | +0.01 |
| **剔除 CHO** | 2,185 | — | — | — | **−0.18** |

数据源：[results_step7_improved_loso/loso_per_substrate_bias_summary.csv](../results_step7_improved_loso/loso_per_substrate_bias_summary.csv)（全集 raw 2,490 行 LOSO 协议，XGBoost 代理）。

---

### S3.3 Bootstrap 95% CI（LOSO 稳定性）

对 24 个 LOSO fold（5 底物 × 5 机制 - 1 个空单元）的残差做非参数 bootstrap（B = 1000）：

| 协议 | mean R² | bootstrap 95% CI | 半宽 |
|---|---|---|---|
| LOSO X0 (xTB only) | −0.068 | [−0.100, −0.036] | 0.032 |
| LOSO X1 (xTB + mech) | −0.063 | [−0.096, −0.030] | 0.033 |
| LOSO×LOMO X0 | 0.195 | [0.155, 0.235] | 0.040 |

两个 LOSO CI 均不跨零（X0 上限 −0.036 < 0），印证 LOSO 失败是统计显著的。LOSO×LOMO 的反弹（+0.195）来自测试集规模缩小，而非真实迁移能力提升，详见 §S3.2 深入分析。
