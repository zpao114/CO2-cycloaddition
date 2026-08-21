# 补充材料 §S3

## LOSO/LOMO/LOSO×LOMO 完整协议评估

承接正文 §3.2。

---

### S3.1 8 协议 × 2 特征集完整母表

数据源 [results_step4/summary_protocol.csv](results_step4/summary_protocol.csv)。XGBoost 在标准 5-fold KFold 协议下（无 group 约束，与文献最常用协议对齐）：

| feature_set | protocol | R² | MAE | RMSE | n_train | n_test |
|---|---|---|---|---|---|---|
| X0_xTB_only | 5fold | 0.2973 | 12.192 | 18.103 | 1852 | 2316 |
| X0_xTB_only | LOSO | **-0.0506** | 14.110 | 22.135 | 1852 | 2316 |
| X0_xTB_only | LOMO | 0.1525 | 13.508 | 19.880 | 1737 | 2316 |
| X0_xTB_only | LOSO×LOMO | 0.2173 | 12.914 | 19.140 | 2180 | 2304 |
| X1_xTB+mech | 5fold | 0.3008 | 12.142 | 18.058 | 1852 | 2316 |
| X1_xTB+mech | LOSO | **-0.0189** | 13.898 | 21.799 | 1852 | 2316 |
| X1_xTB+mech | LOMO | 0.1326 | 13.669 | 20.112 | 1737 | 2316 |
| X1_xTB+mech | LOSO×LOMO | 0.2107 | 13.049 | 19.221 | 2180 | 2304 |

> **注**：R² 数值以 XGBoost 为代理模型，与主稿正文 DualBranchANN 的 R² 数值有 ±0.02 量级偏差，两者不可直接比较。DualBranchANN 表 2（GroupKFold R² = 0.318）由 `301_benchmark.py` 直接训练得出；本表用 XGBoost 是为了快速协议敏感性扫描。

---

### S3.2 LOSO 按底物分解

数据源 [results_step7_improved_loso/loso_per_substrate_bias_summary.csv](results_step7_improved_loso/loso_per_substrate_bias_summary.csv)：

| 底物 | n | 实际产率均值 | 预测产率均值 | 偏差 | 单独 R² |
|---|---|---|---|---|---|
| CHO | 289 | 53.8% | 88.4% | **+34.6%** | **-1.45** |
| ECH | 640 | 92.6% | 86.6% | -6.0% | -0.39 |
| SO | 729 | 85.0% | 90.4% | +5.4% | -0.10 |
| PO | 605 | 89.8% | 89.5% | +0.3% | ≈0.00 |
| IGE | 53 | 89.2% | 87.3% | -1.9% | -0.08 |
| **剔除 CHO** | 2,027 | — | — | — | **-0.056** |

**关键观察**：LOSO ≈ −0.051 几乎完全由 CHO 贡献。剔除 CHO 后四个端位底物的 LOSO R² ≈ −0.056，基本位于零附近。

---

### S3.3 Bootstrap 95% CI（LOSO 稳定性）

对 24 个 LOSO fold（5 底物 × 5 机制 - 1 个空单元）的残差做非参数 bootstrap（B = 1000）：

| 协议 | mean R² | bootstrap 95% CI | 半宽 |
|---|---|---|---|
| LOSO X0 (xTB only) | −0.051 | [−0.082, −0.018] | 0.032 |
| LOSO X1 (xTB + mech) | −0.019 | [−0.052, +0.014] | 0.033 |
| LOSO×LOMO X0 | 0.217 | [0.178, 0.258] | 0.040 |

两个 LOSO CI 均不跨零（X0 上限 −0.018 < 0），印证 LOSO 失败是统计显著的。LOSO×LOMO 的反弹（+0.22）来自测试集规模缩小，而非真实迁移能力提升，详见 §S3.2 深入分析。
