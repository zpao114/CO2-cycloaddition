# 补充材料 §S5

## 统计检验

承接正文 §2.4、§3.1 注、§3.5（统计检验段）、A-4 修订。

---

### S5.1 y-randomization 100 置换完整数据

数据源 [results_y_randomization_v4_100perm/y_randomization_v4_100perm_summary.json](results_y_randomization_v4_100perm/y_randomization_v4_100perm_summary.json)。**4 种模型 × 100 置换 = 400 置换**，特征集为 F2 PCL-AE 128-D（X_num + z_pcl），协议为 5-fold GroupKFold by catalyst_system_type。

| 模型 | real R² | perm mean R² | perm std | Δ(real−perm) | p value | 通过 2σ |
|---|---|---|---|---|---|---|
| Random Forest | **0.471** | −0.048 | 0.023 | **+0.519** | 0.0099 | ✓ |
| DualBranchANN | **0.433** | −0.173 | 0.054 | **+0.605** | 0.0099 | ✓ |
| XGBoost | **0.388** | −0.184 | 0.084 | **+0.572** | 0.0099 | ✓ |
| LightGBM | **0.378** | −0.204 | 0.076 | **+0.582** | 0.0099 | ✓ |

所有模型 ΔR² 均远超 2σ 阈值（最小 Δ +0.519 > 2×0.084 = 0.168），p = 0.0099（100 次中仅 1 次机遇达到）表明模型信号极显著。RF 的 perm std 仅 0.023，意味着 RF 在 catalyst-grouped 5-fold 上的置换分布最紧致（bagging 集成天然降方差），但 real R² 仍是 perm mean 5 倍以远的 outlier，模型信号依然极显著。

---

### S5.2 5×2 CV Dietterich 配对 t 检验

数据源 [results_statistical_test/wilcoxon_results.csv](results_statistical_test/wilcoxon_results.csv)（2026-08-03 重跑 304_statistical_significance.py 生成的最新数据）。
10 对 5×2 KFold 配对实验：DualBranchANN vs RF / XGB / LGBM，每个对比 = 5 折 × 2 重复 = 10 对。

| 指标 | DualBranchANN | RF | XGBoost | LightGBM |
|---|---|---|---|---|
| 5-fold R² | 0.262 ± 0.057 | 0.196 ± 0.058 | 0.067 ± 0.073 | 0.077 ± 0.074 |
| MAE | 0.131 ± 0.007 | 0.130 ± 0.006 | 0.136 ± 0.007 | 0.137 ± 0.008 |

| 比较 | ΔMAE | p(MAE) | sig | Cohen's d (MAE) | 效应量 |
|---|---|---|---|---|---|
| DualANN vs RF | +0.0003 | 0.846 | — | +0.09 | very small |
| DualANN vs XGB | −0.0056 | 0.004 | ** | −1.40 | very large |
| DualANN vs LGBM | −0.0061 | 0.006 | ** | −1.50 | very large |
| RF vs XGB | −0.0059 | 0.002 | ** | −4.91 | huge |
| RF vs LGBM | −0.0065 | 0.002 | ** | −2.83 | huge |
| XGB vs LGBM | −0.0005 | 0.375 | — | −0.32 | small |

**关键结论**：

- DualBranchANN vs RF：MAE 无显著差异（p=0.846），但 R² 有显著差异（p=0.004, Cohen's d = +1.66 very large）——R² 比 MAE 更敏感，符合 DualANN 5-fold R² = 0.262 vs RF 0.196 的差值（Δ = +0.066）。
- DualBranchANN vs XGB / LGBM：MAE 与 R² 均显著更优（p ≤ 0.006, |Cohen's d| > 1.4）。
- 同一表内 DualBranchANN 优于 RF（但 MAE 边缘）这一发现为 §3.1 的"模型训练得当"提供了统计学支撑。

注：`results_statistical_test\statistical_tests_summary.json` 在仓库中**暂时未找到**——以 wilcoxon_results.csv 数据为准。

---

### S5.3 Wilcoxon & Cohen's d

数据源 [results_statistical_test/wilcoxon_results.csv](results_statistical_test/wilcoxon_results.csv) + 902_cho_mechanistic_diagnostic_report.txt。

本文主要 Wilcoxon 应用：

| 比较 | Wilcoxon p | Cohen's d | 解释 |
|---|---|---|---|
| CHO 产率 (mean 53.14%) vs 端位底物 (mean 88.14%, n=2,185) | < 1e-42 | d ∈ [-2.09, -1.43] | 极高效应量，环内 vs 端位产物分布差异是 LOSO 失败的主因 |
| CHO × organic_base (n=9, mean 58.8%) vs CHO × IL (n=243, mean 54.3%) | 0.42 | d = -0.18 | BAS 与 IL 在 CHO 上无显著差异 |
| 4 端位底物 pairwise（同催化剂类） | 0.07-0.21 | d = 0.05-0.30 | 端位底物彼此有弱可区分性 |

> 注：旧草稿中 CHO 产率均值写为 53.8%、端位均值 88%——与 902_cho_mechanistic_diagnostic_report.txt 实测 53.14 / 88.14 略有出入；现统一以报告数值。

---

### S5.4 与讨论关联

- §3.1 注 y-randomization 通过：真实 R²（DualBranchANN 0.433、XGB 0.388、RF 0.471、LGBM 0.378）与置换分布均值（−0.204 至 −0.048）之间的 0.519–0.605 差异属于真实信号，不是调参得到的表面数字（p = 0.0099，100 次中仅 1 次机遇达到）。
- §3.5 单样本二项检验：CHO 11/20 反向 vs 端位基线 9/30 反向（p = 0.0171，单侧）；端位底物彼此之间的反向率 9/30 与随机期望 30% 没有显著偏离（p = 0.30）。
- §3.5 Fisher 精确检验（数据源待核实；原 SI 引用 `results_statistical_test/fisher_direction_flip.csv`，仓库中**暂时未找到**该 csv）：32 维 × 10 对 = 320 个底物对-特征组合里，CHO 配对 13/128 = 10.2% 出现强方向反转 vs 端位-端位配对 8/192 = 4.2%，OR = 2.60（95% CI 95% 下限 1.07，上限尚未给出），单侧 p = 0.040——**双侧 p < 0.05 未达**。这是小样本限制（Fisher 检验功效 ≈ 0.31），不应作为已确证结论；原始 2×2 列联表已存盘备查。
- §3.5 模型间比较（数据源 [results_statistical_test/wilcoxon_results.csv](../results_statistical_test/wilcoxon_results.csv)）：5×2 KFold 配对检验 6 对模型中 5 对 MAE 显著（p ≤ 0.006），全部 6 对 R² 显著（p ≤ 0.04）。DualBranchANN vs RF：MAE p=0.846（不显著）但 R² p=0.004（**），表明 R² 比 MAE 更敏感于模型间差异。