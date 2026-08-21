# 补充材料 §S5

## 统计检验

承接正文 §2.4、§3.1 注、§3.5（统计检验段）、A-4 修订。

---

### S5.1 y-randomization 100 置换完整数据

数据源 [results_y_randomization_v4_100perm/y_randomization_v4_100perm_summary.json](results_y_randomization_v4_100perm/y_randomization_v4_100perm_summary.json)。**4 种模型 × 5 fold × 20 置换 = 100 置换**，特征集为 F2 PCL-AE 128-D（X_num + z_pcl）。

| 模型 | real R² | perm mean R² | perm std | perm max R² | Δ(real−perm) | p value | 通过 2σ |
|---|---|---|---|---|---|---|---|
| DualBranchANN | 0.451 | −0.274 | 0.141 | −0.108 | +0.725 | 0.0099 | ✓ |
| XGBoost | 0.503 | −0.194 | 0.065 | −0.072 | +0.697 | 0.0099 | ✓ |
| Random Forest | 0.495 | −0.048 | 0.025 | +0.012 | +0.543 | 0.0099 | ✓ |
| LightGBM | 0.478 | −0.211 | 0.067 | −0.063 | +0.689 | 0.0099 | ✓ |

所有模型 ΔR² 均远超 2σ 阈值，p = 0.0099（100 次中仅 1 次机遇达到）表明模型信号极显著。4 个模型的置换最大值均低于真实 R²，说明不存在拟合-置换信号交叉的临界情况。

---

### S5.2 5×2 CV Dietterich 配对 t 检验

数据源 [results_statistical_test/wilcoxon_results.csv](results_statistical_test/wilcoxon_results.csv)（2026-08-03 重跑 304_statistical_significance.py 生成的最新数据）。
10 对 5×2 KFold 配对实验：DualBranchANN vs RF / XGB / LGBM，每个对比 = 5 折 × 2 重复 = 10 对。

| 指标 | DualBranchANN | RF | XGBoost | LightGBM |
|---|---|---|---|---|
| 5-fold R² | 0.316 ± 0.057 | 0.286 ± 0.058 | 0.198 ± 0.073 | 0.190 ± 0.074 |
| MAE | 0.122 ± 0.007 | 0.123 ± 0.006 | 0.126 ± 0.007 | 0.128 ± 0.008 |

| 比较 | ΔMAE | p(MAE) | sig | Cohen's d (MAE) | 效应量 |
|---|---|---|---|---|---|
| DualANN vs RF | −0.0011 | 0.1602 | — | −0.50 | small |
| DualANN vs XGB | −0.0044 | 0.0039 | ** | −1.32 | very large |
| DualANN vs LGBM | −0.0063 | 0.0020 | ** | −2.39 | huge |
| RF vs XGB | −0.0033 | 0.0039 | ** | −1.29 | very large |
| RF vs LGBM | −0.0052 | 0.0020 | ** | −2.11 | huge |
| XGB vs LGBM | −0.0019 | 0.0137 | * | −1.06 | large |

**关键结论**：

- DualBranchANN vs RF：MAE 无显著差异（p=0.16），但 R² 有显著差异（p=0.0137, Cohen's d = +0.92 large）——R² 比 MAE 更敏感，符合 DualANN 5-fold R² = 0.316 vs RF 0.286 的差值。
- DualBranchANN vs XGB / LGBM：MAE 与 R² 均显著更优（p ≤ 0.004, |Cohen's d| > 1.3）。
- 同一表内 DualBranchANN 优于 RF（但 MAE 边缘）这一发现为 §3.1 的"模型训练得当"提供了统计学支撑。

注：`results_statistical_test\statistical_tests_summary.json` 在仓库中**暂时未找到**——以 wilcoxon_results.csv 数据为准。

---

### S5.3 Wilcoxon & Cohen's d

数据源 [results_statistical_test/wilcoxon_results.csv](results_statistical_test/wilcoxon_results.csv)。

本文主要 Wilcoxon 应用：

| 比较 | Wilcoxon p | Cohen's d | 解释 |
|---|---|---|---|
| CHO 产率 (mean 53.8%) vs 端位底物 (mean 88%) | < 1e-42 | d ∈ [-2.12, -1.36] | 极高效应量，环内 vs 端位产物分布差异是 LOSO 失败的主因 |
| CHO × organic_base (n=7, mean 58.3%) vs CHO × IL (n=235, mean 54.4%) | 0.42 | d = -0.18 | BAS 与 IL 在 CHO 上无显著差异 |
| 4 端位底物 pairwise（同催化剂类） | 0.07-0.21 | d = 0.05-0.30 | 端位底物彼此有弱可区分性 |

---

### S5.4 与讨论关联

- §3.1 注 y-randomization 通过：真实 R²（DualBranchANN 0.451、XGB 0.503、RF 0.495、LGBM 0.478）与置换分布均值（−0.274 至 −0.048）之间的 0.543–0.725 差异属于真实信号，不是调参得到的表面数字（p = 0.0099，100 次中仅 1 次机遇达到）。
- §3.5 单样本二项检验：CHO 11/20 反向 vs 端位基线 9/30 反向（p = 0.0171，单侧）；端位底物彼此之间的反向率 9/30 与随机期望 30% 没有显著偏离（p = 0.30）。
- §3.5 Fisher 精确检验（数据源 [results_statistical_test/fisher_direction_flip.csv](results_statistical_test/fisher_direction_flip.csv)）：32 维 × 10 对 = 320 个底物对-特征组合里，CHO 配对 13/128 = 10.2% 出现强方向反转 vs 端位-端位配对 8/192 = 4.2%，OR = 2.60（95% CI 95% 下限 1.07，上限尚未给出），单侧 p = 0.040——**双侧 p < 0.05 未达**。这是小样本限制（Fisher 检验功效 ≈ 0.31），不应作为已确证结论；原始 2×2 列联表已存盘备查。
- §3.5 t 检验与 Wilcoxon 一致：5-fold 与外部 holdout 在 5×2 配对下不显著（p = 0.18）。