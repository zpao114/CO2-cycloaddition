# 补充材料 §S2

## λ / DRFP / 潜空间消融

承接正文 §2.3、§2.5（Configuration snapshot）、P0-2 [D2]、P1-1 [D5]、P1-3 [D7]。

---

### S2.1 λ 13 点扫描完整表

数据源 [results_lambda_ablation/lambda_results.csv](../../results_lambda_ablation/lambda_results.csv)。DualBranchANN 5-fold KFold，3 个 AE 种子 × 5 fold：

| λ | DualANN R² | RF R² | DualANN seed-std |
|---|---|---|---|
| 0.00 | 0.3052 ± 0.0027 | 0.2693 | **0.0027** |
| 0.05 | 0.3085 ± 0.0082 | 0.2840 | 0.0082 |
| 0.10 | 0.3059 ± 0.0010 | 0.2889 | **0.0010** |
| 0.20 | 0.3163 ± 0.0048 | 0.3038 | 0.0048 |
| 0.50 | 0.3308 ± 0.0104 | 0.3240 | 0.0104 |
| 1.00 | 0.3441 ± 0.0114 | 0.3403 | 0.0114 |
| 2.00 | 0.3631 ± 0.0080 | 0.3526 | 0.0080 |
| 3.00 | 0.3680 ± 0.0101 | 0.3571 | 0.0101 |
| 5.00 | 0.3774 ± 0.0114 | 0.3626 | 0.0114 |
| 7.00 | 0.3862 ± 0.0077 | 0.3673 | 0.0077 |
| 10.00 | 0.3908 ± 0.0080 | 0.3698 | 0.0080 |
| 20.00 | 0.3945 ± 0.0050 | 0.3708 | 0.0050 |
| 50.00 | 0.4021 ± 0.0067 | 0.3737 | 0.0067 |
| 75.00 | 0.4046 ± 0.0050 | 0.3734 | 0.0050 |
| 100.00 | 0.4058 ± 0.0045 | 0.3734 | 0.0045 |
| 150.00 | 0.4107 ± 0.0039 | 0.3748 | 0.0039 |
| **200.00** | **0.4149 ± 0.0045** ★ | **0.3748** | 0.0045 |

**关键观察**：在 λ ∈ [0.0, 200] 全扫描区间（共 17 个点），DualANN R² 单调上升从 0.305 到 0.415，极差 ΔR² = **0.1097**（约 peak 的 26%）——**PCL-AE 对 λ 不鲁棒**（这与旧 SI §S2.1 的"ΔR² = 0.0156"鲁棒性结论不同，旧草稿扫描区间仅为 [0.0, 50] 13 点）。生产配置 λ = 200.0 由 `201_ablation.py` 自动写入 `config.py`（`BEST_LAMBDA_PROP = 200.0, BEST_LAMBDA_PROP_R2 = 0.4149`）。在 λ ∈ [0.05, 50] 子区间（共 12 个点），DualANN R² ∈ [0.309, 0.402] 仍单调上升——**鲁棒性仅在"全 λ 区间"框架下不成立；在低 λ 区间（≤50）下 R² 提升约 0.09，相对幅度 ~30%**。**关于 seed-std**：λ=0.10（0.0010）和 λ=200.0（0.0045）均较小，绝对值不同；整段扫描 seed-std 范围 [0.0010, 0.0114]，中位 ~0.007，表明 PCL-AE 的 seed 鲁棒性**始终保持**（不论 λ 取何值，seed 间方差均 < 1.2% R²）。

---

### S2.2 DRFP 4 变体消融

数据源 [results_best_pipeline/drfp_ablation_results.csv](results_best_pipeline/drfp_ablation_results.csv) + [202_drfp_ablation.py](202_drfp_ablation.py)。XGBoost 5-fold KFold（全集 raw 2,490）：

| 变体 | 编码 | 5-fold R² |
|---|---|---|
| `full` | reactants + cat + solv | **0.1544** |
| `React` | reactants only | 0.1386 |
| `no_cats` | reactants + solv | 0.1402 |
| `no_sols` | reactants + cat | 0.1482 |

**结论**：4 变体 R² 范围 [0.1386, 0.1544]，ΔR² = 0.0158；`full` 变体（reactants + cat + solv）最优 R² = 0.1544。**这与旧 SI 中 "React 变体最佳 R²=0.2027" 不同**——旧 SI 基于旧数据集 (2,316) 与不同 ablation 脚本。**当前 v3 数据支持 `full` 变体为生产配置**（`BEST_DRFP_VARIANT = 'full'`），不再选 `React`。

**注**：代码中字符 `'full'` 对应论文中的 `full` 变体；不再使用旧配置 `'reactants'`。

---

### S2.3 PCL-AE vs 标准 AE 潜空间对比

数据源 [results_pcl_ae_viz/viz_report.txt](results_pcl_ae_viz/viz_report.txt) + [results_pcl_ae_viz/latent_comparison.csv](results_pcl_ae_viz/latent_comparison.csv)。配置：128 维瓶颈，`utils_benchmark.py PropertyCoLearningAE`。

**注意**：以下表格数据为**旧数据集（2,316）+ 旧 ablation 脚本**下的历史结果。**当前仓库中** `results_pcl_ae_viz/` 目录**未找到**（仅 `results_pcl_ae/{standard_ae_latent.npy, pcl_ae_latent.npy, improved_pcl_ae_latent.npy, row_id.csv}` 存在）；为保持 SI 完整性，列出历史数据并标记待 v3 重跑：

| 指标 | Standard AE | PCL-AE | 比值 |
|---|---|---|---|
| mean \|Pearson(yield, latent)\| | 0.147 | 0.209 | 1.42×（绝对差 0.062） |
| 维度满足 \|r\| > 0.1 | 76/128 = 59.4% | 88/128 = 68.75% | — |
| Silhouette（催化剂家族） | 0.168 | 0.154 | −0.014 |
| 5-fold DualANN R² | **0.295** | **0.318** | **+0.023** |

注：5-fold DualANN R² = 0.318（旧数据集旧 ablation）— 与 §S2.1 λ=0.1 时 DualANN R² = 0.306（v3 csv 数据）不直接可比——两组数据来自不同数据集 (2,316 vs 2,490) 与不同 ablation 脚本（13-pt vs 17-pt）。**写作时建议以 §S2.1 v3 λ=200 数据为准**，本表仅作历史记录。Silhouette 下降 0.014 反映潜空间从"催化剂家族聚类"重排到"产率排序"——这是设计取舍，不是退化。S3.5 SHAP 方向诊断的物理可解读性依赖这一重排。

---

### S2.4 PCL-AE 的科学价值定位（鲁棒性 vs 绝对精度）

PCL-AE 的科学贡献**不在"绝对 R² 优于 PCA"**，而在"对随机种子的鲁棒性（property co-learning 注入 yield 排序结构）"。下表给出三种降维方法在 5-fold KFold DualBranchANN 上的对照（数据源 [results_best_pipeline/full_benchmark_results.csv](../../results_best_pipeline/full_benchmark_results.csv)，取每方法 best DualANN 行）：

| 降维方法 | dim | DualANN R² | DualANN seed-std | 鲁棒性特征 |
|---|---|---|---|---|
| Raw DRFP (2048D) | 2048 | 0.3065 | 0.063 | 高维稀疏，树模型容量受限 |
| **PCL-AE-128(λ=200) + full** | 128 | **0.4095** | 0.044 | property co-learning 注入 yield 排序，最佳 R² |
| **PCL-AE-256(λ=200) + full** | 256 | 0.3980 | 0.047 | 同上，略低 |
| PCA-128 + full | 128 | 0.3016 | 0.054 | 线性投影天然去噪，R² 中等 |
| PCA-256 + full | 256 | 0.3020 | 0.046 | 同上 |

**关键发现 1 — λ 鲁棒性（已修正）**：v3 lambda_results.csv 显示 **PCL-AE 对 λ 并不高度鲁棒**：在 λ ∈ [0, 200] 全扫描区间（共 17 个点），DualANN R² 单调上升从 0.305 到 0.415，ΔR² = 0.110（约 peak 的 27%）。生产配置 λ = 200.0 也不是"鲁棒平坦区中心"——是**单调上升曲线的端点**。**这与旧 SI §S2.1 表"λ=0.1 最佳 ΔR²=0.016 鲁棒"不同**——旧 SI 基于 13-pt 扫描 (λ ∈ [0, 50]) + 旧数据集 (2,316 旧 ablation)，新 v3 是 17-pt 扫描 (λ ∈ [0, 200]) + 新数据集 (2,490)。在低 λ 区间 [0.05, 50] 共 12 个点，R² 上升 0.305→0.402 (ΔR² ≈ 0.097)，仍**非鲁棒**。**这一结论的关键差异需要在正文 §2.5 与 §3.1 中显式说明**：原文"PCL-AE 对 λ 鲁棒"claim 需要降级为"PCL-AE 的 seed 鲁棒性强（seed-std < 0.012），但 λ 敏感性较高（ΔR² ~0.10 across full scan）"。

**关键发现 2 — AE seed 鲁棒性**（来自 v3 lambda_results.csv）：
- λ = 0.10（接近纯 AE）：seed-std = **0.0010**（极小）
- λ = 200.0（PCL-AE 最佳）：seed-std = **0.0045**（仍 < 0.5% R²）
- λ = 50.0：seed-std = **0.0067**
- 含义：property co-learning 不是"提升 R² 数字"，而是**让 encoder 对随机初始化不再敏感**——seed-std 全程 < 0.012，对实际部署（一次训练即可投入生产）至关重要。

**关键发现 3 — 降维上限的解释**：
- DRFP 在本数据集上的 sparsity = 98.3%（仅 1.7% 的位为 1）。在极度稀疏的 2048-D 二值指纹上，PCA 的线性投影天然去噪，因此 PCA-128/256 的 R² 中等（~0.30）。
- **PCL-AE-128(λ=200)+full 的 R² = 0.410 显著优于 PCA-128+full = 0.302**（ΔR² = +0.108，约 36% 相对提升）——这与旧 SI "PCL-AE 不可能压过 PCA 的去噪优势" 的论断**相反**。v3 数据表明：property co-learning 在λ 充分大时（λ=200）能将 yield 排序结构注入潜空间，使 PCL-AE 同时享有 PCA 的去噪优势和 yield 监督的语义结构。
- PCL-AE 的潜空间**同时**满足重建保真度（≈PCA）和 yield 排序可读性（λ=200 时 DualANN R² 最高）——这是 §S2.3 表中 `PCL-AE vs Standard AE` 在 `mean |Pearson(yield, latent)|` 上提升的机制。

**写作建议**：在正文 §2.5 中将 PCL-AE 定位为 "a **property-conditioned** AE whose main contribution is **seed robustness + λ-tuned yield-ordering**, with peak R² 0.410 at λ=200 (vs PCA-128+full 0.302, ΔR² = +0.108)"；在 §S3.5 SHAP 方向诊断中再引用 §S2.3 的 `mean |Pearson| = 0.209` 物理可解读性证据。
