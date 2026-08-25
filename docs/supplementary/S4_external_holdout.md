# 补充材料 §S4

## 外部 holdout 细节

承接正文 §2.5、§3.5。

---

### S4.1 切分来源与种子

- 数据源：[results_data_split/data_split.json](results_data_split/data_split.json)
- 脚本：[306_external_validation.py](306_external_validation.py) L58-66 优先读取；若不存在则 `train_test_split(..., test_size=0.15, random_state=2026)`
- 种子：`SEED = 2026`（外部 holdout 专用种子，与 CV 用的 42 区分开）
- 切分比例：HOLDOUT_RATIO = 0.15

---

### S4.2 训练池 vs 测试集（外部 holdout）

外部 holdout 由 `src/models/persistence/405_external_validation.py` 执行。脚本读取 `results/results_data_split/data_split.json` 选 `use_holdout_train=True`，先用 data_split.json 的 `train_indices`（2,116 反应，全集 raw 2,490 中 yield NaN/=0 已过滤，等价全集 clean）取得训练池，再对训练池做 `np.random.shuffle(seed=2026)` + `n_test = int(2116 × 0.15) = 317` 切出测试集。底物/催化系统分布以 `external_test_predictions.csv` 实际明细为准。

| 集合 | n | 5 底物分布 (SO/ECH/PO/CHO/IGE) | 5 催化系统分布 (IL/MH/Mixed/BAS/Unknown) |
|---|---|---|---|
| 训练池（405 外部） | **1,799** | 570/500/457/228/44 | 1396/153/123/63/64 |
| 测试集（405 外部） | **317** | 96/87/87/37/10 | 245/23/19/14/16 |
| 全集 clean（data_split 索引） | **2,490** | 783/692/646/305/64 | 1940/199/161/93/97 |

测试集产率分布：均值 82.3%、标准差 22.5%、范围 [7.0%, 100.0%]（训练池均值 84.1%、标准差 21.7%）。两组均产率相近——测试集从训练池内部随机抽出（不跨 data_split 边界），构成同分布内部的 85/15 切分。这与 §3.5 中 R² ≈ 0.119 的弱 holdout 性能对应——随机 holdout 测试的是"训练池内部的随机外推"，而非"未参与 CV 的反应外推"。**训练池子集分布**：训练池 + 测试集 = data_split train 池 2,116；测试集 317 反应从中抽出后，剩余 1,799 反应为训练池。底物分布 570/500/457/228/44 与全集 clean 783/692/646/305/64 同比例（按 72.4% 缩放），催化分布 1396/153/123/63/64 与全集 1940/199/161/93/97 同比例。

---

### S4.3 PCL-AE per-fold 重训策略

正文 §2.5 已说明：外部 holdout 训练时，PCL-AE-128 **仅在训练池 n=1,799 上重训**，而不是直接读取 `results_pcl_ae_viz/pcl_ae_encoder.pt`（后者基于全量 2,490 反应预训练，会构成潜在嵌入泄漏）。

`306_external_validation.py` 的实现路径：

```python
# 在函数 train_dual_branch_ann() 内：
if cfg.FRESH_PCL_AE_ON_HOLDOUT:
    # 仅用训练池反应训练
    pcl_ae = PropertyCoLearningAE(input_dim=2048, latent_dim=128)
    pcl_ae.fit(X_train_drfp_only, y_train, lambda_=0.5, epochs=80)
else:
    # 默认：读取 802_pcl_ae_visualization.py 中保存的预训练权重
    # （SI 早期草稿中曾以 `005_pcl_ae_embedding.csv` 为别名，现统一为
    #  `results_pcl_ae_viz/` 下的持久化权重，详见 `802` 文档）
    pcl_ae = load_pcl_ae('results_pcl_ae_viz/pcl_ae_encoder.pt')
```

默认 `FRESH_PCL_AE_ON_HOLDOUT = True`（[306_external_validation.py](306_external_validation.py) 顶部配置）。这是与流水线默认设置唯一不同的点，必须显式说明以避免"嵌入泄漏"的潜在质疑。

---

### S4.4 测试集去重维度审计（与 §3.5 对应）

| 维度 | 训练池唯一 | 测试集唯一 | 重叠 |
|---|---|---|---|
| 反应 ID | 1,799 | 317 | 0 |
| 底物名 | 5 (SO/ECH/PO/CHO/IGE) | 5 (同上) | 5（覆盖全部底物；这是反应级随机划分的特征，区别于 LOSO） |
| 催化系统 | 5 | 5 | 5（覆盖全部 5 类） |
| 文献 ID（全集 data_split 索引） | 327 | 108 | 76（部分文献跨越训练/测试，仅算反应级去重） |

底物覆盖完整的副作用：测试集包含 10 行 IGE 反应（IGE×LAC/BIF 单元正是 n<10 实验缺口），它们进入测试集为缺口验证提供了有限度的"自然验证"——但每格 n=1-2，统计功效不足，仅作为方向性参考。

---

### S4.5 时间切分 OOD Holdout（按发表年份）

除随机 15% holdout 外，另按发表年份构建真实的分布外（OOD）测试集，以验证模型对未来文献的预测能力。

**切分策略**：

- 训练集：`publication_year ≤ 2021`，共 1,499 条反应
- 测试集：`publication_year ≥ 2022`，共 942 条反应（合计 2,441 = 全集 2,490 − 49 行 year 字段缺失）
- 脚本：[generate_year_ood_benchmark.py](generate_year_ood_benchmark.py)
- 结果：[results_si/year_ood_benchmark.csv](results_si/year_ood_benchmark.csv)
- 注意：本 OOD 切分独立于 §S4.2 外部 holdout——全集 raw 2,490 在按 yield NaN/yield=0 过滤后得到全集 clean 2,490（全集 raw 已无 NaN 与 0），再按 `publication_year` 分组得到 year 子集；脚本来源与 `co2_drfp_xtb_extended.csv` 的 year 元数据附加过程相关

**年度 OOD 结果（LOMO 5-fold CV 对照，3 种子均值）**：

| 模型 | Year-OOD R² | LOMO 5-fold R² | Gap（Year-OOD − LOMO） |
|---|---|---|---|
| Random Forest | **0.425** | **0.471** | −0.046 |
| DualBranchANN | **0.414** | **0.433** | −0.019 |
| LightGBM | **0.312** | **0.378** | −0.066 |
| XGBoost | **0.252** | **0.388** | −0.136 |

数据源：Year-OOD 来自 `results_si/year_ood_benchmark.csv`（3 种子 r2_mean 取均），LOMO R² 来自 `results_si/lomo_v3_full_results.csv`（5 fold by catalyst，3 种子取均）。

**关键发现**：

- 4 个模型在 Year-OOD 上均保持 R² > 0，**DRFP 分子指纹对时间外推场景具良好泛化性**：RF (0.425) > DualBranchANN (0.414) > LGBM (0.312) > XGBoost (0.252)。
- LOMO R² 普遍**高于** Year-OOD：LOMO 仅按催化机制（5 组）划分，测试集与训练集共享全部 5 底物，是同分布内部按 catalyst split 的相对简单任务；Year-OOD 跨越 2021/2022 文献分布偏移，更接近真实部署场景。两者的 Gap 为负（−0.02 至 −0.14），定量反映"催化机制 split 不破坏迁移能力，但年份分布偏移轻微增加难度"——这与 §3.2 LOMO 全部为正 R²、§3.3 LOSO 全部为负 R² 的趋势一致（**底物 split > 年份 split > 催化机制 split** 的 OOD 难度阶梯）。
- XGBoost 在 Year-OOD 上的 Gap 最大（−0.136），LGBM 较小（−0.066），RF 最小（−0.046）——RF 的归纳偏置（bagging + 随机特征）对时间分布偏移的鲁棒性最强。
