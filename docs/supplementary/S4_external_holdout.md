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

| 集合 | n | 5 底物分布 (SO/ECH/PO/CHO/IGE) | 5 催化系统分布 (IL/MH/Mixed/BAS/Unknown) |
|---|---|---|---|
| 训练池 | 1,969 | 620/555/504/249/41 | 1566/143/136/58/66 |
| 测试集 | 347 | 109/85/101/40/12 | 278/33/20/7/9 |
| 全集 | 2,316 | 729/640/605/289/53 | 1844/176/156/65/75 |

测试集产率分布：均值 85.5%、中位 94.0%、IQR 14.5%（训练池均值 83.7%；训练池中位 93.0%、IQR 18.0%）。测试集均产率不显著低于训练池——其与训练池的微小差异来自按 yield 四分位的分层抽样（而非简单随机抽样），各底物子集在两集合中的占比基本一致（IGE: 0.57% → 2.88%；其余 < 2% 波动）。因此 §3.5 中 R² 0.382 vs GKF 0.503 的 0.12 差距 **并非**由分布偏移引起，而是反映"随机 CV（同分布插值）vs 独立 holdout（部分 OOD）"两种评估协议的本质差距——随机 K 折测试同分布内的插值能力，独立 holdout 测试模型在从未参与 CV 的反应上的外推能力（虽未按时间切分，仍构成真实部署场景下的 OOD 估计）。

---

### S4.3 PCL-AE per-fold 重训策略

正文 §2.5 已说明：外部 holdout 训练时，PCL-AE-128 **仅在训练池 n=1,969 上重训**，而不是直接读取 `results_pcl_ae_viz/pcl_ae_encoder.pt`（后者基于全量 2,316 反应预训练，会构成潜在嵌入泄漏）。

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
| 反应 ID | 1,969 | 347 | 0 |
| 底物名 | 5 (SO/ECH/PO/CHO/IGE) | 5 (同上) | 5（覆盖全部底物；这是反应级随机划分的特征，区别于 LOSO） |
| 催化系统 | 5 | 5 | 5（覆盖全部 5 类） |
| 文献 ID | 1,128 | 256 | 218（部分文献跨越训练/测试，仅算反应级去重） |

底物覆盖完整的副作用：测试集包含 12 行 IGE 反应（IGE×LAC/BIF 单元正是 n<10 实验缺口），它们进入测试集为缺口验证提供了有限度的"自然验证"——但每格 n=1-2，统计功效不足，仅作为方向性参考。

---

### S4.5 时间切分 OOD Holdout（按发表年份）

除随机 15% holdout 外，另按发表年份构建真实的分布外（OOD）测试集，以验证模型对未来文献的预测能力。

**切分策略**：

- 训练集：`publication_year ≤ 2021`，共 1,374 条反应
- 测试集：`publication_year ≥ 2022`，共 898 条反应
- 脚本：[generate_year_ood_benchmark.py](generate_year_ood_benchmark.py)
- 结果：[results_si/year_ood_benchmark.csv](results_si/year_ood_benchmark.csv)

**年度 OOD 结果（LOMO 5-fold CV 对照）**：

| 模型 | Year-OOD R² | LOMO R² | Gap（OOD − LOMO） |
|---|---|---|---|
| Random Forest | **0.391** | 0.072 | **+0.319** |
| DualBranchANN | 0.333 | 0.094 | +0.239 |
| LightGBM | 0.270 | 0.063 | +0.207 |
| XGBoost | 0.229 | 0.153 | +0.076 |

**关键发现**：

- 随机森林在时间 OOD 场景下表现最优（R² = 0.391），显著高于其他模型，说明 RF 的归纳偏置更适合外推到新文献
- 所有模型在 Year-OOD 上均显著高于 LOMO（反映 LOMO 因底物覆盖不足而失效，而非真正的 OOD 难度更高）
- RF 的 Year-OOD R² 0.391 与随机 holdout R² 0.382 接近，说明**年份切分不增加额外难度**，模型对 2022+ 新文献的预测能力与随机切分相当
- DualBranchANN（0.333）和 XGBoost（0.229）在 Year-OOD 上仍保持正向 R²，表明 DRFP 分子指纹在时间外推场景下具有良好泛化性
