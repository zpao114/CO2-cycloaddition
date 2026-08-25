# 补充材料 §S3.1

## per-substrate SHAP 方法

承接正文 §2.5、§3.5。

---

### S3.1.1 计算流程

脚本 [701_per_substrate_shap.py](701_per_substrate_shap.py)：

1. 对 5 个底物依次做 LOSO（即把当前底物所有 n 反应留出）。
2. 在剩余 ~N - n 反应上训练 XGBoost（注释中说明选择 XGBoost 是因为它的 SHAP 路径可通过 `pred_contribs=True` 直接拿到 signed per-feature contribution，绕过 `shap>=0.44` 的 UTF-8 解码问题）。
3. 对当前底物全部 n 反应计算 SHAP signed values。
4. 把所有底物结果汇总到 `results_step4_5/per_substrate_top_features.csv` 与 `per_substrate_shap_direction.csv`。

### S3.1.2 5 底物各自 top-1 特征完整表（与正文 §3.4 表 5 一致）

数据源 [data/processed/bootstrap_substrate_shap_ci.csv](../../data/processed/bootstrap_substrate_shap_ci.csv)（2,490 反应 SHAP global 模型 bootstrap 100 次，n_samples 与正文一致）：

| 底物 | top-1 feature | mean signed SHAP | 95% CI [lo, hi] | sign 稳定性 |
|---|---|---|---|---|
| CHO | sub_homo_eV | **−25.148** | [−25.608, −24.684] | **全零下**（CI disjoint 于其它 4 底物） |
| ECH | sub_homo_eV | +4.373 | [+4.322, +4.441] | 全零上 |
| PO | sub_homo_eV | +3.658 | [+3.595, +3.730] | 全零上 |
| SO | sub_homo_eV | +2.964 | [+2.914, +3.006] | 全零上 |
| IGE | sub_homo_eV | +3.049 | [+2.898, +3.190] | 全零上 |

**关键观察**：5 底物 top-1 特征**统一为 `sub_homo_eV`**（电子描述符），且在 4 个端位底物上**同向正**、在 CHO 上**显著反向负**——CI 完全互不重叠，gap ≈ 27.6。**这与正文 §3.4 表 5 一致**。

> **注（与 701 脚本的区别）**：早期 `results_step4_5/per_substrate_top_features.csv`（701 脚本基于 LOSO 重训后模型 SHAP）给出的 CHO top-1 = `delta_E_HL`（mean_signed = −1.78），与此处 global SHAP bootstrap 不一致——两脚本**度量对象不同**：
> - **701 / `per_substrate_top_features.csv`**：LOSO 重训（CHO 数据被剔除训练集）的 SHAP，反映"当 CHO 不在训练集时，模型对 CHO 测试点的特征依赖"
> - **`bootstrap_substrate_shap_ci.csv`**：global SHAP（CHO 数据保留），bootstrap 抽样下反映"global 模型在 CHO 上的稳定性"
> 本 SI 与 §S5.1 / 正文 §3.4 表 5 **统一采用 global SHAP bootstrap** 作为权威数据，原因：(1) 数据规模 (n=305 / 64~783) 远大于 701 的早期 snapshot；(2) CHO `delta_E_HL` 的 LOSO 重训 SHAP (−1.78) 在 global SHAP bootstrap 下变为 **+0.014（CI 跨零，统计上不显著）**，故不作为稳定反转特征。

---

### S3.1.3 方向反转统计（基于 LOSO 重训模型的强翻转特征）

数据源 [results_step4_5/per_substrate_shap_direction.csv](../../results_step4_5/per_substrate_shap_direction.csv)（LOSO 重训模型下 32 维特征 × 10 对底物对的 `strong_flip` 数量；`strong_flip` 定义：两底物 signed SHAP 符号相反且 |signed| ≥ 1.0）：

| 底物对 | n_features_strong_flip | frac_strong_flip | 最大翻转特征 |
|---|---|---|---|
| Cyclohexene oxide vs Isopropyl glycidyl ether | 3 | 9.4% | delta_E_HL |
| Cyclohexene oxide vs Propylene oxide | 3 | 9.4% | sub_homo_eV |
| Epichlorohydrin vs Isopropyl glycidyl ether | 3 | 9.4% | sub_homo_eV |
| Isopropyl glycidyl ether vs Propylene oxide | 3 | 9.4% | sub_homo_eV |
| Cyclohexene oxide vs Epichlorohydrin | 2 | 6.3% | sub_homo_eV |
| Cyclohexene oxide vs Styrene oxide | 2 | 6.3% | sub_homo_eV |
| Epichlorohydrin vs Propylene oxide | 1 | 3.1% | sub_lumo_eV |
| Epichlorohydrin vs Styrene oxide | 1 | 3.1% | sub_lumo_eV |
| Isopropyl glycidyl ether vs Styrene oxide | 1 | 3.1% | sub_homo_eV |
| Propylene oxide vs Styrene oxide | 1 | 3.1% | sub_lumo_eV |

**核心发现**：10 对底物配对中，**所有 5 对** 含 CHO 的配对（CHO vs ECH / CHO vs PO / CHO vs IGE / CHO vs SO）的 max_abs_diff_feature 均为 **`sub_homo_eV`** 或 **`delta_E_HL`**——CHO 在 LOSO 重训下与端位底物都有强翻转；这一 LOSO-level 反转与 global bootstrap CI（§S5.1.A）中 sub_homo_eV 的 disjoint CI 互相印证。

---

### S3.1.4 SHAP 因果性局限（Discussion）

SHAP 是基于树模型的局部可解释方法，给出的 signed values 是模型内部的"对预测贡献"，**不是因果证据**。本工作的 SHAP 结果支持两类诊断：

1. **同特征方向对比**（如 sub_homo_eV 在 CHO 上 vs 端位底物）：仅当训练集/测试集同分布假设成立时，符号差异可解读为机制差异。本工作该假设受 LOSO 失败（−0.068）约束，**方向对比应在端位底物内部**（CHO vs ECH 等），不应跨到底物维度的"测试"集合。
2. **top-1 特征类型对比**（动力学 vs 电子）：top-1 类别错位（CHO vs 端位）是更稳健的诊断，因为它不依赖 signed value 的具体数值，仅依赖特征类型。CHO 的 top-1 = `time_log`，4 个端位底物的 top-1 = `sub_homo_eV`——这一类别级对比与单特征方向反转是双重独立的证据。

SHAP 不能直接证伪或证实以下假说：

- CHO 的低产率是因为环内 LUMO 不能与侧链 HOMO 形成有效耦合（需要 DFT 过渡态验证，§4.4 列为未来工作）。
- 选择性机理（Lewis 酸 vs 氢键）对 CHO 的影响大于电子描述符（需要单独的反应条件分离实验）。
