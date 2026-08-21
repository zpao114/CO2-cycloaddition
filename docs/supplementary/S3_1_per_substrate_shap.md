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

### S3.1.2 5 底物各自 top-5 特征完整表（与正文 §3.5 表 5 一致）

| 底物 | top-1 | top-2 | top-3 | top-4 | top-5 |
|---|---|---|---|---|---|
| CHO | time_log (1.93, -0.12) | pressure (1.86, -0.15) | temperature (1.65, +0.04) | delta_E_HL (1.39, -1.14) | sub_homo_eV (1.22, -1.20) |
| ECH | sub_homo_eV (4.81, +4.81) | time_log (2.31, -0.68) | temperature (2.20, +0.17) | pressure (2.07, -0.04) | sub_lumo_eV (0.96, +0.96) |
| SO | sub_homo_eV (4.80, +4.80) | temperature (1.64, +0.36) | time_log (1.52, -0.26) | pressure (1.47, -0.04) | sub_lumo_eV (1.31, +1.31) |
| PO | sub_homo_eV (5.02, +5.02) | sub_lumo_eV (2.69, +2.69) | temperature (2.26, -0.62) | pressure (1.93, -0.18) | time_log (1.89, -0.43) |
| IGE | sub_homo_eV (2.42, +2.42) | temperature (1.60, +0.66) | pressure (1.51, +0.16) | time_log (1.34, +0.26) | delta_E_LL (0.96, +0.84) |

---

### S3.1.3 方向反转统计（`direction_flip`）

数据源 `results_step4_5/per_substrate_shap_direction.csv`：

| 底物对 | n_features_strong_flip | frac_strong_flip | 最大翻转特征 |
|---|---|---|---|
| CHO vs ECH | 4 | 12.5% | sub_homo_eV |
| CHO vs IGE | 4 | 12.5% | sub_homo_eV |
| CHO vs SO | 3 | 9.4% | sub_homo_eV |
| CHO vs PO | 2 | 6.3% | sub_homo_eV |

`strong_flip` 定义：top-10 特征集合中 signed SHAP 在两底物上符号相反且 |signed| ≥ 0.5 的特征。

---

### S3.1.4 SHAP 因果性局限（Discussion）

SHAP 是基于树模型的局部可解释方法，给出的 signed values 是模型内部的"对预测贡献"，**不是因果证据**。本工作的 SHAP 结果支持两类诊断：

1. **同特征方向对比**（如 sub_homo_eV 在 CHO 上 vs 端位底物）：仅当训练集/测试集同分布假设成立时，符号差异可解读为机制差异。本工作该假设受 LOSO 失败（−0.051）约束，**方向对比应在端位底物内部**（CHO vs ECH 等），不应跨到底物维度的"测试"集合。
2. **top-1 特征类型对比**（动力学 vs 电子）：top-1 类别错位（CHO vs 端位）是更稳健的诊断，因为它不依赖 signed value 的具体数值，仅依赖特征类型。CHO 的 top-1 = `time_log`，4 个端位底物的 top-1 = `sub_homo_eV`——这一类别级对比与单特征方向反转是双重独立的证据。

SHAP 不能直接证伪或证实以下假说：

- CHO 的低产率是因为环内 LUMO 不能与侧链 HOMO 形成有效耦合（需要 DFT 过渡态验证，§4.4 列为未来工作）。
- 选择性机理（Lewis 酸 vs 氢键）对 CHO 的影响大于电子描述符（需要单独的反应条件分离实验）。
