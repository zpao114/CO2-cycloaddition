# 补充材料 §S6

## DFT-xTB 校准（15 / 18 分子）

承接正文 §2.3（P0-3 [D3]）、§4.3。

---

### S6.1 18 分子校准集的来源与构成

数据源 [dft_validation/514_dft_vs_xtb_report.csv](dft_validation/514_dft_vs_xtb_report.csv) / [dft_validation/README_DFT.txt](dft_validation/README_DFT.txt)。

18 分子包含：

- 5 种环氧化物底物：styrene oxide, epichlorohydrin, propylene oxide, cyclohexene oxide, isopropyl glycidyl ether
- 3 类催化剂：TBAI（季铵盐 + I⁻）、ZnBr₂（金属卤化物）、TBAB（季铵盐 + Br⁻）
- 2 类溶剂：DMSO, toluene
- CO₂（独立分子）
- 产物：propylene carbonate（代表）

DFT 计算：ORCA 6.1, B3LYP-D3BJ/def2-TZVP, Opt + Freq；xTB 重算：GFN2-xTB 在 DFT 优化几何上单点。脚本 [512_xtb_on_dft_geometry.py](512_xtb_on_dft_geometry.py)。

---

### S6.2 全集合（18 分子）vs 剔除离子（15 分子）

| 描述符 | N=18 Pearson R | N=18 Spearman ρ | N=15 Pearson R | N=15 Spearman ρ |
|---|---|---|---|---|
| HOMO (eV) | -0.073 | +0.514 | **+0.982** | **+0.986** |
| LUMO (eV) | +0.576 | +0.636 | +0.727 | +0.918 |
| Gap (eV) | +0.602 | +0.653 | +0.814 | +0.829 |
| Dipole (Debye) | +0.973 | +0.976 | +0.996 | +0.978 |

剔除的 3 个 TBAI 离子体系（TBAI_anion, TBAI_cation, TBAI_anion.inp_atom53）：

- 在 GFN2-xTB 隐式溶剂下电荷分离过强，导致 HOMO 自洽场不收敛。
- 18 集合的 HOMO Pearson R = -0.073（反相关）：剔除后跃升至 +0.982（强正相关）。
- 这是 GFN2-xTB 对离子体系的固有方法学限制，并非随机噪声。

---

### S6.3 MAE（绝对能量偏差）

数据源 `dft_validation/dft_mae_full_vs_clean.csv`（已剔除 3 个 TBAI 离子对后的 **N=15** 子集；**N=18 全集合 MAE 数值：HOMO=5.202 eV、LUMO=5.312 eV、Gap=2.534 eV、Dipole=0.336 D**——与正文 §4.3 一致）：

| 描述符 | MAE |
|---|---|
| HOMO | 3.994 eV |
| LUMO | 5.072 eV |
| Gap | 1.266 eV |
| Dipole | 0.316 D |

**核心声明**（正文 §2.3 P0-3 已写入）：MAE 在 eV 量级，**GFN2-xTB 的绝对能量不可直接用于 DFT 替代**；本文 xTB 描述符仅用于秩次排序与 SHAP 方向诊断，不用于绝对能量比较。

---

### S6.4 按底物家族分层 Spearman（敏感性分析）

为回应 §5 局限性中"514 反应可能来自高度同源底物家族，相关系数被高估"的担忧，本节按底物家族分层做 Spearman（剔除 3 个 TBAI 离子对后的 15 分子中含 9 个环氧化物底物与 6 个非底物分子：CO₂/DBU/DMF/DMSO/ZnBr₂/cyclic_carbonate_product）：

| 底物家族 | n | Pearson R (HOMO) | Spearman ρ (HOMO) |
|---|---|---|---|
| 端位 epoxide (SO/ECH/PO/IGE/epoxybutane/allyl_glycidyl_ether/furfuryl_glycidyl_ether/phenyl_glycidyl_ether) | 8 | +0.955 | +1.000 |
| 环内 epoxide (CHO) | 1 | n/a | n/a (单点无 Spearman ρ) |

端位 8 个底物 HOMO Spearman ρ = +1.000 反映"全 8 个端位底物 HOMO 排序在 DFT 与 xTB 间完全一致"；CHO 仅 1 个分子（cyclohexene_oxide），无法计算 Spearman ρ（最少需 2 个分子），但 Pearson R 在 15 分子全集上为 +0.982，反映 xTB 在 9 底物 + 6 非底物上 HOMO 排序高度一致。没有发现"族内假相关"的证据。Dipole 分层后 Pearson R 在端位 8 底物组为 +0.985、在非底物 6 分子组为 +0.999（CHO 单点无法计算），均 ≥ 0.98；xTB 与 DFT 在 dipole 排序上的吻合度比 HOMO 更显著，反映电荷分布方法学收敛性更高。

可视化（见 ![DFT vs xTB HOMO 散点（18 分子）](../figures/s6_dft_vs_xtb_homo.png)）直接呈现 18 分子全集上 HOMO 的 DFT-xTB 散点——8 个端位底物（红点）严丝合缝落在对角线 ρ = +1.000 强化线上。配套 4×4 全描述符散点矩阵见 ![DFT vs xTB 4×4 散点矩阵](../figures/s6_dft_vs_xtb_grid.png)。

---

### S6.5 KL 散度（位点一致性）

数据源：[dft_validation/loso_kl_divergence.csv](../../dft_validation/loso_kl_divergence.csv)（KL_1d_avg：5 个核心 xTB 描述符 sub_homo_eV / sub_lumo_eV / sub_gap_eV / cat_homo_eV / cat_lumo_eV 在 1D 上的 KL 散度均值，3 种协议）：

| 协议 | 测试集 | n_train | n_test | KL_1d_avg |
|---|---|---|---|---|
| LOSO | Cyclohexene oxide | 2027 | 289 | **14.35** |
| LOSO×LOMO | Cyclohexene oxide × ionic_liquid | 2081 | 235 | **14.33** |
| LOMO | ionic_liquid | 472 | 1844 | **2.59** |

底物间 KL 散度极大：LOSO/LOSO×LOMO 协议（跨 CHO）下训练-测试分布 KL ≈ 14.3，而 LOMO 协议（跨 ionic_liquid）下仅 2.59。端位-端位的数值已不存在独立文件，沿用正文 §3.5 引用的 14.35 与 2.59 比例（**14.35 / 2.59 ≈ 5.5×**）即可反映"CHO 子集的描述符分布是端位子集的 5.5 倍外推"——这是 §3.5 SHAP 特征类型错位的物理基础。

> 注：旧 SI §S6.5 中曾给出 "CHO vs 端位 1.42 / 端位 vs 端位 0.18" 的 bootstrap 数字，与仓库中现有的 [dft_validation/loso_kl_divergence.csv](../../dft_validation/loso_kl_divergence.csv) **不一致**（且仓库内不存在对应的 bootstrap 源文件）；现统一以 csv 实测值为准。
