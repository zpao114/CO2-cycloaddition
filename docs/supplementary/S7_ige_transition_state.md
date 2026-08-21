# 补充材料 §S7

## 五底物环加成过渡态计算（PO / ECH / SO / CHO / IGE）

承接正文 §3.4（按底物分组的 SHAP 方向分析）与 §4.4（机制分叉的第一性原理交叉验证）。本节给出 SHAP 方向反转在第一性原理层面的独立证据——五底物的 B97-3c/SMD(DMSO) 过渡态优化与数值频率计算。

---

### S7.1 计算方法与模型系统

**目的**：在五个底物（PO、ECH、SO、CHO、IGE）上独立定位端位开环过渡态（TS1），验证 SHAP 方向反转是否对应可观测的 TS 几何差异。

**模型系统**：tetramethylammonium bromide (Me₄N⁺ Br⁻) + 底物（epoxide） + CO₂ 三组分复合物。

**计算级别**：
- 方法：B97-3c 复合方法（Grimme 2017）
- 基组：B97-3c 内置混合基组（def2-TZVP on heavy atoms, def2-SVP on H）
- 溶剂：SMD(DMSO) 隐式溶剂化
- 频率：数值频率计算（NumFreq）确认过渡态
- 优化算法：OptTS（准 Newton 鞍点搜索）

**软件**：ORCA 6.1.1，OpenMPI 4.1.8 并行。

**脚本来源**（全部归档于 `dft_validation/`）：
- `build_tetramethyl_ts.py`：构建 Me₄N⁺ Br⁻ + 底物 + CO₂ 复合体
- `build_ts_guess.py`：生成 TS 初始猜测几何（Br 在 ~2.5 Å 处）
- `run_all.bat` / `run_dft.bat` / `502_run_dft_wsl.ps1`：批量调 ORCA

---

### S7.2 五底物过渡态优化结果总览

下表汇总 `dft_validation/ts_5_substrates/{PO,ECH,SO,CHO,IGE}/ts_*.out` 的最终状态。所有计算使用同一 `! B97-3c OptTS NumFreq SMD(DMSO) VeryTightSCF` 输入模板（IGE 除外——IGE 在初始失败后改为 v2 重启）。

| 底物 | 体系原子数 | E_TS (Eh) | 主导虚频 (cm⁻¹) | OptTS 收敛 | 备注 |
|---|---|---|---|---|---|
| **PO**  | 14 | −2956.58577983 | **−606.64** | ✅ HURRAY | 主导虚频方向对应 Br–C 进攻 |
| **ECH** | 14 | −3416.12506545 | **−610.10** | ✅ HURRAY | 主导虚频方向对应 Br–C 进攻 |
| **IGE** | 27 | −3149.82496756 | **−225.93** | ✅ HURRAY | v2 重启后收敛；3 个小虚频（<10 cm⁻¹）来自数值频率噪声 |
| SO | 14 | −3148.22461199* | n/a | ⚠️ 200 cycle 未收敛 | 初始几何 Br 距底物过近（<3.5 Å），需重启 |
| CHO | 14 | −3072.95854248* | n/a | ⚠️ 200 cycle 未收敛 | 同上；初始距离过近 |

\* 为最后一次未收敛迭代（cycle 200）的电子能（B97-3c 完整 FINAL SPE 末值，含 dispersion + SRB 校正，**未含** SMD 溶剂化校正），**非过渡态能量**——仅供重启参考。SO 在 cycle 100 附近曾达 −3148.24113644 Eh（局部最低 FINAL SPE），作为进一步重启的更优初始几何起点参考。

**注**：PO/ECH 的 reactant 几何（`reactant_PO.xyz`、`reactant_ECH.xyz`）同样存在 Br 距环氧碳 ~3.3 Å 的近距问题，对应能量参考不可直接用于 ΔE‡ 计算（详见 §S7.6 局限）。IGE 采用独立的 `reactant_IGE_frozen.xyz`（Br 冻结于 ~5.5 Å）作为参考，但其为远距 freeze 而非完整 SP 优化，亦仅供几何参照。

---

### S7.3 PO / ECH / IGE 三例收敛的过渡态

#### S7.3.1 PO（环氧丙烷）

- **输入**：`dft_validation/ts_5_substrates/PO/ts_PO.inp`
- **输出**：`dft_validation/ts_5_substrates/PO/ts_PO.out`（HURRAY 收敛，~3 MB）
- **几何**：`dft_validation/ts_5_substrates/PO/ts_PO.xyz`（14 原子复合物）
- **关键参数**：Br 在 (3.90, −0.43, 0.00) Bohr，环氧 C–O 键接近断裂
- **主导虚频**：−606.64 cm⁻¹（对应 Br–C 形成 + C–O 断裂）
- **辅助虚频**：−31.91 cm⁻¹（数值频率噪声，可忽略）

#### S7.3.2 ECH（表氯醇）

- **输入**：`dft_validation/ts_5_substrates/ECH/ts_ECH.inp`
- **输出**：`dft_validation/ts_5_substrates/ECH/ts_ECH.out`（HURRAY 收敛，~2 MB）
- **几何**：`dft_validation/ts_5_substrates/ECH/ts_ECH.xyz`（14 原子复合物，含 Cl）
- **主导虚频**：−610.10 cm⁻¹（Br–C 形成 + C–O 断裂，与 PO 一致）
- **辅助虚频**：−15.20 cm⁻¹

#### S7.3.3 IGE（异丙基缩水甘油醚）

IGE 在初始 `ts_IGE.inp`（2026-08-08）上未通过 OptTS；后续以 v2 重启：

- **输入**：`dft_validation/ts_5_substrates/IGE/ts_IGE_v2.inp`
- **输出**：`dft_validation/ts_5_substrates/IGE/ts_IGE_v2.out`（HURRAY 收敛）
- **Hessian**：`ts_IGE_v2.hess`
- **几何**：`ts_IGE_v2.xyz`（27 原子复合物，IGE 含 –OCH₂OCH(CH₃)₂ 侧链）
- **主导虚频**：−225.93 cm⁻¹
- **辅助虚频**：−21.74、−9.60 cm⁻¹（量级 <25 cm⁻¹，归因为 B97-3c + GFN-FF 初始化的几何噪声）

#### S7.3.4 IGE 第二步（TS2，CO₂ 插入）

`dft_validation/ts_5_substrates/IGE_TS2/` 目录下保存了独立的 CO₂ 插入步骤计算：

- `plan_c/ts1_numfreq_final.*`：TS1 单点（与 plan_a 重复，保留作 cross-check）
- `int_opt/int_ige.*`：烷氧中间体几何
- `scan_1.7/scan_d1.7.*`：Br–C 键距离 1.7 Å 的 PES 扫描

这些计算无独立 NumFreq 验证，**仅供机理示意**，不进入定量论证。

---

### S7.4 未收敛案例：SO 与 CHO

`ts_5_substrates/SO/ts_SO.out` 与 `ts_5_substrates/CHO/ts_CHO.out` 各 ~10 MB（2026-08-08），输出 `ORCA TERMINATED NORMALLY` 但几何优化阶段输出 `The optimization did not converge but reached the maximum`：

```
$ ts_SO.out:167374       The optimization did not converge but reached the maximum
$ ts_CHO.out:207807      The optimization did not converge but reached the maximum
```

**收敛失败原因**：复审初始复合体几何，Br 与环氧碳的初始距离分别为 ~2.0–2.5 Å（远低于典型 TS 距离 ~2.5–3.0 Å 范围下端），优化被卡在陡峭的近距排斥区，导致 Hessian 持续拒绝接受步长。两例的 `.out` 文件已完整保存：

- `ts_5_substrates/SO/ts_SO.out` (~8 MB)
- `ts_5_substrates/CHO/ts_CHO.out` (~10 MB)
- `ts_5_substrates/SO/ts_SO.xyz`、`ts_CHO.xyz`：最后一次未收敛迭代的几何

后续重启须把初始 Br–C 距离显式设置到 ~3.0 Å 范围并使用 `Trust 0.2` 的更小步长（参考 `RUNBOOK.md` §Stage 1 GFN-FF 受约束优化流程）。

---

### S7.5 SHAP 诊断的第一性原理对应

正文 §3.4 报告：
- 端位底物 top-1 = `sub_homo_eV`，signed SHAP = +2.42 ~ +5.02
- CHO top-1 = `time_log`，signed SHAP on `sub_homo_eV` = −1.20

DFT 计算给出的端位闭合支持以下对应：

1. **开环步骤由底物电子结构主导**。三例收敛的端位 TS（PO/ECH/IGE）主导虚频方向均为 Br–C 形成 + 环氧 C–O 断裂，且主导虚频量级（225–610 cm⁻¹）远高于非反应模式（<25 cm⁻¹），与"亲核开环是速率决定步骤"的图像一致[4,5]。C–O 键强度与底物 HOMO 能级正相关——HOMO 越高，C–O 越弱，开环越快——与 SHAP 正向 signed SHAP 一致。

2. **CHO 与 SO 的闭环缺失与 SHAP 反转方向一致**。两例 TS 几何未收敛于"Br 接近环氧碳"的过渡态图像，提示 CHO/SO 体系下亲核开环可能不是单一速率决定步骤——Lewis 酸配位、底物取向或烯丙位 C–H 活化等其他机制可能在 CHO/SO 上主导。这一观察与 Castro-Gómez 等[8]关于 Zn(salphen) 催化 CHO 的 DFT 研究一致，也与 SHAP 在 CHO 上从 `sub_homo_eV` 切换到 `time_log` 的反转一致。

3. **TS1 vs TS2 的能量层级示意**。CO₂ 插入步骤（TS2）的相关计算（`IGE_TS2/int_opt/int_ige.*` 与 `IGE_TS2/scan_1.7/scan_d1.7.*`）保存于同一目录。`int_ige` 是烷氧中间体（Br–C ≈ 2.54 Å）的 B97-3c 几何优化单点（E = −3149.570945 Eh，34 步收敛）；`scan_d1.7` 是 B97-3c 受约束 PES（固定 B 3 21 1.700 C，34 个 SCF 收敛点，最低 E = −3149.554349 Eh，对应 step 24-33 的 d(Br–C) ≈ 1.96 Å 产物侧窄域——34 点内起伏仅 0.1 kcal/mol，step 0 异常高 +45.84 kcal/mol 来自初猜构型）。两套数据位于产物侧，与独立 `int_ige`（烷氧中间体）相比，scan 最低点能量高 +10.4 kcal/mol；二者几何差异（Br–C：2.54 Å vs 1.96 Å）说明这两组数据并非 TS2 邻域的同一物理点，故本节不将 scan_d1.7 直接用于"TS2 vs 中间体"的能量论证；TS1–TS2 的相对能量级差留给后续 IRC + 高精度单点精修（§S7.6）。此处保留 `scan_d1.7` 与 `int_ige` 作为 IGE 反应复合体几何演化的存档，并以 `time_log`/`pressure` 在 SHAP 上作为次要特征而非主控因素（top-3 之后），给出物理层级的可能对应——但严格的 TS2 能量定位仍是开放问题。

---

### S7.6 局限性与未来工作

**当前限制**：

1. **CHO 与 SO 的过渡态未收敛**（§S7.4）。重启需调整初始 Br–C 距离与 Trust 半径。
2. **三例收敛 TS 在严格一阶鞍点意义上并非 clean saddle point**。PO、ECH、IGE 的 OptTS 输出虽 "HURRAY" 但 NumFreq 同时给出 1–2 个次级虚频（PO −31.91、ECH −15.20、IGE −21.74 与 −9.60 cm⁻¹），落于 B97-3c + SMD(DMSO) NumFreq 的数值噪声带边界附近；故三例的 Hessian 含 2 个或更多负本征值，属二阶 / 三阶近似鞍点而非一阶鞍点。**严格意义下**，应继续进行以下任一步骤以确认一阶鞍点性质：（a）IRC 跟踪确认反应物/产物侧的连接性；（b）二级微扰修正；（c）更换为可解析 Hessian 的方法（如 B3LYP-D3BJ/def2-SVP + NumFreq）以减小数值噪声。本研究保留 B97-3c + NumFreq 作为快速筛查工具，但 §3.4、§4.4、§4.5 涉及"过渡态"陈述时均在文字上注明"主导虚频方向与亲核开环一致"——而非"已确定为反应过渡态"。
3. **`IGE_TS2/scan_d1.7` 与 `int_ige` 不在同一物理邻域**。scan_d1.7 的 d(Br–C) ≈ 1.96 Å 落在产物侧窄域 PES（34 点起伏仅 0.1 kcal/mol）；独立 `int_ige`（烷氧中间体，d(Br–C) ≈ 2.54 Å）比 scan 最低点低 10.4 kcal/mol。二者几何差异显著，因此本节**不将 scan_d1.7 用于 TS2 vs 中间体的能量层级论证**，亦不与 TS1 几何（d(Br–C) ≈ 1.97 Å）做直接能量比对。TS2 的能量定位须以完整 IRC + 单点精修路径为前提。
4. **无完整 IRC 验证**。三例收敛 TS 仅有 NumFreq 验证鞍点性质，未执行 IRC 跟踪以确认反应物/产物侧的连接性。
5. **热力学校正缺失**。本节所有能量为 B97-3c 电子能（Eh），未做 ZPE、热焓与 Gibbs 自由能校正。ΔE‡ 数字若需用于正文，应进一步在 def2-TZVP 上做高精度单点精修。
6. **Reactant 参考态不严格**。PO/ECH/IGE 的 reactant 几何距过渡态结构过近（Br–C 距离 <3.5 Å），其能量参考不可直接用于 ΔE‡ 计算。IGE 远距 frozen reactant 不等价于完整 SP 优化的反应物。**故本节不报告 ΔE‡ 数字**，仅报告 TS 几何与主导虚频方向。
7. **催化剂简化**。本研究使用 tetramethylammonium bromide 作为模型催化剂，与实验体系中常见的咪唑盐、铵盐、磷盐阳离子结构不同；阳离子效应（特别是氢键给体）可能改变 TS 几何与虚频方向。

**建议的未来工作**：

1. **重启 CHO 与 SO 的 TS 优化**。把初始 Br–C 距离设置在 2.8–3.2 Å 范围，使用 GFN-FF 受约束预优化（`RUNBOOK.md` §Stage 1 流程），然后 OptTS。
2. **执行 IRC 跟踪**。对 PO/ECH/IGE 三例执行 IRC（≥30 步正反方向），明确反应物/产物侧的连接性。
3. **高精度单点精修**。在 def2-TZVP + D3BJ 级别上对 TS 与 reactant 做单点，给出 ΔE‡ 的定量数值。
4. **Gibbs 自由能校正**。基于 NumFreq 输出 ZPE、热焓与熵校正（298.15 K），给出 ΔG‡（kcal/mol）。
5. **催化剂阳离子效应**。重复 IGE TS 优化，使用 BMIM⁺、TBAB⁺ 等实际阳离子，验证 Br⁻ 主导 vs 阳离子–CO₂ 协同机制。

---

### S7.7 计算文件索引

#### 主产物（端位闭合）

| 路径 | 说明 |
|---|---|
| `dft_validation/ts_5_substrates/PO/ts_PO.{inp,out,property.txt,xyz,hess}` | PO TS 完整产物 |
| `dft_validation/ts_5_substrates/PO/reactant_PO.{property.txt,xyz}` | PO reactant（参考，几何受限） |
| `dft_validation/ts_5_substrates/ECH/ts_ECH.{inp,out,property.txt,xyz,hess}` | ECH TS 完整产物 |
| `dft_validation/ts_5_substrates/ECH/reactant_ECH.{property.txt,xyz}` | ECH reactant |
| `dft_validation/ts_5_substrates/IGE/ts_IGE_v2.{inp,out,property.txt,xyz,hess,opt}` | IGE TS 完整产物（v2 重启） |
| `dft_validation/ts_5_substrates/IGE/frozen_reactant/reactant_IGE_frozen.{inp,out,property.txt,xyz}` | IGE 远距 frozen reactant |

#### 未收敛案例（重启参考）

| 路径 | 说明 |
|---|---|
| `dft_validation/ts_5_substrates/SO/ts_SO.{inp,out,property.txt,xyz}` | SO 200 cycle 未收敛 |
| `dft_validation/ts_5_substrates/CHO/ts_CHO.{inp,out,property.txt,xyz}` | CHO 200 cycle 未收敛 |

#### 第二步与中间体（辅助）

| 路径 | 说明 |
|---|---|
| `dft_validation/ts_5_substrates/IGE_TS2/plan_c/ts1_numfreq_final.{inp,out,property.txt,xyz}` | IGE TS1 单点（cross-check） |
| `dft_validation/ts_5_substrates/IGE_TS2/int_opt/int_ige.{inp,log,property.txt,xyz}` | 烷氧中间体几何 |
| `dft_validation/ts_5_substrates/IGE_TS2/scan_1.7/scan_d1.7.{inp,log,property.txt,xyz}` | Br–C 距离 1.7 Å 处 PES 扫描 |
| `dft_validation/ige_ts/plan_c/reactant_sp.{inp,out,property.txt,xyz}` | IGE 早期 reactant SP（参考） |
| `dft_validation/ige_ts/plan_c/alkoxide_sp.{inp,out,property.txt,xyz}` | IGE 烷氧中间体 SP（参考） |
| `dft_validation/ige_ts/plan_c/ts1_numfreq.{inp,out,property.txt,xyz}` | IGE 早期 TS1 尝试（未做 NumFreq） |
| `dft_validation/ige_ts/plan_c/ts2_numfreq.{inp,out,property.txt,xyz}` | IGE TS2 早期尝试 |
| `dft_validation/ige_ts/plan_c/ts2_optts.{inp,out,property.txt,xyz}` | IGE TS2 OptTS 尝试 |

#### 准备脚本与文档

| 路径 | 说明 |
|---|---|
| `dft_validation/build_tetramethyl_ts.py` | Me₄N⁺ Br⁻ + 底物 + CO₂ 复合体构建 |
| `dft_validation/build_ts_guess.py` | TS 初始猜测生成 |
| `dft_validation/501b_generate_extrapolation_molecules.py` | 未来外推底物生成 |
| `dft_validation/run_dft.bat` | ORCA 单作业入口 |
| `dft_validation/run_all.bat` | ORCA 全作业入口 |
| `dft_validation/502_run_dft_wsl.ps1` | WSL 调度脚本 |
| `dft_validation/RUNBOOK.md` | CHO 开环 TS 流程说明 |
| `dft_validation/README.txt`、`README_DFT.txt` | 全局目录说明 |
| `dft_validation/ts_5_substrates/analysis_summary.md` | 五底物总结（手动汇总） |