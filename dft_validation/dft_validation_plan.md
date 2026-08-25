# DFT验证方案 - CO2环加成反应

> **审计提示（2026-08-22）** — 本文档记录的是计划/方案，实际运行集合存在以下已知问题（不影响作为方法论展示）：
>
> 1. `ige_ts/plan_c/` 与 `ts_5_substrates/` 目录中 **5 个 TS** 使用 `* xyz 0 2`（闭壳层双重态），物理意义应为 `* xyz -1 1`（开壳层阴离子双重态）。能量偏差约 5–10 kcal/mol。
> 2. 所有 TS **未跑 IRC**。TS 标记基于单点负频率，未与反应物/产物极小点相连。
> 3. `extended/` 与 `ige_ts/` / `ts_5_substrates/` 混合了 B97-3c、ωB97X-D3、B3LYP-D3 三个方法/基组。
> 4. `TBAI_anion.xyz`（HOMO=−23.85 eV）与 `TBAI_cation.xyz`（HOMO=−12.10 eV）为单原子 SP，HOMO 实际为 1s 轨道；已从 `s6_dft_vs_xtb_homo.{pdf,png}` 中过滤，但仍在 `dft_xtb_calibration_full.csv` 中。
> 5. GFN2-xTB vs DFT 的 HOMO/LUMO 系统性偏移约 +4 eV（GFN2-xTB 更负）。
> 6. `514b_dft_transition_state.py` 中 CHO / PO 的 TS 几何来自手画猜测，~50 cycle 未收敛。
>
> 详见 [`docs/CODE_AUDIT.md`](../docs/CODE_AUDIT.md) §2。

## 目标
用ORCA DFT计算验证ML模型的预测，特别是：
1. 验证GFN2-xTB前线轨道计算结果的可靠性
2. 计算催化剂-底物相互作用能
3. 计算反应能垒（如果时间允许）

## 建议验证的候选物（从虚拟筛选Top10中选择）

| 编号 | 底物 | 催化剂 | 溶剂 | 预测产率 | 用途 |
|------|------|--------|------|----------|------|
| V1 | Bisphenol A diglycidyl ether | ZnBr2 | DMSO | 97.4% | 高产率验证 |
| V2 | Furfuryl glycidyl ether | ZnBr2 | DMF | 94.0% | 高产率验证 |
| V3 | Bisphenol A diglycidyl ether | ZnBr2 | - | 91.6% | 无溶剂对比 |
| V4 | (训练集低产率样本) | ZnBr2 | - | ~30-50% | 低产率对照 |
| V5 | (训练集中等产率) | ZnBr2 | - | ~60-70% | 中等产率对照 |

## 计算级别建议

### 方案A：快速验证（推荐首选）
```
! B3LYP def2-SVP Opt Freq
! D3BJ
%pal nproc 8 end
```
- 几何优化 + 频率计算
- 验证GFN2-xTB的HOMO/LUMO结果
- 耗时：每体系 ~10-30分钟

### 方案B：中等精度
```
! B3LYP def2-TZVP Opt Freq
! D3BJ
%pal nproc 16 end
```
- 更高精度几何优化
- 适合过渡态搜索
- 耗时：每体系 ~1-2小时

### 方案C：高精度（最终投稿用）
```
! wB97X-D4 def2-TZVP def2/J TightOpt Freq
! SlowConv
%pal nproc 16 end
```
- 色散校正 + 高基组
- 适合最终验证
- 耗时：每体系 ~4-8小时

## 建议工作流程

1. **先做方案A** - 验证GFN2-xTB结果
2. **如有需要** - 方案B做过渡态
3. **最终** - 方案C确认关键结果
