# DFT验证方案 - CO2环加成反应

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
