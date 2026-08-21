# ORCA TS Optimization Results (B97-3c, SMD(DMSO))

| Substrate | Final Energy (Ha) | Imaginary Freq (cm⁻¹) | Status |
|-----------|---------------------|-------------------------|---------|
| PO        | -2956.5858          | -606.64                 | ✅ Complete |
| SO        | -3148.2459          | N/A (200 cyc, not converged) | ⚠️ Restart needed |
| ECH       | -3416.1251          | -610.10                 | ✅ Complete |
| CHO       | -3072.9585          | N/A (200 cyc, not converged) | ⚠️ Restart needed |
| IGE       | -3149.6473          | -595.15                 | ✅ Complete |

## Notes
- 3/5 底物得到 1 个主导虚频（约 -600 cm⁻¹），符合环加成 TS 特征
- SO/CHO 初始结构缺少 S 原子（SMILES 解析错误），需重新生成
- 所有计算使用 B97-3c → 高精度单点另行需要
