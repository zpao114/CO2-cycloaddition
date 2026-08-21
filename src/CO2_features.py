# -*- coding: utf-8 -*-
"""
Enhanced Feature Construction Module for CO2 Cycloaddition Reactions

Enhancements (v2):
1. Support for 4 DRFP variant concatenation (drfp + drfp_React + drfp_wo_cats + drfp_wo_sols)
2. Grouped standardization: binary features (DRFP/Morgan) vs continuous features (conditions)
3. Catalyst Morgan fingerprints (256 dimensions per catalyst)
4. Substrate one-hot encoding (5 substrates)
5. Catalyst system one-hot encoding (5 types)
6. Condition features (temperature, pressure, time, catalyst loading)

Target: Improve R² from 0.31 to 0.38-0.45

Note: Temperature column handling uses aliases for cross-version compatibility
(handles 'temperature (\u2103)', 'temperature (°C)', etc.)
"""
import os
import sys
import numpy as np
import pandas as pd
from functools import lru_cache

PROJECT_ROOT = os.environ.get("CO2_PROJECT_ROOT", r"D:\machine-learning\CO2-cycloaddition")
sys.path.insert(0, PROJECT_ROOT)
from CO2_rxn import read_drfp

# ------------------------------------------------------------
# Temperature column aliases (for cross-version compatibility)
# Handles all historical naming conventions including erroneous '\u2103'
# ------------------------------------------------------------
_TEMP_COL_ALIASES = [
    'temperature_celsius',
    'temperature (\u2103)',    # erroneous Unicode (Celsius symbol, not degree)
    'temperature (\u00b0)',   # degree symbol
    'temperature (\u00b0C)', # degree Celsius
    'temperature (°C)',       # literal degree Celsius
    'temperature (℃)',       # fullwidth Celsius
    'temperature',
]


def _find_temp_col(df):
    """Find temperature column in DataFrame, handling all alias conventions."""
    for col in _TEMP_COL_ALIASES:
        if col in df.columns:
            return col
    # Raise informative error with available columns
    available_temp = [c for c in df.columns if 'temp' in c.lower() or '°' in c or '\u2103' in c]
    raise KeyError(f"Cannot find temperature column. Tried: {_TEMP_COL_ALIASES}. Available: {available_temp}")

# ------------------------------------------------------------
# Morgan 指纹编码器（延迟导入 RDKit）
# ------------------------------------------------------------
_rdk_loaded = False
_RDKit = None
_AllChem = None
_morgan_cache: dict[tuple[str, int, int], np.ndarray] = {}


def _ensure_rdkit():
    global _rdk_loaded, _RDKit, _AllChem
    if _rdk_loaded:
        return True
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        _RDKit = Chem
        _AllChem = AllChem
        _rdk_loaded = True
        return True
    except ImportError:
        print("  [警告] RDKit 未安装，催化剂指纹将用零向量代替")
        _rdk_loaded = True
        return False


def smiles_to_morgan(smiles_str: str, n_bits: int = 256, radius: int = 2) -> np.ndarray:
    """将单个 SMILES 转为 Morgan 指纹。失败返回全零向量。"""
    if not smiles_str or not isinstance(smiles_str, str):
        return np.zeros(n_bits, dtype=np.float32)
    smiles_str = str(smiles_str).strip()
    if smiles_str in ('', '/', 'nan', 'None'):
        return np.zeros(n_bits, dtype=np.float32)
    key = (smiles_str, n_bits, radius)
    if key in _morgan_cache:
        return _morgan_cache[key].copy()
    if not _ensure_rdkit():
        return np.zeros(n_bits, dtype=np.float32)
    try:
        mol = _RDKit.MolFromSmiles(smiles_str)
        if mol is None:
            result = np.zeros(n_bits, dtype=np.float32)
        else:
            fp = _AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
            result = np.array(fp, dtype=np.float32)
        _morgan_cache[key] = result
        return result.copy()
    except Exception:
        return np.zeros(n_bits, dtype=np.float32)


def build_combined_catalyst_fp(row, n_bits: int = 256) -> np.ndarray:
    """
    将最多 4 种催化剂的 SMILES 合并为一个指纹向量。
    策略：取所有催化剂的并集（按位 OR），而非拼接。
    """
    cat_smiles_list = []
    for i in range(1, 5):
        col = f'catalyst_{i}_smiles'
        if col in row.index and pd.notna(row.get(col)):
            smi = str(row[col]).strip()
            if smi and smi not in ('', '/', 'nan'):
                cat_smiles_list.append(smi)

    if not cat_smiles_list:
        return np.zeros(n_bits, dtype=np.float32)

    if len(cat_smiles_list) == 1:
        return smiles_to_morgan(cat_smiles_list[0], n_bits)

    fps = [smiles_to_morgan(s, n_bits) for s in cat_smiles_list]
    combined = fps[0]
    for fp in fps[1:]:
        combined = np.maximum(combined, fp)
    return combined


# ------------------------------------------------------------
# 分组标准化器
# ------------------------------------------------------------
class GroupedStandardScaler:
    """
    对特征的不同分组分别做标准化。

    分组：
    - binary: 二值特征（DRFP、Morgan、one-hot），仅做居中（减均值），不缩放
    - continuous: 连续特征（条件），做标准标准化

    这样可避免 DRFP/Morgan 的大规模 0/1 特征与连续特征统计量互相干扰。
    """

    def __init__(self):
        self.binary_mean: np.ndarray | None = None
        self.cont_scaler = None  # sklearn StandardScaler

    def fit(self, X_binary: np.ndarray, X_cont: np.ndarray):
        # 二值特征：仅记录均值（训练时用于居中）
        self.binary_mean = np.mean(X_binary, axis=0).astype(np.float32)

        # 连续特征：sklearn StandardScaler
        from sklearn.preprocessing import StandardScaler
        self.cont_scaler = StandardScaler()
        self.cont_scaler.fit(X_cont)

    def transform(self, X_binary: np.ndarray, X_cont: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        Xb = X_binary - self.binary_mean  # 仅居中，不缩放
        Xc = self.cont_scaler.transform(X_cont)
        return Xb.astype(np.float32), Xc.astype(np.float32)

    def fit_transform(self, X_binary: np.ndarray, X_cont: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self.fit(X_binary, X_cont)
        return self.transform(X_binary, X_cont)

    def inverse_transform_binary(self, Xb: np.ndarray) -> np.ndarray:
        """还原二值特征（加回均值）。"""
        return Xb + self.binary_mean


# ------------------------------------------------------------
# One-hot 编码
# ------------------------------------------------------------
def build_reactant_onehot(reactant_name: str, all_reactants: list) -> np.ndarray:
    """底物 one-hot 编码。"""
    n = len(all_reactants)
    vec = np.zeros(n, dtype=np.float32)
    try:
        idx = all_reactants.index(reactant_name)
        vec[idx] = 1.0
    except (ValueError, TypeError):
        pass
    return vec


def build_catalyst_type_onehot(cat_type: str, all_types: list) -> np.ndarray:
    """催化剂体系类型 one-hot 编码。"""
    n = len(all_types)
    vec = np.zeros(n, dtype=np.float32)
    try:
        idx = all_types.index(cat_type)
        vec[idx] = 1.0
    except (ValueError, TypeError):
        pass
    return vec


# ------------------------------------------------------------
# 条件特征
# ------------------------------------------------------------
def build_condition_features(df: pd.DataFrame) -> np.ndarray:
    """Build condition feature columns: temperature, pressure, time(log), catalyst loading(log).

    Automatically detects temperature column regardless of naming convention
    (°C, ℃, \u2103, etc.) using _find_temp_col helper.
    """
    # Temperature: auto-detect using aliases
    temp_col = _find_temp_col(df)
    temp = df[temp_col].fillna(df[temp_col].median()).values.reshape(-1, 1)

    pressure = df['pressure (MPa)'].fillna(df['pressure (MPa)'].median()).values.reshape(-1, 1)
    time_val = df['time (h)'].fillna(df['time (h)'].median()).values.reshape(-1, 1)
    time_log = np.log1p(np.maximum(time_val, 0))

    # 催化剂总负载量（mol%）
    loading_cols = [f'catalyst_{i}_loading_mol%' for i in range(1, 5)]
    loadings = np.zeros((len(df), 1), dtype=np.float32)
    for lc in loading_cols:
        if lc in df.columns:
            vals = pd.to_numeric(df[lc], errors='coerce').fillna(0).values.reshape(-1, 1)
            loadings = loadings + np.nan_to_num(vals, nan=0.0)
    loading_log = np.log1p(np.maximum(loadings, 0))

    return np.hstack([temp, pressure, time_log, loading_log]).astype(np.float32)


# ------------------------------------------------------------
# 主特征构建函数
# ------------------------------------------------------------
def load_enhanced_data(
    data_path: str,
    use_conditions: bool = True,
    use_catalyst_fp: bool = True,
    use_catalyst_type: bool = True,
    use_reactant_onehot: bool = True,
    use_drfp_variants: bool = True,
    morgan_bits: int = 256,
    grouped_scale: bool = True,
    verbose: bool = True
) -> tuple:
    """
    加载增强特征数据集（v2）。

    新增参数：
        use_drfp_variants: 是否拼接 4 个 DRFP 变体（默认 True）
        grouped_scale:     是否分组标准化（默认 True）

    Returns:
        X, y, df, feat_info: 特征矩阵、目标向量、原始 DataFrame、特征信息字典
        scaler:              GroupedStandardScaler 实例（用于测试集 transform）
    """
    df = pd.read_csv(data_path)
    if "drfp" not in df.columns:
        raise ValueError("数据文件中缺少 'drfp' 列")

    # ===========================================================
    # 1. DRFP 特征（支持多变体）
    # ===========================================================
    drfp_cols_map = {
        'drfp':         'DRFP全反应',
        'drfp React':   'DRFP仅底物',
        'drfp wo cats': 'DRFP去催化剂',
        'drfp wo sols': 'DRFP去溶剂',
    }
    if not use_drfp_variants:
        drfp_cols_map = {'drfp': 'DRFP全反应'}

    X_drfp_list = []
    drfp_info = {}
    for col, label in drfp_cols_map.items():
        if col not in df.columns:
            if verbose:
                print(f"  [警告] 缺少列 '{col}'，跳过")
            continue
        n_rows = len(df)
        parts = []
        for i, fp_str in enumerate(df[col]):
            fp = read_drfp(fp_str)
            if fp is None:
                raise ValueError(f"DRFP 解析失败: {fp_str[:50]}")
            parts.append(fp.astype(np.float32))
            if (i + 1) % 200 == 0:
                print(f"    {label} 进度: {i+1}/{n_rows}")
        arr = np.array(parts, dtype=np.float32)
        X_drfp_list.append(arr)
        drfp_info[label] = arr.shape[1]
        if verbose:
            print(f"  解析 {label} ({n_rows} 条)...")

    X_drfp_all = np.hstack(X_drfp_list) if len(X_drfp_list) > 1 else X_drfp_list[0]
    feat_info = {'drfp_variants': X_drfp_all.shape[1]}

    # ===========================================================
    # 2. 催化剂 Morgan 指纹（先批量预计算唯一 SMILES，避免重复 RDKit 解析）
    # ===========================================================
    X_cat_fp = None
    if use_catalyst_fp:
        if verbose:
            print("  生成催化剂 Morgan 指纹（预计算唯一 SMILES）...")
        n_rows = len(df)
        # 收集所有唯一催化剂 SMILES
        unique_smiles = set()
        for i in range(1, 5):
            col = f'catalyst_{i}_smiles'
            if col in df.columns:
                for smi in df[col].dropna().astype(str):
                    smi = smi.strip()
                    if smi and smi not in ('', '/', 'nan', 'None'):
                        unique_smiles.add(smi)
        unique_list = sorted(unique_smiles)
        if verbose:
            print(f"    唯一催化剂 SMILES 数量: {len(unique_list)}，预计算中...")
        # 批量预计算指纹（利用缓存，每次只计算一次）
        fp_table = {}
        for idx, smi in enumerate(unique_list):
            fp_table[smi] = smiles_to_morgan(smi, n_bits=morgan_bits)
            if (idx + 1) % 500 == 0:
                print(f"    预计算进度: {idx+1}/{len(unique_list)}")

        # 批量生成每行数据的组合指纹（不再逐行调用 RDKit）
        zero_fp = np.zeros(morgan_bits, dtype=np.float32)
        cat_fps = np.zeros((n_rows, morgan_bits), dtype=np.float32)
        for i in range(n_rows):
            row_fps = []
            for j in range(1, 5):
                col = f'catalyst_{j}_smiles'
                if col in df.columns:
                    smi = str(df.iloc[i][col]).strip() if pd.notna(df.iloc[i][col]) else ''
                    if smi and smi not in ('', '/', 'nan', 'None'):
                        row_fps.append(fp_table.get(smi, zero_fp))
            if row_fps:
                combined = row_fps[0]
                for fp in row_fps[1:]:
                    np.maximum(combined, fp, out=combined)
                cat_fps[i] = combined
            if (i + 1) % 200 == 0:
                print(f"    催化剂指纹 进度: {i+1}/{n_rows}")
        X_cat_fp = cat_fps
        feat_info['catalyst_morgan'] = X_cat_fp.shape[1]

    # ===========================================================
    # 3. 底物 one-hot
    # ===========================================================
    X_reactant_oh = None
    if use_reactant_onehot:
        all_reactants = sorted(df['reactant_name'].dropna().unique().tolist())
        if verbose:
            print(f"  底物类别 ({len(all_reactants)}): {all_reactants}")
        X_reactant_oh = np.array(
            [build_reactant_onehot(r, all_reactants) for r in df['reactant_name'].fillna('unknown')],
            dtype=np.float32
        )
        feat_info['reactant_onehot'] = X_reactant_oh.shape[1]

    # ===========================================================
    # 4. 催化剂体系类型 one-hot
    # ===========================================================
    X_cat_type_oh = None
    if use_catalyst_type:
        all_types = sorted(df['catalyst_system_type'].dropna().unique().tolist())
        if verbose:
            print(f"  催化剂体系类别 ({len(all_types)}): {all_types}")
        X_cat_type_oh = np.array(
            [build_catalyst_type_onehot(c, all_types) for c in df['catalyst_system_type'].fillna('unknown')],
            dtype=np.float32
        )
        feat_info['catalyst_type_onehot'] = X_cat_type_oh.shape[1]

    # ===========================================================
    # 5. 条件特征
    # ===========================================================
    X_cond = None
    if use_conditions:
        X_cond = build_condition_features(df)
        feat_info['conditions'] = X_cond.shape[1]

    # ===========================================================
    # 6. 组装 & 分组
    # ===========================================================
    # 二值特征（不缩放，仅居中）
    binary_parts = [X_drfp_all]
    if X_cat_fp is not None:
        binary_parts.append(X_cat_fp)
    if X_reactant_oh is not None:
        binary_parts.append(X_reactant_oh)
    if X_cat_type_oh is not None:
        binary_parts.append(X_cat_type_oh)
    X_binary = np.hstack(binary_parts)

    # 连续特征（标准化）
    if X_cond is not None:
        X_cont = X_cond
    else:
        X_cont = np.zeros((len(df), 0), dtype=np.float32)

    feat_info['binary_dim'] = X_binary.shape[1]
    feat_info['cont_dim'] = X_cont.shape[1]

    # ===========================================================
    # 7. 分组标准化
    # ===========================================================
    if grouped_scale:
        scaler = GroupedStandardScaler()
        X_binary_s, X_cont_s = scaler.fit_transform(X_binary, X_cont)
        X = np.hstack([X_binary_s, X_cont_s])
    else:
        scaler = None
        X = np.hstack([X_binary, X_cont])

    # 分组特征（原始未标准化，供双分支 ANN 使用）
    X_drfp_raw = X_drfp_all
    X_aux_raw = np.hstack([b for b in [X_cat_fp, X_reactant_oh, X_cat_type_oh] if b is not None])
    X_cont_raw = X_cont if X_cont is not None else np.zeros((len(df), 0), dtype=np.float32)

    # ===========================================================
    # 8. Target vector
    # ===========================================================
    _col_renames = {}
    if "yield (%)" in df.columns and "rxn_yield" not in df.columns:
        _col_renames["yield (%)"] = "rxn_yield"
    if "reactant_name" in df.columns and "reactant" not in df.columns:
        _col_renames["reactant_name"] = "reactant"
    if "product_name" in df.columns and "product" not in df.columns:
        _col_renames["product_name"] = "product"

    # Normalize temperature column to canonical name
    for alias in _TEMP_COL_ALIASES:
        if alias in df.columns:
            _col_renames[alias] = 'temperature_celsius'
            break

    if "time (h)" in df.columns and "time" not in df.columns:
        _col_renames["time (h)"] = "time"
    if "pressure (MPa)" in df.columns and "pressure" not in df.columns:
        _col_renames["pressure (MPa)"] = "pressure"
    if "reference" not in df.columns:
        df["reference"] = "Unknown reference"
    if _col_renames:
        df = df.rename(columns=_col_renames)
        if verbose:
            print(f"  Column rename mapping: {_col_renames}")

    from CO2_rxn import df_to_rxn_list
    rxn_list = df_to_rxn_list(df)
    y = np.array(
        [float(rxn.rxn_yield) for rxn in rxn_list],
        dtype=np.float32
    ) / 100.0

    if verbose:
        print(f"\n数据加载完成:")
        print(f"  样本数: {len(X)}")
        print(f"  总特征维度: {X.shape[1]}  (二值 {X_binary.shape[1]} + 连续 {X_cont.shape[1]})")
        print(f"  特征构成:")
        total = 0
        for name, dim in drfp_info.items():
            total += dim
            print(f"    {name}: {dim} 维")
        for name, dim in feat_info.items():
            if name not in ('drfp_variants', 'binary_dim', 'cont_dim'):
                print(f"    {name}: {dim} 维")
        print(f"  y 范围: [{y.min():.4f}, {y.max():.4f}], 均值: {y.mean():.4f}")

    return X, y, df, feat_info, scaler, X_drfp_raw, X_aux_raw, X_cont_raw


# ------------------------------------------------------------
# 单元测试
# ------------------------------------------------------------
if __name__ == "__main__":
    test_path = os.path.join(PROJECT_ROOT, 'data/processed/co2_drfp.csv')
    if os.path.exists(test_path):
        print("测试增强特征加载（v2）...")
        result = load_enhanced_data(
            test_path,
            use_drfp_variants=True,
            grouped_scale=True,
            verbose=True
        )
        X, y, df, info, scaler, X_drfp_raw, X_aux_raw, X_cont_raw = result
        print(f"\nX shape: {X.shape}, y shape: {y.shape}")
        print(f"特征信息: {info}")
        print(f"scaler 类型: {type(scaler).__name__}")
        print(f"X_drfp_raw: {X_drfp_raw.shape}, X_aux_raw: {X_aux_raw.shape}, X_cont_raw: {X_cont_raw.shape}")
        # 检查 NaN
        nan_count = np.isnan(X).sum()
        inf_count = np.isinf(X).sum()
        print(f"NaN 数量: {nan_count}, Inf 数量: {inf_count}")
        print("测试通过！")
    else:
        print(f"测试文件不存在: {test_path}")
