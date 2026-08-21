# -*- coding: utf-8 -*-
"""
    (translated to English in upstream docstring)
"""
from collections import namedtuple
import os
import numpy as np
import pandas as pd

# Project root (consumed when inferring default paths for drfp_ablation_meta.json,
# which is produced by 201_ablation.py at PROJECT_ROOT/results_best_pipeline/).
PROJECT_ROOT = os.environ.get("CO2_PROJECT_ROOT", r"D:\machine-learning\CO2-cycloaddition")
# ------------------------------------------------------------
# DRFP 解析
# ------------------------------------------------------------
def read_drfp(fp_str) -> np.ndarray | None:
    """
        (translated to English in upstream docstring)
"""
    if fp_str is None or (isinstance(fp_str, float) and np.isnan(fp_str)):
        return None
    raw = str(fp_str).strip()
    if not raw.startswith("["):
        return None
    raw = raw.strip("[]")
    if not raw:
        return np.array([], dtype=np.int64)

    try:
        if " " in raw:
            parts = raw.split()
        else:
            parts = raw.split(",")
        parts = [p.strip() for p in parts if p.strip()]
        arr = np.array([int(p) for p in parts], dtype=np.int64)
        return arr
    except (ValueError, TypeError):
        return None


# ------------------------------------------------------------
# ------------------------------------------------------------
_RxnRecord = namedtuple("RxnRecord", [
    "rxn_yield",
    "reactant",
    "product",
    "catalyst_1_smiles",
    "catalyst_2_smiles",
    "catalyst_3_smiles",
    "catalyst_4_smiles",
    "solvent",
    "temperature",
    "pressure",
    "time",
    "reference",
])


def df_to_rxn_list(df: pd.DataFrame):
    """
    将 DataFrame 转换为命名元组列表。

    每行返回一个 namedtuple，字段名与 DataFrame 列名对齐，
    确保 CO2_features.py 中 rxn.rxn_yield 的访问生效。

    参数:
        df: 包含 rxn_yield 列的 DataFrame（列名与 co2_drfp.csv 对应）

    返回:
        list[RxnRecord] 或 list[namedtuple]
    """
    if df.empty:
        return []

    records = []
    n_rows = len(df)
    for idx, (_, row) in enumerate(df.iterrows()):
        if (idx + 1) % 200 == 0:
            print(f"    df_to_rxn_list 进度: {idx+1}/{n_rows}")
        d = {}
        for field in _RxnRecord._fields:
            if field in row.index:
                val = row[field]
                if val is None or (isinstance(val, float) and np.isnan(val)):
                    d[field] = None
                else:
                    d[field] = val
            else:
                d[field] = None

        # rxn_yield 必须是数值类型
        raw_yield = d.get("rxn_yield")
        if raw_yield is None:
            d["rxn_yield"] = 0.0
        else:
            try:
                d["rxn_yield"] = float(raw_yield)
            except (ValueError, TypeError):
                d["rxn_yield"] = 0.0

        records.append(_RxnRecord(**d))

    return records


# =============================================================================
# DRFP 变体消融结果读取工具
# 所有下游脚本统一调用此函数读取最优 DRFP 变体，禁止硬编码
# =============================================================================
import json as _json
import os as _os

_DRFP_ABLATON_META = None  # 模块级缓存


def get_best_drfp_variant(meta_path=None):
    """
    读取 DRFP 消融实验的元信息，返回最优变体。
    如果 meta 文件不存在，返回默认值 'no_cats' 并打印警告。

    参数:
        meta_path: str，可选。默认从 results_best_pipeline/drfp_ablation_meta.json 读取。

    返回:
        dict: {
            'best_variant': str,       # 'full' / 'reactants' / 'no_cats' / 'no_sols'
            'best_drfp_col': str,      # CSV 列名，如 'drfp wo cats'
            'best_r2': float,          # 最优 R²
            'all_variants': dict,     # 所有变体的 R²
        }
    """
    global _DRFP_ABLATON_META

    if meta_path is None:
        # Auto-infer: PROJECT_ROOT/results_best_pipeline/drfp_ablation_meta.json
        # (produced by 201_ablation.py; was co-located with utils_rxn.py
        # before the 2026-08-14 restructure moved utils_rxn.py into utils/).
        _default_path = _os.path.join(
            PROJECT_ROOT,
            'results_best_pipeline', 'drfp_ablation_meta.json'
        )
        meta_path = _default_path

    if _DRFP_ABLATON_META is not None:
        return _DRFP_ABLATON_META

    if _os.path.exists(meta_path):
        with open(meta_path, 'r', encoding='utf-8') as f:
            meta = _json.load(f)
        _DRFP_ABLATON_META = meta
        return meta
    else:
        # 文件不存在（尚未运行消融实验），返回默认值并警告
        import sys as _sys
        print(
            f"[WARNING] {meta_path} not found. "
            f"DRFP ablation has not been run yet. "
            f"Defaulting to 'no_cats'. "
            f"Run 201_ablation.py first to generate this file.",
            file=_sys.stderr
        )
        default_meta = {
            'best_variant': 'no_cats',
            'best_drfp_col': 'drfp wo cats',
            'best_r2': None,
            'best_label': 'DRFP去催化剂 (default fallback)',
            'all_variants': {},
        }
        _DRFP_ABLATON_META = default_meta
        return default_meta


# =============================================================================
# XTB 列名（与 co2_drfp_xtb_extended.csv 严格一致）
# =============================================================================
XTB_COLS = [
    # 底物
    'sub_homo_eV', 'sub_lumo_eV', 'sub_gap_eV', 'sub_dipole_D',
    # CO2
    'co2_homo_eV', 'co2_lumo_eV', 'co2_gap_eV',
    # 催化剂
    'cat_homo_eV', 'cat_lumo_eV', 'cat_gap_eV', 'cat_dipole_D',
    # 溶剂
    'solv_homo_eV', 'solv_lumo_eV', 'solv_gap_eV',
    # 衍生描述符
    'delta_E_hl_cat_sub', 'global_hardness', 'nucleophilicity_index',
    # IL 阴阳离子
    'cat_homo_eV_min', 'cat_lumo_eV_max', 'cat_gap_eV_min',
    'cat_cation_homo_eV', 'cat_cation_lumo_eV', 'cat_cation_gap_eV',
    'cat_anion_homo_eV', 'cat_anion_lumo_eV', 'cat_anion_gap_eV',
    'cat_cation_dipole_D', 'cat_anion_dipole_D',
    # 催化活性特征
    'activation_proxy', 'charge_transfer_potential', 'ion_pair_interaction',
    'electrophilicity_cat', 'electrodonating_cat',
    # 反应性特征
    'sub_cat_orbital_match', 'gap_ratio', 'hardness_ratio',
    'nucleophilicity_cat', 'reaction_polarity', 'co2_activation_proxy',
    # 溶剂相关
    'solv_cat_interaction', 'solv_sub_interaction',
    'total_polarity_index', 'dielectric_proxy',
]


def get_xtb_cols(df):
    """Return XTB columns that exist in DataFrame (intersection with XTB_COLS)."""
    return [c for c in XTB_COLS if c in df.columns]


# =============================================================================
# 全局随机种子（统一工具，保证复现性）
# =============================================================================
def set_global_seed(seed: int = 42) -> None:
    """
    设置全局随机种子：numpy / torch / Python random。
    各脚本 main() 开头调用一次即可。

    注：默认不开 CuDNN 确定性模式（会显著拖慢 GPU 训练）。如需严格复现，
    各脚本可在调用本函数后自行设置 torch.backends.cudnn.deterministic=True。
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# ------------------------------------------------------------
# ------------------------------------------------------------
if __name__ == "__main__":
    test_cases = [
        "[0 0 1 0 1]",
        "[0,0,1,0,1]",
        "[0, 0, 1, 0, 1]",
        "[ 0   1   0 ]",
    ]
    for tc in test_cases:
        result = read_drfp(tc)
        print(f"read_drfp({tc!r}) -> {result}")

    # DataFrame 测试
    sample_df = pd.DataFrame({
        "rxn_yield": ["85.2", "92.0", "NaN"],
        "reactant":  ["PO", "SO", "CHO"],
        "product":   ["PC", "SC", "CHC"],
    })
    rxns = df_to_rxn_list(sample_df)
    for r in rxns:
        print(f"  rxn_yield={r.rxn_yield}, reactant={r.reactant}")
