# -*- coding: utf-8 -*-
"""
    (translated to English in upstream docstring)
"""
import os
import sys
import numpy as np
import pandas as pd

PROJECT_ROOT = os.environ.get("CO2_PROJECT_ROOT", r"D:\machine-learning\CO2-cycloaddition")
sys.path.insert(0, PROJECT_ROOT)
from utils_rxn import read_drfp

# ================================================================
# ================================================================
_rdk_loaded = False
_RDKit = None
_AllChem = None
_Descriptors = None
_morgan_cache = {}

N_CATION_DESC  = 18
N_ANION_DESC   = 16
N_METAL_DESC   = 8
N_BASE_DESC    = 10
N_COMPONENT_TOTAL = N_CATION_DESC + N_ANION_DESC + N_METAL_DESC + N_BASE_DESC  # 48


# ================================================================
# Canonical experimental-condition column names (single source of truth).
# Used by every training/eval script so that the "conditions" block of the
# model input is identical across ablation, statistical test, sensitivity,
# and external validation -- preventing reviewer attacks on reproducibility.
#
# Following IUPAC/Cheminformatics standards for column naming:
# - snake_case for machine readability
# - explicit units (celsius, MPa, h)
# ================================================================
COND_COLS = ['temperature_celsius', 'pressure_MPa', 'time_h']

# Canonical names -> legacy aliases (for backward compatibility)
# Handles all historical column naming conventions including the erroneous
# 'temperature (\u2103)' (Celsius symbol ℉, not degree symbol °)
COND_COL_ALIASES = {
    'temperature_celsius': [
        'temperature_celsius',
        'temperature (\u2103)',    # erroneous Unicode (Celsius symbol)
        'temperature (\u00b0)',   # degree symbol
        'temperature (\u00b0C)', # degree Celsius
        'temperature (°C)',       # literal degree Celsius
        'temperature (℃)',       # fullwidth Celsius
        'temperature', 'temp', 'Temp', 'T (°C)', 'Temperature (°C)',
    ],
    'pressure_MPa': [
        'pressure_MPa',
        'pressure (MPa)',
    ],
    'time_h': [
        'time_h',
        'time (h)',
        'time',
    ],
}


def find_cond_cols(df, include_loading=False):
    """Return the subset of COND_COLS actually present in df.

    Parameters
    ----------
    df : pd.DataFrame
        Feature matrix.
    include_loading : bool, default False
        If True, also include any column whose name contains 'loading_mol%'
        (catalyst loading), useful for SSTS-style per-catalyst analyses.

    Returns
    -------
    list[str]
        List of canonical column names present in df.
    """
    cols = []
    for canonical in COND_COLS:
        for alias in COND_COL_ALIASES.get(canonical, [canonical]):
            if alias in df.columns:
                cols.append(canonical)  # Return canonical name
                break
    if include_loading:
        cols += [c for c in df.columns if 'loading_mol%' in c]
    return cols


def normalize_column_names(df: pd.DataFrame, inplace: bool = False) -> pd.DataFrame:
    """Normalize column names to canonical form for cross-version compatibility.

    This function ensures that all historical CSV files with varying temperature
    column names (°C, ℃, \u2103, etc.) are handled correctly.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with potentially non-canonical column names.
    inplace : bool, default False
        If True, modify df in place; otherwise return a copy.

    Returns
    -------
    pd.DataFrame
        DataFrame with normalized column names.
    """
    if not inplace:
        df = df.copy()

    rename_map = {}
    for col in df.columns:
        for canonical, aliases in COND_COL_ALIASES.items():
            if col in aliases:
                rename_map[col] = canonical
                break

    if rename_map:
        df = df.rename(columns=rename_map)

    return df


def _ensure_rdkit():
    global _rdk_loaded, _RDKit, _AllChem, _Descriptors
    if _rdk_loaded:
        return _RDKit is not None
    _rdk_loaded = True
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        from rdkit.Chem import Descriptors
        _RDKit = Chem
        _AllChem = AllChem
        _Descriptors = Descriptors
        return True
    except ImportError:
        print("  [警告] RDKit 未安装，催化剂指纹将用零向量代替")
        return False


# ------------------------------------------------------------
# Substrate-specific chemical features
# These features capture substrate properties that are critical for
# understanding CHO-specific behavior (e.g., pi-pi interactions, steric effects)
# ------------------------------------------------------------

_SUBSTRATE_FEAT_NAMES = [
    'has_aromatic_ring',
    'n_aromatic_rings',
    'n_aliphatic_rings',
    'molecular_weight',
    'logp',
    'tpsa',
    'n_rotatable_bonds',
    'n_hydrogen_bond_donors',
    'n_hydrogen_bond_acceptors',
    'n_heteroatoms',
    'fraction_csp3',
    'n_rings',
    'n_bridgehead_atoms',
    'n_spiro_atoms',
]


def _compute_substrate_features_from_smiles(smiles: str) -> dict:
    """
    Compute chemical descriptors for a substrate SMILES.

    These features are specifically selected to help distinguish CHO from other
    substrates and to explain the observed SHAP sign reversal.

    Args:
        smiles: SMILES string of the substrate

    Returns:
        Dictionary of feature names -> values
    """
    if not smiles or not isinstance(smiles, str):
        return {name: 0.0 for name in _SUBSTRATE_FEAT_NAMES}

    mol = _safe_mol(smiles)
    if mol is None:
        return {name: 0.0 for name in _SUBSTRATE_FEAT_NAMES}

    try:
        from rdkit.Chem import rdMolDescriptors
        from rdkit.Chem import Descriptors
        from rdkit.Chem import Crippen
    except ImportError:
        return {name: 0.0 for name in _SUBSTRATE_FEAT_NAMES}

    # Initialize features with zeros
    feats = {name: 0.0 for name in _SUBSTRATE_FEAT_NAMES}

    try:
        # Aromaticity features (critical for CHO vs terminal epoxides)
        aromatic_atoms = sum(1 for a in mol.GetAtoms() if a.GetIsAromatic())
        feats['has_aromatic_ring'] = 1.0 if aromatic_atoms > 0 else 0.0
        feats['n_aromatic_rings'] = mol.GetRingInfo().NumAromaticRings()

        # Ring features
        feats['n_aliphatic_rings'] = mol.GetRingInfo().NumRings() - feats['n_aromatic_rings']
        feats['n_rings'] = mol.GetRingInfo().NumRings()

        # Molecular descriptors
        feats['molecular_weight'] = Descriptors.MolWt(mol)
        feats['logp'] = Crippen.MolLogP(mol)
        feats['tpsa'] = Descriptors.TPSA(mol)

        # Bond flexibility
        feats['n_rotatable_bonds'] = rdMolDescriptors.CalcNumRotatableBonds(mol)

        # H-bonding capacity
        feats['n_hydrogen_bond_donors'] = Descriptors.NumHDonors(mol)
        feats['n_hydrogen_bond_acceptors'] = Descriptors.NumHAcceptors(mol)

        # Composition
        n_atoms = mol.GetNumAtoms()
        n_c = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 6)
        feats['fraction_csp3'] = Descriptors.FractionCSP3(mol)

        # Heteroatom count
        feats['n_heteroatoms'] = sum(1 for a in mol.GetAtoms()
                                      if a.GetAtomicNum() not in (1, 6))

        # Advanced topology
        feats['n_bridgehead_atoms'] = rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
        feats['n_spiro_atoms'] = rdMolDescriptors.CalcNumSpiroAtoms(mol)

    except Exception:
        # Return zeros on any error
        pass

    return feats


# Cache for substrate features
_SUBSTRATE_FEAT_CACHE = {}


def build_substrate_features(df: pd.DataFrame, substrate_col: str = 'substrate_smiles',
                              verbose: bool = True) -> pd.DataFrame:
    """
    Add substrate-specific chemical features to the DataFrame.

    These features capture properties that help distinguish CHO (cyclohexene oxide)
    from other substrates and may explain the observed SHAP sign reversal for
    sub_homo_eV.

    Features include:
    - Aromaticity (critical for CHO's unique behavior)
    - LogP (lipophilicity, affects catalyst binding)
    - TPSA (topological polar surface area)
    - Rotatable bonds (flexibility)
    - Ring counts (structural complexity)

    Args:
        df: Input DataFrame
        substrate_col: Column name containing substrate SMILES
        verbose: Print progress messages

    Returns:
        DataFrame with additional substrate feature columns
    """
    global _SUBSTRATE_FEAT_CACHE

    if substrate_col not in df.columns:
        if verbose:
            print(f"  [警告] Substrate column '{substrate_col}' not found. Skipping substrate features.")
        return df

    df = df.copy()

    # Compute features for each unique substrate
    unique_smiles = df[substrate_col].dropna().unique()
    for smiles in unique_smiles:
        if smiles not in _SUBSTRATE_FEAT_CACHE:
            _SUBSTRATE_FEAT_CACHE[smiles] = _compute_substrate_features_from_smiles(smiles)

    # Add features to DataFrame
    for feat_name in _SUBSTRATE_FEAT_NAMES:
        df[feat_name] = df[substrate_col].map(
            lambda s: _SUBSTRATE_FEAT_CACHE.get(s, {}).get(feat_name, 0.0)
        )

    if verbose:
        print(f"  Added {len(_SUBSTRATE_FEAT_NAMES)} substrate-specific features:")
        for name in _SUBSTRATE_FEAT_NAMES:
            print(f"    - {name}")

    return df


def get_substrate_feature_matrix(df: pd.DataFrame, substrate_col: str = 'substrate_smiles') -> np.ndarray:
    """
    Get substrate feature matrix as numpy array.

    Args:
        df: DataFrame with substrate features (from build_substrate_features)
        substrate_col: Original substrate column name

    Returns:
        numpy array of shape (n_samples, n_features)
    """
    available_feats = [f for f in _SUBSTRATE_FEAT_NAMES if f in df.columns]
    if not available_feats:
        return np.zeros((len(df), 0), dtype=np.float32)
    return df[available_feats].fillna(0.0).values.astype(np.float32)


# Substrate mechanism grouping for chemical interpretation
SUBSTRATE_MECHANISM_GROUPS = {
    'internal_epoxides': ['Cyclohexene oxide', 'Cyclopentene oxide', 'Cyclooctene oxide'],
    'terminal_epoxides': [
        'Epichlorohydrin', 'Isopropyl glycidyl ether', 'Propylene oxide',
        'Styrene oxide', '1,2-epoxy-3-phenoxypropane', 'Glycidyl ether',
        'Epoxymethylstyrene', 'Allyl glycidyl ether',
    ],
    'unknown': [],
}


def get_substrate_group(substrate_name: str) -> str:
    """Get the mechanism group for a substrate."""
    for group, substrates in SUBSTRATE_MECHANISM_GROUPS.items():
        if substrate_name in substrates:
            return group
    return 'unknown'


def build_mechanism_group_features(df: pd.DataFrame, substrate_col: str = 'substrate') -> pd.DataFrame:
    """
    Add mechanism group one-hot encoding to DataFrame.

    This helps the model distinguish between:
    - Internal epoxides (CHO, etc.) - unique mechanism
    - Terminal epoxides (ECH, PO, SO, etc.) - standard mechanism

    Args:
        df: Input DataFrame
        substrate_col: Column name containing substrate names

    Returns:
        DataFrame with mechanism group columns added
    """
    df = df.copy()

    if substrate_col not in df.columns:
        return df

    groups = list(SUBSTRATE_MECHANISM_GROUPS.keys())

    for group in groups:
        col_name = f'is_{group}'
        df[col_name] = df[substrate_col].apply(
            lambda s: 1.0 if get_substrate_group(s) == group else 0.0
        )

    # Add a flag for CHO specifically (since it shows unique SHAP behavior)
    df['is_cho'] = df[substrate_col].apply(
        lambda s: 1.0 if 'cyclohexene' in str(s).lower() else 0.0
    )

    return df


def _safe_mol(smiles):
    if not smiles or not isinstance(smiles, str):
        return None
    s = smiles.strip()
    if s in ('', 'nan', 'None', '/'):
        return None
    if not _ensure_rdkit():
        return None
    try:
        return _RDKit.MolFromSmiles(s)
    except Exception:
        return None


def _has_substruct(mol, smarts):
    if mol is None:
        return False
    try:
        pat = _RDKit.MolFromSmarts(smarts)
        if pat is None:
            return False
        return mol.HasSubstructMatch(pat)
    except Exception:
        return False


def smiles_to_morgan(smiles_str, n_bits=256, radius=2):
    if not smiles_str or not isinstance(smiles_str, str):
        return np.zeros(n_bits, dtype=np.float32)
    s = str(smiles_str).strip()
    if s in ('', '/', 'nan', 'None'):
        return np.zeros(n_bits, dtype=np.float32)
    key = (s, n_bits, radius)
    if key in _morgan_cache:
        return _morgan_cache[key].copy()
    mol = _safe_mol(s)
    if mol is None:
        return np.zeros(n_bits, dtype=np.float32)
    try:
        fp = _AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
        result = np.array(fp, dtype=np.float32)
        _morgan_cache[key] = result
        return result.copy()
    except Exception:
        return np.zeros(n_bits, dtype=np.float32)


def build_combined_catalyst_fp(row, n_bits=256):
    cat_list = []
    for i in range(1, 5):
        col = f'catalyst_{i}_smiles'
        if col in row.index and pd.notna(row.get(col)):
            smi = str(row[col]).strip()
            if smi and smi not in ('', '/', 'nan'):
                cat_list.append(smi)
    if not cat_list:
        return np.zeros(n_bits, dtype=np.float32)
    if len(cat_list) == 1:
        return smiles_to_morgan(cat_list[0], n_bits)
    fps = [smiles_to_morgan(s, n_bits) for s in cat_list]
    combined = fps[0]
    for fp in fps[1:]:
        np.maximum(combined, fp, out=combined)
    return combined


# ================================================================
# ================================================================
def _atom_counts(mol):
    if mol is None:
        return {'C': 0, 'N': 0, 'O': 0, 'hal': 0, 'hetero': 0,
                'ring': 0, 'heavy': 0, 'N_pos': 0, 'aromatic': False, 'N_in_ring': False}
    try:
        atoms = list(mol.GetAtoms())
        C    = sum(1 for a in atoms if a.GetAtomicNum() == 6)
        N    = sum(1 for a in atoms if a.GetAtomicNum() == 7)
        O    = sum(1 for a in atoms if a.GetAtomicNum() == 8)
        hal  = sum(1 for a in atoms if a.GetAtomicNum() in (9, 17, 35, 53))
        het  = sum(1 for a in atoms if a.GetAtomicNum() not in (1, 6))
        Npos = sum(1 for a in atoms if a.GetAtomicNum() == 7 and a.GetFormalCharge() > 0)
        heavy = sum(1 for a in atoms if a.GetAtomicNum() > 1)
        ring = mol.GetRingInfo().NumRings()
        arom = any(b.GetIsAromatic() for b in mol.GetBonds())
        n_in_ring = False
        for ring_bonds in mol.GetRingInfo().BondRings():
            for b_idx in ring_bonds:
                b = mol.GetBondWithIdx(b_idx)
                for a in [b.GetBeginAtom(), b.GetEndAtom()]:
                    if a.GetAtomicNum() == 7:
                        n_in_ring = True
        return {'C': C, 'N': N, 'O': O, 'hal': hal, 'hetero': het,
                'ring': ring, 'heavy': heavy, 'N_pos': Npos,
                'aromatic': arom, 'N_in_ring': n_in_ring}
    except Exception:
        return {'C': 0, 'N': 0, 'O': 0, 'hal': 0, 'hetero': 0,
                'ring': 0, 'heavy': 0, 'N_pos': 0, 'aromatic': False, 'N_in_ring': False}


def _extract_cation_features(smiles):
    """
        (translated to English in upstream docstring)
"""
    if not smiles:
        return np.zeros(N_CATION_DESC, dtype=np.float32)
    mol = _safe_mol(smiles)

    is_imidazolium = 0.0
    is_pyridinium  = 0.0
    is_ammonium    = 0.0
    is_phosphonium = 0.0
    is_sulfonium   = 0.0
    if mol is not None:
        if _has_substruct(mol, '[nH+]1ccnc1') or _has_substruct(mol, '[n+]1c[nH]cc1'):
            is_imidazolium = 1.0
        elif _has_substruct(mol, 'c1ccc[n+]1') or _has_substruct(mol, '[n+]1ccccc1'):
            is_pyridinium = 1.0
        elif _has_substruct(mol, '[N+]([C])([C])([C])[C]') or _has_substruct(mol, 'N+(C)(C)C'):
            is_ammonium = 1.0
        elif _has_substruct(mol, '[P+]([C])([C])([C])[C]'):
            is_phosphonium = 1.0
        elif _has_substruct(mol, '[S+]([C])([C])[C]'):
            is_sulfonium = 1.0

    counts = _atom_counts(mol)
    mol_wt = 0.0
    if mol is not None and _Descriptors is not None:
        try:
            mol_wt = _Descriptors.MolWt(mol)
        except Exception:
            pass

    # 18 维：5(type) + 13(数值) = 18
    return np.array([
        is_imidazolium, is_pyridinium, is_ammonium, is_phosphonium, is_sulfonium,
        min(counts['C']   / 30.0, 1.0),
        min(counts['N']   / 10.0, 1.0),
        min(counts['O']   /  5.0, 1.0),
        min(counts['hal'] /  5.0, 1.0),
        min(counts['ring'] / 5.0, 1.0),
        float(counts['aromatic']),
        float(counts['N_in_ring']),
        min(counts['C'] / 20.0, 1.0),
        min(counts['N'] * 3 / 12.0, 1.0),
        min(mol_wt / 500.0, 1.0),
        min(counts['N_pos'] / 3.0, 1.0),
        min(counts['hetero'] / 20.0, 1.0),
        0.0,   # padding
    ], dtype=np.float32)


def _extract_anion_features(smiles):
    """
        (translated to English in upstream docstring)
"""
    if not smiles:
        return np.zeros(N_ANION_DESC, dtype=np.float32)
    mol = _safe_mol(smiles)
    smi_lower = (smiles or '').lower()

    is_halide    = 0.0
    is_BF4       = 0.0
    is_PF6       = 0.0
    is_bistrif   = 0.0
    is_acetate   = 0.0
    is_phosphate = 0.0

    if mol is not None:
        if (_has_substruct(mol, '[F-]') or _has_substruct(mol, '[Cl-]') or
                _has_substruct(mol, '[Br-]') or _has_substruct(mol, '[I-]')):
            is_halide = 1.0
        if _has_substruct(mol, 'F[B-](F)(F)F') or 'B(F)(F)(F)' in smiles:
            is_BF4 = 1.0
        if _has_substruct(mol, 'F[P-](F)(F)(F)(F)F') or 'P(F)(F)(F)' in smiles:
            is_PF6 = 1.0
        if 'N(S(=O)(=O)C(F)(F)F)S(=O)(=O)C(F)(F)F' in smiles:
            is_bistrif = 1.0
        if _has_substruct(mol, 'CC(=O)[O-]'):
            is_acetate = 1.0
        counts = _atom_counts(mol)
        if counts['N'] > 0 and counts['hal'] >= 4:
            is_phosphate = 1.0
    else:
        if smiles in ('[Cl-]', '[Br-]', '[I-]', '[F-]'):
            is_halide = 1.0
        elif 'B(F)(F)(F)' in smiles:
            is_BF4 = 1.0
        elif 'P(F)(F)(F)' in smiles:
            is_PF6 = 1.0

    n_fluorine = 0
    n_other_hal = 0
    has_S = 0.0
    has_P = 0.0
    mol_wt = 0.0
    n_heavy = 0

    if mol is not None:
        try:
            atoms = list(mol.GetAtoms())
            n_fluorine = sum(1 for a in atoms if a.GetAtomicNum() == 9)
            n_other_hal = sum(1 for a in atoms if a.GetAtomicNum() in (17, 35, 53))
            has_S = 1.0 if any(a.GetAtomicNum() == 16 for a in atoms) else 0.0
            has_P = 1.0 if any(a.GetAtomicNum() == 15 for a in atoms) else 0.0
            n_heavy = sum(1 for a in atoms if a.GetAtomicNum() > 1)
            if _Descriptors is not None:
                mol_wt = _Descriptors.MolWt(mol)
        except Exception:
            pass

    return np.array([
        is_halide, is_BF4, is_PF6, is_bistrif, is_acetate, is_phosphate,
        min(n_fluorine  / 6.0, 1.0),
        min(n_other_hal / 3.0, 1.0),
        has_S, has_P,
        min(mol_wt  / 300.0, 1.0),
        min(n_heavy / 20.0, 1.0),
    ], dtype=np.float32)


def _extract_metal_halide_features(smiles):
    """
        (translated to English in upstream docstring)
"""
    if not smiles:
        return np.zeros(N_METAL_DESC, dtype=np.float32)
    smi_lower = (smiles or '').lower()

    has_zn = 1.0 if ('[zn' in smi_lower or 'zn+' in smi_lower) else 0.0
    has_ni = 1.0 if ('[ni' in smi_lower or 'ni+' in smi_lower) else 0.0

    halide_type = 0
    if '[f-]' in smi_lower:    halide_type = 1
    elif '[cl-]' in smi_lower: halide_type = 2
    elif '[br-]' in smi_lower: halide_type = 3
    elif '[i-]' in smi_lower:  halide_type = 4

    return np.array([0.0, 0.0, 0.0, 0.0, has_zn, has_ni,
                     halide_type / 4.0, 0.0], dtype=np.float32)


def _extract_organic_base_features(smiles):
    """
        (translated to English in upstream docstring)
"""
    if not smiles:
        return np.zeros(N_BASE_DESC, dtype=np.float32)
    mol = _safe_mol(smiles)

    is_aliphatic  = 0.0
    is_pyridine   = 0.0
    is_imidazole  = 0.0
    is_piperidine = 0.0
    is_dbu        = 0.0
    has_tertiary  = 0.0
    has_secondary = 0.0
    n_nitrogen = 0
    n_carbon   = 0
    mol_wt     = 0.0

    if mol is not None:
        if _has_substruct(mol, 'N(C)(C)C') or _has_substruct(mol, 'CCNCC'):
            is_aliphatic = 1.0
        if _has_substruct(mol, 'c1ccc[nH]1') or _has_substruct(mol, 'c1ccncc1'):
            is_pyridine = 1.0
        if _has_substruct(mol, '[nH]1cnc[n+]') or 'ccnc' in smiles.lower():
            is_imidazole = 1.0
        if _has_substruct(mol, 'C1CCNCC1'):
            is_piperidine = 1.0
        counts = _atom_counts(mol)
        n_nitrogen = counts['N']
        n_carbon   = counts['C']
        if n_nitrogen >= 2 and n_carbon >= 8:
            is_dbu = 1.0
        try:
            for a in mol.GetAtoms():
                if a.GetAtomicNum() == 7:
                    deg = a.GetTotalDegree()
                    if deg == 4:    has_tertiary = 1.0
                    elif deg == 3: has_secondary = 1.0
            if _Descriptors is not None:
                mol_wt = _Descriptors.MolWt(mol)
        except Exception:
            pass
    else:
        sl = smiles.lower()
        if sl.startswith('c') and 'n' in sl and ('ccc' in sl or 'cccc' in sl):
            is_pyridine = 1.0
        elif '[nh' in sl or 'n1c' in sl:
            is_imidazole = 1.0
        elif not is_pyridine and not is_imidazole and 'n' in sl:
            is_aliphatic = 1.0
        n_nitrogen = sl.count('n')
        n_carbon   = sl.count('c')

    return np.array([
        is_aliphatic, is_pyridine, is_imidazole, is_piperidine, is_dbu,
        min(n_nitrogen / 5.0, 1.0),
        min(n_carbon   / 20.0, 1.0),
        has_tertiary,
        has_secondary,
        min(mol_wt / 300.0, 1.0),
    ], dtype=np.float32)


def _parse_catalyst_smiles(smiles):
    """
        (translated to English in upstream docstring)
"""
    if not smiles:
        return {'type': 'unknown', 'components': {'cation': None, 'anion': None,
                   'metal': None, 'halide': None, 'organic': None}}
    sl = smiles.lower()
    parts = [p.strip() for p in smiles.split('.')
             if p.strip() and p.strip() not in ('nan', 'None', '/')]

    has_metal   = any(k in sl for k in ['[zn', '[ni', '[co', '[fe', '[cu',
                                          '[mg', '[ca', '[mn', '[li', '[na', '[k+'])
    has_halide  = any(k in sl for k in ['[cl-]', '[br-]', '[i-]', '[f-]'])
    has_il_cat  = any(k in sl for k in ['[nh+]', '[n+]', '[nh]'])

    if has_metal and has_halide:
        cat_type = 'metal_halide'
    elif has_il_cat or ('[N+' in smiles and ('cccc' in sl or 'ccc' in sl)):
        cat_type = 'ionic_liquid'
    elif not has_metal:
        cat_type = 'organic_base'
    else:
        cat_type = 'unknown'

    cation_smi = None; anion_smi = None; metal_smi = None
    halide_smi = None; organic_smi = None

    for p in parts:
        pl = p.lower()
        if ('[nh+]' in pl or '[n+]' in pl or '[nh]' in pl or
                ('[N+' in p and (('cccc' in pl or 'ccc' in pl) and p.count('c') > 2))):
            cation_smi = p
        elif '[cl-]' in pl or '[br-]' in pl or '[i-]' in pl or '[f-]' in pl:
            halide_smi = p
        elif '[bf' in pl or '[pf' in pl or 'n(s(o)(=o)' in pl or pl.count('f') >= 3:
            anion_smi = p
        elif any(k in pl for k in ['[zn', '[ni', '[co', '[fe', '[cu', '[mg', '[na', '[k+']):
            metal_smi = p
        else:
            organic_smi = p

    return {'type': cat_type,
            'components': {'cation': cation_smi, 'anion': anion_smi,
                           'metal': metal_smi, 'halide': halide_smi, 'organic': organic_smi}}


# ================================================================
# Step 2: IL 阳离子细粒度检测（7类，替换粗粒度 catalyst_system_type）
# ================================================================
IL_CATION_SUBTYPES = [
    'imidazolium', 'pyridinium', 'ammonium',
    'phosphonium', 'sulfonium', 'guanidinium', 'other_il',
]


def _detect_cation_subtype(smiles):
    """
        (translated to English in upstream docstring)
"""
    if not smiles or not isinstance(smiles, str):
        return 'other_il'
    smi = smiles.strip()
    if not smi or smi in ('', '/', 'nan', 'None'):
        return 'other_il'
    mol = _safe_mol(smi)

    # ---- RDKit 子结构检测（用于干净 SMILES）----
    if mol is not None:
        if _has_substruct(mol, '[nH+]1ccnc1') or _has_substruct(mol, '[n+]1c[nH]cc1'):
            return 'imidazolium'
        if _has_substruct(mol, 'c1ccc[n+]1') or _has_substruct(mol, '[n+]1ccccc1'):
            return 'pyridinium'
        if _has_substruct(mol, '[N+]([C])([C])([C])[C]'):
            return 'ammonium'
        if _has_substruct(mol, '[P+]([C])([C])([C])[C]'):
            return 'phosphonium'
        if _has_substruct(mol, '[S+]([C])([C])[C]'):
            return 'sulfonium'
        # guanidinium: C(=N)N + NC(=N) 双键检测
        if _has_substruct(mol, 'C(=N)N') and _has_substruct(mol, 'NC(=N)'):
            return 'guanidinium'

    smi_l = smi.lower()
    sl_part = smi_l.split('.')[0]  # 只取第一段（主体）
    if ('imidazol' in smi_l or 'n1ccnc' in sl_part or
            'ccnc1' in sl_part and 'n1' in sl_part or
            '[nh+]1ccnc' in smi_l or 'n+1c[nh]c' in smi_l):
        return 'imidazolium'
    if 'n+(c)(c)(c)c' in smi_l or '[n+](c)(c)(c)c' in smi_l:
        return 'ammonium'
    if 'pyridin' in smi_l or 'c1ccc[n+]1' in smi_l or 'n+1ccccc1' in smi_l:
        return 'pyridinium'
    if 'phosphon' in smi_l or '[p+](c)(c)(c)c' in smi_l:
        return 'phosphonium'
    if 'guanidin' in smi_l or 'tmg' in smi_l:
        return 'guanidinium'
    return 'other_il'


def build_il_subtype_features(df):
    """
    为 DataFrame 生成 IL 阳离子 7分类 one-hot 特征（7维）。
    仅当 catalyst_system_type == 'ionic_liquid' 时推断亚类；其余类型填 'other_il'。
    """
    n = len(df)
    X = np.zeros((n, len(IL_CATION_SUBTYPES)), dtype=np.float32)
    type_map = {t: i for i, t in enumerate(IL_CATION_SUBTYPES)}
    for i in range(n):
        row = df.iloc[i]
        cat_type = str(row.get('catalyst_system_type', '')).strip()
        subtype = 'other_il'
        if cat_type == 'ionic_liquid':
            # 优先用 cation_smi；若为空则回退到 catalyst_1_smiles
            parsed = _parse_catalyst_smiles(str(row.get('catalyst_1_smiles', '')) if pd.notna(row.get('catalyst_1_smiles')) else '')
            cation_smi = parsed['components'].get('cation')
            subtype = _detect_cation_subtype(cation_smi)
        idx = type_map.get(subtype, -1)
        if idx >= 0:
            X[i, idx] = 1.0
    return X


def build_catalyst_component_features(df, verbose=True):
    """
        (translated to English in upstream docstring)
"""
    n_rows = len(df)
    n_slots = 4
    feat_per_slot = N_COMPONENT_TOTAL
    X_comp = np.zeros((n_rows, n_slots * feat_per_slot), dtype=np.float32)

    type_counts = {'ionic_liquid': 0, 'metal_halide': 0, 'organic_base': 0,
                   'unknown': 0, 'none': 0}

    for i in range(n_rows):
        for slot in range(n_slots):
            col = f'catalyst_{slot + 1}_smiles'
            smiles = None
            if col in df.columns:
                raw = df.iloc[i][col]
                if pd.notna(raw):
                    smiles = str(raw).strip()

            if not smiles or smiles in ('', '/', 'nan', 'None'):
                type_counts['none'] += 1
                continue

            parsed = _parse_catalyst_smiles(smiles)
            cat_type = parsed['type']
            comps = parsed['components']
            type_counts[cat_type] = type_counts.get(cat_type, 0) + 1
            offset = slot * feat_per_slot

            if cat_type == 'ionic_liquid':
                cation_feat = _extract_cation_features(comps['cation'])
                X_comp[i, offset:offset + N_CATION_DESC] = cation_feat
                anion_feat = _extract_anion_features(comps['anion'])
                X_comp[i, offset + N_CATION_DESC:
                        offset + N_CATION_DESC + N_ANION_DESC] = anion_feat
            elif cat_type == 'metal_halide':
                metal_feat = _extract_metal_halide_features(comps['metal'] or smiles)
                X_comp[i, offset:offset + N_METAL_DESC] = metal_feat
            elif cat_type == 'organic_base':
                base_feat = _extract_organic_base_features(comps['organic'] or smiles)
                X_comp[i, offset:offset + N_BASE_DESC] = base_feat

        if verbose and (i + 1) % 500 == 0:
            print(f"    组分特征构建进度: {i+1}/{n_rows}")

    if verbose:
        print("  组分感知特征构建完成:")
        for k, v in sorted(type_counts.items()):
            print(f"    {k}: {v}条")

    return X_comp


# ================================================================
# ================================================================
class GroupedStandardScaler:
    def fit(self, X_binary, X_cont):
        self.binary_mean = np.mean(X_binary, axis=0).astype(np.float32)
        from sklearn.preprocessing import StandardScaler
        self.cont_scaler = StandardScaler()
        self.cont_scaler.fit(X_cont)

    def transform(self, X_binary, X_cont):
        Xb = X_binary - self.binary_mean
        Xc = self.cont_scaler.transform(X_cont)
        return Xb.astype(np.float32), Xc.astype(np.float32)

    def fit_transform(self, X_binary, X_cont):
        self.fit(X_binary, X_cont)
        return self.transform(X_binary, X_cont)


# ================================================================
# One-hot 编码
# ================================================================
def build_onehot(val, categories):
    n = len(categories)
    vec = np.zeros(n, dtype=np.float32)
    try:
        vec[categories.index(val)] = 1.0
    except (ValueError, TypeError):
        pass
    return vec


# ================================================================
# ================================================================
def build_condition_features(df):
    """Build condition feature columns: temperature, pressure, time(log), catalyst loading(log).

    Uses canonical column names from COND_COLS for cross-version compatibility.
    Automatically detects temperature column regardless of naming convention
    (°C, ℃, \u2103, etc.).
    """
    # Auto-detect temperature column using aliases
    temp_col = None
    for alias in COND_COL_ALIASES['temperature_celsius']:
        if alias in df.columns:
            temp_col = alias
            break
    if temp_col is None:
        # Collect available temperature-like columns for error message
        temp_candidates = [c for c in df.columns
                         if 'temp' in c.lower() or '°' in c or chr(0x2103) in c]
        raise KeyError(
            f"Cannot find temperature column. Tried: {COND_COL_ALIASES['temperature_celsius']}. "
            f"Available temperature-like columns: {temp_candidates}"
        )

    temp = df[temp_col].fillna(df[temp_col].median()).values.reshape(-1, 1)

    # Pressure: detect using aliases
    pressure_col = None
    for alias in COND_COL_ALIASES['pressure_MPa']:
        if alias in df.columns:
            pressure_col = alias
            break
    if pressure_col is None:
        raise KeyError(f"Cannot find pressure column. Tried: {COND_COL_ALIASES['pressure_MPa']}")
    pressure = df[pressure_col].fillna(df[pressure_col].median()).values.reshape(-1, 1)

    # Time: detect using aliases
    time_col = None
    for alias in COND_COL_ALIASES['time_h']:
        if alias in df.columns:
            time_col = alias
            break
    if time_col is None:
        raise KeyError(f"Cannot find time column. Tried: {COND_COL_ALIASES['time_h']}")
    time_val = df[time_col].fillna(df[time_col].median()).values.reshape(-1, 1)
    time_log = np.log1p(np.maximum(time_val, 0))

    # Catalyst total loading (mol%)
    loading_cols = [f'catalyst_{i}_loading_mol%' for i in range(1, 5)]
    loadings = np.zeros((len(df), 1), dtype=np.float32)
    for lc in loading_cols:
        if lc in df.columns:
            vals = pd.to_numeric(df[lc], errors='coerce').fillna(0).values.reshape(-1, 1)
            loadings = loadings + np.nan_to_num(vals, nan=0.0)
    loading_log = np.log1p(np.maximum(loadings, 0))

    return np.hstack([temp, pressure, time_log, loading_log]).astype(np.float32)


# ================================================================
# ================================================================
_MORDED_CACHE = {}
_MORDED_FEAT_KEYS = None
_MORDED_CALC = None


def _init_mordred():
    """
        (translated to English in upstream docstring)
"""
    global _MORDED_CALC
    if _MORDED_CALC is not None:
        return _MORDED_CALC
    from mordred import Calculator, descriptors as mordred_descs
    NEEDED = [
        'nAtom', 'nHeavyAtom', 'nH', 'nB', 'nC', 'nN', 'nO', 'nS', 'nP',
        'nF', 'nCl', 'nBr', 'nI',
        'nRing', 'n3Ring', 'n4Ring', 'n5Ring', 'n6Ring',
        'nAromAtom', 'nAromBond',
        'nSpiro', 'nBridgehead', 'nHetero',
        'nBase', 'nAcid', 'nHBAcc', 'nHBDon',
        'nRotable', 'LogEE_A', 'TopoPSA',
    ]
    all_descs = Calculator(mordred_descs, ignore_3D=True)
    filtered = [d for d in all_descs.descriptors if str(d) in NEEDED]
    _MORDED_CALC = Calculator(filtered, ignore_3D=True)
    global _MORDED_FEAT_KEYS
    _MORDED_FEAT_KEYS = sorted(NEEDED)
    return _MORDED_CALC


def _extract_cation_simple(smi):
    """
        (translated to English in upstream docstring)
"""
    if not smi or not isinstance(smi, str):
        return None
    smi = smi.strip()
    if not smi or smi.lower() in ('', 'nan', 'none', '/'):
        return None
    parts = smi.split('.')
    for p in parts:
        if '+' in p:
            return p.strip()
    return parts[0].strip()


# ================================================================
# ================================================================
# 论文 "Machine learning for the yield prediction of CO2 cyclization
# reaction catalyzed by the ionic liquids" (Fuel 335, 2023) Table 1:
#   - 4 个反应条件 (T, P, t, catalyst loading) → 已在 build_condition_features
#   - 7 个分子结构: NH, NC, NN, NO, NTol, Nheter, Nring
#   - 15 个电子/DFT 性质: HOMO, LUMO, Dipole, 亲核/亲电指数, 能量差等
#
#   NH  → nH,   NC → nC,   NN → nN,   NO → nO,   NTol → nAtom,
#   Nheter → nHetero, Nring → nRing
#   额外扩展: nHeavy, nAromAtom, nAromBond, nSpiro, nBridgehead,
#             nHBAcc, nHBDon, TopoPSA, nRotable, nBase, nAcid
#   IL阳离子7类one-hot叠加
#
# ================================================================

_PAPER_26_CACHE = {}
_PAPER_26_CALC = None
_PAPER_26_FEAT_KEYS = [
    'nH', 'nC', 'nN', 'nO', 'nAtom', 'nHetero', 'nRing',
    'nHeavyAtom', 'nAromAtom', 'nAromBond', 'nSpiro', 'nBridgehead',
    'nHBAcc', 'nHBDon', 'TopoPSA', 'nRotable', 'nBase', 'nAcid',
]


def _init_paper_26_calc():
    """初始化 Mordred Calculator（25 维，nRotable 单独处理）。"""
    global _PAPER_26_CALC
    if _PAPER_26_CALC is not None:
        return _PAPER_26_CALC
    from mordred import Calculator, descriptors as mordred_descs
    all_calc = Calculator(mordred_descs, ignore_3D=True)
    available = [str(d) for d in all_calc.descriptors]
    needed = [k for k in _PAPER_26_FEAT_KEYS if k in available]
    if 'nRotable' not in needed:
        # nRotable 不在 Mordred 中，用 rdKit 替代
        needed = [k for k in needed if k != 'nRotable']
    filtered = [d for d in all_calc.descriptors if str(d) in needed]
    _PAPER_26_CALC = Calculator(filtered, ignore_3D=True)
    return _PAPER_26_CALC


def _extract_cation_simple(smi):
    """
        (translated to English in upstream docstring)
"""
    if not smi or not isinstance(smi, str):
        return None
    smi = smi.strip()
    if not smi or smi.lower() in ('', 'nan', 'none', '/'):
        return None
    parts = smi.split('.')
    for p in parts:
        if '+' in p:
            return p.strip()
    return parts[0].strip()


def _compute_paper_26_for_cation(cation_smi):
    """
        (translated to English in upstream docstring)
"""
    if not cation_smi:
        return None
    mol = _safe_mol(cation_smi)
    if mol is None:
        return None

    calc = _init_paper_26_calc()
    try:
        result = calc(mol)
        vals = {}
        for desc, val in zip(calc.descriptors, result):
            try:
                v = float(val) if val is not None else np.nan
            except (ValueError, TypeError):
                v = np.nan
            vals[str(desc)] = v
    except Exception:
        return None

    # nRotable: Mordred 没有，用 rdKit 补
    if _RDKit is not None:
        try:
            from rdkit.Chem import rdMolDescriptors
            vals['nRotable'] = float(rdMolDescriptors.CalcNumRotatableBonds(mol))
        except Exception:
            vals['nRotable'] = 0.0

    return vals


def _compute_catalyst_mordred(df, verbose=True):
    """
        (translated to English in upstream docstring)
"""
    calc = _init_mordred()
    cache_path = os.path.join(
        PROJECT_ROOT, '_mordred_catalyst_cache.npy'
    )

    cache = {}
    if os.path.exists(cache_path):
        try:
            cache = np.load(cache_path, allow_pickle=True).item()
        except Exception:
            cache = {}

    rows = []
    valid_idx = []
    n_new = 0
    for i in range(len(df)):
        raw = df.iloc[i].get('catalyst_1_smiles', '')
        smi = str(raw).strip() if pd.notna(raw) else ''
        cation = _extract_cation_simple(smi)
        key = cation or '__empty__'

        if key in cache and cache[key] is not None:
            rows.append(cache[key])
            valid_idx.append(i)
        elif cation:
            try:
                from rdkit import Chem
                mol = Chem.MolFromSmiles(cation)
                if mol is not None:
                    result = calc(mol)
                    vals = {}
                    for desc, val in zip(calc.descriptors, result):
                        try:
                            vals[str(desc)] = float(val) if val is not None else np.nan
                        except (ValueError, TypeError):
                            vals[str(desc)] = np.nan
                    cache[key] = vals
                    rows.append(vals)
                    valid_idx.append(i)
                    n_new += 1
                else:
                    cache[key] = None
            except Exception as e:
                if verbose and i < 3:
                    print(f"    Mordred 跳过: {cation[:50]} ({str(e)[:40]})")
                cache[key] = None

    np.save(cache_path, cache)
    if verbose:
        print(f"    Mordred: 成功={len(valid_idx)}, 新增={n_new}, 缓存={len(cache)}")

    if not rows:
        return None

    feat_keys = _MORDED_FEAT_KEYS
    X = np.zeros((len(df), len(feat_keys)), dtype=np.float32)
    for j, d in enumerate(rows):
        for k, key in enumerate(feat_keys):
            X[valid_idx[j], k] = d.get(key, np.nan)

    for k in range(X.shape[1]):
        col = X[:, k]
        nan_mask = np.isnan(col)
        if nan_mask.any():
            med = np.nanmedian(col)
            X[nan_mask, k] = med if not np.isnan(med) else 0.0

    return X


def build_paper_26_descriptors(df, verbose=True):
    """
    论文 26 描述符（结构/电子部分，21 维 Mordred + rdKit 代理）。

    对应 "Machine learning for the yield prediction of CO2 cyclization
    reaction catalyzed by the ionic liquids" (Fuel 335, 2023) Table 1:
      - 7 原子计数: nH, nC, nN, nO, nAtom, nHetero, nRing
      - 扩展: nHeavy, nAromAtom, nAromBond, nSpiro, nBridgehead,
              nHBAcc, nHBDon, TopoPSA, nRotable, nBase, nAcid

    DFT 值不可用时，用 Mordred 2D 描述符作为电子结构代理。

    叠加 IL 阳离子 7 类 one-hot（Step 2），共 28 维。
    """
    n_rows = len(df)
    feat_keys = _PAPER_26_FEAT_KEYS + ['nRotable']  # 21 + 1 = 22 维
    n_feat = len(feat_keys)
    X_struct = np.zeros((n_rows, n_feat), dtype=np.float32)

    # IL 阳离子 7 类 one-hot
    X_il_subtype = np.zeros((n_rows, len(IL_CATION_SUBTYPES)), dtype=np.float32)
    type_map = {t: i for i, t in enumerate(IL_CATION_SUBTYPES)}

    cache = _PAPER_26_CACHE
    calc = _init_paper_26_calc()

    for i in range(n_rows):
        row = df.iloc[i]
        cat_type = str(row.get('catalyst_system_type', '')).strip()

        # IL 阳离子亚类 one-hot
        subtype = 'other_il'
        if cat_type == 'ionic_liquid':
            smi_raw = row.get('catalyst_1_smiles', '')
            smi = str(smi_raw).strip() if pd.notna(smi_raw) else ''
            parsed = _parse_catalyst_smiles(smi)
            cation = parsed.get('components', {}).get('cation', '')
            subtype = _detect_cation_subtype(cation)
            key = cation or '__empty__'
            if key in cache:
                d = cache[key]
            else:
                d = _compute_paper_26_for_cation(cation)
                cache[key] = d
            if d:
                for k_idx, k_name in enumerate(_PAPER_26_FEAT_KEYS):
                    if k_name in d and not np.isnan(d[k_name]):
                        X_struct[i, k_idx] = d[k_name]
                    elif k_name == 'nRotable' and 'nRotable' in d and not np.isnan(d.get('nRotable', np.nan)):
                        X_struct[i, n_feat - 1] = d['nRotable']
        idx = type_map.get(subtype, -1)
        if idx >= 0:
            X_il_subtype[i, idx] = 1.0

        if verbose and (i + 1) % 500 == 0:
            print(f"    论文26描述符 进度: {i+1}/{n_rows}")

    for k in range(n_feat):
        col = X_struct[:, k]
        nan_mask = np.isnan(col)
        if nan_mask.any():
            med = np.nanmedian(col)
            X_struct[nan_mask, k] = med if not np.isnan(med) else 0.0

    if verbose:
        print(f"  论文26描述符构建完成: 结构={n_feat}维 + IL子类={len(IL_CATION_SUBTYPES)}维")
        subtype_counts = pd.Series(X_il_subtype.argmax(axis=1)).map(
            dict(enumerate(IL_CATION_SUBTYPES))).value_counts()
        print(f"  IL阳离子亚类: {subtype_counts.to_dict()}")

    return X_struct, X_il_subtype


# ================================================================
# ================================================================
def load_enhanced_data(
    data_path,
    use_conditions=True,
    use_catalyst_fp=True,
    use_catalyst_type=True,
    use_reactant_onehot=True,
    use_drfp_variants=False,
    drfp_variant='no_cats',
    use_component_features=False,
    use_mordred=False,
    use_paper_26=False,
    morgan_bits=256,
    grouped_scale=True,
    verbose=True
):
    """
    加载增强特征数据集。

    use_component_features: 启用组分感知催化剂描述符（替代 Morgan 指纹）
    use_drfp_variants:     是否拼接 4 个 DRFP 变体（默认 False；消融实验表明拼接无额外收益）
    drfp_variant:          当 use_drfp_variants=False 时，选择单个 DRFP 变体加载。
                          'full' → DRFP全反应, 'no_cats' → DRFP去催化剂（默认，消融最优），
                          'reactants' → DRFP仅底物, 'no_sols' → DRFP去溶剂
    use_paper_26:          启用论文 26 描述符（IL 阳离子 22 维结构 + 7 类 one-hot）
    grouped_scale:          分组标准化（默认 True）

    Returns:
        X, y, df, feat_info, scaler, X_drfp_raw, X_aux_raw, X_cont_raw
    """
    df = pd.read_csv(data_path)
    if "drfp" not in df.columns:
        raise ValueError("数据文件中缺少 'drfp' 列")

    # DRFP 列名映射（内部名称 → CSV 实际列名，与 103_drfp.py 保持一致）
    DRFP_COLS = {
        'full':      'drfp',
        'reactants': 'drfp React',
        'no_cats':   'drfp wo cats',
        'no_sols':   'drfp wo sols',
    }
    if drfp_variant not in DRFP_COLS:
        raise ValueError(f"drfp_variant 必须是 {list(DRFP_COLS.keys())} 之一")

    # 1. DRFP 特征
    if use_drfp_variants:
        drfp_cols_map = {
            'drfp':         'drfp',
            'drfp React':   'drfp React',
            'drfp wo cats': 'drfp wo cats',
            'drfp wo sols': 'drfp wo sols',
        }
    else:
        drfp_cols_map = {DRFP_COLS[drfp_variant]: DRFP_COLS[drfp_variant]}

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
            if (i + 1) % 500 == 0:
                print(f"    {label} 进度: {i+1}/{n_rows}")
        arr = np.array(parts, dtype=np.float32)
        X_drfp_list.append(arr)
        drfp_info[label] = arr.shape[1]
        if verbose:
            print(f"  解析 {label} ({n_rows} 条)")

    X_drfp_all = np.hstack(X_drfp_list) if len(X_drfp_list) > 1 else X_drfp_list[0]
    feat_info = {'drfp_variants': X_drfp_all.shape[1]}

    X_component = None
    if use_component_features:
        if verbose:
            print("  构建组分感知催化剂描述符...")
        X_component = build_catalyst_component_features(df, verbose=verbose)
        feat_info['component_features'] = X_component.shape[1]

    X_cat_fp = None
    if use_catalyst_fp and not use_component_features:
        if verbose:
            print("  生成催化剂 Morgan 指纹（预计算唯一 SMILES）...")
        n_rows = len(df)
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
            print(f"    唯一催化剂 SMILES: {len(unique_list)}，预计算中...")
        fp_table = {}
        for idx, smi in enumerate(unique_list):
            fp_table[smi] = smiles_to_morgan(smi, n_bits=morgan_bits)
            if (idx + 1) % 500 == 0:
                print(f"    预计算进度: {idx+1}/{len(unique_list)}")

        zero_fp = np.zeros(morgan_bits, dtype=np.float32)
        cat_fps = np.zeros((n_rows, morgan_bits), dtype=np.float32)
        for i in range(n_rows):
            row_fps = []
            for j in range(1, 5):
                col = f'catalyst_{j}_smiles'
                if col in df.columns:
                    raw = df.iloc[i][col]
                    smi = str(raw).strip() if pd.notna(raw) else ''
                    if smi and smi not in ('', '/', 'nan', 'None'):
                        row_fps.append(fp_table.get(smi, zero_fp))
            if row_fps:
                combined = row_fps[0]
                for fp in row_fps[1:]:
                    np.maximum(combined, fp, out=combined)
                cat_fps[i] = combined
            if (i + 1) % 500 == 0:
                print(f"    催化剂指纹 进度: {i+1}/{n_rows}")
        X_cat_fp = cat_fps
        feat_info['catalyst_morgan'] = X_cat_fp.shape[1]

    X_mordred = None
    if use_mordred and not use_component_features:
        if verbose:
            print("  计算催化剂 Mordred 结构描述符（等效 DFT 电子信息近似）...")
        try:
            X_mordred = _compute_catalyst_mordred(df, verbose=verbose)
            feat_info['mordred'] = X_mordred.shape[1]
        except ImportError:
            if verbose:
                print("  [警告] Mordred 未安装，跳过")
        except Exception as e:
            if verbose:
                print(f"  [警告] Mordred 计算失败: {e}")

    # 3.5 论文 26 描述符（IL 阳离子结构/电子信息，Mordred + rdKit 代理 DFT）
    X_paper26_struct = None
    X_il_subtype_paper = None
    if use_paper_26:
        if verbose:
            print("  计算论文 26 描述符（IL 阳离子 Mordred 22 维 + 7 类 one-hot）...")
        X_paper26_struct, X_il_subtype_paper = build_paper_26_descriptors(df, verbose=verbose)
        feat_info['paper26_struct'] = X_paper26_struct.shape[1]
        feat_info['paper26_il_subtype'] = X_il_subtype_paper.shape[1]

    # 4. 底物 one-hot
    X_reactant_oh = None
    if use_reactant_onehot:
        all_reactants = sorted(df['reactant_name'].dropna().unique().tolist())
        if verbose:
            print(f"  底物类别 ({len(all_reactants)}): {all_reactants}")
        X_reactant_oh = np.array(
            [build_onehot(r, all_reactants) for r in df['reactant_name'].fillna('unknown')],
            dtype=np.float32)
        feat_info['reactant_onehot'] = X_reactant_oh.shape[1]

    X_cat_type_oh = None
    X_il_subtype = None
    if use_catalyst_type and not use_component_features:
        all_types = sorted(df['catalyst_system_type'].dropna().unique().tolist())
        X_cat_type_oh = np.array(
            [build_onehot(c, all_types) for c in df['catalyst_system_type'].fillna('unknown')],
            dtype=np.float32)
        feat_info['catalyst_type_onehot'] = X_cat_type_oh.shape[1]
        X_il_subtype = build_il_subtype_features(df)
        if verbose:
            subtype_counts = pd.Series(X_il_subtype.argmax(axis=1)).map(
                dict(enumerate(IL_CATION_SUBTYPES))).value_counts()
            print(f"  IL阳离子亚类: {subtype_counts.to_dict()}")
        feat_info['il_subtype_onehot'] = X_il_subtype.shape[1]

    X_cond = None
    if use_conditions:
        cond_basic = build_condition_features(df)
        has_solvent = (
            df['all_solvents_normalized'].notna() &
            (df['all_solvents_normalized'].astype(str).str.strip() != '')
        ).astype(np.float32).values.reshape(-1, 1)
        has_reagent = (
            (df['reagent_1_name'].notna() & (df['reagent_1_name'].astype(str).str.strip() != '')) |
            (df['reagent_2_name'].notna() & (df['reagent_2_name'].astype(str).str.strip() != ''))
        ).astype(np.float32).values.reshape(-1, 1)
        X_cond = np.hstack([cond_basic, has_solvent, has_reagent]).astype(np.float32)
        feat_info['conditions'] = X_cond.shape[1]

    binary_parts = [X_drfp_all]
    if X_component is not None:
        binary_parts.append(X_component)
    elif X_paper26_struct is not None:
        binary_parts.append(X_paper26_struct)
        if X_il_subtype_paper is not None:
            binary_parts.append(X_il_subtype_paper)
    elif X_cat_fp is not None:
        binary_parts.append(X_cat_fp)
    if X_mordred is not None:
        binary_parts.append(X_mordred)
    if X_reactant_oh is not None:
        binary_parts.append(X_reactant_oh)
    if X_il_subtype is not None and X_paper26_struct is None:
        binary_parts.append(X_il_subtype)
    X_binary = np.hstack(binary_parts)

    if X_cond is not None:
        X_cont = X_cond
    else:
        X_cont = np.zeros((len(df), 0), dtype=np.float32)

    feat_info['binary_dim'] = X_binary.shape[1]
    feat_info['cont_dim'] = X_cont.shape[1]

    if grouped_scale:
        scaler = GroupedStandardScaler()
        X_binary_s, X_cont_s = scaler.fit_transform(X_binary, X_cont)
        X = np.hstack([X_binary_s, X_cont_s])
    else:
        scaler = None
        X = np.hstack([X_binary, X_cont])

    X_drfp_raw = X_drfp_all
    aux_parts = []
    if X_component is not None:
        aux_parts.append(X_component)
    elif X_cat_fp is not None:
        aux_parts.append(X_cat_fp)
    if X_mordred is not None:
        aux_parts.append(X_mordred)
    if X_reactant_oh is not None:
        aux_parts.append(X_reactant_oh)
    if X_il_subtype is not None:
        aux_parts.append(X_il_subtype)
    X_aux_raw = np.hstack(aux_parts) if aux_parts else np.zeros((len(df), 0), dtype=np.float32)
    X_cont_raw = X_cond if X_cond is not None else np.zeros((len(df), 0), dtype=np.float32)

    # Normalize column names to canonical form
    col_renames = {}
    if "yield (%)" in df.columns and "rxn_yield" not in df.columns:
        col_renames["yield (%)"] = "rxn_yield"
    if "reactant_name" in df.columns and "reactant" not in df.columns:
        col_renames["reactant_name"] = "reactant"
    if "product_name" in df.columns and "product" not in df.columns:
        col_renames["product_name"] = "product"
    if "time (h)" in df.columns and "time" not in df.columns:
        col_renames["time (h)"] = "time"

    # Normalize temperature column to canonical name (handles all aliases)
    for alias in COND_COL_ALIASES['temperature_celsius']:
        if alias in df.columns and alias != 'temperature_celsius':
            col_renames[alias] = 'temperature_celsius'
            break

    if "pressure (MPa)" in df.columns and "pressure" not in df.columns:
        col_renames["pressure (MPa)"] = "pressure"
    if "reference" not in df.columns:
        df["reference"] = "Unknown reference"
    if col_renames:
        df = df.rename(columns=col_renames)
        if verbose:
            print(f"  Column rename mapping: {col_renames}")

    from utils_rxn import df_to_rxn_list
    rxn_list = df_to_rxn_list(df)
    y = np.array([float(rxn.rxn_yield) for rxn in rxn_list], dtype=np.float32) / 100.0

    if verbose:
        print(f"\n数据加载完成:")
        print(f"  样本数: {len(X)}")
        print(f"  总特征维度: {X.shape[1]}  (二值 {X_binary.shape[1]} + 连续 {X_cont.shape[1]})")
        print(f"  特征构成:")
        for name, dim in drfp_info.items():
            print(f"    {name}: {dim} 维")
        for name, dim in feat_info.items():
            if name not in ('drfp_variants', 'binary_dim', 'cont_dim'):
                print(f"    {name}: {dim} 维")
        print(f"  y 范围: [{y.min():.4f}, {y.max():.4f}], 均值: {y.mean():.4f}")

    return X, y, df, feat_info, scaler, X_drfp_raw, X_aux_raw, X_cont_raw


# ================================================================
# Mechanism-Aware LOSO Cross-Validation
# ================================================================
# Chemical groupings for substrate mechanism types
MECHANISM_GROUPS = {
    'internal_epoxides': [
        'Cyclohexene oxide', 'Cyclopentene oxide', 'Cyclooctene oxide',
    ],
    'terminal_epoxides': [
        'Epichlorohydrin', 'Isopropyl glycidyl ether', 'Propylene oxide',
        'Styrene oxide', '1,2-epoxy-3-phenoxypropane', 'Glycidyl ether',
        'Epoxymethylstyrene', 'Allyl glycidyl ether',
    ],
}


class MechanismAwareLOSO:
    """
    Mechanism-Aware Leave-One-Substrate-Out Cross-Validation.

    This class implements a chemically-informed LOSO strategy that accounts for
    the different reaction mechanisms between internal and terminal epoxides.

    Key improvements over standard LOSO:
    1. Special handling for CHO (internal epoxide) which has unique behavior
    2. Mechanism-aware training set selection
    3. Similarity-based prediction for unseen substrates

    Chemical rationale:
    - Internal epoxides (CHO) have different rate-determining steps
    - Terminal epoxides follow standard SN2 mechanism
    - Cross-mechanism prediction is harder but chemically meaningful
    """

    def __init__(self, substrates, mechanism_groups=None):
        """
        Args:
            substrates: Array-like of substrate names
            mechanism_groups: Dict of group_name -> list of substrate names
        """
        self.substrates = np.array(substrates)
        self.mechanism_groups = mechanism_groups or MECHANISM_GROUPS

        # Build substrate to group mapping
        self._substrate_to_group = {}
        for group_name, group_substrates in self.mechanism_groups.items():
            for sub in group_substrates:
                self._substrate_to_group[sub] = group_name

        # Classify each substrate
        self.substrate_groups = np.array([
            self._substrate_to_group.get(s, 'other') for s in self.substrates
        ])

    def get_train_val_split(self, substrate_to_leave_out):
        """
        Get train/validation split for LOSO on a specific substrate.

        For internal epoxides (CHO), uses a hybrid strategy that includes
        similar terminal epoxides in training to improve generalization.

        Args:
            substrate_to_leave_out: Name of substrate to leave out

        Returns:
            train_mask, val_mask: Boolean arrays for train/validation selection
        """
        val_mask = self.substrates == substrate_to_leave_out

        substrate_group = self._substrate_to_group.get(substrate_to_leave_out, 'other')

        if substrate_group == 'internal_epoxides':
            # For internal epoxides: use ALL terminal epoxides as training
            # This is chemically motivated as both undergo ring-opening
            train_mask = self.substrate_groups == 'terminal_epoxides'
        elif substrate_group == 'terminal_epoxides':
            # For terminal epoxides: use all other groups
            train_mask = ~val_mask
        else:
            # For unknown substrates: use everything except the held-out one
            train_mask = ~val_mask

        return train_mask, val_mask

    def get_cross_mechanism_split(self, substrate_to_leave_out):
        """
        Get split that specifically tests cross-mechanism generalization.

        This is useful for evaluating whether the model can predict
        internal epoxide behavior from terminal epoxide data and vice versa.

        Args:
            substrate_to_leave_out: Name of substrate to leave out

        Returns:
            train_mask, val_mask: Boolean arrays
        """
        substrate_group = self._substrate_to_group.get(substrate_to_leave_out, 'other')
        val_mask = self.substrates == substrate_to_leave_out

        if substrate_group == 'internal_epoxides':
            # Train only on terminal epoxides
            train_mask = self.substrate_groups == 'terminal_epoxides'
        else:
            # Train on everything except the held-out substrate
            train_mask = ~val_mask

        return train_mask, val_mask

    def get_all_splits(self):
        """
        Get all LOSO splits for all unique substrates.

        Yields:
            substrate_name, train_mask, val_mask
        """
        unique_substrates = np.unique(self.substrates)
        for substrate in unique_substrates:
            train_mask, val_mask = self.get_train_val_split(substrate)
            yield substrate, train_mask, val_mask

    def get_group_splits(self, group_name):
        """
        Get splits for all substrates in a specific mechanism group.

        Args:
            group_name: Name of the mechanism group

        Yields:
            substrate_name, train_mask, val_mask
        """
        if group_name not in self.mechanism_groups:
            return

        for substrate in self.mechanism_groups[group_name]:
            if substrate in self.substrates:
                train_mask, val_mask = self.get_train_val_split(substrate)
                yield substrate, train_mask, val_mask

    def evaluate_model(self, model, X, y, metric_fn=None):
        """
        Evaluate a model using mechanism-aware LOSO.

        Args:
            model: Scikit-learn compatible model with fit/predict methods
            X: Feature matrix (n_samples, n_features)
            y: Target values (n_samples,)
            metric_fn: Function that computes metrics, default r2_score

        Returns:
            dict with per-substrate and aggregate metrics
        """
        from sklearn.metrics import r2_score, mean_absolute_error

        if metric_fn is None:
            metric_fn = r2_score

        results = {
            'per_substrate': {},
            'aggregate': {},
            'by_group': {g: {} for g in self.mechanism_groups.keys()},
        }

        all_y_true = []
        all_y_pred = []
        all_y_pred_by_group = {g: ([], []) for g in self.mechanism_groups.keys()}

        for substrate, train_mask, val_mask in self.get_all_splits():
            if train_mask.sum() == 0:
                results['per_substrate'][substrate] = {
                    'error': 'No training samples',
                    'n_train': 0,
                    'n_val': val_mask.sum(),
                }
                continue

            try:
                model.fit(X[train_mask], y[train_mask])
                y_pred = model.predict(X[val_mask])

                r2 = metric_fn(y[val_mask], y_pred)
                mae = mean_absolute_error(y[val_mask], y_pred)

                results['per_substrate'][substrate] = {
                    'r2': float(r2),
                    'mae': float(mae),
                    'n_train': int(train_mask.sum()),
                    'n_val': int(val_mask.sum()),
                    'group': self._substrate_to_group.get(substrate, 'other'),
                }

                all_y_true.extend(y[val_mask])
                all_y_pred.extend(y_pred)

                # Track by group
                group = self._substrate_to_group.get(substrate, 'other')
                if group in all_y_pred_by_group:
                    all_y_pred_by_group[group][0].extend(y[val_mask])
                    all_y_pred_by_group[group][1].extend(y_pred)

            except Exception as e:
                results['per_substrate'][substrate] = {
                    'error': str(e),
                    'n_train': int(train_mask.sum()),
                    'n_val': int(val_mask.sum()),
                }

        # Compute aggregate metrics
        if all_y_true:
            results['aggregate'] = {
                'r2': float(metric_fn(all_y_true, all_y_pred)),
                'mae': float(mean_absolute_error(all_y_true, all_y_pred)),
                'n_total': len(all_y_true),
            }

        # Compute group-level metrics
        for group, (y_true, y_pred) in all_y_pred_by_group.items():
            if y_true:
                results['by_group'][group] = {
                    'r2': float(metric_fn(y_true, y_pred)),
                    'mae': float(mean_absolute_error(y_true, y_pred)),
                    'n': len(y_true),
                }

        return results


# ================================================================
# ================================================================
if __name__ == "__main__":
    test_path = os.path.join(PROJECT_ROOT, 'data/processed/co2_drfp.csv')
    if os.path.exists(test_path):
        print("测试增强特征加载...")
        X, y, df, info, scaler, X_drfp, X_aux, X_cont = load_enhanced_data(
            test_path, use_drfp_variants=True, use_component_features=False,
            grouped_scale=True, verbose=True)
        print(f"\nX shape: {X.shape}, y shape: {y.shape}")
        print(f"特征信息: {info}")
        print(f"NaN: {np.isnan(X).sum()}, Inf: {np.isinf(X).sum()}")
        print("测试通过！")
    else:
        print(f"测试文件不存在: {test_path}")
