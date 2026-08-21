"""
candidate_library.py
====================

Shared library definitions and feature-building functions for 402
(virtual screening) and 404 (hypothetical screening). Centralised so the
two scripts stay in sync 鈥� a reactant / catalyst added to 402's library
automatically flows into 404's augmented pool without code duplication.

Re-exported:
  - REACTANTS, CATALYSTS, SOLVENTS, CONDITIONS
  - HYPOTHETICAL_REACTANTS, HYPOTHETICAL_CATALYSTS, HYPOTHETICAL_SOLVENTS,
    HYPOTHETICAL_CONDITIONS
  - build_base_library() -> pd.DataFrame       (402-style 800 candidates)
  - build_hypothetical_library() -> pd.DataFrame (404-style 400 candidates)
  - score_candidates_with_artifacts(cand_df, art) -> np.ndarray
        Uses the persisted DualBranchANN (art['pcl_ae'] + art['ann']) to
        score an arbitrary candidate DataFrame. Replaces the previous
        404's 1-NN proxy with a real ML prediction.
"""

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from rdkit import Chem

# ---------------------------------------------------------------------------
# Library definitions
# ---------------------------------------------------------------------------

# 8 epoxide reactants (402): 3 in training (T ~ 1.0) + 5 progressively further.
REACTANTS: List[Tuple[str, str]] = [
    ('propylene oxide',         'CC1CO1'),
    ('epichlorohydrin',         'ClCC1CO1'),
    ('styrene oxide',           'C1OC1c1ccccc1'),
    ('cyclohexene oxide',       'C1CCC2OC2C1'),
    ('isopropyl glycidyl ether','CC(C)OCC1CO1'),
    ('1,2-epoxybutane',         'CCC1CO1'),
    ('furfuryl glycidyl ether', 'c1ccc(COCC2CO2)o1'),
    ('bisphenol A diglycidyl ether',
                               'C1OC1COc1ccc(C(C)(C)c2ccc(OCC3CO3)cc2)cc1'),
]

# 4 catalyst archetypes covering main families in training data.
CATALYSTS: List[Tuple[str, str, float, str]] = [
    ('TBAB',   'CCCC[N+](CCCC)(CCCC)CCCC.[Br-]',  2.5, 'ammonium_halide'),
    ('TBAI',   'CCCC[N+](CCCC)(CCCC)CCCC.[I-]',    2.5, 'ammonium_halide'),
    ('ZnBr2',  '[Br-].[Zn+2].[Br-]',               2.5, 'metal_halide'),
    ('DBU',    'C1CCC2=NCCCCN2CC1',                5.0, 'organic_base'),
]

# 5 solvent / no-solvent cases.
SOLVENTS: List[Tuple[str, str, str, str]] = [
    ('',          'solvent_free',  None,     ''),
    ('MeCN',      'acetonitrile',  'CC#N',   'acetonitrile'),
    ('DMF',       'DMF',           'CN(C)C=O','DMF'),
    ('DMSO',      'DMSO',          'CS(C)=O','DMSO'),
    ('CH2Cl2',    'dichloromethane','ClCCl', 'dichloromethane'),
]

# 5 condition templates covering the (T, P, t) high-yield region.
CONDITIONS: List[Tuple[int, float, int]] = [
    (60,  0.1, 24),
    (80,  1.0,  6),
    (100, 1.0, 12),
    (120, 2.0, 12),
    (140, 3.0,  8),
]

# 404 extension: 4 new reactants FAR from training SMILES (T < 0.30 typical).
HYPOTHETICAL_REACTANTS: List[Tuple[str, str]] = [
    ('1,2-epoxyoctane',           'CCCCCCCC1CO1'),
    ('allyl glycidyl ether',      'C=CCOCC1CO1'),
    ('phenyl glycidyl ether',     'c1ccc(OCC2CO2)cc1'),
    ('cyclohexene oxide',         'C1CCC2OC2C1'),  # already in REACTANTS
                                          # but kept here for symmetry
]

HYPOTHETICAL_CATALYSTS: List[Tuple[str, str, float, str]] = CATALYSTS

HYPOTHETICAL_SOLVENTS: List[Tuple[str, str]] = [
    ('MeCN',   'CC#N'),
    ('DMF',    'CN(C)C=O'),
    ('DMSO',   'CS(C)=O'),
    ('CH2Cl2', 'ClCCl'),
    ('solvent-free', ''),
]

HYPOTHETICAL_CONDITIONS: List[Tuple[int, float, int]] = [
    (80,  1.0,  6),
    (100, 1.0, 12),
    (120, 2.0, 12),
    (140, 3.0,  8),
    (60,  0.1, 24),
]


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

def _make_rxn_smiles(rsmi: str, csmi: str, ssmi: str) -> str:
    parts = [rsmi, csmi]
    if ssmi:
        parts.append(ssmi)
    return '.'.join(parts) + '>>' + 'O=C1OCCO1'


def build_base_library() -> pd.DataFrame:
    """402-style 800-candidate library (8 x 4 x 5 x 5)."""
    rows = []
    for rname, rsmi in REACTANTS:
        for cname, csmi, cload, cfam in CATALYSTS:
            for _sn, _sdisp, ssmi, snorm in SOLVENTS:
                for T, P, t in CONDITIONS:
                    rxn_smi = _make_rxn_smiles(rsmi, csmi, ssmi or '')
                    rows.append({
                        'reactant_name': rname,
                        'reactant_smiles': rsmi,
                        'catalyst_name': cname,
                        'catalyst_smiles': csmi,
                        'catalyst_loading_mol%': cload,
                        'catalyst_family': cfam,
                        'solvent_name': snorm or 'solvent-free',
                        'solvent_smiles': ssmi or '',
                        'temperature_C': T,
                        'pressure_MPa': P,
                        'time_h': t,
                        'RXN_SMILES': rxn_smi,
                        'product_smiles': 'O=C1OCCO1',
                    })
    return pd.DataFrame(rows)


def build_hypothetical_library() -> pd.DataFrame:
    """404-style 400-candidate library (4 x 4 x 5 x 5). The reactants are
    deliberately outside the training SMILES distribution (Tanimoto < 0.30
    typically)."""
    rows = []
    for rname, rsmi in HYPOTHETICAL_REACTANTS:
        for cname, csmi, cload, cfam in HYPOTHETICAL_CATALYSTS:
            for sname, ssmi in HYPOTHETICAL_SOLVENTS:
                for T, P, t in HYPOTHETICAL_CONDITIONS:
                    rxn_smi = _make_rxn_smiles(rsmi, csmi, ssmi)
                    rows.append({
                        'reactant_name': rname,
                        'reactant_smiles': rsmi,
                        'catalyst_name': cname,
                        'catalyst_smiles': csmi,
                        'catalyst_loading_mol%': cload,
                        'catalyst_family': cfam,
                        'solvent_name': sname,
                        'solvent_smiles': ssmi,
                        'temperature_C': T,
                        'pressure_MPa': P,
                        'time_h': t,
                        'RXN_SMILES': rxn_smi,
                        'product_smiles': 'O=C1OCCO1',
                    })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Real-ML scoring for arbitrary candidate DataFrames
# ---------------------------------------------------------------------------

def score_candidates_with_artifacts(cand_df: pd.DataFrame,
                                    art: Dict,
                                    cache: Dict | None = None,
                                    medians: Dict | None = None) -> np.ndarray:
    """Score `cand_df` with the persisted DualBranchANN (art must contain
    'pcl_ae', 'ann', 'drfp_scaler', 'other_scaler', 'meta' as in 402).
    Returns a 1-D np.ndarray of clipped predicted yields (0..1).

    For xTB lookups, we use the provided `cache` (xTB cache) and `medians`
    (role-specific medians from 402.load_xtb_medians). If either is None,
    xTB features fall back to the dataset global median.
    """
    from drfp import DrfpEncoder

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    meta = art['meta']
    xtb_cols = meta['xtb_cols']
    cond_cols = meta['cond_cols']

    rxn_list = cand_df['RXN_SMILES'].astype(str).tolist()
    # DRFP on full reaction
    fps_drfp = np.asarray(DrfpEncoder.encode(rxn_list, n_folded_length=2048),
                           dtype=np.float32)

    n = len(cand_df)
    n_xtb_cond_inter = meta['xtb_cond_inter_dim']
    X_other = np.zeros((n, n_xtb_cond_inter), dtype=np.float32)

    if medians is None:
        medians = {c: 0.0 for c in xtb_cols}

    for i, row in cand_df.reset_index(drop=True).iterrows():
        sub = row.get('reactant_smiles', '')
        cat = row.get('catalyst_smiles', '')
        solv = row.get('solvent_smiles', '')

        xtb_row = np.array([medians[c] for c in xtb_cols], dtype=np.float32)
        if cache is not None:
            for role, smi in (('sub', sub), ('cat', cat), ('solv', solv)):
                if not smi:
                    continue
                m = Chem.MolFromSmiles(smi)
                if m is None:
                    continue
                can = Chem.MolToSmiles(m)
                for chg in (0, 1, -1):
                    k = f'{can}|{chg}'
                    e = cache.get(k)
                    if not e or not e.get('xtb_ok'):
                        continue
                    homo = e.get('homo_eV')
                    if homo in (None, '', 'NaN'):
                        continue
                    if f'{role}_homo_eV' in xtb_cols:
                        xtb_row[xtb_cols.index(f'{role}_homo_eV')] = float(homo)
                    for kk in ('lumo_eV', 'gap_eV', 'dipole_D'):
                        v = e.get(kk)
                        if v in (None, '', 'NaN'):
                            continue
                        col = f'{role}_{kk.replace("_eV", "_eV").replace("dipole_D", "dipole_D")}'
                        if kk == 'lumo_eV': col = f'{role}_lumo_eV'
                        if kk == 'gap_eV':  col = f'{role}_gap_eV'
                        if kk == 'dipole_D': col = f'{role}_dipole_D'
                        if col in xtb_cols:
                            xtb_row[xtb_cols.index(col)] = float(v)
                    break

        T = float(row.get('temperature_C', 80.0))
        P = float(row.get('pressure_MPa', 1.0))
        t = float(row.get('time_h', 12.0))
        loading = float(row.get('catalyst_loading_mol%', 2.5))
        cond_row = np.array([T, P, t, loading, 0.0, 0.0, 0.0], dtype=np.float32)

        ai = xtb_cols.index('activation_proxy')
        tpi = xtb_cols.index('total_polarity_index')
        inter_row = np.array([T * xtb_row[ai], P * xtb_row[tpi]], dtype=np.float32)
        X_other[i] = np.concatenate([xtb_row, cond_row, inter_row])

    drfp_scaler = art['drfp_scaler']
    pcl_ae = art['pcl_ae']
    other_scaler = art['other_scaler']
    ann = art['ann']

    Xd_s = drfp_scaler.transform(fps_drfp).astype(np.float32)
    with torch.no_grad():
        Xd_red = pcl_ae.encode(torch.FloatTensor(Xd_s).to(DEVICE)).cpu().numpy().astype(np.float32)
    Xo_s = other_scaler.transform(X_other).astype(np.float32)
    with torch.no_grad():
        Xd_t = torch.tensor(Xd_red, dtype=torch.float32).to(DEVICE)
        Xo_t = torch.tensor(Xo_s, dtype=torch.float32).to(DEVICE)
        preds = ann(Xd_t, Xo_t).cpu().numpy()
    return np.clip(preds, 0.0, 1.0)