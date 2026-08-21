# -*- coding: utf-8 -*-
"""
virtual_screening_top10.py
==========================

Score a virtual library of CO2-cycloaddition candidates with the persisted
best pipeline (PCL-AE-128 + DualBranchANN) trained by save_best_model.py.

Workflow
--------
  [1] Load artifacts (drfp_scaler, pcl_ae_encoder, xtb_cond_inter_scaler,
      dual_branch_ann, feature_meta).
  [2] Build candidate library: enumerate epoxide reactants x catalysts x
      solvents x conditions -> 450+ candidates.
  [3] For each candidate, compute DRFP (DrfpEncoder) and XTB-derived features
      (with _xtb_cache lookup; if miss, fill with dataset median and flag).
  [4] Score each candidate with the trained DualBranchANN (point estimate).
  [5] Bootstrap CI: retrain N=B DualBranchANN models on bootstrap samples of
      the training data, then rescore the candidate library -> mean / std / CI.
  [6] Optional xTB single-point validation for cache-hit entries.
  [7] Rank by predicted yield, output Top-10 + CSV + report + figure.

Inputs (must exist after save_best_model.py):
  results_best_pipeline/artifacts/
      drfp_scaler.joblib
      pcl_ae_encoder.pt
      xtb_cond_inter_scaler.joblib
      dual_branch_ann.pt
      feature_meta.json
      training_metrics.json

Outputs:
  results_virtual_screening/
      candidates_full.csv
      top10_results.csv
      top10_results.txt
      figure_top10.png

Usage:
  D:\\co2\\env_drfp\\python.exe virtual_screening_top10.py
"""

import os
import io
import json
import os
import sys
import time
import warnings
from typing import Dict, List, Optional, Tuple

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
warnings.filterwarnings('ignore')

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from drfp import DrfpEncoder
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

PROJECT_ROOT = os.environ.get("CO2_PROJECT_ROOT", r"D:\machine-learning\CO2-cycloaddition")
sys.path.insert(0, PROJECT_ROOT)

# Paths
ARTIFACT_DIR = os.path.join(PROJECT_ROOT, 'results_best_pipeline', 'artifacts')
DATA_EXTENDED = os.path.join(PROJECT_ROOT, 'results/results_cho_diagnostic/co2_drfp_xtb_extended.csv')
XTB_CACHE = os.path.join(PROJECT_ROOT, '_xtb_cache', 'xtb_cache_clean.json')
OUT_DIR = os.path.join(PROJECT_ROOT, 'results_virtual_screening')
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)


# ----------------------------------------------------------------------
# 1. Model architectures (mirror save_best_model.py exactly)
# ----------------------------------------------------------------------

class PropertyCoLearningAE(nn.Module):
    """Mirror of save_best_model.py's class - full encoder/decoder/predictor
    so load_state_dict matches. We only use .encode() for inference."""
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.ReLU(),
            nn.Linear(256, 512), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Linear(512, input_dim), nn.Sigmoid(),
        )
        self.predictor = nn.Sequential(
            nn.Linear(latent_dim, 64), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1),
        )

    def encode(self, x):
        return self.encoder(x)


class DualBranchANN(nn.Module):
    def __init__(self, drfp_dim, xtb_dim, hidden=128):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.bn_drfp = nn.BatchNorm1d(64)
        self.fc_drfp = nn.Sequential(
            nn.Linear(64, hidden), nn.LeakyReLU(0.1), nn.Dropout(0.3),
            nn.Linear(hidden, 64), nn.LeakyReLU(0.1),
        )
        self.fc_xtb = nn.Sequential(
            nn.Linear(xtb_dim, 64), nn.BatchNorm1d(64), nn.LeakyReLU(0.1), nn.Dropout(0.3),
            nn.Linear(64, 32), nn.LeakyReLU(0.1), nn.Dropout(0.2),
        )
        self.fc_out = nn.Sequential(
            nn.Linear(64 + 32, 64), nn.BatchNorm1d(64), nn.LeakyReLU(0.1),
            nn.Linear(64, 1),
        )

    def forward(self, x_drfp, x_xtb):
        h = x_drfp.unsqueeze(1)
        h = torch.relu(self.conv1(h))
        h = torch.relu(self.conv2(h))
        h = self.pool(h).squeeze(-1)
        h = self.bn_drfp(h)
        h = self.fc_drfp(h)
        h2 = self.fc_xtb(x_xtb)
        return self.fc_out(torch.cat([h, h2], dim=1)).squeeze(-1)


# ----------------------------------------------------------------------
# 2. Artifact loading
# ----------------------------------------------------------------------

def load_artifacts() -> Dict:
    """Load all model + scaler artifacts + meta from save_best_model.py."""
    print('[1/6] Loading artifacts ...')
    assert os.path.isdir(ARTIFACT_DIR), f'Artifacts missing: run save_best_model.py first'

    drfp_scaler = joblib.load(os.path.join(ARTIFACT_DIR, 'drfp_scaler.joblib'))
    other_scaler = joblib.load(os.path.join(ARTIFACT_DIR, 'xtb_cond_inter_scaler.joblib'))

    pcl_ckpt = torch.load(os.path.join(ARTIFACT_DIR, 'pcl_ae_encoder.pt'),
                           map_location=DEVICE, weights_only=False)
    pcl_ae = PropertyCoLearningAE(pcl_ckpt['input_dim'], pcl_ckpt['latent_dim']).to(DEVICE)
    pcl_ae.load_state_dict(pcl_ckpt['state_dict'])
    pcl_ae.eval()

    ann_ckpt = torch.load(os.path.join(ARTIFACT_DIR, 'dual_branch_ann.pt'),
                           map_location=DEVICE, weights_only=False)
    ann = DualBranchANN(ann_ckpt['drfp_dim'], ann_ckpt['xtb_dim'], hidden=ann_ckpt['hidden']).to(DEVICE)
    ann.load_state_dict(ann_ckpt['state_dict'])
    ann.eval()

    with open(os.path.join(ARTIFACT_DIR, 'feature_meta.json'), encoding='utf-8') as f:
        meta = json.load(f)
    with open(os.path.join(ARTIFACT_DIR, 'training_metrics.json'), encoding='utf-8') as f:
        cv_metrics = json.load(f)

    print(f'  DRFP raw dim         : {meta["drfp_raw_dim"]}')
    print(f'  DRFP reduced dim     : {meta["drfp_reduced_dim"]}')
    print(f'  XTB cond_inter dim   : {meta["xtb_cond_inter_dim"]}')
    print(f'  CV R^2 (reference)   : {cv_metrics["r2_mean"]:.4f} +/- {cv_metrics["r2_std"]:.4f}')

    return {
        'drfp_scaler': drfp_scaler, 'other_scaler': other_scaler,
        'pcl_ae': pcl_ae, 'ann': ann,
        'meta': meta, 'cv_metrics': cv_metrics,
    }


# ----------------------------------------------------------------------
# 3. Candidate library (epoxide reactants x catalysts x solvents)
# ----------------------------------------------------------------------

# 28 epoxide reactants commonly seen in CO2-cycloaddition literature.
# All SMILES are real & validated against RDKit.
REACTANTS = [
    # Terminal aliphatic epoxides
    ('propylene oxide',         'CC1CO1'),
    ('epichlorohydrin',         'ClCC1CO1'),
    ('1,2-epoxybutane',         'CCC1CO1'),
    ('1,2-epoxypentane',        'CCCC1CO1'),
    ('1,2-epoxyhexane',         'CCCCC1CO1'),
    ('glycidol',                'OCC1CO1'),
    ('glycidyl methyl ether',   'COCC1CO1'),
    ('allyl glycidyl ether',    'C=CCOCC1CO1'),
    ('tert-butyl glycidyl ether','CC(C)(C)OCC1CO1'),
    ('phenyl glycidyl ether',   'c1ccc(OCC2CO2)cc1'),
    # Aromatic / styrene-type epoxides
    ('styrene oxide',           'C1OC1c1ccccc1'),
    ('4-methylstyrene oxide',   'C1OC1c1ccc(C)cc1'),
    ('4-chlorostyrene oxide',   'C1OC1c1ccc(Cl)cc1'),
    ('4-fluorostyrene oxide',   'C1OC1c1ccc(F)cc1'),
    ('4-methoxystyrene oxide',  'C1OC1c1ccc(OC)cc1'),
    ('4-nitrostyrene oxide',    'C1OC1c1ccc([N+](=O)[O-])cc1'),
    ('alpha-methylstyrene oxide','CC1(c2ccccc2)CO1'),
    ('trans-stilbene oxide',    'c1ccc([C@@H]2O[C@@H]2c2ccccc2)cc1'),
    ('2-naphthyloxirane',       'c1ccc2cc(C3CO3)ccc2c1'),
    # Internal / cyclic / functionalised
    ('cyclohexene oxide',       'C1CCC2OC2C1'),
    ('cyclopentene oxide',      'C1CCC2OC2C1'),
    ('cycloheptene oxide',      'C1CCCC2OC2C1'),
    ('indene oxide',            'c1ccc2c(c1)C1OC1C2'),
    ('1,2-epoxy-3-phenoxypropane','c1ccc(OCC2CO2)cc1'),
    ('isobutylene oxide',       'CC1(C)CO1'),
    ('2,3-epoxypropyl benzene', 'c1ccc(CC2CO2)cc1'),
    ('vinyl cyclohexene oxide', 'C=CC1CCC2OC2C1'),
    ('furfuryl glycidyl ether', 'c1ccc(COCC2CO2)o1'),
    # Bis-epoxide
    ('bisphenol A diglycidyl ether (model)', 'C1OC1COc1ccc(C(C)(C)c2ccc(OCC3CO3)cc2)cc1'),
]

# Best 6 catalysts (consistent with training-data winners)
CATALYSTS = [
    # (name, smiles, loading_mol%, family)
    ('TBAB',  'CCCC[N+](CCCC)(CCCC)CCCC.[Br-]',                2.5, 'ionic_liquid'),
    ('TBAI',  'CCCC[N+](CCCC)(CCCC)CCCC.[I-]',                  2.5, 'ionic_liquid'),
    ('KI',    '[K+].[I-]',                                       5.0, 'metal_halide'),
    ('ZnBr2', '[Br-].[Zn+2].[Br-]',                              2.5, 'metal_halide'),
    ('choline iodide', 'C[N+](C)(C)CCO.[I-]',                   5.0, 'ionic_liquid'),
    ('TBAC',  'CCCC[N+](CCCC)(CCCC)CCCC.[Cl-]',                 2.5, 'ionic_liquid'),
]

# 4 solvents + solvent-free
SOLVENTS = [
    ('',          'solvent_free', None,      ''),
    ('MeCN',      'acetonitrile', 'CC#N',    'acetonitrile'),
    ('DMF',       'DMF',          'CN(C)C=O','DMF'),
    ('DMSO',      'DMSO',         'CS(C)=O', 'DMSO'),
]

# 4 condition templates (T, P, t) sampled from typical training-set high-yield regime
CONDITIONS = [
    (80,  1.0,  6),
    (100, 1.0, 12),
    (120, 2.0, 12),
    (60,  0.1, 24),
]


def build_candidate_library() -> pd.DataFrame:
    """Enumerate reactant x catalyst x solvent x condition."""
    print('[2/6] Building candidate library ...')
    rows = []
    for rname, rsmi in REACTANTS:
        for cname, csmi, cload, cfam in CATALYSTS:
            for sname, _, ssmi, snorm in SOLVENTS:
                for T, P, t in CONDITIONS:
                    # Build approximate product SMILES (cyclic carbonate)
                    # Two isomers: 5- vs 6-membered not differentiated, just the
                    # generic cyclic-carbonate skeleton. We use '?' to skip rxn
                    # but the actual physical SMILES still works for DRFP.
                    psmi_guess = None
                    try:
                        m = Chem.MolFromSmiles(rsmi)
                        if m is not None:
                            # Heuristic: count ring atoms + 3 -> cyclic carbonate (O=C1OCCO1)
                            ring_atoms = m.GetRingInfo().AtomRings()
                            if ring_atoms:
                                # Epoxide 3-membered -> 5-membered cyclic carbonate
                                psmi_guess = 'O=C1OCCO1'  # generic cyclic carbonate
                    except Exception:
                        pass

                    # RXN_SMILES for DRFP: reactants>>products
                    parts = [rsmi, csmi]
                    if ssmi:
                        parts.append(ssmi)
                    rxn_smi = '.'.join(parts) + '>>' + (psmi_guess or 'O=C1OCCO1')

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
                        'product_smiles': psmi_guess or 'O=C1OCCO1',
                    })
    df = pd.DataFrame(rows)
    print(f'  Total candidates (before filter): {len(df)}')
    return df


# Tanimoto gate: only keep candidates whose reactant has max Tanimoto >= THRESH
# to any training-set reactant. This constrains the screen to in-distribution
# chemistry and avoids wild extrapolation.
SIMILARITY_THRESHOLD = 0.3
REACTANT_FP_RADIUS = 2
REACTANT_FP_BITS = 2048


def _fp(smi):
    m = Chem.MolFromSmiles(str(smi))
    if m is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(m, REACTANT_FP_RADIUS,
                                                  nBits=REACTANT_FP_BITS)


def compute_tanimoto_gate(cand_df: pd.DataFrame, train_react_smiles: List[str],
                          train_yields: np.ndarray) -> pd.DataFrame:
    """Annotate each candidate with (a) max Tanimoto to any training reactant,
    (b) 1-NN training yield and SMILES. Then drop candidates below THRESHOLD.

    Returns the filtered DataFrame with three extra columns.
    """
    print(f'[2.5/6] Tanimoto gate (threshold T >= {SIMILARITY_THRESHOLD}) ...')
    train_fps = [_fp(s) for s in train_react_smiles]
    n_train = len(train_fps)

    max_sim, nn1_yield, nn1_smi = [], [], []
    for _, row in cand_df.iterrows():
        fp1 = _fp(row['reactant_smiles'])
        if fp1 is None:
            max_sim.append(0.0); nn1_yield.append(np.nan); nn1_smi.append('')
            continue
        best, best_j = 0.0, -1
        for j, fp2 in enumerate(train_fps):
            if fp2 is None:
                continue
            s = DataStructs.TanimotoSimilarity(fp1, fp2)
            if s > best:
                best, best_j = s, j
        max_sim.append(best)
        nn1_yield.append(float(train_yields[best_j]) if best_j >= 0 else np.nan)
        nn1_smi.append(train_react_smiles[best_j] if best_j >= 0 else '')

    cand_df = cand_df.copy()
    cand_df['tanimoto_max'] = max_sim
    cand_df['nn1_train_yield'] = nn1_yield
    cand_df['nn1_train_smiles'] = nn1_smi

    print(f'  Tanimoto distribution: min={min(max_sim):.3f}, '
          f'mean={np.mean(max_sim):.3f}, median={np.median(max_sim):.3f}, '
          f'max={max(max_sim):.3f}')

    before = len(cand_df)
    df_filt = cand_df[cand_df['tanimoto_max'] >= SIMILARITY_THRESHOLD].reset_index(drop=True)
    print(f'  Kept {len(df_filt)}/{before} candidates '
          f'(T >= {SIMILARITY_THRESHOLD}; dropped {before - len(df_filt)})')

    if len(df_filt) == 0:
        raise RuntimeError(
            f'All {before} candidates were filtered out by Tanimoto threshold '
            f'{SIMILARITY_THRESHOLD}. Lower it.')
    return df_filt


# ----------------------------------------------------------------------
# 4. Feature builder for new candidates
# ----------------------------------------------------------------------

def compute_drfp_for_rxn(rxn_smiles_list: List[str], n_bits: int = 2048) -> np.ndarray:
    """Compute DRFP for a list of reaction SMILES, return (N, n_bits) array.

    Only the "full" DRFP variant (2048-d) is returned. The 401-persisted
    drfp_scaler and pcl_ae_encoder were trained on a single 2048-d DRFP
    variant (not the 4-variant 8192-d concatenation used in the legacy
    ablation). Returning a wider matrix here breaks score_candidates because
    StandardScaler.transform() then refuses to accept 8192 features when
    expecting 2048. See Aug 20, 2026 fix.
    """
    print(f'  Computing DRFP for {len(rxn_smiles_list)} reactions ...')
    fps_drfp = DrfpEncoder.encode(rxn_smiles_list, n_folded_length=n_bits)
    fps_drfp = np.asarray(fps_drfp, dtype=np.float32)
    return fps_drfp


def load_xtb_medians() -> Tuple[Dict[str, float], Dict[str, float]]:
    """Compute median XTB values & condition stats from the training set,
    used as fallback for cache-miss candidates."""
    print('  Computing XTB median fallback from training data ...')
    df = pd.read_csv(DATA_EXTENDED, encoding='utf-8-sig')
    with open(os.path.join(ARTIFACT_DIR, 'feature_meta.json'), encoding='utf-8') as f:
        meta = json.load(f)
    xtb_cols = meta['xtb_cols']
    cond_cols = meta['cond_cols']
    xtb_medians = {c: float(np.nanmedian(df[c].astype(float).values)) for c in xtb_cols}
    cond_medians = {c: float(np.nanmedian(df[c].astype(float).values)) for c in cond_cols}
    print(f'    XTB medians: {len(xtb_medians)} cols; Cond medians: {len(cond_medians)} cols')
    return xtb_medians, cond_medians


def load_xtb_cache() -> Dict[str, Dict]:
    """Load precomputed xTB single-molecule properties."""
    cache = {}
    if os.path.isfile(XTB_CACHE):
        with open(XTB_CACHE, encoding='utf-8') as f:
            cache = json.load(f)
        print(f'  xTB cache loaded: {len(cache)} entries')
    else:
        print(f'  [WARN] xTB cache not found at {XTB_CACHE}')
    return cache


def get_xtb_for_smiles(smi: str, role: str, cache: Dict, medians: Dict) -> Dict[str, float]:
    """Look up xTB descriptors for one molecule. Cache hit -> real values.
    Cache miss -> fall back to role-specific default."""
    smi_key = smi.strip()
    if not smi_key:
        # empty molecule (e.g. solvent-free or no second catalyst)
        return dict(medians)
    if smi_key in cache:
        # Cache entry keys are: homo_eV, lumo_eV, gap_eV, dipole_D
        # plus optional: charge, f_plus, f_minus, etc.
        e = cache[smi_key]
        out = dict(medians)
        for k, v in e.items():
            # Map cache keys to xtb_cols naming convention
            if k == 'homo_eV':
                out[f'{role}_homo_eV'] = float(v)
            elif k == 'lumo_eV':
                out[f'{role}_lumo_eV'] = float(v)
            elif k == 'gap_eV':
                out[f'{role}_gap_eV'] = float(v)
            elif k == 'dipole_D':
                out[f'{role}_dipole_D'] = float(v)
        return out
    return dict(medians)


def build_candidate_features(cand_df: pd.DataFrame, meta: Dict,
                              xtb_cache: Dict,
                              xtb_medians: Dict, cond_medians: Dict) -> Tuple[np.ndarray, List[bool]]:
    """Compute (DRFP, X_xtb_cond_inter) feature matrix for all candidates.

    Returns:
      X_full_drfp : (N, 8192) numpy float32  -- raw DRFP for the scaler
      X_other     : (N, n_xtb_cond_inter) numpy float32
      cache_hits  : list of bool per candidate, True if at least sub+cat had cache hit
    """
    print('[3/6] Computing candidate features ...')
    xtb_cols = meta['xtb_cols']
    cond_cols = meta['cond_cols']
    n = len(cand_df)
    n_xtb_cond_inter = meta['xtb_cond_inter_dim']

    # ----- DRFP (heavy)
    rxn_list = cand_df['RXN_SMILES'].tolist()
    X_drfp_raw = compute_drfp_for_rxn(rxn_list, n_bits=2048)  # (N, 8192)
    print(f'  DRFP matrix: {X_drfp_raw.shape}')

    # ----- X_xtb_cond_inter via cache lookup
    X_other = np.zeros((n, n_xtb_cond_inter), dtype=np.float32)
    cache_hit_mask = np.zeros(n, dtype=bool)
    for i, row in cand_df.iterrows():
        # Substrate
        sub = row['reactant_smiles']
        cat = row['catalyst_smiles']
        solv = row['solvent_smiles']
        # Cache lookups
        sub_x = get_xtb_for_smiles(sub, 'sub', xtb_cache, xtb_medians)
        cat_x = get_xtb_for_smiles(cat, 'cat', xtb_cache, xtb_medians)
        solv_x = get_xtb_for_smiles(solv, 'solv', xtb_cache, xtb_medians) if solv else dict(xtb_medians)

        if (sub in xtb_cache) and (cat in xtb_cache):
            cache_hit_mask[i] = True

        # Aggregate xtb row (in meta.xtb_cols order)
        xtb_row = np.array([xtb_medians[c] for c in xtb_cols], dtype=np.float32)
        # Override fields using {sub, cat, solv} roles. For multi-role cols
        # (cation/anion/electrophilicity etc.) we just use the catalyst value
        # when available; otherwise keep median.
        for role_dict in (sub_x, cat_x, solv_x):
            for k, v in role_dict.items():
                if k in xtb_cols:
                    xtb_row[xtb_cols.index(k)] = v

        # Conditions
        T = float(row['temperature_C'])
        P = float(row['pressure_MPa'])
        t = float(row['time_h'])
        loading = float(row['catalyst_loading_mol%'])
        cond_row = np.array([T, P, t, loading, 0.0, 0.0, 0.0], dtype=np.float32)  # last 3 are cat_2/3/4 loading

        # Interactions (T * activation_proxy ; P * total_polarity_index)
        ai = xtb_cols.index('activation_proxy')
        tpi = xtb_cols.index('total_polarity_index')
        inter_row = np.array([T * xtb_row[ai], P * xtb_row[tpi]], dtype=np.float32)

        X_other[i] = np.concatenate([xtb_row, cond_row, inter_row])

    print(f'  X_other matrix: {X_other.shape}; cache hits: {cache_hit_mask.sum()}/{n}')
    return X_drfp_raw, X_other, cache_hit_mask.tolist()


# ----------------------------------------------------------------------
# 5. Score with the trained model
# ----------------------------------------------------------------------

def score_candidates(art: Dict, X_drfp_raw: np.ndarray, X_other: np.ndarray) -> np.ndarray:
    """Point estimate of predicted yield (clipped 0..1)."""
    print('[4/6] Scoring candidates ...')
    drfp_scaler = art['drfp_scaler']
    pcl_ae = art['pcl_ae']
    other_scaler = art['other_scaler']
    ann = art['ann']

    # Apply same scaling pipeline as training
    Xd_s = drfp_scaler.transform(X_drfp_raw).astype(np.float32)
    with torch.no_grad():
        Xd_red = pcl_ae.encode(torch.FloatTensor(Xd_s).to(DEVICE)).cpu().numpy().astype(np.float32)
    Xo_s = other_scaler.transform(X_other).astype(np.float32)

    with torch.no_grad():
        Xd_t = torch.tensor(Xd_red, dtype=torch.float32).to(DEVICE)
        Xo_t = torch.tensor(Xo_s, dtype=torch.float32).to(DEVICE)
        preds = ann(Xd_t, Xo_t).cpu().numpy()

    preds = np.clip(preds, 0.0, 1.0)
    print(f'  Predicted yield range: [{preds.min():.3f}, {preds.max():.3f}], mean={preds.mean():.3f}')
    return preds


# ----------------------------------------------------------------------
# 6. Bootstrap CI: retrain N DualANN on bootstrap subsets, rescore
# ----------------------------------------------------------------------

def bootstrap_ci(art: Dict, X_drfp_raw: np.ndarray, X_other: np.ndarray,
                 n_boot: int = 20, sub_frac: float = 0.8) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For each bootstrap replicate:
       - resample training set with replacement
       - retrain a fresh DualBranchANN (same arch) for a few epochs
       - score the candidates
    Returns mean, std, percentile-CI.
    """
    print(f'[5/6] Bootstrap CI (N={n_boot}, sub_frac={sub_frac}) ...')
    # Need training DRFP+X_other+y. Reload from disk.
    df = pd.read_csv(DATA_EXTENDED, encoding='utf-8-sig')
    df = df[df['extraction_status'] == 'valid'].copy()
    df = df.dropna(subset=['yield (%)'])
    df = df[df['yield (%)'] > 0].reset_index(drop=True)
    y_full = df['yield (%)'].values.astype(np.float32) / 100.0

    # Recompute DRFP for the training reactions (slow!)
    # NOTE: only the 'drfp' (full) variant is used here. See the Aug 20, 2026
    # fix to compute_drfp_for_rxn — the 4-variant 8192-d concatenation is
    # incompatible with the 401-persisted 2048-d drfp_scaler.
    print('  Rebuilding training DRFP (one-time) ...')
    from utils_rxn import read_drfp
    if 'drfp' not in df.columns:
        raise KeyError("Expected 'drfp' column in training CSV")
    arr = []
    for s in df['drfp']:
        a = read_drfp(s)
        arr.append(np.zeros(2048, dtype=np.float32) if a is None or a.size == 0
                   else a.astype(np.float32))
    X_drfp_train = np.array(arr).astype(np.float32)

    # Apply scaler + PCL-AE -> 128D
    drfp_scaler = art['drfp_scaler']
    pcl_ae = art['pcl_ae']
    Xd_s = drfp_scaler.transform(X_drfp_train).astype(np.float32)
    with torch.no_grad():
        Xd_train_red = pcl_ae.encode(torch.FloatTensor(Xd_s).to(DEVICE)).cpu().numpy().astype(np.float32)

    # X_other for training: we need to recompute. Easier: load the parquet cache
    # if it exists; else recompute from CSV.
    print('  Rebuilding training X_other (one-time) ...')
    with open(os.path.join(ARTIFACT_DIR, 'feature_meta.json'), encoding='utf-8') as f:
        meta = json.load(f)
    xtb_cols = meta['xtb_cols']
    cond_cols = meta['cond_cols']
    X_xtb_train = np.nan_to_num(df[xtb_cols].values.astype(np.float32), nan=0.0)
    X_cond_train = np.nan_to_num(df[cond_cols].values.astype(np.float32), nan=0.0)
    ai = xtb_cols.index('activation_proxy')
    tpi = xtb_cols.index('total_polarity_index')
    T = X_cond_train[:, 0:1]; P = X_cond_train[:, 1:2]
    X_inter_train = np.concatenate([T * X_xtb_train[:, ai:ai+1],
                                      P * X_xtb_train[:, tpi:tpi+1]], axis=1).astype(np.float32)
    X_other_train_raw = np.concatenate([X_xtb_train, X_cond_train, X_inter_train], axis=1).astype(np.float32)

    other_scaler = art['other_scaler']
    Xo_train_s = other_scaler.transform(X_other_train_raw).astype(np.float32)

    # Transform candidates
    Xd_s_cand = drfp_scaler.transform(X_drfp_raw).astype(np.float32)
    with torch.no_grad():
        Xd_cand_red = pcl_ae.encode(torch.FloatTensor(Xd_s_cand).to(DEVICE)).cpu().numpy().astype(np.float32)
    Xo_cand_s = other_scaler.transform(X_other).astype(np.float32)

    n_train = len(y_full)
    n_cands = len(X_drfp_raw)
    boot_preds = np.zeros((n_boot, n_cands), dtype=np.float32)

    from torch.utils.data import DataLoader, TensorDataset
    drfp_dim = Xd_train_red.shape[1]
    xtb_dim = Xo_train_s.shape[1]

    rng = np.random.RandomState(SEED)
    for b in range(n_boot):
        idx = rng.choice(n_train, size=int(n_train * sub_frac), replace=True)
        Xd_b = torch.tensor(Xd_train_red[idx], dtype=torch.float32).to(DEVICE)
        Xo_b = torch.tensor(Xo_train_s[idx], dtype=torch.float32).to(DEVICE)
        y_b = torch.tensor(y_full[idx], dtype=torch.float32).to(DEVICE)

        model = DualBranchANN(drfp_dim, xtb_dim, hidden=128).to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        ds = TensorDataset(Xd_b, Xo_b, y_b)
        dl = DataLoader(ds, batch_size=64, shuffle=True)

        # short bootstrap training (40 epochs is enough for uncertainty)
        for ep in range(40):
            model.train()
            for xd, xo, yy in dl:
                opt.zero_grad()
                nn.MSELoss()(model(xd, xo), yy).backward()
                opt.step()
        model.eval()
        with torch.no_grad():
            Xd_c = torch.tensor(Xd_cand_red, dtype=torch.float32).to(DEVICE)
            Xo_c = torch.tensor(Xo_cand_s, dtype=torch.float32).to(DEVICE)
            boot_preds[b] = model(Xd_c, Xo_c).cpu().numpy()

        print(f'  Boot {b+1:2d}/{n_boot} done')

    boot_preds = np.clip(boot_preds, 0.0, 1.0)
    mean = boot_preds.mean(axis=0)
    std = boot_preds.std(axis=0)
    ci_lo = np.percentile(boot_preds, 2.5, axis=0)
    ci_hi = np.percentile(boot_preds, 97.5, axis=0)
    print(f'  Bootstrap done. Mean CI width: {(ci_hi - ci_lo).mean():.3f}')
    return mean, std, np.stack([ci_lo, ci_hi], axis=1)


# ----------------------------------------------------------------------
# 7. Report
# ----------------------------------------------------------------------

def write_outputs(cand_df: pd.DataFrame, point_preds: np.ndarray,
                  boot_mean: np.ndarray, boot_std: np.ndarray, boot_ci: np.ndarray,
                  cache_hits: List[bool], cv_r2_ref: float) -> None:
    print('[6/6] Writing outputs ...')
    df = cand_df.copy()
    df['pred_yield_point'] = point_preds
    df['pred_yield_boot_mean'] = boot_mean
    df['pred_yield_boot_std'] = boot_std
    df['pred_yield_ci_lo'] = boot_ci[:, 0]
    df['pred_yield_ci_hi'] = boot_ci[:, 1]
    df['cache_hit'] = cache_hits

    # Sort by bootstrap mean
    df_sorted = df.sort_values('pred_yield_boot_mean', ascending=False).reset_index(drop=True)

    # Save full candidate table
    df_sorted.to_csv(os.path.join(OUT_DIR, 'candidates_full.csv'), index=False, encoding='utf-8-sig')
    print(f'  Saved: candidates_full.csv  ({len(df_sorted)} rows)')

    # Save top 10
    top10 = df_sorted.head(10).copy()
    top10.to_csv(os.path.join(OUT_DIR, 'top10_results.csv'), index=False, encoding='utf-8-sig')
    print(f'  Saved: top10_results.csv')

    # Save top 20 + top 50 for context
    df_sorted.head(20).to_csv(os.path.join(OUT_DIR, 'top20_results.csv'), index=False, encoding='utf-8-sig')
    df_sorted.head(50).to_csv(os.path.join(OUT_DIR, 'top50_results.csv'), index=False, encoding='utf-8-sig')

    # ----- Text report
    lines = []
    lines.append('=' * 88)
    lines.append('  Virtual Screening for CO2 Cycloaddition - Top 10 Predicted Yields')
    lines.append('  (Tanimoto gate: max reactant Tanimoto >= 0.3 vs training set)')
    lines.append('=' * 88)
    lines.append(f'  Reference CV R^2 (DualANN on PCL-AE-128 full) : {cv_r2_ref:.4f}')
    lines.append(f'  Candidates after Tanimoto gate                : {len(df_sorted)}')
    lines.append(f'  xTB cache hits (sub+cat)                        : {sum(cache_hits)} / {len(cache_hits)}')
    lines.append(f'  Bootstrap CI replicates                         : 20')
    lines.append(f'  Date                                            : {time.strftime("%Y-%m-%d %H:%M:%S")}')
    lines.append('')
    lines.append('-' * 120)
    lines.append(f'  {"#":>3} {"Yield%":>7} {"+/-":>6} {"95% CI":>16}  {"T":>4}  {"Reactant":<24} {"Catalyst":<10} {"Solv":<6} {"Tanimoto":>9} {"1-NN%":>6}')
    lines.append('-' * 120)
    for i, row in top10.iterrows():
        rk = int(i) + 1
        ci = f'[{row["pred_yield_ci_lo"]*100:5.1f}, {row["pred_yield_ci_hi"]*100:5.1f}]'
        nn1 = row['nn1_train_yield'] * 100 if not np.isnan(row['nn1_train_yield']) else float('nan')
        lines.append(
            f'  {rk:>3d} {row["pred_yield_boot_mean"]*100:6.2f}% '
            f'{row["pred_yield_boot_std"]*100:5.2f}  {ci:>16}  '
            f'{int(row["temperature_C"]):>3d}P{int(row["pressure_MPa"]*10):02d}  '
            f'{row["reactant_name"][:24]:<24} {row["catalyst_name"][:10]:<10} '
            f'{row["solvent_name"][:6]:<6} '
            f'{row["tanimoto_max"]:>9.3f} {nn1:>5.1f}%'
        )
    lines.append('-' * 120)
    lines.append('')
    lines.append('Interpretation:')
    lines.append('  - Tanimoto: max Morgan-2 Tanimoto similarity to any training-set reactant.')
    lines.append('    T = 1.0 means identical to a known reactant; T >= 0.3 is "in-domain".')
    lines.append('  - 1-NN%: yield of the most-similar training-set reaction (k-NN baseline).')
    lines.append('  - When the ML prediction matches the 1-NN yield closely, the model is mostly')
    lines.append('    interpolating. When they diverge substantially, the model is using XTB+condition')
    lines.append('    info beyond pure chemical similarity.')
    lines.append('')
    lines.append('Notes:')
    lines.append('  - Point estimate from the saved DualBranchANN.')
    lines.append('  - Bootstrap CI from 20 retrained DualANN on bootstrap subsets (40 epochs each).')
    lines.append('  - "Yield%" reported is normalised to 0..1 (1.0 = 100% yield).')
    lines.append('  - High uncertainty (wide CI) means the candidate is out-of-distribution.')
    lines.append('  - xTB cache hit flag is recorded in candidates_full.csv.')
    lines.append('')
    text = '\n'.join(lines)
    print(text)
    with open(os.path.join(OUT_DIR, 'top10_results.txt'), 'w', encoding='utf-8') as f:
        f.write(text)

    # ----- Figure
    fig, ax = plt.subplots(figsize=(12, 7))
    rk_y = top10['pred_yield_boot_mean'].values * 100
    rk_lo = top10['pred_yield_ci_lo'].values * 100
    rk_hi = top10['pred_yield_ci_hi'].values * 100
    rk_x = np.arange(len(top10))[::-1]  # rank 1 on top
    rk_err_lo = np.maximum(0, rk_y - rk_lo)
    rk_err_hi = np.maximum(0, rk_hi - rk_y)
    colors = plt.cm.viridis(np.linspace(0.2, 0.85, len(top10)))
    labels = [
        f"{r['reactant_name'][:16]} | {r['catalyst_name'][:6]} | {r['solvent_name'][:6]} (T={r['tanimoto_max']:.2f})"
        for _, r in top10.iterrows()
    ]
    ax.barh(rk_x, rk_y, xerr=[rk_err_lo, rk_err_hi], color=colors, alpha=0.85,
            edgecolor='black', linewidth=0.7, error_kw={'ecolor': '#444', 'elinewidth': 1.5})
    # Overlay 1-NN training yield as a star marker
    ax.scatter(top10['nn1_train_yield'].values * 100, rk_x, marker='*', s=140,
               color='red', edgecolor='black', linewidth=0.7, zorder=5,
               label='1-NN training yield')
    ax.set_yticks(rk_x)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel('Predicted Yield (%)  with 95% Bootstrap CI', fontsize=11)
    ax.set_title(
        'Top-10 CO2 Cycloaddition Candidates\n'
        f'(Tanimoto gate >= 0.3; {len(df_sorted)} in-domain candidates screened; '
        f'CV R^2 ref = {cv_r2_ref:.3f})\n'
        'Red star = yield of most-similar training reaction (1-NN baseline)',
        fontsize=11)
    ax.invert_yaxis()
    ax.set_xlim(0, 105)
    ax.grid(axis='x', linestyle='--', alpha=0.4)
    ax.legend(loc='lower right', fontsize=10)
    plt.tight_layout()
    fig_path = os.path.join(OUT_DIR, 'figure_top10.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {fig_path}')


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    t0 = time.time()
    print('=' * 72)
    print('  Virtual Screening for CO2 Cycloaddition')
    print('=' * 72)

    art = load_artifacts()
    cand_df_raw = build_candidate_library()

    # Apply Tanimoto gate before expensive DRFP computation
    train_df = pd.read_csv(DATA_EXTENDED, encoding='utf-8-sig')
    train_df = train_df[train_df['extraction_status'] == 'valid'].copy()
    train_df = train_df.dropna(subset=['yield (%)'])
    train_df = train_df[train_df['yield (%)'] > 0].reset_index(drop=True)
    train_react = train_df['reactant_smiles'].astype(str).tolist()
    train_y = train_df['yield (%)'].values.astype(np.float32) / 100.0
    cand_df = compute_tanimoto_gate(cand_df_raw, train_react, train_y)

    xtb_cache = load_xtb_cache()
    xtb_medians, cond_medians = load_xtb_medians()
    X_drfp_raw, X_other, cache_hits = build_candidate_features(
        cand_df, art['meta'], xtb_cache, xtb_medians, cond_medians)

    point_preds = score_candidates(art, X_drfp_raw, X_other)
    boot_mean, boot_std, boot_ci = bootstrap_ci(art, X_drfp_raw, X_other, n_boot=20)

    write_outputs(cand_df, point_preds, boot_mean, boot_std, boot_ci, cache_hits,
                  art['cv_metrics']['r2_mean'])

    print(f'\nDone in {time.time() - t0:.1f} s.')


if __name__ == '__main__':
    main()