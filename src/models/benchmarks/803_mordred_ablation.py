# -*- coding: utf-8 -*-
"""
803_mordred_ablation.py
===================================

Adds a panel of 30 "general-purpose" substrate molecular descriptors
computed by Mordred (2D) and quantifies whether the DualBranchANN R^2
improves when these descriptors are concatenated to the DRFP + xTB
feature stack.

This is the "path A" fix for the two questions:
  - Q2 (substrate extrapolation): the model fails because DRFP only
    encodes the substrate via topology (no logP, no PSA, no HBD/HBA
    complexity).  Adding Mordred enriches the substrate fingerprint
    with chemistry-relevant 2D descriptors.
  - Q4 (R^2 = 0.33 on JC is on the low end): Mordred's 30 general
    descriptors add orthogonal information to the xTB electronic
    descriptors, so the DualBranchANN second branch has more to work
    with.

Workflow
--------
  1. Load the curated dataset (co2_drfp_xtb_extended.csv).
  2. Compute 30 Mordred descriptors per unique reactant SMILES
     (cached on disk).
  3. Build feature matrix:  DRFP(no_cats) + xTB + cond + inter + mordred
  4. Run 5x2 KFold R^2 on DualBranchANN with the original feature set
     (baseline) and the new feature set (with Mordred).
  5. Compute the delta R^2 and report.

The Mordred panel is curated to be chemistry-relevant for epoxide
cycloaddition:
  - Lipophilicity: MolLogP, MolMR
  - Polarity:     TopoPSA, LabuteASA
  - H-bonding:    nHBAcc, nHBDon
  - Complexity:   BertzCT, BalabanJ, HallKierAlpha
  - Ring count:   nRing, n3Ring, n5Ring, n6Ring, nAromRing
  - Heteroatoms:  nHetero, nN, nO, nS, nF, nCl, nBr, nI
  - Atom counts:  nAtom, nHeavyAtom, nH, nC, nB
  - Connectivity: Chi0, Chi1, Chi2v, Chi3v, Chi4v (Kier-Hall)

Cache: _mordred_substrate_cache.npy (one entry per unique SMILES).

Outputs (results_mordred_ablation/):
  - mordred_panel.csv             (sanity dump of the 30 descriptors)
  - mordred_dataset.csv           (2338 x 30 feature matrix joined to rows)
  - ablation_results.csv          (R^2, MAE, Pearson, RMSE per config)
  - ablation_summary.txt          (human-readable)
  - figure_mordred_impact.png     (bar chart of delta R^2)

Runtime: ~10 minutes (Mordred 30 desc * 5 unique substrates + 2 configs * 5x2 fold CV).

Usage:
    python 803_mordred_ablation.py
"""

import io
import os
import sys
import time
import warnings
from typing import Dict, List, Tuple

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = os.environ.get("CO2_PROJECT_ROOT", r"D:\machine-learning\CO2-cycloaddition")
sys.path.insert(0, PROJECT_ROOT)
from utils_rxn import read_drfp, XTB_COLS
from utils_features import COND_COLS, find_cond_cols

DATA_EXTENDED = os.path.join(PROJECT_ROOT, 'results/results_cho_diagnostic/co2_drfp_xtb_extended.csv')
OUT_DIR = os.path.join(PROJECT_ROOT, 'results', 'results_mordred_ablation')
os.makedirs(OUT_DIR, exist_ok=True)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)


# ----------------------------------------------------------------------
# 1. Data loader
# ----------------------------------------------------------------------

def load_data():
    df = pd.read_csv(DATA_EXTENDED, encoding='utf-8-sig')
    df = df[df['extraction_status'] == 'valid'].copy()
    df = df.dropna(subset=['yield (%)'])
    df = df[df['yield (%)'] > 0].reset_index(drop=True)
    y = df['yield (%)'].values.astype(np.float32) / 100.0

    arr = []
    for s in df['drfp']:
        fp = read_drfp(s)
        arr.append(np.zeros(2048, dtype=np.float32) if fp is None or fp.size == 0
                   else fp.astype(np.float32))
    X_drfp = np.array(arr, dtype=np.float32)

    X_xtb = np.nan_to_num(df[XTB_COLS].values.astype(np.float32), nan=0.0)
    cond_cols_present = find_cond_cols(df)
    X_cond = np.nan_to_num(df[cond_cols_present].values.astype(np.float32), nan=0.0) \
        if cond_cols_present else np.zeros((len(df), 0), dtype=np.float32)

    # inter = T * activation_proxy + P * total_polarity
    if 'activation_proxy' in XTB_COLS and 'total_polarity_index' in XTB_COLS:
        ai = XTB_COLS.index('activation_proxy')
        tpi = XTB_COLS.index('total_polarity_index')
        X_inter = np.concatenate([
            X_cond[:, 0:1] * X_xtb[:, ai:ai + 1],
            X_cond[:, 1:2] * X_xtb[:, tpi:tpi + 1],
        ], axis=1).astype(np.float32)
    else:
        X_inter = np.zeros((len(df), 2), dtype=np.float32)

    print(f'  loaded {len(df)} reactions, '
          f'DRFP={X_drfp.shape}, XTB={X_xtb.shape}, Cond={X_cond.shape}, '
          f'Inter={X_inter.shape}')
    return df, X_drfp, X_xtb, X_cond, X_inter, y


# ----------------------------------------------------------------------
# 2. Mordred descriptors (curated panel)
# ----------------------------------------------------------------------

MORDRED_PANEL = [
    # Lipophilicity / polarity
    'MolLogP', 'MolMR', 'TopoPSA', 'LabuteASA',
    # H-bonding
    'nHBAcc', 'nHBDon',
    # Complexity
    'BertzCT', 'BalabanJ', 'HallKierAlpha',
    # Ring counts
    'nRing', 'n3Ring', 'n5Ring', 'n6Ring', 'nAromRing',
    # Heteroatoms
    'nHetero', 'nN', 'nO', 'nS', 'nF', 'nCl', 'nBr', 'nI',
    # Atom counts
    'nAtom', 'nHeavyAtom', 'nH', 'nC', 'nB',
    # Connectivity (Kier-Hall)
    'Chi0', 'Chi1', 'Chi2v', 'Chi3v', 'Chi4v',
]
print(f'  Mordred panel: {len(MORDRED_PANEL)} descriptors')


def compute_mordred_for_unique(smiles_list, cache_path):
    """Compute Mordred descriptors for unique substrate SMILES, cache on disk."""
    cache = {}
    if os.path.exists(cache_path):
        try:
            cache = np.load(cache_path, allow_pickle=True).item()
            print(f'  loaded Mordred cache: {len(cache)} entries')
        except Exception:
            cache = {}

    try:
        from mordred import Calculator, descriptors as mordred_descs
    except ImportError:
        print('  [ERROR] mordred not installed; pip install mordred')
        return None

    try:
        from rdkit import Chem
    except ImportError:
        print('  [ERROR] rdkit not installed')
        return None

    calc = Calculator(mordred_descs, ignore_3D=True)
    wanted = [d for d in calc.descriptors if str(d) in MORDRED_PANEL]
    print(f'  Mordred descriptors available: {len(wanted)}/{len(MORDRED_PANEL)}')
    calc = Calculator(wanted, ignore_3D=True)

    unique_smis = sorted(set(s for s in smiles_list if s and isinstance(s, str)))
    print(f'  unique substrate SMILES: {len(unique_smis)}')

    for smi in unique_smis:
        if smi in cache:
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            cache[smi] = None
            continue
        try:
            res = calc(mol)
            vals = {}
            for desc, val in zip(calc.descriptors, res):
                key = str(desc)
                try:
                    v = float(val) if val is not None else np.nan
                    if v != v:  # NaN
                        v = np.nan
                except (ValueError, TypeError):
                    v = np.nan
                vals[key] = v
            cache[smi] = vals
        except Exception as e:
            print(f'    [WARN] Mordred failed for {smi[:30]}: {str(e)[:60]}')
            cache[smi] = None

    np.save(cache_path, cache)
    print(f'  cached {len(cache)} entries')
    return cache


def build_mordred_matrix(df, cache):
    """Build (N, 30) Mordred matrix for the dataset."""
    n = len(df)
    X = np.zeros((n, len(MORDRED_PANEL)), dtype=np.float32)
    for i, smi in enumerate(df['reactant_smiles'].values):
        if not isinstance(smi, str):
            continue
        d = cache.get(smi)
        if d is None:
            continue
        for k, key in enumerate(MORDRED_PANEL):
            if key in d and not np.isnan(d[key]):
                X[i, k] = d[key]
    # NaN -> median imputation
    for k in range(X.shape[1]):
        col = X[:, k]
        nan_mask = np.isnan(col)
        if nan_mask.any():
            med = np.nanmedian(col)
            X[nan_mask, k] = med if not np.isnan(med) else 0.0
    return X


# ----------------------------------------------------------------------
# 3. DualBranchANN architecture (mirrored from 301)
# ----------------------------------------------------------------------

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
# 4. Cross-validation
# ----------------------------------------------------------------------

def run_5x2_cv(X_drfp, X_xtb, y, hidden=128, epochs=80, batch_size=64):
    """5x2 KFold CV on DualBranchANN. X_xtb already includes cond + inter + (optional Mordred)."""
    drfp_dim = X_drfp.shape[1]
    xtb_dim = X_xtb.shape[1]
    fold_metrics = []

    for split_seed in range(5):
        kf = KFold(n_splits=2, shuffle=True, random_state=SEED + split_seed)
        for fold, (tr, va) in enumerate(kf.split(X_drfp)):
            sc_drfp = StandardScaler()
            sc_xtb = StandardScaler()
            Xd_tr = sc_drfp.fit_transform(X_drfp[tr]).astype(np.float32)
            Xd_va = sc_drfp.transform(X_drfp[va]).astype(np.float32)
            Xo_tr = sc_xtb.fit_transform(X_xtb[tr]).astype(np.float32)
            Xo_va = sc_xtb.transform(X_xtb[va]).astype(np.float32)
            y_tr = y[tr]; y_va = y[va]

            Xd_tr_t = torch.tensor(Xd_tr, dtype=torch.float32).to(DEVICE)
            Xo_tr_t = torch.tensor(Xo_tr, dtype=torch.float32).to(DEVICE)
            Xd_va_t = torch.tensor(Xd_va, dtype=torch.float32).to(DEVICE)
            Xo_va_t = torch.tensor(Xo_va, dtype=torch.float32).to(DEVICE)
            y_tr_t = torch.tensor(y_tr, dtype=torch.float32).to(DEVICE)
            y_va_t = torch.tensor(y_va, dtype=torch.float32).to(DEVICE)

            model = DualBranchANN(drfp_dim, xtb_dim, hidden=hidden).to(DEVICE)
            opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
            ds = TensorDataset(Xd_tr_t, Xo_tr_t, y_tr_t)
            dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

            for ep in range(epochs):
                model.train()
                for xd, xo, yy in dl:
                    opt.zero_grad()
                    nn.MSELoss()(model(xd, xo), yy).backward()
                    opt.step()
            model.eval()
            with torch.no_grad():
                pred = model(Xd_va_t, Xo_va_t).cpu().numpy()

            r2 = r2_score(y_va, pred)
            mae = mean_absolute_error(y_va, pred)
            rmse = float(np.sqrt(mean_squared_error(y_va, pred)))
            pearson, _ = pearsonr(y_va, pred)
            fold_metrics.append({'r2': r2, 'mae': mae, 'rmse': rmse, 'pearson': pearson})
            print(f'    fold {split_seed}.{fold}  R虏={r2:.4f}  MAE={mae:.4f}')

    df = pd.DataFrame(fold_metrics)
    return {
        'r2_mean': df['r2'].mean(), 'r2_std': df['r2'].std(),
        'mae_mean': df['mae'].mean(), 'mae_std': df['mae'].std(),
        'rmse_mean': df['rmse'].mean(), 'rmse_std': df['rmse'].std(),
        'pearson_mean': df['pearson'].mean(), 'pearson_std': df['pearson'].std(),
    }


# ----------------------------------------------------------------------
# 5. Reporting
# ----------------------------------------------------------------------

def plot_impact(baseline, mordred, out_path):
    metrics = ['r2', 'mae', 'pearson']
    labels = ['R虏', 'MAE', 'Pearson r']
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, m, lab in zip(axes, metrics, labels):
        v_b = baseline[f'{m}_mean']; s_b = baseline[f'{m}_std']
        v_m = mordred[f'{m}_mean']; s_m = mordred[f'{m}_std']
        ax.bar(['Baseline', 'Baseline + Mordred'], [v_b, v_m],
                yerr=[s_b, s_m], color=['#4477AA', '#CC6677'],
                edgecolor='black', capsize=8)
        ax.set_title(f'{lab} (5x2 KFold)')
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        if m == 'r2' or m == 'pearson':
            ax.set_ylim(0, max(v_b, v_m) * 1.4 + 0.05)
        for i, (v, s) in enumerate([(v_b, s_b), (v_m, s_m)]):
            ax.text(i, v + s, f'{v:.3f}卤{s:.3f}', ha='center', fontsize=10)
    fig.suptitle('Impact of adding Mordred substrate descriptors '
                  '(5x2 KFold, n=2,338)', fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  saved {out_path}')


def write_report(baseline, mordred, out_path):
    delta_r2 = mordred['r2_mean'] - baseline['r2_mean']
    delta_mae = mordred['mae_mean'] - baseline['mae_mean']
    delta_pearson = mordred['pearson_mean'] - baseline['pearson_mean']

    lines = []
    lines.append('=' * 88)
    lines.append('  Mordred substrate descriptor ablation report')
    lines.append('=' * 88)
    lines.append('')
    lines.append('AIM')
    lines.append('---')
    lines.append('The original feature stack (DRFP + xTB + cond + inter) achieves 5x2 KFold R^2')
    lines.append(f'= {baseline["r2_mean"]:.4f} +/- {baseline["r2_std"]:.4f}. This script re-runs the')
    lines.append('same evaluation with the same dual-branch architecture but with a 30-dimensional')
    lines.append('Mordred substrate descriptor panel concatenated to the xTB branch, and quantifies')
    lines.append('the delta R^2.')
    lines.append('')
    lines.append('BASELINE  (DRFP-no_cats + xTB + cond + inter)')
    lines.append('-' * 60)
    lines.append(f'  R^2     = {baseline["r2_mean"]:.4f} +/- {baseline["r2_std"]:.4f}')
    lines.append(f'  MAE     = {baseline["mae_mean"]:.4f} +/- {baseline["mae_std"]:.4f}')
    lines.append(f'  RMSE    = {baseline["rmse_mean"]:.4f} +/- {baseline["rmse_std"]:.4f}')
    lines.append(f'  Pearson = {baseline["pearson_mean"]:.4f} +/- {baseline["pearson_std"]:.4f}')
    lines.append('')
    lines.append('+ MORDRED  (DRFP-no_cats + xTB + cond + inter + 30 Mordred)')
    lines.append('-' * 60)
    lines.append(f'  R^2     = {mordred["r2_mean"]:.4f} +/- {mordred["r2_std"]:.4f}')
    lines.append(f'  MAE     = {mordred["mae_mean"]:.4f} +/- {mordred["mae_std"]:.4f}')
    lines.append(f'  RMSE    = {mordred["rmse_mean"]:.4f} +/- {mordred["rmse_std"]:.4f}')
    lines.append(f'  Pearson = {mordred["pearson_mean"]:.4f} +/- {mordred["pearson_std"]:.4f}')
    lines.append('')
    lines.append('DELTA')
    lines.append('-' * 60)
    lines.append(f'  Delta R^2     = {delta_r2:+.4f}')
    lines.append(f'  Delta MAE     = {delta_mae:+.4f}')
    lines.append(f'  Delta Pearson = {delta_pearson:+.4f}')
    lines.append('')
    if delta_r2 > 0.02:
        verdict = 'STATISTICALLY MEANINGFUL 鈥?keep Mordred in the final pipeline.'
    elif delta_r2 > 0.005:
        verdict = 'MARGINAL 鈥?keep Mordred only if R^2 variance is also reduced.'
    elif delta_r2 > 0.0:
        verdict = 'NEGLIGIBLE 鈥?Mordred does not improve the regression.'
    else:
        verdict = 'NEGATIVE 鈥?Mordred HURTS the regression. Drop it and report the null result.'
    lines.append(f'VERDICT: {verdict}')
    lines.append('')
    lines.append('DECISION RULE')
    lines.append('-------------')
    lines.append('If delta R^2 > 0.02 (a 2 percentage-point improvement),')
    lines.append('the Mordred panel is kept and becomes part of the xTB branch input.')
    lines.append('If 0.005 < delta R^2 < 0.02, the panel is included but described as')
    lines.append('"marginal orthogonal chemistry information" in the paper.')
    lines.append('If delta R^2 < 0.005, the panel is dropped and the ablation is')
    lines.append('reported as a NEGATIVE result (Mordred does not improve R^2).')
    lines.append('')
    lines.append('RATIONALE')
    lines.append('---------')
    lines.append('The Mordred descriptors are 2D general-purpose chemistry descriptors; they')
    lines.append('are *orthogonal* to the xTB electronic descriptors (which are 3D / orbital).')
    lines.append('If the R^2 gain is negative, the orthogonal information is either already')
    lines.append('captured by the xTB branch or is too noisy to help the regression.')
    lines.append('')

    text = '\n'.join(lines)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(text)
    print(f'  saved {out_path}')
    print('\n' + text[:3000])


# ----------------------------------------------------------------------
# 6. Main
# ----------------------------------------------------------------------

def main():
    t0 = time.time()
    print('=' * 72)
    print('  903 -- Mordred substrate descriptor ablation')
    print('=' * 72)

    df, X_drfp, X_xtb, X_cond, X_inter, y = load_data()

    print('\n[1/4] Computing Mordred descriptors ...')
    cache_path = os.path.join(PROJECT_ROOT, '_mordred_substrate_cache.npy')
    cache = compute_mordred_for_unique(df['reactant_smiles'].tolist(), cache_path)
    if cache is None:
        print('  [ERROR] Mordred not available, aborting')
        return
    X_mordred = build_mordred_matrix(df, cache)
    print(f'  Mordred matrix: {X_mordred.shape}, '
          f'non-zero mean = {X_mordred.mean():.4f}')
    pd.DataFrame(X_mordred, columns=MORDRED_PANEL).to_csv(
        os.path.join(OUT_DIR, 'mordred_panel.csv'), index=False, encoding='utf-8-sig')

    # baseline: xTB + cond + inter
    X_xtb_base = np.concatenate([X_xtb, X_cond, X_inter], axis=1).astype(np.float32)
    print(f'\n  Baseline X_xtb shape: {X_xtb_base.shape}')

    # augmented: xTB + cond + inter + mordred
    X_xtb_mord = np.concatenate([X_xtb_base, X_mordred], axis=1).astype(np.float32)
    print(f'  Mordred-augmented X_xtb shape: {X_xtb_mord.shape}')

    print('\n[2/4] Running 5x2 KFold CV on baseline ...')
    baseline = run_5x2_cv(X_drfp, X_xtb_base, y)

    print('\n[3/4] Running 5x2 KFold CV on baseline + Mordred ...')
    mordred = run_5x2_cv(X_drfp, X_xtb_mord, y)

    print('\n[4/4] Writing outputs ...')
    metrics_df = pd.DataFrame([
        {'config': 'baseline', **baseline},
        {'config': 'baseline+mordred', **mordred},
    ])
    metrics_df.to_csv(os.path.join(OUT_DIR, 'ablation_results.csv'),
                       index=False, encoding='utf-8-sig')

    plot_impact(baseline, mordred, os.path.join(OUT_DIR, 'figure_mordred_impact.png'))
    write_report(baseline, mordred, os.path.join(OUT_DIR, 'ablation_summary.txt'))

    # also save the joined dataset
    mordred_df = pd.DataFrame(X_mordred, columns=[f'mordred_{n}' for n in MORDRED_PANEL])
    pd.concat([df.reset_index(drop=True), mordred_df], axis=1).to_csv(
        os.path.join(OUT_DIR, 'mordred_dataset.csv'), index=False, encoding='utf-8-sig')

    print(f'\nDone in {time.time() - t0:.1f} s.')


if __name__ == '__main__':
    main()
