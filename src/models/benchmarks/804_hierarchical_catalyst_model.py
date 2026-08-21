# -*- coding: utf-8 -*-
"""
804_hierarchical_catalyst_model.py
====================================

Hierarchical / stratified catalyst-family models for problem 8.

The original DualBranchANN is trained on the full 2,490-reaction dataset
and pooled R^2 is reported. This script trains **separate** models on
each major catalyst family and compares per-family R^2 against the
pooled R^2.

Catalyst families and sample counts:
  - ionic_liquid   : 1,940 reactions
  - metal_halide   :   199 reactions
  - mixed_system   :   161 reactions
  - organic_base   :    93 reactions  (too small for stratified model)
  - unknown        :    97 reactions  (excluded)

Workflow
--------
1. Split the dataset by `catalyst_system_type`.
2. Train a per-family DualBranchANN (or fall back to RF if the
   family sample size is < {MIN_SAMPLES_FOR_NN}).
3. Compute per-family 5x2 KFold R^2 and per-family Pearson r.
4. Compute a "weighted-average" R^2 using the family sample sizes as
   weights, and compare against the pooled R^2.

Outputs (results_hierarchical_model/):
  - per_family_metrics.csv
  - pooled_vs_hierarchical.csv
  - figure_hierarchical_compare.png
  - hierarchical_report.txt

Runtime: ~2 minutes (3 family models * 5x2 folds * 60 epochs).

Usage:
    python 804_hierarchical_catalyst_model.py
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
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = os.environ.get("CO2_PROJECT_ROOT", r"D:\machine-learning\CO2-cycloaddition")
sys.path.insert(0, PROJECT_ROOT)
from utils_rxn import read_drfp, XTB_COLS
from utils_features import COND_COLS, find_cond_cols

DATA_EXTENDED = os.path.join(PROJECT_ROOT, 'results', 'results_cho_diagnostic', 'co2_drfp_xtb_extended.csv')
OUT_DIR = os.path.join(PROJECT_ROOT, 'results_hierarchical_model')
os.makedirs(OUT_DIR, exist_ok=True)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

FAMILIES = ['ionic_liquid', 'metal_halide', 'mixed_system']
MIN_SAMPLES_FOR_NN = 300   # below this, fall back to RF (199 is too few)


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
    if 'activation_proxy' in XTB_COLS and 'total_polarity_index' in XTB_COLS:
        ai = XTB_COLS.index('activation_proxy')
        tpi = XTB_COLS.index('total_polarity_index')
        X_inter = np.concatenate([
            X_cond[:, 0:1] * X_xtb[:, ai:ai + 1],
            X_cond[:, 1:2] * X_xtb[:, tpi:tpi + 1],
        ], axis=1).astype(np.float32)
    else:
        X_inter = np.zeros((len(df), 2), dtype=np.float32)
    X_other = np.concatenate([X_xtb, X_cond, X_inter], axis=1).astype(np.float32)
    cat_family = df['catalyst_system_type'].fillna('unknown').values
    print(f'  loaded {len(df)} reactions, '
          f'DRFP={X_drfp.shape}, Other={X_other.shape}')
    print(f'  family distribution:')
    for fam, cnt in pd.Series(cat_family).value_counts().items():
        print(f'    {fam:<15s}: {cnt}')
    return df, X_drfp, X_other, y, cat_family


# ----------------------------------------------------------------------
# 2. DualBranchANN
# ----------------------------------------------------------------------

class DualBranchANN(nn.Module):
    def __init__(self, drfp_dim, xtb_dim, hidden=64):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.bn_drfp = nn.BatchNorm1d(32)
        self.fc_drfp = nn.Sequential(
            nn.Linear(32, hidden), nn.LeakyReLU(0.1), nn.Dropout(0.3),
            nn.Linear(hidden, 32), nn.LeakyReLU(0.1),
        )
        self.fc_xtb = nn.Sequential(
            nn.Linear(xtb_dim, 64), nn.BatchNorm1d(64), nn.LeakyReLU(0.1), nn.Dropout(0.3),
            nn.Linear(64, 32), nn.LeakyReLU(0.1), nn.Dropout(0.2),
        )
        self.fc_out = nn.Sequential(
            nn.Linear(32 + 32, 32), nn.BatchNorm1d(32), nn.LeakyReLU(0.1),
            nn.Linear(32, 1),
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
# 3. Per-family evaluation
# ----------------------------------------------------------------------

def eval_per_family(X_drfp, X_other, y, mask, family_name: str) -> Dict:
    """5x2 KFold CV on the family subset.  Choose DualBranchANN or RF
    depending on sample size."""
    n = int(mask.sum())
    print(f'\n  --- Family "{family_name}" (n={n}) ---')
    if n < 20:
        print(f'  Too few samples, skipping')
        return {'family': family_name, 'n': n,
                'r2_mean': np.nan, 'mae_mean': np.nan,
                'pearson_mean': np.nan, 'model': 'skipped'}
    use_nn = n >= MIN_SAMPLES_FOR_NN
    model_name = 'DualANN' if use_nn else 'RF'
    print(f'  Using model: {model_name}')

    X_drfp_f = X_drfp[mask]
    X_other_f = X_other[mask]
    y_f = y[mask]

    fold_metrics = []
    for split_seed in range(5):
        kf = KFold(n_splits=2, shuffle=True, random_state=SEED + split_seed)
        for fold, (tr, va) in enumerate(kf.split(X_drfp_f)):
            sc_drfp = StandardScaler()
            sc_xtb = StandardScaler()
            Xd_tr = sc_drfp.fit_transform(X_drfp_f[tr]).astype(np.float32)
            Xd_va = sc_drfp.transform(X_drfp_f[va]).astype(np.float32)
            Xo_tr = sc_xtb.fit_transform(X_other_f[tr]).astype(np.float32)
            Xo_va = sc_xtb.transform(X_other_f[va]).astype(np.float32)
            y_tr = y_f[tr]; y_va = y_f[va]

            if use_nn:
                Xd_tr_t = torch.tensor(Xd_tr, dtype=torch.float32).to(DEVICE)
                Xo_tr_t = torch.tensor(Xo_tr, dtype=torch.float32).to(DEVICE)
                Xd_va_t = torch.tensor(Xd_va, dtype=torch.float32).to(DEVICE)
                Xo_va_t = torch.tensor(Xo_va, dtype=torch.float32).to(DEVICE)
                y_tr_t = torch.tensor(y_tr, dtype=torch.float32).to(DEVICE)

                model = DualBranchANN(Xd_tr.shape[1], Xo_tr.shape[1]).to(DEVICE)
                opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
                ds = TensorDataset(Xd_tr_t, Xo_tr_t, y_tr_t)
                dl = DataLoader(ds, batch_size=32, shuffle=True)

                best_loss, patience, wait = float('inf'), 8, 0
                for ep in range(60):
                    model.train()
                    for xd, xo, yy in dl:
                        opt.zero_grad()
                        nn.MSELoss()(model(xd, xo), yy).backward()
                        opt.step()
                    # early stopping on training loss
                    cur_loss = sum(nn.MSELoss()(model(xd, xo), yy).item()
                                   for xd, xo, yy in dl) / len(dl)
                    if cur_loss < best_loss - 1e-6:
                        best_loss = cur_loss
                        wait = 0
                    else:
                        wait += 1
                        if wait >= patience:
                            break
                model.eval()
                with torch.no_grad():
                    pred = model(Xd_va_t, Xo_va_t).cpu().numpy()
            else:
                # RF baseline
                rf = RandomForestRegressor(n_estimators=200, random_state=SEED,
                                             max_depth=10, min_samples_leaf=5,
                                             n_jobs=-1)
                Xf_tr = np.hstack([Xd_tr, Xo_tr])
                Xf_va = np.hstack([Xd_va, Xo_va])
                rf.fit(Xf_tr, y_tr)
                pred = rf.predict(Xf_va)

            r2 = r2_score(y_va, pred) if len(np.unique(y_va)) > 1 else np.nan
            mae = mean_absolute_error(y_va, pred)
            pearson, _ = pearsonr(y_va, pred) if len(np.unique(y_va)) > 1 else np.nan
            fold_metrics.append({'r2': r2, 'mae': mae, 'pearson': pearson})

    df = pd.DataFrame(fold_metrics)
    summary = {
        'family': family_name,
        'n': n,
        'model': model_name,
        'r2_mean': float(df['r2'].mean(skipna=True)),
        'r2_std': float(df['r2'].std(skipna=True)),
        'mae_mean': float(df['mae'].mean()),
        'mae_std': float(df['mae'].std()),
        'pearson_mean': float(df['pearson'].mean(skipna=True)),
        'pearson_std': float(df['pearson'].std(skipna=True)),
    }
    print(f'  Family {family_name}: R^2 = {summary["r2_mean"]:.4f} ± {summary["r2_std"]:.4f}, '
          f'MAE = {summary["mae_mean"]:.4f}, Pearson = {summary["pearson_mean"]:.4f}')
    return summary


# ----------------------------------------------------------------------
# 4. Pooled baseline (DualBranchANN on full dataset)
# ----------------------------------------------------------------------

def eval_pooled(X_drfp, X_other, y) -> Dict:
    """5x2 KFold CV on the full dataset with DualBranchANN."""
    print(f'\n  --- Pooled (n={len(y)}) ---')
    fold_metrics = []
    for split_seed in range(5):
        kf = KFold(n_splits=2, shuffle=True, random_state=SEED + split_seed)
        for fold, (tr, va) in enumerate(kf.split(X_drfp)):
            sc_drfp = StandardScaler()
            sc_xtb = StandardScaler()
            Xd_tr = sc_drfp.fit_transform(X_drfp[tr]).astype(np.float32)
            Xd_va = sc_drfp.transform(X_drfp[va]).astype(np.float32)
            Xo_tr = sc_xtb.fit_transform(X_other[tr]).astype(np.float32)
            Xo_va = sc_xtb.transform(X_other[va]).astype(np.float32)
            y_tr = y[tr]; y_va = y[va]

            Xd_tr_t = torch.tensor(Xd_tr, dtype=torch.float32).to(DEVICE)
            Xo_tr_t = torch.tensor(Xo_tr, dtype=torch.float32).to(DEVICE)
            Xd_va_t = torch.tensor(Xd_va, dtype=torch.float32).to(DEVICE)
            Xo_va_t = torch.tensor(Xo_va, dtype=torch.float32).to(DEVICE)
            y_tr_t = torch.tensor(y_tr, dtype=torch.float32).to(DEVICE)

            model = DualBranchANN(Xd_tr.shape[1], Xo_tr.shape[1]).to(DEVICE)
            opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
            ds = TensorDataset(Xd_tr_t, Xo_tr_t, y_tr_t)
            dl = DataLoader(ds, batch_size=64, shuffle=True)

            best_loss, patience, wait = float('inf'), 8, 0
            for ep in range(60):
                model.train()
                for xd, xo, yy in dl:
                    opt.zero_grad()
                    nn.MSELoss()(model(xd, xo), yy).backward()
                    opt.step()
                cur_loss = sum(nn.MSELoss()(model(xd, xo), yy).item()
                               for xd, xo, yy in dl) / len(dl)
                if cur_loss < best_loss - 1e-6:
                    best_loss = cur_loss
                    wait = 0
                else:
                    wait += 1
                    if wait >= patience:
                        break
            model.eval()
            with torch.no_grad():
                pred = model(Xd_va_t, Xo_va_t).cpu().numpy()

            r2 = r2_score(y_va, pred)
            mae = mean_absolute_error(y_va, pred)
            pearson, _ = pearsonr(y_va, pred)
            fold_metrics.append({'r2': r2, 'mae': mae, 'pearson': pearson})

    df = pd.DataFrame(fold_metrics)
    summary = {
        'family': 'pooled',
        'n': len(y),
        'model': 'DualANN',
        'r2_mean': float(df['r2'].mean()),
        'r2_std': float(df['r2'].std()),
        'mae_mean': float(df['mae'].mean()),
        'mae_std': float(df['mae'].std()),
        'pearson_mean': float(df['pearson'].mean()),
        'pearson_std': float(df['pearson'].std()),
    }
    print(f'  Pooled: R^2 = {summary["r2_mean"]:.4f} ± {summary["r2_std"]:.4f}')
    return summary


# ----------------------------------------------------------------------
# 5. Plot + report
# ----------------------------------------------------------------------

def plot_compare(per_family, pooled, out_path):
    families = per_family['family'].tolist()
    r2 = per_family['r2_mean'].tolist()
    r2s = per_family['r2_std'].tolist()
    pooled_r2 = pooled['r2_mean']
    pooled_r2s = pooled['r2_std']
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(families) + 1)
    y = r2 + [pooled_r2]
    e = r2s + [pooled_r2s]
    cols = ['steelblue'] * len(families) + ['#CC6677']
    ax.bar(x, y, yerr=e, color=cols, edgecolor='black', capsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(families + ['pooled'], rotation=15)
    ax.set_ylabel('R² (5x2 KFold)')
    ax.set_title('Per-family vs pooled R²: hierarchical catalyst model')
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    for i, (v, s) in enumerate(zip(y, e)):
        ax.text(i, v + s + 0.005, f'{v:.3f}', ha='center', fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  saved {out_path}')


def write_report(per_family, pooled, out_path):
    valid = per_family.dropna(subset=['r2_mean'])
    weighted_r2 = float(np.average(valid['r2_mean'].values,
                                     weights=valid['n'].values))
    lines = []
    lines.append('=' * 88)
    lines.append('  Hierarchical Catalyst Model: Per-family vs Pooled R²')
    lines.append('=' * 88)
    lines.append('')
    lines.append('PER-FAMILY R² (5x2 KFold CV)')
    lines.append('-' * 60)
    lines.append(per_family[['family', 'n', 'model', 'r2_mean', 'mae_mean',
                              'pearson_mean']].to_string(index=False))
    lines.append('')
    lines.append('POOLED BASELINE (DualANN on all 2,490 reactions)')
    lines.append('-' * 60)
    lines.append(f'  R² = {pooled["r2_mean"]:.4f} ± {pooled["r2_std"]:.4f}')
    lines.append(f'  MAE = {pooled["mae_mean"]:.4f} ± {pooled["mae_std"]:.4f}')
    lines.append(f'  Pearson = {pooled["pearson_mean"]:.4f} ± {pooled["pearson_std"]:.4f}')
    lines.append('')
    lines.append('WEIGHTED-AVERAGE PER-FAMILY R²')
    lines.append('-' * 60)
    lines.append(f'  Weighted R² = {weighted_r2:.4f}')
    lines.append(f'  (vs pooled R² = {pooled["r2_mean"]:.4f})')
    lines.append(f'  Δ = {weighted_r2 - pooled["r2_mean"]:+.4f}')
    lines.append('')
    lines.append('INTERPRETATION')
    lines.append('-' * 60)
    lines.append('A positive Δ means the hierarchical (per-family) models recover')
    lines.append('more variance than the pooled model.  A negative Δ means the pooled')
    lines.append('model already exploits inter-family regularities (transfer learning).')
    if weighted_r2 > pooled['r2_mean'] + 0.02:
        verdict = 'HIERARCHICAL HELPS — per-family models improve R² by >= 0.02.'
    elif weighted_r2 > pooled['r2_mean']:
        verdict = 'HIERARCHICAL MARGINAL — small improvement, but within CV noise.'
    else:
        verdict = 'POOLED IS BETTER — the hierarchical split loses information.'
    lines.append(f'VERDICT: {verdict}')
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
    print('  804 -- Hierarchical Catalyst Model')
    print('=' * 72)

    df, X_drfp, X_other, y, cat_family = load_data()

    # Per-family eval
    print('\n[1/4] Per-family 5x2 KFold CV ...')
    rows = []
    for fam in FAMILIES:
        mask = (cat_family == fam)
        summary = eval_per_family(X_drfp, X_other, y, mask, fam)
        rows.append(summary)
    per_family = pd.DataFrame(rows)
    per_family.to_csv(os.path.join(OUT_DIR, 'per_family_metrics.csv'),
                       index=False, encoding='utf-8-sig')

    # Pooled baseline
    print('\n[2/4] Pooled 5x2 KFold CV ...')
    pooled = eval_pooled(X_drfp, X_other, y)

    # Save pooled vs hierarchical
    pooled_df = pd.DataFrame([pooled])
    pooled_vs_h = pd.concat([per_family, pooled_df], ignore_index=True)
    pooled_vs_h.to_csv(os.path.join(OUT_DIR, 'pooled_vs_hierarchical.csv'),
                         index=False, encoding='utf-8-sig')

    # Plot
    print('\n[3/4] Plot comparison ...')
    plot_compare(per_family, pooled,
                  os.path.join(OUT_DIR, 'figure_hierarchical_compare.png'))

    print('\n[4/4] Writing report ...')
    write_report(per_family, pooled,
                 os.path.join(OUT_DIR, 'hierarchical_report.txt'))

    print(f'\nDone in {time.time() - t0:.1f} s.')


if __name__ == '__main__':
    main()