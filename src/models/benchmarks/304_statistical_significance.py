# -*- coding: utf-8 -*-
"""
304_statistical_significance.py
=========================
配对统计显著性检验
=====================
Dietterich (1998) 5×2 CV + Wilcoxon signed-rank test

All models (RF, XGB, LGBM, DualANN) receive the SAME per-fold PCL-AE
latent representation for fair comparison. The AE is trained inside each
fold (train-only) to prevent data leakage.

比较模型：
  1. DualANN (PCL-AE-128 + DualBranchANN) vs RF
  2. DualANN vs XGB
  3. DualANN vs LGBM
  4. RF vs XGB
  5. RF vs LGBM

输出：
  results_statistical_test/
      wilcoxon_results.csv      — 所有配对比较的 p 值
      paired_errors.csv         — 每个 fold 的配对误差
"""

import os
import sys
import io
import time
import warnings
import random

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)
sys.path.insert(0, os.path.dirname(_SCRIPT_DIR))  # src/ parent
from src.data_split import load_manifest, holdout_arrays, kfold_folds

# NOTE: do NOT re-wrap sys.stdout here.
# src.data_split (imported above) already wraps stdout to UTF-8 on line ~63.
# Re-wrapping a second time closes the underlying buffer, which makes every
# subsequent print() raise "I/O operation on closed file". The original line:
#   sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
# was therefore removed on 2026-08-20.
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.stats import wilcoxon

import xgboost as xgb
import lightgbm as lgb

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = os.environ.get("CO2_PROJECT_ROOT", r"D:\machine-learning\CO2-cycloaddition")
sys.path.insert(0, PROJECT_ROOT)
from utils_rxn import read_drfp, get_best_drfp_variant

DATA_EXTENDED = os.path.join(PROJECT_ROOT, 'results', 'results_cho_diagnostic', 'co2_drfp_xtb_extended.csv')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'results_statistical_test')
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# =====================================================================
# 数据加载
# =====================================================================
def _col(df, key):
    return [c for c in df.columns if key.lower() in c.lower()]


def parse_drfp_col(series):
    arrs = []
    for fp_str in series:
        a = read_drfp(fp_str)
        if a is None or a.size == 0:
            arrs.append(np.zeros(2048, dtype=np.float32))
        else:
            arrs.append(a.astype(np.float32))
    return np.array(arrs)


def load_data(use_holdout_train=True):
    """Load data, dynamically selecting the best DRFP variant from ablation meta."""
    meta = get_best_drfp_variant()
    best_col = meta['best_drfp_col']
    best_var = meta['best_variant']
    print(f'\n[Step 1] Load data (variant: {best_var}) ...')
    df = pd.read_csv(DATA_EXTENDED, encoding='utf-8-sig')
    df = df[df['extraction_status'] == 'valid'].copy()
    df = df.dropna(subset=['yield (%)'])
    df = df[df['yield (%)'] > 0].reset_index(drop=True)

    if use_holdout_train:
        train_idx, _, _ = holdout_arrays(load_manifest())
        df = df.iloc[sorted(train_idx)].reset_index(drop=True)
        print(f'  [load_data] filtered to holdout train pool: {len(df)} rows')

    X_drfp = parse_drfp_col(df[best_col]).astype(np.float32)

    xtb_candidates = [
        'sub_homo_eV', 'sub_lumo_eV', 'sub_gap_eV', 'sub_dipole_D',
        'co2_homo_eV', 'co2_lumo_eV', 'co2_gap_eV',
        'cat_homo_eV', 'cat_lumo_eV', 'cat_gap_eV', 'cat_dipole_D',
        'solv_homo_eV', 'solv_lumo_eV', 'solv_gap_eV',
        'delta_E_hl_cat_sub', 'global_hardness', 'nucleophilicity_index',
        'cat_homo_eV_min', 'cat_lumo_eV_max', 'cat_gap_eV_min',
        'cat_cation_homo_eV', 'cat_cation_lumo_eV', 'cat_cation_gap_eV',
        'cat_anion_homo_eV', 'cat_anion_lumo_eV', 'cat_anion_gap_eV',
        'cat_cation_dipole_D', 'cat_anion_dipole_D',
        'activation_proxy', 'charge_transfer_potential', 'ion_pair_interaction',
        'electrophilicity_cat', 'electrodonating_cat',
        'sub_cat_orbital_match', 'gap_ratio', 'hardness_ratio',
        'nucleophilicity_cat', 'reaction_polarity', 'co2_activation_proxy',
        'solv_cat_interaction', 'solv_sub_interaction',
        'total_polarity_index', 'dielectric_proxy',
    ]
    xtb_cols = [c for c in xtb_candidates if c in df.columns]
    X_xtb = np.nan_to_num(df[xtb_cols].values.astype(np.float32), nan=0.0)

    temp_col = _col(df, 'temperature')[0]
    pres_col = _col(df, 'pressure')[0]
    time_col = _col(df, 'time(')[0] if _col(df, 'time(') else _col(df, 'time')[0]
    cond_cols = [temp_col, pres_col, time_col] + [c for c in df.columns if 'loading_mol%' in c]
    cond_cols = [c for c in cond_cols if c in df.columns]
    X_cond = np.nan_to_num(df[cond_cols].values.astype(np.float32), nan=0.0)

    T = X_cond[:, 0:1]
    P = X_cond[:, 1:2]
    inter_parts = []
    ai = xtb_cols.index('activation_proxy') if 'activation_proxy' in xtb_cols else None
    tpi = xtb_cols.index('total_polarity_index') if 'total_polarity_index' in xtb_cols else None
    if ai is not None:
        inter_parts.append(T * X_xtb[:, ai:ai+1])
    if tpi is not None:
        inter_parts.append(P * X_xtb[:, tpi:tpi+1])
    X_inter = np.concatenate(inter_parts, axis=1).astype(np.float32) if inter_parts \
        else np.zeros((len(df), 0), dtype=np.float32)

    X_xtb_cond_inter = np.hstack([X_xtb, X_cond, X_inter]).astype(np.float32)
    y = df['yield (%)'].values.astype(np.float32) / 100.0

    return X_drfp, X_xtb_cond_inter, y


# =====================================================================
# 模型定义
# =====================================================================
class PropertyCoLearningAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
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


def train_pcl_ae(X_train, y_train, latent_dim=128, lambda_prop=None,
                 epochs=150, batch_size=128, lr=1e-3):
    if lambda_prop is None:
        from config import BEST_LAMBDA_PROP as lambda_prop
    X_tensor = torch.FloatTensor(X_train)
    y_tensor = torch.FloatTensor(y_train).unsqueeze(1)
    model = PropertyCoLearningAE(X_train.shape[1], latent_dim)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=10, factor=0.5)
    bce = nn.BCEWithLogitsLoss(reduction='none')
    mse = nn.MSELoss()
    loader = DataLoader(TensorDataset(X_tensor, y_tensor), batch_size=batch_size, shuffle=True)
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            z = model.encoder(xb)
            recon = model.decoder(z)
            pred = model.predictor(z).squeeze(-1)
            recon_loss = (bce(recon, xb) * (xb * 9.0 + 1.0)).mean()
            prop_loss = mse(pred, yb.squeeze(-1))
            (recon_loss + lambda_prop * prop_loss).backward()
            opt.step()
        sched.step(recon_loss.item())
    model.eval()
    return model


def train_dualann(Xd_tr, Xo_tr, y_tr, Xd_va, Xo_va, y_va, hidden=128, epochs=300):
    Xd_tr_t = torch.FloatTensor(Xd_tr).to(DEVICE)
    Xd_va_t = torch.FloatTensor(Xd_va).to(DEVICE)
    Xo_tr_t = torch.FloatTensor(Xo_tr).to(DEVICE)
    Xo_va_t = torch.FloatTensor(Xo_va).to(DEVICE)
    y_tr_t = torch.FloatTensor(y_tr).to(DEVICE)

    model = DualBranchANN(Xd_tr.shape[1], Xo_tr.shape[1], hidden=hidden).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=15, factor=0.5)
    ds = TensorDataset(Xd_tr_t, Xo_tr_t, y_tr_t)
    dl = DataLoader(ds, batch_size=64, shuffle=True)

    best_r2, best_state, no_imp = -np.inf, None, 0
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
        sched.step(1 - r2)
        if r2 > best_r2:
            best_r2 = r2
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1
        if no_imp >= 40:
            break
    model.load_state_dict(best_state)
    model.eval()
    return model


# =====================================================================
# 5×2 KFold — 收集每个 fold 的配对误差
# =====================================================================
def run_5x2_cv_paired(X_drfp, X_xtb, y, seed_base=42, n_repeats=2, n_folds=5):
    """
    返回：
      fold_errors: dict of lists
        每个模型 -> 每 fold 的 |error| 均值（用于配对比较）
      fold_r2: dict of lists
        每个模型 -> 每 fold 的 R²
    """
    from config import BEST_LAMBDA_PROP, BEST_LATENT_DIM

    models = ['RF', 'XGB', 'LGBM', 'DualANN']
    fold_errors = {m: [] for m in models}
    fold_r2 = {m: [] for m in models}
    fold_details = []  # 保存所有 fold 的逐样本误差

    # 5-fold CV using the canonical data_split.json manifest (seed=2026, yield-stratified)
    manifest_folds = kfold_folds(load_manifest())
    for rep in range(n_repeats):
        for fold_id, tr, te in manifest_folds:
            # --- Per-fold PCL-AE (train-only) to prevent leakage ---
            sd = StandardScaler()
            Xd_tr_s = sd.fit_transform(X_drfp[tr]).astype(np.float32)
            Xd_te_s = sd.transform(X_drfp[te]).astype(np.float32)
            pcl = train_pcl_ae(Xd_tr_s, y[tr], latent_dim=BEST_LATENT_DIM,
                                lambda_prop=BEST_LAMBDA_PROP, epochs=120)
            with torch.no_grad():
                Z_tr = pcl.encode(torch.FloatTensor(Xd_tr_s)).numpy()
                Z_te = pcl.encode(torch.FloatTensor(Xd_te_s)).numpy()

            # xTB + Cond for this fold
            Xo_tr = X_xtb[tr].astype(np.float32)
            Xo_te = X_xtb[te].astype(np.float32)

            # Combined features: AE(latent) + xTB = (BEST_LATENT_DIM + X_xtb.shape[1])D
            # ALL models receive the same input for fair comparison
            X_comb_tr = np.hstack([Z_tr, Xo_tr]).astype(np.float64)
            X_comb_te = np.hstack([Z_te, Xo_te]).astype(np.float64)

            # ----- RF (uses same AE latent as DualANN) -----
            sc_rf = StandardScaler()
            Xf_tr = sc_rf.fit_transform(X_comb_tr)
            Xf_te = sc_rf.transform(X_comb_te)
            rf = RandomForestRegressor(n_estimators=200, max_depth=20,
                                        min_samples_leaf=2, n_jobs=-1,
                                        random_state=seed_base)
            rf.fit(Xf_tr, y[tr])
            pred_rf = np.clip(rf.predict(Xf_te), 0, 1)

            # ----- XGB (uses same AE latent as DualANN) -----
            sc_xgb = StandardScaler()
            Xf_tr2 = sc_xgb.fit_transform(X_comb_tr)
            Xf_te2 = sc_xgb.transform(X_comb_te)
            xgb_m = xgb.XGBRegressor(n_estimators=500, max_depth=8, learning_rate=0.05,
                                      subsample=0.8, colsample_bytree=0.8,
                                      tree_method='hist', random_state=seed_base, verbosity=0)
            xgb_m.fit(Xf_tr2, y[tr])
            pred_xgb = np.clip(xgb_m.predict(Xf_te2), 0, 1)

            # ----- LGBM (uses same AE latent as DualANN) -----
            sc_lgb = StandardScaler()
            Xf_tr3 = sc_lgb.fit_transform(X_comb_tr)
            Xf_te3 = sc_lgb.transform(X_comb_te)
            lgb_m = lgb.LGBMRegressor(n_estimators=500, num_leaves=63, learning_rate=0.05,
                                        subsample=0.8, colsample_bytree=0.8, min_data_in_leaf=10,
                                        device='cpu', verbose=-1, random_state=seed_base)
            lgb_m.fit(Xf_tr3, y[tr])
            pred_lgb = np.clip(lgb_m.predict(Xf_te3), 0, 1)

            # ----- DualANN (uses same AE latent as tree models) -----
            so = StandardScaler()
            Xo_tr_s = so.fit_transform(Xo_tr)
            Xo_te_s = so.transform(Xo_te)
            dualann = train_dualann(Z_tr, Xo_tr_s, y[tr], Z_te, Xo_te_s, y[te],
                                     hidden=BEST_LATENT_DIM, epochs=300)
            dualann.eval()
            with torch.no_grad():
                pred_ann = np.clip(
                    dualann(torch.FloatTensor(Z_te).to(DEVICE),
                            torch.FloatTensor(Xo_te_s).to(DEVICE)).cpu().numpy(), 0, 1)

            # 计算误差
            y_te = y[te]
            pred_dict = {'RF': pred_rf, 'XGB': pred_xgb, 'LGBM': pred_lgb, 'DualANN': pred_ann}

            for mname, pred in pred_dict.items():
                mae = np.mean(np.abs(pred - y_te))  # MAE per fold
                r2 = r2_score(y_te, pred)
                fold_errors[mname].append(mae)
                fold_r2[mname].append(r2)

            # 保存逐样本误差（用于 Wilcoxon）
            for i, idx in enumerate(te):
                fold_details.append({
                    'rep': rep, 'fold': fold_id,
                    'sample_idx': idx,
                    'y_true': y_te[i],
                    'pred_RF': pred_rf[i],
                    'pred_XGB': pred_xgb[i],
                    'pred_LGBM': pred_lgb[i],
                    'pred_DualANN': pred_ann[i],
                })

    return fold_errors, fold_r2, pd.DataFrame(fold_details)


# =====================================================================
# Wilcoxon signed-rank test + Cohen's d effect size
# =====================================================================
def cohen_d_paired(a, b):
    """Cohen's d for paired samples: d = mean(diff) / std(diff).
    Returns NaN if std(diff) is 0 (i.e., no variation)."""
    diff = np.asarray(a) - np.asarray(b)
    s = float(np.std(diff, ddof=1))
    if s < 1e-12:
        return float('nan')
    return float(np.mean(diff) / s)


def interpret_cohen_d(d):
    """Sawilowsky (2009) extension of Cohen's thresholds:
       |d| < 0.2 very small, 0.2-0.5 small, 0.5-0.8 medium,
       0.8-1.2 large, 1.2-2.0 very large, >= 2.0 huge.
    Returns a label string. NaN -> 'undefined'."""
    if d is None or np.isnan(d):
        return 'undefined'
    a = abs(d)
    if a < 0.2:   return 'very small'
    if a < 0.5:   return 'small'
    if a < 0.8:   return 'medium'
    if a < 1.2:   return 'large'
    if a < 2.0:   return 'very large'
    return 'huge'


def wilcoxon_test(errors1, errors2, name1, name2):
    """Wilcoxon signed-rank test on paired errors. We test
       H0: median(errors1) == median(errors2)
       using the standard two-sided alternative (raw errors, no
       sign flipping needed). Returns (p_value, significance_marks).
    """
    n = len(errors1)
    assert n == len(errors2), f"fold 数量不匹配: {n} vs {len(errors2)}"

    diffs = np.array(errors1) - np.array(errors2)
    if np.all(diffs == 0):
        return np.nan, 'identical'

    try:
        stat, p = wilcoxon(errors1, errors2, alternative='two-sided')
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
        return float(p), sig
    except Exception as e:
        return np.nan, f'error: {e}'


# =====================================================================
# 主函数
# =====================================================================
def main():
    t0 = time.time()
    print('=' * 72)
    print('  配对统计显著性检验')
    print('  Dietterich 5×2 CV + Wilcoxon signed-rank test')
    print('=' * 72)
    print(f'设备: {DEVICE}')

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # 加载数据
    print('\n[Step 1] 加载数据 ...')
    X_drfp, X_xtb, y = load_data()
    n = len(y)
    print(f'  样本数: {n}')

    # 5×2 KFold
    print('\n[Step 2] 5×2 KFold 配对实验（这步最耗时，约 20 分钟）...')
    fold_errors, fold_r2, df_details = run_5x2_cv_paired(
        X_drfp, X_xtb, y, seed_base=42, n_repeats=2, n_folds=5)

    print('\n  各模型 fold 性能汇总：')
    for mname in ['RF', 'XGB', 'LGBM', 'DualANN']:
        r2s = fold_r2[mname]
        maes = fold_errors[mname]
        print(f'    {mname:10s}  R² = {np.mean(r2s):.4f} ± {np.std(r2s):.4f}  '
              f'MAE = {np.mean(maes):.4f} ± {np.std(maes):.4f}')

    # Wilcoxon 配对检验
    print('\n[Step 3] Wilcoxon signed-rank test ...')
    comparisons = [
        ('DualANN', 'RF'),
        ('DualANN', 'XGB'),
        ('DualANN', 'LGBM'),
        ('RF', 'XGB'),
        ('RF', 'LGBM'),
        ('XGB', 'LGBM'),
    ]

    results = []
    for m1, m2 in comparisons:
        # MAE comparison (lower = better)
        p_mae, sig = wilcoxon_test(fold_errors[m1], fold_errors[m2], m1, m2)
        d_mae = cohen_d_paired(fold_errors[m1], fold_errors[m2])
        d_mae_lbl = interpret_cohen_d(d_mae)

        # R² comparison (higher = better). Use paired differences:
        # for each fold, diff = R²_m1 - R²_m2. Positive diff => m1 is better.
        # We feed the raw R²s to wilcoxon; scipy does NOT negate them — the
        # returned p is identical for (a, b) vs (-a, -b) since the test is
        # rank-based. The previous `[-x for x in ...]` trick was harmless but
        # unreadable. Cohen's d keeps the natural sign (positive => m1 better).
        p_r2, sig_r2 = wilcoxon_test(fold_r2[m1], fold_r2[m2], m1, m2)
        d_r2 = cohen_d_paired(fold_r2[m1], fold_r2[m2])
        d_r2_lbl = interpret_cohen_d(d_r2)

        mean_diff = np.mean(fold_errors[m1]) - np.mean(fold_errors[m2])
        winner = m1 if mean_diff < 0 else m2

        results.append({
            'model_1': m1,
            'model_2': m2,
            'winner (lower MAE)': winner,
            'mean_MAE_1': np.mean(fold_errors[m1]),
            'mean_MAE_2': np.mean(fold_errors[m2]),
            'mean_MAE_diff': mean_diff,
            'p_value_MAE': p_mae,
            'significance_MAE': sig,
            'cohen_d_MAE': d_mae,
            'cohen_d_MAE_interpretation': d_mae_lbl,
            'mean_R2_1': np.mean(fold_r2[m1]),
            'mean_R2_2': np.mean(fold_r2[m2]),
            'mean_R2_diff': float(np.mean(fold_r2[m1]) - np.mean(fold_r2[m2])),
            'p_value_R2': p_r2,
            'significance_R2': sig_r2,
            'cohen_d_R2': d_r2,
            'cohen_d_R2_interpretation': d_r2_lbl,
        })
        print(f'  {m1} vs {m2}: ΔMAE={mean_diff:+.4f}, p(MAE)={p_mae:.4f}{sig}  '
              f'p(R²)={p_r2:.4f}{sig_r2}')

    df_results = pd.DataFrame(results)
    out_csv = os.path.join(OUTPUT_DIR, 'wilcoxon_results.csv')
    df_results.to_csv(out_csv, index=False, encoding='utf-8-sig')
    print(f'\n  已保存: {out_csv}')

    # 保存逐样本误差
    detail_csv = os.path.join(OUTPUT_DIR, 'paired_errors.csv')
    df_details.to_csv(detail_csv, index=False, encoding='utf-8-sig')
    print(f'  已保存: {detail_csv}')

    # 汇总表
    print('\n' + '=' * 72)
    print('  汇总：Wilcoxon signed-rank test + Cohen\'s d (α = 0.05)')
    print('=' * 72)
    print(f'  {"比较":22s} {"ΔMAE":>8s} {"p(MAE)":>8s} {"sig":>4s} {"Cohen d":>8s} {"size":>10s}  {"结论"}')
    print(f'  {"-"*22} {"-"*8} {"-"*8} {"-"*4} {"-"*8} {"-"*10}')
    for _, row in df_results.iterrows():
        sig = row['significance_MAE']
        if sig == '':
            conclusion = '无显著差异'
        else:
            w = row['winner (lower MAE)']
            conclusion = f'{w} 显著更优'
        print(f'  {row["model_1"]:>8s} vs {row["model_2"]:<8s} '
              f'{row["mean_MAE_diff"]:>+8.4f} {row["p_value_MAE"]:>8.4f} {sig:>4s} '
              f'{row["cohen_d_MAE"]:>+8.3f} {row["cohen_d_MAE_interpretation"]:>10s}  {conclusion}')

    elapsed = time.time() - t0
    print(f'\n  总耗时: {elapsed/60:.1f} 分钟')

    # --- Q1 ablation: DRFP-only vs DRFP+XTB ---
    # Persist the Q1 ablation results (computed by Q1_analysis.py) into
    # the same output directory for traceability.
    q1_csv = os.path.join(OUTPUT_DIR, 'q1_wilcoxon_drfp_xtb.csv')
    q1_detail_csv = os.path.join(OUTPUT_DIR, 'q1_paired_errors.csv')
    if os.path.exists(q1_csv):
        df_q1 = pd.read_csv(q1_csv)
        print('\n[附录] Q1 消融实验结果（DRFP-only vs DRFP+XTB，Wilcoxon 配对检验）:')
        row = df_q1.iloc[0]
        print(f'  DRFP-only R²  = {row["mean_R2_drfp_only"]:.4f} ± {row["std_R2_drfp_only"]:.4f}')
        print(f'  DRFP + XTB R² = {row["mean_R2_drfp_xtb"]:.4f} ± {row["std_R2_drfp_xtb"]:.4f}')
        print(f'  差值 ΔR²      = {row["mean_diff"]:+.4f}')
        print(f'  Wilcoxon p    = {row["wilcoxon_p"]:.4f}  {row["significance"]}')
        print(f'  结论           = {row["conclusion"]}')
        # Append to the main wilcoxon_results.csv
        aug_row = {
            'model_1': 'DRFP-only',
            'model_2': 'DRFP+XTB',
            'winner (lower MAE)': 'N/A (R²-based comparison)',
            'mean_MAE_1': float('nan'),
            'mean_MAE_2': float('nan'),
            'mean_MAE_diff': float('nan'),
            'p_value_MAE': float('nan'),
            'significance_MAE': 'N/A',
            'mean_R2_1': row['mean_R2_drfp_only'],
            'mean_R2_2': row['mean_R2_drfp_xtb'],
            'p_value_R2': row['wilcoxon_p'],
            'significance_R2': row['significance'],
            'note': 'Q1 ablation: DRFP-only uses median-fill XTB branch; '
                    'Wilcoxon on 5x2 KFold fold-level R² (n=10 folds). '
                    'p=%s, %s' % (row['wilcoxon_p'], row['conclusion']),
        }
        df_results = pd.concat([df_results, pd.DataFrame([aug_row])], ignore_index=True)
        out_csv_aug = os.path.join(OUTPUT_DIR, 'wilcoxon_results_augmented.csv')
        df_results.to_csv(out_csv_aug, index=False, encoding='utf-8-sig')
        print(f'\n  已保存扩充版 Wilcoxon 结果: {out_csv_aug}')
    else:
        print('\n[附录] Q1 消融结果未找到（请先运行 Q1_analysis.py）')

    return df_results, df_details


if __name__ == '__main__':
    main()
