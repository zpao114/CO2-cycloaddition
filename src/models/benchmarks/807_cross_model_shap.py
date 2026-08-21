# -*- coding: utf-8 -*-
"""
807_cross_model_shap.py  (PERFECT v2)
=====================================
#5 — Cross-model consistency check for the CHO sub_homo_eV sign reversal.

Scientific question
-------------------
Is the negative SHAP value for sub_homo_eV in CHO a property of the
PCL-AE / DualBranchANN pipeline, or does it persist when we retrain
Random Forest and XGBoost on the same feature set?

Protocol
--------
1. Load the standard training pool (2116 rows, DRFP-AE latent + XTB/Cond/Inter).
2. For each substrate group G ∈ {CHO, PO, SO, ECH, IGE}:
   - LOCO split: train on rest, evaluate on G.
   - Train RandomForestRegressor and XGBoost on rest-train.
   - Compute SHAP per row on G-test (XGBoost native pred_contribs +
     shap.TreeExplainer for RF).
3. Aggregate per (model, substrate, feature) → mean SHAP.
4. Bar chart: sub_homo_eV mean SHAP across all 5 substrates × 2 models.
5. Permutation test (n_perm configurable, default 500) for CHO sub_homo_eV
   reversal, separately for RF and XGB.  The null is "group labels
   randomly permuted" (natural LOCO null); y is never shuffled because that
   destroys all signal.
6. Bootstrap CI (n_bootstrap configurable) for the CHO sub_homo_eV SHAP mean,
   for both RF and XGB (RF is fast under TreeExplainer).

SHAP backend
------------
* XGBoost : native ``pred_contribs=True`` (exact, fast, no shap-lib dep).
* RF      : ``shap.TreeExplainer`` (exact, polynomial time on trees).
  Note: we deliberately avoid KernelExplainer — it crashes when the
  kmeans background + Lasso surrogate falls into the n_samples < n_features
  regime (180-D features, ~50-row kmeans background).

Outputs
-------
results/results_shap_comprehensive/cross_model/
    cross_model_shap_summary.csv         — shap_mean per (model, substrate, feature)
    cross_model_sub_homo_eV.csv          — pivot: sub_homo_eV shap (substrate × model)
    fig_cross_model_sub_homo.png         — bar chart of sub_homo_eV SHAP
    permutation_results.csv              — p-values for CHO reversal per model
    bootstrap_cho_ci_<model>.csv           — 95% CI for CHO sub_homo_eV per model

Tier placement: tier_si  (depends on tier_main having persisted models)
Typical runtime: ~15–20 min for the default (n_perm=1000, n_bootstrap=1000).
RF permutation uses 100 trees (vs 500 for the canonical fit) to keep the
null distribution under ~10 minutes while preserving stable p-values.

CLI
---
    python 807_cross_model_shap.py --n-perm 500 --n-bootstrap 500
    python 807_cross_model_shap.py --force          # ignore cached outputs
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Path bootstrap ───────────────────────────────────────────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parents[1]
for _p in (_SCRIPT_DIR, _SCRIPT_DIR.parent):  # src/models, src
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent  # repo root
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from paths import RESULTS_DIR                       # noqa: E402
sys.path.insert(0, str(_SCRIPT_DIR / "benchmarks"))  # noqa: E402
from _shap_infra import (                           # noqa: E402
    SUBSTRATE_ORDER, CHO_NAME,
    load_X_y_groups, compute_group_shap,
    permutation_test, bootstrap_shap_ci,
    logger,
)

OUT_DIR = RESULTS_DIR / "results_shap_comprehensive" / "cross_model"
os.makedirs(OUT_DIR, exist_ok=True)


# ============================================================================
# Model factories
# ============================================================================
def train_rf(X_tr, y_tr, X_te, y_te, feat_names, n_estimators=500):
    """Random Forest + TreeExplainer SHAP.

    TreeExplainer is the canonical SHAP algorithm for tree ensembles — it is
    exact (in polynomial time), orders of magnitude faster than KernelExplainer
    on 180-D data, and avoids the "n_samples < n_features" failure mode that
    KernelExplainer exhibits when kmeans background + Lasso regression in the
    surrogate model becomes under-determined (see shap #882).

    Returns
    -------
    shap_vals : (n_test, n_feat) float32
    r2        : float
    """
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import r2_score
    import shap

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=8,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)
    r2 = float(r2_score(y_te, pred))

    explainer = shap.TreeExplainer(model)
    # TreeExplainer for a regressor returns the raw SHAP values (no class axis)
    shap_vals = np.asarray(explainer.shap_values(X_te), dtype=np.float32)
    if shap_vals.ndim == 3:            # very old shap versions return (n,n_feat,1)
        shap_vals = shap_vals[..., 0]
    return shap_vals, r2


def train_xgb(X_tr, y_tr, X_te, y_te, feat_names,
              n_estimators=500, max_depth=5, learning_rate=0.05):
    """XGBoost + native pred_contribs SHAP."""
    import xgboost as xgb
    from sklearn.metrics import r2_score

    dtrain = xgb.DMatrix(X_tr, label=y_tr, feature_names=feat_names)
    dtest = xgb.DMatrix(X_te, label=y_te, feature_names=feat_names)
    params = dict(
        objective="reg:squarederror",
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.5,
        reg_lambda=2.0,
        random_state=42,
        verbosity=0,
        n_jobs=-1,
    )
    model = xgb.train(
        params, dtrain,
        num_boost_round=n_estimators,
        evals=[(dtest, "test")],
        early_stopping_rounds=50,
        verbose_eval=False,
    )
    pred = model.predict(dtest)
    r2 = float(r2_score(y_te, pred))
    contribs = model.predict(dtest, pred_contribs=True, validate_features=False)
    shap_vals = contribs[:, :-1].astype(np.float32)   # drop bias column
    return shap_vals, r2


# ============================================================================
# LOCO driver
# ============================================================================
def loco_one(model_name: str, model_fn, X, y, groups, sub_name, feat_names):
    sub_mask = np.array([g == sub_name for g in groups])
    X_tr, y_tr = X[~sub_mask], y[~sub_mask]
    X_te, y_te = X[sub_mask], y[sub_mask]

    logger.info(
        "[%s] %s — LOCO train=%d, test=%s=%d",
        sub_name, model_name, len(X_tr), sub_name, len(X_te),
    )

    if len(X_te) < 5:
        logger.warning("  Skipping %s (test too small)", sub_name)
        return None
    if len(X_tr) < 20:
        logger.warning("  Skipping %s (train too small)", sub_name)
        return None

    try:
        shap_vals, r2 = model_fn(X_tr, y_tr, X_te, y_te, feat_names)
    except Exception as e:
        logger.error("  Model failed on %s/%s: %s", sub_name, model_name, e)
        return None

    if shap_vals.shape[1] != len(feat_names):
        logger.error(
            "  SHAP column mismatch: got %d expected %d",
            shap_vals.shape[1], len(feat_names),
        )
        return None

    shap_mean = shap_vals.mean(axis=0)
    shap_std = shap_vals.std(axis=0)
    return shap_mean, shap_std, r2


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="807 cross-model SHAP consistency (v2)")
    parser.add_argument("--n-perm", type=int, default=500)
    parser.add_argument("--n-bootstrap", type=int, default=500)
    parser.add_argument("--n-estimators-rf", type=int, default=500)
    parser.add_argument("--n-estimators-xgb", type=int, default=500)
    parser.add_argument("--force", action="store_true",
                        help="Re-run even if outputs exist")
    args = parser.parse_args()

    out_csv = OUT_DIR / "cross_model_shap_summary.csv"
    if out_csv.exists() and not args.force:
        logger.info("Outputs exist; skipping (use --force to re-run)")
        return

    # ── Load full training pool ───────────────────────────────────────────
    X, y, groups, feat_names, df = load_X_y_groups()
    logger.info("Feature dim: %d, train rows: %d", X.shape[1], len(y))

    # ── LOCO × {RF, XGB} ──────────────────────────────────────────────────
    model_fns = {
        "RF": lambda Xtr, ytr, Xte, yte, fn:
            train_rf(Xtr, ytr, Xte, yte, fn,
                     n_estimators=args.n_estimators_rf),
        "XGB": lambda Xtr, ytr, Xte, yte, fn:
            train_xgb(Xtr, ytr, Xte, yte, fn,
                      n_estimators=args.n_estimators_xgb),
    }

    results = []
    perm_results = []

    for sub_name in SUBSTRATE_ORDER:
        sub_mask = np.array([g == sub_name for g in groups])

        for model_name, fn in model_fns.items():
            out = loco_one(model_name, fn, X, y, groups, sub_name, feat_names)
            if out is None:
                continue
            shap_mean, shap_std, r2 = out

            for i, fname in enumerate(feat_names):
                results.append(dict(
                    substrate=sub_name,
                    model=model_name,
                    feature=fname,
                    shap_mean=float(shap_mean[i]),
                    shap_std=float(shap_std[i]),
                    r2=float(r2),
                    n_test=int(sub_mask.sum()),
                ))

            if sub_name == CHO_NAME:
                feat_idx = feat_names.index("sub_homo_eV") if "sub_homo_eV" in feat_names else None
                if feat_idx is not None:
                    logger.info(
                        "  CHO/%s sub_homo_eV mean SHAP=%.5f  R²=%.4f",
                        model_name, float(shap_mean[feat_idx]), r2,
                    )

    # ── Save full summary ────────────────────────────────────────────────
    results_df = pd.DataFrame(results)
    results_df.to_csv(out_csv, index=False)
    logger.info("Saved: %s  (%d rows)", out_csv, len(results_df))

    # ── sub_homo_eV pivot ────────────────────────────────────────────────
    subhomo = results_df[results_df["feature"] == "sub_homo_eV"].copy()
    if subhomo.empty:
        logger.error("No sub_homo_eV rows in summary; cannot build pivot.")
        sys.exit(1)
    subhomo_pivot = subhomo.pivot_table(
        index="substrate", columns="model", values="shap_mean", aggfunc="first",
    ).reindex(SUBSTRATE_ORDER)
    subhomo_pivot.to_csv(OUT_DIR / "cross_model_sub_homo_eV.csv")
    logger.info("\nsub_homo_eV pivot:\n%s", subhomo_pivot.round(5))

    # ── Add chemical interpretation ──────────────────────────────────────
    try:
        from shap_explanation import get_feature_interpretation
        logger.info("\n=== Chemical Interpretation of sub_homo_eV SHAP ===")
        for sub in subhomo_pivot.index:
            for model in subhomo_pivot.columns:
                shap_val = subhomo_pivot.loc[sub, model]
                interp = get_feature_interpretation("sub_homo_eV", shap_val)
                direction = "↑" if shap_val > 0 else "↓"
                logger.info(f"  {sub} ({model}): {shap_val:+.4f} {direction} - {interp.get('main_text', 'N/A')[:60]}")
    except ImportError:
        logger.warning("Could not import shap_explanation for chemical interpretation")

    # ── Permutation test for CHO reversal (separately per model) ─────────
    cho_mask = np.array([g == CHO_NAME for g in groups])
    groups_arr = np.asarray(groups)
    for model_name in model_fns:
        logger.info(
            "[Permutation] CHO sub_homo_eV, model=%s, n_perm=%d",
            model_name, args.n_perm,
        )
        perm = permutation_test(
            X, y, cho_mask, feat_names,
            feature_of_interest="sub_homo_eV",
            n_perm=args.n_perm,
            random_state=42,
            groups=groups_arr,
            model_type="rf" if model_name == "RF" else "xgb",
        )
        perm_results.append(dict(
            model=model_name,
            feature=perm["feature"],
            observed=perm["observed"],
            p_value=perm["p_value"],
            significant=perm["significant"],
            n_perm=perm["n_perm"],
        ))
        logger.info(
            "  observed=%.5f  p=%.4f  significant=%s",
            perm["observed"], perm["p_value"], perm["significant"],
        )

    perm_df = pd.DataFrame(perm_results)
    perm_df.to_csv(OUT_DIR / "permutation_results.csv", index=False)

    # ── Bootstrap CI for CHO — RF and XGB ──────────────────────────────
    for model_name in model_fns:
        logger.info(
            "[Bootstrap] CHO sub_homo_eV CI — %s, n=%d",
            model_name, args.n_bootstrap,
        )
        ci_df = bootstrap_shap_ci(
            X, y, cho_mask, feat_names,
            n_bootstrap=args.n_bootstrap,
            random_state=42,
            model_type="rf" if model_name == "RF" else "xgb",
        )
        out_path = OUT_DIR / f"bootstrap_cho_ci_{model_name.lower()}.csv"
        ci_df.to_csv(out_path, index=False)
        row = ci_df[ci_df.feat_name == "sub_homo_eV"]
        if not row.empty:
            logger.info(
                "  %s sub_homo_eV CI: %.5f [%.5f, %.5f]",
                model_name,
                float(row["shap_mean"].iloc[0]),
                float(row["ci_lo"].iloc[0]),
                float(row["ci_hi"].iloc[0]),
            )

    # ── Plot ────────────────────────────────────────────────────────────
    _plot_cross_model(subhomo_pivot, perm_df, OUT_DIR)
    logger.info("All done. Results in %s", OUT_DIR)


def _short_label(s: str) -> str:
    return s.replace(" oxide", "").replace("Isopropyl glycidyl ether", "IGE")


def _plot_cross_model(pivot: pd.DataFrame, perm_df: pd.DataFrame, out_dir: Path):
    models = list(pivot.columns)
    n_sub = len(pivot)
    fig, axes = plt.subplots(1, len(models), figsize=(7 * len(models), 5), sharey=False)
    if len(models) == 1:
        axes = [axes]
    x = np.arange(n_sub)
    width = 0.55

    for ax, model in zip(axes, models):
        vals = pivot[model].values.astype(float)
        colors = ["tab:red" if v < 0 else "tab:blue" for v in vals]
        ax.bar(x, vals, color=colors, edgecolor="black", alpha=0.85, width=width)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([_short_label(s) for s in pivot.index], rotation=25, ha="right")
        ax.set_ylabel("mean SHAP for sub_homo_eV")
        ax.set_title(f"{model} — sub_homo_eV SHAP by substrate")
        ax.grid(axis="y", alpha=0.3)

        row = perm_df[perm_df["model"] == model]
        if not row.empty:
            p = float(row["p_value"].iloc[0])
            for i, v in enumerate(vals):
                if pivot.index[i] == CHO_NAME and p < 0.05:
                    stars = "***" if p < 0.001 else ("**" if p < 0.01 else "*")
                    ax.annotate(
                        stars, (i, v),
                        xytext=(0, 5 if v >= 0 else -15),
                        textcoords="offset points",
                        ha="center", fontsize=11, color="red", fontweight="bold",
                    )

    blue_patch = mpatches.Patch(color="tab:blue", label="Positive SHAP")
    red_patch = mpatches.Patch(color="tab:red",  label="Negative SHAP (CHO reversal)")
    axes[-1].legend(handles=[blue_patch, red_patch], loc="upper right")

    fig.suptitle(
        "Cross-model consistency: sub_homo_eV SHAP reversal in CHO",
        y=1.02, fontsize=12,
    )
    fig.tight_layout()
    out_path = out_dir / "fig_cross_model_sub_homo.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved figure: %s", out_path)


if __name__ == "__main__":
    sys.exit(main())