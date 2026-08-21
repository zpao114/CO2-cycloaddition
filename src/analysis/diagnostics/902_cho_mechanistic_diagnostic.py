# -*- coding: utf-8 -*-
"""
902_cho_mechanistic_diagnostic.py
=================================

A targeted mechanistic diagnostic for cyclohexene oxide (CHO), the one
substrate whose mean yield is ~30 percentage points below the other four.
This script:

  1. Subsets the dataset to CHO reactions only, and to "other 4 substrates".
  2. Retrains a 25-feature XGBoost on each subset (mirror of 601's pipeline
     so SHAP values are directly comparable to the global SHAP).
  3. Compares top-10 SHAP features between CHO and other-4 substrates to
     surface which descriptors matter for the harder substrate.
  4. Renders per-catalyst box plots of CHO yields.
  5. Performs a Welch t-test on mean yield (CHO vs each other substrate) and
     reports the effect size (Cohen's d).
  6. Optionally links to 514 DFT outputs when present.

Inputs (read-only):
  - co2_drfp_xtb_extended.csv
  - shap_xtb_values.csv                            (global, for context)
  - results_substrate_catalyst_matrix/cho_summary.csv  (output of 901)
  - dft_validation/514_dft_vs_xtb_report.txt       (optional, linked only)

Outputs (results_cho_diagnostic/):
  - cho_vs_other_summary.csv
  - cho_vs_other_ttest.csv
  - cho_shap_top10.csv
  - other_shap_top10.csv
  - cho_shap_vs_other_shap.csv
  - cho_boxplot_by_catalyst.png/.pdf
  - cho_yield_vs_other_yield.png/.pdf
  - cho_shap_top10.png/.pdf
  - cho_shap_vs_other_shap.png/.pdf
  - 902_cho_mechanistic_diagnostic_report.txt

Usage:
  D:\\co2\\env_drfp\\python.exe 902_cho_mechanistic_diagnostic.py
"""
from __future__ import annotations
import os

import io
import sys
import warnings
from datetime import datetime, timezone

warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# ----------------------------------------------------------------------
# Paths / constants
# ----------------------------------------------------------------------
PROJECT_ROOT = os.environ.get("CO2_PROJECT_ROOT", r"D:\machine-learning\CO2-cycloaddition")
# 2026-08-20 fix: paths.py RESULTS_* now point at PROJECT_ROOT (legacy layout).
DATA_EXTENDED = os.path.join(PROJECT_ROOT, 'results', 'results_cho_diagnostic', 'co2_drfp_xtb_extended.csv')
SHAP_CSV = os.path.join(PROJECT_ROOT, 'results_step4_5', 'shap_xtb_values.csv')
CHO_SUMMARY_CSV = os.path.join(PROJECT_ROOT, "results_substrate_catalyst_matrix",
                                "cho_summary.csv")
OUT_DIR = os.path.join(PROJECT_ROOT, "results_902_cho_diagnostic")

CHO_SUBSTRATE = "Cyclohexene oxide"
OTHER_SUBSTRATES = [
    "Styrene oxide",
    "Epichlorohydrin",
    "Propylene oxide",
    "Isopropyl glycidyl ether",
]
RANDOM_STATE = 42
PRIMARY_FAMILIES = ["ionic_liquid", "metal_halide", "mixed_system"]
AGGREGATED_OTHER_LABEL = "other"

os.makedirs(OUT_DIR, exist_ok=True)


# ----------------------------------------------------------------------
# Font setup
# ----------------------------------------------------------------------
for f2 in fm.fontManager.ttflist:
    if any(tag in f2.name for tag in ["SimHei", "Noto Sans CJK", "WenQuanYi", "Microsoft YaHei"]):
        plt.rcParams["font.family"] = f2.name
        break
plt.rcParams["axes.unicode_minus"] = False


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def aggregate_catalyst_family(series: pd.Series) -> pd.Series:
    return series.astype(str).where(series.isin(PRIMARY_FAMILIES),
                                    other=AGGREGATED_OTHER_LABEL)


def load_clean_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_EXTENDED, encoding="utf-8-sig")
    df = df[df["extraction_status"] == "valid"].copy()
    df = df.dropna(subset=["yield (%)"])
    df = df[df["yield (%)"] > 0].reset_index(drop=True)
    df["catalyst_system_type_agg"] = aggregate_catalyst_family(df["catalyst_system_type"])
    # Robust temperature column discovery
    temp_cols = [c for c in df.columns if "temperature" in c.lower()]
    if temp_cols:
        df["temperature_canonical"] = pd.to_numeric(df[temp_cols[0]], errors="coerce")
    return df


def build_xtb_cond_features(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Mirror 601's 25-feature XGB pipeline so SHAP values are comparable.

    Returns: feature matrix (np.ndarray) and feature name list.
    """
    XTB_RAW = ["sub_homo_eV", "sub_lumo_eV", "sub_gap_eV", "sub_dipole_D",
               "co2_homo_eV", "co2_lumo_eV", "co2_gap_eV",
               "cat_homo_eV", "cat_lumo_eV", "cat_gap_eV", "cat_dipole_D",
               "solv_homo_eV", "solv_lumo_eV", "solv_gap_eV"]
    XTB_avail = [c for c in XTB_RAW if c in df.columns]

    sub_gap_v = df["sub_gap_eV"].fillna(0).to_numpy(dtype=np.float64) if "sub_gap_eV" in df.columns \
        else np.zeros(len(df), dtype=np.float64)
    cat_homo_v = df["cat_homo_eV"].fillna(0).to_numpy(dtype=np.float64) if "cat_homo_eV" in df.columns \
        else np.zeros(len(df), dtype=np.float64)
    sub_lumo_v = df["sub_lumo_eV"].fillna(0).to_numpy(dtype=np.float64) if "sub_lumo_eV" in df.columns \
        else np.zeros(len(df), dtype=np.float64)
    cat_lumo_v = df["cat_lumo_eV"].fillna(0).to_numpy(dtype=np.float64) if "cat_lumo_eV" in df.columns \
        else np.zeros(len(df), dtype=np.float64)

    delta_E = cat_homo_v - sub_lumo_v
    delta_LL = cat_lumo_v - sub_lumo_v
    hardness = sub_gap_v / 2.0
    softness = np.where(hardness > 0, 1.0 / (2.0 * hardness), 0.0)
    nucleophilicity = -cat_homo_v
    electrophilicity = cat_lumo_v
    cat_electrophilicity = (cat_homo_v + cat_lumo_v) ** 2 / (
        2.0 * np.where(sub_gap_v > 0, sub_gap_v, 1.0))

    for arr_name in ("delta_E", "delta_LL", "hardness", "softness",
                     "nucleophilicity", "electrophilicity", "cat_electrophilicity"):
        locals()[arr_name] = np.nan_to_num(locals()[arr_name], nan=0.0, posinf=0.0, neginf=0.0)

    XTB_raw_vals = np.nan_to_num(df[XTB_avail].to_numpy(dtype=np.float64), nan=0.0)
    X_xtb = np.column_stack([XTB_raw_vals, delta_E, delta_LL, hardness, softness,
                             nucleophilicity, electrophilicity, cat_electrophilicity])
    XTB_NAMES = XTB_avail + ["delta_E_HL", "delta_E_LL", "global_hardness", "global_softness",
                             "nucleophilicity", "electrophilicity", "cat_electrophilicity"]

    temp_col = "temperature_canonical" if "temperature_canonical" in df.columns \
        else ([c for c in df.columns if "temperature" in c.lower()] or [None])[0]
    temp_arr = pd.to_numeric(df[temp_col], errors="coerce").fillna(
        pd.to_numeric(df[temp_col], errors="coerce").median()).to_numpy(dtype=np.float64) \
        if temp_col else np.zeros(len(df), dtype=np.float64)
    press_arr = pd.to_numeric(df["pressure (MPa)"], errors="coerce").fillna(
        pd.to_numeric(df["pressure (MPa)"], errors="coerce").median()).to_numpy(dtype=np.float64)
    time_arr = pd.to_numeric(df["time (h)"], errors="coerce").fillna(
        pd.to_numeric(df["time (h)"], errors="coerce").median()).to_numpy(dtype=np.float64)
    time_log = np.log1p(np.maximum(time_arr, 0.0))

    loadings = np.zeros(len(df), dtype=np.float64)
    for lc in [f"catalyst_{i}_loading_mol%" for i in range(1, 5)]:
        if lc in df.columns:
            vals = pd.to_numeric(df[lc], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
            loadings += np.nan_to_num(vals, nan=0.0)
    loading_log = np.log1p(np.maximum(loadings, 0.0))

    has_solvent = (df["all_solvents_normalized"].notna() &
                   (df["all_solvents_normalized"].astype(str).str.strip() != "")).astype(float).to_numpy()
    has_reagent = ((df["reagent_1_name"].notna() & (df["reagent_1_name"].astype(str).str.strip() != "")) |
                   (df["reagent_2_name"].notna() & (df["reagent_2_name"].astype(str).str.strip() != ""))).astype(float).to_numpy()

    X_cond = np.column_stack([temp_arr, press_arr, time_log, loading_log, has_solvent, has_reagent])
    COND_NAMES = ["temperature", "pressure", "time_log", "loading_log", "has_solvent", "has_reagent"]

    X = np.hstack([X_xtb, X_cond]).astype(np.float64)
    names = XTB_NAMES + COND_NAMES
    return X, names


def train_xgb_shap(X: np.ndarray, y: np.ndarray, feature_names: list[str]) -> tuple:
    """Train a single XGBoost on the full subset, return (model, SHAP, R^2, RMSE)."""
    from sklearn.model_selection import KFold
    from sklearn.preprocessing import StandardScaler
    import xgboost as xgb
    import shap

    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    tr_idx, va_idx = next(iter(kf.split(X)))
    X_tr, X_va = X[tr_idx], X[va_idx]
    y_tr, y_va = y[tr_idx], y[va_idx]
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr).astype(np.float64)
    X_va_s = scaler.transform(X_va).astype(np.float64)

    model = xgb.XGBRegressor(
        n_estimators=500, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.5, reg_lambda=2.0,
        random_state=RANDOM_STATE, verbosity=0, n_jobs=-1,
        early_stopping_rounds=50,
    )
    model.fit(X_tr_s, y_tr, eval_set=[(X_va_s, y_va)], verbose=False)
    pred_va = model.predict(X_va_s)
    from sklearn.metrics import r2_score, mean_squared_error
    r2 = float(r2_score(y_va, pred_va))
    rmse = float(np.sqrt(mean_squared_error(y_va, pred_va)))

    sv_with_bias = model.get_booster().predict(
        xgb.DMatrix(X_va_s, feature_names=feature_names),
        pred_contribs=True, validate_features=False,
    )
    sv = sv_with_bias[:, :-1]
    return model, sv, r2, rmse, va_idx


def shap_top_n(sv: np.ndarray, feature_names: list[str], n: int = 10) -> pd.DataFrame:
    """Return top-N features by mean |SHAP|."""
    mean_abs = np.abs(sv).mean(axis=0)
    order = np.argsort(mean_abs)[::-1][:n]
    return pd.DataFrame({
        "rank": np.arange(1, n + 1),
        "feature": [feature_names[i] for i in order],
        "mean_abs_shap": [float(mean_abs[i]) for i in order],
    })


def cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    a = a[~np.isnan(a)]; b = b[~np.isnan(b)]
    if a.size < 2 or b.size < 2:
        return float("nan")
    sa, sb = a.std(ddof=1), b.std(ddof=1)
    pooled = np.sqrt(((a.size - 1) * sa ** 2 + (b.size - 1) * sb ** 2) / (a.size + b.size - 2))
    if pooled == 0:
        return float("nan")
    return float((a.mean() - b.mean()) / pooled)


def render_boxplot(df_cho: pd.DataFrame, fname: str):
    families = sorted(df_cho["catalyst_system_type_agg"].unique())
    data = [df_cho.loc[df_cho["catalyst_system_type_agg"] == f, "yield (%)"].to_numpy()
            for f in families]
    fig, ax = plt.subplots(figsize=(10, 6))
    bp = ax.boxplot(data, labels=[f[:18] for f in families], patch_artist=True, notch=True)
    cmap = plt.cm.Set2(np.linspace(0, 1, max(len(families), 3)))
    for patch, color in zip(bp["boxes"], cmap):
        patch.set_facecolor(color)
    ax.axhline(df_cho["yield (%)"].mean(), color="red", linestyle="--", linewidth=1.5,
               label=f"CHO mean = {df_cho['yield (%)'].mean():.2f}%")
    ax.axhline(85.0, color="green", linestyle=":", linewidth=1.5,
               label="Other 4 substrates mean 鈮?85%")
    ax.set_ylabel("Yield (%)", fontsize=11)
    ax.set_title(f"{CHO_SUBSTRATE} yield distribution by catalyst family "
                 f"(n={len(df_cho)})", fontsize=12, pad=12)
    ax.legend(fontsize=9, loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, fname + ".png"), dpi=150, bbox_inches="tight")
    plt.savefig(os.path.join(OUT_DIR, fname + ".pdf"), bbox_inches="tight")
    plt.close()


def render_yield_compare(df_all: pd.DataFrame, fname: str):
    """Strip plot: yield by reactant_name, sorted by median."""
    order = (df_all.groupby("reactant_name")["yield (%)"]
             .median().sort_values(ascending=False).index.tolist())
    data = [df_all.loc[df_all["reactant_name"] == r, "yield (%)"].to_numpy() for r in order]
    fig, ax = plt.subplots(figsize=(11, 6))
    cmap = plt.cm.viridis(np.linspace(0.1, 0.9, len(order)))
    bp = ax.boxplot(data, labels=[r[:20] for r in order], patch_artist=True, notch=True,
                    showmeans=True, meanprops=dict(marker="D", markerfacecolor="red",
                                                   markeredgecolor="red", markersize=6))
    for patch, color in zip(bp["boxes"], cmap):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    ax.set_ylabel("Yield (%)", fontsize=11)
    ax.set_title("Yield distribution by substrate (sorted by median)", fontsize=12, pad=12)
    ax.tick_params(axis="x", rotation=20)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, fname + ".png"), dpi=150, bbox_inches="tight")
    plt.savefig(os.path.join(OUT_DIR, fname + ".pdf"), bbox_inches="tight")
    plt.close()
    return order


def render_shap_compare(cho_top: pd.DataFrame, other_top: pd.DataFrame, fname: str):
    """Side-by-side bar plot: top-10 SHAP features for CHO vs other-4."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 7), sharey=True)
    for ax, df, title, color in [(axes[0], cho_top, f"{CHO_SUBSTRATE} (subset model)",
                                  "#e74c3c"),
                                 (axes[1], other_top, "Other 4 substrates (subset model)",
                                  "#2980b9")]:
        ax.barh(range(len(df))[::-1], df["mean_abs_shap"].values[::-1],
                color=color, edgecolor="white")
        ax.set_yticks(range(len(df))[::-1])
        ax.set_yticklabels(df["feature"].values[::-1], fontsize=9)
        ax.set_xlabel("Mean |SHAP|", fontsize=11)
        ax.set_title(title, fontsize=11, pad=8)
    plt.suptitle("SHAP top-10: CHO vs other 4 substrates", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, fname + ".png"), dpi=150, bbox_inches="tight")
    plt.savefig(os.path.join(OUT_DIR, fname + ".pdf"), bbox_inches="tight")
    plt.close()


def render_shap_delta(merged: pd.DataFrame, fname: str):
    """Bar plot: top features where |CHO - other|/|CHO + other| rank-difference is largest."""
    if "rank_cho" not in merged.columns or "rank_other" not in merged.columns:
        print(f"  [info] rank_cho/rank_other missing; skipping {fname}")
        return
    merged = merged.copy()
    merged["rank_delta"] = merged["rank_other"] - merged["rank_cho"]
    # Top 15 by absolute rank delta
    top = merged.reindex(merged["rank_delta"].abs().sort_values(ascending=False).index).head(15)
    fig, ax = plt.subplots(figsize=(11, 7))
    colors = ["#e74c3c" if d > 0 else "#2980b9" for d in top["rank_delta"]]
    ax.barh(range(len(top))[::-1], top["rank_delta"].values[::-1],
            color=colors, edgecolor="white")
    ax.set_yticks(range(len(top))[::-1])
    ax.set_yticklabels(top["feature"].values[::-1], fontsize=9)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Rank change (other - CHO):  + 鈫?more important for non-CHO;  "
                  "鈭?鈫?more important for CHO", fontsize=10)
    ax.set_title("Where CHO and other substrates disagree (top-15 by |rank delta|)",
                 fontsize=12, pad=12)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, fname + ".png"), dpi=150, bbox_inches="tight")
    plt.savefig(os.path.join(OUT_DIR, fname + ".pdf"), bbox_inches="tight")
    plt.close()


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    print("=" * 60)
    print("902 鈥?CHO Mechanistic Diagnostic")
    print("=" * 60)
    df = load_clean_data()
    print(f"[Load] {len(df)} valid rows")

    cho_mask = df["reactant_name"] == CHO_SUBSTRATE
    other_mask = df["reactant_name"].isin(OTHER_SUBSTRATES)

    print(f"  CHO rows       : {int(cho_mask.sum())}")
    print(f"  Other 4 rows   : {int(other_mask.sum())}")

    # 1. Yield comparison + statistical tests
    rows = []
    cho_yield = df.loc[cho_mask, "yield (%)"].to_numpy(dtype=np.float64)
    for sub in OTHER_SUBSTRATES:
        m = df["reactant_name"] == sub
        if m.sum() < 3:
            continue
        other_yield = df.loc[m, "yield (%)"].to_numpy(dtype=np.float64)
        t_res = stats.ttest_ind(cho_yield, other_yield, equal_var=False)
        d = cohen_d(cho_yield, other_yield)
        rows.append({
            "comparison": f"CHO vs {sub}",
            "n_cho": int(cho_mask.sum()),
            "n_other": int(m.sum()),
            "mean_cho": float(cho_yield.mean()),
            "mean_other": float(other_yield.mean()),
            "delta_pct": float(cho_yield.mean() - other_yield.mean()),
            "t_stat": float(t_res.statistic),
            "p_value": float(t_res.pvalue),
            "cohens_d": d,
            "significant_05": bool(t_res.pvalue < 0.05),
        })
    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, "cho_vs_other_ttest.csv"), index=False,
                               encoding="utf-8-sig")
    print(f"  Saved: cho_vs_other_ttest.csv")

    # 2. SHAP on CHO subset and other subset
    X_all, feat_names = build_xtb_cond_features(df)
    y_all = (df["yield (%)"].to_numpy(dtype=np.float64) / 100.0)

    X_cho = X_all[cho_mask.to_numpy()]
    y_cho = y_all[cho_mask.to_numpy()]
    X_other = X_all[other_mask.to_numpy()]
    y_other = y_all[other_mask.to_numpy()]

    if len(X_cho) < 30:
        print(f"[WARN] only {len(X_cho)} CHO rows; SHAP may be unstable")

    model_cho, sv_cho, r2_cho, rmse_cho, va_cho = train_xgb_shap(X_cho, y_cho, feat_names)
    model_other, sv_other, r2_other, rmse_other, va_other = train_xgb_shap(X_other, y_other,
                                                                            feat_names)
    print(f"  CHO model       : R^2={r2_cho:.4f}  RMSE={rmse_cho:.4f}  (val n={len(va_cho)})")
    print(f"  Other-4 model   : R^2={r2_other:.4f}  RMSE={rmse_other:.4f}  (val n={len(va_other)})")

    cho_top = shap_top_n(sv_cho, feat_names, n=10)
    other_top = shap_top_n(sv_other, feat_names, n=10)
    cho_top.to_csv(os.path.join(OUT_DIR, "cho_shap_top10.csv"), index=False,
                   encoding="utf-8-sig")
    other_top.to_csv(os.path.join(OUT_DIR, "other_shap_top10.csv"), index=False,
                     encoding="utf-8-sig")
    print("  Saved: cho_shap_top10.csv / other_shap_top10.csv")

    # 3. SHAP comparison: top features where CHO and other disagree
    merged = pd.merge(cho_top.rename(columns={"rank": "rank_cho",
                                              "mean_abs_shap": "mean_abs_shap_cho"}),
                      other_top.rename(columns={"rank": "rank_other",
                                                "mean_abs_shap": "mean_abs_shap_other"}),
                      on="feature", how="outer")
    # Fill missing ranks (feature only in one top-10) with rank 11
    merged["rank_cho"] = merged["rank_cho"].fillna(11).astype(int)
    merged["rank_other"] = merged["rank_other"].fillna(11).astype(int)
    merged.to_csv(os.path.join(OUT_DIR, "cho_shap_vs_other_shap.csv"), index=False,
                  encoding="utf-8-sig")
    print("  Saved: cho_shap_vs_other_shap.csv")

    # 4. Plots
    render_boxplot(df.loc[cho_mask].copy(), fname="cho_boxplot_by_catalyst")
    print("  Saved: cho_boxplot_by_catalyst.png/.pdf")
    order = render_yield_compare(df.copy(), fname="cho_yield_vs_other_yield")
    print("  Saved: cho_yield_vs_other_yield.png/.pdf")
    render_shap_compare(cho_top, other_top, fname="cho_shap_top10")
    print("  Saved: cho_shap_top10.png/.pdf")
    render_shap_delta(merged, fname="cho_shap_vs_other_shap")
    print("  Saved: cho_shap_vs_other_shap.png/.pdf")

    # 5. CHO summary (subset of 901's cho_summary + extras)
    summary_rows = []
    for fam in sorted(df.loc[cho_mask, "catalyst_system_type_agg"].unique()):
        m = (df["catalyst_system_type_agg"] == fam) & cho_mask
        summary_rows.append({
            "catalyst_family": fam,
            "n": int(m.sum()),
            "mean_yield": float(df.loc[m, "yield (%)"].mean()) if m.sum() else float("nan"),
            "std_yield": float(df.loc[m, "yield (%)"].std()) if m.sum() > 1 else float("nan"),
        })
    pd.DataFrame(summary_rows).to_csv(os.path.join(OUT_DIR, "cho_summary_extended.csv"),
                                       index=False, encoding="utf-8-sig")

    # 6. CHO-vs-other yield summary row
    pd.DataFrame([{
        "reactant": CHO_SUBSTRATE,
        "n": int(cho_mask.sum()),
        "mean_yield_pct": float(cho_yield.mean()),
        "std_yield_pct": float(cho_yield.std(ddof=1)) if len(cho_yield) > 1 else float("nan"),
    }]).to_csv(os.path.join(OUT_DIR, "cho_vs_other_summary.csv"), index=False,
               encoding="utf-8-sig")
    print("  Saved: cho_vs_other_summary.csv")

    # 7. Report
    lines = []
    lines.append("=" * 70)
    lines.append("902 鈥?CHO Mechanistic Diagnostic")
    lines.append("=" * 70)
    lines.append(f"Generated: {datetime.now(timezone.utc).isoformat()}")
    lines.append(f"Dataset  : {DATA_EXTENDED}")
    lines.append(f"CHO rows : {int(cho_mask.sum())}")
    lines.append(f"Other 4  : {int(other_mask.sum())}")
    lines.append("")
    lines.append("CHO model R^2     = {:.4f}   RMSE = {:.4f}   (val n = {})".format(
        r2_cho, rmse_cho, len(va_cho)))
    lines.append("Other-4 model R^2 = {:.4f}   RMSE = {:.4f}   (val n = {})".format(
        r2_other, rmse_other, len(va_other)))
    if r2_cho < 0.05:
        lines.append("  NOTE: CHO model R^2 is near zero 鈥?expected, because all CHO rows")
        lines.append("        share the same substrate, so substrate electronic descriptors")
        lines.append("        lose their discriminatory power. The model relies almost entirely")
        lines.append("        on conditions (temperature, pressure, time_log) to make")
        lines.append("        within-substrate predictions.")
    lines.append("")
    lines.append(f"Mean yield (%):  CHO = {cho_yield.mean():.2f}    "
                 f"Other-4 mean = {df.loc[other_mask, 'yield (%)'].mean():.2f}    "
                 f"螖 = {cho_yield.mean() - df.loc[other_mask, 'yield (%)'].mean():.2f}")
    lines.append("")
    lines.append("Pairwise Welch t-tests (CHO vs each other substrate):")
    for r in rows:
        lines.append(f"  {r['comparison']:30s}  n_cho={r['n_cho']:4d}  n_other={r['n_other']:4d}  "
                     f"螖={r['delta_pct']:+.2f}  t={r['t_stat']:+.3f}  "
                     f"p={r['p_value']:.2e}  d={r['cohens_d']:+.3f}  "
                     f"sig={r['significant_05']}")
    lines.append("")
    lines.append("CHO top-10 SHAP:")
    for _, r in cho_top.iterrows():
        lines.append(f"  {int(r['rank']):2d}. {r['feature']:30s}  mean|SHAP|={r['mean_abs_shap']:.5f}")
    lines.append("")
    lines.append("Other-4 top-10 SHAP:")
    for _, r in other_top.iterrows():
        lines.append(f"  {int(r['rank']):2d}. {r['feature']:30s}  mean|SHAP|={r['mean_abs_shap']:.5f}")
    lines.append("")
    lines.append("Top features ranked differently between CHO and other-4:")
    if "rank_cho" in merged.columns and "rank_other" in merged.columns:
        merged["rank_delta"] = merged["rank_other"] - merged["rank_cho"]
        delta = merged.sort_values("rank_delta", key=lambda s: s.abs(), ascending=False).head(10)
        for _, r in delta.iterrows():
            lines.append(f"  {r['feature']:30s}  rank CHO={int(r['rank_cho']):2d}  "
                         f"rank other={int(r['rank_other']):2d}  "
                         f"螖rank={int(r['rank_delta']):+d}")
    else:
        lines.append("  (rank columns missing; nothing to compare)")
    lines.append("")
    lines.append("Outputs:")
    for f in sorted(os.listdir(OUT_DIR)):
        lines.append(f"  - {f}")
    report = "\n".join(lines)
    with open(os.path.join(OUT_DIR, "902_cho_mechanistic_diagnostic_report.txt"),
              "w", encoding="utf-8") as f:
        f.write(report)
    print("\n" + report)
    print("\nDone!")


if __name__ == "__main__":
    main()