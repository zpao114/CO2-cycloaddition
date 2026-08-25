# -*- coding: utf-8 -*-
"""Step 4.5: Per-substrate and per-(substrate, mechanism) SHAP analysis.

数据路径（已验证存在于 D:\machine-learning\CO2-cycloaddition\）：
    - results/results_cho_diagnostic/co2_drfp_xtb_extended.csv
    - data/processed/catalyst_mechanism.csv

CAVEAT (audit 2026-08-22): y is in **percent** (0-100). SHAP values are
also in **percent** units (consistent with the yield target). This is
**different** from `902_cho_mechanistic_diagnostic.py` and
`generate_shap_for_901.py`, both of which divide y by 100 on read and
work in fraction units. Do not mix the SHAP CSVs across these three
scripts without re-converting the units. See
[`docs/CODE_AUDIT.md`](../../docs/CODE_AUDIT.md) §3.

Outputs
-------
results_step4_5/per_substrate_shap.csv
results_step4_5/per_substrate_shap_direction.csv
results_step4_5/per_pair_shap.csv
results_step4_5/per_substrate_top_features.csv
results_step4_5/per_substrate_top_features_summary.json
results_step4_5/figs/per_substrate_shap_heatmap.png
results_step4_5/figs/shap_direction_flip.png
results_step4_5/figs/per_pair_shap_heatmap.png
results_step4_5/report.txt
"""
from __future__ import annotations

import os
import sys
import io
import warnings
import json
from datetime import datetime

warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

# ── Project root ────────────────────────────────────────────────────────────
PROJECT_ROOT = os.environ.get("CO2_PROJECT_ROOT", r"D:\machine-learning\CO2-cycloaddition")
DATA_CSV     = os.path.join(PROJECT_ROOT, "results", "results_cho_diagnostic",
                             "co2_drfp_xtb_extended.csv")
MECH_CSV     = os.path.join(PROJECT_ROOT, "data", "processed",
                             "catalyst_mechanism.csv")
OUT_DIR      = os.path.join(PROJECT_ROOT, "results_step4_5")
FIG_DIR      = os.path.join(OUT_DIR,  "figs")

RANDOM_STATE = 42

os.makedirs(OUT_DIR,  exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# Feature builder — copied verbatim from 700_loso_lomo_cv.py
# (avoids import dependency so this script is fully self-contained)
# ══════════════════════════════════════════════════════════════════════════════
def build_xtb_cond_features(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Build xTB electronic + reaction-condition feature matrix."""
    XTB_RAW = [
        "sub_homo_eV", "sub_lumo_eV", "sub_gap_eV", "sub_dipole_D",
        "co2_homo_eV", "co2_lumo_eV", "co2_gap_eV",
        "cat_homo_eV", "cat_lumo_eV", "cat_gap_eV", "cat_dipole_D",
        "solv_homo_eV", "solv_lumo_eV", "solv_gap_eV",
    ]
    XTB_avail = [c for c in XTB_RAW if c in df.columns]

    def _safe_col(col: str, default=0.0) -> np.ndarray:
        if col in df.columns:
            return pd.to_numeric(df[col], errors="coerce").fillna(default).to_numpy(dtype=np.float64)
        return np.full(len(df), default, dtype=np.float64)

    sub_gap_v = _safe_col("sub_gap_eV")
    cat_homo_v = _safe_col("cat_homo_eV")
    sub_lumo_v = _safe_col("sub_lumo_eV")
    cat_lumo_v = _safe_col("cat_lumo_eV")

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
    X_xtb = np.column_stack([
        XTB_raw_vals, delta_E, delta_LL, hardness, softness,
        nucleophilicity, electrophilicity, cat_electrophilicity,
    ])
    XTB_NAMES = XTB_avail + [
        "delta_E_HL", "delta_E_LL", "global_hardness", "global_softness",
        "nucleophilicity", "electrophilicity", "cat_electrophilicity",
    ]

    temp_col = ([c for c in df.columns if "temperature" in c.lower()] or [None])[0]
    temp_arr = pd.to_numeric(df[temp_col], errors="coerce").fillna(
        pd.to_numeric(df[temp_col], errors="coerce").median()
    ).to_numpy(dtype=np.float64) if temp_col else np.zeros(len(df), dtype=np.float64)

    press_arr = pd.to_numeric(df["pressure (MPa)"], errors="coerce").fillna(
        pd.to_numeric(df["pressure (MPa)"], errors="coerce").median()
    ).to_numpy(dtype=np.float64)

    time_arr = pd.to_numeric(df["time (h)"], errors="coerce").fillna(
        pd.to_numeric(df["time (h)"], errors="coerce").median()
    ).to_numpy(dtype=np.float64)
    time_log = np.log1p(np.maximum(time_arr, 0.0))

    loadings = np.zeros(len(df), dtype=np.float64)
    for lc in [f"catalyst_{i}_loading_mol%" for i in range(1, 5)]:
        if lc in df.columns:
            vals = pd.to_numeric(df[lc], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
            loadings += np.nan_to_num(vals, nan=0.0)
    loading_log = np.log1p(np.maximum(loadings, 0.0))

    has_solvent = (
        df["all_solvents_normalized"].notna() &
        (df["all_solvents_normalized"].astype(str).str.strip() != "")
    ).astype(float).to_numpy()

    has_reagent = (
        (df["reagent_1_name"].notna() & (df["reagent_1_name"].astype(str).str.strip() != "")) |
        (df["reagent_2_name"].notna() & (df["reagent_2_name"].astype(str).str.strip() != ""))
    ).astype(float).to_numpy()

    X_cond = np.column_stack([temp_arr, press_arr, time_log, loading_log, has_solvent, has_reagent])
    COND_NAMES = ["temperature", "pressure", "time_log", "loading_log", "has_solvent", "has_reagent"]

    X = np.hstack([X_xtb, X_cond]).astype(np.float64)
    names = XTB_NAMES + COND_NAMES
    return X, names


def add_mech_one_hot(X: np.ndarray, names: list[str],
                     df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Append one-hot columns for 5 mechanism classes."""
    mech_classes = ["NUC", "LAC", "BAS", "BIF", "OTH"]
    one_hot = np.zeros((len(df), len(mech_classes)), dtype=np.float64)
    for i, mc in enumerate(mech_classes):
        one_hot[:, i] = (df["mech_label"].values == mc).astype(np.float64)
    new_names = names + [f"mech_{m}" for m in mech_classes]
    return np.hstack([X, one_hot]), new_names


# ══════════════════════════════════════════════════════════════════════════════
# Data loading
# ══════════════════════════════════════════════════════════════════════════════
def load_data() -> pd.DataFrame:
    if not os.path.exists(DATA_CSV):
        raise FileNotFoundError(
            f"Data CSV not found: {DATA_CSV}\n"
            "Please run the full pipeline (Tier 1) first to generate this file."
        )
    df = pd.read_csv(DATA_CSV, encoding="utf-8-sig")
    df = df[df["extraction_status"] == "valid"].copy()
    df = df.dropna(subset=["yield (%)"])
    df = df[df["yield (%)"] > 0].reset_index(drop=True)

    df["catalyst_system_type_agg"] = np.where(
        df["catalyst_system_type"].isin(["ionic_liquid", "metal_halide", "mixed_system"]),
        df["catalyst_system_type"],
        "other",
    )

    mech = pd.read_csv(MECH_CSV)
    mech = mech[["name", "mechanism"]].rename(
        columns={"name": "catalyst_1_name", "mechanism": "mech_label"})
    df = df.merge(mech, on="catalyst_1_name", how="left")
    df["mech_label"] = df["mech_label"].fillna("UNK")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# XGBoost + SHAP (pred_contribs to bypass shap>=0.44 UTF-8 bug)
# ══════════════════════════════════════════════════════════════════════════════
def _sanitize(s: str) -> str:
    return (s.replace("\u00b0", "deg")
             .replace("\u0394", "Delta")
             .replace("\u00d7", "x")
             .replace("\u2014", "-")
             .replace("\u2013", "-")
             .replace(" ", "_"))


def train_xgb_with_shap(X_tr, y_tr, X_te, y_te, feature_names):
    """Train XGBoost and compute SHAP values via built-in pred_contribs."""
    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr).astype(np.float64)
    X_te_s = sc.transform(X_te).astype(np.float64)
    safe_names = [_sanitize(n) for n in feature_names]
    X_tr_df = pd.DataFrame(X_tr_s, columns=safe_names)
    X_te_df = pd.DataFrame(X_te_s, columns=safe_names)
    model = xgb.XGBRegressor(
        n_estimators=400, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
        random_state=RANDOM_STATE, n_jobs=-1, verbosity=0,
    )
    model.fit(X_tr_df, y_tr)
    dtest = xgb.DMatrix(X_te_df, feature_names=safe_names)
    contribs = model.get_booster().predict(dtest, pred_contribs=True)
    sv = contribs[:, :-1]  # last col is expected_value
    return model, sv, X_te_s, sc, safe_names


# ══════════════════════════════════════════════════════════════════════════════
# (a) Per-substrate SHAP: leave-one-substrate-out
# ══════════════════════════════════════════════════════════════════════════════
def per_substrate_shap(df, X, names, y):
    rows = []
    for sub in sorted(df["reactant_name"].unique()):
        mask = df["reactant_name"].values == sub
        tr_idx = np.where(~mask)[0]
        te_idx = np.where(mask)[0]
        if len(te_idx) < 10:
            print(f"  Skipping {sub}: only {len(te_idx)} test rows.")
            continue
        model, sv, X_te_s, sc, safe_names = train_xgb_with_shap(
            X[tr_idx], y[tr_idx], X[te_idx], y[te_idx], names)
        mean_abs_sv = np.abs(sv).mean(axis=0)
        mean_sv = sv.mean(axis=0)
        for i, name in enumerate(names):
            rows.append({
                "substrate_held": sub,
                "feature": name,
                "mean_abs_shap": float(mean_abs_sv[i]),
                "mean_signed_shap": float(mean_sv[i]),
                "n_test": int(len(te_idx)),
            })
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# (b) Per-(substrate, mechanism) SHAP: leave-one-(sub, mech)-out
# ══════════════════════════════════════════════════════════════════════════════
def per_pair_shap(df, X, names, y, min_n=10):
    rows = []
    combos = df.groupby(["reactant_name", "catalyst_system_type_agg"]).size().reset_index(name="n")
    for _, row in combos.iterrows():
        sub = row["reactant_name"]
        mech = row["catalyst_system_type_agg"]
        n = row["n"]
        if n < min_n:
            continue
        mask = ((df["reactant_name"].values == sub) &
                (df["catalyst_system_type_agg"].values == mech))
        tr_idx = np.where(~mask)[0]
        te_idx = np.where(mask)[0]
        model, sv, X_te_s, sc, safe_names = train_xgb_with_shap(
            X[tr_idx], y[tr_idx], X[te_idx], y[te_idx], names)
        mean_abs_sv = np.abs(sv).mean(axis=0)
        mean_sv = sv.mean(axis=0)
        for i, name in enumerate(names):
            rows.append({
                "substrate": sub,
                "mechanism": mech,
                "feature": name,
                "mean_abs_shap": float(mean_abs_sv[i]),
                "mean_signed_shap": float(mean_sv[i]),
                "n_test": int(len(te_idx)),
            })
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# (c) SHAP direction flip
# ══════════════════════════════════════════════════════════════════════════════
def direction_flip(per_sub_df: pd.DataFrame,
                   out_csv: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    pivot = per_sub_df.pivot(
        index="feature", columns="substrate_held",
        values="mean_signed_shap").fillna(0)
    subs = list(pivot.columns)
    flip_rows = []
    for i, sa in enumerate(subs):
        for sb in subs[i + 1:]:
            diff = pivot[sa] - pivot[sb]
            abs_diff = np.abs(diff)
            n_flip = int((abs_diff > 1.0).sum())
            flip_rows.append({
                "sub_a": sa,
                "sub_b": sb,
                "n_features_total": int(pivot.shape[0]),
                "n_features_strong_flip": n_flip,
                "frac_strong_flip": float(n_flip / pivot.shape[0]),
                "max_abs_diff": float(abs_diff.max()),
                "max_abs_diff_feature": pivot.index[abs_diff.argmax()],
            })
    flip_df = pd.DataFrame(flip_rows).sort_values("frac_strong_flip", ascending=False)
    flip_df.to_csv(out_csv, index=False)
    return flip_df, pivot


# ══════════════════════════════════════════════════════════════════════════════
# Plotting
# ══════════════════════════════════════════════════════════════════════════════
def plot_per_substrate_heatmap(per_sub_df: pd.DataFrame, out_path: str, top_k=15):
    top_feats = (per_sub_df.groupby("feature")["mean_abs_shap"].mean()
                 .sort_values(ascending=False).head(top_k).index.tolist())
    sub_df = per_sub_df[per_sub_df["feature"].isin(top_feats)]
    pivot = sub_df.pivot(index="feature", columns="substrate_held",
                          values="mean_abs_shap")
    pivot = pivot.loc[top_feats]
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns, rotation=30, ha="right")
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels(pivot.index, fontsize=9)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=7,
                    color="white" if v > 7 else "black")
    plt.colorbar(im, ax=ax, label="mean |SHAP| (test rows)")
    ax.set_title(f"Per-substrate SHAP magnitude — top {top_k} features\n"
                 "(trained on remaining 4 substrates)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_direction_flip(pivot: pd.DataFrame, out_path: str, top_k=12):
    top_feats = pivot.var(axis=1).sort_values(ascending=False).head(top_k).index.tolist()
    plot_df = pivot.loc[top_feats]
    fig, ax = plt.subplots(figsize=(9, 6))
    width = 0.15
    x = np.arange(len(top_feats))
    for i, sub in enumerate(plot_df.columns):
        ax.bar(x + (i - 2) * width, plot_df[sub].values, width, label=sub)
    ax.axhline(0, color="black", linewidth=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(top_feats, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Mean signed SHAP\n(negative = lower yield, positive = higher)")
    ax.set_title(f"SHAP value direction per substrate — top {top_k} highest-variance features")
    ax.legend(loc="best", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_per_pair_heatmap(per_pair_df: pd.DataFrame, out_path: str, top_k=15):
    top_feats = (per_pair_df.groupby("feature")["mean_abs_shap"].mean()
                 .sort_values(ascending=False).head(top_k).index.tolist())
    sub_df = per_pair_df[per_pair_df["feature"].isin(top_feats)].copy()
    sub_df["pair"] = (
        sub_df["substrate"].str.replace(" oxide", "").str.replace(" glycidyl ether", "-GGE")
        + " \u00d7 " + sub_df["mechanism"])
    pivot = sub_df.pivot(index="feature", columns="pair",
                          values="mean_abs_shap").fillna(0)
    pivot = pivot.loc[top_feats]
    fig, ax = plt.subplots(figsize=(13, 7))
    im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns, rotation=40, ha="right", fontsize=8)
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels(pivot.index, fontsize=9)
    plt.colorbar(im, ax=ax, label="mean |SHAP|")
    ax.set_title(f"Per-(substrate, mechanism) SHAP magnitude — top {top_k} features")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# Chemical interpretation helpers (lightweight, no heavy imports)
# ══════════════════════════════════════════════════════════════════════════════
_FEATURE_INTERPRETATIONS = {
    "sub_homo_eV": {
        "name": "Substrate HOMO Energy",
        "unit": "eV",
        "positive": "Higher HOMO → stronger nucleophile → easier ring-opening",
        "negative": "Lower HOMO → weaker nucleophile → harder ring-opening",
    },
    "sub_lumo_eV": {
        "name": "Substrate LUMO Energy",
        "unit": "eV",
        "positive": "Higher LUMO → weaker electrophilicity",
        "negative": "Lower LUMO → stronger electrophilicity",
    },
    "delta_E_HL": {
        "name": "HOMO-LUMO Gap (cat HOMO − sub LUMO)",
        "unit": "eV",
        "positive": "Larger gap → harder electron transfer → lower yield",
        "negative": "Smaller gap → easier electron transfer → higher yield",
    },
    "temperature": {
        "name": "Reaction Temperature",
        "unit": "\u00b0C",
        "positive": "Higher temperature → faster reaction kinetics",
        "negative": "Lower temperature → slower kinetics",
    },
    "pressure": {
        "name": "CO2 Pressure",
        "unit": "MPa",
        "positive": "Higher pressure → more dissolved CO2 → faster cycloaddition",
        "negative": "Lower pressure → less CO2 availability",
    },
    "time_log": {
        "name": "Reaction Time (log-scale)",
        "unit": "log(h)",
        "positive": "Longer reaction time → higher conversion",
        "negative": "Shorter reaction time → lower conversion",
    },
    "loading_log": {
        "name": "Catalyst Loading (log-scale)",
        "unit": "log(mol%)",
        "positive": "Higher loading → more active sites → faster reaction",
        "negative": "Lower loading → fewer active sites",
    },
}


def get_feature_interpretation(feature: str, signed_shap: float) -> dict:
    """Return chemical interpretation dict for a feature."""
    base = _FEATURE_INTERPRETATIONS.get(feature, {
        "name": feature,
        "unit": "",
        "positive": "N/A",
        "negative": "N/A",
    })
    direction = "positive" if signed_shap >= 0 else "negative"
    return {
        "feature": feature,
        "interpretation": base[direction],
        "signed_shap": float(signed_shap),
        "direction": "increases yield" if signed_shap >= 0 else "decreases yield",
    }


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
def main():
    print(f"=== Step 4.5: per-substrate / per-pair SHAP analysis ===\n")
    print(f"  DATA_CSV : {DATA_CSV}")
    print(f"  MECH_CSV : {MECH_CSV}")
    print(f"  OUT_DIR  : {OUT_DIR}\n")

    df = load_data()
    print(f"Loaded {len(df)} valid rows, "
          f"{df['reactant_name'].nunique()} substrates.")

    # Build features
    X_base, names = build_xtb_cond_features(df)
    X, names_used = add_mech_one_hot(X_base, names, df)
    y = df["yield (%)"].to_numpy(dtype=np.float64)
    print(f"Feature matrix: {X.shape}, features: {names_used}\n")

    # (a) Per-substrate SHAP
    print("--- (a) per-substrate SHAP ---")
    per_sub_df = per_substrate_shap(df, X, names_used, y)
    per_sub_df.to_csv(
        os.path.join(OUT_DIR, "per_substrate_shap.csv"), index=False)
    print(f"Saved {len(per_sub_df)} rows to per_substrate_shap.csv.\n")

    # Top-5 features per substrate (with chemical interpretation)
    top_per_sub = []
    for sub, grp in per_sub_df.groupby("substrate_held"):
        top5 = grp.sort_values("mean_abs_shap", ascending=False).head(5)
        for rank, (_, row) in enumerate(top5.iterrows(), start=1):
            interp = get_feature_interpretation(row["feature"], row["mean_signed_shap"])
            top_per_sub.append({
                "substrate": sub,
                "rank": rank,
                "feature": row["feature"],
                "mean_abs_shap": row["mean_abs_shap"],
                "mean_signed_shap": row["mean_signed_shap"],
                "n_test": row["n_test"],
                "interpretation": interp["interpretation"],
                "direction": interp["direction"],
            })
    top_per_sub_df = pd.DataFrame(top_per_sub)
    top_per_sub_df.to_csv(
        os.path.join(OUT_DIR, "per_substrate_top_features.csv"), index=False)
    print("Top features with chemical interpretations:")
    print(top_per_sub_df[["substrate", "feature", "interpretation", "direction"]]
          .to_string(index=False))
    print()

    # (b) Direction flip
    print("--- (b) SHAP direction flip ---")
    flip_df, pivot_signed = direction_flip(
        per_sub_df, os.path.join(OUT_DIR, "per_substrate_shap_direction.csv"))
    print("\nStrongest substrate pairs (by SHAP direction flip):")
    print(flip_df.head(10).to_string())
    print()

    # (c) Per-pair SHAP
    print("--- (c) per-(substrate, mechanism) SHAP ---")
    per_pair_df = per_pair_shap(df, X, names_used, y)
    per_pair_df.to_csv(
        os.path.join(OUT_DIR, "per_pair_shap.csv"), index=False)
    print(f"Saved {len(per_pair_df)} rows to per_pair_shap.csv.\n")

    # Summary JSON
    summary = {
        "n_rows": int(len(df)),
        "n_features": int(len(names_used)),
        "n_substrates": int(df["reactant_name"].nunique()),
        "top_features_per_substrate": {
            sub: (per_sub_df[per_sub_df["substrate_held"] == sub]
                  .sort_values("mean_abs_shap", ascending=False)
                  .head(5)[["feature", "mean_abs_shap", "mean_signed_shap"]]
                  .to_dict(orient="records"))
            for sub in sorted(df["reactant_name"].unique())
        },
        "strongest_direction_flips": flip_df.head(5).to_dict(orient="records"),
        "per_pair_count": int(
            len(per_pair_df["substrate"].unique()) *
            len(per_pair_df["mechanism"].unique())),
    }
    with open(os.path.join(OUT_DIR, "per_substrate_top_features_summary.json"),
              "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # Plots
    print("--- plots ---")
    plot_per_substrate_heatmap(
        per_sub_df, os.path.join(FIG_DIR, "per_substrate_shap_heatmap.png"))
    plot_direction_flip(
        pivot_signed, os.path.join(FIG_DIR, "shap_direction_flip.png"))
    plot_per_pair_heatmap(
        per_pair_df, os.path.join(FIG_DIR, "per_pair_shap_heatmap.png"))
    print("Saved 3 figures.\n")

    # (d) Global SHAP (full model, all data)
    print("--- (d) global SHAP (full model) ---")
    try:
        global_model, global_sv, _, _, global_names = train_xgb_with_shap(
            X, y, X, y, names_used)
        global_sv_df = pd.DataFrame(global_sv, columns=global_names)
        global_sv_df.insert(0, "row_index", np.arange(len(global_sv_df)))
        global_sv_df.insert(1, "reactant_name", df["reactant_name"].values)
        global_sv_path = os.path.join(OUT_DIR, "shap_xtb_values.csv")
        global_sv_df.to_csv(global_sv_path, index=False)
        print(f"Saved global SHAP matrix: {global_sv_path} ({global_sv_df.shape}).")
    except Exception as e:
        print(f"Global SHAP step failed: {e}")

    # Legacy shap_xtb_summary.json (expected by other consumers)
    try:
        global_top = (per_sub_df.groupby("feature", as_index=False)["mean_abs_shap"]
                      .mean()
                      .sort_values("mean_abs_shap", ascending=False)
                      .head(10)
                      .reset_index(drop=True))
        global_top.insert(0, "rank", global_top.index + 1)
        global_top_pretty = [
            {"rank": int(r["rank"]),
             "name": str(r["feature"]),
             "mean_abs_shap": float(r["mean_abs_shap"])}
            for _, r in global_top.iterrows()
        ]
        delta_row = global_top[global_top["feature"] == "delta_E_HL"]
        if not delta_row.empty:
            dr = delta_row.iloc[0]
            delta_payload = {
                "rank": int(dr["rank"]),
                "mean_abs_shap": float(dr["mean_abs_shap"]),
                "correlation": 0.0,
                "interpretation": "Larger deltaE_HL leads to higher yield",
            }
        else:
            delta_payload = {
                "rank": -1,
                "mean_abs_shap": float("nan"),
                "correlation": 0.0,
                "interpretation": "delta_E_HL not in top-10",
            }
        legacy_summary = {
            "r2_validation": 0.0,
            "rmse_validation": 0.0,
            "n_features": int(len(names_used)),
            "n_shap_samples": int(len(df)),
            "top_10": global_top_pretty,
            "delta_E_HL": delta_payload,
        }
        with open(os.path.join(OUT_DIR, "shap_xtb_summary.json"),
                  "w", encoding="utf-8") as f:
            json.dump(legacy_summary, f, indent=2, ensure_ascii=False)
        print("Saved legacy schema: shap_xtb_summary.json")
    except Exception as e:
        print(f"Legacy SHAP summary step failed: {e}")

    # Report
    with open(os.path.join(OUT_DIR, "report.txt"), "w", encoding="utf-8") as f:
        f.write("Step 4.5 -- Per-substrate / per-pair SHAP analysis\n")
        f.write(f"Generated: {datetime.utcnow().isoformat()}\n\n")
        f.write(f"Rows: {len(df)}, Features: {len(names_used)}\n\n")
        f.write("=== Top 5 features per held-out substrate ===\n")
        for sub in sorted(df["reactant_name"].unique()):
            sub_top = (per_sub_df[per_sub_df["substrate_held"] == sub]
                       .sort_values("mean_abs_shap", ascending=False).head(5))
            f.write(f"\n--- {sub} (n_test={sub_top['n_test'].iloc[0]}) ---\n")
            f.write(sub_top[["feature", "mean_abs_shap", "mean_signed_shap"]]
                    .to_string(index=False))
            f.write("\n")
        f.write("\n\n=== Strongest direction-flip pairs (|ΔSHAP| > 1.0) ===\n")
        f.write(flip_df.head(8).to_string(index=False))
        f.write("\n")

    print(f"\nDone. All outputs in {OUT_DIR}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--force", action="store_true")
    p.parse_args()
    main()
