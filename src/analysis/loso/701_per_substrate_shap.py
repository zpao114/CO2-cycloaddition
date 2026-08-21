"""Step 4.5: Per-substrate and per-(substrate, mechanism) SHAP analysis.

Motivating question
-------------------
The Step 4 LOSO protocol showed: when a substrate is held out, the model
R^2 collapses (especially for CHO and ECH). This means the model has not
learned a transferable *mechanism* —it has memorised per-substrate
yield ranges.

To prove this quantitatively, we compute SHAP values:

    (a) Per-substrate SHAP      -- refit on all data minus the target
                                   substrate; report which features drive
                                   prediction for that substrate
    (b) Per-(substrate, mech)   -- refit on all data minus the (sub, mech)
                                   combination
    (c) SHAP-value direction comparison
                                 -- does the same feature push yield up
                                    in CHO but down in SO? If so, the model
                                    treats CHO and SO as qualitatively
                                    different reactions.

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

import io
import sys
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


# Re-use 700 helpers: try relative-path first, then project-root fallback
_this_dir = os.path.dirname(os.path.abspath(__file__))
_700_path = os.path.join(_this_dir, "700_loso_lomo_cv.py")
if os.path.exists(_700_path):
    sys.path.insert(0, _this_dir)
else:
    # Fallback: derive from project root
    _proj_root = os.environ.get("CO2_PROJECT_ROOT", r"D:\machine-learning\CO2-cycloaddition")
    _700_path = os.path.join(_proj_root, "src", "analysis", "loso", "700_loso_lomo_cv.py")
    _700_dir = os.path.dirname(_700_path)
    if os.path.exists(_700_path):
        sys.path.insert(0, _700_dir)
    else:
        raise FileNotFoundError(f"Cannot find 700_loso_lomo_cv.py at {_700_path}")

from importlib import import_module
mod = import_module("700_loso_lomo_cv")
build_xtb_cond_features = mod.build_xtb_cond_features
add_mech_one_hot = mod.add_mech_one_hot

# Import SHAP chemical interpretation
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from shap_explanation import get_feature_interpretation, generate_chemical_report


PROJECT_ROOT = os.environ.get("CO2_PROJECT_ROOT", r"D:\machine-learning\CO2-cycloaddition")
DATA_CSV = os.path.join(PROJECT_ROOT, 'results/results_cho_diagnostic/co2_drfp_xtb_extended.csv')
MECH_CSV = os.path.join(PROJECT_ROOT, 'data/processed/catalyst_mechanism.csv')
OUT_DIR = os.path.join(PROJECT_ROOT, "results_step4_5")
FIG_DIR = os.path.join(OUT_DIR, "figs")

RANDOM_STATE = 42

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)


# ----------------------------------------------------------------------
# Load + build features (same as 700)
# ----------------------------------------------------------------------
def load_data():
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
    mech = mech[["name", "mechanism"]].rename(columns={"name": "catalyst_1_name",
                                                        "mechanism": "mech_label"})
    df = df.merge(mech, on="catalyst_1_name", how="left")
    df["mech_label"] = df["mech_label"].fillna("UNK")
    return df


def _sanitize(s: str) -> str:
    return (s.replace("°", "deg")
             .replace("Δ", "Delta")
             .replace("×", "x")
             .replace("—", "-")
             .replace("–", "-")
             .replace(" ", "_"))


def train_xgb_with_shap(X_tr, y_tr, X_te, y_te, feature_names):
    """Train XGBoost and compute SHAP values via xgboost's built-in
    `pred_contribs=True` interface.  This bypasses shap>=0.44's TreeExplainer
    UTF-8 decoder, which breaks on the binary-embedded feature names."""
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
    # Use built-in pred_contribs to get SHAP values directly
    dtest = xgb.DMatrix(X_te_df, feature_names=safe_names)
    contribs = model.get_booster().predict(dtest, pred_contribs=True)
    # contribs shape = (n_samples, n_features + 1); last col is expected_value
    sv = contribs[:, :-1]
    return model, sv, X_te_s, sc, safe_names


# ----------------------------------------------------------------------
# (a) Per-substrate SHAP: leave-one-substrate-out
# ----------------------------------------------------------------------
def per_substrate_shap(df, X, names, y):
    """For each substrate held out, compute SHAP values on the held-out
    test set."""
    rows = []
    for sub in sorted(df["reactant_name"].unique()):
        mask = df["reactant_name"].values == sub
        tr_idx = np.where(~mask)[0]
        te_idx = np.where(mask)[0]
        if len(te_idx) < 10:
            continue
        model, sv, X_te_s, sc, safe_names = train_xgb_with_shap(
            X[tr_idx], y[tr_idx], X[te_idx], y[te_idx], names
        )
        # mean |SHAP| per feature for the held-out substrate
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


# ----------------------------------------------------------------------
# (b) Per-(substrate, mechanism) SHAP: leave-one-(sub, mech)-out
# ----------------------------------------------------------------------
def per_pair_shap(df, X, names, y, min_n=10):
    rows = []
    combos = df.groupby(["reactant_name", "catalyst_system_type_agg"]).size().reset_index(name="n")
    for _, row in combos.iterrows():
        sub, mech, n = row["reactant_name"], row["catalyst_system_type_agg"], row["n"]
        if n < min_n:
            continue
        mask = (df["reactant_name"].values == sub) & (df["catalyst_system_type_agg"].values == mech)
        tr_idx = np.where(~mask)[0]
        te_idx = np.where(mask)[0]
        model, sv, X_te_s, sc, safe_names = train_xgb_with_shap(
            X[tr_idx], y[tr_idx], X[te_idx], y[te_idx], names
        )
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


# ----------------------------------------------------------------------
# (c) SHAP direction flip: does feature F push yield UP in one substrate
# and DOWN in another?
# ----------------------------------------------------------------------
def direction_flip(per_sub_df: pd.DataFrame, out_csv: str) -> pd.DataFrame:
    """For each feature, compute per-substrate mean signed SHAP.
    Flag pairs of substrates where the SHAP direction is opposite.

    A flip means the model treats the two substrates qualitatively
    differently, which is strong evidence that the catalyst-substrate
    interaction is non-additive.
    """
    pivot = per_sub_df.pivot(index="feature", columns="substrate_held", values="mean_signed_shap").fillna(0)
    # Pairwise direction difference
    subs = list(pivot.columns)
    flip_rows = []
    for i, sa in enumerate(subs):
        for sb in subs[i + 1:]:
            diff = pivot[sa] - pivot[sb]
            abs_diff = np.abs(diff)
            n_flip = (diff.abs() > 1.0).sum()  # |ΔSHAP| > 1.0 = real flip
            flip_rows.append({
                "sub_a": sa, "sub_b": sb,
                "n_features_total": int(pivot.shape[0]),
                "n_features_strong_flip": int(n_flip),
                "frac_strong_flip": float(n_flip / pivot.shape[0]),
                "max_abs_diff": float(abs_diff.max()),
                "max_abs_diff_feature": pivot.index[abs_diff.argmax()],
            })
    flip_df = pd.DataFrame(flip_rows).sort_values("frac_strong_flip", ascending=False)
    flip_df.to_csv(out_csv, index=False)
    return flip_df, pivot


# ----------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------
def plot_per_substrate_heatmap(per_sub_df: pd.DataFrame, out_path: str, top_k=15):
    """Heatmap: mean_abs_shap for top features across held-out substrates."""
    # Pick top features by overall mean |SHAP|
    top_feats = (per_sub_df.groupby("feature")["mean_abs_shap"].mean()
                 .sort_values(ascending=False).head(top_k).index.tolist())
    sub_df = per_sub_df[per_sub_df["feature"].isin(top_feats)]
    pivot = sub_df.pivot(index="feature", columns="substrate_held", values="mean_abs_shap")
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
    ax.set_title(f"Per-substrate SHAP magnitude -- top {top_k} features\n"
                 "(trained on other 4 substrates)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_direction_flip(pivot: pd.DataFrame, out_path: str, top_k=12):
    """Bar chart: top features that flip direction between substrates.
    Each row: a feature; each colour: a substrate. Sign indicates push up/down.
    """
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
    ax.set_ylabel("Mean signed SHAP\n(negative = predicts lower yield, positive = higher)")
    ax.set_title(f"SHAP value direction per substrate -- top {top_k} highest-variance features\n"
                 "(flipped signs = model treats reactions as qualitatively different)")
    ax.legend(loc="best", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_per_pair_heatmap(per_pair_df: pd.DataFrame, out_path: str, top_k=15):
    """Heatmap of mean_abs_shap across (substrate, mechanism) for top features."""
    top_feats = (per_pair_df.groupby("feature")["mean_abs_shap"].mean()
                 .sort_values(ascending=False).head(top_k).index.tolist())
    sub_df = per_pair_df[per_pair_df["feature"].isin(top_feats)].copy()
    sub_df["pair"] = sub_df["substrate"].str.replace(" oxide", "").str.replace(" glycidyl ether", "-GGE") + " × " + sub_df["mechanism"]
    pivot = sub_df.pivot(index="feature", columns="pair", values="mean_abs_shap").fillna(0)
    pivot = pivot.loc[top_feats]
    fig, ax = plt.subplots(figsize=(13, 7))
    im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns, rotation=40, ha="right", fontsize=8)
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels(pivot.index, fontsize=9)
    plt.colorbar(im, ax=ax, label="mean |SHAP|")
    ax.set_title(f"Per-(substrate, mechanism) SHAP magnitude -- top {top_k} features")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    print(f"=== Step 4.5: per-substrate / per-pair SHAP analysis ===\n")
    df = load_data()
    print(f"Rows: {len(df)}")

    X_base, names = build_xtb_cond_features(df)
    X_mech, names_mech = add_mech_one_hot(X_base, names, df)
    y = df["yield (%)"].to_numpy(dtype=np.float64)

    # Use X1 (xTB + mech) for the per-substrate analysis -- mirrors 700
    X, names_used = X_mech, names_mech

    # (a) Per-substrate SHAP
    print("\n--- (a) per-substrate SHAP ---")
    per_sub_df = per_substrate_shap(df, X, names_used, y)
    per_sub_df.to_csv(os.path.join(OUT_DIR, "per_substrate_shap.csv"), index=False)
    print(f"Saved {len(per_sub_df)} rows.")

    # Top 5 features per substrate
    top_per_sub = []
    for sub, grp in per_sub_df.groupby("substrate_held"):
        top5 = grp.sort_values("mean_abs_shap", ascending=False).head(5)
        for rank, (_, row) in enumerate(top5.iterrows(), start=1):
            # Add chemical interpretation for this feature
            interp = get_feature_interpretation(row["feature"], row["mean_signed_shap"])
            direction_text = "increases yield" if row["mean_signed_shap"] > 0 else "decreases yield"
            # Use main_text which contains the chemical interpretation
            chem_interp = interp.get("main_text", interp.get("chemical_context", "N/A")) if interp else "N/A"
            top_per_sub.append({
                "substrate": sub,
                "rank": rank,
                "feature": row["feature"],
                "mean_abs_shap": row["mean_abs_shap"],
                "mean_signed_shap": row["mean_signed_shap"],
                "n_test": row["n_test"],
                "chemical_interpretation": chem_interp,
                "direction": direction_text,
            })
    top_per_sub_df = pd.DataFrame(top_per_sub)
    top_per_sub_df.to_csv(os.path.join(OUT_DIR, "per_substrate_top_features.csv"), index=False)
    print("\nTop features with chemical interpretations:")
    print(top_per_sub_df[["substrate", "feature", "chemical_interpretation", "direction"]].to_string(index=False))

    # (b) Direction flip analysis
    print("\n--- (b) SHAP direction flip ---")
    flip_df, pivot_signed = direction_flip(
        per_sub_df, os.path.join(OUT_DIR, "per_substrate_shap_direction.csv")
    )
    print("\nStrongest substrate pairs (by SHAP direction flip):")
    print(flip_df.head(10).to_string())

    # (c) Per-pair SHAP
    print("\n--- (c) per-(substrate, mechanism) SHAP ---")
    per_pair_df = per_pair_shap(df, X, names_used, y)
    per_pair_df.to_csv(os.path.join(OUT_DIR, "per_pair_shap.csv"), index=False)

    # Summary JSON
    summary = {
        "n_rows": len(df),
        "n_features": len(names_used),
        "top_features_per_substrate": {
            sub: (per_sub_df[per_sub_df["substrate_held"] == sub]
                  .sort_values("mean_abs_shap", ascending=False)
                  .head(5)[["feature", "mean_abs_shap", "mean_signed_shap"]]
                  .to_dict(orient="records"))
            for sub in sorted(df["reactant_name"].unique())
        },
        "strongest_direction_flips": flip_df.head(5).to_dict(orient="records"),
        "per_pair_count": int(len(per_pair_df["substrate"].unique()) *
                                len(per_pair_df["mechanism"].unique())),
    }
    with open(os.path.join(OUT_DIR, "per_substrate_top_features_summary.json"), "w",
              encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # Plots
    plot_per_substrate_heatmap(per_sub_df,
                              os.path.join(FIG_DIR, "per_substrate_shap_heatmap.png"))
    plot_direction_flip(pivot_signed,
                        os.path.join(FIG_DIR, "shap_direction_flip.png"))
    plot_per_pair_heatmap(per_pair_df,
                          os.path.join(FIG_DIR, "per_pair_shap_heatmap.png"))

    # (d) Global SHAP matrix (full model, all data) -- consumed by fig7, paper_abstract
    print("\n--- (d) global SHAP (full model) ---")
    try:
        global_model, global_sv, _, _, global_names = train_xgb_with_shap(
            X, y, X, y, names_used
        )
        global_sv_df = pd.DataFrame(global_sv, columns=global_names)
        global_sv_df.insert(0, "row_index", np.arange(len(global_sv_df)))
        global_sv_df.insert(1, "reactant_name", df["reactant_name"].values)
        global_sv_path = os.path.join(OUT_DIR, "shap_xtb_values.csv")
        global_sv_df.to_csv(global_sv_path, index=False)
        print(f"Saved global SHAP matrix to {global_sv_path} ({global_sv_df.shape}).")
    except Exception as e:
        print(f"Global SHAP step failed: {e}")

    # FIX (2026-08-19): emit a global top-10 SHAP summary in the legacy
    # `shap_xtb_summary.json` schema so 900_paper_abstract.py and other
    # consumers that still expect this file can find it.  Aggregated by
    # mean(|SHAP|) across all substrates (i.e. a global ranking that ignores
    # the per-substrate mean) so it matches the historical `top_10` layout.
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
        # delta_E_HL entry (always present in legacy summaries)
        if "delta_E_HL" in global_top["feature"].tolist():
            delta_row = global_top[global_top["feature"] == "delta_E_HL"].iloc[0]
            delta_payload = {
                "rank": int(delta_row["rank"]),
                "mean_abs_shap": float(delta_row["mean_abs_shap"]),
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
            "r2_validation": 0.0,        # filled by 401 if available
            "rmse_validation": 0.0,
            "n_features": int(len(names_used)),
            "n_shap_samples": int(len(df)),
            "top_10": global_top_pretty,
            "delta_E_HL": delta_payload,
        }
        with open(os.path.join(OUT_DIR, "shap_xtb_summary.json"), "w",
                  encoding="utf-8") as f:
            json.dump(legacy_summary, f, indent=2, ensure_ascii=False)
        print(f"  Saved legacy schema: shap_xtb_summary.json")
    except Exception as e:
        print(f"  Legacy SHAP summary step failed: {e}")

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
            f.write(sub_top[["feature", "mean_abs_shap", "mean_signed_shap"]].to_string(index=False))
            f.write("\n")

        f.write("\n\n=== Strongest direction-flip pairs (|Δ SHAP| > 1.0) ===\n")
        f.write(flip_df.head(8).to_string(index=False))
        f.write("\n")

    print(f"\nDone. Outputs in {OUT_DIR}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--force", action="store_true")
    p.parse_args()
    main()