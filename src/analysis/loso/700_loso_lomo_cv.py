"""Step 4: Leave-One-Substrate-Out (LOSO) and Leave-One-Mechanism-Out (LOMO) CV.

Question
--------
If we train the standard XGB on 4 substrates and predict the 5th, do we get
near-zero R^2 (i.e., zero transferability)?

Motivated by Step 1+2+3 finding:
    Cyclohexene oxide (CHO) is a structural outlier.  Removing it from the
    training set should massively improve *within-distribution* R^2 for the
    other 4 substrates, and should *systematically* fail on CHO predictions.

Pipeline
--------
1. Mirror 902_cho_mechanistic_diagnostic.build_xtb_cond_features (25 features).
2. Add the new mechanism labels (NUC / LAC / BAS / BIF / OTH) as one-hot
   columns so the model knows WHICH mechanism class this catalyst belongs to.
3. Three CV protocols:
       A. Standard 5-fold KFold (baseline).
       B. Leave-One-Substrate-Out (LOSO).
       C. Leave-One-Mechanism-Out (LOMO).
       D. LOSO + LOMO combined (LOSO×LOMO).
4. Two feature sets:
       - X0 = original 25 features (no mech label).
       - X1 = X0 + 5 one-hot mech_label columns.
5. Per cell report: R^2, MAE, RMSE, n_train, n_test.
6. Diagnostics:  predicted-vs-actual scatter for LOSO of each substrate
   (highlighting CHO as the systematic outlier).

Outputs
-------
results_step4/loso_results.csv
results_step4/lomo_results.csv
results_step4/lomo_loso_results.csv
results_step4/summary_protocol.csv
results_step4/figs/loso_scatter.png
results_step4/figs/loso_metrics_bar.png
results_step4/figs/heatmap_protocol_R2.png
results_step4/figs/heatmap_protocol_MAE.png
results_step4/report.txt
"""
from __future__ import annotations
import os

import io
import sys
import warnings
import json
from datetime import datetime, timezone

warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import xgboost as xgb


PROJECT_ROOT = os.environ.get("CO2_PROJECT_ROOT", r"D:\machine-learning\CO2-cycloaddition")
DATA_CSV = os.path.join(PROJECT_ROOT, 'results/results_cho_diagnostic/co2_drfp_xtb_extended.csv')
MECH_CSV = os.path.join(PROJECT_ROOT, 'data/processed/catalyst_mechanism.csv')
OUT_DIR = os.path.join(PROJECT_ROOT, "results_step4")

PRIMARY_FAMILIES = ["ionic_liquid", "metal_halide", "mixed_system"]
AGGREGATED_OTHER_LABEL = "other"

RANDOM_STATE = 42
N_SPLITS = 5

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, "figs"), exist_ok=True)


# ----------------------------------------------------------------------
# Feature builder (mirror 902)
# ----------------------------------------------------------------------
def build_xtb_cond_features(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
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

    temp_col = ([c for c in df.columns if "temperature" in c.lower()] or [None])[0]
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


# ----------------------------------------------------------------------
# Load + attach mechanism label
# ----------------------------------------------------------------------
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_CSV, encoding="utf-8-sig")
    df = df[df["extraction_status"] == "valid"].copy()
    df = df.dropna(subset=["yield (%)"])
    df = df[df["yield (%)"] > 0].reset_index(drop=True)
    df["catalyst_system_type_agg"] = np.where(
        df["catalyst_system_type"].isin(PRIMARY_FAMILIES),
        df["catalyst_system_type"],
        AGGREGATED_OTHER_LABEL,
    )
    # Attach mechanism label from Step 1
    mech = pd.read_csv(MECH_CSV)
    mech = mech[["name", "mechanism"]].rename(columns={"name": "catalyst_1_name",
                                                        "mechanism": "mech_label"})
    df = df.merge(mech, on="catalyst_1_name", how="left")
    df["mech_label"] = df["mech_label"].fillna("UNK")
    return df


def add_mech_one_hot(X: np.ndarray, names: list[str], df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Append 5 one-hot columns for the 5 mechanism classes (NUC, LAC, BAS, BIF, OTH)."""
    mech_classes = ["NUC", "LAC", "BAS", "BIF", "OTH"]
    one_hot = np.zeros((len(df), len(mech_classes)), dtype=np.float64)
    for i, mc in enumerate(mech_classes):
        one_hot[:, i] = (df["mech_label"].values == mc).astype(np.float64)
    new_names = names + [f"mech_{m}" for m in mech_classes]
    return np.hstack([X, one_hot]), new_names


# ----------------------------------------------------------------------
# Train + score
# ----------------------------------------------------------------------
def train_xgb(X_tr, y_tr, X_te, y_te, tag="") -> dict:
    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr).astype(np.float64)
    X_te_s = sc.transform(X_te).astype(np.float64)
    model = xgb.XGBRegressor(
        n_estimators=400, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
        random_state=RANDOM_STATE, n_jobs=-1, verbosity=0,
    )
    model.fit(X_tr_s, y_tr)
    y_pred = model.predict(X_te_s)
    return {
        "r2": float(r2_score(y_te, y_pred)),
        "mae": float(mean_absolute_error(y_te, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_te, y_pred))),
        "y_true": y_te.tolist(),
        "y_pred": y_pred.tolist(),
        "n_train": int(len(y_tr)),
        "n_test": int(len(y_te)),
    }


def aggregate(results: list[dict]) -> dict:
    y_true = np.concatenate([np.asarray(r["y_true"]) for r in results])
    y_pred = np.concatenate([np.asarray(r["y_pred"]) for r in results])
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "n_train": int(sum(r["n_train"] for r in results) // len(results)),
        "n_test": int(sum(r["n_test"] for r in results)),
    }


# ----------------------------------------------------------------------
# Protocols
# ----------------------------------------------------------------------
def run_kfold(X, y, df, n_splits=N_SPLITS):
    """Standard 5-fold KFold (baseline)."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    for fold, (tr_idx, te_idx) in enumerate(kf.split(X)):
        res = train_xgb(X[tr_idx], y[tr_idx], X[te_idx], y[te_idx])
        rows.append({"fold": fold, **res,
                     "substrate_held": "all", "mech_held": "all"})
    return pd.DataFrame(rows), aggregate(rows)


def run_loso(X, y, df):
    """Leave-One-Substrate-Out."""
    rows = []
    for sub in sorted(df["reactant_name"].unique()):
        te_mask = df["reactant_name"].values == sub
        tr_idx = np.where(~te_mask)[0]
        te_idx = np.where(te_mask)[0]
        if len(te_idx) < 10:
            continue
        res = train_xgb(X[tr_idx], y[tr_idx], X[te_idx], y[te_idx])
        rows.append({"substrate_held": sub, **res, "mech_held": "all"})
    return pd.DataFrame(rows), aggregate(rows)


def run_lomo(X, y, df):
    """Leave-One-Mechanism-Out (using catalyst_system_type_agg)."""
    rows = []
    for mech in sorted(df["catalyst_system_type_agg"].unique()):
        te_mask = df["catalyst_system_type_agg"].values == mech
        tr_idx = np.where(~te_mask)[0]
        te_idx = np.where(te_mask)[0]
        if len(te_idx) < 10:
            continue
        res = train_xgb(X[tr_idx], y[tr_idx], X[te_idx], y[te_idx])
        rows.append({"mech_held": mech, **res, "substrate_held": "all"})
    return pd.DataFrame(rows), aggregate(rows)


def run_loso_lomo(X, y, df):
    """Double-LO: leave one (substrate, mechanism) combination out."""
    rows = []
    combos = df.groupby(["reactant_name", "catalyst_system_type_agg"]).size().reset_index(name="n")
    for _, row in combos.iterrows():
        sub, mech, n = row["reactant_name"], row["catalyst_system_type_agg"], row["n"]
        if n < 10:
            continue
        te_mask = (df["reactant_name"].values == sub) & (df["catalyst_system_type_agg"].values == mech)
        tr_idx = np.where(~te_mask)[0]
        te_idx = np.where(te_mask)[0]
        res = train_xgb(X[tr_idx], y[tr_idx], X[te_idx], y[te_idx])
        rows.append({"substrate_held": sub, "mech_held": mech, **res})
    return pd.DataFrame(rows), aggregate(rows)


# ----------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------
def plot_loso_scatter(loso_df, out_path):
    """Predicted vs actual per substrate (LOSO)."""
    subs = sorted(loso_df["substrate_held"].unique(),
                  key=lambda s: -aggregate([loso_df[loso_df["substrate_held"] == s].iloc[0].to_dict()])["mae"])
    n = len(subs)
    cols = min(n, 3)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4.5 * rows))
    axes = np.atleast_1d(axes).flatten()

    palette = {"Styrene oxide": "tab:blue", "Epichlorohydrin": "tab:orange",
                "Propylene oxide": "tab:green", "Cyclohexene oxide": "tab:red",
                "Isopropyl glycidyl ether": "tab:purple"}
    for i, sub in enumerate(subs):
        r = loso_df[loso_df["substrate_held"] == sub].iloc[0]
        ax = axes[i]
        ax.scatter(r["y_true"], r["y_pred"], alpha=0.4, c=palette.get(sub, "k"), s=15)
        ax.plot([0, 100], [0, 100], "k--", alpha=0.4)
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.set_xlabel("Actual yield (%)")
        ax.set_ylabel("Predicted yield (%)")
        ax.set_title(f"{sub}\nR²={r['r2']:.3f}, MAE={r['mae']:.1f}, n={r['n_test']}")
        ax.grid(alpha=0.3)
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")
    fig.suptitle("LOSO-CV: predicted vs actual (XGB, 25 xTB features)", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_heatmap_protocol(summary_df, out_path, value="r2", title_suffix="R²"):
    """Heatmap of (feature_set, protocol) × R² or MAE."""
    pivot = summary_df.pivot(index="protocol", columns="feature_set", values=value)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    im = ax.imshow(pivot.values, cmap="RdYlGn" if value == "r2" else "RdYlGn_r", aspect="auto")
    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels(pivot.index)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                    color="white" if v < 0.4 or value == "mae" else "black", fontsize=9)
    plt.colorbar(im, ax=ax, label=title_suffix)
    ax.set_title(f"Protocol × Feature set – {title_suffix}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_loso_bar(loso_df, out_path):
    """Bar chart of MAE per held-out substrate, with trainMAE shown as dotted."""
    subs = sorted(loso_df["substrate_held"].unique(),
                  key=lambda s: -loso_df[loso_df["substrate_held"] == s].iloc[0]["mae"])
    mae = [loso_df[loso_df["substrate_held"] == s].iloc[0]["mae"] for s in subs]
    r2 = [loso_df[loso_df["substrate_held"] == s].iloc[0]["r2"] for s in subs]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = ["tab:red" if s == "Cyclohexene oxide" else "tab:blue" for s in subs]
    ax.bar(subs, mae, color=colors, edgecolor="black")
    for i, (s, m, r) in enumerate(zip(subs, mae, r2)):
        ax.text(i, m + 1, f"R²={r:.2f}", ha="center", fontsize=9)
    ax.set_ylabel("MAE (%)")
    ax.set_title("LOSO-CV: MAE per held-out substrate\n(red = Cyclohexene oxide, the structural outlier)")
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    print(f"=== Step 4: LOSO / LOMO CV -- {datetime.now()} ===\n")
    df = load_data()
    print(f"Loaded rows: {len(df)}")
    print(f"Substrates: {df['reactant_name'].value_counts().to_dict()}")
    print(f"Mechanism families: {df['catalyst_system_type_agg'].value_counts().to_dict()}")
    print(f"Mechanism labels: {df['mech_label'].value_counts().to_dict()}\n")

    X_base, names = build_xtb_cond_features(df)
    X_mech, names_mech = add_mech_one_hot(X_base, names, df)
    y = df["yield (%)"].to_numpy(dtype=np.float64)

    print(f"Feature set X0: {X_base.shape[1]} (xTB + cond)")
    print(f"Feature set X1: {X_mech.shape[1]} (X0 + 5 mech one-hot)\n")

    protocols = [
        ("5fold",  run_kfold),
        ("LOSO",   run_loso),
        ("LOMO",   run_lomo),
        ("LOSO×LOMO", run_loso_lomo),
    ]
    feature_sets = {
        "X0_xTB_only":  (X_base, names),
        "X1_xTB+mech":  (X_mech, names_mech),
    }

    summary = []
    loso_per_sub_stats = {}
    for feat_name, (X, names) in feature_sets.items():
        for proto_name, proto_fn in protocols:
            print(f"--- {feat_name} × {proto_name} ---")
            df_proto, agg = proto_fn(X, y, df)
            print(f"  n_test={agg['n_test']}, R²={agg['r2']:.3f}, "
                  f"MAE={agg['mae']:.2f}, RMSE={agg['rmse']:.2f}\n")
            tag = f"{feat_name}__{proto_name}"
            df_proto.to_csv(os.path.join(OUT_DIR, f"{tag}.csv"), index=False)
            summary.append({
                "feature_set": feat_name,
                "protocol": proto_name,
                **agg,
            })
            if proto_name == "LOSO":
                loso_per_sub_stats[feat_name] = df_proto

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(os.path.join(OUT_DIR, "summary_protocol.csv"), index=False)
    print("=== Summary table ===")
    print(summary_df.to_string())

    # LOSO plots: pick X1 (mech-aware) for the headline figure
    loso_df = loso_per_sub_stats["X1_xTB+mech"]
    plot_loso_scatter(loso_df, os.path.join(OUT_DIR, "figs", "loso_scatter.png"))
    plot_loso_bar(loso_df, os.path.join(OUT_DIR, "figs", "loso_metrics_bar.png"))
    plot_heatmap_protocol(summary_df, os.path.join(OUT_DIR, "figs", "heatmap_protocol_R2.png"),
                          value="r2", title_suffix="R²")
    plot_heatmap_protocol(summary_df, os.path.join(OUT_DIR, "figs", "heatmap_protocol_MAE.png"),
                          value="mae", title_suffix="MAE")

    # Report
    print("\n=== Writing report ===")
    with open(os.path.join(OUT_DIR, "report.txt"), "w", encoding="utf-8") as f:
        f.write(f"Step 4 -- LOSO / LOMO CV report\n")
        f.write(f"Generated: {datetime.now(timezone.utc).isoformat()}\n\n")
        f.write(f"Total dataset: {len(df)} rows\n")
        f.write(f"Substrates: {list(df['reactant_name'].unique())}\n")
        f.write(f"Mechanism families: {list(df['catalyst_system_type_agg'].unique())}\n\n")
        f.write("=== Summary ===\n")
        f.write(summary_df.to_string(index=False))
        f.write("\n\n=== LOSO per substrate (X1) ===\n")
        f.write(loso_df[["substrate_held", "n_train", "n_test", "r2", "mae", "rmse"]].to_string(index=False))
        f.write("\n")
    print(f"Done. Outputs in {OUT_DIR}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--force", action="store_true")
    p.parse_args()
    main()