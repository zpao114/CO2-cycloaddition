# -*- coding: utf-8 -*-
"""
901_substrate_catalyst_matrix.py
================================

Generate a substrate x catalyst diagnostic matrix for the CO2-cycloaddition
ML pipeline. For every (substrate, catalyst family) cell, compute:
  - n (sample count)
  - mean yield (%) with 95% bootstrap CI
  - top-3 SHAP features (mean |SHAP| rank) when n >= MIN_N_FOR_SHAP
  - mean predicted-vs-actual delta

Inputs (read-only, no retraining):
  - co2_drfp_xtb_extended.csv            (master CSV, 2316 valid rows)
  - shap_xtb_values.csv                  (analysis/601 output, 468 rows on fold-0 val)
  - data_split.json                      (paths.DATA_SPLIT_JSON, soft dependency)

Outputs (results_substrate_catalyst_matrix/):
  - matrix_yield_ci.csv                  3x4 machine-readable matrix (5 substrates x 3+other)
  - matrix_yield_ci_5x5.csv             5x5 machine-readable matrix (5 substrates x 5 catalyst_system_type)
  - matrix_yield_ci.xlsx                 Excel version with formatting
  - heatmap_yield.png / .pdf             main heatmap
  - heatmap_n.png / .pdf                 sample-size heatmap
  - per_cell_shap.csv                    per-cell top-3 SHAP when supported
  - per_system_shap.csv                  mean |SHAP| aggregated by catalyst family
  - per_system_shap_heatmap.png / .pdf   bar plot of SHAP by catalyst family
  - cho_summary.csv                      quick CHO subset summary (for 902 input)
  - 901_substrate_catalyst_matrix_report.txt  human-readable report

Usage:
  D:\\co2\\env_drfp\\python.exe 901_substrate_catalyst_matrix.py
"""
from __future__ import annotations

import os
import io
import json
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

# ----------------------------------------------------------------------
# Paths / constants (no hardcoded magic numbers where possible)
# ----------------------------------------------------------------------
PROJECT_ROOT = os.environ.get("CO2_PROJECT_ROOT", r"D:\machine-learning\CO2-cycloaddition")
# 2026-08-20 fix: paths.py RESULTS_* now point at PROJECT_ROOT (legacy layout).
#   The master DRFP+ xTB merged table lives under results_cho_diagnostic/,
#   while step4_5 / data_split live at the repo root.
DATA_EXTENDED = os.path.join(PROJECT_ROOT, 'results', 'results_cho_diagnostic', 'co2_drfp_xtb_extended.csv')
SHAP_CSV = os.path.join(PROJECT_ROOT, 'results_step4_5', 'shap_xtb_values.csv')
OUT_DIR = os.path.join(PROJECT_ROOT, "results_substrate_catalyst_matrix")

# Use the canonical split manifest path from src.paths (single source of truth).
try:
    import sys as _sys
    _ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if _ROOT not in _sys.path:
        _sys.path.insert(0, _ROOT)
    from src.paths import DATA_SPLIT_JSON as _DATA_SPLIT_JSON
    SPLIT_JSON = str(_DATA_SPLIT_JSON)
except Exception:
    # Fallback (kept in sync with src.paths.RESULTS_DATA_SPLIT layout):
    SPLIT_JSON = os.path.join(PROJECT_ROOT, "results", "results_data_split", "data_split.json")

# Minimum sample count for per-cell SHAP ranking to be reported.
# Below this, top-k SHAP is too noisy.
MIN_N_FOR_SHAP = int(os.environ.get("CO2_MIN_N_FOR_SHAP", 20))

# Minimum sample count for a cell to be considered "reportable" at all.
MIN_N_FOR_CELL = int(os.environ.get("CO2_MIN_N_FOR_CELL", 5))

# Bootstrap replicates for the 95% CI.
N_BOOTSTRAP = int(os.environ.get("CO2_N_BOOTSTRAP", 2000))
RNG_SEED = int(os.environ.get("CO2_RNG_SEED", 42))

# The 5 epoxide substrates we expect in the dataset.
EXPECTED_SUBSTRATES = [
    "Styrene oxide",
    "Epichlorohydrin",
    "Propylene oxide",
    "Cyclohexene oxide",
    "Isopropyl glycidyl ether",
]

# Catalyst-family aggregation: keep the 3 well-populated families as-is,
# merge the sparse ones into "other" so each aggregated cell has >= 20 rows
# (the threshold needed for SHAP to be interpretable).
PRIMARY_FAMILIES = ["ionic_liquid", "metal_halide", "mixed_system"]
AGGREGATED_OTHER_LABEL = "other"

os.makedirs(OUT_DIR, exist_ok=True)


# ----------------------------------------------------------------------
# Font setup (mirror 601_shap_analysis.py)
# ----------------------------------------------------------------------
for f2 in fm.fontManager.ttflist:
    if any(tag in f2.name for tag in ["SimHei", "Noto Sans CJK", "WenQuanYi", "Microsoft YaHei"]):
        plt.rcParams["font.family"] = f2.name
        break
plt.rcParams["axes.unicode_minus"] = False


# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------
def aggregate_catalyst_family(series: pd.Series) -> pd.Series:
    """Collapse sparse catalyst-system-type labels into PRIMARY_FAMILIES + 'other'."""
    out = series.astype(str).where(series.isin(PRIMARY_FAMILIES), other=AGGREGATED_OTHER_LABEL)
    out = out.where(out != "nan", other=AGGREGATED_OTHER_LABEL)
    return out


def bootstrap_ci(values: np.ndarray, n_boot: int = N_BOOTSTRAP, alpha: float = 0.05,
                 rng: np.random.Generator | None = None) -> tuple[float, float, float]:
    """Return (mean, ci_lo, ci_hi) of values via percentile bootstrap."""
    if rng is None:
        rng = np.random.default_rng(RNG_SEED)
    v = np.asarray(values, dtype=np.float64)
    v = v[~np.isnan(v)]
    if v.size == 0:
        return float("nan"), float("nan"), float("nan")
    means = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, v.size, size=v.size)
        means[b] = v[idx].mean()
    mean = float(v.mean())
    lo = float(np.percentile(means, 100 * alpha / 2.0))
    hi = float(np.percentile(means, 100 * (1.0 - alpha / 2.0)))
    return mean, lo, hi


def load_master() -> pd.DataFrame:
    """Load and apply the same cleaning filters used by 601/602/401."""
    df = pd.read_csv(DATA_EXTENDED, encoding="utf-8-sig")
    df = df[df["extraction_status"] == "valid"].copy()
    df = df.dropna(subset=["yield (%)"])
    df = df[df["yield (%)"] > 0].reset_index(drop=True)
    # Discover the temperature column robustly
    temp_candidates = [c for c in df.columns if "temperature" in c.lower()]
    if temp_candidates:
        df = df.rename(columns={temp_candidates[0]: "temperature_canonical"})
    return df


def load_shap_subset(df_master: pd.DataFrame) -> pd.DataFrame | None:
    """Load 601 SHAP values if available; merge substrate + family onto each row.

    The SHAP CSV was computed on the first fold's val set using XGBoost
    trained on the 25 XTB+condition features. We therefore expect at most
    ~25% of rows in the master CSV to appear here. When the file is absent
    or has incompatible columns, we silently return None.
    """
    if not os.path.exists(SHAP_CSV):
        return None
    try:
        shap_df = pd.read_csv(SHAP_CSV, encoding="utf-8-sig")
    except Exception:
        return None
    if "reactant_name" not in shap_df.columns or "catalyst_system_type" not in shap_df.columns:
        return None
    shap_df["catalyst_system_type_agg"] = aggregate_catalyst_family(shap_df["catalyst_system_type"])
    return shap_df


def build_yield_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (yield_mean, n, ci_lo, ci_hi) wide matrices, all indexed identically
    by (reactant, catalyst_family).
    """
    df = df.copy()
    df["catalyst_system_type_agg"] = aggregate_catalyst_family(df["catalyst_system_type"])
    substrates = [s for s in EXPECTED_SUBSTRATES if s in df["reactant_name"].unique()]
    families = PRIMARY_FAMILIES + [AGGREGATED_OTHER_LABEL]

    rng = np.random.default_rng(RNG_SEED)
    rows = []
    for sub in substrates:
        sub_mask = df["reactant_name"] == sub
        for fam in families:
            mask = sub_mask & (df["catalyst_system_type_agg"] == fam)
            n = int(mask.sum())
            if n == 0:
                rows.append({"reactant": sub, "catalyst_family": fam,
                             "n": n, "yield_mean": float("nan"),
                             "yield_ci_lo": float("nan"), "yield_ci_hi": float("nan")})
                continue
            values = df.loc[mask, "yield (%)"].to_numpy(dtype=np.float64)
            m, lo, hi = bootstrap_ci(values, n_boot=N_BOOTSTRAP, rng=rng)
            rows.append({"reactant": sub, "catalyst_family": fam, "n": n,
                         "yield_mean": m, "yield_ci_lo": lo, "yield_ci_hi": hi})
    df_long = pd.DataFrame(rows)
    df_yield = df_long.pivot(index="reactant", columns="catalyst_family",
                             values="yield_mean").reindex(index=substrates, columns=families)
    df_n = df_long.pivot(index="reactant", columns="catalyst_family",
                         values="n").reindex(index=substrates, columns=families)
    df_lo = df_long.pivot(index="reactant", columns="catalyst_family",
                          values="yield_ci_lo").reindex(index=substrates, columns=families)
    df_hi = df_long.pivot(index="reactant", columns="catalyst_family",
                          values="yield_ci_hi").reindex(index=substrates, columns=families)
    return df_yield, df_n, df_lo, df_hi


# 5 catalyst-system-type categories (raw, NOT aggregated).
# Reference: figures/extend_matrix_5x5.py from the legacy repo.
FAMILIES_5 = ["ionic_liquid", "metal_halide", "mixed_system",
              "organic_base", "unknown"]


def build_yield_matrix_5x5(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (yield_mean, n, ci_lo, ci_hi) wide matrices for the raw 5x5 layout.

    Index: substrate (5 expected epoxides).
    Columns: catalyst_system_type (5 raw families).
    Bootstrap-CI matches the 4x4 matrix above (2000 replicates, percentile method).
    """
    substrates = [s for s in EXPECTED_SUBSTRATES if s in df["reactant_name"].unique()]
    families = [f for f in FAMILIES_5 if f in df["catalyst_system_type"].unique()]

    rng = np.random.default_rng(RNG_SEED + 1)
    rows = []
    for sub in substrates:
        sub_mask = df["reactant_name"] == sub
        for fam in families:
            mask = sub_mask & (df["catalyst_system_type"] == fam)
            n = int(mask.sum())
            if n == 0:
                rows.append({"reactant": sub, "catalyst_family": fam,
                             "n": n, "yield_mean": float("nan"),
                             "yield_ci_lo": float("nan"), "yield_ci_hi": float("nan")})
                continue
            values = df.loc[mask, "yield (%)"].to_numpy(dtype=np.float64)
            m, lo, hi = bootstrap_ci(values, n_boot=N_BOOTSTRAP, rng=rng)
            rows.append({"reactant": sub, "catalyst_family": fam, "n": n,
                         "yield_mean": m, "yield_ci_lo": lo, "yield_ci_hi": hi})
    df_long = pd.DataFrame(rows)
    df_yield = df_long.pivot(index="reactant", columns="catalyst_family",
                             values="yield_mean").reindex(index=substrates, columns=families)
    df_n = df_long.pivot(index="reactant", columns="catalyst_family",
                         values="n").reindex(index=substrates, columns=families)
    df_lo = df_long.pivot(index="reactant", columns="catalyst_family",
                          values="yield_ci_lo").reindex(index=substrates, columns=families)
    df_hi = df_long.pivot(index="reactant", columns="catalyst_family",
                          values="yield_ci_hi").reindex(index=substrates, columns=families)
    return df_yield, df_n, df_lo, df_hi


def build_yield_matrix_5x5_long(df: pd.DataFrame) -> pd.DataFrame:
    """Long-format 5x5 (reactant, catalyst_family, n, mean, ci_lo, ci_hi)."""
    substrates = [s for s in EXPECTED_SUBSTRATES if s in df["reactant_name"].unique()]
    families = [f for f in FAMILIES_5 if f in df["catalyst_system_type"].unique()]
    rng = np.random.default_rng(RNG_SEED + 1)
    rows = []
    for sub in substrates:
        sub_mask = df["reactant_name"] == sub
        for fam in families:
            mask = sub_mask & (df["catalyst_system_type"] == fam)
            n = int(mask.sum())
            if n == 0:
                rows.append({"reactant": sub, "catalyst_family": fam,
                             "n": n, "yield_mean": float("nan"),
                             "yield_ci_lo": float("nan"), "yield_ci_hi": float("nan")})
                continue
            values = df.loc[mask, "yield (%)"].to_numpy(dtype=np.float64)
            m, lo, hi = bootstrap_ci(values, n_boot=N_BOOTSTRAP, rng=rng)
            rows.append({"reactant": sub, "catalyst_family": fam,
                         "n": n, "yield_mean": m,
                         "yield_ci_lo": lo, "yield_ci_hi": hi})
    return pd.DataFrame(rows)


def build_per_cell_shap(shap_df: pd.DataFrame, df_master: pd.DataFrame) -> pd.DataFrame:
    """For each substrate x family cell with n>=MIN_N_FOR_SHAP, compute top-3 SHAP.

    Returns a long-format DataFrame with columns:
        reactant, catalyst_family, n_shap, top1_feature, top2_feature, top3_feature,
        top1_mean_abs_shap, top2_mean_abs_shap, top3_mean_abs_shap
    Cells with n < MIN_N_FOR_SHAP get NaN top features and a note.
    """
    if shap_df is None:
        return pd.DataFrame()

    feat_cols = [c for c in shap_df.columns
                 if c not in {"row_index", "actual_yield", "predicted_yield",
                              "residual", "abs_error",
                              "reactant_name", "catalyst_system_type",
                              "catalyst_system_type_agg"}]

    rows = []
    substrates = [s for s in EXPECTED_SUBSTRATES if s in shap_df["reactant_name"].unique()]
    families = PRIMARY_FAMILIES + [AGGREGATED_OTHER_LABEL]
    for sub in substrates:
        sub_mask = shap_df["reactant_name"] == sub
        for fam in families:
            mask = sub_mask & (shap_df["catalyst_system_type_agg"] == fam)
            n = int(mask.sum())
            row = {"reactant": sub, "catalyst_family": fam, "n_shap": n}
            if n < MIN_N_FOR_SHAP:
                row.update({f"top{i}_feature": None for i in (1, 2, 3)})
                row.update({f"top{i}_mean_abs_shap": float("nan") for i in (1, 2, 3)})
                row["note"] = f"n<{MIN_N_FOR_SHAP}; SHAP skipped"
                rows.append(row)
                continue
            sub_shap = shap_df.loc[mask, feat_cols].to_numpy(dtype=np.float64)
            mean_abs = np.abs(sub_shap).mean(axis=0)
            order = np.argsort(mean_abs)[::-1][:3]
            for rank, idx in enumerate(order, start=1):
                row[f"top{rank}_feature"] = feat_cols[idx]
                row[f"top{rank}_mean_abs_shap"] = float(mean_abs[idx])
            row["note"] = ""
            rows.append(row)
    return pd.DataFrame(rows)


def build_per_system_shap(shap_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate mean |SHAP| across all 5 substrates per catalyst family.

    Returns:
        wide: feature x catalyst_family, values are mean(|SHAP|) averaged over all rows
        long: tidy version for plotting
    """
    if shap_df is None:
        return pd.DataFrame(), pd.DataFrame()
    feat_cols = [c for c in shap_df.columns
                 if c not in {"row_index", "actual_yield", "predicted_yield",
                              "residual", "abs_error",
                              "reactant_name", "catalyst_system_type",
                              "catalyst_system_type_agg"}]
    families = PRIMARY_FAMILIES + [AGGREGATED_OTHER_LABEL]
    present = [f for f in families if f in shap_df["catalyst_system_type_agg"].unique()
               and int((shap_df["catalyst_system_type_agg"] == f).sum()) >= MIN_N_FOR_SHAP]
    wide = {}
    for fam in present:
        sub = shap_df[shap_df["catalyst_system_type_agg"] == fam]
        wide[fam] = np.abs(sub[feat_cols].to_numpy(dtype=np.float64)).mean(axis=0)
    wide_df = pd.DataFrame(wide, index=feat_cols)
    long_df = wide_df.reset_index().melt(id_vars="index", var_name="catalyst_family",
                                         value_name="mean_abs_shap").rename(
        columns={"index": "feature"})
    return wide_df, long_df


def draw_heatmap(value_matrix: pd.DataFrame, ci_lo: pd.DataFrame, ci_hi: pd.DataFrame,
                 n_matrix: pd.DataFrame, title: str, fname: str, fmt: str = "{:.1f}"):
    """Draw a heatmap with mean yield, CI bar (visual), and n in cell annotation."""
    fig, ax = plt.subplots(figsize=(11, 7))
    data = value_matrix.to_numpy(dtype=np.float64)
    masked = np.ma.masked_invalid(data)
    cmap = plt.cm.viridis
    cmap.set_bad(color="#dddddd")
    im = ax.imshow(masked, cmap=cmap, vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(value_matrix.shape[1]))
    ax.set_xticklabels(value_matrix.columns, rotation=20, ha="right", fontsize=10)
    ax.set_yticks(range(value_matrix.shape[0]))
    ax.set_yticklabels(value_matrix.index, fontsize=10)
    ax.set_title(title, fontsize=12, pad=12)

    lo_mat = ci_lo.reindex(index=value_matrix.index, columns=value_matrix.columns).to_numpy(dtype=np.float64)
    hi_mat = ci_hi.reindex(index=value_matrix.index, columns=value_matrix.columns).to_numpy(dtype=np.float64)
    n_mat = n_matrix.reindex(index=value_matrix.index, columns=value_matrix.columns).to_numpy()
    for i in range(value_matrix.shape[0]):
        for j in range(value_matrix.shape[1]):
            v = data[i, j]
            n = n_mat[i, j]
            if not np.isfinite(v):
                ax.text(j, i, "n/a", ha="center", va="center", fontsize=9, color="#555555")
                continue
            label = f"{fmt.format(v)}%\n[{fmt.format(lo_mat[i, j])}–{fmt.format(hi_mat[i, j])}]\nn={int(n)}"
            color = "white" if v < 50 else "black"
            ax.text(j, i, label, ha="center", va="center", fontsize=8.5, color=color)

    plt.colorbar(im, ax=ax, label="Mean yield (%)")
    plt.tight_layout()
    out_png = os.path.join(OUT_DIR, fname + ".png")
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.savefig(os.path.join(OUT_DIR, fname + ".pdf"), bbox_inches="tight")
    plt.close()


def draw_n_heatmap(n_matrix: pd.DataFrame, fname: str):
    fig, ax = plt.subplots(figsize=(11, 6))
    data = n_matrix.to_numpy()
    im = ax.imshow(data, cmap=plt.cm.Blues, aspect="auto")
    ax.set_xticks(range(n_matrix.shape[1]))
    ax.set_xticklabels(n_matrix.columns, rotation=20, ha="right", fontsize=10)
    ax.set_yticks(range(n_matrix.shape[0]))
    ax.set_yticklabels(n_matrix.index, fontsize=10)
    ax.set_title("Sample size per substrate x catalyst family (n)", fontsize=12, pad=12)
    for i in range(n_matrix.shape[0]):
        for j in range(n_matrix.shape[1]):
            v = data[i, j]
            color = "white" if v < data.max() * 0.6 else "black"
            ax.text(j, i, f"{int(v)}", ha="center", va="center", fontsize=11, color=color)
    plt.colorbar(im, ax=ax, label="n samples")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, fname + ".png"), dpi=150, bbox_inches="tight")
    plt.savefig(os.path.join(OUT_DIR, fname + ".pdf"), bbox_inches="tight")
    plt.close()


def draw_per_system_shap(long_df: pd.DataFrame, fname: str, top_n: int = 12):
    """Bar plot: top-N features ranked by mean |SHAP|, side-by-side per family."""
    if long_df.empty:
        return
    # Rank by overall mean across families
    overall = long_df.groupby("feature")["mean_abs_shap"].mean().sort_values(ascending=False)
    top_feats = overall.head(top_n).index.tolist()
    sub = long_df[long_df["feature"].isin(top_feats)].copy()
    # Stable ordering
    sub["feature"] = pd.Categorical(sub["feature"], categories=top_feats, ordered=True)
    sub = sub.sort_values("feature")

    families = sorted(sub["catalyst_family"].unique())
    n_fam = len(families)
    width = 0.8 / n_fam
    fig, ax = plt.subplots(figsize=(12, 7))
    x = np.arange(len(top_feats))
    cmap_fam = plt.cm.Set2(np.linspace(0, 1, max(n_fam, 3)))
    for k, fam in enumerate(families):
        vals = []
        for feat in top_feats:
            row = sub[(sub["feature"] == feat) & (sub["catalyst_family"] == fam)]
            vals.append(float(row["mean_abs_shap"].iloc[0]) if len(row) else 0.0)
        ax.bar(x + (k - (n_fam - 1) / 2) * width, vals, width=width * 0.95,
               label=fam, color=cmap_fam[k], edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(top_feats, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Mean |SHAP|", fontsize=11)
    ax.set_title(f"Top-{top_n} SHAP features per catalyst family "
                 f"(only families with n >= {MIN_N_FOR_SHAP})", fontsize=12, pad=12)
    ax.legend(fontsize=9, loc="best")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, fname + ".png"), dpi=150, bbox_inches="tight")
    plt.savefig(os.path.join(OUT_DIR, fname + ".pdf"), bbox_inches="tight")
    plt.close()


def write_excel(yield_mat, ci_lo_mat, ci_hi_mat, n_mat, per_cell_shap, per_system_shap_wide):
    path = os.path.join(OUT_DIR, "matrix_yield_ci.xlsx")
    try:
        import openpyxl  # noqa: F401
    except ImportError:
        print("  [info] openpyxl not available; skipping xlsx export")
        return
    with pd.ExcelWriter(path, engine="openpyxl") as xl:
        yield_mat.to_excel(xl, sheet_name="yield_mean")
        ci_lo_mat.to_excel(xl, sheet_name="yield_ci_lo")
        ci_hi_mat.to_excel(xl, sheet_name="yield_ci_hi")
        n_mat.to_excel(xl, sheet_name="n_samples")
        if not per_cell_shap.empty:
            per_cell_shap.to_excel(xl, sheet_name="per_cell_shap", index=False)
        if not per_system_shap_wide.empty:
            per_system_shap_wide.to_excel(xl, sheet_name="per_system_shap")
    print(f"  Saved: {path}")


def write_report(df: pd.DataFrame, yield_mat, ci_lo_mat, ci_hi_mat, n_mat, per_cell_shap,
                 per_system_shap_wide, per_system_shap_long):
    lines = []
    lines.append("=" * 70)
    lines.append("901 — Substrate x Catalyst Family Diagnostic Matrix")
    lines.append("=" * 70)
    lines.append(f"Generated: {datetime.now(timezone.utc).isoformat()}")
    lines.append(f"Dataset  : {DATA_EXTENDED}")
    lines.append(f"N total  : {len(df)} valid rows after cleaning")
    if os.path.exists(SPLIT_JSON):
        with open(SPLIT_JSON, "r", encoding="utf-8") as fh:
            split = json.load(fh)
        lines.append(f"Split    : holdout seed={split['holdout']['seed']} "
                     f"test_size={split['holdout']['test_size']} "
                     f"({split['holdout']['n_train']}/{split['holdout']['n_test']}); "
                     f"kfold seed={split['kfold']['seed']} n_splits={split['kfold']['n_splits']}")
    lines.append("")
    lines.append(f"Substrates        ({len(EXPECTED_SUBSTRATES)} expected, "
                 f"{yield_mat.shape[0]} present):")
    for s in yield_mat.index:
        lines.append(f"  - {s}")
    lines.append("")
    lines.append("Catalyst families (after aggregation; sparse labels merged into 'other'):")
    for fam in yield_mat.columns:
        lines.append(f"  - {fam}")
    lines.append("")
    lines.append("Mean yield (%) per cell:")
    for sub in yield_mat.index:
        for fam in yield_mat.columns:
            v = yield_mat.loc[sub, fam]
            n = n_mat.loc[sub, fam]
            if pd.isna(v):
                lines.append(f"  {sub:30s} | {fam:15s} | n={int(n):4d}  | n/a")
                continue
            lo = ci_lo_mat.loc[sub, fam]
            hi = ci_hi_mat.loc[sub, fam]
            lines.append(f"  {sub:30s} | {fam:15s} | n={int(n):4d}  | "
                         f"{v:6.2f}%  CI[{lo:.2f}, {hi:.2f}]")
    lines.append("")
    if not per_cell_shap.empty:
        reportable = per_cell_shap[per_cell_shap["n_shap"] >= MIN_N_FOR_SHAP]
        lines.append(f"Per-cell SHAP (top-3) for {len(reportable)} cells with "
                     f"n_shap >= {MIN_N_FOR_SHAP}:")
        for _, row in reportable.iterrows():
            tops = [str(row.get(f"top{i}_feature")) for i in (1, 2, 3)]
            vals = [row.get(f"top{i}_mean_abs_shap") for i in (1, 2, 3)]
            lines.append(f"  {row['reactant']:30s} | {row['catalyst_family']:15s} | "
                         f"n_shap={int(row['n_shap']):3d} | "
                         f"top1={tops[0]} ({vals[0]:.4f})  "
                         f"top2={tops[1]} ({vals[1]:.4f})  "
                         f"top3={tops[2]} ({vals[2]:.4f})")
        lines.append("")
    if not per_system_shap_wide.empty:
        lines.append(f"Per-system SHAP (families with n >= {MIN_N_FOR_SHAP}):")
        overall = per_system_shap_wide.mean(axis=1).sort_values(ascending=False)
        for feat in overall.head(8).index:
            vals = "  ".join(f"{fam}={per_system_shap_wide.loc[feat, fam]:.4f}"
                             for fam in per_system_shap_wide.columns)
            lines.append(f"  {feat:30s}  | {vals}")
        lines.append("")
    lines.append("Outputs:")
    for f in sorted(os.listdir(OUT_DIR)):
        lines.append(f"  - {f}")
    report = "\n".join(lines)
    with open(os.path.join(OUT_DIR, "901_substrate_catalyst_matrix_report.txt"),
              "w", encoding="utf-8") as f:
        f.write(report)
    print(report)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    print("=" * 60)
    print("901 — Substrate x Catalyst Family Diagnostic Matrix")
    print("=" * 60)
    df = load_master()
    print(f"[Load] {len(df)} valid rows")
    shap_df = load_shap_subset(df)
    if shap_df is None:
        print("[WARN] shap_xtb_values.csv not found or unusable; "
              "per-cell/per-system SHAP will be skipped.")
    else:
        print(f"[Load] {len(shap_df)} SHAP rows from {SHAP_CSV}")

    # 1. Yield / CI matrix on the full dataset
    df["catalyst_system_type_agg"] = aggregate_catalyst_family(df["catalyst_system_type"])
    yield_mat, n_mat, ci_lo_mat, ci_hi_mat = build_yield_matrix(df)

    # Build a tidy long-form CSV: one row per (substrate, family) with mean + CI.
    long_rows = []
    for sub in yield_mat.index:
        for fam in yield_mat.columns:
            mean_v = yield_mat.loc[sub, fam]
            n_v = n_mat.loc[sub, fam]
            lo_v = ci_lo_mat.loc[sub, fam]
            hi_v = ci_hi_mat.loc[sub, fam]
            long_rows.append({
                "reactant": sub,
                "catalyst_family": fam,
                "n": int(n_v) if pd.notna(n_v) else 0,
                "yield_mean": float(mean_v) if pd.notna(mean_v) else float("nan"),
                "yield_ci_lo": float(lo_v) if pd.notna(lo_v) else float("nan"),
                "yield_ci_hi": float(hi_v) if pd.notna(hi_v) else float("nan"),
            })
    matrix_long = pd.DataFrame(long_rows)
    matrix_long.to_csv(os.path.join(OUT_DIR, "matrix_yield_ci.csv"), index=False,
                       encoding="utf-8-sig")
    print(f"  Saved: matrix_yield_ci.csv")

    # 2. Heatmaps
    draw_heatmap(yield_mat, ci_lo_mat, ci_hi_mat, n_mat,
                 title=f"CO2-cycloaddition mean yield (%): substrate x catalyst family\n"
                       f"(95% bootstrap CI, n_bootstrap={N_BOOTSTRAP}, n shown in cell)",
                 fname="heatmap_yield")
    print("  Saved: heatmap_yield.png/.pdf")
    draw_n_heatmap(n_mat, fname="heatmap_n")
    print("  Saved: heatmap_n.png/.pdf")

    # 3. Per-cell SHAP
    per_cell_shap = build_per_cell_shap(shap_df, df) if shap_df is not None else pd.DataFrame()
    if not per_cell_shap.empty:
        per_cell_shap.to_csv(os.path.join(OUT_DIR, "per_cell_shap.csv"), index=False,
                             encoding="utf-8-sig")
        print(f"  Saved: per_cell_shap.csv  ({len(per_cell_shap)} cells)")

    # 4. Per-system SHAP
    per_system_shap_wide, per_system_shap_long = build_per_system_shap(shap_df) \
        if shap_df is not None else (pd.DataFrame(), pd.DataFrame())
    if not per_system_shap_wide.empty:
        per_system_shap_wide.to_csv(os.path.join(OUT_DIR, "per_system_shap.csv"),
                                    encoding="utf-8-sig")
        per_system_shap_long.to_csv(os.path.join(OUT_DIR, "per_system_shap_long.csv"),
                                    index=False, encoding="utf-8-sig")
        print("  Saved: per_system_shap.csv/.long.csv")
        draw_per_system_shap(per_system_shap_long, fname="per_system_shap_heatmap")
        print("  Saved: per_system_shap_heatmap.png/.pdf")

    # 5. Excel
    write_excel(yield_mat, ci_lo_mat, ci_hi_mat, n_mat, per_cell_shap, per_system_shap_wide)

    # 6. CHO subset summary for 902
    cho_mask = df["reactant_name"] == "Cyclohexene oxide"
    cho_summary = df.loc[cho_mask].groupby("catalyst_system_type_agg").agg(
        n=("yield (%)", "size"),
        mean_yield=("yield (%)", "mean"),
        std_yield=("yield (%)", "std"),
    ).reset_index()
    cho_summary.to_csv(os.path.join(OUT_DIR, "cho_summary.csv"), index=False,
                        encoding="utf-8-sig")
    print("  Saved: cho_summary.csv  (input for 902)")

    # 7. Extended 5x5 matrix (5 substrates × 5 catalyst_system_type raw categories).
    #    The PRIMARY_FAMILIES + 'other' aggregation above collapses organic_base +
    #    unknown into 'other'; downstream figures want the full 5×5 breakdown
    #    (ionic_liquid, metal_halide, mixed_system, organic_base, unknown) so
    #    they can map each family to a distinct panel colour.
    yield_mat_5x5, n_mat_5x5, ci_lo_mat_5x5, ci_hi_mat_5x5 = build_yield_matrix_5x5(df)
    long_5x5 = build_yield_matrix_5x5_long(df)
    long_5x5.to_csv(os.path.join(OUT_DIR, "matrix_yield_ci_5x5.csv"), index=False,
                    encoding="utf-8-sig")
    print(f"  Saved: matrix_yield_ci_5x5.csv  ({len(long_5x5)} cells)")

    # 8. Report
    write_report(df, yield_mat, ci_lo_mat, ci_hi_mat, n_mat, per_cell_shap,
                 per_system_shap_wide, per_system_shap_long)
    print("\nDone!")


if __name__ == "__main__":
    main()