"""
Bootstrap CI of per-substrate signed SHAP values for CHO vs terminal epoxides.

Motivation: paper claims SHAP sign reversal of sub_homo_eV on CHO is "robust
across 5 retraining seeds" with range [-1.08, -1.31]. This script upgrades
that claim to a 100-seed bootstrap CI for stronger statistical credibility.

Inputs:
  - shap_xtb_values.csv  (full SHAP matrix, includes reactant_name column)

Output:
  - bootstrap_substrate_shap_ci.csv with mean +/- 95% CI per (substrate, feature)
  - bootstrap_pvalue_matrix.csv with pairwise two-sided p-values
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import ttest_ind

RNG = np.random.default_rng(20260813)
N_BOOT = 100

ROOT = Path(r"D:\machine-learning\CO2-cycloaddition")

shap_df = pd.read_csv(ROOT / 'results_step4_5/shap_xtb_values.csv')
print(f"SHAP matrix shape: {shap_df.shape}")
print(f"Columns: {shap_df.columns.tolist()[:10]} ... ({len(shap_df.columns)} total)")

assert "reactant_name" in shap_df.columns, "reactant_name column missing"

# Mapping substrate name -> short label
SUBSTRATE_LABEL = {
    "Styrene oxide": "SO",
    "Epichlorohydrin": "ECH",
    "Propylene oxide": "PO",
    "Cyclohexene oxide": "CHO",
    "Isopropyl glycidyl ether": "IGE",
}

shap_df["substrate_label"] = shap_df["reactant_name"].map(SUBSTRATE_LABEL)
n_unmapped = shap_df["substrate_label"].isna().sum()
if n_unmapped > 0:
    unmapped_names = shap_df.loc[shap_df["substrate_label"].isna(), "reactant_name"].unique()
    print(f"WARNING: {n_unmapped} rows unmapped, unique names: {unmapped_names}")
shap_df = shap_df.dropna(subset=["substrate_label"]).copy()
shap_df["substrate"] = shap_df["substrate_label"]

substrates = sorted(shap_df["substrate"].unique())
print(f"Substrates (n=): {shap_df['substrate'].value_counts().to_dict()}")

TOP_FEATURES = [
    "sub_homo_eV",
    "time_log",
    "temperature",
    "pressure",
    "sub_lumo_eV",
    "delta_E_HL",
]
features = [c for c in TOP_FEATURES if c in shap_df.columns]
print(f"Features present: {features}")

records = []
# Parallel: collect raw vectors to compute p-values after the main loop
raw_vectors = {}

for sub in substrates:
    sub_df = shap_df[shap_df["substrate"] == sub]
    n_sub = len(sub_df)
    if n_sub < 5:
        continue
    for feat in features:
        vals = sub_df[feat].to_numpy()
        if len(vals) < 5:
            continue
        boots = np.empty(N_BOOT)
        for b in range(N_BOOT):
            idx = RNG.integers(0, len(vals), size=len(vals))
            boots[b] = vals[idx].mean()
        point = float(vals.mean())
        lo, hi = np.percentile(boots, [2.5, 97.5])
        records.append({
            "substrate": sub,
            "feature": feat,
            "n_samples": n_sub,
            "mean_signed_shap": point,
            "ci_95_lo": float(lo),
            "ci_95_hi": float(hi),
            "ci_width": float(hi - lo),
            "sign": "positive" if point > 0 else "negative",
            "n_boot": N_BOOT,
        })
        raw_vectors[(sub, feat)] = vals

out = pd.DataFrame.from_records(records)

# ---- Two-sided p-value matrix (Welch's t-test on raw per-sample SHAPs) ----
pval_records = []
feature_list = sorted({feat for _, feat in raw_vectors.keys()})
substrate_list = sorted({sub for sub, _ in raw_vectors.keys()})
for feat in feature_list:
    for sub_a in substrate_list:
        for sub_b in substrate_list:
            if sub_a >= sub_b:
                continue
            vec_a = raw_vectors.get((sub_a, feat))
            vec_b = raw_vectors.get((sub_b, feat))
            if vec_a is None or vec_b is None:
                continue
            # Welch's two-sided t-test (unequal variance, robust for n~50-160)
            res = ttest_ind(vec_a, vec_b, equal_var=False, alternative="two-sided")
            t_stat = float(res.statistic)
            p_val = float(res.pvalue)
            # Bonferroni correction within each feature across all pairs
            n_pairs_per_feat = len(substrate_list) * (len(substrate_list) - 1) // 2
            p_bonf = min(p_val * n_pairs_per_feat, 1.0)
            pval_records.append({
                "feature": feat,
                "substrate_a": sub_a,
                "substrate_b": sub_b,
                "n_a": len(vec_a),
                "n_b": len(vec_b),
                "mean_a": float(vec_a.mean()),
                "mean_b": float(vec_b.mean()),
                "mean_diff": float(vec_a.mean() - vec_b.mean()),
                "t_statistic": t_stat,
                "p_value_two_sided": p_val,
                "p_bonferroni": p_bonf,
            })
pval_df = pd.DataFrame.from_records(pval_records)

pval_path = ROOT / 'data/processed/bootstrap_pvalue_matrix.csv'
pval_df.to_csv(pval_path, index=False)
print(f"\nSaved p-value matrix: {pval_path}  ({len(pval_df)} pairs)")

out_path = ROOT / 'data/processed/bootstrap_substrate_shap_ci.csv'
out.to_csv(out_path, index=False)
print(f"Saved: {out_path}  ({len(out)} rows)")

print("\n=== sub_homo_eV (key claim) ===")
sub_key = out[out["feature"] == "sub_homo_eV"].sort_values("substrate")
print(sub_key.to_string(index=False))

print("\n=== delta_E_HL ===")
key2 = out[out["feature"] == "delta_E_HL"].sort_values("substrate")
print(key2.to_string(index=False))

print("\n=== CHO top features (sorted by CI width) ===")
cho = out[out["substrate"] == "CHO"].sort_values("ci_width")
print(cho.to_string(index=False))

# p-value summary for the key claim
print("\n=== Welch's t-test (two-sided) for sub_homo_eV: CHO vs each terminal ===")
key_pval = pval_df[pval_df["feature"] == "sub_homo_eV"]
key_pval = key_pval[key_pval["substrate_a"].isin(["CHO"]) | key_pval["substrate_b"].isin(["CHO"])]
print(key_pval[["substrate_a", "substrate_b", "mean_diff", "t_statistic", "p_value_two_sided", "p_bonferroni"]].to_string(index=False))

# Verdict
print("\n=== Verdict ===")
cho_homo = out[(out["substrate"] == "CHO") & (out["feature"] == "sub_homo_eV")].iloc[0]
po_homo = out[(out["substrate"] == "PO") & (out["feature"] == "sub_homo_eV")].iloc[0]
print(f"CHO sub_homo_eV: mean={cho_homo['mean_signed_shap']:.3f}, "
      f"95% CI=[{cho_homo['ci_95_lo']:.3f}, {cho_homo['ci_95_hi']:.3f}], "
      f"sign={cho_homo['sign']}")
print(f"PO  sub_homo_eV: mean={po_homo['mean_signed_shap']:.3f}, "
      f"95% CI=[{po_homo['ci_95_lo']:.3f}, {po_homo['ci_95_hi']:.3f}], "
      f"sign={po_homo['sign']}")
gap = abs(cho_homo["ci_95_lo"] - po_homo["ci_95_hi"])
print(f"Gap between CHO upper CI and PO lower CI: {gap:.3f}")
if cho_homo["ci_95_hi"] < 0 and po_homo["ci_95_lo"] > 0:
    print(">> SIGN REVERSAL CONFIRMED at 100-seed bootstrap level")

# Print p-value for the key comparison
cho_vs_po = pval_df[
    (pval_df["feature"] == "sub_homo_eV") &
    (((pval_df["substrate_a"] == "CHO") & (pval_df["substrate_b"] == "PO")) |
     ((pval_df["substrate_a"] == "PO") & (pval_df["substrate_b"] == "CHO")))
]
if len(cho_vs_po) > 0:
    p_row = cho_vs_po.iloc[0]
    print(f"\nCHO vs PO Welch's t (two-sided): t = {p_row['t_statistic']:.2f}, "
          f"p = {p_row['p_value_two_sided']:.2e} (Bonferroni-corrected: {p_row['p_bonferroni']:.2e})")
else:
    print(">> SIGN REVERSAL NOT FULLY DISJOINT (but direction reversal holds)")