# -*- coding: utf-8 -*-
"""
906_generate_fisher_flip.py
============================

Generate Fisher's exact test on 2x2 contingency tables built from
per-substrate signed SHAP direction-flip counts.

The script mirrors the historical `fisher_direction_flip.csv` produced
by the v2 ablation pipeline (2,316-row dataset) but re-runs the entire
computation on the v3 2,490-row dataset using the existing
`results_step4_5/per_substrate_shap.csv` file.

Pipeline:
  1. Load per-substrate mean signed SHAP (5 substrates x 32 features).
  2. For each substrate pair (sa, sb), count features where
     |signed_shap_sa - signed_shap_sb| > FLIP_THRESHOLD (= 1.0, same
     threshold as 701_per_substrate_shap.direction_flip).
  3. Build 2x2 contingency table:
       [CHO_pair_reversed, CHO_pair_same]
       [Term_pair_reversed, Term_pair_same]
     plus all pairwise comparisons involving CHO vs each terminal.
  4. For each 2x2 table, compute:
       - Odds ratio (OR)
       - 95% CI for log-OR (Woolf method)
       - Fisher's exact p (two-sided + one-sided)
       - Statistical power via non-central chi-square approximation
         (statsmodels GofChisquarePower).
  5. Write results to
     results_statistical_test/fisher_direction_flip.csv.

Outputs:
  - results/results_statistical_test/fisher_direction_flip.csv

Runtime: < 30 seconds.
"""

import os
import sys
import io
import warnings

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, norm, chi2, ncx2

# Power is computed by non-central chi-square (1 df) under the observed
# effect size (Cramer's V) and a fixed alpha. We avoid a statsmodels
# dependency by integrating the non-central chi-square ourselves.
ALPHA_POWER = 0.05


def noncentral_chi2_sf(x, df, nc, n=4096):
    """Compute P(X > x) for X ~ chi2(df, nc) by simple trapezoidal
    integration of the non-central chi-square density."""
    if df != 1:
        # For df != 1, fall back to survival function from scipy if
        # available; otherwise return NaN.
        try:
            from scipy.stats import ncx2 as _ncx2
            return float(_ncx2.sf(x, df, nc))
        except Exception:
            return float("nan")
    # Series expansion of the non-central chi-square density:
    #   f(x; k=1, nc) = sum_{m=0}^inf exp(-nc/2)(nc/2)^m / m!
    #                    * x^{k/2+m-1} exp(-x/2) / (2^{k/2+m} Gamma(k/2+m))
    # Then SF = sum_{m=0}^inf Poisson(nc/2)[m] * GammaInc(k/2+m, x/2)
    half_nc = nc / 2.0
    # Pre-compute Poisson PMF up to a high m.
    log_pmf = -half_nc
    s = 0.0
    m = 0
    while m < 5000:
        if m == 0:
            pm = np.exp(log_pmf)
        else:
            log_pmf += np.log(half_nc / m)
            pm = np.exp(log_pmf)
        # GammaRegularized upper incomplete for shape (1/2 + m),
        # scale 2 evaluated at x.
        from scipy.special import gammaincc
        s_inc = gammaincc(0.5 + m, x / 2.0)
        s += pm * s_inc
        if pm < 1e-15 and m > half_nc + 10:
            break
        m += 1
    return float(s)

PROJECT_ROOT = os.environ.get(
    "CO2_PROJECT_ROOT", r"D:\machine-learning\CO2-cycloaddition"
)
PER_SUB_CSV = os.path.join(
    PROJECT_ROOT, "results_step4_5", "per_substrate_shap.csv"
)
OUT_DIR = os.path.join(PROJECT_ROOT, "results", "results_statistical_test")
os.makedirs(OUT_DIR, exist_ok=True)
OUT_CSV = os.path.join(OUT_DIR, "fisher_direction_flip.csv")

FLIP_THRESHOLD = 1.0
TERMINAL_SUBSTRATES = [
    "Propylene oxide",        # PO
    "Epichlorohydrin",        # ECH
    "Styrene oxide",          # SO
    "Isopropyl glycidyl ether",  # IGE
]
CHO_NAME = "Cyclohexene oxide"


def log_or_ci(a, b, c, d, alpha=0.05):
    """Compute OR and Woolf 95% CI for log-OR."""
    if 0 in (a, b, c, d):
        # Apply Haldane-Anscombe 0.5 continuity correction.
        a_, b_, c_, d_ = a + 0.5, b + 0.5, c + 0.5, d + 0.5
    else:
        a_, b_, c_, d_ = float(a), float(b), float(c), float(d)
    log_or = np.log(a_ * d_ / (b_ * c_))
    se_log_or = np.sqrt(1.0 / a_ + 1.0 / b_ + 1.0 / c_ + 1.0 / d_)
    z = norm.ppf(1.0 - alpha / 2)
    ci_lo = np.exp(log_or - z * se_log_or)
    ci_hi = np.exp(log_or + z * se_log_or)
    or_ = np.exp(log_or)
    return float(or_), float(ci_lo), float(ci_hi)


def fisher_power(a, b, c, d, alpha=ALPHA_POWER):
    """Compute the post-hoc power of the chi-square (1 df) approximation
    of Fisher's exact test under H1, given the observed 2x2 table.

    Uses the non-central chi-square distribution:
      power = P(chi2_nc(df=1, nc) >= chi2_crit(1 - alpha, 1))
    where chi2_nc non-centrality is the observed chi2 statistic and
    chi2_crit is the critical value at significance level alpha.
    """
    n = a + b + c + d
    if n == 0:
        return float("nan")
    row1, row2 = a + b, c + d
    col1, col2 = a + c, b + d
    # Expected counts under H0.
    exp_a = row1 * col1 / n
    exp_b = row1 * col2 / n
    exp_c = row2 * col1 / n
    exp_d = row2 * col2 / n
    chi2_obs = (
        (a - exp_a) ** 2 / exp_a
        + (b - exp_b) ** 2 / exp_b
        + (c - exp_c) ** 2 / exp_c
        + (d - exp_d) ** 2 / exp_d
    )
    # Critical value for the chi-square(1) test at alpha.
    crit = chi2.ppf(1.0 - alpha, df=1)
    # Power = SF of non-central chi-square(1, nc=chi2_obs).
    try:
        power = noncentral_chi2_sf(crit, 1, chi2_obs)
    except Exception:
        power = float("nan")
    return float(power)


def build_pairwise_flip_table(per_sub_df, substrates, threshold):
    """For every (sa, sb) pair, compute the number of features where
    |signed_shap_sa - signed_shap_sb| > threshold."""
    pivot = per_sub_df.pivot(
        index="feature", columns="substrate_held", values="mean_signed_shap"
    ).fillna(0)
    rows = []
    subs = substrates
    for i, sa in enumerate(subs):
        for sb in subs[i + 1:]:
            if sa not in pivot.columns or sb not in pivot.columns:
                continue
            diff = pivot[sa] - pivot[sb]
            abs_diff = np.abs(diff)
            n_flip = int((abs_diff > threshold).sum())
            n_total = int(pivot.shape[0])
            rows.append({
                "sub_a": sa,
                "sub_b": sb,
                "n_features_total": n_total,
                "n_features_strong_flip": n_flip,
                "frac_strong_flip": float(n_flip / n_total),
            })
    return pd.DataFrame(rows), pivot


def cho_vs_terminal_row(pair_df, cho, terminal):
    """Aggregate a CHO vs single terminal row."""
    mask = ((pair_df.sub_a == cho) & (pair_df.sub_b == terminal)) | (
        (pair_df.sub_a == terminal) & (pair_df.sub_b == cho)
    )
    sel = pair_df[mask]
    if len(sel) == 0:
        return None
    grp_reversed = int(sel["n_features_strong_flip"].sum())
    grp_total = int(sel["n_features_total"].sum())
    grp_same = grp_total - grp_reversed
    return grp_reversed, grp_same, grp_total


def main():
    print("=" * 72)
    print("  906 -- Fisher's exact test on SHAP direction flip")
    print("=" * 72)

    per_sub_df = pd.read_csv(PER_SUB_CSV, encoding="utf-8-sig")
    print(f"  loaded per_substrate_shap.csv: shape={per_sub_df.shape}")
    substrates = sorted(per_sub_df["substrate_held"].unique().tolist())
    print(f"  substrates: {substrates}")

    pair_df, pivot = build_pairwise_flip_table(
        per_sub_df, substrates, FLIP_THRESHOLD
    )
    print(f"\n  Pairwise flip table (threshold |dSHAP| > {FLIP_THRESHOLD}):")
    print(pair_df.to_string(index=False))

    # Aggregate CHO vs each terminal, plus CHO vs Terminal (combined).
    rows_out = []
    cho_rev = cho_same = cho_tot = 0
    for term in TERMINAL_SUBSTRATES:
        if term not in substrates:
            continue
        out = cho_vs_terminal_row(pair_df, CHO_NAME, term)
        if out is None:
            continue
        rev, same, tot = out
        cho_rev += rev
        cho_same += same
        cho_tot += tot
        # The complementary 2x2: CHO pair vs Terminal pair.
        # We treat each comparison symmetrically so the second group
        # here is the single terminal alone (1 pair = tot features).
        # That's not meaningful; instead we report the comparison
        # "CHO_vs_<term>" with no second group, and the omnibus row
        # below aggregates all terminal pairs together.
        rows_out.append({
            "model": "DualANN",
            "comparison": f"CHO_vs_{term.split()[0]}",
            "group_a_reversed": rev,
            "group_a_same": same,
            "group_b_reversed": np.nan,
            "group_b_same": np.nan,
            "fisher_or": np.nan,
            "fisher_or_ci_lower": np.nan,
            "fisher_or_ci_upper": np.nan,
            "fisher_two_sided_p": np.nan,
            "fisher_one_sided_p": np.nan,
            "n_total": tot,
            "note": (
                f"per-substrate SHAP direction flip between CHO and "
                f"{term}, threshold |dSHAP|>{FLIP_THRESHOLD}, "
                f"{tot} feature pairs"
            ),
        })

    # CHO vs Terminal (combined): the headline 2x2 contingency table.
    # Group B = all terminal-terminal pairs.
    term_term = pair_df[
        (~pair_df.sub_a.isin([CHO_NAME])) & (~pair_df.sub_b.isin([CHO_NAME]))
    ]
    tt_rev = int(term_term["n_features_strong_flip"].sum())
    tt_tot = int(term_term["n_features_total"].sum())
    tt_same = tt_tot - tt_rev

    table = np.array(
        [[cho_rev, cho_same], [tt_rev, tt_same]], dtype=int
    )
    print(f"\n  CHO vs Terminal-Terminal 2x2 table:")
    print(f"             reversed  same")
    print(f"  CHO pair:    {cho_rev:3d}      {cho_same:3d}")
    print(f"  Term pair:   {tt_rev:3d}      {tt_same:3d}")

    or_, p_two = fisher_exact(table, alternative="two-sided")
    _, p_one = fisher_exact(table, alternative="greater")
    or_pt, ci_lo, ci_hi = log_or_ci(cho_rev, cho_same, tt_rev, tt_same)
    pow_ = fisher_power(cho_rev, cho_same, tt_rev, tt_same)
    print(f"\n  Fisher OR={or_pt:.4f}, 95% CI=[{ci_lo:.4f}, {ci_hi:.4f}]")
    print(f"  Fisher two-sided p={p_two:.4f}, one-sided p={p_one:.4f}")
    print(f"  Approx power (chi-square, alpha=0.05)={pow_:.3f}")

    rows_out.append({
        "model": "DualANN",
        "comparison": "CHO_vs_Terminal",
        "group_a_reversed": cho_rev,
        "group_a_same": cho_same,
        "group_b_reversed": tt_rev,
        "group_b_same": tt_same,
        "fisher_or": round(or_pt, 4),
        "fisher_or_ci_lower": round(ci_lo, 4),
        "fisher_or_ci_upper": round(ci_hi, 4),
        "fisher_two_sided_p": round(p_two, 4),
        "fisher_one_sided_p": round(p_one, 4),
        "n_total": cho_tot + tt_tot,
        "note": (
            f"{len(TERMINAL_SUBSTRATES)} pairs per substrate group; "
            f"Fisher exact test on 2x2 contingency table "
            f"[reversed,same] x [CHO,Terminal], "
            f"threshold |dSHAP|>{FLIP_THRESHOLD}, "
            f"approx power={pow_:.3f}"
        ),
    })

    out_df = pd.DataFrame(rows_out)
    out_df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\n  saved {OUT_CSV}")
    print(f"\nFinal Fisher direction-flip CSV:")
    print(out_df.to_string(index=False))


if __name__ == "__main__":
    main()