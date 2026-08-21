# -*- coding: utf-8 -*-
from __future__ import annotations
"""
403b_ranking_metrics.py
========================

Ranking-quality metrics for virtual-screening output (705_virtual_screening.py or
legacy 402). Adds three numbers the existing screening report does not include:

  1. NDCG@10 / MAP@10     — standard ranking metrics, treating each
                              candidate's nearest-neighbor training yield
                              (``nn1_train_yield``) as the relevance label.
  2. Spearman / Kendall correlation between predicted yield and
     nearest-neighbor training yield over ALL candidates (not just top-K).
  3. Top-K concordance     — fraction of the model's top-K predictions that
                              are also in the top-K by training NN yield.

Inputs (from 705_virtual_screening.py or legacy 402):
  results_virtual_screening/top10_*.csv   (fallback: top10_results.csv)

Outputs (results_ranking_metrics/):
  - ranking_metrics_per_tier.csv
  - ranking_metrics_summary.csv
  - ranking_correlation.csv
  - 403b_ranking_metrics_report.txt

Usage:
  D:\\co2\\env_drfp\\python.exe 403b_ranking_metrics.py
"""
import os

import io
import os
import sys
import warnings
from datetime import datetime, timezone

warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = os.environ.get("CO2_PROJECT_ROOT", r"D:\machine-learning\CO2-cycloaddition")
VS_DIR = os.path.join(PROJECT_ROOT, "results_virtual_screening")
OUT_DIR = os.path.join(PROJECT_ROOT, "results_ranking_metrics")

TIER_FILES = {
    "in_domain":   os.path.join(VS_DIR, "top10_indomain.csv"),
    "near_domain": os.path.join(VS_DIR, "top10_neardomain.csv"),
    "exploratory": os.path.join(VS_DIR, "top10_exploratory.csv"),
    "diverse":     os.path.join(VS_DIR, "top10_diverse.csv"),
}
# FIX (2026-08-19): the current pipeline runs 310_known_top10_baseline.py which
# produces a flat top-K ranking (no per-domain split).  When the per-domain
# files above are missing, fall back to the 310
# artefacts so 403b can still produce ranking metrics for the paper.
TIER_FILES_310_FALLBACK = {
    "in_domain":   os.path.join(VS_DIR, "top10_results.csv"),
    "near_domain": os.path.join(VS_DIR, "top20_results.csv"),
    "exploratory": os.path.join(VS_DIR, "top50_results.csv"),
}
PRED_COL = "pred_yield_boot_mean"
RELEVANCE_COL = "nn1_train_yield"
CANDIDATES_FULL = os.path.join(VS_DIR, "candidates_full.csv")

os.makedirs(OUT_DIR, exist_ok=True)


# ----------------------------------------------------------------------
# Metric helpers
# ----------------------------------------------------------------------
def dcg_at_k(relevances: np.ndarray, k: int) -> float:
    rel = np.asarray(relevances, dtype=np.float64)[:k]
    if rel.size == 0:
        return 0.0
    discounts = np.log2(np.arange(2, rel.size + 2))
    return float((rel / discounts).sum())


def ndcg_at_k(relevances: np.ndarray, k: int) -> float:
    rel = np.asarray(relevances, dtype=np.float64)
    ideal = np.sort(rel)[::-1][:k]
    ideal_dcg = dcg_at_k(ideal, k)
    if ideal_dcg == 0:
        return 0.0
    return dcg_at_k(rel, k) / ideal_dcg


def average_precision_at_k(relevances: np.ndarray, threshold: float, k: int) -> float:
    """Mean Average Precision @ k where "relevant" = relevance >= threshold."""
    rel = np.asarray(relevances, dtype=np.float64)[:k]
    if rel.size == 0:
        return 0.0
    binary = (rel >= threshold).astype(np.float64)
    if binary.sum() == 0:
        return 0.0
    cumsum = np.cumsum(binary)
    precision_at_k = cumsum / np.arange(1, rel.size + 1)
    return float((precision_at_k * binary).sum() / binary.sum())


def mean_reciprocal_rank(relevances: np.ndarray, threshold: float) -> float:
    rel = np.asarray(relevances, dtype=np.float64)
    binary = (rel >= threshold).astype(np.float64)
    if binary.sum() == 0:
        return 0.0
    first = int(np.argmax(binary > 0)) + 1
    return float(1.0 / first)


def concordance_at_k(pred_top_k_idx: np.ndarray, true_top_k_idx: np.ndarray, k: int) -> float:
    return float(len(set(pred_top_k_idx[:k]) & set(true_top_k_idx[:k]))) / float(k)


def tanimoto_threshold_for_pred(relevance_signal: np.ndarray, threshold: float) -> np.ndarray:
    """Convert a relevance signal into a boolean 'is this in-domain?' mask."""
    return (np.asarray(relevance_signal, dtype=np.float64) >= threshold).astype(int)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    print("403b — Ranking Metrics for Screening Output")
    print("=" * 60)

    # 1. Load full candidate pool
    if not os.path.exists(CANDIDATES_FULL):
        sys.stderr.write(f"[error] {CANDIDATES_FULL} not found; "
                         f"run 705_virtual_screening.py first.\n")
        sys.exit(1)
    full = pd.read_csv(CANDIDATES_FULL, encoding="utf-8-sig")
    print(f"[Load] {len(full)} candidates from candidates_full.csv")

    if PRED_COL not in full.columns:
        sys.stderr.write(f"[error] column '{PRED_COL}' missing in candidates_full.csv\n")
        sys.exit(1)

    # nn1_train_yield is degenerate (constant 1.0 across all 800 candidates) —
    # not useful as a relevance signal. Use Tanimoto similarity to the training
    # set as a 2nd axis instead. The "relevance" for ranking quality is then:
    #   "does the predicted-yield ranking put high-Tanimoto (in-domain)
    #    candidates at the top?"
    if RELEVANCE_COL in full.columns and full[RELEVANCE_COL].nunique() > 1:
        relevance_signal = full[RELEVANCE_COL].to_numpy(dtype=np.float64)
        relevance_label = RELEVANCE_COL
    else:
        print(f"[INFO] {RELEVANCE_COL} is degenerate "
              f"(unique values = {full[RELEVANCE_COL].nunique() if RELEVANCE_COL in full.columns else 'N/A'}). "
              f"Falling back to 'tanimoto_max' as the relevance signal.")
        relevance_signal = full["tanimoto_max"].to_numpy(dtype=np.float64)
        relevance_label = "tanimoto_max"

    relevance_threshold = float(np.median(relevance_signal[~np.isnan(relevance_signal)]))
    print(f"[Relevance]  signal = {relevance_label}")
    print(f"[Threshold]  '{relevance_label} >= {relevance_threshold:.3f}' defines a "
          f"'in-domain-like' candidate for MAP/MRR purposes")

    # 2. Full-pool correlation
    full_pred = full[PRED_COL].to_numpy(dtype=np.float64)
    valid = ~(np.isnan(full_pred) | np.isnan(relevance_signal))
    full_pred_v = full_pred[valid]
    rel_v = relevance_signal[valid]
    if full_pred_v.std() == 0 or rel_v.std() == 0:
        spearman_r = spearman_p = float("nan")
        kendall_tau = kendall_p = float("nan")
        pearson_r = pearson_p = float("nan")
    else:
        spearman_r, spearman_p = stats.spearmanr(full_pred_v, rel_v)
        kendall_tau, kendall_p = stats.kendalltau(full_pred_v, rel_v)
        pearson_r, pearson_p = stats.pearsonr(full_pred_v, rel_v)

    # Top-K by predicted yield vs top-K by relevance
    pred_top_10_idx = np.argsort(full_pred_v)[::-1][:10]
    rel_top_10_idx = np.argsort(rel_v)[::-1][:10]

    corr_rows = [
        {"metric": "spearman_rho", "value": float(spearman_r) if np.isfinite(spearman_r) else None,
         "p_value": float(spearman_p) if np.isfinite(spearman_p) else None,
         "n": int(valid.sum())},
        {"metric": "kendall_tau", "value": float(kendall_tau) if np.isfinite(kendall_tau) else None,
         "p_value": float(kendall_p) if np.isfinite(kendall_p) else None,
         "n": int(valid.sum())},
        {"metric": "pearson_r", "value": float(pearson_r) if np.isfinite(pearson_r) else None,
         "p_value": float(pearson_p) if np.isfinite(pearson_p) else None,
         "n": int(valid.sum())},
        {"metric": "concordance_at_5_pred_vs_relevance_top5",
         "value": concordance_at_k(pred_top_10_idx, rel_top_10_idx, 5),
         "p_value": None, "n": None},
        {"metric": "concordance_at_10_pred_vs_relevance_top10",
         "value": concordance_at_k(pred_top_10_idx, rel_top_10_idx, 10),
         "p_value": None, "n": None},
    ]
    pd.DataFrame(corr_rows).to_csv(os.path.join(OUT_DIR, "ranking_correlation.csv"),
                                    index=False, encoding="utf-8-sig")
    print(f"  Saved: ranking_correlation.csv  "
          f"(Spearman ρ = {spearman_r:+.3f}, p = {spearman_p:.2e})")

    # 3. Per-tier ranking metrics
    per_tier_rows = []
    for tier_name, tier_path in TIER_FILES.items():
        # FIX (2026-08-19): prefer the 402 per-domain split if it exists, else
        # fall back to 310's flat top-K ranking.
        used_path = tier_path
        if not os.path.exists(used_path):
            fb = TIER_FILES_310_FALLBACK.get(tier_name)
            if fb and os.path.exists(fb):
                print(f"  [fallback] {tier_name}: {tier_path} missing -> {fb}")
                used_path = fb
            else:
                print(f"  [skip] {tier_name}: {tier_path} not found")
                continue
        tier = pd.read_csv(used_path, encoding="utf-8-sig")
        if len(tier) == 0:
            continue
        rel = tier[relevance_label].to_numpy(dtype=np.float64) \
            if relevance_label in tier.columns \
            else np.zeros(len(tier), dtype=np.float64)
        pred = tier[PRED_COL].to_numpy(dtype=np.float64) \
            if PRED_COL in tier.columns else rel.copy()

        per_tier_rows.append({
            "tier": tier_name,
            "n_top": int(len(tier)),
            "ndcg_at_10": ndcg_at_k(rel, 10),
            "ndcg_at_5": ndcg_at_k(rel, 5),
            "map_at_10": average_precision_at_k(rel, relevance_threshold, 10),
            "map_at_5": average_precision_at_k(rel, relevance_threshold, 5),
            "mrr": mean_reciprocal_rank(rel, relevance_threshold),
            "mean_signal_in_top": float(rel.mean()) if len(rel) else float("nan"),
            "median_signal_in_top": float(np.median(rel)) if len(rel) else float("nan"),
        })
    per_tier_df = pd.DataFrame(per_tier_rows)
    per_tier_df.to_csv(os.path.join(OUT_DIR, "ranking_metrics_per_tier.csv"), index=False,
                       encoding="utf-8-sig")
    print(f"  Saved: ranking_metrics_per_tier.csv")

    # 4. Tier summary
    summary_rows = []
    if not per_tier_df.empty:
        summary_rows.append({
            "metric": "ndcg_at_10_mean_across_tiers",
            "value": float(per_tier_df["ndcg_at_10"].mean()),
        })
        summary_rows.append({
            "metric": "map_at_10_mean_across_tiers",
            "value": float(per_tier_df["map_at_10"].mean()),
        })
    summary_rows.append({
        "metric": f"spearman_rho_pred_vs_{relevance_label}_all_candidates",
        "value": float(spearman_r) if np.isfinite(spearman_r) else None,
    })
    summary_rows.append({
        "metric": "concordance_pred_top10_vs_relevance_top10",
        "value": concordance_at_k(pred_top_10_idx, rel_top_10_idx, 10),
    })
    pd.DataFrame(summary_rows).to_csv(os.path.join(OUT_DIR, "ranking_metrics_summary.csv"),
                                       index=False, encoding="utf-8-sig")
    print(f"  Saved: ranking_metrics_summary.csv")

    # 5. Report
    lines = []
    lines.append("=" * 70)
    lines.append("403b — Ranking Metrics (screening output evaluation)")
    lines.append("=" * 70)
    lines.append(f"Generated: {datetime.now(timezone.utc).isoformat()}")
    lines.append(f"Pool: {CANDIDATES_FULL}  ({len(full)} candidates)")
    lines.append(f"Prediction column  : {PRED_COL}")
    lines.append(f"Relevance signal   : {relevance_label}")
    lines.append(f"Relevance threshold: {relevance_label} >= {relevance_threshold:.3f}")
    lines.append("")
    if RELEVANCE_COL in full.columns and full[RELEVANCE_COL].nunique() <= 1:
        lines.append(f"NOTE: {RELEVANCE_COL} has only {full[RELEVANCE_COL].nunique()} unique value(s) "
                     f"across all candidates — it is degenerate and cannot serve as a relevance "
                     f"signal. Tanimoto similarity is used instead.")
        lines.append("")
    lines.append("Full-pool ranking correlation (all candidates):")
    for r in corr_rows:
        v = r["value"]
        v_str = f"{v:+.4f}" if isinstance(v, (int, float)) and np.isfinite(v) else "n/a"
        if r["p_value"] is not None and np.isfinite(r["p_value"]):
            lines.append(f"  {r['metric']:50s}  value={v_str}  "
                         f"p={r['p_value']:.2e}  n={r['n']}")
        else:
            lines.append(f"  {r['metric']:50s}  value={v_str}")
    lines.append("")
    lines.append("Per-tier ranking metrics:")
    if not per_tier_df.empty:
        lines.append(f"  {'tier':12s} {'n':4s} {'NDCG@10':9s} {'NDCG@5':9s} {'MAP@10':9s} "
                     f"{'MAP@5':9s} {'MRR':7s} {'mean signal':14s}")
        for _, r in per_tier_df.iterrows():
            lines.append(f"  {str(r['tier']):12s} {int(r['n_top']):4d} "
                         f"{r['ndcg_at_10']:.4f}    {r['ndcg_at_5']:.4f}    "
                         f"{r['map_at_10']:.4f}    {r['map_at_5']:.4f}    "
                         f"{r['mrr']:.4f}  {r['mean_signal_in_top']:.4f}")
    lines.append("")
    lines.append("Interpretation:")
    lines.append(f"  - NDCG@10 close to 1.0  → the predicted ranking puts high-{relevance_label}")
    lines.append("    candidates near the top.")
    lines.append(f"  - MAP@10  → average precision of recommending '{relevance_label} >= "
                 f"{relevance_threshold:.3f}' within the model's top-10.")
    lines.append(f"  - Spearman ρ over all candidates → overall monotonic agreement")
    lines.append(f"    between predicted yield and {relevance_label}.")
    lines.append("")
    lines.append("Outputs:")
    for f in sorted(os.listdir(OUT_DIR)):
        lines.append(f"  - {f}")
    report = "\n".join(lines)
    with open(os.path.join(OUT_DIR, "403b_ranking_metrics_report.txt"), "w",
              encoding="utf-8") as f:
        f.write(report)
    print("\n" + report)
    print("\nDone!")


if __name__ == "__main__":
    main()