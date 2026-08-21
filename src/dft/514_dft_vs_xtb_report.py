#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
514_dft_vs_xtb_report.py
========================
Compare GFN2-xTB descriptors against DFT (B3LYP D3BJ def2-TZVP) results
for the CO2-cycloaddition validation set.

Inputs
------
  --xtb-summary   xtb_results_summary.csv    (default name)
  --dft-summary   dft_results_summary.csv    (produced by 510_parse_dft_outputs.py)
  --output        514_dft_vs_xtb_report.csv  (matched pairs with delta, RMSE)

Output columns:
  name, smiles, role,
  xtb_homo_eV, dft_homo_eV, homo_delta_eV,
  xtb_lumo_eV, dft_lumo_eV, lumo_delta_eV,
  xtb_gap_eV,  dft_gap_eV,  gap_delta_eV,
  xtb_dipole_D, dft_dipole_D, dipole_delta_D,
  mae_homo_eV, mae_lumo_eV, mae_gap_eV, mae_dipole_D,
  rmse_homo_eV, rmse_lumo_eV, rmse_gap_eV, rmse_dipole_D

The matching is done by basename (`name`) of the ORCA .out file vs the
`name` column in the xTB summary. Use `--match-substring` to do a softer
match (e.g. ORCA "01_BisphenolA_diglycidyl_ether_substrate.out" -> xTB
"BisphenolA_diglycidyl_ether").

For the manuscript we report, per descriptor:
  - mean signed error (MSE)  :  mean(DFT - xTB)
  - mean absolute error (MAE):  mean(|DFT - xTB|)
  - root-mean-square error  :  sqrt(mean((DFT - xTB)^2))
  - max absolute error       :  max(|DFT - xTB|)
"""
import argparse
import csv
import math
import os
import sys
from typing import Dict, List, Optional, Tuple


def _safe_float(v) -> Optional[float]:
    try:
        if v is None:
            return None
        f = float(v)
        if math.isnan(f):
            return None
        return f
    except (ValueError, TypeError):
        return None


def _load_csv(path: str) -> List[Dict]:
    if not os.path.isfile(path):
        print(f"[error] {path} not found.", file=sys.stderr)
        return []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _match_rows(xtb_rows: List[Dict], dft_rows: List[Dict],
                match_substring: bool) -> List[Tuple[Dict, Dict]]:
    """Return (xtb_row, dft_row) pairs matched by `name`.

    If `match_substring` is True, we use any xTB name that is a substring of
    the DFT name (with leading number prefix and trailing role tag stripped).
    """
    pairs: List[Tuple[Dict, Dict]] = []
    for dft in dft_rows:
        dft_name = (dft.get("name") or "").strip()
        dft_base = (dft.get("file") or dft_name).replace(".out", "")
        for xtb in xtb_rows:
            xtb_name = (xtb.get("name") or "").strip()
            if match_substring:
                # Match if the xTB name appears anywhere in the ORCA
                # basename, e.g. "BisphenolA_diglycidyl_ether" inside
                # "01_BisphenolA_diglycidyl_ether_substrate".
                if xtb_name and (xtb_name in dft_base or xtb_name in dft_name):
                    pairs.append((xtb, dft))
                    break
            else:
                if xtb_name == dft_name or xtb_name == dft_base:
                    pairs.append((xtb, dft))
                    break
    return pairs


def _stats(values: List[float]) -> Dict[str, float]:
    """Return MSE / MAE / RMSE / MAX over a list of signed errors."""
    n = len(values)
    if n == 0:
        return {"n": 0, "mse": float("nan"), "mae": float("nan"),
                "rmse": float("nan"), "max": float("nan")}
    mse = sum(values) / n
    mae = sum(abs(v) for v in values) / n
    rmse = math.sqrt(sum(v * v for v in values) / n)
    mx = max(abs(v) for v in values)
    return {"n": n, "mse": mse, "mae": mae, "rmse": rmse, "max": mx}


def _corr(xtb_vals: List[float], dft_vals: List[float]) -> Tuple[float, float]:
    """Pearson R and Spearman rank-correlation between xTB and DFT columns.

    These are the headline metrics for xTB vs DFT validation: MAE alone tells
    you the systematic offset, but R/rho tell you whether the method captures
    the *trend* across compounds. Both should be reported in the manuscript.
    """
    if len(xtb_vals) < 2:
        return float("nan"), float("nan")
    n = len(xtb_vals)
    mx = sum(xtb_vals) / n
    md = sum(dft_vals) / n
    num = sum((x - mx) * (y - md) for x, y in zip(xtb_vals, dft_vals))
    denx = math.sqrt(sum((x - mx) ** 2 for x in xtb_vals))
    deny = math.sqrt(sum((y - md) ** 2 for y in dft_vals))
    if denx == 0 or deny == 0:
        return float("nan"), float("nan")
    pearson = num / (denx * deny)
    # Spearman: rank correlation with ties broken at the mean rank.
    def _rank(xs: List[float]) -> List[float]:
        idx = sorted(range(n), key=lambda i: xs[i])
        ranks = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and xs[idx[j + 1]] == xs[idx[i]]:
                j += 1
            r = (i + j + 2) / 2.0  # mean rank, 1-indexed
            for k in range(i, j + 1):
                ranks[idx[k]] = r
            i = j + 1
        return ranks
    rx = _rank(xtb_vals)
    ry = _rank(dft_vals)
    # Direct Pearson on ranks (no recursion into _corr).
    mr = sum(rx) / n
    ms = sum(ry) / n
    num_r = sum((x - mr) * (y - ms) for x, y in zip(rx, ry))
    denr_x = math.sqrt(sum((x - mr) ** 2 for x in rx))
    denr_y = math.sqrt(sum((y - ms) ** 2 for y in ry))
    if denr_x == 0 or denr_y == 0:
        spearman = float("nan")
    else:
        spearman = num_r / (denr_x * denr_y)
    return pearson, spearman


def _shifted_mae(deltas: List[float]) -> float:
    """MAE after subtracting the mean signed error (best constant offset).

    xTB and DFT use different orbital-energy reference frames; the most
    common correction in the literature is a single per-descriptor shift
    equal to the mean delta. After that shift, MAE should drop substantially
    if the deviation is systematic (which it is for HOMO/LUMO).
    """
    if not deltas:
        return float("nan")
    shift = sum(deltas) / len(deltas)
    return sum(abs(v - shift) for v in deltas) / len(deltas)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true",
                        help="Accept the global --force flag from run_pipeline.ps1 "
                             "(this script always overwrites its outputs, so the "
                             "flag is a no-op here).")
    parser.add_argument("--xtb-summary", default='dft_validation/results/xtb_on_dft_geometry_nosolv.csv')
    parser.add_argument("--dft-summary", default='dft_validation/results/dft_results_summary.csv')
    parser.add_argument("--output",      default='dft_validation/results/514_dft_vs_xtb_report.csv')
    parser.add_argument("--report",      default="dft_validation/results/514_dft_vs_xtb_report.txt")
    parser.add_argument("--match-substring", action="store_true",
                        help="Allow substring matching between xTB and DFT names")
    args = parser.parse_args()

    xtb_rows = _load_csv(args.xtb_summary)
    dft_rows = _load_csv(args.dft_summary)

    if not xtb_rows:
        print(f"[error] No xTB rows loaded from {args.xtb_summary}.", file=sys.stderr)
        sys.exit(1)
    if not dft_rows:
        print(f"[warn]  No DFT rows loaded from {args.dft_summary}; "
              f"this is expected before ORCA runs are complete.")

    pairs = _match_rows(xtb_rows, dft_rows, args.match_substring)
    print("=" * 60)
    print(f"DFT-vs-xTB validation | matched pairs: {len(pairs)}")
    print("=" * 60)

    if not pairs:
        print("[error] No matched pairs; check --xtb-summary / --dft-summary "
              "and the name columns.", file=sys.stderr)
        sys.exit(2)

    out_rows: List[Dict] = []
    deltas: Dict[str, List[float]] = {"homo": [], "lumo": [], "gap": [], "dipole": []}

    for xtb, dft in pairs:
        name = xtb.get("name", "?")
        smiles = xtb.get("smiles", "")
        role = xtb.get("role", "")

        xh = _safe_float(xtb.get("homo_eV"))
        xl = _safe_float(xtb.get("lumo_eV"))
        xg = _safe_float(xtb.get("gap_eV"))
        xm = _safe_float(xtb.get("dipole_D"))

        dh = _safe_float(dft.get("homo_eV"))
        dl = _safe_float(dft.get("lumo_eV"))
        dg = _safe_float(dft.get("gap_eV"))
        dm = _safe_float(dft.get("dipole_debye"))

        row = {
            "name": name, "smiles": smiles, "role": role,
            "xtb_homo_eV":  xh, "dft_homo_eV":  dh,
            "homo_delta_eV":  (dh - xh) if (xh is not None and dh is not None) else None,
            "xtb_lumo_eV":  xl, "dft_lumo_eV":  dl,
            "lumo_delta_eV":  (dl - xl) if (xl is not None and dl is not None) else None,
            "xtb_gap_eV":   xg, "dft_gap_eV":   dg,
            "gap_delta_eV":   (dg - xg) if (xg is not None and dg is not None) else None,
            "xtb_dipole_D": xm, "dft_dipole_D": dm,
            "dipole_delta_D": (dm - xm) if (xm is not None and dm is not None) else None,
        }
        out_rows.append(row)
        for key, dv in zip(("homo", "lumo", "gap", "dipole"),
                           (row["homo_delta_eV"], row["lumo_delta_eV"],
                            row["gap_delta_eV"], row["dipole_delta_D"])):
            if dv is not None:
                deltas[key].append(dv)

    # Write per-pair CSV
    with open(args.output, "w", encoding="utf-8-sig", newline="") as f:
        if out_rows:
            w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
            w.writeheader()
            w.writerows(out_rows)
    print(f"Per-pair CSV : {args.output}")

    # Write summary report
    lines = ["DFT-vs-xTB validation report",
             "=" * 60,
             f"Pairs matched : {len(pairs)}",
             f"xTB summary   : {args.xtb_summary}",
             f"DFT summary   : {args.dft_summary}",
             ""]
    # Map key -> (xTB column name, DFT column name)
    col_map = {
        "homo":   ("homo_eV",     "homo_eV"),
        "lumo":   ("lumo_eV",     "lumo_eV"),
        "gap":    ("gap_eV",      "gap_eV"),
        "dipole": ("dipole_D",    "dipole_debye"),
    }
    for key, label in [("homo", "HOMO (eV)"), ("lumo", "LUMO (eV)"),
                       ("gap",  "Gap (eV)"),   ("dipole", "Dipole (Debye)")]:
        s = _stats(deltas[key])
        xname, dname = col_map[key]
        xcol, dcol = [], []
        for xtb, dft in pairs:
            x = _safe_float(xtb.get(xname))
            d = _safe_float(dft.get(dname))
            if x is not None and d is not None:
                xcol.append(x); dcol.append(d)
        n_corr = min(len(xcol), len(dcol))
        if n_corr >= 2:
            pearson, spearman = _corr(xcol[:n_corr], dcol[:n_corr])
        else:
            pearson = spearman = float("nan")
        shifted = _shifted_mae(deltas[key])
        lines.append(f"{label:<14s}  N={s['n']:<3d}  "
                     f"MSE={s['mse']:+.3f}  MAE={s['mae']:.3f}  "
                     f"RMSE={s['rmse']:.3f}  MAX={s['max']:.3f}")
        lines.append(f"{'':<14s}  R={pearson:+.3f}  "
                     f"Spearman={spearman:+.3f}  "
                     f"MAE_after_shift={shifted:.3f}")
    lines.append("")
    lines.append("Interpretation guide (typical for GFN2-xTB vs B3LYP-D3BJ):")
    lines.append("  - HOMO/LUMO MAE typically 0.3-0.6 eV (xTB underestimates gap)")
    lines.append("  - Gap MAE  typically 0.5-1.0 eV")
    lines.append("  - Dipole MAE typically < 0.5 D")
    lines.append("  - R >= 0.90 and Spearman >= 0.85 indicate good trend agreement")
    lines.append("  - MAE_after_shift removes the systematic offset and shows")
    lines.append("    the residual error after a per-descriptor constant shift.")
    with open(args.report, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Report       : {args.report}\n")
    print("\n".join(lines[3:]))


if __name__ == "__main__":
    main()
