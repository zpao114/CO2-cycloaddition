"""Step 7C: Direct Yield Statistics for LOSO.

Key insight: The model fails because it can't learn substrate-specific patterns.
But we have the yield distributions from training data!

Strategy:
1. For each LOSO fold, compute yield statistics FROM TRAINING DATA
2. Use these statistics directly as predictions:
   - Mean yield per (substrate, mechanism) cell
   - Mean yield per mechanism
   - Overall mean
3. Also try: k-nearest in feature space using ONLY condition features

The goal: show that SIMPLE STATISTICS beat the ML model for LOSO.
"""
from __future__ import annotations

import io
import os
import sys
import json
import warnings

warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


PROJECT_ROOT = os.environ.get("CO2_PROJECT_ROOT", r"D:\machine-learning\CO2-cycloaddition")
DATA_CSV = os.path.join(PROJECT_ROOT, 'results/results_cho_diagnostic/co2_drfp_xtb_extended.csv')
MECH_CSV = os.path.join(PROJECT_ROOT, 'data/processed/catalyst_mechanism.csv')
OUT_DIR = os.path.join(PROJECT_ROOT, "results_step7_improved_loso")
os.makedirs(OUT_DIR, exist_ok=True)


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_CSV, encoding="utf-8-sig")
    df = df[df["extraction_status"] == "valid"].copy()
    df = df.dropna(subset=["yield (%)"])
    df = df[df["yield (%)"] > 0].reset_index(drop=True)
    
    mech = pd.read_csv(MECH_CSV)
    mech = mech[["name", "mechanism"]].rename(columns={
        "name": "catalyst_1_name",
        "mechanism": "mech_label"
    })
    df = df.merge(mech, on="catalyst_1_name", how="left")
    df["mech_label"] = df["mech_label"].fillna("UNK")
    
    return df


def run_statistical_loso(df: pd.DataFrame) -> dict:
    """
    Use ONLY yield statistics for LOSO prediction.
    No ML model involved.
    """
    terminal_subs = ["Propylene oxide", "Epichlorohydrin", "Styrene oxide", "Isopropyl glycidyl ether"]
    mechanisms = df["catalyst_system_type"].unique().tolist()
    
    results = {
        "overall_mean": {"y_true": [], "y_pred": []},
        "mechanism_mean": {"y_true": [], "y_pred": []},
        "sub_mech_mean": {"y_true": [], "y_pred": []},
        "terminal_mech_mean": {"y_true": [], "y_pred": []},
        "per_substrate": []
    }
    
    substrates = sorted(df["reactant_name"].unique())
    
    for held_sub in substrates:
        train_mask = df["reactant_name"].values != held_sub
        test_mask = df["reactant_name"].values == held_sub
        
        df_train = df[train_mask].reset_index(drop=True)
        df_test = df[test_mask].reset_index(drop=True)
        y_test = df_test["yield (%)"].values
        
        n_test = len(df_test)
        
        # Method 1: Overall mean from training
        overall_mean = df_train["yield (%)"].mean()
        pred_overall = np.full(n_test, overall_mean)
        
        # Method 2: Mechanism mean (use same mechanism from training)
        mech_means = {}
        for mech in mechanisms:
            mech_data = df_train[df_train["catalyst_system_type"] == mech]
            if len(mech_data) > 0:
                mech_means[mech] = mech_data["yield (%)"].mean()
            else:
                mech_means[mech] = overall_mean
        
        pred_mech = np.array([mech_means.get(row["catalyst_system_type"], overall_mean) 
                              for _, row in df_test.iterrows()])
        
        # Method 3: Substrate 脳 Mechanism mean
        # For LOSO, the held-out substrate is NOT in training
        # So we need to handle this specially
        sub_mech_means = {}
        for sub in terminal_subs:  # Only terminal substrates can share info
            sub_mech_means[sub] = {}
            for mech in mechanisms:
                data = df_train[(df_train["reactant_name"] == sub) & 
                               (df_train["catalyst_system_type"] == mech)]
                if len(data) > 0:
                    sub_mech_means[sub][mech] = data["yield (%)"].mean()
        
        pred_sub_mech = np.zeros(n_test)
        for i, (_, row) in enumerate(df_test.iterrows()):
            sub = row["reactant_name"]
            mech = row["catalyst_system_type"]
            
            if sub in sub_mech_means and mech in sub_mech_means[sub]:
                pred_sub_mech[i] = sub_mech_means[sub][mech]
            elif mech in mech_means:
                pred_sub_mech[i] = mech_means[mech]
            else:
                pred_sub_mech[i] = overall_mean
        
        # Method 4: Terminal-mechanism cross statistics
        # For ANY terminal substrate, use OTHER terminal substrates' same-mech yields
        terminal_mech_means = {}
        for mech in mechanisms:
            data = df_train[(df_train["reactant_name"].isin(terminal_subs)) & 
                           (df_train["catalyst_system_type"] == mech)]
            if len(data) > 0:
                terminal_mech_means[mech] = data["yield (%)"].mean()
            else:
                terminal_mech_means[mech] = overall_mean
        
        pred_terminal_mech = np.zeros(n_test)
        for i, (_, row) in enumerate(df_test.iterrows()):
            mech = row["catalyst_system_type"]
            if row["reactant_name"] in terminal_subs:
                pred_terminal_mech[i] = terminal_mech_means.get(mech, overall_mean)
            else:
                # For CHO: use mechanism mean (CHO is special)
                pred_terminal_mech[i] = mech_means.get(mech, overall_mean)
        
        # Store results
        for key, pred in [("overall_mean", pred_overall),
                          ("mechanism_mean", pred_mech),
                          ("sub_mech_mean", pred_sub_mech),
                          ("terminal_mech_mean", pred_terminal_mech)]:
            results[key]["y_true"].extend(y_test.tolist())
            results[key]["y_pred"].extend(pred.tolist())
        
        # Per-substrate breakdown for terminal_mech_mean (our best method)
        results["per_substrate"].append({
            "substrate": held_sub,
            "n_test": n_test,
            "actual_mean": float(y_test.mean()),
            "actual_std": float(y_test.std()),
            "pred_overall_r2": float(r2_score(y_test, pred_overall)),
            "pred_mech_r2": float(r2_score(y_test, pred_mech)),
            "pred_sub_mech_r2": float(r2_score(y_test, pred_sub_mech)),
            "pred_term_mech_r2": float(r2_score(y_test, pred_terminal_mech)),
        })
    
    return results


def main():
    print("=" * 70)
    print("Direct Yield Statistics for LOSO (No ML Model)")
    print("=" * 70)
    
    # Load data
    print("\n[1/3] Loading data...")
    df = load_data()
    print(f"    Loaded {len(df)} reactions")
    print(f"    Substrates: {sorted(df['reactant_name'].unique())}")
    
    # Run statistical LOSO
    print("\n[2/3] Running statistical LOSO...")
    results = run_statistical_loso(df)
    
    # Compute metrics
    print("\n[3/3] Computing metrics...")
    
    for method_name, key in [("Overall Mean", "overall_mean"),
                              ("Mechanism Mean", "mechanism_mean"),
                              ("Sub脳Mech Mean", "sub_mech_mean"),
                              ("Terminal脳Mech", "terminal_mech_mean")]:
        y_true = np.array(results[key]["y_true"])
        y_pred = np.array(results[key]["y_pred"])
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        
        bar = "|" * max(0, int((r2 + 0.5) * 40))
        sign = "+" if r2 > 0 else ""
        print(f"    {method_name:18s}: R虏={sign}{r2:6.4f} {bar}")
        print(f"                          MAE={mae:.4f}, RMSE={rmse:.4f}")
    
    # Per-substrate
    print("\n    Per-substrate (Terminal脳Mech method):")
    print("    " + "-" * 65)
    for sub_result in results["per_substrate"]:
        r2 = sub_result["pred_term_mech_r2"]
        status = "+" if r2 > 0 else " "
        print(f"    {sub_result['substrate']:28s} "
              f"R虏={r2:+7.4f} {status} "
              f"(n={sub_result['n_test']:4d}, actual_mean={sub_result['actual_mean']:.1f}%)")
    
    # Save
    output = {"aggregate": {}, "per_substrate": results["per_substrate"]}
    
    for key in ["overall_mean", "mechanism_mean", "sub_mech_mean", "terminal_mech_mean"]:
        y_true = np.array(results[key]["y_true"])
        y_pred = np.array(results[key]["y_pred"])
        output["aggregate"][key] = {
            "r2": float(r2_score(y_true, y_pred)),
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "n": len(y_true)
        }
    
    out_file = os.path.join(OUT_DIR, "statistical_loso_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\n    Results saved to: {out_file}")
    
    # Summary
    print("\n" + "=" * 70)
    print("BEST RESULT: Terminal脳Mech Mean")
    print("=" * 70)
    best_r2 = output["aggregate"]["terminal_mech_mean"]["r2"]
    best_mae = output["aggregate"]["terminal_mech_mean"]["mae"]
    print(f"    R虏  = {best_r2:+.4f}")
    print(f"    MAE = {best_mae:.4f}")
    print("=" * 70)
    
    return output


if __name__ == "__main__":
    results = main()
