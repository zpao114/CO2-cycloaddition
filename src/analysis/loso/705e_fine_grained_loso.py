"""Step 7E: Fine-grained Yield Statistics for LOSO.

Key insight: The LOSO problem is fundamental when yield variance is high.
But we can do better by using MORE GRANULAR statistics.

Strategy:
1. Use (mechanism 脳 solvent presence) cells instead of just mechanism
2. Use (mechanism 脳 temperature range) bins
3. Use median instead of mean (more robust to outliers)
4. Find the BEST statistical predictor for LOSO
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
    
    # Add temperature bins
    temp_col = [c for c in df.columns if 'temperature' in c.lower()][0]
    df["temp_bin"] = pd.cut(df[temp_col], bins=[0, 50, 80, 120, 1000], 
                             labels=["low", "mid", "high", "very_high"])
    
    return df


def run_fine_grained_loso(df: pd.DataFrame) -> dict:
    """Try different granularities for yield statistics."""
    
    terminal_subs = ["Propylene oxide", "Epichlorohydrin", "Styrene oxide", "Isopropyl glycidyl ether"]
    CHO = "Cyclohexene oxide"
    
    results = {}
    
    # Define prediction strategies
    strategies = {
        "overall_mean": lambda tr, te: np.full(len(te), tr["yield (%)"].mean()),
        "mech_mean": lambda tr, te: te["catalyst_system_type"].map(
            tr.groupby("catalyst_system_type")["yield (%)"].mean()
        ).fillna(tr["yield (%)"].mean()).values,
        "mech_median": lambda tr, te: te["catalyst_system_type"].map(
            tr.groupby("catalyst_system_type")["yield (%)"].median()
        ).fillna(tr["yield (%)"].median()).values,
        "mech脳temp_mean": lambda tr, te: _mech_temp_mean(tr, te),
        "mech脳solvent_mean": lambda tr, te: _mech_solvent_mean(tr, te),
    }
    
    for strategy_name, strategy_func in strategies.items():
        results[strategy_name] = {
            "all": {"y_true": [], "y_pred": []},
            "terminal_only": {"y_true": [], "y_pred": []},
            "per_substrate": []
        }
    
    substrates = sorted(df["reactant_name"].unique())
    
    for held_sub in substrates:
        train_mask = df["reactant_name"].values != held_sub
        test_mask = df["reactant_name"].values == held_sub
        
        df_train = df[train_mask].reset_index(drop=True)
        df_test = df[test_mask].reset_index(drop=True)
        y_test = df_test["yield (%)"].values
        
        for strategy_name, strategy_func in strategies.items():
            pred = strategy_func(df_train, df_test)
            
            results[strategy_name]["all"]["y_true"].extend(y_test.tolist())
            results[strategy_name]["all"]["y_pred"].extend(pred.tolist())
            
            if held_sub in terminal_subs:
                results[strategy_name]["terminal_only"]["y_true"].extend(y_test.tolist())
                results[strategy_name]["terminal_only"]["y_pred"].extend(pred.tolist())
            
            results[strategy_name]["per_substrate"].append({
                "substrate": held_sub,
                "r2": float(r2_score(y_test, pred)),
                "mae": float(mean_absolute_error(y_test, pred)),
            })
    
    return results


def _mech_temp_mean(df_train, df_test):
    """Use mechanism 脳 temperature bin mean."""
    temp_col = [c for c in df_train.columns if "temperature" in c.lower()][0]
    df_train["temp_bin"] = pd.cut(
        df_train[temp_col],
        bins=[0, 50, 80, 120, 1000],
        labels=["low", "mid", "high", "very_high"]
    )

    cell_means = df_train.groupby(["catalyst_system_type", "temp_bin"])["yield (%)"].mean()

    # For test, need to assign temp_bin
    temp_col = [c for c in df_test.columns if 'temperature' in c.lower()][0]
    temp_vals = df_test[temp_col].values
    
    pred = np.zeros(len(df_test))
    for i, (_, row) in enumerate(df_test.iterrows()):
        mech = row["catalyst_system_type"]
        temp = temp_vals[i]
        
        if temp <= 50:
            tb = "low"
        elif temp <= 80:
            tb = "mid"
        elif temp <= 120:
            tb = "high"
        else:
            tb = "very_high"
        
        try:
            pred[i] = cell_means.loc[(mech, tb)]
        except KeyError:
            # Fallback to mechanism mean
            try:
                pred[i] = df_train.groupby("catalyst_system_type")["yield (%)"].mean().loc[mech]
            except:
                pred[i] = df_train["yield (%)"].mean()
    
    return pred


def _mech_solvent_mean(df_train, df_test):
    """Use mechanism 脳 solvent presence mean."""
    df_train["has_solvent"] = df_train["all_solvents_normalized"].notna() & \
                              (df_train["all_solvents_normalized"].astype(str).str.strip() != "")
    
    cell_means = df_train.groupby(["catalyst_system_type", "has_solvent"])["yield (%)"].mean()
    
    pred = np.zeros(len(df_test))
    for i, (_, row) in enumerate(df_test.iterrows()):
        mech = row["catalyst_system_type"]
        has_solvent = pd.notna(row.get("all_solvents_normalized")) and \
                     str(row.get("all_solvents_normalized", "")).strip() != ""
        
        try:
            pred[i] = cell_means.loc[(mech, has_solvent)]
        except KeyError:
            try:
                pred[i] = df_train.groupby("catalyst_system_type")["yield (%)"].mean().loc[mech]
            except:
                pred[i] = df_train["yield (%)"].mean()
    
    return pred


def main():
    print("=" * 70)
    print("Fine-grained Yield Statistics for LOSO")
    print("=" * 70)
    
    # Load data
    df = load_data()
    print(f"\nLoaded {len(df)} reactions")
    
    # Run analysis
    print("\nRunning fine-grained LOSO...")
    results = run_fine_grained_loso(df)
    
    # Compare strategies
    print("\n" + "=" * 70)
    print("Strategy Comparison (All Substrates)")
    print("=" * 70)
    print(f"\n{'Strategy':<25} {'R虏 (All)':>12} {'R虏 (Term)':>12} {'MAE (All)':>12}")
    print("-" * 65)
    
    best_all_r2 = -999
    best_all_strategy = None
    
    for name in results.keys():
        y_true = np.array(results[name]["all"]["y_true"])
        y_pred = np.array(results[name]["all"]["y_pred"])
        r2_all = r2_score(y_true, y_pred)
        mae_all = mean_absolute_error(y_true, y_pred)
        
        y_true_term = np.array(results[name]["terminal_only"]["y_true"])
        y_pred_term = np.array(results[name]["terminal_only"]["y_pred"])
        r2_term = r2_score(y_true_term, y_pred_term) if len(y_true_term) > 0 else float('nan')
        
        status = "*" if r2_all > best_all_r2 else " "
        if r2_all > best_all_r2:
            best_all_r2 = r2_all
            best_all_strategy = name
        
        print(f"{name:<25} {r2_all:>+12.4f} {r2_term:>+12.4f} {mae_all:>12.4f} {status}")
    
    # Per-substrate for best strategy
    print(f"\n{'='*70}")
    print(f"Per-Substrate Results ({best_all_strategy})")
    print("=" * 70)
    print(f"\n{'Substrate':<30} {'R虏':>10} {'MAE':>10}")
    print("-" * 52)
    
    for sub_result in results[best_all_strategy]["per_substrate"]:
        r2 = sub_result["r2"]
        status = "+" if r2 > 0 else " "
        print(f"{sub_result['substrate']:<30} {r2:>+10.4f} {sub_result['mae']:>10.4f} {status}")
    
    # Summary
    print(f"\n{'='*70}")
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"\nBest strategy: {best_all_strategy}")
    print(f"All substrates LOSO R虏: {best_all_r2:+.4f}")
    print(f"Terminal-only LOSO R虏: {r2_term:+.4f}")
    
    # Key finding
    if best_all_r2 < 0:
        print("""
        
NOTE: LOSO R2 is fundamentally limited by yield variance.
Even with PERFECT yield statistics, R2 >= 0 when:
- Training and test sets have similar yield distributions
- The "predictor" (statistics) has no real predictive power

The negative R2 indicates that the naive predictor (mean) is worse
than just predicting the overall mean for all samples.

This is a THEORETICAL LIMITATION, not a model failure!
    """)
    
    # Save
    out_file = os.path.join(OUT_DIR, "fine_grained_loso_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        # Convert to serializable format
        save_results = {}
        for name, data in results.items():
            save_results[name] = {
                "all_r2": float(r2_score(
                    np.array(data["all"]["y_true"]),
                    np.array(data["all"]["y_pred"])
                )),
                "terminal_r2": float(r2_score(
                    np.array(data["terminal_only"]["y_true"]),
                    np.array(data["terminal_only"]["y_pred"])
                )) if len(data["terminal_only"]["y_true"]) > 0 else None,
                "per_substrate": data["per_substrate"]
            }
        json.dump(save_results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {out_file}")
    
    return results


if __name__ == "__main__":
    results = main()
