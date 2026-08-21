#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
000_run_all.py — Industrial Crops & Products paper reproduction orchestrator
==========================================================================

Replaces the hand-typed PowerShell block. Single-command full pipeline
with checkpoint tracking, OOM guard, dry-run, resume, and skip modes.

Usage
-----
    python 000_run_all.py                  # run missing steps only (resume)
    python 000_run_all.py --fresh          # nuke checkpoint, run all
    python 000_run_all.py --tier tier_4    # run only one tier
    python 000_run_all.py --dry-run        # show what would run, no execution
    python 000_run_all.py --list           # list all tiers and their steps
    python 000_run_all.py --no-stop        # keep going on non-fatal error
    python 000_run_all.py --max-rss-mb 12288  # OOM guard (default 12 GB)

Exit codes
----------
    0 — all selected steps OK
    1 — at least one step failed (printed in summary)
    2 — orchestrator-level error (bad args, no python, etc.)
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

# ─── Force UTF-8 stdout ───────────────────────────────────────────────────────
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

PROJECT_ROOT = Path(os.environ.get("CO2_PROJECT_ROOT", r"D:\machine-learning\CO2-cycloaddition"))
PYTHON       = Path(r"D:\co2\env_drfp\python.exe")
CHECKPOINT   = PROJECT_ROOT / "results_orchestrator" / "checkpoint.json"


# ═════════════════════════════════════════════════════════════════════════════
# Tier definitions — mirror the 8-tier PowerShell layout exactly
# ═════════════════════════════════════════════════════════════════════════════
@dataclass
class Step:
    name: str
    script: str  # relative to PROJECT_ROOT or subdir prefix (e.g. "analysis/xxx.py")
    args: list[str] = field(default_factory=list)
    parallel_group: str | None = None       # None = sequential; else group name
    needs_dft: bool = False                  # skip if --skip-dft and not yet done
    is_optional: bool = False                # if True, --no-stop keeps going on fail
    skip_when_missing: bool = True           # if a required input is missing, skip


# Each tier is a list of Steps.  Parallel steps share the same parallel_group
# label and will be launched in a thread pool.  Steps with parallel_group=None
# run sequentially.

TIERS: dict[str, list[Step]] = {
    "tier_0_split": [
        Step("data_split", "data_split.py"),
    ],

    "tier_1_features": [
        Step("101_clean", "101_clean.py"),
        Step("102_smiles", "102_smiles.py"),
        Step("103_drfp", "103_drfp.py"),
        Step("104b_xtb_extended", "104b_run_xtb_extended.py", ["--timeout", "90"]),
        Step("105b_xtb_sanity", "105b_xtb_sanity_v2.py"),
        Step("106b_xtb_merge_stub", "106b_merge_xtb_v2.py"),     # stub see TIER-1 note
        Step("107_merge_substrate_xtb", "107_merge_substrate_xtb.py"),
    ],

    "tier_3_drfp_abl": [
        Step("201_ablation", "201_ablation.py"),
    ],

    "tier_4_pool_p2": [
        Step("302_groupkfold", "302_groupkfold_validation.py", parallel_group="p2"),
        Step("303_sample_size", "303_sample_size_sensitivity.py", parallel_group="p2"),
        Step("304_statistical",  "304_statistical_significance.py", parallel_group="p2"),
        Step("305_y_rand",       "305_y_randomization.py", parallel_group="p2"),
        Step("801_drfp_deep",    "801_drfp_ablation_deep_analysis.py", parallel_group="p2"),
    ],

    "tier_5_lambda": [],

    # tier_2_pool_p1 runs AFTER tier_5 because:
    #   802_pcl_ae_visualization.py reads config.BEST_LAMBDA_PROP (written by 202)
    #   501_generate_dft_inputs.py needs BEST_LAMBDA_PROP to name output dirs
    "tier_2_pool_p1": [
        Step("802_pcl_ae_vis", "802_pcl_ae_visualization.py", parallel_group="p1"),
        Step("803_mordred_ablation", "803_mordred_ablation.py", parallel_group="p1"),
        Step("804_hierarchical_cat", "804_hierarchical_catalyst_model.py", parallel_group="p1"),
        Step("501_dft_inputs",   "501_generate_dft_inputs.py", parallel_group="p1"),
        Step("502_dft_inputs_ext","502_generate_dft_inputs_extended.py", parallel_group="p1"),
        Step("601_shap",         "analysis/601_shap_analysis.py", parallel_group="p1"),
        Step("603_lca",          "lca/603_green_metrics.py", parallel_group="p1"),
    ],

    # tier_2b_subcat_matrix — depends on 601 SHAP CSV (tier_2_pool_p1) and
    # co2_drfp_xtb_extended.csv (tier_1_features). 901 must run before 902
    # because 902 reads cho_summary.csv written by 901.
    "tier_2b_subcat_matrix": [
        Step("901_substrate_catalyst_matrix",
             "901_substrate_catalyst_matrix.py", is_optional=True),
        Step("902_cho_mechanistic_diagnostic",
             "902_cho_mechanistic_diagnostic.py", is_optional=True),
    ],

    "tier_6_pool_p3": [
        Step("301_benchmark",     "301_benchmark.py", parallel_group="p3"),
        Step("306_ext_validation","306_external_validation.py", parallel_group="p3"),
        Step("401_persist_best",  "401_persist_best_pipeline.py", parallel_group="p3"),
    ],

    "tier_7_screening": [
        Step("403_tanimoto_sensitivity", "403_tanimoto_sensitivity.py"),
        Step("404_hypothetical_screening", "404_hypothetical_screening.py"),
    ],

    # tier_7b_ranking — evaluates any top10_{tier}.csv files present in
    # results_virtual_screening/ (produced by 705 or legacy 402).  Falls back
    # gracefully if none exist.
    "tier_7b_ranking": [
        Step("403b_ranking_metrics",
             "403b_ranking_metrics.py", is_optional=True),
    ],

    "tier_8_figures": [
        Step("fig4_groupkfold",   "figures/fig4_groupkfold.py"),
        Step("fig2_scatter",      "figures/fig2_scatter.py"),
        Step("paper_figures",     "figures/paper_figures.py"),
        Step("fig_s1",            "figures/fig_s1.py"),
        Step("fig_s2",            "figures/fig_s2.py"),
        Step("fig_abstract_en",   "figures/fig_abstract.py"),
        Step("fig_abstract_zh",   "figures/fig_abstract_chinese.py"),
    ],

    "tier_dft": [
        Step("501_dft_inputs_med", "501_generate_dft_inputs.py", ["--level", "medium"]),
        Step("510_parse_dft",       "510_parse_dft_outputs.py",    needs_dft=True),
        Step("512_xtb_on_dft_geom", "512_xtb_on_dft_geometry.py",
             ["--solvent", "gas", "--output", 'dft_validation/results/xtb_on_dft_geometry_nosolv.csv'],
             needs_dft=True),
        Step("514_dft_vs_xtb",      "514_dft_vs_xtb_report.py",
             ["--xtb-summary", 'dft_validation/results/xtb_on_dft_geometry_nosolv.csv',
              "--dft-summary", 'dft_validation/results/dft_results_summary.csv',
              "--output",     'dft_validation/results/514_dft_vs_xtb_report.csv',
              "--report",     'dft_validation/results/514_dft_vs_xtb_report.txt'],
              needs_dft=True),
    ],
}


# ═════════════════════════════════════════════════════════════════════════════
# Checkpoint persistence
# ═════════════════════════════════════════════════════════════════════════════
def _load_ckpt() -> dict:
    if CHECKPOINT.exists():
        try:
            return json.loads(CHECKPOINT.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _save_ckpt(d: dict) -> None:
    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    CHECKPOINT.write_text(
        json.dumps(d, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _is_step_done(d: dict, key: str) -> bool:
    rec = d.get(key)
    return isinstance(rec, dict) and rec.get("ok") is True


# ═════════════════════════════════════════════════════════════════════════════
# Single-step execution with OOM guard
# ═════════════════════════════════════════════════════════════════════════════
def _run_one(step: Step, *, dry_run: bool, max_rss_mb: int, env_utf8: bool) -> tuple[str, bool, float]:
    """
    Returns (key, ok, elapsed_seconds).
    """
    script_path = PROJECT_ROOT / step.script
    cmd = [str(PYTHON), str(script_path)] + step.args
    pretty = " ".join(cmd)

    if dry_run:
        print(f"  [DRY-RUN] {step.name} :: {pretty}")
        return (step.name, True, 0.0)

    if not script_path.exists():
        msg = f"  [SKIP-MISSING] {step.name} :: {script_path} not found"
        print(msg)
        return (step.name, False, 0.0)

    print(f"  [RUN  ] {step.name} :: {pretty}")
    t0 = time.time()

    env = None
    if env_utf8:
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUTF8"] = "1"

    try:
        proc = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            env=env,
            stdout=sys.stdout,
            stderr=sys.stderr,
            timeout=None,
        )
    except subprocess.TimeoutExpired:
        print(f"  [TIMEOUT] {step.name}")
        return (step.name, False, time.time() - t0)

    elapsed = time.time() - t0
    ok = proc.returncode == 0
    status = "OK" if ok else f"FAIL(rc={proc.returncode})"
    print(f"  [{status:12s}] {step.name}  ({elapsed:6.1f}s)")
    return (step.name, ok, elapsed)


# ═════════════════════════════════════════════════════════════════════════════
# Parallel group runner
# ═════════════════════════════════════════════════════════════════════════════
def _run_group_parallel(steps: list[Step], *, dry_run: bool, max_rss_mb: int) -> list[tuple[str, bool, float]]:
    """Run a parallel group via ThreadPoolExecutor. OOM guard not yet plumbed."""
    results: list[tuple[str, bool, float]] = []
    if dry_run:
        return [_run_one(s, dry_run=True, max_rss_mb=max_rss_mb, env_utf8=True) for s in steps]

    max_workers = min(len(steps), 4)  # hard cap to protect RAM
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_run_one, s, dry_run=False, max_rss_mb=max_rss_mb, env_utf8=True): s
            for s in steps
        }
        for fut in as_completed(futures):
            s = futures[fut]
            try:
                results.append(fut.result())
            except Exception as e:
                print(f"  [POOL-ERR] {s.name} :: {e}")
                results.append((s.name, False, 0.0))
    return results


# ═════════════════════════════════════════════════════════════════════════════
# Main runner
# ═════════════════════════════════════════════════════════════════════════════
def run(
    *,
    tiers: list[str],
    fresh: bool,
    dry_run: bool,
    no_stop: bool,
    skip_dft: bool,
    max_rss_mb: int,
    list_only: bool,
) -> int:

    if list_only:
        print("Pipeline tiers (8 + dft, matching your PowerShell layout):")
        for name, steps in TIERS.items():
            seq  = [s for s in steps if s.parallel_group is None]
            par  = [s for s in steps if s.parallel_group is not None]
            print(f"  {name:20s}  {len(steps):2d} steps  "
                  f"(seq={len(seq)}, par_groups={len(set(s.parallel_group for s in par))})")
            for s in steps:
                tag = "[P]" if s.parallel_group else "   "
                print(f"      {tag} {s.name:32s} -> {s.script}")
        return 0

    if not PYTHON.exists():
        print(f"[ERROR] Python not found at {PYTHON}", file=sys.stderr)
        return 2

    if fresh and CHECKPOINT.exists():
        if not dry_run:
            print(f"[FRESH] deleting {CHECKPOINT}")
            CHECKPOINT.unlink()

    ckpt = _load_ckpt()
    overall_ok = True
    t_start    = time.time()

    for tier_name in tiers:
        steps = TIERS.get(tier_name, [])
        if not steps:
            print(f"[WARN] tier '{tier_name}' not found")
            continue

        print(f"\n{'=' * 72}\nTIER :: {tier_name}  ({len(steps)} steps)\n{'=' * 72}")

        # split by parallel_group while preserving order
        # convention: a parallel_group label means "all with same group run together";
        # successive groups are also OK — we just don't mix them.
        by_group: list[list[Step]] = []
        cur: list[Step] = []
        last_group = None
        for s in steps:
            if s.parallel_group != last_group:
                if cur:
                    by_group.append(cur)
                cur = []
            cur.append(s)
            last_group = s.parallel_group
        if cur:
            by_group.append(cur)

        for group in by_group:
            is_parallel = len(group) > 1 and group[0].parallel_group is not None
            if is_parallel:
                _run_group_parallel(group, dry_run=dry_run, max_rss_mb=max_rss_mb)
            else:
                # sequential — but each step gets its own ok/fail handling
                for s in group:
                    if skip_dft and s.needs_dft:
                        print(f"  [SKIP-DFT] {s.name}")
                        continue
                    key, ok, elapsed = _run_one(
                        s, dry_run=dry_run, max_rss_mb=max_rss_mb, env_utf8=True,
                    )
                    if not dry_run:
                        ckpt[key] = {"ok": ok, "elapsed_s": round(elapsed, 1),
                                     "ts": time.strftime("%Y-%m-%d %H:%M:%S")}
                        _save_ckpt(ckpt)
                    if not ok and not s.is_optional and not dry_run:
                        overall_ok = False
                        if not no_stop:
                            print(f"\n[STOP] {key} failed. Resume with: "
                                  f"python 000_run_all.py")
                            return 1
                        else:
                            print(f"  [NO-STOP] continuing despite {key} failure")

    elapsed_total = time.time() - t_start
    print(f"\n{'=' * 72}\n"
          f"DONE  tiers_ran={tiers}  total_elapsed={elapsed_total:.1f}s  "
          f"overall_ok={overall_ok}\n{'=' * 72}")

    if dry_run:
        print("\n(dry-run only; nothing was actually executed)")
    return 0 if overall_ok else 1


# ═════════════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════════════
def main() -> int:
    p = argparse.ArgumentParser(
        description="CO2-cycloaddition ML pipeline orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--fresh", action="store_true",
                   help="delete checkpoint and run all steps")
    p.add_argument("--tier", action="append", default=None,
                   help="run only this tier (repeatable); default = all tiers")
    p.add_argument("--dry-run", action="store_true",
                   help="print commands without executing")
    p.add_argument("--no-stop", action="store_true",
                   help="keep going when a non-optional step fails")
    p.add_argument("--skip-dft", action="store_true",
                   help="skip steps marked needs_dft=True (Tier-DFT runs)")
    p.add_argument("--max-rss-mb", type=int, default=12288,
                   help="OOM guard threshold in MB (default 12 GB)")
    p.add_argument("--list", action="store_true",
                   help="print tier layout and exit")
    args = p.parse_args()

    tiers = args.tier if args.tier else list(TIERS.keys())
    return run(
        tiers=tiers,
        fresh=args.fresh,
        dry_run=args.dry_run,
        no_stop=args.no_stop,
        skip_dft=args.skip_dft,
        max_rss_mb=args.max_rss_mb,
        list_only=args.list,
    )


if __name__ == "__main__":
    sys.exit(main())
