"""
verify_golden.py — re-verify that the optimized tier1 scripts still produce
byte-identical (or near-identical with timestamps) outputs vs the golden baseline.
"""
import hashlib
import json
from pathlib import Path

ROOT = Path(r"D:\machine-learning\CO2-cycloaddition")
GOLD = ROOT / "_golden_baseline_20260820"
LIVE = ROOT

# 11 files: (live_relpath, golden_relpath, mode)
# mode = 'exact'  => byte-identical
# mode = 'ignore_ts' => JSON: ignore 'created_at_utc' field, otherwise identical
FILES = [
    ("data/processed/cleaned.csv",                              "data/processed/cleaned.csv",                              "exact"),
    ("data/processed/co2_smiles.csv",                           "data/processed/co2_smiles.csv",                           "exact"),
    ("data/processed/co2_drfp.csv",                             "data/processed/co2_drfp.csv",                             "exact"),
    ("data/external/extraction_report.csv",                     "data/external/extraction_report.csv",                     "exact"),
    ("data/external/discard_report.csv",                        "data/external/discard_report.csv",                        "exact"),
    ("data/external/cleaned_baseline.json",                     "data/external/cleaned_baseline.json",                     "exact"),
    ("data/external/smiles_baseline.json",                      "data/external/smiles_baseline.json",                      "exact"),
    ("results/results_cho_diagnostic/xtb_results_summary.csv",  "results/results_cho_diagnostic/xtb_results_summary.csv",  "exact"),
    ("results/results_cho_diagnostic/xtb_sanity_summary.csv",   "results/results_cho_diagnostic/xtb_sanity_summary.csv",   "exact"),
    ("results/results_cho_diagnostic/co2_drfp_xtb_extended.csv","results/results_cho_diagnostic/co2_drfp_xtb_extended.csv","exact"),
    ("results/results_data_split/data_split.json",              "results/results_data_split/data_split.json",              "ignore_ts"),
]

def sha1(p: Path) -> str:
    h = hashlib.sha1()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

ok = bad = 0
for live_rel, gold_rel, mode in FILES:
    live_p = LIVE / live_rel
    gold_p = GOLD / gold_rel
    if not live_p.exists():
        print(f"  [MISSING live] {live_rel}")
        bad += 1
        continue
    if not gold_p.exists():
        print(f"  [MISSING gold] {gold_rel}")
        bad += 1
        continue

    if mode == "exact":
        if sha1(live_p) == sha1(gold_p):
            print(f"  [EXACT  ] {live_rel}")
            ok += 1
        else:
            print(f"  [DIFFER ] {live_rel}")
            bad += 1
    elif mode == "ignore_ts":
        live = json.loads(live_p.read_text(encoding="utf-8"))
        gold = json.loads(gold_p.read_text(encoding="utf-8"))
        # Strip timestamp field
        if "metadata" in live and "created_at_utc" in live["metadata"]:
            del live["metadata"]["created_at_utc"]
        if "metadata" in gold and "created_at_utc" in gold["metadata"]:
            del gold["metadata"]["created_at_utc"]
        if live == gold:
            print(f"  [EXACT-ts] {live_rel}  (timestamp-only diff)")
            ok += 1
        else:
            print(f"  [DIFFER ] {live_rel}  (more than timestamp)")
            bad += 1

print(f"\n  matched: {ok} / {ok + bad}")
