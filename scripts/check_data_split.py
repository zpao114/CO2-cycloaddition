import json
import hashlib
from pathlib import Path

ROOT = Path(r"D:\machine-learning\CO2-cycloaddition")
GOLD = ROOT / "_golden_baseline_20260820"

live = json.loads((ROOT / "results/results_data_split/data_split.json").read_text(encoding="utf-8"))
gold = json.loads((GOLD / "results/results_data_split/data_split.json").read_text(encoding="utf-8"))

# Drop timestamps
for d in (live, gold):
    if "metadata" in d and "created_at_utc" in d["metadata"]:
        del d["metadata"]["created_at_utc"]

print(f"live == gold (excluding timestamp): {live == gold}")

def h(p):
    hh = hashlib.sha1()
    with p.open("rb") as f:
        hh.update(f.read())
    return hh.hexdigest()

print(f"live hash:   {h(ROOT / 'results/results_data_split/data_split.json')}")
print(f"golden hash: {h(GOLD / 'results/results_data_split/data_split.json')}")

for k in sorted(set(live.get("metadata", {})) | set(gold.get("metadata", {}))):
    lv = live.get("metadata", {}).get(k)
    gv = gold.get("metadata", {}).get(k)
    if lv != gv:
        print(f"  diff metadata.{k}: live={lv!r} gold={gv!r}")

print(f"train_indices match: {live['holdout']['train_indices'] == gold['holdout']['train_indices']}")
print(f"test_indices match:  {live['holdout']['test_indices'] == gold['holdout']['test_indices']}")
print(f"n_train: live={live['holdout']['n_train']} gold={gold['holdout']['n_train']}")
print(f"n_test:  live={live['holdout']['n_test']}  gold={gold['holdout']['n_test']}")
print(f"n_splits: live={live['kfold']['n_splits']} gold={gold['kfold']['n_splits']}")
print(f"seed: live={live['kfold']['seed']} gold={gold['kfold']['seed']}")
print(f"splits match: {live['kfold']['splits'] == gold['kfold']['splits']}")
