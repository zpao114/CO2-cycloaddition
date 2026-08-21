# CO2 Cycloaddition ML Pipeline

Cross-substrate transferability in CO? / epoxide cycloaddition is bounded by substrate-mechanism orthogonality: a DRFP + GFN2-xTB + PCL-AE + DualBranchANN framework with mechanistic SHAP diagnostics.

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)

---

## Quick Start

```bash
git clone https://github.com/zpao114/CO2-cycloaddition.git
cd CO2-cycloaddition

# Environment
conda create -n co2_ml python=3.10
conda activate co2_ml
pip install -r requirements.txt
conda install -c conda-forge xtb==22.1

# Run pipeline (PowerShell)
.\scripts\run_pipeline_v2.ps1 -List        # show tiers
.\scripts\run_pipeline_v2.ps1 -DryRun     # preview commands
.\scripts\run_pipeline_v2.ps1 -NoXTB      # skip xTB (~6 hr)
.\scripts\run_pipeline_v2.ps1             # run all
```

---

## Key Results

### Headline Performance

| Protocol | Model | R? | Notes |
|----------|-------|-----|-------|
| 5-fold KFold | DualBranchANN | **0.4106** | MAE=0.116 |
| External holdout (15%) | RF | **0.391** | n=347 |
| y-randomization (100 perms) | All models | p < 0.01 | Signal is real |

### LOSO Failure is Structurally Robust

| Protocol | XGB | LGBM | RF | DualANN |
|----------|-----|------|-----|---------|
| LOSO | -0.441 | -0.519 | -2.300 | -5.031 |
| LOMO | +0.153 | +0.063 | +0.072 | +0.094 |

**Key finding**: Cyclohexene oxide (CHO, mean yield 53.8%) sits in a different yield regime than terminal epoxides (mean yield ~88%). CHO drives LOSO failure across all four model architectures.

### Bootstrap 95% CI (B=1000)

| Protocol | R? mean | 95% CI |
|----------|---------|---------|
| LOSO (xTB only) | -0.051 | [-0.082, -0.018] |
| LOSO?LOMO | +0.217 | [+0.178, +0.258] |

---

## Dataset

2,316 CO? cycloaddition reactions across 5 substrates and 5 catalyst mechanism classes.

| Substrate | n | Mean Yield | Unique Catalysts |
|-----------|---|------------|-----------------|
| Styrene oxide | 729 | 85.0% | 84 |
| Epichlorohydrin | 640 | 92.6% | 78 |
| Propylene oxide | 605 | 89.8% | 71 |
| Cyclohexene oxide | 289 | **53.8%** | 45 |
| Isopropyl glycidyl ether | 53 | 89.2% | 17 |

Data source: Reaxys. Funnel: ~12,800 ? 5,263 (PDF parsed) ? 2,316 (final).

---

## Project Structure

```
CO2-cycloaddition/
|-- README.md              # you are here
|-- LICENSE
|-- requirements.txt      # Python dependencies
|
|-- src/                   # source code
|   |-- config.py          # tuned hyperparameters (auto-loaded)
|   |-- data/              # data cleaning, DRFP, xTB
|   |-- models/            # benchmarks, persistence, screening
|   |-- analysis/          # LOSO, mechanism, diagnostics
|   |-- dft/               # ORCA/DFT validation
|   |-- visualization/     # paper figures
|   |-- ci_artifacts/      # SI artifact generators
|
|-- scripts/
|   |-- run_pipeline_v2.ps1  # PowerShell pipeline runner
|
|-- docs/
|   |-- PIPELINE.md        # detailed execution guide
|   |-- RESULTS.md          # ablation & benchmark tables
|   |-- supplementary/      # supplementary information
|   |-- paper/              # manuscript drafts
|
|-- data/                   # input data
|-- results/               # all outputs (gitignored)
```

---

## Method Summary

| Component | Description |
|-----------|-------------|
| **DRFP** | Differential Reaction Fingerprint, 2048-D |
| **GFN2-xTB** | Semi-empirical electronic descriptors (HOMO, LUMO, charges) |
| **PCL-AE** | Property-Co-Learning AutoEncoder, 128-D latent (?=200.0) |
| **DualBranchANN** | Dual-branch ANN: DRFP branch + XTB/condition branch |
| **MechanismAwareLOSO** | Leave-one-substrate-out with mechanistic interpretation |
| **Chemical SHAP** | SHAP values mapped to chemical meaning |

### Tuned Hyperparameters

| Parameter | Value | Source |
|-----------|-------|--------|
| PCL-AE lambda | 200.0 | 17-point scan, 3 seeds |
| Latent dimension | 128 | hard-coded |
| DRFP variant | full | reactants + cat + solv |

---

## Documentation

| Document | Description |
|----------|-------------|
| [README.md](README.md) | This file ? overview, quick start, key results |
| [docs/PIPELINE.md](docs/PIPELINE.md) | Complete pipeline execution guide |
| [docs/RESULTS.md](docs/RESULTS.md) | All ablation and benchmark tables |
| [docs/supplementary/](docs/supplementary/) | Supplementary information (SI) |
| [docs/paper/](docs/paper/) | Manuscript drafts |

---

## Citation

```bibtex
@software{co2_cycloaddition_ml,
  title = {Cross-substrate transferability in CO2 cycloaddition is bounded by
           substrate-mechanism orthogonality},
  author = {zpao114},
  year = {2026},
  url = {https://github.com/zpao114/CO2-cycloaddition}
}
```

Target journal: *ACS Catalysis*.

---

## License

MIT ? see [LICENSE](LICENSE).
