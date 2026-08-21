"""Utility package for the CO2-cycloaddition data pipeline."""

__version__ = "1.0.0"


def print_dependency_check():
    """Print which DRFP-pipeline dependencies are missing (or all OK)."""
    missing = []
    try:
        import drfp  # noqa: F401
    except ImportError:
        missing.append("drfp")
    try:
        import rdkit  # noqa: F401
    except ImportError:
        missing.append("rdkit")
    try:
        import tqdm  # noqa: F401
    except ImportError:
        missing.append("tqdm")

    if missing:
        print(f"[WARN] Missing dependencies: {', '.join(missing)}")
        print("Install with: pip install " + " ".join(missing))
    else:
        print("[OK] All dependencies installed: drfp, rdkit, tqdm")