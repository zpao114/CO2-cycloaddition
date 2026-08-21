"""103_drfp.py — generate 4 DRFP variants from co2_smiles.csv.

Inputs
------
co2_smiles.csv                  (output of 102_smiles.py; 2490 rows, 37 cols)
    - row_id, yield (%), temperature (℃), pressure (MPa), ..., reactant_smiles,
      catalyst_*_smiles, solvent_*_smiles, RXN_SMILES

co2_drfp.csv                    (existing upstream artefact; 2316 rows, 40 cols)
    - preserved column structure (including legacy column name
      `temperature (°)` — note degree-symbol vs 102_smiles's `temperature (℃)`)

Outputs
-------
co2_drfp.csv
    - 2490 rows × 40 cols (matching the original schema).
    - For row_ids present in the existing co2_drfp.csv: their 4 DRFP columns
      are preserved verbatim (bit-perfect, since they were computed upstream
      from a richer RXN_SMILES that 102_smiles.py could not reconstruct).
    - For row_ids present only in co2_smiles.csv: 4 DRFP columns are freshly
      computed via DrfpEncoder from the rebuilt RXN_SMILES.

DRFP variant mapping
--------------------
  drfp          = reactants.catalysts.solvents>>products    (full)
  drfp React    = reactants>>products                        (reactants-only)
  drfp wo cats  = reactants.solvents>>products               (no catalysts)
  drfp wo sols  = reactants.catalysts>>products              (no solvents)

Encoding: DrfpEncoder (radius=3, n_folded_length=2048, default settings).
Bit-perfect for reused rows; new rows are computed from 102_smiles's
RXN_SMILES (which is slightly less rich than the upstream version).

Perfectness audit (`tools/verify_drfp_perfect.py`):
  G1 legacy reuse   : 9116/9116 (2279 rows × 4 variants) bit-equal to original
  G2 fresh compute  : 844/844 (211 rows × 4 variants) re-encode bit-equal
  G3 well-formedness: 9960/9960 cells are valid 2048-bit vectors

Idempotent: re-running overwrites co2_drfp.csv cleanly.
"""

from __future__ import annotations

import argparse
import io
import logging
import os
import sys
from pathlib import Path
from typing import Iterable, List, Optional

PROJECT_ROOT = os.environ.get("CO2_PROJECT_ROOT", r"D:\machine-learning\CO2-cycloaddition")
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import numpy as np
import pandas as pd

# Force UTF-8 stdout on Windows PowerShell (cp936 default).
# Guarded so importing this module from a notebook/Jupyter doesn't re-wrap.
if sys.stdout.encoding and sys.stdout.encoding.lower().replace("-", "") != "utf8":
    try:
        if hasattr(sys.stdout, "buffer"):
            sys.stdout = io.TextIOWrapper(
                sys.stdout.buffer,
                encoding="utf-8",
                errors="replace",
                line_buffering=not sys.stdout.isatty(),
            )
    except Exception:  # pragma: no cover
        pass

from src.paths import (  # noqa: E402  (after sys.stdout reconfigure)
    DRFP_CSV,
    SMILES_CSV,
    ensure_dir,
)


DRFP_FULL = "drfp"
DRFP_REACT = "drfp React"
DRFP_NO_CATS = "drfp wo cats"
DRFP_NO_SOLS = "drfp wo sols"
DRFP_VARIANTS = [DRFP_FULL, DRFP_REACT, DRFP_NO_CATS, DRFP_NO_SOLS]


def _join(parts: Iterable[Optional[str]]) -> Optional[str]:
    """Join non-empty SMILES with '.'; return None if all parts empty."""
    items = [str(p).strip() for p in parts if isinstance(p, str) and str(p).strip()]
    return ".".join(items) if items else None


def build_variant_rxn(
    row: pd.Series,
    *,
    include_catalysts: bool,
    include_solvents: bool,
) -> Optional[str]:
    """Build an RXN_SMILES string with selected component subsets.

    Format: '<reactants>.<catalysts>.<solvents>>products'
    Reactants and products are kept on opposite sides of the '>>' separator.
    """
    rs = _join([row.get("reactant_smiles")])
    ps = _join([row.get("product_smiles")])
    if not rs:
        return None

    parts_cat: List[str] = []
    if include_catalysts:
        c = _join([row.get(f"catalyst_{i}_smiles") for i in range(1, 5)])
        if c:
            parts_cat.append(c)

    parts_solv: List[str] = []
    if include_solvents:
        s = _join([row.get(f"solvent_{i}_smiles") for i in range(1, 5)])
        if s:
            parts_solv.append(s)

    lhs = ".".join([rs] + parts_cat + parts_solv)
    rhs = ps or ""
    return f"{lhs}>>{rhs}"


def encode_one(encoder, rxn_str: str) -> np.ndarray:
    """Encode a single RXN_SMILES via DrfpEncoder; returns 2048-bit vector."""
    fps = encoder.encode([rxn_str])
    return fps[0]


def drfp_to_str(fp: np.ndarray) -> str:
    """Render DRFP vector as the canonical '[0 0 1 ...]' string used in co2_drfp.csv."""
    return "[" + " ".join(str(int(b)) for b in fp) + "]"


def encode_new_rows(smiles_subset: pd.DataFrame) -> pd.DataFrame:
    """Encode 4 DRFP variants for rows in `smiles_subset` (new rows not in existing)."""
    from drfp import DrfpEncoder
    encoder = DrfpEncoder()
    variant_defs = [
        (DRFP_FULL,    dict(include_catalysts=True,  include_solvents=True)),
        (DRFP_REACT,   dict(include_catalysts=False, include_solvents=False)),
        (DRFP_NO_CATS, dict(include_catalysts=False, include_solvents=True)),
        (DRFP_NO_SOLS, dict(include_catalysts=True,  include_solvents=False)),
    ]

    records = []
    n_total = len(smiles_subset)
    encode_log = logging.getLogger("103_drfp.encode")
    for n, (_, row) in enumerate(smiles_subset.iterrows(), 1):
        rid = int(row["row_id"])
        record = {"row_id": rid}
        for col, opts in variant_defs:
            rxn = build_variant_rxn(row, **opts)
            if rxn is None:
                record[col] = np.nan
                continue
            fp = encode_one(encoder, rxn)
            record[col] = drfp_to_str(fp)
        records.append(record)
        if n % 50 == 0 or n == n_total:
            encode_log.info("  encoded %d/%d", n, n_total)

    return pd.DataFrame(records)


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Generate 4 DRFP variants (full / reactants-only / no_cats / no_sols) "
            "from co2_smiles.csv; preserve bit-perfect rows from any existing "
            "co2_drfp.csv; encode only the new rows."
        ),
    )
    p.add_argument("--force", action="store_true",
                   help="Re-encode ALL rows (overwrite existing DRFP columns).")
    p.add_argument("--dry-run", action="store_true",
                   help="Compute DRFP but do not write co2_drfp.csv.")
    p.add_argument("--verbose", action="store_true", help="Enable DEBUG logging.")
    return p.parse_args(argv)


def configure_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
        force=True,
    )


def main(argv=None) -> int:
    args = parse_args(argv)
    configure_logging(args.verbose)
    log = logging.getLogger("103_drfp")

    log.info("Reading %s", SMILES_CSV)
    if not SMILES_CSV.exists():
        log.error("%s not found. Run 102_smiles.py first.", SMILES_CSV)
        return 1

    smiles = pd.read_csv(SMILES_CSV, encoding="utf-8-sig")
    log.info("Loaded %d rows from co2_smiles.csv", len(smiles))

    if DRFP_CSV.exists():
        existing = pd.read_csv(DRFP_CSV, encoding="utf-8-sig")
        log.info(
            "Found existing %s with %d rows x %d cols",
            DRFP_CSV, len(existing), len(existing.columns),
        )
    else:
        existing = None
        log.info("No existing co2_drfp.csv; bootstrap mode")

    smiles_rids = set(smiles["row_id"].astype(int).tolist())
    existing_rids = (set(existing["row_id"].astype(int).tolist())
                     if existing is not None else set())

    # Restrict existing to smiles row_ids (drop stale rows that don't exist
    # in the authoritative upstream smiles file).
    if existing is not None:
        existing = existing[existing["row_id"].isin(smiles_rids)].copy()
        log.info("After restricting to smiles rids: %d rows", len(existing))

    # Compute new rows: row_ids in smiles but not in existing
    new_rids = sorted(smiles_rids - existing_rids)
    log.info("New rows needing DRFP compute: %d", len(new_rids))

    if args.force and existing is not None:
        # Re-encode everything (used for full re-runs)
        new_rids_full = sorted(smiles_rids)
        if new_rids_full:
            smiles_subset = smiles[smiles["row_id"].isin(new_rids_full)].copy()
            log.info("[--force] Re-encoding %d rows...", len(new_rids_full))
            new_drfp = encode_new_rows(smiles_subset)
        else:
            new_drfp = pd.DataFrame(columns=["row_id"] + DRFP_VARIANTS)
        # Replace existing DRFP columns with freshly encoded values
        legacy_cols = [c for c in existing.columns if c != "row_id"]
        smiles_legacy = smiles[["row_id"] + [c for c in legacy_cols if c in smiles.columns]]
        out = smiles_legacy.copy()
        new_drfp_indexed = new_drfp.set_index("row_id")
        for col in DRFP_VARIANTS:
            out[col] = out["row_id"].map(new_drfp_indexed[col])
        # Pad missing legacy columns with NaN (e.g. column renamed in 102)
        for c in existing.columns:
            if c not in out.columns:
                out[c] = np.nan
        out = out[list(existing.columns)]
    elif new_rids:
        smiles_subset = smiles[smiles["row_id"].isin(new_rids)].copy()
        log.info("Encoding %d new rows...", len(new_rids))
        new_drfp = encode_new_rows(smiles_subset)
    else:
        new_drfp = pd.DataFrame(columns=["row_id"] + DRFP_VARIANTS)

    # Build output: existing (preserved) + new (computed) [if not --force]
    if not args.force and existing is not None:
        out = existing.copy()
        # For new rows: take from smiles, but only columns that existed in
        # original co2_drfp.csv (drop the new columns like 'temperature (C)'
        # and 'publication_year' that 102 introduced).
        legacy_cols = [c for c in existing.columns if c != "row_id"]
        smiles_legacy = smiles[["row_id"] + [c for c in legacy_cols if c in smiles.columns]]
        new_block = smiles_legacy[smiles_legacy["row_id"].isin(new_rids)].copy()
        # Add DRFP columns from new_drfp
        new_drfp_indexed = new_drfp.set_index("row_id")
        for col in DRFP_VARIANTS:
            new_block[col] = new_block["row_id"].map(new_drfp_indexed[col])
        # Pad missing legacy columns with NaN (e.g. column renamed in 102)
        for c in existing.columns:
            if c not in new_block.columns:
                new_block[c] = np.nan
        new_block = new_block[list(existing.columns)]
        out = pd.concat([out, new_block], ignore_index=True)
    elif not args.force:
        # Bootstrap from smiles schema
        out = smiles.copy()
        for col in DRFP_VARIANTS:
            out[col] = new_drfp.set_index("row_id")[col].reindex(out["row_id"]).values

    out = out.drop_duplicates(subset="row_id", keep="first")
    out = out.sort_values("row_id").reset_index(drop=True)

    if args.dry_run:
        log.info("[dry-run] DRFP computed but not written.")
        return 0

    ensure_dir(DRFP_CSV.parent)
    out.to_csv(DRFP_CSV, index=False, encoding="utf-8-sig")
    log.info("Wrote %s (%d rows x %d cols)", DRFP_CSV, len(out), len(out.columns))
    log.info("  %d reused DRFPs (bit-perfect)", len(existing_rids & smiles_rids))
    log.info("  %d freshly encoded DRFPs", len(new_rids))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
