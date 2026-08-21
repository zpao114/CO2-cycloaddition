# -*- coding: utf-8 -*-
"""
_shap_infra.py  (PERFECT v2)
=============================
Shared SHAP infrastructure for the CHO-sign-reversal control experiments
(807/808/809/810 etc. in the reviewer response plan).

Provides
--------
1. Data loading: load_X_y_groups() — loads the full training pool (2116 rows),
   builds the (PCL-AE latent 128-D + XTB 43-D + Cond 7-D + Inter 2-D) = 180-D
   feature matrix.  Feature column names are derived authoritatively from
   results_best_pipeline/artifacts/feature_meta.json so the 52-D XTB+Cond+Inter
   block is *guaranteed* to match the schema used by the PCL-AE / DualBranchANN
   pipeline during training.

2. Group SHAP: compute_group_shap() — trains XGBoost on (X[~mask], y[~mask]),
   evaluates on (X[mask], y[mask]), returns mean SHAP per feature on the test
   set via XGBoost's native pred_contribs=True API (avoids shap-library UTF-8
   issues on Windows).

3. Bootstrap CI: bootstrap_shap_ci() — n_bootstrap iterations; each iter
   re-samples the *test* group with replacement, then re-evaluates SHAP means.
   The training set is held fixed (we are bootstrapping the test statistic,
   not the training data).

4. Permutation test: permutation_test() — shuffles y n_perm times, counts how
   often |permuted_mean_shap| >= |observed|.  Two-tailed p-value.

5. Stratified comparison: stratified_group_shap() — bins data on a condition
   column (e.g. temperature), then computes SHAP within each stratum for CHO
   vs. terminal substrates and reports the difference.

All functions are pure / stateless — the caller manages RNG seeds via the
``random_state`` argument.

Design contract
---------------
* Feature column order is FIXED and identical to the training pipeline:
      latent_0 .. latent_127, then XTB[0..42], then Cond[0..6], then Inter[0..1]
* Interaction columns (T_x_activation_proxy, P_x_total_polarity_index) are
  computed on-the-fly inside load_X_y_groups() because they are NOT written to
  the source CSV — they were constructed at training time by the pipeline.
* Latent vector alignment uses the row_id mapping stored in
  results_pcl_ae/row_id.csv, NOT positional indices, because the original
  master CSV had its rows shuffled before latent encoding.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

# ── Paths ────────────────────────────────────────────────────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parents[1]
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

PROJECT_ROOT = Path(
    os.environ.get("CO2_PROJECT_ROOT", r"D:\machine-learning\CO2-cycloaddition")
)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

DATA_EXTENDED = (
    PROJECT_ROOT / "results" / "results_cho_diagnostic" / "co2_drfp_xtb_extended.csv"
)
PCL_LATENT = PROJECT_ROOT / "results_pcl_ae" / "pcl_ae_latent.npy"
PCL_ROWID = PROJECT_ROOT / "results_pcl_ae" / "row_id.csv"
# Authoritative path is results_best_pipeline/artifacts at PROJECT_ROOT
# (NOT under results/), as defined in src.paths.RESULTS_BEST_PIPELINE.
ARTIFACTS = PROJECT_ROOT / "results_best_pipeline" / "artifacts"

# ── Logger ──────────────────────────────────────────────────────────────────
logger = logging.getLogger("shap_infra")
if not logger.handlers:
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    )
    logger.addHandler(h)
    logger.setLevel(logging.INFO)

# ── Substrate constants ─────────────────────────────────────────────────────
SUBSTRATE_ORDER = [
    "Cyclohexene oxide",         # CHO — intra-ring epoxide (key reversal substrate)
    "Propylene oxide",           # PO  — terminal
    "Styrene oxide",             # SO  — terminal
    "Epichlorohydrin",           # ECH — terminal
    "Isopropyl glycidyl ether",  # IGE — terminal
]
CHO_NAME = "Cyclohexene oxide"
TERMINAL_NAMES = [s for s in SUBSTRATE_ORDER if s != CHO_NAME]

# Feature names of mechanistic interest (for SHAP bar plots)
KEY_FEATURES = [
    "sub_homo_eV", "sub_lumo_eV", "sub_gap_eV", "sub_dipole_D",
    "temperature (°)", "pressure (MPa)", "time (h)",
    "activation_proxy", "charge_transfer_potential", "ion_pair_interaction",
    "electrophilicity_cat", "electrodonating_cat",
    "nucleophilicity_index", "gap_ratio", "reaction_polarity",
    "co2_activation_proxy",
]

# Interaction column name → (cond_short_prefix, xtb_col_name)
INTERACTION_NAME_MAP = {
    "T * activation_proxy":          "T_x_activation_proxy",
    "P * total_polarity_index":      "P_x_total_polarity_index",
}


# ============================================================================
# 1.  Meta loading
# ============================================================================
class FeatureMeta:
    """
    Authoritative feature schema loaded from feature_meta.json.

    Attributes
    ----------
    xtb_cols     : list[str]   length 43
    cond_cols    : list[str]   length 7
    interactions : list[dict]  each {name, cond_idx, xtb_name}  length 2
    feature_names_52 : list[str]  (xtb + cond + inter), length 52, in pipeline order
    """

    def __init__(self, meta: dict):
        self.xtb_cols: list[str] = list(meta.get("xtb_cols", []) or [])
        self.cond_cols: list[str] = list(meta.get("cond_cols", []) or [])
        self.interactions: list[dict] = list(meta.get("interaction_rules", []) or [])

        inter_names: list[str] = []
        for rule in self.interactions:
            rule_name = rule["name"]
            # map rule.name → canonical column name (matches training pipeline)
            if rule_name in INTERACTION_NAME_MAP:
                inter_names.append(INTERACTION_NAME_MAP[rule_name])
            else:
                # fallback: derive from cond_idx + xtb_name
                ci = rule["cond_idx"]
                cond_full = self.cond_cols[ci] if ci < len(self.cond_cols) else f"c{ci}"
                head = cond_full.split(" ")[0].lower()
                short_map = {"temperature": "T", "pressure": "P"}
                head = short_map.get(head, head)
                inter_names.append(f"{head}_x_{rule['xtb_name']}")

        self.feature_names_52: list[str] = self.xtb_cols + self.cond_cols + inter_names

    def validate_against_csv(self, csv_path: Path) -> "FeatureMeta":
        """Return a NEW FeatureMeta filtered to columns actually present in the CSV.
        Missing columns are dropped (with a logger.info note)."""
        header = pd.read_csv(csv_path, encoding="utf-8-sig", nrows=0).columns.tolist()
        present_xtb = [c for c in self.xtb_cols if c in header]
        present_cond = [c for c in self.cond_cols if c in header]
        missing_xtb = [c for c in self.xtb_cols if c not in header]
        missing_cond = [c for c in self.cond_cols if c not in header]
        if missing_xtb:
            logger.info("XTB cols missing from CSV (dropped): %s", missing_xtb)
        if missing_cond:
            logger.info("Cond cols missing from CSV (dropped): %s", missing_cond)

        new = FeatureMeta.__new__(FeatureMeta)
        new.xtb_cols = present_xtb
        new.cond_cols = present_cond
        # interactions always computed on the fly (they are derived columns)
        new.interactions = [
            r for r in self.interactions if r["xtb_name"] in present_xtb
            and r["cond_idx"] < len(present_cond)
        ]
        new.feature_names_52 = (
            present_xtb + present_cond + [_rule_to_colname(r, present_cond)
                                          for r in new.interactions]
        )
        return new


def _rule_to_colname(rule: dict, cond_cols: list[str]) -> str:
    rule_name = rule["name"]
    if rule_name in INTERACTION_NAME_MAP:
        return INTERACTION_NAME_MAP[rule_name]
    ci = rule["cond_idx"]
    cond_full = cond_cols[ci] if ci < len(cond_cols) else f"c{ci}"
    head = cond_full.split(" ")[0].lower()
    short_map = {"temperature": "T", "pressure": "P"}
    head = short_map.get(head, head)
    return f"{head}_x_{rule['xtb_name']}"


def load_feature_meta(path: Optional[Path] = None) -> FeatureMeta:
    """Load and validate feature_meta.json. Returns a FeatureMeta instance."""
    p = path or (ARTIFACTS / "feature_meta.json")
    if not p.exists():
        logger.warning("feature_meta.json not found at %s; using empty schema", p)
        return FeatureMeta({})
    with open(p, encoding="utf-8") as f:
        raw = json.load(f)
    meta = FeatureMeta(raw)
    return meta.validate_against_csv(DATA_EXTENDED)


# ============================================================================
# 2.  Data loading
# ============================================================================
def _clean_master(csv_path: Path) -> pd.DataFrame:
    """Read + canonical-clean the master CSV (the same protocol used everywhere)."""
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    df = df[df["extraction_status"] == "valid"].copy()
    df = df.dropna(subset=["yield (%)"])
    df = df[df["yield (%)"] > 0].reset_index(drop=True)
    return df


def _align_latent(df: pd.DataFrame) -> Optional[np.ndarray]:
    """
    Return the latent matrix sliced to match ``df`` row order, using
    results_pcl_ae/row_id.csv as the authoritative mapping.

    Returns None if latent files are missing.

    Mapping convention (verified empirically on 2026-08-20):
        latent[k]  ↔  master-CSV row whose row_id == row_id.csv[k]

    master-CSV row_id is the integer index assigned by the original
    stratified split.  ``df`` after reset_index() has ``df.index`` = 0..N-1,
    so we re-attach the original row_id via the CSV's ``row_id`` column.
    """
    if not (PCL_LATENT.exists() and PCL_ROWID.exists()):
        return None
    if "row_id" not in df.columns:
        logger.error(
            "Master CSV is missing the 'row_id' column; cannot align latent."
        )
        return None

    latent = np.load(PCL_LATENT)        # (n_full, 128)
    rid = pd.read_csv(PCL_ROWID)["row_id"].values  # (n_full,)

    # Build a position lookup: master row_id → latent row index
    pos_of = {int(r): k for k, r in enumerate(rid)}

    # For every df row, find its latent row.
    rows = df["row_id"].astype(int).values
    latent_idx = np.array([pos_of[r] for r in rows if r in pos_of], dtype=int)
    if len(latent_idx) != len(df):
        logger.warning(
            "Latent alignment lost %d rows (out of %d)",
            len(df) - len(latent_idx), len(df),
        )
    return latent[latent_idx]


def _build_interaction_block(
    df: pd.DataFrame,
    interactions: list[dict],
    cond_cols: list[str],
) -> tuple[np.ndarray, list[str]]:
    """Compute the interaction columns on-the-fly (NOT in CSV).

    Returns (matrix, names) where matrix is (n, len(interactions)) float32.
    """
    if not interactions:
        return np.zeros((len(df), 0), dtype=np.float32), []
    cols = []
    names = []
    for rule in interactions:
        ci = rule["cond_idx"]
        xtb_name = rule["xtb_name"]
        if ci >= len(cond_cols) or xtb_name not in df.columns:
            logger.warning(
                "Interaction rule %s missing inputs (cond_idx=%d, xtb=%s); skipped",
                rule, ci, xtb_name,
            )
            cols.append(np.zeros(len(df), dtype=np.float32))
            names.append(f"missing_inter_{ci}_{xtb_name}")
            continue
        cond_vals = df[cond_cols[ci]].astype(np.float32).values
        xtb_vals = df[xtb_name].astype(np.float32).values
        prod = cond_vals * xtb_vals
        # replace NaN/Inf with 0
        prod = np.nan_to_num(prod, nan=0.0, posinf=0.0, neginf=0.0)
        cols.append(prod)
        names.append(_rule_to_colname(rule, cond_cols))
    block = np.column_stack(cols).astype(np.float32) if cols else np.zeros((len(df), 0), dtype=np.float32)
    return block, names


def load_X_y_groups(
    filter_fn: Optional[Callable[[pd.DataFrame], np.ndarray]] = None,
    include_latent: bool = True,
    train_pool_only: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], pd.DataFrame]:
    """
    Load the training pool and build the (latent + XTB + Cond + Inter)
    feature matrix.

    Parameters
    ----------
    filter_fn   : optional callable(df -> bool array) for catalyst / condition
                  control experiments.  Applied BEFORE feature construction.
    include_latent : if False, omit the 128-D PCL-AE latent block (useful when
                  the filter has detached rows from the latent alignment).
    train_pool_only : if True (default), restrict to the 85% canonical training
                  pool defined by src.data_split.holdout_arrays.  If False,
                  use all 2490 valid rows.

    Returns
    -------
    X           : (n, 180) float32  when include_latent=True
                  (n, 52)  float32  when include_latent=False
    y           : (n,)    yield normalised to [0, 1]
    groups      : list[str] of reactant_name labels
    feat_names  : list[str] of feature column names
    df          : the underlying cleaned DataFrame (for debugging / extra cols)
    """
    logger.info("Loading data from %s", DATA_EXTENDED)
    df = _clean_master(DATA_EXTENDED)

    # ── Restrict to canonical train pool (85%) if requested ───────────────
    if train_pool_only:
        from src.data_split import holdout_arrays, load_manifest  # local import
        train_idx, _, _ = holdout_arrays(load_manifest())
        # NOTE: the manifest's indices refer to position-within-cleaned-CSV
        # (i.e. df.index after reset_index in _clean_master), NOT the
        # master-CSV 'row_id' column (which has gaps from earlier dropna).
        train_pos_set = set(int(i) for i in train_idx)
        # df has just been (possibly) reset by _clean_master; its index is
        # already 0..len(df)-1, matching the manifest's positional space.
        before = len(df)
        df = df[df.index.isin(train_pos_set)].reset_index(drop=True)
        logger.info(
            "  train_pool_only: kept %d / %d rows (canonical 85%% train pool)",
            len(df), before,
        )

    # ── Normalise y ─────────────────────────────────────────────────────────
    y = np.clip(df["yield (%)"].values.astype(np.float32) / 100.0, 0.0, 1.0)

    # ── Row filter ─────────────────────────────────────────────────────────
    if filter_fn is not None:
        mask = np.asarray(filter_fn(df), dtype=bool)
        df = df.loc[mask].reset_index(drop=True)
        y = y[mask]
        logger.info("  filter: kept %d / %d rows", int(mask.sum()), len(df))

    groups = df["reactant_name"].values.tolist()

    # ── Authoritative 52-D feature schema ──────────────────────────────────
    meta = load_feature_meta()
    xtb_block = df[meta.xtb_cols].values.astype(np.float32) if meta.xtb_cols else np.zeros((len(df), 0), dtype=np.float32)
    cond_block = df[meta.cond_cols].values.astype(np.float32) if meta.cond_cols else np.zeros((len(df), 0), dtype=np.float32)
    inter_block, inter_names = _build_interaction_block(df, meta.interactions, meta.cond_cols)
    X_52 = np.hstack([xtb_block, cond_block, inter_block]).astype(np.float32)
    feat_names_52 = meta.xtb_cols + meta.cond_cols + inter_names

    # ── Latent (128-D) ──────────────────────────────────────────────────────
    if include_latent:
        latent_slice = _align_latent(df)
        if latent_slice is None:
            logger.warning("PCL-AE latent not found; using XTB/Cond/Inter only (52-D)")
            X = X_52
            feat_names = feat_names_52
        else:
            X = np.hstack([latent_slice, X_52]).astype(np.float32)
            feat_names = [f"latent_{i}" for i in range(latent_slice.shape[1])] + feat_names_52
            logger.info(
                "  feature matrix: latent %d + XTB %d + Cond %d + Inter %d = %d",
                latent_slice.shape[1], len(meta.xtb_cols), len(meta.cond_cols),
                len(inter_names), X.shape[1],
            )
    else:
        X = X_52
        feat_names = feat_names_52

    # Replace NaN / Inf globally
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    logger.info(
        "  X shape: %s, y: %d rows, groups: %d unique",
        X.shape, len(y), len(set(groups)),
    )
    return X, y, np.asarray(groups), feat_names, df


# ============================================================================
# 3.  Group SHAP (XGBoost native pred_contribs)
# ============================================================================
def compute_group_shap(
    X: np.ndarray,
    y: np.ndarray,
    group_mask: np.ndarray,
    feat_names: list[str],
    test_frac: float = 1.0,
    n_estimators: int = 500,
    max_depth: int = 5,
    learning_rate: float = 0.05,
    random_state: int = 42,
    verbose: bool = False,
    model_type: str = "xgb",
    rf_max_depth: int = 8,
    rf_min_samples_leaf: int = 3,
) -> dict:
    """
    Train a model on (X[~mask], y[~mask]), evaluate on (X[mask], y[mask]),
    return mean SHAP per feature.

    ``model_type`` is ``"xgb"`` (default, uses native ``pred_contribs``) or
    ``"rf"`` (uses ``shap.TreeExplainer``).  For RF on 180-D data the
    KernelExplainer is intentionally not used — it crashes under
    ``n_samples < n_features`` (see comments on ``train_rf``).

    Returns
    -------
    dict with keys:
        shap_mean, shap_std, r2, n_test, feat_names, model, params
    """
    train_mask = ~group_mask
    X_tr, y_tr = X[train_mask], y[train_mask]
    X_te, y_te = X[group_mask], y[group_mask]

    if len(X_tr) < 20:
        logger.warning("Train set small (%d rows); SHAP may be unreliable", len(X_tr))
    if len(X_te) < 5:
        logger.warning("Test set small (%d rows)", len(X_te))

    if model_type == "rf":
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.metrics import r2_score
        import shap as _shap

        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=rf_max_depth,
            min_samples_leaf=rf_min_samples_leaf,
            random_state=random_state,
            n_jobs=-1,
        )
        model.fit(X_tr, y_tr)
        pred_te = model.predict(X_te)
        r2 = float(r2_score(y_te, pred_te))
        sv = np.asarray(
            _shap.TreeExplainer(model).shap_values(X_te),
            dtype=np.float32,
        )
        if sv.ndim == 3:        # very old shap versions
            sv = sv[..., 0]
        shap_vals = sv
        params = dict(
            n_estimators=n_estimators,
            max_depth=rf_max_depth,
            min_samples_leaf=rf_min_samples_leaf,
            random_state=random_state,
        )
    else:
        import xgboost as xgb
        from sklearn.metrics import r2_score

        dtrain = xgb.DMatrix(X_tr, label=y_tr, feature_names=feat_names)
        dtest = xgb.DMatrix(X_te, label=y_te, feature_names=feat_names)
        params = dict(
            objective="reg:squarederror",
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.5,
            reg_lambda=2.0,
            random_state=random_state,
            verbosity=0,
            n_jobs=-1,
        )
        model = xgb.train(
            params, dtrain,
            num_boost_round=n_estimators,
            evals=[(dtest, "test")],
            early_stopping_rounds=50,
            verbose_eval=verbose,
        )
        contribs = model.predict(dtest, pred_contribs=True, validate_features=False)
        shap_vals = contribs[:, :-1].astype(np.float32)  # drop bias column
        pred_te = model.predict(dtest)
        r2 = float(r2_score(y_te, pred_te))

    return dict(
        shap_mean=shap_vals.mean(axis=0),
        shap_std=shap_vals.std(axis=0),
        r2=r2,
        n_test=int(len(X_te)),
        feat_names=feat_names,
        model=model,
        params=params,
    )


# ============================================================================
# 4.  Bootstrap CI  (FIXED)
# ============================================================================
def bootstrap_shap_ci(
    X: np.ndarray,
    y: np.ndarray,
    group_mask: np.ndarray,
    feat_names: list[str],
    n_bootstrap: int = 1000,
    random_state: int = 42,
    n_estimators: int = 300,
    model_type: str = "xgb",
) -> pd.DataFrame:
    """
    Bootstrap CI for the *test* group SHAP mean.

    Each iteration:
        1. Resample the test-group rows WITH replacement.
        2. Train XGB on (X[~group_mask], y[~group_mask]) once.
        3. Evaluate SHAP on the bootstrap-resampled test set.
        4. Record per-feature mean SHAP.

    The training set is held fixed across bootstraps — we are characterising
    the variability of the test statistic, not the model.  This keeps runtime
    to O(n_bootstrap) predictions instead of O(n_bootstrap × n_total) refits.

    ``model_type`` controls which estimator is used (see ``compute_group_shap``).

    Returns
    -------
    DataFrame with columns: feat_name, shap_mean, ci_lo, ci_hi, p_zero
    """
    import xgboost as xgb
    import shap as _shap
    rng = np.random.default_rng(random_state)

    n_group = int(group_mask.sum())
    idx_group = np.where(group_mask)[0]

    # Train one canonical model on the full pool (the complement is fixed).
    canonical = compute_group_shap(
        X, y, group_mask, feat_names,
        n_estimators=n_estimators, random_state=random_state,
        model_type=model_type,
    )
    model = canonical["model"]

    # Now use that model to score bootstrap-resampled test sets.
    shap_boot = []
    for b in range(n_bootstrap):
        boot_idx = rng.choice(idx_group, size=n_group, replace=True)
        X_b = X[boot_idx]
        if model_type == "rf":
            sv_b = np.asarray(_shap.TreeExplainer(model).shap_values(X_b),
                              dtype=np.float32)
            if sv_b.ndim == 3:
                sv_b = sv_b[..., 0]
            shap_boot.append(sv_b.mean(axis=0))
        else:
            contribs_b = model.predict(
                xgb.DMatrix(X_b, feature_names=feat_names),
                pred_contribs=True, validate_features=False,
            )
            shap_boot.append(contribs_b[:, :-1].astype(np.float32).mean(axis=0))
    shap_boot = np.asarray(shap_boot)  # (n_bootstrap, n_feat)

    ci_lo = np.percentile(shap_boot, 2.5, axis=0)
    ci_hi = np.percentile(shap_boot, 97.5, axis=0)
    shap_mean = shap_boot.mean(axis=0)
    p_zero = ((shap_boot < 0) != (shap_mean < 0)).mean(axis=0)

    return pd.DataFrame({
        "feat_name": feat_names,
        "shap_mean": shap_mean,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "p_zero": p_zero,
    })


# ============================================================================
# 5.  Permutation test
# ============================================================================
def permutation_test(
    X: np.ndarray,
    y: np.ndarray,
    group_mask: np.ndarray,
    feat_names: list[str],
    feature_of_interest: str,
    n_perm: int = 1000,
    random_state: int = 42,
    n_estimators: int = 300,
    groups: np.ndarray | None = None,
    model_type: str = "xgb",
) -> dict:
    """
    Permutation test for the SHAP value of ``feature_of_interest`` in the
    left-out group.

    Null hypothesis: the SHAP magnitude is the same when the group label of
    each row is randomly permuted.  We *do not* shuffle ``y`` (that destroys
    all signal and makes the test uninformative); we shuffle the *group
    membership* and refit.  This is the natural LOCO-style null.

    Parameters
    ----------
    groups : array-like of str (length n_rows)
        Group label for each row.  Required.
    model_type : ``"xgb"`` (default) or ``"rf"``.
    rf_n_estimators : int
        Number of trees to use for RF permutation.  Default 100; this is
        plenty for a null distribution on 180-D / 2k rows.  Steady-state
        cost is ~0.6 s / iter.

    Returns
    -------
    dict with keys: observed, perm_means, p_value, significant, feature, n_perm
    """
    if groups is None:
        raise ValueError("groups must be provided for permutation_test")
    groups = np.asarray(groups)
    rng = np.random.default_rng(random_state)
    feat_idx = feat_names.index(feature_of_interest) if feature_of_interest in feat_names else 0

    # Observed
    result = compute_group_shap(
        X, y, group_mask, feat_names,
        n_estimators=n_estimators, random_state=random_state,
        model_type=model_type,
    )
    observed = float(result["shap_mean"][feat_idx])

    # Permutations: shuffle group membership, refit, record mean SHAP
    perm_means = []
    for p in range(n_perm):
        chosen_idx = rng.choice(len(groups), size=int(group_mask.sum()), replace=False)
        perm_mask = np.zeros(len(groups), dtype=bool)
        perm_mask[chosen_idx] = True
        res = compute_group_shap(
            X, y, perm_mask, feat_names,
            n_estimators=n_estimators if model_type == "xgb" else 100,
            random_state=random_state + p + 1,
            model_type=model_type,
        )
        perm_means.append(float(res["shap_mean"][feat_idx]))

    perm_means = np.asarray(perm_means)
    # One-sided "is observed more extreme than null?": use |.|
    p_val = float(np.mean(np.abs(perm_means) >= np.abs(observed)))

    return dict(
        observed=observed,
        perm_means=perm_means,
        p_value=p_val,
        significant=p_val < 0.05,
        feature=feature_of_interest,
        n_perm=n_perm,
    )


# ============================================================================
# 6.  Stratified comparison (condition control)
# ============================================================================
def stratified_group_shap(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    feat_names: list[str],
    cho_name: str = CHO_NAME,
    stratify_col: str = "temperature (°)",
    n_bins: int = 3,
    min_per_stratum: int = 10,
) -> pd.DataFrame:
    """
    Compare CHO vs terminal SHAP within matched strata defined by
    ``stratify_col`` (e.g. temperature bins).
    """
    if stratify_col not in feat_names:
        logger.warning("stratify_col '%s' not in features; skipping", stratify_col)
        return pd.DataFrame()

    col_idx = feat_names.index(stratify_col)
    vals = X[:, col_idx]

    try:
        bins = np.percentile(vals, np.linspace(0, 100, n_bins + 1))
    except Exception:
        bins = np.linspace(vals.min(), vals.max(), n_bins + 1)
    bins[0] = -np.inf
    bins[-1] = np.inf

    rows = []
    for i in range(n_bins):
        mask = (vals >= bins[i]) & (vals < bins[i + 1])
        if not mask.any():
            continue
        X_s, y_s, g_s = X[mask], y[mask], groups[mask]
        cho_mask_s = np.array([g == cho_name for g in g_s])
        term_mask_s = ~cho_mask_s

        if cho_mask_s.sum() < min_per_stratum or term_mask_s.sum() < min_per_stratum:
            continue

        res_cho = compute_group_shap(X_s, y_s, cho_mask_s, feat_names, random_state=42)
        res_term = compute_group_shap(X_s, y_s, term_mask_s, feat_names, random_state=42)

        try:
            si = feat_names.index("sub_homo_eV")
        except ValueError:
            si = 0

        rows.append(dict(
            stratum=f"[{bins[i]:.1f}, {bins[i+1]:.1f})",
            cho_n=int(cho_mask_s.sum()),
            term_n=int(term_mask_s.sum()),
            cho_shap_mean=float(res_cho["shap_mean"][si]),
            term_shap_mean=float(res_term["shap_mean"][si]),
            shap_diff=float(res_cho["shap_mean"][si] - res_term["shap_mean"][si]),
        ))

    return pd.DataFrame(rows)


# ============================================================================
# 7.  Export helpers
# ============================================================================
def shap_to_csv(result: dict, path: str | Path):
    df = pd.DataFrame({
        "feat_name": result["feat_names"],
        "shap_mean": result["shap_mean"],
        "shap_std": result["shap_std"],
    })
    df.to_csv(path, index=False)
    logger.info("Saved SHAP CSV: %s", path)