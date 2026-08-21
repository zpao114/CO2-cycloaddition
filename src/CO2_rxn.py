# -*- coding: utf-8 -*-
"""
CO2_rxn.py
==========
Backwards-compatible alias for `utils_rxn.py`.

Historically the project shipped two parallel files (`utils_rxn.py` and
`CO2_rxn.py`) that exported the same surface (`read_drfp`, `df_to_rxn_list`,
etc.).  During the 2026-08-12 repo clean-up the canonical name was kept as
`utils_rxn.py`; this file is restored as a *thin forwarder* so that any
remaining `from CO2_rxn import read_drfp` / `from CO2_rxn import df_to_rxn_list`
in the codebase still resolves to the live implementation.

Downstream consumers:
    103_drfp.py                       (line 64)
    CO2_features.py                   (lines 23, 404)

This module re-exports the public names from `utils_rxn`.  Nothing else is
implemented here on purpose — keeping a single source of truth.
"""

from utils_rxn import (  # noqa: F401
    read_drfp,
    df_to_rxn_list,
    get_best_drfp_variant,
    set_global_seed,
    XTB_COLS,
    get_xtb_cols,
)

__all__ = [
    'read_drfp',
    'df_to_rxn_list',
    'get_best_drfp_variant',
    'set_global_seed',
    'XTB_COLS',
    'get_xtb_cols',
]

if __name__ == '__main__':
    # Quick smoke check: the alias should resolve to the same code object
    # as utils_rxn.
    import utils_rxn as _u
    assert read_drfp is _u.read_drfp, 'read_drfp mismatch'
    assert df_to_rxn_list is _u.df_to_rxn_list, 'df_to_rxn_list mismatch'
    print('CO2_rxn.py alias OK -- all symbols point at utils_rxn.py')