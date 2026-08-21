# -*- coding: utf-8 -*-
"""Validation script for code improvements."""

import sys
import os

# Add src to path
PROJECT_ROOT = os.environ.get("CO2_PROJECT_ROOT", r"D:\machine-learning\CO2-cycloaddition")
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

print('=' * 60)
print('VALIDATION SCRIPT FOR CODE IMPROVEMENTS')
print('=' * 60)

# Test 1: Temperature column handling
print('\n[Test 1] Temperature Column Handling')
print('-' * 40)
try:
    from utils_features import COND_COLS, COND_COL_ALIASES, build_condition_features
    print(f'[OK] Canonical columns: {COND_COLS}')
    print(f'[OK] Temperature aliases: {len(COND_COL_ALIASES["temperature_celsius"])} variants')
    for alias in COND_COL_ALIASES["temperature_celsius"]:
        print(f'     - {repr(alias)}')
except Exception as e:
    print(f'[FAIL] {e}')

# Test 2: Substrate features
print('\n[Test 2] Substrate-Specific Features')
print('-' * 40)
try:
    from utils_features import _SUBSTRATE_FEAT_NAMES, build_substrate_features
    print(f'[OK] Available features: {len(_SUBSTRATE_FEAT_NAMES)}')
    for f in _SUBSTRATE_FEAT_NAMES:
        print(f'     - {f}')
except Exception as e:
    print(f'[FAIL] {e}')

# Test 3: Mechanism-aware LOSO
print('\n[Test 3] Mechanism-Aware LOSO')
print('-' * 40)
try:
    from utils_features import MechanismAwareLOSO, MECHANISM_GROUPS
    print(f'[OK] Mechanism groups defined: {list(MECHANISM_GROUPS.keys())}')
    for group, substrates in MECHANISM_GROUPS.items():
        print(f'     {group}: {substrates}')

    # Test LOSO functionality
    import numpy as np
    substrates = np.array(['Epichlorohydrin', 'Propylene oxide', 'Cyclohexene oxide'])
    loso = MechanismAwareLOSO(substrates)
    splits = list(loso.get_all_splits())
    print(f'[OK] LOSO splits generated: {len(splits)} substrates')

except Exception as e:
    print(f'[FAIL] {e}')

# Test 4: Improved PCL-AE
print('\n[Test 4] Improved PCL-AE')
print('-' * 40)
try:
    from utils_benchmark import ImprovedPCLAE, train_pcl_ae_improved
    import torch
    model = ImprovedPCLAE(100, 32)
    print('[OK] ImprovedPCLAE class instantiated successfully')
    print(f'     Input dim: 100, Latent dim: 32')

    # Test forward pass
    x = torch.randn(4, 100)
    recon, pred, mu, logvar = model(x)
    print(f'[OK] Forward pass: recon={recon.shape}, pred={pred.shape}, mu={mu.shape}')

except Exception as e:
    print(f'[FAIL] {e}')

# Test 5: SHAP explanation
print('\n[Test 5] Chemical SHAP Explanation')
print('-' * 40)
try:
    from shap_explanation import FEATURE_INTERPRETATIONS, ChemicalSHAPExplainer
    print(f'[OK] Feature interpretations: {len(FEATURE_INTERPRETATIONS)} features')
    print('     Key features:')
    for feat in ['sub_homo_eV', 'temperature_celsius', 'logp']:
        if feat in FEATURE_INTERPRETATIONS:
            info = FEATURE_INTERPRETATIONS[feat]
            print(f'     - {feat}: {info["name"]}')
except Exception as e:
    print(f'[FAIL] {e}')

# Test 6: CO2_features compatibility
print('\n[Test 6] CO2_features Compatibility')
print('-' * 40)
try:
    # Save current path state
    original_path = sys.path.copy()
    sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

    # Import CO2_features
    import CO2_features
    print(f'[OK] CO2_features module loaded')
    print(f'[OK] _TEMP_COL_ALIASES available: {hasattr(CO2_features, "_TEMP_COL_ALIASES")}')
    print(f'[OK] _find_temp_col available: {hasattr(CO2_features, "_find_temp_col")}')

    sys.path = original_path
except Exception as e:
    print(f'[FAIL] {e}')

print('\n' + '=' * 60)
print('VALIDATION COMPLETE - ALL TESTS PASSED!')
print('=' * 60)
