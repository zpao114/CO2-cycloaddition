# Quick syntax check
import ast
import sys

files_to_check = [
    'D:/machine-learning/CO2-cycloaddition/src/utils_features.py',
    'D:/machine-learning/CO2-cycloaddition/src/utils_benchmark.py',
    'D:/machine-learning/CO2-cycloaddition/src/CO2_features.py',
    'D:/machine-learning/CO2-cycloaddition/src/shap_explanation.py',
]

all_ok = True
for f in files_to_check:
    try:
        with open(f, encoding='utf-8') as file:
            ast.parse(file.read())
        print(f'[OK] {f.split("/")[-1]}')
    except SyntaxError as e:
        print(f'[FAIL] {f.split("/")[-1]}: {e}')
        all_ok = False

if all_ok:
    print('\nAll files passed syntax check!')
else:
    print('\nSome files have syntax errors!')
    sys.exit(1)
