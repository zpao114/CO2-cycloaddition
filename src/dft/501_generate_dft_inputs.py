"""
501_generate_dft_inputs.py
==========================
Auto-generate ORCA 6.1 DFT input files (.inp + .xyz) for the CO2-cycloaddition
validation set.

Default candidate set focuses on the 5 epoxide chemotypes that account for
>95% of the training set, plus the most common catalysts and solvents seen
in the high-yield top-10 virtual-screening hits.

Run:
    python 501_generate_dft_inputs.py                 # uses defaults
    python 501_generate_dft_inputs.py --level medium  # override level
    python 501_generate_dft_inputs.py --out <dir>

Reference: ORCA 6.1 Manual, sections 2.3 (Run types), 2.4 (Output),
3.4.1 (D3/D4 dispersion). All templates follow the manual's `! simple
keyword line` syntax.
"""
import argparse
import os
import sys
from typing import Dict, List, Optional, Tuple

# RDKit is required for 3D embedding; the script exits early with a clear
# error message if it is missing.
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False


WORK_DIR_DEFAULT = r"D:\machine-learning\CO2-cycloaddition\dft_validation"
ORCA_EXE_DEFAULT = r"D:\orca\Orca6.1.Win64.exe"

# Calculation levels mirror the ORCA manual recommendations:
#   - 'fast'    : B3LYP D3BJ def2-SVP     -- quick screening
#   - 'medium'  : B3LYP D3BJ def2-TZVP    -- recommended for the manuscript
#   - 'accurate': wB97X-D4 def2-TZVP      -- final reference (long, optional)
LEVEL_KEYWORDS = {
    "fast":     "B3LYP D3BJ def2-SVP Opt Freq",
    "medium":   "B3LYP D3BJ def2-TZVP Opt Freq",
    "accurate": "wB97X-D4 def2-TZVP TightOpt Freq",
}


# ---------------------------------------------------------------------------
# 1. Representative validation set
# ---------------------------------------------------------------------------
# Names mirror the 5 training-set chemotypes + the 3 catalyst families that
# dominate the high-yield region of the virtual screen.
CANDIDATES: List[Dict] = [
    # ---- Epoxide substrates (5 chemotypes) ----
    {
        "name": "styrene_oxide",
        "smiles": "C1OC1c1ccccc1",
        "role": "substrate",
        "category": "Styrene-oxide family",
        "ref_xtb_key": "sub_styrene_oxide",
    },
    {
        "name": "epoxybutane",
        "smiles": "CCC1CO1",
        "role": "substrate",
        "category": "Aliphatic epoxide",
        "ref_xtb_key": "sub_epoxybutane",
    },
    {
        "name": "epichlorohydrin",
        "smiles": "ClCC1CO1",
        "role": "substrate",
        "category": "Epichlorohydrin",
        "ref_xtb_key": "sub_epichlorohydrin",
    },
    {
        "name": "propylene_oxide",
        "smiles": "CC1CO1",
        "role": "substrate",
        "category": "Propylene-oxide",
        "ref_xtb_key": "sub_propylene_oxide",
    },
    {
        "name": "isopropyl_glycidyl_ether",
        "smiles": "CC(C)OCC1CO1",
        "role": "substrate",
        "category": "Glycidyl ether",
        "ref_xtb_key": "sub_isopropyl_glycidyl_ether",
    },
    # ---- Catalysts (3 families) ----
    {
        "name": "TBAI",
        "smiles": "CCCC[N+](CCCC)(CCCC)CCCC.[I-]",
        "role": "catalyst_ionic_liquid",
        "category": "Ionic liquid",
        "ref_xtb_key": "cat_TBAI",
    },
    {
        "name": "ZnBr2",
        "smiles": "[Br-].[Zn+2].[Br-]",
        "role": "catalyst_metal_halide",
        "category": "Metal halide",
        "ref_xtb_key": "cat_ZnBr2",
    },
    {
        "name": "DBU",
        "smiles": "N1=C2N(CCCC2)CCCC1",
        "role": "catalyst_organic_base",
        "category": "Organic base",
        "ref_xtb_key": "cat_DBU",
    },
    # ---- Solvents ----
    {
        "name": "DMSO",
        "smiles": "CS(C)=O",
        "role": "solvent",
        "category": "Polar aprotic",
        "ref_xtb_key": "solv_DMSO",
    },
    {
        "name": "DMF",
        "smiles": "CN(C)C=O",
        "role": "solvent",
        "category": "Polar aprotic",
        "ref_xtb_key": "solv_DMF",
    },
    # ---- Reactant / product ----
    {"name": "CO2",   "smiles": "O=C=O", "role": "reactant", "category": "Gas",   "ref_xtb_key": "co2"},
    {"name": "cyclic_carbonate_product", "smiles": "O=C1OCCO1", "role": "product", "category": "Product", "ref_xtb_key": "product"},
]

# Catalysts that are salts -> need an explicit charge in the ORCA input.
# Charged species: TBAI (ionic pair, +1/-1). ORCA expects ONE * xyzfile
# block per input, so we generate TBAI cation and anion separately below.
CATION_OF = {
    "TBAI": ("CCCC[N+](CCCC)(CCCC)CCCC", +1, 1, "TBAI_cation"),
    "TBAB": ("CCCC[N+](CCCC)(CCCC)CCCC", +1, 1, "TBAB_cation"),
    "ZnBr2_neutral": (None, 0, 1, "ZnBr2"),
}


# ---------------------------------------------------------------------------
# 2. ORCA input-file template
# ---------------------------------------------------------------------------
# Template uses placeholders: {keywords}, {charge}, {xyz_block}.
# The `! {keywords}` line is the simple keyword form (ORCA Manual 2.3.3).
# `%pal nprocs N end` is the CORRECT parallel block (ORCA Manual 2.5.1);
# the previous version of this file used `%pal nproc N / nprocs N` which is
# invalid ORCA syntax.
#
# Safety: single atoms (TBAI_anion = I-) cannot be optimized or frequency-
# analyzed; ORCA logs "Geometry optimization for a single atom requested"
# and then hangs. We therefore detect single-atom inputs and substitute
# the Opt+Freq flag with SP (single-point) in the caller side.
def make_inp_text(keywords: str, charge: int, mult: int, xyz_block: str) -> str:
    return f"""! {keywords}

%maxcore 4000

%pal
    nprocs 8
end

%output
    Print[ P_OrbEn ] 2        # full orbital energies -> HOMO/LUMO
    Print[ P_MOs ] 1          # MO coefficients
    Print[ P_Mayer ] 1        # Mayer bond orders
    Print[ P_Hirshfeld ] 1    # Hirshfeld charges
    Print[ P_homolumogap ] 1  # SCF-iteration HOMO/LUMO gap
end

* xyz {charge} {mult}
{xyz_block}*
"""


# ---------------------------------------------------------------------------
# 3. SMILES -> 3D -> XYZ
# ---------------------------------------------------------------------------
# Special-case templates for systems that RDKit cannot 3D-embed meaningfully
# (disconnected SMILES, monoatomic ions, or molecules whose geometry is
# pre-known to a high accuracy). Tuple: (atom_count, comment, lines).
_HARDCODED_XYZ: Dict[str, Tuple[int, str, List[str]]] = {
    # ZnBr2 is a linear molecule (D_inf h); the Zn-Br bond length is 2.27 A
    # in the gas phase. RDKit embeds "[Br-].[Zn+2].[Br-]" with all three
    # atoms at (0,0,0), which ORCA rejects ("Zero distance between atoms").
    "ZnBr2": (
        3,
        "ZnBr2  (linear, gas-phase Zn-Br = 2.27 A)",
        ["Br     0.000000     0.000000    -2.270000",
         "Zn     0.000000     0.000000     0.000000",
         "Br     0.000000     0.000000     2.270000"],
    ),
    # Monoatomic ions are placed at the origin; ORCA only needs the charge
    # + multiplicity to evaluate the SP energy.
    "TBAI_anion": (
        1,
        "I-",
        ["I      0.000000     0.000000     0.000000"],
    ),
}


def smiles_to_xyz(smiles: str, name: str) -> Optional[str]:
    """Embed + MMFF-optimize SMILES, return an XYZ string (or None).

    For a small set of pre-defined molecules (ZnBr2, TBAI_anion, ...) we
    skip the RDKit embedding and write a hard-coded reasonable geometry.
    """
    if name in _HARDCODED_XYZ:
        n, comment, rows = _HARDCODED_XYZ[name]
        body = "\n".join(rows) + "\n"
        return f"{n}\n{comment}\n{body}"

    if not RDKIT_AVAILABLE:
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"  [warn] RDKit could not parse SMILES for {name}")
            return None
        mol = Chem.AddHs(mol)
        if AllChem.EmbedMolecule(mol, randomSeed=42) != 0:
            print(f"  [warn] 3D embedding failed for {name}")
            return None
        AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
        conf = mol.GetConformer()
        lines = [f"{mol.GetNumAtoms()}", name]
        for atom in mol.GetAtoms():
            pos = conf.GetAtomPosition(atom.GetIdx())
            lines.append(f"{atom.GetSymbol():2s} {pos.x:12.6f} {pos.y:12.6f} {pos.z:12.6f}")
        return "\n".join(lines) + "\n"
    except Exception as e:
        print(f"  [error] {name}: {e}")
        return None


# ---------------------------------------------------------------------------
# 4. File writers
# ---------------------------------------------------------------------------
def write_file(path: str, content: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def make_xyz_block(xyz_path: str) -> str:
    """Read a .xyz file and return an inline XYZ block for embedding in .inp."""
    with open(xyz_path, "r", encoding="utf-8") as f:
        lines = f.read().strip().splitlines()
    body = lines[2:]  # discard atom count + comment line
    return "\n".join(body) + "\n"


# ---------------------------------------------------------------------------
# 5. Generate inputs
# ---------------------------------------------------------------------------
def generate_for_molecule(cand: Dict, work_dir: str, keywords: str,
                          charge_override: Optional[int] = None,
                          mult_override: Optional[int] = None,
                          name_override: Optional[str] = None) -> Optional[str]:
    name = name_override or cand["name"]
    xyz_path = os.path.join(work_dir, f"{name}.xyz")
    inp_path = os.path.join(work_dir, f"{name}.inp")

    xyz_text = smiles_to_xyz(cand["smiles"], name)
    if xyz_text is None:
        return None
    write_file(xyz_path, xyz_text)

    xyz_block = make_xyz_block(xyz_path)
    charge = charge_override if charge_override is not None else 0
    mult = mult_override if mult_override is not None else 1

    # Safety: an Opt+Freq job on a single atom (e.g. TBAI_anion = I-) causes
    # ORCA to log a warning and then hang indefinitely because the Hessian
    # of a monoatomic system is undefined. Single-point energies are still
    # informative for ions participating in a reaction mechanism, so we
    # # automatically downgrade the keyword line to SP for 1-atom systems.
    n_atoms = len(xyz_block.splitlines())
    if n_atoms == 1 and ("Opt" in keywords or "Freq" in keywords):
        keywords = keywords.replace("Opt", "").replace("Freq", "").strip()
        keywords = f"{keywords} SP" if keywords else "SP"
        print(f"  [info] {name} is a single atom -> downgraded to '{keywords}'")

    inp_text = make_inp_text(keywords, charge, mult, xyz_block)
    write_file(inp_path, inp_text)
    print(f"  -> {inp_path} (charge={charge}, mult={mult}, level={keywords})")
    return inp_path


def generate_batch_script(inp_paths: List[str], work_dir: str, orca_exe: str) -> str:
    """Generate a Windows .bat runner.

    ORCA 6.1 on Windows is invoked as `orca input.inp`. Each invocation
    creates input.out / input.property.txt / input.gbw in the same directory.
    """
    bat_path = os.path.join(work_dir, "run_dft.bat")
    body = ["@echo off",
            "REM Auto-generated by 501_generate_dft_inputs.py",
            f"SET ORCA_EXE={orca_exe}",
            "SET WORK_DIR=%~dp0",
            "",
            f'cd /d "%WORK_DIR%"',
            ""]
    for inp in inp_paths:
        basename = os.path.splitext(os.path.basename(inp))[0]
        body += [
            f'echo ===========================================',
            f'echo Running {basename}',
            f'echo ===========================================',
            f'"%ORCA_EXE%" "{basename}.inp" > "{basename}.out" 2>&1',
            "if errorlevel 1 (",
            f'    echo [error] {basename} failed',
            ") else (",
            f'    echo [ok]    {basename} converged',
            ")",
            "",
        ]
    body += ["echo All ORCA jobs submitted.", "pause"]
    write_file(bat_path, "\n".join(body) + "\n")
    return bat_path


# ---------------------------------------------------------------------------
# 6. Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--level", choices=list(LEVEL_KEYWORDS), default="medium")
    parser.add_argument("--out", default=WORK_DIR_DEFAULT)
    parser.add_argument("--orca", default=ORCA_EXE_DEFAULT)
    parser.add_argument("--nproc", type=int, default=8)
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing ORCA inputs in the output directory.")
    args = parser.parse_args()

    if not RDKIT_AVAILABLE:
        print("RDKit is required (pip install rdkit). Aborting.")
        sys.exit(1)

    work_dir = args.out
    os.makedirs(work_dir, exist_ok=True)
    keywords = LEVEL_KEYWORDS[args.level]

    print("=" * 60)
    print(f"ORCA 6.1 input-file generator  |  level={args.level}  |  {keywords}")
    print(f"Output directory: {work_dir}")
    print("=" * 60)

    inp_paths: List[str] = []

    # Step 1: 5 epoxide substrates, 3 catalyst families, 2 solvents, 2 small molecules.
    # TBAI is the only ionic-liquid catalyst in CANDIDATES; we skip it here and
    # generate the cation (-1) / anion (+1) split inputs in Step 2 below. The
    # neutral "[N+].[I-]" pair as a single .inp would force ORCA into a
    # non-relativistic doublet treatment (it actually triggers a warning) and
    # the resulting orbital energies are not comparable to the cation / anion
    # split we use everywhere else in the workflow.
    print("\n[1] Single-component input files")
    for cand in CANDIDATES:
        if cand["name"] in CATION_OF:
            # already split into cation/anion downstream; skip the neutral pair
            continue
        print(f"\n-> {cand['name']} ({cand['role']})")
        inp = generate_for_molecule(cand, work_dir, keywords)
        if inp:
            inp_paths.append(inp)

    # Step 2: split TBAI into cation / anion, since ORCA expects one charge per input.
    print("\n[2] Charged-component decomposition")
    tba_cation_smiles = "CCCC[N+](CCCC)(CCCC)CCCC"
    inp = generate_for_molecule(
        {"name": "TBAI_cation", "smiles": tba_cation_smiles},
        work_dir, keywords, charge_override=+1, mult_override=1,
    )
    if inp:
        inp_paths.append(inp)
    inp = generate_for_molecule(
        {"name": "TBAI_anion", "smiles": "[I-]"},
        work_dir, keywords, charge_override=-1, mult_override=1,
    )
    if inp:
        inp_paths.append(inp)

    # Step 3: batch runner script.
    print("\n[3] Batch runner script")
    bat = generate_batch_script(inp_paths, work_dir, args.orca)
    print(f"  -> {bat}")

    # Step 4: README.
    readme = os.path.join(work_dir, "README_DFT.txt")
    write_file(readme, f"""ORCA 6.1 DFT validation set
================================
Level      : {args.level}  ({keywords})
Workdir    : {work_dir}
Generated  : {len(inp_paths)} .inp files
Batch run  : {bat}

Pipeline (use `run_all.bat` to run all of them in one go):

  501  python 501_generate_dft_inputs.py   generate .inp files  (this script)
  ---  run_dft.bat                          invoke ORCA 6.1 on every .inp
  510  python 510_parse_dft_outputs.py     parse .out -> dft_results_summary.csv
                                                  (also writes <name>.xyz.out)
  512  python 512_xtb_on_dft_geometry.py   re-run GFN2-xTB on the DFT geometries
                                                  (apples-to-apples comparison)
  514  python 514_dft_vs_xtb_report.py     xTB vs DFT table + report

Notes:
  - 714_orca_xtb_bridge.py is a utility module (xyz -> ORCA input, etc.) and
    is not invoked directly.
  - xtb.exe is shipped with the conda env_drfp environment and is auto-found
    on PATH by 104/717 via `where xtb`.

Outputs:
  dft_results_summary.csv              13 rows, one per molecule
  xtb_on_dft_geometry_nosolv.csv       13 rows, xTB on the DFT geometries
  514_dft_vs_xtb_report.csv            per-pair descriptor table
  514_dft_vs_xtb_report.txt            MAE / R / Spearman summary
""")
    print(f"  -> {readme}")

    print("\n" + "=" * 60)
    print(f"Done. {len(inp_paths)} input files written.")
    print(f"Next step: run {bat}")
    print("=" * 60)


if __name__ == "__main__":
    main()
