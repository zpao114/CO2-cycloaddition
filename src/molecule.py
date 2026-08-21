# -*- coding: utf-8 -*-
"""
utils/molecule.py
=================

Helpers for SMILES lookup and DRFP fingerprint parsing:
  - Mol_Manager  : query SMILES from multiple sources with persistent cache
  - parse_fingerprint_from_excel : parse DRFP string into a numpy array
"""
import os
import json
import time
import re
import logging
from typing import Optional, Dict, List, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ------------------------------------------------------------
# ---- single (name, smiles, source) cache record
# ------------------------------------------------------------

class SmiEntry:
    """Single (name, smiles, source) cache record."""
    def __init__(self, name: str, smiles: str, source: str = "unknown"):
        self.name = name
        self.smiles = smiles
        self.source = source
    
    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "smiles": self.smiles,
            "source": self.source
        }
    
    @classmethod
    def from_dict(cls, d: Dict) -> 'SmiEntry':
        return cls(
            name=d.get("name", ""),
            smiles=d.get("smiles", ""),
            source=d.get("source", "unknown")
        )


# ------------------------------------------------------------
# ---- Mol_Manager ----
# ------------------------------------------------------------

class Mol_Manager:
    """Lookup SMILES for chemical names with multi-source fallback.

    Lookup priority:
      1. Hard-coded table (handled by caller)
      2. In-memory cache (self.yes_smi)
      3. Persistent JSON cache
      4. Cactus / PubChem REST APIs
      5. Selenium WebDriver (optional)

    Example:
        mol_mgr = Mol_Manager(cache_path="mol_mgr_cache.json")
        smiles = mol_mgr.get_smi("ethanol")
    """
    
    # Cactus REST endpoint
    CACTUS_URL = "https://cactus.wilkinson.science/names2smiles"
    
    def __init__(self, bro=None, cache_path: str = None, manual_cache_path: str = None):
        """Initialise the molecule manager.

        Args:
            bro: Selenium WebDriver instance (optional).
            cache_path: path to the persistent JSON cache file.
            manual_cache_path: path to a manually maintained CSV cache (columns: name, smiles).
        """
        self.bro = bro  # Selenium driver
        self.cache_path = cache_path
        self.manual_cache_path = manual_cache_path

        # In-memory cache
        self.yes_smi: List[SmiEntry] = []  # successfully resolved
        self.no_smi: List[str] = []         # confirmed unresolvable

        # Persistent on-disk cache
        self._persistence_cache: Dict[str, str] = {}

        # Load caches (manual cache > persistent cache)
        self._load_cache()
    
    def _load_cache(self):
        """Load the persistent on-disk cache from a JSON file."""
        if self.cache_path and os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # Backwards-compatible with older format
                if isinstance(data, dict):
                    if "yes_smi" in data:
                        self.yes_smi = [SmiEntry.from_dict(e) for e in data.get("yes_smi", [])]
                    if "no_smi" in data:
                        self.no_smi = data.get("no_smi", [])
                    if "cache" in data:
                        self._persistence_cache = data.get("cache", {})
                    elif "smiles_cache" in data:
                        self._persistence_cache = data.get("smiles_cache", {})
                elif isinstance(data, dict):
                    # New format: a flat name -> smiles mapping
                    self._persistence_cache = data

                logger.info(f"Loaded {len(self.yes_smi)} cached entries")
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")

        # Load the manually maintained CSV cache (highest priority)
        self._load_manual_cache()

    def _load_manual_cache(self):
        """Load the manually maintained CSV cache (highest priority)."""
        if not self.manual_cache_path or not os.path.exists(self.manual_cache_path):
            return
        try:
            import pandas as pd
            df_cache = pd.read_csv(self.manual_cache_path, encoding="utf-8-sig")
            for _, row in df_cache.iterrows():
                name = str(row.iloc[0]).strip()
                smiles = str(row.iloc[1]).strip() if len(row) > 1 else ""
                if name and smiles and smiles.lower() not in ("", "nan", "none"):
                    # Write manual entries straight into the persistent cache (highest priority)
                    self._persistence_cache[name] = smiles
            logger.info(f"Loaded {len(df_cache)} entries from manual cache: {self.manual_cache_path}")
        except Exception as e:
            logger.warning(f"Failed to load manual cache: {e}")
    
    def save(self):
        """Save the in-memory cache to a JSON file."""
        if not self.cache_path:
            return
            
        try:
            data = {
                "yes_smi": [e.to_dict() for e in self.yes_smi],
                "no_smi": self.no_smi,
                "cache": self._persistence_cache
            }
            
            os.makedirs(os.path.dirname(self.cache_path) or '.', exist_ok=True)
            with open(self.cache_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
                
            logger.debug(f"Saved {len(self.yes_smi)} entries to cache")
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")
    
    def _query_cactus(self, name: str) -> Optional[str]:
        """Resolve a chemical name via the Cactus REST API.

        Args:
            name: chemical name.
        Returns:
            SMILES string, or None on failure.
        """
        try:
            import urllib.request
            
            # URL-encode the chemical name
            import urllib.parse
            encoded_name = urllib.parse.quote(name)
            url = f"{self.CACTUS_URL}?name={encoded_name}"
            
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=30) as response:
                smiles = response.read().decode('utf-8').strip()
                
            if smiles and not smiles.startswith("No unique"):
                return smiles
            return None
            
        except Exception as e:
            logger.debug(f"Cactus query failed for '{name}': {e}")
            return None
    
    def _query_pubchem(self, name: str) -> Optional[str]:
        """Resolve a chemical name via the PubChem REST API.

        Args:
            name: chemical name.
        Returns:
            SMILES string, or None on failure.
        """
        try:
            import urllib.request
            import urllib.parse
            search_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{urllib.parse.quote(name)}/property/IsomericSMILES/JSON"
            
            req = urllib.request.Request(search_url)
            with urllib.request.urlopen(req, timeout=30) as response:
                data = json.loads(response.read().decode('utf-8'))
                if "PropertyTable" in data and "Properties" in data["PropertyTable"]:
                    props = data["PropertyTable"]["Properties"]
                    if props:
                        return props[0].get("IsomericSMILES")
                        
            return None
            
        except Exception as e:
            logger.debug(f"PubChem query failed for '{name}': {e}")
            return None
    
    def get_smi(self, name: str, use_cactus: bool = True, use_pubchem: bool = True) -> Optional[str]:
        """Resolve SMILES for a chemical name with full cache + API fallback.

        Args:
            name: chemical name.
            use_cactus: whether to query the Cactus API.
            use_pubchem: whether to query the PubChem API.
        Returns:
            SMILES string, or None if unresolvable.
        """
        if not name or not name.strip():
            return None
            
        name = name.strip()
        
        # Check the "known-unresolvable" cache first
        if name in self.no_smi:
            return None
            
        # Check the persistent cache
        if name in self._persistence_cache:
            smiles = self._persistence_cache[name]
            if smiles:
                self.yes_smi.append(SmiEntry(name, smiles, source="persistence_cache"))
            return smiles if smiles else None
        
        # Try Cactus
        if use_cactus:
            smiles = self._query_cactus(name)
            if smiles:
                self._persistence_cache[name] = smiles
                self.yes_smi.append(SmiEntry(name, smiles, source="cactus"))
                return smiles
        
        # Try PubChem
        if use_pubchem:
            smiles = self._query_pubchem(name)
            if smiles:
                self._persistence_cache[name] = smiles
                self.yes_smi.append(SmiEntry(name, smiles, source="pubchem"))
                return smiles
        
        # Confirm unresolvable
        self.no_smi.append(name)
        self._persistence_cache[name] = ""
        return None
    
    def add_known(self, name: str, smiles: str, source: str = "manual"):
        """Register a known SMILES into the cache.

        Args:
            name: chemical name.
            smiles: SMILES string.
            source: provenance label (e.g. "manual", "literature").
        """
        if name and smiles:
            entry = SmiEntry(name, smiles, source)
            self.yes_smi.append(entry)
            self._persistence_cache[name] = smiles

    def set_manual_cache_path(self, path: str):
        """Set the manual cache path at runtime and reload (for switching cache files mid-run)."""
        self.manual_cache_path = path
        self._load_manual_cache()


# ------------------------------------------------------------
# ---- DRFP fingerprint parsing ----
# ------------------------------------------------------------

def parse_fingerprint_from_excel(fp_string) -> Optional:
    """Parse a DRFP fingerprint string from an Excel/CSV cell into a numpy array.

    Accepted formats:
      1. JSON-style list: "[0.0, 1.0, ...]"
      2. Bracketed space-separated: "[0 0 1 0 ...]" (used by co2_drfp.csv)
      3. Numpy repr: "array([0.0, 1.0, ...])"

    Args:
        fp_string: fingerprint string.
    Returns:
        numpy array, or None on failure.
    """
    if fp_string is None:
        return None

    # If already a numpy array, return a copy
    try:
        import numpy as np
        if isinstance(fp_string, np.ndarray):
            return fp_string
    except ImportError:
        pass

    # String preprocessing
    if isinstance(fp_string, str):
        s = fp_string.strip()

        # Null/empty guard
        if not s or s.lower() in ('nan', 'none', '', '[]'):
            return None

        try:
            import numpy as np
            # Try JSON-style "[0.0, 1.0, ...]"
            if s.startswith('['):
                # First try standard JSON (requires float strings)
                try:
                    data = json.loads(s.replace("'", '"'))
                    return np.array(data, dtype=np.float64)
                except (json.JSONDecodeError, ValueError):
                    pass
                # Fallback: bracket-stripped, whitespace-separated integers
                inner = s.strip().lstrip('[').rstrip(']')
                parts = inner.split()
                if parts:
                    try:
                        return np.array([float(v) for v in parts], dtype=np.float64)
                    except ValueError:
                        pass

            # Try whitespace-separated "0.0 1.0 0.5 ..."
            parts = s.split()
            if len(parts) > 1:
                try:
                    values = [float(v) for v in parts]
                    if values:
                        return np.array(values, dtype=np.float64)
                except ValueError:
                    pass

            # Try comma-separated "0.0,1.0,0.5,..."
            parts2 = s.replace('[', '').replace(']', '').replace('(', '').replace(')', '').split(',')
            if parts2 and len(parts2) > 1:
                try:
                    values = [float(v) for v in parts2 if v.strip()]
                    if values:
                        return np.array(values, dtype=np.float64)
                except ValueError:
                    pass
        except (ValueError, json.JSONDecodeError):
            pass

    return None


def encode_drfp(rxn_smiles: str, n_bits: int = 2048, radius: int = 3) -> Optional:
    """Encode a reaction SMILES into a DRFP fingerprint.

    Args:
        rxn_smiles: reaction SMILES string.
        n_bits: fingerprint length (default 2048).
        radius: atom-radius parameter (default 3).
    Returns:
        DRFP fingerprint array, or None on failure.
    """
    try:
        from drfp import DrfpEncoder
        fingerprint = DrfpEncoder.encode(
            [rxn_smiles],
            n_fingerprints=n_bits,
            radius=radius
        )
        return fingerprint[0]
    except ImportError:
        logger.warning("drfp library not installed. Run: pip install drfp")
        return None
    except Exception as e:
        logger.debug(f"DRFP encoding failed: {e}")
        return None


# ------------------------------------------------------------
# ---- RDKit helpers ----
# ------------------------------------------------------------

def smiles_to_mol(smiles: str):
    """Convert a SMILES string to an RDKit Mol object.

    Args:
        smiles: SMILES string.
    Returns:
        RDKit Mol, or None on failure.
    """
    try:
        from rdkit import Chem
        return Chem.MolFromSmiles(smiles)
    except ImportError:
        logger.warning("RDKit not installed. Run: conda install -c conda-forge rdkit")
        return None
    except Exception as e:
        logger.debug(f"SMILES parsing failed: {e}")
        return None


def is_valid_smiles(smiles: str) -> bool:
    """Check whether a SMILES string is valid by attempting Mol construction.

    Args:
        smiles: SMILES string.
    Returns:
        True if RDKit can parse the SMILES, False otherwise.
    """
    if not smiles or not isinstance(smiles, str):
        return False
        
    mol = smiles_to_mol(smiles)
    return mol is not None


# ------------------------------------------------------------
# ---- self-test ----
# ------------------------------------------------------------

if __name__ == "__main__":
    # Smoke-test Mol_Manager
    print("Testing Mol_Manager...")
    mol_mgr = Mol_Manager()
    
    test_names = ["ethanol", "acetone", "water", "tetrabutylammonium bromide"]
    for name in test_names:
        smiles = mol_mgr.get_smi(name)
        print(f"  {name}: {smiles}")
    
    # Smoke-test fingerprint parsing
    print("\nTesting fingerprint parsing...")
    import numpy as np
    test_fp = "[0.0, 1.0, 0.5, 0.0]"
    fp = parse_fingerprint_from_excel(test_fp)
    print(f"  Parsed: {fp} (type: {type(fp).__name__})")
    
    # Test space-separated format
    test_fp2 = "0.0 1.0 0.5 0.0"
    fp2 = parse_fingerprint_from_excel(test_fp2)
    print(f"  Parsed (space): {fp2} (type: {type(fp2).__name__ if fp2 else None})")
    
    print("\nAll tests completed.")
