"""
Catalog loader for testing modules (YAML-based).
Exposes ALL_TESTING_MODULES: set[str]

This module loads a taxonomy of testing modules from a YAML configuration file.
The taxonomy is extensible beyond the original Pimentel 2019 list.

YAML Format Expected:
    testing_modules:
      legacy:
        - nose
      python_stdlib:
        - unittest
      modern:
        - pytest
        - hypothesis
      mocks:
        - mock
        - unittest.mock

Notes:
- Only top-level module names are stored (before the first '.')
  Example: "unittest.mock" -> "unittest", "pytest.fixtures" -> "pytest"
- The YAML file should be located at: config/testing_modules.yaml
- If the YAML file is missing or invalid, a fallback set is used and a warning is logged
- The fallback set includes: unittest, pytest, nose, hypothesis, mock
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Set
import yaml

logger = logging.getLogger(__name__)

CONFIG_PATH = (Path(__file__).resolve().parent.parent / "config" / "testing_modules.yaml").resolve()

def _load_all_testing_modules() -> Set[str]:
    """
    Loads all testing modules from YAML config file.
    
    Returns:
        Set of top-level module names (e.g., {"unittest", "pytest", "nose", ...})
    
    If YAML is missing or invalid, returns fallback set and logs a warning.
    """
    try:
        with CONFIG_PATH.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        modules = set()
        for group, items in (cfg.get("testing_modules") or {}).items():
            if isinstance(items, list):
                for m in items:
                    if isinstance(m, str) and m.strip():
                        # Extract and store only top-level module name (before first '.')
                        # Example: "unittest.mock" -> "unittest", "pytest.fixtures" -> "pytest"
                        top_level = m.strip().split(".")[0]
                        modules.add(top_level)
        if modules:
            logger.info(f"Loaded {len(modules)} testing modules from {CONFIG_PATH}")
            return modules
        else:
            logger.warning(f"YAML file {CONFIG_PATH} exists but contains no testing modules. Using fallback.")
            return {"unittest", "pytest", "nose", "hypothesis", "mock"}
    except FileNotFoundError:
        logger.warning(f"YAML config not found at {CONFIG_PATH}. Using fallback set of testing modules.")
        return {"unittest", "pytest", "nose", "hypothesis", "mock"}
    except Exception as e:
        logger.warning(f"Error loading YAML config from {CONFIG_PATH}: {e}. Using fallback set.")
        return {"unittest", "pytest", "nose", "hypothesis", "mock"}

ALL_TESTING_MODULES: Set[str] = _load_all_testing_modules()
