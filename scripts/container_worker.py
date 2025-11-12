"""
Container worker: executes a single notebook inside Docker container.

Reads environment variables, calls execute_notebook_in_current_env(),
implements retry logic for missing modules (up to 5 distinct modules),
and writes result.json to /workspace/out/.

Contract:
- Worker never knows about Docker/volumes - only filesystem local (/workspace)
- Worker only cares about "run + try to fix dependencies"
- Host decides image, mounts volumes, copies result.json to CSV
- result.json always contains all expected fields (with null/empty defaults if not applicable)
"""

from __future__ import annotations

import os
import json
import sys
import subprocess
import logging
from pathlib import Path
from typing import Dict, Set

from execution_core import execute_notebook_in_current_env

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    stream=sys.stderr
)
logger = logging.getLogger("container_worker")


def install_module(module_name: str, logger: logging.Logger) -> bool:
    """
    Installs a Python module using pip.
    Returns True if successful, False otherwise.
    """
    try:
        result = subprocess.run(
            ["pip", "install", module_name],
            capture_output=True,
            text=True,
            timeout=300,
            check=False
        )
        if result.returncode == 0:
            logger.info(f"Successfully installed {module_name}")
            return True
        else:
            logger.warning(f"Failed to install {module_name}: {result.stderr[:500]}")
            return False
    except subprocess.TimeoutExpired:
        logger.warning(f"Timeout installing {module_name}")
        return False
    except Exception as e:
        logger.warning(f"Error installing {module_name}: {e}")
        return False


def main():
    """
    Main entry point for container worker.
    Reads env vars, executes notebook with retry logic, writes result.json.
    """
    # Read environment variables
    notebook_rel_path = os.environ.get("NOTEBOOK_REL_PATH")
    timeout_s = int(os.environ.get("TIMEOUT_SECONDS", "300"))
    run_id = os.environ.get("RUN_ID", "unknown")
    declared_python_version = os.environ.get("PYTHON_VERSION_DECLARED", "")
    notebook_id = os.environ.get("NOTEBOOK_ID", "")
    
    if not notebook_rel_path:
        logger.error("NOTEBOOK_REL_PATH environment variable not set")
        sys.exit(1)
    
    # Resolve paths
    repo_dir = Path("/workspace/repo")
    out_dir = Path("/workspace/out")
    notebook_abs_path = repo_dir / notebook_rel_path
    
    if not notebook_abs_path.exists():
        logger.error(f"Notebook not found: {notebook_abs_path}")
        # Ensure all expected fields are present with defaults
        result = {
            "exec_ok": False,
            "elapsed_s": 0.0,
            "outputs_hash_exec": "",
            "n_outputs_exec": 0,
            "outputs_hash_canonical": "",
            "outputs_n_cells": 0,
            "outputs_n_bytes": 0,
            "exec_exception_type": "FileNotFoundError",
            "exec_exception_module": "",
            "exec_exception_str": f"Notebook not found: {notebook_rel_path}",
            "exec_traceback_str": "",
            "missing_module": None,
            "retry_attempted": False,
            "retry_missing_modules_count": 0,
            "retry_missing_modules": [],
            "retry_success": False,
            "exec_env_python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        }
        if notebook_id:
            result["notebook_id"] = notebook_id
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
        sys.exit(1)
    
    logger.info(f"Executing notebook: {notebook_rel_path}")
    logger.info(f"Timeout: {timeout_s}s, Run ID: {run_id}")
    
    # Prepare config for execution
    config = {
        "timeout_s": timeout_s,
        "kernel_name": "python3",
        "declared_python_version": declared_python_version,
        "repo_dir": repo_dir,
    }
    
    # Track installed modules for retry (max 5 distinct modules)
    installed_modules: Set[str] = set()
    max_modules = 5
    retry_attempted = False
    retry_success = False
    
    # Execute with retry loop
    while len(installed_modules) < max_modules:
        result = execute_notebook_in_current_env(notebook_abs_path, config)
        
        if result.get("exec_ok"):
            # Success!
            if retry_attempted:
                retry_success = True
            break
        
        # Do not retry on timeout-like errors
        exc_type = (result.get("exec_exception_type") or "").strip()
        exc_msg = (result.get("exec_exception_str") or "").lower()
        if exc_type in ("TimeoutExpired", "TimeoutError") or "timeout" in exc_msg:
            logger.info("Timeout detected; skipping dependency retry.")
            break
        
        # Check if it's a missing module error
        missing_module = result.get("missing_module")
        if missing_module and missing_module not in installed_modules:
            retry_attempted = True
            logger.info(f"ModuleNotFoundError detected: {missing_module}. Attempting to install...")
            
            if install_module(missing_module, logger):
                installed_modules.add(missing_module)
                logger.info(f"Installed {missing_module}. Re-executing notebook...")
                # Continue loop to retry execution
                continue
            else:
                logger.warning(f"Failed to install {missing_module}. Stopping retry.")
                break
        else:
            # Not a missing module error, or we've already tried this module
            break
    
    # Ensure all expected fields are present (add defaults for missing ones)
    expected_fields = {
        "exec_ok": False,
        "elapsed_s": 0.0,
        "outputs_hash_exec": "",
        "n_outputs_exec": 0,
        "outputs_hash_canonical": "",
        "outputs_n_cells": 0,
        "outputs_n_bytes": 0,
        "exec_exception_type": None,
        "exec_exception_module": None,
        "exec_exception_str": None,
        "exec_traceback_str": None,
        "missing_module": None,
        "retry_attempted": False,
        "retry_missing_modules_count": 0,
        "retry_missing_modules": [],
        "retry_success": False,
        "exec_env_python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
    }
    
    # Add defaults for missing fields
    for key, default_value in expected_fields.items():
        if key not in result:
            result[key] = default_value
    
    # Override with actual values
    result["retry_attempted"] = retry_attempted
    result["retry_missing_modules_count"] = len(installed_modules)
    result["retry_missing_modules"] = sorted(list(installed_modules))  # Ensure JSON-serializable list
    result["retry_success"] = retry_success
    result["exec_env_python_version"] = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    
    if notebook_id:
        result["notebook_id"] = notebook_id
    
    # Save executed notebook if successful
    if result.get("exec_ok") and result.get("executed_notebook"):
        out_dir.mkdir(parents=True, exist_ok=True)
        executed_nb_path = out_dir / notebook_abs_path.name
        try:
            import nbformat
            nb = nbformat.from_dict(result["executed_notebook"])
            nbformat.write(nb, str(executed_nb_path), version=4)
            logger.info(f"Saved executed notebook to {executed_nb_path}")
        except Exception as e:
            logger.warning(f"Failed to save executed notebook: {e}")
    
    # Write result.json
    out_dir.mkdir(parents=True, exist_ok=True)
    result_json_path = out_dir / "result.json"
    
    # Remove executed_notebook from JSON (too large, already saved as file)
    result_for_json = {k: v for k, v in result.items() if k != "executed_notebook"}
    
    result_json_path.write_text(
        json.dumps(result_for_json, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    logger.info(f"Result written to {result_json_path}")
    
    # Exit with appropriate code
    sys.exit(0 if result.get("exec_ok") else 1)


if __name__ == "__main__":
    main()
