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

import os
import json
import sys
import subprocess
import logging
from pathlib import Path
from typing import Dict, Set

from scripts.execution_core import execute_notebook_in_current_env

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    stream=sys.stderr
)
logger = logging.getLogger("container_worker")

# Mapping of import names to pip package names
MODULE_NAME_MAP = {
    "sklearn": "scikit-learn",
    "skimage": "scikit-image",
}


def install_repo_dependencies(repo_dir: Path, logger: logging.Logger) -> bool:
    """
    Attempts to install dependencies from repository specification files.
    Checks for requirements.txt, Pipfile, setup.py, pyproject.toml, environment.yml.
    Returns True if at least one installation attempt was made (regardless of success).
    """
    installed_any = False
    
    # Check for requirements.txt
    requirements_txt = repo_dir / "requirements.txt"
    if requirements_txt.exists():
        logger.info(f"Found requirements.txt, installing dependencies...")
        try:
            if sys.version_info >= (3, 7):
                result = subprocess.run(
                    ["pip", "install", "-r", str(requirements_txt)],
                    capture_output=True,
                    text=True,
                    timeout=600,
                    check=False,
                    cwd=str(repo_dir)
                )
            else:
                result = subprocess.run(
                    ["pip", "install", "-r", str(requirements_txt)],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=600,
                    check=False,
                    cwd=str(repo_dir)
                )
            if result.returncode == 0:
                logger.info("Successfully installed dependencies from requirements.txt")
                installed_any = True
            else:
                logger.warning(f"Failed to install from requirements.txt: {result.stderr[:500]}")
                installed_any = True  # Still count as attempted
        except subprocess.TimeoutExpired:
            logger.warning("Timeout installing from requirements.txt")
            installed_any = True
        except Exception as e:
            logger.warning(f"Error installing from requirements.txt: {e}")
            installed_any = True
    
    # Check for Pipfile (requires pipenv)
    pipfile = repo_dir / "Pipfile"
    if pipfile.exists():
        logger.info(f"Found Pipfile, attempting to install with pipenv...")
        try:
            # Try to install pipenv if not available
            if sys.version_info >= (3, 7):
                subprocess.run(["pip", "install", "pipenv"], capture_output=True, text=True, timeout=60, check=False)
            else:
                subprocess.run(["pip", "install", "pipenv"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60, check=False)
            if sys.version_info >= (3, 7):
                result = subprocess.run(
                    ["pipenv", "install", "--system", "--deploy"],
                    capture_output=True,
                    text=True,
                    timeout=600,
                    check=False,
                    cwd=str(repo_dir)
                )
            else:
                result = subprocess.run(
                    ["pipenv", "install", "--system", "--deploy"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=600,
                    check=False,
                    cwd=str(repo_dir)
                )
            if result.returncode == 0:
                logger.info("Successfully installed dependencies from Pipfile")
                installed_any = True
            else:
                logger.warning(f"Failed to install from Pipfile: {result.stderr[:500]}")
                installed_any = True
        except subprocess.TimeoutExpired:
            logger.warning("Timeout installing from Pipfile")
            installed_any = True
        except Exception as e:
            logger.warning(f"Error installing from Pipfile: {e}")
            installed_any = True
    
    # Check for setup.py (install in editable mode)
    setup_py = repo_dir / "setup.py"
    if setup_py.exists():
        logger.info(f"Found setup.py, installing package in editable mode...")
        try:
            if sys.version_info >= (3, 7):
                result = subprocess.run(
                    ["pip", "install", "-e", "."],
                    capture_output=True,
                    text=True,
                    timeout=600,
                    check=False,
                    cwd=str(repo_dir)
                )
            else:
                result = subprocess.run(
                    ["pip", "install", "-e", "."],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=600,
                    check=False,
                    cwd=str(repo_dir)
                )
            if result.returncode == 0:
                logger.info("Successfully installed package from setup.py")
                installed_any = True
            else:
                logger.warning(f"Failed to install from setup.py: {result.stderr[:500]}")
                installed_any = True
        except subprocess.TimeoutExpired:
            logger.warning("Timeout installing from setup.py")
            installed_any = True
        except Exception as e:
            logger.warning(f"Error installing from setup.py: {e}")
            installed_any = True
    
    # Check for pyproject.toml (PEP 518/621)
    pyproject_toml = repo_dir / "pyproject.toml"
    if pyproject_toml.exists():
        logger.info(f"Found pyproject.toml, attempting to install...")
        try:
            # Try installing with pip (PEP 621 support)
            if sys.version_info >= (3, 7):
                result = subprocess.run(
                    ["pip", "install", "."],
                    capture_output=True,
                    text=True,
                    timeout=600,
                    check=False,
                    cwd=str(repo_dir)
                )
            else:
                result = subprocess.run(
                    ["pip", "install", "."],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=600,
                    check=False,
                    cwd=str(repo_dir)
                )
            if result.returncode == 0:
                logger.info("Successfully installed from pyproject.toml")
                installed_any = True
            else:
                logger.warning(f"Failed to install from pyproject.toml: {result.stderr[:500]}")
                installed_any = True
        except subprocess.TimeoutExpired:
            logger.warning("Timeout installing from pyproject.toml")
            installed_any = True
        except Exception as e:
            logger.warning(f"Error installing from pyproject.toml: {e}")
            installed_any = True
    
    # Check for environment.yml (Conda format - we'll try to install with pip if possible)
    environment_yml = repo_dir / "environment.yml"
    if environment_yml.exists():
        logger.info(f"Found environment.yml (Conda format). Note: Conda not available, skipping.")
        # We don't have conda in the container, so we skip this
        # In the future, we could parse the YAML and try to install pip packages
    
    return installed_any


def install_module(module_name: str, logger: logging.Logger) -> bool:
    """
    Installs a Python module using pip.
    Returns True if successful, False otherwise.
    """
    pip_name = MODULE_NAME_MAP.get(module_name, module_name)
    
    try:
        # Python 3.7+ suporta capture_output
        if sys.version_info >= (3, 7):
            result = subprocess.run(
                ["pip", "install", pip_name],
                capture_output=True,
                text=True,
                timeout=300,
                check=False
            )
        else:
            # Python 3.6
            result = subprocess.run(
                ["pip", "install", pip_name],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                timeout=300,
                check=False
            )
        if result.returncode == 0:
            logger.info(f"Successfully installed {pip_name}")
            return True
        else:
            stderr_text = result.stderr if result.stderr else ""
            logger.warning(f"Failed to install {pip_name}: {stderr_text[:500]}")
            return False
    except subprocess.TimeoutExpired:
        logger.warning(f"Timeout installing {pip_name}")
        return False
    except Exception as e:
        logger.warning(f"Error installing {pip_name}: {e}")
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
    policy = os.environ.get("POLICY", "relaxed").lower()
    
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
    logger.info(f"Timeout: {timeout_s}s, Run ID: {run_id}, Policy: {policy}")
    
    # Install repository dependencies if policy is strict
    repo_deps_installed = False
    if policy == "strict":
        logger.info("Policy is 'strict': attempting to install repository dependencies...")
        repo_deps_installed = install_repo_dependencies(repo_dir, logger)
        if repo_deps_installed:
            logger.info("Repository dependency installation completed (may have succeeded or failed)")
        else:
            logger.info("No repository dependency files found")
    
    # Prepare config for execution
    config = {
        "timeout_s": timeout_s,
        "kernel_name": "python3",
        "declared_python_version": declared_python_version,
        "repo_dir": repo_dir,
        "logger": logger,  # Pass logger for debugging
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
