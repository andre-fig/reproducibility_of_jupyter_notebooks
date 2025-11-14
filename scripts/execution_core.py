"""
Core execution logic for notebooks - environment-agnostic.

This module provides execute_notebook_in_current_env() which assumes
it's running in a Python environment with necessary libraries (nbclient, etc.)
and doesn't know about Docker, venv, or any other execution context.
"""

import json
import hashlib
import re
import time
import traceback
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

try:
    import nbformat
    from nbclient import NotebookClient
except ImportError as e:
    raise ImportError(f"Required libraries not available: {e}. Install nbclient, nbformat.")


# Headless environment variables
HEADLESS_ENV = {
    "DISPLAY": "",
    "QT_QPA_PLATFORM": "offscreen",
    "SDL_VIDEODRIVER": "dummy",
    "SDL_AUDIODRIVER": "dummy",
    "PYGAME_HIDE_SUPPORT_PROMPT": "1",
    "MPLBACKEND": "Agg",
    "PYTHONWARNINGS": "ignore",
    "OPENBLAS_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
}


def canonicalize_outputs_struct(outputs: list) -> list:
    """
    Returns a minimal, ordered representation of outputs for deterministic comparison.
    - Ignores execution_count, volatile metadata, random IDs.
    - Keeps type, stream name (stdout/stderr), and MIME data.
    - Sorts data keys (MIME) and normalizes lists/strings.
    """
    canon = []
    for out in outputs or []:
        otype = out.get("output_type")
        if otype == "stream":
            canon.append({
                "output_type": "stream",
                "name": out.get("name"),
                "text": out.get("text", ""),
            })
        elif otype in ("display_data", "execute_result"):
            data = out.get("data") or {}
            keep = {}
            for k in sorted(data.keys()):
                v = data[k]
                if isinstance(v, list):
                    keep[k] = "".join(str(x) for x in v)
                else:
                    keep[k] = v
            canon.append({
                "output_type": otype,
                "data": keep,
            })
        elif otype == "error":
            canon.append({
                "output_type": "error",
                "ename": out.get("ename"),
                "evalue": out.get("evalue"),
                "traceback": "\n".join(out.get("traceback") or []),
            })
        else:
            data = out.get("data") or {}
            keep = {}
            for k in sorted(data.keys()):
                v = data[k]
                keep[k] = v if not isinstance(v, list) else "".join(str(x) for x in v)
            canon.append({"output_type": otype, "data": keep})
    return canon


ANSI_ESCAPE_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")


def _strip_ansi(text: str) -> str:
    return ANSI_ESCAPE_RE.sub("", text)


def _module_from_text(text: str) -> Optional[str]:
    """
    Returns the top-level module referenced in a ModuleNotFoundError/ImportError message.
    """
    if not text:
        return None
    patterns = [
        r"No module named '([^']+)'",
        r"No module named ([^\s,]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            module_name = match.group(1)
            return module_name.split(".")[0]
    return None


def hash_outputs_from_nbjson(nbjson: dict) -> Tuple[str, int]:
    """
    Computes SHA256 hash of canonicalized outputs from notebook JSON.
    Returns: (hash_hex, count_of_outputs)
    """
    try:
        cells = (nbjson or {}).get("cells") or []
        outs_min = []
        for c in cells:
            if c.get("cell_type") == "code":
                outs_min.extend(canonicalize_outputs_struct(c.get("outputs") or []))
        blob = json.dumps(outs_min, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest(), len(outs_min)
    except Exception:
        return "", 0


def extract_missing_module_name(exc: BaseException) -> Optional[str]:
    """
    Attempts to extract the missing module name from ModuleNotFoundError/ImportError
    *or* wrapper exceptions such as CellExecutionError that embed the message.
    """
    candidates = []
    exc_type = type(exc).__name__

    # Direct exception string (strip ANSI sequences emitted by nbclient)
    candidates.append(_strip_ansi(str(exc)))

    # CellExecutionError exposes additional metadata
    if exc_type == "CellExecutionError":
        candidates.append(_strip_ansi(getattr(exc, "ename", "") or ""))
        candidates.append(_strip_ansi(getattr(exc, "evalue", "") or ""))

    # Include chained exceptions for completeness
    if exc.__cause__:
        candidates.append(_strip_ansi(str(exc.__cause__)))
    if exc.__context__:
        candidates.append(_strip_ansi(str(exc.__context__)))

    for text in candidates:
        module = _module_from_text(text)
        if module:
            return module

    return None


def execute_notebook_in_current_env(
    notebook_abs_path: Path,
    config: Dict,
) -> Dict:
    """
    Executes a notebook in the current Python environment.
    
    This function is environment-agnostic: it doesn't know about Docker, venv, etc.
    It assumes the current Python has nbclient, nbformat, and other required libs.
    
    Args:
        notebook_abs_path: Absolute path to the .ipynb file
        config: Dictionary with:
            - timeout_s: int (timeout in seconds)
            - kernel_name: str (optional, default: "python3")
            - declared_python_version: str (optional, for logging)
            - repo_dir: Path (optional, for PYTHONPATH)
    
    Returns:
        Dictionary with:
            - exec_ok: bool
            - elapsed_s: float
            - outputs_hash_exec: str (SHA256 hex)
            - n_outputs_exec: int
            - exec_exception_type: str (if error)
            - exec_exception_module: str (if error)
            - exec_exception_str: str (if error)
            - exec_traceback_str: str (if error, full traceback; no truncation)
            - missing_module: str (if ModuleNotFoundError/ImportError)
            - executed_notebook: dict (notebook JSON after execution)
    """
    import os
    
    # Apply headless environment variables directly to os.environ
    # (NotebookClient doesn't accept env parameter, so we modify the process environment)
    os.environ.update(HEADLESS_ENV)
    
    # Set PYTHONPATH if repo_dir provided
    if "repo_dir" in config and config["repo_dir"]:
        repo_dir_str = str(config["repo_dir"])
        current_pythonpath = os.environ.get("PYTHONPATH", "")
        os.environ["PYTHONPATH"] = f"{repo_dir_str}:{current_pythonpath}" if current_pythonpath else repo_dir_str
    
    timeout_s = config.get("timeout_s", 300)
    kernel_name = config.get("kernel_name", "python3")
    
    result = {
        "exec_ok": False,
        "elapsed_s": 0.0,
        "outputs_hash_exec": "",
        "n_outputs_exec": 0,
        "executed_notebook": None,
    }
    
    t0 = time.time()
    
    try:
        # Read notebook
        nb = nbformat.read(str(notebook_abs_path), as_version=4)
        
        # Execute with nbclient (environment variables are now set in os.environ)
        client = NotebookClient(
            nb,
            timeout=timeout_s,
            kernel_name=kernel_name,
            allow_errors=False,
        )
        client.execute()
        
        # Serialize executed notebook
        nb_dict = nbformat.writes(nb)
        nb_json = json.loads(nb_dict)
        
        # Compute hash of outputs and canonical fingerprint fields
        outputs_hash, n_outputs = hash_outputs_from_nbjson(nb_json)
        # Count code cells with outputs
        cells = (nb_json or {}).get("cells") or []
        n_output_cells = 0
        outs_min = []
        for c in cells:
            if c.get("cell_type") == "code":
                outs = c.get("outputs") or []
                if outs:
                    n_output_cells += 1
                outs_min.extend(canonicalize_outputs_struct(outs))
        canon_blob = json.dumps(outs_min, ensure_ascii=False, sort_keys=True)
        outputs_hash_canonical = hashlib.sha256(canon_blob.encode("utf-8")).hexdigest()
        outputs_n_bytes = len(canon_blob.encode("utf-8"))
        
        elapsed = round(time.time() - t0, 3)
        
        result.update({
            "exec_ok": True,
            "elapsed_s": elapsed,
            "outputs_hash_exec": outputs_hash,
            "n_outputs_exec": n_outputs,
            "outputs_hash_canonical": outputs_hash_canonical,
            "outputs_n_cells": n_output_cells,
            "outputs_n_bytes": outputs_n_bytes,
            "executed_notebook": nb_json,
        })
        
    except Exception as e:
        elapsed = round(time.time() - t0, 3)
        
        exc_type = type(e).__name__
        exc_module = type(e).__module__
        exc_str = str(e)
        
        # Get full traceback (no truncation for complete traceability)
        try:
            tb_str = traceback.format_exc()
        except Exception:
            tb_str = "Failed to get traceback"
        
        result.update({
            "exec_ok": False,
            "elapsed_s": elapsed,
            "exec_exception_type": exc_type,
            "exec_exception_module": exc_module,
            "exec_exception_str": exc_str,
            "exec_traceback_str": tb_str,
        })
        
        # Extract missing module if applicable
        missing_module = extract_missing_module_name(e)
        if missing_module:
            result["missing_module"] = missing_module
    
    return result
