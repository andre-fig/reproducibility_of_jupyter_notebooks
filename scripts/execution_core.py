"""
Core execution logic for notebooks - environment-agnostic.

This module provides execute_notebook_in_current_env() which assumes
it's running in a Python environment with necessary libraries (nbclient, etc.)
and doesn't know about Docker, venv, or any other execution context.
"""

import contextlib
import json
import hashlib
import re
import time
import traceback
import sys
import textwrap
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

CWD_INJECTION_TAG = "__injected_cwd_fix__"
INJECTION_LOG_PREFIX = "[CWD_FIX]"
INJECTION_LOG_PREVIEW_CHARS = 600

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


def _should_retry_with_cwd_injection(exc: BaseException) -> bool:
    """
    Detects if the exception indicates a missing file/relative path issue.
    Used to decide whether to retry using the injected CWD cell.
    """
    text = _strip_ansi(f"{exc}").lower()
    indicators = (
        "filenotfounderror",
        "no such file or directory",
        "notadirectoryerror",
        "isadirectoryerror",
    )
    return any(token in text for token in indicators)


def _build_cwd_injection_cell(notebook_dir: Path) -> Tuple["nbformat.NotebookNode", str]:
    """
    Creates the injected CWD-fix cell and returns it alongside the generated code
    (useful for logging/debugging).
    """
    notebook_dir_str = repr(str(notebook_dir))
    code = textwrap.dedent(
        f"""
        import os
        import pathlib

        __cwd_fix_target = pathlib.Path({notebook_dir_str})
        __cwd_fix_before = pathlib.Path(os.getcwd())
        print("{INJECTION_LOG_PREFIX} before:", __cwd_fix_before)

        if not __cwd_fix_target.exists():
            raise FileNotFoundError(f"Notebook directory not found: {{__cwd_fix_target}}")

        if not __cwd_fix_target.is_dir():
            raise NotADirectoryError(f"Notebook directory is not a directory: {{__cwd_fix_target}}")

        os.chdir(__cwd_fix_target)
        __cwd_fix_after = pathlib.Path(os.getcwd())
        print("{INJECTION_LOG_PREFIX} after:", __cwd_fix_after)
        os.environ["NBEXEC_CWD"] = str(__cwd_fix_after)
        """
    ).strip()

    cell = nbformat.v4.new_code_cell(code)
    cell.metadata = {"tags": [CWD_INJECTION_TAG]}
    return cell, code


def _summarize_injection_cell(cell: "nbformat.NotebookNode") -> Dict[str, Optional[str]]:
    """
    Extracts lightweight information about the injected cell execution so that we
    can log/debug whether it actually ran.
    """
    outputs = cell.get("outputs") or []
    stdout_chunks = []
    for out in outputs:
        if out.get("output_type") == "stream":
            stdout_chunks.append(out.get("text", ""))
    stdout_text = "".join(stdout_chunks).strip()
    stdout_preview = stdout_text[:INJECTION_LOG_PREVIEW_CHARS]

    before_value = None
    after_value = None
    for line in stdout_text.splitlines():
        clean = line.strip()
        if clean.lower().startswith(f"{INJECTION_LOG_PREFIX.lower()} before:"):
            before_value = clean.split(":", 1)[-1].strip()
        if clean.lower().startswith(f"{INJECTION_LOG_PREFIX.lower()} after:"):
            after_value = clean.split(":", 1)[-1].strip()

    return {
        "execution_count": cell.get("execution_count"),
        "stdout_preview": stdout_preview,
        "cwd_before": before_value,
        "cwd_after": after_value,
        "had_output": bool(stdout_text),
    }


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
        "cwd_strategy_used": None,
        "cwd_fallback_attempted": False,
        "cwd_injection_summary": None,
    }
    
    t0 = time.time()
    logger = config.get("logger")
    notebook_dir = notebook_abs_path.parent
    resources = {"metadata": {"path": str(notebook_dir)}}
    
    strategy_override = config.get("cwd_strategy_override")
    enable_cwd_injection = config.get("enable_cwd_injection", True)
    force_cwd_retry = config.get("force_cwd_injection_retry", False)
    
    if strategy_override == "resources_plus_injection":
        strategy_order = ["resources_plus_injection"]
    elif strategy_override == "resources_only":
        strategy_order = ["resources_only"]
    else:
        strategy_order = ["resources_only"]
        if enable_cwd_injection:
            strategy_order.append("resources_plus_injection")
    
    if logger:
        logger.debug(
            "[CWD_FIX] Strategies=%s, notebook_dir=%s, host_cwd=%s",
            strategy_order,
            notebook_dir,
            Path().resolve(),
        )
    
    executed_nb_json = None
    injection_summary: Optional[Dict[str, Optional[str]]] = None
    used_strategy: Optional[str] = None
    last_exception: Optional[BaseException] = None
    last_traceback_str = ""
    
    for idx, strategy_name in enumerate(strategy_order):
        inject_cwd = strategy_name == "resources_plus_injection"
        nb = nbformat.read(str(notebook_abs_path), as_version=4)
        injection_cell_code_preview = None
        
        if inject_cwd:
            cwd_cell, cwd_code = _build_cwd_injection_cell(notebook_dir)
            injection_cell_code_preview = cwd_code[:240].replace("\n", "\\n")
            nb.cells.insert(0, cwd_cell)
            if logger:
                logger.debug(
                    "[CWD_FIX] Injected cell prepared (strategy=%s, preview=%s...)",
                    strategy_name,
                    injection_cell_code_preview,
                )
        elif logger:
            logger.debug("[CWD_FIX] Executing without injected cell (strategy=%s)", strategy_name)
        
        try:
            client = NotebookClient(
                nb,
                timeout=timeout_s,
                kernel_name=kernel_name,
                allow_errors=False,
            )
            try:
                client.execute(resources=resources)
            except TypeError:
                if logger:
                    logger.warning(
                        "[CWD_FIX] NotebookClient.execute() rejected resources argument; retrying without metadata.path"
                    )
                client.execute()
            used_strategy = strategy_name
            
            if inject_cwd:
                injected_cells = [
                    cell for cell in nb.cells
                    if CWD_INJECTION_TAG in cell.metadata.get("tags", [])
                ]
                if injected_cells:
                    injection_summary = _summarize_injection_cell(injected_cells[0])
                    if logger:
                        logger.debug("[CWD_FIX] Injected cell execution summary: %s", injection_summary)
            
            nb.cells = [
                cell for cell in nb.cells
                if CWD_INJECTION_TAG not in cell.metadata.get("tags", [])
            ]
            
            nb_dict = nbformat.writes(nb)
            executed_nb_json = json.loads(nb_dict)
            break
        except Exception as exc:
            last_exception = exc
            try:
                last_traceback_str = traceback.format_exc()
            except Exception:
                last_traceback_str = "Failed to get traceback"
            
            if logger:
                logger.warning(
                    "[CWD_FIX] Execution failed under strategy %s: %s",
                    strategy_name,
                    exc,
                )
            
            is_last_strategy = idx == len(strategy_order) - 1
            if inject_cwd or is_last_strategy:
                break
            
            should_retry = force_cwd_retry or _should_retry_with_cwd_injection(exc)
            if should_retry:
                result["cwd_fallback_attempted"] = True
                if logger:
                    logger.info(
                        "[CWD_FIX] Retrying with injected CWD cell after %s",
                        type(exc).__name__,
                    )
                continue
            break
        finally:
            with contextlib.suppress(Exception):
                client.shutdown_kernel(cancel_pending_tasks=True)
                if logger:
                    logger.debug("[CWD_FIX] Kernel shutdown complete for strategy=%s", strategy_name)
    
    elapsed = round(time.time() - t0, 3)
    
    if executed_nb_json is not None:
        outputs_hash, n_outputs = hash_outputs_from_nbjson(executed_nb_json)
        cells = (executed_nb_json or {}).get("cells") or []
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
        
        result.update({
            "exec_ok": True,
            "elapsed_s": elapsed,
            "outputs_hash_exec": outputs_hash,
            "n_outputs_exec": n_outputs,
            "outputs_hash_canonical": outputs_hash_canonical,
            "outputs_n_cells": n_output_cells,
            "outputs_n_bytes": outputs_n_bytes,
            "executed_notebook": executed_nb_json,
            "cwd_strategy_used": used_strategy,
            "cwd_injection_summary": injection_summary,
        })
    else:
        exc = last_exception or RuntimeError("Notebook execution aborted with unknown error")
        exc_type = type(exc).__name__
        exc_module = type(exc).__module__
        exc_str = str(exc)
        
        result.update({
            "exec_ok": False,
            "elapsed_s": elapsed,
            "exec_exception_type": exc_type,
            "exec_exception_module": exc_module,
            "exec_exception_str": exc_str,
            "exec_traceback_str": last_traceback_str,
            "cwd_strategy_used": used_strategy,
        })
        
        missing_module = extract_missing_module_name(exc)
        if missing_module:
            result["missing_module"] = missing_module
    
    return result
