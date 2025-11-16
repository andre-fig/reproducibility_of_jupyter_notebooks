"""
Notebook collection script with Pimentel 2019 taxonomy implementation.

This script collects Jupyter notebooks from GitHub and computes metrics based on
the taxonomy of "bad practices" from Pimentel et al. (2019).

Pimentel 2019 Taxonomy Implementation:
- The `bad_*` flags implement the taxonomy of bad practices from the original study
- Field names are STABLE and will not change without migration
- Categories:
  * Structure: bad_no_markdown, bad_low_markdown_ratio, bad_untitled, bad_copy_name, bad_non_portable_name
  * Modularization/Testing: bad_no_functions, bad_no_local_imports, bad_no_testing_imports
  * Execution: bad_out_of_order_exec, bad_has_exec_skips, bad_has_non_executed_code_in_middle

notebook_id Contract:
- Computed via compute_notebook_id(repo_full_name, commit_sha, notebook_rel_path)
- Uses SHA256 hash of: "{repo_full_name}|{commit_sha}|{notebook_rel_path}"
- MUST be consistent with execution layer (container_worker.py, execute_notebook_docker.py)
- Uniqueness is enforced: script fails hard if duplicates are found

repo_python_modules:
- Extracts Python modules from repository git tree (including subfolders)
- Limited to 5,000 Python files per repository (parameterizable trade-off)
- This limit prevents excessive time/memory usage in monorepos
- Only top-level module names are stored (before first '.')

Output:
- Generates collection.csv with all metrics and flags
- All `bad_*` flags are boolean (True/False)
- JSON fields (e.g., nb_imported_modules) are stored as JSON strings
"""

from __future__ import annotations

import argparse
import base64
import csv
import time
import datetime as dt
import json
import os
import re
import sys
import tempfile
import shutil
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
import random
import hashlib

import requests
from requests.adapters import HTTPAdapter, Retry
from tqdm import tqdm
import nbformat
import ast
from testing_modules_catalog import ALL_TESTING_MODULES

GITHUB_API = "https://api.github.com"

# ===============================
# Utilidades HTTP e GitHub API
# ===============================

def request_with_backoff(session: requests.Session, method: str, url: str, **kwargs) -> requests.Response:
    """
    Faz a requisição com tratamento de 403 (rate limit/abuse) e 429, respeitando
    os headers Retry-After e X-RateLimit-Reset. Aplica backoff exponencial + jitter.
    """
    max_attempts = 8
    for attempt in range(max_attempts):
        resp = session.request(method, url, **kwargs)

        if resp.status_code not in (403, 429):
            # OK ou outros erros tratados fora por raise_for_status()
            return resp

        # Extrai dicas de espera dos headers
        retry_after = resp.headers.get("Retry-After")
        reset_epoch = resp.headers.get("X-RateLimit-Reset")
        remaining = resp.headers.get("X-RateLimit-Remaining")

        # Tenta identificar mensagem de abuso
        try:
            msg = resp.json().get("message", "")
        except Exception:
            msg = (resp.text or "")[:200]
        abuse = "abuse" in msg.lower()

        # Calcula sleep
        wait = 0.0
        if retry_after:
            # Servidor mandou esperar N segundos
            try:
                wait = float(retry_after)
            except Exception:
                wait = 30.0
        elif remaining == "0" and reset_epoch:
            # Rate limit hard: espera até reset + colchão
            try:
                reset = float(reset_epoch)
            except Exception:
                reset = time.time() + 60
            wait = max(0.0, reset - time.time()) + 5.0
        elif abuse:
            # Abuse detection: seja conservador
            wait = 30.0 * (attempt + 1)
        else:
            # fallback: backoff exponencial
            wait = (2 ** attempt) * 3.0

        # jitter aleatório para dessaturar
        wait += random.uniform(0.5, 2.5)
        time.sleep(wait)

    # Se chegou aqui, esgotou as tentativas
    resp.raise_for_status()
    return resp  # pragma: no cover

def build_session(token: Optional[str]) -> requests.Session:
    s = requests.Session()
    retries = Retry(
        total=8, backoff_factor=0.6,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"]
    )
    s.headers.update({
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "jupyter-reproducibility-replication/1.0"
    })
    if token:
        s.headers.update({"Authorization": f"Bearer {token}"})
    adapter = HTTPAdapter(max_retries=retries)
    s.mount("https://", adapter)
    return s

def gh_search_code(session: requests.Session, q: str, page: int = 1, per_page: int = 100) -> Dict:
    url = f"{GITHUB_API}/search/code"
    resp = request_with_backoff(session, "GET", url, params={"q": q, "page": page, "per_page": per_page})
    resp.raise_for_status()
    return resp.json()

def gh_get_repo(session: requests.Session, owner: str, repo: str) -> Dict:
    url = f"{GITHUB_API}/repos/{owner}/{repo}"
    r = request_with_backoff(session, "GET", url)
    r.raise_for_status()
    return r.json()

def gh_get_contents(session: requests.Session, owner: str, repo: str, path: str, ref: Optional[str]=None) -> Dict:
    url = f"{GITHUB_API}/repos/{owner}/{repo}/contents/{path}"
    params = {"ref": ref} if ref else None
    r = request_with_backoff(session, "GET", url, params=params)
    r.raise_for_status()
    return r.json()

def gh_get_tree(session: requests.Session, owner: str, repo: str, sha: str) -> Dict:
    url = f"{GITHUB_API}/repos/{owner}/{repo}/git/trees/{sha}"
    r = request_with_backoff(session, "GET", url, params={"recursive": 1})
    r.raise_for_status()
    return r.json()

def compute_notebook_id(repo_full_name: str, commit_sha: str, notebook_rel_path: str) -> str:
    """
    Computes a stable identifier for a notebook based on immutable attributes:
    repo full name, snapshot/commit SHA (we use the blob/file SHA here), and the relative path.
    """
    key = f"{repo_full_name}|{commit_sha}|{notebook_rel_path}".encode("utf-8")
    return hashlib.sha256(key).hexdigest()

def gh_search_repos(session: requests.Session, q: str, page: int = 1, per_page: int = 100) -> Dict:
    url = f"{GITHUB_API}/search/repositories"
    resp = request_with_backoff(session, "GET", url, params={"q": q, "page": page, "per_page": per_page, "sort": "updated", "order": "desc"})
    resp.raise_for_status()
    return resp.json()

def list_ipynb_in_repo(session: requests.Session, owner: str, repo: str, ref: Optional[str]) -> list:
    """
    Retorna lista de caminhos .ipynb no branch padrão via árvore recursiva.
    """
    try:
        if not ref:
            r = session.get(f"{GITHUB_API}/repos/{owner}/{repo}")
            r.raise_for_status()
            ref = r.json().get("default_branch","main")
        b = session.get(f"{GITHUB_API}/repos/{owner}/{repo}/branches/{ref}")
        b.raise_for_status()
        tree_sha = b.json()["commit"]["commit"]["tree"]["sha"]
        tree = gh_get_tree(session, owner, repo, tree_sha)
        return [it.get("path","") for it in tree.get("tree", []) if it.get("type")=="blob" and it.get("path","").lower().endswith(".ipynb")]
    except Exception:
        return []

def as_text(src) -> str:
    """Normaliza cell.source (str | list[str] | None) para uma única string."""
    if isinstance(src, str):
        return src
    if isinstance(src, list):
        return "".join(s for s in src if isinstance(s, str))
    return ""


# ===============================
# Particionamento por datas
# ===============================

def partition_date_range(session: requests.Session, start: dt.date, end: dt.date, max_count: int = 1000) -> List[Tuple[dt.date, dt.date]]:
    ranges = [(start, end)]
    final = []
    while ranges:
        a, b = ranges.pop()
        q = f"created:{a.isoformat()}..{b.isoformat()} is:public fork:true"
        try:
            j = gh_search_repos(session, q, page=1, per_page=1)
            total = min(j.get("total_count", 0), 1_000_000)
        except requests.HTTPError:
            # 422 costuma indicar cap de 1000; force split
            if (b - a).days <= 0:
                continue
            mid = a + (b - a)//2
            ranges.append((a, mid))
            ranges.append((mid + dt.timedelta(days=1), b))
            continue

        if total >= max_count and (b - a).days > 0:
            mid = a + (b - a)//2
            ranges.append((a, mid))
            ranges.append((mid + dt.timedelta(days=1), b))
        else:
            final.append((a, b))
    return sorted(final, key=lambda t: t[0])

def iterate_repo_search(session: requests.Session, date_ranges: List[Tuple[dt.date, dt.date]], max_repos: Optional[int]=None) -> Iterable[Dict]:
    seen = 0
    for a, b in date_ranges:
        q = f"created:{a.isoformat()}..{b.isoformat()} is:public fork:true"
        page = 1
        while True:
            try:
                data = gh_search_repos(session, q, page=page, per_page=100)
            except requests.HTTPError as e:
                # Se for 422 do limite de 1000, não adianta continuar essa janela
                if e.response is not None and e.response.status_code == 422:
                    break
                raise
            time.sleep(3.5 + random.uniform(0.0, 2.0))

            items = data.get("items", [])
            if not items:
                break

            for repo in items:
                yield repo
                seen += 1
                if max_repos and seen >= max_repos:
                    return

            page += 1
            if page > 10:  # 10 * 100 = 1000
                break


# ===============================
# Métricas por notebook
# ===============================


ABS_PATH_PATTERN = re.compile(r"(^\/[^ \n\r]+)|([A-Za-z]:\\[^ \n\r]+)")
TRIPLE_BACKTICKS_IN_CODE = re.compile(r"```(?:python|py)?", re.IGNORECASE)

# Windows invalid characters
WINDOWS_INVALID_CHARS = set('?*<>:|"\\/')


def analyze_notebook_structure(nb_json: dict, file_path: str) -> Dict:
    """
    Analyzes notebook structure: markdown ratios, filename quality.
    Returns metrics and bad_* flags.
    """
    nb = nbformat.from_dict(nb_json)
    cells = nb.cells or []
    
    n_code = sum(1 for c in cells if c.get("cell_type") == "code")
    n_markdown = sum(1 for c in cells if c.get("cell_type") == "markdown")
    n_raw = sum(1 for c in cells if c.get("cell_type") == "raw")
    n_total = len(cells)
    
    # Markdown ratio
    nb_markdown_ratio = (n_markdown / n_total * 100.0) if n_total > 0 else 0.0
    
    # Distribution by thirds
    third_size = max(1, n_total // 3)
    begin_cells = cells[:third_size]
    middle_cells = cells[third_size:2*third_size]
    end_cells = cells[2*third_size:]
    
    nb_markdown_at_begin = sum(1 for c in begin_cells if c.get("cell_type") == "markdown")
    nb_markdown_at_middle = sum(1 for c in middle_cells if c.get("cell_type") == "markdown")
    nb_markdown_at_end = sum(1 for c in end_cells if c.get("cell_type") == "markdown")
    
    nb_markdown_at_begin_ratio = (nb_markdown_at_begin / len(begin_cells) * 100.0) if begin_cells else 0.0
    nb_markdown_at_middle_ratio = (nb_markdown_at_middle / len(middle_cells) * 100.0) if middle_cells else 0.0
    nb_markdown_at_end_ratio = (nb_markdown_at_end / len(end_cells) * 100.0) if end_cells else 0.0
    
    # Filename analysis
    nb_filename = os.path.basename(file_path).replace(".ipynb", "")
    nb_title_len = len(nb_filename)
    
    bad_untitled = nb_filename.startswith("Untitled")
    bad_copy_name = ("-Copy" in nb_filename or "Copy of" in nb_filename)
    
    # Portable characters: A-Z, a-z, 0-9, '.', '-'
    nb_has_non_portable_char = any(
        char not in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.-"
        for char in nb_filename
    )
    bad_non_portable_name = nb_has_non_portable_char
    
    # Windows invalid characters
    bad_windows_invalid_char = any(char in WINDOWS_INVALID_CHARS for char in nb_filename)
    
    # Flags
    bad_no_markdown = (n_markdown == 0)
    bad_low_markdown_ratio = (nb_markdown_ratio < 10.0)  # threshold: 10%
    
    return {
        "nb_n_code_cells": n_code,
        "nb_n_markdown_cells": n_markdown,
        "nb_n_raw_cells": n_raw,
        "nb_markdown_ratio": nb_markdown_ratio,
        "nb_markdown_at_begin_ratio": nb_markdown_at_begin_ratio,
        "nb_markdown_at_middle_ratio": nb_markdown_at_middle_ratio,
        "nb_markdown_at_end_ratio": nb_markdown_at_end_ratio,
        "nb_filename": nb_filename,
        "nb_title_len": nb_title_len,
        "nb_has_non_portable_char": nb_has_non_portable_char,
        "nb_has_windows_invalid_char": bad_windows_invalid_char,
        "bad_no_markdown": bad_no_markdown,
        "bad_low_markdown_ratio": bad_low_markdown_ratio,
        "bad_untitled": bad_untitled,
        "bad_copy_name": bad_copy_name,
        "bad_non_portable_name": bad_non_portable_name,
        "bad_windows_invalid_char": bad_windows_invalid_char,
    }


def analyze_notebook_ast(nb_json: dict, repo_python_modules: Optional[List[str]] = None) -> Dict:
    """
    Analyzes notebook AST: functions, classes, imports, local imports, testing modules.
    Returns metrics and bad_* flags.
    """
    nb = nbformat.from_dict(nb_json)
    cells = nb.cells or []
    
    if repo_python_modules is None:
        repo_python_modules = []
    
    # Concatenate all code cells
    all_code = []
    for c in cells:
        if c.get("cell_type") == "code":
            src = as_text(c.get("source"))
            all_code.append(src)
    
    full_source = "\n".join(all_code)
    
    # AST analysis
    nb_n_function_defs = 0
    nb_n_class_defs = 0
    nb_n_loops = 0
    nb_n_conditionals = 0
    imports = []
    imported_modules = set()
    nb_n_local_imports = 0
    nb_imports_testing_module = 0
    nb_imports_test_like_name = 0
    
    try:
        tree = ast.parse(full_source)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                nb_n_function_defs += 1
            elif isinstance(node, ast.ClassDef):
                nb_n_class_defs += 1
            elif isinstance(node, (ast.For, ast.While)):
                nb_n_loops += 1
            elif isinstance(node, ast.If):
                nb_n_conditionals += 1
            elif isinstance(node, ast.Import):
                for n in node.names:
                    mod = n.name.split(".")[0]
                    imports.append(mod)
                    imported_modules.add(mod)
                    if mod in ALL_TESTING_MODULES:
                        nb_imports_testing_module += 1
                    if "test" in mod.lower() or "mock" in mod.lower():
                        nb_imports_test_like_name += 1
                    # Check if it's a local import
                    if mod in repo_python_modules:
                        nb_n_local_imports += 1
            elif isinstance(node, ast.ImportFrom):
                mod = (node.module or "").split(".")[0]
                if mod:
                    imports.append(mod)
                    imported_modules.add(mod)
                    if mod in ALL_TESTING_MODULES:
                        nb_imports_testing_module += 1
                    if "test" in mod.lower() or "mock" in mod.lower():
                        nb_imports_test_like_name += 1
                    # Check if it's a local import
                    if mod in repo_python_modules:
                        nb_n_local_imports += 1
    except Exception:
        pass  # AST parsing failed, keep defaults
    
    nb_n_imports_total = len(imports)
    nb_imported_modules = sorted(imported_modules)
    
    # Flags
    bad_no_functions = (nb_n_function_defs == 0)
    bad_no_classes = (nb_n_class_defs == 0)
    bad_no_local_imports = (nb_n_local_imports == 0)
    bad_no_testing_imports = not (nb_imports_testing_module > 0 or nb_imports_test_like_name > 0)
    
    return {
        "nb_n_function_defs": nb_n_function_defs,
        "nb_n_class_defs": nb_n_class_defs,
        "nb_n_loops": nb_n_loops,
        "nb_n_conditionals": nb_n_conditionals,
        "nb_n_imports_total": nb_n_imports_total,
        "nb_imported_modules": nb_imported_modules,
        "nb_n_local_imports": nb_n_local_imports,
        "nb_imports_testing_module": nb_imports_testing_module,
        "nb_imports_test_like_name": nb_imports_test_like_name,
        "bad_no_functions": bad_no_functions,
        "bad_no_classes": bad_no_classes,
        "bad_no_local_imports": bad_no_local_imports,
        "bad_no_testing_imports": bad_no_testing_imports,
    }


def analyze_notebook_exec_order(nb_json: dict) -> Dict:
    """
    Analyzes execution order from execution_count: unambiguous order, out-of-order, skips.
    Returns metrics and bad_* flags.
    """
    nb = nbformat.from_dict(nb_json)
    cells = nb.cells or []
    
    code_cells = [c for c in cells if c.get("cell_type") == "code"]
    exec_counts = [c.get("execution_count") for c in code_cells]
    exec_counts_clean = [e for e in exec_counts if isinstance(e, int)]
    
    nb_n_executed_code_cells = len(exec_counts_clean)
    nb_n_non_executed_code_cells = len(code_cells) - nb_n_executed_code_cells
    nb_has_executed_cells = (nb_n_executed_code_cells > 0)
    
    # Check for cells with "*" (executing state)
    has_star = any(
        isinstance(c.get("execution_count"), str) and "*" in str(c.get("execution_count"))
        for c in code_cells
    )
    
    # Unambiguous order: all integers, no repeats, no "*"
    unique_counts = set(exec_counts_clean)
    nb_has_unambiguous_exec_order = (
        len(exec_counts_clean) == len(unique_counts) and
        len(exec_counts_clean) == len([e for e in exec_counts if e is not None]) and
        not has_star
    )
    
    # Out-of-order: check if any count is less than previous in cell order
    nb_has_out_of_order_exec = False
    if exec_counts_clean:
        ordered = sorted(exec_counts_clean)
        prev = None
        for e in exec_counts_clean:
            if prev is not None and e < prev:
                nb_has_out_of_order_exec = True
                break
            prev = e
    
    # Skips: gaps > 1 in sorted execution counts
    nb_n_exec_skips = 0
    skip_sizes = []
    if len(exec_counts_clean) > 1:
        ordered = sorted(exec_counts_clean)
        for i in range(1, len(ordered)):
            gap = ordered[i] - ordered[i-1]
            if gap > 1:
                nb_n_exec_skips += (gap - 1)
                skip_sizes.append(gap - 1)
    
    nb_avg_exec_skip_size = (sum(skip_sizes) / len(skip_sizes)) if skip_sizes else 0.0
    
    # Distribution by thirds
    third_size = max(1, len(code_cells) // 3)
    begin_cells = code_cells[:third_size]
    middle_cells = code_cells[third_size:2*third_size]
    end_cells = code_cells[2*third_size:]
    
    begin_executed = sum(1 for c in begin_cells if isinstance(c.get("execution_count"), int))
    middle_executed = sum(1 for c in middle_cells if isinstance(c.get("execution_count"), int))
    end_executed = sum(1 for c in end_cells if isinstance(c.get("execution_count"), int))
    
    nb_executed_at_begin_ratio = (begin_executed / len(begin_cells) * 100.0) if begin_cells else 0.0
    nb_executed_at_middle_ratio = (middle_executed / len(middle_cells) * 100.0) if middle_cells else 0.0
    nb_executed_at_end_ratio = (end_executed / len(end_cells) * 100.0) if end_cells else 0.0
    
    # Check for non-executed code in middle
    bad_has_non_executed_code_in_middle = False
    if middle_cells:
        for i, c in enumerate(middle_cells):
            if not isinstance(c.get("execution_count"), int):
                # Check if there are executed cells before and after
                has_before = any(
                    isinstance(bc.get("execution_count"), int) for bc in begin_cells
                ) or any(
                    isinstance(mc.get("execution_count"), int) for mc in middle_cells[:i]
                )
                has_after = any(
                    isinstance(mc.get("execution_count"), int) for mc in middle_cells[i+1:]
                ) or any(
                    isinstance(ec.get("execution_count"), int) for ec in end_cells
                )
                if has_before and has_after:
                    bad_has_non_executed_code_in_middle = True
                    break
    
    # Flags
    bad_out_of_order_exec = nb_has_out_of_order_exec
    bad_has_exec_skips = (nb_n_exec_skips > 0)
    
    return {
        "nb_n_executed_code_cells": nb_n_executed_code_cells,
        "nb_n_non_executed_code_cells": nb_n_non_executed_code_cells,
        "nb_has_executed_cells": nb_has_executed_cells,
        "nb_has_unambiguous_exec_order": nb_has_unambiguous_exec_order,
        "nb_has_out_of_order_exec": nb_has_out_of_order_exec,
        "nb_n_exec_skips": nb_n_exec_skips,
        "nb_avg_exec_skip_size": nb_avg_exec_skip_size,
        "nb_executed_at_begin_ratio": nb_executed_at_begin_ratio,
        "nb_executed_at_middle_ratio": nb_executed_at_middle_ratio,
        "nb_executed_at_end_ratio": nb_executed_at_end_ratio,
        "bad_out_of_order_exec": bad_out_of_order_exec,
        "bad_has_exec_skips": bad_has_exec_skips,
        "bad_has_non_executed_code_in_middle": bad_has_non_executed_code_in_middle,
    }


def parse_notebook_metrics(nb_json: dict, file_path: str = "", repo_python_modules: Optional[List[str]] = None) -> Tuple[dict, dict]:
    """
    Extrai métricas de células, execução e outputs a partir do JSON do notebook.
    Usa as funções de análise separadas para estrutura, AST e ordem de execução.
    """
    nb = nbformat.from_dict(nb_json)
    cells = nb.cells or []
    n_code = sum(1 for c in cells if c.get("cell_type") == "code")
    n_markdown = sum(1 for c in cells if c.get("cell_type") == "markdown")
    n_raw = sum(1 for c in cells if c.get("cell_type") == "raw")
    n_total = len(cells)
    
    # Use new analysis functions
    metrics_struct = analyze_notebook_structure(nb_json, file_path)
    metrics_ast = analyze_notebook_ast(nb_json, repo_python_modules)
    metrics_exec = analyze_notebook_exec_order(nb_json)


    # Outputs
    n_cells_with_output = 0
    outputs_text = False
    outputs_image = False
    outputs_html_js = False
    outputs_error = False
    outputs_formatted = False
    outputs_ext = False

    def inspect_output(out):
        nonlocal outputs_text, outputs_image, outputs_html_js, outputs_error, outputs_formatted, outputs_ext
        if out.get("output_type") == "error":
            outputs_error = True
        for mime in (out.get("data") or {}):
            if mime.startswith("text/"):
                outputs_text = True
            if any(mime.startswith(x) for x in ["image/png", "image/jpeg", "image/svg"]):
                outputs_image = True
            if mime in ("text/html", "application/javascript"):
                outputs_html_js = True
            if mime in ("text/latex", "text/markdown"):
                outputs_formatted = True
            # extensões comuns (widgets/plotly/bokeh)
            if any(mime.startswith(x) for x in ["application/vnd.", "application/plotly", "application/vnd.bokeh"]):
                outputs_ext = True

    for c in cells:
        if c.get("cell_type") == "code":
            outs = c.get("outputs") or []
            if outs:
                n_cells_with_output += 1
                for out in outs:
                    if isinstance(out, dict):
                        inspect_output(out)

    # Kernel / linguagem / versão
    kernel = (nb.metadata.get("kernelspec") or {}).get("name")
    lang = (nb.metadata.get("language_info") or {}).get("name")
    raw_pyver = (nb.metadata.get("language_info") or {}).get("version")
    # Normaliza versão para major.minor (ex.: "3.11.9" -> "3.11")
    pyver = None
    if raw_pyver:
        try:
            parts = str(raw_pyver).split(".")
            if len(parts) >= 2:
                pyver = f"{int(parts[0])}.{int(parts[1])}"
            else:
                pyver = str(raw_pyver)
        except Exception:
            pyver = str(raw_pyver)


    # Derived execution aggregates for summary stats
    exec_counts = [c.get("execution_count") for c in cells if c.get("cell_type") == "code"]
    exec_counts_clean = [e for e in exec_counts if isinstance(e, int)]
    n_code_executed = metrics_exec["nb_n_executed_code_cells"]
    percent_code_executed = (n_code_executed / n_code * 100.0) if n_code > 0 else 0.0
    max_exec = max(exec_counts_clean) if exec_counts_clean else 0
    

    triple_bq_in_code = False
    has_abs_data_path = False
    for c in cells:
        if c.get("cell_type") == "markdown":
            src = as_text(c.get("source"))


        elif c.get("cell_type") == "code":
            src = as_text(c.get("source"))
            if TRIPLE_BACKTICKS_IN_CODE.search(src):
                triple_bq_in_code = True
            if ABS_PATH_PATTERN.search(src):
                has_abs_data_path = True

    # Top imports (from AST metrics)
    from collections import Counter
    imported_modules_list = metrics_ast.get("nb_imported_modules") or []
    imp_counts = Counter(imported_modules_list)
    top_imports = imp_counts.most_common(10)
    top_imports_json = json.dumps(top_imports, ensure_ascii=False)
    nb_imported_modules_json = json.dumps(sorted(set(imported_modules_list)), ensure_ascii=False)

    # Merge all metrics
    result = {
        # Notebook-level structural and execution metrics
        "kernel_name": kernel,
        "language": lang,
        "python_version_declared": pyver,
        "n_cells_total": n_total,
        "n_code": n_code,
        "n_markdown": n_markdown,
        "n_raw": n_raw,
        "n_code_executed": n_code_executed,
        "percent_code_executed": percent_code_executed,
        "max_execution_count": max_exec,
        "n_cells_with_output": n_cells_with_output,
        "outputs_text": outputs_text,
        "outputs_image": outputs_image,
        "outputs_html_js": outputs_html_js,
        "outputs_error": outputs_error,
        "outputs_formatted": outputs_formatted,
        "outputs_ext": outputs_ext,
        "imports_total": metrics_ast.get("nb_n_imports_total", 0),
        "top_imports_json": top_imports_json,
        "has_local_imports": (metrics_ast.get("nb_n_local_imports", 0) > 0),
        "defines_function": (metrics_ast.get("nb_n_function_defs", 0) > 0),
        "defines_class": (metrics_ast.get("nb_n_class_defs", 0) > 0),
        "has_control_flow": (metrics_ast.get("nb_n_loops", 0) > 0 or metrics_ast.get("nb_n_conditionals", 0) > 0),
        "uses_testing_module": (metrics_ast.get("nb_imports_testing_module", 0) > 0 or metrics_ast.get("nb_imports_test_like_name", 0) > 0),
        "triple_backticks_in_code": triple_bq_in_code,
        "has_abs_data_path": has_abs_data_path,
        "nb_imported_modules": nb_imported_modules_json,
    }
    
    # Add new metrics from analysis functions
    result.update(metrics_struct)
    # avoid overriding JSONified nb_imported_modules
    result.update({k: v for k, v in metrics_ast.items() if k != "nb_imported_modules"})
    result.update(metrics_exec)
    
    return result, {}

def detect_repo_features(session: requests.Session, owner: str, repo: str, default_branch: str) -> Dict[str, bool]:
    """Inspeciona a árvore do repositório e infere features de ambiente/CI/dados/etc."""
    feats = {
        "repo_has_requirements_txt": False,
        "repo_has_pipfile": False,
        "repo_has_pyproject_toml": False,
        "repo_has_setup_py": False,
        "repo_has_environment_yml": False,
        "repo_has_lockfile": False,
        "repo_has_dockerfile": False,
        "repo_has_binder_config": False,
        "repo_has_ci_workflow": False,
        "repo_has_tests_dir": False,
        "repo_has_data_dir": False,
        "repo_has_env_spec": False,
        "repo_python_modules": [],
        "repo_python_modules_count": 0,
        "repo_default_branch_sha": "",
    }
    try:
        repo_info = gh_get_repo(session, owner, repo)
        ref = repo_info.get("default_branch") or default_branch or "main"
        # Pega o SHA da árvore do default branch
        branch = session.get(f"{GITHUB_API}/repos/{owner}/{repo}/branches/{ref}")
        branch.raise_for_status()
        branch_json = branch.json()
        feats["repo_default_branch_sha"] = branch_json.get("commit", {}).get("sha", "") or ""
        tree_sha = branch_json["commit"]["commit"]["tree"]["sha"]
        tree = gh_get_tree(session, owner, repo, tree_sha)
        items = tree.get("tree", [])
        paths = [item.get("path","") for item in items]
        lower_paths = [p.lower() for p in paths]

        feats["repo_has_requirements_txt"] = any(p.endswith("requirements.txt") for p in lower_paths)
        feats["repo_has_pipfile"] = any(p.endswith("pipfile") for p in lower_paths)
        feats["repo_has_pyproject_toml"] = any(p.endswith("pyproject.toml") for p in lower_paths)
        feats["repo_has_setup_py"] = any(p.endswith("setup.py") for p in lower_paths)
        feats["repo_has_environment_yml"] = any(p.endswith("environment.yml") or p.endswith("environment.yaml") for p in lower_paths)
        feats["repo_has_lockfile"] = any(p.endswith("pipfile.lock") or p.endswith("poetry.lock") for p in lower_paths)
        feats["repo_has_dockerfile"] = any(os.path.basename(p).lower() == "dockerfile" for p in paths)
        feats["repo_has_binder_config"] = any(p.startswith("binder/") or p.endswith("runtime.txt") for p in lower_paths)
        feats["repo_has_ci_workflow"] = any(p.startswith(".github/workflows/") for p in lower_paths)
        feats["repo_has_tests_dir"] = any(p.startswith("tests/") or p.startswith("test/") for p in lower_paths)
        feats["repo_has_data_dir"] = any(p.startswith("data/") or p.startswith("datasets/") or p.startswith("input/") for p in lower_paths)
        feats["repo_has_env_spec"] = any([
            feats["repo_has_requirements_txt"],
            feats["repo_has_pipfile"],
            feats["repo_has_pyproject_toml"],
            feats["repo_has_setup_py"],
            feats["repo_has_environment_yml"]
        ])
        # Detect local python modules (heuristic)
        py_modules: set[str] = set()
        py_files_seen = 0
        for p in paths:
            lp = p.lower()
            if lp.endswith(".py"):
                base = os.path.splitext(os.path.basename(p))[0]
                if base and base != "__init__" and not base.startswith("_"):
                    py_modules.add(base)
                # Add package dir name for __init__.py
                if os.path.basename(lp) == "__init__.py":
                    pkg = os.path.basename(os.path.dirname(p))
                    if pkg and not pkg.startswith("_"):
                        py_modules.add(pkg)
                py_files_seen += 1
                if py_files_seen > 5000:
                    # limite para performance; mantém parciais
                    break
        feats["repo_python_modules"] = sorted(py_modules)
        feats["repo_python_modules_count"] = len(py_modules)
    except Exception:
        pass
    return feats

def decode_notebook_content(item_json: Dict, session: requests.Session) -> Optional[dict]:
    """Baixa o JSON bruto do notebook a partir do endpoint contents."""
    repo = item_json["repository"]
    owner = repo["owner"]["login"]
    name = repo["name"]
    path = item_json["path"]
    default_branch = repo.get("default_branch") or "main"
    try:
        contents = gh_get_contents(session, owner, name, path, ref=default_branch)
        if contents.get("encoding") == "base64":
            raw = base64.b64decode(contents["content"])
            nb_json = json.loads(raw.decode("utf-8", errors="replace"))
            return nb_json
        else:
            # às vezes é um diretório (não deveria com search/code), ou encoding diferente
            return None
    except Exception:
        return None

def safe_join_save_path(base_dir: str, owner: str, repo: str, sha8: str, filename: str) -> str:
    """Monta caminho seguro para salvar o ipynb."""
    owner = re.sub(r"[^A-Za-z0-9_.-]", "_", owner)
    repo = re.sub(r"[^A-Za-z0-9_.-]", "_", repo)
    filename = re.sub(r"[^A-Za-z0-9_.-]", "_", os.path.basename(filename))
    return os.path.join(base_dir, owner, repo, f"{sha8}_{filename}")


def _sanitize_component(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", value)


def download_full_repo_snapshot(
    session: requests.Session,
    owner: str,
    repo: str,
    ref: str,
    dest_root: str,
    commit_sha: Optional[str] = None
) -> Optional[Path]:
    """
    Faz download do zipball do repositório inteiro e extrai para dest_root/owner/repo/{sha_or_ref}
    Retorna o caminho do snapshot salvo ou None em caso de falha.
    """
    dest_root_path = Path(dest_root)
    dest_root_path.mkdir(parents=True, exist_ok=True)

    safe_owner = _sanitize_component(owner)
    safe_repo = _sanitize_component(repo)
    label = commit_sha or ref or "latest"
    label = _sanitize_component(label)[:40] or "latest"
    repo_dest = dest_root_path / safe_owner / safe_repo
    snapshot_dir = repo_dest / label

    if snapshot_dir.exists():
        return snapshot_dir

    url = f"{GITHUB_API}/repos/{owner}/{repo}/zipball/{ref}"
    resp = request_with_backoff(session, "GET", url, stream=True)
    resp.raise_for_status()

    fd, tmp_zip_path = tempfile.mkstemp(suffix=".zip")
    os.close(fd)
    try:
        with open(tmp_zip_path, "wb") as tmp:
            for chunk in resp.iter_content(chunk_size=8192):
                if chunk:
                    tmp.write(chunk)

        tmp_extract_dir = Path(tempfile.mkdtemp(prefix="repo_extract_"))
        try:
            with zipfile.ZipFile(tmp_zip_path) as zf:
                zf.extractall(tmp_extract_dir)

            members = list(tmp_extract_dir.iterdir())
            if len(members) == 1 and members[0].is_dir():
                extracted_root = members[0]
            else:
                extracted_root = tmp_extract_dir

            repo_dest.mkdir(parents=True, exist_ok=True)
            shutil.move(str(extracted_root), str(snapshot_dir))
        finally:
            shutil.rmtree(tmp_extract_dir, ignore_errors=True)
    finally:
        try:
            os.remove(tmp_zip_path)
        except OSError:
            pass

    return snapshot_dir

def collect(
    token: Optional[str],
    date_start: dt.date,
    date_end: dt.date,
    max_items: Optional[int],
    output_csv: str,
    only_python: bool = True,
    require_outputs: bool = False,
    save_notebooks_dir: Optional[str] = None,
    download_full_repo_dir: Optional[str] = None
) -> None:
    session = build_session(token)
    date_ranges = partition_date_range(session, date_start, date_end, max_count=900)
    fields = [
        "repo_full_name","repo_id","repo_default_branch","repo_default_branch_sha","repo_created_at","repo_stars",
        # stable identity fields
        "notebook_id","commit_sha","notebook_rel_path",
        # original file fields
        "file_path","file_sha","file_size","html_url","nb_ok_parse",
        "kernel_name","language","python_version_declared",
        "n_cells_total","n_code","n_markdown","n_raw",
        "n_code_executed","percent_code_executed","max_execution_count",
        "n_cells_with_output","outputs_text","outputs_image","outputs_html_js","outputs_error","outputs_formatted","outputs_ext",
        "imports_total","top_imports_json","has_local_imports","defines_function","defines_class","has_control_flow","uses_testing_module",
        # repo features (expanded)
        "repo_has_requirements_txt","repo_has_pipfile","repo_has_pyproject_toml","repo_has_setup_py","repo_has_environment_yml",
        "repo_has_lockfile","repo_has_dockerfile","repo_has_binder_config","repo_has_ci_workflow","repo_has_tests_dir","repo_has_data_dir","repo_has_env_spec",
        "repo_python_modules_count","repo_python_modules",
        # dependency presence summary (aliases of repo_has_*)
        "deps_requirements_txt","deps_setup_py","deps_pipfile","deps_any",
        # New structure metrics
        "nb_n_code_cells","nb_n_markdown_cells","nb_n_raw_cells","nb_markdown_ratio",
        "nb_markdown_at_begin_ratio","nb_markdown_at_middle_ratio","nb_markdown_at_end_ratio",
        "nb_filename","nb_title_len","nb_has_non_portable_char","nb_has_windows_invalid_char",
        "bad_no_markdown","bad_low_markdown_ratio","bad_untitled","bad_copy_name","bad_non_portable_name","bad_windows_invalid_char",
        # New AST metrics
        "nb_n_function_defs","nb_n_class_defs","nb_n_loops","nb_n_conditionals",
        "nb_n_imports_total","nb_imported_modules","nb_n_local_imports","nb_imports_testing_module","nb_imports_test_like_name",
        "bad_no_functions","bad_no_classes","bad_no_local_imports","bad_no_testing_imports",
        # New exec order metrics
        "nb_n_executed_code_cells","nb_n_non_executed_code_cells","nb_has_executed_cells",
        "nb_has_unambiguous_exec_order","nb_has_out_of_order_exec","nb_n_exec_skips","nb_avg_exec_skip_size",
        "nb_executed_at_begin_ratio","nb_executed_at_middle_ratio","nb_executed_at_end_ratio",
        "bad_out_of_order_exec","bad_has_exec_skips","bad_has_non_executed_code_in_middle",
        "triple_backticks_in_code","has_abs_data_path"
    ]
    # Garantir diretório de saída dos notebooks, se solicitado
    if save_notebooks_dir:
        os.makedirs(save_notebooks_dir, exist_ok=True)
    if download_full_repo_dir:
        os.makedirs(download_full_repo_dir, exist_ok=True)

    print("🔍 Iniciando coleta de notebooks...", flush=True)
    print(f"   Período: {date_start} até {date_end}", flush=True)
    print(f"   Meta: {max_items if max_items else 'ilimitado'} notebooks", flush=True)
    print(f"   Requer outputs: {require_outputs}", flush=True)
    print(f"   Output: {output_csv}\n", flush=True)
    
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        
        repo_count = 0
        notebooks_found_count = 0

        for repo in tqdm(iterate_repo_search(session, date_ranges, max_repos=None), desc="Varredura de repositórios"):
            owner = repo["owner"]["login"]
            name = repo["name"]
            full = repo["full_name"]
            default_branch = repo.get("default_branch") or "main"
            created_at = repo.get("created_at")
            stargazers_count = repo.get("stargazers_count", 0)
            repo_id = repo["id"]
            
            repo_count += 1
            print(f"[Repo {repo_count}] Processando: {full} ...", flush=True)

            ipynb_paths = list_ipynb_in_repo(session, owner, name, default_branch)
            if not ipynb_paths:
                print("  └─ Nenhum notebook encontrado", flush=True)
                continue
            
            print(f"  └─ Encontrados {len(ipynb_paths)} notebook(s), analisando...", flush=True)

            # Detecta features de repo uma vez
            repo_feats = detect_repo_features(session, owner, name, default_branch)
            default_branch_sha = repo_feats.get("repo_default_branch_sha", "")
            try:
                print(f"  └─ Módulos Python locais detectados: {repo_feats.get('repo_python_modules_count', 0)}", flush=True)
            except Exception:
                pass

            full_repo_snapshot = None
            if download_full_repo_dir:
                try:
                    full_repo_snapshot = download_full_repo_snapshot(
                        session,
                        owner,
                        name,
                        default_branch,
                        download_full_repo_dir,
                        commit_sha=default_branch_sha
                    )
                    if full_repo_snapshot:
                        print(f"  └─ Repositório completo salvo em: {full_repo_snapshot}", flush=True)
                except Exception as exc:
                    print(f"  └─ ⚠️ Falha ao baixar repositório completo: {exc}", flush=True)

            for file_path in ipynb_paths:
                if max_items is not None and max_items <= 0:
                    return

                try:
                    contents = gh_get_contents(session, owner, name, file_path, ref=default_branch)
                except Exception:
                    continue

                file_sha = contents.get("sha","")
                html_url = contents.get("html_url","")
                file_size = contents.get("size", 0)

                nb_json = None
                try:
                    if contents.get("encoding") == "base64" and "content" in contents:
                        raw = base64.b64decode(contents["content"])
                        nb_json = json.loads(raw.decode("utf-8", errors="replace"))
                    elif contents.get("download_url"):
                        r = request_with_backoff(session, "GET", contents["download_url"])
                        r.raise_for_status()
                        nb_json = r.json()
                except Exception:
                    nb_json = None

                # Sem JSON legível: registra fallback e segue
                if not nb_json:
                    # notebook_id based on snapshot (use file_sha) and path
                    commit_sha = file_sha or ""
                    notebook_rel_path = file_path
                    notebook_id = compute_notebook_id(full, commit_sha, notebook_rel_path)
                    row = {
                        "repo_full_name": full, "repo_id": repo_id, "repo_default_branch": default_branch,
                        "repo_default_branch_sha": default_branch_sha,
                        "repo_created_at": created_at, "repo_stars": stargazers_count,
                        "notebook_id": notebook_id, "commit_sha": commit_sha, "notebook_rel_path": notebook_rel_path,
                        "file_path": file_path, "file_sha": file_sha, "file_size": file_size,
                        "html_url": html_url, "nb_ok_parse": False,
                    }
                    for k in fields:
                        if k not in row:
                            row[k] = ""
                    w.writerow(row)
                    notebooks_found_count += 1
                    print(f"     ├─ ✗ {file_path}: falha no parse (total: {notebooks_found_count})", flush=True)
                    if max_items is not None:
                        max_items -= 1
                        if max_items <= 0:
                            print(f"\n✅ Meta de {notebooks_found_count} notebooks atingida!", flush=True)
                            return
                    continue

                # Filtra por linguagem
                lang = (nb_json.get("metadata", {}).get("language_info") or {}).get("name", "")
                if only_python and (not lang or "python" not in str(lang).lower()):
                    continue

                # Get repo Python modules for local import detection
                repo_python_modules = repo_feats.get("repo_python_modules") or []
                
                # Extrai métricas
                metrics, _ = parse_notebook_metrics(nb_json, file_path, repo_python_modules)

                # Se exigir outputs salvos (estado executado), filtra aqui
                if require_outputs:
                    if metrics.get("n_cells_with_output", 0) <= 0:
                        continue
                    # opcionalmente, exigir alguma execução numerada
                    if metrics.get("percent_code_executed", 0.0) <= 0.0 and metrics.get("max_execution_count", 0) <= 0:
                        continue

                # Salva o .ipynb bruto, se solicitado
                if save_notebooks_dir:
                    sha8 = (contents.get("sha","") or "")[:8] or "noSHA"
                    out_path = safe_join_save_path(save_notebooks_dir, owner, name, sha8, os.path.basename(file_path))
                    os.makedirs(os.path.dirname(out_path), exist_ok=True)
                    try:
                        with open(out_path, "w", encoding="utf-8") as fh:
                            json.dump(nb_json, fh, ensure_ascii=False)
                    except Exception:
                        # não impede a coleta; segue registrando no CSV
                        pass

                row = {
                    "repo_full_name": full,
                    "repo_id": repo_id,
                    "repo_default_branch": default_branch,
                    "repo_default_branch_sha": default_branch_sha,
                    "repo_created_at": created_at,
                    # identity fields
                    "notebook_id": compute_notebook_id(full, file_sha or "", file_path),
                    "commit_sha": file_sha or "",
                    "notebook_rel_path": file_path,
                    "repo_stars": stargazers_count,
                    "file_path": file_path,
                    "file_sha": file_sha,
                    "file_size": file_size,
                    "html_url": html_url,
                    "nb_ok_parse": True,

                    **metrics,

                    # repo features (expanded)
                    **{k: v for k, v in repo_feats.items() if k not in ("repo_python_modules",)},
                    "repo_python_modules": json.dumps(repo_feats.get("repo_python_modules", []), ensure_ascii=False),
                    # dependency summary aliases (keep alongside repo_has_* for quick filters)
                    "deps_requirements_txt": repo_feats.get("repo_has_requirements_txt", False),
                    "deps_setup_py": repo_feats.get("repo_has_setup_py", False),
                    "deps_pipfile": repo_feats.get("repo_has_pipfile", False),
                    "deps_any": any([
                        repo_feats.get("repo_has_requirements_txt", False),
                        repo_feats.get("repo_has_setup_py", False),
                        repo_feats.get("repo_has_pipfile", False),
                    ]),
                }
                for k in fields:
                    row.setdefault(k, "")
                w.writerow(row)
                notebooks_found_count += 1
                # Se chegou até aqui, parse foi OK (nb_ok_parse=True)
                print(f"     └─ ✓ OK: {file_path} (total: {notebooks_found_count})", flush=True)

                if max_items is not None:
                    max_items -= 1
                    if max_items <= 0:
                        print(f"\n✅ Meta de {notebooks_found_count} notebooks atingida!", flush=True)
                        return
    
    # Verificação obrigatória de unicidade de notebook_id
    try:
        dup_out_csv = os.path.join(os.path.dirname(output_csv) or ".", "duplicate_notebook_ids.csv")
        counts: Dict[str, int] = {}
        sample_rows: List[dict] = []
        with open(output_csv, newline="", encoding="utf-8") as fchk:
            rd = csv.DictReader(fchk)
            for r in rd:
                nid = r.get("notebook_id") or ""
                counts[nid] = counts.get(nid, 0) + 1
                if nid:
                    sample_rows.append(r)
        dup_ids = [nid for nid, c in counts.items() if nid and c > 1]
        if dup_ids:
            with open(dup_out_csv, "w", newline="", encoding="utf-8") as fd:
                if sample_rows:
                    wdup = csv.DictWriter(fd, fieldnames=sample_rows[0].keys())
                    wdup.writeheader()
                    for r in sample_rows:
                        if r.get("notebook_id") in dup_ids:
                            wdup.writerow(r)
            print(f"ERROR: Notebook ID duplicates detected: {len(dup_ids)} ids duplicados. Amostra: {dup_ids[:10]}", file=sys.stderr, flush=True)
            raise RuntimeError("Notebook ID uniqueness check failed. See duplicate_notebook_ids.csv for details.")
        else:
            print(f"Notebook ID uniqueness check passed: {sum(1 for _ in counts)} unique notebooks processed.", flush=True)
    except Exception as e:
        if isinstance(e, RuntimeError):
            raise
        print(f"ERROR: Failed to verify notebook_id uniqueness: {e}", file=sys.stderr, flush=True)
        raise

    print("\n" + "="*60, flush=True)
    print("✅ Coleta concluída!", flush=True)
    print(f"   Total de notebooks coletados: {notebooks_found_count}", flush=True)
    print(f"   Repositórios processados: {repo_count}", flush=True)
    print(f"   Arquivo salvo: {output_csv}", flush=True)
    print("="*60 + "\n", flush=True)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Coleta metadados de notebooks Jupyter no GitHub.")
    p.add_argument("--date-start", type=str, required=True, help="Data inicial (YYYY-MM-DD)")
    p.add_argument("--date-end", type=str, required=True, help="Data final (YYYY-MM-DD)")
    p.add_argument("--max-items", type=int, default=1000, help="Limite de notebooks a coletar (aprox.)")
    p.add_argument("--output", type=str, required=True, help="Caminho do CSV de saída")
    p.add_argument("--include-non-python", action="store_true", help="Não filtrar notebooks que não sejam Python")
    # NOVOS FLAGS
    p.add_argument("--require-outputs", action="store_true",
                   help="Apenas notebooks com outputs salvos (estado executado).")
    p.add_argument("--save-notebooks-dir", type=str, default=None,
                   help="Se definido, salva o .ipynb original decodificado em owner/repo/sha8_nome.ipynb.")
    p.add_argument("--download-full-repos", action="store_true",
                   help="Baixa um snapshot completo (zipball) de cada repositório processado.")
    p.add_argument("--full-repos-dir", type=str, default=None,
                   help="Diretório base para salvar os repositórios completos (default: ao lado do CSV de saída).")
    return p.parse_args(argv)

def main(argv=None):
    args = parse_args(argv)
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        print("ERRO: defina a variável de ambiente GITHUB_TOKEN com um token de acesso do GitHub.", file=sys.stderr)
        sys.exit(2)
    date_start = dt.date.fromisoformat(args.date_start)
    date_end = dt.date.fromisoformat(args.date_end)
    if date_end < date_start:
        print("ERRO: date-end não pode ser anterior a date-start.", file=sys.stderr)
        sys.exit(2)
    download_full_repo_dir = args.full_repos_dir
    if args.download_full_repos or download_full_repo_dir:
        if not download_full_repo_dir:
            base_dir = os.path.dirname(args.output) or "."
            download_full_repo_dir = os.path.join(base_dir, "repositorios_completos")
    collect(
        token=token,
        date_start=date_start,
        date_end=date_end,
        max_items=args.max_items,
        output_csv=args.output,
        only_python=not args.include_non_python,
        require_outputs=args.require_outputs,
        save_notebooks_dir=args.save_notebooks_dir,
        download_full_repo_dir=download_full_repo_dir
    )

if __name__ == "__main__":
    main()
