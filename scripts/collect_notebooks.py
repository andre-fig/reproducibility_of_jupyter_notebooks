from __future__ import annotations

import argparse
import base64
import csv
import time
from dataclasses import dataclass
import datetime as dt
import json
import os
import re
import sys
from typing import Dict, Iterable, List, Optional, Tuple
import random

import requests
from requests.adapters import HTTPAdapter, Retry
from tqdm import tqdm
import nbformat
import ast

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
        except requests.HTTPError as e:
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

TEST_MODULE_HINTS = re.compile(r"(?:^|[.\-_])(test|tests|mock|mocks)(?:$|[.\-_])", re.IGNORECASE)
KNOWN_TEST_MODULES = {
    "unittest", "pytest", "nose", "hypothesis", "behave", "mox", "subunit", "twisted.trial"
}

AI_MARKER_PATTERNS = [
    re.compile(r"\bChatGPT\b", re.IGNORECASE),
    re.compile(r"\bCopilot\b", re.IGNORECASE),
    re.compile(r"\bClaude\b", re.IGNORECASE),
    re.compile(r"\bCodeWhisperer\b", re.IGNORECASE),
    re.compile(r"\bCursor\b", re.IGNORECASE),
    re.compile(r"As an AI language model", re.IGNORECASE),
]

ABS_PATH_PATTERN = re.compile(r"(^\/[^ \n\r]+)|([A-Za-z]:\\[^ \n\r]+)")
TRIPLE_BACKTICKS_IN_CODE = re.compile(r"```(?:python|py)?", re.IGNORECASE)

@dataclass
class NotebookMetrics:
    repo_full_name: str
    repo_id: int
    repo_default_branch: str
    repo_created_at: str
    repo_stars: int
    file_path: str
    file_sha: str
    file_size: int
    html_url: str
    nb_ok_parse: bool

    kernel_name: Optional[str]
    language: Optional[str]
    python_version_declared: Optional[str]

    n_cells_total: int
    n_code: int
    n_markdown: int
    n_raw: int

    n_code_executed: int
    percent_code_executed: float
    has_unambiguous_order: bool
    out_of_order: bool
    n_skips_total: int
    n_skips_middle: int
    max_execution_count: int

    n_cells_with_output: int
    outputs_text: bool
    outputs_image: bool
    outputs_html_js: bool
    outputs_error: bool
    outputs_formatted: bool
    outputs_ext: bool

    imports_total: int
    top_imports_json: str
    has_local_imports: bool
    defines_function: bool
    defines_class: bool
    has_control_flow: bool
    uses_testing_module: bool

    deps_requirements_txt: bool
    deps_setup_py: bool
    deps_pipfile: bool
    deps_any: bool

    ai_marker_found: bool
    triple_backticks_in_code: bool
    has_abs_data_path: bool

def parse_notebook_metrics(nb_json: dict) -> Tuple[dict, dict]:
    """Extrai métricas de células, execução e outputs a partir do JSON do notebook."""
    nb = nbformat.from_dict(nb_json)
    cells = nb.cells or []
    n_code = sum(1 for c in cells if c.get("cell_type") == "code")
    n_markdown = sum(1 for c in cells if c.get("cell_type") == "markdown")
    n_raw = sum(1 for c in cells if c.get("cell_type") == "raw")
    n_total = len(cells)

    # Execução / ordem
    exec_counts = [c.get("execution_count") for c in cells if c.get("cell_type") == "code"]
    exec_counts_clean = [e for e in exec_counts if isinstance(e, int)]
    n_code_executed = sum(1 for e in exec_counts if isinstance(e, int))
    percent_code_executed = (n_code_executed / n_code * 100.0) if n_code > 0 else 0.0
    has_unambiguous = False
    out_of_order = False
    n_skips_total = 0
    n_skips_middle = 0
    max_exec = max(exec_counts_clean) if exec_counts_clean else 0
    if exec_counts_clean:
        # Unambiguous: todos inteiros, sem repetidos e nenhum "*"
        has_unambiguous = (len(exec_counts_clean) == len([e for e in exec_counts if e is not None]))
        # Out-of-order e skips
        ordered = sorted(exec_counts_clean)
        prev = None
        for idx, e in enumerate(exec_counts_clean):
            if prev is not None and e < prev:
                out_of_order = True
            prev = e
        # Skips: diferenças > 1 entre vizinhos em ordem de execução
        for i in range(1, len(ordered)):
            gap = ordered[i] - ordered[i-1]
            if gap > 1:
                n_skips_total += (gap - 1)
                # se o skip não inclui o primeiro valor
                if ordered[i-1] != min(ordered):
                    n_skips_middle += (gap - 1)

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
    pyver = (nb.metadata.get("language_info") or {}).get("version")

    # AST: imports, funções, classes, controle de fluxo, testes
    imports = []
    has_local_imports = False
    defines_function = False
    defines_class = False
    has_control_flow = False
    uses_testing = False
    ai_marker_found = False
    triple_bq_in_code = False
    has_abs_data_path = False

    for c in cells:
        if c.get("cell_type") == "markdown":
            src = as_text(c.get("source"))
            # Heurísticas IA
            for pat in AI_MARKER_PATTERNS:
                if pat.search(src):
                    ai_marker_found = True
                    break
        elif c.get("cell_type") == "code":
            src = as_text(c.get("source"))
            if TRIPLE_BACKTICKS_IN_CODE.search(src):
                triple_bq_in_code = True
            if ABS_PATH_PATTERN.search(src):
                has_abs_data_path = True
            try:
                tree = ast.parse(src)
            except Exception:
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for n in node.names:
                        mod = n.name.split(".")[0]
                        imports.append(mod)
                        if TEST_MODULE_HINTS.search(mod) or mod in KNOWN_TEST_MODULES:
                            uses_testing = True
                elif isinstance(node, ast.ImportFrom):
                    mod = (node.module or "").split(".")[0]
                    if node.level and node.level > 0:
                        has_local_imports = True
                    if mod:
                        imports.append(mod)
                        if TEST_MODULE_HINTS.search(mod) or mod in KNOWN_TEST_MODULES:
                            uses_testing = True
                elif isinstance(node, ast.FunctionDef):
                    defines_function = True
                elif isinstance(node, ast.ClassDef):
                    defines_class = True
                elif isinstance(node, (ast.For, ast.While, ast.If, ast.Try, ast.With)):
                    has_control_flow = True

    # top imports: top-10
    from collections import Counter
    imp_counts = Counter(imports)
    top_imports = imp_counts.most_common(10)
    top_imports_json = json.dumps(top_imports, ensure_ascii=False)

    return {
        "kernel_name": kernel,
        "language": lang,
        "python_version_declared": pyver,
        "n_cells_total": n_total,
        "n_code": n_code,
        "n_markdown": n_markdown,
        "n_raw": n_raw,
        "n_code_executed": n_code_executed,
        "percent_code_executed": percent_code_executed,
        "has_unambiguous_order": has_unambiguous,
        "out_of_order": out_of_order,
        "n_skips_total": n_skips_total,
        "n_skips_middle": n_skips_middle,
        "max_execution_count": max_exec,
        "n_cells_with_output": n_cells_with_output,
        "outputs_text": outputs_text,
        "outputs_image": outputs_image,
        "outputs_html_js": outputs_html_js,
        "outputs_error": outputs_error,
        "outputs_formatted": outputs_formatted,
        "outputs_ext": outputs_ext,
        "imports_total": sum(imp_counts.values()),
        "top_imports_json": top_imports_json,
        "has_local_imports": has_local_imports,
        "defines_function": defines_function,
        "defines_class": defines_class,
        "has_control_flow": has_control_flow,
        "uses_testing_module": uses_testing,
        "ai_marker_found": ai_marker_found,
        "triple_backticks_in_code": triple_bq_in_code,
        "has_abs_data_path": has_abs_data_path,
    }, {}

def detect_deps_in_repo(session: requests.Session, owner: str, repo: str, default_branch: str) -> Tuple[bool, bool, bool]:
    """Procura por requirements.txt, setup.py e Pipfile em qualquer pasta do repo."""
    try:
        repo_info = gh_get_repo(session, owner, repo)
        ref = repo_info.get("default_branch") or default_branch or "main"
        # Pega o SHA da árvore do default branch
        branch = session.get(f"{GITHUB_API}/repos/{owner}/{repo}/branches/{ref}")
        branch.raise_for_status()
        tree_sha = branch.json()["commit"]["commit"]["tree"]["sha"]
        tree = gh_get_tree(session, owner, repo, tree_sha)
        paths = [item.get("path","").lower() for item in tree.get("tree", [])]
        has_req = any(p.endswith("requirements.txt") for p in paths)
        has_setup = any(p.endswith("setup.py") for p in paths)
        has_pipfile = any(p.endswith("pipfile") for p in paths)
        return has_req, has_setup, has_pipfile
    except Exception:
        return False, False, False

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

def collect(
    token: Optional[str],
    date_start: dt.date,
    date_end: dt.date,
    max_items: Optional[int],
    output_csv: str,
    only_python: bool = True,
    require_outputs: bool = False,
    save_notebooks_dir: Optional[str] = None
) -> None:
    session = build_session(token)
    date_ranges = partition_date_range(session, date_start, date_end, max_count=900)
    fields = [
        "repo_full_name","repo_id","repo_default_branch","repo_created_at","repo_stars",
        "file_path","file_sha","file_size","html_url","nb_ok_parse",
        "kernel_name","language","python_version_declared",
        "n_cells_total","n_code","n_markdown","n_raw",
        "n_code_executed","percent_code_executed","has_unambiguous_order","out_of_order",
        "n_skips_total","n_skips_middle","max_execution_count",
        "n_cells_with_output","outputs_text","outputs_image","outputs_html_js","outputs_error","outputs_formatted","outputs_ext",
        "imports_total","top_imports_json","has_local_imports","defines_function","defines_class","has_control_flow","uses_testing_module",
        "deps_requirements_txt","deps_setup_py","deps_pipfile","deps_any",
        "ai_marker_found","triple_backticks_in_code","has_abs_data_path"
    ]
    # Garantir diretório de saída dos notebooks, se solicitado
    if save_notebooks_dir:
        os.makedirs(save_notebooks_dir, exist_ok=True)

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()

        for repo in tqdm(iterate_repo_search(session, date_ranges, max_repos=None), desc="Varredura de repositórios"):
            owner = repo["owner"]["login"]
            name = repo["name"]
            full = repo["full_name"]
            default_branch = repo.get("default_branch") or "main"
            created_at = repo.get("created_at")
            stargazers_count = repo.get("stargazers_count", 0)
            repo_id = repo["id"]

            ipynb_paths = list_ipynb_in_repo(session, owner, name, default_branch)
            if not ipynb_paths:
                continue

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
                    row = {
                        "repo_full_name": full, "repo_id": repo_id, "repo_default_branch": default_branch,
                        "repo_created_at": created_at, "repo_stars": stargazers_count,
                        "file_path": file_path, "file_sha": file_sha, "file_size": file_size,
                        "html_url": html_url, "nb_ok_parse": False,
                    }
                    for k in fields:
                        if k not in row:
                            row[k] = ""
                    w.writerow(row)
                    if max_items is not None:
                        max_items -= 1
                        if max_items <= 0:
                            return
                    continue

                # Filtra por linguagem
                lang = (nb_json.get("metadata", {}).get("language_info") or {}).get("name", "")
                if only_python and (not lang or "python" not in str(lang).lower()):
                    continue

                # Extrai métricas
                metrics, _ = parse_notebook_metrics(nb_json)

                # Se exigir outputs salvos (estado executado), filtra aqui
                if require_outputs:
                    if metrics.get("n_cells_with_output", 0) <= 0:
                        continue
                    # opcionalmente, exigir alguma execução numerada
                    if metrics.get("percent_code_executed", 0.0) <= 0.0 and metrics.get("max_execution_count", 0) <= 0:
                        continue

                # Detecta dependências no repo
                has_req, has_setup, has_pipfile = detect_deps_in_repo(session, owner, name, default_branch)

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
                    "repo_created_at": created_at,
                    "repo_stars": stargazers_count,
                    "file_path": file_path,
                    "file_sha": file_sha,
                    "file_size": file_size,
                    "html_url": html_url,
                    "nb_ok_parse": True,

                    **metrics,

                    "deps_requirements_txt": has_req,
                    "deps_setup_py": has_setup,
                    "deps_pipfile": has_pipfile,
                    "deps_any": has_req or has_setup or has_pipfile,
                }
                for k in fields:
                    row.setdefault(k, "")
                w.writerow(row)

                if max_items is not None:
                    max_items -= 1
                    if max_items <= 0:
                        return


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
    collect(
        token=token,
        date_start=date_start,
        date_end=date_end,
        max_items=args.max_items,
        output_csv=args.output,
        only_python=not args.include_non_python,
        require_outputs=args.require_outputs,
        save_notebooks_dir=args.save_notebooks_dir
    )

if __name__ == "__main__":
    main()
