"""
collect_notebooks.py
--------------------
Coleta metadados de Jupyter Notebooks no GitHub para o estudo de reprodutibilidade,
seguindo e atualizando a metodologia de Pimentel et al. (2019).

Principais funcionalidades:
- Busca notebooks (.ipynb) por intervalo de datas, com divisão automática de janelas
  para respeitar o limite de 1000 resultados por consulta do GitHub Search API.
- Faz download do JSON do notebook (sem clonar repositório) e extrai métricas chave:
  * contagem de células por tipo, execução e outputs;
  * ordem de execução (ambígua vs. não-ambígua), skips e out-of-order;
  * presença de imports, funções, classes, e possíveis módulos de teste;
  * indicadores heurísticos de autoria/colagem por IA (comentários/padrões);
  * presença de arquivos de dependências no repositório (requirements.txt, setup.py, Pipfile).
- Salva um CSV tabular pronto para análises.

Uso (exemplo):
    GITHUB_TOKEN=ghp_xxx \
    python scripts/collect_notebooks.py \
        --date-start 2025-01-01 --date-end 2025-09-28 \
        --max-items 2000 \
        --output data/notebooks_metadata.csv

Requisitos: ver requirements.txt
"""
from __future__ import annotations

import argparse
import base64
import csv
import dataclasses
from dataclasses import dataclass, asdict
import datetime as dt
import json
import os
import random
import re
import sys
from typing import Dict, Iterable, List, Optional, Tuple

import requests
from requests.adapters import HTTPAdapter, Retry
from dateutil.parser import isoparse
from tqdm import tqdm
import nbformat
import ast

GITHUB_API = "https://api.github.com"

# ===============================
# Utilidades HTTP e GitHub API
# ===============================

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
    resp = session.get(url, params={"q": q, "page": page, "per_page": per_page})
    resp.raise_for_status()
    return resp.json()

def gh_get_repo(session: requests.Session, owner: str, repo: str) -> Dict:
    url = f"{GITHUB_API}/repos/{owner}/{repo}"
    r = session.get(url)
    r.raise_for_status()
    return r.json()

def gh_get_contents(session: requests.Session, owner: str, repo: str, path: str, ref: Optional[str]=None) -> Dict:
    url = f"{GITHUB_API}/repos/{owner}/{repo}/contents/{path}"
    params = {"ref": ref} if ref else None
    r = session.get(url, params=params)
    r.raise_for_status()
    return r.json()

def gh_get_tree(session: requests.Session, owner: str, repo: str, sha: str) -> Dict:
    url = f"{GITHUB_API}/repos/{owner}/{repo}/git/trees/{sha}"
    r = session.get(url, params={"recursive": 1})
    r.raise_for_status()
    return r.json()

def gh_search_repos(session: requests.Session, q: str, page: int = 1, per_page: int = 100) -> Dict:
    url = f"{GITHUB_API}/search/repositories"
    resp = session.get(url, params={"q": q, "page": page, "per_page": per_page, "sort": "updated", "order": "desc"})
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


# ===============================
# Particionamento por datas
# ===============================

def partition_date_range(session: requests.Session, start: dt.date, end: dt.date, max_count: int = 1000) -> List[Tuple[dt.date, dt.date]]:
    """
    Divide [start, end] em janelas menores, garantindo < max_count resultados por consulta.
    Estratégia: busca iterativa por dias; quando ainda exceder, quebra pela metade.
    """
    ranges = [(start, end)]
    final = []
    while ranges:
        a, b = ranges.pop()
        q = f'extension:ipynb created:{a.isoformat()}..{b.isoformat()}'
        try:
            j = gh_search_code(session, q, page=1, per_page=1)  # apenas conta
            total = min(j.get("total_count", 0), 1000_000)  # proteção
        except requests.HTTPError as e:
            # Se algo falhar (ex.: permissões), quebra em dois e continua
            if (b - a).days <= 0:
                continue
            mid = a + (b - a)//2
            ranges.append((a, mid))
            ranges.append((mid + dt.timedelta(days=1), b))
            continue

        if total >= max_count:  # dividir mais
            if (b - a).days <= 0:
                # dia único ainda com >= max_count: não há o que fazer; aceitar risco de truncamento
                final.append((a, b))
            else:
                mid = a + (b - a)//2
                ranges.append((a, mid))
                ranges.append((mid + dt.timedelta(days=1), b))
        else:
            final.append((a, b))
    # Ordena cronologicamente
    return sorted(final, key=lambda t: t[0])

def iterate_search_results(session: requests.Session, date_ranges: List[Tuple[dt.date, dt.date]], sample_limit: Optional[int]=None) -> Iterable[Dict]:
    count = 0
    for a, b in date_ranges:
        q = f'extension:ipynb created:{a.isoformat()}..{b.isoformat()}'
        page = 1
        while True:
            data = gh_search_code(session, q, page=page, per_page=100)
            items = data.get("items", [])
            if not items:
                break
            for it in items:
                yield it
                count += 1
                if sample_limit and count >= sample_limit:
                    return
            page += 1

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
            if any(mime in x for x in ["text/html", "application/javascript"]):
                outputs_html_js = True
            if any(mime in x for x in ["text/latex", "text/markdown"]):
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
            src = c.get("source") or ""
            # Heurísticas IA
            for pat in AI_MARKER_PATTERNS:
                if pat.search(src):
                    ai_marker_found = True
                    break
        elif c.get("cell_type") == "code":
            src = c.get("source") or ""
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
                        # Heurística simples de local import: relativo não se aplica aqui; será via ImportFrom com level>0
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
        sha = repo_info.get("default_branch") or default_branch
        ref = repo_info.get("default_branch") or "main"
        # Pega o SH A da árvore do default branch
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

def collect(
    token: Optional[str],
    date_start: dt.date,
    date_end: dt.date,
    max_items: Optional[int],
    output_csv: str,
    only_python: bool = True
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
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()

        for item in tqdm(iterate_search_results(session, date_ranges, sample_limit=max_items), desc="Coletando notebooks"):
            repo = item["repository"]
            owner = repo["owner"]["login"]
            name = repo["name"]
            full = repo["full_name"]
            default_branch = repo.get("default_branch") or "main"
            created_at = repo.get("created_at")
            stargazers_count = repo.get("stargazers_count", 0)
            file_path = item["path"]
            file_sha = item.get("sha","")
            html_url = item.get("html_url","")
            file_size = item.get("score", 0)  # não há size no search/code; manter 0

            # baixar notebook
            nb_json = decode_notebook_content(item, session)
            if not nb_json:
                row = {
                    "repo_full_name": full, "repo_id": repo["id"], "repo_default_branch": default_branch,
                    "repo_created_at": created_at, "repo_stars": stargazers_count,
                    "file_path": file_path, "file_sha": file_sha, "file_size": file_size,
                    "html_url": html_url, "nb_ok_parse": False,
                }
                for k in fields:
                    if k not in row:
                        row[k] = ""
                w.writerow(row)
                continue

            # filtrar por linguagem Python (opcional)
            lang = (nb_json.get("metadata", {}).get("language_info") or {}).get("name", "")
            if only_python and (not lang or "python" not in str(lang).lower()):
                continue

            # métricas
            metrics, _ = parse_notebook_metrics(nb_json)

            # deps no repo
            has_req, has_setup, has_pipfile = detect_deps_in_repo(session, owner, name, default_branch)

            row = {
                "repo_full_name": full,
                "repo_id": repo["id"],
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
            # garantir todas as chaves
            for k in fields:
                row.setdefault(k, "")
            w.writerow(row)

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Coleta metadados de notebooks Jupyter no GitHub.")
    p.add_argument("--date-start", type=str, required=True, help="Data inicial (YYYY-MM-DD)")
    p.add_argument("--date-end", type=str, required=True, help="Data final (YYYY-MM-DD)")
    p.add_argument("--max-items", type=int, default=1000, help="Limite de notebooks a coletar (aprox.)")
    p.add_argument("--output", type=str, required=True, help="Caminho do CSV de saída")
    p.add_argument("--include-non-python", action="store_true", help="Não filtrar notebooks que não sejam Python")
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
        only_python=not args.include_non_python
    )

if __name__ == "__main__":
    main()
