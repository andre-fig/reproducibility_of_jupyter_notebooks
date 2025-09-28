#!/usr/bin/env python3
"""
Executa notebooks listados no CSV de coleta com nbclient, sob timeout real,
headless (sem janelas/GUI) e isolamento por venv por repositório.

- Preenche 'error' com a exceção real (ModuleNotFoundError, ImportError, etc.)
- Mantém logs e tails de stdout/stderr quando falha.

Políticas de deps:
- strict: tenta instalar deps do repo (requirements.txt / Pipfile / setup.py/pyproject).
- relaxed: instala baseline mínima (nbclient, ipykernel, nbformat).

Observações:
- Clona o repositório (shallow).
- Executa no caminho relativo do CSV.
- Headless: evita abrir janelas (pygame, Qt, matplotlib interativo etc.).
"""

from __future__ import annotations
import argparse, csv, logging, os, shutil, subprocess, sys, tempfile, time, json, re
from pathlib import Path
from typing import Dict, Tuple

import nbformat
try:
    from nbclient import NotebookClient
    try:
        from nbclient.exceptions import CellExecutionError
    except Exception:
        class CellExecutionError(Exception): ...
except Exception:
    print("ERRO: nbclient não está instalado. Rode: pip install nbclient", file=sys.stderr)
    raise

try:
    from jupyter_client.kernelspec import KernelSpecManager
except Exception:
    KernelSpecManager = None

GIT_BASE = "https://github.com"
RESULT_PREFIX = "NBEXEC_RESULT:"

# -------------------------
# Logging
# -------------------------
def setup_logging(log_file: Path | None):
    logger = logging.getLogger("nbexec")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    if logger.handlers:
        for h in list(logger.handlers):
            logger.removeHandler(h)
    h_console = logging.StreamHandler(sys.stdout)
    h_console.setFormatter(fmt)
    logger.addHandler(h_console)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        h_file = logging.FileHandler(str(log_file), encoding="utf-8")
        h_file.setFormatter(fmt)
        logger.addHandler(h_file)
    return logger

# -------------------------
# Shell util
# -------------------------
def sh(cmd, cwd=None, env=None, check=True, timeout=None) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd, cwd=cwd, env=env, text=True,
        capture_output=True, check=check, timeout=timeout
    )

# -------------------------
# Ambientes / deps
# -------------------------
def venv_bin(env_dir: Path, name: str) -> str:
    if sys.platform.startswith("win"):
        return str(env_dir / "Scripts" / name)
    return str(env_dir / "bin" / name)

def make_env(env_dir: Path, policy: str, logger: logging.Logger) -> tuple[str, str]:
    env_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Criando venv: {env_dir}")
    sh([sys.executable, "-m", "venv", str(env_dir)])
    pip = venv_bin(env_dir, "pip")
    pybin = venv_bin(env_dir, "python")
    base = ["pip", "setuptools", "wheel", "nbclient", "ipykernel", "nbformat"]
    logger.info("Instalando baseline no venv (relaxed baseline)")
    sh([pip, "install", "--upgrade"] + base, check=True)
    return pip, pybin

def install_reqs(pip: str, repo_dir: Path, logger: logging.Logger) -> bool:
    reqs = list(repo_dir.rglob("requirements.txt"))
    if reqs:
        pth = str(reqs[0])
        logger.info(f"Instalando requirements.txt: {pth}")
        try:
            sh([pip, "install", "-r", pth], check=True, timeout=600)
            return True
        except Exception as e:
            logger.warning(f"Falha ao instalar requirements.txt: {e}")
    pipfile = list(repo_dir.rglob("Pipfile"))
    if pipfile:
        logger.info("Encontrado Pipfile; tentando pipenv (se disponível)")
        try:
            sh(["pipenv", "install", "--system", "--deploy"], cwd=repo_dir, check=True, timeout=600)
            return True
        except Exception as e:
            logger.warning(f"Falha ao instalar via pipenv: {e}")
    setup_py = list(repo_dir.rglob("setup.py"))
    pyproject = list(repo_dir.rglob("pyproject.toml"))
    if setup_py or pyproject:
        logger.info("Instalando pacote local (setup.py/pyproject)")
        try:
            sh([pip, "install", "-e", "."], cwd=repo_dir, check=True, timeout=600)
            return True
        except Exception as e:
            logger.warning(f"Falha ao instalar pacote local: {e}")
    return False

# -------------------------
# Git
# -------------------------
def clone_repo(full_name: str, dest: Path, branch: str, logger: logging.Logger):
    url = f"{GIT_BASE}/{full_name}.git"
    logger.info(f"Clonando {full_name}@{branch}")
    sh(["git", "clone", "--depth", "1", "--branch", branch, url, str(dest)], check=True, timeout=300)

# -------------------------
# Kernel fallback
# -------------------------
def resolve_kernel_name(requested: str | None) -> str:
    req = (requested or "python3").strip()
    if not KernelSpecManager:
        return "python3"
    try:
        ksm = KernelSpecManager()
        specs = ksm.find_kernel_specs()
        return req if req in specs else "python3"
    except Exception:
        return "python3"

# -------------------------
# Execução headless + timeout real
# -------------------------
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
    "OPENAI_API_KEY": "",
    "HF_TOKEN": "",
    "HF_HUB_DISABLE_TELEMETRY": "1",
}

def build_exec_code(nb_path: Path, kernel: str, timeout_s: int) -> str:
    """
    Código Python (executado no subprocesso) que:
      - executa o notebook com nbclient
      - captura a exceção real
      - imprime uma linha JSON com prefixo RESULT_PREFIX
    """
    return f"""
import json, sys, re, nbformat, traceback
from nbclient import NotebookClient
PREFIX = {RESULT_PREFIX!r}
try:
    nb = nbformat.read(r'''{nb_path}''', as_version=4)
    client = NotebookClient(nb, timeout={timeout_s}, kernel_name=r'''{kernel}''', allow_errors=False)
    client.execute()
    print(PREFIX + json.dumps({{"ok": True}}))
except Exception as e:
    # nbclient.CellExecutionError traz e.ename/e.evalue; outras exceções usam type/str
    name = getattr(e, "ename", type(e).__name__)
    msg  = getattr(e, "evalue", str(e))
    info = {{"ok": False, "error": name, "exc_msg": msg}}
    # heurísticas úteis
    m = re.search(r"No module named '([^']+)'", msg)
    if m:
        info["missing_module"] = m.group(1)
    if "401" in msg or "AuthenticationError" in name or "invalid_api_key" in msg.lower():
        info["auth_error"] = True
    if "FileNotFoundError" in name or "No such file or directory" in msg or "not found" in msg:
        info["file_missing"] = True
    print(PREFIX + json.dumps(info, ensure_ascii=False))
    sys.exit(1)
""".strip()

def parse_result_from_stdout(stdout: str) -> dict | None:
    """
    Varre o stdout e procura a linha NBEXEC_RESULT:{json}
    """
    for line in stdout.splitlines():
        if line.startswith(RESULT_PREFIX):
            try:
                return json.loads(line[len(RESULT_PREFIX):].strip())
            except Exception:
                return None
    return None

def run_nb_with_timeout(pybin: str, repo_dir: Path, nb_path: Path, kernel: str, timeout_s: int,
                        logger: logging.Logger) -> tuple[bool, Dict]:
    code = build_exec_code(nb_path, kernel, timeout_s)
    env = os.environ.copy()
    env.update(HEADLESS_ENV)
    t0 = time.time()
    try:
        p = sh([pybin, "-c", code], cwd=repo_dir, env=env, check=False, timeout=timeout_s)
        elapsed = round(time.time() - t0, 3)

        # tenta extrair resultado estruturado do stdout
        info = parse_result_from_stdout(p.stdout) or {}

        if p.returncode == 0 and info.get("ok") is True:
            return True, {"error": None, "exc_name": None, "failed_cell_index": None, "elapsed_s": elapsed}

        # Falhou: preenche com a exceção real quando disponível
        err_name = info.get("error") or ("SubprocessError" if p.returncode != 0 else "UnknownError")
        out = {
            "error": err_name,
            "exc_name": None,
            "failed_cell_index": None,
            "elapsed_s": elapsed,
            "stdout_tail": p.stdout[-2000:],
            "stderr_tail": p.stderr[-2000:],
        }
        # campos auxiliares úteis (não entram no CSV, só para log)
        if "missing_module" in info:
            out["missing_module"] = info["missing_module"]
        if "auth_error" in info:
            out["auth_error"] = info["auth_error"]
        if "file_missing" in info:
            out["file_missing"] = info["file_missing"]
        return False, out

    except subprocess.TimeoutExpired:
        return False, {
            "error": "TimeoutExpired",
            "exc_name": None,
            "failed_cell_index": None,
            "elapsed_s": round(time.time() - t0, 3),
        }
    except Exception as e:
        return False, {
            "error": type(e).__name__,
            "exc_name": None,
            "failed_cell_index": None,
            "elapsed_s": round(time.time() - t0, 3),
        }

# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", "--input-csv", dest="input_csv", required=True, help="CSV de coleta (notebooks)")
    ap.add_argument("--output", "--output-csv", dest="output_csv", default="data/outputs/execution_results.csv")
    ap.add_argument("--log-file", default="data/outputs/logs/execute_notebooks.log")
    ap.add_argument("--policy", choices=["strict", "relaxed"], default="relaxed")
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    logger = setup_logging(Path(args.log_file) if args.log_file else None)
    logger.info("== Iniciando executor de notebooks (headless + timeout real) ==")
    logger.info(f"Parâmetros: input={args.input_csv} output={args.output_csv} policy={args.policy} timeout={args.timeout}s limit={args.limit}")

    outp = Path(args.output_csv)
    outp.parent.mkdir(parents=True, exist_ok=True)

    tmp_root = Path(tempfile.mkdtemp(prefix="nbexec_"))
    logger.info(f"Workspace temporário: {tmp_root}")

    processed = 0
    try:
        with open(args.input_csv, newline="", encoding="utf-8") as f_in, \
             open(outp, "w", newline="", encoding="utf-8") as f_out:

            rd = csv.DictReader(f_in)
            fields = [
                "repo_full_name", "repo_default_branch", "file_path",
                "nb_ok_parse", "kernel_name", "language", "python_version_declared"
            ]
            out_fields = fields + ["exec_ok", "error", "exc_name", "failed_cell_index", "elapsed_s"]
            wr = csv.DictWriter(f_out, fieldnames=out_fields)
            wr.writeheader()

            env_cache: dict[str, tuple[str, str]] = {}

            for row in rd:
                if row.get("nb_ok_parse") != "True":
                    continue

                full = row["repo_full_name"]
                branch = row.get("repo_default_branch") or "main"
                rel = row["file_path"]
                nb_kernel = resolve_kernel_name(row.get("kernel_name"))

                repo_dir = tmp_root / full.replace("/", "__")
                if not repo_dir.exists():
                    try:
                        clone_repo(full, repo_dir, branch, logger)
                    except Exception as e:
                        logger.warning(f"Falha ao clonar {full}: {e}")
                        continue

                if full not in env_cache:
                    env_dir = tmp_root / (full.replace("/", "__") + "_env")
                    try:
                        pip, pybin = make_env(env_dir, args.policy, logger)
                        if args.policy == "strict":
                            try:
                                installed = install_reqs(pip, repo_dir, logger)
                                logger.info(f"Deps strict: {'instaladas' if installed else 'não encontradas/instalação falhou'}")
                            except Exception as e:
                                logger.warning(f"Falha ao instalar deps strict: {e}")
                        env_cache[full] = (pip, pybin)
                    except Exception as e:
                        logger.warning(f"Falha ao preparar venv para {full}: {e}")
                        continue
                else:
                    pip, pybin = env_cache[full]

                nb_path = repo_dir / rel
                if not nb_path.exists():
                    logger.warning(f"Notebook não encontrado no clone: {nb_path}")
                    continue

                logger.info(f"Executando notebook: {nb_path}")
                ok, info = run_nb_with_timeout(pybin, repo_dir, nb_path, nb_kernel, args.timeout, logger)

                wr.writerow({
                    "repo_full_name": full,
                    "repo_default_branch": branch,
                    "file_path": rel,
                    "nb_ok_parse": True,
                    "kernel_name": nb_kernel,
                    "language": row.get("language"),
                    "python_version_declared": row.get("python_version_declared"),
                    "exec_ok": ok,
                    "error": info.get("error"),
                    "exc_name": info.get("exc_name"),
                    "failed_cell_index": info.get("failed_cell_index"),
                    "elapsed_s": info.get("elapsed_s"),
                })

                if not ok and ("stdout_tail" in info or "stderr_tail" in info):
                    logger.warning(f"STDOUT(tail): {info.get('stdout_tail','')}")
                    logger.warning(f"STDERR(tail): {info.get('stderr_tail','')}")

                processed += 1
                if args.limit and processed >= args.limit:
                    logger.info("Limite atingido; encerrando loop.")
                    break

    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)
        logger.info("Limpeza do workspace temporário concluída.")

    logger.info(f"Concluído. Notebooks processados: {processed}")
    logger.info(f"Resultados em: {outp}")

if __name__ == "__main__":
    main()
