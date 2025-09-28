"""
Executa notebooks listados no CSV de coleta com nbclient, sob timeout
e isolamento por env. Saída: data/execution_results.csv

Políticas de deps:
- strict: se houver requirements.txt / Pipfile / setup.py ⇒ instala (pip).
- relaxed: usa um env base (ex.: conda env com numpy/pandas/matplotlib).

Obs:
- Clonamos o repositório (shallow) para respeitar paths relativos.
- Executamos o notebook no caminho relativo registrado no CSV.
"""

import argparse, csv, json, os, shutil, subprocess, sys, tempfile, time
from pathlib import Path
from typing import Dict, Tuple

import nbformat
from nbclient import NotebookClient, CellExecutionError

GIT_BASE = "https://github.com"

def sh(cmd, cwd=None, env=None, check=True):
    p = subprocess.run(cmd, cwd=cwd, env=env, text=True, capture_output=True)
    if check and p.returncode != 0:
        raise RuntimeError(f"cmd failed: {' '.join(cmd)}\nSTDOUT:\n{p.stdout}\nSTDERR:\n{p.stderr}")
    return p

def make_env(env_dir: Path, policy: str):
    env_dir.mkdir(parents=True, exist_ok=True)
    py = sys.executable  # usa o venv atual
    sh([py, "-m", "venv", str(env_dir)])
    pip = str(env_dir / "bin" / "pip")
    # baseline mínima (relaxed)
    base = ["pip", "setuptools", "wheel", "nbclient", "ipykernel"]
    sh([pip, "install", "--upgrade"] + base, check=True)
    return pip

def install_reqs(pip: str, repo_dir: Path):
    # Estrita: tenta na ordem: requirements.txt, Pipfile (via pipenv), setup.py/pyproject
    reqs = list(repo_dir.rglob("requirements.txt"))
    if reqs:
        sh([pip, "install", "-r", str(reqs[0])], check=True)
        return True
    pipfile = list(repo_dir.rglob("Pipfile"))
    if pipfile:
        # fallback simples: pipenv seria o ideal; aqui tentamos pipenv se disponível
        try:
            sh(["pipenv", "install", "--system", "--deploy"], cwd=repo_dir, check=True)
            return True
        except Exception:
            pass
    setup_py = list(repo_dir.rglob("setup.py"))
    pyproject = list(repo_dir.rglob("pyproject.toml"))
    if setup_py or pyproject:
        try:
            sh([pip, "install", "-e", "."], cwd=repo_dir, check=True)
            return True
        except Exception:
            return False
    return False

def clone_repo(full_name: str, dest: Path, branch: str):
    url = f"{GIT_BASE}/{full_name}.git"
    sh(["git", "clone", "--depth", "1", "--branch", branch, url, str(dest)])

def execute_notebook(nb_path: Path, timeout_s: int, kernel: str|None) -> Tuple[bool, Dict]:
    t0 = time.time()
    info = {"error": None, "failed_cell_index": None, "elapsed_s": None, "exc_name": None}
    try:
        nb = nbformat.read(nb_path, as_version=4)
        client = NotebookClient(
            nb,
            timeout=timeout_s,
            kernel_name=kernel or "python3",
            allow_errors=False,
            record_timing=True,
        )
        client.execute()
        ok = True
    except CellExecutionError as e:
        ok = False
        info["error"] = "CellExecutionError"
        info["exc_name"] = e.ename
        info["failed_cell_index"] = getattr(e, "cell_index", None)
    except Exception as e:
        ok = False
        info["error"] = type(e).__name__
    finally:
        info["elapsed_s"] = round(time.time() - t0, 3)
    return ok, info

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-csv", required=True, help="CSV de coleta (notebooks)")
    ap.add_argument("--output-csv", default="data/execution_results.csv")
    ap.add_argument("--policy", choices=["strict","relaxed"], default="relaxed")
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    outp = Path(args.output_csv)
    outp.parent.mkdir(parents=True, exist_ok=True)

    with open(args.input_csv, newline="", encoding="utf-8") as f_in, \
         open(outp, "w", newline="", encoding="utf-8") as f_out:
        rd = csv.DictReader(f_in)
        fields = ["repo_full_name","repo_default_branch","file_path",
                  "nb_ok_parse","kernel_name","language","python_version_declared"]
        out_fields = fields + ["exec_ok","error","exc_name","failed_cell_index","elapsed_s"]
        wr = csv.DictWriter(f_out, fieldnames=out_fields)
        wr.writeheader()

        tmp_root = Path(tempfile.mkdtemp(prefix="nbexec_"))
        try:
            count = 0
            for row in rd:
                if row.get("nb_ok_parse") != "True":
                    continue
                full = row["repo_full_name"]
                branch = row["repo_default_branch"] or "main"
                rel = row["file_path"]
                nb_kernel = row.get("kernel_name") or "python3"

                repo_dir = tmp_root / full.replace("/", "__")
                if not repo_dir.exists():
                    try:
                        clone_repo(full, repo_dir, branch)
                    except Exception:
                        continue

                # env isolado por repo
                env_dir = tmp_root / (full.replace("/", "__") + "_env")
                pip = str(env_dir / "bin" / "pip")
                pybin = str(env_dir / "bin" / "python")
                if not env_dir.exists():
                    pip = make_env(env_dir, args.policy)
                    if args.policy == "strict":
                        try:
                            install_reqs(pip, repo_dir)
                        except Exception:
                            pass

                nb_path = repo_dir / rel
                if not nb_path.exists():
                    continue

                # Executa no env isolado
                cmd = [
                    pybin, "-c",
                    (
                        "import nbformat;"
                        "from nbclient import NotebookClient;"
                        f"nb=nbformat.read(r'''{nb_path}''', as_version=4);"
                        f"NotebookClient(nb, timeout={args.timeout}, kernel_name=r'''{nb_kernel}''', allow_errors=False).execute()"
                    )
                ]
                t0 = time.time()
                try:
                    sh(cmd, cwd=repo_dir, check=True)
                    ok, info = True, {"error": None, "exc_name": None, "failed_cell_index": None}
                except Exception as e:
                    ok, info = False, {"error": type(e).__name__, "exc_name": None, "failed_cell_index": None}
                info["elapsed_s"] = round(time.time() - t0, 3)

                wr.writerow({
                    "repo_full_name": full,
                    "repo_default_branch": branch,
                    "file_path": rel,
                    "nb_ok_parse": True,
                    "kernel_name": nb_kernel,
                    "language": row.get("language"),
                    "python_version_declared": row.get("python_version_declared"),
                    "exec_ok": ok,
                    **info
                })

                count += 1
                if args.limit and count >= args.limit:
                    break
        finally:
            # limpe se quiser
            shutil.rmtree(tmp_root, ignore_errors=True)

if __name__ == "__main__":
    main()
