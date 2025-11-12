"""
Executa notebooks listados no CSV de coleta via Docker containers.

Execução oficial é via Docker; sem Docker o pipeline não roda.

- Cada notebook é executado em um container Docker isolado por versão de Python
- Worker interno (container_worker.py) executa o notebook e implementa retry de módulos ausentes
- Preenche campos crus de erro (exec_exception_type, exec_exception_str, exec_traceback_str)
- Compara outputs do notebook executado vs. o notebook original salvo

Observações:
- Clona o repositório (shallow) no host
- Monta repositório e diretório de saída como volumes no container
- Headless: evita abrir janelas (pygame, Qt, matplotlib interativo etc.)
- Retry automático: até 5 módulos distintos podem ser instalados durante execução
"""

from __future__ import annotations
import argparse
import csv
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
import json
import hashlib
from pathlib import Path
from typing import Dict, Optional

try:
    from jupyter_client.kernelspec import KernelSpecManager
except Exception:
    KernelSpecManager = None

GIT_BASE = "https://github.com"
RESULT_PREFIX = "NBEXEC_RESULT:"
DOCKERFILE_MAP = {
    "notebook-executor": "Dockerfile.python310",
    "notebook-executor-py27": "Dockerfile.python27",
    "notebook-executor-py35": "Dockerfile.python35",
    "notebook-executor-py36": "Dockerfile.python36",
    "notebook-executor-python38": "Dockerfile.python38",
    "notebook-executor-python39": "Dockerfile.python39",
    "notebook-executor-python311": "Dockerfile.python311",
    "notebook-executor-python312": "Dockerfile.python312",
    "notebook-executor-py313": "Dockerfile.python313",
}

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
## Docker-only runner: venv/host dependency management removed from main flow.

# -------------------------
# Git
# -------------------------
def clone_repo(full_name: str, dest: Path, branch: str, logger: logging.Logger):
    url = f"{GIT_BASE}/{full_name}.git"
    logger.info(f"Clonando {full_name}@{branch}")
    sh(["git", "clone", "--depth", "1", "--branch", branch, url, str(dest)], check=True, timeout=300)

def ensure_docker_image(image: str, logger: logging.Logger):
    """
    Garante que a imagem docker necessária exista localmente. Se não existir e
    houver Dockerfile correspondente em scripts/dockers/, executa docker build.
    """
    try:
        sh(["docker", "image", "inspect", image], check=True)
        return
    except subprocess.CalledProcessError:
        pass

    dockerfile = DOCKERFILE_MAP.get(image)
    scripts_dir = Path(__file__).resolve().parent
    dockerfile_path = scripts_dir / "dockers" / dockerfile if dockerfile else None
    if not dockerfile or not dockerfile_path or not dockerfile_path.exists():
        logger.warning(f"Imagem {image} não encontrada e nenhum Dockerfile mapeado. "
                       "Construa manualmente antes de prosseguir.")
        raise RuntimeError(f"Missing Docker image {image}")

    logger.info(f"Construindo imagem Docker {image} (Dockerfile: {dockerfile_path}) ...")
    project_root = scripts_dir.parent
    try:
        sh(["docker", "build", "-t", image, "-f", str(dockerfile_path), str(project_root)], check=True)
    except subprocess.CalledProcessError as exc:
        logger.error(f"Falha ao construir a imagem {image}: {exc.stderr}")
        raise
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

# -------- Canonicalização & hash de outputs (compartilhado) ----------
def canonicalize_outputs_struct(outputs: list) -> list:
    """
    Retorna uma representação mínima e ordenada dos outputs para comparação determinística.
    - Ignora execution_count, metadata volátil, ids aleatórios.
    - Mantém tipo, stream name (stdout/stderr), e dados MIME.
    - Ordena chaves de data (MIME) e normaliza listas/strings.
    """
    canon = []
    for out in outputs or []:
        otype = out.get("output_type")
        if otype == "stream":
            canon.append({
                "output_type": "stream",
                "name": out.get("name"),            # stdout/stderr
                "text": out.get("text", ""),        # texto
            })
        elif otype in ("display_data", "execute_result"):
            data = out.get("data") or {}
            # seleciona apenas MIME estáveis
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
            # outros tipos raros
            data = out.get("data") or {}
            keep = {}
            for k in sorted(data.keys()):
                v = data[k]
                keep[k] = v if not isinstance(v, list) else "".join(str(x) for x in v)
            canon.append({"output_type": otype, "data": keep})
    return canon

def hash_outputs_from_nbjson(nbjson: dict) -> tuple[str, int]:
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

# Execution pipeline always runs through Docker + container_worker for consistency across hosts.

# -------------------------
# Localização do original salvo
# -------------------------
def find_original_notebook(originals_dir: Path, owner: str, repo: str, file_basename: str) -> Optional[Path]:
    """
    Procura por arquivos no padrão: originals_dir/owner/repo/*_{basename}
    Retorna o primeiro encontrado (ou None).
    """
    root = originals_dir / owner / repo
    if not root.exists():
        return None
    # múltiplas versões (sha8_...). Escolhemos arbitrariamente a primeira.
    candidates = sorted(root.glob(f"*_{file_basename}"))
    return candidates[0] if candidates else None

# -------------------------
# Main
# -------------------------
def analyze_python_versions_from_csv(csv_path: Path) -> dict:
    """
    Analisa as versões Python declaradas no CSV e retorna estatísticas.
    """
    versions_count = {}
    total_notebooks = 0
    
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("nb_ok_parse") == "True":
                    total_notebooks += 1
                    version = row.get("python_version_declared", "").strip()
                    if version:
                        versions_count[version] = versions_count.get(version, 0) + 1
                    else:
                        versions_count["unknown"] = versions_count.get("unknown", 0) + 1
    except Exception as e:
        print(f"Erro ao analisar CSV: {e}")
    
    return {
        "versions_count": versions_count,
        "total_notebooks": total_notebooks,
        "most_common_version": max(versions_count.items(), key=lambda x: x[1])[0] if versions_count else "unknown"
    }

def suggest_docker_image(python_version: str) -> str:
    """
    Sugere a imagem Docker baseada na versão Python.
    """
    if not python_version or python_version == "unknown":
        return "notebook-executor"  # Python 3.10 padrão
    
    version_parts = python_version.split('.')
    major = int(version_parts[0]) if version_parts else 0
    minor = int(version_parts[1]) if len(version_parts) > 1 else 0
    
    if major == 2:
        if minor == 7:
            return "notebook-executor-py27"
    elif major == 3:
        if minor == 5:
            return "notebook-executor-py35"
        elif minor == 6:
            return "notebook-executor-py36"
        if minor == 8:
            return "notebook-executor-python38"
        elif minor == 9:
            return "notebook-executor-python39"
        elif minor == 10:
            return "notebook-executor"
        elif minor == 11:
            return "notebook-executor-python311"
        elif minor == 12:
            return "notebook-executor-python312"
        elif minor == 13:
            return "notebook-executor-py313"
    
    return "notebook-executor"  # fallback


def run_worker_in_docker(image: str, repo_dir: Path, out_dir: Path, notebook_rel_path: str,
                         timeout_s: int, declared_python_version: str, logger: logging.Logger,
                         notebook_id: str | None = None) -> Dict:
    """
    Orquestra a execução no container chamando scripts.container_worker.
    Lê o result.json gerado no volume /workspace/out.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    envs = [
        "-e", f"NOTEBOOK_REL_PATH={notebook_rel_path}",
        "-e", f"TIMEOUT_SECONDS={timeout_s}",
        "-e", f"PYTHON_VERSION_DECLARED={declared_python_version or ''}",
    ]
    if notebook_id:
        envs += ["-e", f"NOTEBOOK_ID={notebook_id}"]
    cmd = [
        "docker", "run", "--rm",
        "-v", f"{str(repo_dir.absolute())}:/workspace/repo:ro",
        "-v", f"{str(out_dir.absolute())}:/workspace/out",
    ] + envs + [
        image, "python", "-m", "scripts.container_worker"
    ]
    logger.info(f"docker run: {' '.join(cmd)}")
    try:
        p = subprocess.run(cmd, text=True, capture_output=True, check=False, timeout=max(timeout_s + 60, 120))
        if p.returncode != 0:
            logger.warning(f"Docker returned {p.returncode}. STDOUT/ERR tail:\n{p.stdout[-1000:]}\n{p.stderr[-1000:]}")
        # Read result.json
        result_json = out_dir / "result.json"
        if result_json.exists():
            try:
                return json.loads(result_json.read_text(encoding="utf-8"))
            except Exception as e:
                logger.warning(f"Falha ao ler result.json: {e}")
                return {"exec_ok": False, "exec_exception_type": "ResultReadError", "exec_exception_str": str(e)}
        else:
            return {"exec_ok": False, "exec_exception_type": "ResultMissing", "exec_exception_str": "result.json não encontrado"}
    except subprocess.TimeoutExpired:
        return {"exec_ok": False, "exec_exception_type": "TimeoutExpired", "exec_exception_str": "docker run timeout"}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", "--input-csv", dest="input_csv", required=True, help="CSV de coleta (notebooks)")
    ap.add_argument("--output", "--output-csv", dest="output_csv", default="data/outputs/execution_results.csv")
    ap.add_argument("--log-file", default="data/outputs/logs/execute_notebooks.log")
    ap.add_argument("--policy", choices=["strict", "relaxed"], default="relaxed")
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--originals-dir", type=str, default=None,
                    help="Diretório onde estão os .ipynb originais salvos pelo coletor (--save-notebooks-dir).")
    ap.add_argument("--analyze-versions", action="store_true",
                    help="Apenas analisar versões Python no CSV e sugerir imagem Docker.")
    args = ap.parse_args()

    logger = setup_logging(Path(args.log_file) if args.log_file else None)
    logger.info("== Iniciando executor de notebooks (headless + timeout real) ==")
    logger.info(f"Parâmetros: input={args.input_csv} output={args.output_csv} policy={args.policy} timeout={args.timeout}s limit={args.limit} originals_dir={args.originals_dir}")

    # Análise de versões Python se solicitado
    if args.analyze_versions:
        logger.info("🔍 Analisando versões Python nos notebooks...")
        analysis = analyze_python_versions_from_csv(Path(args.input_csv))
        
        print("\n📊 Análise de versões Python:")
        print(f"   Total de notebooks: {analysis['total_notebooks']}")
        print("   Versões encontradas:")
        
        for version, count in sorted(analysis['versions_count'].items()):
            percentage = (count / analysis['total_notebooks']) * 100 if analysis['total_notebooks'] > 0 else 0
            print(f"     - Python {version}: {count} notebooks ({percentage:.1f}%)")
        
        most_common = analysis['most_common_version']
        suggested_image = suggest_docker_image(most_common)
        
        print("\n🎯 Recomendações:")
        print(f"   Versão mais comum: Python {most_common}")
        print(f"   Imagem Docker sugerida: {suggested_image}")
        
        if most_common != "unknown":
            print("\n🚀 Comando para construir e executar:")
            print(f"   docker build -t {suggested_image} -f Dockerfile.python{most_common.replace('.', '')} .")
            print(f"   docker run --rm -v $(pwd)/scripts/data:/workspace/data -v $(pwd)/scripts/data/run_20251028_144659/raw_ipynb:/workspace/originals {suggested_image}")
        else:
            print("\n⚠️  Nenhuma versão Python declarada encontrada. Usando imagem padrão.")
        
        return

    outp = Path(args.output_csv)
    outp.parent.mkdir(parents=True, exist_ok=True)

    # Análise rápida de versões Python para logging
    logger.info("🔍 Analisando versões Python nos notebooks...")
    analysis = analyze_python_versions_from_csv(Path(args.input_csv))
    most_common = analysis['most_common_version']
    suggested_image = suggest_docker_image(most_common)
    
    logger.info(f"📊 Versões Python encontradas: {dict(analysis['versions_count'])}")
    logger.info(f"🎯 Versão mais comum: Python {most_common}")
    logger.info(f"🐳 Imagem Docker sugerida: {suggested_image}")
    
    if most_common != "unknown":
        logger.info("💡 Para usar a versão correta, execute:")
        logger.info(f"   docker build -t {suggested_image} -f Dockerfile.python{most_common.replace('.', '')} .")

    originals_dir = Path(args.originals_dir) if args.originals_dir else None

    tmp_root = Path(tempfile.mkdtemp(prefix="nbexec_"))
    logger.info(f"Workspace temporário: {tmp_root}")

    processed = 0
    ensured_images = set()
    try:
        with open(args.input_csv, newline="", encoding="utf-8") as f_in, \
             open(outp, "w", newline="", encoding="utf-8") as f_out:

            rd = csv.DictReader(f_in)
            base_fields = [
                "notebook_id", "repo_full_name", "repo_default_branch", "repo_created_at",
                "file_path", "commit_sha", "notebook_rel_path",
                "nb_ok_parse", "kernel_name", "language", "python_version_declared"
            ]
            out_fields = base_fields + [
                "exec_ok", "elapsed_s",
                "exec_env_python_version", "exec_docker_image",
                "exec_legacy_env",
                "exec_error_type",
                "retry_attempted", "retry_missing_modules_count", "retry_missing_modules", "retry_success",
                "exec_exception_type", "exec_exception_module", "exec_exception_str", "exec_traceback_str",
                "original_found", "outputs_equal", "outputs_hash_orig", "outputs_hash_exec",
                "n_outputs_orig", "n_outputs_exec",
                "outputs_hash_canonical", "outputs_n_cells", "outputs_n_bytes"
            ]
            wr = csv.DictWriter(f_out, fieldnames=out_fields)
            wr.writeheader()

            for row in rd:
                if row.get("nb_ok_parse") != "True":
                    continue

                full = row["repo_full_name"]          # owner/repo
                owner, repo = full.split("/", 1)
                branch = row.get("repo_default_branch") or "main"
                rel = row["file_path"]
                notebook_id = row.get("notebook_id") or ""
                commit_sha = row.get("commit_sha") or ""
                if not commit_sha:
                    raise RuntimeError(
                        f"Missing commit_sha for notebook_id={notebook_id} ({full}/{rel}). "
                        "Re-run collection to ensure commit SHA is captured."
                    )
                notebook_rel_path = row.get("notebook_rel_path") or rel
                nb_kernel = resolve_kernel_name(row.get("kernel_name"))

                # Localiza original salvo (se informado)
                original_found = False
                outputs_hash_orig = ""
                n_outputs_orig = ""
                if originals_dir:
                    orig_path = find_original_notebook(originals_dir, owner, repo, os.path.basename(rel))
                    if orig_path and orig_path.exists():
                        try:
                            import nbformat
                            nb_orig = nbformat.read(str(orig_path), as_version=4)
                            # nbformat.read retorna NotebookNode; converte p/ dict consistente
                            nb_orig_json = json.loads(nbformat.writes(nb_orig))
                            outputs_hash_orig, n_orig = hash_outputs_from_nbjson(nb_orig_json)
                            n_outputs_orig = str(n_orig)
                            original_found = True
                        except Exception as e:
                            logger.warning(f"Falha ao ler original {orig_path}: {e}")

                repo_dir = tmp_root / full.replace("/", "__")
                if not repo_dir.exists():
                    try:
                        clone_repo(full, repo_dir, branch, logger)
                    except Exception as e:
                        logger.warning(f"Falha ao clonar {full}: {e}")
                        continue

                nb_path = repo_dir / rel
                if not nb_path.exists():
                    logger.warning(f"Notebook não encontrado no clone: {nb_path}")
                    continue

                # Execução via Docker
                declared_python_version = (row.get("python_version_declared") or "").strip()
                image = suggest_docker_image(declared_python_version)
                if image not in ensured_images:
                    ensure_docker_image(image, logger)
                    ensured_images.add(image)
                logger.info(f"Executando via Docker: image={image} nb={rel}")
                out_dir = repo_dir / "_out" / rel.replace("/", "__")
                info = run_worker_in_docker(image, repo_dir, out_dir, rel, args.timeout, declared_python_version, logger, notebook_id=notebook_id)

                outputs_hash_exec = info.get("outputs_hash_exec","")
                n_outputs_exec = info.get("n_outputs_exec",0)
                outputs_equal = ""
                if original_found and outputs_hash_orig:
                    outputs_equal = str(outputs_hash_orig == outputs_hash_exec)
                
                # Flag notebooks targeting deprecated Python versions (major.minor)
                ver_full = info.get("exec_env_python_version","")
                try:
                    parts = str(ver_full).split(".")
                    major_minor = f"{int(parts[0])}.{int(parts[1])}" if len(parts) >= 2 else ""
                except Exception:
                    major_minor = ""
                exec_legacy_env = str(major_minor in {"2.7", "3.5", "3.6"})

                # Error taxonomy (derived; raw fields permanecem a fonte da verdade)
                exc_type = (info.get("exec_exception_type") or "").strip()
                exec_error_type = "ok" if info.get("exec_ok") else "other"
                if not info.get("exec_ok"):
                    msg = (info.get("exec_exception_str") or "").lower()
                    if exc_type in ("ModuleNotFoundError","ImportError"):
                        exec_error_type = "missing_dependency"
                    elif exc_type in ("FileNotFoundError",) or "no such file or directory" in msg:
                        exec_error_type = "file_not_found"
                    elif exc_type in ("TimeoutExpired","TimeoutError"):
                        exec_error_type = "timeout"
                    elif exc_type in ("SyntaxError","IndentationError"):
                        exec_error_type = "syntax_error"

                wr.writerow({
                    "notebook_id": notebook_id,
                    "repo_full_name": full,
                    "repo_default_branch": branch,
                    "repo_created_at": row.get("repo_created_at"),
                    "file_path": rel,
                    "commit_sha": commit_sha,
                    "notebook_rel_path": notebook_rel_path,
                    "nb_ok_parse": True,
                    "kernel_name": nb_kernel,
                    "language": row.get("language"),
                    "python_version_declared": row.get("python_version_declared"),
                    "exec_ok": info.get("exec_ok"),
                    "elapsed_s": info.get("elapsed_s"),
                    "exec_env_python_version": info.get("exec_env_python_version",""),
                    "exec_docker_image": image,
                    "exec_legacy_env": exec_legacy_env,
                    "exec_error_type": exec_error_type,
                    "retry_attempted": str(info.get("retry_attempted", False)),
                    "retry_missing_modules_count": info.get("retry_missing_modules_count", 0),
                    "retry_missing_modules": json.dumps(info.get("retry_missing_modules", []), ensure_ascii=False) if isinstance(info.get("retry_missing_modules"), (list, dict)) else (info.get("retry_missing_modules") or ""),
                    "retry_success": str(info.get("retry_success", False)),
                    "exec_exception_type": info.get("exec_exception_type",""),
                    "exec_exception_module": info.get("exec_exception_module",""),
                    "exec_exception_str": info.get("exec_exception_str",""),
                    "exec_traceback_str": info.get("exec_traceback_str",""),
                    "original_found": str(original_found),
                    "outputs_equal": outputs_equal,
                    "outputs_hash_orig": outputs_hash_orig,
                    "outputs_hash_exec": outputs_hash_exec,
                    "n_outputs_orig": n_outputs_orig,
                    "n_outputs_exec": n_outputs_exec,
                    "outputs_hash_canonical": info.get("outputs_hash_canonical",""),
                    "outputs_n_cells": info.get("outputs_n_cells",""),
                    "outputs_n_bytes": info.get("outputs_n_bytes",""),
                })

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
