"""
Executa notebooks listados no CSV de coleta com nbclient, sob timeout real,
headless (sem janelas/GUI) e isolamento por venv por repositório.

- Preenche 'error' com a exceção real (ModuleNotFoundError, ImportError, etc.)
- Mantém logs e tails de stdout/stderr quando falha.
- (NOVO) Compara outputs do notebook executado vs. o notebook original salvo.

Políticas de deps:
- strict: tenta instalar deps do repo (requirements.txt / Pipfile / setup.py/pyproject).
- relaxed: instala baseline mínima (nbclient, ipykernel, nbformat).

Observações:
- Clona o repositório (shallow).
- Executa no caminho relativo do CSV.
- Headless: evita abrir janelas (pygame, Qt, matplotlib interativo etc.).
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

# -------------------------
# Código a ser executado no subprocesso
# -------------------------
def detect_python_version_compatibility(declared_version: str, current_version: tuple) -> tuple[bool, str]:
    """
    Verifica se a versão Python declarada é compatível com a versão atual.
    Retorna: (is_compatible, warning_message)
    """
    if not declared_version:
        return True, ""
    
    try:
        # Parse da versão declarada (ex: "3.11.9" ou "3.12.0")
        declared_parts = declared_version.split('.')
        declared_major = int(declared_parts[0])
        declared_minor = int(declared_parts[1]) if len(declared_parts) > 1 else 0
        
        current_major, current_minor = current_version[:2]
        
        # Python 2.x é incompatível
        if declared_major == 2:
            return False, f"Python 2.x não suportado (declarado: {declared_version})"
        
        # Python 3.x - verificar compatibilidade
        if declared_major == 3:
            # Versões muito antigas podem ter problemas
            if declared_minor < 7:
                return False, f"Python {declared_version} muito antigo (mínimo: 3.7)"
            
            # Versões muito novas podem ter recursos não disponíveis
            if declared_minor > current_minor + 2:
                return False, f"Python {declared_version} muito novo (atual: {current_major}.{current_minor})"
            
            # Avisos para diferenças menores
            if declared_minor != current_minor:
                return True, f"Versão diferente: declarado {declared_version}, atual {current_major}.{current_minor}"
        
        return True, ""
    
    except (ValueError, IndexError):
        return True, f"Versão malformada: {declared_version}"

def build_exec_code(nb_path: Path, kernel: str, timeout_s: int, declared_python_version: str = None) -> str:
    """
    Código Python (executado no subprocesso) que:
      - executa o notebook com nbclient
      - calcula hash de outputs do notebook executado
      - captura a exceção real
      - imprime uma linha JSON com prefixo RESULT_PREFIX
    """
    return f"""
import json, sys, re, nbformat, traceback, hashlib
from nbclient import NotebookClient
PREFIX = {RESULT_PREFIX!r}

def canonicalize_outputs_struct(outputs):
    canon = []
    for out in outputs or []:
        otype = out.get("output_type")
        if otype == "stream":
            canon.append({{"output_type":"stream","name":out.get("name"),"text":out.get("text","")}})
        elif otype in ("display_data","execute_result"):
            data = out.get("data") or {{}}
            keep = {{}}
            for k in sorted(data.keys()):
                v = data[k]
                if isinstance(v, list):
                    keep[k] = "".join(str(x) for x in v)
                else:
                    keep[k] = v
            canon.append({{"output_type":otype,"data":keep}})
        elif otype == "error":
            canon.append({{"output_type":"error","ename":out.get("ename"),"evalue":out.get("evalue"),"traceback":"\\n".join(out.get("traceback") or [])}})
        else:
            data = out.get("data") or {{}}
            keep = {{}}
            for k in sorted(data.keys()):
                v = data[k]
                keep[k] = v if not isinstance(v, list) else "".join(str(x) for x in v)
            canon.append({{"output_type": otype, "data": keep}})
    return canon

def hash_outputs_from_nb(nb):
    cells = nb.get("cells") or []
    outs_min = []
    for c in cells:
        if c.get("cell_type") == "code":
            outs_min.extend(canonicalize_outputs_struct(c.get("outputs") or []))
    blob = json.dumps(outs_min, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest(), len(outs_min)

try:
    # Verificar compatibilidade de versão Python
    import sys
    current_version = sys.version_info
    declared_version = {declared_python_version!r}
    
    if declared_version:
        declared_parts = declared_version.split('.')
        declared_major = int(declared_parts[0])
        declared_minor = int(declared_parts[1]) if len(declared_parts) > 1 else 0
        
        # Python 2.x é incompatível
        if declared_major == 2:
            print(PREFIX + json.dumps({{"ok": False, "error": "Python2NotSupported", "exc_msg": f"Python 2.x não suportado (declarado: {{declared_version}})"}}))
            sys.exit(1)
        
        # Versões muito antigas
        if declared_major == 3 and declared_minor < 7:
            print(PREFIX + json.dumps({{"ok": False, "error": "PythonTooOld", "exc_msg": f"Python {{declared_version}} muito antigo (mínimo: 3.7)"}}))
            sys.exit(1)
    
    nb = nbformat.read(r'''{nb_path}''', as_version=4)
    client = NotebookClient(nb, timeout={timeout_s}, kernel_name=r'''{kernel}''', allow_errors=False)
    client.execute()
    # nb foi modificado in-place; serialize para dict
    nb_dict = nbformat.writes(nb)
    nb_json = json.loads(nb_dict)
    h, n = hash_outputs_from_nb(nb_json)
    print(PREFIX + json.dumps({{"ok": True, "outputs_hash": h, "n_outputs": n}}))
except Exception as e:
    name = getattr(e, "ename", type(e).__name__)
    msg  = getattr(e, "evalue", str(e))
    info = {{"ok": False, "error": name, "exc_msg": msg}}
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
    for line in stdout.splitlines():
        if line.startswith(RESULT_PREFIX):
            try:
                return json.loads(line[len(RESULT_PREFIX):].strip())
            except Exception:
                return None
    return None

def run_nb_with_timeout(pybin: str, repo_dir: Path, nb_path: Path, kernel: str, timeout_s: int,
                        logger: logging.Logger, declared_python_version: str = None) -> tuple[bool, Dict]:
    env = os.environ.copy()
    env.update(HEADLESS_ENV)
    
    # Criar diretório isolado para evitar conflitos com módulos locais
    isolated_dir = repo_dir.parent / f"{repo_dir.name}_isolated"
    isolated_dir.mkdir(exist_ok=True)
    
    # Copiar apenas o notebook para o diretório isolado
    isolated_nb_path = isolated_dir / nb_path.name
    shutil.copy2(nb_path, isolated_nb_path)
    
    # Adicionar o diretório original ao PYTHONPATH para imports relativos funcionarem
    env["PYTHONPATH"] = f"{repo_dir}:{env.get('PYTHONPATH', '')}"
    
    code = build_exec_code(isolated_nb_path, kernel, timeout_s, declared_python_version)
    
    t0 = time.time()
    try:
        # Executar no diretório isolado, mas com PYTHONPATH apontando para o repo original
        p = sh([pybin, "-c", code], cwd=isolated_dir, env=env, check=False, timeout=timeout_s)
        elapsed = round(time.time() - t0, 3)

        info = parse_result_from_stdout(p.stdout) or {}

        if p.returncode == 0 and info.get("ok") is True:
            return True, {
                "error": None, "exc_name": None, "failed_cell_index": None, "elapsed_s": elapsed,
                "outputs_hash_exec": info.get("outputs_hash",""), "n_outputs_exec": info.get("n_outputs",0)
            }

        err_name = (info.get("error") or "SubprocessError") if p.returncode != 0 else (info.get("error") or "UnknownError")
        out = {
            "error": err_name,
            "exc_name": None,
            "failed_cell_index": None,
            "elapsed_s": elapsed,
            "stdout_tail": p.stdout[-2000:],
            "stderr_tail": p.stderr[-2000:],
            "outputs_hash_exec": info.get("outputs_hash",""),
            "n_outputs_exec": info.get("n_outputs",0)
        }
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
            "outputs_hash_exec": "",
            "n_outputs_exec": 0
        }
    except Exception as e:
        return False, {
            "error": type(e).__name__,
            "exc_name": None,
            "failed_cell_index": None,
            "elapsed_s": round(time.time() - t0, 3),
            "outputs_hash_exec": "",
            "n_outputs_exec": 0
        }

# -------------------------
# Mapeamento de módulos relacionados
# -------------------------
MODULE_FAMILIES = {
    "torch": ["torch", "torchvision", "torchaudio", "torchinfo"],
    "tensorflow": ["tensorflow", "tensorflow-gpu"],
    "sklearn": ["scikit-learn", "sklearn"],
    "cv2": ["opencv-python", "opencv-contrib-python"],
    "PIL": ["Pillow"],
    "bs4": ["beautifulsoup4"],
    "yaml": ["PyYAML"],
    "dotenv": ["python-dotenv"],
    "dateutil": ["python-dateutil"],
    "pytz": ["pytz"],
    "tzdata": ["tzdata"],
    "psycopg2": ["psycopg2-binary"],
    "pymongo": ["pymongo"],
    "redis": ["redis"],
    "celery": ["celery"],
    "flask": ["Flask"],
    "fastapi": ["fastapi"],
    "django": ["Django"],
    "pytest": ["pytest"],
    "black": ["black"],
    "flake8": ["flake8"],
    "mypy": ["mypy"],
    "jupyter": ["jupyter", "jupyterlab"],
    "ipywidgets": ["ipywidgets"],
    "tqdm": ["tqdm"],
    "click": ["click"],
    "toml": ["toml"],
    "networkx": ["networkx"],
    "sympy": ["sympy"],
    "statsmodels": ["statsmodels"],
    "plotly": ["plotly"],
    "bokeh": ["bokeh"],
    "altair": ["altair"],
    "plotnine": ["plotnine"],
    "transformers": ["transformers"],
    "openai": ["openai"],
    "langchain": ["langchain"],
    "streamlit": ["streamlit"],
}

def get_related_modules(module_name: str) -> list[str]:
    """Retorna uma lista de módulos relacionados que podem ser instalados juntos."""
    return MODULE_FAMILIES.get(module_name, [module_name])

# -------------------------
# Retry com instalação de módulo
# -------------------------
def retry_with_module_install(pip: str, pybin: str, repo_dir: Path, nb_path: Path, 
                               kernel: str, timeout_s: int, missing_module: str,
                               logger: logging.Logger, max_retries: int = 5) -> tuple[bool, Dict, bool, str]:
    """
    Tenta instalar o módulo ausente e re-executar o notebook.
    Se ainda houver ModuleNotFoundError, tenta instalar recursivamente até max_retries.
    Retorna: (success, info_dict, retry_attempted, installed_modules)
    """
    installed_modules = []
    retry_count = 0
    
    while retry_count < max_retries:
        logger.info(f"Tentativa {retry_count + 1}/{max_retries}: instalando módulo ausente: {missing_module}")
        try:
            # Instala o módulo e módulos relacionados
            related_modules = get_related_modules(missing_module)
            logger.info(f"Instalando módulos relacionados: {related_modules}")
            
            result = sh([pip, "install"] + related_modules, check=False, timeout=300)
            if result.returncode != 0:
                logger.warning(f"Falha ao instalar {related_modules}: {result.stderr[:500]}")
                # Tenta instalar apenas o módulo principal
                result = sh([pip, "install", missing_module], check=False, timeout=120)
                if result.returncode != 0:
                    logger.warning(f"Falha ao instalar {missing_module}: {result.stderr[:500]}")
                    return False, {"error": "ModuleInstallFailed", "elapsed_s": 0}, True, ",".join(installed_modules)
            
            installed_modules.extend(related_modules)
            logger.info(f"Módulo {missing_module} instalado com sucesso. Tentando executar novamente...")
            
            # Re-executa o notebook
            ok, info = run_nb_with_timeout(pybin, repo_dir, nb_path, kernel, timeout_s, logger, None)
            
            if ok:
                return True, info, True, ",".join(installed_modules)
            
            # Se ainda há ModuleNotFoundError, tenta instalar o próximo módulo
            if "missing_module" in info:
                missing_module = info["missing_module"]
                retry_count += 1
                continue
            else:
                # Outro tipo de erro, não é mais ModuleNotFoundError
                return False, info, True, ",".join(installed_modules)
        
        except subprocess.TimeoutExpired:
            logger.warning(f"Timeout ao instalar {missing_module}")
            return False, {"error": "ModuleInstallTimeout", "elapsed_s": 0}, True, ",".join(installed_modules)
        except Exception as e:
            logger.warning(f"Erro ao instalar {missing_module}: {e}")
            return False, {
                "error": f"ModuleInstallError_{type(e).__name__}",
                "elapsed_s": 0
            }, True, ",".join(installed_modules)
    
    # Se chegou aqui, esgotou as tentativas
    logger.warning(f"Esgotadas {max_retries} tentativas de instalação de módulos")
    return False, {"error": "MaxRetriesExceeded", "elapsed_s": 0}, True, ",".join(installed_modules)

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
    
    if major == 3:
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
    
    return "notebook-executor"  # fallback

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
    try:
        with open(args.input_csv, newline="", encoding="utf-8") as f_in, \
             open(outp, "w", newline="", encoding="utf-8") as f_out:

            rd = csv.DictReader(f_in)
            fields = [
                "repo_full_name", "repo_default_branch", "file_path",
                "nb_ok_parse", "kernel_name", "language", "python_version_declared"
            ]
            out_fields = fields + [
                "exec_ok", "error", "exc_name", "failed_cell_index", "elapsed_s",
                "retry_attempted", "retry_success", "installed_modules",
                "original_found", "outputs_equal", "outputs_hash_orig", "outputs_hash_exec",
                "n_outputs_orig", "n_outputs_exec"
            ]
            wr = csv.DictWriter(f_out, fieldnames=out_fields)
            wr.writeheader()

            env_cache: dict[str, tuple[str, str]] = {}

            for row in rd:
                if row.get("nb_ok_parse") != "True":
                    continue

                full = row["repo_full_name"]          # owner/repo
                owner, repo = full.split("/", 1)
                branch = row.get("repo_default_branch") or "main"
                rel = row["file_path"]
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

                # Verificar compatibilidade de versão Python
                declared_python_version = row.get("python_version_declared", "")
                if declared_python_version:
                    is_compatible, warning = detect_python_version_compatibility(declared_python_version, sys.version_info)
                    if not is_compatible:
                        logger.warning(f"Incompatibilidade de versão Python: {warning}")
                        # Pular notebook incompatível
                        wr.writerow({
                            "repo_full_name": full,
                            "repo_default_branch": branch,
                            "file_path": rel,
                            "nb_ok_parse": True,
                            "kernel_name": nb_kernel,
                            "language": row.get("language"),
                            "python_version_declared": declared_python_version,
                            "exec_ok": False,
                            "error": "PythonVersionIncompatible",
                            "exc_name": None,
                            "failed_cell_index": None,
                            "elapsed_s": 0,
                            "retry_attempted": False,
                            "retry_success": False,
                            "installed_modules": "",
                            "original_found": str(original_found),
                            "outputs_equal": "",
                            "outputs_hash_orig": outputs_hash_orig,
                            "outputs_hash_exec": "",
                            "n_outputs_orig": n_outputs_orig,
                            "n_outputs_exec": 0,
                        })
                        processed += 1
                        continue
                    elif warning:
                        logger.info(f"Aviso de versão: {warning}")

                logger.info(f"Executando notebook: {nb_path}")
                ok, info = run_nb_with_timeout(pybin, repo_dir, nb_path, nb_kernel, args.timeout, logger, declared_python_version)
                
                # Retry logic se ModuleNotFoundError
                retry_attempted = False
                retry_success = False
                installed_modules = ""
                
                if not ok and "missing_module" in info:
                    missing_mod = info["missing_module"]
                    logger.info(f"ModuleNotFoundError detectado: {missing_mod}. Tentando instalar e re-executar...")
                    ok, info, retry_attempted, installed_modules = retry_with_module_install(
                        pip, pybin, repo_dir, nb_path, nb_kernel, args.timeout, missing_mod, logger
                    )
                    retry_success = ok
                    logger.info(f"Resultado do retry: {'sucesso' if ok else 'falha'}")

                outputs_hash_exec = info.get("outputs_hash_exec","")
                n_outputs_exec = info.get("n_outputs_exec",0)
                outputs_equal = ""
                if original_found and outputs_hash_orig:
                    outputs_equal = str(outputs_hash_orig == outputs_hash_exec)

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
                    "retry_attempted": str(retry_attempted),
                    "retry_success": str(retry_success) if retry_attempted else "",
                    "installed_modules": installed_modules,
                    "original_found": str(original_found),
                    "outputs_equal": outputs_equal,
                    "outputs_hash_orig": outputs_hash_orig,
                    "outputs_hash_exec": outputs_hash_exec,
                    "n_outputs_orig": n_outputs_orig,
                    "n_outputs_exec": n_outputs_exec,
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
