#!/bin/bash
# Shell script para executar o pipeline completo de coleta e execução de notebooks
# Prepara os dados necessários para o pipeline_master.ipynb

set -euo pipefail

# Cores para output (opcional, mas útil)
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Função para imprimir mensagens formatadas
info() {
    echo -e "${BLUE}[INFO]${NC} $*"
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $*"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $*"
}

error() {
    echo -e "${RED}[ERROR]${NC} $*" >&2
}

# Diretório raiz do projeto (assumindo que o script está em scripts/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Diretórios padrão
DATA_OUTPUTS="$PROJECT_ROOT/data/outputs"
CONFIG_DIR="$PROJECT_ROOT/config"
LOGS_DIR="$DATA_OUTPUTS/logs"
NOTEBOOKS_ORIG_DIR="$DATA_OUTPUTS/notebooks_originais"
REPOS_FULL_DIR="${REPOS_FULL_DIR:-$DATA_OUTPUTS/repositorios_completos}"
DOCKERS_DIR="$SCRIPT_DIR/dockers"

declare -A DOCKERFILE_MAP=(
    ["notebook-executor"]="Dockerfile.python310"
    ["notebook-executor-py27"]="Dockerfile.python27"
    ["notebook-executor-py35"]="Dockerfile.python35"
    ["notebook-executor-py36"]="Dockerfile.python36"
    ["notebook-executor-python38"]="Dockerfile.python38"
    ["notebook-executor-python39"]="Dockerfile.python39"
    ["notebook-executor-python311"]="Dockerfile.python311"
    ["notebook-executor-python312"]="Dockerfile.python312"
    ["notebook-executor-py313"]="Dockerfile.python313"
)

# Arquivos de saída esperados
COLLECTION_CSV="$DATA_OUTPUTS/collection.csv"
EXECUTION_CSV="$DATA_OUTPUTS/execution_results.csv"
PIMENTEL_REF="$CONFIG_DIR/pimentel2019_reference.json"

# Valores padrão
MAX_ITEMS="${MAX_ITEMS:-400}"
EXEC_TIMEOUT="${EXEC_TIMEOUT:-300}"
EXEC_POLICY="${EXEC_POLICY:-strict}"
SAVE_NOTEBOOKS="${SAVE_NOTEBOOKS:-true}"
DOWNLOAD_FULL_REPOS="${DOWNLOAD_FULL_REPOS:-true}"
SKIP_COLLECT="${SKIP_COLLECT:-false}"
SKIP_EXECUTE="${SKIP_EXECUTE:-false}"
RUN_SUMMARY="${RUN_SUMMARY:-true}"
REBUILD_IMAGES="${REBUILD_IMAGES:-false}"

# Variáveis que serão definidas via argumentos
DATE_START=""
DATE_END=""
EXEC_LIMIT=""

# Função para exibir ajuda
show_help() {
    cat << EOF
Uso: $0 [OPÇÕES]

Executa o pipeline completo de coleta e execução de notebooks Jupyter.

OPÇÕES:
  --date-start DATE      Data inicial no formato YYYY-MM-DD (obrigatório)
  --date-end DATE        Data final no formato YYYY-MM-DD (obrigatório)
  --max-items N          Limite de notebooks a coletar (padrão: 1000)
  --exec-timeout N       Timeout para execução de notebooks em segundos (padrão: 300)
  --exec-limit N         Limite de notebooks a executar (opcional)
  --exec-policy POLICY   Política de execução: strict ou relaxed (padrão: relaxed)
                         strict = instala dependências declaradas do repositório antes da execução
                         relaxed = usa imagem base apenas com pacotes essenciais
  --no-save-notebooks    Não salvar notebooks originais
  --save-full-repos      Fazer download completo dos repositórios processados (tarball do branch padrão)
                         Quando ativo, os repositórios completos são automaticamente usados na execução
                         para reduzir erros de "file_not_found" e evitar clonagens desnecessárias
  --repos-dir DIR        Diretório onde os repositórios completos serão salvos (padrão: data/outputs/repositorios_completos)
  --skip-collect         Pular etapa de coleta se collection.csv já existe
  --skip-execute         Pular etapa de execução se execution_results.csv já existe
  --no-summary           Não executar summarize_collection.py
  --rebuild-images       Forçar rebuild das imagens Docker antes da execução
  -h, --help             Exibir esta ajuda

VARIÁVEIS DE AMBIENTE:
  GITHUB_TOKEN           Token de acesso do GitHub (obrigatório)
  MAX_ITEMS              Limite de notebooks a coletar
  EXEC_TIMEOUT           Timeout para execução
  EXEC_LIMIT             Limite de notebooks a executar
  EXEC_POLICY            Política de execução (strict/relaxed)
  SAVE_NOTEBOOKS         Salvar notebooks originais (true/false)
  DOWNLOAD_FULL_REPOS    Baixar repositórios completos (true/false)
  REPOS_FULL_DIR         Caminho para salvar repositórios completos
  SKIP_COLLECT           Pular coleta (true/false)
  SKIP_EXECUTE           Pular execução (true/false)
  RUN_SUMMARY            Executar resumo (true/false)
  REBUILD_IMAGES         Forçar rebuild das imagens Docker (true/false)

EXEMPLOS:
  # Uso básico
  $0 --date-start 2025-01-01 --date-end 2025-01-31

  # Com parâmetros customizados
  $0 --date-start 2025-01-01 --date-end 2025-01-31 --max-items 500 --exec-limit 100

  # Pular etapas já executadas
  $0 --date-start 2025-01-01 --date-end 2025-01-31 --skip-collect --skip-execute
EOF
}

# Parsing de argumentos
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --date-start)
                DATE_START="$2"
                shift 2
                ;;
            --date-end)
                DATE_END="$2"
                shift 2
                ;;
            --max-items)
                MAX_ITEMS="$2"
                shift 2
                ;;
            --exec-timeout)
                EXEC_TIMEOUT="$2"
                shift 2
                ;;
            --exec-limit)
                EXEC_LIMIT="$2"
                shift 2
                ;;
            --exec-policy)
                EXEC_POLICY="$2"
                if [[ "$EXEC_POLICY" != "strict" && "$EXEC_POLICY" != "relaxed" ]]; then
                    error "Política de execução deve ser 'strict' ou 'relaxed'"
                    exit 1
                fi
                shift 2
                ;;
            --no-save-notebooks)
                SAVE_NOTEBOOKS="false"
                shift
                ;;
            --save-full-repos)
                DOWNLOAD_FULL_REPOS="true"
                shift
                ;;
            --repos-dir)
                REPOS_FULL_DIR="$2"
                shift 2
                ;;
            --skip-collect)
                SKIP_COLLECT="true"
                shift
                ;;
            --skip-execute)
                SKIP_EXECUTE="true"
                shift
                ;;
            --no-summary)
                RUN_SUMMARY="false"
                shift
                ;;
            --rebuild-images)
                REBUILD_IMAGES="true"
                shift
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            *)
                error "Opção desconhecida: $1"
                show_help
                exit 1
                ;;
        esac
    done

    # Validar argumentos obrigatórios
    if [[ -z "$DATE_START" || -z "$DATE_END" ]]; then
        error "As opções --date-start e --date-end são obrigatórias"
        show_help
        exit 1
    fi

    # Validar formato de datas
    if ! date -d "$DATE_START" &>/dev/null && ! date -j -f "%Y-%m-%d" "$DATE_START" &>/dev/null 2>&1; then
        error "Formato de data inválido para --date-start: $DATE_START (use YYYY-MM-DD)"
        exit 1
    fi
    if ! date -d "$DATE_END" &>/dev/null && ! date -j -f "%Y-%m-%d" "$DATE_END" &>/dev/null 2>&1; then
        error "Formato de data inválido para --date-end: $DATE_END (use YYYY-MM-DD)"
        exit 1
    fi
}

# Validação de pré-requisitos
check_prerequisites() {
    info "Verificando pré-requisitos..."

    # Verificar Docker
    if ! command -v docker &> /dev/null; then
        error "Docker não está instalado. Por favor, instale o Docker primeiro."
        exit 1
    fi
    if ! docker info &> /dev/null; then
        error "Docker não está rodando. Por favor, inicie o serviço Docker."
        exit 1
    fi
    success "Docker está instalado e rodando"

    # Verificar GITHUB_TOKEN
    if [[ -z "${GITHUB_TOKEN:-}" ]]; then
        error "GITHUB_TOKEN não está definido. Defina a variável de ambiente GITHUB_TOKEN."
        exit 1
    fi
    success "GITHUB_TOKEN está definido"

    # Verificar Python
    if ! command -v python3 &> /dev/null; then
        error "Python 3 não está instalado."
        exit 1
    fi
    PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
    PYTHON_MAJOR=$(echo "$PYTHON_VERSION" | cut -d. -f1)
    PYTHON_MINOR=$(echo "$PYTHON_VERSION" | cut -d. -f2)
    if [[ $PYTHON_MAJOR -lt 3 ]] || [[ $PYTHON_MAJOR -eq 3 && $PYTHON_MINOR -lt 13 ]]; then
        warning "Python 3.13+ é recomendado. Versão atual: $PYTHON_VERSION"
    else
        success "Python $PYTHON_VERSION está disponível"
    fi

    # Verificar diretórios
    mkdir -p "$DATA_OUTPUTS"
    mkdir -p "$LOGS_DIR"
    mkdir -p "$CONFIG_DIR"
    if [[ "$SAVE_NOTEBOOKS" == "true" ]]; then
        mkdir -p "$NOTEBOOKS_ORIG_DIR"
    fi
    if [[ "$DOWNLOAD_FULL_REPOS" == "true" ]]; then
        mkdir -p "$REPOS_FULL_DIR"
    fi
    success "Diretórios criados/verificados"

    info "Dados serão salvos em:"
    info "  - CSV coleta: $COLLECTION_CSV"
    info "  - CSV execução: $EXECUTION_CSV"
    if [[ "$SAVE_NOTEBOOKS" == "true" ]]; then
        info "  - Notebooks originais: $NOTEBOOKS_ORIG_DIR"
    else
        info "  - Notebooks originais: desabilitado (--no-save-notebooks)"
    fi
    if [[ "$DOWNLOAD_FULL_REPOS" == "true" ]]; then
        info "  - Repositórios completos: $REPOS_FULL_DIR"
    else
        info "  - Repositórios completos: desabilitado (use --save-full-repos)"
    fi

    # Verificar pimentel2019_reference.json
    if [[ ! -f "$PIMENTEL_REF" ]]; then
        warning "Arquivo de referência não encontrado: $PIMENTEL_REF"
        warning "O pipeline_master.ipynb pode não funcionar corretamente sem este arquivo"
    else
        success "Arquivo de referência encontrado: $PIMENTEL_REF"
    fi

    # Verificar se scripts Python existem
    if [[ ! -f "$SCRIPT_DIR/collect_notebooks.py" ]]; then
        error "Script não encontrado: $SCRIPT_DIR/collect_notebooks.py"
        exit 1
    fi
    if [[ ! -f "$SCRIPT_DIR/execute_notebook_docker.py" ]]; then
        error "Script não encontrado: $SCRIPT_DIR/execute_notebook_docker.py"
        exit 1
    fi
    if [[ ! -f "$SCRIPT_DIR/summarize_collection.py" ]]; then
        warning "Script não encontrado: $SCRIPT_DIR/summarize_collection.py (etapa de resumo será pulada)"
        RUN_SUMMARY="false"
    fi

    info "Todos os pré-requisitos foram verificados"
}

# Validar arquivo CSV
validate_csv() {
    local csv_file="$1"
    local min_lines="${2:-2}"  # Mínimo: cabeçalho + 1 linha

    if [[ ! -f "$csv_file" ]]; then
        error "Arquivo CSV não encontrado: $csv_file"
        return 1
    fi

    local line_count=$(wc -l < "$csv_file" | tr -d ' ')
    if [[ $line_count -lt $min_lines ]]; then
        error "Arquivo CSV tem poucas linhas ($line_count < $min_lines): $csv_file"
        return 1
    fi

    # Verificar se tem cabeçalho (primeira linha não vazia)
    if ! head -n 1 "$csv_file" | grep -q .; then
        error "Arquivo CSV parece estar vazio ou sem cabeçalho: $csv_file"
        return 1
    fi

    success "Arquivo CSV validado: $csv_file ($line_count linhas)"
    return 0
}

get_required_images() {
    if [[ ! -f "$COLLECTION_CSV" ]]; then
        echo "notebook-executor"
        return
    fi

    python3 - "$COLLECTION_CSV" <<'PY'
import csv, re, sys
from pathlib import Path

path = Path(sys.argv[1])

def image_for(version: str) -> str:
    fallback = "notebook-executor"
    if not version:
        return fallback
    match = re.match(r"\s*(\d+)(?:\.(\d+))?", version.strip())
    if not match:
        return fallback
    major = int(match.group(1))
    minor = int(match.group(2) or 0)
    if major == 2 and minor == 7:
        return "notebook-executor-py27"
    if major == 3:
        if minor == 5:
            return "notebook-executor-py35"
        if minor == 6:
            return "notebook-executor-py36"
        if minor == 8:
            return "notebook-executor-python38"
        if minor == 9:
            return "notebook-executor-python39"
        if minor == 10:
            return fallback
        if minor == 11:
            return "notebook-executor-python311"
        if minor == 12:
            return "notebook-executor-python312"
        if minor == 13:
            return "notebook-executor-py313"
    return fallback

images = set()
if path.exists():
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            version = (row.get("python_version_declared") or "").strip()
            images.add(image_for(version))

if not images:
    images.add("notebook-executor")

for image in sorted(images):
    print(image)
PY
}

ensure_docker_images() {
    if [[ "$REBUILD_IMAGES" == "true" ]]; then
        warning "Flag --rebuild-images ativa: todas as imagens necessárias serão reconstruídas agora."
    fi
    info "Verificando imagens Docker necessárias..."
    local -a required_images=()
    mapfile -t required_images < <(get_required_images)

    if [[ ${#required_images[@]} -eq 0 ]]; then
        required_images=("notebook-executor")
    fi

    for image in "${required_images[@]}"; do
        if [[ "$REBUILD_IMAGES" != "true" ]]; then
            if docker image inspect "$image" >/dev/null 2>&1; then
                info "Imagem disponível: $image"
                continue
            fi
        else
            info "Forçando rebuild da imagem $image"
        fi
        local dockerfile="${DOCKERFILE_MAP[$image]:-}"
        if [[ -z "$dockerfile" ]]; then
            warning "Imagem $image não possui Dockerfile conhecido. Disponibilize-a manualmente antes de prosseguir."
            continue
        fi
        local dockerfile_path="$DOCKERS_DIR/$dockerfile"
        if [[ ! -f "$dockerfile_path" ]]; then
            warning "Dockerfile não encontrado para $image: $dockerfile_path"
            continue
        fi
        info "Construindo imagem $image (Dockerfile: $dockerfile_path)..."
        # Usar DOCKER_BUILDKIT=0 para evitar problemas de credenciais
        # --pull=false usa cache local quando possível, evitando necessidade de autenticação
        if DOCKER_BUILDKIT=0 docker build --pull=false -t "$image" -f "$dockerfile_path" "$PROJECT_ROOT"; then
            success "Imagem construída: $image"
        else
            error "Falha ao construir a imagem $image"
            exit 1
        fi
    done
}

# Etapa 1: Coleta de notebooks
step_collect() {
    info "=== ETAPA 1: Coleta de Notebooks ==="

    if [[ "$SKIP_COLLECT" == "true" && -f "$COLLECTION_CSV" ]]; then
        warning "Pulando coleta (SKIP_COLLECT=true e collection.csv existe)"
        if validate_csv "$COLLECTION_CSV"; then
            return 0
        else
            error "collection.csv existe mas é inválido. Execute sem --skip-collect para regenerar."
            exit 1
        fi
    fi

    info "Executando collect_notebooks.py..."
    info "Parâmetros:"
    info "  --date-start: $DATE_START"
    info "  --date-end: $DATE_END"
    info "  --max-items: $MAX_ITEMS"
    info "  --output: $COLLECTION_CSV"
    if [[ "$DOWNLOAD_FULL_REPOS" == "true" ]]; then
        info "  --full-repos-dir: $REPOS_FULL_DIR"
    fi

    local collect_args=(
        "$SCRIPT_DIR/collect_notebooks.py"
        --date-start "$DATE_START"
        --date-end "$DATE_END"
        --max-items "$MAX_ITEMS"
        --output "$COLLECTION_CSV"
    )

    if [[ "$SAVE_NOTEBOOKS" == "true" ]]; then
        collect_args+=(--save-notebooks-dir "$NOTEBOOKS_ORIG_DIR")
        info "  --save-notebooks-dir: $NOTEBOOKS_ORIG_DIR"
    fi
    if [[ "$DOWNLOAD_FULL_REPOS" == "true" ]]; then
        collect_args+=(--download-full-repos --full-repos-dir "$REPOS_FULL_DIR")
    fi

    if ! python3 "${collect_args[@]}" 2>&1 | tee "$LOGS_DIR/collect_$(date +%Y%m%d_%H%M%S).log"; then
        error "Falha na coleta de notebooks"
        exit 1
    fi

    if ! validate_csv "$COLLECTION_CSV" 10; then
        error "collection.csv não foi gerado corretamente"
        exit 1
    fi

    success "Coleta de notebooks concluída: $COLLECTION_CSV"
}

# Etapa 2: Execução de notebooks
step_execute() {
    info "=== ETAPA 2: Execução de Notebooks ==="

    if [[ "$SKIP_EXECUTE" == "true" && -f "$EXECUTION_CSV" ]]; then
        warning "Pulando execução (SKIP_EXECUTE=true e execution_results.csv existe)"
        if validate_csv "$EXECUTION_CSV"; then
            return 0
        else
            error "execution_results.csv existe mas é inválido. Execute sem --skip-execute para regenerar."
            exit 1
        fi
    fi

    if [[ ! -f "$COLLECTION_CSV" ]]; then
        error "collection.csv não encontrado. Execute a etapa de coleta primeiro."
        exit 1
    fi

    ensure_docker_images

    info "Executando execute_notebook_docker.py..."
    info "Parâmetros:"
    info "  --input: $COLLECTION_CSV"
    info "  --output: $EXECUTION_CSV"
    info "  --timeout: $EXEC_TIMEOUT"
    info "  --policy: $EXEC_POLICY"

    local exec_args=(
        "$SCRIPT_DIR/execute_notebook_docker.py"
        --input "$COLLECTION_CSV"
        --output "$EXECUTION_CSV"
        --timeout "$EXEC_TIMEOUT"
        --policy "$EXEC_POLICY"
        --log-file "$LOGS_DIR/execute_$(date +%Y%m%d_%H%M%S).log"
    )

    if [[ "$SAVE_NOTEBOOKS" == "true" && -d "$NOTEBOOKS_ORIG_DIR" ]]; then
        exec_args+=(--originals-dir "$NOTEBOOKS_ORIG_DIR")
        info "  --originals-dir: $NOTEBOOKS_ORIG_DIR"
    fi

    if [[ "$DOWNLOAD_FULL_REPOS" == "true" && -d "$REPOS_FULL_DIR" ]]; then
        exec_args+=(--full-repos-dir "$REPOS_FULL_DIR")
        info "  --full-repos-dir: $REPOS_FULL_DIR"
    fi

    if [[ -n "$EXEC_LIMIT" ]]; then
        exec_args+=(--limit "$EXEC_LIMIT")
        info "  --limit: $EXEC_LIMIT"
    fi

    if ! python3 "${exec_args[@]}" 2>&1 | tee "$LOGS_DIR/execute_$(date +%Y%m%d_%H%M%S).log"; then
        error "Falha na execução de notebooks"
        exit 1
    fi

    if ! validate_csv "$EXECUTION_CSV" 10; then
        error "execution_results.csv não foi gerado corretamente"
        exit 1
    fi

    success "Execução de notebooks concluída: $EXECUTION_CSV"
}

# Etapa 3: Resumo e health checks
step_summary() {
    info "=== ETAPA 3: Resumo e Health Checks ==="

    if [[ "$RUN_SUMMARY" != "true" ]]; then
        warning "Pulando resumo (RUN_SUMMARY=false)"
        return 0
    fi

    if [[ ! -f "$SCRIPT_DIR/summarize_collection.py" ]]; then
        warning "summarize_collection.py não encontrado. Pulando resumo."
        return 0
    fi

    if [[ ! -f "$COLLECTION_CSV" ]]; then
        warning "collection.csv não encontrado. Pulando resumo."
        return 0
    fi

    info "Executando summarize_collection.py..."

    local summary_args=(
        "$SCRIPT_DIR/summarize_collection.py"
        --collection-csv "$COLLECTION_CSV"
    )

    if [[ -f "$EXECUTION_CSV" ]]; then
        summary_args+=(--exec-csv "$EXECUTION_CSV")
    fi

    if ! python3 "${summary_args[@]}" 2>&1 | tee "$LOGS_DIR/summary_$(date +%Y%m%d_%H%M%S).log"; then
        warning "Falha no resumo (não crítico, continuando...)"
        return 0
    fi

    success "Resumo concluído"
}

# Função principal
main() {
    info "=========================================="
    info "Pipeline de Coleta e Execução de Notebooks"
    info "=========================================="
    info "Diretório do projeto: $PROJECT_ROOT"
    info ""

    # Parsing de argumentos
    parse_args "$@"

    info "Política de execução selecionada: $EXEC_POLICY"
    if [[ "$EXEC_POLICY" == "strict" ]]; then
        info "  strict = instala dependências declaradas do repositório antes da execução (resultados mais fiéis, porém mais lentos)."
    else
        info "  relaxed = usa apenas a imagem base com pacotes essenciais (mais rápido, pode falhar se faltarem dependências)."
    fi

    # Validação de pré-requisitos
    check_prerequisites

    info ""
    info "Iniciando pipeline..."
    info ""

    # Executar etapas
    step_collect
    info ""
    step_execute
    info ""

    if [[ "$RUN_SUMMARY" == "true" ]]; then
        step_summary
        info ""
    fi

    # Resumo final
    info "=========================================="
    success "Pipeline concluído com sucesso!"
    info "=========================================="
    info ""
    info "Arquivos gerados:"
    if [[ -f "$COLLECTION_CSV" ]]; then
        local coll_lines=$(wc -l < "$COLLECTION_CSV" | tr -d ' ')
        info "  - $COLLECTION_CSV ($coll_lines linhas)"
    fi
    if [[ -f "$EXECUTION_CSV" ]]; then
        local exec_lines=$(wc -l < "$EXECUTION_CSV" | tr -d ' ')
        info "  - $EXECUTION_CSV ($exec_lines linhas)"
    fi
    if [[ "$DOWNLOAD_FULL_REPOS" == "true" ]]; then
        info "  - Repositórios completos em: $REPOS_FULL_DIR"
    fi
    info ""
    info "Próximos passos:"
    info "  1. Execute o pipeline_master.ipynb para análises RQ1, RQ2 e RQ3"
    info "  2. Verifique os logs em: $LOGS_DIR"
    info ""
}

# Executar função principal
main "$@"
