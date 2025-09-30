# Replicação — Coleta e Execução (Jupyter Notebooks Reproducibility)

Este diretório contém scripts e artefatos para **coletar metadados** e **executar notebooks** Jupyter públicos no GitHub, seguindo (e atualizando) a metodologia de Pimentel et al. (2019). Inclui também um **summarizer** para estatísticas descritivas.

## ⚙️ Pré-requisitos

- Python 3.10+
- Um **token do GitHub** com leitura pública (`GITHUB_TOKEN`)
- Dependências do `requirements.txt`

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Crie a pasta de saídas:

```bash
mkdir -p data/outputs
```

## 🔑 Token do GitHub

```bash
export GITHUB_TOKEN=ghp_XXXXXXXXXXXXXXXXXXXXXXXXXXXX
```

> Dica: use token **só de leitura pública**. Se vazou, **revogue** no GitHub.

---

## 🚀 1) Coleta de metadados (sem executar notebooks)

Script: `scripts/collect_notebooks.py`  
Descobre repositórios por `created:` (GitHub Search **repositories**), varre a árvore do branch padrão e processa cada `.ipynb` (sem clonar o repo).

### Exemplos de uso

**Jan–Set/2025 (exemplo, ~2000 notebooks):**

```bash
python scripts/collect_notebooks.py \
  --date-start 2025-01-01 \
  --date-end   2025-09-28 \
  --max-items  1000 \
  --output     data/outputs/notebooks_2025_jan-set.csv
```

```bash
python3 scripts/collect_notebooks.py \
 --date-start 2025-01-01 --date-end 2025-01-31 \
 --max-items 200 --output data/notebooks_jan.csv \
 --require-outputs
```

**Janela curta (sanidade):**

```bash
python scripts/collect_notebooks.py \
  --date-start 2025-03-01 \
  --date-end   2025-03-07 \
  --max-items  100 \
  --output     data/outputs/notebooks_2025_mar_1w.csv
```

**Coleta “histórica” (comparabilidade 2019):**

```bash
python scripts/collect_notebooks.py \
  --date-start 2019-01-01 \
  --date-end   2019-01-05 \
  --max-items  200 \
  --output     data/outputs/notebooks_2019_jan_1w.csv
```

> O script divide automaticamente o intervalo para respeitar o limite de **1000 resultados por consulta** da API de busca.

### Saída

CSV com **uma linha por notebook** contendo:

- Identificação do repositório (`repo_full_name`, `repo_stars`, datas)
- Caminho do `.ipynb` no repositório e URL
- Contagem de células por tipo; execução e outputs
- Métricas de ordem de execução (ambiguidade, _skips_, _out-of-order_)
- AST de código: imports, funções, classes, controle de fluxo, _testing_ hints
- Heurísticas de **IA** (marcadores em markdown/código)
- Presença de arquivos de dependências (`requirements.txt`, `setup.py`, `Pipfile`)
- Heurística de **paths absolutos**

Validações embutidas:

- `nbformat` (parse do JSON)
- Filtro por **Python** (padrão)
- _Retry/backoff_ para _rate limits_
- _Fallback_ para `download_url` em arquivos grandes (LFS)

---

## 🧪 2) Executor controlado (nbclient) — **opcional nesta fase**

Script: `scripts/execute_notebooks.py`  
Clona cada repositório (shallow), cria **env isolado por repositório** (venv), aplica política de dependências e **executa** o notebook com **timeout**.

Políticas:

- `--policy relaxed` (padrão): instala baseline mínima (pip, wheel, nbclient, ipykernel).
- `--policy strict`: tenta instalar `requirements.txt` / `Pipfile` / `setup.py`/`pyproject` do repositório.

```bash
python scripts/execute_notebooks.py \
  --input-csv  data/outputs/notebooks_2025_mar_1w.csv \
  --output-csv data/outputs/execution_results_2025_mar_1w.csv \
  --policy     strict \
  --timeout    30 \
  --limit      50
```

**Saída (`data/outputs/execution_results_*.csv`):**

- `repo_full_name`, `repo_default_branch`, `file_path`
- Metadados relevantes (`kernel_name`, `language`, `python_version_declared`)
- **Resultados de execução**: `exec_ok`, `error`, `exc_name`, `failed_cell_index`, `elapsed_s`

> Observação: esta etapa é **computacionalmente cara** e comparável ao desenho de Pimentel et al. (ambiente limpo + timeout + ordem de execução do notebook).

---

## 📊 3) Summarizer (estatísticas descritivas)

Script: `scripts/summarize_collection.py`  
Lê o CSV de coleta (e opcionalmente o CSV de execução) e imprime estatísticas descritivas.

```bash
python scripts/summarize_collection.py \
  --collection-csv data/outputs/notebooks_2025_mar_1w.csv \
  --exec-csv       data/outputs/execution_results_2025_mar_1w.csv \
  | tee data/outputs/summary_2025_mar_1w.txt
```

Produz (exemplos):

- Contagens e percentuais: `nb_ok_parse=True`, `language=python`, `deps_any=True`, `has_unambiguous_order=True`
- Mediana/média de `n_code`, `n_markdown`, `% code executed`
- **Top imports** agregados (a partir de `top_imports_json`)
- Se execuções fornecidas: taxa de sucesso e erros mais comuns

---

## 🧹 Boas práticas / Qualidade

- **Garbage in, garbage out**: use `nb_ok_parse=True` e `language=python` para análises comparáveis a 2019.
- Para _reprodutibilidade_, rode o executor com política **estrita** e relacione erros a `deps_*`, _ordem de execução_, _paths absolutos_, etc.
- Faça janelas menores (semanas/meses) e **consolide**:

```bash
# exemplo de concatenação mantendo apenas um cabeçalho
awk 'FNR==1 && NR!=1 {next} {print}' data/outputs/notebooks_2025_*.csv > data/outputs/notebooks_2025_all.csv
```

---

## 🔒 Ética e limites

- Apenas repositórios **públicos**
- Coleta é **somente metadados**; execução é opcional e controlada
- Não coletamos dados sensíveis; respeitamos _rate limits_ e termos da API

---

### Troubleshooting rápido

- **Mostra “Coletando notebooks: 0it”** → você está na versão antiga (busca por **código** com `created:`). A correta mostra **“Varredura de repositórios”**.
- **`nb_ok_parse=False` em muitos casos** → notebooks grandes (LFS). Garanta que seu coletor está com o **fallback para `download_url`**.
- **`TypeError: expected string, got list`** → normalizamos `cell.source` com helper `as_text(...)`.
