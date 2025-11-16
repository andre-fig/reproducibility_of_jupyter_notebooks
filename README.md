# Replicação 

Este repositório documenta o pipeline empregado baseando-se no estudo de reprodutibilidade de notebooks de Pimentel et al. (2019), atualizado para o cenário de 2025. Toda a replicação deve ser feita via `scripts/run_pipeline.sh`, que automatiza a coleta, execução controlada e sumarização dos notebooks públicos analisados. As análises (RQs e exploração) devem ser conduzidas com `scripts/pipeline_master.ipynb`, utilizando os dados consolidados em `data/outputs/`.

---

## 1. Pré-requisitos e preparação

- Linux ou WSL2 com Docker ativo
- Python **3.13+** instalado localmente
- Token GitHub com permissão **read-only** (`GITHUB_TOKEN`)
- Dependências listadas em `requirements.txt`

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
mkdir -p data/outputs
```

Configure o token:

```bash
export GITHUB_TOKEN=ghp_xxxxxxxxxxxxxxxxxxxxx
```

Recomendações metodológicas:

- Use ambientes isolados por repositório (a execução já aplica isso via Docker).
- Monitore `data/outputs/logs` para garantir rastreabilidade.
- Nunca edite os arquivos em `data/outputs/` manualmente sem documentar.

---

## 2. Replicação com `scripts/run_pipeline.sh`

Esse script é a entrada única para todo o pipeline. Ele inclui:

1. **Coleta (`collect_notebooks.py`)** — busca notebooks com filtros temporais e salva `data/outputs/collection.csv`.
2. **Execução controlada (`execute_notebook_docker.py`)** — executa notebooks em contêineres específicos por versão de Python e gera `data/outputs/execution_results.csv`.
3. **Resumo (`summarize_collection.py`)** — executa validações e health checks, gerando relatórios em `data/outputs/logs/`.

### Execução básica

```bash
GITHUB_TOKEN=ghp_xxx \
bash scripts/run_pipeline.sh \
  --date-start 2025-01-01 \
  --date-end   2025-01-31 \
  --max-items  400 \
  --exec-policy strict \
  --exec-limit 120
```

Principais parâmetros:

- `--date-start`, `--date-end`: intervalo obrigatório (UTC) para busca por `created:`.
- `--max-items`: limite de notebooks coletados (default 400).
- `--exec-policy`: `strict` (instala dependências do repo) ou `relaxed`.
- `--exec-timeout`: timeout em segundos (default 300).
- `--exec-limit`: restringe quantos notebooks executar (útil para amostras).
- `--save-full-repos`: baixa o repositório completo e reduz erros `file_not_found`.
- `--skip-collect`, `--skip-execute`, `--no-summary`: permitem retomar etapas.

Variáveis de ambiente úteis:

- `MAX_ITEMS`, `EXEC_TIMEOUT`, `EXEC_POLICY`, `EXEC_LIMIT`
- `SAVE_NOTEBOOKS`, `DOWNLOAD_FULL_REPOS`
- `REBUILD_IMAGES` para forçar rebuild das imagens Docker descritas no script

Saídas esperadas (todas em `data/outputs/`):

- `collection.csv`: metadados e métricas de cada notebook.
- `execution_results.csv`: diagnóstico de execução notebook a notebook.
- `logs/*.log`: log completo das etapas para auditoria.
- `notebooks_originais/` (opcional): cópias `.ipynb` obtidas da coleta.
- `repositorios_completos/` (quando `--save-full-repos`).

Caso o script detecte inconsistências (CSV incompleto, ausência de token, Docker inativo), ele interrompe com mensagens claras para correção.

---

## 3. Análises com `scripts/pipeline_master.ipynb`

Após o pipeline, abra `scripts/pipeline_master.ipynb` para reproduzir:

- **RQ1–RQ3**: qualidade, erros e fatores associados.
- **RQ6–RQ8**: correlações, distribuição de falhas e comparação com literatura.
- **Análise exploratória**: consolida métricas adicionais em `execution_exploratory.csv`.

Orientações:

- Use o mesmo ambiente virtual que contém as dependências do projeto.
- Atualize os caminhos internos apenas se mover os arquivos; por padrão, o notebook lê diretamente de `data/outputs/collection.csv` e `data/outputs/execution_results.csv`.
- Execute as células na ordem, documentando quaisquer modificações metodológicas.
- Gere as versões exportadas (`pipeline_master.html` / `pipeline_master.pdf`) para registro dos resultados.

---

## 4. Dados de pesquisa em `data/outputs/`

Todo o material usado no artigo está em `data/outputs/`. Principais artefatos:

- **CSV mestre de coleta**: `collection.csv` (metadados brutos).
- **Resultados de execução**: `execution_results.csv`, `execution_exploratory.csv`.
- **Consolidações e métricas**:
  - `master_notebooks_merged.csv`: junção pronta para análises.
  - `cellexecution_error_analysis.csv`, `error_*` CSVs: diagnósticos de exceções.
  - `file_not_found_analysis.csv`: casos de paths quebrados.
  - `rq*_*.csv`: tabelas finais usadas nas seções do artigo (RQ1–RQ8).
- **Representações gráficas e logs**:
  - `figures/`: imagens exportadas das análises.
  - `logs/` e `old_logs/`: rastreabilidade de execuções anteriores.
  - `result_json/`: registros estruturados por notebook/repositório.
- **Documentos finais**: `pipeline_master.html`, `pipeline_master.pdf`.
- **Material bruto**: `notebooks_originais/` (cópias dos `.ipynb`) e `repositorios_completos/` (tarballs dos repositórios quando habilitado).

Nunca altere esses arquivos manualmente; gere novas versões via `run_pipeline.sh` e **guarde os logs** para comprovar as condições de execução.

---

## 5. Boas práticas e troubleshooting

- Valide sempre se `collection.csv` e `execution_results.csv` têm >=10 linhas antes de prosseguir.
- Em casos de `rate limit`, o script aplica _retry/backoff_, mas você pode reduzir `--max-items` para janelas menores.
- Para erros massivos de dependência, prefira `--exec-policy strict` ou habilite `--save-full-repos`.
- Execute novamente apenas etapas necessárias com `--skip-collect` ou `--skip-execute` para economizar tempo.
- Registre versões de Docker e Python utilizadas no relatório para assegurar comparabilidade longitudinal.

Com esses passos, qualquer pesquisador consegue replicar o pipeline end-to-end, reexecutar as análises no notebook mestre e validar os dados já publicados em `data/outputs/`.
