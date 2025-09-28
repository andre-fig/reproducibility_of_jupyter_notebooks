# Replicação — Coleta de Dados (Jupyter Notebooks Reproducibility)

Este diretório contém o script e os artefatos para **coletar metadados** de notebooks Jupyter públicos no GitHub, seguindo (e atualizando) a metodologia de Pimentel et al. (2019).

## ⚙️ Pré-requisitos

- Python 3.10+
- Um **token de acesso do GitHub** com permissão de leitura pública (`GITHUB_TOKEN`)
- Dependências do `requirements.txt`

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## 🚀 Uso

Coletar notebooks criados em 2025 (exemplo), limitando a ~2000 itens e salvando o CSV:

```bash
export GITHUB_TOKEN=ghp_XXXXXXXXXXXXXXXXXXXXXXXXXXXX
python scripts/collect_notebooks.py   --date-start 2025-01-01   --date-end   2025-09-28   --max-items  2000   --output     data/notebooks_metadata.csv
```

> Observação: o GitHub Search API retorna no máximo 1000 resultados por consulta. O script **divide automaticamente o intervalo de datas** para respeitar esse limite.

## 📦 Saída

Um CSV com **uma linha por notebook** e colunas como:

- Identificação do repositório, datas e estrelas
- Caminho do arquivo `.ipynb`
- Contagem de células por tipo, execução e outputs
- Métricas de ordem de execução (ambiguidade, _skips_, _out-of-order_)
- AST de código: imports, funções, classes, controle de fluxo, _testing_ hints
- Heurísticas de **uso/colagem por IA** (marcadores em markdown/código)
- Presença de arquivos de dependências (`requirements.txt`, `setup.py`, `Pipfile`)
- Heurística de **paths absolutos**

## 🧪 Validação / Qualidade

- JSON de notebook validado com `nbformat`
- Filtros para notebooks **Python** por padrão (pode desabilitar)
- _Backoff_ e _retry_ para lidar com _rate limits_
- Detecção de dependências varre a árvore completa do branch padrão

## 🔒 Ética e limites

- Apenas repositórios **públicos**
- Sem execução de notebooks neste estágio de coleta (apenas **metadados**)
- Dados sensíveis não são coletados
