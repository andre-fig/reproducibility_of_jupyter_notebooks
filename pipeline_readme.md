# 📑 Pipeline de Coleta, Sumarização e Execução — `run_all.sh`

Este script automatiza **todo o fluxo** da pesquisa:  
1. Coleta metadados de notebooks públicos no GitHub.  
2. Consolida os CSVs mensais.  
3. Gera estatísticas descritivas.  
4. Executa os notebooks em ambiente controlado.  

---

## ⚙️ Pré-requisitos

- Python 3.10+  
- Token do GitHub com permissão de leitura pública:  
  ```bash
  export GITHUB_TOKEN=ghp_xxxxxxxxxxxxxxxxxxxxx
  ```
- Dependências do projeto já instaladas:  
  ```bash
  python -m venv .venv && source .venv/bin/activate
  pip install -r requirements.txt
  ```

---

## 🚀 Uso

### Exemplo simples
Coletar de **janeiro a setembro de 2025**:
```bash
START_DATE=2025-01-01 END_DATE=2025-09-28 ./scripts/run_all.sh
```

### Configurações opcionais
- **MAX_ITEMS**: limite aproximado de notebooks por mês (default = 2000).  
  Útil para rodadas de teste mais rápidas:
  ```bash
  MAX_ITEMS=200 START_DATE=2025-03-01 END_DATE=2025-03-31 ./scripts/run_all.sh
  ```

- **PYTHON**: binário Python a ser usado (default = `python3`).  
  Exemplo com virtualenv:
  ```bash
  PYTHON=.venv/bin/python START_DATE=2025-01-01 END_DATE=2025-01-31 ./scripts/run_all.sh
  ```

---

## 📦 Saídas

Na pasta `data/outputs/` você terá:

- **`notebooks_YYYY_MM_mon.csv`** → CSV de notebooks coletados em cada mês.  
- **`notebooks_STARTDATE_to_ENDDATE.csv`** → CSV consolidado com todos os meses.  
- **`summary_STARTDATE_to_ENDDATE.txt`** → estatísticas descritivas da coleta.  
- **`executions_STARTDATE_to_ENDDATE.csv`** → resultados de execução controlada.  
- **`logs/collect_*.log`** → logs de coleta por mês.  
- **`logs/exec_STARTDATE_to_ENDDATE.log`** → log da execução controlada.

---

## 🔒 Ética e Limites

- Apenas notebooks de **repositórios públicos**.  
- Coleta inicial só de **metadados**, sem execução imediata.  
- Execução controlada (nbclient/papermill) usa timeout, isolamento e política de dependências para reduzir riscos.  

---

## 🔄 Fluxo do pipeline

```text
┌───────────────┐      ┌─────────────────┐      ┌─────────────────────┐      ┌─────────────────────┐
│  Coleta mês a │ ───▶ │ Consolidação de │ ───▶ │ Sumarização         │ ───▶ │ Execução controlada │
│  mês (CSV)    │      │ CSVs            │      │ estatística (txt)   │      │ notebooks (CSV/log) │
└───────────────┘      └─────────────────┘      └─────────────────────┘      └─────────────────────┘
```