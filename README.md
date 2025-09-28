# Analisador da Pesquisa Stack Overflow 2017 (Python)

Ferramenta em **Python** para processar o CSV da [Stack Overflow Developer Survey 2017](https://insights.stackoverflow.com/survey/), calcular métricas, gerar **gráficos** e produzir um **PDF** com todas as respostas organizadas.

---

## Funcionalidades

- Leitura do CSV da pesquisa de **2017** e normalização de dados
- **Q1** – Registra a ferramenta de análise utilizada (Python + libs)
- **Q2** – Estatísticas de `Salary`: média, máximo, mediana e 3º quartil
- **Q3** – Geração de **histograma** e **curva de densidade** de `Salary`
- **Q4** – Correlação (**Pearson r** e **p-valor**) entre `CareerSatisfaction` e `JobSatisfaction`
- **Q5** – Identifica o `Professional` com **maior média** de `CareerSatisfaction`
- **Q6** – Para quem escolheu **apenas uma** plataforma em `MobileDeveloperType`:
  - **média salarial** Android vs iOS e **teste t** com p-valor
- **Q7** – `PronounceGIF`:
  - **Boxplots** lado a lado; **ANOVA**; **Tukey HSD** (comparações múltiplas)
- **Q8** – `TabsSpaces`:
  - salário médio por grupo; discussão de **variáveis de confusão**;
  - **regressão OLS** controlando por `YearsProgram` (quando disponível);
  - tabela opcional de diferença média **por linguagem** (`HaveWorkedLanguage`, _Spaces − Tabs_)
- **Q9** – Análise extra sugerida: **remoto vs não remoto** via `HomeRemote` (histograma e teste t)
- Geração de **PDF** com todas as respostas a questões e figuras

**Colunas utilizadas (2017):**
`Salary`, `CareerSatisfaction`, `JobSatisfaction`, `Professional`, `MobileDeveloperType`, `PronounceGIF`, `TabsSpaces`, `YearsProgram`, `HaveWorkedLanguage`, `HomeRemote`.

---

## Pré-requisitos

- **Python 3.9+** instalado
- Arquivo `.env` configurado (ver exemplo acima)

---

## Estrutura do projeto

```
.
├── so_survey_2017_analysis.py
├── requirements.txt
├── .env
└── data/
    └── inputs/
        └── survey_results_public.csv
```

---

## Ambiente Virtual

1. Criar o ambiente:

```bash
python -m venv .venv
```

2. Ativar:

**Linux/Mac**

```bash
source .venv/bin/activate
```

**Windows (PowerShell)**

```bash
.\venv\Scripts\Activate
```

---

## Dependências

Instale via `requirements.txt`:

```bash
pip install -r requirements.txt
```

Se adicionar novas bibliotecas:

```bash
pip freeze > requirements.txt
```

---

## Como rodar

1. Configure `.env` com grupo e participantes.
2. Execute:

```bash
python so_survey_2017_analysis.py
```

Saídas em data/outputs

---
