#!/usr/bin/env bash
set -euo pipefail

# === Config ===
OUTDIR="data/outputs"
LOGDIR="${OUTDIR}/logs"
MAX_ITEMS=2000    # ajuste se quiser limitar por mês
PYTHON="python3"  # ou .venv/bin/python

# === Checks ===
if [[ -z "${GITHUB_TOKEN:-}" ]]; then
  echo "ERRO: defina GITHUB_TOKEN antes de rodar (export GITHUB_TOKEN=...)" >&2
  exit 1
fi

mkdir -p "${OUTDIR}" "${LOGDIR}"

# === Jan -> Set 2025 (mês a mês) ===
declare -a STARTS=(
  "2025-01-01"
  "2025-02-01"
  "2025-03-01"
  "2025-04-01"
  "2025-05-01"
  "2025-06-01"
  "2025-07-01"
  "2025-08-01"
  "2025-09-01"
)

declare -a ENDS=(
  "2025-01-31"
  "2025-02-28"
  "2025-03-31"
  "2025-04-30"
  "2025-05-31"
  "2025-06-30"
  "2025-07-31"
  "2025-08-31"
  "2025-09-28"  # até a data atual do estudo
)

declare -a TAGS=(
  "2025_01_jan"
  "2025_02_fev"
  "2025_03_mar"
  "2025_04_abr"
  "2025_05_mai"
  "2025_06_jun"
  "2025_07_jul"
  "2025_08_ago"
  "2025_09_set_ate_28"
)

for i in "${!STARTS[@]}"; do
  S="${STARTS[$i]}"
  E="${ENDS[$i]}"
  T="${TAGS[$i]}"

  OUT="${OUTDIR}/notebooks_${T}.csv"
  LOG="${LOGDIR}/collect_${T}.log"

  echo "==> Coletando ${S}..${E} -> ${OUT}"
  # Tenta até 3 vezes com backoff simples
  for attempt in 1 2 3; do
    if ${PYTHON} scripts/collect_notebooks.py \
        --date-start "${S}" \
        --date-end   "${E}" \
        --max-items  ${MAX_ITEMS} \
        --output     "${OUT}" \
        2>&1 | tee "${LOG}"; then
      break
    fi
    echo "Falha (tentativa ${attempt}) — aguardando antes de retry..." >&2
    sleep $(( 10 * attempt ))
  done
done

# === Consolidação: notebooks_2025_all.csv ===
ALL="${OUTDIR}/notebooks_2025_all.csv"
echo "==> Consolidando CSVs mensais em ${ALL}"
# Junta todos os CSVs do período (ordem cronológica pelos TAGS)
awk 'FNR==1 && NR!=1 {next} {print}' \
  "${OUTDIR}"/notebooks_2025_01_jan.csv \
  "${OUTDIR}"/notebooks_2025_02_fev.csv \
  "${OUTDIR}"/notebooks_2025_03_mar.csv \
  "${OUTDIR}"/notebooks_2025_04_abr.csv \
  "${OUTDIR}"/notebooks_2025_05_mai.csv \
  "${OUTDIR}"/notebooks_2025_06_jun.csv \
  "${OUTDIR}"/notebooks_2025_07_jul.csv \
  "${OUTDIR}"/notebooks_2025_08_ago.csv \
  "${OUTDIR}"/notebooks_2025_09_set_ate_28.csv \
  > "${ALL}"

echo "OK. Linhas totais:"
wc -l "${ALL}"

echo "Dica: rode o summarizer para estatísticas:"
echo "${PYTHON} scripts/summarize_collection.py --collection-csv ${ALL} | tee ${OUTDIR}/summary_2025_all.txt"
