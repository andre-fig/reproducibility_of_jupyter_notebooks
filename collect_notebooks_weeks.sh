#!/bin/bash
# Script para rodar coleta semanal de notebooks (Jan - Set 2025)
set -euo pipefail
mkdir -p data/outputs/raw_ipynb

python3 scripts/collect_notebooks.py --date-start 2025-01-01 --date-end 2025-01-05 --max-items 10 --output data/outputs/notebooks_2025_jan_w1.csv --require-outputs --save-notebooks-dir data/outputs/raw_ipynb
python3 scripts/collect_notebooks.py --date-start 2025-01-06 --date-end 2025-01-12 --max-items 10 --output data/outputs/notebooks_2025_jan_w2.csv --require-outputs --save-notebooks-dir data/outputs/raw_ipynb
python3 scripts/collect_notebooks.py --date-start 2025-01-13 --date-end 2025-01-19 --max-items 9  --output data/outputs/notebooks_2025_jan_w3.csv --require-outputs --save-notebooks-dir data/outputs/raw_ipynb
python3 scripts/collect_notebooks.py --date-start 2025-01-20 --date-end 2025-01-26 --max-items 10 --output data/outputs/notebooks_2025_jan_w4.csv --require-outputs --save-notebooks-dir data/outputs/raw_ipynb
python3 scripts/collect_notebooks.py --date-start 2025-01-27 --date-end 2025-02-02 --max-items 9  --output data/outputs/notebooks_2025_jan_w5.csv --require-outputs --save-notebooks-dir data/outputs/raw_ipynb

python3 scripts/collect_notebooks.py --date-start 2025-02-03 --date-end 2025-02-09 --max-items 10 --output data/outputs/notebooks_2025_feb_w1.csv --require-outputs --save-notebooks-dir data/outputs/raw_ipynb
python3 scripts/collect_notebooks.py --date-start 2025-02-10 --date-end 2025-02-16 --max-items 9  --output data/outputs/notebooks_2025_feb_w2.csv --require-outputs --save-notebooks-dir data/outputs/raw_ipynb
python3 scripts/collect_notebooks.py --date-start 2025-02-17 --date-end 2025-02-23 --max-items 10 --output data/outputs/notebooks_2025_feb_w3.csv --require-outputs --save-notebooks-dir data/outputs/raw_ipynb
python3 scripts/collect_notebooks.py --date-start 2025-02-24 --date-end 2025-03-02 --max-items 10 --output data/outputs/notebooks_2025_feb_w4.csv --require-outputs --save-notebooks-dir data/outputs/raw_ipynb

python3 scripts/collect_notebooks.py --date-start 2025-03-03 --date-end 2025-03-09 --max-items 10 --output data/outputs/notebooks_2025_mar_w1.csv --require-outputs --save-notebooks-dir data/outputs/raw_ipynb
python3 scripts/collect_notebooks.py --date-start 2025-03-10 --date-end 2025-03-16 --max-items 9  --output data/outputs/notebooks_2025_mar_w2.csv --require-outputs --save-notebooks-dir data/outputs/raw_ipynb
python3 scripts/collect_notebooks.py --date-start 2025-03-17 --date-end 2025-03-23 --max-items 10 --output data/outputs/notebooks_2025_mar_w3.csv --require-outputs --save-notebooks-dir data/outputs/raw_ipynb
python3 scripts/collect_notebooks.py --date-start 2025-03-24 --date-end 2025-03-30 --max-items 9  --output data/outputs/notebooks_2025_mar_w4.csv --require-outputs --save-notebooks-dir data/outputs/raw_ipynb
python3 scripts/collect_notebooks.py --date-start 2025-03-31 --date-end 2025-04-06 --max-items 10 --output data/outputs/notebooks_2025_mar_w5.csv --require-outputs --save-notebooks-dir data/outputs/raw_ipynb

python3 scripts/collect_notebooks.py --date-start 2025-04-07 --date-end 2025-04-13 --max-items 9  --output data/outputs/notebooks_2025_apr_w1.csv --require-outputs --save-notebooks-dir data/outputs/raw_ipynb
python3 scripts/collect_notebooks.py --date-start 2025-04-14 --date-end 2025-04-20 --max-items 10 --output data/outputs/notebooks_2025_apr_w2.csv --require-outputs --save-notebooks-dir data/outputs/raw_ipynb
python3 scripts/collect_notebooks.py --date-start 2025-04-21 --date-end 2025-04-27 --max-items 9  --output data/outputs/notebooks_2025_apr_w3.csv --require-outputs --save-notebooks-dir data/outputs/raw_ipynb
python3 scripts/collect_notebooks.py --date-start 2025-04-28 --date-end 2025-05-04 --max-items 10 --output data/outputs/notebooks_2025_apr_w4.csv --require-outputs --save-notebooks-dir data/outputs/raw_ipynb

python3 scripts/collect_notebooks.py --date-start 2025-05-05 --date-end 2025-05-11 --max-items 9  --output data/outputs/notebooks_2025_may_w1.csv --require-outputs --save-notebooks-dir data/outputs/raw_ipynb
python3 scripts/collect_notebooks.py --date-start 2025-05-12 --date-end 2025-05-18 --max-items 10 --output data/outputs/notebooks_2025_may_w2.csv --require-outputs --save-notebooks-dir data/outputs/raw_ipynb
python3 scripts/collect_notebooks.py --date-start 2025-05-19 --date-end 2025-05-25 --max-items 9  --output data/outputs/notebooks_2025_may_w3.csv --require-outputs --save-notebooks-dir data/outputs/raw_ipynb
python3 scripts/collect_notebooks.py --date-start 2025-05-26 --date-end 2025-06-01 --max-items 10 --output data/outputs/notebooks_2025_may_w4.csv --require-outputs --save-notebooks-dir data/outputs/raw_ipynb

python3 scripts/collect_notebooks.py --date-start 2025-06-02 --date-end 2025-06-08 --max-items 10 --output data/outputs/notebooks_2025_jun_w1.csv --require-outputs --save-notebooks-dir data/outputs/raw_ipynb
python3 scripts/collect_notebooks.py --date-start 2025-06-09 --date-end 2025-06-15 --max-items 9  --output data/outputs/notebooks_2025_jun_w2.csv --require-outputs --save-notebooks-dir data/outputs/raw_ipynb
python3 scripts/collect_notebooks.py --date-start 2025-06-16 --date-end 2025-06-22 --max-items 10 --output data/outputs/notebooks_2025_jun_w3.csv --require-outputs --save-notebooks-dir data/outputs/raw_ipynb
python3 scripts/collect_notebooks.py --date-start 2025-06-23 --date-end 2025-06-29 --max-items 9  --output data/outputs/notebooks_2025_jun_w4.csv --require-outputs --save-notebooks-dir data/outputs/raw_ipynb
python3 scripts/collect_notebooks.py --date-start 2025-06-30 --date-end 2025-07-06 --max-items 10 --output data/outputs/notebooks_2025_jun_w5.csv --require-outputs --save-notebooks-dir data/outputs/raw_ipynb

python3 scripts/collect_notebooks.py --date-start 2025-07-07 --date-end 2025-07-13 --max-items 10 --output data/outputs/notebooks_2025_jul_w1.csv --require-outputs --save-notebooks-dir data/outputs/raw_ipynb
python3 scripts/collect_notebooks.py --date-start 2025-07-14 --date-end 2025-07-20 --max-items 9  --output data/outputs/notebooks_2025_jul_w2.csv --require-outputs --save-notebooks-dir data/outputs/raw_ipynb
python3 scripts/collect_notebooks.py --date-start 2025-07-21 --date-end 2025-07-27 --max-items 10 --output data/outputs/notebooks_2025_jul_w3.csv --require-outputs --save-notebooks-dir data/outputs/raw_ipynb
python3 scripts/collect_notebooks.py --date-start 2025-07-28 --date-end 2025-08-03 --max-items 9  --output data/outputs/notebooks_2025_jul_w4.csv --require-outputs --save-notebooks-dir data/outputs/raw_ipynb

python3 scripts/collect_notebooks.py --date-start 2025-08-04 --date-end 2025-08-10 --max-items 10 --output data/outputs/notebooks_2025_aug_w1.csv --require-outputs --save-notebooks-dir data/outputs/raw_ipynb
python3 scripts/collect_notebooks.py --date-start 2025-08-11 --date-end 2025-08-17 --max-items 9  --output data/outputs/notebooks_2025_aug_w2.csv --require-outputs --save-notebooks-dir data/outputs/raw_ipynb
python3 scripts/collect_notebooks.py --date-start 2025-08-18 --date-end 2025-08-24 --max-items 10 --output data/outputs/notebooks_2025_aug_w3.csv --require-outputs --save-notebooks-dir data/outputs/raw_ipynb
python3 scripts/collect_notebooks.py --date-start 2025-08-25 --date-end 2025-08-31 --max-items 9  --output data/outputs/notebooks_2025_aug_w4.csv --require-outputs --save-notebooks-dir data/outputs/raw_ipynb

python3 scripts/collect_notebooks.py --date-start 2025-09-01 --date-end 2025-09-07 --max-items 10 --output data/outputs/notebooks_2025_sep_w1.csv --require-outputs --save-notebooks-dir data/outputs/raw_ipynb
python3 scripts/collect_notebooks.py --date-start 2025-09-08 --date-end 2025-09-14 --max-items 9  --output data/outputs/notebooks_2025_sep_w2.csv --require-outputs --save-notebooks-dir data/outputs/raw_ipynb
python3 scripts/collect_notebooks.py --date-start 2025-09-15 --date-end 2025-09-21 --max-items 10 --output data/outputs/notebooks_2025_sep_w3.csv --require-outputs --save-notebooks-dir data/outputs/raw_ipynb
python3 scripts/collect_notebooks.py --date-start 2025-09-22 --date-end 2025-09-28 --max-items 9  --output data/outputs/notebooks_2025_sep_w4.csv --require-outputs --save-notebooks-dir data/outputs/raw_ipynb
python3 scripts/collect_notebooks.py --date-start 2025-09-29 --date-end 2025-09-30 --max-items 10 --output data/outputs/notebooks_2025_sep_w5.csv --require-outputs --save-notebooks-dir data/outputs/raw_ipynb
