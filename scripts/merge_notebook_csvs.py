#!/usr/bin/env python3
from __future__ import annotations
import argparse
import glob
import os
import sys
from typing import List

import pandas as pd

# Esquema canônico (na ordem do seu exemplo)
SCHEMA = [
    "repo_full_name","repo_id","repo_default_branch","repo_created_at","repo_stars",
    "file_path","file_sha","file_size","html_url","nb_ok_parse","kernel_name",
    "language","python_version_declared","n_cells_total","n_code","n_markdown",
    "n_raw","n_code_executed","percent_code_executed","has_unambiguous_order",
    "out_of_order","n_skips_total","n_skips_middle","max_execution_count",
    "n_cells_with_output","outputs_text","outputs_image","outputs_html_js",
    "outputs_error","outputs_formatted","outputs_ext","imports_total",
    "top_imports_json","has_local_imports","defines_function","defines_class",
    "has_control_flow","uses_testing_module","deps_requirements_txt","deps_setup_py",
    "deps_pipfile","deps_any","ai_marker_found","triple_backticks_in_code",
    "has_abs_data_path",
]

def read_csv_safe(path: str) -> pd.DataFrame:
    # Lê como string quando possível para evitar inflexões de tipo
    df = pd.read_csv(path, dtype=str, low_memory=False)
    # Normaliza colunas: adiciona as que faltam, na ordem do SCHEMA; mantém extras no final
    for col in SCHEMA:
        if col not in df.columns:
            df[col] = pd.NA
    # Reordena: primeiro SCHEMA (na ordem), depois quaisquer colunas extras
    extras = [c for c in df.columns if c not in SCHEMA]
    df = df[SCHEMA + extras]
    return df

def main():
    ap = argparse.ArgumentParser(description="Merge de CSVs semanais de notebooks em um único CSV.")
    ap.add_argument("--glob", default="data/outputs/notebooks_2025_*.csv",
                    help="Padrão de arquivos para ler (glob). Ex: data/outputs/notebooks_2025_*.csv")
    ap.add_argument("--output", default="data/outputs/notebooks_2025_all.csv",
                    help="Caminho do CSV final.")
    ap.add_argument("--dedup-keys", default="repo_full_name,file_path,file_sha",
                    help="Chaves para remover duplicatas, separadas por vírgula.")
    args = ap.parse_args()

    files: List[str] = sorted(glob.glob(args.glob))
    if not files:
        print(f"Nenhum arquivo encontrado com o padrão: {args.glob}", file=sys.stderr)
        sys.exit(1)

    print(f"Encontrados {len(files)} arquivos. Lendo e unindo...")
    dfs = []
    for f in files:
        try:
            df = read_csv_safe(f)
            df["__source_file"] = os.path.basename(f)
            dfs.append(df)
        except Exception as e:
            print(f"[WARN] Falha lendo {f}: {e}", file=sys.stderr)

    if not dfs:
        print("Nenhum CSV válido lido.", file=sys.stderr)
        sys.exit(2)

    big = pd.concat(dfs, ignore_index=True)

    # Deduplicação
    keys = [k.strip() for k in args.dedup_keys.split(",") if k.strip()]
    missing = [k for k in keys if k not in big.columns]
    if missing:
        print(f"[WARN] Chaves de deduplicação ausentes nas colunas: {missing}. "
              f"Pulando dedup.", file=sys.stderr)
    else:
        before = len(big)
        big = big.drop_duplicates(subset=keys, keep="first")
        after = len(big)
        print(f"Removidas {before - after} duplicatas (chaves: {keys}).")

    # Ordena para facilitar inspeção
    sort_cols = [c for c in ["repo_full_name", "file_path"] if c in big.columns]
    if sort_cols:
        big = big.sort_values(sort_cols, kind="stable").reset_index(drop=True)

    # Salva
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    big.to_csv(args.output, index=False)
    print(f"CSV final salvo em: {args.output} ({len(big)} linhas)")

if __name__ == "__main__":
    main()
