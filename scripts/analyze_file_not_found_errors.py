"""
Script para análise exploratória de erros FileNotFoundError.

Extrai padrões dos erros, verifica estrutura de snapshots e gera CSV de análise.
"""

import csv
import json
import re
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Optional

def extract_missing_file_pattern(exception_str: str) -> Optional[str]:
    """
    Extrai padrão do arquivo faltante da mensagem de exceção.
    """
    if not exception_str:
        return None
    
    # Padrões comuns de FileNotFoundError
    patterns = [
        r"\[Errno 2\] No such file or directory: ['\"]([^'\"]+)['\"]",
        r"FileNotFoundError: \[Errno 2\] No such file or directory: ['\"]([^'\"]+)['\"]",
        r"cannot open file ['\"]([^'\"]+)['\"]",
        r"file ['\"]([^'\"]+)['\"] not found",
        r"pd\.read_csv\(['\"]([^'\"]+)['\"]",
        r"open\(['\"]([^'\"]+)['\"]",
        r"with open\(['\"]([^'\"]+)['\"]",
    ]
    
    for pattern in patterns:
        match = re.search(pattern, exception_str, re.IGNORECASE)
        if match:
            return match.group(1)
    
    return None

def analyze_file_not_found_errors(
    master_csv_path: Path,
    repos_completos_dir: Path,
    output_csv_path: Path
) -> None:
    """
    Analisa erros file_not_found e gera CSV exploratório.
    """
    file_not_found_rows = []
    all_rows = []
    snapshot_usage = {"completo": 0, "clone": 0, "nao_encontrado": 0}
    
    # Ler CSV principal
    with open(master_csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            all_rows.append(row)
            
            # Verificar uso de snapshot (baseado no log ou inferir)
            # Se exec_ok está vazio ou False e não há erro, pode ser snapshot não encontrado
            exec_error_type = row.get("exec_error_type", "").strip()
            
            if exec_error_type == "file_not_found":
                file_not_found_rows.append(row)
    
    print(f"Total de notebooks: {len(all_rows)}")
    print(f"Notebooks com erro file_not_found: {len(file_not_found_rows)}")
    
    # Análise de padrões
    missing_files = []
    repos_with_errors = defaultdict(int)
    error_patterns = Counter()
    
    for row in file_not_found_rows:
        repo = row.get("repo_full_name", "")
        repos_with_errors[repo] += 1
        
        exception_str = row.get("exec_exception_str", "")
        missing_file = extract_missing_file_pattern(exception_str)
        
        if missing_file:
            missing_files.append({
                "repo": repo,
                "notebook": row.get("notebook_rel_path", ""),
                "missing_file": missing_file,
                "exception": exception_str[:200] if exception_str else ""
            })
            
            # Classificar tipo de arquivo
            if missing_file.endswith('.csv'):
                error_patterns['csv'] += 1
            elif missing_file.endswith('.json'):
                error_patterns['json'] += 1
            elif missing_file.endswith(('.png', '.jpg', '.jpeg', '.gif')):
                error_patterns['image'] += 1
            elif missing_file.endswith('.txt'):
                error_patterns['txt'] += 1
            elif 'data' in missing_file.lower():
                error_patterns['data_dir'] += 1
            else:
                error_patterns['other'] += 1
    
    print(f"\nRepositórios com mais erros file_not_found:")
    for repo, count in sorted(repos_with_errors.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {repo}: {count} erros")
    
    print(f"\nTipos de arquivos faltantes:")
    for file_type, count in error_patterns.most_common():
        print(f"  {file_type}: {count}")
    
    # Verificar estrutura de snapshots para repositórios problemáticos
    print(f"\nVerificando estrutura de snapshots...")
    snapshot_analysis = []
    
    for repo, count in sorted(repos_with_errors.items(), key=lambda x: x[1], reverse=True)[:5]:
        owner, repo_name = repo.split('/', 1) if '/' in repo else (repo, "")
        repo_base = repos_completos_dir / owner / repo_name if repo_name else repos_completos_dir / owner
        
        if repo_base.exists():
            snapshots = list(repo_base.glob("*"))
            snapshot_dirs = [s for s in snapshots if s.is_dir()]
            
            if snapshot_dirs:
                snapshot = snapshot_dirs[0]  # Usar primeiro snapshot encontrado
                
                # Listar estrutura de diretórios
                data_dirs = list(snapshot.rglob("data"))
                csv_files = list(snapshot.rglob("*.csv"))
                
                snapshot_analysis.append({
                    "repo": repo,
                    "snapshot_path": str(snapshot),
                    "has_data_dir": len(data_dirs) > 0,
                    "data_dirs": [str(d.relative_to(snapshot)) for d in data_dirs[:5]],
                    "csv_count": len(csv_files),
                    "csv_files_sample": [str(f.relative_to(snapshot)) for f in csv_files[:5]]
                })
    
    # Gerar CSV exploratório
    exploratory_data = []
    
    for row in file_not_found_rows:
        exception_str = row.get("exec_exception_str", "")
        missing_file = extract_missing_file_pattern(exception_str)
        
        # Verificar se snapshot existe
        repo = row.get("repo_full_name", "")
        owner, repo_name = repo.split('/', 1) if '/' in repo else (repo, "")
        repo_default_branch_sha = row.get("repo_default_branch_sha", "")
        
        snapshot_found = False
        snapshot_path = ""
        snapshot_structure_sample = ""
        
        if repo_name and repo_default_branch_sha:
            repo_base = repos_completos_dir / owner / repo_name
            if repo_base.exists():
                # Procurar snapshot pelo SHA
                sha_prefix = repo_default_branch_sha[:40]
                candidates = list(repo_base.glob(f"{sha_prefix}*"))
                dirs = [c for c in candidates if c.is_dir()]
                
                if dirs:
                    snapshot = dirs[0]
                    snapshot_found = True
                    snapshot_path = str(snapshot)
                    
                    # Amostra de estrutura (primeiros 10 arquivos/diretórios)
                    try:
                        items = list(snapshot.iterdir())[:10]
                        snapshot_structure_sample = ", ".join([item.name for item in items])
                    except Exception:
                        snapshot_structure_sample = "erro_ao_listar"
        
        exploratory_data.append({
            "notebook_id": row.get("notebook_id", ""),
            "repo_full_name": repo,
            "notebook_rel_path": row.get("notebook_rel_path", ""),
            "snapshot_found": str(snapshot_found),
            "snapshot_path": snapshot_path,
            "repo_default_branch_sha": repo_default_branch_sha,
            "exec_error_type": "file_not_found",
            "exec_exception_str": exception_str[:500] if exception_str else "",
            "missing_file_pattern": missing_file if missing_file else "",
            "snapshot_structure_sample": snapshot_structure_sample,
            "exec_ok": row.get("exec_ok", ""),
            "elapsed_s": row.get("elapsed_s", ""),
            "exec_env_python_version": row.get("exec_env_python_version", "")
        })
    
    # Escrever CSV exploratório
    if exploratory_data:
        fieldnames = list(exploratory_data[0].keys())
        with open(output_csv_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(exploratory_data)
        
        print(f"\nCSV exploratório gerado: {output_csv_path}")
        print(f"Total de registros: {len(exploratory_data)}")
    else:
        print("\nNenhum dado para gerar CSV exploratório")

def main():
    project_root = Path(__file__).parent.parent
    master_csv = project_root / "data" / "outputs" / "master_notebooks_merged.csv"
    repos_dir = project_root / "data" / "outputs" / "repositorios_completos"
    output_csv = project_root / "data" / "outputs" / "file_not_found_analysis.csv"
    
    if not master_csv.exists():
        print(f"Erro: {master_csv} não encontrado")
        return
    
    if not repos_dir.exists():
        print(f"Aviso: {repos_dir} não encontrado. Análise de snapshots limitada.")
    
    analyze_file_not_found_errors(master_csv, repos_dir, output_csv)

if __name__ == "__main__":
    main()

