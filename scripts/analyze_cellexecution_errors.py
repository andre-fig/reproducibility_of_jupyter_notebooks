"""
Script para análise detalhada de erros CellExecutionError.

Extrai e categoriza os erros reais mascarados dentro de CellExecutionError.
"""

import csv
import re
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

def extract_real_error_from_cellexecutionerror(
    exception_str: str,
    traceback_str: str
) -> Tuple[Optional[str], Optional[str]]:
    """
    Extrai o tipo de erro real e mensagem de dentro de CellExecutionError.
    
    Returns:
        (error_type, error_message) ou (None, None) se não encontrado
    """
    if not exception_str and not traceback_str:
        return None, None
    
    combined = f"{exception_str or ''} {traceback_str or ''}".lower()
    
    # Padrões para identificar tipos de erro (ordem importa - mais específicos primeiro)
    error_patterns = [
        # Erros de arquivo
        (r"filenotfounderror\s*:\s*([^\n]+)", "file_not_found"),
        (r"\[errno\s+2\]\s+no\s+such\s+file\s+or\s+directory:\s*['\"]([^'\"]+)['\"]", "file_not_found"),
        (r"cannot\s+open\s+file\s+['\"]([^'\"]+)['\"]", "file_not_found"),
        (r"isadirectoryerror\s*:\s*([^\n]+)", "file_not_found"),
        (r"notadirectoryerror\s*:\s*([^\n]+)", "file_not_found"),
        # Erros de dependências
        (r"modulenotfounderror\s*:\s*no\s+module\s+named\s+['\"]([^'\"]+)['\"]", "missing_dependency"),
        (r"importerror\s*:\s*([^\n]+)", "missing_dependency"),
        (r"no\s+module\s+named\s+['\"]([^'\"]+)['\"]", "missing_dependency"),
        # Erros de sintaxe
        (r"syntaxerror\s*:\s*([^\n]+)", "syntax_error"),
        (r"indentationerror\s*:\s*([^\n]+)", "syntax_error"),
        # Timeouts
        (r"timeoutexpired\s*:\s*([^\n]+)", "timeout"),
        (r"timeouterror\s*:\s*([^\n]+)", "timeout"),
        (r"cell\s+execution\s+timeout", "timeout"),
        # Outros erros comuns
        (r"keyerror\s*:\s*['\"]([^'\"]+)['\"]", "key_error"),
        (r"attributeerror\s*:\s*([^\n]+)", "attribute_error"),
        (r"typeerror\s*:\s*([^\n]+)", "type_error"),
        (r"valueerror\s*:\s*([^\n]+)", "value_error"),
        (r"indexerror\s*:\s*([^\n]+)", "index_error"),
        (r"nameerror\s*:\s*name\s+['\"]([^'\"]+)['\"]", "name_error"),
        (r"zerodivisionerror\s*:\s*([^\n]+)", "zero_division_error"),
        (r"permissionerror\s*:\s*([^\n]+)", "permission_error"),
        (r"oserror\s*:\s*([^\n]+)", "os_error"),
        (r"connectionerror\s*:\s*([^\n]+)", "connection_error"),
        (r"httperror\s*:\s*([^\n]+)", "http_error"),
        (r"urlerror\s*:\s*([^\n]+)", "url_error"),
        (r"runtimeerror\s*:\s*([^\n]+)", "runtime_error"),
        (r"memoryerror\s*:\s*([^\n]+)", "memory_error"),
        (r"recursionerror\s*:\s*([^\n]+)", "recursion_error"),
    ]
    
    for pattern, error_type in error_patterns:
        match = re.search(pattern, combined, re.IGNORECASE | re.MULTILINE)
        if match:
            error_msg = match.group(1) if match.groups() else match.group(0)
            return error_type, error_msg[:200]  # Limitar tamanho da mensagem
    
    return None, None

def analyze_cellexecution_errors(
    master_csv_path: Path,
    output_csv_path: Path
) -> None:
    """
    Analisa erros CellExecutionError e gera CSV detalhado.
    """
    cellexecution_rows = []
    all_rows = []
    
    # Ler CSV principal
    with open(master_csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            all_rows.append(row)
            
            exec_exception_type = row.get("exec_exception_type", "").strip()
            if exec_exception_type == "CellExecutionError":
                cellexecution_rows.append(row)
    
    print(f"Total de notebooks: {len(all_rows)}")
    print(f"Notebooks com CellExecutionError: {len(cellexecution_rows)}")
    
    # Análise de padrões
    real_error_types = Counter()
    real_error_messages = defaultdict(list)
    repos_with_errors = defaultdict(int)
    
    detailed_data = []
    
    for row in cellexecution_rows:
        repo = row.get("repo_full_name", "")
        repos_with_errors[repo] += 1
        
        exception_str = row.get("exec_exception_str", "")
        traceback_str = row.get("exec_traceback_str", "")
        
        real_error_type, real_error_msg = extract_real_error_from_cellexecutionerror(
            exception_str, traceback_str
        )
        
        if real_error_type:
            real_error_types[real_error_type] += 1
            if real_error_msg:
                real_error_messages[real_error_type].append(real_error_msg[:100])
        else:
            real_error_types["unclassified"] += 1
        
        detailed_data.append({
            "notebook_id": row.get("notebook_id", ""),
            "repo_full_name": repo,
            "notebook_rel_path": row.get("notebook_rel_path", ""),
            "exec_error_type": row.get("exec_error_type", ""),
            "exec_exception_type": "CellExecutionError",
            "real_error_type": real_error_type or "unclassified",
            "real_error_message": real_error_msg or "",
            "exec_exception_str": exception_str[:500] if exception_str else "",
            "exec_traceback_str": traceback_str[:1000] if traceback_str else "",
            "exec_ok": row.get("exec_ok", ""),
            "elapsed_s": row.get("elapsed_s", ""),
            "exec_env_python_version": row.get("exec_env_python_version", "")
        })
    
    print(f"\nTipos de erro real encontrados dentro de CellExecutionError:")
    for error_type, count in real_error_types.most_common():
        print(f"  {error_type}: {count} ({count/len(cellexecution_rows)*100:.1f}%)")
    
    print(f"\nRepositórios com mais CellExecutionError:")
    for repo, count in sorted(repos_with_errors.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {repo}: {count} erros")
    
    # Exemplos de mensagens por tipo
    print(f"\nExemplos de mensagens de erro por tipo:")
    for error_type, messages in list(real_error_messages.items())[:5]:
        unique_messages = list(set(messages))[:3]
        print(f"  {error_type}:")
        for msg in unique_messages:
            print(f"    - {msg[:80]}...")
    
    # Escrever CSV detalhado
    if detailed_data:
        fieldnames = list(detailed_data[0].keys())
        with open(output_csv_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(detailed_data)
        
        print(f"\nCSV detalhado gerado: {output_csv_path}")
        print(f"Total de registros: {len(detailed_data)}")
    else:
        print("\nNenhum dado para gerar CSV detalhado")

def main():
    project_root = Path(__file__).parent.parent
    master_csv = project_root / "data" / "outputs" / "master_notebooks_merged.csv"
    output_csv = project_root / "data" / "outputs" / "cellexecution_error_analysis.csv"
    
    if not master_csv.exists():
        print(f"Erro: {master_csv} não encontrado")
        return
    
    analyze_cellexecution_errors(master_csv, output_csv)

if __name__ == "__main__":
    main()

