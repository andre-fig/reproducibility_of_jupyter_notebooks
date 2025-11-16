"""
Health check script for collection and execution data.

This script performs basic health checks and descriptive statistics on the collected
and executed notebooks. It does NOT perform RQ analysis - that is done in pipeline_master.ipynb.

Dependency metrics note:
- `repo_has_*` fields capture the presence of concrete environment/spec files in the repo tree.
- `deps_*` fields are convenient aliases derived from the same detection (e.g., `deps_any` is true
  whenever any of requirements/setup/pipfile exist). They coexist for quick summaries vs. richer views.

Main functions:
- Health checks: notebook counts, parsing stats, basic execution stats
- Basic reproducibility sanity check (labeled as such, not final results)
- Merge collection and execution CSVs into master_notebooks_merged.csv
"""

import argparse
import csv
import json
import statistics as stats
from collections import Counter
from pathlib import Path

# ---------- utils ----------
def load_csv(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for r in rd:
            rows.append(r)
    return rows

def pct(a, b): 
    return 0.0 if not b else 100.0*a/b

def to_int(x, d=0):
    try:
        return int(float(x))
    except (ValueError, TypeError):
        return d

def to_float(x, d=0.0):
    try:
        return float(x)
    except (ValueError, TypeError):
        return d

def to_bool(x):
    if x is None:
        return False
    s = str(x).strip().lower()
    return s in ("true","1","yes","y","t")

def save_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

def load_pimentel_reference(path="config/pimentel2019_reference.json"):
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if "metrics" not in data:
            raise ValueError("missing 'metrics' key")
        return data
    except Exception as e:
        print(f"WARNING: could not load Pimentel reference from {path}: {e}")
        return {"metrics": {}}

def merge_collection_and_exec(coll_rows, exec_rows, out_dir="data/outputs"):
    """
    Two-stage join:
      1) by notebook_id
      2) fallback by repo_full_name|notebook_rel_path
    """
    coll_by_id = {}
    for r in coll_rows:
        nid = r.get("notebook_id") or ""
        if nid and nid not in coll_by_id:
            coll_by_id[nid] = r
    merged = []
    left_only = []
    right_only = []

    exec_seen_ids = set()
    for ex in exec_rows:
        nid = ex.get("notebook_id") or ""
        if nid and nid in coll_by_id:
            merged.append({**coll_by_id[nid], **ex})
            exec_seen_ids.add(nid)
        else:
            right_only.append(ex)

    coll_ids = set(r.get("notebook_id") or "" for r in coll_rows)
    for c in coll_rows:
        nid = c.get("notebook_id") or ""
        if nid and nid not in exec_seen_ids:
            left_only.append(c)

    coll_by_key = {}
    for c in left_only:
        key = f"{c.get('repo_full_name','')}|{c.get('notebook_rel_path') or c.get('file_path','')}"
        if key not in coll_by_key:
            coll_by_key[key] = c
    recovered = 0
    still_right = []
    for ex in right_only:
        key = f"{ex.get('repo_full_name','')}|{ex.get('notebook_rel_path') or ex.get('file_path','')}"
        if key in coll_by_key:
            merged.append({**coll_by_key[key], **ex})
            recovered += 1
        else:
            still_right.append(ex)

    print("\nJoin report:")
    print(f"  matched via notebook_id: {len(merged) - recovered}")
    print(f"  recovered via fallback:  {recovered}")
    print(f"  left_only (collection only): {len(left_only)}")
    print(f"  right_only remaining (execution only): {len(still_right)}")

    outp = Path(out_dir) / "master_notebooks_merged.csv"
    outp.parent.mkdir(parents=True, exist_ok=True)
    save_csv(str(outp), merged)
    print(f"Master merged CSV written: {outp} ({len(merged)} rows)")
    return merged
# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--collection-csv", required=True)
    ap.add_argument("--exec-csv", required=False)
    args = ap.parse_args()

    # ---------- coleção ----------
    coll = load_csv(args.collection_csv)
    n = len(coll)
    print(f"# Notebooks (linhas): {n}")

    parsed = [r for r in coll if to_bool(r.get("nb_ok_parse"))]
    py = [r for r in coll if (r.get("language") or "").lower().startswith("python")]
    deps_any = [r for r in coll if to_bool(r.get("deps_any"))]
    env_spec = [r for r in coll if to_bool(r.get("repo_has_env_spec"))]
    unamb = [r for r in coll if to_bool(r.get("nb_has_unambiguous_exec_order"))]

    print(f"nb_ok_parse=True: {len(parsed)} ({pct(len(parsed), n):.1f}%)")
    print(f"language=python: {len(py)} ({pct(len(py), n):.1f}%)")
    print(f"deps_any=True (alguma especificação simples de deps): {len(deps_any)} ({pct(len(deps_any), n):.1f}%)")
    print(f"repo_has_env_spec=True (qualquer spec reconhecida): {len(env_spec)} ({pct(len(env_spec), n):.1f}%)")
    print(f"nb_has_unambiguous_exec_order=True: {len(unamb)} ({pct(len(unamb), n):.1f}%)")

    repo_dep_fields = [
        "repo_has_requirements_txt",
        "repo_has_setup_py",
        "repo_has_pipfile",
        "repo_has_environment_yml",
        "repo_has_lockfile",
    ]
    for field in repo_dep_fields:
        count = sum(1 for r in coll if to_bool(r.get(field)))
        print(f"{field}: {count} ({pct(count, n):.1f}%)")

    n_code = [to_int(r.get("n_code", 0)) for r in coll if r.get("n_code") not in ("", None)]
    n_md = [to_int(r.get("n_markdown", 0)) for r in coll if r.get("n_markdown") not in ("", None)]
    perc_exec = [to_float(r.get("percent_code_executed", 0.0)) for r in coll if r.get("percent_code_executed") not in ("", None)]
    if n_code:
        print(f"n_code: median={stats.median(n_code):.0f}, mean={stats.mean(n_code):.2f}")
    if n_md:
        print(f"n_markdown: median={stats.median(n_md):.0f}, mean={stats.mean(n_md):.2f}")
    if perc_exec:
        print(f"%code executed: median={stats.median(perc_exec):.1f}, mean={stats.mean(perc_exec):.1f}")

    # imports agregados
    counter = Counter()
    for r in coll:
        tj = r.get("top_imports_json") or "[]"
        try:
            items = json.loads(tj)
            for mod, cnt in items:
                counter[mod] += int(cnt)
        except Exception:
            pass
    if counter:
        print("Top imports (agregado):", counter.most_common(10))

    # ---------- execuções ----------
    if args.exec_csv:
        ex = load_csv(args.exec_csv)
        # Build master join with fallback for downstream analyses
        try:
            merged_rows = merge_collection_and_exec(coll, ex)
        except Exception as e:
            print(f"WARNING: master join failed: {e}")
        m = len(ex)
        ok = [r for r in ex if to_bool(r.get("exec_ok"))]
        fail = [r for r in ex if not to_bool(r.get("exec_ok"))]
        print(f"# Execuções: {m}  |  sucesso: {len(ok)} ({pct(len(ok), m):.1f}%)")

        by_err = Counter((r.get("exec_error_type") or "None") for r in fail)
        print("Erros mais comuns:", by_err.most_common(12))

        elapsed_all = [to_float(r.get("elapsed_s")) for r in ex if r.get("elapsed_s")]
        elapsed_ok = [to_float(r.get("elapsed_s")) for r in ok if r.get("elapsed_s")]
        elapsed_fail = [to_float(r.get("elapsed_s")) for r in fail if r.get("elapsed_s")]
        if elapsed_all:
            print(f"elapsed_s (todos): median={stats.median(elapsed_all):.2f}s, mean={stats.mean(elapsed_all):.2f}s")
        if elapsed_ok:
            print(f"elapsed_s (sucesso): median={stats.median(elapsed_ok):.2f}s, mean={stats.mean(elapsed_ok):.2f}s")
        if elapsed_fail:
            print(f"elapsed_s (falha): median={stats.median(elapsed_fail):.2f}s, mean={stats.mean(elapsed_fail):.2f}s")

        fail_by_repo = Counter(r.get("repo_full_name") for r in fail)
        if fail_by_repo:
            print("Repos com mais falhas:", fail_by_repo.most_common(10))

        fail_by_file = Counter(f"{r.get('repo_full_name')}/{r.get('file_path')}" for r in fail)
        if fail_by_file:
            print("Notebooks com falha (top):", fail_by_file.most_common(10))

        # ---------- Reprodutibilidade: original vs executado ----------
        # Validar presença de colunas necessárias
        needed = {"original_found","outputs_equal","outputs_hash_orig","outputs_hash_exec","n_outputs_orig","n_outputs_exec"}
        if not ex:
            print("⚠️  Sem dados de execução para análise de reprodutibilidade")
        elif not needed.issubset(ex[0].keys()):
            missing = needed - set(ex[0].keys())
            print(f"⚠️  Colunas ausentes para análise de reprodutibilidade: {missing}")
            print("   Pulando análise de reprodutibilidade")
        elif needed.issubset(ex[0].keys()):
            orig_found = [r for r in ex if to_bool(r.get("original_found"))]
            equal_all = [r for r in ex if to_bool(r.get("outputs_equal"))]
            equal_ok  = [r for r in ok if to_bool(r.get("outputs_equal"))]

            print(f"originais encontrados: {len(orig_found)} ({pct(len(orig_found), m):.1f}%)")
            print(f"outputs iguais (todas execuções): {len(equal_all)} ({pct(len(equal_all), m):.1f}%)")
            print(f"outputs iguais (entre sucessos): {len(equal_ok)} ({pct(len(equal_ok), len(ok) or 1):.1f}%)")

            # exemplos de divergência
            diffs = [r for r in ex
                     if r.get("outputs_hash_orig") and r.get("outputs_hash_exec")
                     and r["outputs_hash_orig"] != r["outputs_hash_exec"]]
            if diffs:
                print("Exemplos de divergência (até 10):")
                for r in diffs[:10]:
                    print(" -", f"{r.get('repo_full_name')}/{r.get('file_path')}",
                          f"orig={r.get('outputs_hash_orig')[:8]} exec={r.get('outputs_hash_exec')[:8]}",
                          f"n_orig={r.get('n_outputs_orig')} n_exec={r.get('n_outputs_exec')}")

            # distribuição de número de outputs
            n_orig_vals = [to_int(r.get("n_outputs_orig")) for r in ex if r.get("n_outputs_orig")]
            n_exec_vals = [to_int(r.get("n_outputs_exec")) for r in ex if r.get("n_outputs_exec")]
            if n_orig_vals:
                print(f"n_outputs_orig: median={stats.median(n_orig_vals):.0f}, mean={stats.mean(n_orig_vals):.1f}")
            if n_exec_vals:
                print(f"n_outputs_exec: median={stats.median(n_exec_vals):.0f}, mean={stats.mean(n_exec_vals):.1f}")

            # KPI: reprodutibilidade entre originais
            orig_and_equal = [r for r in ex if to_bool(r.get("original_found")) and to_bool(r.get("outputs_equal"))]
            orig_and_exec_ok = [r for r in ex if to_bool(r.get("original_found")) and to_bool(r.get("exec_ok"))]
            orig_execok_and_equal = [r for r in ex if to_bool(r.get("original_found")) and to_bool(r.get("exec_ok")) and to_bool(r.get("outputs_equal"))]
            if orig_found:
                print(f"Reprodutibilidade (entre originais): {pct(len(orig_and_equal), len(orig_found)):,.1f}% "
                      f"({len(orig_and_equal)}/{len(orig_found)})")
            if orig_and_exec_ok:
                print(f"Reprodutibilidade (entre originais + exec_ok): {pct(len(orig_execok_and_equal), len(orig_and_exec_ok)):,.1f}% "
                      f"({len(orig_execok_and_equal)}/{len(orig_and_exec_ok)})")

        # ---------- Sanity Check: Reprodutibilidade Básica (não resultados finais) ----------
        print("\n" + "="*60)
        print("SANITY CHECK: Reprodutibilidade Básica (não resultados finais)")
        print("="*60)
        print("NOTA: Análises detalhadas de RQ1/RQ2/RQ3 devem ser feitas no pipeline_master.ipynb")
        
        # Apenas checagem básica de ordem de grandeza
        if orig_found:
            print(f"\nOrdem de grandeza (sanity check):")
            print(f"  - Originais encontrados: {len(orig_found)} ({pct(len(orig_found), m):.1f}%)")
            print(f"  - Outputs iguais (entre originais): {len(orig_and_equal)} ({pct(len(orig_and_equal), len(orig_found)):.1f}%)")
            if orig_and_exec_ok:
                print(f"  - Outputs iguais (originais + exec_ok): {len(orig_execok_and_equal)} ({pct(len(orig_execok_and_equal), len(orig_and_exec_ok)):.1f}%)")
            print("\n  (Valores finais para o paper devem ser calculados no pipeline_master.ipynb)")


if __name__ == "__main__":
    main()
