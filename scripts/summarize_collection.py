import argparse, csv, json, statistics as stats
from collections import Counter, defaultdict

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
    except:
        return d

def to_float(x, d=0.0):
    try:
        return float(x)
    except:
        return d

def to_bool(x):
    if x is None: return False
    s = str(x).strip().lower()
    return s in ("true","1","yes","y","t")

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
    unamb = [r for r in coll if to_bool(r.get("has_unambiguous_order"))]

    print(f"nb_ok_parse=True: {len(parsed)} ({pct(len(parsed), n):.1f}%)")
    print(f"language=python: {len(py)} ({pct(len(py), n):.1f}%)")
    print(f"repos com deps_any=True: {len(deps_any)} ({pct(len(deps_any), n):.1f}%)")
    print(f"unambiguous order: {len(unamb)} ({pct(len(unamb), n):.1f}%)")

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
        m = len(ex)
        ok = [r for r in ex if to_bool(r.get("exec_ok"))]
        fail = [r for r in ex if not to_bool(r.get("exec_ok"))]
        print(f"# Execuções: {m}  |  sucesso: {len(ok)} ({pct(len(ok), m):.1f}%)")

        by_err = Counter((r.get("error") or "None") for r in fail)
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
        # só roda se o CSV tiver as colunas
        needed = {"original_found","outputs_equal","outputs_hash_orig","outputs_hash_exec","n_outputs_orig","n_outputs_exec"}
        if ex and needed.issubset(ex[0].keys()):
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

if __name__ == "__main__":
    main()
