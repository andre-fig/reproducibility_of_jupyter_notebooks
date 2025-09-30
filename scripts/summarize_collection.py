import argparse, csv, json, statistics as stats
from collections import Counter

def load_csv(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for r in rd:
            rows.append(r)
    return rows

def pct(a, b): 
    return 0.0 if b==0 else 100.0*a/b

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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--collection-csv", required=True)
    ap.add_argument("--exec-csv", required=False)
    args = ap.parse_args()

    # ---------- coleção ----------
    coll = load_csv(args.collection_csv)
    n = len(coll)
    print(f"# Notebooks (linhas): {n}")

    parsed = [r for r in coll if r.get("nb_ok_parse") == "True"]
    py = [r for r in coll if (r.get("language") or "").lower().startswith("python")]
    deps_any = [r for r in coll if r.get("deps_any") == "True"]
    unamb = [r for r in coll if r.get("has_unambiguous_order") == "True"]

    print(f"nb_ok_parse=True: {len(parsed)} ({pct(len(parsed), n):.1f}%)")
    print(f"language=python: {len(py)} ({pct(len(py), n):.1f}%)")
    print(f"repos com deps_any=True: {len(deps_any)} ({pct(len(deps_any), n):.1f}%)")
    print(f"unambiguous order: {len(unamb)} ({pct(len(unamb), n):.1f}%)")

    n_code = [to_int(r.get("n_code", 0)) for r in coll]
    n_md = [to_int(r.get("n_markdown", 0)) for r in coll]
    perc_exec = [
        to_float(r.get("percent_code_executed", 0.0))
        for r in coll if (r.get("percent_code_executed") or "") != ""
    ]
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
        ok = [r for r in ex if r.get("exec_ok") == "True"]
        fail = [r for r in ex if r.get("exec_ok") != "True"]
        print(f"# Execuções: {m}  |  sucesso: {len(ok)} ({pct(len(ok), m):.1f}%)")

        # erros exatos (já vêm do executor)
        by_err = Counter((r.get("error") or "None") for r in fail)
        print("Erros mais comuns:", by_err.most_common(12))

        # tempo total / por status
        elapsed_all = [to_float(r.get("elapsed_s", 0.0)) for r in ex if (r.get("elapsed_s") or "") != ""]
        elapsed_ok = [to_float(r.get("elapsed_s", 0.0)) for r in ok if (r.get("elapsed_s") or "") != ""]
        elapsed_fail = [to_float(r.get("elapsed_s", 0.0)) for r in fail if (r.get("elapsed_s") or "") != ""]
        if elapsed_all:
            print(f"elapsed_s (todos): median={stats.median(elapsed_all):.2f}s, mean={stats.mean(elapsed_all):.2f}s")
        if elapsed_ok:
            print(f"elapsed_s (sucesso): median={stats.median(elapsed_ok):.2f}s, mean={stats.mean(elapsed_ok):.2f}s")
        if elapsed_fail:
            print(f"elapsed_s (falha): median={stats.median(elapsed_fail):.2f}s, mean={stats.mean(elapsed_fail):.2f}s")

        # repos com mais falhas (útil para priorizar depuração)
        fail_by_repo = Counter(r.get("repo_full_name") for r in fail)
        if fail_by_repo:
            print("Repos com mais falhas:", fail_by_repo.most_common(10))

        # top arquivos que falharam (pra achar “piores do lote”)
        fail_by_file = Counter(f"{r.get('repo_full_name')}/{r.get('file_path')}" for r in fail)
        if fail_by_file:
            print("Notebooks com falha (top):", fail_by_file.most_common(10))

if __name__ == "__main__":
    main()
