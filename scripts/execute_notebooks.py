import argparse, csv, json, statistics as stats
from collections import Counter, defaultdict

# -------------------------
# utils
# -------------------------
def load_csv(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for r in rd:
            rows.append(r)
    return rows

def pct(a, b):
    return 0.0 if not b else 100.0 * a / b

def to_int(x, d=0):
    try:
        # aceita "3.0" também
        return int(float(x))
    except Exception:
        return d

def to_float(x, d=0.0):
    try:
        return float(x)
    except Exception:
        return d

def to_bool(x):
    if x is None:
        return False
    s = str(x).strip().lower()
    return s in ("true", "1", "yes", "y", "t")

def safe_stats_mean(vals):
    return None if not vals else stats.mean(vals)

def safe_stats_median(vals):
    return None if not vals else stats.median(vals)

def print_stat(label, val, fmt="{}"):
    if val is not None:
        print(f"{label}: {fmt.format(val)}")

# -------------------------
# main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--collection-csv", required=True, help="CSV gerado pelo coletor")
    ap.add_argument("--exec-csv", required=False, help="CSV gerado pelo executor (opcional)")
    args = ap.parse_args()

    # ---------- coleção ----------
    coll = load_csv(args.collection_csv)
    n = len(coll)
    print(f"# Notebooks coletados (linhas): {n}")

    parsed = [r for r in coll if to_bool(r.get("nb_ok_parse"))]
    py = [r for r in coll if (r.get("language") or "").strip().lower().startswith("python")]
    deps_any = [r for r in coll if to_bool(r.get("deps_any"))]
    unamb = [r for r in coll if to_bool(r.get("has_unambiguous_order"))]

    print(f"nb_ok_parse=True: {len(parsed)} ({pct(len(parsed), n):.1f}%)")
    print(f"language=python: {len(py)} ({pct(len(py), n):.1f}%)")
    print(f"repos com deps_any=True: {len(deps_any)} ({pct(len(deps_any), n):.1f}%)")
    print(f"unambiguous order: {len(unamb)} ({pct(len(unamb), n):.1f}%)")

    n_code = [to_int(r.get("n_code")) for r in coll if r.get("n_code") not in (None, "")]
    n_md = [to_int(r.get("n_markdown")) for r in coll if r.get("n_markdown") not in (None, "")]
    n_outputs_cells = [to_int(r.get("n_cells_with_output")) for r in coll if r.get("n_cells_with_output") not in (None, "")]
    perc_exec = [to_float(r.get("percent_code_executed")) for r in coll if r.get("percent_code_executed") not in (None, "")]
    file_sizes = [to_int(r.get("file_size")) for r in coll if r.get("file_size") not in (None, "")]
    stars = [to_int(r.get("repo_stars")) for r in coll if r.get("repo_stars") not in (None, "")]

    m = safe_stats_median; a = safe_stats_mean
    print_stat("n_code median", m(n_code), "{:.0f}")
    print_stat("n_code mean", a(n_code), "{:.2f}")
    print_stat("n_markdown median", m(n_md), "{:.0f}")
    print_stat("n_markdown mean", a(n_md), "{:.2f}")
    print_stat("n_cells_with_output median", m(n_outputs_cells), "{:.0f}")
    print_stat("n_cells_with_output mean", a(n_outputs_cells), "{:.2f}")
    print_stat("%code executed median", m(perc_exec), "{:.1f}%")
    print_stat("%code executed mean", a(perc_exec), "{:.1f}%")

    # tamanhos brute do .ipynb (bytes)
    if file_sizes:
        print_stat("file_size median (KB)", m([s/1024 for s in file_sizes]), "{:.1f}")
        print_stat("file_size mean (KB)", a([s/1024 for s in file_sizes]), "{:.1f}")
        # percentis simples
        for q in (50, 75, 90, 95, 99):
            idx = int(len(file_sizes) * q / 100)
            val = sorted(file_sizes)[min(max(idx-1,0), len(file_sizes)-1)] / 1024.0
            print(f"file_size P{q} (KB): {val:.1f}")

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

    # ---------- execuções (opcional) ----------
    if args.exec_csv:
        ex = load_csv(args.exec_csv)
        mexec = len(ex)
        ok = [r for r in ex if to_bool(r.get("exec_ok"))]
        fail = [r for r in ex if not to_bool(r.get("exec_ok"))]

        print(f"# Execuções: {mexec}  |  sucesso: {len(ok)} ({pct(len(ok), mexec):.1f}%)")

        # erros mais comuns
        by_err = Counter((r.get("error") or "None") for r in fail)
        print("Erros mais comuns:", by_err.most_common(12))

        # tempos
        elapsed_all = [to_float(r.get("elapsed_s")) for r in ex if r.get("elapsed_s")]
        elapsed_ok = [to_float(r.get("elapsed_s")) for r in ok if r.get("elapsed_s")]
        elapsed_fail = [to_float(r.get("elapsed_s")) for r in fail if r.get("elapsed_s")]

        print_stat("elapsed_s (todos) median", safe_stats_median(elapsed_all), "{:.2f}s")
        print_stat("elapsed_s (todos) mean", safe_stats_mean(elapsed_all), "{:.2f}s")
        print_stat("elapsed_s (sucesso) median", safe_stats_median(elapsed_ok), "{:.2f}s")
        print_stat("elapsed_s (sucesso) mean", safe_stats_mean(elapsed_ok), "{:.2f}s")
        print_stat("elapsed_s (falha) median", safe_stats_median(elapsed_fail), "{:.2f}s")
        print_stat("elapsed_s (falha) mean", safe_stats_mean(elapsed_fail), "{:.2f}s")

        # repos/arquivos com mais falhas
        fail_by_repo = Counter(r.get("repo_full_name") for r in fail)
        if fail_by_repo:
            print("Repos com mais falhas:", fail_by_repo.most_common(10))

        fail_by_file = Counter(f"{r.get('repo_full_name')}/{r.get('file_path')}" for r in fail)
        if fail_by_file:
            print("Notebooks com falha (top):", fail_by_file.most_common(10))

        # --------- ORIGINAL vs EXECUTADO (reprodutibilidade) ----------
        # colunas esperadas do executor:
        # original_found, outputs_equal, outputs_hash_orig, outputs_hash_exec, n_outputs_orig, n_outputs_exec
        have_orig_col = "original_found" in ex[0] if ex else False
        have_equal_col = "outputs_equal" in ex[0] if ex else False

        if have_orig_col or have_equal_col:
            orig_found = [r for r in ex if to_bool(r.get("original_found"))]
            print(f"originais encontrados: {len(orig_found)} ({pct(len(orig_found), mexec):.1f}%)")

            equal_all = [r for r in ex if to_bool(r.get("outputs_equal"))]
            print(f"outputs iguais (todas execuções): {len(equal_all)} ({pct(len(equal_all), mexec):.1f}%)")

            # só entre execuções com sucesso
            equal_ok = [r for r in ok if to_bool(r.get("outputs_equal"))]
            print(f"outputs iguais (entre sucessos): {len(equal_ok)} ({pct(len(equal_ok), len(ok) or 1):.1f}%)")

            # divergências (útil mostrar alguns exemplos)
            diffs = [r for r in ex if r.get("outputs_hash_orig") and r.get("outputs_hash_exec") and r.get("outputs_hash_orig") != r.get("outputs_hash_exec")]
            if diffs:
                print("Exemplos de divergência (até 10):")
                for r in diffs[:10]:
                    print(" -", f"{r.get('repo_full_name')}/{r.get('file_path')}",
                          f"orig={r.get('outputs_hash_orig')[:8]} exec={r.get('outputs_hash_exec')[:8]}",
                          f"n_orig={r.get('n_outputs_orig')} n_exec={r.get('n_outputs_exec')}")

            # distribuição de n_outputs (orig vs exec) para investigar mudanças de quantidade
            n_orig_vals = [to_int(r.get("n_outputs_orig")) for r in ex if r.get("n_outputs_orig") not in (None, "")]
            n_exec_vals = [to_int(r.get("n_outputs_exec")) for r in ex if r.get("n_outputs_exec") not in (None, "")]
            print_stat("n_outputs_orig median", safe_stats_median(n_orig_vals), "{:.0f}")
            print_stat("n_outputs_orig mean", safe_stats_mean(n_orig_vals), "{:.1f}")
            print_stat("n_outputs_exec median", safe_stats_median(n_exec_vals), "{:.0f}")
            print_stat("n_outputs_exec mean", safe_stats_mean(n_exec_vals), "{:.1f}")

            # crosstab simples: (exec_ok x outputs_equal)
            crosstab = defaultdict(int)
            for r in ex:
                key = (to_bool(r.get("exec_ok")), to_bool(r.get("outputs_equal")))
                crosstab[key] += 1
            if crosstab:
                print("Matriz (exec_ok x outputs_equal):")
                # linhas: exec_ok False/True; colunas: outputs_equal False/True
                for exec_ok in (False, True):
                    row_vals = []
                    for eq in (False, True):
                        row_vals.append(crosstab[(exec_ok, eq)])
                    print(f"  exec_ok={exec_ok}: {row_vals[0]} (equal=False), {row_vals[1]} (equal=True)")

if __name__ == "__main__":
    main()
