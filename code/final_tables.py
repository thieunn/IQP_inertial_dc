"""final_tables.py [RESULTS_DIR] -- builds the LaTeX tables of the revised manuscript and the response letter
from the JSON files written by run_all_methods.py (default RESULTS_DIR = results3):

  tables/table5.tex   Table 5: DCA vs InDCA1 (projection decomposition), start 'cutest', seed 0
  tables/table6.tex   Table 6: DCA2 vs InDCA2 (proximal decomposition) and BDCA, start 'cutest', seed 0
  tables/table7.tex   Table 7: multi-start comparison (4 starts per problem, 5 methods)
  tables/tableR1.tex  Table R1 (response letter, Sec. 3.1): import audit
  tables/tableR2.tex  Table R2 (response letter, Sec. 3.1): NCVXQP vs NCVXBQP from the same start
"""
import sys, json
from pathlib import Path
import numpy as np

HERE = Path(__file__).parent
RES = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "results3"
OUT = HERE / "tables"
PROBLEMS = ("BLOCKQP2 BLOCKQP3 BLOCKQP4 BLOCKQP5 BLOWEYA BLOWEYB BLOWEYC NCVXQP1 NCVXQP2 NCVXQP3 NCVXQP4 "
            "NCVXQP5 NCVXQP6 NCVXQP7 NCVXQP8 NCVXQP9 QPNBAND QPNBLEND").split()
STARTS = (("cutest", 0), ("uniform", 1), ("uniform", 2), ("uniform", 3))
METHODS = ("DCA", "InDCA1", "DCA2", "InDCA2", "BDCA")
HIT_TOL = 1e-6          # relative tolerance for "reached f*" and for ties


def load(p, start, seed):
    return json.load(open(RES / f"{p}_{start}_s{seed}.json"))


def load_all():
    return {(p, st, s): load(p, st, s) for p in PROBLEMS for st, s in STARTS}


def e(v, d):
    """$-9.938006\\mathrm{e}{+02}$ style (d digits after the point)."""
    m, x = f"{v:.{d}e}".split("e")
    return f"${m}\\mathrm{{e}}{{{int(x):+03d}}}$"


def e0(v, d=1):
    return "$0$" if v == 0 else e(v, d)


def short(v):
    """5e-4, 8e-1, 2e+00, 0 (below 1e-12): the style of Table 7."""
    if v < 1e-12:
        return "0"
    m, x = f"{v:.0e}".split("e")
    x = int(x)
    return f"{m}e{x}" if x < 0 else f"{m}e+{x:02d}"


def it(r):
    return f"{r['iterations']}{'$^*$' if not r['converged'] else ''}"


def rel_gap(f, fstar):
    return (f - fstar) / abs(fstar)


def table5(R):
    L = [r"\begin{tabular}{l|cc|cc|cc|cc|cc}", r"\toprule",
         r"\multirow{2}{*}{Problem} & \multicolumn{2}{c|}{Dim} & \multicolumn{2}{c|}{Obj} & \multicolumn{2}{c|}{KKT residual $r$} & \multicolumn{2}{c|}{Iter} & \multicolumn{2}{c}{Time (s)}\\",
         r"& $n$ & $m$ & DCA & \textbf{InDCA1} & DCA & \textbf{InDCA1} & DCA & \textbf{InDCA1} & DCA & \textbf{InDCA1}\\",
         r"\midrule"]
    for p in PROBLEMS:
        r = R[(p, "cutest", 0)]; D, I = r["DCA"], r["InDCA1"]
        L.append(f"{p} & {r['n']} & {r['m_cutest']} & {e(D['obj'], 6)} & {e(I['obj'], 6)} & {e(D['kkt_res'], 1)} & "
                 f"{e(I['kkt_res'], 1)} & {it(D)} & {it(I)} & {D['time_s']:.2f} & {I['time_s']:.2f}\\\\")
    L += [r"\bottomrule\end{tabular}"]
    return "\n".join(L)


def table6(R):
    L = [r"\begin{tabular}{l|ccc|ccc}", r"\toprule",
         r"\multirow{2}{*}{Problem} & \multicolumn{3}{c|}{Obj} & \multicolumn{3}{c}{KKT residual $r$}\\",
         r"& DCA2 & \textbf{InDCA2} & BDCA & DCA2 & \textbf{InDCA2} & BDCA\\", r"\midrule"]
    for p in PROBLEMS:
        r = R[(p, "cutest", 0)]
        L.append(f"{p} & " + " & ".join(e(r[t]["obj"], 6) for t in ("DCA2", "InDCA2", "BDCA")) + " & "
                 + " & ".join(e(r[t]["kkt_res"], 1) for t in ("DCA2", "InDCA2", "BDCA")) + r"\\")
    L += [r"\bottomrule", r"\end{tabular}", "", r"\bigskip",
          r"\begin{tabular}{l|rr|rrr|rrr}", r"\toprule",
          r"\multirow{2}{*}{Problem} & \multicolumn{2}{c|}{Dim} & \multicolumn{3}{c|}{Iter} & \multicolumn{3}{c}{Time (s)}\\",
          r"& $n$ & $m$ & DCA2 & \textbf{InDCA2} & BDCA & DCA2 & \textbf{InDCA2} & BDCA\\", r"\midrule"]
    for p in PROBLEMS:
        r = R[(p, "cutest", 0)]
        L.append(f"{p} & {r['n']} & {r['m_cutest']} & " + " & ".join(it(r[t]) for t in ("DCA2", "InDCA2", "BDCA"))
                 + " & " + " & ".join(f"{r[t]['time_s']:.2f}" for t in ("DCA2", "InDCA2", "BDCA")) + r"\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L)


def multistart(R):
    """per problem: f*, and per method the list of relative gaps over the 4 starts."""
    S = {}
    for p in PROBLEMS:
        vals = {t: [R[(p, st, s)][t]["obj"] for st, s in STARTS] for t in METHODS}
        fstar = min(min(v) for v in vals.values())
        S[p] = dict(fstar=fstar, gaps={t: [rel_gap(f, fstar) for f in v] for t, v in vals.items()})
    return S


def table7(R):
    S = multistart(R)
    L = [r"\begin{tabular}{l|c|ccc|ccc|ccc|ccc|ccc}", r"\toprule",
         r"\multirow{2}{*}{Problem} & \multirow{2}{*}{$f^*$} & \multicolumn{3}{c|}{DCA} & \multicolumn{3}{c|}{\textbf{InDCA1}} & \multicolumn{3}{c|}{DCA2} & \multicolumn{3}{c|}{\textbf{InDCA2}} & \multicolumn{3}{c}{BDCA}\\",
         r"& & " + " & ".join(["hits & med & max"] * 5) + r"\\", r"\midrule"]
    for p in PROBLEMS:
        cells = []
        for t in METHODS:
            g = S[p]["gaps"][t]
            cells.append(f"{sum(x <= HIT_TOL for x in g)}/4 & {short(float(np.median(g)))} & {short(max(g))}")
        L.append(f"{p} & {e(S[p]['fstar'], 6)} & " + " & ".join(cells) + r"\\")
    runs = {t: [R[(p, st, s)][t] for p in PROBLEMS for st, s in STARTS] for t in METHODS}
    L.append(r"\midrule")
    L.append(r"\multicolumn{2}{l|}{runs stopped by the test} & " + " & ".join(
        rf"\multicolumn{{3}}{{c{'|' if t != 'BDCA' else ''}}}{{{sum(r['converged'] for r in runs[t])}/{len(runs[t])}}}" for t in METHODS) + r"\\")
    L.append(r"\multicolumn{2}{l|}{median / max KKT residual} & " + " & ".join(
        rf"\multicolumn{{3}}{{c{'|' if t != 'BDCA' else ''}}}{{{short(float(np.median([r['kkt_res'] for r in runs[t]])))} / {short(max(r['kkt_res'] for r in runs[t]))}}}" for t in METHODS) + r"\\")
    L.append(r"\multicolumn{2}{l|}{median iterations} & " + " & ".join(
        # medians of 72 integers can be half-integers; Python's round() rounds them to the nearest even integer
        rf"\multicolumn{{3}}{{c{'|' if t != 'BDCA' else ''}}}{{{round(float(np.median([r['iterations'] for r in runs[t]])))}}}" for t in METHODS) + r"\\")
    L.append(r"\multicolumn{2}{l|}{median / total time (s)} & " + " & ".join(
        rf"\multicolumn{{3}}{{c{'|' if t != 'BDCA' else ''}}}{{{float(np.median([r['time_s'] for r in runs[t]])):.2f} / {sum(r['time_s'] for r in runs[t]):.0f}}}" for t in METHODS) + r"\\")
    L.append(r"\bottomrule\end{tabular}")
    return "\n".join(L)


def tableR1(R):
    L = [r"\begin{tabular}{l|rrrrr|cc|cc|cc}", r"\toprule",
         r"Problem & $n$ & $m_{\rm CUTEst}$ & $m_{\rm imp}$ & \#eq & rows & Viol$_{\rm lin}$ & Viol$_{\rm bnd}$ & Obj (DCA) & Obj (InDCA1) & $r$ (DCA) & $r$ (InDCA1)\\",
         r"\midrule"]
    for p in PROBLEMS:
        r = R[(p, "cutest", 0)]; D, I = r["DCA"], r["InDCA1"]
        vl = max(r[t]["lin_viol"] for t in METHODS); vb = max(r[t]["bound_viol"] for t in METHODS)
        L.append(f"{p} & {r['n']} & {r['m_cutest']} & {r['m_imported']} & {r['m_equalities']} & "
                 f"{r['rows_Axb_general']}+{r['rows_Axb_bounds']} & {e0(vl)} & {e0(vb)} & {e(D['obj'], 6)} & "
                 f"{e(I['obj'], 6)} & {e(D['kkt_res'], 1)} & {e(I['kkt_res'], 1)}\\\\")
    L.append(r"\bottomrule\end{tabular}")
    return "\n".join(L)


def tableR2(R):
    L = [r"\begin{tabular}{l|cc|c|c|c}", r"\toprule",
         r"Problem & NCVXQP (DCA) & NCVXQP (InDCA1) & comp. & NCVXBQP & viol.\\", r"\midrule"]
    for p in PROBLEMS:
        if not p.startswith("NCVXQP"):
            continue
        r = R[(p, "cutest", 0)]; c = r["companion"]
        if abs(c["DCA"]["obj"] - c["InDCA1"]["obj"]) > 1e-6 * abs(c["DCA"]["obj"]):
            print(f"note: {p}: DCA and InDCA1 differ on {c['name']}: {c['DCA']['obj']} vs {c['InDCA1']['obj']}")
        L.append(f"{p} & {e(r['DCA']['obj'], 6)} & {e(r['InDCA1']['obj'], 6)} & {c['name']} & "
                 f"{e(c['InDCA1']['obj'], 7)} & {c['InDCA1']['lin_viol_in_NCVXQP']:.1f}\\\\")
    L.append(r"\bottomrule\end{tabular}")
    return "\n".join(L)


def main():
    R = load_all()
    OUT.mkdir(exist_ok=True)
    for name, fn in (("table5", table5), ("table6", table6), ("table7", table7),
                     ("tableR1", tableR1), ("tableR2", tableR2)):
        (OUT / f"{name}.tex").write_text(fn(R) + "\n")
        print(f"wrote tables/{name}.tex")


if __name__ == "__main__":
    main()
