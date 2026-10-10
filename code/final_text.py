"""final_text.py [RESULTS_DIR] -- computes every number quoted in the text of Section 3.3 (items (a)-(e))
and in the response letter (Sec. 3.1, 3.10, minor 10) from the JSON files of run_all_methods.py."""
import sys
import numpy as np
from final_tables import load_all, multistart, PROBLEMS, STARTS, METHODS, HIT_TOL, RES

R = load_all()
runs = [(p, st, s) for p in PROBLEMS for st, s in STARTS]
print(f"results directory: {RES};  {len(runs)} paired runs x {len(METHODS)} methods = {len(runs) * len(METHODS)} returned points")

# feasibility of the starts and of all returned points
x0l = max(R[k]["x0_lin_viol"] for k in runs); x0b = max(R[k]["x0_bound_viol"] for k in runs)
vl = max(R[k][t]["lin_viol"] for k in runs for t in METHODS); vb = max(R[k][t]["bound_viol"] for k in runs for t in METHODS)
print(f"starts: max linear violation {x0l:.1e}, max bound violation {x0b:.1e}")
print(f"returned points: max linear violation {vl:.1e}, max bound violation {vb:.1e}")
print(f"m_imported == m_CUTEst for all problems: {all(R[(p, 'cutest', 0)]['m_imported'] == R[(p, 'cutest', 0)]['m_cutest'] for p in PROBLEMS)}")

# QP solver statistics (minor 10)
qs = [R[k]["qp_stats"] for k in runs]
nsol = sum(q["proj_solves"] + q["sub_solves"] for q in qs)
nfb = sum(q["proj_fallbacks"] + q["sub_fallbacks"] for q in qs)
nrej = sum(q["proj_polish_rejected"] + q["sub_polish_rejected"] for q in qs)
print(f"QP solves (projections + InDCA2 subproblems): {nsol}; retries with the second setting: {nfb}; "
      f"polishing rejected: {nrej}")

# (a) runs stopped by the iteration limit
lim = sorted({p for (p, st, s) in runs for t in METHODS if not R[(p, st, s)][t]["converged"]}, key=PROBLEMS.index)
print("\n(a) problems with runs stopped by the iteration limit:", ", ".join(lim))
r0 = [R[(p, "cutest", 0)][t]["kkt_res"] for p in lim for t in METHODS]
ra = [R[k][t]["kkt_res"] for k in runs if k[0] in lim for t in METHODS if not R[k][t]["converged"]]
print(f"    KKT residuals on these problems, start of Tables 5-6: {min(r0):.1e} .. {max(r0):.1e}; "
      f"all runs stopped by the limit: {min(ra):.1e} .. {max(ra):.1e}")
worst = max(((R[k][t]["kkt_res"], k, t) for k in runs for t in METHODS))
print(f"    largest KKT residual over all {len(runs) * len(METHODS)} points: {worst[0]:.1e} ({worst[1][0]}, "
      f"{worst[1][1]} seed {worst[1][2]}, {worst[2]}, eta1 = {R[worst[1]]['eta1']:.0f}, eta2 = {R[worst[1]]['eta2']:.0f}, "
      f"converged = {R[worst[1]][worst[2]]['converged']})")


def compare(a, b):
    """counts: b lower than a / same / b higher (relative difference HIT_TOL), and the problems involved"""
    lower = same = higher = 0; where = set(); rel = {}
    for k in runs:
        fa, fb = R[k][a]["obj"], R[k][b]["obj"]; d = (fa - fb) / abs(fa)
        if d > HIT_TOL:
            lower += 1; where.add(k[0]); rel.setdefault(k[0], []).append(d)
        elif d < -HIT_TOL:
            higher += 1
        else:
            same += 1
    return lower, same, higher, where, rel


print("\n(b) paired comparisons of objective values (relative threshold %g)" % HIT_TOL)
for a, b in (("DCA", "InDCA1"), ("DCA2", "InDCA2")):
    lo, sa, hi, where, rel = compare(a, b)
    print(f"    {b} vs {a}: lower in {lo}, same in {sa}, higher in {hi} of {len(runs)} runs; lower on "
          + ", ".join(sorted(where, key=PROBLEMS.index)))
    for grp, ps in (("BLOCKQP3/5", ("BLOCKQP3", "BLOCKQP5")), ("BLOWEY", ("BLOWEYA", "BLOWEYB", "BLOWEYC"))):
        v = [d for p in ps for d in rel.get(p, [])]
        if v:
            print(f"        max relative improvement on {grp}: {max(v):.2e}")
for p in ("BLOCKQP2", "BLOCKQP4"):
    med = {t: float(np.median([R[(p, st, s)][t]["iterations"] for st, s in STARTS])) for t in METHODS}
    print(f"    {p}: median iterations over the 4 starts  DCA {med['DCA']:g}, InDCA1 {med['InDCA1']:g}, "
          f"DCA2 {med['DCA2']:g}, InDCA2 {med['InDCA2']:g}, BDCA {med['BDCA']:g}")

print("\n(c) NCVXQP: inertial weight gamma/eta")
lm = [R[(p, "cutest", 0)]["lambda_max"] for p in PROBLEMS if p.startswith("NCVXQP")]
ln = [-R[(p, "cutest", 0)]["lambda_min"] for p in PROBLEMS if p.startswith("NCVXQP")]
g = R[(PROBLEMS[0], "cutest", 0)]["gamma"]
print(f"    lambda_n(Q) in [{min(lm):.0f}, {max(lm):.0f}]  ->  gamma/eta1 <= {g / (min(lm) + 2):.1e}")
print(f"    -lambda_1(Q) in [{min(ln):.0f}, {max(ln):.0f}]  ->  gamma/eta2 <= {g / (min(ln) + 2):.1e}")

print("\n(d) BDCA vs DCA")
lo, sa, hi, _, _ = compare("DCA", "BDCA")
print(f"    BDCA lower in {lo}, same in {sa}, higher in {hi} of {len(runs)} runs")
ratio = [R[k]["BDCA"]["iterations"] / R[k]["DCA"]["iterations"] for k in runs
         if R[k]["BDCA"]["converged"] and R[k]["DCA"]["converged"]]
print(f"    runs where both stopped by the test: {len(ratio)}; median BDCA/DCA iteration ratio {np.median(ratio):.2f}")
S = multistart(R)
print("    runs reaching f* (Table 7): " + ", ".join(
    f"{t} {sum(x <= HIT_TOL for p in PROBLEMS for x in S[p]['gaps'][t])}" for t in METHODS))

print("\n(e) different decompositions, start of Tables 5-6: NCVXQP instances where DCA and DCA2 return different values")
dif = [p for p in PROBLEMS if p.startswith("NCVXQP") and
       abs(R[(p, "cutest", 0)]["DCA"]["obj"] - R[(p, "cutest", 0)]["DCA2"]["obj"]) > HIT_TOL * abs(R[(p, "cutest", 0)]["DCA"]["obj"])]
print("    " + ", ".join(dif))
