"""run_all_methods.py PROBLEM START SEED
Runs DCA (gamma=0) and InDCA1 (gamma=0.5) [projection decomposition, eta = lambda_n(Q) + rho],
DCA2 (gamma=0) and InDCA2 (gamma=0.5) [proximal decomposition, eta = -lambda_1(Q) + rho],
and the linearly constrained BDCA of Aragon Artacho-Campoy-Vuong [projection decomposition]
from ONE common feasible start x0 (rho = 2, Tol = 1e-6, at most 5000 iterations).
START = cutest : x0 = P_C(x0_CUTEst + 0.01 N(0,I)),  START = uniform : x0 = P_C(U[max(bl,-10), min(bu,10)]).
For NCVXQP instances with START=cutest, DCA/InDCA1 are also run on the bound-constrained companion NCVXBQP
from the same x0 (Table R2). Writes results3/<P>_<START>_s<SEED>.json and archives all returned points.
Example:  python3 run_all_methods.py NCVXQP3 cutest 0"""
import sys, json, numpy as np, scipy.sparse as sp, scipy.sparse.linalg as spla
from pathlib import Path
import pycutest
from run_audit import decode, build_projector, evaluate, COMPANION
from indca_qp import QPSolver, indca1, indca2, bdca, lin_violation, bound_violation
name, start, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
TOL, RHO, GAMMA, MAXIT = 1e-6, 2.0, 0.5, 5000
OUT = Path(__file__).parent / "results3"; OUT.mkdir(exist_ok=True)
D = decode(name); proj = build_projector(D); Q, q, n = D["Q"], D["q"], D["n"]
# extreme eigenvalues of Q (computed once; not included in the reported run times)
if n <= 3000:
    ev = np.linalg.eigvalsh(Q.toarray()); lmin, lmax = float(ev[0]), float(ev[-1])
else:
    lmin = float(spla.eigsh(Q, k=1, which="SA", return_eigenvectors=False, maxiter=50000, tol=1e-10)[0])
    lmax = float(spla.eigsh(Q, k=1, which="LA", return_eigenvectors=False, maxiter=50000, tol=1e-10)[0])
eta1, eta2 = lmax + RHO, -lmin + RHO
rng = np.random.default_rng(seed)
if start == "cutest":
    x_raw = D["x0"] + 0.01 * rng.standard_normal(n)
else:
    lo = np.maximum(D["bl"], -10.0); hi = np.minimum(D["bu"], 10.0)
    x_raw = lo + (hi - lo) * rng.random(n)
x0 = proj(x_raw)
row = dict(problem=name, start=start, seed=seed, n=n, m_cutest=D["m_cutest"], m_imported=D["m_imported"],
           m_equalities=D["n_eq"], rows_Axb_general=D["rows_Axb_general"], rows_Axb_bounds=D["rows_Axb_bounds"],
           quad_check=D["quad_err"], linearity_check=D["lin_err"], lambda_min=lmin, lambda_max=lmax,
           eta1=eta1, eta2=eta2, rho=RHO, gamma=GAMMA, tol=TOL, max_iter=MAXIT,
           x0_lin_viol=lin_violation(x0, D["J"], D["cl"], D["cu"]), x0_bound_viol=bound_violation(x0, D["bl"], D["bu"]),
           qp_solver=f"piqp {__import__('piqp').__version__} + active-set polishing")
np.save(OUT / f"{name}_{start}_s{seed}_x0.npy", x0)

def record(tag, r, S):
    e = evaluate(D, r["x"], proj)
    row[tag] = dict(e, iterations=r["iterations"], converged=r["converged"], time_s=round(r["time_s"], 3),
                    **({"boosted_steps": r["boosted_steps"]} if "boosted_steps" in r else {}))
    np.save(OUT / f"{name}_{start}_s{seed}_{tag}.npy", r["x"])
    print(tag, "%.10e" % e["obj"], "kkt %.1e" % e["kkt_res"], r["iterations"], r["converged"], flush=True)

for tag, g in (("DCA", 0.0), ("InDCA1", GAMMA)):
    record(tag, indca1(Q, q, proj, x0, eta1, g, TOL, MAXIT), proj)
sub = QPSolver(Q + eta2 * sp.eye(n), D["J"], D["cl"], D["cu"], D["bl"], D["bu"])
for tag, g in (("DCA2", 0.0), ("InDCA2", GAMMA)):
    record(tag, indca2(Q, q, sub, x0, eta2, g, TOL, MAXIT), sub)
record("BDCA", bdca(Q, q, proj, D["J"], D["cl"], D["cu"], D["bl"], D["bu"], x0, eta1, TOL, MAXIT), proj)
row["qp_stats"] = dict(proj_solves=proj.nsolves, proj_polished=proj.npolished, proj_polish_rejected=proj.npolish_rejected,
                       proj_fallbacks=proj.nfallback, sub_solves=sub.nsolves, sub_polished=sub.npolished,
                       sub_polish_rejected=sub.npolish_rejected, sub_fallbacks=sub.nfallback)
comp = COMPANION.get(name)
if comp and start == "cutest":
    C = decode(comp); cproj = build_projector(C)
    ceta = float(np.linalg.eigvalsh(C["Q"].toarray())[-1]) + RHO
    cres = {}
    for tag, g in (("DCA", 0.0), ("InDCA1", GAMMA)):
        r = indca1(C["Q"], C["q"], cproj, x0, ceta, g, TOL, MAXIT)
        e = evaluate(C, r["x"], cproj)
        cres[tag] = dict(e, iterations=r["iterations"], converged=r["converged"],
                         lin_viol_in_NCVXQP=lin_violation(r["x"], D["J"], D["cl"], D["cu"]))
        np.save(OUT / f"{name}_{start}_s{seed}_{comp}_{tag}.npy", r["x"])
    row["companion"] = dict(name=comp, eta=ceta, max_abs_diff_Q=float(abs(C["Q"] - Q).max()),
                            max_abs_diff_q=float(np.abs(C["q"] - q).max()), **cres)
(OUT / f"{name}_{start}_s{seed}.json").write_text(json.dumps(row, indent=1, default=float))
