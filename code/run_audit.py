"""
run_audit.py PROBLEM [--seed S] [--tol T] -- CUTEst decoding and the Sec. 3.1 audit for one instance.

Module part (imported by the other scripts):
  decode(name)          decodes a CUTEst problem with explicit SIF parameters, checks its dimensions,
                        checks that the objective is quadratic and the constraints are linear, and returns
                        Q, q, J, cl <= Jx <= cu, bl <= x <= bu (the SIF -> Ax >= b mapping, see indca_qp.py).
  build_projector(D)    P_C for the decoded constraint set.
  evaluate(D, x, proj)  objective, linear/bound violations and KKT residual at x.

Script part: runs DCA (gamma=0) and InDCA1 (gamma=0.5) from ONE common feasible start and, for NCVXQP
instances, the bound-constrained companion NCVXBQP from the SAME start.
Writes results/<PROBLEM>_<START>_s<S>_t<T>.json and archives the returned points (.npy).
Used for the Tol = 1e-3 observations of referee minor comment 9 (default --tol 1e-3).
"""
import sys, json, time, argparse
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import pycutest
from indca_qp import (Projector, indca1, lin_violation, bound_violation, kkt_residual)

SIF = {**{f"BLOCKQP{i}": {"N": 1000} for i in range(2, 6)},
       **{f"BLOWEY{c}": {"N": 1000} for c in "ABC"},
       **{f"NCVXQP{i}": {"N": 1000} for i in range(1, 10)},
       **{f"NCVXBQP{i}": {"N": 1000} for i in range(1, 4)},
       "QPNBAND": {"N": 10000}, "QPNBLEND": None}
TABLE3 = {**{f"BLOCKQP{i}": (2010, 1001) for i in range(2, 6)},
          **{f"BLOWEY{c}": (2002, 1002) for c in "ABC"},
          **{f"NCVXQP{i}": (1000, 500) for i in (1, 2, 3)},
          **{f"NCVXQP{i}": (1000, 250) for i in (4, 5, 6)},
          **{f"NCVXQP{i}": (1000, 750) for i in (7, 8, 9)},
          **{f"NCVXBQP{i}": (1000, 0) for i in (1, 2, 3)},
          "QPNBAND": (10000, 5000), "QPNBLEND": (83, 74)}
COMPANION = {f"NCVXQP{i}": f"NCVXBQP{(i - 1) % 3 + 1}" for i in range(1, 10)}
OUT = Path(__file__).parent / "results"


def decode(name):
    p = pycutest.import_problem(name, sifParams=SIF[name])
    n, m = p.n, p.m
    assert (n, m) == TABLE3[name], f"{name}: decoded {(n, m)} != Table 3 {TABLE3[name]}"
    z = np.zeros(n)
    f0 = float(p.obj(z)); q = p.grad(z)
    Q = sp.csr_matrix(p.isphess(z) if m else p.sphess(z)); Q = (0.5 * (Q + Q.T)).tocsr()
    # check the objective really is quadratic: compare with a random point
    xr = p.x0 + np.random.default_rng(1).standard_normal(n)
    fq = 0.5 * xr @ (Q @ xr) + q @ xr + f0
    quad_err = abs(fq - p.obj(xr)) / max(1.0, abs(fq))
    if m:
        c0, J = p.scons(z, gradient=True)
        J = sp.csr_matrix(J)
        # linearity check of the constraints
        cr, _ = p.scons(xr, gradient=True)
        lin_err = float(np.max(np.abs(cr - (c0 + J @ xr))))
        cl, cu = p.cl - c0, p.cu - c0          # cl <= Jx <= cu
        big = 1e19
        cl = np.where(cl <= -big, -np.inf, cl); cu = np.where(cu >= big, np.inf, cu)
        n_eq = int(np.sum(np.isfinite(cl) & (cl == cu)))
        rows_Axb = int(np.sum(np.isfinite(cl)) + np.sum(np.isfinite(cu)))
    else:
        J = sp.csr_matrix((0, n)); cl = cu = np.zeros(0); lin_err = 0.0; n_eq = 0; rows_Axb = 0
    bl = np.where(p.bl <= -1e19, -np.inf, p.bl); bu = np.where(p.bu >= 1e19, np.inf, p.bu)
    nb = int(np.isfinite(bl).sum() + np.isfinite(bu).sum())
    return dict(name=name, n=n, m_cutest=m, Q=Q, q=q, f0=f0, J=J, cl=cl, cu=cu, bl=bl, bu=bu,
                x0=p.x0.copy(), n_eq=n_eq, m_imported=J.shape[0], rows_Axb_general=rows_Axb,
                rows_Axb_bounds=nb, quad_err=float(quad_err), lin_err=lin_err)


def build_projector(D):
    return Projector(D["J"], D["cl"], D["cu"], D["bl"], D["bu"])


def lam_max(Q):
    n = Q.shape[0]
    if n <= 3000:
        return float(np.linalg.eigvalsh(Q.toarray())[-1])
    return float(spla.eigsh(Q, k=1, which="LA", return_eigenvectors=False, maxiter=50000, tol=1e-10)[0])


def evaluate(D, x, proj):
    Q, q = D["Q"], D["q"]
    return dict(obj=float(0.5 * x @ (Q @ x) + q @ x + D["f0"]),
                lin_viol=lin_violation(x, D["J"], D["cl"], D["cu"]),
                bound_viol=bound_violation(x, D["bl"], D["bu"]),
                kkt_res=kkt_residual(x, Q, q, proj))


def run(D, proj, x0, eta, gamma, tol, max_iter):
    n0, f0 = proj.nsolves, proj.nfallback
    r = indca1(D["Q"], D["q"], proj, x0, eta, gamma, tol, max_iter)
    ns, nf = proj.nsolves - n0, proj.nfallback - f0
    ev = evaluate(D, r["x"], proj)
    return dict(ev, iterations=r["iterations"], converged=r["converged"], qp_solves=ns, qp_fallbacks=nf,
                time_s=round(r["time_s"], 2)), r["x"]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("name"); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rho", type=float, default=2.0); ap.add_argument("--gamma", type=float, default=0.5)
    ap.add_argument("--tol", type=float, default=1e-3); ap.add_argument("--max-iter", type=int, default=5000)
    ap.add_argument("--start", choices=["cutest", "uniform"], default="cutest",
                    help="cutest: x0_CUTEst + 0.01*N(0,I); uniform: U[max(bl,-10), min(bu,10)]; both then projected onto C")
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    D = decode(a.name)
    proj = build_projector(D)
    lmax = lam_max(D["Q"]); eta = lmax + a.rho
    rng = np.random.default_rng(a.seed)
    if a.start == "cutest":
        x_raw = D["x0"] + 0.01 * rng.standard_normal(D["n"])
    else:
        lo = np.maximum(D["bl"], -10.0); hi = np.minimum(D["bu"], 10.0)
        x_raw = lo + (hi - lo) * rng.random(D["n"])
    x0 = proj(x_raw)                                # feasible common start x0 in C
    row = dict(problem=a.name, seed=a.seed, start=a.start, n=D["n"], m_cutest=D["m_cutest"],
               m_imported=D["m_imported"], m_equalities=D["n_eq"],
               rows_Axb_general=D["rows_Axb_general"], rows_Axb_bounds=D["rows_Axb_bounds"],
               quad_check=D["quad_err"], linearity_check=D["lin_err"],
               lambda_max=lmax, eta=eta, rho=a.rho, gamma=a.gamma, tol=a.tol, max_iter=a.max_iter,
               x0_lin_viol=lin_violation(x0, D["J"], D["cl"], D["cu"]),
               x0_bound_viol=bound_violation(x0, D["bl"], D["bu"]),
               qp_solver=f"piqp {__import__('piqp').__version__}")
    res_dca, x_dca = run(D, proj, x0, eta, 0.0, a.tol, a.max_iter)
    res_in, x_in = run(D, proj, x0, eta, a.gamma, a.tol, a.max_iter)
    row["DCA"] = res_dca; row["InDCA1"] = res_in
    np.save(OUT / f"{a.name}_{a.start}_s{a.seed}_t{a.tol:g}_x0.npy", x0)
    np.save(OUT / f"{a.name}_{a.start}_s{a.seed}_t{a.tol:g}_DCA.npy", x_dca)
    np.save(OUT / f"{a.name}_{a.start}_s{a.seed}_t{a.tol:g}_InDCA1.npy", x_in)
    comp = COMPANION.get(a.name)
    if comp:
        C = decode(comp)
        cproj = build_projector(C)
        ceta = lam_max(C["Q"]) + a.rho
        same_Q = float(abs(C["Q"] - D["Q"]).max()); same_q = float(np.abs(C["q"] - D["q"]).max())
        cres = {}
        for tag, g in (("DCA", 0.0), ("InDCA1", a.gamma)):
            r, cx = run(C, cproj, x0, ceta, g, a.tol, a.max_iter)   # SAME start x0 (it lies in the box)
            r["lin_viol_in_NCVXQP"] = lin_violation(cx, D["J"], D["cl"], D["cu"])
            cres[tag] = r
            np.save(OUT / f"{a.name}_{a.start}_s{a.seed}_t{a.tol:g}_{comp}_{tag}.npy", cx)
        row["companion"] = dict(name=comp, eta=ceta, max_abs_diff_Q=same_Q, max_abs_diff_q=same_q,
                                start_is_same=True, **cres)
    (OUT / f"{a.name}_{a.start}_s{a.seed}_t{a.tol:g}.json").write_text(json.dumps(row, indent=1, default=float))
    print(json.dumps(row, default=float))
