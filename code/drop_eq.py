"""drop_eq.py PROBLEM TOL -- forensic re-run for referee Sec. 3.1.
SAME code, SAME start as run_audit.py, but with the CUTEst EQUALITY rows removed (inequality rows and simple
bounds kept). Tests whether the values of the old Table 3 are reproduced when the equality constraints are
omitted at import. Run `python3 run_audit.py PROBLEM --tol TOL` first (it stores the start x0).
Example:  python3 run_audit.py NCVXQP1 --tol 1e-3 && python3 drop_eq.py NCVXQP1 0.001"""
import sys, json, numpy as np, scipy.sparse as sp
from pathlib import Path
from run_audit import decode, lam_max, evaluate
from indca_qp import Projector, indca1, lin_violation
name, tol = sys.argv[1], float(sys.argv[2])
D = decode(name)
eq = np.isfinite(D["cl"]) & (D["cl"] == D["cu"])
J2, cl2, cu2 = D["J"].tocsr()[~eq], D["cl"][~eq], D["cu"][~eq]
if J2.shape[0] == 0: J2 = sp.csr_matrix((0, D["n"])); cl2 = cu2 = np.zeros(0)
red = Projector(J2, cl2, cu2, D["bl"], D["bu"])
x0 = np.load(Path(__file__).parent / "results" / f"{name}_cutest_s0_t{tol:g}_x0.npy")
eta = lam_max(D["Q"]) + 2.0
out = {}
for tag, g in (("DCA", 0.0), ("InDCA1", 0.5)):
    r = indca1(D["Q"], D["q"], red, x0, eta, g, tol, 5000)
    x = r["x"]
    out[tag] = dict(obj=float(0.5*x@(D["Q"]@x)+D["q"]@x+D["f0"]), iters=r["iterations"],
                    eq_viol=float(np.abs(D["J"].tocsr()[eq]@x - D["cl"][eq]).max()) if eq.any() else 0.0)
print(name, "n_eq_dropped=", int(eq.sum()), json.dumps(out))
