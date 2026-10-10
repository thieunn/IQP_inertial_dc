"""check_kkt_clarabel.py -- independent check of the KKT residuals r(x) = ||x - P_C(x - (Qx+q))||:
the projection is recomputed with a different QP solver (Clarabel) for archived returned points
results3/<P>_cutest_s0_<METHOD>.npy produced by run_all_methods.py.
Usage: python3 check_kkt_clarabel.py [RESULTS_DIR]"""
import sys, json, numpy as np, scipy.sparse as sp, clarabel
from pathlib import Path
from run_audit import decode
RES = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).parent / "results3")

def clarabel_projection(D, v):
    n = D["n"]; J = D["J"].tocsr(); cl, cu, bl, bu = D["cl"], D["cu"], D["bl"], D["bu"]
    eq = np.isfinite(cl) & (cl == cu); ne = ~eq
    blocks, rhs = [J[eq]], [cl[eq]]
    for M, b in ((-J[ne], -cl[ne]), (J[ne], cu[ne]), (-sp.eye(n), -bl), (sp.eye(n), bu)):
        f = np.isfinite(b); blocks.append(sp.csr_matrix(M)[f]); rhs.append(b[f])
    A = sp.vstack(blocks).tocsc(); b = np.concatenate(rhs)
    st = clarabel.DefaultSettings(); st.verbose = False
    st.tol_gap_abs = st.tol_gap_rel = st.tol_feas = 1e-12; st.max_iter = 500
    cones = [clarabel.ZeroConeT(int(eq.sum())), clarabel.NonnegativeConeT(A.shape[0] - int(eq.sum()))]
    r = clarabel.DefaultSolver(sp.eye(n, format="csc"), -v, A, b, cones, st).solve()
    return np.array(r.x), str(r.status)

if __name__ == "__main__":
    for p, tag in (("NCVXQP1", "DCA"), ("NCVXQP4", "InDCA1"), ("NCVXQP9", "InDCA1"), ("QPNBLEND", "DCA"),
                   ("BLOCKQP2", "InDCA1"), ("QPNBLEND", "InDCA2"), ("BLOCKQP4", "BDCA")):
        D = decode(p); x = np.load(RES / f"{p}_cutest_s0_{tag}.npy")
        y, st = clarabel_projection(D, x - (D["Q"] @ x + D["q"]))
        rep = json.load(open(RES / f"{p}_cutest_s0.json"))[tag]["kkt_res"]
        print(f"{p:9s} {tag:7s} Clarabel status={st:12s} r(x) Clarabel = {np.linalg.norm(x - y):.2e}   r(x) PIQP (reported) = {rep:.2e}")
