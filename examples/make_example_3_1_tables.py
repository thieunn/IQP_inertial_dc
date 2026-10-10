"""
make_example_3_1_tables.py

Example 1 (Section 3.3):  min 1/2 x'Qx + q'x  s.t.  Ax >= b, with
    Q = [[1, 1], [1, -1]],  q = (1/2, 1/2),
    C = {x : 0 <= x1 <= 2, 0 <= x2 <= 2, x1 + 2 x2 >= 1},
    x^0 = (1, 0)  (a vertex of C),  eta = 4 for both algorithms,
    Tol = 1e-6,  gamma in {0, 0.9}.

1. Computes every iterate of InDCA1 and InDCA2 in EXACT rational arithmetic
   (each subproblem is a strongly convex QP on a polygon, solved exactly by
   enumerating active sets of its KKT system), and the KKT set of the IQP.
2. Re-runs the same four cases with the floating-point solvers of ../code/indca_qp.py
   (PIQP + active-set polishing) and checks that the two computations agree.
3. Writes the LaTeX tables  tables_example_1.tex  (Tables 1-4 of the manuscript).

Usage:  python3 make_example_3_1_tables.py
"""
from fractions import Fraction as F
from itertools import combinations
import math, os, sys

Q = [[F(1), F(1)], [F(1), F(-1)]]
q = [F(1, 2), F(1, 2)]
A = [[F(v) for v in r] for r in ([1, 0], [0, 1], [-1, 0], [0, -1], [1, 2])]
b = [F(v) for v in (0, 0, -2, -2, 1)]
X0 = [F(1), F(0)]
ETA = F(4)
TOL2 = F(1, 10**12)            # Tol = 1e-6, compared on squared norms
GAMMAS = (F(0), F(9, 10))


def mv(M, x):
    return [M[i][0] * x[0] + M[i][1] * x[1] for i in range(2)]


def f(x):
    Qx = mv(Q, x)
    return F(1, 2) * (x[0] * Qx[0] + x[1] * Qx[1]) + q[0] * x[0] + q[1] * x[1]


def feasible(x):
    return all(A[i][0] * x[0] + A[i][1] * x[1] >= b[i] for i in range(len(b)))


def solve_linear(M, r):
    n = len(r)
    M = [row[:] + [r[i]] for i, row in enumerate(M)]
    for c in range(n):
        p = next((i for i in range(c, n) if M[i][c] != 0), None)
        if p is None:
            return None
        M[c], M[p] = M[p], M[c]
        for i in range(n):
            if i != c and M[i][c] != 0:
                t = M[i][c] / M[c][c]
                M[i] = [u - t * v for u, v in zip(M[i], M[c])]
    return [M[i][n] / M[i][i] for i in range(n)]


def kkt_points(P, c):
    """All (x, lambda, W): Px + c - A_W^T lambda = 0, A_W x = b_W, x in C, lambda >= 0."""
    out = []
    for k in range(3):
        for W in combinations(range(len(b)), k):
            n = 2 + k
            M = [[F(0)] * n for _ in range(n)]
            r = [F(0)] * n
            for i in range(2):
                M[i][0], M[i][1], r[i] = P[i][0], P[i][1], -c[i]
                for j, w in enumerate(W):
                    M[i][2 + j] = -A[w][i]
            for j, w in enumerate(W):
                M[2 + j][0], M[2 + j][1], r[2 + j] = A[w][0], A[w][1], b[w]
            s = solve_linear(M, r)
            if s is not None and feasible(s[:2]) and all(l >= 0 for l in s[2:]):
                out.append((tuple(s[:2]), tuple(s[2:]), W))
    return out


def qp_argmin(P, c):
    xs = {x for x, _, _ in kkt_points(P, c)}
    assert len(xs) == 1, xs            # strongly convex => unique KKT point
    return list(xs.pop())


def step(alg, g, x, xm):
    d = [g * (x[i] - xm[i]) for i in range(2)]
    if alg == 1:   # (26): x^{k+1} = P_C(x^k + d^k/eta - (Qx^k+q)/eta)
        Qx = mv(Q, x)
        u = [x[i] + d[i] / ETA - (Qx[i] + q[i]) / ETA for i in range(2)]
        return qp_argmin([[F(1), F(0)], [F(0), F(1)]], [-u[0], -u[1]]), d
    P = [[Q[0][0] + ETA, Q[0][1]], [Q[1][0], Q[1][1] + ETA]]   # (27)
    return qp_argmin(P, [q[i] - ETA * x[i] - d[i] for i in range(2)]), d


def run_exact(alg, g):
    """[x^0, ..., x^{K+1}] and the stopping index K of Algorithm 1."""
    seq, xm, x = [X0], X0, X0
    for k in range(1000):
        xn, d = step(alg, g, x, xm)
        seq.append(xn)
        if sum((xn[i] - x[i]) ** 2 for i in range(2)) <= TOL2 and sum(v * v for v in d) <= TOL2:
            return seq, k
        xm, x = x, xn
    raise RuntimeError("no stop")


def num(v):
    v = float(v)
    return f"{0.0 if abs(v) < 5e-7 else v:.6f}"


def sci(v):
    return "0" if v == 0 else f"{v:.2e}"


LABELS = {(1, F(0)): "InDCA1-gamma0", (1, F(9, 10)): "InDCA1-gamma09",
          (2, F(0)): "InDCA2-gamma0", (2, F(9, 10)): "InDCA2-gamma09"}


def table(alg, g, runs):
    """One table per (algorithm, gamma), in the manuscript's style. Row k lists
    ||x^k - x^{k-1}|| and ||d^{k-1}|| = gamma ||x^{k-1} - x^{k-2}|| (x^{-1} = x^0)."""
    seq, K = runs[(alg, g)]
    lines = [r"\begin{table}[h]", r"\centering",
             rf"\caption{{\textbf{{InDCA{alg}}} with $\gamma={float(g):g}$ for "
             rf"Example~\ref{{eg3.1}} (stops at $k={K}$)}}",
             rf"\label{{{LABELS[(alg, g)]}}}", r"\small",
             r"\begin{tabular}{|c|c|c|c|c|}", r"\hline",
             r"$k$ & $x^k$ & $f(x^k)$ & $\|x^k-x^{k-1}\|$ & $\|d^{k-1}\|$ \\", r"\hline"]
    for k, x in enumerate(seq):
        if k == 0:
            st = dn = "--"
        else:
            st = sci(math.sqrt(sum(float(x[t] - seq[k - 1][t]) ** 2 for t in range(2))))
            prev = seq[k - 2] if k >= 2 else seq[0]
            dn = sci(float(g) * math.sqrt(sum(float(seq[k - 1][t] - prev[t]) ** 2 for t in range(2))))
        lines += [f"{k} & ({num(x[0])}, {num(x[1])}) & {num(f(x))} & {st} & {dn} \\\\", r"\hline"]
    lines += [r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def float_check(runs):
    """Same four runs with the floating-point solvers of ../code/indca_qp.py."""
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "code"))
    import numpy as np, scipy.sparse as sp
    from indca_qp import Projector, QPSolver, indca1, indca2
    Qf = np.array([[1., 1.], [1., -1.]]); qf = np.array([.5, .5])
    J = sp.csr_matrix([[1., 2.]]); cl = np.array([1.]); cu = np.array([np.inf]); bl = np.zeros(2); bu = 2 * np.ones(2)
    P = Projector(J, cl, cu, bl, bu); S = QPSolver(Qf + 4 * np.eye(2), J, cl, cu, bl, bu)
    for (alg, g), (seq, K) in runs.items():
        xs = []
        wrap = (lambda v: (xs.append(P(v)) or xs[-1])) if alg == 1 else (lambda c: (xs.append(S(c)) or xs[-1]))
        r = (indca1 if alg == 1 else indca2)(Qf, qf, wrap, np.array([1., 0.]), 4.0, float(g), 1e-6, 1000)
        assert r["iterations"] - 1 == K, (alg, g, r["iterations"], K)
        dev = max(max(abs(xf[i] - float(xe[i])) for i in range(2)) for xf, xe in zip(xs, seq[1:]))
        print(f"[float check] InDCA{alg}, gamma={float(g):g}: same stopping index k={K}, "
              f"max |float - exact| = {dev:.1e}")


if __name__ == "__main__":
    print("KKT points (x, f(x), multipliers of the active constraints):")
    seen = set()
    for x, lam, W in kkt_points(Q, q):
        if x not in seen:
            seen.add(x)
            print(f"  x = ({x[0]}, {x[1]}), f = {f(x)}, active rows {W}, lambda = "
                  f"{tuple(str(l) for l in lam)}")
    runs = {(alg, g): run_exact(alg, g) for alg in (1, 2) for g in GAMMAS}
    for (alg, g), (seq, K) in runs.items():
        x = seq[K]
        print(f"InDCA{alg}, gamma={g}: stops at k={K} and returns x^{K} = "
              f"({x[0]}, {x[1]}), f = {f(x)};  x^1 = ({seq[1][0]}, {seq[1][1]})")
    float_check(runs)
    with open("tables_example_1.tex", "w") as fh:
        fh.write("\n".join(table(alg, g, runs) for alg in (1, 2) for g in GAMMAS))
    print("Wrote tables_example_1.tex")
