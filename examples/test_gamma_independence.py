"""Unit test (Referee report Sec. 3.4, minimum requirement 4): the first iterate x^1 of InDCA1 and InDCA2
does not depend on gamma (because x^{-1} = x^0 gives d^0 = 0). Checked on Example 1 against the exact values
x^1 = (17/20, 3/40) (InDCA1) and (16/19, 3/38) (InDCA2), for gamma in {0, 0.3, 0.9}, and on random data.
Usage:  python3 test_gamma_independence.py      (or:  python3 -m pytest test_gamma_independence.py)"""
import sys, numpy as np, scipy.sparse as sp
import os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "code"))
from indca_qp import Projector, QPSolver, indca1, indca2

def ex1():
    Q = np.array([[1., 1.], [1., -1.]]); q = np.array([.5, .5])
    J = sp.csr_matrix([[1., 2.]]); return Q, q, J, np.array([1.]), np.array([np.inf]), np.zeros(2), 2 * np.ones(2)

def test_example1():
    Q, q, J, cl, cu, bl, bu = ex1(); x0 = np.array([1., 0.])
    P = Projector(J, cl, cu, bl, bu); S = QPSolver(Q + 4 * np.eye(2), J, cl, cu, bl, bu)
    for g in (0.0, 0.3, 0.9):
        x1a = indca1(Q, q, P, x0, 4.0, g, 0.0, 1)["x"]; x1b = indca2(Q, q, S, x0, 4.0, g, 0.0, 1)["x"]
        assert np.allclose(x1a, [17/20, 3/40], atol=1e-10), (g, x1a)
        assert np.allclose(x1b, [16/19, 3/38], atol=1e-10), (g, x1b)

def test_random(trials=20):
    rng = np.random.default_rng(0)
    for t in range(trials):
        n = 6; M = rng.standard_normal((n, n)); Q = (M + M.T) / 2; q = rng.standard_normal(n)
        J = sp.csr_matrix(rng.standard_normal((3, n))); cl = -np.ones(3); cu = np.full(3, np.inf)
        bl, bu = -2 * np.ones(n), 2 * np.ones(n); x0 = rng.uniform(-1, 1, n)
        lam = np.linalg.eigvalsh(Q); e1, e2 = lam[-1] + 2, -lam[0] + 2
        P = Projector(J, cl, cu, bl, bu); S = QPSolver(Q + e2 * np.eye(n), J, cl, cu, bl, bu)
        ref1 = indca1(Q, q, P, x0, e1, 0.0, 0.0, 1)["x"]; ref2 = indca2(Q, q, S, x0, e2, 0.0, 0.0, 1)["x"]
        for g in (0.3, 0.9):
            assert np.allclose(indca1(Q, q, P, x0, e1, g, 0.0, 1)["x"], ref1, atol=1e-9)
            assert np.allclose(indca2(Q, q, S, x0, e2, g, 0.0, 1)["x"], ref2, atol=1e-9)

if __name__ == "__main__":
    test_example1(); test_random(); print("all gamma-independence tests passed")
