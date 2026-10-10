"""
indca_qp.py -- InDCA1 (gamma>0) / DCA (gamma=0) for
        min 1/2 x'Qx + q'x   s.t.  x in C = {x : cl <= Jx <= cu, bl <= x <= bu}
with C handled EXACTLY by a convex QP solver (PIQP) at every iteration.

Mapping to the paper's one-sided form Ax >= b (eq. (13)):
    equality  J_i x = c_i          ->  J_i x >= c_i  and  -J_i x >= -c_i
    range     cl_i <= J_i x <= cu_i ->  J_i x >= cl_i and  -J_i x >= -cu_i (finite sides only)
    bounds    bl_j <= x_j <= bu_j   ->  x_j >= bl_j   and  -x_j >= -bu_j  (finite sides only)
PIQP takes the two-sided form l <= Mx <= u (plus equality rows and variable bounds) directly,
which describes the same set C.

P_C(v) in formula (26) is computed as the unique solution of the strongly convex QP
    min 1/2||x||^2 - v'x   s.t. x in C
(see class Projector).
"""
import time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


class _PolishedQP:
    """argmin 1/2 x'Px + c'x  s.t.  x in C = {Jx = c_eq (equality rows), hl <= Gx <= hu, bl <= x <= bu}.
    Stage 1: PIQP (proximal interior-point method, sparse KKT factorisation; eps_abs = 1e-9, eps_rel = 1e-10;
             set up once, only the linear term changes between calls; if the status is not 'solved' the call
             is repeated with the Ruiz preconditioner switched off, and if that fails a RuntimeError is raised).
    Stage 2: active-set polishing (as in OSQP): the constraints whose slack at the PIQP point is <= tau_act
             are taken as equalities and the resulting equality-constrained QP is solved exactly (sparse LU of
             its KKT matrix + 2 steps of iterative refinement). The polished point is accepted only if it is
             feasible (violation <= 1e-9 (1+|rhs|)), all inequality multipliers are >= -1e-9 (1+||c||_inf),
             and it lies within 1e-3 of the PIQP point; then it is the EXACT solution of the QP (KKT conditions
             hold). Otherwise the PIQP point is returned. Interior-point points are only sqrt(gap)-accurate
             (about 1e-5 near degenerate vertices), which is why polishing is used."""
    def _init(self, P, J, cl, cu, bl, bu, eps_abs=1e-9, eps_rel=1e-10, max_iter=3000):
        import piqp
        n = J.shape[1]
        eq = np.isfinite(cl) & np.isfinite(cu) & (cl == cu)
        J = sp.csr_matrix(J)
        self.P = sp.csc_matrix(P); self.n = n
        self.Aeq, self.beq = J[eq], cl[eq]
        self.G, self.hl, self.hu = J[~eq], cl[~eq], cu[~eq]
        self.bl, self.bu = bl, bu
        # all inequality rows in the form  M x >= r  (for polishing / checks)
        I = sp.eye(n, format="csr")
        parts, rhs = [], []
        for M, r in ((self.G, self.hl), (-self.G, -self.hu), (I, bl), (-I, -bu)):
            fin = np.isfinite(r)
            if fin.any():
                parts.append(sp.csr_matrix(M)[np.where(fin)[0]]); rhs.append(r[fin])
        self.Min = sp.vstack(parts, format="csr") if parts else sp.csr_matrix((0, n))
        self.rin = np.concatenate(rhs) if rhs else np.zeros(0)
        self.solvers = []
        for precond in (None, 0):
            s = piqp.SparseSolver()
            s.settings.eps_abs = eps_abs; s.settings.eps_rel = eps_rel; s.settings.max_iter = max_iter
            if precond is not None:
                s.settings.preconditioner_iter = precond
            s.setup(self.P, np.zeros(n),
                    sp.csc_matrix(self.Aeq) if self.Aeq.shape[0] else None, self.beq if self.Aeq.shape[0] else None,
                    sp.csc_matrix(self.G) if self.G.shape[0] else None,
                    self.hl if self.G.shape[0] else None, self.hu if self.G.shape[0] else None, bl, bu)
            self.solvers.append(s)
        self.piqp = piqp; self.nsolves = 0; self.nfallback = 0; self.primary = 0
        self.npolished = 0; self.npolish_rejected = 0

    def _polish(self, x, c):
        n = self.n
        slack = self.Min @ x - self.rin
        act = slack <= 1e-6 * (1.0 + np.abs(self.rin))
        MA = sp.vstack([self.Aeq, self.Min[np.where(act)[0]]], format="csc")
        rA = np.concatenate([self.beq, self.rin[act]])
        m = MA.shape[0]
        if m == 0:
            K = self.P.tocsc(); rhs = -c
        else:
            K = sp.bmat([[self.P, -MA.T], [MA, -1e-13 * sp.eye(m)]], format="csc")
            rhs = np.concatenate([-c, rA])
        try:
            lu = spla.splu(K)
            z = lu.solve(rhs)
            Kt = sp.bmat([[self.P, -MA.T], [MA, None]], format="csc") if m else K
            for _ in range(2):
                z = z + lu.solve(rhs - Kt @ z)
        except Exception:
            return None
        xp = z[:n]; mu = z[n:]
        scale_c = 1.0 + float(np.max(np.abs(c))) if c.size else 1.0
        if not np.all(np.isfinite(xp)):
            return None
        viol_in = self.Min @ xp - self.rin
        if viol_in.size and np.min(viol_in + 1e-9 * (1.0 + np.abs(self.rin))) < 0:
            return None
        if self.Aeq.shape[0] and np.max(np.abs(self.Aeq @ xp - self.beq) - 1e-9 * (1.0 + np.abs(self.beq))) > 0:
            return None
        neq = self.Aeq.shape[0]
        if mu[neq:].size and np.min(mu[neq:]) < -1e-9 * scale_c:
            return None
        if np.linalg.norm(xp - x) > 1e-3 * (1.0 + np.linalg.norm(x)):
            return None
        return xp

    def solve_c(self, c):
        c = np.asarray(c, dtype=float); self.nsolves += 1
        for j, idx in enumerate([self.primary, 1 - self.primary]):
            s = self.solvers[idx]; s.update(c=c)
            if s.solve() == self.piqp.PIQP_SOLVED:
                if j == 1:
                    self.nfallback += 1; self.primary = idx
                x = s.result.x.copy()
                xp = self._polish(x, c)
                if xp is None:
                    self.npolish_rejected += 1
                    return x
                self.npolished += 1
                return xp
        raise RuntimeError("PIQP failed with both settings")


class Projector(_PolishedQP):
    """P_C(v) = argmin 1/2||x||^2 - v'x over C (see _PolishedQP)."""
    def __init__(self, J, cl, cu, bl, bu, **kw):
        self._init(sp.eye(J.shape[1], format="csc"), J, cl, cu, bl, bu, **kw)

    def __call__(self, v):
        return self.solve_c(-np.asarray(v, dtype=float))


class QPSolver(_PolishedQP):
    """argmin 1/2 x'Px + c'x over C (see _PolishedQP)."""
    def __init__(self, P, J, cl, cu, bl, bu, **kw):
        self._init(P, J, cl, cu, bl, bu, **kw)

    def __call__(self, c):
        return self.solve_c(c)


def lin_violation(x, J, cl, cu):
    """||(b - Ax)_+||_inf over the general (non-bound) rows."""
    if J.shape[0] == 0:
        return 0.0
    Jx = J @ x
    lo = np.where(np.isfinite(cl), cl - Jx, 0.0)
    hi = np.where(np.isfinite(cu), Jx - cu, 0.0)
    return float(max(0.0, lo.max(initial=0), hi.max(initial=0)))


def bound_violation(x, bl, bu):
    lo = np.where(np.isfinite(bl), bl - x, 0.0)
    hi = np.where(np.isfinite(bu), x - bu, 0.0)
    return float(max(0.0, lo.max(initial=0), hi.max(initial=0)))


def kkt_residual(x, Q, q, proj):
    """r(x) = ||x - P_C(x - (Qx+q))||_2 ; zero iff x is a KKT point (AVI (17))."""
    return float(np.linalg.norm(x - proj(x - (Q @ x + q))))


def indca1(Q, q, proj, x0, eta, gamma, tol, max_iter):
    """InDCA1 exactly as in the manuscript: x^{-1} = x^0, d^k = gamma(x^k - x^{k-1}),
    x^{k+1} = P_C((1+gamma/eta)x^k - (gamma/eta)x^{k-1} - (Qx^k+q)/eta),
    stop when ||x^{k+1}-x^k|| <= Tol and ||d^k|| <= Tol (Euclidean norms).
    gamma = 0 gives the (non-inertial) projection DCA."""
    t0 = time.time()
    x_prev = x0.copy(); x = x0.copy()
    k = 0; converged = False
    while k < max_iter:
        d = gamma * (x - x_prev)
        v = x + d / eta - (Q @ x + q) / eta
        x_new = proj(v)
        step = np.linalg.norm(x_new - x); dn = np.linalg.norm(d)
        x_prev, x = x, x_new
        k += 1
        if step <= tol and dn <= tol:
            converged = True
            break
    return dict(x=x, iterations=k, converged=converged, time_s=time.time() - t0)


def indca2(Q, q, sub, x0, eta, gamma, tol, max_iter):
    """InDCA2 exactly as in the manuscript: x^{k+1} = argmin_{x in C} 1/2 x'(Q+eta I)x
    + <q - eta x^k - gamma(x^k - x^{k-1}), x>; same stopping rule as InDCA1.
    gamma = 0 gives the (non-inertial) proximal DCA (DCA2)."""
    t0 = time.time()
    x_prev = x0.copy(); x = x0.copy(); k = 0; converged = False
    while k < max_iter:
        d = gamma * (x - x_prev)
        x_new = sub(q - eta * x - d)
        step = np.linalg.norm(x_new - x); dn = np.linalg.norm(d)
        x_prev, x = x, x_new; k += 1
        if step <= tol and dn <= tol:
            converged = True; break
    return dict(x=x, iterations=k, converged=converged, time_s=time.time() - t0)


def bdca(Q, q, proj, J, cl, cu, bl, bu, x0, eta, tol, max_iter, lam0=2.0, alpha=0.1, beta=0.5,
         lam_min=1e-8, act_tol=1e-9):
    """Boosted DCA for linearly constrained DC programs (Aragon Artacho, Campoy, Vuong, SVVA 2022),
    with the projection decomposition f1 = eta/2||x||^2 + q'x + delta_C, f2 = 1/2 x'(eta I - Q)x:
      y^k = P_C(x^k - (Qx^k+q)/eta)                (DCA step)
      d^k = y^k - x^k; if d^k is a feasible direction at y^k (A_i d^k >= 0 for the constraints active
      at y^k) do a backtracking line search from lam = min(lam0, lam_max) (lam_max: largest step keeping
      y^k + lam d^k in C) until f(y^k+lam d^k) <= f(y^k) - alpha lam^2 ||d^k||^2; else lam = 0.
      x^{k+1} = y^k + lam d^k.   Stop when ||y^k - x^k|| <= Tol."""
    t0 = time.time()
    f = lambda z: 0.5 * z @ (Q @ z) + q @ z
    J = sp.csr_matrix(J)
    x = x0.copy(); k = 0; converged = False; nboost = 0
    while k < max_iter:
        y = proj(x - (Q @ x + q) / eta)
        d = y - x; nd = np.linalg.norm(d); k += 1
        if nd <= tol:
            x = y; converged = True; break
        Jy, Jd = J @ y, J @ d
        # slacks s >= 0 and their rates along d, for every finite side
        S = [Jy - cl, cu - Jy, y - bl, bu - y]; R = [Jd, -Jd, d, -d]
        ok = True; lam_max = np.inf
        for s_, r_ in zip(S, R):
            fin = np.isfinite(s_)
            s_, r_ = s_[fin], r_[fin]
            act = s_ <= act_tol * np.maximum(1.0, 1.0)
            if np.any(r_[act] < -1e-12):
                ok = False; break
            neg = (~act) & (r_ < 0)
            if neg.any():
                lam_max = min(lam_max, float(np.min(s_[neg] / -r_[neg])))
        lam = 0.0
        if ok:
            lam = min(lam0, lam_max); fy = f(y)
            while lam >= lam_min and f(y + lam * d) > fy - alpha * lam * lam * nd * nd:
                lam *= beta
            if lam < lam_min:
                lam = 0.0
        if lam > 0:
            nboost += 1
        x = y + lam * d
    return dict(x=x, iterations=k, converged=converged, time_s=time.time() - t0, boosted_steps=nboost)
