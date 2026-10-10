"""example2.py -- Example 2 (Section 4): exact (rational) recomputation of Cases 1-5 for InDCA1
(eta = 3, gamma = 1/3, rho = eta - lambda_2(Q) = 1) and Figures 1-2 (referee minor comment 11).

    min x1^2 - x2^2   s.t.  x in C = {x1 >= |x2|, x1 >= 1/4}

Every iterate is computed in exact rational arithmetic (fractions.Fraction) with an exact metric projection
onto C, the values stated in the manuscript are asserted, and the figures are drawn from these iterates.
Usage:  python3 example2.py        (writes Figure_1.pdf/.png and Figure_2.pdf/.png)
"""
from fractions import Fraction as F
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt


def proj(u):
    """exact metric projection onto C = {x1 >= |x2|, x1 >= 1/4} (polygon with 2 rays + 1 segment)."""
    y1, y2 = u
    if y1 >= abs(y2) and y1 >= F(1, 4):
        return (y1, y2)
    cands = []
    for s in (1, -1):                      # ray x2 = s x1, x1 >= 1/4
        t = max((y1 + s * y2) / 2, F(1, 4)); cands.append((t, s * t))
    cands.append((F(1, 4), min(max(y2, F(-1, 4)), F(1, 4))))   # segment x1 = 1/4
    return min(cands, key=lambda p: (p[0] - y1) ** 2 + (p[1] - y2) ** 2)


def indca1(x0, eta=F(3), gamma=F(1, 3), tol=F(1, 10**6), max_iter=200):
    """InDCA1 (formula (36)); returns the iterates [x^0, ..., x^{K+1}] and the stopping index K."""
    xs = [x0]; xp = x0; x = x0
    for k in range(max_iter):
        g = (2 * x[0], -2 * x[1])
        v = tuple((1 + gamma / eta) * x[i] - gamma / eta * xp[i] - g[i] / eta for i in range(2))
        xn = proj(v); xs.append(xn)
        step2 = sum((xn[i] - x[i]) ** 2 for i in range(2))
        d2 = gamma ** 2 * sum((x[i] - xp[i]) ** 2 for i in range(2))
        stop = (step2 == 0 and d2 == 0) if tol == 0 else (step2 <= tol ** 2 and d2 <= tol ** 2)
        xp, x = x, xn
        if stop:
            return xs, k
    return xs, None


CASES = {  # name: (x0, expected prefix x^0, x^1, ..., expected stopping index K)
    "Case 1 (t = 1/2)": ((F(3, 4), F(3, 4)), [(F(3, 4), F(3, 4))] * 2, 0),
    "Case 2": ((F(1, 4), F(0)), [(F(1, 4), F(0))] * 2, 0),
    "Case 3": ((F(1, 4), F(1, 8)),
               [(F(1, 4), F(1, 8)), (F(1, 4), F(5, 24)), (F(1, 4), F(1, 4)), (F(109, 432), F(109, 432))], None),
    "Case 4": ((F(1), F(0)), [(F(1), F(0)), (F(1, 3), F(0)), (F(1, 4), F(0)), (F(1, 4), F(0))], 3),
    "Case 5": ((F(1), F(1, 8)),
               [(F(1), F(1, 8)), (F(1, 3), F(5, 24)), (F(1, 4), F(1, 4)), (F(1, 4), F(1, 4)), (F(1, 4), F(1, 4))], 3),
}


def check_exact():
    runs = {}
    for name, (x0, expected, K_exp) in CASES.items():
        seq, K = indca1(x0)
        assert seq[:len(expected)] == expected, (name, seq[:len(expected)])
        if K_exp is not None:
            assert K == K_exp, (name, K)
        runs[name] = (seq, K)
        print(f"{name}: x^0..x^{len(expected) - 1} agree with the manuscript; stops at k = {K}; "
              f"last iterate = ({float(seq[-1][0]):.6f}, {float(seq[-1][1]):.6f})")
    for t in (F(0), F(1, 10), F(2), F(7, 3)):                         # Case 1 for several t >= 0
        x0 = (F(1, 4) + t, F(1, 4) + t)
        assert indca1(x0)[0][1] == x0
    seq, _ = indca1(CASES["Case 3"][0], tol=F(0), max_iter=40)        # Case 3: t_k increases to 97/384
    t = [x[0] for x in seq[2:]]
    assert all(a[0] == a[1] for a in seq[2:])
    assert all(t[i + 1] > t[i] for i in range(len(t) - 1)) and all(v < F(97, 384) for v in t)
    assert abs(float(t[-1]) - 97 / 384) < 1e-12
    print("Case 1 (x^1 = x^0 for t in {0, 1/10, 2, 7/3}) and Case 3 (t_k strictly increasing, t_k < 97/384, "
          "t_k -> 97/384) verified")
    # the projection arguments quoted in the text
    print("Case 3, argument of P_C for x^2:", tuple(str(v) for v in (F(4, 9) * F(1, 4) - F(1, 9) * F(1, 4),
                                                                     F(16, 9) * F(5, 24) - F(1, 9) * F(1, 8))))
    print("Case 4, argument of P_C for x^3: (%s, 0)" % (F(4, 9) * F(1, 4) - F(1, 9) * F(1, 3)))
    return runs


# ---- figures ------------------------------------------------------------------------------------------
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#d9d8d2"
C_GLOBAL, C_KKT = "#2a78d6", "#eb6834"
plt.rcParams.update({"font.size": 12, "axes.titlesize": 12, "axes.labelsize": 12,
                     "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 10})
XL, YL = (0.0, 1.15), (-0.75, 0.75)


def base_axes(ax, xl=XL, yl=YL, label_C=(0.8, -0.45)):
    x1 = np.linspace(*xl, 500); x2 = np.linspace(*yl, 500)
    X1, X2 = np.meshgrid(x1, x2)
    inC = (X1 >= np.abs(X2)) & (X1 >= 0.25)
    ax.contourf(X1, X2, inC.astype(float), levels=[0.5, 1.5], colors=["#efeee9"])
    lev = [-0.6, -0.3, -0.1, -0.02, 0.02, 0.1, 0.3, 0.6, 0.9]
    cs = ax.contour(X1, X2, X1 ** 2 - X2 ** 2, levels=lev, colors=MUTED, linewidths=0.8, linestyles="dotted")
    ax.plot([], [], color=MUTED, lw=0.8, ls="dotted", label=r"level curves $x_1^2-x_2^2=c$")
    s = np.linspace(0.25, 3, 50)
    ax.plot([0.25, 0.25], [-0.25, 0.25], color=INK, lw=1.2)
    for sg in (1, -1):
        ax.plot(s, sg * s, color=C_GLOBAL, lw=3.2, solid_capstyle="round", zorder=2,
                label=r"global solutions $F_1\cup F_2$ (KKT)" if sg == 1 else None)
    ax.plot([0.25], [0.0], "o", ms=9, mfc="white", mec=C_KKT, mew=2, zorder=6,
            label=r"KKT point $(\frac{1}{4},0)$ (not a local solution)")
    ax.set_xlim(*xl); ax.set_ylim(*yl); ax.set_aspect("equal")
    ax.set_xlabel(r"$x_1$"); ax.set_ylabel(r"$x_2$")
    ax.grid(color=GRID, lw=0.5)
    if label_C:
        ax.text(*label_C, r"$C$", fontsize=14, color=INK)


def figure_1():
    fig, ax = plt.subplots(figsize=(5.2, 5.6), layout="constrained")
    base_axes(ax)
    ax.set_title(r"Example 2: $C$ and level curves of $f$")
    fig.legend(*ax.get_legend_handles_labels(), loc="outside lower center", ncol=1, frameon=False)
    for ext in ("pdf", "png"):
        fig.savefig(f"Figure_1.{ext}", dpi=300)


def draw_path(ax, xs, labels):
    ax.plot(xs[:, 0], xs[:, 1], "-", color=INK, lw=1.0, zorder=3)
    for i in range(len(xs) - 1):
        if np.linalg.norm(xs[i + 1] - xs[i]) > 1e-3:
            ax.annotate("", xy=xs[i + 1], xytext=xs[i], zorder=4,
                        arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.0, shrinkA=4, shrinkB=4))
    ax.plot(xs[1:, 0], xs[1:, 1], "o", ms=5, color=INK, zorder=5, label=r"iterates $x^k$ (arrows: $x^k\to x^{k+1}$)")
    ax.plot(*xs[0], "*", ms=15, mfc="white", mec=INK, mew=1.4, zorder=6, label=r"initial point $x^0$")
    for k, off in labels.items():
        ax.annotate(rf"$x^{{{k}}}$", xs[k], xytext=off, textcoords="offset points", fontsize=12)


def figure_2(runs):
    fl = {k: np.array([[float(a), float(b)] for a, b in seq]) for k, (seq, _) in runs.items()}
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 6.0), layout="constrained")
    ax = axes[0]
    base_axes(ax, xl=(0.12, 0.42), yl=(-0.06, 0.34), label_C=(0.36, 0.02))
    draw_path(ax, fl["Case 3"], {0: (8, -4), 1: (8, -4), 2: (-22, 4)})
    lim = 97 / 384
    ax.annotate(r"$x^k\to(\frac{97}{384},\frac{97}{384})$", (lim, lim), xytext=(0.29, 0.30),
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8), fontsize=11)
    ax.set_title(r"(a) Case 3: $x^0=(\frac{1}{4},\frac{1}{8})$ (enlarged)")
    ax = axes[1]
    base_axes(ax)
    draw_path(ax, fl["Case 5"], {0: (-6, -20), 1: (4, 6), 2: (-24, 4)})
    ax.set_title(r"(b) Case 5: $x^0=(1,\frac{1}{8})$")
    fig.legend(*axes[1].get_legend_handles_labels(), loc="outside lower center", ncol=3, frameon=False)
    fig.suptitle(r"InDCA1 on Example 2 with $\eta=3$, $\gamma=\frac{1}{3}$ ($\rho=1$)")
    for ext in ("pdf", "png"):
        fig.savefig(f"Figure_2.{ext}", dpi=300)


if __name__ == "__main__":
    runs = check_exact()
    figure_1(); figure_2(runs)
    print("Wrote Figure_1.pdf/.png and Figure_2.pdf/.png")
