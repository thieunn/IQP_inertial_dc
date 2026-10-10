# Code for "Solving a Class of Nonconvex Quadratic Programs by Inertial DC Algorithms"

T.H. Cuong, Y. Lim, N.N. Thieu, N.D. Yen — Journal of Global Optimization, manuscript JOGO-D-24-00212

This folder reproduces every numerical result of the revised manuscript (Tables 1–7, Figures 1–2) and the
audit tables R1–R2 of the response letter.

```
code/            algorithms, CUTEst interface, batch runs, table builders
code/results3/   archived results of all 72 paired runs: one JSON file per (problem, start) and the
                 start and returned points of every method (.npy)
examples/        Example 1 (Tables 1-4), Example 2 (Figures 1-2), unit test
```

The archived results in `code/results3/` were regenerated with these scripts in October 2026. Running
`python3 assemble.py` on them reproduces Tables 5, 6, 7, R1 and R2 and all numbers quoted in Section 3.3
exactly, except the run times, which depend on the machine.

## 1. Requirements

| Component | Version used |
|---|---|
| Python | 3.13 |
| numpy / scipy | 2.5 / 1.18 |
| piqp (QP solver for all projections and subproblems) | 0.6.4 |
| pycutest | 1.8 |
| CUTEst, SIFDecode, ARCHDefs, MASTSIF | github.com/ralna (2025–2026), compiler gfortran, architecture `pc64.lnx.gfo`, double precision |
| clarabel (independent KKT check only) | 0.11 |
| matplotlib (figures only) | 3.11 |

```bash
pip install -r requirements.txt
```

CUTEst must be installed as described in the pycutest documentation, with the environment variables
`ARCHDEFS`, `SIFDECODE`, `CUTEST`, `MASTSIF` (the directory of the SIF files, github.com/ralna/sif) and
`MYARCH=pc64.lnx.gfo` set. The examples in `examples/` do not need CUTEst.

## 2. Which script reproduces which part of the referee report

Run the commands from the folder indicated in the first column.

| Referee point / result | Command | Output |
|---|---|---|
| §3.4, minor 8: Example 1, Tables 1–4 (exact rational arithmetic, plus a check with the floating-point solvers) | `examples/`: `python3 make_example_3_1_tables.py` | `tables_example_1.tex`; KKT set and stopping indices 6, 22, 6, 19 printed |
| §3.4, minimum requirement 4: unit test, x¹ does not depend on γ | `examples/`: `python3 test_gamma_independence.py` | `all gamma-independence tests passed` |
| minor 11: Example 2, Cases 1–5 in exact arithmetic, Figures 1–2 | `examples/`: `python3 example2.py` | `Figure_1.pdf`, `Figure_2.pdf` |
| §3.1, §3.5, §3.10, minor 10, 12: all CUTEst runs (18 problems × 4 starts × 5 methods = 360 runs, plus the NCVXBQP companions) | `code/`: `bash run_cutest_batch.sh` | `results3/*.json`, archived start and returned points `results3/*.npy` |
| one run only, e.g. NCVXQP3 from the start of Tables 5–6 | `code/`: `python3 run_all_methods.py NCVXQP3 cutest 0` | `results3/NCVXQP3_cutest_s0.json` |
| §3.1: Tables R1 (import audit) and R2 (NCVXQP vs NCVXBQP); §3.5, §3.10, minor 12: Tables 5, 6, 7 | `code/`: `python3 assemble.py` (after the batch) | `tables/table5.tex`, `table6.tex`, `table7.tex`, `tableR1.tex`, `tableR2.tex` |
| numbers quoted in Section 3.3, items (a)–(e), and in the letter (feasibility maxima, number of QP solves and retries) | included in `python3 assemble.py` (or `python3 final_text.py`) | `tables/section33_numbers.txt` |
| §3.1: decoding with explicit SIF parameters, dimension, quadratic and linearity checks, SIF → `Ax ≥ b` mapping | `code/run_audit.py`, function `decode` | — |
| §3.1: P_C computed as a strongly convex QP; minor 10: failure handling | `code/indca_qp.py`, classes `_PolishedQP`, `Projector`, `QPSolver` | — |
| minor 9: observations with Tol = 10⁻³ | `code/`: `python3 run_audit.py NCVXQP3 --tol 1e-3` | `results/NCVXQP3_cutest_s0_t0.001.json` |
| §3.1: forensic rerun with the CUTEst equality rows removed | `code/`: `python3 run_audit.py NCVXQP1 --tol 1e-3` then `python3 drop_eq.py NCVXQP1 0.001` | printed |
| §3.1: independent check of the KKT residuals with a second QP solver (Clarabel) | `code/`: `python3 check_kkt_clarabel.py` (after the batch) | printed |

The batch runs two problems in parallel and skips pairs that are already finished, so it can be restarted.
It takes several hours; the run times reported in Tables 5–7 were measured on
[fill in: CPU, number of threads, operating system]. Run times depend on the machine; objective values,
iteration counts and KKT residuals do not.

## 3. Settings (Section 3.3 of the manuscript)

- **Problems**: BLOCKQP2–5, BLOWEYA–C, NCVXQP1–9 (SIF parameter N = 1000), QPNBAND (N = 10000), QPNBLEND.
- **Conversion to `Ax ≥ b`**: each equality row `J_i x + c0_i = cl_i` gives the two rows
  `J_i x ≥ cl_i − c0_i` and `−J_i x ≥ −(cl_i − c0_i)`; each finite side of an inequality row and each finite
  bound gives one row. All general constraints are passed to the solver (m_imported = m_CUTEst).
- **Methods**: DCA (γ = 0) and InDCA1 (γ = 0.5) with decomposition (a), η = λ_n(Q) + 2; DCA2 (γ = 0) and
  InDCA2 (γ = 0.5) with decomposition (b), η = −λ_1(Q) + 2; BDCA of Aragón Artacho–Campoy–Vuong (2022) with
  decomposition (a), line search from λ = min{2, λ̄_k}, test f(y + λd) ≤ f(y) − 0.1 λ²‖d‖², factor 0.5,
  performed only when d is a feasible direction at y.
- **Stopping**: Tol = 10⁻⁶ (both tests of Algorithm 1; BDCA stops when ‖y^k − x^k‖ ≤ Tol), at most 5000 iterations.
- **Subproblems**: PIQP (eps_abs = 10⁻⁹, eps_rel = 10⁻¹⁰), set up once per run, followed by exact active-set
  polishing; if PIQP fails, one retry with the preconditioner switched off, then the run is aborted.
- **Eigenvalues**: dense eigensolver for n ≤ 3000, Lanczos (`scipy.sparse.linalg.eigsh`) otherwise; not included
  in the reported times.
- **Starts** (identical for all methods): seed 0: x⁰ = P_C(x⁰_CUTEst + 0.01 ξ), ξ ~ N(0, I);
  seeds 1–3: x⁰ = P_C(ξ), ξ uniform in [max(b_l, −10), min(b_u, 10)].
- **Reported accuracy**: KKT residual r(x) = ‖x − P_C(x − (Qx + q))‖, linear violation ‖(b − Ax)₊‖_∞, bound violation.
- **Table 7**: f* is the best value found by any method from any start; a start "hits" f* if (f − f*)/|f*| ≤ 10⁻⁶.
  Medians over an even number of runs can be half-integers; the median iteration counts are rounded to the
  nearest even integer (Python `round`).

## 4. Note on the previous version

The results in Table 3 of the previous version were produced by an earlier code in which the linear equality
constraints of the CUTEst problems were lost during the conversion to `Ax ≥ b`. That code is not included;
all results of the revised manuscript come from the scripts in this folder.
