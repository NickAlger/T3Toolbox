# Exact vs Gauss-Newton Hessian: shared vs unshared Tucker factors

**Question (Nick):** if the shared manifold is more nonlinear, does the exact Hessian help more there than unshared?

## Setup
- Snapshot `b6ad78b`, numpy 2.5.3, eager, single thread, backend `bopt.newton_cg` (check-free, so no gauge false-fails). probe_derivatives order 2, ω = [1, .5, .25], d = 4, shape (6,6,6,6), W = 60 (plus a W = 120 check), 1 % noise.
- Target (5,5,5,5)/(1,3,4,3,1). It is tied for sharing (0,0,1,1) and for (0,0,0,0), and untied for baseline (c). Level 1 is (2,2,2,2)/(1,2,2,2,1) with 40 GN steps, then a zero-padded `resize` restart. Shared restart: (3,3,3,3)/(1,2,3,2,1), so group rank 3 exceeds the own ceiling of modes 0 and 3 (1·2 = 2): the group-ceiling case. Unshared restart: (2,3,3,2)/(1,2,3,2,1), the same TT ranks with Tucker clipped per mode. Horizontal dims are 49 / 40 / 61, matching `manifold_dim`.
- Matched pairs: shared and unshared each run their own continuation on the SAME data. Exact: TR, power 0.5, gtol 1e-8 (zero reference). GN: line search, power 0.5, one 72-row run read up to e ≤ 1e-8, capped at 60 steps. "warm" means 10 GN steps first.
- At the exact minimizer x\* I built the full pencil (K, H_GN) on an orthonormal basis of the (tied) horizontal space. Its ρ(H_GN⁻¹K) is §2.4's GN linear-rate constant. I also used 10 random gauged tangents. κ = pKp/(pHp−pKp) is taken on accepted exact steps.
- Run table: seeds 1–3, medians [range]. The spectrum table uses 9 seeds (W = 60) and 3 seeds at W = 120. e = ‖g‖/‖g₀‖.

## Table 1: runs (W = 60, seeds 1–3; GN "60" means capped)
| cell | start | exact: outer / acc / CG | exact tail e (last 4) | GN: outer / CG to e≤1e-8 | GN capped | GN tail e_{k+1}/e_k | GN steps to exact's final ‖g‖ | rel misfit | wall exact / GN s |
|---|---|---|---|---|---|---|---|---|---|
| (0,0,1,1) shared | restart | 11 / 11 / 44 [42–55] | 2e-03→5e-05→2e-07→4e-11 | 60 / 227 | 2/3 | 0.74 [0.64–0.92] | >60,>60,>60 | 0.31 | 1.4 / 2.2 |
| (0,0,1,1) shared | warm | 6 / 6 / 34 [20–40] | 1e-03→3e-05→9e-08→1e-11 | 60 / 265 | 1/3 | 0.79 [0.64–0.92] | 34,>60,>60 | 0.31 | 0.8 / 2.3 |
| (0,0,1,1) data, unshared | restart | 10 / 9 / 44 [43–44] | 2e-03→5e-05→2e-07→6e-11 | 46 / 289 | 1/3 | 0.71 [0.71–0.86] | 50,>60,58 | 0.22 | 1.3 / 2.3 |
| (0,0,1,1) data, unshared | warm | 5 / 5 / 31 [21–36] | 2e-03→6e-05→4e-07→2e-10 | 36 / 270 | 1/3 | 0.71 [0.71–0.86] | >60,>60,40 | 0.22 | 0.8 / 2.2 |
| (0,0,0,0) shared | restart | 13 / 13 / 42 [41–45] | 2e-03→6e-05→3e-07→9e-11 | 51 / 248 | 1/3 | 0.70 [0.63–0.76] | >60,>60,>60 | 0.48 | 1.1 / 2.1 |
| (0,0,0,0) shared | warm | 5 / 5 / 25 [22–34] | 3e-03→1e-04→2e-06→1e-09 | 41 / 234 | 0/3 | 0.70 [0.63–0.76] | 60,34,45 | 0.48 | 0.8 / 1.9 |
| (0,0,0,0) data, unshared | restart | 11 / 11 / 50 [39–65] | 2e-03→8e-05→5e-07→2e-10 | 60 / 274 | 2/3 | 0.83 [0.67–0.89] | >60,53,>60 | 0.25 | 1.3 / 2.7 |
| (0,0,0,0) data, unshared | warm | 6 / 6 / 39 [23–44] | 3e-03→2e-04→2e-06→2e-09 | 60 / 300 | 2/3 | 0.84 [0.67–0.89] | >60,38,>60 | 0.25 | 1.0 / 2.7 |
| untied, unshared | restart | 14 / 13 / 49 [47–56] | 3e-03→7e-05→6e-07→4e-10 | 49 / 297 | 0/3 | 0.69 [0.66–0.70] | >60,60,50 | 0.26 | 1.3 / 2.5 |
| untied, unshared | warm | 8 / 8 / 32 [28–41] | 4e-03→1e-04→1e-06→7e-10 | 39 / 279 | 0/3 | 0.69 [0.66–0.70] | 33,56,43 | 0.26 | 0.9 / 2.2 |

Every exact run stopped on `gradient`, with at most 2 rejections. Exact tail ratios were 0.02–0.05 → 3e-3–1e-2 → 2e-4–9e-4, which is superlinear. GN stops were `gradient` or `max_newton`. The usual reason GN cannot reach exact's final ‖g‖ is its Armijo floor (around e ≈ 4e-10, `line_search`) or its rate.

## Table 2: how much curvature matters (W = 60; ρ over 9 seeds, other columns are 9-seed medians)
| cell | dim | κ on accepted exact steps: median ‖κ‖ (max) | random p: ‖pKp/pH_GNp‖ ; ‖Kp‖/‖H_GN p‖ | λ-RMS of H_GN⁻¹K | **ρ(H_GN⁻¹K) at x\*** | GN measured tail rate | ρ at W=120 (3 seeds) | cond H_GN |
|---|---|---|---|---|---|---|---|---|
| (0,0,1,1) shared | 49 | 0.29 (0.83) | 0.033 ; 0.19 | 0.22 | 0.76 [0.44–0.92] | 0.74 [0.42–0.92] | 0.79 [0.75–0.82] | 6.4 |
| (0,0,1,1) data, unshared | 61 | 0.22 (0.69) | 0.019 ; 0.15 | 0.19 | 0.71 [0.46–0.86] | 0.71 [0.43–0.86] | 0.88 [0.79–0.90] | 6.9 |
| (0,0,0,0) shared | 40 | 0.27 (0.67) | 0.029 ; 0.19 | 0.24 | 0.69 [0.40–0.76] | 0.70 [0.40–0.80] | 0.56 [0.55–0.63] | 5.5 |
| (0,0,0,0) data, unshared | 61 | 0.35 (0.74) | 0.017 ; 0.17 | 0.21 | 0.73 [0.52–1.01] | 0.73 [0.52–0.89] | 0.72 [0.68–0.84] | 7.6 |
| untied, unshared | 61 | 0.34 (0.84) | 0.018 ; 0.18 | 0.21 | 0.69 [0.31–1.02] | 0.69 [0.31–0.82] | 0.64 [0.60–0.74] | 7.1 |

Paired ρ_shared − ρ_unshared on the same data:
- W = 60: (0,0,1,1) +0.06 [−0.10, +0.22], shared larger in 5/9 seeds; (0,0,0,0) −0.13 [−0.49, +0.21], shared larger in 3/9.
- W = 120: (0,0,1,1) −0.11 (1/3); (0,0,0,0) −0.16 (0/3).

GN/exact outer-step ratio from the restart (9 seeds, GN capped at 60, so these are lower bounds):
- shared: 3.5 [2.4–6.0] and 4.6 [2.0–5.7]
- unshared: 4.6 [2.9–6.7] and 5.0 [3.1–7.5]
- untied: 4.4 [2.1–5.7]

## Interpretation
- **No: the shared manifold is not measurably more nonlinear where it counts, so the exact Hessian's advantage is not larger there.** What matters is the GN rate constant ρ(H_GN⁻¹K) at the minimizer. It is about 0.7 in every cell (shared 0.76 / 0.69, unshared on the same data 0.71 / 0.73, untied 0.69). The paired differences fall on both sides of zero: shared is larger in 8/18 seed pairs at W = 60 and 1/6 at W = 120.
- **The exact Hessian's benefit is large everywhere and about the same size.** From the restart, exact takes 10–14 outer steps and 42–50 CG iterations; GN takes 46–60+ outer steps and 230–300 CG. That is 3.5–5× fewer outer steps, 5–6.6× fewer CG iterations and 1.6–2× less wall time. From the warm start, exact takes 5–8 outer steps against 36–60 for GN. The ratio is not larger when shared. These numbers match T21/T22 (exact 10 / 41 against unconverged GN).
- **The average curvature fraction is somewhat higher when shared.** The random-tangent ‖pKp/pH_GNp‖ is about 1.7× higher, and the λ-RMS is about 1.15× higher. This fits the shared fits' larger residual (rel misfit 0.31 / 0.48 vs 0.22 / 0.25; a smaller tangent space leaves more of the residual normal). The extreme eigenvalue is not larger, though, and GN's rate depends only on that one. GN on the shared manifold is also slightly better conditioned (cond 5.5–6.4 vs 6.9–7.6).
- **§2.4 holds quantitatively.** GN's measured tail ratio equals ρ(H_GN⁻¹K) within 0.03 in 51 of 58 runs. In 2 of the 27 unshared runs at W = 60 (none of the 18 shared), ρ ≥ 1: GN stalled at e ≈ 1e-2 after 60 steps while exact converged in 9 and 18 steps. These are the only cases where the exact Hessian's gain is unbounded, and both were unshared.
- **The evidence is thin in several ways:**
  - one template: probe_derivatives order 2, zero-padded continuation, one rank pattern per partition, group ranks far below N_g;
  - small sizes (dimensions 40–61, N = 6, d = 4) and 9 / 3 seeds, with a seed spread (0.3–1.0) much larger than any shared/unshared effect;
  - a heavily misspecified, high-residual regime (rel misfit 0.2–0.5), which is what makes K large; at low residual ρ → 0 for both;
  - numpy eager only, and wall times taken on a shared, contended CPU.

## Re-run (from the snapshot root, about 10 s per cell at W = 60 and about 16 s at W = 120)
```
cd <scratch>
for s in 1 2 3 4 5 6 7 8 9; do for c in "s0011 shared" "s0011 unshared" "s0000 shared" "s0000 unshared" "untied unshared"; do
  OMP_NUM_THREADS=1 PYTHONPATH=$PWD python run_case.py $s $c; done; done      # prefix T3EXP_W=120 for the W=120 check
python make_tables.py     # both tables -> stdout; python aggregate.py results_W60 1,2,3 gives per-seed detail
```
Scripts: `exp_common.py` (builders, summaries, pencil), `run_case.py`, `make_tables.py`, `aggregate.py`. Raw per-run histories are in `results_W60/`, `results_W120/`.
