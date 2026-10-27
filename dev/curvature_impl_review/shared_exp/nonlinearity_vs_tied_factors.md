# Nonlinearity vs the number of tied Tucker factors

**Question (Nick):** how does the degree of nonlinearity scale with the number of tied factors m in one sharing group?

## Setup
- Snapshot `b6ad78b`, numpy eager, `bopt.newton_cg`. probe_derivatives order 2, ω = [1, .5, .25], N = 6, W = 60, 1 % noise, seeds 1–5. Measures and pencil as in `shared_vs_unshared_exact.md`.
- One group = modes 0..m−1, other modes singletons; m = 1 is untied on the plain MANIFOLD. Target: Tucker 5, TT (1,3,4,…,4,3,1), tied by averaging over the group. Level 1: (2,…)/(1,2,…,2,1), 40 GN steps, then the zero-padded restart at TT (1,2,3,…,3,2,1). Shared restart: group rank 3 at each group mode, which is above mode 0's own ceiling of 2 (the group-ceiling case). Unshared restart: Tucker (2,3,…,3,2).
- Rows: "tied / shared" fits a tied target on `shared_manifold(sharing)`. "tied / unshared" fits the same data on MANIFOLD; this is the control for "different target". "untied / unshared" is m = 1.
- Exact = TR, ≤ 40 steps, gtol 1e-8, from the restart (the GN warm-start fallback was needed 2/90). GN = line search from the restart, capped at 60. The pencil (K, H_GN) is taken on an orthonormal basis of the tied horizontal space at the exact x\*. Its dimension equals `manifold_dim` in every run, with asymmetry ≤ 2e-15.
- d = 6 has 55 new cells. d = 4 has 20 new cells (m = 2, 3); its m = 1, 4 rows are seeds 1–5 of the earlier run's untied / (0,0,0,0) cells, which are identical problems (ρ = 0.759 reproduced exactly). d = 8 was not run (cost). Each entry is the median [range] over 5 seeds. κ is the seed-median of the per-run median |κ|, with (max) the per-run maxima.

## d = 6 (dim = tied horizontal-space dimension; wall in s on a contended CPU; GN "(n)" = runs capped at 60 of 5)
| m | target / fit | dim | rel misfit | **ρ(H_GN⁻¹K)** | λ-RMS | ‖Kp‖/‖H_GN p‖ | κ med (max) | exact outer / CG / wall | GN outer / CG / wall (capped) | GN tail rate | cond H_GN |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | untied / unshared | 115 | 0.29 | 0.90 [0.82–0.95] | 0.26 [0.23–0.29] | 0.12 [0.08–0.16] | 0.44 (0.96) | 19 / 121 / 8.3 | 60 / 551 / 14.3 (5) | 0.90 [0.84–0.93] | 53 |
| 2 | tied / **shared** | 109 | 0.35 | 0.78 [0.67–0.96] | 0.26 [0.23–0.32] | 0.13 [0.08–0.23] | 0.33 (0.90) | 18 / 101 / 7.6 | 60 / 546 / 12.8 (4) | 0.77 [0.67–0.93] | 34 |
| 2 | tied / unshared | 115 | 0.33 | 0.81 [0.70–0.94] | 0.24 [0.23–0.32] | 0.14 [0.09–0.20] | 0.45 (0.83) | 16 / 101 / 7.1 | 60 / 545 / 15.3 (3) | 0.81 [0.67–0.91] | 32 |
| 3 | tied / **shared** | 100 | 0.40 | 1.03 [0.78–1.06] | 0.32 [0.29–0.33] | 0.18 [0.13–0.19] | 0.48 (0.84) | 23 / 149 / 10.3 | 60 / 307 / 7.0 (5) | 0.87 [0.79–0.90] | 41 |
| 3 | tied / unshared | 115 | 0.33 | 0.87 [0.76–0.97] | 0.27 [0.23–0.31] | 0.14 [0.11–0.17] | 0.49 (0.90) | 19 / 111 / 7.4 | 60 / 459 / 8.7 (5) | 0.88 [0.76–0.92] | 34 |
| 4 | tied / **shared** | 91 | 0.46 | 0.83 [0.77–1.01] | 0.26 [0.25–0.38] | 0.14 [0.10–0.18] | 0.32 (0.75) | 15 / 92 / 5.7 | 60 / 466 / 9.0 (5) | 0.80 [0.77–0.88] | 62 |
| 4 | tied / unshared | 115 | 0.32 | 0.92 [0.73–0.95] | 0.27 [0.26–0.30] | 0.12 [0.10–0.16] | 0.42 (0.91) | 18 / 147 / 5.9 | 60 / 518 / 9.2 (4) | 0.91 [0.72–0.94] | 81 |
| 5 | tied / **shared** | 82 | 0.48 | 0.88 [0.68–0.89] | 0.30 [0.22–0.33] | 0.16 [0.07–0.19] | 0.47 (0.89) | 16 / 100 / 7.1 | 60 / 484 / 8.3 (4) | 0.88 [0.68–0.89] | 33 |
| 5 | tied / unshared | 115 | 0.31 | 0.87 [0.74–1.10] | 0.28 [0.22–0.30] | 0.12 [0.09–0.18] | 0.44 (0.89) | 18 / 149 / 7.0 | 60 / 649 / 9.8 (5) | 0.87 [0.76–0.98] | 93 |
| 6 | tied / **shared** | 76 | 0.48 | 0.87 [0.76–1.17] | 0.31 [0.18–0.34] | 0.18 [0.07–0.19] | 0.37 (0.85) | 21 / 120 / 4.5 | 60 / 398 / 10.0 (5) | 0.87 [0.76–1.14] | 35 |
| 6 | tied / unshared | 115 | 0.30 | 0.84 [0.65–0.90] | 0.26 [0.17–0.28] | 0.14 [0.06–0.15] | 0.32 (0.82) | 20 / 113 / 4.3 | 60 / 462 / 6.0 (4) | 0.84 [0.64–0.90] | 56 |

## d = 4 (walls omitted: rows m = 1, 4 come from a different session)
| m | target / fit | dim | rel misfit | **ρ(H_GN⁻¹K)** | λ-RMS | ‖Kp‖/‖H_GN p‖ | κ med (max) | exact outer / CG | GN outer / CG (capped) | GN tail rate | cond H_GN |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | untied / unshared | 61 | 0.26 | 0.66 [0.31–0.79] | 0.19 [0.10–0.24] | 0.14 [0.08–0.18] | 0.27 (0.84) | 13 / 49 | 49 / 297 (1) | 0.69 [0.31–0.79] | 7.1 |
| 2 | tied / **shared** | 55 | 0.23 | 0.56 [0.43–0.94] | 0.20 [0.12–0.25] | 0.16 [0.09–0.21] | 0.15 (0.86) | 9 / 39 | 32 / 149 (1) | 0.56 [0.42–0.93] | 8.4 |
| 2 | tied / unshared | 61 | 0.13 | 0.54 [0.44–0.99] | 0.15 [0.12–0.25] | 0.12 [0.09–0.20] | 0.12 (0.78) | 10 / 46 | 29 / 159 (1) | 0.54 [0.41–0.97] | 7.7 |
| 3 | tied / **shared** | 46 | 0.38 | 0.72 [0.42–0.76] | 0.24 [0.16–0.28] | 0.19 [0.12–0.23] | 0.32 (0.70) | 12 / 43 | 56 / 292 (2) | 0.72 [0.44–0.76] | 7.1 |
| 3 | tied / unshared | 61 | 0.19 | 0.72 [0.48–0.96] | 0.20 [0.15–0.31] | 0.15 [0.11–0.27] | 0.24 (0.71) | 10 / 43 | 47 / 197 (1) | 0.61 [0.50–0.76] | 8.5 |
| 4 | tied / **shared** | 40 | 0.43 | 0.69 [0.52–0.76] | 0.24 [0.18–0.30] | 0.19 [0.16–0.25] | 0.27 (0.72) | 12 / 42 | 51 / 248 (2) | 0.70 [0.51–0.80] | 5.8 |
| 4 | tied / unshared | 61 | 0.21 | 0.73 [0.52–0.89] | 0.21 [0.16–0.27] | 0.16 [0.10–0.22] | 0.29 (0.67) | 10 / 46 | 53 / 274 (2) | 0.73 [0.52–0.89] | 8.3 |

## Trends vs m (Δ = paired per-seed difference between consecutive m, shared series, d = 6, "k/5" = seeds with Δ > 0)
- **ρ: flat, noisy, non-monotone.** Δ: 1→2 −0.13 (2/5), 2→3 +0.11 (5/5), 3→4 −0.17 (0/5), 4→5 −0.04 (1/5), 5→6 +0.07 (4/5). Per-seed Spearman(m, ρ) is +0.49, +0.20, −0.14, −0.14, −0.37. The control on the same data is also flat (0.81–0.92). Shared − unshared per m: −0.03, +0.09 (5/5), −0.03, −0.01, +0.09 (4/5). At d = 4 the shared series is 0.66 → 0.56 → 0.72 → 0.69, and the control moves in step (0.54 → 0.72 → 0.73).
- **λ-RMS: mildly up, noisy** (0.26 → 0.31, about +19 % by m = 6; d = 4: 0.19 → 0.24, Δ +0.03 / +0.04 / +0.03, each 4/5). The control is flat. Shared − unshared at d = 6: +0.02 (5/5), +0.03 (5/5), −0.01 (2/5), +0.02 (4/5), +0.05 (5/5).
- **‖Kp‖/‖H_GN p‖: mildly up, noisy** (0.12 → 0.18; d = 4: 0.14 → 0.19). The control is flat (0.12–0.14). Shared − unshared at d = 6: −0.00, +0.03 (5/5), +0.02, +0.01, +0.03 (5/5).
- **κ: flat** (median 0.32–0.48, max 0.75–0.96, in no order with m). **Exact Newton: flat** (15–23 outer, 92–149 CG at d = 6; 9–13 at d = 4). GN is capped at 60 in 49/55 d = 6 runs, so GN/exact ≥ about 3 at every m.
- **Misfit: monotone up when shared** (d = 6: 0.29 → 0.35 → 0.40 → 0.46 → 0.48 → 0.48). The control stays at 0.30–0.33. The gap over the control is +0.03, +0.10, +0.09, +0.12, +0.18, 5/5 seeds at every m; at d = 4 it is +0.04, +0.10, +0.23 (5/5). **Dimension: monotone down, exactly** (115, 109, 100, 91, 82, 76; d = 4: 61, 55, 46, 40).

## Interpretation
- **The nonlinearity that sets GN's rate does not grow with the number of tied factors.** ρ(H_GN⁻¹K) is flat within seed noise from m = 1 to m = d at both d. The m-dependence it does show is shared by the unshared control on the same data, so it comes from the target, not the ties. κ and the exact-Newton step counts are flat too, and the exact Hessian's advantage is the same at every m.
- **Average curvature rises mildly with m, and entirely through the residual.** λ-RMS rises about 20–25 % and the random-tangent ratio about 35–50 % by m = d. The cause is that one group-rank-3 subspace has to serve m modes of a group-rank-5 target, so the shared misfit grows monotonically, and more of the residual is normal to a smaller manifold. Average curvature scales as about misfit^0.5–1 (R² 0.5–0.85). With log misfit in the regression, m adds nothing: ×0.96–1.03 per tied factor, R² gain ≤ 0.04.
- **No single quantity explains the extreme eigenvalue.** At d = 6, ρ is uncorrelated with misfit (Spearman +0.14) and with relative σ_min/‖X‖, the group/bond rank deficiency (≈ 0). Misfit, σ_min, cond H_GN and dimension together give R² ≤ 0.13. At d = 4 the correlation with misfit is +0.55. One explanation fits the data but is not shown. The tied space shrinks by 6–9 dimensions per factor, and restricting a pencil cannot raise its extreme eigenvalue, so this would offset the larger normal residual. That would explain why the average curvature rises while the maximum does not. Strictly, the shared tangent space is not a subspace of the unshared one (mode 0 has rank 3 vs 2, plus a Weingarten term). The seed (probes and target) dominates ρ: seed 5 is lowest at every m ≥ 2.
- **d matters more than m.** At every m, ρ is about 0.8–0.9 at d = 6 and about 0.55–0.75 at d = 4, with cond H_GN 30–90 at d = 6 against 6–9 at d = 4 (W = 60 is fixed while the dimension doubles). GN stalls (ρ ≥ 1) in 5/25 shared d = 6 fits (m = 3 ×3, m = 4, m = 6), 1/25 controls and 0/5 untied. At d = 4 the count is 0/30, and the earlier experiment had 0/18 shared against 2/27 unshared, so the counts are too small to call this a tail effect.
- **The evidence is thin.** There are 5 seeds per cell, and ρ's seed spread (about 0.3) exceeds every m-effect. There is one template: the residual growth is built in by group rank 3 < target 5, and a well-specified tied target would not show it. The group is always the leading modes. The regime is heavily misspecified (misfit 0.2–0.6). d = 8 was not run, and σ_min/‖X‖ is available for 70/90 runs.

## Re-run (from this directory, the snapshot root)
```
bash run_scaling_all.sh 6 "1 2 3 4 5"        # 55 cells; 2 workers; ~13-25 s per cell uncontended
for s in 1 2 3 4 5; do for m in 2 3; do for g in shared unshared; do OMP_NUM_THREADS=1 PYTHONPATH=$PWD python run_scaling.py 4 $s $m $g; done; done; done
PYTHONPATH=$PWD python scaling_tables.py 6; PYTHONPATH=$PWD python scaling_mechanism.py 6    # and 4 (m = 1, 4 read results_W60/)
```
Scripts: `run_scaling.py` (one cell, reuses `exp_common.py`), `run_scaling_all.sh`, `scaling_tables.py`, `scaling_mechanism.py`. Raw JSON is in `results_scaling/`.
