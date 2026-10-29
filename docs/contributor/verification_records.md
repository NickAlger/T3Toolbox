# Verification records

Recorded numerical evidence behind claims the user docs state qualitatively. These are lab-notebook
entries: the numbers are from the referenced verification runs, and the corresponding automated
tests are the living guarantee.

## The three sampling transposes are Riemannian-compatible

(Stated qualitatively in `../entries_apply_probe.md` §4.) For the least-squares objective
`f(X) = ½‖S(X) − b‖²` on the fixed-rank manifold, with `S` each of the three sampling operations,
the matrix-free Riemannian gradient
`g = MANIFOLD.project(op_transpose(r, …, sum_over_probes=True))` and the Gauss-Newton
Hessian `H V = (𝒥ᵀ 𝒥 V)` gauged were checked end-to-end at an orthogonal, minimal-rank frame:

| operator | adjoint `⟨r, 𝒥V⟩ = ⟨𝒥ᵀr, V⟩` | gradient vs finite-diff *along the retraction* | Gauss-Newton Hessian   |
|----------|-------------------------------|-----------------------------------------------|------------------------|
| entries  | exact                         | rel. err ~5e-8                                | symmetric ~5e-17, PSD  |
| apply    | exact                         | rel. err ~3e-7                                | symmetric ~1e-12, PSD  |
| probe    | exact                         | rel. err ~5e-7                                | symmetric ~3e-13, PSD  |

The adjoint identity and gradient/Hessian checks are pinned by the fitting and manifold test
suites (`tests/test_fitting.py`, `tests/test_manifold.py`).

## The exact Riemannian Hessian (2026.3.0)

(Stated qualitatively in `../fitting_and_optimization.md` §4.12; the design and the facts table are
[`curvature_internals.md`](curvature_internals.md).) The measurements below come from the writer/critic
verification scripts of the implementation plan and from each implementation slice's handoff and
independent review (the slice records, archived under `dev/archive/`). The automated guarantees are
`tests/backend/test_curvature.py`, `tests/backend/test_uniform_curvature.py`, the exact oracle sweep
(`tests/oracle_sweeps/sweep_exact.py`) and the dispatch tests.

### Correctness

| check | result |
|---|---|
| probe-derivative two-hole vs the block-sum oracle, orders 0–3, `K ∈ {(2,), (2,3)}`, `C ∈ {(), (2,)}`, d ∈ {4, 3, 1}, slack frames | ≤ 4.4e-16 (plan bar 1.0e-15) |
| the recurrence kernels vs the dense `_trs` twins (orders 0–4) | ≤ 5.8e-16 (bar 1e-14) |
| apply / entries derivative two-holes vs an independent dense oracle (the review's own einsum, no FD, no library kernel), orders 0–3 | 9.7e-16 |
| plain probe / apply / entries two-holes: entries == apply at the one-hot vectors | bitwise |
| `<p, H p>` vs `d²/dt² f(R(tp))` (Richardson), resolving the curvature term itself | within 1e-3 of the curvature term (T7, every kind) |
| the exact `hvp` vs the dense extrinsic oracle (Weingarten form through the exact Jacobians), every kind × geometry × representation × sharing × weight × regularizer, d ∈ {3, 2, 1} | always-on 720 cases, max 2.5e-14; slow tier 8000 cases (4 seeds, `W ∈ {(12,), (3, 4)}`, orders 1–3), max 2.6e-11 (bar 1e-9); a Gauss-Newton mutant fails 72/72, a per-mode-lift mutant 9/9 |
| `jax.jvp` of each corewise transpose along θ̂ vs its two-hole (float64, 7 ragged + the uniform probe derivative) | ≤ 1e-12 |
| uniform two-holes vs ragged, per element (varying-rank stacks, forced padding, a TT-slack element) | 5.4e-16 (kernels), 1.4e-14 (model hvp); finite garbage in θ, θ̂ (boundary bonds included), the frame and the samples: bitwise |
| shared: the tied lift identity / the shared retraction's FD ratios / the dense Lagrangian of `Φ ∘ T` | 4.7e-16 / 4.0, 4.25 (O(h²)) / 1.1e-14 (a per-mode lift: off by 0.68–3.4) |
| downstream (T3Polynomial e01, L3 checkpoint, dof 1001): `‖H_fd p − hessian(p)‖ / ‖hessian(p)‖` along the retraction, `t = 1e-2 … 3e-6` | O(t²) down to a ~1e-9 floor (4.3e-2 → 4.3e-9); Gauss-Newton off by 0.10–0.94 at every `t` |

### Convergence

- **E2 — the critic's first restart experiment was ONE unseeded draw.** `check_restart_tr.py` seeded only a
  `default_rng(7)` stream, never the legacy `np.random` stream `TuckerTensorTrain.randn` draws from, so its
  "exact + TR 13 / 69, 0 rejected, vs GN + Armijo 54 and GN + TR 52 / 352" is irreproducible at any commit. Cite
  the seeded rows instead (F19, F27).
- **F19 (seeded; probe derivatives, order 2, ω = `[1, .5, .25]`, restart rank 2 → 3, target 4, 5 % noise):**
  Gauss-Newton + Armijo 39 steps / 130 Hessian-applies; exact + TR 8 / 23 at forcing power 1.0, 10 / 28 at
  0.5, 0 rejected. **F26** (its reference interplay): zero 7.01e4 vs warm start 2.07e4 (ratio 0.30); GN 36 vs 39
  steps at `gtol_rel` 1e-8, 13 vs 15 at 1e-3; exact + TR 9/27 vs 10/28 (power 0.5), 7/24 vs 8/23 (1.0).
- **F27 (E2's three apply restarts, seeded):** warm/zero reference ratios 1.9e-2, 1.6e-14, 3.1e-1. The library's
  exact `newton_cg` reproduces the critic harness row for row (outer / rejected / CG): A 18/1/117 at `gtol_rel`
  1e-8, 17/1/92 at 1e-3; B 4/1/12 (`objective_floor`, noise-free: the zero reference declares the start
  converged); C 11/1/53 and 10/1/33.
- **F28 / F38 (no minimal reduction):** Tucker slack 20 outer / 4 rejected / 78 CG iterations → 2.6e-9; TT slack
  34 / 9 / 272 → 2.2e-12; the same counts from the same tensor at minimal ranks; `ε = 0` ≡ `1e-6‖X‖`;
  reproduced **exactly** by the library on plain `apply`; `apply_derivatives` order 1: 17/4/56 (2.3e-9),
  31/10/240 (2.0e-9). (Gauss-Newton + Armijo's 80-step stall from the TT-slack start is a property of the
  problem — identical from the minimal-rank tensor, f = 1.335520e3, E11 — not of the frame.)
- **F22 / F29 (plain apply, target `(4,4,4)`, rank-2 fit, W = 200, 1 % noise; the GN warm-up uses
  `g0_reference='warm_start'`):** GN ratios 0.56–0.68; exact power 1.0 `e_{k+1}/e_k² ≤ 0.917`; power 0.5
  `e_{k+1}/e_k^{1.5}` 0.43–0.98 in `[1e-6, 1e-1]`. With the zero-reference warm-up the power-1.0 constant is
  0.941 — the warm-up ends at a different point; the kernels are bit-identical. T16 itself is re-based on
  probe derivatives (order 1, W = 200, 6 GN warm-start steps, ≥ 2 window points per seed); an `apply_derivatives`
  rate test, if ever added, must draw its start by a predicate (exact Hessian PD on `range(Π)` and
  `ρ(H_GN⁻¹K) < 0.9`: one measured seed was genuinely indefinite, `ρ = 1.03–1.10`).
- **T21 / T25 re-measured (S5, S8):** T21 (restart, W = 60) ragged 8 outer / 0 rejected / 27 CG; uniform 8/0/27,
  the same final objective. T25 Tucker slack 14/1/26, TT slack 22/6/70 — uniform backend and frontend identical
  counts, losses row by row ≤ 1.9e-13. T22 (shared restart) ragged 10/0/41, uniform 13/0/40 (same final
  objective, tied), Gauss-Newton 25 steps to `max_newton`.

### Cost and memory

| measurement | result |
|---|---|
| exact / GN Hessian-apply, probe derivatives (numpy CPU; d5 r8 o2 W300 and d4 r16 o3 W2000) | S3: 2.2–2.7×; after the recurrence conversion (S3b) 2.8–3.5× — Gauss-Newton got 2.3× faster, the two-hole's dominant pieces were recurrences already (plan gate ≤ 3.5×) |
| the same for the other kinds (S10–S12) | probe 2.8–2.9×, apply 3.1–3.2×, entries 3.1–3.3×; apply derivatives 3.0–3.1× (d5) / 3.5–3.9× (d4) |
| downstream (T3Polynomial, W = 208, dof 153 / 425 / 1001) | exact hvp 3.0 / 1.9 / 3.5× a GN hvp; an exact model build 2.5–2.7× |
| two-hole per-row scratch vs the GN transpose (uniform, jax f32) | the TT-assembly kernel alone (unchunked `memory_analysis`): 2.2–2.8× after S3b (was 2.5–3.5×); the running chunked program (the estimator's slope, current jax): ~1.3–2.2× (~1× on jaxlib 0.4.30); unchunked peak 1.85–2.1× |
| the chunked probe-derivative two-hole at `estimate_chunk_size(hessian='exact')` | 1.25–1.61× the GN assembly at its own default chunk, each held at its own jet floor (near, not strictly under, the plan's 1.5× gate; recorded, not changed) |
| compile (fresh process, CPU x64, small problems) | the exact uniform CG compiles once per run, ~14.5–14.8 s vs ~9.2–9.7 s for Gauss-Newton |
| S0c uniform-transpose re-mask overhead | F34: 0.08–0.61 % of the transpose; re-measured ~1.1 % eager on a small transpose including the host mask build, ~0.13 % large, 0 under jit |

### The zero reference

- F30: per-layer LAPACK zero references differ ragged vs uniform by 0.56–1.93×, and 6 of 16 seeded `newton_cg`
  pairs drift; with one reference, 0 of 16. F31 / F37: the canonical frame is valid (orthogonality residual 0,
  base point 0, Π idempotent) at d = 1, 3, 4 and shared; ragged == packed uniform to ≤ 3.9e-16; a mode's factor
  is bit-identical when another mode's rank or size changes; level `r`'s factor is exactly the prefix of level
  `r + 1`'s (error 0.0 over r = 1…7); the 44 O2 ragged/uniform pairs agree to 3.5e-12.
- F32: on data with no energy on the leading coordinates, the LAPACK and identity-prefix completions give a
  reference of exactly 0; Haar seeds 0–5 give 9.5–24 there and 4.4e1–7.0e1 on generic data (LAPACK 7.0e1).
  E9: a mode rotation moves the reference by ≤ 1.6–1.7×; a nonzero start's `‖g‖` is invariant.
- Downstream (T3Polynomial e01): the library's zero reference vs the downstream's own `_zero_gradient_norm`
  (a zero frame from `zeros()`' LAPACK completion) is 2.18 / 0.73 / 0.94 / 0.97 by level — not a drop-in.

### The downstream measurement (S0b; T3Polynomial against the 2026.3.0 release candidate, small sizes)

- **Tail** (e01 L3 checkpoint, `gtol_rel` 1e-9, ≤ 30 steps): Gauss-Newton 30 steps (ρ 0.02–0.03), 817 CG,
  `‖g‖` 4.2e-2 → 2.2e-2; exact **6 steps**, 205 CG, → 1.1e-8 (3.5e-12 at power 1.0). Damping 0 / 1e-6 / 1e-3 give
  the same steps (206 / 205 / 207 CG): **1e-6 ≡ 0** for the default.
- **Per-level restarts from identical warm starts** (steps (rejected) / CG / final f): e01 L0 GN 60(cap)/149
  vs exact 15/54 (same f); L1 44/310 vs 21(1)/91; L2 60(cap)/584/1.8050 vs 23(3)/203/1.7999; L3 60(cap)/888/0.8354
  vs 24(4)/284/0.8367. o01 L1: GN 0.987 vs exact **1.108** — exact converged to a verified minimum (dense Hessian
  PSD, the 81 zero eigenvalues the gauge) of a **worse basin** from the zero-padded start; the trust region is
  radius-limited for ~13 early steps (Δ0 = the Cauchy length, doubling per boundary step). On the small-residual
  b01 toy exact took 27/26/15 steps vs GN's 20/17/10 at 2.4× the wall.

### The Newton-schedule experiments (I-36 inputs; the records are archived under `dev/archive/`)

- **Shared vs unshared** (probe derivatives order 2, d = 4, shape `(6,6,6,6)`, W = 60, 9 seeds (+3 at W = 120), a
  heavily misspecified high-residual target): Gauss-Newton's rate constant `ρ(H_GN⁻¹K)` at the minimizer is ≈ 0.7
  in every cell — shared `(0,0,1,1)` 0.76, `(0,0,0,0)` 0.69; unshared on the same data 0.71 / 0.73; untied 0.69 —
  with paired differences on both sides of zero. The exact Hessian's gain is the same size shared or unshared:
  3.5–5× fewer outer steps, 5–6.6× fewer CG, 1.6–2× less wall. The shared manifold is mildly more curved on
  average (1.7× on random tangents, λ-RMS 1.15×) but not in the top eigenvalue that sets GN's rate. GN's
  measured tail ratio matched `ρ` within 0.03 in 51 of 58 runs (`ρ_tr − 1 ≈ −⟨p, Kp⟩/⟨p, H_GN p⟩`).
- **Tied-factor scaling** (one group of `m = 1…d` modes, d ∈ {6, 4}, 5 seeds): `ρ(H_GN⁻¹K)` flat within seed noise
  (d = 6: 0.90 / 0.78 / 1.03 / 0.83 / 0.88 / 0.87; d = 4: 0.66 / 0.56 / 0.72 / 0.69), the same-data unshared control
  moving the same way; κ and the exact step counts flat. Average curvature rises mildly with `m` when shared
  (λ-RMS +20–25 %, random-tangent ratio +35–50 % at `m = d`), entirely through the shared fit's larger misfit
  (×0.96–1.03 per tied factor once misfit is in the regression). `ρ` depends on `d` more than on `m` (≈ 0.85 at
  d = 6, ≈ 0.65 at d = 4); no single quantity explains it (R² ≤ 0.13). Untested hypothesis: the tied space
  shrinks 6–9 dimensions per tied factor, capping the top eigenvalue while the residual raises the average.
- Both are thin: one setup each, numpy eager, a contended CPU, seed spreads (~0.3) larger than any effect.
