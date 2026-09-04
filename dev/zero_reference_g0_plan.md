# The Newton termination reference in rank continuation — make the equal-ranks zero tensor the default

> **STATUS: PROPOSED (2026-09-03), for the next breaking change.** Nick's position, from long
> experience: the right reference for the relative Newton stop and the CG forcing term is basically
> always the gradient at the ZERO tensor of the level's own ranks. T3Polynomial made this structural
> in its fitting layer on 2026-09-03 (its `fitting._newton_fit` computes it unless `g0norm_newton` is
> passed; the warm-start reference is selectable only by name and warns). Downstream evidence in §0;
> a read-only survey of the library (Sonnet agent, 2026-09-03) in §1–§3; the proposed change in §4;
> open questions in §5. Nothing implemented upstream.

## 0. Why (the downstream evidence)

T3Polynomial's eigen example, 0.1 cell at the input scale κ_auto, adaptive continuation, 30 Newton
iterations per level (T3Polynomial `dev/eigenproblem_design_2026_09_02.md` §8.5):

| level | dof | ‖g₀‖ at the zero tensor of these ranks | warm-start ‖g₀‖ / zero reference | ‖g‖/‖g₀_zero‖ at iteration 10 / 20 / 30 |
|---|---|---|---|---|
| 1 | 417 | 10.0 | 1.03 | 3.8e-1 / 2.7e-2 / 9.0e-4 |
| 3 | 1617 | 14.4 | 0.37 | 1.0e-1 / 6.3e-2 / 4.2e-2 |
| 5 | 4609 | 17.7 | 0.21 | 6.1e-2 / 1.6e-2 / 1.3e-2 |
| 7 | 10545 | 22.2 | 0.13 | 3.3e-2 / 1.4e-2 / 9.1e-3 |
| 9 | 17917 | 27.3 | 0.053 | 1.7e-2 / 8.5e-3 / 5.4e-3 |

The zero reference grows with rank (the projected data gradient sees more tangent directions); the
warm-start gradient after a zero-padded `resize` is deflated to 5% of it at Tucker rank 17. A
tolerance stated against the warm start therefore means nothing consistent across levels — the
library default `gtol_rel = 1e-8` is unreachable and every level runs to `max_newton` — and the CG
forcing term η = min(½, √(‖g‖/g₀)) is referenced to a shrinking number. With 100 iterations instead of
30 the weighted validation misfit moved by < 1% at every level (the objective plateaus by iteration
~20), so the budget was never the issue.

**The warm-start-vs-zero A/B (same cell, same data, 30 iterations, `gtol_rel` 1e-3 against each
reference):** identical accuracy at every level (q 0.0792 vs 0.0784), the deflation measured directly
(warm-start g₀ 1.55 vs zero-tensor g₀ 27.3 at rank 17), and **1.9× the wall time for the zero
reference** (3,179 s vs 1,681 s): neither arm reaches its stop above rank 3, so both run to the cap,
and the only operative difference is the forcing term η = min(½, √(‖g‖/g₀)) — loose (η ≈ 0.4, 8–9 CG
steps) against the deflated warm start, tight (η ≈ 0.11, 22 CG steps) against the zero reference,
with the loose solves just as effective per Newton step here. Also measured: the zero-referenced
ratio at the objective plateau varies from 5e-3 (rank 17) to 6e-2 (ranks 7–9) across levels, so no
single `gtol_rel` tracks the plateau. Consequences for §4: (a) the zero reference is the right
*reference* (its ratios are comparable across levels; the warm start's are not); (b) a
relative-gradient stop alone is a poor stopping rule in continuation — an **objective-plateau
criterion** (relative decrease over k iterations; the library has none) is the needed complement;
(c) the forcing term's dependence on g₀ means the reference choice also sets the CG work — an η
floor (≈ 0.3–0.4 lost nothing here) or a smaller `cg_forcing_power` should ship alongside the
default swap, or the swap makes continuation ~2× slower at a fixed cap.

## 1. Inventory — every place a reference gradient norm enters

- **`newton_cg`** (`backend/optimizers.py` L646–770) is the only optimizer with the two-tier relative
  design: `computed_g0norm` (L704–716) = ‖g‖ at the actual start `x0`, cached from iteration 0
  (`gnorm if gnorm > 0 else 1.0`); `newton_ref` (L717) = `g0norm_newton` if given else that; the stop
  is `‖g‖ ≤ gtol_rel · newton_ref` (L719); `cg_ref` (L718) = `g0norm_cg` if given else `newton_ref`
  (chained fallback); the forcing term η = min(0.5, (‖g‖/cg_ref)**cg_forcing_power) (L727), so
  `cg_tol = η·‖g‖` (L728). Both overrides are plain `float | None` (L654–656); no other spelling
  exists. `NewtonInfo.g0norm` (L614) records the `newton_ref` in effect. The frontend
  `optimizers.newton_cg` (`optimizers.py` L318–439) forwards them via `**kwargs` (L333, L339–342).
- **`gradient_descent`** (L386–439): the same pattern with **no override**: `g0norm` (L413, L420–421)
  is always the initial ‖g‖ at `x0`, feeding the stop `gg**0.5 <= gtol_rel*g0norm` (L422). It would
  suffer the identical deflation with no escape hatch.
- **`mc_sgd`** (L442–484) and **`adam`** (L487–524): no relative test, no g0 (plateau of the smoothed
  full-batch loss; `max_iter` only).
- There is **no library `lbfgs`**; `examples/fit_hilbert_from_entries_lbfgs.py` L136–152 wraps scipy's
  L-BFGS-B with its own absolute `gtol`.
- **The display** (`backend/optimizer_display.py` L131–135, L168–185) only reports `gnorm/g0norm` and
  the "converged (‖g‖ ≤ gtol·‖g₀‖)" line; it does not say WHICH reference was in effect.

## 2. Continuation today

- `continuation_ranks` (`tucker_tensor_train.py` L4397–4515) and `resize` (L1123) are purely
  structural (T3-SVD / masks) — no coupling to optimizers or gradients. **No helper anywhere computes
  a reference gradient norm**; `docs/contributor/fitting_internals.md` L103–109 confirms the outer
  loop is deferred ("the Goal-1 `fit(...)` facade"), so "the rank-continuation machinery" today is
  four independent copy-pasted loops.
- `docs/rank_continuation.md` L103–109 and `docs/fitting_and_optimization.md` L464–472 **already
  document the deflation** and recommend pinning `g0norm_newton` to *"the initial ‖g‖ from the first
  continuation stage"* — ONE scalar carried from level 0's start (which, by the library's convention,
  is usually the rank-1 zero tensor). That is not the per-level zero tensor; it does not grow with
  rank (2 → 27 above), so it over-tightens later levels in the other direction.
- Only one of four shipped continuation examples follows even that recipe:
  `examples/fit_shared_factors_jetted_probes.py` L121–130 (`g0 = stats['history'][0]['gnorm']`,
  reused every level) and its mirror `tests/test_sharing.py` L1077–1102. The other three pin nothing
  and exhibit the pathology: `examples/fit_varied_rank_tensor_newton_cg.py` L167–207
  (`riemannian_newton_cg` resets `g0norm = None` at L172 on EVERY call — each level recomputes g₀ from
  its deflated warm start); `examples/fit_hilbert_uniform_newton_cg.py` L147–148 (no
  `g0norm_newton`, although `rank_continuation.md` L184 names it as the reference uniform pattern);
  `examples/fit_hilbert_uniform_probe_derivatives_newton_cg.py` L163–168 (same gap).

## 3. Computing the zero-tensor reference

- **No new construction is needed.** A continuation warm start is `X0 = X.resize(shape, tucker, tt)`
  — cores already shaped to the level's ranks — so the zero tensor at these ranks is
  `cw.corewise_zeros_like(x0)` (`corewise.py` L116–129, `= corewise_scale(X, 0)`, already used for
  `adam`'s moment init at `backend/optimizers.py` L515–516). Identical for ragged `.data`, uniform
  `(tucker_supercore, tt_supercore)` (masks live in the geometry, untouched) and shared points.
- Then `g0 = geom.inner(g, g)**0.5` with `g = problem.local_model(corewise_zeros_like(x0)).gradient`
  — **one extra `LocalModel` build per level** (frame + sweep + `kind.transpose`), the cost of one
  Newton gradient. It must **reuse the level's own `Problem`** (same sample/data/kind/geom/chunk):
  on uniform `probe_derivatives`, `chunk_size='auto'` compiles to measure scratch
  (`optimizers.py` L145–166, `_resolve_chunk_size`), so a fresh `_setup`/`newton_cg(x0=zeros,
  max_newton=0)` would pay that compile twice per level.
- **Sharing.** `t3_sharing_residual` (`backend/sharing.py` L198–262) guards the zero/zero case
  (`pos = denom > 0`, returns 0.0 at L255–259), so `zeros(shape, tucker, tt)` with the group's rank
  repeated at every member mode is **exactly tied** by construction (`has_shared_tucker_factors`
  True) — no `.share()` needed to sit on the shared submanifold; `tests/test_sharing.py` L957–958
  already fits from a plain `zeros` under `shared_manifold`. The "gated tied channel" warning
  (`docs/sharing.md` L184–188; `fit_shared_factors_jetted_probes.py` L37) is about the trajectory
  escaping a lower stratum, not about evaluating a gradient AT the zero point
  (`test_restart_escape_activates_new_directions`, `test_sharing.py` L1057–1072, and the g0norm
  test `test_optimizers.py` L262–275 both do exactly that). **So "same-ranks zero tensor" for a
  shared T3 = the zero tensor at the GROUP ranks, its gradient taken in the shared geometry** (the
  tied gradient is a different vector from the free one; T3Polynomial passes
  `shared_manifold(sharing)` for this reason).
- **Uniform.** Minimality is structural (`backend/uniform_fitting.py` L347–374, `compute_minimal_ranks`
  from shape/nominal ranks), so a zero tensor at ranks from `continuation_ranks` is minimal;
  `uniform_minimal` is a no-op on it.
- **One real trap (`docs/fitting_and_optimization.md` §4.11, L438–450):** at a fully degenerate zero
  point the ragged and uniform layers pick different orthonormal completions — same objective,
  **different gradient** (measured: ragged 8.8e-16 vs uniform stalled at 8.49 on an ill-conditioned
  `entries` fit). The zero-tensor gradient is representation-dependent: **compute it in the same
  layer/geometry as the level's fit**, never once in ragged for a uniform fit.

## 4. The change (proposed)

1. **Additive now, in any release:** a string spelling `g0norm_newton='zero'` (and `g0norm_cg='zero'`)
   on `newton_cg`, resolved once after `_prepare_jit_inputs` (L702) as
   `‖problem.local_model(corewise_zeros_like(x0)).gradient‖` in the problem's geometry, so it
   inherits x0's converted dtype/device and the level's `Problem`. The same spelling on
   `gradient_descent`, which today has no override at all. `NewtonInfo` gains the reference KIND
   so the display can print it ("converged (‖g‖ ≤ gtol·‖g₀‖, g₀ = zero-tensor)").
2. **The breaking change (next `YYYY.MINOR`):** `g0norm_newton` **defaults to `'zero'`** (and so, by
   the chained fallback, `g0norm_cg`), on `newton_cg` and `gradient_descent`. Nick: the equal-ranks
   zero tensor is the right reference basically always; a fresh nonzero start has no deflation to
   fix, but the start-point gradient is an arbitrary scale there while the zero-tensor gradient is
   the problem's intrinsic data-gradient scale, so nothing is lost. For a zero start — every
   non-continuation example (`fit_probe_display.py` L55, `fit_hilbert_regularized.py` L104,
   `fit_per_mode_weight_probes.py` L87) and the backend baseline test (`test_optimizers.py`
   L266–280) — the swap is a numeric no-op. It changes behaviour only for nonzero starts: a handful
   of tests (`test_optimizers_frontend.py` L341–355 `randn`/`x_true` starts;
   `backend/test_optimizers.py` L679, L692) — audit, and pin them to `'start'` where the test is
   about the old semantics. `'start'` is the explicit spelling of today's behaviour (a float stays
   a float).
   *(The surveying agent preferred keeping the opt-in spelling and making only the documented
   recommendation + examples the breaking move, on the grounds that `newton_cg` cannot know it is
   inside a continuation loop. Nick's ruling — default to the zero reference — is what §4.2 states;
   the agent's concern is met by the explicit `'start'` spelling and the audit.)*
3. **Docs/examples:** rewrite `docs/rank_continuation.md`'s recipe (L76–97, L172–182) and its
   shared-factor section, `docs/sharing.md`'s continuation section, and
   `docs/fitting_and_optimization.md` L464–472 to the per-level zero reference (dropping the "reuse
   level-0's g₀" advice, which the new default supersedes); add the pin to the two uniform examples
   (`fit_hilbert_uniform_newton_cg.py` L147, `fit_hilbert_uniform_probe_derivatives_newton_cg.py`
   L165) — under the default swap they need no change, only their prose; simplify
   `fit_shared_factors_jetted_probes.py` L121–130 and `test_sharing.py` L1077–1102 from manual
   level-0 tracking to nothing; leave `fit_varied_rank_tensor_newton_cg.py`'s hand-rolled loop with a
   comment that it does not inherit library fixes, or port it to `newton_cg`.
4. **CHANGELOG `### Changed — breaking`** with the measured table above, and a `release_notes.md`
   "Upgrading" paragraph: *pass `g0norm_newton='start'` to keep the start-point reference; the
   default is now the equal-ranks zero tensor; `stats['history'][0]['gnorm']` bookkeeping in
   continuation loops can be deleted.* Verification record: the deflation table (§0) and the
   downstream A/B when it lands.

## 5. Risks / open questions

- **The forcing term and the reference are coupled** (§0's A/B): switching the default reference
  tightens CG ~2× at high rank for the same accuracy at a fixed cap. Ship an η floor or a lower
  `cg_forcing_power` default with the swap, and add an objective-plateau stop (`ftol`-style:
  relative decrease of the objective over k iterations) so levels can terminate where the objective
  actually flattens — the gradient ratio at the plateau spans an order of magnitude across ranks.

- **`g0norm_cg` inherits `g0norm_newton`** by the chained fallback (L718): `g0norm_newton='zero'`
  alone changes BOTH tests; `g0norm_cg='zero'` alone leaves the Newton stop on the old reference
  while tightening CG — state it in the docstring.
- **The regularizer is harmless at the zero reference**: `point_tangent` / `v_X`
  (`fitting_internals.md` L69–78) vanishes at X = 0, so `IdentityRegularizer` adds nothing to g₀ —
  it stays a pure data-gradient scale.
- **jit:** the outer Newton loop, including the existing initial `local_model` call, runs eagerly;
  only `_cg_solve` is jitted (L598). One more eager `local_model` per level needs no static changes.
- **`chunk_size='auto'` double compile** if the helper is not implemented backend-first on the
  level's `Problem` (§3).
- **Cross-layer mixing** (§3's trap): never reuse a ragged g₀ for a uniform fit.
- **The `history[0]['gnorm']` pattern** becomes redundant; flag in the migration note rather than
  orphan it.
- **A nonzero-start fit with `gtol_rel` tuned against the start-point reference** (any user script)
  changes its stopping point under the default swap — the one genuine behavioural break; the
  `'start'` spelling and the CHANGELOG entry are the mitigation.

## 6. Addendum (2026-09-03, late): the "plateau" is Gauss–Newton's linear rate on a large residual

Nick's hypothesis, checked against the downstream Newton diagnostics: ρ = actual / GN-predicted decrease
starts at 0.6–0.9 and decays to 0.01–0.1 by the end of every continuation level (100-iteration runs
included), with α = 1 always accepted and the objective decrements shrinking geometrically at
0.90–0.99 per iteration; the gradient keeps decreasing slowly (to ~1e-2 of the zero reference)
rather than stopping. That is the textbook signature of GN on a nonzero-residual problem: the omitted
Σᵢ rᵢ∇²rᵢ (plus the manifold curvature term) dominates late, the model over-predicts 10–100×, and
convergence is linear at a rate set by the omitted term. Consequences for this plan:

- **Stopping.** A relative-gradient test is the wrong primary stop in continuation regardless of
  the reference (the gradient decays slowly by construction). Add an objective-decrement test —
  relative decrease over k iterations — and consider making it ρ-aware (stop when ρ has collapsed
  and the decrement is below tolerance). The zero reference stays right for the *forcing term* and
  for reading ratios across levels.
- **Rate.** The fix is a second-order correction to GN: a full Riemannian Newton step (needs the
  Hessian of the residual and the curvature term) or a structured quasi-Newton secant for
  Σ rᵢ∇²rᵢ (Dennis–Gay–Welsch / NL2SOL style) added to JᵀJ inside the same CG. The display already
  prints ρ; the extrapolated remaining decrease at fixed rank downstream was ≤ 0.1–1%, so the payoff
  is a faster, well-terminated level, not a different answer.

**Measured downstream (2026-09-03, T3Polynomial `scripts/x06_hessian_fd_probe.py`, `x07_hessian_free_newton.py`,
eigen note §8.5):** at a near-converged continuation level (rank 17, ρ 0.06) the true Riemannian
curvature along the GN direction is 1.95× the GN curvature (the neglected terms equal the GN action
there; 9% along random tangents), which reproduces ρ ≈ 0.05 exactly; the T3-SVD retraction is
second order tangentially (acceleration ∝ t², ⟨grad f, R″⟩ negligible); and a Hessian-free
Riemannian Newton-CG (central differences of the projected gradient along the retraction) has
ρ = 1.00 at every step and cuts ‖grad f‖ 10× in ten iterations — while the entire remaining
decrease at that rank was 3e-5 (1e-4 relative) and the test error did not move. So the second-order
correction's payoff is a *terminating* level and a short tail, not accuracy; the cheap version
(structured secant on the two r-proportional terms, or the FD action for a final polish) is the one
worth having, and an objective-decrement stop remains the right primary termination.
