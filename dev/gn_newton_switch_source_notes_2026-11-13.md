# GN -> exact-Hessian switch: source notes from nicks-optimizers (2026-11-13)

Read-only research for the 2026.5.0 `newton_cg` switch plan (`dev/HANDOFF.md:233-240`). Source:
https://github.com/NickAlger/nicks-optimizers at `1ab10dc` (2026-11-11), cloned for reading only (nothing
in it was run). Paths below marked **[no]** are relative to that repo; **[T3]** paths are this repo at
`04356d38` with the R6 working-tree edits, which leave line numbers unchanged in the two optimizer files.
Statements marked *(observation)* are mine, not the source's.

## 0. The arc

Thirteen commits on 2026-11-11, `8113dc2` (plan) to `1ab10dc` (plans archived): phase 0 vendored baseline
(`72b5f0b`), 1 forcing power (`935ba43`), 2 second operator + tracking product (`73920b7`), 3-4
`hessian="switch"` in both Newton optimizers (`9df0d63`), 5 strict backend + JAX (`49b246b`), 6 experiments
(`3381c3f`), D10 post-switch power 0.5 (`4d045c3`), 7 README/example/TODO (`e8a8271`), baseline deleted
(`7853012`). Before that, the repo's arc was L-BFGS (2026-10-26).

- Plan, with decisions D1-D10 and progress: **[no]** `dev/archive/gauss_newton_switch_plan.md` (501 lines).
- Code: `src/nicks_optimizers/newton_switch.py` (keywords, tracker, work split, display), `criteria.py`
  (`switch_sample`), `newton_core.py` (`PointOperators.gn_hessvec`, `CoreProblem.model`, `tracked_models`),
  `trust_region_core.py` / `newton_line_search_core.py` (`track=`), `newton_driver.py` (`PhasePieces`),
  `trust_region.py` / `newton_line_search.py` (drivers), `trust_region_diagnostics.py` (records, summaries).
- Tests: `tests/test_gauss_newton_switch.py`, `tests/test_forcing_power.py`, `tests/test_jax_switch.py`.
- Experiments: `experiments/gn_newton_switch/` (README, `tables.md`, `results.jsonl`).
- User docs: `README.md:333-380`; open items in `dev/TODO.md:320-348`.
- The original study, cited by the plan (`plan:10-17`): the claude.ai artifact "Gauss-Newton to full-Newton
  switching: figures" (V9rpRsqpod3aSuBVNbxhLo; Nick Alger and Claude). It was read for section (c). The TODO
  credits the idea and the study to Nick (`dev/TODO.md:322`).

## (a) The strategy

### Mathematics

Least squares `f = 1/2 ||r(x)||^2`. Full Hessian `H = J^T J + S`, `S = sum_k r_k Hess(r_k)`. Model Hessian
`H_GN` = `gn_hessvec`: "any positive semidefinite model Hessian" whose error is of second order in the step,
usually `J^T J` plus regularization (`plan:22-26`, D1 `plan:313-321`). On a manifold, `hessvec` is the Hessian
of the pullback `s -> f(R_x(s))`, and `gn_hessvec` is its Gauss-Newton part (`plan:256-259`).

At each iteration of the Gauss-Newton phase, for the step tried, `s` (the trust-region step, or `alpha d` for
the line search):

```
change   = f(R_x(s)) - f(x)                              (already known)
m_GN(s)  = <g, s> + 1/2 <H_GN s, s>                       (already known)
m_N(s)   = <g, s> + 1/2 <H s, s>                          (ONE extra product with H)
e_GN     = |change - m_GN(s)|,    e_N = |change - m_N(s)|
valid    = isfinite(e_GN) and isfinite(e_N) and e_GN > 100 * 10 eps max(1, |f(x)|)
hit      = valid and e_N < tau * e_GN                     (tau = 0.25)
hits     = hits + 1 if (hit and accepted) else 0
if hits >= k: switch, effective at the next iteration      (k = 2)
```

Source: `plan:56-87`; code **[no]** `criteria.py:44-69` (`SWITCH_ROUNDING = 100.0`; `eps` is float64's,
`np.finfo(float).eps`, line 67) and `newton_switch.py:126-141` (`SwitchTracker.observe`).

Why it works (`plan:89-101`, `criteria.py:62-64`): `e_GN ~ 1/2 |<S s, s>| = O(||s||^2)`, while
`e_N = O(||s||^3)` along a second-order retraction. Near a minimizer with a nonzero residual `e_N / e_GN`
tends to 0. In the noisy early iterations it is about 1 or larger, so a threshold well below 1, required on
k consecutive steps, rarely fires too early. The rounding gate: when Gauss-Newton's own error is at the
rounding level of `f`, no model can be measured to be better, so there is nothing to gain. An `e_N` at the
rounding level counts as a hit.

*(observation)* An equivalent form: with `c = m_N(s) - m_GN(s) = 1/2 <S s, s>` and
`delta = change - m_GN(s)` (Gauss-Newton's signed error), `hit <=> |delta - c| < tau |delta|`. That is, the
curvature term predicts Gauss-Newton's error to a relative accuracy of tau.

### In words

Start with the Gauss-Newton model. After every step, apply the full Hessian once more and check which
quadratic model predicted the actual change of f better. Once the full model's error has been below a
quarter of Gauss-Newton's for two consecutive accepted steps, use the full Hessian for the rest of the run.

### Parameters, defaults, direction

- `switch_tau = 0.25`, `switch_k = 2`, both keywords (**[no]** `newton_switch.py:33-34`). In code they default
  to `None`, so that passing them with `hessian="full"` raises (`plan:252-254`, `newton_switch.py:78-84`).
  Checks: `0 < tau < 1`; `k` a positive int, not a bool (`newton_switch.py:93-98`).
- `c_round = 100`: a module constant, not a keyword (D4, `plan:337-341`).
- **One-way.** "switching back after the switch (Nick: one-way)" is out of scope (`plan:38`). On firing, the
  tracker sets `switched=True` and `hits=0` (`newton_switch.py:137-139`). Nothing reads the errors afterwards.
- **What resets the count:** a miss, an invalid sample (D3, `plan:333-336`) and a rejected trust-region trial
  (D5, `plan:342-344`). Each is tested (`tests/test_gauss_newton_switch.py:176-182`).
- **Tracking starts at iteration 0**, with no delayed start (D8, `plan:351-355`, kept after phase 6).

### Interaction with step control

- **Trust region.** Before the switch, rho and the radius update use the Gauss-Newton model (`plan:51-54`).
  `m_GN` is the solver's `true_model_value` (`trust_region_core.py:130`), and the product is applied to the
  trial step whether or not it is accepted. **The radius carries over unchanged** at the switch, and from then
  on rho uses the full model (`plan:109-110`). Negative curvature after the switch is handled as in any
  full-Hessian run (Steihaug or Lanczos boundary steps). Radius rule (`trust_region_core.py:91-97`): shrink to
  `0.25 ||s||` when rho < 1/4; double, up to `radius_max`, when rho > 3/4 on a boundary step. Accept when
  rho > `eta = 1e-4`.
- **Line search** (backtracking Armijo). `m_GN = -predicted = alpha <g,d> + 1/2 alpha^2 <H_GN d, d>`
  (`newton_line_search_core.py:157-163`). A fallback direction (steepest descent when the Newton direction is
  not a descent direction) has `gn_model = nan`, so its sample is invalid and resets the count (`plan:205-208`).
  A failed line search ends the run, so it never reaches the detector (`plan:74-75`). After the switch the
  solve truncates at nonpositive curvature (or uses `saddle_free`).
- **Before the switch, a switch run is exactly a Gauss-Newton run**, and after it exactly a full-Hessian run
  resumed at the switch point. Tracking never changes the trajectory (`plan:114-117`), and this is tested bit
  for bit (section c).
- **Mechanism** (*observation*: this mirrors T3Toolbox's compile-budget concern). There are two
  `CoreProblem`s that differ only in `model` ("gauss_newton" / "full") and `forcing_power`. Their
  `OuterState` is shared (same x, f, g, point data), so the switch costs nothing at the switch point
  (`plan:151-165`). The post-switch pieces are built and compiled lazily, only if the switch fires
  (`newton_driver.py:137-171`). The driver picks the phase every iteration
  (`trust_region.py:367`, `pieces, finish_piece = phases(tracker.tracking)`).

### Interaction with the CG forcing sequence

The forcing term is `eta = min(forcing_max, (||g|| / gnorm0)^p)`, with `forcing_max = 0.5` and
`gnorm0 = ||g(x0)||` by default (shared by both phases). The relative tolerance is
`eta * ||g||_{M^{-1}}` (`criteria.py:15-26`, `newton_core.py:132-140`). Powers by phase
(`newton_switch.py:59-73`; D2 `plan:322-332`; D10 `plan:361-365`):

| `pre_switch_forcing_power` | `forcing_power` | before the switch | after the switch |
|---|---|---|---|
| None | None | 0.5 | 0.5 |
| p | None | p | 0.5 |
| None | q | q | q |
| p | q | p | q |

The approved plan had 1.0 after the switch, for the full Hessian's q-quadratic tail. Phase 6 changed it to 0.5
(D10): "0.5 after the switch was cheaper than 1.0 in 46 of 58 runs" (`newton_switch.py:69-70`). A power of
exactly 0.5 keeps the `sqrt` path, so default runs are unchanged bit for bit (`plan:141-146`).

### At a rank change in a continuation

**The arc does not address it.** The detector state belongs to one optimizer call. A new call starts in the
Gauss-Newton phase with zero hits, unless `resume=` passes a `ResumeState(switched, switch_hits)`
(`trust_region.py:319-322`). Resuming a switched state with `hessian="full"` raises
(`newton_driver.py:174-179`). The only continuation in the repo, `examples/matrix_completion.py:440-450`, runs
every stage with `hessian="full"`. The switch appears only in a demo that re-runs the last stage three ways
(GN, switch, full) from the same start (`examples/matrix_completion.py:481-493`).

*(observation)* The resume check compares only f at x0 (`trust_region.py:167-175`). A zero-padded point has
the same f, so a state saved at the old rank would pass the check, but nothing in the arc sanctions that use.

### How the study's detector differs from the library's

The study (artifact, section "Detector"):

- gated on `||g|| > 1e-12 ||g0||` instead of the rounding gate;
- used an absolute forcing term `min(0.5, sqrt(||g||))`, a radius cap of `1e3`, CG capped at `4n` steps, and
  cancellation-free differences of f;
- did not charge the tracking product;
- had an optional "step did not hit the trust-region boundary" condition, dropped as harmful: at diffusion
  beta = 1e-9, 214 of 216 steps hit the boundary and the gated detector never fired.

The library computes the change of f by subtraction, as `f(trial) - f(x)` (`experiments/gn_newton_switch/README.md:129-131`).

## (b) The API in nicks-optimizers

```python
trust_region_newton(fun, grad, hessvec, x0, *, ...,
        forcing_power: Optional[float] = None,
        hessian: Hessian = "full",                      # Literal["full", "switch"]
        gn_hessvec: Optional[Callable[[PointT, VectorT], CovectorT]] = None,
        switch_tau: Optional[float] = None, switch_k: Optional[int] = None,
        pre_switch_forcing_power: Optional[float] = None, ...)   # trust_region.py:199-203
line_search_newton(...)                                  # same five keywords, newton_line_search.py:195-199
```

- With `operators_at`, `gn_hessvec` goes in the returned `PointOperators` (`newton_core.py:47-68`). Giving it
  both ways raises (`newton_switch.py:88-90`).
- `CoreProblem.model: Literal["full", "gauss_newton"]` selects which operator `operators()` returns as
  `hessvec` (`newton_core.py:100-104, 165-170`). Everything downstream sees one `hessvec`.
- `criteria.switch_sample(f, change, gn_model, full_model, tau) -> (valid, hit, e_gn, e_full)` is public
  (exported in `__init__.py:56, 244`).
- Docstring (`trust_region.py:252-264`): "hessian="switch" (with gn_hessvec, see the module docstring) starts
  with the Gauss-Newton model Hessian and switches to hessvec, for good, after switch_k consecutive accepted
  steps (default 2) on which the full model's error |change - m(s)| was below switch_tau (default 0.25) times
  the Gauss-Newton model's; a rejected step or a sample at the level of rounding in f resets the count
  (criteria.switch_sample). Until the switch each iteration costs one more product with hessvec."
- **What it reports.**
  - `IterationRecord` / `LineSearchRecord`: `model` ("gauss_newton" | "full"), `error_gn`, `error_full`,
    `switch_hits` (the count after this iteration) and `switch_fired`. All are `None` without the switch, so
    the records of a `"full"` run compare equal to the old ones (`trust_region_diagnostics.py:61-71, 86-90`).
    These replace the plan's single `switch_hit` (`plan:477-479`).
  - `RunSummary`: `hessian` ("full" | "switch"), `switch_iteration` (the first iteration that used the full
    Hessian), and `gn_hessvec_calls`. `hessvec_calls` includes the tracking products
    (`trust_region_diagnostics.py:162-167, 184-186`; split by `newton_switch.split_hessvec_calls:158-167`).
  - `ResumeState`: `switched`, `switch_hits` (`trust_region_diagnostics.py:106-120`).
  - The display, in switch mode only, adds columns `H` (GN/N) and `eN/eGN` ("-" when invalid), with `*` on
    the row where the switch fired. The summary line reads "switched to the full Hessian at iteration j; a
    Gauss-Newton and b full Hessian-vector products" (`newton_switch.py:170-200`).
  - `history_array` keeps its 8 columns (D6).

## (c) Tests and experiments

**Tests** (`tests/test_gauss_newton_switch.py`):

- `switch_sample` hit/miss/tau, the rounding gate scaling with |f|, and nan (134-150).
- The tracker fires after exactly k hits; a miss, an invalid sample or a rejection resets it; resumed
  counts (167-190).
- **Trajectory identity**: before the switch, the run is bit for bit the GN run with
  `max_iter = switch_iteration` (242-258). After it, the run is the full run resumed at the switch point with
  the same radius and gnorm0 (261-280). Both cover TR and LS × CG, Lanczos, Lanczos with P, flexible with P.
- Resume before, at and after the switch reproduces the run exactly (283-307). A switched state resumed with
  "full" raises (310-316).
- **Work**: one `hessvec` product per GN-phase iteration, rejected trials included, and every product in
  the right counter (319-337).
- **Problems** (`tests/problems.py:273-`):
  - regularized extended Rosenbrock, nonzero residual: fires, converges to GN's f, in no more iterations, and
    the last two GN-phase samples satisfy `e_full < 0.25 e_gn` (340-352);
  - zero residual: converges (355-359);
  - linear least squares, S = 0: **never switches**, with no valid sample (362-369).
- Also `operators_at`, keyword checks, display, and the forcing powers of both phases (372-465).

`tests/test_forcing_power.py` covers the power alone. `tests/test_jax_switch.py:97-139` checks that compiled
runs match numpy, and the trace counts: before the switch `hessvec` is traced only by the tracking finish
piece, and the full-phase pieces compile only when the switch fires.

**Library experiments** (`experiments/gn_newton_switch/README.md`, `tables.md`). The study's three problems:

- Rosenbrock n = 200-400 (7 configurations);
- a 32x32 diffusion inversion, 1024 parameters, beta = 1e-4 to 1e-9, cost in PDE solves (GN product = 2,
  full = 3);
- rank-10 matrix completion 200x200 (16 variants).

Both methods, five variants each (gn, switch with post-power 1.0, switch p=0.5 (today's default), full,
full p=1), rtol 1e-9, max_iter 500. Summary (README:69-79):

| | rosenbrock (14) | diffusion (12) | mc (32) | all (58) |
|---|---|---|---|---|
| switch p=0.5 < gn | 9 | 11 | 31 | 51 |
| switch p=0.5 < switch (post 1.0) | 14 | 9 | 23 | 46 |
| switch p=0.5 < full | 6 | 9 | 12 | 27 |
| gn did not converge (500 its) | 0 | 1 | 11 | 12 |

Observations (README:81-127):

- Against GN, on mc and diffusion the switch saves 52-94%. On diffusion the saving grows as beta shrinks.
- Rosenbrock with the TR fires one or two iterations before convergence. Tracking is a median 16% of the
  products, so the switch costs 17-52% more than GN in 5 of 7 runs.
- Against full Newton from the start: the switch p=0.5 wins in 21 of 29 LS runs but only 6 of 29 TR runs.
  Worst case: mc rank 5 with the LS, 4.4x the full run's products.
- It fired at iteration 2-4 on 6 of 16 mc TR runs (then the run is the full run plus a few tracking
  products), and never on linear LS.

Nick's outcome: power 0.5 after the switch, no delayed tracking, tau stays 0.25. The weaker showing against
full-from-start "is taken as a property of these examples" (D10).

**The study** (artifact), with tracking not charged:

- 29 of 29 cases beat pure GN, with 5-92% fewer products.
- Full Newton from the start cost 23-55% more than switching on ill-posed diffusion (beta 1e-6 to 1e-9).
- tau = 0.5-1 fires early: cheaper on mc, about 49% worse on diffusion at beta = 1e-8. tau = 0.25 "never fired
  in the noisy transient on any problem tried".
- Figures 5-7 plot `e/||s||^2` per model. GN's flattens while Newton's keeps falling.
- In 2D the switch came within 1e-2 to 3e-4 of the solution.

**Library example** (`README.md:367-370`): the last matrix-completion stage takes 43 iterations / 436
products with GN, 15 / 82 with the switch, and 16 / 73 with the full Hessian from the start.

## (d) The arc's open questions, caveats, TODOs

From `dev/TODO.md:334-348`:

> - **A delayed start of the tracking.** On Rosenbrock with the trust region the detector fires one or two
>   iterations before convergence, and the tracking products (a median 16% of the run) buy nothing. Starting
>   the tracking only once Gauss-Newton is visibly linear (its `|1 - rho|`, or the ratio of successive
>   gradient norms, levels off) would remove most of that cost. Deferred by Nick (2026-11-11).
> - **tau** stays 0.25. Larger values fire earlier, which helped matrix completion and hurt the diffusion
>   inversion in the study.
> - A rule with no tracking product (Fletcher-Xu: the relative decrease of f), as an alternative trigger to
>   compare against.
> - The comparison under compilation and on GPUs, and weighted by the real cost of a full product (1.6
>   Gauss-Newton products in matrix completion, 1.5 PDE solves in the diffusion inversion).

Plan risks (`plan:432-455`):

- **Tracking costs more than the study charged.** "30% to 40% more work" when GN solves are 1-5 Krylov steps,
  "more if a full product costs more than a Gauss-Newton one (1.7 times in the study's matrix completion,
  **about 3 to 4 times in T3Toolbox**)".
- **The retraction must be of second order**, or `e_N` stays O(||s||^2) and the switch may never fire. That
  is a safe failure, and the docstring must say so.
- **The full Hessian may be indefinite at the switch.** The study checked definiteness only in 2D.
- **The best tau is problem-dependent.**

Not in scope (`plan:36-42`): switching back, NL2SOL-style secant S, Fletcher-Xu, delayed tracking.

Experiment caveats (README:129-131): one seed per configuration; unweighted counts; f differences by
subtraction.

The study's own follow-ups and caveats:

- Fletcher-Xu (`|F_{k-1} - F_k| / F_{k-1} < 0.2`) as a baseline; an NL2SOL secant S; framing the rule as a
  model-error Dennis-More condition.
- Stress-test early switching near saddles (Absil et al.).
- Negative-curvature handling after the switch is "largely untested".

## (e) Mapping to T3Toolbox

**Where the Hessian is chosen today.**

- Modes: `_HESSIAN_MODES = ('gauss_newton', 'exact')`, checked by `_check_hessian`
  (**[T3]** `t3toolbox/backend/optimizers.py:150-163`).
- `newton_cg(..., hessian='gauss_newton', curvature_damping=None, step_control=None, tr_radius0=None, ...)`
  (`:1323-1326`).
- The mode is fixed for the run: `model_options = dict(hessian='exact', curvature_damping=...) if exact`
  (`:1527`). `lm = problem.local_model(x, frame=start_frame, **model_options)` (`:1539`) is rebuilt only
  after an accepted step (`:1529, 1666-1668`).
- `Problem.local_model(hessian='exact')` runs `_require_exact_capable` (`:166-184`, called at `:419-420`).
  Its exact branch does the one misfit transpose plus `geom.curvature_precompute` into a `CurvatureState`
  (`:435-440`). On the manifold that means damped lift factors, the misfit gradient as a doubled-rank T3 and
  its sweep (`t3toolbox/backend/geometry.py:431-467`).
- `step_control=None` resolves to `'line_search'` for GN and `'trust_region'` for exact (`:1245-1257`).
- Frontend: `t3toolbox/optimizers.py:529-532` (same keywords) passes them straight through (`:733-739`), and
  checks `_check_hessian` early (`:714`). `hessian` also sizes `chunk_size='auto'` via
  `least_squares_problem(..., hessian=...)` (`:159-181, 214, 236-238, 715-716`).

**The two modes of `LocalModel`** (`:203-331`).

- `hvp` takes the GN branch whenever `not self._has_curvature_term` (`:285-298`). CG calls only `lm.hvp`
  (`:1083`).
- `gn_quadratic(p) = ||omega J Pi p||^2 (+ reg)` (`:279-283`) and
  `curvature_quadratic(p) = <Pi p, L^T K_N L Pi p>` (`:323-328`) are separate methods.
- The exact-mode gradient is the GN gradient bit for bit, read from `CurvatureState.misfit_gradient`
  (`:264-273`).
- In exact mode `newton_cg` already computes `pKp = lm.curvature_quadratic(p)` and
  `pHp = gn_quadratic(p) + pKp` (`:1600-1603`).
- *(observation)* So the switch's tracking quantity is `m_N - m_GN = 1/2 alpha^2 pKp`. It costs one
  curvature-block application (lift, `corewise_hessian`, `tangential_curvature`, lift transpose), not a full
  hvp, **plus an exact-mode `CurvatureState` at every GN-phase point**. The per-frame `curvature_precompute`
  is a tracking cost that nicks-optimizers does not have.
- `e_GN` can be computed from today's row fields: `|delta_f - (alpha*slope + 1/2 alpha^2 pHp)|` (LS) or
  `|delta_f - (slope + 1/2 pHp)|` (TR). Then `e_N = |delta_f - predicted - 1/2 alpha^2 pKp|`.

**Step control** (`newton_cg`, `:1608-1646`).

- LS:
  - Armijo with 40 halvings and `c_armijo = 1e-4`;
  - `predicted = alpha*slope + 0.5*alpha^2*pHp` and `rho = delta_f / predicted` (`:1621-1622`);
  - `on_line_search_failure='stop'` rejects and terminates, while `'accept'` takes the last trial and
    continues (`:1623`). The second case does not exist in nicks-optimizers.
- TR, with constants as module constants (`:1234-1242`):
  - accept when `rho > 0.1` (nicks-optimizers: 1e-4);
  - shrink to `0.25 ||p||` when `rho < 0.25` or rho is non-finite;
  - double, up to `1e4 Delta0`, when `rho > 0.75` on a boundary step;
  - `Delta0` = the GN Cauchy length `||g||^3 / <g, H_GN g>` whatever the Hessian (`:1286-1296`);
  - `objective_floor` and `trust_region_floor` are checked only after a rejection (`:1639-1646`).
- Non-descent fallback (`:1593-1599`): `p <- -g` (scaled into the region under the TR), and `pHp` is then
  computed on the fallback `p`. So unlike nicks-optimizers, the GN model value is defined there.

**Forcing sequence.**

- `eta = min(0.5, (gnorm / cg_ref) ** cg_forcing_power)` and `cg_tol = eta * gnorm` (`:1580-1581`).
  `forcing_max` is hard-coded at 0.5.
- `cg_forcing_power: float = 0.5`, a float default, not `None` (`:1332`).
- `cg_ref` is resolved once at iteration 0 (`:1548-1567`): by default `'zero'` on a manifold (the zero tensor
  of x0's ranks) and `'warm_start'` on corewise; also `'cone'` and the overrides `g0norm_newton` /
  `g0norm_cg`. nicks-optimizers uses `||g(x0)||`.
- The docstring says 1.0 gives the exact tail q-quadratically (`:1366-1368, 1431-1435`). T3Polynomial's
  documented schedule uses `{'hessian': 'exact', 'cg_forcing_power': 1.0}` past rank 10
  (`T3Polynomial/t3polynomial/fitting.py:526-530`). nicks' D10 chose 0.5 after the switch.

**`NewtonInfo`** (`:1147-1218`).

- Fields: `iteration, objective, gnorm, g0norm, converged, x_cores, lm, misfit, regularization, forcing_eta,
  cg_tol, cg_iters, cg_maxiter, cg_resid, cg_converged, cg_truncated, ls_steps, ls_failed, alpha, slope, pHp,
  delta_f, rho, step_rel, wall_time, g0_reference, cg_reference, step_control, stop_reason, hessian,
  tr_radius, tr_accepted, cg_boundary, pKp, lift_sigma_min, lift_eps`.
- `hessian` is one of the documented **run-level** strings set on every row (`:1157-1168`).
- `pKp` / `lift_*` are "set only with `hessian='exact'`" (`:1179-1181`).
- `stats = {'losses', 'newton_steps', 'newton_accepted', 'stop_reason', 'history', +extra}` (`:1452-1458,
  1669-1670`). There is no Hessian-product count; `cg_iters` is per row.
- The display computes `kappa = pKp / (pHp - pKp)` (`t3toolbox/backend/optimizer_display.py:150-160`), which
  assumes `pHp` includes `pKp`. On a GN-phase row carrying a tracking `pKp`, that formula would be wrong.
- There is no resume state in `newton_cg`, so nicks' `ResumeState` fields have no counterpart.

**What a switch would read.** Per row: `f`, `delta_f`, `slope`, `alpha`, `pHp` (GN), the tracking `pKp`, and
acceptance (`tr_accepted`; LS `not ls_failed`); the rounding scale is discussed below. It would keep `hits` and
`switched` across rows.

**What it would report** (by analogy):

- per row, the model used, `e_GN`, `e_N` (or their ratio), the hit count and a fired flag;
- in stats, the switch iteration, the CG iterations split by Hessian, and the number of tracking
  applications.

**Further mapping observations** (*observation*):

- **The switch always lands on a fresh linearization.** Hits count only accepted steps, so the switch fires
  right after an accepted step, when `lm` is `None` and is rebuilt anyway (`:1666-1668`). The next model can
  simply be built in exact mode, with no model upgrade in the middle of a point.
- **One exact-mode build per point is possible.** In the GN phase a GN view `dataclasses.replace(lm,
  curvature=None)` shares the sweep and residual, so CG runs the GN `hvp` while the exact model supplies
  `pKp` and the gradient. Under `use_jit` the curvature's presence is pytree structure (`:1699-1709`), so the
  exact phase adds one compile of `_cg_core` when the switch fires. That fits the compile-budget rule ("a
  small fixed number inside a loop tolerable").
- **The rounding gate.** nicks-optimizers' gate is float64-only (`criteria.py:67`) and absolute when |f| < 1.
  T3Toolbox already has a dtype-aware resolution scale: `_resolution_eps(lm.residual) * (||omega y|| *
  ||omega r|| + |f|)` (`:1273-1280, 1641-1643`). That is the cancellation error of `delta_f` for
  `f = 1/2||r||^2`, and it matters under float32, the default with `use_jit` on jax.
- **The second-order caveat applies.**
  - The T3-SVD retraction is second order "on a minimal, numerically full-rank frame"
    (`backend/optimizers.py:385-387`).
  - The exact Hessian uses a damped lift, `eps = curvature_damping * ||X||_HS` (`geometry.py:438-441`).
  - "A zero-padded restart's new `s_g` levels filter to exactly 0" (`geometry.py:455-456`).
  - So after a rank increase, or near small singular values, `e_N` may keep an O(||s||^2) floor, and the
    detector would hold off: a safe failure.
- **Where S comes from.** On the manifold the ambient objective is exactly quadratic, so `S` is purely the
  manifold (Weingarten) curvature term (`docs/fitting_and_optimization.md:322-336`). On corewise geometries it
  is `K_{A^T omega^2 r}`. Both scale with the residual, as the detector's argument requires.
- **Related open item:** I-36, the Newton schedule, GN vs exact as the continuation default
  (`docs/contributor/deferred_and_rejected.md:204-`; inputs in `docs/contributor/verification_records.md:116-131`:
  GN's rate `rho(H_GN^-1 K) ~ 0.7`, and exact cuts outer steps 3.5-5x). Nick leans toward exact with the warm
  start.

## (f) Questions for the maintainer

1. **Keyword shape and name.** Should it be a third mode, `hessian='switch'` (nicks' name), or another value or
   keyword? The default stays `'gauss_newton'`, so existing runs are unchanged bit for bit.
2. **Detector keywords.** Should `switch_tau=0.25` and `switch_k=2` be exposed with nicks' `None`-defaults that
   raise outside switch mode? Should `c_round` stay a module constant?
3. **Forcing powers.** `cg_forcing_power` defaults to the float 0.5 today, not `None`, so nicks' resolution
   table needs adapting.
   - What should the pre-switch keyword be called (e.g. `pre_switch_cg_forcing_power`)?
   - What should the post-switch default be: 0.5 (nicks D10, 46 of 58 runs) or 1.0 (T3Polynomial's exact
     schedule, and this library's "q-quadratic tail" guidance)?
4. **Default `step_control` in switch mode.**
   - The TR (exact's default) handles indefiniteness after the switch; the LS (GN's default) won 21 of 29
     against full-from-start in nicks.
   - Both keep one step control for the whole run, as nicks does, so the TR radius carries over unchanged.
5. **The rounding gate.** Use nicks' `100 * 10 eps_f64 * max(1, |f|)`, or T3Toolbox's dtype-aware
   `eps_dtype * (||omega y|| ||omega r|| + |f|)` times a factor, and which factor?
6. **Cases nicks never meets.** Is a non-descent fallback to `-g` a valid sample (its GN model is defined
   here) or a reset? Does an `on_line_search_failure='accept'` step count as rejected (reset)? Is a TR
   rejection a reset, as in nicks' D5?
7. **Tracking cost and start.** In T3Toolbox tracking costs an exact-mode `curvature_precompute` per point
   plus one `curvature_quadratic`, and a full product is about 3-4x a GN one (nicks `plan:438-439`).
   - Should tracking start at iteration 0 (nicks D8), or be delayed until GN is visibly linear (`|1 - rho|`
     levelling off)? That option is deferred in nicks but may matter more here.
   - Is per-point wall time the acceptance metric? That is the "measure at scale" concern.
8. **Implementation.**
   - Build one exact-mode model per point and give CG a GN view, or build two models?
   - Is one extra jit compile when the switch fires acceptable?
   - Should `_require_exact_capable` run up front, so that switch mode fails at the call?
9. **Reporting.**
   - Should `NewtonInfo.hessian` stay run-level (`'switch'`) with a new per-row model field, or become
     per-row (`'gauss_newton'` / `'exact'`) with a new run-level field?
   - What should the new fields be called (`error_gn`, `error_exact`, `switch_hits`, `switch_fired`), and
     what stats key (`switch_iteration`)?
   - Should GN-phase rows carry the tracking `pKp` / `lift_*`? If so, the display's kappa must be fixed.
   - Should the display get nicks-style `H` and `eN/eGN` columns (compare `dev/newton_display_plan.md`)?
10. **Rank continuation.** Should each `newton_cg` call (each level) start in the GN phase, as in nicks, or
    should there be a way to start switched or carry the state? How does this interact with I-36 and
    T3Polynomial's per-level `newton_kwargs` schedule? Is the expected non-firing at padded or
    near-rank-deficient starts acceptable as the safe failure?
11. **Scope.**
    - Do all geometries × layouts (ragged/uniform, manifold/corewise, shared) get the switch, as the exact
      Hessian does?
    - Should `least_squares_problem(hessian='switch')` size `chunk_size='auto'` as `'exact'`?
    - `curvature_damping` would be allowed in switch mode.
12. **Evidence before the tag.** Which battery should the switch be measured on (e.g. the Darcy laptop
    config, few points × many directions)? Against which baselines (GN, exact, exact with power 1.0)? Should
    an alternative trigger with no extra product (Fletcher-Xu) be compared?
