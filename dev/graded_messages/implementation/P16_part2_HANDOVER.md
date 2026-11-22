# P16, part 2 (the backend model and optimizers): handover

Implementer (the third of P16: a fresh one for part 2), 2026-11-21, on `p16-wip` at `5be5dc28`. Built to
`P16_DESIGN_NOTE.md`, `P16_DESIGN_REVIEW.md` and Claude's rulings in `LOG.md` (D1 to D8; the alignment projector; the
review's test additions). Nothing is committed. **Round 1 (review `P16_part2_REVIEW_r1.md`, response
`P16_part2_RESPONSE_r1.md`): D9 was ruled (d) and is built (section 4); both D9 tests are green.**

Required reading done first (`TuckerTensorTrain`): `probe` (to `probing.t3_probe`, `sampling_variables`, `_run`,
`sampling_steps.mu_step`, `contractions.jet_contract`), `t3svd` (to `t3_svd.t3svd`, `t3_down_svd_tt_core`), `__add__`
(to `t3_linalg.t3_add`) and `inner` (to `t3_linalg.t3_inner_product`).

## 1. What changed

**`t3toolbox/backend/optimizers.py`** (the core of the part)
- **`LocalModel`** has the fields of plan 2.2: `geom`, `oracle`, `batch`, `frame`, `centers`, `held`, `f`, then
  `regularizer`, `geom_aux`, `curvature`. `sweep`, `residual`, `sample`, `kind` and `n_w` are gone. Every method is one
  `oracle.query` at the model's frame on its batch, with `held=self.held`, the centers' `Vc`, and a literal `want`:
  - `gradient`: `('g',)`, projected (exact: the build's `g`, projected once, read from `CurvatureState`);
  - `gn_quadratic`: `('Jp_norm_sq',)` at `variations=Pi p`;
  - `hvp`: Gauss-Newton `('JtJp',)`; exact `('JtJp', 'Kp')` in ONE query, `variations=Pi p`,
    `core_directions=lift(Pi p)`, then `Pi [JtJp + L^T (Kp - K_grad)]`;
  - `curvature_hvp` / `curvature_quadratic`: `('Kp',)` at the lifted direction;
  - `jacobian(p)`: `('Jp',)`, checked: needs `'Jp'` in `optional_outputs`;
  - `residual` (a property): `('r',)`, checked: needs `'r'` (free when `r` is held);
  - `misfit` is the build's `f`; `objective = f + rho(base_point)`.
- **`local_model(geom, oracle, batch, x_cores, regularizer=None, hessian='gauss_newton', curvature_damping=None,
  frame=None, centers=None)`** (D3): the frame and centers from `geom.frame`, else the given frame (its structure
  checked by `require_frame_structure`, as before) with the given centers; **centers without a frame raise**. One
  query `('f',)` (exact `('f', 'g')`). Its docstring says a wrong frame or wrong centers are a silent wrong answer and
  names the residual checkers. Exact mode checks `'Kp'` in `oracle.optional_outputs` and the four geometry members
  before any work.
- **`point_objective(geom, oracle, batch, x_cores, regularizer=None)`**: a query `('f',)` at `geom.point_data(x_cores)`
  plus `rho(x_cores)`; the trial points (policy a) and the full-batch checks of MC-SGD and Adam.
- **The optimizers** take `(geom, oracle, batch, x0, ...)`, `regularizer=None` after the positional arguments, today's
  keywords in today's order: `newton_cg(geom, oracle, batch, x0, regularizer=None, max_newton=...)`,
  `gradient_descent(geom, oracle, batch, x0, regularizer=None, ...)`, `mc_sgd(geom, oracle, batch, x0, rng,
  batch_size, regularizer=None, draw=None, ...)`, `adam(...)` likewise. `flat_draw(oracle, batch, batch_size)` returns
  `draw(rng) -> batch` through `oracle.take` (the same `rng.choice(n, size=min(b, n), replace=False)` call as before).
  MC-SGD and Adam without a `draw` on an oracle without `take`/`n_measurements` raise at entry (`_require_draw`).
- **The minibatch rule**: each stochastic step builds a model afresh on the drawn batch; the oracle scales the misfit
  (`minibatch_scale`), the regularizer is **not** scaled (O-21, as corrected 2026-11-21). The Cauchy step is
  invariant under the factor (verified: MC-SGD iterates match the pre-P16 library to 1e-13); Adam differs only through
  its `eps` (matched to 1e-10).
- **The phases** are module-level functions of pytrees and statics: build (`local_model` / `_local_model_phases`),
  gradient (`_model_gradient`), CG (`_cg_core`, unchanged), trial (`point_objective`), the stochastic steps
  `_mc_sgd_step(geom, oracle, regularizer, cores, batch_B)` and `_adam_step(...)`, jitted with the three statics as
  `static_argnums`. `_prepare_jit_inputs(use_jit, x0, batch)` moves every array of the batch (any pytree) to jax;
  `_jit_engaged` reads the batch's array leaves through jax's own flattening (`_array_leaves`).
- **The references and the padded start through the oracle**: the zero reference is a query `('g',)` at the
  zero-point frame (its base point is exactly 0); `padded_start(geom, oracle, batch, x0, x_old, <options>,
  use_jit=False) -> (frame, centers, movements)` (D2); the alignment projector is `_alignment_projector(geom, oracle)`,
  memoized, which at each pass queries `('g',)` at that pass's frame and projects with `geom.with_frame(frame)` (the
  ruling); the batch is the loop's pytree hint; nothing is held across passes. `newton_cg(padded_from=)` passes the
  padded centers to the iteration-0 build internally; `start_frame=` has none. The cone likewise.
- **D8, the timing laps**: `PhaseSeconds` has `oracle` in place of `sweep` and `residual` (fields `previous_callback,
  frame, oracle, companion, curvature, gradient, reference, cg, forms, trial_retraction, trial_objective, tracking,
  remainder`); the build laps `frame`, `oracle`, `companion`, then in exact mode `gradient` (the projection of the
  build's `g`) and `curvature`. The `Problem`-subclass override path is gone with `Problem` (`_STOCK_LOCAL_MODEL`).
- **D8, the floor**: `_objective_resolution(lm, regularization)` = the oracle's `f_resolution` (queried only on a
  rejection or a tracked switch row, with `held`) `+ eps |rho|`; without `f_resolution`, `eps (|f| + |rho|)`; `eps` read
  off the dtype of the model's `f` (`_resolution_eps`, which still accepts any tree). This reproduces the 2026.5.0 floor
  (`eps (|wy| |wr| + |misfit + rho|)`).
- **Removed**: `Problem`, `least_squares_problem`, `_minibatch_step_problem`, `_STOCK_LOCAL_MODEL`, the `Problem`
  pytree registration, the import of `_ScaledRegularizer`. `__all__` gains `local_model`, `point_objective`.
- The `Geometry` protocol text gains `point_data`; the `Oracle` protocol text says what the optimizers rely on (`held`
  for one frame and one batch; the optional outputs each feature needs; the draw members).

**`t3toolbox/backend/geometry.py`**
- **D1**: `point_data(x_cores)` on the four geometries (ragged: the cores; uniform: `(B_sc, G_sc, shape, masks)` with
  the geometry's point masks).
- **D5**: `UniformManifoldGeometryOps.from_point` raises a `ValueError` for a non-minimal point (structural, from the
  host masks, the shared notion with `sharing`; the message names `uniform_minimal` and `from_frame`). `from_frame` and
  the corewise geometry do not check (no retraction desync there).

**`t3toolbox/backend/regularization.py`**: `_ScaledRegularizer` removed; docstrings name `point_objective` and the
unscaled rule.

**`t3toolbox/backend/uniform_fitting.py`**: `pack_sample(kind, order, sample, N)` and `pack_data(kind, order, y, N)`
(the plan's forms; doctest); `pack_batch` calls them; `uniform_least_squares_problem` and its doctest removed (it was
annotated `-> bopt.Problem`); the `optimizers` import dropped; the module docstring rewritten. The old uniform kinds
stay (P20).

**`t3toolbox/backend/optimizer_display.py`** (the lap rename, D8): `_PHASE_LABELS` and `_MODEL_PHASES` name `oracle`;
the `PhaseSeconds` doctest re-run and its output pasted. Nothing else of the display (part 3).

**`t3toolbox/optimizers.py`** (one line, outside the part): the field annotation `problem: bopt.Problem` of
`LeastSquaresSetup` became `typ.Any`; it is evaluated at import, so without this `import t3toolbox` fails. Part 4 deletes
the class.

**`tests/_fits.py`** (new, test-only): `Fit(geom, oracle, batch, regularizer)` forwarding each call to the backend
function of the same name by keyword, and `fit`, `derivative_fit` (derivative data along lines through
`prepare_derivative_data`: the old derivative kinds' objective exactly; entries at the one-hot jet) and `uniform`
(`from_point` + `pack_batch`). It keeps the ported tests close to their originals without the old kinds.

## 2. Tests

**`tests/backend/test_optimizers.py`** (rewritten on the seam). New `TestOracleSeam`:
- `test_the_heads`: the heads of the four optimizers, `local_model`, `point_objective`, `padded_start`, `flat_draw`;
  `Problem` and friends gone.
- `test_want_is_a_literal_at_every_query` (AST): every `.query(` in `backend/optimizers.py` has a literal tuple of
  strings as `want` (>= 10 sites).
- **`held` never crosses a draw, a trial or a build**: `test_mc_sgd_builds_each_draw_afresh` and
  `test_adam_builds_each_draw_afresh` compare the iterates with reference loops built from DIRECT oracle queries on each
  drawn batch with no held and no `LocalModel` (MC-SGD regularized, so the unscaled rule is in the reference);
  `test_minibatched_newton_builds_each_point_afresh` (newton_cg per drawn batch: every row's objective is the DENSE
  objective of that call's batch, and accepted `f + delta_f` is the next row's); `test_a_trial_is_scored_with_nothing_held`
  (every trial's `f` is the dense objective, trust region and line search).
- **The review's structural test** `test_a_query_given_the_models_held_runs_no_plain_sweep`: the engine's `_run`
  recorded during `gradient`, `gn_quadratic` and `hvp` of a Gauss-Newton model at a frame with centers runs none of
  `xi, mu, nu, eta, z, r, nu_L, eta_D`; probe, apply, entries x ragged, uniform x manifold, corewise point. Without
  centers the probe kind holds `('xi', 'mu', 'r')` and its hvp recomputes exactly `{nu, eta}` (asserted).
- **The review's four frames** `test_the_models_f_is_point_objective_at_four_frames`: the model's `f` equals
  `point_objective` and the dense objective at a corewise point, `frame(x)` with centers, a given frame without
  centers, and a padded start with its zero-padded centers.
- **D1** `test_point_data_reads_garbage_padding_as_nothing`: uniform `point_data` is the bare pair with the geometry's
  shape and point masks; `point_objective` at supercores with garbage (1e3) in every padded slot equals the dense
  objective, both uniform geometries, every kind; ragged `point_data` is the cores.
- **D8** `test_the_floor_reads_eps_off_f`: float64 and float32 data give `f` and `eps` of that dtype; the floor equals
  `eps (|y| |r| + |f + rho|)` by hand; an oracle without `f_resolution` gets `eps (|f| + |rho|)`.
- **D9 (round 1)** `test_the_acceptance_compares_like_with_like`: row 0's objective is `point_objective(x0)` bitwise;
  every row's misfit and regularization are `point_objective`'s terms at that row's point, bitwise; on an accepted row
  the next objective is `objective + delta_f` to its rounding, on a rejected one the same bitwise; trust region (a
  scripted rejection), line search, the `'accept'` after exhaustion, regularized, ragged and uniform, both geometries,
  and `gradient_descent` (`losses[k]` is `point_objective` at its k-th point). `check_bookkeeping` checks the same chain
  on every trust-region test.
- `test_optional_outputs_are_checked_where_needed`: no draw and no `take` raises before any work (mc_sgd, adam); with a
  draw it runs; `jacobian` needs `Jp`, `residual` needs `r`, exact needs `Kp`.
- Rewritten against independent constructions: `test_model_matches_dense_ground_truth` (replaces the comparison with
  the frontend model, which is broken until part 4): objective, `<g, p>`, `gn_quadratic`, `<p, hvp(p)>`, `jacobian`,
  `residual` against the dense samplers and the tangent's dense direction (corewise: an exact polynomial difference
  quotient), every geometry x kind; `test_derivative_data` (the derivative objective on the derivatives, an FD of the
  corewise gradient, the flat draw's scale, recovery); `test_stochastic_regularizer_is_not_scaled` (the drawn model's
  objective, gradient and gn_quadratic are `s` times the unscaled batch's plus the regularizer's own terms).
- Ported, the same assertions: Steihaug CG, the trust region and its floors (the trial-objective offset is now a user
  oracle, `_OffsetOracle`, the extension surface), the timing records (the phase attribution now traces the oracle's
  query; the waits test checks the build's `(frame, centers)`, `(f, held)`, companion, misfit gradient, curvature),
  the switch (the "one transpose per point" spy counts queries wanting `g` or `JtJp`; `_NoKpOracle` in place of the
  kind without a two-hole; `_switch_case` built on the backend, uniform starts reduced by `uniform_minimal`).
- Removed: the `Problem`-override timing tests (no `Problem`), `test_chunk_size_does_not_change_the_gradient` (no
  chunking), the two frontend-model stacked tests (part 4).

**`tests/backend/test_zero_reference.py`**: the resolution tests on backend fits (ragged, uniform, shared, corewise);
the hand-computed reference is now a DENSE projection (`tv_project_dense_onto_tangent_space` of `A^T(0 - y)`); the
custom geometry has ten members (`point_data`); the uniform template uses `from_frame` (D5: a d = 1 template at Tucker
rank 3 is non-minimal).

**`tests/backend/test_padded_start.py`**: `fit_case` builds backend fits; `padded_start` returns three; the projector
patch (`_no_projection`) raises when a pass projects; `padded_from` against `start_frame=` is now equal to 1e-12, not
bitwise (D3: one reads `z` through the padded centers, the other through the base point); `_KwargsMember` (no centers)
likewise against the stock member.

**`tests/backend/test_uniform_fitting.py`**: the uniform model and optimizers against the ragged ones through
`F.uniform`; D5's guard (`test_nonminimal_x0_rejected`, `test_shared_minimality_is_the_shared_notion` on a structure
that is shared-minimal and not minimal per mode, `test_a_local_model_at_a_nonminimal_point` through `from_frame`, the
review's requirement); `TestPackSampleAndData` (the new forms slot by slot); `TestSlackPaddedStart` from a minimal slack
start (the old one was non-minimal and silently reduced by the frontend). The old uniform kinds' own tests are kept.

**`tests/backend/test_curvature.py`** (the model parts): fits through `F.derivative_fit` (data from `dd.forward`, the
data of one point broadcast over `C`: the oracle's `y` has no `C`); `K_{A^T w^2 r}` read as the oracle's `Kp` (`_K_r`);
the dense Lagrangian tests measure the Taylor coefficients with the Taylor weight; the block-sum oracle is now a user
oracle (`_BlockSumApplyOracle`); the "kind operation" tests became `test_the_curvature_block_is_the_oracles_Kp_at_the_
weighted_residual` (Kp against the block-sum oracle or the merged corewise Hessian at `w^2 r`; the weight is not
dropped; an exact hvp is ONE query `('JtJp', 'Kp')` with the model's frame and held); `TestRequiredSixthOperation` is
now about `Kp` (an oracle without it works for Gauss-Newton and is refused for exact and switch; every layout builds an
exact model). The tests of the old kinds' sixth-operation marker went (no caller left).

**`tests/backend/test_uniform_curvature.py`** (the model parts): models at a given frame through `local_model(frame=)`
(`_uniform_exact_model`, `_ragged_exact_model`), random data in place of a random residual, the packed batch with
garbage beyond each `N_i` (`_uniform_batch`); the quadratic comparisons normalized by their terms
(`_quadratic_term_scale`, as `_hvp_term_scale` does for the hvp): at 'd4 varying' `|q|` is 0.16 of its terms and the
error is 2.2e-14 of them (1.4e-13 of `|q|`); the old kinds' own tests keep the pre-P16 packing (`_old_pack_sample`).
The frontend-equals-backend assertions of `TestUniformExactNewton` and the shared restart moved to part 4 (comments
mark them).

**`tests/test_dispatch.py`** (the backend rows): the exact models in the kind two-hole jit tests, the uniform whole-step
jit, the model cache keys, the uniform exact ops (the old uniform kind built locally for its own two-hole rows), the CG
staleness rows, the float32 floor, the zero-point frame, the padded-start seam (three elements, centers jax on jax), the
alignment timing, the Newton timing rows and the switch's float32 gate. Mixed rows that call the frontend stay for part 4.

## 3. Runs (current env, `-n 2`, one run at a time)

Python 3.11, numpy 2.4.6, jax 0.10.2 (`t3toolbox`); `PYTHONPATH=$PWD ... -m pytest <files> -q -n 2`.

| Run | Result |
| --- | --- |
| `tests/backend/test_optimizers.py` | round 1: 76 passed, 1 skipped (slow tier), 354 subtests (round 0: 74 passed, 1 failed, the D9 test) |
| `tests/backend/test_zero_reference.py` | 17 passed, 27 subtests |
| `tests/backend/test_padded_start.py` | 37 passed, 1380 subtests |
| `tests/backend/test_uniform_fitting.py` | 30 passed, 190 subtests |
| `tests/backend/test_curvature.py` | 96 passed, 1407 subtests |
| `tests/backend/test_uniform_curvature.py` | 57 passed, 2 skipped (slow tier), 383 subtests |
| `tests/backend/test_geometry.py`, `tests/test_import_graph.py`, `tests/test_least_squares_oracle.py` | 55 passed |
| `tests/test_dispatch.py`, the backend rows (`-k`: the kind two-holes, the uniform whole step, the model cache keys, the uniform exact ops, CG staleness, the float32 floor, the zero-point frame, `TestPaddedStartSeam`, `TestAlignmentTiming`, `TestNewtonTiming`, `TestSwitchNewton`, the oracle row) | round 1: 21 passed, 1 failed (the float32 floor: its hand scale read the reported misfit, now the point route's; fixed to read the model's, see the response), then that test alone: passed (round 0: 22 passed, 1 failed, D9) |
| Round 1, the other files above | unchanged: 17, 37, 30, 96, 57 (+2 skipped), 55 passed |
| Module doctests (`python -m doctest`): `backend/optimizers.py`, `regularization.py`, `uniform_fitting.py`, `geometry.py`, `optimizer_display.py` | pass (the display's `PhaseSeconds` line re-run and pasted) |
| **Full suite** (`tests/`, `-n 2`, once, to measure the red list) | 244 failed (subtests and tests), 1796 passed, 5 skipped, 1 xfailed, 90,658 subtests, 31 min; **lowest MemAvailable 4.27 GB**; 106 red ids, one of them (`test_geometry`'s forced padding, made red by D5) fixed after it (55 passed above) |

The slow tier was not run (it is the gate's, part 4). The floor environment was not run.

**Mutation pass** (a scratch copy of `t3toolbox/` and `tests/`, deleted; `scratchpad/implementer3/mutate.py`), 20
mutants (M16 to M20 added for D9 in round 1, the whole pass re-run then), all killed:

| Mutant | Killed by |
| --- | --- |
| M1 a build reuses the previous build's `held` (across draws and points) | `TestOracleSeam` (12 tests) |
| M2 the trust-region trial scored with the model's `held` | `test_a_trial_is_scored_with_nothing_held` |
| M3 the stochastic step scales the regularizer by `b/n` | `test_mc_sgd_builds_each_draw_afresh` |
| M4 uniform `point_data` with all-true masks | `test_point_data_reads_garbage_padding_as_nothing` |
| M5 `local_model` drops the centers | the structural test, the four frames |
| M6 the floor's `eps` always float64 | `test_the_floor_reads_eps_off_f` |
| M7 the floor without `eps |rho|` | the same |
| M8 exact mode does not check `Kp` | `TestSwitchNewton.test_options`, `TestRequiredSixthOperation` |
| M9 the alignment projector with the level geometry, not `with_frame` | `test_padded_start.py` |
| M10 no guard in `UniformManifoldGeometryOps.from_point` | `TestUniformMinimalRank` |
| M11 centers without a frame accepted | `test_centers_need_their_frame` |
| M12 exact `hvp`: `variations` and `core_directions` swapped | `TestExactLocalModel` |
| M13 Gauss-Newton `hvp` not projected | **survived the first pass** (the dense test read `hvp` only through `<p, hvp(p)>`, blind to the projection); `test_model_matches_dense_ground_truth` now also asserts `Pi hvp(p) = hvp(p)`, and kills it |
| M14 `point_objective` drops the regularizer | `TestOracleSeam`, `TestProblemContracts` |
| M15 the zero reference off by 1e-6 | `test_newton_cg_g0norm_and_forcing_overrides` (now a dense projection), `test_zero_reference.py` |
| M16 D9 mixed routes (the reference re-read off the model, frame route) | `test_the_acceptance_compares_like_with_like`, `test_gradient_descent_line_search_exhaustion` |
| M17 D9 the accepted trial's value not carried | `test_the_acceptance_compares_like_with_like` |
| M18 D9 a rejected trial's value carried | the same |
| M19 D9 `gradient_descent` re-reads its reference off the model | the same, and the exhaustion test (its first form, `f = None` after a step, was an equivalent mutant: it re-reads `point_objective(x)`, the same value; replaced) |
| M20 the shrink a quarter of the radius (the offset oracle's tests still bite) | `test_shrink_is_a_quarter_of_the_step_not_of_the_radius` |

## 4. Departures, choices, and the decision needed

**D9 (ruled (d) in round 1, built).** The problem: the model's `f` (the frame route, row 4.3) and a trial's
`point_objective` (the point route, policy a) are equal to rounding but rounded differently, and at the floor the
offset is as large as the changes tested (the line-search exhaustion never fired; the float32 exact run accepted a
spurious step). **The fix, as ruled:** `newton_cg` and `gradient_descent` carry the point-route objective of exactly the
current `x`: one `point_objective` at each start (`x0`, `padded_from`, `start_frame`, every call; `newton_cg` times it in
the `reference` lap), then the accepted trial's value. A rejected trust-region trial keeps `x` and the value; the
`'accept'` after an exhausted search carries the last trial's; the `-g` fallback is a trial; the switch's exact rebuild is
at the same `x`. The history's `objective`, `misfit`, `regularization` and `losses` are that value and its two terms
(`_point_objective_parts`, private; `point_objective` is their sum; `_scored` the host floats), so the next row's
objective is the accepted trial's value itself and `objective + delta_f` equals it to `delta_f`'s rounding. The model's
frame-route `f` stays `LocalModel.misfit` (the floor's dtype, the display). The invariant is written at the carry: any
future step that changes `x` between iterations must recompute it. Cost: one forward per run.

Choices (mine, within the rulings):
- **`require_frame_structure` kept for `local_model(frame=)`** (structural, as before; the plan's "checks nothing" is
  about numerics, O-6).
- **The exact and switch checks run at `newton_cg`'s entry** for both modes (the old code checked `exact` only at the
  first build, after a padded start's alignment).
- **The floor's `f_resolution` is queried, not cached** (once per rejection or tracked row; it re-reads `y`'s block
  sums, O(data)); 2026.5.0 cached `|w y|` once per run.
- **`LocalModel.residual` is a property** (it was a field); `jacobian(p)` projects `p` first, as before.
- **Stacked data**: the tests now use one `y` for every element of a stacked point (the oracle's `y` broadcasts over
  `C`); the old kinds took per-element data. Per-element residuals still differ, through the points.
- **D5** is in `from_point` only (the corewise geometry cannot desync: its retraction is additive).

## 5. Side by side against the pre-P16 library

`scratchpad/implementer3/side_by_side.py` runs 36 cases in the old library (`git archive 0acf7868`) and in this tree
(separate processes, the same data and seeds; derivative data through `prepare_derivative_data`), and compares the
stop reason, the step counts, the losses and the final dense tensor:

Its copy is kept beside the benchmarks, `dev/graded_messages/implementation/bench/p16_part2_side_by_side.py` (run it
with `old <pkl>` on the archive's path, `new <pkl>` on this tree's, then `compare`). The cases: Gauss-Newton, exact
and switch (`'countable'` and `'all'`); line search and trust region; the zero, cone and warm references;
`padded_from`; regularized; a per-mode weight; derivative data at orders 1 and 2; gradient descent; MC-SGD with the
flat draw and a custom draw step for step (the same `rng.choice` calls), regularized; Adam; ragged and uniform.
Columns: SAME/DIFF (stop reason, step counts, switch row, history length), steps old/new, stop old/new, the worst
relative difference of the losses, of the final dense tensor, of the cone / start movements, and any history field
beyond rtol 1e-6:

```
gn_ls_probe_manifold_zero          SAME  steps 12/12 stop max_newton/max_newton losses 8.4e-15 x 8.2e-16 fields:rho
gn_ls_apply_manifold_warm          SAME  steps 12/12 stop max_newton/max_newton losses 1.0e-12 x 4.4e-09 
gn_ls_entries_manifold             SAME  steps 12/12 stop max_newton/max_newton losses 8.9e-11 x 1.2e-11 
gn_ls_probe_corewise               SAME  steps 10/10 stop max_newton/max_newton losses 2.1e-09 x 1.2e-10 
gn_tr_apply_manifold               SAME  steps 12/12 stop max_newton/max_newton losses 3.7e-14 x 3.4e-14 
exact_tr_probe_manifold            SAME  steps 10/10 stop gradient/gradient losses 2.5e-14 x 1.5e-15 fields:rho
exact_ls_apply_manifold            SAME  steps 10/10 stop max_newton/max_newton losses 8.8e-15 x 4.6e-14 
exact_tr_entries_corewise          SAME  steps 10/10 stop max_newton/max_newton losses 4.2e-16 x 1.7e-15 
switch_probe_manifold              SAME  steps 10/10 stop gradient/gradient losses 2.3e-14 x 2.8e-15 fields:rho
switch_all_apply_manifold          SAME  steps 12/12 stop max_newton/max_newton losses 1.1e-12 x 2.1e-11 fields:delta_f,pHp,pKp,rho,slope
cone_probe_manifold                SAME  steps 8/8 stop max_newton/max_newton losses 8.4e-15 x 2.7e-15 cone 6.1e-16 
padded_probe_manifold              SAME  steps 8/8 stop max_newton/max_newton losses 1.7e-13 x 1.7e-15 start 3.0e-16 
padded_exact_apply_manifold        SAME  steps 8/8 stop max_newton/max_newton losses 4.2e-15 x 1.1e-14 start 3.1e-16 
reg_gn_probe_manifold              SAME  steps 10/10 stop max_newton/max_newton losses 5.4e-15 x 2.5e-15 
reg_exact_apply_corewise           SAME  steps 10/10 stop max_newton/max_newton losses 1.2e-15 x 1.3e-15 
reg_switch_probe_manifold_tr       SAME  steps 10/10 stop gradient/gradient losses 1.0e-14 x 2.3e-15 fields:rho
weighted_probe_manifold            SAME  steps 10/10 stop max_newton/max_newton losses 1.7e-14 x 1.9e-15 
deriv2_probe_manifold              SAME  steps 10/10 stop max_newton/max_newton losses 2.3e-14 x 2.4e-15 
deriv1_exact_probe_manifold        SAME  steps 9/9 stop gradient/gradient losses 1.1e-14 x 2.7e-15 
gd_probe_manifold                  SAME  steps 15/15 stop None/None losses 7.2e-14 x 3.8e-15 
gd_apply_corewise_reg              SAME  steps 15/15 stop None/None losses 3.6e-16 x 4.6e-16 
gd_cone_apply_manifold             SAME  steps 6/6 stop None/None losses 1.6e-14 x 4.3e-14 cone 8.0e-16 
mcsgd_flat_probe_manifold          SAME  steps 40/40 stop None/None losses 1.3e-13 x 3.7e-15 
mcsgd_flat_reg_apply_manifold      SAME  steps 40/40 stop None/None losses 3.8e-14 x 8.0e-14 
mcsgd_custom_entries_manifold      SAME  steps 30/30 stop None/None losses 1.9e-13 x 5.2e-14 
adam_flat_apply_corewise           SAME  steps 100/100 stop None/None losses 8.4e-11 x 4.0e-10 
adam_reg_probe_corewise            SAME  steps 100/100 stop None/None losses 2.0e-11 x 1.0e-10 
u_gn_probe_manifold                SAME  steps 10/10 stop max_newton/max_newton losses 2.3e-14 x 2.2e-15 
u_exact_apply_manifold             SAME  steps 8/8 stop max_newton/max_newton losses 5.4e-15 x 3.1e-14 
u_switch_probe_manifold_reg        DIFF  steps 10/10 stop max_newton/max_newton losses 1.6e+00 x 1.4e-06 fields:cg_iters,delta_f,gnorm,hessian,misfit,objective,pHp,pKp,regularization,rho,slope,switch_count,switch_fired,tr_radius
u_exact_entries_corewise           SAME  steps 8/8 stop max_newton/max_newton losses 3.7e-16 x 9.6e-16 
u_cone_apply_manifold              SAME  steps 6/6 stop max_newton/max_newton losses 4.1e-14 x 2.0e-14 cone 6.9e-16 
u_padded_probe_manifold            SAME  steps 6/6 stop max_newton/max_newton losses 4.0e-14 x 3.6e-15 start 3.1e-15 
u_mcsgd_probe_manifold             SAME  steps 30/30 stop None/None losses 6.9e-14 x 5.3e-15 
u_adam_apply_corewise              SAME  steps 100/100 stop None/None losses 8.4e-11 x 4.0e-10 
u_deriv1_probe_manifold            SAME  steps 8/8 stop max_newton/max_newton losses 8.8e-15 x 2.7e-15 
```

35 of 36 are the same run to 1e-9 (most to 1e-13; MC-SGD 1e-13, the regularizer now unscaled; Adam 1e-10, its `eps`).
The `rho` fields that differ are rows at the objective's resolution. Under D9's (d) (round 1) the same 36 cases re-run give the same table (35 SAME, the
same DIFF), as expected: 2026.5.0 compared point route with point route. The one DIFF, `u_switch_probe_manifold_reg`, is
the radius tie of section 6: row 0's CG step lands exactly on the default radius, the boundary flag is `True` in the old
run and `False` in the new one (`pHp` 6804.738785484038 against 6804.738785484036), so the old run doubles the radius
for row 1; both converge to the same point (1.4e-6) and objective. The translation of the old weight forms is part 4's.

## 6. Found outside the part

- **The default trust-region radius is a rounding tie** (already documented in `test_optimizers._untied_radius`): the
  Gauss-Newton Cauchy length is exactly the length of CG's first iterate, so the iteration-0 boundary flag, and with it
  the first expansion, is decided by the last bit. It is the one side-by-side difference above. A relative tolerance on
  the boundary test would remove the tie; not changed. **For the backlog** (the reviewer confirmed it, round 1).
- `UniformTuckerTensorTrain.from_t3(..., n=, r=)` truncation (the slice after P16): `TestSlackPaddedStart`'s old start
  was non-minimal and was silently reduced by the frontend; now its start is minimal.

## 7. Expected red until parts 3 and 4

`P16_part2_EXPECTED_RED.txt`, measured on the full suite above (the reviewer's matched it exactly): 103 ids since round
1 (the two D9 ids are green), each with its reason in the file's header.
- **Part 3** (13): all of `tests/backend/test_optimizer_display.py` (`make_newton_display` reads the old model).
- **Part 4** (90): the frontend files (`test_fitting`, `test_optimizers_frontend`, `test_sharing`'s fitting cases,
  `test_aligned_padded_frame_frontend`, `test_chunk_size_estimator`'s exact case, `test_jax_absent`'s script,
  `test_oracle_sweep`'s O2) and the frontend rows of `test_dispatch`: the frontend builds the old backend model or the
  removed `Problem`.
- **Against part 1's list**: 43 of its 67 ids are green now (the backend files); the other 24 are red for the part 4
  reason, none for a new one. One test outside the list turned red through D5 and is fixed:
  `tests/backend/test_geometry.py::TestTheFrameReturnsTheCenters::test_uniform_with_forced_padding` (P15's; its shared
  point packed at `n = 4` below its ranks was truncated into a non-minimal point, which `from_point` now refuses): it
  sizes `n`, `r` from the point and passes the sharing to `from_point`, as the from_t3 slice prescribes.
