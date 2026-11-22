NOT YET

# P16 part 4a, review round 1

Reviewer (the third of P16), 2026-11-21, on the checkpoint `4047ed53` against its parent (`git diff 4047ed53~1 4047ed53`),
read only. Briefed by `BRIEF.md`, `P16_REVIEWER_HANDOFF.md`, `CLAUDE.md`, the plan (2.5, P16, P17), the P16 design note
and review, `LOG.md` from "P16 started", `TARGET_DECISIONS.md`'s 2026-11-21 notes, and `P16_part4_HANDOVER.md`.
Required reading done: `TuckerTensorTrain.probe` (to `jets.check_sample`, `probing.t3_probe`,
`sampling_variables.sampling_variables`, `contractions.jet_contract`), `__add__` (to `t3_linalg.t3_add`) and `t3svd`
(to `t3_svd.t3svd` and `_t3svd_shared`).

**Overall.** The library half of part 4 is right where it matters most. The translation keeps the old meaning: against
the pre-P16 library, on 25 old-form calls, every case runs to the same point and the same history to rounding, except one
weight form that now raises (finding 1). The display gets the run's batch, also under `use_jit`. The frontend checks sit
where the plan puts them. What remains are should-fix items at the check sites and in one test shim, and no blocking one.
Items 2 to 4 of the 4b implementer's read-only notes were relayed to me; I checked each independently and they hold
(findings 1, 2, 3, 6, 7).

## What I checked, and how

1. **The translation against the pre-P16 library** (handoff items 3 and 4). `scratchpad/reviewer3/fe_sbs.py`: the same
   data from the same seeds (targets built from numpy cores, derivative data from my own dense Taylor coefficients times
   `t!`); old-form frontend calls run on `reviewer2/old_lib` and on the new tree, compared on the final dense tensor, the
   losses, the history's objectives and the verbose records. All to rounding (dx at most 1.4e-14) unless noted:
   - plain probe with a per-mode weight, and with extra rows;
   - bare weight vectors on derivative kinds at `d == n + 1` (`apply_derivatives` and `probe_derivatives`, order 2):
     read per order, as before;
   - `probe_derivatives` with the full `(d, n+1)` matrix, extra rows, a `(d, 1)` column and a `(1, n+1)` row;
   - `entries_derivatives` (now apply at the one-hot jet); order 0 for probe and apply (apply at order 0: dx 3e-10,
     loss 2e-14; a flat minimizer);
   - positional `order` and `weight` with `hessian='exact'`; corewise exact with a regularizer; a shared manifold;
   - `val_sample`/`val_data` with `verbose`: the records equal the old ones to 6e-15; one of the two alone raises the
     same error in both;
   - `mc_sgd` with a custom draw and a regularizer, the positional `batch`; `mc_sgd` with the flat draw and `batch=`;
     `adam` (corewise, `entries_derivatives`, custom draw, regularizer): 9e-8, the ruled `eps` effect;
     `gradient_descent` with `chunk_size`; uniform Newton and uniform `mc_sgd` with a custom draw and a regularizer.
   - Four new-form calls built independently (Taylor data, weight times `t!`) give the old runs too.
   - **One divergence:** finding 1.
2. **Batch identity under `use_jit`.** The fast path needs the display's `batch` to be `info.lm.batch`. The frontend
   converts the batch before building the display, and `_prepare_jit_inputs` now returns an all-jax tree as the same
   object. Two mutants of mine are killed by `test_verbose_error_table_and_the_runs_batch`: `_prepare_jit_inputs`
   always converting, and the display handed `s.batch` in place of `run_batch`.
3. **Check placement.** `check_batch` runs before any work in the optimizers and in `local_model`. A mutant of mine that
   deletes it in `_setup` is killed by `test_a_bad_batch_raises_before_any_work`. A supplied `start_frame` and
   `local_model(frame=)` are checked in safe mode only, with the structural type check always (but see findings 2 and
   3). The old-signature stubs' positional order is the 2026.5.0 signatures', `*` included (an extra positional raises
   as before).
4. **The tests run.** `tests/test_optimizers_frontend.py` and `tests/test_aligned_padded_frame_frontend.py`, at `-n 2`:
   61 passed, 122 subtests, 96 s.
5. **Translation mutants against 4a's own suite.**
   - The custom draw's `minibatch_scale` dropped: 49 passed (it survives). My side-by-side catches it with a
     regularizer (2.6e-3, 7.8e-2, 7.1e-3), not without one (4e-15), as the handoff predicted.
   - The `t!` weight conversion skipped: 49 passed (it survives). My side-by-side catches it at order 2 (dx 0.17,
     0.023), not at order 1, where `1! = 1`.

   The translation's own tests are 4b's work. See finding 9.

## Findings

**1. Should fix. The translation rejects a plain-probe weight of length 1, which 2026.5.0 accepted.**
- **Place:** `t3toolbox/optimizers.py`, `_old_fit`, plain branch, `w = ... wm[:d, 0]`.
- **Evidence:**
  - `newton_cg(MANIFOLD, 'probe', ww, y, x0, weight=[2.])` runs on `old_lib`, with the same history and the same point
    as `weight=[2., 2., 2.]` (my `b_check.py`: identical objectives, point difference 0.0). `_canonical_weight` gives
    `(1, 1)` and the old probe kind broadcast it over the modes.
  - Through the translation it raises in `check_batch`: "the weight of the probe kind has shape (1,); expected exactly
    (d,)+F = (3,)". The 4b implementer's `w1.py` found the same.
- **Resolve:** broadcast to `(d,)`: `np.broadcast_to(wm[:d, 0], (d,))` (`wm` has 1 or at least `d` rows). Add the case to
  the translation test.

**2. Should fix. The safe-mode orthonormality check of a given frame uses the fixed `atol=1e-9`, so a float32 frame the
library itself built fails it.**
- **Place:** `t3toolbox/fitting.py`, `check_given_frame`: `frame.is_orthogonal()` (default `atol=1e-9`). The
  consistency check two lines below uses `safety.effective_rtol(*arrays)`.
- **Evidence:** `t3m.MANIFOLD.frame(x.to_jax())` has orthogonality residual 6.6e-7, and `effective_rtol` is 1e-5 there.
  Both `local_model(MANIFOLD, oracle, batch, xj, frame=fr)` and `check_given_frame` alone raise "the frame is not
  orthonormal" (my `items23.py`; the 4b implementer's `f32_frame.py` likewise).
- **Why it matters:** CLAUDE.md's rule for check sites is `safety.require(<checker>(atol=safety.effective_rtol(...)))`,
  so that jax float32 works in safe mode. The fixed `1e-9` is inherited from 2026.5.0's `start_frame` check. Part 4
  moved it into a shared helper, where it now also guards the new `local_model(frame=)`, beside a consistency check that
  uses the right tolerance.
- **Resolve:** `frame.is_orthogonal(atol=rtol)` with the same `rtol`. Add a test: a float32 jax frame from
  `MANIFOLD.frame` passes, and a perturbed one still raises.

**3. Should fix. `local_model(frame=)` on a corewise geometry is a dead path, and its frontend check would pass a frame
in another gauge.**
- **Place:** `t3toolbox/fitting.py`, `local_model` (the `if not manifold` branch that builds `(B, G)`) and
  `check_given_frame` (the corewise case checks only `base.allclose(x)`).
- **Evidence:**
  - The backend `local_model(frame=)` needs `require_frame_structure`, which only the two manifold geometries have.
    `local_model(COREWISE, ..., frame=COREWISE.frame(x))` raises "frame= needs a geometry with a
    require_frame_structure member; CorewiseGeometryOps has none". That is a backend message, raised after the frontend
    check.
  - A corewise frame is the point, so its coordinates depend on the gauge. Yet a frame built from
    `x.left_orthogonalize_tt_cores()` (cores differ, same tensor) passes the frontend check (`items23.py`; the 4b
    implementer's `cw_frame.py`).
- **Resolve:** recommended, a structural `ValueError` at the top of the frontend `local_model`: `frame=` is for a
  manifold geometry, since a corewise frame is the point itself (`frame=None`). Delete the dead branch and the corewise
  case of `check_given_frame` (the optimizers already refuse `start_frame` on corewise). If Claude prefers to keep
  corewise frames, the check must be `corewise_equal`, not `allclose`, and the backend corewise geometries need
  `require_frame_structure`; that is more surface for no use I can see.

**4. Should fix (Claude to rule on scope). On a shared geometry, a given frame is not checked for tied down cores, and an
untied one gives a silent wrong model.**
- **Place:** `t3toolbox/fitting.py`, `local_model` (with `frame` given, `x` is not tied) and `check_given_frame`.
- **Evidence:** `local_model(shared(MANIFOLD, (0, 1, 0)), o, b, x, frame=MANIFOLD.frame(x))` with an untied `x` runs.
  Its `f` is the untied point's, and its "tied" gradient has different blocks on modes 0 and 2 (`shared_frame.py`). The
  shared projection assumes tied down cores.
  - Without `frame=`, `x` is tied first (documented, as `SharedGeometry.frame`), so only the given-frame path is open.
  - The same gap exists for `newton_cg(start_frame=)`, inherited from 2026.5.0. A minimal untied `x0` skips
    `_ragged_minimal`'s safe-mode tie check.
- **Resolve:** in `check_given_frame`, on a shared geometry, safe mode requires
  `base.has_shared_tucker_factors(sharing)` (numerical, `effective_rtol`). This covers both entry points. It is a small,
  numerical precondition at an existing check site. If Claude would rather not widen 2026.5.0's `start_frame` behaviour
  in P16, log it as a backlog item instead.

**5. Should fix. The test shim `old_model` silently uses element 0's data for a stacked point.**
- **Place:** `tests/_fits.py`, `old_model`: `y = [yi[... (0,) * stack] ...]` and `y = y[(Ellipsis,) + (0,) * stack]`.
- **Evidence:** the oracle's `y` cannot carry `C`, so for a stacked `x` the shim takes element 0's
  `y = z(x) - residual`. Every other element then gets a model whose residual is not the caller's `residual`. A test that
  checks a property holding for any residual (an adjoint identity, a model consistency) would pass while testing
  another residual than it states. The docstring states the precondition, but nothing enforces it. The handover's §3
  (the ~11 "False is not true" failures in `TestLocalModel` at `C=(2,)`) shows the shim is already used this way.
- **Resolve:** the shim raises unless `y` is equal (allclose) across the stack, so a test must build its residual as
  `z(x) - y` with one `y`, as the handover's §3 plans. The check fails loudly instead of fitting another residual.

**6. Minor. `_is_packed` decides from the sample alone.** `fitting._is_packed` treats any non-list sample as packed, so a
ragged `x` with an ndarray sample skips `check_batch`.
- In my checks (`item4.py`) the cases I could build failed loudly in the engine (an ndarray of jets gave the right `f`;
  a sample missing its jet axis gave a numpy broadcast error, not the checker's message).
- So this is a lost message rather than a silent answer. Condition the skip on a uniform `x` (a ragged `x` always gets
  `check_batch`).

**7. Minor. The batch logic is written twice.** `fitting.prepare_batch` (its `who` argument unused) and
`optimizers._setup` with `_pack` do the same check-then-pack. Have `_setup` call `prepare_batch` (or the reverse), and
drop or use `who`. A single site would also carry finding 6's fix once.

**8. Minor. Form in `tests/test_optimizers_frontend.py`.**
- **Continuation lines.** Several rewritten calls have a trailing space before the newline and continuation lines left at
  the old indentation:
  - `topt.newton_cg(t3m.MANIFOLD, oracle, batch, restart, ` with `max_newton=40` on the next line;
  - `_twin(geom, oracle, batch, restart, ` followed by `hessian='exact')` far to the right;
  - `x0,  max_newton=60`.
- **Stale text in `TestFrontendExactNewtonPlainKinds`.** Its docstring still speaks of `chunk_size='auto'`. Its uniform
  subtest still patches `sd.estimate_chunk_size` and asserts it is not called, which is now vacuous: the frontend no
  longer reaches it. Drop the spy and the sentence.
- **Duplicated helper.** `_ob` duplicates `tests/_fits.old_vocabulary` line for line; use the latter.
- **The `hessian` argument of `_twin`** is accepted and ignored (its docstring says so). Drop it from the calls.

**9. For 4b (not a defect of 4a). The translation has no test yet, and two of its parts are invisible to weak cases.**
Both translation mutants survive 4a's suite, as expected, since the translation's tests are planned for 4b. When they
are written:
- the `t!` conversion of the weight is visible only at order 2 or higher;
- the custom draw's `minibatch_scale` is visible only with a regularizer (Newton or `mc_sgd`; for Adam it is otherwise
  only `eps`).

`fe_sbs.py` holds the cases and a pre-P16 run (`old.pkl`). The 4b implementer can pin a few numbers from it, to a
tolerance.

**10. Minor (cost, not correctness). The frontend `LocalModel.retract` on a shared geometry recomputes
`geometry.precompute(self.frame)` on every call.** 2026.5.0 reused the model's companion. The backend model already holds
`geom_aux`; if its type is the one the frontend `retract(shared_data=)` takes, pass it. Otherwise leave it, with a
comment.

## Not findings (checked, right)

- **The old signature stubs** match 2026.5.0's positional order for all four optimizers (`adam` shares `mc_sgd`'s). The
  var-keyword bucket (`kwargs` / `keywords`) carries `start_frame`, `padded_*`, `timing`, `switch_*` and `n_iter`
  through to the new call. `chunk_size` is bound and dropped.
- **The old custom draw** is converted in full per draw (jets, one-hot for entries, Taylor data, weight), packed for a
  uniform `x0` by `_frontend_draw`, and scaled `n / min(batch, n)`, with `n` the full batch's `n_measurements`, as in
  2026.5.0's `_minibatch_step_problem`.
- **`_prepare_jit_inputs`** now converts with `tree_map(jnp.asarray)` in place of `tree_to_jax` for `x0`. On the
  backend's bare core pairs the two are the same (same leaves, same structure).
- **D5's path.** A uniform manifold `local_model` builds its geometry `from_frame` on
  `ut3_orthogonal_representations(x.data)`. That is exactly the backend's `frame(x)`, so the centers belong to that frame.
- **The validation batch** under `use_jit` stays numpy, and the display's held-free query promotes it. That is correct,
  and unchanged from 2026.5.0.

## Scratch (`/tmp/claude-1000/-home-nick-repos-T3Toolbox/352ad37b-5ac8-40a1-a326-a7fc3a369cb2/scratchpad/reviewer3/`)

- `fe_sbs.py`: the frontend side-by-side. `old.pkl` is from `reviewer2/old_lib`; `new.pkl` is from `4047ed53`.
- `b_check.py`: finding 1.
- `items23.py`: findings 2 and 3.
- `shared_frame.py`: finding 4.
- `item4.py`: finding 6.

The mutation copy is deleted. The working tree is as I found it, apart from this file.
