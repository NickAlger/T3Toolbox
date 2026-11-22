# P16 part 4a, response to review round 1

Implementer (4b, the fresh one), 2026-11-21, on `p16-wip` at `4047ed53` plus the working tree. Claude ruled on all ten
findings. Each finding is below, with what changed and the evidence. 4b has not started: nothing outside the 4a fixes
was touched.

## Findings

**1. Accepted. A plain-probe weight of length 1 now broadcasts over the modes.**
- **Change:** `optimizers._old_fit` now sets `w = np.broadcast_to(wm[:d, 0], (d,))`. `wm` has one row or at least `d`
  rows.
- **Test:** `TestReview4aChecks.test_the_old_plain_probe_weight_of_length_one_broadcasts`. `weight=[2.]` gives the
  same losses as `weight=[2., 2., 2.]`, and the translated batch's weight is `(2, 2, 2)`.

**2. Accepted. The orthonormality check of a given frame now uses `safety.effective_rtol`.**
- **Change:** `check_given_frame` calls `frame.is_orthogonal(atol=rtol)`, with the same `rtol` as the consistency check.
- **Test:** `test_a_float32_frame_the_library_built_passes_and_a_perturbed_one_raises`.
  - The float32 jax frame from `MANIFOLD.frame` has an orthogonality residual above 1e-9. The test asserts this, so it
    would fail a fixed tolerance.
  - That frame passes, and the model holds it.
  - The same frame with its down cores scaled by 1.01 raises "not orthonormal". Skipped without jax.

**3. Accepted (decision A). `frame=` on a corewise geometry raises a structural error in the frontend.**
- **Change:**
  - `local_model` raises at its top: "local_model: frame= needs a manifold geometry ...; on a corewise geometry the
    frame is the point itself: pass frame=None". This happens before any check, in both safety modes.
  - The dead corewise branch is deleted from `local_model` (the `(B, G)` rebuild) and from `check_given_frame`, which
    now serves manifold geometries only. The optimizers already refuse `start_frame` on a corewise geometry.
  - Docstring and signature comment updated.
- **Test:** `test_frame_on_a_corewise_geometry_raises_structurally` covers `COREWISE`, `UNIFORM_COREWISE` and
  `shared_corewise`, in safe and in unsafe mode.

**4. Accepted (ruled in scope). On a shared geometry, a given frame must have tied down cores, in safe mode.**
- **Change:** `check_given_frame` adds, on a shared geometry,
  `safety.require(np.all(base.has_shared_tucker_factors(sharing, rtol=rtol)), ...)`, where `base` is the frame's
  `(D, L)` point, ragged or uniform.
  - It covers `local_model(frame=)` and `newton_cg(start_frame=)`, which share the helper.
  - `newton_cg`'s docstring now lists the tie among the `start_frame` checks.
- **Test:** `test_a_shared_geometrys_given_frame_must_be_tied`.
  - For both entry points, on `shared(MANIFOLD)` and `shared(UNIFORM_MANIFOLD)`: the frame of an untied point raises
    "not tied" in safe mode and runs under `safety.unsafe()`.
  - The frame of the tied point passes.

**5. Accepted. The shim `_fits.old_model` raises unless `y` is the same over the stack.**
- **Change:** a new helper `_one_over_the_stack` checks that every element's `y = z(x_c) - residual_c` equals
  element 0's, to rounding (`rtol` 1e-9, `atol` 1e-9 times the scale), before taking it. Otherwise it raises "a
  stacked point needs one y for every element". The docstring states the rule (decision D: one `y` per stacked model).
- **Test:** `test_the_old_model_shim_needs_one_y_over_the_stack`.
  - A residual `z(x) - y` with one `y` at `C = (2,)` builds a model whose `residual` is the stated one.
  - A random per-element residual raises.

**6 and 7. Accepted. There is one batch helper, and `check_batch` is skipped only for a uniform point whose batch is
already packed.**
- **Change:** `fitting.prepare_batch(oracle, batch, x, check=True)` is the single helper.
  - It checks with `oracle.check_batch` unless `x` is uniform and the batch is already packed.
  - Then it packs a `LeastSquaresBatch` for a uniform `x`.
  - `check=False` packs only. The uniform draw wrapper uses it: a draw's batch is packed every step and not re-checked,
    as before.
- **Callers:**
  - `optimizers._setup` calls it before the reduction of `x0`. The reduction keeps `N`, checked numerically
    (`uniform_minimal`: `N` 7 before and after).
  - `newton_cg`'s `val_batch` calls it before the setup, and the display takes its result.
  - `local_model` calls it.
- **Removed:** `optimizers._pack`; the unused `who` arguments of `prepare_batch` and of `optimizers._setup` (call sites
  and the three test calls updated).
- **Test:** `test_the_batch_is_checked_unless_packed_for_a_uniform_point`.
  - It spies on `LeastSquaresOracle.check_batch` in four cases: a ragged point with a list sample (checked); a ragged
    point with an ndarray sample (checked, the case finding 6 found unchecked); a uniform point with a ragged batch
    (checked and packed); a uniform point with a packed batch (not checked).
  - `check=False` does not check.

**8. Accepted. The test-file form items are fixed.**
- **Continuation lines:** the trailing-space continuation lines are joined, and four continuation lines left at the old
  indentation are realigned (`newton_cg(..., restart,` / `max_newton=12, **kw)`, and the like). `x0,  max_newton` now
  has one space. The 128-character `bopt.newton_cg(..., hessian='switch', **kw)` line is wrapped.
- **`_twin`:** loses its ignored `hessian` argument, and its calls lose `hessian='exact'` / `'switch'`.
- **Stale text in two classes:** `TestFrontendExactNewtonPlainKinds` and `TestFrontendExactNewtonDerivativeKinds` lose
  the vacuous `sd.estimate_chunk_size` spy and its imports. Their docstrings lose the `chunk_size='auto'` sentence and
  the mention of `least_squares_problem`.
- **`_ob`:** deleted; its five call sites use `F.old_vocabulary`, which was line for line the same. The comment on the
  `_fits` import is corrected.

**9. Taken into 4b.** The translation tests will have:
- a regularized custom draw (the scale);
- order 2 or more (the `t!` conversion);
- pinned numbers from `scratchpad/reviewer3/fe_sbs.py` / `old.pkl`.

**10. Done (trivial and clearly right). The shared `LocalModel.retract` reuses the build's companion.**
- **Change:** `retract(p, shared_data=self.backend.geom_aux)`.
  - The backend model's `geom_aux` is `geom.precompute(frame)`. That is `fv_shared_frame_data` / `ufv_shared_frame_data`
    of the same frame and groups, the same function the frontend `SharedGeometry.precompute` calls.
  - On a shared corewise geometry it is `None`, as the frontend's is.
- **Test:** `test_shared_retract_reuses_the_builds_companion`, ragged and uniform shared manifold.
  - `SharedGeometry.precompute` is not called during `retract`.
  - The result is bit for bit the retraction with the companion recomputed from the frame.

## Tests of the tests

These are mutants in a scratch copy of `t3toolbox/` and `tests/`, since deleted. Each was run against
`TestReview4aChecks`, and all seven are killed:
- the old `wm[:d, 0]`;
- `is_orthogonal()` at its default tolerance;
- the tie check removed;
- the packed skip for any point;
- `retract` recomputing `precompute`;
- the shim's check removed;
- the corewise `frame=` raise removed.

## Runs

- `tests/test_optimizers_frontend.py` and `tests/test_aligned_padded_frame_frontend.py`, `-n 2`: **68 passed, 135
  subtests**, 99 s (was 61 / 122; the 7 tests and 13 subtests are `TestReview4aChecks`). MemAvailable afterwards
  11.9 GB.
- Doctests, `--doctest-modules t3toolbox/fitting.py t3toolbox/optimizers.py t3toolbox/backend/optimizers.py`: 10
  passed.

## Files changed

- `t3toolbox/fitting.py`: `_is_packed`, `prepare_batch`, `check_given_frame`, `local_model`, `LocalModel.retract`.
- `t3toolbox/optimizers.py`: `_setup`, `_frontend_draw`, `_pack` (removed), `_old_fit`, `newton_cg`'s `val_batch` and
  its docstring.
- `tests/_fits.py`: `old_model`, `_one_over_the_stack`.
- `tests/test_optimizers_frontend.py`: the form items, and the new `TestReview4aChecks`.

## For the 4b handover (decision C)

`tests/_fits.old_factory`, `old_model` and `old_vocabulary` keep the 2026.5.0 vocabulary in the tests. P17's search for
the old call form will not find them, so they are listed for conversion or renaming in P17.
