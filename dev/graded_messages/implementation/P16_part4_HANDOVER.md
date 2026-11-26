# P16, part 4 (the frontend and the translation): handover IN PROGRESS

Implementer (parts 2 to 4), 2026-11-21, on `p16-wip` at `322baefd` (part 3 committed). **Not finished.** The implementer
stops at its context limit with the library done and two test files migrated; a fresh implementer continues from
here. Nothing is committed. Nothing is running.

## 1. Done: the library (all module doctests pass)

**`t3toolbox/fitting.py`**
- **Removed:** the six factories, `_check_ragged_sample`, `_uniform_model`, `_with_curvature`, `_ragged_frame`,
  `_ragged_geometry_aux`, `_backend_geometry_ops`, and the old `LocalModel`. `_canonical_weight` is kept for the
  translation (P17 deletes it).
- **New `LocalModel(geometry, backend)` (D6).**
  - `frame` is a cached property derived from the backend frame (`_frontend_frame`): manifold `T3Frame` / `UT3Frame`;
    corewise, the `(B, G, G, G)` frame of the point, via `t3_corewise_frame` / `ut3_corewise_frame`.
  - Read-only `oracle`, `batch` and `held`, and `residual`.
  - Every method keeps its name and is one backend call. `gn_hessian` on an exact model goes through
    `bopt._gauss_newton_view`. `evaluate` is `c + <g, Πp> + ½ quadratic(p)`. `retract` is the frontend geometry's
    (shared: with `precompute`).
  - The same-frame guard is unchanged.
  - Pytree: the child is `backend`, the aux is `geometry`.
- **New `local_model(geometry, oracle, batch, x, regularizer, hessian, curvature_damping, frame)`.**
  - **Checks:** the layer, then `oracle.check_batch` on a ragged batch, then the batch is packed for a uniform `x`.
  - **A given frame:** checked in safe mode by `check_given_frame`: orthonormal, `consistency_residual <=
    safety.effective_rtol`, represents `x`; a corewise frame only for representing `x`. It is passed to the backend
    without centers.
  - **Shared geometry:** `x` is tied first (`_tied`), as `SharedGeometry.frame` does.
  - **D5's path:** a uniform manifold model builds its geometry with `from_frame`, from
    `ut3_orthogonal_representations(x.data)`, and passes those centers. A non-minimal `x` therefore works; the
    optimizers instead reduce to minimal and use `from_point`.
- **Helpers shared with the optimizers** (private): `_require_layer`, `_is_manifold`, `_sharing_of`,
  `backend_geometry`, `_is_packed`, `prepare_batch`, `check_given_frame`.
- **Docs:** the module docstring is rewritten, `__all__` is updated, and the new doctests were run.

**`t3toolbox/__init__.py`** (D7): the factories are out; `local_model`, `LeastSquaresOracle` and `LeastSquaresBatch`
are in.

**`t3toolbox/optimizers.py`** (rewritten)
- **The heads, plan 2.5:**
  - `gradient_descent(geometry, oracle, batch, x0, regularizer=None, g0_reference=None, **kwargs)`;
  - `mc_sgd` and `adam(geometry, oracle, batch, x0, rng, batch_size, regularizer=None, draw=None, use_jit=False,
    **kwargs)`;
  - `newton_cg(geometry, oracle, batch, x0, regularizer=None, verbose=False, error_table=True, val_batch=None,
    callback=None, use_jit=False, <today's keywords>, *, <today's keyword-only>)`.
- **Gone:** `least_squares_problem`, `LeastSquaresSetup`, `_resolve_chunk_size`, `_check_kind`, the kind maps,
  `chunk_size`.
- **`_setup` (before any work):**
  - the layer;
  - `check_batch` on a ragged batch (structural, before anything else);
  - `x0` reduced to minimal ranks (a manifold geometry; always on the uniform layer);
  - the backend geometry;
  - the batch packed.
- **`newton_cg`:**
  - `val_batch` is checked at entry.
  - Under `use_jit` it converts `x0` and the batch to jax **before** building the display, so the display holds the
    very batch the run uses (the part 3 ruling).
  - A supplied `start_frame` goes through `check_given_frame`, which adds the consistency check (O-5).
  - `padded_from` is unchanged.
- **Draws:** a new-form custom `draw` is wrapped to pack its batch for a uniform `x0` (`_frontend_draw`).
- **The translation, a decorator `@_translates`** on the four optimizers. A string in the oracle's position, or
  `kind=`, marks the old form. The old arguments are bound to old-signature stubs (`_old_*`) and rebuilt:
  - **`_old_fit`:**
    - plain kinds at `()`: the old canonical weight, probe `(d,)`; apply and entries with a weight raise as before;
    - derivative kinds at `(n,)`, `(0,)` included: `affine_jet`; `entries_derivatives` becomes the apply kind at
      `affine_jet(index_to_one_hot(index, shape), pp)`; the weight is canonicalized and broadcast to the exact
      shape, then `prepare_derivative_data`.
  - **Validation:** both-or-neither on `val_sample` / `val_data`, which become `val_batch`.
  - **mc_sgd / adam:** `batch` becomes `batch_size`. An old custom draw is wrapped (`_old_draw`), with the full
    per-draw conversion and `minibatch_scale = n / min(batch, n)`.
  - **`chunk_size`** is accepted and dropped.
- **Docstrings:** `newton_cg`'s 26 KB docstring is kept and its examples are converted to the new form; the doctest
  outputs are unchanged, so the translation and the new form give the same numbers there. The other three
  docstrings are shortened; `gradient_descent` gets a new-form example.

**`t3toolbox/backend/optimizers.py`**: `_prepare_jit_inputs` returns an already-jax tree **as is** (the same object),
so a caller that converted first keeps the display's fast path under `use_jit`. This is the only backend change. It
is checked by `test_verbose_error_table_and_the_runs_batch`: under `use_jit` each display numerator carries the
model's `held`.

## 2. Done: tests

- **`tests/test_optimizers_frontend.py`: 49 passed, 102 subtests.**
  - New `TestFrontendHeadsAndKeywords`: the heads; minimal `x0` on the manifold only; gradient against the frontend
    `local_model`; all four optimizers go through `_setup`; the removed names; the D7 exports; a misspelt option
    raises before setup; **a bad batch raises before any work, and the backend runs it unchecked**; the keyword
    mirror; forwarding; the timing keywords.
  - New `test_verbose_error_table_and_the_runs_batch`:
    - `error_table=False` prints the header lines only, and its `diagnostics` equal the run with the table;
    - at `(1, 1)` the records have the weight's shape and the columns are the multi-indices in row-major order;
    - the display's numerators carry the model's `held`, under `use_jit` too.
  - **The "frontend run is the backend run bitwise" tests** now compare against an independently built backend twin
    (`_twin`, not `_setup`); all bitwise, ragged, uniform and shared.
  - **Problem builders:** `_restart_problem`, `_shared_restart_problem` and `_switch_problem` are rebuilt on the
    oracle; `_ob` turns the old vocabulary into an oracle and a batch with `tests/_fits`.
  - **Removed:** the two `chunk_size='auto'` tests.
  - Verbose tests: records of shape `(d,)` (was `(d, 1)`), new form.
- **`tests/test_aligned_padded_frame_frontend.py`: 12 passed.** The twin test is built directly on the backend; the
  safe-mode message is now `does not represent the point`.
- **`tests/_fits.py`** gains the test-only shims for the 2026.5.0 vocabulary, independent of the translation:
  - `old_vocabulary(kind, sample, data, shape, order, weight)` returns `(oracle, batch)`;
  - `old_model(...)`;
  - `old_factory(kind)`: the old factory's **exact signature** over `fitting.local_model`, with data
    `y = z(x) - residual`.

## 3. Started, not finished: `tests/test_fitting.py`

The factory calls are replaced by `F.old_factory('<kind>')(...)`; the module imports. **145 tests and subtests
fail**, for these reasons:
- **`bopt.least_squares_problem` / `uf.uniform_least_squares_problem` (~85).** These are the "frontend equals the
  backend model" comparisons, built on the removed `Problem`. Rewrite each to compare with
  `bopt.local_model(<backend geometry>, oracle, batch, x.data or the supercore pair)` on the same oracle and batch
  (`F.old_vocabulary`), as part 2 did for the backend files.
- **`model.kind` / `model.curvature` / `model._bgeom` / the old 7-argument `LocalModel(...)` (~30).** These test the
  old model's internals. Use `model.backend.curvature`, `model.backend.geom`, the oracle; or delete where only the
  old internals are tested (`TestBlockSumsq`, the parts of `TestResidualWeighting` and `TestPerModeWeightRowCount`
  that test the old kinds' `block_sumsq` and weights: the old kinds keep their own tests until P20).
- **"False is not true" (~11), mostly `TestLocalModel` at `C=(2,)` and `test_jacobian_gradient_adjoint`.** The
  oracle's `y` cannot carry the point's stack (it broadcasts over `C`; checked: `check_batch` rejects a `y` with `C`,
  and the engine cannot broadcast it), so `old_model` takes element 0's data. **`_setup` must draw `y` without `C`
  and set `r = z(x) - y`**, so that each element's residual is right. Check the adjoint test's residual likewise.
- **`test_curvature_damping_...`:** the error messages name the old factory (`probe_derivatives_model: ...`); the new
  message comes from the backend `local_model`.

## 4. Not started

- **Tests:**
  - `tests/test_manifold.py` (its `fitting.apply_model` call);
  - `tests/test_sharing.py`'s fitting cases;
  - `tests/test_oracle_sweep.py` with `tests/oracle_sweeps/sweep_models.py`, `sweep_exact.py`,
    `sweep_optimizers.py`, `oracle.py` (the kind strings, the factories, `least_squares_problem`);
  - `tests/test_jax_absent.py` (its script calls `bopt.least_squares_problem`, `probe_derivatives_model` and
    `newton_cg`);
  - `tests/test_chunk_size_estimator.py` (remove `test_resolver` and `test_newton_cg_exact_across_chunk_size`);
  - the frontend rows of `tests/test_dispatch.py` (part 3's expected-red list names them).
- **New tests:**
  - the translation: each old call form equals the new one, with a few pinned numbers; plain and derivative kinds;
    `entries_derivatives`; order 0 as `(0,)`; a weight; a custom draw with its scale; `val_sample` / `val_data` both
    or neither; `chunk_size` dropped;
  - the frontend-only fit: derivative data, importing nothing from `t3toolbox.backend`;
  - the `local_model` frame checks: a perturbed `start_frame` or `frame` raises in safe mode and runs under
    `safety.unsafe()`; a library-built frame is not checked;
  - a user oracle without a draw raises in `mc_sgd`;
  - the floor `eps |f|` on an oracle without `f_resolution`, through the frontend.
- **Pages:** the factory doctests of `docs/fitting_and_optimization.md` (lines 147, 182, 424, 430) and
  `docs/numerical_contracts.md` (line 94).
- **Examples:** these move to `fitting.local_model`: `fit_hilbert_from_entries_lbfgs.py`,
  `fit_hilbert_from_probes_adam.py`, `fit_hilbert_from_probes_optax.py`, `fit_hilbert_tensor_newton_cg.py`,
  `fit_varied_rank_tensor_newton_cg.py`. The examples and pages that use the old optimizer form keep working through
  the translation (P17).
- **Side by side:** add the translation's cases to `dev/graded_messages/implementation/bench/p16_part2_side_by_side.py`
  (old form through the translation against the pre-P16 library).
- **The gate:** the full gate at `-n 2` (six steps), then the slow tier serially; MemAvailable; the final handover;
  the end of P16.

## 5. A decision to rule

**The old call form inside the frontend test files.** Many tests in `tests/test_optimizers_frontend.py` (for example
`TestFrontendOptimizers`, `TestFrontendUniformOptimizers`) still call the optimizers in the old form. They pass, now
through the translation (so they test it too). Converting them is mechanical but runs to hundreds of calls across the
frontend files. They must convert before P17 deletes the translation. I recommend doing it with the translation's
deletion (P17), where every remaining old-form call (pages, examples, tests) is found by deleting it. The alternative
is to do it in this slice. Ruled conversions done here so far: every test of a removed name, every bitwise
frontend-equals-backend comparison, the verbose tests.

## 6. Notes for the next implementer

- `tests/_fits.old_factory(kind)` keeps the old factory signatures exactly. Use it to keep each test's arithmetic as
  it was, and compare with `bopt.local_model` on `F.old_vocabulary(...)`'s oracle and batch.
- `_twin` in `tests/test_optimizers_frontend.py` is the pattern for "frontend equals backend": an independent backend
  geometry, reduced cores and packed batch.
- The expected-red ids of part 3 (`P16_part3_EXPECTED_RED.txt`, 90, all frontend) are the targets. Run them by file.
