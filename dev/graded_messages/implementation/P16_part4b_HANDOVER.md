# P16, part 4b (the frontend tests, the translation tests, pages, examples): handover

Implementer (4b, the fresh one), 2026-11-21, on `p16-wip` at `c6b62ba0` (part 4a committed after its review). 4a's
record is `P16_part4_HANDOVER.md` with `P16_part4a_REVIEW_r1.md`, `P16_part4a_RESPONSE_r1.md` and `_r2`. Nothing is
committed. Nothing is running.

The ruling that bounds this part (Claude, per the plan): tests that call the optimizers in the old form, which the
translation keeps working, are converted in P17. P16 moves only what the translation does not cover: the six
factories, `least_squares_problem`, `Problem` and the other removed names, the backend corewise geometries, and the
kind strings with `_derivatives` where they reach removed code.

## 1. What changed

### Tests rewritten

**`tests/test_fitting.py`** (rewritten in the new form: `fitting.local_model(geometry, oracle, batch, x)`)
- **Helpers:**
  - `_backend_geometry` and `_backend_model` build the backend twin here (the uniform geometry by `from_point`, the
    batch packed), independently of `fitting.backend_geometry` and `prepare_batch`.
  - `_model` and `_pair` state a fit in the 2026.5.0 vocabulary through `F.old_vocabulary` (the kind names,
    derivative data, a derivative-convention weight).
  - The dense oracle `tests/oracle_sweeps/oracle.py` (`dense_oracle`: exact polynomial interpolation, no library jet
    code) is imported for derivative data.
- **`TestLocalModel`:**
  - `_setup` now draws data `y` without `C` (one `y` per stacked model, decision D); each element's residual is
    `z(x_c) - y` from the dense forward. This fixes every `C = (2,)` failure.
  - `test_dense_truth` also checks `objective_value == ½‖r‖²` with shape `C`. New: `test_residual`.
  - `test_matched_pair` and `test_jacobian_gradient_adjoint` use the `T3Tangent` transposes at the model's frame in
    place of `model.kind.transpose`. At the corewise frame `(B, G, G, G)` that is the bare corewise transpose.
  - `test_caching`: the frame and `held` replace the old `sweep`.
  - `test_derivative_models` is rewritten against ground truth. For apply / entries / probe derivatives at order 2
    with a per-order ω, on both geometries:
    - the objective equals the dense oracle's `½ Σ_t ω_t² ‖D_t(x) − y_t‖²`;
    - `evaluate(t p)` equals the dense objective at `x + t dense(Πp)` for `t = 1, 2`, which checks the gradient and the
      Gauss-Newton term;
    - the gradient, `gn_quadratic` and `gn_hessian` equal the backend twin's, bit for bit.
- **`TestBlockSumsq` is deleted** (decision B; the old kinds are deleted in P20 and nothing calls them).
- **`TestResidualWeighting`** (new form; ω placed by explicit indexing):
  - the full ω matrix against the dense objective, the ω²r gradient identity (data built so the residual is ω²⊙r)
    and `pᵀHp == ‖ω ⊙ t! Jp‖²`;
  - the apply / entries order-only weight against the dense objective, and a per-mode weight raising;
  - the plain probe per-mode weight.
  - The 2026.5.0 broadcasting forms (row, column, bare vector) moved to the translation test.
- **`TestPerModeWeightRowCount`:** in the new form a probe weight has exactly `d` rows, so more rows and fewer rows
  both raise in the batch check. 2026.5.0's truncation of extra rows is the translation's, and is tested there.
  `test_over_long_is_truncated` became `test_over_long_is_rejected`.
- **`TestUniformLayerLocalModel`:**
  - `test_matches_backend_local_model` compares against the backend twin, bit for bit (objective, gradient,
    `gn_quadratic`, `gn_hessian`), and against the ragged model's objective at the same point.
  - The other three tests are moved to the new form.
- **`TestFactoryValidation`:** each old mismatch is now a `ValueError` from `local_model`'s batch check: the data shape,
  `len(ww)`, `N_i`, the probe data length, the index rows, the jet axis.
- **`TestUniformBackendGeometryFromFrame`** (D5):
  - the frontend uniform manifold model makes one orthogonalization (a spy);
  - its geometry equals `from_point`'s (every field) on a minimal, a stacked and a shared point;
  - on a Tucker-slack and a TT-slack point `from_point` raises, the model works, and its objective, gradient norm and
    `pᵀHp` match the ragged model's. This is D5's test.
  - The regularized-numbers test compares against the backend twin, bit for bit.
- **`TestExactLocalModel`:**
  - every "matches the backend" test compares against `_pair`'s backend twin through one helper, `_check_actions`:
    the frame, objective, gradient, misfit gradient, `curvature_hessian`, `curvature_quadratic`, `quadratic`,
    `hessian`, `gn_quadratic`, bit for bit, with `K = ()` and `(2,)`;
  - the companion spies wrap only the model's actions;
  - internals are read through `model.backend` (`curvature`, `geom_aux`, `frame`);
  - `test_gradient_is_the_gauss_newton_gradient_bitwise` now spies on `LeastSquaresOracle.query`: one build query
    wanting `g`, and the gradient is read, not recomputed;
  - the curvature-damping message is the backend `local_model`'s.
- **`TestFiniteDifferenceHessianCheckInSafeMode`:** `model_at` builds on the derivative data.

**Other frontend test files**
- `tests/test_manifold.py`: one `local_model` call.
- `tests/test_sharing.py`:
  - a helper `_apply_model(geometry, x, ww, y)` replaces the eleven `apply_model` calls (the data `y` in place of the
    residual);
  - `geometry_aux` becomes `backend.geom_aux`;
  - the old-form optimizer calls stay (P17).
- `tests/test_jax_absent.py`: the script's T28 part builds the backend model with `bopt.local_model` on a
  `LeastSquaresOracle('probe', (1,))` and the Taylor data, and the frontend model with `fitting.local_model`. The
  old-form `topt.newton_cg` calls stay.
- `tests/test_chunk_size_estimator.py`: `TestFittingChunkSizeWiring` (`test_resolver`,
  `test_newton_cg_exact_across_chunk_size`) is removed, as the plan says, with the now unused imports.
- `tests/oracle_sweeps/sweep_models.py` and `sweep_exact.py`:
  - `build_model` / `build_exact_model` take the data and go through `F.old_vocabulary` + `fitting.local_model`;
  - the model sweep's Jacobian check converts the model's Taylor `J p` to derivatives (`taylor_to_derivatives`)
    before comparing with the oracle's derivative-convention `J p`. This was the only red: a convention change,
    0.5 = 1 − 1/2! at order 2.
  - `sweep_optimizers.py` (old form through the translation) is unchanged (P17).
- `tests/test_dispatch.py`:
  - a helper `_fmodel` replaces every factory call (twelve tests, the part-3 list's frontend rows and the ones 4a's
    removal of the factories turned red after part 3's measurement);
  - the two removed `least_squares_problem` / `uniform_least_squares_problem` uses become `F.Fit` / `F.uniform`;
  - a local `F = lambda` in `test_jit_shared_uniform_exact` is renamed `F4`; it shadowed `_fits` once the test used
    it.
  - `test_a_derived_kind_does_not_inherit_its_parents_cache_key` now tests an oracle subclass (a `HalfOracle` that
    halves `Jp_norm_sq`): unequal, a different hash, and the jit result is the subclass's.
  - **`test_jit_ragged_local_model_parameterized_kind`: changed assertion.** The weight is now a batch LEAF (plan
    P16: "a new weight ... does not recompile"). The old assertion was "different weights must not share a
    compilation".
    - Two weight values and the unweighted fit share ONE compile. Derivative data always carry
      `prepare_derivative_data`'s `t!` Taylor weight, so the tree structure is the same; measured: my first version
      expected the unweighted fit to compile apart, and it did not.
    - Each jitted result is checked against its own model's eager one, so a shared compile cannot return another
      model's numbers.

**`tests/test_optimizers_frontend.py`**
- `test_a_bad_batch_raises_before_any_work`: a bad `val_batch` raises before the setup (the reviewer's addition 2),
  and the backend spy is the optimizer called (`gradient_descent` was spying on `newton_cg`).
- **New `TestTheTranslation`:** the old form through the translation against numbers PINNED from the pre-P16 library
  on the same arrays (the 4a reviewer's `fe_sbs.py` cases and `old.pkl`), built from dense coefficients with no
  library code. Pinned: the last loss and the norm of the result, at rtol 1e-8; Adam at 1e-6 (its `eps`).
  - The cases:
    - every old weight form: probe per mode on noisy data (addition 1), a weight of length 1, extra rows; a bare
      vector on a derivative kind at `d == n + 1`; the matrix, a column, a row;
    - order 2 for apply and probe (the `t!` conversion), `entries_derivatives`, order 0;
    - positional `order` / `weight` with `hessian='exact'`;
    - `val_sample` / `val_data`: the verbose records pinned, and one without the other raising;
    - `mc_sgd` with a custom draw and a regularizer (the scale) and the positional `batch`;
    - the flat draw with `batch=` by keyword;
    - `adam` with a custom draw and a regularizer;
    - `chunk_size` dropped;
    - the uniform layer, Newton and `mc_sgd` with a custom draw.
  - `test_each_old_form_is_the_new_form` checks `_old_fit` against the new form by hand:
    - the jet shape and the lines;
    - the Taylor data and the weight times `t!`, exactly, for the bare, column, row and extra-rows forms;
    - entries derivatives as apply at the one-hot jet;
    - the plain probe's extra rows dropped;
    - the old structural errors;
    - a custom draw's batch carries `n / min(batch, n)`, with batch below, equal to and above `n`.
- **New `TestFrontendOnlyAndUserOracles`:**
  - The frontend-only fit: probe derivatives to order 2 with a (mode, order) weight, fitted from zero to a relative
    error below 1e-8 with `t3toolbox.fitting` and `t3toolbox.optimizers` only; the test asserts that its fit body's
    source names no backend module.
  - A user oracle without a draw (`take`) raises in `mc_sgd` and `adam` before any model is built, and runs with a
    draw.
  - An oracle without `f_resolution`:
    - an exact trust-region run with a large first radius, so trials are rejected and the floor is read, never asks
      for `f_resolution`;
    - every floor read equals `eps (|f| + |rho|)`, with `eps` of `f`'s dtype;
    - the run is the backend twin's.
- `TestFrontendUniformOptimizers.test_nonminimal_x0_handled_transparently`: the stale reference to
  `uniform_least_squares_problem` in its docstring is corrected.

### Pages and examples (the factory calls, which the translation does not cover)

- **`docs/fitting_and_optimization.md`:**
  - §2.2's two doctests build the model with `local_model` on an oracle and a batch, and the paragraph under them
    describes `local_model` in place of the six factories.
  - §4.6's doctest builds the weighted probe-derivative model with `prepare_derivative_data` (the ω matrix → the
    Taylor weight) and the plain probe one with a `(d,)` weight; its pasted output `(3, 4)` is unchanged.
  - No page output changed. The pages' old-form optimizer calls are untouched (P17).
- `docs/numerical_contracts.md`: the finite-difference example builds its models with `local_model` on the Taylor
  data (`prepare_derivative_data`); its output is unchanged.
- **Examples:**
  - `fit_hilbert_from_entries_lbfgs.py`: one oracle and batch, `local_model` per scipy evaluation.
  - `fit_hilbert_from_probes_adam.py` and `fit_hilbert_from_probes_optax.py`: `local_model` on the drawn batch (inside
    `jax.jit` for optax).
  - `fit_hilbert_tensor_newton_cg.py` and `fit_varied_rank_tensor_newton_cg.py`: `model_builder(X, b)` is
    `local_model` on the data, so the residual is no longer computed by hand.
- `dev/graded_messages/implementation/bench/p16_part4_frontend_side_by_side.py`: the 4a reviewer's `fe_sbs.py`, kept
  with the slice (README updated). The translation's cases against the pre-P16 library live there and in
  `TestTheTranslation`.

## 2. The tests of the tests

Mutants run in a scratch copy (since deleted) against `TestTheTranslation`. All are killed:
- the custom draw's scale dropped;
- the `t!` weight conversion skipped;
- the plain probe weight replaced by its first entry;
- the probe derivative weight's rows replaced by the first;
- the validation batch built from the training sample;
- `n / min(batch, n)` replaced by `n / batch`. This one survived at first, with every case at `batch < n`; it is now
  killed by the batch-above-`n` check.

4a's seven mutants were run against `TestReview4aChecks` (4a's response).

## 3. The gate

The full gate at `-n 2`
(`run_gate.sh --detach -n 2 -l scratchpad/implementer4/gate_4b`, the six steps one after another):

| step | result | counts | seconds | lowest MemAvailable (MB) |
| --- | --- | --- | --- | --- |
| suite | FAIL (1, fixed below) | 1 failed, 1896 passed, 5 skipped, 1 xfailed, 90942 subtests | 2327 | 4021 |
| doctest-modules | PASS | 321 passed | 53 | 9902 |
| doctest-quickstart | PASS | 1 file, 0 failed | 3 | 11576 |
| doctest-pages | PASS | 47 files, 0 failed | 19 | 11514 |
| examples | PASS | 14 examples, 0 failed | 567 | 11199 |
| sphinx | PASS | 0 warning/error lines | 1012 | 11106 |

**The one failure, new and mine, now fixed:**
`TestReview4aChecks::test_a_float32_frame_the_library_built_passes_and_a_perturbed_one_raises`.
- In the full suite another module (`test_dispatch`) turns on jax x64 at import, so `x.to_jax()` was float64.
- The test now builds float32 jax cores explicitly. It passes with x64 on, in a fresh process, and alone.
- Its mutant (`is_orthogonal()` at the default tolerance) is still killed: the residual is above 1e-9, which the test
  asserts.

**Against part 3's expected-red list (90 ids):**
- Every id that exists is green; the suite's only red was the test above.
- Eleven ids no longer exist:
  - `TestFittingChunkSizeWiring::test_newton_cg_exact_across_chunk_size`, removed by the plan;
  - the nine `TestLeastSquaresProblemAndKeywords` ids and `TestFrontendSwitchNewton::test_least_squares_problem_sizes_switch_as_exact`,
    whose tests 4a moved to `TestFrontendHeadsAndKeywords` with the removal of `least_squares_problem`;
  - `TestFrontendExactNewton::test_chunk_size_auto_is_sized_for_the_exact_mode`, removed in 4a with `chunk_size`.
- The laptop baseline has no failure, and nothing else is red.

Earlier by file, at `-n 2`, one at a time:
- `test_fitting` + `test_optimizers_frontend` + `test_aligned_padded_frame_frontend` + `test_manifold` +
  `test_jax_absent`: 181 passed, 561 subtests;
- `test_sharing`: 78 passed;
- `test_oracle_sweep`: 5 passed, 2 skipped (the slow tier). The model sweep has 992 cases and 0 failures; the exact
  sweep 720 and 0.
- `test_dispatch`: 107 passed after one fix (above).

**The slow tier, serially** (`run_gate.sh --detach -l scratchpad/implementer4/gate_4b_slow slow`):
- PASS: 66 passed, 13173 subtests, 1438 s, lowest MemAvailable 11502 MB.
- It covers the full optimizer matrix through the translation, and the exact sweep's seeds, layouts and order 3.

Lowest MemAvailable over the whole gate: **4021 MB** (the suite at `-n 2`).

Re-run after the float32 fix: that test alone (with x64 on and off), and `test_optimizers_frontend` by
`-k float32`. The full suite was not re-run after this one-test change.

## 4. Departures, choices

- **The changed compile assertion in `test_dispatch`** (above): the plan's "a new weight does not recompile", now
  that the weight is a batch leaf; jit is checked against eager per model.
- **The old kinds' own tests are deleted** (decision B). `TestBlockSumsq`, and the old-kind halves of
  `TestResidualWeighting` and `TestPerModeWeightRowCount`, tested code nothing calls.
- **Prose that names the removed factories is left for P19,** as the plan says: `docs/api_reference.rst`,
  `docs/glossary.md`, `docs/user_guide.rst`, the contributor pages, `docs/fitting_and_optimization.md` §4.6's
  paragraph above its doctest, and the comments of `examples/fit_per_mode_weight_probes.py` and
  `examples/fit_hilbert_from_apply_derivatives.py`, which are P17's.
- **For P17 (decision C):**
  - `tests/_fits.old_factory`, `old_model` and `old_vocabulary`, and `tests/test_dispatch.py`'s `_fmodel` and
    `tests/test_fitting.py`'s `_model` / `_pair`, keep the 2026.5.0 vocabulary (kind names with `_derivatives`,
    derivative data) as a test convenience. P17's search for the old call form will not find them.
  - Convert or rename them there, with the old-form optimizer calls of `test_optimizers_frontend`, `test_sharing`,
    `test_jax_absent`, `test_dispatch`, `oracle_sweeps/sweep_optimizers.py` and the pages.
  - `TestTheTranslation` and the side-by-side script go with the translation.
- **For the CHANGELOG (decision D):** a stacked local model now has one `y` for all its elements. The 2026.5.0
  factories took a residual per element.

## 5. Found outside the slice

- Stale docstrings that name `uniform_least_squares_problem` in `tests/backend/test_uniform_curvature.py` (lines 26,
  1250, 1267, 2057, 2173) and `tests/backend/test_uniform_fitting.py` (641). They are part 2's files; the text only.
