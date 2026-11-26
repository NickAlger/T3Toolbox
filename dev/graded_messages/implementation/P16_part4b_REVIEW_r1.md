SIGN OFF

# P16 part 4b, review round 1

Reviewer (the third of P16), 2026-11-21. Reviewed: the working tree on `p16-wip` (4b, uncommitted, on top of
`c6b62ba0` and the `dev/` commits after it), against `P16_part4b_HANDOVER.md`, the plan's P16 and Claude's rulings.
Read only, apart from the full-suite run below.

**Verdict.** 4b does what part 4 left to do, and its tests compare with independent constructions. I ran the full suite
once on the final tree, and it is green. With the implementer's other five gate steps and its serial slow tier, that
completes P16's gate. Nothing blocks and nothing needs fixing; three minor notes follow.

## The gate

**The full suite, at `-n 2`, on the final tree** (my run, the only test run at the time):
`1897 passed, 5 skipped, 1 xfailed, 1 warning, 90942 subtests passed in 2359.44s (0:39:19)`, exit 0. MemAvailable
afterwards was 11.6 GB.
- That is the implementer's count (1896 passed and 1 failed), with its one failure fixed: the float32 frame test
  depended on test order, because `test_dispatch` turns on x64.
- So nothing is red, and each id of part 3's list that still exists is green.

**The other steps**, from the handover's gate log:
- module doctests: 321 passed;
- the quickstart;
- 47 pages;
- 14 examples;
- Sphinx with 0 warnings;
- the slow tier, serially: 66 passed, 13173 subtests.

## What I checked

1. **`TestTheTranslation` against the pre-P16 library.** I read the 19 pinned `(last loss, result norm)` pairs and the
   `J_TRAIN` / `J_VAL` records out of the test and compared them with my own pre-P16 results (`old.pkl`, from
   `reviewer2/old_lib`). All equal exactly, relative difference 0.0. The pins are the pre-P16 library's numbers, not the
   new tree's, so the test is not circular.
   - Both of my additions are in: a per-mode probe weight on noisy data, and a bad `val_batch`.
   - Two of my mutants against `TestTheTranslation` alone are killed: the per-mode weight replaced by its first entry
     (3 tests fail), and the custom draw's scale dropped (4 tests fail).
   - `test_each_old_form_is_the_new_form` checks the batch's weight against `full * t!` exactly. It also checks the
     draw's scale with the batch below, equal to and above `n` (the last kills `n / batch`).
2. **`test_jit_ragged_local_model_parameterized_kind`: can the shared compile hide a wrong number? No.**
   - The three models (`w`, `2w`, and none, which gives the `t!` default) are built from the same seed, so `x`, `y` and
     the trial `p` are the same in all three. (`p` comes from `MANIFOLD.randn` right after `build` reseeds.)
   - Only the weight differs, and the Gauss-Newton product is quadratic in it: `2w` gives four times `w`'s product, and
     the default gives another.
   - So a compile that baked in the first model's weight would return `w`'s numbers for the second and third models.
     Each jitted result is compared with its own model's eager result, so that would fail.
   - The eager path's own numbers are tested elsewhere: the weighting tests of `test_fitting`, against the dense
     objective.
3. **`test_fitting` is judged against independent constructions.**
   - **Derivative fits:** the objective and `evaluate(t p)` at `t = 1, 2` are compared with
     `tests/oracle_sweeps/oracle.py`, which uses exact polynomial interpolation on the dense tensor and no library jet
     code, at `x + t dense(Π p)`.
   - **Plain kinds:** the dense einsum forwards.
   - **The backend twins:** `_backend_geometry` and `_backend_model` are built in the test, from `from_point`,
     `with_sharing` and `pack_batch`, not through `fitting.backend_geometry` or `prepare_batch`.
   - **Stacked models:** `TestLocalModel._setup` now draws one `y` without `C`, and takes each element's residual from
     the dense forward.
4. **The frontend-only fit** is judged by recovering the target tensor to 1e-8, which is ground truth. Its body names no
   backend module (an assertion on its source). The user-oracle tests check two things:
   - a missing `take` raises before any `bopt.local_model`;
   - with no `f_resolution`, every floor read equals `eps (|f| + |rho|)`, with `eps` of `f`'s dtype, and the run is the
     backend twin's.
5. **The conversions keep their meaning.**
   - `test_sharing`'s `apply_model(geom, x, ww, r)` with `r = x.apply(ww) - b` became `_apply_model(geom, x, ww, b)`,
     the data and not the residual. The same holds at every converted call site I read.
   - `sweep_models`' Jacobian check converts the model's Taylor `J p` with `taylor_to_derivatives` before comparing
     with the oracle's derivative-convention `J p`. That is the right direction.
6. **Pages and examples.**
   - The doc diff changes prose and doctest inputs only; no pasted output line changed.
   - The examples move to `local_model` on an oracle and a batch. The residual is no longer formed by hand; the model
     computes it from the data. All 14 pass in the gate.
7. **The side-by-side script** is in `bench/` (`p16_part4_frontend_side_by_side.py`), with the README updated.

## Minor notes (none blocks)

1. **`test_dispatch`, the weight compile test.**
   - What the check rests on: the three models' eager Gauss-Newton products differ. That holds by construction (point
     2), but the test does not assert it.
   - What would make that explicit: assert, for example, that the `2w` product is four times the `w` product. That
     would also pin that the weight reaches the jitted product.
   - Optional.
2. **Stale docstrings that name `uniform_least_squares_problem`.** In `tests/backend/test_uniform_curvature.py` and
   `test_uniform_fitting.py` (the handover's §5). Text only; P19's sweep or the next touch of those files.
3. **For P17 and the CHANGELOG, as the handover records.** The test-only 2026.5.0 vocabulary (`_fits.old_vocabulary`,
   `old_model`, `old_factory`, `test_dispatch._fmodel`, `test_fitting._model` / `_pair`) is invisible to P17's search
   for the old call form, so list it there. A stacked local model now takes one `y` for all its elements (decision D),
   which the CHANGELOG should state.

The mutation copy is deleted. Nothing is running. The working tree is as the implementer left it, plus this file.
