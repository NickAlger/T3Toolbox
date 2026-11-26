# P4, response to review round 1

To `P4_REVIEW_r1.md` and the orchestrator's rulings on it, 2026-11-17. Every finding 1 to 9 is accepted. The
public surface (names, signatures, formulas) is unchanged; the changes are inside `backend/fv_operations.py` and
`backend/ufv_operations.py`, in docstrings, and in the tests. `P4_HANDOVER.md` is brought up to date.

## Findings

**1. Overflow of the norm (should fix). Accepted, fixed.**
- `_tt_left_orthogonal_norm` divides the last core by its largest magnitude before squaring and multiplies it
  back, so a norm overflows only where it is not representable in the dtype.
- `_relative_residual(num, den)` returns `nan` wherever either norm is not finite, so `num / inf` can no longer
  read 0.
- Measured, float32 numpy, `d = 5`: at norm `4e20` and `1.2e37` a consistent frame reads `2.5e-7` to `3.0e-7`
  (centers `3.1e-7` to `3.4e-7`); `R[0]` off by `1e-3` reads `9.99e-4`; `Vc[1]` off by `1e-3` reads `1.000e-3`.
  jax float32 at norm `1e20`, eager and under `jit`: `9e-7` consistent, `9.99e-4` with `R[0]` off.
- Tests: the class `TestNormsThatDoNotOverflow` (4 tests): consistent float32 frames at norms near `1e20` and
  `1e37` pass, with a residual above 0; `R[0]`, `Vc[1]`, `Hc[2]` off by `1e-3` read `1e-3`; a stack of three whose
  middle element has a norm beyond the largest float32 (finite cores) reads `nan` there and keeps the others; the
  uniform twins at norm `1e20`. `test_a_norm_that_is_not_finite_gives_nan_and_never_zero` tests
  `_relative_residual` by itself. A row in `test_dispatch.py` (`R[0]` off by `1e-3` at norm near `1e21`, eager and
  under `jit`).
- The docstring now says where the residual is `nan` and that it is never 0.

**2. The `1` of `max(1, ||T||)`. Accepted.** `test_below_norm_one_the_residual_is_absolute`: hand-made frames with
every core scaled down so that both tensors have norm below 0.5 (asserted), over the six structures and three
stacks; the residual equals the norm of the dense difference itself, and is asserted to be at least twice away
from the ratio. The mutation `max(1e-3, den)` now fails 21 cases; `num / den` fails 22.

**3. The guard's test must fail, not hang. Accepted, both ways.**
- `test_the_guard_zeroes_the_stack_elements_that_are_not_finite` tests `_zero_stack_elements_that_are_not_finite`
  by itself, with no SVD in reach.
- In the test that goes through the checkers, the two functions that lead to an SVD (`_tt_left_orthogonal_form`,
  `_tucker_down_factors`) are wrapped by a mock that raises `AssertionError` on any core that is not finite. A
  checker that does not call the guard, or a guard that does not zero, now fails in about two seconds.
- Checked by mutation: the zeroing removed, the flag always true, the flag over one family only: each caught as a
  failure, none as a hang.

**4. Cost. Accepted; `D` is now shared.**
- The docstrings state the cost as counts (one SVD of each Tucker core, linear in the mode size; three left
  orthogonalizations of a TT, one at doubled ranks; the centers' check growing like `d**2`), with no seconds and
  without "whatever the mode sizes". `T3Frame.consistency_residual` says it costs several times the frame's
  construction.
- Sharing `D` was simple, so it is done, and more follows from it. Each Tucker core is factored once,
  `D[i] = R[i]^T Q[i]` with `Q[i]` of orthonormal rows; `R[i]` is absorbed into the TT cores above; since `Q` is
  orthonormal the whole computation is then on TTs (bonds doubled in the difference, the Tucker edge not). For the
  centers, `Vc[i]` and `D[i]` are factored together at site `i` so that both terms stand on one basis there.
- One thing had to be got right in that: `Vc[i]` has the scale of the tensor and `D[i]` scale one, so factoring
  them stacked as they stand brings back the accuracy loss of the handover's 4.1 (a mutation that removes the
  balancing fails 7 tests). Each is divided by its largest magnitude first and the scale is given back after.
- Accuracy after the change (float64, 20 seeds, norms `4e-2` to `4e14`): frame at most `1.7e-14`, centers at most
  `1.9e-14`; against the dense definition as before. Time at `d = 8`, `N = 2000`, `n = 60`, `r = 40`: 0.36 s
  where the frame's construction takes 0.16 s (the reviewer measured 0.90 to 0.98 s on the round-0 code).

**5. `nan` or `inf` in uniform padding. I made the code match the docstring.** It was cheap and it is sound: the
padding is don't-care by the equivalence contract. `_zero_the_padding` selects the real slots with `where`, using
the indicator the library's masking function gives on ones, so the library's masking itself is not touched and
the masks stay the library's. `test_a_nan_or_an_inf_in_the_padding_changes_nothing`: every padded slot of the
frame and of the centers set to `nan`, `inf` or `-inf`; each residual is the clean one bit for bit. A mutation
back to multiplication fails 6 cases.

**6. The `LinAlgError` branch. Kept, narrowed, and the backends now agree.**
- It is reachable after the guard: finite cores whose products overflow inside the sweep (every core of an
  element at `1e200`) still make numpy's SVD raise. So it stays.
- On that failure numpy now recomputes one stack element at a time (`_per_stack_element_if_the_svd_fails`): the
  elements that raise get `nan`, the others their residual, which is what jax gives by itself. Unstacked: `nan`.
- It takes `LinAlgError` only; any other error passes through (tested at both levels of the helper, and by two
  mutations).
- What remains as the reviewer recorded it: an SVD that does not converge on finite input is also `nan`, which
  reads as "inconsistent". The docstring says "where the orthogonalization overflows or fails".

**7. numpy 1.x promotion. Made explicit; not verified on numpy 1.x.** No Python float meets an array any more:
`xnp.maximum(den, xnp.ones_like(den))`, `xnp.where(..., residual, xnp.full_like(residual, xnp.nan))`, and
`xnp.ones_like` / `xnp.zeros_like` elsewhere. An unstacked float32 frame is now tested for a float32 0-d residual
(both residuals), which passes here on numpy 2.5; CI's floor leg will run it on numpy 1.x.

**8. Wall-clock assertions. Removed** from the four tests. The tests still check `d = 12`, `N = 100` and get the
right numbers.

**9. Duplicated helpers. Kept, each with a comment naming its twin**, under one comment block that says why
(`t3_linalg` and `geometry` reach `fv_operations` through `ranks` and `sharing`):
`_tucker_down_factors` (twin: `t3_orthogonalization._down_orthogonalize_tucker_cores_step`),
`_tt_left_orthogonal_norm` (twin: `geometry.t3_left_orthogonal_norm_sq`), `_tt_core_direct_sum` (twin: the TT
cores of `t3_linalg.t3_add`). `fv_operations` now imports `tt_orthogonalization` in place of
`t3_orthogonalization` (no cycle; the acyclicity test passes).

## What changed in the code, in one place

Private helpers of `fv_operations` now: `_tucker_down_factors`, `_tt_absorb_down_factors`,
`_tt_left_orthogonal_form`, `_largest_magnitude`, `_tt_left_orthogonal_norm`, `_tt_core_direct_sum`,
`_tt_difference_norm`, `_zero_stack_elements_that_are_not_finite`, `_relative_residual`,
`_per_stack_element_if_the_svd_fails`, `_frame_residual_of`, `_centers_residual_of`. The round-0 helpers
`_t3_left_orthogonal_form`, `_t3_left_orthogonal_norm`, `_t3_core_direct_sum`, `_t3_difference_norm` are gone.
In `ufv_operations`: `_zero_the_padding` added beside `_masked_frame_as_ragged`.

## Tests

`tests/backend/test_frame_consistency.py`: 39 tests (30 before). Nine new: the four of
`TestNormsThatDoNotOverflow`, `test_below_norm_one_the_residual_is_absolute`,
`test_a_nan_or_an_inf_in_the_padding_changes_nothing`,
`test_the_guard_zeroes_the_stack_elements_that_are_not_finite`,
`test_the_failure_path_takes_one_stack_element_at_a_time`,
`test_a_norm_that_is_not_finite_gives_nan_and_never_zero`. Changed: the float32 test also takes an unstacked
frame; the non-finite test runs under the mock; the overflow test asserts per-element `nan`; no clocks.

Mutations in a scratch copy on the final code: 26, all caught as failures, none as a hang. Among them: the norm
by a plain sum of squares; `nan` not set where a norm is not finite (either norm); the floor `1e-3`; no floor;
the guard broken three ways; the pre-orthogonalization removed; the balancing of the stacked factor removed; the
scale not given back; the failure path removed, made all-`nan`, or made to swallow every error (each level);
padding zeroed by multiplication.

## The gate

```
step                result counts                                                      seconds   peak_MB  minavl_MB
quick               PASS   166 passed, 619 subtests passed                                 4.0       548       7240

suite               PASS   1631 passed, 5 skipped, 1 xfailed, 1 warning, 66233 subtests passed    925.9      3915       3964
doctest-modules     PASS   310 passed                                                     25.1      1359       6328
doctest-quickstart  PASS   1 file(s), 0 with failures, 0 failed example(s)                 1.8       230       7394
doctest-pages       PASS   48 file(s), 0 with failures, 0 failed example(s)               12.2       447       7215
examples            PASS   14 examples, 0 failed                                           186       361       7304
sphinx              PASS   0 warning/error line(s)                                       193.2       728       6804
```

The slice's tests first (`run_gate.sh --detach -n 2 -l /tmp/gate_P4_r1_quick quick` on the new file, the two frame
test files, `test_import_graph.py`, `test_jax_absent.py`, `tests/backend/test_fv_operations.py`; the two edited
tests of `test_dispatch.py` were run by hand, 2 passed), then the whole gate
(`run_gate.sh --detach -n 2 -l /tmp/gate_P4_r1`), on the final tree: no file of the library or the tests changed
after it started. 1,631 passed = 1,582 before the slice + 49 (39 + 5 + 4 + 1). 5 skipped and 1 xfailed as before;
the one warning is the baseline's (adam on a manifold geometry). No failure.
