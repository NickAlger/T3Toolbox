# P16 part 2, response to review round 1

Implementer, 2026-11-21. Review: `P16_part2_REVIEW_r1.md` (NOT YET, on D9 only). Ruling: D9 is option (d). Nothing is
committed.

## D9: built as ruled

`t3toolbox/backend/optimizers.py`:
- **`_point_objective_parts(geom, oracle, batch, x_cores, regularizer=None) -> (f, rho)`** (private). `point_objective`
  is their sum. **`_scored(...)`** returns the host floats `(misfit, regularization, objective)`.
- **`newton_cg`** carries `scored`, the point-route objective of exactly the current `x`:
  - **At the start**, one `_scored` at `x0`, in row 0's `reference` lap. That covers every start (`x0`, `padded_from`,
    `start_frame`), once per call.
  - **The trials** are `_scored` (policy a, as before).
  - **On acceptance**, `scored = trial_scored`. A rejected trust-region trial keeps `x` and `scored`. Under `'accept'`
    after an exhausted search, the last trial's value is carried. The `-g` fallback is a trial like any other. The
    switch's exact rebuild happens at the same `x`.
  - **The history's** `objective`, `misfit`, `regularization` and `losses` are `scored`. The model's frame-route `f`
    stays `LocalModel.misfit`, which the floor's `eps` and the display read.
- **The invariant** is written as a comment at the carry: any future step that changes `x` between iterations must
  recompute it.
- **`gradient_descent`** carries the same way: `point_objective(x0)` once, then the accepted (or `'accept'`-ed) trial's
  value.
- **Docs.** `newton_cg`'s stats docstring and `NewtonInfo.objective` say which value is reported. One correction to
  the ruling's wording: on an accepted row the next objective IS the trial's value (bitwise), and
  `objective + delta_f` equals it to the rounding of `delta_f = f_new - f`. It is exact only when Sterbenz applies
  (the two values within a factor of 2). The tests check the chain to `1e-15` relative, and the per-row values bitwise.

## The tests that relied on the mixed routes

Each one now has a mechanism that does not depend on which route reads the current point's value.
- **`_OffsetOracle`** (shrink, floor (d)) raises `f` at every corewise point other than the run's start. The start is
  fingerprinted as the sum of squares of its cores (`_point_key`), passed through `_offset(fit, offset, x0)`. While the
  run stays at its start, every trial is an ascent on either route. M20 (the shrink a quarter of the radius) still
  kills the shrink test.
- **`_objective_stub_at(n, change)`** (the switch's reset runs) now patches `_point_objective_parts` and counts trials
  only (the start's value is not a trial). It is the same scripted ascent of row `n - 1`'s trial as before.
- **The switch's `test_stats_and_rows` rejection** used the 0.05·f0 offset oracle. Under (d) that offset is
  rejection-neutral once the run leaves its start, so it now uses `_objective_stub_at(3, f + f0)`, which rejects row 2's
  trial. The assertions are unchanged: after a rejection the next row has the same `x_cores` and the same `lm`, which is
  not exact.
- **`test_a_trial_is_scored_with_nothing_held`** records `_point_objective_parts`. It also asserts that the first scored
  point is `x0` itself.
- **The timing attribution test** traces `_point_objective_parts` and expects it in `{reference, trial_objective}`.
- **`check_bookkeeping`** (every trust-region test) now checks the carried chain.
- **New: `test_the_acceptance_compares_like_with_like`.** It is described in the handover, section 2.
- **`test_dispatch`'s float32 floor.** It now passes its `(1, 0)` assertion, but its hand scale `y·sqrt(2·misfit)` read
  the reported misfit, which is now the point route's. At that floor the two float32 misfits are 3.5e-9 (point) and
  2.0e-7 (model). The floor sizes `|r|` from the model's residual (`f_resolution` with `held`). So the scale now reads
  `seen[0].lm.misfit` (a callback). The bound and the assertion are unchanged.

## Runs (`-n 2`, one at a time)

| File | Result |
| --- | --- |
| `test_optimizers` | 76 passed, 1 skipped |
| `test_zero_reference` | 17 passed |
| `test_padded_start` | 37 passed |
| `test_uniform_fitting` | 30 passed |
| `test_curvature` | 96 passed |
| `test_uniform_curvature` | 57 passed, 2 skipped |
| `test_geometry`, `test_import_graph`, `test_least_squares_oracle` | 55 passed |
| `test_dispatch` backend rows | 21 passed, 1 failed (the float32 floor, before its fix above); that test alone after the fix: passed |
| Module doctest of `optimizers.py` | passes |

- **Side by side:** the 36 cases, re-run, give the same table as round 0 (35 SAME, the same radius-tie DIFF).
- **Mutation:** M16 to M20 were added for D9 and the whole pass of 20 was re-run; all are killed. M19's first form
  (`f = None` after a step) was an equivalent mutant, since it re-reads `point_objective(x)`, the same value. It was
  replaced by a real mixed route (`f = lm.objective`). M2 and M14 were re-targeted to the new code.
- **Expected-red list:** 103 ids. The two D9 ids are removed and the header is updated. The frontend and display reds
  are untouched by a backend-only change, so the full suite was not re-run.

## The reviewer's minor points

- `_alignment_projector`'s unbounded `lru_cache` now has a comment: one entry per (geometry, oracle), so one per rank
  level, which is negligible.
- **The default-radius tie is noted for the backlog** (handover section 6). Row 0's CG step equals the default radius to
  the last bit. A relative tolerance on the boundary test would remove it.
