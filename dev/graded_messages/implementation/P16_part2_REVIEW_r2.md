SIGN OFF: D9 is built as ruled and matches my prototype bitwise. The rewritten tests still bite: my four mutants are all killed. The floor's frame-route resolution is the right scale.

# P16 part 2 review, round 2

Reviewer, 2026-11-21, on `P16_part2_RESPONSE_r1.md`, the updated handover and the diff against round 1. Runs were one at
a time, at `-n 2`. The mutant copy was deleted. The working tree is untouched apart from this file.

## The build of D9

- **`_point_objective_parts` and `_scored`.**
  - `point_objective` is the sum of its two parts.
  - The start is scored once, in the `reference` block of row 0, which is entered once per call.
  - A trial is scored by `_scored`.
  - On acceptance, `scored = trial_scored`.
  - A rejected trust-region trial keeps `x` and `scored`.
  - `'accept'` after an exhausted search carries the last trial.
  - `gradient_descent` carries the same way.
  - The history's `objective`, `misfit` and `regularization` are the carried point-route terms. The model's
    frame-route `f` stays `LocalModel.misfit`.
  - The invariant is written at the carry.
- **The implementer's correction holds.** The next row's objective IS the trial's value, bitwise. `objective +
  delta_f` equals it only to the rounding of the subtraction, so the tolerance of `1e-15` relative in the chain test is
  right.
- **My prototype's cases against the build.** On the 36 side-by-side cases, the build and my prototype of (d) agree
  **bitwise**: the final points differ by 0.0, and so do the cone and start movements. Against the pre-P16 library the
  build gives round 1's table again: 35 SAME, plus the known radius tie.

## The rewritten tests still bite

**The baseline passes.** `test_optimizers`, the float32 floor test of `test_dispatch`, `test_padded_start` and
`test_zero_reference`: 131 passed, 1 skipped.

**Four mutants of my own,** each run against `test_optimizers` and the float32 floor test:

| Mutant | Result |
| --- | --- |
| no carry on acceptance (the start's value kept for every row) | killed, 37 failures |
| `gradient_descent` scoring its current point by the frame route each iteration | killed, 3 (the exhaustion test among them) |
| the trust-region radius floor that never fires | killed, 2 (the `_OffsetOracle` floor (d) tests) |
| the switch's countable row without its full-step condition | killed, 6 (the reset, count and record tests on the scripted stubs) |

**The mechanisms that replaced the route mix are route-independent.**
- `_OffsetOracle` fingerprints the start point, so the start is unshifted on either route and every trial is shifted.
- `_objective_stub_at` scripts the n-th trial through `_point_objective_parts`.

**`test_the_acceptance_compares_like_with_like`** covers trust region, line search and `'accept'`, regularized, ragged
and uniform, manifold and corewise. It checks:
- every row's misfit and regularization **bitwise** against `_point_objective_parts` at that row's own `x_cores`;
- row 0 bitwise against the start's value;
- the chain to 1e-15.

So any frame-route value reported, or any stale carry, fails it.

## The float32 floor: is the frame-route `f_resolution` the right scale for point-route values?

Yes.

**The noise that matters.** The floor must bound the rounding noise of the compared values, `f_new - f`, both on the
point route. Each is `1/2 |omega r|^2` with `r = z - y`. The computed `z` carries an absolute error of order
`eps |omega y|`, so either route's `f` is off by about `eps (|omega y| |omega r| + |f|)` (plus `eps^2 |omega y|^2`).
That is the oracle's `f_resolution` formula, and it depends on the route only through `|r|`.

**What the route changes.** Both routes give `|r_true| + O(eps |omega y|)`. They can differ by more than a small
relative amount only when `|r|` is itself at the rounding level: the implementer's float32 misfits 3.5e-9 against
2.0e-7, a factor of about 7.5 in `|r|`.
- There, every term of the bound is `O(eps^2 |omega y|^2)`.
- The O(1) factor between the routes sits inside the floor's own margin, `_TR_RESOLUTION_FACTOR = 10`, and the
  switch's 1000.
- When the frame route's `|r|` is the larger, the floor is the more conservative. It stops at
  `'objective_floor'`, which is not a failure, slightly earlier, rather than chasing noise.

**The test's hand scale** reads `lm.misfit`, the quantity the floor reads. That is the right check: the test pins the
code's formula and its `(1, 0)` outcome. The values compared are still asserted on the point route by the like-with-like
test.

**A note worth keeping** in `_objective_resolution`'s docstring, one sentence:
- the resolution is a scale estimate;
- either route's `|r|` serves;
- they differ only at the floor, by a factor inside the margin.

This is optional, not a finding.

## Expected red

The two D9 ids are removed (103 ids). The change touches the backend only. The frontend and display reds are
untouched, as the response says: my round-1 measurement found them red for reasons outside the step acceptance. I did
not re-run the full suite.
