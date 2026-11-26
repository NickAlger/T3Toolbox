SIGN OFF

# M2 part 5 review, round 1: the frontend, the old forms, and M2's gate

Reviewer (the sub-arc's second), 2026-11-23. Reviewed: the uncommitted part 5 on `m2-wip` (over `fd29eae1`), together
with `M2_part5_HANDOVER.md`. My scripts are in `scratchpad/m2rev2/`:
- `translation_runs.py`, with its records `tr_old.npz` and `tr_new.npz`;
- `stacked_routes.py`;
- `base/`, my own `git archive` of `85c6e54d`, identical to `m2impl/base` (`diff -rq`);
- `bat_old.npz` and `bat_new.npz`, my own battery records.

**Verdict: SIGN OFF.**
- The translation reproduces the pre-M2 library on every old weight form.
- The stacked-`y` routes are right against dense values.
- I re-recorded the battery myself on both trees, and it reproduces bitwise. It is AS EXPECTED.
- The tally and the red trail check out, and the diff audit's flagged lines are what the handover says.
- One minor point corrects the battery's gauge explanation. It does not change the verdict.

## 1. The translation against the pre-M2 library (`translation_runs.py`)

**The setup.** Old-form `newton_cg` (2 steps) and `mc_sgd` (3 steps, a seeded draw) from a random point, on the
pre-M2 base and on this tree, with every old weight form:
- plain probe: `(d,)`, an over-long `(d+1,)` (extra rows dropped), none;
- `probe_derivatives` at order 2: `(d, n+1)`, a bare `(n+1,)`, `(1, n+1)`, `(d, 1)`, none;
- `apply_derivatives`: `(n+1,)`, `(1, n+1)`, `(1, 1)`, none;
- `entries_derivatives` at order 1: `(n+1,)`, none.

**The results, over 196 arrays.**
- 163 bitwise, including every Newton core and every unweighted run.
- Every loss to `≤ 1.4e-15`.
- Four weighted `mc_sgd` runs end with cores that differ by signs (whole cores, or rows of them). Their represented
  tensors agree to `≤ 1.1e-14`, and every loss to `1.4e-15`. See minor 1.

So placing the old blocks with `blocks_to_weight` reproduces the old objective. For the derivative kinds the blocks go
on the derivative data before `prepare_derivative_data`, so the `k!` is applied once, as before. The rounding-level
differences come only from `r·(ω²r)` against `ω²(r·r)`, as ruled.

I read the diff of `_old_fit`. It is the pre-M2 canonicalization and broadcast to blocks, unchanged, then
`blocks_to_weight` on the (stack-shared) data. `_old_draw` and the validation batch go through the same function.

## 2. The stacked-`y` routes (`stacked_routes.py`)

Frontend `fitting.local_model` on a stacked point with `C = (2,)`, apply and probe, ragged and uniform. The objective
per element is compared both with the unstacked model at that element's data and with my dense
`½ Σ ω² (z − y)²` (einsum of `to_dense()`). All agree to 1e-12, for two kinds of data:
- a shared `y` with size-1 `C` axes;
- a per-element `y` with a weight full in `C`.

The refusals:
- A `y` without `C` raises a `ValueError` at `prepare_batch → check_batch`. Before M2 it was right-aligned.
- The translation on a stacked `x0` passes the batch check and then raises the optimizers' `NotImplementedError`.

## 3. The battery

**My own records.** I recorded both forms myself with the committed `battery.py`: `old` on my base archive, `new` on
this tree. Both are bytewise identical to the implementer's `battery_old5.npz` and `battery_new5.npz` (25,476 keys
each, 0 differences). `--compare` on my records gives **AS EXPECTED**:
- 24,427 bitwise;
- 956 to rounding, at most `3.59e-16`;
- 39 jax, at most `1.01e-16`;
- 11 exact optimizer counts;
- 43 optimizer arrays, at most `1.93e-16`.

**The re-recorded base.** `battery_old5.npz` equals part 1's `battery_old.npz` bytewise in every array except the 20 of
the two `mc_sgd` runs (`opt/mc_sgd/ragged/{new,old}`). The changed runs restore the global numpy state after their
draw, and nothing else moved.

**The gauge explanation: right in kind, too narrow in cause (minor 1).** The original mismatch is a gauge
difference, not a numerical one: the losses, counts and stop reasons agree, and the tensors agree to `4e-15`. But my
runs in §1 show that it is not specific to a start at the zero tensor:
- four of the eight derivative-weighted `mc_sgd` runs flip the signs of singular vectors in their cores from a
  generic random start;
- all eight Newton runs, by contrast, are bitwise in the cores.

The cores' sign gauge after an `mc_sgd` retraction can turn on a rounding-level change of the step (here the Cauchy
step's `Jp_norm_sq`), from any start. The re-seeded battery run happens not to flip. That is allowed under the
knife-edge ruling ("choose another run"), and the gate result stands. But the battery's comment and the handover's
§4 should not say the cause is the zero start's degenerate spectrum.

## 4. The tally, the red trail and the gate

**The log.**
- `suite5_v.log` ran from about 22:48 to 23:28. The last changed file, `tests/test_dispatch.py`, is from 22:43.
- 1,954 passed, 0 failed; no `FAILED`, `SUBFAILED` or `ERROR` lines.

**My own tally against part 1's log.** 1,934 nodes there, 1,960 here; 33 differ:
- 29 new: 12 in `test_engine_data_space`, 11 in `test_oracle_data_space`, part 1's refusal test, and part 3's 5 in
  `test_jets` and `test_data_preparation`;
- 3 renamed or replaced in part 3;
- 1 with more subtests: the factorials by hand, 11 → 58.

Every other node has exactly part 1's outcome and count. The red trail 194 → 123 → 23 → 0 is consistent.

**Checks I reran:**
- the page `docs/fitting_and_optimization.md` (`python -m doctest`): passes;
- `backend/fitting.py`'s doctests under plain `python -m doctest`: pass (part 4's minor 1 is fixed);
- `test_oracle_data_space`, `TestTheTranslation` and `TestLocalModel`: 31 passed, 366 subtests.

I did not rerun Sphinx or the slow tier: the implementer's gate logs report them PASS, and nothing in them touches
this part beyond the page.

## 5. The diff audit

I read the 129 flagged lines in `m2/diff_audit_85c6e54d.txt` and spot-checked them against the diffs. No tolerance
changed.

**The four translation expectations** (`test_optimizers_frontend`) are the same blocks through `blocks_to_weight`. In
particular, `blocks_to_weight(full * fact2, y)` equals placing the blocks on the derivatives and then multiplying by
`k!` on the leading jet axes, bitwise: one product per entry either way.

**The other flagged lines:**
- `test_optimizers.py`: one re-wrapped `assertAlmostEqual`, whose reference is the element's own model (right: the
  stacked fit's `y` now carries `C`);
- `test_data_metric`: part 3's re-pin;
- `test_dispatch`:
  - the full-form weight `[[1],[2],[3]]`, at the same `1e-5 * 1e3`;
  - the compile case: the weight is broadcast on axis 2, which is `W` in `(d,) + F + W + (edge,)`; one new program;
    the same value again compiles nothing; Hessian products equal to 1e-10;
- `test_jets` and `test_sampling_definitions`: parts 1 and 3's re-ruled tables, block-shape assertions and refusals,
  reviewed there.

## 6. The rest of the part

- `prepare_batch` passes `stack_shape=x.stack_shape`.
- `_fits.old_model` gives the one `y` size-1 `C` axes after its one-`y` check.
- The page's example code moved to `blocks_to_weight`; the printed `(3, 4, 1, 1)` is `(d,) + F + W + edge`, with `W`
  and the edge of size 1.
- Part 4's two minor points are done: the `IGNORE_EXCEPTION_DETAIL` directive, and the `ValueError` for
  `f_resolution` without `y`, tested with a held `r`.
- The implementer's five mutants cover each library change of the part. My translation comparison covers the
  placement independently.

## Minor

1. **Correct the gauge explanation** in `battery.py`'s comment and in the handover's §4 (and in the squash message if
   it repeats it).
   - Suggested text: "an `mc_sgd` run's cores carry the sign gauge of its SVD retraction, which a rounding-level change
     of the step can flip from any start; the tensors and losses agree. This run was chosen because it does not flip.
     Gauge-invariant quantities (losses, the represented tensor) are what M2 promises equal."
   - Optionally, the battery could also store the dense tensor of each optimizer run, a gauge-invariant check that
     does not depend on the choice of run. This is not required: the current run passes, and loosening was ruled out.

Nothing is left running. My scratch records stay in `scratchpad/m2rev2/` (about 45 MB, regenerable). The working tree
is as I found it, apart from this file.
