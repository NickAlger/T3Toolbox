# P16 part 3, response to review round 1

Implementer, 2026-11-21. Review: `P16_part3_REVIEW_r1.md` (NOT YET, finding 1). Ruling: the display reports its
errors on the batch it is given, with numerator and denominator both on `batch`. Nothing is committed.

## Finding 1: built as ruled

In `make_newton_display`, the training numerator is now read on `batch`:
- **`info.lm.batch is batch`** (the run's batch): `r_block_sumsq` at the model's frame, with its centers' `Vc` and its
  `held`. This is the residual the build computed, so no sweep runs (unchanged).
- **Otherwise** (a drawn batch, other data, `use_jit`'s jax copy): `r_block_sumsq` at the model's frame on `batch`,
  with `Vc` and **no** `held`, as validation does. That is one forward more per row.
- **The denominator** is unchanged: `y_block_sumsq` of `batch`, queried once.
- **The docstring** states the two paths, the `use_jit` cost, and that the caller should hand the display the very
  batch object the run uses.

One consequence, my choice: on the second path the header's `(unwt …)` is not shown. It annotates the run's misfit (a
value on the run's batch), and the second path's `½ Σ r_block_sumsq` is on another batch. That makes the
figure incomparable, not a weighting gap. On the run's own batch it is shown as before.

## Tests (`tests/backend/test_optimizer_display.py`)

- **`test_the_errors_are_on_the_displays_batch`** (the reviewer's measured case, extended). The display is on a noisy
  60-sample apply batch, and `newton_cg` runs on two different draws (every third sample, offsets 0 and 1) with one
  callback. The test checks:
  - every row's `train_err` equals the dense relative error on the full batch at the row's point (`1e-8`);
  - every numerator query on the full batch has `held=None`;
  - `y_block_sumsq` is queried once;
  - no `unwt` is printed.
- **`test_on_the_runs_batch_no_sweep`.** The display is on the run's own batch (probe along lines). The engine's `_run`
  is recorded during each display call, and none of `xi, mu, nu, eta, z, r` runs.
- **`test_the_queries`** (unchanged) still pins the fast path's frame, `held` and centers, and the validation query's
  `held=None`.

**Mutants** (`scratchpad/implementer3/mutate3r1.py`), all killed:

| Mutant | Killed by |
| --- | --- |
| Q1: the fast path without `held` | `test_on_the_runs_batch_no_sweep` |
| Q2: the slow path given the run's `held` | `test_the_errors_are_on_the_displays_batch` |
| Q3: always the slow path | `test_on_the_runs_batch_no_sweep` |
| Q4: the numerator on the run's batch (the reviewed bug) | `test_the_errors_are_on_the_displays_batch` |
| Q5: `unwt` shown on another batch | `test_the_errors_are_on_the_displays_batch` |

Q1 replaces round 0's P2, whose target line changed. The other eleven of round 0's mutants target code this change did
not touch.

## Runs (`-n 2`, one at a time)

- `tests/backend/test_optimizer_display.py` and `tests/test_import_graph.py`: 49 passed, 118 subtests.
- Module doctest of `backend/optimizer_display.py`: passes.
- The expected-red list is unchanged (90 frontend ids). The change is confined to the display and its tests, which
  are green.

## For part 4 (added to the handover's list)

- **"Both or neither" for validation.** When the translation builds `val_batch` from the old `val_sample=`/`val_data=`,
  it must keep the check that the two are given together, as the old frontend raised.
- **The fast path under `verbose=True`.** The frontend should hand `make_newton_display` the same batch object
  `newton_cg` runs on (under `use_jit`, the converted one). Otherwise each row pays one forward more.
