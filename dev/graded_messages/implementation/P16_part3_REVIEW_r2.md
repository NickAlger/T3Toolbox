SIGN OFF: the display reports its errors on the batch it is given, numerator and denominator both. My measured case is now right. The slow path never receives the run's `held`. Leaving out `unwt` there is correct. Nothing else changed.

# P16 part 3 review, round 2

Reviewer, 2026-11-21, on `P16_part3_RESPONSE_r1.md` and the code as built under Claude's ruling. Runs were one at a
time, at `-n 2`. The working tree is untouched apart from this file.

## My measured case, under the new semantics

**The setup is round 1's:**
- the display on the 60-sample apply batch;
- `newton_cg` on `oracle.take(full, every third)`;
- the oracle's `query` spied.

**The errors are now right.** Each recorded `train_err` equals the dense relative error on the display's batch at the
row's point:

| Row | Recorded | Dense |
| --- | --- | --- |
| 0 | 1.3520452144155604 | 1.3520452144155604 |
| 1 | 1.1941532038546119 | 1.1941532038546117 |

Round 1's value of 0.550 was the drawn batch's residual over the full batch's data, and it is gone.

**The slow path never gets the run's `held`.**
- Both numerator queries on the full batch went out with `held=None`.
- In the code, the `else` branch makes its query with `Vc` only. The centers belong to the frame, not to a batch, so
  reading `z` through them is right on any batch.
- The implementer's mutant Q2 (the slow path given `held`) is killed by `test_the_errors_are_on_the_displays_batch`.
  Q4 (the numerator on the run's batch, round 1's bug) is killed by the same test.

**The fast path.** On `info.lm.batch is batch`, the query is unchanged: `held` and `Vc`, no sweep. It is pinned by
`test_on_the_runs_batch_no_sweep`, which records `_run`. Mutants Q1 and Q3 are killed by it.

## The `unwt` choice: right

- `(unwt …)` annotates the run's misfit, a value on the run's batch, with its unweighted counterpart on that same
  batch.
- On another batch, `½ Σ r_block_sumsq` is a different data set's sum, so the comparison has no meaning.
- Dropping it there is correct, and it is better than any substitute: a value to show would need the run's own block
  sums, which is exactly the held query the slow path must not make.
- On the run's own batch the guard is round 1's same-route comparison.
- In the measured case no `unwt` is printed. Mutant Q5 (`unwt` shown on another batch) is killed.

## Nothing else changed

**`git diff --stat` touches the same four files as round 1:**
- the display;
- the display's tests;
- `test_import_graph`'s re-pointed example;
- the one docstring sentence in `optimizers.py` (I checked its diff: those three lines only).

**The display's `errors`** differs from my round-1 reading only in three places:
- the `own` branch;
- the `held`-free `else` branch;
- `not own` added to the `unwt` guard.

The denominators (once, on `batch` and on `val_batch`), validation (no `held`), the records, `error_table` and the
reuse after a rejection are as in round 1.

**Runs:**
- the display and import-graph tests: 49 passed;
- the module doctest passes;
- the expected-red list is unchanged at 90.

## Carried to part 4 (from the response, agreed)

- **Validation: both or neither.** The translation must keep the check that `val_sample=` and `val_data=` are given
  together when it builds `val_batch` from them.
- **The display's batch.** The frontend should hand `make_newton_display` the very batch object `newton_cg` runs on
  (under `use_jit`, the converted one), or every row pays one forward more.
