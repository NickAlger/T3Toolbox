NOT YET, on one should-fix finding: the training table's numerator is read on the row's model batch and its denominator on the display's own `batch`. A run on another batch (the minibatched Newton the notes describe) then prints and records wrong errors, silently. Everything else checks out.

# P16 part 3 review, round 1: the display and the record

Reviewer, 2026-11-21, on the uncommitted part 3 over `eec7b0fb`. Runs were one at a time, at `-n 2`. The mutant copy was
deleted. The working tree is untouched apart from this file.

## What I checked

**The runs.**
- `test_optimizer_display` and `test_import_graph`: 47 passed.
- The module doctest of `optimizer_display.py` passes.
- `P16_part3_EXPECTED_RED.txt` is exactly part 2's list less its 13 display ids. That matches a change confined to
  the display, the import-graph test and one docstring sentence.

**Unweighted and unscaled sums, against my own dense residuals** (`reviewer2/p3_check.py`). The setup:
- probe at `order=(1, 1)` along planes `affine_jet(ww, p, q)`;
- noisy data, a random weight of shape `(d, 2, 2)`;
- a validation batch of another size `W`;
- three Newton rows.

The results:
- The records' `train_err` and `val_err` have shape `(3, 2, 2)`, the weight's shape, nothing reduced.
- Every cell equals `sqrt(sum (z - y)^2 / sum y^2)` per (mode, slot), computed from `dense_probe` at the row's
  `x_cores` with no weight, to 8.8e-16.
- The header shows `(unwt 2.63e+06)` against a weighted `obj 3.199e+06`, as it should.

**The `m >= 2` columns, read back.** In the printed table, the cell at row `m2`, column `ord(1,0)` reads
`1.9e+00|1.4e+00`, which equals `train_err[2, 1, 0]` and `val_err[2, 1, 0]`. The columns are `ord(0,0) ord(0,1)
ord(1,0) ord(1,1)`, row-major (O-42). The implementer's test reads every cell of a non-square `F = (2, 3)`.

**The training numerator and validation.**
- The numerator is `oracle.query(lm.frame, lm.batch, ('r_block_sumsq',), Vc, held=lm.held)`, and the build holds `r`,
  so no sweep runs.
- Validation queries at `lm.frame` on `val_batch` with **no** `held`.
- The implementer's mutants P1, P2 and P12 cover these three facts.

**The denominators** are queried once per callback, at the first row's frame. The queries test counts them, and mutant
P3 dies.

**Records, `error_table=False`, header-only oracles, uniform against ragged, and the drawn batch:** each is tested in
`TestMakeNewtonDisplay` against an independent construction, and each has a killed mutant (P7 to P11). I read the
code paths and agree.

**The `unwt` guard: the right comparison.**
- `(unwt …)` should appear exactly when a weight or a draw's scale makes the weighted misfit differ from the plain
  `½‖r‖²`.
- Comparing `½ Σ r_block_sumsq` with `lm.misfit` compares two values of **one** route, the frame route, through the
  same held `r`. They differ only by the weight and the scale.
- Comparing with the reported misfit, which since D9 is on the point route, mixes routes and prints a spurious `unwt`
  near the floor.
- My mutant doing exactly that (the guard against `info.misfit`) is **killed** by `test_no_unwt_on_an_unweighted_fit`.
- `format_newton_iter` keeps its own pure check against the reported misfit. That is harmless, since the guard
  already drops the equal case.

**`TestValidationArgsGoTogether`: nothing is lost in the backend.** `val_batch` is one object, so there is no pair to
mismatch. A batch with no `y` raises in the engine ("`r` needs the argument `y`"). The pairing rule survives in one
place: the old frontend form `val_sample=`/`val_data=`. **Part 4's translation** must keep "both or neither" when it
builds `val_batch` from them (the old frontend raised there).

**The floor reading (section 4 of the handover): agreed.** The display prints no floor. The floor is part 2's, built
and tested.

## Finding

**1. (Should fix.) The denominator and numerator of the training errors can come from different batches.**

*The cause.* The training denominator is `y_block_sumsq` of the display's `batch`, queried once. The numerator is
`r_block_sumsq` on `info.lm.batch`, the batch the run actually uses. When the two differ, the error is
`sqrt(sum r_b^2 / sum y_full^2)`: wrong, with no error raised.

*When they differ:*
- a display built on the full batch and a Newton run on a drawn batch, which is the "minibatched Newton" of the
  settled notes;
- a callback reused across runs on different batches;
- any caller that builds the display from another batch than the one passed to `newton_cg`.

*Measured.* A display on a 60-sample apply batch; `newton_cg` on `oracle.take(full, every third)`. Row 0's recorded
`train_err` is 0.550; the true relative error on the run's batch, computed dense, is 2.469.

*Before P16.* 2026.5.0 had the same exposure if a callback met another problem's run. But minibatching was then only
`mc_sgd`'s, which takes no display. The new layer makes a run on a drawn batch an ordinary call, with both batches in
the user's hands.

*Resolution (recommended).* Read the training denominator on the same batch as the numerator:
- `y_block_sumsq` of `info.lm.batch`;
- memoized by the batch object's identity (keep a reference and compare with `is`), so a fixed batch is still queried
  once;
- a new batch (a draw, a continuation level on other data, `use_jit`'s jax copy) is queried once more.

Keep `batch` in the signature (plan 8.4); in this design it only states the training data. The first row's query can
reuse it when `info.lm.batch is batch`.

A structural check that `info.lm.batch is batch` would be simpler, but it fails under `use_jit`: `newton_cg` converts
the batch to jax, a new object. Validation already reads its numerator and denominator on the one `val_batch`.

*Test:* the measured case above. A display on the full batch and a run on a drawn batch must give the dense error on
the drawn batch. A second run with the same callback on another draw recomputes.

## Mutants

| Mutant | Result |
| --- | --- |
| my own: the `unwt` guard compared with the reported (point-route) misfit | killed |
| the implementer's 12 (P1 to P12) | not repeated; they cover the remaining risks |
