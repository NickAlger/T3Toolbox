# P16, part 3 (the display and record): handover

Implementer, 2026-11-21, on `p16-wip` at `eec7b0fb` (part 2 committed). Built to plan P16 item 3, `P16_DESIGN_NOTE.md`
(part 3), the rulings in `LOG.md` (D4: `order` explicit) and the coordinator's brief. Nothing is committed. No decision
is needed; one reading is flagged (section 4). **Round 1** (review `P16_part3_REVIEW_r1.md`, response
`P16_part3_RESPONSE_r1.md`): the training errors are on the display's `batch`, numerator and denominator both.
Section 1's train-numerator bullet is updated.

## 1. What changed

**`t3toolbox/backend/optimizer_display.py`**
- **`make_newton_display(oracle, batch, val_batch=None, print_fn=print, fmt='%.1e', record=True,
  error_table=True)`**, which replaces `(problem, val_sample, val_data, ...)`.
  - **Denominators.** `y_block_sumsq` of `batch`, and of `val_batch` when one is given, is queried once for the life of
    the callback, at the first row's frame (a packed probe `y` reads its mode sizes off the frame).
  - **Train numerator, on `batch` (round 1).** When `info.lm.batch is batch` (the run's batch), each new point's
    `r_block_sumsq` is read at the model's frame with its centers' `Vc` and its `held`. The build holds `r`, so no
    sweep runs. Otherwise (a drawn batch, other data, `use_jit`'s jax copy) it is a forward at the model's frame on
    `batch` with no `held`: one forward more per row, and no `(unwt …)`, which would annotate the run's misfit with
    another batch's value.
  - **Validation numerator.** `r_block_sumsq` at the same frame on `val_batch`, with no `held` (the training `held`
    belongs to the training batch).
  - **Unweighted and unscaled.** The block sums carry no weight and no draw's scale (verified by script for every kind
    and for jet shapes `()`, `(1,)`, `(1, 1)`), so the errors are the per-block recovery errors.
  - **The `(unwt …)` figure** is `½ Σ r_block_sumsq`. It is passed to the header only when it differs (`rel_tol 1e-9`)
    from the model's own misfit `lm.misfit`, which comes from the same route. A nontrivial weight or a draw's scale
    shows it. Near the floor an unweighted fit no longer shows a spurious one: the history's misfit is the point
    route's since D9, and its rounding there differs from the frame route's block sums by more than `1e-9` (the test
    asserts that this happens).
  - **An oracle without either block sum** (checked against `optional_outputs`) prints the header lines and no table.
    Its records carry the scalars and no error arrays.
  - **Records.** `train_err` and `val_err` are kept in the weight's shape: `(d,) + F` probe, `F` apply, `()` entries.
    Nothing is reduced; 2026.5.0 stored the canonical `(n_mode, n_order)` matrix. There is no `val_err` key without a
    `val_batch`.
  - **Rows after a rejected trial** reuse the previous row's errors (keyed on the model object), as before.
- **`format_newton_iter(info, train_err, val_err=None, obj_unweighted=None, fmt='%.1e', order=None,
  error_table=True)`.**
  - `train_err` may be `None`, which gives the header lines only.
  - `error_table=False` gives the header lines only: the header, plus the time line on a timed row.
  - `order` is the jet shape (D4: given explicitly, never read off the model); `make_newton_display` passes
    `oracle.order`.
- **`_format_table(train_err, val_err, fmt, order=None)`** works on weight-shaped arrays:
  - **At `m = len(order) <= 1`** it is today's table, through the old body renamed `_format_matrix`, of the
    `(n_mode, n_order)` matrix:
    - probe `()`: mode columns;
    - probe `(n,)`: mode rows and order columns;
    - apply `(n,)`: order columns;
    - apply and entries `()`: one line.
  - **At `m >= 2`** there is one column per multi-index of the jet axes, in row-major order (`np.ndindex`), labelled
    `ord(j,k)` (O-42). Probe gets mode rows with `train|val` cells; apply gets `train` / `val` rows.
  - **Without `order`** (an oracle that names none) the axes are labelled by position: a scalar on one line; one axis
    in columns `[j]`; otherwise the first axis in rows `[i]` and the others row-major in columns `[j,k]`.
  - **A shape that does not fit `order`** raises a `ValueError` (structural), as does a validation shape that differs
    from the training shape.
- **`relative_errors`**: same function; its docstring now says the inputs are weight-shaped.
- **The module docstring** describes the new layout. One `corewise` import was dropped (the old validation forward is
  gone).
- **Doctests.**
  - `format_newton_iter`'s examples now pass weight-shaped arrays with `order`, and print the same text.
  - New: a two-variable example (multi-index columns) and an `error_table=False` example. Their output was run and
    pasted.
  - `make_newton_display` has a backend-only doctest (probe with validation; records of shape `(3,)`; the validation
    error falls).

**`tests/test_import_graph.py`**: the example of a `from package import module` edge re-pointed (section 5).

**`t3toolbox/backend/optimizers.py`** gets one docstring sentence on `_objective_resolution`, the reviewer's: "The
resolution is a scale estimate, so either route's `|r|` serves: the frame route's here sizes the point-route values
the acceptance compares (D9) as well."

**`tests/backend/test_optimizer_display.py`** (rewritten where the API changed; the formatter, timing and switch tests
kept):
- **The formatter tests** of the 2026.5.0 layouts now call `_format_matrix`, the same layouts. The timed-line test
  passes `order=(1,)`.
- **The timing tests** use the `oracle` lap: `_seconds` uses `oracle=0.07` in place of `sweep` and `residual`; the
  expected time lines and the closing table's row list follow.
- **New `TestTables`:**
  - `m <= 1` maps to the 2026.5.0 table, per kind;
  - `m >= 2` on a non-square `F = (2, 3)`: every cell is read back by its row and column labels (row-major told from
    column-major), probe and apply;
  - positional labels without `order`;
  - a mismatched shape raises;
  - `error_table=False` and `train_err=None` give exactly the header lines (timed and untimed).
- **New `TestMakeNewtonDisplay`**, against independent constructions:
  - `test_records_in_the_weights_shape`: probe, apply, entries at `()`; probe and apply at `(1,)` and `(1, 1)`; with
    and without validation. The `(1, 1)` table's columns are checked.
  - `test_the_errors_against_dense_residuals`: per row, train and validation errors against dense residuals at the
    row's point (einsum probes and applies, dense entries), on a weighted noisy fit, so the errors are shown to be
    unweighted. The header's `(unwt …)` equals ½ the dense unweighted squared residual.
  - `test_no_unwt_on_an_unweighted_fit`: run to the floor (Armijo exhausted). The test asserts that the two routes
    differ beyond `1e-9` at some row and that no `unwt` is printed.
  - `test_the_queries`: over two runs with one callback, `y_block_sumsq` is queried exactly once per batch. Each row's
    numerator is at `info.lm.frame` with `info.lm.held` and `info.lm.centers[0]`. Each validation query is at the same
    frame with `held=None`.
  - `test_rows_after_a_rejection_reuse_the_errors`: ported. The validation queries counted equal the distinct points;
    a shared callback recomputes the next run's first row.
  - `test_an_oracle_without_block_sums_gets_the_header_lines`: without both block sums, and without
    `y_block_sumsq` only, with and without validation.
  - `test_error_table_false_changes_no_record`: the printed block is the header line alone; the records are equal,
    arrays bitwise.
  - `test_uniform_equals_ragged`: packed batches give the ragged run's errors, in the same unpadded shape.
  - `test_a_drawn_batch`: on `oracle.take`'s batch the errors equal those of the same subset unscaled; `unwt` is the
    unscaled ½‖r‖², with objective/unwt = 3 (the scale 60/20).
- **The timing and switch end-to-end problems** are built with `F.derivative_fit` and `dd.forward` (probe along lines,
  noisy) in place of the old kind.
- **Removed:**
  - `test_needs_block_sumsq`: kinds are gone; the header-only test covers the oracle form.
  - `TestValidationArgsGoTogether`: `val_batch` is one object, so there is no pair to mismatch.
  - `test_records_and_silent`: subsumed by the records and dense tests.

## 2. Runs (`-n 2`, one at a time)

| Run | Result |
| --- | --- |
| `tests/backend/test_optimizer_display.py` | 39 passed, 68 subtests (part 2: 13 red, all expected); round 1, with `test_import_graph.py`: 49 passed, 118 subtests |
| Module doctests: `backend/optimizer_display.py`, `backend/optimizers.py` | pass |
| Full suite (once, to measure the red list) | 229 failed (subtests and tests), 1821 passed, 5 skipped, 1 xfailed, 90,717 subtests, 33 min; 91 red ids: the 90 frontend ids of part 2's list, and one new, fixed after it (below) |
| `tests/test_import_graph.py` (after its fix) | 8 passed, 50 subtests |

**Mutation pass** (scratch copy, deleted; `scratchpad/implementer3/mutate3.py`): 12 mutants, all killed:

| Mutant | Killed by |
| --- | --- |
| P1 the validation query given the training `held` | the queries test, the dense test (8 failures) |
| P2 the train numerator without `held` | `test_the_queries` |
| P3 the denominators re-queried at every row | `test_the_queries` |
| P4 the train numerator weighted | the dense test (3 kinds) |
| P5 no `unwt` guard | `test_no_unwt_on_an_unweighted_fit` |
| P6 multi-index columns column-major | `test_two_or_more_jet_variables` |
| P7 `error_table` ignored | `TestTables`, `test_error_table_false_changes_no_record` |
| P8 the record reduced to a 2-D matrix | `test_records_in_the_weights_shape` |
| P9 header-only decided on `r_block_sumsq` alone | `test_an_oracle_without_block_sums_gets_the_header_lines` |
| P10 no reuse after a rejection | `test_rows_after_a_rejection_reuse_the_errors` |
| P11 `unwt` from the scaled sums | `test_a_drawn_batch` |
| P12 the validation numerator on the training batch | the dense and queries tests |

## 3. Choices (mine, within the plan and rulings)

- **The `unwt` guard** compares `½ Σ r_block_sumsq` with `lm.misfit`, both from the model's frame route, rather than
  with the reported misfit. Without it, D9's point-route misfit makes an unweighted fit print a spurious
  `(unwt …)` near the floor. `format_newton_iter` keeps its own pure check, unchanged.
- **The validation query passes the centers' `Vc`** (not a held; centers belong to the frame), so validation's `z` is
  read through the same frame as the training numerator.
- **`format_newton_iter`'s `train_err` stays the second positional parameter** and now accepts `None`.
- **Positional labels** for an oracle without `order`: `[j]`, and `[i]` / `[j,k]`. The plan only says "labelled by
  position".
- **`error_table=False` keeps the timing closing table** after the last row. It is not the error table.

## 4. One reading to confirm

The brief's "the display reads … `f_resolution`" (plan P16 item 3) refers to the trust-region floor. The floor itself
is computed by the optimizer and moved to part 2 by D8: `_objective_resolution` gives `f_resolution + eps |rho|`, and
`eps (|f| + |rho|)` without `f_resolution`, `eps` from the dtype of `f`. Part 2 built and tested it
(`test_the_floor_reads_eps_off_f`), and it is unchanged here. The display never printed a floor, in 2026.5.0 or now,
so nothing of it is in the display. If the brief meant a printed floor cell, that is a new display feature to rule
on.

## 5. Expected red until part 4

`P16_part3_EXPECTED_RED.txt`, measured: 90 ids, all for part 4 (the frontend). They are exactly part 2's 90 frontend
ids, so part 2's 13 display ids are green and none is new.

One test outside the part turned red and is fixed: `tests/test_import_graph.py::TestImportGraph::test_the_graph_is_on_module_files`.
It used the display's `from t3toolbox import corewise` (the old validation forward's import, dropped) as its example of
a `from package import module` edge. It now uses the one remaining such import, `t3toolbox/__init__.py`'s
`from t3toolbox import safety`, and keeps the display's `optimizers` edge.

## 6. For part 4

- **"Both or neither" for validation.** When the translation builds `val_batch` from the old
  `val_sample=`/`val_data=`, keep the check that the two are given together. The old frontend raised there; the
  backend has one `val_batch`, so the check lives only in the translation.
- **The fast path under `verbose=True`.** The frontend should hand `make_newton_display` the same batch object that
  `newton_cg` runs on (under `use_jit`, the converted one); otherwise each row pays one forward more.
- **`stats['diagnostics']`'s `train_err` and `val_err` now have the weight's shape**, where 2026.5.0 stored a
  `(n_mode, n_order)` matrix. This needs an upgrade note.

## 7. Backlog (carried)

- The default trust-region radius tie (part 2's handover, section 6).
