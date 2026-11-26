# M2 part 3 handover: `jets` on the rule

Implementer, 2026-11-22, on `m2-wip` at `a80f3a62` (part 2 committed). Built on `M2_DESIGN_NOTE.md` (part 3),
`M2_DESIGN_REVIEW.md` and decisions 13 to 15 of `M_ARC_DECISIONS.md`. Nothing committed.

**Scope.** `backend/jets.py`: the checkers, the dual-jet converters, the factorials and `prepare_derivative_data` on the
data-space rule (plan §2 to §4); `check_batch`'s one call updated; the tests of `jets` and the `jets` rows of
`test_dispatch` and `test_jax_absent`. The fitting layer (part 4) and the frontend and old forms (part 5) are not
switched.

## 1. What changed

**`t3toolbox/backend/jets.py`.**
- **The checkers** (plan §2; corrections (d)): `check_data(kind, order, sample, y, stack_shape=())` and
  `check_weight(kind, order, sample, weight, stack_shape=())`, one private rule `_check_of_type_of_z`: `z`'s shape
  read from the sample (`F + W + C + (N_i,)`, `F + W + C`, `W + C`), exact rank, each axis 1 or `z`'s size; for probe
  exactly `d` arrays, a tuple, a list, or one array whose first axis is exactly `d` (a mode axis of 1 refused); a
  tuple for apply or entries refused. `check_weight`'s signature changes from `(kind, order, weight, d)` (unreleased:
  P16). Messages name the object, the kind, the mode and both shapes.
- **The converters through `data_metric`** (plan §3): `jet_to_dual_jet` is `flip(data_metric(jet, omega, None))` per
  mode; `dual_jet_to_jet` flips back and divides by `_omega_squared` (`M^{-1}` at `gamma = 1`). Both read the weight
  with the new private `_mode_weights` (probe: a tuple, a list, or a mode-leading array, exactly `d`; else one array)
  and check the pair with `_check_pair`, which gains a `name` for the message. A `K`-stacked jet raises when a weight is
  given and is turned when it is `None`. `_squared_weights` is deleted. Signatures unchanged.
- **The factorials** (plan §4): `derivative_weight_to_taylor_weight(kind, order, weight)` gains `kind` (unreleased).
  `_taylor_weight` puts the factorials on the **leading** jet axes (after the mode axis of a mode-leading probe array),
  so a size-1 jet axis comes back with the full `F`; the form comes back as given (a tuple for a tuple or a list). The
  shared private `_taylor_weights(kind, order, weight, xnp)` lets `prepare_derivative_data` dispatch once over all its
  inputs.
- **`prepare_derivative_data`**: the weight is of the type of `z` and is checked against `y` (`_check_pair` per mode);
  `weight=None` returns `k!` through `blocks_to_weight` (one mode-leading array for probe), the decided default. Q7's
  docs: Taylor coordinates everywhere, one conversion at the boundary, the recipe for a data-relative weight, the
  coordinate dependence of an equalizing weight's floor.
- **The contragredient sentence** (audit D4) in `derivatives_to_taylor` and `taylor_to_derivatives`: an element of `Y`
  converts from derivative to Taylor coordinates by `taylor_to_derivatives`.
- The module text: the type of `z` with `C`, the weight of the type of `z`, the coordinates; its examples moved to the
  full form. 14 module doctests pass.

**`t3toolbox/backend/fitting.py`.** `check_batch` calls `check_weight(kind, order, batch.sample, batch.weight)`. No
`stack_shape` yet: part 4 adds it (correction (e)).

## 2. Tests

- **`test_jets.py`** (39 tests):
  - `TestDualJet`: weights per entry (`random_weight` now of the type of `z`, a list for probe) through the
    definition, the round trip, the pairing identity, the zero weight, the packed probe, the dtype matrix.
    `test_a_backend_function_given_a_bad_weight_checks_nothing` is replaced by
    `test_a_converter_refuses_a_weight_not_of_the_type_of_the_jet`: the block forms, a `K`-stacked jet with a weight,
    a wrong size, a tuple for apply, `d - 1` arrays and a mode axis of 1 raise with the converter's own name; with no
    weight a `K`-stacked jet is turned; `prepare_derivative_data` refuses a block weight.
  - `TestDerivativeData`: **the factorials by hand** (`test_the_weight_goes_the_opposite_way_on_its_jet_axes`): every
    entry of the converted weight against `given[source] * k!`, read at 0 on its size-1 axes; per entry, per sample
    (jet axes of 1, which come back full) and a mode-leading probe array; sizes distinct (`W = (5,)`, the edges
    `4, 5, 6`) and equal (`F = W = (3,)` against edges of 3). The objective written on derivatives is reproduced with
    per-entry weights; no weight gives the factorials in `blocks_to_weight`'s form and passes `check_weight`.
  - `TestCheckers`: **the re-ruled rejection table** (19 cases, `d = 3`, `W = (7,)`, `n = 3`): 2 legal (apply `(1,)` at
    `()`, entries `(1,)`), 17 raise, each with its reason (the mode count, or the shape by rank or size), pinned by
    regex; the acceptance table is the full forms (`z`'s shape, every axis 1, from blocks, mode-leading per sample;
    `C = ()` and `(2,)`); `check_data` refusals re-ruled (a stacked `y` needs `stack_shape`; a mode axis of 1; a
    padded mode-leading array); size-1 axes and a mode-leading array pass; **new:** an object larger than `z` where
    `z`'s axis is 1 (a one-sample batch, the jet axis at `(0,)`) raises; the transposed-square limit, re-stated on a
    full form.
- **`test_data_metric.py`: the bitwise bridge re-pinned.** `_old_block_weighting`, written out in the test, is the
  pre-M2 seed before its turn (each array and its block row cast to the lower precision of the array and the whole
  block array, squared, placed on the leading jet axes, times the array). `data_metric(r, blocks_to_weight(...), None)`
  and the turn undone of `jet_to_dual_jet(..., same weight)` both equal it bitwise, dtypes included, in the four
  float32/float64 pairings, on `C`, packed and ragged. It no longer compares `data_metric` with itself.
- **`test_type_laws.py`**: the converter laws with full-form weights (`blocks_to_weight`); the swap catcher likewise.
- **`test_data_preparation.py`**: the block norms keep the block shape; placed by `blocks_to_weight` they pass
  `check_weight` (renamed `test_the_block_norms_have_the_block_shape`).
- **`test_dispatch.py`**: the `jets` rows of `_jets_and_preparation_cases` (full-form weights from the same random
  blocks; `equalizing_weight` keeps the blocks; `derivative_weight_to_taylor_weight` with `kind`; entries `(1, 1)`),
  `test_one_dispatch_over_all_the_inputs_of_a_call`, the x64-off row, and `test_the_checkers_run_under_jit` (full-form
  weights; the "the data y" refusal now passes a `y` with a wrong `W`, since a full-form weight is also legal data; the
  bad weight is `(3,)`, which no kind accepts).
- **`test_jax_absent.py`**: the `jets` rows with full-form weights.

## 3. Mutants: 13 of 13 killed

Script `scratchpad/m2impl2/mutants_p3.py` on a scratch copy (deleted), against `test_jets`, `test_data_metric`,
`test_type_laws`, `test_data_preparation`.

| Mutant | Killed by |
| --- | --- |
| the checker accepts any size where `z` is 1 | the new larger-than-`z` test (4); **survived before it** |
| the checker's rank test dropped | the rejection table (6) |
| a mode axis of 1 broadcast by the checker | `check_data` refusals, the rejection table |
| `stack_shape` ignored | 41 |
| the factorials on the trailing axes (the old alignment) | the factorials by hand, the objective, the given weight (8) |
| `lead = 0` for a mode-leading probe weight | the factorials by hand |
| `jet_to_dual_jet` unturned | 149 |
| `dual_jet_to_jet` divides by `omega`, not `omega**2` | 53 |
| `dual_jet_to_jet` divides before the flip | 43 |
| `jet_to_dual_jet`'s own pair check dropped | the converter refusals by name (4); **survived before** (`data_metric` raised the same error under its own name) |
| no weight gives ones, not `k!` | 12 |
| `prepare_derivative_data`'s weight check dropped | the converter refusals |
| the weight squared before the cast (M1's L4) | the re-pinned bitwise bridge (8: `r` float32, weight float64) |

## 4. The runs

| Run | Result |
| --- | --- |
| `jets` module doctests | 14 passed |
| `test_jets`, `test_data_metric`, `test_type_laws`, `test_data_preparation` | 94 passed, 988 subtests |
| `test_dispatch` / `test_jax_absent`, the `jets`, checker and dispatch rows | 11 passed, 1 failed (`one_dispatch`: the weight conversion dispatched on the weight alone; fixed by `_taylor_weights`), then that test 1 passed, 13 subtests |
| **the full suite, `pytest tests/ -v -n 2`** | **434 failed, 1,892 passed, 5 skipped, 1 xfailed, 91,210 subtests passed, 31:53** |

**The red, `M2_part3_EXPECTED_RED.txt`:** 123 ids, each tagged with its cause (all old-form objects, or the old
`f_resolution`, that parts 4 and 5 switch). Against part 2's 194: 121 still red, 2 new, 73 green (listed in
`m2/part3_green_since_part2.txt`).
- **The 2 new:** `test_check_batch` (check_batch now checks the weight by the rule; its batch has a block weight) and
  the translation's `test_each_old_form_is_the_new_form` (`prepare_derivative_data` refuses the translation's block
  weight). Both predicted by the design note.
- **Where the rest now raise:** 70 at `prepare_derivative_data`'s check (block weights from `tests/_fits.py:92` and the
  translation), 14 at `check_batch`, 34 still at the engine, 3 sweeps (every collected error is one of the first two),
  1 the old signature (`test_least_squares_oracle.py:535`), 1 the `f_resolution` gate (below).
- **The 73 green, and a caveat (decision for Claude, §6).** They are fits whose only weight is
  `prepare_derivative_data`'s default, which is now full form, so the oracle passes the engine a legal weight; their
  pre-M2 assertions pass unchanged. But the oracle's `f_resolution` (`backend/fitting.py:958`, part 4's) still
  computes `weight**2 * y_block_sumsq`, the block formula: on a full-form weight it broadcasts into cross terms
  (`(sum w^2)(sum y_b)` where `sum w_b^2 y_b` is meant). Checked by hand: a factor 1.021 on an apply case; the T5
  float32 gate (`test_dispatch.py::TestSwitchNewton::test_rounding_gate_at_float32`) sees it, 412.98 against 224.98,
  and is the one id red for this reason (R). The floor only governs the `objective_floor` stop and the switch's
  rounding gate, and no green test asserts it closely enough to see; part 4 replaces it (decision 13(c)).

**The tally against part 2's `-v` log** (`m0/tally.py`, kept as `m2/tally_part2_vs_part3.txt`): 100 nodes differ: the
red-to-green and green-to-red ids above, and 10 `test_jets` / `test_data_preparation` / `test_engine_data_space` nodes,
each a renamed, replaced or added test of this part (or part 2's round-1 test, added after part 2's log was taken). No
other node changed.

## 5. Departures and choices

- `prepare_derivative_data` checks its weight against `y` (rank and sizes, `_check_pair`): it is the one function that
  has both, and a block weight would otherwise be converted silently and fail later in the engine.
  `derivative_weight_to_taylor_weight` alone cannot tell a block from a full form (it has no `y`), so it checks only
  that the weight has its jet axes.
- The converters raise under their own names (`jet_to_dual_jet: ...`, `dual_jet_to_jet: ...`) rather than
  `data_metric`'s, so the message names the function called.
- `test_the_checkers_run_under_jit`'s "the data y" refusal changed its bad `y`: the old one was the weight passed as
  `y`, and a full-form weight is legal data. That changes a test's input beyond the weight and `y` forms; recorded here
  for the diff audit.

## 6. Found outside the part

- **Decision needed (small): the interim `f_resolution`.** Part 3 makes the default derivative weight full form before
  part 4 rewrites `f_resolution`, so on `m2-wip` between the two the floor is silently off for those fits (see §4).
  Recommendation: leave it to part 4, whose first item it is; the branch is squashed before it reaches
  `graded-messages`, and the T5 gate guards it. The alternative, a one-line interim fix here, would be rewritten in
  part 4 anyway.

- Parts 4 and 5 meet the new signature of `derivative_weight_to_taylor_weight` and `check_weight`
  (`test_least_squares_oracle.py:535`, the frontend translation `optimizers.py:196` through `prepare_derivative_data`,
  `tests/_fits.py:92`, the page `fitting_and_optimization.md:426`): all already red from part 2 for the block weight.
