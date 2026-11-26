# M3 handover: the block sums and the display

Implementer, 2026-11-23, on `b85a6fda` (the library and the tests as at `68c10621`). Built from `M3_DESIGN_NOTE.md`,
with the design review (`M3_DESIGN_REVIEW.md`) and Claude's rulings: D1, D2, the validation table (a), a private `C`
reader, and the five additions. Nothing is committed.

## 1. What changed

**The engine and the reference**
- `sampling_steps.r_block_sumsq_step(r, weight, ...)` gives `Σ r · data_metric(r, ω, None)` per block. It sums over
  the axes after the `m` jet axes (`W`, `C`, the edge) and keeps the mode and `F`. There is no `γ` here: the engine
  never sees it.
- The engine's row reads `('r', 'weight')`. The variable table and the module text say `<r, M r>` per block.
- `sampling_definitions.r_block_sumsq` multiplies by its own `_weight_squared` and still does not import `jets`. Its
  docstring gains a weighted example, a sample weighted zero.

**The oracle (`backend/fitting.py`)**
- Private `_stack_of_frame(frame)` reads `C` off the frame: `D[0]` or `B[0]` on the ragged layout, the Tucker
  supercore's axes after `d` on the uniform one. `y_block_sumsq` and `f_resolution` both use it; `f_resolution` no
  longer reads `np.shape(f)`.
- Private `_y_times_metric_y` builds `y · data_metric(y, ω, None)` at `z`'s shape, with `y` broadcast first. It is
  the old `_y_metric_y` body, unchanged.
  - `_y_metric_y` (`f_resolution`) reduces it as before; bitwise, see §3.
  - New `_y_block_metric_y` (`y_block_sumsq`) reduces it per block.
- `query`:
  - `y_block_sumsq` is `_y_block_metric_y`. It raises with a message if `batch.y is None`.
  - `_SCALED` gains `r_block_sumsq`, and the scaling loop covers `y_block_sumsq` too, so both carry `γ`.
- Removed: the `data_block_norms` route and the now unused `data_preparation` import.
- The class docstring now says:
  - the scale is part of `M`;
  - what the block sums are, that `γ` cancels in their ratio;
  - the drawn-batch sentence (review point 1);
  - each batch's sums use its own weight.
- The `Oracle` protocol docstring (`backend/optimizers.py`) says the display divides the two block sums, so a user
  oracle returning plain sums gets plain ratios.

**The display (`backend/optimizer_display.py`)**
- `relative_errors`: same code; the docstring says `‖ω⊙r_b‖/‖ω⊙y_b‖`, with a zero-weighted block shown as `—`.
- Removed:
  - `_format_header(info)` loses `obj_unweighted` and the `(unwt …)` branches;
  - `format_newton_iter` loses `obj_unweighted` with no shim (D1);
  - `make_newton_display.errors()` returns `(train_err, val_err)`, and the callback's plumbing follows.
- `make_newton_display`'s docstring states the weighted ratio and the validation rule (ruling (a)): each table is
  under its own batch's weight.
- The module docstring, the shape comments and two error messages say "the block shape".

**Text elsewhere**
- `data_preparation.data_block_norms`: "the block shape".
- The frontend `optimizers.newton_cg` docstring: the weighted ratio, the block shape and the validation rule.
- `examples/fit_probe_display.py`: the two `unwt` passages (D2).

**Tests**
- `tests/oracle_sweeps/oracle.py`: two additive helpers, `weighted_full` and `block_sumsq_full`.

## 2. Tests

**`tests/test_block_sums.py`** (new).
- `TestAgainstDenseSums`:
  - What is checked: the engine's `r_block_sumsq`, and the oracle's two sums on a manifold frame, a corewise point
    and the uniform layout. The packed probe batch has finite garbage written into the padding of the sample, `y`
    and the weight wherever their edge is `N`; edges of 1 come from the per-sample weights and the `edge 1` form
    of `y`.
  - The independent reference: loops over the blocks of `ω⊙(z_c − y_c)` and `ω⊙y_c`, written out by `np.repeat`,
    summed over the stack elements. `z_c` comes from `sampling_definitions.z` per element.
  - The grid:
    - 7 (kind, order) cases: `()`, `(2,)`, `(1, 1)` for probe and apply, plus entries;
    - `C` = `()` and `(2,)`;
    - weights: none, per entry with zeros, per sample with a zero, constant per block;
    - `y`: full, shared by `W`, shared by `C`, and (probe) an edge of 1.
  - Each case also checks the law `½ Σ_b r_block_sumsq = Σ_C f`, a consistency check between two engine routes.
- `TestSharedData`: a `y` shared by the stack, or by the samples, gives `y_block_sumsq = |C|` (or `|W|`) times one
  copy's (the `√|C|` fix).
- `TestNoWeight`:
  - `weight=None`, full-form ones and `blocks_to_weight(ones)` give `Σ r²` bitwise;
  - `y_block_sumsq` on a full `y` with no weight is `data_block_norms(y)**2` to 1e-14.
- `TestTheScale`: on a draw of 2 of 6 samples (per-sample weight), both sums are `3 ×` the dense sums over the drawn
  samples, and their ratio is the subset's.

**`tests/backend/test_optimizer_display.py`**
- New:
  - `test_a_zero_weighted_outlier_counts_in_neither_sum`, apply and probe `(1,)`:
    - one sample is corrupted by `1e3 ‖y‖` and weighted zero;
    - every row's `train_err` equals the plain dense ratio on the batch with that sample deleted (`np.delete`), to
      `rtol 1e-8, atol 1e-12`; the `atol` covers the rounding of the converged rows, where the ratio is about 1e-9;
    - the plain ratio with the outlier stays above 0.5;
    - the probe fit ends below 1e-6.
  - `test_each_table_is_under_its_own_batch_weight`, apply and probe:
    - the training weight is per sample, the `val_batch` has none, and both `W` are 40, so the lent-weight mutant is
      well formed;
    - `train_err` equals the dense weighted ratio and `val_err` the dense unweighted one;
    - the ratio under a lent weight differs by more than 1e-3.
  - `test_no_unwt_figure_on_any_run`: weighted, regularized and drawn runs.
  - `test_no_unweighted_figure`: the header, and `format_newton_iter(..., obj_unweighted=...)` raising `TypeError`.
- Converted:
  - every `_format_header` call drops its second argument;
  - the four pinned 2026.2.0 header strings lose `(unwt 1.70e-04)`, and their comment says so;
  - `test_the_errors_against_dense_residuals`: the docstring says its weight is block-constant (so the dense plain
    ratio is exact), and the `unwt` assertion becomes "not printed";
  - `test_a_drawn_batch`: the errors still equal the plain subset's; it checks the run's misfit is 3× the subset's
    and that `unwt` is not printed;
  - `test_header_misfit_reg_breakdown` keeps only the split;
  - the "weight's shape" wording becomes "block shape".
- Removed: `test_header_unweighted_shown_only_when_differs` and `test_no_unwt_on_an_unweighted_fit` (the feature is
  gone).

**`tests/test_least_squares_oracle.py`**
- `y_block_sumsq` is checked against `y_block_metric_y`, built from the dense helpers at `z`'s shape, to 1e-13.
- On a drawn batch the block sums move to the scaled list. New check: `r_block_sumsq = s·<r, w² r>` per block,
  written out.
- In the float32 test, `y_block_sumsq` is float32.

**Other files**
- `tests/test_sampling_definitions.py`: the reference's formula is `Σ ω² r²` per block.
- `tests/test_dispatch.py`: `r_block_sumsq` with a weight inside the jitted engine call, against numpy; the oracle's
  two block sums under `jax.jit`, ragged and uniform, against eager.
- `tests/test_optimizers_frontend.py`: two comments ("block shape").

**Mutation pass** (`scratchpad/m3impl/mutants.sh`). Each mutant ran on a fresh copy of `t3toolbox/` and `tests/`, over
`test_block_sums`, `test_least_squares_oracle`, `test_optimizer_display`, `test_sampling_definitions`,
`test_sampling_stacks` and `test_engine_data_space`.

| Mutant | Killed by |
| --- | --- |
| `γ` on `y_block_sumsq` only (`r_block_sumsq` out of `_SCALED`) | `TestTheScale`, the oracle's drawn test (7 subtests), the display's drawn test |
| `r_block_sumsq` unweighted (engine) | `TestAgainstDenseSums` (124), the outlier test, the validation test, the dense-residuals test, both drawn tests |
| `y_block_sumsq` over `y`'s own shape | `TestAgainstDenseSums` (109), `TestSharedData` |
| `val_batch` lent the training weight | `test_each_table_is_under_its_own_batch_weight` (and two query tests, by shape) |
| the reference unweighted | the reference's formula test (64 subtests); and, run separately, `test_sampling_variables::TestByDefinition::test_every_name_against_the_reference` (the engine against the reference, 21 failures with `test_sampling_stacks`) |

`test_sampling_stacks` alone does not see the reference mutant: its `r_block_sumsq` branch compares the stacked
engine with the engine per element, not with the reference. That is by design (stack consistency).

## 3. The gate

**The suite**, `pytest tests/ -n 2 -v` (under `timed.py`): **1960 passed, 5 skipped, 1 xfailed, 92,325 subtests
passed**, 38:07. Lowest MemAvailable 3.5 GB (`scratchpad/m3impl/gate/suite_v.log`).

**The node tally** (`m0/tally.py`, against M2 part 5's all-green `-v` log; `scratchpad/m3impl/tally_m2part5_vs_m3.txt`).
That log has 1,960 nodes; this run has 1,966. The 10 nodes that differ are all expected:
- 2 removed: `test_header_unweighted_shown_only_when_differs` and `test_no_unwt_on_an_unweighted_fit`;
- 8 new: the four of `test_block_sums.py` (164, 49, 6 and 3 subtests) and the four new display tests.

Every converted node has exactly its old outcome and subtest count.

**The other steps** (`run_gate.sh -n 2 doctest-modules doctest-quickstart doctest-pages examples sphinx`, one after
another; logs in `scratchpad/m3impl/gate_rest/`; Sphinx's `sphinx_html` and `sphinx_doctrees` deleted after reading):

| Step | Result | Seconds | Lowest MemAvailable (MB) |
| --- | --- | --- | --- |
| module doctests | PASS, 325 passed | 104 | 9,687 |
| `getting_started.rst` | PASS, 0 failed examples | 5 | 11,054 |
| pages (47 files) | PASS, 0 failed examples | 33 | 11,149 |
| examples | PASS, 14 examples, 0 failed | 983 | 10,523 |
| Sphinx `-W` | PASS, 0 warning or error lines | 1,329 | 10,359 |

**The battery** (M2's `battery.py new`, 25,476 arrays, base `68c10621` against M3):
- every `f_resolution` array (84, `orc/…`) is bitwise identical (the review's condition);
- everything that is not a block sum is bitwise identical (24,871 arrays, the jitted run and the optimizer runs
  included);
- all 437 `r_block_sumsq` and 84 `y_block_sumsq` arrays differ, as designed: the battery's weights are nonuniform per
  block, and its `C = (2,)` cases share `y` over the stack.

**The display records** (`scratchpad/m3impl/display_records.py`, recorded on `68c10621`, then on M3):
- Runs, each with a weight constant per block (`blocks_to_weight` of nonuniform blocks):
  - probe `(1,)` with validation, plain probe with validation, apply `(2,)`, entries with validation, apply `(1,)`
    regularized with validation, and apply on a drawn batch;
  - `y` and `ω` are full on `W`, nothing is stacked, and every validation batch carries the same block form.
- Result:
  - `train_err` and `val_err`: 40 arrays, worst relative difference 2.3e-16 against `rtol 1e-12`;
  - every other record field, the final cores, the stop reasons and the step counts: bitwise (682 arrays;
    `wall_time` excluded);
  - two base runs agreed with each other exactly.

**Expected red** (`M3_EXPECTED_RED.txt`, one run after the library change and before the test conversions, on the 18
test files that read the block sums, the display or `f_resolution`):
- 117 failures, all explained:
  - the header's removed argument and the `unwt` assertions (`test_optimizer_display`);
  - the oracle's old `y_block_sumsq` formula and the old "plain" list (`test_least_squares_oracle`);
  - the reference's unweighted formula (`test_sampling_definitions`).
- Every other node of those files was green.

## 4. Departures and choices

- **One private helper for `C`**, used by both sums (ruling). `f_resolution` reads `C` off the frame too; its numbers
  are bitwise unchanged.
- **`y_block_sumsq` without `y`** now raises a `ValueError` naming the need, as `f_resolution` does. Before, it would
  have failed inside `data_block_norms`.
- **The `(1, 1)` dense `z`** comes from `sampling_definitions.z`, the library's dense definitions, because
  `oracle.S_taylor` takes one jet variable only. The weighting, the broadcast and the per-block reduction, which are
  what M3 changes, are all written out in the test.
- **The display records** compare to `rtol 1e-12` with a `wall_time` exclusion. These are the gate conditions of
  review minor 3.
- **Not run:** the slow tier. M3 touches nothing in the sweeps except two additive helpers in `oracle.py`.

## 5. Found outside the slice

- **For P19's list.** `docs/fitting_and_optimization.md:114` should read: "the weighted ratio `‖ω⊙r_b‖/‖ω⊙y_b‖`; the
  same as the plain one for a weight constant within each block". Next to it, §4.6's prose.
- **For P19's upgrade note (D1).** "Positional callers of `format_newton_iter`: remove the fourth argument". With
  no table, the old call passes silently.
- **The old layer.** `SamplingKind.block_sumsq`, `block_sumsq_over_*` and `has_block_sumsq` stay unweighted and
  untouched; nothing reads them now, and they go with the old kinds.
