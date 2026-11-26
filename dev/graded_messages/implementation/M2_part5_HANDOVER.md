# M2 part 5 handover: the frontend, the old forms, the pages, and M2's gate (the final handover of M2)

Implementer (the third of M2), 2026-11-22, on `m2-wip` at `7eda5d14` (part 4 committed). Built on `M2_DESIGN_NOTE.md`
(part 5 and §2, the gate), `M2_DESIGN_REVIEW.md` (finding 2: the battery's expectations), `M2_part4_HANDOVER.md` §6 and
`M2_part4_EXPECTED_RED.txt`, and Claude's brief for part 5 (the review's two minor points). Nothing committed.

**Scope.** The frontend passes the model's stack to `check_batch`. The old-form translation and `_old_draw` place
their block weights with `blocks_to_weight`, and give the old shared data size-1 `C` axes. `_fits.old_model` likewise.
The stacked-`y` frontend tests, the page's doctest, the compile test's shape case, and the two minor points of part 4's
review. Then M2's gate: the battery, the diff audit, the full gate with the node tally.

## 1. What changed

**`t3toolbox/fitting.py`.** `prepare_batch` calls `oracle.check_batch(batch, x.shape, stack_shape=x.stack_shape)`
(correction (e): the frontend always passes the model's stack); its docstring says so.

**`t3toolbox/optimizers.py`, the translation (deleted in P17).**
- `_old_fit(..., stack_shape=())`: the old weight, canonicalized as before (`_canonical_weight`) and broadcast to one
  value per block (probe `(d,)` or `(d, n+1)`, else `(n+1,)`), is placed on the data with `blocks_to_weight`; for a
  derivative kind on the derivative data, before `prepare_derivative_data` (which converts data and weight together).
- New `_shared_by_the_stack`: the old data, one dataset for a stacked point, gain size-1 `C` axes (before the probe
  edge, else last). `_translate` passes `x0.stack_shape` to `_old_fit` and to `_old_draw`, which gains the keyword.
- Both-or-neither on `val_sample` / `val_data` is unchanged (its check precedes the conversion); the validation batch
  goes through the same `_old_fit`, with no weight, as before.

**`t3toolbox/backend/fitting.py`** (part 4's review, minor points).
- `query` raises a `ValueError` naming the cause when `f_resolution` is wanted and the batch has no `y`. `f` alone can
  come from a held `r`; `<y, M y>` cannot. Before, it failed deep in `broadcast_to(None)`. The class docstring says
  so.
- `check_batch`'s traceback doctest carries `+IGNORE_EXCEPTION_DETAIL`.

**`docs/fitting_and_optimization.md`** (section 4.6's example, code only; the surrounding prose describes the old
call forms and is P19's to rewrite): the per-(mode, order) blocks and the per-mode vector placed with
`fitting.blocks_to_weight`; the printed `np.shape(w)` is now `(3, 4, 1, 1)` (pasted from the run).
`docs/numerical_contracts.md` needed no change (its weight goes through the translation, which now places it).

**`dev/graded_messages/implementation/m2/battery.py`**: the two `mc_sgd` runs start at a seeded random point
instead of the zero tensor (§3, the battery), with the global numpy state restored after the draw so that every other
section's inputs are unchanged.

## 2. Tests

Each converted test changes only how its `y` or weight is built and passed, never an assertion's tolerance (the
audit, §3).
- `tests/test_fitting.py`:
  - `_setup` (the eight `TestLocalModel` tests): the shared `y` has size-1 `C` axes, and `r = z - y` by position.
  - `TestUniformBackendGeometryFromFrame`: its `y` has size-1 `C` axes.
  - `test_plain_probe_per_mode` and `TestPerModeWeightRowCount.test_over_long_is_rejected` (the accepted call): the
    per-mode vector placed with `blocks_to_weight`. The refused five-row weight still raises.
- `tests/test_optimizers_frontend.py`:
  - The translation's four `batch.weight` expectations are the same blocks placed with `blocks_to_weight`.
  - `_fit_body` places its weight with the frontend's `fitting.blocks_to_weight`, so its source still names no backend
    module.
- `tests/_fits.py`: `old_model` keeps the one-`y`-over-the-stack check, then gives that `y` (and the weight) size-1 `C`
  axes with `shared_over`. Its docstring said "the data cannot carry the point's stack", which is no longer true.
- `tests/test_oracle_data_space.py`: new `test_it_needs_y_even_with_a_held_r`. With no `y`, a held `r` gives `f`, equal
  to the full call's; `f_resolution` raises with the message.
- `tests/test_dispatch.py`, `test_jit_ragged_local_model_parameterized_kind` (P16's batch-leaf compile test) gains the
  shape case of plan §4:
  - The same weight written out full on `W`, where it had size 1, is exactly one new program, and gives the same
    Hessian product to `1e-10`.
  - Another value of that shape compiles nothing (trace counts `2`, then `2`).

**Mutants** (`scratchpad/m2impl3/mutants_p5.py`, on a scratch copy, deleted after; run against
`test_oracle_data_space` and the frontend classes of `test_fitting` and `test_optimizers_frontend`). Unmutated: 43
passed. **5 of 5 killed:**

| Mutant | Killed by |
| --- | --- |
| `prepare_batch`: `stack_shape` not passed | `TestLocalModel` and the uniform `from_frame` test (41) |
| `_old_fit` (plain): the blocks passed as the weight | the translation tests, the length-one weight (5) |
| `_old_fit` (derivative): the blocks placed on the Taylor data after the conversion (no `k!`) | the translation tests (6) |
| `f_resolution`: the missing-`y` check dropped | `test_it_needs_y_even_with_a_held_r` |
| `_fits.old_model`: no size-1 `C` axes | `test_the_old_model_shim_needs_one_y_over_the_stack` |

## 3. The gate

**Every one of part 4's 23 red ids is green, and nothing else is red.** Run by id first: the 22 non-sweep ids passed,
105 subtests; the optimizer sweep passed in 2:14. Then the full suite.

| Step | Result |
| --- | --- |
| suite, `pytest tests/ -v -n 2` (run directly, for the node log; `scratchpad/m2impl3/suite5_v.log`) | **1,954 passed, 5 skipped, 1 xfailed, 92,096 subtests passed, 0 failed**, 39:43 |
| module doctests (`run_gate.sh`) | PASS, 325 passed |
| `getting_started.rst` | PASS, 0 failed examples |
| pages (47 files) | PASS, 0 failed examples |
| examples | PASS, 14 examples, 0 failed |
| Sphinx `-W` | PASS, 0 warning or error lines, 19:20 (its `sphinx_html` and `sphinx_doctrees` deleted) |
| slow tier (`T3TOOLBOX_SLOW_TESTS=1`, serial; `run_gate.sh slow`) | PASS, 66 passed, 13,173 subtests, 25:07 |

The lowest MemAvailable during the gate's steps was 9.4 GB (Sphinx); the logs are in `scratchpad/m2impl3/gate5/`.

**The node tally against part 1's all-green log** (`m0/tally.py`; kept as `m2/tally_part1_vs_part5.txt`). Part 1's tree
is the last all-green tree of M2, equal to pre-M2 in `tests/` except part 1's own additions. No file in `t3toolbox/` or
`tests/` is newer than the log. 1,934 nodes there, 1,960 here; 33 differ, all expected:
- 29 new nodes:
  - `test_engine_data_space.py` (part 2): 12;
  - `test_oracle_data_space.py` (parts 4 and 5): 11;
  - part 1's round-1 refusal test: 1;
  - part 3's new and renamed tests in `test_jets` (4) and `test_data_preparation` (1);
- 3 nodes only in part 1's log, each renamed or replaced in part 3:
  - `test_the_block_norms_have_the_shape_of_the_weight` → `…_the_block_shape` (27 subtests both);
  - `test_check_weight_passes_the_exact_shape_and_none` → `…_the_full_form_and_none`;
  - `test_a_backend_function_given_a_bad_weight_checks_nothing` → `test_a_converter_refuses_…`;
- 1 node in both with more subtests: part 3's factorials by hand, `test_the_weight_goes_the_opposite_way_on_its_jet_axes`,
  11 → 58.

**Every other node, the converted ones included, has exactly part 1's outcome and subtest count**, so no conversion
dropped a comparison.

**The battery** (`implementation/m2/battery.py --compare`, the expectations as ruled in the design review's finding 2):
- **AS EXPECTED** over all 25,476 arrays:
  - 24,427 bitwise;
  - 956 to rounding (`f`, `Jp_norm_sq`, `f_resolution`), largest relative error 3.6e-16, against 1e-13;
  - 39 jitted, 1.0e-16, against 1e-12;
  - 11 optimizer counts and stop reasons exact;
  - 43 optimizer losses and iterates, 1.9e-16, against 1e-10.
- The `mc_sgd` runs were changed first (§4).
- The old record (`scratchpad/m2impl3/battery_old5.npz`) was made by the same script on the pre-M2 base tree. It equals
  part 1's record bitwise in every array but the 20 of the two changed `mc_sgd` runs. The new record is
  `battery_new5.npz`.

**The diff audit** (`m2/diff_audit.py`, output `m2/diff_audit_85c6e54d.txt`): every changed line of the 25
pre-existing test files modified since `85c6e54d` (the three new test files are additions). A line is flagged when its
removed or added side touches an assertion or a tolerance; 129 flagged lines, all read.
- **No tolerance changed anywhere.** The two flagged lines carrying tolerances keep `1e-5 * 1e3`, and a new line is new
  code (the compile case).
- **The assertion lines that changed are**:
  - part 3's re-ruled checker tables and the dual-jet refusals (`test_jets`, 38 removed, 50 added, reviewed there);
    spot-checked that the removed behaviours are still asserted (the `0 / 0` NaN, the entries refusal, the float32
    dtype);
  - part 1's block-shape assertions (`test_sampling_definitions`, 2 lines: "the weight's shape" became the block shape);
  - M1's re-pinned bridge (`test_data_metric`);
  - this part's four translation expectations (the same blocks placed with `blocks_to_weight`) and one re-wrapped
    `assertAlmostEqual` (`test_optimizers.py`, the element model passed in its place).
- Every other changed line builds or passes `y` or the weight, or is a helper's docstring.

## 4. Departures and choices

- **The battery's `mc_sgd` runs start at a random point, not the zero tensor.** With the original runs,
  `--compare` reported 8 arrays UNEXPECTED, all `mc_sgd` cores (relative differences up to 2).
  - The losses agreed at every step to `4e-16`, the counts and stop reasons exactly. The dense tensors the cores
    represent agreed to `4.3e-15`.
  - So no decision of the run moved, and no tensor did; the cores' gauge did. **Corrected by the part-5 review:** the
    cause is not the zero start. A rounding-level change of the step can flip the sign gauge of the SVD retraction
    from any start (the reviewer saw it in 4 of 8 derivative-weighted `mc_sgd` runs from generic random starts,
    tensors agreeing to 1.1e-14); the re-seeded battery run happens not to flip.
  - Per the ruling (a run that decides at rounding level is changed, not loosened), the run was changed. The
    expectation (cores to `1e-10`) was not relaxed, and comparing dense tensors instead was not adopted.
  - The old record was re-recorded with the same script on the pre-M2 base tree (`scratchpad/m2impl/base`, `85c6e54d`).
    It equals part 1's record bitwise everywhere but those two runs (the global numpy state is restored after the
    draw).
  - The Newton runs kept bitwise cores (here and in the reviewer's translation runs).
- **The translation gives the old data size-1 `C` axes on a stacked `x0`**, which is the old meaning (one dataset
  broadcast over the stack). The optimizers reject a stacked point anyway (`NotImplementedError`, after the batch
  check), so today this only keeps that check from failing first with a less helpful `ValueError`.
- **The page's prose (section 4.6) is left as it is**, with its "bare vector means per-order" wording of the old call
  forms. Only its example's code moved to the new form. The prose is P19's, the documentation slice.

## 5. M2 as a whole, and what follows

**M2 in one paragraph.** Every data-space object (`y`, the weight, a residual) is of the type of `z`: each axis 1 or
`z`'s size, with the model's stack `C`, never `K`. The parts:
- part 1: the reference module and the dense oracle read the rule, pinned against brute force;
- part 2: the engine reads it and checks it structurally on entry;
- part 3: the checkers, the dual-jet converters and the derivative conversions;
- part 4: the oracle's batch: `check_batch` with the stack, `take`, packing, and `f_resolution` as `<y, M y>` over `z`'s
  shape;
- part 5: the frontend and the old forms.

The weighted numbers are the pre-M2 ones (the battery). Per-element stacked data and per-sample weights, impossible
before, are tested against independent oracles: a zero weight deletes a sample, stacked data are separate fits, and a
drawn batch is unbiased under a partly broadcast weight.

**For the squash and what follows.**
- The 123 → 23 → 0 red trail is in `M2_part3_EXPECTED_RED.txt` and `M2_part4_EXPECTED_RED.txt`, with
  `m2/part4_green_since_part3.txt`.
- The tallies are in `m2/`: `tally_part1_vs_part2`, `tally_part2_vs_part3`, `tally_part3_vs_part4`,
  `tally_part1_vs_part5`.
- **M3** (the block sums and the display): `y_block_sumsq` is still the plain block sum over `y`'s own shape. With a `y`
  that has size-1 axes it undercounts against `r_block_sumsq` (decision 13(c)), and the display's ratio inherits it.
  The display's `(unwt …)` figure is untouched. Both are M3's, as planned.
- **P17** deletes the translation; `_shared_by_the_stack` and the `blocks_to_weight` calls in `_old_fit` go with it.
  The tests' `_fits.fit` / `derivative_fit` keep the block vocabulary (a test convenience, documented there).
- **P19** (the documentation slice): `fitting_and_optimization.md` section 4.6's prose still describes the old weight
  forms (only its example's code moved).
- The battery's records (about 20 MB each) stay in the scratch folder, `scratchpad/m2impl3/battery_{old5,new5}.npz`.
  They are regenerable bitwise with the committed script, from `85c6e54d` and this tree.
- Nothing is left running. The scratch copies are deleted.
