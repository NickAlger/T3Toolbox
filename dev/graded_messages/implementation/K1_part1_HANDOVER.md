# K1 part 1 handover: the independent references

Implementer (the third), 2026-11-26, on the local branch `k1-wip` (head `78a34d77`, part 0 committed), uncommitted.
Part 1 of `K1_DESIGN_NOTE.md` (§2.2 part 1, §4.1, §6.1 to §6.3, §7.4 (c)) with `k1/prose_edits.md` §6's texts, round
2's finding 12, and the orchestrator's four points of 00:35 (the reviewer's first look). No engine code changes.

**Status:** complete; suite green; the gate's lines are copied from the logs in §3.

**Methods traced first** (the maintainer's instruction): `TuckerTensorTrain.probe` (`tucker_tensor_train.py:3660`) →
`probing.t3_probe` (`backend/probing.py:46`) → `sampling_variables.sampling_variables` (`:1678`), its
`_check_data_space` (`:1172`, the engine's data-space rule, today's "1 or full"); `probe_corewise_transpose`
(`:3817` → `t3_probe_corewise_transpose`, `probing.py:260`, the engine with a held `Jp`); `__add__` (`:1949`); and the
weight's path: `jets.data_metric` (`jets.py:579`), `_omega_squared`, `_check_pair`, `blocks_to_weight` (`:767`). Then
the reference whole where it reads `y` and the weight (`_r`, `_modes_of_data`, `_check_data_space`,
`_weight_squared`, `_dual_jet`, `_the_seed`, and the bodies of `w_tilde`, `JtJp`, `f`, `r_block_sumsq`,
`Jp_norm_sq`), `tests/oracle_sweeps/oracle.py`, `tests/test_dense_oracle.py`, and the tests that compare the engine
with the reference.

## 1. What changed

### Library (1 module): `t3toolbox/backend/sampling_definitions.py`

- **The 37 public functions:** `weight=None` becomes `block_weight=None, full_weight=None` at the same place of the
  keyword-only group; `y`'s comment says "z's shape exactly". Scripted (`k1/part1_reference_rewrite.py`: whole-line
  replacements, refusing a file whose counts are not 37 / 37 / 17); the 17 `_the_seed(...)` calls pass both weights.
  The block weight's comment is the note's §4.1 text with "None = ones" on a `#   ` continuation line under the
  comment column (the orchestrator's point 4: the one-line form is 124 columns), the same in all 37.
- **The private helpers, by hand** (`prose_edits.md` §6.7):
  - `_r`: `y` is checked exact (`may_be_one=()`).
  - `_check_data_space(what, i, shape, of_shape, may_be_one)`: each axis `z`'s size, 1 allowed only at
    `may_be_one`; the message says which rule (exact, or "each sample axis 1 or its size, the jet axes and the mode
    edge exactly").
  - `_check_block_weight(S, block_weight)` (new): exactly `(d,) + F` / `F` / `()`; a tuple or a list is a `TypeError`.
  - `_weight_squared(S, block_weight, full_weight, i, shape)`: the mode's block row squared, reshaped onto the
    leading jet axes (`row.shape + (1,) * (len(shape) - m)`), times the full weight squared, read by position after
    `_check_data_space(..., may_be_one = the sample axes W)`; `1.0` for a factor that is `None`. Its own code: no
    import of `jets` (the module's structural test, `test_the_module_calls_nothing_of_the_library_but_the_dense_functions`,
    still passes).
  - `_dual_jet(S, jet, block_weight, full_weight)`, `_the_seed(S, sample, y, block_weight, full_weight, w_tilde, name)`.
  - The bodies of `JtJp`, `f`, `r_block_sumsq`, `Jp_norm_sq` pass both weights (calls wrapped by hand, no reflow).
- **Docstrings** exactly as `prose_edits.md` §6.1 to §6.6: the module docstring's seed paragraph and its argument list
  (`y`, `block_weight`, `full_weight`, the data-space paragraph); `w_tilde`'s formula and example (`block = 0.5 +
  rng.random((3, 3))`); the six tilde examples (`block_weight=0.5 + rng.random(3)`); `JtJp`'s first line and its
  written-out full weight; `f`'s first line, its full-weight example and the refusal of a per-mode value given as a
  full weight beside the block form; `r_block_sumsq`'s first line, the full weight per sample written out, and the
  new block example; `Jp_norm_sq`'s first line and example. Doctest outputs: the run reproduced every expected line as
  it stands (shapes and `True`), so nothing needed pasting beyond what is there; the module doctests pass (§3).
  Where a first line grew past the column (`f`, `r_block_sumsq`, `Jp_norm_sq`, `JtJp`, the module's seed formula),
  only the lines of that sentence's paragraph that the new words push were re-wrapped; every other line is byte for
  byte (`k1/part1_kept_text.txt`, §3).

### Tests (9 files)

- `tests/oracle_sweeps/oracle.py`: additive `weighted_residual_two(kind, Sx, y, block_weight, full_weight)`,
  `weighted_two(kind, like, value, block_weight, full_weight)` and their helper `_block_factor` (the block row placed
  by a loop over the slots; the full weight and `y` by `expand`, `np.repeat`). The full-form functions stay.
- `tests/test_dense_oracle.py`: new `TestTwoWeights` (2 tests) and the module docstring's item 4.
- `tests/test_sampling_definitions.py`:
  - `Problem.full_weight` (the old placed weight) became `Problem.block_weight` (the blocks); `Problem.arguments`
    passes `block_weight=`; `Problem.data` (the engine's old weight, `blocks_to_weight` of the same blocks) stays.
  - `TestOutputsAgainstDense` (3 places) and `TestKp.test_Kp_is_the_second_derivative_of_the_dense_objective`
    compute their expectations with `P.data(kind, case)[1]` (the placed table) where they read `args['weight']`;
    `TestJetsTwoWays.test_a_second_variable_of_order_zero_is_one_variable`, `TestKp.jax_problem` read
    `block_weight`; `TestStructure.test_every_function_has_the_same_arguments` expects the two keywords.
  - `TestTheWeightIsOfTheTypeOfZ` became **`TestTheTwoWeights`** (its four tests keep their names; one test added).
- `tests/test_engine_data_space.py`: `TestAgainstTheReference` rewritten (finding 12, §2 below); module docstring
  item 0 extended by one line.
- `tests/test_sampling_variables.py` (`reference`), `test_sampling_corewise.py` (`reference`),
  `test_sampling_apply.py` (`TestByDefinition.check`), `test_sampling_entries.py` (`TestByDefinition.test_every_name`):
  the reference gets `block_weight=problem.block_weight(...)`, the engine `problem.data(...)`'s
  `blocks_to_weight` of the same blocks, unchanged.
- `tests/test_sampling_stacks.py` (`TestTheReferenceAnchor`): `full_weight=weight_c` (the case's weight is per entry,
  `box + W + C + (N_i,)`, exact).
- The other reference-importing files (`test_block_sums`, `test_sampling_uniform`, `_identities`, `_graph`,
  `test_jax_absent`) pass the reference no weight: unchanged.

### Records (`dev/graded_messages/implementation/k1/`)

`diff_audit.py` (audit (c), `--scope-map`, `--files`, `subTest` labels skipped), `audit_c_planted.py`,
`full_weight_reasons.tsv` (40 lines) and the script that writes it from a hand table, `part1_reasons.py`;
`part1_reference_rewrite.py`; `part1_mutations.py` and `part1_mutations.txt`; `part1_audit.txt`;
`part1_kept_text.txt`; `tally.py --part=1`; `part1_suite_v.log.gz`, `tally_part0_vs_part1.txt`.

## 2. The tests added or rewritten, and what each establishes

Every oracle below is loops over dense arrays (`np.ndindex`) or the old engine; none calls a function of the
reference (the orchestrator's point 1).

- **`TestTheTwoWeights.test_against_brute_force`** (the pins of §2.2 part 1): problems `distinct` (new: `F = (4,)`,
  `W = (2,)`, `d = 3`, `N = (5, 6, 7)` at the line `(3,)`: every size different), `small` (at the curve `(2,)`:
  `F = W = (3,)`, `d = N_1 = 3`, all coinciding), `tails` (`W = (2, 2)`), `two_sites` (`W = ()`); jet cases none,
  line `(3,)`, plane `(1, 2)`, curve `(2,)`; three kinds; six patterns: neither, a block weight alone (random:
  asymmetric on every jet axis; a Python float for entries), a full weight per entry, per slot written out on `W`
  and the edge (`np.broadcast_to`), with its `W` axes 1, both (a block weight and a full weight with `W` of 1). `y`
  random, exact. Checked against index loops: `w_tilde` (every entry, the turned slot), `f`, `r_block_sumsq` (per
  block), `Jp_norm_sq`, `<g, p> = Σ b² f² r Jp(p)` and `<JtJp(p), q> = Σ b² f² Jp(p) Jp(q)` (pairings by loops, not
  through `d/dt f`: point 1).
- **`TestTheTwoWeights.test_every_name_that_reads_the_seed_reads_both_weights`** (new; point 2): each of the 17
  functions that build the seed (`w_tilde`, the six tilde, the four tilde-dot, `g`, `Kp`, `V_tilde`, `H_tilde`,
  `V_tilde_dot`, `H_tilde_dot`) with `(y, block_weight, full_weight)` equals the same function at `w_tilde=` a seed
  written by loops (`b² f² r` at the turned slot); probe at a line and a plane, apply, entries. The remaining weight
  readers are `f`, `r_block_sumsq`, `Jp_norm_sq` and `JtJp`, each pinned by the brute force above with both weights.
- **`TestTheTwoWeights.test_a_weight_of_another_form_raises`** (rewritten): block weights of another rank, with a
  slot more, the old placed form (`ValueError`), a list (`TypeError`); full weights of another rank, an axis neither
  1 nor its size, one per sample with the rest 1, an edge of 1, one per mode `(d,) + (1,)*…`, a mode axis of 1,
  `d - 1` arrays, a tuple for apply/entries, a jet axis of 1 (`ValueError`); the right forms pass; a tuple and a list
  are one object; the plan's E3b coincidence: a `(d,)` block at `order=(n,)`, `d = n + 1`, refused for probe, legal
  (per slot) for apply.
- **`…test_an_object_larger_than_z_where_z_has_a_size_of_one_raises`**: `weight=` → `full_weight=`, unchanged
  otherwise.
- **`…test_data_of_another_form_raises`**: plus `y` constant in `W` (a size-1 `W`), an edge of 1, a jet axis of 1.
- **`test_dense_oracle.TestTwoWeights`**: `weighted_residual_two` and `weighted_two` against a loop over every entry
  (neither, block, full per entry, full with `W` of 1, both; three kinds; `order` `()` and `(2,)`); a block row of the
  old placed shape refused.
- **`test_engine_data_space.TestAgainstTheReference`** (finding 12): M2's patterns, as pairs. Per entry → full; per
  sample → full written out (values identical); per slot → block; one per mode → block (constant along `F`); the
  edge and the first jet axis → full written out; plus "with `W` of 1" (a legal full weight) and **"a block and a full
  weight"** (point 3: the reference gets `(b, f)`, the old engine the one weight `blocks_to_weight(b) * f`, mode by
  mode; agreement to `defs.TOL`, as every row). `y` constant in `W` is written out. The engine gets the same
  written-out `y` and `blocks_to_weight(block)`, the full weight, or the product. Same subjects, tolerance and
  assertions as before.
- The four `TestByDefinition`s and the stacks anchor: the old engine (on `blocks_to_weight(b)` or the per-entry
  weight) pins the reference's new reading of the block weight / full weight; same tolerances.

## 3. The gate (part 1's: the suite, the reference's doctests, the audits, the mutations)

- **The suite, `-v`, `-n 2`** (`k1/part1_suite_v.log.gz`): `1890 passed, 5 skipped, 1 xfailed, 1 warning, 90398
  subtests passed in 1971.50s (0:32:51)`; no failure, no failed subtest. MemAvailable seen during the run: 10.0 GB
  free of 13.9 GB at 30% (one reading; not a minimum).
- **Tally** against part 0's log (`k1/tally_part0_vs_part1.txt`, `tally.py --part=1`): 1,893 → 1,896 nodes, 10
  differ, every one accounted for:
  - new: `test_dense_oracle.TestTwoWeights` ×2, `TestTheTwoWeights.test_every_name_that_reads_the_seed_reads_both_weights`;
  - more subtests in the rewritten tests: `TestTheTwoWeights.test_against_brute_force` 135 → 216,
    `test_a_weight_of_another_form_raises` 16 → 30, `test_data_of_another_form_raises` 4 → 10,
    `test_engine_data_space.TestAgainstTheReference.test_the_engine_is_the_reference` 135 → 189 (two patterns more);
  - not part 1's: `backend/test_optimizers.TestOracleSeam.test_optional_outputs_are_checked_where_needed` 6 → 8
    subtests and `test_least_squares_oracle`'s `test_reachable_from_the_frontend` under `TestTheOracle` instead of
    `TestTheBatchConstructor`: both are part 0's post-review fixes (`K_ARC_DECISIONS.md` 32; the carry-over's
    "one node had moved class by mistake: fixed after the run"), committed in `78a34d77` after part 0's log was taken.
- **Module doctests of the reference:** `pytest --doctest-modules t3toolbox/backend/sampling_definitions.py`:
  37 passed (`k1/part1_doctests.txt`, rerun after the continuation-line edit).
- **Targeted run** before the full suite (the 12 files that import the reference and `test_dense_oracle.py`): 139
  passed, 15,992 subtests, 0 failed; `test_sampling_definitions.py` alone 71 passed, 5,773 subtests; then the law
  test: 1 passed, 68 subtests.
- **Audit (c)** (`k1/part1_audit.txt`, `diff_audit.py HEAD --part 1 --scope-map
  TestTheWeightIsOfTheTypeOfZ=TestTheTwoWeights`): PASS. W-BLOCK 21 (every old `weight=` to the reference that became
  `block_weight=`: each was `blocks_to_weight` of the blocks now passed, or a per-mode/per-slot table), W-FULL 9,
  B-PAIRED 1, B-REASON 25, F-REASON 15, W-GONE 1; no `blocks_to_weight` call removed (part 1 keeps the engine on one
  weight). The planted test (`audit_c_planted.py`): a right conversion passes; a block table passed as `full_weight=`,
  a block weight of another value, and a removed table each fail.
  - **The audit's blind spots, as they occur in part 1** (the orchestrator's list): *a weight dropped altogether*
    shows as W-GONE and passes: one site, `test_engine_data_space.TestAgainstTheReference.test_the_engine_is_the_reference`,
    where `dict(weight=w_arg)` moved into the helper `weights` (its `dict(block_weight=…, full_weight=…)` is seen
    there, B-PAIRED / F-REASON) and reaches the reference as `**reference`, an invisible keyword: read by hand.
    *String keys*: `args['block_weight']` / `flat_args['block_weight']` read back the dict `Problem.arguments` builds
    (seen there); no `**{'full_weight': …}` in the diff. *Bare callee*: the reason lines name `f`, `getattr(sdef,
    name)` etc.; within a scope each callee's sites are listed by line in the audit file.
- **Line audit (a):** no tolerance changed (no `TOL`/`tol` assignment in the diff; every tolerance argument of a
  changed line is in a new or rewritten test). One removed assertion line, in the refusal test the note's §6.2 lists
  (`assertEqual(f(weight=list(full)), f(weight=full))` → the same with `full_weight=tuple(full)` / `full`).
- **Mutations** (`k1/part1_mutations.txt`; the note's §6.4 has no row for part 1, so these are the reference's twins
  of part 2's rows and its own rules), each in a scratch copy, `python -B`, all **CAUGHT**: the block weight
  unsquared; the block row right-aligned; the weights applied after the turn; the full weight refused a `W` of 1;
  the full weight allowed a structural 1; `y` allowed a `W` of 1; the block weight unchecked; the block weight
  unsquared seen by the old engine through `TestByDefinition` ×2 and `TestAgainstTheReference` (420 failures); the
  oracle's block row on the wrong slot; `nu_tilde` swapping the two weights in its own call; `V_tilde_dot` dropping
  the full weight (both caught only by the new law test).
- **Kept text** (`k1/part1_kept_text.txt`, `kept_text_check.py HEAD`): `sampling_definitions.py` 69 unchanged, 43
  changed (the module docstring, `_r`, `_check_data_space`, `_weight_squared`, `_dual_jet`, `_the_seed`, the 37
  public functions), 1 added (`_check_block_weight`), 0 removed. In the 37, outside the signature lines and the
  seed calls, the changed lines are the docstrings and bodies listed in §1.

## 4. Departures from the note, and choices

1. The block weight's signature comment on two lines (`#   None = ones`), the orchestrator's point 4.
2. `TestTheWeightIsOfTheTypeOfZ` → `TestTheTwoWeights` (its subject is no longer "of the type of z"); the four test
   names kept, so the tally maps them (`tally.py --part=1`).
3. `Problem.full_weight` → `Problem.block_weight` (the old name returned the placed table, which the reference now
   refuses; four callers in other files).
4. The pin of `g` is the pairing by loops, not `d/dt f` (point 1: `f` and `g` share `_weight_squared`).
5. `TestAgainstTheReference` gained two patterns ("with `W` of 1", "a block and a full weight"); the note's §6.2
   "plus the refusal of each old structural size-1 pattern, once" is the reference's refusal table
   (`TestTheTwoWeights`); the engine's refusals are part 2's.
6. The oracle's two-weight functions read the jet count `m` off the block weight (`ndim - 1` for probe); with no block
   weight they need none.
7. `TestOutputsAgainstDense` and the second-derivative test of `Kp` compute their expected values with the placed
   table `P.data(kind, case)[1]` (`jets.blocks_to_weight`, pinned in `test_jets`/`test_data_metric`): the reference
   shares no code with it; part 5, which deletes `blocks_to_weight`, must give `Problem.data` and these three a
   placement of their own (listed in §5).
8. The mutations of part 1 are my own list (the note has none for part 1).

## 4b. `typ.Any`s

The only library module part 1 touches is `backend/sampling_definitions.py`, which is exempt (`K_BRIEF.md` §5).

## 5. For the orchestrator, and outside the part

- **For part 5:** `tests/test_sampling_definitions.py`'s `Problem.data` and the three expectations of §4.7 use
  `jets.blocks_to_weight`; once part 2 moves the engine to the two weights, `Problem.data`'s second element is only
  those expectations' and can become a loop placement (or go). The audit's scan of part 5 would otherwise find them.
- The note's §6.1 lists `test_dense_oracle.py` under "the reference's two weights": it pins the oracle, not the
  reference; part 1 adds the oracle's two-weight pins there.
- `test_block_sums.py`'s engine half passes the reference no weight (its weights go through the oracle's full-form
  functions), so part 1 changes nothing there.
