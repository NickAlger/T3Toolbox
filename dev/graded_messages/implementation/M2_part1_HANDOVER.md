# M2 part 1 handover: the independent oracles, and the pre-M2 battery

Implementer, 2026-11-22, on `m2-wip` at `cec3afbe`. Built on `M2_DESIGN_NOTE.md`, `M2_DESIGN_REVIEW.md` and Claude's
rulings. Nothing committed.

**Scope.** The reference module's full-form reading, pinned against brute force; `oracle.py` extended and pinned;
the battery recorded on the untouched tree. No line of the engine, `jets`, the fitting layer or the frontend changed.
The reference module, `t3toolbox/backend/sampling_definitions.py`, does sit in the package, but nothing in the
library calls it (it imports only the three dense functions; `TestStructure` pins that). I read "change no library
code in part 1" as "no code the library runs", since the plan's part 1 is the reference's own switch.

## 1. What changed

**`t3toolbox/backend/sampling_definitions.py`** (the reference).
- `_weight_squared(S, weight, i, shape)`: the weight of the type of `z`, read by position with no reshape. For probe,
  exactly `d` arrays: a tuple, a list, or one array whose first axis is the mode (`_modes_of_data`; a mode axis of 1
  raises). Each array checked by `_check_data_space`: the rank of `z`'s array, each axis 1 or `z`'s size. Returns
  `omega**2`.
- `_r` reads `y` by the same rule, with the same check. So a data or a weight of another form (the old block form, a
  wrong rank, a wrong size, a missing `W`) raises a `ValueError` naming it, instead of being right-aligned.
- Docs: the module text (the data-space rule, with what a size of 1 means), the 37 signature comments of `weight`,
  `w_tilde`'s formula, `r_block_sumsq` "the block shape" (design decision 6).
- Doctests: the weighted examples moved to the full form (per mode and slot, per mode and sample, per entry); `f`
  gains the gotcha (a `(d,)` weight raises) and the right form, `(d,) + (1,) + (1,)`. 37 doctests pass.
- No import of `jets` (the independence test).

**`tests/oracle_sweeps/oracle.py`**, additive: `expand` (size-1 axes written out by `np.repeat`, no broadcasting;
any other shape raises), `S_taylor` (Taylor coordinates: plain, or along a line `(main, pp)` at `(n,)` as the
`_poly_derivs` derivatives over `t!`), `weighted_residual_full`, `misfit_full`. The old-form functions are unchanged
(the sweeps use them).

**Tests.**
- `tests/test_dense_oracle.py` (new, 7 tests, 91 subtests): pins the new `oracle.py` with no library call.
  - `expand` against an explicit loop, and its refusals.
  - `S_taylor` along a line against `S` evaluated on the line, `sum_t c_t s^t` at two points off the nodes (order 3,
    the whole cubic), and the truncations at orders 1, 2 as prefixes of order 3; `order = ()` is `S`.
  - `misfit_full` against an explicit `np.ndindex` loop for weights per entry, per sample, per slot, one per mode,
    on the edge and one slot axis, `y` full or constant in `W`, `None`; a `d`-leading array weight equal to the list.
  - The old form bridged: `misfit(kind_derivatives, w)` equals `misfit_full` with `omega_t = w_t t!` on the Taylor
    data, for probe, apply and entries (as apply at the one-hot vectors).
  - Block weights and a wrong mode count raise.
- `tests/test_sampling_definitions.py`:
  - **New `TestTheWeightIsOfTheTypeOfZ`** (3 tests, 155 subtests), the brute-force pin of gate (2). `z` and `Jp` are
    the dense tensors sampled by the tests' own route (`sample_dense_jet`); the residual, the seed `w_tilde`, `f`,
    `Jp_norm_sq`, `<g, p> = sum omega**2 r Jp(p)` and `<JtJp(p), q> = sum omega**2 Jp(p) Jp(q)` are explicit loops
    over every entry, the weight and the data read at index 0 on their size-1 axes (`at`), with no broadcasting.
    Problems `small`, `tails` (`W = (2, 2)`), `W = ()`; the three kinds; jets none, a line at `(3,)`, two directions
    at `(1, 2)`, a curve at `(2,)` (on `small`, `F = W = N_1 = (3,)`: sizes equal; the line: distinct). Five weight
    patterns; the data constant in `W` on alternate patterns; probe weights alternately as a list and as one
    mode-leading array. Plus the refusals of a weight (the block form, another rank, an axis neither 1 nor its size,
    a mode axis of 1, `d - 1` arrays, a tuple for apply/entries) and of the data (a missing `W` axis, a mode axis
    of 1), each checked by script to raise for the reason named, not elsewhere. Round 1 added the refusals where
    `z`'s own axis is 1 (a one-sample batch, the jet axis at order `(0,)`) and the object's is 3, for the weight and
    `y`, probe and apply (8 subtests; the reviewer's mutant R1 now fails 8).
  - `Problem.data` still returns the blocks (the engine's form until part 2); new `Problem.full_weight` (through
    `blocks_to_weight`); `Problem.arguments`, the reference's arguments, now carry the full form. `Jets.weighted` reads
    the full form positionally (asserting equal rank). Three lines adapted: the block-sum shape (now the block shape,
    not the weight's), the order `(3, 0)` case (the size-1 jet axis inserted after the mode axis), and the old-kernels
    test (the full form to the reference, the blocks kept for the old kernel's `forward`).
- **Engine against reference** (`test_sampling_variables`, `_apply`, `_entries`, `_corewise`, `_stacks`): the
  reference gets `Problem.full_weight` (or `blocks_to_weight`), the engine keeps the blocks. So the old engine now
  also pins the new reading, over the jet matrix. A mechanical change: only the `weight=` argument of the reference
  calls (and the one import in `_stacks`).

**`dev/graded_messages/implementation/m2/battery.py`** (new). See section 3.

## 2. Mutants: 9 of 9 killed

On a scratch copy of `t3toolbox/` and `tests/`, against `TestTheWeightIsOfTheTypeOfZ`, `TestOutputsAgainstDense`
and `test_dense_oracle.py` (script `scratchpad/m2impl/mutants_p1.py`; the copy deleted).

| Mutant | Result |
| --- | --- |
| reference: `omega` not squared | killed (135 subtests) |
| reference: mode 0's weight for every mode | killed (60) |
| reference: the weight not turned with the residual | killed (54) |
| reference: the weight's check removed | killed (the refusals) |
| reference: the data's check removed | killed (the refusals: a missing `W` axis otherwise broadcasts silently) |
| reference: `y` read reversed along its last axis | killed (115) |
| oracle: no factorial in `S_taylor` | killed |
| oracle: the weight applied to `y` only | killed (50) |
| oracle: `expand` repeating the wrong axis | killed |

## 3. The battery (gate (1)), recorded

`battery.py OUT old` on a `git archive` of `85c6e54d` (`t3toolbox/` only, in `scratchpad/m2impl/base/`):
**25,476 arrays** in 28 s (`scratchpad/m2impl/battery_old.npz`, 20 MB; re-recorded in round 1 with the `mc_sgd`
runs' loss history, `check_every=1`): the engine 23,528, the oracle 1,804, derivative data 51, the optimizer runs 54,
the jitted run 39. Run twice: `--compare` gives every array bitwise equal,
so the record is deterministic and can be regenerated from the commit if the scratch folder is lost.

The expectations are those of the review's finding 2, coded in `--compare`: bitwise for everything but `f`,
`Jp_norm_sq`, `f_resolution` (rtol 1e-13), the jitted run (1e-12), and the optimizer runs (iteration counts and stop
reasons exact: the three Newton runs stop at `max_newton` after 3 steps, the two `mc_sgd` runs after 4; losses
(for `mc_sgd`, every step's) and iterates 1e-10). The `new` form adapters (`weight_form`, `y_form`, `taylor_weight`, `full`) are written now and run
first in part 5. Every stored weight is normalized to the full form, so the two runs have the same keys.

## 4. The gate

| Step | Result |
| --- | --- |
| suite, `pytest tests/ -v -n 2` (run directly, for the per-node log) | **1,928 passed, 5 skipped, 1 xfailed, 91,525 subtests, exit 0**, 35:21. The one warning is the known `adam on a MANIFOLD` warning of the sweeps, also in M1's log |
| module doctests of `sampling_definitions` (the only library file changed) | 37 passed |
| the touched test files alone, at `-n 2`, before the suite | 160 passed, 17,938 subtests |

No page, example or other module changed, so the other gate steps were not rerun.

**The tally against M1's `-v` log** (`m0/tally.py`; the output kept as `m2/tally_m1_vs_part1.txt`). The trees
`c16aa640` and `85c6e54d` have the same `t3toolbox/` and `tests/`, so M1's log is the pre-M2 log. 11 nodes differ:
- the 7 new nodes of `test_dense_oracle.py` and the 3 of `TestTheWeightIsOfTheTypeOfZ`;
- `test_data_metric.py::...::test_through_the_metric_it_is_the_old_block_weighting_bitwise`, 10 to 40 subtests. This is
  not part 1's: M1's log was taken before M1's review round 1 widened that test to the four precision pairings
  (`M1_RESPONSE_r1.md`), so the committed tree already has 40.

Every converted engine-against-reference node keeps its outcome and its subtest count, so no comparison was dropped.
Nothing is left running.

## 5. Departures and choices

- The reference module is changed in part 1 (see Scope).
- `_check_data_space` and `_modes_of_data` are the reference's own (private) code, so the independence holds: they
  are not the engine's coming check.
- The pins of `oracle.py` live in a new file, `tests/test_dense_oracle.py`, beside the oracle's other users.
- The battery's pre-M2 record stays in the scratch folder, not in the repository (20 MB); it is regenerable bitwise.
- The design review's expected-red rule starts with part 2: part 1 is green.
