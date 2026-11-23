# M2 part 2 handover: the engine on the rule

Two implementers, 2026-11-22, on `m2-wip`. The first built the library change and most of the test conversions and
stopped at a clean point (checkpoint commit `fe7c7250`, unreviewed); the second finished the part (this tree, on top of
`e782fd79`). Built on `M2_DESIGN_NOTE.md` (part 2), `M2_DESIGN_REVIEW.md` and decision 13 of `M_ARC_DECISIONS.md`.
Nothing committed by either.

**Scope.** The engine (`sampling_steps`, `sampling_variables`) reads `y` and the weight `omega` by the rule of plan §2:
each of the type of `z`, every axis 1 or `z`'s size, never `K`. Every engine-level test moved to that form. `jets`
(part 3), the fitting layer (part 4) and the frontend and old forms (part 5) are not switched: whatever reaches the
engine through them with a block weight, or with a stacked `y` without `C`, is red, and is listed in
`M2_part2_EXPECTED_RED.txt`.

## 1. What changed

**`t3toolbox/backend/sampling_steps.py`.**
- New `_modes` (pairs each array of a data-space value with its weight's array: a tuple, a list, or a mode-leading
  array) and `_with_K` (`lens.K` size-1 axes inserted at `lead + m + len(W)`, `lead` 1 for a mode-leading array, 0
  otherwise: the weight placed against `Jp` and the adjoint it seeds; `K` is the engine's to place, never the
  caller's).
- New `_weighted_sum`: `sum value * data_metric(value, omega, None)` over the jet axes, `W` and the edge, keeping
  `[K] + C`.
- `r_step` is `z - y` by position (the old alignment of a `y` without `C` is gone).
- `f_step` and `Jp_norm_sq_step` go through `_weighted_sum`; `w_tilde_step` is `_flipped(data_metric(r, omega_K,
  None))` per mode, no longer `jet_to_dual_jet`.
- `r_block_sumsq`'s comment: "the block shape" (design decision 6).

**`t3toolbox/backend/sampling_variables.py`.**
- New `_check_data_space` (correction (a)): `y` and the weight of the type of `z`: for probe exactly `d` arrays (a
  tuple, a list, or one array whose first axis is the mode, of length `d`), each of `z`'s rank, each axis 1 or `z`'s
  size; the edge `N_i` ragged, `N` packed; a tuple for apply or entries raises. Run on the objects as given, before
  any packing, after `_check_stacks`. Static, so it runs under `jit`.
- On the uniform layout the probe `y` and weight are packed by `ut3_operations._pack_data_space`.
- Masks (correction (b)): the probe `y` always (an edge-1 `y` is broadcast to `N` with zero padding by its mask); the
  probe weight only when its edge is `N`; apply's and entries' never (no edge). `'weight'` added to
  `_UNIFORM_MASKS`.
- **Fixed in this round:** the weight's mask test read `np.shape(None)[-1]` when the weight was `None` (an
  `IndexError` on every uniform probe call with `y` and no weight). The guard now skips `None`.
- Docs of `y`, the weight, the masks, the check and the block shape.

**`t3toolbox/backend/ut3_operations.py`.** New private `_pack_data_space(value, N)`: each axis of `F + W + C` the join
across the modes; the edge kept at 1 if every mode's is 1; otherwise a full edge zero-padded to `N` and an edge of 1
broadcast to `N` (never padded: that would put its value on entry 0 only). A packed object passes through.

## 2. Tests

**Converted to the full form** (by the first implementer; the diff of each changes only how `y` and the weight are
built and passed, never an assertion or a tolerance; I read every hunk with `git diff -U0 HEAD~2`):
- `test_sampling_definitions.Problem`: `data()` returns the full-form weight (the blocks through
  `blocks_to_weight`), `blocks()` the blocks; the refusal test uses `blocks()`. So `test_sampling_variables`,
  `_apply`, `_entries`, `_corewise` (over the jet matrix and the problems, `small` included) pass the full form to the
  engine and the reference alike.
- `test_sampling_identities`, `test_sampling_apply`: the weight used positionally; the block reshapes gone.
- `test_sampling_uniform.UniformCase`: `y` and the weight per entry and per element of `C`, packed with finite
  garbage in the mode padding (probe); the ragged twin gets its element (`at_element`).
- `test_sampling_stacks.Case`: `y` and the weight per entry and per element of `C`; the reference anchor and the
  per-sample blocks take the element.
- The bridge (`tests/bridge/test_old_kernels_bridge.py`): the engine gets `placed(...)` (`blocks_to_weight` on a
  template), the old kernels keep their blocks.

**Converted in this round:**
- `test_contractions_sharding.TestTheEngineShardsOverW`: the weight (per entry, so it carries `W`) is sharded over
  `W` with the sample and `y`, instead of closed over at full `W` (which the check rightly refused inside a shard).
- `test_dispatch`'s five engine rows with a weight (`test_jit_sampling_variables`, `_with_a_stack_of_directions`,
  `_curvature_and_the_corewise_point`, `_the_apply_kind`, `_the_entries_kind`): the same random blocks through
  `blocks_to_weight`. The uniform rows already used `UniformCase`. `test_jax_absent`'s engine calls pass no weight
  and an unstacked `y`, so they are unchanged (its `check_weight` call is part 3's).

**`tests/test_engine_data_space.py`** (new; the first implementer wrote 8 tests, I fixed one and added three):
- `TestAgainstTheReference` (**added**): the engine against the reference module on `small` (where the curve at
  `(2,)` has `F = W = N_1 = (3,)`), `tails` (`W = (2, 2)`) and `W = ()`; three kinds, four jet cases; the weight per
  entry, per sample, per slot, one per mode, and on the edge and the first jet axis; `y` full or constant in `W`
  alternately; probe weights alternately a list and a mode-leading array; a stack of two directions `K` (both
  forms): `f`, `r`, `w_tilde`, `g`, and per element of `K` `Jp`, `Jp_norm_sq`, `JtJp`, `Kp`. 135 subtests. This is the
  design note's "against the reference, over the jet matrix, `omega` in the patterns" at the engine; until now the
  engine met only block-constant weights against the reference (the stacks anchor has per-entry weights, but not the
  jet matrix or the equal-sizes problem).
- `TestTheCheck`: the silent `W = C` case (plain broadcasting is shown to accept the `y` without `C`; the engine
  raises; size-1 `C` passes), three kinds; the other forms (block weight, `K`, a wrong size, a mode axis of 1,
  `d - 1` arrays, a tuple for apply, a `y` larger than `z` where `z` is 1); a packed edge of 1 or `N` passes, `N - 1`
  raises.
- `TestAZeroWeightDeletesASample`: `f`, `g`, `Jp_norm_sq`, `JtJp` with one sample's weight zero equal the batch
  without it; the sample's deletion is visible (with weight 1 `f` changes); three kinds, ragged and uniform.
- `TestStackedDataAreSeparateFits`: `C = (2,)`, `y` and the weight different per element: `f`, `r`, `w_tilde`, `g`
  element by element equal the unstacked call at that element's data; the other element's data give another `f`.
- `TestTheUniformEdgeOfOne` (**fixed and extended**):
  - **Fixed:** it compared the `g` blocks of `UNIFORM_MANIFOLD.frame(ux)` with those of `MANIFOLD.frame(x)`. The two
    frames are gauge-equivalent, not equal (`docs/uniform_equivalence_contract.md`, "Gauge-carrying operations"),
    and `g`'s blocks carry the gauge, so the test failed by about 100% on correct code (`f` agreed). It now packs the
    ragged frame (`UT3Frame.from_t3frame`), and compares the `H` blocks too.
  - **Added `test_mixed_edges`:** a ragged probe `y` and weight with the edge 1 in some modes and `N_i` in others.
    The mutant that zero-pads an edge of 1 when the modes are mixed survived every test before it.
  - **Added `test_a_weight_whose_padding_would_overflow_when_squared`:** a packed weight with an `N` edge whose padding
    holds `1e200` (finite; its square is `inf`, and `inf * 0` is NaN). The engine's weight mask is what keeps the
    result finite. Before it, dropping the weight mask survived every test, and for finite garbage of ordinary size
    the mask is indeed redundant (`r`, `Jp` and a held variable have zero padding, so `omega**2 * 0 = 0`); the
    overflow is the case where it is not.
- `TestTheUniformEdgeOfOne.test_modes_with_different_size_1_axes` (**added in review round 1**): a ragged probe `y` and
  weight whose modes differ in their size-1 axes of `F + W + C` as well as their edges (per slot with a full edge, per
  sample and element with an edge of 1, per entry; rotated over the modes, the weight's pattern never `y`'s), uniform
  against ragged. It pins the join across the modes in `_pack_data_space`: the reviewer's mutant that takes the body
  from mode 0 fails all 3 subtests.
- `TestTheLaws`: the jet law and the input law on `f`, `r`, `w_tilde`, `g`, `Jp`, `Jp_norm_sq`, `JtJp`, `Kp`, the
  weight moved by its own law (`P_star`, `V`).

## 3. Mutants

On a scratch copy of `t3toolbox/` and `tests/` (`scratchpad/m2impl2/mutants_p2.py`, `mut_one.py`; the copy deleted).
The orchestrator's "the weight squared after the cast" is read as its inversion, the square formed before the cast.
"New file" is `test_engine_data_space.py` in full; "engine files" are `test_sampling_uniform`, `_stacks`,
`_variables`, `_identities`, `_apply`, `_entries`, `_corewise` and the bridge, run with `--maxfail=1`.

| Mutant | New file | Engine files |
| --- | --- | --- |
| the data-space check dropped | killed (7 subtests: the `W = C` case, three kinds; a mode axis of 1; `d - 1` arrays; a tuple for apply; larger than `z` where `z` is 1) | |
| the check accepts any size where `z` is 1 | killed (1) | |
| the probe `y` mask dropped | killed (10: the edge-1 and mixed-edge tests) | |
| the weight mask dropped | killed (2: the overflow test; survived before it) | survives (finite garbage of ordinary size: redundant there, see above) |
| a ragged `y` on the uniform layout checked against the padded `N` | killed (2: the zero-weight test, uniform) | |
| `_with_K` before `W` | killed (36: `TestAgainstTheReference`; equivalent where `W = ()` or the weight's `W` is 1) | killed (`TestEquivalence`, `TestTheStackMatrix`) |
| `_with_K` after `C` | survives (no `C` there: equivalent) | killed (`TestTheStackMatrix`, `TestEquivalence`) |
| the square dropped in `_weighted_sum` (`omega`, not `omega**2`) | killed (143) | |
| the weight squared before the cast (in `_omega_squared`) | survives | survives: **equivalent in the engine** (below); killed by `test_data_metric`'s bitwise four-pairing test (8 subtests, `r` float32 with a float64 weight) |
| `w_tilde` unturned | killed (6: the jet law) | |
| mode 0's weight for every mode | killed (7: the laws, stacked = separate) | |
| `r = y - z` | killed (135: against the reference) | killed (the reference anchor) |
| an edge of 1 zero-padded when the modes are mixed | killed (2: `test_mixed_edges`; survived before it) | survives |
| an edge of 1 zero-padded always | killed (10) | |

**Why "squared before the cast" is equivalent in the engine.** The engine casts every floating input, the weight
included, to the call's dtype on entry (the lowest precision among the inputs), and `r`, `Jp` are formed in that
dtype. So inside `data_metric` the weight and the array it weighs already share one dtype, and squaring before or
after the (now trivial) cast gives the same bits. The order that matters, cast then square, is the entry cast; the
one place a pair of mixed precisions meets `_omega_squared` is a direct call of `data_metric`, which M1's test pins
bitwise.

## 4. The runs

| Run | Result |
| --- | --- |
| the first implementer's engine rerun (`p2_eng2.log`: `test_sampling_definitions`, `_variables`, `_apply`, `_entries`, `_corewise`, `_stacks`, `_identities`, `_uniform`, `test_contractions_sharding`, at `-n 2`) | 207 passed, 4 failed (the 4 sharding subtests), 22,761 subtests, 5:32 |
| `test_contractions_sharding -k TestTheEngineShardsOverW`, after the fix | 1 passed, 4 subtests |
| `test_engine_data_space.py` + `tests/bridge/` (first run) | 6 failed: the `None`-weight `IndexError` and the gauge comparison |
| `test_engine_data_space.py` + `tests/bridge/`, after both fixes | 17 passed, 154 subtests |
| `test_dispatch.py -k "sampling_variables or eager_uniform"` | 7 passed, 8 subtests |
| `test_engine_data_space.py` final | 11 passed, 182 subtests, 15 s; after review round 1, 12 passed, 185 subtests |
| changed modules' doctests (`sampling_variables`, `sampling_steps`, `ut3_operations`) | 6 passed |
| **the full suite, `pytest tests/ -v -n 2`** | **664 failed, 1,839 passed, 5 skipped, 1 xfailed, 90,765 subtests passed, 26:07**; MemAvailable about 6 GB at the midpoint |

**The red, `M2_part2_EXPECTED_RED.txt`:** 194 test ids, each with its reason. All 194 are the engine's new check
refusing an old-form object that reaches it through the oracle (`LeastSquaresOracle.query` passes `batch.y` and
`batch.weight`): 170 a block weight, 24 a stacked model's `y` without `C`. 190 raise at `_check_data_space` with
`backend/fitting.py` in the traceback; the other 4 (three oracle sweeps that collect failing configurations, and
`test_jax_absent`'s subprocess) were read, and every error inside them is the same check. None is red for any other
reason; parts 4 and 5 turn them green. `test_zero_reference.py:415` is among them, as the design review predicted.

**The tally against part 1's `-v` log** (`m0/tally.py`; kept as `m2/tally_part1_vs_part2.txt`): 1,934 nodes there,
1,946 here; 206 differ: the 194 red, the 11 nodes of `test_engine_data_space.py`, and part 1's round-1 refusal test
(added after part 1's log was taken). Every converted engine-level node keeps its outcome and its subtest count.

## 5. Departures and choices

- `test_engine_data_space.py` imports `test_sampling_definitions` (as `test_sampling_variables` does) for the
  reference's problems and jet matrix, and `sampling_definitions` for the reference.
- `TestTheUniformEdgeOfOne` packs the ragged frame instead of orthogonalizing the uniform point, so that `g`'s
  blocks compare; the gauge-equivalence of the two frames is the documented contract, not a defect.
- The weight mask is kept although it is redundant for ordinary finite garbage: it is what keeps an overflowing
  padding value from becoming NaN, and plan §4 asks for it. The test documents why it exists.
- The design note's part-2 list named "the old-kernels bridge with `blocks_to_weight`": done (first implementer).

## 6. Found outside the part

- None new in the library. The expected-red list (§4) names what parts 4 and 5 must turn green.
- For part 4's design: the oracle tests' fixture batches (`test_least_squares_oracle`'s `batch(...)`) feed the engine
  directly as well as through `query`, so converting the batches converts both.
