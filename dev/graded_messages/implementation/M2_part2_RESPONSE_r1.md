# M2 part 2 response to review round 1

Implementer, 2026-11-22, on `M2_part2_REVIEW_r1.md`. Both points accepted.

## Should fix

1. **The join across the modes in `_pack_data_space`.** New
   `TestTheUniformEdgeOfOne.test_modes_with_different_size_1_axes` in `tests/test_engine_data_space.py`:
   - order `(2,)`, `C = (2,)`, `W = (3,)`, on the packed ragged frame (`UT3Frame.from_t3frame`);
   - a ragged probe `y` and weight whose modes take three patterns: per slot with a full edge `F + (1,) + (1,) + (N_i,)`,
     per sample and element with an edge of 1 `(1,) + W + C + (1,)`, and per entry `F + W + C + (N_i,)`;
   - the patterns rotated over the modes in three subtests, the weight's pattern in each mode never `y`'s;
   - uniform against ragged on `f`, `r_block_sumsq` and every `V` and `H` block of `g`.

   **Your mutant** (the body `F + W + C` from mode 0, `body = tuple(arrays[0].shape[k] ...)`), on a scratch copy:
   **fails all 3 subtests**; the copy is deleted. Unmutated: passes.

## Minor

1. **The expected-red header.** It now says how the one cause shows: most ids raise at the check; about 7 fail at the
   test level through a downstream consequence, each named (`bond_wins` 0, `m` unbound, the empty covered set, the
   `assertRaisesRegex` receiving the check's error, the three sweeps' collected lists, `test_jax_absent`'s subprocess).
   The ids and their tags are unchanged.
2. **One review unit with the checkpoint `fe7c7250`:** noted for the squash; nothing to change in the tree.

## Rerun

`tests/test_engine_data_space.py` at `-n 2`: **12 passed, 185 subtests**, 15 s. It is the only test file touched this
round. `M2_part2_HANDOVER.md` is updated (the new test, the count). Nothing is left running.
