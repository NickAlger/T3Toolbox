# M2 part 1 response to review round 1

Implementer, 2026-11-22, on `M2_part1_REVIEW_r1.md` and Claude's rulings (the reference module's change is in
part 1's scope). Every point is accepted.

## Should fix

1. **The rule where `z`'s axis is 1.** New `TestTheWeightIsOfTheTypeOfZ.test_an_object_larger_than_z_where_z_has_a_size_of_one_raises`
   in `tests/test_sampling_definitions.py`. It covers two places where `z` has an axis of size 1:
   - a one-sample batch, `W = (1,)`, on a new problem with the shapes of `small`;
   - the jet axis at order `(0,)` (a line clipped to one slot).

   For each place, probe and apply, the test does three things:
   - it asserts that `z`'s axis there is 1;
   - it checks that a weight of `z`'s own shape is accepted;
   - it checks that a weight, and a `y`, with 3 on that axis raise a `ValueError`.

   That is 8 subtests. A script confirms that each raise comes from the rule, naming the axis. It does not come from
   the mode count or anywhere else: the messages read "has shape (3, 4); … here has shape (1, 4)".

   **Your mutant R1** (`a not in (1, b) and b != 1`), on a scratch copy, now **fails all 8 subtests**. The copy is
   deleted.

## Minor

1. **The `mc_sgd` loss history in the battery.** `mc_sgd` records its full-batch loss only every `check_every` steps
   (default 25), so a 4-step run had none. Both runs now pass `check_every=1`, and the loss of every step is stored.
   - I re-recorded the pre-M2 record on the base tree: 25,476 arrays, 8 more, all `mc_sgd` losses.
   - Two runs give `AS EXPECTED` with every array bitwise equal.
   - The stop is unchanged: 4 iterations each.
   - The single record is kept as `scratchpad/m2impl/battery_old.npz`; the earlier copies are deleted.
2. **The equal-sizes problem.** It stays in the brute-force class when part 2 extends it to `C`. Noted for part 2's
   design: `small` with the curve at `(2,)`, where `F = W = N_1 = (3,)`.

## Reruns (`-n 2`)

`tests/test_sampling_definitions.py` and `tests/test_dense_oracle.py`: **79 passed, 5,772 subtests**, 2:01. They are
the only test files touched this round. The module doctests of `sampling_definitions` are unchanged since round 0.
`M2_part1_HANDOVER.md` is updated: the refusals, and the battery count and runs. Nothing is left running.
