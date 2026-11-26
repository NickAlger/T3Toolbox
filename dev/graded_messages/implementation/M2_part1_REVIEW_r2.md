SIGN OFF

# M2 part 1 review, round 2

Reviewer, 2026-11-22, on `M2_part1_RESPONSE_r1.md` and the uncommitted tree on `m2-wip`.

1. **Finding 1 (the reference's check where `z`'s axis is 1): resolved.**
   - The new refusal test covers a weight and a `y` larger than `z` against a size-1 axis of `z`: a one-sample batch
     (`W = (1,)`) and an order-0 jet axis, probe and apply, 8 subtests.
   - I reran my mutant R1 (the reference accepting any size against a size-1 axis of `z`) on a fresh scratch copy of
     the tree, against `TestTheWeightIsOfTheTypeOfZ`. It fails all 8 new subtests: weight and `y`, `W = (1,)` and
     order `(0,)`, probe and apply.
   - The unmutated copy passes: 4 tests, 163 subtests.
2. **Minor 1 (`mc_sgd`'s loss history in the battery): taken.** The record now holds `opt/mc_sgd/…/losses/0..3`.
   - I re-recorded the battery myself on the base tree (`scratchpad/m2impl/base`, equal to `85c6e54d`) with the
     committed script.
   - My recording and the implementer's `battery_old.npz` hold the same 25,476 keys, with 0 bytewise differences.
     The record is deterministic and regenerable from the commit.

My round-1 conclusions stand:
- the reference agrees with my brute force;
- `oracle.py` is right;
- the engine-against-reference conversions are argument-only, with every count kept;
- the battery's `--compare` implements my design-review finding 2.

Nothing is left running; the scratch copies are deleted, apart from `scratchpad/m2rev/new`.
