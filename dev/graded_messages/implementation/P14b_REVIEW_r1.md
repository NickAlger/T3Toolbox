SIGN OFF: `computed_names` and the engine share one reading, so they cannot diverge; the oracle holds exactly what is free and adds no work; `frame_shape` is right on the four forms; and the plan edits extend the text without rewriting history.

# P14b review, round 1: `computed_names` in place of `keep=`; `frame_shape`

Reviewer, 2026-11-21. This reviews the uncommitted P14b on top of `327ca436`, against `TARGET_DECISIONS.md` ("The
hold list: `computed_names`, not `keep=`"), the rulings in `LOG.md`, `P14b_DESIGN_NOTE.md` and the handover.
Everything was run single-process. The mutant copies have been deleted, and the working tree is untouched apart from
this file.

## `computed_names` and the engine cannot diverge

By construction:
- Both call the one private `_closure_facts(kind, frame, want, order, y, variations, core_directions, Vc, held)`.
- The engine unpacks its `rules` and runs exactly those (`for rule in rules: _run(...)`). No row is added or dropped
  after it: the array-reading checks that follow (the index, packing, the setting, the stacks, the held shapes) only
  raise.
- `computed_names` returns that closure's public names. Held names are never in a closure, and private names are
  filtered.
- The arguments that do not decide the closure (`sample`, `weight`, `sum_over_samples`) are accepted and unread.
  None of them is a needs fact: `keep_W` changes shapes, not rows.

The test `test_computed_names_is_what_the_engine_runs` pins this empirically by recording the rules `_run` sees,
over more than 150 calls. Its companion shows the same exception and message for each of seven bad calls.

## The oracle holds what is free and adds no work

`free = computed_names(..., engine_want, **args)`. The extras are `HOLD ∩ free` minus the wanted names, and the call is
for `engine_want + extras`. Adding names already in the closure cannot change it: each was reached, by the same
first-applicable row and the same dependencies, so the closure of `want + extras` is the closure of `want`.
- **Held:** the `held` passed in, plus `HOLD ∩ free`.
- **The test** asserts that the rules run equal `computed_names(want)` exactly. That kills the "whole hold list in
  `want`" mutant, which had survived the implementer's first pass.
- **Under `jit`** the slots are static: `computed_names` reads no arrays.

## `frame_shape`

It reads the four forms:
- **ragged:** the last axis of each Tucker core;
- **uniform:** the static `shape` element `frame[-2]`, as plain `int`s, so on a padded supercore it reads the real
  sizes, not `N`.

It raises as `frame_form` does. The oracle uses it to unpack a packed probe `y`, replacing `frame[4]` / `frame[2]`.

## The plan edits

The diff only extends existing sentences, re-adding each changed line with its addition. Nothing is rewritten as
history:
- the oracle's text names `computed_names`, "in place of an engine keyword `keep=`";
- 2.4 has a row for the three readers;
- P14 has one sentence;
- P16 gets three additions:
  - the aligned padded frame's `(frame, centers)`, with zero-padded old centers and a consistency-residual test;
  - the held-across-draws test and `pack_sample` / `pack_data`'s `(kind, order, ...)` forms;
  - `eps |rho|` in the floor.

These match the decisions. `P15_HANDOVER.md`'s now-superseded note on the aligned padded frame is marked as
superseded by this handover's §6.

## Mutants (3, mine): all killed

| Mutant | Killed by |
| --- | --- |
| `frame_shape` reading the padded `N` | 7 subtests |
| the oracle's `computed_names` taken on `want` rather than `engine_want` (so `f_resolution`'s added `f` is missing) | 23 subtests |
| `computed_names` ignoring `held` | 121 subtests |

## Runs

- `test_least_squares_oracle` and `test_sampling_variables`: 33 passed, 2,640 subtests.
- `test_least_squares_oracle` on the floor: 18 passed.
- The implementer's gate before `frame_shape`, then the oracle tests on both envs, the doctests and Sphinx after it,
  all pass.
