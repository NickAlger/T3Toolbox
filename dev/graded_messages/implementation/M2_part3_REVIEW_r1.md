SIGN OFF

# M2 part 3 review, round 1: `jets` on the rule

Reviewer, 2026-11-22, on the uncommitted part 3 on `m2-wip` (over `7b2bcf49`) and `M2_part3_HANDOVER.md`. My
script is `scratchpad/m2rev/check_p3.py`.

**Verdict: SIGN OFF.** The checkers, the converters and the factorials agree with my own constructions. The bridge
test is now independent of the library. The 123 red ids are exactly the log's failing set, each with a stated cause.
The one changed test input is justified. Two minor notes follow.

## 1. The checkers against my own construction

`check_data` and `check_weight` on probe (ragged), apply and entries, at orders `()`, `(2,)` and `(1,1)`, with
`C = ()` and `(2,)`.
- **Legal: 70 objects, all passed.** Each axis is drawn at random 1 or `z`'s size, with `stack_shape=C`.
- **Illegal: 68 objects, all raised.**
  - rank ±1;
  - an axis neither 1 nor the size;
  - `d − 1` arrays;
  - a mode-leading array with a mode axis of 1;
  - a tuple for apply or entries;
  - a stacked object checked without `stack_shape`;
  - an object larger than `z` where `z`'s axis is 1 (a one-sample batch).

## 2. The converters' `K` behaviour and their numbers

- **The `K` behaviour.** A `K`-stacked apply jet (`F + W + K + C`) with a full-form weight raises; with `weight=None`
  it is turned (bitwise the flipped jet). This is r7's point 2, as the plan states it.
- **The numbers.** `jet_to_dual_jet` equals my written-out `flip((ω·ω)·r)`, cast to the lower precision first,
  bitwise and with its dtype, in all four float32/float64 pairings. This holds for ragged probe with a tuple weight
  and with a mode-leading weight. `dual_jet_to_jet` round-trips to `1e-13` (float64) or `1e-5` (float32).

## 3. The factorials by hand

`derivative_weight_to_taylor_weight(kind, order, ω)` against `expand(ω) · k!`, written out with the factorials on the
leading jet axes:
- apply at `(2,)` and `(1,2)`: a per-entry weight, and a per-sample one whose jet axes of 1 come back full;
- probe at `(2,)` and `(1,2)`: a tuple per sample, and a mode-leading array per slot;
- entries at `()`.

All agree to `1e-15`, with the right shapes, and the form comes back as given (a tuple for a tuple).

## 4. The re-pinned bridge test is independent

`_old_block_weighting` in `test_data_metric.py` is numpy only: per array, the block row cast to the lower precision
of the array and the whole block array, squared, reshaped onto the leading jet axes, times the cast array. It calls
no library function. `data_metric(r, blocks_to_weight(...), None)` and the unturned `jet_to_dual_jet` are each
compared with it, so the test no longer compares `data_metric` with itself. It covers the four pairings, and the
implementer reports that M1's L4 mutant still fails it (8 subtests), consistent with M1.

## 5. The expected-red list

- **The set.** Progress and summary lines of `suite3_v.log` give a failing set equal to the 123 listed ids. No file in
  `t3toolbox/` or `tests/` is newer than the log.
- **The causes.** All 434 exception lines fall in the stated causes:
  - `prepare_derivative_data`'s check on a block weight: 251;
  - the engine's check (`y`, the weight): 152;
  - `check_weight` through `check_batch`: 24;
  - the old signature of `derivative_weight_to_taylor_weight`: 3, all at `test_least_squares_oracle.py:535`;
  - the three sweeps' collected lists: 3;
  - the `f_resolution` gate: 1, `412.98 != 224.98`.

  Nothing else is red.

**Claude's ruling on the interim `f_resolution`: I agree.** The block formula
`weight**2 * y_block_sumsq` on a full-form weight broadcasts into cross terms. It is part 4's first item, the branch
is squashed before `graded-messages`, and the float32 gate is red meanwhile as its guard. Part 4 should pin the new
`f_resolution` against a dense `<y, M y>` over `z`'s shape, and the battery compares it to `1e-13` in part 5. The 73
tests that turned green here pass without asserting `f_resolution` closely, so they say nothing about it.

## 6. The one changed test input

`test_the_checkers_run_under_jit`, the "the data y" refusal: the old input passed the weight as `y`. A block weight
was an illegal `y`, but a full-form weight is now legal data, so the refusal needs another bad `y`. Its new `bad_y`
has a `W` of 5 against the sample's 2, which is illegal for every kind. The bad weight changes from `(1,)` to `(3,)`;
`(1,)` could now be a legal all-ones form. With `W = (2,)`, `(3,)` is illegal for every kind, entries included. The
purpose of the test, refusals under `jit`, is kept.

## Minor

1. **`derivative_weight_to_taylor_weight` cannot tell a block form from a full form** (it has no `y`). Its docstring
   already says so ("nothing else is checked here", pointing to `prepare_derivative_data`), so nothing is needed.
2. **The handover's §4 count**, "1 the old signature", is one id, which shows as 3 exception lines (subtests). Fine
   as is; noted for the tally reader.

Nothing is left running.
