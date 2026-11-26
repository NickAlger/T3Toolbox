SIGN OFF

# Review of the revised `DATA_METRIC_PLAN.md`, round 2

Reviewer, 2026-11-22. I checked the revision and `DATA_METRIC_PLAN_RESPONSE_r1.md` against `aaee0b04`, and ran one
script, described under "The reading rule, verified".

The revision settles every round-1 finding, and the writer's five departures are each justified. I found nothing
blocking. Four minor points follow; the implementers can take them in the slice design notes, and none needs another
round before the maintainer reads the plan.

## The round-1 findings

| # | Finding | Status |
| --- | --- | --- |
| 1 | The reading rule | **Resolved.** §2 states it exactly (rules 1 to 6). The refinement, that the probe mode count stays exact, is right: the mode axis is the tuple index, not an axis of `y`, so broadcasting it would make the per-mode reading drop modes. It also keeps the one point of 6.7 the maintainer confirmed explicitly. |
| 2a | `W` and the stacks | **Resolved** by rule 5; no group length needed (verified below). |
| 2b | The scale | **Resolved.** The scale is in `W`; `weight` and `minibatch_scale` are both required, and `None` must be written out. The arrays-not-batch argument holds: a `jets` function cannot import `LeastSquaresBatch`. The batch form is put to the maintainer as option (b) of Q3. |
| 2c | Home | **Resolved.** `W` is defined in `jets` and re-exported by `t3toolbox.fitting`. Not re-exporting it from `backend/fitting.py` is fine: that would add a second import path and no capability. |
| 2d | One `ω²` | **Resolved,** seven sites on one helper. One more site arrives with the display: minor point 1. |
| 3 | Questions | **Resolved.** Eight questions, each with a recommendation. The case against `covector` is correct under the maintainer's typing: the measurements in `Y'` are the covectors, and `W: Y' -> Y` is his "co-vector to vector". |
| 4 | D4 | **Resolved.** §1 "Coordinates" derives `c_t = D c_d` (checked: `<c_d, D r_t> = <D c_d, r_t>`). It adds docstring sentences, comments in the two examples, and Q7. |
| 5 | Factorials | **Resolved.** The jet axes are the leading `m` axes per array, and `kind` is added to the converter. M1 tests it with equal and distinct sizes. |
| 6 | Padding | **Resolved.** `pack_batch` never pads a size-1 edge, and the engine masks a full-`N` edge on entry. |
| 7 | `take` | **Resolved.** An all-ones `W` part becomes `(1,)`; otherwise the weight is broadcast, flattened and gathered. |
| 8 | Display | **Resolved** as Q4. Option (b) adds new weighted, unscaled block sums, so O-21's plain sums and the `unwt` figure are untouched. The `0/0 -> —` caveat is stated. |
| 9 | Reversed decisions | **Resolved.** 2.5 and 6.7 are named. The writer is right that the engine's stack check is not reversed: the weight was already outside it (`sampling_variables.py:1097`). |
| 10 | Oracles | **Resolved.** The reference is extended independently (M2), and so is the dense oracle (M3). |
| 11 | Counts | **Resolved.** 17 module-doctest calls is right; my 18 included a hit in `docs/_build`. |
| 12 to 14 | Translation; decision 4; jit and sharing | **Resolved.** |
| 15 | Slicing | **Resolved.** M1 to M4 each have an oracle independent of the code under test: hand-written arrays; the reference module (which may not import `jets`); `oracle.py` with its own alignment; a difference quotient of the dense misfit on a drawn batch. M2's added check, that `ω = 0` gives the same `f` and `g` as deleting the sample, is a good independent one. |

## The reading rule, verified

I ran a prototype of rules 1, 2 and 5 that is given no group length. It compares `ω²·r` with a brute-force,
entry-by-entry evaluation of `ω(mode, F, W, edge)`. The maximum error is exactly 0 in every case:

- apply:
  - the full form `F+(4,1)` (`W` partly broadcast), on an `r` with both `K` and `C`;
  - the block form `F`;
  - the per-sample form `(1,)+W`;
- entries: the full form `W`, on an `r` with `C`;
- probe, ragged:
  - the full form `F+(1,2)+(N_i,)`, with `K` and `C`;
  - the full form with an edge of size 1;
  - the block form `(d,)+F`;
- probe, packed: the full form `(d,)+F+W+(N,)`, with `C`;
- probe at `m = 0`: the full form `W+(N_i,)`, with `C`. This is the case that was ambiguous by rank in round 1; requiring the edge resolves it.

**The 19-case rejection table, worked through by hand under the rule** (`D = 3`, `W = (7,)`). Four cases become legal
and fifteen still raise, as §2 says.

**Two more checks that hold:**

- `f_resolution` with `s` inside both `W` and `f` equals today's value: `√(s<ω²y,y>)·√(2sf) + s|f| = s·(…)`.
- `jet_to_dual_jet = flip ∘ W`, with `dual_jet_to_jet` flipping first and then dividing by the unflipped helper. This
  matches `dual[k] = ω[order-k]² r[order-k]`.

## Minor points, for the slice designs

1. **An eighth `ω²` site.** Option (b) of Q4 adds weighted block sums of `r` and of `y`. They must go through the same
   helper. Add a row to the §3 table, or a sentence in M3.
2. **Keeping M1 green.**
   - M1 changes two signatures, so it must also update their callers.
     - `check_weight(kind, order, sample, weight)`: its call in `check_batch` (`fitting.py:958`). The plan lists that
       call under M3.
     - `derivative_weight_to_taylor_weight(kind, order, weight)`: `test_dispatch.py:600` and
       `test_least_squares_oracle.py:535`. Neither file is named in §6.
   - From M1 on, `check_batch` accepts full-form weights, but the engine reads them only from M2. On the branch this
     is a transient state, harmless if M1 and M2 are not released apart. One sentence in the plan would do.
3. **The uniform engine's weight input.** State that the uniform engine takes a block-form weight or a packed one,
   `(d,)+F+W+(e,)`, and never a ragged tuple: `pack_batch` makes the packed form, and the mask of §4 assumes a
   leading `d` axis.
4. **The rejection test needs a sample now.** Once `check_weight` takes the sample, the rejection test must choose its
   `W` distinct from every `F` size. Otherwise apply `(1, n+1)` would pass as a full form (the plan's `W = (7,)` does
   this; say so in M1).

## Readability

The plan reads well for the maintainer:

- the typed table and the six outputs come first, then the rule as a table with numbered rules, then the signature
  and a table of the sites;
- each slice has an "oracle" column;
- each question puts its recommendation first, with the reason.

The only dense passage is §4 "Uniform padding". It is correct, but it is implementer detail, and the maintainer can
skim it.
