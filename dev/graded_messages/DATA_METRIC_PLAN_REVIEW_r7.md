SIGN OFF

# Review of the plan with every question decided, round 7

Reviewer, 2026-11-22. I checked the rewritten `DATA_METRIC_PLAN.md` (working tree after `b5a848d5`) against the
maintainer's notes of 2026-11-22 in `TARGET_DECISIONS.md` (first answers, Q1/Q12, Q5/Q7/Q8, Q3) and against "Round 7"
of the response. I ran one script, `scratchpad/r2/rule7.py`, and recounted the blast radius.

**Verdict.**
- Every decision is applied as decided.
- The rule is exact in every case I tried.
- The counts match.
- M2's one-step switch is justified.

Four things in the plan were settled by the writer rather than the maintainer. Two of them should be shown to him at
his final read, not left to a design note (points 1 and 2). One point strengthens M2's gate (point 5). None of them
blocks.

## The decisions, applied

| Q | Decided | In the plan |
| --- | --- | --- |
| 1 | `data_metric` | §3, the body |
| 2 | `c` | §5 |
| 3 | the signature and its rules (listed below) | §3 |
| 4 | the weighted ratio | §4, M3 |
| 5, 8 | the type-of-`z` rule | §2 |
| 6 | `ω` | §2 |
| 7 | Taylor coordinates, one conversion at the boundary, the recipe, documented in three places | §1, §4 |
| 9 | frontend-private, backend-public | 9.2 |
| 10 | `*` | throughout |
| 11 | types in both backend and frontend signatures | 9.1 |
| 12 | `F_in,i` | M0 |

**Q3 in full.** `data_metric(r, weight, minibatch_scale)`:
- elementwise, with no `kind` or `order`;
- both factors required, with `None` written out;
- no `K`;
- `batch.data_metric(r)`;
- defined in `jets` and re-exported by `t3toolbox.fitting`.

**Q5 and Q8 in full:**
- each axis 1 or `z`'s size, never `K`;
- no block form; a constructor builds a weight from blocks;
- `y` and `ω` carry `C`;
- `r_step`'s special alignment of a `y` without `C` goes.

**Q10:** a grep finds no `X'`, `Y'`, `P'`, `E'` or `V_i'` left.

**What follows from the decisions, correctly drawn:**
- O-20 is superseded;
- the rejection table is re-ruled: 2 cases legal, 17 raise;
- the translation, the bridge and the pages go through the constructor;
- `f_resolution` is per element of `C`;
- the factorials sit on the leading jet axes.

## The rule, by script

The plan's `data_metric` is equal-rank elementwise broadcasting, plus the engine's `_with_K`. I compared it with an
entry-by-entry evaluation; the largest error in any case is 9e-16.

**Cases checked:**
- probe, ragged, with `C` at full size and the edge at full size;
- probe, ragged, a `K`-stacked `Jp` through `_with_K`;
- probe, packed, with `C` of size 1, and its `K`-stacked `Jp`;
- apply, a per-sample weight with `C` of size 1;
- entries, the full `W+C`;
- `blocks_to_weight`'s placement: `(d,)+F` blocks become `(d,)+F+(1,)*(len W + len C)+(1,)`, so the per-mode shapes
  are equal and a `d`-leading array is well defined.

**Cases that raise:**
- a `K`-stacked `r` raises by rank;
- a bare-`W` apply weight at `m = 1` raises by rank. The reading trap of the block rule is gone.

## Counts

All match my recount. The units differ in one place:

| Item | Plan | Recount |
| --- | --- | --- |
| `weight=` lines in tests | about 170 | 174 |
| `LeastSquaresBatch(` in tests | 39 | 39 |
| `check_weight(` / `check_data(` / `prepare_derivative_data(` / `derivative_weight_to_taylor_weight(` calls in tests | 12 / 7 / 20 / 7 | 12 / 7 / 20 / 7 |
| converter calls in tests | 73 | 73 |
| `J_in` lines | 180 | 163 in the library + 17 in tests |

The units point is minor: 9.2's table counts *lines naming* the converters (94 in tests), and §6 counts *calls* (73).
Both are right, but §6 says "lines". Write "calls" there.

## M2 as one switch

**Necessary, and safely buildable.**
- Once the block form is gone, the checkers, the engine, `take`, the packing and the translation must agree on one
  representation. A committed intermediate state would be the silent wrong answer the plan names: a full-form weight
  through an old `take`.
- Building M2 in parts on a sub-branch and squashing, as P16 was, avoids such a state on `graded-messages`.
- Its oracles are independent:
  - the reference module;
  - the dense `oracle.py`;
  - zero-weight equals deletion;
  - per-element stacked data equal the separate fits;
  - unbiasedness by exact enumeration;
  - the factorials by hand;
  - the translation's 19 pins.

See point 5 for two cheap strengthenings.

## Points

1. **`blocks_to_weight` is settled by the writer, not the maintainer.**
   - Its name, its signature `(kind, order, blocks, y)`, and its output form (a `d`-leading array for probe) are the
     writer's. The plan says it is "confirmed in M1's design note".
   - It is a new public name, and on this branch the maintainer decides public names.
   - He reads the plan next, so list it in §8 as the one open item for his final read, with the alternatives
     (`weight_from_blocks`, `block_weight`) and the recommendation.
2. **The converters "now reject a `K`-stacked jet": also settled by the writer, and inconsistent as written.**
   - Q3's "raises by rank" is about `data_metric` with a weight. The plan's own `data_metric(r, None, γ)` returns
     `γ r` for any `r`, `K` included, because there is no rank to compare.
   - Routed through `data_metric`, the converters then raise on `K` only when a weight is given. With `weight=None`
     they flip a `K`-stacked jet correctly, as today.
   - Recommendation: state exactly that behaviour, which follows with no new rule, instead of a blanket rejection the
     maintainer did not decide.
   - Also fix M1's test line "a `K`-stacked `r` raises": it raises when a weight is given.
3. **Other writer-settled public details.** None needs a question; each should be named as proposed in the design
   note that builds it.
   - the checkers' new keyword `stack_shape=()`, which uses the library's existing name for `C`;
   - `prepare_derivative_data(weight=None)` still returning `k!`, now in full form;
   - the names of M3's new engine variable and oracle output (the weighted block sums). Engine variable names are the
     maintainer's (the locked notation), so propose them to him in M3's design.
4. **Edge cases of `data_metric`'s pairing**, for M1's design note:
   - a packed `r` with a ragged tuple weight is not covered: raise, or read the tuple;
   - a packed weight whose mode axis has size 1 would broadcast unchecked. The checkers take the ragged form only, and
     `pack_batch` builds packed weights from checked ragged ones, so this is a documented limit, not a hole.
5. **M2's gate, two cheap strengthenings.**
   - **(a) Numbers unchanged on the old weights.** Every existing weighted test, its block weight passed through
     `blocks_to_weight`, gives the numbers it gave before M2, to rounding. That covers far more than the 19 pins, and
     it is the direct check that the representation switch changed nothing it should not.
   - **(b) Build the reference module's full-form reading first.** Pin it against brute-force dense sums before the
     engine switches. The reference and the engine are changed in the same slice, so their errors could otherwise
     correlate. The independence test forbids imports, not shared mistakes.
6. **Readability.**
   - The plan reads well: "In brief" at the top, §8 now a compact decision record, one notation table.
   - The status line still says "the reviewer checks this version once more". Update it after this round.
