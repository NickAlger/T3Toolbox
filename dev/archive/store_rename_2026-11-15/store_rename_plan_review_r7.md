# The store rename plan — critic review, round 7 (the width ruling)

Reviewed: draft 7 (851 lines): §0, A3 and its scoped swaps, L5, §2, D-3/D-8, K2, K13, the battery's adapter and
fixtures, §6's migration recipe.

**Consistent.**
- The swap is simultaneous with each suffix kept verbatim, `nO`→`nU` is one-way, and the explicit forms are listed.
  No file holds both `nD` and `nO`, so mapping both to `nU` merges nothing; `sampling_derivatives` has only `nO` and
  `nU`, and with the `n` exception `nO = nU` becomes `nU = n`.
- The scoped `kU`↔`kD` … `WU`↔`WD` swaps are listed.
- The adapter's "A3 reversed" is the swap itself.
- No gate bans `\bnU\b`/`\bnD\b` any more.
- §2's silent list grows by exactly 16 keywords and 6 properties, and §6's snippet swaps `nU`↔`nD` and drops `rL`/`rR`.
- K13's table is right where §5.3 states it. Against `UT3VariationsMasks`' own comments, `variations_up_mask` is V's row
  mask (the old `variations_down_mask`, `(nD,)` today) and `variations_down_mask` is H's middle mask; the left and right
  cores and H carry the down width.

## MINOR

1. **The width regex corrupts an escape sequence.** `examples/fit_hilbert_regularized.py:167` prints
   `f"\nOn the fixed-rank manifold …"`. In the source text, `\b` holds before the `n` of `\n`, and `nO` followed by
   `n` satisfies `[ijkdn]\b`. The one-way `nO`→`nU` would therefore print "\nUn the fixed-rank …", and K2's `\bnO` would
   flag the correct line. This is the only such hit in the tree (scanned with the plan's own pattern). Add
   `(?<![\\\w])` before the A3 family, the `nO` rule and K2's `nO`.
2. **K13's generic-entry rule contradicts R2's output.** K13 says a T3's own `tucker_cores`/`tt_cores` carry `n`, and
   K13 is an R2 gate. R2's swap writes `nD`/`nU` into about ten generic kernel shape lines that the `n` exception does
   not list: `sampling_derivatives.py:388, 416, 473, 518, 872, 995, 3236`, `probing.py:550`, and the `ufv_conversions.py:334–335`
   comments. Some of those parameters receive frame stores, not a T3: `sampling_derivatives.py:473`'s
   `tt_cores … (rLi,nOi,rR(i+1))` holds the up store, so after the swap its `nUi` is right and `n` would be wrong.
   Make K13 recognise only the store-named entries and V/H, and report generic `tucker_cores`/`tt_cores` lines to the
   reviewer. Alternatively, list them with a disposition each: `n` for a T3's cores, the swapped letter where a kernel
   receives a frame store.

## NIT

- The battery passes widths positionally. Add the default-width path on `xS` (`UT3Frame.from_t3frame(F)`,
  `UT3Variations.from_t3variations(V)`, `UT3Tangent.from_t3tangent(v)`). Their `nU = int(max(b.up_ranks)) if nU is None`
  lines pair two swapped names, and `from_t3tangent` forwards them as keywords (`dict(N=N, nU=nU, …)`), so a half-applied
  swap there shows only through those calls.

VERDICT: REVISE (0 MAJOR, 2 MINOR)
