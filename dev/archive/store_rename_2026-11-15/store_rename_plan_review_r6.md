# The store rename plan — critic review, round 6 (diagrams and equations)

Reviewed: draft 6 (832 lines, 01:18): §4A, §1.4, §3.4's `O` family, K9 to K12 and the slice gates.

**All five r5 minors are resolved.**

- **Fences are spans.** K11 compares span text, so the field comments no longer contradict it.
- **K12 is the completion gate.** It runs after R4b on the code tree and after R5 on the docs. A forgotten block,
  such as `frame_variations_format.py:86–96`, now fails.
- **K10b checks re-laid-out blocks mechanically.** Every old alignment column must map to one consistent new column,
  and a field comment's start column is checked.
- **Tier E's locator skips fenced lines.** Its rows inside a fence feed the hand rewrite only, and tier D runs over
  unfenced lines only.
- **§3.4 disposes of the `O` family**: `Uio`→`Bio`, then `O…`→`U…`, function-scoped. `O_g` is kept and carved out of
  K8.
- **The nits are applied.** The store-word recurrence rule `\b(?:up|down)_[a-z]\b` is in both the scanner and K9. The
  seed list is extended. Companion doctests are inventoried but not fenced, and checked against their diagrams. K10
  and K10b compare raw text.

**Protection, checked end to end.**

- Every script pass skips the fenced spans: pass 1, tier A on the docs, tier D, the tier-E locator, and the R4/R5
  regexes.
- K11 proves the fences untouched through R2 and R3.
- K10 and K10b hold the geometry of every rewritten block, K12 proves every block was done, and K2a/K8 ban `nO` and `O`
  inside fences too.
- The reviewer reads every block and re-derives each relation, and Nick sees every before/after at R4b and R5.

I found no path by which a regex reaches a diagram or a block escapes.

## NIT

- **Some seed ranges cut through a block.** `frame_variations_format.py:1346–1362` stops at the third diagram's top
  line (1362), so its `U0 U1 V2 U3` row (1364) is outside; the block is 1360–1365. Likewise `1262–1273` should be
  1262–1274. R1's per-line candidate gate catches this, but the inventory should take block bounds from the scanner,
  not from the seeds.
- **Companion doctests: "their code is renamed by R2" (§4A a).** Tier C maps `frame_variations_format.py` only at
  483–573, so `:1302–1322`'s `U0…`/`D0…` are §4.4 rows, done in R4a. K12 still checks them, since they are
  inventoried.
- **A merged bullet.** §4A(c)'s last bullet runs the "typo inside a diagram" paragraph into the field-comment bullet
  (line 551).

VERDICT: AGREE
