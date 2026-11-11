# The store rename plan — critic review, round 21 (draft 21, confirmation)

Reviewed: the draft-21 diff against `c2d53b41`, which records R1's findings.

**Spot-checked against the code:**
- **`_lost_directions` (`test_uniform_frame_variations_format.py:891–896`).** `um, dm, lm, rm = uframe.data[5]` beside
  A2's `mup, mdown`. With `um↔dm` the pair stays matched (`int(dm[i].sum()) - rank(mdown[i])`), since A2 swaps
  `mup`/`mdown` on the same lines.
- **`frame_variations_format.py:821` (`T3Variations.shape`).** `[U.shape[-1] for U in self.tucker_variations]` iterates
  Tucker variations, so `U→V` is right.
- **`test_t3_orthogonalization.py:123–125`.** `V, Oc = O.t3_up_orthogonalize_tt_cores(x)`: `Oc` holds the up-orthogonal
  TT cores, so `Oc→Uc` is right. The `O.` is the oracle module alias, item 13, and stays.
- **`test_uniform_fitting.py:360–361`.** `P`/`Q` are comprehension variables over the left/right cores, so `→L`/`R` is
  right. The nested `fam`'s `R` is a separate scope.
- **`LETTERS_NEW`.** It is keyed on the type of `tv_align_padded_frame`'s `update` default (a `str` through R2, a
  `tuple` from R3), so the R2 preview is well-defined.
- **DNT item 14.** It records kept locals that a reader might take for renamed ones (QR's `R`, the factored
  `P`/`Pd`/`Q`/`Qd` products, `sh`/`psh`/`pshd`, `shift = lambda`). It is disjoint from the σ̂ rows, and K14 now
  covers items 1–14.
- **The rest:** the three new §4A blocks, the widened K8 tex pattern, item 12's three kernels, the corrected
  `ufv_conversions.py:334–335` note, and the `U_3` comment-only mark are as listed.

No MAJOR or MINOR finding.

VERDICT: AGREE
