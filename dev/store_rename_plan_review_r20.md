# The store rename plan — critic review, round 20 (draft 20, confirmation)

Reviewed: the draft-20 diff against `edb669de`, checked against `dev/store_rename_sample_edits_3.md` §12–§13 and the
code.

**All sixteen §12 items are fixed as the session ruled.** I checked these against the code:

- **The tied ambient direction takes `D` forms.**
  - The listed functions (`fv_share_tucker_variations`, `_tied_filtered_solve`, `_tied_filtered_solve_transpose`,
    `fv_tied_ambient_directions`, `utv_to_ut3`'s docstring, and `test_degenerate_point_min_norm`) exist and hold the
    `Udot` named, and `Ddot` is free in each (`ast`).
  - Their earlier `B` rows are gone, so nothing is listed twice.
  - The lift's raw direction stays `B`: `tv_corewise_lift`, `utv_corewise_lift`, `fv_tied_lift_factors`' `Udot_g`, and
    the `sharing.py:788–802` doctest, whose `Udot` is `tv_corewise_lift`'s output.
- **The `update` helper is amended.**
  - The 2026.4.0 sentence now fires for every `str`, and for any collection holding `'O'`. Its example points to
    `('D', 'L')`, not `('U', 'L')`.
  - §1.3, §2 and D-2 now say plainly that tuple callers of 2026.4.0 run silently, with only `'O'`-holding collections
    raising, and the release notes say so.
- **Battery B5's fixture produces a real pad.** `from_t3frame(F_G, None, 7, 6)` on `xG`: the Tucker store
  `(6, 6, 2)` and the up store `(3, 6, 2)` fit `nD = 7`, `nU = 6`, so `nD ≠ nU` and pad = 1. Positional, so the old
  tree reads it identically.
- **The rest:**
  - L7 and K3 cover prose word tuples, with ownership assigned to R3.
  - Tier D exempts T4S citations.
  - Carve-outs are by block, with N17's alias included.
  - K9 checks `UP`/`DOWN`.
  - `rem:embedding` is one range.
  - The wordings ("need not raise", "letter tuple", "four pairs") and R3's module doctests are as listed.

No MAJOR or MINOR finding remains. I concur with the session's §13 recommendation: start R1. R1's hit table,
freeness check, `letter_sites` list and diagram inventory are the mechanical form of the enumeration that the three
sample batches did by hand.

VERDICT: AGREE
