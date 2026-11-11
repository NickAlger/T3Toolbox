# The store rename plan — critic review, round 17 (draft 17, confirmation)

Reviewed: the draft-17 diff against `4e5cd26c`, checked against `dev/store_rename_sample_edits_2.md` §11–§12 and the
code.

**All fifteen §12 issues are fixed.** I checked these against the code:
- **§12.2, the `ufv_masking.py` rows.** I traced both functions token by token after A2/A3. In
  `ufv_apply_frame_masks` the down Tucker store and the left/right middle legs end up masked by `DM_*` (`nD`) and the
  up core by `UM_t` (`nU`). In `ufv_apply_variations_masks`, V's rows end up masked by `UM_k` (`nU`) and H's middle leg
  by `DM_t` (`nD`). Both match K13's table. The images are free (`UM_t↔DM_t` as a pair).
- **§12.6.** `K8` now bans `Ud`, carving out `o1_ragged.py` and `o1_uniform.py`. Both read `Ud = np.asarray(u.to_dense())`.
- **§12.8.** `Z` is unused in `shared_t3_math.tex`. The `K_i` range `:279–292` stops before `:306`'s factor `B_i`.
- **§12.4.** The kernels' letters now rest on their A1 keywords, and `_tangent_z_jets_step`'s `dD` on its
  `var_tucker_cores` (a tangent's V).
- **§12.7.** The corewise content/slot rule (`B` content, `D` slot) is stated once for §6 and applied at the listed C4
  sites.
- **§12.5 and §12.10–§12.15.** B6, the HISTORICAL corrections, the lift written in words beside `L`, the two
  P-blocks, and item 12's one-size-one-label rule are as listed.

§12.9 is held for Nick with two options, as it should be.

## NIT

- §4 item 10 ("R4a also realigns the bodies and doctest examples the rename touched") may touch neighbouring lines to
  restore a column. Say "whitespace only; K15 confirms the NAME stream is unchanged by it".

VERDICT: AGREE
