# Glossary review, round 2 (critic)

I re-reviewed the items the writer changed after round 1 (`dev/glossary_review_r1.md`). That covers 241 entries, and
the work list now parses to 398 rows. Everything below was checked against the source.

## Round-1 findings: status

- **MAJOR 1 (orthogonality adjectives): resolved.**
  - The "orthogonal core" entry now states the method convention: a Tucker core is down-orthogonal
    (`B Bᵀ = I`), and a TT core is up/outer-orthogonal in its mode-leg unfolding.
  - The entry flags the frame-page clash. I confirmed it in source: `U` is called "up orthogonal" at
    `frame_variations_format.py:1371`, `uniform_frame_variations_format.py:1122` and `probing.py:646`, and `O` is
    called "down orthogonal".
  - N17 is a decision for Nick, with its rows HELD. It includes `tucker_tensor_train.py:2978`, `:3002` and `:3108`,
    which contradict even the method convention.
- **MAJOR 2 (the `:962` row): resolved.** The row is dropped, and N4 now names it as the boundary-bond exception.
  - The N4 recheck is sound. I read every remaining N4 row against its line: `frame_variations.md:71, 83`,
    `tv_operations.py:164, 181`, `ufv_operations.py:277, 294`, and `frame_variations_format.py:63, 72, 355`. Each one
    renames a core or store name or symbol, and none touches an orthogonality adjective.
  - N4 is only incomplete (see the first NIT).
- **MINOR 1 to 12: resolved.** Notes on specific items:
  - Group ceiling: now `r_i·r_{i+1}`, with E19 rows for `sharing.md`.
  - Movement: the NaN padding is now attributed only to `ufv_aligned_padded_frame`'s returned array.
  - Absorb: `.absorb` is replaced by `T3Tangent.absorb_weights`.
  - Minimal ranks: `x0` is reduced by the fitting layer.
  - Gappy mask: `+` only, plus the weights' `kronecker`. New E4 rows are at `user_guide.rst:255` and
    `uniform_masks_vs_ranks.md:44, 53`.
  - Store: the sweep order is right (`ranks.py:644-674`).
  - Sampling kind: the six operations match `backend/fitting.py:336-380`.
  - Stratum / variety: the masks now name the "nominal" stratum.
  - Frame cores: down is now TT-shaped.
  - New entries "left-/right-orthogonal form" and "shape mask": both true (`tucker_tensor_train.py:780, 814`;
    `ut3_masking.py:70-79`).
  - N12 is now a decision for Nick, with no rows.
  - New rows: E22 (now 13 rows), the E17 shape rows (`fv_operations.py:98` is right: `right[i].shape[-1]` is
    `rR(i+1)`), and E1 at `optimizers.py:551`.
- **MINOR 13: resolved.** All 74 E6 rows now give replacement text. A scripted check found every "current" fragment at
  its line, and no review code left in any replacement. There are eight runtime messages; the eighth,
  `common.py:628` (`ValueHashedFields._value_key`'s `TypeError`), is a real runtime message I missed.
- **NITs 1 to 11:** done, except that `HISTORICAL` is kept. I accept that, since the coordinator's brief names it.

## New findings

### MINOR

1. **The E4 row for `uniform_frame_variations_format.py:52` (`UT3FrameMasks`) can't be applied mechanically, and the
   replacement it points to would be wrong for frames.**
   - The row says "same sentence as above / same replacement". The nearest replacement above it is `ut3_masking.py:10`'s
     "possibly gappy after arithmetic". The one at `UT3Masks` says "gappy after `+`".
   - Frames have no `+`. A `UT3Frame`'s masks go gappy in a padded frame's **block layout**: `UT3Frame.padded` and
     `ufv_padded_frame` return `[old block | new block]` per slot, each block with its own mask, and the old block
     keeps any mask pattern (`uniform_padded_frames.py:7-9, 179`).
   - Fix: give the row its own text, for example "Slot ``j`` of an edge is real iff its mask is ``True`` there;
     canonically a prefix, gappy in a padded frame's block layout."

### NIT

1. **N4 is incomplete for `D` → `O` in the same docstrings.** These should also change:
   - `frame_variations_format.py:101`: the rank diagram's `D1` (the line below already says `nO1`).
   - `:355`: the `einsum(..., D, D)` on the same line as the listed row.
   - `:1271`: the `fv_to_t3` diagram's `D2`.
2. **The E6 row at `fitting.py:642` keeps "(the 2026-08-22 review)".** That is still an internal reference a user
   cannot resolve; drop the parenthesis.
3. **C1: `'cone': 'cone'` reads flat beside 'zero tensor' and 'warm start'.** Something like
   `'cone': 'cone (aligned zero tensor)'` would fit better. This is Nick's call.

**`sphinx -W`:** the rebuild of the round-2 tree is recorded below.

Verdict: REVISE (0 MAJOR, 1 MINOR, 3 NIT)
