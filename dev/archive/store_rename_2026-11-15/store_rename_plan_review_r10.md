# The store rename plan — critic review, round 10 (over-reach)

Reviewed: draft 10 (1,051 lines, 02:04), the r9 changes, read literally.

**Resolved.**
- **O1, the closure gate.** K15 is positional after R2 and hunk-listed after R3–R5, with the set-difference
  cross-check.
- **O2, the two missing classes.** §1.5 items 11 and 12 add them. §3.1's T3 bullet is bounded ("Only these become
  `D`", then "every other T3 `U` stays"). The T3-level `O_ab_x`/`Oaxb` and the T3 test's `dU`/`'U'` are kept and carved
  out of K8.
- **MINOR 1.** The catch-all row is now R1's enumerated `letter_sites.md`.
- **MINOR 2.** Every sentence I listed is bounded to its sites.
- **MINOR 3.** D-2 lays out (c).
- **The nit.**

The withdrawals (B5, `Uio`→`Bio`, the T3 pages' `B` renames in code) shrink the change set, which is the safe
direction; Nick should see them as a change.

## MINOR

1. **The doctest-example letter rows are scheduled twice, and K15 will trip on one of them.**
   - `letter_sites.md` (tier C, applied in R2) takes "every CL row of inv 1 §7". That includes the doctest-example
     rows, which are NAME tokens to K15: `frame_variations_format.py:1302–1322`'s `U0…`/`D0…`, about 40 doctest lines
     in that file, about 37 in `uniform_frame_variations_format.py`, and `manifold.py:202`.
   - But §4A(a) says the companions "are §4.4 rows in R4a", and §4.4 takes "CL rows … that fall in … docstrings".
   - After R2, K15 expects exactly the hit table's positions. After R4a, it allows only the hunks it lists (§4.11,
     item 12, §3.4), and those do not include §4.4's doctest rows.
   - Fix: schedule them once, in R2 through `letter_sites.md`, where they get simultaneity and the positional check.
     Limit §4.4 to comment and prose letters, and correct §4A(a)'s companion sentence.
2. **§1.5 item 12's "every `u` … prefix" collides with store names.**
   - In tier C, `u` means *up*: `uc`→`c_down` and `um`↔`dm` (`ufv_conversions.py`,
     `uniform_frame_variations_format.py`).
   - Under a `_u_` prefix, A2 renames `_u_up_messages`↔`_u_down_messages`.
   - Read literally, item 12 keeps `uc`/`um`, and R1's proof that the map and the list are disjoint fails.
   - Fix: say the prefix `u` (= uniform) and the `ut3_`/`ufv_`/`utv_` module and function prefixes are untouched.
     Names under a `_u_` prefix change only by the token map, and `uc`/`um` are up-store locals mapped by tier C, not
     uniform.

## NIT

- Item 11 names "the `TuckerTensorTrain` doctests' `gU`". Add the `UniformTuckerTensorTrain` doctests
  (`uniform_tucker_tensor_train.py:566–567, 631–632`).

VERDICT: REVISE (0 MAJOR, 2 MINOR)
