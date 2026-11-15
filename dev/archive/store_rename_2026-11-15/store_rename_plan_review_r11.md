# The store rename plan — critic review, round 11 (over-reach)

Reviewed: draft 11 (1,063 lines, 02:06), the r10 changes.

**Both r10 minors and the nit are resolved.**

- **Doctest-example letters are scheduled once.** They are renamed in R2 through `letter_sites.md` and checked by K15
  and K12. §4.4 is limited to comment and docstring prose, and §4A(a)'s companion sentence agrees.
- **§1.5 item 12 is bounded.** It keeps the prefix `u` (= uniform) and the `ut3_`/`ufv_`/`utv_` prefixes. Names under
  `_u_` change only by the token map, and `uc`/`um` are up-store locals changed by tier C.
- **Item 11 cites the uniform doctests' `gU`.**

**Read literally, the plan now leaves no room to over-reach.**
- §0's list is exhaustive.
- Every letter rename is a listed row: tier C, R1's `letter_sites.md`, §3.4's classified hits, or a §4A block.
- §1.5 names what stays.
- The brief forbids extending a pattern.
- K14 holds the DO-NOT-TOUCH names, and K15 fails on any NAME token change that no hit-table row or listed hunk
  accounts for.

## NIT

- Item 12 places `um` in `uniform_frame_variations_format.py`, which has none; tier C maps `um↔dm` in
  `ufv_conversions.py` and `ufv_operations.py`. Name those two files.

VERDICT: AGREE
