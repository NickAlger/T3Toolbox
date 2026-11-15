# Critic review r5 — `dev/aligned_padded_frame_impl_plan_2026-11-09.md` (round 4, CHANGES rows 4.1–4.3)

Scope: rows 4.1–4.3 only (§0, §2.3's Newton table and zero-passes paragraph, §3.5's `max_passes=0` line, §7, §10,
§11, T6, T14(d), D-36, D-38).

The design change is sound. D-38 is consistent:
- `max_passes = 0` builds the start directly at the target ranks, the same call as `fv_padded_frame` /
  `ufv_padded_frame`, so T6's exactness claim holds;
- `project` is never called, and T6 checks this with a raising `project`;
- the uniform result is compacted, and at the no-op level the two returns coincide;
- the `max_passes = 1` cases are right;
- the `'cone'` at zero passes is `fv_zero_point_frame`'s frame, so T14(d) "= `'zero'` to rounding" stands for any
  `cone_oversampling`;
- D-36 is amended without contradiction.

The table cells match the stated reading except where noted below.

## What remains

### 1. MINOR — §0 and §2.3 "The reading": "ANY injected generic frame … 49 → 21–33" overstates the table

Row (f) (p = 0, 4 passes) has level-2 Newton counts **42** / 29. 42 lies outside 21–33 and within ~7 of the
library's 49. The "~5-iteration noise" also has no stated source, and the table's own seed spreads exceed it: (c)
level 1 is 17 vs 11, and (f) level 2 is 42 vs 29.

**Fix:** "every arm but one run: 21–33 (row (f): 42 / 29)". Either derive the noise from the seed pairs or say "seed
spread up to ~13". §0's basin bullet should also carry §2.3's exception: one of the two (c) runs ends in the other
basin.

### 2. MINOR — §2.3, D-33, D-38: zero passes adds ignored options, so two statements need updating

- §2.3 still says "The D-36 clamp below is the only other place a set value is not honoured". D-38 adds another:
  `oversampling`, `oversampled_passes`, `tol` and `alternate` are ignored at `max_passes = 0`.
- D-33's house rule ("an option where it would have no effect is a structural error") then applies at the optimizer
  level: `padded_max_passes=0` together with an explicitly set `padded_tol` / `padded_alternate` /
  `padded_oversampling` / `padded_oversampled_passes`, and likewise for `cone_*`.

Decide one of two:
- raise there (consistent with `cone_*` under `g0norm_newton`), with a T14(d) row;
- or record it as a documented exception in D-33 / D-38.

Then fix the §2.3 sentence either way.

### 3. NIT — §11: a duplicated bullet

"T3Polynomial adopts `padded_from` and `g0_reference='cone'` …" appears twice: before the baseline table and again
under "Acceptance".

### 4. NIT — T14(d): the uniform comparison is after compaction

"The iteration-0 frame equals `fv_padded_frame` / `ufv_padded_frame` at the target, ≤ 1e-14". On the uniform layer,
compare against `ufv_compact_frame(ufv_padded_frame(...)[0], level_widths)`: the iteration-0 frame is the compacted
one, and the block-layout start has other widths.

### 5. NIT — §0, "established value" (2)

The Newton table does not compare `'cone'` with `'zero'`; every arm used the cone as its absolute stop. Cite the
evidence for the reference's value (the 2026-11-02 zero-reference investigation; D-5 / D-7) beside the claim, or
call it "the design's second deliverable" rather than "established".

VERDICT: REVISE
