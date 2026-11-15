# R6 alignment sweep (store rename, 2026.5.0)

Nick's requirement (`dev/HANDOFF.md`, 2026-11-12): every block the rename touched is re-aligned before the arc
completes, "within reason", readability for humans the principle, `t3toolbox/tucker_tensor_train.py` the style
reference. The orchestrator's R6 brief widens it to every aligned block of every file the arc touched: the 21
one/two-column near-misses R4a's reviewer counted in touched lines, the two in R3's new test lines, R5's review N3,
and any other. Whitespace only. Nothing is committed.

## Result

- **277 lines in 56 files** (Revision 1 included), every one whitespace-only: `git diff -w HEAD` is empty; every
  changed `.py` file has the same token stream as HEAD (docstring whitespace aside); K15 against HEAD **0 NAME
  hunks**; no expected doctest output changed (37 doctest source lines moved a trailing comment; 3,177 examples, the
  same wants as the tag).
- **Adjacent lines:** no comment or doctest column within two columns of its neighbour's is left (NEAR 0 for those
  kinds over the 109 scanned files); of 3,334 adjacent pairs that shared a column at the tag, 0 drifted by the arc,
  and 2 were moved on purpose by this sweep (J1). "0 near-misses / 0 drifted" in the first pass meant adjacent pairs
  only.
- **Across one or two comment-less lines** (`--bridge`, Revision 1): 74 bridged pairs 1–2 columns apart before
  Revision 1, 19 after, every one two separate groups (J3b); bridged pairs that shared a column at the tag and no
  longer do: 0.
- 42 of the changed lines were touched by the arc; 235 were as at the tag (pre-existing slips, the reference
  module's included).
- No inventoried diagram or equation block was touched (K10/K10b/K11/K12 PASS, 301/301 verbatim). The six `.tex`
  notes and the PDFs are out of scope (no comment columns; their blocks are hand-only and K12-checked).

## Revision 1 (the R6 review's bridged findings, 2026-11-13)

The first pass paired adjacent lines only, so a column split by a comment-less line went unseen, both as a near-miss
and as drift. The reviewer's bridge scan (1–2 columns apart across one or two comment-less non-blank lines) found 74
pairs. Every one was judged; 62 lines moved ("bridged (review)" in the table), all whitespace:

- **A1, columns the first pass split** (one column at HEAD): `test_padded_frame.py:1041, 1043` join the running
  sentence `:1038–1039` at 90; `manifold.py:1089–1091` join the J5 re-pad of `:1093–1095` at 59; `test_sharing.py:556`
  joins `:553–554` at 76.
- **A2, arc drift across a comment-less line** (one column at the tag): `fv_conversions.py:207` ("1. Tucker
  down-orth") joins steps 2–4 at 72; `geometry.py:129` joins `:132` at 70; `padded_frames.py:297` joins `:292`,
  `:294` at 89 (R2's `up` → `down` had moved `:294`).
- **A3, twins of first-pass fixes:** `t3_svd.py:720, 725, 727` (`dense_t3svd`, the twin of J8's `:510`, `:620`) at
  77 with `:721–722`; `test_dispatch.py:1811` joins `:1809` at 76.
- **A4, staircases that are one block** (one indent, one doctest or statement group): `tv_operations.py:807, 809,
  810` → 82 with `:813`; `manifold.py:1332`, `:1569` → 50; `fitting_and_optimization.md:421` → 80, `:707` → 63;
  `fitting.py:938` → 88; `test_uniform_frame_variations_format.py:816` → 77, `:96` → 73, `:239` → 98, `:252` → 86,
  `:461` → 60, `:898` → 72; `test_padded_start.py:206` → 77; `test_frame_variations_format.py:307` → 80;
  `optimizers.py:806` → 75; `sampling_derivatives.py:1771`, `:2944` → 56; `ufv_operations.py:236` → 53;
  `uniform_padded_frames.py:574` → 80; `utv_operations.py:743` → 84; `padded_frames.py:777` → 65 (the function's
  running comment column); `safety.py:60` → 62; `tucker_tensor_train.py:4566` → 63, `:4479`, `:4481` → 76 (with
  `:4484`, `:4486`; moving `:4481` up to `:4479` would have split it from `:4484`, which shared its column at the tag);
  `uniform_manifold.py:1100` → 59; `uniform_tucker_tensor_train.py:986`, `:988` → 71; `test_ranks.py:141` → 72;
  `test_chunk_size_estimator.py:124` → 69; `test_dispatch.py:1730` → 62, `:1857` → 72; `test_fitting.py:675` → 88
  (one list of cases, though `:677` on is nested in `with`); `test_manifold.py:1166` → 74, `:1188`, `:1192` → 87;
  `test_uniform_probing.py:938` → 91; `test_weighted.py:255` → 70.
- **Signature blocks across a multi-line type:** the `variations:` line of `entries.py:118`, `fv_operations.py:632,
  660, 740` (→ 80) and `:752–753` (→ 82) joins the block's column below the `weights:` continuation; the return-line
  comments of `ut3_sampling.py:241` (→ 40) and `tucker_tensor_train.py:3645` (→ 42), `:3720`, `:3781` (→ 41) join
  their argument blocks (padded, so meant to align, as in the first pass's `fitting.py:406`).
- **NIT:** `test_padded_frame.py:1091` → 98, so the three-line comment `:1091–1093` is one column (a 3-column move of
  one line).

The remaining 19 bridged pairs are J3b. `r6_align_scan.py --bridge` lists them (report only).

## Method

`dev/store_rename_slices/r6_align_scan.py` (new, one-off, in `dev/` like the other gate tools; the t3toolbox env)
scans every `.py`, `.md`, `.rst` file in `git diff --name-only v2026.4.0 HEAD` outside `dev/` (109 files) for three
column features: a trailing comment after code (code lines, doctest source lines, markdown code blocks), the type
column of a `name:   type` signature/field line, the `=` of a padded assignment. A pair of consecutive lines with the
same feature (a doctest comment across up to two output lines; a type line across its own multi-line type) is an
**intended** alignment when one line is padded (comment gap ≥ 3 spaces, type/`=` gap ≥ 2). It reports drift against
the tag, NEAR (1–2 columns apart), FIT (≥ 3 apart, both lines could take the other's column) and counts LONG (≥ 3
apart, one line's code too long for the other's column: a stick-out).

The fixes ran in three passes, each reviewed as a diff by eye before it reached the tree:

1. **Chains** (automatic, then read): a maximal run of consecutive feature lines whose columns differ by ≤ 2 goes to
   the run's majority column ("chain"); when a line's code is too long for it by 1–2 columns, the whole run moves to
   that line instead ("re-pad", the reference's own habit, e.g. `tucker_tensor_train.py:691–692`). Type columns
   moved only where a padded line slipped ("type slip"); a long NAME that cannot reach the block's column is left
   (J1). A run touching an inventoried block would have been skipped; none did.
2. **Hand fixes** ("hand"): the FIT pairs judged to be slips, chains broken by a comment-less line, one arc-created
   near-miss in an assignment block, one first-pass move reverted (J6).
3. **Re-scan** until only the judgement classes below remain.
4. **Revision 1** (above): the bridged pairs, judged one by one.

## The known items

The reviews quote 0-based columns; this table and the appendix use 1-based ones.

| item | disposition |
|---|---|
| R4a review: `optimizers.py:228`, `frame:` one column right of its block (the backend `Problem` fields) | type column 16 → 15, its comment with it (31 → 30): on the block's columns |
| R4a review: `utv_operations.py:276–279` (0-based 80/79/78/79) | `:276`, `:278` → 80, with `:277`, `:279`; the twin runs `:288–291` and `:298–302` likewise (`:291`, `:301` → 80) |
| R4a review (revision 1): `utv_operations.py:486–487` (0-based 74/75) and `:300–302` (79/78/79), the tag's own | `:487` → 75 with `:485–486`, and `:484` joins them by hand (72 → 75); `:301` above |
| R4a review: `ufv_conversions.py:407–408` (the return-annotation pair) | `:407` 47 → 48 |
| R4a review: `test_uniform_manifold.py:111–112` | the whole mask block `:111–118` to `:112`'s column, 84 (`:112` sticks out by 2; `:114–118` by hand across the comment-less `:113`, `:117`) |
| R3's new test lines `test_frame_variations_format.py:544–545` | `:544` 86 → 87, on `:545–547`'s column |
| R5 review N3: `numerical_contracts.md:105–107` doctest comments (0-based 88/87) | `:107` 88 → 89, on `:103`, `:105`'s column |
| the rest of the 21 | the review did not enumerate them; every comment near-miss in the touched files is fixed (0 left) |

## Judgement calls

- **J1 — long names stick out of a type column (left).** 24 type pairs stay 1–2 columns apart because one NAME is too
  long for its block's type column and is written `name: type` with one space: `sum_over_probes` (5 signatures in
  `probing.py`, `sampling_derivatives.py`), `frame_sweep` (6 `*_from_sweep` signatures in `apply.py`, `entries.py`,
  `probing.py`), `variations_data` (3, `sharing.py`), `shared_data` (`tv_operations.py:350`), `NewtonInfo`'s
  `regularization` and `lift_sigma_min` (`backend/optimizers.py`), `uniform_least_squares_problem`'s `regularizer`,
  `chunk_size` (`uniform_fitting.py`), `t3_svd`'s `assume_orthogonal`. Re-padding a whole signature (every argument,
  its comment, the continuation and return lines) for one name was judged not worth it; the reference lets a long
  name stand. Two consequences, both deliberate and reported by the tool as the only drift:
  - `NewtonInfo.misfit` now sits on the dataclass's type column; at the tag it stood two columns right, paired with
    `regularization` (two ragged lines become one);
  - `uniform_least_squares_problem`'s `sharing` likewise joins its block's column (at the tag it was paired with
    `chunk_size`'s forced column).
  In the six `*_from_sweep` signatures `frame:` was one column right of the arguments above it, matching the later
  `frame_sweep:`; it now sits on the arguments' column and `frame_sweep:` alone sticks out (the other 29
  `frame:` lines the sweep moved, in signatures without a `frame_sweep`, were a copied slip of one or two columns).
- **J2 — comment stick-outs of ≥ 3 columns (left).** 455 LONG pairs: a line whose code is longer than its block's
  comment column carries its comment two spaces after the code, as in the reference (`tucker_tensor_train.py:2066–2075`).
  Re-padding a block to a line 3–40 columns longer would push its other lines far right.
- **J3 — separate groups (left).** The remaining 26 FIT pairs are two groups with their own columns, not a slip: a
  return-annotation tuple's elements against the `]:` line, whose comment follows the argument block (14 of them); the argument block against the return line (`fitting.py:77`, `probing.py:1663`, `padded_frames.py:226`,
  `uniform_padded_frames.py:137`); `NewtonInfo`'s fields without defaults against those with; a multi-line
  expression's comments against the statement above (`sampling_derivatives.py:1821`, `utv_operations.py:351`);
  doctest sub-blocks separated by output lines (`tucker_tensor_train.py:3418–3422`, `:3620–3624`); an unpadded one-space
  comment (`tucker_tensor_train.py:978`); `self:` (`:4419`); two nesting levels (`test_fitting.py:1406–1407`).
- **J3b — bridged pairs that are separate groups (left, Revision 1).** The 19 bridged pairs still 1–2 columns apart
  all join lines of different indents: a statement and a nested or enclosing block (`apply.py:119/122`,
  `optimizers.py:1644/1647`, `utv_operations.py:293/295`, `test_optimizers.py:443/445`, `:1051/1054`,
  `test_uniform_curvature.py:2130/2133`, `test_manifold.py:1043/1045`); a call's continuation line and the next
  statement (`test_curvature.py:1578/1581`, `:1837/1839`, `test_optimizers.py:561/563`,
  `test_aligned_padded_frame_frontend.py:271/274`); an argument block and the return-annotation tuple or an inner
  tuple's elements (`padded_frames.py:98/100`, `sharing.py:1178/1180`, `:1199/1201`, `ufv_conversions.py:202/204`,
  `:208/210`, `:309/311`, `:315/317`); a constant and the next statement's list (`test_jet_recurrence.py:32/35`).
- **J4 — unpadded assignments beside an aligned pair (left).** Six `=` pairs (`t3_linalg.py:468–469`,
  `utv_operations.py:320–321`, `uniform_manifold.py:532–533`, `test_optimizers.py:528–529`, `test_ranks.py:137–138`,
  `:191–192`): a statement next to an aligned pair, never meant to share its column.
- **J5 — re-padding to a line that sticks out by 1–2 columns (done).** 38 lines moved right so a block keeps one
  column (e.g. `least_squares_problem`'s signature, the two stop-rule tests, `t3_linalg.py:388–389`). Seven lines
  that were already 119–138 characters grow by 1–2 to 121–139 (`ufv_operations.py:119`, `utv_operations.py:569`,
  `fitting.py:89`, `test_padded_frame.py:1030, 1031, 1033`, `test_uniform_padded_frame.py:823`); nothing becomes
  absurdly long.
- **J6 — `t3_svd.py:45` `sharing` (reverted).** The first pass moved it under `assume_orthogonal`'s unpadded type; it
  is aligned with `max_tt_ranks` / `max_tucker_ranks` above, so it stays as at the tag.
- **J7 — `test_manifold.py:693–695`** had three columns (104/74/99). They now share the most compact column that fits
  all three (85), instead of moving two lines to 104 (lines of ~125 characters).
- **J8 — ≥ 3-column slips fixed by hand.** Adjacent lines of one statement kind, one indentation and no structural
  boundary: `weighting.md:34–37` (two at 44, two at 47 → 44) and `:169` (54 → 50); `t3_svd.py:510`, `:620` (the
  graded-spectrum line joins the two below it); `ufv_operations.py:505`; `ut3_orthogonalization.py:52` and `:65`
  (the `elif` under its `if`); `utv_operations.py:261` and `:484`; `fitting.py:626`;
  `sampling_derivatives.py:1893` (a sub-term's comment six columns right of its expression's); the type column of
  `ufv_conversions.py:68` (`squash_tails` under `already_left_orthogonal`).
- **J9 — one near-miss the rename created in an `=` block.** `test_uniform_frame_variations_format.py:103`: the tag's
  unpadded `up = np.ones(...)` became `down = ...`, one column short of the block's `=`; now `down  =`.
- **Bridging a comment-less line.** Two chains split by a line without a comment were joined by hand
  (`test_padded_frame.py:1035, 1038–1039` with the stop-rule block above; `test_uniform_manifold.py:114–118`). A
  general bridge was tried and dropped: it merged argument and return-tuple groups.

## Every fix

Column numbers are 1-based. "chain", "re-pad", "type slip": pass 1; "hand": pass 2; "bridged (review)": Revision 1.
"touched by the arc": the line differs between `v2026.4.0` and HEAD. Counts: 114 chain, 38 re-pad, 39 type slip
(2 of them also chain), 24 hand, 62 bridged (review).

| file | line | column (1-based) before → after | class | the line at the tag |
|---|---|---|---|---|
| `docs/entries_apply_probe.md` | 61 | comment 41 → 42 | chain | unchanged since the tag |
| `docs/fitting_and_optimization.md` | 416 | comment 81 → 80 | chain | unchanged since the tag |
| `docs/fitting_and_optimization.md` | 421 | comment 81 → 80 | bridged (review) | unchanged since the tag |
| `docs/fitting_and_optimization.md` | 707 | comment 64 → 63 | bridged (review) | unchanged since the tag |
| `docs/numerical_contracts.md` | 107 | comment 88 → 89 | chain | touched by the arc |
| `docs/weighting.md` | 36 | comment 47 → 44 | hand | unchanged since the tag |
| `docs/weighting.md` | 37 | comment 47 → 44 | hand | unchanged since the tag |
| `docs/weighting.md` | 169 | comment 54 → 50 | hand | unchanged since the tag |
| `docs/weighting.md` | 203 | comment 69 → 70 | chain | unchanged since the tag |
| `examples/fit_hilbert_from_entries_lbfgs.py` | 87 | comment 39 → 38 | chain | unchanged since the tag |
| `examples/fit_hilbert_from_entries_lbfgs.py` | 88 | comment 39 → 38 | chain | unchanged since the tag |
| `examples/fit_hilbert_from_entries_lbfgs.py` | 147 | comment 68 → 67 | chain | unchanged since the tag |
| `t3toolbox/backend/apply.py` | 147 | type 22 → 21 | type slip | unchanged since the tag |
| `t3toolbox/backend/apply.py` | 178 | type 22 → 21 | type slip | unchanged since the tag |
| `t3toolbox/backend/apply.py` | 199 | type 18 → 17 | type slip | unchanged since the tag |
| `t3toolbox/backend/apply.py` | 239 | type 22 → 21 | type slip | unchanged since the tag |
| `t3toolbox/backend/apply.py` | 270 | type 22 → 21 | type slip | unchanged since the tag |
| `t3toolbox/backend/entries.py` | 118 | comment 79 → 80 | bridged (review) | unchanged since the tag |
| `t3toolbox/backend/entries.py` | 119 | type 22 → 21 | type slip | unchanged since the tag |
| `t3toolbox/backend/entries.py` | 145 | type 22 → 21 | type slip | unchanged since the tag |
| `t3toolbox/backend/entries.py` | 165 | type 18 → 17 | type slip | unchanged since the tag |
| `t3toolbox/backend/entries.py` | 200 | type 22 → 21 | type slip | unchanged since the tag |
| `t3toolbox/backend/entries.py` | 222 | type 22 → 21 | type slip | unchanged since the tag |
| `t3toolbox/backend/fitting.py` | 76 | comment 61 → 62 | chain | unchanged since the tag |
| `t3toolbox/backend/fitting.py` | 201 | comment 43 → 44 | chain | unchanged since the tag |
| `t3toolbox/backend/fv_conversions.py` | 207 | comment 74 → 72 | bridged (review) | touched by the arc |
| `t3toolbox/backend/fv_operations.py` | 289 | type 16 → 18 | type slip | unchanged since the tag |
| `t3toolbox/backend/fv_operations.py` | 295 | comment 61 → 62 | chain | unchanged since the tag |
| `t3toolbox/backend/fv_operations.py` | 632 | comment 79 → 80 | bridged (review) | unchanged since the tag |
| `t3toolbox/backend/fv_operations.py` | 660 | comment 79 → 80 | bridged (review) | unchanged since the tag |
| `t3toolbox/backend/fv_operations.py` | 740 | comment 79 → 80 | bridged (review) | unchanged since the tag |
| `t3toolbox/backend/fv_operations.py` | 752 | comment 81 → 82 | bridged (review) | unchanged since the tag |
| `t3toolbox/backend/fv_operations.py` | 753 | comment 81 → 82 | bridged (review) | unchanged since the tag |
| `t3toolbox/backend/geometry.py` | 129 | comment 72 → 70 | bridged (review) | touched by the arc |
| `t3toolbox/backend/geometry.py` | 638 | comment 91 → 92 | chain | unchanged since the tag |
| `t3toolbox/backend/optimizers.py` | 228 | type 16 → 15 | type slip | touched by the arc |
| `t3toolbox/backend/optimizers.py` | 461 | comment 37 → 39 | re-pad | unchanged since the tag |
| `t3toolbox/backend/optimizers.py` | 462 | comment 37 → 39 | re-pad | unchanged since the tag |
| `t3toolbox/backend/optimizers.py` | 463 | comment 37 → 39 | re-pad | unchanged since the tag |
| `t3toolbox/backend/optimizers.py` | 464 | comment 37 → 39 | re-pad | unchanged since the tag |
| `t3toolbox/backend/optimizers.py` | 806 | comment 74 → 75 | bridged (review) | unchanged since the tag |
| `t3toolbox/backend/optimizers.py` | 929 | comment 60 → 61 | chain | unchanged since the tag |
| `t3toolbox/backend/optimizers.py` | 963 | comment 36 → 38 | re-pad | unchanged since the tag |
| `t3toolbox/backend/optimizers.py` | 964 | comment 36 → 38 | re-pad | unchanged since the tag |
| `t3toolbox/backend/optimizers.py` | 965 | comment 36 → 38 | re-pad | unchanged since the tag |
| `t3toolbox/backend/optimizers.py` | 967 | comment 36 → 38 | re-pad | unchanged since the tag |
| `t3toolbox/backend/optimizers.py` | 1190 | type 21 → 19 | chain, type slip | unchanged since the tag |
| `t3toolbox/backend/padded_frames.py` | 297 | comment 87 → 89 | bridged (review) | touched by the arc |
| `t3toolbox/backend/padded_frames.py` | 715 | comment 46 → 47 | chain | unchanged since the tag |
| `t3toolbox/backend/padded_frames.py` | 746 | comment 64 → 65 | chain | unchanged since the tag |
| `t3toolbox/backend/padded_frames.py` | 772 | comment 64 → 65 | chain | touched by the arc |
| `t3toolbox/backend/padded_frames.py` | 777 | comment 64 → 65 | bridged (review) | touched by the arc |
| `t3toolbox/backend/probing.py` | 595 | type 22 → 21 | type slip | unchanged since the tag |
| `t3toolbox/backend/probing.py` | 638 | type 22 → 21 | type slip | unchanged since the tag |
| `t3toolbox/backend/probing.py` | 778 | type 18 → 17 | type slip | unchanged since the tag |
| `t3toolbox/backend/probing.py` | 813 | comment 46 → 47 | re-pad | unchanged since the tag |
| `t3toolbox/backend/probing.py` | 814 | comment 46 → 47 | re-pad | touched by the arc |
| `t3toolbox/backend/probing.py` | 815 | comment 48 → 47 | re-pad | unchanged since the tag |
| `t3toolbox/backend/probing.py` | 1176 | type 22 → 21 | type slip | unchanged since the tag |
| `t3toolbox/backend/probing.py` | 1213 | type 26 → 25 | type slip | unchanged since the tag |
| `t3toolbox/backend/probing.py` | 1295 | comment 55 → 56 | chain | unchanged since the tag |
| `t3toolbox/backend/probing.py` | 1317 | comment 55 → 56 | chain | unchanged since the tag |
| `t3toolbox/backend/probing.py` | 1509 | comment 63 → 64 | chain | touched by the arc |
| `t3toolbox/backend/sampling_derivatives.py` | 191 | comment 46 → 47 | chain | touched by the arc |
| `t3toolbox/backend/sampling_derivatives.py` | 443 | comment 71 → 72 | chain | unchanged since the tag |
| `t3toolbox/backend/sampling_derivatives.py` | 560 | type 19 → 18 | type slip | unchanged since the tag |
| `t3toolbox/backend/sampling_derivatives.py` | 579 | type 19 → 18 | type slip | unchanged since the tag |
| `t3toolbox/backend/sampling_derivatives.py` | 602 | type 18 → 17 | type slip | unchanged since the tag |
| `t3toolbox/backend/sampling_derivatives.py` | 624 | type 18 → 17 | type slip | unchanged since the tag |
| `t3toolbox/backend/sampling_derivatives.py` | 644 | type 18 → 17 | type slip | unchanged since the tag |
| `t3toolbox/backend/sampling_derivatives.py` | 784 | comment 55 → 56 | chain | unchanged since the tag |
| `t3toolbox/backend/sampling_derivatives.py` | 904 | type 22 → 21 | type slip | unchanged since the tag |
| `t3toolbox/backend/sampling_derivatives.py` | 935 | type 22 → 21 | type slip | unchanged since the tag |
| `t3toolbox/backend/sampling_derivatives.py` | 1081 | type 22 → 21 | type slip | unchanged since the tag |
| `t3toolbox/backend/sampling_derivatives.py` | 1109 | type 22 → 21 | type slip | unchanged since the tag |
| `t3toolbox/backend/sampling_derivatives.py` | 1165 | type 22 → 21 | type slip | unchanged since the tag |
| `t3toolbox/backend/sampling_derivatives.py` | 1193 | type 22 → 21 | type slip | unchanged since the tag |
| `t3toolbox/backend/sampling_derivatives.py` | 1771 | comment 55 → 56 | bridged (review) | unchanged since the tag |
| `t3toolbox/backend/sampling_derivatives.py` | 1875 | comment 97 → 99 | chain | touched by the arc |
| `t3toolbox/backend/sampling_derivatives.py` | 1891 | comment 70 → 72 | chain | touched by the arc |
| `t3toolbox/backend/sampling_derivatives.py` | 1893 | comment 78 → 72 | hand | touched by the arc |
| `t3toolbox/backend/sampling_derivatives.py` | 2018 | type 22 → 21 | type slip | unchanged since the tag |
| `t3toolbox/backend/sampling_derivatives.py` | 2057 | type 22 → 21 | type slip | unchanged since the tag |
| `t3toolbox/backend/sampling_derivatives.py` | 2252 | type 22 → 21 | type slip | unchanged since the tag |
| `t3toolbox/backend/sampling_derivatives.py` | 2280 | type 22 → 21 | type slip | unchanged since the tag |
| `t3toolbox/backend/sampling_derivatives.py` | 2312 | type 22 → 21 | type slip | unchanged since the tag |
| `t3toolbox/backend/sampling_derivatives.py` | 2355 | type 22 → 21 | type slip | unchanged since the tag |
| `t3toolbox/backend/sampling_derivatives.py` | 2514 | comment 55 → 56 | chain | unchanged since the tag |
| `t3toolbox/backend/sampling_derivatives.py` | 2552 | comment 93 → 95 | chain | touched by the arc |
| `t3toolbox/backend/sampling_derivatives.py` | 2944 | comment 55 → 56 | bridged (review) | unchanged since the tag |
| `t3toolbox/backend/sampling_derivatives.py` | 3155 | comment 54 → 55 | chain | unchanged since the tag |
| `t3toolbox/backend/sharing.py` | 193 | comment 61 → 63 | chain | unchanged since the tag |
| `t3toolbox/backend/sharing.py` | 579 | comment 45 → 47 | chain | unchanged since the tag |
| `t3toolbox/backend/sharing.py` | 713 | comment 45 → 47 | chain | unchanged since the tag |
| `t3toolbox/backend/sharing.py` | 1030 | comment 45 → 47 | chain | unchanged since the tag |
| `t3toolbox/backend/t3_linalg.py` | 388 | comment 60 → 62 | re-pad | touched by the arc |
| `t3toolbox/backend/t3_linalg.py` | 389 | comment 60 → 62 | re-pad | unchanged since the tag |
| `t3toolbox/backend/t3_operations.py` | 112 | type 15 → 16 | type slip | unchanged since the tag |
| `t3toolbox/backend/t3_svd.py` | 510 | comment 60 → 68 | hand | unchanged since the tag |
| `t3toolbox/backend/t3_svd.py` | 620 | comment 56 → 60 | hand | unchanged since the tag |
| `t3toolbox/backend/t3_svd.py` | 720 | comment 56 → 77 | bridged (review) | unchanged since the tag |
| `t3toolbox/backend/t3_svd.py` | 725 | comment 76 → 77 | bridged (review) | unchanged since the tag |
| `t3toolbox/backend/t3_svd.py` | 727 | comment 76 → 77 | bridged (review) | unchanged since the tag |
| `t3toolbox/backend/tv_operations.py` | 659 | comment 47 → 48 | chain | touched by the arc |
| `t3toolbox/backend/tv_operations.py` | 666 | comment 29 → 28 | chain | unchanged since the tag |
| `t3toolbox/backend/tv_operations.py` | 681 | comment 28 → 29 | chain | unchanged since the tag |
| `t3toolbox/backend/tv_operations.py` | 807 | comment 80 → 82 | bridged (review) | unchanged since the tag |
| `t3toolbox/backend/tv_operations.py` | 809 | comment 81 → 82 | bridged (review) | unchanged since the tag |
| `t3toolbox/backend/tv_operations.py` | 810 | comment 79 → 82 | chain | unchanged since the tag |
| `t3toolbox/backend/ufv_conversions.py` | 68 | type 40 → 34 | hand | unchanged since the tag |
| `t3toolbox/backend/ufv_conversions.py` | 407 | comment 47 → 48 | chain | touched by the arc |
| `t3toolbox/backend/ufv_operations.py` | 119 | comment 41 → 43 | re-pad | unchanged since the tag |
| `t3toolbox/backend/ufv_operations.py` | 120 | comment 41 → 43 | re-pad | unchanged since the tag |
| `t3toolbox/backend/ufv_operations.py` | 236 | comment 54 → 53 | bridged (review) | touched by the arc |
| `t3toolbox/backend/ufv_operations.py` | 459 | comment 75 → 76 | chain | unchanged since the tag |
| `t3toolbox/backend/ufv_operations.py` | 505 | comment 79 → 74 | hand | touched by the arc |
| `t3toolbox/backend/uniform_fitting.py` | 475 | type 21 → 20 | type slip | unchanged since the tag |
| `t3toolbox/backend/uniform_padded_frames.py` | 54 | comment 64 → 63 | chain | unchanged since the tag |
| `t3toolbox/backend/uniform_padded_frames.py` | 574 | comment 82 → 80 | bridged (review) | unchanged since the tag |
| `t3toolbox/backend/uniform_padded_frames.py` | 936 | comment 64 → 65 | chain | unchanged since the tag |
| `t3toolbox/backend/ut3_orthogonalization.py` | 52 | comment 64 → 59 | hand | unchanged since the tag |
| `t3toolbox/backend/ut3_orthogonalization.py` | 63 | comment 74 → 75 | chain | unchanged since the tag |
| `t3toolbox/backend/ut3_orthogonalization.py` | 65 | comment 73 → 75 | hand | unchanged since the tag |
| `t3toolbox/backend/ut3_sampling.py` | 241 | comment 39 → 40 | bridged (review) | unchanged since the tag |
| `t3toolbox/backend/utv_operations.py` | 58 | comment 27 → 28 | re-pad | touched by the arc |
| `t3toolbox/backend/utv_operations.py` | 59 | comment 27 → 28 | re-pad | unchanged since the tag |
| `t3toolbox/backend/utv_operations.py` | 68 | comment 43 → 44 | chain | touched by the arc |
| `t3toolbox/backend/utv_operations.py` | 261 | comment 52 → 55 | hand | unchanged since the tag |
| `t3toolbox/backend/utv_operations.py` | 276 | comment 81 → 80 | chain | touched by the arc |
| `t3toolbox/backend/utv_operations.py` | 278 | comment 79 → 80 | chain | touched by the arc |
| `t3toolbox/backend/utv_operations.py` | 291 | comment 79 → 80 | chain | touched by the arc |
| `t3toolbox/backend/utv_operations.py` | 301 | comment 79 → 80 | chain | touched by the arc |
| `t3toolbox/backend/utv_operations.py` | 484 | comment 72 → 75 | hand | unchanged since the tag |
| `t3toolbox/backend/utv_operations.py` | 487 | comment 76 → 75 | chain | touched by the arc |
| `t3toolbox/backend/utv_operations.py` | 569 | comment 79 → 80 | chain | unchanged since the tag |
| `t3toolbox/backend/utv_operations.py` | 743 | comment 83 → 84 | bridged (review) | unchanged since the tag |
| `t3toolbox/backend/utv_sampling.py` | 163 | comment 27 → 28 | chain | unchanged since the tag |
| `t3toolbox/backend/utv_sampling.py` | 174 | comment 27 → 28 | chain | unchanged since the tag |
| `t3toolbox/backend/utv_sampling.py` | 186 | comment 27 → 28 | chain | unchanged since the tag |
| `t3toolbox/backend/utv_sampling.py` | 197 | comment 23 → 24 | chain | unchanged since the tag |
| `t3toolbox/backend/utv_sampling.py` | 209 | comment 23 → 24 | chain | unchanged since the tag |
| `t3toolbox/backend/utv_sampling.py` | 221 | comment 23 → 24 | chain | unchanged since the tag |
| `t3toolbox/backend/utv_sampling.py` | 291 | comment 28 → 27 | chain | unchanged since the tag |
| `t3toolbox/backend/utv_sampling.py` | 303 | comment 28 → 27 | chain | unchanged since the tag |
| `t3toolbox/backend/utv_sampling.py` | 316 | comment 28 → 27 | chain | unchanged since the tag |
| `t3toolbox/backend/utv_sampling.py` | 328 | comment 24 → 23 | chain | unchanged since the tag |
| `t3toolbox/backend/utv_sampling.py` | 341 | comment 24 → 23 | chain | unchanged since the tag |
| `t3toolbox/backend/utv_sampling.py` | 357 | comment 24 → 23 | chain | unchanged since the tag |
| `t3toolbox/fitting.py` | 89 | comment 52 → 53 | chain | unchanged since the tag |
| `t3toolbox/fitting.py` | 310 | type 18 → 19 | chain, type slip | unchanged since the tag |
| `t3toolbox/fitting.py` | 406 | comment 20 → 21 | chain | unchanged since the tag |
| `t3toolbox/fitting.py` | 420 | comment 20 → 21 | chain | unchanged since the tag |
| `t3toolbox/fitting.py` | 626 | comment 56 → 60 | hand | unchanged since the tag |
| `t3toolbox/fitting.py` | 938 | comment 89 → 88 | bridged (review) | unchanged since the tag |
| `t3toolbox/frame_variations_format.py` | 1038 | comment 69 → 71 | re-pad | unchanged since the tag |
| `t3toolbox/frame_variations_format.py` | 1040 | comment 69 → 71 | re-pad | unchanged since the tag |
| `t3toolbox/frame_variations_format.py` | 1164 | type 22 → 21 | type slip | unchanged since the tag |
| `t3toolbox/frame_variations_format.py` | 1414 | comment 61 → 62 | chain | unchanged since the tag |
| `t3toolbox/manifold.py` | 521 | comment 75 → 74 | chain | unchanged since the tag |
| `t3toolbox/manifold.py` | 815 | comment 89 → 91 | chain | unchanged since the tag |
| `t3toolbox/manifold.py` | 816 | comment 90 → 91 | chain | unchanged since the tag |
| `t3toolbox/manifold.py` | 1089 | comment 57 → 59 | bridged (review) | unchanged since the tag |
| `t3toolbox/manifold.py` | 1090 | comment 57 → 59 | bridged (review) | unchanged since the tag |
| `t3toolbox/manifold.py` | 1091 | comment 57 → 59 | bridged (review) | unchanged since the tag |
| `t3toolbox/manifold.py` | 1093 | comment 57 → 59 | re-pad | unchanged since the tag |
| `t3toolbox/manifold.py` | 1094 | comment 57 → 59 | re-pad | unchanged since the tag |
| `t3toolbox/manifold.py` | 1332 | comment 49 → 50 | bridged (review) | unchanged since the tag |
| `t3toolbox/manifold.py` | 1566 | comment 49 → 50 | chain | unchanged since the tag |
| `t3toolbox/manifold.py` | 1569 | comment 48 → 50 | bridged (review) | unchanged since the tag |
| `t3toolbox/optimizers.py` | 139 | comment 70 → 71 | chain | unchanged since the tag |
| `t3toolbox/optimizers.py` | 342 | comment 40 → 41 | chain | unchanged since the tag |
| `t3toolbox/optimizers.py` | 374 | comment 40 → 41 | chain | unchanged since the tag |
| `t3toolbox/optimizers.py` | 399 | comment 40 → 41 | chain | unchanged since the tag |
| `t3toolbox/optimizers.py` | 486 | comment 73 → 75 | chain | unchanged since the tag |
| `t3toolbox/safety.py` | 60 | comment 61 → 62 | bridged (review) | unchanged since the tag |
| `t3toolbox/tucker_tensor_train.py` | 768 | comment 57 → 58 | chain | unchanged since the tag |
| `t3toolbox/tucker_tensor_train.py` | 962 | comment 53 → 54 | chain | unchanged since the tag |
| `t3toolbox/tucker_tensor_train.py` | 3645 | comment 41 → 42 | bridged (review) | unchanged since the tag |
| `t3toolbox/tucker_tensor_train.py` | 3720 | comment 40 → 41 | bridged (review) | unchanged since the tag |
| `t3toolbox/tucker_tensor_train.py` | 3781 | comment 40 → 41 | bridged (review) | unchanged since the tag |
| `t3toolbox/tucker_tensor_train.py` | 4234 | comment 73 → 74 | chain | unchanged since the tag |
| `t3toolbox/tucker_tensor_train.py` | 4479 | comment 77 → 76 | bridged (review) | unchanged since the tag |
| `t3toolbox/tucker_tensor_train.py` | 4566 | comment 62 → 63 | bridged (review) | unchanged since the tag |
| `t3toolbox/uniform_frame_variations_format.py` | 413 | comment 48 → 49 | chain | unchanged since the tag |
| `t3toolbox/uniform_manifold.py` | 1100 | comment 58 → 59 | bridged (review) | unchanged since the tag |
| `t3toolbox/uniform_manifold.py` | 1297 | comment 58 → 59 | chain | unchanged since the tag |
| `t3toolbox/uniform_tucker_tensor_train.py` | 977 | comment 71 → 72 | chain | unchanged since the tag |
| `t3toolbox/uniform_tucker_tensor_train.py` | 986 | comment 70 → 71 | bridged (review) | unchanged since the tag |
| `t3toolbox/uniform_tucker_tensor_train.py` | 988 | comment 70 → 71 | bridged (review) | unchanged since the tag |
| `tests/backend/test_curvature.py` | 2521 | comment 94 → 95 | chain | unchanged since the tag |
| `tests/backend/test_curvature.py` | 2805 | comment 80 → 81 | chain | touched by the arc |
| `tests/backend/test_optimizers.py` | 354 | comment 68 → 69 | chain | unchanged since the tag |
| `tests/backend/test_optimizers.py` | 1067 | comment 87 → 88 | chain | unchanged since the tag |
| `tests/backend/test_padded_frame.py` | 325 | comment 73 → 75 | chain | unchanged since the tag |
| `tests/backend/test_padded_frame.py` | 1025 | comment 89 → 90 | re-pad | unchanged since the tag |
| `tests/backend/test_padded_frame.py` | 1026 | comment 89 → 90 | re-pad | unchanged since the tag |
| `tests/backend/test_padded_frame.py` | 1028 | comment 89 → 90 | re-pad | unchanged since the tag |
| `tests/backend/test_padded_frame.py` | 1029 | comment 89 → 90 | re-pad | unchanged since the tag |
| `tests/backend/test_padded_frame.py` | 1030 | comment 89 → 90 | re-pad | touched by the arc |
| `tests/backend/test_padded_frame.py` | 1031 | comment 89 → 90 | re-pad | touched by the arc |
| `tests/backend/test_padded_frame.py` | 1032 | comment 89 → 90 | re-pad | touched by the arc |
| `tests/backend/test_padded_frame.py` | 1033 | comment 89 → 90 | re-pad | touched by the arc |
| `tests/backend/test_padded_frame.py` | 1035 | comment 88 → 90 | hand | unchanged since the tag |
| `tests/backend/test_padded_frame.py` | 1038 | comment 89 → 90 | hand | unchanged since the tag |
| `tests/backend/test_padded_frame.py` | 1039 | comment 89 → 90 | hand | unchanged since the tag |
| `tests/backend/test_padded_frame.py` | 1041 | comment 89 → 90 | bridged (review) | unchanged since the tag |
| `tests/backend/test_padded_frame.py` | 1043 | comment 89 → 90 | bridged (review) | unchanged since the tag |
| `tests/backend/test_padded_frame.py` | 1091 | comment 95 → 98 | bridged (review) | unchanged since the tag |
| `tests/backend/test_padded_frame.py` | 1093 | comment 97 → 98 | chain | unchanged since the tag |
| `tests/backend/test_padded_start.py` | 206 | comment 76 → 77 | bridged (review) | unchanged since the tag |
| `tests/backend/test_ranks.py` | 141 | comment 71 → 72 | bridged (review) | unchanged since the tag |
| `tests/backend/test_uniform_curvature.py` | 151 | comment 70 → 71 | chain | unchanged since the tag |
| `tests/backend/test_uniform_curvature.py` | 1784 | comment 80 → 81 | chain | unchanged since the tag |
| `tests/backend/test_uniform_fitting.py` | 102 | comment 60 → 62 | chain | unchanged since the tag |
| `tests/backend/test_uniform_fitting.py` | 108 | comment 89 → 90 | chain | unchanged since the tag |
| `tests/backend/test_uniform_fitting.py` | 217 | comment 56 → 57 | chain | unchanged since the tag |
| `tests/backend/test_uniform_fitting.py` | 460 | comment 107 → 108 | re-pad | unchanged since the tag |
| `tests/backend/test_uniform_fitting.py` | 461 | comment 107 → 108 | re-pad | unchanged since the tag |
| `tests/backend/test_uniform_fitting.py` | 656 | comment 57 → 58 | chain | unchanged since the tag |
| `tests/backend/test_uniform_padded_frame.py` | 820 | comment 89 → 90 | re-pad | unchanged since the tag |
| `tests/backend/test_uniform_padded_frame.py` | 821 | comment 89 → 90 | re-pad | unchanged since the tag |
| `tests/backend/test_uniform_padded_frame.py` | 823 | comment 89 → 90 | re-pad | touched by the arc |
| `tests/backend/test_uniform_padded_frame.py` | 824 | comment 89 → 90 | re-pad | unchanged since the tag |
| `tests/backend/test_uniform_padded_frame.py` | 825 | comment 89 → 90 | re-pad | touched by the arc |
| `tests/backend/test_uniform_padded_frame.py` | 826 | comment 89 → 90 | re-pad | unchanged since the tag |
| `tests/backend/test_uniform_padded_frame.py` | 827 | comment 88 → 90 | re-pad | unchanged since the tag |
| `tests/test_chunk_size_estimator.py` | 124 | comment 70 → 69 | bridged (review) | unchanged since the tag |
| `tests/test_chunk_size_estimator.py` | 161 | comment 93 → 95 | chain | unchanged since the tag |
| `tests/test_chunk_size_estimator.py` | 229 | comment 89 → 90 | chain | unchanged since the tag |
| `tests/test_chunk_size_estimator.py` | 253 | comment 101 → 102 | chain | unchanged since the tag |
| `tests/test_dispatch.py` | 773 | comment 65 → 67 | chain | unchanged since the tag |
| `tests/test_dispatch.py` | 1730 | comment 61 → 62 | bridged (review) | unchanged since the tag |
| `tests/test_dispatch.py` | 1770 | comment 76 → 77 | chain | unchanged since the tag |
| `tests/test_dispatch.py` | 1771 | comment 75 → 77 | chain | unchanged since the tag |
| `tests/test_dispatch.py` | 1780 | comment 74 → 75 | chain | unchanged since the tag |
| `tests/test_dispatch.py` | 1811 | comment 75 → 76 | bridged (review) | unchanged since the tag |
| `tests/test_dispatch.py` | 1857 | comment 73 → 72 | bridged (review) | unchanged since the tag |
| `tests/test_dispatch.py` | 2044 | comment 80 → 81 | chain | unchanged since the tag |
| `tests/test_dispatch.py` | 2091 | comment 75 → 74 | chain | unchanged since the tag |
| `tests/test_dispatch.py` | 2395 | comment 96 → 97 | chain | touched by the arc |
| `tests/test_fitting.py` | 367 | comment 80 → 81 | chain | unchanged since the tag |
| `tests/test_fitting.py` | 575 | comment 72 → 73 | chain | unchanged since the tag |
| `tests/test_fitting.py` | 675 | comment 87 → 88 | bridged (review) | unchanged since the tag |
| `tests/test_fitting.py` | 856 | comment 79 → 80 | chain | unchanged since the tag |
| `tests/test_frame_variations_format.py` | 307 | comment 81 → 80 | bridged (review) | unchanged since the tag |
| `tests/test_frame_variations_format.py` | 544 | comment 86 → 87 | chain | touched by the arc |
| `tests/test_manifold.py` | 693 | comment 104 → 85 | hand | unchanged since the tag |
| `tests/test_manifold.py` | 694 | comment 74 → 85 | hand | unchanged since the tag |
| `tests/test_manifold.py` | 695 | comment 99 → 85 | hand | unchanged since the tag |
| `tests/test_manifold.py` | 1166 | comment 73 → 74 | bridged (review) | unchanged since the tag |
| `tests/test_manifold.py` | 1188 | comment 85 → 87 | bridged (review) | unchanged since the tag |
| `tests/test_manifold.py` | 1192 | comment 82 → 87 | bridged (review) | unchanged since the tag |
| `tests/test_sharing.py` | 554 | comment 75 → 76 | chain | unchanged since the tag |
| `tests/test_sharing.py` | 556 | comment 75 → 76 | bridged (review) | unchanged since the tag |
| `tests/test_tucker_tensor_train.py` | 2143 | comment 48 → 50 | chain | unchanged since the tag |
| `tests/test_tucker_tensor_train.py` | 2157 | comment 79 → 80 | chain | unchanged since the tag |
| `tests/test_uniform_frame_variations_format.py` | 96 | comment 72 → 73 | bridged (review) | unchanged since the tag |
| `tests/test_uniform_frame_variations_format.py` | 103 | assign 14 → 15 | hand | touched by the arc |
| `tests/test_uniform_frame_variations_format.py` | 239 | comment 99 → 98 | bridged (review) | unchanged since the tag |
| `tests/test_uniform_frame_variations_format.py` | 252 | comment 87 → 86 | bridged (review) | touched by the arc |
| `tests/test_uniform_frame_variations_format.py` | 461 | comment 59 → 60 | bridged (review) | unchanged since the tag |
| `tests/test_uniform_frame_variations_format.py` | 525 | comment 72 → 73 | chain | unchanged since the tag |
| `tests/test_uniform_frame_variations_format.py` | 816 | comment 76 → 77 | bridged (review) | unchanged since the tag |
| `tests/test_uniform_frame_variations_format.py` | 818 | comment 75 → 77 | chain | unchanged since the tag |
| `tests/test_uniform_frame_variations_format.py` | 876 | comment 68 → 69 | chain | unchanged since the tag |
| `tests/test_uniform_frame_variations_format.py` | 898 | comment 71 → 72 | bridged (review) | unchanged since the tag |
| `tests/test_uniform_manifold.py` | 111 | comment 82 → 84 | chain | touched by the arc |
| `tests/test_uniform_manifold.py` | 114 | comment 82 → 84 | hand | touched by the arc |
| `tests/test_uniform_manifold.py` | 115 | comment 82 → 84 | hand | touched by the arc |
| `tests/test_uniform_manifold.py` | 116 | comment 82 → 84 | hand | touched by the arc |
| `tests/test_uniform_manifold.py` | 118 | comment 82 → 84 | hand | touched by the arc |
| `tests/test_uniform_manifold.py` | 614 | comment 76 → 77 | chain | unchanged since the tag |
| `tests/test_uniform_manifold.py` | 1223 | comment 79 → 80 | chain | unchanged since the tag |
| `tests/test_uniform_probing.py` | 377 | comment 97 → 98 | chain | unchanged since the tag |
| `tests/test_uniform_probing.py` | 794 | comment 102 → 101 | chain | unchanged since the tag |
| `tests/test_uniform_probing.py` | 938 | comment 92 → 91 | bridged (review) | unchanged since the tag |
| `tests/test_weighted.py` | 255 | comment 69 → 70 | bridged (review) | unchanged since the tag |
