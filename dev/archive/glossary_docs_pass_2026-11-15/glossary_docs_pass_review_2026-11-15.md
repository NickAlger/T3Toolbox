# Review: the glossary update and the docs pass (2026.5.0), 2026-11-15

Reviewer's record for bundle item (C): the implementer's working tree on HEAD `27a9216c`, uncommitted, described in
[`glossary_docs_pass_2026-11-15.md`](glossary_docs_pass_2026-11-15.md) (the record), with the row statuses in
[`glossary_naming_decisions_status.md`](glossary_naming_decisions_status.md) and the list in
[`glossary_naming_decisions.md`](glossary_naming_decisions.md) § "Work list by file". Every run below is mine, one heavy
job at a time, pytest at `-n 3`.

## Round 1

### Findings

**MAJOR**: none.

**MINOR**

- **R1-M1, glossary "rounding gate": the first clause is wrong.** "Both model errors must be measurable" contradicts
  the code and §4.14. `_switch_sample` (`backend/optimizers.py:1754`) needs only the *larger* error above the floor
  (`max(e_GN, e_N) > floor`, both finite), and §4.14 says outright that "`e_N` at the floor with `e_GN` well above it is a
  hit". The entry's own second clause (`max(e_GN, e_N)` above the floor) is right. Suggested: "A tracked step counts only
  if the larger model error, `max(e_GN, e_N)`, is above a hundred times the trust region's `objective_floor` threshold;
  both at the floor resets the count, and `e_N` at the floor with `e_GN` above it is a hit. In float32 few steps pass,
  so enable jax x64."
- **R1-M2, glossary "Gauss-Newton view": "A contributor's word; user text says 'the row's model'" is false.** The user
  page uses the term twice (§4.14: "the callback's `info.lm` is the Gauss-Newton view of the point's exact build", and
  "CG, the step control and the row use its Gauss-Newton view", `fitting_and_optimization.md:1045, 1063`), and so does
  the frontend `newton_cg` docstring (`optimizers.py:690`). The sentence came from the W4 review's list
  (`W4_review.md:153`), which was wrong about it. Suggested: drop the sentence, or "§4.14 and the `newton_cg`
  docstrings use it too; the `hessian` row field names the row's model". Linking §4.14 beside the contributor anchor
  would serve a user reader.
- **R1-M3, glossary symbols row `τ`: "the continuation's growth factor `tau`" misnames it.** `tau` is the conditioning
  threshold: edge `i` grows iff `κi < κmax / tau` (`compute_continuation_ranks`, `ranks.py:370`; `rank_continuation.md`
  calls it the "sensitivity of the 'well-conditioned' test"). The ranks grow by `n_chunk`, not by a factor. The
  glossary's own `continuation_ranks` entry has it right. Suggested: "the continuation's threshold `tau` (grow an edge
  whose `κi` is a factor `tau` below the worst; rank continuation)".
- **R1-M4, for the maintainer, not the implementer: the list missed sites of two of its own errors.** The pass rightly
  stayed inside its rows. These are list gaps I found while checking E17 and N15:
  - E17's TT-variation off-by-one, `(rLi, nDi, rRi)` for `(rLi, nDi, rR(i+1))`, at nine more sites:
    `frame_variations_format.py:241` (a doctest comment whose own printed output reads `rR(i+1)`), `:786` (inside the
    K12 block FVF-0784), `:814`; `backend/apply.py:145, 236`; `backend/sampling_derivatives.py:900, 933, 1077`;
    `backend/fv_operations.py:637`.
  - N15: `backend/uniform_fitting.py:503` "a 2x2x2 central Tucker core" (present at `daa1e23d`, not on the list).
  
  A closed, enumerated follow-up (eight plain edits and one hand edit with a K12 re-record) could go with the seven
  in-block fixes below, or wait for 2026.6.0.

**NIT**

- **R1-N1, glossary "countable step":** "Only these move the switch's count; a rejected trial, … resets it" contradicts
  itself (a reset moves the count). "Only these can add to the count; …" says it.
- **R1-N2, glossary "`switched_at`":** "The first history row whose model is exact" is the row; the value is its index.
  "The index of the first history row whose model is exact: …".
- **R1-N3, `entries_apply_probe.md:22–25`:** the realigned table's first-column pipes now sit at columns 36, 36, 36, 37
  (all four were at 35 before). The one-hot row gained a space before its pipe. Drop it, or widen the other three by
  one.
- **R1-N4, line lengths:** about twenty changed lines now run past 120 columns where the old line did not (for example
  `user_guide.rst:8` 126, `frame_variations.md:113` 123, `curvature_internals.md:347` 129, `deferred_and_rejected.md:203`
  127, `uniform_equivalence_contract.md:82` 122, `backend/linalg.py:224` 123, `backend/uniform_padded_frames.py:11–13,
  103, 118, 183, 267`). Some are K9-pinned (line alignment), but the rendered-prose ones can be rewrapped where K9
  allows. Optional.
- **R1-N5, the examples' `oracle_relerr` docstrings (three files):** "the quasi-optimal rank-(tucker,tt) T3-SVD
  truncation of A (dense T3-SVD truncation)" now says "T3-SVD truncation" twice. This is the row's text verbatim, so it
  is optional: "(dense)" would do.
- **R1-N6, the record:** "Three of these notes have a committed PDF". Four do (`exact_hessian_two_hole`,
  `padded_frame_cone`, `pad_safe_svd`, `shared_t3_math`); only `symmetric_probe_derivatives` has none. This is in the
  record only.

### (1) The glossary

Each new entry checked against the code (`backend/optimizers.py`: `_check_switch_options`, `_switch_sample`,
`_gauss_newton_view`, the switch block of `newton_cg`, `_newton_scalar_record`, `PhaseSeconds`) and against §4.13–§4.14:

| entry | verdict |
|---|---|
| countable step | right (accepted, not the `−g` fallback, TR accepted or `ls_steps == 0 and not ls_failed`); wording R1-N1 |
| Gauss-Newton view | the mechanism is right (`dc.replace(lm, curvature=None)`, one exact build per point); R1-M2 |
| model errors | right (`e_GN`, `e_N`, row fields, hit on a countable step) |
| phase timing | right: the phases, the compile wait, the one-offs, "changes no number", the serialized jax run (§4.13 "What it costs") |
| rounding gate | the floor is right (`_SWITCH_ROUNDING · _TR_RESOLUTION_FACTOR · resolution`); the first clause is wrong, R1-M1 |
| switch | right (defaults 0.25 / 2, for good, per call, trust region by default, the four keywords) |
| `switched_at` | right (`it + 1`, so `len(history)` when the last row fired; `0` with `switch_k=0`; `None`); R1-N2 |
| tracking product | right (`switch_pKp` outside `pHp`, `'countable'` default, the `tracking` phase) |

The fixed stale entries are now true: stats (the three switch keys in switch mode only, `timing` only with
`timing=True`), history (`hessian` on every row; the five `switch_*` fields in switch mode only; `NewtonInfo.timing`
dropped by `_newton_scalar_record`), step control, forcing term, verbose display, `newton_cg`, local model, exact
Hessian, same-frame (`FRAMES`). Symbols: `K` (the derivative-order meaning retired; I found no remaining `K`-as-order
site in docs, code or examples), `m`, `ε` are right; `τ` has R1-M3. The five re-pointed anchors
(`#a-tangent-is-a-sum-of-2d-terms`, `#d-cores-d1-bonds--and-the-carriers-ld-1-r0`) match the renamed headings. Every
retired alias is still in an "also:" list (sample stack, A-block / P-block, seeded start, padded / continuation restart,
test / measurement vectors, `Ld`, frame-internal padding, central Tucker core, `SF`). The maintainer's earlier rulings are
untouched and intact: matricization (Tucker–TT) vs unfolding (TT–TT); the edge condition number over the first `r`
singular values, structural; the equivalence contract with its exceptions (the alignment); orthogonal *and* oblique
gauge projection; "padding" with "junk slot" as the padded-frame code's alias. The new entries are alphabetical and in
the page's style.

### (2) The pass

I read every row of the work list against its diff for 35 of the 88 touched files, about 380 rows (stratified):
- user pages: `entries_apply_probe`, `frame_variations`, `numerical_contracts`, `rank_continuation`, `release_notes`,
  `user_guide.rst`, `sharing`, `t3m_methods`, `t3svd_minimal_ranks`, `uniform_masks_vs_ranks`,
  `uniform_equivalence_contract`, `fitting_and_optimization`;
- contributor pages: `curvature_internals` (incl. the facts table), `deferred_and_rejected`, `padded_frame_internals`,
  `verification_records`, `numerical_contract_catalog`, `testing_strategy`;
- docstrings and messages: `tucker_tensor_train`, `frame_variations_format`, `sampling_derivatives`, `probing`,
  `shared_geometry`, `manifold`, `stacking`, `linalg`, `common`, both `optimizers`, both `fitting`, `ranks`, `ut3_svd`,
  `geometry`, `__init__`;
- all nine examples.

I scanned the rest of the diff for anything wider than a row. Each fix is the row's ask and no wider, apart from the
record's listed judgement calls. Checked against the code where the row makes a claim:
- E22: `compute_mu` is a forward `xscan`; `compute_nu` is `compute_mu` on the reversed train; `τ` gathers the
  variation on its right through `P_{i+1}τ`.
- E11: `xwhile` (`common.py:341`).
- M/U: `dense_t3svd`'s first SVD is the mode-0 matricization.
- `t3_down_svd_tt_core` / `up_svd` are mode-leg matricizations.
- E17: `fv_variation_shapes` reads `right[i].shape[-1]`.
- E8: the carriers.

The facts table (L358) keeps every row one for one, and its reworded cross-references point the right way ("above",
"below"). The two blockquote reflows (`rank_continuation.md`, `uniform_svd_prefix_orthogonalization.md`) keep their
`>` markers. No residual "sample stack", "A-/P-block", "padded/continuation restart", "test/measurement vectors",
`numerically_equal`, `_jax_or_warn`, `tWCi_Cio_to_tWCo` or "Gauss-Newton models" remains outside the glossary's
alias lists. (`release_notes.md:76` "seeded starts" means random seeds and is right as it is.)

**The six K9-driven rewordings are sound.**
- L630 "matricization or unfolding" is true without the store words.
- L739/L740 and L946 "TT-core matricization" are true: `up_svd` on the mode leg.
- L941 "Up TT core matricizations" parallels Left/Right and is true. It differs from the user guide's
  "up-matricizations", which is acceptable.
- L359 credits both files in one row.

**The deferrals.** The 62 `.tex` rows follow the list's own rule (math notes change at their next rebuild; four of
the five notes have a committed PDF). **The seven in-block rows are each a real error, and each proposed hand fix is
right:**

| row | verdict |
|---|---|
| L201 `frame_variations.md:119` | an error: the `nᵢ × (rᵢ·rᵢ₊₁)` reshape puts the mode leg against both bonds, which is a Tucker–TT cut, so a matricization. The fix is right (a markdown table, so nothing to align). |
| L873/L874 `frame_variations_format.py:74–75` | an error: the left/right middle leg is the frame's `nDi` (it must match `D_i`'s rows). `ni` is undefined in this docstring and differs from `nDi` at a slack frame, e.g. a Tucker rank above `Ni`. The inventory shows R4a kept `ni` deliberately ("the rewriter decides"), so this is a correction of that choice, not a slip. The fix `nDi` is right. |
| L875 `:79` | an error, and pre-rename too (`(rLi, nUi, rRi)` at 2026.4.0). `variation_shapes` zips `right_ranks[1:]`, and the doctest at `:241` prints `(1, 10, 4)` = `rR1`. The fix `rR(i+1)` is right (see R1-M4 for its nine siblings). |
| L877 `:94` | an error: `Dd` in a 0-based diagram whose column reads `nD(d-1)`, `N(d-1)`. `D(d-1)` at column 48 is right; it is the last label on the line. |
| L879 `:106` | an error plus a misalignment: `Dd` starts at column 49 under the `\|` at 48 (I measured both lines). The record's layout (17 spaces, `D0`, 7, `(V1)`, 18, `D(d-1)`) puts it at 48, matching the first diagram's `D(d-1)` / `R(d-1)` column. Right. |
| L884 `:356` | a retired alias. The replacement keeps the backquoted `einsum` at column 40, and I checked the column arithmetic. Optional, the maintainer's call: the line above reads `down_tucker D_i`, which he may want as `down D_i` in the same hand edit for parallelism (not a list row). |

**Fix before the tag: yes, I recommend it.** Three are mathematical errors in `T3Frame`'s class docstring, the
library's main shape reference, and 2026.5.0 is the release that rewrote exactly these blocks. The cost is
six hand edits plus re-recording six inventory blocks' "new text" (DOC-FV-0080, FVF-0071, FVF-0077, FVF-0088, FVF-0100,
FVF-0354) and re-running K12.

**C1.** These four agree:
- the label `'cone': 'cone (aligned zero tensor)'`;
- the display test's new subcase;
- the CHANGELOG `### Fixed` wording ("…, g₀ = cone (aligned zero tensor))");
- §4.13's pasted line.

My battery's ragged cone run prints exactly §4.13's line, character for character. `### Fixed` between `### Changed` and
`### Added` follows the file's 2026.3.0 precedent. The "eight messages" count matches E6's eight message sites.

**The four hand-edited diagrams and equations, re-derived whole.**
- **User guide probe diagram:** probe `m` leaves mode `m` free. The vectors under the other two `B`s are `u` of those
  modes, each in the old one's column. The lengths read `N0`..`N2`.
- **`TuckerTensorTrain` diagram:** `n(d-1)`, `N(d-1)` are the column's last labels, so nothing moves.
- **`sampling_derivatives`:** `t = 0, 1, ..., order.`
- **`sharing.md`:** `n_g ≤ min(N_g, Σ_{i∈g} min(N_g, r_i·r_{i+1}))`, consistent with `get_minimal_ranks`' docstring.

### (3) The gates (my runs)

- `store_rename_check.py --stage R6 --hits dev/store_rename_slices --prev HEAD`: **exit 0**, every gate PASS.
  - K9, K9-converse, K9-registered and the three K9-docs gates;
  - K10/K10b, K11, **K12 301 of 301**, K13, K14, K15-cross, K17 and its three shape checks;
  - **K15 0 NAME hunks**; K16 reports its 21 forms.
- `store_rename_diagram_check.py all`: **exit 0**. The self-test, K10/K10b, K11 (94 fenced spans byte-identical) and
  K12 301/301 all pass.
- `git diff` on `diagrams_inventory.md` shows only the four `new span` lines (PF-0041 43, PF-0043 45, UPF-0047 48–49,
  UPF-0050 51), and K12 confirms the bytes.
- The `idiom` clause entries are each a real idiom and none is a store word: "up to rounding" ×2, "from here down" (a
  field-list position), "one level up" (the next rank level), "up to j".
- The one clause-`c` entry (`frame_variations_format.py:366`) is the L885 rewrite, store words in the new meaning.

### (4) Non-breaking (my runs)

- **AST comparison**, 27a9216c vs the tree (docstrings and comments stripped): the only code changes are message
  strings and the `_REFERENCE_NAMES` dict.
  - The message strings are in `common`, `linalg`, `sampling_derivatives`, `stacking`, `uniform_padded_frames`,
    `manifold` and `shared_geometry`.
  - The tests change only `test_optimizer_display`'s loop.
- **Bitwise battery**, `git archive 27a9216c` vs the tree, numpy, ten seeded runs:
  - The runs: ragged GN verbose; exact + `padded_from` + `'cone'`, verbose and timed; switch verbose; switch `'all'`
    on the line search; uniform GN verbose; uniform exact; uniform `'cone'` verbose; corewise entries;
    `gradient_descent`.
  - **Every returned array is bitwise identical.** History rows, `stats` and the verbose `diagnostics` are equal
    (`wall_time` aside).
  - **The display text is identical except the converged line of the two `'cone'` runs** (ragged and uniform), which
    gains ", g₀ = cone (aligned zero tensor)".
- **Display tests:** `tests/backend/test_optimizer_display.py`, 29 passed, 36 subtests.
- **Suite**, current env, `-n 3`: **1,407 passed, 5 skipped, 49,058 subtests**. The one warning is the designed
  `adam`-on-MANIFOLD notice.
- **Module doctests:** 241 passed (py3.11) / 241 passed (`tttt`, py3.9). `getting_started.rst` passes in both.
- **CI's docs page set** (`docs/*.md docs/contributor/*.md` but `doctest_style.md`): passes in both envs
  (`fitting_and_optimization.md` 108 examples).
- `sphinx -W --keep-going -b html docs`: **exit 0, build succeeded, no warning**. In the rendered glossary, the six
  §4.14 links, the §4.13 link, `contributor/curvature_internals.html#model-level-rules` and the five re-pointed
  `frame_variations` anchors all resolve to existing ids.
- `git diff --check` clean. `py_compile` passes on every changed `.py`.

### Round 1 verdict

**REVISE.** The implementer should fix R1-M1, R1-M2 and R1-M3, which are glossary text only. R1-N1 to R1-N3 are cheap
and recommended; R1-N4 to R1-N6 are optional. R1-M4 goes to the maintainer. No other change is asked, and nothing in
the code, the gates or the pass needs another round beyond rerunning the glossary page's checks.

## Round 2

The implementer's fixes landed in `docs/glossary.md` and `docs/entries_apply_probe.md`, and I read each one:
- R1-M1, the rounding gate: the larger error, both at the floor resets the count, `e_N` at the floor with `e_GN` above
  it is a hit. Right.
- R1-M2: the false sentence is gone, and a §4.14 link now sits beside the contributor link.
- R1-M3: `tau` is now "the continuation's threshold". Right.
- R1-N1, "can add to the count", and R1-N2, "The index of": done.
- R1-N3: the table's pipes are at column 36 on all four lines.

The implementer also took two of the optional NITs:
- R1-N5: the three examples now read "the quasi-optimal rank-(tucker,tt) dense T3-SVD truncation of A".
- R1-N6: the record now says four notes have a PDF.

R1-N4 is left, and R1-M4 stays with the maintainer. The implementer's own `sphinx -W` after the new link built with
no warning.

My reruns:
- `store_rename_check.py --stage R6`: exit 0 (K9 and K9-docs PASS, K12 301/301, K15 0 hunks).
- `store_rename_diagram_check.py all`: PASS.
- The glossary and `entries_apply_probe` page doctests: pass.
- `git diff --check`: clean.

The new link is the same §4.14 anchor that six glossary links already resolve in my `sphinx -W` build, and nothing
else changed, so I did not rebuild.

VERDICT: APPROVE
