# The glossary update and the docs pass (2026.5.0), 2026-11-15

Bundle item (C) of 2026.5.0, run on HEAD `27a9216c` (store rename, timing and switch all in). Nothing is committed.

## Where the list lives

The work list is the § "Work list by file" of [`glossary_naming_decisions.md`](glossary_naming_decisions.md)
(committed with the glossary in `3fc91855`, re-expressed by the rename's R5). It has **599 rows** (the "601" in the
handoff is the commit message's figure; the table held 599 then too). The file stays as committed; each row's status
is in the sibling [`glossary_naming_decisions_status.md`](glossary_naming_decisions_status.md), keyed by the row's
line in the list file.

## Tally

| status | rows |
|---|---|
| done in this pass | 505 |
| done earlier (the store rename; the list already said so) | 25 |
| deferred | 69 |
| not an error | 0 |

Done by item: E6 185, N1 69, M/U 43, N7 30, N3 23, N5 22, N2 16, E22 13, N8 12, N9 11, E4 11, E19 8, N6 6, E8 6, E17 6,
E1 5, E2 4, N4 4, E18 4, E16 3, N11 3, and one or two each of N10, N15, E12, N14, E20, E9, E3, E13, E21, E5, N13,
E11, E14, C1, E15, E7, N17.

**Deferred, 69:**

- **62 math-note rows** (`exact_hessian_two_hole.tex` 37, `padded_frame_cone.tex` 12, `pad_safe_svd.tex` 8,
  `symmetric_probe_derivatives.tex` 4, `shared_t3_math.tex` 1). The list's own rule: math-note rows wait for the
  note's next rebuild. Four of these notes have a committed PDF, so editing only the `.tex` would leave the two out of
  step. Several rows also sit beside fenced equation lines (K11/K12).
- **7 rows inside the store rename's hand-rewritten blocks**, which K12 checks verbatim against the inventory. All
  seven are real errors. By the session's ruling they stay for the maintainer, who makes the hand edit and re-records
  the inventory's new text. Each one, with the exact hand fix:

| row | file:line (block) | the error | severity | the hand fix |
|---|---|---|---|---|
| L201 | `docs/frame_variations.md:119` (DOC-FV-0080) | the up core's `nᵢ × (rᵢ·rᵢ₊₁)` reshape is called an unfolding | wording | "`nᵢ × (rᵢ·rᵢ₊₁)` unfolding, rows orthonormal" → "`nᵢ × (rᵢ·rᵢ₊₁)` matricization, rows orthonormal" |
| L873 | `t3toolbox/frame_variations_format.py:74` (FVF-0071) | left cores' middle width written `ni` (the T3's Tucker rank), not the frame's `nDi`; wrong at a slack frame | mathematical | "`elm_shape=(rLi, ni, rL(i+1))`" → "`elm_shape=(rLi, nDi, rL(i+1))`" |
| L874 | `t3toolbox/frame_variations_format.py:75` (FVF-0071) | the same for the right cores | mathematical | "`elm_shape=(rRi, ni, rR(i+1))`" → "`elm_shape=(rRi, nDi, rR(i+1))`" |
| L875 | `t3toolbox/frame_variations_format.py:79` (FVF-0077) | the TT variation's right bond off by one (`fv_variation_shapes` reads `right[i].shape[-1]` = `rR(i+1)`) | mathematical | "`elm_shape=(rLi, nDi, rRi)`" → "`elm_shape=(rLi, nDi, rR(i+1))`" |
| L877 | `t3toolbox/frame_variations_format.py:94` (FVF-0088) | the last Tucker core labelled `Dd` in a 0-based diagram whose other labels read `nD(d-1)`, `N(d-1)` | typo | "`Dd`" → "`D(d-1)`" at the same column 48; it is the line's last label, so nothing else moves |
| L879 | `t3toolbox/frame_variations_format.py:106` (FVF-0100) | the same label, and it sits one column right of its edge (col 49, the `\|` at 48) | typo | "`Dd`" → "`D(d-1)`" starting at column 48: the line becomes 17 spaces, `D0`, 7 spaces, `(V1)`, 18 spaces, `D(d-1)` |
| L884 | `t3toolbox/frame_variations_format.py:356` (FVF-0354) | the retired alias "outer" beside the up store | wording | "`- up/outer U_i (all i):     ``einsum(...)```" → "`- up U_i (all i):           ``einsum(...)```" (11 spaces, so `einsum` stays at column 40 with its block) |

## The glossary (`docs/glossary.md`)

- **New entries** (Optimization section, alphabetical, each with its link): **countable step**, **Gauss-Newton view**
  (contributor-facing; links `contributor/curvature_internals.md#model-level-rules`), **model errors** (`e_GN`,
  `e_N`, hit/miss), **phase timing** (`timing=True`, `PhaseSeconds`, `NewtonTiming`, `stats['timing']`; aliases phase,
  window, one-offs, compile wait, compile column; links §4.13), **rounding gate**, **switch** (`hessian='switch'`,
  `switch_tau`, `switch_k`, `switch_cg_forcing_power`, `switch_track`), **`switched_at`**, **tracking product**. The
  switch entries link §4.14's anchor `#414-starting-with-gauss-newton-and-switching-to-the-exact-hessian`. A new
  symbols row for **`τ`**.
- **Stale entries fixed** (the W4 review's list and P3's note): **stats** (adds `hessian`, `switch_track`,
  `switched_at`, `timing`), **history** (`hessian` per row, the five `switch_*` fields, `NewtonInfo.timing` not a
  history field), **step control** (trust region is the switch default too), **forcing term**
  (`switch_cg_forcing_power`), **verbose display** (the `sw` segment, the `time` line and closing table),
  **`newton_cg`** (the three Hessian modes), **local model** (one Hessian, so `'switch'` is `newton_cg`-only),
  **exact Hessian** (pointer to the switch), **same-frame** (N3: `FRAMES` in the contract tables, formerly `SF`).
  Symbols: `K` loses "the derivative order" (N2 fixed every such site), gains `K_N`. `m` gains `m_GN`, `m_N`. `ε`
  gains the rounding gate. The "reserved" `tracking` phase was already fixed in §4.13 and the CHANGELOG by W4.
- **Kept:** every "also:" alias the renames retire (sample stack, A-block / P-block, seeded start, padded restart,
  Riemannian Jacobian, …) and the 2026.4.0 store-word alias groups.
- **Links:** the E9/E8 headings in `frame_variations.md` changed, so the glossary's five links to them were updated
  (`#a-tangent-is-a-sum-of-2d-terms`, `#d-cores-d1-bonds--and-the-carriers-ld-1-r0`).

## Judgement calls

1. **K9-safe wording.** Five fixes are worded so the rename gate keeps passing:
   - L630, `_seed_rows`: "matricization or unfolding", without the list's "(up)" / "(left, right)".
   - L739/L740 (backend) and L946 (frontend `up_svd_tt_core`): "TT-core matricization", kept on one line. The
     list's longer wording split the line, which breaks K9's line alignment and loses a keep row's "down".
   - L941, `get_minimal_ranks`: "Up TT core matricizations are full rank". This keeps the Left/Right/Up parallel and
     the capital "Up" that K9 aligns with the tag's "Down".
   - L359 (E5), curvature facts table: one row now credits both test files rather than being split in two. A split
     row shifts the table block, and K9-docs then loses the "down / left masks" row's alignment.
2. **C1 implemented** (it is a CODE row with Nick's label ruling):
   - `optimizer_display._REFERENCE_NAMES` gains `'cone': 'cone (aligned zero tensor)'`, and the display test's
     reference loop gains the cone case.
   - §4.13's pasted display, which shows a cone run, has its converged line updated to what the display now prints.
     That line is deterministic text, not a measured number.
   - A `### Fixed` section in `CHANGELOG.md` `[2026.5.0]` (not the intro) records C1 and the reworded messages.
     **Drop it if the release commit should carry this instead.**
3. **Just beyond a row's text:**
   - L912, frontend `newton_cg`: the same paragraph's earlier "arbitrary on the padding" changed with the listed
     "aligning the padding". It is the same referent, and the backend row L621 changes both.
   - L229: "the restart's new directions" became "its", to fit the listed phrase.
   - L248: "the deflated warm-start gradient of the padded restart" became "the deflated gradient of the zero-padded
     warm start".
   - L438: its line has two "A-block"s, and both changed.
4. **Facts table** (`curvature_internals.md`, L358): the fact-ID column and every plan or test code are gone. Codes
   that stood alone in a cell now read "tested". Cross-references by ID are reworded ("the probe-derivative restart
   above", "the canonical frame (below)"). The rows map one for one to the old ones.
5. **Runtime messages** (E6 ×8, plus the "sample stack" and "A-block" wording in three more): the text changed, no
   test matches it, and the `%` arguments are unchanged.
6. **Diagrams and equations touched, by hand (Edit, re-derived, alignment checked), none of them rename-inventoried
   blocks:**
   - the user guide's probe diagram (L309: `u0..u2`, `N0..N2`; probe `m` leaves mode `m` free);
   - `TuckerTensorTrain`'s docstring diagram (L938/939: `n(d-1)`, `N(d-1)` in its last column, so nothing moves);
   - `sampling_derivatives`' module-comment jet equation (L674: `t = 0..order`);
   - `sharing.md`'s group-ceiling equation (L247: `r_i·r_{i+1}`).

## Gates (`dev/store_rename_check.py --stage R6 --hits dev/store_rename_slices --prev HEAD`)

**Every gate is green (exit 0)**, end to end at stage R6:
- K0–K9, K9-converse, K9-registered, K9-docs (all three), K10/K10b, K11, K12 (301 of 301), K13, K14, K15-cross and
  K17 (all three shape checks) PASS.
- K15 reports 0 NAME hunks (no identifier changed); K16 reports its 21 LaTeX width forms, signed off at R5.
- `store_rename_diagram_check.py all` passes its self-test, K10/K10b, K11 (94 fenced spans byte-identical) and K12.

**How they got green.** Before this pass, three gates already failed on HEAD `27a9216c`. Nothing in the pass caused
them; by the session's ruling the pass cleared them as bookkeeping:

- **K9 ×7**: idioms the timing and switch slices added — "shows up", "up to rounding", "from here down", "one level
  up", "up to j" — at `backend/optimizers.py:1256, 1273, 1287, 1311`, `optimizers.py:732, 834` and
  `tests/backend/test_optimizers.py:1932`.
- **K9-docs ×1**: §4.13's "a heavy display shows up a row late" (`fitting_and_optimization.md:863`).
- **K12 ×4**: PF-0041, PF-0043, UPF-0047, UPF-0050. P1 (`cbcdc1d6`: `import time`, among other lines) shifted the
  line-located field-comment blocks in `padded_frames.py` and `uniform_padded_frames.py` by one line. The text is
  unchanged; the recorded spans are stale.

The feature slices never ran the rename gates. What the pass did:

- **K12:** re-recorded the four blocks' `new span` lines in `dev/store_rename_slices/diagrams_inventory.md`:
  PF-0041 42 → 43, PF-0043 44 → 45, UPF-0047 47–48 → 48–49 and UPF-0050 50 → 51. No block content changed (K11 and
  K12 confirm the bytes).
- **Reworded "shows up", the three places it reads poorly:**
  - `backend/optimizers.py:1256` "display shows up on the next row" → "display appears on the next row";
  - `optimizers.py:731–732` "a heavy display shows up a row late" → "a heavy display appears a row late";
  - `docs/fitting_and_optimization.md:863`, the same phrase, the same rewording.
- **Registered in `dev/store_rename_slices/k9_registered.tsv`** (new clause `idiom`, documented by a comment line;
  reason `idiom`):
  - `backend/optimizers.py:1273` and `:1287` "up to rounding";
  - `:1311` "from here down";
  - `optimizers.py:834` "one level up";
  - `tests/backend/test_optimizers.py:1932` "up to j".
- **L885**, `frame_variations_format.py:366`: now applied, "(left/right/outer/Tucker)" → "(left/right/up/down)", and
  registered (clause `c`, reason `rewrite`), so that row is done.

## Verification

| check | result |
|---|---|
| docs pages' doctests, CI's set (`docs/*.md docs/contributor/*.md` but `doctest_style.md`), current env (py3.11, numpy 2.4, jax 0.10.2) | pass (`fitting_and_optimization.md` 108 examples) |
| the same, floor env (`tttt`: py3.9, numpy 1.22, jax 0.4.30) | pass |
| module doctests `pytest --doctest-modules t3toolbox`, current / floor | 241 passed / 241 passed; `getting_started.rst` passes in both |
| `tests/backend/test_optimizer_display.py` (C1) and the suite | the display tests pass (the cone case included); full suite, current env, `-n 3`: 1,407 passed, 5 skipped, 49,058 subtests |
| `sphinx -W --keep-going -b html docs` | exit 0, build succeeded, no warning; the glossary's new links resolve (§4.13, §4.14, the renamed `frame_variations.md` headings) |
| `git diff --check` | clean |
| after the gate bookkeeping: `fitting_and_optimization.md` doctests and the module doctests of `optimizers.py`, `backend/optimizers.py`, `frame_variations_format.py`, both envs | pass (21 module examples each env) |
| after the gate bookkeeping: `sphinx -W` | exit 0, build succeeded, no warning |

## For the maintainer's review stop

- **Read the new glossary entries:** countable step, Gauss-Newton view, model errors, phase timing, rounding gate,
  switch, `switched_at`, tracking product, and the `τ` symbols row. Also the updated stats, history, step control,
  forcing term, verbose display, `newton_cg`, local model, exact Hessian and same-frame entries.
- **Rows whose fix changes a claim, not just wording:**
  - E9: a tangent is a sum of `2d` terms.
  - E8: the carriers are `L(d-1)`, `R0`; their bonds are `rLd`, `rR0`.
  - E17/E19, index fixes:
    - `tucker_ranks = (n0, n1, …)`;
    - `fv_variation_shapes`' `rR_(i+1)`;
    - 0-based `N0…N(d-1)` in `T3Frame`, `t3svd_dense` and the user guide;
    - the minimal-rank matricization/unfolding shapes in the user guide and `get_minimal_ranks`;
    - the per-mode and group Tucker ceilings `r_i·r_{i+1}` in `sharing.md`.
  - E4: masks go gappy under `+` and a weights' `kronecker`; there is no uniform T3×T3 product.
  - E3: `use_jit` without jax warns, it does not raise.
  - E11: `xwhile` is not from `get_backend`.
  - E12: local models.
  - E14: value-hashed.
  - E15: the `Geometry` protocol.
  - E18: the T3-SVD truncation is quasi-optimal.
  - E21: stack-leading is not frame-inner.
  - E22: the edge variables are named for their side; `compute_mu` / `compute_tau_tilde` run left to right.
  - E1: `allclose`, `jax_or_warn`.
  - E2: the `'cone'` reference in `NewtonInfo` / `newton_cg`.
  - E5: the import-graph test.
  - E7: a dead link dropped.
  - E20: a link to `CLAUDE.md` now points at the user guide.
  - C1: the display output.
  - The 43 M/U rows (matricization at a Tucker–TT edge, unfolding at a TT–TT edge).
- **Diagrams and equations touched** (judgement call 6); and the 7 deferred rows inside rename blocks, which are real
  errors awaiting a hand edit plus an inventory re-record.
- **The gate bookkeeping** (above): six new `k9_registered.tsv` rows (clause `idiom` is new) and four re-recorded
  K12 spans.

## Review round 1 (`glossary_docs_pass_review_2026-11-15.md`): the fixes

- **R1-M1, rounding gate:** the entry now says a tracked step counts only if `max(e_GN, e_N)` is above the floor. Both
  errors at the floor resets the count, and `e_N` at the floor with `e_GN` above it is a hit. The old text ("both
  must be measurable") was wrong.
- **R1-M2, Gauss-Newton view:** dropped "a contributor's word …", which was false: §4.14 and the frontend docstring
  use the term. The §4.14 link is added beside the contributor one.
- **R1-M3, `τ` symbols row:** the continuation's `tau` is a threshold (grow an edge whose `κi` is a factor `tau` below
  the worst), not a growth factor.
- **N1 to N3:**
  - N1, countable step: "Only these can add to the count".
  - N2, `switched_at`: "The index of the first history row whose model is exact".
  - N3: `entries_apply_probe.md`'s table pipes are back in one column (36).
- **N5:** the three examples' `oracle_relerr` docstring now reads "the quasi-optimal rank-(tucker,tt) dense T3-SVD
  truncation of A", without the duplicated parenthesis.
- **N6:** four of the deferred notes have PDFs, not three.
- **Not done:** N4 (optional rewraps) and R1-M4 (a list gap for the maintainer).
- **Reruns:** `git diff --check` clean; the doctests of the glossary and `entries_apply_probe.md` pass; the gates exit
  0, every gate green; `sphinx -W` (a link was added): exit 0, build succeeded, no warning.
