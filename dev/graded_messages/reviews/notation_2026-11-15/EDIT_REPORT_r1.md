# Edit report, round 1 (phase A: the folder)

Editor's report on bringing `dev/graded_messages/` into line with the notation of 2026-11-15, 23:23. Branch
`graded-messages`. The edits were made on a working tree based on `8222abe`, and committed by the supervisor as
`27fb13a` (parent `e1213c7`; the commits `ab77fab` and `e1213c7` in between touch none of the files edited here).
The Claude doc was not touched (phase B). (Header corrected in round 2.)

**Rulings applied.** The supervisor's rulings Q1-Q10, then the change of 2026-11-16, 00:23 (the maintainer: "There
is no need for extra step functions if they literally do the exact same things with different arguments."). After
that change a step function is named by letter and grade and serves both bases; its parameters are basis-free.
So, in the scripts, the ONLY function rename is `gamma_tilde_dot_step` -> `nu_tilde_dot_step`; `eta_tilde_dot_step`
keeps its name; every step parameter is basis-free. The first pass had renamed two functions with a basis
(`nu_L_tilde_dot_step`, `eta_D_tilde_dot_step`) and some parameters (`z_step`'s `nu_L`, `Jp_step`'s `eta_D_dot`,
`V_tilde_dot_step`'s `eta_D_*`); all were set back to the basis-free form. In `DESIGN_AFTER_REVIEW.md` 4.1 the
definition keeps `mu_dot_step(mu_dot, ...)` and only the call site has the full names; the "22 step functions
with 60 terms" count carries no caveat.

## 1. Every survey finding

| Id | What was done |
| --- | --- |
| R1 | Done: a status bullet "Notation decided on 2026-11-15 at 23:23" |
| R2 | Done: a contents row for `NOTATION_HISTORY.md`; also a row for `reviews/notation_2026-11-15/` (as asked) |
| R3 | No text change: `jet_steps_check.py` is renamed (Q3) and its step names are decided (00:23), so "the decided names" holds |
| R4 | Done as recommended; also "(type catalog, section 14, now `NOTATION_HISTORY.md` section 6)" |
| R5 | Done: "eleven tabs"; "Variable diagrams" now "28 rows, one drawing per variable, output and block", no "prototype" (paragraph rewrapped) |
| R6 | Done as recommended |
| R7 | Done: "`nu` (in the frame's `R` basis) and `nu_L` (in the base point's `L` basis) differ by ..." |
| R8 | Done differently (wording): heading "the letter does not decide the bond space"; the lesson "separate arrays, not stacked on an axis" kept. Section 2 |
| R9 | Done as recommended |
| R10 | Done: "no dot-grade core gradient". Section 2 |
| S1-S3 | Not mine to do (`SAMPLE_VARIABLES.md` out of bounds; S1 and S2 already fixed per Q9) |
| T1 | Done: one clause under "Since the pause" (9.6 at 23:23) |
| T2 | Gloss only (Q10): "`gamma` (now `nu_L`)" |
| T3 | Gloss as recommended |
| T4 | Gloss with the survey's text |
| T5 | Gloss as recommended |
| T6 | Gloss only (Q10): row 3.2 "Is `gamma` (now `nu_L`) a first-class variable, ..." |
| T7 | No change |
| T8 | Gloss: "(Named at 23:23: `eta_D`; `mu_dot`, `xi_dot`, `nu_L_dot`, `eta_D_dot`.)" |
| T9, T10 | Glosses as recommended |
| T11-T14 | Moved to `NOTATION_HISTORY.md` sections 3.2, 4, 5, 6; one pointer bullet with four sub-bullets left in place. Section 3 |
| T15 | Gloss "(The notation of 9.6 uses neither.)" |
| C0 | Done: a "Names updated" paragraph after "Checked" |
| C1 | Renamed. Rows for `eta_D`, `eta_D_tilde` and the raw dots NOT added (substance; see section 6) |
| C2 | Done; the gloss reads "(these two names meant the frame's dots until 2026-11-15, 23:23)" |
| C3 | Done as recommended |
| C4 | Done: pairs rewritten. Section 2 |
| C5 | Done: "Each of the 30 arrays of the program". Section 2 |
| C6 | Done: reworded per the arrow rule. Section 2 |
| C7 | No change (counts unchanged) |
| C8 | Done: table rewritten to the checker's output. Section 2 |
| C9 | Done. Section 2 |
| C10, C11 | Done |
| C12 | Done: `mu + mu_R_dot` (in `TYPE_CATALOG.md` section 9 and in `arrow_type_check.py`) |
| C13 | Done; "`mu_tilde` of `nu_L`, which a call on the corewise frame itself calls `nu`" (that is `SAMPLE_VARIABLES.md` section 9) |
| C14 | Moved whole to `NOTATION_HISTORY.md` section 5; pointer left under the old heading |
| C15, C16 | Moved whole (with the written-out table) to `NOTATION_HISTORY.md` section 6; pointer left |
| C17 | No change |
| D0 | Done: "A fourth, since 2026-11-15, 23:23 ..." in the update note |
| D1 | Done (bullet rewrapped) |
| D2 | Done: split into two rows. Section 2 |
| D3 | No change |
| D4 | Done |
| D5 | Done: heading "3.5 `nu_L` (`gamma` until 2026-11-15, 23:23)"; facts renamed; rationale and the "name is free" bullet replaced by a statement and a pointer. Section 2 |
| D6 | Done per the 00:23 ruling: definition unchanged; call site `mu_R_dot = left_to_right(mu_dot_step, zeros, d, mu, xi, xi_U_dot, R, U, H)`; one sentence added after the code |
| D7 | No change (no caveat, per the 00:23 ruling) |
| D8 | Done, and `eta_D` added to the first row. Section 2 |
| D9 | No change |
| D10 | Done: "`xi`, `mu`, `nu_L` (through `eta_D`)". Section 2 |
| D11, D12 | Done |
| D13 | Moved to `NOTATION_HISTORY.md` section 3.1; replaced by a two-sentence statement with pointer. Section 2 |
| D14 | Done: "`xi`, `mu` and `nu_L` (through `eta_D`)" (bullet rewrapped) |
| D15, D16, D17 | Done |
| L1 | Done: bullet added to "Words changed", with a pointer to `NOTATION_HISTORY.md` |
| L2 | Gloss "`gamma` (now `nu_L`)" in the bold lead of entry 8 (the line was split in two to keep the width) |
| L3 | Covered by L1 |
| L4 | Done (optional): a line under the headings of entries 53 and 54 |
| L5 | No change |
| P1 | Done: a new bullet "The names, since 2026-11-15 (23:23)" in the status note, with the survey's map plus `w_z`, `phi`, `z_t`, `phi_t` and the four blocks |
| P2 | Covered by P1, which points to `NOTATION_HISTORY.md` section 1 |
| P3 | Done in the P1 bullet (Q8: notice only) |
| P4 | Done in the P1 bullet, not under the table (snapshot body left alone). Section 2 |
| P5 | No change |
| W1 | Done: "the decided names (`xi_term`, `mu_dot_step`, `_tilde`, `nu_L`: the notation of SAMPLE_VARIABLES.md ...)" |
| W2 | Done in the status note; the sentence on step 2 says "seen from a call on a manifold frame its `nu` and `eta` are `nu_L` and `eta_D` (`SAMPLE_VARIABLES.md` section 9)" in place of the survey's "where the names have no basis" |
| W3 | Done in the status note (Q8) |
| E1 | Done; also the corewise `_t` -> `_dot`, `_tz` -> `_tilde_dot`, and the figure note (F2, F3, F4) |
| E2 | Done as recommended |
| E3 | Gloss "(`nu_R` is `nu`)" in the body (listed by the survey) |
| E4 | Gloss in the body, as recommended |
| E5 | Covered by E1 |
| E6 | Done in E1 (Q8) |
| A1 | Done differently: a status-note bullet (one notice per snapshot) instead of an inline gloss |
| Scripts (a)-(g) | Done, with the 00:23 changes; see section 4 |
| (h) | Not touched (out of bounds) |
| (i) V1 | Not done: `variable_diagrams_gen.py` is out of bounds. Its docstring still says "prototype", "Seven rows so far" and "TYPE_CATALOG.md, section 14" |
| (j) | Done: the `plan_mockup.py` docstring gloss and pointer |
| F1 | No change |
| F2, F3 | Note in the `EVALUATOR.md` status note and in the README contents rows; files and captions untouched |
| F4 | README row 88: "Names before the review (`_t`, `_z`)" |
| F5, DOC-1..4 | Phase B |

## 2. Changes of substance

Each changes a statement, not only a name.

1. **C4, `TYPE_CATALOG.md` section 2.** Before: "On every edge the two partners are in dual network spaces: `xi`
   and `eta`, `mu` and `gamma`, `w` and `z`." After: "On every edge the two partners, in the same basis, are in
   dual network spaces: `xi` and `eta_D`, `mu` and `nu_L` in the base point's bases; `xi_U_dot` and `eta`,
   `mu_R_dot` and `nu` in the frame's other bases (`SAMPLE_VARIABLES.md` section 8); `w` and `z`."
2. **C5, section 7.** "Each of the 30 variables" -> "Each of the 30 arrays of the program".
3. **C6, section 8.** Before: "The catalog's first rule is better said as: the letter fixes the direction of the
   arrow." After: "... the letter fixes the direction in which a variable travels along its edge. Whether that is
   with the arrow of the edge or against it is fixed by the basis, once the cores carry arrows (section 9)."
4. **C8, section 9 table.** Before: With = `w`, `xi`, `mu`, `nu`, `eta` and their tildes; Against = `gamma`, `z`,
   the dot variables `xi_dot`, `mu_dot`, `nu_dot`, `eta_dot`, `Jp`, `gamma_tilde`. After (exactly the checker's
   output): With adds the raw form's `xi_dot`, `mu_dot`, `xi_tilde_dot`, `mu_tilde_dot`; Against = `nu_L`, `z`;
   the frame's form's `xi_U_dot`, `mu_R_dot`, `nu_L_dot`, `eta_D_dot`, and `Jp`; `nu_L_tilde`; the raw form's
   `nu_L_dot`, `eta_D_dot`, `nu_L_tilde_dot`, `eta_D_tilde_dot`.
5. **C9, section 9.** Before: "So what flows with the arrows has not yet met a hole. What flows against them has
   passed a hole (the dot grade) or is on its way out (`gamma`, `z`)." After: "So what flows against the arrows is
   exactly what carries a basis in its name, and the outputs `z` and `Jp`; everything else flows with them, the
   raw form's dots included. (Until 2026-11-16 this paragraph said ...; the raw form's dots ... are the exception
   to that.)" I kept the old statement in a parenthesis rather than moving it, since it is one sentence.
6. **D2, `DESIGN_AFTER_REVIEW.md` 3.2.** Before: one row, proposed `xi_dot`, `mu_dot`, `nu_dot`, `eta_dot`, today
   `dxi` / `xi_dot`, `sigma` / `mu_dot`, `tau` / `nu_dot`, `deta` / `eta_dot`. After: two rows, the frame's form
   `xi_U_dot`, `mu_R_dot`, `nu_L_dot`, `eta_D_dot` (today `dxi`, `sigma`, `tau`, `deta`), and the raw form `xi_dot`,
   `mu_dot`, `nu_L_dot`, `eta_D_dot` (today `xi_dot`, `mu_dot`, `nu_dot`, `eta_dot`, the two-hole kernel).
7. **D5, 3.5.** Before: "`nu` means the frame's right edge variable, swept through `R`, and nothing else. The right
   edge variable swept through `L` ... is a different variable: `gamma`. The maintainer's view is that it is an
   auxiliary variable, not a `nu`." After: "`nu` is the right of a bond in the frame's `R` basis ... The right of a
   bond in the `L` basis ... is `nu_L`: the same side as `nu`, written in the base point's basis." Plus a pointer.
   The "name is free" bullet was removed (moved).
8. **D8 and D10, 4.3.** `eta_D` added to "The point and the samples"; `z` reads "`xi`, `mu`, `nu_L` (through
   `eta_D`)" (was `xi`, `mu`, `gamma`). Same "(through `eta_D`)" in D14 (4.4).
9. **D13, 4.3.** The list "Variables with no public name" replaced by: "Since 2026-11-15 every variable is public
   (target decision 6.4, confirmed at 21:57) and named (`SAMPLE_VARIABLES.md`); which application of the transpose
   a tilde name refers to is target decision 3.3." Plus pointer.
10. **P4.** In the `PROPOSAL.md` status note: `sigmas`/`mus_dot` and `dxis`/`xis_dot` are two variables each (one
    side in two bases); `taus`/`nus_dot` and `detas`/`etas_dot` are one each.
11. **R8.** Before: "In a manifold frame the grade decides the bond space. `mu` and `mu_z` live on `rL`; `mu_t`
    lives on `rR`. Components must be keyed by grade, not stacked on an axis." After: "In a manifold frame the
    letter does not decide the bond space. `mu` and `mu_tilde` live on `rL`; the frame's dot `mu_R_dot` lives on
    `rR`. The basis is part of the name ..., and the grades are separate arrays, not stacked on an axis."
12. **R10.** "no grade-`t` core gradient" -> "no dot-grade core gradient".
13. **C12 / arrow checker.** The reject row is now `mu + mu_R_dot` (was `mu + mu_dot`, which under the new meaning
    of `mu_dot` is well typed). Same row and number in the recorded output.

## 3. What moved to `NOTATION_HISTORY.md`

All passages are taken programmatically from `git show 8222abe`, and checked to appear verbatim in the new file.
The only alteration: headings inside a moved passage are one `#` lower (stated in the file's banner).

| From | Lines at `8222abe` | To | Pointer left |
| --- | --- | --- | --- |
| `DESIGN_AFTER_REVIEW.md` 3.5 `gamma` (whole section, as it stood) | 141-152 | section 2 | parenthesis in the new 3.5 |
| `DESIGN_AFTER_REVIEW.md` 4.3 "Variables with no public name" | 265-270 | section 3.1 | two sentences in 4.3 |
| `TARGET_DECISIONS.md` notes to 6, the note of 21:57, head to "Also without a name" | 829-853 | section 3.2 | bullet with four sub-bullets |
| the same note, "`zeta`, decided" | 854-857 | section 4 | same bullet |
| the same note, "Rows 5 and 6, two options" | 858-864 | section 5 | same bullet |
| the same note, "The naming problem widened" | 865-874 | section 6 | same bullet |
| `TYPE_CATALOG.md` section 13 | 379-428 | section 5 | heading kept, two lines |
| `TYPE_CATALOG.md` section 14, with the table of 22:41 | 430-548 | section 6 | heading kept, three lines |

New in the file (not moved): the banner, a timeline (section 1) with a three-column table decided / 2026-11-14 /
phase 1, short superseded-by notes before each moved passage, and section 7 (script identifiers renamed, and the
00:23 step-function decision quoted).

Pointers that name a section of `NOTATION_HISTORY.md`: README (section 6), DESIGN 3.5 (2), DESIGN 4.3 (3.1),
TARGET (3.2, 4, 5, 6), TYPE_CATALOG 13 (5) and 14 (6), PROPOSAL (1). All exist.

## 4. Scripts

**Renamed identifiers.**

- `jet_steps_check.py`: function `gamma_tilde_dot_step` -> `nu_tilde_dot_step`; step parameters `gamma_tilde_dot`,
  `gamma_dot`, `gamma_tilde`, `gamma` -> `nu_tilde_dot`, `nu_dot`, `nu_tilde`, `nu` in `nu_tilde_dot_step`,
  `eta_tilde_dot_step`, `H_tilde_dot_step`; `z_step(mu, L, gamma, D)` -> `z_step(mu, L, nu, D)`. In `main()`:
  `gamma` -> `nu_L`, `gamma_tilde` -> `nu_L_tilde`, `gamma_tilde_dot` -> `nu_L_tilde_dot`, `eta_tilde_dot` ->
  `eta_D_tilde_dot`, `gamma_dot` -> `raw_nu_L_dot`, `c_eta_dot` -> `raw_eta_D_dot`, `c_mu_dot` -> `mu_dot`,
  `c_xi_dot` -> `xi_dot`, and the frame's `xi_dot`, `mu_dot`, `nu_dot`, `eta_dot` -> `xi_U_dot`, `mu_R_dot`,
  `nu_L_dot`, `eta_D_dot`. Comment above the dot steps per the 00:23 ruling; comment on `raw_`; docstring.
- `type_catalog_check.py`: the same dict keys and `CATALOG` names; `J.gamma_tilde_dot_step` ->
  `J.nu_tilde_dot_step`; docstring "eta, nu (in either basis) and z are COVECTORS"; comment on `raw_`.
- `arrow_type_check.py`: string key `'gamma_tilde_dot_step'` -> `'nu_tilde_dot_step'`; `v['gamma']` ->
  `v['nu_L']`; `v['mu_dot']` -> `v['mu_R_dot']`; the two reject labels; comment `gamma` -> `nu_L`; docstring.
- `subtree_gauge_check.py`: `gamma` -> `nu_L`, `zeta` -> `eta_D`, `f_xi_dot`/`f_mu_dot`/`f_nu_dot`/`f_eta_dot` ->
  `xi_U_dot`/`mu_R_dot`/`nu_L_dot`/`eta_D_dot`, `gamma_dot`/`zeta_dot` -> `raw_nu_L_dot`/`raw_eta_D_dot`; labels;
  docstring A and B rewritten in the new names.
- `tilde_multiplier_check.py`: frame's dots renamed in locals, bump strings, dict keys, the loop table, docstring.
- `order0_assessment.py`: `gamma` (and its lambda parameter `g`) -> `nu_L`; frame's dots renamed.
- `kind_readout_check.py`: locals `xi_dot`, `mu_dot` -> `xi_U_dot`, `mu_R_dot`; import aliases kept.
- `plan_mockup.py`: docstring only.

**Runs.** `./run_all.sh`: output identical to `scripts/expected_output_3dc15d9.txt` (not changed).
`./review/run_review_checks.sh` (jax 0.12.0.1, numpy 2.5.3): before any edit it reproduced the record exactly;
after the edits the diff is label lines only, below; `scripts/review/expected_review_output_3dc15d9.txt` was
replaced by the new output (old copy at `/home/claude/scratch/notation/old_expected_review_output.txt`). A
mechanical comparison of every numeric token of the 316 lines found no difference. Line numbers are 4 higher than
the survey's (the file grew at `8222abe`). `python3 -I scripts/review/sample_variables_check.py
/home/claude/t3toolbox`: "all checks passed". The count line "22 step functions, 60 terms" is unchanged.

```
166-169  type_catalog_check: the four lists of names
   gamma -> nu_L; xi_dot, mu_dot, nu_dot, eta_dot -> xi_U_dot, mu_R_dot, nu_L_dot, eta_D_dot;
   c_xi_dot, c_mu_dot, gamma_dot, c_eta_dot -> xi_dot, mu_dot, raw_nu_L_dot, raw_eta_D_dot;
   gamma_tilde -> nu_L_tilde; gamma_tilde_dot, eta_tilde_dot -> nu_L_tilde_dot, eta_D_tilde_dot
184-185  "frame nu_dot   and gamma_dot"  -> "nu_L_dot,  frame and raw form"
         "frame eta_dot  and zeta_dot"   -> "eta_D_dot, frame and raw form"
187      "frame mu_dot (R)  and mu_dot (L)" -> "mu_R_dot (R)  and mu_dot (L)"
189-191  "xi_dot (D)  and frame xi_dot (U)" -> "xi_dot (D)    and xi_U_dot (U)";
         "nu (R) and gamma (L)" -> "nu (R) and nu_L (L)"; "eta (U) and zeta (D)" -> "eta (U) and eta_D (D)"
212-213  arrow_type_check: with / against lists, renamed as in 166-169
224      "gamma used where nu is wanted: eta_term(mu, U, gamma)" -> "nu_L used ...: eta_term(mu, U, nu_L)"
226      "a sum of mu and mu_dot: mu + mu_dot" -> "a sum of mu and mu_R_dot: mu + mu_R_dot"
238-241  tilde_multiplier_check: eta_dot, xi_dot, nu_dot, mu_dot -> eta_D_dot, xi_U_dot, nu_L_dot, mu_R_dot
```

Every number on these lines (`1e-15`, `equal`, `REJECTED: ... AL2 and AR2`, `(AL1-; order in) and (AR1+; order
in)`, `1e-11`, `3e+01`, ...) is the same.

## 5. Remaining grep hits

`gamma|zeta` (outside `SURVEY.md`, `DIAGRAM_TEST.md`):

- `NOTATION_HISTORY.md` (74): moved passages and the history tables.
- `SAMPLE_VARIABLES.md` (11): its old-name column and "Four things to watch".
- `REVIEW_DECISIONS.md` (20): the L1 bullet (line 12); dated text at 311-323 (entry 8, with the L2 gloss), 384, 468,
  779, 785, 1150-1177 (entries 53-55, with the L4 lines) and 1214-1215 (entry 57).
- `TARGET_DECISIONS.md` (7): glosses (lines 54, 297, 322, 592), the note "3.2, what the notation changes" and the
  9.6 note (existing), the pointer sub-bullet "22:11: `zeta` ... became `eta_D`".
- `DESIGN_AFTER_REVIEW.md` (3): the update note, the 3.5 heading gloss, the pointer parenthesis.
- `EVALUATOR.md` (1): status note "(named `gamma` from 2026-11-14 to 2026-11-15)".
- `scripts/plan_mockup.py` (1): gloss "`gamma` (now `nu_L`)".
- `scripts/graded_messages_prototype.py:127`, `reviews/review_1_scratch/proto_tails.py:127`: `zeta` as a symbol for
  a reversed residual jet, unrelated (records).

`\b(mu|xi|nu|eta)_dot\b`: in live files every remaining hit is the raw form (`TYPE_CATALOG.md` 63, 230;
`DESIGN_AFTER_REVIEW.md` 91 row; `subtree_gauge_check.py` raw block; `type_catalog_check.py`, `jet_steps_check.py`
raw blocks), a basis-free step parameter (`jet_steps_check.py` steps, `DESIGN_AFTER_REVIEW.md` 205-206), the
library's names (`DESIGN_AFTER_REVIEW.md` 73, 91 "today" column), a gloss or its subject (`TARGET_DECISIONS.md`
476, 812, 816-817, 827; the 9.6 note 1048-1049), or a status-note map (`PROPOSAL.md`, `README.md` 134). Records
(`REVIEW_DECISIONS.md`, reviews, snapshot bodies, `NOTATION_HISTORY.md`, `SAMPLE_VARIABLES.md`) keep theirs.

`sigma`, `tau`: the library's names (`README.md` table of code locations; `DESIGN_AFTER_REVIEW.md` 79, 88, 90,
186, 313) and the 21:59 note with its gloss (`TARGET_DECISIONS.md` 1036). `frame's`: every remaining
"the frame's `<x>_dot`" is in the 9.6 note (existing) or a gloss. "without a name" / "no name": `TARGET_DECISIONS.md`
810 (dated 21:44 note, glossed), 833, 836 (my pointer), 1050 (9.6 note); `DESIGN_AFTER_REVIEW.md` 230 (the
lifted-direction argument: still unnamed, consistent with `SAMPLE_VARIABLES.md` section 11).

## 6. What the survey missed, or I found

- `TYPE_CATALOG.md` section 9: "Each block is the outer product of variables that flow with the arrows into a
  hole" is not true for the `Kp` blocks, which read `nu_L` (it flows against). Pre-existing; not touched.
- `DESIGN_AFTER_REVIEW.md` 4.1: "The corewise frame calls the same functions with `G, G, G`." Left as is (rule 3 of
  the 00:23 message: nothing on the corewise frame).
- C1: the type table of `TYPE_CATALOG.md` section 2 still lacks rows for `eta_D`, `eta_D_tilde` and the raw
  `xi_dot`, `mu_dot`; the paragraph below the table names the raw ones. Not added (substance).
- The README "How to resume" step 2 reading order does not mention `SAMPLE_VARIABLES.md`. Not changed.
- `SAMPLE_VARIABLES.md` section 10 says the old name of `eta_D_dot` includes "the prototype's `c_eta_dot`" and of
  `nu_L_dot` `gamma_dot`: right. Section 11's third bullet ("Two step functions for one variable ... what the two
  functions are called is open") is overtaken by the 00:23 ruling; the supervisor said he will update it.
- `variable_diagrams_gen.py` docstring (V1) is stale; out of bounds.

## 7. Judgment calls and doubts

- Moved passages: headings demoted one level; the `TARGET_DECISIONS.md` note of 21:57 split across sections 3.2-6
  in its own order (concatenated, it is the original). I put the whole of `DESIGN_AFTER_REVIEW.md` 3.5 into the
  history (not only the retired sentences), so the record is complete; its facts also stay, renamed, in DESIGN.
- `raw_` comment is in each script that uses it (`jet_steps_check.py`, `type_catalog_check.py`,
  `subtree_gauge_check.py`). In `subtree_gauge_check.py` the two arrays are equal (lifted direction) but are still
  two arrays in one program, so `raw_` is used there too.
- Glosses in snapshot bodies only where the survey lists them (E3, E4); A1, P4, W2/W3, figures handled in status
  notes. The C9 old sentence kept in a parenthesis in place.
- Paragraphs I changed were rewrapped at 115 (only those); one dated line in `REVIEW_DECISIONS.md` entry 8 was
  split for width after the gloss.
- `jet_steps_check.py` docstring still says "the per-site functions are the ones of the order-0 example with the
  names of 2026-11-14"; true apart from `nu_tilde_dot_step`. Could be reworded if the reviewer prefers.
- Unsure: whether the T4 gloss and the C9 rule ("what carries a basis in its name, and the outputs") should say
  `z` and `Jp` explicitly in T4 (it says "the outputs").
