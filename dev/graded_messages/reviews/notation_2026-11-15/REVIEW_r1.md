# Review of editor round 1 (commit `27fb13a`, parent `e1213c7`)

Reviewer's report. The only file the reviewer wrote is this one. Nothing else in the repository or the Claude doc
was changed.

## Verdict: NOT SATISFIED

The round is close. Every survey finding is handled or the departure is explained. Every rename of a dot
variable is in the right basis. The moved passages are word for word. No quotation was altered. Both runners
reproduce their records, and in the re-recorded file only labels changed. Three new or retained statements are
wrong, though, and must be fixed (B1 to B3). The rest is non-blocking, and the `(D_dot, L_dot)` list below must
be applied in round 2.

Counts: 3 blocking, 16 non-blocking, 4 observations on out-of-bounds files.

---

## Blocking findings

### B1. `TYPE_CATALOG.md` section 9: the sentence under the flow table is wrong about the raw form's dots

The exact words: "So what flows against the arrows is exactly what carries a basis in its name, and the outputs
`z` and `Jp`; everything else flows with them, the raw form's dots included. (Until 2026-11-16 this paragraph
said that what flows with the arrows has not yet met a hole, and what flows against them has passed a hole or is
on its way out; the raw form's dots, which flow with the arrows, are the exception to that.)"

What is wrong: the raw form has four dots, `xi_dot`, `mu_dot`, `nu_L_dot`, `eta_D_dot`, plus their tilde-dots.
The table directly above, and the checker's output (`expected_review_output_3dc15d9.txt`, lines 212-213), put
the raw `nu_L_dot`, `eta_D_dot`, `nu_L_tilde_dot` and `eta_D_tilde_dot` AGAINST the arrows. So "the raw form's
dots, which flow with the arrows" is false for half of them. Those four are also not an exception to the old
rule, which they obey. The first sentence, "everything else ..., the raw form's dots included", reads the same way.

Correction: "... everything else flows with them, the raw form's `xi_dot` and `mu_dot` and their tilde-dots
included. (Until 2026-11-16 this paragraph said that what flows with the arrows has not yet met a hole, and what
flows against them has passed a hole or is on its way out; the raw form's `xi_dot`, `mu_dot`, `xi_tilde_dot` and
`mu_tilde_dot` have passed a hole and flow with the arrows, so they are the exception to that.)"

### B2. `TYPE_CATALOG.md` section 9: "Each block is the outer product of variables that flow with the arrows into a hole"

The editor found this and left it (its report, section 6). It is pre-existing, but it sits in the paragraph the
editor rewrote, and the new table directly above now contradicts it. `H_tilde_dot` reads `nu_L`,
`nu_L_tilde`, `nu_L_tilde_dot` and the raw `nu_L_dot`. `V_tilde_dot` reads `eta_D_tilde_dot` and the raw
`eta_D_dot` (`SAMPLE_VARIABLES.md` section 6). The table lists all of these against the arrows. A newcomer
sees the contradiction at once.

Correction: "Each block of `g`, `V_tilde` and `H_tilde`, is the outer product of variables that flow with the
arrows into a hole. The `Kp` blocks also read variables that flow against them (`nu_L` and its grades, and
`eta_D_dot`, `eta_D_tilde_dot`). Every block's legs are the legs of the hole's direction with every sign flipped:
`V_tilde` fits `V`, `H_tilde` fits `H`, and the `Kp` blocks fit the raw core directions." (The supervisor may
downgrade this to non-blocking because it is pre-existing. I count it as blocking because it is false and the
rewrite put it next to the evidence against it.)

### B3. `DESIGN_AFTER_REVIEW.md` section 4.3: the pointer says the wrong decision superseded the list

The exact words: "The list of "variables with no public name" that stood here (2026-11-14) was superseded on
2026-11-15 at 23:23 by target decision 9.6; it is in `NOTATION_HISTORY.md` section 3.1."

What is wrong: the list had three items. Only items 2 and 3 (the `eta`-like variable through `L`, and the dots of
`Kp`) were answered by 9.6. Item 1 ("The tilde variables seeded by `Jp` ... stay internal to `JtJp`") is not a
naming matter. Target decision 6.4 overtook it at 21:57, and it is still open under 3.3, which is "To confirm".
`NOTATION_HISTORY.md` section 3.1 has this right ("every variable can be wanted and held (6.4 ...), and every
variable has a name (9.6 ...)"). The pointer in DESIGN does not. The same document also says the opposite
further down: the table of findings has the row "D13 | Decision 8: seeds (only residual-seeded tilde variables
are public) and gauge (`nu_L`)".

Correction: "... was superseded on 2026-11-15: every variable of a kind's graph can be wanted and held (target
decision 6.4, confirmed at 21:57), and every variable is named (9.6, 23:23); how the `Jp`-seeded tilde copies
are reached is target decision 3.3, to confirm. The list is in `NOTATION_HISTORY.md` section 3.1." Also put a
gloss on the D13 row of the findings table: "(item 1 superseded by 6.4; see 4.3)". The update note at the top
("the parts of sections 3.5 and 4.3 whose subject was a retired name") should say "a retired name or an
overtaken decision".

---

## Non-blocking findings

N1. **`TYPE_CATALOG.md` section 8, first bullet (change of substance C6).** "Whether that is with the arrow of
the edge or against it is fixed by the basis, once the cores carry arrows (section 9)." This holds for the
internal edges. It does not hold for the mode edge, which has one basis: there the letter decides, `w` with the
arrow, `z` and `Jp` against. Add: "(on the mode edge, which has one basis, by the letter: `w` with, `z` and `Jp`
against)". The statement is otherwise true and agrees with `SAMPLE_VARIABLES.md` section 7.

N2. **`scripts/review/jet_steps_check.py`, docstring:** "The per-site functions are the ones of the order-0
example with the names of 2026-11-14". This is no longer true: `gamma_tilde_dot_step` is now `nu_tilde_dot_step`,
and the `gamma*` parameters are `nu*`. Reword: "... the ones of the order-0 example, named by letter and grade
(one, `nu_tilde_dot_step`, renamed on 2026-11-16) ...".

N3. **"basis-free parameters": the core parameters are store names.** The new comment in `jet_steps_check.py`
says "So the parameters of every step function are basis-free (letter and grade)". `NOTATION_HISTORY.md`
section 7 says the same. Yet `nu_step(nu, R, xi)` is called with `L`, and `mu_dot_step(..., R, U, H)` with
`(L, L, G_dot)`. Say "the variable parameters"; the core parameters keep the stores of the frame's form as
placeholders. Taste: in `order0_assessment.py` the inline step lambdas now have basis names as parameters
(`lambda mu_R_dot, mu, xi, xi_U_dot, R, U, H`). They are only ever called in the frame's form, so this is
harmless, but it differs from the ruling for step functions.

N4. **`DESIGN_AFTER_REVIEW.md`, update note:** "`gamma` is `nu_L`, and the dot variables carry a basis in the
frame's form". The most important change for a reader is missing: the bare `mu_dot` and `xi_dot` now mean the
raw form's. Add it.

N5. **`DESIGN_AFTER_REVIEW.md` 3.2, the column "Is the multiplier of the constraint defining".** It pairs
`xi_tilde` with `eta`, `eta_tilde` with `xi`, `mu_tilde` with `nu` and `nu_tilde` with `mu`. That is true on the
corewise frame, where it was checked. On a manifold frame `SAMPLE_VARIABLES.md` section 8 pairs `xi_tilde` with
`eta_D_dot`, `eta_tilde` with `xi_U_dot`, `mu_tilde` with `nu_L_dot` and `nu_tilde` with `mu_R_dot`. A reader
who knows only the page sees `xi_tilde` (`D` basis) paired with `eta` (`U` basis). This is the error that C4
fixed in `TYPE_CATALOG.md`. Add "(on the corewise frame; on a manifold frame the partners are the dots of
`SAMPLE_VARIABLES.md` section 8)" to the header or under the table.

N6. **`TYPE_CATALOG.md` section 2, the type table (the editor's item C1).** It has no rows for `eta_D`,
`eta_D_tilde`, or the raw `xi_dot` and `mu_dot`. The README calls the file "the types of all the sampling
variables". Do not add rows the check does not cover. Add one sentence under the table instead: "The table
lists the arrays of the checked program; `eta_D` and `eta_D_tilde` are not among them (the program forms `eta_D`
inside `z_step`). The type of every variable is in `SAMPLE_VARIABLES.md` section 7."

N7. **`NOTATION_HISTORY.md` section 5 has no key from its old names to the new ones.** The passage from
`TYPE_CATALOG.md` section 13 uses `gamma`, `zeta`, "the frame's `mu_dot`", "the frame's `xi_dot`", "the frame's
`nu_dot`", "the frame's `eta_dot`", and a bare `mu_dot` meaning the raw dot ("`mu + mu_dot` is well typed").
Section 6 has a key and section 5 does not. Give section 5 the same sentence, or "Read with the table of section
1". Without it, the line "the checker of section 9 rejects `mu + mu_dot` for it" is a trap for the reader, which
is exactly what C12 avoided.

N8. **`TARGET_DECISIONS.md`, the pointer bullet, sub-bullet "22:11: options A and B ..."** It does not say what
became of them, though the other sub-bullets do. Add "; neither was taken: the dots are `xi_dot`/`mu_dot` (raw
form) and `xi_U_dot`/`mu_R_dot` (frame's form)".

N9. **`TARGET_DECISIONS.md`, the T4 gloss (the editor's doubt).** Yes: write "and the outputs `z` and `Jp`" to
match `TYPE_CATALOG.md`. The T8 gloss ("Named at 23:23: `eta_D`; `mu_dot`, `xi_dot`, `nu_L_dot`,
`eta_D_dot`") could also name `eta_D_tilde_dot`, which was the unnamed row 4 of the 21:57 list. Optional.

N10. **`README.md`, "How to resume", step 2.** The reading order does not include `SAMPLE_VARIABLES.md`, which
is now the reference for every name. Put it first, or right after `TARGET_DECISIONS.md`, with
`NOTATION_HISTORY.md` "when an older text is read".

N11. **`README.md`, the paragraph on the Claude doc:** "Variable diagrams" (28 rows, one drawing per variable,
output and block ...). This describes the tab, not the generator. The survey found 7 rows in the tab, and I did
not open the doc. If the tab has not been republished with 28 rows, say "28 rows in the generator" or leave the
count for phase B. Also, "the older tabs with the same status notes as the files here" stops being true until
phase B mirrors the new status-note bullets.

N12. **`REVIEW_DECISIONS.md` entry 8:** the line "cores) is not `nu`. The maintainer's view: ... it gets" is
now 121 characters. The editor split the line "to keep the width", so the split did not do its job. Trivial.

N13. **`DESIGN_AFTER_REVIEW.md` 4.1, "The corewise frame calls the same functions with `G, G, G`." (the editor's
item 2).** Leave the corewise statement as it is (the corewise question is open, `SAMPLE_VARIABLES.md` section
11). After it, add the case that is decided: "At the base point of a manifold frame the raw form calls them with
`(L, L, L_dot)` and `D_dot`." Do this together with the `(D_dot, L_dot)` change below.

N14. **`NOTATION_HISTORY.md` section 1, the timeline.** It stops at 23:23 and "2026-11-16". Add 00:23 (step
functions by letter and grade, serving both bases; also in section 7) and 00:30 (the raw core direction at the
base point is `(D_dot, L_dot)`; `(B_dot, G_dot)` remains on the corewise frame). Add a row for the raw direction
to the three-period table: decided `(D_dot, L_dot)` | `(B_dot, G_dot)` in the prototype, `core_directions` in the
library | phase 1: the `_t` cores at the corewise frame.

N15. **`reviews/notation_2026-11-15/EDIT_REPORT_r1.md`, header:** "Base: ... head `8222abe`. Nothing committed."
The work is commit `27fb13a`, whose parent is `e1213c7`. (`ab77fab` and `e1213c7` came in between. I checked that
neither touches a file the editor changed, so nothing was lost.) Correct the header so that the record is right.

N16. **`EVALUATOR.md` status note:** "a dot in the frame's form carries the other basis". This is right only if
"other" means other than the letter's usual basis: `nu_L_dot` and `eta_D_dot` carry the base point's bases.
Clearer: "a dot in the frame's form carries a basis in its name: `xi_t`, `mu_t`, `nu_t`, `eta_t` here are
`xi_U_dot`, `mu_R_dot`, `nu_L_dot`, `eta_D_dot`". The `PROPOSAL.md` notice has the same wording.

---

## Round 2: every place where `B_dot`, `G_dot`, `(B_dot, G_dot)` (or `B`, `G` for the base point) stand for the raw direction at the base point of a manifold frame

The maintainer decided at 00:30 that this is `(D_dot, L_dot)`; on the corewise frame it stays `(B_dot, G_dot)`.
The `SAMPLE_VARIABLES.md` at `f183e46` already says so (sections 1, 5, 5.2, 5.4, 6, 11). The generic parameters
`B`, `G`, `B_dot`, `G_dot` of the term and step functions are NOT listed. Only main-level variables, program
parameters, labels and text are listed.

**Documents**

| File | Where | Words now | Change |
| --- | --- | --- | --- |
| `TYPE_CATALOG.md` | section 2, row "blocks / `V_tilde_dot`, `H_tilde_dot` (`Kp`)" | "covector on the space of `B`, of `G`" | "of `D`, of `L`" (the space of `D_dot`, `L_dot`) |
| `DESIGN_AFTER_REVIEW.md` | 4.2, first paragraph | "the argument for the lifted direction has no name yet" | now `(D_dot, L_dot)` on the page; update, or say what is still open (the engine's keyword?) as the supervisor rules |
| `DESIGN_AFTER_REVIEW.md` | section 10, list of undecided items (line ~592) | "The name of the lifted-direction argument of `sampling_variables`." | same as above |
| `DESIGN_AFTER_REVIEW.md` | 4.1, after the code (N13) | (nothing yet) | add `(L, L, L_dot)` and `D_dot` for the raw form |
| `NOTATION_HISTORY.md` | section 3.2, editor's note | "The argument that carries the raw core direction is still without a name (`SAMPLE_VARIABLES.md` section 11)." | "... was named `(D_dot, L_dot)` on 2026-11-16 at 00:30 (`SAMPLE_VARIABLES.md` sections 1 and 11)" |
| `NOTATION_HISTORY.md` | section 7, last paragraph | "given `(L, L, G_dot)` and the raw `xi_dot` it returns `mu_dot`" | `(L, L, L_dot)` |
| `NOTATION_HISTORY.md` | section 1 (N14) | | timeline entry and table row for the raw direction |
| `NOTATION_HISTORY.md` | section 7, table of renamed identifiers | | add the round-2 renames (`B_dot`/`G_dot` -> `D_dot`/`L_dot` in the scripts, the labels `fits B_dot`/`fits G_dot`) |

Not to change: `NOTATION_HISTORY.md` line 157 (moved passage, verbatim: "`(B_dot, G_dot)` in the prototype");
`DESIGN_AFTER_REVIEW.md` 4.4 "`(B, G)` for a trial point" (corewise); `WORKED_EXAMPLE.md` `B, G` (the point
before orthogonalization, corewise); `TYPE_CATALOG.md` section 1 "On the corewise frame `(B, G, G, G)`".

**Scripts**

- `scripts/review/jet_steps_check.py`
  - line 93 (the comment above the dot steps): "given (L, L, G_dot) and the raw xi_dot" -> `(L, L, L_dot)`.
  - line 131 (section comment of the tilde-dot steps): "at the base point (B, G) with a raw core direction (B_dot,
    G_dot)". Optional: the parameters stay generic, but the comment could say "(D, L) and (D_dot, L_dot) on a
    manifold frame; (B, G) and (B_dot, G_dot) on the corewise frame".
  - `main()`: line 208 `B_dot, G_dot = ...  # a raw core direction at (D, L)`; line 250 comment "raw core
    direction (B_dot, G_dot)"; line 252 `B, G = D, L` (the alias; then `G` stands for `L` in lines 253, 258-268:
    `G[-1].shape`, `G[0].shape`, `G, G, G_dot`, `G, G_dot`); lines 254 and 256 comments; lines 257 and 262
    `B_dot`; lines 258-268 `G_dot`; line 279 `kind.corewise_hessian((B_dot, G_dot), ...)`.
- `scripts/review/type_catalog_check.py`
  - `program(frame, V, H, B_dot, G_dot, ...)` (line 32): a whole program, not a step function -> `D_dot, L_dot`.
  - lines 58-60 comment ("at the base point (B, G) = (D, L), with the raw core direction (B_dot, G_dot)"; "(B_dot,
    G_dot) is not the lift"); line 61 `G = L` (alias) and its uses in lines 62-78; `B_dot`/`G_dot` in lines 63-78.
  - `main()`: lines 150, 169, 171, 176-177 (`B_dot2`, `G_dot2`), 194-196, 208-209 (`B_dot3`, `G_dot3`), 225, 227.
- `scripts/review/arrow_type_check.py`
  - `typed_program(frame, V, H, B_dot, G_dot, ...)` (line 113) and the call at line 152.
  - `raw['B_dot']`, `raw['G_dot']` (lines 168-169); the typed keys and comments at lines 177-178 ("a direction of
    D: the type of D"); lines 188, 205.
  - line 217, the printed hole names `('V_tilde_dot', 'B_dot'), ('H_tilde_dot', 'G_dot')`. **These are labels in
    the recorded output**: the lines "fits B_dot True" and "fits G_dot True" of
    `scripts/review/expected_review_output_3dc15d9.txt` (in section 2 of `arrow_type_check.py`) will change.
    Re-record again and recheck that the numbers are the same.
- `scripts/review/subtree_gauge_check.py`
  - line 57 `B_dot, G_dot = (list(c) for c in tvo.tv_corewise_lift(...))`; lines 84-88 (`B_dot`, `G_dot`).
  - The docstring speaks of "the LIFTED raw core direction" with no names; nothing to change there.
- No occurrence in `tilde_multiplier_check.py`, `order0_assessment.py`, `kind_readout_check.py` or
  `plan_mockup.py`.

---

## The three things the editor left untouched (its report, section 6)

1. **The "each block ... flow with the arrows" sentence in `TYPE_CATALOG.md` section 9.** A real problem. See B2.
2. **"The corewise frame calls the same functions with `G, G, G`" in `DESIGN_AFTER_REVIEW.md` 4.1.** Imprecise
   (the dot steps take `G, G, G_dot`) but not wrong, and the corewise question is open. Leave the sentence and
   add the raw form at the base point (N13) in round 2.
3. **The rows missing from the type table of `TYPE_CATALOG.md` section 2.** A real gap, but not an error. Add
   one sentence pointing to the page, not unchecked rows (N6).

The editor also listed the README reading order (N10: fix it) and the `variable_diagrams_gen.py` docstring. The
supervisor has since updated that docstring at `f183e46`, and it now says 28 rows and the page's notation.

---

## What I checked and how

**Context read in full:** `SAMPLE_VARIABLES.md` (as at `27fb13a` and then as at `f183e46`, the supervisor's
commit made during this review), `SURVEY.md`, `EDIT_REPORT_r1.md`, the whole diff
(`git diff e1213c7 27fb13a -- dev/graded_messages`, 2293 lines), `NOTATION_HISTORY.md`, `TYPE_CATALOG.md`, and
the changed regions of `DESIGN_AFTER_REVIEW.md`, `README.md` (status, contents, how to resume, established
facts), `TARGET_DECISIONS.md`, `WORKED_EXAMPLE.md` (with its code, to check the notice's mapping through `V`,
`R`, `U`, `H`), and `PROPOSAL.md` (its name table, to check the notice and the phase-1 column of the history
table). I also checked that the intervening commits `ab77fab` and `e1213c7` touch only `SAMPLE_VARIABLES.md` and
`variable_diagrams_gen.py` (`git diff --stat 8222abe e1213c7`), so the editor's base `8222abe` lost nothing.

**1. Completeness.** Every survey id (R1-R10, S1-S3, T1-T15, C0-C17, D0-D17, L1-L5, P1-P5, W1-W3, E1-E6, A1,
scripts (a)-(j), F1-F5) was traced to the diff. All are handled as ruled. The departures (R3 no change, R8
wording, A1 in the status note, C1 rows not added, W2 wording, the 00:23 changes to D6/D7 and (a) step 7) are
sound. Then this grep, outside `NOTATION_HISTORY.md`, `SAMPLE_VARIABLES.md`, `REVIEW_DECISIONS.md` and
`reviews/`, with every hit read in context:
`grep -rnE "<pattern>" --include=*.md --include=*.py --include=*.sh --include=*.dot .` for `gamma|zeta`,
`\bsigma|\btau`, `\b(mu|xi|nu|eta)_dot\b`, `c_(mu|xi|eta)_dot`, `frame's`, `tangent variable`,
`without a name|no name|unnamed|no public name`, `first-class`, `eight letters`, `option A|option B|options A`,
`_step`. The `REVIEW_DECISIONS.md` hits were read too. Every remaining hit is legitimate: a gloss, the 9.6 note
or the 3.1/3.2 notes (the supervisor's), a dated log entry, a snapshot body with its notice, a library name (the
README table of code locations, `DESIGN_AFTER_REVIEW.md` 3.2 and 4.5, `API_IMPACT.md`), a record script
(`graded_messages_prototype.py:127` and `proto_tails.py:127`, `zeta` as an unrelated symbol; `correspondence_check`,
`index_check2`, `bad_scaling_example` with the library's `sigma`/`tau`), a basis-free step parameter, or the raw
form. `c_*_dot` has no hits left. The exceptions found are reported above (DESIGN 4.2 and section 10 "no name
yet", now stale after 00:30: round 2).

**2. Correctness of every dot rename.** I decided each one myself from the cores:
- `TYPE_CATALOG.md` section 2: dot rows, frame's form, with spaces XU/AR/AL/XD; raw paragraph `xi_dot` XD,
  `mu_dot` AL; section 9 table against the checker output; reject row `mu + mu_R_dot` (the checker:
  `(AL1-; order in) and (AR1+; order in)`); section 11 pairings, against page section 8.
- `DESIGN_AFTER_REVIEW.md`: 3.2 two rows and their library names, against page section 10; 4.1 call site
  (`R, U, H` -> `mu_R_dot`, `xi_U_dot`); 4.3 `Jp` reads `eta_D_dot`.
- `TARGET_DECISIONS.md` glosses T4, T5, T8, T9, T10. The raw `xi_dot` is affine: `xi_term(B_dot, w)` with an
  affine `w`.
- README R4, R8; the notices in `PROPOSAL.md`, `EVALUATOR.md`, `WORKED_EXAMPLE.md`.
- Scripts. `jet_steps_check.py`: frame block through `(V, R, U, H)`, raw block through `(B_dot, G, G, G_dot)`.
  `type_catalog_check.py` likewise. `subtree_gauge_check.py`: `f_*` -> frame's form, bare -> raw form.
  `tilde_multiplier_check.py`, `order0_assessment.py` and `kind_readout_check.py`: frame's form only. Arguments
  of every call are in the same order as before. All correct; no wrong basis found.
- Left as they were and checked: the step parameters (basis-free), `DESIGN` 3.2 line 73 (library names, raw
  meaning), `DESIGN` 4.1 definition.

**3. Changes of substance.** The editor listed 13. All are true except: item 5 (C9), wrong in part, see B1;
item 9 (D13), the attribution is wrong, see B3; item 3 (C6), incomplete, see N1. Not in the editor's list:
the claims of the new `NOTATION_HISTORY.md` sections 1 and 7 (checked against `PROPOSAL.md`'s name table and
the scripts: correct); the gloss on entry 54 of `REVIEW_DECISIONS.md` ("neither option was taken", defensible:
9.6 marks the frame's dots by basis, which is neither A's mark on the raw dots nor B's new letters); and the
`WORKED_EXAMPLE.md` sentence on step 2 (consistent with page section 9).

**4. Fidelity of what moved.** A script extracted each block from `git show e1213c7:...` (DESIGN 3.5 whole, DESIGN
4.3 list, the TARGET note from "The variables that still have no name" to before "6.3.", TYPE_CATALOG sections 13
and 14 to the end of the file). It normalised whitespace and heading levels, then tested containment in
`NOTATION_HISTORY.md`. All five blocks are present word for word. The TARGET note is split in four, and each of
the four parts is present, in its own order. Every pointer names a section that exists: README to section 6;
DESIGN 3.5 to section 2; DESIGN 4.3 to section 3.1; TARGET to 3.2, 4, 5 and 6; TYPE_CATALOG 13 and 14 to 5 and
6; PROPOSAL to section 1. Each says what was there. The banner is clear, and each passage states its source file,
section, commit and date. The file can be used to look up an old name through the table of section 1 and the
key of section 6. Section 5 lacks a key (N7), and the time line lacks 00:23 and 00:30 (N14). The promised
`[ed.: ...]` glosses are never used, which is harmless.

**5. Quotations.** A script took every double-quoted span (8 to 700 characters, no backticks or table pipes) in
the old version of each changed `.md` file and checked that it is still in the new file or in
`NOTATION_HISTORY.md`. All 320 spans are present. The quotations in the reflowed TARGET notes of 05:26 and 21:59
and in the moved 21:57/21:59/22:11/22:24/22:27 passages were also read by eye: unchanged. The 00:23 quotation in
`jet_steps_check.py` and `NOTATION_HISTORY.md` section 7 is an exact substring of the maintainer's words as the
5.1 note gives them.

**6. Scripts.**
- `cd scripts && ./run_all.sh > /home/claude/scratch/notation/rev_run_all.txt 2>&1`, then `diff` against
  `expected_output_3dc15d9.txt`: identical. That file is not in the commit.
- `./review/run_review_checks.sh > /home/claude/scratch/notation/rev_review.txt 2>&1`, then `diff` against the
  re-recorded `review/expected_review_output_3dc15d9.txt`: identical, byte for byte (jax 0.12.0.1, numpy 2.5.3).
- The old record (`git show e1213c7:...`) against the new one, `diff`: only lines 166-169, 184-185, 187, 189-191,
  212-213, 224, 226 and 238-241 changed. A script compared the numeric tokens of all 315 lines, and the counts of
  "equal", "REJECTED", "differ": 0 differences. Every new label agrees with the page: the four type lists; "nu_L_dot,
  frame and raw form equal"; "mu_R_dot (R) and mu_dot (L)"; "xi_dot (D) and xi_U_dot (U)"; "nu (R) and nu_L (L)";
  "eta (U) and eta_D (D)"; the with/against lists, which match page section 7 exactly; the two reject labels; the
  four multiplier rows, which match page section 8.
- `python3 -I review/sample_variables_check.py /home/claude/t3toolbox`: "all checks passed".
- The script diffs, read in full: nothing beyond renaming. The only reshaping is the return of `F` in
  `tilde_multiplier_check.py`, put in a tuple with parentheses (same value), and the lambda parameter `g` -> `nu_L`
  in `order0_assessment.py`.

**7. What the diff should not contain.** No out-of-bounds file is in the diff. The counts of `TARGET_DECISIONS.md`
("56 rows", "22 are decided", "The other 30") are untouched. The 9.6 note, the notes "3.1 ..." and "3.2, what the
notation changes", and `REVIEW_DECISIONS.md` entries 57/58 are untouched. The edits to dated entries are the
listed ones only (L1, L2, L4; T1, T3-T5, T8-T10, T15). A word diff (`git diff --word-diff`) shows no
whitespace-only reflow of an unchanged paragraph. Changed paragraphs were rewrapped (the 05:26 and 21:59 TARGET
notes, with the quotations intact), which is acceptable. One overlong line: N12.

**8. Readability.** `TYPE_CATALOG.md`, `DESIGN_AFTER_REVIEW.md`, and the changed parts of `TARGET_DECISIONS.md`
and `README.md` were read as a reader who knows only the page. Findings: B1, B2, B3, N1, N4, N5, N6, N8, N10.
Otherwise the shapes, the reads of each output, the pairings and the rule for the arrows agree with the page.

**Not checked.** I did not open the Claude doc, so the "28 rows" of N11 is unverified. I did not run the jax-only
review scripts by hand outside the runner; the runner ran them, with jax installed. I did not check the
pre-existing content of the snapshot bodies beyond the places the notices map. I did not recheck the
supervisor's commit `f183e46` beyond reading its diff of `SAMPLE_VARIABLES.md`, `README.md` and
`TARGET_DECISIONS.md`.

---

## Out-of-bounds files: for the supervisor

O1. **`SAMPLE_VARIABLES.md` section 10** (at `f183e46`). The old-to-new table and its "Unchanged" list do not
mention the raw direction, which changed name at 00:30 (`(B_dot, G_dot)` -> `(D_dot, L_dot)`; section 11 records
it). Add one line under "Four things to watch" or after "Unchanged". Otherwise an older text's `(B_dot, G_dot)` at
a manifold frame has no entry in the map.

O2. **`TARGET_DECISIONS.md` row 3.3 against `SAMPLE_VARIABLES.md` section 5.3.** The row asks "Which seeded
copies of the tilde variables are public?". The page says that 3.3 "is only about names ... not a question about
the algorithm". A newcomer gets two descriptions of one row. This is not notation, but B3 touches it.

O3. **`scripts/variable_diagrams_gen.py`** has no row in the README contents table. It appears only in the
paragraph on the doc. Minor.

O4. Nothing wrong found in `sample_variables_check.py` (it passes) or in the `variable_diagrams_gen.py` docstring
at `f183e46`.
