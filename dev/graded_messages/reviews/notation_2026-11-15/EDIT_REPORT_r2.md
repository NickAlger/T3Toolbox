# Edit report, round 2

Editor's round 2, on the working tree over `621b9a1` (round 1 is `1dd65d6`). Nothing committed. The Claude doc
was not touched. Inputs: `REVIEW_r1.md`, the supervisor's rulings of round 2, and `SAMPLE_VARIABLES.md` at
`621b9a1` (sections 1, 5 and 11 re-read).

## 1. The reviewer's findings

| Id | What I did |
| --- | --- |
| B1 | `TYPE_CATALOG.md` section 9: the reviewer's correction, word for word ("the raw form's `xi_dot` and `mu_dot` and their tilde-dots included"; the parenthesis now names `xi_dot`, `mu_dot`, `xi_tilde_dot`, `mu_tilde_dot` as the exception) |
| B2 | The reviewer's correction. Each clause checked first (section 2 below); all are supported. One change: the last clause names the holes, "the `Kp` blocks fit the raw core directions `D_dot` and `L_dot`" (round-2 decision) |
| B3 | `DESIGN_AFTER_REVIEW.md` 4.3: the pointer replaced by the reviewer's text. The D13 row of the findings table has the gloss "(item 1 superseded by 6.4; see 4.3)". The update note now says "a retired name or an overtaken decision" |
| N1 | `TYPE_CATALOG.md` section 8: added "; on the mode edge, which has one basis, by the letter: `w` with, `z` and `Jp` against" |
| N2 | `jet_steps_check.py` docstring: "named by letter and grade (one, nu_tilde_dot_step, renamed on 2026-11-16)" |
| N3 | "the variable parameters" in the `jet_steps_check.py` comment and in `NOTATION_HISTORY.md` section 7 (text and table row). The core parameters are described as "placeholders named after one set of cores they are called with". The lambdas of `order0_assessment.py` are left as they are |
| N4 | `DESIGN_AFTER_REVIEW.md` update note: "and the bare `mu_dot` and `xi_dot` now mean the raw form's (the base point's)" |
| N5 | `DESIGN_AFTER_REVIEW.md` 3.2: a paragraph under the table. The column is for the corewise frame, where it was checked; on a manifold frame the partners are the four dots of page section 8 |
| N6 | `TYPE_CATALOG.md` section 2: one paragraph under the table. The table and the paragraph above it cover the arrays of the checked program; `eta_D` is formed inside `z_step`, and `eta_D_tilde` is not formed at all; for all types see page section 7. I reworded "The table lists the arrays" to "The table, with the paragraph above, covers the arrays", because the raw `xi_dot`, `mu_dot`, `nu_L_dot`, `eta_D_dot` are arrays of the program that are in that paragraph and not in the table |
| N7 | `NOTATION_HISTORY.md` section 5: the same key as section 6, plus "Read with the table of section 1" |
| N8 | `TARGET_DECISIONS.md` pointer: "; neither was taken: the dots are `xi_dot`/`mu_dot` (raw form) and `xi_U_dot`/`mu_R_dot` (frame's form)" |
| N9 | T4 gloss now ends "and the outputs `z` and `Jp`". T8 gloss adds `eta_D_tilde_dot`. Both notes rewrapped |
| N10 | README "How to resume", step 2: `SAMPLE_VARIABLES.md` comes right after `TARGET_DECISIONS.md`, with `NOTATION_HISTORY.md` "when an older text is read" |
| N11 | The count stays, as ruled. Its second point is also done: "(the notation notices added to the status notes on 2026-11-16 are not yet in the tabs)" |
| N12 | Entry 8 lead rewrapped. `git diff --word-diff` shows no word changed |
| N13 | `DESIGN_AFTER_REVIEW.md` 4.1: "At the base point of a manifold frame the raw form calls them with `(L, L, L_dot)` and `D_dot`." The corewise sentence is unchanged |
| N14 | `NOTATION_HISTORY.md` section 1: timeline entries for 00:23 and 00:30, and a row for the raw direction in the three-period table |
| N15 | `EDIT_REPORT_r1.md` header corrected: the work is commit `1dd65d6`, parent `70073af` |
| N16 | `EVALUATOR.md` and `PROPOSAL.md` notices: "carries a basis in its name" in place of "carries the other basis" |
| O3 | README contents: a row for `scripts/variable_diagrams_gen.py`, with its usage line |

## 2. B2: each claim checked

The claims were checked against `arrow_type_check.py` output section 2, lines 212-218 of the record, and against
`SAMPLE_VARIABLES.md` section 6:

- **`V_tilde` and `H_tilde`, the blocks of `g`, read only variables that flow with the arrows.** Supported.
  `V_tilde` reads `eta_tilde`, `w`, `eta`, `w_tilde`, and `H_tilde` reads `mu_tilde`, `xi`, `nu`, `mu`, `xi_tilde`,
  `nu_tilde`. The checker lists all of them under "with".
- **The `Kp` blocks also read variables that flow against the arrows.** Supported. `H_tilde_dot` reads `nu_L`,
  `nu_L_tilde`, the raw `nu_L_dot` and `nu_L_tilde_dot`. `V_tilde_dot` reads the raw `eta_D_dot` and
  `eta_D_tilde_dot`. The checker lists all of them under "against" (as `nu_L`, `nu_L_tilde`, `raw_nu_L_dot`,
  `nu_L_tilde_dot`, `raw_eta_D_dot`, `eta_D_tilde_dot`).
- **Every block's legs are the legs of its hole with every sign flipped.** Supported: the checker prints "fits ...
  True" for all four blocks.

No clause was left out.

## 3. Every `(D_dot, L_dot)` change, by file

**Scripts** (the generic parameters `B`, `G`, `B_dot`, `G_dot` of the term and step functions are unchanged):

- `scripts/review/jet_steps_check.py`
  - The comment above the dot steps: `(L, L, G_dot)` is now `(L, L, L_dot)`.
  - The tilde-dot section comment: "(D, L) and (D_dot, L_dot) at the base point of a manifold frame; (B, G) and
    (B_dot, G_dot) on the corewise frame."
  - In `main()`, `B_dot, G_dot = ...` is now `D_dot, L_dot`. The alias `B, G = D, L` is gone, and every `G` in the
    `Kp` block is now `L`. Every `B_dot` and `G_dot` there is now `D_dot` and `L_dot`, including in the comments
    and in `kind.corewise_hessian((D_dot, L_dot), ...)`.
- `scripts/review/type_catalog_check.py`
  - `program(frame, V, H, D_dot, L_dot, ...)`. The alias `G = L` is gone; its uses are now `L`, and the program's
    `B_dot`/`G_dot` are now `D_dot`/`L_dot`. The comment is "at the base point (D, L), with the raw core direction
    (D_dot, L_dot)".
  - The `CATALOG` comments now say "dual of the space of D" and "dual of the space of L".
  - In `main()`, `B_dot`, `G_dot`, `B_dot2`, `G_dot2`, `B_dot3`, `G_dot3` are now `D_dot`... and `L_dot`.... The
    parameters `B` and `G` of the local lambdas `core` and `tucker` stay: they are generic.
- `scripts/review/arrow_type_check.py`
  - `typed_program(frame, V, H, D_dot, L_dot, ...)` and its call of `T.program`.
  - In `main()`: `raw['D_dot']`, `raw['L_dot']`; the `arrows` keys `D_dot`, `L_dot`; `typed[...]`; the untyped
    call.
  - The printed hole names are now `'D_dot'` and `'L_dot'`.
- `scripts/review/subtree_gauge_check.py`
  - `D_dot, L_dot = (list(c) for c in tvo.tv_corewise_lift(...))`, and their uses.
  - The comment is "along the lifted raw core direction (D_dot, L_dot), through L".

**Documents:**

- `TYPE_CATALOG.md`:
  - Section 2, the `Kp` blocks row: "covector on the space of `D`, of `L` (of `D_dot`, `L_dot`)".
  - Section 2, the paragraph under the table: "the raw core direction `(D_dot, L_dot)`".
  - Section 9 (B2): "fit the raw core directions `D_dot` and `L_dot`".
  - Section 11: "types the raw core directions `(D_dot, L_dot)` this way".
- `DESIGN_AFTER_REVIEW.md`:
  - 3.2, the raw-form row: "along the raw core direction `(D_dot, L_dot)`".
  - 4.1 (N13).
  - 4.2: "The lifted direction is `(D_dot, L_dot)` (decided 2026-11-16, 00:30); the keyword of
    `sampling_variables` that takes it was suggested as `core_direction` with it, and is not confirmed."
  - 10.3: the same, as the undecided item.
- `NOTATION_HISTORY.md`:
  - Section 1: the timeline and the table row (N14).
  - Section 3.2: the editor's note now says the direction "was named `(D_dot, L_dot)` on 2026-11-16 at 00:30".
  - Section 7: `(L, L, L_dot)`, and two new rows for the second pass.

**Left as they are, deliberately:**

- In `NOTATION_HISTORY.md`, the moved passage with "`(B_dot, G_dot)` in the prototype".
- `DESIGN_AFTER_REVIEW.md` 4.4, "`(B, G)` for a trial point".
- `WORKED_EXAMPLE.md`, `B, G` (the point before orthogonalization).
- `TYPE_CATALOG.md` section 1, `(B, G, G, G)`.
- The record scripts. These include `slot_tuple_examples.py`, which calls the direction at `(D, L)` `(Bd, Gd)`. It
  is a record of the declined form (Q3), so the reviewer's list missed nothing that I would change there.

**Places the list missed:** in the scripts and documents above, `type_catalog_check.py`'s `CATALOG` comments
("space of B / G"), `TYPE_CATALOG.md` section 2 (the paragraph) and section 11, and `DESIGN_AFTER_REVIEW.md` 3.2
(the raw-form row). All four are changed.

## 4. Recorded output and checks

- `scripts/review/expected_review_output_3dc15d9.txt` was recorded again. Against the version in `621b9a1`, only
  two lines differ, both labels:

```
217,218c217,218
<       V_tilde_dot  (XD1+, N1-; order real)          fits B_dot  True
<       H_tilde_dot  (AL1-, XD1-, AL2+; order real)   fits G_dot  True
---
>       V_tilde_dot  (XD1+, N1-; order real)          fits D_dot  True
>       H_tilde_dot  (AL1-, XD1-, AL2+; order real)   fits L_dot  True
```

- `run_review_checks.sh` was run once more after the last script edit: its output is byte-identical to the new
  record.
- `run_all.sh`: its output is identical to `scripts/expected_output_3dc15d9.txt`, which is unchanged.
- `python3 -I scripts/review/sample_variables_check.py /home/claude/t3toolbox`: "all checks passed".
- The count line "22 step functions, 60 terms" is unchanged.

## 5. Disagreements, and what I could not do

- None of substance.
- N6: the sentence is reworded slightly; see section 1.
- N7: the key line breaks inside a code span (`mu +` / `mu_dot`). It renders as one span with a space.
- O1 and O2: the supervisor's; not touched.

## 6. Noticed

- Live documents that take a step function's core parameter as literally "the frame's form" may mislead. The new
  comment in `jet_steps_check.py` says the core parameters are placeholders. The same is true of
  `DESIGN_AFTER_REVIEW.md` 4.1's sketch `mu_dot_step(..., R, U, H)`, which the added sentence N13 now covers.
- `TARGET_DECISIONS.md` row 3.3 and page section 5.3 (O2) remain the reviewer's open point for the supervisor.
