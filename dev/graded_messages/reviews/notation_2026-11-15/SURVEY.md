# Survey: where the folder and the Claude doc must change for the notation of 2026-11-15 (23:23)

Survey only. Nothing in the folder or in the doc was changed. The reference is `SAMPLE_VARIABLES.md` (read in
full). Folder state: branch `graded-messages`, head `cad5cbb` (23:35), with one uncommitted change in the working
tree: `scripts/variable_diagrams_gen.py` (21 further diagram rows "added 2026-11-16", i.e. step 2 of the
maintainer's plan is in progress). Doc read at: Tab 1 rev 29, Target decisions rev 128, Sample variables rev 1,
Type catalog rev 20, Variable diagrams rev 5, Design after review rev 10, Worked example rev 7, Evaluator rev 33.

Line numbers are those of the files at `cad5cbb`. "frame's form" = direction `(V, H)` at the frame; "raw form" =
direction `(B_dot, G_dot)` at the base point.

---

## 1. Summary

### 1.1 Counts

Counts are of findings (a finding may cover a block of lines or a whole table), not of occurrences.

| File / place | LIVE | DOT-IN-CONTEXT | SUPERSEDED-IDEA | DATED-RECORD | SCRIPT | FIGURE | INCONSISTENT | Total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `README.md` | 7 | 1 | 0 | 0 | 0 | 0 | 2 | 10 |
| `SAMPLE_VARIABLES.md` | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 3 |
| `TARGET_DECISIONS.md` | 2 | 3 | 5 | 5 | 0 | 0 | 1 | 16 |
| `TYPE_CATALOG.md` | 7 | 3 | 2 | 1 | 0 | 0 | 5 | 18 |
| `DESIGN_AFTER_REVIEW.md` | 11 | 3 | 2 | 0 | 0 | 0 | 2 | 18 |
| `REVIEW_DECISIONS.md` | 0 | 0 | 0 | 5 | 0 | 0 | 0 | 5 |
| `PROPOSAL.md` | 0 | 0 | 1 | 1 | 0 | 0 | 2 | 4 |
| `WORKED_EXAMPLE.md` | 1 | 0 | 0 | 1 | 0 | 0 | 1 | 3 |
| `EVALUATOR.md` | 2 | 0 | 1 | 1 | 0 | 0 | 2 | 6 |
| `API_IMPACT.md` | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 1 |
| scripts (section 5) | 0 | 0 | 0 | 0 | 9 | 0 | 0 | 9 |
| recorded output | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 1 |
| figures, `.dot` | 0 | 0 | 0 | 0 | 0 | 3 | 0 | 3 |
| Claude doc (tab-only or tab-lag items) | 3 | 0 | 0 | 0 | 0 | 1 | 2 | 6 |
| **Total** | **33** | **10** | **11** | **16** | **9** | **4** | **20** | **103** |

A finding with two classes (for example LIVE + INCONSISTENT) is counted in both columns, so a row's classes can
sum to more than the number of its numbered items.

Files with no finding: `EXECUTION_PLANS.md`, `T3POLYNOMIAL_SCALING_SURVEY.md`, `reviews/review_1_design.md`,
`reviews/review_2_library_fit.md` (records), `scripts/parked/*`, `reviews/review_1_scratch/*`, the phase-1 scripts
(section 5.3), `scripts/expected_output_*.txt`, `scripts/timings_2core_cpu_3dc15d9.txt`, the runners.

### 1.2 The places that most need care

1. **`TYPE_CATALOG.md` §9, the checker's reject row `mu + mu_dot` (line 244) and `arrow_type_check.py` line 231.**
   Under the new names `mu + mu_dot` is well typed (both are `L`-basis vectors; §13 line 402 already says so).
   The rejected sum is `mu + mu_R_dot`. A mechanical "keep `mu_dot`" here inverts the meaning of the row.
2. **`jet_steps_check.py` and the four scripts that import it.** The same step functions are called in the frame's
   form and in the raw form, and inside the tilde-dot steps the parameters `mu_dot`, `xi_dot` ALREADY mean the raw
   form (lines 129-151). A blanket `mu_dot -> mu_R_dot` would break them. Also two pairs of arrays collide under
   the new names in one program (`nu_dot`/`gamma_dot` both -> `nu_L_dot`; `eta_dot`/`c_eta_dot` both ->
   `eta_D_dot`): the scripts need a second spelling (open question Q2). `arrow_type_check.py` keys on step
   function names as strings (`'mu_dot_step'`, `'gamma_tilde_dot_step'`).
3. **`TYPE_CATALOG.md` §14 (lines 430-548).** It uses BOTH meanings: bare `mu_dot`/`xi_dot` there already mean the
   raw form and "the frame's `mu_dot`" means `mu_R_dot` (lines 467-480). Read each occurrence.
4. **`DESIGN_AFTER_REVIEW.md` §3.2 table row (line 86).** The "Today" column pairs `dxi`/`xi_dot` and
   `sigma`/`mu_dot` as one variable. Under the new notation they are two (`xi_U_dot` vs `xi_dot`, `mu_R_dot` vs
   `mu_dot`); only `tau`/`nu_dot` and `deta`/`eta_dot` coincide. `PROPOSAL.md` line 475 makes the same claim.
5. **`TARGET_DECISIONS.md` notes to section 6, lines 824-869** (variables without names, `zeta`, options A and B,
   naming widened): the main block of superseded ideas, inside dated notes that quote the maintainer.
6. **The "Target decisions" tab lags its file by the whole evening** (no 21:35-23:23 notes, no row 9.6, 3.1 still
   "to confirm", counts 55/20 against 56/22). Notation edits cannot simply be mirrored; the tab needs the missing
   content first (open question Q7).
7. **`DESIGN_AFTER_REVIEW.md` §3.5 `gamma` and §4.3 "Variables with no public name"**: live text whose subject is
   a retired idea ("an auxiliary variable, not a `nu`"; "no public name").
8. **`TARGET_DECISIONS.md` line 819-820 (decision of 21:57)**: the affine variables "`xi`, `xi_dot`". At 21:57
   `xi_dot` was the frame's; the decided list is now `w`, `xi`, `xi_dot`, `xi_U_dot` (`SAMPLE_VARIABLES.md` §7).
9. **"Mode edge" means two different edges** in the folder: in `PROPOSAL.md`, `WORKED_EXAMPLE.md`, `EVALUATOR.md`
   ("mode leg") it is the edge `SAMPLE_VARIABLES.md` calls the Tucker edge; there "ambient edge" is the page's mode
   edge. Not a name of a variable, but it will mislead a reader of the new page.
10. **The arrow statements.** "`gamma`, `z` and the dot variables flow against the arrows" (`TYPE_CATALOG.md`
    §9, `TARGET_DECISIONS.md` line 322, `REVIEW_DECISIONS.md` line 774) is false for the raw dots under the new
    names: the checker's own output puts `c_xi_dot`, `c_mu_dot` (now `xi_dot`, `mu_dot`) WITH the arrows. In the
    new notation the rule is clean: a variable flows against the arrows exactly when its name carries a basis
    (or it is an output).

---

## 2. Proposed policy for the editor

**LIVE.** Rename in place, using the exact new names of `SAMPLE_VARIABLES.md` §10. Where a sentence makes a claim
that was true of the old name only (for example "`gamma` is not a `nu`"), do not rename it; treat it as
SUPERSEDED-IDEA. Add `NOTATION_HISTORY.md` to the Contents table of `README.md`.

**DOT-IN-CONTEXT.** Decide per occurrence from the cores the variable goes through: through `R`, `U`, `H`, `V`
(the direction `(V, H)`) it is the frame's form: `mu_R_dot`, `xi_U_dot`, `nu_L_dot`, `eta_D_dot`; through `G`,
`G_dot`, `B_dot` or "the raw core direction" it is the raw form: `mu_dot`, `xi_dot`, `nu_L_dot`, `eta_D_dot`. Never
run a global replace of `mu_dot` or `xi_dot`. Use word-boundary matching (`\bmu_dot\b` does not match inside
`c_mu_dot`, `mu_dot_step` or `mu_tilde_dot`, which is what is wanted).

**SUPERSEDED-IDEA.** Keep the passage verbatim, add a one-line banner "Deprecated (2026-11-15, 23:23): superseded
by the notation of `SAMPLE_VARIABLES.md`", and move it to `NOTATION_HISTORY.md` when it is a self-contained block
of Claude's analysis (TYPE_CATALOG §13-14, TARGET notes lines 824-869, DESIGN §3.5 rationale, DESIGN §4.3 list).
Leave behind one line: "`<what>` (date): superseded at 23:23; the passage is in `NOTATION_HISTORY.md` §n." Where
the passage is a dated log entry (`REVIEW_DECISIONS.md`), do not move it: a log is history in place.

**DATED-RECORD.** Never edit a quotation of the maintainer. Never rename inside a dated log entry, a reviewer's
report, a recorded output or a snapshot document (`PROPOSAL.md`, `API_IMPACT.md`, `WORKED_EXAMPLE.md`,
`EVALUATOR.md`, `EXECUTION_PLANS.md`, `scripts/plan_mockup.py`). Instead: (a) add one notice per document (at the
existing status note or at the "Words changed" header) that maps the document's names to the new ones; (b) add
an inline gloss `(now \`nu_L\`)` only where a dated note is still read as the current statement of a decision
(listed per finding below).

**SCRIPT.** Live prototypes are renamed (open question Q3 decides which are live); records get at most a one-line
note in the docstring. Rename in the safe order of section 5; re-run both runners; re-record
`scripts/review/expected_review_output_3dc15d9.txt` (open question Q4). Do not touch
`scripts/variable_diagrams_gen.py` while step 2 of the maintainer's plan has uncommitted work in it, except the
docstring finding V1, in coordination.

**FIGURE.** The three PNGs and `dependency_graph.dot` illustrate pre-review documents and use `_t`/`_z`; leave
them, add a one-line caption note where they are embedded.

**INCONSISTENT.** Report to the supervisor; fix only what is decided.

### 2.1 Proposed outline of `NOTATION_HISTORY.md`

```
# Notation history (deprecated names and naming ideas)
Banner: everything here is superseded by SAMPLE_VARIABLES.md (2026-11-15, 23:23); kept as the record.
1. Timeline of the names
   - Phase 1 (89ac203f, 2026-11-13/14): grades _t, _z, _tz; letters xi, mu, nu, eta, w, z, phi
     (PROPOSAL.md "Edge variables"; pointer only, PROPOSAL stays a snapshot).
   - 2026-11-14 (decision 8): _dot, _tilde, _tilde_dot; gamma; seven *_term functions (REVIEW_DECISIONS entry 8).
   - 2026-11-15 22:11: zeta. 22:41: four letters and a basis written out. 23:23: decided.
   - One table: phase-1 name | 2026-11-14 name | 22:11 name | decided name (extends SAMPLE_VARIABLES §10
     with the _t/_z column: mu_t -> mu_R_dot in a manifold frame / mu_dot at the corewise frame, etc.).
2. gamma: "an auxiliary variable, not a nu"            <- DESIGN_AFTER_REVIEW §3.5 lines 143-145, 151-152
3. "Variables with no public name" / "without a name"  <- DESIGN_AFTER_REVIEW §4.3 lines 265-270;
                                                          TARGET_DECISIONS lines 824-848
4. zeta (22:11)                                        <- TARGET_DECISIONS lines 849-852
5. The dot collision: options A (a mark) and B (sigma/tau letters for the frame's tangent variables)
                                                       <- TARGET_DECISIONS lines 853-859; TYPE_CATALOG §13 (379-428)
6. Naming widened: core-and-leg as a name; eight letters vs four letters and a basis
                                                       <- TARGET_DECISIONS lines 860-869; TYPE_CATALOG §14 (430-548)
7. Prototype stand-ins and old script identifiers      (c_mu_dot, c_xi_dot, c_eta_dot, eta_tilde_dot, gamma*,
                                                        f_*_dot, zeta*, the step-function names), if scripts are renamed
```

---

## 3. Findings by file

### 3.1 `README.md`

| # | Location | Class | What | Action |
| --- | --- | --- | --- | --- |
| R1 | 23-64 "Status" | LIVE | The last status bullet is "Resumed 15:34". Nothing on 2.5/2.3/2.9/6.7 or on the notation decided at 23:23 | Add a bullet: "Notation decided (2026-11-15, 23:23): four letters and a basis; reference `SAMPLE_VARIABLES.md`; older names in `NOTATION_HISTORY.md`" |
| R2 | 71-131 Contents | LIVE | No row for `NOTATION_HISTORY.md` (to be created) | Add a row |
| R3 | 109 "`scripts/review/jet_steps_check.py` \| **The explicit form with the decided names**" | LIVE | Its names are those of 2026-11-14 | If renamed (Q3): keep text; if frozen: "the explicit form with the names of 2026-11-14"; and mark `sample_variables_check.py` (line 128) as the program in the decided names |
| R4 | 127 `subtree_gauge_check.py`: "Checks that the frame's dot variables and the base point's dots along the lifted direction are one tangent vector" | DOT-IN-CONTEXT (sure) | "the frame's dot variables" = frame's form | "that the frame's form (`xi_U_dot`, `mu_R_dot`, `nu_L_dot`, `eta_D_dot`) and the raw form along the lifted direction (`xi_dot`, `mu_dot`, `nu_L_dot`, `eta_D_dot`) are one tangent vector" |
| R5 | 133-136 "ten tabs" | INCONSISTENT | The doc has eleven tabs (the list in the same sentence names eleven) | "eleven tabs"; and "Variable diagrams" status per Q7 |
| R6 | 199-200 "`scripts/review/jet_steps_check.py` is the starting point for the explicit functions and the names" | LIVE | The names are now in `sample_variables_check.py` | "... starting point for the explicit functions; the names are those of `SAMPLE_VARIABLES.md` and `scripts/review/sample_variables_check.py`" |
| R7 | 243-244 "The frame's `nu` and the base point's `nu` differ by a bond-sized gauge matrix" | LIVE | The "base point's `nu`" is `nu_L` | "`nu` and `nu_L` differ by a bond-sized gauge matrix" (the "established" list is read as current) |
| R8 | 382-383 "`mu` and `mu_z` live on `rL`; `mu_t` lives on `rR`. Components must be keyed by grade, not stacked on an axis." | LIVE | Old `_z`/`_t` grades; the lesson is now carried by the basis in the name | "`mu` and `mu_tilde` live on `rL`; the frame's dot `mu_R_dot` lives on `rR`: the basis is part of the name (`SAMPLE_VARIABLES.md` §2)" |
| R9 | 403-404 "the adjoint needs only `nu_z` and `eta_z`. For them `nu_z` is the residual times the plain `nu`" | LIVE | `_z` grade | `nu_tilde`, `eta_tilde`; "`nu_tilde` is the residual times the plain `nu`" |
| R10 | 380-381 "A manifold frame has no plain probe output and no grade-`t` core gradient" | INCONSISTENT (minor) | `t` grade = old name of the dot grade | "no dot-grade core gradient" (optional) |

No change: 105, 245-246 ("frame's components", gauge), 351-372 (open questions of phase 1, with `_dot`/`_tilde`
noted as decided), 419-420 and 422-439 (library names: `dxis`, `compute_sigma`, `sigma_hat`; facts about the
library).

### 3.2 `SAMPLE_VARIABLES.md`

No renames. Seen, need no change: §10 table column "In this folder until now" (old names on purpose); the
"Four things to watch" list; line 9 pointer to `NOTATION_HISTORY.md` (file to be created).

| # | Location | Class | What |
| --- | --- | --- | --- |
| S1 | §7 line 288: "the seed `w_tilde[i]` is `T + W + C + (N_i,)`" | INCONSISTENT (possible omission) | The `K` row (line 259) gives `K` to "a tilde variable whose seed was made from `Jp`"; the seed of `JtJp` itself is made from `Jp` and so carries `K`: `T + W + [K] + C + (N_i,)` |
| S2 | §7 line 245 "The letter gives the variance" | INCONSISTENT (note, not an error) | True in the catalog's coordinates (`type_catalog_check.py`). Under the arrow convention of 9.5 the letter does not fix with/against: the checker's output puts every variable in its usual basis WITH the arrows and every variable whose name carries a basis (and `z`, `Jp`) AGAINST. Worth one sentence, so the two readings are not confused |
| S3 | §1 line 30 "Mode edge `i` joins Tucker core `i` to the input" | INCONSISTENT (terminology across files) | Correct on this page, but older documents use "mode edge" for the Tucker edge (see P3, W3, E6) |

Checked and consistent: the recurrences of §5 (bases of every core leg), the 22 of 32 count, the outputs and what
each reads, the pairings of §8, the shapes of §7 (apart from S1), §9.

### 3.3 `TARGET_DECISIONS.md`

| # | Location | Class | What | Action |
| --- | --- | --- | --- | --- |
| T1 | 10-29 "Where this stands (pause of 2026-11-15, 05:53)" | DATED-RECORD | Stale state at 05:53 (no notation); dated heading | No rename. Optional: one line under "Since the pause": "9.6, the notation, decided at 23:23 (`SAMPLE_VARIABLES.md`)" |
| T2 | 54 group B: "3.2 whether `gamma` is first-class (to confirm, depends on 4.1)" | LIVE | `gamma` retired; "first-class" is the old question | "3.2 whether the second basis of the right of a bond, `nu_L`, is needed or a different frame convention removes it (to confirm, depends on 4.1)" |
| T3 | 297-298 (note of 04:57): "`eta`, `nu`, `gamma`, `z` covectors" | DATED-RECORD | Summary of the catalog at 04:57 | Inline gloss `gamma` -> "`gamma` (now `nu_L`)" |
| T4 | 321-323 (note of 05:26): "`gamma`, `z` and the dot variables flow against them" | DATED-RECORD + INCONSISTENT | Under the new names the raw dots `mu_dot`, `xi_dot` flow WITH the arrows | Gloss: "(now: `nu_L`, `z`, and the frame's dots `mu_R_dot`, `xi_U_dot`, `nu_L_dot`, `eta_D_dot`; in the new names, what flows against the arrows is what carries a basis in its name, and the outputs)" |
| T5 | 474 (2.9 decided, 19:08): "A product with an affine operand (`xi`, `w`, `xi_dot`)" | DOT-IN-CONTEXT (sure) | At 19:08 `xi_dot` was the frame's | Gloss: "`xi_dot` (now `xi_U_dot`; the raw `xi_dot` is affine too)". This note is read as the content of row 2.9 |
| T6 | 590 row 3.2: "Is `gamma` a first-class variable, or does a different frame convention remove the need for it?" | LIVE | Row question in a retired name | "Is `nu_L` (the right of a bond in the `L` basis; `gamma` until 23:23) needed, or does a different frame convention remove the need for a second basis?" The note at 602-605 already says this |
| T7 | 771-773 (21:35): "The tilde variables seeded by `Jp` and the dot variables of `Kp` stay internal" | DATED-RECORD | Withdrawn at 21:44 (line 792) | No change |
| T8 | 803-804 (21:44): "**Still without a name:** the `eta`-like variable through `L` that `z` reads, and the dot variables of `Kp`" | SUPERSEDED-IDEA (inside a dated note) | Answered at 23:23 | Keep; gloss "(named at 23:23: `eta_D`; `mu_dot`, `xi_dot`, `nu_L_dot`, `eta_D_dot`)" |
| T9 | 808-811 (21:44): "The input `w + s p`, `xi` and `xi_dot` have two coefficients" | DOT-IN-CONTEXT (sure) | Frame's `xi_dot` | Gloss "(`xi_dot`: now `xi_U_dot`; the raw `xi_dot` too)" |
| T10 | 819-820 (confirmed 21:57): "The affine variables (the input `w + s p`, `xi`, `xi_dot`) carry a two-slot group" | DOT-IN-CONTEXT (sure: at 21:57 the raw dot was still `c_xi_dot`) | A decision record still read as the rule | Gloss "(now `w`, `xi`, `xi_dot`, `xi_U_dot`: `SAMPLE_VARIABLES.md` §7)" |
| T11 | 824-848 "The variables that still have no name" (21:57; table rows 1-6; "Also without a name") | SUPERSEDED-IDEA | List of unnamed variables; all named at 23:23 | Move to `NOTATION_HISTORY.md` §3 (quotes of 21:57 and 21:59 verbatim). Pointer: "21:57: Claude listed six unnamed variables; all were named at 23:23 (9.6). See `NOTATION_HISTORY.md` §3." |
| T12 | 849-852 "`zeta`, decided (22:11)" | SUPERSEDED-IDEA | `zeta` retired; "The base point has the four letters `xi`, `mu`, `gamma`, `zeta`" | Move to §4 with the quote "zeta is great!". Pointer: "`zeta` (22:11) became `eta_D` at 23:23." |
| T13 | 853-859 "Rows 5 and 6, two options, not decided. A ... B ..." | SUPERSEDED-IDEA | Options A and B | Move to §5 |
| T14 | 860-869 "The naming problem widened (22:24 and 22:27)" | SUPERSEDED-IDEA | Core-and-leg naming, "eight letters; or four letters and a basis", "the frame's `nu_dot` is `gamma_dot`" | Move to §6; leave: "22:24-22:41: the naming was rethought from first principles; the result is 9.6 (`SAMPLE_VARIABLES.md` §2); the path is in `NOTATION_HISTORY.md` §6." |
| T15 | 1060-1063 (21:59): "`sigma`, `tau` and the other letters the refactor frees may be given new meanings" | DATED-RECORD (Claude's consequence sentence after the quote) | Recycling `sigma`/`tau` was not taken | Gloss: "(The notation of 9.6 uses neither.)" The library's `sigma`/`tau` names stay facts about the library |

Also: the count paragraph at 33-46 (56 rows, 22 decided, 30 open) is consistent with the rows (checked). Lines
60-61, 589, 599-607, 1056, 1064-1081 already use the new notation; no change. Lines 692-710 (library table:
`compute_sigma`, `compute_tau_tilde`) and 729-733 (`dxi_tilde` = the target's `eta_tilde`, correct) and 870-872
(`xi_tilde` against `dxi_tilde`, still a real near miss) need no change.

### 3.4 `TYPE_CATALOG.md`

Status lines 3-10 say "Nothing here is a decision". Sections 1-7 are the catalog (read as the current account of
types); 8-12 are dated sections of Claude's analysis that are still the account of the arrows; 13-14 are the
naming analysis that 9.6 overtook. Recommendation (Q5): rename in place in 1-12 with a banner at the top; move
13-14 to `NOTATION_HISTORY.md`.

| # | Location | Class | What | Action |
| --- | --- | --- | --- | --- |
| C0 | 3-10 status | LIVE | No mention of the notation | Add: "Names updated to `SAMPLE_VARIABLES.md` (2026-11-15, 23:23); sections 13-14 moved to `NOTATION_HISTORY.md`; its section 7 is the reference for types and shapes" |
| C1 | 32-57 §2 table | LIVE + DOT (sure) | `gamma` -> `nu_L`; dot rows (manifold frame, line 59): `xi_dot` (XU) -> `xi_U_dot`, `mu_dot` (AR) -> `mu_R_dot`, `nu_dot` (AL) -> `nu_L_dot`, `eta_dot` (XD) -> `eta_D_dot`; `gamma_tilde` -> `nu_L_tilde`; `gamma_tilde_dot` -> `nu_L_tilde_dot`; `eta_tilde_dot` (XD) -> `eta_D_tilde_dot` | Rename. Consider adding the rows the table lacks: `eta_D`, `eta_D_tilde` and the raw `xi_dot` (XD), `mu_dot` (AL) |
| C2 | 59-61 "there the dot variables are `xi_dot` in `XD`, `mu_dot` in `AL`, `gamma_dot` a covector of `AL`, and `eta_dot` a covector of `XD`" | DOT-IN-CONTEXT (sure: raw form) | | "`xi_dot` in `XD`, `mu_dot` in `AL` (names unchanged, now with this meaning), `nu_L_dot` a covector of `AL`, `eta_D_dot` a covector of `XD`" |
| C3 | 65-66 rule 1: "`eta`, `nu`, `gamma` and `z` are covectors" | LIVE | | "`eta` and `nu`, in either basis, and `z` are covectors" |
| C4 | 72-74 "On every edge the two partners are in dual network spaces: `xi` and `eta`, `mu` and `gamma`, `w` and `z`" | LIVE + INCONSISTENT | `gamma` -> `nu_L`. But `xi` (vector of `XD`) and `eta` (covector of `XU`) are not dual to each other; in the new names the partners are `xi`/`eta_D` and `mu`/`nu_L` (base point's bases), `xi_U_dot`/`eta` and `mu_R_dot`/`nu` (frame's other bases), `w`/`z` | Rewrite with those pairs (`SAMPLE_VARIABLES.md` §8) |
| C5 | 136 "Each of the 30 variables" | INCONSISTENT (count) | 30 arrays of `type_catalog_check.py` (incl. `w`, `z`, `Jp`, `w_tilde`, four blocks). After the rename two pairs share a name (`nu_L_dot`, `eta_D_dot` in both forms) and `eta_D`, `eta_D_tilde` are not among them | "each of the 30 arrays of the program"; same count at `REVIEW_DECISIONS.md` 747 (record, no change) |
| C6 | 160-162 "the letter fixes the direction of the arrow" | INCONSISTENT | The checker shows the letter fixes the direction of travel; with/against the arrow is fixed by whether the basis is the usual one | Reword per the checker (see C8) |
| C7 | 213-214 "424 contractions ... 176 sums, in the 22 step functions" | (count; no change unless the step functions are split, Q1) | | |
| C8 | 220-223 §9 table "With the arrows ... Against the arrows `gamma`, `z`; the dot variables `xi_dot`, `mu_dot`, `nu_dot`, `eta_dot`, `Jp`; `gamma_tilde`" | LIVE + DOT (sure) + INCONSISTENT | The table omits the raw dots, which the checker lists WITH the arrows (`c_xi_dot`, `c_mu_dot`, `xi_tilde_dot`, `mu_tilde_dot`) and the raw ones against (`gamma_dot`, `c_eta_dot`, `gamma_tilde_dot`, `eta_tilde_dot`) | With: `w`, `xi`, `mu`, `nu`, `eta`, their tildes, and the raw `xi_dot`, `mu_dot`, `xi_tilde_dot`, `mu_tilde_dot`. Against: `nu_L`, `z`, `xi_U_dot`, `mu_R_dot`, `nu_L_dot` (both forms), `eta_D_dot` (both forms), `Jp`, `nu_L_tilde`, `nu_L_tilde_dot`, `eta_D_tilde_dot` |
| C9 | 225-226 "What flows against them has passed a hole (the dot grade) or is on its way out (`gamma`, `z`)" | INCONSISTENT | False for the raw dots | "What flows against the arrows is exactly what carries a basis in its name, and the outputs" (Claude's reading; for the supervisor) |
| C10 | 231-232 "`gamma` enters `L` through the leg where `L`'s arrow leaves" | LIVE | | `nu_L` |
| C11 | 242 reject row "`gamma` where `nu` is wanted, in `eta_term(mu, U, gamma)`" | LIVE | | "`nu_L` where `nu` is wanted, in `eta_term(mu, U, nu_L)`" |
| C12 | 244 reject row "`mu + mu_dot` \| A sum of unlike types" | DOT-IN-CONTEXT (sure; critical) | The checker's `mu_dot` is the frame's | `mu + mu_R_dot`. Do NOT leave `mu_dot`: with the new meaning the sum is well typed |
| C13 | 323-331 §11 point 2: "`xi_tilde` of `eta_dot`, `eta_tilde` of `xi_dot`, `mu_tilde` of `nu_dot`, `nu_tilde` of `mu_dot`"; "the dot variables"; "`mu_tilde` of `gamma`" | DOT-IN-CONTEXT (sure: frame's form, the program `(V, H) -> Jp`) | | `eta_D_dot`, `xi_U_dot`, `nu_L_dot`, `mu_R_dot`; "the dot variables of the frame's form"; "`mu_tilde` of `nu_L`" (on the corewise frame: of `nu`). §14 line 542-544 already has the new list |
| C14 | 379-428 §13 "The letters by edge, and the variables without names (22:11)" | SUPERSEDED-IDEA | Options A/B, slots without a letter, `zeta`, `gamma`, "makes a letter fix the edge" | Move whole to `NOTATION_HISTORY.md` §5 (maintainer's quotes verbatim). Pointer: "§13 (22:11), the letters by edge and options A and B: superseded by 9.6; in `NOTATION_HISTORY.md` §5." |
| C15 | 430-505 §14 (22:24-22:27) | SUPERSEDED-IDEA (partly adopted) | "Three things identify a variable", the 22-of-32 table and "One dot" were adopted into `SAMPLE_VARIABLES.md` §2, §3, §5.2; the core-and-leg naming survives as a definition (§5.1); "Two schemes" (494-505: eight letters with `sigma`/`tau`) is superseded. Mixed meanings: bare `mu_dot`/`xi_dot` at 469-480 are already the raw form | Move whole to `NOTATION_HISTORY.md` §6 (it is the record of how 9.6 was reached). If kept instead: rename "the frame's `mu_dot`" -> `mu_R_dot`, "the frame's `xi_dot`" -> `xi_U_dot`, "the frame's `nu_dot`"/`gamma_dot` -> `nu_L_dot`, "the frame's `eta_dot`"/`zeta_dot` -> `eta_D_dot`, `gamma` -> `nu_L`, `zeta` -> `eta_D`, and leave bare `mu_dot`, `xi_dot` |
| C16 | 507-548 "The second scheme written out (22:41) ... Not decided" | DATED-RECORD | "Not decided" (509) is stale; the table duplicates `SAMPLE_VARIABLES.md` §10 | Move with §14, or replace by "Decided at 23:23; the table is `SAMPLE_VARIABLES.md` §10" |
| C17 | 222 and 225-228, 312-361 | (no change beyond C8-C13) | Sections 8, 10, 12 have no old variable names | |

### 3.5 `DESIGN_AFTER_REVIEW.md`

The document states the design of 2026-11-14 by topic, and README says it "wins" over the older ones; it is
Claude's summary (no quotes in the renamed spots). Recommendation (Q6): treat as LIVE, rename, and add a line to
the top update note.

| # | Location | Class | What | Action |
| --- | --- | --- | --- | --- |
| D0 | 3-11 update note | LIVE | Lists what is overtaken; nothing on names | Add: "A fourth, since 2026-11-15, 23:23: the names of the sampling variables (section 3 and below) follow `SAMPLE_VARIABLES.md`; `gamma` is `nu_L`, and the dot variables carry a basis in the frame's form" |
| D1 | 57-58 "edge variables (`xi`, `mu`, `nu`, `eta` at every grade, and `gamma`)" | LIVE | | "... at every grade, in either basis (`nu_L`, `eta_D`, `mu_R_dot`, `xi_U_dot`)" |
| D2 | 86 row "`xi_dot`, `mu_dot`, `nu_dot`, `eta_dot` \| `dxi` / `xi_dot`, `sigma` / `mu_dot`, `tau` / `nu_dot`, `deta` / `eta_dot`" | DOT-IN-CONTEXT (sure) + INCONSISTENT | Proposed column: frame's form -> `xi_U_dot`, `mu_R_dot`, `nu_L_dot`, `eta_D_dot`. "Today" column joins different variables: library `dxis`/`sigmas` are `xi_U_dot`/`mu_R_dot`, but `xis_dot`/`mus_dot` (two-hole kernel) are the raw `xi_dot`/`mu_dot` | Split into two rows: `xi_U_dot`, `mu_R_dot`, `nu_L_dot`, `eta_D_dot` (today `dxis`, `sigmas`, `taus`, `detas`); `xi_dot`, `mu_dot`, `nu_L_dot`, `eta_D_dot` (today `xis_dot`, `mus_dot`, `nus_dot`, `etas_dot`) |
| D3 | 69-70 "`_dot` is already the library's spelling with this meaning (`xi_dot`, `mu_dot`, `mus_dot`: 149 uses)" | (no change) | Those library uses are the raw form; consistent with the new bare `mu_dot` | |
| D4 | 105 `LEFT` -> `nu_term`: "the `nu` family, and `gamma`" | LIVE | | "the `nu` family, in either basis (`nu`, `nu_L`)" |
| D5 | 141-152 "### 3.5 `gamma`" | LIVE + SUPERSEDED-IDEA | Facts stay (read by `z` and `Kp`; companion `gamma_tilde`; library `nu_c = T nu`; coincide with `nu` at a trial point). Retired: "a different variable ... an auxiliary variable, not a `nu`" (143-145) and "The name is free ..." (151-152) | Heading "3.5 `nu_L` (`gamma` until 2026-11-15, 23:23)"; rename the facts (`nu_L`, `nu_L_tilde`); move 143-145's rationale and 151-152 to `NOTATION_HISTORY.md` §2; replace with "The right of a bond in the `L` basis is `nu_L`: the same side as `nu`, written in the base point's basis" |
| D6 | 200-207 code: `def mu_dot_step(mu_dot, mu, xi, xi_dot, R, U, H)`, `mu_dot = left_to_right(mu_dot_step, zeros, d, mu, xi, xi_dot, R, U, H)` | DOT-IN-CONTEXT (sure: `R, U, H`) | Variables -> `mu_R_dot`, `xi_U_dot`. Function name depends on Q1 (`mu_dot_step` if steps are named by letter and grade; else `mu_R_dot_step`) | Rename after Q1 |
| D7 | 212, 336 "22 step functions with 60 terms" | (count; depends on Q1) | | |
| D8 | 249-250 table: `gamma`, `gamma_tilde` | LIVE | | `nu_L`, `nu_L_tilde` (and add `eta_D` to the first row: it depends on the point and the samples) |
| D9 | 251 "the dot variables" | (no change) | Covers both forms | |
| D10 | 255 `z` reads "`xi`, `mu`, `gamma`" | LIVE + INCONSISTENT | `SAMPLE_VARIABLES.md` §6: `z` reads `xi`, `mu`, `nu_L`, `eta_D` | `xi`, `mu`, `nu_L` (and `eta_D`) |
| D11 | 258 `Jp` reads "`eta`, `eta_dot`" | DOT-IN-CONTEXT (sure: `Jp = z_term(eta_D_dot, D) + z_term(eta, V)`) | | `eta`, `eta_D_dot` |
| D12 | 260 `Kp` reads "`xi`, `mu`, `gamma`, `r`, `xi_tilde`, `mu_tilde`, `gamma_tilde`" | LIVE | | `nu_L`, `nu_L_tilde` |
| D13 | 265-270 "Variables with no public name" (Jp-seeded tildes; the `eta`-like variable through `L`; the dot variables of `Kp`) | SUPERSEDED-IDEA | Contradicts 6.4 confirmed at 21:57 (every variable can be wanted and held) and 9.6 (all named) | Mark deprecated and move to `NOTATION_HISTORY.md` §3; pointer: "Since 2026-11-15 every variable is public (target decision 6.4) and named (`SAMPLE_VARIABLES.md`); the `Jp`-seeded copies are target decision 3.3" |
| D14 | 275 "For a frame it reads `xi`, `mu` and `gamma`" | LIVE | | "`xi`, `mu`, `nu_L` (through `eta_D`)" |
| D15 | 303 "Plain: `mu`, `nu`, `gamma`" | LIVE | | `nu_L` |
| D16 | 564 10.1 item 7 "**`gamma` and `gamma_tilde` as public variables.**" | LIVE | | `nu_L` and `nu_L_tilde` |
| D17 | 617 "D13 \| ... gauge (`gamma`)" | LIVE | | `nu_L` |

No change: 75-95 (library `tau_tilde` etc. and the four tilde pairs: consistent with §10 of the page), 186-195,
406 ("'sweep' ... `xi`, `mu`, `nu`, `eta`"), 585 (lifted-direction argument still unnamed: consistent with the
page's §11).

### 3.6 `REVIEW_DECISIONS.md` (the dated log)

All findings DATED-RECORD. No moves; no renames.

| # | Location | Action |
| --- | --- | --- |
| L1 | 8-11 "Words changed during the evening" | Add one bullet: "On 2026-11-15 (23:23, entry 57) the names were fixed again: `gamma` is `nu_L`, `zeta` (entry 54) is `eta_D`, the frame's `mu_dot`, `xi_dot`, `nu_dot`, `eta_dot` (this log's `_dot` of entry 8) are `mu_R_dot`, `xi_U_dot`, `nu_L_dot`, `eta_D_dot`, and the bare `mu_dot`, `xi_dot` now mean the base point's. Entries before 57 use the words of their time; `SAMPLE_VARIABLES.md` §10 has the table." |
| L2 | 306-318 entry 8 "The gauge part of D13: `gamma`" | Inline gloss in the bold lead: "`gamma` (now `nu_L`)". Target decision 3.2 cites this decision. Lines 317-318 ("`gamma_tilde`"; "The `eta`-like helper that only `z` reads stays internal") covered by L1 |
| L3 | 188 `mu_t_step(mu_t, mu, xi, xi_t, R, U, H)`; 256-266 (`mu_t -> mu_dot`); 463-464; 747 ("all 30 variables"); 774, 780 (`gamma`, dot variables against the arrows) | Covered by L1. No inline change |
| L4 | 1135-1175 entries 53-55 (variables without names; `zeta` decided; options A/B; eight letters) | Optional gloss at the head of 53 and 54: "(superseded by entry 57)" |
| L5 | 1177-1197 entry 56 ("the other fifteen variables are not drawn"); 1199-1218 entry 57 | No change (true at the time; 57 is the decision) |

### 3.7 `PROPOSAL.md` (snapshot of 2026-11-14 at `89ac203f`, with a status note)

| # | Location | Class | Action |
| --- | --- | --- | --- |
| P1 | 3-15 status note, bullet "The names" (8) | DATED-RECORD | Add: "Since 2026-11-15 (23:23) the names are those of `SAMPLE_VARIABLES.md`. This document's grades map as: `_z` -> `_tilde`; `_t` in a manifold frame carries the other basis: `xi_t` -> `xi_U_dot`, `mu_t` -> `mu_R_dot`, `nu_t` -> `nu_L_dot`, `eta_t` -> `eta_D_dot`; `_t` at the corewise frame -> `_dot`; `_tz` -> `_tilde_dot` (`eta_tz` -> `eta_D_tilde_dot`, `nu_tz` -> `nu_L_tilde_dot`)" |
| P2 | 85 "The base letter fixes the edge and direction" (and 136 "four letters match the paper"; 143 "The `_t` grade flips the space") | SUPERSEDED-IDEA (inside a snapshot) | No move (snapshot). Covered by P1; optional pointer to `NOTATION_HISTORY.md` §1 |
| P3 | 61, 70-71, 126, 398, 401, 560 "mode edge" (= Tucker edge), "ambient edge" (= the page's mode edge); 49, 233 "mode leg" | INCONSISTENT (terminology) | Add to P1: "This document's 'mode edge' is the page's Tucker edge, and its 'ambient edge' the page's mode edge" |
| P4 | 444-471 table and 475 "`sigmas` and `mus_dot` are both `mu_t` ... `dxis` and `xis_dot`" | INCONSISTENT | In the new notation `sigmas` = `mu_R_dot` and `mus_dot` = `mu_dot` (one side in two bases, not one variable); likewise `dxis` = `xi_U_dot`, `xis_dot` = `xi_dot`. `taus`/`nus_dot` and `detas`/`etas_dot` do coincide (`nu_L_dot`, `eta_D_dot`). Gloss in P1 or under the table |
| P5 | 51 `network_diagram.png` | FIGURE (no change) | see section 6 |

### 3.8 `WORKED_EXAMPLE.md`

| # | Location | Class | Action |
| --- | --- | --- | --- |
| W1 | 3 status: "`scripts/review/jet_steps_check.py` is the same idea with the decided names (`xi_term`, `mu_dot_step`, `_tilde`, `gamma`)" | LIVE | "... with the names of 2026-11-14 (`xi_term`, `mu_dot_step`, `_tilde`, `gamma`); the decided names are those of `SAMPLE_VARIABLES.md` (`scripts/review/sample_variables_check.py`)". Adjust if Q3 renames `jet_steps_check.py` |
| W2 | 85-139 code and table: `xi_z`, `mu_z`, `nu_z`, `eta_z`, `w_z`, `xi_t`, `mu_t`, `nu_t`, `eta_t` | DATED-RECORD (code of `probe_fit_step_example.py`) | No rename. Add to the status note the map: `w_z`, `xi_z`, `mu_z`, `nu_z`, `eta_z` -> `w_tilde`, `xi_tilde`, `mu_tilde`, `nu_tilde`, `eta_tilde`; `xi_t`, `mu_t`, `nu_t`, `eta_t` (through `V`, `R`, `U`, `H`) -> `xi_U_dot`, `mu_R_dot`, `nu_L_dot`, `eta_D_dot`. Step 2 runs on the point's own cores, where the names have no basis |
| W3 | 67, 70 comments "# mode edges, up/down" | INCONSISTENT (terminology) | Note with W2 ("mode edge" here = Tucker edge) |

### 3.9 `EVALUATOR.md` (before the review; status note)

| # | Location | Class | Action |
| --- | --- | --- | --- |
| E1 | 8 "The suffixes `_t` and `_z` are `_dot` and `_tilde`" | LIVE (status note) | Append: "and since 2026-11-15 (23:23) a dot in the frame's form carries the other basis: `xi_t`, `mu_t`, `nu_t`, `eta_t` here are `xi_U_dot`, `mu_R_dot`, `nu_L_dot`, `eta_D_dot` (`SAMPLE_VARIABLES.md`)" |
| E2 | 9 "with the new variable `gamma`" | LIVE (status note) | "with the variable `nu_L` (named `gamma` from 2026-11-14 to 2026-11-15)" |
| E3 | 120 "`nu_L[i] = S_i nu_R[i]`" | INCONSISTENT (minor) | `nu_L` here happens to agree with the new name; `nu_R` is the new bare `nu`. Gloss "(`nu_R` is `nu`)" |
| E4 | 416 Open question "Names for copies ... `nu` exists through `L` and through `R`. A naming convention is needed" | SUPERSEDED-IDEA | Gloss: "Answered for `nu`: `nu` and `nu_L` (target decision 9.6). The `Jp`-seeded copies are 3.3" |
| E5 | 30-101, 111-114, 168, 241, 261-264, 282, 288, 368, 406 `_t`/`_z`/`_tz` names; 264 "`xi_t` ... yes, at the corewise frame" | DATED-RECORD | Covered by E1. (264 mixes forms: in the manifold frame `xi_U_dot` etc.; at the corewise frame the raw `xi_dot` etc.) |
| E6 | 30, 42, 72 "mode leg" (= Tucker edge) | INCONSISTENT (terminology) | Note with E1 |

### 3.10 `API_IMPACT.md`

| # | Location | Class | Action |
| --- | --- | --- | --- |
| A1 | 119 "The tilde, hat, `d` and dot vocabulary, and the letters `sigma` and `tau` \| Four letters plus a grade suffix"; 88 `eta_z` | DATED-RECORD | No change; optional gloss "(now four letters, a basis where not the usual one, and a grade)". `sigma`, `tau` here are the library's names |

### 3.11 No change, seen

`EXECUTION_PLANS.md` (211 "the eta jet combine" is the library's kernel name; no variable names);
`T3POLYNOMIAL_SCALING_SURVEY.md`; `reviews/review_1_design.md` (D13 "`nu` through `L` against through `R`", 155
`sigma_hats`), `reviews/review_2_library_fit.md` (F4 on `_t`/`_z`, 42 and 218 on `sigma`/`tau` as the library's
canonical names): reviewer reports, never edited; `scripts/parked/README.md`.

---

## 4. The Claude doc

Doc: <https://claude.ai/artifact/PvTCvDZBKaPKqRTnntH42s>, project `b99f9eba-1b65-429e-8ace-9354c1587be8`.
Comparison method: section outline plus targeted searches (`gamma`, `_dot`, `_t`, `public`, `overtaken`, ...)
against the files; not a character diff. Tabs marked "not compared" were only checked for notation content.

### 4.1 The tabs

| Tab (order) | Tab id | Body node | Mirrors | In step? |
| --- | --- | --- | --- | --- |
| Tab 1 (a0) | `dec4b934-b5e3` | `a6460089-fed7` | `PROPOSAL.md` | Yes for notation (`mu_t` etc. as in the file). Wording differs ("in the handoff package"). Tab only: an embedded drawing of the network, node `b3dc09f3-022a`, caption "the sampled T3 network for d = 3 · bonds, sites and the messages around site 1", and a sentence "the three highlighted ones are the inputs of TT core 1" (the file uses `network_diagram.png`) |
| Target decisions (a03) | `feb9b581-ceeb` | `a85e3927-c6f2` | `TARGET_DECISIONS.md` | **No: lags by the evening.** Rev 128 stops at about 20:00 (last notes: the library read). Missing: all notes to section 6 from 21:35 ("public", the two objections, "confirmed", "The variables that still have no name", `zeta`, "The naming problem widened"); notes 3.1 and "3.2, what the notation changes"; row 9.6 and its note; "Names and the T4S paper". Differs: "Of the 55 rows ... 20 are decided" (file 56/22); "The ten rows marked to confirm" (file nine); group C still lists "3.1 letters and grades as the primitives (to confirm)". The 2.4 notes appear condensed (the file's 04:57 and 05:26 notes with `gamma` are not found). Tab only: a paragraph "The working copy ... `main` has the snapshot ... `graded-messages` ..." under the intro |
| Sample variables (a035) | `73afb9f0-92de` | `7ac6b36f-4b0a` | `SAMPLE_VARIABLES.md` | Yes (rev 1; sections 1-11 match) |
| Type catalog (a04) | `6482921c-dc7b` | `afad8585-4577` | `TYPE_CATALOG.md` | Yes (rev 20; through §14 and its written-out table; 35 `gamma` hits in both) |
| Variable diagrams (a042) | `264abe5d-db78` | `c65936d9-a667` | no file; drawings from `scripts/variable_diagrams_gen.py` | Tab has 7 rows (widgets `c8107aca-b112` mu, `ec2244ff-a8f8` nu, `fcd631c8-c9ea` nu_L, `8dbb0b13-edce` xi, `c941a1a1-3d0e` eta, `300ec000-3b68` mu_R_dot, `5afa6c82-6970` mu_tilde). The script's working tree now has 28 specs (21 rows added "2026-11-16", uncommitted). The widget of `mu_R_dot` matches the script's module exactly |
| Scaling survey (a045) | `12a4a1de-347d` | `a6b8b243-5cb6` | `T3POLYNOMIAL_SCALING_SURVEY.md` | Not compared (no notation content) |
| Design after review (a05) | `95b29a91-04e3` | `87438a19-079c` | `DESIGN_AFTER_REVIEW.md` | Yes for notation (21 `gamma`/`Gamma` and 29 `_dot` hits, as the file). Lags in the update note: lacks the third overtaken item of 19:32 ("No option for orders") |
| API impact (a1) | `643e294c-926d` | `12b13006-6726` | `API_IMPACT.md` | Not compared (no notation change) |
| Worked example (a2) | `17e0f9b0-1623` | `2f976be2-375e` | `WORKED_EXAMPLE.md` | Yes (status note identical up to "in the handoff package" and a tab mention) |
| Execution plans (a3) | `0a3ebf21-b3a7` | `8d895a76-1d6d` | `EXECUTION_PLANS.md` | Not compared (no notation content) |
| Evaluator (a4) | `3331cc9d-03a1` | `f49b0266-157c` | `EVALUATOR.md` | Yes for notation (status note with `gamma`; the `_t`/`_z` tables). Its drawings (sections "One pattern" and "The probe case, drawn") were not inspected node by node |

There is no tab for `REVIEW_DECISIONS.md`, the reviews or `README.md` (as README says).

### 4.2 Findings per tab

- **Tab 1.** As `PROPOSAL.md` P1-P4 (status-note additions; the "mode edge" terminology note). The network widget
  `b3dc09f3-022a` (FIGURE, tab only) labels `mu`, `nu`, `xi`, `eta`, `w`, `z` (plain names): no change needed.
- **Target decisions.** (DOC-1, INCONSISTENT) Bring the tab in step with the file before or with the notation
  edit (Q7): the missing notes and row 9.6 must be added, and the superseded blocks of the file (T11-T14) should
  arrive in the tab already in their deprecated/moved form. Then, as file: T2 (group B line, tab block
  `m97gdk51jj2.39302`), T6 (row 3.2, tab block `.5342`), T5 (2.9 note, tab block `.64494`, `xi_dot` -> gloss
  `xi_U_dot`). Group C's "3.1 ... (to confirm)" (tab block `.39615`) is stale against the file (LIVE).
- **Sample variables.** No change. (If S1 is accepted, mirror it.)
- **Type catalog.** As `TYPE_CATALOG.md` C0-C16. Tab blocks with the critical rows: §9 flow table `.14248`, the
  reject table `.15193` (row `gamma` where `nu` is wanted `.15562`; the `mu + mu_dot` row is in the same table),
  §11 point 2 `.21706`, §13 table `.26902`, §14 lists `.32531`, `.34296`, written-out table `.35509`.
- **Variable diagrams (tab only).** (DOC-2, LIVE) The intro paragraph `.30`: "Variable diagrams (prototype)" (H1
  `.0`), "Seven rows are drawn, to settle the form before the rest are made. The names follow the
  four-letters-and-a-basis scheme of the Type catalog, section 14. Nothing here is decided." -> the scheme is
  decided (9.6); reference `SAMPLE_VARIABLES.md` / the "Sample variables" tab; drop "prototype" and "Nothing here
  is decided" once step 2 lands. (DOC-3, LIVE) "Not drawn yet" `.1456`: "The other fifteen variables of the table
  in the Type catalog, section 14; the output `z` ...; and the blocks `V_tilde` and `H_tilde`" -> reference the
  "Sample variables" tab; the gen script now also draws `V_tilde_dot`, `H_tilde_dot`. The "How to read a row" list
  and the shapes table are consistent with the page. The seven widgets are in the new notation (FIGURE: no change).
- **Design after review.** As `DESIGN_AFTER_REVIEW.md` D0-D17 (tab blocks: `.3434` D1, `.5631` D4, heading
  `.7583` and `.7593`-`.8241` D5, code block `.10346` D6, tables `.13257`/`.13586` D8-D12, `.14450` D14, `.16073`
  D15, `.31123` D16, `.34086` D17, table `.4465` row `.4711` D2). (DOC-4, INCONSISTENT) Update note `.35141` lacks
  the 19:32 item.
- **Worked example.** As W1 (tab block `.8615`).
- **Evaluator.** As E1, E2 (tab block `.63177` list: `.63380`, `.63582`), E4. Drawings: check whether they carry
  `_t`/`_z` labels (FIGURE; expected to, as the PNGs).
- **API impact, Execution plans, Scaling survey.** No notation change.

---

## 5. Scripts

### 5.1 Import dependencies (who breaks when what is renamed)

| Script | Imports | Uses |
| --- | --- | --- |
| `review/sample_variables_check.py` | `jet_steps_check as J`; `from jet_steps_check import xi_term, z_term, mu_term, nu_term, eta_term, H_term, V_term` | `J.ones_start`, `J.zeros_start`, `J.input_jet`, `J.derivatives`, `J.seed_jet`, `J.top`, `J.truncated_product`. No step function: unaffected by step renames |
| `review/type_catalog_check.py` | `jet_steps_check as J` | `J.xi_step`, `J.mu_step`, `J.nu_step`, `J.eta_step`, `J.z_step`, `J.xi_dot_step`, `J.mu_dot_step`, `J.nu_dot_step`, `J.eta_dot_step`, `J.Jp_step`, `J.xi_tilde_step`, `J.mu_tilde_step`, `J.nu_tilde_step`, `J.eta_tilde_step`, `J.V_tilde_step`, `J.H_tilde_step`, `J.xi_tilde_dot_step`, `J.mu_tilde_dot_step`, `J.gamma_tilde_dot_step`, `J.eta_tilde_dot_step`, `J.V_tilde_dot_step`, `J.H_tilde_dot_step`, drivers, starts |
| `review/arrow_type_check.py` | `jet_steps_check as J`; `type_catalog_check as T` | `T.program` (its dict keys `'gamma'`, `'mu_dot'`, ...); monkeypatches `J.ones_start`, `J.left_to_right`, `J.right_to_left` and the seven terms; keys on `step.__name__`: `'mu_dot_step'`, `'nu_dot_step'`, `'mu_tilde_step'`, `'nu_tilde_step'`, `'mu_tilde_dot_step'`, `'gamma_tilde_dot_step'` (lines 119-133). Its start-vector iterator (115-118) depends on the ORDER in which `T.program` calls `ones_start` |
| `review/subtree_gauge_check.py` | `jet_steps_check as J` | `J.xi_step`, `J.mu_step`, `J.nu_step`, `J.eta_step`, `J.eta_term`, `J.z_term`, `J.xi_dot_step`, `J.mu_dot_step`, `J.nu_dot_step`, `J.eta_dot_step`, `J.Jp_step`, drivers, starts |
| `review/tilde_multiplier_check.py` | `jet_steps_check as J` | `J.xi_dot_step`, `J.mu_dot_step`, `J.nu_dot_step`, `J.eta_dot_step`, `J.Jp_step`, the plain and tilde steps, drivers, starts |
| `review/order_scan_check.py` | `jet_steps_check as J` | `J.mu_term` only |
| `review/kind_readout_check.py`, `review/tilde_adjoint_check.py` | `from unbundled_examples import ...` (`xi_t_site as xi_dot_step`, `mu_t_step as mu_dot_step`, ...) | record module (earlier names) |
| `review/slot_tuple_examples.py` | `graded_messages_prototype`, `plan_mockup` | records |
| `component_graphs.py`, `reuse_check.py`, `sharing_check.py`, `review/levels_sketch.py` | `graded_messages_prototype` / `plan_mockup` by path | records |
| `gn_fusion_check.py` | `probe_fit_step_example` by path | records |

So renaming any step function of `jet_steps_check.py` requires the same edit in `type_catalog_check.py`,
`subtree_gauge_check.py`, `tilde_multiplier_check.py`, and the string keys in `arrow_type_check.py`.

### 5.2 Live scripts: rename plans (subject to Q1-Q3)

Notation in the plans: `F:` the frame's form; `R:` the raw form; `RAW_NU`, `RAW_ETA` stand for the spelling chosen
in Q2 for the raw `nu_L_dot` and `eta_D_dot` when the frame's ones are in the same program
(`sample_variables_check.py` uses `raw_nu_L_dot`, `raw_eta_D_dot`).

**(a) `scripts/review/jet_steps_check.py`** (SCRIPT)

Step functions are called in both forms: `xi_dot_step(V, w)` (F) and `xi_dot_step(B_dot, w)` (R);
`mu_dot_step(..., R, U, H)` (F) and `(..., G, G, G_dot)` (R); `nu_dot_step`, `eta_dot_step` likewise. Parameter
names inside `mu_tilde_dot_step` (129-131) and `H_tilde_dot_step` (146-151) named `mu_dot`, `xi_dot` are ALREADY
the raw form: keep them.

Safe order:
1. Parameters and body of the dot steps (89-102), frame names: `mu_dot_step(mu_dot, mu, xi, xi_dot, R, U, H)` ->
   `(mu_R_dot, mu, xi, xi_U_dot, R, U, H)`; `nu_dot_step(nu_dot, nu, xi, xi_dot, L, U, H)` -> `(nu_L_dot, nu, xi,
   xi_U_dot, L, U, H)`; `eta_dot_step(mu_dot, mu, nu_dot, nu, L, R, H)` -> `(mu_R_dot, mu, nu_L_dot, nu, L, R, H)`;
   `Jp_step(eta_dot, eta, D, V)` -> `(eta_D_dot, eta, D, V)`. (Comment 88 "the variable varied in the cores along
   p = (V, H)" stays; add "called with (G, G, G_dot) and B_dot it gives the raw form".)
2. `main`, frame block (236-240): `xi_dot` -> `xi_U_dot`, `mu_dot` -> `mu_R_dot`, `nu_dot` -> `nu_L_dot`,
   `eta_dot` -> `eta_D_dot` (F, sure). Restrict the replace to these lines.
3. `main`, raw block: `gamma_dot` (250-259, 262) -> `RAW_NU`; `c_eta_dot` (252, 260) -> `RAW_ETA` (before step 5).
4. `gamma_tilde_dot` -> `nu_L_tilde_dot` (133-151, 256-262); then `gamma_tilde` -> `nu_L_tilde` (133-151, 245,
   247, 257-262); then `gamma` -> `nu_L` (85, 133-151, 217, 223, 226, 245, 247, 250-262). Longest first.
5. `eta_tilde_dot` -> `eta_D_tilde_dot` (138-144 parameters, 258, 260). All occurrences are at the base point.
6. `c_mu_dot` -> `mu_dot`, `c_xi_dot` -> `xi_dot` (248-261). Only after step 2.
7. Function names (only if Q1 says the basis is in the step name): `gamma_tilde_dot_step` -> `nu_L_tilde_dot_step`
   (or `nu_tilde_dot_step` under letter-and-grade naming); `eta_tilde_dot_step` -> `eta_D_tilde_dot_step` (or
   unchanged); `xi_dot_step`/`mu_dot_step`/`nu_dot_step`/`eta_dot_step`: unchanged under letter-and-grade naming,
   otherwise they must be split in two (frame and raw), which changes "22 step functions, 60 terms".
8. Docstring 1-7 ("tonight's names") and comments 174, 217, 244-245, 248: names of 2026-11-15.

Output: only the count line "22 step functions, 60 terms, 7 term functions" (unchanged unless step 7 splits
functions).

**(b) `scripts/review/type_catalog_check.py`** (SCRIPT; output labels change)

1. `program()` dict keys and `CATALOG` names, frame: `'xi_dot'` -> `'xi_U_dot'`, `'mu_dot'` -> `'mu_R_dot'`,
   `'nu_dot'` -> `'nu_L_dot'`, `'eta_dot'` -> `'eta_D_dot'` (44-48, 94-97).
2. Raw collisions: `'gamma_dot'` -> `RAW_NU`, `'c_eta_dot'` -> `RAW_ETA` (63-65, 72-79, 101-102).
3. `'gamma_tilde_dot'` -> `'nu_L_tilde_dot'`, `'gamma_tilde'` -> `'nu_L_tilde'`, `'gamma'` -> `'nu_L'` (40-41,
   60, 63-79, 92, 108, 111).
4. `'eta_tilde_dot'` -> `'eta_D_tilde_dot'` (72-75, 112).
5. `'c_mu_dot'` -> `'mu_dot'`, `'c_xi_dot'` -> `'xi_dot'` (61-77, 99-100). After step 1.
6. `J.*_step` names per (a) step 7.
7. Docstring line 5 "eta, nu, gamma, z are COVECTORS" -> "eta, nu (either basis), z".
Check the `GRADE` lambda (118-119) still classifies (it tests substrings `tilde_dot`, `tilde`, `dot`; `RAW_*`
names must contain `dot`). Output lines 166-169 of the recorded file change.

**(c) `scripts/review/arrow_type_check.py`** (SCRIPT; output labels change)

1. String keys of `zero_legs` and `seen` (119-133) follow (a) step 7.
2. `v['gamma']` -> `v['nu_L']` (229); `v['mu_dot']` -> `v['mu_R_dot']` (231).
3. Printed labels: "gamma used where nu is wanted: eta_term(mu, U, gamma)" -> "nu_L used where nu is wanted:
   eta_term(mu, U, nu_L)"; "a sum of mu and mu_dot: mu + mu_dot" -> "a sum of mu and mu_R_dot: mu + mu_R_dot"
   (critical: see 1.2 item 1).
4. Comment 117 "gamma : right to left through L" -> `nu_L`. Docstring "22 step functions" (23) per Q1.
Output lines 208-209 (with/against lists, derived from `T.program` keys), 220, 222 change.

**(d) `scripts/review/subtree_gauge_check.py`** (SCRIPT; output labels change)

1. `gamma_dot` -> `RAW_NU`; `zeta_dot` -> `RAW_ETA` (75-77, 84-85, docstring 12-13).
2. `gamma` -> `nu_L`; `zeta` -> `eta_D` (61-62, 74-77, 90-93, docstring 18-19).
3. `f_xi_dot` -> `xi_U_dot`, `f_mu_dot` -> `mu_R_dot`, `f_nu_dot` -> `nu_L_dot`, `f_eta_dot` -> `eta_D_dot`
   (66-70, 81-94). Bare `xi_dot`, `mu_dot` (73-74) are already raw: keep.
4. Printed labels (80-93) and docstring 9-21 ("the frame's `nu_dot` == `gamma_dot`", "the frame's `eta_dot` ==
   `zeta_dot`", "the frame's `mu_dot` (R basis) == `mu_dot` (L basis)", "`xi_dot` (D basis) == the frame's
   `xi_dot` (U basis)", "`nu` (R) == ... `gamma` (L)", "`eta` (U) == ... `zeta` (D)", "frame's tangent variables")
   -> the new names; with the new names, A's two "equal" lines become "`nu_L_dot`: frame form and raw form equal".
5. `J.*_step` names per (a) step 7.
Output lines 184-191 change.

**(e) `scripts/review/tilde_multiplier_check.py`** (SCRIPT; output labels change)

All dots are the frame's form (`V`, `R`, `U`, `H`). `xi_dot` -> `xi_U_dot`, `mu_dot` -> `mu_R_dot`, `nu_dot` ->
`nu_L_dot`, `eta_dot` -> `eta_D_dot`, in locals, the bump-name strings (`'xi_dot'` ...), the dict keys of `F`'s
return (53-63) and the loop table (73-76); docstring 4-8 ("the dot variables", the four pairings). `J.*_step` per
(a) step 7. Output lines 234-237 change.

**(f) `scripts/review/order0_assessment.py`** (SCRIPT; needs jax; output unaffected)

`gamma` -> `nu_L` (176-177); lambda parameters and locals, frame's form: `xi_dot` -> `xi_U_dot`, `mu_dot` ->
`mu_R_dot`, `nu_dot` -> `nu_L_dot`, `eta_dot` -> `eta_D_dot` (179-192). Live or record: Q3 (it is the evidence
of decided 2.7).

**(g) `scripts/review/kind_readout_check.py`** (SCRIPT or record, Q3; output unaffected)

Aliases (16) `xi_t_site as xi_dot_step, mu_t_step as mu_dot_step` and locals `xi_dot`, `mu_dot` (51-61) are the
frame's form -> `xi_U_dot`, `mu_R_dot` (aliases per Q1).

**(h) `scripts/review/sample_variables_check.py`** (already in the new names)

No rename. It establishes the `raw_` prefix for the raw `nu_L_dot`/`eta_D_dot` (Q2). Docstring 16 "imported from
jet_steps_check.py; they have no names of variables in them" stays true.

**(i) `scripts/variable_diagrams_gen.py`** (live; already in the new names; uncommitted work in progress)

V1 (LIVE): docstring 1 "(prototype of 2026-11-15, 23:02)", 6-7 "Seven rows so far: mu, nu, nu_L, xi, eta,
mu_R_dot, mu_tilde, in the names of the four-letters-and-a-basis scheme (TYPE_CATALOG.md, section 14)" -> the
row count of the file and "the notation of `SAMPLE_VARIABLES.md`". Coordinate with whoever holds the uncommitted
change. The 21 new specs were spot-checked against the page (nu_L_dot, eta_D_dot, mu_dot, H_tilde, eta_tilde):
consistent.

**(j) `scripts/plan_mockup.py`** (record, before the review)

Docstring 3-5: "new names (`_dot`, `_tilde`, `gamma`, `JtJp`) ... The explicit functions with the decided names
are in review/jet_steps_check.py" -> gloss "`gamma` (now `nu_L`)" and point to `review/sample_variables_check.py`
for the decided names. Body (`_t`/`_z`, `--dot` writer): leave.

### 5.3 Records (leave; at most a one-line docstring note)

`review/unbundled_examples.py` (earlier names, README says so), `review/tilde_adjoint_check.py`,
`review/slot_tuple_examples.py`, `review/slot_tuple_check.py` (declined form), `graded_messages_prototype.py`
(note: its docstring 127 uses `zeta` as a symbol for a reversed residual jet, unrelated to the retired letter; no
change, but an optional word to avoid confusion), `probe_fit_step_example.py`, `correspondence_check.py`,
`pairing_check.py`, `index_check.py`, `index_check2.py`, `extra_core_experiment.py`,
`frame_consistency_check.py`, `frame_consistency_nondense_check.py`, `f32_experiment.py`, `gn_fusion_check.py`,
`component_graphs.py`, `reuse_check.py` ("the frame's components": plain, fine), `sharing_check.py`,
`order_numpy.py`, `eta_axes.py`, `sweep_blocks.py`, `review/pad_reshape_check.py`, `review/levels_sketch.py`,
`review/levels_jax_check.py`, `review/merged_frontend_prototype.py`, `review/tilde_jet_pairing_check.py`,
`review/gauge_arrow_check.py`, `review/bad_scaling_example.py`, `review/adapter_doctest_draft.py`,
`review/order_scan_check.py` (uses only `J.mu_term`), `scripts/parked/*`, `reviews/review_1_scratch/*`
(`proto_tails.py` 127 has the same unrelated `zeta` symbol). The `__pycache__` folders hold stale bytecode of
`jet_steps_check`, `type_catalog_check`, `unbundled_examples`, `plan_mockup`, `graded_messages_prototype`;
harmless.

### 5.4 Recorded output affected

`scripts/review/expected_review_output_3dc15d9.txt` (DATED-RECORD; re-record after the renames, Q4):

| Lines | Script | Labels that change |
| --- | --- | --- |
| 166-169 | `type_catalog_check.py` | the four lists of variable names (`gamma`, `xi_dot`, `mu_dot`, `nu_dot`, `eta_dot`, `c_*_dot`, `gamma_*`, `eta_tilde_dot`) |
| 184-191 | `subtree_gauge_check.py` | "frame nu_dot and gamma_dot", "frame eta_dot and zeta_dot", "frame mu_dot (R) and mu_dot (L)", "xi_dot (D) and frame xi_dot (U)", "nu (R) and gamma (L)", "eta (U) and zeta (D)" |
| 208-209 | `arrow_type_check.py` | with/against lists |
| 220, 222 | `arrow_type_check.py` | the `gamma` and `mu + mu_dot` reject rows |
| 234-237 | `tilde_multiplier_check.py` | `eta_dot`, `xi_dot`, `nu_dot`, `mu_dot` |
| 11, 205 | `jet_steps_check.py`, `arrow_type_check.py` | only if step functions are split (Q1) |

The checks were re-run read-only against this repository: `jet_steps_check`, `type_catalog_check`,
`subtree_gauge_check`, `sample_variables_check`, `arrow_type_check`, `tilde_multiplier_check`,
`kind_readout_check` all reproduce the recorded output today.

Not affected: `scripts/expected_output_3dc15d9.txt`, `scripts/expected_output_89ac203f.txt` (phase-1 `_t`/`_z`
and library names: records), `scripts/timings_2core_cpu_3dc15d9.txt`.

---

## 6. Figures

| # | File | Labels | Generated by | Class / action |
| --- | --- | --- | --- | --- |
| F1 | `network_diagram.png` | `mu[1]`, `mu[2]`, `nu[1]`, `nu[2]`, `xi[1]`, `eta[1]`, `w[1]`, `z[1]`, `G[i]`, `B[i]`, `b_L`, `b_R`, "sample i", "bond 0..3" | no script in the folder (hand-made); the doc's Tab 1 has a widget version | No change (plain names, `nu[i+1]` on bond `i+1` as on the page) |
| F2 | `dependency_pattern.png` | `w`, `xi`, `mu`, `nu`, `eta`, `phi`, `z`; grades `plain`, `_t`, `_z`, `_tz`; "xi, eta: mode legs, up and down" | no script found | FIGURE: pre-review (`EVALUATOR.md`). Leave; caption note via E1. "mode legs" = Tucker edge |
| F3 | `dependency_graph.png` | `xi_z`, `mu_z`, `nu_z`, `eta_z`, `xi_t`, `mu_t`, `nu_t`, `eta_t`, `xi_tz` ... `Jp.xi_z` ..., `J p`, `K p`, `Hp` | no script found (hand-drawn from the mockup's graph) | FIGURE: leave; caption note. In the new names the tangent group is `xi_U_dot`, `mu_R_dot`, `nu_L_dot`, `eta_D_dot` |
| F4 | `dependency_graph.dot` | `xi_t`, `mu_t`, `nu_t`, `eta_t`, `xi_z`, ..., `Jp.xi_z` ... | `scripts/plan_mockup.py --dot` | FIGURE (record): leave; README row 88 may add "(pre-review names)" |
| F5 (doc) | Variable diagrams widgets (7); Tab 1 network widget `b3dc09f3-022a` | new notation / plain names | `variable_diagrams_gen.py` / by hand | No change |

---

## 7. Open questions for the supervisor

1. **Q1. Step-function names.** `SAMPLE_VARIABLES.md` §11 leaves open whether a step is named by letter and grade
   alone (`nu_step` given `L` cores returns `nu_L`). The scripts call one function in both forms
   (`mu_dot_step` with `R, U, H` and with `G, G, G_dot`). Needed before renaming `jet_steps_check.py`, its four
   dependents, and the code sketch of `DESIGN_AFTER_REVIEW.md` §4.1. Splitting the functions changes the
   recorded count "22 step functions, 60 terms" (also quoted in `DESIGN_AFTER_REVIEW.md`, `TYPE_CATALOG.md` §9,
   `arrow_type_check.py`).
2. **Q2. Two arrays, one name.** Where a program holds both forms of `nu_L_dot` or `eta_D_dot` (not the lifted
   direction, so the arrays differ), what is the second spelling? `sample_variables_check.py` uses
   `raw_nu_L_dot`, `raw_eta_D_dot`. Adopt it?
3. **Q3. Which scripts are live?** `jet_steps_check.py` is described in README as "the explicit form with the
   decided names" and is imported by five scripts. Rename it (and `type_catalog_check`, `arrow_type_check`,
   `subtree_gauge_check`, `tilde_multiplier_check`, `order0_assessment`, `kind_readout_check`), or freeze them as
   records of the 2026-11-14 names with `sample_variables_check.py` as the live program?
4. **Q4. Recorded output.** Re-record `expected_review_output_3dc15d9.txt` in place (labels only; numbers agree
   to rounding), or keep it as a record and add a new file at the current commit?
5. **Q5. `TYPE_CATALOG.md`.** Rename Claude's analysis in place in sections 1-12 (including the dated sections
   8-11), or gloss them? Move sections 13-14 whole to `NOTATION_HISTORY.md`, or keep §14's adopted parts in
   place?
6. **Q6. `DESIGN_AFTER_REVIEW.md`.** Live (rename) or a record of 2026-11-14 (notice only)? README says it wins
   over the older documents, which argues for renaming.
7. **Q7. The doc.** The "Target decisions" tab lags its file by the whole evening (21:35-23:23) and the "Design
   after review" tab lacks the 19:32 item: bring them in step as part of this edit, or notation only? Should
   `NOTATION_HISTORY.md` get a tab? The "Variable diagrams" tab and its generator are being changed now (step
   2, uncommitted): who edits that tab's text (DOC-2, DOC-3)?
8. **Q8. "Mode edge".** Older documents use "mode edge" for the page's Tucker edge and "ambient edge" for the
   page's mode edge. Notice only (recommended for the snapshots), or rewrite?
9. **Q9. Possible issues in `SAMPLE_VARIABLES.md` itself:** the seed `w_tilde` of `JtJp` lacks `[K]` in §7 (S1);
   whether to add a sentence on the arrows (S2: with the arrows exactly when the name carries no basis).
10. **Q10. Row 3.2 of `TARGET_DECISIONS.md`.** With `gamma` retired, the question "is `gamma` first-class"
    becomes "is a second basis for the right of a bond needed". The proposed rewording (T6) changes the row's
    question, not only its name: confirm.
