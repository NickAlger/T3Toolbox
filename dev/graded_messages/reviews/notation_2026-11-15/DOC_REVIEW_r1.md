# Review of the Claude doc (phase B), round 1

Reviewer's report. I only read the doc: I made no update, create, batch, delete or comment. This file is the only
file I wrote. The standard is the files at `f67b1dc`. The tabs were read at these revs: Target decisions 138,
Type catalog 22, Design after review 12, Notation history 5, Sample variables 7, Evaluator 34, Tab 1 30, Worked
example 8, API impact 9, Variable diagrams 34.

## Verdict: SATISFIED

In every tab the editor changed, each notation passage reads exactly as it does in its file. I compared the text
by script, word for word, and the remaining differences are all tab-only wordings that were there before.
- The Notation history tab and the Sample variables tab match their files word for word, with the same headings,
  table rows and bold.
- No old name is left where it does not belong.
- `(D_dot, L_dot)` stands wherever the files have it.

There are no blocking findings. There are three small non-blocking points, and a list of differences outside the
notation for the supervisor.

## Blocking findings

None.

## Non-blocking findings

DN1. **Variable diagrams (the supervisor's tab), embed `m97gdk51jj2.1613`, caption "mu_R_dot · left of a bond, R
basis, with a direction".** The other frame-form captions say "with the direction" (`xi_U_dot` `.1935`,
`nu_L_dot` `.1929`, `eta_D_dot` `.1938`, `Jp` `.1942`). There is only one direction `(V, H)`, so the caption
should be "with the direction". The other 27 captions agree with the page: each gives the right side and basis,
"raw direction" exactly for the raw-form and tilde-dot rows, and "seed" exactly for the tilde rows. The text of
the tab also agrees with the page:
- "28 rows: the 22 variables, the two outputs and the four blocks", in five groups: 5 `mu`, 6 `nu`, 5 `xi`,
  6 `eta`, and 6 outputs and blocks.
- The key: the direction at the frame is "`H`, or `V` under `U`"; the raw core direction is "`L` or `D` with a
  dot"; one tilde variable per kind, summed over the modes on its side.
- The letters table.

The tab does not say "five groups" in words, but its headings show them. Wording point: it calls the library's
array `ztilde`, where the page says `ztildes`.

DN2. **Notation history, sections 4, 5 and 6.** The moved TARGET sub-bullets are top-level list items in the tab.
In the file they are indented two spaces. This is a whitespace difference, the editor disclosed it, and it reads
better in the tab. No change needed. The nesting inside section 3.2 is as in the file: the table sits inside the
first sub-bullet (checked in the export).

DN3. **Target decisions, "Where this stands".** The file adds "and 9.6 ... at 23:23" to its "Since the pause"
bullet. The tab has no such bullet, so the editor added a new one, `m97gdk51jj2.81909`, after "Held": "Since the
pause: 9.6, the notation of the sampling variables, was decided on 2026-11-15 at 23:23 (Sample variables)." It
says the same thing in the tab's own structure. Acceptable.

## Checks, by item of the brief

**1. Old names.** I searched each export for `gamma`, `zeta`, `sigma`, `tau`, `B_dot`, `G_dot`,
`c_(mu|xi|eta)_dot`, "frame's `<x>_dot`", "no name", "without a name", "no public name", "first-class", "not
decided" and `\b(mu|xi|nu|eta)_dot\b`, and read every hit in context. Every hit is legitimate:
- **Type catalog:** only the raw form's `xi_dot`, `mu_dot` and their gloss.
- **Design after review:** the update-note gloss, the 3.5 heading and parenthesis, library names (`tau_tilde`,
  `sigma_hat`, `dxi`, `sigma`, `tau`, `deta`, the two-hole `nu_dot`, `eta_dot`), and the generic step definition
  `mu_dot_step(mu_dot, ...)`.
- **Target decisions:** the group-B and row-3.2 glosses, the dated 21:44 note with its gloss, the pointer bullet,
  the 21:59 note, the 9.6 and "3.2, what the notation changes" notes, and the library's `sigma`/`tau`.
- **Evaluator:** the `nu_L` gloss and the snapshot body's `_t`/`_z` names, under its notice.
- **Tab 1:** the snapshot body's library names under the new notice.
- **Sample variables:** the page's own old-name column, section 10 and section 11.

`Execution plans`: a search for `_dot` gives 0 hits.

**2. Changed passages against the files.** Each tab was exported (markdown, or html where a markdown export was
too small to be saved). A script reduced both sides to plain words, dropping markup, code ticks and table pipes,
and compared them with `difflib`. Results:
- **Type catalog:** 10 differences. All are the chips "Sample variables" and "Notation history" in place of the
  file names, plus the tab's own "of the handoff package". Everything else is identical: the status paragraph
  "Names updated", section 2 with the `(D_dot, L_dot)` row and paragraph and the N6 paragraph, rule 1, the
  partner sentence, section 7's "30 arrays", section 8's mode-edge clause, section 9 (the flow table, the
  B1/B2 paragraph now "Each block of `g` (`V_tilde` and `H_tilde`)", the reject row `mu + mu_R_dot`), section
  11, and sections 13 and 14 as pointers.
- **Design after review:** 21 differences, all tab-only wordings that were there before ("this tab", "the tab
  Target decisions", "File names in this tab refer to the handoff package", the 10.4 list). Everything else is
  identical: the update note (all four items, including the 19:32 one), 3.1, 3.2 with both dot rows and the
  multiplier note, 3.3, 3.5, 4.1 (code, sentence, `(L, L, L_dot)` bullet), 4.2 (`core_direction` wording as fixed
  at `f67b1dc`), 4.3 (tables and pointer), 4.4, 4.5, 10.1, 10.3 and the D13 row.
- **Target decisions:** the notation passages are word for word as in the file, apart from chips. These are: the
  counts "56 rows ... 23 are decided ... The other 29", "The eight rows marked to confirm" (status "To confirm"
  appears the same number of times as in the file), group B's gloss, group C's 3.1, rows 3.1, 3.2, 3.6 and 9.6,
  the notes "3.1, decided with the notation", "3.2, what the notation changes", "3.6, decided" and "5.1, step
  functions and the basis", the four section-6 notes with the pointer bullet, the 2.9 gloss, the 21:44 glosses,
  the 21:59 gloss and the 9.6 note. The group table keeps its tab-only "0" cells. Other differences are listed
  under "For the supervisor" below.
- **Status notes:**
  - Tab 1 (`m97gdk51jj2.40001`) and Evaluator (`.65070` and the three body glosses): identical apart from chips
    and tab wordings.
  - Worked example (`.9131`) and API impact (`.14718`): read in full by a search read. Both are the file's text,
    with chips and "in the handoff package".

**3. Nothing lost or mangled.**
- **Headings:** equal in Type catalog (15), Design after review (42), Target decisions (15), Sample variables
  (16), Notation history (13) and Tab 1 (51). The Evaluator differences are the tab's own title and section names,
  which were there before (plus code comments my parser took for headings).
- **Table rows:** equal in Type catalog (71), Design after review (122), Sample variables (122) and Notation
  history (82). Target decisions has 104 against 108 because of the 2.9 table, which is a difference outside the
  notation (below).
- **Bold and code blocks:** counts equal in the changed tabs, except Target decisions (188 against 196 bold,
  from its condensed and reworded notes, which were already so).
- **Duplicates and half-replaced paragraphs:** no duplicated block, and no paragraph with old and new text side
  by side. The word diff would have shown either.
- **Tab-only content still present:** Tab 1's drawing (the export notes one widget) and its sentence on the
  highlighted inputs; Evaluator's two drawings; the "working copy" paragraph in Target decisions; the
  "handoff package" wordings.

**4. Notation history.**
- **Placement:** order `a6`, after every other tab (`a0` to `a5`).
- **Text:** the word comparison with `NOTATION_HISTORY.md` at `f67b1dc` gives 0 differences over the whole file.
  That covers the banner, the timeline (including 00:23 and 00:30), the three-period table with the
  raw-direction row, the moved passages with every maintainer quotation (21:57, 21:59, 22:11 "zeta is great!",
  22:24, 22:27), the written-out scheme table and the section 7 identifier table.
- **Structure:** headings equal (13), table rows equal (82), bold equal (53), and the banner is a blockquote.

**5. Sample variables.** The word comparison with `SAMPLE_VARIABLES.md` at `f67b1dc` gives 0 differences.
Headings (16), table rows (122), bold (39) and code blocks (1) are all equal. This covers sections 1 to 11:
section 5's step-function paragraph with `(L, L, L_dot)`, 5.2, 5.3, 5.4, section 6's table, section 7, section
10's paragraph on the raw core direction, and section 11's "Settled since ...".

**6. Variable diagrams.** See DN1. I read the captions of all 28 embeds and the whole text. I did not open the
drawings themselves.

**7. Contradictions with the page.** None found. The tabs equal files that rounds 1 and 2 checked against the
page, and the Sample variables tab is the page.

## For the supervisor: the Target decisions tab and its file differ outside the notation

These are older than this phase; the editor applied only `d8a2b62..f67b1dc`. I list them because the brief asked
for anything in a tab that is behind its file. I did not judge which of them are meant to be tab-only.
- **Row 2.3, column "Depends on":** the file has "2.2, 6.7"; the tab has "2.2, 1.1".
- **The intro to "What the assessment did not cover":** the file's sentence "The maintainer gave rules for how
  the functions are written on 2026-11-15, 19:50 (notes to section 5)." is missing from the tab.
- **The 2.9 notes:** the file has the "Counted properly" table (three forms, multiplications, traced program)
  and the struck-through "~~With the default ...~~" line. The tab has a prose note "2.9, three forms compared"
  and a "(Wrong; corrected by the maintainer at 19:50 ...)" line instead. The 2.9 lead paragraph also sits in a
  different place, and parts of the library read ("The binomial weights come from the same shift ...", the
  docstring quotation "The order loop being a real scan is what forces this") are missing or reworded in the tab.
- **The 19:50 rules note:** the tab says "his words are quoted in full in the note to 2.9" in place of the
  quotation.
- **The 2.5, 6.2 and 6.7 notes:** worded and ordered differently. The tab has the 04:26 quotation "Yes, all of
  these decisions are approved, including the two ones which I did not say."; the file does not, at that place.
- **The 2.4 notes:** condensed in the tab (known, left by design). The file's 04:57 and 05:26 glosses therefore
  have no counterpart.

The 3.6 note's `omega^2` sub-bullet and the new note "The sample stack on the outputs" (`312a068`) are not in
the tab, as expected, and not reported.

## How I checked, and what I did not

- **Exports:** `mcp__Claude_Docs__export` of each tab, decoded from the saved results to
  `/home/claude/scratch/notation/docrev/`. I compared them with
  `git show f67b1dc:dev/graded_messages/<file>` using two scripts:
  - `cmp.py` normalises both sides to words (markup, code ticks and table pipes removed; html unescaped) and
    runs `difflib` over the word sequences.
  - `struct.py` counts headings, table rows, bold and code blocks.
  The Worked example and API impact exports came back inline, too small to be saved. I read their changed
  paragraphs through `read` with a search payload. I read the Variable diagrams tab through a full `read` view.
- **Limits of the word comparison:** it ignores markup, so a code span lost or added inside a sentence would not
  show. The counts of bold, headings and table rows cover formatting only coarsely. Within table cells, column
  boundaries were not compared, only the words in order.
- **Not checked:**
  - The drawings in Tab 1, Evaluator and Variable diagrams (only their captions and the text around them).
  - The Scaling survey tab.
  - The Execution plans tab, beyond a search for `_dot`.
  - The non-notation differences in Target decisions, which I listed but did not classify.
