# Doc edit report (phase B): the Claude doc brought in step with the folder at `9b89f0d`

Editor's running record. Doc: <https://claude.ai/artifact/PvTCvDZBKaPKqRTnntH42s>, project
`b99f9eba-1b65-429e-8ace-9354c1587be8`. All edits are written directly, not as suggestions. Every whole-block
replace or delete is guarded with `ifHash` and `ifRev`. No `allowDetach` and no `force` were used. The markdown
sent comes from the files at `9b89f0d`, with wrapped lines joined
(`/home/claude/scratch/notation/tools/unwrap.py`). A link to another file of the folder becomes a mention chip
for its tab, as the tabs already do (for example `Design after review`).

## Notation history (new tab)

- **Tab and body.** The tab is `a69c51c5-e4aa` and its body `fcaee97f-f774`, placed with order `a6`. No other
  tab's name or order was changed.
- **How it was built.** Five calls took it from rev 1 to rev 5:
  - The birth carried the title, the banner, the contents and section 1 (rev 1).
  - Four `insert`s at the root's end, in file order: sections 2 and 3 (rev 2), sections 4 and 5 (rev 3),
    section 6 (rev 4), and section 7 (rev 5).
- **Content.** It is `NOTATION_HISTORY.md` at `9b89f0d`, whole. One change of whitespace: the moved TARGET
  sub-bullets that open sections 4, 5 and 6 are sent as top-level list items (the file indents them two spaces,
  under a parent bullet that is in section 3.2). Within section 3.2 the nesting is as in the file: the head
  bullet, its three sub-bullets, and the table inside the first sub-bullet (checked by a `view` of list
  `m97gdk51jj2.9217`).
- **Check.** I ran a search for numbered headings. The outline is 1, 2, 3.5 (h4), 3, 3.1, 3.2, 4, 5, 13 (h3), 6,
  14 (h3), 7, the same as the file's headings.
- **The links into this tab.** A mention chip `file/a69c51c5-e4aa` is used in the status notes of the other
  tabs, where the files link `NOTATION_HISTORY.md`.

## API impact (body `12b13006-6726`)

- Rev 8 -> 9. Inserted one list item, `m97gdk51jj2.14718`, after `.13849` ("The jet normalization"): the file's
  bullet "The names, since 2026-11-15 (23:23)". Its links are chips to Sample variables and Notation history.
- Nothing replaced or deleted.

## Worked example (body `2f976be2-375e`)

- Rev 7 -> 8. Replaced the status paragraph `m97gdk51jj2.8615` (guarded `9b72394a` @ 7); it is now `.9131`. The
  new text is the file's, with the tab's own wording kept: "in the handoff package", and "See the tab"
  [Design after review]. The link to the page is a chip.

## Evaluator (body `f49b0266-157c`)

- Rev 33 -> 34. Four guarded paragraph replaces, all to the file's text:
  - `.63380`, "The names" (now `.65070`; chips to Sample variables and Notation history). The tab's own words
    "See section 3 of that tab" are kept.
  - `.63583`, "The `centers` argument ...": the gloss on `gamma`.
  - `.52301`, "The two `nu` differ ...": "(`nu_R` is `nu`)".
  - `.54700`, "Names for copies": "(Answered for `nu` ...)".
- The editor applied these as in-place differences, so the ids of the last three paragraphs are kept.
- The drawings in sections "One pattern" and "The probe case, drawn" were not opened or changed (tab-only
  figures). The status note now says that the figures use the names from before the review.

## Tab 1 (body `a6460089-fed7`)

- Rev 29 -> 30. Inserted one list item, `m97gdk51jj2.40001`, after `.38459` ("The names."): the file's bullet "The
  names, since 2026-11-15 (23:23)", with chips. Nothing replaced.
- The tab-only drawing and its sentence, and the "handoff package" wording, are left as they are.

## Design after review (body `87438a19-079c`)

- **Revs.** Rev 10 -> 12. The file diff `ee23f44^`..`9b89f0d` has 15 hunks; all are applied.
- **Rev 11: one call, all guarded @ 10, every guard accepted.**
  - Update note `.35141`: now `.35791`, with all four items (the 19:32 item was missing from the tab). The tab's
    wording "this tab" and the chip to Target decisions are kept.
  - The 3.1 bullet `.3434`.
  - The 3.2 multiplier table, row `.4711`: its two cells `.4713` and `.4746`.
  - A new paragraph `.37108` after table `.4465`: the multiplier note (N5).
  - The 3.3 table, cell `.5632`.
  - The 3.5 heading `.7583` (now `.37343`) and its paragraph `.7593` (now `.37384`), with a chip to Notation
    history. Its bullets `.7887`, `.7991`, `.8141`. The bullet `.8240` ("The name is free ...") is deleted.
  - The 4.1 code block `.10346`: now `.37860`, plus a new paragraph `.38443` ("A step is named by letter and
    grade ...").
  - The 4.1 bullet `.11033`; the 4.2 paragraph `.11675`.
  - The 4.3 tables: cells `.13322`, `.13391`, `.13630`, `.13899`. Paragraph `.14103` is now the pointer `.38593`;
    the list `.14134` (the three "variables with no public name") is deleted.
  - The 4.4 bullet `.14451`; the 4.5 cell `.16113`; the 10.1 item `.31124`; the 10.3 item `.32213`; the D13 cell
    `.34087`.
- **Rev 12.**
  - The `Jp` cell `.13818` (guarded @ 11): `eta_D_dot`.
  - A new table row `.39692` after `.4711`: the raw-form row of 3.2.
- **Note.** The editor applied most paragraph replaces as in-place differences: the ids are kept, and the
  "dropped" notices list marks re-sent unchanged.
- **Check.** A search of the tab for `gamma`, `zeta`, `B_dot`, `G_dot`, `\beta_dot\b`, `\bnu_dot\b` and "no name",
  after the edits:
  - `gamma` appears only in the three new glosses: the update note, the 3.5 heading, and the 3.5 parenthesis.
  - The rest appear only in the new raw-form row, where `nu_dot`, `eta_dot` are the library's two-hole names.

## Type catalog (body `afad8585-4577`)

- **Revs.** Rev 20 -> 22. The file diff `6e0910b`..`9b89f0d` is applied in full.
- **Rev 21: one call, all guarded @ 20, every guard accepted.**
  - Inserted the status paragraph "Names updated" (`.38116`) after `.577`.
  - Section 2, the table: the cells `.1755`, `.1882`, `.1915`, `.1948`, `.1983`, `.2230`, `.2426`, `.2481`, and
    the `Kp` row's cell `.2635`.
  - Section 2, the paragraph `.2680`: now `.38279`, plus the new N6 paragraph `.38631`.
  - Section 2: rule 1 `.2984`, and "Why, and where the reversal comes from" `.3260` (now `.38865`).
  - Section 7, Part A `.7937`; section 8, first bullet `.10010`.
  - Section 9: the flow-table cells `.14287` and `.14401`, the paragraph `.14478` (B1 and B2), the paragraph
    `.14861`, and the reject-table cells `.15564` and `.15705` (`mu + mu_R_dot`).
  - Section 11: point 2 `.21707` and the base-point bullet `.23991`.
- **Rev 22: sections 13 and 14.**
  - Deleted the bodies of sections 13 and 14: 25 blocks, guarded by `ifRev` 20 (several ids), from `.26155`
    through `.37063`, including the written-out table `.35509`.
  - Kept the headings `.26078` and `.29892`. After each, inserted the file's pointer (`.40929` and `.41135`), with
    chips to Notation history and Sample variables.
- **Tab-only content left.** The status note's wording "of the handoff package", and the tab's own word
  differences in sections 10 and 11.
- **Check.**
  - A search of the tab for `gamma`, `zeta`, `B_dot`, `G_dot`, "frame's `", "no name", `\b(nu|eta)_dot\b`: 0
    hits.
  - The outline from section 12 on is: 12, 13 (heading + pointer), 14 (heading + pointer), as in the file.

## Target decisions (body `a85e3927-c6f2`)

Rev 128 -> 138. Applied: the file diff `ee23f44`..`9b89f0d` (11 hunks, including your 3.6 decision of 01:04,
commit `5ca2030`). The tab already held the content of `ee23f44` itself: the 20:00 library-patterns table is
`.77626`.

- **Top** (rev 128 -> 134):
  - **"Where this stands".** The tab has no "Since the pause" bullet; it marks 2.4 as decided inside the "Held"
    bullet instead. I inserted a new bullet `.81909` after "Held": "Since the pause: 9.6 ... decided on
    2026-11-15 at 23:23 ([Sample variables])". This adapts the file's sentence to the tab's own structure.
  - **The counts paragraph `.38200`.** Edited by quoted words to "56 rows ... 23 are decided ... The other 29".
  - **The group table.** Guarded cell replaces: A rows 3 -> 2 (`.38534`) and to confirm 3 -> 2 (`.38542`); C
    rows 8 -> 7 (`.38634`) and to confirm 2 -> 1 (`.38643`); the total rows 31 -> 29 (`.38836`) and to confirm
    10 -> 8 (`.38847`). The tab-only cells "0" and "0 (2.4 decided at 16:10)" in row A are kept.
  - **The groups list, by quoted words.** A: "(3.6, the weight inside the engine, was decided on 2026-11-16 at
    01:04.)", in the tab's own parenthesis style, with the 3.6 clause removed. B: the gloss "(now `nu_L`)". C:
    3.1 moved to "(Decided on 2026-11-15 at 23:23 ...)".
  - **"The ten rows" -> "The eight rows"** (by quoted words; the bold is kept).
- **Section 2** (rev 135): the 2.9 note `.64495` (guarded) has the gloss on `xi_dot`.
  - The file's other two 2.4 glosses (the notes of 04:57 and 05:26) have no place in the tab. The tab's notes to
    2.4 are condensed, and those two notes are not in it (a search for `gamma` and "flow against" found
    nothing). Left as they are.
- **Section 3** (rev 135):
  - The row cells (all guarded): 3.1 question `.5209`, status `.5329` -> Decided, depends `.5341` -> 9.6; 3.2
    `.5349` (gloss only, per Q10); 3.6 question `.5680` -> the decided text, status `.5769` -> Decided.
  - Inserted the notes "3.1, decided with the notation" and "3.2, what the notation changes" before `.5875`
    (`.82350`, `.82652`).
  - Inserted "3.6, decided" with its six sub-bullets (`.82996`) before `.6424`.
  - The paragraph `.6425` is now "3.6, the earlier record. ...".
- **Section 5** (rev 136): inserted the note "5.1, step functions and the basis" (list `.85862`) after the notes
  list `.8601` (5.3, 5.5), before the 19:50 rules, as in the file.
- **Section 6** (rev 137): inserted, before the 6.3 note `.9611`, the file's four new notes as list `.86576`:
  "What public means" (21:35), "the maintainer's two objections" (21:44), "Public, two points confirmed" (21:57),
  and the pointer bullet that replaced the moved naming notes (with chips to Notation history and Sample
  variables).
- **Section 9** (rev 138):
  - Inserted the row 9.6 (`.94458`) after `.34361` (9.5). The cell's reference is in code, `SAMPLE_VARIABLES.md`,
    as in the file.
  - Inserted the notes "Names and the T4S paper" and "9.6, decided" (list `.94776`) before the 9.4 note
    `.21237`.
- **Tab-only content left.** The paragraph "The working copy ..." (`.59538`); the wording "in the handoff
  package"; the condensed 2.4 notes; the parenthesis style of group A.
- **Check.**
  - A search for `gamma`, `zeta`, `B_dot`, `G_dot`, "frame's `", "no name", "without a name": every hit is a gloss
    (B, row 3.2, "3.2, what the notation changes"), the pointer bullet, the dated 21:44 note with its gloss, or
    the 9.6 note.
  - The rows with status "To confirm" are 3.2, 3.3, 3.7, 4.5, 5.5, 6.4, 8.1 and 8.4: eight, as the paragraph and
    the group table say.
  - The section headings (0 to 9, "After the target") are unchanged.

## Sample variables (tab 73afb9f0-92de; verify only)

- Rev 7 before and after: no edit made.
- Checked: exported the tab as markdown (rev 7, 30,527 bytes) and compared it with `SAMPLE_VARIABLES.md` at 9b89f0d with the wrapped lines joined (30,528 bytes; the one extra byte is the file's final blank line). The sizes agree to that byte. Spot checks of the decoded export:
  - the §10 paragraph "**The raw core direction** was `(B_dot, G_dot)` in this folder until 2026-11-16 (00:30) ..." (commit ed574ac), word for word;
  - the §11 "Settled since this page was first written ..." sentence, and the §5 step-function paragraph with the 00:23 quotation.
- This was not a line-by-line diff. The export arrives as base64 and cannot be piped into a shell, so the check rests on the byte count plus these spot checks. A difference of equal length somewhere else would not have been caught.
- Blocks replaced, inserted or deleted: none. Nothing was refused.

## Not done, by design

- Variable diagrams, Execution plans, Scaling survey: not touched, per the brief.
- Evaluator: the drawings were not inspected.
- Target decisions: the 04:57/05:26 glosses had no place in the tab's condensed notes, so they were not added.
- No guard was refused, and no `anchors_affected` came up in any tab.

## Follow-up: the decisions after 9b89f0d (files at 24e5174)

**Target decisions** (a85e3927-c6f2), rev 138 -> 145.
- Counts paragraph .38200: 25 decided, the other 27 (by quoted words).
- Group table: A Rows 2 -> 0 (.38534), A To confirm 2 -> blank (char deleted), Total 29 -> 27 (.38836), To confirm 8 -> 6 (.38847). The tab-only "Held" texts in the table were left alone.
- "The six rows marked to confirm" (.41263, by quoted words).
- Group-A bullet .38855: the 3.7/3.3 tail replaced in the tab's own condensed style, "(3.7, what happens to the sample stack on each output, was decided at 01:32; 3.3, which adjoint sweep a tilde name refers to, at 01:36.) Group A is finished." This is not the file's wording.
- Row 3.3: question cell .5467, status .5523 Decided, depends .5535 "6.5". Row 3.7: question cell .5792, status .5851 Decided.
- Row 2.3 depends cell .3367: "2.2, 1.1" -> "2.2, 6.7".
- 3.6 note, "So:" sub-bullet .84514 replaced (omega / omega^2, and the note on `W`).
- Inserted after the 3.6-decided item .82996: "3.3, decided" (.97512), "3.7, decided" with its three-row table (.98726), and "The sample stack on the outputs" (.100169). The maintainer's quotations were pasted from the file word for word.
- All whole-block replaces were guarded (ifHash + ifRev 138). Nothing was refused.
- Checked by search and view at rev 145:
  - "23 are", "2.2, 1.1" and "seeded tilde copies" no longer occur;
  - the new items, the table and the 3.6 sub-bullet read as the file does;
  - the order matches the file (3.6 decided, 3.3, 3.7, sample stack, 3.6 earlier record).
- Still behind the file, left alone:
  - The §6 note "The seeds (3.3)" (.91378) still says "which seeded copies are public". That text was not in this diff, so it was left alone.

**Sample variables** (7ac6b36f-4b0a), rev 7 -> 11.
- §5.3 paragraph .11941: the closing two sentences ("Target decision 3.3 is only about names ...") deleted by quoted words. The new paragraph "The tilde names always mean the residual's application ..." was inserted after it (.38855).
- §11 "Settled since ..." .36962: extended by quoted words.
- §11 bullet .38239 ("Which application of the transpose ...") deleted (guarded, ifHash cc0b5cc2, ifRev 7).
- Checked at rev 11: §5.3 now ends "... linear in its seed." and is followed by the new paragraph. §11 reads as the file does, with three bullets left. Nothing was refused.
