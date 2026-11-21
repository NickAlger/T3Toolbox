# Doc sync: four tabs to the folder at 229da9d

- Source: `git diff 9f6ac57 229da9d` on the four files named below.
- Only these four tabs were edited. The one repository file written is this report.
- Every whole-block replace was guarded (ifHash plus the rev of the read). Nothing was refused. No `anchors_affected`, no `allowDetach`, no suggestions.
- No email address appears anywhere.

## 1. Target decisions (node a85e3927-c6f2): rev 162 -> 164

**At the top**
- The old heading retitled "Where this stood at the earlier pause (2026-11-15, 05:53)". Only the heading's words changed; the section's content is kept.
- The new section "Where this stands (pause of 2026-11-16, 04:46)" inserted before it, after the intro paragraph: the heading, the opening sentence, and the eight bullets with their sub-lists.

**What remains, in groups**
- The counts paragraph: 41 decided, the other 11.
- The F row: 2 / 1 / blank / 1.
- The Total row: 11 / 7 / 3 / 1.
- The F bullet.
- "The row marked to confirm (8.4) ...".

**`Jp_sumsq` to `Jp_norm_sq`** (in place, the code mark kept)
- Row 3.7.
- The 3.7 note: the table row, the `Jp_norm_sq = <Jp, omega^2 Jp>` sub-bullet, and the "Said with the question" sub-bullet.
- The `weight=` comment in the 6.2 code block.
- The two sub-bullets of "8.2, decided, the name apart".

**Rows**
- 8.1: new question text; status Decided; depends on "1.1, 6.1, 8.2".
- 8.2: the amended text (required and optional outputs).

**Notes inserted before "8.2, decided, the name apart"**
- "8.1, decided", with 10 sub-bullets.
- "8.1, the discussion", with five sub-bullets: the numbered list of three, and the three sub-items under "The fused quadratic".
- The maintainer's quotations are word for word.

## 2. Type catalog (node afad8585-4577): rev 22 -> 23
- The two occurrences in the misfit rows of the two tables renamed to `Jp_norm_sq`.

## 3. Design after review (node 87438a19-079c): rev 12 -> 13
- Six occurrences renamed, in place:
  - the variables table;
  - the "Also the direction" row;
  - the readout table;
  - the `LocalModel` bullet;
  - the `jet_steps_check.py` row;
  - the F9 row.

## 4. Notation history (node fcaee97f-f774): rev 5 -> 6
- Section 8, "An output renamed on 2026-11-16 (04:46)", appended at the end of the tab: the heading and its paragraph, from the file.

## Checks

**"Jp_sumsq" in every tab** (search of all twelve):
- Target decisions: three hits.
  - Two are inside the maintainer's quotation in "8.1, the discussion", as expected.
  - The third is in "8.1, decided": "The name `Jp_norm_sq` replaces `Jp_sumsq` (the name in every record before 04:46)". That is the file's wording, so it is correct, but the brief expected only two hits.
- Notation history: three hits, all in the new section 8, as in the file:
  - the old name at the start ("`Jp_sumsq` became `Jp_norm_sq`");
  - in the maintainer's quotation;
  - in "the log ... and quotations keep `Jp_sumsq`".
- Every other tab: none.

**Status cells of Target decisions that are not Decided, Guidance or Parked:**
- 5.3 Reopened, 5.4 Open, 5.5 Open;
- 7.2 and 7.3 Reopened; 7.5, 7.6 and 7.7 Open;
- 8.3 Open, 8.4 To confirm, 9.3 Open.

That is eleven, exactly the expected list, and it agrees with the group table: 7 open + 3 reopened + 1 to confirm. (The status legend table near the top also contains the words "To confirm", "Reopened" and "Open"; it is not a row of decisions.)

**Read-back:**
- The new top section: an 8-item list under the heading.
- The two 8.1 notes: inserted with their nested lists.
- `||Jp||=<Jp,Jp>` in the quotations: renders as text in both tabs.

## Not carried over

Nothing. The blank line in the file after "8.1, the discussion" has nothing to correspond to in the doc.

## Checked by Claude (2026-11-16, at the pause)

- The Target decisions tab was read back as its changes since revision 162 and compared with the file: the new
  top section, the counts and the group table, the renames, rows 8.1 and 8.2, and the two notes on 8.1 agree.
- The Notation history tab's new section was read back; the six and the two renames in the Design after review
  and Type catalog tabs were found by search.
- One correction made afterwards, in the file and in the tab alike: the note "8.1, the discussion" pointed to the
  decided note as "next note", but that note stands above it. Both places now say "the note above". The Target
  decisions tab is at revision 166.
