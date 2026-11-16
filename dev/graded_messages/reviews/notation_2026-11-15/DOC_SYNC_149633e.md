# Doc sync: "Target decisions" tab to TARGET_DECISIONS.md at 149633e

- Tab: body node a85e3927-c6f2, project b99f9eba-1b65-429e-8ace-9354c1587be8.
- Source: `git diff 7268d54 149633e -- dev/graded_messages/TARGET_DECISIONS.md` (17 commits, 355 insertions, 42 deletions).
- Revision: **146 before, 160 after**. No other tab was touched, and no repository file other than this report.
- Every whole-block replace was guarded (ifHash plus the rev of the read that showed it). Nothing was refused. No `anchors_affected`, no `allowDetach`, no suggestions.
- No email address appears in the file or in anything sent to the tab.

## What changed

**What remains, in groups**
- The counts paragraph: 40 decided, the other 12.
- The group table:
  - B, C: Rows 0, other cells blank.
  - D: 3 / 2 / 1 / blank.
  - Total: 12 / 7 / 3 / 2.
  - A's tab-only "Held" text and the E, F, G rows are unchanged.
- The B, C, D and E bullets, replaced from the file. The D bullet now carries the file's sentence on the rules of 19:50, which the tab had left out before.
- "The two rows marked to confirm".
- "Outside the list", with forward-only probing and the squashed-tails audit.

**Rows** (question cell, status cell and, where the file changed it, "Depends on")
- 3.2, 3.4: Decided.
- 4.2, 4.3, 4.4, 4.5: Decided.
- 5.2, 5.6: Decided.
- 5.5: now Open, and "Depends on" set to 7.6.
- 6.2, 6.3, 6.4, 6.5: Decided.
- 8.2: Decided (`Oracle`).
- 9.2: Decided.
- 5.1 and 7.6 are untouched: the diff does not change them.

**Notes inserted, at the places the file has them**
- 3.2, decided: after the 3.1 note.
- 3.4, decided: before the old 3.4 note. It has five sub-bullets, including "Forward-only probing, left off this arc".
- 4.2, decided (five sub-bullets) and 4.3, the trial point, decided (with its three-policy table): before "4.1 to 4.3.".
- 4.4, decided, with six sub-bullets.
- 4.5, decided: after "4.3, after the geometry boundary".
- Section 5, in file order (5.3; then the three 5.5 notes; 5.6; 5.4; 5.1; 5.2; the rules):
  - 5.5, reversed (four sub-bullets);
  - 5.5, granularity and helpers (five sub-bullets, the last with its three nested items);
  - 5.6, decided, and 5.4, how the library does it today: after the 5.5 earlier record;
  - 5.2, decided: after the 5.1 note.
- 6.2, decided up to three names, with the signature sketch as a python code block; then 6.4 and 6.5, decided. All three go before "What 'public' means".
- 6.3, decided, with its table (empty first header cell, as in the file): before the old 6.3 note.
- 8.2, decided, the name apart: before the 8.5 note.
- 9.2, decided: before "Names and the T4S paper".

**Notes renamed** to "the earlier record", with the body unchanged as in the file: 3.4, 4.4, 5.5 and 6.3.

The maintainer's quotations were pasted from the file word for word, typos included ("prececision", "out the outer level", "work the through").

## How it was checked

- Each inserted note was read back. 6.2 (code block, three sub-bullets) and 4.3 (table and three sub-bullets) were read whole. 3.4, 5.5 and 6.3 were checked by search for their nested structure.
- A search for the old question texts of every changed row finds none.
- The status cells that are not Decided, Guidance or Parked are 5.3, 5.4, 5.5, 7.2, 7.3, 7.5, 7.6, 7.7, 8.1, 8.3, 8.4 and 9.3. That is the same list the file gives (`grep` of the row tables), and it matches the group table: 12 = 7 open + 3 reopened + 2 to confirm.

**Limit of the check:** it covers the passages the diff changed. The whole tab was not diffed line by line against the file, because the export comes back as base64, which cannot be piped into a diff.

## Not carried over

Nothing in the diff was left out. The one blank line inside the file's 8.2 note list (before the 8.5 note) has nothing to correspond to in the doc.

## Checked, and carried on to 2ee26ff (Claude, 2026-11-16, 04:2x)

- The editor's edits were read back as the tab's changes since revision 146 and compared with the file: the counts,
  the group table, the bullets, the rows and the inserted notes agree with the file at `149633e`.
- One difference of form, left as it is: the last sentence of "Claude's proposal" under 5.5 ("Checked on small
  arrays ...") sits inside the third nested bullet in the tab. The file has no blank line before it, so markdown
  reads it the same way.
- The two commits made after `149633e` (`2ee26ff`: the granularity of the contraction order moves to the
  performance discussion) were carried to the tab by Claude in three edits: the D bullet, the row 5.5, and the
  granularity sub-bullet. The tab is at revision 162 and in step with the file at `2ee26ff`.
