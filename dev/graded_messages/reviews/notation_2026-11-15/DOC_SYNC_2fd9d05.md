# Doc sync: "Target decisions" tab to TARGET_DECISIONS.md at 2fd9d05

- Tab: body node a85e3927-c6f2, project b99f9eba-1b65-429e-8ace-9354c1587be8.
- Source: `git diff 1065955 2fd9d05 -- dev/graded_messages/TARGET_DECISIONS.md`.
- Revision: **170 before, 176 after.** No other tab and no repository file other than this report was touched.
- Every whole-block replace and the one block delete were guarded (ifHash plus rev 170). Nothing was refused. No `anchors_affected`, no `allowDetach`, no suggestions.
- No email address appears anywhere.

## What changed

**"Where this stands (pause of 2026-11-16, 04:46)"**
- The "Since the pause" bullet replaced with the file's longer text.

**"What remains, in groups"**
- The counts paragraph: 43 decided, the other 9.
- F row: Rows 0, To confirm blank.
- Total: Rows 9, To confirm blank (Open 6 and Reopened 3 unchanged).
- The F bullet.
- The paragraph "The row marked to confirm (8.4) ..." deleted.

**Rows**
- 2.2, 2.3, 6.7, 8.1, 8.2 and 8.3: question cells replaced with the new texts.
- 8.4: new text; status Decided; depends on "3.7, 8.2".

**Notes**
- Section 2: "2.3, amended on 2026-11-16 (15:12) ..." inserted before "2.3, decided".
- 6.7: "6.7, amended on 2026-11-16 (15:12 and 15:44) ..." inserted before "6.7, decided".
  - In this tab the 6.7 notes have always sat with the 2.5 notes, in the date order of the discussion, not in section 6 as in the file. The new note is in that same place, directly before "6.7, decided".
- Section 8: "The oracle is a description and the data are an argument" inserted first, before "8.1, decided". It carries:
  - the python code block, taken from the raw file with its two-line `oracle.evaluate(...)` call as written there;
  - the eleven-row table "What changes in each decision";
  - every sub-bullet.
- Section 8: "8.4, the first part" and "Derivative mode moved out of the fitting layer; 8.4 decided" inserted before "8.2, decided, the name apart".

The maintainer's quotations are word for word, typos included ("commplexity", "for,e.g.", "the through" and the others as in the file).

## Checks

- **Read-back by search at rev 176:**
  - "Since the pause" ends with the `DESIGN_AFTER_REVIEW.md` / `dev/HANDOFF.md` sentence;
  - the code block shows both lines of the `evaluate` call;
  - "commplexity" is present in the 15:12 quotation;
  - "The row marked to confirm" no longer occurs.
- **Status cells that are not Decided, Guidance or Parked:**
  - 5.3 Reopened, 5.4 Open, 5.5 Open;
  - 7.2 and 7.3 Reopened; 7.5, 7.6 and 7.7 Open;
  - 9.3 Open.

  That is nine, exactly the expected list, and it agrees with the group table: 6 open + 3 reopened. (The status legend near the top uses the same words; it is not a row of decisions.)

## Not carried over

Nothing from the diff. The blank lines between notes in the file, and the empty line left in the file where the "row marked to confirm" paragraph was, have nothing to correspond to in the doc.

## Checked by Claude (2026-11-16)

- This pass was spot-checked, not read back whole: the new texts of rows 2.3 and 8.3 and two passages of the new
  notes were found by search and agree with the file. The tab is at revision 176.
- **An older difference came to light.** The notes on 6.7 stand among the notes of section 2 in the tab, not in
  section 6 as in the file, and they carry an older wording in places (for example "in the handoff package", and
  "the directions are 100 times too short" without the `p`). The passes so far carried over the differences
  between two commits of the file, so a difference older than the first of those commits was never seen. The
  editor agent has been asked to compare the order and placement of every note with the file and to bring the 6.7
  notes in step. Until a whole-tab comparison is done, "in step" is established only for what changed since
  commit `235a045`.
