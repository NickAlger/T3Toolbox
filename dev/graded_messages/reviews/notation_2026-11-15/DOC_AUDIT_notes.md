# Audit of note placement: "Target decisions" tab against TARGET_DECISIONS.md (dd6be86)

- Tab: body node a85e3927-c6f2, project b99f9eba-1b65-429e-8ace-9354c1587be8.
- Revision: **176 before, 180 after.** No other tab, and no repository file except this report, was touched.
- Guards: every block delete was guarded (ifHash plus the rev of the read that showed it). Nothing was refused. No `anchors_affected`, no `allowDetach`, no suggestions.
- No email address appears anywhere.

## How the lists were taken

- **From the file:** every `## ` heading and the bold lead-in of every top-level note (`- **...**`), taken from the joined-line text so that wrapped lead-ins are whole. That is 169 entries, 154 of them notes.
- **From the tab:** an outline read gave each section's top-level blocks and how many items each list holds. Searches anchored at the start of a block gave every top-level lead-in with its list and position. Each section was then compared, entry by entry, in order.

## Differences found

### Sections that agree with the file

These sections agree with the file in membership and order:
- "Where this stands (pause of 2026-11-16, 04:46)", "Where this stood at the earlier pause", "What remains, in groups";
- 1. Scope; 3; 4 except for the one extra noted below; 7; 8; 9.

### 2. Jets

**Fixed:**
- **"2.9."** stood before "2.6, decided (2026-11-15)."; in the file it follows "2.7, decided in structure (2026-11-15).". It was moved (one `move` op) to after that note.
- **Missing, now inserted** after "2.4, the type catalog" from the file:
  - "2.4, the maintainer's arrow picture";
  - "2.4, arrows on the cores";
  - "2.4, the convention for the arrows";
  - "Parked for another arc";
  - "2.4, resumed".

**Present in the tab but not in the file, left in place:**
- **"2.4, since the catalog"**, a tab-only summary that pointed to the Type catalog tab instead of the five notes above. It was kept as tab-only condensation in phase B. It now stands after the five inserted notes and duplicates them. I suggest deleting it; I did not, because it is not a misplacement or a missing note.
- **"2.9, three forms compared"**, after "2.9, the scan reconsidered". The file removed this note in 4377301, which replaced it with "2.9, how the library does it today" and the corrected comparison. Stale in the tab; I suggest deleting it.
- **"T3Toolbox 2026.5.0 (17:10)"**, a one-line item that sat among the old 6.7 notes. It is not in any version of the file I checked. Left where it was, now among the section 2 notes.

**Lead-in or form differs (reported, not changed):**
- The file's **"2.7, assessment"** (a list item with five sub-bullets) is in the tab a plain paragraph, **"Assessment of 2.7"** (with "in the handoff package"), followed by a separate top-level list of the five points.
- **"2.5, the survey."** (tab) against **"2.5, the survey"** (file): the tab has a period inside the bold, and its body differs (it cites the tab, the file cites `T3POLYNOMIAL_SCALING_SURVEY.md`).

### 4. The frame and the value

- **Present in the tab but not in the file:** **"4.5."** ("Carried over from the present `LocalModel` without being examined."), last in the first notes list, before "4.3, after the geometry boundary". It is in no version of the file. Left in place.

### 5. Functions and layouts

- **Nesting differs (reported, not changed):** under "The same rules as the library has them" the file nests its table, four bullets (Dispatch, Masks, Bodies, Dtype), the line "What the reading bears on..." and five more bullets (5.3; 2.9 with 5.3; A question for group E; What the explicit prototypes are; One difference that matters for the target). The tab has them all, in the same order, but at top level after that note. That gives the tab nine extra top-level lead-ins in section 5, all of which are sub-bullets in the file.

### 6. The interface

**Fixed: the ten 6.7 notes.** They stood among the section 2 notes (after "2.5, decided" and before "2.3, a proposal and an example"), with older wording in places. They are now the last notes of section 6, after "6.6, the wording of the comment." and the tab's "Approved at 04:26" item, and before "## 7". Their wording is the file's.

The method was insert first, verify, then delete:
1. One insert of the ten notes from the file. Read back: the example's table has its six rows, the bracket remark follows it, "the directions `p` are 100 times too short", and every sub-bullet of "what the prototype shows" is the file's.
2. Then guarded deletes of the old blocks:
   - the three 6.7 items of the list that holds "2.5, decided";
   - the list holding "the equalizing weight, decided", "a required example" and "a correction";
   - the paragraph "6.7, the example prototyped" (with "in the handoff package"), its table and the paragraph "The numbers in brackets...";
   - the item "6.7, what the prototype shows";
   - the list holding "6.7, a word withdrawn";
   - the item "6.7, two more functions decided".
3. A search afterwards finds the "6.7, " lead-ins only in section 6, ten of them, in the file's order.

**Present in the tab but not in the file, left in place:**
- **"6.2, where each form is used."**, after "6.2, the sample and the direction, decided". The file has this content as sub-bullets of that note.
- **"Approved at 04:26"**, after "6.6, the wording of the comment.". The file folds this approval into the two notes.

**Lead-in matches, body differs:** "6.6, the wording of the comment." ends "He raised no objection to it." in the tab, and "Approved at 04:26." in the file. Not changed.

## What I did not do, and why

- **I did not delete the tab-only notes** ("2.4, since the catalog", "2.9, three forms compared", "T3Toolbox 2026.5.0", "4.5.", "6.2, where each form is used.", "Approved at 04:26"). The brief asked me to fix misplacements and missing notes and to report the rest. Two of them are now plain duplicates or stale: "2.4, since the catalog" and "2.9, three forms compared". Deleting them would be a one-call follow-up if you want it.
- **I did not re-nest the section 5 sub-bullets** or rebuild "Assessment of 2.7" as a list item. Both are structural rewrites of content that is otherwise present and in order, and the brief asked for such cases to be listed, not changed.
- **I did not change wording outside the moved 6.7 notes and the five inserted 2.4 notes.** The section 2 notes in particular keep older wording in many places (for example "in the handoff package"), as they did before.

## Claude's reading of the audit (2026-11-16)

- Fixed by the audit: the ten 6.7 notes (place and wording), the place of "2.9.", five notes that were missing
  from section 2.
- Not fixed: six notes that are in the tab and not in the file; three differences of shape; and older wording
  in notes that no pass has touched, in section 2 above all. The audit compared lead-ins and order, not the body
  of every note, so the full extent of the older wording is not known.
- Conclusion: the tab is a faithful copy of the file for everything recorded since commit `7268d54` and for the
  notes named above, and not for the older notes. The remedy is to rebuild the tab from the file, section by
  section, at a pause. Until then the file is the text to trust.
