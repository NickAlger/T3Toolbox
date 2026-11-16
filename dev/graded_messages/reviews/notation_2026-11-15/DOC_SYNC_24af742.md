# Doc sync: "Target decisions" tab to TARGET_DECISIONS.md at 24af742

- Tab: body node a85e3927-c6f2, project b99f9eba-1b65-429e-8ace-9354c1587be8.
- Source: `git diff dcda1d5 24af742 -- dev/graded_messages/TARGET_DECISIONS.md`.
- Revision: **166 before, 170 after.** No other tab and no repository file other than this report was touched.
- Every whole-block replace was guarded (ifHash plus rev 166). Nothing was refused. No `anchors_affected`, no `allowDetach`, no suggestions.
- No email address appears anywhere.

## What changed

**"Where this stands (pause of 2026-11-16, 04:46)"**
- The "For the branch" bullet now ends with the `docs/contributor/precompute_and_caching.md` amendment (guarded replace).
- The new last bullet "Since the pause", inserted after "The Claude doc".

**"What remains, in groups"**
- The counts paragraph: 42 decided, the other 10.
- F row: Rows 1, Open blank.
- Total: 10 / 6 (Reopened 3 and To confirm 1 unchanged).
- The E bullet: the parenthesis after "7.2 the defaults".
- The F bullet.

**Row 8.3**
- The new text, status Decided, depends on "1.1, 8.1, 8.2, 7.2".

**Note "8.1, decided"**
- In the "What each method asks the oracle for" sub-bullet, the parenthesis "(corrected on 2026-11-16, 13:35: ...; note to 8.3)" added after "share the dotted sweep". This was a guarded replace of that sub-bullet, from the file.

**Three new notes before "8.2, decided, the name apart"**, in the file's order:
- "8.3, part 1, decided: the contract, and how `held` passes". It includes the python code block (`values, held = oracle.evaluate(...)`) inside the sub-bullet "The oracle's one operation ...". The block was taken from the raw file, because the line-joining tool flattens fenced code.
- "8.3, part 2, decided: a minibatch is an oracle".
- "8.3, part 3, decided: what is compiled as one program", with the five-phase table inside the sub-bullet "Decided: the phases are the units ...".

The maintainer's quotations are word for word, including the typographic apostrophes in "Let’s go with that" and "I’m getting it mixed up". The straight apostrophe in "Here I'm wondering" is also as in the file.

## Checks

- **Read-back by search at rev 170:**
  - the code block holds the eight lines of the sketch;
  - the table ends with its "stochastic step" row;
  - both typographic apostrophes are present;
  - "Since the pause" is the last bullet of the pause section.
- **Status cells that are not Decided, Guidance or Parked:**
  - 5.3 Reopened, 5.4 Open, 5.5 Open;
  - 7.2 and 7.3 Reopened; 7.5, 7.6 and 7.7 Open;
  - 8.4 To confirm, 9.3 Open.

  That is ten, exactly the expected list, and it agrees with the group table: 6 open + 3 reopened + 1 to confirm. (The status legend near the top uses the same words; it is not a row of decisions.)

## Not carried over

Nothing from the diff. The blank lines between the notes in the file have nothing to correspond to in the doc.

## Checked by Claude (2026-11-16)

The tab was read back as its changes since revision 166 and compared with the file at `24af742`: the two bullets
of the pause section, the counts and the group table, the E and F bullets, row 8.3, the parenthesis in "8.1,
decided", and the three notes on 8.3 with the code block and the table of phases. They agree. The tab is at
revision 170.
