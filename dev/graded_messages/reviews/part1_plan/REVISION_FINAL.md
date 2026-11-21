# PART1_PLAN.md: the final pass

This pass folds in the maintainer's answers to O-29 to O-35 and his approval of the added names (2026-11-16,
20:59). It also fixes the reviewer's findings 39 to 44 (`REVIEW_ANSWERS.md`). Nothing in the plan is open now.

- **Status line.** It says only what the document is: the implementation plan of Part 1, complete, every point
  answered and every name approved, agreed by the reviewer, awaiting the maintainer's sign-off, dated 2026-11-16,
  written against `a9c31b2` (finding 44).
- **O-29.** `check_batch` is an optional member of the protocol: in the protocol sketch, in the oracle text, in the
  frontend checks of 2.5, in P14 and in P16 part 4.
- **O-30.** `held` must be something jax can carry in a compiled program: in 2.2, the protocol sketch and P14.
- **O-31.** The geometry's consistency member is removed from 2.1, 2.4 and P15; the P15 test is now the centers'
  residual. N21's row reads "Withdrawn", with no renumbering, and the paragraph of section 3 says so.
- **O-32.** 7.1 "The compiled phases" cites rows 4.2 and 8.3 as amended at 19:30 (finding 41).
- **O-33.** The paper-to-code page takes the notation paragraph of `CLAUDE.md` and the probing map of
  `entries_apply_probe.md` (section 8; "proposed" removed).
- **O-34.** New: `ut3_unscale_coordinates` in `backend/data_preparation.py`, and
  `UniformTuckerTensorTrain.unscale_coordinates`. They appear in 2.1, the 2.4 table and sketch, 2.5, P3 (with a
  test against the ragged one and garbage padding), 7.5 and 11.3.
- **O-35.** `scales` is one per coordinate, stated in 2.4 with the prototype as the source; nothing asks about it
  any more (finding 42).
- **Finding 39.** `'scan'` uses the `xscan` of `get_backend(True, use_jax)`: a real `lax.scan` on jax operands and
  a loop on numpy, with no reference to the layout.
- **Finding 40.** P13 corrected:
  - five library calls of `dense_*_derivatives`, each inside a local helper;
  - three examples' helpers carry the removed name `dense_apply_derivatives`, at lines 116, 110 and 74;
  - the helpers are replaced by `dense_apply` and `dense_probe`, which take `W` themselves, so the loops go;
  - the seven local `dense_apply` and `dense_probe` helpers may stay.
- **Finding 43.** P16's translation reads the old positional arguments as the old signature names them, with the
  `topt.adam(...)` page call as the example.
- **Finding 44.** 7.5 quotes only the sentence that carries the rule, with its citation.
- **5.2.** The "Needs" column is gone, and the paragraph says nothing is open. P3 and P15 have updated content.
- **Section 11.** 11.1 has a line for each of O-1 to O-35, with "as proposed" and "the plan's reading" dropped.
  11.2 says nothing is open. 11.3 records the added names, all approved at 20:59, plus the uniform twin of O-34.
- **New section 12, "How this plan is carried out."**
  - The maintainer's sign-off comes first.
  - The implementer and the reviewer go one slice at a time until the reviewer signs off.
  - Claude looks over and commits each slice before the next one starts.
  - What a slice hands over: the code, its tests, the gate's results, and a note of any departure from the plan.
  - Major issues go to the maintainer; a departure from a decided row, a public name or the slice order is major.
    Minor issues are Claude's to decide and record.
- **Consistency read.** No "open point", "proposed", "until answered" or "the plan's reading" is left for anything
  decided. "Awaiting" now appears only in the status line, for the sign-off. Slice numbers are P0 to P20
  throughout.

The plan is 1,891 lines and has 21 slices.
