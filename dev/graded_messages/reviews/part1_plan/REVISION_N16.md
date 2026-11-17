# PART1_PLAN.md: revision for the maintainer's decision on N16

The decision: `TARGET_DECISIONS.md`, "The two parts", the note "The names of Part 1, and the form of the merged
sampling functions" (2026-11-16, 18:13 and 18:18). The plan was agreed by the reviewer in round 4; this revision
is to be checked by the reviewer. Counts are from an AST scan of `t3toolbox/`, `tests/`, `docs/`, `examples/` and
every doctest at `a9c31b2`, by call site.

## What changed, section by section

- **Status line.** "Final draft, agreed by the reviewer in round 4; revised on 2026-11-16 for the maintainer's
  decision on N16; this revision is to be checked by the reviewer." The sentence "the names are proposals until the
  maintainer settles the table" now says the names are settled.
- **2.2.** "The three placeholder keywords of 6.2 are proposed here as" now reads "are named".
- **2.4, "Changed".** Rewritten: the N16 rule and the one-name-per-thing list, a table "Function, today | Becomes"
  for all backend and uniform functions, and new sketches for `t3_probe(x, ww, ...)`,
  `tv_probe(frame, variations, ww, ...)`, `tv_probe_transpose(frame, ww, residual, ...)`,
  `t3_probe_corewise_hessian(x, ww, residual, core_directions, ...)`. The corewise Hessian's order is now N16's,
  and only residual-versus-seed is left to O-22. The ambient transposes stay as they are, with O-28. N30 is
  stated as decided ("as N30 has it").
- **2.5.** Rewritten: the methods follow N16 with `self` as the object; the tangent transposes stay static with the
  frame first (constructors). The renames are `residual` for `ztildes`/`c` and `ww` for `TuckerTensorTrain.apply`'s
  `vecs`, plus the sum flag. The table of today's forms now has a "Today" and a "Becomes" column. The sketches are
  `probe(self, ww, pp, order)`, static `T3Tangent.probe_transpose(frame, ww, residual, ...)` and
  `probe_corewise_transpose(self, ww, residual, ...)`. The lines "keeps `vecs`" and "no frontend caller passes it
  positionally" are removed; the section now says every call of a transpose moves (P13). N30 is stated as decided.
- **3. Table of names.** The heading is now "(settled)". The paragraph cites the note (18:13 and 18:18, `query`
  confirmed). The column "Proposed" is now "Name". The N16 row is the decided rule, citing the note.
- **5.2.** "The table of names must be settled before P1" now says the table is settled. In the overview, P12 now
  needs O-3, O-22, O-28, and P13 needs O-27, O-28.
- **5.3, P12.** Rewritten:
  - The `point_forward` sentence: of the six calls, the two of probe move (`backend/fitting.py:604`,
    `backend/uniform_fitting.py:207`); apply and entries already have the object first.
  - "What breaks" now says every positional call of a reordered function breaks, which replaces "positional calls
    do not break".
  - It adds a table of calls by function (calls, in tests, in doctests) and the list of where they are, with the
    by-reference calls in `tests/backend/test_uniform_curvature.py` given by line.
  - The two prototype scripts are pointed to O-26.
- **5.3, P13.**
  - The intro names the reorder and the renames.
  - "What breaks": every call of a frontend transpose or corewise transpose breaks. The 91 calls are listed by
    method and by file.
  - The ambient transposes and `.probe`/`.apply`/`.entries` are unchanged.
- **8 (docs).** The `docs/entries_apply_probe.md` and `docs/transposes.md` entry names their tables of call forms,
  to be put in N16's order (P18).
- **9 (dry port).** The `polynomial_predict` row notes that `probe(ww)` keeps its form. New sentence: T3Polynomial
  calls no transpose, corewise transpose or backend sampling function (searched), so N16 does not reach it.
- **11.** O-22 is restated (the order is decided by N16; only residual turned inside versus seed is open). O-28 is
  new (the ambient transposes).

Unchanged: the slice list, the order of the slices and their sizes (P12 and P13 were large and stay large; the
extra work is mechanical edits at sites the slices already visit); section 10 (it concerns removed names, not
reordered ones).

## Inventory: calls by function

Backend and uniform functions (P12). A call counts once per call site in `t3toolbox/` (the twins' bodies and the
frontend included), `tests/` and doctests.

| Function | Calls | In tests | In doctests | Breaks |
| --- | --- | --- | --- | --- |
| `t3_probe` / `ut3_probe` | 9 / 2 | 3 / 0 | 3 / 0 | every call |
| `tv_probe`, `tv_apply`, `tv_entries` | 9, 6, 6 | 3, 4, 4 | 4, 0, 0 | every call |
| `utv_probe`, `utv_apply`, `utv_entries` | 1 each | 0 | 0 | every call |
| `tv_probe_transpose`, apply, entries | 7, 6, 6 | 2, 3, 3 | 2, 0, 0 | every call |
| `utv_*_transpose` | 2 each | 1 each | 0 | every call |
| `t3_*_corewise_transpose` (probe, apply, entries) | 6, 7, 6 | 3, 4, 3 | 1 each | every call |
| `ut3_*_corewise_transpose` | 2 each | 1 each | 0 | every call |
| `t3_*_corewise_hessian` | 4, 5, 4 + by reference | 1, 3, 2 | 2, 1, 1 | every call |
| `ut3_*_corewise_hessian` | 1, 0, 0 + by reference | by reference | 1, 0, 0 | every call |
| `t3_apply`, `t3_entries`, `ut3_apply`, `ut3_entries` | 9, 3, 2, 2 | 5, 0, 0, 0 | 0 | none (renames only) |
| `t3_*_ambient_transpose` | 4 | 0 | 0 | none (O-28) |

Frontend methods (P13):

| Method | Calls | In tests | In doctests |
| --- | --- | --- | --- |
| `probe_transpose` (T3Tangent, UT3Tangent) | 23 | 15 | 8 |
| `apply_transpose` | 14 | 13 | 1 |
| `entries_transpose` | 10 | 10 | 0 |
| `probe_corewise_transpose` (TT, UTT) | 12 | 12 | 0 |
| `apply_corewise_transpose` | 17 | 14 | 3 |
| `entries_corewise_transpose` | 15 | 15 | 0 |
| ambient transposes (unchanged, O-28) | 6, 6, 7 | 5, 5, 6 | 1 each |

By file, the 91 frontend transpose calls: `tests/test_uniform_probing.py` 24, `tests/test_dispatch.py` 18,
`tests/test_tucker_tensor_train.py` 13, `tests/oracle_sweeps/o1_uniform.py` 10, `tests/test_manifold.py` 6,
`tests/oracle_sweeps/o1_ragged.py` 6, `tests/test_fitting.py` 2. Doctests: `manifold.py` 7, `uniform_manifold.py` 2,
`uniform_tucker_tensor_train.py` 2, `tucker_tensor_train.py` 1.

## Surprises

- The old kinds' `point_forward` calls `t3_probe(ww, x_cores)` and `ut3_probe(ww, ...)` (`backend/fitting.py:604`,
  `backend/uniform_fitting.py:207`). The r4 plan said they were "unchanged"; now they move. Apply and entries do
  not.
- `t3_apply`, `t3_entries` and their uniform twins already follow the rule; only their argument names change, and no
  caller passes those names by keyword. No caller anywhere passes any renamed argument by keyword (`vecs=`,
  `variation=`, `frame_data=`, `data=`, `core_pair=`, `ztildes=`, `c=`).
- `tests/backend/test_uniform_curvature.py` reaches the six corewise Hessians only by reference (a table and a
  helper, lines 595 to 608; calls at 646, 647, 656, 673, 676), so a name search misses its calls.
- `tests/test_probe_derivatives.py` and `tests/backend/test_from_sweep_reuse.py` call `t3_probe`/`tv_*` forwards
  positionally; both are deleted later (P13 and P19) but must be updated in P12.
- No page doctest and no example calls a reordered function or method. Pages name them only in prose and tables
  (P18).
- T3Polynomial calls none of them. Its `T.probe(ww)` keeps its form, and its `X.probe_derivatives` is the dry
  port's business, as before. N16 does not reach it.
- Two prototype scripts (`dev/graded_messages/scripts/extra_core_experiment.py`, `graded_messages_prototype.py`)
  call the old forms; O-26 already covers them.

## Open points

- **O-22, restated.** The order is decided by N16. Open is only whether `residual` of the corewise Hessian is a
  residual turned inside (proposed) or a seed. Needed before P12.
- **O-28, new.** The ambient transposes: do they take N16's order, `residual` and `sum_over_samples`, and does their
  default turn to `True`? Kept as they are until answered. Needed before P12 and P13.
