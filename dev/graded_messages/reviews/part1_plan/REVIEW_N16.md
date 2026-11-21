AGREE

# Review of the N16 revision of `PART1_PLAN.md`

Reviewer: Claude, 2026-11-16. Branch `graded-messages` at `c875224`. The library, `tests/`, `docs/` and `examples/`
are unchanged since `4d38df7`.

The maintainer's decision is carried out exactly and everywhere I looked. No blocking or should-fix finding
remains. Three minor items remain (36 to 38 below). The writer can fix them without another review round.

## What I checked, and how

- **The decision.** I read the note "The names of Part 1, and the form of the merged sampling functions" in
  `TARGET_DECISIONS.md`, then `REVISION_N16.md`, then the plan in full.
- **Every "Function, today" and "Becomes" entry and every sketch.** Sections 2.4 and 2.5, checked against the rule:
  - the object first; then the sample; then what is applied (`residual`, then `core_directions`);
  - then `pp=None, order=0`; then `sum_over_samples=True`;
  - one name for one thing.

  The "today" columns I checked against an `ast` listing of every public function of `probing`, `apply`,
  `entries`, `ut3_sampling` and `utv_sampling`, and every sampling method of the four frontend classes.
- **Leftover assumptions.** I searched the whole plan for any place that still assumes today's order or names. I
  checked N16, P12, P13, sections 8 and 9, O-22 and O-28.
- **Callers, by my own scans** over `t3toolbox/`, `tests/`, `examples/`, `docs/` and every module and page doctest:
  - positional calls, with keywords recorded, of the 30 changed functions and the six frontend transposes;
  - every keyword call that passes a renamed argument (`vecs`, `variation`, `variations_data`, `frame_data`, `data`,
    `core_pair`, `ztildes`, `c`), and the new names as keywords;
  - references as values: tables, lambdas, `op(*args, ...)` patterns, `getattr`, `partial`, `jit`, `vmap`;
  - functions that stay in the tree until P19 and call a changed function.
- **T3Polynomial**, read only: a grep of its package and scripts for these functions.

## The decision, carried out

- **Backend (section 2.4).** Every "Becomes" entry follows the rule and matches the note's seven example forms:
  - `t3_probe(x, ww)`, `t3_apply(x, ww)`, `t3_entries(x, index)`;
  - `tv_probe(frame, variations, ww)`, the tangent's object being frame then variations, as the note writes it;
  - `tv_*_transpose(frame, ww or index, residual)`;
  - `t3_*_corewise_transpose(x, ww or index, residual)`;
  - `t3_*_corewise_hessian(x, ww or index, residual, core_directions)`;
  - `pp=None, order=0`, then `sum_over_samples=True`, in that order everywhere.

  The "today" column matches the code for all 15 ragged and 15 uniform functions: for example `tv_probe(ww,
  variation, frame)` against `utv_probe(ww, frame_data, variations_data)`, and `ut3_apply(data, vecs)`. The four
  sketches are in the house style and follow the rule.
- **Frontend (section 2.5).**
  - The "Method, today" column matches the code, method by method.
  - The instance methods follow the rule with `self` as the object; for example `probe_corewise_transpose(ww,
    residual, ...)` and `entries_corewise_transpose(index, residual, ...)`.
  - The tangent transposes are static, with the frame first, as the note decides.
  - `apply`'s `vecs` becomes `ww` on both layouts.
  - The tangent forwards `T3Tangent.probe(ww)` and the uniform ones keep `self` as the object and gain the
    keywords.
- **Nothing still assumes today's form.** N16's row states the decided rule and cites the note. P12 and P13 now say
  that every positional call of a reordered function breaks, and list them.
- **No silent choice.** I looked for places the rule leaves open:
  - the position of `variations`, and `index` for entries: decided by the note;
  - the engine's `(kind, frame, sample, ...)` and the oracle's `(frame, batch, ...)`: the note cites them as the
    origin of the rule;
  - the ambient transposes: open, as O-28.

  The one function of the family that no table covers is `dense_probe` (finding 37). That is a gap, not a silent
  choice: the plan leaves it as 6.3 does.

## The inventory, tested

- **The counts are exact.** My scan finds 91 calls of the six frontend transposes and corewise transposes:
  `probe_transpose` 23, `apply_transpose` 14, `entries_transpose` 10, `probe_corewise_transpose` 12,
  `apply_corewise_transpose` 17, `entries_corewise_transpose` 15. By file it gives the same numbers as P13.
- **"No caller passes a renamed argument by keyword": confirmed.** My keyword scan, over code and doctests, finds
  none, for the old names or the new.
- **"No page doctest or example calls a reordered function": confirmed.** The only page doctests that call a
  transpose are those of `docs/chunking.md`, which call the derivative transpose that P13 removes with the page.
- **Functions kept until P19 that call a changed function:** the six `point_forward` methods (P12 moves the two
  probe calls), and `ut3_full_sum`, which calls `t3_apply`, whose position does not change. No `sampling_derivatives`
  function and no `*_from_sweep` function calls one.
- **T3Polynomial is not reached.** It calls none of these functions. Its `T.probe(ww)` keeps its form, and its
  `apply_transpose` calls are its own covariance class's.
- **Calls by reference.** Three sites call these functions through a table or an `op(*args, ...)` pattern, which a
  search of call names misses. The plan names only one of them (finding 36). All three files are already on the
  slices' lists.

## Slices

P12 and P13 stay reviewable as single slices: the extra work is mechanical, at call sites the slices already
visit. The suite stays green between them:

- **P12** reorders the backend and moves the frontend's call sites to the new backend order. It also moves the two
  probe `point_forward` calls.
- **The frontend between P12 and P13** keeps its own old signatures until P13, as the plan says.
- **The bridge** compares only with code that is never rerouted, so it is unaffected.
- **The old derivative kernels** call none of the reordered functions.

## O-22 and O-28

- **O-22** is narrowed correctly. N16 fixes the corewise Hessian's order. Only "residual turned inside, or seed" is
  open, and that is answerable as put.
- **O-28** is genuinely open: the note says so explicitly. It is answerable as three yes-or-no questions, with the
  call counts the maintainer needs.

## Form

The body reads as a plan. The only history is in the status line, which records the round of agreement and this
revision (finding 38).

## Findings

### 36. Minor. Two of the three call sites by reference are not named (P12, P13)

A search of call names misses these sites, because the function is held in a table or passed as a value:

- `tests/test_dispatch.py:619-626`: the three ragged and three uniform corewise Hessians, held in a table.
- `tests/backend/test_curvature.py:889-896`: the three ragged corewise Hessians, returned from a helper.
- `tests/test_uniform_probing.py:600-605`: the three `UT3Tangent` transposes, called as
  `op(*args, v.frame, sum_over_probes=True)`.

The plan names only the analogous site in `tests/backend/test_uniform_curvature.py`. The `op(*args, ...)` site in
particular must be rewritten so that the frame comes first. All three files are on the slice lists already; name
these sites beside the `test_uniform_curvature.py` lines.

### 37. Minor. `dense_probe` is not covered (section 2.4; O-28)

`dense_probe(vectors, T)` is a public sampling function in `probing` that 6.3 keeps "as it is". Its order is
sample-then-object and its sample is named `vectors`, against N16's rule and names. The note leaves it unaddressed,
as it does the ambient transposes. Add it to O-28. If O-27 is answered with its alternative of `dense_probe` taking
`pp` and `order`, its form follows from the same answer.

### 38. Minor. The status line will need updating

It says "agreed by the reviewer in round 4; revised ... ; this revision is to be checked by the reviewer". Once
this review is recorded, state the plan's status as it then stands.

## What is sound (keep it)

- **The tables.** The "Function, today" and "Becomes" table of section 2.4 and the "Method, today" and "Becomes"
  table of section 2.5 are exact, and checkable at a glance.
- **The inventory.** It is by function and by file, and its counts match an independent scan.
- **O-22 and O-28** are stated so that the maintainer can answer each with one choice.
