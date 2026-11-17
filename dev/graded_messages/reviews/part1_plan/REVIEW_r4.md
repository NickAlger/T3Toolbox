AGREE

# Review of `PART1_PLAN.md`, draft r4

Reviewer: Claude, 2026-11-16. Branch `graded-messages` at `bbea637`. The library, `tests/`, `docs/` and `examples/`
are unchanged since `4d38df7`.

No blocking or should-fix finding remains, so the plan can go to the maintainer. Three minor items remain (33 to 35
below). The writer can fix them without another review round.

## What I checked, and how

- **Findings 28 to 32**, in the plan's text.
- **The table of today's frontend forms (section 2.5).** I read it against an `ast` listing of every sampling
  method of `TuckerTensorTrain`, `T3Tangent`, `UniformTuckerTensorTrain` and `UT3Tangent`. For each method I
  checked the decorator, the positional arguments, their names and the defaults. I then checked the three sketches
  against the rule the section states.
- **The placement of `regularizer`.** I read the argument lists of today's four frontend optimizers
  (`t3toolbox/optimizers.py`) and the backend ones, and read them against 8.3 and the note "The oracle is a
  description".
- **Section 10's table of remaining references.** I wrote my own alias-aware search:
  - It maps every `import t3toolbox.backend.<m> as <alias>`, and every `from t3toolbox.backend import <m>`, for the
    modules P19 deletes from: `sampling_derivatives`, and the removed names of `probing`, `apply`, `entries`,
    `ut3_sampling`, `utv_sampling`, `fitting` and `uniform_fitting`.
  - It then lists every attribute of each alias that names a deleted object, and every `from ... import` of one.
  - It reads `t3toolbox/`, `tests/`, `examples/` and `docs/`, in code and in every module and page doctest.
  - Text greps cover scripts inside strings.
- **O-26 and O-27**, against rows 6.3, 9.3 and 9.7.
- **One more whole read**, for contradictions between sections, the slice order, and the "Needs" column.

## Findings 28 to 32

| # | Status | In r4 |
| --- | --- | --- |
| 28 | Resolved | Section 2.5 states one rule: each method keeps its kind, its positional arguments in today's order and their names, and gains `pp=None, order=0`. The three renames or changes are each tied to a row or a name (N19, N25 with the 3.7 note, 6.6). The table of today's forms matches the code method by method: `apply(vecs)`, `entries(index)`, `probe(ww)`; the instance corewise transposes `(ztildes or c, ww or index, sum_over_probes=False)`; the static tangent transposes `(ztildes or c, ww or index, frame, sum_over_probes=False)`; and the uniform classes as stated. The three sketches follow the rule, and `probe_transpose` is now static with `frame`. |
| 29 | Resolved | P13 moves the five examples' dense ground truth and waits for O-27. Section 10's table now names `examples/`. My alias-aware search finds no reference that the table and the slice lists miss (one table omission, finding 34, sits in a file P16 rewrites anyway). |
| 30 | Resolved | N16 now says what is kept, renamed, added and moved. The positional sum flags are exactly `tests/test_uniform_probing.py:832-836` and `:906-910`; my scan finds no other, frontend or backend. |
| 31 | Resolved | The definition goes from `backend/regularization.py` and the import from `backend/optimizers.py` (section 2.4; P16 part 2). |
| 32 | Resolved | Above the table of 5.2: section 3 must be settled before P1. |

## The placement of `regularizer`

- **What 8.3 writes.** 8.3 gives the form for `newton_cg` only: `newton_cg(geom, oracle, batch, x0, regularizer=None, ...)`.
- **Today's placement.** `regularizer` follows `chunk_size` in `newton_cg` and `gradient_descent`, and follows
  `use_jit` in `mc_sgd` and `adam`.
- **What the plan does.** It puts `regularizer=None` right after the positional arguments of all four, for the
  other three "with the same head" as 8.3's `newton_cg`, and says so in sections 2.4 and 2.5.
- **My judgement.** For `newton_cg` this is 8.3. For the other three it is a choice made by analogy. It is stated
  with its reason, so it is not silent, but it is not marked as a proposal (finding 33).

## O-26 and O-27

- **O-26** (the folder's prototype scripts after P19) is genuinely open. No row covers `dev/` scripts. CI runs none
  of them, so no gate depends on the answer. It is answerable as put: keep them as records against `v2026.5.0`, or
  port the ones still worth running.
- **O-27** (the jets of a dense tensor) is genuinely open:
  - 6.3 removes every `*_derivatives` function and keeps `dense_probe` "as it is";
  - 9.7's reference module computes from its definition;
  - no row says what computes the jets of a tensor that is not a T3.

  The alternatives are stated fairly, including the conflict of extending `dense_probe` with "as it is". 9.3's
  wording, "`z`, the dense tensor contracted with the inputs", supports the proposal. P13 is correctly marked as
  waiting for it.

  Note also that `tests/oracle_sweeps/oracle.py` already holds dense jet oracles (`S_probe_derivatives` and
  others). That is precedent for the alternative where the examples build the jets themselves.

## New findings

### 33. Minor. The `regularizer` placement for three optimizers is not marked as a proposal (sections 2.4 and 2.5)

8.3 fixes `regularizer=None` after the positional arguments for `newton_cg` only. Its placement in
`gradient_descent`, `mc_sgd` and `adam`, which in `mc_sgd` and `adam` puts it before `draw`, follows by analogy. Add
a row to the table of names, as N16 does for the merged functions, so that the maintainer sees it as a proposal.

### 34. Minor. Section 10's table omits `tests/test_jax_absent.py`

Its subprocess script refers to the old kind `bfit.probe_derivatives_kind(order)` (line 71), a string reference of
the kind the table says its grep adds. P16 rewrites that file, so nothing breaks. Add the row so the table is
complete as claimed.

### 35. Minor. Where O-27's proposal would land

If O-27 is answered with "the reference module's `z` also accepts a dense tensor", the change goes into
`backend/sampling_definitions.py`, the module of P2. Say in P13 that it extends that module, with its tests and its
doctest, or give P2 the dependency instead.

## The whole read

- **No contradictions between sections.** The status line, the 5.2 table, the head lines of the slices, the table
  of 5.4, the test tables of section 6, and sections 7 to 11 agree on P0 to P19.
- **The slice order still works.** Each slice's "What breaks" list is consistent with my searches in this round and
  the last.
- **The "Needs" column** now includes O-27 for P13 and O-26 for P19. Both are correct.
- **Form.** The plan reads as a plan. The responses stay in their own files.

## What is sound in r4 (keep it)

- **Section 2.5.** The rule plus a table of today's forms makes the frontend surface checkable at a glance.
- **Section 10.** The alias-aware table of remaining references, each with the slice that removes it.
- **N16** now records exactly which calls move and why.
- **Every earlier finding** (1 to 32) is resolved or handed to the maintainer as an open point.
