NOT YET

# Review of `PART1_PLAN.md`, draft r3

Reviewer: Claude, 2026-11-16. Branch `graded-messages` at `f6910c8`. The library, `tests/`, `docs/` and `examples/`
are unchanged since `4d38df7`.

## What I checked, and how

- **Findings 21 to 27**, in the plan's text. I also checked the writer's correction of `backend/sharing.py:876`.
- **The lists of what each slice breaks.** I tested them with my own searches, not the response's:
  - an `ast` scan of calls, with their keywords, over `t3toolbox/`, `tests/`, `examples/` and `docs/`;
  - the same scan over every module doctest and every page doctest (`doctest.DocTestParser`);
  - text greps for references that are not calls (by reference, in strings, in comments).
- **The names I sampled:**
  - P12: the 15 changed ragged names and the 15 uniform twins;
  - P13: the 24 removed frontend methods and the six frontend transposes;
  - P15: every `.frame(` on a backend geometry;
  - P16: `_resolve_chunk_size`, `_canonical_weight`, `_require_exact_capable`, `LeastSquaresSetup`,
    `uniform_least_squares_problem`, `_ScaledRegularizer`, `_minibatch_step_problem`;
  - P19: every file that still names `sampling_derivatives`, a `compute_*` or `assemble_*` kernel, a
    `*_from_sweep` function, or a chunk estimator.
- **Four items the writer removed from r2's lists:** `tests/oracle_sweeps/oracle.py`,
  `tests/backend/test_optimizer_display.py`, `tests/backend/test_uniform_curvature.py` (for P13), and the pages
  `rank_continuation.md` and `aligned_padded_frame.md` (for P16).
- **The signature sketches of section 2.4**, against today's argument lists, read with `ast`, of all 20 changed
  functions I sampled; and N16 against them.
- **The "Needs" column**, slice by slice, against what each slice builds.
- **One more read of the whole plan**, for contradictions between sections and for text that answers the review.
- **The AST vocabulary floor.** `test_contractions_sharding.py:90` and `test_contractions_interpreter.py:518`
  require at least 100 grouped strings. I counted what is left after P19: 84 literals today outside the deleted and
  rewritten modules, plus the 41 strings of the new engine. The floor holds.

## Findings 21 to 27

| # | Status | In r3 |
| --- | --- | --- |
| 21 | Resolved | P15 lists every caller of a backend geometry's `frame`; my own search finds no others. The writer's correction is right: `backend/sharing.py:876` calls `sg.shared_manifold(sh).frame(x)`, a frontend geometry, which does not change. The frontend `fitting.py` is dropped from the list. |
| 22 | Resolved | P16 lists `test_manifold.py:994`, `test_resolver`, and `uniform_least_squares_problem` with its doctest, and gives the reason (the annotation is evaluated at import). |
| 23 | Resolved | O-1 and O-19 match the corrected decisions. The "Needs" column and the line at the head of each slice are added. |
| 24 | Resolved | Section 10 item 3. |
| 25 | Resolved | O-22 is narrowed to the corewise Hessian. |
| 26 | Resolved | Section 5.1 excludes `point_forward` and `Problem.objective` from P12 on. |
| 27 | Resolved | Section 5.1 names P12, P13, P15 and P16. `test_newton_cg_exact_across_chunk_size` is kept until P16, with its data made through `taylor_to_derivatives`. |

## Sampling the lists

- **Complete:**
  - P12: every call with `sum_over_probes=`, every positional call with the sum flag, and the library's own call
    sites;
  - P13: the removed methods in 16 test files, the module doctests, two pages, the six examples; the frontend
    transposes in 7 test files and 4 modules' doctests; the eight `:doc:` links to `/chunking`;
  - P15: all of the backend `frame` callers;
  - P16: the private names I sampled.
- **The four removals from r2's lists were right:**
  - `oracle.py` only defines dense oracles named `S_*_derivatives`;
  - `test_uniform_curvature.py` and `test_optimizer_display.py` call `sd.t3_probe_derivatives`, a backend
    function that stays until P19, and no removed frontend method;
  - `rank_continuation.md` and `aligned_padded_frame.md` have no optimizer call in a doctest.
- **One gap:** five examples call `sampling_derivatives` directly, and no slice removes those calls (finding 29).

## New findings

### 28. Should fix. The frontend transpose sketch changes a static method into an instance method, without a row or an open point (section 2.5, lines 545 to 552)

- **What is wrong.** The sketch is `def probe_transpose(self, residual, ww, pp=None, order=0, sum_over_samples=True) -> 'T3Tangent'`,
  "at `self.frame`".
- **Today.** `T3Tangent.probe_transpose`, `apply_transpose` and `entries_transpose`, and the `UT3Tangent` twins,
  are `@staticmethod`s: `(ztildes, ww, frame, sum_over_probes)`, or `(c, ww, frame, ...)` for apply and entries
  (`t3toolbox/manifold.py`, `uniform_manifold.py`). Their callers pass a `frame`:
  - tests: `test_manifold.py:556-663`, `test_uniform_probing.py:343-387`, `test_fitting.py:287`;
  - the oracle sweeps: `o1_ragged.py:195`, `o1_uniform.py:223`;
  - the doctests of `manifold.py` and `uniform_manifold.py`.

  The old derivative transpose is static too: `(ztildes, ww, pp, frame, order, ...)`.
- **What the rows say.** 6.6's prototype covers only `TuckerTensorTrain.probe` and its corewise transpose, which are
  instance methods today. For the tangent transposes the 6.6 note says only "the same pattern ... to cover ... the
  `T3Tangent` methods" (Claude's reading). No row turns a static method that takes a frame into an instance
  method. An instance method also has no tangent to stand on: the transpose's input is a residual, not a tangent.
- **The plan contradicts itself.** N16 keeps "today's positional arguments, in today's order", and P13 lists these
  callers only for the keyword and the default, not for a change of call form.
- **What would resolve it.** Sketch `@staticmethod probe_transpose(residual, ww, frame, pp=None, order=0, sum_over_samples=True)`,
  today's form with the keywords added as N16 says. Or raise the change as an open point and add it to P13's
  "Needs".

### 29. Should fix. Five examples call `sampling_derivatives` directly, and no slice removes it (P13, lines 918 to 920; P16, lines 1014 to 1015; section 10, the list of remaining references)

- **What is wrong.** These examples import `t3toolbox.backend.sampling_derivatives as pd` and call
  `pd.dense_apply_derivatives` or `pd.dense_probe_derivatives` to make the dense ground truth:
  - `examples/fit_hilbert_from_apply_derivatives.py:65`, `:124`;
  - `examples/fit_hilbert_from_apply_derivatives_flat.py:47`, `:118`;
  - `examples/fit_hilbert_from_apply_derivatives_topt.py:32`, `:79`;
  - `examples/fit_hilbert_uniform_probe_derivatives_newton_cg.py:54`, `:92`;
  - `examples/fit_shared_factors_jetted_probes.py:61`, `:99`.
- **Why it matters.** P13 and P16 rewrite these examples only for the removed methods and the optimizers. Section
  10 lists the remaining references "outside the defining modules" as complete, but does not name `examples/`. P19
  deletes the module, so CI's examples step fails at P19. The gate's grep would find these calls, but no slice
  plans their removal.
- **What would resolve it.** Name the five in P13 or P16. Replace the dense oracle with the reference module of P2
  (`sampling_definitions`), with `derivatives_to_taylor` or `taylor_to_derivatives` as needed. Add `examples/` to
  section 10's list.

### 30. Minor. N16's reason is not quite true (section 3, N16; section 2.4)

- **"With today's names."** N16 says the positional arguments keep "today's names". But the residual argument
  changes from `ztildes` or `c` to `residual`, as section 2.4 itself says ("Two names change").
- **"No caller's positional call changes."** Inserting `pp` and `order` before `sum_over_samples` moves a sum flag
  that was passed positionally. `tests/test_uniform_probing.py:832-836` and `:906-910` pass it fourth; P12 lists
  them. Say so in N16.

### 31. Minor. `_ScaledRegularizer` is defined in `backend/regularization.py` (line 99), not in `backend/optimizers.py` (section 2.4, line 522; P16 part 2)

`backend/optimizers.py` only imports it. Delete it from `regularization.py`. Its mention in
`docs/contributor/fitting_internals.md:97` belongs to P18.

### 32. Minor. The table of names is not in the "Needs" column (section 5.2)

Small item 6 settles the names when the plan is looked over, and P1, P3, P5 and P7 to P16 create the proposed
names. Note once, above the table, that section 3 must be settled before P1.

## The "Needs" column

- **Correct.** Every slice that builds on a proposal names its open point:
  - O-10 for P1; O-14 and O-20 for P3; O-5 for P4; O-6 and O-18 for P5; O-25 for P6; O-12 and O-13 for P8;
  - O-11 for P10; O-3 and O-19 for P11; O-3 and O-22 for P12; O-7, O-8 and O-21 for P14; O-4 and O-5 for P15;
  - O-9, O-17 and O-24 for P16; O-15, O-16 and O-23 for P18; O-1 and O-2 for P19.
- **Every named dependency is real.** P4's O-5 is the weakest: P4 can build both residuals before O-5 is answered.
  The dependency does no harm.
- **Missing:** finding 28 (P13) if it becomes an open point, and the names (finding 32).

## The open points

- O-1, O-19 and O-22 are restated correctly and match the decisions as they stand.
- O-24 and O-25 are put so the maintainer can answer them.
- No open point is decided by a row.
- No earlier silent decision has reappeared elsewhere, apart from finding 28. That one was already in r2 and I
  missed it then.

## Form

r3 reads as a plan. The search notes at the head of each "What breaks" list say what was searched, which an
implementer can re-run. That is plan material, not a reply. The response stays in its own file.

## What is sound in r3 (keep it)

- **The breakage lists of P12, P13, P15 and P16** match my searches, apart from finding 29. Their search
  descriptions make them reproducible.
- **The signature sketches of section 2.4** match today's positional arguments for all 20 changed functions I
  checked. The explicit note on the differing orders (`t3_apply(x, vecs)` but `t3_probe(ww, x)`) prevents a real
  mistake.
- **The `:doc:` links.** The handling of the links to `/chunking`, including those in the docstrings of
  `sampling_derivatives` that survive to P19, is correct for `sphinx -W`.
- **`uniform_least_squares_problem`** going in P16, with its reason.
- **O-19** now names the cost of both forms.
