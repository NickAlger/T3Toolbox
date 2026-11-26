NOT YET

# Review of `PART1_PLAN.md`, draft r2

Reviewer: Claude, 2026-11-16. Branch `graded-messages` at `78b14dd`. The library and `tests/` are unchanged since
`4d38df7`, as the plan's status line says.

## What I checked, and how

- **The 20 findings of r1.** I checked each one in the plan's text, not in `RESPONSE_r1.md`, and again in the code
  where the finding rested on code. Table below.
- **The whole of r2, read again as a new document.** Every slice cross-reference against the new numbering P0 to
  P19:
  - section 5.1; the overview table; every slice; the table of 5.4; the test tables of section 6;
  - sections 7, 8, 9 and 10; the open points.

  All of them are consistent.
- **Green after each slice, in the new order.** For P12 to P16 I searched the tree again for the callers of what
  each slice changes or removes: the backend `geom.frame(...)`; the frontend factories; `_resolve_chunk_size`;
  `Problem` and `least_squares_problem`; the `:doc:` links to `/chunking`; the doctests of the old modules that
  stay until P19.
- **New and changed open points.** I checked O-1, O-4, O-8, O-9, O-18 and O-19 to O-25 against the rows. For O-1
  and O-19 I also checked against the two statements of "The two parts" corrected since r1
  (`TARGET_DECISIONS.md`, lines 127 to 138).
- **Code checked for r2's new statements:**
  - P4's route: `t3_operations.t3_sum`, `t3_orthogonalization`, and the cycle through `t3_linalg`.
  - `ufv_masking.ufv_variation_masks` as the uniform geometry uses it.
  - Where `stats['diagnostics']` comes from (`t3toolbox/optimizers.py:929`, the display's records).
  - The consumer's `stats` keys (`t3polynomial/fitting.py:391-395`, `:604-611`) and what `b01_backend_bench.py`
    calls.

## The findings of r1

| # | Status | In r2 |
| --- | --- | --- |
| 1 | Resolved | Section 9 replaces `_newton_fit` and gives back `probe_derivatives` at run time, and checks the `stats` keys the consumer reads. (One wrong detail about the benchmark: finding 24.) |
| 2 | Resolved | The frontend local model moves in P16, and the reason a private copy of the old model cannot work is right: the old kinds read a four-store frame. Its test list still misses a file (finding 22). |
| 3 | Resolved | P11 adds the `shard_map` class on the engine (`g`, `Kp`, orders 0 and 3, both layouts). P19 deletes the old class. Table 5.4, section 6 and section 7.6 agree. |
| 4 | Resolved | `t3_probe_corewise_hessian(residual, ww, x, core_directions, pp, order)`: today's order, no weight. Raised as O-22, but O-22 is half decided (finding 25). |
| 5 | Resolved | P13 lists the callers, removes `docs/chunking.md` with its toctree entry and its links, and removes the `newton_cg` case of `test_chunk_size_estimator.py`. Another case of that file breaks at P16 (finding 22). |
| 6 | Partly | P15 is now the geometry change on its own. The rest of the switch stays one commit, and the split with transitional code is put to the maintainer as O-24. The reason given is sound, and it is the maintainer's call, so this no longer counts against the plan. |
| 7 | Resolved | Section 2.3 step 2 names the masks: the frame's variation masks for `variations` and `Vc`, and the frame's `D` and `L` masks for `core_directions` and `Kp`. P10 and P11 take the full `_CONFIGS` matrix. P11 adds the TT-slack test and the mutation check. |
| 8 | Resolved | Section 1 is corrected. The parenthetical in O-1 is now stale (finding 23). |
| 9 | Resolved | O-19, with its memory cost in section 1. Its wording predates the correction of the decisions (finding 23). |
| 10 | Resolved | N6 and N7 follow `a_to_b`; N25 to N29 added; the history in N12 is gone. |
| 11 | Resolved | House-style sketches for the changed backend functions, `jets`, `data_preparation`, `point_objective`, one frontend method, one transpose and the frontend `newton_cg`. |
| 12 | Resolved | O-20. The sketch no longer says "raises". |
| 13 | Resolved | The seven pages are listed; the `__init__.py` re-exports go in P16; O-23 for the diagrams; arrows (9.5). |
| 14 | Resolved | P16 part 3: the stored record in the full shape, and the floor `eps |f|`, each with a test. |
| 15 | Partly | The bridge now compares only with code that is never rerouted, except one case it misses (finding 26). O-25 asks for the reading of "used once". |
| 16 | Resolved | P4 goes through `t3_operations` and `t3_orthogonalization`, keeps `max(1, ...)`, and tests a shape too large to make dense. I checked that neither module reaches `fv_operations`. |
| 17 | Resolved | Section 2.4 and P1. |
| 18 | Resolved | Section 10. |
| 19 | Resolved | Section 4, "Tails"; section 2.3 step 6. |
| 20 | Resolved | All eleven items carried out (P8, P14, section 7.5 with O-18, section 5.1, section 2.2, P16 part 2, P2, O-21, O-8, the split of P5, P0). |

Of the 20: 18 resolved, 2 partly resolved, none not resolved. Neither partial one stays open as a finding of r1;
what is left of finding 15 is finding 26.

## New findings

### 21. Should fix. P15 does not list the callers it breaks, and it names one that is not a caller (P15, lines 892 to 900)

- **What is wrong.** P15 makes the backend `frame(x)` return `(frame, centers)`. It adapts four callers and rewrites
  one test file, `tests/backend/test_geometry.py`. Every other caller that uses the result as a frame breaks.
- **Callers it does not list.**
  - In the library:
    - `backend/geometry.py:411` (`old = self.frame(x_old)`) and `:803`
      (`self.point_tangent(self.frame(x_sc))`): the geometry's own calls;
    - the doctest of `backend/sharing.py:876` (`frame = geom.frame(x)`);
    - the other calls in `backend/optimizers.py` besides `_local_model_phases`.
  - In the tests:
    - `tests/backend/test_curvature.py:2976`, `3122`, `3152`, `3175`;
    - `tests/backend/test_uniform_curvature.py:1043`, `1055`;
    - `tests/test_dispatch.py:1055`, `1140`, `1290`, `1952`;
    - `tests/backend/test_optimizers.py:577`, `700`;
    - `tests/backend/test_padded_start.py:290`, `296`, `309`;
    - `tests/backend/test_uniform_fitting.py` and `tests/backend/test_zero_reference.py`. Each has calls of the
      same pattern; I did not record their line numbers.
- **A caller that is not one.** The frontend `t3toolbox/fitting.py` is named as reading
  `frame, _ = geom.frame(x)`. It calls the frontend geometry's `frame` (`fitting.py:530`, `:625`), which returns a
  `T3Frame` and does not change (O-4).
- **What would resolve it.** List the callers above, both the library and the tests, in P15. Drop the frontend
  `fitting.py` from P15.

### 22. Should fix. Three things break at P16 that P16 does not list (P16, lines 902 to 947; section 6, lines 1000 and 1005 to 1006; table 5.4, lines 964 to 965)

- **`tests/test_manifold.py:994`.** It calls `fitting.apply_model(...)`, a factory P16 removes. P16 lists
  `test_manifold.py` nowhere; section 6 lists only its sampling parts, for P12 and P13.
- **`tests/test_chunk_size_estimator.py`, `TestFittingChunkSizeWiring.test_resolver`.** It calls
  `t3toolbox.optimizers._resolve_chunk_size` (lines 246 to 285). That goes in P16, with the frontend `chunk_size`.
  The plan removes only `test_newton_cg_exact_across_chunk_size` (P13) and keeps "its estimator cases" until P19,
  so `test_resolver` falls between the two.
- **`backend/uniform_fitting.uniform_least_squares_problem`.** It is annotated `-> bopt.Problem` (line 476). The
  module has no `from __future__ import annotations`, so the annotation is evaluated when the module is imported.
  Its doctest (lines 507 and 516) calls it.
  - P16 deletes `Problem`. Table 5.4 deletes the old kinds of `uniform_fitting.py` only in P19. Section 2.4 lists
    `uniform_least_squares_problem` as removed without saying when.
  - If it is still in the module after P16, importing `uniform_fitting` fails. The new frontend needs that module
    for `pack_sample` and `uniform_minimal`.
  - "`least_squares_problem` (both layers)" in table 5.4 does not say that it covers
    `uniform_least_squares_problem`.
- **What would resolve it.** Add `test_manifold.py:994` and `test_resolver` to P16's list. Say that
  `uniform_least_squares_problem` is deleted in P16, with its doctest.

### 23. Minor. Two open points use text the decisions have since corrected, and the slice each proposal is needed by is not stated (O-1, lines 1241 to 1246; O-19, lines 1300 to 1307)

- **O-1.** Its parenthetical says "The cost" in the decisions claims a default of `chunk_size=None`. That was
  corrected at 18:00 (`TARGET_DECISIONS.md`, lines 130 to 132). The parenthetical can go.
- **O-19.** It says "The two parts" fixes 5.3 "as the library does today", and that the plan "follows" it. The
  decisions now say that "as today" does not pick one, and mark the question open (lines 133 to 138).
- **The slices that wait.** Several slices are built on a proposal of an open point:
  - O-10 before P1;
  - O-6 before P5;
  - O-19 before P11, where the uniform map form is written;
  - O-22 before P12.

  Mark these "needed before Pn", as O-9 already is for P16.

### 24. Minor. The benchmark drives `fit_rank_continuation` (section 10 item 3, lines 1215 to 1216)

`scripts/b01_backend_bench.py` imports and calls `fit_rank_continuation` (lines 136 to 142 and 270 to 273). Only
its docstring says `_adaptive`. The port patches both functions, so nothing else changes; correct the sentence.

### 25. Minor. O-22 is decided for the transposes (lines 1315 to 1320)

- **What the rows already decide.** By 6.6 the backend function behind each frontend method has the method's
  signature. By 2.4 the frontend transposes take a residual of the type of `z` and turn it inside. So for
  `tv_probe_transpose` and the corewise transposes the convention is decided.
- **What is still open.** Only the corewise Hessian, which has no frontend method.
- **What would resolve it.** Restate O-22 as that one question.

### 26. Minor. One bridge comparison goes through rerouted code (section 5.1, lines 663 to 667)

"The old kinds" includes the plain kinds' `point_forward`. It calls `t3_probe`, `t3_apply` and `t3_entries`
(`backend/fitting.py:538`, `565`, `604`), which P12 routes onto the engine. From P12 on, exclude `point_forward`,
and the `objective` built on it, from the bridge.

### 27. Minor. Two small consistencies of section 5

- **The switching slices.** Section 5.1 says "P12 to P16 move one surface each". P14 is additive and P15 changes
  the geometry; say "P12, P13, P15, P16".
- **A gap in coverage.** P13 deletes `test_newton_cg_exact_across_chunk_size`, but the frontend `chunk_size` lives
  until P16, so for three slices a live keyword has no test. The alternative is to keep the test, feeding it
  `taylor_to_derivatives(x.probe(...))` as P13 does for the other callers, until P16.

## The open points

- **O-4, O-8, O-9, O-18.** Restated correctly: only the form of the result, or only the placement of the check, is
  open.
- **O-20, O-21, O-23, O-24, O-25.** Genuinely open, and each answerable as put. O-21 is open because 8.3's "and
  the rest" does not say. O-25 is open because 9.3 says "not a permanent test" and nothing more.
- **O-19.** Genuine; its wording needs finding 23.
- **O-22.** Half decided (finding 25).
- **None moved silently.** No silent decision of r1 was moved to a place where it is still decided:
  - zero weights, the minibatch scale, the diagrams and the uniform jetted maps each now default only
    "until answered", and say so;
  - the corewise Hessian's convention is proposed and marked.

## Form

r2 still reads as a plan. Where a reason appears, it is one sentence: P4's cycle, P16's reason for being one
commit. The response is in a separate file. The one place that reads as a reply is O-1's parenthetical
(finding 23).

## What is sound in r2 (keep it)

- **Slice numbering.** Every slice reference and the table of 5.4 are consistent with P0 to P19.
- **The switch.** P16 takes the frontend local model, its test list now includes `test_jax_absent.py` and
  `test_aligned_padded_frame_frontend.py`, and it states the order constraint O-9 puts on it.
- **P4** is now implementable as written.
- **The dry port** replaces both places where T3Polynomial meets the removed API.
- **The uniform masks** and the TT-slack test with its mutation check are correct against `testing_strategy.md`
  and the two-hole note.
- **The `shard_map` pin** on the engine is in place before the old class goes.
