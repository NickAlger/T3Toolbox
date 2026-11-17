# Log of the implementation of Part 1

Kept by Claude, who looks over each slice after the reviewer signs off and commits it. One entry per slice: the
rounds, what Claude checked, the minor issues Claude decided, and what went to the maintainer.

## P0. Baseline (2026-11-16)

- **Rounds:** two. Round 1: not yet, four should-fix findings, none blocking (two in the gate script's tracking of
  runs, one in the inventory script, one a gap in the plan). Round 2: sign off, two minor findings.
- **Result:** the unchanged tree passes every step of the gate in this workspace (Python 3.13, numpy 2.5, jax
  0.12; 2 cores, 8 GB): 1,407 passed, 5 skipped (the slow tier), 49,058 subtests. Two independent full runs agree.
  A whole gate takes about 27 minutes with two workers. Details in `P0_BASELINE.md`.
- **Claude checked:** the reviewer planted a failure of each kind twice (before and after the script was
  reworked) and each step failed on it; Claude read the script's header and option parsing, ran one step, and
  confirmed the tree holds no change outside `dev/graded_messages/`.
- **Minor issues decided by Claude:**
  - The root `README.md` holds the old optimizer call form and no slice named it. It goes into P17, and the
    inventory searches it. `CHANGELOG.md` and `docs/release_notes.md` stay out of the search on purpose.
  - Three counts in the plan were corrected from the inventory (seven factory calls in the doctests of
    `t3toolbox/fitting.py`; two links to `chunking.md` in the glossary; four functions call `_wchunked_sum`).
  - Round 2's finding 9 (an option without its value made the script loop without end) Claude fixed directly.
    Finding 10 (a descendant that ignores SIGTERM can outlive a step stopped at the time limit) is recorded as a
    known limit in `P0_BASELINE.md`, not fixed.
- **To know:** the machine restarted once under the first suite run, cause unknown, not memory. The gate script
  detects a run that vanished (`wait` exits 2); start it again.
- **To the maintainer:** nothing major.

## P1. `jet_contract` and `GroupLengths` (2026-11-17)

- **Rounds:** two, and a small third pass. Round 1: not yet, one should-fix finding in a test, eight minor; no
  defect in the code. Round 2: sign off, two minor findings in the tests, fixed without another round.
- **Result:** `GroupLengths(T, W, K, C)` and `jet_contract(subscripts, *operands, lens, over_orders)` in
  `backend/contractions.py`, additions only (`contract` untouched). 36 more tests than the baseline. The whole
  gate green on the tree of round 1's fixes (1,442 passed, 5 skipped, 1 expected failure, 60,982 subtests; sphinx
  clean); after that only `tests/test_jet_contract.py` and one comment changed, checked by the slice's tests and
  the module doctests.
- **Checked independently by the reviewer:** the mathematics by two methods of its own (polynomial interpolation,
  then roots of unity and an inverse FFT; 5,214 cases, largest relative error 5.9e-15); 66 mutations of the code
  over the two rounds, every real one caught in the end.
- **Claude checked:** read the code and the docstring; the diff of `contractions.py` removes no line; nothing
  treats a site axis as a group; the Part 2 markers are in place; ran the slice's tests and the module doctests.
  Claude reworded one signature comment (`over_orders`).
- **Went to the maintainer** (all recorded in `TARGET_DECISIONS.md` and the log, entries 111 to 114):
  - The order of the contractions inside one step is `contract`'s default, which on numpy does the expensive
    contraction on every slot for 16 of the 20 product strings. Deferred to Part 2 (row 5.5), marked by comments
    `PART 2 (row 5.5)` and an expected-failure test.
  - The site group `S`, which Claude had put into row 5.4 without raising it, is removed. A rule: new groups only
    after deliberation. The affine group `A` is kept after deliberation, with a definition.
  - Jets in several variables noted as a future feature.
- **Minor issues decided by Claude:**
  - A readout that takes a third full jet one coefficient at a time is a real scan under `'scan'` (the library's
    scheme at 2026.5.0, `_tt_det_lag_step`), where the implementer had a Python loop under both forms. The
    maintainer asked about it and it stood.
  - Eleven choices the plan left to the implementer were judged sound by the reviewer (only `T`, `A`, `W`, `K`,
    `C` are accepted in a string; the order group leads its term; the record validates its fields; and so on).
- **For P5:** the `HISTORICAL` table of `tests/test_contractions_interpreter.py` holds one supplement letter per
  string, and 12 of the 41 strings need two or three once they stand at library call sites. The handover's
  section 6 gives the table's format and two ways to change it. It is a test format; Claude decides it at P5.
- **One interruption:** the server process behind the session stopped once during the slice; the implementer was
  resumed from its transcript and nothing was lost.

## P2. The reference module and the dense functions (2026-11-17)

- **Rounds:** two. Round 1: not yet, two should-fix findings (the test that guards the module's independence
  could be walked around; a docstring of `mu_tilde` was false for apply and entries), six minor; no definition
  wrong. Round 2: sign off, four minor.
- **Result:** `backend/sampling_definitions.py`, 37 functions named as the variables, one signature, each a
  dense contraction of its subnetwork, jets by evaluation at the roots of unity and a discrete Fourier transform;
  `dense_probe(T, ww, pp=None, order=0)` in its new form with its 12 callers, and the new `dense_apply`,
  `dense_entries`. 59 more tests than P1. The whole gate green on the final tree (1,501 passed, 5 skipped, 1
  expected failure, 65,157 subtests; sphinx clean).
- **Checked independently by the reviewer:** every one of the 37 names recomputed by a route that shares nothing
  with the module (the page's recurrences in exact polynomial arithmetic; its own dense Jacobian; an exact double
  sum for `Kp`), three kinds, orders 0 to 5 and 16, one to four sites, tails: worst relative difference 1.4e-14.
  Against the library at 2026.5.0: 7e-15. The turn of the order leg checked from first principles. 77 mutations
  over the two rounds, every real one caught in the end.
- **Claude checked:** read the module's docstring (it reads as documentation); the diff's shape; ran the slice's
  tests, the doctests of the four touched backend modules, and sphinx after adding one clause to the docstring
  (complex arrays are not checked for; their imaginary part is dropped at order 1 and above).
- **Minor issues decided by Claude:**
  - `Kp`'s route is read off the arrays (finite differences, exact for this polynomial, on numpy; jax's second
    derivative on jax arrays), by the house dispatch rule, where the plan said "jax where present". The plan's
    sentence is corrected.
  - For apply and entries the reference returns zeros for `xi_tilde`, `mu_tilde` and their dots (the seed enters at
    the right end only, the note to row 4.4; the reviewer showed zero is the only value consistent with the
    readouts). Which names the engine accepts for these kinds is P9's, with a test that reference and engine
    agree; tell the maintainer then.
  - The reference takes four stores and the manifold names; a corewise point is `(B, G, G, G)`. Enough for P8.
  - Interpolation changed to the roots of unity (the implementer's choice, invited by Claude): right to 1e-14 at
    order 32 where Chebyshev nodes gave 2e-8 at order 16.
- **Edits to the variables page from this slice:** section 8 says which pairings hold for which kind (two of the
  four are of the probe kind only); section 9 lists the names of a corewise call (`nu_dot`, `eta_dot`,
  `nu_tilde_dot`, `eta_tilde_dot` among them), which follow from the rule of that section and await the
  maintainer's confirmation before P8.
- **For later slices:** P9's identity tests must not assume the pairings of `mu_tilde` and `xi_tilde` for apply
  and entries. P10 (the dtype rule): float64 cores with a float32 sample give float64 that is right only to 9e-8
  in the reference. P19: `docs/api_reference.rst` lists backend modules by hand and lacks the new one.
- **To know when perturbing code in place:** a same-size change written and restored within a second leaves
  stale bytecode that makes later runs lie. Work in a scratch copy, or run with `python -B` and
  `PYTHONDONTWRITEBYTECODE=1`.
- **Two interruptions** of the session during the slice (once with a machine restart); the implementer was
  resumed both times and its detached gate run survived the second.
- **To the maintainer:** the corewise names, for confirmation before P8.

## P3. Jets, the checkers and data preparation (2026-11-17)

- **Rounds:** two. Round 1: not yet, four should-fix findings (a sample with `pp = None` accepted above order 0;
  the check of `scales` in the frontend only; `rescale_direction` failing under `jit` with a traced length; a
  tolerance in the frontend's zero check passing every test), twelve minor. Round 2: not yet on one finding with
  a one-line fix (`ut3_unscale_coordinates` had dropped out of `__all__`), which the reviewer said needed no
  further round; Claude made it, with a test that pins the module's seven public names, and saw the page back
  in the sphinx build.
- **Result:** `backend/jets.py` (the dual-jet helper and its inverse, the four functions for derivative data,
  the checkers `check_sample`, `check_data`, `check_weight`, the two validators of `pp` copied) and
  `backend/data_preparation.py` (block norms, the equalizing weight, the two rescalings, `t3_unscale_coordinates`
  and its uniform twin, `check_coordinate_scales`); from `t3toolbox.fitting` nine backend functions and two of
  its own (`dual_jet_to_jet`, `rescale_coordinates`); `unscale_coordinates` on both tensor classes. 81 more tests
  than P2. The whole gate green on the implementer's final tree (1,580 passed, 5 skipped, 1 expected failure,
  65,875 subtests; sphinx clean); after it Claude's one-line fix and its two tests, checked by the slice's tests,
  the module doctests and sphinx.
- **Checked independently by the reviewer:** the dual jet, its inverse and the pairing identity (216 cases,
  3.6e-15); the derivative conversions against its own derivatives and the library's old derivative functions;
  both rescalings against dense tensors; the packed probe array against a definition of its own (81 cases); 167
  mutations over the two rounds, all caught in the end.
- **Went to the maintainer** (log entry 115): the four functions whose result depends on the type of the data
  take `kind` (`jet_to_dual_jet(kind, order, jet, weight)` and the three others); he said such a function is
  three functions in one. The names of a corewise call were confirmed.
- **Minor issues decided by Claude:**
  - `check_coordinate_scales(scales, shape)`, a public backend checker, by the rule that the backend offers the
    checkers. The frontend's `rescale_coordinates` is its own function that calls it.
  - The frontend's zero check in `dual_jet_to_jet` is on the divisor as the backend computes it (one private
    helper serves both tiers), so a weight whose square underflows is refused like a zero. Exact, no tolerance.
  - One dispatch per call over all the inputs; `rescale_direction` returns `ww` / `index` as given.
  - Choices of the implementer that touch public behaviour, accepted: the lowest floating precision wins where
    two user arrays meet (row 5.6); `prepare_derivative_data` accepts order 0; the uniform backend function also
    takes packed scales; the checkers take the ragged form only.
- **For later slices:**
  - P11: the functions of `data_preparation` that take `kind` are for the ragged type; `rescale_coordinates`
    cannot take a packed array.
  - P14: a backend user with a packed batch has no checker (the checkers take the ragged form).
  - P19: cross-references to the re-exported functions render as plain text; a frontend-only teaching example;
    `docs/api_reference.rst` lacks the new backend modules.
  - Left as they are: the frontend `rescale_coordinates` gives a poor message for a sample of the wrong form
    (review r2, finding 18); a 0-dimensional float32 `length` does not lower the dtype (finding 19).
- **One interruption** during the slice, coinciding with a scheduled check-in; the detached gate run finished and
  the agent was resumed automatically.

## A planned pause after P4 (the maintainer, 2026-11-17, 12:58)

"Let's have a planned pause after P4 is complete." When P4 is signed off, looked over and committed: do NOT start
P5. Refresh `dev/HANDOFF.md` and the folder's README for the state at the pause, stop the scheduled check-ins, and
report to the maintainer. P5 starts when he says.

**Added at 13:49:** at the pause, before P5, a plan phase for the generalization of the jet group (one jet group
of any number of axes; `TARGET_DECISIONS.md`, "One jet group, of any number of axes"): an Opus writer and an Opus
critic create the new slices and update `PART1_PLAN.md`; the maintainer reviews; then the new slices are
implemented by the usual pair; then P5. The brief for that plan phase is `reviews/jet_group_plan/BRIEF.md`.

### P4, review round 1 (2026-11-17, 14:35)

The reviewer (agent `ab209a5de9d210d82`): NOT YET on one should-fix finding, eight minor (`P4_REVIEW_r1.md`). Sent
to the implementer with my rulings:

- (1, should fix) a norm that overflows made an inconsistent frame read 0.0 and pass (float32 from norm 1.8e19):
  a norm that does not overflow, `nan` where it is not finite, a test, the docstring.
- (2, 3, 8) tests: pin the `1` of `max(1, ||T||)`; the guard's test must fail and not hang; no wall-clock
  assertions.
- (4, 5) docstrings on cost and on padding; sharing `D` in the difference if simple.
- (6, 7) the `LinAlgError` branch and the dtype of an unstacked float32 frame under numpy 1.x.
- (9) the duplicated helpers stay, with a comment naming the twin. **A note on the plan:** it names
  `t3_operations` for the difference; the addition lives in `t3_linalg`.

For the maintainer at the pause, from the review:

- The centers' residual: normalized by `max(1, ||T(D, L)||)` (absolute below norm 1, as the frame residual
  already was), reduced by the maximum over sites and families; a family that is `None` is skipped and
  `(None, None)` raises. Its cost grows like `d**2` and it needs no orthogonal frame (row 4.1's note expected a
  core-sized check).
- `UT3Frame.is_consistent` changes from `rtol * ||T||` to `rtol * max(1, ||T||)`: looser below norm 1, and more
  exact (the old subtraction read 3.9e-10 on a consistent frame of norm 3.7e6; the new residual 4e-15). Its only
  callers are tests. For the changelog in P19.
- `(x - y).norm()` loses accuracy across gauges; whether `_t3_difference_norm` gets a public name.

For P16: the frame residual costs six to eight frame constructions (fine once per model build); the enforcement
must pass `safety.effective_rtol(frame.data)`, since a float32 numpy frame reads 2.5e-7.
