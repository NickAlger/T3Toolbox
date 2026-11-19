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

### P4, signed off and committed (2026-11-17, 15:14)

- **Rounds:** two. Round 2 (`P4_REVIEW_r2.md`): SIGNED OFF, all nine findings resolved, three minor notes.
- **What round 1 changed beyond tests and docstrings:** the norm of the last core is taken after dividing by its
  largest magnitude (no overflow below the dtype's maximum; `nan`, never 0, where a norm is not finite); the Tucker
  cores are shared (each factored once, its factor absorbed into the TT cores, so every difference is a difference
  of TTs; the frame residual went from about six frame constructions to about two and a half, the centers' from
  13.6 s to 4.5 s on the reviewer's frame); the uniform twins select the real slots with `where`; on
  `LinAlgError` numpy recomputes one stack element at a time.
- **My look:** read the backend code, the frontend properties and the reviews. One addition of mine, for the
  reviewer's note 1 (the balancing of `D[i]` against `Vc[i]` was pinned by no test):
  `test_the_accuracy_does_not_depend_on_the_scale_of_a_down_core` in `tests/backend/test_frame_consistency.py`;
  it fails in a scratch copy with the balancing removed. Notes 2 and 3 need nothing (a zero stack element reads
  7e-16 in the centers' residual; a count in the response).
- **Gate:** the implementer's whole gate on its final tree, all steps PASS (suite 1,631 passed, 5 skipped,
  1 xfailed; `/tmp/gate_P4_r1`); the slice's quick tests after my test, 167 passed. The suite is at 1,632.
- **Not run here:** numpy 1.x (finding 7: an unstacked float32 frame's residual dtype). CI's floor leg runs it.
- **The pause begins.** P5 is not started. The items for the maintainer are under "P4, review round 1" above.

### At the pause: the plan phase for the generalized jets begins (2026-11-17, 15:15)

The maintainer settled its inputs between 13:31 and 14:44 (log entries 116 to 120). The Opus writer (agent
`ade730504190cf063`) was started at 15:15 on `reviews/jet_group_plan/BRIEF.md`; it changes `PART1_PLAN.md` and
writes `reviews/jet_group_plan/ACCOUNT.md`. When it reports, a fresh Opus critic reviews
(`reviews/jet_group_plan/REVIEW_r<n>.md`); they iterate until the critic agrees; then the maintainer reads and signs
off; then the new slices are implemented by a Fable 5.2 medium pair; then P5. No implementation runs meanwhile.

A correction: the two headings above first carried 15:20 and 15:30, written from an estimate; the commit times
are 15:14 and 15:15, and the headings now say so.

### The maintainer on P4's three items (2026-11-17, 15:18)

"1. Ok. 2. Ok. 3. Note for consideration later, after this refactor." So the centers' residual stands as built,
`UT3Frame.is_consistent` stands with `max(1, ||T||)` (for the changelog in P19), and the accuracy of
`(x - y).norm()` across gauges is in the backlog of `dev/HANDOFF.md` for after the refactor.

### The writer was stopped with the server and resumed (2026-11-17, 15:19)

The server process behind the session stopped about three minutes after the writer started; resumed by its id.
It did not coincide with a check-in.

### The plan phase: the writer's draft and the critic's first round (2026-11-17, 15:34)

- The writer (agent `ade730504190cf063`) reported at 15:33: four slices P4a to P4d (the jet contraction over
  `J`; the input jets and the dense functions; the reference module in several variables; `jets` and
  `data_preparation` in the jet order), the rest of the plan brought in line, `ACCOUNT.md`, eight open points
  O-36 to O-43 and proposed names N31 to N39. Its draft is committed as a69e497, marked as a draft under review.
- A fresh Opus critic (agent `ad73f7b1794a3f870`) was started at 15:34 on it; it writes
  `reviews/jet_group_plan/REVIEW_r1.md`. Its findings go back to the writer; rounds until the critic agrees.

### The plan phase: the critic's first review (2026-11-17, 15:49)

The critic: NOT YET (`reviews/jet_group_plan/REVIEW_r1.md`, committed as 50e2b62). It confirmed the mathematics
with its own scripts (the box product, the corner readout, the dual jet in two and three variables, the product of
factorials, the grid of roots of unity), the table of 29 strings and the contraction counts against P1. One
blocking finding: with input jets a forgotten `order` is no longer caught by the form of the sample, and with the
oracle's default `order=()` a fit at order 1 without a weight passes every check and fits another problem; the
default is the maintainer's to decide. Seven should-fix (the jet axes on the uniform layout are data, not
don't-care; the reference module must not import the helpers; the old derivative kinds at order 0 translate to
`(0,)`; a wider test matrix at `m = 2`; the one-program-size test kept; packed inputs specified; the open points
corrected) and twelve minor. Sent to the writer at 15:49 with my guidance: the default of `order` and the rule on the
jet axes of the uniform layout become open points for the maintainer, the second with the rule proposed from entry
116 (every slot of a jet axis is a coefficient; a zero written when packing is data).

### The plan phase: the critic agrees (2026-11-17, 16:04)

Three rounds (`reviews/jet_group_plan/REVIEW_r1.md` to `REVIEW_r3.md`, `RESPONSE_r1.md`, `RESPONSE_r2.md`).
Round 2 left one finding (the extension of `pack_vectors` would have zero-padded a mismatch in `W` that raises
today; now only the jet axes are padded, through a keyword `len_J=0`). Round 3: AGREED. The plan change stands in
`PART1_PLAN.md` as a draft awaiting the maintainer: slices P4a to P4d, eleven open points (O-36 to O-38, O-40 to
O-47; O-39 withdrawn as decided by rows 2.1 and 2.2) and the proposed names N32, N33, N35 to N41. My look: read
the head, the semantics and the algorithm of `jet_contract` in 2.4, the open points and the names; corrected the
plan's status line, which still said the whole plan awaited sign-off. Not yet done, and mine once he signs: the
amendments to `TARGET_DECISIONS.md` and `SAMPLE_VARIABLES.md` that the account's section 7 lists. Nothing is
implemented; no check-in is scheduled beyond the one pending.

### The plan for the generalized jets is complete; the pause (2026-11-17, 16:30)

- The maintainer answered the open points (16:21, 16:26; log entries 121 and 122) and approved the names. The
  writer folded the answers into `PART1_PLAN.md` in one editing pass (commit 48a1af4): section 11.2 has nothing open,
  11.1 has the answers, 11.4 the approved names. I checked the pass by search (no open mark left; every
  `LeastSquaresOracle(...)` passes `order` but the one test that it raises without; no sentinel; no fallback by
  total degree); the critic did not review this last pass, to save the maintainer's budget, since it settles
  nothing new.
- Two things in it for his eye: `error_table=True` (N42) is taken as approved with the other names; and the writer
  put it on `format_newton_iter` as well as on the frontend `newton_cg` and `make_newton_display`, because that
  public function is where the table is printed.
- **Paused at his request** ("commit it, update the handoff, and pause before implementation ... We will resume
  in a couple days"). Not done, on purpose: the amendments to `TARGET_DECISIONS.md` rows and
  `SAMPLE_VARIABLES.md` listed in `reviews/jet_group_plan/ACCOUNT.md` section 7 (mine, after his sign-off of the
  whole). No agent is running, no check-in is scheduled, nothing is uncommitted.
- To resume: `dev/HANDOFF.md`, the block "STATE AT THE PAUSE OF 2026-11-17", its NEXT.

### The work moves to Claude Code on the maintainer's laptop (2026-11-19, 10:18)

The maintainer: "I'm going to continue this locally with Claude code on my laptop. Can you make sure the handoff
and other documents are updated so that Claude code will have the right context?" The branch was unchanged since the
pause (`680ca36`). Done, documents only:

- `implementation/ORCHESTRATION.md`, new: where things stand; what the orchestrating session does for a slice; what
  was particular to the cloud session (agent ids, check-ins, the ten-minute limit, paths, versions, the commit
  trailer, the stale Claude doc); the first steps on another machine (the rewritten history; the environment; a
  baseline there); the amendments owed before P4a; the notes carried to later slices, gathered from this log;
  things learned the hard way.
- `dev/HANDOFF.md`: a START HERE block at the head of "The next arc" with the reading order for the work now; the
  sentence that no library code had been changed, corrected.
- Banners at the top of `TARGET_DECISIONS.md` and `SAMPLE_VARIABLES.md` naming what is not yet amended for the jets
  and what wins meanwhile. The amendments themselves are still owed, after his sign-off of the changed plan.
- `BRIEF.md`: its workspace section marked as the cloud workspace's. The folder `README.md`: a pointer for a new
  session; the Claude doc marked as far behind. `CLAUDE.md` (on this branch): a pointer at the head of "Current
  state" to the handoff, by its own handoff ritual.

Nothing of the library or the tests was touched. No agent is running and nothing is scheduled.

### Resumed on the laptop: the plan signed off, the records amended, the baseline (2026-11-19, 10:50 to 11:45)

- **The maintainer** (his message came before 10:56): "Part 1 of the plan is confirmed as a whole. But we may
  encounter issues along the way which require adjusting the plan as we go." Pauses after P4d, after P11 and after
  P15. Small or clear issues: Claude decides and logs; big, unforeseen or uncertain ones go to him at once. "Think
  from first principles ... use judgement rather than blindly following rules ... go slow and do things right."
  N42 (`error_table=True` on the three places) confirmed. CI through a draft pull request. The agents: two Opus 5.5
  on high effort, the same implementer and reviewer kept from slice to slice. Required reading for all: a handful
  of `TuckerTensorTrain` methods and the backend functions they call. Recorded in `../TARGET_DECISIONS.md`, the note
  "The changed plan signed off".
- **The clone** was already level with the remote at `10e680d` (it had the rewritten history); nothing to move.
- **Draft pull request #26** (`graded-messages` into `main`), opened 10:57: the workflows run only on `main` and on
  pull requests to it, so the branch had never run in CI and P1 to P4 had never run on the floor. First run, on
  `10e680d`: docs build PASS, examples PASS, current leg PASS; the floor leg was still running at 11:45.
- **The amendments owed for the jets** (`ORCHESTRATION.md` section 5; `../reviews/jet_group_plan/ACCOUNT.md`
  section 7), made by Claude: `../TARGET_DECISIONS.md` rows 2.1, 2.3 to 2.9, 3.4, 5.4, 6.2, 6.6 and 9.4 (row 9.4, the
  letter of the order group, was not on the account's list and is amended too); the notes on N16, O-10, O-27 and
  O-28, the group `A` (marked superseded in place) and jets in several variables as a future feature (likewise);
  "Carried to Part 2", item 1 restated and items 4 to 6 added (the scan over several jet axes; the zero slots of an
  input affine in several variables; data by total degree); the banner. `../SAMPLE_VARIABLES.md` sections 4, 6 and
  7 brought in line in place (it is the reference, not a log), with a dated note and a new banner.
- **`PART1_PLAN.md`:** the status line (signed off), N42 confirmed (11.4), section 12 (the agents, the pauses, CI).
  `BRIEF.md`: the laptop's workspace in place of the cloud's, the required reading, how the pair works with Claude.
  `ORCHESTRATION.md`: section 0, "Since the move to the laptop".
- **The baseline**, `BASELINE_laptop.md`: the whole gate at `-n 3` on `10e680d`, 10:55 to 11:44, every step PASS;
  the suite 1,632 passed, 5 skipped, 1 xfailed, the same as the cloud workspace's. The suite takes 23 minutes and
  peaks at 9.8 GB (2.5 GB left): nothing may run beside it.
- **Required reading, Claude:** `to_dense`, `probe`, `probe_derivatives`, `norm` of `TuckerTensorTrain`; the backend
  `t3_probe` and its sweep (`compute_xi`, `_mu_step`, `compute_mu`, `compute_nu`, `_eta_step`, `assemble_z`),
  `common.get_backend`, the jetted kernels of `sampling_derivatives` (`t3_probe_derivatives` and its steps), and
  P1's `jet_contract` whole.
- **The agents** (Opus 5.5, high), started during the gate (after 10:56) on an orientation turn (read only; no Python,
  no edits): the implementer (about 270k tokens) and the reviewer (about 280k). Both reported a sound reading of P4a.
  Independently, both raised the same two gaps in the plan's section 2.4, which Claude decided as minor and wrote
  into the plan (marked "2026-11-19, at the start of P4a"):
  - **"Most slots" is counted after clipping to the box** (for a readout, to the corner's box): slots beyond the
    box are never read, so they must not decide the carrier or the pair. Values are the same either way; the cost
    and the counts are not.
  - **A string with no `J` raises at every `m`**, `m = 0` included. The plan had it raise at `m >= 1` only and run as
    plain `contract` at `m = 0`, which would make a string's validity depend on the order, against P1's rule that a
    call site is wrong at every order or at none. A string with no `J` is a `contract` string.
  - The implementer found that P1's `test_jax_default_optimizer_is_the_one_read_here` pins jax's einsum default
    as `'auto'`, which is jax 0.10's; jax 0.4.30, the floor's, has `'optimal'`. Ruling in the message that starts
    P4a; the floor leg's result decides whether the behaviour tests beside it hold on 0.4.30 too.
