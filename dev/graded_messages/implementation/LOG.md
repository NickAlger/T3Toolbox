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

### P4a: `jet_contract` over one jet group (2026-11-19, 11:45 to 14:08)

- **Rounds:** two. The implementer (Opus 5.5, high) built the slice after its orientation turn; the reviewer (the
  same agent that oriented, Opus 5.5, high) wrote `P4a_REVIEW_r1.md` (NOT YET, one should-fix) and
  `P4a_REVIEW_r2.md` (SIGN OFF). The implementer's answers: `P4a_RESPONSE_r1.md`. Handover: `P4a_HANDOVER.md`.
- **What changed:** `backend/contractions.py`, the `jet_contract` section rewritten (`contract` untouched):
  `GroupLengths(J, W, K, C)` with `J` any count; `jet_contract(subscripts, *operands, lens, order, over_orders)`, the
  one behaviour of section 2.4; the string read and cached on the string, the roles (ride, product, readout;
  carrier, pair, partners) cached on `(string, jet sizes, order)`; the scan body memoized on every static it reads;
  `_read_order`, the one reader of `order`, which P4b reuses; the markers `PART 2 (row 5.4, jets in several
  variables)` and `PART 2 (row 5.5)`. Tests: `tests/_jet_contract_cases.py` (the 29 strings, by family xi 5, z 3,
  mu 4, nu 4, eta 4, V 5, H 4, checked; the library's operand shapes; an independent multi-index reference),
  `tests/test_jet_contract.py` rebuilt (47 tests), and the `jet_contract` rows of the interpreter, sharding and
  dispatch tests.
- **Decided by Claude at the start (minor, logged above):** "most slots" after clipping; a string with no `J`
  raises at every `m`; the jax default pin accepts `{'auto', 'optimal'}`; a product with several partners at `m = 1`
  under `'scan'` scans the first full one; an unreachable corner gives zeros; `order` a tuple of nonnegative
  integers (Python or numpy integer scalars, normalized to Python ints; `bool` rejected), read before any cache.
- **Decided by Claude during the slice (minor):** accepted the implementer's one departure, **an input term of `J`
  alone raises** (from the string, at every `m`): taken one coefficient at a time it would leave `contract` an empty
  term; a scalar polynomial is written `'JW'` with `W = 0`. The reviewer agreed.
- **Review round 1** (NOT YET): the code held up under an independent roots-of-unity oracle (1,296 calls per seed,
  three seeds, `m = 1` to 3, worst 2.3e-15), NaN beyond the box (bit-identical results), the edge cases, P1's
  contraction counts on all 41 operand shapes, constant program size under `'scan'`, 25 of 26 mutants caught (the
  survivor equivalent). One should-fix: the lone readout and the unreachable-corner fallback were never compiled
  under jit (host-numpy mutants there passed every test). Three minor (a docstring calling `n_contractions`
  independent; a signature's comment column; the `J`-alone rule not in the docstring). All fixed in round 1;
  round 2 confirmed the mutants are now caught.
- **The row-5.5 recount** (`_pairwise_path` on the derived step strings): at order 4, 16 of the 20 product cases
  pair the carried jet with the core first, the cheap 4 being the `nu` strings with a two-slot leaf, as in P1; at
  order 1, 20 of 20. As the plan predicted. The expected failure is restated and still xfails.
- **Gate:** the implementer's full gate at `-n 3` on its round-0 tree, every step PASS (suite 1,647 passed, 5
  skipped, 1 xfailed; 15 tests more than the baseline; peak 10.1 GB, 2.2 GB left). After round 1 (tests and
  docstrings only): the slice's tests and the module doctests, single-process (implementer, reviewer, and Claude:
  67 passed, 1 xfailed; doctests 3 passed). Not rerun as a whole: the changes after the gate are two test rows and
  docstrings.
- **The floor, locally** (`tttt`, jax 0.4.30, numpy 1.22), on P4a's tree: `tests/test_jet_contract.py` 46 passed,
  1 xfailed (the reviewer; the jax behaviour tests of `TestOrderInsideAStep` hold under jax 0.4.30's default
  `'optimal'`, so the restated pin is enough); `tests/test_dispatch.py -k jet_contract` 2 passed in 44 s (Claude).
- **Found outside the slice: the CI floor leg hangs on the branch.** CI's first run on PR #26 (head `10e680d`): the
  floor leg's suite step ran for hours, against 11 to 16 minutes on `main`. Reproduced locally on the floor env,
  file by file, on the committed head: P1's `test_jax_default_optimizer_is_the_one_read_here` FAILS there (fixed
  by P4a); `tests/test_sampling_definitions.py` does not finish in 15 minutes, stalling at
  `TestKp::test_jax_against_finite_differences` (P2's reference module; 36 s on jax 0.10); `tests/test_dispatch.py`
  serially reaches `test_jit_fitting` (pre-existing) at 15 minutes, which may only be its known serial slowness.
  One `Kp` case by jax takes 1 to 2 s on the floor (0.05 to 3.6 s on jax 0.10), with or without the x64 context,
  so the jax route is not itself pathological; the cause is being looked for. Every other file P1 to P4 added
  passes on the floor.

### P4b: the input jets and the dense functions; the floor hang found and fixed (2026-11-19, 14:08 to 15:39)

- **Rounds:** two (`P4b_REVIEW_r1.md` NOT YET, one should-fix; `P4b_REVIEW_r2.md` SIGN OFF; `P4b_RESPONSE_r1.md`;
  handover `P4b_HANDOVER.md`). The same implementer and reviewer as P4a. The reviewer read the slice while the
  implementer's gate ran, and ran its checks after.
- **What changed:** `backend/jets.py`: `affine_jet(ww, *directions)` (tuple or packed form, as given) and
  `index_to_one_hot(index, shape)`. `backend/probing.py`: `dense_probe(T, ww, order=())` on input jets; `_dense_jet`
  multiplies the tensor by each mode's input jet one mode at a time, by explicit loops over the pairs of
  multi-indices inside the box (never `jet_contract`); `_dense_contract` and `_check_pp_and_order` deleted (no other
  caller). `backend/apply.py`: `dense_apply(T, ww, order=())`. `backend/entries.py`: `dense_entries(T, index)`,
  indexing only, its docstring showing the recipe of entry 120. `backend/sampling_definitions.py`: builds its own
  two-slot input jet until P4c, never imports `jets`. The callers with `pp` updated (`data_preparation`'s doctests,
  `test_jax_absent`, `test_data_preparation`, `test_jets`). The dense functions read `order` through
  `contractions._read_order` (the implementer's choice: `jets` does not re-expose it; P4d's functions import it
  from `contractions` too).
- **The CI floor hang, found by Claude and fixed in this slice (decided as minor; P4b already edited the module):**
  `sampling_definitions._einsum` called `jnp.einsum` without `optimize`. On jax 0.4.30 the default is `'optimal'`,
  opt_einsum's exhaustive path search, exponential in the number of operands; `_tensor` contracts a whole network
  in one einsum, so the four-site test problem `long()` never finished (faulthandler stack: `w_tilde` -> `_z` ->
  `_tensor` -> `_einsum` -> `opt_einsum.paths._optimal_iterate`). jax 0.10's default `'auto'` hid it. The jax path now
  passes `optimize='greedy'`, with a docstring saying why the house convention does not apply there. Proof:
  `tests/test_sampling_definitions.py` on the floor env, single-process, 65 passed in 107 s (it hung before); 132 s
  on the `t3toolbox` env. The lesson is in `ORCHESTRATION.md` section 7.
- **Review round 1:** the reviewer's own oracle (unclipped input jets at roots of unity, one einsum per node,
  inverse DFT; about 640 outputs over three seeds, `d = 1..4`, `m = 1..3`, mixed and discrete jet sizes, inputs
  beyond the box, `W` and `C` stacks) agrees to 2e-15; NaN beyond the box changes nothing; the recipe of entry 120
  matches, and its slot 0 equals `dense_entries` exactly; packed `affine_jet` right at `m = 3`. 5 of 8 mutants caught,
  the 3 others equivalent. One should-fix, **ruled by Claude** (minor): `index_to_one_hot` gave a zero vector for an
  index outside `[-N, N)`, silently. Now it follows numpy indexing and `dense_entries`: `[-N, 0)` wraps; a numpy
  index outside raises `IndexError` naming the mode and the values; a jax index (possibly traced) is unchecked and
  gives zeros, as its docstring says. The reason: an index outside the grid has no one-hot vector, and zeros would
  turn a bad index into plausible data.
- **Round 2:** SIGN OFF, with one minor note: no test had `-N - 1` alone in its mode (a mutant narrowing the lower
  bound survived). **Claude added the assertion** (`TestInputJets`) and checked in a scratch copy that it fails on
  that mutant and passes on the code.
- **Gate:** the implementer's full gate at `-n 3`, with the floor fix in and before round 1, every step PASS (suite
  1,657 passed, 5 skipped, 1 xfailed; 24.9 min; peak 9.8 GB; module doctests 312). After round 1 (`index_to_one_hot`
  and its tests): the affected tests and doctests single-process (implementer: 54 + 2 passed, doctests 21; reviewer:
  150 passed in round 1; Claude: `TestInputJets` 4 passed after the added assertion).
- **For P4d** (the reviewer): `data_preparation`'s doctest comments still use the old `T` group notation.
- **Pushed after this commit**, with P4a and the records, so that CI's floor leg runs on the fix.

### CI on P4b's head: the floor leg finishes; two subTest labels fixed (2026-11-19, 16:05)

- CI run 44620999298 on `d26f1d4c`: docs build, examples and the current leg PASS; **the floor leg no longer hangs**
  (14 min 55 s, against hours before the `_einsum` fix), and FAILS on two tests of P4a:
  `TestStrictness::test_the_jet_shape_is_a_tuple_of_nonnegative_integers` and
  `TestGroupLengths::test_counts_are_nonnegative_ints_and_J_any_count`, both `execnet DumpError: can't serialize
  <class 'numpy.int64'>`: their `subTest` labels carried numpy scalars, which xdist cannot send on numpy 1.x (the
  rule "subTest params are builtins" of `dev/HANDOFF.md`; neither local env reproduces it: `t3toolbox` is numpy 2,
  `tttt` has no xdist). 1,654 passed otherwise. **Fixed by Claude** (minor): the labels are `repr(...)`. Checked on
  both envs; CI is the judge of the serialization.

### CI green on the branch, the floor included (2026-11-19, 16:17)

CI run 44624219934 on `3ccff5da`: docs build, examples, the current leg (17 min 42 s) and **the floor leg (13 min 43 s,
in the historical range of `main`)** all PASS. The first time the branch has passed on the floor.

### P4c: the reference module in several variables (2026-11-19, 15:39 to 17:17)

- **Rounds:** one. `P4c_REVIEW_r1.md`: SIGN OFF, no blocking or should-fix finding. Handover `P4c_HANDOVER.md`.
- **What changed:** `backend/sampling_definitions.py`: the 37 functions take `order=()` (the jet shape) and a sample
  of input jets, or `index` for entries, no `pp`; a private reader of the jet shape of the module's own (imports
  neither `jets` nor `contractions`), tested to agree with `contractions._read_order` on 15 inputs; an entries call
  with jets raises; inputs clipped to the box; a grid of `M_j = d(a_j - 1) + n_j + 1` roots of unity per axis and a
  DFT along each; every function returns its declared jet sizes (plan, section 2.3); the seed flipped on every jet
  axis; the `_einsum` fix kept. The module docstring and doctests rewritten. Its independence test unchanged
  (byte-identical to `HEAD`, the reviewer checked) and passing. `tests/test_sampling_definitions.py` rebuilt over
  the plan's jet matrix with a real-grid dense route of its own; new: `TestAgainstTheOldKernels` (the bridge), the
  readers agree, a second variable of order 0 equals one variable for every name, entries equals apply at
  `index_to_one_hot` and raises with jets.
- **The bridge, derived twice independently** (the reviewer, and Claude while the review ran): the old `z` is `t!`
  times the new; the old transpose `sum_t t! J_t^T c_t` equals the new `g = sum_t J_t^T omega_t^2 r_t` at
  `c_t = omega_t^2 r_t / t!`, in FORWARD order (the turn is inside the new `g`). The test asserts exactly that, with a
  nonuniform weight. **Claude added that sentence to the plan's P6** (the bridge slice), so its implementer need not
  derive it again; the plan's older "`t!` per slot, the turn" was ambiguous on where the turn goes.
- **The nodes, checked by both:** the largest degree in `s_j` is `d(a_j - 1)` for the plain variables,
  `(d - 1)(a_j - 1) + n_j` for probe's tilde ones, `d(a_j - 1) + n_j` for apply's `nu_tilde[0]` (the seed and all `d`
  inputs); so `M_j` is exact for every variable and tight only for apply at full-size inputs (one node fewer breaks
  apply's `nu_tilde[0]` only).
- **The reviewer's independent check:** `xi`, `mu`, `nu`, `eta` and their tilde forms recomputed by their
  recurrences, each step one P4a `jet_contract`, on a frame of four unrelated stores, probe and apply, `d = 3, 4`,
  seven jet cases (mixed sizes, clipping, `m = 3`): agree to 2e-15 over three seeds. Mutants: the implementer's 16
  of 17 caught (the survivor equivalent), the reviewer's 4 of 4.
- **The test budget:** `tests/test_sampling_definitions.py` single-process 132 -> 217 s (current env), 107 -> 224 s
  (floor). The implementer trimmed only cases of no distinct coverage (the reviewer agreed); the next cut, if one is
  wanted, is the four-site half of `TestJetsTwoWays` (101 s of the floor's total). Left as is: in the `-n 3` suite
  it is spread over workers.
- **Gate:** the implementer's full gate at `-n 3`, every step PASS (suite 1,660 passed, 5 skipped, 1 xfailed; 25.6
  min; peak 9.9 GB). The floor env: the file passes (68 passed). The reviewer: 72 passed, doctests 37; Claude:
  doctests 37.

### CI on P4c's head: one more numpy subTest label (2026-11-19, 17:45)

CI run 44636416721 on `5a102126`: everything PASS but one test on the floor,
`TestStructure::test_the_jet_shape_is_read_as_the_library_reads_it` (P4c), the same `DumpError` on a `numpy.int64`
`subTest` label. Fixed by Claude (`repr(...)`). The rule was only in `dev/HANDOFF.md`; it is now in `BRIEF.md` ("Rules
for both agents"), and the agents are told. A local guard (a `conftest.py` check that every `subTest` label is a
builtin, so the trap fails on any environment) is a candidate for the maintainer at the pause.

### P4d: `jets` and `data_preparation` in the jet shape; their frontend reach (2026-11-19, 17:17 to 19:17)

- **Rounds:** one. `P4d_REVIEW_r1.md`: SIGN OFF, no blocking or should-fix finding, two optional minor ones, which
  Claude made (below). Handover `P4d_HANDOVER.md`.
- **What changed:** every function of `backend/jets.py` and `backend/data_preparation.py` reads the jet shape with
  `contractions._read_order` (through one helper, `jets._kind_and_order`; the entries kind takes `()` only and
  returns its data unchanged, O-47). The dual jet weighs by `weight**2`, then flips every jet axis; the derivative
  conversions use `k_1! ... k_m!` (with no weight `prepare_derivative_data` returns the factorials, `F` for apply,
  `(d,) + F` for probe); `check_sample`, `check_data`, `check_weight` take `(kind, order, ...)` and input jets, and
  state the trap of a wrong jet shape; `check_perturbation_*` deleted from `jets`. `data_block_norms` in the jet
  shape; `rescale_direction` multiplies slot `k` by `length_1**k_1 ... length_m**k_m` (one length or one per
  variable); `rescale_coordinates` scales every slot of every input jet. Shape comments and docstrings in `F`, `J`,
  `J_in_i` (N39; the reviewer's note of P4b). Frontend: `affine_jet` and `index_to_one_hot` reachable from
  `t3toolbox` and `t3toolbox.fitting` as the backend objects (O-37); the frontend `dual_jet_to_jet` reads the jet shape
  before its safe-mode check; the frontend `rescale_coordinates` reads the mode sizes off the input jets.
- **Departures, accepted by Claude as minor** (the reviewer agreed): `tests/test_import_graph.py` changed (`jets`
  imports `contractions`; a pin that `contractions` does not import `jets`); a doctest of
  `TuckerTensorTrain.unscale_coordinates` the plan's search missed now passes `()`; `rescale_direction` raises for
  the entries kind (the plan's signature is probe and apply); a length array of size 1 is accepted at `m = 1`.
- **The review:** the pairing identity at `m = 1, 2, 3` with a nonuniform weight (the reviewer's oracle and P4a's
  `jet_contract`); weigh-then-flip slot by slot; the round trip exact, on a packed probe array too; the factorials
  against hand-written mixed partials of `exp(a s)(1 + c s_1 s_2)` at `(2, 3)`; `rescale_direction` gives the data of
  the same tensor at four jet shapes, both kinds, one length and per-variable lengths; `rescale_coordinates` the data
  of `T / (s_0 x s_1 x s_2)`. The checkers: every wrong form raises. The frontend: the razor holds; safe mode raises
  on a zero weight, `safety.unsafe()` and jit divide, equal to the backend. The search for leftover `pp` and integer
  orders: only the old model layer of `fitting.py`, which P16 replaces. The implementer's mutants 20 of 20, the
  reviewer's 3 of 3; no numpy `subTest` label (both checked with `subTest` patched).
- **Claude's two edits after the sign-off** (the reviewer's optional minors): `check_sample`'s docstring states the
  other direction of the trap too (a jet shape with one variable too many reads an axis of `W` as a jet axis); the
  frontend `rescale_coordinates` reads the kind and the jet shape before it checks the scales, as `dual_jet_to_jet`
  does (checked: an integer `order` with wrong scales now raises the jet-shape `TypeError`). `test_jets`,
  `test_data_preparation` 73 passed; the doctests of `fitting` and `jets` 18 passed.
- **Gate:** the implementer's full gate at `-n 3`: the first suite run had 6 failures in `tests/test_dispatch.py`
  (four tests outside the helper the plan counted still called the old forms), fixed; the suite step again: 1,662
  passed, 5 skipped, 1 xfailed. The other five steps passed on the first run (module doctests 312, quickstart, 48
  pages, 14 examples, sphinx 0 warnings); not rerun, as the fix touched only `tests/test_dispatch.py`.
- **After P4d nothing in P1 to P4's code takes `pp` or reads an integer order** (the implementer's search, repeated by
  the reviewer).

### The jet sub-arc P4a to P4d is complete: the planned pause (2026-11-19, 19:17)

The first of the maintainer's three planned pauses. Pushed and CI checked before the report to him. Summary in
`dev/HANDOFF.md`, START HERE.
CI run 44652602834 on `53d525fb` (P4d): both test legs, the examples and the docs build PASS (floor 17 min 32 s).
Nothing is running; the two agents are idle and are resumed for P5 when the maintainer says so.

### The pause ends: the maintainer's answers; the `subTest` guard; P5 designed (2026-11-20)

- **The maintainer's answers** (recorded in `../TARGET_DECISIONS.md`, the note "The first planned pause"): Claude's
  minor decisions of P4a to P4d agreed; the plain-integer `order` raising (O-36) confirmed after Claude's
  explanation; the many-operand einsum order carried to Part 2 (item 7 there); the `subTest` guard approved; the test
  cut held in reserve, recorded in `dev/HANDOFF.md` with its reasoning, consequences and measured savings (the
  four-site half of `TestJetsTwoWays`: 122 s -> 48 s on the floor env, 67 s -> 32 s on `t3toolbox`, single process);
  the same agents continue.
- **The `subTest` guard** (`tests/conftest.py`, Claude): wraps `unittest.TestCase.subTest` and raises `TypeError` on
  any label (keyword or `msg`) that execnet cannot send, by exact type as execnet dispatches (`save_<type>`: `None`,
  `bool`, `int`, `float`, `complex`, `str`, `bytes`, and tuples, lists, sets, frozensets and dicts of them); pinned by
  `tests/test_conftest_guard.py` (3 tests, both envs). The full suite at `-n 3` with it: 1,665 passed, 5 skipped, 1
  xfailed: no existing label violates it. Peak 10.4 GB, 1.9 GB left at the lowest: the margin is thinner than at the
  baseline. `BRIEF.md`'s rule points at it.
- **P5's design, before any code:** the implementer's sketch (`P5_DESIGN_SKETCH.md`) and the reviewer's critique
  (`P5_DESIGN_REVIEW.md`: sound, with conditions). **Ruled by Claude** (plan details within decided rows; the plan
  amended where it said otherwise):
  - routes as several records per name chosen by `needs`, the precedence declared and tested, a held variable never
    changing the route; `present_args` the one carrier of a call's static facts, the call type included; the corewise
    graph (P8) derived from the manifold one by a declared rename table over names and `reads`, frame-form rows
    dropped (the reviewer's alternative to an alias table on `want`, since the corewise routes differ, not only the
    names);
  - **one `GroupLengths` per node**, each graph row declaring the call's record or the same with `K = 0` (a node with
    no direction), so one tilde step serves the residual's and the `Jp`-seeded sweeps with one literal; every call
    still passes one whole record and a stray `K` axis still raises (row 5.4). The plan's 2.4 "one record per
    computation" amended;
  - `w` is an argument, not a slot of `HeldVariables`; the plan's rule amended to "the computed variables some other
    record reads";
  - the `HISTORICAL` table: option (b), `jet_contract` literals out of the interpreter's identifiability sweep, with
    two guards (every library literal with `J` is a `jet_contract` first argument; every library `jet_contract`
    literal is in `STRINGS`) (`ORCHESTRATION.md` section 6 left this to Claude);
  - P5 computes the result dtype by the rule; `Vc` is given, never derived; P6's short form of `eta_tilde` will be a
    graph record.
  - **The reviewer's Finding A**, a gap in the plan: `sum_over_samples=False` needs readout strings with `W` in the
    output (the `V` and `H` families, about nine), not among the 29. Within row 3.7, no new group: a sentence added to
    the plan's P10 (added to `STRINGS`, chosen by a static flag, as the library's two bodies do today).

### P5: Engine A (2026-11-20, from the design to 02:57)

- **Rounds:** a design sketch and its review before any code (above), then two review rounds on the build
  (`P5_REVIEW_r1.md` NOT YET, one should-fix; `P5_REVIEW_r2.md` SIGN OFF; `P5_RESPONSE_r1.md`; `P5_HANDOVER.md`).
- **What was built:** `backend/sampling_steps.py` (private: `xi_step`, `mu_step`, `nu_step`, `eta_step`, `z_step`,
  the whole steps `r_step`, `f_step`, `r_block_sumsq_step`, `start_ones`; each term one `jet_contract` on a literal
  in `STRINGS`, `C` written); `backend/sampling_variables.py` (`sampling_variables`, `HeldVariables`; the probe graph,
  11 rows of `Rule(name, step, reads, needs, carries_K)`, two routes for `z`; the declared tables `_BOND_OFFSET`,
  `_SHAPES`, checked against the docstring by a test; `_closure` cached on hashables, `present` carrying the call
  type; bodies memoized on `(step_name, lens, order, form)`; the start joined to a sweep in `_with_end`; the result
  dtype over what the closure reads; the checks of section 2.3 step 1 and nothing more). Tests: four new files (36
  tests) and option (b) in the interpreter test with the two guards. The parts not yet built raise
  `NotImplementedError`.
- **The floor env caught a real bug in the build** (the implementer): numpy 1.x promotes `0.5 *` a float32 scalar to
  float64, and `f` came back as a numpy scalar; fixed, and a mutant reverting it is caught on `tttt`.
- **Sphinx, ruled by Claude:** the implementer had first silenced 18 ambiguous-`z` warnings with an
  `autoapi-skip-member` hook hiding `HeldVariables`' fields. The cause was prose in numpy-style type fields (`y: of
  the type of z`), which Sphinx turns into cross-references, until then resolving silently to
  `ManifoldCurvatureData.z`, a wrong link. Fixed at the source instead: `docs/conf.py` unchanged; 22 type fields in
  `jets`, `data_preparation`, `fitting` and one in `sampling_variables` are real types with the prose in the
  description; the implementer found 31 more return types in `sampling_definitions` naming `d`, fixed alike. Sphinx 0
  warnings with the fields rendered; no page links to `ManifoldCurvatureData.z` but its own.
- **Review round 1:** the engine checked against the P4c reference over eight jet cases with a nonuniform weight, `d =
  3` and `d = 4` with tails of different `L` and `R` end bonds, and against the reviewer's own dense oracle (worst
  2.7e-15; 2.9e-15 on the floor); a deliberately modified held `mu` changes exactly what the recurrences predict;
  step counts from the reviewer's own reading. Should-fix: the held-shape check was pinned on one slot only (fixed:
  two slots, the bad one second and unread). **Ruled by Claude (from rows 5.6 and O-11):** every output of a call but
  the sample `w` is in the result dtype; a held variable counts when a row reads it or it is wanted, and is cast once;
  one neither read nor wanted is not counted. Written into the plan's section 2.3 step 4.
- **Round 2:** SIGN OFF, with three wording notes, made by Claude (the engine docstring "used, not recomputed (cast
  once ...)"; the plan's step 4 "every output but `w`"; P10 states that `use_jax` reads every held value).
- **Gate:** the implementer's full gate before round 1 (suite 1,705 passed; Sphinx after the fix 0 warnings; module
  doctests 314); **Claude's run on the final tree:** suite 1,706 passed, 5 skipped, 1 xfailed (27.5 min, peak 8.9 GB),
  module doctests 314. Floor env: the four new files pass (implementer, reviewer).
- **A stuck wait loop** (`while pgrep -f "pytest tests/ -q -rfEs"; do sleep 20; done`, the implementer's, matching its
  own command line) ran two hours and kept the implementer's reports from being delivered; killed by PID. The rule
  is in `BRIEF.md`.

### P6 designed (2026-11-20)

- The implementer's note (`P6_DESIGN_NOTE.md`) and the reviewer's critique (`P6_DESIGN_REVIEW.md`). The reviewer
  confirmed the short form of `eta_tilde` in any number of variables (`V_tilde` at the corner reads exactly the
  slots `[c - a + 1, c]` of `eta_tilde` per axis, `a = min(J_in, F)`) and its worth (for a line at order `n`: 2 calls and
  `2n + 1` slot pairs per site and term, against `n + 1` calls and `(n + 1)(n + 2)/2` or `(n + 1)^2`).
- **The maintainer's rule, given during this design** (recorded in `../TARGET_DECISIONS.md`): "the same value" means
  equal to within rounding; two algorithms equal to rounding and roughly equally stable are both acceptable; bitwise
  identity has low weight. In `BRIEF.md` too.
- **Ruled by Claude** (written into the plan's P6 and section 2.2): `V_tilde` always reads the private short node, by
  one row, and the full `eta_tilde` has its own row, run only when wanted (the reviewer's alternative to two routes
  chosen by a "wanted" fact; chosen for simplicity, at a negligible cost: `2 prod(a)` readouts per site when both are
  wanted); each row declares its own start value (`'ones'`, `'zeros'` or a node: apply's `nu_tilde[d]` starts at the
  seed, P9); `HeldVariables`' slots are every public computed variable (row 6.4), `w` and private nodes excepted; P7's
  `Jp`-seeded rows are private twins named by a rule, with a test; the bridge lists the old tilde kernels it leaves
  out; the generic twin computes `V_tilde` from the full `eta_tilde`; the new readout string goes into `STRINGS`.

### P6: Engine B, the tilde grade (2026-11-20, to 04:07)

- **Rounds:** one on the build (`P6_REVIEW_r1.md`: SIGN OFF, two minor findings), after the design note and its review
  (above). Handover `P6_HANDOVER.md`.
- **What was built:** the steps `xi_tilde`, `mu_tilde`, `nu_tilde`, the full `eta_tilde`, the private short node
  `_eta_tilde_top` (`eta_tilde_top_step`: the top `a = min(J_in, F)` slots per jet axis, two readouts each at a
  lowered corner), `V_tilde` (reads the short node, by one row), `H_tilde` (three corner readouts), the seed
  `w_tilde_step` (probe; `jet_to_dual_jet`), `g_step`, `start_zeros`. The row type gains `start`, declared per row; 9
  new rows, all `carries_K=False`; private nodes (leading underscore) can be neither wanted nor held.
  `HeldVariables`: 18 slots, every public computed variable; the held-shape check covers one array, a tuple over the
  index, and `g`'s pair, its messages naming the entry (`g[1][0]`). `STRINGS` gains the family `eta_top` (two
  `W`-keeping readout strings, written with `K`); the pinned counts updated. Tests: every new name against the
  reference over the jet matrix with nonuniform weights; the generic twin (`tests/_generic_twin.py`, numpy and
  itertools only, AST-pinned, `V_tilde` from the full `eta_tilde`, five jet cases incl. mixed modes); the short node
  against the slice of the full `eta_tilde`; `g` against central differences of `f`; the graph tables, slots,
  starts and step counts; the temporary bridge (`tests/bridge/test_old_kernels_bridge.py`, deleted in P20).
- **The bridge's conversion of the old adjoint variables, derived by the reviewer** (the implementer had found it by
  script, the only one of five candidates that held): the old objective is `sum_t t! <c_t, z_t>`, its adjoint taken
  against `X^(r) = r! X_r`, so it is `(1/r!) dL/dX_r`; the new multiplier, stored turned and read at the corner, is
  `dL/dX_r` at slot `n - r`. So old slot `r` = new slot `n - r` divided by `r!`, with `deta -> xi_tilde`, `tau ->
  mu_tilde`, `sigma -> nu_tilde`, `dxi -> eta_tilde`.
- **The reviewer's own checks:** the short node against the full slice by calling the step on the engine's own
  variables (mixed modes at `(1, 2)`, mixed sizes at `(2,)` and `(1, 1, 1)`, 1e-15); `g` against a dense transposed
  Jacobian built from a dense tangent tensor per coordinate and probed by its roots-of-unity oracle, no engine or
  reference code (1e-15, four jet cases, two seeds); seven jet cases with `d = 3, 4` and tails, worst 2.6e-15;
  floor env 46 passed. Mutants: the implementer's 17 of 17 (the reviewer's four among them), the reviewer's 2 of 2.
- **Minor 1, made by Claude:** the test that `g` does not depend on wanting `eta_tilde` asserted bit-identity;
  under the maintainer's rule it now pins the ROUTE exactly (the short node made once per site in both calls) and the
  value to rounding (1e-13). **Minor 2:** `tests/bridge/` is staged by file name, not with its `__pycache__`.
- **Gate:** the implementer's full gate at `-n 3`, every step PASS (suite 1,715 passed, 5 skipped, 1 xfailed; peak
  8.9 GB; module doctests 314; Sphinx 0 warnings). After Claude's test change: `TestTheShortForm` 2 passed.

### P7 designed (2026-11-20)

- The implementer's note (`P7_DESIGN_NOTE.md`) and the reviewer's critique (`P7_DESIGN_REVIEW.md`: sound, two changes).
  The reviewer checked the four dot recurrences term by term against `SAMPLE_VARIABLES.md` 5.2, bases included, and
  the mechanical derivation of the `Jp`-seeded twins for the seed and the short node.
- **Ruled by Claude** (written into the plan's P7): the dot rows, `Jp`, `Jp_norm_sq`, the twins (`'_Jp.' +
  name.lstrip('_')`, mechanical, `JtJp` the twin of `g`, none of the full `eta_tilde` or of `g`) as the implementer
  proposed, with a `_base(name)` helper; **the reviewer's design for the two names with both dot forms** (private
  `_frame.nu_L_dot` / `_frame.eta_D_dot` read by `Jp`, the public names identity route rows needing `variations`, P8
  adding the raw route in front; no public name moves between slices); **one structural check at entry**: every
  `V[i]`, `H[i]` (and a held `Jp`) has the `K + C` prefix the engine reads from `V[0]`. The reviewer showed that a
  size-1 `K` on `H` broadcasts silently in the contraction (a wrong `Jp`, no error). Read as within O-18 (the engine
  reads the size of `K` from one operand; checking that the others agree is part of reading it) and `CLAUDE.md`'s
  "structural problems always raise". **P10 applies the same rule to `W` and `C`: to raise with the maintainer at
  the pause after P11**, since it bears on O-18's "nothing more" (the plan's 2.3 step 1 leaves the sample's sizes to
  the checkers). P10's unsummed readout strings need their `K` forms too (added to the plan's P10).

### P7: Engine C, the frame-form dot grade (2026-11-20, to 05:12)

- **Rounds:** one on the build (`P7_REVIEW_r1.md`: SIGN OFF, one minor finding), after the design (above). Handover
  `P7_HANDOVER.md`.
- **What was built:** the dot steps `xi_dot_step`, `mu_dot_step`, `nu_dot_step`, `eta_dot_step` (written for both
  direction forms, for P8), `Jp_step`, `Jp_norm_sq_step` (keeps `K + C`, no half), `identity_step`; start values with
  `[K] + C` after `W`. Rows: `xi_U_dot`, `mu_R_dot` (through `R`, zeros), the private frame nodes
  `_frame.nu_L_dot`/`_frame.eta_D_dot` read by `Jp`, the public route rows `nu_L_dot`/`eta_D_dot` over them,
  `Jp_norm_sq`; the `Jp`-seeded twins derived from `_JP_PATH` (`JtJp` the twin of `g`); `_base`, `_letter`,
  `_table_name` for a name's letter; `_read_K` and the structural `K` check (every `V[i]`, `H[i]` exactly `K + C` plus
  its core axes; a held `Jp` with the call's `K`); `HeldVariables` 25 slots.
- **The reviewer's checks:** the seven new names against the reference per direction (8.6e-15); `Jp` against a dense
  tangent tensor built from its definition and probed by its oracle (all four dot recurrences and bases, independent
  of engine and reference); the identities per direction; a held, changed public `nu_L_dot`/`eta_D_dot` never feeds
  `Jp`; a size-1 `K` on `H[1]` now raises; floor 52 passed. The implementer's 17 mutants all killed.
- **Departures accepted:** `Jp_norm_sq` with `carries_K=True`; the `K` check counting total axes (catches a wrong
  core rank); the bridge checking the frame dots through `Jp` and `JtJp`.
- **Minor, made by Claude:** without `variations`, `JtJp`'s error named a chain of six private nodes; the `JtJp` twin
  now carries `needs={'variations'}` itself (the docstring table and the twin test state the one exception), and a
  test pins the message (`'JtJp' needs the argument variations, which was not given`). Engine tests 48 passed,
  doctests 2.
- **Gate:** the implementer's full gate at `-n 3`, every step PASS (suite 1,722 passed, 5 skipped, 1 xfailed; module
  doctests 314; 48 pages; 14 examples; Sphinx 0 warnings). **Memory: peak 10.4 GB, 1.9 GB left at the lowest.** If it
  falls further the gate moves to `-n 2`.
- **P8's design note** (`P8_DESIGN_NOTE.md`) was written while P7 was reviewed; it goes to the reviewer next.

### P8 designed (2026-11-20)

- The implementer's note (`P8_DESIGN_NOTE.md`, written while P7 was reviewed) and the reviewer's critique
  (`P8_DESIGN_REVIEW.md`: sound, one must-fix for the corewise derivation). The reviewer confirmed the tilde-dot
  terms (`mu_tilde_dot`'s six = 3 places of the direction x 2 of the seed, both-in-the-leaf being `xi_tilde_dot`;
  `H_tilde_dot`'s nine) and the cost (12 readouts against `6(n + 1)` contractions per site for a line), and derived
  the bridge to the old `corewise_hessian` (forward `omega^2 r / t!`).
- **Ruled by Claude** (written into the plan's P8 and section 2.4): the rows and O-12 routes as proposed; one shared
  `K` for the four direction operands, a mismatch raising; the corewise graph derived at import with the rename
  applied to `needs` too (Finding A: else `JtJp` is unreachable there), two import-time assertions, the twin path
  derived through the rename; one table of direction operands for P11's masks; `Kp` tested against the reference
  (exact), not a central difference of `g`.
- **For the maintainer at the pause after P11:** on a corewise call the direction argument is `core_directions` and
  `variations` raises (option (a) of the review; both agents recommended it, against Claude's first lean to accept
  `variations` there). Reason: at `(B, G, G, G)` the frame form of a direction is its raw form, so accepting both would
  run two identical families of raw dots when both are given (the exact Hessian on a corewise geometry, lift = the
  identity); the least-squares oracle passes the direction as `core_directions` on a corewise frame, told from the
  frame's structure, not the geometry. No decided row changes in letter (4.5, 8.2, 8.3); it is a rule on one call
  type of a public backend function, like O-13. Reversible inside P8.

### P8: Engine D, the raw dots, the tilde-dot grade, `Kp`, the corewise call (2026-11-20, to 06:33)

- **Rounds:** one on the build (`P8_REVIEW_r1.md`: SIGN OFF, one minor), after the design (above). Handover
  `P8_HANDOVER.md`.
- **What was built:** the tilde-dot steps (`xi_tilde_dot`, `mu_tilde_dot`, `nu_tilde_dot`, `eta_tilde_dot`, a short
  node with the new literal `'JWCa,KCaib,JWCb->WKCi'`, `V_tilde_dot`, `H_tilde_dot` of nine readouts); `STRINGS` at 32;
  the raw dot rows, the O-12 routes, a raw route for `Jp`, the tilde-dot rows and `Kp`; one shared `K` checked across
  `V`, `H`, `D_dot`, `L_dot`; the corewise `(B, G)` graph derived at import (drop, rename over names, reads and needs,
  twins through the renamed path) with import-time invariants and a reachability loop; `variations` refused on a
  corewise call (one `if`); `_DIRECTION_MASKS`, the one table of direction operands for P11; `HeldVariables` 40
  slots. Tests: `tests/test_sampling_corewise.py` (8) and additions elsewhere; `Kp` against the reference exactly;
  the bridge to the old `corewise_hessian`.
- **The reviewer's checks:** `Kp` against its own dense second derivative (the `eps^1` coefficient of
  `Phi(x + eps xdot + e) - Phi(x + eps xdot)`, exact at `2d + 1` points), 8.8e-15, covering the raw dots, the tilde-dot
  grade, the short node and the nine readouts; the corewise call at `(D, L)` against the manifold call; corewise `JtJp`
  as `g` with `Jp_k` held; each sweep exactly `d` times on corewise `('z', 'g')` and `('Kp',)`; the refusals (a `K`
  mismatch, O-13, `variations` on corewise). The implementer's mutants: 20 non-equivalent, all killed.
- **Minor, made by Claude:** `_corewise`'s import-time invariants were `assert`s (stripped by `python -O`); now `raise
  RuntimeError`, its test updated; the module imports under `-O`. Corewise, graph and variable tests 42 passed.
- **Gate:** the implementer's full gate at `-n 3`, every step PASS (suite 1,734 passed; module doctests 314; Sphinx 0
  warnings). Memory: the summed peak RSS of the run rose to 13.4 GB (it counts pages shared between workers more than
  once); the lowest MemAvailable stayed 1.9 GB, the real margin. P8's own files peak at 0.68 GB in one process (the
  reviewer). Rule from here: if the lowest MemAvailable of a gate falls below 1.5 GB, the gate moves to `-n 2`.
- **P9's design note** (`P9_DESIGN_NOTE.md`) was written while P8 was reviewed; to the reviewer next.

### P9 designed, and split into P9a and P9b (2026-11-20)

- The implementer's note (`P9_DESIGN_NOTE.md`, written while P8 was reviewed) and the reviewer's critique
  (`P9_DESIGN_REVIEW.md`: sound). The reviewer confirmed the founding fact with tails (apply's value is `<mu[d], 1>`,
  so every side reaching bond `d` is the plain recurrence started from `s (x) 1`; `close_step` must sum, `z` closing with
  `rL_d` ones and `Jp` with `rR_d`), and that `H_tilde_step` on the dot and tilde-dot arguments gives exactly the three
  surviving terms of apply's `H_tilde_dot`.
- **Ruled by Claude** (written at the end of the plan's P9): the design as proposed with the reviewer's requests
  (twin paths keyed with `_GRAPHS`; invariants raise; scatter-add for entries' `V`, low memory, duplicates tested, a
  static "keep `W`" option for P10; a `Vc` given to apply/entries accepted, as any unread argument). **P9 split in two
  commits in place, P9a (apply, both call types, the shared machinery) and P9b (entries):** granularity, not scope or
  order.
- **For the maintainer at the pause after P11:** (1) on apply and entries, do `xi_tilde`, `mu_tilde`,
  `xi_tilde_dot`, `mu_tilde_dot` return zeros, or are they unknown names whose message says they are zero? (Default
  built: unknown, both agents' lean; zeros rows would need a false shape read or a new "zero" driver. The reference
  module returns zeros, so the engine's names and the reference's differ by exactly these, and a test says so.)
  (2) A public reader of a call's names, so generic code can discover them (a new public name; private meanwhile).

### P9a: Engine E, the apply kind on both call types (2026-11-20, to 07:59)

- **Rounds:** one (`P9a_REVIEW_r1.md`: SIGN OFF, no findings). Handover `P9a_HANDOVER.md`.
- **What was built:** the rows split into shared plain, residual, dot and `Jp_norm_sq` blocks plus each kind's own;
  apply's rows (`z` and `Jp` by `close_step`, summing over the end bond: `rL_d` ones for `z`, `rR_d` for `Jp`;
  `nu_tilde`/`nu_L_tilde` as `nu_step` started from the node `w_tilde`; `eta_tilde` by `eta_step`; the tilde-dots
  reusing `nu_dot_step`, `eta_dot_step`, `H_tilde_step`; one-term readouts `eta_top_step`, `eta_dot_top_step`,
  `V_top_step`, `H_readout_step`; no new literal); `_deps` and `start_node`; `_JP_PATHS` keyed `(kind, call)`;
  `_corewise(rows, kind)` and a new `_manifold(rows, kind)`, both raising on a broken invariant; `K` on the carry of
  `nu_step`/`eta_step`; every whole step takes `kind`; the zero names off apply's graphs (`_ZERO_NAMES`, a message
  saying they are zero there); a private `_names(kind, call)`; a `Vc` given to apply accepted and unread. Tests:
  `tests/test_sampling_apply.py` (13) incl. the names comparison against the reference (the difference exactly the
  four zero names), tails with `rL_d != rR_d`, the corewise rename carrying a start node.
- **The reviewer's checks:** on a frame with tails `rL_d = 2`, `rR_d = 4` at both ends, `()`, `(3,)` mixed sizes,
  `(1, 2)` mixed modes, nonuniform weight, `K = (2,)`: `z` against the full dense contraction, `g` against the
  transposed dense Jacobian (pins `close_step` and the seed start), `Jp` against the dense tangent tensor, `Kp`
  against its dense second derivative; worst 8.5e-15 (8.1e-15 on the floor). **The probe graphs unchanged row for
  row** (both call types, 97 rows, all six fields, against `HEAD` by `git archive`). A wrong-shape `Vc` changes
  nothing. The implementer's 20 mutants killed (two equivalent for apply, now killed by a test on synthetic rows the
  reviewer judged meaningful).
- **Gate:** the implementer's full gate at `-n 3`, every step PASS (suite 1,755 passed; 1,367 s). **The lowest
  MemAvailable fell to 1.56 GB**: from P9b the gate runs at `-n 2` (`BRIEF.md` updated).

### P9b: Engine E, the entries kind (2026-11-20, to 09:20)

- **Rounds:** two (`P9b_REVIEW_r1.md` NOT YET, one should-fix from Claude's ruling; `P9b_REVIEW_r2.md` SIGN OFF;
  `P9b_RESPONSE_r1.md`; handover `P9b_HANDOVER.md`).
- **What was built:** `xi_gather_step` (the leaf as `D[..., index]`, serving the dot leaves too); `V_scatter_step`
  (scatter-add, `W x n` memory, duplicates added by `np.add.at` / `.at[].add`, a static `keep_W` for P10 tested at step
  level); entries derived from apply by one declared table `_ENTRIES_STEPS` (every read of `w` becomes `index`, the
  scatter also reads `D` for the mode size, the short nodes gone at `order = ()`); entries' twin path through the full
  `eta_tilde`; six graphs; the index out of the dtype count and never cast; the zero names off entries' graphs; the
  `NotImplementedError` gone (all three kinds built). Tests: `tests/test_sampling_entries.py` (11+), the names
  comparison over three kinds, the bridge incl. the migration of entry 120 (`t3_entries_derivatives` against apply
  at `affine_jet(index_to_one_hot(index, shape), pp)`).
- **Ruled by Claude, built in round 1 the reviewer's way:** `'w'` means its definition: on entries the one-hot vectors
  `e_index` (a row reading `index` and `D`, through `index_to_one_hot`, cast once, built only when wanted), on probe and
  apply the sample as given. `w` is not a `HeldVariables` slot (the input; a stated exception, recorded in the plan).
- **The reviewer's checks:** dense constructions sharing no code with the engine or the reference, on tails with
  unequal end bonds, a Tucker rank 3 over a mode of size 2, `W = 6` with duplicate indices in every mode, `K = 2`:
  `z = T[index]`; `g` against the transposed dense Jacobian at the entries (pins that duplicates add); `Jp`; `Kp`; the
  corewise call; `keep_W` at step level. Worst 6.8e-15 on both envs. The implementer's 12 non-equivalent mutants
  killed (one first survived, the mode size as `max(N, n)`, killed by a test with a Tucker rank above the mode size).
- **Gate:** the implementer's full gate at **`-n 2`**, every step PASS (suite 1,770 passed; 1,804 s; lowest
  MemAvailable 4.14 GB). After round 1: the affected tests and doctests single-process on both envs.
- **P10 designed meanwhile** (`P10_DESIGN_NOTE.md`, `P10_DESIGN_REVIEW.md`; settled in the plan's P10): the reviewer
  found that comparing a multi-axis `W` against the unstacked engine at the same `W` would let a `W` bug cancel;
  multi-axis `W` is compared against the engine with `W` flattened. The `_check_stacks` extension of P7's check to `W`
  and `C` is pending the maintainer (O-18).

### Paused by the maintainer mid-P10 (2026-11-20, 11:31)

"I need to use this laptop for a meeting in 30 minutes from now. Please pause and we will resume in a couple hours.
You can let any currently running gates complete." P10 is built; the implementer's gate (at `-n 2`) was on its last
step, Sphinx, and was allowed to finish; the implementer was told to start nothing new and to write
`P10_HANDOVER.md` at whatever stage it is. The reviewer is idle. P10 is not reviewed or committed. Nothing else runs.

**The questions gathered for the maintainer at the pause after P11** (from the entries above): (1) the corewise call
takes `core_directions` and refuses `variations` (P8; reversible); (2) on apply and entries, are `xi_tilde`, `mu_tilde`,
`xi_tilde_dot`, `mu_tilde_dot` zeros or unknown names (built: unknown, P9); (3) a public reader of a call's names (P9);
(4) the structural stack checks of P7 (`K`) and P10 (`W`, `C`) under O-18 ("the engine checks what it needs to run,
and nothing more"; removable).
P10 finished before the pause: the implementer's full gate at `-n 2` PASS on its final tree (suite 1,782 passed, 5
skipped, 1 xfailed; 1,870 s; lowest MemAvailable 3.81 GB; module doctests 314; Sphinx 0 warnings); `P10_HANDOVER.md`
marked COMPLETE. Not yet reviewed or committed. One departure, accepted by Claude as minor: each of the six unsummed
readout steps has two bodies (`if keep_W:` and the summed one), each with its literal at its own `jet_contract` call:
the interpreter test's guard requires every `J` literal to be a call's first argument, and literals at call sites are
the house rule (a first build that named the literals at the top of the step failed it). Nothing runs.

### Resumed (2026-11-20, 14:10); P10: Engine F, the stacks (to 14:14+)

- **Resumed** at the maintainer's word ("The laptop is free. Let's continue").
- **P10, rounds:** one (`P10_REVIEW_r1.md`: SIGN OFF, no findings). Handover `P10_HANDOVER.md` (COMPLETE).
- **What was built:** six unsummed readout literals (three `V`, three `H`, `W`-first); `sum_over_samples=False` wired
  through `keep_W` to block rows only (read off the shape table, `_Jp` twins included), joining the memo key; `g`,
  `JtJp`, `Kp` as `W + [K] + C` unsummed; `f`, `r_block_sumsq`, `Jp_norm_sq` always summed; `_check_stacks` (one
  function, one call, one test class; exact tuple equality of `C` on every store core and `Vc`, of `W` on every
  `w[i]`; removable, pending the maintainer under O-18). Tests: `tests/test_sampling_stacks.py` (11): the covering
  design against the unstacked engine, multi-axis `W` against the engine with `W` FLATTENED, a maximal case per kind,
  held stacked variables, the unsummed tier (a multi-axis `K`; a non-square `W` with an off-diagonal sample), a thin
  reference anchor, the stack check, the dtype matrix over four backends (jax under `jit`).
- **Departure, accepted by Claude:** each unsummed readout step has two bodies, each literal at its own `jet_contract`
  call (the interpreter guard; the house rule). The reviewer: the right price, since a literal held in a name would be
  hidden from the AST sweeps.
- **The reviewer's checks:** the flattening is real (`(2, 2)` against `(4,)`, different expanded strings and offsets;
  the `W`-offset mutant killed); on `C = (2,)`, `K = (2,)`, a non-square `W = (2, 3)`, probe at `(1, 1)` mixed jets and
  apply at `(2,)`: `z` and `f` per frame element against its dense oracle; each sample's unsummed `g`, `JtJp`, `Kp`
  equals the summed call on that sample alone at diagonal and off-diagonal samples, the blocks summing back; the
  stack check raising on a size-1 `C`, a size-1 `W` axis, a `Vc` without `C`, and passing a consistent size-1 stack.
  Worst 4.1e-16 (4.8e-16 on the floor). The implementer's 13 mutants killed.
- **Gate:** the implementer's full gate at `-n 2` before the pause, every step PASS (suite 1,782 passed; 1,870 s;
  lowest MemAvailable 3.81 GB).
- **P11's design note** (`P11_DESIGN_NOTE.md`) was written while P10 was reviewed; to the reviewer next.

### P11 designed (2026-11-20)

- The implementer's note (`P11_DESIGN_NOTE.md`, written while P10 was reviewed) and the reviewer's critique
  (`P11_DESIGN_REVIEW.md`: the structure right, three findings, one change).
- **Ruled by Claude** (written into the plan's P11), all the reviewer's: **A** the shape mask on `N` for every
  Tucker-shaped input and `V`-type block, and the sample's `N` masked on entry (jet axes untouched): "the sample needs
  no mask" held only because `D` carries the shape mask, while `w` also meets `V`, `D_dot`, `Vc` and the `V` readouts,
  whose masks are rank masks; **B** held variables masked on entry by their table's masks (a held `z` or `r` with
  garbage padding would give a wrong `f`; padding is not a value); **no `layout` static** on the whole steps, which infer
  the layout from their operands (the house rule); **C** a concrete numpy `index >= N_i` raises `IndexError` on uniform
  as on ragged (the equivalence contract; one comparison, as for `index_to_one_hot`), a jax index reading a zero,
  documented. Tests: garbage everywhere on a covering design, negative controls by mask family, the uniform maximal
  case unsummed.

### P11: Engine G, the uniform layout and sharding over `W`; the additive phase closes (2026-11-20)

- **Rounds:** two (`P11_REVIEW_r1.md` NOT YET, two should-fix; `P11_REVIEW_r2.md` SIGN OFF; `P11_RESPONSE_r1.md`;
  handover `P11_HANDOVER.md`), after the design (above).
- **What was built:** the engine tells ragged from uniform by `is_ndarray`, then the element count (6 a frame, 4 a
  point), a traced mask raising; one table of entry masks `_UNIFORM_MASKS` (each store, `V`/`H`/`Vc`, `D_dot`/`L_dot` by
  the masks of the object they are defined at, the directions by the frame's `D`/`L` masks; the shape mask on `N` for
  every Tucker-shaped input, the sample (jet axes untouched) and probe's `y`; held variables by their own table; start
  values with ones on the real slots of their end bond); every output masked on exit; the engine packs a ragged sample
  itself (`len_J`); a numpy index outside `[-N_i, N_i)` raises on uniform as on ragged; the whole steps infer the layout
  from their operands; `_check_stacks`' uniform branch (the masks' `C`, the packed sample's `d` and `N`);
  `pack_vectors`/`pack_if_ragged` with `len_J=0` (a `W` mismatch now a named `ValueError`). Tests:
  `tests/test_sampling_uniform.py` (14+: the covering design of 81 uniform calls with garbage in every masked slot,
  every name per element and per site against ragged, the real block read off the ragged shape and the rest exactly
  zero; the maximal case per kind unsummed; packed jet zeros as data; held garbage; the index; `Kp` at TT slack with
  permanent negative controls; the uniform stack check per family); dispatch (a jitted uniform call; jaxpr size equal
  at `(2,)` and `(6,)`; a second eager call compiles nothing); sharding (`g` and `Kp`, 4 programs, no all-gather).
- **Departures accepted:** every output masked on exit; the program-size test counts jaxpr lines (the HLO text shifted
  with what the process had compiled before); a user who jits the engine on raw `.data` keeps the masks and `shape`
  static (docstring).
- **Review round 1** (the reviewer's checks: a TT-slack point built by the library with garbage in every padding slot,
  every name equal to ragged with exactly zero padding, `z` and `f` against its dense oracle to 1e-12; all 98 holdable
  variables held with garbage padding give the clean call; sharding on 4 virtual devices, zero all-gathers; 25
  mutants, 24 killed). **Finding 1, a silent wrong answer:** the entries index's leading `d` was never checked; on
  uniform an index of shape `(1,) + W` broadcast to every mode and returned `T[i, i, i]`, and ragged silently ignored an
  extra row (since P9b). **Fixed, ruled by Claude:** one structural check on both layouts, outside `_check_stacks` (the
  sample's length `d`, which the engine needs to read the sample at all). Finding 2: a mutant returning the uniform
  one-hot `'w'` as a tuple survived; an `isinstance` assertion kills it. The masking's three redundant layers (the
  sample's `N` mask, the start mask, the exit masks) judged sound by the reviewer: they protect only the padding, each
  pinned alone by a permanent test; kept.
- **Round 2:** SIGN OFF; the reviewer's 26 mutants over two rounds all caught.
- **Gate:** the implementer's full gate at `-n 2` before round 1, every step PASS (suite 1,799 passed; 1,936 s; lowest
  MemAvailable 3.69 GB; module doctests 315; Sphinx 0 warnings). **Claude's run on the final tree** (after round 1's
  engine change): below.
  Suite 1,800 passed, 5 skipped, 1 xfailed (1,886 s at `-n 2`; lowest MemAvailable 4.0 GB); module doctests 315.

### The engine sub-arc P5 to P11 is complete: the second planned pause (2026-11-20, 17:13)

The additive phase is closed: the engine `sampling_variables` computes every variable of the three kinds, on manifold
and corewise calls, ragged and uniform, with jets in any number of variables, the stacks `W`, `K`, `C`, and
`sum_over_samples`. Nothing of the old sampling layer is rerouted or deleted yet (P12 on). Summary and the questions
for the maintainer: `dev/HANDOFF.md`, START HERE.

### The maintainer's answers at the second pause (2026-11-20, evening)

In `../TARGET_DECISIONS.md`, the note "The second planned pause": `sum_over_samples` aligned with his intent, the
oracle not asking for unsummed blocks; corewise first-class (P12, P13, P17/P19 additions approved; a check-in between
P19 and P20 on `t3_corewise_frame`); re-gauging recommended and filed in the handoff's backlog; the backlog to be
considered after Parts 1 and 2 (a possible Part 3); **the zero tilde names stay unknown names; no public reader of
names for now; the structural stack checks are kept** (O-18 read to include agreement of a group across the operands
that carry it). Question 1 (the corewise call refusing `variations`) answered the same evening: kept. A writer (the implementer) and a
critic (the reviewer) are updating the plan for the corewise surface: the critic's round 1 NOT YET (the bridge to
external optimizers must go through the geometry: shared factors, the regularizer, the minibatch scale), in revision.

### The plan updated for the corewise surface (2026-11-20, evening)

Writer the implementer, critic the reviewer (`PLAN_UPDATE_corewise.md`, `_REVIEW.md` NOT YET, `_RESPONSE_r1.md`,
`_REVIEW_r2.md` AGREED with one amendment, folded in). The plan now has: in P12 a corewise forward derivative
(`t3_*_corewise_jacobian` or `_jvp`, the maintainer's choice, O-48) with uniform twins, its primary check a numpy dense
oracle (`J x_dot = dense_probe(T_dot)`, `T_dot` the sum of the `2d` one-core-replaced tensors), `jax.jvp` second, a
structural step count; `sum_over_samples=True` on the six corewise Hessians (none on the forward derivative: it has no
sum); in P13 frontend methods for both on `TuckerTensorTrain` (15 methods) and `UniformTuckerTensorTrain` (12),
directions as the pair of core tuples only (O-49; a `TuckerTensorTrain` holding a direction is a trap, its `+`
concatenates ranks), the Hessian checked by symmetry and a polynomial-exact difference quotient; in P17 the bridge to
external optimizers THROUGH THE GEOMETRY (the critic's main finding: a raw query flattened by hand unties shared
factors, misses the geometry's regularizer and the minibatch scale), a helper `point_value_and_gradient(geom, oracle,
batch, x_cores, regularizer=None)` proposed (O-50) returning the pair, on a shared geometry the gradient the mean over
each group replicated (the projection onto the tied subspace; the sum would overstate the slope by the group size,
caught by a tied-direction gradient check), four suite tests; in P19 the docs steering corewise users; in P20 the check
of `t3_corewise_frame`, with a check-in with the maintainer between P19 and P20. The counts unchanged (removal table
166 / 30 / 5 / 131). Also: the plan's three "pending the maintainer" notes (P7/P10 stack checks, P9 zero names)
resolved by his answers.
O-49 (the pair of core tuples only) and O-50 (the helper adopted) approved by the maintainer the same evening; O-48
waits on a signature prototype he asked for.

### O-48 decided; the sub-arc P12 to P15 begins (2026-11-20, late)

O-48: `_corewise_tangent` (the maintainer, after a signature prototype). Substituted for `_jacobian` in the plan
(the writer tidies the candidate text in 11.5 as part of P12's preparation). The handoff's START HERE brought up to
date. P12 starts design-first.

### P12 designed; split into P12a and P12b (2026-11-20, late)

`P12_DESIGN_NOTE.md`, `P12_DESIGN_REVIEW.md`. The reviewer caught that the implementer's plan tidy for O-48 had
deleted sections 11.3 and 11.4 (being restored from `HEAD`). **Ruled by Claude:** a transpose's residual with a stack
`K` goes through a `JtJp` route from a held `Jp` (`K` read off it; the contract "holding a vector as `Jp` gives `J^T` of
it" stated in 6.5 and `HeldVariables`); the bridge's `z` rows use the order-0 derivative kinds or `dense_*`, with
`sampling_variables` patched to raise while the old side runs; the `utv_*` twins mask variations with the frame's
masks; **a ragged `ww[i]` with a mode axis of size 1 against `N_i > 1` raises** (the structural rule the maintainer
endorsed for the stack checks; else a new silent broadcast); the docstrings point repeated calls to `held=`. **Split
in place: P12a** (the engine addition, the ragged functions, call sites) **then P12b** (the uniform twins, the corewise
tangent, the Hessian's `sum_over_samples`). The implementer writes the rulings into the plan and builds P12a.

### P12a, review round 1 (2026-11-20, late)

Built (handover `P12a_HANDOVER.md`): the held-`Jp` route for `JtJp`, the ragged mode-size check, the 15 ragged
functions as engine calls, the bridge guard, the call sites; five test comparisons moved from bitwise to rounding
(engine against old kernel; padding and garbage checks still exact). The implementer found that a size-1 rank axis on
`V`, `H`, `D_dot`, `L_dot` broadcasts silently today; **ruled by Claude: the engine checks their trailing shapes
against the stores** (the maintainer's structural rule). `P12a_REVIEW_r1.md` NOT YET: that check (must), tests of the
corewise transposes with a residual carrying `K` (three mutants survived), the full gate rerun. In round 1.

### P12a committed (2026-11-21)

Round 1 (`P12a_RESPONSE_r1.md`): the core-size check (`V`, `H`, `Vc`, `D_dot`, `L_dot` against the stores by exact
shape, both layouts, inside `_check_stacks`; `TestTheCoreSizesOfTheDirections`), the transposes with a residual
carrying `K` tested (`TestAResidualWithAStack`), the full gate at `-n 2` PASS (suite 1,816 passed, 5 skipped, 1
xfailed; lowest MemAvailable 2.6 GB, down from 4.6: watch; if below 1.5 GB the gate goes to `-n 1`). Round 2
(`P12a_REVIEW_r2.md`): SIGN OFF. For P20: `test_uniform_curvature`'s direction-mask control now runs on the old kernel
and is deleted with it (P11's `TestKpAtTTSlack` is the lasting control). Next: P12b (the uniform twins' signatures and
mirroring, `*_corewise_tangent`, the Hessians' `sum_over_samples`, the automatic-differentiation tests).

### P12b committed; P12 complete (2026-11-21)

`P12b_HANDOVER.md`, `P12b_REVIEW_r1.md` (SIGN OFF, no findings, one round). Built: `t3_*_corewise_tangent` and
`ut3_*_corewise_tangent` (O-48; one engine call wanting `Jp` at the corewise point; bitwise equal to the old
substitution through `tv_*`, against the dense oracle 5.9e-16, `jax.jvp` 7.8e-16); `sum_over_samples=True` on the
corewise Hessians; the 18 uniform twins on the ragged signatures, mirroring the sample's form; the call sites moved;
`test_jit_the_sampling_functions`. The reviewer: the tangent against its own one-core-replaced construction, the
unsummed Hessian per sample, the 18 twins with garbage everywhere (padding exactly zero, judged by ragged shapes), the
corewise step count (9 rows, no frame rows), 5 more mutants killed. Gate at `-n 2`: suite 1,824 passed; lowest
MemAvailable 3.9 GB. For P20: the `_by_old_order` test adapter. Next: P13 (the frontend methods), design-first.

### P13 designed; split into P13a and P13b; one question for the maintainer (2026-11-21)

`P13_DESIGN_NOTE.md`, `P13_DESIGN_REVIEW.md`. Ruled by Claude: split P13a (merged signatures, the six new corewise
methods per class, `check_sample` in every method taking a sample, the transpose call sites; the `*_derivatives`
methods kept) then P13b (their removal, every caller, the examples, pages, `chunking.md` and its 13 links); a
`TypeError` for any `core_directions` that is not a pair of core sequences (or two supercores); `np.asarray` on a
non-array entries index; a test pinning the derivative conversion against an exact derivative (survives P20); a
`d = 3` test that swapping `ww` and the residual changes the result. **Asked of the maintainer (finding A of the
review):** old positional calls `x.probe_corewise_transpose(ztildes, ww)` and `probe_ambient_transpose(ztildes, ww)` would
silently swap residual and sample (same shapes for probe); recommended: `residual` and what follows keyword-only on
the frontend transposes and corewise methods, plus a CHANGELOG upgrade note. P13a is being built meanwhile.
Finding A decided by the maintainer: not keyword-only; documented by right/wrong doctests and the CHANGELOG note
(`TARGET_DECISIONS.md`, the note "Old positional calls to the probe transposes").

### P13a committed (2026-11-21)

`P13a_HANDOVER.md`, `P13a_REVIEW_r1.md` (SIGN OFF, one round). The merged frontend signatures on the four classes; the
six new corewise methods per tensor-train class (`*_corewise_tangent`, `*_corewise_hessian`); `check_sample` in every
method taking a sample; the directions guard; `np.asarray` on entries indices; finding A's right/wrong adjoint doctests
on the four probe transposes; 98 transpose calls rewritten (every one checked by the reviewer); a new
`tests/test_frontend_sampling.py`. Gate at `-n 2`: suite 1,835 passed; lowest MemAvailable 3.9 GB. Carried into P13b:
five entries methods into the nested-list test; the derivative-conversion pin (independent of `affine_jet` and
`taylor_to_derivatives`, outliving P20).

### P13b committed; P13 complete (2026-11-21)

`P13b_HANDOVER.md`, `P13b_REVIEW_r1.md` (SIGN OFF, one round). The 24 `*_derivatives*` methods removed from the four
classes; 162 test calls through a test-only helper `tests/_derivative_data.py` (the merged method at `affine_jet`, then
`taylor_to_derivatives`; the transposes' residual scaled by `t!`), derived and checked independently by the reviewer;
the doctests of `optimizers.py`, `fitting.py` and two pages, and the six examples' dense ground truth moved;
`docs/chunking.md` removed with its links; the nested-list test on all 13 entries methods; the derivative-conversion
pin (independent of `affine_jet` and `taylor_to_derivatives`, survives P20). `test_probe_derivatives.py` stays until
P20 (it tests the old kernels). Gate at `-n 2`: suite 1,836 passed; lowest MemAvailable 3.7 GB. **For the maintainer
(the third pause) and Part 2:** `fit_hilbert_from_apply_derivatives.py` takes about 11 s instead of 5.8 s (the engine at
the jet on every minibatch, in place of the old dedicated derivative kernels; the plan accepts Part 1 being slower,
section 1); the stochastic examples differ from the old tree in the last digits (MC-SGD amplifying rounding; the
reviewer checked the operator agrees to 4e-16). P19: prose mentions of the removed methods and of `chunking.md` remain
in two pages, code comments and the glossary.

### P14 designed (2026-11-21)

`P14_DESIGN_NOTE.md`, `P14_DESIGN_REVIEW.md`. `point_value_and_gradient` stays in P17. Ruled by Claude: `keep=` on
`sampling_variables` (the held record plus the kept names the call computed anyway; extra return only when given;
unknown names raise) rather than a separate `computed_names`; a public frame reader `frame_form(frame) -> (layout,
call)` (not `structure`, which the tensor-train classes use); on a corewise frame given both `variations` and
`core_directions` (the exact Hessian, lift = identity), pass `core_directions`, require the two to be the same arrays,
else raise; `take` sets `minibatch_scale = n / len(indices)` (documented as a draw); `pack_sample`/`pack_data`
unchanged until P16; `f_resolution` from unscaled values times the scale once. **Asked of the maintainer:** the two new
public names (`keep=`, `frame_form`); P14 is built with them meanwhile. For P16: `eps|rho|` in the resolution; a test
that `held` is not carried across MC-SGD draws.

### P14 built; review round 1 (2026-11-21)

Built (`P14_HANDOVER.md`): `frame_form`, `keep=`, `LeastSquaresBatch`, `LeastSquaresOracle` (query, check_batch,
n_measurements, take), `pack_batch`, the `Oracle` protocol text, reachable from `t3toolbox.fitting`; the P3 test that
bans the checkers in the backend now allows them inside `check_batch` only (the reviewer agrees). Gate at `-n 2`: 1,851.
`P14_REVIEW_r1.md` NOT YET: `y_block_sumsq` and `f_resolution` summed a packed probe `y`'s padding (silent; the
reviewer found it with garbage padding; must fix), float32 `eps` untested, the P3 test to cover every backend module.
In round 1.

### P14 committed (2026-11-21)

Round 1 (`P14_RESPONSE_r1.md`): `query` unpacks a packed probe `y` with the frame's static shape before reducing (the
reviewer's garbage case now exact), a float32 `eps` test, the P3 test over every backend module. `P14_REVIEW_r2.md`:
SIGN OFF. Gate at `-n 2`: suite 1,852; lowest MemAvailable 3.5 GB. **Still open with the maintainer:** the names
`keep=` and `frame_form` (committed as built; a one-line rename each; to be raised at the third pause if not answered
before). Next: P15 (the backend geometry's `frame(x)` returns `(frame, centers)`; small), design-first.

### P15 designed (2026-11-21)

`P15_DESIGN_NOTE.md`. The slice is small, so Claude ruled without a separate design review (the build is reviewed as
usual). The plan's call-site list is complete and exact after P12 to P14. Ruled: the uniform centers are the pair of
supercores `(Vc_sc, Hc_sc)` (the ragged pair's mirror; `ufv_centers_consistency_residual`'s form); the corewise
geometries return `None`; the aligned padded frame keeps returning a frame only (it has no centers). For P16: at a
padded-start frame the oracle runs without `Vc`.

### P15 built; review round 1 (2026-11-21)

Built (`P15_HANDOVER.md`): `frame(x)` returns `(frame, centers)` on the four backend geometries; the 3 library and 27
test call sites unpack; gate at `-n 2` 1,854. `P15_REVIEW_r1.md` NOT YET: the implementer's "equivalent" surviving
mutant (`point_norm_sq` from the centers) is NOT equivalent. Each of the 2d centers reproduces X, so the tangent
(frame, centers) is 2d X, and the mutant makes the regularizer 2d times too large; it survived because the only uniform
test of `point_norm_sq` was circular. In round 1: a dense |X|^2 test on all four geometries; the P16 suggestion
retracted; docstrings calling the centers "the point's own coordinates" corrected.

### P15 committed; the sub-arc P12 to P15 complete; the third planned pause (2026-11-21)

Round 1 (`P15_RESPONSE_r1.md`, tests and docstrings only): `TestPointNormSq` on all four geometries (manifold against
dense |X|^2, corewise against the raw cores' sum of squares, uniform with garbage padding; the centers mutant dies);
the P16 suggestion retracted; the three `frame` docstrings say (frame, centers) read as a tangent is 2d X. Touched
files on both envs, module doctests and Sphinx pass (the full gate passed on round 0's code). `P15_REVIEW_r2.md`: SIGN
OFF. **The work stops here for the maintainer's third planned pause, before the switch (P16).**

### The third planned pause: `keep=` replaced by `computed_names` (2026-11-21)

The maintainer decided: drop `keep=` from the engine; add the pure query `computed_names(kind, frame, want,
**arguments)`; the oracle uses it to hold exactly what is free (`TARGET_DECISIONS.md`, the note "The hold list").
To be built as a small slice (P14b) before P16. `frame_form` under discussion with the maintainer.
P14b designed (`P14b_DESIGN_NOTE.md`; small, ruled by Claude without a design review): `computed_names` with the
engine's exact signature (`sample` accepted and unread), presence only, public names only, held names excluded, a
frozenset, the engine's errors; one private `_closure_facts` shared by the engine and `computed_names`, so they read the
same facts; the oracle passes one shared dictionary to both. Being built.
The maintainer keeps `frame_form` (structure inference, two strings) and approves a small `frame_shape` reader, added
in P14b. Next at the pause: Claude investigates why `fit_hilbert_from_apply_derivatives.py` got slower (5.8 s to
about 11 s): is it architectural (a limit at scale) or incidental (compilation, call overhead, einsum paths)?
The maintainer settled the four notes carried to P16 (`TARGET_DECISIONS.md`, "The notes carried to P16, settled"),
including one change: the aligned padded frame returns `(frame, centers)`, the old centers zero-padded (reversing the
P15 ruling).
The slowdown investigated (the maintainer's question at the pause): not architectural. Overhead per call at small
sizes; at scale the forward step's contraction order (row 5.5), and with it fixed the engine is 2-3x faster than the
old kernels. Numbers and scripts in `implementation/bench/`; `TARGET_DECISIONS.md` "Carried to Part 2".
Decided by the maintainer for P16: a FRESH implementer and reviewer from P16 to the end of Part 1 (the first pair
finishes P14b), with the required reading (methods of `TuckerTensorTrain` traced into the backend) and the goal of the
refactor stressed; P16 committed in parts on a local side branch `p16-wip`, squashed into one commit on
`graded-messages` at the end. `BRIEF.md` updated (the new sections "What we are doing", the fresh pair, the traps, the
side branch).

### P14b committed (2026-11-21)

`P14b_HANDOVER.md`, `P14b_REVIEW_r1.md` (SIGN OFF, one round). `keep=` removed; the pure public `computed_names` (the
engine's signature; one private `_closure_facts` shared with the engine, so the two cannot diverge); the oracle holds
the hold-list names `computed_names` says the call computes anyway (the rules run equal `computed_names(want)`
exactly: holding adds no work); the public `frame_shape` beside `frame_form`; the plan's text brought in line with the
decisions of 2026-11-21. Gate at `-n 2` before `frame_shape`: suite 1,857; after it, the oracle tests on both envs,
the module doctests and Sphinx. **The first pair of agents ends here; P16 starts with a fresh pair.**

### P16 started with a fresh pair (2026-11-21), on the local branch `p16-wip`

A fresh implementer and reviewer (Opus 5.5, high effort), briefed on `BRIEF.md` (the goal first, the required reading,
the traps, the side branch). Both orient first; the implementer then writes `P16_DESIGN_NOTE.md`. If the session is
lost, start a fresh pair on the same brief: the state is in the files.
The P16 reviewer's orientation found the plan's P16 ("the regularizer is not scaled") and P17 ("the minibatch scale
on f, g and the regularizer") in conflict. Ruled by Claude under O-21: the regularizer is not scaled (the batch's misfit
is scaled by n/b; the objective is the unbiased estimator (n/b) misfit_b + rho). P17's two sentences and the
TARGET_DECISIONS note of 2026-11-21 (item 2, which Claude had worded wrongly) corrected; the maintainer told.

### P16 designed (2026-11-21)

`P16_DESIGN_NOTE.md` (the fresh implementer), `P16_DESIGN_REVIEW.md` (the fresh reviewer: the design is right; four
changes; five tests added). Ruled by Claude: D2 (`padded_start(geom, oracle, batch, x0, x_old, ...) -> (frame,
centers, movements)`), D3 (backend `local_model(..., frame=None, centers=None)`; centers without a frame raise), D4
(`order` explicit in the display), D5 (the uniform minimal-rank guard placed by the frontend path), D6 (frontend
`LocalModel(geometry, backend)`, the frame derived), D8 (the lap rename and the floor in part 2); the alignment
projector memoized but `with_frame` at every pass; a direct additive uniform corewise retraction; the translation a
decorator, its custom-draw wrapper doing the full per-draw conversion; the review's five test additions, among them a
side-by-side script against the pre-P16 library. **With the maintainer:** D1 (a required geometry member, proposed
name `point_data(x_cores)`) and D7 (top-level exports: `LeastSquaresOracle`, `LeastSquaresBatch`, `local_model` in
place of the six factories). Part 1 (the geometry) is being built.
Decided by the maintainer: `from_t3` raises for `n` or `r` below a rank and points to `resize` (TARGET_DECISIONS,
"`UniformTuckerTensorTrain.from_t3` raises"). A small slice after P16. D1 and D7 still with the maintainer.
D1 and D7 decided by the maintainer: `point_data(x_cores)` (a required geometry member); the top level exports
`LeastSquaresOracle`, `LeastSquaresBatch`, `local_model`.

### P16 part 1 (the geometry), review round 1 (2026-11-21)

Built (`P16_part1_HANDOVER.md`): the corewise frame is the point, ragged and uniform; a direct additive uniform
corewise retraction (`ut3_corewise_frame` no longer used); `aligned_padded_frame` returns `(frame, centers,
movements)`, the old centers zero-padded. `P16_part1_REVIEW_r1.md` NOT YET (the geometry right; the reviewer's full
suite: 66 red ids, all corewise fits through the old kinds or model, except four of part 1's own): the two
`test_dispatch` member tests to update; the expected-red list replaced by the measured one; and a silent wrong answer,
`CorewiseGeometryOps.base_point` accepting the old four-store frame (the regularizer then counts G three times), to be
closed by a structural check through `frame_form`. In round 1.
Round 1 (`P16_part1_RESPONSE_r1.md`): the two `test_dispatch` tests and `TestUniformGeometryOps` fixed; a private
`_require_frame_form` (through `frame_form`) on the corewise and manifold `base_point`s and the manifold
`point_tangent`s, so the old four-store frame (and a uniform point given to the manifold geometry) raises; the measured
expected-red list. `P16_part1_REVIEW_r2.md`: four more ids red for the stated reason, missing from the list; added by
Claude (the reviewer signs off on that alone). **P16 part 1 committed to `p16-wip`.** The implementer's context is
near its limit: a fresh implementer for part 2, oriented on the brief, P16's design note and review and part 1's
handover.
P16 part 2 (the backend model and optimizers) started with a fresh implementer (the part 1 implementer's context was
near its limit); the reviewer stays. Brief: `BRIEF.md`, the required reading, P16's design note and review, the
rulings above, part 1's handover, reviews and expected-red list.

### P16 part 2 (the backend model and optimizers) built (2026-11-21)

`P16_part2_HANDOVER.md`: `LocalModel` (one `query` per method, literal wants, the exact hvp one `('JtJp', 'Kp')`
query), `local_model(..., frame, centers)`, `point_objective`, `point_data` on the four geometries, the optimizers on
`(geom, oracle, batch, x0, regularizer)`, a fresh model per draw, the regularizer unscaled, `padded_start` returning
centers, the floor `f_resolution + eps|rho|`, the `oracle` lap; `Problem` and its kin removed. Full suite: 105 red ids
measured (`P16_part2_EXPECTED_RED.txt`: 2 for D9, 13 display, 90 frontend; 43 of part 1's ids green). Side by side
with the pre-P16 library: 35 of 36 cases the same to 1e-9 (one tie in the default radius). **D9 (open):** near the
floor the acceptance compares the frame-route `f` of the current point with the raw-cores `f` of a trial, rounded
differently (line-search exhaustion never fires; float32 exact Newton accepts a spurious step). The implementer
recommends (a) `point_objective(x)` per accepted point; Claude proposes (d) carrying the accepted trial's
`point_objective` value forward as the next reference (no extra cost); the reviewer assesses both with part 2.
`P16_part2_REVIEW_r1.md`: NOT YET on D9 only (the red list exact; the reviewer's 18 side-by-side cases and the
implementer's 36 match; routing, `held`, the regularizer and compile keys right). **D9 ruled by Claude: (d)**, the
accepted trial's `point_objective` value carried forward as the next reference (one evaluation per run start), sound
per the reviewer's prototype and the same comparison 2026.5.0 made, without its extra forward per accepted point; the
history reports the carried value; tests that relied on the mixed routes get a route-independent mechanism. Backlog:
the pre-existing default-radius tie. In round 1.
Round 1 (`P16_part2_RESPONSE_r1.md`): D9 built as (d) (`_point_objective_parts`, `_scored`; one scoring per start;
the carried value through every path; the history reports it; route-independent mechanisms in the tests that relied on
mixed routes; a like-with-like test). `P16_part2_REVIEW_r2.md`: SIGN OFF (the build bitwise equal to the reviewer's
prototype on 36 cases; no test weakened, four reviewer mutants killed; the frame-route `f_resolution` is the right
scale for point-route values, by the rounding analysis). Expected red: 103 ids (display 13, frontend 90). **P16 part 2
committed to `p16-wip`.** Next: part 3 (the display and record), the same implementer.

### P16 part 3 (the display and record) built (2026-11-21)

`P16_part3_HANDOVER.md`: `make_newton_display(oracle, batch, val_batch, ...)` (denominators once per batch; the
training numerator through `info.lm`'s held `r`, no sweep; validation without `held`; unweighted, unscaled block sums;
weight-shaped records; header-only for an oracle without block sums); `format_newton_iter(..., order, error_table)`;
the multi-index table for `m >= 2`. Full suite: 90 red, all frontend (`P16_part3_EXPECTED_RED.txt`). Claude confirmed
the implementer's reading that the plan's "the display reads `f_resolution`" is the floor (part 2), no printed cell.
In review.
`P16_part3_REVIEW_r1.md` NOT YET on one finding: the training table's numerator (`info.lm.batch`) and denominator (the
display's `batch`) could come from different batches (0.550 recorded against a true 2.469). Ruled by Claude: the
display reports its errors on the batch it is given, numerator and denominator both; when that batch is the run's
(`is`), the numerator reuses the held `r` (no sweep), otherwise it is queried at the model's frame without `held`, as
validation is. Keeps the planned signature and gives `batch` its meaning. For part 4: the frontend hands the display
the object the optimizer runs on; the translation keeps the both-or-neither check on `val_sample`/`val_data`.
Round 1 (`P16_part3_RESPONSE_r1.md`): the display's errors on its own `batch` (fast path on the run's batch, a
held-free forward otherwise; no unwt on another batch); `P16_part3_REVIEW_r2.md`: SIGN OFF (the reviewer's case now
equals the dense errors on the display's batch). Expected red: 90, all frontend. **P16 part 3 committed to
`p16-wip`.** Next: part 4 (the frontend and the translation), the same implementer; a fresh reviewer for part 4 (the
reviewer's context is near its limit), briefed by the reviewer's own handoff note.

### P16 part 4 (the frontend), first half built; the implementer's context full (2026-11-21)

`P16_part4_HANDOVER.md` (in progress). Built (4a): the frontend `LocalModel(geometry, backend)` and `local_model`; the
optimizers' heads `(geometry, oracle, batch, x0, ...)`, `_setup`, `start_frame`'s consistency check, the display handed
the run's batch (also under `use_jit`, via `_prepare_jit_inputs` returning a jax tree as is); the translation as a
decorator; the removals; D7; `test_optimizers_frontend` and `test_aligned_padded_frame_frontend` migrated. Not done
(4b): `test_fitting` and the other frontend test files, the translation and frontend-only tests, the pages and
examples that call a factory, the side-by-side translation cases, the full gate and the slow tier. **Ruled by Claude:**
tests that call the optimizers in the old form, which the translation keeps working, are converted in P17, as the plan
already says ("What the translation keeps working moves in P17"). Committed as an UNREVIEWED checkpoint on `p16-wip`;
a fresh reviewer reviews 4a (briefed by `P16_REVIEWER_HANDOFF.md`) while a fresh implementer orients for 4b.

### A planned pause after P16 (the maintainer, 2026-11-21)

The maintainer asked why `t3_probe_corewise_hessian` has no weights. Claude's answer: by O-22 the transposes and
Hessians take a data-space covector under the plain pairing (for a weighted fit, `c = omega^2 r`; the engine forms it
as `w_tilde` from `r` and `weight` inside the fitting layer), and the parameter-space Riesz map is the geometry's.
**The maintainer: "almost all fits will involve data-space weights of some kind, so weighting should be first-class in
all these functions. Let's work in a pause after P16 to discuss this more."** So: a PAUSE after P16 (before P17), on
first-class data-space weights in the sampling functions. Also for that pause (from the 4b implementer's orientation):
the oracle's `y` cannot carry the frame stack `C`, so a stacked local model now shares one `y` across its elements,
where the old factories took a residual per element (a capability narrowing, accepted for P16, noted for the
CHANGELOG, to discuss with the data-space design).
`P16_part4a_REVIEW_r1.md` (the fresh reviewer): NOT YET, nothing blocking; the translation keeps the old meaning on 25
old-form calls against the pre-P16 library (dx at most 1.4e-14; Adam 9e-8 by its eps). Ruled by Claude: fix the
`(1,)` probe weight broadcast, the frame orthogonality tolerance (effective_rtol), the corewise `frame=` (structural
raise), an untied given frame on a shared geometry (safe-mode check, for `local_model(frame=)` and
`newton_cg(start_frame=)`; the same gap existed in 2026.5.0), the stacked-`y` shim, `_is_packed`, the duplicated batch
logic, test form; the translation tests in 4b to cover a regularized custom draw and order >= 2. The 4b implementer
fixes 4a first; the reviewer checks; then 4b.
For the pause after P16, the maintainer's principle (2026-11-21): "any place where there is a silent conversion of a
data space co-vector to vector without the option to use weights should be viewed as a kind of type error."
And (the maintainer): the jet reversal and the weighting are two parts of one Riesz map. Claude confirmed in the code:
the engine's `w_tilde_step` calls `jets.jet_to_dual_jet(kind, order, r, weight)`, weight squared then every jet axis
flipped, one function; the standalone transposes and Hessians reach it with no weight, so they apply the flip and fix
the weight to the identity, with no option. The pause's audit covers every `jet_to_dual_jet` / `dual_jet_to_jet` call
site and whether a weight reaches it.
Round 1 (`P16_part4a_RESPONSE_r1.md`): all ten 4a findings dealt with (`prepare_batch` the one batch helper; the tie
check on given frames; `effective_rtol`; the corewise `frame=` raises). `P16_part4a_REVIEW_r2.md`: SIGN OFF (25
old-form cases and 4 new-form cases still match the pre-P16 library). **P16 part 4a committed to `p16-wip`.** For 4b
from the review: a per-mode probe weight on noisy data in the translation test; a bad `val_batch` in the bad-batch
test. Next: 4b, the same implementer.
Also for the pause after P16 (the maintainer, thinking aloud, not final): which corewise operations are methods of
`TuckerTensorTrain` and which standalone functions. His lean: the raw sampling operations, their derivatives and
transposes as methods (independent of an objective); the Hessian-related ones as functions (they combine sampling with
an objective). Claude's view: the corewise "Hessian" is the second derivative of phi_c(x) = <c, z(x)> (O-22), the
derivative of the transpose along a direction, needing only a data covector, like the transpose; so it belongs with
the methods, and the objective-dependent Hessian of a fit (Gauss-Newton plus the curvature term, c = omega^2 r) is
already in the fitting layer. The real issue may be the name "Hessian" (candidate: `*_corewise_transpose_derivative`).
To decide together with first-class weights.
And (the maintainer): "the transpose should take a vector. J: X -> Y', J^T: Y'' -> X' or Y -> X'. The (Gauss-Newton)
Hessian: Hgn = J^T W J, where W: Y' -> Y is the Riesz map. We need to get this all settled and typed consistently. I
think we may be mixing and matching Banach space transposes with Hilbert space transposes." The audit for the pause
types every operation this way (data side and parameter side), and marks each place where a Riesz map (data: the
weight and the jet flip; parameters: the geometry's metric and projection) is applied implicitly or partly.
The maintainer: the audit for the pause is done by an agent (to preserve Claude's context), when the pause arrives.
P16 part 4b built (`P16_part4b_HANDOVER.md`): `test_fitting` rewritten (one `y` per stacked model; derivative fits
against the dense `oracle.py`); the other frontend files converted; `TestTheTranslation` against pinned pre-P16 numbers;
the frontend-only fit; user oracles; the pages' and examples' factory calls moved to `local_model`. Gate at `-n 2`: the
suite 1,896 passed with one order-dependent 4a test failing (fixed, passes alone; the suite not re-run); the other five
steps pass; the slow tier passes; lowest MemAvailable 4.0 GB. Ruled by Claude: a new weight does not recompile (the
weight is a batch leaf), each jitted result checked against its eager model. In review; the reviewer re-runs the suite.

### For the pause after P16: where the sampling functions live (the maintainer, 2026-11-21; a proposal, not a decision)

The maintainer's proposal, to be assessed: separate the sampling functions and methods (1) by the object they act on,
a manifold point (with the base modules) or a tangent (in the tangent layer); corewise is a flat manifold with another
geometry, as the MANIFOLD and COREWISE singletons already treat it; and (2) by whether they need a metric or weighting
(Riesz maps). Data-space metrics and weights are used only for fitting and optimization (e.g. the Hessian operations),
so the functions that need them would be confined to the fitting and optimization layers (and the engine), and the rest
of the library stays valid in the Banach setting. The input space has no weights today (to be checked); input weighting
can be applied by preparing the data. Data weights could be handled the same way for least squares, but not for
future nonlinear losses, whose Hessian acts as a data-space metric. "This refactor is the easiest time to reorganize
sampling functions. Let's get it right now instead of having to break things again if we do it later."

**The audit at the pause (one agent, Claude's judgement: combined with the typing audit, one inventory):** every
sampling-related function and method, backend and frontend: where it lives; what it acts on (point or tangent;
corewise as a flat manifold); its typing under the maintainer's scheme (J: X -> Y', W: Y' -> Y, J^T: Y -> X', R_X:
X -> X'; input space); which metric or Riesz map it applies (data weight, jet flip, parameter metric, input weights),
fully, partly or silently as the identity; whether weights can be passed; and where it would live under the proposal.
Then the findings and the options, not decisions.
The maintainer refines (2026-11-21): the ambient tensor space is implicitly Hilbert-Schmidt almost everywhere, so the
dividing line is not whether a function needs a metric but **whether it depends on the objective function**.
**Claude's correction of its own earlier statement** (that the standalone transposes apply "half of the Riesz map",
the jet flip without the weight): checked in the code, the transposes' contract is the plain slot-wise pairing
`phi(x) = <residual, z(x)>` (their adjoint doctests pair `sum(r * J x_dot)` slot by slot), and `jet_to_dual_jet`
stores the functional `a -> sum_k weight[k]^2 r[k] a[k]` turned so that the engine can read it as the corner
coefficient of a jet product. So the flip is a change of representation of that pairing inside the engine, needed for
any weight, not a metric; the only metric choice is the weight, which is the objective's. With no weight, the
standalone transposes are exact metric-free transposes for the slot-wise pairing on Taylor coefficients. The remaining
type issue is the name: the argument is a data-space element paired slot by slot, not a residual.
`P16_part4b_REVIEW_r1.md`: SIGN OFF. **P16's gate:** the reviewer's full suite on the final tree at `-n 2`: 1,897
passed, 5 skipped, 1 xfailed, 90,942 subtests, exit 0; the implementer's module doctests (321), quickstart, 47 pages,
14 examples, Sphinx with 0 warnings; the slow tier serially (66 passed). The translation's 19 pins equal the pre-P16
library's own numbers. Notes: the weight compile test may also assert the eager products differ (optional); stale
`uniform_least_squares_problem` docstrings in two test files (P19); the test-vocabulary helpers and the stacked `y`
for P17 and the CHANGELOG. **P16 complete; squashed into one commit on `graded-messages`.**

### The pause after P16: the sampling audit (2026-11-21)

`dev/graded_messages/SAMPLING_AUDIT.md` (an agent). Main findings: the jet flip is representation, not a metric
(verified numerically); the transposes and Hessians are correct as maps from `Y` but are named and documented as
taking a residual (`J^T r`), which reads as `J^T W r` with `W = I` silently; derivative coordinates convert the opposite
way for `Y` and `Y'` (a trap); `jet_to_dual_jet` is public but no frontend function takes its turned form; the weight
is block-scalar only (no per-sample weight; entries one global scalar); no input-space weighting; the oracle applies
the scheme in full; on the parameter side `T3Tangent` holds both `X` and `X'`; corewise sampling exists twice (raw
core pairs on the TT methods; `T3Tangent` at `(B,G,G,G)` in the frontend local model). Four decisions for the
maintainer: the type of the transposes' data argument; how much `W` expresses; corewise as point methods or as
tangent operations of `COREWISE`; then the names.
Decisions 1 and 2 of the audit taken by the maintainer (TARGET_DECISIONS, "The data metric"); 3 and 4 still with him.
A sub-arc between P16 and P17 is planned for 1 and 2: Claude's first draft `dev/graded_messages/DATA_METRIC_PLAN.md`,
refined by a writer/reviewer pair; the maintainer reads it before any build.
P16's CI failed on `test_dispatch.py::TestSwitchNewton::test_rounding_gate_at_float32` (both legs; passed on the
laptop). Cause (an agent, `P16_CI_FIX.md`): a route mix in the test after D9, the hand floor read from the reported
point-route misfit while the gate's floor reads the model's frame route; normal float32 variation between routes
(reproduced under 1e-7 perturbations of the start). Fixed in the test only: the model's misfit through a callback, a
tolerance derived from float32 rounding (tighter than before); six gate mutants killed.
The data-metric plan refined: `DATA_METRIC_PLAN_REVIEW_r1.md` (15 findings, 3 blocking), the writer's revision and
`DATA_METRIC_PLAN_RESPONSE_r1.md`, `DATA_METRIC_PLAN_REVIEW_r2.md`: SIGN OFF (the reading rule checked by script, all
exact; four minor points for the slice designs). **With the maintainer to read before any build** (eight questions).
Lightweight typing (2026-11-22): the maintainer folds Claude's ideas 1 (the algebraic type in signature comments), 2
(one public representation of P*, the flip engine-private) and 6 (transformation-law tests) into the data-metric plan
(its new section 9, Claude's draft); 3 and 4 (types in `jet_contract`'s strings; types on the declared graph) carried
to Part 2 (TARGET_DECISIONS). Claude corrected one claim in the draft: the jet-rescaling law does not catch a
Taylor/derivative mix-up. Another writer/reviewer cycle on the plan. **The maintainer decides tomorrow:** the audit's 3
(corewise representation) and 4 (names), and the plan's questions.
`DATA_METRIC_PLAN_REVIEW_r3.md` on section 9: NOT YET, 7 findings, two of them false claims in Claude's draft: (1) "one
public representation of P*" is false in the backend (`w_tilde` and the tilde variables are public engine variables
in the turned form; the converters stay backend-public, frontend-private); (2) no scaling law catches the probe
residual/sample swap (both arguments carry the input type; the laws are homogeneity tests, necessary not
sufficient). The reviewer derived and verified the parameter-side law. The writer revises.
`DATA_METRIC_PLAN_REVIEW_r4.md`: SIGN OFF on section 9 (four minor points for M1's design note). Claude added a short
"In brief" at the plan's head (the reviewer's readability suggestion). **The plan is final for the maintainer's
reading:** eleven questions (section 8), plus the audit's decisions 3 and 4.
Notation decided by the maintainer (2026-11-22; TARGET_DECISIONS, "Notation for the jets and the data metric"): `P`/`Q`
for the jet/dual-jet groups, `F` the box sizes, `M` the data metric (accepting the collision with the manifold), `[G]`
structural presence (every group may be empty). The writer applies it across the plan; the reviewer checks.
The plan's notation pass: `DATA_METRIC_PLAN_REVIEW_r5.md` (NOT YET: the rename's scope, M0, letter collisions, Q1),
the writer's round 6, `DATA_METRIC_PLAN_REVIEW_r6.md` (SIGN OFF), then three small edits (`ℓ_j` for the jet rescaling;
Q1's cost of `data_metric`; Q12 on `J_in_i`). Six slices M0 (the `J -> P` rename) to M5. **Final for the maintainer:
twelve questions** (section 8), plus the audit's decisions 3 and 4.
The maintainer answered the plan's questions 2, 4, 6, 9, 10 (`*` for the algebraic dual), 11; Q1 (`data_riesz_map`)
to confirm; Q3, Q5, Q8 (weights) and Q7 (derivative coordinates, guided by T3Polynomial's use) to discuss; Q12 open.
An agent studies T3Polynomial's use of jetted probe fitting (read only).
Q7 decided by the maintainer: option (i), Taylor coordinates everywhere, one conversion at the boundary (data and
weight together; a data-relative weight computed after converting the data), documented in `prepare_derivative_data`,
`data_metric` and the pages. Claude recommended for Q5 and Q8 together: data-space objects (`y`, `omega`, residuals)
have the type of `z`, each axis 1 or `z`'s size (the full form only, with a constructor helper for block weights;
per-element data on a stack then removes a special case). The maintainer asked for a T3Polynomial-style example.
All the maintainer's decisions folded into the plan (the writer's round 7); `DATA_METRIC_PLAN_REVIEW_r7.md`: SIGN OFF
(the rule checked by script for every kind, layout and stack; M2 a single switch, justified); the writer's last edits
(`blocks_to_weight` as the one open question in section 8; the converters' `K` behaviour stated exactly; writer
proposals listed for their design notes; two strengthenings of M2's gate). **Final for the maintainer's read.**
The plan is COMPLETE (2026-11-22): the last decisions folded in (`blocks_to_weight`; `stack_shape=()`; defaults are
ones in the natural coordinates, so `prepare_derivative_data(weight=None)` returns the factorials; the weighted block
sums supersede the plain ones under the same names; the `unwt` figure dropped). Section 8 has no open question. Next,
on the maintainer's go: M0 (the `J -> P` and `J_in_i -> F_in,i` rename), with fresh agents.

### M0 started (2026-11-22): the `J -> P` and `J_in_i -> F_in,i` rename

The maintainer's go. A fresh implementer and reviewer (Opus 5.5, high), briefed on `BRIEF.md` and the plan; the
reviewer is kept for the sub-arc's slices. Design-first: the implementer's `M0_DESIGN_NOTE.md` (the rename map, the
script, the bit-for-bit evidence), the reviewer's critique, Claude's rulings, then the build.
M0 designed (`M0_DESIGN_NOTE.md`, `M0_DESIGN_REVIEW.md`: sound). Ruled by Claude: the test-local ring `J` becomes `ring`;
a guard against `J` in library subscripts literals; two small doc additions; the script and edit log kept under
`implementation/m0/`; HEAD's per-test counts from one local `-v` run of a `git archive` copy before the build (the
reviewer's reference too); the shape rule (a concrete shape spells the sizes `F`, `min(F_in_i, F)`; a family spells
the group `P`); `F_in,i` spelled `F_in_i` in code. Building.

### M0 committed (2026-11-22)

`M0_HANDOVER.md`, `M0_REVIEW_r1.md` (SIGN OFF, one round). The rename `J -> P` (the jet group, `GroupLengths.P`,
`len_P`, the suffixed locals), `J_in_i -> F_in_i` (`F_in,i` in prose), the test-local ring `J -> ring`, by one logged
script (`m0/rename.py`, 1,579 edits); the guard against `J` in library literals, a floor in the literal scan, the
retired `J` raising; the glossary and naming-conventions entries. Evidence: the round trip empty; the implementer's
battery (23,577 outputs bitwise equal, 154,550 einsum calls identical) and the reviewer's own (24,038 calls
byte-identical incl. jitted float32 jax, 743 outputs bitwise equal); per-test counts against a local baseline, exactly
the two expected nodes differ; 8 reviewer mutants killed. Gate at `-n 2`: suite 1,898 passed (90,945 subtests), 321
module doctests, 47 pages, 14 examples, Sphinx 0 warnings. The baseline log is kept gzipped. Next: M1.

### M1 designed (2026-11-22), in critique

`M1_DESIGN_NOTE.md`: `data_metric` and `blocks_to_weight` in `jets`, the batch method, the frontend converters made
private (O-20's frontend check deleted), the law helper `tests/_typing.py` with negative controls, the type-first
comment grammar. Five decisions to rule after the reviewer's critique.
`M1_DESIGN_REVIEW.md`: sound. Ruled by Claude: `_with_K` moves to M2 (nothing calls it in M1); a packed `r` with a
ragged weight raises; the batch method raises by rank on block weights until M2 (pinned by a test); the frontend
converters leave `fitting` entirely (not in 2026.5.0); M0's log is M1's reference (verified). Folded in: exact mode
count for a ragged `r`; the static axis check (`data_metric` never grows `r`); the docstrings (Q7's recipe; the
converters retyped); an input-law negative control. Building.
M1 built (`M1_HANDOVER.md`; gate at `-n 2`: suite 1,918 passed, 91,249 subtests; doc steps green; the tally against
M0's log: 37 nodes, all expected). `M1_REVIEW_r1.md`: NOT YET on two small points (a mixed-precision case missing from
the bitwise bridge test; a comment block deleted by accident). Ruled by Claude: type comments use Unicode for the
algebraic types, identifiers stay ASCII. In round 1.
Round 1 (`M1_RESPONSE_r1.md`): the bridge test over the four float32/float64 pairings (the reviewer's mutant now
fails 8); the comment block restored verbatim; the Unicode/ASCII ruling in `signature_style.md`. `M1_REVIEW_r2.md`:
SIGN OFF. **M1 committed.** Next: M2, the switch to the rule (one change, built in parts on a side branch as P16 was).

### The M slices continue without stopping (the maintainer, 2026-11-22)

"Go ahead with M2 and continue through the rest of the M slices, unless a major issue arises. Please track decisions
you make and we will discuss them after the M slice arc is complete." Claude's decisions are tracked in
`implementation/M_ARC_DECISIONS.md` (M0 and M1 entered). M2 starts on a local side branch `m2-wip` with a fresh
implementer; the sub-arc's reviewer is kept.

### M2 designed (2026-11-22, on `m2-wip`), in critique

`M2_DESIGN_NOTE.md` (a fresh implementer): five parts (independent oracles and the pre-M2 battery first; the engine;
`jets`; the backend fitting layer; the frontend and old forms); the gate with the reviewer's two strengthenings; six
corrections to the plan found in design (three of them silent wrong answers the plan would have allowed: a `y` without
`C` right-aligned on a stacked model; an edge-1 `y` unmasked on the uniform layout; `f_resolution` and `y_block_sumsq`
not over `z`'s shape).
M2 part 1 (the independent oracles and the pre-M2 battery): `M2_part1_HANDOVER.md`; review r1 NOT YET (a size-1 axis
of `z` untested: a larger weight was accepted on a one-sample batch, `f` five times too large); round 1 added the
refusals; `M2_part1_REVIEW_r2.md` SIGN OFF (the battery re-recorded independently, bytewise equal). **Committed to
`m2-wip`.** Next: part 2, the engine.
M2 part 2 (the engine) in progress: the library changes are in (`r_step` by position, `_weighted_sum`, the seed as
`flip(data_metric(...))`, `_with_K`, `_check_data_space`, the uniform packing and masking); the implementer's context
filled, so an UNREVIEWED checkpoint is committed on `m2-wip` and a fresh implementer finishes part 2 from
`M2_part2_HANDOVER.md`.
M2 part 2 finished by the second implementer (`M2_part2_HANDOVER.md`): a `None`-guard bug in the weight mask fixed; a
gauge-blind test corrected; `TestAgainstTheReference` (the engine against the part-1 reference over the jet matrix,
135 subtests); 14 mutants killed or justified; the full suite's 194 red ids recorded, all one cause (old-form objects
reaching the new check through the oracle), for parts 4 and 5. In review.
M2 part 2 (the engine): `M2_part2_REVIEW_r1.md` NOT YET (the join of the modes' shapes in packing untested); round 1
added `test_modes_with_different_size_1_axes`; `M2_part2_REVIEW_r2.md` SIGN OFF. **Committed to `m2-wip`** (with the
earlier unreviewed checkpoint `fe7c7250`, one review unit). 194 ids red, one cause, for parts 4 and 5. The reviewer's
handoff note for a successor: `M_REVIEWER_HANDOFF.md`. Next: part 3 (`jets`).
M2 part 3 (`jets`): `M2_part3_HANDOVER.md`, `M2_part3_REVIEW_r1.md` SIGN OFF (the checkers against 138 objects of the
reviewer's own construction; the converters bitwise in four precision pairings; the factorials by hand; the
re-pinned bridge independent). **Committed to `m2-wip`.** 123 red, for parts 4 and 5. Parts 4 and 5 go to a fresh
implementer and a fresh reviewer (both previous contexts long), from the handovers and `M_REVIEWER_HANDOFF.md`.
M2 part 4 (the backend fitting layer): `M2_part4_HANDOVER.md`, `M2_part4_REVIEW_r1.md` SIGN OFF (the fresh reviewer's
2,135 independent cases, 0 failures; 7 mutants killed; the diff audit clean). **Committed to `m2-wip`.** 23 red, all
part 5's frontend routes. Two minor points carried to part 5 (a doctest flag; a clear error for `f_resolution` with a
held `r` and no `y`). Next: part 5, the frontend, the old forms, the pages, the battery comparison, the full gate.
M2 part 5 (the frontend, the old forms, the pages, the gate): `M2_part5_HANDOVER.md`, `M2_part5_REVIEW_r1.md` SIGN OFF
(the translation against the pre-M2 library over every old weight form: 163 of 196 arrays bitwise, losses within
1.4e-15; stacked `y` against dense misfits; the battery re-recorded independently, bytewise equal). The battery's
gauge explanation corrected per the review (a sign-gauge flip of the SVD retraction, possible from any start).
**M2 gate:** suite 1,954 passed, 92,096 subtests, 0 failed; 325 module doctests; quickstart; 47 pages; 14 examples;
Sphinx 0 warnings; slow tier 66 passed; the battery AS EXPECTED (24,427 of 25,476 arrays bitwise, the rest within
3.6e-16); the node tally against part 1's log only the expected differences; the diff audit clean. **M2 squashed onto
`graded-messages`.** Next: M3 (the display and the weighted block sums).
M3 designed (`M3_DESIGN_NOTE.md`, `M3_DESIGN_REVIEW.md`: sound); rulings in `M_ARC_DECISIONS.md` 20 to 23. Building.
M2's CI: green on the current leg, the examples and the docs; the floor leg fails one test,
`test_uniform_curvature.py::TestUniformSharedExactNewton::test_zero_padded_shared_restart` (17 Newton steps against a
bound of 16; passed before M2). Likely a knife-edge iteration count moved by M2's rounding-level regrouping; an agent
investigates (`M2_CI_FIX.md`), under the rule "change the run, not the tolerance".
M3 built (`M3_HANDOVER.md`): suite green (1,960 passed); `f_resolution` and every non-block-sum battery output bitwise
equal to `68c10621`; display records against `68c10621` within 2.3e-16; the tally only the expected 10 nodes; 5
mutants killed. The remaining gate steps (doctests, pages, examples, Sphinx) are being run, then the review.
M3: gate green (suite 1,960 passed; module doctests 325; quickstart; 47 pages; 14 examples; Sphinx 0 warnings);
`M3_REVIEW_r1.md` SIGN OFF (888 independent block-sum checks; `f_resolution` and 24,871 other battery arrays bitwise
equal to `68c10621`; the validation rule checked; 5 mutants killed). **M3 committed.** Next: M4 (the transposes typed,
`residual` -> `c`).
M4 (the transposes typed) started: the M3 implementer writes `M4_DESIGN_NOTE.md`.
