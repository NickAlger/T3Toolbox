# Decisions Claude made during the K slices, for discussion with the maintainer after the arc

The maintainer (2026-11-25): "The orchestrator is advised to use their judgement for decisions along the way, except
for big decisions or major unforseen problems which they can pause and talk to me about. The orchestrator logs all
decisions, and then after the entire arc is complete, discusses it with me." Each entry: the decision, the reason,
where it is recorded. Larger rulings are also in `LOG.md`. The design's decisions are not here: they are in
`PACKING_PLAN.md` §11 to §14, already ruled.

## The arc's start (2026-11-25)

1. **The agents' brief is a new page, `K_BRIEF.md`, read before `BRIEF.md`.** `BRIEF.md` is written for
   `PART1_PLAN.md`, names an older scratch folder, three planned pauses and 19 GB of disk, and tells the agents to
   decide nothing ("Issues: stop and report it. Do not decide it"). The maintainer's process asks for judgement, so
   `K_BRIEF.md` states the process in his words, gives the precedence rule, and replaces that paragraph: small choices
   are the agents', recorded; the cases that reach the orchestrator are listed.
2. **Every slice has a reviewed plan phase,** the small ones too (the M arc built M4 and M5 without a design review).
   His process says "each slice includes a plan phase".
3. **The orchestrator relays and rules; it does not edit the library or the tests.** `ORCHESTRATION.md` §2 step 4 had
   the orchestrator add a missing test itself. Here a gap goes back to the implementer ("approves … or sends it back
   for more work"; and the standing rule that the library changes only through the process).
4. **The plan phase's records:** `K<n>_DESIGN_NOTE.md`, `K<n>_DESIGN_REVIEW_r<k>.md`, `K<n>_DESIGN_RESPONSE_r<k>.md`,
   with the reviewer's verdict on the first line, as the build's reviews have it. The M arc had one design review and
   Claude's rulings in the log; with rounds "until the reviewer signs off", the rounds need numbers.
5. **The two agents start together.** The implementer orients and writes K0's design note; the reviewer orients at
   the same time and writes down its own view of K0's risks before it sees the note (a review not anchored on the
   note). Only the implementer may run the suite in that time (one `-v` run of the starting tree, the baseline of
   K0's per-test tally).
6. **The two planning commits (`1eda7fc8`, `b5125f76`) are pushed with K0's commit,** not before: they change `dev/`
   only, and a push starts both CI legs. *Overtaken the same day:* the maintainer asked for the push at once, so that
   the plan is safe on the remote; pushed as `5848e687`, with a third commit holding the arc's start records
   (`K_BRIEF.md`, this file, the log entry).
7. **Disk.** 8 GB are free (19 GB when `BRIEF.md` was written). The agents delete a gate's Sphinx output once it is
   read and keep kept logs gzipped. To raise with the maintainer if it falls further.

## K0, the plan phase (2026-11-25)

8. **The floor of two scan tests is lowered from 100 to 60** (`tests/test_contractions_interpreter.py:551`,
   `tests/test_contractions_sharding.py:102`: "the library holds more than 100 grouped subscripts"). After the
   deletion 86 remain (209 today; measured by both agents on their own simulated trees). The floor guards against a
   scan that silently finds nothing; every scanned string is still checked. A lowered assertion, so logged.
9. **The non-vacuity clause of `TestTransposeOutputHygiene` is replaced, not kept.** It asserted that the old seam
   left garbage in the padding; the engine returns zero there even with its exit mask removed (measured by both), so
   the clause cannot pass. In its place a negative control: with every engine mask removed, the zero-padding check
   fails on garbage-padded input.
10. **`sum_over_probes` stays in the tests after K0** (the plan's K0 says it "goes here", having measured it in
    deleted code only): it is a keyword of the test helper `tests/_derivative_data.py`, passed in about 60 test
    lines, and a label in a few others. Not renamed in a deletion slice. **For P17's list, by name.**
11. **The per-kind negative control for the uniform direction mask is deleted with the old kernel it tested.** In
    general position the engine has no such property (removing the direction masks leaves the real blocks unchanged,
    since every other operand is masked on entry); at TT slack the probe kind is pinned by `TestKpAtTTSlack`.
    **For K5: its design note adds the apply and entries cases of `TestKpAtTTSlack` first, against the engine as it
    then is, before it changes the masking** (the reviewer tried the parametrization: the harness is probe-shaped, so
    it is real work).
12. **Every docstring or comment that K0 rewrites has its new text in the design phase** (`k0/prose_edits.md`, before
    and after), reviewed with the note; a deleted comment needs only its lines. The maintainer: "comments are
    important … when in doubt, do the review."
13. **No permanent test with frozen export lists or name patterns for the deletion** (the implementer's proposal):
    K1 and K5 change those lists. At most one assertion that the deleted module cannot be found.
14. **K0 removes from `docs/api_reference.rst` the names of things that no longer exist** (the deleted module at
    `:130`, and six model factories at `:75-80` that P16 deleted and nobody removed: Sphinx is silent because
    `nitpicky` is off), where the removal leaves the text correct. Adding the new names is P19's. Strictly the six
    are outside K0; the file is edited anyway and the fix is of the same kind.
15. **K0's design note approved** after three rounds (`K0_DESIGN_REVIEW_r1.md` to `_r3.md`; NOT YET, NOT YET, SIGN
    OFF). With it: the rewritten `backend/fitting.py` module docstring keeps the factoring `J = 𝒥∘Π`,
    `Jᵀr = Π∘𝒥ᵀr` that the old one stated (my addition: it is what says why the oracle and the geometry are two
    halves), with `Π` named as `geometry.project` (the gauge projection on the manifold; on the corewise geometry the
    identity, or the per-group mean of tied factors: the reviewer's correction).
16. **One `-v` run of the suite may serve as both the gate's suite step and the tally's input** (the gate's own suite
    step is the same command with `-q`), to spare a second 40-minute run on a machine that holds one at a time.
17. **A second reviewer takes K0's build review** (and stays for K1). The first reviewer's context filled in the plan
    phase and it expected to hand over partway through the build review; a fresh one, oriented while the build runs
    and started from `K_REVIEWER_CARRYOVER.md`, avoids a handover in the middle of a review and has not watched the
    design being made. It writes its own tools and its own list of what the build review must establish
    (`k0/REVIEWER2_PLAN.md`). The maintainer (2026-11-25): "Don't worry about the token cost."

## K0, the build (2026-11-25)

18. **A third scan floor is lowered** (`tests/test_contractions_sharding.py:136`, `checked > 400`, to 150): the
    sweep checks 228 cases after the deletion, 930 before. The design missed it because the vocabulary floor fails
    first in the same test. Same kind as decision 8 (`checked` is a fixed function of the scanned vocabulary; the
    reviewer verified that the vocabulary floor already implies at least 122). The reviewer searched every count
    asserted against a literal in `tests/`: no fourth floor trips at K0. **For K5:** the call-site floor
    (`> 90`, 113 at K0) is the one the removal of the per-mode loops is most likely to reach.
    *Correction to decision 8:* the tests' own scan on the built tree gives 84 subscripts, not 86 (the 86 was a
    simulated tree that still held two dead scan bodies).
19. **K0's suite count rests on one `-v` run taken before the third floor's fix, plus that one file rerun** (1,881
    passed and 1 failed, then 4 passed in the fixed file: 1,882). Not rerun in full: the reviewer checked that the
    library was last written before the run started and that the only file written after it is the fixed test file,
    and ran that file on the final tree. Both CI legs run the final tree after the push.
20. **A second implementer takes K1** (the first one's context filled building K0; it finishes K0's three minor
    fixes). Started on its orientation and K1's design note while K0's fixes are made, with no test runs until K0 is
    committed.

## K1, the plan phase (2026-11-25)

21. **A third reviewer takes K1's design review.** The second reviewer, after K0's two build-review rounds, judged
    that K1's note, its rounds and its six part reviews would not fit its context and stopped before reading the note
    (no opinion given), at a clean point, as asked. Its tools are kept in `k0/reviewer2_tools/`, its view of K1's
    risks in `k1/REVIEWER_PRENOTES.md`, its handover in `K_REVIEWER_CARRYOVER.md`. Observed so far: a reviewer or an
    implementer lasts about one phase of a slice after its orientation (the orientation is the maintainer's
    instruction and is kept); rotations are planned at clean points, and each agent is asked after every turn how much
    room it has.
22. **`flat_draw(batch, batch_size)`**: the default draw loses its `oracle` argument, which nothing reads once the draw
    calls the batch (the plan's §4.3 moves `take` and `n_measurements` to the batch). A public backend signature, on
    this branch only (P16 introduced the oracle form; no release has it).
23. **The guards on a second batch arrive with the move of the kind, in K1's part 0, not in K4:** the frontend entries
    refuse a validation batch whose kind or order differs from the batch's, and `_frontend_draw` compares each drawn
    batch's kind and order with the full batch's. The plan's §4.3 lists them without a slice. Reason: before part 0
    a validation batch was read under the oracle's kind; after it, under its own, so without the guards the "pure
    refactor" would change a reading silently. (The reviewer found that a user's `draw=` receives only `rng`; what
    must agree is what it returns.)
24. **No sentinel default to make the old two-argument `LeastSquaresBatch(ww, y)` "fail on its kind"** (the plan's
    §2.4 says an old positional call fails on its kind; with two arguments Python refuses first, "missing
    'sample'"). The form exists only on this branch, it fails at once and loudly, and a default on `sample` would
    make the signature lie.
25. **The frontend re-exports the renamed helpers** (`t3toolbox.fitting`: the three metrics,
    `derivative_block_weight_to_taylor`, the two `prepare_*_derivative_data`, in place of `data_metric`,
    `blocks_to_weight`, `derivative_weight_to_taylor_weight`, `prepare_derivative_data`). The names are the plan's
    (§5); their place in the frontend follows that module's rule that a fit needs no import from the backend. It
    changes the frontend's public surface, so logged.
26. **K1's parts are amended** (the implementer's proposal, checked by the reviewer by experiment): each old name is
    deleted with its last library caller (the frontend re-exports four of `jets`' names, so deleting them in the
    `jets` part would stop `import t3toolbox`); one temporary line in the oracle, `full_weight=batch.weight`,
    between parts 2 and 4 (with it every weight is either refused by the new rule or read as before: the new rules are
    subsets of the old, with the same meaning, so no intermediate tree runs silently wrong); the translation of the
    old call form moves to part 4 with the batch's fields.
27. **Two audits by construction are part of K1's evidence** (the reviewer's findings): part 0 pairs every removed
    `LeastSquaresOracle(kind, order)` with the batch constructions it served (a wrong order passes every shape
    check); part 4 maps every removed `blocks_to_weight(...)` to a `block_weight=` and lists every `weight=` that
    became `full_weight=` (for apply and entries the wrong one of the two is legal and bitwise equal).
28. **A part of K1 is built only when its public docstrings are written in full and reviewed** (parts 0 and 1 for
    the design's sign-off; parts 2 to 5 before each is built). K0 had every rewritten text in the design phase; K1
    is too large to write every docstring twice, so the public contract is written first and the rest is reviewed
    with each part's build.
29. **For the maintainer at K2:** the plan's `check_probe_data` is documented for "y, a full weight, a residual", but
    `y` and a residual are exact on `W` and a full weight may be 1 there; a public full-weight checker (three, one
    per kind) is proposed. Public names the plan does not have. In K1 the batch's `check` uses a private helper.
30. **K1's design note approved** (revision 2; `K1_DESIGN_REVIEW_r1.md` NOT YET with no error in the design,
    `K1_DESIGN_REVIEW_r2.md` SIGN OFF), with round 2's five minor findings to be folded into the note before part 0 is
    built and checked with part 0's review, not in a third round: the audit of the kind and the order follows one
    level of test helper and lists what it cannot pair (11); part 1 takes over the rewrite of the tests that hand the
    dense reference a form it now refuses (12); the expected red of parts 2 and 3 is recorded per subtest, and part
    4's green run is named as the first check of the engine on cases hidden behind a red node (13); the backend
    `mc_sgd` and `adam` say that a draw's batch is not checked there (14); part 0's interim docstrings state part 0's
    rule for `y`, not the final one (15).
31. **K1 is built on the local side branch `k1-wip`**, one commit per reviewed part, squashed onto `graded-messages`
    when part 5's gate passes (as M2 and P16 were). The branch is pushed to the remote as a backup (the maintainer:
    the session ends at 03:00 CT and nothing should live only on the laptop); CI does not run on it.

## The maintainer's instructions during the arc

- **2026-11-25, on `typ.Any`:** "I have been skimming the code and see a lot of typ.Any's. Many of these are going
  to change to have clear types by the time this K slice arc is over. However, at the end of the arc (once all
  planned K slices are done) I would like to do an audit of typ.Any's in the sampling layer and downstream
  operations, and assess why they are there and see if we can make them strict on a case-by-case basis. Sometimes
  typ.Any may be unavoidable, but often it signals a sub-optimal design." **Done about it now** (Claude): the audit
  is on the list after K7 (`K_ORCHESTRATOR_HANDOFF.md` §1); from K1's part 0 on, each handover lists the `typ.Any`s
  left in the signatures its slice touched, with the reason for each (`K_BRIEF.md` §5), so the audit starts from a
  reasoned list and not from a grep. **The count at K0's tree** (`grep -c 'typ\.Any'`, annotations and the few
  mentions in prose alike): `sampling_definitions` 185, `backend/optimizers` 108, `jets` 47, `sampling_variables` 28,
  `backend/fitting` 28, `sampling_steps` 23, `t3toolbox/optimizers` 19, `data_preparation` 17, `t3toolbox/fitting` 10,
  `uniform_fitting` 6, `geometry` 5, `optimizer_display` 1; none in `probing`, `apply`, `entries`, `ut3_sampling`,
  `utv_sampling`; 492 in the library as a whole.
- **2026-11-25, the same day:** "sampling_definitions is exempt from the typ.Any audit" (the dense reference
  `backend/sampling_definitions.py`, 185 of the 492). The handovers' lists leave it out too.
- **2026-11-25:** "Don't worry about the token cost." The session ends at 03:00 CT on 2026-11-26 and is resumed later
  that day; everything is committed and pushed before it ends.

## K1, the build (2026-11-25 and 26)

32. **Part 0 approved and committed to `k1-wip`** (`K1_part0_HANDOVER.md`; `K1_part0_REVIEW_r1.md` SIGN OFF with five
    minor findings; `K1_part0_RESPONSE_r1.md`). The four fixes made after the sign-off (an aligned comment pair in a
    page, a blank line before a doctest, a test for an untested branch of `_require_draw`, the layout of one module
    docstring) were checked by Claude by a diff of the two diffs and by running what they touch, not by a second
    review round: the reviewer's context was three quarters used, and the changes are four small hunks.
33. **Prose is not re-flowed incidentally** (told to the implementer, written into its carry-over): where a part
    rewrites a sentence of a docstring, the lines that do not change stay byte for byte, so that the diff shows the
    change; a paragraph is re-wrapped only where an inserted sentence forces it. Cause: a wrapping script of the
    implementer's re-filled the `backend/fitting.py` module docstring (and, earlier, long lines already in the test
    files, which it reverted before the handover). The reviewer confirmed by text and token comparison that no other
    line moved. Found by Claude reading the library diff.
34. **Towards precise types, small departures from the approved signatures are the implementer's** (the maintainer's
    `typ.Any` rule): in part 0, `minibatch_scale: typ.Optional[typ.Union[float, NDArray]]` and two typed frontend
    helpers. The reviewer checked each is true under jit. For the audit after K7, noted by the reviewer: the batch's
    `sample`, `y` and weights could be `Union[Sequence[NDArray], NDArray, None]` today; left to K2's `PerSite`.
35. **A third implementer takes part 1** (the second one's context: the design, its two rounds, part 0 and its
    fixes), started on `K_IMPLEMENTER_CARRYOVER.md`; a fourth reviewer will take part 1's review from
    `K_REVIEWER_CARRYOVER.md`.
