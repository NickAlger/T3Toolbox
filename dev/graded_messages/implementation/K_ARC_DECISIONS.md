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
