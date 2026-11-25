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
   only, and a push starts both CI legs.
7. **Disk.** 8 GB are free (19 GB when `BRIEF.md` was written). The agents delete a gate's Sphinx output once it is
   read and keep kept logs gzipped. To raise with the maintainer if it falls further.
