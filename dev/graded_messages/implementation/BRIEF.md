# Brief for the implementation of Part 1

Written by Claude on 2026-11-16, after the maintainer approved `PART1_PLAN.md` ("Plan approved. Go ahead.",
21:16). For two agents: the **implementer**, who writes one slice at a time, and the **reviewer**, who critiques
it. They go back and forth until the reviewer signs off on the slice. Claude then looks over the slice and commits
it; the next slice starts only after that.

## What to follow

- **`dev/graded_messages/PART1_PLAN.md` says what to build and how.** Read sections 1 to 4 once, then the slice
  you are on (section 5), then the parts of sections 6 to 10 it points to. Section 12 is the process.
- Behind the plan stand the decided rows and notes of `dev/graded_messages/TARGET_DECISIONS.md`, and
  `dev/graded_messages/SAMPLE_VARIABLES.md` for every variable's name, definition, recurrence and shape. Where the
  plan and a decided row disagree, stop and report it (see "Issues").
- **`CLAUDE.md` at the repository root is the project guide and its rules bind every line you write**: the razor
  between backend and frontend, the signature shape comments, names, scan bodies at module level, the dispatch
  between numpy and jax, "masks are numpy", how tests and doctests are written, and **no auto-formatter**. The
  contributor notes it points to (`docs/contributor/`) are the reference for the form of the code. The prototypes
  in `dev/graded_messages/scripts/review/` give the algebra and the names; they are **not** the form of the code.
- The maintainer's rules that have gone wrong before, so check them every time: the backend has no safe mode and
  gives checkers, not guarantees; the frontend enforces. One set of step functions for both layouts. Dispatch is
  inferred from the arrays, never threaded as a flag. Uniform masks are host numpy. Shared Tucker factors keep
  working. Nothing a backend user needs lives only in the frontend. Scan bodies are module-level functions or
  memoized factories. Code is statically readable: literal contraction strings at the call sites.

## The workspace

*(This section describes the cloud workspace in which P0 to P4 were built. On another machine the rules stand
(one run at a time; never sphinx beside tests; the consumer is read-only) and the particulars differ: the paths, the
versions, the baseline, the ten-minute limit. See `ORCHESTRATION.md` beside this file, sections 3 and 4.)*

- The repository is `/home/claude/t3toolbox`, on the branch `graded-messages`. Run Python as
  `PYTHONPATH=/home/claude/t3toolbox python3 ...` from the repository root.
- Python 3.13, numpy 2.5, jax 0.12. **This is ahead of what CI runs** (CI's current leg is Python 3.11, numpy 2.4,
  jax 0.10), and `CLAUDE.md` warns that results depending on the BLAS have differed. So the baseline of P0 (what
  passes and fails here on the unchanged tree) is the reference: a slice must not add a failure, and a failure that
  is already in the baseline is not the slice's to fix. Record it, do not hide it.
- The machine has **2 cores and 7 GB**. The maintainer has run out of memory on a larger machine by running sphinx
  and a parallel test run together. So: **the gate's steps run one after another, never two at once**; do not
  start a long run while another is going; use the worker count that P0 measured as safe. The implementer and the
  reviewer never run tests at the same time (they alternate, so this holds if nobody leaves a run in the
  background at the end of their turn: do not).
- A foreground command is cut off after ten minutes. Use the gate script that P0 writes
  (`dev/graded_messages/implementation/run_gate.sh`), which runs the steps in pieces that fit, and writes logs.
- `/home/claude/t3polynomial` is the consumer. Read it; never change it; do not run its code unless the slice
  says so (P18).

## Rules for both agents

- **Do not commit, push, switch branches or rewrite history.** Claude commits. Leave the working tree as the
  result of your work.
- Change only what the slice changes. If something outside the slice is wrong, write it down; do not fix it.
- Do not weaken a test to make it pass. Tolerance tests for numerical things, exact tests for structural things.
  Verify a mathematical property with a quick script before asserting it in a test. Doctest output is pasted from
  a real run, never written by hand.
- **Issues.** If the slice cannot be done as the plan says; if the plan contradicts a decided row or the code; if
  doing it right needs a change to a decided row, a public name or signature, or the order or scope of the slices;
  or if a test can only pass by being weakened: **stop and report it.** Do not decide it. Smaller choices the plan
  leaves to the implementer are yours; say what you chose in the handover.
- Do not put anyone's email address in any file.

## What the implementer hands over

`dev/graded_messages/implementation/P<n>_HANDOVER.md`:

1. What changed: files and functions, added, changed, deleted.
2. The tests added or rewritten, and what each establishes.
3. The gate: each command and its result, copied from the logs (counts of passed, failed, skipped; the failures
   by name, each marked as in the baseline or new).
4. Departures from the plan, each with its reason; choices the plan left open and what was chosen.
5. Anything found outside the slice that should be looked at.

After a review round, the implementer answers each finding in `P<n>_RESPONSE_r<k>.md` (accepted and what changed,
or not accepted and why, with evidence) and updates the handover.

## What the reviewer does

`dev/graded_messages/implementation/P<n>_REVIEW_r<k>.md`, with the verdict on the first line: `SIGN OFF` or
`NOT YET`.

- Read the slice in the plan, then the diff (`git status`, `git diff`, and the new files), then the handover. Judge
  the code, not the handover's account of it.
- Check against the plan and the decided rows (names, signatures, argument order, what is checked where). Check
  against `CLAUDE.md` and the contributor notes for form. Look for silent decisions, for logic in the wrong tier,
  for a numpy call on what can be a jax array, for a mask treated as data, for a closure where a module-level body
  is required.
- **Test the tests:** would each new test fail if the code were wrong? Perturb the code or the expectation in a
  scratch copy and see (and restore the tree exactly). Run the slice's tests yourself, and the gate where the
  slice requires it, respecting the memory rule.
- Check the mathematics independently where the slice has any: against the definitions in
  `SAMPLE_VARIABLES.md`, by your own small dense computation, not by re-running the implementer's test.
- Findings are numbered, each with a severity (blocking, should fix, minor), the place, the evidence and what
  would resolve it. Sign off only when no blocking or should-fix finding remains. Do not fix the code yourself.
- Leave the working tree exactly as you found it, apart from your review file.
