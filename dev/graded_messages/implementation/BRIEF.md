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

## The workspace (the maintainer's laptop, from P4a on, 2026-11-19)

*(P0 to P4 were built in a cloud workspace: `/home/claude/t3toolbox`, 2 cores, 7 GB, Python 3.13, numpy 2.5, jax 0.12.
Paths of that form in the older handovers, reviews and the plan's section 9 mean "the repository root" and "the
maintainer's checkout of T3Polynomial". `ORCHESTRATION.md` section 3 says what else does not carry over.)*

- The repository is `/home/nick/repos/T3Toolbox`, on the branch `graded-messages`. Run Python with the project's
  environment and the repository on the path, from the repository root:
  `PYTHONPATH=$PWD /home/nick/miniconda3/envs/t3toolbox/bin/python ...` (Python 3.11, numpy 2.4.6, jax 0.10.2: the
  twin of CI's current leg). The gate script calls `python3`, so put the environment first on `PATH`:
  `PATH=/home/nick/miniconda3/envs/t3toolbox/bin:$PATH dev/graded_messages/implementation/run_gate.sh -n 3 -l <logdir>`.
  An agent's shell still stops a foreground command after ten minutes, so run the gate with `--detach` and `wait`
  (the script's header shows how), and leave nothing running at the end of a turn.
- **The reference is the laptop baseline**, `BASELINE_laptop.md` beside this file, taken on the untouched head
  before P4a: what passes and fails here. A slice must not add a failure; one already in the baseline is not the
  slice's to fix. Record it, do not hide it.
- **The floor is checked by CI, not locally.** A draft pull request (#26) runs CI on every push of the branch: the
  current leg and the floor leg (Python 3.9, numpy 1.26, jax 0.4.30), the examples and the docs build. Claude pushes
  after each commit and reads the result. So write code that runs on Python 3.9 and numpy 1.x: no `X | Y` type unions
  evaluated at run time, no `match`, nothing new in numpy 2. (A local floor environment exists, `tttt`: Python 3.9,
  numpy 1.22, jax 0.4.30, no xdist, no optax. Use it only to reproduce a floor failure CI reported.)
- **The machine has 8 cores and 13 GB**, and memory is the limit: the maintainer has run it out of memory with
  sphinx beside a parallel test run. So: **at most `-n 3`** test workers; the gate's steps one after another, never
  two at once; never start a long run while another is going; the implementer and the reviewer never run tests at the
  same time (they alternate, so this holds if nobody leaves a run in the background at the end of their turn: do not).
  A quick script that imports jax is fine; a second test run is not.
- **Disk:** about 19 GB free. Put scratch work (scripts, logs, copies) under
  `/tmp/claude-1000/-home-nick-repos-T3Toolbox/352ad37b-5ac8-40a1-a326-a7fc3a369cb2/scratchpad/<your role>/`. A copy of
  the code for mutation testing holds `t3toolbox/` and `tests/` only (about 35 MB), not the repository (377 MB, mostly
  `docs/`); delete it when done. Delete a gate's `sphinx_html` and `sphinx_doctrees` once its result is read.
- `/home/nick/repos/T3Polynomial` is the consumer. Read it; never change it; do not run its code before
  P18.

## Required reading, once, before the first slice

The maintainer (2026-11-19): pick a handful of methods of the flagship class `TuckerTensorTrain`
(`t3toolbox/tucker_tensor_train.py`), read those methods and the backend functions they call, all the way down.
This gives the style and the conventions of the library: the frontend as a thin wrapper over the backend; the
signature shape comments; the module-level scan and map bodies (`xscan`, `xmap` from `common.get_backend`); dispatch
inferred from the arrays; one literal grouped contraction string per contraction; doctests that check against dense
ground truth. Choose the methods yourself; `probe` and `probe_derivatives` (the jets of today) are good among them.

## How the two agents work with Claude (from P4a on)

The same implementer and reviewer are kept from slice to slice (the maintainer, 2026-11-19: "Keep the same agents
going so they have more context"). Claude relays between them: the implementer reports to Claude when its handover
is written; Claude starts the reviewer on it; the reviewer reports its verdict to Claude; Claude passes the findings
to the implementer with a ruling on any that needs one; and so on until `SIGN OFF`. Then Claude looks the slice over
and commits it, or sends it back. Then the same pair takes the next slice. Keep your notes in your files, not only in
your memory of the conversation: a slice's handover, reviews and responses are the record, and a later slice may be
read by someone who has not seen the conversation.

The work stops at three planned pauses for the maintainer (after P4d, after P11, after P15). In between, small and
clear issues are decided (by the implementer where the plan leaves the choice to it, by Claude otherwise) and
written down; anything large or uncertain is raised (see "Issues"). The maintainer: "think from first principles
about what we are trying to do here, and use judgement rather than blindly following rules. Please go slow and do
things right rather than cutting corners for expediency."

## Rules for both agents

- **Do not commit, push, switch branches or rewrite history.** Claude commits. Leave the working tree as the
  result of your work.
- Change only what the slice changes. If something outside the slice is wrong, write it down; do not fix it.
- **`subTest` labels are builtins** (`str`, `int`, `float`, `bool`, `None`, and tuples of them): CI's floor leg runs
  xdist on numpy 1.x, which cannot send a numpy scalar or array as a label, and the test then fails there only
  (neither local environment shows it). Write `self.subTest(order=repr(order))` where a value may be numpy. This bit
  P4a and P4c on 2026-11-19. Since 2026-11-20 `tests/conftest.py` enforces it everywhere: a label that is not a
  builtin raises `TypeError` at the `subTest` call.
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
