# Brief for the packing sub-arc (K0 to K7)

Written by Claude (the orchestrator) on 2026-11-25, for the implementer and the reviewer of the slices K0 to K7, and
for any agent that replaces one of them. Read this first, then `BRIEF.md` beside it (the brief of the P and M slices),
which still holds except where this page says otherwise.

## 1. The process, in the maintainer's words (2026-11-25)

"The implementation will take the normal approach: an orchestrator oversees two Opus 5.5 agents on high effort, an
implementer and a reviewer, both of which are instructed to get familiar with the library before starting, and to use
judgement and first principles thinking rather than blindly following rules. Each slice includes a plan phase and an
implementation phase. In the plan phase the implementer writes a more detailed plan about the specific slice, the
reviewer critiques it, and they go back and forth until the reviewer signs off. Then they send it to the
orchestrator, who can approve or send it back. Once approved, the implementer implements the slice, the reviewer
reviews their implementation, and they go back and forth. Once the reviewer signs off, they send it to the
orchestrator who either approves and initiates the next slice, or sends it back for more work. The orchestrator is
advised to use their judgement for decisions along the way, except for big decisions or major unforseen problems
which they can pause and talk to me about. The orchestrator logs all decisions, and then after the entire arc is
complete, discusses it with me."

**Precedence.** The maintainer: "my instructions here in chat supercede those in BRIEF.md and ORCHESTRATION.md
wherever they conflict." The paragraph above is the process. `BRIEF.md` and `ORCHESTRATION.md` are the older
practice: they supply what the paragraph does not say (the gate, the form of the records, the traps), and where they
differ from it, it wins.

## 2. What to follow

- **`dev/graded_messages/PACKING_PLAN.md` (revision 5) is the specification.** It is signed off by its critic and by
  the maintainer. Read it whole once; then the slice you are on (§7) with the sections it points to. Its §11 to §14
  list who decided what.
- **`dev/graded_messages/PACKING_PLAN_NOTES.md`** has the rulings in the maintainer's words (§7 to §9 are his review
  of the plan). Where the plan and the notes disagree, the notes win; say so when you find a case.
- `PART1_PLAN.md`, `TARGET_DECISIONS.md` and `SAMPLE_VARIABLES.md` are the background of the engine the arc works on
  (`SAMPLE_VARIABLES.md` is still the reference for every sampling variable). The packing plan supersedes them where
  it says so by name.
- **`CLAUDE.md` at the repository root binds every line,** as in `BRIEF.md`. Its "Current state" section describes
  the released library; the branch is ahead of it.
- The plan's `file:line` references are at commit `3375d213`. **Every slice measures again on its own tree.** A claim
  about the code is made with `file:line` at a stated commit, a count with its method.

## 3. How to think

The maintainer asks for judgement and first principles, not rules followed blindly; simplicity, performance and a
coherent mathematical framework are the goals. The plan was made in conversation and reviewed four times, and each
round still found real seams (a slice order that would have turned the frontend red; a wrapper class that broke the
shared batch; a metric that could not check its block weight). Expect the same here:

- Know the reason of a rule before applying it. When the reason does not fit the case in front of you, say so.
- **Small choices are yours.** Where the plan leaves a choice open, or a detail of it is plainly wrong for the code as
  it is, decide, and record the decision and its reason (a section "Decisions" in the design note or the handover).
- **Stop and report to the orchestrator** (with the reasoning and a recommendation, not a bare question) for: a
  departure from the plan's design or from a ruling in the notes; a public name or signature the plan does not have;
  a change of the order or the scope of the slices (the scope is sampling and what is downstream of it); a silent
  wrong answer found in released behaviour; a test that can only pass by being weakened; anything that needs a new
  principle. This replaces `BRIEF.md`'s "Issues" paragraph, which asked you to decide nothing.
- The reviewer checks claims by running small experiments of its own, as the plan's critic did
  (`dev/graded_messages/packing_audit/critic_experiments/` are models). An agreement is worth what was checked.

## 4. The two phases of a slice, and their records

All records go in `dev/graded_messages/implementation/`; working files of a slice (scripts, logs worth keeping,
expected-red lists) in a folder `k<n>/` there.

**The plan phase.** No file under `t3toolbox/`, `tests/`, `docs/` or `examples/` changes in this phase.

1. The implementer writes `K<n>_DESIGN_NOTE.md`: what changes, file by file and function by function, with exact
   signatures where there are any; what is measured on today's tree; the tests and what each is checked against
   (never the engine against itself); what breaks and how it is repaired; the order of the work, and the parts if the
   slice is built in parts; the gate and the evidence (batteries, tallies, mutation passes); the decisions taken; the
   open points for the orchestrator.
2. The reviewer writes `K<n>_DESIGN_REVIEW_r<k>.md`, the verdict on the first line: `SIGN OFF` or `NOT YET`. Findings
   are numbered, each with a severity (blocking, should fix, minor), the place, the evidence and what would resolve
   it. Points that need a ruling are marked "for the orchestrator".
3. The implementer revises the note in place and answers each finding in `K<n>_DESIGN_RESPONSE_r<k>.md` (accepted and
   what changed, or not accepted and why, with evidence). Rounds until `SIGN OFF`.
4. The orchestrator approves the note or sends it back.

**The implementation phase**, as in `BRIEF.md` ("What the implementer hands over", "What the reviewer does"), with
the names `K<n>_HANDOVER.md`, `K<n>_REVIEW_r<k>.md`, `K<n>_RESPONSE_r<k>.md` (and `K<n>_part<j>_...` when a slice is
built in parts). The implementer builds what the approved note says; a departure found while building is reported in
the handover, or at once if it is one of the cases of section 3. The orchestrator approves the slice or sends it back,
then writes the log entry, commits and pushes, and reads both CI legs.

The orchestrator relays between the two agents: each turn ends with a report to the orchestrator, who starts the
other agent on it. Keep what matters in the files, not in the conversation: a replacement, and the maintainer, read
the files.

## 5. Rules for both agents

`BRIEF.md`'s "Rules for both agents" hold: **do not commit, push, switch branches or rewrite history**; change only
what the slice changes; `subTest` labels are builtins; never wait with `pgrep -f`; "the same value" means equal to
within rounding, with bitwise identity given low weight when designs are weighed; do not weaken a test; doctest
output is pasted from a real run; no email address in any file. In addition:

- **Tests are run by one agent at a time.** The orchestrator's message says whose turn it is. When both agents are
  active at once (the orientation), nobody runs the test suite unless the orchestrator's message says so; a quick
  script that imports the library is always fine.
- **The machine:** 8 cores, 13 GB. The suite peaks near 10 GB at `-n 2`, so: the gate at `-n 2`, one heavy run at a
  time, never Sphinx beside tests, nothing left running at the end of a turn. Run the gate detached and wait for it
  (`run_gate.sh --detach`, then `run_gate.sh wait`), since a foreground command is cut off after ten minutes:
  `PATH=/home/nick/miniconda3/envs/t3toolbox/bin:$PATH dev/graded_messages/implementation/run_gate.sh --detach -n 2 -l <logdir>`.
- **Disk: about 8 GB free** (measured 2026-11-25; less than `BRIEF.md` says). Delete a gate's `sphinx_html` and
  `sphinx_doctrees` once its result is read; delete scratch copies of the code when done; keep logs small (gzip a
  `-v` log that is kept).
- **Scratch work** goes under
  `/tmp/claude-1000/-home-nick-repos-T3Toolbox/7a03b046-9177-4c43-923a-b31595edb27e/scratchpad/<your role>/`
  (`BRIEF.md` names an older session's folder). Mutate code only in a scratch copy of `t3toolbox/` and `tests/`, and
  run with `python -B` there (stale bytecode has lied before).
- **The reference for "no new failure"** is the tree the slice starts from. The last full gate on record is M5's
  (`LOG.md`: suite 1,979 passed, every doc step green, Sphinx with no warning); `BASELINE_laptop.md` is far older.
  A slice that changes which tests exist records the per-test outcome before and after (a `-v` run of the starting
  tree, kept gzipped in `k<n>/`, and a tally of the differences, as `m0/tally.py` does).
- **The floor** (Python 3.9, numpy 1.26, jax 0.4.30) is checked by CI after the push; write for it. The local floor
  environment `tttt` is only for reproducing a failure CI reported.
- **Expected-red lists are recorded by a run,** never predicted.
- **Comments are content.** The signature shape comments, their aligned columns and the ASCII diagrams are part of
  the code's contract. A scripted edit that touches comments is checked for them before it is called mechanical.
- **`typ.Any` in a signature is a debt, and it is counted** (the maintainer, 2026-11-25: after K7 he audits every
  `typ.Any` of the sampling layer and what is downstream, case by case; "sometimes typ.Any may be unavoidable, but
  often it signals a sub-optimal design"). When a slice writes or changes a signature, annotate what Python can
  express (`CLAUDE.md`, "Signature shape comments": real `Union` / `Optional`, a sequence type, the `PerSite`
  `TypeVar` from K2 on) and do not reach for `typ.Any` out of habit. Each handover lists the `typ.Any`s left in the
  signatures the slice touched, each with its reason (a per-mode object in two forms; a pytree of the caller's; a
  duck-typed batch; ...). Do not go hunting for the ones a slice does not touch: that is the audit's.
- **The planned pauses of `BRIEF.md` do not apply.** The orchestrator pauses for the maintainer when section 3's
  cases need him.
- **When your context is getting long,** say so in your report, and write what a successor needs in
  `K_IMPLEMENTER_CARRYOVER.md` or `K_REVIEWER_CARRYOVER.md` (what you know that is not in the slice's records).

## 6. Traps of this arc (each was found by a review, or would have shipped)

- A size-1 axis that broadcasts where an exact size was meant. In this arc a size of 1 is legal on the stack axes
  `W` and `C` only (the plan's §2.2).
- Junk in the uniform padding reaching a reduction. Every packed operand is tested with junk in its padding; from K5
  with `inf` and `nan` in the padding of the data.
- A test that compares a function with itself, or the engine with a helper that calls the engine. K0 removes the old
  kernels, which several staying tests use as their oracle.
- Masking by selection (`where(mask, x, 0)`): the zero must have the array's dtype (on numpy 2 a `np.float64(0)`
  promotes a float32 array); never boolean indexing; a comparison with the tree before is on values, not bytes (the
  two maskings differ in the sign of a zero).
- An `ast` scan that reads top-level definitions only misses a reference inside a module-level `if` or a decorator
  (`K_MEASUREMENTS.md` M1(a2) lists `_BATCH_FIELDS` as unreferenced; the batch's pytree registration uses it).
- A swapped pair of same-shaped arguments; a float64 constant on a float32 run; numpy reading an object's `.shape`
  attribute as an array shape.
- `jnp.einsum`'s default `optimize` differs between jax generations (`ORCHESTRATION.md` §7).
- The program-size gate (K5) measures the optimized program at `d = 3` and `5`, never `1` or `2`.

## 7. What each report to the orchestrator says

Short. The file you wrote; the verdict, if you are the reviewer; the decisions you took; the points that need the
orchestrator; what you ran, and that nothing is left running; whether your context is getting long. In your first
report, the methods of `TuckerTensorTrain` you read and traced (`BRIEF.md`, "Required reading").
