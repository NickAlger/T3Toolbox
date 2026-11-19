# How the implementation of Part 1 is run, and how to pick it up on another machine

Written by Claude on 2026-11-19, when the maintainer moved the work from a cloud session of Claude to Claude Code on
his laptop. Everything a new session needs is in files; nothing lives in the old session. This page says what the
orchestrating session did for each slice, what was particular to the cloud workspace, and what a session on another
machine has to do first. `BRIEF.md` (beside this file) is the brief of the implementer and the reviewer; `LOG.md`
is the record, slice by slice; `PART1_PLAN.md` section 12 is the process as the maintainer set it.

## 1. Where things stand (2026-11-19)

- Branch `graded-messages`. Slices P0 to P4 are built, reviewed, committed and pushed (P4 is `5991725`).
- The plan was then changed for the generalized jets (one jet group `J`, the tuple `order`, input jets, entries of
  order 0 only): four new slices **P4a to P4d** between P4 and P5, and P5 to P20 brought in line
  (`PART1_PLAN.md`, commit `48a1af4`; its account is `../reviews/jet_group_plan/ACCOUNT.md`).
- **Nothing of P4a to P4d is implemented. The next step is the maintainer's:** his review and sign-off of the changed
  plan as a whole. Do not start P4a before he says so.
- Two things he is to confirm at that review: the keyword `error_table=True` (name N42, taken as approved with the
  other names), and that it is also put on `format_newton_iter` (the writer's addition: that public function is
  where the table is printed).
- After his sign-off, before P4a: amend the decision records (section 5 below).

## 2. What the orchestrating session does for a slice

The maintainer's process (2026-11-16, 20:51; `TARGET_DECISIONS.md`, "How Part 1 is implemented"): an implementer
writes the slice, a reviewer critiques it, they go back and forth until the reviewer signs off; then the
orchestrator looks it over and commits; major issues go to the maintainer, minor ones the orchestrator decides and
records. Both agents were Fable 5.2 on medium effort; the plan itself was written and criticized by two Opus
agents. Agents are spawned only as the maintainer has asked.

What was done each time, in order:

1. **Start a fresh implementer** for the slice. Its prompt: read `BRIEF.md`, `CLAUDE.md`, the plan's sections 1 to 4
   and the slice; the state of the tree (branch, head, clean or not); the notes `LOG.md` carries for this slice
   (section 6 below); how to run the gate; "do not commit; write `P<n>_HANDOVER.md`; report when done".
2. **Start a fresh reviewer** when the implementer hands over (a new agent per slice, so that it has not seen the
   work being made). Its prompt: `BRIEF.md`, the slice in the plan, judge the diff and not the handover, test the
   tests by mutation in a scratch copy (never in the tree), check the mathematics with its own code, write
   `P<n>_REVIEW_r1.md` with the verdict on the first line.
3. **Pass the findings back** to the same implementer, with a ruling on each finding that needs one (which to fix,
   which is the maintainer's, which is left). It answers in `P<n>_RESPONSE_r<k>.md`, reruns the gate, updates the
   handover. Then the same reviewer again. Rounds until `SIGN OFF`. P0 to P4 took one to three rounds each.
4. **Look it over**: read the diff of the library code and the reviews; check names and signatures against the plan;
   where the reviewer left a note that no test pins, add the test (and check in a scratch copy that it fails when
   the code is broken); run the slice's quick tests after any change of one's own.
5. **Write the slice's entry in `LOG.md`**: rounds, what changed, what was decided and by whom, the gate's counts,
   what was not run, notes for later slices, anything for the maintainer.
6. **Commit, staging files by name** (never `-a`: other files may be in the tree), one commit per slice, then push.
   Report to the maintainer only a slice committed, an interruption recovered, or an issue that needs him.

Rules that held throughout: the implementer and the reviewer never run tests at the same time; the gate's steps run
one after another (never sphinx beside a test run: the maintainer has run a 13 GB machine out of memory that way);
agents do not commit; a departure from a decided row, a public name or the order of the slices is a major issue
and goes to the maintainer; nobody's email address goes into any file; T3Polynomial (the consumer) is read, never
changed, and its code is not run before P18.

## 3. What was particular to the cloud session, and does not carry over

- **Agent ids.** `LOG.md` names agents by id (`a015b46da53847b5c` and others). They belonged to that session and
  cannot be resumed from another. Start fresh agents; the handovers, reviews and responses on disk are the context.
- **Scheduled check-ins.** The cloud session's server stopped now and then and ended running agents, so the
  maintainer asked for a scheduled check-in every 45 minutes while agents worked. This is not needed where the
  session does not stop. The "interruption" lines of `LOG.md` are about this.
- **The ten-minute limit on a foreground command**, which is why `run_gate.sh` has `--detach` and `wait`. On a
  machine without that limit the gate can run in the foreground.
- **Paths.** `/home/claude/t3toolbox` and `/home/claude/t3polynomial` in `BRIEF.md`, in the handovers and in the
  plan's section 9 are the cloud workspace's. Read them as "the repository root" and "the maintainer's checkout of
  T3Polynomial". Logs under `/tmp/gate_P<n>` and scripts under `/tmp/p4rev`, `/tmp/critic`, `/tmp/jg` are gone; what
  mattered from them is in the handovers and reviews.
- **The machine and the versions.** 2 cores, 7 to 8 GB; Python 3.13, numpy 2.5, jax 0.12 (ahead of CI). Every count
  and timing in `P0_BASELINE.md` and in the handovers was measured there.
- **The commit trailer.** Commits were authored by NickAlger (the maintainer's rule of 2026-11-16) with
  `Co-Authored-By: Claude Fable 5.2` and a `Claude-Session:` link of the cloud session. The link belongs to that
  session; on the laptop the maintainer's own settings (`~/.claude/`) say how commits are signed.
- **The Claude doc** that mirrored this folder tab by tab (linked from `../README.md`) was last brought in step on
  2026-11-16 and is far behind. The files are the durable copy. Rebuilding it is optional and low priority.

## 4. First steps on another machine

1. **Get the branch.** The history of this arc was rewritten on 2026-11-16 to fix the author of the commits: the
   last two commits of `main` and every commit of `graded-messages` have new hashes and were force-pushed
   (`../COMMIT_HASH_MAP_2026-11-16.md` maps old to new). A clone fetched before that has the old commits. Run
   `git fetch origin`, then compare `git log --oneline -3 origin/main` and `main`: if the local `main` or
   `graded-messages` holds the old hashes, check first that it has no work of its own (`git status`,
   `git log origin/main..main`), and only then move it to the remote's. The remote heads on 2026-11-19: `main` at
   `c96b19d`, `graded-messages` at the commit that added this file. Ask the maintainer before discarding anything.
2. **Use the project's environment.** `CLAUDE.md` says to run tests with the project's env Python and
   `PYTHONPATH=$PWD`; where that Python lives is in the maintainer's `~/.claude/`. `run_gate.sh` calls `python3` and
   sets `PYTHONPATH` itself, so run it with that environment activated (first on `PATH`). The script needs Linux
   (`/proc`, `setsid`). The gate needs pytest, pytest-xdist, pytest-subtests, jax, optax, sphinx, sphinx-autoapi,
   pydata-sphinx-theme and myst-parser.
3. **Take a baseline on that machine before the first slice.** `P0_BASELINE.md` is the cloud workspace's. Run the
   whole gate once on the untouched branch head and keep the result beside it (for example `BASELINE_laptop.md`:
   versions from `versions.py`, counts per step, time, peak memory). Expected on the cloud workspace at this head:
   suite 1,632 passed, 5 skipped (the slow tier), 1 xfailed (the contraction-order marker carried to Part 2); the
   other steps pass with 0 warnings from sphinx. A failure that is in the new baseline and not in the old one is a
   difference of the environment (versions, BLAS): record it and tell the maintainer; it is not a slice's to fix.
   `CLAUDE.md` and `dev/HANDOFF.md` say which local environments twin which CI leg.
4. **Choose the worker count for that machine** (`-n`): the cloud workspace used 2. Whatever it is, one run at a
   time, and never sphinx beside tests.

## 5. Owed before P4a, after the maintainer's sign-off

The decided rows of `../TARGET_DECISIONS.md` and three sections of `../SAMPLE_VARIABLES.md` still read as before the
jet change. Both files carry a banner at the top saying so. The list of amendments is
`../reviews/jet_group_plan/ACCOUNT.md` section 7; it was written before the maintainer answered the open points, so
where it says "if O-45 is confirmed" and the like, take the answers from `PART1_PLAN.md` section 11.1. Amend rows by
adding an amendment under the row, as the earlier amendments were made (rows 4.2, 5.4 and 8.3 show the form); do not
rewrite the history of a row. Until this is done, for anything about jets, `order`, `pp`, the letters `T` and `A`,
or the entries kind: **the notes of 2026-11-17 and `PART1_PLAN.md` win over the rows.**

## 6. Notes carried to slices not yet built

From `LOG.md`, gathered here. Each was written before the jet change; the plan's slice is the authority, and these
are things to check when the slice is reached.

- **P4a.** The expected-failure test of `tests/test_jet_contract.py` (the contraction order on numpy, carried to
  Part 2 as "row 5.5") is restated, not removed. The one-program-size test of the scan in `tests/test_dispatch.py`
  is kept, with its "a leaf has no scan" assertion restated for orders above 1.
- **P5.** The `HISTORICAL` table of `tests/test_contractions_interpreter.py` holds one supplement letter per
  string, and several strings need two or three at library call sites (12 of the old 41; recount on the 29).
  `P1_HANDOVER.md` section 6 gives two ways to change the format. A test format: the orchestrator decides.
- **P8.** The corewise call passes `(B, G)` where a frame goes; the names without a basis (`nu_dot`, `eta_dot`,
  `nu_tilde_dot`, `eta_tilde_dot`; `SAMPLE_VARIABLES.md` section 9) were confirmed by the maintainer (2026-11-17,
  12:02). The reference module takes four stores `(B, G, G, G)`.
- **P9.** The identity tests must not assume the pairings of `mu_tilde` and `xi_tilde` for apply (two of the four
  pairings are of the probe kind only; `SAMPLE_VARIABLES.md` section 8). Add a test that the reference module and
  the engine accept the same names per kind, and tell the maintainer which names the engine accepts.
- **P10.** The dtype rule: in the reference module float64 cores with a float32 sample give float64 that is right
  only to about 9e-8.
- **P11.** `pack_vectors` and `pack_if_ragged` get `len_J=0` (only the leading jet axes are padded with zeros; a
  mismatch in `W` still raises). The functions of `data_preparation` that take `kind` are for the ragged form;
  `rescale_coordinates` cannot take a packed array.
- **P14.** A backend user with a packed batch has no checker (the checkers take the ragged form).
- **P16.** The frame's consistency is enforced once at model build, in safe mode, on a frame the caller supplied.
  The check must pass `safety.effective_rtol(frame.data)`: `is_consistent`'s default `rtol=1e-9` cannot pass a
  float32 frame (it reads about 2.5e-7). The frame residual costs about two and a half frame constructions.
- **P19.** `docs/api_reference.rst` lists backend modules by hand and lacks the new ones; cross-references to the
  functions re-exported from `t3toolbox.fitting` render as plain text; a frontend-only teaching example is wanted;
  the changelog gets `UT3Frame.is_consistent` thresholding against `max(1, ||T||)`, and `entries_derivatives`
  removed (apply with `index_to_one_hot` in its place); the uniform equivalence contract gets the sentence that
  the slots of a jet axis are data; the letters table; the paper-to-code map moves out of `CLAUDE.md` (O-15).
- **Left as they are, on purpose:** the frontend `rescale_coordinates` gives a poor message for a sample of the
  wrong form; a 0-dimensional float32 `length` does not lower the dtype (P3, review r2, findings 18 and 19).
- **Part 2** (the section "Carried to Part 2" of `TARGET_DECISIONS.md`): the contraction order inside one step of
  `jet_contract` on numpy; map-type operations on uniform one site at a time; the benchmark against 2026.5.0. New
  with the jets, and not yet written into that section (it is among the amendments of section 5): the scan over
  several jet axes, with the program size given up for two or more jet variables until then (O-38); and the zero
  slots of an input affine in several variables (its box has `2^m` slots for `m + 1` coefficients, and the simple
  algorithm visits them all; Claude to the maintainer, 2026-11-17, 14:20).
- **After the refactor** (the backlog of `dev/HANDOFF.md`): an accurate norm of a difference of two T3s.

## 7. Things learned the hard way

- **Stale bytecode.** A same-size change to a source file, written and restored within a second, leaves bytecode
  that makes later runs lie. Mutate code only in a scratch copy, or run with `python -B` and
  `PYTHONDONTWRITEBYTECODE=1`.
- **Times in the records come from the clock**, not from memory: take them from `date` or from commit times. Three
  times in this arc a time written from an estimate had to be corrected.
- **A plan agreed by a reviewer can still hold a decision nobody made.** The site group `S` entered row 5.4 that
  way and was removed by the maintainer. His rule since: a new group in the contraction strings, or any new
  abstraction of that weight, is raised with him first.
- **`git add` by name.** A directory-wide `git add` once swept an agent's unfinished draft into a commit.
