SIGN OFF

# P0: review, round 2

Reviewer, 2026-11-16, on `graded-messages` at `ad8c1c7`. All four should-fix findings of round 1 are resolved,
tested by me. Two new minor findings (9, 10) are listed for the implementer to fix without another round.

## Findings 1 to 8

| # | Round 1 | State | How I tested it |
| --- | --- | --- | --- |
| 1 | `stop` removes the lock whatever run holds it | resolved | With a detached run live, `stop` on another directory printed "nothing was signalled", and the lock stayed. A second run and a `--detach` onto the live directory were both refused (exit 2), and the live run's `PID` file was kept. The live run then finished with PASS |
| 2 | a run is identified by a bare pid | resolved | Five stale records, each put into both `LOGDIR/PID` and the lock: bare `1`; the bare pid of a live `sleep`; that pid with a wrong start time; that pid with its true start time and another boot id; pid 1 with its true start time and another boot id. In every case `wait` exited 2, `stop` signalled nothing (the `sleep` and pid 1 alive afterwards), and a new run started, passed and left no lock |
| 3 | star patterns in backticks not listed | resolved | A made-up page: `` `compute_*` ``, ``(`*_from_sweep`)`` and ``` ``tv_*_transpose`` ``` are listed; `**bold**`, `*emphasis*`, `a*b` and `2*chunk_size` are not. On the tree the table has `*_from_sweep` 9, `*_model` 4, `*_derivatives` 8, `ut3_*` 10 |
| 4 | the root `README.md` not searched | resolved | 12 occurrences, among them the four calls (lines 72 to 74, 84) and the kind string of line 87. The plan gap is in the handover (2b), and `PART1_PLAN.md` now has the README in P17 (see "The tree") |
| 5 | non-ASCII before a name loses the kind | resolved | My two lines and three more (a dictionary with non-ASCII keys, a doctest line, a string): `call`, `ref`, `kwarg` and the bindings are right in each |
| 6 | `doctest-pages` passes on an empty glob | left as CI's behaviour, documented | `P0_BASELINE.md`, "Known limits", states 48 pages (47 after P13). Accepted |
| 7 | other gate behaviours | resolved | `-n two`, `-n 0`, `-t x`: exit 2 with a message. `kill -9` of the gate: 5 processes left, `wait` exit 2 with the hint, a new run refused, `stop` on another directory left the lock, `stop` on the run's directory left 0 processes and no lock. `-t 6` on `tests/test_dispatch.py` with two workers: FAIL "stopped at the time limit of 6s.", no process, no lock. But see 9 and 10 |
| 8 | statements in the notes | resolved | 616 in the handover and the inventory note; the 47 star imports are in the docstring and the handover; `import apply as bapply` is `import-module` |

## What the changes could have broken: nothing found

Planted again in a scratch worktree, each through the changed script:

| Planted | Step | Outcome |
| --- | --- | --- |
| a test where only one `subTest` fails, `-n 2` | `quick` | FAIL, `1 failed, 2 passed, 2 subtests passed`, exit 1 |
| a wrong expected output in `t3toolbox/safety.py` | `doctest-modules` | FAIL, `1 failed, 240 passed` |
| a wrong expected output in the last page (`docs/weighting.md`) | `doctest-pages` | FAIL, `48 file(s), 1 with failures` |
| the first of three examples raises | `examples` | FAIL, `3 examples, 1 failed`, exit 1 |
| a link to a page that does not exist | `sphinx` | FAIL, `1 warning/error line(s)`, exit 1 |

In the same run as the two planted doctest failures, `doctest-quickstart` passed, and the run exited 1.

- The cheap steps pass on the unchanged content (241 passed; 1 file; 48 files).
- `quick` runs exactly what it is given: a node id plus two files gave 21 passed; three files gave 30 passed and
  67 subtests with one worker and with two, the same as plain pytest on those files.
- SIGTERM to a foreground run with two workers, and SIGINT to a foreground `doctest-modules`: exit 143, no
  process, no lock, no `DONE`; `wait` on that directory exits 2.
- The foreground exit code is the run's (0, 1 and 2 seen).

## The inventory

- `P0_INVENTORY.md` reproduces exactly from the script.
- Taking `docs/release_notes.md` out changed no count the plan relies on. I compared the per-name tables of the
  round 1 script's output with the new one: nine names differ, only in the `text` column (page prose) and the
  total; no `def`, `import`, `call`, `kw`, `ref`, doctest or `string` cell changed. The total moved from 7,387
  to 7,390 (9 out with `release_notes.md`, 12 in with `README.md`). My AST-only count still matches (1,503
  calls, 395 keyword arguments). The exclusion follows the plan's section 10 (line 1807), which leaves
  `CHANGELOG.md` and `docs/release_notes.md` out of the final search.

## New findings

### 9. Minor: an option without its value hangs the script

Place: `run_gate.sh` lines 157 to 159 (`-n) WORKERS="${2:-}"; shift 2`).

Evidence: `run_gate.sh -n` (nothing after it) loops without end at full CPU: `shift 2` fails when one argument
is left, so the loop never advances. The same holds for a trailing `-l` or `-t`. It starts no step and takes no
lock. The round 1 script exited with "unbound variable" here.

Resolve: check that a value follows (`[ $# -ge 2 ]`), else exit 2.

### 10. Minor: under the time limit, a descendant that ignores SIGTERM outlives its step unnoticed

Place: `timed.py`, the sampler (the SIGKILL after ten seconds is sent only while the step's main process is
still alive).

Evidence: a made-up example that starts a child ignoring SIGTERM, run with `-t 2`: the example is recorded FAIL,
the gate goes on and finishes, the lock is released, `wait` exits 1, and the child is still running; a new run
then starts beside it. The suite's own workers do end (seen with two workers), so this needs a process that
ignores SIGTERM. Without `-t` the same holds for any step that leaves a process behind, as before.

Resolve (optional): after the main process ends on a time limit, send SIGKILL to what is left of its tree; or
have the gate check its process group for leftovers after each step and say so.

## The tree

- `git status`: the nine files under `dev/graded_messages/implementation/` (and this review), and one modified
  file, `dev/graded_messages/PART1_PLAN.md`, changed at 23:19 during this round, not by me. Its diff is the
  three counts corrected as P0 found them (`_wchunked_sum` called from four functions; two glossary links; seven
  factory calls) and the root `README.md` added to P17 and to P17's final search. Nothing in `t3toolbox/`,
  `tests/`, `docs/` or `examples/` changed, so the round 1 baseline stands and I did not rerun the full suite.
- No process running, no lock file, the scratch worktree removed (`git worktree list` shows one tree), no build
  output.
