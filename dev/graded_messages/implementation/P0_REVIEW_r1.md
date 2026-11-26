NOT YET

# P0: review, round 1

Reviewer, 2026-11-16, on `graded-messages` at `ad8c1c7`. The baseline itself is confirmed: I reran every gate
step and both check suites and got the handover's results. Four should-fix findings remain, two in the gate
script's lock and run identity, two in the inventory. None is blocking; all are small to resolve.

## What I ran myself

All runs strictly one after another; nothing left running; scratch worktree removed.

| What | Where | Result |
| --- | --- | --- |
| `run_gate.sh doctest-modules doctest-quickstart doctest-pages` | the real tree | pass: 241 passed; 1 file; 48 files |
| `pytest tests/ --collect-only -q` | the real tree | 1,412 collected = 1,407 + 5 skips |
| `run_gate.sh --detach examples sphinx`, then `wait` | a scratch worktree at `ad8c1c7` | pass: 14 of 14; sphinx 0 warnings; 215 s and 188 s |
| `run_gate.sh --detach -n 2 suite`, then `wait` twice | the real tree | pass: 1407 passed, 5 skipped, 1 warning, 49058 subtests passed; 1,152.6 s; peak 3,113 MB; lowest free 4,777 MB. No restart |
| `run_all.sh`, `run_review_checks.sh`, diffed with the recorded outputs | the real tree | 372 lines identical; 355 of 356 identical, line 95 `8e-16` against `7e-16` |
| the slow command with `--collect-only` | the real tree | 66 tests collected (the archive records 66 passed) |
| planted failures, each through the gate script (scratch worktree) | see below | every one makes its step FAIL and the run exit 1 |
| lock, `wait`, `stop`, a killed run, a stale PID | scratch worktree | see findings 1 and 2 |
| `inventory.py` on made-up files (`.py`, `.md`, `.rst`), and against my own AST-only counter on the real tree | scratch worktree, real tree | see findings 3 to 5 |

Planted failures (the script takes its repository from its own location, so a copy of the five script files in
the worktree runs against the worktree):

| Planted | Step | Outcome |
| --- | --- | --- |
| a test where only one `subTest` fails | `quick`, `-n 1` and `-n 2` | FAIL, `1 failed, 2 passed, 2 subtests passed` |
| a test file with an import error | `quick` | FAIL, `1 error` |
| a file that does not exist | `quick` | FAIL, `no tests ran` |
| a node id plus a file | `quick` | runs exactly those (16 passed) |
| a wrong expected output in `t3toolbox/safety.py` | `doctest-modules` | FAIL, `1 failed, 240 passed` |
| a wrong expected output in `docs/getting_started.rst` | `doctest-quickstart` | FAIL |
| a wrong expected output in the last page of the list (`docs/weighting.md`) | `doctest-pages` | FAIL (the last file is reached) |
| the first of three examples raises, the others pass | `examples` | FAIL, `3 examples, 1 failed: ...`; the later passes do not clear it |
| a link to a page that does not exist | `sphinx` | FAIL, `1 warning/error line(s)` |

The gate's commands are CI's (`tests.yaml`, `build-sphinx-docs.yaml`) apart from the stated additions
(`-rfEs -p no:cacheprovider`, the worker count, sphinx's output and doctree directories). Results come from exit
codes, not from parsing logs. After `examples` and `sphinx` the worktree held no build output
(`git status --ignored`: only `__pycache__`).

## Findings

### 1. Should fix: `stop` removes the lock whatever run holds it

Place: `run_gate.sh` line 104 (`rm -f "$LOCK"`, unconditional).

Evidence: with a detached run going in `/tmp/rv_copy_clean` (lock held by pid 2994), I ran
`run_gate.sh stop /tmp/rv_second`, a directory with no run. It printed "no run is going in /tmp/rv_second" and
the lock file was gone while the first run was still going. A second run could then start beside it. The lock is
the script's only guard for the memory rule.

Resolve: remove the lock only when the pid in it is the pid in `LOGDIR/PID` (or is dead).

### 2. Should fix: a run is identified by a bare pid, which is wrong after a restart

Place: `run_gate.sh` lines 76 (`wait`), 95 to 96 (`stop`), 152 (the lock).

Evidence: `/tmp` survives a restart of this machine (the baseline says so, and pids start again from low
numbers). I put the pid of a live unrelated process (1) into `LOGDIR/PID`: `wait LOGDIR 3` answered "still
running after 3s" (exit 3), not exit 2. With the same pid in the lock file, a new run was refused ("another gate
run is going (pid 1)"). In that state `stop LOGDIR` would send `kill -TERM -- -<pid>` to an unrelated process
group (not exercised). The restart is the one failure P0 actually met, and `P0_BASELINE.md` tells later slices
that `wait` reports it with exit 2; that holds only when the old pid is not in use again.

Resolve: record something that identifies the run besides the pid (the process start time from
`/proc/PID/stat`, or the boot id from `/proc/sys/kernel/random/boot_id`) and compare it in `wait`, `stop` and the
lock check; or check that `/proc/PID/cmdline` is this script.

### 3. Should fix: star patterns written in backticks are not listed

Place: `inventory.py`, the `STAR` pattern (the lookbehind excludes a backtick), and the table "Star patterns in
prose" in `P0_INVENTORY.md`.

Evidence: `STAR.finditer` finds nothing in ``the `compute_*` kernels``, ``(`*_from_sweep`)`` or
``` ``tv_*_transpose`` ```, and finds `compute_*` in `the compute_* kernels`. On the tree, my grep finds 60
backticked star patterns over the name families, among them `` `*_from_sweep` `` 9 times, `` `*_model` `` 6 and
`` `*_derivatives` `` 6; the table lists `*_derivatives` as written twice and does not have `*_from_sweep` or
`*_model` in backticks at all. The script's docstring says the matching star patterns are listed. Section 5.1
asks each slice's inventory for glob prose, so later slices will rely on this table.

Resolve: let a pattern start after a backtick; regenerate the tables.

### 4. Should fix: the root `README.md` is not searched, and it holds the old call forms

Place: `inventory.py`, `ROOTS`; `P0_INVENTORY.md`, "Searched".

Evidence: `README.md` lines 72 to 74 and 84 call `x.probe(ww)`, `x.apply(ww)`, `x.entries(...)`, `A.apply(ww)`,
and line 87 is `t3t.newton_cg(t3t.MANIFOLD, 'apply', ww, b, x0, max_newton=30)`, the optimizer call form that
P16 and P17 change. `CHANGELOG.md` has 60 lines with these names. The implementer followed the plan's list
(`t3toolbox/`, `tests/`, `docs/`, `examples/`, `CLAUDE.md`), so this is a gap in the plan, not a departure:
no slice names the root `README.md`, and no CI step runs it. It needs Claude's decision on which slice rewrites
it (P17 by its content).

Resolve: add `README.md` (and `CHANGELOG.md`, or say why not) to the searched files and regenerate; record the
plan gap in the handover's section 5.

### 5. Minor: an identifier after a non-ASCII character on the same line loses its kind

Place: `inventory.py`, `lookup` (token columns count characters, `ast` columns count UTF-8 bytes).

Evidence: on made-up lines, `print('‖r‖ =', x.probe(ww))` and `print('ω', compute_xi(1), chunk_size)` are
reported as `ref` with binding `-`, not `call`. No line of the tree is affected today: my AST-only counter gives
1,503 calls and 395 keyword arguments in code, the inventory gives the same, with no per-name difference, and no
call in the tree has a non-ASCII character before it on its line. A later slice could add one.

Resolve: convert the byte offset to a character offset, or note the limit under "What it cannot see".

### 6. Minor: `doctest-pages` passes when one of its two globs matches nothing

Place: `run_gate.sh` line 223.

Evidence: with `docs/contributor` moved aside the step printed `ls: cannot access` to the terminal and recorded
`PASS 23 file(s)`. This is CI's own command and behaviour. The page count is shown but compared with nothing.

Resolve (optional): fail the step when either glob is empty.

### 7. Minor: other behaviours of the gate script

- `-n two` prints `[: two: integer expression expected` and runs serially with a PASS (line 186).
- After `kill -9` of the gate's own process, its step (`timed.py` and pytest) keeps running, the lock is stale,
  and a new run starts beside it (seen). `stop` handles the ordinary case correctly (seen: run, step and lock all
  gone; `wait` then exits 2).
- `--detach -l LOGDIR` on the directory of a live run deletes that run's `PID` file before being refused (line
  141), after which `wait` on it reports a vanished run.
- No step has a time limit; a hung test shows as `wait` exit 3 without end.
- The default invocation (all six steps in one run) was never executed as one; I ran two and three steps
  together without a problem.
- The step `slow` is unexercised, as the handover says. Its command collects 66 tests here.

### 8. Minor: statements in the notes

- Handover section 5.4 and the inventory note: "622 of the 677" strings are kind names. I count 677 method
  names in strings and 616 strings that are exactly a kind name; the remainder is kind names inside longer
  strings. The point stands.
- Handover section 5.5 and the script's docstring name two files with star imports. Forty-seven library modules
  also have `from t3toolbox.backend.common import *`. It has no effect (`common` defines none of the names), but
  the docstring should say so.
- `from t3toolbox.backend import apply as bapply` is counted as an `import` of the method name `apply`, while
  uses of `bapply` are not counted as the method; `import-module` would be consistent.

## Checked and confirmed

- **The three disagreements with the plan: the implementer is right in all three.** `t3toolbox/fitting.py` has
  seven factory calls in doctests (lines 244, 270, 285, 770, 779, 839, 931). `docs/glossary.md` has two links to
  `chunking.md` (398, 406); line 769 does not mention it as a link. `_wchunked_sum` is called from four
  functions (lines 1538, 1776, 2826, 2948).
- The inventory's code counts agree with an independent AST-only count: 1,503 calls, 395 keyword arguments, 55
  references to sampling names held without a call (all in tests), no import of a name under another name.
  The made-up cases (import alias, module alias, `from` import of a module, table of functions, `getattr`
  string, script in a string, f-string, multi-line chains, backslash continuation, nested docstrings, markdown
  and rst pages, brace patterns, star import) land where the docstring says. The tables in `P0_INVENTORY.md`
  reproduce exactly from the script.
- The five skips are the slow tier: the only other skips in `tests/` are conditional on jax being absent or on
  jax internals, and none fired. The one warning is the Adam-on-manifold warning.
- The one differing line of the review checks is rounding in the last digit (line 95, `levels_jax_check.py`).
- The slow test at `tests/backend/test_optimizers.py:1758` is outside the documented slow command, as reported.
- The restart: the machine's uptime (booted 21:28) agrees with the baseline's account, and the note says what a
  later slice does if it recurs. It did not recur in my 19-minute suite run or the other runs.
- `git status` shows only the eight new files under `dev/graded_messages/implementation/` (and this review);
  `git diff` is empty; no process is running; no lock file; no build output; `git worktree list` shows one tree.
