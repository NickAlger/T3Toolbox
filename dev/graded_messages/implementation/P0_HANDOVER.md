# P0: handover

Slice P0, the baseline (`PART1_PLAN.md` section 5.3), with the two parts added for this workspace (the gate script
and the measured baseline). Implementer's handover, 2026-11-16. Branch `graded-messages` at `ad8c1c7`.

## 1. What changed

Nothing in the library, the tests, the docs or the examples. Eight new files, all in
`dev/graded_messages/implementation/` (this note and the seven below):

| File | What it is |
| --- | --- |
| `run_gate.sh` | The gate every later slice uses: the six steps of section 5.1 and the two CI workflows, strictly one after another, each with a log, its time and its peak memory, and a summary table. Modes: the whole gate or named steps; `quick` for the tests of given files; `--detach` with `wait` and `stop` for runs longer than ten minutes; `-n N` for the test workers; `-t SECONDS` for an optional time limit per step; a lock so that two runs cannot go at once, with a run named by pid, process start time and boot id so that nothing left from before a restart is mistaken for a run; an extra step `slow` for the slow tier, run only when named |
| `timed.py` | Runs one command and records its wall time, the peak resident memory of its process tree and the lowest free memory of the machine. It stands in for `/usr/bin/time -v`, which is not installed here |
| `versions.py` | Prints the versions a run used (one line, into the run's logs) |
| `inventory.py` | The inventory as a script (static, standard library only). Its docstring says what it searches, the kinds it tells apart, and what it can and cannot see |
| `inventory_names.json` | The names the inventory is about, frozen at P0, so a later re-run means the same thing after modules are gone |
| `P0_INVENTORY.md` | The inventory: the names, how to read the counts, the check against the plan's counts, and the script's tables (by group and area, by name, by file) |
| `P0_BASELINE.md` | The baseline: each gate step's result, counts, time and memory on the unchanged tree; the worker count; the restart; the drift of the check suites; the slow tier |

## 2. Tests

None added or changed. P0 has no library change to test.

The tools were tested as follows. `run_gate.sh`: every step run through it (below); `quick` with one and with two
workers; an unknown step is refused; a second run is refused while one is going; `stop` ends a detached run with
its workers and frees the lock; `wait` reports a run that is still going (exit 3) and one that vanished (exit 2).
After the first review (`P0_RESPONSE_r1.md` has the list): `stop` on another directory leaves a live run's lock;
a `PID` file or a lock holding a bare pid, a live pid with another start time, a live pid with another boot id, or
pid 1 counts as dead for `wait`, `stop` (which signals nothing) and the lock; `--detach` onto the directory of a
live run is refused; `-n two` and `-n 0` are errors; a gate process killed with `-9` is detected and cleaned up by
`stop`; a step over its `-t` limit fails; a foreground run that receives SIGTERM ends with its workers.
The step `slow` is the one part not exercised (P0 was told not to run the slow tier). `inventory.py`: a made-up
file with an import alias, a module alias, a table of functions, a `getattr` string, an f-string, a brace pattern,
a star pattern, a doctest and a fenced block, each landing in the expected kind; and its counts against the
plan's (`P0_INVENTORY.md`, "Checked against the plan"), where 26 of 29 statements agree to the number and three differ slightly (section 5).

## 3. The gate

P0's gate is "none" in the plan. The gate was nevertheless run on the unchanged tree, as the baseline. From the
logs (`P0_BASELINE.md` has the details):

| Step | Result | Counts | Time | Peak memory |
| --- | --- | --- | --- | --- |
| suite, `-n 1` | pass | 1,407 passed, 0 failed, 5 skipped, 0 errors; 49,058 subtests passed; 1 warning | 1,412 s | 2,268 MB |
| suite, `-n 2` | pass | the same counts | 1,149 s | 3,150 MB |
| doctest-modules | pass | 241 passed | 28 s | 1,482 MB |
| doctest-quickstart | pass | 1 file, no failed example | 2 s | 236 MB |
| doctest-pages | pass | 48 files, no failed example | 13 s | 458 MB |
| examples | pass | 14 of 14 exit 0 | 210 s | 370 MB |
| sphinx `-W` | pass | build succeeded, no warning | 186 s | 690 MB |

No failure on the unchanged tree, so there is no baseline failure to carry: every failure a later slice sees is
new. The five skips are the slow tier; the one warning is the library's intended Adam-on-manifold warning.

The suite was run as one step alone in each of two runs, and the other five steps together in a third run; the
whole gate was not run as a single invocation (it would repeat the same commands for 27 more minutes).

Part A, the folder's check suites: `run_all.sh` has no drift (372 lines identical);
`run_review_checks.sh` differs in one line by one last digit (`levels_jax_check.py`, part D: `8e-16` against the
recorded `7e-16`), which is rounding.

## 4. Departures from the plan, and choices

Departures: none from the plan's P0. The task added the gate script and the measured baseline to it.

Choices the task left open:

- **Long runs are detached, not cut in pieces.** The suite is one `pytest tests/` invocation, as in CI, started
  with `--detach` and awaited with `wait` (each `wait` call returns within ten minutes). Cutting the suite by file
  would have changed what is run (45 invocations, no state carried between files) for no gain in memory: the whole
  suite in one process peaks at 2.3 GB.
- **`-n 1` means plain pytest**, without xdist; `-n 2` and above pass `-n N` to xdist.
- **The pytest steps add `-rfEs -p no:cacheprovider`** to CI's commands: failures, errors and skips are listed by
  name at the end of the log, and no `.pytest_cache` is written into the repository. Neither changes what is run.
- **Sphinx builds under the log directory**, not `docs/_build`, with CI's flags; the `docs/autoapi` folder that
  sphinx-autoapi makes during a build is removed afterwards when the run created it.
- **The page doctests are one `python -m doctest` command over all 48 pages**, as in CI (one process), not one
  command per page.
- **The inventory's unit** is the call site in the syntax tree, the plan's own unit, and every occurrence of a
  name is put in exactly one context and kind, so the columns add up to the total.
- **The frozen name list.** The script reads the names from `inventory_names.json`, not from the modules, so that
  it still runs after P20.
- **`slow` as a step of the gate script**, off by default, because P16, P17 and P20 need it and it is longer than
  ten minutes.

## 5. Found outside the slice

1. **The machine restarted once during a run** (the first suite run was lost; the files on disk were kept).
   Memory was not the cause as far as the evidence goes: at the same point the second run held 0.65 GB of 8 GB.
   It did not recur. `P0_BASELINE.md` has the times and what it means for later slices: a detached run can
   vanish, `wait` says so with exit 2, and the run is started again.
2. **Three counts in the plan differ from the tree** (`P0_INVENTORY.md`, "Checked against the plan"). None changes
   what a slice does:
   - P16 says the doctests of `t3toolbox/fitting.py` hold six factory calls; there are seven (lines 244, 270, 285,
     770, 779, 839, 931; line 779 is the one shown raising).
   - P13 says `docs/glossary.md` has three links to `docs/chunking.md`; it has two (lines 398, 406). The third
     mention (line 769) is the file name in backticks, which `sphinx -W` does not check. In the two pages the links
     are markdown links, not `:doc:` roles.
   - Section 1 speaks of "the library's two chunked uniform assemblies (`_wchunked_sum`)"; `_wchunked_sum` is
     called from four functions (the Tucker and TT assemblies of the transpose and of the tangent form).
2b. **The root `README.md` is in no slice** (found by the reviewer). It holds the old call forms, among them
   `t3t.newton_cg(t3t.MANIFOLD, 'apply', ww, b, x0, max_newton=30)` at line 87, and no CI step runs it. Claude
   decided: the inventory searches it, and the plan will name it in P17. `CHANGELOG.md` and
   `docs/release_notes.md` stay out of the inventory on purpose (P20's final search excludes them); their counts
   are in a separate table of `P0_INVENTORY.md`.
3. **A slow-tier test outside the documented slow command.** `tests/backend/test_optimizers.py:1758` is switched
   on by `T3TOOLBOX_SLOW_TESTS` but the command in `CLAUDE.md` names only `tests/test_oracle_sweep.py` and
   `tests/backend/test_uniform_curvature.py`. The slices whose gate includes the slow tier (P16, P17, P20) should
   say whether that test runs too. `run_gate.sh slow` runs the documented command only.
4. **The old kind strings are spelled like the methods** (`'probe_derivatives'`, `'apply'`, ...). A search for
   the removed methods by name also finds the kind strings: of the 677 occurrences of method names inside
   strings, 616 are a string that is exactly a kind name, and most of the rest are kind names inside longer strings. They belong to P16 and P17, not P13. The inventory keeps them apart (context
   `string`, and kind `str` inside doctests).
5. **Two test files import with `*`** (`tests/oracle_sweeps/o1_ragged.py` and `o1_uniform.py`, from
   `o1_common`), so a static search cannot tell where their names come from. The inventory counts the calls and
   marks the binding as not determined. (Forty-seven library modules also have
   `from t3toolbox.backend.common import *`; it has no effect here, since `common` defines none of the names.)
6. `/usr/bin/time` is not installed; nothing else the gate needs is missing.

## State of the tree

`git status` shows only new files in `dev/graded_messages/implementation/`: the eight of this slice, the review
and the response. Nothing was committed. No
build output is left in the tree and no process is left running. The logs of the runs were under `/tmp/p0/`.
