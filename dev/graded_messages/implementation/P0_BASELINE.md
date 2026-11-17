# P0: the baseline

Slice P0 of `PART1_PLAN.md`. Measured on 2026-11-16 on the branch `graded-messages` at `ad8c1c7`, on the unchanged
tree (the library and `tests/` as at `4d38df7`). This is the reference for every later slice: a slice must not
add a failure to what is recorded here.

**Result: everything passes on the unchanged tree.** No failure, no error, in any step. So from here on, any
failure in the gate is new.

## The machine and the versions

2 cores, 8,024 MB of memory (about 7.6 GB free when idle), no swap.

    python 3.13.18, numpy 2.5.3, jax 0.12.0.1, scipy 1.18.1, optax 0.2.9, pytest 9.1.1, pytest-xdist 3.8.0,
    pytest-subtests 0.15.0, sphinx 9.2.0, sphinx-autoapi 3.9.0, pydata-sphinx-theme 0.24.0, myst-parser 5.2.0

This is ahead of CI (its current leg is Python 3.11, numpy 2.4, jax 0.10; its floor leg Python 3.9, numpy 1.x,
jax 0.4.30). Nothing that CI uses is missing. `/usr/bin/time` is not installed, so the time and memory of each
step are measured by `timed.py` in this folder (it sums the resident memory of the step's process tree twice a
second, and reads the machine's `MemAvailable`).

## The gate, step by step

Each step was run alone, through `run_gate.sh`, one after another. "Peak" is the largest sampled sum of resident
memory over the step's process tree; "lowest free" is the smallest `MemAvailable` of the machine during the step.

| Step | Command | Result | Counts | Time | Peak | Lowest free |
| --- | --- | --- | --- | --- | --- | --- |
| suite, 1 worker | `pytest tests/ -q` | pass | 1,407 passed, 0 failed, 5 skipped, 0 errors, 49,058 subtests passed, 1 warning | 1,412 s (23 min 32 s) | 2,268 MB | 5,585 MB |
| suite, 2 workers | `pytest tests/ -q -n 2` | pass | the same: 1,407 passed, 5 skipped, 49,058 subtests passed, 1 warning | 1,149 s (19 min 9 s) | 3,150 MB | 4,768 MB |
| doctest-modules | `pytest --doctest-modules t3toolbox -q` | pass | 241 passed, 0 failed, 0 skipped | 28 s | 1,482 MB | 6,271 MB |
| doctest-quickstart | `python -m doctest docs/getting_started.rst` | pass | 1 file, no failed example | 2 s | 236 MB | 7,466 MB |
| doctest-pages | `python -m doctest` on the 48 pages (`docs/*.md`, `docs/contributor/*.md`, without `doctest_style.md`) | pass | 48 files, no failed example | 13 s | 458 MB | 7,305 MB |
| examples | each `examples/*.py` | pass | 14 examples, 0 failed | 210 s | 370 MB (the largest single example) | 7,390 MB |
| sphinx | `python -m sphinx -W --keep-going -b html docs <out>` | pass | build succeeded, 0 warnings | 186 s | 690 MB | 6,933 MB |

The whole gate, one step after another: about 31 minutes with one worker, about 27 minutes with two.

The counts of the suite are those `CLAUDE.md` records for 2026.5.0 (1,407 tests, 49,058 subtests). The two runs
of the suite gave the same counts; no test was seen to be flaky.

**The five skips** are the slow tier, all skipped for the same reason (`T3TOOLBOX_SLOW_TESTS` is not set):

    tests/backend/test_optimizers.py:1758         the remainder bound
    tests/backend/test_uniform_curvature.py:1467  the full-length run (> 3 s)
    tests/backend/test_uniform_curvature.py:1499  the full-length run (> 3 s)
    tests/test_oracle_sweep.py:105                seeds x sample layouts x order 3
    tests/test_oracle_sweep.py:138                full optimizer matrix

**The one warning** is intended (the library warns that Adam on the manifold geometry depends on the
representation): `tests/test_oracle_sweep.py::TestO2OptimizerOracleSweep::test_pairwise_subset`, a
`RuntimeWarning` raised at `tests/oracle_sweeps/sweep_optimizers.py:102`.

**The examples**, each alone:

    PASS     5.3s    207 MB  examples/fit_hilbert_from_apply_derivatives.py
    PASS     4.8s    207 MB  examples/fit_hilbert_from_apply_derivatives_flat.py
    PASS     4.1s    163 MB  examples/fit_hilbert_from_apply_derivatives_topt.py
    PASS    15.1s    204 MB  examples/fit_hilbert_from_entries_lbfgs.py
    PASS    20.2s    164 MB  examples/fit_hilbert_from_probes_adam.py
    PASS    28.9s    370 MB  examples/fit_hilbert_from_probes_optax.py
    PASS    30.5s    164 MB  examples/fit_hilbert_regularized.py
    PASS     7.4s    244 MB  examples/fit_hilbert_tensor_newton_cg.py
    PASS    15.2s    206 MB  examples/fit_hilbert_uniform_newton_cg.py
    PASS     9.3s    334 MB  examples/fit_hilbert_uniform_probe_derivatives_newton_cg.py
    PASS     4.5s    162 MB  examples/fit_per_mode_weight_probes.py
    PASS     0.7s    163 MB  examples/fit_probe_display.py
    PASS    48.3s    213 MB  examples/fit_shared_factors_jetted_probes.py
    PASS    14.6s    214 MB  examples/fit_varied_rank_tensor_newton_cg.py

## Failures on the unchanged tree

None. The differences between this environment and CI's (Python 3.13 against 3.11, numpy 2.5 against 2.4, jax
0.12 against 0.10) produced no failure in any step.

## How many workers

Both one and two workers are safe for the suite on this machine, when the suite runs alone: with two, the
process tree peaked at 3.2 GB and the machine never had less than 4.8 GB free; with one, 2.3 GB and 5.6 GB.

**Recommended: `-n 2` for the full suite, the default (`-n 1`) for `quick`.** Two workers save only 4 minutes of
23 (19 %), because the numerical libraries already use both cores with one worker. So `-n 1` costs little, and it
is the choice whenever anything else is using memory. Do not go above 2 (there are 2 cores). The memory rule
stands whatever the count: one gate step at a time, and nothing else heavy while it runs. The steps other than the
suite are single-process, and none of them came near the limit (the largest is the module doctests at 1.5 GB).

## One thing that went wrong: the machine restarted once

The first run of the suite (one worker) was lost: the machine restarted under it. The run started at 21:25:36;
its log was last written at 21:28:19; at 21:29:09 the machine reported an uptime of one minute, with every process
gone and the files on disk intact. The run was started again and passed.

It does not look like memory. The second run, at the same point (2 min 43 s in), held about 0.65 GB with 7.1 GB
free, and the whole run never held more than 2.3 GB. Nothing else was running except a five-second static script
(the inventory). The kernel log does not survive a restart, so the cause is not known; it looks like an event of
the workspace's infrastructure, not of the tests. It did not happen again in the 54 minutes of runs that followed.

What it means for the later slices:
- A detached gate run can vanish. `run_gate.sh wait LOGDIR` then says that the run is not running and did not
  finish, and exits 2. Start the run again; it is not a test failure.
- `timed.py` now writes the memory history of each step to `LOGDIR/<step>.stats.trace` every five seconds and
  syncs it to disk, so that if it happens again the trace shows whether memory was the cause.
- Work that exists only in memory is lost by a restart; files on disk are kept.
- `/tmp` is kept too, so the lock file and a run's `LOGDIR/PID` outlive a restart, while process ids start again
  from low numbers. Since the review of P0 the gate script names a run by its pid, the start time of that
  process and the machine's boot id, so a lock or a log directory from before a restart is recognised as dead: it
  does not block a new run, `wait` answers exit 2 for it, and `stop` signals nothing.

## Known limits of the gate script

- **A gate process killed with `kill -9` leaves its step running.** Every run is the leader of a process group
  that holds its steps, so this is detected: the lock still refuses a new run, `wait LOGDIR` says the gate process
  is gone and exits 2, and `run_gate.sh stop LOGDIR` ends the step and frees the lock. Nothing ends the step by
  itself; run `stop`.
- **No step has a time limit by default.** A hung test shows as `wait` answering exit 3 without end. `-t SECONDS`
  sets a limit for each step (each example, in `examples`): a step that reaches it is ended with its process tree
  and counts as FAIL. For the suite here, `-t 2400` is about twice the measured time with two workers.
- **Under the time limit, a descendant that ignores SIGTERM can outlive its step** (the reviewer's finding 10,
  round 2). The kill that follows the SIGTERM is sent only while the step's main process is alive, so a child that
  ignores SIGTERM stays, the lock is released, and a new run could start beside it. The suite's own workers do end
  on SIGTERM. Not fixed; after a step has been stopped at the time limit, look for leftovers (`pgrep -f pytest`)
  before starting another run.
- **`doctest-pages` runs whatever the two globs match**, as CI's command does: if a folder of pages disappeared,
  the step would pass on fewer pages. The summary shows the number of pages; it is **48** at P0, so a smaller
  number in a later slice (other than P13 removing `docs/chunking.md`, which makes it 47) is a finding.
- The whole gate has not been run as one invocation by the implementer; the reviewer ran two and three steps
  together, and the suite alone, without a problem.

## The check suites of the folder (part A)

Both were run against the branch head and compared with the recorded outputs.

| Script | Compared with | Result | Time | Peak |
| --- | --- | --- | --- | --- |
| `dev/graded_messages/scripts/run_all.sh` (15 scripts) | `scripts/expected_output_3dc15d9.txt` | **no drift**: the 372 lines are identical | 27 s | 266 MB |
| `dev/graded_messages/scripts/review/run_review_checks.sh` (23 scripts, the jax ones included: jax is installed) | `scripts/review/expected_review_output_3dc15d9.txt` | **one line differs, in the last digit**: line 95, in `levels_jax_check.py`, part D ("entries, 1000 samples padded to 1024"), `one device, jit:  difference 8e-16` here against `7e-16` recorded. The other 355 lines are identical | 114 s | 586 MB |

The one difference is rounding (the scripts say "numbers agree to rounding; last digits may differ"), in a jax
computation, on a jax newer than the one the output was recorded with. No script failed and none was skipped.

## The slow tier (not run in P0)

`T3TOOLBOX_SLOW_TESTS=1` turns on the five tests skipped above. It is a **required release-gate step, run
serially** (`CLAUDE.md`), and the plan makes it part of the gate of P16, P17 and P20:

    T3TOOLBOX_SLOW_TESTS=1 PYTHONPATH=$PWD python3 -m pytest tests/test_oracle_sweep.py tests/backend/test_uniform_curvature.py -q -p no:cacheprovider

`run_gate.sh` has it as the step `slow`, which runs only when named and always with one process
(`run_gate.sh --detach slow`, then `wait`). The step was written but **not exercised** in P0, as instructed.

Expected time: the last records of the maintainer's runs of this command, at 2026.5.0, are 66 passed and 13,173
subtests in 14 min 45 s and 17 min 32 s (`dev/archive/store_rename_2026-11-15/store_rename_slices/R6_review.md`
and `R6_handoff.md`). This machine has 2 cores and ran the ordinary suite serially in 23.5 minutes, so expect
**15 to 30 minutes** here; it has not been measured. It is longer than the ten minutes a foreground command gets,
so it must be run detached. Its memory has not been measured either; it runs in one process, like the suite with
one worker (2.3 GB).

One of the five slow tests is outside that command: `tests/backend/test_optimizers.py:1758` ("the remainder
bound") is switched on by the same variable, but the documented command names only the other two files. The
maintainer's records run it separately (`T3TOOLBOX_SLOW_TESTS=1 ... test_the_remainder_is_small`). A slice whose
gate includes the slow tier should say whether it runs that one too.

## How to repeat this

    G=dev/graded_messages/implementation/run_gate.sh
    $G --detach -n 2 -l /tmp/gate_Pn        # the whole gate; returns at once
    $G wait /tmp/gate_Pn                    # repeat until it prints the table (each call waits up to 570 s)
    $G -l /tmp/quick_Pn quick tests/test_jet_contract.py tests/test_dispatch.py     # only these files

The logs of the runs behind this note were under `/tmp/p0/` and are not kept.
