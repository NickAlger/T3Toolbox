# The gate on the maintainer's laptop, before P4a (baseline)

Taken by Claude on 2026-11-19 (10:55 to 11:44, by the times of the run's files), on the untouched head of
`graded-messages`, `10e680d` (the library and `tests/` as P4 left them at `5991725`). The reference for every slice
from P4a on: a slice must not add a failure. `P0_BASELINE.md` is the cloud workspace's and is kept as history.

## The machine and the versions

- 8 cores, 13 GB of memory, about 19 GB of free disk.
- `python 3.11.15, numpy 2.4.6, jax 0.10.2, scipy 1.17.1, optax 0.2.8, pytest 9.1.1, pytest-xdist 3.8.0,
  pytest-subtests absent, sphinx 9.0.4, sphinx-autoapi 3.8.0, pydata-sphinx-theme 0.20.0, myst-parser 5.1.0`
  (`versions.py`; the conda environment `t3toolbox`, the twin of CI's current leg). pytest 9 reports `subTest`
  failures itself, so `pytest-subtests` is not needed here.
- The command: `PATH=/home/nick/miniconda3/envs/t3toolbox/bin:$PATH dev/graded_messages/implementation/run_gate.sh -n 3
  -l <logdir>`, nothing else running.

## The result: every step passes

| Step | Result | Counts | Seconds | Peak RSS (MB) | Lowest MemAvailable (MB) |
| --- | --- | --- | --- | --- | --- |
| suite (`-n 3`) | PASS | 1,632 passed, 5 skipped, 1 xfailed, 1 warning, 66,243 subtests passed | 1,401 | 9,826 | 2,502 |
| doctest-modules | PASS | 310 passed | 53 | 2,062 | 10,033 |
| doctest-quickstart | PASS | 1 file, 0 failed examples | 3 | 269 | 11,847 |
| doctest-pages | PASS | 48 files, 0 failed examples | 15 | 374 | 11,753 |
| examples | PASS | 14 examples, 0 failed | 421 | 637 | 11,443 |
| sphinx | PASS | 0 warning or error lines | 1,052 | 635 | 9,956 |

About 49 minutes in all. The suite's counts equal the cloud workspace's at this head (`ORCHESTRATION.md` section 4:
1,632 passed, 5 skipped, 1 xfailed), so no difference of the environment shows in the suite.

- **The 5 skipped** are the slow tier's, as expected (`T3TOOLBOX_SLOW_TESTS=1`): `tests/backend/test_optimizers.py:1758`,
  `tests/backend/test_uniform_curvature.py:1467` and `:1499`, `tests/test_oracle_sweep.py:105` and `:138`.
- **The 1 xfailed** is the contraction-order marker carried to Part 2 (`tests/test_jet_contract.py`,
  `TestOrderInsideAStep`; row 5.5).
- **The 1 warning** is the library's own, deliberate `RuntimeWarning` for adam on a manifold geometry, raised from
  `tests/oracle_sweeps/sweep_optimizers.py:102`.

## Memory

The suite at `-n 3` peaked at 9.8 GB of resident memory and left 2.5 GB available at its lowest. That is the
margin: **`-n 3` is the most this machine takes, and nothing else may run beside the suite** (not sphinx, not a
second test run, not an agent's quick test). `-n 2` is the choice if anything else must run.

## The floor

Not run locally. CI's floor leg (Python 3.9, numpy 1.26, jax 0.4.30) runs on every push through the draft pull
request #26; its first run is on this same head (`LOG.md`, the entry of 2026-11-19).
