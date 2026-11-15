# CI floor-leg fix (2026-11-15): four failing tests, three causes, a test-only fix

HEAD `e97a839b` (the 2026.5.0 release commit `52843bfc` is pushed, not tagged). Uncommitted: `tests/backend/test_optimizers.py`,
`tests/test_dispatch.py`, `tests/test_optimizers_frontend.py`, this note. **No library change.** One library property
surfaced along the way (the row-0 trust-region tie, §4); it is not a counting bug and does not block the tag, but it is a
design question for Nick.

## 1. What CI actually showed

The brief said the floor leg (py3.9 / numpy 1.26.4 / jax 0.4.30) has been red since `c1a42c24` while the current leg
(py3.11 / numpy 2.4.6 / jax 0.10.2) passes. The job history of the last eleven runs says something different:

| commit | floor leg | current leg |
|---|---|---|
| `c1a42c24` (P3; the switch tests not yet written) | (b) | pass |
| `3dc15d96` (W1 lands the switch tests) | (b) | (a) + (c) `7 != 6` |
| `c97c754f`, `6b01889c`, `a1fb9923`, `27a9216c` | (a) + (b) + (c) `6 != 7` | (a) + (c) `7 != 6` |
| `7807b238`, `923178ff` | (b) | pass |
| `51782061` | (a) + (b) + (c) `6 != 7` | (a) + (c) `7 != 6` |
| `52843bfc`, `e97a839b` | (a) + (b) + (c) `6 != 7` | pass |

(a) = the two `TestSwitchNewton` reset tests, (b) = the frontend `timing` subtest label, (c) = `test_use_jit_ragged`.

- **(b) is the only deterministic failure**, and it is floor-only.
- **(a) and (c) are flaky on BOTH legs**: they fail together or pass together, per job, on either leg (6 of 11 current-leg
  jobs failed them). Two jobs of the same run can disagree (`3dc15d96`). That is a per-runner effect, not a version effect.
- **(c) is not a compile count.** `6 != 7` is `eager['switched_at']` against the jit run's `switched_at`; the floor jobs
  failed `6 != 7`, the current jobs `7 != 6` (eager and jit swapped sides).

## 2. Reproduction

CI-like floor env in the session scratchpad: `python3.9 -m venv` (from the `tttt` interpreter, 3.9.19) +
`pip install --only-binary=:all:` of the exact versions CI resolved (read from the job's install log):

    numpy==1.26.4 jax==0.4.30 jaxlib==0.4.30 scipy==1.13.1 ml-dtypes==0.5.4 opt-einsum==3.4.0
    pytest==8.4.2 pytest-subtests==0.15.0 pytest-xdist==3.8.0 execnet==2.1.2

(565 MB, about a minute to build.) The four tests at `-n 3`, `PYTHONPATH=$PWD` as the workflow: **(b) reproduced, (a) and
(c) did not** (3 passed). Neither did they reproduce under three OpenBLAS kernels (`OPENBLAS_CORETYPE=Haswell / SandyBridge
/ Prescott`; the trajectory agrees to six digits). They were then reproduced by reasoning from the code (§3).

## 3. Root causes and fixes

### (b) `test_newton_cg_forwards_timing_only_when_set`: an `np.bool_` subtest label

pytest-subtests ships each subtest report, labels included, from the xdist worker over execnet. execnet's serializer
dispatches **by type name** (`"save_" + type(obj).__name__`). Under numpy 2 the bool scalar's class is named `bool`, so it
falls into `save_bool` by accident; under numpy 1.x it is `bool_`, which has no saver: `DumpError`. (`np.int64` fails on
both, which is how the repo met this class before: `test_dispatch.py`'s `test_frame_widths_dispatch`.)

**Fix:** the label is builtins only, `subTest(timing=None | bool(value), numpy_bool=isinstance(value, np.bool_))`, the
`test_frame_widths_dispatch` pattern. The call still passes the `np.bool_` itself to the frontend, so the test's meaning
("a numpy bool is a bool") is unchanged. The sibling `test_newton_cg_timing_must_be_a_bool` labels are builtins already.

### (a) and (c): the row-0 tie of the default trust-region radius

Under Gauss-Newton (and in the switch's Gauss-Newton phase) the default `tr_radius0` is the GN Cauchy length
`‖g‖³ / ⟨g, H_GN g⟩` (`_initial_radius`). Steihaug CG starts at 0 with `d = −g`, so its **first iterate is the Cauchy point,
whose length is exactly that radius**. `_cg_body`'s boundary test `‖p + αd‖ > Δ` therefore compares two roundings of the
same number, and its verdict decides:

- whether row 0 is flagged `cg_boundary` (when one CG iteration meets the forcing tolerance the step is the same to
  rounding either way; when it does not, the tie also decides whether CG stops at the Cauchy point or continues), and so
- whether row 1's radius doubles (`ρ > 0.75` on a boundary step), and from there the whole trajectory.

Measured on this laptop: the default radius and the first iterate's length are **bitwise equal** on both problems
(`127.18324426819628` on `_pd_problem(0)` from `_rank2_start()`; `90.11942936873201` on T9's ragged case). Which way the
`>` falls depends on the reduction order of the inner products -- the BLAS kernel (the runner's CPU) and eager vs XLA.

Forcing each side by patching `_initial_radius` to return `Δ0·(1 ∓ 4ε)` (a `unittest.mock.patch.object` around a
`unittest` run of the named tests; for the split, by call order: the eager run is `check_use_jit`'s fifth
`_initial_radius` call) reproduces CI exactly on the unchanged tests (a `git archive` of HEAD):

| forced | result on HEAD's tests |
|---|---|
| row 0 on the boundary | `test_reset_rows`: `AssertionError: False is not true` at `reset_runs` line 1990 (CI's line) |
| row 0 interior | passes (this laptop's side, the side the tests were written on) |
| jit on the boundary, eager interior | `test_use_jit_ragged`: `AssertionError: 6 != 7` (the floor jobs' message) |
| jit interior, eager on the boundary | `AssertionError: 7 != 6` (the current jobs' message) |

A 1e-15 relative perturbation of `x0` flips the tie in 2 of 5 draws, with the same consequence: in the 'rejected trial'
run row 0 becomes a boundary step, the radius doubles, and row 2's `ρ` falls from 0.587 to 0.055 (rejected), so the
"three accepted steps, then rejections" scenario the test assumed is gone.

**Is it a library bug that blocks the release? No.** The counting is host Python on the row's own values and is right on
both trajectories; both trajectories are valid trust-region runs; bitwise agreement across machines was never promised.
The tests were structural tests written numerically: they relied on a trajectory producing specific events, and the
first decision on that trajectory is a coin toss. See §4 for the library side.

**Fix (tests only): drive every event, and never start a trajectory-dependent run on the tie.**

- `_untied_radius(problem, x0)` (new helper, `tests/backend/test_optimizers.py`): half the default radius. CG's first
  iterate is then twice the radius, so row 0 stops on the boundary by a factor of 2 by construction.
- `_objective_stub_at(n, change)` (new helper, the `_cg_stub_at` idiom): the `n`-th `Problem.objective` call returns
  `change(f)`. Under the trust region `newton_cg` calls it once per row, on the trial, so call 4 is row 3's trial.
- `reset_runs` (feeds T4's two tests):
  - 'rejected trial': was `_OffsetProblem` (5 % of `f0` on every trial; rejections arrive once the true decrease falls
    under the offset -- a trajectory property). Now the base problem with row 3's trial objective raised by `f(x0)`:
    `f(trial) + f(x0) − f(x) > 0` from any accepted point, so that trial is an ascent, `ρ < 0`, rejected, whatever the
    numbers. Rows 0-2 are natural steps with `ρ` far above 0.1.
  - 'fallback, trust region' and 'boundary': start at `_untied_radius`. 'boundary' needs row 1 to be an accepted
    boundary stop after a countable row 0: at half the default rows 0 and 1 both stop on the boundary.
  - the line-search runs (backtrack, fallback, forced accept) run without a radius and keep their CG stubs.
  The precondition asserts stay; they now hold by construction or by margin, not by luck.
- `check_use_jit` (T9, `tests/test_dispatch.py`): every run starts at `_untied_radius` (computed once on numpy, passed to
  the jit runs and the eager run alike; `tr_radius0` is a traced CG argument, so the program-count checks are unchanged).
  The eager-equals-jit check keeps its meaning: same switch row, objectives to 1e-9.

The plan's budget (the Gauss-Newton phase reuses the GN CG program; a firing run adds at most one, the exact one; a
repeat adds none) was never the failing assertion and is untouched.

**Margins** (a recorder wrapping `_cg_body` and `_switch_sample`: every discrete decision's relative distance from its threshold, over every CG
iteration and row): CG boundary `‖p+αd‖/Δ` vs 1, CG convergence `rs/tol²` vs 1, `ρ` vs 0.1 / 0.25 (and 0.75 on boundary
rows), the switch's `e_N/(τ·e_GN)` vs 1 and `max(e)/floor` vs 1. Smallest values, at half the default radius:

| run | CG boundary | CG converge | ρ | hit | gate |
|---|---|---|---|---|---|
| reset: rejected trial | 0.58 | 0.74 | 0.35 | 3 | 1e9 |
| reset: fallback, trust region | 0.58 | 0.74 | 0.35 | 37 | 8e9 |
| reset: boundary | 0.58 | 0.77 | 0.35 | 37 | 8e9 |
| T9 ragged / uniform | 0.35 | 0.0074 | 0.51 | 0.58 | 3e9 |
| T9 shared uniform | 0.15 | 0.0096 | 0.20 | 0.37 | 5e8 |

against `1e-15`-scale rounding (the default radius' CG-boundary margin is `~1e-15`). T9 switches with `switched_at` 7
(ragged, uniform) and 10 (shared).

**Mutations** (the W1 review's T4 list, each a one-line change of `newton_cg` in a copy of the working
tree, the two reset tests rerun; the unmutated tree as control): **11 / 11 caught, each by its own case**, control green.
Rejected trial / backtrack / fallback / forced accept counted (3 subtests each, the named case); boundary CG resets (6);
capped CG resets (the two 'capped CG' subtests); `count += 2` (12); no reset across a non-countable row (8); 'all' tracking
only the countable rows (10); 'countable' tracking every row (10); untracked rows reporting `e_N` (10).

## 4. The library side: the tie is a property of the default radius

Every Gauss-Newton trust-region run with the default `tr_radius0`, and every `hessian='switch'` run, starts on this tie
(exact mode does not: its CG uses `H`, not `H_GN`, so its first iterate's length differs from the GN Cauchy length).
Consequence: from row 1 on, such runs differ between machines and between eager and jit by more than rounding (the
radius is doubled or not). Not wrong -- but it is a reproducibility wart, and any test that reads a default-radius GN
trajectory is exposed to it. Options, for after the tag:

1. Leave it; document that GN-TR runs are machine-dependent from row 1, and keep trajectory-reading tests off the
   default radius (what this fix does).
2. Change the default to a multiple of the Cauchy length (2× is the natural one: CG continues past the Cauchy point and
   the boundary is reached only by a genuinely longer step). Principled, but it changes every default GN-TR trajectory
   and step count, so it needs the Darcy ladders before it ships.
3. A tolerance on the boundary test. Rejected in advance: it moves the tie rather than removing it, and it is the
   ε-content kind of fix the project avoids.

Recommendation: 1 now (tag on this fix), 2 as a post-tag decision with a benchmark.

## 5. Why the local floor env did not catch it

- **(b)** needs three things at once: numpy < 2, pytest-xdist and pytest-subtests. `tttt` is numpy 1.22.4 but has
  **neither plugin**, and the release checks ran it single-process, so no subtest report was ever serialized.
- **(a)/(c)** are not version effects. This laptop's CPU breaks the tie on the side the tests were written on, so no env
  on this machine -- `tttt`, the current env, or the CI-like env -- reproduces them. They show on CI only on runners whose
  CPU breaks it the other way (both legs, about half the jobs).
- numpy 1.22 vs 1.26 played no part in either.

## 6. Recommendation: a CI-like floor env locally?

Keep the **recipe** (the pins in §2 plus `pytest-xdist`, `pytest-subtests`), not a permanent env: it builds in about a
minute and costs 565 MB, and it catches exactly one class of failure -- what numpy 1.x + xdist + pytest-subtests together
reject (serialization of labels, plugin interplay). Cheaper still: add `pytest-xdist pytest-subtests` to `tttt` and run
its floor checks with `-n 3`, which would have caught (b). What no local env catches is (a)/(c): the defence there is the
test rule this fix applies -- a test that needs an event drives it (stub, script, or a parameter set away from any tie),
and a run whose rows a test reads never starts on a rounding tie.

**Both sides of the tie, after the fix:** with `_initial_radius` forced to `Δ0·(1 − 4ε)` and to `Δ0·(1 + 4ε)`, all 34
tests of `test_optimizers.TestSwitchNewton`, `test_optimizers.TestTrustRegionNewton`,
`test_optimizers_frontend.TestFrontendSwitchNewton` and `test_dispatch.TestSwitchNewton` pass on each side (the reset
tests, T9 and the T4 scripts included). So no other test in these classes reads a trajectory past the tie in a way that
breaks; the default-radius runs left in them (`test_the_junction`, `test_a_miss_resets`, `test_reset_rows`' closing
switching run, ...) hold on both trajectories or compare runs on the same machine.

## 7. Verification

One job at a time.

| what | env | result |
|---|---|---|
| the four tests, `-n 3` | CI-like floor (py3.9.19 / numpy 1.26.4 / jax 0.4.30, xdist + subtests) | 4 passed, 25 subtests (51 s) |
| `tests/backend/test_optimizers.py`, `tests/test_dispatch.py`, `tests/test_optimizers_frontend.py`, `-n 3` | CI-like floor | 203 passed, 2 skipped, 533 subtests (13 min) |
| full suite, `-n 3` | current (py3.11 / numpy 2.4.6 / jax 0.10.2) | 1407 passed, 5 skipped, 49,058 subtests (23 min) |
| the three touched modules, single process | `tttt` (py3.9 / numpy 1.22.4 / jax 0.4.30) | 203 passed, 2 skipped (26 min) |
| the four failures, unfixed, tie forced each way | current | reproduced CI's messages (§3) |
| `git diff --check` | | clean |

CI runs `-n auto` (4 workers on GitHub's 4-vCPU runners); `-n 3` here is the project's local cap. The CI-like env was
deleted after the runs; §2 has the recipe.
