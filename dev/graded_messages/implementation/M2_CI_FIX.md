# M2 CI fix: `test_zero_padded_shared_restart` (and three siblings)

2026-11-23. CI run 45125027667 (PR #26, at `68c10621` = M2) failed on the floor leg only (py3.9, numpy 1.26.4,
jax 0.4.30): `tests/backend/test_uniform_curvature.py::TestUniformSharedExactNewton::test_zero_padded_shared_restart`,
`AssertionError: 17 not less than or equal to 16` at `self.assertLessEqual(ub['newton_steps'], 16)`. The same leg
passed at `85c6e54d`, before M2.

**Not a library bug, and not M2.** The outer count of an exact Newton run from a zero-padded (rank-deficient) shared
restart is a draw over the rounding: the frame's completion of the zero block is whatever LAPACK returns
(`docs/uniform_equivalence_contract.md` §"The full-rank qualifier"), and the globalization phase that follows it
lasts 2 to 11 steps depending on that choice. At the test's template seed (7) the uniform count ranged over 7 to 17,
so a bound of 16 sat in the tail. Test-only fix: a template seed whose count is robust. The group-ceiling sibling had
the same defect, worse (its bound 9 failed on about one rounding draw in five), and is fixed the same way.

The same CI run's second floor-leg failure, `tests/backend/test_curvature.py::TestExactNewtonRate::
test_gauss_newton_control_is_linear` (seed 1, a ratio 0.25 outside `(0.3, 0.9)`), is also rounding and also not M2:
its window counted a step whose predicted decrease was below the objective's resolution (§ "The Gauss-Newton rate
control"). A latent flake in the ragged T22 test is fixed too (§ "The ragged T22 stop reason").

## What the bound is for

T22's uniform twin: from a zero-padded shared restart (the new `s_g` levels exactly zero, a singular tied lift) the
exact Newton run must converge on the gradient test, with at most two rejections, to the ragged run's minimizer, and
stay tied. The step bound is the "fast" part: Gauss-Newton does not converge there in 25 steps (the ragged T22 test
asserts that), and the exact model gets there in about 8. So the bound must sit between the exact run's count
distribution and a degraded model's. The sibling's bound 9 ("the per-mode lift needs 12 here") must separate the
tied lift from the per-mode lift on a group-ceiling structure.

## Evidence

Scripts in the scratch folder `/tmp/claude-1000/-home-nick-repos-T3Toolbox/352ad37b-5ac8-40a1-a326-a7fc3a369cb2/scratchpad/cifix2/`
(`measure.py`: the test's two fits, optionally at a perturbed start; `tsweep.sh`: sweeps; `counts.py`: outer /
rejected / CG). Runs on `git archive` copies of HEAD and of `85c6e54d`, not the working tree.

**1. The local floor env reproduces the 17, at both commits.** The `tttt` env (numpy 1.22, OpenBLAS auto-detected
on this Ryzen) gives uniform 17 / 1 rejection at `85c6e54d` and at M2; the `t3toolbox` env gives 8 / 0 at both. The
unfixed test fails locally on `tttt` with the CI message.

**2. M2 changes nothing here.** Pre-M2 and M2 runs, compared per history row (objective, gradient norm, accepted
flag), are **bitwise identical** in both envs, ragged and uniform, every row. And over 4 OpenBLAS kernels
(`OPENBLAS_CORETYPE` = Haswell, Zen, Sandybridge, Nehalem) × 1 or 4 threads × both envs (32 runs), pre-M2 and M2
give the same count in every configuration. The count moves with the kernel instead:

| env | Haswell | Zen | Sandybridge | Nehalem |
|---|---|---|---|---|
| `tttt` (numpy 1.22) | 17 | 17 | 9 | 11 |
| `t3toolbox` (numpy 2.4) | 8 | 8 | 8 | 9 |

CI: the passing pre-M2 run and the failing M2 run installed identical packages (numpy 1.26.4, scipy 1.13.1,
jax 0.4.30) on different runner images (`20261116.133` vs `20261116.376`) and took 19 vs 27 minutes, i.e. different
hosts. GitHub runners vary in CPU, so OpenBLAS picks different kernels. The M2 commit coinciding with the flip is a
coincidence of host, not of code.

**3. The count is a draw.** The 17-step history: 3 good steps, a rejection, then 9 accepted steps at a halved trust
radius descending from 6.5e4 to 3.7e4, then 5 steps of quadratic convergence. The 8-step run reaches the same
minimizer (same objective to the last bit) by a shorter globalization. Already at row 0 the two envs report
different gradient norms at the same point (3515 vs 4454): the tangent space itself differs with the completion.

Perturbing the restart's cores by 1e-15 relative (the tie kept exact; zeros stay zero) and rerunning, uniform
`newton_steps`:

| template seed | runs | uniform counts | ragged counts | stops |
|---|---|---|---|---|
| 7 (old) | 96 + 24 | 7 to 16 (two above 13), plus 17 unperturbed on the floor | 9 to 11 | 3 of 96 `objective_floor` (1 uniform, 2 ragged) |
| **9 (new)** | 160 (128 current env, 32 floor env over the 4 kernels) | **6 to 12**: 6×5, 7×55, 8×63, 9×22, 10×9, 11×5, 12×1 | **8 every time** | all `gradient`; ≤ 1 rejection |

The seed was chosen by a sweep of 16 template seeds × 8 starts each; seed 9 had the tightest uniform spread and a
fully deterministic ragged run.

(Perturbing the start *without* keeping the tie exact makes both layers fail outright: 20 straight rejections, the
trust-region floor. At a rank-deficient point the two group modes' completions are chosen independently once their
factors differ by an ulp, so the tied directions stop being tied. Expected: ties are exact by construction,
`t3_tie_tucker_factors`. Not a bug, but worth knowing before writing a perturbation study.)

**4. The sibling (group-ceiling restart, bound 9).** Same class, measured the same way. At seed 7 the tied run's
count was 6 to 13 (9 of 48 above 9) and the per-mode lift's 11 to 17: the distributions overlap, so no bound at seed
7 separates them. At seed 5 (a sweep of 12 seeds): tied **6 to 13** over 72 runs (56 current env, 16 floor env over
the 4 kernels; all `gradient`, ≤ 1 rejection, minimizers equal to 2.4e-16), per-mode lift **22 to 32** over 16 runs. Bound 17 sits in the gap.

## Fix (`tests/backend/test_uniform_curvature.py`, the test only)

- `test_zero_padded_shared_restart`: `_shared_restart_template(*_T22, seed=9)`. The bound stays 16.
- `test_group_ceiling_restart_backend_and_frontend`: `seed=5`, bound 9 → 17, the comment states both
  distributions. Here the bound moves too, because its purpose is separation, and at the new seed the gap is
  [14, 21]; 17 leaves 4 above the tied tail and 5 below the per-mode floor.
- The class docstring: the new seeds, why, the measured ranges, the new observed counts. The full-rank-start test
  (its own construction, seed 7) is unchanged.

## Mutants (scratch copies of HEAD, each against the fixed tests)

| mutant | effect | fixed tests |
|---|---|---|
| GN: the curvature block `Lᵀ(K_{Aᵀω²r} − K_{grad f})` × 0 (`optimizers.py`) | T22 29 steps (ragged 28) | killed, `29 not ≤ 16` (+ the ceiling run hits `max_newton`) |
| half: the curvature block × 0.5 | T22 18 to 19, ceiling 29 | killed, `19 not ≤ 16`, `29 not ≤ 17` |
| per-mode lift on the shared uniform path (`geometry.py`) | ceiling 27 | killed, `27 not ≤ 17` (+ the full-rank-start test) |
| (the same, against the OLD ceiling test, floor env) | ceiling 11 | killed by 2 steps only: `11 not ≤ 9` |
| the `K_{grad f}` term alone × 0 or × 0.5 (`geometry.py`) | T22 8 to 11 | survives: that term is small on this problem; the count cannot see it (neither could the old test). The Hessian-level tests catch it. |
| none | | passes |

The unfixed test on HEAD in the `tttt` env fails with `17 not less than or equal to 16`: the CI failure, reproduced.

## The Gauss-Newton rate control (`test_gauss_newton_control_is_linear`)

**What it is for.** The control for T16: on the misspecified fit, Gauss-Newton + line search converges *linearly*
(every ratio `e_{k+1}/e_k` in `(0.3, 0.9)`, at least 10 of them), so the exact runs' superlinear rates mean something.

**Cause.** The window was `1e-10 < e_k < 1e-1`. The objective here is ~1.3e6, so its resolution
`res = ε(‖ω⊙y‖‖ω⊙r‖ + |f|)` is reached long before `e = 1e-10`. On the failing step (seed 1, row 23,
`e = 1.1e-7`) the predicted decrease is `|pred| = 0.09·res`: the Armijo test compares two objectives that differ in
the last bit. On the floor env it rejected `α = 1` and took `α = 0.5`, giving the ratio 0.25. The current env took
`α = 1` and the ratio 0.52. The steps before that are fine: `|pred|/res` falls by 4 per step, from 2e5 at
`e = 1.8e-4` to 1.2 at `e = 4e-7`. The class's own Newton tests already exclude such steps with the n15 criterion,
`|pred| > 1e3·res`; the control did not.

**Not M2.** The local floor env reproduces the 0.25 at `85c6e54d` and at M2, with printed histories identical
between the two commits in both envs. Under 1e-15 start perturbations (`gn_rate.py`), the old window fails 2 of 20
floor-env runs at seed 1 and 0 of 17 current-env runs.

**Fix.** The control uses the n15 window, the same expression as `_newton_steps`: `e_k < 1e-1` and
`|slope + ½pHp| > 1e3·res`. The bounds `(0.3, 0.9)` and the count `>= 10` are unchanged. This is not a looser
tolerance. It stops measuring a rate on steps where none is measurable, which is the class's stated criterion.
Measured with the new window, over 74 runs (seeds 1, 2 × 17 current-env runs + 20 floor-env runs over 4 kernels):
0 failures, at least 12 ratios per run, all ratios in [0.454, 0.681]. The docstring says why.

**Mutant.** The control run with `hessian='exact'` (a superlinear method in place of Gauss-Newton) fails, with
`3 not >= 10` on both seeds.

## The ragged T22 stop reason (`TestSharedExactNewtonRestart`)

**What it is for.** The ragged T22 test asserts that exact Newton from a zero-padded shared restart "converges on
the gradient test", fast (at most 14 steps, at most 2 rejections), while Gauss-Newton does not converge in 25 steps.

**Cause.** It ran at the default `gtol_rel = 1e-8`, and this objective reaches its resolution near `e = 1e-8`. The
last quadratic step lands at `1e-8` give or take a factor of a few, so the stop reason depended on where it landed.
Below 1e-8 it is `'gradient'`. Just above, the next step's predicted decrease is under the resolution and the run
stops `'objective_floor'`, which is a documented success. Over 48 tied 1e-15 restarts (both envs) at seed 7: 47
stopped `'gradient'` and 1 stopped `'objective_floor'` (floor env, with `e` going 8e-6 → 1e-8, 11 steps, 2
rejections).

**Decision: change the run, not the assertion.** The test's claim is convergence on the gradient test, and a
gradient test placed exactly at the floor cannot make that claim robustly. Both runs, exact and Gauss-Newton, now
stop at `gtol_rel = 1e-6`, two decades above the floor. The last step crosses it from about 1e-5 to about 1e-8. Even
an iterate landing on 1e-6 would take one more step with `|pred| ≈ 1e4·res`, which is measurable, and then stop on
the gradient. Over 108 runs (48 + 48 in the two envs, plus 12 under Sandybridge and Nehalem): **108 `'gradient'`**,
9 or 10 steps, at most 1 rejection. Gauss-Newton is at `e ≈ 4e-4` after 25 steps, so the `'max_newton'` contrast
holds at 1e-6. Accepting both success stops would also be honest, but it gives up the gradient-test claim, and the
coordinator preferred changing the run.

**Mutants.** With the curvature block × 0 (Gauss-Newton under the trust region), the run stops `'max_newton'`:
killed. With the curvature block × 0.5 it takes 27 steps: killed (`27 not ≤ 14`).

## Runs (all four tests)

Laptop, no `-n`, waiting out the other agent's `-n` runs:

| run | result |
|---|---|
| `TestUniformSharedExactNewton`, working tree, `t3toolbox` (py3.11, numpy 2.4, jax 0.10) | 3 passed (20 s) |
| the same, `tttt` (py3.9, numpy 1.22, jax 0.4.30) | 3 passed (26 s) |
| `TestExactNewtonRate` + `TestSharedExactNewtonRestart`, working tree, `t3toolbox` | 4 passed, 6 subtests (22 s) |
| the same, `tttt` | 4 passed (24 s) |
| the unfixed uniform test, HEAD, `tttt` | fails, 17 > 16 |

Not run here: the full suite, the CI floor leg (numpy 1.26). CI will run both.
