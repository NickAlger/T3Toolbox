# S14 review — T3Polynomial update for T3Toolbox 2026.3.0-rc (1f8ab24), tree uncommitted on 388090d
**Verdict: APPROVE.** I made no edits to the T3Polynomial tree. Its 12 modified files are exactly as the implementer left them, still uncommitted.
Housekeeping: I removed the stale `git worktree` registration that the killed reviewer had left at `scratchpad/review/base388` (metadata only). For the baseline I used `git archive 388090d` into `scratchpad/t3poly_base`.
My scripts and logs are in `scratchpad/review/s14/` and `scratchpad/review/s14_suite_*.log`.

1. **Nothing breaking left.** The grep finds nothing that needs a fix. What remains is the docstring history (fitting.py:201, :208–210), the guard (:215–219), the d07 docstring (:63), the test's guard assertion (test:110), and the `**kwargs` passthrough in `_newton_fit` (:184/:226/:235). Each is justified. No `stats['newton']`, `_setup(`, `_geometry_ops` call or `GaussNewtonModel` remains in code. The RC really does lack `optimizers._setup`, and the CHANGELOG confirms the `LocalModel` rename.
2. **K.1 semantics: OK.**
   - The library's zero reference is the default. History rows read `g0_reference == 'zero'`, and the toy run gives the same g0 on ragged and uniform (6.5680).
   - `g0_reference='warm_start'` warns, and keeps the library's own `gtol_rel`.
   - The guard that RAISES on `g0norm_newton=None` (fitting.py:215) is reasonable. Since 2026.3.0 that spelling silently means the opposite. The message names both alternatives: `g0_reference='warm_start'`, or omit the argument.
   - Records keep `newton` (= `newton_steps`) and add `newton_accepted` / `newton_stop_reason`.
   - The docstrings are truthful (module :28–42/:65, `_newton_fit`, both ladders). The ratios are as measured.
3. **I-36 hook: OK.** `_level_newton_kwargs` (:105) accepts None, a dict, or a callable returning a dict or None. Any other return raises TypeError. The level's options are merged LAST (:379, :566), and `info` carries level / tucker_ranks / tt_ranks / dof / records.
   **Default-unchanged comparison** (toy fixture; adaptive n_chunk=1 × 4 levels, plus uniform levels 1..3, max_newton=10; RC 1f8ab24):
   - The new tree's default, `newton_kwargs={}` and `lambda i: None` give **bitwise-identical** records.
   - The base tree plus the rename only (`s0b/t3p_renameonly`; the untouched base KeyErrors on the RC) is **bitwise-identical** to the new tree when a schedule passes the old hand-rolled `_zero_gradient_norm` as `g0norm_newton`. That holds for newton, train, val, hessian, step_control and every gnorm on every level.
   - So the only default change is the intended K.1 reference swap. New/old g0 is 1.606, 1.257, 0.952 and 1.000 (L0–L3); Newton steps go 10/6/4/0 → 10/5/4/0; val at the best level is 1.660e-4 → 1.574e-4. Still GN + line search on every level.
   - The 2-line callable `lambda info: {} if info['level'] < 2 else {'hessian': 'exact'}` runs. Adaptive: L0–1 GN/LS, then L2–3 exact/TR, stop 'gradient'. Uniform ladder: L1 GN, then L2–3 exact.
4. **I-37: OK.** `test_cold_zero_start` passes (ragged; GN and exact `_newton_fit` from `zeros`, plus both ladders from a rank-1 zero seed). I also checked by hand (`review/s14/cold.py`):
   - `_newton_fit(zeros)` runs on ragged and uniform × GN and exact. Loss goes 3.50 → 0.105; the ref is 'zero' with an identical g0; GN takes 14 steps, exact 7.
   - The adaptive ladder from a zero seed runs with the GN → exact schedule on ragged and uniform.
   - Optional suggestion (not a defect): add a `backend='uniform'` subTest to the test, which today covers ragged only.
5. **K.2 test: non-vacuous.** It asserts `'zero'` on every ladder record and on a direct `_newton_fit` call, and `'warm_start'` on the opt-out (ladder and direct). It also checks warm g0 < 0.5·zero g0, *strictly* more warm steps, and that `g0norm_newton=None` raises. Warm-starting from a fitted rank-2 tensor is a sound strengthening (a resize of the unfitted seed is not deflated).
6. **K.3 scripts: OK.**
   - d07: `.problem`. `LeastSquaresSetup` is a frozen dataclass (`problem`, `x0_cores`, `rewrap`; t3toolbox/optimizers.py:186).
   - d05: keeps `ls_steps is not None` AND checks `step_control`, and adds `cg_boundary`.
   - b01: the histogram counts LS rows only, plus `n_rows`.
   - x06: the `EXACT` column. x07: `HESS=fd|exact`, `CONTROL_GTOL`, `EXACT_CONTROL`.
   - d08: the public constructor with named fields.
   - The `implicit_probing` shim exists at `scratchpad/s0b/shim`; `implicit_probing` does not import without it. With the shim I re-ran two scripts on the implementer's e01 L3 data:
     - **x06** (STEPS=1e-3,1e-4): the exact column converges at O(t²), e.g. rel 4.75e-4 → 4.76e-6.
     - **x07** (HESS=exact, N_ITER=3): exit 0; GN control at max_newton, exact control 3/3 accepted, JSON written.
   - Judgment call worth noting: x07's controls now default to gtol 1e-12 instead of the inherited 1e-3. This is documented, and it keeps the controls from being a 1-step no-op.
   - I did not re-run e01, o01, b01, d05, s05, d07 or d08. I accept the implementer's logs; the Darcy data is absent.
7. **Suite** (system python: py3.13 / numpy 2.5.3 / jax 0.12; `-n 2 --continue-on-collection-errors`):
   - **Baseline 388090d:** 10 failed / 22 passed / 34 skipped / 2 errors (test_eigen and test_ot cannot import implicit_probing).
   - **After:** 31 passed / 0 failed / 34 skipped / the same 2 errors (206 s).
   - py311latest (py3.11 / numpy 2.4.6 / jax 0.10.2): 31 passed / 0 failed / 34 skipped / the same 2 errors (213 s).
8. **Notes: accurate.** HANDOFF.md has a dated top block (2026-10-29), and upstream-notes has two ✅ FIXED banners and a dated section at :815.
   - The e01 ratios read 2.175 / 0.727 / 0.943 / 0.967, plus 0.95 on the toy, consistent with my 0.952 at rank 3.
   - The hook API and the cold-start options are described correctly. README and the pyproject comment now require ≥ 2026.3.0.
   - x01's `NameError` (`K`, x01:55) is pre-existing and was left untouched (verified unchanged). The 3 pyflakes warnings in x06/d08 are also pre-existing (verified on base).
