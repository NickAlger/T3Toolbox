# W1 handoff: the backend of `newton_cg(hessian='switch')` (the Gauss-Newton-to-exact switch, 2026.5.0)

Implementer's record for the session. Base: HEAD `c1a42c24` (timing P1-P3 committed, the store-rename arc complete; the
plan `dev/gn_newton_switch_plan_2026-11-13.md` at draft 3, every §9 decision resolved). Nothing is committed. Scope: the
plan's §8 row W1, exactly -- the backend modes and checks, `_check_switch_options`, `_objective_resolution` (a bitwise
refactor), `_switch_sample`, `_gauss_newton_view`, the loop of §1, the `NewtonInfo` fields and `_newton_scalar_record`,
the `stats` keys, the tracking product lapped into `PhaseSeconds.tracking`; tests T1-T9 at W1's level, T12, G-bit.

Not pulled forward (W2): the display segment (`optimizer_display.py` is untouched), the frontend keywords, early checks,
forwarding and `least_squares_problem` sizing (the frontend `optimizers.py` is untouched; it still rejects `'switch'`
through `_check_hessian`'s default modes), the frontend docstrings. Not pulled forward (W4): docs pages, CHANGELOG,
contributor records.

**Changed:** `t3toolbox/backend/optimizers.py`, `tests/backend/test_optimizers.py`, `tests/test_dispatch.py`,
`tests/test_optimizers_frontend.py` (the self-expiring exemption, §6). **New:** this file.

## 1. What was built

| where | what |
|---|---|
| modes | `_HESSIAN_MODES` unchanged (a local model's two); `_NEWTON_HESSIAN_MODES = _HESSIAN_MODES + ('switch',)`. `_check_hessian(hessian, who, modes=_HESSIAN_MODES)`: its message is built from `modes`, byte-identical for the default (`"'gauss_newton' or 'exact'"`), `"'gauss_newton', 'exact' or 'switch'"` for newton_cg's. `_require_exact_capable(kind, geom, hessian='exact')`: the mode named in its two messages (byte-identical for `'exact'`). `_check_step_control`: `None` -> `'trust_region'` for `'exact'` and `'switch'`. |
| `_check_switch_options` | §2.1's rules, raised before any work: each `switch_*` option without `hessian='switch'` (`"newton_cg: switch_k, switch_track set without hessian='switch'"`, names sorted, as the padded group's message); `switch_tau` a real number in `[0, 1)` (not a bool); `switch_k` an integral `>= 0` (not a bool); `switch_track` `'countable'` / `'all'`, case-insensitive (a non-`str` raises); `switch_tau` / `switch_track` with `switch_k=0`; `switch_k` / `switch_cg_forcing_power` with `switch_tau=0` (so `(0, 0)` raises either way). Returns `None` outside switch mode, else `(tau, k, power, track)` with `track` `None` iff `k = 0`. `switch_cg_forcing_power` unchecked and passed through unconverted, as `cg_forcing_power` is. Defaults are the constants `_SWITCH_TAU = 0.25`, `_SWITCH_K = 2`, `_SWITCH_CG_FORCING_POWER = 1.0`, `_SWITCH_TRACKS = ('countable', 'all')`; `_SWITCH_ROUNDING = 100.0`. |
| `_objective_resolution(eps, data_norm, misfit, f)` | `eps * (data_norm * math.sqrt(2.0 * misfit) + abs(f))`, the trust region's expression verbatim; the `objective_floor` check now calls it (bitwise: battery, §5). |
| `_switch_sample(delta_f, predicted, curvature, floor, tau)` | `e_gn = abs(delta_f - predicted)`, `e_n = abs(delta_f - (predicted + curvature))`, `informative = isfinite(e_gn) and isfinite(e_n) and max(e_gn, e_n) > floor`, `hit = informative and e_n < tau * e_gn` (strict). |
| `_gauss_newton_view(lm)` | `dc.replace(lm, curvature=None)`. |
| `NewtonInfo` | five fields appended after `timing`: `switch_pKp`, `switch_e_gn`, `switch_e_n`, `switch_count`, `switch_fired` (all `None` by default). `_SWITCH_FIELDS` is excluded from `_NEWTON_SCALAR_FIELDS` (equal to 2026.4.0's tuple: `test_off_is_today` unedited and green); `_newton_scalar_record` appends the five iff `info.switch_fired is not None`. `hessian`'s comment and docstring now say "this row's model". |
| `newton_cg` | four keyword-only options after `timing` in their own blank-line group (`switch_tau`, `switch_k`, `switch_cg_forcing_power`, `switch_track`, each `None` by default). Validation: `_check_hessian(..., _NEWTON_HESSIAN_MODES)`, then `_check_switch_options`, then (switch mode) `_require_exact_capable(kind, geom, 'switch')`, all before `curvature_damping`'s check and before any work. Loop (switch mode only; every other mode's path is today's): each point is built in exact mode (`model_hessian = 'exact'` for `'exact'` and `'switch'`; the timed stock build gets it too); in the Gauss-Newton phase `lm_track, lm = lm, _gauss_newton_view(lm)`, so CG, `_initial_radius`, `pHp`, the retraction, the lift diagnostics (`None`) and the row's `lm` are the view's, while `g` is read from `lm_track` (its one transpose); forcing power `switch_power if switched else cg_forcing_power`; `pHp` takes the exact branch on `exact or switched`; `fallback` names the existing descent test; after step control, a GN-phase step row gets `e_gn = |Δf − predicted|`, and a tracked row (countable, or every step row under `'all'`) one `lm_track.curvature_quadratic(p)` lapped as `'tracking'`, the floor `_SWITCH_ROUNDING · _TR_RESOLUTION_FACTOR · _objective_resolution(_resolution_eps(lm.residual), data_norm, misfit, f)` (`data_norm` lazily, once per run, shared with the floors) and `_switch_sample(delta_f, predicted, ½α²·pKp, floor, tau)`; `count = count + 1 if (countable and hit) else 0`; at `count >= k`, `switched, switched_at = True, it + 1`. Countable = `accepted and not fallback and (tr_accepted if trust_region else (ls_steps == 0 and not ls_failed))` (§1.2 verbatim; CG's boundary / truncation does not enter). Row fields: GN-phase step rows all five (`switch_fired` = the row switched); exact-phase rows `switch_fired=False`, the other four `None`; the converged row `switch_fired=False` and, in the GN phase, the carried count. Accepted: `x, lm, lm_track = x_trial, None, None`. After the loop, switch mode only: `stats` gains `hessian='switch'`, `switch_track`, `switched_at`, after the other keys and before `timing`. |
| timing | `PhaseSeconds.tracking` (declared by timing P2) carries the tracking product on tracked rows; the GN phase's exact build is lapped as `frame` .. `gradient` .. `curvature` by the stock `_local_model_phases` (plan §3.3). The floor's arithmetic stays in `remainder` (only the product is the phase). The field's docstring and comment, and `curvature` / `gradient`'s bullets, say so. |
| docstrings | `newton_cg`: the signature comments of `hessian` / `curvature_damping` / `step_control` and the new group; a **Switching from Gauss-Newton to the exact Hessian** paragraph (the rule in math and words, the counting, the gate with its float32 advice, `'all'`, step control and radius, the two forcing powers, `tau=0` / `k=0`, each call starts in Gauss-Newton, when it does not fire, reporting, the option rules); the stats paragraph's three keys. `NewtonInfo`: `hessian` per row, a **The switch** paragraph (`lm` on GN-phase rows is the view, the five fields), `pKp` / lift fields "exact rows". `PhaseSeconds`, `_check_step_control`, the new helpers. |

## 2. Judgement calls

1. **`_require_exact_capable` gains `hessian='exact'`** so that switch mode's message names `'switch'` (plan §2.1:
   "raises at the call, naming `'switch'`"; §2.2 lists no signature for it). `'exact'`'s two messages are byte-identical
   (battery). Exact mode keeps its check inside `Problem.local_model` (unchanged); only switch mode checks up front.
2. **The backend `newton_cg`'s invalid-`hessian` message now lists `'switch'`** (`"newton_cg: hessian must be
   'gauss_newton', 'exact' or 'switch' (case-insensitive); got 'newton'"`). It was round 1's only battery difference from
   `c1a42c24` (two entries: a bad string and a bool; round 2 adds the damping message, §7) and is inherent to the feature. The frontend message is unchanged
   in W1 (the frontend still calls `_check_hessian` with the default modes); see §6 for what W2 meets.
3. ~~**`curvature_damping`'s Gauss-Newton message is kept byte-identical**~~ -- superseded in round 2 (§7): the session
   ruled the message widened to name both modes.
4. **Option types where the plan is silent.** `switch_tau`: a `numbers.Real` that is not a `bool`, in `[0, 1)` (a
   string or `False` raises; numpy floats pass; NaN fails the range). `switch_k`: a `numbers.Integral` that is not a
   `bool` (numpy ints pass; `2.0` raises). `switch_cg_forcing_power` is passed through without `float()`, exactly as
   `cg_forcing_power` reaches the forcing term, so T2's bitwise identity holds for any numeric type.
5. **Field and key placement.** The five `NewtonInfo` fields are appended after `timing` (the plan: "appended"); the
   history rows carry them after the scalar fields; the three `stats` keys come after the other optional keys
   (`start_frame`, `start_movements`, `cone_movements`) and before `timing`, which stays last.
6. **Backend docstrings are W1's**, for what W1 builds (the timing feature's P2 precedent, its judgement call 12): a
   keyword without a docstring would ship undocumented between slices. No doctest is added: §7 puts the switch's
   doctest in docs §4.14 (W4); the frontend docstrings are W2's.
7. **`check_timing` (the shared timing helper in `test_optimizers.py`) generalized in two assertion lines**: `tracking`
   ran iff the row has `switch_pKp`; `curvature` ran on a built row iff its `hessian` is `'exact'` or it is a switch row
   (`switch_fired is not None`). For every non-switch row the two conditions are the old ones (`switch_pKp` and
   `switch_fired` are `None` there), so the timing tests' assertions are unchanged in meaning; the switch's timed test
   reuses the helper rather than duplicating its 80 lines. T12 says "existing tests pass unedited": no test was edited,
   but the helper they share was -- the session's call.
8. **T4 through runs, with two test-side devices.** Natural reset rows are almost never would-be hits (a rejected or
   backtracked step's errors are large and alike), so a mutation that drops a reset condition would pass unnoticed.
   (a) `TestSwitchNewton.forced_hits()` patches `_switch_sample` to report every tracked row a hit (errors as
   computed): the count then rises on every countable row and resets on every other, so each reset condition, the
   boundary rule and the `countable` term of the count update are observable. (b) `_cg_stub_at(n, change)` replaces
   CG's `n`-th step (an overshoot `4p` for an Armijo backtrack; `+g` for the −g fallback), in the style of the existing
   descent-guard test. A rejected trial comes from `_OffsetProblem` (an offset of 5 % of `f0`: three accepted steps,
   then rejections), a forced accept from `c_armijo=10`, an accepted boundary step from the default radius. The
   unmocked `test_reset_rows` covers the plan's literal T4 rows and the `'all'` equality, and pins `e_GN` / `e_N`
   against the row's own fields on every tracked row (it is what catches a dropped `α²` on a backtracked `'all'` row).
9. **T5's literal criterion is vacuous against mutations on its own** (§3); `test_rounding_gate_at_float32` adds a spy
   on `_switch_sample` that records each tracked row's `floor` and checks it against
   `1000·ε₃₂·(‖y‖·√(2·misfit) + |f|)` from the row (relative 1e-4: the test's `‖y‖` is float64, the run's float32). The
   gate's logic itself is pinned by `test_switch_sample`.
10. **T6's problem** is `newton_cg`'s docstring problem (a noisy rank-3 target, rank-2 fit, probe derivatives of order 1,
    warm-started by five Gauss-Newton steps), not §4.12's doctest, whose `ww3` / `x0` come from the page's earlier
    setup. Same structure as the plan names. It fires at row 1 (`switched_at = 2`): 6 Newton steps against GN-TR's 25.
11. **T8's G-bit is the battery** (§5), not a test with hard-coded digests (float bits differ across BLAS / jax
    versions, so a digest test would fail on CI's other legs). Recorded here: numpy 319 entries / 27,101 arrays, jax 27
    entries / 1,976 arrays.
12. **Tests beyond §5's rows**, each because a mutation needed it: `test_one_transpose_per_point` (a switch_tau=0 run
    makes exactly the Gauss-Newton run's `kind.transpose` calls: the view's `gradient` is never read -- numerically
    invisible otherwise), `test_defaults` (the constants, and the defaulted run equals the spelled-out one; a changed
    `_SWITCH_TAU` fires at the same rows on these problems), and the regularized ragged manifold case added to T1/T2's
    matrix (the view must carry the regularizer).
13. **T9 split by layer** (`test_use_jit_ragged` / `_uniform` / `_shared_uniform`, one shared `check_use_jit`): the
    three together take ~95 s, mostly the first Gauss-Newton compile per structure; split, xdist spreads them.
14. **Equivalent mutants kept as the plan writes them**: in `countable`, `not ls_failed` is implied by `ls_steps == 0`
    (a failed search has `ls_steps = 40`), and `accepted` is implied by either branch (`tr_accepted` in the trust
    region, `ls_steps == 0` on the line search). Kept for readability (they say what §1.2 says); no test can see them.

## 3. Where the plan could not be followed literally

- **T5's criterion alone cannot fail under a gate mutation.** On a noise-free true-rank float32 run the at-floor rows
  are never would-be hits (there `e_N ≥ e_GN`, measured: rows 8-10 of the test's run), so "every at-floor countable row
  has count 0" holds with or without the gate. Asserted as written (with at least one such row required), and
  supplemented as judgement call 9 says. Nothing in the library differs from the plan.
- Otherwise nothing blocked. Readings recorded above: `_require_exact_capable`'s parameter (1), the backend message
  (2), the kept damping message (3), the option types (4), placement (5), docstrings (6), the helper (7), T4's devices
  (8), T6's problem (10), G-bit (11).

## 4. The printed switch run (for the maintainer)

The problem is `newton_cg`'s docstring example (probe derivatives of order 1 on (5, 5, 5), a rank-3 target with 1 %
noise, the rank-2 fit warm-started by five Gauss-Newton steps), run with `hessian='switch'` and `g0_reference=
'warm_start'` (the docstring's comparison runs use it), numpy, the maintainer's laptop. The display is today's
`make_newton_display` (W2 adds the `sw` segment; the exact rows already show their `κ σ ε` segment, the Gauss-Newton
rows none); below it, `stats`' switch keys and the switch fields of each history row (script: the session scratchpad
`w1/s/example.py`). The exact model predicted Δf 78x better than Gauss-Newton on row 0 and 337x better on row 1, so
the switch fires on row 1 (the **switch row**: count 2, `switch_fired=True`) and row 2 is the first exact row
(`switched_at = 2`); the run converges in 6 Newton steps, against 25 for Gauss-Newton with the trust region and 5 for
the exact Hessian at power 1.0 from the start (the docstring's two runs).

```
iter  0 | obj 2.547e+05  ‖g‖ 9.84e+02 (1.0e+00·g₀) | CG 1/200 tol 4.9e+02 resid 2.3e+02 ✓ | TR Δ 2.51e+00 ✓ ‖Δx‖/‖x‖ 1.9e-02 | Δf -1.07e+03 ρ 0.86 | 0.04s
  rel err  rows=mode  cols=order
           ord0     ord1
    m0  3.7e-01  1.9e-01
    m1  2.7e-01  2.8e-01
    m2  2.8e-01  2.6e-01
iter  1 | obj 2.537e+05  ‖g‖ 4.17e+02 (4.2e-01·g₀) | CG 1/200 tol 2.1e+02 resid 1.2e+02 ✓ | TR Δ 2.51e+00 ✓ ‖Δx‖/‖x‖ 8.7e-03 | Δf -2.03e+02 ρ 0.85 | 0.04s
  rel err  rows=mode  cols=order
           ord0     ord1
    m0  3.6e-01  1.9e-01
    m1  2.8e-01  2.8e-01
    m2  2.8e-01  2.5e-01
iter  2 | obj 2.535e+05  ‖g‖ 2.43e+02 (2.5e-01·g₀) | CG 2/200 tol 6.0e+01 resid 1.9e+01 ✓ | TR Δ 2.51e+00 ✓ ‖Δx‖/‖x‖ 7.2e-03 | Δf -9.75e+01 ρ 1.00 | 0.06s | κ -3.5e-01 σ 5.8e+01 ε 1.3e-04
  rel err  rows=mode  cols=order
           ord0     ord1
    m0  3.6e-01  1.9e-01
    m1  2.8e-01  2.8e-01
    m2  2.8e-01  2.5e-01
iter  3 | obj 2.534e+05  ‖g‖ 1.87e+01 (1.9e-02·g₀) | CG 4/200 tol 3.5e-01 resid 1.9e-01 ✓ | TR Δ 2.51e+00 ✓ ‖Δx‖/‖x‖ 3.9e-04 | Δf -4.24e-01 ρ 1.00 | 0.11s | κ -1.7e-02 σ 5.8e+01 ε 1.3e-04
  rel err  rows=mode  cols=order
           ord0     ord1
    m0  3.6e-01  1.9e-01
    m1  2.8e-01  2.8e-01
    m2  2.8e-01  2.5e-01
iter  4 | obj 2.534e+05  ‖g‖ 1.91e-01 (1.9e-04·g₀) | CG 8/200 tol 3.7e-05 resid 2.8e-05 ✓ | TR Δ 2.51e+00 ✓ ‖Δx‖/‖x‖ 5.7e-06 | Δf -6.38e-05 ρ 1.00 | 0.19s | κ -3.0e-01 σ 5.8e+01 ε 1.3e-04
  rel err  rows=mode  cols=order
           ord0     ord1
    m0  3.6e-01  1.9e-01
    m1  2.8e-01  2.8e-01
    m2  2.8e-01  2.5e-01
iter  5 | obj 2.534e+05  ‖g‖ 2.82e-05 (2.9e-08·g₀) | CG 14/200 tol 8.1e-13 resid 5.3e-13 ✓ | TR Δ 2.51e+00 ✓ ‖Δx‖/‖x‖ 6.4e-10 | Δf -2.91e-11 ρ 26.73 | 0.31s | κ -1.4e-02 σ 5.8e+01 ε 1.3e-04
  rel err  rows=mode  cols=order
           ord0     ord1
    m0  3.6e-01  1.9e-01
    m1  2.8e-01  2.8e-01
    m2  2.8e-01  2.5e-01
iter  6 | obj 2.534e+05  ‖g‖ 1.97e-10 (2.0e-13·g₀) | converged (‖g‖ ≤ gtol·‖g₀‖, g₀ = warm start)
  rel err  rows=mode  cols=order
           ord0     ord1
    m0  3.6e-01  1.9e-01
    m1  2.8e-01  2.8e-01
    m2  2.8e-01  2.5e-01
stats: newton_steps 6, newton_accepted 6, stop_reason 'gradient', hessian 'switch', switch_track 'countable', switched_at 2
row  hessian       tr_acc  e_GN       e_N        e_N/e_GN   pKp(track)  count  fired
  0  gauss_newton  True    1.670e+02  2.130e+00  1.275e-02  3.297e+02   1      False
  1  gauss_newton  True    3.703e+01  1.100e-01  2.971e-03  7.429e+01   2      True
  2  exact         True    —          —          —          —           —      False
  3  exact         True    —          —          —          —           —      False
  4  exact         True    —          —          —          —           —      False
  5  exact         True    —          —          —          —           —      False
  6  exact         —       —          —          —          —           —      False
```

The same problem from the random rank-2 start that the docstring's five warm-up steps begin from: a longer Gauss-Newton
phase whose first three countable rows are misses (the exact model 3.5-9.6x WORSE: the noisy transient), then two hits
on rows 3 and 4. Row 0's step stopped on the trust-region boundary (`∂`) and was tracked like any other countable row
(a miss). Display headers and the tables:

```
iter  0 | obj 3.707e+06  ‖g‖ 1.85e+04 (1.0e+00·g₀) | CG 1/200 tol 9.3e+03 resid 5.2e+03 ∂ | TR Δ 5.13e+01 ✓ ‖Δx‖/‖x‖ 7.4e+00 | Δf -3.62e+05 ρ 0.76 | 0.04s
iter  1 | obj 3.345e+06  ‖g‖ 3.17e+04 (1.7e+00·g₀) | CG 1/200 tol 1.6e+04 resid 8.1e+03 ✓ | TR Δ 1.03e+02 ✓ ‖Δx‖/‖x‖ 1.8e+00 | Δf -1.27e+06 ρ 1.07 | 0.03s
iter  2 | obj 2.080e+06  ‖g‖ 3.17e+04 (1.7e+00·g₀) | CG 1/200 tol 1.6e+04 resid 7.4e+03 ✓ | TR Δ 1.03e+02 ✓ ‖Δx‖/‖x‖ 9.9e-01 | Δf -1.52e+06 ρ 1.15 | 0.03s
iter  3 | obj 5.621e+05  ‖g‖ 1.37e+04 (7.4e-01·g₀) | CG 1/200 tol 6.9e+03 resid 3.5e+03 ✓ | TR Δ 1.03e+02 ✓ ‖Δx‖/‖x‖ 3.4e-01 | Δf -2.89e+05 ρ 1.07 | 0.03s
iter  4 | obj 2.727e+05  ‖g‖ 3.17e+03 (1.7e-01·g₀) | CG 1/200 tol 1.3e+03 resid 1.2e+03 ✓ | TR Δ 1.03e+02 ✓ ‖Δx‖/‖x‖ 8.5e-02 | Δf -1.64e+04 ρ 0.96 | 0.03s
iter  5 | obj 2.562e+05  ‖g‖ 1.48e+03 (8.0e-02·g₀) | CG 3/200 tol 1.2e+02 resid 5.7e+01 ✓ | TR Δ 1.03e+02 ✓ ‖Δx‖/‖x‖ 3.3e-02 | Δf -2.84e+03 ρ 1.00 | 0.09s | κ -2.1e-01 σ 5.8e+01 ε 1.3e-04
iter  6 | obj 2.534e+05  ‖g‖ 6.22e+01 (3.4e-03·g₀) | CG 6/200 tol 2.1e-01 resid 7.3e-02 ✓ | TR Δ 1.03e+02 ✓ ‖Δx‖/‖x‖ 1.9e-03 | Δf -7.16e+00 ρ 1.00 | 0.14s | κ -1.7e-01 σ 5.8e+01 ε 1.3e-04
iter  7 | obj 2.534e+05  ‖g‖ 9.10e-02 (4.9e-06·g₀) | CG 11/200 tol 4.5e-07 resid 2.0e-07 ✓ | TR Δ 1.03e+02 ✓ ‖Δx‖/‖x‖ 2.6e-06 | Δf -1.37e-05 ρ 1.00 | 0.25s | κ -1.1e-01 σ 5.8e+01 ε 1.3e-04
iter  8 | obj 2.534e+05  ‖g‖ 2.32e-07 (1.2e-11·g₀) | converged (‖g‖ ≤ gtol·‖g₀‖, g₀ = warm start)
stats: newton_steps 8, newton_accepted 8, stop_reason 'gradient', hessian 'switch', switch_track 'countable', switched_at 5
row  hessian       tr_acc  e_GN       e_N        e_N/e_GN   pKp(track)  count  fired
  0  gauss_newton  True    1.134e+05  1.093e+06  9.635e+00  -1.959e+06  0      False
  1  gauss_newton  True    8.361e+04  6.778e+05  8.107e+00  -1.523e+06  0      False
  2  gauss_newton  True    2.010e+05  7.011e+05  3.487e+00  -1.804e+06  0      False
  3  gauss_newton  True    1.971e+04  1.599e+03  8.113e-02  -3.623e+04  1      False
  4  gauss_newton  True    6.687e+02  5.981e+01  8.944e-02  1.457e+03   2      True
  5  exact         True    —          —          —          —           —      False
  6  exact         True    —          —          —          —           —      False
  7  exact         True    —          —          —          —           —      False
  8  exact         —       —          —          —          —           —      False
```

## 5. Verification

One job at a time, current env `t3toolbox` (py3.11, numpy 2.4.6, jax 0.10.2) unless stated; scripts and logs in the
session scratchpad `w1/`.

| check | result |
|---|---|
| **Non-breaking, bitwise against `c1a42c24`** (a `git archive` copy of the commit in the scratchpad, no worktree; `w1/battery/battery.py` = the P3 reviewer's battery extended, run in both trees; `w1/battery/compare.py`: container types, dict key order, dtypes, shapes, float bits). Covers: `newton_cg` on numpy, ragged and uniform, unshared and shared, probe and probe-derivative kinds -- GN line search and trust region, exact TR with rejections, exact line search, padded start, `'cone'`, the `padded_*` / `cone_*` options, a given `start_frame`, warm-start overrides, Armijo exhausted under `'stop'` / `'accept'`, converged, `max_newton=0`; corewise, regularized, `_OffsetProblem` to `trust_region_floor`; every `NewtonInfo` row (all fields but `timing`, with its model's parts) and the history's key tuples; the display's records and texts; `local_model` both modes; `padded_start`, `gradient_descent`; the frontend `newton_cg` (verbose texts, `stats`, errors) on both layers, shared and corewise; hand-built `format_newton_iter` rows. Added for W1: **timed runs** (x, `stats` but `timing`, the rows, each record's `None` pattern, `stats['timing']`'s keys) on numpy (GN LS / TR, exact TR with rejections, exact LS, corewise exact, converged, exhausted accept, padded + cone exact on both layers) and jax `use_jit` (GN, exact TR); **errors**: `newton_cg`'s twelve structural messages (`hessian` bad / bool, damping with GN, the radius / accept rules, step control, `on_line_search_failure`, `g0_reference`, padded / cone without their group), a kind without `corewise_hessian` under `'exact'` (in `newton_cg` and `local_model`), `local_model`'s bad `hessian`, `_check_step_control`. jax x64: `use_jit` GN / padded + cone / exact TR on both layers and kinds, eager ragged GN / cone / exact, frontend `use_jit` | **numpy 319 entries, 27,101 arrays: 2 mismatches, both the expected message** (`newton_cg`'s invalid `hessian`, a string and a bool, now listing `'switch'`; judgement call 2). **jax 27 entries, 1,976 arrays: 0 mismatches.** The switch fields are asserted `None` on every row the battery records. A planted one-ulp change is caught. |
| **Mutation testing** (`w1/mut/mutate.py`: each mutation on a fresh copy of `t3toolbox/` + `tests/`, the targeted tests rerun) | **47 of 47 caught** (numpy 43, jax 4). The view as the exact model; the gradient read from the view (caught by the transpose count); the GN phase on the switch power; the exact phase on `cg_forcing_power`; the fallback / backtrack / rejection counted; both line-search conditions dropped; the count update without `countable`; boundary steps resetting; tracking only under `'all'`, or always; the curvature without `α²`; `switched_at = it`; firing at `k + 1`; `switch_fired` never set; the converged row's count `None` / 0; the row's `hessian` the run mode; no `stats` keys; the switch fields in every history row; the default step control the line search; no capability check at the call; no group check; `tau` range open at 0; a bool `tau` / `k`; `switch_track` not lowered; each cross rule removed; damping dropped in switch mode; the radius reset at the switch; `lm_track` not reset on accept; the gate without the floor, on `e_GN` only, non-strict hit, non-finite informative; no tracking lap, the product lapped as `forms`; the exact phase tracking; `_SWITCH_TAU = 0.5`; `e_GN` missing on reset rows; the mirror exemption outliving W2 (`switch_k` added to the frontend signature: red). jax (T5 / T9): the floor without `_SWITCH_ROUNDING`, at float64's ε, without the data norm; a view whose structure is not Gauss-Newton's (compiles a second program in the Gauss-Newton phase). Equivalent mutants not run: judgement call 14 |
| new tests | `test_optimizers.py::TestSwitchNewton` **13 passed** (110 subtests, 33 s); `test_dispatch.py::TestSwitchNewton` **4 passed** (T5 22 s; T9 ragged 42 s, uniform 24 s, shared uniform 31 s, measured separately) |
| full suite, `pytest tests/ -q -n 3` | **1,397 passed, 5 skipped**, 49,003 subtests, 0 failed (26 min); the one warning is the pre-existing adam-on-manifold one |
| module doctests, `pytest --doctest-modules t3toolbox -n 3` | **241 passed** (no doctest added or edited) |
| `python -m sphinx -W --keep-going -b html docs <scratch>` | **exit 0, no warning**; `newton_cg`, `NewtonInfo`, `PhaseSeconds` render the new text; the private helpers stay out |
| floor env `tttt` (py3.9, numpy 1.22.4, jax 0.4.30), single process: `test_optimizers.py`, `test_optimizers_frontend.py`, `test_dispatch.py::TestSwitchNewton`, `::TestNewtonTiming` | **117 passed, 1 skipped** (5.5 min): the skip is the timing feature's slow-tier row. T9's program-cache checks run there (0.4.30 has `_cache_size`); T5 runs at float32 there too. (`test_reset_rows`' last check was tidied after the full suite, before this run; rerun alone on the current env: passed) |

No `docs/*.md` page changed, so the page doctests were not rerun (the battery shows every non-switch run's numbers and
texts unchanged).

## 6. W2 must delete (and meet)

- **Delete** -- `tests/test_optimizers_frontend.py`, `TestLeastSquaresProblemAndKeywords.
  test_newton_cg_keywords_mirror_backend`: the two-line `not_yet = (...)` tuple of the four switch keywords, the loop
  below it (`for name in not_yet: self.assertNotIn(...)`), and the `or name in not_yet` in the mirror loop. The
  `assertNotIn` makes it self-expiring: once W2 adds any of the four frontend keywords the test fails with "`<name>` is
  now a frontend keyword: delete not_yet (the switch plan, W2)" (checked: a scratch copy with `switch_k` added to the
  frontend signature is red, §5's mutation list).
- **Meet** -- once the frontend calls `bopt._check_hessian(hessian, 'newton_cg', bopt._NEWTON_HESSIAN_MODES)` (plan
  §2.3), its message lists `'switch'`, so the regex `"newton_cg: hessian must be 'gauss_newton' or 'exact'"` in
  `tests/test_optimizers_frontend.py`, `TestFrontendExactNewton.test_options_with_no_effect_and_unbuilt_geometries_raise`,
  must change with it. `least_squares_problem`'s own
  `_check_hessian` call needs the three modes too (§2.3: `'switch'` sizes as `'exact'`).

## 7. Round 2 (review `W1_review.md`: REVISE, four MINOR, three NIT; and the session's ruling)

No library code path changed. The library diff this round is the `curvature_damping` message (the session's ruling) and
one docstring sentence (M4), whose paragraph is re-wrapped to 120 columns (the reviewer's follow-up NIT: line 1870 was
188 characters; whitespace only, checked word for word).

| finding | change | checked |
|---|---|---|
| **M1** a countable miss's reset untested in the loop | `TestSwitchNewton.test_a_miss_resets`: `_switch_sample` scripted (the `forced_hits` pattern, one `(informative, hit)` per tracked row) hit, miss, hit, uninformative, hit, hit on `_pd_problem(0)` from `_rank2_start()` under the default trust region; every one of the six rows countable; counts `[1, 0, 1, 0, 1, 2]`, `switch_fired` on row 5 only, `switched_at = 6` | mutations "a countable miss does not reset" (`... else (count if countable else 0)`) and "an uninformative countable row does not reset" (`... else (count if (countable and not informative) else 0)`): **caught** |
| **M2** a capped CG step untested | `test_what_counts_and_what_resets` gains a forced-hit `'capped CG'` subtest per step control (`cg_maxiter=2`, `g0norm_cg=1e12`, `switch_k=50`): every row has `cg_iters == 2` and is neither converged, on the boundary nor truncated; counts 1..6. Under the trust region it takes `tr_radius0=1e3`: with the default radius row 0's CG stops on the boundary at its second iteration (measured), so the review's "every row capped" holds for the line search and, with the larger radius, for the trust region | "capped CG resets" (`... and (cg_converged or cg_boundary) and ...`): **caught** |
| **M3** T5 did not test its gate | the spy also records the real sample's verdict; per tracked row it must be uninformative where `max(e_GN, e_N) <= 0.99·floor` and informative above `1.01·floor` (the margin absorbs the test's float64 `‖y‖`); at least one uninformative row | "gate deleted, T5 alone": **caught** |
| **M4** two overstated docstring clauses | "(the run is the Gauss-Newton run with the same step control, bit for bit, plus the tracking)"; "(the exact run with ``cg_forcing_power=switch_cg_forcing_power``)"; the paragraph re-wrapped | read; the over-120 scan of the paragraph is empty |
| **N1** the view's companion | `test_one_transpose_per_point` also counts `ManifoldGeometryOps.precompute`, on the ragged manifold and the shared manifold (the companion count > 0 there): a switch_tau=0 run makes the Gauss-Newton run's counts exactly | "view drops the companion" (`geom_aux=None`): **caught** |
| **N2** `check_timing`'s mode | `check_timing(..., switch=False)`: `tracking` ran iff `switch and switch_pKp is not None`; `curvature` on a built row iff `hessian == 'exact' or switch`. The pre-W1 callers' assertions are the old ones by construction; the switch's timed test passes `switch=True` | the timing tests green (below) |
| **N3** §6's line number | §6 names `TestFrontendExactNewton.test_options_with_no_effect_and_unbuilt_geometries_raise` | -- |
| **Session ruling** the damping message | backend `newton_cg` with `curvature_damping` and Gauss-Newton -- before: `"newton_cg: curvature_damping is only meaningful with hessian='exact'"`; after: `"newton_cg: curvature_damping is only meaningful with hessian='exact' or 'switch'"`. `Problem.local_model`'s and the `fitting` factories' messages are unchanged (`'switch'` is not a model's mode). T7 (`test_options`) pins the new string (`re.escape`, with the before-any-work spy) | the frontend test that regex-matches the old text as a prefix (`TestFrontendExactNewton.test_options_with_no_effect_and_unbuilt_geometries_raise`) passes; `test_curvature.py`'s matching assertion is on `Problem.local_model` (unchanged) |

**The battery's expected mismatches are now two messages** (three entries): the invalid `hessian` (a string and a bool;
judgement call 2) and the `curvature_damping` message above. Rerun on the round-2 tree against `c1a42c24`: numpy 319
entries / 27,101 arrays, exactly those three; nothing else.

| run (round 2, one job at a time) | result |
|---|---|
| touched tests, `-n 3`: `test_optimizers.py`, `test_optimizers_frontend.py`, `test_dispatch.py::TestSwitchNewton`, `::TestNewtonTiming` | **118 passed, 1 skipped** (the slow-tier timing row), 363 subtests (2.6 min). (The first run, before the `tr_radius0` fix, failed the trust-region capped-CG subtest on row 0's boundary stop: the M2 row above) |
| round-2 mutations (`w1/mut/mutate.py r2`) | **5 of 5 caught**; with round 1's 47, **52 of 52** |
| numpy battery vs `c1a42c24` | as above: 3 entries, the two expected messages |
| floor env `tttt`, single process, the same four targets | **118 passed, 1 skipped** (5.5 min; the skip is the slow-tier timing row) |
| module doctests | **241 passed** |

**N4 (the reviewer's round-2 follow-up)**: `test_a_miss_resets` reset only from a count of 1, so a leaky count
(`... else (count // 2 if countable else 0)`, or `max(count - 1, 0)`) passed it. Rewritten: `switch_k=3`, the script
hit, hit, miss, hit, hit, uninformative, hit, hit, hit; counts `[1, 2, 0, 1, 2, 0, 1, 2, 3]`, fired on row 8 only,
`switched_at = 9` (`max_newton=11`; the nine rows' countability asserted at run time). Edited only, per the reviewer's
machine rule; the reviewer runs it and the two leaky-count mutations after its jobs.

Not rerun this round: the full suite and `sphinx -W` (round 1's are green; this round changed tests, one message and
one docstring paragraph's wording and wrapping), left to the reviewer's round-2 checks.

