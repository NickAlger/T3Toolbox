# W2 handoff: the display segment and the frontend of `newton_cg(hessian='switch')` (2026.5.0)

Implementer's record for the session. Base: HEAD `3dc15d96` (W1 committed as `ca13b37c`; the plan
`dev/gn_newton_switch_plan_2026-11-13.md` at draft 4). Nothing is committed. Scope: the plan's §8 row W2, exactly --
the display segment, the frontend keywords with their early checks and forwarding, `least_squares_problem`'s sizing
for `'switch'`, the docstrings of the touched functions; tests T7 (frontend), T10, T11, T12; the `not_yet` exemption
deleted and the `hessian` regex updated (W1's §6). Not pulled forward: docs pages, CHANGELOG, contributor records
(W4); the benchmark (W3).

**Changed:** `t3toolbox/backend/optimizer_display.py`, `t3toolbox/optimizers.py`, `t3toolbox/backend/optimizers.py`
(one docstring sentence, judgement call 8), `tests/backend/test_optimizer_display.py`,
`tests/test_optimizers_frontend.py`. **New:** this file.

## 1. What was built

| where | what |
|---|---|
| `optimizer_display.py` | `_switch_segment(info)`, pure: `None` unless the row carries `switch_count` (a Gauss-Newton-phase row of a switch run), else `sw <e_N/e_GN> ×<count>` in `%.1e`, the ratio `—` on an untracked row (`switch_e_n` `None`), on a non-finite error or at `e_GN = 0`; the row with `switch_fired` appends ` ⇒ exact`. `_format_header` appends it after the `κ σ ε` position (a Gauss-Newton-phase row has no `κ`: its product is `switch_pKp`, not `pKp`) and before `stop: …`. The converged line returns before the segments, so a converged Gauss-Newton-phase row (which carries the count) prints its short line. Docstrings: the module's (one sentence), `format_newton_iter`'s (a switch-row doctest, the firing row), `make_newton_display`'s (a **Switch rows** paragraph). |
| frontend `newton_cg` | Four keyword-only options after `timing`, one blank-line group, default `None`, the backend's order: `switch_tau`, `switch_k`, `switch_cg_forcing_power`, `switch_track`. Before the setup work: `mode = bopt._check_hessian(hessian, 'newton_cg', bopt._NEWTON_HESSIAN_MODES)` then `bopt._check_switch_options(mode, …)` (the lower-cased mode; the original `hessian` string is still what reaches `least_squares_problem` and the backend, as today), then the `timing` type check, then `least_squares_problem`. Forwarding: each switch keyword as given, only when not `None` (a `switching` dict beside `timed`), so a call without them reaches the backend exactly as at `3dc15d96`. The signature comments of `hessian`, `curvature_damping`, `step_control` name `'switch'`. Docstring: the `hessian` paragraph names `'switch'`; the solver-options paragraph says the `switch_*` options are checked before the setup and forwarded only when set; a **Switching from Gauss-Newton to the exact Hessian** section (the rule in words and math, a literal block of the four formulas, what counts and resets, the rounding gate with the x64 advice, each option and its default, the unread-option rules, step control and radius, damping, each call starting in Gauss-Newton, when it does not fire, the cost, the row fields / `stats` keys / the display segment with two sample header lines); an example after the exact-Hessian example (§4). |
| `least_squares_problem` | `bopt._check_hessian(hessian, 'least_squares_problem', bopt._NEWTON_HESSIAN_MODES)`; `'switch'` is then sized as `'exact'` (two lines: `if hessian == 'switch': hessian = 'exact'`), so `_resolve_chunk_size` and `estimate_chunk_size` keep their two modes. Signature comment and docstring name `'switch'`; `_resolve_chunk_size`'s docstring says the caller passes `'exact'` for `'switch'`. |
| `backend/optimizers.py` | One clause in `NewtonInfo`'s switch paragraph: the display prints a Gauss-Newton-phase row's fields as its `sw` segment (a link to `format_newton_iter`). No code. |
| tests (new) | `test_optimizer_display.py::TestSwitchDisplay` (T10, 4 tests): the segment on hand-built rows (ratio, count, mark, every `—` case incl. `e_GN = inf / nan`, `e_N = inf / nan`, `e_GN = 0`); only on Gauss-Newton-phase rows (an exact-phase switch row prints exactly the exact-mode row; no `κ` on a row carrying `switch_pKp`; after the wall clock, before `stop:`; the converged line unchanged); unchanged outside switch mode (the 2026.2.0 texts); end to end on `newton_cg`'s docstring problem (each Gauss-Newton-phase row's ratio and count from its own fields, the mark on the row before `switched_at` only, `κ` on the exact rows; and a `switch_tau=0` run's text minus the segment equals the Gauss-Newton trust-region run's text, wall clocks masked). `test_optimizers_frontend.py::TestFrontendSwitchNewton` (T7 frontend + T11, 5 tests): the keywords' order / kind / defaults beside the backend's; each §2.1 rule raising with the backend's message (`re.escape`) before `least_squares_problem` is called (17 cases, the mode read case-insensitively); forwarding as given and only when set (all four at non-default values with `hessian='Switch'`; each alone, zeros included, alone in the call with its type kept; none forwarded when unset, in each mode); `least_squares_problem` sizing `'switch'` / `'SWITCH'` as `'exact'` on both layers (a spy on `_resolve_chunk_size`; with jax, the estimator itself through `least_squares_problem` and `newton_cg`), an unknown mode raising before any work with the three-mode message; a frontend switch run, ragged (defaults) and uniform (`switch_track='ALL'`), equal to the backend run on `least_squares_problem`'s problem bit for bit (losses, every history field with the switch fields, the cores, the `stats` keys), `step_control` resolved to the trust region, `switch_track` lower-cased, the switch firing, and `verbose=True` printing one segment per Gauss-Newton-phase row and the mark on the switching row only. |
| tests (edited) | The three forced edits, §3: the `not_yet` exemption deleted (`test_newton_cg_keywords_mirror_backend`); the regex in `TestFrontendExactNewton.test_options_with_no_effect_and_unbuilt_geometries_raise` now `"newton_cg: hessian must be 'gauss_newton', 'exact' or 'switch'"`; `test_newton_cg_forwards_timing_only_when_set`'s `list(front)[-2:] == ['cone_seed', 'timing']` now `names[names.index('timing') - 1] == 'cone_seed'` (and its docstring's "the last keyword"). |

## 2. Judgement calls

1. **The segment's text** follows §3.2 literally: `sw 3.0e-03 ×2`, `%.1e` (the house cell), ` ⇒ exact` on the firing
   row. "`—` when `switch_e_n` is None, non-finite, or `e_GN = 0`" is read as "either error non-finite": with only
   `e_N` checked, an infinite `e_GN` would print `0.0e+00`, a ratio that means nothing. Both are finite on every real
   tracked row (`_switch_sample` treats a non-finite error as uninformative); the case is a hand-built row's.
2. **Placement**: after the `κ σ ε` position and before `stop: …` (a run's last row in the Gauss-Newton phase reads
   `… | 0.31s | sw 2.5e-01 ×1 | stop: max_newton`). A row never carries both `κ` and `sw`.
3. **The count after the row**, `×0` included (a reset reads `sw … ×0`; an untracked reset `sw — ×0`), so a reader
   sees the count climb `×1 ×2 ⇒ exact`. No `/k` (the plan's text has none; `switch_k` is not on the row).
4. **The early checks take the lower-cased mode** (`_check_hessian`'s return), while the original `hessian` string
   is forwarded unchanged to `least_squares_problem` and the backend (the existing forwarding test asserts `'Exact'`
   reaches `least_squares_problem` as given). The `timing` type check stays after them.
5. **Forwarded as given**: `switch_track='All'` reaches the backend as `'All'` (the backend lower-cases it, as it
   does `hessian`); numeric types are not converted (W1 passes the power through unconverted).
6. **`least_squares_problem` maps `'switch'` to `'exact'` in place** rather than in `_resolve_chunk_size` (whose
   contract, and `estimate_chunk_size`'s, stays two modes: §2.3 "`estimate_chunk_size` keeps its two modes").
7. **The frontend docstring is a full section** (the brief: "may be detailed"), where §2.3 says one paragraph; it
   sits after "After a rank increase" and before "Timing a run" (each call starts in Gauss-Newton, which a rank
   continuation meets); the example sits after the exact-Hessian example and reuses its problem (`x0`, `opts`, `gn`),
   asserting only structure: the three `stats` values, the rows' `hessian` around `switched_at`, the firing row's
   `switch_fired` / count / `e_N < 0.25·e_GN`, and `newton_steps` under half of Gauss-Newton's (6 against 29 here).
   W4's §4.14 doctest is untouched.
8. **One sentence in `NewtonInfo`'s docstring** (`backend/optimizers.py`, not in §8's W2 file list) pointing at the
   display segment, as P3 did for `timing` (§7 lists `NewtonInfo` among the switch's docstrings). The backend
   `newton_cg` docstring is not touched (its switch paragraph points to `NewtonInfo`).
9. **Messages that change on a non-switch path** (the battery, §5): the frontend `newton_cg`'s invalid-`hessian`
   message and `least_squares_problem`'s now list `'switch'` -- the same `_check_hessian` message W1 changed on the
   backend, now reached from the frontend's two call sites (inherent to accepting the mode). And a frontend call with
   a `switch_*` keyword but no `hessian='switch'` was a `TypeError` (unknown keyword) at `3dc15d96`; it is now the
   house `ValueError` ("`switch_tau set without hessian='switch'`").
10. **T7 (frontend) has no kind-without-`corewise_hessian` case**: unreachable from the frontend (every built-in
    kind implements it); the backend's T7 covers it. The `Problem.local_model` / `fitting` factory rejections are
    W1's tests, unchanged.
11. **Test problems**: the display's end-to-end test and the frontend tests use `newton_cg`'s docstring problem (W1's
    T6, where the switch fires on row 1); the frontend's option tests use a tiny `apply` problem (they raise before
    the setup).

## 3. Where the plan could not be followed literally

- **T12 "the existing tests pass unedited" against §2.1's placement.** §2.1 appends the switch group after `timing`
  in both signatures ("identical"); P3's `test_newton_cg_forwards_timing_only_when_set` asserted
  `list(front)[-2:] == ['cone_seed', 'timing']`, i.e. that `timing` is the last keyword. Both cannot hold. I followed
  §2.1 and edited that one assertion to its purpose (`timing` directly after `cone_seed`, still keyword-only; P3's
  N1 mutation, "`timing` moved before the `*`", stays caught), and its docstring's "the last keyword". The new
  `test_the_keywords` pins the switch group after `timing` on both signatures. With the two edits W1 required (the
  exemption, the regex) that makes three edited existing assertions; no other existing test changed.
- **§2.3's "one paragraph"**: a section instead, per the session's brief (judgement call 7).
- Otherwise nothing blocked.

## 4. The printed switch run through the FRONTEND (for the maintainer)

`newton_cg`'s docstring problem through `optimizers.newton_cg`: probe derivatives of order 1 on (5, 5, 5), a rank-3
target with 1 % noise, the rank-2 fit warm-started by five Gauss-Newton steps; `hessian='switch'`,
`g0_reference='warm_start'`, `verbose=True`; numpy, the maintainer's laptop (script: the session scratchpad
`w2/front_run.py`). The exact model predicted `Δf` 78× better than Gauss-Newton on row 0 (`sw 1.3e-02 ×1`) and 337×
on row 1 (`sw 3.0e-03 ×2 ⇒ exact`), so row 2 is the first exact row (`switched_at = 2`) and shows the `κ σ ε` cells.
6 Newton steps, against 29 for Gauss-Newton with its line search and 25 with the trust region. (In each block below,
the `stats:` line and the line after it are the script's summary, not the display.)

```
iter  0 | obj 2.547e+05  ‖g‖ 9.84e+02 (1.0e+00·g₀) | CG 1/200 tol 4.9e+02 resid 2.3e+02 ✓ | TR Δ 2.51e+00 ✓ ‖Δx‖/‖x‖ 1.9e-02 | Δf -1.07e+03 ρ 0.86 | 0.03s | sw 1.3e-02 ×1
  rel err  rows=mode  cols=order
           ord0     ord1
    m0  3.7e-01  1.9e-01
    m1  2.7e-01  2.8e-01
    m2  2.8e-01  2.6e-01
iter  1 | obj 2.537e+05  ‖g‖ 4.17e+02 (4.2e-01·g₀) | CG 1/200 tol 2.1e+02 resid 1.2e+02 ✓ | TR Δ 2.51e+00 ✓ ‖Δx‖/‖x‖ 8.7e-03 | Δf -2.03e+02 ρ 0.85 | 0.03s | sw 3.0e-03 ×2 ⇒ exact
  rel err  rows=mode  cols=order
           ord0     ord1
    m0  3.6e-01  1.9e-01
    m1  2.8e-01  2.8e-01
    m2  2.8e-01  2.5e-01
iter  2 | obj 2.535e+05  ‖g‖ 2.43e+02 (2.5e-01·g₀) | CG 2/200 tol 6.0e+01 resid 1.9e+01 ✓ | TR Δ 2.51e+00 ✓ ‖Δx‖/‖x‖ 7.2e-03 | Δf -9.75e+01 ρ 1.00 | 0.05s | κ -3.5e-01 σ 5.8e+01 ε 1.3e-04
  rel err  rows=mode  cols=order
           ord0     ord1
    m0  3.6e-01  1.9e-01
    m1  2.8e-01  2.8e-01
    m2  2.8e-01  2.5e-01
iter  3 | obj 2.534e+05  ‖g‖ 1.87e+01 (1.9e-02·g₀) | CG 4/200 tol 3.5e-01 resid 1.9e-01 ✓ | TR Δ 2.51e+00 ✓ ‖Δx‖/‖x‖ 3.9e-04 | Δf -4.24e-01 ρ 1.00 | 0.08s | κ -1.7e-02 σ 5.8e+01 ε 1.3e-04
  rel err  rows=mode  cols=order
           ord0     ord1
    m0  3.6e-01  1.9e-01
    m1  2.8e-01  2.8e-01
    m2  2.8e-01  2.5e-01
iter  4 | obj 2.534e+05  ‖g‖ 1.91e-01 (1.9e-04·g₀) | CG 8/200 tol 3.7e-05 resid 2.8e-05 ✓ | TR Δ 2.51e+00 ✓ ‖Δx‖/‖x‖ 5.7e-06 | Δf -6.38e-05 ρ 1.00 | 0.14s | κ -3.0e-01 σ 5.8e+01 ε 1.3e-04
  rel err  rows=mode  cols=order
           ord0     ord1
    m0  3.6e-01  1.9e-01
    m1  2.8e-01  2.8e-01
    m2  2.8e-01  2.5e-01
iter  5 | obj 2.534e+05  ‖g‖ 2.82e-05 (2.9e-08·g₀) | CG 14/200 tol 8.1e-13 resid 5.3e-13 ✓ | TR Δ 2.51e+00 ✓ ‖Δx‖/‖x‖ 6.4e-10 | Δf -2.91e-11 ρ 26.73 | 0.23s | κ -1.4e-02 σ 5.8e+01 ε 1.3e-04
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
Gauss-Newton line search: 29 Newton steps; Gauss-Newton trust region: 25
```

The same problem from the random rank-2 start, `switch_track='all'` (headers only): three misses (the exact model
3.5–9.6× worse in the transient: `sw 9.6e+00 ×0`, …), then two hits.

```
iter  0 | obj 3.707e+06  ‖g‖ 1.85e+04 (1.0e+00·g₀) | CG 1/200 tol 9.3e+03 resid 5.2e+03 ∂ | TR Δ 5.13e+01 ✓ ‖Δx‖/‖x‖ 7.4e+00 | Δf -3.62e+05 ρ 0.76 | 0.03s | sw 9.6e+00 ×0
iter  1 | obj 3.345e+06  ‖g‖ 3.17e+04 (1.7e+00·g₀) | CG 1/200 tol 1.6e+04 resid 8.1e+03 ✓ | TR Δ 1.03e+02 ✓ ‖Δx‖/‖x‖ 1.8e+00 | Δf -1.27e+06 ρ 1.07 | 0.03s | sw 8.1e+00 ×0
iter  2 | obj 2.080e+06  ‖g‖ 3.17e+04 (1.7e+00·g₀) | CG 1/200 tol 1.6e+04 resid 7.4e+03 ✓ | TR Δ 1.03e+02 ✓ ‖Δx‖/‖x‖ 9.9e-01 | Δf -1.52e+06 ρ 1.15 | 0.02s | sw 3.5e+00 ×0
iter  3 | obj 5.621e+05  ‖g‖ 1.37e+04 (7.4e-01·g₀) | CG 1/200 tol 6.9e+03 resid 3.5e+03 ✓ | TR Δ 1.03e+02 ✓ ‖Δx‖/‖x‖ 3.4e-01 | Δf -2.89e+05 ρ 1.07 | 0.03s | sw 8.1e-02 ×1
iter  4 | obj 2.727e+05  ‖g‖ 3.17e+03 (1.7e-01·g₀) | CG 1/200 tol 1.3e+03 resid 1.2e+03 ✓ | TR Δ 1.03e+02 ✓ ‖Δx‖/‖x‖ 8.5e-02 | Δf -1.64e+04 ρ 0.96 | 0.03s | sw 8.9e-02 ×2 ⇒ exact
iter  5 | obj 2.562e+05  ‖g‖ 1.48e+03 (8.0e-02·g₀) | CG 3/200 tol 1.2e+02 resid 5.7e+01 ✓ | TR Δ 1.03e+02 ✓ ‖Δx‖/‖x‖ 3.3e-02 | Δf -2.84e+03 ρ 1.00 | 0.06s | κ -2.1e-01 σ 5.8e+01 ε 1.3e-04
iter  6 | obj 2.534e+05  ‖g‖ 6.22e+01 (3.4e-03·g₀) | CG 6/200 tol 2.1e-01 resid 7.3e-02 ✓ | TR Δ 1.03e+02 ✓ ‖Δx‖/‖x‖ 1.9e-03 | Δf -7.16e+00 ρ 1.00 | 0.12s | κ -1.7e-01 σ 5.8e+01 ε 1.3e-04
iter  7 | obj 2.534e+05  ‖g‖ 9.10e-02 (4.9e-06·g₀) | CG 11/200 tol 4.5e-07 resid 2.0e-07 ✓ | TR Δ 1.03e+02 ✓ ‖Δx‖/‖x‖ 2.6e-06 | Δf -1.37e-05 ρ 1.00 | 0.19s | κ -1.1e-01 σ 5.8e+01 ε 1.3e-04
iter  8 | obj 2.534e+05  ‖g‖ 2.32e-07 (1.2e-11·g₀) | converged (‖g‖ ≤ gtol·‖g₀‖, g₀ = warm start)
stats: newton_steps 8, newton_accepted 8, stop_reason 'gradient', hessian 'switch', switch_track 'all', switched_at 5
```

Reset rows with no tracking product print `sw — ×0`. Natural rejections are rare in a Gauss-Newton phase, so this
one is the backend on the test fixture `_OffsetProblem` (the objective shifted by 5 % of `f0`: three accepted
steps, then rejected trials), with the default `'countable'` and then `'all'` (which reports `e_N` on the rejected
rows, `sw 1.2e+00 ×0`, and moves no count):

```
======== switch_track=None
iter  0 | obj 1.857e+07  ‖g‖ 7.73e+04 (1.3e+00·g₀) | CG 1/200 tol 3.9e+04 resid 2.0e+04 ✓ | TR Δ 1.27e+02 ✓ ‖Δx‖/‖x‖ 2.1e+01 | Δf -4.30e+06 ρ 0.87 | 0.05s | sw 1.8e+02 ×0
iter  1 | obj 1.334e+07  ‖g‖ 1.21e+05 (2.0e+00·g₀) | CG 1/200 tol 6.1e+04 resid 4.3e+04 ∂ | TR Δ 1.27e+02 ✓ ‖Δx‖/‖x‖ 1.1e+00 | Δf -9.29e+06 ρ 0.92 | 0.03s | sw 1.6e+00 ×0
iter  2 | obj 3.122e+06  ‖g‖ 4.79e+04 (7.8e-01·g₀) | CG 1/200 tol 2.4e+04 resid 1.2e+04 ✓ | TR Δ 2.54e+02 ✓ ‖Δx‖/‖x‖ 4.9e-01 | Δf -1.24e+06 ρ 0.59 | 0.03s | sw 1.1e+00 ×0
iter  3 | obj 9.523e+05  ‖g‖ 1.64e+04 (2.7e-01·g₀) | CG 1/200 tol 8.2e+03 resid 4.7e+03 ✓ | TR Δ 2.54e+02 ✗ ‖Δx‖/‖x‖ 1.2e-01 | Δf +6.19e+05 ρ -2.47 | 0.02s | sw — ×0
iter  4 | obj 9.523e+05  ‖g‖ 1.64e+04 (2.7e-01·g₀) | CG 1/200 tol 8.2e+03 resid 1.2e+04 ∂ | TR Δ 7.65e+00 ✗ ‖Δx‖/‖x‖ 3.1e-02 | Δf +8.12e+05 ρ -7.41 | 0.01s | sw — ×0
iter  5 | obj 9.523e+05  ‖g‖ 1.64e+04 (2.7e-01·g₀) | CG 1/200 tol 8.2e+03 resid 1.5e+04 ∂ | TR Δ 1.91e+00 ✗ ‖Δx‖/‖x‖ 7.7e-03 | Δf +8.98e+05 ρ -29.59 | 0.01s | sw — ×0
iter  6 | obj 9.523e+05  ‖g‖ 1.64e+04 (2.7e-01·g₀) | CG 1/200 tol 8.2e+03 resid 1.6e+04 ∂ | TR Δ 4.78e-01 ✗ ‖Δx‖/‖x‖ 1.9e-03 | Δf +9.21e+05 ρ -118.52 | 0.01s | sw — ×0 | stop: max_newton
stats: stop_reason 'max_newton', switched_at None
======== switch_track='all'
iter  0 | obj 1.857e+07  ‖g‖ 7.73e+04 (1.3e+00·g₀) | CG 1/200 tol 3.9e+04 resid 2.0e+04 ✓ | TR Δ 1.27e+02 ✓ ‖Δx‖/‖x‖ 2.1e+01 | Δf -4.30e+06 ρ 0.87 | 0.04s | sw 1.8e+02 ×0
iter  1 | obj 1.334e+07  ‖g‖ 1.21e+05 (2.0e+00·g₀) | CG 1/200 tol 6.1e+04 resid 4.3e+04 ∂ | TR Δ 1.27e+02 ✓ ‖Δx‖/‖x‖ 1.1e+00 | Δf -9.29e+06 ρ 0.92 | 0.03s | sw 1.6e+00 ×0
iter  2 | obj 3.122e+06  ‖g‖ 4.79e+04 (7.8e-01·g₀) | CG 1/200 tol 2.4e+04 resid 1.2e+04 ✓ | TR Δ 2.54e+02 ✓ ‖Δx‖/‖x‖ 4.9e-01 | Δf -1.24e+06 ρ 0.59 | 0.03s | sw 1.1e+00 ×0
iter  3 | obj 9.523e+05  ‖g‖ 1.64e+04 (2.7e-01·g₀) | CG 1/200 tol 8.2e+03 resid 4.7e+03 ✓ | TR Δ 2.54e+02 ✗ ‖Δx‖/‖x‖ 1.2e-01 | Δf +6.19e+05 ρ -2.47 | 0.03s | sw 1.2e+00 ×0
iter  4 | obj 9.523e+05  ‖g‖ 1.64e+04 (2.7e-01·g₀) | CG 1/200 tol 8.2e+03 resid 1.2e+04 ∂ | TR Δ 7.65e+00 ✗ ‖Δx‖/‖x‖ 3.1e-02 | Δf +8.12e+05 ρ -7.41 | 0.02s | sw 1.0e+00 ×0
iter  5 | obj 9.523e+05  ‖g‖ 1.64e+04 (2.7e-01·g₀) | CG 1/200 tol 8.2e+03 resid 1.5e+04 ∂ | TR Δ 1.91e+00 ✗ ‖Δx‖/‖x‖ 7.7e-03 | Δf +8.98e+05 ρ -29.59 | 0.02s | sw 1.0e+00 ×0
iter  6 | obj 9.523e+05  ‖g‖ 1.64e+04 (2.7e-01·g₀) | CG 1/200 tol 8.2e+03 resid 1.6e+04 ∂ | TR Δ 4.78e-01 ✗ ‖Δx‖/‖x‖ 1.9e-03 | Δf +9.21e+05 ρ -118.52 | 0.02s | sw 1.0e+00 ×0 | stop: max_newton
stats: stop_reason 'max_newton', switched_at None
```

## 5. Verification

One job at a time, current env `t3toolbox` (py3.11, numpy 2.4.6, jax 0.10.2) unless stated; scripts and logs in the
session scratchpad `w2/`.

| check | result |
|---|---|
| **Non-breaking, bitwise against `3dc15d96`** (a `git worktree` of the commit, since removed, against a byte copy of the W2 tree; `w2/battery.py` = the W1 reviewer's battery -- itself the P3 / P2 reviewers' -- with the frontend signature entry filtering the four new names, plus a W2 section; `w2/compare.py`: container types, key order, dtypes, shapes, float bits). The inherited part: every `NewtonInfo` row with its model parts, `x`, the whole `stats`, the display's records and texts (wall clock masked) for numpy ragged / uniform, unshared / shared, probe / probe-derivative runs in every non-switch mode and option group, timed runs, `local_model`, `padded_start`, `gradient_descent`, hand-built `format_newton_iter` rows at three layouts, the frontend `newton_cg` on both layers (verbose, validation, regularizer, weight, exact LS / TR, padded + cone, exhausted stop / accept, converged at row 0, `max_newton=0`, custom callback, warm overrides, explicit `timing=False`), corewise and shared, and 50+ structural-error messages (backend and frontend). The W2 section: **switch runs** on the backend with the display (defaults, line search, `'all'`, `tau=0`, `k=0`, `k=3` with a power, corewise, regularized, `_OffsetProblem` rejections under both tracks, exhausted `'accept'`, ragged / uniform × unshared / shared, from a padded start and not), compared WITH every switch field and with the printed text's `sw` segment stripped (the reference prints none); **cross-tree frontend runs**: the new frontend `newton_cg(hessian='switch', verbose=True)` against the reference's backend switch run on `least_squares_problem(hessian='exact')`'s problem with the frontend's display and diagnostics rebuilt by hand -- ragged (warm / random start, `'ALL'` with validation, line search with `k=3` and a power, `tau=0`, `k=0`), corewise, uniform with `chunk_size='auto'` (the estimator, sized `'exact'`) and with `None` plus validation: `x`, `stats` with `diagnostics`, the stripped text. jax x64: the inherited `use_jit` / eager runs plus `use_jit` switch runs on both layers (TR; `'all'` + line search) | **numpy 374 entries, 36,622 arrays: 7 mismatches, all expected** (judgement call 9): the frontend `newton_cg`'s invalid-`hessian` message (ragged, uniform) and `least_squares_problem`'s now list `'switch'`; `least_squares_problem(hessian='switch')` and `newton_cg(hessian='switch')` on the frontend raised at `3dc15d96` and now run; a frontend `switch_tau=` without the mode was a `TypeError` (unknown keyword), now the house `ValueError`. **jax 35 entries, 3,131 arrays: 0 mismatches.** Every switch run's stripped text, rows (switch fields included), `x` and `stats` are bitwise the reference's; every frontend switch run is bitwise the reference backend run. Side check on the new tree: in each of the 23 backend switch runs (19 numpy, 4 jax) the number of `sw` segments equals the number of Gauss-Newton-phase step rows, and `⇒ exact` appears once iff the switch fired (`switched_at` > 0); the 8 frontend runs print one mark each, none under `tau=0` or `k=0` (which prints no segment). A planted one-character text change is caught |
| **Mutation testing** of the new tests (`w2/mutate.py`: one mutation at a time on a scratch copy of the tree, the targeted tests rerun: `TestSwitchDisplay`; `TestFrontendSwitchNewton` + the mirror and timing-forwarding tests) | **29 of 29 caught**. Display (14): the segment never appended; after the stop reason; before the wall clock; the ratio inverted; `%.2e`; the finite guard dropped (`e_GN = inf` printed `0.0e+00`); the `e_GN = 0` guard dropped; the mark on every switch row; never; the count + 1; the count missing; the segment keyed on `switch_fired` (on exact-phase rows); on every row; the ratio as `e_N` alone. Frontend (15): no early switch check; the check on the raw (unlowered) mode; `newton_cg` checking two modes; the check after the setup; forwarded always; a zero not forwarded; `tau` / `k` swapped; `switch_track` lowered; not forwarded; `least_squares_problem` checking two modes; sizing `'switch'` as itself; as `'gauss_newton'`; a frontend default of `0.25`; the group before `timing`; the group positional. Each failed on the intended test (log: `w2/mutate.log`) |
| new tests | `TestSwitchDisplay` **4 passed** (15 subtests, 1.3 s); `TestFrontendSwitchNewton` with the mirror / forwarding / timing / regex tests **9 passed** (73 subtests, 4.9 s) |
| full suite, `pytest tests/ -q -n 3` | **1,407 passed, 5 skipped**, 49,057 subtests (22 min); W1's 1,398 + the 9 new tests; the one warning is the pre-existing adam-on-manifold one |
| module doctests, `pytest --doctest-modules t3toolbox -n 3` | **241 passed** (the new examples are in existing docstrings) |
| `python -m sphinx -W --keep-going -b html docs <scratch>` | **exit 0, no warning** (15 min); the frontend section's literal blocks, the display example and the `NewtonInfo` link render |
| floor env `tttt` (py3.9, numpy 1.22.4, jax 0.4.30), single process: `test_optimizer_display.py`, `test_optimizers_frontend.py`; the doctests of `optimizers.py`, `backend/optimizer_display.py`, `backend/optimizers.py` | **78 passed** (1.5 min; the sizing test's jax branch included); **6 passed** |

Not run: the `docs/*.md` page doctests (no page changed; every non-switch run and text is bitwise the reference's,
above) and the slow tier (release gate).

## 6. For the session and the maintainer

- The display's look: `sw <ratio> ×<count>`, ` ⇒ exact`, `—` (calls 1–3); printed in §4.
- The three edited existing assertions (§3), one of them forced by §2.1 against T12.
- The message changes on non-switch paths (call 9).
- The frontend docstring section (call 7) is the longest text this slice adds; W4's §4.14 can lift from it.

## 7. Round 2 (review `W2_review.md`: REVISE, two MINOR, four NIT)

Docstrings, tests and whitespace only; no code path changed, so round 1's battery stands by construction.

| finding | change | checked |
|---|---|---|
| **M1** the frontend's timing phases | `curvature`: "with ``hessian='exact'`` or ``'switch'`` the curvature data"; a `tracking` bullet after the trials ("``hessian='switch'``: a tracked Gauss-Newton-phase row's curvature product (``switch_pKp``); ``None`` on every other row"); and `gradient`'s "(in both modes)" now "(in every mode)", the same staleness one line down | the frontend doctests pass |
| **M2** G-3 unpinned | `TestSwitchDisplay.test_the_segment` gains a row with count 2 and `switch_fired=False` (`sw 2.5e-01 ×2`, no mark) and one with count 1 and `switch_fired=True` (`… ×1 ⇒ exact`, `switch_k=1`) | the reviewer's mutation `" ⇒ exact" if count >= 2 else ""` (my D15): **caught** |
| **N1** a vacuous type check | the alone-values are `np.float64(0.0)`, `np.int64(0)`, `np.float32(0.875)`, `'all'` | `float()` on the forwarded power (F16) and `int()` on `switch_k` (F17): **caught** |
| **N2** blank lines | two before `def _switch_problem` and before `if __name__` | read |
| **N3** "ends each Gauss-Newton-phase header" | the module docstring, `make_newton_display`'s **Switch rows** and the frontend's *What it reports* now say each Gauss-Newton-phase **step** row's header carries the segment, before a last row's stop reason (the module docstring also lists the converged line among what prints as before); `_switch_segment`'s docstring says the converged row carries `switch_count` but its short line never reaches the segments | read |
| **N4** the `NewtonInfo` clause | "(the display prints them as a Gauss-Newton-phase row's ``sw`` segment: :py:func:`…format_newton_iter`)" | read |

Also: two test lines over 120 columns re-wrapped (`_switch_problem`'s docstring, a comment).

| run (round 2) | result |
|---|---|
| touched tests (`TestSwitchDisplay`, `TestFrontendSwitchNewton`, the mirror / timing-forwarding / regex tests), `-n 3` | **12 passed**, 88 subtests |
| doctests of `optimizers.py`, `backend/optimizer_display.py`, `backend/optimizers.py` | **6 passed** |
| mutations (`w2/mutate.py`, round 1's 29 plus D15, F16, F17) | **32 of 32 caught** (`w2/mutate_r2.log`) |
| floor env `tttt`, the same touched tests | **12 passed** |

Not rerun this round: the full suite and `sphinx -W` (docstring bullets and sentences, tests; round 1's are green).
