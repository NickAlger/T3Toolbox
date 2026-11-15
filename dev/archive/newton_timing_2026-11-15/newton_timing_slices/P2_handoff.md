# P2 handoff: the `newton_cg` side of phase timing (Newton timing, 2026.5.0)

Implementer's record for the session. Base: HEAD `89ac203f` (the plan at draft 4, P1 committed in `cbcdc1d6`).
Nothing is committed. Scope: plan §7 row P2, exactly --

- `PhaseSeconds`, `NewtonTiming`, `NewtonInfo.timing`, `_NEWTON_SCALAR_FIELDS`;
- `_CompileSeconds`, `_PhaseClock`;
- `_local_model_phases`, with `Problem.local_model` delegating;
- `_padded_start`; `_cone_reference(timing)` and `gradient_descent`'s call site;
- the loop's laps; `stats['timing']`;
- tests T1-T7 at P2's level; T10.

Not pulled forward (P3): the display (`format_newton_iter`'s `time` line, `make_newton_display`'s closing table), the
frontend keyword and docstring, docs pages (§4.13, `aligned_padded_frame.md`, `scan_body_principles.md`), the
figures, CHANGELOG. `optimizer_display.py` and the frontend `optimizers.py` are untouched.

**Changed:** `t3toolbox/backend/optimizers.py`, `tests/backend/test_optimizers.py`, `tests/backend/test_padded_start.py`,
`tests/test_dispatch.py`, `tests/test_optimizers_frontend.py` (one temporary line, §3). **New:** this file.
(`git diff -w` shows the logic: `newton_cg`'s body after the listener's registration is re-indented by one level
under `try:`, judgement call 5.)

## 1. What was built

| where | what |
|---|---|
| records | `PhaseSeconds` (§3.1's fourteen fields in run order, `tracking` after `trial_objective`, always `None` here) and `NewtonTiming` (window, seconds, compile_seconds, trials, accepted_retraction, then the six row-0 one-offs defaulting to `None`): frozen dataclasses of floats / tuples of floats, in `__all__`, with docstrings stating what each phase covers. `NewtonInfo.timing: Optional[NewtonTiming] = None`, the last field; `_NEWTON_SCALAR_FIELDS` excludes it, so history keys and their order are 2026.4.0's. |
| `_CompileSeconds` | The `jax.monitoring` duration listener: `total`, `__call__(event, duration, **kwargs)` adding exactly the three `/jax/core/compile/*_duration` events, `register(x0)` (a registered listener on a jax run, `tree_contains_jax(x0)` after `_prepare_jit_inputs`, whose jax has `unregister_event_duration_listener`; else `None`, so the floor's 0.4.30 and numpy runs report `None`) and `unregister()`. |
| `_PhaseClock` | `(listener, t_entry)`; `mark()`, `lap(phase, outputs) -> float` (waits on `outputs` with `common.tree_block_until_ready` first, then reads the clock and the listener), `close(t) -> (seconds, compile)` (ends a *stretch*: its laps by phase plus `remainder`), `row(t, trials, accepted_retraction, **one_offs) -> NewtonTiming`. Module helpers `_mark(clock)` / `_lap(clock, phase, outputs)` return at once for `clock=None`; `_phase_seconds(laps)` builds the record (an unknown phase is a `TypeError`); `_timing_stats(...)` builds `stats['timing']`. |
| `_local_model_phases` | `Problem.local_model`'s body moved verbatim (`self` -> `problem`), with `_mark` after the validation and laps `frame`, `sweep`, `companion`, `residual`, and in exact mode `gradient` (the misfit transpose, Q-1 (a)) then `curvature`. `Problem.local_model` is now `return _local_model_phases(self, ..., None)`; its docstring and doctest stay on the method. |
| alignment seam | `padded_start` (public, unchanged signature and return) = member check, option validation, `_padded_start(..., use_jit, False)[:2]`. `_padded_start(problem, x0, x_old, options, use_jit, timing)` returns `(frame, movements, pass_seconds)`. Both it and `_cone_reference` call the member through `_aligned_padded_frame(member, x0, x_old, hint, options, use_jit, timing)`, which passes `timing=True` only when `timing` and `inspect.signature(member).parameters['timing']` exists as a positional-or-keyword or keyword-only parameter (a `**kwargs` catch-all is not evidence; an unreadable signature counts as absent); otherwise the untimed call and `None`. `_cone_reference(..., timing)` returns `(value, movements, pass_seconds)`; `gradient_descent` passes `False` and drops the third. `_alignment_member(geom)` holds the member lookup and its error, shared by `padded_start` and `_padded_start`. |
| `newton_cg(..., timing=False)` | Keyword-only, after `cone_seed`. With `timing`: the entry reading `t_entry` (first statement); after `_prepare_jit_inputs`, `tree_block_until_ready((x0, problem))` and the clock with its listener; a `try` / `finally` that unregisters the listener (also when a callback or the padded start raises). Laps: `alignment` around `_padded_start`; the pre-loop stretch closed at the loop's own `t_prev` (its remainder is `setup`); per row `previous_callback` (top of the next iteration), the model (split, or an override's whole build in `frame`), `gradient` (misfit, regularization, gradient, `‖g‖`, lift diagnostics), `reference` (row 0), `forms` (the first trust-region radius, before CG), `cg` (waits on CG's five outputs), `forms` (slope, descent fallback, `pᵀHp`, `‖p‖`; waits on `p`), each trial's `trial_retraction` (waits on the trial point) and `trial_objective`; each row closed at the loop's own `t_now` by `clock.row`, so `window` is `t_now - t_prev`, the same float as `wall_time`. The converged row is closed at its own reading (only with timing). The run's exit reading closes the last callback's stretch and gives `total`. Without timing: no reading, no synchronization; `problem.local_model` is called exactly as today. |
| `stats['timing']` | `{'total', 'setup', 'alignment', 'alignment_passes', 'reference_passes', 'last_callback', 'phases', 'compile', 'cg_per_iteration', 'rows'}` as §3.2; `phases` sums each `PhaseSeconds` field over the rows (`None` as 0); `compile` the same for the compile column plus `setup`, `alignment`, `last_callback` (`None` without a listener); `rows` is `[dc.asdict(record) for each row]`. Added after the other keys (`start_frame`, `start_movements`, `cone_movements`). |
| docstrings | `newton_cg`: the `timing` signature comment, one sentence in the `stats` paragraph and a **Timing** paragraph (what is timed, `stats['timing']`'s keys and identity, the synchronization and async caveats, the process-global listener, the override rule, the member rule). `NewtonInfo`: `timing` is not a history field. `_newton_scalar_record`: drops `timing`. The `Geometry` protocol's member rule was written by P1 and is what `_aligned_padded_frame` implements. |

## 2. Judgement calls

1. **The clock's interface differs from §3.1's sketch in three private details.** (a) `close(t)` is added: one
   mechanism ends every stretch (the pre-loop setup, each row's window, the last callback), so `setup`,
   `last_callback` and each row's `remainder` are the same computation and the `total` identity holds by tiling.
   (b) `mark()` takes no `t`: a stretch's start is set by `close(t)` (the plan's `mark(t)` "at the loop's own t_now"
   is exactly that). (c) `row(t, ...)` takes the row's end reading, not `window`, so `window = t - start` is
   computed from the same two readings as `wall_time` and is the same float (T2 asserts `==`). The constructor
   takes the entry reading.
2. **`_mark` / `_lap` are module helpers that accept `clock=None`**, instead of an `if clock is not None:` around each
   of the eighteen laps; with `None` they return at once (no reading, no synchronization), which is what keeps
   `Problem.local_model`'s call (`clock=None`) the same operations in the same order.
3. **Where the listener is registered.** §2.3 says "registered once at entry" and "on a jax run
   (`tree_contains_jax(x0)` after `_prepare_jit_inputs`)". The jax-run test needs the converted `x0`, so the listener
   is registered right after `_prepare_jit_inputs` (and its synchronization). `setup_compile` therefore covers the
   rest of the setup (`_prepare_jit_inputs` converts arrays and compiles nothing). The plan's test is applied
   literally: `x0` only (numpy `x0` with jax data reports `None`, "not measured").
4. **`previous_callback` is lapped at the top of the next iteration.** It covers the previous row's record
   (`NewtonInfo`, `clock.row`), its callback and history append, plus the loop's stop test and the accepted
   bookkeeping (`x, lm = x_trial, None`: microseconds). The last row's `last_callback` is the same stretch closed by
   the exit reading, so the two mean the same.
5. **`try` / `finally` around the body after registration.** The listener must be unregistered when anything after it
   raises (a callback, the padded start), so `newton_cg`'s body from `extra = {}` to the loop's end is indented one
   level under `try:`. Comment columns were kept where the code allowed (two lines with a staircase comment moved
   by the indentation); the `return` follows the `finally`. `git diff -w` shows only the changed lines.
6. **The first trust-region radius is a second `forms` lap before `cg`** (W-7 puts `_initial_radius` in `forms`; it runs
   before CG in code order). Laps sum by phase, so the record is unchanged in shape.
7. **What each lap waits on.** The model's parts by object (`frame`, `sweep`, `geom_aux`, `residual`, the misfit
   gradient, the `CurvatureState`; an override's whole `LocalModel`, a registered pytree); CG's five outputs; the
   step `p` at the end of `forms`; each trial point. Laps ending in `float(...)` pass nothing (§2.4). The setup waits
   on `(x0, problem)` (`Problem` is a registered pytree), and the alignment on `(frame, movements)`.
8. **`trials` is the line search's loop count `bt + 1`** (= `ls_steps + 1`, and 40 when exhausted), 1 under the trust
   region, `None` on the converged row; `accepted_retraction` is the last trial's lap when the step is taken (so
   `'accept'` after an exhausted search reports it), `None` when rejected.
9. **Compile column.** `compile_seconds` mirrors `seconds`' `None` pattern (a phase that ran has a float, 0.0 when
   nothing compiled); its `remainder` is the window's listener change minus the phases'. `stats['timing']['compile']
   ['alignment']` is `None` without `padded_from`. `cg_per_iteration` divides by the history's `cg_iters`.
10. **`padded_start` keeps its error precedence.** Its member check stays before the option validation
    (`_alignment_member`), and `_padded_start` repeats it for `newton_cg`'s path, so a call that is wrong in both
    ways raises today's message. `padded_start` now returns the slice `(frame, movements)` of the private triple
    rather than the member's own tuple object (same elements).
11. **The converged row's window is read only with timing** (W-4); the untimed converged row reads no clock, as today.
    (The mutation battery showed my first version's `t_prev = t_now` there was dead code; removed.)
12. **Backend docstrings are P2's**, for what P2 builds (the records, `NewtonInfo.timing`, `newton_cg`'s keyword):
    a keyword without its docstring would ship undocumented between slices. §6's pages, the frontend docstring and the
    CHANGELOG stay P3's. `PhaseSeconds.tracking` is documented as reserved for the switch.
13. **Test placement and cost.** numpy rows in `tests/backend/test_optimizers.py` (`TestNewtonTiming`, with the shared
    checker `check_timing` and `untimed`) and `tests/backend/test_padded_start.py` (`TestNewtonTimingAtTheSeam`,
    importing `test_optimizers` as a module: no re-collection, P1's precedent); jax rows in `tests/test_dispatch.py`
    (`TestNewtonTiming`, x64). Eager jax is the plain run only: eager `padded_from` / `'cone'` compile per operation
    (+40 s, measured); their timed paths are covered on numpy and under `use_jit`. Under `use_jit`, ragged plain and
    uniform `padded_from` (65 s for the jax class, against 235 s for padded + cone on both layers). T5's "a structure no
    other test uses" is realized by clearing jax's program caches (`_release_jax_programs`) before the run.
14. **Two tests go beyond §5's rows**, because the rows' criteria (sums and `None` patterns) cannot see a phase
    attributed to the wrong lap or a lap that does not wait: `test_each_phase_is_its_own_work` (records the run's
    calls of each part and the clock's marks / laps / closes in order, and attributes each call to the lap that ends
    after it; exact sets per part, for exact + trust region, Gauss-Newton + line search and an override) and
    `test_each_phase_waits_on_its_outputs` (a mocked `tree_block_until_ready`: the setup's `(x0, problem)`, the
    model's parts by identity, CG's five outputs, the step, the trial point; nothing when off). Also
    `test_untimed_calls_ask_for_nothing` (only a timed `newton_cg` passes `timing` to the member; `padded_start`,
    `gradient_descent` and an untimed `newton_cg` call it as 2026.4.0 did).
15. **The review stop's "§4.13 example".** §4.13 does not exist until P3, so §4 below uses `newton_cg`'s own docstring
    problem (probe derivatives, d = 3, noisy), from a rank-2 fit zero-padded to rank 3 with `padded_from` and
    `'cone'`, exact + trust region, so every phase and one-off appears. P3 may pick another example for §4.13.

## 3. Where the plan could not be followed literally

**One conflict the plan did not foresee (for the session to decide).** `tests/test_optimizers_frontend.py::
test_newton_cg_keywords_mirror_backend` asserts that every backend `newton_cg` option is an explicit frontend keyword.
P2 adds the backend `timing`; §7 gives the frontend keyword to P3. So the full suite failed on that one subtest
(`name='timing'`). I did not pull the frontend keyword forward; I added one temporary line to that test,
`not_yet = ('timing',)` (skipped in the mirror loop, commented "TEMPORARY: the timing plan's slice P3 adds the
frontend keyword and removes this"). P3 must delete it; the alternative is moving the frontend keyword (forwarded only
when `True`) into P2.

Otherwise nothing blocked. Readings recorded above: the clock's private interface (call 1), the listener's
registration point (call 3), the `forms` lap before CG (call 6), the backend docstrings (call 12), the example
(call 15).

## 4. The review stop: a printed timed run

The problem (`newton_cg`'s docstring example, judgement call 15): probe derivatives, order 1, d = 3, shape (5, 5, 5),
40 probes, 1 % noise; `x_old` = 5 Gauss-Newton steps at rank (2, 2, 2) from a random start, zero-padded to rank
(3, 3, 3); `newton_cg(problem, x0, max_newton=40, hessian='exact', cg_forcing_power=1.0, padded_from=x_old,
g0_reference='cone', timing=True, callback=make_newton_display(problem)[0])`, numpy, the maintainer's laptop. The
display is today's (`format_newton_iter` per row: P3 adds the `time` line); then `stats['timing']`, its floats rounded
to 3 significant digits for print (script: the session scratchpad `p2/example.py`).

```
iter  0 | obj 2.547e+05  ‖g‖ 1.26e+04 (2.2e-01·g₀) | CG 1/200 tol 2.8e+03 resid 4.9e+03 ∂ | TR Δ 3.46e+01 ✓ ‖Δx‖/‖x‖ 2.6e-01 | Δf -2.19e+05 ρ 1.00 | 0.16s | κ -1.2e-03 σ 0.0e+00 ε 1.3e-04
  rel err  rows=mode  cols=order
           ord0     ord1
    m0  3.7e-01  1.9e-01
    m1  2.7e-01  2.8e-01
    m2  2.8e-01  2.6e-01
iter  1 | obj 3.617e+04  ‖g‖ 4.79e+03 (8.5e-02·g₀) | CG 3/200 tol 4.1e+02 resid 2.3e+02 ✓ | TR Δ 6.92e+01 ✓ ‖Δx‖/‖x‖ 1.2e-01 | Δf -3.57e+04 ρ 1.01 | 0.09s | κ +3.6e-03 σ 1.3e+01 ε 1.4e-04
  rel err  rows=mode  cols=order
           ord0     ord1
    m0  1.1e-01  8.9e-02
    m1  1.2e-01  1.1e-01
    m2  1.1e-01  8.3e-02
iter  2 | obj 4.622e+02  ‖g‖ 5.23e+02 (9.3e-03·g₀) | CG 5/200 tol 4.8e+00 resid 2.5e+00 ✓ | TR Δ 6.92e+01 ✓ ‖Δx‖/‖x‖ 1.3e-02 | Δf -4.62e+02 ρ 1.01 | 0.11s | κ +1.2e-02 σ 1.0e+01 ε 1.4e-04
  rel err  rows=mode  cols=order
           ord0     ord1
    m0  1.4e-02  9.8e-03
    m1  1.3e-02  1.1e-02
    m2  1.2e-02  1.0e-02
iter  3 | obj 3.746e-01  ‖g‖ 1.47e+01 (2.6e-04·g₀) | CG 8/200 tol 3.8e-03 resid 3.0e-03 ✓ | TR Δ 6.92e+01 ✓ ‖Δx‖/‖x‖ 3.3e-04 | Δf -3.14e-01 ρ 1.00 | 0.19s | κ +6.0e-04 σ 1.0e+01 ε 1.4e-04
  rel err  rows=mode  cols=order
           ord0     ord1
    m0  4.2e-04  2.4e-04
    m1  3.7e-04  3.2e-04
    m2  3.7e-04  3.0e-04
iter  4 | obj 6.024e-02  ‖g‖ 1.58e-02 (2.8e-07·g₀) | CG 14/200 tol 4.4e-09 resid 3.0e-09 ✓ | TR Δ 6.92e+01 ✓ ‖Δx‖/‖x‖ 3.6e-07 | Δf -3.76e-07 ρ 1.00 | 0.27s | κ +4.1e-05 σ 1.0e+01 ε 1.4e-04
  rel err  rows=mode  cols=order
           ord0     ord1
    m0  1.6e-04  9.5e-05
    m1  1.9e-04  1.2e-04
    m2  1.8e-04  1.0e-04
iter  5 | obj 6.024e-02  ‖g‖ 2.65e-08 (4.7e-13·g₀) | converged (‖g‖ ≤ gtol·‖g₀‖)
  rel err  rows=mode  cols=order
           ord0     ord1
    m0  1.6e-04  9.5e-05
    m1  1.9e-04  1.2e-04
    m2  1.8e-04  1.0e-04

stats['stop_reason'] = gradient ; rows: 6
stats['timing'] (floats rounded to 3 significant digits for print):
{'total': 0.913,
 'setup': 0.00013,
 'alignment': 0.0796,
 'alignment_passes': (0.0185, 0.0124, 0.0136, 0.0137, 0.0137),
 'reference_passes': (0.0113, 0.0137, 0.0119, 0.0106, 0.0108, 0.0119, 0.0102, 0.0101, 0.00973),
 'last_callback': 0.000204,
 'phases': {'previous_callback': 0.00137,
            'frame': 0.00286,
            'sweep': 0.00663,
            'companion': 2.84e-05,
            'residual': 0.00688,
            'curvature': 0.00895,
            'gradient': 0.0193,
            'reference': 0.111,
            'cg': 0.585,
            'forms': 0.0789,
            'trial_retraction': 0.00546,
            'trial_objective': 0.0065,
            'tracking': 0.0,
            'remainder': 0.00104},
 'compile': None,
 'cg_per_iteration': 0.0189,
 'rows': [{'window': 0.159,
           'seconds': {'previous_callback': None,
                       'frame': 9.77e-05,
                       'sweep': 0.00115,
                       'companion': 8.01e-06,
                       'residual': 0.00115,
                       'curvature': 0.00204,
                       'gradient': 0.00322,
                       'reference': 0.111,
                       'cg': 0.0234,
                       'forms': 0.0154,
                       'trial_retraction': 0.000814,
                       'trial_objective': 0.00112,
                       'tracking': None,
                       'remainder': 0.000205},
           'compile_seconds': None,
           'trials': 1,
           'accepted_retraction': 0.000814,
           'setup': 0.00013,
           'setup_compile': None,
           'alignment': 0.0796,
           'alignment_compile': None,
           'alignment_passes': (0.0185, 0.0124, 0.0136, 0.0137, 0.0137),
           'reference_passes': (0.0113, 0.0137, 0.0119, 0.0106, 0.0108, 0.0119, 0.0102, 0.0101, 0.00973)},
          {'window': 0.0906,
           'seconds': {'previous_callback': 0.000304,
                       'frame': 0.000516,
                       'sweep': 0.00099,
                       'companion': 3.35e-06,
                       'residual': 0.00106,
                       'curvature': 0.00141,
                       'gradient': 0.00293,
                       'reference': None,
                       'cg': 0.0577,
                       'forms': 0.0218,
                       'trial_retraction': 0.00198,
                       'trial_objective': 0.00168,
                       'tracking': None,
                       'remainder': 0.000206},
           'compile_seconds': None,
           'trials': 1,
           'accepted_retraction': 0.00198,
           'setup': None,
           'setup_compile': None,
           'alignment': None,
           'alignment_compile': None,
           'alignment_passes': None,
           'reference_passes': None},
          {'window': 0.112,
           'seconds': {'previous_callback': 0.000277,
                       'frame': 0.000592,
                       'sweep': 0.00116,
                       'companion': 6.23e-06,
                       'residual': 0.00115,
                       'curvature': 0.00133,
                       'gradient': 0.00317,
                       'reference': None,
                       'cg': 0.0899,
                       'forms': 0.0126,
                       'trial_retraction': 0.000816,
                       'trial_objective': 0.0011,
                       'tracking': None,
                       'remainder': 0.000179},
           'compile_seconds': None,
           'trials': 1,
           'accepted_retraction': 0.000816,
           'setup': None,
           'setup_compile': None,
           'alignment': None,
           'alignment_compile': None,
           'alignment_passes': None,
           'reference_passes': None},
          {'window': 0.192,
           'seconds': {'previous_callback': 0.000241,
                       'frame': 0.000494,
                       'sweep': 0.00102,
                       'companion': 3.25e-06,
                       'residual': 0.00108,
                       'curvature': 0.00144,
                       'gradient': 0.00334,
                       'reference': None,
                       'cg': 0.166,
                       'forms': 0.0155,
                       'trial_retraction': 0.000962,
                       'trial_objective': 0.00136,
                       'tracking': None,
                       'remainder': 0.000232},
           'compile_seconds': None,
           'trials': 1,
           'accepted_retraction': 0.000962,
           'setup': None,
           'setup_compile': None,
           'alignment': None,
           'alignment_compile': None,
           'alignment_passes': None,
           'reference_passes': None},
          {'window': 0.272,
           'seconds': {'previous_callback': 0.000286,
                       'frame': 0.000584,
                       'sweep': 0.00118,
                       'companion': 3.81e-06,
                       'residual': 0.00126,
                       'curvature': 0.00143,
                       'gradient': 0.00345,
                       'reference': None,
                       'cg': 0.248,
                       'forms': 0.0137,
                       'trial_retraction': 0.000896,
                       'trial_objective': 0.00125,
                       'tracking': None,
                       'remainder': 0.000193},
           'compile_seconds': None,
           'trials': 1,
           'accepted_retraction': 0.000896,
           'setup': None,
           'setup_compile': None,
           'alignment': None,
           'alignment_compile': None,
           'alignment_passes': None,
           'reference_passes': None},
          {'window': 0.00769,
           'seconds': {'previous_callback': 0.000266,
                       'frame': 0.000572,
                       'sweep': 0.00114,
                       'companion': 3.73e-06,
                       'residual': 0.00118,
                       'curvature': 0.00131,
                       'gradient': 0.0032,
                       'reference': None,
                       'cg': None,
                       'forms': None,
                       'trial_retraction': None,
                       'trial_objective': None,
                       'tracking': None,
                       'remainder': 2.31e-05},
           'compile_seconds': None,
           'trials': None,
           'accepted_retraction': None,
           'setup': None,
           'setup_compile': None,
           'alignment': None,
           'alignment_compile': None,
           'alignment_passes': None,
           'reference_passes': None}]}
```

The same run on jax (x64) with `use_jit=True`, twice: the compile column on empty program caches, then at the same
structure (script `p2/example_jax.py`; `rows` abbreviated). Row 0 carries CG's one program; the ragged alignment is a
host loop around its jitted step, so its passes are timed (the uniform whole-loop jit would give `None`); the eager
phases (`forms`' exact `pᵀHp`, `gradient`, `curvature`, the retraction) compile per operation on the first run only.

```
jax first run (empty caches), use_jit=True, x64 -- stats['timing'] (floats to 3 significant digits):
{'total': 39.8,
 'setup': 0.179,
 'alignment': 18.8,
 'alignment_passes': (6.95, 0.107, 1.79, 4.65, 1.1),
 'reference_passes': (1.1, 0.101, 1.18, 1.09, 0.988, 0.101, 0.0945, 0.0861, 0.0882),
 'last_callback': 0.000333,
 'phases': {'previous_callback': 0.00147,
            'frame': 1.14,
            'sweep': 0.133,
            'companion': 4.65e-05,
            'residual': 0.123,
            'curvature': 2.46,
            'gradient': 1.07,
            'reference': 6.3,
            'cg': 3.27,
            'forms': 2.51,
            'trial_retraction': 3.03,
            'trial_objective': 0.117,
            'tracking': 0.0,
            'remainder': 0.608},
 'compile': {'previous_callback': 0.0,
             'frame': 1.05,
             'sweep': 0.0,
             'companion': 0.0,
             'residual': 0.0,
             'curvature': 2.25,
             'gradient': 0.608,
             'reference': 4.97,
             'cg': 3.15,
             'forms': 0.992,
             'trial_retraction': 2.85,
             'trial_objective': 0.0,
             'tracking': 0.0,
             'remainder': 0.575,
             'setup': 0.0,
             'alignment': 17.6,
             'last_callback': 0.0},
 'cg_per_iteration': 0.0039,
 'rows': '[6 rows; row 0 compile_seconds.cg = 3.15]'}
jax second run (same structure), use_jit=True, x64 -- stats['timing'] (floats to 3 significant digits):
{'total': 4.1,
 'setup': 0.00126,
 'alignment': 0.533,
 'alignment_passes': (0.0903, 0.094, 0.113, 0.107, 0.0908),
 'reference_passes': (0.0929, 0.107, 0.099, 0.0922, 0.093, 0.0915, 0.0878, 0.0945, 0.0993),
 'last_callback': 0.000394,
 'phases': {'previous_callback': 0.00158,
            'frame': 0.0357,
            'sweep': 0.126,
            'companion': 3.62e-05,
            'residual': 0.14,
            'curvature': 0.0885,
            'gradient': 0.44,
            'reference': 1.01,
            'cg': 0.0301,
            'forms': 1.52,
            'trial_retraction': 0.0567,
            'trial_objective': 0.113,
            'tracking': 0.0,
            'remainder': 0.00937},
 'compile': {'previous_callback': 0.0,
             'frame': 0.0,
             'sweep': 0.0,
             'companion': 0.0,
             'residual': 0.0,
             'curvature': 0.0,
             'gradient': 0.0,
             'reference': 0.0,
             'cg': 0.0,
             'forms': 0.0,
             'trial_retraction': 0.0,
             'trial_objective': 0.0,
             'tracking': 0.0,
             'remainder': 0.0,
             'setup': 0.0,
             'alignment': 0.0,
             'last_callback': 0.0},
 'cg_per_iteration': 0.00097,
 'rows': '[6 rows; row 0 compile_seconds.cg = 0]'}
```

What the split already shows on this toy (laptop CPU; not a benchmark, P3 measures): under `use_jit` only CG is one
program, so on the cached run `forms` (the exact `pᵀHp` = `gn_quadratic` + `curvature_quadratic`, eager) is 1.5 s against
CG's 0.03 s, and `gradient` / `reference` are the next largest; on the first run the `remainder` carries 0.58 s of
compile (first dispatches of eager operations outside the phases, such as `-g` and `‖x‖`), which is what the
compile column is for.

## 5. Verification

One job at a time, current env `t3toolbox` (py3.11, numpy 2.4.6, jax 0.10.2) unless stated; scripts and logs in the
session scratchpad `p2/`.

| check | result |
|---|---|
| **Non-breaking, bitwise against HEAD** (`89ac203f` in a scratchpad worktree, since removed; the pickles stay in `p2/`): `p2/battery.py` run in both trees, `p2/compare.py` (every float by its bytes, arrays by dtype / shape / bytes, container types and dict key order kept; the history's `wall_time` dropped, the display text with its `%.2fs` cell masked). numpy, ragged and uniform, each unshared and shared `(0, 1, 0)`: `newton_cg` Gauss-Newton line search (capped; converged), exact + trust region with rejections, exact + line search, Gauss-Newton + trust region, `padded_from` + `'cone'`, `padded_from` + exact; `gradient_descent('cone')`; `padded_start` (defaults, `tol=0`, the zero tensor); `Problem.local_model` both modes (frame, sweep, residual, companion, gradient, `hvp`, quadratic, misfit, objective); corewise; regularized; Armijo exhausted under `'stop'` and `'accept'`; `max_newton=0`; the overrides without a display; the corewise `padded_from` error. jax x64: `use_jit` Gauss-Newton, exact, and `padded_from` + `'cone'` on both layers; eager ragged `'cone'`. Each `newton_cg` run with `make_newton_display`'s text and records | **numpy 54 entries (745 arrays), 0 mismatches; jax 7 entries (52 arrays), 0 mismatches** (rerun on the final code). A planted one-ulp change is caught |
| **Mutation testing** of the new tests (`p2/mutate.py`: each mutation on a fresh copy of the tree, the targeted tests rerun) | **33 of 33 caught**: the lap without its wait; no setup wait; the frame lap waiting on nothing / on the sweep; `timing` always passed; `**kwargs` counted as naming it; never passed; the override guard removed; `timing` a history field; no `previous_callback` lap; the gradient lap before the lift diagnostics; no mark before CG; `‖x‖` in `forms`; the one-offs on every row; `setup` including the alignment; the listener unregistered only on success (jax); no compile accounting (jax); two of the three events; a listener on numpy runs; `trials` off by one; `accepted_retraction` on rejections; the window from the last lap; a zero remainder; `gradient_descent` / `padded_start` / an untimed `newton_cg` asking for timing; the converged row not closed at its reading; `cg_per_iteration` with the compile (jax); the CG lap waiting on nothing; Q-1 (b) (the transpose in `curvature`); `phases` without the remainder; the cone ignoring `timing`; the objective lap before the objective. (A first-round mutation, the converged row not updating `t_prev`, was MISSED because that line was dead code; it is removed, call 11) |
| new tests | `test_optimizers.py::TestNewtonTiming` 7 passed + the slow row skipped (8.4 s); `T3TOOLBOX_SLOW_TESTS=1` `test_the_remainder_is_small` passed (1.7 s); `test_padded_start.py::TestNewtonTimingAtTheSeam` 4 passed; `test_dispatch.py::TestNewtonTiming` 3 passed (65 s: eager 11 s, ragged `use_jit` 21 s, uniform `use_jit` + `padded_from` 32 s) |
| full suite, `pytest tests/ -q -n 3` | **1,368 passed, 1 failed, 5 skipped**, 48,856 subtests (22 min). The failure is the frontend mirror test of §3 (`name='timing'`); after its temporary line, the touched numpy files (`test_optimizers_frontend`, `test_optimizers`, `test_padded_start`) **126 passed** and `test_dispatch -k TestNewtonTiming` + `test_jax_absent` **4 passed**. The fifth skip is the new slow-tier row |
| module doctests, `pytest --doctest-modules t3toolbox -n 3` | **241 passed** (unchanged: no doctest added or edited) |
| `python -m sphinx -W --keep-going -b html docs <scratch>` | **exit 0, no warning**; `autoapi/t3toolbox/backend/optimizers/{NewtonTiming,PhaseSeconds}.html` rendered; the private helpers stay out |
| floor env `tttt` (py3.9, numpy 1.22.4, jax 0.4.30), single process: `test_optimizers`, `test_padded_start`, `test_optimizers_frontend`, `test_dispatch` | **209 passed, 2 skipped** (29 min): the skips are the new slow-tier row and the pre-existing `test_eager_jax_compiles_nothing_twice`; `TestNewtonTiming` runs there on its `None` branch (0.4.30 cannot unregister a listener, so `compile_seconds` and `stats['timing']['compile']` are `None`), the program-cache checks included |

No `docs/*.md` page changed, so the page doctests were not rerun.

## 6. P3 must delete

- `tests/test_optimizers_frontend.py`, `TestLeastSquaresProblemAndKeywords.test_newton_cg_keywords_mirror_backend`: the
  `not_yet = ('timing',)` line and the two-line loop below it (`for name in not_yet: self.assertNotIn(...)`), and the
  `or name in not_yet` in the mirror loop. The `assertNotIn` makes the line self-expiring: once P3 adds the frontend
  `timing` keyword, this test fails with "timing is now a frontend keyword: delete not_yet (the timing plan, P3)"
  until the exemption is removed (checked on a scratch copy, round 1 below).

## 7. Round 1 (review `P2_review.md`: REVISE, four MINOR, all test-side; four NIT)

The reviewer's full suite was running, so per the machine rule only single-process numpy targeted tests were run this
round; the jax change (M4) is written and awaits a jax run (below).

| finding | change | checked |
|---|---|---|
| **M1** the mirror exemption could outlive P3 | `test_newton_cg_keywords_mirror_backend` now also asserts `assertNotIn(name, front, '%s is now a frontend keyword: delete not_yet (the timing plan, P3)')` for each exempt name; §6 above lists what P3 deletes | scratch copy with `timing: bool = False` added to the frontend `newton_cg`: **red** (the self-expiry assertion) |
| **M2** no test saw a lap reading the clock before waiting; three branches' waits unchecked | `test_each_phase_waits_on_its_outputs` rewritten: `tree_block_until_ready` is mocked to sleep `WAIT` = 30 ms on every call and record `(phase, tree)` (the phase from a wrapped `_PhaseClock.lap`, rows split at each `close`). `check_waits_inside_laps`: every phase lasts at least `WAIT` per wait it ended on its row, no row's `remainder` reaches `WAIT`, `setup` >= `WAIT` (its own wait) and < 2 `WAIT` with a padded start (the alignment's wait is not in it), `alignment` >= its waits. Four runs: exact + trust region (the full per-row lap sequence and identities, as before), Gauss-Newton + line search (every trial's wait is a point; the last is `seen[1].x_cores`), a padded start (the alignment waits on `(frame, movements)`, `frame is seen[0].lm.frame`), an `_ScaledModelProblem` (the override's `frame` lap waits on `seen[0].lm` itself). Untimed: no wait | mutations (single process): the lap reading before it waits, the line-search trial / the alignment / the override's frame lap waiting on nothing: **4 of 4 caught**; round 0's six wait mutations rerun on the new test: **6 of 6 caught** |
| **M3** `accepted_retraction` on a backtracked row not pinned | `check_timing`: with `trials > 1` and an accepted step, `accepted_retraction < seconds.trial_retraction` (strict); with one trial, equal | mutation "the accepted share is the row's whole `trial_retraction`": **caught** (the `'exhausted, accept'` case, 40 trials) |
| **M4** the listener's unregistration on success untested | `TestNewtonTiming.test_eager_jax`: the listener count is taken before the first timed run and asserted after it, and again after the raising run | written; the jax run and the mutation "unregister only on an exception" (`mutate.py`, `r1jax`) wait for the reviewer's go (machine rule) |
| **N1** the unreadable-signature branch untested | `_UnreadableMember` (its member is a callable whose `__signature__` is not a `Signature`; `inspect.signature` raises `TypeError` on both envs, checked) joins `test_a_member_without_timing` | mutation "no `try` / `except` around the signature read": **caught** |
| **N2** no timed run converges on row 0 | `TestNewtonTiming.CASES` gains `'converged at row 0'` (`gtol_rel=10.0`); `check_timing` sees the one-offs on a converged row 0 | mutation "the converged row drops the one-offs": **caught** |
| **N3** the staircase comment on the line-search `retraction = _lap(...)` | dropped (its trust-region twin has none) | -- |
| **N4** a class-level patch of `Problem.local_model` passed the guard and was bypassed under timing | taken (library, timing path only): `_STOCK_LOCAL_MODEL = Problem.local_model`, captured once after the class; the guard is `type(problem).local_model is _STOCK_LOCAL_MODEL`. This departs from §2.5's literal `is Problem.local_model` for the reason §2.5 gives ("timing never bypasses an override"). New `test_a_patched_local_model_is_called_whole` (the patch is called once per model, the build reported whole in `frame`) | mutation "the guard against the live attribute": **caught**. The untimed path does not read the guard, so the bitwise battery is unaffected |

Runs this round (current env, single process): `TestNewtonTiming` 8 passed + the slow row skipped; the seam tests and
the mirror test 5 passed; the touched numpy files in full (`test_optimizers`, `test_padded_start`, `test_optimizers_frontend`): **127 passed, 1 skipped** (6 min).
