# Opt-in phase timing for `newton_cg` — plan (draft 2, 2026-11-13)

> **Status: DRAFT 2** (writer, after critic r1 `dev/newton_timing_plan_review_r1.md`), for the critic, then Nick. Signature level (code in the house signature style; blocks
> here compress today's arguments). T-1..T-7 settled (§8); §9 holds the one open question. Lines (working tree,
> 2026-11-13): `backend/optimizers.py` `Problem.local_model` ~360, `padded_start` ~763, `_cone_reference` ~841,
> `NewtonInfo` ~1148, `newton_cg` ~1313 (loop ~1536–1670); `geometry.py` members ~376 / ~818; `padded_frames.py` ~721;
> `uniform_padded_frames.py` ~760–911. Store words are 2026.5.0's. Every agent first reads a handful of
> `TuckerTensorTrain` methods and their backend functions, the only precedent for code shape.

## CHANGES

*History only. Where a CHANGES line and a section differ, the section governs.*

**Draft 2 (2026-11-13, writer; critic r1: no MAJOR, 4 MINOR, 4 NIT, all applied).**

| row | from | change | where |
|---|---|---|---|
| 2.1 | r1 M1 | §2.7 rewritten against switch draft 2 (five `_SWITCH_FIELDS`, excluded from `_NEWTON_SCALAR_FIELDS`, appended to history only in switch mode); `PhaseSeconds.tracking` added after `trial_objective`, its execution order (r2 nit; the switch laps it; `None` here); a `track` cell; timing keeps §4.13 and creates `### Added`, the switch takes §4.14 and appends | §2.1, §2.7, §3.1, §4, §6, T4 |
| 2.2 | r1 M2 | W-3 narrowed: `timing=True` reaches the member only when its signature names `timing` explicitly (`**kwargs` is not evidence); T7 adds a `**kwargs`-forwarding member | §2.6, §8, T7 |
| 2.3 | r1 M3 | W-6 guarded: the kernel raises on `clock` with `use_jit`; its partial-binding note names the clock slot; the start frame is synchronized before pass 1, and each pass is [synced start, synced end], ragged and uniform | §2.6, §3.3, T7 |
| 2.4 | r1 M4 | A `Problem` subclass that overrides `local_model` is called, not bypassed: its whole build is one lap, reported in `frame`, the other split phases `None`; T3 adds the case | §2.5, T3, T4 |
| 2.5 | r1 N1 | Async work dispatched in `remainder` (`neg_g`) or by a callback without waiting is charged to a later lap: one caveat | §2.4 |
| 2.6 | r1 N2 | T2 keeps only construction identities (tiling, `remainder ≥ −1e-6`, the `total` identity, now exact by construction); the 10 % wall-clock bound moves to the slow tier | §3.2, T2 |
| 2.7 | r1 N3 | `_PhaseClock` / `_CompileSeconds` are host-side accumulators, not parameter-holding classes: one line | §3.1 |
| 2.8 | r1 N4 | `stats['timing']['phases']` sums `PhaseSeconds` only; the pass tuples are breakdowns, never added | §3.2 |

**Draft 1 (2026-11-13, writer; the session's decisions on T-1..T-7, HANDOFF item (A)).**

- T-1..T-7 applied (§8). T-6 changed: the member shipped in 2026.4.0, so it gains `timing: bool = False` and a third
  element only when set, not a new arity (§2.6).
- Records flattened (W-1; `ModelSeconds`, `OneOffSeconds` dropped); `step_*` renamed `trial_*`. New: W-2..W-7, Q-1
  (§9), `common.tree_block_until_ready`, switch coexistence (§2.7), three slices with a stop after P2, measured figures.

**Draft 0 (2026-11-10, writer).** First draft, with T-1..T-7 open.

## 1. What ships

| layer | name | what |
|---|---|---|
| backend `optimizers.py` | `newton_cg(..., timing=False)` | keyword-only, appended after `cone_seed` |
| | `PhaseSeconds`, `NewtonTiming` | frozen dataclasses of floats (and tuples of floats), in `__all__` |
| | `NewtonInfo.timing` | `None` unless `timing=True`; not a history field |
| | `stats['timing']` | a plain dict, present only with `timing=True` (§3.2) |
| backend `geometry.py` | `aligned_padded_frame(..., timing=False)` (ragged and uniform) | a third element, the per-pass seconds, only when set |
| backend `common.py` | `tree_block_until_ready(tree)` | in `__all__`, beside `tree_to_jax` |
| backend `optimizer_display.py` | `format_newton_iter`, `make_newton_display` | a `time` line per timed row; a closing table on the last row |
| frontend `optimizers.py` | `newton_cg(..., timing=False)` | keyword-only after `cone_seed`; forwarded only when `True` |

**The non-breaking rule.** With `timing=False`, a seeded run is bitwise identical to today's: `x`, the whole `stats`
dict and every history row, the `wall_time` clock readings aside. No synchronization and no clock reading is added. The
loop calls `problem.local_model` as today. The member is called without the `timing` keyword, as today. `NewtonInfo`
gains `timing` (default `None`), excluded from `_NEWTON_SCALAR_FIELDS`, so history keys and their order are unchanged.
`timing=True` changes no number either (T3).

## 2. What is timed

### 2.1 Per row: the `wall_time` window, split

The window is today's: from the previous row's `t_now` (before the loop on row 0) to this row's `t_now`.
`NewtonTiming.window` is that difference, the same float as `wall_time` on a step row; the converged row gets one too
(W-4). The clock's laps tile it:

| phase | code it covers | `None` when |
|---|---|---|
| `previous_callback` | the previous row's `NewtonInfo` + `NewtonTiming`, its `callback`, `history.append` | row 0 |
| `frame` | `geom.frame(x)`, or `require_frame_structure` of a start frame | the row reuses the model (after a rejected trust-region trial) |
| `sweep` | `kind.precompute(frame, sample)` | as `frame` |
| `companion` | `geom.precompute(frame)` | as `frame` |
| `residual` | `kind.point_forward(x, sample) − data` | as `frame` |
| `curvature` | `geom.curvature_precompute(...)` | Gauss-Newton mode; as `frame` |
| `gradient` | the misfit transpose `Π 𝒥ᵀ r` (in exact mode it runs inside the model build and is clocked here, §9), `lm.misfit`, `lm.regularization`, `lm.gradient`, `‖g‖`, `_lift_diagnostics` | as `frame` |
| `reference` | `_resolve_g0_reference` (`'zero'`: one sweep + one transpose at the zero-point frame) or `_cone_reference` (the cone loop + one projection); a few µs for `'warm_start'` and the overrides | every row but row 0 |
| `cg` | `_cg_solve` (under `use_jit`: one compiled call) | the converged row |
| `forms` | the first trust-region step's `_initial_radius` (one Gauss-Newton quadratic), `slope`, the descent fallback, `pᵀHp` (exact: `gn_quadratic` + `curvature_quadratic`), `‖p‖` | the converged row |
| `trial_retraction` | every trial's `lm.retract(...)`: the line search's trials, or the trust region's one | the converged row |
| `trial_objective` | every trial's `problem.objective(...)` | the converged row |
| `tracking` | the GN-to-Newton switch's tracking `curvature_quadratic` on a countable Gauss-Newton-phase row (§2.7); it runs after the trials | every row outside switch mode (always, until the switch lands) |
| `remainder` | `window` minus the phases above: the forcing term, `‖x‖`, the floors (`‖ω⊙y‖` once, at the first rejection), Python bookkeeping | never |

Per row also `trials` (line search `ls_steps + 1`, or 40 when exhausted; trust region 1) and `accepted_retraction`
(the accepted trial's share of `trial_retraction`; `None` when none was accepted): the loop keeps the accepted trial's
point (`x, lm = x_trial, None`) and the next row's `frame` re-orthogonalizes it, so there is no separate phase (T-3).

### 2.2 Per run: the one-offs

Carried on row 0's `NewtonTiming` (so a callback sees them) and in `stats['timing']`: `setup` (entry → the loop:
validation, `_prepare_jit_inputs` synchronized, the start-frame bookkeeping; the alignment excluded); `alignment`
(`_padded_start`, the member's whole call, `padded_from` only); `alignment_passes` (the member's per-pass seconds,
§2.6; `None` under the uniform whole-loop jit or a member without `timing`); `reference_passes` (the same for the cone
loop, `'cone'` only; its total is inside row 0's `reference`). `setup` and `alignment` have compile twins. After the
loop, `stats['timing']` adds `last_callback` (the last row's record, callback and append, outside every window, closed
by the reading at the loop's exit) and `total` (entry → that same reading; building `stats` after it is µs).

### 2.3 Compile seconds (T-2: the full wait)

- A private `_CompileSeconds` listener keeps a running total of the three events
  `/jax/core/compile/{jaxpr_trace,jaxpr_to_mlir_module,backend_compile}_duration`. In jax 0.10.2 they never nest (the
  trace event fires only at a clean trace state; a persistent-cache retrieval sits inside the backend event; read in
  `jax/_src/{pjit,dispatch,interpreters/pxla}.py`). A phase's compile seconds are the total's change across its lap;
  the `remainder`'s twin is the window's change minus the phases'.
- It runs only with `timing=True`, on a jax run (`tree_contains_jax(x0)` after `_prepare_jit_inputs`), on a jax with
  `jax.monitoring.unregister_event_duration_listener` (0.10.2 yes; the floor's 0.4.30 no; checked 2026-11-13):
  registered once at entry, unregistered in a `finally`. Otherwise `compile_seconds` and `stats['timing']['compile']`
  are `None`, numpy runs included.
- Caveat (documented): the listener is process-global. Compiles in other threads, or in a nested timed run inside a
  callback, are counted too.

### 2.4 Synchronization

jax dispatch is asynchronous, so with `timing=True` each lap first calls `common.tree_block_until_ready(outputs)`
(`jax.block_until_ready` when the tree holds a jax array, else nothing; both envs accept mixed trees, checked). Laps
ending in `float(...)` / `int(...)` (`gradient`, `reference`, `forms`, `trial_objective`) pass no outputs. Documented:
this serializes dispatch (the host waits at each boundary instead of queueing the next phase), so a timed jax run is
slower than an untimed one and its `wall_time` describes the serialized run. numpy is synchronous: a no-op. Caveat
(documented): work dispatched asynchronously in an unclocked stretch is charged to the next lap that waits for it.
Today that is `neg_g = corewise_scale(g, -1)`, dispatched in `remainder` and finished inside `cg` (tiny); likewise,
jax work a custom callback dispatches without waiting lands in later phases.

### 2.5 The local model's split (T-1)

`_local_model_phases(...)` holds today's `Problem.local_model` body: the validation, then the parts in today's order,
lapping `frame`, `sweep`, `companion`, `residual` and, in exact mode, `gradient` after the misfit transpose and then
`curvature` (§9). `Problem.local_model` becomes `return _local_model_phases(self, ..., None)` (no clock: the same
operations in the same order); its docstring and doctest stay on the method. `newton_cg` keeps today's
`problem.local_model` line when timing is off. When it is on, it calls `_local_model_phases(..., clock)` only if
`type(problem).local_model is Problem.local_model`. A subclass that overrides `local_model` (subclassing `Problem` is a
tested pattern: `_OffsetProblem` overrides `objective`) is called as today, its whole build one lap reported in
`frame`, with `sweep`, `companion`, `residual` and `curvature` `None` (documented). Timing never bypasses an override.

### 2.6 The alignment's per-pass times (T-6)

- **Loops.** The public `fv_aligned_padded_frame` / `ufv_aligned_padded_frame` keep signatures and returns; their
  bodies move into private twins `_fv_aligned_padded_frame(..., timing)` / `_ufv_aligned_padded_frame(..., timing)`
  returning `(frame, movements, pass_seconds)`, which the public ones call with `timing=False` and slice to two.
  - Each pass's interval is [synchronized start, synchronized end]: the start frame (`fv_padded_frame` /
    `ufv_padded_frame`) is synchronized once before pass 1's clock starts, and each pass ends with
    `tree_block_until_ready((frame, movement))` (its cores and movement on the uniform layer), on both layers.
  - Ragged (a host loop in both modes): each `_aligning_pass` call is clocked so.
  - Uniform eager: `_ufv_aligned_loop_kernel` gains `clock: Optional[list] = None` and the pass statics a last slot
    `clock`; `_aligning_pass`, through which every pass runs (the truncating pass, `_oversampled_body`, both passes of
    `_pairs_body`, `_final_pass`, `_single_body`), appends its seconds when `clock` is a list (W-6). The kernel raises
    a `ValueError` when `clock is not None and use_jit` (under a trace the append would record tracing time and the
    sync would meet tracers). Its docstring's note accepting the `functools.partial` binding "in this instance only"
    is extended to name the clock slot and why it is eager-only.
  - Uniform under `use_jit`: the jitted call never passes `clock`, so the program and its cache key are today's (T6);
    `pass_seconds` is `None`. The early returns (zero passes, nothing to complete) give `()`.
- **Members.** `aligned_padded_frame(..., use_jit=False, timing=False)`: off, today's call to the public loop; on, the
  private twin, all three elements. The `Geometry` protocol docstring lists the keyword; a custom member need not
  have it: `newton_cg` passes `timing=True` only when `inspect.signature(member).parameters` names `timing` explicitly,
  else calls it as today and reports `alignment_passes` / `reference_passes` `None` (W-3).
- **Drivers.** The public `padded_start` keeps `(frame, movements)`; its body after option validation moves into
  `_padded_start(..., timing)` (three elements), which `newton_cg` calls with its flag. `_cone_reference` gains
  `timing` and returns `(value, movements, pass_seconds)`; `gradient_descent` passes `False` and drops the third.

### 2.7 Coexistence with the GN-to-Newton switch (bundle item (B))

Read against `dev/gn_newton_switch_plan_2026-11-13.md` draft 2 (its §3.3). Both live on `NewtonInfo` and are inert
when off. The switch's five per-row fields (`_SWITCH_FIELDS`: `switch_pKp`, `switch_e_gn`, `switch_e_n`,
`switch_count`, `switch_fired`) are excluded from `_NEWTON_SCALAR_FIELDS`, as `timing` is, and
`_newton_scalar_record` appends them only on switch-mode rows. So neither feature changes a default run's history, and
`NewtonInfo`'s field order touches no history. The switch's work maps onto phases:

- its Gauss-Newton-phase point's exact build is `curvature` (`curvature_precompute` alone), with the transpose in
  `gradient` (Q-1(a));
- its tracking `curvature_quadratic` on a countable row is `PhaseSeconds.tracking`, so `forms` keeps its meaning. This
  plan adds the field (always `None` until the switch lands); the switch plan adds the lap;
- the `_cg_core` compile when it fires is in the first exact row's `compile_seconds.cg`.

A row's tracking overhead is `seconds.curvature + seconds.tracking`, with the compile twins beside it. Docs: timing
writes §4.13 and creates CHANGELOG `[2026.5.0]`'s `### Added`; the switch takes §4.14 and appends.

## 3. API (house style)

### 3.1 Records, keyword, private helpers (`backend/optimizers.py`)

```python
@dc.dataclass(frozen=True)
class PhaseSeconds:                              # one history row's window by phase, seconds; None = the phase did not run on this row
    previous_callback:  typ.Optional[float]      # the previous row's record, callback and history append; None on row 0
    frame:              typ.Optional[float]      # frame .. gradient: None on a row that reuses the model; an override's whole build (§2.5)
    sweep:              typ.Optional[float]
    companion:          typ.Optional[float]
    residual:           typ.Optional[float]
    curvature:          typ.Optional[float]      # also None in Gauss-Newton mode
    gradient:           typ.Optional[float]      # the misfit transpose (wherever it runs, §9) .. the lift diagnostics
    reference:          typ.Optional[float]      # row 0 only
    cg:                 typ.Optional[float]      # cg .. trial_objective: None on the converged row
    forms:              typ.Optional[float]
    trial_retraction:   typ.Optional[float]      # every trial's retraction
    trial_objective:    typ.Optional[float]      # every trial's objective
    tracking:           typ.Optional[float]      # the switch's tracking product (§2.7), after the trials; None outside switch mode
    remainder:          float                    # window minus the phases above (>= 0 up to rounding)

@dc.dataclass(frozen=True)
class NewtonTiming:                                              # NewtonInfo.timing (timing=True only)
    window:               float                                  # this row's window (== wall_time on a step row)
    seconds:              PhaseSeconds
    compile_seconds:      typ.Optional[PhaseSeconds]             # jax's compile wait per phase; None off jax / on the floor (§2.3)
    trials:               typ.Optional[int]                      # retraction trials; None on the converged row
    accepted_retraction:  typ.Optional[float]                    # the accepted trial's share of trial_retraction; None if none accepted

    setup:                typ.Optional[float]                 = None   # row 0 only, from here down (§2.2)
    setup_compile:        typ.Optional[float]                 = None
    alignment:            typ.Optional[float]                 = None   # padded_from only
    alignment_compile:    typ.Optional[float]                 = None
    alignment_passes:     typ.Optional[typ.Tuple[float, ...]] = None   # per pass; None under the uniform whole-loop jit
    reference_passes:     typ.Optional[typ.Tuple[float, ...]] = None   # 'cone' only

class NewtonInfo:                                  # (existing) + one field, last
    timing:  typ.Optional[NewtonTiming] = None     # timing=True only; not a history field

_NEWTON_SCALAR_FIELDS = tuple(f.name for f in dc.fields(NewtonInfo) if f.name not in ('x_cores', 'lm', 'timing'))

def newton_cg(problem, x0, ..., callback=None, *, start_frame=None, ..., cone_seed=None,   # unchanged
        timing:  bool = False,                     # NEW: NewtonInfo.timing per row, stats['timing'] per run
) -> typ.Tuple[Tangent, dict]:

def _local_model_phases(problem, x_cores, sample, data, hessian, curvature_damping, frame,   # Problem.local_model's
        clock:  typ.Optional['_PhaseClock'],       # None: no synchronization, no clock reading (local_model's call)
) -> LocalModel:

# _CompileSeconds and _PhaseClock are host-side mutable accumulators, not parameter-holding backend classes
# (parameters_not_closures.md does not apply): they never ride as aux and never enter a trace.
class _CompileSeconds:                             # the run's jax.monitoring duration listener
    total:  float                                  # compile seconds so far
    def __call__(self, event: str, duration: float, **kwargs) -> None:   # adds the three counted events (§2.3)

class _PhaseClock:                                 # timing=True only; laps tile the window
    def __init__(self, listener: typ.Optional[_CompileSeconds]) -> None:   # None: no compile column
    def mark(self, t: typ.Optional[float] = None) -> None:          # start a lap at t (the loop's own t_now) or now; unattributed time -> remainder
    def lap(self, phase: str, outputs: typ.Any = None) -> float:    # sync outputs; add the seconds (and compile) since the mark to phase; re-mark; return the lap
    def row(self, window: float, trials, accepted_retraction, **one_offs) -> NewtonTiming:   # close the row; reset the per-row sums

def _padded_start(problem, x0, x_old, options: dict, use_jit: bool, timing: bool
) -> typ.Tuple[typ.Any, typ.Any, typ.Optional[typ.Tuple[float, ...]]]:   # (frame, movements, pass_seconds)

def _cone_reference(problem, x0, options, use_jit, who, hint, timing: bool
) -> typ.Tuple[float, typ.Tuple[float, ...], typ.Optional[typ.Tuple[float, ...]]]:   # (value, movements, pass_seconds)
```

`newton_cg` passes `timing=True` to the member only when timing is on and the member accepts it (W-3); otherwise
`pass_seconds` is `None`.

### 3.2 `stats['timing']` (a plain dict)

```python
{'total': float, 'setup': float, 'alignment': Optional[float], 'alignment_passes': Optional[tuple],
 'reference_passes': Optional[tuple], 'last_callback': float,
 'phases': {phase: float},              # each PhaseSeconds field summed over the rows (None as 0.0); one-offs and pass tuples excluded
 'compile': Optional[dict],             # the same for compile_seconds, plus 'setup', 'alignment', 'last_callback'
 'cg_per_iteration': Optional[float],   # Σ (cg − its compile, when known) / Σ cg_iters; None without a CG iteration
 'rows': [dict, ...]}                   # dc.asdict(info.timing) for each history row, in order
```

The pass tuples are breakdowns of `alignment` and of row 0's `reference`; nothing is counted twice. Identity, exact by
construction up to `perf_counter`'s float rounding: `total == setup + (alignment or 0) + Σ window + last_callback` (the
readings tile entry → the loop's exit).

### 3.3 Loops, members, helper

```python
def tree_block_until_ready(                        # backend/common.py, in __all__
        tree:  typ.Any,                            # any pytree (jax / numpy arrays, ints, None)
) -> None:                                         # jax.block_until_ready(tree) if it holds a jax array; else nothing

def _fv_aligned_padded_frame(<fv_aligned_padded_frame's arguments>, timing: bool)     # padded_frames.py
def _ufv_aligned_padded_frame(<ufv_aligned_padded_frame's arguments>, timing: bool)   # uniform_padded_frames.py
# both -> (frame, movements, pass_seconds: Optional[Tuple[float, ...]])
def _ufv_aligned_loop_kernel(dynamic, tol, skeleton, project, schedule, use_jit,
        clock:  typ.Optional[list] = None,         # eager only: one appended float per pass; with use_jit, a ValueError
)

def aligned_padded_frame(self, x0, x_old, project, hint, ..., use_jit=False,   # both members; today's arguments
        timing:  bool = False,                     # NEW: True appends the seconds per pass (None under the uniform whole-loop jit)
) -> typ.Tuple:                                    # (frame, movements) as today; timing=True: (frame, movements, pass_seconds)
```

The frontend `optimizers.newton_cg(..., timing: bool = False)` appends the keyword after `cone_seed` and forwards it
only when `True`. Its docstring covers: what is timed; the synchronization and compile caveats; that the frontend's own
setup (`least_squares_problem`, `padded_from`'s reduction) runs before the backend clock starts (W-5); an example
reading `stats['timing']`; and `verbose=True` showing the split.

## 4. Display (`backend/optimizer_display.py`)

- **`format_newton_iter`**: when `info.timing` is not `None`, one `time` line between the header and the error table:

      time  model .12 [frame .05 sweep .06 comp .00 resid .01] | grad .03 | cg .40 (33 ms/it) | forms .02 |
            trials .21 ×3 (accepted .07) | track .05 | cb .01 | rest .01 | compile cg 1.9

  `track` shows only when `tracking` is set. Row 0 adds `setup`, `alignment` with its pass count, and `reference`.
  Built by a private `_format_timing(info)`.
- **`make_newton_display`**: the callback keeps the rows' `NewtonTiming` (reset at `iteration == 0`, so it can be
  reused across levels) and on the last row (`converged`, or `stop_reason` set) prints a closing table from a private
  `_format_timing_table(rows)`: phases × (seconds, % of Σ window, compile, per-row mean), then the one-offs. The last
  row's own callback is not in it (it is running); `stats['timing']` has it.
- With `timing=False` the strings and records are today's (records come from `_newton_scalar_record`, which skips
  `timing`).

## 5. Tests

| # | file | test | criterion |
|---|---|---|---|
| T1 | `tests/backend/test_optimizers.py` | **Off = today.** `NewtonInfo.timing is None` on every row; `'timing' not in stats`; `_NEWTON_SCALAR_FIELDS` equals today's tuple. G-bit: seeded runs (ragged and uniform; line search, exact + trust region with a rejection, `padded_from`, `'cone'`) compared with the parent commit, recorded once in the slice handoff by the aligned-frame plan's G-bit procedure | `x` and `stats` bitwise, `wall_time` aside |
| T2 | same | **The identities**, `timing=True` (numpy; jax eager; jax `use_jit`; line search; trust region with a rejected trial; exact; `padded_from`; `'cone'`): per row, Σ non-`None` phases + `remainder` == `window`; `window == wall_time` on step rows; `remainder ≥ −1e-6` (the laps lie inside the window; `perf_counter` float rounding); the `total` identity of §3.2 to 1e-6 s. No wall-clock bound in the always-on suite. **Slow tier** (`T3TOOLBOX_SLOW_TESTS=1`), on a problem whose window is well above Python's per-row overhead: Σ `remainder` ≤ 0.1·Σ `window` | as stated |
| T3 | same | **Timing changes no number**: `x` and `stats` (minus `timing` and the `wall_time` fields) on vs off; plus a `Problem` subclass overriding `local_model` (its override runs under both, so on == off, and differs from the stock build) | bitwise: numpy, jax eager, jax `use_jit` (CPU) |
| T4 | same | **The record's shape**: model phases `None` on the row after a rejected TR trial; `curvature` `None` in GN mode; `reference` on row 0 only; `previous_callback` `None` on row 0; the converged row has `window` with `wall_time is None` and `cg`..`trial_objective` `None`; `trials` = `ls_steps + 1`, 40 when exhausted, 1 under TR; `accepted_retraction` `None` on a rejection and ≤ `trial_retraction`; `stats['timing']['rows'] == [dc.asdict(i.timing) for i in seen_by_callback]`; `max_newton=0` gives `rows == []`; `tracking` `None` on every row; the overriding subclass's build in `frame`, `sweep`..`curvature` `None` | as stated |
| T5 | same + `tests/test_dispatch.py` | **Compile** (jax 0.10): `use_jit=True` at a structure no other test uses, so row 0's `compile_seconds.cg` > 0, and a second run at that structure has `cg` compile == 0 on every row. A callback that raises still unregisters the listener (the listener count is unchanged). Floor env and numpy: `compile_seconds is None` and `stats['timing']['compile'] is None` | as stated |
| T6 | `tests/test_dispatch.py` | **Dispatch unaffected**: the existing compile-count rows unchanged and green; `timing=True` adds no program (`_jitted(_cg_core)` and the uniform whole-loop kernel's `_cache_size()` unchanged between an off and an on run at one structure); `tree_block_until_ready` rows on jax, numpy and mixed trees | as stated |
| T7 | `tests/backend/test_padded_start.py` | **Per-pass times**: both members return 2 elements off and 3 on; the frame and movements bitwise equal off vs on; `len(pass_seconds)` = the passes run (ragged `len(movements)`; uniform eager the non-NaN count), positive floats; uniform `use_jit` → `None`; public loops and `padded_start` return 2; through `newton_cg(timing=True)`, row 0's `alignment_passes` and `reference_passes` match; custom members without `timing`, one with a fixed signature and one forwarding `**kwargs` to the public loop → both runs succeed, equal to `timing=False`, with `alignment_passes is None`; the start frame synchronized before pass 1 (a mock of `tree_block_until_ready` sees it first); `_ufv_aligned_loop_kernel(..., use_jit=True, clock=[])` raises | as stated |
| T8 | `tests/backend/test_optimizer_display.py` | `timing` rows give the `time` line, and the closing table appears once, at the end (both structure-checked, no printed floats); a reused callback resets at row 0; `timing=False` output equals today's (the existing doctests unchanged) | as stated |
| T9 | `tests/test_optimizers_frontend.py` | The frontend forwards `timing` only when `True`; ragged and uniform `timing=True` runs give `stats['timing']` | as stated |
| T10 | — | The full suite on the floor env (jax 0.4.30) and the current env (py3.11 / numpy 2.4 / jax 0.10.2), `-n 3` at most; doctests on both; `sphinx -W` | green |

## 6. Docs

- **`docs/fitting_and_optimization.md` §4.13 "Timing a Newton run"**:
  - the phases (§2.1's table), the one-offs, reading `stats['timing']`, the display;
  - **the costs**: synchronization serializes dispatch; the figures are timing on vs off (median of 3 after a warm-up)
    for numpy, jax eager and jax `use_jit`, ragged and uniform, on the §4.13 example and a d = 6 `probe_derivatives`
    case, measured in P3 on the maintainer's laptop and stated with the config (a GPU loses more overlap: measure there);
  - **jit or eager, as the compile column shows it**: under `use_jit` row 0's `cg` carries CG's one program and a rerun
    at the same structure compiles nothing; the uniform alignment's whole-loop program sits in `alignment_compile`;
    eager jax spreads many per-operation compiles over row 0's phases (cross-reference `aligned_padded_frame.md`);
  - the caveats (compile `None` on jax without `unregister_event_duration_listener`; the listener is process-global;
    `previous_callback` is why a heavy display shows up on the next row) and a platform-stable doctest (keys, the
    window identity, `None` when off).
- **`docs/aligned_padded_frame.md`**: the members' `timing` (one paragraph; host loops only). **Docstrings**: the
  `Geometry` protocol, `newton_cg` (its `stats` paragraph), `NewtonInfo`.
- **CHANGELOG `[2026.5.0]`, a new `### Added`** (timing creates it; the switch appends): `newton_cg(timing=True)` (backend and frontend); `NewtonInfo.timing`
  (`NewtonTiming`, `PhaseSeconds`); `stats['timing']`; the display's time lines and closing table; the members'
  `timing`; `common.tree_block_until_ready`. Add a line to `docs/release_notes.md` only if its 2026.5.0 section lists
  additions.

## 7. Slices

Process (house): per slice an implementer and a reviewer (Opus, xhigh) iterate through
`dev/newton_timing_slices/P<n>_{handoff,review}.md` to `VERDICT: APPROVE`; the session reads the diff, runs the suite
on both envs (`-n 3`), commits by name, pushes, checks CI. Unforeseen issues: judgment calls in the handoff.

| slice | content | gates |
|---|---|---|
| P1 | `tree_block_until_ready`; the loops' private twins (ragged; the uniform `clock` slot); the members' `timing`; the protocol docstring | T6 (helper rows, kernel cache), T7 (member level), T10 |
| P2 | `PhaseSeconds`, `NewtonTiming`, `NewtonInfo.timing`, `_NEWTON_SCALAR_FIELDS`; `_CompileSeconds`, `_PhaseClock`; `_local_model_phases` (with `local_model` delegating); `_padded_start`; `_cone_reference(timing)` and `gradient_descent`'s call site; the loop's laps; `stats['timing']`. **REVIEW STOP with Nick**: the handoff shows a printed timed run (`format_newton_iter` per row + `stats['timing']`) on the §4.13 example | T1–T7, T10 |
| P3 | the display; the frontend keyword and docstring; the overhead and compile figures (§6); docs, CHANGELOG. **FINAL STOP with Nick** | T8, T9, doctests, `sphinx -W`, T10 |

## 8. Decisions record

Settled (the session's recommendations, HANDOFF item (A), 2026-11-13): **T-1** split the local model into frame /
sweep / companion / residual / curvature through `_local_model_phases`, which `local_model` uses (bitwise path);
**T-2** compile = the full wait, one number per phase, via `jax.monitoring` when available, `None` on the floor;
**T-3** no accepted-retraction phase, the accepted trial's share reported; **T-4** keep `wall_time`'s window, report
`previous_callback`; **T-5** no `gradient_descent`; **T-6** the member's `timing: bool = False`, a third element only
when set, the public loops and `padded_start` unchanged; **T-7** `stats['timing']` a plain dict, `NewtonInfo.timing`
the `NewtonTiming` dataclass, `None` when off, not a history field.

Writer's calls in draft 1 (for the critic):

| # | call | why |
|---|---|---|
| W-1 | Flat records: the model parts are phases of `PhaseSeconds`; the one-offs are row-0 fields of `NewtonTiming` | every phase gets a compile twin from one record type; two classes instead of four |
| W-2 | `stats['timing']['rows']`, the per-row dicts | a run without a callback still gets per-row times; history rows stay unchanged |
| W-3 | A member without `timing` keeps working: `newton_cg` passes `timing=True` only when `inspect.signature(member).parameters` names `timing` explicitly (a `**kwargs` member forwarding to the public loop would raise or return two), else reports the pass tuples `None` | a diagnostic must never make an otherwise-working run fail; 2026.4.0 custom geometries predate the keyword |
| W-4 | The converged row gets `window`; `wall_time` stays `None` there | timing changes no existing field |
| W-5 | The frontend's own setup is not timed; the docstring says so | `stats['timing']` describes the backend call, the same for a raw-`.data` user |
| W-6 | The uniform eager per-pass clock is a list in the pass statics, under the kernel's existing `functools.partial` binding; a `ValueError` with `use_jit`; the binding note names it | the one place every pass runs in both modes; under jit it is `None`, so the jitted program is today's |
| W-7 | The trust region's `_initial_radius` is in `forms`; `‖ω⊙y‖` (once) is in `remainder` | a Jacobian apply should not hide in `remainder`; the data norm is one cheap pass |

## 9. For the maintainer

**Q-1. Where exact mode's misfit transpose is reported.** In Gauss-Newton mode `lm.gradient` runs the misfit transpose
`Π 𝒥ᵀ r`, inside the `gradient` phase; in exact mode `Problem.local_model` runs it once, inside the model build, and
`lm.gradient` only reads it.

- **(a), recommended and written into this plan:** clock it as `gradient` in both modes (§2.5). `gradient` then means
  the same work in both modes and `curvature` is `curvature_precompute` alone, so a GN-vs-exact comparison reads
  directly: the modes differ by `curvature` (setup) and `cg` (per iteration).
- **(b):** report it inside `curvature` (code order, as draft 0 did): exact's `gradient` is then near zero and its
  `curvature` mixes the transpose with the curvature data.

Either is one line in `_local_model_phases`.
