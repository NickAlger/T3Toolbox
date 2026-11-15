# Critic review r1 — `dev/newton_timing_plan_2026-11-10.md` (draft 1)

Read for this review: `CLAUDE.md`; `dev/HANDOFF.md`'s 2026.5.0 bundle paragraph (T-1..T-7 as adopted);
`backend/optimizers.py` (`Problem.local_model`, `padded_start`, `_cone_reference`, `gradient_descent`, `NewtonInfo`,
`newton_cg` whole, `_mc_sgd_step`/`_adam_step`, which call `local_model` under jit); `backend/geometry.py` (both
`aligned_padded_frame` members); `backend/padded_frames.py` (`_aligning_pass`, `fv_aligned_padded_frame`);
`backend/uniform_padded_frames.py` (`_aligning_pass` to `ufv_aligned_padded_frame`, the kernel and its partial-binding
note); `backend/optimizer_display.py`; `tests/test_dispatch.py` (`test_eager_jax_compiles_nothing_twice`, the
monitoring precedent); `parameters_not_closures.md`, `scan_body_principles.md`; the switch plan (draft 2) for §2.7.

**(checked)**:
- jax 0.10.2's three event names are in `jax/_src/dispatch.py:59-61`. The trace event fires only under
  `core.trace_state_clean()` (`pjit.py:524-528`).
- The floor's jax 0.4.30 `jax.monitoring` has `register_event_duration_secs_listener` and `clear_event_listeners`, but
  no `unregister_event_duration_listener`, so `None` there is right.
- `jax.block_until_ready` exists on 0.4.30 and accepts a mixed tree `(jax array, numpy array, int, None)`.

## Summary

T-1..T-7 are applied faithfully. T-6 is read as "a keyword whose default keeps today's arity", which is the natural
reading. The non-breaking claim with `timing=False` holds:
- `_local_model_phases(..., None)` runs today's operations in today's order.
- `_padded_start(timing=False)` calls the member without the keyword.
- `_cone_reference`'s third element is dropped by `gradient_descent`.
- `NewtonInfo.timing` is excluded from `_NEWTON_SCALAR_FIELDS`.

The compile accounting is sound and correctly `None` on the floor. Synchronization is added only under `timing=True`,
and every model phase passes its outputs.

**Q-1:** I agree with (a). It also matters to the switch plan: with (a), a Gauss-Newton-phase point's tracking setup
is `seconds.curvature` alone.

**W-choices:**
- W-1, W-2, W-4, W-5, W-7: sound.
- W-6: clean enough. The clock is eager-only and `None` under jit, so it adds no cache-key and no staleness hazard,
  but it needs the three guards of finding 3.
- W-3: needs a narrower probe (finding 2).

There is no MAJOR finding. There are four MINOR findings and four NITs.

---

## MINOR

### 1. MINOR — §2.7: stale against the switch plan (draft 2)

The switch plan has five per-row fields: `switch_pKp`, `switch_e_gn`, `switch_e_n`, `switch_count` and `switch_fired`.
They are excluded from `_NEWTON_SCALAR_FIELDS`, as `timing` is, and `_newton_scalar_record` appends them only on
switch-mode rows (`switch_fired is not None`). So they are not always-present history fields that are `None` outside
switch mode, and the order of `NewtonInfo` fields does not touch the history at all. The switch also needs one phase,
agreed in its review loop.

**Fix.** Rewrite §2.7:
- Add `PhaseSeconds.tracking: Optional[float]`: the switch's `curvature_quadratic` on a countable Gauss-Newton-phase
  row, `None` elsewhere. It sits after `forms`, so `forms` keeps its meaning (slope, `pHp`, `‖p‖`).
- The Gauss-Newton-phase point's exact build lands in `curvature` (its transpose in `gradient`, per Q-1(a)).
- The firing's `_cg_core` compile lands in the first exact row's `compile_seconds.cg`.
- §4's `time` line gets a `track` cell, shown only when the phase is set.
- The switch plan implements the lap; this plan adds the field.
- Docs: timing keeps §4.13 and creates CHANGELOG's `### Added`. The switch takes §4.14 and appends (already in switch
  draft 2).

### 2. MINOR — W-3, §3.1: `**kwargs` is not evidence that a member accepts `timing`

A member written as `def aligned_padded_frame(self, x0, x_old, project, hint, **kwargs)` typically forwards
`**kwargs` to `fv_aligned_padded_frame`. That function does not take `timing`, so it raises `TypeError`, or the member
drops the keyword and returns a 2-tuple, which `newton_cg` then fails to unpack as three. Either way, a diagnostic
makes a working run fail, which is what W-3 exists to prevent.

**Fix.** Pass `timing=True` only when `inspect.signature(member).parameters` names `timing` explicitly. Otherwise call
without it and report `alignment_passes=None`. Add a `**kwargs`-forwarding custom member to T7's custom-member case.

### 3. MINOR — W-6, §2.6: the uniform clock slot needs three guards

The list in the pass statics is acceptable: it is read only on the eager path (Python loops, nothing traced), and
under jit the slot is `None`, so the jitted program and its cache key are today's (T6). Three things are missing:

- **(a) A structural guard.** `_ufv_aligned_loop_kernel` raises when `clock is not None and use_jit`. Under a trace
  the append would record tracing time once per trace, and `block_until_ready` would meet tracers. "Never passed under
  jit" should be enforced, not just observed.
- **(b) The documented exception.** The kernel's docstring accepts the `functools.partial` binding "in this instance
  only and sets no precedent". Extend that note to name the clock slot and why it is eager-only, so the exception
  stays one, written down in one place.
- **(c) Pass 1's start.** Each pass is clocked "after `tree_block_until_ready((frame, movement))`" at its end. Pass
  1's interval also contains the start frame's (`fv_padded_frame` / `ufv_padded_frame`) asynchronous tail unless the
  start is synchronized before pass 1's clock starts. Define each pass's interval as [synchronized start,
  synchronized end], and synchronize the start frame once, on the ragged and uniform loops alike.

### 4. MINOR — T-1, §2.5, T3: a `Problem` subclass that overrides `local_model` is bypassed under `timing=True`

With timing on, `newton_cg` calls `_local_model_phases(problem, ...)` instead of `problem.local_model(...)`. A user's
`Problem` subclass that overrides `local_model` then runs the stock build with `timing=True` and its own build without
it, so "timing changes no number" fails silently. The tests already subclass `Problem` (`_OffsetProblem` overrides
`objective`), so it is a supported pattern.

**Fix.** Use `_local_model_phases` only when `type(problem).local_model is Problem.local_model`. Otherwise call the
override, lap the whole build once (say which phase carries it, with the split phases `None`), and document it. Add
the subclass case to T3.

---

## NIT

1. **§2.4.** Work that `remainder` dispatches asynchronously runs inside the next lap and is charged there. Today
   that is `neg_g = corewise_scale(g, -1)` before `_cg_solve`; it is tiny, but say which lap it is in. Likewise, async
   jax work a custom callback dispatches without waiting lands in later phases. One sentence in the caveats covers
   both.
2. **T2.** "Σ remainder ≤ 0.1·Σ window + 0.02 s" is a wall-clock assertion in the always-on suite under `-n auto`.
   Keep the tiling identity and `remainder ≥ −1e-9`, and move the 10 % bound to the slow tier or to P3's measured
   figures. Otherwise, at minimum, run it on a problem whose window is well above Python's per-row overhead.
3. **§3.1.** `_PhaseClock` and `_CompileSeconds` are mutable host-side accumulators, not parameter-holding backend
   classes. Say so in one line, so a reviewer does not apply `parameters_not_closures.md` to them. They never ride
   as aux and never enter a trace.
4. **§2.2.** `reference_passes` is part of row 0's `reference`, and `alignment_passes` of `alignment`. State in §3.2
   that `stats['timing']['phases']` is computed without those inner tuples, so nothing is counted twice.

VERDICT: REVISE
