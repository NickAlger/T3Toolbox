# N6 review: the compaction and the uniform loop (review stop)

Reviewer: Claude Opus 5.5, 2026-11-09. I reviewed the uncommitted working tree on top of 056ace8b against:
- the plan, round 7: §2.3, §3.4 (compaction, the whole-loop jit), §4.5, §4.6, §5.3, D-39, §6 T10–T13 and T16;
- `docs/contributor/scan_body_principles.md` and `parameters_not_closures.md`;
- `N6_handoff.md`.

## Summary

The loop is correct, in both modes. On my own scatter of the old frame (gappy masks, garbage in every junk entry), for
seven cases, the uniform loop eagerly, the uniform loop under `use_jit=True`, and the ragged loop give:
- the same pass counts;
- the same movements to ≤ 2.3e-14;
- the same frames by the subspace measure to ≤ 1.9e-14;
- level prefix masks exactly, at the level widths;
- `utv_retract` of the zero tangent equal to X_old to ≤ 4.4e-15;
- `from_frame` variation masks equal to `from_point`'s.

The seven cases are the defaults, tol 1e-10 at cap 14, the zero tensor at d = 4, p = 0, the simultaneous loop,
sharing, and the final `'UL'`. `ufv_compact_frame` holds its claims bitwise on 120 random block-layout frames
compacted to narrower, equal and wider widths.

On the main question, the `functools.partial`-bound loop bodies are **acceptable as they are**: the justification is
already written in the module and holds against both docs.

One test, however, fails on the CI floor leg (jax 0.4.30). That leg runs `pytest tests/` on every push, so this is a
MAJOR (finding 1). The fix is one guard.

## The main question: `functools.partial` around module-level loop bodies

**Verdict: acceptable as is, with the justification documented.** It is documented already, in the section comment
above `ufv_compact_frame`. Two small additions are recommended below. A registered pytree carrying the static half
would add a backend class for no measured benefit.

The reasoning, against the two docs:

1. **Why `partial` is a hazard at all.** `scan_body_principles.md` names `functools.partial` and `lambda` as CAUSES of
   recompilation because eager `lax.scan` / `while_loop` / `cond` key their trace caches on the body's IDENTITY.
   That hazard needs a lax primitive called eagerly with a fresh body. Here that cannot happen:
   - With `use_jit=False`, `_fori`, `common.xwhile` and `_cond` are plain Python loops and branches (`xwhile` takes
     the Python path whenever `use_jit` is False, even on jax arrays). No partial ever reaches a lax primitive.
   - The lax primitives inside the step are its module-level scan bodies (N5), whose identities are stable. T13 (c)
     shows a second eager run issuing 0 backend compilations.
2. **With `use_jit=True`, the partials exist only during the trace of ONE `jax.jit`.** That jit is
   `_jitted(_ufv_aligned_loop_kernel, (2, 3, 4, 5))`, memoized. Its cache key is the kernel plus its static arguments:
   the value-hashed skeleton, `project` by identity, the hashable schedule and `use_jit`.
   - That is the documented remedy for bodies whose bindings are awkward to defunctionalize.
     `parameters_not_closures.md`: "put the jit boundary where a value-based cache already exists, rather than
     hand-defunctionalizing". `_cg_solve` is the precedent.
   - Inside the trace, a fresh partial is traced once as part of the enclosing computation, whatever its identity.
     T13 (b)'s `[1, 0, 1]` compiles confirm it.
3. **Principle 5** (a value that changes between calls belongs in the state). Every bound value is either part of
   that jit's key or a tracer of the current trace:
   - `project`: a static argument;
   - the masks, `shape`, `split`, the output ranks, `sharing`, the colour: from the skeleton or the schedule, so in
     the key;
   - the hint's arrays: tracers rebuilt from the dynamic leaves by `rebuild_static`, so never stale.

   The values that do change between calls ride the state or the dynamic arguments: `tol`, the counter, the stop
   flag and the cap. Nothing that differs between two calls with the same key is captured.
4. **Principle 4** (a cached factory never closes over arrays). Not applicable: nothing is cached by the partials
   themselves. The host mask arrays they bind are static structure, already part of the jit key through the
   skeleton.
5. **`parameters_not_closures.md`'s concern** is that a callable field hashes by identity and hides parameters from
   readers. It does not apply either:
   - the partials are never fields, never cache keys, never returned;
   - the bodies are module-level functions with an explicit `statics` parameter, so the math stays reachable.

**Recommended additions** (NIT 3):
- **(a)** One sentence in `_ufv_aligned_loop_kernel`'s docstring: `use_jit=True` is valid only inside its own
  `_jitted` wrapper, because the bound bodies are fresh objects and an eager call with `use_jit=True` would recompile
  every call. The public function guarantees this; the kernel is private.
- **(b)** For N9: add this case to `scan_body_principles.md`'s section "Where jitting the caller IS the right remedy",
  as its second instance after `_cg_solve`.

## Correctness

- **The schedule** follows §2.3 r7 in both modes, from one core:
  1. `fori` oversampled passes at the working structure;
  2. the truncating pass at the target;
  3. `(UL, RO)` pairs while a whole pair fits, stopping on the pair (`maximum`, so NaN never stops);
  4. the final `'UL'` under a `cond` on the stop flag;
  5. without `alternate`, single `'ULRO'` passes starting from the truncating pass's stop.
- **Carry shapes are invariant.**
  - The start builds P-widths = the counts' maxima; a `None`-output step keeps them (N5).
  - The truncating pass sets the target's maxima, and every later pass keeps them.
  - So the `fori` and `while` carries keep their shapes, which the jit needs. The masks bound for the later passes
    are the truncating pass's output masks, which the later steps reproduce.
- **Return contract.**
  - Movements are `(max_passes,)+C`, NaN after the last pass, in both modes. `(0,)+C` at zero passes; all NaN at a
    no-op level (§2.3).
  - The frame is compacted to `level_widths` with the level's prefix masks.
  - Under `use_jit` numpy inputs are converted (N3b's ruling), the hint stays untouched, and the zero-pass and no-op
    returns are jax too (judgment call 6).
- **`ufv_compact_frame`.**
  - The structural check comes first; junk is appended where a store is narrower than its level.
  - One host index array per edge mask (`argsort(~mask, stable)[..., :W]`) serves every core axis that mask
    governs, exactly as §3.4 lists.
  - The result goes through prefix `ufv_make_frame_masks` and the masking chokepoint.
  - The read-back is bitwise equal and the junk is exactly 0, at narrower, equal and wider widths (my check).
- **Judgment calls.**
  - 2 (the final `'UL'` via `lax.cond`; private `_fori` / `_cond`) and 6: accepted.
  - 4 (a T10 tolerance of 1e-13 for a stack whose elements run at the stack's padded widths): accepted. That is
    roundoff from different SVD sizes, not a defect.
  - 5 (the compile-count repeat negates the old frame, so the same masks with other values): accepted. Masks are
    structure, so a fresh scatter must recompile.
  - 1 and 3: see the main question and the maintainer list.

## Findings

### 1. MAJOR: T13 (c) fails on the jax 0.4.30 floor leg of CI

`tests/test_dispatch.py`, `TestUniformAlignedLoopJit.test_eager_jax_compiles_nothing_twice`, the `finally` clause at
about line 2509.

**Problem.** jax 0.4.30's `jax.monitoring` has `register_event_duration_secs_listener` and `clear_event_listeners`,
but no `unregister_event_duration_listener`. That function arrived later; jax 0.10 has it.
- Run in the `tttt` env (py3.9, numpy 1.22.4, jax 0.4.30), the test fails:
  `AttributeError: module 'jax.monitoring' has no attribute 'unregister_event_duration_listener'`.
- The registered listener also stays alive for the rest of that xdist worker.
- `.github/workflows/tests.yaml` runs `pytest tests/` on `jax==0.4.30`, so CI would go red.

**Fix.** Before registering, skip when the jax at hand cannot unregister:
`if not hasattr(monitoring, 'unregister_event_duration_listener'): self.skipTest('this jax cannot unregister a
duration listener')`.

This mirrors the `_cache_size` skip in T13 (b). Do not fall back to `clear_event_listeners()`, which would drop other
listeners too. Then re-run the four N6 jit and dispatch tests on the floor env. The other three passed there.

### 2. NIT: the kernel's docstring should state its `use_jit` invariant

`_ufv_aligned_loop_kernel`.

**Problem.** `use_jit=True` is only valid inside its `_jitted` wrapper, and the module comment says so only
implicitly.

**Fix.** Add one sentence to the docstring (the main question's recommendation (a)).

### 3. NIT, for N9: the `partial`-bound bodies in the scan-body doc

**Fix.** Record them in `scan_body_principles.md` as the second "jit the caller" instance (recommendation (b)).

## What I ran (env `t3toolbox`: py3.11, numpy 2.x, jax 0.10.2; `OPENBLAS_NUM_THREADS=1` for the scripts)

1. **The loop, three ways,** `n6_check.py`. Seven cases with my own gappy, garbage-filled scatter of the old frame
   (the zero tensor uses `ufv_empty_frame`), the full-rank T3 hint, and both layers' T3 projections.

   | case | passes ragged / eager / jit | vs ragged: frames, movements | eager vs jit: frames, movements |
   |---|---|---|---|
   | defaults d = 3 | 5 / 5 / 5 | 1.2e-14, 3.1e-15 | 1.9e-14, 3.9e-15 |
   | tol 1e-10, cap 14 | 14 / 14 / 14 | 7.6e-15, 6.3e-15 | 8.9e-15, 8.3e-15 |
   | zero old, d = 4 | 9 / 9 / 9 | 1.6e-15, 1.1e-15 | 2.1e-15, 4.4e-15 |
   | p = 0, cap 7, tol 0 | 7 / 7 / 7 | 7.8e-15, 6.8e-15 | 1.6e-14, 4.2e-15 |
   | simultaneous, cap 8 | 4 / 4 / 4 | 1.7e-15, 9.8e-15 | 3.3e-15, 4.4e-16 |
   | sharing `(0,0,1)`, cap 6 | 6 / 6 / 6 | 9.5e-15, 2.3e-14 | 1.6e-14, 1.5e-14 |
   | final `'UL'`, cap 8, tol 0 | 8 / 8 / 8 | 1.4e-14, 1.6e-14 | 1.2e-14, 1.2e-14 |

   On every case: level prefix masks exact (eager and jit); supercores at the level widths; `utv_retract` of the
   zero tangent within 4.4e-15 of X_old (relative); `from_frame(compacted).var_masks == from_point(level).var_masks`.

   Whole-loop jit wall time per first call: 13–25 s at d = 3–4, informational, on a machine loaded by the parallel
   Darcy runs. The plan quotes about 6 s at d = 6 and 24 from the prototype.
2. **Compaction,** `n6_compact.py`. 120 random block-layout frames (gappy in both blocks, garbage, d = 1..4)
   compacted to random widths from the counts upward. **Read-back bitwise equal, prefix masks exact, junk 0: all
   120.**
3. **Tests.** `pytest tests/backend/test_uniform_padded_frame.py tests/test_import_graph.py -n 4`: **43 passed, 191
   subtests**. Doctests of `uniform_padded_frames`: **1 passed**.
4. **Floor env** `tttt` (py3.9, numpy 1.22.4, jax 0.4.30): `tests/test_dispatch.py -k "TestUniformAlignedLoopJit or
   compact_frame_dispatch or uniform_aligned_padded_frame_dispatch"`: **4 passed, 1 failed** (finding 1), in 7.6 min.
   The eager = jit comparison, T13 (b)'s compile count and both T16 rows pass on the floor.

The full suite: the implementer's run was still pending at review time.

## For the maintainer (review stop after N6)

1. **The `partial` question.** My answer is above: keep it, add the docstring sentence, and record the case in
   `scan_body_principles.md` (N9).
2. **The NaN-padded movement record (judgment call 3).**
   - A NaN movement cannot be told apart from a pass that never ran. A NaN movement means NaN content, so the frame
     is NaN as well, which makes the ambiguity mostly academic.
   - Options: state it in N9's docs (the implementer's proposal), or add a pass count to the return, which would
     change §5.3's signature.
   - I lean to the docs.
3. **Whole-loop compile time** on this loaded machine is 13–25 s at d = 3–4, against the plan's ~6 s (d = 6 and 24,
   the prototype). Re-measure unloaded before N10's acceptance numbers.
4. **Plan text.**
   - §4.5's compaction now pads first, so wider levels work.
   - The T10 loop tolerance is 1e-13.
   - The jit tests live in `test_dispatch.py`, per N3b's x64 ruling.
5. **The floor leg.** jax 0.4.30's API is narrower than 0.10's (finding 1). Every new jax-dependent test should be
   run once on the `tttt` env before commit. N7 and N8 add more of them.

## Counts

MAJOR 1, MINOR 0, NIT 2.

VERDICT: REVISE
