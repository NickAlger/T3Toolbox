# N3b review: `use_jit` and the pair stop rule on the ragged loop

Reviewer: Claude Opus 5.5, 2026-11-09. I reviewed the uncommitted working tree on top of c5e1f253 against:
- the plan, round 7: §2.3 (the pair rule), §3.1(6), §4.3, §5.1, D-39, §6 T6/T13/T16;
- `N3b_handoff.md`.

## Summary

The pair rule is implemented exactly as §2.3 (r7) words it. That covers:
- the truncating pass outside the pairs;
- the stop tested only after a whole `('UL', 'RO')` pair;
- the odd remainder as one final `'UL'`, run only when not stopped;
- `alternate=False` unchanged.

My own property check agrees with it on 3,000 random schedules. The NaN rule is sound. The jitted step is the
library's `_jitted` pattern, and the two N4 NITs are fixed.

One item needs a ruling. With numpy inputs, `use_jit=True` silently runs eagerly. The plan's text says exactly that,
but it contradicts the precedent D-39 cites, the library's documented "loud, never silent" jit policy, and what the
uniform loop's whole-loop `jax.jit` will do with the same inputs (finding 1).

## Style (checked first)

- **The wrapper.** `_jitted(fn, static_argnums)` is a module-level, `lru_cache`d `jax.jit` getter keyed on the
  function object and an int tuple: parameters, not closures. It is a twin of `optimizers._jitted`, which this
  module may not import (§3.4). jax is imported inside it.
  - `_STEP_STATICS = (2..7)` names the old ranks, the output ranks, `sharing` and `update`.
  - `_aligning_pass` passes them positionally, and `sharing` is turned into a tuple so that it hashes.
  - The memoization is asserted (`_jitted(f, s) is _jitted(f, s)`). This matters because on jax 0.10 a missing
    memo would not show in `_cache_size` (judgment call 5).
- **Helpers.** `_aligning_pass` and `_peak` are module-level, fully annotated, with one-line docstrings.
- **No plan labels.** §5.1's "(D-39)" was dropped from the `use_jit` comment. A grep of the diff finds none.
- **N4 NITs fixed.**
  - `ufv_padded_frame`'s `sharing:` line is aligned, with the blank line before the option group (§5.3).
  - `_coordinate_seeds` and `_store_completion` have one-line docstrings.

## The stop rule against §2.3 (r7)

- **Before the pairs.** The `n_over` oversampled `'ULRO'` passes and the truncating `'ULRO'` pass run
  unconditionally, so no stop happens during oversampling or on the truncating pass.
- **`alternate=True`.**
  - `('UL', 'RO')` pairs run while `len + 2 ≤ max_passes`.
  - The stop is tested after each pair: the maximum over the pair and the stack must be below `tol`.
  - Then `if not stop and len < max_passes`, one final `'UL'` runs. This tests the stop flag, not parity, so
    `max_passes` is reached when nothing stops, as §2.3 says.
  - With the defaults the run is 2 + 1 + 3 pairs + the final `'UL'` = 10 passes.
- **`alternate=False`.** `'ULRO'` passes, stopping after the first pass, from the truncating one on, whose movement
  is below `tol`. That is the r1 rule, which r7 leaves unchanged.
- **NaN** (judgment call 2). `_peak` takes `np.max` over the stack and then over the pair, so a NaN propagates and
  `NaN < tol` is False. A NaN anywhere in a pair, or in a stack element, never stops the loop, under either rule.
  The flaw it fixes was real: Python's `max` keeps the first of incomparable values, so N3's `(lo, NaN)` could stop
  the loop. "Never stops" is the reading consistent with "both movements below tol".
- **The oracle.** `dense_loop` follows the same rule, written separately, and T6 runs both to an early stop with the
  default tol: same pass counts, movements ≤ 1e-9.

## Findings

### 1. MINOR (needs a ruling): numpy inputs with `use_jit=True` silently run eagerly

`padded_frames.py`, `jitted = use_jit and jax_or_warn(...) and tree_contains_jax(old_frame)`. Judgment call 1.

**What the plan says.** The implementation follows the plan's literal wording: §4.3, and §5.1's comment "jax inputs:
the host loop calls a memoized jitted step; else eager". But the plan also says:
- D-39 and §3.1(6): "Jit is under the user's control, **as everywhere in the library** … the `newton_cg` / `mc_sgd`
  / `adam` precedent".
- That precedent is the BACKEND `optimizers._prepare_jit_inputs`, whose docstring reads: "**auto-convert** …
  rather than silently dropping the request … Loud, not silent: the 2026.1.0 silent eager drop is what this
  replaces".
- The uniform loop's `use_jit=True` is a whole-loop `jax.jit` (§3.4). `jax.jit` takes numpy arrays and returns jax
  ones, so on the same numpy inputs the uniform layer WILL jit and return jax, while the ragged layer will silently
  stay eager and numpy.
- So the ragged behaviour re-creates the silent drop, and it makes the two layers disagree on one flag.

**Recommendation.** Mirror `_prepare_jit_inputs`. When `use_jit` and `jax_or_warn(...)`, convert the old frame with
`tree_to_jax`; the hint stays opaque and is handed on as is. The library's projections infer jax from the frame, and
`jax.jit` converts a numpy variation tree anyway. Without jax, the loop warns and runs eagerly, as now.

**Checked by hand.** I converted only the old frame (`tree_to_jax`), kept a numpy hint, and ran the library's dense
projection with `use_jit=True` at `(3,4,5)` for 6 passes. Every leaf comes back jax, and the movements equal numpy's
to 2.6e-16.

**Size.** About three lines. Flip the `('numpy', True, False)` subtest in `TestLoopJit` and the numpy leg of the T16
row to expect jax.

**For the session.** If Nick rules that the plan's literal text stands, this finding drops to a doc note for N9 (say
it in the "jit or eager" section) and the verdict becomes APPROVE.

### 2. NIT: `TestLoopJit` costs about 60 s on one worker

`tests/backend/test_padded_frame.py`, `TestLoopJit.test_eager_equals_jit`, which runs T6's six cases at their caps,
two of them at 40 passes.

**Problem.** Under `-n auto` this is a single long test that sets the suite's tail.

**Fix.** Capping the jit comparison at about 12 passes would keep every pass type: oversampled, truncating, pairs,
the final `'UL'`, and an early stop.

### 3. NIT: the jax x64 flag is toggled process-wide in the numpy-only test file

`TestLoopJit` toggles `jax_enable_x64` in `setUpClass` / `tearDownClass`.

**Why it is fine today.** It is restored, and xdist keeps a class's tests together. But it puts a global-state
toggle in a file the testing strategy keeps numpy-only.

**Alternative.** Move the class into `tests/test_dispatch.py`, where x64 is on for the whole module, or note the
choice in the docstring.

## What I ran (env `t3toolbox`: py3.11, numpy 2.x, jax 0.10.2)

1. **The schedule against my own reading of §2.3 r7,** `n3b_rule.py`. 3,000 random schedules: p ∈ {0, 5};
   `oversampled_passes` 0..3; `max_passes` 1..12; alternate on and off; prescribed movements with 5 % NaN; a quarter
   of them stacked `C = (2,)`. The step is mocked, and I compare the recorded update sequence and pass count.
   **0 mismatches.**
2. **Tests.** `pytest tests/backend/test_padded_frame.py tests/backend/test_uniform_padded_frame.py
   tests/test_import_graph.py -n 4`: **74 passed, 292 subtests** (83 s). The slowest:
   - `test_eager_equals_jit`, 60.5 s;
   - `test_against_the_dense_loop`, 31.8 s;
   - `test_compile_count`, 29.2 s.
3. **Dispatch.** `tests/test_dispatch.py -k "aligned_padded_frame or align_padded or padded_frame_dispatch"`: **4
   passed, 12 subtests**. That includes the loop's row with `use_jit` (jax in → jax out; numpy in → numpy out).
4. **Doctests** of `padded_frames` and `uniform_padded_frames`: **2 passed**.
5. **The recommended auto-convert,** `n3b_autoconv.py`: see finding 1. It works as described.
6. **A plan-label grep of the diff:** none.

The full suite was not run; the implementer reports 1,277 passed, 0 failed.

## Judgment calls

- 1: see finding 1.
- 2 (NaN never stops), 3 (frames compared by the subspace measure), 4 (the eager reference is numpy), 5 (memoization
  asserted directly) and 6 (`dense_loop` follows the pair rule): accepted.

## Counts

MAJOR 0, MINOR 1 (a ruling), NIT 2.

VERDICT: REVISE
