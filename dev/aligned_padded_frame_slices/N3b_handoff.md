# N3b handoff: `use_jit` and the pair stop rule on the ragged loop

Slice N3b of `dev/aligned_padded_frame_impl_plan_2026-11-09.md` (round 7: §2.3, §4.3, §5.1, D-39):

- `fv_aligned_padded_frame(..., use_jit=False)` with a memoized jitted step;
- the PAIR stop rule;
- T6 (pair rule; eager = jit), T13 (a), and the T16 row with `use_jit`;
- the two N4 review NITs.

Implementer: Claude Opus 5.5, 2026-11-09, on top of c5e1f253. Everything is uncommitted; nothing outside the repository
was touched.

## Files

| file | change |
|---|---|
| `t3toolbox/backend/padded_frames.py` | The loop changes (below). New private helpers: `_jitted`, `_STEP_STATICS`, `_aligning_pass`, `_peak`. `import functools as ft`. The module docstring names the pairs and `use_jit`. |
| `t3toolbox/backend/uniform_padded_frames.py` | The N4 NITs: `ufv_padded_frame`'s `sharing:` line aligned, with the blank line before the option group (§5.3); `_coordinate_seeds` and `_store_completion` have one-line docstrings, and the order argument moved into a comment. |
| `tests/backend/_padded_frame_oracles.py` | `dense_loop` rewritten from §2.3's text to the pair rule. |
| `tests/backend/test_padded_frame.py` | T6 updates; the jax-absent `use_jit` test; the module docstring. Numpy-only again (round 2). |
| `tests/test_dispatch.py` | `test_aligned_padded_frame_dispatch` gains `use_jit=True`; new `TestAlignedLoopJit` (eager = jit, the step choice, T13 (a)), moved here in round 2 because x64 is on module-wide; `import unittest.mock`. |

## The loop

- **`use_jit`.**
  - Round 2: when `use_jit and jax_or_warn('use_jit=True')`, the old frame is converted with `tree_to_jax` (the
    optimizers' auto-convert) and the host loop calls `_jitted(tv_align_padded_frame, (2, 3, 4, 5, 6, 7))`. The
    hint is opaque and handed on untouched. The conversion happens before the zero-pass and no-op returns, so every
    leaf comes back jax.
  - The static arguments are the old ranks, the output ranks, `sharing` and `update`. `sharing` is turned into a
    tuple so that it hashes.
  - Otherwise the step runs eagerly. Without jax, `jax_or_warn` warns once and the loop runs eagerly on numpy, as the
    optimizers do. `project` and the loop are always eager.
- **`_jitted(fn, static_argnums)`.** The module's own `lru_cache`d `jax.jit`, with parameters rather than closures,
  as optimizers' `_jitted` is written. jax is imported inside it.
- **The schedule (§2.3).**
  1. `n_over` oversampled `'ULRO'` passes, then the truncating `'ULRO'` pass.
  2. With `alternate`: `('UL', 'RO')` pairs while a whole pair fits under `max_passes`. The stop is tested after each
     pair: both movements below `tol`, as the maximum over the pair and the stack.
  3. If the cap leaves one slot and the loop has not stopped, one final `'UL'` pass runs.
  4. With `alternate=False`: as before. `'ULRO'` passes stop after the first pass, from the truncating one on, whose
     movement is below `tol`.
  - With the defaults the run is 2 oversampled + 1 truncating + 3 pairs + the final `'UL'` = 10 passes.
- **`_aligning_pass`** calls `step(frame, project(frame, hint), *old, *out, sharing, update)` positionally, because
  the jitted step's statics are positional.

### Judgment calls

1. **Numpy inputs with `use_jit=True` are CONVERTED** (round 2; the coordinator's ruling on review finding 1).
   - Round 1 ran them eagerly with numpy out, following the plan's literal text ("jax inputs: … a memoized jitted
     step; else eager", §4.3, §5.1).
   - The ruling: D-39 says "as everywhere in the library". The backend optimizers auto-convert
     (`_prepare_jit_inputs`) because the silent eager drop was once a defect, and the uniform whole-loop `jax.jit`
     will jit numpy inputs too, so the layers must agree. The plan's text will be corrected.
   - Built: the old frame goes through `tree_to_jax`; the hint stays opaque and untouched. A numpy hint works through
     the library projections, which infer jax from the frame, and `jax.jit` converts numpy variations. Every leaf
     comes back jax. In the T16 row the movements equal numpy's (≤ 1e-12 asserted).
   - With jax absent: exactly the optimizers' behaviour, a one-time `use_jit=True` RuntimeWarning, then eager on
     numpy. A new numpy-only test simulates it as `test_optimizers` does, by setting `common.jax_available = False`.
     It checks the warning, numpy leaves, the eager run's result bitwise, and no warning when `use_jit=False`.
2. **NaN never stops the loop.**
   - Found by mutation: `max(float(...) for m in ...)` ignores a NaN in second place, because Python's `max` keeps
     the first of incomparable values. So `(lo, NaN)` stopped a pair, and N3's code had the same flaw.
   - `_peak` now uses `np.max`, which propagates NaN. The simultaneous branch uses `not stop` rather than `>= tol`.
   - Tests cover NaN in either place of a pair, in a stack element, and in the simultaneous loop.
3. **Frames eager vs jit are compared by the plan's measure** (the largest principal-angle sine per directed edge),
   not entrywise.
   - The step's gauge follows its SVD's: `pinned_completion`'s sign rule ties each column's sign to its seed's.
   - The jitted and eager kernels can flip a column: on 'two-colour, p=0' the frames differ entrywise by 1.0 (jit
     vs numpy) and 2.0 (jit vs eager jax), while the subspaces agree to 1.3e-15.
4. **The eager reference is the numpy loop.**
   - Eager jax costs about 100 s over T6's six cases (op-by-op compilation), against about 43 s for jit and 0.6 s
     for numpy. I measured eager jax too: it agrees with jit as well (subspaces ≤ 2e-15, movements ≤ 1.1e-15).
   - That jax eager equals numpy is the dispatch row's claim; this slice extends that row.
5. **Memoizing the wrapper is the house contract, but jax 0.10 does not need it.** A mutant without `lru_cache`
   passed the cache-size test, because jax 0.10 shares compiled programs across `jax.jit` wrappers of one function
   object (checked). The test now also asserts the memoization (`_jitted(f, s) is _jitted(f, s)`).
6. **`dense_loop` follows the pair rule** too, written from §2.3. A new T6 test runs both loops with the default tol
   and a 40-pass cap on three cases: they stop after the same pass, early, with movements ≤ 1e-9.

## Verification

- **T6 against the dense loop:** unchanged at tol 0. The pass sequences are the same under both rules when nothing
  stops. With the default tol, the pass counts match the oracle.
- **T6 schedule, recorded call by call.** As before, plus `max_passes = 2` at working = target → `ULRO@t, UL`. A stop
  after the first pair leaves exactly `ULRO ×3, UL, RO`.
- **The stop rule on prescribed movements:**
  - the first pair stops at 5;
  - the truncating pass never counts;
  - two low movements straddling two pairs never stop (10);
  - no stop during oversampling;
  - the stack maximum decides;
  - the cap of 10 runs 3 pairs and a final `'UL'`; an even cap of 9 runs no final `'UL'`;
  - cap 6: stopped at 5 leaves no final `'UL'`, not stopped runs one;
  - simultaneous as before;
  - NaN in five places.
- **Eager = jit** (`tests/test_dispatch.py::TestAlignedLoopJit`, x64 on module-wide).
  - T6's six cases, capped at 12 passes (round 2). They keep every pass type: oversampled, truncating, pairs, the
    final `'UL'` (two cases run 12 = 3 + 4 pairs + 1 and 1 + 5 pairs + 1), and early stops (d = 4 at 11; the
    simultaneous case at tol 0.1, at 8). The pass counts are asserted.
  - Results: same pass counts; frames ≤ 1e-12; movements ≤ 1e-10; every leaf of the jitted run is jax.
  - The sharing case also runs on numpy inputs (converted).
- **The step choice:** the step and `_jitted` are mocked.
  - `use_jit=True` gets `_jitted(tv_align_padded_frame, _STEP_STATICS)` on jax inputs AND on numpy ones, and the
    step sees jax frames.
  - Without `use_jit`, `_jitted` is never touched, and the frame keeps its type.
- **T13 (a):** a 10-pass jitted run at a structure no other test uses compiles 4 programs (≤ 4 asserted, > 0); a
  second run with other values compiles 0; the wrapper is one object.
- **T16:** the loop's row runs jax eager, jax `use_jit=True` (jax leaves, movements = numpy ≤ 1e-10), and numpy
  `use_jit=True` (converted: jax leaves, movements = numpy ≤ 1e-12).
- **T4:** the cycle case's two-colour loop still converges at pass 59 (last pair 7.2e-9, 5.1e-9).
- **Mutations, 11, all killed** (two first survived and led to judgment calls 2 and 5):
  - the per-pass (old) stop; no final `'UL'`; the final `'UL'` even when stopped; the pair test on `'RO'` only;
  - simultaneous over all passes; NaN stopping; `_peak` dropping a later NaN;
  - never jit; jit on numpy inputs; jit without `use_jit`; the wrapper unmemoized.
  - (Round 1's "jit on numpy inputs" mutant is now the specified behaviour.) Round 2 adds 4 more, all killed: no
    conversion; never jit; jit without `use_jit`; no jax-absent warning.
- **Round 2 reruns:** `tests/test_dispatch.py`, `tests/backend/test_padded_frame.py` and `tests/test_import_graph.py`
  in full: 127 passed, 243 subtests. The full suite was not rerun after round 2 (round 1: 1,277 passed).
- **Costs:** eager = jit is 43 s and the compile count 12 s, on one worker.
  - The 12-pass cap barely moves it (44 s before), because the time is compilation, not passes: T6's six cases are
    six distinct static structures, about 16 step programs at roughly 2.5 s each.
  - Halving it would mean dropping two structures (d = 4 and the empty old frame, about 22 s). I kept all six as
    "T6's cases"; it is the coordinator's call.

## Full suite

`PYTHONPATH=$PWD python -m pytest tests -q -n auto`, on the final code: **1,277 passed, 4 skipped, 48,390
subtests passed** in 11 min 17 s (N4: 1,271 passed). The one warning is the existing `adam` one.

## Plan problems

- None blocking. §2.3's pseudo-code and §4.3 match what is built. The plan's NaN behaviour is unstated. "Never
  stops" is the reading consistent with "both movements below tol".
