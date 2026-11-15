# N6 handoff: the compaction and the uniform loop (review stop)

Slice N6 of `dev/aligned_padded_frame_impl_plan_2026-11-09.md` (§8, round 7 / D-39):

- `ufv_compact_frame` (§3.4, §4.5) and `ufv_aligned_padded_frame(..., use_jit=False)` (§3.4, §4.6, §5.3);
- tests T11, T12 (incl. eager = jit), T13 (b, c), the T10 loop row and the T16 rows;
- N5's nit: the four transport wrappers are annotated.

Implementer: Claude Opus 5.5, 2026-11-09, on top of 056ace8b (N5 committed). Everything is uncommitted; nothing
outside the repository was touched.

## Files

| file | change |
|---|---|
| `t3toolbox/backend/uniform_padded_frames.py` | New public functions: `ufv_compact_frame` and `ufv_aligned_padded_frame` (in `__all__`, one-line docstrings: both get twins in N8). New private helpers: the loop core `_ufv_aligned_loop_kernel`, the pass `_aligning_pass`, the module-level loop bodies (`_oversampled_body`, `_pairs_cond` / `_pairs_body`, `_final_pass` / `_no_pass`, `_single_cond` / `_single_body`), `_record`, and the dispatches `_fori` / `_cond`. Also `_nested_tuple`. The module imports the `ALIGN_*` constants and `_jitted` from `padded_frames`. N5's nit: the four transport wrappers are annotated. |
| `tests/backend/test_uniform_padded_frame.py` | T11 (`TestCompaction`); T12 eager (`TestLoopAgainstRagged`, `TestLoopSchedule`); helpers `full_rank_hint`, `level_of`, `loop_case`, `passes_of`. Still numpy-only. |
| `tests/test_dispatch.py` | `test_compact_frame_dispatch`, `test_uniform_aligned_padded_frame_dispatch`, and `TestUniformAlignedLoopJit` (eager = jit; T13 (b); T13 (c)). They live here because x64 is on module-wide (N3b's ruling). |

No contraction strings: the compaction is gathers and a mask multiply, and the loop calls the step.

## The code

- **`ufv_compact_frame(frame_data, level_widths)`.**
  - Structural check: every real count ≤ its level width.
  - Junk slots are appended where a store is narrower than its level width.
  - Host index arrays `argsort(~mask, stable)[..., :W]`, one `take_along_axis` per rank axis. An edge's index array
    serves every core axis its mask governs: up for U rows and the L/R middle; down for the O middle; left for the
    L first/last and the O first; right for the R first/last and the O last.
  - The masks become `ufv_make_frame_masks(counts, level_widths)`, and junk is zeroed through
    `ufv_apply_frame_masks`.
- **`ufv_aligned_padded_frame`, the public function** (host work).
  - Validates the three non-negative ints, the output boundary, and the target stores (≥ the old ones, ≤ the level
    widths).
  - Under `use_jit` (when jax is present), the old frame's supercores go to jax: N3b's ruling, the optimizers'
    auto-convert. The hint is untouched.
  - Zero passes return the start at the target, compacted, with movements `(0,)+C`. A no-op level returns
    `ufv_compact_frame(old)` with all-NaN `(max_passes,)+C` movements.
  - Otherwise: working ranks per element (`oversampled_ranks` on arrays); `n_over` by the STORES rule over every
    element, capped at `max_passes − 1`; a hashable `schedule`; `partition_static((old_frame_data, hint))`.
  - It then calls the core, eagerly or through `_jitted(_ufv_aligned_loop_kernel, (2, 3, 4, 5))`. Static arguments
    are the skeleton, `project`, the schedule and `use_jit`; `tol` and the dynamic leaves are traced.
  - The output masks are the level's prefix masks.
- **`_ufv_aligned_loop_kernel`: one core for both modes** (the CG precedent: the same cond/body pair drives the
  Python loop and `lax.while_loop`).
  1. The start: `ufv_padded_frame` at the working counts.
  2. `_fori(n_over, …)`: the oversampled passes.
  3. The truncating pass, at the target ranks.
  4. With `alternate`: `xwhile(_pairs_cond, _pairs_body)`, a `'UL'` then an `'RO'` pass per iteration, the
     counter in the state, the stop on the pair (`max(a, b) < tol`; NaN never stops). Then
     `_cond(not stopped and under the cap, _final_pass('UL'), _no_pass)`.
  5. With `alternate=False`: `xwhile(_single_cond, _single_body)` from the truncating pass's stop.
  6. `ufv_compact_frame`.
  - Movements: `_record` writes each pass into a NaN-filled `(max_passes,)+C` array (branch-free, so a traced
    index works).
  - With `use_jit=False` the loops are Python loops and every step runs eagerly. With `use_jit=True` they are
    `lax.fori_loop` / `while_loop` / `cond` inside the one jit.

### Judgment calls (for the review stop)

1. **The loop bodies are module-level; each phase's static structure is bound with `functools.partial`.** The static
   structure is project, the hint, the masks, the split, the output ranks, sharing and the colour.
   - The masks must stay concrete host arrays for the step's host work, so they cannot ride the loop state.
     `scan_body_principles` §4 forbids closing over arrays in a cached factory.
   - The lax loops run only inside the one `jax.jit` of the core, whose cache key is the core's identity plus its
     static arguments. A fresh partial is therefore traced once per compiled program, never per call. Eager runs
     use Python loops, where body identity is irrelevant.
   - The dynamic values that change between calls (`tol`, the counter, the stop flag, the cap) ride the state,
     per principle 5. The hint's traced leaves are bound, because a hint can be a tuple with static masks, which
     cannot ride the state.
   - T13 confirms it: one compile per structure, none on a repeat.
   - **For Nick:** the alternative is a registered pytree carrying the static half (the CG state's `local_model`
     pattern). That would be a new backend class.
2. **The final `'UL'` uses `lax.cond`, as the plan says.** The `fori` / `cond` dispatches are private helpers here;
   `common` has only `xwhile`, and N6 does not name `common.py`.
3. **The NaN record cannot tell a NaN movement from an unrun pass.** The stop-rule test therefore counts the step's
   calls (`stop_after`) and checks that the record equals those movements followed by NaN. Callers needing the pass
   count with possible NaN movements cannot read it off the record. **For the docs (N9).**
4. **T10's loop row compares the stack with its elements at 1e-13 for frames** (movements 1e-14).
   - With different ranks per element, the stack runs every element at the stack's padded widths (the maxima),
     while each element's own call runs at its own. Six passes then agree to roundoff, measured 1.0e-14, not to
     the plan's 1e-14.
5. **T13 (b)'s second run negates the old frame.** The masks are static structure, so a new scatter is a new
   structure: it recompiles, correctly. The negated frame keeps the masks and changes the values, and the run also
   uses another `tol` and another hint of the same ranks.
6. **Conversion under `use_jit` also covers the eager returns.** Zero passes and the no-op level return jax arrays
   from numpy input, so all `use_jit` outputs agree in type. The whole-loop jit itself would convert numpy inputs
   anyway; the explicit conversion only matters for these returns, and the T16 row pins them.

7. **The compaction pads first when the level is wider than the frame** (for the plan update, §4.5). The
   prototype's `compact` assumed `W ≤` the frame's widths. The library appends junk slots, then gathers, so any level
   width at least the real counts works (tested at wider and mixed widths).
8. **The stacked-loop tolerance is 1e-13** (for the plan update, T10). See 4.

### Round 2 (the review: REVISE on one item)

- **MAJOR fixed:** `test_eager_jax_compiles_nothing_twice` now skips, naming the jax version, when
  `jax.monitoring.unregister_event_duration_listener` is missing (jax 0.4.30, the CI floor). It is the guard
  idiom of the `_cache_size` checks and `test_contractions_sharding`.
- **The partial binding is accepted.** The kernel's docstring now states the `use_jit` invariant: the partials reach a
  lax primitive only under `use_jit=True`, inside the one memoized jit keyed on the skeleton, `project`, the schedule
  and `use_jit`; eagerly the loops are Python loops.
- **Floor env** (`tttt`: jax 0.4.30, numpy 1.22, Python 3.9).
  - Every new jax test passes: the dispatch rows of N3b–N6 for the padded frames, `TestAlignedLoopJit` and
    `TestUniformAlignedLoopJit`. That is 16 passed, with 1 skipped by name, the listener test, as intended.
  - The new numpy files (`test_uniform_padded_frame.py`, `test_padded_frame.py`, `test_linalg.py`, the import graph,
    the contraction interpreter) pass: 135.
  - On the main env the guarded test still runs and passes.

## Verification

- **T11:**
  - masks exactly `arange(W) < count` at the level widths, at wider widths and at mixed widths;
  - the read-back bitwise equal to the input's (real entries moved bitwise); junk exactly 0;
  - a stack equals its elements bitwise;
  - `utv_retract` of the zero tangent at the compacted frame gives X_old (≤ 1e-12 relative);
  - `from_frame(compacted).var_masks == from_point(zero-padded point).var_masks`;
  - a count above its width raises.
- **T12 eager, against `fv_aligned_padded_frame`:**
  - Setup: one full-rank hint, projected by `tv_project_t3_onto_tangent_space` and
    `utv_project_ut3_onto_tangent_space`; the old frame scattered gappy with garbage.
  - Cases: the defaults, tol 1e-10 (13 passes), the zero point, non-minimal + uneven, p = 0 at d = 2,
    simultaneous, sharing.
  - Results: the same pass counts, movements ≤ 1e-12 (measured ≤ 1.1e-14), NaN after; spans ≤ 1e-9 (measured
    ≤ 1e-14); the level's prefix masks exactly; junk 0.
  - Also: the no-op level, zero passes (bitwise the compacted start, no projection, p = 0 and 5), the recorded
    schedule (incl. `max_passes = 2` at working = target → `ULRO@t, UL`, and the cap counting the truncation), the
    pair stop rule (the ragged cases, incl. NaN), and structural errors.
- **T12 eager = jit** (`TestUniformAlignedLoopJit.test_eager_equals_jit`, 52 s):
  - Cases: the seven T12 cases plus "the final UL" (tol 0, cap 10).
  - The same pass counts, NaN slots identical, frames ≤ 1e-12, movements ≤ 1e-10.
  - The defaults stop early at 9; the final-UL case runs 10.
- **T13:**
  - (b) compiles `[1, 0, 1]` (first; same structure with other values, `tol` and hint; another `max_passes`), 16 s;
  - (c) eager jax: the first run issues the backend compilations (about 700 at d = 3), a second run at the same
    structure issues 0 (a `jax.monitoring` listener), 44 s.
- **T16:**
  - `ufv_compact_frame`: numpy, eager jax and jit, bitwise equal (gathers); masks host; a stack.
  - `ufv_aligned_padded_frame` at d = 2:
    - numpy → numpy;
    - jax eager → jax; jax `use_jit` → jax;
    - numpy `use_jit` → jax (converted), incl. the zero-pass and no-op returns;
    - movements = numpy's ≤ 1e-10.
  - 41 s, mostly eager jax's first calls.
- **Mutations, 14, all killed:**
  - compaction: junk first; junk kept; count check;
  - the loop: a pair past the cap; no final `'UL'`; the final `'UL'` even when stopped; the stop on `'RO'` only;
    the cap ignoring the truncation; no stores skip; no truncating pass; the no-op branch; the simultaneous stop;
  - jit: never jit; no conversion.
- **Informational timings:** see the next section.

## First-call times (informational, T13 (c))

Each mode ran in a fresh process. Setup: the whole loop at the defaults with tol 0 (10 passes), shape `(5,)*d`, old
ranks 2, target ranks 3, a hint of Tucker rank 4 / TT rank 6. The machine was quiet: load about 4 after the
suite.

| d | numpy (first / second) | jit, whole loop (first / second) | eager jax (first / second) |
|---|---|---|---|
| 6 | 0.2 s / 0.16 s | 8.8 s (15.1 s at load 6) / 0.02 s | 41.5 s / 0.57 s |
| 24 | 0.6 s / 0.54 s | 9.0 s / 0.08 s | 42.6 s / 0.64 s |

- The prototype's whole-loop compile was about 6 s at both sizes; the library's is about 9 s.
- Eager jax's first call is op-by-op compilation (about 700 compilations at d = 3, per T13 (c)).
- The jit's second call is the fastest of all.

## Full suite

`PYTHONPATH=$PWD python -m pytest tests -q -n auto`: **1,308 passed, 4 skipped, 48,631 subtests passed** in
19 min 54 s (on a loaded machine; N5: 1,292 passed). The one warning is the existing `adam` one. This run predates
round 2's two edits (the floor skip guard and a docstring); the touched test was rerun after them.

## Plan problems

- §4.5 says "The compaction: `P/uniform_loop.compact`". The prototype assumed `W ≤` the frame's widths; the library
  pads first, so wider levels work (tested).
- The N6 row of §8 names `tests/test_dispatch.py` but not the moved jit tests' reason. They are there because of
  N3b's x64 ruling.
- Judgment call 3 (the NaN record) should be stated in the N9 docs.
