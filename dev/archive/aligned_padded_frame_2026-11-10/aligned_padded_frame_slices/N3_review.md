# N3 review: the ragged alignment loop (review stop)

Reviewer: Claude Opus 5.5, 2026-11-09. I reviewed the uncommitted working tree on top of N2 (196fcfa2) against:
- the plan: §2.3, §4.3, §5.1, §6 T4/T6/T16, §12;
- the prototype `P/ragged_loop.py`;
- `N3_handoff.md`.

## Summary

The loop follows §2.3 line by line. It equals the prototype loop to 2.7e-15 (frames) and 5.7e-15 (movements), with
the same pass counts, on a random d = 4 case and on the cycle case, under both schedules, at p = 0 and 5. T4 and T6
are thorough.

One gap needs a fix in this slice: no test checks that the loop's working start uses `seed`. A mutation that drops it
survives. The rest is notes for the maintainer at this review stop.

## Style (checked first)

- **Types.** Basic types in and out. The old frame is read for its stores because it is used. The return is
  `(frame, tuple of C arrays)`.
- **`hint` and `project`.** `hint` is typed `Any` and handed to `project` unread. `project(frame, hint)` is called
  exactly as given, once per pass, before the step. At zero passes and in the no-op case it is never called; the
  tests check this with a raising `project`.
- **Code shape.** It is a host loop, as §4.3 says. There is no contraction and no array arithmetic beyond the stop's
  host `max`, and the docstring is one line.
- **Signature.** §5.1 verbatim, minus the "(D-38)" label (ruling). My grep for plan labels finds none in
  `padded_frames.py`. (It hits only "T3" and "N0", which are not labels.)
- **Factoring.** `_require_contains` is factored out with `fv_padded_frame`'s message unchanged, and T5 still passes.
- **N2 finding 1 is fixed in the commit.** `_fold` now takes `core_widths` as ints.

## The schedule against §2.3, line by line

- **Working ranks.** `working = ranks.oversampled_ranks(shape, target, p, sharing)`, and `F_0` is `fv_padded_frame`
  at the working ranks with `sharing` and `seed`.
- **Passes at the working ranks.** For `it < n_over`, the step runs at the frame's own (working) stores with
  `'ULRO'`.
- **The truncating pass.** At `it = n_over` the step outputs at the target with `'ULRO'`.
- **After it, with `alternate`.** `'UL'` when `it − n_over` is odd and `'RO'` when even; with `alternate=False`,
  `'ULRO'`.
- **Stop with `alternate`.** It stops when `it > n_over` and the movements of passes `it−1` and `it` are both below
  `tol`. At `it = n_over + 1` the pair is the truncating pass and the first `'UL'` pass, as §2.3 says.
- **Stop without `alternate`.** It stops when `it ≥ n_over` and the pass's movement is below `tol`.
- **Stack maximum.** Both stop rules compare `np.max` over the stack, and the cap is the loop bound.
- **The cap counts the truncating pass.** `n_over = min(n_over, max_passes − 1)`.
  - With `oversampled_passes ≥ max_passes ≥ 1`, the last pass truncates.
  - `max_passes = 1` runs one truncating `'ULRO'` pass, whether working ≠ target or working = target.
- **`max_passes = 0`** returns `fv_padded_frame(old_frame, target, sharing, seed)` and `()`: the hint-free start AT
  THE TARGET. `project`, `hint`, `tol` and `alternate` are not read; the seed is.
- **"Nothing to complete"** (target stores equal the old stores) returns the same call, which equals the old frame
  value for value (T6 checks this bitwise), and `()`.
- **Truncation outside `update`.** The step raises when a store outside `update` is truncated. The loop truncates
  only at pass `n_over`, always with `'ULRO'`. Every later pass runs at the target stores, which are the frame's own,
  so the error cannot fire. My mutation that gives the truncating pass a colour fails 11 tests, the step raising
  among them.
- **Old ranks** are the old frame's own `(up, left)` widths, and `_frame_stores`' layout check guarantees they
  reproduce its stores.

## The two judgment calls

1. **The oversampled phase is decided on stores, not ranks.**
   - I searched 20,000 random structures (d = 1..4, ranks 0..7 including non-minimal ones, p = 0..5) plus 20,000 with
     sharing groups. No case has working ranks ≠ target ranks with equal stores, so the two criteria agree on
     everything sampled.
   - Stores are the natural criterion, because the passes act on them. Accepted.
   - **For N6:** the uniform loop must use the same criterion (T12 "same pass counts").
2. **No T16 row for the loop.** See "For the maintainer". Accepted as an open question for Nick, not as a finding.

## Findings

### 1. MINOR: no test checks the loop's working start for `seed`

`tests/backend/test_padded_frame.py`, the loop tests.

**Problem.** In a scratch copy, the mutation `frame = fv_padded_frame(old_frame, *working, sharing)` (seed dropped)
leaves the loop tests green: 12 passed. The dense comparisons start from the library's own start, `seen[0]`, and
every `max_passes > 0` test uses the default seed 0. The seed is what makes the `'cone'` reference deterministic
(D-7), so it needs pinning. The zero-pass path's seed is pinned, and dropping it there is caught.

**Fix.** In `test_against_the_dense_loop`, or a small new test, run with `seed=3` and assert that
`seen[0] == fv_padded_frame(old, *working, sharing=sharing, seed=3)` bitwise and that it differs from the seed-0
start.

### 2. NIT: `oversampling` is validated only when the loop reaches `oversampled_ranks`

`padded_frames.py`, the head of `fv_aligned_padded_frame`.

**Problem.** `max_passes` and `oversampled_passes` are checked up front, but `oversampling=-1` passes silently at
`max_passes=0` or in the no-op case, while `oversampled_passes=-1` raises there.

**Fix.** Add `oversampling` to the up-front non-negative-int loop, so a structural error does not depend on
`max_passes`.

### 3. NIT, pre-existing from N2 (committed): a shipped comment points at an unnamed external note

`padded_frames.py:307`: `# One alignment step: the note's unrooted message passing …`.

**Problem.** "The note" names nothing a library reader can find.

**Fix.** N9's math note can be cited there, or the comment can say "unrooted message passing over the directed
edges of the T3 tree".

## What I ran (env `t3toolbox`: py3.11, numpy 2.x, jax 0.10.2, unless stated)

1. **Against the prototype.** A scratch script, `compare_loop.py`, runs `ragged_loop.ragged_aligned_padded_frame`
   from the library's start against `fv_aligned_padded_frame`.
   - Cases: a random d = 4 point and the cycle case; p ∈ {0, 5}; alternate on and off; (tol, cap) ∈ {(0.05, 10),
     (1e-6, 25)}. That is 16 runs.
   - Pass counts are equal in every run. Final frames agree to ≤ 2.7e-15 (largest principal-angle sine) and
     movements to ≤ 5.7e-15.
2. **Tests.** `pytest tests/backend/test_padded_frame.py`: **48 passed, 134 subtests** (16 s). T6's dense loop takes
   6.7 s and T4's simultaneous cycle 3.5 s.
3. **The rest.** Doctests of `padded_frames.py`: **1 passed**. `test_import_graph.py` and
   `test_contractions_interpreter.py`: **20 passed**.
4. **Floor env** `tttt` (py3.9, numpy 1.22.4): `test_padded_frame.py` **48 passed**. `call.kwargs` in the schedule
   test works on 3.9.
5. **The cycle data.** `_padded_frame_cycle_case.py` is bitwise equal to T3Polynomial's
   `data/g0_reference/g06c_cycle_case.npz` on all 15 items (`S`, seed, shape, ranks, ten old cores).
6. **Judgment call 1.** The 40,000-structure search above.
7. **My mutations** (`-k Loop`, scratch copy):
   - dropping `seed` from the working start **survives** (finding 1);
   - dropping `sharing` from the working start, dropping `seed` from the no-op return, giving the truncating pass a
     colour, and dropping `sharing` from the step are all **red**.

The full suite was not run; the implementer reports 1,245 passed, 4 skipped, 0 failed.

## For the maintainer (review stop after N3)

1. **Plan text to update.**
   - §2.3's "working ≠ target" should say stores. In practice it makes no difference (see above).
   - T4 and Appendix A still name `tests/data/padded_frame_cycle_case.npz`.
2. **A new test-data convention.** The cycle case lives as a Python module of float literals,
   `tests/backend/_padded_frame_cycle_case.py`: 7 KB, `repr` round-trip, provenance in the docstring. It replaces a
   binary fixture under the no-binary-fixtures ruling. If you like it, `testing_strategy.md` could record it in N9
   as the way to keep small regression cases.
3. **No T16 row for the ragged loop.**
   - It is a host loop, so it cannot be jitted.
   - An eager-jax row costs about 18 s even at d = 2 (implementer, on a loaded machine).
   - Its two building blocks have jitted rows.
   - Your call whether T16 should say "jittable functions" or whether a slow eager row is wanted anyway.
4. **Eager jax on the ragged layer is slow on first call.** The step took 37.7 s eagerly at d = 4 (N2 review,
   measured), against 0.024 s in numpy. A ragged-layer user on jax arrays pays this unless they jit the step. The
   loop cannot be jitted as a whole. Decide before N8 whether the ragged twins jit the step internally or the docs
   just say so.
5. **Machine load.** The parallel Darcy runs inflate multithreaded-BLAS timings about 50× on small SVDs (handoff). My
   comparison runs used `OPENBLAS_NUM_THREADS=1`; my pytest runs did not.

## Counts

MAJOR 0, MINOR 1, NIT 2.

VERDICT: REVISE
