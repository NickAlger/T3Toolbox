# N8b handoff: Gaussian seeds everywhere, slot-indexed (D-41)

Slice N8b, between N8 and N9: the maintainer's decision D-41 (`dev/HANDOFF.md`). Implementer: Claude Opus 5.5,
2026-11-10, on top of a9073bf2. Everything is uncommitted; nothing outside the repository was touched.

## The new convention (both layers)

- **The seeds.** Every store's completion seeds -- the Tucker factors AND the TT and down cores -- are rows of a keyed
  legacy `RandomState` stream: `_seed_rows(store, key, seed, w, k, p)` =
  `RandomState([seed, tag(store), key]).standard_normal((w + k, p))[w:]`.
  - The tags: up 0, down 1, left 2, right 3.
  - The key: the core's mode; for a Tucker factor, its sharing group's first mode (so a group's members share
    seeds).
  - **Slot s is seeded by row s.** A store at old width `w` gaining `k` slots uses rows `w .. w+k-1`, so start-from-start
    chains never reuse a row. The zero point (`w = 0`) uses rows `0 .. k-1`.
  - The seed length `p` is the mode size (Tucker) or the real rows of the core's ragged unfolding.
- **The completion.** The pinned completion of `[old block embedded | seeds]`, as before (`pinned_completion`; the
  uniform twin's `pad_safe_pinned_completion`).
- **The uniform twin** generates the same rows on the HOST, per core (Tucker: per group) and per stack element, at that
  element's ragged size. It places them at the unfolding's real rows in flat slot order, which is the ragged row-major
  order. Junk rows are zero, and the scatter tests stay at 1e-12 (bitwise in practice).
- **Deleted:** `_haar_seed_rows` (now `_seed_rows`), `_coordinate_seed_order`, `_seeded_completion` and its
  outside/inside branch (folded into `_padded_core`), and the uniform `_units` / `_coordinate_seeds` /
  unit-vector machinery. `_padded_core` and `_padded_tucker_factor` keep their structure. The uniform
  `_store_completion` is now one pad-safe completion over the whole unfolding, and `_store_seeds` is the host
  bookkeeping.

## Files

| file | change |
|---|---|
| `t3toolbox/backend/padded_frames.py` | The convention above; `fv_padded_frame` passes each core's store and key. The `seed` comments are updated. |
| `t3toolbox/backend/uniform_padded_frames.py` | Per-element slot-indexed Tucker seeds; `_store_seeds` + `_store_completion`; the module docstring. |
| `t3toolbox/frame_variations_format.py`, `t3toolbox/uniform_frame_variations_format.py` | Docstrings and `seed` comments describe the new convention. |
| `t3toolbox/optimizers.py`, `t3toolbox/backend/optimizers.py` | The `'cone'` docs no longer claim that `cone_max_passes=0` reproduces `'zero'` (decision point 1). |
| `tests/backend/test_padded_frame.py` | T5 rewritten (below). New helpers: `stream_rows`, `seeded_qr`, `convention_reference` (the convention rebuilt independently), `worst_difference`. |
| `tests/backend/test_padded_start.py` | T14 (d) and (e) updated (below). |
| `tests/backend/test_uniform_padded_frame.py` | The chain test (`TestStartChains`, `coordinate_point`). |
| `dev/aligned_padded_frame_slices/N1_handoff.md` | A note at the top: the S2-parity record is superseded by D-41 (kept as history). |

## Every changed expectation

1. **T5, `TestConvention`:**
   - **removed:** `test_outside_seeds_are_placed_exactly`, `test_in_block_columns_come_from_coordinate_seeds`,
     `test_tucker_completion_is_the_keyed_stream_completed` (the old key `[seed, mode]`, rows from 0),
     `test_the_seed_moves_the_tucker_completions_only`, and `test_sign_gauge_covariance` with its `sign_gauge`
     helper. Gaussian seeds are not covariant under sign flips of an unfolding's ROW axes (coordinate seeds were), so
     the property no longer holds.
   - **added:** `test_each_store_is_the_seeded_pinned_qr`. Every store's completion equals the independent reference to
     1e-12: unshared, shared, a single-bond growth (no new rows elsewhere), d = 1, two seeds. Also added:
     `test_the_seed_moves_every_completion`.
   - **kept:** `test_tucker_completion_is_nested` (slot indexing keeps prefixes) and `test_stack_equals_per_element`.
2. **T5, `TestZeroTensor`:** "the start at the empty frame is `fv_zero_point_frame`'s frame, bitwise" becomes "it is
   the convention" (reference, 1e-12), plus stacked = per element bitwise. `fv_zero_point_frame` (the `'zero'`
   reference's frame) is UNCHANGED (decision point 1).
3. **T14 (d):** "`cone_max_passes=0` equals `'zero'`" becomes "equals the misfit-gradient norm at the zero tensor's
   hint-free start" (computed directly, 1e-12). The cone exceeds that value.
4. **T14 (e):** probe_derivatives at d = 4, uniform vs ragged `padded_start`.
   - Before: per-pass movements and equal pass counts.
   - Now: the final aligned spans, ≤ 1e-9 at tol 1e-10 (both runs stopping on tol) and ≤ 0.2 at the default stop.
   - Why: the two layers' generic old frames agree only up to column signs, and Gaussian seeds are not
     sign-covariant, so the starts and the per-pass trajectories differ. The aligned results agree (the N7
     washout).
5. **T8 / T9 / T10 start cases:** unchanged expectations. They pass with the new convention through the scatter at
   1e-12.
6. **`tests/test_dispatch.py::TestAlignedLoopJit` (eager = jit):** its 'two-colour, d=4' case no longer stops early
   within the 12-pass cap at tol 0.05 under the new start (12 passes instead of 11). That case now runs at tol 0.1,
   where it stops on a pair at pass 9 (pair movements 0.085 and 0.052, against 0.157 before), and the pinned counts
   are `(9, 8)`. The eager-equals-jit comparisons themselves were unchanged and passed.
7. **The N1 S2-parity record:** superseded (note added). `dev/aligned_padded_frame_slices/N1_check_s2_equivalence.py`
   no longer matches and is not a test.

## The chain test (`TestStartChains`)

- **Setup:** start → start → start without a fit, at targets `((3,3,3,3),(1,3,3,3,1))` → `((3,3,3,3),(1,3,4,3,1))`
  (a single bond grows: no new rows in the other unfoldings) → `((4,4,4,4),(1,4,4,4,1))`. Two points: a
  coordinate-structured one (identity-prefix Tucker factors, 0/1 TT cores: its frame has unit-vector columns) and a
  generic one. The uniform chain starts from a gappy scatter with garbage and runs block layout to block layout.
- **At every level:** the ragged start equals the independent reference (1e-12); the uniform read-back equals the ragged
  start (1e-12); junk is zero; **every seed's pinned-QR pivot** (its norm off the old block and the earlier seeds,
  `|R_jj|` of the reference QR) is above **1e-3**. The smallest measured is 0.036.

## Decision points for the session

1. **`fv_zero_point_frame` / the `'zero'` reference were left unchanged.**
   - D-41's text says "the zero-point frame changes (random orthonormal cores)"; the N8b brief names only the start's
     functions.
   - I changed the START (including the zero tensor's start, so `'cone'` changes) and left `fv_zero_point_frame`
     alone: it is the DEFAULT `'zero'` Newton reference, used by every manifold run.
   - Consequence: `cone_max_passes=0` no longer equals `'zero'` (D-38's "= 'zero' to rounding"). It is now the
     gradient norm at the zero tensor's Gaussian hint-free frame. The docs say so.
   - If `fv_zero_point_frame` should follow D-41, it is a small change: it would call the start's convention. But it
     changes every default Newton run's `g0` and its stats.
2. **The layers' starts agree only for identical old frames** (signs included). T14 (e) now asserts the aligned
   agreement instead. Plan §2.4's cross-layer wording needs this in round 8.

## Verification

- `test_padded_frame.py`: 50 passed. `test_uniform_padded_frame.py` with the chain test: 39 passed.
  `test_padded_start.py`: 24 passed. The touched doctests: 28 passed.
- The `-W` Sphinx dummy build: exit 0, no warnings. The AST scan: clean.

## Full suite, floor env, numpy-1.26 env

- **Full suite** (`-n 3`): 1 failed, 1,337 passed, 4 skipped, 48,697 subtests (15 min 11 s).
  - The failure was the pinned pass count in item 6 above, now fixed.
  - Rerun of that test: passed. Every other test passed in that run.
- **numpy 1.26.4 env** (the scratch venv mirroring the CI floor: py3.9, numpy 1.26.4, jax 0.4.30), `-n 2`: all
  padded-frame files, `test_linalg.py`, `test_aligned_padded_frame_frontend.py` and the dispatch file. **227 passed,
  1 skipped, 2,341 subtests.** That includes the T8 block-layout case that failed on CI, and the frame-widths row.
- **Floor env `tttt`:**
  - the padded-frame and frontend files: 125 passed;
  - the padded-frame dispatch rows and classes: 16 passed, 1 skipped (the `jax.monitoring` listener test).
