# N3 handoff: the ragged alignment loop

Slice N3 of `dev/aligned_padded_frame_impl_plan_2026-11-09.md` (§8): `fv_aligned_padded_frame` (§2.3, §4.3, §5.1),
tests T4 and T6. Implementer: Claude Opus 5.5, 2026-11-09, on top of N2 (196fcfa2). Everything is uncommitted;
nothing outside the repository was touched.

## Files

| file | change |
|---|---|
| `t3toolbox/backend/padded_frames.py` | `fv_aligned_padded_frame` (the §5.1 signature, minus the "(D-38)" label per the N1 ruling on plan labels), in `__all__`; the module docstring names it. The "target contains the old structure" check is factored out of `fv_padded_frame` into `_require_contains`, which the loop calls too (`fv_padded_frame`'s message is unchanged). |
| `tests/backend/_padded_frame_oracles.py` | `+ dense_loop`: the §2.3 schedule on the dense step, written from the plan's text. |
| `tests/backend/_padded_frame_cycle_case.py` | **new**: the 243-number cycle case as TEXT (`S`, the old cores, the target ranks, the seed), exported from T3Polynomial `data/g0_reference/g06c_cycle_case.npz`; provenance in the docstring. Every number is the stored float64 written with `repr`, and the round trip was checked bitwise against the npz. A Python module because `tests/` has no data files and `testing_strategy.md` sets no convention for them; it follows the `_curvature_oracles.py` helper-module precedent. |
| `tests/backend/test_padded_frame.py` | T4 and T6 appended (three classes); the module docstring extended. |

### The loop

- **Order of work.** Validate `max_passes` and `oversampled_passes` (non-negative ints). Read the old stores
  (`_frame_stores`); the old ranks are the old frame's own up and left widths. Then compute the target stores
  (`_rank_stores`, boundary 1) and check that they contain the old ones.
- **Zero passes, or target stores = old stores.** Return `fv_padded_frame(old_frame, target, sharing, seed)` and
  `()`. For zero passes that is the hint-free start at the target (D-38). For the no-op it equals the old frame
  value for value. `project` is not called and `hint` is not read.
- **Working ranks and the oversampled phase.** `working = ranks.oversampled_ranks(shape, target, p, sharing)`.
  `n_over = 0` when the working stores equal the target stores, else `oversampled_passes`, then capped at
  `max_passes − 1`, because the cap counts the truncating pass.
- **The passes.** The start is `fv_padded_frame` at the working ranks. Pass `it` runs
  `tv_align_padded_frame(frame, project(frame, hint), old ranks, out ranks, sharing, update)`:
  - out ranks: the frame's own while `it < n_over`, then the target (pass `n_over` truncates);
  - update: `'ULRO'` up to and including the truncating pass, then `'UL'` / `'RO'` alternately when `alternate`.
- **The stop.**
  - Alternate: `it > n_over` and both of the last two movements below `tol`.
  - Otherwise: `it ≥ n_over` and the movement below `tol`.
  - The stack maximum decides in both; the cap is the loop bound.
- **Return.** `(frame, tuple(movements))`, each movement a `C` array (numpy's scalar at C = ()).

## Verification

- **Against the prototype** (`ragged_loop.ragged_aligned_padded_frame`, given the library's start): both schedules,
  12 passes, frames ≤ 1.2e-15, movements ≤ 2.2e-15, same pass counts.
- **T6 against the dense loop** (`TestLoopAgainstDense`, `tol = 0` so every case runs its full pass count): pass by
  pass, the frames (the largest principal-angle sine per directed edge) and the movements are each within 1e-9.
  Measured: frames ≤ 1.3e-14, movements ≤ 2.2e-15. The final frames are valid padded frames.
  - Two-colour with p = 0 and with p = 5 (two oversampled passes, the truncating pass): 40 passes each.
  - The simultaneous schedule, p = 5: 12 passes. Two-colour at d = 4: 6 passes. The empty old frame (the cone):
    20 passes. Sharing `(0, 0, 1)`: 20 passes.
  - Each run checks that the start sits at the working stores.
  - The two-colour rows are T4's "two-colour = transport of the other stores": the dense loop transports through
    the oracle's `dense_pass`, the `select_pass` rule.
- **T6, the schedule** (`TestLoopSchedule`), recorded call by call by wrapping the real step with `mock.patch.object`
  (`wraps=`), as (update, output ranks):
  - working = target: `ULRO@target, UL, RO, UL`.
  - The defaults: `ULRO@working ×2, ULRO@target, UL, RO, …`; with `alternate=False`, all `ULRO`.
  - `oversampled_passes=5, max_passes=3`: two oversampled passes and the truncating one, ending at the target
    stores.
  - `max_passes=1`, p = 5 or 0: one `ULRO@target` pass.
  - `max_passes=0`, p = 0 and 5: no step and no projection (a raising `project`); `()`; bitwise
    `fv_padded_frame(old, target, seed)`.
  - Target = old: no step, `()`, the old frame bitwise.
  - The stop rule on prescribed movements (the step replaced by a fake): the truncating-pass pair; no stop during
    oversampling; the simultaneous rule; the stack maximum under both rules; the default cap 10; a given cap.
  - Structural errors: negative or bool `max_passes`; negative `oversampled_passes` and `oversampling`; a target
    below the old frame; a boundary TT rank.
- **T6, stacks** (`test_stacked`): C = (2,); the movements are a tuple of `(2,)` arrays; each element equals the
  per-element loop to 1e-12.
- **T4, the cycle** (`TestLoopCycle`, the text case; p = 0, the seeded start, the dense projection of `S`):
  - simultaneous, 200 passes: the last two movements are both above 0.05 (measured 0.0948; the prototypes reported
    0.092 / 0.095);
  - two-colour, `tol = 1e-8`: stops after 59 passes with both last movements below 1e-8 (the prototypes reported 72
    passes to 1e-10).
- **Mutation run** (scratch; the file restored and checked): 13 mutations of the loop, each turning the loop tests
  red.
  - Schedule: the cap not counting the truncating pass; the colours swapped; an oversampled phase at
    working = target; no truncating pass; `alternate` ignored.
  - Stop rule: the alternate stop on one pass; the alternate stop allowed at the truncating pass; the simultaneous
    stop during oversampling; min instead of max over the stack, under each rule.
  - Short circuits and checks: zero passes running the loop; the no-op not short-circuited; the containment check
    deleted.
  - The simultaneous stack mutant survived the first version of the test; a stacked simultaneous case now kills it.

## Judgment calls

| # | issue | decision | why |
|---|---|---|---|
| 1 | §2.3 decides the oversampled phase on "working ≠ target" (ranks). | On the STORES: `n_over = 0` when the working stores equal the target stores. | The frames' structure is what the passes act on; ranks that differ while the stores agree would give "oversampled" passes at the target structure and a "truncating" pass that truncates nothing. For minimal targets the two criteria agree. |
| 2 | Where the cycle case lives (no binary fixtures). | `tests/backend/_padded_frame_cycle_case.py`, a Python module of literals with the provenance in its docstring. | No text-data convention exists under `tests/`; a module needs no file I/O and follows the helper-module precedent. |
| 3 | §5.1's "(D-38)" in `max_passes`' comment. | Dropped. | The N1 ruling removes plan labels from shipped code. |
| 4 | The loop's T16 dispatch row. | **Not added**; flagged. | The loop is a host loop (its stop reads the movement on the host), so it cannot be jitted. The eager jax check costs about 18 s even at d = 2 with 3 passes on this loaded machine (verified by hand: jax in, jax out). Its two calls, `fv_padded_frame` and `tv_align_padded_frame`, have their own rows. N3's brief names T4 and T6 only. Say if you want the row anyway. |
| 5 | Validation of `tol` / `alternate`. | Not validated (any float; truthiness). | They are tunables, not structure; the D-38 rule about unread options applies at the optimizer level (N7), not in the loops (§2.3). |

## For the session

- **Timing on this machine.** While the parallel Darcy runs load it (load 9 on 8 cores), multithreaded OpenBLAS makes
  small SVDs about 50× slower (an 80 × 170 SVD: 0.28 s against 6 ms single-threaded). So the dense-loop test took
  38 s here, against 3.2 s with `OPENBLAS_NUM_THREADS=1`; the test file takes 15 s single-threaded. The full suite's
  wall time is inflated the same way.

## Round 2 (review `N3_review.md`: REVISE, one MINOR; the judgment calls accepted)

| finding | fix |
|---|---|
| MINOR: nothing tested that the working start uses `seed` | `test_the_working_start_uses_the_seed`: the first frame the loop projects at equals `fv_padded_frame(old, *working, seed=s)` bitwise for s = 3 and 0, and the two differ. With the seed dropped from the loop's start, the test goes red (checked). |
| NIT: a negative `oversampling` passed silently at `max_passes=0` | Validated up front with `max_passes` and `oversampled_passes`; a test at `max_passes=0`. |
| NIT: the N2 comment "the note's unrooted message passing" | Rewritten to describe the construction: every directed edge gets a new completion from its content, assembled from messages passed inward (step 1) and the feeding edges' new completions (step 2), with no root. |

Rerun after round 2: `test_padded_frame.py` and `test_import_graph.py`, 54 passed with 137 subtests; `padded_frames`'
doctests pass. The full suite was not rerun (one validation line, one comment and two tests); the result below is
round 1's.

## In the plan: wrong or ambiguous

- T4's data path `tests/data/padded_frame_cycle_case.npz` (also Appendix A) contradicts the no-binary-fixtures ruling;
  the case is text now (call 2).
- §2.3's "working ≠ target" is on ranks; the stores are the natural criterion (call 1).
- T16 asks for a row for every new backend function, but the ragged loop cannot be jitted and the eager row is
  expensive (call 4).

## Full suite

`PYTHONPATH=$PWD python -m pytest tests -q -n auto` (env `t3toolbox`: py3.11, numpy 2.x, jax 0.10.2), the complete
`tests/` tree on the final working tree: **1,245 passed, 4 skipped, 48,210 subtests passed, 0 failed** in 15 min 20 s
on 8 workers (on the loaded machine).
- The one warning is the existing `adam`-on-MANIFOLD `RuntimeWarning` from the oracle sweep.
- N2 had 1,233 passed / 48,195 subtests; N3 adds T4 and T6.
- `padded_frames`' module doctests pass.
