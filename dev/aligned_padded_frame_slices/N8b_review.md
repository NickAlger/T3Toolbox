# N8b review: Gaussian seeds everywhere, slot-indexed (D-41)

Reviewer: Claude Opus 5.5, 2026-11-10. I reviewed the uncommitted working tree on top of 9761a75e (plan round 8)
against:
- the plan: §2.4, §3.4, §4.2, §4.5, §5.2–§5.3, T5, T20, D-38 (amended), D-41, D-42;
- `N8b_handoff.md`;
- the session's CI rules: the `-W` docs build, shadowed imports, cross-path tolerances, and `-n 3`.

## Summary

The new convention is implemented exactly as D-41 states it, on both layers:
- one slot-index rule: slot `s` uses stream row `s` of `RandomState([seed, store tag, key])`;
- host-generated seed rows as bookkeeping only;
- one pinned completion per store;
- no numerical decision anywhere.

I rebuilt it independently (my own flat stream draw, reshaped by hand; my own sign-fixed numpy QR) for every store:
- the ragged start equals my rebuild **bitwise** on all 14 cases: generic, shared, the zero tensor, a single-bond
  step, a coordinate-structured point, d = 1, a `t3svd` point; two seeds each;
- the uniform start, through MY gappy, garbage-filled scatter, equals it to **≤ 1.1e-16**.

Further:
- **Chains** of four starts agree on both layers at every level, with every pivot ≥ 0.003, at a coordinate point and
  at a generic one.
- **Tied groups** keep one array.
- **Default and `'zero'` runs are bitwise unchanged vs HEAD**: 8 of 8 seeded runs, both layers.
- **The cone at zero passes** equals the hint-free start's gradient norm exactly, on both layers.

I have no MAJOR or MINOR findings. There is one decision for the maintainer that this slice makes more pressing:
since D-41, the hint-free start of a NONZERO old point depends on the old frame's column SIGNS. So it now differs
between the layers for generic points too (span distance 1.0, where it was 3e-15), and it can differ across LAPACK
builds.

## Style (checked first)

- **One slot-index rule.**
  - `_seed_rows(store, key, seed, w, k, p)` = rows `w .. w+k−1` of
    `RandomState([seed, tag(store), key]).standard_normal((w + k, p))`.
  - The tags: up 0, down 1, left 2, right 3.
  - The key: the mode, or the sharing group's first mode for the Tucker store.
  - `w` is the old width, i.e. the old block's real column count on the uniform layer.
  - It is the single seed source of both layers (the uniform module imports it, as it imported `_haar_seed_rows`
    before).
- **Seeds are host bookkeeping only.**
  - The uniform `_store_seeds` loops over cores and stack elements on the host, generates each element's rows at its
    ragged unfolding length, and places them at the real rows in flat slot order, which is the ragged row-major order.
  - The mathematics is one batched `pad_safe_pinned_completion` per store and one over the Tucker groups, so the
    start stays truly uniform.
  - The ragged `_padded_core` is one `pinned_completion` of `[embedded old block | seeds]`. The outside/inside branch
    is gone.
- **No numerical decisions.** There are no thresholds and no rank choices. Degenerate seeds would be completed by the
  primitives, as before (D-12).
- **Deleted cleanly.**
  - `_haar_seed_rows`, `_coordinate_seed_order`, `_seeded_completion`, `_units` and `_coordinate_seeds` are gone,
    with no leftover reference in `t3toolbox/`.
  - The remaining "Haar" mentions are `fv_zero_point_frame` (unchanged by ruling) and an unrelated `linalg` sketch.
  - The AST scan finds no shadowed import or duplicate definition, and every name the uniform module imports from
    `padded_frames` is used.
- **No plan labels** in the shipped diff.

## Correctness (my own checks, `n8b_check.py`, `n8b_cone.py`, `gbit.py`)

1. **Every store, both layers, against an independent rebuild.**
   - My rebuild: for each store, embed the old block, draw `RandomState([seed, tag, key]).standard_normal((w+k)·p)`
     flat and reshape it by hand, take rows `w..`, and complete with a sign-fixed `np.linalg.qr`. Carriers are the
     old content.

   | case | ragged against rebuild | uniform (gappy + garbage) against rebuild |
   |---|---|---|
   | generic d = 4 | 0.0 | ≤ 1.1e-16 |
   | shared `(0,1,1,0)` (one array per group) | 0.0 | ≤ 5.6e-17 |
   | the zero tensor | 0.0 | 0.0 |
   | single bond grows (no new rows elsewhere) | 0.0 | ≤ 5.6e-17 |
   | coordinate point (identity-prefix Tucker, 0/1 TT cores) | 0.0 | 0.0 |
   | d = 1 | 0.0 | 0.0 |
   | a `t3svd` output | 0.0 | 0.0 |

   - Every case was run at seeds 0 and 5.
   - The smallest seed pivot over these 14 is 0.029 (the zero tensor, seed 5).
2. **The slot-index rule on chains.**
   - The chain: four hint-free starts in a row, each from the previous start as the old frame. The targets are
     `(3,…)/(1,3,3,3,1)`, then a single bond to 4, then all to 4, then a further mixed step.
   - Points: the coordinate point (Tucker 2, bonds 2) and a generic one.
   - Ragged against rebuild: 0.0 at every level. Uniform read-back against ragged: ≤ 5.6e-17. Orthogonality
     ≤ 1.1e-15.
   - Pivots: min 0.036 on the first three levels. My extra fourth level reaches **0.003**: generic, `(5,4,5,4)`, a
     store filled to its last free direction. That is above T20's 1e-3, but it shows that the bound is
     probabilistic, not structural (see maintainer item 2).
   - Rows are fresh by construction. Each level starts at the old width `w`, which grew by the previous `k`. The
     underlying stream elements used at a level begin at `w·p ≥ (w_prev + k_prev)·p_prev`, so with non-decreasing
     unfolding lengths no element is reused.
3. **Tied seeds.** A sharing group's Tucker store is completed once, keyed by its first mode, and every member holds
   the same array (`is`), on the ragged layer and through the uniform scatter. TT and down stores are keyed per mode,
   as D-41 says.
4. **The deleted sign-gauge test.** The claim checks out. I flipped bond 2's signs in a generic old frame (same point,
   same subspaces, 2.1e-15): the two starts differ in span by **0.987**. Gaussian seeds are not covariant under a
   row-sign flip of an unfolding, so deleting `test_sign_gauge_covariance` is right.
5. **`'zero'` and every default Newton run, bitwise unchanged vs HEAD.** HEAD came out by `git archive`. The 8 seeded
   runs: ragged and uniform; default Gauss-Newton (`'zero'` by default); exact with `'zero'`; exact with
   `'warm_start'`; `gradient_descent`. **8 of 8: cores bitwise, `repr(stats)` equal.** `fv_operations.py`
   (`fv_zero_point_frame`) is untouched.
6. **The cone at zero passes.** `newton_cg(g0_reference='cone', cone_max_passes=0)` records exactly my direct
   `‖Π 𝒥ᵀ r(0)‖` at the zero tensor's hint-free start: 5679.45993618 on both layers, relative difference 0.0. It
   differs from `'zero'` (6845.42), as the amended D-38 says.
7. **The changed expectations** (handoff items 1–7) are each justified.
   - T5 is restated against an independent reference, at 1e-12, a prudent cross-path bound.
   - T14 (d) is restated as above.
   - T14 (e) now compares aligned frames, because the starts differ across the layers (item 1 below). T12 (N6) still
     pins the uniform loop pass by pass against the ragged loop from the SAME old frame.
   - The eager-vs-jit case changed only its pinned stop count.

## What I ran

1. `n8b_check.py`, `n8b_cone.py` and `gbit.py` (HEAD against the working tree): results above.
2. The AST scan and a grep for leftover helpers: clean.
3. **Tests** (env `t3toolbox`, `-n 3`):
   - `test_padded_frame.py`, `test_uniform_padded_frame.py`, `test_padded_start.py`,
     `test_aligned_padded_frame_frontend.py` and `test_import_graph.py`: **130 passed, 1,663 subtests**.
   - The padded-frame rows and classes of `tests/test_dispatch.py`, incl. the eager-equals-jit classes: **22 passed,
     66 subtests** (4 min 50 s).
   - The doctests of the six touched modules: **30 passed**.
4. **Docs.** `python -m sphinx -W --keep-going -b dummy docs <scratch>`: **exit 0**, no warnings.
5. **Floor env** `tttt` (py3.9, numpy 1.22.4, jax 0.4.30): the three padded-frame test files, **113 passed**.

The full suite was not run (the implementer: 1,338 passed at `-n 3`, with the one pinned-count fix rerun).

## For the maintainer

1. **The hint-free start now depends on the old frame's SIGNS, not only on bases at ties.**
   - Under the coordinate convention, a sign flip of an old column only flipped a seed, so the start was
     sign-covariant. The two layers' starts then differed only at singular-value ties (`t3svd` and shared points;
     generic points agreed to 3e-15).
   - With Gaussian seeds, any sign difference changes the span. Ragged against uniform hint-free starts of generic
     randn old points now differ by **1.0** (5 seeds), as do `t3svd` and down-orthogonalized points.
   - The orthogonal representations take their column signs from plain LAPACK SVDs (`t3_orthogonalization.py`
     118/157; no sign fix). So the start of a nonzero old point is also not guaranteed reproducible ACROSS LAPACK
     builds or versions.
   - Within one platform and one layer it is deterministic. The `'cone'` reference (empty old frame) is unaffected.
     Aligned results agree once converged (T14 (e)).
   - Plan §2.4's D-42 text ("happens at singular-value ties"; "1e-15 on generic points") understates this and should
     be restated.
   - **Cheap remedy, if you want reproducible starts:** canonicalize the old frame's column signs before seeding,
     e.g. a deterministic per-column sign rule applied in sweep order (Tucker rows; left bonds left to right; right
     bonds right to left; down columns). It restores sign covariance on both layers and across BLAS builds. Ties
     still need the audit item.
   - Otherwise, document it in N9 next to the basis dependence.
2. **T20's pivot bound (≥ 1e-3) is probabilistic.** When a store's completion takes the last free direction of its
   unfolding, the pivot is `|N(0,1)|`-like: I measured 0.003 on a fourth chain level. That is harmless numerically
   (rounding sensitivity ≈ eps / pivot ≈ 1e-13), and the test's fixed seeds make it deterministic. But the bound is a
   property of the chosen seeds, not a guarantee, and the T20 docstring should say so.
3. **`fv_zero_point_frame` unchanged (decision point 1).** Agreed: it is every default run's reference, and the
   non-breaking rule holds (verified bitwise). D-41's sentence "the zero-point frame changes" should be read as the
   ZERO TENSOR'S START changes, and the plan text should say so.

## Counts

MAJOR 0, MINOR 0, NIT 0; and the maintainer items above.

VERDICT: APPROVE
