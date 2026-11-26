NOT YET: the geometry is right, and the centers check out independently on both layers. Three should-fix findings: four red subtests that are part 1's own (the member's new return, unpacked by two `test_dispatch` tests), a list of expected-red tests that differs from what is red, and a silent wrong answer on the old four-store corewise frame that a structural check would turn into an error.

# P16 part 1 review, round 1: the geometry

Reviewer (the fresh pair), 2026-11-21, on the uncommitted part 1 at `1baf1335`. I read it against the plan's P16 item 1, the
settled note 3 of 2026-11-21, my design review and Claude's rulings ("P16 designed"), and the handover. I judged the
diff, not the handover. Runs were one at a time, at `-n 2`. The mutant copy (`t3toolbox/` and `tests/` only) was
deleted. The working tree is untouched apart from this file.

## What I ran

| Check | Result |
| --- | --- |
| The part's six files (`test_geometry`, `test_padded_start`, `test_padded_frame`, `test_uniform_padded_frame`, `test_zero_reference`, `test_import_graph`) | 181 passed, 5 failed, as the handover says; the 5 are red for the stated reason (below) |
| **The full suite** (`tests/`, `-n 2`, 33 min; MemAvailable about 4 GB near the end) | 194 failed, 1847 passed, 5 skipped, 1 xfailed; **66 test ids red**, listed in my scratch `reviewer2/actual_red.txt` (finding 2) |
| My uniform centers check (`reviewer2/p16p1_centers_uniform.py`) | right; detail below |
| 2 mutants of my own | both killed |

**The uniform twin of my centers check.** The setup:
- a stack `C = (2,)` with the old ranks **different per element**, `(2, 2, 2)/(1, 2, 2, 1)` and `(1, 2, 2)/(1, 1, 2, 1)`;
- the old point packed **wider than the level** (`n = r = 6` against widths of 4), which takes the truncation path
  of `_ufv_padded_centers`;
- garbage `±1e3` in the old point's padding;
- 0, 1, 3 and the default number of passes.

The results:
- The uniform centers' residual is at most 9e-15, per element.
- The probe `z` read through `Vc` by the engine equals `dense_probe` of each element's `x0` to 3.2e-15.
- Garbage changes nothing.

**A wrong-centers mutant must really be wrong.** The obvious candidate, `x0`'s own centers (`frame(x0)[1]`), **equals**
the zero-padded old centers to 1.1e-14. It is an equivalent mutant: orthogonalizing a zero-padded point reproduces the
old centers. So nobody should count it as a killed mutant. Genuinely wrong centers give residuals of order 1:
- the centers of another point: residual 2.3;
- `Vc` rolled by one rank slot: residual 1.4.

So the residual test discriminates, and the handover's two mutants (trailing slots, all zero) are genuine.

**My mutants.** Both were killed:
- Ragged `Hc` set to zeros with `Vc` right: killed by `test_consistent_and_reproducing_x0` (4 subtests). The engine
  never reads `Hc`, so only the residual sees it.
- `_ufv_padded_centers` with the left and right masks swapped in the `H` gather: killed by
  `test_the_uniform_helper_reads_real_slots_anywhere` only. On the member's own path the masks are prefixes, so the
  swap is equivalent there, and that test is what pins the gather.

## What is right

**The corewise frame is the point.**
- *Ragged:* `frame(x) = ((B, G), None)`, `base_point` is the frame, and `retract` and `point_tangent` follow.
- *Uniform:* `frame(x_sc) = ((B_sc, G_sc, shape, masks), None)`, using the geometry's shape and plain masks, and
  `base_point = (frame[0], frame[1])`.
- `frame_form` reads both as corewise.
- The tests check identity of the arrays. The retraction at `K = ()` and `(3,)` is checked against a dense
  construction (ragged) and against the ragged retraction through dense tensors (uniform, with garbage in the
  padding). The shared tie is checked both ways.

**The direct uniform additive retraction** inserts `len(K)` axes after `d`. That is right, and the `K` misalignment
mutant dies. Its padding is the sum of the operands' padding, as before; it is don't-care, and the engine masks on
entry.

**`aligned_padded_frame` returns `(frame, centers, movements[, seconds])`.**
- The centers are the old point's, zero-padded. Ragged uses `_embed`. Uniform masks the old centers, gathers the
  real slots first, then pads or truncates.
- They are zeros from the zero tensor.
- `timing` is bitwise neutral.
- The protocol text says all this, including centers that may be `None`.

**The bridge in the old `_aligned_padded_frame`** is two lines that drop the centers. Its tests pass.

**`from_t3` with a width below a rank: no part-1 test relies on it.**
- `test_geometry` sizes `n` and `r` from the point (`max + 1`); its shared point's ranks are read after `share`.
- `test_padded_start` packs `x_old` at `n = r = 4`, which is at least its ranks (2), and `x0` and the hint at their
  default widths, which equal their maximum ranks.
- P15's `test_uniform_with_forced_padding` (shared case) is the one affected, as the handover says; it is outside part 1.

## Findings

**1. (Should fix.) Four red subtests are part 1's own, and are not on the expected-red list.**
- *The tests:* `tests/test_dispatch.py::TestPaddedStartSeam::test_members_and_padded_start` and
  `::TestAlignmentTiming::test_timing_adds_no_program`, both layers.
- *Why they fail:*
  - the first unpacks the member as `F, m = geom.aligned_padded_frame(...)`, line 3429: "too many values to unpack
    (expected 2)";
  - the second compares `on[1]` with `off[1]` as movements, line 3513, where `[1]` is now the centers.
- *Why they are part 1's:* they test the member's contract, which part 1 changed. They are not old-kind fits.
- *Resolution:* update them in part 1, as `TestMemberTiming` was: three elements off and four on, centers compared
  bitwise, the movement and seconds indices moved by one.

**2. (Should fix.) The expected-red list differs from what is red.** I ran the whole suite.
- **Everything red is red for the stated reason, except finding 1.** Every other red subtest is labelled corewise,
  or is a corewise `adam` or derivative fit. Every traceback I opened ends in an old kind or the old backend model
  unpacking a four-store frame:
  - "expected 4, got 2" in `apply.py`, `entries.py`, `probing.py` or `sampling_derivatives.py`;
  - "expected 6, got 4" in `ufv_masking`;
  - the frontend's `T3Variations` built from the two-store frame;
  - one uniform geometry test comparing the backend frame with the frontend's `(B, G, G, G)`.

  No manifold subtest is red.
- **Predicted red but green:**
  - `test_uniform_fitting`: `TestKindTransposeOutputHygiene::test_kind_transposes`,
    `TestProblemStringArgs::test_geometry_string`;
  - `test_dispatch`: `test_jit_fitting`, `test_jit_geometry_as_arg`, `test_jit_uniform_geometry`,
    `test_jit_uniform_probing`, `test_jit_uniform_derivatives`;
  - `test_optimizers_frontend`: `TestLeastSquaresProblemAndKeywords::test_x0_cores_minimal_on_manifold_only`;
  - `test_sharing`: `TestSharedGeometry::test_gradient_consistency`;
  - `test_fitting`: `TestLocalModel::test_uniform_regularizer`, `test_matched_pair`.
- **Red but not predicted:**
  - `test_optimizers`: `TestBackendOptimizers::test_gradient_descent_descends`,
    `test_identity_regularizer_contributions`, `test_oracle_matches_frontend`, and
    `TestSwitchNewton::test_forced_at_zero_is_exact`, `test_threshold_disabled_is_gauss_newton`;
  - `test_uniform_fitting`: `TestUniformGeometryOps::test_ops_match_frontend`,
    `test_masks_loop_invariant_across_points`, `TestUniformProblem::test_plain_kinds_match_ragged`,
    `test_derivative_kinds_match_ragged`, `TestUniformOptimizers::test_adam_descends`;
  - `test_dispatch`: `test_jit_exact_local_model`, `test_jit_shared_uniform_exact`,
    `test_jit_uniform_exact_local_model_is_one_cache_key`;
  - `test_fitting`: `TestUniformLayerLocalModel` (both tests), and `TestExactLocalModel`'s corewise and shared
    subtests;
  - `test_uniform_curvature`: `test_m27_*`, `test_shared_uniform_subcase`.
- **Resolution:** replace the predicted list in the handover with the observed one. The 66 ids are in
  `reviewer2/actual_red.txt`. Part 2's reviewer will judge regressions against it, so it has to be the measured
  list. `TestUniformGeometryOps::test_ops_match_frontend` and `test_masks_loop_invariant_across_points` compare a
  geometry's frame with the frontend's. They are geometry tests: either fix their corewise case in part 1 (compare
  the point), or list them for part 4 with that reason.

**3. (Should fix.) The old four-store corewise frame gives a silent wrong regularizer value.**
- *The cause:* `CorewiseGeometryOps.base_point(frame)` returns any tuple it is given.
- *How it happens:* hand it the frame of `t3_corewise_frame(x)`, `(B, G, G, G)`. That is the form today's frontend
  corewise models still pass to the backend geometry, and the form a backend user's existing code builds. Then
  `IdentityRegularizer.value(geom, geom.base_point(frame))`:
  - sums `‖B‖² + 3‖G‖²`: measured 54.41 against the right 33.47, with no error;
  - while `retract` on the same frame raises (a length mismatch in `corewise_add`).
- *The uniform form:* the six-element frame happens to give the right `(B, G)`, by position.
- *Why it matters:* the plan's interim keeps the four-store frame in the frontend until part 4, and P20 may delete
  `t3_corewise_frame`. A structural check costs nothing and makes the wrong form loud now.
- *Resolution:*
  - ragged `base_point` raises a `ValueError` unless `frame_form(frame) == ('ragged', 'corewise')` (two tuples);
  - uniform `base_point` raises unless the frame has four elements.

  Structural, so it raises in both safety modes and under `jit`. Test: the four-store frame raises on each.

## Minor

- **Expect the frontend corewise models to stay red until part 4** (finding 2's `T3Variations` failures). The
  structural check of finding 3 makes the remaining interim paths raise instead of passing by luck.
- **The full suite took 33 minutes at `-n 2`** on this tree (many failures and slow tracebacks). Plan the part 2 and
  part 3 runs on the touched files.
