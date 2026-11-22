NOT YET, on one bookkeeping item only: four ids are missing from `P16_part1_EXPECTED_RED.txt`. They are red for the stated reason. Add them and I sign off; the code needs nothing more.

# P16 part 1 review, round 2

Reviewer, 2026-11-21, on `P16_part1_RESPONSE_r1.md`, the updated handover, `P16_part1_EXPECTED_RED.txt` and the diff.
Runs were one at a time, at `-n 2`. The working tree is untouched apart from this file.

## Round 1's findings

**1. The two `test_dispatch` tests: resolved.**
- `TestPaddedStartSeam::test_members_and_padded_start` unpacks `(frame, centers, movements)`, compares the timed call
  bitwise (the centers included), and checks that the centers are jax on jax.
- `TestAlignmentTiming::test_timing_adds_no_program` asserts 3 elements off and 4 on, with the indices moved.
- Both pass in my full run.
- `TestUniformGeometryOps` now compares the corewise backend point with the frontend's `(B, G)`, its shape and masks,
  and `U is G`. Both of its tests pass.

**3. The silent case: resolved.** `_require_frame_form` reads the frame with `frame_form`, the engine's reader. My own
probe:
- **The four-store `(B, G, G, G)`** on the ragged corewise geometry: `base_point`, `IdentityRegularizer.value` (the old
  54.41), `.gradient` and `retract` all raise a `ValueError` that names the member and the form.
- **The right form** gives 33.47, the cores' weight decay exactly.
- **On the uniform layers:** the six-element corewise frame raises on the uniform corewise geometry, and a uniform point
  raises in the uniform manifold `base_point` and `point_tangent`.
- **Under `jit`:** the check is structural and holds. No `test_dispatch` jit test went red because of it.

**The members left unchecked cannot change a value.**
- *Corewise, both layers:* `project`, `precompute`, `lift`, `lift_transpose`, `tangential_curvature`,
  `curvature_precompute`, `zero_point_frame`, `inner`, `stack_shape` and `point_norm_sq` never read the frame (or read
  only the geometry's own masks). `retract` and `point_tangent` go through the checked `base_point`. Verified on the
  ragged one: `project` gives bitwise the same result on the four-store and the two-store frames; `lift` returns its
  argument; the others return `None` or `()`.
- *Manifold:* `project`, `retract`, `precompute`, `curvature_precompute` and `lift` unpack four or six stores
  themselves, so a corewise point raises there. `tangential_curvature` goes through `base_point`.
- One form passes every check by design: the old four-store frame given to a *manifold* geometry. That is the
  substitution, a valid ragged manifold structure. Whether such a frame is orthonormal is a numerical question, for the
  frontend's safe mode (`is_orthogonal`), not this check's.

**2. The expected-red list: four ids are missing.** My full suite on this tree (31 min): 196 failed, 1848 passed,
**67** red ids. Against the file's 63:
- **None is listed and green.**
- **Four are red and not listed:**
  - `tests/backend/test_uniform_curvature.py::TestUniformExactLocalModel::test_garbage_in_the_frame_padding`
  - `tests/backend/test_uniform_curvature.py::TestUniformPlainKindsExactLocalModel::test_garbage_in_the_frame_padding`
  - `tests/backend/test_uniform_curvature.py::TestUniformDerivativeKindsExactLocalModel::test_garbage_in_the_frame_padding`
  - `tests/backend/test_uniform_curvature.py::TestUniformSharedExactModel::test_garbage_in_the_frame_padding`
- **They are red for the stated reason**, the same as `test_uniform_regularizer`:
  - their corewise subtests build the old backend `LocalModel` on a hand-made six-element `(B, G, G, G, shape, masks)`
    frame;
  - its exact `hvp` reaches `UniformCorewiseGeometryOps.base_point`, which now raises: "expected a uniform corewise
    point; got the form ('uniform', 'manifold')";
  - before round 1 they passed only because `(frame[0], frame[1])` happened to be `(B, G)`;
  - their manifold subtests pass.
- **Why they were missed:** the response re-measured `test_fitting.py` and `test_sharing.py` only. These sit in
  `test_uniform_curvature.py`.
- **Resolution:** add the four ids, with that reason, to the file. Part 2 rebuilds these models on the point.

The rest of the red set is unchanged from round 1's measured list, less the four ids fixed: every id is a corewise path
through the old kinds, the old backend model or a frontend model.

My lists: `reviewer2/actual_red_r2.txt` (measured) and `reviewer2/expected.txt`, in my scratch folder.
