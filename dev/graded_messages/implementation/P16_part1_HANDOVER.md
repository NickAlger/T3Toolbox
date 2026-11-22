# P16, part 1 (geometry): handover

Implementer, 2026-11-21, on `p16-wip` at `1baf1335`. Updated after review round 1 (`P16_part1_RESPONSE_r1.md`). Built to `P16_DESIGN_NOTE.md`, `P16_DESIGN_REVIEW.md` and
Claude's rulings (D2; the uniform corewise retraction made direct). Nothing is committed.

## 1. What changed

**`t3toolbox/backend/geometry.py`**
- **`CorewiseGeometryOps`:**
  - `frame(x)` returns `((B, G), None)`, the point as two tuples of cores. The engine reads it as a ragged corewise call.
  - `base_point(frame)` is the frame itself.
  - `point_tangent`, `retract` and the shared post-pass are unchanged in code, and follow from `base_point`. The
    class docstring says the frame is the point.
- **`UniformCorewiseGeometryOps`:**
  - `frame(x_sc)` returns `((B_sc, G_sc, shape, masks), None)`: the point's `.data`, with the geometry's shape and
    plain masks.
  - `base_point(frame)` returns `(frame[0], frame[1])`.
  - `retract` is the new private `_uniform_additive_retraction` (about ten lines, as ruled): the point gets `len(K)`
    size-1 axes after the site axis `d`, then the direction is added. It no longer goes through
    `utv_corewise_retract` or `ut3_corewise_frame`.
  - The `var_masks` docstring now describes a core direction's masks at the point.
- **The `t3_corewise_frame` import is dropped:** `geometry.py` no longer uses it.
- **The `aligned_padded_frame` members** (ragged and uniform) return `(frame, centers, movements)`. With `timing=True`
  they return `(frame, centers, movements, seconds)`. The frame, centers and movements are bitwise those of
  `timing=False`. The centers are the old point's, zero-padded on every rank axis; from the zero tensor (or a
  structurally zero old point) they are zeros. The docstrings say why this is valid.
- **Structural form checks (round 1, finding 3).** A private `_require_frame_form(frame, expected, who)` reads the
  frame with the engine's `frame_form` and raises a `ValueError` on any other form. It is called by:
  - the four geometries' `base_point`;
  - the two manifold geometries' `point_tangent`.

  The corewise `point_tangent` and `retract` read the frame through `base_point`. The old four-store corewise frame
  `(B, G, G, G)` therefore raises instead of passing `G` three times (the regularizer's value was 54.41 against
  33.47, silently). Members that do not read the frame are not checked (response r1, finding 3).
- **The new private `_ufv_padded_centers(old_variations, frame_data)`** lives here, beside the uniform member:
  - it masks the old centers with their own variation masks;
  - it gathers their real slots first along every rank axis, in order (as `ufv_compact_frame` does for the frame);
  - it pads or truncates to the frame's variation shapes.

  It lives here and not in `uniform_padded_frames.py`, because `ufv_variation_shapes` is in `ufv_operations`, which
  `tests/test_import_graph.py` does not allow the padded-frame modules to import.

**`t3toolbox/backend/padded_frames.py`:** the new private `_fv_padded_centers(old_centers, frame)` embeds the old
centers in the leading slots (`_embed`), or returns zeros for `None`. It works in the frame's array type and dtype.
The module now imports `fv_operations` as a module, which is within the pinned import graph.

**`t3toolbox/backend/optimizers.py`:**
- **The `Geometry` protocol text:**
  - the corewise frame is the point, in both forms, with `None` centers;
  - the `aligned_padded_frame` member returns `(frame, centers, movements)` (`None` centers allowed), and a fourth
    element with `timing`.
- **`_aligned_padded_frame`** unpacks the new return and drops the centers. This is a two-line bridge so the old
  padded-start path keeps running until part 2 rewrites it and reads the centers.

## 2. Tests

**`tests/backend/test_geometry.py`:**
- **`TestTheCorewiseFrameIsThePoint`** (new), on the points `C=()`, `C=(2,)` and shared `(0, 0, 1)`:
  - **ragged:**
    - the frame is the point's own arrays, `frame_form` reads it as ragged corewise, and the centers are `None`;
    - `base_point` and `point_tangent` give the cores back;
    - `retract` at `K=()` and `K=(3,)` matches a dense construction of point plus projected direction, per `k`;
    - shared: the factors come out aliased although the direction was untied.
  - **uniform**, with forced padding on every axis and garbage in the padding of the point and the direction:
    - the frame is the point's arrays with this geometry's shape and masks, read as uniform corewise;
    - `retract` at `K=()` and `(3,)` equals the ragged retraction through dense tensors;
    - shared: the factors stay tied.
- **The P15 test's corewise assertions** now expect the point. They had checked the four-store frame.
- **`TestTheCorewiseFrameIsThePoint.test_a_frame_of_another_form_raises`** (round 1). These four cases raise in
  `base_point`, `point_tangent` and `IdentityRegularizer.gradient`:
  - the old four-store frame, on both corewise geometries;
  - a corewise point, on both manifold geometries.

  With the right form, the regularizer's value equals the cores' weight decay, and the manifold `base_point` returns
  `(D, L)`.
- **`tests/backend/test_uniform_fitting.py::TestUniformGeometryOps`** (round 1): its corewise case compares the
  backend point with the frontend frame's `(B, G)` (the frontend frame is `(B, G, G, G)` until part 4).
- **`tests/test_dispatch.py`** (round 1): `TestPaddedStartSeam::test_members_and_padded_start` and
  `TestAlignmentTiming::test_timing_adds_no_program` unpack the member's three and four elements. They compare the
  centers bitwise, and check that the centers are jax on jax.

**`tests/backend/test_padded_start.py`:**
- **`TestThePaddedFrameCenters`** (new):
  - **both layers, with no pass, one pass and the defaults, from the old point and from the zero tensor:**
    - the consistency residual is below `1e-12`;
    - the probe `z` read through the centers (`Vc` route of the engine) equals `dense_probe` of `x0`;
    - from the zero tensor the centers are exactly zero.
  - **a stack `C = (2,)` on both layers,** with garbage in the uniform old point's padding: residuals of shape `C`,
    below `1e-12`.
  - **the uniform helper directly:**
    - garbage in the old centers' padding changes nothing, bitwise;
    - the old centers with the last (padded) slot of every rank axis moved to the front give the same output. The
      real slots are then not a prefix, but keep their order.
- **`TestMemberTiming`:** the members return three elements off and four on. The centers are compared bitwise like
  the frame; the movements and seconds indices move by one.
- **The test-side custom member `_KwargsMember`** returns `(frame, None, movements)` under the new contract.

**Mutation pass** (scratch copy of `t3toolbox/` and `tests/`, deleted). All 9 mutants were killed:

| Mutant | Result |
| --- | --- |
| the ragged centers padded at the trailing slots | killed |
| the ragged centers all zero | killed |
| the uniform helper without masking | killed |
| the uniform helper without the real-first gather | killed (by the slot-moving test) |
| the uniform retraction without the `K` reshape (`d = 3`, `K = (3,)`: a silent misalignment) | killed |
| ragged corewise `retract` without the shared projection | killed |
| uniform corewise `retract` without the shared projection | killed |
| (round 1) ragged corewise `base_point` without the form check | killed |
| (round 1) uniform corewise `base_point` without the form check | killed |

Not counted: `x0`'s own centers (`frame(x0)[1]`) at the padded frame is an **equivalent** mutant. Orthogonalizing a
zero-padded point reproduces the old centers, to 1.1e-14 (review r1). Genuinely wrong centers give residuals of
order 1.

## 3. Runs (current env, `-n 2`, one run at a time)

**Round 1:**
- **The part's six files,** with `test_uniform_fitting.py::TestUniformGeometryOps` and `test_dispatch.py`'s
  `TestPaddedStartSeam` and `TestAlignmentTiming`: **187 passed, 5 failed** (1829 subtests). The 5 failures are on
  the expected list.
- **`tests/test_fitting.py` and `tests/test_sharing.py`,** to measure the form check's effect: red exactly on the
  list's ids in those files, among them the one new id (section 4).
- **Page doctests** (`python -m doctest`): `docs/numerical_contracts.md`, `docs/sharing.md` and
  `docs/contributor/curvature_internals.md` pass. `docs/fitting_and_optimization.md` fails at lines 73, 75 and 1190:
  corewise `adam` and `newton_cg` through the old `Problem`, expected.
- **Module doctests** of `geometry.py`, `padded_frames.py` and `optimizers.py`, and `test_import_graph`: pass.

**Round 0:** the part's six files gave 181 passed and 5 failed. The floor env (`tttt`) passed the new and changed
test classes (15).

**The reviewer ran the full suite on round 0's code:** 66 ids red, all for the stated reason except the two
`test_dispatch` tests of the member's contract (fixed in round 1).

## 4. Expected red, measured

`dev/graded_messages/implementation/P16_part1_EXPECTED_RED.txt` has 63 test ids, plus three page lines. It is:
- the reviewer's measured 66;
- less the four round 1 fixed (`TestUniformGeometryOps` ×2, and the two `test_dispatch` member tests);
- plus `tests/test_fitting.py::TestLocalModel::test_uniform_regularizer`. The form check turned it red: the uniform
  corewise frontend model passed the frontend's six-element frame to the backend `point_tangent`, which was right
  by luck and is now a `ValueError`.

Every id is a corewise fit or model that reaches one of:
- the old kinds or the old backend `LocalModel` (which unpack a four-store frame);
- the backend corewise geometry with the frontend's four-store frame.

No manifold test is red. Part 2 judges regressions against this list. The frontend models' ids stay red until
part 4.

## 5. Choices and departures

- **The uniform centers helper is in `geometry.py`**, not `uniform_padded_frames.py`, because of the import pin
  (section 1). It is private, as is the ragged one; I added no public name. A backend user gets the centers from
  the member, and in part 2 from `padded_start`.
- **The uniform helper gathers real slots first** rather than assuming prefix masks. On the member's path the old
  frame comes from `ut3_orthogonal_representations`, whose masks are prefixes, so there the gather is a no-op. It
  makes the helper correct for any mask pattern, matching `ufv_compact_frame`, and a test pins it.
- **Retraction padding:** the uniform corewise retraction does not mask its result; its padding is the sum of the
  operands'. That matches the old `utv_corewise_retract`, and the padding is don't-care.

## 6. Found outside the part

- **`UniformTuckerTensorTrain.from_t3(x, n=..., r=...)` silently truncates** a rank above the given width. At Tucker
  ranks `(5, 5, 2)` and `n = 4` it returns ranks `(4, 4, 2)`, with 30% relative error, and no error is raised.
- **P15's `TestTheFrameReturnsTheCenters.test_uniform_with_forced_padding` is affected,** in its 'shared' case: the
  shared point has Tucker ranks `(5, 5, 2)` and is packed with `n = 4`. The test checks a truncated tensor (still
  self-consistent, so it passes). My new tests size `n` and `r` from the point. I propose `from_t3` raise on a width
  below a rank (structural); this is outside P16. **Decided by the maintainer** (2026-11-21): `from_t3` raises for
  `n` or `r` below a rank, pointing to `resize`, in a small slice after P16; not changed here.

## 7. For part 2 (recorded as ruled or found)

- **D5, the frontend path.** Today's uniform frontend model builds its backend geometry with `from_frame`
  (`fitting.LocalModel._bgeom`); `uniform_least_squares_problem` uses `from_point`.
  - The new frontend `local_model` can compute the frame (and centers) once with `ut3_orthogonal_representations`,
    build the geometry with `from_frame`, and pass `frame=` and `centers=` to the backend `local_model`.
  - So the minimal-rank guard can go in `UniformManifoldGeometryOps.from_point`, with shared minimality when
    `sharing` is given.
  - Part 2 tests that `gradient`, `hvp` and `gn_quadratic` work at a non-minimal uniform point.
- **The `Kp` re-sweep of `nu_L`.** In exact mode on a manifold frame, every `Kp` query recomputes `nu_L` and its tilde
  (not on Part 1's hold list), as today's two-hole kernels re-sweep. That is parity, not a regression; one line for
  "Carried to Part 2".
- **Centers with a given frame** (D3): `local_model(..., frame=None, centers=None)`; centers without a frame raise;
  the docstring says wrong centers are a silent wrong answer. Test: the model's `f` at `padded_from`'s start equals
  `point_objective(x0)`.
- **The bridge in `_aligned_padded_frame`** that drops the centers goes when part 2 rewrites the padded start.
