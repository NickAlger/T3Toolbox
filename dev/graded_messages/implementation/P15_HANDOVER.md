# P15: the geometry returns the centers — handover

Slice P15 of `PART1_PLAN.md`, built to `P15_DESIGN_NOTE.md` and the coordinator's two rulings: the uniform centers are
the pair of supercores, and `aligned_padded_frame` keeps returning only a frame. Written by the implementer,
2026-11-21, against `graded-messages` at `97d12113`. Nothing is committed.

## 1. What changed

**The four backend geometries' `frame(x)`** (`backend/geometry.py`) return `(frame, centers)`:

| Geometry | `centers` |
| --- | --- |
| `ManifoldGeometryOps` | the second result of `t3_orthogonal_representations`, which it already computed |
| `UniformManifoldGeometryOps` | `(Vc_sc, Hc_sc)`, the two supercores of the second result of `ut3_orthogonal_representations` |
| `CorewiseGeometryOps`, `UniformCorewiseGeometryOps` | `None`; the frame is still the four-store `(B, G, G, G)` |

The signatures carry the new return. Sharing (the `groups` field) does not touch `frame`.

**The library call sites:**
- `backend/geometry.py`: the aligned padded frame's `old, _ = self.frame(x_old)`, and `point_norm_sq`'s
  `self.frame(x_sc)[0]`;
- `backend/optimizers.py`: `_local_model_phases`' `frame, _ = problem.geom.frame(x_cores)`.

`aligned_padded_frame` itself still returns a frame only.

**The `Geometry` protocol** (`backend/optimizers.py`): the `frame` line and the class docstring state the pair, and
that a corewise geometry returns `None`.

**The test call sites** are the plan's list, unchanged since P12 to P14, each now taking `[0]`:
- `test_geometry` 5, `test_curvature` 4, `test_uniform_curvature` 2, `test_dispatch` 4, `test_optimizers` 2;
- `test_padded_start` 7, the subclass's `aligned_padded_frame` included;
- `test_uniform_fitting` 1;
- `test_zero_reference` 2.

The zero-reference wrapper geometry delegates `frame`, so it passes the pair through unchanged. No doctest called a
backend `frame`.

## 2. Tests (`tests/backend/test_geometry.py::TestTheFrameReturnsTheCenters`)

- **Ragged** (`C = ()`, `C = (2,)`, and a shared geometry on a tied point):
  - frame and centers bitwise equal `t3_orthogonal_representations(x)`;
  - `fv_centers_consistency_residual(frame, centers)` < 1e-12;
  - the corewise geometry (also with sharing) returns `None`, and its frame is `t3_corewise_frame(x)`, the same arrays.
- **Uniform with forced padding** (`N = 10`, `n = r = 4`; the same three points):
  - the frame's supercores and shape, and the two-supercore centers, bitwise equal `ut3_orthogonal_representations`;
  - `ufv_centers_consistency_residual` < 1e-12;
  - the uniform corewise geometry returns `None` and the corewise frame.

## 3. Mutation pass (scratch copy, deleted)

5 mutants:

| Mutant | Result |
| --- | --- |
| the manifold centers' two parts swapped | killed |
| the uniform centers as the 4-element data | killed |
| a ragged corewise geometry returning centers | killed |
| a uniform corewise geometry returning centers | killed |
| `point_norm_sq` reading the centers in place of `point_tangent(frame)` | survived the first pass; killed after round 1 |

**Retracted (round 1).** I first called the survivor equivalent and suggested that `point_norm_sq` could read the
centers. **Both are wrong.** At every site the center in the frame's hole gives the point, so the tangent
`(frame, centers)` is the sum of `2d` copies of `X`: `2d X`, with corewise inner `2d |X|^2`. Reading the centers would
make the regularizer `2d` times too large. The mutant survived because the only uniform test of `point_norm_sq`
compared it with `R.value`, which calls the same function. `TestPointNormSq` (round 1) now kills it: the relative
error is 5 at `d = 3`, that is `6 |X|^2` against `|X|^2`.

## 4. For P16 (recorded as ruled)

**At a padded-start frame the oracle runs without `Vc`.** `aligned_padded_frame` returns no centers. I checked that the
engine then computes `z` by a route without `Vc`, as on a corewise point. The closure for `('z', 'f', 'g')` on a
manifold call with no `Vc`:
- probe `z` is `z_step` reading `(eta_D, D)`, with no need;
- apply and entries `z` is `close_step` reading `mu`;
- no row of any of the three closures reads `Vc`.

A P16 test should assert this route at a padded start: the probe `z` there must equal the one at the frame with
centers.

## 5. Environments and gate

- **`t3toolbox`**: the gate's `quick` step over the 11 affected files passed (604 tests, 4,976 subtests).
- **`tttt`**: `test_geometry`, `test_zero_reference` and `test_padded_start` pass (75).
- **The full gate at `-n 2`**: below.

The full gate at `-n 2` (logs in `scratchpad/gate_P15`) passed every step:
- **suite**: 1854 passed, 5 skipped, 1 xfailed, 90,562 subtests, in 2012 s; peak 7.7 GB; **lowest MemAvailable
  4247 MB**;
- **doctest-modules**: 319;
- **quickstart**: pass;
- **pages**: 47;
- **examples**: 14;
- **sphinx**: 0 warnings.

## 6. Round 1 (`P15_RESPONSE_r1.md`)

- **`TestPointNormSq`** tests `point_norm_sq` on all four geometries against an independent reference:
  - manifold: the dense `|X|^2`;
  - corewise: the real cores' sum of squares;
  - both stacked and unstacked, the uniform ones with garbage in the padding;
  - it kills the centers mutant.
- **The handover's equivalence claim and P16 suggestion** are retracted (§3).
- **Three docstrings corrected:** `ManifoldGeometryOps.frame`, `UniformManifoldGeometryOps.frame` and the `Geometry`
  protocol. The centers each give the point in the frame's hole; `(frame, centers)` as a tangent is `2d X`; the point
  as a tangent is `point_tangent(frame)`.

**Runs.** Only tests and docstrings changed, so the full gate was not re-run. Instead:
- `test_geometry` (26) and the module doctests of the two touched modules pass on both environments;
- the gate's `doctest-modules` step passes (319), and its `sphinx` step with 0 warnings, the new fully qualified
  `:py:meth:` included. Logs are in `scratchpad/gate_P15r1`.
