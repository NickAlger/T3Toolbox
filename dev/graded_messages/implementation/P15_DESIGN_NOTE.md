# P15 design note: the geometry returns the centers

Implementer, 2026-11-21, against `graded-messages` at `97d12113` (P14 committed). Read-only so far.

## 1. What `centers` is, per geometry

The backend `frame(x)` returns `(frame, centers)`, with `centers = (Vc, Hc)` or `None`.

| Geometry | `centers` |
| --- | --- |
| `ManifoldGeometryOps` | the second result of `t3_orthogonal_representations(x)`, which it computes and discards today: the pair of tuples `(Vc, Hc)` |
| `UniformManifoldGeometryOps` | the second result of `ut3_orthogonal_representations` is a `UT3Variations.data` (four elements); `centers` is its **two supercores**, `(Vc_sc, Hc_sc)`, the form `ufv_centers_consistency_residual` takes. The shape and masks are the frame's (derivable), so they are not repeated |
| `CorewiseGeometryOps`, `UniformCorewiseGeometryOps` | `None`; the frame stays the four-store `(B, G, G, G)` until P16 |

**Shared geometries.** Sharing is the `groups` field of these same four classes (`with_sharing`), and `frame` ignores
it, so the rows above hold unchanged.

**Two other objects are touched:**
- **`test_zero_reference`'s wrapper geometry** delegates `frame` to its base, so it passes the pair through with no
  change; only its callers unpack.
- **The aligned padded frame** (`aligned_padded_frame`) is a different member. It returns a frame not built from an
  orthogonal representation of `x`, so it has no centers, and **it keeps returning a frame only**. Inside it,
  `old = self.frame(x_old)` becomes `old, _ = ...`. In P16 a padded start then queries with `Vc=None`, and the probe
  `z` goes through the base point's sweep, as it does today.

**The `Geometry` protocol docstring** (`backend/optimizers.py`, `frame(self, x_cores)`) gets the new return. The
frontend geometries and `SharedGeometry` do not call the backend `frame` and are unchanged (O-4).

## 2. Is the plan's call-site list complete today?

I searched again after P12 to P14 and separated backend from frontend calls by the argument (raw `.data` or a
supercore pair, against a `TuckerTensorTrain` object) and by the object.

**The list is complete and exact:**
- **the library:**
  - `backend/optimizers.py:509` (`_local_model_phases`);
  - `backend/geometry.py:411` (the aligned padded frame's `old`);
  - `backend/geometry.py:803` (`point_norm_sq`);
- **the tests**, with the plan's counts:
  - `test_geometry` 5, `test_curvature` 4, `test_uniform_curvature` 2, `test_dispatch` 4, `test_optimizers` 2;
  - `test_padded_start` 7 (one in its subclass's `aligned_padded_frame`);
  - `test_uniform_fitting` 1 (`_compare_ops`);
  - `test_zero_reference` 2, plus the wrapper.

**Not touched:**
- The P14 code (`LeastSquaresOracle`) does not call a backend `frame`.
- The `.frame(` calls in `test_sharing`, `test_fitting`, `test_uniform_manifold` and the other `test_dispatch` sites
  are on frontend geometries.
- No module or page doctest calls a backend `frame`. The one in `backend/sharing.py` is on `sg.shared_manifold`, a
  frontend geometry.

## 3. Tests (in `tests/backend/test_geometry.py`)

- **Ragged:** on `C = ()` and `C = (2,)` and a shared geometry, the manifold `centers` equal the second result of
  `t3_orthogonal_representations` (bitwise), and `fv_centers_consistency_residual(frame, centers)` is at rounding
  level (< 1e-12).
- **Uniform:** the same with `ut3_orthogonal_representations`' supercores and `ufv_centers_consistency_residual`, at
  forced padding.
- **The corewise geometries:** both return `None`, and their frame is unchanged.
- **The frame:** the first result is bitwise the frame `frame(x)` returned before (against
  `t3_orthogonal_representations(x)[0]`).
- **`test_dispatch`:** the existing jitted `frame` calls now unpack the pair, so the centers come out of a compiled
  `frame` as jax leaves.

## 4. What I would change

Nothing in the plan's scope. Two small points to rule on:
1. **The uniform centers are the two supercores,** not the four-element `UT3Variations.data` (section 1).
2. **`aligned_padded_frame` keeps returning a frame only** (section 1). The alternative, returning
   `(frame, None)` for symmetry with `frame`, would add a second unpacking at its four call sites for no information.
