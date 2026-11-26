SIGN OFF: `point_norm_sq` is now pinned on all four geometries, my mutant is killed, the P16 suggestion is retracted and the docstrings say what the centers are.

# P15 review, round 2

Reviewer, 2026-11-21, on `P15_RESPONSE_r1.md` and the updated tree (uncommitted, on top of `99f65dc7`; only tests and
docstrings changed). Everything was run single-process. The mutant copy has been deleted, and the working tree is
untouched apart from this file.

## The test: right, with the right corewise reference

`TestPointNormSq` checks `point_norm_sq` against independent references, at `C = ()` and `(2,)`, per element:
- **The manifold geometries:** the dense Hilbert-Schmidt `|X|^2`.
- **The corewise geometries:** `sum_i |core_i|^2` over the real cores. That is right: on `COREWISE` the regularizer
  `1/2 lambda |x|^2` is weight decay on the raw cores. `CorewiseGeometryOps.point_norm_sq` is defined as exactly that
  (`corewise_stack_dot(x, x, n_stack)`), and the uniform twin is the masked coordinate inner product. The reference
  is computed directly from `x.data`, not through the geometry, so it is not circular.
- **The uniform geometries:** forced padding (`N = 10`, `n = r = 4`), with garbage (1e3) in every padded slot of the
  point. So the masked inner products are also shown to ignore the padding.

## My mutant: killed

The uniform `point_norm_sq` reading the centers (`v_x = self.frame(x_sc)[1]`, which gives `2d |X|^2`) now fails 2
subtests of `test_geometry` (the uniform manifold case at both `C`). The ragged `point_norm_sq` does not go through the
frame at all (it left-orthogonalizes and reads the last core), so it has no such mutant.

## The retraction and the docstrings

- The P16 suggestion is retracted (the response's §2).
- `ManifoldGeometryOps.frame` now says what the centers are:
  - each, in the frame's hole at its site, gives the point;
  - so `(frame, centers)` read as a tangent is `2d` times the point;
  - the point as a tangent is `point_tangent(frame)`.
- The uniform twin and the protocol text follow it.

The implementer reports the touched files, the doctests and Sphinx passing on both envs. My run of `test_geometry`
passes, 26 tests.
