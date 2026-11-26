# P15, response to review round 1

Implementer, 2026-11-21, answering `P15_REVIEW_r1.md`. Nothing is committed. Only tests and docstrings changed; no
code did.

## 1. `point_norm_sq` against an independent reference: done; the mutant dies

`tests/backend/test_geometry.py::TestPointNormSq` covers all four backend geometries:
- **the manifold geometries:** `point_norm_sq` against the Hilbert-Schmidt `|X|^2` of the dense tensor;
- **the corewise geometries:** against the weight decay `sum_i |core_i|^2` of the real cores, which is what
  `point_norm_sq` is there, so the reference is that sum and not `|X|^2`.

It runs at `C = ()` and `C = (2,)`, per element, with the uniform two at forced padding (`N = 10`, `n = r = 4`) and
garbage (1e3) in the padding of the point. The tolerance is 1e-12 relative.

The mutant `point_norm_sq` reading the frame's centers now fails it, with a relative error of 5 at `d = 3`: `6 |X|^2`
against `|X|^2`, which is the `2d |X|^2` of the review.

## 2. The P16 suggestion: retracted

`P15_HANDOVER.md` §3 now records:
- that the survivor is not equivalent;
- why it survived (the circular `R.value` comparison);
- that `TestPointNormSq` kills it.

The suggestion that `point_norm_sq` read the centers is withdrawn.

## 3. The docstrings: corrected

I searched the tree (`t3toolbox/`, `docs/`) for "own coordinates", "point's coordinates", "tangent-shaped" and
similar phrases. The only offenders were my three P15 texts.

`fv_operations.fv_centers_consistency_residual` already says it right ("the base point written into the frame's
hole: for every site"), and `docs/contributor/fitting_internals.md` §"The base point as a tangent" is about
`point_tangent`, and correct.

| Text | Before | Now |
| --- | --- | --- |
| `ManifoldGeometryOps.frame` (comment and docstring) | the point's own coordinates | at every site the center in the frame's hole is the point; read as a tangent, `(frame, centers)` is `2d` times the point; the point as a tangent is `point_tangent` of the frame (a fully qualified reference) |
| `UniformManifoldGeometryOps.frame` (comment) | the point's coordinates | the supercores of the centers, as the ragged |
| The `Geometry` protocol's docstring (`backend/optimizers.py`) | the point's own coordinates | the same correction |

## 4. Runs

Only tests and docstrings changed, so I did not re-run the full gate. I ran:
- **the touched files on both environments:** `tests/backend/test_geometry.py`, 26 tests on each;
- **the module doctests** of `backend/geometry.py` and `backend/optimizers.py` on both;
- **the gate's `doctest-modules` and `sphinx` steps**, for the new cross-reference: `doctest-modules` 319 passed, `sphinx` 0 warnings.
