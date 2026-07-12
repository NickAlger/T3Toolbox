# Verification records

Recorded numerical evidence behind claims the user docs state qualitatively. These are lab-notebook
entries: the numbers are from the referenced verification runs, and the corresponding automated
tests are the living guarantee.

## The three sampling transposes are Riemannian-compatible

(Stated qualitatively in `../entries_apply_probe.md` §4.) For the least-squares objective
`f(X) = ½‖S(X) − b‖²` on the fixed-rank manifold, with `S` each of the three sampling operations,
the matrix-free Riemannian gradient
`g = op_transpose(r, …, sum_over_probes=True).orthogonal_gauge_projection()` and the Gauss-Newton
Hessian `H V = (𝒥ᵀ 𝒥 V)` gauged were checked end-to-end at an orthogonal, minimal-rank frame:

| operator | adjoint `⟨r, 𝒥V⟩ = ⟨𝒥ᵀr, V⟩` | gradient vs finite-diff *along the retraction* | Gauss-Newton Hessian   |
|----------|-------------------------------|-----------------------------------------------|------------------------|
| entries  | exact                         | rel. err ~5e-8                                | symmetric ~5e-17, PSD  |
| apply    | exact                         | rel. err ~3e-7                                | symmetric ~1e-12, PSD  |
| probe    | exact                         | rel. err ~5e-7                                | symmetric ~3e-13, PSD  |

The adjoint identity and gradient/Hessian checks are pinned by the fitting and manifold test
suites (`tests/test_fitting.py`, `tests/test_manifold.py`).
