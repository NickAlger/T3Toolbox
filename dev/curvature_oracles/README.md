# `dev/curvature_oracles/` — dense oracles for the curvature (Weingarten) term

Standalone numpy scripts (no `t3toolbox` import, no jax) backing the derivations in
[`../riemannian_newton_curvature_plan.md`](../riemannian_newton_curvature_plan.md). They work on
*dense* tensors at toy sizes, build tangent projectors from first principles (Koch–Lubich formula
or the Jacobian of the parameterisation), and compare hand formulas against finite differences.
They are the reference a future `hvp`-with-curvature test should be checked against; once that
lands they can be ported to `tests/` against the real `MANIFOLD`/`tv_retract` surface.

```
python -I dev/curvature_oracles/check_tucker_weingarten.py
python -I dev/curvature_oracles/check_t3svd_retraction_second_order.py
```

## `check_tucker_weingarten.py` — the Tucker-layer Weingarten map (plan §2.2)

Random Tucker point, d = 3, sizes (7, 6, 8), ranks (3, 2, 3). Verifies the explicit block formula for
`W_X(ξ, N)` (plan §2.2) against finite differences of the tangent projector, the identity
`⟨W(ξ,N), η⟩ = ⟨N, D²Φ[ξ̂, η̂]⟩` + symmetry, and the Hessian quadratic forms along three retractions.
Captured output (2026-10-25):

```
check 1  |W_formula - W_fd| / |W| = 1.864047016176575e-09
check 2  <W(xi,N),eta> = -23.143461592076118  <W(eta,N),xi> = -23.143461592076115  <N, D2Phi[xi,eta]> = -23.143461592076104
check 3  Riemannian Hessian quadratic form:
   formula  |A xi|^2 + <N, D2Phi[xi,xi]>           = 9968.185888614067
   FD along exact projection retraction (HOOI)     = 9968.150331701509
   FD along HOSVD retraction                        = 9968.147534777927
   FD along parameterisation retraction             = 9607.964060705854
   level-1 formula |A xi|^2 + <A^T r, D2Phi[xi,xi]> = 9607.911866424914
   Gauss-Newton  |A xi|^2                            = 10806.920630247569
check 4  exact projection : |P acc| = 8.249e-04   |Pperp acc| = 5.139e+01   |Pperp acc - Pperp D2Phi| = 1.092e-03
check 4  HOSVD            : |P acc| = 9.750e-04   |Pperp acc| = 5.139e+01   |Pperp acc - Pperp D2Phi| = 1.092e-03
check 4  parameterisation : |P acc| = 2.571e+01   |Pperp acc| = 5.139e+01   |Pperp acc - Pperp D2Phi| = 9.546e-05
```

Reading it: the FD second-difference noise floor with h = 1e-3 is ~1e-3 (the exact projection's
`|P acc|`, analytically zero, sits there), so HOSVD's tangential acceleration is also zero at this
resolution — HOSVD is a second-order retraction on the Tucker manifold (proved in the plan), and its
FD Hessian agrees with the exact-projection one and with the formula. The parameterisation
retraction is first order (`|P acc| = 25.7`) and reproduces the level-1 (core-space) form instead.

## `check_t3svd_retraction_second_order.py` — T3-SVD is second order (plan §2.3)

Random **minimal-format** T3 point, d = 4, sizes (5, 6, 6, 5), Tucker ranks (2, 3, 3, 2), TT ranks
(2, 3, 2); the TT core is the TT-SVD of a random dense core so it is well conditioned. The tangent
projector comes from the SVD of the FD Jacobian of `Φ` (dimension check: 57 = formula). Captured
output (2026-10-25):

```
Jacobian singular values: max 9.156e+00, min nonzero 9.164e-01, next 4.091e-10
tangent space dim from Jacobian = 57, expected = 57

Tangential part of the retraction acceleration (-> 0 like h^2 for a second-order retraction):
  T3-SVD: h=4.0e-02  |P acc| = 3.551e-05  |P acc_richardson| = 4.517e-08   |acc| = 2.768e-01
  T3-SVD: h=2.0e-02  |P acc| = 8.879e-06  |P acc_richardson| = 2.824e-09   |acc| = 2.768e-01
  T3-SVD: h=1.0e-02  |P acc| = 2.220e-06  |P acc_richardson| = 2.121e-10   |acc| = 2.769e-01
  T3-SVD: h=5.0e-03  |P acc| = 5.547e-07  |P acc_richardson| = 7.654e-10   |acc| = 2.769e-01

Contrast — parameterisation retraction Phi(theta + t theta_dot) (first order only):
  h=1.0e-02  |P acc| = 1.371e-01   |acc| = 3.089e-01
  h=5.0e-03  |P acc| = 1.371e-01   |acc| = 3.089e-01

Quadratic forms <xi, H xi> for f = 0.5|A x - y|^2:
  Gauss-Newton  |A xi|^2                                   = 80.949262
  Riemannian Hessian  |A xi|^2 + <N, II(xi,xi)>            = 87.752890
  FD d2/dt2 f(T3-SVD retraction)                           = 87.752811
  core-space (level 1)  |A xi|^2 + <A^T r, D2Phi[xi,xi]>   = 85.637604
  FD d2/dt2 f(parameterisation retraction)                 = 85.637646
  |Riemannian grad| = 373.075,  |normal part N| = 1066.232
```

Reading it: `|P acc|` falls by 4× per halving of h (pure O(h²) FD error, Richardson → 1e-10), so the
tangential acceleration of the T3-SVD retraction is exactly zero: second order. Hence the FD Hessian
along `tv_retract` is the exact Riemannian Hessian — the test oracle for the curvature term. The
parameterisation retraction's tangential acceleration is a constant 0.137 (first order only).

**Pitfalls met while writing these.** (i) A point whose Tucker rank exceeds what the TT ranks allow
(`r_1 > ρ_1`) is not in minimal format: the Jacobian rank drops (43 vs 52 in a first attempt) and the
"manifold" is the smaller one. (ii) A random Gaussian TT core can have tiny bond singular values; a
unit tangent vector then lifts to core variations of norm ~1e3 and every second difference is
roundoff. Both are exactly the regimes in which the curvature term itself is large (plan §4.4).
