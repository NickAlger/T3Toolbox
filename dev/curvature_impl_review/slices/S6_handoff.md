# S6a + S6 handoff (implementer) -- DONE; base main @ 8036fb2; tree UNCOMMITTED (8 files). Scratch (session-local, not archived): impl/
## Files
- backend/geometry.py: `UniformManifoldGeometryOps.from_frame(frame_data, plain_masks, sharing)` (C.8: var_masks =
  ufv_variation_masks(frame_data[5]), no re-orthogonalisation, no damping arg).
- fitting.py: `_bgeom` -> from_frame (uniform manifold; corewise keeps from_point, which never orthogonalised);
  `LocalModel.curvature` (one new leaf field) + derived `is_exact` property; `hessian(p)`, `quadratic(p)`,
  `curvature_hessian(p)`, `curvature_quadratic(p)` (raise unless exact; delegate to a cached backend-LocalModel view
  `_blm` = the backend's own curvature_hvp / curvature_quadratic), mode-aware `evaluate`, `retract(p)` (geometry.retract,
  shared_data=geometry_aux when set); exact `gradient` = `_tangent_at(frame, curvature.misfit_gradient)` (+ g_R);
  `_with_curvature(model, hessian, curvature_damping, who)`; all 7 sites take `hessian`/`curvature_damping`; pytree
  flatten via `partition_static` with `curvature` as a child; doctest in probe_derivatives_model (exact quadratic ==
  d2/dt2 f(retract(tp)), GN misses it; hessian matches quadratic; curvature_* raises on GN).
- optimizers.py: `least_squares_problem(..., hessian='gauss_newton')` (validated first; threaded to
  `_resolve_chunk_size` on both paths) + doctest; `newton_cg(+hessian, curvature_damping, step_control, tr_radius0)`
  (appended after on_line_search_failure, backend defaults; hessian validated as 'newton_cg' before setup, passed to
  least_squares_problem and the backend) + exact doctest; docstring. manifold.py: retract "second order" sentence (C.13).
- tests: test_fitting.py +TestUniformBackendGeometryFromFrame (2), +TestExactLocalModel (10: T18 every clause, x07 FD);
  test_optimizers_frontend.py: `not_yet_frontend` DELETED (S6 gate), T3POLYNOMIAL_KEYS += chunk_size, callback, hessian,
  forwarding test += 4 keywords + hessian->least_squares_problem spy, +TestFrontendExactNewton (4: bitwise vs backend,
  T21, chunk 'auto' spy, no-effect/unbuilt errors); test_dispatch.py +2 (exact frontend model jit once across points x
  dampings, int/mask leaves gone; frontend newton_cg exact use_jit). CHANGELOG [2026.3.0] (below).
## Frontend == backend (ragged, probe_derivatives o2, W=40, restart): `topt.newton_cg(**kw)` vs `bopt.newton_cg(
  least_squares_problem(...,hessian=).problem, .x0_cores, **kw)`: losses ==, every history field == except wall_time,
  final cores array_equal -- 5 configs (exact default; exact cd=0 p=1 Δ0=.3; exact+line_search+warm; GN+TR; GN) and
  uniform GN. Exact model == backend exact model: gradient / misfit gradient / curvature_hvp / curvature_quadratic /
  quadratic / gn_quadratic bitwise, hessian 1e-12 (frontend sums two gauged halves; backend projects once).
## T21 through the frontend (defaults; = S5): W=60 exact 8 outer / 0 rej / 27 CG vs GN 28 / 124 CG; W=120 exact 9/0
  (zero ref), warm 10 outer / 28 CG, power 1: 7 / 24, GN 37. Test pins <=10 and GN > 2x (BLAS-robust).
## x07 FD (manifold): |hessian(p) - FD_h of P_x grad f(R(hp))| / |FD| = O(h^2): 1.3e-5 @1e-4 -> 1.3e-7 @1e-5 (min.),
  8.6e-8 (d4), 2.2e-9 (reg.); GN off by 0.65-1.1. Test h = 5e-4*sigma_min, bound 1e-5, GN > 1e3x.
## Mutants (impl/mutate.py on a scratch copy): 15/15 RED (gradient recompute, hessian/quadratic/evaluate w/o curvature,
  _bgeom->from_point, from_frame w/o gauge shift, retract w/o companion, no kind check, damping dropped, GN damping
  accepted, hessian not to least_squares_problem / resolver, step_control dropped, pytree drops curvature, flatten w/o
  partition_static (caught by the int-leaf assert)).
## Gates (final, after all edits): targeted 11 files 479 passed / 1,645 sub; module doctests 210 (S5: 209); pages +
  quickstart OK; sphinx -W exit 0 (clean build); full suite -n 2: dispatch+backend 302 passed / 1,963 sub; the rest 728
  passed, 1 skipped / 42,465 sub (S5: 300 / 712); slow tier SERIAL 5 passed / 12,790 sub.
## Conflicts / judgment calls
1. B1/C.13: no `hessian` field; `curvature` leaf + `is_exact` property. Frontend reuses the backend PRIVATE
   `bopt._check_hessian` / `_require_exact_capable` (same checks + messages, plan "with the same messages"; no
   frontend copy). No precedent of frontend->backend-private calls; alternative = a public backend helper (n1, S4).
2. curvature_hessian/quadratic delegate to a backend LocalModel view (`_blm`) rather than re-coding the lift/two-hole
   (frontend wraps .data and delegates); hessian = gn_hessian + curvature_hessian literally (C.13 table; 2 Pi's).
3. hessian/quadratic/evaluate skip the add when the term vanishes (d=1 manifold), mirroring backend `quadratic`.
4. Mode checks run AFTER the frame/sweep build (one helper, C.13); errors are structural either way.
5. Frontend flatten switched to partition_static (plan M3): FIXES a latent trace of the uniform sweep's mode-size ints
   (CHANGELOG Fixed bullet). New keywords appended at the end of newton_cg (no positional call breaks).
6. least_squares_problem(hessian=) only sizes chunk 'auto' (S4 m6); docstring says it does not fix the mode.
7. CHANGELOG: least_squares_problem(hessian=) noted in the existing breaking `_setup` bullet + Added signature (it is
   new in 2026.3.0, so not itself breaking vs 2026.2.0); **kwargs bullet lists the 4 keywords; S4/S5 "frontend
   follows" phrases updated; new Added bullets (frontend exact mode; from_frame); Fixed (pytree ints).
8. T3POLYNOMIAL_KEYS lacked chunk_size/callback (d05/o01/b01 pass them) -- added; every §K key is explicit.
9. Uniform exact: ValueError naming UniformProbeDerivativesKind (the stub's marker) at model construction / newton_cg
   iteration 0; shared manifold: NotImplementedError naming S7; shared corewise exact: live and == backend.
## Draft commit message
frontend: the exact Hessian on fitting.LocalModel and optimizers.newton_cg (S6a, S6) -- UniformManifoldGeometryOps.
from_frame (the model's backend geometry without re-orthogonalising); the factories' hessian= / curvature_damping=,
the derived is_exact, hessian / quadratic / curvature_hessian / curvature_quadratic / retract, a mode-aware evaluate;
newton_cg's hessian / curvature_damping / step_control / tr_radius0; least_squares_problem(hessian=) for 'auto' chunks
  Body: Slices S6a + S6 (C.8, C.13, C.16 per DEFERRED S4 m6; T18). The exact model is the backend's on the same
  data (curvature actions bitwise); the frontend newton_cg run is the backend run bit for bit; T21 reproduces through
  the frontend (8 vs 28 outer); hessian(p) == FD of the Riemannian gradient along the retraction (x07). The frontend
  pytree partitions statics (a uniform model's mode sizes were traced). Drops the S5 not_yet_frontend exclusion.
