# S8 handoff (implementer) -- DONE, all gates green; base main @ fdcff71; tree UNCOMMITTED (16 modified + 1 new). Scratch: impl/s8/
## Files
- backend/ufv_operations.py: `ufv_lift_factors` (mask frame once; polymorphic zipper + 'dC..' strings; ONE batched damped
  pinv per family; rcond per slot/element at the REAL dims `_real_rcond`; pinvs re-masked; padded svals zeroed by host
  masks `_real_values`; bond_pinvs has d slots, slot d-1 = boundary pass-through E00) + doctest. fv_operations: LiftFactors doc.
- backend/utv_operations.py: `utv_corewise_lift[_transpose]` (entry mask = own masks; out = frame (up,left) / gauge masks
  over K, re-masked) + doctest; `_check_uniform_lift_factors` (structural; tied factors -> NIE).
- backend/t3_linalg.py: `t3_precompute_inner_product_sweep` / `..._hessian_from_sweep` polymorphic (supercores ->
  d-prefixed strings + stacked zippers); ragged branch unchanged (bitwise, below).
- backend/ut3_linalg.py: `ut3_inner_product_corewise_{transpose, hessian[_from_sweep]}`, `ut3_precompute_inner_product_sweep`
  (mask z, theta; theta_hat with THETA's masks; re-mask) + doctest. New edge ut3_linalg -> t3_linalg.
- backend/ut3_sampling.py: `ut3_probe_corewise_derivatives_hessian[_from_sweep]` (M4) + doctest (incl. negative control).
- backend/uniform_fitting.py: `UniformProbeDerivativesKind.corewise_hessian` (masks = frame_data[5][0],[2]); its
  `_UNIFORM_HESSIAN_SLICE` row removed (stub + table stay for S10-S12); `_UniformKind._variations` K-stack (S0c m1).
- backend/geometry.py: uniform `_variations` K-broadcast (`_var_masks_for`), `inner` keeps K+C; four exact members on both
  uniform geometries (`_uniform_point`, `_uniform_lift_sigma_min` real slots, `_uniform_manifold_curvature_data`); shared
  uniform (manifold AND corewise) -> NotImplementedError naming S9, raised in curvature_precompute. ufv_masking: `_masks_over_stack`.
- fitting.py: `_tangent_at` broadcasts masks over K; docstrings flipped. CHANGELOG [2026.3.0]: Added (uniform exact),
  Fixed (uniform tangent stacks), S4/S6 "uniform raises" phrases flipped.
- tests: NEW tests/backend/test_uniform_curvature.py (22); test_dispatch +3; test_curvature T23 window test (5 window kinds,
  UPDK out); test_fitting +uniform frontend==backend & m27, flips (shared uniform -> S9 NIE); test_optimizers_frontend
  flip; test_contractions_interpreter +17 d-prefixed HISTORICAL rows ('' each; sharding sweep green, 778 sub).
## Per element vs ragged (worst over d=1,3,4, varying-rank C=(2,), forced padding, K=(2,), TT slack; tol 1e-13)
lift factors 6.0e-16; lift 6.3e-16 / transpose 8.0e-16; T3-T3 one-hole 3.1e-16, two-hole 5.9e-16, tangential 5.2e-16;
probe two-hole 5.9e-16 (dense + chunked); kind 3.9e-16; model gradient 5.5e-16, hvp 3.4e-15, curvature_hvp 3.5e-15;
GN hvp on K 9.4e-16. Exact output masks from the ragged shapes; garbage in theta/theta_hat/z/frame (corewise boundary
slots incl.)/residual/ww/pp -> BITWISE unchanged (layout preserved). Problem path (own gauge-equiv frame): dense hvp <1e-12.
## T21 / T25 uniform vs ragged
T21 (restart W=60): ragged 8 outer / 0 rej / 27 CG; uniform backend 8/0/27 (per-step CG 1,1,1,2,3,4,6,9 vs 1,1,2,2,3,4,6,8),
same final objective (rel 0); frontend == backend bitwise. T25 tucker slack 14/1/26, tt slack 22/6/70: uniform backend
AND frontend identical counts, losses row by row <= 1.9e-13.
## Compile (fresh process, CPU x64, first minus cached run): exact ~14.5-14.8 s vs GN ~9.2-9.7 s ((4,5,6) W7 chunk 2;
(6,6,6,6) W200 o2 chunk 50) -- inside the 11-43 s precedent; dispatch test guards < 300 s. Dispatch tests 22/18/13 s.
## Bit-identity vs fdcff71 (git archive, separate process): uniform GN (both geoms: grad/hvp/proj/fwd/transpose/newton_cg/
GD/frontend) + ragged GN + uniform apply: 184 arrays, 0 differ; ragged EXACT (manifold/corewise/shared manifold hvp,
curvature_quadratic, newton_cg runs): 108 arrays, 0 differ.
## Mutants (impl/s8/mutate.py): 12/16 RED. 4 survivors EQUIVALENT by construction: M5/M6 lift entry masks (factors already
masked + output re-masked), M8 pinv re-mask (padded singular vectors exactly 0), M16 transpose re-mask (output already
clean) -- kept as defence in depth (plan C.10). M7 (padded rcond) needed a new test -> RED.
## Gates (final): targeted runs green; test_dispatch in half 1; module doctests 217 passed; doc pages + getting_started OK;
sphinx -W exit 0 (clean); floor leg (py3.11/np1.26/jax0.4.30): estimator+dispatch+uniform files 217 passed / 678 sub,
test_uniform_curvature 22 passed (final), docs/chunking.md doctest OK; full suite -n 2: dispatch+backend 343 passed /
2210 sub (S7: 314/2042); rest 746 passed, 1 skipped / 42839 sub (1 warning = pre-existing adam-on-manifold); slow tier
SERIAL 5 passed / 12,790 sub. `git diff --check` clean.
## Conflicts / judgment calls
1. T21 uniform row-by-row == ragged is IMPOSSIBLE by the equivalence contract ("full-rank qualifier": at a rank-deficient
   start the layers must pick different completions; row-0 ||g|| differs 7.6e-2). Asserted T21's own properties + same
   minimizer + frontend==backend; row-by-row + counts asserted from full-rank starts (T25, backend AND frontend).
2. Shared uniform corewise blocked too (NIE naming S9) per the scope guard, though it would likely just work.
3. K-stack: beyond S0c m1 the uniform geometries' project/inner and the frontend `_tangent_at` also crashed on K+C; fixed.
   Geometry `inner` now keeps K+C (kept |C| leading axes = K's on a stacked pair). One CHANGELOG Fixed bullet.
4. LiftFactors reused (supercore-valued, bond_pinvs d slots). Real-dim rcond: a padded clip drops directions ragged keeps.
5. ut3_*_from_sweep take the UNIFORM sweep tuple (packed ww/pp ride in it), like utv_*_from_sweep.
6. No uniform `_trs` twins written (I-33's "written once in S8"): the polymorphic kernel is compared per element against the
   ragged production kernel, itself pinned to the ragged twins -- not needed.
7. New test file (test_curvature.py is 2.2k lines). T21 uniform 3.6 s / TT-slack 4.5 s > 3 s budget (hvp-bound; S6 n2
   precedent); slow-tier candidates if the budget tightens.
## Draft commit message
backend+frontend: the exact Hessian on the uniform layer (S8) -- ufv_lift_factors, utv_corewise_lift[_transpose],
ut3_inner_product_corewise_*, ut3_probe_corewise_derivatives_hessian[_from_sweep]; UniformProbeDerivativesKind.corewise_hessian
replaces its stub; the uniform geometries' exact members; uniform tangent stacks reach the local model
  Body: Slice S8 (C.10; T17, T24, T20, T21/T25 uniform, T23). Theta and theta_hat masked with the FRAME's masks, every
  output re-masked; per element == ragged <= 3.5e-15 on varying-rank stacks with forced padding, garbage-padded inputs
  bitwise invariant. T25 slack starts: uniform backend and frontend == ragged (counts, losses <= 1.9e-13); T21 restart
  converges (8/0/27, same minimizer; frames differ there by contract). Jit: one cache key across points and frames;
  exact CG compiles once (~15 s vs GN ~9.5 s). GN uniform and ragged exact bit-identical to fdcff71. Shared uniform
  exact raises NotImplementedError until S9. Fixes the K-stacked uniform forward/project/inner (S0c m1).
