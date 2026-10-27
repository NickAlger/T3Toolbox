# S7 handoff (implementer) -- DONE, all gates green; base main @ 3158778; tree UNCOMMITTED (16 files). Scratch (session-local, not archived): impl/s7/
## Files
- backend/sharing.py: `_tied_solve` -> `_tied_filter(sd, gi, eps, rcond)` (the solve's rcond default, `filtered_reciprocal`)
  + `_tied_filtered_solve(sd, gi, V, s_filter)` (einsums unchanged: bit-identity) + `_tied_filtered_solve_transpose`
  (grouped `contract`, split by row_splits); public `fv_tied_lift_factors(frame, curvature_damping, shared_data, rcond)`
  (+ doctest). Edges sharing -> linalg, contractions (plan M lists linalg).
- backend/fv_operations.py: `LiftFactors` += `shared_data=None`, `tied_filters=()` (pytree children; partition = the
  companion's own aux). tv_operations.py: `tv_corewise_lift[_transpose]` tie the group slots when the factors carry a
  companion (T copies / T^T sums), `_tied_lift_groups` structural check. geometry.py: S7 NotImplementedError GONE;
  `curvature_precompute` builds tied factors on `aux` (fallback `precompute`), `_lift_sigma_min`. linalg/fitting/
  optimizers: docstrings. docs/sharing.md (exact-mode paragraph, prose), sharing_internals.md ("Padded restarts"),
  CHANGELOG (Added bullet; S4/S6 bullets no longer say the shared manifold raises).
- tests: test_curvature +TestSharedExactLift (3) / TestSharedExactLocalModel (7) / TestSharedExactNewtonRestart (T22),
  S1 tied-solve test re-frozen on the pre-S7 body, damping test flipped (tied factors bitwise); _curvature_oracles
  +`tied_lift[_transpose]` (dense numpy-pinv tied lift); test_fitting +`test_shared_manifold_matches_the_backend`
  (S7 gate), :1031 / :1133 flipped; test_optimizers_frontend :654 flipped + frontend T22; test_dispatch
  +`test_jit_shared_exact_local_model`; contractions HISTORICAL +3 rows ('' each).
## Numbers
- F14: c15 runs unchanged at base and here. Through the REAL `Problem.local_model` (impl/s7/f14_real.py): lift 4.7e-16,
  GN 1058.03248707, exact 1058.19431605 (curvature 0.16182898), cd 0 == default; FD err 1.12e-4, 2.80e-5, 6.89e-6,
  1.98e-6 (h 2e-2..2.5e-3; O(h^2)). Test asserts the ratios in (3, 5) (values LAPACK-sign dependent; S4 c4 precedent).
- T12: tied lift identity <= 1e-12 (obs. ~5e-16), group arrays identical; adjoint 1e-12 at eps 0..10; retraction FD
  (obs. |exact-FD| 6.5e-7..1.4e-5); FD of P_x grad_sh f (exact for ANY retraction; obs. 5e-8..4e-7 rel); dense
  Lagrangian of Phi o T via an independent dense tied lift < 1e-10; per-mode twin off by 5.6..19 (quadratic),
  7e-2..2.7e-1 (hvp) on every discriminating structure. Padded restart: s_g zero level -> f exactly 0 (both eps),
  lift ⊥ null(M_g) (1e-14), transpose blind to it, TT/singleton lift bitwise unshared, sigma_min = min(s_g) ~0.
- T22 backend |W|=60: exact 10 outer / 0 rej / 41 CG, 'gradient', row0 lift_sigma_min 0.0; GN 25 'max_newton'
  (5.5e-4; 40 steps: 7.6e-6, 131 CG). F19 (unshared): 8-10 / 23-28. Serial 2.5 s. Frontend |W|=40: 8 / 0 / 31,
  == backend run bitwise, 1.9 s.
- Bit-identity vs 3158778 (git archive, separate process, impl/s7/bitident.py; GN shared ragged+uniform+corewise,
  shared probe_derivatives GN + restart, shared-corewise exact, projection / ambient dirs / retraction / residual at a
  padded restart, UNSHARED exact model + run; 2 seeds): 2074 arrays, 0 differ.
- Mutants (impl/s7/mutants.py, mutant_n.py) 7/7 RED: `_blm` w/o geometry_aux (DEFERRED gate), per-mode lift, untied N,
  sigma_min reading group modes, transpose scaling, undamped tied filter, project w/o tie.
## Gates (final)
targeted 7 files 302 passed / 3051 sub; module doctests 211 (S6 210); pages + quickstart OK; sphinx -W exit 0;
full -n 2: dispatch+backend 314 / 2042 sub (S6 302 / 1963); rest 730 passed, 1 skipped / 42509 sub (S6 728 / 42465;
1 warning = the oracle sweep's pre-existing adam-on-manifold); slow tier SERIAL 5 passed / 12,790 sub (= S6).
After two cosmetic edits (direct SharedFrameData annotations; "common frame" wording): sharing + curvature tests
144 / 801 sub, their doctests, sphinx -W exit 0 re-run.
## Conflicts / judgment calls
1. Tied factors live IN `LiftFactors` (plan C.2 "lift factors gain the per-group filtered tied factors"): a companion
   REFERENCE + per-group filters, built by `sharing.fv_tied_lift_factors` (fv_operations may not import sharing, §M).
   The companion's arrays thus appear twice among a model's jax leaves (geom_aux, curvature) -- harmless.
2. `_tied_solve` removed, not wrapped; its rcond default moved to `_tied_filter`. `_tied_filtered_solve` infers its
   dispatch (was handed xnp): differs only for numpy variations + jax companion (now jax out).
3. Forward tied solve keeps '...' einsums (bit-identity); the NEW transpose uses grouped contract (plan's new-kernel rule).
4. sigma_min: group modes' own spectra are REPLACED by s_g, not merely joined ("includes min(s_g)", n12): the tied
   lift never inverts a group mode's S_i^T, which can be tiny on a regular shared point (test: >= 10x smaller).
5. Finding: the per-mode lift == the tied lift wherever every group mode's S_i^T is square (F14's own structure!), and
   it satisfies DPhi[Lv] = Ev even where it differs (min-norm); only the curvature tells. T12 therefore runs
   group-ceiling structures (group rank > a member's ceiling; and n_g < N -- at n_g = N there is no Tucker variation).
6. Extra oracle beyond plan: FD of the projected shared gradient (needs no 2nd-order retraction).
7. `_blm` mutant is value-invisible (a recomputed companion is bitwise equal): killed by `assertIs(_blm.geom_aux,
   geometry_aux)` + a 0-rebuild spy. Frontend `evaluate` takes unstacked p only (pre-existing) -- gate calls it so.
8. Dispatch test 20 s (precedent 11-43 s); its eager twin runs on numpy (eager jax two-hole: 11 s).
9. docs/sharing.md paragraph is prose (no code; S13 owns examples). Private cross-module calls tv_operations ->
   sharing._tied_filtered_solve[_transpose] (precedent t3_linalg._inner_product_frame_transpose).
## Draft commit message
backend+frontend: the exact Hessian on the shared manifold (S7) -- the tied lift (sharing.fv_tied_lift_factors;
LiftFactors carries the companion and one damped filter of s_g per group; tv_corewise_lift[_transpose] lift each
group to one factor direction and sum its copies back); the tied projection's solve refactored into
_tied_filter / _tied_filtered_solve[_transpose] (bit-identical); lift_sigma_min reads s_g
  Body: Slice S7 (C.11, n12/I-13; T12, T22). Replaces the shared-manifold NotImplementedError. Reproduces F14
  through the real model (exact 1058.19431605 vs GN 1058.03248707, FD O(h^2)); exact == the dense Lagrangian of
  Phi o T, == FD of the projected shared gradient; padded restarts gate the zero s_g level; T22 10 outer / 0
  rejected vs GN unconverged at 25. GN shared runs bit-identical to 3158778. Frontend shared exact == backend
  (the DEFERRED S6 gate); dispatch: one compile across points and dampings.
