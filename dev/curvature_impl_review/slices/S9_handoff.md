# S9 handoff (implementer) -- DONE, gates green; base main @ 2c42d4c; tree UNCOMMITTED (12 files). Scratch: impl/s9/
## Files
- backend/ufv_operations.py: NEW public `ufv_tied_lift_factors(frame_data, curvature_damping, shared_data, rcond)`
  (+doctest) = ufv_lift_factors + per-group `_tied_filter` at the group's REAL-dim clip `eps*max(sum nD_i, n_g)`,
  times the real-level host mask (`_real_group_levels`: first min(sum nD_i, nU) levels). New edge ufv_operations -> sharing.
- backend/utv_operations.py: `_check_uniform_lift_factors` returns the tied groups (`tv_operations._tied_lift_groups`)
  + structural companion-layout check (row_splits must be padded-nD blocks); `utv_corewise_lift` writes
  `sharing._tied_filtered_solve` into every group mode's slice (`_tie_group_slices`, xnp.stack); transpose sums the
  group slices (`_group_sum`, ragged order) -> `_tied_filtered_solve_transpose`; outputs re-masked as before.
- backend/geometry.py: S9 NIEs + `_uniform_shared_exact_error` gone (both uniform geoms); d=1 shortcut now precedes
  the shared branch (S8 nit); `_uniform_manifold_curvature_data(..., shared_data)`; `_uniform_lift_sigma_min` reads
  each group's REAL s_g levels in place of its modes' own spectra. Shared uniform COREWISE: verified live (no code).
- backend/sharing.py: `ufv_share_tucker_variations` RE-MASKS its output (see call 2); docstrings (ufv_shared_frame_data,
  fv_tied_lift_factors twin pointer); fv_operations.LiftFactors doc. docs/sharing.md "ragged" -> "ragged and uniform";
  docs/contributor/sharing_internals.md lesson 1 (the companion SVD's padding is roundoff; mask, real-dim clip,
  real-level sigma_min). CHANGELOG [2026.3.0]: Added bullet (exact on every geometry x representation for
  probe_derivatives; S8 bullet's "shared uniform raises" sentence removed), Fixed bullet (tied projection padding).
- tests: backend/test_uniform_curvature.py +18 (TestUniformSharedTiedLift 7, TestUniformSharedExactModel 8,
  TestUniformSharedExactNewton 3), `test_d1_and_shared_and_corewise` flipped, `_uniform/_ragged_exact_model` build
  the companion (None unshared: existing tests unchanged); test_fitting +`test_shared_uniform_matches_the_backend_
  exact_local_model`, :1185 NIE flipped; test_optimizers_frontend +uniform frontend T22 (W=40), options test runs the
  two shared uniform geoms; test_dispatch +`test_jit_shared_uniform_exact` (32 s). No new contract strings.
## Numbers (per element vs ragged shared, varying-rank stacks, padding forced, K in {(),(2,)}; tol 1e-13)
tied filters 5.8e-16 (padded levels exactly 0); lift 6.6e-16, transpose 7.9e-16; model hvp 1.6e-14, curvature_hvp
2.2e-14, gradient 1.1e-14 (manifold+corewise; outputs tied); sigma_min == ragged (rtol 1e-13), > 0 on every element.
Garbage (frame, theta_hat/p, sample, residual, companion padded U_M rows/level cols, s_g levels, Vt_M) -> BITWISE.
FD along the shared uniform retraction: |fd-q| 2.7e-8..3.2e-5 abs (<= 1e-5 rel), per-mode twin off 0.02..8.3;
O(h^2): 5.4e-5, 1.35e-5, 3.2e-6 (ratios 4.0, 4.25); dense Lagrangian of Phi o T 1.1e-14 (per-mode 0.68..3.4).
Padded restart (2,2,2)->(3,3,2): new real level f == 0 both layers, uniform == ragged filter; uniform-built frame:
real zero level 2.4e-31, padded 0.0 -> sigma_min reads the real level. Group clip: level at 1.8e-15 kept on both.
## End to end (W=60 unless noted)            outer / rejected / CG
T22 ragged 10/0/41; uniform backend 13/0/40 (same final objective, rel 0; tied). Frontend W=40: 11/0/44 (ragged 8/0/31).
Group-ceiling restart (7,7,4) (3,3,2)->(4,4,2): uniform 7/0/24 (ragged 7/0/28), per-mode lift 12/0/59 -> asserts <= 9.
Group-ceiling FULL-RANK start: uniform == ragged 9/0/28, losses row by row <= 1.7e-14 (per-mode 14/0/63).
Times: T22 2.9 s, ceiling restart 2.6 s, full-rank 1.4 s, frontend T22 2.95 s (all always-on, <= 3 s).
## Bit-identity vs 2c42d4c (git archive, separate process; impl/s9/bitident.py): shared uniform GN both geoms (grad,
hvp, proj, retract, newton_cg, GD, frontend runs, apply newton_cg), unshared uniform exact, ragged shared exact
(manifold+corewise), 2 seeds: 240 arrays, 10 differ -- ONLY the tied projection's padded slots (<= 1.7e-13 abs,
~1e-16 rel) now exactly 0; every real value and every run bitwise identical.
## Mutants (impl/s9/mutants.py, tree copies): 17/17 RED -- padded s_g not zeroed, clip at padded dims, sigma_min reads
padded / joins group modes, per-mode lift, T^T first copy / mean / untied, lift untied, undamped filter, tangential
with self.masks (m29, TT-slack shared test), projection not re-masked, no layout check, curvature ignores aux,
unmasked V / Y into the tied solve, lift output not re-masked.
## Gates
targeted 9 files -n 2: 334 passed, 2 skipped / 4306 sub; module doctests 218; pages + getting_started OK; sphinx -W 0;
floor leg (jax0430): 215 passed, 2 skipped / 599 sub, chunking.md OK; full -n 2: dispatch+backend 363 passed,
2 skipped / 2298 sub (S8 344/2212); rest 748 passed, 1 skipped / 42845 sub (1 warning = pre-existing adam); slow
tier SERIAL: see reply.
## Conflicts / judgment calls
1. `ufv_tied_lift_factors` lives in ufv_operations, not sharing (sharing -> ufv_operations would cycle via
   ufv_conversions -> ranks -> sharing). Plan §M: add ufv_operations -> sharing. Private cross-module calls
   `sharing._tied_filter/_tied_filtered_solve[_transpose]`, `ufv_operations._real_group_levels` (S7 precedent).
2. Pre-existing hygiene gap fixed: the uniform tied PROJECTION left SVD roundoff in padded nD rows (its docstring
   claimed masked-clean); the T17 exact-output-mask rule fails through any shared model without it. Re-mask only.
3. NOT changed (GN bit-identity): the uniform tied projection still clips at the PADDED dims and does not mask padded
   levels -> a level between the two clips (~1e-15 rel) is kept by the lift/ragged, dropped by that projection; and
   it is not robust to garbage in a companion's padded levels (only a hand-built companion). DEFERRED candidate.
4. T22 uniform row-by-row == ragged impossible by contract (rank-deficient restart; S8 T21 ruling): asserted its own
   properties (gradient stop, <= 2 rej, sigma0 < 1e-12, <= 16 outer, same minimizer 1e-9, tied); row-by-row asserted
   from the full-rank group-ceiling start instead. T22 is non-discriminating for the lift (S7); the ceiling restart is.
5. Uniform frontend T22 at W=40 with chunk_size=None ('auto' costs ~5 s eager; not under test).
## Draft commit message
backend+frontend: the exact Hessian on the shared uniform geometries (S9) -- ufv_operations.ufv_tied_lift_factors
(each group's damped filter zeroed on the uniform companion's padded s_g levels by the masks, clipped at the real
dims); utv_corewise_lift[_transpose] tie the group slices; the uniform geometries' shared exact mode; real-level
s_g in lift_sigma_min; the uniform tied projection re-masks its output
  Body: Slice S9 (C.10/C.11; T12 uniform twin under T17, T22). Per element == ragged shared <= 2.2e-14, exact masks,
  garbage-invariant incl. the companion's padding; FD O(h^2) along the shared uniform retraction, dense Lagrangian
  1.1e-14; full-rank shared start == ragged run; group-ceiling restart 7 outer vs per-mode 12. Frontend == backend;
  one jit key, CG compiles once. GN bit-identical to 2c42d4c except padded roundoff now 0. probe_derivatives now
  runs exact on every geometry x representation.
