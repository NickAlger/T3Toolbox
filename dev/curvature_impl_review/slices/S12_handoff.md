# S12 handoff (implementer) -- DONE; base f5c0e1b (main now 4c908d8: dev/-only, no overlap); UNCOMMITTED. Scratch: impl/s12/
## Files
- backend/sampling_derivatives.py: section "Corewise two-hole: apply / entries derivatives" -- t3_{apply,entries}_corewise_
  derivatives_hessian[_from_sweep]; public compute_sigma_hat_tangent_jets (scan body _sigma_hat_tangent_jets_step = product
  rule of _sigma_hat_jets_step); private dotted two-row dxi^ (_dxi_hat_tangent_order_step / _core / step_{uniform,ragged});
  _apply_tangent_assemble_step (dG factored: banded pair + dotted twin, 2 finals; dU = dxi^dot (x) w); shared tail
  _apply_derivatives_hessian_from_jets (xmap over cores on both layouts, as the GN transpose). REFACTORS, bit-identical:
  _adj_prop_tangent extracted from _adj_tilde_tangent_step (its prop half, now shared); _entries_ambient_jets extracted from
  the GN entries transpose (shared). +5 __all__. 0 new contraction strings (all 18 used are already HISTORICAL rows).
- backend/fitting.py: ApplyDerivativesKind / EntriesDerivativesKind.corewise_hessian (omega^2 r; the model's lean jet sweep);
  backend/ut3_sampling.py: ut3_{apply,entries}_corewise_derivatives_hessian[_from_sweep] (S10 five-step wrapper; +1 doctest).
- backend/uniform_fitting.py: both uniform overrides via _frame_theta (FRAME masks); stub + _UNIFORM_HESSIAN_SLICE DELETED;
  _UniformKind -> _UniformLayerMixin (fields, from_point, _point, _variations, _remask kept); `import abc` dropped.
- fitting.py / optimizers.py / backend/optimizers.py docstrings: "every sampling kind"; CHANGELOG [2026.3.0]: the plain-kinds
  bullet -> "The exact Hessian for every sampling kind" (+ the new API), 3 flips (for now / interim stub / keep their stub).
- tests: _curvature_oracles +4 dense-trs twins (sigma^dot, dxi^dot all rows, 3-term assembly, states); test_curvature
  TestDerivativeKindsTwoHole (10), TestDerivativeKindsExactLocalModel (6, incl. one ragged-SHARED subcase per kind), T23:
  test_the_uniform_window_stub -> test_no_uniform_stub_remains (+ all six ragged kinds implemented, own class); uniform:
  TestUniformDerivativeKinds{TwoHole (2), CorewiseHessian (3), ExactLocalModel (subclass of the plain class via _KINDS, with
  its shared-uniform subcase)}; S8/S10 override tests rewritten w/o the stub; test_fitting +2 (frontend == backend, 5 setups;
  every-kind-builds replaces the S12-pending test); test_optimizers_frontend +2; test_dispatch +1; the chunk resolver test.
## Numbers (ragged; d = 1, 3, 4 + Tucker/TT slack (+ raw unsquashed for block-sum), K=(2,3), C=(),(2,); impl/s12/measure.py)
apply_derivatives: == trs twins 4.3e-16 (orders 0-4; bar 1e-14) | block-sum 7.0e-16 | exact Richardson FD 1.5e-14 | order 0
vs t3_apply_corewise_hessian 3.9e-16 (NOT bitwise) | == probe_derivatives two-hole at z~_i^(t)=a_i(c^(t)w_i+(t+1)c^(t+1)p_i),
sum a=1: 3.0e-16. entries_derivatives: twins 5.8e-16 | block-sum 5.7e-16 | FD 6.9e-15 | order 0 3.1e-16 | == apply_derivatives at
the one-hot vectors BITWISE. T6 Gram asym < 1e-12 rel; T11 per-order omega: two-hole(omega^2 r) == FD(grad) - GN < 1e-6, the
unweighted one off > 1e-2; a per-mode omega raises ("no mode axis"). T7 seeds: no redraw needed (sigma_min 4e-3..1.9e-1).
F11 c13 breaks on sd.binomial_combine_tensor (-> tests/_jet_trs_twins.py, S3b) + dropped trs args; shimmed (impl/s12/
run_c13_shimmed.py) <= 3.6e-16 (F11 2.5e-16); c9 runs (1.6e-10). Cost (numpy): exact/GN hvp 3.0-3.1x (d5 r8 o2 W300), 3.5-3.9x
(d4 r16 o3 W2000); two-hole/GN T 3.7-4.8x. Memory (jax uniform f32, unchunked): two-hole/GN T 1.85-1.98x (probe_deriv 1.85-2.2x).
## F22/F28/F29 re-measured through the LIBRARY (impl/s12/rerate.py; .out). NB those rows were PLAIN `apply` (c20/c30/c31/c42)
- T25 (F28/F38), apply: Tucker slack 20 outer / 4 rej / 78 CG, 2.6e-9; TT slack 34 / 9 / 272, 2.2e-12 -- F28 EXACTLY; slack ==
  minimal start (F38). apply_derivatives (o1): Tucker 17/4/56 (2.3e-9), TT 31/10/240 (2.0e-9); slack == minimal; frames as F28.
- T16 (F22/F29), apply, 15 GN warm steps, n15 window: GN ratios 0.50-0.80 (F22 0.56-0.68); exact p=1.0 e1/e0^2 0.79 (s1),
  0.94/0.86 (s2) (F22 <= 0.92); p=0.5 e1/e0^1.5 0.79-0.92 (F29 0.43-0.98); |1-rho| <= 2e-4. Target (3,3,3): at the floor already.
- apply_derivatives (o1): seed 1 GN 0.36-0.44; exact p=1.0 0.41 (15 GN) / 0.81, 0.48 (6 GN); p=0.5 0.40-0.99. Seed 2 is
  near-degenerate for GN (ratios 0.89-1.01): exact converges (16-17 rows, 'gradient'/'objective_floor') but its window
  constants are large (e1/e0^2 up to 47 / 280, |1-rho| to 0.5) -- verified NOT a bug: the exact form matches the retraction FD
  to 1.4e-6 / 2.3e-6 of the curvature term at x15 and at the end (impl/s12/seed2.py).
## Uniform (T17, worst): two-hole vs ragged 5.4e-16, kind 4.4e-16, exact hvp/curvature_hvp 1.4e-14, gradient 3.9e-15 (bar 1e-13);
exact masks; finite garbage bitwise (theta, theta_hat, frame incl. corewise boundary bonds, vectors beyond N_i); TT-slack frame
masks (point masks differ); shared-uniform subcase per kind; frontend == backend.
## Bit-identity vs f5c0e1b (git archive, separate processes, impl/s12/bitident.py): 772 arrays, 0 differ (GN corewise/tangent
transposes of both kinds, kinds' transposes, GN grad/hvp/newton_cg 6 kinds x 4 geometries, the other 4 kinds' exact hvp/forms/
newton_cg x 4 geometries, the probe-derivative two-hole orders 0-3 + uniform chunked). Mutants (impl/s12/mutate.py): 16/16 RED.
## Conflicts / judgment calls
1. chunk_size='auto' for these kinds: None in BOTH modes, estimator never reached (spy-tested) -- the GN transposes have no
   chunked path and the two-hole is ~1.9x their unchunked peak (above); the orchestrator's "reaches the estimator" presumes a
   chunked program that does not exist (S10/S11 ruling 4, same reasoning). Not built.
2. Twins in tests/backend/_curvature_oracles.py (beside S3's two-hole twins, as plan F says), built on tests/_jet_trs_twins.py.
3. F22/F28/F29 were plain-apply rows; re-measured on BOTH plain apply (their own kind) and apply_derivatives.
4. Mixin renamed (_UniformLayerMixin: dead-code grep empty); _from_sweep keeps an unused `order` (parity with the probe two-hole
   and the GN tv_*_transpose_derivatives_from_sweep). 5. Public pieces: only compute_sigma_hat_tangent_jets (GN twin public).
6. Uniform quadratic-form bar in the S10/S11 helper _per_element (now shared): 1e-12 rel-|q| -> 1e-13 of |p||Hp| (the floor
   leg tripped the old bar at 1.7e-12, an entries-derivative form; |q| cancels to ~1e-3 |p||Hp|; observed 5e-15 np 2.x /
   1.1e-14 np 1.26 -- the plain kinds reach 8.9e-13 rel-|q| on 1.26 too).
## Gates
targeted 11 files -n 4: 367 passed, 2 skipped / 5995 sub; test_dispatch 55 passed / 45 sub; module doctests 225 (S11: 222);
doc pages + getting_started OK; sphinx -W rc 0; floor leg (py3.11/np1.26/jax0.4.30) 222 passed, 2 skipped / 1872 sub +
chunking.md OK (run 1 failed the form bar above; fixed, rerun green); full -n 2: dispatch+backend 411 passed, 2 skipped /
3252 sub; rest 755 passed, 1 skipped / 43094 sub (1 warning, as at base); slow tier SERIAL 64 passed / 13173 sub (3 old
warnings). diff --check clean; no dev/ or CLAUDE.md edits; no new import edge; use_jit exact newton_cg runs (both kinds, both layers).
## Draft commit message
backend+frontend: the exact Hessian for apply_derivatives and entries_derivatives (S12) -- every sampling kind now runs
hessian='exact'; sampling_derivatives.t3_{apply,entries}_corewise_derivatives_hessian[_from_sweep] (+ compute_sigma_hat_tangent_jets),
the ut3_* masked twins; the derivative kinds + Uniform* twins implement corewise_hessian; the _UniformKind stub + table deleted
  Body: Slice S12 (C.5, C.6; T5, T6, T11, T17, T23). The adjoint-state transpose's product rule in the GN kernels' forms
  (sigma^ re-swept through G, its derivative a two-term reverse recurrence -- the dotted adjoint step's prop half, now shared;
  two-row dxi^ order scan; factored assembly). == trs twins <= 5.8e-16, block-sum <= 7e-16; order 0 == plain; entries == apply
  at one-hots bitwise; apply == probe_derivatives at the paired jets. Uniform == ragged per element. Rest bit-identical to f5c0e1b.
