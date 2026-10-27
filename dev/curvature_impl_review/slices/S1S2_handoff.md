# S1 + S2 handoff (implementer) -- DONE, all green, tree UNCOMMITTED (base main @ 0ec8d45)
Split: scratchpad/impl/S1.patch (base->S1), S2.patch (S1->final); scratchpad/impl/split/ = scratch git repo
(base/S1/S2; final tree == repo's; `git apply -R --check S2.patch` passes on the repo). S1 alone verified green.

## Files per slice
S1: CHANGELOG.md (1 Added bullet); backend/linalg.py (filtered_reciprocal, damped_pseudo_inverse); backend/fv_operations.py
(fv_center_factors, LiftFactors + `if jax_available:` pytree, fv_lift_factors); backend/sharing.py (fv_shared_frame_data
calls fv_center_factors); backend/tv_operations.py (tv_corewise_lift[_transpose], _check_lift_lengths);
tests/backend/test_curvature.py (T1-T4), tests/backend/_curvature_oracles.py (whole helper: block-sum, dense
gradient/Hessian/Jacobian/Lagrangian, vertical directions); tests/test_contractions_interpreter.py (AXIS_SIZE.update
+ 6 rows); tests/test_contractions_sharding.py (SINGLE += l,n,x,z); tests/test_dispatch.py (test_jit_curvature_lift).
S2: CHANGELOG.md (1 bullet); backend/t3_linalg.py (moved kernel as _inner_product_{rebasis,variations}_step +
_inner_product_frame_transpose; t3_inner_product_corewise_transpose, t3_precompute_inner_product_sweep,
t3_inner_product_corewise_hessian[_from_sweep]); backend/tt_operations.py (_tt_zipper_tangent_step,
tt_zipper_tangent_{left_to_right,right_to_left}); backend/tv_operations.py (projection delegates, old steps removed);
test_curvature.py (+TestTangentZipper, T5/T6 TestInnerProductTwoHole, T20 TestFlatSlotsAndVerticalDirections);
interpreter (+14 rows); test_dispatch.py (test_jit_inner_product_two_hole).

## Reproduced numbers (through the new functions; scratch impl/repro_s1.py, repro_s2.py)
c1: |H-H0|<=4.4e-13, |T-<L,H>|<=6.5e-13, |S^T-VU^T|<=2.9e-12; all spectra match; sum s(S0)^2/|X|^2-1<=5.6e-16.
c2 lift: 6.04e-15/6.22e-15 and 1.22e-15/1.27e-15 [c2 6.04/6.21, 1.20/1.27]; one-hole -393.9702957658 vs FD
-393.9702920206 and -34939.8090286372 vs -34939.8090209888 (identical). c12: 6.8e-16/1.0e-15 [6.3e-16/9.2e-16].
c7 is UNSEEDED -> its verbatim lift vs tv_corewise_lift in-process (slack/padded/TT-slack, cd 0/1e-3/1e-1): <=4.9e-17;
zero point identical (1.03e+00 last slot, 0 elsewhere). c6 T3-T3 (seeded): vs prototype <=3.1e-16, vs block-sum
<=4.7e-16. c40 on final tree: 90/90 strings, 143/143 HISTORICAL rows. c34 DAG: ACYCLIC. array_equal bit-identity:
fv_/ufv_shared_frame_data (6 cases); tv_project_t3_onto_tangent_space + MANIFOLD/shared project_ambient (15 cases).

## Tests
test_curvature.py 22 tests/270 subtests (S1 13/199). Mutation: dropping any of 3 two-hole terms or a zipper term -> red.
Final: suite halves 344+609 passed, 1 skipped, 43,779 subtests; sharding 1/716 standalone; slow O2 5/12,790; module
doctests 206; all docs pages. S1 tree: halves 333+609 passed, sharding 1/666, module doctests 203.
NOT run: sphinx -W (sphinx not installed here).

## Conflicts / judgment calls
- Plan gap: tests/test_contractions_sharding.py has its own SINGLE size table; extended with l/n/x/z (else KeyError).
- env/envd/Tucker-block strings are mine (plan §D: kernels record their own): 'Cxa,Cxzc,Ccb->Cazb' + dotted twins,
  'KCazb,Caub->KCzu', 'Cazb,KCaub->KCzu', 'KCzu,Czn->KCun', sweep 'Czn,Cun->Czu', 'Cazb,Czu->Caub'; all '' supplement.
- Moved steps renamed (private). Committed checks c2/c4/c6/c10 + critic checks import the OLD tv._project_t3_* -> they
  AttributeError after S2 (all ran green before); no alias kept.
- damped_pseudo_inverse default rcond = eps*max(m,n) (pinv/_tied_solve); filtered_reciprocal's = eps*k per plan;
  filtered_reciprocal(s,None,rcond) == _tied_solve's filter bitwise (tested incl. a padded shared restart).
- ||X||_HS = ||S_0^T||_F (== sqrt(sum s(S0)^2); avoids an SVD before eps). Negative scalar damping -> ValueError.
- tv_corewise_lift[_transpose] keep the plan's unused `frame` (checked structurally: mode counts).
- tt_zipper_tangent_*: optional zippers= (None -> computed), polymorphic ragged/uniform like the parent.
- CHANGELOG bullets added (S0c-S0e precedent); drop them if the orchestrator prefers S13-only.

## Draft commit messages
S1: backend: the exact Hessian's lift (S1) -- linalg.filtered_reciprocal / damped_pseudo_inverse (clipped, Tikhonov-damped);
fv_operations.fv_center_factors (H_i, S_i^T, T_i by zipper; fv_shared_frame_data now calls it, bit-identical), LiftFactors
(pytree of leaves, eps a leaf) + fv_lift_factors(frame, curvature_damping) (eps relative to ||X||_HS);
tv_operations.tv_corewise_lift / tv_corewise_lift_transpose (exact at eps=0 on minimal, Tucker-slack and TT-slack frames)
  Body: Slice S1 of dev/riemannian_newton_curvature_impl_plan.md (C.1, C.2, C.3 lifts; I-1, I-22). T1-T4 in
  tests/backend/test_curvature.py + oracle helper; AXIS_SIZE (and the sharding test's table) += l,n,x,z, 6 HISTORICAL
  rows; dispatch: one compile across frames and dampings. Full suite 942 passed; doctests; pages.
S2: backend: the T3-T3 two-hole (S2) -- tv_project_t3_onto_tangent_space's ungauged kernel moves to
t3_linalg._inner_product_frame_transpose (bit-identical); t3_inner_product_corewise_transpose, t3_precompute_inner_product_sweep,
t3_inner_product_corewise_hessian[_from_sweep]; tt_operations.tt_zipper_tangent_left_to_right / _right_to_left
  Body: Slice S2 (C.3 T3-T3). T5 (block-sum, dense multilinear Hessian, FD), T6 symmetry, T20 backend half (flat
  structures cancel; vertical directions null); 14 HISTORICAL rows; dispatch incl. uniform scan + d=1. Full suite
  953 passed / 43,779 subtests; slow O2 5 / 12,790.
