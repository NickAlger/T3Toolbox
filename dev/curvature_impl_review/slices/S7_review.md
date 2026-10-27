# S7 review -- exact mode on the SHARED manifold (ragged): tied lift, adjoint, s_g, tied-solve refactor
Base 24ee56d; tree UNCOMMITTED (16 files + my 1-line fix). Scripts: review/s7/

## Verdict: APPROVE-WITH-FIXES (one doc fix applied; no blocking finding)
Fix applied: `CHANGELOG.md:249` dropped "where it raised `NotImplementedError` before" (that state never shipped; the
[2026.3.0] section is unreleased). Nothing else changed.

## Independent checks (all mine)
1. **Math, from first principles** (`check_math.py`): dense Jacobian of Phi∘T (T copies the group factor), P_sh = J J^+,
   N = (I-P_sh)A^T r, a min-norm lift J^+ xi (ANY lift of Phi∘T is valid: D2Phi_sh along ker J_sh is tangent), D2Phi by
   multilinearity. Production `hvp` bilinear form == dense Riemannian Hessian <= 2.3e-12 rel on ceiling / two-group /
   all-mode / square structures; tied misfit gradient == P_sh A^T r <= 1.9e-14; DPhi[Lv]=Ev <= 8.4e-16; adjoint <= 3.6e-15
   at eps 0/1e-2/1. Per-mode twin off by 5e-2 / 22 / 9 (ceiling / two_groups / one_group), == tied (2e-15) on square
   S_i^T. Why per-mode is wrong: it is tied + a kernel direction k of the UNSHARED Phi; D2Phi[a,k] lies in the unshared
   tangent space, which N (normal only to the SHARED tangent space) does not annihilate. Multi-axis K=(2,2),C=(2,3)
   (`stack_check.py`): stacked == per element bitwise, adjoint 2.1e-14/element. Oracle independence: `_curvature_oracles.
   tied_lift[_transpose]` use np.linalg.pinv on fv_center_factors' S^T (pinned in T1); no production lift / companion SVD.
2. **F14 through the real model** (`f14_rev.py` = c15 verbatim + Problem.local_model): exact 1058.19431605 (c15 - real
   -2.3e-13), GN 1058.03248707, cd 0 == default to 1e-11; FD errs 1.12e-4, 2.80e-5, 6.89e-6, 1.98e-6 (ratios 4.01 4.07
   3.48). Per-mode vs tied lift on F14's structure: 3e-18 (down ranks == up ranks: judgment call 5 confirmed).
3. **Bit-identity vs 24ee56d** (`bitident_rev.py`, `git archive` tree, separate processes, 2 seeds): tied projection /
   ambient dirs / gauge / retraction on 4 structures x {(),C,K,K+C}, padded restart, shared GN probe_derivatives (+
   restart), shared corewise GN AND exact (+hvp), frontend shared apply ragged AND uniform newton_cg, unshared exact
   model+run: **1144 arrays, 0 differ**. The only differences (312 compared, 160 differ) are the MIXED case (numpy
   variations + jax companion): `fv_tied_ambient_directions` now returns jax (float32 without x64), and
   `fv_share_tucker_variations` stays numpy but its solve now runs in jax f32 (1e-7). Judgment call 2's stated change.
4. **Mutants** (`mutants_rev.py`, tree copies, S7 subset of 43 tests): all 9 RED -- per-mode lift (31 sub), untied N_sh
   (20, incl. T22 back+front), T^T as group MEAN (58), T^T first copy only (58), undamped tied filter (1: the filter
   test), sigma_min JOINING group modes' own spectra (3), tied-but-wrong first-member lift (66), transpose without the
   filter (60), `_blm` without geometry_aux (3: the identity/spy asserts).
5. **Pytree/aux** (`pytree_check.py`): backend + frontend exact shared models: treedefs equal and hash-equal across points
   and dampings; 5 duplicated companion leaves (geom_aux + lift_factors.shared_data; identity lost on a jax round trip,
   values equal, round-trip hessian bitwise); no non-array leaves. `curvature_precompute(aux=None)` fallback == with aux
   bitwise (untested in the suite -- n2). Frontend `newton_cg(..., hessian='exact', use_jit=True)` on shared_manifold
   runs, tied output, same losses to 4 digits (`jit_run.py`). Import DAG (`imports_dag.py`, AST): backend acyclic; new
   edges sharing->linalg (planned, §M) and sharing->contractions (contractions imports only common); fv_operations
   imports no sharing (LiftFactors.shared_data is `typ.Any` + comment; no TYPE_CHECKING import needed).
6. **T22** (`t22_discrim.py`): the per-mode-lift mutant runs IDENTICALLY on T22's structure (10/0/41; square S_i^T) --
   T12 catches it, T22 cannot. A group-ceiling restart discriminates end to end: (7,7,4), (3,3,2)->(4,4,2), target
   (5,5,3)/(1,3,3,1): tied 7 outer / 28 CG vs per-mode 12 / 57; GN 25 unconverged.

## Findings
- n1 (nit, orchestrator): `fv_share_tucker_variations` (sharing.py:625-626) infers its redistribution's backend from the
  variations only, its solve now from all inputs: mixed-case output is numpy via a jax f32 solve. Harmless; align later.
- n2 (nit): the `curvature_precompute` aux=None fallback (geometry.py:358-359) has no suite test (verified bitwise here).
- n3 (doc, S13): `fv_tied_lift_factors` "Exactness" (sharing.py:753) says full column rank of M_g; sufficient, not
  necessary: at a padded restart the group part is still exact (8.8e-16, `padded_exact.py`; tied V lies in range(M_g)),
  while the singleton/TT parts are not (0.45, the pre-existing S1 non-full-rank caveat). Prose only.
- n4 (S13): the [2026.3.0] CHANGELOG narrates slices ("now runs there too", "LiftFactors gains"); consolidate at S13.

## Rulings on the 9 judgment calls
1 AGREE (companion+filters as LiftFactors leaves; x2 leaves is memory/transfer only, treedef stable, compile-once holds).
2 AGREE (`_tied_solve` private, only two callers + the re-frozen S1 test; inferred dispatch is the library convention;
  no caller/test relied on the mixed numpy output -- see n1). 3 AGREE (forward einsums kept for bit-identity; the three
  HISTORICAL rows are exactly sharing.py:697-699). 4 AGREE -- the tied lift inverts M_g (spectrum s_g) and never a group
  mode's S_i^T; a plan CLARIFICATION for S13, not a deviation. 5 AGREE (verified: per-mode == tied iff S_i^T square;
  T12 structures discriminate). 6 AGREE (FD of P_x grad_sh f is retraction-free and correct). 7 AGREE -- the mutant is
  value-invisible by construction (the curvature reads lift_factors.shared_data; project recomputes a bitwise-equal
  companion), so identity + 0-rebuild spy is the right (cost) kill. 8 AGREE (15.6 s here). 9 AGREE (sharing does not
  import tv_operations; precedent t3_linalg._inner_product_frame_transpose).
T12 "ratios in (3,5)": justified -- q(v) for a coordinate v depends on the frame's SVD signs; a wrong q plateaus the
error (ratio -> 1), so the window detects any q error above ~7e-6 (rel 7e-9). Budget: T22 2.79 s, frontend 2.21 s.
## Plan-text amendments for S13
(a) C.11 / C.7 / I-13: `lift_sigma_min` = min(bond spectra, untied modes' Tucker spectra, s_g IN PLACE OF the group
modes' own). (b) §M: add sharing -> contractions. (c) C.2: LiftFactors fields `shared_data`, `tied_filters`; builder
`sharing.fv_tied_lift_factors`. (d) §F T12: structures must be group-ceiling (n_g > some member's rL*rR, n_g < N);
add the retraction-free oracle. (e) §F T22: non-discriminating for the lift type by construction; optionally add the
group-ceiling restart above. (f) DEFERRED rows 24 and 36 resolved by S7.

## For S8 / S9 (uniform shared twin)
- The uniform companion holds padded slices (row_splits multiples of padded nD); padded s_g levels sit at roundoff, so
  zero them by the group rank mask, not only by the relative clip (and keep a real zero level at a padded restart).
- `_lift_sigma_min`'s twin must read REAL s_g levels only (else every varying-rank point reports 0).
- Uniform tied lift/transpose: one supercore slice per group mode (aliasing as here), masks re-applied on output.

## Gates (re-run by me; full suite and slow tier LAST, after my CHANGELOG fix)
Targeted 8 files -n 2: 305 passed / 3771 sub. test_dispatch: 49 passed / 36 sub (new test 15.6 s). Module doctests
211 passed; docs pages + getting_started.rst OK; sphinx -W exit 0. Full: dispatch+backend 314 / 2042 sub; rest 730
passed, 1 skipped / 42509 sub (1 warning = the pre-existing adam-on-manifold). Slow tier SERIAL: 5 passed / 12790 sub.
`git diff --check` clean; no dev/ or CLAUDE.md edits; long lines all pre-existing. Tree left uncommitted.
