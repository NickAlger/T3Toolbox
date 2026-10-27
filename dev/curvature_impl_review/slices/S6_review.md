# S6a + S6 independent review -- VERDICT: APPROVE-WITH-FIXES
Base 8036fb2, tree uncommitted (8 files; my edits F1-F2, prose only). No BLOCKER/MAJOR. Scripts: `scratchpad/review/s6/`.

## Independent checks
- **S6a `from_frame`** (`c1_from_frame.py`): `var_masks = ufv_variation_masks(frame_data[5])`, no re-orthogonalisation, no
  damping arg. == `from_point` by value AND hash, every mask equal (shape, dtype, read-only), 24 cases (7 structures incl.
  Tucker/TT slack, d = 1, d = 4, maximal rank; C = (), (2,), (2,3); shared) + a non-minimal uniform x. A jitted fn of a backend
  LocalModel whose geom alternates from_point / from_frame traces ONCE (same aux key). Corewise keeps `from_point`: the code
  (`geometry.py:604-611`) stores the masks only, never orthogonalises -- right.
- **GN bit-identity vs 8036fb2** (`c1_bitident.py`, `c1_compare.py`): 192 keys, 2 seeds -- frontend `newton_cg` (uniform
  manifold / corewise / shared manifold / regularized + per-mode weight; ragged manifold / corewise / shared manifold) and the
  frontend models' objective / gradient / gn_hessian / gn_quadratic / evaluate / jacobian, with and without a regularizer, on
  all six geometries (the uniform regularized model is the `_bgeom` consumer): **0 diffs** (bytes).
- **Cost spy** (`c2_spy.py`): exact construction = 1 frame, 1 sweep, 1 transpose, 1 project, 1 `curvature_precompute`, 1
  `fv_lift_factors`; `.gradient` = nothing; `_blm` is lazy and reuses the sweep / curvature (no second sweep). `hessian(p)` = 1
  forward + 1 transpose + 1 two-hole + 4 projections (backend `hvp`: 2); `quadratic` / `evaluate` = 1 forward + 1 two-hole.
- **Pytree** (`c5_leaves.py`, `c5b_where.py`): at 8036fb2 a uniform model had int leaves `[5, 6, 7]`, all from the **sweep**;
  now none (no int / bool leaf); jit fine on both. The CHANGELOG Fixed bullet is accurate.
- **Exact model** (`c6_stack_jit.py`): C-stacked == per element (manifold, corewise; <= 2e-16); built INSIDE a jit trace ==
  eager (1.6e-16; H dense 1e-15); jax-hidden smoke: exact model + `newton_cg(hessian='exact', use_jit=True)` run, with warning.
- **Frontend == backend `newton_cg`** (`c3_front_back.py`, 6 own configs, order 2, tied x0: exact + reg + omega matrix; GN;
  'Exact' + power 1 + warm + cd 1e-3; corewise exact + reg; shared corewise exact + LS; GN + TR r0 .2): all **bitwise**. Every
  key reaching `topt.newton_cg` from T3Polynomial (`order weight max_newton verbose val_sample val_data g0norm_newton gtol_rel
  use_jit chunk_size` [d05/o01/e01/n01/x01/b01] `callback` [b01:275], + `g0_reference` §K, `hessian` S0b) is explicit and in
  `T3POLYNOMIAL_KEYS`. `**kwargs` gone (TypeError test `test_optimizers_frontend.py:440`); `not_yet_frontend` deleted;
  `gradient_descent` has `g0_reference` (S0d).
- **x07 FD** (`c4_fd.py`, 6 seeds x {minimal, d4, regularized, TT slack, Tucker slack}, h = 5e-2..5e-6 sigma_min): clean
  O(h^2) to a 1e-9..3e-7 roundoff floor; at the test's h = 5e-4 sigma_min worst exact error **2.8e-7** (bound 1e-5: 36x
  margin), GN/exact >= **3.6e6** (bound 1e3). Sound.
- **Mutants** (`mutate.py`, 17 on a scratch copy): 16 RED -- `_bgeom`->from_point (spy), hessian / quadratic / evaluate without
  the curvature term, damping dropped before `fv_lift_factors`, `hessian` not passed to `least_squares_problem`, uniform
  resolver without `hessian`, unflatten drops `curvature`, flatten without `partition_static`, retract without `geometry_aux`,
  GN+damping accepted, `newton_cg` dropping `curvature_damping` / `tr_radius0`, curvature_hessian unprojected, from_frame
  without the gauge shift, exact gradient recomputed. **1 survivor, equivalent today**: `_blm` with `geometry_aux=None` (only a
  shared manifold has an aux, and it raises S7).

## Fixes applied (in the tree)
- **F1 (MINOR)** `CHANGELOG.md:58-59`: the rename bullet said "the class's methods and fields are unchanged", false for the
  release once S6 adds `curvature` / `is_exact` / `hessian` ...; now "the rename leaves [them] as they were (the exact mode
  adds to them, under Added)".
- **F2 (NIT)** `t3toolbox/optimizers.py:528-531`: the exact doctest's prose implied GN reaches the gradient test; measured GN
  28 steps ending `'line_search'`, exact 5 (`c7_doctest_margin.py`). Prose fixed (names power 1.0); output unchanged.

## Remaining findings (NIT, not fixed)
n1 the handoff's "no precedent of frontend->backend-private calls" is wrong: `shared_geometry.py:134` calls
`backend_sharing._groups_from_labels` (at 8036fb2 too). n2 frontend T21 takes 3.06 s (budget <= 3 s, incl. the GN control);
the bitwise test 6.0 s. n3 `geometry.py:458` still says "build it ... with from_point"; `fitting.py:457-458` (`quadratic`)
lacks the "minimal, numerically full-rank" qualifier the retract docstrings carry. n4 uniform exact via `newton_cg` pays the
jax-measured `'auto'` estimate before the stub raises (window only, until S8).

## Rulings on the 9 judgment calls
1 **agree, keep private**: public names add backend `__all__` / autoapi surface for no raw-`.data` need (`Problem.local_model`
runs them); precedent exists (n1). 2 **agree**: lazy, no duplicated sweep (spy); `hessian` = gn + curvature is the C.13 table;
backend `hvp` to <= 1e-12 is fine (T18 asks "mode-dependent"; curvature actions and `quadratic` ARE bitwise). Optional S13:
exact `hessian` via `_blm.hvp` (2 fewer projections, bitwise). 3 agree (d = 1 bitwise == GN, tested). 4 **agree**: only error
paths pay frame + sweep; `newton_cg` validates `hessian` before setup. 5 **agree**, verified above. 6 **agree**: DEFERRED S4 m6
asks exactly "size 'auto'"; the docstring says it does not fix the mode. 7 agree (new in 2026.3.0). 8 agree (call sites
checked). 9 agree (tests + my shared-corewise bitwise run). **T21 pins (<= 10, GN > 2x)**: acceptable -- a GN-like
regression takes ~28 steps and fails `<= 10`; backend T21 pins 8/28 and the bitwise test ties the frontend to it.
**Dispatch**: `curvature_damping` is a LEAF (only `LiftFactors.eps`), as plan §F intends ("a different damping does not
recompile"); the test asserts 1 trace across None / 1e-3 / 0.0 x three points -- the intended thing.

## Plan-text amendments to queue for S13
(a) C.13 / T18: exact `hessian(p)` == backend `hvp` to rounding (two gauged halves); curvature actions bitwise. (b) C.13
`_with_curvature(model, hessian, curvature_damping, who)` via `dc.replace` (cached `_bgeom` rebuilt; cheap after S6a). (c) C.16
signature gains `hessian` (S4 m6 done). (d) `docs/fitting_and_optimization.md:256` "verified bit-identical": qualify for exact
`hessian`. (e) n3's two docstring touches.

## For S7 / S8
S7: a frontend shared-MANIFOLD exact == backend test must kill the `_blm geometry_aux=None` survivor; flip the S7 skips /
asserts (`test_fitting.py:1031`, `:1133`; the 'S7' assert in `test_optimizers_frontend.py`). S8: the uniform `_blm` (6-tuple
frame) and `_tangent_at` on bare supercores are unreached today -- add uniform frontend exact == backend and the uniform m27
(exact gradient bitwise == GN, masks included); flip the uniform ValueError asserts.

## Gates (re-run by me; targeted before F1-F2 (prose only), all others after)
Targeted (test_fitting, test_optimizers_frontend, backend/test_optimizers, test_curvature, test_sharing + the `from_point`
consumers backend/test_geometry, test_uniform_fitting, test_zero_reference) `-n 2`: 320 passed / 1,249 subtests.
test_dispatch: 48 passed / 36 subtests. Module doctests 210 passed; docs pages + getting_started.rst OK; sphinx -W exit 0.
FULL SUITE (last): dispatch + backend 302 passed / 1,963 subtests; the rest 728 passed, 1 skipped / 42,465 subtests. Slow
tier (serial): 5 passed / 12,790 subtests. Tree left uncommitted.
