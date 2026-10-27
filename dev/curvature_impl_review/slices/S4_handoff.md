# S4 handoff (implementer) -- DONE + review m1-m5 folded in; UN-PARKED onto main @ 4eacbcc (S3b); UNCOMMITTED

## Files (9)
- backend/fitting.py: `SamplingKind.corewise_hessian` (@abc.abstractmethod + NotImplementedError guidance, C.6 contract;
  "six operations"); `ProbeDerivativesKind.corewise_hessian` = the S3 `_from_sweep` kernel at omega^2 r, model sweep, chunk_size=self.chunk_size.
- backend/uniform_fitting.py: `_UNIFORM_HESSIAN_SLICE` + the marked raising stub on `_UniformKind`; module/:60 comments.
- backend/geometry.py: `DEFAULT_CURVATURE_DAMPING`, `ManifoldCurvatureData` (registered), private `_manifold_curvature_data`
  (no d=1 shortcut; T20 uses it); the four members on the two ragged geometries. New edge geometry -> t3_linalg.
- backend/optimizers.py: `CurvatureState`, `_check_hessian`, `_require_exact_capable`, `LocalModel` exact mode (+ private
  `_curvature_block`), `Problem.local_model(..., hessian, curvature_damping)` + doctest; flatten adds `m.curvature`;
  protocol docstring lists the four members. Imports unchanged (common, regularization, corewise).
- optimizers.py (frontend): `_resolve_chunk_size(..., hessian='gauss_newton')` -> `estimate_chunk_size(hessian=)`.
- tests: test_curvature.py +4 classes/19 tests (T7-T10, T6, stacks, mode/damping, sigma_min, T13 + shared corewise, T20,
  T23); test_dispatch.py +2 (jit every exact method + CG; one cache key); estimator resolver; CHANGELOG (2 bullets).

## Reproduced (scratch `impl/s4/repro_c4_c10.py`, through `Problem.local_model(hessian='exact').hvp`)
- c4: <v, hvp v> = **252.85288420** at eps=0 AND at the default 1e-6 (c4: 252.85288420); GN 285.03004656 (= c4);
  FD 252.85288211 (h=2.5e-3). c10: |K_N L p_vert| = 1.1e-11 vs 1.3e4. Mutants (`impl/s4/mutate.py`) all RED.

## Gates (pre-un-park: full suite 364 + 619 passed / 44,779 subtests, slow tier 5 / 12,790, doctests 208, pages OK)

## Review follow-ups (tests only; no code change)
m1 test_gauge_laws: curvature_quadratic / quadratic of an UNGAUGED p == of Pi p == <Pi p, curvature_hvp(Pi p)>.
m2 one misfit transpose: spy (autospec) -> 1 call per exact local_model, 0 on .gradient (GN: 0 then 1).
m3 sigma_min == min(sigma(S_i^T), sigma(T_i)), shape C; d=3 end bonds equal Tucker spectra, so the bond-carried cases
   are d=4 with a weakened middle bond. m4 Tucker/TT slack in T7 and T8. m5 TestCorewiseExact.test_shared_corewise:
   tied point, Hp tied, == Pi H_untied Pi (1e-13), == FD (1e-8). Mutants (`impl/s4/mutate2.py`) m1/m2/m3/m5 all RED.

## Un-park onto S3b (4eacbcc)
No S4 code touches `trs` / `compute_mu_jets` / `compute_nu_jets` / dxi~ (diff grepped; the kernel signature is unchanged).
Re-pinned `test_resolver`: S3b's measured estimator admits all of W at n_tangent = 1 in BOTH modes (floor/per-row >= 2.0
over 12 shapes, d 3-5, order 0-6, ranks <= 36, N <= 1000; this STRUCT o2: 2.89 GN / 2.55 exact), and the resolver
always sizes K = 1 -- so no problem makes the answers differ through it. Pinned instead: a spy on the forwarded mode,
the estimator's ValueError reached through the resolver, equality with direct calls. Mode-dropping mutant -> RED.
Re-run: chunk-size + curvature + optimizers + fitting + dispatch 164 passed / 805 subtests; module doctests of the
touched modules pass; sphinx -W exit 0; c4 = 252.85288420 and c10 reproduce on the new kernels.

## Conflicts / judgment calls
1. Uniform geometries UNTOUCHED (S8): the four-member check rejects them structurally. With the stub deleted, the
   uniform kind passes the kind check and the GEOMETRY check fires instead (verified) -- defence in depth until S8.
2. Shared MANIFOLD exact raises NotImplementedError('... lands in slice S7') in curvature_precompute (the untied lift
   would be silently wrong). Shared COREWISE is allowed: the generic body gives Pi(GN + K)Pi, the tied Hessian (m5 test).
3. Chunk size: 'auto' resolves only in the frontend `_resolve_chunk_size` (private `hessian=` kwarg). **S6 must add
   `hessian=` to `optimizers.least_squares_problem` (C.16 lacks it; reviewer m6) and pass it through.**
4. `_check_hessian` = S3's isinstance-str form (not C.9's `str(h).lower()`); structural checks before the GN build;
   negative damping validated before the d=1 shortcut; `float()` cast (C.7). All ACCEPTED by the reviewer.
6. `sigma_min` absolute, shape C. `curvature_quadratic` = <Pi p, y> (Pi self-adjoint). Stub takes `*args, **kwargs`.
7. c4 pinned only in scratch (LAPACK signs); T7 asserts exact vs FD. 8. Docs untouched (S13; :228 true for GN).

## S13 must append to dev/OPEN_QUESTION_extension_surface.md (§"What to watch")
"2026.3.0 adds a **sixth required operation**, `corewise_hessian`. Kinds without it keep working for Gauss-Newton and
fail early, with a structural error, for `hessian='exact'`." Also: the stub count (:30 "nine abstract stubs", :135
"there are nine") becomes ten, and the abc.abstractmethod marker is one more piece of the open surface's price.

## Draft commit message
backend: the exact local model (S4) -- SamplingKind.corewise_hessian, the required sixth operation (abc marker;
ProbeDerivativesKind implements it; _UniformKind stub until S8-S12); the geometries' exact-mode members,
ManifoldCurvatureData, DEFAULT_CURVATURE_DAMPING; CurvatureState + LocalModel exact mode; Problem.local_model(hessian=)
  Body: Slice S4 (C.6, C.7, C.9). T7-T10, T13, T20, T23, full-hvp symmetry, shared corewise; dispatch: exact jit, one
  cache key. Reproduces c4 (252.85288420) and c10. GN default unchanged; newton_cg untouched.
