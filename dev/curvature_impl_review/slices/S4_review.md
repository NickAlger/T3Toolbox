# S4 review (reviewer) -- base main @ 8c87123, implementer's tree UNCOMMITTED (9 files)

Status: COMPLETE. **VERDICT: APPROVE** -- the implementation needs no change; fold in the five small test additions (m1-m5) before the commit (no re-review needed) and record the plan amendment m6.


### Reproductions (done first)
- Implementer's scratch `impl/s4/repro_c4_c10.py` (through `Problem.local_model(hessian='exact').hvp`, kind = ApplyKind + block-sum
  oracle corewise_hessian): <v,hvp v> = **252.85288420** at eps=0 and at the default 1e-6; GN 285.03004656; c10 (a) 0.61, (b)
  |K_N L p_vert| = **1.1e-11** vs 1.3e4. Native c4 / c10: 252.85288420 (FD 252.85288182), 1.3e-11 / 2.4e-11. CONFIRMED.

### Item 1 -- the Hessian, independently (scratch `s4rev/indep_hessian.py`, log `s4rev/indep_hessian.log`)
Three oracles on my own seeded problems (data from a second unit point + 10% noise, so the curvature term is O(1): GN
is off by 73-110% everywhere except d = 1), eps = 0:
- (c) a FULLY DENSE extrinsic Riemannian Hessian sharing no curvature code with the library: my own einsum Phi, my own
  subset-expansion probe-derivative operator (its objective == `prob.objective` to 1e-10), the tangent space as
  range(J_Phi) by SVD (no gauge machinery), II = P_N D^2Phi via `jax.hessian` (float64) of theta -> <P_N grad f, Phi>;
  `E(hvp p)` vs `Q (Q^T nabla^2 f Q + Theta^T H_psi Theta) Q^T E p`.
- (b) jax float64 Hessian of the Lagrangian f o Phi - <G, Phi> with G = MY dense P_T grad f, through the library lift.
- (a) d^2/dt^2 f(tv_retract(t p)), one Richardson step, h = 0.05 min(sigma_min, 1).

| case | (c) dense extrinsic (gauged / ungauged p) | (b) Lagrangian | (a) retraction FD |
|---|---|---|---|
| minimal d3, order 0 / 1 (omega matrix) / 2 (per order) / 1 (per mode) | <= 2.5e-13 | <= 4.3e-14 | <= 3.0e-7 |
| Tucker slack (6,6,6),(4,4,4),(1,2,2,1), order 1 | 1.9e-14 | 2.2e-14 | 1.1e-7 |
| TT slack (6,6,6),(4,4,4),(1,9,9,1), order 2, omega matrix | 1.2e-14 | 5.8e-15 | 5.8e-8 |
| d = 1 order 2 (the shortcut; hvp == GN) | 5.9e-16 | 4.3e-16 | 5.0e-12 |
| d = 2, d = 4 (orders 0, 2, omega matrix) | <= 4.1e-14 | <= 3.2e-14 | <= 4.8e-7 |
| regularized lambda = 0.5 (minimal) / 20 (TT slack) | <= 9.2e-14 | <= 5.1e-15 | <= 4.4e-7 |

Also on every case: the dense Hessian matrix symmetric to <= 1e-14; `<p,Hq> = <q,Hp>` to <= 4e-13;
`curvature_quadratic(p) == <p, curvature_hvp(p)>` to <= 8e-15 (gauged AND ungauged p); exact `gradient` bitwise == GN
`gradient` (all 12 cases); `misfit_gradient` == my dense `P_T grad f_amb` to <= 2e-14; the regularized curvature uses
the misfit gradient only (lambda enters as lambda*xi, matching the dense oracle). **The exact Hessian is right.**

### Items 2-5 -- protocols, model, jit (scratch `s4rev/model_checks.py`, `s4rev/jit_checks.py`, `s4rev/mutate_src.sh`)
- **Stacks**: multi-axis K = (2,2), C = (2,3) (C-stacked point with per-element data), manifold and corewise: hvp /
  curvature_hvp / quadratic / curvature_quadratic per element == the unstacked model to <= 1.7e-15; quadratic shape K+C.
- **One misfit transpose** (spied): `local_model(exact)` calls frame, precompute, point_forward, transpose x1, project x1,
  curvature_precompute; `.gradient` calls nothing; GN `local_model` unchanged (no transpose). **hvp body order**
  (spied): project -> forward -> lift -> corewise_hessian -> tangential_curvature -> lift_transpose -> transpose ->
  project (ONE output Pi) -- exactly C.9. `hvp - curvature_hvp == GN hvp` to 2.7e-15 (ungauged p).
- **Gauge laws on slack frames** (TT slack, Tucker slack): hvp(p) == hvp(Pi p), Pi hvp == hvp, curvature_hvp(p) ==
  curvature_hvp(Pi p), all <= 1.1e-15 (the suite covers d2/d3/d4 minimal only).
- **Kind protocol**: `type(SamplingKind)` is `type` (no ABCMeta) -- a five-operation kind instantiates; the marker is
  read off `type(kind)`; MRO puts `_UniformKind` before the ragged kind for all six uniform kinds (T23 asserts it).
  **T23 mutation reproduced** (stub deleted): 7 failures (6 subtests + the structural-error assertion); with the stub
  gone `UniformProbeDerivativesKind.corewise_hessian is ProbeDerivativesKind.corewise_hessian` (True) and the GEOMETRY
  check then fires ("UniformManifoldGeometryOps does not implement curvature_precompute, ...") -- the implementer's
  defence-in-depth claim holds.
- **Shared corewise exact (adjudication: ACCEPT)**: on a tied point (sharing (0,0,1), order 2, omega per order) the exact
  hvp == Pi H_full Pi (jax float64 Hessian of f o Phi over all core copies, Pi = group mean) to **1.3e-15**; quadratic ==
  d^2/dt^2 f(retract(t p)) to 3.6e-7 (Richardson); output tied bitwise; symmetric to 3e-16; GN is off by 6%. The
  generic body IS the tied Hessian (linear tied subspace, Euclidean metric, additive retraction). **Shared manifold
  raising NotImplementedError naming S7 (ACCEPT)**: Pi_sh Hess_untied Pi_sh would miss the tied submanifold's own
  second fundamental form (DP/F14: the tied lift is needed), so raising is the only safe window behaviour.
- **Uniform geometries untouched (ACCEPT)**: they lack the four members, so `_require_exact_capable` rejects them
  structurally; the kind stub fires first today.
- **jit**: the exact CG solve (`_cg_solve(..., use_jit=True)`) over 4 models rebuilt at new points with dampings
  None / 1e-3 / 0.0 / 1e-6 traces hvp ONCE (manifold; corewise likewise), with chunking engaged (chunk_size=3);
  jit == eager CG to 1.8e-16; quadratic / curvature_quadratic / curvature_hvp jitted across 5 dampings: 1 trace; every
  leaf a jax array; the curvature subtree's treedef holds no array (the only array in the model's aux is the kind's
  pre-existing value-hashed `weight`). `partition_static` keeps host-bool masks / ints static, so S8's uniform
  `ManifoldCurvatureData` (z with masks) will flatten with masks in the skeleton -- the path does not preclude it.
- `backend/optimizers.py` imports unchanged (common, regularization, corewise; jax lazily). geometry -> t3_linalg is the
  only new edge (plan M lists it).
- **Kind protocol, verified**: `type(SamplingKind) is type`, `SamplingKind()` itself instantiates, no
  `__abstractmethods__`; value hash/eq of kinds unchanged. UniformProbeDerivativesKind MRO = [UPDK, _UniformKind,
  ProbeDerivativesKind, _WwPpSample, ProbeOutputKind, ...] (plan's claim). The `_UNIFORM_HESSIAN_SLICE` table names
  S8 / S10 / S11 / S11 / S12 / S12 (C.6 / H).
- **Item 6 (safe mode)**: no `safety` call added anywhere; every new error is structural (ValueError for a bad mode,
  a no-effect damping, a negative damping, a missing kind op / geometry member; NotImplementedError for the two
  window stubs). Caveats only. CONFORMS to C.12.
- **jax-hidden smoke** (sys.modules['jax'] = None): exact models on manifold + corewise construct CurvatureState /
  ManifoldCurvatureData, quadratic / curvature_quadratic evaluate, `_cg_solve(..., use_jit=True)` runs eager.

### Mutations
- Implementer's, reproduced: delete the `_UniformKind` stub -> 7 failures (T23); drop K_{grad f} -> 29 (18 F + 11 E);
  omega instead of omega^2 -> 12; sign flip on K_{grad f} -> 29. All match the handoff.
- Mine (source-level, `s4rev/mutate_src.sh`, S4 classes + the two dispatch tests): lift p un-projected -> 4 RED;
  GN part forward(p) un-projected -> 5 RED; misfit gradient un-projected -> 24 RED; wrong base point (U, R) in the
  kind call -> 21 RED; wrong base point in `tangential_curvature` -> 24 RED; d = 1 shortcut removed -> 1 RED;
  `_flatten_local_model` drops `curvature` -> 3 RED (dispatch); CurvatureState flattened as aux -> 1 RED (cache key).
  **Survivors (test gaps, m1-m3 below):** `curvature_quadratic` with `inner(p, y)` (un-projected p) -> all green;
  exact `.gradient` re-running the transpose -> all green; `sigma_min` without the bond spectra -> all green.

## Findings

No BLOCKER, no MAJOR.

**MINOR**
- **m1 (test gap) `curvature_quadratic` / `quadratic` on an UNGAUGED p are untested.** Every S4 test draws p through
  `geom.project` (`_unit_tangent`), so the mutation `return self.geom.inner(p, self._curvature_block(pp))`
  (optimizers.py:309) survives. The code is right (my check: `curvature_quadratic(p_ungauged) == <p, curvature_hvp(p)>`
  to 4e-16). Fix: in `test_gauge_laws` (test_curvature.py, the ungauged `p` already drawn there) add
  `curvature_quadratic(p) == curvature_quadratic(Pi p)` and `quadratic(p) == quadratic(Pi p)` (rel 1e-12).
- **m2 (test gap) the "ONE misfit transpose per model" contract (C.9, A.4, M2) is not pinned.** Mutating `.gradient`
  to recompute the transpose in exact mode (optimizers.py:247 `if False:`) keeps every test green (the result is
  bitwise the same). Fix: in `test_mode_is_derived_and_the_gradient_is_the_gauss_newton_one`, spy
  `mock.patch.object(bfit.ProbeDerivativesKind, 'transpose', wraps=...)`: `local_model(x, hessian='exact')` -> 1 call,
  `.gradient` -> 0 calls (the GN model: 0 and 1).
- **m3 (test gap) `sigma_min`'s value is not pinned** (C.7 "the real-slot minimum"; it is T25/T19's lift_sigma_min
  later, and it sizes T7's own FD step). Dropping the bond spectra (geometry.py:185) keeps everything green. Fix: assert
  `curvature.geometry_data.sigma_min == min(min sigma(S_i^T), min sigma(T_i))` from `fv_center_factors` (T1's
  quantities) on a C-stacked model (shape C) and on TT slack (where a bond spectrum is the minimum).
- **m4 (coverage) T7 / T8 never run on the non-minimal frames.** Plan F lists Tucker slack and TT slack among the
  structures, and they are exactly where the lift is a one-sided pinv (F3). T7's cases are minimal / d2 / d4 / rank-1;
  T8 is d2 / d3 / d4 minimal (slack appears only in the symmetry test). My oracles pass there (dense extrinsic 1.9e-14 /
  1.2e-14, FD 1.1e-7 / 5.8e-8; gauge laws <= 1.1e-15), so this is cheap: add `('tucker_slack', _TUCKER_SLACK)` and
  `('tt_slack', _TT_SLACK)` to T7's `cases` and T8's structures.
- **m5 (coverage) shared COREWISE exact is enabled but untested.** The implementer allows it (`CorewiseGeometryOps.
  curvature_precompute` ignores `groups`) and defers tests to S7's T12. I verified it is correct (1.3e-15 vs Pi H Pi,
  output tied), but a live path should carry its regression test in the commit that enables it: add one subTest to
  `TestCorewiseExact` on `_COREWISE.with_sharing((0, 0, 1), shape)` at a tied point (FD of the gradient along a tied
  p, as the existing test does, plus `Hp` tied). Alternatively make shared-corewise exact raise until S7 like the shared
  manifold -- but it is correct, so the test is the better fix.
- **m6 (plan amendment, record; no S4 code change)** CONFIRMED: `'auto'` reaches `estimate_chunk_size(hessian=)` only
  through the private `_resolve_chunk_size` (t3toolbox/optimizers.py:154), and its only callers (:287, :300, inside
  `least_squares_problem`) pass no mode -- so a uniform exact fit would size the chunk for GN. Plan C.16's signature
  lacks the keyword. **Amend C.16**: `least_squares_problem(..., hessian: str = 'gauss_newton')` (validated like every
  `hessian=`; it changes only the `'auto'` resolution, because the chunk is a FIELD of the kind fixed at construction),
  and **C.13**: frontend `newton_cg(hessian=...)` passes it through. Lands in S6 (the frontend wiring); it first changes
  numbers in S8 (ragged resolves to None either way).

**NIT**
- n1: `_check_hessian` + `_HESSIAN_MODES` (backend/optimizers.py:131-144) duplicate S3's `_check_hessian_mode` +
  `_HESSIAN_MODES` (sampling_derivatives.py:2016) verbatim, and S5/S6 will need it again (backend + frontend
  `newton_cg`, the factories). One T3-agnostic `check_hessian_mode` in `backend/common.py` keeps the optimizers
  import invariant and one message. (The `isinstance(str)` form vs C.9's `str(h).lower()`: ACCEPT -- same outcome for
  every str and bool, consistent with S3.)
- n2: `Problem.local_model` signature (optimizers.py:341-347): the `curvature_damping` comment staircases 4 columns past
  the others and is truncated ("None -> the geometry's"); signature_style.md: align or group. -> "None -> the
  geometry's default", aligned.
- n3: `hvp` (optimizers.py:266-267): the `# exact: ...` comment sits on the GN branch's `if not
  self._has_curvature_term:` line; move it to the `else:`.
- n4: the shared-manifold `NotImplementedError` fires inside `curvature_precompute`, i.e. after the misfit transpose,
  unlike the other exact-mode structural checks ("before any work"). Window-only (S7 deletes it) and optimizers.py must
  not read `groups`; no action needed.
- n5: 8 new test lines exceed 120 columns (max 134; HEAD had 1 in the file). Cosmetic.
- n6 (advice for S5, not an S4 defect): C.9 step 6 evaluates `lm.quadratic(p)` and `lm.curvature_quadratic(p)`
  separately -- two two-hole sweeps per Newton step (spied: `quadratic` runs its own lift/two-hole/lift_transpose).
  S5 can take `pKp = curvature_quadratic(p)` once and `pHp = gn_quadratic(p) + pKp`.

### Adjudications requested
1. Uniform geometries untouched: ACCEPT (structural rejection by the four-member check; defence in depth behind the
   kind stub -- verified with the stub deleted).
2. Shared manifold exact raises naming S7: ACCEPT (the untied lift would be silently wrong). Shared corewise exact
   allowed: ACCEPT on the math (verified independently) -- but see m5 (add its test now).
3. Chunk size: CONFIRMED; recorded as plan amendment m6 (C.16 + C.13, S6).
4. `_check_hessian` isinstance form: ACCEPT (n1). Structural checks before the GN build: ACCEPT (earlier error, same
   semantics). Negative damping validated before the d = 1 shortcut: ACCEPT. `float()` cast: ACCEPT (C.7).
5. `curvature_quadratic` as `<Pi p, y>` with y ungauged: ACCEPT -- Pi is self-adjoint in `geom.inner` (measured: 0.0 on
   the manifold, 5e-16 on the SHARED manifold too, so it stays valid in S7); C.9 specifies exactly this. (m1 pins it.)
6. c4 pinned only in scratch: ACCEPT (LAPACK sign dependence; T7 is the BLAS-independent form).
7. Docs untouched until S13; OPEN_QUESTION "six operations" text queued for S13: ACCEPT (plan G routes both to S13;
   docs/fitting_and_optimization.md:228's "five operations" remains true for GN).

### Item 7 -- tests / style
- T7-T10, T13, T20, T23 present; full-hvp symmetry (T6), K+C stacks, mode / damping rules extra. Seeded (setUp
  `np.random.seed` + per-test `default_rng`), `subTest` over structures / weights / dampings, numpy-only (no jax import
  in test_curvature.py). Dispatch: `test_jit_exact_local_model` (every exact method + the CG loop, manifold and
  corewise) and `test_exact_local_model_rebuilt_each_step_is_one_jit_cache_key` (points x dampings, one trace; GN is
  its own treedef). Chunked jit of the kernel is S3's dispatch row (W = 7 at chunk 2); I also ran the exact CG with
  chunk_size = 3 under jit (one trace).
- House style: signature columns and shape comments on the new public functions; `Problem.local_model` doctest is a
  seeded wrong-vs-right example (GN misses the FD second derivative, exact matches); CHANGELOG `[2026.3.0]` Changed
  (sixth operation) + Added (exact backend local model) bullets are accurate; every deleted line in the diff is a
  purposeful edit (no formatter churn). Only nits n2, n3, n5.

### Item 8 -- runs (all on the implementer's uncommitted tree; numpy 2.5.3, jax 0.12, 2 cores, `-n 2`)
- Targeted: test_curvature + test_optimizers + test_fitting + test_uniform_fitting + test_dispatch +
  test_chunk_size_estimator: **188 passed, 910 subtests** (157 s).
- `pytest --doctest-modules t3toolbox`: **208 passed**. `docs/getting_started.rst`: OK. Every `docs/*.md` +
  `docs/contributor/*.md` page (minus doctest_style.md): OK.
- Full suite in two halves (first 19 / last 18 test files): **310 passed, 4,635 subtests** + **673 passed, 1 skipped,
  40,144 subtests** = 983 passed / 44,779 subtests -- identical to the implementer's 364 + 619 / 44,779. The one
  warning is the designed adam-on-manifold RuntimeWarning (pre-existing).
- Slow tier (`T3TOOLBOX_SLOW_TESTS=1 tests/test_oracle_sweep.py`): **5 passed, 12,790 subtests**.
- **`sphinx -W --keep-going -b html` (sphinx 9.2.0 + autoapi + pydata theme ARE installed here, contrary to the
  handoff's "not installed"): exit 0, zero warnings** (built from a scratch copy of the tree).

## Must-fix before the commit (tests only; no re-review needed)
1. m1 -- ungauged p in `curvature_quadratic` / `quadratic` (kills the surviving `inner(p, y)` mutant).
2. m2 -- spy the transpose count: 1 per exact `local_model`, 0 per `.gradient` (kills the recompute mutant).
3. m3 -- pin `sigma_min` against the S_i / T_i spectra, C-stacked and TT slack (kills the bond-spectra mutant).
4. m4 -- add Tucker slack and TT slack to T7 and T8.
5. m5 -- one shared-corewise exact subTest in TestCorewiseExact (the path is live in this commit).
Plan amendment to record (m6): C.16 `least_squares_problem(..., hessian='gauss_newton')`, threaded by C.13's frontend
`newton_cg`, in S6. Nits n1-n6 optional (n6 is advice for S5).

VERDICT: APPROVE

---

## Re-review after the un-park (S3b landed under S4: main @ 4eacbcc; S4 staged + one unstaged test edit)

**(1) The re-pinned `test_chunk_size_estimator.py::TestFittingChunkSizeWiring::test_resolver` -- ACCEPT.**
- The old `assertLess(exact, gn)` is now unsatisfiable through the resolver: I printed both answers for its STRUCT at
  W = 40000, order 2 -> **exact 40000, gn 40000** (the resolver sizes one CG direction, K = 1, and the measured
  balance admits all of W in both modes, as the comment says).
- The spy is the right instrument and is specific: `[c.kwargs['hessian'] for c in spy.call_args_list] ==
  ['exact', 'gauss_newton']` pins the exact string forwarded for each call (and the GN default), plus a bad mode
  raising through the resolver and equality of both answers with direct estimator calls. The spy wraps
  `sampling_derivatives.estimate_chunk_size`, which the resolver looks up at call time via `_bfit_pd()` (same module
  object), so it sees the real call.
- Non-vacuous, by mutation of `t3toolbox/optimizers.py`'s resolver: drop the `hessian=` keyword -> RED; hard-code
  `'gauss_newton'` -> RED; hard-code `'exact'` -> RED; forward `'EXACT'` for `'exact'` -> RED (strict about the
  verbatim pass-through, which is what the resolver promises).
- NIT (no action): without jax the resolver returns `DEFAULT_CHUNK_SIZE` before the estimator, so a bad `hessian`
  is not validated there; the class is jax-gated and S6's frontend `newton_cg` validates its own `hessian` anyway.

**(2) CHANGELOG merge -- reads correctly.** The staged diff vs 4eacbcc is **additions only** (0 deleted lines): S3b's
breaking bullet (the recurrences, the four sub-bullets: `trs` gone, `dx̃i` two rows, the removed twins, the
estimator measuring the running program) and its `### Fixed` section (jet outputs keep their dtype) are intact;
S4's "sixth operation" bullet closes `### Changed -- breaking`, and S4's "exact Riemannian Hessian in the backend
local model" bullet closes `### Added`, each exactly once and verbatim as reviewed. S3's own Added bullet carries
S3b's updated "~1.3-2.2x" figure. NIT: the blank line between S3b's nested sub-list and S4's "sixth operation"
bullet makes that Markdown list "loose" (every item rendered as a paragraph); delete the blank line for consistency.
- m1-m5 as folded in: my three earlier surviving mutants are now RED -- `curvature_quadratic` with un-projected p
  (4 failures, incl. the new TT-slack T8 row), the exact `.gradient` re-running the transpose (the transpose spy),
  `sigma_min` without the bond spectra (the new sigma_min test, 3 failures). T7/T8 include both slack structures;
  `test_shared_corewise` is present.

**(3) The hvp is unchanged by S3b, to roundoff.**
- Direct before/after (`s4rev/before_after.py`): the S4 code run once with S3's kernels (8c87123's
  `sampling_derivatives.py` + `uniform_tucker_tensor_train.py` swapped into a copy of the tree; S3b touched none of
  S4's files) and once with S3b's, same seeded problems: 7 cases (minimal order 0 and order 3 with an omega matrix,
  Tucker slack order 2 chunked, TT slack order 2, d = 4 order 2 with K = C = (2,) chunked, d = 1, minimal with
  K = C = (2, 3) chunked) x {manifold, corewise} x {hvp, curvature_hvp, gradient, quadratic, curvature_quadratic}:
  **max relative difference 9.5e-15** over 68 groups (the two d = 1 curvature groups are exactly 0 in both).
- T7 (retraction FD) and T9 (dense Lagrangian): pass (20 subtests). My dense extrinsic oracle on the new kernels
  (`s4rev/indep_hessian_s3b.log`): every number matches the pre-S3b run to its printed digits -- (c) <= 2.6e-13,
  (b) <= 4.3e-14, (a) <= 4.8e-7, all 12 cases incl. slack, d = 1/2/4, regularized. c4 = **252.85288420**,
  c10 = **1.1e-11** through the scratch route.

**(4) Runs on the re-review tree (staged S4 + the unstaged resolver test; numpy 2.5.3, jax 0.12, 2 cores).**
- Full suite, two halves (`-n 2`, the same 19 / 18 file split as before): **317 passed, 4,562 subtests** +
  **669 passed, 1 skipped, 39,756 subtests** = **986 passed, 1 skipped, 44,318 subtests**, 0 failures / errors (the
  only warning is the designed adam-on-manifold RuntimeWarning). (Counts differ from the first review's 983 / 44,779
  because S3b reshaped the jet tests and S4 gained m1-m5.)
- Slow tier, serial (`T3TOOLBOX_SLOW_TESTS=1 pytest tests/test_oracle_sweep.py`, no xdist): **5 passed, 12,790
  subtests** (210 s).
- `tests/test_chunk_size_estimator.py`: 14 passed / 18 subtests.

**Nits from this round (optional):** the CHANGELOG blank line inside the Changed list; the resolver's jax-absent
early return skips `hessian` validation.

FINAL VERDICT (re-review): APPROVE -- commit the staged S4 files together with the unstaged
`tests/test_chunk_size_estimator.py` edit.
