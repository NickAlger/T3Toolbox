# Critic checklist — Riemannian-Newton curvature implementation plan

Repo `/home/claude/t3toolbox` @ `920d776`. Every line: the requirement · where it is stated. "DP" = the
settled design `dev/riemannian_newton_curvature_plan.md`. I hold the plan to these; anything a plan cites
(function, file:line, doc §) is verified by grep/read/run — **an invented name is a BLOCKER**.

Pre-verified facts (read-only run, scratchpad `check_lift_objects.py`, minimal + non-minimal frames):
- `T_i := einsum('abi,abj->ij', L_i, H_i^X)` **==** `tt_zipper_right_to_left(L, R)[i+1]` exactly (i ≤ d−2);
  centers `H_i^X == L_i Z_{i+1}` (== construction's tt_variations); `S_iᵀ = einsum('axb,aub->xu', O_i, H_i)`
  (`sharing.py:552-554`) and construction `V_i^X == S_iᵀ U_i`. Non-minimal frame: `S_iᵀ` is `nD×nU` (e.g. 2×4) → wide.
- Lift orientation (DP §4.1) checks out: Tucker term `L_{<i} H_i^X(U̇_i) R_{>i} = L_{<i} O_i (S_iᵀU̇_i) R_{>i}` ⇒
  `V_i = S_iᵀU̇_i`; TT `H_i = Ġ_i ×₃ T_i` ⇒ `Ġ_i = einsum('abr,rc->abc', H_i, pinv(T_i))`; `Ġ_{d−1} = H_{d−1}`.
  `Lᵀ(Y) = (S_i⁻¹ Y_U, Y_G ×₃ T_i⁻ᵀ)` (Frobenius adjoint). DΦ∘L = E holds term-by-term (any gauge).
- Raw left-canonical gauge θ = `ManifoldGeometryOps.base_point(frame) = (frame[0], frame[2]) = (U, P)`
  (`backend/geometry.py:183-189`); corewise frame of θ = `t3_corewise_frame` (`fv_conversions.py:30-41`) = (U,P,P,P).
- **Reuse fact:** at the corewise frame (U,G,G,G) the existing tangent-forward sweep IS the one-hole companion of the
  frame sweep: `compute_sigma` (Q=O=G) = d/ds `compute_mu`; `compute_tau` = d`nu`; `compute_deta` = d`eta`;
  `compute_dxi(θ̂_U, ww)` = d`xi` (`probing.py:336-523`); same for `compute_{sigma,tau,deta}_jets`
  (`sampling_derivatives.py:958-1050`). mu (and xi) at the corewise frame of (U,P) == the orthogonal frame's mu
  (both sweep through P); nu/eta/sigma_hat differ (P vs Q/O).
- Reuse VERIFIED numerically (`check_reuse.py`): `tv_apply(ww, θ̂, t3_corewise_frame(θ))` and
  `tv_probe(ww, θ̂, t3_corewise_frame(θ))` == central-FD d/ds of `t3_apply`/`t3_probe` at θ + sθ̂.
- jax is NOT installed in this env (numerics checkable numpy-only).

## 0b. Minimal format, continuation, varieties (the regimes the lift lives in)
- [ ] Lift is a genuine inverse only in MINIMAL format: `S_i` square iff `nD_i == nU_i` (Tucker ceiling
  `n_i ≤ rL_i·rR_i`), `T_i` square iff `rL_{i+1} == rR_{i+1}` (TT bond ceilings); otherwise wide/tall → pinv,
  and the slack directions must be handled (they carry no curvature) · docs/t3svd_minimal_ranks.md "What minimal ranks
  means"; docs/frame_variations.md "Four rank stores"; DP §4.1
- [ ] Minimal-format square S/T can still be numerically singular (rank-deficient point) → damping regime, distinct
  from the structural slack case; the plan distinguishes the two · numerical_contracts.md retract row; DP §4.4
- [ ] Ragged manifold `x0` is reduced to minimal on entry (`_ragged_minimal`), uniform via `uniform_minimal` — so the
  optimizer path sees minimal frames, but the backend/model surface does NOT (non-minimal must still be correct) ·
  optimizers.py:116-126, 206-217
- [ ] Rank continuation: every level starts at a zero-padded `resize` warm start ⇒ new Tucker/bond singular values
  EXACTLY zero ⇒ S_i/T_i singular ⇒ the damped lift / phase-in is the COMMON case, not an edge case; continuation is a
  ragged-layer activity (uniform per level via `from_t3`/`to_t3`); g0norm pinning guidance interacts with any forcing-
  term change · docs/rank_continuation.md "warm-start gotcha", "On the uniform layer"; DP §4.4
- [ ] Shared restarts: new `s_g` levels exactly zero, tied channel gated, escape via TT variations — curvature/damping
  must not block that escape · rank_continuation.md §Shared; sharing_internals.md "Padded restarts"
- [ ] Varying-rank C stacks (variety, not one manifold): curvature per element with per-element masks/damping; the
  uniform hvp must not couple stack elements · docs/uniform_ranks_and_varieties.md "tangent layer keeps the varying-rank
  stack"
- [ ] W-chunking (`chunk_size`, uniform+jax only, ≥3 chunks for the memory bound, `estimate_chunk_size` sizes the
  LARGER assembly): any new two-hole assembly either chunks with the same reducer seam or documents why not; 'auto'
  estimator coverage stated · docs/chunking.md "What chunk_size does", "Choosing chunk_size"

## 0. Process / scope (CLAUDE.md "Workflow"; DP status)
- [ ] Incremental reviewable slices, each with its own gate (full suite + doctests + `sphinx -W`) and a commit · CLAUDE.md Workflow
- [ ] Genuine decisions listed for Nick (not silently made); MINORs either fixed or deferred to that list · task brief; CLAUDE.md Workflow
- [ ] Implements the SETTLED math (L2 exact Riemannian Hessian, hand-written two-hole sweep shipped, routes 1–2 as oracles/prototypes only, S_i/T_i lift, Steihaug–Toint + TR, damped lift); re-litigation only with new evidence · DP status banner, §4
- [ ] Stage 0 "measure first" (DP §4.7) either kept or explicitly deferred with reason · DP §4.7
- [ ] Blast-radius rule: grep ALL consumers of any changed protocol/signature (LocalModel ctor, GaussNewtonModel ctor + pytree, NewtonInfo, SamplingKind) · CLAUDE.md Workflow; contributor/refactoring_methodology.md

## 1. Backend/frontend razor + parameters-as-fields
- [ ] All nontrivial math (lift, damped pinv, K_Z two-hole sweeps, Steihaug step, TR update) is a public backend function on raw `.data`; frontend thin · CLAUDE.md Architecture razor; contributor/fitting_internals.md §Backend-first
- [ ] A raw-`.data` user is not left "one fiddly step short" (e.g. lift + K + Lᵀ + Π composed in one backend entry point) · CLAUDE.md razor corollary
- [ ] Backend classes: parameters = FIELDS, behaviour = METHODS; every method's math also a standalone function · contributor/parameters_not_closures.md "The decision"
- [ ] Anything riding as jax aux (curvature mode, kind, geometry) is `ValueHashedFields` (frozen, eq=False); no callable fields, no closures/partials as aux · parameters_not_closures.md "Honest limits"; common.py:597-675
- [ ] Per-iteration-changing VALUES (ε damping, Δ trust radius, CG tol, forcing η) are traced leaves/args, never aux (aux ⇒ recompile per iteration) and never closed over (⇒ stale) · scan_body_principles.md §5; parameters_not_closures.md "Where to put a jit boundary"
- [ ] Per-frame companion (S_i, T_i, their damped pinvs/svals) derived on demand (zipper), NOT retained on the frame (mechanism 3 rejected) · contributor/precompute_and_caching.md "Examined and left alone"; sharing_internals.md §S_i machinery
- [ ] New precomputed object: names its invalidation scope + channel (Problem field / aux / leaf beside `sweep`); a NEW channel is a design decision to weigh; keep an `aux=None`-style recompute fallback; audit every consumer seam · precompute_and_caching.md "Checklist"
- [ ] Residual-dependent adjoint state for K_{Aᵀr} (fixed during inner CG) computed once per local model, not per matvec — OR the store-vs-recompute choice explicitly weighed vs the low-memory preference · fitting_internals.md §"Low-memory transpose"
- [ ] New companion data class holding arrays: frozen, eq=False, registered pytree with arrays as leaves + statics as aux (the `SharedFrameData` precedent) · backend/sharing.py:445-476, :1081-1091
- [ ] Geometry-specific logic (lift needs the manifold frame; shared lift needs companion) lives in/through the geometry, kind-specific (sampling) logic in the kind — the two axes stay independent · fitting_and_optimization.md §1, §4.1

## 2. Ragged vs uniform
- [ ] Every new op has a uniform mirror satisfying `to_uniform→op→to_ragged == op_ragged` on real parts (or an explicit "uniform deferred" decision for Nick) · uniform_equivalence_contract.md; CLAUDE.md "uniform mirror is COMPLETE"
- [ ] Polymorphic vs `ut3_`/`utv_`/`ufv_` twin chosen by the lenses (read-out → polymorphic via mask-once; T3-like output → twin) · contributor/uniform_polymorphism.md Lens 1–2
- [ ] Masks host numpy via `np` (never `xnp`), `require_concrete_masks` at chokepoints; supercores via `xnp` · CLAUDE.md dispatch exception; contributor/uniform_pytree_composition.md ⚠️
- [ ] Transpose-like outputs carry the frame's gauge-shifted variation masks broadcast over K_new (`ufv_variation_masks_over_stack` / `utv_sampling._gauge_masks_over_Knew`) · utv_sampling.py:419-429; ufv_masking.py:170-189
- [ ] Lift output (core coordinates) carries the corewise-frame variation masks (`ufv_variation_masks((tk,tk,tt,tt))`), nD-vs-nU and rR-vs-rL padded dims handled · geometry.py:458-464; uniform_fitting.py:88-91
- [ ] Mask-once on entry, re-mask after any SVD; padded S/T singular values map to 0 (guarded double-`where`), never `1/0=inf → 0·inf=nan`; real-slot zeros are NOT clamped silently (that is the damping's job, explicit) · contributor/uniform_internals.md masking schedule; weighted_internals.md §reciprocal
- [ ] Anything SVD-based that yields a KEPT BASIS routes through `pad_safe_svd`; spectra-only/pinv SVDs of masked small matrices may use plain SVD but must be pad-transparent · contributor/uniform_svd_prefix_orthogonalization.md "pad-safe"; linalg.pad_safe_svd
- [ ] K_{grad f} on uniform: the doubled-rank Z = `utv_to_ut3` has GAPPY `[Q|P]` TT masks + honest-zero boundaries — consumed through the mask, not a prefix slice · utv_operations.py:184-322; testing_strategy.md "phantom rank"
- [ ] d-leading supercore layout `(d,)+C+(...)`; d-vectorized einsums for maps, `xscan` for sweeps; no Python loop over d on uniform (unroll trap) · uniform_supercore_layout.md; sampling_derivatives.py:271-277
- [ ] Varying-rank-across-C stacks: per-element masks/damping; masked-out directions in ker(J) so CG reductions unpolluted · uniform_ranks_and_varieties.md "tangent layer"
- [ ] Uniform jit recipe: masks loop-invariant (from the geometry/kind fields), only supercores traced; compile once across Newton iterations · uniform_backend_jit_recipe.md; contributor/uniform_internals.md "masks as loop-invariant state"
- [ ] Uniform requires minimal-rank start (`uniform_minimal`); zero-padded restarts gauge-EQUIVALENT not bit-identical → cross-layer tests compare gauge-invariants, start nonzero full-rank · uniform_fitting.py:347-374; uniform_equivalence_contract.md §"full-rank qualifier"; fitting_and_optimization.md §4.11
- [ ] No uniform `vmap` assumptions (masks are aux) · batching_and_stacking.md §7

## 3. Safe mode
- [ ] Structural problems hard-error always (shape/rank/stack); numerical PRECONDITIONS only in safe mode via `safety.checks_active(...)` → `safety.require(...)`, skipped under unsafe/jit; CAVEATS never enforced · CLAUDE.md House philosophy; numerical_contracts.md "classification rule"
- [ ] Lift invertibility / full Tucker & bond rank (σ_min(S_i), σ_min(T_i) > 0) is a CAVEAT (exactness), never a precondition — zero-padded restarts and the zero start live there by construction · numerical_contracts.md "Minimal ranks verdict"; sharing.md "full shared rank is a diagnostic, never a precondition"
- [ ] Minimal rank NOT a precondition; non-minimal frames handled by pseudo-inverse (documented caveat) · numerical_contract_catalog.md "Minimal ranks"; frame_variations.md "Four rank stores"
- [ ] Same-frame (SF) guard on any new frontend `GaussNewtonModel` method taking `p` (`_require_at_frame`) · numerical_contracts.md §GaussNewtonModel; fitting.py:101-118
- [ ] ORTH is a construction invariant of the model (no per-matvec check); any new frontend geometry-level op that needs ORTH uses `_require_orthogonal_frame` · numerical_contract_catalog.md GaussNewtonModel row; manifold.py:1188-1204
- [ ] Backend stays check-free (frontend-only enforcement; backend mirror deferred) · numerical_contract_catalog.md sign-off 3
- [ ] New checker/diagnostic names follow checker grammar (frontend `is_`/`has_`; backend `<family>_<subject>_<predicate>`, `_residual` pairs), per-stack-element · naming_conventions.md §Checkers; batching_and_stacking.md §9

## 4. Batching and stacking (hvp MUST carry K and C)
- [ ] Every new signature states its stack contract frame-inner `W + K + C` (variations `K+C`, frame/lift companion `C`, samples `W`, residual `W+C` (+order leading)) · batching_and_stacking.md §2-3; CLAUDE.md batch blocks
- [ ] Machinery choice: one broadcast prefix → leading `'...'`; independent blocks on different operand subsets (W on samples, K on θ̂, C on cores) → `contract('WKCa,...')` with `len_*` only where the string requires · batching_and_stacking.md §4; grouped_contractions.md §3
- [ ] Per-frame quantities (S_i, T_i, pinvs, gradient coords g for K_{grad f}) carry C only and broadcast over K for free (C innermost) — never tiled to K+C · batching_and_stacking.md §3; weighted_internals.md "batches with the FRAME"
- [ ] Two-hole sweep: residual/adjoint state `W+C` (no K), θ̂ and one-hole companions `W+K+C`, output `K+C` (W summed) — the `_sigma_step` t2-style `len_C=` pattern where K rides only on a variation core · probing.py:363-377; apply.py:357-400
- [ ] Coordinate inner: `corewise_stack_dot` (keeps stack) vs `corewise_dot` (collapses all) chosen deliberately; geometry `inner` is per-element · CLAUDE.md gotcha; geometry.py:218-222
- [ ] Optimizer loop may stay single-point (stacked optimizers raise NotImplementedError; regularizer on stacked raises) — but LocalModel/GaussNewtonModel curvature hvp must work for stacked C and K-stacked p · deferred_and_rejected.md "Stacked optimization"; fitting_and_optimization.md §4.10
- [ ] Tests over the stack matrix (no-stack, C, K, K+C, multi-axis, forced pad, varying-C) · testing_strategy.md "The stack matrix"

## 5. Naming
- [ ] Family prefixes on public backend fns (`t3_`/`ut3_`/`tv_`/`utv_`/`fv_`/`ufv_`/`tt_`), u-modifier first; polymorphic fn = ragged name · naming_conventions.md grammar + polymorphism
- [ ] `corewise` marker on core-coordinate math (K_Z acts on raw core variations); `numerically_` only for SVD-grade checks; markers rename-invariant · naming_conventions.md §Semantic markers; contributor/naming_rules.md
- [ ] `_from_sweep` / `precompute_*` hook naming mirrors existing seams; `_trs` reserved for dense-binomial reference twins · naming_conventions.md `_trs`; fitting_internals.md "precompute/*_from_sweep split"
- [ ] Index letters: scalar `order`; jet axis lowercase `t` (`r`,`s` binomial); `K` = tangent stack ONLY; any new axis letter (e.g. hole-count) documented, no collision with W/K/C · naming_conventions.md §"Index letters"
- [ ] No "outer" for the down core (`down_tt_cores`); no "basis" for frame · CLAUDE.md gotchas; naming_conventions.md "frame not basis"
- [ ] `gn_*` names keep meaning Gauss–Newton — curvature must not silently enter `gn_hessian`/`gn_quadratic` (misnomer + breaks Cauchy steps); new names for the full model · fitting.py:356-387; optimizers.py:344-358, :424
- [ ] Frontend methods unprefixed; frontend free functions prefixed; new module (if any) placed in the module map / outside-the-matrix list · naming_conventions.md module map, exceptions
- [ ] Parameter names encode representation (`x`/`data`, `frame`/`frame_data`) · contributor/naming_rules.md

## 6. Signature style
- [ ] One arg per line; three aligned columns name·type·`# contract`; one return element per line (tuples expanded); `# len=d, elm_shape=K+C+(...)`; `HOST bool, static` for masks · contributor/signature_style.md mechanics, micro-grammar; reading_signatures.md
- [ ] Annotate what Python can express (Optional, Union incl. None); comment only the inexpressible; grouped alignment when type-length spread is large; no staircase · signature_style.md "Annotations vs comments", "Aligning"
- [ ] `contract(...)` call sites need no restating comment · signature_style.md "Within reason"
- [ ] Docstrings cite the local `t4s.pdf` equation numbers / external refs (Absil et al., Psenka–Boumal, Steihaug) — no `dev/` paths in shipped text · CLAUDE.md "cite it in the docstring"; HANDOFF cluster 5 ruling

## 7. Dispatch
- [ ] No `use_jax` threading; infer via `tree_contains_jax(inputs)` / `is_jax_ndarray` at the lowest level; `xnp, xmap, xscan = get_backend(is_uniform, use_jax)` · CLAUDE.md numpy-vs-jax dispatch; common.py:314-337
- [ ] Scan/map/while bodies module-level, closure-free, dispatch inside on ALL operands; statics recovered from shapes; `lru_cache` factories close over hashables only, never arrays · contributor/scan_body_principles.md §1-6
- [ ] `'...'` einsums; grouped contractions via `contract` (numpy greedy-pairwise path handled inside) · CLAUDE.md code style; contractions.py:97-109
- [ ] `xwhile` same cond/body for eager and jit; branch-free (`xnp.where`) Steihaug boundary step · common.py:340-358; optimizers.py:548-567
- [ ] jax-absent policy: `use_jit=True` warns + runs eager · fitting_and_optimization.md §4.5; common.jax_or_warn

## 8. Jet machinery
- [ ] Two-index jet (vector order t ≤ `order` × core-hole order ≤ 1) composes with the actual code: affine input jets (s∈{0,1}), banded recurrences (`_mu_jets_step`, `_sigma_banded_step`), full order-scans for eta/tilde (`_adj_tilde_step`), `binomial_combine_tensor` · sampling_derivatives.py:240-407, 958-1030, 1518-1602
- [ ] Reuse: the one-hole companions = existing `compute_{dxi,sigma,tau,deta}[_jets]` at the corewise frame with variations θ̂ (see pre-verified fact); only the adjoint-sweep companions are genuinely new · probing.py:336-523; sampling_derivatives.py:958-1050
- [ ] Layout consistent: order axis leading `(order+1,)+W+K+C+…` ragged; `(d,)+(order+1,)+…` uniform; frame jets `W+C`, variation jets `W+K+C` · sampling_derivatives.py:54-65, 271-277
- [ ] Standard recurrence form is the default; `_trs` dense twins only as reference/oracle · deferred_and_rejected.md "bidiagonal-trs ... standard form"; tests/test_jet_recurrence.py
- [ ] W-chunking: probe_derivatives assembly chunking threaded (`chunk_size`), memory model for the larger two-hole assembly stated; `estimate_chunk_size`/'auto' accounts for it or the gap is documented; chunking engages uniform+jax only, ≥3 chunks · docs/chunking.md; sampling_derivatives.py:1942-1987, 2063-2130
- [ ] Sharding: map-reduce over W via `shard_map` + psum still valid; no dynamic-slice-on-sharded-W regression · docs/chunking.md §Sharding

## 9. Geometry / SamplingKind / LocalModel protocols
- [ ] Extends the ACTUAL `Geometry` protocol (frame/base_point/stack_shape/precompute/project/retract/inner/point_norm_sq/point_tangent) — any new member is a breaking protocol change → CHANGELOG + release_notes upgrade note · optimizers.py:60-98; release_notes.md "custom backend geometry needs three more members"
- [ ] `precompute(frame) -> aux` slot semantics preserved: `None` for MANIFOLD/COREWISE today; SharedGeometry's `SharedFrameData`; if lift data rides there, shared + unshared both handled, `aux=None` fallback kept · geometry.py:191-216; shared_geometry.py:215-220; sharing_internals.md "The geometry precompute slot"
- [ ] `SamplingKind` new method(s) declared on the base with a helpful NotImplementedError + optional-capability flag (the `has_block_sumsq` pattern); uniform twins override with the `(v, sample, frame_data, sweep)` protocol shape · backend/fitting.py:305-413; uniform_fitting.py:17-21
- [ ] Acknowledges it WIDENS the open SamplingKind surface (standing open question) and asks Nick · dev/OPEN_QUESTION_extension_surface.md "What to watch"
- [ ] `LocalModel` new fields appended with defaults; `_flatten_local_model`/`_unflatten_local_model` updated (arrays as leaves via `partition_static`, statics aux); `Problem.local_model` builds them · optimizers.py:105-161, 190-201, 799-808
- [ ] Compile-once preserved: rebuilt LocalModel same treedef; `_cg_solve` jit cache key = model treedef + traced (tol, maxiter, Δ, …); static flags in `static_argnums` only if truly static · optimizers.py:589-599; tests/test_dispatch.py:556-598
- [ ] Frontend `GaussNewtonModel` twin: fields + pytree registration + all 7 construction sites (6 factories + `_uniform_model`) updated; stays bit-identical to the backend LocalModel · fitting.py:184-282, 454-488, 534-730, 748-754; test_fitting `test_matches_backend_local_model`
- [ ] Optimizer default (GN, plain CG) bitwise unchanged when curvature off — the slow-tier O2 optimizer-trajectory oracle sweep must stay green unchanged · tests/test_oracle_sweep.py:106-119
- [ ] `gradient_descent`/`mc_sgd`/`adam` Cauchy/first-order paths keep the GN quadratic · DP §4.3; optimizers.py:344-358, 424
- [ ] Frontend `optimizers.newton_cg` kwargs pass-through + `_setup` kind construction (string kinds only today) consistent · optimizers.py:174-229, 318-439

## 10. Shared factors
- [ ] Shared lift = the companion's clipped solve (`fv_tied_ambient_directions` / `_tied_solve`) — not a re-derived pinv; output tied-projected (`geom.project(..., aux=companion)`) · sharing.py:644-686; DP §4.3 "Shared factors"
- [ ] N_shared uses the TIED gradient (`grad_shared f`), consistent with the tied embedding · DP §4.3; sharing.md "Optimizing over shared factors"
- [ ] Manifold (tilted solve) vs corewise (mean) formulas not swapped · sharing_internals.md "One principle, two formulas"
- [ ] Companion via `precompute` once per local model (no per-matvec rebuild — the regularizer-leak lesson) · precompute_and_caching.md "Fixed — the regularizer leak"
- [ ] Padded-restart gated channel: clipped pinv returns 0 there (escape via TT variations) — damping must not inject spurious curvature · sharing_internals.md "Padded restarts"
- [ ] Uniform shared mirror masks the frame first; cross-layer comparisons gauge-invariant from tied, nonzero, shared-minimal starts · sharing_internals.md "uniform mirror" lessons 1-2

## 11. Regularizer, ω, g0
- [ ] IdentityRegularizer: H_R = λΠ is already EXACT Riemannian (Weingarten term 0); the L2 correction must use the MISFIT-only gradient — `LocalModel.gradient` includes g_R = λ v_X, so using it in K_{grad f} adds a spurious −Lᵀ K_{λX} L · fitting_internals.md §Regularization ("H_R = λI is the exact Riemannian Hessian"); optimizers.py:134-140
- [ ] Regularizer protocol (value/gradient/hessian/quadratic, `aux=`) unchanged or extended with a stated story for future curvature-carrying regularizers (GK `SingularValueRegularizer`) · regularization.py:54-68; fitting_internals.md "What's deferred"
- [ ] ω: K_{Aᵀr} uses ω²r (routed through the kind's `_apply_weight(r, 2)`, residual stays RAW); ω host-numpy static · fitting_and_optimization.md §4.6; fitting.py:199-285; backend/fitting.py:553-555
- [ ] `_ScaledRegularizer` / minibatch scaling untouched (stochastic optimizers unaffected) · regularization.py:98-118
- [ ] g0 references (`g0norm_newton`/`g0norm_cg`, chained fallback) preserved; forcing-term change η = O(‖g‖) only when curvature on, stated relative to which g0; no conflict with the queued zero-reference g0 default plan · optimizers.py:646-728; dev/zero_reference_g0_plan.md §4-5

## 12. Tests
- [ ] Numerical correctness numpy-only vs dense ground truth; jax only in `test_dispatch` (jit each new op; stray np on tracer raises) + compile-once / not-stale tests for new traced values (Δ, ε) · CLAUDE.md Verification; testing_strategy.md; test_dispatch.py:556-598
- [ ] Oracles: polynomial-exact route-1 FD and dense Lagrangian Hessian (numpy); FD along `tv_retract` (2nd order ⇒ exact up to FD O(h²), valid only at numerically full-rank MINIMAL frames — not at padded restarts / non-minimal shared frames); jax-jvp route 2 only in a jax-gated test · DP §4.6; numerical_contracts.md retract caveat; test_oracle_sweep.py:30-34
- [ ] Symmetry of full hvp; vertical (pure-gauge) directions in its null space (generators `tests/backend/test_tv_operations.py:121-137`) · DP §4.6
- [ ] Uniform: dense-vs-ragged per element over `_CONFIGS`; EXACT output masks (non-circular); garbage-padded-input robustness; isolated rejection tests for any new structural check (+ mutation check) · testing_strategy.md "Checklist for a new uniform op"
- [ ] Degenerate/edge cases: d = 1; zero start x0 = 0 (S=T=0 everywhere ⇒ graceful GN fallback, no NaN); zero-padded restart; non-minimal frame; rank-1 bonds · HANDOFF rulings "d = 1 supported degenerate"; fitting_and_optimization.md §2.4 zero start
- [ ] Convergence-rate test (none exists): seeded, cheap enough for always-on or placed in the slow tier deliberately; asserts ρ_tr → 1 and q-quadratic tail with GN linear-tail control · DP §4.6; HANDOFF suite-time budget
- [ ] New `contract` strings recorded in `HISTORICAL` (`tests/test_contractions_interpreter.py:70`), supplement decided by hand · contributor/contractions_internals.md §9; batching_internals.md
- [ ] Per-test seeding; subTest params primitives (xdist DumpError); order-independent under `-n auto` · HANDOFF Phase D slice 8, CI note
- [ ] Use `allclose`/`np.allclose`, never `==` on runtime classes · numerical_contracts.md "Equality is explicit"
- [ ] Oracle sweep tier: decide whether curvature joins the O2 optimizer sweep (always-on subset vs slow REQUIRED tier) · tests/test_oracle_sweep.py

## 13. Docs / doctests / routing
- [ ] User doc: `docs/fitting_and_optimization.md` new section (curvature, TR, damping, when it helps: |1−ρ_tr| tail) + pointers; contributor record in `docs/contributor/` (fitting_internals or a new note added to `contributor_guide.rst` toctree) · CLAUDE.md "Where things live"
- [ ] DP promoted to `docs/contributor/` when settled; `dev/` plan → `dev/archive/` dated with banner; `dev/curvature_oracles/` ported to `tests/`; `dev/HANDOFF.md` + CLAUDE.md Current-state refreshed · CLAUDE.md routing + handoff ritual; dev/curvature_oracles/README.md
- [ ] Rejected alternatives (L3 AD-through-T3-SVD; Weingarten projector-derivative form; NRO AD) → `docs/contributor/deferred_and_rejected.md`; measured numbers → `verification_records.md` · deferred_and_rejected.md purpose
- [ ] Doctests are examples-first, seeded, print structure/bools not raw floats, RUN-AND-PASTE; page examples `>>>` in python fences with blank line before closing fence; CI on both numpy generations · contributor/doctest_style.md
- [ ] Any change to `format_newton_iter`/NewtonInfo header re-runs + re-pastes its doctests · optimizer_display.py:160-187
- [ ] `sphinx -W` zero warnings; new backend module added to `api_reference.rst` "backend surface" + naming module map · CLAUDE.md Current state; docs/api_reference.rst:97-135
- [ ] CHANGELOG `[Unreleased]` Added (+ `Changed — breaking` + `release_notes.md` "Upgrading" for any protocol/default change) · CHANGELOG format; release_notes.md
- [ ] `docs/numerical_contracts.md` table rows for any new op (precondition/caveat) · numerical_contracts.md
- [ ] `docs/batching_and_stacking.md`/`naming_conventions.md` updated if a new axis letter or block convention appears · batching_internals.md "Maintenance note"

## 14. Math-to-code map
- [ ] Lift orientation per pre-verified facts (S_i⁻ᵀ on Tucker, T_i⁻¹ on the 3rd TT leg, Ġ_{d−1}=H_{d−1}); Lᵀ the Frobenius adjoint · DP §4.1
- [ ] S_i, T_i derived exactly (zipper; `S_iᵀ = ⟨O_i,H_i⟩`), no fresh SVD for the factors themselves (sign/degenerate hazards); damped pinv via an SVD of the small S/T is fine · sharing.py:491-568; sharing_internals.md decision 2
- [ ] Raw gauge θ = `base_point(frame)` = (U, P) with left-canonical P; K_Z evaluated at the corewise frame of θ · geometry.py:183-189
- [ ] hvp = Π𝒥ᵀω²𝒥Πp + Π Lᵀ(K_{Aᵀr} − K_{grad_misfit})L Π p + λΠp — Π on input AND output so H maps gauged→gauged and the coordinate dot stays HS (CG Krylov space gauged) · DP §4.2; docs/frame_variations.md "Gauge conditions"; DP §1.3
- [ ] Sign: + K_{Aᵀr} with r = S(x) − y (LocalModel.residual), − K_{grad f} · DP §2.1(b); optimizers.py:199
- [ ] D²Φ[θ̂,θ̂] = Σ_{a≠b} (factor 2 per unordered pair) — K_Z normalisation consistent with the quadratic form ⟨p, H p⟩ · DP §2.1(b)
- [ ] Vertical directions in L2's null space (claimed; must be TESTED in coordinates, incl. after Π) · DP §3 notes
- [ ] Retraction-FD oracle exactness claim scoped (second order proven at minimal full-rank frames; not first-order accurate at rank-deficient frames) · DP §2.3; numerical_contracts.md retract row
- [ ] Predicted reduction/ρ_tr consistent with the model actually solved (TR) · DP §4.3; optimizers.py:750-751
- [ ] Damped lift `σ/(σ²+ε²)`: ε scale rule stated (per-mode/per-bond?), continuous to GN as ε→∞, exact as ε→0; ε=0 path guarded at σ=0 · DP §4.4
- [ ] COREWISE geometry: K_{Aᵀr} with no lift / no K_{grad f}; gauge-singular ⇒ TR/damping only · DP §4.3

## 15. Anything else Nick would send back
- [ ] Equality: new runtime array-carrying frontend classes are `ExplicitEquality` (`==` raises); aux classes value-hashed · numerical_contracts.md "Equality is explicit"; common.py:523-558
- [ ] Frozen dataclasses throughout (minimal: derive, don't store redundant fields) · CLAUDE.md "Prefer minimal dataclasses"
- [ ] NewtonInfo additions (Δ, curvature flag, neg-curvature hit, σ_min(S/T)) flow into `stats['history']` and the display; `cg_truncated` semantics under Steihaug (boundary vs negative curvature) redefined explicitly · optimizers.py:602-643; optimizer_display.py:137-139
- [ ] Callback stays host-side; composes with `use_jit` (inner CG only jitted) · optimizers.py:693-697; deferred_and_rejected.md "Jitting the outer Newton loop"
- [ ] Line-search interplay: Armijo-along-retraction fallback vs TR acceptance; `on_line_search_failure` semantics preserved · optimizers.py:683-692, 738-769
- [ ] Default-off first (API choice `curvature=`/Hessian enum is a decision for Nick); forcing-term change opt-in · DP §5
- [ ] Release gates: full suite (`-n auto`), module + page doctests on both numpy generations, `sphinx -W`, slow-tier oracle gate before any release · CLAUDE.md Verification; HANDOFF release ritual
- [ ] No auto-formatter near nonstandard code style; uppercase core names kept · CLAUDE.md Code style
- [ ] Performance claims (≈2–2.5× GN matvec; m-independent K_{grad f}) measured, regime-qualified · fitting_and_optimization.md §4.7 tone; DP §4.5
