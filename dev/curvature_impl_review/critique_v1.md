# Critique of `plan_v1.md` (round 1)

Critic review of the exact-Riemannian-Hessian plan. Repo `/home/claude/t3toolbox` @ `920d776` (read-only).
Every repo citation below was checked by grep, read or a read-only run. "DP" means
`dev/riemannian_newton_curvature_plan.md`; "SCR" means the `plan_review` scratch directory.

**Bottom line.** The math and the code mapping are sound, and the reuse claims check out:

- the lift orientation is right;
- the one-sided-inverse argument for non-minimal frames is right;
- the block-sum oracle is exact;
- X3, X4 and X7 are real.

No invented names: every existing function, file:line and doc section the plan cites exists.

My restart experiment (§0) **supports** the plan's riskiest choice (I-8): the trust region with ε = 1e-6‖X‖
converges in 4–5× fewer Newton steps than GN from zero-padded continuation restarts. Damping more hurts.

Three MAJOR findings block agreement:

| # | finding |
|---|---|
| M1 | the default exact-mode display crashes |
| M2 | the misfit transpose is duplicated, justified by a rationale that contradicts a contributor doc |
| M3 | the frontend pytree would trace the uniform host masks the curvature companion carries |

The rest are MINOR and NIT.

---

## 0. What I verified (beyond the plan's own scripts)

**The writer's scripts.** I re-ran `checks/c1`–`c13` read-only. All the numbers the plan quotes reproduce, with two
exceptions (m1):

- `c1` prints `|T-Z|=8.4e+05` at the last slot;
- `c2`'s lift block is broken and unseeded. It printed `rel err 3.92e+18` on one run and `2.44e+17` on another,
  because `t3_corewise_randn` draws from the unseeded global RNG.

My own scripts are in SCR and were run with `python -I -B`.

**E1 — `check_small_sigma.py`.** A minimal d = 4 point with one bond singular value δ (spectrum `[1, .5, δ]`,
‖X‖ = 1.12), a 30%-noise residual, and one unit tangent:

| δ | GN | curv (ε=0) | curv (ε=1e-6‖X‖) | curv (ε=1e-2‖X‖) | exact vs FD along `tv_retract` | exact-model err at t = δ / 10δ / 100δ | GN-model err |
|---|---|---|---|---|---|---|---|
| 1e-2 | 261.4 | +35.0 | +35.0 | +15.2 | 296.34 / 296.34 | 5e-3 / 0.23 / 0.47 | 3e-2 / 0.12 / 0.29 |
| 1e-3 | 212.0 | +695.7 | +695.7 | +14.0 | 907.67 / 907.66 | 3e-2 / 1.1 / 7.1 | 6e-2 / 0.12 / 0.28 |
| 1e-4 | 285.5 | +2942 | +2942 | −10.0 | 3227.5 / 3227.4 | 1e-3 / 9e-2 / 1.8 | 2e-2 / 7e-2 / 6e-2 |

- The math is right: exact matches FD to 5–6 digits even at δ = 1e-4.
- The exact model's validity radius is about σ.
- **E1b** (`check_small_sigma_signs.py`): at δ = 1e-3, 4 of 12 random directions have net negative curvature, so the
  exact Hessian is indefinite there.

**E2 — `check_restart_tr.py`, `check_restart_tr2.py`.** This is the decisive one.

- **Method.** I ran an exact-mode Steihaug–TR Newton loop with the plan's TR rules, Δ0 and forcing powers. The
  exact hvp uses library kernels plus the block-sum oracle, the same math as plan C.9. The GN parts, gradient,
  retraction and objective are the library's backend `LocalModel`.
- **Problem.** d = 3, `(6,6,6)`, apply, W = 400. Level 1 is a converged rank-(2,2,2)/(1,2,2,1) fit; level 2 is its
  zero-padded `resize` to (3,3,3)/(1,3,3,1). Outer = Newton steps, including rejected ones; hvps = total
  Hessian-vector products.

| scenario | GN+Armijo (library) | GN+TR | exact+TR, ε = 1e-6‖X‖ (plan default) | ε = 0 | ε = 1e-2‖X‖ | ε = Δ |
|---|---|---|---|---|---|---|
| target rank 4, 5% noise | 54 steps | 52 / 352 hvps | **13 / 69 hvps, 0 rejected** | 13 / 69 | 17 / 173 | 56 / 634, 9 rejected, **not converged** |
| target rank 2 (level over-ranked), 5% noise | 80, not converged | 80, not converged | **18 / 117, 1 rejected** | — | 80, not converged | — |
| target rank 4, 30% noise | 32 | 42, 10 rejected | **11 / 53, 1 rejected** | — | 11 / 53 | — |

What E2 shows:

- **The new σ jump to about Δ on the first step.** min σ goes from 0 to 14, with Δ = 28. So the validity radius
  from E1 tracks Δ automatically, and the zero slots are already handled at ε = 0 (f(0) = 0).
- **Steihaug handled negative-curvature stops** twice per run, with no collapse.
- **Larger ε or Δ-coupled ε is harmful.** It kills the tail curvature, which is exactly J-2. That is evidence
  *for* I-8's small default. (I had considered proposing a Δ-coupled ε; it is measured harmful, so I do not.)
- **A degenerate case.** A noise-free over-ranked level (f ≈ 1e-22 at the start, so g₀ ≈ 1e-11) ends in the TR
  collapse stop for both GN+TR and exact. That is the correct outcome; it must just be reported as such (m17).

**E3 — `check_float32_cancellation.py`.** The curvature quadratic form in float32 vs float64, at the E1 point:

| δ | relative error vs the curvature | relative error vs GN |
|---|---|---|
| 1e-2 | 2e-6 | 3e-7 |
| 1e-3 | 5e-6 | 2e-5 |
| 1e-4 | 2e-5 | 2e-4 |

The `K_{Aᵀr}` / `K_{grad f}` cancellation is benign in float32; plan J-5 is fine as written.

**E4 — `check_d1_normal.py`.** At d = 1 (n = 1 and n = 3), `‖Aᵀr − grad f‖/‖Aᵀr‖ = 5e-17` and `3e-16`. The d = 1
manifold is open, so N ≡ 0.

**E5 — `check_full_rank_N0.py`.** On maximal-rank structures `(3,3,3)` and `(2,3,4)`, `‖K_N L v‖/‖K_{Aᵀr} L v‖ =
7e-16` and `8e-16`. Each term is O(1), so this is a non-circular check of the sign and normalisation of the T3–T3
term against the sampling term.

**E6 — uniform d = 1 today.** At d = 1:

- uniform `MANIFOLD` `newton_cg` and `gradient_descent` **already raise** in
  `utv_retract → utv_to_ut3` (`utv_operations.py:347` → `:305`);
- uniform `COREWISE` and ragged `MANIFOLD` run;
- the uniform `MANIFOLD` *model* plus gradient builds.

**Repo facts behind the findings:**

- **The `ls` display segment crashes on `None`.** `optimizer_display.py:140-141` formats `ls_steps` with `%d`
  unconditionally, and `'%d' % None` raises `TypeError`.
- **Boundary stops would count as truncations.** `optimizers.py:733` sets
  `cg_truncated = (not cg_converged) and (not cg_ok)`.
- **The frontend pytree does not partition statics.** `fitting.py:748-754` registers raw children, with no
  `partition_static`; the backend `LocalModel` does (`optimizers.py:800`).
- **Import graph.** `geometry.py:45-46` imports `fv_operations` and `ufv_operations`. Only the frontend
  `tucker_tensor_train.py` and `uniform_tucker_tensor_train.py` import `t3_linalg`.
- **`_cg_solve` consumers.** Only `optimizers.py:730`, `tests/backend/test_optimizers.py:218,223` and
  `tests/test_dispatch.py:595` use it; nothing in `examples/` or `tests/oracle_sweeps/`. The blast-radius list is
  complete.

**Repo-hygiene note.** `t3toolbox/__pycache__` and `t3toolbox/backend/__pycache__` (timestamp 19:14:43) come from my
first non-`-B` read-only run. They are gitignored (`.gitignore:2`). I left them, because deleting is also a write.
All later runs used `-B`, and the implementer's runs should too.

---

## 1. Razor and parameters-as-fields

- **OK.** All math lands in backend functions on raw `.data`. `LocalModel.hvp` composes lift → K → Lᵀ → Π, so a raw
  user is not one step short.
- **OK.** `LiftFactors` is frozen with `eq=False` and is a pytree (the `SharedFrameData` precedent,
  `sharing.py:445`, `:1081-1091`).
- **OK.** The geometry and kind gain methods only, no fields, so hash/eq are unchanged.
- **M2 — MAJOR: the misfit gradient is transposed twice per Newton step, and the rationale contradicts a house
  rule.**
  - In exact mode, `Problem.local_model` computes `g_misfit = geom.project(kind.transpose(r, …))` (plan C.9 step 2).
    Then `newton_cg` reads `lm.gradient`, which runs the same W-scaling transpose again (`optimizers.py:134-140`,
    `:713`).
  - That costs about 0.5 GN matvec per Newton step. With E2's CG counts (1–4 exact hvps per step early) it is
    roughly 5–10% of an early step; it is worst for probe and probe_derivatives.
  - The stated reason, "caching rejected because the model is a jit argument", contradicts
    `docs/contributor/precompute_and_caching.md:46-53`:
    - a per-frame item lives as "a **leaf** beside `sweep`", and being a jit argument is exactly why leaves work;
    - "once the channel exists, hoisting one more item is nearly free — do it". The curvature leaf is a channel
      the plan creates anyway.
  - **Fix.** Carry `g_misfit` in the curvature companion. In exact mode `LocalModel.gradient` returns
    `g_misfit (+ g_R)` from it, bit-identical. Equivalently, set `g_misfit = lm.gradient − g_R`, with
    `g_R = λ·point_tangent`, which is cheap. Either way there is one transpose per step.
  - Audit the frontend `GaussNewtonModel.gradient` too. It is a `cached_property`, so the same seam applies
    (`precompute_and_caching.md` checklist item 4).
- **m18 — MINOR: the shared geometry builds the centres twice.** `fv_lift_factors` re-runs the zipper and the
  centres, which `geometry.precompute` already holds in `SharedFrameData.centers` (`sharing.py:547-548`).
  - **Fix:** take them from `aux` when present.

## 2. Ragged vs uniform

- **OK.** The polymorphic-kernel-plus-wrapper split follows `uniform_polymorphism.md`. Host masks use `np`. The
  `'d...'` maps and `xscan` avoid the unroll trap. Masked Z feeds the T3–T3 kernel. Varying-rank C works per
  element. Uniform `T` is square-bond `(r,r)` (verified).
- **m9 — MINOR: the uniform lift masks are derived from the wrong mask set.**
  - C.10 gives `utv_corewise_lift` the output masks `ufv_variation_masks((tk, tk, tt, tt))`, built from the plain
    point masks.
  - The raw θ is `base_point(frame) = (frame[0], frame[2])`, whose masks are the frame's `(um, lm)`
    (`ufv_conversions.py:115-118`). The two sets coincide only at a minimal point.
  - The fitting path is minimal (`uniform_minimal`), but the backend `utv_*` surface must be right off that path.
  - **Fix:** use `ufv_variation_masks((um, um, lm, lm))` from `frame_data[5]`.
  - `lift_sigma_min` must take its minimum over **real** slots only. Otherwise every padded or varying-rank stack
    reports 0.
- **n7 — NIT: say where the uniform lift output is masked.** A plain-SVD pinv can leak roundoff into padded
  coordinates. That is harmless under "consumers mask on entry", but say so, or re-mask the output.
- **m21** (uniform d = 1) is under §15.

## 3. Safe mode

- **OK.** No new enforced preconditions; `_require_at_frame` on the new methods. The three caveats are correctly
  caveats under `numerical_contracts.md` §"The classification rule", and lift invertibility is never a
  precondition.
- **m3b — MINOR: one caveat is missing.** Record in `numerical_contracts.md` that on `MANIFOLD` the exact Hessian
  equals GN at d = 1, and on any open full-rank format, where N ≡ 0 (E4, E5).

## 4. Batching (W + K + C)

- **OK.** Per-frame data carries C only, the residual carries W+C, and K enters only through θ̂. K placement follows
  the `_sigma_step` precedent (`probing.py:363-377`). `curvature_quadratic` keeps K+C. The Steihaug norms are per
  element.
- **n8 — NIT: the HISTORICAL table uses an undefined notation.** It lists `'WCo,KCio->WKCi' : W|C`, but a
  HISTORICAL row maps to **one** letter or `''` (`tests/test_contractions_interpreter.py:70-90`).
  - **Fix:** state the chosen letter. The analog `'WCi,KCio->WKCo'` records `'C'` (`:90`).

## 5. Naming

- **m8 — MINOR: module placement.** `t3_inner_product_corewise_transpose`, `t3_precompute_inner_product_sweep` and
  `t3_inner_product_corewise_hessian[_from_sweep]` are placed in `tv_operations.py`.
  - That module has no `t3_` functions today, and the module map (`naming_conventions.md` §"The module map") puts
  the t3 family × linalg in `t3_linalg.py`, next to `t3_inner_product` (`:176`).
  - There is no cycle: only the two frontend modules import `t3_linalg`.
  - The uniform twins go in `ut3_linalg.py`.
- **n2 — NIT: constant name.** `LIFT_DAMPING_DEFAULT` should be `DEFAULT_LIFT_DAMPING`, following the precedents
  `DEFAULT_CHUNK_SIZE` (`sampling_derivatives.py:28`) and `DEFAULT_RTOL_NUMPY/JAX` (`safety.py:89-90`).
- **OK.** The `corewise` marker, the `has_corewise_hessian` flag (the `has_block_sumsq` pattern, `fitting.py:326`),
  `gn_*` keeping its GN meaning (X1), `quadratic`/`curvature_*`, and the chain-helper exception for
  `tt_zipper_tangent_*`.

## 6. Signature style

- **n1 — NIT: argument order differs between one-shot and `_from_sweep`.** Compare
  `t3_apply_corewise_hessian(c, ww, core_pair, core_directions)` with
  `_from_sweep(core_directions, c, ww, core_pair, frame_sweep)`, and the inner-product pair similarly.
  - **Fix:** one rule. Recommended: the one-shot order with the sweep appended, the `tv_apply_transpose` /
  `tv_apply_transpose_from_sweep` precedent (`apply.py:172`, `:264`).
- **n5 — NIT: some signatures are one-liners.** The Geometry members and the one-shot wrappers are written on one
  line. Implementation must expand them per `signature_style.md`.

## 7. Dispatch

- **OK.** Dispatch is inferred everywhere. Bodies are module-level. The radius rides in the CG state
  (`scan_body_principles.md` §5). `_jitted(_cg_core, (4,))` keeps `use_jit` as the only static argument.
- **m7 — MINOR: import cycle.**
  - `fv_lift_factors` computes ‖X‖ via `t3_left_orthogonal_norm_sq` (`geometry.py:73-85`), and the uniform twin via
    `ufv_base_point_tangent` (`geometry.py:105-123`). But `geometry.py:45-46` imports `fv_operations` and
    `ufv_operations`, so this creates a cycle.
  - It is also the wrong layering direction.
  - **Fix:** compute `‖X‖² = Σ last-left-core²` inline (the frame is left-orthogonal), or move the helper down into
    `fv_operations` or `t3_linalg` and re-export it.
- **n6 — NIT: carry over the `_src_step` justification.** The inner `_src_step` closure in the new
  `_adj_tilde_tangent_step` must carry the existing exception reasoning (`sampling_derivatives.py:1549-1550`: it
  captures per-core scan operands, never a changing Python value).

## 8. Jet machinery

- **OK.** The two-index composition is right: a second Leibniz index with s ≤ 1, the same `trs` wiring, and dotted
  input jets affine in t. Using companion arrays (I-10) is right by `batching_and_stacking.md` §3.
- **OK.** The reuse of `compute_sigma_jets` and `build_input_jets`.
- **OK.** The finding that `_adj_tilde_step` cannot take a K-carrying `Ġ` is right: it infers
  `nf = len(P.shape[:-3])` from the core (`sampling_derivatives.py:1530`).
- **OK.** S5b's memory measurement plus the conditional `estimate_chunk_size` extension. The chunking scope is
  stated.

## 9. Geometry / SamplingKind / LocalModel protocols

- **M3 — MAJOR: the frontend pytree would trace uniform host masks.**
  - C.13 adds `GaussNewtonModel.curvature` to the registration at `fitting.py:748-754` "as a leaf".
  - That registration passes **raw children**. Unlike the backend `LocalModel`, it does not use `partition_static`
    (`optimizers.py:800`).
  - On the uniform layer the companion holds `Z = utv_to_ut3(...)`, i.e. `(tk_sc, tt_sc, shape, (tucker_mask,
    tt_mask))` with host-numpy masks (`utv_operations.py:322`). C.10 then masks Z at use.
  - Under `jax.jit(lambda m, p: m.hessian(p))`, the documented roll-your-own pattern that
    `tests/test_dispatch.py:515` pins for GN, those masks become tracers. `ut3_apply_masks` →
    `require_concrete_masks` then raises (`ut3_masking.py:74`). This is the failure `common.py:704-711` documents
    as the reason `partition_static` exists.
  - Today's frontend children (UT3Frame, the uniform sweep's bare supercores, `SharedFrameData`) carry no masks,
    which is why GN is safe.
  - The plan's dispatch tests jit only the **backend** exact `LocalModel.hvp`, so this would ship unnoticed.
  - **Fix:**
    - Give the companion a pytree registration that keeps statics in aux (`partition_static` in its flatten, or
      masks-as-aux like `UT3Frame`). Alternatively, store only Z's supercores and rebuild its masks on the host
      from the geometry's masks.
    - Add `test_jit_uniform_exact_gauss_newton_model`, mirroring `:515`: one trace across rebuilt models.
- **m10 — MINOR: one protocol member too many.**
  - `corewise_inner` is redundant: `curvature_quadratic(p) = ⟨LΠp, K_N LΠp⟩_raw = geom.inner(Πp, Lᵀ K_N LΠp)`,
    because Lᵀ is the Frobenius adjoint.
  - Lᵀ is m-independent, O(d[n²N + nr³]).
  - Each new protocol member is an upgrade-note item for custom geometries (`release_notes.md`), and this one also
    forces a third uniform mask set.
  - **Fix:** drop the member.
- **m11 — MINOR: drop the reserved `None` slot.** `curvature = (geom_curv, None)` reserves a slot for a future
  I-11 flip. That is YAGNI under CLAUDE.md "Prefer minimal dataclasses". Store the companion directly; a later flip
  appends a defaulted field.
- **OK.** `SamplingKind.corewise_hessian` plus the capability flag, the geometry-agnostic kinds, the
  `Problem.local_model` kwargs and the backend pytree. Compile-once holds: the GN and exact treedefs differ but are
  each stable.

## 10. Shared factors

- **OK.** The tied lift through the stacked-S SVD with the shared filter; `N_sh` from the tied misfit gradient; the
  manifold and corewise formulas not swapped (`sharing_internals.md` §"One principle, two formulas"); the
  padded-restart gating (`sharing_internals.md:98`); the honest "shared retraction not proven second order" oracle
  caveat; T12 via the dense Lagrangian of Φ∘T.
- **m18** (§1): reuse `SharedFrameData.centers`.

## 11. Regularizer, ω, g0

- **OK.** The curvature uses the **misfit** gradient (X11). `H_R = λΠ` is exact, because λX is tangent, so its
  normal part is zero. ω² enters via `_apply_weight(r, 2)` (`fitting.py:405`, `:554`). The g0 chain is untouched,
  and the plan composes with `dev/zero_reference_g0_plan.md` §4.
- **m5 — MINOR: the `cg_forcing_power` default change has unlisted doc consumers.**
  - `docs/fitting_and_optimization.md:470` says "(default `0.5`)", as do `optimizers.py:656`, `:677` and the
    frontend `t3toolbox/optimizers.py:341`.
  - **Fix:** list them in G, plus a CHANGELOG "Changed" line.
- **m19 — MINOR: no story for curvature-carrying regularizers.**
  - The plan relies on ∇ρ being tangent, which holds for ridge and weight decay only.
  - A future Grasedyck–Kramer `SingularValueRegularizer` (`fitting_internals.md` §"What's deferred") has a normal
    component and would need a Weingarten term.
  - **Fix:** record the assumption and the hook it would need in `deferred_and_rejected.md`.

## 12. Tests

- **m15 — MINOR: missing tests.**
  - **(a) Open-manifold oracle.** On a maximal-rank structure and on ragged d = 1, assert `curvature_hvp(p) ≈ 0`
    relative to `‖Lᵀ K_{Aᵀr} L Πp‖` (E4, E5: 7e-16). It is a free, non-circular end-to-end check of the T3–T3
    sign and normalisation.
  - **(b) ε neither stale nor recompiling.** Two `lift_damping` values give one compile and distinct results
    (mirror `test_cg_tolerance_is_not_stale`, `test_dispatch.py:581`).
  - **(c) Exact mode with `verbose=True`** (M1).
  - **(d) Frontend uniform exact jit** (M3).
  - **(e) The restart regime, pinned.** An exact-mode fit from a zero-padded `resize` of a converged lower-rank fit
    reaches `gtol` with no TR collapse and few rejections (E2's first row is a ready template: 13 steps,
    0 rejections). Ragged plus one uniform case; slow tier if it costs more than ~3 s.
- **m6 — MINOR: the oracle sweeps' edit policy.**
  - `tests/oracle_sweeps/README.md` says they are "Verbatim copies … keep these byte-close to the archived
    originals — the only local change is `sweep_optimizers.main`'s `case_filter` hook".
  - Plan F adds exact rows to those files.
  - **Fix:** a new module, e.g. `tests/oracle_sweeps/sweep_exact.py`, driven from new classes in
    `tests/test_oracle_sweep.py` in both tiers. Keep the GN rows untouched; they are the regression guard.
- **m16 — MINOR: T16 is fragile as specified.**
  - `log‖g_{k+1}‖/log‖g_k‖ ≥ 1.6 over the last two steps` is meaningless without normalising by ‖g₀‖, and the last
    step hits the noise or float64 floor.
  - E2's exact run shows a clean quadratic tail: ‖g‖/‖g₀‖ = 2.6e-2 → 5.9e-4 → 4.4e-7, then 3e-13.
  - **Fix:** use ‖g‖/‖g₀‖. Require at least 3 points inside a window such as [1e-1, 1e-10], and assert
    `‖g_{k+1}‖ ≤ C‖g_k‖²` with a stated C.
- **m1 (also §14) — do not port broken checks.** Do not port `c2`'s lift block or its unseeded RNG into the test
  helpers (per-test seeding under `-n auto`).
- **OK.** T1–T18 cover the right axes:
  - the stack matrix;
  - minimal and both slack kinds;
  - padded and zero points;
  - the non-circular uniform masks plus garbage padding plus the mutation check (`testing_strategy.md:143`);
  - numerics numpy-only, and a jax-gated jvp oracle.

## 13. Docs, doctests, CHANGELOG routing

- **m14 — MINOR: doc gaps.**
  - **§4.2 needs a qualifier.** `docs/fitting_and_optimization.md` §4.2 says "the only approximation in a Newton-CG
    fit is the manifold linearization itself", which is the term this feature adds. Add "exact on the affine
    tangent space; `hessian='exact'` adds the manifold's curvature (§4.12)".
  - **Stale `_cg_solve` prose.** Update the signature text at `docs/contributor/scan_body_principles.md:190` and
    `parameters_not_closures.md:111` ("a plain function of `(local_model, rhs, tol, maxiter)`").
  - **The open question.** Append to `dev/OPEN_QUESTION_extension_surface.md` §"What to watch" (`:131`). Do not
    archive it.
  - **CHANGELOG.** It has no `[Unreleased]` header (the top is `[2026.2.0]`). Create it in S1 and add lines **per
    slice**.
  - **Record the evidence.** Put E2's numbers into `verification_records.md`, alongside S0.
- **OK.** `curvature_internals.md` in the contributor toctree, the `deferred_and_rejected.md` entries, run and
  pasted doctests that print booleans and structure, and the dev archive plan.

## 14. Math-to-code map

- **OK, verified.**
  - The lift orientation (`U̇ = (Sᵀ)⁺V`, `Ġ = H ×₃ T⁺`, `Ġ_{d−1} = H_{d−1}`).
  - Lᵀ as the Frobenius adjoint.
  - The one-sided inverses (Sᵀ wide, T tall on every `t3_orthogonal_representations` frame).
  - θ = (U, P).
  - The `D²Φ = Σ_{a≠b}` normalisation (F6, E1).
  - The signs (E5).
  - The lift-independence argument and the corrected vertical-space statements (X7, F8).
- **m1 — MINOR: the facts table overstates two results.**
  - **F1** says "all ≤ 3e-12", but `c1` prints `|T-Z|=8.4e+05` (and `3.7e+04` non-minimal) at `i = d−1`. That
    comparison is not an identity the lift uses; the claim should say `i ≤ d−2`.
  - **F5** cites `c2`, whose lift section is broken and unseeded. Only its block-sum, symmetry and one-hole lines
    support F5.
  - **Fix:** correct both rows.

## 15. Everything else

- **M1 — MAJOR: exact mode plus `verbose=True` crashes.**
  - C.9 sets `ls_steps = ls_failed = None` in TR mode, which is the exact default (I-6).
  - C.14 only **appends** a TR segment. The existing `"ls %d%s α %.2e …" % (info.ls_steps, …)`
    (`optimizer_display.py:140-141`) raises `TypeError: %d format: a real number is required, not NoneType`.
  - The frontend `verbose=True` reaches it through `make_newton_display`.
  - **Fix:**
    - In TR mode, replace the `ls …` segment with `TR Δ %.2e ✓/✗`.
    - Guard κ = pKp/(pHp − pKp): the denominator can be 0.0, which raises `ZeroDivisionError` on Python floats.
    - Add an exact-mode `_format_header` test in `tests/backend/test_optimizer_display.py` and a run-and-pasted
      doctest.
- **m20 — MINOR (I-8): make the S0 gate explicit, and record E2.**
  - E2 supports the provisional `1e-6`: ε = 0 and ε = 1e-6‖X‖ are identical and best; 1e-2 hurts; Δ-coupled fails.
  - It is one apply problem at d = 3, so S0 still decides the default. The plan should say what ships if S0 is not
    run before S6. Recommended: ship the measured E2 default and record it.
  - Keep ε a traced leaf (m15b).
  - The plan's own "phase-in" language (A, C.12 caveat 1, G `rank_continuation.md`) should drop the implication
    that damping is what makes restarts safe. Per E2, the f(0) = 0 clip handles the exact-zero slots, the first
    step lifts the new σ to about Δ, and the TR does the rest.
- **m21 — MINOR: uniform d = 1 (I-12).**
  - Raising mirrors an existing gap: uniform `MANIFOLD` d = 1 cannot be optimized today (E6), so no optimizer path
    regresses.
  - But two cases should not raise:
    1. **Uniform `COREWISE` d = 1** runs today, and its exact Hessian (`GN + K_{Aᵀr}`) needs no T3–T3 and no
       `utv_to_ut3`.
    2. **On `MANIFOLD` at any d = 1** (and on open formats) N ≡ 0 (E4), so exact = GN. The model can set
       `curvature=None` at d = 1 instead of raising, and the ragged d = 1 T3–T3 branch (C.3) computes two terms
       that cancel to roundoff.
  - That matches the standing ruling "d = 1 is a supported *degenerate* case" (`dev/HANDOFF.md:150-154`;
    `deferred_and_rejected.md:49-54`).
  - **Fix:** rewrite I-12. COREWISE d = 1 runs. MANIFOLD d = 1 exact uses the GN model, documented, with the
    caveat m3b. The existing uniform-manifold d = 1 *retraction* gap is noted as pre-existing, not introduced.
- **m2 — MINOR: `cg_truncated` would also fire on boundary stops.**
  - With `keep = pos & ~hit`, a boundary stop clears `ok`, so `optimizers.py:733` reports boundary stops as
    truncations; "keeps its meaning" needs that line changed.
  - **Fix:** `cg_truncated = not converged and not ok and not boundary`. Order the display's ✓ / ∂ / ⌇ / ⋯
    accordingly. Initialise `boundary=False` in `state0`.
- **m3 — MINOR: rejected TR steps are unspecified.** State:
  - that a rejected step counts toward `max_newton`;
  - whether `losses` gets a duplicate `f` (`:712`);
  - what `stats['newton']` counts (`:726`);
  - that callback and history fire with `tr_accepted=False` and the unchanged `x_cores`, which keeps O2's
    hand-objective check valid.
- **m4 — MINOR: the descent guard ignores the radius in TR mode.** It substitutes −g (`:734-736`), which can lie
  outside Δ.
  - **Fix:** scale it to `min(1, Δ/‖g‖)`.
- **m17 — MINOR: TR constants and units.**
  - `radius < _TR_MIN_REL·max(‖x‖, 1)` uses the coordinate norm `cw.corewise_norm(x)` (`:756`). That includes the
    √n of the orthonormal factors, while Δ is in the HS / tangent metric. Use ‖X‖_HS (`geom.point_norm_sq`).
  - Give the value of `_TR_MIN_REL`.
  - Add a Δ_max cap; Nocedal–Wright Alg. 4.1 has one.
  - Report a collapse at a zero-residual point (E2's degenerate case) as converged-at-floor in `NewtonInfo`, not
    as a failure.
- **m12 — MINOR: bit-identity holds for eager numpy only (I-5).**
  - At Δ = ∞ the eager numpy path is bit-identical; I traced every `where`, and no inf or NaN arises.
  - Under jit the compiled program changes, so agreement is to tolerance only. That is fine for O2
    (1e-8 / 1e-7, `sweep_optimizers.py:132`, `:143`).
  - **Fix:** say so.
- **m13 — MINOR: X13 is a performance fix, not a jit-safety fix.**
  - `from_point` in a trace is jit-safe: `ut3_orthogonal_representations` is polymorphic and pad-safe. The cost is
    one eager re-orthogonalisation per regularized uniform model build.
  - **Fix:** correct the wording, and make `_bgeom` → `from_frame` its **own** commit before S8 (it touches the GN
    path, J-8).
- **n4 — NIT: a string in the scalar history.** The `hessian` string field enters `_NEWTON_SCALAR_FIELDS`, whose
  rows are documented as scalar-only. Say so.
- **OK.**
  - The `newton_cg` frontend kwargs.
  - `on_line_search_failure` stays line-search-only.
  - The callback stays host-side.
  - The equality semantics (no new frontend runtime class).
  - The slow-tier release gate in S12.

---

## Responses to the flagged decisions

| # | verdict | note |
|---|---|---|
| I-1 S/T by zipper | **Agree** | `sharing_internals.md` §"The S_i machinery"; reuse `SharedFrameData.centers` when shared (m18). |
| I-2 `hessian=` enum | Defer to Nick | "exact" is right for both geometries. |
| I-3 mode from `curvature is not None` | **Agree**, minus the reserved slot (m11) | |
| I-4 keep `GaussNewtonModel` | Defer to Nick | It becomes a misnomer in exact mode; at minimum the §4.2 wording (m14). |
| I-5 one CG body | **Agree**, with m12 and m2 | The 5-tuple blast radius is complete. |
| I-6 TR default in exact | Agree in principle; defer to Nick | E2 shows 0–1 rejections per level with these constants. Add Δ_max and the m17 fixes. |
| I-7 forcing 1.0 in exact | Agree; defer to Nick | E2 shows the quadratic tail with power 1.0. Update the docs (m5). |
| I-8 ε relative to ‖X‖, 1e-6 | **Agree, provisionally** | E2 supports it, and larger or Δ-coupled ε is worse. S0 confirms (m20). |
| I-9 GN stays default | Agree | |
| I-10 companion arrays | **Agree** | |
| I-11 recompute per matvec | Agree as default; defer to Nick | Drop the reserved slot. |
| I-12 uniform d = 1 raises | **Revise (m21)** | COREWISE must not raise; MANIFOLD d = 1 = GN (N ≡ 0). |
| I-13 slot norms deferred | **Agree** | ‖N‖ is not free. |
| I-14 one optional kind method | **Agree** | Plus the OPEN_QUESTION note (m14). |
| I-15 block-sum oracle as a test helper | Agree | |
| I-16 S0 before S6 | Agree | E2 is a first data point for it. |

| # | verdict |
|---|---|
| X1 | Correct (`optimizers.py:357`, `:424`, `:737`). |
| X2 | Correct. |
| X3 | Correct (`tv_operations.py:417-419`). |
| X4 | Correct (`probing.py:794-799`). |
| X5, X6 | Correct. |
| X7 | Correct (c10 reproduced). |
| X8 | An estimate, measured later. |
| X9 | Correct (`sharing.py:552`). |
| X10 | Correct. The GN Rayleigh quotient along g, already computed for Δ0, is a free λ proxy for S0. |
| X11, X12 | Correct. |
| X13 | Real, but a performance issue (m13). |

---

## What must change for AGREE (prioritized)

1. **M1.** A TR-aware display header with no `%d` on `None`; a guarded κ; an exact-mode display test and doctest.
2. **M3.** The frontend curvature companion keeps host masks static (or carries no masks); add a frontend uniform
   exact-jit one-trace test.
3. **M2.** One misfit transpose per Newton step, via the curvature companion or `lm.gradient − g_R`; correct the
   rationale.
4. **The MINORs.** Fix each or list it in §I for Nick:
   - m1 facts table;
   - m2 `cg_truncated`;
   - m3 rejected-step bookkeeping;
   - m3b the d = 1 / open-format caveat;
   - m4 guard scaling;
   - m5 forcing-power docs;
   - m6 oracle-sweep edit policy;
   - m7 import cycle;
   - m8 `t3_linalg` placement;
   - m9 uniform lift masks and the real-slot σ_min;
   - m10 drop `corewise_inner`;
   - m11 drop the reserved slot;
   - m12 bit-identity wording;
   - m13 X13 wording and its own commit;
   - m14 the §4.2 qualifier, the `_cg_solve` prose, OPEN_QUESTION, per-slice CHANGELOG, recording E2;
   - m15 the five tests;
   - m16 T16 robustness;
   - m17 TR constants, units, Δ_max and collapse reporting;
   - m18 reuse the centres;
   - m19 the regularizer assumption recorded;
   - m20 the S0 gate and fallback stated, and the phase-in wording;
   - m21 the d = 1 policy.
5. **The NITs** (n1, n2, n4–n8) can be fixed during implementation; noting them in the plan is enough.

VERDICT: REVISE
