# Implementation plan — the exact Riemannian Hessian (curvature term) for `newton_cg`

**v2, 2026-10-26. Writer revision after critic round 1 (`critique_v1.md`, verdict REVISE, 3 MAJOR / ~22 MINOR / 7 NIT)
and the maintainer's T3Polynomial input. Nothing implemented.**

- **Repo:** `/home/claude/t3toolbox` @ `920d776` (`main`).
- **Downstream consumer:** `/home/claude/t3polynomial` @ `388090d`, read-only.
- **Design source:** `dev/riemannian_newton_curvature_plan.md` ("DP").
- **[V] claims:** re-checked by read-only scripts under `SCR/checks/`, where
  `SCR = /tmp/claude-0/-home-claude/29728d3c-c433-5028-9de9-1af7ead3c123/scratchpad/plan_review`. Run them as
  `cd /home/claude/t3toolbox && python -I -B SCR/checks/<script>`; always use `-B` so no `__pycache__` is written
  into the repo.
- **[E] claims:** the critic's experiments, `SCR/check_*.py`.
- **References:** file:line is at `920d776`.

Target (Nick's decision, DP status banner): **level 2, the exact Riemannian Hessian, computed by a hand-written
two-hole sweep**:

```
H p = Π 𝒥ᵀω²𝒥 Π p  +  Π Lᵀ (K_{Aᵀω²r} − K_{grad f}) L Π p  +  λ Π p
```

- `L` is the lift from gauged frame coordinates to raw core variations.
- `K_Z` is the two-hole sweep: the directional derivative of a corewise transpose.
- `grad f` is the **misfit** Riemannian gradient, as a doubled-rank T3.
- Companions: Steihaug–Toint CG plus a trust region, a damped lift, and minimal diagnostics.

**Primary target: `probe_derivatives`.** T3Polynomial fits only that kind, on all three backends and with sharing. It
is therefore the first kind built, and the plan pre-verifies it (F13–F15).

---

## CHANGES (v1 → v2)

### MAJOR findings — all fixed

| # | finding | change | where |
|---|---|---|---|
| M1 | exact mode + `verbose=True` crashes: TR sets `ls_steps=None`; `optimizer_display.py:140-141` formats it with `%d` | In TR mode the header **replaces** the `ls …` segment with `TR Δ %.2e ✓/✗`. Every new cell goes through a `None`/non-finite guard (the `_cell` rule, `:63-67`). κ = pKp/(pHp−pKp) is guarded against a zero or non-finite denominator. GN output stays byte-identical. Added an exact-mode `_format_header` test and a run-and-paste doctest. | C.14, F-T19 |
| M2 | the misfit gradient is transposed twice per Newton step; v1's "model is a jit argument" rationale contradicts `precompute_and_caching.md:46-53` | **Retracted the rationale** (a leaf beside `sweep` is exactly the per-frame channel). The misfit gradient is computed once and carried in the curvature companion (`CurvatureState.misfit_gradient`). In exact mode `LocalModel.gradient` and frontend `GaussNewtonModel.gradient` read it and add `g_R` as today, bit-identical to the GN computation. One transpose per Newton step. | C.9, C.13 |
| M3 | the frontend pytree (`fitting.py:748-754`) passes raw children, so the uniform companion's host masks (Z from `utv_to_ut3`, `utv_operations.py:322`) would be traced under `jit(m.hessian)` and raise in `require_concrete_masks` (`ut3_masking.py:74`) | The companion classes (`CurvatureState`, `ManifoldCurvatureData`) are **registered pytrees whose flatten runs `partition_static`** (`common.py:698-724`). Masks, shape ints and Python bools go into a value-hashed `StaticSkeleton` aux (`:680-683`). One registration fixes both layers: the backend `LocalModel` and the frontend model just pass the companion as a child, and jax recurses into its own flatten. Added `test_jit_uniform_exact_gauss_newton_model`, mirroring `test_dispatch.py:515`: one trace across rebuilt models. | C.9, C.13, F-dispatch |

### MINOR findings

| # | resolution |
|---|---|
| m1 | Facts table corrected. F1 now states `i ≤ d−2`; `c1` no longer compares the nonexistent `T_{d−1}`. `c2`'s lift block is **fixed** (it used the tangent's `H` instead of the frame center `L_i T_i`) and **seeded** (`np.random.seed(1)`); it now gives 6.0e-15 / 1.2e-15. The test helpers will not port the old code (F-preamble). |
| m2 | `cg_truncated = (not cg_converged) and (not cg_ok) and (not cg_boundary)` (`optimizers.py:733` changes). `boundary` is initialised as `rs0 < 0.0`, a dtype-stable False. Display order: ✓ converged, ∂ boundary, ⌇ truncated, ⋯ maxiter. |
| m3 | Rejected TR steps are specified (C.9 "TR bookkeeping"). They count toward `max_newton`. `losses` gets one entry per history row, so `losses[k] == history[k]['objective']`, duplicates included. `stats['newton']` counts CG solves, as today. New `stats['newton_accepted']`. The callback and history fire with `tr_accepted=False` and the unchanged `x_cores`. |
| m3b | New caveat: on `MANIFOLD`, `N ≡ 0` at `d = 1` and on any open (maximal-rank) structure, so exact = GN there [E4, E5]. Recorded in `numerical_contracts.md` (G). |
| m4 | In TR mode the descent guard uses `p = −g·min(1, Δ/‖g‖)`. |
| m5 | Listed every place the `cg_forcing_power` default appears: `docs/fitting_and_optimization.md:470`, backend `optimizers.py:656`, `:677`, frontend `t3toolbox/optimizers.py:341`. Plus a CHANGELOG "Changed" line (G). |
| m6 | `tests/oracle_sweeps/` stay byte-close per their README edit policy. New module `tests/oracle_sweeps/sweep_exact.py`, driven by new classes in `tests/test_oracle_sweep.py`, in both tiers. The GN rows are untouched (F). |
| m7 | No import of `geometry.py` from `fv_operations`/`ufv_operations`. ε comes from the lift factors' own spectrum, `‖X‖² = Σσ(S_0)²` [V `c18`: 1.4e-15]. |
| m8 | `t3_inner_product_corewise_transpose`, `t3_precompute_inner_product_sweep` and `t3_inner_product_corewise_hessian[_from_sweep]` move to **`t3_linalg.py`**, next to `t3_inner_product` (`:176`) per the module map. The uniform twins go to `ut3_linalg.py`. The shared ungauged kernel moves down into `t3_linalg` and `tv_operations` imports it. That is acyclic: nothing in `t3_linalg`'s chain imports `tv_operations` [V]. |
| m9 | The uniform lift's output masks come from **`frame_data[5]`**, as `ufv_variation_masks((um, um, lm, lm))`, not the plain point masks. `sigma_min` is taken over **real** slots only, computed once in the companion. |
| m10 | Dropped `corewise_inner` from the protocol: `curvature_quadratic(p) = geom.inner(Πp, Lᵀ K_N L Πp)`. Lᵀ is the Frobenius adjoint and costs nothing m-dependent. |
| m11 | Dropped the reserved `None` slot. The companion is stored directly; a later store-mode flip would append a defaulted field. |
| m12 | Bit-identity at Δ = ∞ is claimed for **eager numpy only**. Under jit the program changes, so agreement is to tolerance (O2 uses 1e-8 / 1e-7, `sweep_optimizers.py:132`, `:143`). |
| m13 | X13 reworded as a **performance** fix: `from_point` is jit-safe but re-orthogonalises. It is now its own commit, S6a, before the frontend slice. |
| m14 | All accepted (G): the §4.2 qualifier; `_cg_solve` prose at `scan_body_principles.md:190` and `parameters_not_closures.md:111`; append to `dev/OPEN_QUESTION_extension_surface.md` §"What to watch" (do not archive); create `[Unreleased]` in S1 with lines per slice; record E2 in `verification_records.md`. |
| m15 | All five tests added (F-T20…T24). |
| m16 | T16 is rewritten on `‖g‖/‖g₀‖` inside a window of `[1e-10, 1e-1]`, with ≥ 3 points, asserting `e_{k+1} ≤ C e_k²` with C = 10. E2's tail fits it: 2.6e-2 → 5.9e-4 → 4.4e-7. |
| m17 | Floor measured in HS units: `‖X‖_HS = √geom.inner(v_X, v_X)` with `v_X = geom.point_tangent(frame)`. Constants: `_TR_RADIUS_FLOOR_REL = 1e-12`, `Δ_max = 1e4·Δ0`. Collapse is reported through a new `stop_reason`: `'objective_floor'` (the model's predicted change is below f's resolution — E2's zero-residual case) vs `'trust_region_floor'`. |
| m19 | Recorded in `deferred_and_rejected.md` and `curvature_internals.md`: the curvature uses the misfit gradient because ∇ρ is tangent for ridge and weight decay. A curvature-carrying regularizer such as Grasedyck–Kramer would need an optional `Regularizer.normal_gradient` hook. Not built. |
| m20 | The S0 gate is explicit (H). If S0 has not run before S4, ship E2's measured default `DEFAULT_LIFT_DAMPING = 1e-6` and record E2. The "phase-in" wording is gone: per E2, the `f(0) = 0` clip handles exact-zero slots, the first step lifts the new σ to about Δ, and the TR does the rest. Damping is a safety margin, not the mechanism. |
| m21 | d = 1 policy rewritten (I-12). `MANIFOLD` d = 1, ragged and uniform: the exact model **is** the GN model (structural: no curvature data; `N ≡ 0`, documented caveat). `COREWISE` d = 1 runs exact (`GN + K_{Aᵀr}`, no T3–T3, no `utv_to_ut3`). The uniform-manifold d = 1 *retraction* gap is noted as pre-existing [E6]. |

### NITs

All are noted for implementation:

- n1: one argument rule — the one-shot order, with the sweep appended (`tv_apply_transpose` / `_from_sweep` precedent,
  `apply.py:172`, `:264`).
- n2: the constant is `DEFAULT_LIFT_DAMPING`.
- n4: history rows carry two short strings, `hessian` and `stop_reason`; the docstring says so.
- n5: signatures are expanded at implementation.
- n6: the `_src_step` justification is carried over (`sampling_derivatives.py:1549-1550`).
- n7: the uniform lift output is re-masked, at the cost of one multiply.
- n8: §D gives one letter per HISTORICAL row, decided by analog.

### What I pushed back on

- **m18 (reuse `SharedFrameData.centers` in the shared lift).** Kept the plain zipper. The companion holds centers only
  for **group** modes (`sharing.py:472`, `:546-560`). The lift needs `T_i` for every bond and `H_i` for every
  non-group mode, so any structure with a singleton mode needs the full zipper anyway. Reusing only helps when every
  mode is in a group, and even then it saves one m-independent `O(d·n·r³)` zipper per Newton step.
  `precompute_and_caching.md`'s scope-ladder rule ("Creating a NEW state channel for a sub-percent win is where to
  stop") applies: the item is not in the channel today, and putting all-mode centers in would change the public
  `SharedFrameData` for a sub-percent win. The alternative is listed for Nick (I-21).
- **I-4 (keep the `GaussNewtonModel` name).** The critic deferred it. T3Polynomial's scripts use
  `tfit.probe_derivatives_model(...).gn_hessian/.gn_quadratic/.gradient/.objective_value`
  (`scripts/x06_hessian_fd_probe.py:61-122`, `x07_hessian_free_newton.py:74-131`). A rename buys no capability and
  costs a deprecation cycle on the most-used class, so I still recommend keeping the name.

### Folded in from the T3Polynomial input

- **probe_derivatives first.** It is pre-verified, ragged and uniform-padded, K- and C-stacked, orders 0–3 (F13, F15).
  The slices are re-ordered so that by S9 the complete T3Polynomial path — probe_derivatives × {ragged, uniform,
  uniform_jax} × {unshared, shared} — is exact-capable (H).
- **Shared exact pre-verified** with the probe_derivatives kind, against FD along the **shared** retraction. It
  converges at O(h²), so it is measured (not proven) second order there (F14).
- **The downstream surface is preserved.** Unchanged:
  - `topt._geometry_ops` (`t3polynomial/fitting.py:108`, `scripts/d07_curvature_dispersion.py:64`);
  - `topt._setup` (`d08_profile_ops.py:113`);
  - `bopt.least_squares_problem` / `Problem.local_model(x).gradient` / `geom.inner` (the zero reference,
    `t3polynomial/fitting.py:86-112`);
  - `lm.hvp` / `lm.gn_quadratic` / `lm.retract` with GN meaning (`d07:120-163`, `d08:115-130`);
  - `g0norm_newton` / `gtol_rel` / `max_newton` / `verbose` / `val_sample` / `val_data`.

  The public equivalent of `_geometry_ops` already exists (`fitting.ragged_backend_geometry`, `fitting.py:426-441`).
  Documenting it is I-18.
- **Research-repo measurement bed** (S0b, after S6): `e01_eigen_fit.py`, `o01_ot_fit.py`, `d05_darcy_fit_assess.py`
  and `s05_fit.py`, plus `x06`/`x07` as the FD-Hessian cross-check. They are referenced, never copied: research
  never accretes into the library (CLAUDE.md "Where things live").

### New findings this round

- **F16.** The base point's corewise-frame sweep and adjoint state are exact `T`/`S` transforms of the manifold-frame
  ones (`c18`): `ν_c = Tν`, `η_c = Sη`, `σ̃_c = Tσ̃`, `dx̃i_c = S dx̃i`, `σ̂_c = Tσ̂`, all ≤ 1e-15, order-wise for jets.
  This refines X4. In recompute mode, v1's "extra corewise frame sweep" for probe kinds is a recomputation through
  `G`; the transform matters only for the store variant (I-11).
- **F17.** `‖X‖² = Σσ(S_0)² = Σσ(T_i)²` gives ε for free (m7).

---

## Verified facts this plan rests on

| # | claim | script | result |
|---|---|---|---|
| F1 | `T_i := ⟨L_i, H_i⟩ = tt_zipper_right_to_left(L, R)[i+1]` for `i ≤ d−2`; centers `H_i = L_i T_i`; `S_iᵀ := einsum('axb,aub->xu', O_i, H_i) = V_i^X U_iᵀ`; σ(S_i) = mode-i Tucker spectrum, σ(T_i) = bond-(i+1) TT spectrum | `c1_identities.py` | ≤ 3e-12; spectra match the dense unfoldings |
| F2 | Lift identity `tv_to_dense(t3_corewise_frame(U, L), Lv) == tv_to_dense(frame, v)`, gauged and ungauged, minimal frame | `c2` (fixed), `c3`, `c4` | 6.0e-15, 1.2e-15; per slot ≤ 3.3e-15 |
| F3 | The same on non-minimal frames: Tucker slack (`Sᵀ` wide 2×4) and TT slack (`T_1` tall 9×4). The pinv is an exact one-sided inverse on every `t3_orthogonal_representations` frame (`nD ≤ nU`, `rR ≤ rL`) | `c7`, `c8`, `c12` | 9.4e-16, 9.2e-16 |
| F4 | Zero-padded `resize`: the guarded pinv is finite; at X = 0 the lift is identically 0 except the undamped last TT slot | `c7` | finite; 0.0 |
| F5 | **Block-sum oracle** (no new code): `K_c θ̂ = Bᵀ t3_*_corewise_transpose(c, sample, tv_to_t3(t3_corewise_frame(θ), θ̂))`, vs central FD of the one-hole; symmetric | `c2`, `c9` | 1.1e-9 (FD-limited) all six kinds; ⟨a,Kb⟩ = ⟨b,Ka⟩ to 13 digits |
| F6 | Exact = GN + ⟨θ̂,K_{Aᵀr}θ̂⟩ − ⟨θ̂,K_{grad f}θ̂⟩ vs FD along `tv_retract` (minimal, TT-SVD-conditioned, d = 4) | `c4` | 252.85288420 vs 252.85288182 (h = 2.5e-3); GN 285.03 |
| F7 | Lagrangian null space `K_N L p_vert = 0` | `c4`, `c10` | 1.3e-11 vs `‖K_{Aᵀr}L p_vert‖ = 1.3e4` |
| F8 | Under the orthogonal Π, `hvp(p_vert) ≠ 0` (`‖Πp_vert‖/‖p_vert‖ = 0.61`) — DP §4.6 wording is wrong | `c10` | — |
| F9 | apply two-hole from existing kernels; the lean sweep is identical at both frames | `c5` | 4.7e-16 |
| F10 | probe two-hole and T3–T3 two-hole from existing kernels (+1 new string), C = (2,) | `c6` | ≤ 4.4e-16 |
| F11 | apply_derivatives two-hole (order 2), C = (2,) | `c13` | ≤ 2.5e-16 |
| F12 | `len_*` sufficiency of the new strings | `c11`, `c17` | §D |
| **F13** | **probe_derivatives two-hole**, orders 0, 1, 2, 3, K = 2, C ∈ {(), (2,)}, from existing jet kernels plus one 6-term adjoint step, a 6-term `dx̃i` dot and 2+2+9-term assemblies. `two_hole()` in `c14` is the S3 template | `c14_probe_derivs_twohole.py` | ≤ 3.6e-16 vs F5 |
| **F14** | **Shared exact**, sharing `(0,0,1)`, probe_derivatives order 2. Tied lift `U̇_g = M_g⁺[V_{i}]` (replicated) satisfies `DΦ[Lv] = Ev`. Exact quadratic vs FD of f along the **shared** retraction | `c15_shared_exact.py` | lift 5.3e-16. Exact 1058.19431605; FD at h = 2e-2…2.5e-3 gives 1058.1942037 / …288 / …3092 / …3141, errors falling 4× per halving (O(h²)). GN 1058.0325 |
| **F15** | Padding transparency: zero-padded θ, θ̂, ww, pp, z̃ (what mask-on-entry produces) give the same real block and **exactly 0** in padded output slots | `c16_padding_transparency.py` | 0.0 / 0.0 |
| **F16** | Corewise-frame sweep and adjoint state = T/S transforms of the manifold ones (probe jets, apply σ̂) | `c18` + inline | ≤ 9.5e-16 |
| **F17** | `‖X‖² = Σσ(S_0)² = Σσ(T_i)²` | `c18` | ≤ 1.4e-15 |

### Critic's experiments [E]

| # | claim | script | result |
|---|---|---|---|
| E1 | Exact matches FD to 5–6 digits at a small bond σ = δ ∈ {1e-2, 1e-3, 1e-4}; the model's validity radius is about σ; the Hessian is indefinite there (4 of 12 directions) | `check_small_sigma.py`, `check_small_sigma_signs.py` | — |
| **E2** | **Zero-padded restart (decisive for I-8).** d = 3 `(6,6,6)` apply, W = 400, rank 2 → 3. ε = 0 and 1e-6‖X‖ both give **13 steps / 69 hvps, 0 rejections**, vs GN+Armijo 54 and GN+TR 52 / 352. ε = 1e-2‖X‖: 17 / 173. Δ-coupled ε: 56 / 634, not converged. Over-ranked level: exact 18 / 117 vs GN 80, not converged. 30% noise: 11 / 53 vs 32. Steihaug stopped on negative curvature twice per run with no collapse; the new σ jumped from 0 to ~Δ on step 1 | `check_restart_tr.py`, `_tr2`, `_tr3` | — |
| E3 | float32 curvature cancellation is benign: relative error vs GN ≤ 2e-4 at δ = 1e-4 | `check_float32_cancellation.py` | — |
| E4, E5 | `N ≡ 0` at d = 1 and on maximal-rank structures (7e-16, 8e-16) — a non-circular sign/normalisation check | `check_d1_normal.py`, `check_full_rank_N0.py` | — |
| E6 | Uniform `MANIFOLD` d = 1 optimizers already raise in `utv_retract → utv_to_ut3` | — | — |

---

## A. Scope and non-goals

**In scope**, in build order (H):

1. The lift `L` and its adjoint `Lᵀ`, plus the damped filter: ragged, uniform and shared; C and K stacks.
2. The T3–T3 two-hole `K_{grad f}`, in `t3_linalg` / `ut3_linalg`.
3. Two-hole corewise sweeps `K_c θ̂` for all six kinds. **`probe_derivatives` comes first** (the T3Polynomial path),
   then probe, apply, entries and the apply/entries derivatives. Each is a standalone backend function plus a
   `SamplingKind.corewise_hessian` method behind a `has_corewise_hessian` capability flag.
4. `LocalModel` exact mode (`hessian='exact'`): compile-once, one misfit transpose per Newton step, static-safe
   companions.
5. Steihaug–Toint CG plus a trust region in `newton_cg`, a TR-aware display, `stop_reason`.
6. Frontend: `GaussNewtonModel` / factories / `optimizers.newton_cg` wiring; `_bgeom → from_frame`.
7. Minimal diagnostics: Δ, accept, CG boundary, `pKp`, real-slot `σ_min`, ε.
8. Tests, docs, CHANGELOG, and the research-repo measurement plan (S0a/S0b).

**Non-goals.** All v1 non-goals stand:

- no AD through the T3-SVD (L3);
- the jvp and polynomial-FD routes only as oracles;
- no preconditioner, LM damping or NL2SOL β;
- no N-slot-norm diagnostic;
- no curvature for `gradient_descent` / `mc_sgd` / `adam`;
- no stacked *optimization* (`_require_unstacked`, `optimizers.py:269-298`) — the stacked C and K paths live at the
  model level;
- no nonlinear-residual Hessian term (T3Polynomial's `Σ rᵢ∇²rᵢ` stays theirs; DP §5);
- no `fit(...)` facade.

Revised:

- **d = 1 on MANIFOLD** is not an error. The exact model is the GN model (`N ≡ 0`), on both layers (I-12).
- **No change to rank continuation.** Restarts need no phase-in mechanism beyond the trust region (E2).

---

## B. Math-to-code map

Notation: `X = Φ(θ)`; frame `(U, O, L, R)` = `T3Frame.data`; gauged coordinates `v = (V_i, H_i)`;
`Π = tv_orthogonal_gauge_projection` (`tv_operations.py:83-137`).

| math | code (new unless cited) | shapes (per C element) |
|---|---|---|
| raw gauge `θ = (U, L)` | `geom.base_point(frame)` (`geometry.py:183-189`, `:274-276`, `:378-380`, `:483-485`) | — |
| corewise frame of θ | `t3_corewise_frame` (`fv_conversions.py:30-41`); `ut3_corewise_frame` (`ufv_conversions.py:122-137`) | — |
| `T_i` | `tt_zipper_right_to_left(L, R)[i+1]` (`tt_operations.py:192-199`), `i ≤ d−2` | `(rL_{i+1}, rR_{i+1})`, `rR ≤ rL` |
| `H_i^X = L_i T_i`, `S_iᵀ = ⟨O_i, H_i^X⟩` | `fv_center_factors` (extracted from `sharing.py:546-554`) | `(rL_i,nU_i,rR_{i+1})`, `(nD_i,nU_i)`, `nD ≤ nU` |
| `‖X‖_HS` | `√Σσ(S_0)²` (F17) | — |
| filter `f(σ) = σ/(σ²+ε²)` with a clip, `f(0) = 0` | `linalg.filtered_reciprocal`, `linalg.damped_pseudo_inverse` | — |
| `L`: `U̇_i = (S_iᵀ)^{+ε}V_i`, `Ġ_i = H_i ×₃ T_i^{+ε}` (`i ≤ d−2`), `Ġ_{d−1} = H_{d−1}` | `tv_corewise_lift` | `(nD,N)→(nU,N)`; `(rL,nU,rR)→(rL,nU,rL')` |
| `Lᵀ`, the Frobenius adjoint | `tv_corewise_lift_transpose` | inverse shapes |
| `K_c θ̂ = d/ds ∇_θ⟨c, S(Φ(θ+sθ̂))⟩` | `t3_{probe,apply,entries}_corewise[_derivatives]_hessian[_from_sweep]` | raw `K+C` |
| `⟨θ̂,K_cθ̂⟩ = ⟨c, S(D²Φ[θ̂,θ̂])⟩`, `D²Φ = Σ_{a≠b}` ordered pairs | normalisation F6, E5 | — |
| `Z = grad f` as a T3 | `tv_to_t3(frame, g_misfit)` (`tv_operations.py:199-330`) / `utv_to_ut3` | doubled rank |
| `∇_θ⟨Z, Φ(θ)⟩` (one hole) | `t3_inner_product_corewise_transpose` (`t3_linalg`) | raw `C` |
| `K_Z θ̂` (two holes) | `t3_inner_product_corewise_hessian[_from_sweep]` (`t3_linalg`), dotted zippers | raw `K+C` |
| `K_N = K_{Aᵀω²r} − K_{grad f}` | never formed | — |
| Riemannian Hessian | `LocalModel.hvp` (exact) = `Π[𝒥ᵀ𝒥Πp + Lᵀ K_N LΠp] + H_R p` | `K+C` |
| COREWISE exact | `GN_corewise + K_{Aᵀω²r}`; no lift, no grad term | — |
| shared lift | `_tied_filtered_solve` (refactor of `sharing.py:644-664`) | — |
| identity regularizer | `H_R = λΠ` is already exact (`fitting_internals.md` §Regularization). The curvature uses the **misfit** gradient only | — |

**Exactness of the lift on any frame from `t3_orthogonal_representations`** (F2, F3).

- `S_iᵀ` is wide or square and `T_i` is tall or square, so the pseudo-inverses are exact right and left inverses at a
  numerically full-rank point.
- A wide `T` (`rL < rR`) arises only on hand-built frames (`frame_variations.md` "Four rank stores"). There `L⁺` is the
  least-squares lift: a caveat (C.12).

**Coordinate-level gauge laws** (F7, F8; replaces DP §4.6):

1. `K_N L p_vert = 0`.
2. `hvp(p) = hvp(Πp)`.
3. `Π hvp(p) = hvp(p)`.

**Open structures** (E4, E5): at d = 1 and at maximal ranks `N ≡ 0`. The curvature term vanishes exactly, and there
`K_{Aᵀr}` and `K_{grad f}` cancel to roundoff — the non-circular oracle of F-T20.

---

## C. File-by-file changes

Conventions for every signature:

- one argument per line, three aligned columns (name · type · `#` shape), one return element per line, the
  `len=d, elm_shape=…` micro-grammar (`signature_style.md`; n5);
- argument order is the one-shot order with the sweep appended (n1);
- shipped docstrings cite local `t4s.pdf` App. A and external references (Absil–Mahony–Trumpf 2013;
  Psenka–Boumal arXiv:2011.13395; Steihaug 1983; Nocedal–Wright Alg. 4.1, 7.2), with no `dev/` paths.

### C.1 `backend/linalg.py` — the filtered reciprocal

```python
def filtered_reciprocal(
        s:      NDArray,                     # singular values, shape=C+(k,), descending
        eps:    typ.Optional[NDArray],       # damping, shape=C (broadcast over k); None -> exact clipped 1/s
        rcond:  typ.Optional[float] = None,  # relative clip vs s[..., :1]; None -> dtype eps * k
) -> NDArray:                                # f(s), shape=C+(k,); 0 where clipped
```

- The guard is a branch-free double `where`: `keep = s > rcond·s_max`; `f = where(keep, s / where(keep, s²+eps², 1),
  0)`.
- `eps=None` reproduces `_tied_solve`'s filter bit-for-bit (`sharing.py:658-659`).
- Why a clip at ε = 0: a zero-padded slot's σ comes back as ~1e-17, not 0. Per E2 the clip is what handles restarts.

```python
def damped_pseudo_inverse(
        M:      NDArray,                     # shape=C+(m, n)
        eps:    typ.Optional[NDArray],       # shape=C; None -> clipped pinv
        rcond:  typ.Optional[float] = None,
) -> typ.Tuple[
        NDArray,                             # M^{+eps}, shape=C+(n, m)
        NDArray,                             # s, shape=C+(min(m, n),)
]:
```

- A plain thin batched SVD: no basis is kept, and it is pad-transparent (padding gives zero rows/cols, σ = 0, f = 0).
  So it is not `pad_safe_svd` (`uniform_svd_prefix_orthogonalization.md`: pad-safety is needed only for a kept
  basis).
- Dispatch is inferred.

### C.2 `backend/fv_operations.py` — frame factors and lift factors

```python
def fv_center_factors(
        frame:  typ.Tuple[
            typ.Sequence[NDArray],  # up_tucker_cores. len=d, elm_shape=C+(nUi, Ni)
            typ.Sequence[NDArray],  # down_tt_cores.   len=d, elm_shape=C+(rLi, nDi, rR(i+1))
            typ.Sequence[NDArray],  # left_tt_cores.   len=d, elm_shape=C+(rLi, nUi, rL(i+1))
            typ.Sequence[NDArray],  # right_tt_cores.  len=d, elm_shape=C+(rRi, nUi, rR(i+1))
        ],
) -> typ.Tuple[
    typ.Tuple[NDArray, ...],        # centers H_i = L_i T_i.  len=d,   elm_shape=C+(rLi, nUi, rR(i+1))
    typ.Tuple[NDArray, ...],        # tucker factors S_i^T.   len=d,   elm_shape=C+(nDi, nUi)
    typ.Tuple[NDArray, ...],        # bond factors T_i.       len=d-1, elm_shape=C+(rL(i+1), rR(i+1))
]:
```

- This is steps 1–2 of `fv_shared_frame_data` (`sharing.py:546-554`), extracted. `fv_shared_frame_data` calls it, and
  its existing doctest pins bit-identity.
- Imports: `sharing → fv_operations` is acyclic (`fv_operations.py:13-21` imports only `tt_operations`,
  `t3_operations`, `t3_conversions`, `corewise`, `common`) [V].

```python
@dc.dataclass(frozen=True, eq=False)       # identity hash/eq: array fields (SharedFrameData precedent, sharing.py:445)
class LiftFactors:
    tucker_pinvs:  typ.Tuple[NDArray, ...]  # (S_i^T)^{+eps}. len=d,   elm_shape=C+(nUi, nDi)
    bond_pinvs:    typ.Tuple[NDArray, ...]  # T_i^{+eps}.     len=d-1, elm_shape=C+(rR(i+1), rL(i+1))
    tucker_svals:  typ.Tuple[NDArray, ...]  # σ(S_i), elm_shape=C+(nDi,)
    bond_svals:    typ.Tuple[NDArray, ...]  # σ(T_i), elm_shape=C+(rR(i+1),)
    eps:           NDArray                  # the absolute damping used, shape=C (a traced LEAF: not stale, no recompile)

def fv_lift_factors(
        frame:          typ.Tuple,              # (U, O, L, R) orthogonal frame, stack C
        lift_damping:   float,                  # RELATIVE: eps = lift_damping * ‖X‖_HS, ‖X‖² = Σ σ(S_0)² (F17)
        rcond:          typ.Optional[float] = None,
) -> LiftFactors:
```

- Registered as a jax pytree with arrays as leaves and no aux (`sharing.py:1081-1091` precedent).
- **I-1 stands** (the critic agrees): derive by zipper, never store on the frame, never re-SVD H/O/R
  (`sharing_internals.md` §"The S_i machinery", decisions 1–2). The only SVDs are of the small factors, to filter
  them.

### C.3 Lift and lift adjoint (`tv_operations.py`); T3–T3 (`t3_linalg.py`)

```python
def tv_corewise_lift(
        frame:          typ.Tuple,              # (U, O, L, R), stack C
        variations:     typ.Tuple[
            typ.Sequence[NDArray],              # tucker_variations. len=d, elm_shape=K+C+(nDi, Ni)
            typ.Sequence[NDArray],              # tt_variations.     len=d, elm_shape=K+C+(rLi, nUi, rR(i+1))
        ],
        lift_factors:   'fv_operations.LiftFactors',   # stack C
) -> typ.Tuple[
    typ.Tuple[NDArray, ...],                    # Udot. len=d, elm_shape=K+C+(nUi, Ni)
    typ.Tuple[NDArray, ...],                    # Gdot. len=d, elm_shape=K+C+(rLi, nUi, rL(i+1))
]:

def tv_corewise_lift_transpose(
        frame:            typ.Tuple,            # (U, O, L, R), stack C
        core_variations:  typ.Tuple[
            typ.Sequence[NDArray],              # len=d, elm_shape=K+C+(nUi, Ni)
            typ.Sequence[NDArray],              # len=d, elm_shape=K+C+(rLi, nUi, rL(i+1))
        ],
        lift_factors:     'fv_operations.LiftFactors',
) -> typ.Tuple[
    typ.Tuple[NDArray, ...],                    # UNGAUGED frame coords, elm_shape=K+C+(nDi, Ni)
    typ.Tuple[NDArray, ...],                    # elm_shape=K+C+(rLi, nUi, rR(i+1))
]:
```

- Two `'...'` einsums: the factors are C-only and broadcast over K for free.
- The last TT slot passes through undamped.
- The same factors in both directions keep `Lᵀ_ε K L_ε` symmetric for every ε.

**T3–T3, moved to `t3_linalg.py` (m8).** The rebasis/variations map bodies (`tv_operations.py:333-362`) and the
ungauged core of `tv_project_t3_onto_tangent_space` (`:388-415`, including its d = 1 branch `:402-407`) move to
`t3_linalg.py` as the private kernel `_inner_product_frame_transpose(frame, z)`.

- `tv_project_t3_onto_tangent_space` becomes `tv_orthogonal_gauge_projection(frame,
  t3_linalg._inner_product_frame_transpose(frame, x), shared_data)`. This is bit-identical and pinned by the existing
  tests. The callers (`manifold.py:1410`, `shared_geometry.py:353`) are unchanged.
- **Conflict with DP §4.2 kept (X3):** the public function gauge-projects (`:417-419`), so it is not ∇⟨Z,Φ⟩ at
  `(U,G,G,G)`.

```python
def t3_inner_product_corewise_transpose(
        z:          typ.Tuple[typ.Sequence[NDArray], typ.Sequence[NDArray]],  # Z, stack C
        core_pair:  typ.Tuple[typ.Sequence[NDArray], typ.Sequence[NDArray]],  # θ, stack C
) -> typ.Tuple[
    typ.Tuple[NDArray, ...],    # ∇_U ⟨Z, Φ(θ)⟩, shapes of tucker_cores
    typ.Tuple[NDArray, ...],    # ∇_G ⟨Z, Φ(θ)⟩, shapes of tt_cores
]:                              # = _inner_product_frame_transpose(t3_corewise_frame(core_pair), z)

def t3_precompute_inner_product_sweep(
        z:          typ.Tuple,  # Z, stack C (e.g. tv_to_t3(frame, g_misfit): doubled rank)
        core_pair:  typ.Tuple,  # θ, stack C
) -> typ.Tuple[
    typ.Tuple[NDArray, ...],    # BU_i = B_Z U_iᵀ.     len=d, elm_shape=C+(nZi, nUi)
    typ.Tuple[NDArray, ...],    # G2_i = G_Z ×₂ BU_i.  len=d, elm_shape=C+(rZi, nUi, rZ(i+1))
    typ.Tuple[NDArray, ...],    # ZL.                  len=d, elm_shape=C+(rZi, rLi)
    typ.Tuple[NDArray, ...],    # ZR.                  len=d, elm_shape=C+(rZ(i+1), rL(i+1))
    typ.Tuple[NDArray, ...],    # env_i = ZL G_Z ZR.   len=d, elm_shape=C+(rLi, nZi, rL(i+1))
]:

def t3_inner_product_corewise_hessian(
        z:                typ.Tuple,   # Z, stack C
        core_pair:        typ.Tuple,   # θ, stack C
        core_directions:  typ.Tuple,   # θ̂ = (Udot, Gdot), stack K+C
) -> typ.Tuple[typ.Tuple[NDArray, ...], typ.Tuple[NDArray, ...]]:   # raw K+C

def t3_inner_product_corewise_hessian_from_sweep(
        z:                typ.Tuple,
        core_pair:        typ.Tuple,
        core_directions:  typ.Tuple,
        sweep:            typ.Tuple,   # = t3_precompute_inner_product_sweep(z, core_pair)
) -> typ.Tuple[typ.Tuple[NDArray, ...], typ.Tuple[NDArray, ...]]:
```

- Per matvec (F10): `BUd`, `G2d`, the dotted zippers, `envd`, `dG = envd ×_i BU + env ×_i BUd`,
  `dU = (envd·G + env·Ġ) B_Z`.
- The dotted zippers come from new polymorphic chain helpers `tt_zipper_tangent_left_to_right` / `_right_to_left`
  (`tt_operations.py`), whose module-level scan body `_tt_zipper_tangent_step` has 3 terms on
  `'...ij,...iak,...jal->...kl'`. The descriptive chain names are a cataloged exception
  (`naming_conventions.md` §"Cataloged exceptions").
- The ragged d = 1 branch is kept: general T3 math. The **model path never reaches it**, because MANIFOLD d = 1 has no
  curvature data (I-12).
- The uniform twins live in `ut3_linalg.py`. `ut3_linalg` must not import `utv_operations`; it does not today [V].

### C.4 `backend/sampling_derivatives.py` — `probe_derivatives` two-hole (PRIMARY; F13)

```python
def t3_probe_corewise_derivatives_hessian(
        ztildes:          typ.Sequence[NDArray],  # residual jets (ω² applied), len=d, elm_shape=(order+1,)+W+C+(Ni,)
        ww:               typ.Sequence[NDArray],  # len=d, elm_shape=W+(Ni,)
        pp:               typ.Sequence[NDArray],  # len=d, elm_shape=W+(Ni,)
        core_pair:        typ.Tuple,              # θ, stack C
        core_directions:  typ.Tuple,              # θ̂, elm_shape=K+C+(...)
        order:            int,
        chunk_size:       typ.Optional[int] = DEFAULT_CHUNK_SIZE,
) -> typ.Tuple[typ.Tuple[NDArray, ...], typ.Tuple[NDArray, ...]]:   # raw K+C, W summed

def t3_probe_corewise_derivatives_hessian_from_sweep(
        ztildes, ww, pp, core_pair, core_directions, order,
        frame_sweep:      typ.Tuple,              # the model's (xi, mu, nu, eta)_jets; xi/mu reused (F9)
        chunk_size:       typ.Optional[int] = DEFAULT_CHUNK_SIZE,
) -> ...:
```

The composition, exactly `c14.two_hole` (F13):

1. **Undotted state at the corewise frame (recompute default, I-11).**
   - `xi_j`, `mu_j` are reused from the model sweep: identical at both frames (F9).
   - `nu_j = compute_nu_jets(G, xi_j, trs)` and `eta_j = compute_eta_jets(G, mu_j, nu_j, trs)` are recomputed through
     `G`.
   - The tildes come from the existing `compute_deta_tilde_jets` (`:1440`), `compute_tau_tilde_jets` (`:1605`),
     `compute_sigma_tilde_jets` (`:1611`) and `compute_dxi_tilde_jets` (`:1632`), at `(U, G, G, G)`.
2. **Dotted forward, existing kernels.**
   - `xid_j = build_input_jets(compute_xi(Udot, ww), compute_xi(Udot, pp))` (`:258-277`);
   - `mud_j = compute_sigma_jets(Gdot, G, G, …)` (`:1002`);
   - `nud_j = compute_tau_jets(Gdot, G, G, …, nu_j)` (`:1033`);
   - `etad_j = compute_deta_jets(Gdot, G, G, mu_j, nu_j, mud_j, nud_j, trs)` (`:919`).
3. **Dotted adjoint-up lift.** `detd = 'tWCo,KCio->tWKCi'(z̃, Udot)`, new string.
4. **Dotted adjoint sweeps.** A new module-level scan body. `_adj_tilde_step` cannot be reused: it infers
   `nf = len(P.shape[:-3])` from a C-only core (`:1530`).

   ```python
   def _adj_tilde_tangent_step(
           carry:  NDArray,       # tilde-dot jet: (order+1,)+W+K+C+(rLi,)
           data:   typ.Tuple,     # (P, Pdot, xi_jet, xid_jet, deta_tilde, deta_tilde_dot, edge_jet, edge_dot_jet,
                                  #  tilde_jet (the undotted carry of this core), trs_r), one core
   ) -> typ.Tuple[NDArray, typ.Tuple[NDArray]]:
   ```

   Its six terms are the product rule of `_adj_sweep_step`'s `prop + src` (`:1460-1475`):

   | term | on `trs` | factors | K sits on |
   |---|---|---|---|
   | 1 | `trs[:, :s2, :]` | `carry_dot · P · xi` | the carry |
   | 2 | `trs[:, :s2, :]` | `carry · Ṗ · xi` | the core, `len_C` |
   | 3 | `trs[:, :s2, :]` | `carry · P · xid` | `xid` |
   | 4 | `trs` | `edge_dot · P · deta` | the edge |
   | 5 | `trs` | `edge · Ṗ · deta` | the core |
   | 6 | `trs` | `edge · P · detd` | `detd` |

   - Production keeps `_adj_tilde_step`'s memory shape: `prop` as the fused two-term reverse recurrence, `src` as
     the inner order-scan with `W·r²` peak. The `_src_step` closure comment is carried over (n6). There are
     ragged/uniform scan-body variants (`:1563-1582` pattern).
   - `compute_tau_tilde_tangent_jets` and `compute_sigma_tilde_tangent_jets` (by `tt_reverse`) wrap it.
   - **Implementation order:** the c14 dense-`trs` form lands first as `_trs` reference twins
     (`naming_conventions.md` "`_trs` suffix"). The recurrence/scan form then lands and is tested equal to them.
5. **Dotted `dx̃i`.** The product rule of `_dxi_tilde_jets_step`'s two terms (`:1619-1629`): six terms (c14).
6. **Assemblies.**
   - Tucker: `'tWKCa,tWCo->KCao'(etad, z̃)` + `'uWKCa,uWo->KCao'(dxtd, w_jets)`.
   - TT: 9 terms, the product rule of the three `_tt_variation_jets_trs_core` strings (`:1748-1777`).
   - Both are W-chunked through `_wchunked_sum` (`:1908-1930`), with new module-level chunk functions
     `_tt_tangent_chunk_summed` / `_tucker_tangent_chunk_summed`. These are stable objects for the memoized
     `_wchunked_sum_step` (`:1866-1885`), so the kind's `chunk_size` is honoured.
   - Peak per W row: each dotted term has a GN assembly term's shape. S3 **measures** the peak with
     `memory_analysis`. If it exceeds 1.5× the GN assembly, `estimate_chunk_size` gains a `hessian=` argument and
     `_resolve_chunk_size` (`t3toolbox/optimizers.py:145-166`) passes it. T3Polynomial's `e01`/`o01` pass
     `chunk_size=None` (dense) via `newton_kwargs`, which is unaffected.

### C.5 Plain probe (`probing.py`), apply/entries (`apply.py`, `entries.py`), apply/entries derivatives

All are as in v1, with their arguments reordered per n1:

- **Probe** (F10): two `_sigma_step` calls per dotted tilde step; `compute_deta` ×2 for the dotted `dxi_tilde`; one
  new string `'WCo,KCio->WKCi'`; the Tucker assembly adds `'WCo,WKCa->KCao'`.
- **Apply/entries** (F9): `compute_sigma_hat` through G; then `compute_dxi`, `compute_sigma(Gdot,G,G,…)` and
  `compute_tau(Gdot,G,G,…, sigma_hat)`; then the three-term assemblies. Entries uses `_entry_xis` and
  `_onehot_vectors` (`entries.py:264-309`). The shared tail is `_apply_corewise_hessian_adjoint`, the analog of
  `apply.py:357-400`.
- **Apply/entries derivatives** (F11): one new scan body, `_sigma_hat_tangent_jets_step`, with 3 terms in the
  adjoint-wired `trs[:, :s2, :]` form of `:2226-2234`, plus the dotted 3 + 3 assemblies.

These land **after** the T3Polynomial path (H). Until each lands, its kind has `has_corewise_hessian = False`, so
exact mode raises a structural error naming the kind.

### C.6 `backend/fitting.py` — `SamplingKind`

On `SamplingKind` (`:305-412`):

```python
    has_corewise_hessian:  typ.ClassVar[bool] = False   # set True by whoever IMPLEMENTS corewise_hessian (the
    #                                                     has_block_sumsq pattern, :326)

    def corewise_hessian(
            self,
            core_directions,   # θ̂ = (Udot, Gdot) at base_point, elm_shape=K+C+(...)
            residual,          # RAW r; ω² applied HERE, as in transpose (:553-555, :643-646)
            sample,
            base_point,        # (tucker_cores, tt_cores), stack C
            sweep,             # this model's precompute(frame, sample); its xi/mu half is reused (F9)
    ):                         # -> raw (dU, dG), K+C
        raise NotImplementedError(...)   # names hessian='exact' and has_corewise_hessian
```

- `ProbeDerivativesKind` (`:627-652`) implements it first (S4), passing `chunk_size=self.chunk_size`. The others
  follow in S10–S12.
- The kinds stay geometry-agnostic: they receive the point, never the geometry.
- This widens the open `SamplingKind` surface. The note goes in `dev/OPEN_QUESTION_extension_surface.md`
  §"What to watch" (`:131`), which stays open (I-14).

### C.7 `backend/geometry.py` — the Geometry protocol (`optimizers.py:60-98`)

Four exact-mode members, needed only for `hessian='exact'`. `corewise_inner` is dropped (m10).

```python
    def curvature_precompute(
            self,
            frame,              # this geometry's frame
            misfit_gradient,    # Π 𝒥ᵀω²r (tied when shared) -- the model's ONE misfit transpose
            aux=None,           # geom.precompute(frame) (SharedFrameData when shared)
            lift_damping=...,   # DEFAULT_LIFT_DAMPING
    ):                          # -> per-frame curvature data; None iff the curvature term VANISHES here
    def lift(self, frame, variations, curvature_data):              # frame coords -> raw θ̂ at base_point
    def lift_transpose(self, frame, core_variations, curvature_data): # raw -> UNGAUGED frame coords
    def tangential_curvature(self, frame, core_directions, curvature_data):  # raw K_{grad f} θ̂, or None (static)
```

**`ManifoldGeometryOps`** (`:146-236`):

- **d = 1:** `curvature_precompute` returns `None` — the term vanishes, `N ≡ 0` (I-12).
- **Otherwise** it returns `ManifoldCurvatureData(lift_factors, z, z_sweep, sigma_min)`:
  - `z = tv_to_t3(frame, misfit_gradient)`. The untied embedding represents the same ambient tensor (F14).
  - `z_sweep = t3_precompute_inner_product_sweep(z, base_point)`.
  - `sigma_min` is the real-slot minimum over `tucker_svals ∪ bond_svals`, per C element (m9). The ragged and uniform
    twins compute it with the masks.
- **With `groups`:** the lift factors gain the per-group filtered tied factors (C.11).

**`CorewiseGeometryOps`** (`:239-308`):

- `curvature_precompute` returns `()`: an empty pytree, not None, so the curvature term is **present** with no data.
- `lift` / `lift_transpose` are the identity; `tangential_curvature` returns None (DP §4.3).

**`ManifoldCurvatureData`** is `@dc.dataclass(frozen=True, eq=False)`.

- **It is registered via `partition_static`** (M3): the flatten is `dyn, skel = partition_static((lift_factors, z,
  z_sweep, sigma_min))`, with `aux = skel`, and the unflatten is `rebuild_static`.
- On the uniform layer `z` carries `(shape, (tucker_mask, tt_mask))` host statics (`utv_operations.py:315-322`). They
  land in the value-hashed `StaticSkeleton` (`common.py:680-695`), so a rebuilt-identical companion is the same jit
  cache key — the `UT3Frame` "masks as aux" precedent (`uniform_pytree_composition.md`).
- No hidden state (mechanism 3 is avoided: `precompute_and_caching.md` "Three mechanisms").

### C.8 Uniform-specific geometry member

**`UniformManifoldGeometryOps.from_frame(frame_data, plain_masks, sharing)`** builds `var_masks =
ufv_variation_masks(frame_data[5])` without orthogonalising — the same derivation as the kinds
(`uniform_fitting.py:88-91`).

- **X13 is a performance fix (m13).** `_bgeom` (`fitting.py:311-314`) calls `from_point`, which runs
  `ut3_orthogonal_representations`. That is jit-safe but one eager re-orthogonalisation per regularised uniform model
  build.
- It lands as its **own commit, S6a**, asserting identical masks to `from_point` (it touches the GN path; J-8).

### C.9 `backend/optimizers.py` — `CurvatureState`, `LocalModel`, `Problem`, Steihaug, trust region

```python
@dc.dataclass(frozen=True, eq=False)
class CurvatureState:
    """The per-model curvature companion (exact mode): a LEAF beside sweep / geom_aux (the per-frame channel,
    docs/contributor/precompute_and_caching.md scope ladder). Registered via partition_static (statics -> aux)."""
    misfit_gradient:  typ.Any   # Π 𝒥ᵀω²r -- computed ONCE per Newton step; LocalModel.gradient reads it (M2)
    geometry_data:    typ.Any   # geom.curvature_precompute(...): None => the curvature term vanishes (MANIFOLD d = 1)
```

**`LocalModel`** (`:105-161`) gains `curvature: typ.Any = None` (appended, defaulted).

- Exact mode is `curvature is not None`; the curvature term is present iff `curvature.geometry_data is not None`.
  Both are static Python facts resolved at trace time (I-3; m11 — no reserved slot).

```python
    @property
    def gradient(self):              # exact mode: curvature.misfit_gradient (+ g_R), bit-identical to the GN path
    def hvp(self, p):                # Π[𝒥ᵀ𝒥Πp + Lᵀ K_N LΠp] + H_R p   (curvature term iff present)
    def curvature_hvp(self, p):      # Π Lᵀ K_N L Π p                    (raises if not exact)
    def quadratic(self, p):          # gn_quadratic(p) + curvature_quadratic(p) -- the model actually solved
    def curvature_quadratic(self, p):  # geom.inner(Πp, Lᵀ K_N LΠp)  (m10; keeps K+C)
    # gn_quadratic / jacobian / retract: UNCHANGED (gn_* keeps its Gauss-Newton meaning: X1; T3Polynomial d07/d08)
```

The curvature body inside `hvp` / `curvature_hvp`:

```python
    cd = self.curvature.geometry_data
    base = self.geom.base_point(self.frame)
    th = self.geom.lift(self.frame, Pp, cd)
    K = self.kind.corewise_hessian(th, self.residual, self.sample, base, self.sweep)
    Kt = self.geom.tangential_curvature(self.frame, th, cd)
    if Kt is not None:                        # static: None for COREWISE
        K = cw.corewise_sub(K, Kt)
    y = self.geom.lift_transpose(self.frame, K, cd)
```

`hvp` projects `𝒥ᵀz + y` once, through the shared post-pass via `aux`.

**`Problem.local_model(x_cores, sample=None, data=None, hessian='gauss_newton', lift_damping=DEFAULT_LIFT_DAMPING)`**
(`:190-201`):

1. The GN build is unchanged.
2. If `hessian == 'exact'`:
   - check `kind.has_corewise_hessian` (structural `ValueError`);
   - `g = geom.project(frame, kind.transpose(residual, sample, frame, sweep), aux)` — the **only** misfit transpose
     (M2);
   - `cd = geom.curvature_precompute(frame, g, aux, lift_damping)`;
   - `curvature = CurvatureState(g, cd)`.
3. `hessian` is validated: `'gauss_newton' | 'exact'`, otherwise `ValueError`.
4. **Default unchanged**, so T3Polynomial's `prob.local_model(zero.data).gradient` (`t3polynomial/fitting.py:110`) and
   `d07`/`d08` behave identically.

**Pytree.** `_flatten_local_model` (`:799-801`) adds `m.curvature` to the `partition_static` tuple. A registered
`CurvatureState` is a dynamic leaf, and jax recurses into its own static-safe flatten. The unflatten passes it by
keyword. `CurvatureState`'s registration is `partition_static((misfit_gradient, geometry_data))`.

**Steihaug–Toint in the existing body** (I-5). The state grows to `(p, r, d, rs, i, ok, lm, tol2, maxiter, radius,
boundary)`.

- `boundary0 = rs0 < 0.0` (m2).
- `_cg_cond` is unchanged (`:540-545`).
- The body is as v1 (`hit`, `tau`, `keep = pos & ~hit`), with these properties:
  - all-`where`, no inf arithmetic (`rad2 = where(isfinite(radius), radius², 0)`);
  - module-level and closure-free, with the radius in state (`:527-539`; `scan_body_principles.md` §5).
- **At `radius = inf`:** p/r/d/rs are bit-identical **on the eager numpy path**. Under jit the compiled program
  differs and agreement is to tolerance (m12).
- **Signatures:** `_cg_core(local_model, rhs, tol, maxiter, use_jit, radius)` with `_jitted(_cg_core, (4,))`, so
  `radius` is traced. It returns `(p, i, rs, ok, boundary)`; `_cg_solve(local_model, rhs, tol, maxiter, use_jit,
  radius=math.inf)` likewise.
- **Blast radius** (the critic confirms it complete): `optimizers.py:730`, `tests/backend/test_optimizers.py:218`,
  `:223`, `tests/test_dispatch.py:595`.
- **Prose updates:** `scan_body_principles.md:190` and `parameters_not_closures.md:111` (m14).

**`newton_cg`** (`:646-770`) new keywords, appended:

```python
        hessian:          str = 'gauss_newton',              # 'exact': Riemannian (manifold) / Euclidean (corewise) Hessian
        lift_damping:     float = DEFAULT_LIFT_DAMPING,      # exact only: eps = lift_damping * ‖X‖_HS (1e-6, E2)
        step_control:     typ.Optional[str] = None,          # None -> 'line_search' (GN) / 'trust_region' (exact)
        tr_radius0:       typ.Optional[float] = None,        # None -> GN Cauchy length ‖g‖³ / gᵀH_GN g
```

- `cg_forcing_power: typ.Optional[float] = None` resolves to 0.5 for GN (unchanged) and 1.0 for exact (I-7).
- The `g0norm_newton`/`g0norm_cg` chain (`:717-718`) is untouched, so T3Polynomial's zero-referenced
  `g0norm_newton` + `gtol_rel=1e-3` (`t3polynomial/fitting.py:189-194`) composes unchanged.
- Loop changes:
  1. `lm` is rebuilt **only after an accepted step**.
  2. TR: Δ0 = `tr_radius0` or `‖g‖³/max(gn_quadratic(g), tiny)`, falling back to `‖g‖` if not finite.
     `Δ_max = _TR_RADIUS_MAX_FACTOR·Δ0`, with the factor 1e4 (m17).
  3. `p, i, rs, ok, hit = _cg_solve(..., radius)`.
  4. Descent guard: `p = −g·min(1, Δ/‖g‖)` in TR mode (m4).
  5. `pHp = lm.quadratic(p)`; `pKp = lm.curvature_quadratic(p)` (exact).
  6. Line search unchanged, with ρ from the model's `pHp`.
  7. TR:
     - single trial at α = 1;
     - `ρ = Δf/pred` with `pred = slope + ½pHp < 0`;
     - accept iff ρ > 0.1;
     - ρ < 0.25 or non-finite: Δ ← ¼‖p‖;
     - ρ > 0.75 and `hit`: Δ ← min(2Δ, Δ_max).
  8. **Floor (m17):** if `Δ < _TR_RADIUS_FLOOR_REL·max(‖X‖_HS, Δ0)` with `_TR_RADIUS_FLOOR_REL = 1e-12` and
     `‖X‖_HS = √geom.inner(v_X, v_X)`, `v_X = geom.point_tangent(lm.frame)`, the run stops.
     - `stop_reason = 'objective_floor'` if the last `|pred| ≤ 16·ε_mach·max(|f|, tiny)` — the model cannot resolve f
       any more, E2's zero-residual case.
     - Otherwise `'trust_region_floor'`.
- **TR bookkeeping (m3).**
  - A rejected step counts toward `max_newton`, keeps `x`, `lm`, `g` and `gnorm`, and emits a NewtonInfo row with
    `tr_accepted=False` and the unchanged `x_cores`.
  - `losses` appends each row's `f`, so `losses[k] == history[k]['objective']`.
  - `stats['newton']` counts CG solves (as today) and the new `stats['newton_accepted']` counts accepted steps.
  - The new `stats['stop_reason']` is one of `'gradient' | 'max_newton' | 'line_search' | 'trust_region_floor' |
    'objective_floor'`.

**`NewtonInfo`** (`:602-635`), appended, all defaulting to `None`:

| field | meaning |
|---|---|
| `hessian` | `'gauss_newton'` / `'exact'` (string, n4) |
| `tr_radius` | Δ |
| `tr_accepted` | the TR step was accepted |
| `cg_boundary` | CG stopped on the TR boundary |
| `pKp` | curvature part of `pHp` |
| `lift_sigma_min` | real-slot `σ_min` (`float(cd.sigma_min)`) |
| `lift_eps` | the ε used |
| `stop_reason` | final row only (string, n4) |

- `cg_truncated` is redefined per m2.
- In TR mode `ls_steps = ls_failed = None` and `alpha = 1.0`. T3Polynomial's `d05_darcy_fit_assess.py:232` already
  filters `ls_steps is None`; `b01_backend_bench.py:275` would collect `None`s, which the release note mentions.
- `_NEWTON_SCALAR_FIELDS` grows automatically; `test_newton_cg_diagnostics` compares sets.

### C.10 Uniform mirror

- **`ufv_lift_factors`** (`ufv_operations.py`):
  - mask once (`ufv_apply_frame_masks`, `ufv_masking.py:64`), then use the polymorphic zipper on supercores
    (`utv_operations.py:582-583` precedent) and `'d...'` einsums;
  - padded σ = 0 gives f = 0;
  - ε comes from `Σσ(S_0)²`, real slots only by construction (padding has σ = 0);
  - `sigma_min` is computed over real slots via masks: `xnp.where(real_mask, σ, inf)`, with host-numpy masks.
- **`utv_corewise_lift` / `_transpose`:**
  - mask on entry;
  - the lift output carries **`ufv_variation_masks((um, um, lm, lm))` from `frame_data[5]`** (m9) and is re-masked
    (n7);
  - the transpose output carries `ufv_variation_masks_over_stack(frame_data[5], C, K+C)` (`ufv_masking.py:170-195`);
  - masks via `np`, supercores via `xnp`.
- **`ut3_inner_product_corewise_*`** (`ut3_linalg.py`): Z is **masked first** (`ut3_apply_masks`). The gappy `[Q|P]`
  masks and honest-zero boundaries are consumed through the mask (`testing_strategy.md` "phantom rank").
  `utv_to_ut3` raises at d < 2, but the model never calls it there (I-12).
- **Sampling** (`ut3_sampling.py`, object-type grouping): `ut3_probe_corewise_derivatives_hessian` and its siblings
  mask once, pack, and delegate to the polymorphic ragged-type functions.
  - Those functions gain d-prefixed branches: no Python loop over d (the unroll trap, `sampling_derivatives.py:271-277`);
    sweeps via `xscan`.
  - **Mask-on-entry suffices: the two-hole is padding-transparent** (F15: 0.0 real-block error, exactly 0 in padded
    output).
- **Kinds** (`uniform_fitting.py:69-233`): `corewise_hessian(v_sc, r, sample, base_sc, sweep)` uses
  `(base_sc[0], base_sc[1], self.shape, self.masks)` as the point (`:83-86`) and the uniform sweep layout
  `(mb, packed_ww, packed_pp, shape, psweep)` (`utv_sampling.py:238-281`).
- **Varying-rank C:** every factor, ε and mask is per element, and masked-out directions lie in `ker L ∩ ker Lᵀ`
  (`uniform_ranks_and_varieties.md` "tangent layer").

### C.11 Shared factors

- **Refactor** `_tied_solve` (`sharing.py:644-664`) into `_tied_filtered_solve(shared_data, gi, V, s_filter)` plus its
  adjoint `_tied_filtered_solve_transpose`.
  - The filter is `filtered_reciprocal(s_g, eps, rcond)`; `eps=None` is bit-identical to today.
- **Lift.** Group modes get `U̇_g = M_g^{+ε}[V_{i₁};…]`, replicated over the group (F14: 5.3e-16). Singletons get the
  per-mode lift. TT slots are untied.
- **Adjoint.** `Y_g = Σ_{i∈g}Y_{U_i}`, then `U_M diag(f(s_g)) Vt_M Y_g`, split by `row_splits`; then `Π_sh` in `hvp`
  via `aux`.
- **N_sh** uses the tied misfit gradient that `geom.project(…, aux)` returns (DP §4.3); verified end-to-end in F14.
- **Padded restarts** (`sharing_internals.md` §"Padded restarts"): the new `s_g` levels are 0, so f = 0. The gated
  channel gets no curvature; the TT-variation escape is untouched.
- **Shared COREWISE:** the curvature is `project_mean(K_{Aᵀr}θ̂)`, the same post-pass as its GN (`sharing_internals.md`
  "One principle, two formulas").
- **Oracle:** the shared retraction's second order is **measured** (F14, O(h²) on probe_derivatives order 2), not
  proven. T12 uses FD along the shared retraction as a measured oracle, alongside the dense Lagrangian of Φ∘T.

### C.12 Safe mode — preconditions vs caveats

**No new enforced preconditions.**

- The frontend `hessian` / `curvature_hessian` / `quadratic` / `curvature_quadratic` take `_require_at_frame`
  (`fitting.py:101-118`).
- Orthogonality stays a construction invariant of the model.

Caveats, documented and never enforced:

1. Exact at ε = 0 on numerically full-rank points; damped toward GN in directions with σ ≲ ε, i.e. σ ≲ 1e-6‖X‖ by
   default (E2).
2. A hand-built wide `T_i` gives the least-squares lift.
3. Shared: exact where `s_g > 0`.
4. **m3b:** on MANIFOLD at d = 1 and on maximal-rank structures `N ≡ 0` and exact = GN. At d = 1 the model uses the GN
   model structurally.

The backend stays check-free.

### C.13 Frontend (`t3toolbox/fitting.py`, `t3toolbox/optimizers.py`, `t3toolbox/manifold.py`)

- **`GaussNewtonModel`** (`:184-403`):
  - Field `curvature: typ.Any = None`, appended. The registration (`:748-754`) adds it as a child, and its own
    `partition_static` flatten keeps the statics static (M3). The positional unflatten is updated.
  - `gradient` (`cached_property`, `:329-340`): in exact mode it wraps `curvature.misfit_gradient` and adds `g_R` —
    the same seam audited (M2; `precompute_and_caching.md` checklist item 4).
  - New methods `hessian`, `curvature_hessian`, `quadratic`, `curvature_quadratic`.
  - `evaluate` is mode-aware.
  - `gn_hessian` / `gn_quadratic` / `objective_value` are **unchanged** (T3Polynomial `x06`/`x07`).
  - The name stays `GaussNewtonModel` (I-4).
- **Factories** (`:534`, `:554`, `:574`, `:633`, `:684`, `:707`) and `_uniform_model` (`:454-488`) gain
  `hessian='gauss_newton'` and `lift_damping=DEFAULT_LIFT_DAMPING`, via one helper `_with_curvature(model, hessian,
  lift_damping)`.
  - The helper uses the backend geometry: `ragged_backend_geometry`, or `from_frame` on uniform.
  - All 7 sites are covered.
  - **Downstream payoff:** `tfit.probe_derivatives_model(M, Y, ww, pp, J, res, weight=w, hessian='exact').hessian(p)`
    replaces `x07`'s central-difference Hessian, which costs 2 model builds + 2 retractions per action
    (`x07_hessian_free_newton.py:102-106`).
- **`optimizers.newton_cg`** (`:318-439`): explicit `hessian=` and `lift_damping=`. `step_control` / `tr_radius0` ride
  `**kwargs`. `_setup` and `_geometry_ops` are **unchanged** (used downstream).
- **`manifold.py:1381-1386`:** the retract docstring adds "second order on a minimal, numerically full-rank frame". The
  matching sentence goes in the `numerical_contracts.md` retract row.

### C.14 Diagnostics (`backend/optimizer_display.py`) — M1

`_format_header` (`:121-144`):

- **Segment.** When `info.tr_radius is not None`, the `ls %d…` segment is replaced by `"TR Δ %.2e %s ‖Δx‖/‖x‖ %.1e" %
  (Δ, '✓' if info.tr_accepted else '✗', step_rel)`. Otherwise the `ls` segment stays exactly as today.
- **CG symbol.** `✓` if converged, `∂` if `cg_boundary`, `⌇` if truncated, `⋯` otherwise (m2).
- **κ and σ.** When `info.hessian == 'exact'`, the header appends `" | κ %s σ %s"`:
  - κ = `pKp/(pHp − pKp)`, printed only when that denominator is finite and nonzero, else `—`;
  - σ = `lift_sigma_min` via `_cell` semantics, so `None` and non-finite values show `—`.
- **Byte-identity.** GN-mode output is byte-identical, so the doctests at `:160-187` and
  `tests/backend/test_optimizer_display.py` are unchanged.
- **Tests.** New test `test_exact_mode_header` (TR row, rejected row, κ with a zero denominator, boundary symbol). New
  doctest in `format_newton_iter`, run and pasted.

---

## D. Batching contract (frame-inner `W + K + C`)

| function | inputs (stacks) | output | machinery |
|---|---|---|---|
| `filtered_reciprocal`, `damped_pseudo_inverse` | s C+(k); M C+(m,n); eps C | C+… | elementwise, batched SVD |
| `fv_center_factors`, `fv_lift_factors` | frame C | C | `'...'`, zipper scan |
| `tv_corewise_lift[_transpose]` | frame/factors C; variations K+C | K+C | `'...'` (C innermost → free broadcast over K) |
| `t3_*inner_product*` | Z, θ, sweep C; θ̂ K+C | K+C | `'...'`, tangent-zipper scan |
| `t3_probe_corewise_derivatives_hessian_from_sweep` | jets `(o+1,)+W+C`; dotted `(o+1,)+W+K+C`; input jets `(2,)+W[+K]+C`; z̃ `(o+1,)+W+C+(N,)` | K+C (W summed, chunked) | grouped `contract` with `trs` |
| plain / apply / entries two-holes | θ̂ K+C; c/z̃ W+C; ww W; θ C | K+C | grouped `contract` |
| `LocalModel.hvp / curvature_hvp` | p K+C (CG: K = ()) | K+C | composition |
| `LocalModel.curvature_quadratic` | p K+C | K+C | `geom.inner` → `corewise_stack_dot(n_stack = len(K+C))` (never `corewise_dot`) |

- Per-frame data carries **C only** (`batching_and_stacking.md` §3).
- The residual carries W+C and no K; K enters only through θ̂.
- K placement follows `_sigma_step` (`probing.py:363-377`).

**New `HISTORICAL` rows** (`tests/test_contractions_interpreter.py:70`). The `c17` inventory: 36 strings used, 16
already recorded, **20 new**. One letter per row, decided by the existing analog (n8):

```
'WKCa,WCi,WCb->KCaib': 'W'          'WCa,WKCi,WCb->KCaib': 'W'          (analog 'WCa,WCi,WKCb->KCaib': 'W')
'WCo,KCio->WKCi': 'C'               'tWCo,KCio->tWKCi': 'C'             (analog 'WCi,KCio->WKCo': 'C')
'WCo,WKCa->KCao': 'W'               'tWKCa,tWCo->KCao': 'W'             (analog 'WKCo,WCa->KCao' / 'tWCa,tWKCo->KCao': 'W')
'trs,tWCa,KCaib,rWCi->sWKCb': 'C'   'trs,rWCa,KCaib,tWCi->sWKCb': 'C'   (analog 'trs,rWCa,KCaib,sWCi->tWKCb': 'C')
'trs,rWCa,KCaib,tWCb->sWKCi': 'C'   'trs,tWCa,KCaib,sWCb->rWKCi': 'C'   (analog 'trs,rWCa,KCaib,sWCb->tWKCi': 'C')
'trs,rWKCa,sWCi,tWCb->KCaib': 'W'   'trs,rWCa,sWKCi,tWCb->KCaib': 'W'   (analog 'trs,rWCa,sWCi,tWKCb->KCaib': 'W')
'trs,tWCa,rWKCi,sWCb->KCaib': 'W'   'trs,tWCa,rWCi,sWKCb->KCaib': 'W'
'trs,rWKCa,tWCi,sWCb->KCaib': 'W'   'trs,rWCa,tWCi,sWKCb->KCaib': 'W'   (analog 'trs,rWCa,tWKCi,sWCb->KCaib': 'W')
'trs,tWCa,Caib,rWKCi->sWKCb': ''    'trs,rWKCa,Caib,tWCi->sWKCb': ''
'trs,rWKCa,Caib,tWCb->sWKCi': ''    'trs,tWCa,Caib,sWKCb->rWKCi': ''    (analog 'trs,rWCa,Caib,tWKCb->sWKCi': '')
```

Each was checked sufficient on the test's generic point W = (2,3), K = (2,2), C = (2,3) (`c17`). The d-prefixed
uniform twins use the same letters and are recorded in S8. The scan-form production kernels may use a different
string set from the dense-`trs` reference twins; every string is recorded the commit it appears
(`TestVocabularyCompleteness`, `:297-310`).

**W-chunking.** Only the probe-derivatives tangent assembly is chunked (`chunking.md`: the r² blow-up lives in the
assembly). It engages on uniform + jax only and needs ≥ 3 chunks. The `shard_map` + `psum` recipe applies unchanged.

---

## E. Costs and memory (estimates; measured in S3/S8 and S0b; recorded in `verification_records.md`)

| operation | per sample, leading order | notes |
|---|---|---|
| GN matvec (apply) | `2dNn + 6dnr²` | — |
| apply exact / GN | ≈ 2× (N-dominated) – 3.5× (r-dominated) | DP said 2–2.5× |
| probe exact / GN | ≈ 3–4× | ν_c/η_c recompute through G (~10% of the two-hole) |
| **probe_derivatives exact / GN** | ≈ 3–4× at order 1–2, jets × banded (o+1) / full convolutions (o+1)² as for GN | dominated by the 6-term dotted tilde `src` order-scans and the 9-term TT assembly; the S3 measurement decides |
| `K_{grad f}` (T3–T3, doubled rank) | m-independent `O(d[Nn·n_Z + n²r² + nr³])` | — |
| lift / `Lᵀ` / factors | m-independent; `d` small SVDs per Newton step | — |
| per Newton step | + `Z` build + T3–T3 sweep; **no** extra transpose (M2) | — |

**Memory.**

- Recompute default: no new resident W-scaling state. Transients are the dotted edge variables, `(o+1)·W·K·r` each with
  K = 1 in CG.
- Store variant (I-11): probe kinds add ν_c, η_c and the three tildes (≈ +75% of the full jet sweep). These are
  obtained by **T/S transforms** of the manifold-frame quantities (F16) plus one adjoint pass at build.
- Compile time (uniform_jax, `use_jit=True`, T3Polynomial's third backend): the exact `_cg_core` body includes the
  probe-derivatives two-hole. It compiles once per run, but the first-iteration latency grows; measured in S8.

---

## F. Tests (numpy-only numerics; jax in dispatch / the jvp oracle)

New file `tests/backend/test_curvature.py`.

- Test pattern:
  - `subTest` over structures × stacks: none, C, K, K+C, multi-axis C, multi-axis W;
  - primitive subTest params (xdist);
  - per-test seeding: no global RNG reliance, and never port `c2`'s old unseeded block (m1).
- Structures:
  - minimal well-conditioned (TT-SVD core);
  - Tucker-slack `(6,6,6),(4,4,4),(1,2,2,1)` and TT-slack `(1,9,9,1)`;
  - d ∈ {1, 2, 4};
  - rank-1 bonds;
  - zero-padded `resize`;
  - the zero point;
  - maximal-rank `(3,3,3)` / `(2,3,4)`.
- The shared helper `tests/backend/_curvature_oracles.py` holds the block-sum oracle (F5), the dense Lagrangian
  Hessian, and `c14.two_hole` as the dense-`trs` reference.

| test | asserts | oracle |
|---|---|---|
| T1 factors | `T_i == zipper` (`i ≤ d−2`), `S_iᵀ == V_iU_iᵀ`, spectra == dense unfoldings; `Σσ(S_0)² = ‖X‖²` | F1, F17 |
| T2 lift identity | minimal + both slack kinds, gauged + ungauged, ε = 0 | F2, F3 |
| T3 lift adjoint | `⟨Lv, y⟩_raw == ⟨v, Lᵀy⟩` per K+C element, several ε | — |
| T4 damping | ε → 0 continuity; finite at a padded restart; zero point → lift 0 except last slot | F4 |
| T5 two-holes | every `*_corewise_[derivatives_]hessian` + T3–T3 vs **block-sum** (exact) and vs FD of the one-hole; probe_derivatives at orders 0–3 | F5, F13 |
| T6 symmetry | `⟨a,Kb⟩ = ⟨b,Ka⟩`; `⟨p, hvp q⟩ = ⟨q, hvp p⟩` | — |
| T7 retraction FD | `⟨p, hvp p⟩` vs `d²/dt² f(tv_retract(tp))`, Richardson over h ∈ {1e-2, 5e-3, 2.5e-3}; unshared; minimal, conditioned frames | F6 |
| T8 gauge laws | `K_N L p_vert = 0` (generators `tests/backend/test_tv_operations.py:121-132`); `hvp(p) = hvp(Πp)`; `Π hvp = hvp` | F7, F8 |
| T9 dense Lagrangian | hvp vs the FD Hessian of `f∘Φ − ⟨grad f, Φ⟩` through L; level-1 control | DP §4.6 |
| T10 regularized | exact + `IdentityRegularizer(λ)` vs FD of f + ρ along `tv_retract` | — |
| T11 weights | ω[mode,order] (probe_derivatives), per mode (probe), per order (apply/entries derivatives): two-hole uses `ω²r` | block-sum |
| T12 shared | tied lift identity; adjoint; exact vs FD along the **shared** retraction (measured O(h²), F14); exact vs dense Lagrangian of Φ∘T; padded shared restart: finite, gated channel curvature-free | F14 |
| T13 corewise exact | `GN + K_{Aᵀr}` (block-sum); FD Hessian of `θ ↦ f(Φ(θ))` | — |
| T14 Steihaug unit | ToyModel (`test_optimizers.py:201-225`): indefinite → boundary with `‖p‖ = Δ`; PD + small Δ → boundary; Δ = ∞ → bit-identical (eager numpy); `cg_truncated` excludes boundary stops (m2) | — |
| T15 TR loop | rejection keeps x and reuses the model (counter on `local_model`); radius rules incl. Δ_max; descent-guard scaling (m4); `losses[k] == history[k]['objective']`; `stats['newton_accepted']`; `stop_reason` values (m3, m17) | — |
| T16 convergence rate (m16) | seeded `(6,6,6)` ranks 2, W = 200 apply fit, 1% noise, after 15 GN steps. Exact: ≥ 3 points of `e_k = ‖g_k‖/‖g₀‖` inside `[1e-10, 1e-1]` with `e_{k+1} ≤ 10·e_k²`; `|1−ρ| < 0.05` on those steps. GN control: `e_{k+1}/e_k ∈ (0.05, 0.95)` | E2 tail |
| T17 uniform | per element vs ragged over `_CONFIGS` incl. varying-rank C, from nonzero full-rank starts; **exact output masks** derived non-circularly (lift: `(um, um, lm, lm)` from the frame, m9; Lᵀ/hvp: frame var masks over K); **garbage-padded** inputs (`1e6·(1−mask)`); `sigma_min` ignores padded slots; isolated rejections with a mutation check (`testing_strategy.md:143`) | ragged, F15 |
| T18 frontend | exact `GaussNewtonModel` == backend `LocalModel` (extends `tests/test_fitting.py:595`); exact `gradient` == GN `gradient` bitwise (ragged, M2); `_bgeom` via `from_frame` == via `from_point` masks; same-frame guard | backend |
| T19 display (M1) | `test_exact_mode_header` (accepted/rejected TR rows, ∂, κ zero-denominator, `σ —`) + exact `make_newton_display` end-to-end run (`verbose=True` path) | — |
| T20 open structures (m15a, m3b) | maximal-rank `(3,3,3)`, `(2,3,4)` and ragged d = 1 via the backend functions: `‖Lᵀ K_N LΠp‖ ≤ 1e-12·‖Lᵀ K_{Aᵀr} LΠp‖`; MANIFOLD d = 1 exact model has `geometry_data is None` and `hvp ==` GN (ragged + uniform); COREWISE d = 1 uniform exact runs | E4, E5 |
| T21 restart regime (m15e) | exact-mode `newton_cg` from a zero-padded `resize` of a converged lower-rank fit (E2 row 1 template): reaches `gtol` with `stop_reason == 'gradient'` and ≤ 2 rejections; ragged + one uniform; slow tier if > 3 s | E2 |
| T22 shared restart | the same, on `shared_manifold((0,0,1))` with probe_derivatives order 2 | — |

**Jax** (`tests/test_dispatch.py`):

- jit each new op (lift, Lᵀ, every `*_corewise_hessian`, T3–T3, exact `LocalModel.hvp`; ragged + uniform).
- `test_exact_local_model_rebuilt_each_step_is_one_jit_cache_key`, mirroring `:556`.
- **`test_jit_uniform_exact_gauss_newton_model`** (M3, m15d): mirror `:515` with `hessian='exact'`; three rebuilt models,
  one trace, jax leaves.
- `test_cg_radius_is_not_stale` (mirror `:581`).
- **`test_lift_damping_is_not_stale_and_does_not_recompile`** (m15b): two `lift_damping` values give one `_cg_core`
  compile and distinct results.
- The probe_derivatives exact jit with `chunk_size` engaged (≥ 3 chunks).
- `tests/test_curvature_jvp_oracle.py` (`skipUnless(jax)`, float64): two-hole == `jax.jvp` of the corewise transpose,
  all six kinds + T3–T3.

**Oracle sweeps (m6).**

- New `tests/oracle_sweeps/sweep_exact.py`, driven by `TestExactModelOracleSweep` (always-on: exact model rows vs
  block-sum and dense) and `TestExactOptimizerOracleSweep` (always-on pairwise subset + full matrix under
  `T3TOOLBOX_SLOW_TESTS=1`).
- The existing O1/O2 files stay byte-close; their GN rows must pass unchanged.

**Research-repo measurement (never copied in).**

- **S0a** (prototype, now): extend E2 to probe_derivatives with block-sum prototypes. Log `1 − ρ_tr`, tail rate,
  real-slot σ_min, ‖N‖, and the GN Rayleigh quotient along g (the free λ proxy for X10).
- **S0b** (after S6, with the real ragged exact mode): T3Polynomial's `e01_eigen_fit.py`, `o01_ot_fit.py`,
  `d05_darcy_fit_assess.py`, `s05_fit.py`:
  - Newton steps per level, hvps, wall time, rejections and stop reasons, GN vs exact, at fixed `g0norm_newton` +
    `gtol_rel=1e-3`;
  - `x06_hessian_fd_probe.py`'s FD quadratic forms vs `model.quadratic(p)` — the documented 1.95× GN curvature at
    ρ 0.06 (`dev/zero_reference_g0_plan.md:208-218`) should be reproduced by `κ`;
  - `x07_hessian_free_newton.py`'s FD Hessian vs `model.hessian`.

  The numbers go to `verification_records.md`.

**Budget.** The always-on additions are small (d ≤ 4, ranks ≤ 3, order ≤ 2). T16/T21/T22 are ≤ 3 s each or move to
the slow tier.

---

## G. Docs, CHANGELOG, doctests

**User docs.**

- **`docs/fitting_and_optimization.md`:**
  - **§4.2 qualifier (m14):** "the GN model is exact on the affine tangent space; `hessian='exact'` adds the
    manifold's curvature (§4.12)".
  - **New §4.12 "The exact Hessian and the trust region":** when it helps (ρ far from 1, a linear tail; the rate is
    about |1 − ρ|), API, cost, restarts (the clip and the TR handle them, no phase-in), reading Δ/κ/σ, d = 1 /
    open-structure caveat, and a seeded doctest printing booleans only.
  - **§5:** a guidance bullet.
  - **The "default 0.5" sentence at `:470` (m5).**
- **`docs/rank_continuation.md`:** exact mode at restarts needs nothing special (E2); `tr_radius0` across levels is
  optional; with power 1.0 the forcing term tightens under the `g0norm` pin.
- **`docs/chunking.md`:** the exact-mode probe_derivatives tangent assembly honours the same `chunk_size`, plus the S3
  measurement.
- **`docs/numerical_contracts.md`:** exact-Hessian rows (caveats 1–4; no preconditions) and the retract row's
  second-order sentence.
- **`docs/naming_conventions.md`:** `corewise_hessian` as a `corewise` example; `gn_*` = Gauss–Newton only; the new
  `tt_zipper_tangent_*` in the cataloged exceptions.
- **`docs/transposes.md`:** pointer section "Differentiating a corewise transpose".
- **`docs/sharing.md`:** exact on shared geometries (the tied lift; zero `s_g` levels gate the curvature).

**Contributor docs.**

- **New `docs/contributor/curvature_internals.md`**, added to the `contributor_guide.rst` toctree:
  - DP §2 promoted;
  - the lift and its one-sided exactness;
  - the block-sum oracle;
  - the two-index-jet composition rule (c14);
  - the T/S transform identities (F16);
  - the companion pytree rule (M3);
  - the misfit-gradient-once rule (M2);
  - decisions as settled;
  - E1–E6 and F13–F17.
- **`deferred_and_rejected.md`:**
  - rejected: L3; jvp/FD as shipped implementations;
  - deferred: storing curvature state (I-11), preconditioning / LM / NL2SOL β, slot norms of N, per-mode ε, and
    reusing shared centers (I-21);
  - **the curvature-carrying-regularizer hook (m19).**
- **`verification_records.md`:** E2 now (m14), S0a/S0b numbers, the S3/S8 cost and memory measurements.
- **`numerical_contract_catalog.md`:** caveat entries.
- **Prose (m14):** `scan_body_principles.md:190`, `parameters_not_closures.md:111`.
- **`dev/OPEN_QUESTION_extension_surface.md`:** append to "What to watch". Do not archive.

**CHANGELOG** (m14). Create `## [Unreleased]` in S1 and add lines per slice:

- Added: `hessian='exact'`, trust region, the backend functions, `NewtonInfo` fields, `stats['stop_reason']` /
  `['newton_accepted']`.
- Changed: the `cg_forcing_power` default `None` (resolves to 0.5 in GN, as before); `GaussNewtonModel.evaluate`
  mode-aware (identical in GN); `cg_truncated` no longer counts TR boundary stops; in TR mode `ls_steps` is `None`.
- Internal: `_cg_solve` returns 5 values.
- `docs/release_notes.md` upgrade note: custom `Geometry` / `SamplingKind` need the exact-mode members only for
  `hessian='exact'`.

**`dev/` routing:**

- The DP goes to `dev/archive/riemannian_newton_curvature_plan_2026-10-25.md` with a resolution banner (S13).
- `dev/curvature_oracles/` goes into T7/T9, then is archived.
- `dev/HANDOFF.md` and the CLAUDE.md Current-state pointer are refreshed.

**Gates:** `sphinx -W`; module and page doctests on both numpy generations.

---

## H. Ordered slices (each one commit, independently green: `pytest tests/ -n auto` + doctests + `sphinx -W`)

**Order principle:** T3Polynomial's path first. **By S9, probe_derivatives × {ragged, uniform, uniform_jax} ×
{unshared, shared} runs exact end to end.**

| slice | content | gate tests |
|---|---|---|
| **S0a** (research repo) | E2 extended to probe_derivatives via block-sum prototypes; ε / TR constants / Δ0 / forcing power | numbers → `verification_records.md`. **Gate rule (m20):** if S0a has not run when S4 lands, S4 ships `DEFAULT_LIFT_DAMPING = 1e-6` (E2) and records E2 |
| S1 | `filtered_reciprocal` / `damped_pseudo_inverse`; `fv_center_factors` (+ `fv_shared_frame_data` refactor, bit-identical); `LiftFactors`, `fv_lift_factors`; `tv_corewise_lift[_transpose]` (ragged); CHANGELOG `[Unreleased]` | T1–T4; sharing tests unchanged |
| S2 | move `_inner_product_frame_transpose` + map bodies to `t3_linalg` (bit-identical); `t3_inner_product_corewise_transpose`; `tt_zipper_tangent_*`; `t3_precompute_inner_product_sweep`; `t3_inner_product_corewise_hessian[_from_sweep]` (ragged, incl. d = 1) | T5 (T3–T3), T6, T20 (backend half) |
| **S3** | **probe_derivatives two-hole (ragged)**: `_trs` reference twins (c14 form) → scan-form production; `_adj_tilde_tangent_step`; dotted `dx̃i`; chunked tangent assembly; HISTORICAL rows; memory measurement | T5 (orders 0–3), T6, T11 |
| S4 | protocols: `SamplingKind.corewise_hessian` + flag (ProbeDerivativesKind only); Geometry exact members (Manifold/Corewise, d = 1 policy); `ManifoldCurvatureData`, `CurvatureState` (partition_static pytrees); `LocalModel` exact (misfit gradient once); `Problem.local_model(hessian, lift_damping)` | T7–T10, T13, T20; dispatch: exact hvp jit + one cache key + ε not stale |
| S5 | Steihaug body/state; `_cg_core/_cg_solve` 5-tuple (+3 call sites); TR loop + bookkeeping + `stop_reason`; `NewtonInfo` fields; `cg_truncated` redefinition; **TR-aware display (M1)**; `cg_forcing_power=None` | T14–T16, T19, T21 (ragged); `test_cg_radius_is_not_stale`; existing GN optimizer tests + O2 unchanged |
| S6a | `UniformManifoldGeometryOps.from_frame`; `_bgeom` → `from_frame` (GN path, own commit, m13) | T18 (mask equality); all uniform GN tests unchanged |
| S6 | frontend: `GaussNewtonModel.curvature` + methods + 7 sites + pytree; `optimizers.newton_cg` kwargs | T18; **S0b can start** (ragged) |
| S7 | shared (ragged): `_tied_filtered_solve[_transpose]` refactor (bit-identical), tied lift, shared corewise | T12, T22 |
| S8 | uniform: `ufv_lift_factors`, `utv_corewise_lift[_transpose]`, `ut3_inner_product_*`, d-prefixed probe_derivatives two-hole branches, uniform kinds/geometries exact members | T17, T20 (uniform), T21 (uniform); **`test_jit_uniform_exact_gauss_newton_model` (M3)**; chunked jit; compile-time measurement |
| S9 | shared (uniform) | T12 uniform twin + T17 rules. **T3Polynomial path complete** |
| S10 | plain probe (ragged + uniform) | T5, T6, T11, T17 |
| S11 | apply + entries (ragged + uniform) | T5, T6, T17 |
| S12 | apply/entries derivatives (ragged + uniform) | T5, T6, T11, T17 |
| S13 | docs, CHANGELOG finalised, release notes, doctests, `sweep_exact.py` (both tiers), the jvp oracle, `dev/` archive + HANDOFF/CLAUDE.md | full release gate incl. `T3TOOLBOX_SLOW_TESTS=1` |

- S1–S3 are pure additions.
- S4 is the first consumer change, GN-default throughout.
- No release happens between S4 and S13; until S12 the kinds without `has_corewise_hessian` raise a clear structural
  error in exact mode.

---

## I. Decisions for Nick (recommendation · alternative · critic's verdict)

| # | decision | recommendation | alternative | critic |
|---|---|---|---|---|
| I-1 | S/T source | zipper on demand, once per model | return them from `t3_orthogonal_representations` | Agree |
| I-2 | mode API | `hessian='gauss_newton' \| 'exact'` | `curvature: bool`; `hessian='newton'` | Defer |
| I-3 | mode encoding | derived from `curvature is not None`; term present iff `geometry_data is not None`; no reserved slot | explicit static field | Agree (minus the slot — done) |
| I-4 | frontend name | keep `GaussNewtonModel` (T3Polynomial uses it heavily) | rename + deprecated alias | Defer |
| I-5 | Steihaug | one CG body; Δ = ∞ ≡ today (eager numpy bit-identical; jit to tolerance) | separate body | Agree |
| I-6 | TR policy | TR default in exact; accept ρ > 0.1; ¼‖p‖ / 2Δ on the boundary; Δ0 = GN Cauchy; `Δ_max = 1e4·Δ0`; floor `1e-12·max(‖X‖_HS, Δ0)`; module constants | Armijo for exact too; exposed constants | Agree in principle (E2: 0–1 rejections) |
| I-7 | forcing power | `None` → 0.5 GN / 1.0 exact | 0.5 everywhere | Agree (E2 quadratic tail) |
| I-8 | ε | relative to ‖X‖_HS (= √Σσ(S_0)²), `DEFAULT_LIFT_DAMPING = 1e-6`; **E2: ε = 0 ≡ 1e-6 best; 1e-2 worse; Δ-coupled fails** | per-factor scales; the DP's ‖N‖/λ_min | Agree, S0 confirms |
| I-9 | default | GN stays default | exact by default after S0b | Agree |
| I-10 | jet layout | companion arrays | stacked length-2 axis | Agree |
| I-11 | store vs recompute | recompute per matvec (house low-memory preference, `fitting_internals.md` §"Low-memory transpose"); the probe kinds re-sweep ν/η through G (~10%) | store per model: ν_c, η_c by **T/S transform** of the model sweep (F16, free) + one adjoint pass for the tildes; ≈ +75% jet-sweep memory, ~25–35% faster probe matvec. Needs a `SamplingKind.curvature_precompute` member and a `CurvatureState` field | Defer |
| I-12 | d = 1 (revised, m21) | MANIFOLD d = 1 (ragged + uniform): exact model = GN model structurally; COREWISE d = 1 exact runs; the pre-existing uniform-manifold d = 1 retraction gap is noted, not fixed | raise | Revise — done |
| I-13 | diagnostics | Δ, accept, boundary, pKp, κ, real-slot σ_min, ε, stop_reason; defer N-slot norms | include slot norms | Agree |
| I-14 | kind surface | one optional method + flag; note in OPEN_QUESTION | close the surface | Agree |
| I-15 | block-sum oracle | test helper only | public reference function | Agree |
| I-16 | S0 timing | S0a before S4 (with the m20 fallback); S0b on T3Polynomial after S6 | implement first, tune later | Agree |
| **I-17** | TR bookkeeping | rejected steps count toward `max_newton`; `losses` one per row (duplicates on rejection); `stats['newton']` = CG solves, + `newton_accepted` | count only accepted steps | new (m3) |
| **I-18** | `topt._geometry_ops` | leave it (private, used by T3Polynomial); document `fitting.ragged_backend_geometry(geometry, shape)` (public, `fitting.py:426`) as the public equivalent in `fitting_and_optimization.md` §2.2 | promote `_geometry_ops` to public | new (downstream) |
| **I-19** | `stop_reason` | new NewtonInfo/stats key with five values; `'objective_floor'` distinguishes E2's zero-residual collapse | reuse `converged=True` for the floor | new (m17) |
| **I-20** | slice order | T3Polynomial path first (S3–S9), other kinds S10–S12, single release at S13 | kinds in v1 order (probe_derivatives last) | new (downstream) |
| **I-21** | shared centers (m18) | do **not** reuse `SharedFrameData.centers` (see pushback) | add all-mode centers/T to `SharedFrameData` and derive lift factors from them | **pushback** |

---

## J. Risks and open questions

1. **Indefinite, ill-conditioned CG.** The TR bounds steps, not CG counts. S0b measures CG iterations per step on
   T3Polynomial. The deferred follow-up is a GN-CG flexible preconditioner or LM. E2 saw 5–7 hvps per Newton step.
2. **ε.** E2 supports 1e-6, but it is one apply problem at d = 3. S0a/S0b decide.
3. **probe_derivatives kernel complexity** (S3): now pre-verified (F13, orders 0–3). The remaining risk is the
   scan-form rewrite, so the dense `_trs` twins land first and pin it to 1e-15.
4. **S3 tangent assembly memory** may exceed the GN assembly. It is measured; the estimator is extended if over 1.5×.
5. **float32** cancellation is benign (E3).
6. **Shared retraction order.** It is measured O(h²) on one case (F14) but unproven. ρ could deviate from 1 on harder
   shared points; S0b measures it on `s05_fit.py` (sharing).
7. **Compile time** of the exact uniform_jax CG body (T3Polynomial's third backend). Measured in S8. If it is large,
   the follow-up is a `jax.checkpoint`/`remat` split of the two-hole inside the body.
8. **The `_bgeom → from_frame` switch** touches the GN path. It is isolated in S6a and asserted.
9. **Downstream consumers of `ls_steps`.** In TR mode it is `None`. `d05` already filters; `b01`'s histogram would
   collect `None`s. This is a release-note item; GN mode is unchanged.
10. **Open: subsampled exact Newton (mc_sgd)** — out of scope.
11. **Open: the zero-reference g0 plan composes.** The forcing term is reference-relative, and the objective-decrement
    stop it proposes is still wanted.
12. **Open: `tr_radius0` across continuation levels.** Rec: per-level Cauchy Δ0 (default).

---

## Conflicts between the design plan (DP) and the code — consolidated

| # | DP says | code reality | resolution |
|---|---|---|---|
| X1 | §4.3: curvature into `gn_quadratic` *and* Cauchy on it | the same method serves both (`optimizers.py:357`, `:424`, `:737`); T3Polynomial relies on `gn_quadratic = ‖Jp‖²` (`d07:15`, `:124`) | new `quadratic`; `gn_*` keeps its GN meaning |
| X2 | §4.1/§4.3: S/T from the frame construction | the construction discards them; caching docs forbid frame state | zipper inside the curvature leaf |
| X3 | §4.2: the T3 projection at `(U,G,G,G)` is ∇⟨Z,Φ⟩ | it gauges at the end (`tv_operations.py:417-419`) | ungauged kernel moved to `t3_linalg` |
| X4 | §4.2: frame sweep reused | true for xi/mu (F9); probe ν/η run through R/O | recompute through G; T/S transform for store mode (F16) |
| X5 | §4.2 route 3 "NumPy backend" | `xnp`-agnostic | shipped on all backends |
| X6 | §4.7: jvp first | Nick: the two-hole is the deliverable | block-sum oracle + jax-gated jvp oracle |
| X7 | §4.6: hvp of pure gauge = 0 | false under orthogonal Π (F8) | the three coordinate laws |
| X8 | §4.5: 2–2.5× | 2–3.5× (apply), 3–4× (probe, probe_derivatives) | measured S3/S8/S0b |
| X9 | §4.1: S from the "centre-core SVD" | contraction (`sharing.py:552`) | contraction; SVD only to filter |
| X10 | §4.4: ε ~ ‖N‖/λ_min | needs an eigensolve | ε relative to ‖X‖ (E2); Rayleigh proxy logged in S0 |
| X11 | §4.3: residual raw, ω² applied | `LocalModel.gradient` includes `g_R` (`:138-139`) | misfit gradient carried in `CurvatureState` |
| X12 | §4.1: non-minimal → slack carries no curvature | the pinv is an exact one-sided inverse (F3) | caveat only for a hand-built wide `T` |
| X13 | (frontend) | `_bgeom` re-orthogonalises per uniform model build — **performance, not jit-safety** | `from_frame`, own commit S6a |
| X14 | §4.4: "phase-in" via damping at restarts | E2: the f(0) = 0 clip + TR handle restarts; damping is a margin | wording removed; default 1e-6 |
