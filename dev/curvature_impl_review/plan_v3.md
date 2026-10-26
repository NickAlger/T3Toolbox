# Implementation plan — the exact Riemannian Hessian (curvature term) for `newton_cg`

**v3, 2026-10-26. Writer revision after critic round 2 (`critique_v2.md`: verdict REVISE; 1 new MAJOR, 7 MINOR, 5
NIT). Nothing implemented.**

- **Repos:** `/home/claude/t3toolbox` @ `920d776` (`main`); downstream consumer `/home/claude/t3polynomial` @ `388090d`.
  Both read-only.
- **Design source:** `dev/riemannian_newton_curvature_plan.md` ("DP").
- **[V] claims:** read-only scripts under `SCR/checks/`, with
  `SCR = /tmp/claude-0/-home-claude/29728d3c-c433-5028-9de9-1af7ead3c123/scratchpad/plan_review`. Run them as
  `cd /home/claude/t3toolbox && python -I -B SCR/checks/<script>` (`-B`: no `__pycache__` in the repo).
- **[E] claims:** the critic's experiments, `SCR/check_*.py`.
- **References:** file:line at `920d776`.

Target (Nick's decision, DP status banner): **level 2, the exact Riemannian Hessian, computed by a hand-written
two-hole sweep**:

```
H p = Π 𝒥ᵀω²𝒥 Π p  +  Π Lᵀ (K_{Aᵀω²r} − K_{grad f}) L Π p  +  λ Π p
```

- `L` is the lift from gauged frame coordinates to raw core variations.
- `K_Z` is the two-hole sweep, the directional derivative of a corewise transpose.
- `grad f` is the **misfit** Riemannian gradient, as a doubled-rank T3.
- Companions: Steihaug–Toint CG plus a trust region, a damped lift, and minimal diagnostics.

**Primary target: `probe_derivatives`**, the only kind T3Polynomial fits — on all three backends and with sharing. It is
built first and pre-verified end to end (F13, F14, F18, F19).

---

## CHANGES (v2 → v3)

### MAJOR

**M4 — the uniform two-hole is not padding-transparent in its direction θ̂. Fixed.**

- **Accepted, reproduced [V].** `check_c16_boundary.py`: padded output 1.3e3. `check_garbage_theta_hat.py`: 1.5e3 and
  1.6e3. `c19`: the mutation that skips the θ̂ mask gives 2.6e3.
- **What changed:**
  - The uniform `corewise_hessian` (kind override and `ut3_*_corewise_*hessian` wrappers) masks **θ and θ̂ on entry**,
    using the base point's own real-extent masks: `(frame_data[5][0], frame_data[5][2]) = (um, lm)`, the frame's
    up/left masks. That also resolves n13: θ takes its masks from the frame, as the lift does.
  - `c23`: those masks equal the plain point masks at minimal points and are a **subset** of them at the TT-slack
    non-minimal point (frame left `[1,4,9,1]` vs point `[1,9,9,1]`). Applying the point masks would leave real-looking
    slots that are frame padding, so the frame masks are the right set.
  - `ut3_apply_masks` broadcasts a `K+C` direction against `C`-only masks [V `c23`].
  - With θ̂ masked, real-block error is **5.5e-17**; the mutation gives **2.6e3** (`c19`, real `UniformTuckerTensorTrain`
    layout with every edge padded, boundary bonds included).
  - The wrapper also **re-masks its output** (one multiply), so the curvature term never puts garbage into the CG state.
    This is stricter than today's uniform COREWISE GN Hessian, which leaves 2.7e5 there [E].
  - Padded *output* is otherwise garbage don't-care (1.3e3 before re-masking), as for the existing one-hole (2.2e2).
- **F15 restated:** the real block is exact when θ and θ̂ are masked on entry, boundary bonds included. Padded output is
  don't-care, re-masked by the wrapper. Unmasked θ̂ is a real-block error.
- **Tests:** T17 gains the boundary-slot θ̂ garbage case on `UNIFORM_COREWISE` (where θ̂ is the raw CG direction), plus
  a mutation check that removing the θ̂ mask fails it (`testing_strategy.md:143`).
- **Where:** C.6, C.10, F15, F18, T17.

### MINOR

| # | resolution |
|---|---|
| m23 | **Fixed.** In S4, `_UniformKind.has_corewise_hessian = False` (a `ClassVar`). It shadows the ragged `True` through the MRO, which is verified as `[UniformProbeDerivativesKind, _UniformKind, ProbeDerivativesKind, …]`. Each uniform class sets `True` in the slice that adds its override (S8 for probe_derivatives, S10–S12 for the rest). New isolated-rejection test, T23: a uniform kind without its override raises the structural error, with a mutation check (drop the shadow and the test fails). |
| m22 | **Fixed, measured** [V `c21`]. The floor is tested **only on the rejection branch**, so it never ends a productive step. First check: `'objective_floor'` iff `|pred| ≤ 10·ε_mach·(‖ω⊙y‖·‖ω⊙r‖ + |f|)`, which is f's evaluation resolution, computed from `kind.sumsq` and so geometry- and T3-agnostic. `c21` measured the `‖ω⊙y‖·‖ω⊙r‖` term; the `|f|` term covers a regularizer's ρ, only raises the threshold, and changes neither classification below. Second check: `'trust_region_floor'` iff `Δ < 1e-12·Δ0` (Δ and Δ0 share HS units, so the m17 units point is satisfied with no ‖X‖ term). On E2's degenerate case: three accepted steps (ρ 1.04, 0.96, 0.79; `|pred|/res` 15, 9.3, 3.2), then at the first rejection `|pred|/res = 1.5` → **`objective_floor`**, f = 7.7e-25. On a near-stationary misspecified start: the first rejection has `|pred|/res = 4.9e-4` → `objective_floor`. The v2 rule had stopped E2's case after one productive step with the wrong label. |
| m27 | **Fixed.** The exact frontend `gradient` is `_tangent_at(self.frame, curvature.misfit_gradient)` plus `g_R` — no second projection. T18's bitwise assertion now holds by construction. |
| m24 | **Adopted as recommended** (also listed as I-22): `lift_damping` becomes a **field** of `ManifoldGeometryOps` and `UniformManifoldGeometryOps` (`lift_damping: float = DEFAULT_LIFT_DAMPING`, value-hashed like `groups`), per `parameters_not_closures.md` "parameters are FIELDS". `DEFAULT_LIFT_DAMPING` lives in `backend/geometry.py`; `backend/optimizers.py` never imports it, so the invariant "imports no T3-specific module" holds (its imports today: `common`, `regularization`, `corewise`, `optimizers.py:30-40`). `curvature_precompute(frame, misfit_gradient, aux=None)`, `Problem.local_model(…, hessian=)` and backend `newton_cg(hessian=…)` carry no damping. The frontend `newton_cg(lift_damping=None)` / factories / `_setup(…, lift_damping=None)` (a defaulted trailing kwarg, so `d08_profile_ops.py:113` is unaffected) set it with `dc.replace`. `with_sharing` uses `dc.replace`, so the field survives tying. The absolute ε that changes every Newton step stays the `LiftFactors.eps` **leaf**. Old m15b is replaced by "ε changes across Newton steps with one compile" (covered by the one-cache-key test); a changed `lift_damping` is a new aux, recompiled once, which is acceptable because it is constant per run. |
| m25 | **Decided, no public-surface change in this feature** (I-18). v2's "public equivalent" claim is withdrawn. `fitting.ragged_backend_geometry` stays out of `__all__` (`fitting.py:86-88`) and `topt._geometry_ops` stays as it is. The downstream dependency — T3Polynomial calls it at `t3polynomial/fitting.py:108` and `scripts/d07:64` — is recorded in the HANDOFF backlog as a promotion candidate. A curvature PR is the wrong place to widen the API. The alternative (add it to `__all__` + an API-page entry + a CHANGELOG "Added" line) is listed for Nick. |
| m26 | **Partly accepted.** Dropped the corewise η and dx̃i from the recompute recipe. They were computed and never used; the derivative needs only the dotted ones. Verified by `c22` (the v3 recipe, ≤ 1.0e-15, orders 0–3, `K = (2,)` and `(2,3)`, `C = ()` and `(2,)`, d = 4/3/1). **Pushed back on `ν_c = Tν` in the recompute default** (below). E's cost rows are now measured (`c25`). |
| m28 | **Fixed, measured on T16's own problem** [V `c20`]. With target rank `(3,3,3)` the start is already stationary (`‖g0‖ = 9.6e-7`), so it is a floor case, not a rate case. T16 uses a target of `(4,4,4)`. Measured there: GN linear ratios 0.56–0.68 (seeds 1, 2); exact tail `e_{k+1}/e_k² ≤ 0.92` inside `[1e-8, 1e-1]`; ρ ∈ [1.00, 1.26] on those steps. One pair leaves the window at the float floor (e = 1.2e-9 → 2.4e-13, constant 1.6e5), so the window's lower end is 1e-8, not 1e-10. **Pinned:** `C = 2`, GN band `(0.3, 0.9)`, `|1−ρ| < 0.3`. All are ≥ 2× the observed values; they are re-pinned run-and-paste on the implemented code in S5. |

### NITs

All are noted, and each is either fixed in the text or deferred to implementation:

- **n9:** F13 now cites `c22` (orders 0–3, multi-axis K, d = 1) alongside the critic's extra checks.
- **n10:** `hessian=` and `step_control=` are lower-cased before validation (HANDOFF ruling "geometry/kind strings are
  case-insensitive"; precedents `t3toolbox/optimizers.py:199`, `backend/optimizers.py:699`).
- **n11:** test labels corrected to T19–T24.
- **n12:** on a shared geometry `lift_sigma_min` includes the group spectra `s_g`.
- **n13:** resolved under M4.
- **n14:** on rejected TR rows the display reuses the previous row's table, so there is no validation forward for an
  unchanged x.

### What I pushed back on, and why

**`ν_c = Tν` in the recompute default (part of m26).** The identity is a **manifold-frame** fact: T is the geometry's
bond factor, the zipper of the orthogonal L against R.

- At the corewise frame `(U,G,G,G)` the stored ν already runs through G, and applying "T" there is wrong by 6.2
  relative [V `c26`].
- So a **geometry-agnostic** kind cannot use the transform without the geometry handing T into the kind — a new
  geometry→kind channel through `SamplingKind.corewise_hessian`.
- The benefit is small: the ν re-sweep through G is **2.4–3.5 % of the two-hole** (`c25`, d = 4–5, ranks 6–8,
  order 2), against 0.2–0.3 % for the transform.
- `precompute_and_caching.md:50-52`: "Creating a NEW state channel for a sub-percent win is where to stop" — here a
  ~1–2 % win of the whole exact matvec.
- The transform stays where a channel exists anyway: the store variant (I-11), whose saving was measured as 13–15 % of
  the two-hole.

### New evidence this round

| # | finding | script |
|---|---|---|
| F18 | the M4 fix (θ̂ masked with the frame masks) on the real uniform layout: 5.5e-17; mutation 2.6e3 | `c19` |
| F19 | **E2 on T3Polynomial's kind:** probe_derivatives order 2 with ω weight, zero-padded restart rank 2→3 (target 4), 5% noise. GN+Armijo **39** steps / 130 hvps; exact+TR **8 / 23, 0 rejected** (ε = 0 ≡ 1e-6‖X‖); ε = 1e-2‖X‖ 9 / 36. Supports I-8 on the primary kind | `c24` |
| F20 | probe_derivatives cost, numpy CPU, dense-trs prototype: exact/GN matvec **3.1× – 3.4×**; ν re-sweep 2.4–3.5 % of the two-hole; undotted tildes 13–15 % | `c25` |
| F21 | the v3 TR floor rule on both degenerate cases | `c21` |
| F22 | T16 calibration on T16's own problem | `c20` |
| F23 | the frame up/left masks ⊆ the point masks (equal except TT slack); K+C mask broadcast | `c23` |

Earlier rounds' changes are recorded in `SCR/plan_v2.md` §CHANGES. All of them are carried forward here.

---
## Verified facts this plan rests on

| # | claim | script | result |
|---|---|---|---|
| F1 | `T_i := ⟨L_i, H_i⟩ = tt_zipper_right_to_left(L, R)[i+1]` for `i ≤ d−2`; centers `H_i = L_i T_i`; `S_iᵀ := einsum('axb,aub->xu', O_i, H_i) = V_i^X U_iᵀ`; σ(S_i) is the mode-i Tucker spectrum and σ(T_i) the bond-(i+1) TT spectrum | `c1` | ≤ 3e-12; the spectra match the dense unfoldings |
| F2 | Lift identity `tv_to_dense(t3_corewise_frame(U, L), Lv) == tv_to_dense(frame, v)`, gauged and ungauged, minimal frame | `c2`, `c3`, `c4` | 6.0e-15, 1.2e-15; per slot ≤ 3.3e-15 |
| F3 | The same on non-minimal frames: Tucker slack (`Sᵀ` wide, 2×4) and TT slack (`T_1` tall, 9×4). The pinv is an exact one-sided inverse on every `t3_orthogonal_representations` frame (`nD ≤ nU`, `rR ≤ rL`) | `c7`, `c8`, `c12` | 9.4e-16, 9.2e-16 |
| F4 | Zero-padded `resize`: the guarded pinv is finite. At X = 0 the lift is identically 0, except the undamped last TT slot | `c7` | finite; 0.0 |
| F5 | **Block-sum oracle** (no new code): `K_c θ̂ = Bᵀ t3_*_corewise_transpose(c, sample, tv_to_t3(t3_corewise_frame(θ), θ̂))`, vs central FD of the one-hole; symmetric | `c2`, `c9` | 1.1e-9 (FD-limited), all six kinds; ⟨a,Kb⟩ = ⟨b,Ka⟩ to 13 digits |
| F6 | Exact = GN + ⟨θ̂,K_{Aᵀr}θ̂⟩ − ⟨θ̂,K_{grad f}θ̂⟩, vs FD along `tv_retract` (minimal, TT-SVD-conditioned point, d = 4) | `c4` | 252.85288420 vs 252.85288182 (h = 2.5e-3); GN 285.03 |
| F7 | Lagrangian null space: `K_N L p_vert = 0` | `c4`, `c10` | 1.3e-11 vs `‖K_{Aᵀr}L p_vert‖ = 1.3e4` |
| F8 | Under the orthogonal Π, `hvp(p_vert) ≠ 0` (`‖Πp_vert‖/‖p_vert‖ = 0.61`), so DP §4.6's wording is wrong | `c10` | — |
| F9 | apply two-hole from existing kernels; the lean sweep is identical at both frames | `c5` | 4.7e-16 |
| F10 | probe two-hole and T3–T3 two-hole from existing kernels (+1 new string), C = (2,) | `c6` | ≤ 4.4e-16 |
| F11 | apply_derivatives two-hole (order 2), C = (2,) | `c13` | ≤ 2.5e-16 |
| F12 | `len_*` sufficiency of the new strings | `c11`, `c17` | §D |
| **F13** | **probe_derivatives two-hole.** The v3 recipe is fed from the manifold model's stored sweep. Existing jet kernels plus one 6-term adjoint step, a 6-term `dx̃i` dot and 2+2+9-term assemblies; **no corewise η or dx̃i**; ν re-swept through G. Covers orders 0–3, `K ∈ {(2,), (2,3)}`, `C ∈ {(), (2,)}`, d ∈ {4, 3, 1}. `c22.two_hole_v3` is the S3 template | `c22`, `c14`; critic `check_c14_extra.py`, `check_c14_d1.py` | ≤ 1.0e-15 vs F5 (re-sweep ≤ 4.1e-16) |
| **F14** | **Shared exact**: sharing `(0,0,1)`, probe_derivatives order 2. The tied lift `U̇_g = M_g⁺[V_i]` (replicated over the group) satisfies `DΦ[Lv] = Ev`. Exact quadratic vs FD of f along the **shared** retraction | `c15` | lift 5.3e-16. Exact 1058.19431605; FD at h = 2e-2…2.5e-3 errs by 1.1e-4, 2.8e-5, 6.9e-6, 2.0e-6 (O(h²)). GN 1058.0325 |
| **F15** | Padding: with θ **and θ̂ masked on entry** (boundary bonds included, the real uniform layout), the real block is exact. The padded *output* is don't-care (1.3e3) and the wrapper re-masks it. An **unmasked** θ̂ corrupts the real block | `c16`, critic `check_c16_boundary.py`, `check_garbage_theta_hat.py`, `c19` | real block 1.4e-16 / 5.5e-17; unmasked θ̂: 1.6e3 / 2.6e3 |
| F16 | At the **manifold** frame the corewise sweep and adjoint state are T/S transforms of the manifold ones (`ν_c = Tν`, `η_c = Sη`, `σ̃_c = Tσ̃`, `dx̃i_c = S dx̃i`, apply `σ̂_c = Tσ̂`). At the corewise frame `ν_c = Tν` fails by 6.2 | `c18`, `c26` | ≤ 9.5e-16 |
| F17 | `‖X‖² = Σσ(S_0)² = Σσ(T_i)²` | `c18` | ≤ 1.4e-15 |
| **F18** | M4 fix on a real `UniformTuckerTensorTrain` (every edge padded): θ and θ̂ masked with the frame's `(um, lm)` → real block 5.5e-17; mutation (θ̂ not masked) → 2.6e3 | `c19` | — |
| **F19** | **E2 on probe_derivatives** (order 2, ω = `[1, .5, .25]`, zero-padded restart rank 2→3, target 4, 5% noise, W = 120): GN+Armijo 39 steps / 130 hvps; **exact+TR 8 / 23, 0 rejected** (ε = 0 ≡ 1e-6‖X‖); ε = 1e-2‖X‖ 9 / 36 | `c24` | — |
| **F20** | probe_derivatives cost (numpy CPU, dense-trs prototype, order 2, d = 4–5, n = r = 6–8, W = 300–400): exact/GN matvec **3.1–3.4×**; ν re-sweep 2.4–3.5 % of the two-hole (T-transform 0.2–0.3 %); undotted tildes 13–15 % | `c25` | — |
| **F21** | v3 TR floor rule (m22): E2's degenerate restart → 3 accepted, then `objective_floor` at the first rejection (`|pred|/res = 1.5`); a near-stationary misspecified start → `objective_floor` at the first rejection (`4.9e-4`) | `c21` | — |
| **F22** | T16 calibration on T16's own problem (target `(4,4,4)`, rank-2 fit, W = 200, 1% noise, 15 GN steps): GN ratios 0.56–0.68; exact `e_{k+1}/e_k² ≤ 0.92` in `[1e-8, 1e-1]`; ρ ∈ [1.00, 1.26] | `c20` | — |
| **F23** | Uniform frame up/left masks equal the point masks at minimal, varying-rank-C and Tucker-slack points, and are a **strict subset** at TT slack (`[1,4,9,1]` vs `[1,9,9,1]`). `ut3_apply_masks` broadcasts K+C directions | `c23` | — |

### The critic's experiments [E]

| # | claim | script |
|---|---|---|
| E1 | Exact matches FD to 5–6 digits at a small bond σ = δ ∈ {1e-2, 1e-3, 1e-4}. The model's validity radius is about σ, and the Hessian is indefinite there | `check_small_sigma*.py` |
| **E2** | Zero-padded restart, apply, d = 3: ε ∈ {0, 1e-6‖X‖} gives **13 steps / 69 hvps, 0 rejected**, vs GN+Armijo 54 and GN+TR 52 / 352. ε = 1e-2‖X‖: 17 / 173. Δ-coupled ε: 56 / 634, not converged. Over-ranked: 18 / 117 vs 80, not converged. 30% noise: 11 / 53 vs 32 | `check_restart_tr*.py` |
| E3 | float32 cancellation is benign (≤ 2e-4 vs GN at δ = 1e-4) | `check_float32_cancellation.py` |
| E4, E5 | `N ≡ 0` at d = 1 and at maximal rank (7e-16, 8e-16) | `check_d1_normal.py`, `check_full_rank_N0.py` |
| E6 | Uniform `MANIFOLD` d = 1 optimizers already raise (`utv_retract → utv_to_ut3`) | — |
| E7 | Today's uniform `COREWISE` GN Hessian leaves 2.7e5 in θ's padded boundary slots; the one-hole leaves 2.2e2 in padded output | `check_onehole_boundary_leak.py` |

---

## A. Scope and non-goals

**In scope**, in build order (H):

1. The lift `L` and its adjoint `Lᵀ`, plus the damped filter. Ragged, uniform and shared; C and K stacks.
2. The T3–T3 two-hole `K_{grad f}`, in `t3_linalg` / `ut3_linalg`.
3. Two-hole corewise sweeps `K_c θ̂` for all six kinds, **`probe_derivatives` first**. Each is a standalone backend
   function plus `SamplingKind.corewise_hessian`, behind the `has_corewise_hessian` capability flag. On the uniform
   layer the flag is `False` until that kind's own masking override exists.
4. `LocalModel` exact mode (`hessian='exact'`):
   - compile-once;
   - one misfit transpose per Newton step;
   - static-safe companions;
   - damping as a geometry field.
5. Steihaug–Toint CG plus a trust region in `newton_cg`, a TR-aware display, and `stop_reason`.
6. Frontend: `GaussNewtonModel`, the factories, and `optimizers.newton_cg` wiring; `_bgeom → from_frame`.
7. Minimal diagnostics: Δ, accept, CG boundary, `pKp`, real-slot `σ_min` (with `s_g` when shared), and ε.
8. Tests, docs, CHANGELOG, and the research-repo measurement plan (S0a / S0b).

**Non-goals.**

- No AD through the T3-SVD (L3).
- The jvp and polynomial-FD routes are oracles only.
- No preconditioner, LM damping or NL2SOL β.
- No N-slot-norm diagnostic.
- No curvature for `gradient_descent` / `mc_sgd` / `adam`.
- No stacked *optimization* (`_require_unstacked`, `optimizers.py:269-298`). The C/K stacks live at the model level.
- No nonlinear-residual Hessian term (T3Polynomial's `Σ rᵢ∇²rᵢ`).
- No `fit(...)` facade.
- No public-surface change beyond the feature itself. `ragged_backend_geometry` stays private (I-18).
- **d = 1 on MANIFOLD** is not an error: the exact model is the GN model, since `N ≡ 0` (I-12).
- **No change to rank continuation.** Restarts need nothing beyond the f(0) = 0 clip and the TR (E2, F19).

---

## B. Math-to-code map

Notation: `X = Φ(θ)`; frame `(U, O, L, R)` = `T3Frame.data`; gauged coordinates `v = (V_i, H_i)`;
`Π = tv_orthogonal_gauge_projection` (`tv_operations.py:83-137`).

| math | code (new unless cited) | shapes (per C element) |
|---|---|---|
| raw gauge `θ = (U, L)` | `geom.base_point(frame)` (`geometry.py:183-189`, `:274-276`, `:378-380`, `:483-485`) | — |
| corewise frame of θ | `t3_corewise_frame` (`fv_conversions.py:30-41`); `ut3_corewise_frame` (`ufv_conversions.py:122-137`) | — |
| real extent of θ (uniform) | `(frame_data[5][0], frame_data[5][2]) = (um, lm)` — ⊆ the point masks (F23) | host bool |
| `T_i` | `tt_zipper_right_to_left(L, R)[i+1]` (`tt_operations.py:192-199`), `i ≤ d−2` | `(rL_{i+1}, rR_{i+1})`, `rR ≤ rL` |
| `H_i^X = L_i T_i`, `S_iᵀ = ⟨O_i, H_i^X⟩` | `fv_center_factors` (extracted from `sharing.py:546-554`) | `(rL_i,nU_i,rR_{i+1})`, `(nD_i,nU_i)`, `nD ≤ nU` |
| `‖X‖_HS` | `√Σσ(S_0)²` (F17) | — |
| filter `f(σ) = σ/(σ²+ε²)` with clip, `f(0) = 0` | `linalg.filtered_reciprocal`, `linalg.damped_pseudo_inverse` | — |
| `L`: `U̇_i = (S_iᵀ)^{+ε}V_i`, `Ġ_i = H_i ×₃ T_i^{+ε}` (`i ≤ d−2`), `Ġ_{d−1} = H_{d−1}` | `tv_corewise_lift` | `(nD,N)→(nU,N)`; `(rL,nU,rR)→(rL,nU,rL')` |
| `Lᵀ` (Frobenius adjoint) | `tv_corewise_lift_transpose` | inverse shapes |
| `K_c θ̂ = d/ds ∇_θ⟨c, S(Φ(θ+sθ̂))⟩` | `t3_{probe,apply,entries}_corewise[_derivatives]_hessian[_from_sweep]` | raw `K+C` |
| `⟨θ̂,K_cθ̂⟩ = ⟨c, S(D²Φ[θ̂,θ̂])⟩`, `D²Φ = Σ_{a≠b}` ordered | normalisation F6, E5 | — |
| `Z = grad f` as a T3 | `tv_to_t3(frame, g_misfit)` (`tv_operations.py:199-330`) / `utv_to_ut3` | doubled rank |
| `∇_θ⟨Z, Φ(θ)⟩` (one-hole) | `t3_inner_product_corewise_transpose` (`t3_linalg`) | raw `C` |
| `K_Z θ̂` (two-hole) | `t3_inner_product_corewise_hessian[_from_sweep]` (`t3_linalg`) | raw `K+C` |
| Riemannian Hessian | `LocalModel.hvp` (exact) = `Π[𝒥ᵀ𝒥Πp + Lᵀ K_N LΠp] + H_R p` | `K+C` |
| COREWISE exact | `GN_corewise + K_{Aᵀω²r}`; no lift, no gradient term | — |
| shared lift | `_tied_filtered_solve` (refactor of `sharing.py:644-664`) | — |
| identity regularizer | `H_R = λΠ` is already exact. The curvature uses the **misfit** gradient only | — |

- **Lift exactness.** On every frame from `t3_orthogonal_representations` the lift is exact (F2, F3): `S_iᵀ` is wide
  or square and `T_i` is tall or square, so the pinvs are exact one-sided inverses at numerically full-rank points.
  A wide `T` arises only on hand-built frames, where the lift is least-squares (caveat, C.12).
- **Gauge laws** (F7, F8): `K_N L p_vert = 0`, `hvp(p) = hvp(Πp)`, and `Π hvp = hvp`.
- **Open structures** (E4, E5): at d = 1 and at maximal ranks `N ≡ 0`. That is the non-circular oracle of T20.

---

## C. File-by-file changes

Signature conventions:
- one argument per line;
- three aligned columns;
- one return element per line;
- `len=d, elm_shape=…`;
- one-shot argument order, with the sweep appended (n1).

Every signature is expanded at implementation (n5). Docstrings cite the local `t4s.pdf` App. A and external references
(Absil–Mahony–Trumpf 2013; Psenka–Boumal arXiv:2011.13395; Steihaug 1983; Nocedal–Wright Alg. 4.1, 7.2), with no `dev/`
paths.

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
        lift_damping:   float,                  # RELATIVE: eps = lift_damping * ‖X‖_HS, ‖X‖² = Σ σ(S_0)² (F17);
        #                                         the caller passes its geometry FIELD (m24), never a module default
        rcond:          typ.Optional[float] = None,
) -> LiftFactors:
```

- Registered as a jax pytree with arrays as leaves and no aux (`sharing.py:1081-1091` precedent).
- The relative `lift_damping` is a plain argument here, because this is a backend function on raw data. In the
  optimizer path it comes from the geometry's `lift_damping` field (C.7, m24).
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

### C.4 `backend/sampling_derivatives.py` — the `probe_derivatives` two-hole (PRIMARY; F13)

```python
def t3_probe_corewise_derivatives_hessian(
        ztildes:          typ.Sequence[NDArray],  # residual jets (ω² applied), len=d, elm_shape=(order+1,)+W+C+(Ni,)
        ww:               typ.Sequence[NDArray],  # len=d, elm_shape=W+(Ni,)
        pp:               typ.Sequence[NDArray],  # len=d, elm_shape=W+(Ni,)
        core_pair:        typ.Tuple,              # θ, stack C
        core_directions:  typ.Tuple,              # θ̂, elm_shape=K+C+(...)
        order:            int,
        chunk_size:       typ.Optional[int] = DEFAULT_CHUNK_SIZE,
) -> typ.Tuple[
    typ.Tuple[NDArray, ...],                      # raw dU, elm_shape=K+C+(nUi, Ni)        (W summed)
    typ.Tuple[NDArray, ...],                      # raw dG, elm_shape=K+C+(rLi, nUi, rL(i+1))
]:

def t3_probe_corewise_derivatives_hessian_from_sweep(
        ztildes:          typ.Sequence[NDArray],  # as above
        ww:               typ.Sequence[NDArray],
        pp:               typ.Sequence[NDArray],
        core_pair:        typ.Tuple,
        core_directions:  typ.Tuple,
        order:            int,
        frame_sweep:      typ.Tuple,              # the model's (xi, mu, nu, eta)_jets; only xi and mu are reused (F9)
        chunk_size:       typ.Optional[int] = DEFAULT_CHUNK_SIZE,
) -> typ.Tuple[typ.Tuple[NDArray, ...], typ.Tuple[NDArray, ...]]:
```

The composition is exactly `c22.two_hole_v3` (F13; the `'resweep'` path).

1. **Undotted state at the corewise frame (recompute default, I-11).**
   - `xi_j`, `mu_j` are reused from the model sweep (identical at both frames, F9).
   - `nu_j = compute_nu_jets(G, xi_j, trs)` is re-swept through `G`. It is 2.4–3.5 % of the two-hole (F20). The
     manifold-only `ν_c = Tν` transform is not used, because it would need a geometry→kind channel (CHANGES pushback,
     F16).
   - **No corewise η and no corewise dx̃i** (m26): the derivative needs only the dotted ones.
   - The undotted tildes come from the existing `compute_deta_tilde_jets` (`:1440`), `compute_tau_tilde_jets`
     (`:1605`) and `compute_sigma_tilde_jets` (`:1611`), at `(U, G, G, G)`.
2. **Dotted forward**, using existing kernels:
   - `xid_j = build_input_jets(compute_xi(Udot, ww), compute_xi(Udot, pp))` (`:258-277`);
   - `mud_j = compute_sigma_jets(Gdot, G, G, …)` (`:1002`);
   - `nud_j = compute_tau_jets(Gdot, G, G, …, nu_j)` (`:1033`);
   - `etad_j = compute_deta_jets(Gdot, G, G, mu_j, nu_j, mud_j, nud_j, trs)` (`:919`).
3. **Dotted adjoint-up lift.** `detd = 'tWCo,KCio->tWKCi'(z̃, Udot)`, a new string.
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
   | 2 | `trs[:, :s2, :]` | `carry · Ṗ · xi` | the core (`len_C`) |
   | 3 | `trs[:, :s2, :]` | `carry · P · xid` | `xid` |
   | 4 | `trs` | `edge_dot · P · deta` | the edge |
   | 5 | `trs` | `edge · Ṗ · deta` | the core |
   | 6 | `trs` | `edge · P · detd` | `detd` |

   - Production keeps `_adj_tilde_step`'s memory shape: `prop` as a fused two-term reverse recurrence, `src` as the
     inner order-scan with a `W·r²` peak. The `_src_step` closure comment carries over (n6).
   - Ragged and uniform scan-body variants follow the `:1563-1582` pattern.
   - Wrappers: `compute_tau_tilde_tangent_jets` and `compute_sigma_tilde_tangent_jets` (the latter by `tt_reverse`).
   - **Order of implementation:**
     1. The `c22` dense-`trs` form lands first, as `_trs` reference twins.
     2. The scan form lands next and is tested equal to the twins.
5. **Dotted `dx̃i`.** Six terms: the product rule of `_dxi_tilde_jets_step`'s two terms (`:1619-1629`).
6. **Assemblies.**
   - Tucker: `'tWKCa,tWCo->KCao'(etad, z̃)` + `'uWKCa,uWo->KCao'(dxtd, w_jets)`.
   - TT: 9 terms, over the three `_tt_variation_jets_trs_core` strings (`:1748-1777`).
   - Both are W-chunked through `_wchunked_sum` (`:1908-1930`), with new module-level chunk functions
     `_tt_tangent_chunk_summed` / `_tucker_tangent_chunk_summed` (stable objects for the memoized `_wchunked_sum_step`,
     `:1866-1885`).
   - S3 measures the peak with `memory_analysis`. If it exceeds 1.5× the GN assembly, `estimate_chunk_size` gains a
     `hessian=` argument, which `_resolve_chunk_size` (`t3toolbox/optimizers.py:145-166`) passes.

**Padding.** This polymorphic function is **mask-free by contract**: it assumes masked θ and θ̂ (F15). Masking is the
uniform wrapper's job (C.10). Its docstring says so, in the voice of `utv_sampling._mask_once`
(`utv_sampling.py:65-72`).

---

### C.5 Plain probe (`probing.py`), apply/entries (`apply.py`, `entries.py`), apply/entries derivatives

All are as in v2, arguments per n1:

- **Probe** (F10): two `_sigma_step` calls per dotted tilde step, and `compute_deta` ×2 for the dotted `dxi_tilde`.
  - **m26 applies here too:** no corewise η and no corewise `dxi_tilde`; ν is re-swept through G.
  - New strings: `'WCo,KCio->WKCi'`, `'WCo,WKCa->KCao'`.
- **Apply/entries** (F9):
  - `compute_sigma_hat` through G (the adjoint's right context; there is no ν/η);
  - then `compute_dxi`, `compute_sigma(Gdot,G,G,…)` and `compute_tau(Gdot,G,G,…, sigma_hat)`;
  - then the three-term assemblies.
  - Entries uses `_entry_xis` / `_onehot_vectors` (`entries.py:264-309`).
- **Apply/entries derivatives** (F11): one new scan body, `_sigma_hat_tangent_jets_step`, plus the 3 + 3 dotted
  assemblies.

**Rollout.** These land after the T3Polynomial path (H). Until then `has_corewise_hessian = False`, so exact mode
raises a structural error that names the kind.

---

### C.6 `backend/fitting.py` and `backend/uniform_fitting.py` — the `SamplingKind` capability

On `SamplingKind` (`fitting.py:305-412`):

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

- `ProbeDerivativesKind` (`:627-652`) implements it first, in S4, passing `chunk_size=self.chunk_size`. The other
  kinds follow in S10–S12.
- **m23 — the uniform shadow.** In the same S4 commit, `_UniformKind` (`uniform_fitting.py:69-91`) declares:

  ```python
      has_corewise_hessian:  typ.ClassVar[bool] = False   # SHADOWS the ragged True through the MRO
      #   (_UniformKind precedes the ragged kind: UniformProbeDerivativesKind -> _UniformKind -> ProbeDerivativesKind);
      #   each uniform class re-sets True in the slice that adds its masking override (S8, S10-S12)
  ```

  A uniform exact model with no override therefore fails `Problem.local_model`'s structural check. It never runs the
  ragged method unmasked. T23 pins this, with a mutation check.
- The uniform override (S8) is in C.10.
- This widens the open `SamplingKind` surface. The note goes in `dev/OPEN_QUESTION_extension_surface.md` §"What to
  watch" (`:131`), which stays open (I-14).

---

### C.7 `backend/geometry.py` — the Geometry protocol (`optimizers.py:60-98`) and the damping field

There are four exact-mode members, needed only for `hessian='exact'`. **None carries damping** (m24).

```python
    def curvature_precompute(
            self,
            frame,              # this geometry's frame
            misfit_gradient,    # Π 𝒥ᵀω²r (tied when shared) -- the model's ONE misfit transpose
            aux=None,           # geom.precompute(frame) (SharedFrameData when shared)
    ):                          # -> per-frame curvature data; None iff the curvature term VANISHES here
    def lift(
            self,
            frame,
            variations,         # gauged frame coordinates, K+C
            curvature_data,
    ):                          # -> raw θ̂ at base_point, K+C
    def lift_transpose(
            self,
            frame,
            core_variations,    # raw, K+C
            curvature_data,
    ):                          # -> UNGAUGED frame coordinates, K+C
    def tangential_curvature(
            self,
            frame,
            core_directions,    # raw θ̂, K+C
            curvature_data,
    ):                          # -> raw K_{grad f} θ̂ (K+C), or None (static: COREWISE)
```

**The damping is a field (m24, I-22).**

```python
DEFAULT_LIFT_DAMPING = 1e-6          # relative to ‖X‖_HS; E2, F19 (module constant HERE, never in optimizers.py)

@dc.dataclass(frozen=True, eq=False)
class ManifoldGeometryOps(ValueHashedFields):
    groups:        typ.Tuple[typ.Tuple[int, ...], ...] = ()
    lift_damping:  float = DEFAULT_LIFT_DAMPING     # value-hashed like groups: constant per run -> compile once
```

- `UniformManifoldGeometryOps` gets the same field, defaulted and appended after `groups`; `from_point` and `from_frame`
  pass it through.
- `with_sharing` uses `dc.replace`, so the field survives.
- **Field, not argument** (`parameters_not_closures.md`, "parameters are FIELDS"): the relative damping is a parameter
  of the geometry, constant per run. The per-step absolute `ε = lift_damping·‖X‖_HS` is the `LiftFactors.eps` leaf.

**`ManifoldGeometryOps`** (`:146-236`):

- **d = 1:** `curvature_precompute` returns `None`; the term vanishes because `N ≡ 0` (I-12).
- **Otherwise** it returns `ManifoldCurvatureData(lift_factors, z, z_sweep, sigma_min)`:
  - `lift_factors = fv_lift_factors(frame, self.lift_damping)`;
  - `z = tv_to_t3(frame, misfit_gradient)` (F14);
  - `z_sweep = t3_precompute_inner_product_sweep(z, base_point)`;
  - `sigma_min` is the real-slot minimum over `tucker_svals ∪ bond_svals`, plus `s_g` when `groups` is set (n12).
- **With `groups`**, the lift factors gain the per-group filtered tied factors (C.11).

**`CorewiseGeometryOps`** (`:239-308`):

- `curvature_precompute` returns `()`, so the curvature term is present with no data.
- `lift` / `lift_transpose` are the identity; `tangential_curvature` returns `None`.
- There is no damping field; nothing to damp.

**`ManifoldCurvatureData`** is `@dc.dataclass(frozen=True, eq=False)`, registered via `partition_static`
(`common.py:698-724`; M3).

- Its flatten is `partition_static((lift_factors, z, z_sweep, sigma_min))`, and `aux` is the value-hashed
  `StaticSkeleton`.
- The uniform `z`'s host masks and shape ints are static.

---

### C.8 Uniform-specific geometry member

**`UniformManifoldGeometryOps.from_frame(frame_data, plain_masks, sharing, lift_damping=DEFAULT_LIFT_DAMPING)`** builds
`var_masks = ufv_variation_masks(frame_data[5])` without orthogonalising.

- X13 / m13 is a performance fix: it removes the re-orthogonalisation per uniform model build.
- It is its **own commit, S6a**, asserting identical masks to `from_point`.

---

### C.9 `backend/optimizers.py` — `CurvatureState`, `LocalModel`, `Problem`, Steihaug, trust region

```python
@dc.dataclass(frozen=True, eq=False)
class CurvatureState:
    """The per-model curvature companion (exact mode): a LEAF beside sweep / geom_aux (the per-frame channel,
    docs/contributor/precompute_and_caching.md scope ladder). Registered via partition_static (statics -> aux)."""
    misfit_gradient:  typ.Any   # Π 𝒥ᵀω²r -- computed ONCE per Newton step; LocalModel.gradient reads it (M2)
    geometry_data:    typ.Any   # geom.curvature_precompute(...): None => the curvature term vanishes (MANIFOLD d = 1)
```

**`LocalModel`** (`:105-161`) gains `curvature: typ.Any = None`.

- Exact mode is `curvature is not None`. The term is present iff `curvature.geometry_data is not None`. Both are static
  at trace time.

```python
    @property
    def gradient(self):                # exact: curvature.misfit_gradient (+ g_R); bit-identical to the GN path
    def hvp(self, p):                  # Π[𝒥ᵀ𝒥Πp + Lᵀ K_N LΠp] + H_R p
    def curvature_hvp(self, p):        # Π Lᵀ K_N L Π p   (raises if not exact)
    def quadratic(self, p):            # gn_quadratic(p) + curvature_quadratic(p)
    def curvature_quadratic(self, p):  # geom.inner(Πp, Lᵀ K_N LΠp)  (keeps K+C)
    # gn_quadratic / jacobian / retract: UNCHANGED (X1; T3Polynomial d07/d08 rely on gn_quadratic = ‖Jp‖²)
```

The curvature body:

```python
    cd = self.curvature.geometry_data
    base = self.geom.base_point(self.frame)
    th = self.geom.lift(self.frame, Pp, cd)
    K = self.kind.corewise_hessian(th, self.residual, self.sample, base, self.sweep)  # uniform: masks θ, θ̂ (M4)
    Kt = self.geom.tangential_curvature(self.frame, th, cd)
    if Kt is not None:
        K = cw.corewise_sub(K, Kt)
    y = self.geom.lift_transpose(self.frame, K, cd)
```

`hvp` projects `𝒥ᵀz + y` once.

**`Problem.local_model(x_cores, sample=None, data=None, hessian='gauss_newton')`** (`:190-201`):

1. `hessian = str(hessian).lower()`, validated against `{'gauss_newton', 'exact'}` (n10).
2. The GN build is unchanged.
3. Exact mode:
   - the `kind.has_corewise_hessian` structural check;
   - `g = geom.project(frame, kind.transpose(…), aux)` — the only misfit transpose;
   - `cd = geom.curvature_precompute(frame, g, aux)`;
   - `curvature = CurvatureState(g, cd)`.
4. **No damping argument (m24).** The default path is unchanged for T3Polynomial (`t3polynomial/fitting.py:110`,
   `d07`, `d08`).

**Pytree.** `_flatten_local_model` (`:799-801`) adds `m.curvature` to the `partition_static` tuple, and the unflatten
passes it by keyword. `CurvatureState` registers `partition_static((misfit_gradient, geometry_data))`.

**Steihaug–Toint in the existing body** (I-5). The state grows to `(p, r, d, rs, i, ok, lm, tol2, maxiter, radius,
boundary)`.

- `boundary0 = rs0 < 0.0`.
- The body is as v2: all `where`, no inf arithmetic, closure-free, radius in state.
- At `radius = inf` it is bit-identical on the eager numpy path; under jit, equal to tolerance.
- `_cg_core(local_model, rhs, tol, maxiter, use_jit, radius)` is compiled with `_jitted(_cg_core, (4,))` and returns
  `(p, i, rs, ok, boundary)`. `_cg_solve(…, radius=math.inf)` likewise.
- Blast radius: `optimizers.py:730`, `tests/backend/test_optimizers.py:218` and `:223`, `tests/test_dispatch.py:595`.
- Prose to update: `scan_body_principles.md:190`, `parameters_not_closures.md:111`.

**`newton_cg`** (`:646-770`), new keywords appended:

```python
        hessian:          str = 'gauss_newton',              # 'exact' (case-insensitive, n10)
        step_control:     typ.Optional[str] = None,          # None -> 'line_search' (GN) / 'trust_region' (exact)
        tr_radius0:       typ.Optional[float] = None,        # None -> GN Cauchy length ‖g‖³ / gᵀH_GN g
```

- No `lift_damping` here (m24). A backend user sets it on the geometry: `dc.replace(geom, lift_damping=…)`.
- `cg_forcing_power: typ.Optional[float] = None` resolves to 0.5 (GN) or 1.0 (exact).
- The `g0norm_newton` / `g0norm_cg` chain (`:717-718`) is untouched. T3Polynomial's zero-referenced
  `g0norm_newton` + `gtol_rel=1e-3` (`t3polynomial/fitting.py:189-194`) composes unchanged.

Loop changes:

1. `lm` is rebuilt only after an accepted step.
2. `Δ0 = tr_radius0`, or `‖g‖³ / max(gn_quadratic(g), tiny)`, falling back to `‖g‖` if that is not finite.
   `Δ_max = 1e4·Δ0`.
3. `p, i, rs, ok, hit = _cg_solve(..., radius)`.
4. Descent guard: `p = −g·min(1, Δ/‖g‖)` in TR mode.
5. `pHp = lm.quadratic(p)`; `pKp = lm.curvature_quadratic(p)` (exact mode only).
6. The line search is unchanged, with ρ taken from the model's `pHp`.
7. **TR acceptance.** One trial at α = 1. `ρ = Δf/pred`, with `pred = slope + ½pHp < 0`; accept iff ρ > 0.1.
   - ρ < 0.25 or non-finite: `Δ ← ¼‖p‖`.
   - ρ > 0.75 and `hit`: `Δ ← min(2Δ, Δ_max)`.
8. **Floor (m22, F21), tested only on the rejection branch** (after Δ is updated), in this order:
   1. `res = ε_mach·(‖ω⊙y‖·‖ω⊙r‖ + |f|)`, with `‖ω⊙y‖ = √kind.sumsq(data, n_w)` (computed once per run) and
      `‖ω⊙r‖ = √(2·misfit)`. If `|pred| ≤ _TR_RESOLUTION_FACTOR·res` (`= 10`): stop, `stop_reason =
      'objective_floor'` — the model change is below f's evaluation resolution.
   2. Else, if `Δ < _TR_RADIUS_FLOOR_REL·Δ0` (`= 1e-12`): stop, `stop_reason = 'trust_region_floor'`.
   - An accepted step is never a stop condition, so a small but productive Δ0 runs. Δ and Δ0 share HS units.
   - The objective-floor stop is reported with `converged=False` and `stop_reason='objective_floor'`. It is not a
     failure; the docs say so.

**TR bookkeeping** (I-17, unchanged from v2):

- Rejected steps count toward `max_newton` and emit a row with `tr_accepted=False` and the unchanged `x_cores`.
- `losses[k] == history[k]['objective']`.
- `stats['newton']` counts CG solves; `stats['newton_accepted']` counts accepted steps.
- `stats['stop_reason']` ∈ `'gradient' | 'max_newton' | 'line_search' | 'trust_region_floor' | 'objective_floor'`.
- T3Polynomial's `stats['newton']` consumers (`t3polynomial/fitting.py:353`, `:513`;
  `tests/test_fitting_selection.py:67-91`) keep their meaning.

**`NewtonInfo`** (`:602-635`), new fields appended, all defaulting to `None`:

- `hessian`, `tr_radius`, `tr_accepted`, `cg_boundary`, `pKp`, `lift_sigma_min`, `lift_eps`, `stop_reason`.
- The two strings (`hessian`, `stop_reason`) are documented in the `history`-row docstring (n4).
- `cg_truncated = (not cg_converged) and (not cg_ok) and (not cg_boundary)`.
- In TR mode `ls_steps = ls_failed = None` and `alpha = 1.0`.

---

### C.10 Uniform mirror

- **`ufv_lift_factors`** (`ufv_operations.py`):
  - mask once, then the polymorphic zipper and `'d...'` einsums;
  - padded σ = 0 gives f = 0;
  - ε from `Σσ(S_0)²`;
  - real-slot `sigma_min` via host masks: `xnp.where(real_mask, σ, inf)`, with `s_g` included when shared.
- **`utv_corewise_lift` / `_transpose`:**
  - mask on entry;
  - the lift output carries `ufv_variation_masks((um, um, lm, lm))` from `frame_data[5]` and is re-masked;
  - the transpose output carries `ufv_variation_masks_over_stack(frame_data[5], C, K+C)`.
- **`ut3_inner_product_corewise_*`** (`ut3_linalg.py`): Z and the direction θ̂ are masked first (M4 applies here too:
  θ̂ is the direction).
- **Sampling wrappers** (`ut3_sampling.py`), e.g. `ut3_probe_corewise_derivatives_hessian` (**M4**):

  ```python
  def ut3_probe_corewise_derivatives_hessian(
          ztildes:          typ.Sequence[NDArray],   # packed or ragged; (order+1,)+W+C+(Ni,)
          ww:               typ.Sequence[NDArray],
          pp:               typ.Sequence[NDArray],
          data:             UT3Data,                 # θ: (tk_sc, tt_sc, shape, (tucker_mask, tt_mask)) -- the REAL extent
          core_directions:  typ.Tuple[NDArray, NDArray],   # θ̂ supercores, (d,)+K+C+(...)
          order:            int,
          chunk_size:       typ.Optional[int] = sampling_derivatives.DEFAULT_CHUNK_SIZE,
  ) -> typ.Tuple[NDArray, NDArray]:            # (dU, dG) supercores, (d,)+K+C+(...), RE-MASKED with data's masks
  ```

  The body, in order:
  1. `masked = ut3_apply_masks(data)`, as the existing `ut3_probe_corewise_derivatives_transpose` does
     (`ut3_sampling.py:239-256`).
  2. **`masked_dir = ut3_apply_masks((core_directions[0], core_directions[1], data[2], data[3]))`** — the direction
     masked with θ's own masks; `K+C` against `C`-only masks broadcasts (F23).
  3. Pack.
  4. Delegate to the polymorphic d-prefixed kernel.
  5. Re-mask the output with the same masks.
- **Kinds** (`uniform_fitting.py:69-233`): `corewise_hessian(v_sc, r, sample, base_sc, sweep)`.
  - The point and its masks: `(base_sc[0], base_sc[1], self.shape, (um, lm))`, with `(um, lm) = frame_data[5][0],
    frame_data[5][2]` from the masked frame `mb` in the sweep (`utv_sampling.py:238-281`). This is the **frame's**
    real extent (n13, F23). It equals `self.masks` at the minimal points `uniform_minimal` guarantees, and is a strict
    subset at TT slack, where `self.masks` would leave real-looking slots that are frame padding.
  - The direction is `v_sc`, masked with the same `(um, lm)`.
  - Each uniform class sets `has_corewise_hessian = True` here, in S8 for probe_derivatives (m23).
- **Where θ̂ garbage comes from** (so the mask is not optional):
  - On `UNIFORM_MANIFOLD`, θ̂ is the re-masked lift output.
  - On **`UNIFORM_COREWISE`** the lift is the identity, so θ̂ is the raw CG direction, assembled from `−g` and `Hd`.
    Today's uniform corewise GN Hessian leaves 2.7e5 in θ̂'s padded boundary slots [E7].
  - `geom.inner` masks, so CG's scalars are unaffected; without M4's mask the curvature term would be silently wrong.
- **Varying-rank C:** factors, ε and masks are all per element; masked-out directions lie in `ker L ∩ ker Lᵀ`.

---

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
- **Diagnostics (n12):** `lift_sigma_min` includes `min(s_g)` over the nontrivial groups. Those values gate the tied lift.
- **Oracle:** the shared retraction's second order is **measured** (F14, O(h²) on probe_derivatives order 2), not
  proven. T12 uses FD along the shared retraction as a measured oracle, alongside the dense Lagrangian of Φ∘T.

### C.12 Safe mode — preconditions vs caveats

**No new enforced preconditions.**

- The frontend `hessian` / `curvature_hessian` / `quadratic` / `curvature_quadratic` take `_require_at_frame`
  (`fitting.py:101-118`).
- Orthogonality stays a construction invariant of the model.
- The backend stays check-free.

**Caveats** (documented, never enforced):

1. The Hessian is exact at ε = 0 on numerically full-rank points. It is damped toward GN for σ ≲ ε (default
   `1e-6·‖X‖`).
2. A hand-built wide `T_i` gives a least-squares lift.
3. Shared: exact where `s_g > 0`.
4. On MANIFOLD at d = 1 and on maximal-rank structures, `N ≡ 0`, so exact = GN; at d = 1 this is structural (E4, E5).
5. **New:** `objective_floor` is a resolution limit, not a convergence certificate. The model change fell below f's
   floating-point resolution (F21).

---

### C.13 Frontend (`t3toolbox/fitting.py`, `t3toolbox/optimizers.py`, `t3toolbox/manifold.py`)

**`GaussNewtonModel`** (`:184-403`):

- Field `curvature: typ.Any = None`, appended. The registration (`:748-754`) adds it as a child, and its own
  `partition_static` flatten keeps statics static (M3). The positional unflatten is updated.
- **`gradient`** (`cached_property`, `:329-340`). Exact mode returns **`_tangent_at(self.frame,
  curvature.misfit_gradient)`** plus `g_R`, exactly as today's tail (m27). It does not use `_wrap`, which would
  re-project (`:293-294`). So it is bitwise equal to the GN-mode gradient, which is `_wrap(transpose)`: one Π, the
  same Π that `Problem.local_model` applied.
- New methods: `hessian`, `curvature_hessian`, `quadratic`, `curvature_quadratic`. `evaluate` becomes mode-aware.
- `gn_hessian` / `gn_quadratic` / `objective_value` are unchanged (T3Polynomial `x06`/`x07`).
- The name stays `GaussNewtonModel` (I-4).

**Factories** (`:534`, `:554`, `:574`, `:633`, `:684`, `:707`) and `_uniform_model` (`:454-488`):

- They gain `hessian='gauss_newton'` (case-insensitive) and `lift_damping: typ.Optional[float] = None`.
- One helper, `_with_curvature(model, hessian, lift_damping)`:
  - builds the backend geometry (`ragged_backend_geometry`, or `from_frame` on uniform);
  - applies `dc.replace(bgeom, lift_damping=lift_damping)` when the value is not None and the geometry is a manifold
    one. On COREWISE a non-None `lift_damping` raises a `ValueError`: nothing to damp.
- Downstream payoff: `probe_derivatives_model(..., hessian='exact').hessian(p)` replaces `x07`'s FD Hessian
  (`x07_hessian_free_newton.py:102-106`).

**`optimizers.newton_cg`** (`:318-439`):

- Explicit `hessian=` and `lift_damping: typ.Optional[float] = None` (m24); `step_control` / `tr_radius0` ride `**kwargs`.
- **`_setup(…, lift_damping=None)`** is a defaulted trailing kwarg, so `topt._setup(geom, 'probe_derivatives', …,
  chunk_size=None)` at `d08:113` is unaffected. After building the backend geometry it applies the field via
  `dc.replace` and puts the geometry into the `Problem`. `_geometry_ops` is unchanged.

**`manifold.py:1381-1386`:** the retract docstring adds "second order on a minimal, numerically full-rank frame".

---

### C.14 Diagnostics (`backend/optimizer_display.py`)

`_format_header` (`:121-144`):

- **TR rows.** When `info.tr_radius is not None`, the `ls %d…` segment is replaced by `"TR Δ %.2e %s ‖Δx‖/‖x‖ %.1e"`.
  Otherwise the `ls` segment is unchanged.
- **CG symbol:** `✓`, `∂` (boundary), `⌇` (truncated) or `⋯` (maxiter).
- **Exact rows** append `" | κ %s σ %s"`:
  - κ = `pKp/(pHp − pKp)`, guarded;
  - σ = `lift_sigma_min`, with `_cell` semantics.
- **`stop_reason`** appears on the final row: `objective_floor` / `trust_region_floor`.
- **GN output is byte-identical.**
- **Rejected TR rows (n14):** `make_newton_display` (`:194-242`) reuses the previous row's error tables when
  `info.tr_accepted is False`, so there is no validation forward for an unchanged x.
- **Tests:** `test_exact_mode_header` and a run-and-paste doctest.

---

## D. Batching contract (frame-inner `W + K + C`)

| function | inputs (stacks) | output | machinery |
|---|---|---|---|
| `filtered_reciprocal`, `damped_pseudo_inverse` | s C+(k); M C+(m,n); eps C | C+… | elementwise, batched SVD |
| `fv_center_factors`, `fv_lift_factors` | frame C | C | `'...'`, zipper scan |
| `tv_corewise_lift[_transpose]` | frame/factors C; variations K+C | K+C | `'...'` (C innermost, so it broadcasts over K for free) |
| `t3_*inner_product*` | Z, θ, sweep C; θ̂ K+C | K+C | `'...'`, tangent-zipper scan |
| `t3_probe_corewise_derivatives_hessian_from_sweep` | jets `(o+1,)+W+C`; dotted `(o+1,)+W+K+C`; input jets `(2,)+W[+K]+C`; z̃ `(o+1,)+W+C+(N,)` | K+C (W summed, chunked) | grouped `contract` with `trs` |
| `ut3_*_corewise_*hessian` (M4) | θ C masks; θ̂ `(d,)+K+C+(...)` masked with θ's masks | K+C, re-masked | `ut3_apply_masks` (`'d...'`), then delegate |
| plain / apply / entries two-holes | θ̂ K+C; c/z̃ W+C; ww W; θ C | K+C | grouped `contract` |
| `LocalModel.hvp / curvature_hvp` | p K+C | K+C | composition |
| `LocalModel.curvature_quadratic` | p K+C | K+C | `geom.inner` → `corewise_stack_dot` (never `corewise_dot`) |

- Per-frame data carries **C only**.
- The residual is W+C with no K; K enters only through θ̂.
- K placement follows `_sigma_step` (`probing.py:363-377`).

**New `HISTORICAL` rows** (`tests/test_contractions_interpreter.py:70`): 20 new, each with the analog's letter (`c17`).

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

- Each is sufficient on the test's generic point (`c17`). The uniform d-prefixed twins use the same letters (S8).
- The scan-form kernels record their own strings in the commit they appear (`TestVocabularyCompleteness`, `:297-310`).

**W-chunking.** Only the probe-derivatives tangent assembly is chunked; it engages on uniform + jax only, with ≥ 3
chunks. The `shard_map` + `psum` recipe applies unchanged (`chunking.md`).

---

## E. Costs and memory

Measured where marked [M], estimated otherwise. Final numbers come from S3, S8 and S0b, recorded in
`verification_records.md`.

| operation | cost | notes |
|---|---|---|
| GN matvec (apply), per sample | `2dNn + 6dnr²` | — |
| apply exact / GN | ≈ 2× (N-dominated) – 3.5× (r-dominated) | estimate |
| probe exact / GN | ≈ 3–4× | estimate; ν re-sweep ~2–3 % of the two-hole (by analogy with F20) |
| **probe_derivatives exact / GN** | **[M] 3.1–3.4×** (numpy CPU, dense-trs prototype, order 2, d = 4–5, n = r = 6–8, W = 300–400; F20) | the scan-form kernel may differ, and S3 re-measures. The ν re-sweep is 2.4–3.5 % of the two-hole; the undotted tildes 13–15 % |
| `K_{grad f}` (T3–T3, doubled rank) | m-independent `O(d[Nn·n_Z + n²r² + nr³])` | — |
| lift / `Lᵀ` / factors | m-independent; `d` small SVDs per Newton step | — |
| per Newton step | + Z build + T3–T3 sweep; **no** extra transpose (M2) | — |

**Memory.**

- **Recompute default:** no new resident W-scaling state. The transients are the dotted edge variables, of size
  `(o+1)·W·K·r` each (K = 1 in CG), plus the re-swept ν_c (`(o+1)·W·r` per core).
- **Store variant** (I-11): stores ν_c (by the T transform, F16 — free given the channel) and the three undotted
  tildes, at about +60 % of the full jet sweep. The saving is 13–15 % of the two-hole (F20). The channel cost is a
  `SamplingKind.curvature_precompute` member plus a `CurvatureState` field.
- **Compile time** for the exact uniform_jax CG body is measured in S8.

---

## F. Tests (numpy-only numerics; jax in dispatch / the jvp oracle)

New file `tests/backend/test_curvature.py`.

- Test pattern:
  - `subTest` over structures × stacks: none, C, K, K+C, multi-axis C, multi-axis W, multi-axis K;
  - primitive subTest params;
  - per-test seeding.
- Structures:
  - minimal well-conditioned;
  - Tucker slack `(6,6,6),(4,4,4),(1,2,2,1)` and TT slack `(1,9,9,1)`;
  - d ∈ {1, 2, 4};
  - rank-1 bonds;
  - zero-padded `resize`;
  - the zero point;
  - maximal rank `(3,3,3)` / `(2,3,4)`.
- Helper `tests/backend/_curvature_oracles.py`:
  - the block-sum oracle (F5);
  - the dense Lagrangian Hessian;
  - `c22.two_hole_v3` as the dense-`trs` reference. It is re-written there seeded, and never ports the old unseeded
    `c2` block (m1).

| test | asserts | oracle |
|---|---|---|
| T1 factors | `T_i == zipper` (`i ≤ d−2`), `S_iᵀ == V_iU_iᵀ`, spectra == dense unfoldings; `Σσ(S_0)² = ‖X‖²` | F1, F17 |
| T2 lift identity | minimal + both slack kinds, gauged + ungauged, ε = 0 | F2, F3 |
| T3 lift adjoint | `⟨Lv, y⟩_raw == ⟨v, Lᵀy⟩` per K+C element, several ε | — |
| T4 damping | ε → 0 continuity; finite at a padded restart; at the zero point the lift is 0 except the last slot | F4 |
| T5 two-holes | every `*_corewise_[derivatives_]hessian` + T3–T3 vs **block-sum** and vs FD of the one-hole; probe_derivatives at orders 0–3, K = (2,3), d = 1 | F5, F13 |
| T6 symmetry | `⟨a,Kb⟩ = ⟨b,Ka⟩`; `⟨p, hvp q⟩ = ⟨q, hvp p⟩` | — |
| T7 retraction FD | `⟨p, hvp p⟩` vs `d²/dt² f(tv_retract(tp))`, Richardson | F6 |
| T8 gauge laws | `K_N L p_vert = 0`; `hvp(p) = hvp(Πp)`; `Π hvp = hvp` | F7, F8 |
| T9 dense Lagrangian | hvp vs the FD Hessian of `f∘Φ − ⟨grad f, Φ⟩` through L | DP §4.6 |
| T10 regularized | exact + `IdentityRegularizer(λ)` vs FD of f + ρ along `tv_retract` | — |
| T11 weights | ω[mode,order] / per mode / per order: the two-hole uses `ω²r` | block-sum |
| T12 shared | tied lift identity; adjoint; exact vs FD along the shared retraction (measured O(h²)); dense Lagrangian of Φ∘T; padded shared restart | F14 |
| T13 corewise exact | `GN + K_{Aᵀr}` vs block-sum; FD Hessian of `θ ↦ f(Φ(θ))` | — |
| T14 Steihaug unit | ToyModel (`test_optimizers.py:201-225`): indefinite → boundary; PD + small Δ → boundary; Δ = ∞ → bit-identical (eager numpy); `cg_truncated` excludes boundary stops | — |
| T15 TR loop | rejection keeps x and reuses the model; radius rules incl. Δ_max; descent-guard scaling; `losses[k] == history[k]['objective']`; `newton_accepted`; **floor (m22, F21):** (a) E2's degenerate restart runs ≥ 1 accepted step and then stops with `'objective_floor'`; (b) a near-stationary start stops with `'objective_floor'` at the first rejection; (c) a forced-tiny `tr_radius0` with accepted steps does **not** stop on the floor; (d) `'trust_region_floor'` only after rejections drive Δ below 1e-12·Δ0 | F21 |
| T16 convergence rate (m28) | **calibrated on its own problem (F22):** `(6,6,6)`, target `(4,4,4)` / `(1,4,4,1)`, fit rank 2, W = 200 apply, 1% noise, start = 15 GN steps, seeds 1 and 2. Exact: ≥ 3 points of `e_k = ‖g_k‖/‖g₀‖` in `[1e-8, 1e-1]` with `e_{k+1} ≤ 2·e_k²` and `|1−ρ| < 0.3` on those steps (observed ≤ 0.92, ρ ≤ 1.26). GN control: ratios in `(0.3, 0.9)` (observed 0.56–0.68). All constants re-pinned run-and-paste on the implemented code in S5 | F22 |
| T17 uniform (M4) | per element vs ragged over `_CONFIGS` incl. varying-rank C, from nonzero full-rank starts; **exact output masks** (lift `(um, um, lm, lm)`; Lᵀ/hvp frame var masks over K; the two-hole wrapper output re-masked with `(um, lm)`); **garbage-padded inputs** `1e6·(1−mask)` in θ, the samples, **and θ̂** — explicitly θ̂'s padded **boundary-bond** slots on `UNIFORM_COREWISE` (where θ̂ is the raw CG direction); `sigma_min` ignores padded slots; **mutation:** removing the θ̂ mask fails the garbage case (`testing_strategy.md:143`) | ragged, F15, F18 |
| T18 frontend | exact `GaussNewtonModel` == backend `LocalModel`; **exact `gradient` == GN `gradient` bitwise** (ragged + uniform; holds by construction after m27); `_bgeom` via `from_frame` == via `from_point`; same-frame guard; `lift_damping` reaches the geometry field (`dc.replace`), and a non-None value on COREWISE raises | backend |
| T19 display (M1, n14) | `test_exact_mode_header` (accepted/rejected TR rows, ∂, κ zero denominator, σ —, `stop_reason`); an exact `make_newton_display` run with `val_data`: a rejected row does not call `point_forward` (counter) | — |
| T20 open structures | maximal rank and ragged d = 1 (backend): `‖Lᵀ K_N LΠp‖ ≤ 1e-12·‖Lᵀ K_{Aᵀr} LΠp‖`; MANIFOLD d = 1 exact model has `geometry_data is None` and `hvp ==` GN (ragged + uniform); COREWISE d = 1 uniform exact runs | E4, E5 |
| T21 restart regime | exact `newton_cg` from a zero-padded `resize` (E2 row 1 template): `stop_reason == 'gradient'`, ≤ 2 rejections; ragged + uniform; slow tier if > 3 s | E2 |
| T22 shared restart | the same on `shared_manifold((0,0,1))`, probe_derivatives order 2 (F19 template) | F19 |
| **T23 capability shadow (m23)** | a uniform kind with ragged `has_corewise_hessian = True` but no uniform override is **rejected** by `Problem.local_model(hessian='exact')` with the structural error; **mutation:** deleting the `_UniformKind` shadow makes the test fail. Checked per uniform kind until its override lands | — |
| **T24 θ̂ masking unit (M4)** | `ut3_probe_corewise_derivatives_hessian` with garbage in θ̂'s padded slots == the same with zeroed padding (bitwise on the real block), every edge padded including boundary bonds | F18 |

**Jax** (`tests/test_dispatch.py`):

- jit every new op, ragged and uniform;
- `test_exact_local_model_rebuilt_each_step_is_one_jit_cache_key` (`:556`) — this also covers "ε changes every Newton
  step with one compile", replacing v2's m15b test now that `lift_damping` is a field (m24);
- `test_jit_uniform_exact_gauss_newton_model` (M3);
- `test_cg_radius_is_not_stale`;
- a geometry rebuilt with an equal `lift_damping` is one cache key;
- the probe_derivatives exact jit with chunking engaged;
- `tests/test_curvature_jvp_oracle.py` (jax-gated, float64).

**Oracle sweeps.** New `tests/oracle_sweeps/sweep_exact.py` in both tiers. The existing files stay byte-close, and their
GN rows are unchanged.

**Research-repo measurement** (never copied in):

- **S0a**, in progress: F19 is its first data point.
- **S0b**, after S6: T3Polynomial `e01`/`o01`/`d05`/`s05`, plus `x06`/`x07` as FD cross-checks.

**Budget.** The always-on additions are small. T16/T21/T22 are ≤ 3 s each, or they move to the slow tier.

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
  - the two-index-jet composition rule (`c22.two_hole_v3`);
  - the T/S transform identities (F16);
  - the companion pytree rule (M3);
  - the misfit-gradient-once rule (M2);
  - decisions as settled;
  - the θ̂-masking rule for uniform direction arguments (M4, F15/F18);
  - E1–E7 and F13–F23.
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
  `['newton_accepted']`, the `lift_damping` field on the manifold geometries (frontend `lift_damping=` kwarg).
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

## H. Ordered slices

Each slice is one commit and is independently green: `pytest tests/ -n auto`, the doctests, and `sphinx -W`.

**Order principle:** T3Polynomial's path goes first. By S9, probe_derivatives runs exact end to end on ragged, uniform
and uniform_jax, both unshared and shared.

| slice | content | gate tests |
|---|---|---|
| **S0a** (research repo) | E2 extended to probe_derivatives (F19 is the first point); ε, the TR constants, Δ0 and the forcing power | numbers go to `verification_records.md`. **Gate:** if S0a has not finished when S4 lands, ship `DEFAULT_LIFT_DAMPING = 1e-6` (E2, F19) and record that |
| S1 | `filtered_reciprocal` / `damped_pseudo_inverse`; `fv_center_factors` (and the `fv_shared_frame_data` refactor, bit-identical); `LiftFactors`, `fv_lift_factors`; `tv_corewise_lift[_transpose]` (ragged); the CHANGELOG `[Unreleased]` header | T1–T4; the sharing tests pass unchanged |
| S2 | move `_inner_product_frame_transpose` to `t3_linalg` (bit-identical); `t3_inner_product_corewise_transpose`; `tt_zipper_tangent_*`; `t3_precompute_inner_product_sweep`; `t3_inner_product_corewise_hessian[_from_sweep]` | T5 (T3–T3), T6, T20 (backend half) |
| **S3** | **the probe_derivatives two-hole, ragged** (the `c22` recipe): `_trs` twins, then the scan form; `_adj_tilde_tangent_step`; the dotted `dx̃i`; the chunked tangent assembly; HISTORICAL rows; the memory measurement | T5 (orders 0–3, K = (2,3), d = 1), T6, T11 |
| S4 | the protocols: `SamplingKind.corewise_hessian` with its flag (ProbeDerivativesKind only), **plus the `_UniformKind` shadow `has_corewise_hessian = False` (m23)**; the Geometry exact members; the **`lift_damping` field** and `DEFAULT_LIFT_DAMPING` in `geometry.py` (m24); `ManifoldCurvatureData` and `CurvatureState`; `LocalModel` exact mode; `Problem.local_model(hessian)` | T7–T10, T13, T20, **T23**; dispatch: exact hvp jit and one cache key |
| S5 | Steihaug body and state; `_cg_core` / `_cg_solve` 5-tuple (and the 3 call sites); the TR loop and **its floor rule (m22)**; bookkeeping; `stop_reason`; the `NewtonInfo` fields; the `cg_truncated` redefinition; the TR-aware display (and n14); `cg_forcing_power=None` | T14–T16 (T16 constants re-pinned here), T19, T21 (ragged); `test_cg_radius_is_not_stale`; existing GN tests and O2 pass unchanged |
| S6a | `UniformManifoldGeometryOps.from_frame`; `_bgeom` → `from_frame` (GN path, its own commit) | T18 (mask equality); uniform GN tests unchanged |
| S6 | frontend: `GaussNewtonModel.curvature`, the new methods, all 7 construction sites and the pytree; `gradient` via `_tangent_at` (m27); `optimizers.newton_cg` / `_setup` `hessian=` / `lift_damping=` | T18; **S0b can start** (ragged) |
| S7 | shared (ragged): `_tied_filtered_solve[_transpose]` (bit-identical), the tied lift, `s_g` in `sigma_min`, shared corewise | T12, T22 |
| S8 | uniform: `ufv_lift_factors`, `utv_corewise_lift[_transpose]`, `ut3_inner_product_*`, **`ut3_probe_corewise_derivatives_hessian` with θ and θ̂ masked on entry and the output re-masked (M4)**, `UniformProbeDerivativesKind.corewise_hessian` (flag → True), the uniform geometries' exact members | T17 (incl. the θ̂ boundary garbage and its mutation), **T24**, T20, T21 (uniform); `test_jit_uniform_exact_gauss_newton_model`; chunked jit; compile time |
| S9 | shared (uniform) | T12 uniform twin + T17 rules. **T3Polynomial path complete** |
| S10 | plain probe (ragged + uniform; flag per class) | T5, T6, T11, T17, T23 |
| S11 | apply + entries (ragged + uniform) | T5, T6, T17, T23 |
| S12 | apply/entries derivatives (ragged + uniform) | T5, T6, T11, T17, T23 |
| S13 | docs, the final CHANGELOG, release notes, doctests, `sweep_exact.py` (both tiers), the jvp oracle, the `dev/` archive and HANDOFF / CLAUDE.md | the full release gate, including `T3TOOLBOX_SLOW_TESTS=1` |

- **S1–S3** are pure additions.
- **S4** is the first consumer change; GN stays the default throughout.
- **S4–S8 window (m23):** a uniform probe_derivatives exact model raises the structural error, and T23 pins that.
- **No release before S13.**

---

## I. Decisions for Nick (recommendation · alternative · critic's verdict)

| # | decision | recommendation | alternative | critic |
|---|---|---|---|---|
| I-1 | S/T source | zipper on demand, once per model | return them from `t3_orthogonal_representations` | Agree |
| I-2 | mode API | `hessian='gauss_newton' \| 'exact'`, case-insensitive | `curvature: bool`; `hessian='newton'` | Defer |
| I-3 | mode encoding | derived from `curvature is not None`; the term is present iff `geometry_data is not None` | an explicit static field | Agree |
| I-4 | frontend name | keep `GaussNewtonModel` | rename + deprecated alias | Pushback accepted; Nick's call |
| I-5 | Steihaug | one CG body; Δ = ∞ ≡ today (bit-identical on eager numpy; equal to tolerance under jit) | a separate body | Agree |
| I-6 | TR policy | TR by default in exact mode; accept ρ > 0.1; ¼‖p‖ / 2Δ on the boundary; Δ0 = GN Cauchy; Δ_max = 1e4·Δ0; **floor only on rejection: `objective_floor` at `|pred| ≤ 10·ε·(‖ω⊙y‖·‖ω⊙r‖ + |f|)`, else `trust_region_floor` at `Δ < 1e-12·Δ0`** (F21); module constants | Armijo for exact too; exposed constants | Agree with m22 (now done) |
| I-7 | forcing power | `None` → 0.5 (GN) / 1.0 (exact) | 0.5 everywhere | Agree |
| I-8 | ε | relative to ‖X‖_HS, `DEFAULT_LIFT_DAMPING = 1e-6`. **E2 (apply) and F19 (probe_derivatives): ε = 0 ≡ 1e-6 is best; 1e-2 is worse** | per-factor scales; the DP's ‖N‖/λ_min | Agree |
| I-9 | default | GN stays the default | exact by default after S0b | Agree |
| I-10 | jet layout | companion arrays | a stacked length-2 axis | Agree |
| I-11 | store vs recompute | recompute per matvec: ν re-sweep 2.4–3.5 %, no new channel; the store variant saves 13–15 % of the two-hole (F20) at about +60 % jet-sweep memory and a new kind member | store per model, with ν_c via the T transform (F16) | Defer |
| I-12 | d = 1 | MANIFOLD d = 1: exact model = GN model, structurally; COREWISE d = 1 runs exact | raise | Agree |
| I-13 | diagnostics | Δ, accept, boundary, pKp, κ, real-slot σ_min (+ `s_g` when shared), ε, stop_reason; defer the N-slot norms | include slot norms | Agree (+ n12, done) |
| I-14 | kind surface | one optional method + flag, with the uniform shadow (m23); note it in the OPEN_QUESTION | close the surface | Agree |
| I-15 | block-sum oracle | test helper only | public reference function | Agree |
| I-16 | S0 timing | S0a before S4 (F19 already in hand); S0b after S6 | implement first, tune later | Agree |
| I-17 | TR bookkeeping | rejected steps count toward `max_newton`; `losses` gets one entry per row; `stats['newton']` = CG solves, plus `newton_accepted` | count only accepted steps | Agree |
| **I-18** (revised, m25) | `ragged_backend_geometry` / `topt._geometry_ops` | **no public-surface change in this feature.** Both stay private; T3Polynomial's dependency (`t3polynomial/fitting.py:108`, `d07:64`) is recorded in the HANDOFF backlog as a promotion candidate | promote `ragged_backend_geometry` now: `fitting.__all__` + API page + CHANGELOG "Added" | Agree in substance |
| I-19 | `stop_reason` | five values; `objective_floor` is a resolution limit, reported as not-a-failure | reuse `converged=True` | Agree (criterion fixed) |
| I-20 | slice order | T3Polynomial path first (S3–S9); the other kinds S10–S12; one release at S13 | kinds in v1 order | Agree (+ m23 window) |
| I-21 | shared centers | do not reuse `SharedFrameData.centers` | add all-mode centers and T to `SharedFrameData` | Pushback accepted |
| **I-22** (new, m24) | where `lift_damping` lives | **a field of `ManifoldGeometryOps` / `UniformManifoldGeometryOps`** (value-hashed; constant per run, so it compiles once); `DEFAULT_LIFT_DAMPING` in `geometry.py`; `backend/optimizers.py` stays free of T3 imports; the frontend sets it with `dc.replace` | thread it as an argument through `curvature_precompute` / `local_model` / `newton_cg` (one more protocol argument for every custom geometry, and a T3 import into `optimizers.py`) | Recommend the field |
| **I-23** (new) | uniform θ̂ masks: frame `(um, lm)` vs kind `self.masks` | the **frame's** `(um, lm)`: the base point's real extent, which equals `self.masks` at minimal points and is a strict subset at TT slack (F23) | the kind's plain masks (correct only at minimal points) | n13 |

---

## J. Risks and open questions

1. **Indefinite, ill-conditioned CG.** The TR bounds the step but not the CG count. S0b measures this; the deferred
   follow-up is a preconditioner or LM. E2 and F19 saw 3–7 hvps per Newton step.
2. **ε.** Two problems agree so far (E2 apply, F19 probe_derivatives). S0a/S0b on T3Polynomial's real fits decide.
3. **probe_derivatives scan-form rewrite** (S3). It is pinned to the `c22` `_trs` twins at 1e-15.
4. **S3 tangent-assembly memory.** Measured in S3; the estimator is extended if it exceeds 1.5× the GN assembly.
5. **float32** is benign (E3).
6. **Shared retraction order** is measured, not proven (F14). S0b measures ρ on `s05_fit.py` (sharing).
7. **Compile time** of the exact uniform_jax CG body. Measured in S8; the fallback is `remat`.
8. **The `_bgeom → from_frame` switch** is isolated in S6a.
9. **`ls_steps` is `None` in TR mode.** `d05` filters it; `b01`'s histogram is a release-note item.
10. **Uniform direction garbage (M4) beyond curvature.** Today's uniform COREWISE GN Hessian already leaves 2.7e5 in
    padded output slots [E7]. That is harmless today, because every consumer masks on entry. The curvature wrappers
    now re-mask their output; whether the GN `utv_*_transpose` wrappers should re-mask too (hygiene) is an open
    backlog item, not part of this feature.
11. **Open:** subsampled exact Newton (mc_sgd) — out of scope.
12. **Open:** composition with the zero-reference g0 plan. It composes; its objective-decrement stop is still wanted.
13. **Open:** `tr_radius0` across continuation levels. The default recomputes Δ0 per level.

---

## Conflicts between the design plan (DP) and the code — consolidated

| # | DP says | code reality | resolution |
|---|---|---|---|
| X1 | §4.3: curvature into `gn_quadratic` *and* Cauchy on it | the same method serves both (`optimizers.py:357`, `:424`, `:737`); T3Polynomial relies on `gn_quadratic = ‖Jp‖²` (`d07:15`, `:124`) | new `quadratic`; `gn_*` keeps its GN meaning |
| X2 | §4.1/§4.3: S/T from the frame construction | the construction discards them; caching docs forbid frame state | zipper inside the curvature leaf |
| X3 | §4.2: the T3 projection at `(U,G,G,G)` is ∇⟨Z,Φ⟩ | it gauges at the end (`tv_operations.py:417-419`) | ungauged kernel moved to `t3_linalg` |
| X4 | §4.2: frame sweep reused | true for xi/mu (F9); probe ν runs through R | re-sweep ν through G (2.4–3.5 %, F20); no corewise η or dx̃i is needed (m26); the T/S transform only in store mode (F16, manifold-only) |
| X5 | §4.2 route 3 "NumPy backend" | `xnp`-agnostic | shipped on all backends |
| X6 | §4.7: jvp first | Nick: the two-hole is the deliverable | block-sum oracle + jax-gated jvp oracle |
| X7 | §4.6: hvp of pure gauge = 0 | false under orthogonal Π (F8) | the three coordinate laws |
| X8 | §4.5: 2–2.5× | apply est. 2–3.5×; probe_derivatives **measured 3.1–3.4×** (F20, prototype) | re-measured S3/S8/S0b |
| X9 | §4.1: S from the "centre-core SVD" | contraction (`sharing.py:552`) | contraction; SVD only to filter |
| X10 | §4.4: ε ~ ‖N‖/λ_min | needs an eigensolve | ε relative to ‖X‖ (E2); Rayleigh proxy logged in S0 |
| X11 | §4.3: residual raw, ω² applied | `LocalModel.gradient` includes `g_R` (`:138-139`) | misfit gradient carried in `CurvatureState` |
| X12 | §4.1: non-minimal → slack carries no curvature | the pinv is an exact one-sided inverse (F3) | caveat only for a hand-built wide `T` |
| X13 | (frontend) | `_bgeom` re-orthogonalises per uniform model build — **performance, not jit-safety** | `from_frame`, own commit S6a |
| X14 | §4.4: "phase-in" via damping at restarts | E2: the f(0) = 0 clip + TR handle restarts; damping is a margin | wording removed; default 1e-6 |
| X15 | (uniform layer) | the existing one-hole wrappers mask only the point θ — there is no direction to mask (`ut3_sampling.py:239-256`) | the two-hole wrappers mask θ **and θ̂**, then re-mask the output (M4) |
