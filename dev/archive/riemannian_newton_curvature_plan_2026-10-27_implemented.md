> **RESOLVED — IMPLEMENTED in 2026.3.0 (slices S0c–S13, 2026-10-25 → 2026-10-27); archived at S13.** This is the
> design plan ("DP") the implementation plan `dev/riemannian_newton_curvature_impl_plan.md` built on: the target it
> proposed — level 2, the exact Riemannian Hessian by a hand-written two-hole sweep — is what shipped, as
> `newton_cg(hessian='exact')` with a Steihaug–Toint trust region, for every sampling kind on every geometry and
> representation. Where it and the code disagreed, the implementation plan's "Conflicts" table (X1–X18) records the
> resolution (notably: DP §4.6's "hvp of pure gauge = 0" is false under the orthogonal Π, F8; the cost is ≈ 3× GN, not
> 2–2.5×; jvp/FD are oracles only). The durable record is `docs/fitting_and_optimization.md` §4.12 (users) and
> `docs/contributor/curvature_internals.md` (contributors); the measurements are in
> `docs/contributor/verification_records.md`. Its dense oracles (`dev/curvature_oracles/`) are archived beside it as
> `dev/archive/curvature_oracles_2026-10-27/` (folded into T7 / T9 and the exact oracle sweep). Line references below
> are to `45c6742` and are historical.

# Curvature in the Newton subproblem — from Riemannian Gauss–Newton to Riemannian Newton on the T3 manifold

> **STATUS: PROPOSED (2026-10-25). Nothing implemented.** Decision with Nick (2026-10-25): the target is
> **level 2 below — the exact Riemannian Hessian — computed by a hand-written "two-hole" sweep** (§4.2),
> with the JAX `jvp` route and the polynomial-FD route (§4.2, routes 1–2) serving as prototypes and test
> oracles, not as the shipped implementation. The mathematics in §2 was derived from first principles
> and verified against dense oracles (`dev/curvature_oracles/`, captured output in its README); the
> code survey in §1 is from four read-only Opus 5.5 agent passes over the repo (geometry/retraction,
> sampling/jets, fitting loop) and the literature (2026-10-25); line references are to `main` at
> `45c6742`. Open questions in §5.

**Notation** (paper ↔ code). Point `X = Φ(θ)` with raw cores `θ = (B_i, G_i)` (Tucker `n_i × N_i`,
TT `r_i × n_i × r_{i+1}`); frame `(U, O, L, R)` = `(up, down, left, right)`; gauged tangent coordinates
`(V_i ∈ ℝ^{nD_i×N_i}, H_i ∈ ℝ^{rL_i×nU_i×rR_{i+1}})`; `E: (V,H) ↦ ξ` the coordinate map; `Π` =
`tv_orthogonal_gauge_projection`; `𝒥` the sampling Jacobian in frame coordinates; `A = ω∘S` the weighted
(linear) sampling operator; `r = ω⊙(S(X) − y)`; `P_X = EΠEᵀ` the tangent projector; `N := (I − P_X)Aᵀr`
the normal part of the Euclidean gradient; `d` modes, `m` samples, `N_i` mode sizes, `n_i` Tucker
ranks, `r_i` TT ranks.

---

## 0. Summary

1. **Exactly one term is missing.** Every sampling kind — entries, applies, probes and their jets — is
   *linear* in the ambient tensor, so the ambient Hessian of `f = ½‖ω⊙(S(X) − y)‖²` is exactly `AᵀA` and
   the Riemannian Hessian is

   ```
   Hess f(X)[ξ] = Π 𝒥ᵀ ω² 𝒥 Π ξ   +   𝔚_X(ξ, N),        N = (I − P_X) Aᵀr.
   ```

   The first term is the current `LocalModel.hvp` (`backend/optimizers.py:152-158`). The second —
   the Weingarten / second-fundamental-form term — is the *entire* non-convexity of the problem: the
   GN model is the convex part, which is why GN "has no negative-curvature escape"
   (`examples/fit_hilbert_tensor_newton_cg.py:53-66`) and stalls at saddles it cannot see.

2. **The "levels" collapse to two.** The quadratic model of `f` in exponential coordinates *is* the
   Riemannian Newton model, and it is also `Hess(f∘R_X)(0)` for *any* second-order retraction `R`. The
   T3-SVD retraction (`tv_retract`, Algorithm 10) **is second order** — proved in §2.3 (a composition of
   exact metric projections) and verified numerically on dense oracles and on `tv_retract` itself. So
   "AD through the retraction" would deliver exactly the Riemannian Hessian — but it is the worst way
   to compute it (§3). Nothing beyond the Riemannian Hessian exists at quadratic order; retractions
   differ only at third order.

3. **The curvature term is the Hessian of a *multilinear* scalar in core coordinates.** For any lifts
   `ξ̂, η̂` of `ξ, η` to raw core variations,

   ```
   ⟨𝔚_X(ξ, N), η⟩ = ⟨N, D²Φ[ξ̂, η̂]⟩,      D²Φ[ξ̂, η̂] = Σ_{a≠b} Φ(…ξ̂_a…η̂_b…)   (double core replacement).
   ```

   No SVD derivatives, no frame derivatives: the matvec is the directional derivative of the existing
   corewise transpose along a core direction — a "two-hole" sweep with the same banded Leibniz
   recurrence as the jets. The only new geometric object is the **lift** from frame coordinates to raw
   core variations (§4.1); it is where the `1/σ_min` of the Weingarten map lives, and the two small
   matrices it needs (`S_i`, `T_i`) fall out of the frame construction for free.

4. **It matters exactly where the measured tails are linear.** The GN iteration's local rate is
   `ρ(H_GN⁻¹ 𝔚(·, Aᵀr*))` — zero for zero-residual fits (the exact-rank test: superlinear) and
   `≈ |1 − ρ_tr|` otherwise, where `ρ_tr` is the actual/predicted reduction ratio the loop already
   computes. The Hilbert runs show `ρ_tr ≈ 1.2` with rate 0.21–0.4 per iteration; T3Polynomial measured
   true curvature 1.95× GN with rate 0.90–0.99 (`dev/zero_reference_g0_plan.md:187-218`); GN *diverges*
   once that ratio exceeds 2. Full Newton makes these tails quadratic.

5. **Cost.** With a dedicated two-hole sweep, ≈ 2–2.5× per CG matvec (the `m`-dependent part roughly
   doubles); nothing new per outer iteration; the level-2 correction is `m`-independent and essentially
   free. The zero-new-code prototype costs ~`2d` corewise transposes per matvec.

6. **Mandatory companions.** Steihaug–Toint CG + trust region (the Hessian becomes indefinite), and a
   phase-in/damping mechanism at rank-deficient frames (zero-padded continuation starts), where the
   curvature term is unbounded (§4.4).

---

## 1. What the library does today (survey, 2026-10-25)

### 1.1 Objective and operators

- `f(X) = ½‖ω⊙(S(X) − y)‖² + ρ(X)`. One sample kind per `Problem`; the weight `ω[mode, order]` enters
  only as `ω` in `sumsq` and `ω²` on the residual in `transpose` (`backend/fitting.py:199-285`,
  `:553-555`, `:594-595`, `:644-646`), so the GN operator is `Π𝒥ᵀω²𝒥Π`. The only regularizer,
  `IdentityRegularizer(λ)` = `½λ‖X‖²`, has Riemannian Hessian exactly `λΠ` because its gradient `λX` is
  tangent and the Weingarten term vanishes (`docs/contributor/fitting_internals.md:76-78`).
- All six kinds are linear in the ambient tensor (`docs/fitting_and_optimization.md:258-267`,
  tested to ~1e-13 in `tests/test_fitting.py:118-129`). The jets are symmetric directional derivatives
  with respect to the *sampling vectors* along `W + sP` (`docs/symmetric_probe_derivatives.tex`,
  `backend/sampling_derivatives.py:30-37`); order `t` is the linear functional
  `⟨T, t!·[sᵗ] ⊗_j (w_j + s p_j)⟩` of `T`. The recurrence is a Leibniz/binomial convolution over the
  order axis, banded for the pushthroughs (`_mu_jets_step` `:358`, `_sigma_banded_step` `:958`,
  `binomial_combine_tensor` `:240`).
- Transposes (`docs/transposes.md`): the **tangent** transpose `tv_*_transpose` returns raw, ungauged
  variations at the frame (`apply.py:172`, `entries.py:139`, `probing.py:1200`; jet versions
  `sampling_derivatives.py:2173/2372/2435`); the **corewise** transpose `t3_*_corewise_transpose` is the
  same code at the frame `(U, G, G, G)` and gives core gradients (`fv_conversions.py:30`); the **ambient**
  transpose gives `Aᵀr` as a CP tensor of rank `m` (apply) / `d·m` (probe) and does not exist for jets
  (`docs/contributor/ambient_derivative_transpose_note.md`). Nothing computes second derivatives of a
  sample map with respect to the point (ragged or uniform).

### 1.2 The `newton_cg` loop (`backend/optimizers.py:646-770`)

1. `frame = geom.frame(x)`: full re-orthogonalization by SVDs (Algorithm 11,
   `fv_conversions.py:123-267`): squash tails → down-orthogonalize Tucker cores → left-orthogonalize
   the chain (`L`) → right-orthogonalize it (`R` + per-slot centre cores `H_i^X` with
   `X = L_{<i} H_i^X R_{>i}`) → SVD each centre core `(rL·rR) × nU`: `H_i^X = O_i S_iᵀ`.
2. `sweep = kind.precompute(frame, sample)` (frame-dependent part of the Jacobian, once per frame).
3. `g = Π 𝒥ᵀ(ω²r) (+ λ v_X)` (`:134-140`); `‖g‖` in the coordinate dot = Hilbert–Schmidt metric.
4. Forcing term `η = min(½, (‖g‖/g0)^{0.5})`, `cg_tol = η‖g‖` (`:727-728`).
5. Plain CG on `Hp = −g` (`_cg_body`/`_cg_core`/`_cg_solve`, `:548-599`, under `xwhile`), from `p = 0`;
   stops on tolerance, `maxiter`, or **non-positive curvature, with a zero step** (`:559-561`). No
   preconditioner, no LM, no trust region.
6. Descent guard (`gᵀp ≥ 0` → `p = −g`, `:734-736`); `pHp = gn_quadratic(p)` (`:146-150`).
7. Armijo backtracking **along the retraction**: `x_trial = lm.retract(αp)`, `f_new = objective(x_trial)`
   (raw-core `S(x)`, no sweep), halving up to 40× (`:738-748`); on exhaustion `'stop'|'accept'`.
8. `ρ = Δf / (α·slope + ½α²·pHp)` computed as a **diagnostic only** (`:750-751`).

### 1.3 Tangent coordinates and metric

- Frame cores: `U_i` (`nU_i × N_i`, orthonormal rows), `O_i` (`rL_i × nD_i × rR_{i+1}`, orthonormal
  `nD × (rL·rR)` unfolding), `L_i` left-orthogonal (`i ≤ d−2`), `R_i` right-orthogonal (`i ≥ 1`)
  (`frame_variations_format.py:150-153`, `:295-334`).
- `ξ = Σ_i L_{<i} H_i R_{>i} (all U) + Σ_i L_{<i} O_i R_{>i} (U_i → V_i)` (`tv_operations.py:45-80`,
  T4S eq. (47)); gauge `U_iV_iᵀ = 0`, `L_iᵀH_i = 0` for `i ≤ d−2` (`:610-648`).
- Each slot map is an isometry (orthonormal environment), so on an orthonormal frame with gauged
  variations the coordinate dot **equals** the ambient Hilbert–Schmidt inner product
  (`docs/frame_variations.md:55-66`). The GN system therefore carries no Gram matrices; the CG metric
  is the ambient metric restricted to `T_X M`.
- `Π` (`:83-137`) is coordinate-orthogonal: `V_i ← V_i(I − U_iᵀU_i)`, `H_i ← (I − L_iL_iᵀ)H_i`; it
  changes the represented vector and is what turns `𝒥ᵀr` into the Riemannian gradient. CG iterates
  stay gauged because the Krylov space generated from a gauged `−g` by a gauged `H` stays gauged.
- `tv_to_t3(frame, variations, include_shift=True)` (`:199-330`, T4S (50)–(53)) builds `X + ξ` exactly as
  a T3 of ranks `(nU+nD, rL+rR)`: Tucker factors `[U_i; V_i]`, TT bonds ordered `[R | L]`, middle core
  `[[R_i, 0], [H_i, L_i]]` on the `nU` leg and `[[0,0],[O_i,0]]` on the `nD` leg. Sampling it costs
  ~8× a point sample because the zero blocks are not exploited.

### 1.4 Retraction (`tv_retract`, `tv_operations.py:651-690`; `t3svd`, `backend/t3_svd.py:35-127`)

`X + ξ` from `tv_to_t3(include_shift=True)`, then `t3svd(max_tucker_ranks=nU, max_tt_ranks=rL)`: squash
tails → down-orthogonalize the Tucker cores by SVD → right-orthogonalize the chain → one left-to-right
truncating sweep: per mode an up-SVD truncated to `nU_i` (rotating the Tucker factor) then a left-SVD
truncated to `rL_{i+1}`, pushed right. The docstring claims first order only (`manifold.py:1383`); the
`test_sharing.py:786-818` "second-order agreement" assertion proves only first order. Measured on the
real `tv_retract` (geometry agent): the tangential part of `R(tv) − x − tv` scales as `t³`
(1.16e-6 / 1.21e-9 / 1.2e-12 at `t` = 1e-3 / 1e-4 / 1e-5), the normal part as `t²` — i.e. second
order; the asymptotic regime starts once `t ≪ σ_min`. Proof in §2.3.

**Differentiability.** The ragged path calls plain `xnp.linalg.svd` with static slicing
(`linalg.py:194, 221-223`); the uniform path uses `pad_safe_svd` (Haar sketch, QR of a rank-deficient
sketch, `argsort`, `σ > c/2` threshold, `where` masks; `:558-715`). No `custom_jvp`/`stop_gradient`
anywhere in the frame or retraction code; the repo already recorded NaN gradients through SVD-based
orthogonalization (`backend/ut3_linalg.py:188-241`). At `t = 0` the doubled representation
`[U_i; V_i]`, `V → 0` has `n` exactly repeated zero singular values, which is precisely where JAX's
`_svd_jvp_rule` (`1/((s_j+s_i)(s_j−s_i))`, masking only the diagonal) returns NaN — confirmed by the
literature agent in JAX 0.12 on the `2r → r` rounding; evaluating at `t = 1e-3` instead gives an
`O(t)`-biased Hessian. **There is no supported way to autodiff through the retraction**, and §3 shows
there is no reason to want one.

### 1.5 Second-order code: none

The only Hessian actions are `LocalModel.hvp = Π𝒥ᵀω²𝒥Π + λΠ` and its frontend twin
`GaussNewtonModel.gn_hessian` (`fitting.py:373-387`). Transport is projective
(`manifold.py:1424-1438`). `dev/zero_reference_g0_plan.md:187-239` records downstream (T3Polynomial)
measurements: true Riemannian curvature 1.95× the GN curvature along the GN direction near
convergence; an FD full-Hessian Newton-CG reached `ρ = 1.00` and converged 10× in 10 iterations;
unpreconditioned CG stalled on the near-singular full Hessian. **One correction to that note:** it says
GN "omits `Σ rᵢ∇²rᵢ`"; for the library's own linear sampling kinds that term is identically zero, and
the *only* omitted term is the Weingarten term.

### 1.6 Measured convergence (fitting agent, Hilbert 16⁴, 800 applies, library `newton_cg`)

- Level 1 (rank 1): `‖g‖` contracts ≈ 0.21× per iteration (1.0e-3 → 1.3e-4 → 2.5e-5 → … → 5.8e-10) with
  `ρ_tr ≈ 1.18–1.21`; CG grows from 6 to 16 iterations as `η → 0`. Levels 2–3: ratio 0.25–0.4,
  `ρ_tr ≈ 1.2–1.3`. In every converging level **the tail rate ≈ |1 − ρ_tr|**.
- Level 4 with 1 % noise: not converged in 30 iterations, `ρ_tr ≈ 0.55–0.9`.
- Exact-rank random target (zero residual): ~40-iteration global phase (`ρ_tr` 0.3–1.47), then
  superlinear (2.8 → 0.5 → 2.4e-2 → 2.6e-4 → 4.0e-7 → 2.1e-11) with `ρ_tr = 1.000`. With 1 % noise the
  tail is linear at ≈ 0.04×.
- Nothing in `tests/` measures convergence *rates*.

---

## 2. The mathematics

### 2.1 The curvature term — three equivalent forms

With `f` quadratic in the ambient tensor (`∇f = Aᵀr`, `∇²f = AᵀA`), the Riemannian gradient is
`grad f = P_X Aᵀr` and (Absil–Mahony–Trumpf 2013)

```
Hess f(X)[ξ] = P_X AᵀA ξ + 𝔚_X(ξ, N),        𝔚_X(ξ, N) := P_X (D_ξ P)(N),
⟨𝔚_X(ξ,N), η⟩ = ⟨N, II(ξ,η)⟩   (symmetric),     II = second fundamental form.
```

**(a) Weingarten / projector-derivative form.** `𝔚(ξ, N) = P_X(D_ξ P) N`. Since `P(DP)P = 0`
(differentiate `P² = P`), the full `Aᵀr` may be substituted for `N`. With `P = EΠEᵀ`, this is
"differentiate the Riemannian-gradient map `X ↦ P_X Z` along `ξ` with `Z` frozen"; it needs
`D_ξ(U, O, L, R)` — first-order SVD perturbation of the frame, with `S_i⁻¹`, `T_i⁻¹` factors. Correct,
but the frame derivatives are delicate. Not the chosen route.

**(b) Parameterisation / second-fundamental-form form (the chosen route).** `Φ` is multilinear in its
`2d` cores. For any lift `θ̂` of `ξ` (raw core variations with `DΦ[θ̂] = ξ`), the curve `Φ(θ + tθ̂)`
lies in `M` with velocity `ξ` and acceleration `D²Φ[θ̂,θ̂] = Σ_{a≠b} Φ(…θ̂_a…θ̂_b…)`. The Gauss formula
says the normal part of the acceleration of *every* curve in `M` with velocity `ξ` is `II(ξ,ξ)`.
Hence, by polarisation,

```
⟨𝔚_X(ξ,N), η⟩ = ⟨N, D²Φ[ξ̂, η̂]⟩      for any lifts ξ̂, η̂   (lift-independent).
```

Equivalently (the literature agent found this form independently): in core coordinates the
Riemannian Hessian is the Hessian of the **Lagrangian**

```
∇²_θ [ f(Φ(θ)) − ⟨grad f(X), Φ(θ)⟩ ]  =  (DΦ)ᵀ Hess f(X) (DΦ),
```

because `∇_θ⟨N, Φ(θ)⟩ = (DΦ)ᵀN = 0` makes `θ` a critical point of the Lagrangian, so its Hessian is
tensorial and the vertical (gauge) directions lie exactly in its null space. Replacing `grad f` by `0`
(i.e. using the full `Aᵀr`) gives `∇²_θ(f∘Φ)` = `(DΦ)ᵀAᵀA(DΦ) + ⟨Aᵀr, D²Φ[·,·]⟩`, which differs from the
Riemannian Hessian by `⟨grad f, P_X D²Φ[·,·]⟩ = O(‖grad f‖)` — level 1 in §3.

**(c) Second-order-retraction form.** For any retraction, `(f∘R_X)''(0)[ξ,ξ] = ⟨Hess f ξ, ξ⟩ +
⟨grad f, P_X R''(0)⟩` (Absil–Mahony–Sepulchre §5.5). So `Hess(f∘R_X)(0) = Hess f(X)` at critical points
for *every* retraction, and everywhere iff `R` is second order (`P_X R''(0) = 0`). §2.3: T3-SVD is.

**Verification** (`dev/curvature_oracles/`): the explicit Tucker formula of §2.2 matches finite
differences of the projector to 2e-9; `⟨N, D²Φ⟩` is symmetric and equals `⟨𝔚(ξ,N), η⟩`;
`‖Aξ‖² + ⟨N, D²Φ[ξ̂,ξ̂]⟩` matches `d²/dt² f(T3SVD(X + tξ))` to 6–7 digits on a d = 4 T3
(87.75289 vs 87.75281; GN alone 80.95); the level-1 form matches `d²/dt² f(Φ(θ + tθ̂))` (85.6376 vs
85.6376).

### 2.2 Structure of 𝔚 for T3

**Tucker layer** (core `G`, orthonormal `U_k`, tangent coordinates `(Ġ; V_k)`, `U_kᵀV_k = 0`). For
normal `N` define the slot contractions

```
N_k          := [N ×_{j≠k} U_jᵀ]_(k)                                  (N_k × Π_{j≠k} n_j)
N_k^⊥        := P_{U_k}^⊥ N_k (I − G_(k)⁺ G_(k))                        single-mode normal slot k
N_{k;l}(V_l) := [N ×_l V_lᵀ ×_{j≠k,l} U_jᵀ]_(k)                         doubly-orthogonal slot (k,l)
```

Then (derived by differentiating the Koch–Lubich projector along a horizontal curve; verified)

```
Ġ_out   = Σ_k fold_k( V_kᵀ N_k^⊥ )                                   (for T3: then project onto T_G M_TT)
V_k,out = Σ_{l≠k} P_{U_k}^⊥ N_{k;l}(V_l) G_(k)⁺  +  N_k^⊥ Ġ_(k)ᵀ (G_(k) G_(k)ᵀ)⁻¹
```

For `d = 2` the single-mode slots are empty (the slot `V_k ↦ V_k G_(k)` fills `U_k^⊥ ⊗ U_l` when
`Π_{j≠k} n_j = n_k`) and this is Vandereycken's `U Σ⁻¹ U_pᵀ N + N V_p Σ⁻¹ Vᵀ`. Reading it: the
Weingarten map couples *different* slots — factor `k` with factor `l` through the doubly-orthogonal
component of `N`, factor `k` with the core through the single-mode component — and every term carries
one inverse Tucker singular value (`G_(k)⁺`, `(G_(k)G_(k)ᵀ)⁻¹`). The normal residual `N` decomposes into
three kinds of slots: doubly-orthogonal (`U_k^⊥ ⊗ U_l^⊥ ⊗ …`), single-mode (`U_k^⊥ ⊗ ⊗_{j≠k} U_j`,
orthogonal to the row space of `G_(k)`), and the core slot's TT-normal part `(I − P^TT_G)(N ×_j U_jᵀ)`.
Their norms tell which rank is binding (§4.4).

**TT layer.** Psenka & Boumal (arXiv:2011.13395) give the TT Weingarten map in a left-orthogonal gauge
with "diagonal" and "cross" terms carrying `R_k⁻¹` — the inverse bond-singular-value matrices, the TT
analogue of `Σ⁻¹` (no Gram inverses). Their `O(d²nr³)` cross-term cost is an artifact of computing
each `(k,l)` pair separately; with prefix/suffix environments carrying "0 or 1 replacement" it is
`O(d)` — the same two-term banded recurrence as `_mu_jets_step`.

**T3.** The full Weingarten map is the Tucker-layer map above with the core slot's output projected
onto `T_G M_TT`, plus the TT-layer map lifted by the `U`'s, plus the coupling that arises because the
core slot's normal component is `(I − P^TT_G)(·)` rather than zero. No published formula exists
(Molozhavenko & Rakhuba, arXiv:2508.20928, prove the ETT/shared-factor manifold is smooth and give
projections and the Novikov-style AD gradient, but stay first order). **With route (b) none of this
has to be assembled by hand**: the double-replacement sum *is* the formula. §2.2 is kept for
intuition, for the cost bound `‖𝔚(·,N)‖ ≲ ‖N‖/σ_min` (σ_min over Tucker singular values `S_i` and bond
singular values `T_i`), and for a possible future "partial blocks" variant (§3).

### 2.3 The T3-SVD retraction is second order

Algorithm 10 (`t3svd`) applied to `X + tξ` is a composition `Π_K ∘ ⋯ ∘ Π_1` of truncations, each of which
is an **exact metric projection** onto a variety `𝒱_j = {Y : rank of one unfolding ≤ k_j}`: at every
stage the environment of the active core is isometric (left-orthogonal prefix, right-orthogonal
suffix, orthonormal Tucker factors, down-orthogonalized active factor), so the SVD of the small active
core *is* the SVD of the full unfolding and truncating it is Eckart–Young for that unfolding. Each `𝒱_j
⊇ M` and is smooth at `X` (`X` has exactly the ranks `k_j` on a minimal frame), and metric projections
onto smooth manifolds are second-order retractions (Absil–Malick 2012).

Induction: suppose `c_k(t) := Π_k∘⋯∘Π_1(X + tξ) = X + tξ + (t²/2) a_k + O(t³)` with `a_k ∈ N_X M`
(true for `k = 0` with `a_0 = 0`). Since `Π_{k+1}` is smooth near `X`, `DΠ_{k+1}(X) = P_{T𝒱_{k+1}}`, and
`ξ ∈ T_X M ⊆ T_X 𝒱_{k+1}`,

```
Π_{k+1}(X + tξ + (t²/2) a_k) = X + tξ + (t²/2)[ II_{k+1}(ξ,ξ) + P_{T𝒱_{k+1}} a_k ] + O(t³).
```

`II_{k+1}(ξ,ξ) ∈ N_X 𝒱_{k+1} ⊆ N_X M`, and `P_{T𝒱_{k+1}} a_k ∈ T𝒱_{k+1} ⊖ TM ⊆ N_X M` (decompose
`a_k ∈ N_X M = (T𝒱_{k+1} ⊖ T M) ⊕ N𝒱_{k+1}`). So `a_{k+1} ∈ N_X M`, and `R''(0) = a_K ⊥ T_X M`. ∎

(The same argument covers HOSVD / ST-HOSVD on the Tucker manifold and TT-SVD on the TT manifold.
For standard HOSVD an explicit expansion also works: the tangential parts of `2Σ_k ξ ×_k P_k'` and
`Σ_k X ×_k P_k''` cancel, using the second-order singular-subspace perturbation
`P_U^⊥ U'' = −2 V_k Ġ_(k) G_(k)⁺` in the horizontal gauge.)

Numerics: dense oracle, tangential acceleration `∝ h²` (Richardson → 1e-10, d = 4, minimal format);
on the real `tv_retract`, `t³` scaling of the tangential part (§1.4). Heidel & Schulz (2018) wrote of
HOSVD that their numerics "can be seen as an indication that the retraction by HOSVD preserves
second-order information although we cannot prove this" — the composition argument is the proof.

**Consequences.** (i) `Hess(f∘tv_retract)(0)` is the exact Riemannian Hessian, so FD of `f` along
`tv_retract` is an *exact* oracle for the new `hvp` (§4.6). (ii) Predicted vs actual reductions of the
level-2 model agree to `O(‖p‖³)`, so trust-region theory (Absil–Baker–Gallivan RTR) applies verbatim
with `tv_retract`. (iii) The parameterisation "retraction" `Φ(θ + θ̂)` (add the lifted variations,
re-orthogonalize later) is only first order — its tangential acceleration is `P_X D²Φ[θ̂,θ̂] ≠ 0`
(measured constant 0.137 on a unit tangent) — which is exactly the level-1/level-2 difference.

### 2.4 Why GN stalls, quantitatively

At a minimizer `X*`, `grad f = 0`, so `Aᵀr*` is entirely normal and the GN fixed-point iteration has
`e_{k+1} ≈ (I − H_GN⁻¹ Hess f) e_k = −H_GN⁻¹ 𝔚(·, Aᵀr*) e_k`: local linear rate

```
ρ_GN = spectral radius of H_GN⁻¹ 𝔚(·, Aᵀr*)   ≲  ‖Aᵀr*‖ / (σ_min · λ_min(H_GN)),
```

zero residual ⇒ quadratic (Breiding & Vannieuwenhoven's `cκ²‖F(x*)‖` bound, arXiv:1708.02488, is this
with the curvature hidden in `c`; their "Sensitivity of low-rank matrix recovery" Lemma 1 makes
`‖H⁻¹‖ = max_i |1 − ‖N‖λ_i|⁻¹` with principal curvatures `λ_i` explicit). The loop's `ρ_tr` *measures
it*: near convergence with `α = 1` and the GN step `p` (`gᵀp = −pH_GNp`), the predicted reduction is
`−½pH_GNp` and the actual one is `−½pH_GNp − ½⟨p,𝔚p⟩ + O(‖p‖³)`, so

```
ρ_tr − 1 ≈ −⟨p, 𝔚p⟩ / ⟨p, H_GN p⟩,        tail rate ≈ |1 − ρ_tr|.
```

`ρ_tr ≈ 1.2` (Hilbert) means the true curvature along the GN step is 0.8× GN, the steps are 20 % short
and the error contracts by 0.2 — matching the measured 0.21. The downstream 1.95× case gives
`e_{k+1} ≈ −0.95 e_k` (oscillating, rate 0.95 — matching their 0.90–0.99); at 2× GN diverges. So the
expected gain can be predicted from existing logs before anything is built (§4.7, stage 0).

---

## 3. The hierarchy of Hessian models

| Level | Operator on `T_X M` (frame coordinates) | Exact when | Error elsewhere | Extra cost / matvec |
|---|---|---|---|---|
| **L0** GN (current) | `Π𝒥ᵀω²𝒥Π` | `N = 0` | `𝔚(·,N)` | — |
| **L0.5** damped / partial curvature | L0 + `L_εᵀ K L_ε` (damped lift, §4.4) or `β·K`, or Tucker-only / TT-only blocks | — | tunable | ≤ L1 |
| **L1** core-space Newton | L0 + `Lᵀ K_{Aᵀr} L`, `K_Z θ̂ := jvp_θ(∇_θ⟨Z, Φ(θ)⟩)[θ̂]` | at critical points | `⟨grad f, P_X D²Φ[·,·]⟩ = O(‖grad f‖)`, lift/gauge-dependent | ≈ one adjoint-jvp (2–3 adjoints) |
| **L2** Riemannian Newton (**target**) | L1 − `Lᵀ K_{grad f} L` | always | 0 | L1 + one `m`-independent T3–T3 contraction |
| **L3** AD through T3-SVD | = L2 (T3-SVD is second order) | always | 0 | SVD second derivatives, NaN at `t = 0`; not viable (§1.4) |
| beyond quadratic | exact polynomial `f(Φ(θ + tθ̂))` (degree `4d` in `t`) or cubic model | — | — | one doubled-rank sampling pass |

Notes.

- L1 is `Hess(f∘(parameterisation retraction))`; L2 is `Hess(f∘tv_retract) = Hess(f∘Exp)`. Both give
  quadratic local convergence (a Hessian perturbation of size `O(‖grad f‖)` preserves it: the step error
  is `O(‖g‖²)`). L2 is lift/gauge-independent, has the vertical directions exactly in its null space
  (L1 needs `Π` on both sides), and its model is consistent with the *actual* retraction for trust-region
  management. Since the L2 correction is `m`-independent and cheap (≈ 7 % of a GN matvec for the Hilbert
  setup `m = 800, N = 16, n = r = 6`), **L2 is the target.**
- "Exponential coordinates" add nothing at quadratic order; T3 geodesics have no closed form and are
  not needed.
- Novikov–Rakhuba–Oseledets (SISC 2022, arXiv:2103.14974) is *not* a source of curvature: they
  differentiate `f∘T_X` with the orthogonalization outside the trace and a `stop_gradient` on the
  projector, which is exactly what drops the Weingarten term ("can be interpreted as a constrained
  Gauss–Newton method"). Their trick gives Riemannian GN cheaply; curvature needs a curve that stays on
  the manifold — the parameterisation.
- Partial blocks (L0.5) are a legitimate intermediate: Tucker blocks dominate when a Tucker singular
  value is the smallest, TT blocks when a bond singular value is. Not planned unless the full term
  proves too expensive somewhere.
- Beyond quadratic: `S(Φ(θ + tθ̂))` is a polynomial of degree `2d` in `t`, computable exactly from the
  doubled-rank representation or from core-direction jets, so an *exact* line search along the
  parameterisation retraction is one sampling pass. Noted; not planned.

---

## 4. Implementation plan (target: L2 via the two-hole sweep)

### 4.1 The lift `L`: frame coordinates → raw core variations

Work in the left-canonical raw gauge `θ = (U_0..U_{d−1}; L_0..L_{d−2}, P_{d−1})` (the base point as the
frame stores it, `geometry.py:73-102`). From the frame construction (§1.2 step 1) keep two small
matrices per mode that are currently discarded:

- `S_i` (`nU_i × nD_i`) from the centre-core SVD `H_i^X = O_i S_iᵀ` — its singular values are the mode-`i`
  Tucker singular values of `X`;
- `T_i := einsum('abi,abj->ij', L_i, H_i^X)` (`rL_{i+1} × rR_{i+1}`) — because `X = L_{<i} H_i^X R_{>i} =
  L_{<i} L_i Λ_{>i}` with `Λ_{>i}` the left-canonical tail, `Λ_{>i} = T_i R_{>i}` and `H_i^X = L_i ×₃ T_i`;
  its singular values are the bond-`i` TT singular values of `X`.

Then the slot terms of `ξ` equal raw single-replacement terms iff

```
U̇_i = S_i⁻ᵀ V_i                 (the frame coordinate is V_i = S_iᵀ U̇_i — the same relation SharedGeometry uses, sharing.py:446-664)
Ġ_i = H_i ×₃ T_i⁻¹  (i < d−1),   Ġ_{d−1} = H_{d−1}        (Psenka–Boumal's R_k⁻¹; einsum('abr,rc->abc', H_i, inv(T_i)))
```

so `L: (V, H) ↦ (U̇, Ġ)` and `Lᵀ: (U̇_out, Ġ_out) ↦ (S_i⁻¹ U̇_out, Ġ_out ×₃ T_i⁻ᵀ)`. **This is the only place
`1/σ` appears and the only new geometric object.** On a non-minimal frame (`nD < nU` or `rR < rL`,
`docs/frame_variations.md:68-111`) use pseudo-inverses: the slack directions carry no curvature
information. `S_i`, `T_i` ride on `LocalModel` as pytree leaves (jit constraint, `:799-805`).

### 4.2 The curvature matvec `K`

For a fixed ambient `Z` (never formed densely),

```
K_Z θ̂ = d/ds ∇_θ ⟨Z, Φ(θ + sθ̂)⟩ |_{s=0}   =   ∇_η̂ ⟨Z, D²Φ[θ̂, η̂]⟩.
```

Two instances are needed: `Z = Aᵀr` (sample-structured; `∇_θ⟨Aᵀr,Φ⟩` is `t3_*_corewise_transpose` with
residual `ω²r`) and, for L2, `Z = grad f` (a doubled-rank T3, `tv_to_t3(frame, Πcoords,
include_shift=False)`; `∇_θ⟨Z, Φ⟩` is the corewise version of the T3→T3 projection sweep,
`tv_project_t3_onto_tangent_space` at the frame `(U, G, G, G)`). Then

```
hvp(p) = Π 𝒥ᵀ ω² 𝒥 Π p  +  Lᵀ ( K_{Aᵀr} − K_{grad f} ) L p  +  λ Π p.
```

Three ways to compute `K_Z`, in increasing order of effort — the first two are prototypes/oracles, the
third is the deliverable:

1. **Polynomial finite differences (zero new sweep code).** `∇_θ⟨Aᵀr, Φ(θ + sθ̂)⟩` is a polynomial of
   degree `2d − 1` in `s`, so its `s`-derivative at 0 is recovered *exactly* from `2d` corewise
   transposes at `θ + s_jθ̂` (plain T3s, no doubled rank) with polynomial-differentiation weights, or to
   ~1e-8 with one complex-step evaluation. ~`2d` adjoints per matvec. Enough to validate the whole
   scheme and measure the convergence gain on real problems before any sweep is written.
2. **JAX `jvp` of the existing transposes.** `jax.jvp(lambda cores: corewise_transpose(cores, ω²r),
   (cores,), (L p,))` — forward-over-reverse through the existing closure-free scan bodies; works on
   the uniform/jit path as is. ~2–3× one adjoint. Same for `Z = grad f`. Makes L2 available on the JAX
   backend essentially immediately and is the second oracle.
3. **Hand-written two-hole sweep (NumPy backend; the shipped version).** Every quantity in the
   adjoint sweeps is multilinear in `(cores, vectors)`, so its directional derivative along a core
   direction is the same Leibniz algebra as the jets, with the input jets `(G_k, Ġ_k)` of orders
   `{0, 1}` beside `(ξ_k, δξ_k)`. Each environment carries a companion "exactly one core replaced":

   ```
   μ^{(1)}_k = μ^{(1)}_{k−1} · G_k(ξ_k)  +  μ^{(0)}_{k−1} · [ Ġ_k(ξ_k) + G_k(V_kᵀ w_k) ]      (and likewise ν, σ, τ, η)
   ```

   and the `dU`/`dG` assemblies take all order-1 combinations (`μ^{(1)} ⊗ ξ ⊗ ν^{(0)} + μ^{(0)} ⊗ ξ^{(1)} ⊗ ν^{(0)}
   + μ^{(0)} ⊗ ξ ⊗ ν^{(1)}`, …). For the derivative kinds this is a **two-index jet** (vector order ≤ `order`,
   core order ≤ 1); the existing order-scan over one axis gains a second axis of length 2, so the cost is
   ≈ 2× the jetted adjoint. The `Z = grad f` instance is the two-hole version of the one-hole T3
   projection sweep (`tv_operations.py:333-419`). Apply/entries use the lean sweep `(ξ, μ)`, probe the
   full `(ξ, μ, ν, η)` (`apply.py:195-227`, `probing.py:767`); each state variable gets its order-1
   companion. Frame-dependent parts (the `sweep`) stay precomputed once per frame; `r` is fixed during
   the inner solve; only `θ̂ = Lp` varies.

### 4.3 Plug points

- `LocalModel.hvp` (`backend/optimizers.py:152-158`): add `Lᵀ(K_{Aᵀr} − K_{grad f})L` following the
  `regularizer.hessian` pattern. `LocalModel` already carries `residual` (raw — apply `ω²`), `sample`,
  `frame`, `sweep`; add `S_i`, `T_i`, the gradient coordinates, and the two-hole sweep state as leaves.
- `LocalModel.gn_quadratic` (`:146-150`): add `⟨p, Lᵀ(K − K')L p⟩` so the predicted reduction and `ρ_tr`
  are consistent with the model. Keep the Cauchy step in `gradient_descent`/`mc_sgd` (`:386-439`,
  `:344-358`) on the GN quadratic.
- Frontend twins `GaussNewtonModel.gn_hessian/.gn_quadratic/.evaluate` (`fitting.py:356-403`); uniform
  mirrors (`backend/uniform_fitting.py`, `UniformManifoldGeometryOps`, `geometry.py:389-409`).
- CG → **Steihaug–Toint**: in `_cg_body` (`:548-567`) on `dᵀHd ≤ 0` or `‖p + αd‖ > Δ` step to the
  trust-region boundary instead of taking a zero step; radius update from `ρ_tr` (already computed,
  `:750-751`); keep Armijo-along-retraction as the fallback. Bodies must stay module-level and
  closure-free with every value in state (`:527-539`). For *quadratic* (not just superlinear)
  convergence the forcing term should be `η = O(‖g‖)` rather than `‖g‖^{0.5}` once curvature is on.
- Preconditioning: downstream saw unpreconditioned CG stall on the near-singular full Hessian. Options:
  LM damping `H + μΠ` with `μ` from `ρ_tr`; a few inner GN-CG steps as a flexible preconditioner; or
  just the trust region, which bounds the ill-conditioned directions. Start with trust region + LM and
  measure (§5).
- Shared factors (`SharedGeometry`): with route (b) the tied parameterisation `Φ_shared = Φ∘T` (`T`
  linear) gives `D²Φ_shared = D²Φ[T·, T·]` automatically — lift with the group's `U̇_g` and apply `Tᵀ`
  on the way back — and `N_shared = Aᵀr − grad_shared f` is automatic if the L2 correction uses the
  shared gradient. No separate second-fundamental-form derivation.
- `COREWISE` geometry gets full Newton for free: `∇²_θ(f∘Φ) = GN_corewise + K_{Aᵀr}` with no lift;
  gauge-singular, so only with damping.

### 4.4 Rank-deficient frames and phase-in

At zero-padded continuation warm starts and at `x + x` (`docs/numerical_contracts.md:145`) `S_i` or
`T_i` is singular: the lift is undefined and `𝔚 ~ ‖N‖/σ_min` is unbounded. Curvature **must** be phased
in. The clean mechanism is a **damped lift** `L_ε`: replace `1/σ` by `σ/(σ² + ε²)` in `S_i⁻¹`, `T_i⁻¹`.
It interpolates continuously to GN in the nearly-flat directions (where the quadratic model is
unreliable at step lengths `≈ σ` anyway — that is the curvature radius) and is exact as `ε → 0`. The
natural scale is `ε ~ ‖N‖/λ_min(H_GN)` (curvature exceeds GN curvature when `σ < ‖N‖/λ`); tune on the
Hilbert levels. The cruder alternative is an adaptive scalar `β` on `K` (NL2SOL-style: keep whichever
of GN / full model predicted the last reduction better).

Diagnostics that come for free: `1 − ρ_tr` and the tail rate (curvature strength); `σ_min(S_i)`,
`σ_min(T_i)` (where it lives); the slot norms of `N` (§2.2) — a large single-mode slot `k` says "raise
`n_k`", a large TT-normal core slot says "raise a bond rank" — which is also a rank-continuation signal.

### 4.5 Costs (apply kind, per sample, leading order; jets multiply by ≈ `order + 1`)

| | cost |
|---|---|
| GN matvec (`𝒥` + `𝒥ᵀ`, given sweep) | `2d(Nn + 3nr²)` |
| `K_{Aᵀr}` matvec, two-hole sweep | ≈ `2–3 · d(Nn + 3nr²)` → full Newton matvec ≈ **2–2.5× GN** |
| `K_{Aᵀr}` via route 1 | ≈ `2d` corewise transposes ≈ 4–8× GN matvec |
| `K_{grad f}` (L2 correction) | `m`-independent, `O(d[N n² + n r³])` |
| per outer iteration | unchanged (frame, sweep, gradient); `S_i`, `T_i` free; trust-region bookkeeping |
| L3 (AD through T3-SVD) | SVD second derivatives at doubled ranks; NaN at `t = 0`; not viable on `pad_safe_svd` |

Expect each Newton iteration at ≈ 2× a GN iteration and the linear tail (15–20 iterations per level
today) to collapse to ~5–8, plus the ability to leave saddles GN cannot see. Memory: nothing dense;
the two-hole sweep doubles the per-sample state (`d(r+ρ)` → `2d(r+ρ)` for apply); the existing
chunking over `W` applies unchanged.

### 4.6 Validation

- **Oracle 1 (core coordinates, SVD-free):** on small dense problems compare `hvp` with the exact
  Lagrangian Hessian `∇²_θ[f∘Φ − ⟨grad f, Φ⟩]` (dense `jax.hessian` or finite differences); L1 against
  `∇²_θ(f∘Φ)`.
- **Oracle 2 (retraction):** `⟨p, Hp⟩` vs `d²/dt² f(tv_retract(tp))` — *exact* because the retraction is
  second order (§2.3); and `⟨p, H_L1 p⟩` vs `d²/dt² f(Φ(θ + tLp))`, which distinguishes L1 from L2.
  `dev/curvature_oracles/` has the dense prototypes of both.
- Symmetry of the full `hvp`; vertical null space for L2 (`hvp` of a pure-gauge direction is zero up to
  roundoff, `tests/backend/test_tv_operations.py:121-137` has the generators).
- Route 3 against route 2 (jvp) on the JAX path, numpy-only numerics per the testing strategy, dispatch
  in `test_dispatch`.
- A **convergence-rate test** (there is none today): Hilbert with 1 % noise; assert `ρ_tr → 1` and
  q-quadratic `‖g‖` decay once curvature is on; assert GN's linear tail as the control.

### 4.7 Order of work

0. **Measure first (no hot-path code).** Log `1 − ρ_tr` and the tail rate per level across the real
   problems, plus `σ_min(S_i)`, `σ_min(T_i)`, `‖N‖`; route 1 gives `⟨p, 𝔚p⟩` directly. This says where
   curvature matters and how much is at stake (§2.4 predicts the gain).
1. **L2 on the JAX backend via route 2**, lift with `S_i`/`T_i`, Steihaug-CG + trust region, damped lift
   for continuation starts. Validate with both oracles; measure iteration counts on the Hilbert levels.
2. **Two-hole sweep (route 3)** for the NumPy backend and for speed, reusing the jet-recurrence engine;
   two-index jets for the derivative kinds; uniform mirror.
3. Diagnostics / adaptivity: switch curvature on when `1 − ρ_tr` exceeds a threshold; slot norms of `N`
   as a rank-continuation signal; interaction with the zero-reference `g0` plan
   (`dev/zero_reference_g0_plan.md`).

---

## 5. Open questions

- Trust region vs LM damping vs hybrid for the indefinite system — and the preconditioner question
  (downstream stall on the near-singular full Hessian). Needs the stage-1 experiments.
- The damped-lift scale `ε` and the phase-in rule after a zero-padded `resize`; whether `ε` should be
  per-mode (`S_i`) / per-bond (`T_i`).
- API: `newton_cg(..., curvature=False)` default off at first? A `Hessian` enum (`'gauss_newton' |
  'riemannian'`) on `LocalModel`? The forcing-term change (`η = O(‖g‖)`) only when curvature is on?
- Two-index jet layout for the derivative kinds (a second order axis of length 2 vs interleaving).
- Whether the `K_{grad f}` correction should be computed from the gauged gradient coordinates (needs
  the doubled-rank `Z`) or folded into the same sweep by linearity (`K` is linear in `Z`; `Aᵀr − grad f`
  cannot be represented as samples, so two sweeps it is — unless the T3–T3 two-hole contraction is
  cheap enough to never matter, which the cost estimate says it is).
- Downstream (T3Polynomial) has nonlinear residuals (`Σ rᵢ∇²rᵢ ≠ 0` there); the sweep gives the
  manifold curvature term only — the ambient Hessian term is theirs.

## 6. Literature

- Novikov, Rakhuba, Oseledets, "Automatic differentiation for Riemannian optimization on low-rank
  matrix and tensor-train manifolds", SIAM J. Sci. Comput. 44(2), 2022 (arXiv:2103.14974) — AD
  Riemannian gradient and *Gauss–Newton* Hessian-vector product; curvature term dropped by design
  (stop-gradient on the projector); Hvp ≈ 2× gradient.
- Psenka & Boumal, "Second-order optimization for tensors with fixed tensor-train rank",
  arXiv:2011.13395 (2020) — TT Weingarten map (`R_k⁻¹`, diagonal + cross terms); trust-region Newton
  beats ALS/RTTC on hard, ill-conditioned completion problems.
- Heidel & Schulz, "A Riemannian trust-region method for low-rank tensor completion", NLAA 2018
  (arXiv:1703.10019) — Tucker Hessian via Golub–Pereyra pseudoinverse derivatives; GN "not robust if we
  add some noise"; HOSVD second-order "indication".
- Absil, Mahony, Trumpf, "An extrinsic look at the Riemannian Hessian", GSI 2013 — Weingarten form;
  Vandereycken, SIAM J. Optim. 2013 (arXiv:1209.3834) — fixed-rank matrix Hessian.
- Absil & Malick, "Projection-like retractions on matrix manifolds", SIAM J. Optim. 2012 — projective
  retractions are second order; Absil & Oseledets, Comput. Optim. Appl. 2015 — survey.
- Absil, Mahony, Sepulchre, *Optimization Algorithms on Matrix Manifolds*, §5.5 — `Hess(f∘R)` vs
  `Hess f`; Absil, Baker, Gallivan — RTR convergence with second-order models.
- Breiding & Vannieuwenhoven, "Convergence analysis of Riemannian Gauss–Newton methods and its
  connection with the geometric condition number", Appl. Math. Lett. 2018 (arXiv:1708.02488); "Sensitivity
  of low-rank matrix recovery" (arXiv:2103.00531), Lemma 1 (residual × principal curvatures).
- Molozhavenko & Rakhuba, "Optimization on the Extended Tensor-Train Manifold with Shared Factors",
  arXiv:2508.20928 (2025) — ETT/SF-ETT smooth manifold, projections, AD gradient; first order only.
- Alger, Christierson, Chen, Ghattas, "Tucker Tensor Train Taylor Series", arXiv:2603.21141 (2026) —
  T4S; Appendix A for the frame/tangent/retraction conventions the code follows.
- Dennis, Gay, Welsch, NL2SOL (1981) — adaptive switching between GN and augmented models (the `β`
  heuristic of §4.4).
