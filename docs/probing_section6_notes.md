# Probing — Section 6 (Riemannian Jacobian) notes ↔ code

Working notes from reading **Section 6 (pp. 30–34)** of `t4s.pdf` (Tucker Tensor Train Taylor
Series, arXiv:2603.21141) and mapping it to `t3toolbox/backend/probing.py`. For resuming the
probing work.

## Big picture (6 + 6.1)
Riemannian fitting (TR-RMGN, MC-SGD) applies the manifold least-squares Jacobian `J` and transpose
`Jᵀ`. Full tangent vectors are too big, so everything runs on **gauged variations**
`δV = ((δUᵢ), (δGᵢ))`. Factorization:

- **`J = 𝒥 ∘ Π`**, **`Jᵀ = Π ∘ 𝒥ᵀ`**.
- **`Π`** = orthogonal projector onto gauged variations: `δÛₗ=(I−UₗUₗᵀ)δUₗ`,
  `δĜₗ=(I−Pₗᴸ(Pₗᴸ)ᵀ)δGₗ` (last δĜ untouched) = **`orthogonal_gauge_projection`** (enforces gauge
  conditions (48)–(49)).
- **`𝒥`** = probe the tangent vector against training data. Cost `O(dNn + dnr² + Mm)` per sample;
  `N`, `d` enter only linearly.
- `𝒥⁽ˢ⁾` = single-sample version; full `𝒥` loops/vectorizes over samples. Notation: hat `^` =
  base-point, `δ` = perturbation, tilde `~` = adjoint (transpose sweep).

## A "probe" (6.2.1, Algorithm 5)
`i`-th probe `zᵢ` of multilinear `T` represents `t ↦ T(w₁,…,wᵢ₋₁,t,wᵢ₊₁,…,w_d)` — contract with all
probing vectors except the `i`-th, leaving index `i` free. Output mode = extra tensor mode `m`.

Algorithm 5 edge variables (Figure 7) ↔ code:

| Paper | code | meaning |
|---|---|---|
| `ξᵢ = Uᵢᵀwᵢ` (contract up) | `compute_xis` | probe vec through Tucker basis |
| `μᵢᵀ = μᵢ₋₁ᵀ Gᵢ(ξᵢ)` (left sweep) | `compute_mus` | left partial product |
| `νᵢ = Gᵢ(ξᵢ) νᵢ₊₁` (right sweep) | `compute_nus` | right partial product |
| `ηᵢ = μᵢ₋₁ᵀ Gᵢ νᵢ` (central) | `compute_etas` | leaves Tucker-middle free |
| `zᵢ = Uᵢ ηᵢ` (contract down) | `assemble_zs` | back to ambient index |

`probe_t3` = Algorithm 5 (already wrapped by `TuckerTensorTrain.probe`).

## Tangent probing = `𝒥⁽ˢ⁾` (6.2.2, Algorithms 6 + 7)
A tangent vector is a **doubled-rank** T3 (A.3.1), so each edge variable splits into base + pert:
`ξ→(ξ̂,δξ)`, `μ→(σ,μ̂)`, `ν→(τ,ν̂)`, `η→(δη,η̂)`.

- **Algorithm 6** = Algorithm 5 on the *base point* with gauge-appropriate cores **`P` (left),
  `Q` (right), `O` (central)** in place of `G` → base probes `ẑ` and base edge vars `ξ̂,μ̂,ν̂,η̂`.
  In code: `compute_xis/mus/nus/etas` on `(U,P,Q,O)`.
- **Algorithm 7** (perturbation sweep) ↔ code, verified term-by-term:
  - `δξᵢ = δUᵢᵀwᵢ` → `compute_dxis`
  - `σᵢᵀ = σᵢ₋₁ᵀQᵢ(ξ̂) + μ̂ᵢ₋₁ᵀδGᵢ(ξ̂) + μ̂ᵢ₋₁ᵀOᵢ(δξ)` → `compute_sigmas`
  - `τᵢ = δGᵢ₊₁(ξ̂)ν̂ + Oᵢ₊₁(δξ)ν̂ + Pᵢ₊₁(ξ̂)τ` → `compute_taus`
  - `δηᵢ = σᵢ₋₁ᵀQᵢν̂ + μ̂Pᵢτ + μ̂δGᵢν̂` → `compute_detas`
  - `δzᵢ = Uᵢδηᵢ + δUᵢη̂ᵢ` → `assemble_tangent_zs`

`probe_tangent` = Algorithm 6 + Algorithm 7 = `𝒥⁽ˢ⁾`.

## Transpose `(𝒥⁽ˢ⁾)ᵀ` (6.2.3, Algorithm 8)
Given residual-like `z̃`, adjoint sweep over the same edge variables → `((δŨᵢ),(δG̃ᵢ))`:
`δη̃ᵢ=Uᵢᵀz̃ᵢ`; adjoint `τ̃,σ̃`; central `δξ̃`; assemble `δŨᵢ=z̃ᵢη̂ᵢᵀ+wᵢδξ̃ᵢᵀ`,
`δG̃ᵢ = τ̃⊗ξ̂⊗ν̂ + μ̂⊗ξ̂⊗σ̃ + μ̂⊗δη̃⊗ν̂` (`⊗` = tensor product `(a⊗b⊗c)[j,k,l]=a[j]b[k]c[l]`).
= **`probe_tangent_transpose`** (`compute_deta_tildes/tau_tildes/sigma_tildes/dxi_tildes` +
`assemble_tucker_variations/assemble_tt_variations`).

## 6.3 corewise (non-manifold)
Same Algorithms 7/8 with **plain cores `Gᵢ` substituted for `(Pᵢ,Qᵢ,Oᵢ)`** give the corewise
Jacobian for Adam/L-BFGS-style optimizers (cores as free variables, no gauge). Probing code is
dual-use (manifold vs corewise) by substitution.

## Work items / things to reconcile (flagged, not yet done)
1. **Stale paper refs in `probing.py`**: docstrings cite "Section 5.2, Figure 9" and
   "Formula (34)/(36)/(38)/(40)/(41)/(43)–(46)" — earlier-draft numbering. Current paper:
   **Section 6.2, Figure 7, Algorithms 5–8**. (Same kind of cross-ref pass we did for manifold.)
2. **Base-core ordering**: `probe_tangent`/`probe_tangent_transpose` take
   `base = (up, left, right, outer) = (U,P,Q,O)`, but `T3Basis.data = (up, down, left, right) =
   (U,O,P,Q)`. Same reorder mismatch as manifold.py; not yet wired to `T3Basis`/`T3Tangent`.
3. **Factorization not assembled**: `probe_tangent` is `𝒥⁽ˢ⁾`, `orthogonal_gauge_projection` is `Π`,
   but the Riemannian `J = 𝒥∘Π` / `Jᵀ = Π∘𝒥ᵀ` aren't composed into a single callable yet.
4. **Base edge-var caching**: `ξ̂,μ̂,ν̂,η̂` depend on `(p,s)` not on the tangent vector → can be
   precomputed once per (base, sample) and reused (compute↔memory trade-off). `probe_tangent`
   currently recomputes them every call.
5. **Stacking/vectorization**: both directions now batch over all three blocks — `F` probes, `V`
   tangent stack, `G` base stack. Forward (`probe_tangent`): the `V`-stacked case via 3-group
   contractions in `compute_sigmas`/`detas`, `assemble_tangent_zs` (slice 5c; output `F + V + G`).
   Transpose (`probe_tangent_transpose`): accepts `V`-stacked residuals `F + V + G` and carries `V`
   to the result's tangent stack (`sum_over_probes=True` → `V`; `=False` → `F + V`); the adjoint
   sweep reuses the forward's contractions and the assembly adds 10 outer-product builders.

## Likely build targets (to confirm with Nick)
- Wire `probe_tangent` / `probe_tangent_transpose` into `T3Tangent` (`.probe()` and the transpose),
  reconciling the base order, like the manifold port.
- Compose the Riemannian `J` / `Jᵀ` (`𝒥∘Π`, `Π∘𝒥ᵀ`) as callables on `T3Tangent`.
- Tests against the dense reference (`probe_dense`) + the `Jᵀ` adjoint identity
  `⟨z̃, 𝒥(δV)⟩ = ⟨𝒥ᵀ(z̃), δV⟩`.
- Update the stale `probing.py` paper references.
- (Open) directionally-symmetric probe form `wᵢ = (x,…,x,ω)` used in fitting — build on top or stay
  general.

## Future idea — exploit special structure in the tangent stack `V` (deferred, NOT in current scope)

Slice 5c probes a `V`-stacked tangent (a batch of `k` tangent vectors at *each* base point) via a
general **3-group contraction** (`F` probes x `V` tangents x `G` bases). It assumes the `V` vectors
are an *arbitrary* collection -- no relationship between them -- and the code must always support that.

Open question (Nick, 2026-06-12): when the `V` tangent vectors carry special structure -- e.g. they
form an **orthonormal block** within the one tangent space `T_x M` -- can `J^(s)` be applied more
cheaply than the general 3-group sweep? Plausible angles: shared sub-contractions across the block, a
factored/low-rank form of the perturbation edge variables, or work that cancels under orthonormality.
Not pursued now; recorded for later. The general (arbitrary-`V`) path stays the default regardless.
