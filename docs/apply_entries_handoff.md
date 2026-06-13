# Handoff — `apply` / `entries` for tangent vectors, and their adjoints

Working notes to resume this feature. Audience: future contributors / future Claude. Written
2026-06-12. Letters are the `W` (probe/vec/index stack) / `K` (tangent stack) / `C` (base/core stack)
scheme — see `docs/batching_and_stacking.md`. Paper math: `t4s.pdf` Section 6, Algorithms 5–8; the
notation map is in `docs/probing_section6_notes.md`.

## Goal

`apply` and `entries` are the **all-modes special case of probing** (probing leaves one index free;
these contract *every* index). They compute applications / entries of the dense tensor a `T3Tangent`
represents, **without forming it**, exploiting structure to be cheaper than probing. Scope:

- `T3Tangent.apply` / `entries` (forward) — **DONE**.
- `T3Tangent.apply_transpose` / `entries_transpose` (adjoint) — **DONE**.
- `TuckerTensorTrain.apply_transpose` / `entries_transpose` (adjoint of the existing plain
  `apply`/`entries`) — **TODO (slice 3)**.

## Status

- **Slice 1 (DONE, committed `5ac8db22`)** — forward `T3Tangent.apply(ww)` and `entries(index)`.
  Backend `apply_tangent` / `entries_tangent` in `backend/probing.py`; frontend methods in
  `manifold.py`; tests in `tests/test_manifold.py` (`test_tangent_apply`, `test_tangent_entries`) +
  jit dispatch in `tests/test_dispatch.py`. Verified vs dense to ~1e-16 across all `W/K/C` stacks.
- **Slice 2 (DONE, committed `f25e3d14`)** — adjoint `T3Tangent.apply_transpose(c, ww, basis)` and
  `entries_transpose(c, index, basis)`. Backend `apply_tangent_transpose` / `entries_tangent_transpose`
  in `backend/probing.py` (base sweep + single-term scatter `_apply_transpose_assemble`, reusing the
  existing `WKCi_WCa_WCj_to_{WKCiaj|KCiaj}` and `Wo_WKCa_to_{WKCao|KCao}` contractions); frontend
  static methods in `manifold.py`; adjoint-identity tests in `tests/test_manifold.py` + jit dispatch in
  `tests/test_dispatch.py`. The clean-reuse insight (below) held: it is **one** contraction per output,
  not the zero-the-other-terms reuse of `assemble_*` originally sketched. Verified `⟨applyᵀc, v⟩ ==
  Σ_W c·apply(v)` to ~1e-16 across all `W/C` stacks and both `sum_over_probes` modes.
- **Slice 3 — NOT STARTED.** Math derived (plain-`TuckerTensorTrain` adjoints, below); implementation
  not yet written.

## The algorithms

The base point's orthogonal cores are `T3Basis.data = (U, O, P, Q)` = (up-tucker, down/central-tt,
left-tt, right-tt). The variation is `T3Variations.data = (δU, δG)`. The base sweep (Algorithm 6):

```
ξ̂_i = Uᵢᵀ wᵢ            (compute_xis)        # for entries: ξ̂_i = Uᵢ[..., idxᵢ]  (fiber slice)
μ̂   = left  sweep via P   (compute_mus)      # μ̂[k] = μ̂_{k-1}
ν̂   = right sweep via Q   (compute_nus)      # ν̂[k] = ν̂_{k+1}
η̂_k = μ̂_{k-1}·O_k·ν̂_{k+1}  (compute_etas)
```

### Forward (DONE) — single left-to-right pass, no right/central sweep

`apply(v, ww)` = the perturbation-left sweep `σ` (Algorithm 7, `compute_sigmas`, propagates via `Q`)
run to its **terminal carry**, then contract the last bond. No `ν̂`/`τ`/`η̂`/`δη` and no per-mode
assembly — roughly half of `probe_tangent`. Because there is no free index, the rightward `Q`
propagation folds into the one left sweep:
```
σ_i = σ_{i-1}·Q_i(ξ̂_i) + μ̂_{i-1}·δG_i(ξ̂_i) + μ̂_{i-1}·O_i(δξ_i)        # δξ_i = δUᵢᵀ wᵢ (or sliced)
apply(v, ww) = sum_last_bond(σ_terminal)
```
`entries` = same with `ξ̂`/`δξ` from slicing Tucker-core fibers (no contraction, no `N` factor).
Implemented via the shared `_sigma_step`, plus `_apply_from_xis` (terminal scan + contract) and
`_entry_xis` (fiber slicing). Result shape `W + K + C`.

### Adjoints (TODO) — base sweep + a single-term scatter assembly

The forward *consumes* the perturbation; the adjoint *produces* it. Given a scalar `c` (shape `W (+C)`),
scatter it into each variation core. **Verified formulas** (`⟨applyᵀ(c), v⟩ = c·apply(v)` to 2e-14):

```
applyᵀ(c):    δG̃_k = c · μ̂_{k-1} ⊗ ξ̂_k ⊗ ν̂_{k+1}          # over (rL, nU, rR)
              δŨ_k = c · η̂_k ⊗ w_k                          # over (nO, N)

entriesᵀ(c):  same, ξ̂ sliced; δŨ_k = c·η̂_k scattered into column idxₖ   (wₖ → e_{idxₖ})
```
Needs the full base sweep (`ξ̂, μ̂, ν̂, η̂`) but **skips** `probe_transpose`'s adjoint perturbation sweep
(`σ̃, τ̃, δη̃, δξ̃`) — so it is cheaper than a general transpose.

### Plain `TuckerTensorTrain` adjoints (TODO) — pure construction, no sweep

```
applyᵀ(c, ww) = c · (w_0 ⊗ … ⊗ w_{d-1})       # rank-1 T3.
entriesᵀ(c, idx) = c · (e_{idx_0} ⊗ … )       # rank-1 T3 (scatter c at idx; Tucker cores = scaled one-hots).
```
Batch-summed over `W` → a **rank-`|W|`** T3 (the back-projection `Σ_s c_s·W_s`). Build via CP→TT:
Tucker core `B_i = [w_i^s]_s` (shape `(|W|, N_i)`); tt cores = diagonal "copy" tensors of rank `|W|`;
`c` absorbed into a boundary core. This is the natural `Jᵀ` for least-squares fitting.

## Clean implementation plan for slice 2 (DONE — the key insight, verified)

The shipped slice 2 went one better than the sketch below: each adjoint output is a **single existing
contraction** (`WKCi_WCa_WCj_to_{WKCiaj|KCiaj}` for `δG̃`, `Wo_WKCa_to_{WKCao|KCao}` for `δŨ`, with
`c` folded into `μ̂`/`η̂`), not a re-call of `assemble_*` with the other terms zeroed. Kept for the
record — the original sketch was to take **one term** of the existing transpose-assemble functions
(zero the others, fold `c`), inheriting `W`/`C` stacking + `sum_over_probes` for free:

```python
# δG̃ — only assemble_tt_variations' middle term (τ̃⊗ξ̂⊗ν̂) survives:
dG_t = assemble_tt_variations(
    sigma_tildes=zeros_like_rR, tau_tildes=[c[...,None]*mu_k for mu_k in mus],
    deta_tildes=zeros_like_nU,  xis=xis, mus=mus, nus=nus,
    sum_over_probes=sum_over_probes, n_probe=n_probe)
# δŨ — only assemble_tucker_variations' second term (w⊗δξ̃) survives:
dU_t = assemble_tucker_variations(
    ztildes=zeros_like_N, dxi_tildes=[c[...,None]*eta_k for eta_k in etas], ww=ww,
    etas=etas, sum_over_probes=sum_over_probes)        # etas unused (multiplied by ztildes=0)
```
**VERIFY THIS REUSE numerically before trusting it** (leg orders, the zeroing, `c` broadcast). If it
fights the leg conventions, fall back to dedicated `'...'` einsums (clean when `C=()`; broadcast `w`
over `C` otherwise). For `entries_transpose`, `ξ̂` is sliced and `δŨ` scatters into column `idxₖ`
instead of an outer product with `w` — likely a dedicated assembly (slicing on the *output* side).

## API (mirror `probe` / `probe_transpose`)

- `T3Tangent.apply_transpose(c, ww, basis, sum_over_probes=False) -> T3Tangent`
- `T3Tangent.entries_transpose(c, index, basis, sum_over_probes=False) -> T3Tangent`
- `TuckerTensorTrain.apply_transpose(c, ww, sum_over_probes=False) -> TuckerTensorTrain`
- `TuckerTensorTrain.entries_transpose(c, index, sum_over_probes=False) -> TuckerTensorTrain`
- Backend: **dedicated** `apply_tangent_transpose` / `entries_tangent_transpose` in `probing.py`;
  the plain ones can live in `backend/apply.py` / `entries.py` or as constructors.
- `sum_over_probes=False` keeps `W` as the output tangent stack (one tangent per probe-set, like
  `probe_transpose`); `=True` sums `W` (the `Jᵀr` back-projection).

## Verification recipe (scripts were in /tmp — recreate)

Build a random orthogonal tangent, then check, across `W/K/C` stacks (+ `sum_over_probes` for adjoints):

```python
import numpy as np, t3toolbox.tucker_tensor_train as t3, t3toolbox.basis_variations_format as bvf
import t3toolbox.manifold as t3m, t3toolbox.corewise as cw
x = t3.TuckerTensorTrain.randn((6,7,5,8),(2,3,2,2),(1,2,3,2,1)); base,var = bvf.t3_orthogonal_representations(x)
v = t3m.T3Tangent(base, var); ww = tuple(np.random.randn(N) for N in (6,7,5,8))
# forward:  v.apply(ww)  == einsum(v.to_dense(), ww...)     (DONE, ~1e-16)
# adjoint:  cw.corewise_dot(applyT(c).variations.data, v.variations.data) == c*v.apply(ww)   (verified 2e-14)
# plain:    cw/Frobenius  <applyT(c), x_dense> == c*x.apply(ww)
```
Tests go in `tests/test_manifold.py` (tangent) and `tests/test_tucker_tensor_train.py` (plain); jit
dispatch in `tests/test_dispatch.py`.

## Open decisions to confirm with Nick
1. ~~`sum_over_probes` default~~ — **RESOLVED**: `False` (keep `W`), matching `probe_transpose`. Shipped.
2. Plain-T3 batched adjoint output (slice 3) — rank-`|W|` (CP→TT) by default when summed? Or keep-`W` (a
   `W`-stacked rank-1 T3) default with a sum option? **Still open — confirm before slice 3.**
3. `K`-stacking for the adjoints (input `c` carrying an extra `K` from a `K`-stacked forward apply) —
   `probe_transpose`-style; **deferred** in slice 2 (`c` assumed shape `W (+C)`). Revisit per use case.
4. ~~Naming `apply_transpose` / `entries_transpose`~~ — **RESOLVED**: mirrors `probe_transpose`. Shipped.

## File map
- Backend sweeps/assembly: `t3toolbox/backend/probing.py` (`compute_xis/mus/nus/etas`,
  `compute_sigmas`, `_sigma_step`, `apply_tangent`, `entries_tangent`, `assemble_tt_variations`,
  `assemble_tucker_variations`, `probe_tangent_transpose`).
- Base (plain) apply/entries: `t3toolbox/backend/apply.py`, `t3toolbox/backend/entries.py`.
- Frontend: `t3toolbox/manifold.py` (`T3Tangent`), `t3toolbox/tucker_tensor_train.py`.
- Tests: `tests/test_manifold.py`, `tests/test_dispatch.py`, `tests/test_tucker_tensor_train.py`.
- Background: `docs/probing_section6_notes.md`, `docs/batching_and_stacking.md`.
