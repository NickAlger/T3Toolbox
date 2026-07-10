# Entries, applies, and probes — the three sampling operations

> **If you are choosing how to "sample" or "measure" a Tucker tensor train — for fitting, sketching,
> evaluation, or a Riemannian Jacobian — read this first.** `entries`, `apply`, and `probe` are three
> closely related operations that are easy to confuse (they share most of their machinery and their
> stacking conventions). This doc explains exactly how they differ, how they relate, what they cost,
> and how each lifts to the manifold (tangent) setting with a transpose. The terse version is in
> `CLAUDE.md`; the stacking details live in [`batching_and_stacking.md`](batching_and_stacking.md).

Math reference: **Section 6.2 (Algorithms 5–8)** of Alger, Christierson, Chen & Ghattas (2026),
*"Tucker Tensor Train Taylor Series"* (arXiv:2603.21141) — probing and its Riemannian Jacobian — and
**Appendix A.3** for the tangent/manifold picture. (As elsewhere, the local `t4s.pdf` numbering is
newer than the public arXiv.)

---

## TL;DR

All three operations evaluate the **dense tensor** a Tucker tensor train represents — *without ever
forming it* — by contracting it against test vectors, one per mode. They differ in **how many modes
are left free** and **what kind of test vectors** are used:

|                                  | **all modes contracted → scalar** | **one mode left free → vector(s)** |
|----------------------------------|-----------------------------------|------------------------------------|
| **general test vectors `wᵢ`**    | **`apply`**                       | **`probe`**                        |
| **one-hot test vectors `e_{iᵢ}`**| **`entries`** (point evaluation)  | *(fiber extraction; = `probe` with one-hot — not a named API)* |

- **`entries`** — read off `X[i₀,…,i_{d-1}]` at given multi-indices. A scalar per index.
- **`apply`** — the full multilinear form `X(w₀,…,w_{d-1}) = Σ X[i] · w₀[i₀]···w_{d-1}[i_{d-1}]`. A
  scalar per probe.
- **`probe`** — contract *all but one* index, for each index in turn, leaving `d` free modes. Returns
  `d` vectors, the `i`-th being `X(w₀,…,wᵢ₋₁,·,wᵢ₊₁,…,w_{d-1}) ∈ ℝ^{Nᵢ}`.

> **Probing is the original exemplar.** This code was first written *for* Riemannian fitting against
> **probes** (Section 6 of the paper) — that is the problem `T3Targent.probe` / `probe_transpose` were
> built around. `apply` and `entries` were added later as the cheaper all-modes special cases, as part
> of making the library **general-purpose**: they are equally first-class, and (as verified below) just
> as Riemannian-ready. Pick the operation your problem actually calls for — none is privileged.

---

## Notation

Shapes use the library's standard symbols (see [`batching_and_stacking.md`](batching_and_stacking.md)):

- `d` — the tensor order (number of modes).
- `Nᵢ` — ambient mode dimensions; `nᵢ` — Tucker ranks; `rᵢ` — TT bond ranks.
- `W` — the **probe / sample stack** (a batch of test-vector sets, or of multi-indices). Lives on the
  test vectors `ww` / the `index` array **only**, never on the cores.
- `C` — the **frame/core stack** (`= stack_shape`; a batch of whole T3s). Lives on every core.
- `K` — the **tangent stack** (a batch of tangent directions at one frame; tangent operations only).
- Output stacking is always **base-inner**: `W + K + C` (probe stack outer, frame stack inner).

---

## 1. The three operations, precisely

Let `X` be the dense order-`d` tensor represented by a `TuckerTensorTrain`.

### `entries(index) → scalar`
```
index : int array, shape (d,) + W       # index[m] = the (stack W of) indices into mode m
result: X[index[0], …, index[d-1]]       # shape W + C
```
The all-modes special case of `probe` with one-hot vectors, but computed by **slicing** Tucker-core
fibers `Uᵢ[…, idxᵢ]` (no contraction over `Nᵢ`, no `Nᵢ` cost factor), then the same left-to-right TT
sweep as `apply`. Frontend: `TuckerTensorTrain.entries`. Dense check: `result == X.to_dense()[index]`.

### `apply(ww) → scalar`
```
ww    : list of d arrays, ww[m].shape = W + (Nm,)
result: Σ_{i₀…i_{d-1}} X[i] · ww[0][i₀] ··· ww[d-1][i_{d-1}]   # shape W + C  (a scalar, unstacked)
      = X(w₀, …, w_{d-1})    (the multilinear form)
```
The all-modes special case of `probe` (probing leaves one index free; `apply` contracts them all). A
single left-to-right sweep — no right/central sweep, no per-mode assembly. Frontend:
`TuckerTensorTrain.apply`. Dense check: `result == einsum('i...z,i,...,z->', X, *ww)`.

### `probe(ww) → d vectors`
```
ww    : list of d arrays, ww[m].shape = W + (Nm,)
result: list of d arrays, result[m].shape = W + C + (Nm,)
        result[m] = X(w₀, …, w_{m-1}, ·, w_{m+1}, …, w_{d-1})   # contract all but mode m
```
The general operation: `d` separate contractions, each leaving one mode free. Needs left + right +
central sweeps plus a per-mode assembly back to the ambient index (`zᵢ = Uᵢ ηᵢ`). Frontend:
`TuckerTensorTrain.probe`. (Algorithm 5; backend `probing.probe_t3`.)

---

## 2. How they relate (the containment hierarchy)

```
            probe  ⊃  apply  ⊃  entries
         (one mode free) (all contracted)  (all contracted, one-hot)
```

Three exact identities make the picture precise:

- **`apply` is `probe` with the last free index also contracted.** For *any* mode `m`,
  `apply(ww) == ⟨probe(ww)[m], w_m⟩`. (Contracting any one probe's free index against its own test
  vector collapses it to the scalar.)
- **`entries` is `apply` with one-hot test vectors.** `entries(i) == apply(e_{i₀}, …, e_{i_{d-1}})`.
  (`entries` realizes this by fiber slicing rather than an actual contraction, which is why it is
  cheaper.)
- **Fiber extraction is `probe` with one-hot test vectors** (`probe(e_i)[m]` is the mode-`m` fiber of
  `X` through `i`). This is the empty cell in the TL;DR table — useful, but not given its own API.

So if you understand `probe`, the other two are restrictions of it; if you only need a scalar, reach
for `apply` (or `entries` for point evaluation) and skip probing's extra sweeps.

---

## 3. Cost

Per sample (one probe / index, ignoring stacks), with `N ~ max Nᵢ`, `n ~ max nᵢ`, `r ~ max rᵢ`:

| op        | cost (per sample)            | why                                                            |
|-----------|------------------------------|---------------------------------------------------------------|
| `entries` | cheapest: `O(d·n·r²)`        | fiber **slice** (no `Nᵢ` factor) + one L→R TT sweep            |
| `apply`   | `O(d·N·n + d·n·r²)`          | contract each `wᵢ` through `Uᵢ` (`N·n`) + one L→R TT sweep      |
| `probe`   | `O(d·N·n + d·n·r²)` + assembly | left **and** right **and** central sweeps + `d` assemblies `Uᵢηᵢ` (output `Σ Nᵢ`) |

Rule of thumb: **`entries` < `apply` ≪ `probe`**. `entries` and `apply` return one scalar; `probe`
returns `d` vectors and does ~3× the sweeping plus the ambient assembly. Use the least general
operation that gives you what you need.

---

## 4. The Riemannian / tangent versions (and their transposes)

Each operation has a **tangent** form on `T3Tangent` — the directional derivative of the operation as
the T3 moves along a tangent vector at a base point — and a **transpose** (adjoint). These are what a
manifold least-squares fit needs: the forward applies the Jacobian `J`, the transpose applies `Jᵀ`.

Each op has a forward (`TuckerTensorTrain` Euclidean, `T3Tangent` the Riemannian Jacobian `𝒥`) and
**three** transpose flavors — **tangent / corewise / ambient**. Full taxonomy, costs, and the decision
guide live in **[`docs/transposes.md`](transposes.md)**; the quick map:

| op | forward | tangent `𝒥ᵀ` (`T3Tangent` → tangent) | corewise (`TuckerTensorTrain` → core grads) | ambient (`TuckerTensorTrain` → CP factors) |
|---|---|---|---|---|
| entries | `entries(index)` | `entries_transpose(c, index, frame, …)` | `entries_corewise_transpose(c, index, …)` | `entries_ambient_transpose(c, index, shape, …)` |
| apply   | `apply(ww)`      | `apply_transpose(c, ww, frame, …)`      | `apply_corewise_transpose(c, ww, …)`      | `apply_ambient_transpose(c, ww, …)`            |
| probe   | `probe(ww)`      | `probe_transpose(ztildes, ww, frame, …)` | `probe_corewise_transpose(ztildes, ww, …)` | `probe_ambient_transpose(ztildes, ww, …)`      |

Notes:

- **The forward tangent op is linear in the tangent direction** (it *is* the Jacobian `𝒥⁽ˢ⁾` applied to
  a gauged variation), so the same `.entries/.apply/.probe` names on `T3Tangent` mean "apply `𝒥`,"
  not "evaluate a point."
- **The three transpose flavors** (see `docs/transposes.md`): **tangent** = Riemannian gradient (a
  `T3Tangent`); **corewise** = gradient w.r.t. a frame's cores (raw core-grad tuple, for Adam/L-BFGS);
  **ambient** = the frame-free adjoint on the full tensor space (CP factors, via `from_canonical`). All
  three exist for all three ops (the earlier "probe has no ambient transpose" was true only before the
  transpose-grid redesign).
- **Bare `𝒥` / `𝒥ᵀ`, no gauge projector.** The tangent methods are the bare single-sample Jacobian and
  its adjoint. The full Riemannian operators factor as **`J = 𝒥 ∘ Π`** and **`Jᵀ = Π ∘ 𝒥ᵀ`**, where
  `Π = orthogonal_gauge_projection` projects onto gauged variations. Compose `Π` yourself: take the
  gradient as `op_transpose(r, …, sum_over_probes=True).orthogonal_gauge_projection()`.
- **`sum_over_probes`** (transpose only): `False` (primary) keeps the probe/sample stack `W` — one
  tangent per sample; `True` sums it — the Gauss-Newton data-misfit gradient `Jᵀr` (a single tangent
  when `W` is the only extra stack). See [`batching_and_stacking.md`](batching_and_stacking.md) §11.

### All three are Riemannian-compatible (verified)

For the least-squares objective `f(X) = ½‖S(X) − b‖²` on the fixed-rank manifold, with `S` any of the
three operations, the matrix-free Riemannian gradient
`g = op_transpose(r, …, sum_over_probes=True).orthogonal_gauge_projection()` and the Gauss-Newton
Hessian `H V = (𝒥ᵀ 𝒥 V)` gauged were checked end-to-end at an orthogonal, minimal-rank frame:

| operator | adjoint `⟨r, 𝒥V⟩ = ⟨𝒥ᵀr, V⟩` | gradient vs finite-diff *along the retraction* | Gauss-Newton Hessian   |
|----------|-------------------------------|-----------------------------------------------|------------------------|
| entries  | exact                         | rel. err ~5e-8                                | symmetric ~5e-17, PSD  |
| apply    | exact                         | rel. err ~3e-7                                | symmetric ~1e-12, PSD  |
| probe    | exact                         | rel. err ~5e-7                                | symmetric ~3e-13, PSD  |

So any of the three can drive a Riemannian fit (gradient descent, inexact Newton-CG, …); the choice is
about what measurements your problem provides, not about what the manifold code supports.

---

### Symmetric directional derivatives (a fourth dimension)

Each sampling op also has a **derivative** version: `y⁽ᵗ⁾ = dᵗ/dsᵗ [op(X + sP)]|₀` for `t = 0..order`,
the symmetric directional derivatives in one repeated direction `P` (orders ≥ tensor degree vanish).
They **strictly generalize** the plain op (`order 0` == the plain op) by adding a leading **order axis**
(`(order+1)+W+K+C+…`). Frontend: separate methods `{probe,apply,entries}_derivatives(ww | index, pp,
order)` on both `TuckerTensorTrain` and `T3Tangent` (the points `X` and directions `P` share the sample
stack `W`), with the same transpose flavors — **tangent** (`*_derivatives_transpose`) and **corewise**
(`*_corewise_derivatives_transpose`); the **ambient** derivative transpose is deferred
([`docs/ambient_derivative_transpose_note.md`](ambient_derivative_transpose_note.md)). Full `W+K+C`
stacking. Math + plan: [`docs/symmetric_probe_derivatives.tex`](symmetric_probe_derivatives.tex),
[`dev/archive/derivatives_mirror_plan.md`](../dev/archive/derivatives_mirror_plan.md). Why fit from derivatives:
[`docs/derivative_order_information_and_conditioning.md`](derivative_order_information_and_conditioning.md).

## 5. Measurement spaces (for fitting)

When `S` is the forward operator of a fit, its **measurement space** (where residuals `r = S(X) − b`
live) is:

- `entries`, `apply` → **a scalar per sample**: `r ∈ ℝ^{|W|}`.
- `probe` → **`d` vectors per sample**: `r ∈ ℝ^{N₀} ⊕ ··· ⊕ ℝ^{N_{d-1}}` (a list of `d` arrays, each
  `W + (Nᵢ,)`).

The measurement-space inner product (needed for residual norms and the adjoint identity) is the plain
sum-of-products: `Σ aᵢbᵢ` for the scalar cases, `Σ_modes Σ a[m]·b[m]` for `probe`.

---

## 6. When to use which

- **`entries`** — you observe (or can query) individual tensor entries: tensor completion, point
  evaluation, sampling on a grid. Cheapest.
- **`apply`** — you observe linear functionals / sketches `⟨A, w₀⊗…⊗w_{d-1}⟩` of the tensor, or you
  want a single scalar contraction (e.g. a quadratic/multilinear form `A(u,…,u)`). Clean scalar data
  points for a train/validation split.
- **`probe`** — you need the all-but-one-mode contractions: the original Riemannian probing/fitting
  exemplar, building per-mode sketches, or anything where the natural measurement is `d` vectors rather
  than a scalar. Most information per sample, most expensive.

---

## 7. Code pointers

- Frontend: `TuckerTensorTrain.{entries,apply,probe}` and `{entries,apply}_transpose`
  (`t3toolbox/tucker_tensor_train.py`); `T3Tangent.{entries,apply,probe}` and
  `{entries,apply,probe}_transpose` (`t3toolbox/manifold.py`).
- Backend: `backend/entries.py`, `backend/apply.py`, `backend/probing.py`.
- Stacking semantics: [`batching_and_stacking.md`](batching_and_stacking.md) (§11 for the transpose /
  `sum_over_probes` modes). Algorithm ↔ code map: [`probing_section6_notes.md`](probing_section6_notes.md).
  Apply/entries build history: [`dev/archive/apply_entries_handoff.md`](../dev/archive/apply_entries_handoff.md).
