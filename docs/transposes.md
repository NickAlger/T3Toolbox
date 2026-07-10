# Transposes of `apply` / `entries` — ambient, corewise, and tangent

> **If you need the transpose / adjoint / "back-projection" of `apply` or `entries`** — for a gradient,
> a Gauss–Newton solve, or to build a tensor from measurements — **read this first.** There are
> **three** different transposes. They are **not interchangeable**, they return different things, and
> the one most people reach for by default (the *corewise* gradient) is deliberately **not** given the
> unqualified name. This doc explains what each one is, which to use, and what each costs. For the
> forward operations see [`entries_apply_probe.md`](entries_apply_probe.md); for the `sum_over_probes`
> stacking mechanics see [`batching_and_stacking.md`](batching_and_stacking.md) §11.

Math reference: **§6.2.3** ("Transpose of tangent vector to probes map", Algorithm 8) and **§6.3**
("Corewise (non-manifold) simplification") of Alger, Christierson, Chen & Ghattas (2026), *"Tucker
Tensor Train Taylor Series"* (arXiv:2603.21141). (As elsewhere, the local `t4s.pdf` numbering is newer
than the public arXiv.)

---

## TL;DR — three transposes, one picture

The forward `apply` is **linear in the tensor `X`**: `apply(X) = ⟨X, w₀ ⊗ ⋯ ⊗ w_{d-1}⟩`. Its honest
adjoint sends a residual `c` to the rank-1 tensor

```
c · (w₀ ⊗ w₁ ⊗ ⋯ ⊗ w_{d-1})            # the "ambient back-projection"
```

The other two transposes are *that same object, **projected** onto a smaller space*:

| transpose | adjoint of … | returns | base point? | reach for it when … |
|---|---|---|---|---|
| **ambient** — `TuckerTensorTrain.apply_ambient_transpose(c, ww)` | `apply` on the full tensor space | raw tuple of **canonical (CP) factors** (`c·⊗w`) | **no** (static) | you want the literal back-projection: a building block, a rank-1 update, to combine back-projections across base points, or to hand off to a CP/general-tensor library |
| **corewise** — `X.apply_corewise_transpose(c, ww)` | the **core parametrization** at `X` | raw tuple `(tucker_grads, tt_grads)` | **yes** (`self`) | you optimize the cores directly with a **core-wise optimizer** (Adam, L-BFGS, plain SGD) |
| **tangent** — `T3Tangent.apply_transpose(c, ww, frame)` | the **manifold tangent** map at `X` | a `T3Tangent` | **yes** (orthogonal, gauged) | you do **Riemannian** optimization on the fixed-rank manifold |

The relationships:

- **corewise** = the ambient back-projection **projected onto the core-parametrized tangent space** at
  `X` (the span of single-core perturbations).
- **tangent** = the ambient back-projection **projected onto the gauged manifold tangent space** at `X`.

Everything below is identical for `entries` (`entries_ambient_transpose`,
`entries_corewise_transpose`, `T3Tangent.entries_transpose`) and for `probe`
(`probe_ambient_transpose`, `probe_corewise_transpose`, `T3Tangent.probe_transpose`) — all three
sampling operations have all three transposes. `probe` has one structural twist; see *Probe transposes*
below.

---

## Why the explicit names (`ambient` vs `corewise`)

Most users — reasonably — expect "the transpose of `apply`" to be the **gradient of a loss with
respect to the cores**: you have cores, you computed a measurement, you want `∂loss/∂cores`. That is
the **corewise** transpose. But it is *not* the literal adjoint of `apply`:

- `apply` is the linear map `X ↦ ⟨X, ⊗w⟩`. Its adjoint, in the textbook sense `⟨Aᵀc, X⟩ = ⟨c, A X⟩`,
  is `c·⊗w` — frame-free, representation-free. That is the **ambient** transpose.
- The corewise gradient is the adjoint of a *different* (composite, frame-dependent) map
  `cores ↦ measurement`.

If `apply_transpose` were unqualified it would have to mean one of these, and whichever we picked, half
of all users would silently get the wrong object. So both carry an explicit qualifier. The
static-vs-instance split reinforces it: the ambient transpose is **frame-free** (a `@staticmethod` — it
depends on nothing but `c` and the probe vectors), while the corewise transpose differentiates *your*
cores (an instance method, `self` is the base point).

---

## The three, in detail

### Ambient — `apply_ambient_transpose(c, ww, sum_over_probes=…)`  *(static)*

The literal adjoint of `apply`, returned as a **raw tuple of canonical (CP) factors** — `len = d`, one
factor per mode, `c` folded into the first. It depends on nothing but `c` and the probe vectors; there
is no base point.

CP is the *natural* type here, by duality: `apply` consumes one vector per mode and returns a scalar;
its adjoint takes a scalar and returns one (scaled) vector per mode. In CP form the operation is almost
a relabelling —

```
apply_ambient_transpose(c, ww)  =  (c⊙w₀, w₁, …, w_{d-1})        # CP factors, c folded into factor 0
```

— which is exactly what an honest adjoint of a multilinear form should look like. (Wrapping this in a
`TuckerTensorTrain` would *add* an identity copy-tensor as the TT cores — pure structure, no
information.)

- **Return convention:** factors match what `from_canonical` consumes — `len = d`,
  `factor[i].shape = stack_shape + (R, Nᵢ)`. So `from_canonical(apply_ambient_transpose(c, ww))` round-trips
  to the back-projection as a `TuckerTensorTrain`, if you want it in T3 form (see *Cost* for when that
  conversion is expensive).
- **Defining property (and how you test it):** once realized as a tensor,
  `⟨from_canonical(apply_ambient_transpose(c, ww)), X⟩ == c · X.apply(ww)` for every `X`.
- **`sum_over_probes`** picks where the probe stack `W` lands — and both are cheap in CP:
  - `False` (default): `W` is a **stacking** axis; the result is a `W`-stack of rank-1 CP tensors
    (`R = 1`) — one back-projection per probe.
  - `True`: `W` becomes the **CP rank**; the result is one rank-`|W|` CP tensor `Σ_W c_W (⊗w^W)`,
    stored in `O(d·|W|·N)` (see *Cost* — no blow-up, because CP keeps the shared rank index implicit).
- **Composable / interop.** It lives in the full ambient tensor space, so you can add ambient
  back-projections from *different* base points or iterations, project onto *any* tangent space, etc.
  And because it's a plain factor tuple — not a bespoke class — it hands off directly to dedicated CP /
  general-tensor libraries. (This library is deliberately laser-focused on T3, not a general CP
  library; the raw tuple is the clean boundary.)
- **`entries` variant** (`entries_ambient_transpose(c, index, shape, sum_over_probes=…)`): the CP
  factors are the one-hot vectors `e_{indexᵢ}`, fully determined by the indices (`shape` supplies the
  ambient dims, which `c` and `index` alone don't). Same semantics otherwise.

### Corewise — `X.apply_corewise_transpose(c, ww, sum_over_probes=…)`  *(instance)*

The gradient of the measurement with respect to `X`'s cores, treated as independent optimization
variables. Returns a **raw tuple** `(tucker_grads, tt_grads)` whose arrays have the **exact shapes of
`X`'s cores** — so each entry maps one-to-one onto a core you are optimizing.

- This is the operation behind standard (non-manifold) optimizers: `cores ← cores − lr · grad`.
- It is implemented by the §6.3 substitution into the tangent machinery (Algorithm 8 with the TT cores
  used in place of the orthogonal left/right/down frames, and the Tucker cores no longer required to be
  orthogonal). You do **not** need an orthogonal representation to call it.
- **Returns a raw tuple, not a `TuckerTensorTrain`,** on purpose: the result is a *gradient*, not a
  tensor. Wrapping it in a `TuckerTensorTrain` would invite `X − lr·grad`, which silently does
  tensor-space arithmetic (rank concatenation), not the core-wise step you want.
- **Caveat (from §6.3):** the cores are a *representation*, not an intrinsic object, so a corewise
  gradient is meaningful only at the base point where it was taken. Combining corewise gradients from
  different base points (as L-BFGS history does) is, strictly, ill-defined from the manifold viewpoint.
  Core-wise optimizers do it anyway and it works in practice; just know that is what is happening. If
  you want the principled version, use the **tangent** transpose.

### Tangent — `T3Tangent.apply_transpose(c, ww, frame, sum_over_probes=…)`  *(static, takes a frame)*

The Riemannian Jacobian transpose: back-projects `c` into a **tangent vector** at a base point on the
fixed-rank manifold. Returns a `T3Tangent`. Requires an orthogonal (gauged) frame; the resulting tangent
is the Riemannian gradient after the gauge projection. This is the operation for Riemannian
optimization (Newton-CG, RGD); see the worked example `examples/fit_hilbert_tensor_newton_cg.py`. It
lives on `T3Tangent`, so the class name already says "tangent" — no qualifier needed.

---

## Cost — and why the representation matters

The crucial fact is **where the cost lives**, and CP versus T3 puts it in different places.

- **Ambient (CP), both modes:** `O(d·|W|·N)`. Unsummed is a `W`-stack of rank-1 CP; summed is one
  rank-`|W|` CP. Summed is *not* more expensive, because **CP keeps the shared rank index implicit** —
  `Σ_W c_W (⊗w^W)` is just the factor matrices `[wᵢ^W] ∈ ℝ^{|W|×Nᵢ}` plus the weights `c`. Linear in
  `|W|`. This is the whole reason CP is the right output type, and why `apply_ambient_transpose` *keeps*
  `sum_over_probes`.
- **The `|W|²` cost is real, but it lives in `from_canonical` (CP → T3), not in the transpose.** A
  *dense Tucker tensor train* of a rank-`|W|` object must materialize the "all factors share one index"
  coupling as a diagonal/copy core — `≥ |W|²` storage (and there's no exact, data-independent way
  around it: the minimal ranks are data-dependent, which would break stacking and give non-exact
  results). By returning CP, the transpose stays cheap and that conversion cost becomes the user's
  **explicit opt-in** — incurred only if and when they actually convert to T3, which is exactly where it
  belongs.
- **Corewise / tangent, summed (`sum_over_probes=True`):** the gradient is shaped like the **cores**
  (corewise) or the **tangent space** (tangent) — a *fixed* size set by `X`, not by `|W|`. Summing over
  `W` collapses the `|W|` rank-1 contributions **into** those fixed-shape objects. **No `|W|` ever
  appears in a rank.** `sum_over_probes=True` here is the Gauss–Newton `Jᵀr`, and it is cheap.

So for an optimization gradient, prefer corewise/tangent (fixed-size, fused). The summed *ambient* CP
is the honest "ambient Euclidean gradient as a tensor" — cheap to form, but to *use* it (project, fit)
you'd pay `O(|W|)` anyway, so it's mainly for the general/unpredictable case, sketching, or interop, not
the optimization hot path.

### The forward/transpose `N`-asymmetry (especially for `entries`)

Forward `entries` is **`N`-free**: it reads `X[i]` by *slicing* the cores at the index, so the ambient
mode dimension `N` never enters the arithmetic. The transposes are not `N`-free — but that is intrinsic
to the *output*, not a flaw of the construction. Their outputs are tensors / tangents with
`N`-dimensional modes, so emitting the result is unavoidably `Ω(N)`. The forward gets to skip `N`
because its output (a scalar) contains no `N`; the transpose's output does. (Inside a Riemannian solve
even this is sidestepped: you never need the standalone back-projection, only its inner products
against tangent directions, and those fuse back to `N`-free slicing by adjointness.)

---

## Which one do I want?

```
Do you want the back-projected TENSOR itself (a building block, a rank-1
update, something to combine across base points, or to hand to a CP library)?
        └── yes → apply_ambient_transpose            (raw CP factor tuple; → from_canonical for a T3)

Are you optimizing the cores directly with Adam / L-BFGS / SGD?
        └── yes → X.apply_corewise_transpose         (raw (tucker_grads, tt_grads))

Are you doing Riemannian optimization on the fixed-rank manifold?
        └── yes → T3Tangent.apply_transpose          (a T3Tangent)
```

All three have `entries` **and `probe`** counterparts with the same semantics, and **all three take
`sum_over_probes`** (`True` = sum the probe stack `W`; `False` = keep it as a stack, one result per
probe — useful for assembling `JᵀJ` matrix-vector products). The difference is *where the sum lands*:
cheap in all three (CP rank for ambient; the fixed-size cores/tangent for corewise/tangent), but only
corewise/tangent keep the result at a fixed size independent of `|W|`.

---

## Probe transposes (the one structural twist)

`probe` contracts all-but-one mode, returning **`d` vectors** (one free mode each) instead of a scalar.
So its transposes take a residual that is **`d` vectors** `ztildes = (ž₀,…,ž_{d-1})` (plus the probe
vectors `ww`), where `apply`/`entries` took a scalar residual `c`. Everything else is the same three
transposes:

- **`probe_ambient_transpose(ztildes, ww)`** (static) — the literal adjoint of `probe`. The
  back-projection is `Σᵢ (w₀ ⊗ … ⊗ žᵢ ⊗ … ⊗ w_{d-1})` (residual `žᵢ` in slot `i`, probe vectors
  elsewhere) — a **rank-`d` CP tensor** (vs rank-1 for `apply`), returned as CP factors. Summed → rank
  `d·|W|`, still `O(d|W|N)`.
- **`X.probe_corewise_transpose(ztildes, ww)`** (instance) — the gradient of `X.probe(ww)` w.r.t. `X`'s
  cores, for non-manifold optimizers fitting from probes. Raw `(tucker_grads, tt_grads)`, no `|W|`
  blow-up.
- **`T3Tangent.probe_transpose`** — the Riemannian gradient (the paper's original probing Jacobian
  transpose).

Probing is the paper's exemplar operation, so this completes the full 3×3 grid
(`entries`/`apply`/`probe` × ambient/corewise/tangent).
