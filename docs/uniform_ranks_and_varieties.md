# Why uniform T3s let ranks vary across the stack (but not shape)

> A design-philosophy note for `UniformTuckerTensorTrain` and the `ut3_*` backend. It records *why*
> we let the ranks of a stacked uniform T3 differ from one stack element to the next — and why the
> physical shape is, by contrast, locked across the stack. The aim is the reasoning, not a rulebook:
> if you understand the manifold-vs-variety picture below, the rules fall out on their own.

---

## The representation, and what the mask actually means

A `UniformTuckerTensorTrain` pads a (possibly ragged) Tucker tensor train up to common sizes `n` (Tucker
rank), `r` (TT rank), `N` (mode dimension), stacks the `d` cores onto a leading axis to form
*supercores*, and carries boolean **edge masks** marking the real extent of each padded edge.

The first thing to be clear about: **the mask does not define the dense tensor — the zeros do.** A
stack element whose Tucker rank is `n_s < n` simply has zero columns in the padded slots, so it
represents a tensor of rank `≤ n_s` regardless of the mask. Masking-down and "a rank-`n` core that
happens to have zero columns" are the same tensor.

What the mask *does* define is the **rank** — equivalently, *which stratum of the rank hierarchy the
point sits on*. That is exactly the information that rank-aware operations need: where SVD truncation
should cut, how `ut3_to_t3` extracts the real sub-block, whether the ranks are minimal. So the mask is
per-element metadata: each stack element carries its own "official rank," static and known at trace
time (which is why masks ride along as static aux data under `jit`).

## The real question: a batch on a manifold, or a batch in a variety?

Because the mask is a stratum label, "should ranks vary across the stack?" is really:

> Is a stacked uniform T3 a batch of points on **one smooth manifold**, or a batch of points in **one
> bounded-rank variety**?

Both have a clean mathematical home:

- **Fixed rank (uniform across the stack).** The set of T3s with ranks *exactly* `(n, r)` is a smooth
  **manifold** `M₍ₙ,ᵣ₎`. This is the home of everything Riemannian — tangent spaces, retraction,
  gauge, the Taylor-series machinery the library is named for (T4S). A uniform-rank stack is a batch of
  points on a single such manifold.

- **Varying rank (padded to `(n, r)`, one mask per element).** Tucker ranks are matricization ranks
  and TT ranks are unfolding ranks — both ordinary *matrix* ranks — so the set of tensors with ranks
  **≤ `(n, r)`** is a **determinantal variety** `V₍≤ₙ,≤ᵣ₎`, cut out by the vanishing of unfolding/
  matricization minors. It is the closure `⋃₍ranks ≤ (n,r)₎ M₍ranks₎`, stratified by the lower-rank
  manifolds and singular along them. The padded-supercore-plus-mask representation is *exactly* a
  parametrization of a point in this variety, with the mask naming the stratum.

We choose the **variety**. The decisive reason is a use case the manifold cannot serve well: **ideal
ranks are rarely known a priori.** Sweeping a batch of candidate ranks at once — to see which performs
best on a task — is a batch of points at *different* ranks, i.e. a batch in the variety. The mask
layout expresses this for free (the padded supercore shapes stay fixed, so `jit` is unbothered), and
it is a strict superset of the fixed-rank case: you can always restrict to one stratum when you want
the manifold.

## Why shape is locked, and the principled asymmetry

Shape is treated oppositely — it is **fixed across the stack** — and the contrast is not arbitrary:

- **Shape determines the *ambient space*** `ℝ^{N₀×…×N_{d-1}}`. Stacking means "a batch of objects living
  in the *same* space." Let the shape vary and `to_dense` could not even return one
  `stack_shape + (N₀, …)` array; "stack" would lose its meaning.
- **Rank determines *model complexity within* that ambient space** — which point of the variety, which
  stratum. That is free to vary, because every element still lives in the one shared space.

So shape consistency is a **structural invariant** (a hard error if violated), while a varying rank is
just a different point of the same variety. The asymmetry is the right one: same ambient space,
different models inside it.

(One nuance, since "shape fixed" is easy to misread: it means fixed *across the stack*. Shape still
varies *across modes* — `N₀,…,N_{d-1}` differ — which the supercore handles by padding each mode to
`N=max(Nᵢ)` plus `shape_mask`. That cross-*mode* padding is a separate axis from the cross-*stack*
invariant; only the latter is the structural hard error.)

## Relationship to ragged `TuckerTensorTrain`, and the one real cost

A ragged stacked `TuckerTensorTrain` *physically cannot* carry varying ranks — its cores are
single-shape arrays, so uniform rank is forced on it. `UniformTuckerTensorTrain` exists precisely
because the mask layout can express more. The relationship is clean and worth keeping in mind:

- uniform-rank UT3 stacks ↔ ragged stacked T3 (a faithful round-trip), and
- UT3 *additionally* represents varying-rank stacks that a single ragged stacked T3 cannot.

This is exactly why the **structural** rank checker is shaped differently in the two layers: `has_minimal_ranks`
is a **scalar in ragged** (one core shape ⇒ ranks shared across the stack, so there is nothing per-element to
report) but **per-element (shape `stack_shape`) in uniform** (ranks vary across the stack). The *numerical*
checkers (`is_orthogonal`, etc.) are per-element in both layers — see `docs/batching_and_stacking.md` §9.

The honest cost shows up at exactly one place — the ragged round-trip:

1. **Round-trip.** `ut3_to_t3` of a varying-rank stack can only return a *tree* of distinct T3s, not
   one stacked `TuckerTensorTrain` — which is the truthful answer, since the elements really do have
   different ranks. (`to_dense` is unaffected: the ambient shape is uniform across the stack.)

## The tangent layer keeps the varying-rank stack — uniform rank is a `K`-stack property, not a `C`-stack one

It is tempting to think a *tangent* forces uniform rank across the stack ("a tangent needs a single
tangent space, so the stack must be one smooth manifold point"). That conflates a **single** tangent
vector with a **stack** of them. The distinction is `K` (tangent stack) vs `C` (base stack); see
`docs/batching_and_stacking.md`:

- A single tangent vector — or a **tangent (`K`) stack**, a bundle of tangents sharing *one* base — lives
  in **one** tangent space, so its rank is uniform. This is automatic (they share a base) and is the *one*
  genuine uniform-rank requirement. The bv-pair check enforces it structurally: the variation rank masks
  must be **constant along `K`** (= the base's gauge-shifted masks broadcast over `K`).
- A **base (`C`) stack** is a batch of tangents at *different* bases. It is a tangent vector to the
  **product** manifold `M_{n_1,r_1} × … × M_{n_C,r_C}`, which is smooth even when the factors are different
  fixed-rank manifolds; the per-element tangent spaces may have different dimensions and nothing couples
  them. So **ranks may vary across `C` exactly as in the plain layer** — the rank-sweep use case (a batch
  of candidate models at different ranks, compared against each other) is a varying-`C` stack of tangents.

The price of varying `C` is a **discipline, not a precondition**: every tangent / manifold op is
per-element-mask-aware (it must be anyway for the masks to mean anything), so it vectorizes over `C` with
each element applying its own mask — `tangent_space_dimension`, `inner`/`norm`, gauge, retraction are all
per-element. The one subtle spot is a batched Riemannian solve (Gauss-Newton / CG) over a varying-rank `C`
stack: the masked-out directions must sit in `ker(J)` so they contribute zero to the CG reductions
(`rᵀr`, `pᵀAp`) and never pollute the active per-element solve — which holds as long as the gauge
projection and `J`/`Jᵀ` respect the masks. If some specific op ever turns out to *genuinely* need uniform
`C` rank, add a **narrow non-enforced precondition there** (the orthogonal / gauged pattern), never a
blanket construction-time restriction; shape consistency stays the only structural hard error.

*(Decided 2026-06-30, after the earlier draft of this note over-restricted tangents to uniform `C`.
Varying-`C` is verified to work through the `UT3Tangent` skeleton and is a first-class case in the
uniform-tangent equivalence-contract tests.)*
