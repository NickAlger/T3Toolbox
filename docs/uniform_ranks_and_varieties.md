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

## Relationship to ragged `TuckerTensorTrain`, and the one real cost

A ragged stacked `TuckerTensorTrain` *physically cannot* carry varying ranks — its cores are
single-shape arrays, so uniform rank is forced on it. `UniformTuckerTensorTrain` exists precisely
because the mask layout can express more. The relationship is clean and worth keeping in mind:

- uniform-rank UT3 stacks ↔ ragged stacked T3 (a faithful round-trip), and
- UT3 *additionally* represents varying-rank stacks that a single ragged stacked T3 cannot.

The honest cost shows up at exactly the two places where the manifold matters:

1. **Round-trip.** `ut3_to_t3` of a varying-rank stack can only return a *tree* of distinct T3s, not
   one stacked `TuckerTensorTrain` — which is the truthful answer, since the elements really do have
   different ranks.
2. **The future tangent layer.** A tangent vector needs a single tangent space, i.e. a base sitting at
   a *smooth* point — uniform ranks across the base (`C`) stack. A varying-rank stack has no single
   tangent space.

We handle this the way the project already handles orthogonal / minimal / gauged: as a **non-enforced
precondition** of the manifold-flavored operations — documented and checkable, not a construction-time
error — while shape consistency stays a structural hard error. For the plain UT3 layer (linear
algebra, SVD-with-mask truncation, entries/apply/probe), varying ranks cost nothing and simply work;
the precondition only bites once we reach tangents, where it is the natural requirement anyway.
