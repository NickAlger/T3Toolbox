# Rank masks — the decision record (why they exist, and why boolean)

> The consolidated mask decision record, in two parts: **why the masks exist at all** (they cannot
> be dropped — they are the functional rank constraint that makes variable-rank optimization work),
> and **why the representation is boolean masks rather than integer ranks** (the closure argument,
> the honest ledger, and the rejected alternatives). The user-facing "how masks behave" story is
> `docs/uniform_masks_vs_ranks.md`. Pairs with `docs/uniform_ranks_and_varieties.md` (what the
> object represents), `numerical_contract_catalog.md` (the minimal-rank audit),
> `uniform_svd_prefix_orthogonalization.md`, and `uniform_pytree_composition.md`.

## Part I — why the masks exist (and why "just inflate to uniform rank" fails)

## What the masks do

A uniform Tucker tensor train pads each edge to a common size `R` — so a whole *stack* of T3s can live in
one array, `lax.scan`-friendly — and marks slots `[0, rank)` real with a prefix mask. The real rank may
differ **per edge and per stack element**: that is the *determinantal variety* — fixed shape, ranks that
vary (`docs/uniform_ranks_and_varieties.md`). In the tangent / optimization layer the masks **zero the
variation in the padded slots**, so a gradient step cannot put content beyond the real rank.

## Why variable ranks matter

Real data rarely wants one rank everywhere. A good T3 fit may need a small rank on one edge and a large
one on another; forcing a single rank either **underfits** (rank too low — accuracy left on the table) or
**overfits** (rank too high — fitting noise, wasting compute). *Rank continuation* exploits exactly this:
start low and grow ranks edge-by-edge under a conditioning criterion
(`docs/rank_continuation.md`, `examples/fit_varied_rank_tensor_newton_cg.py`). All of it presumes you can
hold ranks at chosen, possibly-varied values — i.e. that the optimizer **respects the rank structure**.

## The tempting simplification — and why it fails

The minimal-rank audit (`numerical_contract_catalog.md`, empirically verified against the dense oracle)
found that **no manifold operation requires minimal rank**: `inner`/`norm` need only an orthonormal,
gauged frame; `project`/`transport`/gauge need only orthonormality; `retract` is valid on a non-minimal
frame. Equivalently, for a **given** tangent vector, inflating its ranks to the padded size (orthonormally
completing the frame, zero-extending the variations) leaves every operation's result unchanged:

```
op(inflate(tangent)).to_dense()  ==  op(tangent).to_dense()
```

So why not drop the masks entirely, work on the padded supercores, and treat every slot as real? It would
be a large simplification — no static structure in the jit cache key, so the per-step-recompile problem
would vanish for the tangent layer.

**Because optimization does not operate on a fixed tangent — it computes a new one (the gradient) every
step.** At a rank-deficient frame the projected gradient generally **has content in the padded
("completion") directions** — moving there *increases* the rank. The masks are exactly what zero that
content. Drop them and the gradient grows every edge toward the common padded rank `R`: you lose rank
control and overfit — the precise failure the variable-rank feature exists to prevent.

So the maskless equivalence is genuine, but only for **operations on a fixed tangent**. The masks are not
bookkeeping; they are a **functional rank constraint** that makes variable-rank fitting work. Keeping them
is the deliberate choice.

## The cost of keeping masks, and how it is paid

Masks are static structure, so they ride in jax `aux_data` and become part of the `jit` cache key — which
*risks a recompile* every time the frame is rebuilt (i.e. every optimization step). Two design choices
remove that cost:

- **SVD-based orthogonalization** places the real content in the leading slots, so the masks come out a
  **deterministic prefix** — bit-identical every step at fixed rank
  (`uniform_svd_prefix_orthogonalization.md`).
- **Value-hashed mask holders** key the `jit` cache on mask *content*, so a rebuilt-but-identical holder
  is a cache hit (`uniform_pytree_composition.md`).

Together: **no recompile within a fixed-rank stage; a recompile only when the ranks genuinely change**
(e.g. a rank-continuation step) — which is correct, and rare.

## Part II — why boolean masks rather than integer ranks (the representation decision)

*(Excised from the pre-split `uniform_masks_vs_ranks.md`, whose user-facing successor carries the
"how masks behave" story; this part preserves the decision ledger.)*

### The two candidate representations

The rank metadata of a uniform T3 must be *stored*, not derived: a padded core cannot tell you its
rank (a "real" slot may be numerically zero; a padded slot may carry garbage). The question is the
*format*:

- **(a) Boolean edge masks** (chosen): for each edge, a 0/1 vector over the padded size marking
  which slots are real. Equivalently, a diagonal **projector** onto the real subspace.
- **(b) Integer rank arrays**: store the counts (`tucker_ranks`, `tt_ranks`) and rebuild a prefix
  mask `arange(pad) < k` on demand. This presumes the real block is always the **prefix** `[0, k)`.

These coincide *only* if every mask is prefix-contiguous — and the natural operations do not
preserve prefix-contiguity (the concat/Kronecker gap examples are in the user doc). The **boolean
algebra of subsets** (under disjoint union and product) is *closed* under exactly the operations
T3 algebra is made of; the **integers-as-prefixes** representation is not: every `+` and `×`
breaks the prefix invariant, and the only way to restore it is to gather/scatter the data to the
top-left. In the stacked setting that compaction is *ragged per element* — precisely the
irregularity the uniform layout exists to avoid. Read this way, the mask is "more minimal" in the
sense that matters: it is the **minimal state the operations preserve**. The integer rank is
minimal *per object*, but only stays valid if you continuously renormalize positions — i.e. pay
for data movement, or secretly re-derive the positions (reinventing the mask).

### Honest pros and cons

**Boolean masks (a):**
- ✅ Closed under ⊕/⊗ (concatenate / Kronecker); zero data movement. The composition niceties
  (add → concat masks, multiply → Kronecker masks, sweep → apply masks) are consequences, not
  coincidences.
- ✅ Directly usable in the masking einsums.
- ✅ Minimal in the operationally relevant sense (state the algebra preserves).
- ➖ The rank is implicit (`mask.sum(-1)`), so it reads less directly than an integer field.
- ➖ A mask can sit in a non-canonical or meaningless pattern that `validate` cannot cheaply vet —
  the same non-enforced status as orthogonal/minimal/gauged.
- ➖ Masks do **not** save you from rank growth (stated user-side too).

**Integer ranks (b):**
- ✅ Smallest per object; reads literally as the *stratum label* of the determinantal variety.
- ✅ Trivially the canonical form.
- ➖ **Not closed**: every `+`/`×` requires a per-stack-element compaction to restore the prefix —
  ragged data movement — or it must carry positions anyway, which is the mask.

### Other representations considered (and why they're off the map)

Each fails one of the required properties — **closure under ⊕/⊗**, **uniform shape across the
stack**, and (for jit) **host-static** (`uniform_pytree_composition.md`):

| representation | closed under +/× | uniform shape across stack | notes |
|---|---|---|---|
| **bool mask (dense array)** ✅ chosen | ✅ concat / Kronecker | ✅ fixed padded size | the working representation; rank reported as `mask.sum(-1)` |
| integer rank **counts** | ❌ gappy ⇒ per-element compaction | ✅ | a count can't denote a gap; compaction is ragged data movement *and* data-dependent shapes ⇒ not jittable |
| hot-**position** lists / "hot rank" tuples | ✅ position arithmetic | ❌ varying length ⇒ ragged | a *sparse* mask; per-element lengths differ ⇒ jagged, defeating the uniform layout (you'd pad+remask — reinvent the mask) |
| **bool tuples** (vs arrays) | ✅ | ✅ | value-hashable, but non-contiguous/boxed ⇒ must materialize for the multiply; slower eager, worse stacked memory; no win over a bool array (which can be value-hashed via its bytes) |
| **float** mask doubling as edge **weights** | — | — | conflates *structure* (static aux, non-differentiable, defines rank, value-neutral) with a *parameter* (traced leaf, differentiable, scales the contraction, value-affecting): opposite jax treatment, and autodiff would silently differentiate — and a grad step corrupt — the "mask." Weights are a separate (parked) concept |

The bool mask is the unique fixed point: **closed ∧ uniform-shaped ∧ host-static.** The one
attractive property of the integer/tuple forms — value-based hashing for jit-cache hits — is real
but *separable*: get it on the bool mask via a byte hash (`uniform_pytree_composition.md`).
(Caveat: a rank *count* is an insufficient cache key off canonical form — two gappy masks with
equal counts but different positions are different computations — so the value hash keys on the
mask bytes; in canonical form the count suffices.)

### Maintainer guard: host numpy is intentional

Whichever encoding, the structure must live as **numpy (host)** static `aux_data`: any jax op on
it inside a trace becomes a tracer. Under jit the host masks fold into the compiled program as
device constants (no per-call transfer); the eager cost and the deferred `jax.device_put` option
are in `uniform_pytree_composition.md`. **The `np.*` in mask code is intentional — do not "fix" it
to `xnp`.**
