# Why uniform T3s store boolean masks, not integer ranks

> A design-philosophy note for `UniformTuckerTensorTrain` and the `ut3_*` backend. It records *why* the
> real extent of each padded edge is tracked with a **boolean mask** rather than an **integer rank**,
> even though the integer is smaller and reads more directly as "the rank." The short answer is that
> the mask is the unique representation closed under the operations of T3 algebra without moving data.
> As with the other notes, this is an honest accounting — including what the mask choice costs — not
> advocacy.

---

## The two candidate representations

The rank metadata of a uniform T3 must be *stored*, not derived: a padded core cannot tell you its rank
(a "real" slot may be numerically zero; a padded slot may carry garbage). The question is the *format*:

- **(a) Boolean edge masks** (current / `OLD_uniform`): for each edge, a 0/1 vector over the padded
  size marking which slots are real. Equivalently, a diagonal **projector** onto the real subspace.
- **(b) Integer rank arrays**: store the counts (`tucker_ranks`, `tt_ranks`) and rebuild a prefix mask
  `arange(pad) < k` on demand. This presumes the real block is always the **prefix** `[0, k)`.

These coincide *only* if every mask is prefix-contiguous. The heart of the matter is that the natural
operations do not preserve prefix-contiguity.

## The deep reason: closure under ⊕ and ⊗ without data movement

T3 algebra is built from two primitives acting on the underlying core spaces:

- **addition = direct sum (⊕):** block-concatenate cores; ranks add.
- **multiplication = tensor/Kronecker product (⊗):** Kronecker the cores; ranks multiply.

The real subspace is a projector (the mask), and projectors transform *functorially*: under ⊕ they
**concatenate**, under ⊗ they **Kronecker**. Crucially, these produce index subsets that are **not
intervals**, even from interval inputs:

- *Multiply.* `x` real in `{0,1}` of `n_x=3`, `y` real in `{0}` of `n_y=2`. The product's real set in
  the flattened Kronecker index `i·n_y + j` is `{0·2+0, 1·2+0} = {0, 2}` — strided, with a gap at 1.
- *Add.* Concatenating a prefix-with-slack `[0, k_x) ⊂ [0, n_x)` to another leaves a gap on `[k_x, n_x)`
  before the second block — not a prefix.

So the **boolean algebra of subsets** (under disjoint union and product) is *closed* under exactly the
operations T3 algebra is made of. The **integers-as-prefixes** representation is *not*: every `+` and
`×` breaks the prefix invariant, and the only way to restore it is to **gather/scatter the data to the
top-left**. In the stacked setting that compaction is *ragged per element* (each stack slice has a
different rank), which is precisely the irregularity the uniform layout exists to avoid.

Read this way, the mask is "more minimal" in the sense that matters: it is the **minimal state the
operations preserve**. The integer rank is minimal *per object*, but only stays valid if you
continuously renormalize positions — i.e. pay for data movement, or secretly re-derive the positions
(reinventing the mask).

## Why the "must it be a prefix?" worry dissolves

Exactly one operation needs a prefix: **SVD truncation** keeps the top-`k` singular values, which come
out in prefix order, so the truncation mask must be `[0, k)`. But the T3-SVD *re-bases* everything — it
orthogonalizes, runs fresh SVDs, and applies a freshly built prefix mask (`make_uniform_masks` of the
minimal ranks); it ignores whatever pattern the input mask had. So:

- between SVDs, `+` / `×` / `apply` / `entries` carry whatever (possibly gappy) masks the algebra
  produces — no compaction, data stays in place;
- the SVD **re-canonicalizes** to a prefix mask whenever the minimal form is wanted.

The prefix is the *canonical form*; the general mask is the *working form*; the only op that requires the
canonical form is the one that produces it. There is no global prefix invariant to maintain.

**One asymmetry between the two mask kinds, worth stating:** `shape_mask` never goes gappy — the physical
shape is invariant under `+`/`×` — so it stays prefix, which is exactly what lets `to_dense` take a
*static* prefix-slice on the physical axes. The **edge (rank) masks** are the ones that go gappy. So a
gappy mask is fine for *masking-to-zero inside a contraction* (any pattern works), but *extracting a
dense real sub-block* through it (e.g. `ut3_to_t3`) is a gather/`argwhere`, not a slice — or you
re-canonicalize via `ut3svd` first. (The integer-rank alternative would pay that same gather on *every*
`+`/`×`; masks pay it only where a compacted block is actually demanded.)

## Honest pros and cons

**Boolean masks (a):**
- ✅ Closed under ⊕/⊗ (concatenate / Kronecker); zero data movement. The composition niceties
  (add → concat masks, multiply → Kronecker masks, sweep → apply masks) are consequences of this, not
  coincidences.
- ✅ Directly usable in the masking einsums; matches the existing `ut3_*` and `OLD_uniform` machinery.
- ✅ Minimal in the operationally relevant sense (state the algebra preserves).
- ➖ The rank is implicit (`mask.sum(-1)`), so it reads less directly than an integer field.
- ➖ A mask can sit in a non-canonical or meaningless pattern that `validate` cannot cheaply vet — the
  same non-enforced status as orthogonal/minimal/gauged (a numerical-style property, not structural).
- ➖ Masks do **not** save you from rank growth: `n` and `r` still grow under `+`/`×` exactly as ragged
  ranks do, so the padded supercore grows and JIT may retrace. The mask only keeps the bookkeeping clean
  during that growth; it is not a compression.

**Integer ranks (b):**
- ✅ Smallest per object; makes "ranks vary across the stack" visually obvious in the fields; reads
  literally as the *stratum label* of the determinantal variety (`docs/uniform_ranks_and_varieties.md`).
- ✅ Trivially the canonical form.
- ➖ **Not closed**: every `+`/`×` requires a per-stack-element compaction to restore the prefix —
  ragged data movement, the very thing the uniform layer is built to avoid — or it must carry positions
  anyway, which is the mask.

## Other representations we considered (and why they're off the map)

Beyond integer ranks (b), several encodings were weighed and rejected. They each fail one of the two
properties the uniform layer requires — **closure under ⊕/⊗** (so `+`/`×` move no data) and **uniform
shape across the stack** (so `lax.scan` sees one fixed shape) — and, for jit, **host-static**
(resolvable on the host without tracing; see `uniform_pytree_composition.md`).

| representation | closed under +/× | uniform shape across stack | notes |
|---|---|---|---|
| **bool mask (dense array)** ✅ chosen | ✅ concat / Kronecker | ✅ fixed padded size | the working representation; rank reported as `mask.sum(-1)` |
| integer rank **counts** | ❌ gappy ⇒ per-element compaction | ✅ | a count can't denote a gap; compaction is ragged data movement *and* data-dependent shapes ⇒ not jittable |
| hot-**position** lists / "hot rank" tuples | ✅ position arithmetic | ❌ varying length ⇒ ragged | a *sparse* mask; per-element lengths differ ⇒ jagged, defeating the uniform layout (you'd pad+remask — reinvent the mask) |
| **bool tuples** (vs arrays) | ✅ | ✅ | value-hashable, but non-contiguous/boxed ⇒ must materialize for the multiply; slower eager, worse stacked memory; no win over a bool array (which can be value-hashed via its bytes) |
| **float** mask doubling as edge **weights** | — | — | conflates *structure* (static aux, non-differentiable, defines rank, value-neutral) with a *parameter* (traced leaf, differentiable, scales the contraction, value-affecting): opposite jax treatment, and autodiff would silently differentiate — and a grad step corrupt — the "mask." Weights are a separate (parked) concept |

The bool mask is the unique fixed point: **closed ∧ uniform-shaped ∧ host-static.** Counts give up
closure; positions/tuples give up uniformity (or contiguity); float-weights conflate two objects with
opposite autodiff semantics. The one attractive property of the integer/tuple forms — value-based
hashing for jit-cache hits — is real but *separable*: get it on the bool mask via a byte/rank hash
(`uniform_pytree_composition.md`), without surrendering closure or uniformity. (Caveat: a rank *count*
is an insufficient cache key off canonical form — two gappy masks with equal counts but different
positions are different computations — so a general value hash keys on the mask bytes; in canonical
form the count suffices.)

> **Host vs device & jit.** Whichever encoding, the structure must live as **numpy (host)** static
> `aux_data`: any jax op on it inside a trace becomes a tracer. Under jit the host masks fold into the
> compiled program as device constants (no per-call transfer); the eager cost and the deferred
> `jax.device_put` option are in `uniform_pytree_composition.md`. **The `np.*` in mask code is
> intentional — do not "fix" it to `xnp`.**

## A different algebra: tangent (variation) vector-space ops

Everything above describes **tensor algebra** on a `UniformTuckerTensorTrain`: addition is the direct
sum (⊕), multiplication the Kronecker product (⊗), and the masks *concatenate* / *Kronecker* precisely
because the ranks genuinely change. The **basis-variations / tangent layer** (`UT3Variations`, and the
tangent space of a `UT3Basis`) runs a **different algebra**, and the mask behavior is different —
**identical, not combined.**

A variation is a **tangent vector**: a point in the fixed-dimensional tangent space `T_B` at a base
point `B`. Its rank structure (the mask) is a property of **`B`'s gauge**, shared by *every* tangent at
`B`. So the vector-space operations there are **corewise at a fixed rank**, not direct sums:

| op | mask |
|---|---|
| `v + w`, `v - w` (both tangents at `B`) | **identical** — `v` and `w` carry `B`'s mask; the sum is another vector of `T_B` with the *same* mask. No concat. |
| `α · v`, `-v` | unchanged (scaling cannot change which slots are real) |
| `sum_stack` (sum a batch of tangents) | OR the mask over the summed stack axes — a **no-op** when the batch shares a base (`K`-stack, one mask); a genuine union only for a varying-rank stack |

This mirrors the ragged layer exactly: `T3Variations.__add__` is a *corewise* add of equal-shaped cores
(it has no direct-sum/concat at all), because tangent addition stays inside one fixed-dimensional space.
The ⊕/⊗ closure argument above is about *tensors*; the tangent algebra never invokes it.

**Consequence — why the masks-as-`aux_data` worry does not bite here.** Because add/sub/scale **do not
change the mask**, they create no new static structure: no fresh jit cache key, no recompile, and the
result still pairs with its base (`check_ubv_pair`) since two tangents at one base share that base's
mask. Mask *changes* are confined to the tensor layer — where a rank change is a real new structure, and
`ValueHashedMasks` keeps a rebuilt-but-identical mask from recompiling anyway (see
`uniform_pytree_composition.md`).

**One thing uniform must enforce that ragged gets for free.** In ragged, adding mismatched variations
fails loudly (a numpy shape error). In uniform the supercores are *padded to a common shape*, so two
variations with **different masks but equal padded shape** would `corewise_add` silently to a **wrong**
result — the padding hides the mismatch. So uniform variation add/sub carries an explicit **same-mask
structural precondition** (`shape` and the masks must match; masks are host numpy, so it is a cheap
`array_equal` valid even under jit). It is the variation-level analog of the same-frame guard on tangents.

## Recommendation and the relationship between the two

Store **boolean masks**, treat the prefix form as the *canonical form the SVD produces* rather than a
maintained invariant, and still *report* the rank as `mask.sum(-1)`. The integer rank remains the right
way to **name** a stratum of the variety; the mask is the right way to **track which coordinates realize
that stratum** through a computation. They agree in canonical form and diverge only in the working forms
that `+`/`×` create — which is exactly where the mask earns its place.

See also `docs/uniform_supercore_layout.md` (why the core index leads) and
`docs/uniform_ranks_and_varieties.md` (what a stacked uniform T3 represents).
