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

## Recommendation and the relationship between the two

Store **boolean masks**, treat the prefix form as the *canonical form the SVD produces* rather than a
maintained invariant, and still *report* the rank as `mask.sum(-1)`. The integer rank remains the right
way to **name** a stratum of the variety; the mask is the right way to **track which coordinates realize
that stratum** through a computation. They agree in canonical form and diverge only in the working forms
that `+`/`×` create — which is exactly where the mask earns its place.

See also `docs/uniform_supercore_layout.md` (why the core index leads) and
`docs/uniform_ranks_and_varieties.md` (what a stacked uniform T3 represents).
