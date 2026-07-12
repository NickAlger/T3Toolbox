# How the rank masks behave

> Every `UniformTuckerTensorTrain` (and uniform frame/variation) tracks the *real* extent of each
> padded edge with a **boolean mask**. This page is how to read and work with those masks: what a
> mask means, how it changes under arithmetic (it can go non-prefix — that is normal), what the
> masks do for you during optimization, and the one structural precondition they add. The decision
> record behind the representation — why boolean masks rather than integer ranks, and the
> alternatives that were rejected — is
> [`contributor/uniform_rank_masks_rationale.md`](contributor/uniform_rank_masks_rationale.md).

## What a mask means

A padded core cannot tell you its rank by itself: a "real" slot may be numerically zero, and a
padded slot may carry garbage. The mask is the stored answer — for each edge, a 0/1 vector over the
padded size marking which slots are real; equivalently, a diagonal **projector** onto the real
subspace. The integer rank is *derived*: `mask.sum(-1)`. The two views cooperate: the integer
**names** the stratum of the determinantal variety your object lives in
(`docs/uniform_ranks_and_varieties.md`); the mask **tracks which coordinates realize that stratum**
through a computation.

## What the masks do for you in optimization

The masks are not bookkeeping — they are a **functional rank constraint**. In the tangent /
optimization layer the masks zero the variation in the padded slots, so **a gradient step cannot
put content beyond the real rank**: at a rank-deficient frame the projected gradient generally has
components in the padded ("completion") directions, and without the masks every edge would grow
toward the common padded rank — losing rank control and overfitting. Variable ranks are the point:
a good fit may need a small rank on one edge and a large one on another, and *rank continuation*
(`docs/rank_continuation.md`) grows ranks edge-by-edge on exactly this machinery.

The compile-time consequence is equally observable: masks are static structure (jax `aux_data`),
and the library arranges — SVD-prefix orthogonalization plus value-hashed mask holders — that
**nothing recompiles within a fixed-rank stage; a recompile happens exactly when the ranks
genuinely change** (e.g. a rank-continuation step), which is correct and rare.

Note what masks are **not**: a compression. `n` and `r` still grow under `+`/`×` exactly as ragged
ranks do, so the padded supercore grows and jit may retrace; the mask only keeps the bookkeeping
clean during that growth.

## Masks change under arithmetic — and can go non-prefix

T3 **tensor** algebra is built from two primitives on the underlying core spaces: addition is a
direct sum (cores block-concatenate; ranks add) and multiplication is a Kronecker product (ranks
multiply). The masks transform the same way — under `+` they **concatenate**, under `×` they
**Kronecker** — and these produce real-slot patterns that are **not intervals**, even from interval
inputs:

- *Multiply.* `x` real in `{0,1}` of `n_x=3`, `y` real in `{0}` of `n_y=2`. The product's real set
  in the flattened Kronecker index `i·n_y + j` is `{0·2+0, 1·2+0} = {0, 2}` — strided, with a gap.
- *Add.* Concatenating a prefix-with-slack `[0, k_x) ⊂ [0, n_x)` to another block leaves a gap on
  `[k_x, n_x)` before the second block — not a prefix.

So after `+`/`×` your object's `.masks` may be **gappy**. This is expected and correct — the data
stays in place (no compaction), and a gappy mask works fine for what most operations do with it
(masking-to-zero inside contractions, where any pattern works).

## Canonical form vs working form

The **prefix** mask is the *canonical form*; the general (possibly gappy) mask is the *working
form*. Exactly one operation needs the canonical form — SVD truncation keeps top-`k` singular
values, which come out in prefix order — and the T3-SVD *produces* it: `ut3svd` orthogonalizes,
runs fresh SVDs, and applies freshly built prefix masks, ignoring whatever pattern the input had.
So **re-canonicalize with `ut3svd` whenever you want the minimal/prefix form**; between SVDs the
working form flows freely.

Two practical consequences:

- **Extracting a dense real sub-block through a gappy mask** (e.g. `ut3_to_t3`) is a
  gather/`argwhere`, not a slice — or re-canonicalize via `ut3svd` first.
- **`shape_mask` never goes gappy** — the physical shape is invariant under `+`/`×` — which is
  exactly what lets `to_dense` take a *static* prefix-slice on the physical axes. Only the edge
  (rank) masks go gappy.

## Tangent (variation) masks: a different algebra

Everything above is **tensor** algebra. The frame-variations / tangent layer (`UT3Variations`, the
tangent space of a `UT3Frame`) runs a **different algebra**, and the mask behavior is different —
**identical, not combined.** A variation is a tangent vector: a point in the fixed-dimensional
tangent space at a base point `B`, whose rank structure (the mask) is a property of **`B`'s
gauge**, shared by every tangent at `B`:

| op | mask |
|---|---|
| `v + w`, `v - w` (both tangents at `B`) | **identical** — `v` and `w` carry `B`'s mask; the sum is another vector of the same space with the *same* mask. No concat. |
| `α · v`, `-v` | unchanged (scaling cannot change which slots are real) |
| `sum_stack` (sum a batch of tangents) | OR over the summed stack axes — a **no-op** when the batch shares a frame (`K`-stack, one mask); a genuine union only for a varying-rank stack |

This mirrors the ragged layer exactly: tangent addition is a *corewise* add of equal-shaped cores,
staying inside one fixed-dimensional space — the concat/Kronecker story never applies.

Because add/sub/scale **do not change the mask**, they create no new static structure: no fresh
jit cache key, no recompile, and the result still pairs with its frame.

**The one precondition uniform adds.** In ragged, adding mismatched variations fails loudly (a
shape error). In uniform the supercores are *padded to a common shape*, so two variations with
**different masks but equal padded shape** would silently add to a **wrong** result — the padding
hides the mismatch. So uniform variation add/sub carries an explicit **same-mask structural
precondition** (`shape` and the masks must match — a cheap host-side `array_equal`, valid even
under jit). It is the variation-level analog of the same-frame guard on tangents.

## Masks are host numpy

The masks on any uniform object are **host numpy arrays** — always, even when the supercores are
jax. This is part of the API: backend users supply numpy masks (the signature tag is
`HOST bool, static` — see [`reading_signatures.md`](reading_signatures.md)), and under jit the
masks fold into the compiled program as constants at zero per-call cost. The full jit story:
[`uniform_backend_jit_recipe.md`](uniform_backend_jit_recipe.md);
why this is load-bearing: [`contributor/uniform_pytree_composition.md`](contributor/uniform_pytree_composition.md).

See also `docs/uniform_supercore_layout.md` (why the core index leads) and
`docs/uniform_ranks_and_varieties.md` (what a stacked uniform T3 represents).
