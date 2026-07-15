# Weighting (edge weights)

It is often useful to imagine **diagonal matrices inserted on the internal edges** of a Tucker tensor
train. Put the inverse singular values there and the norm of the weighted network penalises the
*less-informed* rank directions — the idea behind Grasedyck–Kramer preconditioning. This layer gives you
a lightweight data format for those diagonals, the ability to **absorb** them into cores, and the
weighted `norm` / `inner` (plus `concatenate` / `kronecker`). Two objects, one per geometry — and they
play two genuinely different roles:

| object | pairs with | role | absorbed into |
|---|---|---|---|
| `T3Weights` | `TuckerTensorTrain` | weights of a **tensor** (all internal edges) | the cores |
| `T3FrameWeights` | a tangent (`T3Tangent`) | a **metric** on the tangent coordinates | the variation cores |

## `T3Weights` — a weighted tensor

A `TuckerTensorTrain` has two internal edge families: **Tucker-rank** edges `nᵢ` (`d` of them) and
**TT-bond** edges `rᵢ` (`d+1`, ends trivial). `T3Weights` holds one diagonal vector per edge —
`(tucker_weights[d], tt_weights[d+1])`, which is *exactly* the shape `t3svd` returns, so the singular
values are the canonical weight:

```python
import t3toolbox.tucker_tensor_train as t3

W  = t3.T3Weights.from_t3svd(x)        # the singular values of x (unmodified)
Wp = W.reciprocal()                    # inverse-σ (Grasedyck–Kramer) weighting; also .sqrt()
xw = t3.t3_absorb_weights(x, Wp)          # a plain TuckerTensorTrain: the fully-weighted network
n  = t3.t3_weighted_norm(x, Wp)           # = xw.norm(); the inserted diagonal is squared by the norm
g  = t3.t3_weighted_inner(x, W, y, W2)    # <absorb(x,W), absorb(y,W2)>  (same physical shape)
```

`absorb` is shape-preserving (the diagonals fold into the cores). Convention: Tucker weights go into the
Tucker cores; TT-bond weights are absorbed leftward (the leftmost bond rightward into the first core).

**The `+` / `⊙` duality.** Two ways to combine weighted tensors map onto two elementary vector operations
on the per-edge diagonals:

- `A + B` → cores block-diagonal, ranks **add** → weights **`concatenate`** (`W_A.concatenate(W_B)`).
- `A ⊙ B` (Hadamard) → cores Kronecker on internal legs, ranks **multiply** → weights **`kronecker`**
  (`W_A.kronecker(W_B)`).

So the weighted objects form a closed algebra (the weighted `+`/`⊙` operations themselves are not yet
packaged, but the primitives are here).

## `T3FrameWeights` — a metric on a tangent

For a tangent the natural object is a **Riemannian metric on the coordinates**: a reweighting of the `d`
variation directions (penalise the poorly-informed ones with `1/σ`). There are `d` weights per family —
one per variation core — in four families matching the tangent's rank types:

| family | weights | length |
|---|---|---|
| `up` | `H`'s Tucker (`nU`) leg | `d` |
| `down` | `V`'s complement (`nD`) leg | `d` |
| `left` | `H`'s left bond (`rL`) | `d` |
| `right` | `H`'s right bond (`rR`) | `d` |

```python
n  = tangent.weighted_norm(W)            # absorb W into the variations V,H; take the coordinate norm
g  = tangent.weighted_inner(other, W)    # one metric W; the same-frame precondition is checked
vw = tangent.absorb_weights(W)           # the weighted tangent itself (vw.corewise_norm() == n)
```

The weights are absorbed into the **variation** cores (`down`→`V`, `up`/`left`/`right`→`H`); the frame
stays orthonormal and untouched, so this is `O(ranks)` and does not disturb the tangent space.
`tangent.absorb_weights(W)` returns the weighted tangent at the same frame — but note it is **not gauged**
(scaling the coordinates breaks the gauge, though not the frame's orthogonality), so use `corewise` ops on
it, or re-gauge with `MANIFOLD.project_oblique` for Hilbert–Schmidt semantics. (The standalone
`frame_variations_format.absorb_weights(variations, W)` returns the weighted `T3Variations`.) All-ones
weights recover `tangent.corewise_norm()`. Like `T3Weights`, `T3FrameWeights` has `reciprocal` / `sqrt` /
`concatenate` / `kronecker` / `reverse` / `stack` / `unstack` / `is_consistent_with`.

**From a base-point weight.** `T3FrameWeights.from_t3weights(W)` builds a tangent metric from a
`T3Weights` — `up = down =` the Tucker weights, and `left` / `right` are the TT-bond weights sliced by the
`Hᵢ` bond convention (`left = tt[:-1]`, `right = tt[1:]`). So the **Grasedyck–Kramer metric from a point's
singular values** is one line:

```python
import t3toolbox.frame_variations_format as bvf
gk = bvf.T3FrameWeights.from_t3weights(t3.T3Weights.from_t3svd(x)).reciprocal()
```

It pairs with a minimal-rank tangent at `x` (where the complement rank `nD` equals the Tucker rank, as for
`t3svd` output — see [`frame_variations.md`](frame_variations.md)).

## Batching

Every weight vector is `stack_shape + (rank,)`, one object holds a stack of weights, and **both classes
carry the base-point stack `C`** — never the tangent stack `K`. Every operation is a single broadcast
prefix, so stacking rides along for free.

- **`T3Weights`** batches like the `TuckerTensorTrain` it weights: stack `C`.
- **`T3FrameWeights`** batches like **`T3Frame`**: stack `C` — one metric per base point. This is worth
  pausing on, because the weight is *absorbed into* the variations (which carry `K + C`) while it *batches
  with* the frame. The two are different questions: a `K`-batch of tangents at one frame shares the **one**
  metric, and the leading `'...'` broadcasts `C` over `K + C` for free (`C` is innermost). So the
  Grasedyck–Kramer metric of a `C`-stacked point is `C`-stacked, and pairs directly with any `K`-stack of
  tangents there.

Pairing a `T3FrameWeights` with variations therefore follows the **same trailing-stack rule** as pairing a
`T3Frame` with them: the weight's stack must be the trailing (inner) part of the variation stack. Like the
variations, that check is blind to the frame — so at the tangent level, where the frame *is* present, the
stricter `weights.stack_shape == frame.stack_shape` is enforced (a `K + C`-stacked weight would otherwise
silently weight one frame's `K` tangents with `K` different metrics). See
[`batching_and_stacking.md`](batching_and_stacking.md).

## On the uniform layer

`UT3Weights` is the uniform mirror of `T3Weights` — the same weights, padded for `jit`/GPU. It is *the
ragged weight, padded*: `to_ragged(op_uniform(to_uniform(W))) == op_ragged(W)`, as for every uniform op
([`uniform_equivalence_contract.md`](uniform_equivalence_contract.md)). It holds two weight supercores
plus **the same edge masks as the train it weights** (a weight's edges *are* the tensor's edges), and
carries no `shape` — weights live only on internal edges.

```python
import t3toolbox.uniform_tucker_tensor_train as ut3

ux = ut3.UniformTuckerTensorTrain.from_t3(x)
W  = ut3.UT3Weights.from_ut3svd(ux)          # or .from_t3weights(ragged_W, n=ux.n, r=ux.r)
gk = W.reciprocal()                          # inverse-σ; also .sqrt() / .concatenate() / .kronecker()
n  = ut3.ut3_weighted_norm(ux, gk)               # + absorb_weights / weighted_inner
```

Three differences from ragged are worth knowing, and none of them are ports-in-progress:

- **The masks must match.** Ragged catches a rank mismatch as a shape error; uniform pads both sides to a
  common width, so a mismatched weight would *silently* zero a real slot. Hence `is_consistent_with`, and
  every `(train, weights)` op enforces it — the one precondition uniform adds.
- **`reciprocal` guards the padding.** The padding is a canonical zero, and `1/0 = inf` would poison every
  masked reduction downstream (`0 × inf = nan`). Real-slot zeros are deliberately *not* guarded: a zero
  singular value is real data, and clamping it would hide a rank-deficient point.
- **`concatenate` / `kronecker` produce gappy masks.** Ranks add / multiply as usual, but the real slots
  stop being a prefix ([`uniform_masks_vs_ranks.md`](uniform_masks_vs_ranks.md)). Expected and correct; the
  T3-SVD re-canonicalizes. The ops themselves need no mask cleverness — combining the weights and combining
  the masks *the same way* is exactly right, because both concatenation and the Kronecker product commute
  with elementwise multiply.

`UT3FrameWeights` mirrors `T3FrameWeights` the same way, and the frame-like batching is where it earns its
keep: the metric carries the frame stack `C`, so the singular-value metric of a `C`-stacked point pairs
directly with a `K`-stack of tangents there, broadcasting over `K` for free.

```python
import t3toolbox.uniform_frame_variations_format as ubvf

frame, _ = ubvf.ut3_orthogonal_representations(ux)
gk = ubvf.UT3FrameWeights.from_ut3weights(ut3.UT3Weights.from_ut3svd(ux)).reciprocal()
n  = tangent.weighted_norm(gk)     # + weighted_inner / absorb_weights; gk.stack_shape == frame.stack_shape
```

## Scope

Shipped: the two ragged weight classes, `absorb`, `weighted_norm` / `weighted_inner`, `concatenate` /
`kronecker`, `from_t3svd` / `from_t3weights`; and the **full uniform mirror** — `UT3Weights` and
`UT3FrameWeights` with the same operations, plus `from_ut3svd` / `from_ut3weights` and the ragged↔uniform
conversions. Not yet packaged (but reachable from these primitives): weighted `+` / `−` / scale / `⊙` as
operations, and the **Grasedyck–Kramer singular-value regularizer** — a `SingularValueRegularizer` that
builds `W` from the frame's singular values and applies it through the `T3FrameWeights` metric (see the
regularization notes). Both layers now have everything that regularizer needs.
