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
xw = t3.absorb_weights(x, Wp)          # a plain TuckerTensorTrain: the fully-weighted network
n  = t3.weighted_norm(x, Wp)           # = xw.norm(); the inserted diagonal is squared by the norm
g  = t3.weighted_inner(x, W, y, W2)    # <absorb(x,W), absorb(y,W2)>  (same physical shape)
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

Both weight classes **mirror the batching of their paired object** — `T3Weights` like `TuckerTensorTrain`,
`T3FrameWeights` like `T3Variations`: every vector is `stack_shape + (rank,)`, and one object holds a
stack of weights. Every operation is a single broadcast prefix (a weight always shares exactly its
object's `C` stack), so stacking rides along for free. See
[`batching_and_stacking.md`](batching_and_stacking.md).

## Scope

Shipped: the two weight classes, `absorb`, `weighted_norm` / `weighted_inner`, `concatenate` /
`kronecker`, `from_t3svd`. Not yet packaged (but reachable from these primitives): weighted `+` / `−` /
scale / `⊙` as operations, and the **Grasedyck–Kramer singular-value regularizer** — a
`SingularValueRegularizer` that builds `W` from the frame's singular values and applies it through the
`T3FrameWeights` metric (see the regularization notes).
