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

> **Not to be confused with `weight=` in the fitting layer.** The optimizers' `weight=` is the
> **residual weight `ω`** in the objective `½‖ω⊙r‖²` — a per-`(mode, order)` scaling of the
> *measurements*, described in [`fitting_and_optimization.md`](fitting_and_optimization.md) §4.6. It is
> unrelated to this page: nothing here touches the objective, and nothing there touches a tensor's
> edges. Throughout the library, the plural `weights` and the `*Weights` classes always mean **edge**
> weights.

## `T3Weights` — a weighted tensor

A `TuckerTensorTrain` has two internal edge families: **Tucker-rank** edges `nᵢ` (`d` of them) and
**TT-bond** edges `rᵢ` (`d+1`, ends trivial). `T3Weights` holds one diagonal vector per edge —
`(tucker_weights[d], tt_weights[d+1])`, which is *exactly* the shape `t3svd` returns, so the singular
values are the canonical weight:

```python
>>> import numpy as np
>>> import t3toolbox.tucker_tensor_train as t3
>>> np.random.seed(0)
>>> x = t3.TuckerTensorTrain.randn((6, 7, 8), (2, 2, 2), (1, 2, 2, 1))   # a minimal-rank point
>>> W  = t3.T3Weights.from_t3svd(x)        # the singular values of x (unmodified)
>>> Wp = W.reciprocal()                    # inverse-σ (Grasedyck–Kramer) weighting; also .sqrt()
>>> xw = t3.t3_absorb_weights(x, Wp)          # a plain TuckerTensorTrain: the fully-weighted network
>>> n  = t3.t3_weighted_norm(x, Wp)           # = xw.norm(); the inserted diagonal is squared by the norm
>>> print(bool(np.allclose(n, xw.norm())))
True
>>> print(xw.shape == x.shape, xw.ranks == x.ranks)   # absorb is shape- and rank-preserving
True True
>>> y  = t3.TuckerTensorTrain.randn((6, 7, 8), (2, 3, 3), (1, 2, 3, 1))
>>> W2 = t3.T3Weights.from_t3svd(y)
>>> g  = t3.t3_weighted_inner(x, W, y, W2)    # <absorb(x,W), absorb(y,W2)>  (same physical shape)
>>> print(bool(np.allclose(g, t3.t3_absorb_weights(x, W).inner(t3.t3_absorb_weights(y, W2)))))
True

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
>>> import t3toolbox.frame_variations_format as bvf
>>> import t3toolbox.manifold as t3m
>>> frame, _, sigma = bvf.t3svd_orthogonal_representations(x)   # the frame at x in the T3-SVD gauge + x's σ's
>>> tangent = t3m.MANIFOLD.randn(frame)                 # two tangent vectors there
>>> other   = t3m.MANIFOLD.randn(frame)
>>> W = bvf.T3FrameWeights.from_t3weights(sigma)        # a metric: x's σ's, per coordinate of THIS frame
>>> n  = tangent.weighted_norm(W)            # absorb W into the variations V,H; take the coordinate norm
>>> g  = tangent.weighted_inner(other, W)    # one metric W; the same-frame precondition is checked
>>> vw = tangent.absorb_weights(W)           # the weighted tangent itself (vw.corewise_norm() == n)
>>> print(bool(np.allclose(vw.corewise_norm(), n)))
True
>>> print(bool(np.allclose(g, vw.corewise_inner(other.absorb_weights(W)))))
True

```

**The gauge matters.** A singular-value metric is per *coordinate*, and the coordinates are the frame's
basis. `t3svd_orthogonal_representations` builds the frame with `already_left_orthogonal=True`, so its
Tucker basis *is* the singular basis the σ's belong to (and it costs one SVD, not two). The default
`t3_orthogonal_representations(x)` re-orthogonalizes the already-orthonormal Tucker factors, whose spectrum
is degenerate, and comes back rotated by an arbitrary orthogonal matrix — σ-weights applied on that frame
weight the wrong directions, silently (this was the documented recipe before 2026.2.0).

The weights are absorbed into the **variation** cores (`down`→`V`, `up`/`left`/`right`→`H`); the frame
stays orthonormal and untouched, so this is `O(ranks)` and does not disturb the tangent space.
`tangent.absorb_weights(W)` returns the weighted tangent at the same frame — but note it is **not gauged**
(scaling the coordinates breaks the gauge, though not the frame's orthogonality), so use `corewise` ops on
it, or re-gauge with `MANIFOLD.project_oblique` for Hilbert–Schmidt semantics. (The standalone
`frame_variations_format.fv_absorb_weights(variations, W)` returns the weighted `T3Variations`.) All-ones
weights recover `tangent.corewise_norm()`. Like `T3Weights`, `T3FrameWeights` has `reciprocal` / `sqrt` /
`concatenate` / `kronecker` / `reverse` / `stack` / `unstack` / `is_consistent_with`.

**From a base-point weight.** `T3FrameWeights.from_t3weights(W)` builds a tangent metric from a
`T3Weights` — `up = down =` the Tucker weights, and `left` / `right` are the TT-bond weights sliced by the
`Hᵢ` bond convention (`left = tt[:-1]`, `right = tt[1:]`). So the **Grasedyck–Kramer metric from a point's
singular values** is one line:

```python
>>> import t3toolbox.frame_variations_format as bvf
>>> gk = bvf.T3FrameWeights.from_t3weights(t3.T3Weights.from_t3svd(x)).reciprocal()
>>> print(gk.up_ranks, gk.down_ranks)              # up = down = the Tucker weights
(2, 2, 2) (2, 2, 2)
>>> print(gk.left_ranks, gk.right_ranks)           # left = tt[:-1], right = tt[1:]
(1, 2, 2) (2, 2, 1)

```

It pairs with a minimal-rank tangent at `x` (where the complement rank `nD` equals the Tucker rank, as for
`t3svd` output — see [`frame_variations.md`](frame_variations.md)).

**Shared Tucker factors.** Weighting composes with the shared-factor format
([`sharing.md`](sharing.md)) with one condition: absorbing keeps a tied T3 tied **iff the Tucker
weight vectors are equal within each sharing group** (Tucker weights scale the factors; TT-bond
weights are absorbed into the TT cores and never touch them). `W.has_shared_tucker_weights(sharing)`
checks that compatibility (non-enforcing — absorbing group-unequal weights is legal, it just unties
the result). `from_t3svd(x, sharing=…)` produces group-equal weights by construction, and the whole
weight algebra (`reciprocal`/`sqrt`/`concatenate`/`kronecker`) preserves group-equality.

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
>>> import t3toolbox.uniform_tucker_tensor_train as ut3
>>> ux = ut3.UniformTuckerTensorTrain.from_t3(x)
>>> W  = ut3.UT3Weights.from_ut3svd(ux)          # or .from_t3weights(ragged_W, n=ux.n, r=ux.r)
>>> gk = W.reciprocal()                          # inverse-σ; also .sqrt() / .concatenate() / .kronecker()
>>> n  = ut3.ut3_weighted_norm(ux, gk)               # + absorb_weights / weighted_inner
>>> print(W.is_consistent_with(ux))              # a weight declares the SAME edge masks as its train
True
>>> print(bool(np.allclose(n, t3.t3_weighted_norm(x, t3.T3Weights.from_t3svd(x).reciprocal()))))
True

```

Three differences from ragged are worth knowing, and none of them are ports-in-progress:

- **The masks must match.** Ragged catches a rank mismatch as a shape error; uniform pads both sides to a
  common width, so a mismatched weight would *silently* zero a real slot. Hence `is_consistent_with`, and
  every `(train, weights)` op enforces it — the one precondition uniform adds. For a train padded *above*
  its minimal ranks (the rank-continuation warm start), build the weights at the train's own widths —
  `UT3Weights.from_ut3svd(ux, n=ux.n, r=ux.r)`, mirroring `from_t3weights` — since the default pads
  tightly to the t3svd result and would be rejected against `ux` and its frame.
- **`reciprocal` guards the padding.** The padding is a canonical zero, and `1/0 = inf` would poison every
  masked reduction downstream (`0 × inf = nan`). Real-slot zeros are deliberately *not* guarded: a zero
  singular value is real data, and clamping it would hide a rank-deficient point.
- **`concatenate` / `kronecker` produce gappy masks.** Ranks add / multiply as usual, but the real slots
  stop being a prefix ([`uniform_masks_vs_ranks.md`](uniform_masks_vs_ranks.md)). Expected and correct; the
  T3-SVD re-canonicalizes. The ops themselves need no mask cleverness — combining the weights and combining
  the masks *the same way* is exactly right, because both concatenation and the Kronecker product commute
  with elementwise multiply.

`UT3FrameWeights` mirrors `T3FrameWeights` the same way (the arithmetic; not `reverse`/`stack`/`unstack`),
and the frame-like batching is where it earns its keep: the metric carries the frame stack `C`, so the singular-value metric of a `C`-stacked point pairs
directly with a `K`-stack of tangents there, broadcasting over `K` for free.

```python
>>> import t3toolbox.uniform_frame_variations_format as ubvf
>>> import t3toolbox.uniform_manifold as ut3m
>>> xs  = t3.TuckerTensorTrain.randn((6, 7, 8), (2, 2, 2), (1, 2, 2, 1), stack_shape=(2,))
>>> uxs = ut3.UniformTuckerTensorTrain.from_t3(xs)                   # C = 2 base points
>>> frame, _, sigma = ubvf.ut3svd_orthogonal_representations(uxs)   # the T3-SVD gauge, as in ragged
>>> gk = ubvf.UT3FrameWeights.from_ut3weights(sigma).reciprocal()
>>> tangent = ut3m.UNIFORM_MANIFOLD.randn(frame, stack_shape=(3,))   # K = 3 tangents at each base point
>>> n  = tangent.weighted_norm(gk)     # + weighted_inner / absorb_weights; gk.stack_shape == frame.stack_shape
>>> print(gk.stack_shape == frame.stack_shape)   # the metric is FRAME-like: it carries C, not K + C
True
>>> print(n.shape)                               # ...and broadcasts over K for free: K + C
(3, 2)

```

## Scope

Shipped: the two ragged weight classes, `absorb`, `weighted_norm` / `weighted_inner`, `concatenate` /
`kronecker`, `from_t3svd` / `from_t3weights`, `reverse` / `stack` / `unstack`; and the uniform mirror —
`UT3Weights` and `UT3FrameWeights` with the same arithmetic (`absorb`, `weighted_norm` / `weighted_inner`,
`reciprocal` / `sqrt`, `concatenate` / `kronecker`), plus `from_ut3svd` / `from_ut3weights` and the
ragged↔uniform conversions. The uniform classes have no `reverse` / `stack` / `unstack` yet (deliberately
deferred; go through the ragged classes for those). Not yet packaged (but reachable from these primitives): weighted `+` / `−` / scale / `⊙` as
operations, and the **Grasedyck–Kramer singular-value regularizer** — a `SingularValueRegularizer` that
builds `W` from the frame's singular values and applies it through the `T3FrameWeights` metric (see the
regularization notes). Both layers now have everything that regularizer needs.
