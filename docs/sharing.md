# Shared Tucker factors (SF-T3)

Sometimes several modes of a tensor should use the **same basis**: the axes are the same physical
space (a derivative tensor's input modes, a multi-time snapshot's spatial modes), or you simply want
one dictionary serving several roles. Constraining the Tucker factors to be **equal within
user-specified groups of modes** removes parameters, couples the information from all the group's
modes into one basis, and — for optimization — restricts the search to a smooth submanifold. This is
the *shared-factor* format: SF-Tucker (Peshekhonov, Arzhantsev & Rakhuba, AISTATS 2024) brought it to
Tucker models, SF-ETT (Molozhavenko & Rakhuba, Comput. Appl. Math. 45:221, 2026) to extended tensor
trains with one trailing shared block; T3Toolbox implements it for an **arbitrary partition** of the
modes into sharing groups. The partition is always yours to choose — automatic selection is out of
scope.

Everything is driven by one spec: **`sharing` — a length-`d` tuple of hashable labels, one per
mode**; modes with equal labels share one factor. `(0, 0, 1)` ties the first two modes of a 3-mode
tensor; `('a', 'b', 'a', 'b')` ties non-adjacent pairs; group modes must have equal mode sizes.
`sharing=None` (or all-distinct labels) is the plain unshared behavior, exactly.

## The format: an ordinary T3, with tied factors

A shared T3 is **not a new class** — it is a `TuckerTensorTrain` whose group factors are equal (the
same array at every group mode). The storage is deliberately redundant: the SF-ETT paper's memory saving is
forgone so the whole existing API keeps working on shared points unchanged; what sharing buys here is
**fewer optimization parameters and one jointly-informed basis**, not a smaller object.

```python
>>> import numpy as np
>>> import t3toolbox as t3t
>>> np.random.seed(0)
>>> x = t3t.TuckerTensorTrain.randn((6, 6, 4), (4, 4, 3), (1, 2, 2, 1))

>>> xs = x.share((0, 0, 1))                         # enter the format: the quasi-optimal projection
>>> print(xs.tucker_ranks, xs.tt_ranks)             # the group basis must serve BOTH modes
(6, 6, 2) (1, 2, 2, 1)
>>> print(bool(np.all(xs.has_shared_tucker_factors((0, 0, 1)))))    # the checker (per stack element)
True
>>> print(xs.data[0][0] is xs.data[0][1])           # one factor ARRAY per group
True

```

`share(sharing, max_tucker_ranks=…, max_tt_ranks=…, rtol=…)` is the way **in** from an arbitrary
tensor: per group it takes the dominant left singular subspace of the concatenated matricizations
`[X_(i₁) | … | X_(iₖ)]` — the quasi-optimal shared basis — projects every group mode onto it, and
rounds with the grouped T3-SVD below. With no caps it is the lossless common-span rewrite.

## Grouped truncation: `t3svd(sharing=…)`

Truncating a shared T3 must pick **one** basis per group, so the grouped T3-SVD is a different
algorithm from the unshared interlaced sweep (it is Molozhavenko & Rakhuba's Algorithm 1, our
arbitrary-partition form): first the TT bonds are rounded with the Tucker steps skipped (skipping is
what keeps tied factors tied), then every mode's center core of that *same* TT-rounded tensor is
collected at once, then all Tucker truncations happen simultaneously — each group truncates one SVD
of its concatenated centers, applying the rotation to the shared factor once. Two consequences worth
knowing:

- **Dispatch rule.** `sharing=None` and all-singleton partitions run the *literal* unshared sweep
  (bit-identical). Any partition with a real group runs the two-phase algorithm for **all** modes —
  so under truncation, even the singleton modes are handled slightly differently than the unshared
  sweep would (all Tucker steps see the same TT-rounded tensor). Lossless calls agree exactly.
- **The reported Tucker singular values at group modes are the group spectrum `s_g`** — the next
  section says precisely what that is.

`rank_adjustment_sweep(direction, sharing=…)` is the matching lossless rank-minimization step: the
per-mode reduction can clip a shared rank at one mode but not another (untying the group), while the
grouped reduction uses the group's own structural ceiling (below).

## What the group spectrum is

The unshared `t3svd` reports, per edge, the singular values of a matrix unfolding across that edge.
A shared group has no single tree edge — the same factor sits on `k` edges — so its spectrum is a
genuinely different (but exactly characterized) object. `s_g` has four equivalent faces:

1. **The concatenated-matricization spectrum**: the singular values of
   `[T_(i₁) | T_(i₂) | … | T_(iₖ)]` — a property of the represented tensor and the partition alone
   (no gauge, no representation in it). This is the object SF-Tucker and SF-ETT compute.
2. **The summed-Gram spectrum**: `s_{g,j}² = λ_j(Σ_{i∈g} Γ_i)` with `Γ_i = T_(i) T_(i)ᵀ` the mode
   Grams.
3. **The Jacobian spectrum of the shared factor**: for a gauged tied motion of the group's factor,
   the `k` first-order tensor changes are mutually orthogonal, so `s_g` is exactly the sensitivity
   spectrum of the tensor with respect to the shared parameter — the same role a per-mode Tucker
   spectrum plays for an unshared factor. Hence `κ_g = s_{g,1}/s_{g,n_g}` is *the* conditioning of
   the tied-factor subproblem, and `s_{g,min} → 0` exactly when the point approaches the stratum of
   lower shared rank.
4. **An honest single-cut spectrum — of a lifted tensor**: stack the `k` mode-permutations of `T`
   along a new copy axis; the cut separating the shared leg from everything else has unfolding
   exactly the concatenation of face 1. The unshared "cut spectrum" intuition survives, one level
   up.

Three facts that follow, worth keeping in mind when reading `ss_tucker`:

- **Scale**: every mode carries the full norm, so `Σ_j s_{g,j}² = k·‖T‖²` — a group of `k` modes is
  `√k`-inflated relative to a single-mode spectrum. The factor cancels in every condition-number
  *ratio* (group and singleton edges compete fairly in rank continuation), but the unshared per-mode
  invariant `‖ss_tucker[i]‖ = ‖T‖` becomes the group version at group modes.
- **Elementwise domination**: `s_{g,j} ≥ σ_{i,j}` for every mode `i` in the group — the shared edge
  sees at least as much signal per level as any single mode, and its conditioning is never worse
  than the group's worst per-mode conditioning (often far better: under tying, a direction is
  well-determined if *some* mode informs it).
- **One tensor, one family**: under truncation, all the reported Tucker spectra (group and
  singleton) describe the *same* phase-1 TT-rounded tensor — a mutually consistent family, unlike
  the unshared sweep's per-edge moving target. For a fully symmetric tensor with all modes grouped,
  `s_g = √k·σ` exactly and `κ_g` equals the per-mode condition number.

The matrix case is a good picture: for a square `T` with both modes tied, `s_g² = eig(TTᵀ + TᵀT)` —
the joint spectrum of the one basis that must serve rows *and* columns.

## Optimizing over shared factors

The headline: wrap a geometry, and every optimizer works unchanged.

```python
>>> sharing = (0, 0, 1)
>>> geom  = t3t.shared_manifold(sharing)      # = t3t.shared(t3t.MANIFOLD, sharing)
>>> geomc = t3t.shared_corewise(sharing)      # = t3t.shared(t3t.COREWISE, sharing)

>>> A  = t3t.TuckerTensorTrain.randn((5, 5, 4), (2, 2, 2), (1, 2, 2, 1)).share(sharing)
>>> ww = [np.random.randn(200, N) for N in A.shape]
>>> ww = [w / np.linalg.norm(w, axis=1, keepdims=True) for w in ww]
>>> b  = A.apply(ww)
>>> x0 = t3t.TuckerTensorTrain.zeros(A.shape, A.tucker_ranks, A.tt_ranks)

>>> x_fit, stats = t3t.newton_cg(geom, 'apply', ww, b, x0, max_newton=30)
>>> print(bool(np.all(x_fit.has_shared_tucker_factors(sharing))))         # every iterate stays tied
True
>>> err = np.linalg.norm(x_fit.to_dense() - A.to_dense()) / np.linalg.norm(A.to_dense())
>>> print(bool(err < 1e-8))
True

```

Every iterate stays exactly tied (one factor array per group). At a shared point the tied tangent
directions are: **TT variations unrestricted**, and per group **one common gauged ambient direction
`U̇`** driving all the group's Tucker variations (`V_i = S_iᵀ U̇` in the frame's coordinates — the
`S_i` factors are part of the orthogonal frame's own factorization). The wrapper's projection lands
on this subspace, and its retraction goes through a *tied* doubled-rank embedding truncated by the
grouped T3-SVD, so tying is preserved by construction rather than repaired after the fact.

**One principle, two formulas.** Each geometry ties by orthogonal projection onto *its* tied
subspace in *its* metric on *its* coordinates — and the formulas genuinely differ:

- **`shared(MANIFOLD, …)`** works in gauged coordinates, which carry the frame's `S_i` factors; the
  tied projection is a *tilted* least-squares solve against the group's stacked-`S` SVD (solved at
  sensitivity `κ_g`, with a clipped pseudoinverse — well-defined even at rank-deficient points).
- **`shared(COREWISE, …)`** works on raw core perturbations, where the tied subspace is simply
  "the `δU_i` are equal" and the projection is the per-group **arithmetic mean**.

Swapping the formulas would give wrong projections in both directions; the wrapper owns this so you
never have to. Inner products are untouched (the tied subspace is linear — the restricted metric
*is* the metric), and `IdentityRegularizer` and the residual weights compose unchanged.

## Rank machinery: minimal ranks, dimension, continuation

**Minimal ranks.** A shared basis column is useless only if it is useless for *every* mode of the
group, so the per-mode Tucker ceiling `n_i ≤ min(N_i, r_{i-1}·r_i)` is replaced by the **group
ceiling** — per-mode ceilings *add* across the group:

    n_g ≤ min(N_g, Σ_{i∈g} min(N_g, r_{i-1}·r_i))

A shared rank may legitimately exceed an individual mode's local ceiling; the unshared reduction
would clip it there and untie the group. `TuckerTensorTrain.get_minimal_ranks(…, sharing=…)`:

```python
>>> t3t.TuckerTensorTrain.get_minimal_ranks((6, 6, 4), (4, 4, 3), (1, 2, 2, 1))
((2, 4, 2), (1, 2, 2, 1))
>>> t3t.TuckerTensorTrain.get_minimal_ranks((6, 6, 4), (4, 4, 3), (1, 2, 2, 1), sharing=(0, 0, 1))
((4, 4, 2), (1, 2, 2, 1))

```

The per-mode reduction clips mode 0 to 2 and so **unties** the group; the group ceiling keeps the
shared rank at 4.

**Dimension.** `manifold.manifold_dim(s, sharing=…)` is the shared submanifold's dimension: the
reduction to minimal ranks is the *shared* one, the TT-core term is unchanged (TT cores are never
tied), and each group contributes **one** Stiefel term `n_g(N_g − n_g)` instead of one per mode.
(Both papers prove the single-trailing-block case; the arbitrary-partition formula is our extension,
validated against dense tangent-space ranks.)

**Continuation.** In rank continuation ([`rank_continuation.md`](rank_continuation.md)) a group's
Tucker edges are **one edge**: one `κ_g` from the group spectrum competes in the pool, one growth
decision applies group-wide (a group counts as one `max_grow` candidate), and useless-rank removal
is the shared reduction. `continuation_ranks(sharing=…)` pairs with `resize(…, sharing=…)`, which
pads the group factor once (the same array at every mode) for the zero-padded warm start. A freshly
padded shared point sits *on* the lower-rank stratum — the new levels of `s_g` are exactly zero and
the tied Tucker channel is momentarily gated — but the escape runs through the unrestricted TT
variations within the first optimization steps, which is why full shared rank is a **diagnostic,
never a precondition** anywhere in the library.

## Batching

The `sharing` spec is static structure (like masks): a stacked shared T3 is a `C`-stack of tied
points with **one partition** for the whole stack, and the checkers give per-stack-element verdicts.
The per-frame companion the manifold wrapper derives (the group centers and stacked-`S` SVD) carries
the frame stack `C` and broadcasts over a tangent stack `K` for free — the standard frame-inner
`W + K + C` story of [`batching_and_stacking.md`](batching_and_stacking.md).

## On the uniform layer

The optimization surface mirrors, under the usual contract
([`uniform_equivalence_contract.md`](uniform_equivalence_contract.md)): `t3svd(sharing=…)` and
`rank_adjustment_sweep(…, sharing=…)` on `UniformTuckerTensorTrain` (the backend function is
`ut3svd`; mask-only truncation — one group rank mask at every group mode; no `rtol`/`atol`, as
always on uniform),
`has_shared_tucker_factors` on masked content, and `shared(UNIFORM_MANIFOLD, sharing)` /
`shared(UNIFORM_COREWISE, sharing)` running the packed, compile-once fitting path (the sharing
partition is a value-hashed *field* of the geometry beside the masks, so a rebuilt geometry is the same
jit cache key; the per-frame companion flows as traced data, so a shared fit compiles once like an
unshared one). Ragged vs uniform is inferred from `x0`,
as everywhere:

```python
>>> ux0 = t3t.UniformTuckerTensorTrain.from_t3(x0)
>>> ux_fit, _ = t3t.newton_cg(t3t.shared(t3t.UNIFORM_MANIFOLD, sharing), 'apply', ww, b, ux0,
...                           max_newton=30)
>>> print(bool(np.all(ux_fit.has_shared_tucker_factors(sharing))))
True
>>> uerr = np.linalg.norm(ux_fit.to_t3().to_dense() - A.to_dense()) / np.linalg.norm(A.to_dense())
>>> print(bool(uerr < 1e-8))
True

```

One uniform-specific note: a shared uniform start is reduced to *shared*-minimal ranks
transparently (`uniform_minimal(x0, sharing=…)`) — the per-mode reduction would silently untie it.

## Scope

- **Compute, not memory.** Storage stays redundant (one array per group, but `d` slots); the wins
  are parameter count, joint information, and the shared-manifold geometry.
- **Sharing is not symmetry.** Tying factors does *not* make the tensor symmetric — the TT cores
  are free, so a shared T3 can represent any tensor whose group modes admit a common basis of the
  given rank. (Conversely, symmetric tensors are the flagship *use case*: their mode matricizations
  are equal, so one shared basis is exactly right — see the symmetric fitting example in
  `examples/`.) Do not expect `sharing=(0,0,0)` to enforce `T[i,j,k] = T[j,i,k]`.
- **Weights compose with sharing already** ([`weighting.md`](weighting.md)): absorbing edge weights
  keeps a tied T3 tied **iff the Tucker weight vectors are equal within each group** (Tucker weights
  scale the factors; TT-bond weights never touch them) — checkable with
  `T3Weights.has_shared_tucker_weights(sharing)` (+ the `UT3Weights` twin), non-enforcing: absorbing
  group-unequal weights is legitimate, it just unties the result (repair with
  `t3_tie_tucker_factors` or re-enter with `share`). `from_t3svd(x, sharing=…)` builds group-equal
  weights by construction (the group spectrum at every group mode — `√k`-inflated relative to
  per-mode spectra, a per-group constant that cancels within each group's reciprocal ratios), and
  `reciprocal`/`sqrt`/`concatenate`/`kronecker` all preserve group-equality. What remains deferred is
  the weighted tied *metric* — a Grasedyck–Kramer preconditioner for the shared geometry (the
  `T3FrameWeights` route is not even well-formed under sharing: the `down` family lives on the `nD`
  legs, which can differ in *length* within a group) — future work alongside the
  `SingularValueRegularizer`.
- **The partition is user-provided.** Automatic selection of what to share is permanently out of
  scope.
- **Safe mode** checks tied factors at the shared entry points (`t3svd(sharing=…)`,
  `rank_adjustment_sweep(…, sharing=…)`, the wrapper's `frame`/`retract`/`transport`,
  `resize(…, sharing=…)`) and tied tangent *coordinates* at the
  manifold retraction; see [`numerical_contracts.md`](numerical_contracts.md). Full shared rank is
  never enforced (the continuation-restart argument above).

## References

The shared-factor idea and the two algorithms this page generalizes:

- Peshekhonov, I., Arzhantsev, A., and Rakhuba, M. "Training a Tucker Model With Shared Factors: a
  Riemannian Optimization Approach." *Proceedings of the 27th International Conference on Artificial
  Intelligence and Statistics (AISTATS)*, PMLR 238, 2024.
  [proceedings.mlr.press/v238/peshekhonov24a.html](https://proceedings.mlr.press/v238/peshekhonov24a.html)
  — **SF-Tucker**: shared factors for the Tucker format, and the Riemannian machinery for optimizing
  over them.
- Molozhavenko, A. and Rakhuba, M. "Optimization on the extended tensor-train manifold with shared
  factors." *Computational and Applied Mathematics* 45(6):221, 2026.
  [doi.org/10.1007/s40314-025-03605-0](https://doi.org/10.1007/s40314-025-03605-0) — **SF-ETT**:
  the same idea for extended tensor trains (= Tucker tensor trains), with **one trailing shared
  block**. Its Algorithm 1 is the two-phase grouped truncation implemented here; its Theorem 5 is the
  manifold-dimension result. The generalization to an **arbitrary partition** of the modes — and
  therefore the group ceiling on minimal ranks, the group spectrum as a per-group object, and the
  grouped rank continuation — is this library's extension.

*Design records — the `S_i` machinery and its float32 measurements, the two-phase decision, the
tied embedding, and the restart analysis — are in*
[`contributor/sharing_internals.md`](contributor/sharing_internals.md).
