# Why T3-SVD returns minimal ranks (and why that took a re-tighten)

A short note for anyone wondering about the ranks that come back from
`TuckerTensorTrain.t3svd` — in particular why they are **structurally minimal**, what "minimal"
means precisely, and why a naive truncating sweep does *not* give minimal ranks even though it
computes the right tensor. If you write your own truncation (or work on the backend / the uniform
layer), this is the subtlety to know about.

## What `t3svd` guarantees

`t3svd` returns a representation with **structurally minimal ranks**: for the returned
`(tucker_ranks, tt_ranks)`, `x2.has_minimal_ranks` is always `True`. Concretely, every core satisfies
the no-redundancy inequalities

```
n_i  <= N_i           (Tucker rank <= mode dimension)
n_i  <= rL_i * rR_i   (Tucker rank <= product of its two TT bonds)
rL_i <= n_i * rR_i    (each TT bond <= product of the other two legs of its core)
rR_i <= n_i * rL_i
```

with `rL_i = r_i`, `rR_i = r_{i+1}`. These are exactly the **matricization ranks** of a generic
tensor with that structure: cutting a single edge of the T3 tensor network splits the physical indices
into two groups, and the minimal dimension of that edge is the rank of the corresponding matricization.
(The T3 network is a *tree* — a TT backbone with a pendant Tucker matrix on each node — so every
single-edge cut is a clean bipartition with no hidden multi-cut degeneracy.) `compute_minimal_ranks`
computes these by propagating the structural bottlenecks; see
`test_compute_minimal_ranks_matches_matricization`.

## Why a naive truncating sweep is *not* minimal

T3-SVD sweeps left to right. At mode `i` it first truncates the **Tucker** rank `n_i` (against the
current right bond `rR_i`), then truncates the **TT bond** `rR_i`. With no truncation — or with a
tolerance (`rtol`/`atol`) — each SVD keeps exactly `min(rows, cols)`, the structural rank, and a bond
only ever shrinks to its structural value `rL_i * n_i`. That can never orphan the Tucker rank computed
just before it (the check `n_i <= rL_i * rR_i^new = rL_i^2 * n_i` holds automatically). **So the
no-truncation and tolerance paths are already minimal.**

A **hard rank cap** breaks this. A `max_tt_ranks` cap can force `rR_i` *below* its structural value
`rL_i * n_i`. Then `n_i`, fixed moments earlier against the larger pre-cap bond, ends up `> rL_i *
rR_i` — a Tucker rank pointing into a subspace the truncated bond can no longer reach. It is
**structurally redundant**: the represented tensor is unaffected, but the reported rank is not minimal.
(The symmetric thing happens to a bond when a downstream `max_tucker_ranks` cap bites.)

Worked example — `shape=(5,6,7)`, truncate the TT bonds to 2 but leave the Tucker ranks uncapped:

```
n_0 is fixed to 3 (= 1 * r_1 with the pre-cap bond r_1 = 3)
then the cap forces r_1 = 2  <  rL_0 * n_0 = 1 * 3
=> n_0 = 3  >  rL_0 * r_1 = 1 * 2 = 2     # orphaned: minimal n_0 is 2
```

## How `t3svd` fixes it: a lossless re-tighten

After the sweep, `t3svd` runs a structural **right-to-left re-tightening pass**
(`_shrink_to_minimal_ranks` in `backend/t3_svd.py`): re-SVD each Tucker edge, then each bond, with no
cap. Each edge unfolding now has only as many rows as the *post-truncation* structure allows, so the
SVD keeps the structural rank and discards no real content. The result:

- **the represented tensor is unchanged** — only redundant directions are dropped (verified to ~1e-15);
- **the ranks become minimal** — `has_minimal_ranks` is `True` for every cap pattern;
- **already-minimal input pays nothing** — the pass is gated behind a minimality check, so the
  no-truncation and `rtol`/`atol` paths are untouched.

This keeps the *best* approximation: the bond truncation still sees the full Tucker rank (more columns
→ a better rank-`k` bond), and the now-redundant Tucker direction is dropped afterward — strictly no
worse than truncating the Tucker rank up front, and identical to the tensor the sweep already produced.

## Choosing: `minimize_ranks=True` (default) vs `False`

The re-tighten is an **extra right-to-left sweep of SVDs**. The default (`True`) pays for it to
guarantee minimal ranks; `False` skips it and returns the raw sweep output — the **same represented
tensor**, with the possibly-redundant ranks left in (`has_minimal_ranks` may be `False`).

Whether that extra sweep is worth it depends on the problem, and the library does not assume one regime.
It is cheap when a truncation compresses aggressively (it then sweeps small, heavily-truncated cores) or
when the result is already minimal (it is gated, so no-truncation and most tolerance truncations skip
it). It is most significant when each truncation compresses the ranks only **slightly** — the cores it
sweeps are still large — and you truncate **repeatedly**: e.g. an iterative solver or an ODE integrator
whose T3-valued state is rounded a little every step, where a near-full extra SVD sweep per step can
dominate. A real cost comparison would have to sweep `d`, `N`, ranks, tolerances, spectra, numpy-vs-jax,
and CPU-vs-GPU, so this is genuinely the user's trade-off to make — which is why it is an option rather
than a fixed policy. (Note that under `jit` the rank bookkeeping — `compute_minimal_ranks` and the
minimality gate — is static and traced away; only the extra SVDs themselves have a runtime cost.)

Keep in mind the cost of *not* minimizing: non-minimal ranks make every later operation more expensive
(storage, contraction) and violate the minimal-rank precondition of `inner`/`norm` Hilbert–Schmidt
faithfulness and some manifold operations. With `minimize_ranks=False` you take responsibility for
managing the ranks — e.g. a later `t3svd()` (with the default) re-tightens losslessly when convenient.

## This is not specific to the Tucker extension

Plain TT-SVD / TT-rounding (Oseledets) has the same phenomenon: hard per-bond rank caps applied in a
single directional sweep can leave an upstream bond above its structural minimum, because a downstream
cap shrinks what that bond can support *after* it was already fixed. Tolerance-based rounding does not
exhibit it — each truncation, after orthogonalization, sees the *true* matricization singular values
and adapts. The cure in both settings is the same: a final structural re-tightening that propagates the
post-cap bottlenecks. T3 simply adds one more flavor (a bond cap orphaning a Tucker rank).

## See also

- `backend/t3_svd.py` — `t3svd` and `_shrink_to_minimal_ranks`.
- `backend/ranks.py` — `compute_minimal_ranks` (the structural minimal ranks).
- Tests: `test_t3svd_truncation_is_minimal`, `test_t3svd_lossless_compression_of_degenerate`,
  `test_compute_minimal_ranks_matches_matricization`, `test_compute_minimal_ranks_inequalities`.
- `docs/t3svd_verification.md` — the accuracy + rank-parsimony test method for the truncation itself.
