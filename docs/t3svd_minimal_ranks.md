# T3-SVD ranks: why `t3svd` may return non-minimal ranks, and how to minimize

A short note for anyone wondering about the ranks that come back from `TuckerTensorTrain.t3svd`. The
short version: **`t3svd` is the basic algorithm (Algorithm 10 / Oseledets' TT-SVD); under truncation it
does *not* guarantee minimal ranks.** It always returns a **left-orthogonal** result, and to reduce to
minimal ranks you call the separate `rank_adjustment_sweep`. This note explains what "minimal" means,
why the truncating sweep can leave non-minimal ranks, and why one `rank_adjustment_sweep` fixes it only
in the matching direction.

## What "minimal ranks" means

A T3 has **minimal ranks** when every core satisfies the no-redundancy inequalities

```
n_i  <= N_i           (Tucker rank <= mode dimension)
n_i  <= rL_i * rR_i   (Tucker rank <= product of its two TT bonds)
rL_i <= n_i * rR_i    (each TT bond <= product of the other two legs of its core)
rR_i <= n_i * rL_i
```

with `rL_i = r_i`, `rR_i = r_{i+1}`. These are exactly the **matricization ranks** of a generic tensor
with that structure: cutting a single edge of the T3 tensor network splits the physical indices into two
groups, and the minimal dimension of that edge is the rank of the corresponding matricization. (The T3
network is a *tree* — a TT backbone with a pendant Tucker matrix on each node — so every single-edge cut
is a clean bipartition with no hidden multi-cut degeneracy.) `compute_minimal_ranks` computes these by
propagating the structural bottlenecks; check with `has_minimal_ranks`. See
`test_compute_minimal_ranks_matches_matricization`.

## Why the truncating sweep can leave non-minimal ranks

`t3svd` sweeps left to right. At mode `i` it first truncates the **Tucker** rank `n_i` (against the
current right bond `rR_i`), then truncates the **TT bond** `rR_i`. With **no truncation** — or with a
tolerance (`rtol`/`atol`) — each SVD keeps exactly `min(rows, cols)`, the structural rank, and a bond
only ever shrinks to its structural value `rL_i * n_i`. That can never orphan the Tucker rank computed
just before it (`n_i <= rL_i * rR_i^new = rL_i^2 * n_i` holds automatically), so **the no-truncation and
tolerance results are already minimal**.

A **hard rank cap** breaks this. A `max_tt_ranks` cap can force `rR_i` *below* its structural value
`rL_i * n_i`. Then `n_i`, fixed moments earlier against the larger pre-cap bond, ends up `> rL_i * rR_i`
— a Tucker rank pointing into a subspace the truncated bond can no longer reach. It is **structurally
redundant**: the represented tensor is unaffected, but the reported rank is not minimal. (The symmetric
thing happens to a bond when a downstream `max_tucker_ranks` cap bites.)

Worked example — `shape=(5,6,7)`, truncate the TT bonds to 2 but leave the Tucker ranks uncapped:

```
n_0 is fixed to 3 (= 1 * r_1 with the pre-cap bond r_1 = 3)
then the cap forces r_1 = 2  <  rL_0 * n_0 = 1 * 3
=> n_0 = 3  >  rL_0 * r_1 = 1 * 2 = 2     # orphaned: minimal n_0 is 2
```

This is deliberate: `t3svd` is the basic algorithm and does the minimum work. Re-tuning ranks is a
separate, opt-in step.

## How to minimize: `rank_adjustment_sweep`

`rank_adjustment_sweep(direction)` is a single **lossless** sweep that drops structurally-redundant
ranks (re-SVD each Tucker edge and TT bond with no cap). The represented tensor is unchanged.

- `'right_to_left'` returns a **right-orthogonal** result;
- `'left_to_right'` returns a **left-orthogonal** result.

**A single sweep reaches minimal ranks only if the input is already orthogonal in the *opposite*
direction.** A left-orthogonal input already satisfies the forward (left) bounds, so a `'right_to_left'`
sweep adds the backward (right) bounds and the result is fully minimal — and a `t3svd` result *is*
left-orthogonal, so this is the common path:

```python
x2, _, _ = x.t3svd(max_tt_ranks=2)        # left-orthogonal, possibly non-minimal
x3 = x2.rank_adjustment_sweep('right_to_left')   # minimal, right-orthogonal (lossless)
```

On a general input (orthogonal in neither direction, or the same direction as the sweep) one sweep is a
*partial* reduction; compose both directions (`'right_to_left'` then `'left_to_right'`) for guaranteed
minimal ranks — e.g. to get a **minimal, left-orthogonal** result. This is why the operation is named a
"sweep", not "minimize": a single sweep only fully minimizes given the matching precondition.

Cost: it is one extra sweep, paid only when you ask for it. Keep in mind the cost of *not* minimizing —
non-minimal ranks make every later operation more expensive (storage, contraction), and retraction on a
non-minimal frame drops the redundant rank rather than preserving ranks strictly. (Minimal rank is **not**
a correctness precondition for any operation — `inner`/`norm` Hilbert–Schmidt faithfulness needs only an
orthogonal frame and gauged variations; the settled audit is
[`numerical_contracts.md`](numerical_contracts.md).)

## This is not specific to the Tucker extension

Plain TT-SVD / TT-rounding (Oseledets) has the same phenomenon: hard per-bond rank caps applied in a
single directional sweep can leave an upstream bond above its structural minimum, because a downstream
cap shrinks what that bond can support *after* it was already fixed. Tolerance-based rounding does not
exhibit it — each truncation, after orthogonalization, sees the *true* matricization singular values and
adapts. The cure is the same: a structural sweep that propagates the post-cap bottlenecks. T3 simply adds
one more flavor (a bond cap orphaning a Tucker rank).

## See also

- [`contributor/t3svd_design_rationale.md`](contributor/t3svd_design_rationale.md) — *why* T3-SVD and minimization are split, and
  why `rank_adjustment_sweep` is a directional sweep rather than a `minimize()` method.
- `backend/t3_svd.py` — `t3svd` (basic algorithm) and `rank_adjustment_sweep` (the minimization).
- `backend/ranks.py` — `compute_minimal_ranks` (the structural minimal ranks), `compute_raw_sweep_ranks`
  (the ranks the sweep produces under caps).
- Tests: `test_t3svd_is_left_orthogonal_not_necessarily_minimal`, `test_rank_adjustment_sweep`,
  `test_compute_minimal_ranks_matches_matricization`, `test_compute_minimal_ranks_inequalities`.
- `docs/contributor/t3svd_verification.md` — the accuracy + rank-parsimony test method for the truncation itself.
