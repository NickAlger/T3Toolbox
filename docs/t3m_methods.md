# Elementwise multiplication: the three `t3m` methods

If you multiply two Tucker tensor trains elementwise, the ranks multiply: the product of a
`(n_x, r_x)`-rank and a `(n_y, r_y)`-rank T3 has Tucker ranks `n_x·n_y` and TT bonds `r_x·r_y`.
Plain `x * y` gives you exactly that -- the **exact** full product, stack-aware, no truncation.
`x.t3m(y, ...)` is the same product with **optional rank truncation**, computed by one of three
interchangeable algorithms that cover complementary cost regimes. All three return the same
tensor; they differ in how much memory and compute they spend getting there.

## TL;DR

| `method=` | algorithm | sweet spot |
|---|---|---|
| `'form_then_round'` | form the full product (`t3_mult`), then round with `t3svd` | small bonds; the forming step is embarrassingly parallel; the reference/oracle |
| `'inplace_fused'` (default) | fused left-to-right sweep; truncates as it goes, never materializes the full product | large `d` -- the workhorse default |
| `'swap'` | swap-based chain merge (the TTM generalization), `O(d²·r³)` compute, `O(r̃²)` memory | `r ≫ d` (huge bonds, few modes); has the `oversample` knob |

With **no truncation requested, every method short-circuits to the exact full product** -- so
`t3m(method=...)` with no tolerances is fast and exact regardless of method, and `*` is method
`'form_then_round'` with no truncation.

## What there is to truncate

Write a T3 as `T[x] = Σ_n (∏_i U_i[n_i, x_i]) · C[n]`, with the central Tucker core `C` stored as
a tensor train. The elementwise product splits into **two truncatable rank families**:

- **Tucker factors:** `W_i = U_i^x ⊙ U_i^y` (row-wise Khatri-Rao), rank `n_x·n_y` → truncate to `ñ_i`;
- **central core-TT bonds:** `G̃_i = G_i^x ⊗ G_i^y` (Kronecker), bond `r_x·r_y` → truncate to `r̃_i`.

Both families blow up quadratically, which is why an untruncated product chain gets expensive fast
and why `t3m` exists.

## Choosing a method

The design analysis (constants depend on whether the Tucker rank `n` or the bond `r` dominates;
full derivations in
[`ttm_t3m_ht_note.tex`](https://github.com/NickAlger/T3Toolbox/blob/main/docs/ttm_t3m_ht_note.tex)):

| method | compute | memory |
|---|---|---|
| `'form_then_round'` | `O(d·r⁶)` | `O(d·r⁴·n²)` (the whole product) |
| `'inplace_fused'` | `O(d·r⁴)` | `O(r̃·n²·r²)` (one active site) |
| `'swap'` | `O(d²·r³)` | `O(r̃²)` |

The crossover is **`d` versus `r`**: the fused and swap methods both avoid forming the full
product (so they beat `'form_then_round'` on memory everywhere); between the two,
`'inplace_fused'` wins for `d ≳ r` and `'swap'` wins for `r ≫ d`. `'form_then_round'`'s virtue is
that its forming step has no sweep at all -- fully parallel -- and it doubles as the oracle the
other two are tested against.

## The truncation spec (shared with `t3svd`)

Any combination; default none = exact product:

- `max_tucker_ranks`: an `int` (caps every mode) or a length-`d` sequence;
- `max_tt_ranks`: an `int` (caps every bond) or a length-`d+1` sequence (list, tuple or array);
- `rtol` / `atol`: scalar tolerances, applied **per truncation step** -- as with `t3svd`, the
  per-step errors accumulate in quadrature, so the realized error can exceed the request by up to
  `√(2d−1)`;
- `rtol`/`atol` **require unstacked input** (different stack elements could truncate to different
  ranks, which one stacked object cannot represent); **max-rank truncation is stacking-compatible**.

**Boundary bonds.** A `TuckerTensorTrain` may carry boundary TT ranks `r0, rd != 1` (e.g. a `segment`,
a `resize`, or a train built from a TT with open ends). The rule is: an **exact** product (`t3_mult`,
the `*` operator) keeps the Kronecker structure on every bond, boundary bonds included (`r0 = r0_x · r0_y`),
because a downstream operation may rely on that algebraic structure; a **truncating** product has no
structure to preserve, so all three `t3m` methods canonicalize both operands' boundary bonds to 1 on entry
(`squash_tails`) and the result always has `r0 = rd = 1`. Before 2026.2.0 `inplace_fused` and `swap` did not
squash, so an unsquashed trailing bond escaped `max_tt_ranks` (and `swap` contracted the two trailing bonds
against each other).

## The quality guarantee (joint truncation)

All three methods truncate the Tucker ranks **jointly** -- weighted by the central TT environment,
exactly as `t3svd`'s per-site SVDs do -- rather than by an unweighted per-mode SVD. The practical
consequence: for the same tolerance, `'inplace_fused'` and `'swap'` never keep larger ranks than
`'form_then_round'` -- the memory-efficient methods compress at least as well as the reference,
not worse. (This property is pinned by `tests/test_t3m.py::test_swap_joint_quality`.)

## Oversampling (`method='swap'` only)

The swap method has one T3-specific wrinkle: the merged Tucker leg of the last-merged mode is
still full-rank at the moment its bonds are fixed, so purely in-process truncation is only
quasi-optimal under aggressive truncation. `oversample = k ≥ 1` resolves it with a uniform rule:
**relax every active criterion by `k` in-process** (rank caps to `⌈k·target⌉`, tolerances to
`tol/k`) purely to bound memory, **then run a single `t3svd` cleanup at the exact targets** for
the decisive truncation.

- `k = 1` (default): no cleanup -- the lowest-memory, lowest-quality corner. With `rtol` at
  `k = 1` the per-step tolerances accumulate to roughly `d·rtol` overall; use `oversample > 1`
  when you need tighter quality from tolerance-based truncation.
- A modest `k ≈ 2` is a good default (near-`'form_then_round'` quality); the backend records
  `k ≈ 3` as recovering full `t3svd` quality. Memory stays `O((k·r̃)²)`, still far below the
  `O(r²)` full product when `r̃ ≪ r`.
- A **per-position** `max_tt_ranks` sequence also triggers the cleanup: the swaps themselves can
  only cap bonds uniformly, and the cleanup is what honors the sequence exactly.

Why the tension is intrinsic (and why converting to a balanced hierarchical format would not
avoid it):
[`ttm_t3m_ht_note.tex`](https://github.com/NickAlger/T3Toolbox/blob/main/docs/ttm_t3m_ht_note.tex).

## Code pointers

- Frontend: `TuckerTensorTrain.t3m(other, method=..., max_tucker_ranks=..., max_tt_ranks=...,
  rtol=..., atol=..., oversample=...)`; plain `*` is the exact product.
- Backend: `backend/t3_linalg.py` -- `t3_mult` (the exact product) and `t3m_form_then_round` /
  `t3m_inplace_fused` / `t3m_swap`.
- Tests: `tests/test_t3m.py` (dense-oracle harness, cross-method agreement, the joint-quality
  guard, oversample and stacking cases).
