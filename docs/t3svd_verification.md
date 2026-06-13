# Verifying a fixed-rank T3-SVD: error *and* rank checks

A short note on how the test suite checks a Tucker-tensor-train SVD (T3-SVD: truncate a tensor `X`
to a fixed-rank T3 `X̃`, with the ranks chosen from `rtol`/`atol` and/or hard caps). The method
combines a generalized Oseledets error bound with a rank upper bound proved by a projection argument.
The combination — in particular *which ranks each check uses* — is the subtle part, and getting it
wrong produces a rare, input-dependent test failure.

## Why two checks

A correct truncation must be both **accurate** and **parsimonious**:

1. **Accuracy** — `X̃` approximates `X` to within the requested tolerance.
2. **Parsimony** — `X̃` uses ranks no larger than necessary for that tolerance.

Neither alone suffices: a routine that returns `X` unchanged (maximal ranks) passes any accuracy
check; a routine that returns `0` (rank 1) passes any rank check. So we check both. The trap is that
the two checks are linked through the ranks, and if the accuracy bound is evaluated at the wrong ranks
a faulty routine can borrow slack from one check to pass the other (see the last section).

## Notation

Work per stacked slice (batch axes drop out). `X ∈ ℝ^{N₁×···×N_d}`; `‖·‖` is the Frobenius norm.

- **TT unfoldings** `X⁽ᵏ⁾`, `k = 1..d−1`: reshape `X` to `(N₁···N_k) × (N_{k+1}···N_d)`; its singular
  values `σ⁽ᵏ⁾` govern the TT rank `r_k`.
- **Tucker matricizations** `X_[i]`, `i = 1..d`: mode-`i` matricization `N_i × ∏_{j≠i} N_j`; its
  singular values `σ_[i]` govern the Tucker rank `n_i`.
- `X̃`: the routine's output, with **chosen** ranks `r̂_k` (TT) and `n̂_i` (Tucker).
- Threshold `τ = max(rtol·‖·‖, atol)`.

## 1. Accuracy check (Oseledets bound, generalized to T3)

Oseledets (2011), *Tensor-Train Decomposition*, SIAM J. Sci. Comput. 33(5):2295–2317, **Theorem 2.2**:
the TT-SVD approximation satisfies `‖X − X̃‖² ≤ Σ_{k=1}^{d−1} εₖ²`, where `εₖ` is the Frobenius norm of
the singular values discarded when truncating the `k`-th unfolding. A T3 truncates the Tucker factors
*and* the TT core; both are orthogonal projections (in complementary directions), so by Pythagoras the
errors add in quadrature, giving the generalized bound

```
            ┌  d−1                          d                          ┐
‖X − X̃‖² ≤ │  Σ   ‖ σ⁽ᵏ⁾_{> r̂_k} ‖²   +    Σ   ‖ σ_[i]_{> n̂_i} ‖²   │
            └ k=1   (TT unfoldings)        i=1   (Tucker matricizations)┘
```

**The discarded tails are taken beyond the ranks the routine *actually returned* (`r̂_k`, `n̂_i`),
evaluated on the *original* `X`'s singular values.** This is a valid upper bound on the true error:
each sequential truncation discards a tail of a *projected* unfolding, whose singular values are `≤`
those of `X` (Lemma below), so the discarded mass is `≤ ‖σ⁽ᵏ⁾_{> r̂_k}‖`.

> **Algorithm 1 — accuracy check.**
> ```
> input: X (reference), X̃ (result), tol_num (numerical slack)
> E² ← 0
> for k in 1..d−1:                       # TT unfoldings
>     σ ← svals(unfold_k(X));  E² += Σ_{j > r̂_k} σ_j²
> for i in 1..d:                         # Tucker matricizations
>     σ ← svals(matricize_i(X));  E² += Σ_{j > n̂_i} σ_j²
> assert ‖X − X̃‖ ≤ sqrt(E²) + tol_num · ‖X‖
> ```

## 2. Rank check (the projection argument)

Because Algorithm 1 evaluates the bound at the chosen ranks, a routine that returns huge ranks has
vanishing tails and passes accuracy trivially. Parsimony is enforced by bounding the chosen ranks
*above* by the optimal ranks of the original tensor:

> **Claim.** `r̂_k ≤ ρ_k := #{ j : σ⁽ᵏ⁾_j ≥ τ }` with `τ = max(rtol·‖X̃‖, atol)`, and likewise
> `n̂_i ≤ #{ j : σ_[i]_j ≥ τ }`. The routine uses no more rank than truncating the *original* unfolding
> at threshold `τ` would.

**Lemma (singular values contract under projection).** For orthogonal projections `P, Q` (so
`‖P‖₂, ‖Q‖₂ ≤ 1`) and any matrix `A`, `σ_j(P A Q) ≤ ‖P‖₂ · σ_j(A) · ‖Q‖₂ ≤ σ_j(A)` for every `j`.

**Proof of the claim.** The sequential T3-SVD processes the modes in turn. Just before truncating mode
`k` it holds a reduced tensor `C_k` obtained from `X` by the earlier truncations, each of which is an
orthogonal projection onto retained dominant singular subspaces. Hence the `k`-th unfolding is
`unfold_k(C_k) = P · unfold_k(X) · Q` for orthogonal projections `P, Q`, and by the Lemma

```
        σ_j( unfold_k(C_k) )  ≤  σ⁽ᵏ⁾_j        for all j.            (∗)
```

The routine chooses `r̂_k = #{ j : σ_j(unfold_k(C_k)) ≥ τ_k }`, where `τ_k` is its per-step threshold.
Orthogonal projection does not increase the Frobenius norm, so the running norm is monotone
non-increasing, `‖X̃‖ ≤ ‖C_k‖`; hence for relative truncation `τ_k ≥ rtol·‖C_k‖ ≥ rtol·‖X̃‖`, and
`τ_k ≥ atol` for the absolute part — so `τ_k ≥ τ := max(rtol·‖X̃‖, atol)`. Combining with `(∗)`,

```
r̂_k = #{ σ_j(unfold_k(C_k)) ≥ τ_k }  ≤  #{ σ_j(unfold_k(C_k)) ≥ τ }  ≤  #{ σ⁽ᵏ⁾_j ≥ τ } = ρ_k.   ∎
```

The Tucker matricization truncations are likewise orthogonal projections onto dominant mode subspaces,
so the same bound holds for `n̂_i`. (Floor `ρ` at 1, the minimal core rank, and intersect with any
user max-rank cap; both preserve the inequality.)

> **Algorithm 2 — rank check.**
> ```
> input: X, X̃, rtol, atol  (and optional max-rank caps)
> τ ← max(rtol · ‖X̃‖, atol)
> for k in 1..d−1:
>     ρ ← max(1, #{ j : svals(unfold_k(X))_j ≥ τ });   ρ ← min(ρ, cap_k)   # if a cap was given
>     assert r̂_k ≤ ρ
> for i in 1..d:
>     ρ ← max(1, #{ j : svals(matricize_i(X))_j ≥ τ }); ρ ← min(ρ, cap_i)
>     assert n̂_i ≤ ρ
> ```

## The consistency pitfall (the bug this prevents)

The two checks must use ranks consistently:

- The **accuracy** bound (Alg. 1) uses the **chosen** ranks `r̂` — this is the only choice that yields a
  true upper bound on `‖X − X̃‖`.
- The **rank** bound (Alg. 2) uses the **optimal** ranks `ρ` — the projection argument certifies `r̂ ≤ ρ`.

It is tempting to compute the accuracy bound at the optimal ranks `ρ` instead (doing both in one pass).
That is wrong: when the sequential truncation's chosen `r̂_k` differs from `ρ_k` — which happens at a
singular value sitting near `τ` — the bound evaluated at `ρ_k` can be *smaller* than the true error,
so accuracy fails spuriously on a small fraction of inputs. Keeping the two checks decoupled (error at
`r̂`, rank at `ρ`) is what makes the test both correct and a genuine guard against inflated ranks.

---

*See `tests/test_tucker_tensor_train.py::test_t3svd` and `::test_t3svd_dense` for the implementation.*
