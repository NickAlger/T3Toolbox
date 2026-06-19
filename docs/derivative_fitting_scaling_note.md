# Scaling & normalization when fitting from derivatives — a short note

*Reference note. Captures the per-order scaling issue that arises when fitting a tensor from its
symmetric directional derivatives, and how the example handles it. No new code is implied — the
example's per-order normalization already does the right thing; this records the reasoning (and the
input-side "length scale" alternative) for the future.*

Context: `examples/fit_hilbert_from_apply_derivatives.py`,
`docs/derivative_order_information_and_conditioning.md` (caveat 1).

## The problem: the orders come out at wildly different magnitudes

Fitting from apply/probe **derivatives** measures, per `(X, P)` sample, a jet `y⁽⁰⁾, …, y⁽ᴷ⁾`. The
order-`t` component carries a `t!`/binomial-type weight, so the orders span many decades. Left alone,
this inflates `σ_max(JᵀJ)` and wrecks the Gauss-Newton conditioning — high orders are effectively
unsensed. The fit needs the graded information **balanced**.

## The homogeneity fact (the lever)

The restriction of a degree-`d` multilinear form to the line `X + sP` is a degree-`d` polynomial in `s`,
and `y⁽ᵗ⁾ = t!·[sᵗ]` of it. The order-`t` term is **degree `t` in `P`** (and, for apply, degree `d−t`
in `X`). Hence under input scaling,

```
y⁽ᵗ⁾(μX, λP) = μ^{d−t} · λᵗ · y⁽ᵗ⁾(X, P)        (apply; probe is degree d−1)
```

So **scaling the perturbation `P` by `λ` multiplies output order `t` by `λᵗ`** — only the `λ/μ` ratio
(perturbation magnitude relative to base-point magnitude) affects the *order balance*.

## Two ways to balance the orders (same imbalance, opposite ends)

| | where | what | how exact |
|---|---|---|---|
| **(A) Per-order output normalization** *(what the example does)* | on the **outputs** | for each order `t`, divide that order's measurements (and the operator's order-`t` output) by their RMS `m_t` over the training samples | **exact** — `K+1` separate factors, every order → unit RMS |
| **(B) Input length scale `λ`** *(not implemented)* | on the **input** `P` | scale `P` by one length `λ`, multiplying output order `t` by `λᵗ` | **geometric** — one number; flattens the trend, exact only if `m_t` is geometric |

Estimating (B) from data: regress `log m_t` against `t`; `λ = exp(slope)` flattens the geometric
component. (A richer version: a per-mode `λ_j` from the mode-`j` order-1 sensitivity, for tensors whose
modes have different natural scales.)

The two are **not separately needed**: also note they interact. If you per-order-normalize the outputs
(A), any input scale `λ` is *divided out* — so for fitting **synthetic, normalized** data, (B) changes
nothing. (A) is what makes the fit well-conditioned.

## When the input-side `λ` actually matters (and why we didn't implement it)

(B) earns its keep only where (A) can't reach — i.e. it is a **data-design / measured-data** tool, not a
fit knob:

1. **Measured data / SNR** — you cannot normalize away signal-to-noise. If derivatives are physically
   measured, `λ` sets each order's measurement magnitude, so balancing it balances the SNR across orders.
2. **Raw numerical range** — keeps raw orders from spanning `1e±8` before any division (precision during
   data generation).
3. **Physical interpretability** — `λ` is one meaningful length scale rather than `K+1` opaque factors.

For the synthetic example, none of these bite (the data is exact and we normalize the outputs), so the
input-side `λ` is omitted. **Decision: rely on per-order output normalization (A); estimate an input
`λ` via the `log m_t`-vs-`t` slope only if a future use case needs measured-data SNR balancing or a
physical scale.**
