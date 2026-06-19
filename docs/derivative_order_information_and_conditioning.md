# Why fit from probe *derivatives*? Information content & Hessian conditioning

*Idea-capture note (a hypothesis with a clean supporting argument, not yet experimentally verified).*

**The question.** When fitting a tensor (or a manifold tangent vector) to sampled data, how does using
higher-order *probe derivatives* — `y_i^(k) = d^k/ds^k y_i(X + sP)|_0` — instead of (or alongside)
ordinary order-0 probes enrich the information content of the data, and how does the derivative order
affect the conditioning of the Riemannian Gauss-Newton Hessian `J^T J`?

**The intuition we're explaining.** All else equal (cost ignored), one "ought" to use a number of
derivative orders equal to the tensor order — e.g. to fit a 5th-order tensor, use derivative data from
orders `1..5`. Below is why that intuition is not a heuristic but an exact dimension count, plus where
it helps conditioning and where the caveats are.

## 1. The grounding identity: a line-restriction is a polynomial of degree = #contracted modes

Fix a base point `X` and a direction `P` and restrict the action to the line `ℓ = {X + sP}`. Because the
multilinear form is affine in each argument, the restriction to `ℓ` is a **univariate polynomial in `s`**
whose degree equals the number of *contracted* modes:

- **apply** (all `d` modes contracted): degree `d`  → derivative orders `0..d`;
- **probe** (one mode left free): degree `d-1`        → derivative orders `0..d-1`.

The symmetric derivatives are exactly its coefficients, `f^(k) = k! · [s^k] f(X + sP)`. So "orders
`1..5` for a 5th-order tensor" is the **apply** count (`d=5` ⇒ degree-5 polynomial, orders `0..5`, with
`1..5` the new derivative info). **Above the tensor order the derivatives are identically zero** — a hard
ceiling. That is the precise statement of the intuition: the number of useful orders equals the tensor
order because that is where the line-polynomial saturates. Fewer orders under-determine the line; more
orders are exact zeros.

## 2. Information content: derivatives ≡ line-sampling, saturating at the tensor order

A degree-`D` line-restriction has `D+1` degrees of freedom. The derivatives `0..D` at one point pin it
down **exactly and minimally** (`D+1` numbers for `D+1` DOF). Equivalently:

> derivatives at one `(X,P)`  ≡  probing the line `ℓ` at `D+1` distinct points.

They carry the **same** information (the line-restriction of `T`), just re-encoded — Taylor/monomial vs.
Lagrange/point. So derivatives add **no information beyond dense sampling along that line**; what they
buy is extracting the whole line in one swept computation, letting you choose *few, maximally informative*
base points instead of many scattered ones.

## 3. Conditioning: derivatives are the best-conditioned encoding of the line

Recovering the `D+1` line coefficients from:

- **derivatives `0..D`**: the map is **diagonal** (`f^(k) = k! · c_k`) — condition number ≈ 1 for that
  line's block;
- **collinear point-probes**: the map is an **inverse Vandermonde** — conditioning grows with `D` and
  degrades badly for clustered/equispaced points.

So for a fixed line's worth of information, derivatives are the **best-conditioned encoding** of it. In
Gauss-Newton terms, each `(X,P)` contributes a block to `J^T J`; using the full `0..D` set makes that
block **full-rank and well-conditioned in the graded (symmetric-power) basis**, whereas truncating leaves
the high-degree directions unsensed (a rank-deficient block). Adding orders can only **raise `σ_min`**
(more directions sensed) and it **plateaus at the tensor order** (no rows beyond). That is the conditioning
version of the count: `σ_min` improves with order up to the tensor order, then saturates.

## 4. The two caveats that decide whether it actually helps

1. **Per-order normalization is essential.** `f^(k)` carries a `C(d-1,k) · ‖p‖^k · k!`-type weight, so raw
   orders span wildly different magnitudes — that alone inflates `σ_max` and wrecks the conditioning. With
   per-order normalization (cf. the unit-norm-rows fitting recipe), the graded information comes in balanced
   and `σ_max` stays controlled. "Well-conditioned" is conditional on this. (The output-side per-order
   normalization and the equivalent input-side "length scale" `λ` — scaling `p` so order `t` scales by `λ^t`
   — are written up in [`docs/derivative_fitting_scaling_note.md`](derivative_fitting_scaling_note.md).)
2. **One line ≠ the whole tensor.** Each `(X,P)` constrains only its line's slice of `T` (the graded
   combinations along it). Covering the manifold's full DOF still needs **spatial diversity** — many `X`,
   many `P`. Derivatives make each base point *maximally* informative, so the benefit is largest when
   **base points are the scarce resource** (exactly when you'd reach for them, cost aside). With unlimited
   cheap order-0 probes scattered everywhere, low orders already span — derivatives then mostly buy
   conditioning, not rank.

(For *measured* data there is a third caveat: high-order derivatives are harder to measure accurately, so
their rows carry lower SNR. For *synthetic* data generated from a known tensor — our backend — this does
not apply; the derivatives are exact.)

## 5. A cheap experiment to test it

Take a small `T` (say `d=5`, modest ranks). Build `J` (the forward operator we implemented) for a fixed
pool of random `(X,P)` and a cutoff `K_max`, and plot, as `K_max` runs `0 → d`:

- `rank(J)` / the smallest few singular values, and `cond(J^T J)` — **per-order-normalized**;
- repeated with base-point counts from "few" to "many".

Prediction: with **few** base points, `σ_min` climbs and `cond` improves up to `K_max = d` then flattens
(a visible **knee at the tensor order**); with **many** base points the knee moves to lower `K_max`
(saturation arrives earlier). A knee-at-`d` would be the evidence. It is a one-screen `numpy.linalg.svd`
loop on top of the existing forward.

## Summary

- The count "number of derivative orders = tensor order" is **exact** — it is the dimension of the
  line-restriction polynomial (degree `d` for apply, `d-1` for probe); beyond it the derivatives vanish.
- Per `(X,P)` line, the full order set extracts the entire line-restriction, in the **most
  well-conditioned form** possible (diagonal recovery, vs. the inverse-Vandermonde of collinear point
  sampling) — so the per-line Hessian block is full-rank and well-conditioned.
- The win is **real but contingent**: requires per-order normalization, and it complements (does not
  replace) spatial diversity in `(X,P)`. It matters most when base points are precious.
