# Fitting & optimization — implementation internals

Contributor-facing rationale behind the fitting stack. The user-facing reference — structure,
usage, and the design properties users rely on — is
[`../fitting_and_optimization.md`](../fitting_and_optimization.md); this note carries the
engineering decisions, tradeoffs, and deferred work.

## Backend-first (the razor)

The **algorithms live in the backend** (`backend/optimizers.py`), operating on raw cores / tangent tuples
via backend functions only, **free of the numerical safety preconditions** (which live in the frontend).
The frontend `optimizers.py` is a thin validate-once adapter. **Why:** an important minority of users
bypass the OO frontend and work on raw `.data` tuples — they must be able to run the *same* optimizer code.
It also means `jit` just works (no `unsafe()` wrapping — the backend has no checks to skip).

## Low-memory transpose: adjoint-state over scatter (store-vs-recompute)

The gradient `𝒥ᵀr` for apply/entries needs a **right context** beyond the forward's left sweep. Two ways:
the **scatter** *stores* the full frame sweep `(xi,mu,nu,eta)` (cheap matvec, more memory), or the
**adjoint-state** method *recomputes* the right context as a seeded `sigma_hat` reverse sweep (stores only
`(xi,mu)`, costs a sweep per matvec). T3Toolbox uses **adjoint-state** — **exactly 2× less `W`-scaling
memory**. **Why:** at real scale the `W`-batched edge variables get large; memory is the binding constraint
on a 40GB GPU, *worst* for minibatched Newton-CG (smaller batches → more ill-conditioned `H` → wants more
data → larger `W`). It is the classic **checkpointing** tradeoff, and the project's standing preference is
low memory over compute (trade ~2× compute for ~2× memory on `W`-scaling ops). *Probe can't use it* — its
residual is a vector (one free mode), which must be propagated by the full adjoint sweeps + `nu`/`eta`;
the scalar-seed shortcut is apply/entries-only. This is why probe's precompute is full and
apply/entries' is lean.

## numpy einsum: force BLAS-eligible pairwise paths; jax: one big einsum

The grouped contractions (`backend/contractions.py`) route through `_grouped_einsum`. **numpy:** a forced
greedy-pairwise path — because numpy's `optimize=True` minimizes FLOP *count*, and on a FLOP-tie runs a
single multi-operand contraction as one `c_einsum` loop **with no BLAS** (10–55× slower for the high-
dimensional order-combines). **jax:** one big einsum — XLA's opt_einsum + fusion is BLAS-aware and *beats*
any path we force. **Why it matters here:** the derivative forward/transpose are dominated by those
order-combines; the fix is 11–19× on them, numerically identical.

## Design-rationale tails (history and naming)

- **Geometry as a bundle** was the geometry refactor's core move: one geometry-generic
  `GaussNewtonModel`/`Problem`, so optimizers are written *once, not per geometry* (six classes
  collapsed to one model + two geometries).
- **The `precompute` / `*_from_sweep` split** is deliberately *named and public* in the backend
  (rather than a private cache) so a raw-`.data` user composes public functions with docstrings —
  the razor's "capability, not line count" corollary.
- **Order-slicing is not a `draw`**: order is an *output-only* axis — the forward computes the whole
  jet jointly, so subsetting orders is output-*masking*, not input-*slicing*. Deferred, and most
  naturally an outer continuation loop anyway.

## What's deferred (not built)

- **The Goal-1 `fit(...)` facade** — a "just fit my tensor" entry point that picks a sensible geometry +
  optimizer, supplies the geometry-correct `x0`, and runs **rank continuation** with validation. The
  current layer is a clean *mid-level toolkit*; the facade is what delivers "standard user, no fiddling".
  Rank continuation + validation currently live in the examples (the right defaults: manifold → zero start
  + warm continuation; corewise → nonzero start + cold per level — see `dev/archive/optimizers_plan.md` §7).
- **A Riemannian L-BFGS** (with vector transport) — the quasi-Newton method only this library could
  provide; the Euclidean case is deliberately left to the scipy bridge.
- **Order-slicing minibatches** (output-masking) and order/polynomial-degree continuation — research,
  likely outer loops.
- **The example pass** — deciding which `examples/fit_hilbert_*` use the library optimizers vs keep inline
  to illustrate the hidden hooks (`gn_hessian`, `gn_quadratic`, `corewise_map`); `dev/archive/optimizers_plan.md` §10.
- **Per-sample gradients / multi-source fits** (SVRG-style; fitting from applies *and* entries together) —
  reachable at the backend level (`sum_over_probes=False`; sum two local models), not packaged.

## Plans and research pointers

- `dev/archive/optimizers_plan.md` (the optimizers + example two-track plan),
  `dev/archive/derivative_fitting_plan.md` (the D1–D4 derivative-fitting build),
  `dev/archive/geometry_refactor_plan.md` (the geometry abstraction, incl. the corewise-vs-autodiff
  benchmarks behind the user doc's §4.7), `dev/archive/fitting_plan.md` (the original fitting-layer
  build; its §9 records the exact-GN dense-truth oracle behind the user doc's §4.2),
  `dev/archive/safe_unsafe_mode_plan.md` (the safe-mode preconditions the frontend enforces).
- The polynomial/derivative fitting studies (MC-SGD minibatch findings, the symmetric-fit "halo")
  live in a separate research repo (maintainer-local); the user doc's §5 carries their practical
  takeaways.
