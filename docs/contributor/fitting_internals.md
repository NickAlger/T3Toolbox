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

## Regularization

A `regularizer` is an **objective term** `ρ(x)` folded into the backend `LocalModel`
(`objective`/`gradient`/`gn_quadratic`/`hvp`) and `Problem.objective`, so it reaches **every** optimizer and
kind for free (they only ever touch those surfaces). It is **backend-homed** (`backend/regularization.py`,
check-free) and re-exported, so a raw-`.data` user attaches it with the same one kwarg as the frontend.
**Why an objective term and not Newton-Hessian-only damping** (decision D2): damping the Hessian alone is
invisible to the objective and gradient, so the first-order optimizers (`gradient_descent`/`mc_sgd`/`adam`)
and even Newton's line search would be *unregularized*. To reach all of them, regularization must be a term
in the objective, hence contribute to the gradient.

**Identity reg is `ρ = ½λ‖x‖²` in the geometry's own tangent metric** (decision D1) → it contributes
`H_R = λ·project` (`= λp` for a gauged tangent). On the **manifold** the tangent coordinate metric *is* the
Hilbert–Schmidt metric, so this is exactly the ambient ridge `½λ‖x‖²_HS`; on **corewise** it is weight decay
`½λΣ‖core‖²` (which, as a bonus, makes the otherwise gauge-singular corewise Newton Hessian strictly PD).
Regularizing in each geometry's own metric — rather than an ambient ridge bolted on some other way — is also
what makes the **uniform** layer safe (below), because it inherits the mask-correctness of `inner`/`project`.

**The base point as a tangent (`v_X`) — the non-obvious part.** The reg gradient needs `x` in tangent
coordinates. Key fact: the attachment point `X` is a *single gauged tangent term* — **all variations zero
except the last TT variation, set to the frame's last left core `P_last`** (the one slot with no gauge
condition, so it is already gauged; `dense(v_X)=X`). Hence `point_tangent(frame)=v_X` is a direct
construction and `point_norm_sq = ‖P_last‖²` — **not** the naive "sum of frame-core norms," which is *wrong*
(orthogonal frame cores have norm `√rank`, not content: measured `42291.8` vs the true `‖X‖²=42280.8`). The
tangent isometry (paper Appendix A.3) is stated for the zero-centered space but extends to `X` because `v_X`
is itself gauged. A consequence worth knowing: on the manifold `H_R = λI` is the **exact** Riemannian Hessian
(grad `= x` is fully tangent — scaling preserves rank — so the Weingarten curvature term vanishes), not just
a Gauss–Newton approximation.

**`value(x)` (the line-search objective) rides the retraction's orthogonality.** `MANIFOLD.retract` → `t3svd`
always emits **left-orthogonal** cores, so `‖x‖_HS = ‖x.tt_cores[-1]‖` — one core norm, no re-orthogonalization,
and the left-orthogonalization is already paid to form the candidate for the data term. "Read the last core"
is correct *only* for a left-orthogonal input — a numerical precondition, so per the house rule the backend
`value` is **check-free** and the matching checker/orthogonalizer (`t3_orthogonality_residual` /
`t3_left_orthogonalize`) are already public: a backend user self-guards with the same tools the frontend
safe-mode uses. No new checker, no backend/frontend inequality.

**Uniform mask-safety.** `ρ` is a *reduction*, so a naive `½λ·Σ(supercore²)` would sum the garbage padding in
the out-of-mask rank slots — wrong *and* nondeterministic (the phantom-rank failure class,
[`testing_strategy.md`](testing_strategy.md)). Design rule: the regularizer routes **only** through mask-aware
primitives (`geom.inner`/`geom.project`/the uniform tangent norms) and never indexes raw supercores, so it
inherits their mask-correctness. `g_R`/`H_R p` are tangents (garbage inert) but their output masks must stay
exact — `project` guarantees it. Tested with large-finite garbage padding (`1e6·(1−mask)`), not just
dense-vs-ragged (which is blind to it).

**Stochastic scaling.** `ρ` is deterministic, but the minibatch data-gradient is a `batch/n` estimate of the
full gradient, so `mc_sgd`/`adam` scale the reg by `min(batch,n)/n` per step (the generic `_ScaledRegularizer`
wrapper), keeping the minibatch step ≈ the full step and `λ`'s meaning batch-invariant; the full-batch stop
keeps full-strength reg. A custom `draw` of a non-nominal batch size may want `λ` retuned.

## What's deferred (not built)

- **The Goal-1 `fit(...)` facade** — a "just fit my tensor" entry point that picks a sensible geometry +
  optimizer, supplies the geometry-correct `x0`, and runs **rank continuation** with validation. The
  current layer is a clean *mid-level toolkit*; the facade is what delivers "standard user, no fiddling".
  The rank *ladder* is library surface (`x.continuation_ranks(...)` / `backend.ranks.compute_continuation_ranks`,
  user page [`../rank_continuation.md`](../rank_continuation.md)); what is still example-only is the outer
  continuation **loop** and validation (the right defaults: manifold → zero start + warm continuation;
  corewise → nonzero start + cold per level — see `dev/archive/optimizers_plan.md` §7).
- **A Riemannian L-BFGS** (with vector transport) — the quasi-Newton method only this library could
  provide; the Euclidean case is deliberately left to the scipy bridge.
- **Order-slicing minibatches** (output-masking) and order/polynomial-degree continuation — research,
  likely outer loops.
- **The example pass** — deciding which `examples/fit_hilbert_*` use the library optimizers vs keep inline
  to illustrate the hidden hooks (`gn_hessian`, `gn_quadratic`, `corewise_map`); `dev/archive/optimizers_plan.md` §10.
- **Per-sample gradients / multi-source fits** (SVRG-style; fitting from applies *and* entries together) —
  reachable at the backend level (`sum_over_probes=False`; sum two local models), not packaged.
- **Grasedyck–Kramer regularization** — a `SingularValueRegularizer` implementing the same four
  `Regularizer` methods with a frame-dependent SPD `M` built from the cores' unfolding singular values
  (penalize poorly-determined, small-σ directions more). Drops in with **no** change to
  `Problem`/`LocalModel`/optimizers/kinds — the payoff of designing the protocol now — and is the sharper
  tool for denoising a *good* fit (where identity λ is only a modest secondary knob; user doc §5). Reuses
  `point_tangent`/`project` composed with an `M`-apply on tangents.
- **Base-point-as-tangent as a public op.** Representing `X` as its gauged tangent `v_X` (last TT variation
  `= P_last`, else zero — already gauged) is broadly useful beyond the reg gradient; worth a first-class
  `T3Tangent`/`UT3Tangent` factory + backend helper. The reg's `point_tangent`
  (`backend/geometry.py::fv_base_point_tangent` / `ufv_base_point_tangent`) is the
  current internal implementation to promote.
- **`already_left_orthogonal` retraction amortization** — `MANIFOLD.retract` (`t3svd`) emits left-orthogonal
  cores, but the next step's `t3_orthogonal_representations` re-orthogonalizes from scratch;
  `already_left_orthogonal=True` skips that sweep. Passing it after a retraction speeds **every** Newton step
  (needs the optimizer to carry a "came from retract" flag). Surfaced by the reg `value(x)` design, which
  relies on the same left-orthogonality.

## Plans and research pointers

- `dev/archive/optimizers_plan.md` (the optimizers + example two-track plan),
  `dev/archive/derivative_fitting_plan.md` (the D1–D4 derivative-fitting build),
  `dev/archive/geometry_refactor_plan.md` (the geometry abstraction, incl. the corewise-vs-autodiff
  benchmarks behind the user doc's §4.7), `dev/archive/fitting_plan.md` (the original fitting-layer
  build; its §9 records the exact-GN dense-truth oracle behind the user doc's §4.2),
  `dev/archive/safe_unsafe_mode_plan.md` (the safe-mode preconditions the frontend enforces),
  `dev/archive/regularization_design.md` (the full regularization build — the D1/D2/D3 decisions, the
  `v_X`/`value` derivations verified this section summarizes, the uniform mask-safety spec, and the S4
  Option-A regime exploration).
- The polynomial/derivative fitting studies (MC-SGD minibatch findings, the symmetric-fit "halo")
  live in a separate research repo (maintainer-local); the user doc's §5 carries their practical
  takeaways.
