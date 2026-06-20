# Optimizers (G3) — design plan

*Design locked with Nick (2026-06-20), branch `geometry-refactor`. This is slice **G3** of
[`geometry_refactor_plan.md`](geometry_refactor_plan.md) (G1 geometries + G2 generic `GaussNewtonModel`
are done). It refines that plan's §4 sketch — what it drew as a single `optimizers.py(geometry,
model_builder, x0)` splits into a **backend algorithm + a thin frontend adapter**. The design is grounded
in the example-building study: [`entries_completion_findings.md`](entries_completion_findings.md) (the
optimizer bake-off, the geometry-aware init/continuation), [`mcsgd_apply_derivatives.md`](mcsgd_apply_derivatives.md)
(the Cauchy step), and the safe-mode mechanism ([`safe_unsafe_mode_plan.md`](safe_unsafe_mode_plan.md),
[`numerical_contract_catalog.md`](numerical_contract_catalog.md)).*

## 0. Resuming from a fresh context (read this first)

This document is self-contained. The design is decided — do not re-derive it. Build §6 in order;
confirm small choices with Nick, but build. The empirical grounding for every choice is in the findings
docs above (notably the rank-1..10 optimizer bake-off on probe data).

## 1. Decisions (locked)

1. **`use_jit` flag, default `False`.** jax-only; on the numpy path (or if jax is not installed) it
   **silently falls back to eager** — *not* a hard error, so code is portable across machines /
   collaborators with different environments without twiddling the flag. `True` + jax inputs compiles the
   per-iteration numerical kernel.
2. **No safe/unsafe-mode management in the optimizers — it dissolves.** The numerical preconditions
   (orthogonal frame, gauged, same-frame) live **only in the frontend**; `backend/` functions are
   check-free of them. The optimizer **algorithm runs in the backend**, so there is nothing to bypass:
   `jit` just works, no `unsafe()` wrapping, no trace-skip subtlety. (See §4.)
3. **Newton-CG jits the whole inner CG loop** via `jax.lax.while_loop` (not merely the Hessian matvec).
4. **The razor: optimizer algorithms go in `backend/optimizers.py`.** They are non-trivial and easy to
   get wrong, so a raw-`.data` user must be able to call them. The frontend `optimizers.py` is a thin
   adapter. (See §3.)
5. **Library optimizers: `gradient_descent`, `adam`, `mc_sgd`, `newton_cg`.** **L-BFGS is *not* a library
   optimizer** — it stays the scipy-bridge example (the own-vs-external decision: the value is scipy's
   battle-tested Wolfe line search). A *Riemannian* L-BFGS (only we could write) is deferred.

## 2. Structural vs numerical checks (the precise line)

Per the house philosophy, and Nick's clarification for the jit context:

- **Structural / static checks** — shape/rank/length consistency. **Always on, both modes, and jit-safe**
  (shapes are concrete at trace time, so `if mismatch: raise` works under `jit`). Keep them wherever they
  are. *Example: "do the input arrays have consistent shapes?" — fine in safe mode and under jit.*
- **Numerical checks** — orthogonal frame, gauged variations, same-frame, etc. **Frontend-only safe-mode
  preconditions**, skipped under unsafe/jit. *Example: "is this matrix orthogonal?" — not run in the hot
  loop.* These are the checks the backend simply does not have.

So "the backend optimizer is check-free" means **free of numerical checks**; structural guards may remain
(they are harmless and jit-safe). The frontend adapter does the one upfront numerical validation of the
*input* (§4).

## 3. Architecture (backend-first, frontend adapter)

```
backend/optimizers.py     the ALGORITHMS — on raw cores + a backend-built "problem oracle"
    gradient_descent(problem, x0_cores, *, use_jit=False, ...)
    adam(problem, x0_cores, ...)            mc_sgd(problem, x0_cores, ...)
    newton_cg(problem, x0_cores, ...)       # jit-the-CG-loop (lax.while_loop)
  · tangent vectors are raw (tucker_var, tt_var) tuples; vector arithmetic via the corewise
    backend ops (add / scale / inner / zeros) — fixed, not injected
  · use_jit compiles the per-iteration kernel; NO numerical-safety logic at all
  · NO frontend imports (no T3Tangent / GaussNewtonModel / geometry classes)

optimizers.py (frontend)  the thin adapter
    newton_cg(geometry, model_builder, x0)  (+ adam / mc_sgd / gradient_descent)
  · validate the input ONCE (structural always; numerical in safe mode)
  · build the backend `problem` oracle from (geometry, model_builder) using BACKEND functions
    (backend/fitting.py + the geometry's backend ops) -- NOT by wrapping model.gradient /
    geometry.retract (those carry the checks)
  · call the backend optimizer; re-wrap the result cores as a TuckerTensorTrain
```

The crucial constraint (and it is the *correct* layering): the oracle is assembled from **backend**
functions, so the backend user and the frontend user run **identical, check-free** code — the only
difference is who assembles the oracle and whether the input got the upfront validation. The frontend
`GaussNewtonModel` stays a purely *interactive* convenience; the optimizer reuses `backend/fitting.py`
directly.

## 4. The problem oracle (what the backend optimizer consumes)

A small, geometry/sampling-agnostic bundle of callables on **raw cores / tangent tuples**:

```
problem.n_samples                              # int -- for minibatch draws (deterministic: full set)
problem.local_model(x_cores, sample_idx=None)  # idx=None -> full batch; builds residual + base sweep ONCE
    -> local with:
         .gradient                  # raw tangent tuple  = Π 𝒥ᵀ r   (Π=gauge for manifold, identity for corewise)
         .objective                 # scalar             = ½‖r‖²
         .hvp(p_tangent)            # raw tangent tuple  = Π 𝒥ᵀ 𝒥 Π p   (Newton-CG; reuses the sweep)
         .gn_quadratic(p_tangent)   # scalar             = ‖𝒥 Π p‖²      (Cauchy step / line-search curvature)
problem.retract(x_cores, p_tangent)            # -> x_cores  (geometry retraction; rebuilds the frame internally)
```

- `local_model` is the **backend** form of `GaussNewtonModel` (built from `backend/fitting.py`); the
  sweep is computed once and shared across `.gradient` / `.hvp` / `.gn_quadratic` (the reuse the fitting
  layer exists for).
- `retract` rebuilds the frame from `x_cores` internally, so the backend optimizer never sees a
  `T3Basis` — it passes only cores and tangent tuples.
- **Vector ops are fixed, not part of the oracle:** tangent data is *always* `(tucker, tt)` tuples
  regardless of geometry, so the optimizer uses the corewise backend ops directly. Existing:
  `corewise.corewise_add`, `corewise_dot`/`corewise_norm`. **To add:** tiny `corewise_scale(c, a)` and
  `corewise_zeros_like(a)` helpers (trivial-but-named, so a backend user finds them).
- **Stochastic vs deterministic** is just `sample_idx`: stochastic optimizers draw a minibatch
  (`rng.choice(n_samples)`, host) and pass it; deterministic optimizers pass `None`. *What a "sample" is*
  (a probe, an entry, an `(X,P)` pair) is the user's choice in how the oracle slices — the library only
  draws indices (per the flat-vs-X-only finding: policy stays out of the library).

## 5. The optimizers

All four consume `(problem, x0_cores)`, return optimized `x_cores` + a small stats dict. Driven purely by
the oracle hooks — confirmed sufficient by the example study.

- **`gradient_descent`** — `x ← retract(x, −α·g)`; fixed or backtracked `α`. Simplest; validates the stack.
- **`adam`** — per-coordinate moment EMAs + bias correction over the tangent tuples (the pytree map from
  the probe example, generalized; optional schedule). Corewise's dependency-free first-order method.
- **`mc_sgd`** — Manifold Cauchy SGD: minibatch, `g = problem.local_model(x, idx).gradient`, Cauchy step
  `α = ‖g‖² / gn_quadratic(g)`, `x ← retract(x, −α·g)`. Stopping: smoothed full-batch loss, **absolute-
  iteration window** (the G3 fix — decouples from batch size; the epoch-based window made batch=3 fragile,
  see `mcsgd_apply_derivatives.md`). The tuning-free workhorse; matched Newton-CG on probes.
- **`newton_cg`** — outer Newton loop (host, ~30 iters): build `local_model` at `x`; solve `H p = −g` by
  **CG jit-compiled as one `lax.while_loop`** (state = the tangent tuples + `rs`; body = `hvp` + CG
  update; cond = `‖res‖ > η·‖g‖ ∧ i < maxiter`, the inexact forcing term); Armijo backtracking line search
  on the host (a few `retract`+`objective` evals, each jit-able). `use_jit` compiles the CG solve (once
  per shape). Tolerates the **gauge-singular corewise `H`** via the existing nonpositive-curvature guard
  (truncated CG); a geometry may advertise `hessian_is_degenerate` to steer corewise users to first-order.

## 6. numpy/jax dispatch × jit (how `use_jit` composes with the dispatch model)

The library's dispatch model — each op infers `use_jax = tree_contains_jax(inputs)` then
`xnp, xmap, xscan = get_backend(...)`, jax calls guarded by `has_jax()` (reference: `backend/probing.py`)
— **composes with jit for free**: under a trace the inputs are tracers (which *are* jax arrays), so
`tree_contains_jax → True → jnp`, and that dispatch is a **trace-time decision (once)**. One
dispatch-written kernel therefore runs all three ways with no special-casing:

| inputs | `use_jit` | path |
|---|---|---|
| numpy | False | numpy eager — the numpy-only install (no jax needed) |
| jax | False | jnp eager (correct, slow) |
| jax | True | jnp, jit-compiled (fast) |

So `use_jit` is a thin **jax-only layer on top of** the dispatch model. Rules:

- **`backend/optimizers.py` imports no jax at top level** (numpy-only safe); `jax.jit` / `lax.while_loop`
  live only inside `has_jax()`-guarded branches. `use_jit=True` with no jax / numpy inputs **silently runs
  eager** (decision 1).
- **The one real interaction is data-dependent control flow** — `newton_cg`'s inner CG `while` (iterate
  until the forcing-term tolerance). Eager it is a Python `while` with a host predicate (`float(‖res‖) >
  tol`); under jit it must be `lax.while_loop` with a *traced* predicate. The **loop body is shared**
  (dispatch-written, pure arithmetic); only the **driver** differs. This is the `xscan` precedent for a
  `while`, factored as a new **`common.xwhile(cond, body, state, use_jit)`** helper next to `get_backend`,
  so a backend user writing their own iterative solver gets the same numpy / eager-jax / jit treatment.
- **The stochastic optimizers (`mc_sgd`/`adam`) don't hit this** — their loop is a host Python loop
  (host-RNG minibatch draws); `use_jit` jits only the per-step kernel (gradient + update) called inside it
  (the optax-example shape). No `lax` loop.
- Vector ops and the oracle are dispatch-written, so they already work numpy/jax; minibatch indices stay
  host numpy.

## 7. Geometry-aware init / continuation (a helper, not the optimizer's job)

The optimizers fit at **fixed rank** from a given `x0`. Rank continuation + the start policy is a separate
concern — an optional `rank_continuation(optimizer, geometry, levels, ...)` driver, or left to the user.
The empirically-found defaults (record them in docs + the helper, not hard-code in the optimizer):

| geometry | start | continuation |
|---|---|---|
| **manifold** | the **zero** tensor (orthonormal frame completion makes `J≠0`) | **warm** (zero-pad the converged previous level) — robust, clean |
| **corewise** | **nonzero** small random, rescaled (zero cores ⇒ `J=0`) | **cold** per level (zero-pad freezes the new block at a vanishing-Jacobian saddle) |

Model selection: **held-out validation** picks the rank (overfitting severity is data-source-set, so
validation is essential and works off a gentle turnover for well-conditioned sources).

## 8. Build plan (slices)

1. **G3.1 — oracle + vector helpers + `gradient_descent`, end-to-end. ✅ DONE.** The `Problem` /
   `LocalModel` / `GeometryOps` oracle (both geometries) + `corewise_zeros_like` + backend
   `gradient_descent` (Cauchy + Armijo) + the frontend adapter. Backend oracle == `GaussNewtonModel`
   bit-identical across both geometries × all three kinds.
2. **G3.2 — `mc_sgd` + `adam`. ✅ DONE.** Minibatch on the oracle (kind-aware slicing + `n_samples`);
   `mc_sgd` (Cauchy step + absolute-iteration stopping window); `adam` (moments via `corewise_map`,
   cosine schedule). Both recover the tensor (backend + adapter).
3. **G3.3 — `newton_cg` + `common.xwhile` + the jit `lax.while_loop` CG. ✅ DONE.** `xwhile` driver;
   `_cg_solve` with a branch-free `xnp.where` curvature guard (truncated CG); host Armijo line search.
   Recovers to <1e-4 eager and **2.9e-7 via the real `lax.while_loop` CG** with jax inputs.
4. **G3.4 — `use_jit` across the first-order optimizers + jit tests. ✅ DONE.** `mc_sgd`/`adam` jit their
   per-step kernel (`_maybe_jit`; silent eager fallback unless x0/sample/data are all jax); jit dispatch
   test confirms `newton_cg`/`mc_sgd`/`adam` compile + recover with jax inputs (no hidden numpy).
   `gradient_descent` stays eager (host Armijo loop). Tests: `tests/backend/test_optimizers.py`,
   `tests/test_optimizers_frontend.py` (9 tests, 17 subtests).
5. **G3.5 — examples + docs (IN PROGRESS).** Re-point the inline-optimizer examples at the library
   optimizers where it sharpens them (confirm they reproduce the inline results); keep the bridge examples
   (scipy L-BFGS, optax) as integration demos. Refresh `geometry_refactor_plan.md` /
   `entries_apply_probe.md`.

## 9. Open questions / deferred

- **Rank-continuation helper scope** — a library helper vs leave to the user/examples. Decide in G3.5
  once the optimizers exist.
- **jit the Armijo line search** (vs host) — host first; revisit if the per-step retract/objective sync
  dominates.
- **`hessian_is_degenerate` geometry hint** — expose to steer corewise users away from `newton_cg`, or
  just document. Decide in G3.3.
- **Riemannian L-BFGS** (own, with vector transport) — deferred; the only quasi-Newton only we could
  write (scipy/optax can't do the manifold). Revisit if an example demands it.
- **Stopping criteria as first-class** — `mc_sgd`'s smoothed-loss test, Newton-CG's gradient-norm test;
  whether to factor a shared "convergence policy" object. Likely premature; inline per optimizer for now.
