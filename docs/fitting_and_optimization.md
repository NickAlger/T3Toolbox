# Fitting & optimization — architecture, usage, and design rationale

How T3Toolbox fits a fixed-rank Tucker tensor train to **sampled** data — applies, entries, probes, and
their symmetric directional derivatives — by minimizing the least-squares misfit `½‖S(x) − y‖²`. This is
the reference for the *structure* (what the pieces are), the *usage* (how to drive it), and the *design
decisions* (what we chose and **why**). It complements the per-feature plans
([`dev/archive/optimizers_plan.md`](https://github.com/NickAlger/T3Toolbox/blob/main/dev/archive/optimizers_plan.md), [`dev/archive/derivative_fitting_plan.md`](https://github.com/NickAlger/T3Toolbox/blob/main/dev/archive/derivative_fitting_plan.md),
[`dev/archive/geometry_refactor_plan.md`](https://github.com/NickAlger/T3Toolbox/blob/main/dev/archive/geometry_refactor_plan.md)) — read this first for the whole picture.

---

## 1. The one idea: a fit factors into four orthogonal axes

Every fit is a choice along **four independent axes**, and the code is structured so each varies without
touching the others — `4 optimizers × 2 geometries × N sampling kinds × any minibatch` with **no
combinatorial code**:

| axis | what it answers | the object |
|---|---|---|
| **kind** | *what* you measure | `SamplingKind` — the bare Jacobian `𝒥`, its transpose `𝒥ᵀ`, the point op `S(x)`, the `‖·‖²` reduction |
| **geometry** | *where* you optimize | `Geometry` — the frame `frame(x)`, the gauge projector `Π`, the `retract` |
| **draw** | *how* you subsample | a `draw(rng) → (sample_B, data_B)` function |
| **optimizer** | *how* you step | `gradient_descent` / `mc_sgd` / `adam` / `newton_cg` |

The Gauss-Newton model that the optimizers consume — `m(p) = c + gᵀp + ½ pᵀ(JᵀJ)p` — is built by composing
the **kind** (the bare `𝒥`/`𝒥ᵀ`) with the **geometry** (the gauge `Π`): the Riemannian forward is
`J = 𝒥∘Π` and the gradient is `Jᵀr = Π∘𝒥ᵀr`. Neither axis knows about the other.

---

## 2. How to use it

There are three entry points, from highest-level to lowest.

### 2.1 Drive a library optimizer (the common case)

```python
import t3toolbox.manifold as t3m
import t3toolbox.optimizers as topt

# fit from applies, on the fixed-rank manifold, by Newton-CG (zero start is fine on the manifold):
x_opt, stats = topt.newton_cg(t3m.MANIFOLD, 'apply', ww, data, x0)

# fit from probes, on the raw cores (corewise), by Adam:
x_opt, stats = topt.adam(t3m.COREWISE, 'probe', ww, data, x0, rng, batch)

# fit from apply-DERIVATIVE jets, by Manifold Cauchy SGD, with a per-order weight and a custom minibatch:
x_opt, stats = topt.mc_sgd(t3m.MANIFOLD, 'apply_derivatives', (ww, pp), data, x0,
                           rng, batch, order=K, weight=ω, draw=my_draw)
```

- **Optimizers:** `gradient_descent` (Cauchy + Armijo), `mc_sgd` (Manifold Cauchy SGD — tuning-free
  stochastic), `adam` (corewise first-order), `newton_cg` (inexact Riemannian, 2nd-order). All return
  `(x_opt: TuckerTensorTrain, stats: dict)`.
- **Kind** is a string: `'apply'` / `'entries'` / `'probe'`, or `'apply_derivatives'` /
  `'entries_derivatives'` / `'probe_derivatives'`.
- **`sample`** is the measurement spec: `ww` (apply/probe — a `len=d` list of `W+(Nᵢ,)` vectors), `index`
  (entries — `(d,)+W` integer grid), or the paired `(ww, pp)` / `(index, pp)` for derivatives (`pp` = the
  perturbation directions).
- **`data`** is the observed `y` — **raw** (the kind applies any weighting internally).
- **Derivative kinds** take keyword `order` (highest derivative order) and optional `weight` (the per-order
  residual weight `ω`, length `order+1`; see §4.6). `mc_sgd`/`adam` also take `draw` (§2.3).

### 2.2 Build the Gauss-Newton model (roll your own optimizer)

If you want to write your own iteration (a custom trust region, a different line search), grab the model:

```python
import t3toolbox.fitting as fitting

model = fitting.apply_model(t3m.MANIFOLD, x, ww, residual)     # residual r = apply(x) − y
g  = model.gradient                       # Π 𝒥ᵀr   (a gauged T3Tangent)
Hv = model.gn_hessian(v)                  # Π 𝒥ᵀ𝒥Π v (the GN normal operator; symmetric PSD)
q  = model.gn_quadratic(v)                # ‖Jv‖²   (ONE forward — the cheap Cauchy/line-search denominator)
α  = g.corewise_inner(g) / q
x_new = t3m.MANIFOLD.retract(-α * g)
```

Factories: `apply_model` / `entries_model` / `probe_model`, and `apply_derivatives_model` /
`entries_derivatives_model` / `probe_derivatives_model` (which additionally take `order` + optional
`weight`). The model is generic over the geometry — pass `t3m.MANIFOLD` or `t3m.COREWISE`. Everything in
and out is a `T3Tangent` at `model.frame`. The frame sweep is computed once and reused across every
`gradient`/`gn_hessian`/`evaluate` (so an inner CG pays for it once, not per matvec).

### 2.3 Custom minibatching — the `draw`

Stochastic optimizers (`mc_sgd`, `adam`) draw a fresh minibatch each step. You can hand them **any**
function that returns a random sub-batch of the measurements:

```python
def my_draw(rng):
    idx = rng.choice(n_x, size=batch, replace=False)   # e.g. slice base points X
    return (sample_B, data_B)                           # the restricted (sample, data)

x, stats = topt.mc_sgd(..., draw=my_draw)
```

`draw(rng) → (sample_B, data_B)` returns the subset of measurement vectors **and** their measured values.
*Every* slicing scheme — flat random pairs, slice-on-X, slice-on-P — is just a different index expression
*you* write, on *your* arrays. **If you don't pass one, you get the flat default** (`flat_draw`: a uniform
random subset across the whole flattened sample stack `W`). The optimizer never compiles the draw, so it's
unconstrained Python/numpy — or write it in jax on device-resident data to keep the minibatch on the GPU
(only the per-step *kernel* is jitted; see §4.5).

### 2.4 Choosing a geometry

| | **`MANIFOLD`** | **`COREWISE`** |
|---|---|---|
| optimizes | on the fixed-rank manifold (the gauge `Π` makes `g`/`H` Riemannian) | the raw cores `(U,G,G,G)`, Euclidean metric |
| retraction | implicit truncated T3-SVD | additive (`cores += step`) |
| start `x0` | the **zero** tensor (orthonormal frame completion makes `J≠0`) | **nonzero** small random (zero cores ⇒ `J=0`) |
| pairs with | `newton_cg`, `mc_sgd` (the gauged `H` is PD) | `adam`, scipy/optax bridges (the corewise `H` is gauge-singular) |

The geometry is **structural, never a flag** — `MANIFOLD ⟺ Π`, `COREWISE ⟺ no Π` is bundled in the
geometry object. Mixing them silently corrupts the result, so they cannot be mixed by accident.

---

## 3. The pieces (structure)

```
        t3toolbox/optimizers.py   (frontend adapter: kind-string + order/weight/draw -> TuckerTensorTrain)
        t3toolbox/fitting.py      (frontend: GaussNewtonModel + *_model factories; T3Tangent in/out)
   ─────────────────────────────  the backend razor: a raw-.data user runs the SAME check-free code ──────
        backend/optimizers.py     (algorithms; Problem/LocalModel oracle; GeometryOps; flat_draw)
        backend/fitting.py        (SamplingKind: APPLY/ENTRIES/PROBE + *_derivatives_kind; sumsq helpers)
        backend/probing.py        (the bare 𝒥/𝒥ᵀ for apply/entries/probe + frame-sweep reuse hooks)
        backend/sampling_derivatives.py  (the derivative 𝒥/𝒥ᵀ + frame-sweep-jets reuse hooks)
        manifold.py               (MANIFOLD/COREWISE geometries; T3Tangent; retract/project/frame)
```

- **`SamplingKind`** (`backend/fitting.py`) — bundles the kind-specific functions the GN model needs:
  `precompute` (the reusable frame sweep), `forward` (`𝒥v`), `transpose` (`𝒥ᵀr`, summed over `W`), `sumsq`
  (the `‖·‖²` reduction), `w_axes`, plus `point_forward` (`S(x)`, for the residual) and the minimal layout
  for the default draw (`n_measurements`, `take`). It carries **no gauge** — that's the geometry's.
  Singletons `APPLY`/`ENTRIES`/`PROBE`; parameterized constructors `*_derivatives_kind(order, weight)`.
- **`Geometry`** (`manifold.py` `MANIFOLD`/`COREWISE`; backend `GeometryOps`) — `frame(x)` (the frame),
  `project` (the gauge `Π`), `retract`, plus the Hilbert-Schmidt `inner`/`norm`.
- **`Problem` + `LocalModel`** (`backend/optimizers.py`) — the backend oracle. `Problem(geom, kind, sample,
  data)` is **layout-agnostic**: `local_model(x [, sample_B, data_B])` linearizes at a point on the full
  data or an explicit minibatch, returning a `LocalModel` with `.gradient` / `.objective` / `.hvp` /
  `.gn_quadratic` / `.retract`. The frame sweep is computed once and shared. **`GaussNewtonModel`**
  (`fitting.py`) is its interactive frontend twin (`T3Tangent` in/out), verified bit-identical.
- **The optimizers** (`backend/optimizers.py`) consume the oracle hooks + (for the stochastic ones) a
  `draw`. `flat_draw(problem, batch)` builds the default minibatch draw.

---

## 4. Design decisions — and why

### 4.1 Backend-first (the razor)

The **algorithms live in the backend** (`backend/optimizers.py`), operating on raw cores / tangent tuples
via backend functions only, **free of the numerical safety preconditions** (which live in the frontend).
The frontend `optimizers.py` is a thin validate-once adapter. **Why:** an important minority of users
bypass the OO frontend and work on raw `.data` tuples — they must be able to run the *same* optimizer code.
It also means `jit` just works (no `unsafe()` wrapping — the backend has no checks to skip).

### 4.2 Geometry as a bundle, not a flag

`MANIFOLD ⟺ gauge Π`, `COREWISE ⟺ no Π` is **structural** — bundled in the geometry object, never a boolean.
**Why:** the gauge projection is not an option you toggle; it *defines* whether you're optimizing on the
manifold or the raw cores. A flag invites mixing the gauged gradient with the ungauged Hessian, which
silently corrupts the result. Bundling makes that unrepresentable. (This is the geometry refactor's core
move: one geometry-generic `GaussNewtonModel`/`Problem`, so optimizers are written *once, not per
geometry*.)

### 4.3 Frame-sweep reuse (the `precompute` / `*_from_sweep` split)

The Jacobian's expensive, `W`-scaling part — the **frame edge variables** (`xi`/`mu`/`nu`/`eta`, and their
order-jets for derivatives) — depends only on the frame frame + the sample vectors, **not** on the tangent
direction or the residual. So it is computed **once per frame** (`SamplingKind.precompute`) and reused by
every `J` / `Jᵀ`. **Why:** an inner CG solve fixes the frame across many matvecs; recomputing the sweep each
matvec would dominate. Per-kind it is **lean or full**: apply/entries need only `(xi, mu)`, probe needs all
four (§4.7).

### 4.4 Minibatching as a user-supplied `draw` (not baked in)

Minibatching is *not* a library policy — it's a function the user hands the optimizer. **Why:** a
stochastic step needs, each iteration, a random *sub-problem* (a subset of the measurements + their data),
and the most flexible way to produce one is to let the user index their own arrays. Slice-on-X,
slice-on-P, flat random pairs — all are one-line index expressions the user controls; the library needs
**zero** knowledge of their data layout. The `Problem` is correspondingly **layout-agnostic** (no
`kind.name` dispatch). The kind retains only the *minimal* layout (`n_measurements`/`take`) needed to build
the **default** flat draw; a custom draw bypasses it. (Order-slicing is the one exception — order is an
*output-only* axis, the forward computes the whole jet jointly, so subsetting orders is output-*masking*
not input-*slicing*; deferred, and most naturally an outer continuation loop anyway.)

### 4.5 jit composes for free; the draw stays on the host (or device, your choice)

The numpy/jax dispatch is inferred from the input array types at the lowest level (no `use_jax` threading),
so a single dispatch-written kernel runs numpy-eager, jax-eager, or jit-compiled. **`use_jit`** is a thin
jax-only layer on top: it jits the per-step kernel (gradient → step → retract; or the inner CG loop as one
`lax.while_loop` via `common.xwhile`). **The draw runs *outside* the compiled kernel** — its fixed-size
minibatch arrays flow in as the kernel's inputs, so the kernel compiles **once** (constant shapes) and is
reused. **Why this split:** it keeps the user's draw unconstrained (any numpy/jax indexing, never compiled)
while the heavy kernel is jitted. For GPU scale: keep the data resident on device, write the draw in jax →
the minibatch is produced on-device and the kernel consumes it on-device, with no per-step host↔device
transfer (only a random key crosses).

### 4.6 Normalization is a per-order residual *weight* `ω`, owned by the kind

Fitting from derivatives, the orders span many decades (the order-`t` term carries a `t!`/binomial weight),
which wrecks the Gauss-Newton conditioning. The fix is a **per-order residual weight** `ω`: the objective is
`½‖ω ⊙ (S(x) − y)‖²`, so `ω` enters **only** `sumsq` (×ω) and `transpose` (×ω², the gradient `𝒥ᵀ(ω²r)`),
while `forward` / `point_forward` / `data` stay **raw**. **Why not fold `1/ω` into the forward + pre-
normalize the data?** Because then a user who writes a custom `draw` returning *raw* `data_B` (the natural
thing) would silently break the residual — a nasty footgun for exactly the power-user audience. Centralizing
`ω` in the kind means the user passes **raw data + a weight vector** and a custom draw has nothing to
remember. `ω` is *created outside the optimization* (the user's choice — per-order RMS, a physical length
scale, …; default `ω=1`).

### 4.7 Low-memory transpose: adjoint-state over scatter (store-vs-recompute)

The gradient `𝒥ᵀr` for apply/entries needs a **right context** beyond the forward's left sweep. Two ways:
the **scatter** *stores* the full frame sweep `(xi,mu,nu,eta)` (cheap matvec, more memory), or the
**adjoint-state** method *recomputes* the right context as a seeded `sigma_hat` reverse sweep (stores only
`(xi,mu)`, costs a sweep per matvec). T3Toolbox uses **adjoint-state** — **exactly 2× less `W`-scaling
memory**. **Why:** at real scale the `W`-batched edge variables get large; memory is the binding constraint
on a 40GB GPU, *worst* for minibatched Newton-CG (smaller batches → more ill-conditioned `H` → wants more
data → larger `W`). It is the classic **checkpointing** tradeoff, and the project prefers low memory (see
the `prefer-low-memory-over-compute` memory). *Probe can't use it* — its residual is a vector (one free
mode), which must be propagated by the full adjoint sweeps + `nu`/`eta`; the scalar-seed shortcut is
apply/entries-only. This is why probe's precompute is full and apply/entries' is lean.

### 4.8 numpy einsum: force BLAS-eligible pairwise paths; jax: one big einsum

The grouped contractions (`backend/contractions.py`) route through `_grouped_einsum`. **numpy:** a forced
greedy-pairwise path — because numpy's `optimize=True` minimizes FLOP *count*, and on a FLOP-tie runs a
single multi-operand contraction as one `c_einsum` loop **with no BLAS** (10–55× slower for the high-
dimensional order-combines). **jax:** one big einsum — XLA's opt_einsum + fusion is BLAS-aware and *beats*
any path we force. **Why it matters here:** the derivative forward/transpose are dominated by those
order-combines; the fix is 11–19× on them, numerically identical.

---

## 5. What's deferred (not built)

- **The Goal-1 `fit(...)` facade** — a "just fit my tensor" entry point that picks a sensible geometry +
  optimizer, supplies the geometry-correct `x0`, and runs **rank continuation** with validation. The
  current layer is a clean *mid-level toolkit*; the facade is what delivers "standard user, no fiddling".
  Rank continuation + validation currently live in the examples (the right defaults: manifold → zero start
  + warm continuation; corewise → nonzero start + cold per level — see `dev/archive/optimizers_plan.md` §7).
- **Order-slicing minibatches** (output-masking) and order/polynomial-degree continuation — research,
  likely outer loops.
- **The example pass** — deciding which `examples/fit_hilbert_*` use the library optimizers vs keep inline
  to illustrate the hidden hooks (`gn_hessian`, `gn_quadratic`, `corewise_map`); `dev/archive/optimizers_plan.md` §10.
- **Per-sample gradients / multi-source fits** (SVRG-style; fitting from applies *and* entries together) —
  reachable at the backend level (`sum_over_probes=False`; sum two local models), not packaged.

---

## 6. Pointers

- **Examples:** `examples/fit_hilbert_tensor_newton_cg.py` (apply, manifold, Newton-CG — *inline*, shows
  `gn_hessian`); `fit_hilbert_from_probes_adam.py` / `_optax.py` (probes, corewise, hand-Adam / optax);
  `fit_hilbert_from_entries_lbfgs.py` (entries, corewise, scipy bridge); `fit_hilbert_from_apply_
  derivatives.py` / `_flat.py` (apply-derivatives, *inline* MC-SGD); `fit_hilbert_from_apply_derivatives_
  topt.py` (the **library** apply-derivatives pilot).
- **Research study (branch `polynomial_fitting_experiments`):** `experiments/` recovers a polynomial from
  function + derivative samples via the `apply_derivatives` fit (`symmetric_polynomial_fitting.tex` is the
  writeup). **Practical takeaways:** on ill-conditioned high-rank symmetric fits, **prefer `newton_cg`** —
  MC-SGD's first-order convergence is too slow and its under-converged iterate is *tilted toward the
  function-space metric* (good function error, poor Frobenius / poor tensor recovery), whereas Newton-CG
  recovers the true tensor (balanced across norms). **Rank continuation alone** (constant seed +
  rank-1-first) suffices; order continuation adds nothing. When the data only constrains a *symmetric*
  (or otherwise structured) subspace, the fit fills the unconstrained null space with a large "halo" —
  **project/symmetrize the fit** to read off the meaningful part.
- **Plans:** `dev/archive/optimizers_plan.md` (the optimizers + example two-track plan), `dev/archive/derivative_fitting_plan.md`
  (the D1–D4 derivative-fitting build), `dev/archive/geometry_refactor_plan.md` (the geometry abstraction).
- **Adjacent:** `entries_apply_probe.md` (the three sampling ops + their transposes), `transposes.md`
  (ambient/corewise/tangent taxonomy), `batching_and_stacking.md` (the `W`/`K`/`C` stack design — read
  before touching anything with stack axes), `mcsgd_apply_derivatives.md` (the Cauchy step + minibatch
  findings), `dev/archive/safe_unsafe_mode_plan.md` (the safe-mode preconditions the frontend enforces).
