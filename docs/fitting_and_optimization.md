# Fitting & optimization

How T3Toolbox fits a fixed-rank Tucker tensor train to **sampled** data — applies, entries, probes,
and their symmetric directional derivatives — by minimizing the least-squares misfit `½‖S(x) − y‖²`.
This is the user reference: the structure (what the pieces are), the usage (how to drive it), and
the design properties you can rely on. (The implementation-side rationale — memory tradeoffs,
einsum paths, deferred work — is [`contributor/fitting_internals.md`](contributor/fitting_internals.md).)

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

# fit from probes with a PER-MODE weight (discount a noisier mode; probe-only, see §4.6):
x_opt, stats = topt.newton_cg(t3m.MANIFOLD, 'probe', ww, data, x0, weight=ω_mode)   # ω_mode shape (d,)
```

- **Optimizers:** `gradient_descent` (Cauchy + Armijo), `mc_sgd` (Manifold Cauchy SGD — tuning-free
  stochastic), `adam` (corewise first-order), `newton_cg` (inexact Riemannian, 2nd-order). All return
  `(x_opt: TuckerTensorTrain, stats: dict)`.
- **Live diagnostics (`newton_cg`):** `newton_cg(..., verbose=True)` prints a per-iteration block —
  objective / gradient norm, CG iterations / tolerance / status, line search, the actual-vs-predicted
  reduction `ρ`, and a per-`(mode, order)` **relative-error table** (`‖S(x)_ij − y_ij‖/‖y_ij‖`); pass
  `val_sample`/`val_data` to add a validation column. The records are also returned in
  `stats['diagnostics']`. The table layout follows the kind's axes — plain probe has modes in columns,
  `probe_derivatives` has modes × orders. Worked demo (both layouts): `examples/fit_probe_display.py`.
  Backend users get the identical display via `backend.optimizer_display.make_newton_display` +
  `backend.optimizers.newton_cg(..., callback=...)`.
- **Kind** is a string: `'apply'` / `'entries'` / `'probe'`, or `'apply_derivatives'` /
  `'entries_derivatives'` / `'probe_derivatives'`.
- **`sample`** is the measurement spec: `ww` (apply/probe — a `len=d` list of `W+(Nᵢ,)` vectors), `index`
  (entries — `(d,)+W` integer grid), or the paired `(ww, pp)` / `(index, pp)` for derivatives (`pp` = the
  perturbation directions).
- **`data`** is the observed `y` — **raw** (the kind applies any weighting internally).
- **`weight`** (optional residual weight `ω`, see §4.6) is an `ω[mode, order]` matrix with broadcasting:
  **derivative** kinds take a per-order vector `(order+1,)`; **probe** additionally takes a per-**mode**
  weight — `probe` (plain) a 1-D `(d,)`, `probe_derivatives` the full `(d, order+1)` matrix. apply/entries
  contract every mode into a scalar, so they have no mode axis (mode weighting is probe-only).
- **Derivative kinds** take keyword `order` (highest derivative order). `mc_sgd`/`adam` also take `draw`
  (§2.3).

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

Factories: `apply_model` / `entries_model` / `probe_model` (the last takes an optional per-mode
`weight`), and `apply_derivatives_model` / `entries_derivatives_model` / `probe_derivatives_model` (which
additionally take `order` + optional `weight`; see §4.6). The model is generic over the geometry — pass
`t3m.MANIFOLD` or `t3m.COREWISE`. Everything in
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

## 4. Design properties you can rely on

### 4.1 Geometry is a bundle, not a flag

`MANIFOLD ⟺ gauge Π`, `COREWISE ⟺ no Π` is **structural** — bundled in the geometry object, never a
boolean. The gauge projection is not an option you toggle; it *defines* whether you're optimizing on
the manifold or the raw cores. A flag would invite mixing the gauged gradient with the ungauged
Hessian, which silently corrupts the result; bundling makes that unrepresentable. One
geometry-generic model serves both, so every optimizer works with either geometry.

### 4.2 The Gauss-Newton model is *exact*, not an approximation

The sampling forwards (`apply`/`entries`/`probe` and their jets) are **linear in the ambient
tensor**, so the least-squares objective `φ(T) = ½‖A·T − y‖²` is *exactly quadratic*, and its
Gauss-Newton Hessian **is** the true ambient Hessian — the second-order term Gauss-Newton normally
drops is identically zero. The model `m(p)` is therefore the **exact** restriction of the objective
to the affine tangent space `x + T_xM`, not a second-order approximation. Two practical
consequences: the only approximation in a Newton-CG fit is the manifold linearization itself (the
model is trustworthy at any step size the retraction tolerates), and dense ground-truth oracles can
check the model to machine precision (~1e-13) — which is how the fitting layer is verified.

### 4.3 Frame-sweep reuse: inner iterations are cheap

The Jacobian's expensive, `W`-scaling part — the **frame edge variables** (`xi`/`mu`/`nu`/`eta`, and
their order-jets for derivatives) — depends only on the frame + the sample vectors, **not** on the
tangent direction or the residual. So it is computed **once per frame** (`SamplingKind.precompute`)
and reused by every `J`/`Jᵀ` (the `*_from_sweep` backend hooks). An inner CG solve fixes the frame
across many matvecs, so it pays for the sweep once, not per matvec. Per kind the stored sweep is
lean (apply/entries) or full (probe) — a deliberate memory tradeoff; the reasoning is in
[`contributor/fitting_internals.md`](contributor/fitting_internals.md).

### 4.4 Minibatching is your function, not library policy

A stochastic step needs, each iteration, a random *sub-problem* (a subset of the measurements +
their data), and the most flexible way to produce one is to let you index your own arrays.
Slice-on-X, slice-on-P, flat random pairs — all are one-line index expressions you control; the
library needs **zero** knowledge of your data layout. The `Problem` is correspondingly
layout-agnostic, and the kind retains only the minimal layout needed to build the **default** flat
draw; a custom draw bypasses it entirely.

### 4.5 jit composes for free; the draw stays outside the compiled kernel

The numpy/jax dispatch is inferred from the input array types at the lowest level (no `use_jax`
threading), so a single kernel runs numpy-eager, jax-eager, or jit-compiled. **`use_jit=True`** (on
`mc_sgd`/`adam`/`newton_cg`) jits the per-step kernel — gradient → step → retract, or the inner CG
loop as one `lax.while_loop`. Asking for jit is opting into "jax world": the optimizer
**auto-converts** the inputs (`x0`, `sample`, `data`) onto jax, so a `use_jit=True` call on numpy
data **returns a jax-backed result** in jax's default **float32** (enable `jax.config.update(
"jax_enable_x64", True)` first if you need float64), and **raises** if jax is not installed — the
request is honored or explained, never silently dropped. (Passing jax inputs yourself is equivalent;
the convert is a no-op.) **The draw runs *outside* the compiled kernel** — its fixed-size minibatch arrays flow in
as the kernel's inputs, so the kernel compiles **once** (constant shapes) and is reused. This keeps
your draw unconstrained (any numpy/jax indexing, never compiled) while the heavy kernel is jitted.
For GPU scale: keep the data resident on device and write the draw in jax → the minibatch is
produced on-device and consumed on-device, with no per-step host↔device transfer (only a random key
crosses).

### 4.6 The residual weight `ω` is a per-order / per-mode matrix, owned by the kind

The objective carries an optional **residual weight** `ω`: `½‖ω ⊙ (S(x) − y)‖²`, so `ω` enters **only**
the `‖·‖²` reduction and the gradient (`𝒥ᵀ ω²r`), while `forward` / `point_forward` / `data` stay
**raw**. You pass **raw data + a weight**, and a custom `draw` returning raw `data_B` (the natural thing)
can never silently break the residual — the alternative (fold `1/ω` into the forward and pre-normalize the
data) would be a footgun for exactly the power-user audience. `ω` is created outside the optimization
(your choice; default `ω = 1`). It is host-numpy static structure (like the uniform masks), so it folds
into the compiled program as a device constant.

`ω` is a matrix over two axes — **`ω[mode, order]`**, conceptual shape `(d, order+1)` — applied with
plain NumPy right-aligned broadcasting, so **order is the innermost (rightmost) axis**:

| you pass | shape | means |
|---|---|---|
| a bare vector | `(order+1,)` → `(1, order+1)` | **per-order** (broadcast over modes) |
| a column | `(d, 1)` | **per-mode** (broadcast over orders) |
| a matrix | `(d, order+1)` | an independent weight per `(mode, order)` |

**Per-order weighting** is the derivative-kind use case: the symmetric-derivative orders span many
decades (the order-`t` term carries a `t!`/binomial weight), which wrecks the Gauss-Newton conditioning;
a per-order `ω` (e.g. per-order RMS, a physical length scale) rebalances them. A bare vector always means
per-order — this is the backward-compatible reading.

**Per-mode weighting** is a **probe** feature: only probing produces a per-mode output (one vector per
mode), so only `probe_model` / `probe_derivatives_model` (and `topt` with `'probe'` /
`'probe_derivatives'`) accept a mode weight. It up- or down-weights each mode's data — invaluable when the
modes are measured at very different scales/precisions (e.g. the "forward"/"reverse" modes of a PDE
operator), where inverse-noise weighting `ω_i = 1/σ_i` is the Gauss–Markov/GLS estimate. See
[`examples/fit_per_mode_weight_probes.py`](https://github.com/NickAlger/T3Toolbox/blob/main/examples/fit_per_mode_weight_probes.py),
where it turns an order-`1e-3` recovery into `1e-4` from the same data.

apply/entries contract every mode into a scalar — they have **no mode axis**, so their weight is
order-only (a per-mode weight is a structural error). Plain `probe_model` has **no order axis**, so its
weight is a 1-D `(d,)` per-mode vector (a 2-D `(d, 1)` is rejected — this keeps `(d,)` the single,
forward-stable spelling if a less-important weighting axis is ever added).

```python
# probe_derivatives: down-weight the noisy high orders AND discount a coarse mode, independently:
mode_w, order_w = [1.0, 0.6, 0.3], [1.0, 0.5, 0.3, 0.2]                    # (d=3,) and (order+1=4,)
model = fitting.probe_derivatives_model(t3m.MANIFOLD, x, ww, pp, order=3, residual=r,
                                        weight=np.outer(mode_w, order_w))   # ω[mode,order], (3, 4)
# plain probe: a per-mode vector is 1-D (d,):
model = fitting.probe_model(t3m.MANIFOLD, x, ww, r, weight=[0.5, 1.0, 2.0])
```

### 4.7 The corewise gradient is hand-rolled — and outperforms autodiff

A natural question: why does the library hand-code the corewise (Euclidean) gradient instead of
just using `jax.grad`? Because the hand-rolled sweeping/probing contractions, jit-compiled,
**outperformed `jax.grad` for the corewise gradient in the maintainer's benchmarks — sometimes
substantially** (frame-sweep reuse plus no autodiff-graph overhead). As with all performance
statements in this library, treat it as regime-dependent rather than absolute — but corewise is not
a "just use autodiff" stand-in. It also gives you a matrix-free Gauss-Newton operator
(`JᵀJ`-vector products, awkward for autodiff via double-backprop) and an autodiff-free numpy path,
inside the same geometry framework as the manifold.

### 4.8 Where's L-BFGS?

Deliberately not a library optimizer. For a *Euclidean* (corewise) L-BFGS, the value is scipy's
battle-tested Wolfe line search — so the intended pattern is the **scipy bridge**
(`examples/fit_hilbert_from_entries_lbfgs.py`: the corewise gradient feeds
`scipy.optimize.minimize(method='L-BFGS-B')` directly). A *Riemannian* L-BFGS (with vector
transport — the quasi-Newton method only this library could provide) is deferred.

---

## 5. Practical guidance from the field

Distilled from fitting studies with the library (the studies themselves live in a separate research
repo, maintainer-local):

- **On ill-conditioned, high-rank fits, prefer `newton_cg`.** The SGD-family's first-order
  convergence is slow there, and an under-converged iterate is *tilted toward the function-space
  metric* — good function error, poor Frobenius error / poor tensor recovery — whereas Newton-CG
  recovers the true tensor, balanced across norms.
- **Rank continuation alone suffices** (constant seed, rank-1-first, warm-started by zero-padding —
  see [`rank_continuation.md`](rank_continuation.md)); order continuation added nothing in the
  studies.
- **In a warm-start continuation loop, pin the reference gradient norm.** `newton_cg`'s Newton stop
  (`‖g‖ ≤ gtol_rel·‖g0‖`) and CG forcing term (`η = min(0.5, (‖g‖/‖g0‖)**cg_forcing_power)`) are both
  *relative* to a reference `‖g0‖` that defaults to the **initial** gradient norm. After a warm start
  that norm is already small, so the Newton stop over-tightens and CG slackens. Pass `g0norm_newton` /
  `g0norm_cg` a reference reflecting the problem's true gradient scale (e.g. the initial `‖g‖` from the
  first continuation stage) — `g0norm_newton` also feeds CG unless `g0norm_cg` is given; `g0norm_cg`
  alone touches only CG. Separately, `cg_forcing_power` (default `0.5`) trades CG effort for Newton
  steps: a **larger** power (`0.75`, `1.0`) tightens CG per step — worth it on the manifold when the
  retraction is expensive relative to a Hessian-apply.
- **When the data only constrains a structured subspace** (e.g. symmetric tensors), the fit fills
  the unconstrained null space with a large "halo" — **project/symmetrize the fitted tensor** to
  read off the meaningful part.

---

## 6. Pointers

- **Examples:** `examples/fit_hilbert_tensor_newton_cg.py` (apply, manifold, Newton-CG — *inline*, shows
  `gn_hessian`); `fit_hilbert_from_probes_adam.py` / `_optax.py` (probes, corewise, hand-Adam / optax);
  `fit_hilbert_from_entries_lbfgs.py` (entries, corewise, scipy bridge); `fit_hilbert_from_apply_
  derivatives.py` / `_flat.py` (apply-derivatives, *inline* MC-SGD); `fit_hilbert_from_apply_derivatives_
  topt.py` (the **library** apply-derivatives pilot); `fit_hilbert_uniform_newton_cg.py` /
  `_probe_derivatives_newton_cg.py` (the uniform layer end-to-end); `fit_per_mode_weight_probes.py`
  (**per-mode residual weighting** — inverse-noise probe fit, §4.6); `fit_probe_display.py`
  (**`verbose=True` Newton-CG diagnostics** — both relative-error table layouts);
  `fit_hilbert_regularized.py` (**identity/Tikhonov regularization** — denoising a noisy Hilbert-tensor
  fit, with λ chosen by held-out validation; shows the `obj = misfit + reg` split).
- **Adjacent:** [`entries_apply_probe.md`](entries_apply_probe.md) (the three sampling ops + their
  transposes), [`transposes.md`](transposes.md) (ambient/corewise/tangent taxonomy),
  [`batching_and_stacking.md`](batching_and_stacking.md) (the `W`/`K`/`C` stack design),
  [`rank_continuation.md`](rank_continuation.md) (growing ranks during a fit).
- **Implementation rationale** (memory tradeoffs, einsum paths, deferred work):
  [`contributor/fitting_internals.md`](contributor/fitting_internals.md).
