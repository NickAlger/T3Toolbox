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

A small end-to-end problem to drive — a rank-`(2,2,2)`/`(1,2,2,1)` tensor `A` we pretend not to know,
measured 120 times by each sampling kind (normalizing every measurement row to unit norm is what keeps
the least-squares well conditioned):

```python
>>> import numpy as np
>>> import t3toolbox.tucker_tensor_train as t3
>>> import t3toolbox.manifold as t3m
>>> import t3toolbox.optimizers as topt
>>> np.random.seed(0)
>>> shape, tucker_ranks, tt_ranks = (6, 7, 8), (2, 2, 2), (1, 2, 2, 1)
>>> A = t3.TuckerTensorTrain.randn(shape, tucker_ranks, tt_ranks)
>>> ww = [np.random.randn(120, N) for N in shape]                    # measurement vectors, len=d
>>> ww = [w / np.linalg.norm(w, axis=1, keepdims=True) for w in ww]  # unit-norm rows
>>> pp = [np.random.randn(120, N) for N in shape]                    # perturbation directions
>>> pp = [p / np.linalg.norm(p, axis=1, keepdims=True) for p in pp]
>>> K, rng = 2, np.random.default_rng(0)
>>> data  = A.apply(ww)                      # the apply measurements,        shape W
>>> pdata = A.probe(ww)                      # the probe measurements,        len=d
>>> jets  = A.apply_derivatives(ww, pp, K)   # the apply-derivative jets, shape (K+1,)+W
>>> x0 = t3.TuckerTensorTrain.zeros(shape, tucker_ranks, tt_ranks)

```

Every fit is then one call:

```python
>>> # fit from applies, on the fixed-rank manifold, by Newton-CG (zero start is fine on the manifold):
>>> x_opt, stats = topt.newton_cg(t3m.MANIFOLD, 'apply', ww, data, x0, max_newton=30)
>>> bool(np.linalg.norm(x_opt.to_dense() - A.to_dense()) < 1e-8 * np.linalg.norm(A.to_dense()))
True

>>> # fit from probes, on the raw cores (corewise), by Adam (corewise needs a NONZERO start):
>>> x0c = t3.TuckerTensorTrain.randn(shape, tucker_ranks, tt_ranks)
>>> x_ad, stats = topt.adam(t3m.COREWISE, 'probe', ww, pdata, x0c, rng, 20, max_iter=50)
>>> misfit = lambda z: float(sum(np.sum((zi - yi) ** 2) for zi, yi in zip(z.probe(ww), pdata)))
>>> bool(misfit(x_ad) < misfit(x0c))                    # 50 Adam steps descend
True

>>> # fit from apply-DERIVATIVE jets, by Manifold Cauchy SGD, with a per-order weight ω (§2.3 adds a draw):
>>> w_order = [1.0, 0.5, 0.25]                          # ω, shape (K+1,) -- per order
>>> x_mc, stats = topt.mc_sgd(t3m.MANIFOLD, 'apply_derivatives', (ww, pp), jets, x0,
...                           rng, 20, order=K, weight=w_order, max_iter=60, check_every=20)
>>> bool(stats['losses'][-1] < stats['losses'][0])      # the (EMA-smoothed) full-batch loss checks descend
True

>>> # fit from probes with a PER-MODE weight (discount a noisier mode; probe-only, see §4.6):
>>> w_mode = [1.0, 1.0, 0.5]                            # ω_mode, shape (d,)
>>> x_pm, stats = topt.newton_cg(t3m.MANIFOLD, 'probe', ww, pdata, x0, weight=w_mode, max_newton=20)
>>> bool(np.linalg.norm(x_pm.to_dense() - A.to_dense()) < 1e-6 * np.linalg.norm(A.to_dense()))
True

>>> # add identity (Tikhonov) regularization ½λ‖x‖² to any fit (any optimizer/geometry/kind, see §4.9):
>>> x_rg, stats = topt.newton_cg(t3m.MANIFOLD, 'apply', ww, data, x0,
...                              regularizer=topt.IdentityRegularizer(1e-3), max_newton=10)
>>> last = stats['history'][-1]                         # the objective splits as misfit + reg
>>> bool(np.allclose(last['objective'], last['misfit'] + last['regularization']))
True

```

- **Optimizers:** `gradient_descent` (Cauchy + Armijo), `mc_sgd` (Manifold Cauchy SGD — tuning-free
  stochastic), `adam` (corewise first-order), `newton_cg` (inexact Riemannian, 2nd-order). All return
  `(x_opt: TuckerTensorTrain, stats: dict)`. `newton_cg`'s `stats`:

  | key | meaning |
  |---|---|
  | `losses` | the objective at each iterate (`losses[k] == history[k]['objective']`) |
  | `newton_steps` | CG solves — the history rows that took a step, rejected trust-region trials included (named `newton` before 2026.3.0) |
  | `newton_accepted` | steps taken: `newton_steps` minus the rejections (an Armijo-exhausted step under `on_line_search_failure='stop'`, or a rejected trust-region trial) |
  | `stop_reason` | `'gradient'` (the relative stop fired), `'max_newton'` (the cap), `'line_search'` (Armijo exhausted), or — trust region only, §4.12 — `'objective_floor'` (the attainable accuracy is reached: **not a failure**) / `'trust_region_floor'` (the radius collapsed) |
  | `history` | one scalar row per iteration (`backend.optimizers.NewtonInfo`), including the reference kinds `g0_reference` / `cg_reference` (§5), `hessian`, `step_control`, the trust-region fields (`tr_radius`, `tr_accepted`, `cg_boundary`), the exact-mode diagnostics (`pKp`, `lift_sigma_min`, `lift_eps`; §4.12), and `stop_reason` on the last row |

- **Live diagnostics (`newton_cg`):** `newton_cg(..., verbose=True)` prints a per-iteration block —
  objective / gradient norm, CG iterations / tolerance / status, line search, the actual-vs-predicted
  reduction `ρ`, and a per-`(mode, order)` **relative-error table** (`‖S(x)_ij − y_ij‖/‖y_ij‖`); pass
  `val_sample`/`val_data` to add a validation column. The records are also returned in
  `stats['diagnostics']`. The table layout follows the kind's axes — plain probe has modes in columns,
  `probe_derivatives` has modes × orders. Worked demo (both layouts): `examples/fit_probe_display.py`.
  An exact-Hessian run adds its trust-region and curvature columns (§4.12). Backend users get the identical display via `backend.optimizer_display.make_newton_display` +
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
- **`regularizer`** (optional, see §4.9) adds a term `ρ(x)` to the objective; `IdentityRegularizer(λ)`
  is `½λ‖x‖²`, and composes with every optimizer / geometry / kind.
- **Derivative kinds** take keyword `order` (highest derivative order). `mc_sgd`/`adam` also take `draw`
  (§2.3).
- **`hessian='exact'`** (`newton_cg` only) solves with the exact Riemannian Hessian under a trust region
  instead of Gauss-Newton under a line search — the superlinear Newton tail on a nonzero-residual fit
  (§4.12).

### 2.2 Build the local model (roll your own optimizer)

If you want to write your own iteration (a custom trust region, a different line search), grab the model:

```python
>>> import t3toolbox.fitting as fitting
>>> x = t3.TuckerTensorTrain.randn(shape, tucker_ranks, tt_ranks)   # the current point
>>> residual = x.apply(ww) - data                                   # residual r = apply(x) − y
>>> model = fitting.apply_model(t3m.MANIFOLD, x, ww, residual)
>>> g  = model.gradient                   # Π 𝒥ᵀr   (a gauged T3Tangent)
>>> v  = t3m.MANIFOLD.randn(model.frame)  # any tangent at the model's frame
>>> Hv = model.gn_hessian(v)              # Π 𝒥ᵀ𝒥Π v (the GN normal operator; symmetric PSD)
>>> q  = model.gn_quadratic(v)            # ‖Jv‖²   (ONE forward — the cheap Cauchy/line-search denominator)
>>> bool(np.allclose(float(q), float(v.corewise_inner(Hv))))        # vᵀHv = ‖Jv‖², one forward vs two
True
>>> α  = float(g.corewise_inner(g)) / float(model.gn_quadratic(g))  # the Cauchy step length
>>> x_new = t3m.MANIFOLD.retract(-α * g)
>>> bool(0.5 * np.sum((x_new.apply(ww) - data) ** 2) < float(model.objective_value))
True

```

Factories: `apply_model` / `entries_model` / `probe_model` (the last takes an optional per-mode
`weight`), and `apply_derivatives_model` / `entries_derivatives_model` / `probe_derivatives_model` (which
additionally take `order` + optional `weight`; see §4.6). The model is generic over the geometry — pass
`t3m.MANIFOLD` or `t3m.COREWISE`. Everything in
and out is a `T3Tangent` at `model.frame`. The frame sweep is computed once and reused across every
`gradient`/`gn_hessian`/`evaluate` (so an inner CG pays for it once, not per matvec).

Every factory also takes `hessian='gauss_newton'` (the default) or `'exact'`, and the model it returns is a
`fitting.LocalModel` in that **mode** — derived, never stored: `model.is_exact`. The exact model adds the
curvature term of §4.12 to the Hessian actions and leaves everything else as it was:

| member | Gauss-Newton model | exact model (`hessian='exact'`) |
|---|---|---|
| `objective_value`, `gradient`, `jacobian(p)` | `½‖ω⊙r‖²` (+ `ρ`), `Π𝒥ᵀω²r` (+ `g_R`), `𝒥Πp` | the same (the gradient bit for bit) |
| `gn_hessian(p)`, `gn_quadratic(p)` | the Gauss-Newton action / form | the same — the Gauss-Newton term alone |
| `hessian(p)`, `quadratic(p)` | `gn_hessian(p)`, `gn_quadratic(p)` | plus the curvature term: the exact Riemannian Hessian |
| `curvature_hessian(p)`, `curvature_quadratic(p)` | raise (structural) | the curvature term alone (it can be negative) |
| `evaluate(p)` | `c + gᵀp + ½ gn_quadratic(p)` | `c + gᵀp + ½ quadratic(p)` |
| `retract(p)` | `geometry.retract(p)` at the model's frame | the same |

```python
>>> emodel = fitting.apply_model(t3m.MANIFOLD, x, ww, residual, hessian='exact')
>>> emodel.is_exact, model.is_exact
(True, False)
>>> bool(np.allclose(float(emodel.quadratic(v)),                    # exact = Gauss-Newton + curvature
...                  float(emodel.gn_quadratic(v) + emodel.curvature_quadratic(v))))
True
>>> bool(np.allclose(float(emodel.quadratic(v)), float(v.corewise_inner(emodel.hessian(v)))))
True

```

### 2.3 Custom minibatching — the `draw`

Stochastic optimizers (`mc_sgd`, `adam`) draw a fresh minibatch each step. You can hand them **any**
function that returns a random sub-batch of the measurements:

```python
>>> def my_draw(rng):
...     idx = rng.choice(120, size=16, replace=False)   # any index expression on YOUR arrays
...     return ([w[idx] for w in ww], data[idx])        # the restricted (sample, data)
>>> x_mb, stats = topt.mc_sgd(t3m.MANIFOLD, 'apply', ww, data, x0,
...                           rng, 16, draw=my_draw, max_iter=60, check_every=20)
>>> bool(stats['losses'][-1] < stats['losses'][0])
True

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

Each has a uniform twin (`UNIFORM_MANIFOLD` / `UNIFORM_COREWISE`, chosen to match a uniform `x0`), and
**any of the four can be wrapped** to constrain the Tucker factors equal within groups of modes:
`shared(MANIFOLD, sharing)` — shorthand `shared_manifold(sharing)` — restricts the search to the
shared-factor submanifold, and every optimizer works on it unchanged. See [`sharing.md`](sharing.md).

---

## 3. The pieces (structure)

```
        t3toolbox/optimizers.py   (frontend adapter: kind-string + order/weight/draw -> TuckerTensorTrain;
                                   least_squares_problem -> the backend Problem)
        t3toolbox/fitting.py      (frontend: LocalModel + *_model factories; T3Tangent in/out)
   ─────────────────────────────  the backend razor: a raw-.data user runs the SAME check-free code ──────
        backend/optimizers.py     (algorithms; Problem/LocalModel oracle; flat_draw) -- no T3 imports
        backend/geometry.py       (the T3 geometries: Manifold/Corewise + their uniform twins)
        backend/fitting.py        (SamplingKind: APPLY/ENTRIES/PROBE + *_derivatives_kind; sumsq helpers)
        backend/probing.py        (the bare 𝒥/𝒥ᵀ for apply/entries/probe + frame-sweep reuse hooks)
        backend/sampling_derivatives.py  (the derivative 𝒥/𝒥ᵀ + frame-sweep-jets reuse hooks)
        manifold.py               (MANIFOLD/COREWISE geometries; T3Tangent; retract/project/frame)
```

- **`SamplingKind`** (`backend/fitting.py`) — the operations the GN model needs for one measurement
  operator: `precompute` (the reusable frame sweep), `forward` (`𝒥v`), `transpose` (`𝒥ᵀr`, summed over
  `W`), `sumsq` (the `‖·‖²` reduction), `w_axes`, plus `point_forward` (`S(x)`, for the residual) and the
  minimal layout for the default draw (`n_measurements`, `take`). It carries **no gauge** — that's the
  geometry's. Singletons `APPLY`/`ENTRIES`/`PROBE`; parameterized
  `ProbeKind` / `*DerivativesKind`, also spelled `*_derivatives_kind(order, weight)`.

  **Writing your own** — subclass. Which methods you must supply depends on where you start: from a
  concrete kind (`ApplyKind`, `ProbeKind`, a `*DerivativesKind`) you override only what differs; from
  `ScalarOutputKind` or `ProbeOutputKind` — pick the one matching your output shape, and you inherit the
  `‖·‖²` reductions — you supply the five operations plus `w_axes` / `n_measurements`; from bare
  `SamplingKind` you supply all nine. Each unimplemented one raises with a message saying so. A sixth
  operation, `corewise_hessian` (the derivative of `transpose` in the raw cores — §4.12), is needed only for
  `hessian='exact'`: a kind without it works for Gauss-Newton, and an exact model rejects it up front with a
  structural error naming the operation.
  Keep your parameters as dataclass **fields**: a kind rides as jax `aux_data`, so its hash/eq are part
  of the compilation cache key, and fields give you value identity (a rebuilt kind is the *same* key, a
  differently-parameterized one is not) for free. A parameter stashed on the instance instead of
  declared — captured in a hand-written `__init__`, or on a class that forgot its `@dataclass`
  decorator — is invisible to that identity, and raises a `TypeError` saying so the first time the
  object is hashed or compared.
  Deriving a variant by copying an existing kind and swapping a function is prevented outright: a
  variant is a subclass, hence a distinct type.
- **`Geometry`** (`manifold.py` `MANIFOLD`/`COREWISE`; `backend/geometry.py`) — the nine members the
  `Geometry` protocol in `backend/optimizers.py` declares: `frame(x)`, `base_point(frame)`,
  `stack_shape(x)`, `project` (the gauge `Π`), `retract`, the coordinate `inner`, `point_norm_sq` /
  `point_tangent` (the regularizer's hooks), and `precompute(frame)` returning a per-frame **geometry
  aux** that `project`/`retract` then receive. The
  aux exists so a geometry with per-frame setup pays for it once per local model rather than once per
  matvec (the shared-factor wrapper's SVD companion is the motivating case); it is `None` for
  `MANIFOLD`/`COREWISE`, and a custom geometry need only accept and ignore `aux=None`. It is
  carried as a leaf on `backend.optimizers.LocalModel` (`geom_aux`) and on `fitting.LocalModel`
  (`geometry_aux`) — [`contributor/precompute_and_caching.md`](contributor/precompute_and_caching.md). One more member is
  **optional** (looked up duck-typed): `zero_point_frame(frame)`, the canonical frame at the zero tensor
  where the default Newton reference is evaluated (§5). The manifold geometries define it, the corewise
  ones return `None`, and a custom geometry without it gets the warm-start reference. The exact Hessian
  (§4.12) needs four more, checked only when `hessian='exact'` asks for them: `curvature_precompute`, `lift`,
  `lift_transpose` and `tangential_curvature` (the `Geometry` protocol documents them); the four built-in
  geometries, and their shared wrappers, have them.
- **`Problem` + `backend.optimizers.LocalModel`** (`backend/optimizers.py`) — the backend oracle.
  `Problem(geom, kind, sample, data)` is **layout-agnostic**: `local_model(x [, sample_B, data_B])`
  linearizes at a point on the full data or an explicit minibatch, returning a
  `backend.optimizers.LocalModel` with `.gradient` / `.objective` / `.hvp` / `.gn_quadratic` /
  `.retract` / `.quadratic` (the mode's form; built with `hessian='exact'` it adds the curvature term, and
  `.curvature_hvp` / `.curvature_quadratic` give that term alone; §4.12). The frame sweep is computed once and shared. **`fitting.LocalModel`** (`fitting.py`) is its
  interactive frontend twin (same name, one level up; `T3Tangent` in/out), verified bit-identical — except
  the exact `hessian(p)`, which sums its two gauged halves (`gn_hessian(p) + curvature_hessian(p)`) where
  the backend `hvp` projects once, so the two agree to rounding (~1e-12), not bitwise; the curvature actions,
  the gradient and the quadratic forms are bitwise.
  To get a `Problem` from the frontend vocabulary (a geometry singleton, a kind name, `order` / `weight`,
  a ragged or uniform `x0`) — say, to drive a backend optimizer or build local models in a research
  script — call **`optimizers.least_squares_problem(geometry, kind, sample, data, x0, ...)`**, the
  constructor the four frontend optimizers use. It returns a `LeastSquaresSetup` with `.problem`, the
  backend starting cores `.x0_cores` (reduced to minimal ranks where the optimizers would reduce them),
  and `.rewrap` (backend cores → the frontend tensor).
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

**The qualifier:** the Gauss-Newton model is exact on the *affine tangent space*, not on the manifold. The
manifold curves away from its tangent space, and where the residual is nonzero that curvature is a
genuine second-order term of the objective along the retraction; Gauss-Newton drops it, and converges
linearly at a rate it sets. `hessian='exact'` adds it (§4.12).

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
"jax_enable_x64", True)` first if you need float64), and, if jax is not installed, **warns and runs eager on numpy** — the
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
forward-stable spelling if a less-important weighting axis is ever added). A per-mode weight may have
**more** rows than modes — the extra rows are ignored, so one weight can ride through a continuation
scheme that adds modes — but fewer rows than modes (other than the broadcast `1`) is a structural error,
on both layers.

```python
>>> # probe_derivatives: down-weight the noisy high orders AND discount a coarse mode, independently:
>>> ww3, pp3 = [w[:20] for w in ww], [p[:20] for p in pp]                       # 20 probes is plenty here
>>> r = [zi - yi for zi, yi in zip(x.probe_derivatives(ww3, pp3, 3),           # RAW r = S(x) − y,
...                                A.probe_derivatives(ww3, pp3, 3))]          #   len=d, (order+1)+W+(Ni,)
>>> mode_w, order_w = [1.0, 0.6, 0.3], [1.0, 0.5, 0.3, 0.2]                    # (d=3,) and (order+1=4,)
>>> model = fitting.probe_derivatives_model(t3m.MANIFOLD, x, ww3, pp3, order=3, residual=r,
...                                         weight=np.outer(mode_w, order_w))   # ω[mode,order], (3, 4)
>>> np.outer(mode_w, order_w).shape
(3, 4)
>>> # plain probe: a per-mode vector is 1-D (d,):
>>> rp = [zi - yi for zi, yi in zip(x.probe(ww), pdata)]
>>> model = fitting.probe_model(t3m.MANIFOLD, x, ww, rp, weight=[0.5, 1.0, 2.0])
>>> print(model.gradient.is_gauged())
True

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

### 4.9 Regularization: an optional objective term

Any fit takes an optional **regularizer** that adds a term `ρ(x)` to the objective:
`½‖ω⊙(S(x) − y)‖² + ρ(x)`. Pass it as `regularizer=` to any optimizer — it composes with every
geometry, sampling kind, and the ragged/uniform layers, because it acts on `x`, not on the
measurements. The one shipped regularizer is **identity (Tikhonov)**, `ρ(x) = ½λ‖x‖²`:

```python
>>> x_opt, stats = topt.newton_cg(t3m.MANIFOLD, 'apply', ww, data, x0, max_newton=10,
...                               regularizer=topt.IdentityRegularizer(1e-3))   # λ = 1e-3
>>> last = stats['history'][-1]
>>> sorted(k for k in last if k in ('objective', 'misfit', 'regularization'))
['misfit', 'objective', 'regularization']
>>> bool(last['regularization'] > 0.0)          # λ‖x‖²/2 -- the term λ contributes
True

```

`ρ` is measured in the **geometry's own metric**: on `MANIFOLD` it is the Hilbert–Schmidt ridge
`½λ‖x‖²_HS` (its Gauss–Newton term is exactly `+λI`, conditioning the poorly-determined tangent
directions); on `COREWISE` it is weight decay `½λ Σ_c ‖core_c‖²`.

- **Choosing λ.** There is no universal value — sweep it and pick by a **held-out validation** set
  (the training misfit only ever prefers `λ = 0`, so it can't choose λ). Worked recipe:
  [`examples/fit_hilbert_regularized.py`](https://github.com/NickAlger/T3Toolbox/blob/main/examples/fit_hilbert_regularized.py).
- **Watching it.** `newton_cg(..., verbose=True)` prints the objective split `obj = misfit + reg` each
  iteration; the same `misfit` / `regularization` fields ride along in `stats['history']`
  (`regularization` is `None` when no regularizer is attached) — read them to see how hard λ is pulling.
- **Stochastic optimizers.** `mc_sgd` / `adam` automatically scale `ρ` by `batch/n` each minibatch step,
  so λ keeps the same meaning whatever batch size you use — you do **not** rescale λ by hand.
- **Custom regularizers.** `IdentityRegularizer` is one implementation of the small `Regularizer` protocol
  (`backend.regularization`); a custom term supplies its value / gradient / Gauss–Newton contribution and
  drops into the same slot. (An inverse-unfolding-singular-value prior, after Grasedyck–Kramer, is future
  work.)

---

### 4.10 Stacked points: build the model, not the fit

A **stacked** point carries a batch of base points `C`, and every objective is then an array of shape
`C` — one value per element. The Gauss-Newton model handles that fine: `Problem.local_model` and the
frontend `fitting.LocalModel` accept a stacked point and return a shape-`C` `objective` with a matching
per-element gradient, so **driving your own loop over a stacked problem works**.

The four library optimizers do **not** support it, and raise `NotImplementedError` if you try. Their
convergence tests, Armijo line searches, and plateau detection all reduce the objective with a Python
`float()`, which a shape-`C` array cannot satisfy. Fit the elements separately
(`backend.stacking.unstack`) or roll your own loop. Stacked optimization is a possible future feature.

Separately, a **regularizer on a stacked point** raises wherever it is attached — including on the
model path that otherwise supports stacking. The data misfit keeps `C` but every regularizer scalar
collapses it, so `objective = misfit + ρ` would add the whole-stack regularization total to each
element, inflating the effective `λ` by about `|C|` and doing it unevenly.

---

### 4.11 A zero start makes the ragged and uniform layers diverge

A zero point has no preferred orthonormal frame, so the completion the two layers pick differs — same
objective, different gradient, and from there different iterates. On a well-conditioned `apply` fit both
still converge to the same answer; on an ill-conditioned `entries` fit they need not (measured: from
zero, ragged reached `8.8e-16` where uniform stalled at `8.49`; from a small random start both reached
`~2e-10`). The layers agree to ~1e-14 per step from **any nonzero** start.

So "a zero start is fine on the manifold" — true, and still the simplest thing to do — is a statement
about one layer at a time, not a promise that the two layers will track each other. If you are comparing
representations, start nonzero. (The optimizers' *reference* `‖g₀‖` is not part of this divergence: it
is evaluated at a canonical zero-point frame whose real blocks are the same on both layers — §5.)

### 4.12 The exact Hessian and the trust region

`newton_cg(..., hessian='exact')` solves each Newton step with the **exact Riemannian Hessian**,

```
H p = Π𝒥ᵀω²𝒥Π p  +  Π Lᵀ(K_{Aᵀω²r} − K_{grad f}) L Π p  (+ H_R p)
```

the Gauss-Newton term plus the manifold's curvature (the Weingarten term of Absil, Mahony & Trumpf 2013):
`L` lifts the gauged frame coordinates to raw core directions, `K_{Aᵀω²r}` is the kind's
`corewise_hessian` (the derivative of its corewise transpose at the fixed weighted residual) and
`K_{grad f}` the same second derivative of `⟨grad f, Φ(θ)⟩` at the misfit gradient — together, the
residual's normal part paired with the manifold's second fundamental form. On a corewise geometry it is
the plain Hessian of `f ∘ Φ` in the cores (no lift, no gradient term). Every sampling kind implements it,
on every geometry and representation — `MANIFOLD` / `COREWISE`, the uniform twins and the shared wrappers
of all four. The same `hessian='exact'` builds an exact `fitting.LocalModel` from any `*_model` factory
(§2.2) or an exact backend model from `Problem.local_model`. The derivation is the math note
[`exact_hessian_two_hole.pdf`](exact_hessian_two_hole.pdf); the design record is
[`contributor/curvature_internals.md`](contributor/curvature_internals.md).

**When it helps.** On a **nonzero-residual** fit — noisy data, or a target the ranks cannot represent —
Gauss-Newton converges linearly, at a rate set by the term it drops (on the history rows, `|1 − ρ|` and the
ratio of successive gradient norms both measure it), while the exact Hessian gives the Newton tail. On a
probe-derivative continuation restart (orders 0–2, 5 % noise, rank 2 → 3), Gauss-Newton with its line
search took 39 steps and 130 Hessian-applies, the exact Hessian with its trust region 10 and 28; on a
seeded set of shared and unshared restarts of one such problem, 3.5–5× fewer outer steps and 5–6.6× fewer
CG iterations, 1.6–2× less wall time. Where the residual vanishes at the solution (noise-free data at the
true ranks) Gauss-Newton is already fast and the curvature term only adds cost. Far from a minimizer — a
cold zero start, or the first steps after a zero-padded restart — the exact Hessian can be indefinite, so
the trust region limits the early steps, and it can settle in a different local minimum than Gauss-Newton
would (measured downstream from a zero-padded restart: a verified minimum about 12 % above Gauss-Newton's).
A Gauss-Newton warm start followed by exact steps is the natural pattern — the schedule is yours to choose
(every knob is a `newton_cg` keyword, per call).

```python
>>> np.random.seed(1)
>>> target = t3.TuckerTensorTrain.randn(shape, (3, 3, 3), (1, 3, 3, 1))     # rank 3: our rank-2 fit cannot match it
>>> y = [z + 0.01 * np.random.randn(*z.shape) for z in target.probe_derivatives(ww3, pp3, 1)]
>>> xw, _ = topt.newton_cg(t3m.MANIFOLD, 'probe_derivatives', (ww3, pp3), y, x0, order=1, max_newton=8)
>>> kw = dict(order=1, max_newton=60, gtol_rel=1e-10, g0_reference='warm_start')   # a tight stop, from xw
>>> _, gn = topt.newton_cg(t3m.MANIFOLD, 'probe_derivatives', (ww3, pp3), y, xw, **kw)
>>> _, ex = topt.newton_cg(t3m.MANIFOLD, 'probe_derivatives', (ww3, pp3), y, xw, hessian='exact',
...                        cg_forcing_power=1.0, **kw)
>>> ex['stop_reason'] in ('gradient', 'objective_floor')    # exact converges to the resolution floor ...
True
>>> ex['newton_steps'] <= 10 and ex['newton_steps'] < gn['newton_steps']    # ... in fewer steps ...
True
>>> rel = lambda st, k: st['history'][k]['gnorm'] / st['history'][0]['gnorm']
>>> rel(ex, 5) < 1e-8 < rel(gn, 5)    # ... because GN is linear to the end: five steps in, exact is at the floor
True
>>> row = ex['history'][0]
>>> row['hessian'], row['step_control'], row['tr_accepted']
('exact', 'trust_region', True)

```

**The API.** Four `newton_cg` keywords, the same on the frontend and the backend:

| keyword | default | meaning |
|---|---|---|
| `hessian` | `'gauss_newton'` | `'exact'` for the exact Riemannian Hessian (case-insensitive) |
| `step_control` | `None` | `'line_search'` (Armijo along the retraction; the Gauss-Newton default) or `'trust_region'` (the exact default, since the exact Hessian is indefinite away from a minimizer); either works with either Hessian |
| `tr_radius0` | `None` | the initial trust radius `Δ0`; `None` is the Gauss-Newton Cauchy step length `‖g‖³/⟨g, H_GN g⟩` |
| `curvature_damping` | `None` | the lift's damping `ε`, relative to `‖X‖_HS`; `None` is the geometry's default `1e-6` |

An option where it would have no effect raises (structural): `tr_radius0` with the line search,
`on_line_search_failure='accept'` with the trust region, `curvature_damping` with Gauss-Newton or on a
corewise geometry. The trust region is Steihaug–Toint: CG inside `‖p‖ ≤ Δ` steps to the boundary on
nonpositive curvature or when it would leave the region, then **one** trial retraction is accepted iff
the actual-over-predicted decrease `ρ > 0.1`; `Δ` shrinks to `¼‖p‖` when `ρ < 0.25` and doubles (capped at
`10⁴·Δ0`) when `ρ > 0.75` on a boundary step. A rejected trial keeps `x` and its local model, and counts
toward `max_newton`.

**The forcing power.** `cg_forcing_power` is `0.5` by default in **both** modes. With `hessian='exact'`,
`cg_forcing_power=1.0` gives the **q-quadratic** tail (measured: `e_{k+1} ≤ 0.92·e_k²` over the asymptotic
steps), and `0.5` a superlinear one of order about 1.5 — less CG per step, a step or two more.

**Cost.** An exact Hessian-apply costs about 3–4× a Gauss-Newton one (measured 2.8–3.9× on numpy CPU
across the six kinds and two problem sizes; 1.9–3.5× on downstream probe-derivative fits); its memory peak
is about twice the Gauss-Newton transpose's. Per Newton step the model does **one** misfit transpose
(shared by the gradient and the curvature term) plus one small T3-to-T3 sweep and `2d − 1` small SVDs (the
lift's `d` Tucker and `d − 1` bond pseudo-inverses); it adds no `W`-sized state. Under `use_jit=True` the
exact CG compiles once per run, like Gauss-Newton's (about 1.5× the compile time). For a uniform
`probe_derivatives` fit — the one chunked path — `chunk_size='auto'` sizes the chunk so the exact Hessian's
assembly fits too ([`chunking.md`](chunking.md)); ragged fits and the other five kinds do not chunk, in
either mode.

**Restarts need nothing special.** From a zero-padded continuation restart ([`rank_continuation.md`](rank_continuation.md))
the new directions have zero singular values; the lift's clipped pseudo-inverse drops them (the Hessian
is Gauss-Newton in those directions, while the gradient still moves along them) and the trust region
bounds the step, so exact mode runs from the restart as from any other point (on a shared geometry, see
[`sharing.md`](sharing.md)); it can still settle in a different local minimum than Gauss-Newton from
there ("When it helps" above). The exact model is also exact at frames that are not minimal (Tucker or TT
slack), and the backend `newton_cg` runs from them without reducing them (the frontend reduces a ragged
manifold `x0` to minimal ranks first, in either mode).

**Reading the display.** With `verbose=True` an exact trust-region row prints its step segment as
`TR Δ … ✓` (accepted) or `✗` (rejected — the next row starts from the same point), CG's status `∂` for a
boundary stop, and appends `κ σ ε`:

- `κ = pKp / (pHp − pKp)` — the curvature term's curvature along the step relative to the Gauss-Newton
  term's (`pKp` in the history; `pHp − pKp` is the Gauss-Newton part). Near 0, Gauss-Newton would have predicted the step well; `|κ|` of order 1 or more is where
  the exact Hessian earns its cost.
- `σ` (`lift_sigma_min`) — the smallest singular value the lift inverts (the bond and Tucker spectra; a
  shared group's spectrum `s_g` in place of its modes' own). The curvature term carries `1/σ`, so the
  quadratic model's validity radius is about `σ`: a small `σ` means a nearly rank-deficient point, where
  the Hessian is strongly indefinite and the trust region stays small.
- `ε` (`lift_eps`) — the absolute lift damping, `curvature_damping · ‖X‖_HS`: the Hessian is exact in the
  directions with `σ ≫ ε` and damped toward Gauss-Newton in those with `σ ≲ ε`.

The last row reads `| stop: <reason>` when the run did not end on the gradient test. **`objective_floor`
is not a failure**: a rejected trial predicted a change below the objective's floating-point resolution,
so the attainable accuracy is reached (the gradient test simply cannot fire at that tolerance);
`trust_region_floor` means rejections drove `Δ` below `10⁻¹²·Δ0`.

**Where the curvature vanishes.** The curvature term lives where the representable set is curved. At a
slot of **maximal** rank (a Tucker rank equal to its mode size, a TT rank equal to the product of its
neighbours) the set is flat in that slot's directions and the term has no component there; at `d = 1`
every slot is maximal, so on the manifold geometries the exact model *is* Gauss-Newton (it skips the
work), while the corewise `d = 1` exact Hessian is still the plain Hessian of `f ∘ Φ`. No special case
is needed in between: the formula gives zero there by itself.

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
- **The relative stop is referenced to the zero tensor of the level's ranks.** `newton_cg`'s Newton stop
  (`‖g‖ ≤ gtol_rel·‖g₀‖`) and CG forcing term (`η = min(0.5, (‖g‖/‖g₀‖)**cg_forcing_power)`) are both
  *relative* to a reference `‖g₀‖`, chosen by `g0_reference` (`gradient_descent` takes it too):

  | `g0_reference` | the reference `‖g₀‖` |
  |---|---|
  | `None` (default) | the geometry's own: `'zero'` on `MANIFOLD` / `UNIFORM_MANIFOLD` and their shared wrappers, `'warm_start'` on the corewise geometries |
  | `'zero'` | the gradient norm at the **zero tensor of `x0`'s ranks** (one extra sweep and transpose per run) |
  | `'warm_start'` | the initial gradient norm `‖g(x0)‖` — the 2026.2.0 default |

  *Why the zero tensor.* After a warm start — a zero-padded `resize` in rank continuation — the initial
  gradient is already small (the guess is near the solution), so a warm-start reference over-tightens
  the Newton stop (the default `gtol_rel = 1e-8` becomes unreachable and every level runs to
  `max_newton`) and slackens CG. The zero-tensor gradient measures the data's pull at the level's ranks
  regardless of how good the guess is, and it grows with the ranks, so ratios against it are comparable
  across continuation levels. On a corewise geometry the zero cores are a critical point (the gradient
  there is exactly 0), so the default stays the warm start there, and an explicit `'zero'` raises.

  *What exactly the reference is: a convention.* At `X = 0` every orthonormal frame completion is valid,
  and the gradient's norm depends on which one is chosen. The library fixes the choice — the **canonical
  zero-point frame** (`backend.fv_operations.fv_zero_point_frame`): seeded Haar-random Tucker factors, one
  per sharing group, drawn from a stream keyed by the group's first mode with the fixed seed
  `ZERO_FRAME_SEED = 0`, and identity-prefix TT cores. So the zero reference is a deterministic
  data-gradient scale at these ranks, **good to an O(1) factor** (another seed, or a rotation of the
  problem, moved it by up to ~1.7× in the measured cases) — harmless for a relative stop, but not an
  intrinsic property of the problem. Three properties make it a good convention:
  - **It is not coordinate-aligned.** A coordinate-aligned completion — an identity prefix, or LAPACK's
    completion of a zero core — gives a reference of **exactly 0** whenever the data have no energy on
    its leading coordinates. A Haar completion is generic (the exact-zero trap does not occur).
  - **It is identical on the ragged and uniform layers.** The uniform frame has the same real blocks, so
    a ragged fit and its uniform twin are referenced to the same number (the start's own frame at a zero
    `x0` is another matter, §4.11).
  - **It is nested across rank-continuation levels.** A mode's factor at rank `r` is the leading block of
    its factor at rank `r + 1`, and is unchanged when another mode's rank grows.

  A **vanishing** zero reference (e.g. all-zero data) is never replaced silently: by default it falls
  back to the warm start with a `RuntimeWarning`, and an explicit `'zero'` raises. Every history row
  records which reference was used:

  ```python
  >>> _, st = topt.newton_cg(t3m.MANIFOLD, 'apply', ww, data, x0, max_newton=30)
  >>> st['history'][0]['g0_reference'], st['stop_reason']     # the default on the manifold
  ('zero', 'gradient')
  >>> _, st = topt.newton_cg(t3m.COREWISE, 'apply', ww, data, x0c, max_newton=2)
  >>> st['history'][0]['g0_reference']                         # ... and on corewise
  'warm_start'

  ```

  *Overriding it.* `g0norm_newton` / `g0norm_cg` pass an explicit reference (it must be `> 0`) and win
  over `g0_reference`: `g0norm_newton` sets both tests (CG inherits it, and the zero reference is then not
  computed at all), `g0norm_cg` alone touches only CG (`cg_reference='override'`). Separately,
  `cg_forcing_power` (default `0.5`) trades CG effort for Newton steps: a **larger** power (`0.75`, `1.0`)
  tightens CG per step — worth it on the manifold when the retraction is expensive relative to a
  Hessian-apply. The zero reference makes `η` tighter than a deflated warm start does (downstream
  continuation measured ~2× more CG work per step at high rank, for the same accuracy); that trade is
  accepted, and an `η` floor or an objective-plateau stop remain open questions.
- **The Newton schedule is yours.** Gauss-Newton (the default) or the exact Hessian (§4.12), the forcing
  power, the reference, the caps — every knob is a per-call `newton_cg` keyword, so a continuation loop can
  run each level with its own schedule (e.g. Gauss-Newton until the step stalls, then exact; or exact at
  every warm-started level). Which default is best is an open, empirical question; on the measured
  restarts the exact Hessian paid off whenever the fit had a nonzero residual.
- **A failed line search now stops the run (or, on request, pushes through).** If `newton_cg`'s
  Armijo search exhausts all 40 halvings, the objective's floor has been reached; the default
  `on_line_search_failure='stop'` rejects the step and terminates (the iteration's history row carries
  `ls_failed=True`, and `alpha` is the step of the last trial actually evaluated). Pass
  `'accept'` to take the last (smallest) trial step and keep iterating even though it may climb — a
  deliberate escape hatch: on some landscapes that small non-descent step bumps the iterate out of a
  stagnant region and convergence resumes. Same option on `gradient_descent`
  (`stats['line_search_failed']`).
- **When the data only constrains a structured subspace** (e.g. symmetric tensors), the fit fills
  the unconstrained null space with a large "halo" — **project/symmetrize the fitted tensor** to
  read off the meaningful part.
- **On the fixed-rank manifold, the rank is already your main regularizer.** Before reaching for a
  `regularizer` (§4.9), check whether an appropriate rank — or rank continuation — gives what you need;
  in the maintainer's toy studies identity `λ` was a modest secondary denoiser (most useful on
  ill-conditioned or noisy fits), not a substitute for choosing the rank. Worked example:
  [`examples/fit_hilbert_regularized.py`](https://github.com/NickAlger/T3Toolbox/blob/main/examples/fit_hilbert_regularized.py).

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
  fit, with λ chosen by held-out validation; shows the `obj = misfit + reg` split);
  `fit_shared_factors_jetted_probes.py` (**shared Tucker factors** — a groupwise-symmetric target fit
  from noisy probe-derivative jets, shared vs unshared under the same rank continuation).
- **Adjacent:** [`entries_apply_probe.md`](entries_apply_probe.md) (the three sampling ops + their
  transposes), [`transposes.md`](transposes.md) (ambient/corewise/tangent taxonomy),
  [`batching_and_stacking.md`](batching_and_stacking.md) (the `W`/`K`/`C` stack design),
  [`rank_continuation.md`](rank_continuation.md) (growing ranks during a fit),
  [`sharing.md`](sharing.md) (the shared-factor geometries), [`weighting.md`](weighting.md) (edge
  weights and the Grasedyck–Kramer tangent metric), [`chunking.md`](chunking.md) (`chunk_size` for the
  probe-derivative transpose and its exact Hessian).
- **Implementation rationale** (memory tradeoffs, einsum paths, deferred work):
  [`contributor/fitting_internals.md`](contributor/fitting_internals.md); for the exact Hessian (the lift,
  the two-hole sweeps, the masking rules, the measurements) [`contributor/curvature_internals.md`](contributor/curvature_internals.md).
