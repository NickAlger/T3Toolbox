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
>>> ww = [np.random.randn(120, N) for N in shape]                    # probe vectors, len=d
>>> ww = [w / np.linalg.norm(w, axis=1, keepdims=True) for w in ww]  # unit-norm rows
>>> pp = [np.random.randn(120, N) for N in shape]                    # perturbation directions
>>> pp = [p / np.linalg.norm(p, axis=1, keepdims=True) for p in pp]
>>> order, rng = 2, np.random.default_rng(0)
>>> data  = A.apply(ww)                          # the apply measurements,        shape W
>>> pdata = A.probe(ww)                          # the probe measurements,        len=d
>>> from t3toolbox.fitting import affine_jet, taylor_to_derivatives
>>> derivative_data = lambda kind, X, ww, pp, n: taylor_to_derivatives(       # the derivatives along w + s p,
...     kind, (n,), getattr(X, kind)(affine_jet(ww, pp), (n,)))               #   the old optimizers' data form
>>> jets  = derivative_data('apply', A, ww, pp, order)   # the apply-derivative jets, shape (order+1,)+W
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
>>> w_order = [1.0, 0.5, 0.25]                          # ω, shape (order+1,) -- per order
>>> x_mc, stats = topt.mc_sgd(t3m.MANIFOLD, 'apply_derivatives', (ww, pp), jets, x0,
...                           rng, 20, order=order, weight=w_order, max_iter=60, check_every=20)
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
  An exact-Hessian run adds its trust-region and curvature columns (§4.12), a `hessian='switch'` run an `sw` segment
  on its Gauss-Newton-phase rows (§4.14), and `timing=True` a `time` line per row and a closing table (§4.13).
  Backend users get the identical display via
  `backend.optimizer_display.make_newton_display` + `backend.optimizers.newton_cg(..., callback=...)`.
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
- **`hessian='switch'`** (`newton_cg` only) starts with Gauss-Newton and switches to the exact Hessian inside the
  run, once the exact model has predicted the objective clearly better (§4.14).

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

`draw(rng) → (sample_B, data_B)` returns the subset of probe vectors **and** their measured values.
*Every* slicing scheme — flat random pairs, slice-on-X, slice-on-P — is just a different index expression
*you* write, on *your* arrays. **If you don't pass one, you get the flat default** (`flat_draw`: a uniform
random subset across the whole flattened probe stack `W`). The optimizer never compiles the draw, so it's
unconstrained Python/numpy — or write it in jax on device-resident data to keep the minibatch on the GPU
(only the per-step *kernel* is jitted; see §4.5).

### 2.4 Choosing a geometry

| | **`MANIFOLD`** | **`COREWISE`** |
|---|---|---|
| optimizes | on the fixed-rank manifold (the gauge `Π` makes `g`/`H` Riemannian) | the raw cores `(B,G,G,G)`, Euclidean metric |
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
  ones return `None`, and a custom geometry without it gets the warm-start reference. Three more optional members
  serve a start frame after a rank increase ([`aligned_padded_frame.md`](aligned_padded_frame.md)), defined on the
  manifold geometries only: `with_frame(frame)`, the geometry at a frame's structure;
  `require_frame_structure(x_cores, frame)`, a structural check from integers and held masks (needed by
  `local_model(frame=)` and `start_frame`); and `aligned_padded_frame(x0, x_old, project, hint, ...)`, the frame of
  `x0` aligned with a hint (needed by `padded_from` and `g0_reference='cone'`). An absent member raises a
  `ValueError` naming it, only when its feature is asked for. The exact Hessian
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
>>> ww3, pp3 = [w[:20] for w in ww], [p[:20] for p in pp]                      # 20 probes is plenty here
>>> r = [zi - yi for zi, yi in zip(derivative_data('probe', x, ww3, pp3, 3),   # RAW r = S(x) − y,
...                                derivative_data('probe', A, ww3, pp3, 3))]  #   len=d, (order+1)+W+(Ni,)
>>> mode_w, order_w = [1.0, 0.6, 0.3], [1.0, 0.5, 0.3, 0.2]                    # (d=3,) and (order+1=4,)
>>> model = fitting.probe_derivatives_model(t3m.MANIFOLD, x, ww3, pp3, order=3, residual=r,
...                                         weight=np.outer(mode_w, order_w))  # ω[mode,order], (3, 4)
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
  drops into the same slot. (An inverse-singular-value prior on the matricizations and unfoldings, after
  Grasedyck–Kramer, is future work.)

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
probe-derivative fit from a zero-padded warm start (orders 0–2, 5 % noise, rank 2 → 3), Gauss-Newton with its line
search took 39 steps and 130 Hessian-applies, the exact Hessian with its trust region 10 and 28; on a
seeded set of shared and unshared restarts of one such problem, 3.5–5× fewer outer steps and 5–6.6× fewer
CG iterations, 1.6–2× less wall time. Where the residual vanishes at the solution (noise-free data at the
true ranks) Gauss-Newton is already fast and the curvature term only adds cost. Far from a minimizer — a
cold zero start, or the first steps after a zero-padded warm start — the exact Hessian can be indefinite, so
the trust region limits the early steps, and it can settle in a different local minimum than Gauss-Newton
would (measured downstream from a zero-padded warm start: a verified minimum about 12 % above Gauss-Newton's).
A Gauss-Newton warm start followed by exact steps is the natural pattern — the schedule is yours to choose
(every knob is a `newton_cg` keyword, per call), or `hessian='switch'` picks the moment to change inside one call
(§4.14).

```python
>>> np.random.seed(1)
>>> target = t3.TuckerTensorTrain.randn(shape, (3, 3, 3), (1, 3, 3, 1))     # rank 3: our rank-2 fit cannot match it
>>> y = [z + 0.01 * np.random.randn(*z.shape) for z in derivative_data('probe', target, ww3, pp3, 1)]
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
assembly fits too; ragged fits and the other five kinds do not chunk, in
either mode.

**Restarts need nothing special.** From a zero-padded warm start ([`rank_continuation.md`](rank_continuation.md))
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

### 4.13 Timing a Newton run

`newton_cg(..., timing=True)` records where a run's time goes. Each history row's `wall_time` window is split into
**phases**, with jax's compile wait beside each phase on a jax run, and the run's one-off costs are recorded beside
them. It answers questions like: is CG the cost, or the model build? the retraction or the line search's extra trials?
the alignment of a padded start? the compiler? It changes no number: `x` and every other `stats` field are bitwise
those of the untimed run, the `wall_time` clock readings aside. It is off by default, and the frontend and the backend
take the same keyword.

A worked example, one level of a rank continuation: a noisy probe-derivative fit at rank 2, zero-padded to rank 3
and restarted from the aligned frame (`padded_from`, [`aligned_padded_frame.md`](aligned_padded_frame.md)) with the
`'cone'` reference and the exact Hessian:

```python
>>> np.random.seed(0)
>>> rng = np.random.default_rng(0)
>>> probes = tuple([rng.standard_normal((40, 5)) for _ in range(3)] for _ in range(2))   # (ww, pp), d = 3
>>> truth = t3.TuckerTensorTrain.randn((5, 5, 5), (3, 3, 3), (1, 3, 3, 1))
>>> noisy = [z + 0.01 * rng.standard_normal(z.shape) for z in derivative_data('probe', truth, *probes, 1)]
>>> x_old, _ = topt.newton_cg(t3m.MANIFOLD, 'probe_derivatives', probes, noisy,
...                           t3.TuckerTensorTrain.randn((5, 5, 5), (2, 2, 2), (1, 2, 2, 1)), order=1, max_newton=5)
>>> x_new = x_old.resize((5, 5, 5), (3, 3, 3), (1, 3, 3, 1))              # the next level, zero-padded
>>> level = dict(order=1, max_newton=40, hessian='exact', cg_forcing_power=1.0, padded_from=x_old,
...              g0_reference='cone')
>>> _, st = topt.newton_cg(t3m.MANIFOLD, 'probe_derivatives', probes, noisy, x_new, timing=True, **level)
>>> T = st['timing']
>>> list(T)
['total', 'setup', 'alignment', 'alignment_passes', 'reference_passes', 'last_callback', 'phases', 'compile', 'cg_per_iteration', 'rows']
>>> rows = T['rows']                                           # one record per history row
>>> len(rows) == len(st['history'])
True
>>> all(abs(sum(s for s in row['seconds'].values() if s is not None) - row['window']) < 1e-9 for row in rows)
True
>>> all(row['window'] == h['wall_time'] for row, h in zip(rows, st['history']) if h['wall_time'] is not None)
True
>>> abs(T['total'] - (T['setup'] + T['alignment'] + sum(row['window'] for row in rows) + T['last_callback'])) < 1e-6
True
>>> T['compile'] is None, len(T['alignment_passes']) == len(st['start_movements'])     # numpy: no compiler to time
(True, True)
>>> _, plain = topt.newton_cg(t3m.MANIFOLD, 'probe_derivatives', probes, noisy, x_new, **level)
>>> 'timing' in plain, plain['losses'] == st['losses']        # the same run, untimed
(False, True)

```

The phases of one row tile its window (the second check above) and the windows, the setup, the alignment and the
last callback tile the run (the fourth); the window is the same float as the row's `wall_time` (the third).

**The phases** (`backend.optimizers.PhaseSeconds`, in run order; `None` when the phase did not run on that row):

| phase | what it covers | `None` when |
|---|---|---|
| `previous_callback` | the previous row's record, its `callback` (a display) and its history append | row 0 |
| `frame` | the orthonormal frame of the point (`geom.frame`), or the start frame's structure check | the row reuses the model (after a rejected trust-region trial) |
| `sweep` | the sampling kind's sweep at the frame (`kind.precompute`) | as `frame` |
| `companion` | the geometry's per-frame data (`geom.precompute`) | as `frame` |
| `residual` | the residual `S(x) − y` | as `frame` |
| `curvature` | the exact Hessian's curvature data (`geom.curvature_precompute`): with `hessian='exact'`, and with `'switch'` on every build, the Gauss-Newton phase's included (§4.14) | `hessian='gauss_newton'`; as `frame` |
| `gradient` | the misfit transpose `Π 𝒥ᵀ r` (in every mode; exact and switch mode run it inside the build), the objective's parts, the gradient, `‖g‖`, the lift diagnostics | as `frame` |
| `reference` | the reference `‖g0‖` (`g0_reference`): one sweep and transpose for `'zero'`, the alignment loop for `'cone'` | every row but row 0 |
| `cg` | the CG solve (under `use_jit`, one compiled call) | the converged row |
| `forms` | the step's scalars: the first trust radius, the slope, the descent fallback, `pᵀHp`, `‖p‖` | the converged row |
| `trial_retraction` | every trial's retraction (the line search may try several; the trust region tries one) | the converged row |
| `trial_objective` | every trial's objective | the converged row |
| `tracking` | `hessian='switch'`: the tracking product `⟨p, Lᵀ K_N L p⟩` on a tracked Gauss-Newton-phase row (its `switch_pKp`, §4.14) | every other row |
| `remainder` | what no phase covers: the forcing term, `‖x‖`, the trust region's data norm, bookkeeping | never |

A row's record (`NewtonInfo.timing`, a `NewtonTiming`; `stats['timing']['rows']` holds them as dicts) adds `window`,
`compile_seconds` (the same phases, jax's compile wait), `trials` (the retractions tried) and `accepted_retraction`
(the accepted trial's share of `trial_retraction`, `None` on a rejection). Row 0 also carries the run's **one-offs**,
which no window contains: `setup` (from the call to the loop: validation, the jax conversion, the start frame's
bookkeeping) and `alignment` (`padded_from`'s padded start) with their compile waits, and the seconds per pass of the
alignment and of the `'cone'` reference's loop (`alignment_passes`, `reference_passes`; breakdowns of `alignment` and
of row 0's `reference`).

**`stats['timing']`** is a plain dict:

| key | what |
|---|---|
| `total` | the call's entry to the loop's exit: `setup + (alignment or 0) + Σ window + last_callback` |
| `setup`, `alignment`, `alignment_passes`, `reference_passes` | the one-offs, as row 0 carries them (`alignment` `None` without `padded_from`) |
| `last_callback` | the last row's callback, which is in no window |
| `phases` | each phase summed over the rows |
| `compile` | the same for the compile wait, with `setup`, `alignment` and `last_callback`; `None` when not measured |
| `cg_per_iteration` | CG's seconds less its compile wait, per CG iteration |
| `rows` | every row's record, as a dict |

**The display.** With `verbose=True` (or `make_newton_display` as the backend callback) each timed row prints a
`time` line under its header, and the run ends with a closing table. The example's run on the maintainer's laptop
(numpy; timings vary run to run, so this block is not checked):

```
iter  0 | obj 2.547e+05  ‖g‖ 1.26e+04 (2.2e-01·g₀) | CG 1/200 tol 2.8e+03 resid 4.9e+03 ∂ | TR Δ 3.46e+01 ✓ ‖Δx‖/‖x‖ 2.6e-01 | Δf -2.19e+05 ρ 1.00 | 0.17s | κ -1.2e-03 σ 0.0e+00 ε 1.3e-04
  time  model 3.9e-03 [frame 7.9e-05 sweep 1.1e-03 comp 4.8e-06 resid 1.0e-03 curv 1.7e-03] | grad 2.8e-03 | ref 1.2e-01 ×9 | cg 2.1e-02 (2.1e-02/it) | forms 1.6e-02 | trials 1.9e-03 ×1 [retr 7.9e-04 obj 1.1e-03] | rest 1.9e-04 | setup 1.9e-04 | align 5.8e-02 ×5
  rel err  rows=mode  cols=order
           ord0     ord1
    m0  3.7e-01  1.9e-01
    m1  2.7e-01  2.8e-01
    m2  2.8e-01  2.6e-01
iter  1 | obj 3.617e+04  ‖g‖ 4.79e+03 (8.5e-02·g₀) | CG 3/200 tol 4.1e+02 resid 2.3e+02 ✓ | TR Δ 6.92e+01 ✓ ‖Δx‖/‖x‖ 1.2e-01 | Δf -3.57e+04 ρ 1.01 | 0.07s | κ +3.6e-03 σ 1.3e+01 ε 1.4e-04
  time  model 3.6e-03 [frame 4.9e-04 sweep 9.9e-04 comp 3.3e-06 resid 1.0e-03 curv 1.1e-03] | grad 2.8e-03 | cg 4.5e-02 (1.5e-02/it) | forms 1.2e-02 | trials 1.9e-03 ×1 [retr 7.9e-04 obj 1.2e-03] | cb 2.9e-04 | rest 1.8e-04
  rel err  rows=mode  cols=order
           ord0     ord1
    m0  1.1e-01  8.9e-02
    m1  1.2e-01  1.1e-01
    m2  1.1e-01  8.3e-02
…
iter  5 | obj 6.024e-02  ‖g‖ 2.65e-08 (4.7e-13·g₀) | converged (‖g‖ ≤ gtol·‖g₀‖, g₀ = cone (aligned zero tensor))
  time  model 3.6e-03 [frame 4.7e-04 sweep 9.5e-04 comp 2.9e-06 resid 1.0e-03 curv 1.1e-03] | grad 2.8e-03 | cb 2.6e-04 | rest 1.9e-05
  rel err  rows=mode  cols=order
           ord0     ord1
    m0  1.6e-04  9.5e-05
    m1  1.9e-04  1.2e-04
    m2  1.8e-04  1.0e-04
  timing  6 rows, Σ window 7.0e-01 s (the last row's callback is not in it)
                       seconds  % window  compile  mean/row
    previous_callback  1.6e-03       0.2        —   3.2e-04
    frame              2.6e-03       0.4        —   4.3e-04
    sweep              5.9e-03       0.8        —   9.8e-04
    companion          2.1e-05       0.0        —   3.4e-06
    residual           6.1e-03       0.9        —   1.0e-03
    curvature          7.4e-03       1.1        —   1.2e-03
    gradient           1.7e-02       2.4        —   2.8e-03
    reference          1.2e-01      17.4        —   1.2e-01
    cg                 4.7e-01      66.3        —   9.3e-02
    forms              6.4e-02       9.1        —   1.3e-02
    trial_retraction   3.9e-03       0.6        —   7.8e-04
    trial_objective    5.5e-03       0.8        —   1.1e-03
    remainder          8.8e-04       0.1        —   1.5e-04
    setup              1.9e-04         —        —         —
    alignment          5.8e-02         —        —         —
    alignment 5 passes, reference 9 passes, cg 1.5e-02 s per CG iteration
```

`model` is the build with its parts in brackets, `grad` the gradient phase, `ref` the reference with the cone loop's
pass count, `cg` its seconds and its seconds per CG iteration, `trials` the retractions and objectives with the trial
count (on a backtracked row, `accepted` is the accepted trial's retraction), `track` the switch's tracking product
(§4.14), `cb` the previous row's callback, `rest` the remainder; row 0 adds `setup` and `align` with its pass count,
and a jax run a `compile` cell (the phases that compiled, or `0`). In this run CG took two thirds of the rows' time
and row 0's `'cone'` reference a sixth.

**What it costs.** Every phase ends by waiting for its results (`common.tree_block_until_ready`), because jax
dispatches asynchronously: without the wait, a phase's work would be charged to whichever later phase first needed
its result. On numpy nothing is asynchronous and the waits are free. On jax they **serialize the run**: the host can
no longer queue the next phase while the device works, so a timed jax run is slower than an untimed one, and its
times describe the serialized run. Measured, timing on against off, in seconds per call (the median of 3 interleaved
pairs after a warm-up, 7 on numpy; the maintainer's laptop: AMD Ryzen 7 3700U, 4 cores, 13 GB, CPU only; Python 3.11,
numpy 2.4, jax 0.10.2 in x64):

| case | layer | numpy | jax, `use_jit` | jax, eager |
|---|---|---|---|---|
| the example above (d = 3; 6 rows, 31 CG iterations) | ragged | 0.77 → 0.74 (−3.9 %) | 3.49 → 3.55 (+1.7 %) | 13.98 → 13.82 (−1.1 %) |
| | uniform | 0.86 → 0.86 (−0.1 %) | 0.82 → 0.81 (−0.8 %) | 3.21 → 3.24 (+1.1 %) |
| d = 6: shape `(8,)*6`, 200 probes, order 1, rank 3, the default Gauss-Newton from a random start (10 rows, 27 CG iterations) | ragged | 0.81 → 0.81 (+0.2 %) | 3.96 → 3.97 (+0.3 %) | 9.64 → 9.68 (+0.4 %) |
| | uniform | 1.02 → 1.02 (+0.1 %) | 0.95 → 0.91 (−3.6 %) | 1.27 → 1.28 (+1.0 %) |

Every difference is within the run-to-run spread (single runs of one numpy call varied by up to 15 %): on this CPU,
timing costs nothing measurable. The waits themselves cost 0.1 % of a numpy run (71 waits of 10–13 µs in the
example), and on a CPU, jax's dispatch of these small operations overlaps little with the host's work, so there is
little to lose.

On a GPU (an A100-SXM4-80GB, jax 0.10.2 in x64, the same protocol) the two cases above and one level of a downstream
fit (dof 12,352) measured −1.4 % to +3.1 %, jit and eager, each within its run-to-run spread (up to 10 %): the waits
cost nothing measurable there either. The small cases are dispatch-bound on the GPU and run slower than on the laptop
(the example, uniform `use_jit`: 1.62 s against 0.82 s), with CG at 1–4 ms per iteration and the eager phases around
it dominating the split, while on the downstream level CG takes 66 % of the rows' time. Cold runs compile a little
longer than on the laptop (the example under `use_jit`: 43 s ragged and uniform, against 31 and 36 s).

**Jit or eager, as the compile column shows it.** On a cold run (empty program caches) the compile wait is most of the
run, and where it sits tells jit from eager. The example above on jax (x64), `use_jit=True`, cold and then again at
the same structure (rows 0 and 1 of the cold run, row 0 of the second):

```
  time  model 2.3e+00 [frame 2.1e-04 sweep 2.2e-02 comp 1.1e-05 resid 2.0e-02 curv 2.3e+00] | grad 6.0e-01 | ref 5.6e+00 ×9 | cg 3.1e+00 (8.4e-02/it) | forms 1.3e+00 | trials 2.8e+00 ×1 [retr 2.7e+00 obj 3.0e-02] | rest 5.5e-01 | setup 1.2e-01 | align 1.7e+01 ×5 | compile curv 2.1e+00 grad 5.1e-01 ref 4.4e+00 cg 3.0e+00 forms 9.4e-01 retr 2.6e+00 rest 5.2e-01 align 1.6e+01
  time  model 1.1e+00 [frame 1.0e+00 sweep 2.1e-02 comp 7.3e-06 resid 2.0e-02 curv 1.3e-02] | grad 6.5e-02 | cg 2.8e-03 (9.2e-04/it) | forms 2.4e-01 | trials 2.9e-02 ×1 [retr 9.8e-03 obj 1.9e-02] | cb 1.0e-01 | rest 1.6e-03 | compile cb 9.9e-02 frame 9.6e-01

  time  model 5.0e-02 [frame 1.3e-04 sweep 1.8e-02 comp 5.6e-06 resid 1.9e-02 curv 1.3e-02] | grad 6.3e-02 | ref 8.7e-01 ×9 | cg 1.8e-03 (1.8e-03/it) | forms 2.9e-01 | trials 3.3e-02 ×1 [retr 1.0e-02 obj 2.2e-02] | rest 1.8e-03 | setup 1.1e-03 | align 4.5e-01 ×5 | compile 0
```

- **`use_jit=True`**: CG is one program, compiled on its first call, so row 0's `cg` carries it (3.0 s here) and a
  second run at the same structure compiles nothing (`compile 0` on every row). The phases around CG stay eager and
  compile their operations the first time they run: mostly on row 0, but also later (above, row 1's first
  `geom.frame`, since row 0 started from the aligned frame, and the display's first error table in `cb`). The ragged
  alignment is a host loop around jitted steps, so its passes are timed (`×5`) and its compile is in `alignment`;
  the uniform alignment is one whole-loop program, compiled inside `alignment` (9.0 s of 9.4 s in the example) with
  `alignment_passes` `None`, and so is the `'cone'` loop inside row 0's `reference`.
- **Eager jax** compiles every operation of a new structure on first use, spread over the phases that run them; a
  rerun at the same structure compiles nothing, but every operation is dispatched one by one.

The cold runs, in seconds (compile = the sum of the compile column, its four largest cells beside it, then `cg`'s
when it is not among them), and the same call again at the same structure:

| case | layer | backend | cold run | compile | the four largest compile cells; `cg` | rerun |
|---|---|---|---|---|---|---|
| the example | ragged | `use_jit` | 36 | 31 | `alignment` 15.8, `reference` 4.5, `cg` 2.9, `trial_retraction` 2.6 | 3.5 |
| | uniform | `use_jit` | 39 | 36 | `alignment` 9.0, `reference` 8.0, `trial_retraction` 5.9, `frame` 4.2; `cg` 1.6 | 0.82 |
| | ragged | eager | 71 | 54 | `alignment` 32.7, `reference` 13.2, `trial_retraction` 2.2, `curvature` 2.1; `cg` 1.7 | 14.0 |
| | uniform | eager | 80 | 73 | `alignment` 34.9, `reference` 28.0, `curvature` 3.3, `trial_retraction` 2.1; `cg` 1.9 | 3.2 |
| d = 6 | ragged | `use_jit` | 17 | 12 | `gradient` 3.8, `trial_retraction` 2.7, `cg` 2.1, `sweep` 1.4 | 4.0 |
| | uniform | `use_jit` | 17 | 15 | `trial_retraction` 5.4, `frame` 5.4, `gradient` 1.4, `reference` 1.1; `cg` 0.8 | 0.95 |
| | ragged | eager | 21 | 11 | `gradient` 3.7, `trial_retraction` 2.7, `frame` 1.4, `sweep` 1.4; `cg` 1.0 | 9.6 |
| | uniform | eager | 18 | 16 | `frame` 5.5, `trial_retraction` 5.5, `gradient` 1.4, `reference` 1.2; `cg` 1.1 | 1.3 |

The warm split is what the timing is for. In the ragged example, `use_jit` brought CG to 0.9 ms per iteration
(numpy: 15 ms; eager jax: 300 ms), yet the run took 3.5 s against numpy's 0.74 s: the eager phases around CG cost
more on jax than on numpy at this size, the exact `pᵀHp` in `forms` (42 % of the rows' time), the `'cone'` loop
(28 %) and `gradient` (12 %). See also [Jit or eager](aligned_padded_frame.md#jit-or-eager) for the alignment's
compiles and memory.

**Caveats.**

- The compile column is `None` on a numpy run, and on a jax whose `jax.monitoring` cannot unregister a listener (the
  jax 0.4.30 floor): the listener is registered for the call and must be removed after it.
- The listener is process-global: compiles in other threads, or in a timed run nested inside a callback, are counted
  too.
- A callback runs between two windows, so its time is the **next** row's `previous_callback` (the last row's is
  `last_callback`): a heavy display appears a row late.
- Work dispatched without a wait is charged to the next phase that waits for it: `-g`, dispatched in the remainder,
  finishes inside `cg` (tiny), and so does any jax work a callback starts without waiting.
- A `Problem` subclass that overrides `local_model` is called as without timing; its whole build is reported in
  `frame`, and `sweep` through `curvature` are `None`.
- A custom geometry's `aligned_padded_frame` is asked for its seconds per pass only when its signature names `timing`;
  otherwise `alignment_passes` and `reference_passes` are `None` (the alignment is still timed whole). Under the
  uniform whole-loop jit they are `None` too ([`aligned_padded_frame.md`](aligned_padded_frame.md#jit-or-eager)).
- The frontend's own setup (building the problem, reducing `x0` and `padded_from` to minimal ranks, building the
  display) runs before the backend's clock starts: `stats['timing']` describes the backend call, the same for a
  raw-`.data` user.

### 4.14 Starting with Gauss-Newton and switching to the exact Hessian

Gauss-Newton is cheap per step but converges only linearly on a nonzero-residual fit; the exact Hessian (§4.12) gives
the superlinear tail but costs 3–4× per apply and is indefinite far from a minimizer. A Gauss-Newton start followed by
exact steps is the natural pattern, and `newton_cg(..., hessian='switch')` picks the moment inside one call: it starts
with Gauss-Newton, checks after each clean step whether the exact model would have predicted the actual change of the
objective better, and switches to the exact Hessian for good once it clearly would have. The frontend and the backend
take the same keywords.

**The two models.** Along a step `s` from `x` (`s = αp` on the line search, `s = p` in the trust region, where
`α = 1`), each quadratic model predicts the change of the objective:

```
Δf       = f(R_x(s)) − f(x)                    the actual change                (the row's delta_f)
m_GN(s)  = ⟨g, s⟩ + ½⟨s, H_GN s⟩               Gauss-Newton's predicted change  (the row's α·slope + ½α²·pHp)
m_N(s)   = m_GN(s) + ½⟨s, Π Lᵀ K_N L Π s⟩      the exact model's                (one curvature product)
```

with `K_N = K_{Aᵀω²r} − K_{grad f}` and `L` the lift of §4.12 (on a corewise geometry `L = I` and
`K_N = K_{Aᵀω²r}`). Gauss-Newton's prediction is already on the history row; the exact model's costs one curvature
product, the **tracking product**.

**The rule.**

```
e_GN   = |Δf − m_GN(s)|                        how far Gauss-Newton missed
e_N    = |Δf − m_N(s)|                         how far the exact model missed
hit    : e_N < τ·e_GN on a countable step      τ = switch_tau, default 0.25
switch : k consecutive hits                    k = switch_k, default 2; for good, from the next point on
```

In words, with the defaults: **the exact model must have predicted the objective at least four times better than
Gauss-Newton, on two clean steps in a row.** Why it works: near a minimizer with a nonzero residual, along a
second-order retraction, `e_GN = ½|⟨s, Π Lᵀ K_N L Π s⟩| + O(‖s‖³)` (the term Gauss-Newton drops) while
`e_N = O(‖s‖³)`, so `e_N / e_GN → 0`; in the transient far from it the ratio is about 1 or more, and a threshold well
below 1, required on consecutive steps, rarely fires early. Equivalently, with `δ = Δf − m_GN(s)` (Gauss-Newton's
signed miss) and `c = m_N(s) − m_GN(s)` (the curvature term's prediction of it), a hit is `|δ − c| < τ·|δ|`: the
curvature term predicted Gauss-Newton's own error to a relative accuracy `τ`.

**What counts, what resets.** A Gauss-Newton-phase step is **countable** iff it took the full model step: accepted,
not the `−g` fallback, and either an accepted trust-region trial or an Armijo step with `α = 1`. Only countable steps
add to the count; anything that suggests the local quadratic model went wrong starts the evidence over:

| row | the count |
|---|---|
| countable, a hit | plus one |
| countable, a miss, or both errors at the rounding floor (below) | reset to 0 |
| a rejected trust-region trial; a line-search backtrack (`α < 1`); the `−g` fallback (CG's step was not a descent direction); a step forced by `on_line_search_failure='accept'` | reset to 0 (by default without a tracking product) |
| CG stopped on the trust region's boundary, truncated, or at its iteration cap | counted like any step: both models are judged at the step actually taken |
| the converged row | unchanged (no step) |

Tracking starts at iteration 0, with no delay.

**The rounding gate.** `Δf` is a difference of two objective values, so below the objective's floating-point
resolution neither error can be measured. The gate's floor is a hundred times the trust region's `objective_floor`
threshold (§4.12):

```
resolution  = ε·(‖ω⊙y‖·‖ω⊙r‖ + |f|)          ε: the residual dtype's machine epsilon
floor       = 100 · 10 · resolution          the trust region's objective_floor is at 10 · resolution
informative = e_GN and e_N finite, and max(e_GN, e_N) > floor
hit         = informative and e_N < τ·e_GN
```

Both errors at the floor is uninformative, a reset; `e_GN` at the floor with `e_N` above it is a miss, also a reset;
`e_N` at the floor with `e_GN` well above it is a hit. In float32 (`use_jit=True` without jax x64) the floor is
`1.2e-4·(‖ω⊙y‖·‖ω⊙r‖ + |f|)`, so only large early steps are informative and the switch rarely fires: **enable jax x64
with `hessian='switch'`.**

**At and after the switch.** The switch is one-way and lands on a fresh linearization: only accepted steps count, so
the row that switches is followed by a new point, which is built in exact mode (no model changes in the middle of a
point). Before the switch the run is the Gauss-Newton run with the same step control, step for step (`ρ` and the
radius rules read `m_GN`); from the switch on it is the exact run resumed at that point (`ρ` reads `m_N`). The trust
radius carries over, `Δ0` and the cap `10⁴·Δ0` stay iteration 0's, and the references `‖g0‖` are resolved once, at
iteration 0, for both phases (§5). The CG forcing power is `cg_forcing_power` (default `0.5`) before the switch and
`switch_cg_forcing_power` (default `1.0`, the exact Hessian's q-quadratic tail) after it. Each `newton_cg` call starts
in Gauss-Newton with a count of 0: nothing carries across calls.

**The options.** Four keyword-only options after `timing`, each `None` by default and each a structural error, raised
before any work, when set without `hessian='switch'`:

| keyword | default | meaning |
|---|---|---|
| `hessian` | `'gauss_newton'` | `'switch'` (case-insensitive): Gauss-Newton, then the exact Hessian |
| `switch_tau` | `None` = `0.25` | the hit threshold `τ`, a number in `[0, 1)`; `0` tracks and never switches (the Gauss-Newton run with the same step control, bit for bit, plus the tracking) |
| `switch_k` | `None` = `2` | the consecutive hits `k` that switch, an int `≥ 0`; `0` is exact from iteration 0 (the exact run with `cg_forcing_power=switch_cg_forcing_power`) |
| `switch_cg_forcing_power` | `None` = `1.0` | the CG forcing power from the switch on; `cg_forcing_power` stays the Gauss-Newton phase's |
| `switch_track` | `None` = `'countable'` | the steps the tracking product runs on: the countable ones, or `'all'` (case-insensitive), every Gauss-Newton-phase step, which reports `e_N` on the non-countable steps too, at one product each, and moves no count, so the switch fires at the same row |

An option the run would not read raises too: `switch_tau` or `switch_track` with `switch_k=0` (nothing is tracked),
`switch_k` or `switch_cg_forcing_power` with `switch_tau=0` (nothing switches). The rest is exact mode's: the kind and
the geometry must implement the exact Hessian (checked at the call), the default step control is the trust region
(the line search works too), `curvature_damping` applies to every exact build, the tracking ones included, and
`tr_radius0` sets the first radius. A local model has one Hessian, so `Problem.local_model` and the `fitting`
`*_model` factories do not take `'switch'`; `least_squares_problem(hessian='switch')` sizes `chunk_size='auto'` as
`'exact'`.

**A worked run.** §4.13's problem at rank 2 (a noisy rank-3 target, probe derivatives of order 1; `x_old` is five
Gauss-Newton steps from a random start), with the warm-start reference, so the stop is relative to `x_old`'s own
gradient:

```python
>>> warm = dict(order=1, max_newton=40, g0_reference='warm_start')
>>> _, sw = topt.newton_cg(t3m.MANIFOLD, 'probe_derivatives', probes, noisy, x_old, hessian='switch', **warm)
>>> sw['hessian'], sw['switch_track'], sw['stop_reason'] in ('gradient', 'objective_floor')
('switch', 'countable', True)
>>> j = sw['switched_at']                                     # the first row whose model is exact
>>> [row['hessian'] for row in sw['history'][j - 1:j + 1]]    # the row that switched, and the next
['gauss_newton', 'exact']
>>> fired = sw['history'][j - 1]
>>> fired['switch_fired'], fired['switch_count'], fired['switch_e_n'] < 0.25 * fired['switch_e_gn']
(True, 2, True)
>>> fired['pKp'], fired['switch_pKp'] is not None             # the tracking product is not in the row's pHp
(None, True)
>>> _, gntr = topt.newton_cg(t3m.MANIFOLD, 'probe_derivatives', probes, noisy, x_old, step_control='trust_region',
...                          **warm)                          # Gauss-Newton under the same step control
>>> sw['newton_steps'] < gntr['newton_steps'] / 2
True

```

The same run with `verbose=True` on the maintainer's laptop (numpy; the header lines only, the error tables left out;
this block is not checked):

```
iter  0 | obj 2.547e+05  ‖g‖ 9.84e+02 (1.0e+00·g₀) | CG 1/200 tol 4.9e+02 resid 2.3e+02 ✓ | TR Δ 2.51e+00 ✓ ‖Δx‖/‖x‖ 1.9e-02 | Δf -1.07e+03 ρ 0.86 | 0.03s | sw 1.3e-02 ×1
iter  1 | obj 2.537e+05  ‖g‖ 4.17e+02 (4.2e-01·g₀) | CG 1/200 tol 2.1e+02 resid 1.2e+02 ✓ | TR Δ 2.51e+00 ✓ ‖Δx‖/‖x‖ 8.7e-03 | Δf -2.03e+02 ρ 0.85 | 0.03s | sw 3.0e-03 ×2 ⇒ exact
iter  2 | obj 2.535e+05  ‖g‖ 2.43e+02 (2.5e-01·g₀) | CG 2/200 tol 6.0e+01 resid 1.9e+01 ✓ | TR Δ 2.51e+00 ✓ ‖Δx‖/‖x‖ 7.2e-03 | Δf -9.75e+01 ρ 1.00 | 0.05s | κ -3.5e-01 σ 5.8e+01 ε 1.3e-04
iter  3 | obj 2.534e+05  ‖g‖ 1.87e+01 (1.9e-02·g₀) | CG 4/200 tol 3.5e-01 resid 1.9e-01 ✓ | TR Δ 2.51e+00 ✓ ‖Δx‖/‖x‖ 3.9e-04 | Δf -4.24e-01 ρ 1.00 | 0.08s | κ -1.7e-02 σ 5.8e+01 ε 1.3e-04
iter  4 | obj 2.534e+05  ‖g‖ 1.91e-01 (1.9e-04·g₀) | CG 8/200 tol 3.7e-05 resid 2.8e-05 ✓ | TR Δ 2.51e+00 ✓ ‖Δx‖/‖x‖ 5.7e-06 | Δf -6.38e-05 ρ 1.00 | 0.14s | κ -3.0e-01 σ 5.8e+01 ε 1.3e-04
iter  5 | obj 2.534e+05  ‖g‖ 2.82e-05 (2.9e-08·g₀) | CG 14/200 tol 8.1e-13 resid 5.3e-13 ✓ | TR Δ 2.51e+00 ✓ ‖Δx‖/‖x‖ 6.4e-10 | Δf -2.91e-11 ρ 26.73 | 0.23s | κ -1.4e-02 σ 5.8e+01 ε 1.3e-04
iter  6 | obj 2.534e+05  ‖g‖ 1.97e-10 (2.0e-13·g₀) | converged (‖g‖ ≤ gtol·‖g₀‖, g₀ = warm start)
```

The exact model's miss was 78 times smaller than Gauss-Newton's on row 0 (`sw 1.3e-02 ×1`, a hit) and 337 times
smaller on row 1 (`sw 3.0e-03 ×2 ⇒ exact`), so row 2 is the first exact row (`switched_at = 2`) and shows exact mode's
`κ σ ε` cells. The run took 6 Newton steps and 30 CG iterations, against 25 steps and 90 CG iterations for
Gauss-Newton with the same trust region (29 steps with its line search) and 5 steps and 27 CG iterations for the exact
Hessian at power 1.0 from the start: one Newton step and two tracking products more than choosing the exact Hessian
in advance. From the random rank-2 start that `x_old`'s five steps began at, the Gauss-Newton phase is longer: in the
transient the exact model's miss was 3.5–9.6 times *larger* (`sw 9.6e+00 ×0`: three misses), then came two hits (the
run used `switch_track='all'`; every Gauss-Newton-phase row here is countable, so the default prints the same; row 0's
CG stopped on the boundary, `∂`, and counted like any step):

```
iter  0 | obj 3.707e+06  ‖g‖ 1.85e+04 (1.0e+00·g₀) | CG 1/200 tol 9.3e+03 resid 5.2e+03 ∂ | TR Δ 5.13e+01 ✓ ‖Δx‖/‖x‖ 7.4e+00 | Δf -3.62e+05 ρ 0.76 | 0.03s | sw 9.6e+00 ×0
iter  1 | obj 3.345e+06  ‖g‖ 3.17e+04 (1.7e+00·g₀) | CG 1/200 tol 1.6e+04 resid 8.1e+03 ✓ | TR Δ 1.03e+02 ✓ ‖Δx‖/‖x‖ 1.8e+00 | Δf -1.27e+06 ρ 1.07 | 0.03s | sw 8.1e+00 ×0
iter  2 | obj 2.080e+06  ‖g‖ 3.17e+04 (1.7e+00·g₀) | CG 1/200 tol 1.6e+04 resid 7.4e+03 ✓ | TR Δ 1.03e+02 ✓ ‖Δx‖/‖x‖ 9.9e-01 | Δf -1.52e+06 ρ 1.15 | 0.02s | sw 3.5e+00 ×0
iter  3 | obj 5.621e+05  ‖g‖ 1.37e+04 (7.4e-01·g₀) | CG 1/200 tol 6.9e+03 resid 3.5e+03 ✓ | TR Δ 1.03e+02 ✓ ‖Δx‖/‖x‖ 3.4e-01 | Δf -2.89e+05 ρ 1.07 | 0.03s | sw 8.1e-02 ×1
iter  4 | obj 2.727e+05  ‖g‖ 3.17e+03 (1.7e-01·g₀) | CG 1/200 tol 1.3e+03 resid 1.2e+03 ✓ | TR Δ 1.03e+02 ✓ ‖Δx‖/‖x‖ 8.5e-02 | Δf -1.64e+04 ρ 0.96 | 0.03s | sw 8.9e-02 ×2 ⇒ exact
iter  5 | obj 2.562e+05  ‖g‖ 1.48e+03 (8.0e-02·g₀) | CG 3/200 tol 1.2e+02 resid 5.7e+01 ✓ | TR Δ 1.03e+02 ✓ ‖Δx‖/‖x‖ 3.3e-02 | Δf -2.84e+03 ρ 1.00 | 0.06s | κ -2.1e-01 σ 5.8e+01 ε 1.3e-04
iter  6 | obj 2.534e+05  ‖g‖ 6.22e+01 (3.4e-03·g₀) | CG 6/200 tol 2.1e-01 resid 7.3e-02 ✓ | TR Δ 1.03e+02 ✓ ‖Δx‖/‖x‖ 1.9e-03 | Δf -7.16e+00 ρ 1.00 | 0.12s | κ -1.7e-01 σ 5.8e+01 ε 1.3e-04
iter  7 | obj 2.534e+05  ‖g‖ 9.10e-02 (4.9e-06·g₀) | CG 11/200 tol 4.5e-07 resid 2.0e-07 ✓ | TR Δ 1.03e+02 ✓ ‖Δx‖/‖x‖ 2.6e-06 | Δf -1.37e-05 ρ 1.00 | 0.19s | κ -1.1e-01 σ 5.8e+01 ε 1.3e-04
iter  8 | obj 2.534e+05  ‖g‖ 2.32e-07 (1.2e-11·g₀) | converged (‖g‖ ≤ gtol·‖g₀‖, g₀ = warm start)
```

**What it reports.** The history rows (`backend.optimizers.NewtonInfo`):

| field | on | value |
|---|---|---|
| `hessian` | every row | the model that solved the row's step: `'gauss_newton'` before the switch, `'exact'` from it on (outside switch mode, the run's mode, as before) |
| `switch_e_gn` | Gauss-Newton-phase rows with a step | `e_GN` |
| `switch_pKp` | tracked rows | the tracking product `⟨p, Lᵀ K_N L p⟩`; not part of the row's `pHp`, which stays Gauss-Newton's |
| `switch_e_n` | tracked rows | `e_N`, with `m_N(s) = m_GN(s) + ½α²·switch_pKp` |
| `switch_count` | Gauss-Newton-phase rows | the count after this row (0: a reset); the converged row carries it |
| `switch_fired` | every row of a switch run | `True` on the row whose count reached `k`, else `False` |

The five `switch_*` fields are `None` outside switch mode and appear in `stats['history']` only in switch mode, so the
rows of every other mode keep their keys. On an exact-phase row `switch_fired` is `False` and the other four `None`;
the row carries exact mode's `pKp`, `lift_sigma_min` and `lift_eps`. On a Gauss-Newton-phase row those three are
`None` and the callback's `info.lm` is the Gauss-Newton view of the point's exact build: such a row equals the pure
Gauss-Newton run's row apart from the `switch_*` fields and `wall_time`. (The view's `gradient` recomputes the misfit
transpose; the loop itself reads the exact build's.)

`stats` gains three keys, in switch mode only: `hessian` (`'switch'`), `switch_track` (the resolved `'countable'` or
`'all'`; `None` with `switch_k=0`) and `switched_at`, the index of the first row whose model is exact (`len(history)`
when the switch fired on the last row; `0` with `switch_k=0`; `None` if it never fired).

**The display.** With `verbose=True` (or `make_newton_display` as the backend callback) each Gauss-Newton-phase step
row's header ends with `sw`, the ratio `e_N / e_GN` (`—` where no product ran, as on a reset row under the default
`switch_track`), then `×` and the count; the switching row adds `⇒ exact`, and a last row's `stop:` comes after it.
The exact-phase rows show `κ σ ε` (§4.12), and every other mode prints as before. With `timing=True` (§4.13) the
tracking product is its own phase, `tracking` (the `track` cell of the `time` line), and every build of a switch run
times its curvature data under `curvature`, the Gauss-Newton phase's included: a Gauss-Newton-phase row's tracking
overhead is `curvature + tracking`, with their compile twins on jax.

**What it costs.** Each Gauss-Newton-phase point is linearized once, in exact mode: one misfit transpose, as in either
pure mode, plus the geometry's curvature data (on the manifold `2d − 1` small SVDs and one small T3-to-T3 sweep,
nothing `W`-sized). CG, the step control and the row use its Gauss-Newton view, so the steps are Gauss-Newton's.
Each tracked step adds one tracking product, about an exact Hessian-apply less a Gauss-Newton one: two to three
Gauss-Newton applies, by §4.12's 3–4×. Against a Gauss-Newton step, which costs one apply per CG iteration plus a
transpose, the trial retractions and their objectives, that is a few percent when CG runs many iterations per step and
a large fraction when it runs one or two (as in the worked run's Gauss-Newton phase). Under `use_jit` CG compiles at
most two programs per call: the Gauss-Newton one, which the Gauss-Newton phase shares with a pure Gauss-Newton run at
the same structure, and the exact one when the switch fires (measured: two at each level of a zero-padded
continuation, none when a level is run again). The per-point build and the tracking product run eagerly in the host
loop, like exact mode's build and `pᵀHp`, so on a small jax problem their dispatch, not their arithmetic, can dominate
their time.

**Measured downstream.** A rank continuation of a Darcy-flow surrogate (directional derivatives up to order 4 at 10
points × 100 directions, six levels to rank 24, `‖g‖ ≤ 10⁻³·‖g0‖` on the zero reference, at most 300 steps a level;
jax x64, `use_jit`, an A100 GPU). Ladder totals, apply-equivalents in Gauss-Newton applies at measured cost:

| `newton_cg` | Newton steps | CG iterations | apply-equivalents | wall s, less compile and display | levels at the cap |
|---|---|---|---|---|---|
| Gauss-Newton, line search (the default) | 1,625 | 24,682 | 24,682 | 427 | 4 of 6 |
| Gauss-Newton, trust region | 1,618 | 27,023 | 27,023 | 436 | 4 of 6 |
| `hessian='exact'` | 267 | 10,751 | 33,309 | 275 | 0 |
| `hessian='switch'` | 223 | 6,303 | 19,794 | 177 | 0 |

The switch fired on every level after 5–8 Gauss-Newton steps, its tracking took 3 % of its time, and at the rank level
the validation error selected its errors were no worse than Gauss-Newton's. Restarted from the aligned frame
(`padded_from`) it fired everywhere too, but took 303 Newton steps to the exact Hessian's 231; on a 20 × 1000 ladder
of 11 levels the two were even, at both starts.

**When to use it, and when not.**

- **Use it** on a nonzero-residual fit (noisy data, or a target the ranks cannot represent) when you would otherwise
  run Gauss-Newton first and then the exact Hessian, and do not know when to change.
- **Each call starts in Gauss-Newton**, by design: after a rank increase the run may have left the quadratic basin.
  So in a rank continuation every level pays for at least `k` tracked Gauss-Newton steps, with their exact-mode
  builds, before it switches. Where the exact Hessian is known to pay off from a level's first step (a warm-started
  level near its minimizer), `hessian='exact'` saves them: in the worked run, 5 steps against 6.
- **It may never fire**, and then the run is Gauss-Newton plus the tracking cost: safe, but not free. That happens
  where the curvature term vanishes (the manifold at `d = 1`, where every slot is flat, §4.12, and `e_N = e_GN`);
  where the residual is zero at the solution (noise-free data at the true ranks: both errors shrink to the floor
  together, and Gauss-Newton is fast there anyway); where the retraction is only first order at the point (near small
  singular values, and at a zero-padded start, whose new directions the lift drops: `e_N` keeps an `O(‖s‖²)` part, and
  the switch can hold off until the new singular values have grown, though on the downstream ladders above it fired
  on every zero-padded level); and in float32 (the gate above). The rows cannot say which: the lift diagnostics are
  `None` on Gauss-Newton-phase rows, the price of those rows equalling the pure Gauss-Newton run's.

**Where the rule comes from.** The rule, its defaults and the name `'switch'` come from the maintainer's
[`nicks-optimizers`](https://github.com/NickAlger/nicks-optimizers) (`hessian="switch"` in its trust-region and
line-search Newton solvers, as of commit `1ab10dc`, 2026-11-11; its experiments are in `experiments/gn_newton_switch/`).
Here it differs in six ways:

- The rounding gate is dtype-aware and tests both errors. The source's is float64's `100·10·ε·max(1, |f|)`, tested on
  `e_GN` alone; testing both moves the count exactly as testing `e_GN` alone would.
- The forcing power after the switch defaults to `1.0`, for the exact Hessian's q-quadratic tail (§4.12); the source
  settled on `0.5` after its experiments (`switch_cg_forcing_power=0.5` gives that).
- Two more kinds of row reset the count: a line-search backtrack and a step forced by `on_line_search_failure='accept'`
  (which the source does not have). The `−g` fallback resets in both.
- By default the tracking product runs on countable steps only; the source runs it on every Gauss-Newton-phase
  iteration, rejected trials included, as `switch_track='all'` does.
- `switch_tau = 0` and `switch_k = 0` are allowed (the source requires `0 < τ < 1` and `k ≥ 1`).
- There is no resume state: each call starts in Gauss-Newton.

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
  | `'warm_start'` | the initial gradient norm `‖g(x0)‖` — the 2026.2.0 default; with a start frame (`start_frame` / `padded_from`), `‖g‖` at that frame |
  | `'cone'` | the gradient norm at the zero tensor of `x0`'s ranks on the frame **aligned with the data gradient there** (options `cone_*`), like-for-like with the gradient norms the stop compares; use it with a looser `gtol_rel` (about `1e-4`). Manifold geometries only; [`aligned_padded_frame.md`](aligned_padded_frame.md) |

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

  *The start frame after a rank increase.* `newton_cg(..., padded_from=x_old)` starts from the frame of `x0`
  (`x_old` zero-padded) whose new directions are aligned with the misfit gradient, and `start_frame=F` starts from
  a frame you built; `cone_max_passes=0` / `padded_max_passes=0` use the hint-free padded frame instead. The
  `'cone'` reference with zero passes is measured at a different frame from `'zero'`'s, so the two differ. See
  [`aligned_padded_frame.md`](aligned_padded_frame.md).

  A **vanishing** zero reference (e.g. all-zero data) is never replaced silently: by default it falls
  back to the warm start with a `RuntimeWarning`, and an explicit `'zero'` raises. Every history row
  records which reference was used:

  ```python
  >>> _, st = topt.newton_cg(t3m.MANIFOLD, 'apply', ww, data, x0, max_newton=30)
  >>> st['history'][0]['g0_reference'], st['stop_reason']     # the default on the manifold
  ('zero', 'gradient')
  >>> _, st = topt.newton_cg(t3m.COREWISE, 'apply', ww, data, x0c, max_newton=2)
  >>> st['history'][0]['g0_reference']                        # ... and on corewise
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
- **The Newton schedule is yours.** Gauss-Newton (the default), the exact Hessian (§4.12) or the automatic
  `hessian='switch'` (§4.14), the forcing power, the reference, the caps — every knob is a per-call `newton_cg`
  keyword, so a continuation loop can run each level with its own schedule (e.g. Gauss-Newton until the step stalls,
  then exact; exact at every warm-started level; or `'switch'` at every level, which changes over inside each call
  once the exact model predicts the objective clearly better). Which default is best is an open, empirical question;
  on the measured restarts the exact Hessian paid off whenever the fit had a nonzero residual.
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
  weights and the Grasedyck–Kramer tangent metric).
- **Implementation rationale** (memory tradeoffs, einsum paths, deferred work):
  [`contributor/fitting_internals.md`](contributor/fitting_internals.md); for the exact Hessian (the lift,
  the two-hole sweeps, the masking rules, the measurements) [`contributor/curvature_internals.md`](contributor/curvature_internals.md).
