# Geometry-based fitting — restructure plan

*Design locked with Nick (2026-06-19), branch `fitting`. This supersedes the corewise structure in
[`docs/fitting_plan.md`](fitting_plan.md) (the six parallel `*GaussNewtonModel` classes + ~38 backend
functions): tangent and corewise fitting are unified under an explicit **`Geometry`** abstraction, with
`T3Tangent` as the universal tangent type and **geometry-agnostic optimizers**. The cost of the refactor
is accepted; this is the target structure, chosen for mathematical coherence, code simplicity, and
user utility — not to preserve the existing code.*

## 1. The core idea (one screen)

Manifold fitting and corewise fitting are **not two feature-families — they are two manifolds you
optimize on**, both representing the same tensor:

- **Manifold geometry** — optimize on the fixed-rank manifold `M`. Tangents in `T_xM` (orthonormal
  frame, gauged), the Hilbert–Schmidt metric, the manifold retraction.
- **Corewise geometry** — optimize on the core **parameter space** `P` (an over-parametrized cover of
  `M`, `π: cores ↦ X(cores)`). Tangents in `T_cores P` (the raw cores `(U,G,G,G)`), the Euclidean
  metric, **additive** retraction (`cores += p`).

Three observations make this a code structure, not just a picture:

1. **`T3Tangent` is the universal tangent — in any frame.** Verified: at the non-orthonormal
   `(U,G,G,G)`, `to_dense` equals the sum-of-single-core-swaps **exactly (err = 0.0)**, and
   `inner`/`norm`/`apply`/`apply_transpose`/`+`/`·` are all correct. A corewise gradient *is* a
   `T3Tangent` at `(U,G,G,G)`, not a raw tuple. Only the *embedding* operations (retraction, gauge
   projection, ambient→tangent projection) require orthonormality.
2. **`T3Tangent.inner` is the universal metric.** It computes `corewise_dot` of the variations either
   way, and that *is* the right optimization metric for both: it equals HS on the manifold's
   orthonormal-gauged frame (Appendix A.3 identity) and is trivially Euclidean for raw cores.
3. **The only genuine manifold↔corewise differences are three chart-level choices** — the base, whether
   to gauge-project (`Π`), and the retraction — and they are *linked* (orthonormal frame ⟺ `Π`; raw
   cores ⟺ no-`Π`). Everything else (the GN-model assembly, the sampling-kind probing, the base-sweep
   reuse) is identical.

So the code wants to be: **probing primitives (bare `𝒥`/`𝒥ᵀ`, per sampling kind) → a `Geometry` that
bundles (base, project, retract) → one generic GN model → generic optimizers.** The §6.3 substitution
`(O,P,Q)→G` stops being a trick and becomes *the change of geometry* (the relation between the gradient
on `P` and the gradient on `M`).

## 2. Mathematical foundation (the precise picture)

- **Two spaces, not two charts.** `M` = bounded-rank tensors (Riemannian). `P` = the core parameter
  space; `π: P → M` is a submersion (the gauge group acts on its fibers). Manifold fitting optimizes
  `φ` on `M`; corewise fitting optimizes `φ∘π` on `P`. The corewise tangent is a tangent to `P`; its
  pushforward `dπ` (= `T3Tangent.to_dense`) lands in `T_xM`, with the **gauge directions in its kernel**
  (core-variations that change the representation but not the tensor).
- **The retraction is vector addition in the chart.** Corewise: `cores ← cores + p`, a flat Euclidean
  step. `π` is multilinear, so `X(cores + t·p)` is a genuine polynomial curve on `M` — an exact
  retraction whose curvature is the higher-order multilinear cross-terms. The over-parametrization is
  exactly what makes the chart flat and the constraint (bounded rank) automatic.
- **The corewise Gauss–Newton Hessian is gauge-singular.** Since `𝒥(gauge) = apply(dX=0) = 0`, the gauge
  directions lie in `ker(𝒥ᵀ𝒥)`; the gradient itself is gauge-orthogonal (`⟨𝒥ᵀr, gauge⟩ = ⟨r, 𝒥 gauge⟩ =
  0`) but the *operator* is rank-deficient. First-order methods (Adam, L-BFGS) tolerate this; Newton/GN-CG
  must regularize or truncate. `Π` is precisely the cure: `Π𝒥ᵀ𝒥Π` is non-degenerate on the gauged tangent
  space — which is why Newton-CG is natural on the manifold side. The matched pair maps cleanly onto a
  method choice:

  | | frame | metric | retraction | GN Hessian | natural optimizer |
  |---|---|---|---|---|---|
  | **manifold** (Π) | orthonormal `(U,O,P,Q)` | ambient / HS | manifold retract | non-degenerate | Newton-CG |
  | **corewise** (no Π) | raw `(U,G,G,G)` | Euclidean | additive in cores | gauge-singular | Adam / L-BFGS |

## 3. Why this structure (weighing the evidence)

- **Riemannian-optimization libraries (Manopt, Pymanopt, Geomstats, McTorch)** are built *exactly* on
  this: a `Manifold` (retraction, inner, gradient conversion) + a `Problem` (cost/grad/hess) + a
  **generic** solver. Our "corewise" is literally their **Euclidean manifold**. The geometry abstraction
  is the field's proven organization, not our invention.
- **Tensor libraries (TensorLy, tntorch, t3f, TensorNetwork)** mostly do corewise via **autodiff** (cores
  as parameters → `∂loss/∂cores` free → Adam/SGD) or **ALS** (exact per-core block least-squares). t3f is
  closest to us — it has both Riemannian and Euclidean paths, but as *two separate code paths*, not a
  unified geometry; its Euclidean path is just autodiff.
- **Decisive point:** wherever autodiff exists, the corewise gradient is *free*. So framing corewise as a
  "compute core gradients" feature is redundant — autodiff does it better. Our corewise earns its place
  only as **(a)** the Euclidean geometry inside a unified framework, **(b)** a matrix-free Gauss–Newton
  operator (`JᵀJ`-vector products, awkward for autodiff via double-backprop), and **(c)** autodiff-free
  numpy. All three *are* the geometry-abstraction framing. So the abstraction isn't only cleaner code —
  it's the correct *positioning* of the feature.
- **Net on the three criteria:** coherence — strongest (it is the established framework, and corewise is
  its Euclidean instance); simplicity — strongest *for the whole system* (fewer backend functions, six
  classes → one model + two geometries, and — the real win — **optimizers written once, not per
  geometry**); utility — highest (any optimizer × any geometry × any sampling; extensible to custom
  geometries/metrics/retractions, the framework's whole value proposition).

## 4. Target architecture

```
probing.py            bare 𝒥 / 𝒥ᵀ from_sweep, per sampling kind   (shared by both geometries)
   │
manifold.py           T3Tangent (universal tangent)  +  Geometry {Manifold, Corewise}
   │
fitting.py            one generic GaussNewtonModel(geometry, x, data)  (+ sampling-kind factories)
   │
optimizers.py         newton_cg / lbfgs / gradient_descent / trust_region   (geometry-agnostic)
```

```python
# manifold.py -- stateless geometry singletons (Manopt-style; the point lives in the model)
class ManifoldGeometry:                # optimize ON the fixed-rank manifold M (Riemannian)
    def base(self, x):        return t3_orthogonal_representations(x)[0]      # orthonormal frame
    def project(self, v):     return v.orthogonal_gauge_projection()         # Π  (raw cotangent -> Riemannian grad)
    def retract(self, x, p):  ...                                            # today's T3Tangent.retract logic
    def project_dense(self, x, T):  ...                                      # HS projection ambient -> T_xM (manifold-only)

class CorewiseGeometry:                # optimize ON the core parameter space P (Euclidean)
    def base(self, x):        return T3Basis(x.tucker_cores, x.tt_cores, x.tt_cores, x.tt_cores)   # (U,G,G,G)
    def project(self, v):     return v                                       # identity
    def retract(self, x, p):  return TuckerTensorTrain(x.tucker + p.dU, x.tt + p.dG)               # additive

# fitting.py -- one model, generic over geometry; sampling kind by factory
model = fitting.apply_model(geometry, x, ww, residual)     # entries_model / probe_model likewise
#   .objective_value          scalar c = ½‖r‖²
#   .gradient                 T3Tangent at geometry.base(x)        =  geometry.project(𝒥ᵀ r)
#   .gn_hessian(p)            T3Tangent -> T3Tangent               =  geometry.project(𝒥ᵀ 𝒥 geometry.project(p))
#   .evaluate(p)              scalar  =  c + g.inner(p) + ½‖𝒥(geometry.project(p))‖²

# optimizers.py -- one implementation each, any geometry
x_opt = optimizers.newton_cg(geometry, model_builder, x0)   # Riemannian on M, or regularized GN on P
x_opt = optimizers.lbfgs(geometry, model_builder, x0)
```

The model's three methods are generic; the sampling kind's bare `𝒥`/`𝒥ᵀ` (the `*_from_sweep` primitives)
are bound at construction; the base sweep is cached on the model (the reuse). The `2×3×4 ≈ 24`
tangent/corewise backend functions collapse to **6 bare probing primitives + the geometry's `project`
(2) + 3 generic assembly steps**; the **6 frontend classes → one model + two geometries**.

**Matched pair, preserved more safely.** The geometry *constructs* `base` and binds `project` together,
so "Π on non-orthonormal cores" is unrepresentable — there is no flag to set wrong. Crossing is also
caught by the same-base identity guard (a corewise tangent lives at the `(U,G,G,G)` basis object, a
manifold tangent at the orthonormal one).

## 5. `T3Tangent` restructuring

**Keep the definition `(T3Basis, T3Variations)` — do not change it.** A tangent as *coordinates in a
frame* is universal: the same object is correct in both geometries (the frame is orthonormal or raw).
Carrying the frame is a feature — it makes the tangent self-contained (`to_dense`/`inner` need no
context) and lets `+` enforce the same-tangent-space guard. (The pure-Manopt alternative — bare
`T3Variations`, frame passed to every op — was considered and rejected: it discards that
self-containment and guard for no gain.)

**The restructuring is of the method surface, not the data.** The deciding question is *"intrinsic to a
coordinate vector, or a chart/embedding choice?"*

| `T3Tangent` member | verdict | why |
|---|---|---|
| `+ − ·`, `_check_same_tangent_space` | **keep** | linear structure; the same-base invariant is the tangent's |
| `inner`, `norm` | **keep** | the *coordinate* metric — universal (see §5.1) |
| `to_dense`, `to_t3` | **keep** | realization; frame-faithful, geometry-independent |
| `apply`/`probe`/`entries` (+ transposes) | **keep** | bare `𝒥`/`𝒥ᵀ` sampling; no metric/gauge |
| `shape`, `ranks`, `stack`/`unstack` | **keep** | structure |
| `randn`, `zeros` | **keep**, **drop** their `apply_gauge_projection=` | raw constructor; gauging is `geometry.project` |
| `retract` | **move → `Geometry.retract(x, p)`** | the chart's "how to move" — the defining difference |
| `orthogonal_gauge_projection` / `oblique_gauge_projection` | **move → `Geometry.project`** | `Π` is the manifold's gradient map; identity for corewise; invalid on a non-orthonormal frame |
| `project` (static, dense → tangent) | **move → `ManifoldGeometry.project_dense`** | HS projection ambient→`T_xM`; manifold-only |
| `is_gauged`, `T3Basis.is_orthogonal` | **keep as checkers** | numerical property checks (house philosophy); the geometry *uses* them |

So exactly three operations leave `T3Tangent` — **retract, gauge-projection, project-from-ambient** — the
embedding-dependent triple. The backend algorithms for these already live in `tangent_operations.py`; the
geometry classes are thin bundlers that select them.

**The principle (the answer to "how much should a tangent know about its geometry"):** a tangent knows
**its frame and the frame-faithful operations on its coordinates** (it is a vector, has a coordinate
metric, realizes to a tensor, samples). It does **not** know **how its geometry moves or projects** —
retraction and gauge/ambient projection are the geometry's, applied from outside.

### 5.1 Why the metric stays on the tangent (the one subtle call)

Differential-geometrically the metric is *the* Riemannian structure, so it "should" be the geometry's.
But `T3Tangent.inner` computes `corewise_dot` — the Euclidean structure on the *coordinate* arrays, which
every coordinate vector space has intrinsically — and **this single coordinate metric is the correct
optimization metric for both geometries** (HS on the orthonormal-gauged frame; Euclidean on raw cores). A
separate `geometry.inner` would be *identical* in both cases, so the geometry's metric simply *is*
`T3Tangent.inner`. The geometry's only metric-determining choice is the *frame* (`base(x)`), which fixes
what `corewise_dot` *means* ambient-wise; the metric *computation* is intrinsic to the vector. Document
`inner` honestly: "the coordinate inner product — Hilbert–Schmidt when the frame is orthonormal and
gauged, Euclidean otherwise."

## 6. Design invariants

1. **Matched pair stays structural.** `Π`/no-`Π` is never a flag; it is bundled into the geometry with the
   base it is valid for. Corewise composes the bare probing primitives; manifold composes `Π` around them.
2. **Metric is one computation** (`corewise_dot`), exposed as `T3Tangent.inner`, correct for both
   geometries; its ambient meaning is set by the geometry's frame choice.
3. **Non-orthonormal-frame footgun is contained.** Users reach retraction/projection *through the
   geometry*, never `T3Tangent.retract` etc. directly; `is_orthogonal`/`is_gauged` report honestly. The
   three embedding methods no longer exist on `T3Tangent`, so they can't be called wrongly on a corewise
   tangent.
4. **Base-sweep reuse is unchanged.** The model still caches the `precompute_*` sweep and injects it via
   the `*_from_sweep` primitives; the geometry is a thin wrapper, not a recompute.
5. **Geometries are stateless singletons.** The point + cached sweep are the model's state.

## 7. Build plan (slices, refactor cost accepted)

1. **Slice G1 — `Geometry` in `manifold.py` + thin `T3Tangent`.** Introduce `ManifoldGeometry` /
   `CorewiseGeometry`; move `retract`, `*_gauge_projection`, `project(dense→tangent)` off `T3Tangent`
   onto the geometries (backend algorithms unchanged); drop `apply_gauge_projection=` from the
   constructors. Update existing `T3Tangent` consumers (the Hilbert example, `manifold`/`tucker`
   tests, doctests). Self-contained `manifold.py` refactor; no `backend/` change. Verify the full suite +
   doctests.
2. **Slice G2 — generic `GaussNewtonModel`.** Collapse the six `*GaussNewtonModel` classes and the ~24
   tangent/corewise backend functions into one geometry-generic model + sampling-kind factories
   (`apply_model`/`entries_model`/`probe_model`). Corewise gradients/Hessians now return `T3Tangent` at
   `(U,G,G,G)`. Re-point the (kind-parameterized) fitting tests at the new surface; keep the exact
   dense-truth + matched-pair oracles.
3. **Slice G3 — `optimizers.py`.** One geometry-agnostic `newton_cg` (truncated/regularized for the
   singular corewise `H`), one `lbfgs`/`gradient_descent`. Consume `(geometry, model_builder, x0)`.
4. **Slice G4 — example + docs.** Run `examples/fit_hilbert_tensor_newton_cg.py` through *both* geometries
   from the same optimizer; confirm the manifold path matches today's iterates and a corewise (L-BFGS)
   path converges. Refresh `fitting_plan.md` / `entries_apply_probe.md` to the geometry framing.

## 8. Risks / open questions

- **Singular corewise `H`.** `newton_cg` must tolerate it (truncated CG / Levenberg–Marquardt damping),
  or steer corewise users to first-order. A geometry may advertise `hessian_is_degenerate` as a hint.
- **`oblique_gauge_projection`** (the ambient-preserving gauge fix) — a second manifold projection
  variant; expose as a geometry method/option, decide during G1.
- **Scope discipline.** Keep the `Geometry` interface to the three methods (`base`, `project`, `retract`,
  + manifold's `project_dense`); resist a manifold zoo.
- **Naming.** `manifold.py` now hosts a non-manifold (`CorewiseGeometry`); consider renaming to
  `geometry.py` later. `Geometry` vs `Chart` vs `Parametrization` — `Geometry` chosen (captures
  metric + retraction; Manopt-familiar).
- **Derivative variants** (`probe_derivatives`, blocked on that branch's merge) inherit the same geometry
  structure when unblocked — the jet `𝒥`/`𝒥ᵀ` compose with `geometry.project` identically.
