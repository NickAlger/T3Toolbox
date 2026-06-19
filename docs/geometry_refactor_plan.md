# Geometry-based fitting — restructure plan

*Design locked with Nick (2026-06-19), branch `fitting`. This supersedes the corewise structure in
[`docs/fitting_plan.md`](fitting_plan.md) (the six parallel `*GaussNewtonModel` classes + ~38 backend
functions): tangent and corewise fitting are unified under an explicit **`Geometry`** abstraction, with
`T3Tangent` as the universal tangent type and **geometry-agnostic optimizers**. The cost of the refactor
is accepted; this is the target structure, chosen for mathematical coherence, code simplicity, and
user utility — not to preserve the existing code.*

## 0. Resuming from a fresh context (read this first)

*If you are a future Claude with little/no memory of the conversation that produced this: this document
is self-contained. Read §1–§5 for the reasoning, §7 for the build order, then execute. Do not re-derive
the design or re-litigate the structure — it is decided. Confirm small choices with Nick, but build.*

**Branch & git state (as of 2026-06-19).** The plan was written on branch `fitting`. The intended flow:
`probe-derivatives` and `fitting` are both merged to `main` (they are tested and work); **this refactor
runs on a fresh branch off `main`** (e.g. `geometry-refactor`). Check `git branch`/`git log --oneline -15`
to see where you actually are. The fitting layer described below is the *starting point* you are
refactoring; everything is committed.

**Current state — what exists to refactor (the starting point):**
- `t3toolbox/fitting.py` — **six** frozen-dataclass models: `ApplyGaussNewtonModel`,
  `EntriesGaussNewtonModel`, `ProbeGaussNewtonModel`, and `Corewise{Apply,Entries,Probe}GaussNewtonModel`.
  Each holds a base (`T3Basis` for tangent, `TuckerTensorTrain` for corewise) + sample data + residual;
  exposes `objective_value`, `gradient`, `gn_hessian(p)`, `evaluate(p)`; caches `_base_sweep`. Tangent
  models return `T3Tangent`; corewise models return **raw `(tucker, tt)` tuples** (this is what the
  refactor changes — corewise should return `T3Tangent` at `(U,G,G,G)`).
- `t3toolbox/backend/fitting.py` — ~38 functions: `<kind>_{jacobian,gradient,gn_hessian,model_value}` for
  tangent (compose bare `𝒥`/`𝒥ᵀ` + `Π`); `<kind>_corewise_*` (same, no `Π`, substituted base via
  `_corewise_base`); `precompute_base_sweep` / `precompute_entries_base_sweep` /
  `precompute_corewise_base_sweep` / `precompute_entries_corewise_base_sweep`; `_sumsq_over_samples` /
  `_sumsq_over_probes`.
- `t3toolbox/backend/probing.py` — the **bare** `𝒥`/`𝒥ᵀ` reuse primitives (KEEP, share across geometries):
  `precompute_base_sweep`, `precompute_entries_base_sweep`, and
  `{apply,entries,probe}_{jacobian,transpose}_from_sweep`. These compose with `Π`-or-not in the fitting
  layer; they are correct and do not change.
- `t3toolbox/manifold.py` — `T3Tangent` (the class to thin per §5) and `T3Basis`. The backend algorithms
  for the three methods that move (`tangent_operations.orthogonal_gauge_projection`, the retraction,
  `project_*_onto_tangent`) live in `t3toolbox/backend/tangent_operations.py` and **do not change** — the
  geometry classes select them.
- `examples/fit_hilbert_tensor_newton_cg.py` — uses `fitting.ApplyGaussNewtonModel` via a `model_builder`;
  the inner CG is the reference. (G4 re-runs it through both geometries.)
- Tests: `tests/test_fitting.py` (kind-parameterized: backend dense-truth/two-form/gauge/razor/adjoint +
  frontend), `tests/test_dispatch.py` (`test_jit_fitting*`).

**Verification commands** (run from repo root; conda env `tttt`):
```
python -m pytest tests/test_fitting.py tests/test_manifold.py tests/test_tucker_tensor_train.py \
                 tests/test_basis_variations_format.py tests/backend/test_contractions.py -q
python -m pytest tests/test_dispatch.py -q              # jax dispatch (jit)
python -m doctest t3toolbox/manifold.py t3toolbox/fitting.py
```
The exact dense-truth oracles in `test_fitting.py` (manifold: `½‖r + 𝒥Πp‖²`; corewise: `½‖r +
apply_dense(Σ_core dense(core→δcore))‖²`) are the correctness gold standard — preserve them through the
refactor (they should still pass against the geometry-generic model). Corewise oracles use a *relative*
tolerance (large raw-core magnitudes).

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
- **Decisive point (corrected per Nick's benchmarks).** Autodiff makes the corewise gradient *convenient*
  (free, no hand-coding) but **not** necessarily fast: the hand-rolled sweeping/probing contractions,
  jit-compiled, **outperform `jax.grad` for the corewise gradient, sometimes substantially** (the
  base-sweep reuse + no autodiff-graph overhead — empirically observed). So corewise is *not* a redundant
  autodiff stand-in. Our corewise earns its place as **(a)** the Euclidean geometry inside a unified
  framework, **(b)** a matrix-free Gauss–Newton operator (`JᵀJ`-vector products, awkward for autodiff via
  double-backprop), **(c)** autodiff-free numpy, and **(d)** competitive-or-faster performance even where
  autodiff exists. All four reinforce the geometry-abstraction framing: corewise belongs as a first-class,
  *performant* Euclidean geometry — not a "just use autodiff" afterthought.
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

**Decisions locked in G1 (refining the sketch above — this is the as-built API).** (1) `randn` /
`random_orthogonal` / `randn_like` **moved onto the geometries** (the old `apply_gauge_projection=` flag
*is* the geometry choice: `MANIFOLD.randn` gauges → i.i.d. Gaussian on `T_xM`; `COREWISE.randn` is raw
cores). This supersedes "keep randn, drop the flag" in §5. `zeros` / `unit` stay on `T3Tangent`
(geometry-independent). (2) The ambient projection is **unified** as
`ManifoldGeometry.project_ambient(basis, grad, method='contraction')` accepting a `TuckerTensorTrain`
*or* dense `grad` — it absorbs and **retires** the old `T3Tangent.project`, `project_dense_onto_tangent`,
and `riemannian_gradient`. (3) `transport` and the oblique gauge fix **moved to `ManifoldGeometry`**
(`transport(v, new_basis)` / `project_oblique(v)`). (4) **`retract(p)` takes the tangent only** — the
point is carried by `p.basis`; corewise recovers `(U,G)` from its `(U,G,G,G)` frame. Singletons:
`MANIFOLD`, `COREWISE`. Methods take the frame (basis) or the tangent (which carries its frame), never a
bare point — except `base(x)`, which builds the frame from a point.

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
| `zeros`, `unit` | **keep** | geometry-independent (zero / canonical units have no gauge ambiguity) |
| `randn`, `random_orthogonal`, `randn_like` | **move → `Geometry.randn(...)`** (G1 refinement) | the `apply_gauge_projection=` flag *is* the geometry: `MANIFOLD.randn` gauges, `COREWISE.randn` is raw |
| `retract` | **move → `Geometry.retract(p)`** (tangent only) | the chart's "how to move" — the defining difference; point carried by `p.basis` |
| `orthogonal_gauge_projection` / `oblique_gauge_projection` | **move → `Geometry.project` / `ManifoldGeometry.project_oblique`** | `Π` is the manifold's gradient map; identity for corewise; invalid on a non-orthonormal frame |
| `project` (static, T3 → tangent), `project_dense_onto_tangent`, `riemannian_gradient` | **move → `ManifoldGeometry.project_ambient(basis, grad, method=)`** | unified ambient→`T_xM` projection (T3 or dense); manifold-only |
| `transport` | **move → `ManifoldGeometry.transport(v, new_basis)`** | projective transport; orthonormality-requiring, manifold-only |
| `is_gauged`, `T3Basis.is_orthogonal` | **keep as checkers** | numerical property checks (house philosophy); the geometry *uses* them |

The embedding-dependent operations leave `T3Tangent` — **retract, gauge-projection, project-from-ambient,
transport** — plus (G1 refinement) the **gauged random constructors** (`randn` family), since the
gauge-vs-raw choice is itself the geometry. The backend algorithms for the embedding ops already live in
`tangent_operations.py`; the geometry classes are thin bundlers that select them.

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

1. **Slice G1 — `Geometry` in `manifold.py` + thin `T3Tangent`. ✅ DONE.** Introduced `ManifoldGeometry` /
   `CorewiseGeometry` + singletons `MANIFOLD` / `COREWISE`; moved `retract`, `*_gauge_projection`,
   the ambient projection (unified as `project_ambient`), `transport`, and (refinement) the `randn`
   family off `T3Tangent` onto the geometries (backend algorithms unchanged). `zeros` / `unit` stay.
   Updated all consumers (both Hilbert examples, `manifold` / `fitting` / `dispatch` / `probe_derivatives`
   tests, doctests across `manifold` / `fitting` / `backend.probing`). Self-contained `manifold.py`
   refactor; no `backend/` change. Verified: 210 core tests + 10 dispatch + all doctests pass; both
   examples run end-to-end. See the "Decisions locked in G1" note under §4 for the as-built API.
2. **Slice G2 — generic `GaussNewtonModel`. ✅ DONE.** Collapsed the six `*GaussNewtonModel` classes and
   the ~38 tangent/corewise backend functions into **one** geometry-generic `GaussNewtonModel(geometry,
   base, kind, sample, residual)` + factories `apply_model`/`entries_model`/`probe_model(geometry, x,
   sample, residual)`. The backend is now a `SamplingKind` bundle (bare `𝒥`/`𝒥ᵀ` from `probing` +
   `sumsq` reducer) per kind (`APPLY`/`ENTRIES`/`PROBE`); the geometry supplies `Π` (`geometry.project`)
   and the frame (`geometry.base`). Two unifications fell out: `geometry.base(x)` **subsumes** the old
   `_corewise_base` substitution (corewise base sweep = `precompute_base_sweep((U,G,G,G), ·)`), and the
   dense oracle is `½‖r + forward(geometry.project(p).to_dense())‖²` for **both** geometries (corewise
   `project(p).to_dense()` = the sum-of-core-swaps). Corewise gradients/Hessians now return `T3Tangent`
   at `(U,G,G,G)`; the same-base guard applies to both geometries. `c = ½‖r‖²` folds into `½·sumsq(r)`.
   Re-pointed `test_fitting.py` to one class parameterized over (kind × geometry × C) — dense-truth,
   two-form, razor, matched-pair (manifold gauges / corewise no-Π via bare-transpose compare), GN
   symmetry, adjoint, same-base guard, caching, + cross-checks vs the established `T3Tangent` /
   `TuckerTensorTrain` transposes. Merged the two `test_dispatch` fitting jit tests into one (model ×
   kind × geometry). Verified: 206 core + 9 dispatch + doctests pass; the Newton-CG example reproduces
   **bit-for-bit** the pre-refactor iterates. *(`docs/fitting_plan.md` prose is now historical — defer
   to G4.)*
3. **Slice G3 — `optimizers.py`.** One geometry-agnostic `newton_cg` (truncated/regularized for the
   singular corewise `H`), one `lbfgs`/`gradient_descent`. Consume `(geometry, model_builder, x0)`.
4. **Slice G4 — example + docs.** Run `examples/fit_hilbert_tensor_newton_cg.py` through *both* geometries
   from the same optimizer; confirm the manifold path matches today's iterates and a corewise (L-BFGS)
   path converges. Refresh `fitting_plan.md` / `entries_apply_probe.md` to the geometry framing.

## 8. Risks / open questions

- **RESOLVED — the jit/recompile/OO question (the safe-mode arc, S1–S6, done 2026-06-19).** The whole
  predicament traced to one root cause: the same-frame guard was a *numerical* property faked as
  *structural* via object identity (`self.basis is other.basis`). Identity forced `T3Tangent`'s basis to
  be jax **aux_data** (→ recompile every base change) and false-failed on a jit round-trip. The fix:
  numericalize the guard (`safety.frames_equal`, an `is`-fast-path then value compare), which lets the
  basis become a pytree **leaf** and the whole `GaussNewtonModel` a registered pytree (base/sweep/sample/
  residual leaves; geometry/kind aux). Now **`jit(lambda model, p: model.gn_hessian(p))` compiles once
  across all bases** (`traces=1`) — you jit the frontend matvec directly; no per-function toggle, no
  recompile. Numerical guards are eager-only (skip under unsafe/jit). The OO layer stays. The geometry
  singletons remain zero-leaf pytrees. `jacobian(p)` + `gn_quadratic(p)` (= `pᵀHp = ‖Jp‖²`, one forward)
  added for cheap Cauchy / line-search step lengths. Full design + build:
  [`docs/safe_unsafe_mode_plan.md`](safe_unsafe_mode_plan.md),
  [`docs/numerical_contract_catalog.md`](numerical_contract_catalog.md). **G3 is unblocked.**
- **MC-SGD (Manifold Cauchy SGD) — prototyped, promising, stopping/batch heuristics still finicky at
  small scale (2026-06-19).** Validated inline in `examples/fit_hilbert_from_apply_derivatives.py`: the
  tuning-free Cauchy step (`alpha = ‖g‖²/‖Jg‖²`, exactly `model.gradient` + `model.gn_quadratic`) fits
  apply-derivatives to the noise floor, ~8× faster than full-batch Newton-CG. The **core optimizer is
  robust**; the finickiness is in the *auxiliary heuristics*: (1) batch size — the paper's ~10%-of-samples
  rule degenerates to a single, too-noisy base point at `N_X=10` (floored to 2 in the example); (2) the
  **stopping window is epoch-based** (`lag = C_t · n_s/|B|`), so a larger batch shrinks the epoch and the
  stop fires too early (batch=3 failed after ~12 iters — a real early non-monotonicity in the *deterministic
  full-batch* loss, caught by a ~12-iteration window; batch=3 converged fine when not stopped). Likely a
  small-scale artifact (at scale a minibatch is small-*fraction* yet large-*absolute* → clean gradient, and
  epochs are large in absolute iterations → robust window; the paper's MC-SGD is robust at scale on *probe*
  fitting). **Open for G3:** when lifting MC-SGD into `optimizers.py`, make the stopping window
  **absolute-iteration-based** (or add a min-iterations guard) so it decouples from batch size; and decide
  whether a derivative `GaussNewtonModel` (an `apply_derivatives_model` in `fitting.py`) replaces the inline
  closures so apply/entries/probe get MC-SGD for free. Robustness of apply-derivative MC-SGD *at scale* is
  unproven (future research). **Full write-up: [`mcsgd_apply_derivatives.md`](mcsgd_apply_derivatives.md).**
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
