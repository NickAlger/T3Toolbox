# Numerical contracts: what safe mode checks

The library's ambient **safety mode** (see the user guide's "Safe and unsafe mode") checks genuine
numerical **preconditions** and skips them under `safety.unsafe()` or a jax trace. This page is the
op-by-op reference: which operations check what, which properties are only *caveats*, and the
settled answer to "which operations need minimal ranks" (none — a reference result that does not
exist in the literature).

## The classification rule

- **Precondition** — the operation is *undefined or numerically wrong* without the property.
  **Safe mode checks it** and raises; unsafe mode / under-jit skips the check.
- **Caveat** — the operation is *valid and correct as computed*; the property only governs *what
  the result means* (e.g. "this coordinate inner product equals Hilbert–Schmidt"). **Never
  enforced** — enforcing it would reject legitimate use.

The classic example: orthogonality/gaugedness is a **caveat** for a raw coordinate inner product
(it is valid on any frame; the properties only make it *equal HS*) — which is why the semantic
`inner`/`norm` live on the `MANIFOLD` geometry (where the properties are preconditions) while
`T3Tangent.corewise_inner`/`corewise_norm` are the raw coordinate ops (no HS claim, no check).

## The numerical properties and their checkers

| property | meaning | checker |
|---|---|---|
| **same-frame** (`SF`) | two tangents share one tangent space — their frames are *numerically equal* (`safety.frames_equal`, with an object-identity fast path; a jit round-trip of the same frame passes) | built into the checked ops |
| **orthogonal** (`ORTH`) | the frame is orthonormal (U/O/L/R conditions) | `T3Frame.is_orthogonal(atol)` |
| **gauged** (`GAUGE`) | the variations satisfy the gauge conditions (48)–(49) | `T3Tangent.is_gauged(atol)` |
| **minimal rank** | no stored rank is redundant | `has_minimal_ranks` (structural) / `has_numerically_minimal_ranks` (numerical) — **diagnostic only, never enforced** (see the verdict below) |
| **tied factors** (`TIED`) | the Tucker factors are numerically equal within each sharing group ([`sharing.md`](sharing.md)) | `has_shared_tucker_factors(sharing, rtol)` (ragged and uniform; the residual is `backend.sharing.t3_sharing_residual` / `ut3_sharing_residual`) |

Checkers are **per stack element** (they return a verdict of shape `stack_shape`); a safe-mode
precondition requires **all** elements to pass. The checks are cheap in loops: the underlying
residuals (`T3Frame.orthogonality_residual`, `T3Tangent.gauge_residual`) are cached on the frozen
objects, so a fixed frame or tangent reused across an inner loop is contracted once. The tolerance
is jax-aware (`rtol_jax`, looser, when any operand is a jax array — jax defaults to float32 — else
`rtol_numpy`).

## What each surface checks in safe mode

`—` = no numerical precondition (structural checks — shapes, ranks, pair consistency — still apply
always, in both modes).

### `T3Tangent`

| op | precondition | caveat (never checked) |
|---|---|---|
| `+`, `−` | **SF** | |
| `* scalar`, `__neg__`, `normalized()` | — | |
| `allclose(other)` | **SF** | |
| `corewise_inner(other)` | **SF** | "= HS iff orthogonal + gauged" (that claim belongs to `MANIFOLD.inner`) |
| `corewise_norm()` | — (unary) | "= HS iff orthogonal + gauged" |
| `to_dense`, `to_t3` | — | realization is gauge-invariant, valid on any frame |
| `probe`/`apply`/`entries` (+ transposes), `*_derivatives` (+ transposes) | — | bare `𝒥`/`𝒥ᵀ` — gauge-invariant on any frame; the Riemannian projector `Π` is the geometry's, applied separately |
| `to_vector`/`from_vector`, `zeros`/`unit`/`zeros_like`, `reverse`, `sum_tangents` | — | |
| `stack_tangents` | **SF** (all leaves) | |
| `stack_frame`, `unstack_*` | — | `stack_frame` deliberately stacks *different* bases |
| the checkers (`is_orthogonal`, `is_gauged`, `has_minimal_ranks`, …) | — | they *are* the checks; non-enforcing by design |

### The geometries (the semantic `inner`/`norm` live here)

| op | precondition | caveat |
|---|---|---|
| `MANIFOLD.inner(t1, t2)` | **SF + ORTH + GAUGE** (both operands) | minimal (for *strict* HS exactness — see the verdict: not needed) |
| `MANIFOLD.norm(t)` | **ORTH + GAUGE** (unary — no SF) | minimal |
| `COREWISE.inner(t1, t2)` | **SF** | — (Euclidean by design; no ORTH/GAUGE) |
| `COREWISE.norm(t)` | — | |
| `MANIFOLD.project` (`Π`), `project_oblique` | **ORTH** | minimal (for oblique's HS-matching purpose) |
| `MANIFOLD.retract(p)` | **ORTH** (gauge-invariant, so ORTH only) | minimal (strict rank preservation — without it, retract drops numerically-redundant rank) |
| `MANIFOLD.project_ambient(frame, grad)` | **ORTH** | — |
| `MANIFOLD.transport(v, new_frame)` | **ORTH** (the new frame) | — |
| `MANIFOLD.randn`/`random_orthogonal`/`randn_like` | **ORTH** (via the projection) | minimal (for a *true* Gaussian on `T_xM`) |
| `MANIFOLD.frame`, `COREWISE.frame`, `COREWISE.{randn, project, retract, randn_like}` | — | `frame` *produces* the orthonormal frame; corewise ops are gauge-free by design |

### The shared (SF-T3) surface

The sharing partition itself is structural (validated always: lengths, hashable labels, equal mode
sizes and Tucker ranks within groups). The numerical precondition is **TIED** — grouped operations
assume the factors are already tied (they pick ONE basis per group; running them on untied input
would silently discard the differences), so safe mode checks it at the entry points:

| op | precondition | caveat (never checked) |
|---|---|---|
| `t3svd(sharing=…)`, `rank_adjustment_sweep(…, sharing=…)`, `resize(…, sharing=…)` (ragged and uniform) | **TIED** | |
| `continuation_ranks(sharing=…)` | **TIED** (via its internal grouped `t3svd`) | |
| `shared(…)` wrapper: `frame(x)`, `transport(v, new_frame)` | **TIED** (the point / the new frame) + the base geometry's preconditions | |
| `shared(MANIFOLD/UNIFORM_MANIFOLD).retract(p)` | **ORTH + TIED frame factors + TIED tangent coordinates** (the embedding's solve would silently tied-project an untied tangent) | minimal (the base retract's rank-preservation caveat) |
| `shared(…).project`, `project_ambient`, `randn` | the base geometry's preconditions only (they *produce* tied output) | |
| `shared(…).inner` / `norm` | the base geometry's (sharing never reweights the metric) | |
| `x.share(sharing, …)` | — (it is how you *enter* the format; untied input is the point) | |
| **full shared rank** (`s_{g,min} > 0`) | **never enforced** — a zero-padded continuation restart sits *on* the lower-rank stratum by construction ([`sharing.md`](sharing.md)); the projection solve is a clipped pseudoinverse, well-defined there | diagnostic via the group spectrum |

### `GaussNewtonModel`

| op | precondition |
|---|---|
| `gradient`, `gn_hessian(p)`, `jacobian(p)`, `gn_quadratic(p)`, `evaluate(p)` | **SF** — `p` must live at the model's frame (checked numerically, identity fast path). The frame's orthogonality is a construction invariant (`geometry.frame(x)` produces it), not a per-op check. |

### Precondition-free surfaces

| surface | why |
|---|---|
| `TuckerTensorTrain.inner`/`norm` | **exact HS for any cores** (`use_orthogonalization` is a stability option, not a correctness requirement) |
| `TuckerTensorTrain` arithmetic, sampling, `t3svd`, `t3m`, conversions, constructors | exact for any cores; only structural checks |
| `corewise.*` | the raw coordinate layer; structural shape-match only |
| the backend sampling machinery (`probing`/`apply`/`entries` sweeps and transposes) | bare `𝒥`/`𝒥ᵀ` contractions, frame/gauge-agnostic |
| `t3_orthogonal_representations`, the `T3Frame`/`T3Variations` constructors | structural validation always (both modes); the orthogonal representation *produces* an orthonormal frame |

## Minimal ranks: the settled verdict

Minimal rank splits into **structural** (`has_minimal_ranks` — ranks equal the structural minimum;
cheap integer arithmetic) and **numerical** (`has_numerically_minimal_ranks` — no stored rank
numerically redundant). Which operations actually *need* minimal ranks was settled empirically
(each op compared on a minimal vs a non-minimal *orthogonal* frame against the dense oracle):

| op | needs minimal? | evidence |
|---|---|---|
| `MANIFOLD.inner`/`norm` | **NO** | the gauged coordinate inner product equals the dense HS value *exactly* on a non-minimal orthonormal frame — **orthogonal + gauged suffices** |
| `manifold_dim` / `tangent_space_dimension` | **NO** | the formula matches the true tangent-space dimension for any ranks |
| `MANIFOLD.retract` | **caveat only** | still a valid first-order retraction on a non-minimal frame; it just *drops* the numerically-redundant rank (desirable cleanup) — minimal buys only strict rank preservation |
| `MANIFOLD.randn` / `project_oblique` | **NO** | correct on any orthonormal frame |
| `MANIFOLD.project` / `project_ambient` / `transport`, the gauge projections | **NO** | orthogonality suffices |

**Verdict: minimal rank is not a correctness precondition for any verified operation**, so it is
never checked — a minimal-rank gate would reject legitimate rank-redundant frames for no
correctness reason. It survives as the `retract` rank-preservation caveat and as the diagnostic
checkers. Naming rule: bare `minimal_ranks` means the structural notion; `numerically_minimal` means
the SVD-grade notion.

**Why excess rank is inert** (the mechanism, worth knowing before you reach for a rank check): a frame
carries *four* rank stores — `up`/`down` and `left`/`right` — and an over-ranked request is absorbed as
slack between them rather than breaking any single core's orthonormality. So an orthonormal frame need
not be minimal: `t3_orthogonal_representations` does not guarantee it, and neither does
`T3Frame.random_orthogonal`, which happily returns `up_ranks=(4,4,4)` with `down_ranks=(2,4,2)` if you
ask for a Tucker rank the TT bonds cannot support. Such a frame is exactly orthogonal and every tangent
operation on it is exact — the tangent space, its dimension, and the HS identities are unchanged. Full
derivation and the measured numbers: [`frame_variations.md`](frame_variations.md). Build from a
minimal-rank point only if you need retraction to *preserve* the rank tuple.

---

*The decision record behind this page — the op-by-op sweep, the experiments, and the sign-off —
is preserved as [`contributor/numerical_contract_catalog.md`](contributor/numerical_contract_catalog.md).*
