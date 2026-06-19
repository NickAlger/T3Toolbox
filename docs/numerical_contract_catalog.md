# S2 — Numerical-contract catalog (precondition vs caveat sweep)

*The make-or-break input to [`docs/safe_unsafe_mode_plan.md`](safe_unsafe_mode_plan.md) §5. Sweep of the
**verified** modules (`tucker_tensor_train`, `basis_variations_format`, `manifold`, `corewise`, `fitting` +
backend, `probing`) classifying each op's numerical assumptions as an enforceable **precondition** or a
non-enforced **caveat**.*

> **APPROVED by Nick, 2026-06-19**, with these decisions:
> 1. `retract` precondition = **ORTH only** (not GAUGE — retract is gauge-invariant). ✓
> 2. `MANIFOLD.inner` checks **all** of SF + ORTH + GAUGE (+ structural-minimal, below), the frame ones
>    cached on the frozen objects. ✓
> 3. **Frontend-only** enforcement for now; a backend mirror is deferred and needs careful discussion
>    (current lean: *no* backend numerical checks). ✓
> 4. No reclassifications. ✓
> 5. **Minimal ranks: do NOT retire** — keep this catalog as the answer to the long-standing "which ops
>    need minimal ranks" TBD (a reference that does not exist in the literature). In safe mode, check
>    ranks are **structurally** minimal (`has_minimal_ranks`, cheap rank arithmetic); **never** run the
>    numerical (SVD) check; **document** the requirement on each op. See the revised "Minimal ranks"
>    section. The structurally-but-not-numerically-minimal gap is an adversarial edge case (failure mode:
>    NaN / wrong result) we accept.

## The classification rule

- **Precondition** — the op is *undefined or numerically wrong* without it. **Safe mode checks it** (and
  raises); unsafe mode / under-jit skips it.
- **Caveat** — the op is *valid and correct as computed*; the property only governs *what the result
  means* (e.g. "this equals Hilbert–Schmidt"). **Never enforced** — it would reject legitimate use.

The classic trap: orthogonality/gaugedness for `inner`/`norm` is a **caveat** (they're valid on any frame;
those properties only make them *equal HS*), so it moves to the geometry's `inner`/`norm`, not the raw op.

## The numerical properties (and their checkers / cost)

| property | meaning | checker (exists) | cost | notes |
|---|---|---|---|---|
| **same-frame** | two tangents share one frame (= same tangent space) | *new* `same_frame(b1,b2)` | **O(1) common case** | `b1 is b2 or frames_equal(b1.data,b2.data, rtol)` — identity is the fast sufficient path; `frames_equal` (a value compare) is the complete fallback that also accepts the jit round-trip. |
| **orthogonal** | the frame is orthonormal (U/O/L/R conditions) | `T3Basis.is_orthogonal()` | contractions | a `@cached_property` candidate (frame is frozen → check once). |
| **gauged** | variations satisfy the gauge conditions (48)–(49) | `T3Tangent.is_gauged()` | contractions | per (basis,variations) pair; cacheable. |
| **minimal rank** | structurally-and-numerically minimal ranks | `has_minimal_ranks` (structural) | SVD (numerical) | **numerical check intentionally skipped** (§ "Minimal ranks"). |

## Master table (by surface)

`SF` = same-frame, `ORTH` = orthogonal base, `GAUGE` = variations gauged. "—" = no numerical precondition
(structural checks like shapes/ranks/`check_bv_pair` still apply, always, in both modes).

### `T3Tangent` (after the §S3 rename)
| op | precondition (checked in safe mode) | caveat (never checked) | note |
|---|---|---|---|
| `+`, `−` | **SF** | — | today an `is`-identity "structural" guard → becomes `same_frame` (the root-cause fix) |
| `* scalar`, `__neg__` | — | — | |
| `corewise_inner(other)` | **SF** | "= HS iff ORTH+GAUGE+minimal" | renamed from `inner`; the caveat moves to `MANIFOLD.inner` |
| `corewise_norm()` | — (unary) | "= HS iff ORTH+GAUGE+minimal" | renamed from `norm` |
| `normalized()`, `allclose(other)` | `allclose`: **SF** | — | `normalized` is unary |
| `to_dense`, `to_t3` | — | — | realization is gauge-invariant, valid on any frame |
| `probe`/`apply`/`entries` (+transposes), `*_derivatives` (+transposes) | — | — | **bare 𝒥/𝒥ᵀ** — gauge-invariant, any frame; the Riemannian `Π` is the geometry's, applied separately |
| `to_vector`/`from_vector`, `zeros`/`unit`/`zeros_like`, `reverse`, `sum_tangents` | — | — | structural / constructors |
| `stack_tangents` | **SF** (all leaves) | — | today identity |
| `stack_basis`, `unstack_*` | — | — | `stack_basis` deliberately stacks *different* bases |
| `is_orthogonal`/`is_gauged`/`has_minimal_ranks`, `minimal_ranks`, `tangent_space_dimension` | — | — | **checkers** — they *are* the checks; keep non-enforcing |

### Geometries (the semantic `inner`/`norm` live here — Nick's refinement)
| op | precondition | caveat | note |
|---|---|---|---|
| `MANIFOLD.inner(t1,t2)` | **SF + ORTH + GAUGE(both)** | minimal (for exact HS) | the HS inner product; *checks* ORTH+GAUGE, can't cheaply check minimal |
| `MANIFOLD.norm(t)` | **ORTH + GAUGE** | minimal | unary (no SF) |
| `COREWISE.inner(t1,t2)` | **SF** | — | Euclidean; **no ORTH/GAUGE** (the corewise frame is non-orthonormal by design) |
| `COREWISE.norm(t)` | — | — | Euclidean, unary |
| `MANIFOLD.project` (Π), `project_oblique` | **ORTH** | minimal (for oblique's HS-matching purpose) | the orthogonal-gauge projection needs an orthonormal frame to *be* the HS-orthogonal one |
| `MANIFOLD.retract(p)` | **ORTH** | minimal (rank preservation) | precondition *implied, not currently documented* — confirm |
| `MANIFOLD.project_ambient(basis,grad)` | **ORTH** | — | docstring already states "Requires orthogonal; minimal NOT required" ✓ |
| `MANIFOLD.transport(v,new_basis)` | **ORTH** (new_basis) | — | = `project_ambient` onto the new frame |
| `MANIFOLD.randn`/`random_orthogonal`/`randn_like` | **ORTH** (via `project`) | minimal (for a *true* Gaussian on `T_xM`) | docstring already careful: "for a non-orthogonal basis it is merely gauged" |
| `MANIFOLD.base`, `COREWISE.base`, `COREWISE.{randn,project,retract,randn_like}` | — | — | `base` *produces* the frame; corewise ops are gauge-free by design |

### `GaussNewtonModel` (fitting)
| op | precondition | note |
|---|---|---|
| `gradient`, `gn_hessian(p)`, `jacobian(p)`, `gn_quadratic(p)`, `evaluate(p)` | **SF** (`p` vs `model.base`) | today `_require_at_base` identity → `same_frame`. The base's ORTH (manifold) is guaranteed by `geometry.base(x)` at construction — a *construction invariant*, not a per-op check. `evaluate`/two-form use `corewise_inner` (coordinate metric), so **SF**, not HS. |

### `TuckerTensorTrain`, `corewise`, `probing` — **precondition-free**
| surface | precondition | why |
|---|---|---|
| `TuckerTensorTrain.inner`/`norm` | **—** | **exact HS for any cores** (`use_orthogonalization` is stability, not correctness) |
| `TuckerTensorTrain.apply`/`entries`/`probe` (+ ambient/corewise transposes), `t3svd`/`t3svd_dense`, `t3m`/`__mul__`/`+`/`−`, `to_dense`/`from_canonical`/constructors | **—** | exact for any cores; only structural shape/rank checks |
| `corewise.*` (`corewise_dot`/`norm`/`add`/`sub`/`scale`/`stack_*`) | **—** | the raw coordinate ("basic types") layer; structural shape-match only |
| `probing.*` (`*_from_sweep`, `precompute_*`, `*_tangent`/`*_transpose`) | **—** | bare 𝒥/𝒥ᵀ contractions; gauge/frame-agnostic |
| `basis_variations_format`: `t3_orthogonal_representations`, constructors, `check_bv_pair`, checkers | **—** (structural) | `check_bv_pair` is **structural** (shapes) → always enforced; orthogonal-rep *produces* an orthonormal (and squashed→minimal) frame |

## The blurs this fixes (current code conflations)

1. **The same-base guard is labelled "structural" but is numerical** (`fitting._require_at_base` docstring
   "Structural guard"; `T3Tangent._check_same_tangent_space` via `is`). **The root cause.** → `same_frame`
   (identity fast-path + `frames_equal`), safe-mode, eager-only. *This is what unblocks basis-as-leaf.*
2. **`T3Tangent.inner`/`norm` fold the HS caveat into the op** (`.. warning:: equals HS only when
   orthogonal and gauged`). → split: `MANIFOLD.inner`/`norm` (HS, *checks* ORTH+GAUGE+SF) vs
   `corewise_inner`/`corewise_norm` (raw, *checks* SF). The op no longer pretends to be HS.
3. **`retract` / `MANIFOLD.project` ORTH precondition is silent** (not in the docstrings). → state + check
   ORTH. (Confirm `retract` needs only ORTH, not GAUGE — gauge is a redundancy retract is invariant to.)
4. **"some tangent ops only correct when minimal ranks — which exactly is TBD"** (`manifold.py`/`bvf`).
   → resolved as a **caveat everywhere** (see below); the TBD note can be replaced with the table above.

## Minimal ranks — the resolution of the long-standing TBD (keep this; it's the reference)

Minimal rank splits into two tests: **structural** (`has_minimal_ranks` — the ranks equal the structural
minimum for the shape; cheap integer arithmetic) and **numerical** (would require an SVD — are the cores
actually full-rank). The decision:

- **Safe mode checks the *structural* test** (`has_minimal_ranks`) on the ops that require minimal ranks,
  and raises if it fails. It is a precondition, skipped in unsafe / under jit like the others.
- **The numerical test is *never* run** (no SVD). The structurally-but-not-numerically-minimal gap is an
  adversarial edge case (failure mode: NaN / wrong result) we accept.
- **The requirement is documented on each op below** (Nick (b)), and this table *is* the answer to the
  "which ops need minimal ranks — TBD" note in `manifold.py` / `basis_variations_format.py` (replace that
  note with a pointer here, do not delete the knowledge).

**The complete list of minimal-rank-requiring ops** (each: precondition checked *structurally* in safe
mode; numerical caveat documented):

| op | what minimal rank buys | without it (the caveat) |
|---|---|---|
| `MANIFOLD.inner` / `norm` | exact Hilbert–Schmidt | the coordinate dot ≠ HS even when orthogonal+gauged |
| `MANIFOLD.retract` | **rank preservation** (stay on the same fixed-rank `M`) | lands on a different-rank manifold |
| `MANIFOLD.randn` / `random_orthogonal` / `randn_like` | a *true* standard Gaussian on `T_xM` | merely a gauged direction |
| `MANIFOLD.project_oblique` | corewise LA on the gauged result matches HS | the gauge-fix preserves the vector but the LA isn't HS-faithful |
| `manifold_dim` / `tangent_space_dimension` | the *true* tangent-space dimension | over-counts (uses the structural-minimal ranks) |
| **not** `MANIFOLD.project`/`project_ambient`/`transport`, gauge `Π` | — | confirmed: **orthogonal suffices, minimal NOT required** |

In practice `MANIFOLD.base` returns a squashed→minimal orthonormal frame, so manifold objects are
structurally minimal by construction and the check rarely fires (matching Nick's "adversarial only" read).

## Implementation notes (for S3–S5, not decisions here)

- **`same_frame(b1, b2)` = `b1 is b2 or frames_equal(b1.data, b2.data, rtol=safety_rtol)`** — the `is`
  fast-path keeps the common eager case O(1); `frames_equal` only runs (and only matters) when objects
  differ but values match (jit round-trip) or genuinely differ.
- **Where checks live (razor):** to serve raw-`.data` users, the ORTH/GAUGE checks belong at the **backend**
  functions that consume raw data (`tangent_operations.*`, which already have `is_orthogonal`-style
  residual helpers), with the frontend methods inheriting them. `same_frame` is a frontend concept (it
  needs the two `T3Basis` objects); the backend's analogue is "the caller passed one shared base", so the
  same-frame check is naturally frontend-only.
- **Cost mitigation:** make `is_orthogonal` (frame-only) and `is_gauged` (pair) `@cached_property`s on the
  frozen objects so a fixed base/tangent in an inner loop is checked once, not per matvec.

## Sign-off questions for Nick

1. **`retract` precondition = ORTH only** (not GAUGE)? My read: yes — retract is gauge-invariant; gauge
   only relabels the variations, not the represented step. Confirm.
2. **`MANIFOLD.inner` should check both operands GAUGE + the frame ORTH** (3 checks) — accept the cost in
   safe mode (mitigated by caching), or check only SF + GAUGE and treat ORTH as a base-construction
   invariant (since `MANIFOLD.base` guarantees it)? Leaning: check all three for honesty, cache the frame
   ones.
3. **Backend-level enforcement** for raw-`.data` users (ORTH/GAUGE checks in `tangent_operations`), or
   frontend-only for now? Leaning: frontend-first (S3–S5), backend mirror later.
4. Any op above you'd reclassify (precondition ⇄ caveat)?
