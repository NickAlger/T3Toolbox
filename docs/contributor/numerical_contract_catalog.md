# S2 — Numerical-contract catalog (precondition vs caveat sweep)

_The decision record that designed safe mode, preserved as approved (2026-06-19). The
**current-state statement** of these contracts — present tense, migration arrows resolved — is the
user page [`../numerical_contracts.md`](../numerical_contracts.md)._

*The make-or-break input to [`dev/archive/safe_unsafe_mode_plan.md`](https://github.com/NickAlger/T3Toolbox/blob/main/dev/archive/safe_unsafe_mode_plan.md) §5. Sweep of the
**verified** modules (`tucker_tensor_train`, `frame_variations_format`, `manifold`, `corewise`, `fitting` +
backend, `probing`) classifying each op's numerical assumptions as an enforceable **precondition** or a
non-enforced **caveat**.*

> **APPROVED by Nick, 2026-06-19**, with these decisions:
> 1. `retract` precondition = **ORTH only** (not GAUGE — retract is gauge-invariant). ✓
> 2. `MANIFOLD.inner` checks **all** of SF + ORTH + GAUGE (+ structural-minimal, below), the frame ones
>    cached on the frozen objects. ✓ *— the minimal-rank parenthetical was superseded the same day; see
>    the note under this banner.*
> 3. **Frontend-only** enforcement for now; a backend mirror is deferred and needs careful discussion
>    (current lean: *no* backend numerical checks). ✓
> 4. No reclassifications. ✓
> 5. **Minimal ranks: do NOT retire** — keep this catalog as the answer to the long-standing "which ops
>    need minimal ranks" TBD (a reference that does not exist in the literature). In safe mode, check
>    ranks are **structurally** minimal (`has_minimal_ranks`, cheap rank arithmetic); **never** run the
>    numerical (SVD) check; **document** the requirement on each op. See the revised "Minimal ranks"
>    section. The structurally-but-not-numerically-minimal gap is an adversarial edge case (failure mode:
>    NaN / wrong result) we accept. *— superseded the same day; see the note below.*

**Superseded (same session, 2026-06-19).** Decisions 2 and 5 approved a safe-mode **structural
minimal-rank check**. The op-by-op experiment run about an hour later — the "Minimal ranks" section
below — disproved its premise: minimal rank is a correctness precondition for *nothing*. The check was
never wired, and the banner above is kept as written only because it records what was approved at the
time. What stands is the verdict in that section and the user page
[`../numerical_contracts.md`](../numerical_contracts.md).

The mechanism behind the result was worked out later and is written up in
[`../frame_variations.md`](../frame_variations.md): a frame carries **four** rank stores
(`up`/`down`, `left`/`right`) and absorbs an over-ranked request as slack between them, so an
orthonormal frame need not be minimal — `T3Frame.random_orthogonal` returns exactly such a frame for any
over-ranked request, and every tangent operation on it is exact. Orthonormality plus the gauge is the
whole precondition; a minimal-rank gate would reject legitimate frames.

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
| **orthogonal** | the frame is orthonormal (U/O/L/R conditions) | `T3Frame.is_orthogonal()` | contractions | a `@cached_property` candidate (frame is frozen → check once). |
| **gauged** | variations satisfy the gauge conditions (48)–(49) | `T3Tangent.is_gauged()` | contractions | per (frame,variations) pair; cacheable. |
| **minimal rank** | structurally-and-numerically minimal ranks | `has_minimal_ranks` (structural) | SVD (numerical) | **numerical check intentionally skipped** (§ "Minimal ranks"). |

> **Checkers are per-stack-element.** Every checker/residual above returns one verdict per stack element
> (shape `stack_shape`, scalar when unstacked) — see `docs/batching_and_stacking.md` §9. So a safe-mode
> precondition reduces with **`.all()`** at the call site (`safety.require(frame.is_orthogonal(atol).all(),
> …)`): it requires **all** stack elements to pass; `safety.require` itself stays a scalar gate. (The
> *structural* `has_minimal_ranks` is the one scalar-in-ragged exception — see that §9 note.)

## Master table (by surface)

`SF` = same-frame, `ORTH` = orthogonal frame, `GAUGE` = variations gauged. "—" = no numerical precondition
(structural checks like shapes/ranks/`check_fv_pair` still apply, always, in both modes).

> **Read the `minimal` entries in the caveat columns below as pre-experiment.** The "Minimal ranks"
> section further down measured every one of them and found none to hold: `inner`/`norm` are exactly HS
> without minimality, and only `retract`'s *strict rank preservation* survives as a real caveat. The
> rows are left as written because this is the record of the sweep as approved.

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
| `stack_frame`, `unstack_*` | — | — | `stack_frame` deliberately stacks *different* bases |
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
| `MANIFOLD.project_ambient(frame,grad)` | **ORTH** | — | docstring already states "Requires orthogonal; minimal NOT required" ✓ |
| `MANIFOLD.transport(v,new_frame)` | **ORTH** (new_frame) | — | = `project_ambient` onto the new frame |
| `MANIFOLD.randn`/`random_orthogonal`/`randn_like` | **ORTH** (via `project`) | minimal (for a *true* Gaussian on `T_xM`) | docstring already careful: "for a non-orthogonal frame it is merely gauged" |
| `MANIFOLD.frame`, `COREWISE.frame`, `COREWISE.{randn,project,retract,randn_like}` | — | — | `frame` *produces* the frame; corewise ops are gauge-free by design |

### `GaussNewtonModel` (fitting)
| op | precondition | note |
|---|---|---|
| `gradient`, `gn_hessian(p)`, `jacobian(p)`, `gn_quadratic(p)`, `evaluate(p)` | **SF** (`p` vs `model.frame`) | today `_require_at_frame` identity → `same_frame`. The frame's ORTH (manifold) is guaranteed by `geometry.frame(x)` at construction — a *construction invariant*, not a per-op check. `evaluate`/two-form use `corewise_inner` (coordinate metric), so **SF**, not HS. |

### `TuckerTensorTrain`, `corewise`, `probing` — **precondition-free**
| surface | precondition | why |
|---|---|---|
| `TuckerTensorTrain.inner`/`norm` | **—** | **exact HS for any cores** (`use_orthogonalization` is stability, not correctness) |
| `TuckerTensorTrain.apply`/`entries`/`probe` (+ ambient/corewise transposes), `t3svd`/`t3svd_dense`, `t3m`/`__mul__`/`+`/`−`, `to_dense`/`from_canonical`/constructors | **—** | exact for any cores; only structural shape/rank checks |
| `corewise.*` (`corewise_dot`/`norm`/`add`/`sub`/`scale`/`stack_*`) | **—** | the raw coordinate ("basic types") layer; structural shape-match only |
| `probing.*` (`*_from_sweep`, `precompute_*`, `*_tangent`/`*_transpose`) | **—** | bare 𝒥/𝒥ᵀ contractions; gauge/frame-agnostic |
| `frame_variations_format`: `t3_orthogonal_representations`, constructors, `check_fv_pair`, checkers | **—** (structural) | `check_fv_pair` is **structural** (shapes) → always enforced; orthogonal-rep *produces* an orthonormal (and squashed→minimal) frame |

## The blurs this fixes (current code conflations)

1. **The same-frame guard is labelled "structural" but is numerical** (`fitting._require_at_frame` docstring
   "Structural guard"; `T3Tangent._check_same_tangent_space` via `is`). **The root cause.** → `same_frame`
   (identity fast-path + `frames_equal`), safe-mode, eager-only. *This is what unblocks frame-as-leaf.*
2. **`T3Tangent.inner`/`norm` fold the HS caveat into the op** (`.. warning:: equals HS only when
   orthogonal and gauged`). → split: `MANIFOLD.inner`/`norm` (HS, *checks* ORTH+GAUGE+SF) vs
   `corewise_inner`/`corewise_norm` (raw, *checks* SF). The op no longer pretends to be HS.
3. **`retract` / `MANIFOLD.project` ORTH precondition is silent** (not in the docstrings). → state + check
   ORTH. (Confirm `retract` needs only ORTH, not GAUGE — gauge is a redundancy retract is invariant to.)
4. **"some tangent ops only correct when minimal ranks — which exactly is TBD"** (`manifold.py`/`bvf`).
   → resolved as a **caveat everywhere** (see below); the TBD note can be replaced with the table above.

## Minimal ranks — the resolution of the long-standing TBD (EMPIRICALLY verified — this is the reference)

Minimal rank splits into **structural** (`has_minimal_ranks` — ranks equal the structural minimum; cheap
integer arithmetic) and **numerical** (`has_numerically_minimal_ranks` — no stored rank numerically
redundant; needs an SVD for a tensor, *free* for an orthonormal frame). **An experiment settled which ops
actually need it** (`tests/`-style script run 2026-06-19; each op compared on a minimal vs a non-minimal
*orthogonal* frame against the dense oracle):

| op | needs minimal? | evidence |
|---|---|---|
| `MANIFOLD.inner` / `norm` (= `corewise_inner` when gauged) | **NO** | `corewise_inner(gauged) == dense HS` *exactly* on the non-minimal orthonormal frame. **Orthogonal + gauged suffices** (the CLAUDE.md "+ minimal" was wrong). |
| `manifold_dim` / `tangent_space_dimension` | **NO** | the formula matches the true SVD rank (103) on the non-minimal frame. Correct for any ranks. |
| `MANIFOLD.retract` | **soft caveat only** | still a valid first-order retraction on a non-minimal frame (rel.err 6e-6); it just **drops the numerically-redundant rank** (tucker 4→3) — desirable cleanup, not an error. Minimal buys only *strict* rank preservation. |
| `MANIFOLD.randn` / `project_oblique` | **NO (correctness)** | gauged direction is valid on any orthonormal frame; "true Gaussian / HS-matching" is on whatever the stored-rank tangent space is, which is correct. |
| `MANIFOLD.project`/`project_ambient`/`transport`, gauge `Π` | **NO** | orthogonal suffices (already known). |

**Verdict: minimal rank is NOT a correctness precondition for any verified op.** So we **do not wire it as
a checked precondition** — a structural-minimal check would raise on legitimate rank-redundant bases for
no correctness reason. (Also corrects an earlier claim: `t3_orthogonal_representations` does **not**
guarantee structural minimality — e.g. `randn((10,11,12),(4,5,4),(1,2,3,1))` → `has_minimal_ranks=False`.)

Minimal rank lives on as a **caveat** (`retract` strict rank-preservation) and as **diagnostic checkers**
(built, not wired into any op): structural `has_minimal_ranks` (existing); numerical
`TuckerTensorTrain.has_numerically_minimal_ranks(rtol)` (structural-first → `t3svd` compare) and
`T3Frame.has_numerically_minimal_ranks(atol)` (orthogonal + structurally-minimal ⟹ numerically-minimal, no
SVD; non-orthogonal frames return `False` — the SVD path for them is deferred, "maybe we don't need it").
**Super-safe mode was considered and dropped** (no op needs it; it would be a ~no-op for orthogonal
frames). Naming convention: bare `minimal_ranks` = structural; `numerically_minimal` = numerical.

## Implementation notes (for S3–S5, not decisions here)

- **`same_frame(b1, b2)` = `b1 is b2 or safety.frames_equal_or_skip(b1.data, b2.data)`** — the `is`
  fast-path keeps the common eager case O(1); the value compare only runs (and only matters) when objects
  differ but values match (jit round-trip) or genuinely differ. The tolerance is **jax-aware** (`rtol_jax`
  if any input is a jax array, else `rtol_numpy`); under a trace it skips (returns pass). Mechanism in
  `t3toolbox/safety.py` (S1, done).
- **Where checks live (razor):** to serve raw-`.data` users, the ORTH/GAUGE checks belong at the **backend**
  functions that consume raw data (`tv_operations.*`, which already have `is_orthogonal`-style
  residual helpers), with the frontend methods inheriting them. `same_frame` is a frontend concept (it
  needs the two `T3Frame` objects); the backend's analogue is "the caller passed one shared frame", so the
  same-frame check is naturally frontend-only.
- **Cost mitigation (done in S5):** cache the atol-*independent* **residual**, not the atol-bool —
  `T3Frame.orthogonality_residual` and `T3Tangent.gauge_residual` are `@cached_property`s, and
  `is_orthogonal(atol)`/`is_gauged(atol)` compare against them. A fixed frame/tangent in an inner loop is
  contracted once; the atol stays a free parameter.
- **Trace detection (done in S5):** the eager-only skip must fire even when the checked operand is a
  *closed-over concrete* array (not a tracer) while globally inside a trace — a jnp op then still yields a
  constant tracer. `safety.is_tracing` detects the global trace state (`jax.core.trace_state_clean`, with a
  committed-array-probe fallback), not just whether the passed arrays are tracers.

## Sign-off resolutions (Nick, 2026-06-19 — recorded here so the questions don't look open)

1. **`retract` precondition = ORTH only** — confirmed. Retract is gauge-invariant (gauge relabels the
   variations, not the represented step); GAUGE is not checked there.
2. **`MANIFOLD.inner`/`norm` check all three** (same-frame + both operands GAUGE + frame ORTH), with the
   frame/tangent residuals cached (`orthogonality_residual` / `gauge_residual`) so a fixed frame in an
   inner loop is contracted once.
3. **Frontend-only enforcement** — the backend stays check-free; a backend mirror for raw-`.data` users
   is deferred (the checkers themselves are public, so a backend user self-guards with the same tools).
4. No reclassifications. The shipped state is the user page
   [`../numerical_contracts.md`](../numerical_contracts.md).
