# Precompute and caching: the design principle

> Where derived data may be computed once and reused during optimization, where it must not be, and
> why — the first-principles audit (2026-08-20) behind `GeometryOps.precompute`, the regularizer
> `aux=` threading, and the decisions recorded at the bottom. Written after the SF-T3 build, whose
> per-frame companion made the question concrete; the principle is general.

## The economics, and the real cost axis

Precompute-and-store pays when `(cost of X) × (reuses within X's lifetime)` exceeds computing it
once plus *carrying* it. In this library the carrying cost is **not memory** — sweeps, companions,
and masks are tiny next to a matvec's working set. The real cost is **staleness risk and state
channels**: a cache that can be stale is the classic source of silent wrongness, and a cache in the
wrong place breaks the execution model. So the design question is never "should we precompute?"
(usually yes) but "**where does the cached object live, and what invalidates it?**"

## Three mechanisms, in increasing order of danger

1. **Lazy caches on immutable objects** — `functools.cached_property` on the frozen dataclasses
   (`T3Frame.orthogonality_residual`, `T3Tangent.gauge_residual`). Invalidation is impossible
   because mutation is impossible. Safe, but limited to things derivable from one object alone.
2. **Explicit aux threading at scope boundaries** — compute X at the moment its inputs settle and
   pass it along as a value: the sampling `sweep`, the SF-T3 companion (`GeometryOps.precompute` →
   `LocalModel.geom_aux` / the frontend models' `geometry_aux`), the packed sample. This is the
   functionally *pure* form of caching: reuse follows **provenance** (the same object flowing
   through the loop), invalidation is structural (a new frame simply *is* a new aux), and it
   composes with jit (the aux is a pytree **leaf**, like `sweep`).
3. **Hidden caches on data objects** — a memo dict on a frame, keyed by whatever. In a jax-pytree
   world this is wrong by construction, not by convention: flatten/unflatten drops or duplicates
   hidden state, a trace poisons it with tracers, and it silently perturbs recompile keys. We
   re-derived this from first principles during the audit (deliberately setting the house
   convention aside) and landed in the same place: **mechanism 2 is the only cache that is correct
   by construction here.** A related non-mechanism: *value-memoization* (hash the arrays, cache by
   content) — rejected outright; hashing costs what it saves, float-equality semantics are a swamp,
   and reuse should follow provenance, not value coincidence.

## The scope ladder

Match each cached object to the widest scope whose invalidation is *structural* — keyed by an
object identity/value the loop already threads:

| scope | invalidated by | lives as | examples |
|---|---|---|---|
| per-problem | new data/sample | `Problem` fields | packed sample/data, ω, canonical `groups`, `row_splits` |
| per-rank-level | a genuine rank change | value-hashed jit aux / closure statics | uniform masks, compilations |
| per-frame (outer step) | the iterate moving | a **leaf** beside `sweep` | the sampling sweep, the SF-T3 companion |
| per-call | — | nothing cached | the matvec itself |

The refined form of "if we can precompute, we should":

> **Precompute at the widest structurally-invalidated scope, through a channel the loop already
> threads. Once the channel exists, hoisting one more item is nearly free — do it. Creating a NEW
> state channel for a sub-percent win is where to stop. Never hide state in pytree data objects.**

## The audit's findings (2026-08-20)

**Fixed — the regularizer leak.** `Regularizer.hessian`/`quadratic` called `geom.project(frame, p)`
*without* aux, inside `gn_hessian`/`gn_quadratic` — i.e. once per CG matvec. For the shared
geometries that recomputed the whole companion (a lossless right sweep + `d` small GEMMs + the
group SVDs) **per matvec** — exactly the serial-small-SVDs pattern `precompute` exists to avoid,
leaking back through one seam. (The original "accepted cost" note was mis-scoped: acceptable per
*model*, not per *matvec*.) The protocol now threads `aux=None` through
`gradient`/`hessian`/`quadratic`, the backend `LocalModel` and both frontend models pass their
stored companion, and a regression test asserts zero companion rebuilds across repeated regularized
matvecs. This also pre-builds the channel a future shared-geometry Grasedyck–Kramer
`SingularValueRegularizer` needs — its natural input *is* the companion's `svd_s`, a fourth
consumer of the same per-frame object (projection, retraction, regularizer, spectrum diagnostics).

**Recorded, not built — retract→frame fusion.** The manifold retraction's internal T3-SVD returns
a **left-orthogonal** point, and the outer loop's next act is `frame(x)`, whose first stage is a
left-orthogonalization (`t3_orthogonal_representations` even has an unused
`already_left_orthogonal=True` fast path). Deeper: frame construction computes the centers `HH`
and discards them; the SF-T3 companion then reproduces them by the bit-identical re-sweep. A fused
`retract → (point, frame, companion)` seam would save roughly one lossless sweep plus a third of
frame construction per outer step — low single-digit percent of a Newton step (gradient + 10–30
matvecs), but a *real fraction* of a first-order manifold step, where frame construction is ~⅓ of
the work. Cost: a new fused seam, and the loss of a genuine virtue of the current split — the
companion derives from **any** frame however obtained (transport targets, user-built frames), one
code path. Verdict: build only if first-order manifold fitting becomes a hot path.

**Examined and left alone:**

- *Retaining `HH`/`S` on the frame* — mechanism 3; see above. The companion's re-sweep is the
  pure-form equivalent at the cost of one cheap sweep per outer step (and its **bit-identity** with
  the construction is what makes the `⟨O, H⟩` pairing exact — see
  [`sharing_internals.md`](sharing_internals.md)).
- *Continuation reading `κ_g` off the last companion* instead of a fresh grouped `t3svd` — true
  that `svd_s` *is* `s_g`, but continuation is a between-solves decision (once per level), the
  fresh call also supplies the TT spectra, and the coupling buys nothing measurable.
- *Sharing-specific sampling precompute* — probe vectors differ per mode, so tied factors share no
  contractions; nothing to hoist.
- *Standalone frontend calls* (`geom.project(v)` outside a model) recompute the companion by
  design — the `aux=None` fallback keeps every entry point correct without a model; interactive
  use is not the hot path.

## Checklist for adding a precomputed object

1. Name its exact invalidation scope; find the existing channel at that scope (`Problem` field /
   value-hashed aux / a leaf beside `sweep`). If none exists, that is a design decision, not a
   refactor — stop and weigh it.
2. Keep an `aux=None` recompute fallback so standalone calls stay correct (the
   `shared_geometry_ops` pattern).
3. The object flows as a pytree **leaf** if it holds arrays; only hashable statics go in aux keys.
4. Audit *every* consumer seam — the regularizer leak lived precisely in the one seam nobody
   re-checked after the channel was added.
