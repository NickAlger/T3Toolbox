# OPEN QUESTION — what extension is the fitting layer actually meant to support?

*Opened 2026-08-21, out of the optimization-layer restructuring. A standing question, not a thread:
per `CLAUDE.md` it is unresolved rather than superseded, and must not be archived until it is answered.*

## The unease

Nick, reviewing the restructuring: *"it feels like there is something off with the architecture here.
Like, we have reached for generic tools rather than carefully considering what we are trying to do, what
we want to support, then crafting a specific architecture with this in mind. Now we have to consider all
these weird cases which the generic machinery technically allows."* The unease predates the
restructuring; the restructuring made it visible.

**The decision is to defer**, on the reasoning that this is hard to settle in the abstract and much
easier with a real downstream requirement in hand. This arc is itself the evidence: it started from a
concrete need — jitting a fit in T3Polynomial — and produced a better answer than speculation would
have. This file exists so the next person starts from the facts rather than re-deriving them.

## The concrete evidence, gathered while it was fresh

**The extension point is already only half-wired.** `t3toolbox/optimizers.py` accepts only the six kind
*strings*; a user-written `SamplingKind` cannot be driven by the frontend optimizers at all. So the
"N sampling kinds" axis advertised in `docs/fitting_and_optimization.md` §1 is, end to end, "six kinds
plus a backend escape hatch that nothing has used." We pay the full price of an open extension point
without delivering it.

**The price, itemized.** Every one of these exists to serve open subclassing, and would mostly evaporate
under a closed set of kinds with explicit registration for a genuinely new operator:

- nine abstract stubs on `SamplingKind`, each needing a message;
- the mixin lattice (`ScalarOutputKind` / `ProbeOutputKind` × the four sample-layout mixins ×
  `_DerivativesKind`) — which exists to factor behaviour across the six built-ins, not for users;
- `has_block_sumsq`, a flag declaring whether a method is implemented, because Python has no way to say
  "optional method";
- `ValueHashedFields.require_parameters_are_fields`, a guard against a user writing a subclass wrongly;
- the subclass identity questions in `parameters_not_closures.md` → *Honest limits* (a mutated class
  attribute or a monkeypatched method silently defeats the cache).

**What is genuinely load-bearing** and would survive any redesign: value identity for the *parameterized*
built-ins (a `probe_derivatives` kind at order 2 with weight ω is a distinct compiled program), and the
geometry carrying its fixed rank. Those are requirements, not generality.

## The purpose, stated (Nick, 2026-08-21)

> *"the point of the open surface is to allow the user to use our machinery to fit tensors with their
> custom objective function."*

That is much more specific than "open subclassing", and it changes one thing immediately: **the
half-wired frontend is a real gap, not a curiosity.** If the goal is user-supplied objectives, then
`t3toolbox/optimizers.py` accepting only the six kind *strings* is the single thing most in the way —
a user's kind can reach `backend.optimizers` but not `newton_cg(geometry, kind, ...)`.

## Candidate extensions, and what each would cost

Nick's brainstorm, with what the current architecture would require of each. Ordered by how hard they
push on it, easiest first. None of this is a design — it is what a designer would need to know.

### 4. Fusion of several data sources — **fits today**

Fit against apply data *and* probe data *and* entries in one objective. Mathematically clean: the
Gauss-Newton model is linear in the residual, so `JᵀJ = Σ Jᵢᵀ Jᵢ` and the sources just add. A composite
kind holding the sub-kinds as a field is the natural shape, and the identity composes correctly —
verified: a kind whose field is a tuple of kinds compares and hashes by value, and differs when any
sub-kind differs. `sample` / `data` / `residual` become per-source tuples; `sumsq` sums; `transpose`
sums the gradients. Nothing in the current machinery objects.

### 2. Different sampling types — **what the kind abstraction is for**

- **Forward-only probing** — probe output for the last mode only, rather than all `d`. This is not just
  a new kind: the reductions assume `d` outputs, so it needs its own output-shape base beside
  `ScalarOutputKind` / `ProbeOutputKind`. Worth noting it is *not hypothetical* — the T3Polynomial
  surrogate has exactly this asymmetry (its weighted misfit carries a 4:1 reverse:forward multiplicity,
  which makes it a poor judge of a symmetry-constrained model), and forward-only probing is the operator
  that would fix it.
- **Probing by contraction with a structured tensor** — the existing `apply` is contraction with a
  rank-1 tensor; contraction with another T3 generalizes it. A genuinely new operator, whose math is the
  library's own (T3 inner products) but whose sweep is new. This is the strongest single argument for
  keeping the surface open rather than closing it to six.

### 3. Data from a general linear functional — **stresses the interface's shape**

Any linear functional `A·T`; apply / entries / probe are all special cases. The user supplies `forward`
and `transpose`, which is exactly what a `SamplingKind` is — so the *concept* fits perfectly. What does
not fit is the interface's built-in assumption that there is a **reusable frame sweep**:
`precompute(frame, sample) → sweep`, then `forward`/`transpose` consume it. That assumption is where the
library's efficiency comes from, and a general functional has no structure to precompute. Supporting it
means either a documented `sweep = None` path or a second, thinner protocol for unstructured operators —
a real interface question, not a subclass.

### 1. Custom weighting and scaling — **the sharpest stressor, and it hits what was just built**

Two cases that look alike and are not:

- **Static schemes** fit today: `ω[mode, order]` is a field, part of the value identity, folded into the
  compiled program as a constant. That is correct *because* it does not change.
- **Iterate-dependent schemes — weights from the tensor's singular values, or from an estimated Hessian
  diagonal — do not.** They change every Newton step, so as static `aux_data` they would force a
  recompile per step, which is precisely the cost this refactor removed. Making them work means the
  weight moves from **aux to leaf** (traced data), which means the kind stops being purely static and
  becomes a pytree with leaves of its own. That is a genuine design change and it cuts against the
  "parameters are fields, fields are the identity" rule.

  **There is a trap here today.** `_weight_matrix` calls `np.array(weight, dtype=float)`, so passing a
  **jax** array as a weight silently converts it to a host-numpy constant snapshotted at construction.
  For a static weight that is exactly right. For an iterate-dependent one it would silently freeze at
  the first value — the same failure shape as the aliasing bug fixed in `cb36ef9f`, but arriving through
  the front door.

  Also worth keeping straight, because the word collides (`CLAUDE.md` warns about this): a
  singular-value scheme sounds like a **metric** on the tangent — the Grasedyck–Kramer preconditioner,
  which already has machinery in the weighted layer (`T3FrameWeights`, `fv_absorb_weights`) and an
  already-deferred `SingularValueRegularizer` — not a **residual** weight `ω` on the measurements. The
  two enter the math in different places. Whichever a scheme needs, decide which one it is first.

## What this changes about the question

The four together say the surface should probably stay **open**, but that it is open in the wrong
places: closed where it matters (the frontend cannot take a user's kind) and generic where it does not
(arbitrary subclassing, with the hazards catalogued in
[`parameters_not_closures.md`](../docs/contributor/parameters_not_closures.md) → *Honest limits*). A
redesign informed by these would likely narrow the subclass surface and widen the frontend one — closer
to "here is the protocol for an objective, registered explicitly" than to "subclass anything".

## What would answer it

A real downstream case that needs a seventh sampling kind, a custom geometry, or neither. Until then the
honest position is that we do not know whether the extension point should be open, closed, or
registration-based, and closing it now would be designing for imagined users — the same failure the
unease is about.

## What to watch

- Each release that ships the open surface makes closing it more expensive. This release widened it
  slightly: `ScalarOutputKind` / `ProbeOutputKind` became public, because the documented "write your own"
  recipe was wrong without them (it said five operations; there are nine). That was a correctness fix to
  the docs, but it is also a wider promise.
- If a downstream need turns out to be for a new *geometry* rather than a new *kind*, the answers may
  differ: geometries have four implementations and a much smaller surface.
