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
