# Handoff — jit / OO-layer architecture (branch `geometry-refactor`)

> **RESOLVED (2026-06-19) — see [`docs/safe_unsafe_mode_plan.md`](safe_unsafe_mode_plan.md).** The root
> cause was the same-base guard being a *numerical* check faked as *structural* (object identity); the
> fix is an honest numerical same-frame check gated by a safe/unsafe mode, which makes `basis`-as-leaf
> viable (no recompile) and lets you jit the frontend directly. This handoff is kept for the reasoning
> and empirical findings that led there.

*Paused 2026-06-19, mid design-discussion. **Nothing is broken; G1 and G2 are done, tested (218 pass),
committed, and pushed.** This note captures an OPEN design question about jax/jit and the OO frontend
that we did not resolve. Read before resuming — and resume rested; this is a design-taste call, not a
regression.*

## Where the branch is
- `geometry-refactor` off `main`. Commits: **G1** geometries + thin `T3Tangent` (`3b64c997`), **G2**
  generic `GaussNewtonModel` (`de8a51d1`), **jax-compat + GN forward ops** (`0da6f51d`). All pushed.
- Working: `MANIFOLD`/`COREWISE` singletons; one geometry-generic `GaussNewtonModel` + `apply_model`/
  `entries_model`/`probe_model`; `jacobian(p)`/`gn_quadratic(p)`. See `docs/geometry_refactor_plan.md`.
- **G3 (`optimizers.py`) NOT started** — it's downstream of the question below (the optimizer is where
  jit matters and where the tangent interface is consumed).

## The problem we were chewing on
A frontend user should be able to **jit an optimizer inner loop** (Cauchy-SGD / Newton-CG). Three things
in tension: **(1)** jit the GN-Hessian matvec (the hot kernel); **(2)** avoid recompiling every Newton
step (the base changes each step); **(3)** keep the eager same-base guards (e.g. `T3Tangent` addition
denies mixing tangent spaces).

### What we established (empirically; scratch scripts run, not committed)
- **Recompile happens iff a base-carrying object CROSSES a jit boundary.** To preserve the identity
  guard the base must be aux/constant → baked in → recompiles per base; closing the model over also
  bakes the base in → recompiles. Either way, crossing a varying base ⇒ recompile.
- **Whole-step jit works (Pattern 1):** `jit(step)(X)`, build base/model *inside* → base is
  traced-internal → **compiles once, reused across steps** (verified `traces == 1`). The natural
  per-step jit for Cauchy-SGD; no recompile.
- **A `T3Tangent` cannot round-trip a jit boundary and keep its basis identity** — each crossing
  reconstructs the basis as a fresh object. So `jit(lambda p: model.gn_hessian(p))` returns `Hp` at a
  *different* basis object than the eager `r`/`p`, and the next eager `p.inner(Hp)` / `r − α·Hp`
  **false-fails** the guard. Making the base a traced leaf (to kill recompile) is exactly what destroys
  the output identity. **So "jit the frontend matvec" cannot give all three at once.**
- **Clean resolution — jit the FUNCTIONAL CORE, keep the frontend EAGER, re-home the output.** Core =
  `core(base_data, sweep, variations) -> variations` with base/sweep/variations as **traced arrays** →
  compiles ONCE, no recompile. Frontend `gn_hessian` stays eager: guard the input, call the core,
  re-home the result at the eager base (`T3Tangent(self.base, hp_vars)`). Then every eager tangent
  shares the one eager base object → all guards hold and nothing recompiles. **Verified: 1 compile
  across 3 different bases, every eager guard passes.**
  - Principle: **base = *context*** (eager-owned identity; passed into the jit as a traced VALUE for
    no-recompile); **variations = *payload*** (the only thing crossing). Frontend owns identity; the jit
    sees values. This is the library's thin-eager-frontend / functional-backend split, taken seriously.
  - **Consequence: this needs NO pytree-registration change and KEEPS `T3Tangent`.** The recompile only
    ever threatened us because we eyed "jit the frontend." **So the jit problem does NOT force an
    architecture change** — jit-the-core works for whatever shape the frontend takes.

### Constraints on a "jit the core under the hood" mechanism (Nick)
1. **jit must be OPTIONAL, default off.** Ragged core shapes → XLA unrolls the core loop before
   compiling → very slow compile with many (or even moderate) cores. (A motivation for the uniform
   layer.) jit must never engage unless explicitly asked.
2. **The user invokes jit from the FRONTEND** — not by dropping to the functional backend.
3. **REJECTED: any global / ambient jit toggle** (context manager, module flag, contextvar). The bar for
   global toggles is very high; the jit choice must be **per-function, at the call site**.

### Where we got stuck
Per-function opt-in jit **and** frontend-invoked **and** no global toggle **and** no flag-on-every-
function (there are *many* numerically-intensive functions) — these pull hard against each other. A
per-call `use_jit=` arg is "per-function" but is the proliferation we want to avoid; a global toggle
avoids proliferation but is rejected. **No clean mechanism agreed.**

## The deeper question (the real reason we paused)
Nick's read: the **OO layer is the source of the friction** — "my code was all good when it was
functional; after adding this OO layer it's a sequence of neverending headaches." The functional backend
(basic types) jits cleanly, per-function, with no global state and no identity wall — *because there is
no bundled `(basis, variations)` object to round-trip.* The bundle is what creates the aux-vs-leaf /
identity / recompile tangle.

So the architectural question from earlier in the discussion is live (and **decoupled from jit** by the
jit-the-core finding — it's now an ergonomics / taste call, not forced by performance):

- **Keep** `T3Tangent` as-is (basis-as-aux); jit via the functional core behind an eager frontend. Jit
  problem solved; the "four-way grab-bag" aesthetic concern remains.
- **Shrink** `T3Tangent` to its honest core (the geometry-independent both-ops: `to_dense`/`inner`/
  `probe`/`+`), moving basis-only ops → `T3Basis`, variations-only → `T3Variations` (geometry-dependent
  already on the geometries). Optionally flip basis-as-leaf.
- **Delete** `T3Tangent` entirely (Nick's four-way split): basis-only → `T3Basis`, variations-only →
  `T3Variations`, geometry-independent both-ops → a namespace/singleton, geometry-dependent → the
  geometries, binary ops (`add`) → free functions taking `(var1, var2, shared_basis)`. Pure functional.
  **Cost:** the same-base *hard guard* downgrades to "by construction / trust the caller" (bare
  variations don't carry their basis, so there's nothing to identity-check); plus ergonomics +
  discoverability. (See `geometry_refactor_plan.md` discussion + the four-way assessment in chat.)

## Open decisions for next session
1. **The per-function opt-in jit mechanism** (no global toggle): how does a frontend user say "jit *this*
   matvec" at the call site, given the jit must live on the functional core? Not-yet-evaluated
   directions: a per-operation **jitted-operator builder** the user constructs only where wanted (e.g.
   the model hands back a compiled `T3Tangent -> T3Tangent` matvec for the inner CG, eager re-home
   inside); exposing the functional cores as first-class with thin frontend-typed wrappers; or
   re-examining whether constraint (2) ("don't bypass the backend") is the right line, *given the
   functional layer is precisely what jits cleanly per-function with no global state* — which is the
   crux of Nick's "functional is the way."
2. **OO-vs-functional architecture:** keep / shrink / delete `T3Tangent`. Aesthetic + ergonomic, no
   longer forced by jit.
3. **G3 (`optimizers.py`)** depends on (1)+(2): the optimizer interface differs sharply between an OO
   tangent and basic-typed variations, and on how the matvec is jitted.

## Decided / committed (not in question)
- G1, G2 done, 218 tests pass. Geometries pytree-registered; `GaussNewtonModel` deliberately **not**
  registered; the two jit patterns + the `T3Tangent` basis-as-aux **REVISIT** note are documented
  (`fitting.py` "Jitting an optimizer", `manifold.py` registration block, plan §8). `jacobian`/
  `gn_quadratic` added. Both Hilbert examples run; the Newton-CG example reproduces pre-refactor
  iterates bit-for-bit.
