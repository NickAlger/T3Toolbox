# Safe / unsafe mode + numerical contracts — the resolution to the jit / OO predicament

*Design locked with Nick (2026-06-19), branch `geometry-refactor`. This **resolves** the open question
parked in [`docs/jit_oo_handoff.md`](jit_oo_handoff.md) (per-function jit + the OO-vs-functional tension)
and supersedes it. It adjusts two house rules (see §8). Read §0–§3 for the reasoning, §4–§5 for the
critical execution rule, §7 for the build order.*

## 0. Resuming from a fresh context (read this first)

If you are a future Claude with little memory of the conversation that produced this: the library hit a
multi-session wall trying to make the OO frontend (`T3Tangent`, `GaussNewtonModel`) jit-friendly. Every
attempt traded one of {jit the frontend, no recompile-per-step, keep the same-base guard} against
another. The **root cause** (this document) is that the same-base guard was a *numerical* check faked as a
*structural* one (object identity), and that fake is what forced `basis`-as-`aux_data` → recompile, and
what made jit round-trips false-fail the guard. The fix is to do the honest numerical check, gated by a
**safe/unsafe mode**. This dissolves the jit problem *and* strengthens the library (enforced numerical
contracts). The decision is made; §7 is the build order. Confirm the **precondition catalog** (§5) with
Nick before wiring checks — that is the make-or-break step.

**Branch/state:** `geometry-refactor`. G1 (geometries + thin `T3Tangent`) and G2 (geometry-generic
`GaussNewtonModel`) are done, tested (218 pass), pushed. This plan is the next phase; G3 (`optimizers.py`)
is downstream of it. Plan for the geometry refactor itself: [`docs/geometry_refactor_plan.md`](geometry_refactor_plan.md).

## 1. The root-cause insight

"Are these two tangents in the same tangent space?" means "are these two frames the **same frame**?" —
which is `frames_equal(b1.data, b2.data)`, a **numerical** comparison. We implemented it as `b1 is b2`
(object identity) only because the house rule forbade numerical checks. That identity proxy is the seed of
the entire jit predicament:

- identity must survive `flatten`/`unflatten` ⟹ `basis` must be `aux_data` ⟹ a jit compile-time constant
  ⟹ **recompile every time the base changes** (every Newton step);
- identity is **lost on a jit round-trip** (each crossing reconstructs the basis as a new, value-equal
  object) ⟹ a matvec *output* combined with an eager residual **false-fails** the guard.

Replace the costume with the real check and the chain unwinds. Empirically (scratch, 2026-06-19):

```
IDENTITY guard  (base is base2) : False   # value-equal rebuild / jit round-trip -> FALSE-FAILS
NUMERICAL guard (frames_equal)  : True    # tolerates value-equality -> passes
NUMERICAL guard, different base  : False  # still catches the real error
```

So `frames_equal` accepts exactly the case identity rejects (the jit round-trip), while still catching a
genuinely different base point (measure-zero false positives — different base points have different frames).

## 2. The design: two modes

- **safe mode** — perform **both** structural and numerical checks before an operation.
- **unsafe mode** — perform **only** structural checks.

Controlled by an ambient **`safety_rtol: float | None`**: a float is safe mode at that tolerance;
`None` is unsafe mode. (Prefer this over a `safe_mode: bool` — the checks need a tolerance anyway.)

- **Mechanism: a `contextvars` context manager**, not a bare module global — thread/async-safe and
  scopable: `with t3.unsafe(): ...` / `with t3.safe(rtol=1e-9): ...`, plus a module default. A single
  `check(condition, msg)` / `check_frames_equal(b1, b2)` helper reads the ambient `safety_rtol` and
  no-ops when unsafe **or when tracing** (see below).
- **Numerical checks are eager-only.** Inside a jit trace you cannot `if not allclose(tracer): raise`
  (data-dependent control flow on a tracer is illegal). So checks are **skipped under tracing**. This is
  *correct*, not a compromise: **jit is for performance, which lives squarely in unsafe mode.** You
  validate eagerly in safe mode, then jit (effectively unsafe) for speed.
- **The numbers are identical in both modes.** The mode only governs *whether errors are caught*, never
  the computed result. This is the `assert` / `python -O` precedent, and is what makes the one ambient
  global acceptable (see §8).

## 3. What it dissolves (the jit story, end to end)

With the same-base guard numerical and value-based, the basis no longer needs identity, so:

- **`T3Tangent`: `basis` becomes a leaf** (`children = (basis, variations)`, `aux = None`). The base flows
  as traced data ⟹ **no recompile** when the base changes.
- **A matvec output passes the eager guard** against an eager residual because the *values* match
  (`frames_equal` True) even though the objects differ ⟹ **no re-homing needed**.
- **`jit(lambda model, p: model.gn_hessian(p))` just works** — per-function, from the frontend, no global
  jit toggle, no special operator plumbing. To get *no recompile*, also **register `GaussNewtonModel`
  with `(base, sweep, sample, residual)` as leaves and `(geometry, kind)` as `aux`** (stateless statics),
  and **store the sweep as a field** (computed once, carried as a leaf ⟹ reused across CG iters, not
  recomputed). Then the matvec compiles **once for the whole solve**.
- **Settles keep/shrink/delete `T3Tangent`: KEEP it, basis-as-leaf.** The bundle was never the problem;
  the identity hack was. (The four-way-split idea remains an *optional* aesthetic cleanup, not forced.)

basis-as-leaf re-exposes the frame to `jax.grad`/`tree_map`. **This is fine, by design:** a user who wants
grad-w.r.t.-variations only writes `g = lambda v: f(T3Tangent(b, v)); jax.grad(g)` (basis closed over);
and grad-w.r.t.-the-basis becomes an available feature rather than a hazard.

## 4. The critical execution rule: enforce *preconditions*, not *caveats*

Safe mode must check the genuine **precondition** of each op (the op is *undefined / numerically wrong*
without it), **never** a semantic **caveat** (the op is *valid* but its *meaning* depends on a property).
Getting this wrong turns a "safety feature" into a regression that rejects legitimate use.

The canonical trap is **`inner`/`norm`**: they are valid on *any* frame; orthogonality is only the
condition under which they *equal* Hilbert–Schmidt. `CorewiseGeometry` deliberately operates on the
**non-orthonormal `(U,G,G,G)`** frame, so "`inner` requires orthogonal" would reject the entire corewise
path.

### 4.1 `inner`/`norm` move onto the *Geometry* (Nick's refinement)

The semantics belong to the geometry; the raw computation stays on `T3Tangent`:

- **`ManifoldGeometry.inner(t1, t2)` / `.norm(t)`** — the **Hilbert–Schmidt** inner product / norm. In safe
  mode it checks **same-frame** *and* **gaugedness** (gaugedness is the precondition for the corewise dot
  to *equal* HS — see Appendix A.3), then does the corewise computation.
- **`CorewiseGeometry.inner(t1, t2)` / `.norm(t)`** — the **Euclidean** inner product / norm. Checks
  **same-frame** only; **no gauge check** (Euclidean does not need it).
- **`T3Tangent.corewise_inner(other)` / `.corewise_norm()`** — the **raw coordinate** dot / norm, *renamed*
  from today's `inner`/`norm`. Honest: no HS claim. (Same-frame is still its precondition for the binary
  `corewise_inner`; the catalog in §5 fixes exactly where that check sits.)

This cleanly separates **contract** (the geometry's `inner` carries and checks it) from **computation**
(the renamed raw op). It is the model for every other precondition/caveat blur we find.

## 5. The contract catalog (the make-or-break work — do this first, with Nick)

> **The full catalog is drafted in [`docs/numerical_contract_catalog.md`](numerical_contract_catalog.md)**
> (op → genuine precondition, the flagged blurs, the minimal-rank resolution, sign-off questions). The
> starter table below is kept for orientation.

Sweep the **verified** modules (`tucker_tensor_train`, `basis_variations_format`, `manifold`,
`corewise`, the fitting layer, `probing`, and their verified backends) and, for each operation, record its
**genuine numerical precondition** — distinguishing it from any semantic caveat. Starter table (to be
completed and signed off):

| op(s) | genuine numerical precondition | notes |
|---|---|---|
| `+`, `−`, `corewise_inner` | **same frame** (`frames_equal`) | binary tangent linalg; `corewise_norm` is unary (none) |
| `MANIFOLD.inner` / `.norm` | **same frame** + **gauged** | HS-faithfulness; gaugedness is the precondition for HS |
| `COREWISE.inner` / `.norm` | **same frame** | Euclidean; no gauge |
| `MANIFOLD.retract` | **orthogonal** base | the doubled-rank truncation needs it |
| `MANIFOLD.project` (gauge Π), `project_oblique`, `project_ambient`, `transport` | **orthogonal** base | manifold embedding ops |
| `_require_at_base` (fitting model) | **same frame** (input tangent vs model.base) | replaces today's identity guard |
| minimal-rank-sensitive ops (some `inner` HS faithfulness, `retract` rank preservation, …) | **structural** minimal rank only | see §6 — no numerical minimal-rank check |

**Caveats that are NOT preconditions** (do *not* enforce): "`inner` equals HS only when orthogonal+gauged"
(that's the geometry split above); anything documented as "works but the result means X under Y".

## 6. Refinements / decisions (settled with Nick)

- **Under jit: checks skipped (eager-only).** jit = performance = unsafe. Accepted.
- **`contextvars` context manager** over a bare module global (thread-safe, scopable). Accepted.
- **`safety_rtol: float | None`** over `safe_mode: bool` (the checks need a tolerance; `None` = off).
- **Minimal ranks:** keep the **structural** minimal-rank check; **skip the numerical** one (it's an SVD to
  verify and the requirement is often soft). Do **not** rely on "NaN is the visible sign" — it is not
  universal (some non-minimal failures are wrong-but-finite); instead **document the sensitive ops**.
- **basis-as-leaf grad/`tree_map` exposure:** accepted as fine / a feature (see §3).

## 7. Build plan (slices)

1. **S1 — the safety mechanism.** `safety_rtol` as a `contextvars` var + `with t3.safe(rtol=...)` /
   `t3.unsafe()` context managers + a module default; a `check(...)` / `check_frames_equal(...)` helper
   that no-ops when unsafe **or** tracing (detect tracers). Pure plumbing; no behavior change yet.
2. **S2 — the precondition catalog (§5).** Sweep verified modules; produce the full precondition table;
   **Nick signs off** before any check is wired. This is the make-or-break step.
3. **S3 — move `inner`/`norm` to the geometries.** Add `MANIFOLD.inner`/`.norm` (same-frame + gauge check)
   and `COREWISE.inner`/`.norm` (same-frame). Rename `T3Tangent.inner`/`norm` → `corewise_inner`/
   `corewise_norm` (raw). Re-point consumers (fitting `evaluate`/two-form, examples, tests).
4. **S4 — numerical same-frame guard + basis-as-leaf.** Replace the `is`-identity guard
   (`_check_same_tangent_space`, `stack_tangents`, fitting `_require_at_base`) with `check_frames_equal`
   (safe-mode, eager-only). Flip `T3Tangent` to basis-as-leaf; register `GaussNewtonModel` with all-leaf
   data + statics as aux; store the sweep as a field. Delete the aux/recompile machinery + the
   basis-as-aux REVISIT note.
5. **S5 — wire the remaining preconditions** from the §5 catalog (orthogonal for retract/project/…, etc.).
6. **S6 — verify.** `jit(matvec)` compiles once across bases (no recompile); eager guards fire in safe
   mode, catch real errors, and tolerate value-equal frames; numbers identical to today; both Hilbert
   examples + the full suite pass. Update CLAUDE.md (§8) and the geometry plan.

## 8. House-rule changes (record + update CLAUDE.md when implemented)

Two principles change. Both are justified, and consistent with each other and with prior decisions.

1. **"Numerical problems → warn, don't enforce" → "numerical problems → enforce in safe mode, skip in
   unsafe / under jit."** A strict improvement: the old rule's real motivations (cost; tolerance-fuzziness;
   deliberate violation) are each answered by *unsafe mode* + *`safety_rtol`*. And we were already
   violating it covertly (the identity hack *is* a hidden numerical check) — we are replacing a brittle
   one with an honest one. (This rule is currently stated in CLAUDE.md "House philosophy".)
2. **"No globals" → "one ambient `safety_rtol` (`contextvars`)."** The narrow, justified exception:
   - it is **correctness-neutral** (eager and the numbers are identical; only error-catching changes) —
     the `assert`/`-O` precedent, which is exactly why it is the one global worth having;
   - it is **naturally a coarse mode** (develop-with-checks / ship-without), unlike the *jit* toggle we
     rejected, which is naturally **per-kernel** — so a global fits safety and not jit, and rejecting one
     while accepting the other is coherent;
   - `contextvars` makes it thread-safe and scopable, the disciplined form of ambient config.

## 9. Open questions

- The **full §5 catalog** — the one piece that needs real design care + Nick's sign-off.
- Whether any safe-mode check is **too expensive even for development** (e.g. `is_orthogonal` is
  contractions) → possibly a tiered level, or just rely on `safety_rtol=None` locally. Decide during S2.
- Exact placement of the same-frame check across `corewise_inner` vs the geometry `inner` (S3/S5).
