# T3Toolbox — current handoff

_Updated 2026-07-10._

## Where we are

**Two big things closed this session (2026-07-10): the uniform frontend (U7), and the `basis`/`base` →
`frame` rename.**

1. **The uniform layer is CLOSED** — backend, optimizers, AND the frontend (U7). The four optimizers and the
   six `fitting.*_model` factories accept a `UniformTuckerTensorTrain` (**inferring** ragged-vs-uniform from
   `x0`, fully packed, **jit-compile-once**, verified == ragged); the roll-your-own surface
   (`fitting.UniformGaussNewtonModel`) and two worked examples ship too.
2. **The `basis`/`base` → `frame` rename is DONE** (`naming_review.md` §2): `T3Basis`→`T3Frame`,
   `.basis`→`.frame`, `bv_`→`fv_`/`ubv_`→`ufv_`, module + file renames, and the `base`-half (frame accessors
   + the **C stack → frame stack**). `T3Variations` and the math "basis" (Tucker factor matrices, vector-space
   bases) preserved. Verified functionally (593 tests, twice) **and** semantically (a two-agent word-level
   diff review — see the rename section below).

Plus a **packaging fix** (`pip install -e .` now works: `[tool.setuptools.packages.find]` + `readme =
README.md`). Branch `main`, direct commits, **pushed**. Full suite green (593 tests / 40,215 subtests).

**→ Uniform + the frame rename are complete. Next: the rest of the naming pass (cross-class method sweep +
small auto-fixes), then the doc pass** (see "Next steps").

## Done this session (2026-07-10) — U7: the uniform frontend surface

Closes the uniform-optimizer thread. Backend-mirroring, suite-gated per slice; history in
`dev/archive/uniform_optimizers_plan.md`.

- **U7a — optimizers frontend.** The four `t3toolbox/optimizers.*` accept a `UniformTuckerTensorTrain` x0 +
  the uniform geometry singletons, **inferring** the representation from x0's type (library-wide dispatch); a
  consistency guard requires the geometry to match. Refactored the shared `_problem` → `_setup` returning
  `(problem, init, rewrap)` (ragged path byte-identical); the uniform path calls `uniform_minimal(x0)`
  transparently and rewraps the bare supercore pair with x0's held shape+masks. Verified frontend uniform ==
  backend == ragged (tolerance).
- **U7b — `fitting.UniformGaussNewtonModel`** (the roll-your-own surface). The same-named `fitting.*_model`
  factories dispatch on x's type; the uniform model surfaces **UT3Tangent**-valued gradient / gn_hessian /
  gn_quadratic / jacobian / evaluate. **jit-compile-once:** the aux is **value-hashed** (`geometry`,
  `kind_name`, x0 rank masks, `order`, `weight`) and the packed kind is **rebuilt lazily** from it (a
  fresh-closure kind as aux would recompile) — verified `traces==1` for both the model-as-arg matvec and the
  whole-step patterns. Promoted `backend/uniform_fitting.pack_sample`/`pack_data` to public (the razor). Verified
  vs the backend `LocalModel` (all six kinds × both geometries), gauged gradient, GN self-consistency, guards.
- **U7c — examples + a doctest.** `examples/fit_hilbert_uniform_newton_cg.py` (minimal on-ramp: the ragged
  Hilbert apply fit + rank continuation, on the uniform layer — only x0/geometry change) and
  `fit_hilbert_uniform_probe_derivatives_newton_cg.py` (showcase: probe-derivative jets + uniform Newton-CG +
  rank continuation + per-order weight ω). Continuation keeps the `resize`/zero-pad glue **ragged** and drops
  into **uniform** only for the fit (a zero-pad is structurally minimal → `uniform_minimal` no-op → the
  gradient grows the rank); verified it matches the ragged run. Added a runnable `newton_cg` doctest (the
  frontend optimizers had none).
- **U7d** — this refresh + archived the plan.

**Design decisions settled with Nick this session (carry forward):** (1) same-named frontend functions infer
ragged/uniform from x0 (not separate uniform functions); (2) **build** the uniform roll-your-own model
surface (`UniformGaussNewtonModel`), not defer it — the "support frontend AND backend users equally"
principle (a frontend user rolling manifold L-BFGS needs UT3Tangent gradient/Hessian + the geometry's
inner/retract/transport); a **twin class** (not a polymorphic single class) — frontend precedent is distinct
classes with identical method names, and two pytree registrations are cleaner for jit.

**Architecture check (conceptual, no code):** confirmed the optimizer/fitting seams (`GeometryOps` /
`SamplingKind` / `corewise`) would accommodate a future **manifold L-BFGS** (needs one additive
`GeometryOps.transport` field, default `None`) and a **shared-core T3** (a new representation = new seam
impls closing over a sharing map, like uniform's masks) — and both are **non-breaking minor releases** for
users if seam fields are appended-with-defaults. Not built; recorded for when it comes up.

## Done this session (2026-07-10) — the `basis`/`base` → `frame` rename

Rationale: the orthogonal representation is overcomplete and doesn't require minimal rank, so "frame"
(admits redundancy) is right where "basis" (implies minimality) was not. Four commits (`basis`-half,
`base`-half, cross-line + doc fixes, doc reconciliation) + one review-fix commit.

- **Scope done:** `T3Basis`→`T3Frame`, `UT3Basis`→`UT3Frame`, `.basis`→`.frame`, `basis_*`→`frame_*`, prefixes
  `bv_`→`fv_` / `ubv_`→`ufv_`, module `basis_variations_format.py`→`frame_variations_format.py` (+ file
  renames), and the `base`-half: frame accessors (`model.base`→`.frame`, `geometry.base(x)`→`.frame(x)`,
  `_ragged_base`→`_ragged_frame`, …) and the **C stack → frame stack** (`base_stack_shape`→`frame_stack_shape`,
  `n_base`→`n_frame`) — the C stack is structurally a batch of frames.
- **Kept as `base` / `basis` (deliberate):** the plain manifold **point** (`base_point`, `base_masks`, "base
  point(s)"); the math **basis** (Tucker factor matrices "Tucker basis"/"bases", "standard/orthonormal
  basis", "basis vectors"); the named **`base-inner`** axis-ordering convention (could still → `frame-inner`
  for full consistency — a flagged nicety). `T3Variations` kept ("variation" is paper-confirmed).

> **METHODOLOGY LESSON (reuse for the cross-class sweep — it's another rename).** A big mechanical rename is
> NOT a safe blind `sed`. What worked: (1) an up-front **exhaustive token/phrase inventory** to separate the
> target sense from homonyms (`basis` = frame vs the math basis; `base` = frame-object vs C-stack vs plain
> point vs math "bases"); (2) **placeholder-protected** substring transforms; (3) a **whitespace-normalized
> (line-collapsed) audit** for corruption signatures — plain grep is line-based and misses phrases split
> across a line break; (4) a **word-level semantic review** (here: one subagent per commit diff). Real traps
> we hit and fixed: the "Tucker basis" factor-matrices (missed by the initial 3-phrase protection),
> **line-split** protected phrases (`base\n> point` → `frame point`), and — the one the review caught — the
> protection regex treating a **hyphen as a word boundary**, so `base-point`/`single-base-point` flipped to
> `frame-point` (my verification only checked the space form). Net: functional correctness rode on the test
> suite (loud failures); the *naming/prose* correctness rode on the audit + review, which found real defects
> the suite structurally cannot.

## Active thread (2026-07-09): the T3Toolbox software reference paper

A **new, separate** effort from the uniform work above — outlining a general-purpose **software /
algorithms reference paper** for T3Toolbox (TOMS-style: the T3 algorithms + the hard-won implementation /
correctness insights, consolidating the scattered tensor-train / hierarchical-Tucker literature and
specializing it to T3). Scope + curation live in **`dev/paper_scope.md`** — the 11 operation groups, the
`docs/` curation (which docs are paper-worthy), the archive scan, and two settled math findings.

> **This is NOT the T4S paper — keep them straight when writing.** **T4S** = Alger, Christierson, Chen &
> Ghattas (2026), *"Tucker Tensor Train Taylor Series"* (arXiv:2603.21141; local `t4s.pdf`) — the existing
> research preprint, a historical algorithm reference. The **toolbox paper** is a distinct, not-yet-written
> document: different purpose (a reusable **library reference**, *not* a research contribution), different
> venue (ACM TOMS + a Zenodo release-DOI). Some material was *cut from* T4S and lands in the toolbox paper
> (e.g. `docs/symmetric_probe_derivatives.tex`) — a relationship, not an identity.

**Status:** scoping only — no paper drafted. Two settled findings (in `dev/paper_scope.md`): the
**canonical-Gaussian-tangent** result (an orthonormal-gauged frame makes `MANIFOLD.randn` the standard
Gaussian on `T_xM` = the projected ambient normal) and that **minimal rank is a correctness precondition
for nothing**.

### Repo changes made this session (committed)
- **Restored** `dev/archive/t3svd_verification.md` → `docs/` — a proof-complete reference (generalized
  Oseledets error bound + a projection-argument rank bound) the knowledge-reorg had misfiled; it is cited
  by live code docstrings as the authority for "the bound and its proof."
- **Repaired the reorg's dead-link cluster** — ~30 `docs/<name>` references across ~15 code/doc files that
  pointed at files now in `dev/archive/`: the `t3m` plan refs → `docs/ttm_t3m_ht_note.tex` (the live theory
  home); every history/plan pointer → `dev/archive/<name>`. (Asserted script; re-grep clean; all `.py`
  recompile.)
- **Tightened** the `MANIFOLD.randn` docstring — orthogonality suffices; minimal rank not required (verified).
- **Struck** the already-resolved `entries_apply_probe.md` refresh line from CLAUDE.md's doc-pass TODO.

**Next on this thread:** walk the 11 groups (start at **Group 6** — `symmetric_probe_derivatives.tex` is
nearly a drop-in chapter), then the small nugget-extractions + cosmetic cleanups listed in
`dev/paper_scope.md`. Independent of the uniform / naming / docs work.

## Prior session (2026-07-06 → 07) — the uniform optimizer backend (U1–U6 + U5.6)

Built the backend the U7 frontend sits on; **full slicing detail in `dev/archive/uniform_optimizers_plan.md`.**
All backend-first (the geometry-generic optimizer bodies in `backend/optimizers.py` are unchanged apart from
one `inner`-seam swap); new code in `backend/uniform_fitting.py`, `backend/ufv_sampling.py`,
`backend/ut3_operations.py`; tests in `tests/backend/test_uniform_fitting.py` (+ jit in `test_dispatch.py`).
The load-bearing pieces that the frontend + future work rely on:

- **`GeometryOps.inner` seam** + `ufv_corewise_inner` (the honest masked/stacked coordinate dot; ragged =
  `corewise_dot`, byte-identical). **Arch-B1:** optimizer state = the bare supercore pair, masks are
  **loop-invariant state closed over** (only supercores traced) → jit-compile-once.
- **`uniform_{manifold,corewise}_ops`** + the `uniform_*_kind` `SamplingKind` builders (plain + jet twins) +
  **`uniform_least_squares_problem`** (packs the loop-invariant sample+data **once**).
- **The packedness-mirror convention** (U3.5): user-facing sampling ops infer & mirror packedness; the
  fitting split-seam is **packed-only** so the inner loop stays packed (`ut3_operations.{is_packed,pack_if_ragged}`).
- **U5.6 — the minimal-rank requirement** (still a standing constraint): the uniform optimizer needs a
  **minimal-rank base** (a non-minimal nominal rank desyncs the retraction from the fixed masks → mid-loop
  crash). `uniform_minimal(x0)` reduces to it (no-op if already minimal); `uniform_least_squares_problem`
  rejects a non-minimal x0. **U7's frontend calls `uniform_minimal` transparently.**

### Two investigation findings (from Nick's benchmark questions) — carry forward
- **Eager timing:** uniform-eager ≈ ragged-eager is NOT a bug. The uniform supercores carry ~2.6× more
  elements (padding) yet pipeline better; the two roughly offset (pipelining slightly wins on CPU). The real
  speedup (scan over `d` + recompile-free step) is a **GPU** story — re-run `dev/bench_uniform_vs_ragged.py`
  on a GPU server (bump `SHAPE`/`TR`/`W`).
- **No hidden unrolled loops in the major ops:** the TT sweeps use `lax.scan` (8 scans), the tucker
  orthogonalizations use a batched SVD, the sampling ops are fully vectorized (all constant in `d`). A small
  **residual ~6 jaxpr-eqns/mode** in orthogonalization+retract remains (not a naive Python loop — subtler;
  linear, not ballooning; amortized since compile-once). A follow-on only if very-high-`d` compile matters.
  Aside: `d ≥ 24` hits einsum's 26-letter label budget (a shared ragged/uniform edge case).

## Next steps

The uniform layer is closed. The agreed sequence (with Nick, this session) is **naming pass → doc pass**:

**1. Naming pass** (`dev/naming_review.md`).
- **`basis`/`base` → `frame` — DONE (2026-07-10).** Two commits: the `basis`-half (`T3Basis`→`T3Frame`,
  `UT3Basis`→`UT3Frame`, `.basis`→`.frame`, `basis_*`→`frame_*`, prefixes `bv_`→`fv_` / `ubv_`→`ufv_`,
  module + file renames) and the `base`-half (frame accessors `model.base`→`.frame` /
  `geometry.base(x)`→`geometry.frame(x)`, the **C stack → frame stack**: `base_stack_shape`→`frame_stack_shape`,
  `n_base`→`n_frame`). `T3Variations` kept. **Math "basis" preserved** — the standard/orthonormal basis of
  the variation cores, "basis vectors", the **Tucker basis** (factor matrices) — verified via an exhaustive
  cross-line audit (a real trap: placeholder protection missed line-split phrases, corrupting a few
  "base point"→"frame point" / "Tucker basis"→"Tucker frame"; all found + fixed). "base" kept for the plain
  manifold **point** (`base_point`, `base_masks`, "base point(s)"), the math "Tucker bases", and the
  named **`base-inner`** axis-ordering convention (could still become `frame-inner` for full consistency —
  a flagged nicety). Full suite green each half.
- **Still open:** the **cross-class method-name sweep** (verify parallel methods are truly the same op
  before unifying — do NOT force-merge subtly-different ones), the small `naming_review.md` items
  (singular alignment like `compute_mus`→`compute_mu`, drop `has_jax`, `probe_t3`→`t3_probe`-style
  auto-fixes), and the deferred cosmetic `Sequence`→`Union` hint relaxation.
- **Backend module reorg** (§4's family×op-kind matrix + per-op polymorphism triage): **DEFERRED** (Nick's
  call) — much more invasive; decide fold-in-or-drop later.

**2. Doc pass** (R3/R4): README (drop "WORK IN PROGRESS DO NOT USE" only at ship), fix the Sphinx build
(`conf.py` autoapi exclusions, committed `_build`, `modules.rst` still titled "TuckerTensorTrainTools"),
and **fold the design rationale from `docs/` into user-facing Sphinx docs**.

Plus the rest of the **release-hygiene roadmap** (R1–R6) below. (Independent thread: the toolbox reference
paper, `dev/paper_scope.md` — above.)

## The 1.0 roadmap (mid-level-toolkit scope) — summary
- **R1** packaging correctness (`readme = README.md`; create `CHANGELOG.md`; numpy range).
- **R2** public API surface (curate `__init__.py`) **+ the naming/organization review** (`dev/naming_review.md`).
  The `T3Basis→T3Frame` / `bv_→fv_` / `ubv_→ufv_` rename is **DONE**; what remains is the cross-class
  method-name sweep + the small auto-fixes (see "Next steps → 1"), and the deferred cosmetic
  `Sequence`→`Union` hint relaxation.
- **R3** README + quickstart (remove the "DO NOT USE" banner **only at the moment of shipping**).
- **R4** docs build (fix autoapi exclusions + `modules.rst` title; **fold design rationale from `docs/` into
  user-facing Sphinx docs**).
- **R5** test CI (pytest numpy-1.x/2.x matrix + **wire doctests in**); no auto-formatter near the curated style.
- **R6** cleanup — delete `OLD_*` / stray artifacts **only after confirming functionality is preserved**.
- **R7 — DONE** (the uniform tangent layer + optimizers/fitting on it + the U7 frontend). Still: document the
  absent weighted layer; do **not** ship research caveats as user guidance.
- **→ 1.1:** the Goal-1 `fit(...)` facade (auto geometry/optimizer/ranks/`x0` + rank-continuation).

## Don't-trip constraints (the maintainer's standing rules)
- **The uniform optimizer requires a minimal-rank base** — a non-minimal (unrealizable) nominal rank desyncs
  the retraction from the fixed masks and crashes mid-loop. `uniform_least_squares_problem` rejects it with a
  clear error; the frontend `optimizers.*` call `uniform_minimal(x0)` transparently.
- **The packedness-mirror convention** (U3.5): user-facing sampling ops infer & mirror packedness; the
  fitting split-seam is packed-only (the optimizer inner loop stays packed). Don't "normalize" it to a flag.
- **A uniform op needs more than dense-vs-ragged** — also exact output masks + garbage-robustness
  (`docs/testing_strategy.md`). Masks are host numpy (`np`, not `xnp`); supercores are `xnp`.
- Numerical test assertions use a **tolerance** (`np.allclose`/`np.isclose`); exact comparison only for
  structure (shapes, masks, types, counts) — never rely on floating-point bit-exactness across hardware.
- Never delete an `OLD_*` (or anything) until its functionality is **confirmed preserved**.
- "DO NOT USE" banner stays until the literal moment of shipping.
- **No automated tool rewrites the code style** (esp. the shape comments). No `manifold.py` rename.
- Research caveats are not user-facing. Notes are preserved/relocated, never lost.
