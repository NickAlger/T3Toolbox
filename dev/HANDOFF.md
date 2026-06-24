# T3Toolbox — current handoff

_Updated 2026-06-23._

## Where we are
- **`geometry-refactor` is merged to `main`** (merge commit `bc8692f6`): the geometry
  abstraction, safe/unsafe mode, the four library optimizers, and derivative fitting.
  Suite green (310 passed / ~39k subtests); doctests swept clean for numpy 2.x.
- **Knowledge-architecture reorganization DONE** (this session) — docs sorted by audience ×
  lifetime: `dev/` + `dev/archive/` created (25 process notes archived); research migrated to a
  separate research repo (maintainer-local; incl. the two cordoned research branches); CLAUDE.md split
  (personal → `~/.claude/CLAUDE.md`) + "Where things live" routing rule added + slimmed; stale
  branches deleted (`fitting`, `probe-derivatives`, `geometry-refactor`, the 2 research branches).
- **Naming/organization review CONVERGED** (this session) — conventions locked in
  `dev/naming_review.md` (backend prefix grammar + `tv_`; `T3Basis→T3Frame` / `fv_`; cross-class
  consistency; the target module matrix). **The module reorg + per-op polymorphism triage fold into
  the uniform-layer fix** (ragged/uniform is *inferred* via `is_ndarray`, like numpy/jax).
- **Rank-continuation detour SHIPPED** (this session, direct to `main`; a time-sensitive collaborator
  request, orthogonal to the uniform work) — the Section 5.4.1 condition-number rank-continuation scheme:
  `compute_continuation_ranks` + `edge_condition_numbers` (`backend/ranks.py`) and the
  `TuckerTensorTrain.continuation_ranks` frontend method (params `tau`/`n_chunk`/`kappa_guard`/`max_grow`;
  `max_grow=1` = one edge at a time). Tests (`tests/backend/test_ranks.py` + a frontend wiring test),
  a doctest, the worked example `examples/fit_varied_rank_tensor_newton_cg.py` (adaptive vs uniform
  continuation), and the user doc `docs/rank_continuation.md`. Suite green. New public API to fold into
  the API-surface/doc passes (R2/R4).
- **In progress:** the **uniform-layer fix** (the 1.0 centerpiece) — Slices 1 & 2 ✅ done; Slice 3 (the
  tangent-layer rebuild) is next. See "Next steps" below.

## Knowledge architecture (decided this session)
- `docs/` = durable **design / reference / style** docs → to be distilled into user docs
  later (Track B / R4).
- `dev/` = **working notes** (this dir); `dev/archive/` = dated, superseded notes.
- a separate **research repo** (maintainer-local) = research detours / experiments / findings
  (the apply-derivative polynomial study, conditioning, scaling, the old probe-derivative
  code, the TTM paper, etc.).
- `~/.claude/CLAUDE.md` = the maintainer's **personal** prefs (commit signature, machine paths,
  work style); in-repo `CLAUDE.md` = **shared**, addressed to "any contributor's AI",
  with a lean current-state that points here.
- A **routing rule** + **handoff ritual** go into CLAUDE.md (Slice 2).

## Next steps
1. **Fix the uniform layer** — the 1.0 centerpiece. **Fully designed + triaged** in
   **`dev/uniform_fix_plan.md`** (polymorphism lenses, packed-vector I/O, `shape`→int-tuple, the
   strict-sampling + norm/inner-polymorphism decisions; reorg context: `dev/naming_review.md` §4), with 4
   agreed slices. **Progress:**
   - Slice 1 (fix the 3 broken imports in `uniform_basis_variations_format.py`) ✅ DONE.
   - **Slice 2 — `shape_mask` → shape int tuple ✅ DONE (2026-06-23).** Plain `ut3_` layer (frontend + 8
     `ut3_*` backend modules + tests) migrated to **4-arity-flat `.data = (tk_sc, tt_sc, shape, (tkm,
     ttm))`**; `UT3Masks` now holds only the two rank masks; `shape` is a value-hashed pytree-aux field.
     Notable wrinkle solved: the int-tuple `shape` is a `Sequence`, so `stacking.py`'s leaf-walker needed a
     **dynamic leaf template** (`ut3_operations.ut3_leaf_structure(d)`) + a manual first-leaf drill in
     `ut3_stack`/`ut3_unstack` + frontend `unstack`. `save`/`load` gained a third (shape) family. **Full
     suite green: 327 passed / 39 198 subtests; jax dispatch + doctests green.**
   - **Cleanup before the rebuild ✅ DONE (2026-06-23):** removed the ambiguous
     `UniformTuckerTensorTrain.{from_canonical, from_tensor_train, to_tensor_train}` + their backend twins
     (they round-tripped *ragged* CP/TT through `TuckerTensorTrain` — ragged-vs-uniform ambiguity; compose
     `t3_to_ut3`/`ut3_to_t3` explicitly instead). Suites green.
   - **Value-hashed mask holders ✅ DONE (2026-06-23):** all mask holders (`UT3Masks`, `UT3BasisMasks`,
     future `UT3VariationsMasks`) now hash/compare by mask **content** via the `common.ValueHashedMasks`
     mixin, so a rebuilt-but-identical holder is the *same* jit cache key. This **retires** the Slice-2
     "value-hashing is shape-only" caveat and fixes the optimization-loop recompile (re-orthogonalizing the
     frame each iteration was creating fresh identity-hashed holders → recompile every step). Empirically
     5→1 compiles; regression test `test_dispatch.py::test_mask_rebuild_does_not_recompile`. Docs updated.
   - **Slice 3 split → 3a + 3b** (decided 2026-06-23; details in `dev/uniform_fix_plan.md`):
     - **3a — frame/variations foundation (IN PROGRESS):** rebuild `UT3Basis` + `UT3Variations` directly in
       the target shape (int-tuple `shape` + the plain-layer pytree composition — masks in a value-hashed
       aux holder, not pytree children as before; supercores the only children), mirroring ragged
       method-for-method (~50 missing methods), + `ubv_*` backend + tests.
       - **Increment 1 ✅ DONE (2026-06-23):** `UT3Basis` + `UT3BasisMasks` rebuilt on the new design
         (flat holder, int-tuple shape, pytree); `ubv_masking.apply_basis_masks` migrated;
         `unstack`/`stack` stubbed (increment 2); `ut3_orthogonal_representations`' `UT3Basis` construction
         forward-ported (but that fn still has stale `use_jax` calls + builds the old `UT3Variations` →
         increment 2). Tests: `tests/test_uniform_basis_variations_format.py`. Suite 334 green.
       - **Increment 2a ✅ DONE (2026-06-24):** `UT3Variations` + `UT3VariationsMasks` rebuilt on the new
         design (value-hashed holder, int-tuple shape, pytree); `ubv_masking.apply_variations_masks`
         migrated **and a latent bug fixed** (it used the 5-axis TT reshape for the 4-axis tucker core →
         wrong output shape); `check_ubv_pair` fixed (its `base.up_mask` accessors broke at increment 1,
         and the `(a!=b).all()` logic was wrong → now `np.array_equal`); `ut3_orthogonal_representations`'
         `UT3Variations` construction forward-ported; `unstack`/`stack` stubbed. Tests + doctests green;
         full suite 342.
       - **Increment 2b (NEXT):** port `ut3_orthogonal_representations` **fully** (drop the stale `use_jax`
         calls → inferred dispatch; the function still doesn't run); migrate conversions
         (`ubv_conversions.ut3basis_to_t3basis` → int-tuple + a `t3basis_to_ut3basis` constructor) →
         **`to_t3`/`to_dense` round-trip anchor** (the equivalence-contract test deferred since increment 1);
         rebuild `unstack`/`stack` (dynamic-leaf-template); then the method buildout.
     - **3b — tangent/manifold:** `UT3Tangent` + `uniform_manifold` off the new types (drop `OLD_uniform` +
       ~600 lines of `if False:` dead code), un-stub geometry/`ubv_to_ut3`, derivative probing, tests.
     - **Naming DEFERRED:** 3a keeps `UT3Basis`/`ubv_` names; the global `T3Basis→T3Frame` + `bv_→fv_` +
       `ubv_→ufv_` rename (naming_review.md §2) is its own later mechanical, suite-gated pass.
   - Then Slice 4 (close the ragged-poly gaps: `_apply_transpose_adjoint` + `Sequence` signatures).
   - _(`uniform_manifold.py` still imports `OLD_uniform` and does not import — pre-existing, the 3b rebuild
     target; not exercised by the suite.)_
2. Then **release hygiene** (the R1–R7 roadmap below). **1.0 = honest mid-level toolkit; the `fit()`
   facade is deferred to 1.1.**

_(Knowledge-arch ✅ and naming review ✅ this session — see "Where we are".)_

## The 1.0 roadmap (mid-level-toolkit scope) — summary
- **R1** packaging correctness (`readme = README.md`; create `CHANGELOG.md`; numpy range).
- **R2** public API surface (curate `__init__.py`) **+ the naming/organization review**.
- **R3** README + quickstart (remove the "DO NOT USE" banner **only at the moment of shipping**).
- **R4** docs build (fix autoapi exclusions + `modules.rst` title; excise the `t3toolbox.jax`
  fiction; **fold design rationale from `docs/` into user-facing Sphinx docs**).
- **R5** test CI (pytest matrix + **wire doctests in**); no auto-formatter near the curated style.
- **R6** cleanup — delete `OLD_*` / stray artifacts **only after confirming the functionality
  is preserved elsewhere** (the maintainer's standing caution).
- **R7** **fix the uniform layer** (A: make broken code work · B: refactor to OO-frontend +
  functional-backend mirroring the ragged layer · C: make the optimizers/fitting work on it
  (its whole point is speed) · D: add derivative probing (ragged was built polymorphism-ready) ·
  E: tests/docstrings/doctests). **Document** the absent weighted layer; do **not** ship the
  research caveats as user guidance.
- **→ 1.1:** the Goal-1 `fit(...)` facade.

## Don't-trip constraints (the maintainer's standing rules)
- Never delete an `OLD_*` (or anything) until its functionality is **confirmed preserved**.
- "DO NOT USE" banner stays until the literal moment of shipping.
- **No automated tool rewrites the code style** (esp. the shape comments).
- No `manifold.py` rename.
- Research caveats are not user-facing. Notes are preserved/relocated, never lost.
