# T3Toolbox — current handoff

_Updated 2026-06-30._

## Where we are

**The uniform-layer fix (the 1.0 centerpiece) is well advanced: the entire uniform *tangent backend* is
built and exhaustively tested.** Branch `main`, direct commits; full suite green throughout
(**467 passed / 39 393 subtests**).

Done in earlier sessions:
- Slices 1, 2 (the `shape_mask`→int-tuple migration), value-hashed mask holders, and the cleanup before the
  rebuild — see git history / `dev/uniform_fix_plan.md`.
- **Increment 2c (Slice 3a) — the `UT3Basis`/`UT3Variations` foundation** — complete and verified against
  the ragged layer (converters, base-point conversions, stack/unstack, vector space, reverse/orthogonalize,
  save/load, per-element checkers). 2c-A…2c-G.

Done **this session (2026-06-30) — Increment 3b, the uniform tangent + manifold layer**, all per-element
verified against the ragged `manifold`/`tangent_operations`, mask-strict, garbage-robust, multi-axis,
varying-`C`, and jit-clean:
- **3b-0** — `check_ubv_pair` accepts the tangent (`K`) stack (stack-free structure compare + `C`-suffix +
  broadcast-over-`K` mask check).
- **3b-1a/b** — `UT3Tangent` (`t3toolbox/uniform_manifold.py`, rebuilt; the old graveyard preserved as
  `OLD_uniform_manifold.py`): structure + `K`/`C` inference, vector-space ops with the numerical same-frame
  guard, corewise inner/norm/allclose/normalized, the delegating checkers, per-element
  `tangent_space_dimension`, constructors, `reverse`, pytree (basis-as-leaf); **3b-1b** the tangent
  stack/unstack conversions + `sum_tangents` (new `backend/ubv_tangent_operations.py`).
- **3b-2a/b — the doubled-rank keystone** — `tangent_to_ut3` (verified block-for-block against the paper's
  eqs 50–53, Appendix A.3.1; honest boundary masks) + `UT3Tangent.to_ut3`/`to_dense` + `retract`.
- **3b-3 — gauge** — `orthogonal`/`oblique_gauge_projection` (oblique's TT step is an `xscan`, not an
  unrolled loop) + `gauge_residual` + `UT3Tangent.gauge_residual`/`is_gauged`.
- **3b-4a/b** — cross-layer `to_t3tangent`/`from_t3tangent` converters + `project_ut3_onto_tangent_space`
  (the TT zippers `tt_zipper_*` made `is_uniform`-polymorphic in `tangent_operations.py`).
- **3b-4c — the test-hardening pass** — exact (non-circular) output-mask assertions + garbage-padded-input
  robustness + the `_CONFIGS` stack matrix (forced-pad, multi-axis, `K`, varying-`C`). Closed the
  **clean-padding blind spot** (dense tests are blind to too-permissive/phantom masks). Found **no impl
  bug** (two issues surfaced, both in the tests); cross-checked by an independent adversarial cold-read
  audit agent (also no bug). New durable rationale: **`docs/testing_strategy.md`**.

## Next steps (finishing increment 3b)

The full slicing + design lives in **`dev/uniform_fix_plan.md`** (the living plan); status here.
1. **3b-5 — the geometries.** `UniformManifoldGeometry` / `UniformCorewiseGeometry`: give the backend ops
   (project / inner / norm / retract / project_ambient / transport / base / randn) their frontend home,
   behind the per-element safe-mode preconditions (orthogonal frame, gauged — `.all()`) and the
   jit-recompile constraint (`docs/uniform_backend_jit_recipe.md`). `project_ut3_onto_tangent_space` and
   `retract` get exposed here.
2. **3b-6 — probing + the `WKC` contractions.** Build the `d`-prefixed uniform `WKC` grouped-block
   contractions in `backend/contractions.py` and fix the map-style uniform-tangent probing branches
   (`compute_detas`/`assemble_*`/`compute_dxi_tildes`/`compute_deta_tildes` — NOT `compute_dxis`); wire
   `UT3Tangent.probe`/`apply`/`entries` (+ derivatives). Inventory in the plan's "Validation hardening".
3. **3b-7 — sweep + cleanup.** Tests/doctests final sweep; **delete `OLD_uniform*.py` + the `if False:`
   graveyard** once functionality is confirmed preserved (the standing caution).
4. Then make the optimizers/fitting work on the uniform layer (speed is its whole point), and the
   release-hygiene roadmap below.

## The 1.0 roadmap (mid-level-toolkit scope) — summary
- **R1** packaging correctness (`readme = README.md`; create `CHANGELOG.md`; numpy range).
- **R2** public API surface (curate `__init__.py`) **+ the naming/organization review** (`dev/naming_review.md`:
  backend prefix grammar; `T3Basis→T3Frame` / `bv_→fv_` / `ubv_→ufv_` — its own mechanical, suite-gated pass).
- **R3** README + quickstart (remove the "DO NOT USE" banner **only at the moment of shipping**).
- **R4** docs build (fix autoapi exclusions + `modules.rst` title; **fold design rationale from `docs/` into
  user-facing Sphinx docs**).
- **R5** test CI (pytest numpy-1.x/2.x matrix + **wire doctests in**); no auto-formatter near the curated style.
- **R6** cleanup — delete `OLD_*` / stray artifacts **only after confirming functionality is preserved**.
- **R7** finish the uniform layer (3b-5/6/7 above) + optimizers/fitting on it. Document the absent weighted
  layer; do **not** ship research caveats as user guidance.
- **→ 1.1:** the Goal-1 `fit(...)` facade.

## Don't-trip constraints (the maintainer's standing rules)
- Never delete an `OLD_*` (or anything) until its functionality is **confirmed preserved**.
- "DO NOT USE" banner stays until the literal moment of shipping.
- **No automated tool rewrites the code style** (esp. the shape comments).
- No `manifold.py` rename.
- Research caveats are not user-facing. Notes are preserved/relocated, never lost.
- **A uniform op needs more than dense-vs-ragged** — also exact masks + garbage-robustness
  (`docs/testing_strategy.md`). Masks are host numpy (`np`, not `xnp`); supercores are `xnp`.
