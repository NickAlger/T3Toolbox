# T3Toolbox — current handoff

_Updated 2026-07-01._

## Where we are

**The uniform-layer fix (the 1.0 centerpiece) is well advanced: the uniform *tangent backend*, the *two
geometries*, the *tangent + corewise probing* (`𝒥` / `𝒥ᵀ`), AND now the *derivative (jet) probing*
(`probe_derivatives`) are all built and exhaustively tested.** Branch `main`, direct commits; full suite
green throughout (**553 passed / 40 134 subtests**).

Done **this session (2026-07-01) — Increment 3b-6′, uniform `probe_derivatives`** (the jet/derivative twin
of 3b-6), both the plain `UniformTuckerTensorTrain` and `UT3Tangent` layers, per-element verified vs ragged
+ adjoint-identity + mask-strict + garbage-robust + jit-clean:
- **3b-6′a** — the 20 `d`-prefixed uniform JET contractions in `backend/contractions.py` (the binomial-jet
  `trs_*` twins with `d` prepended; `d` leads, then the order axis, then W/K/C). Oracle + order-0-anchor tested.
- **3b-6′b** — forward `𝒥`: `build_input_jets` unroll-trap fix (stack at axis 1, not a per-core loop over
  `d`) + the `reverse_tt` unroll fix in `compute_nu_jets`/`compute_tau_jets` (→ `uniform_ops.reverse_utt`) +
  4 map-style branches → `d`-prefixed + 2 scan-style flag-flips; new plain
  `UniformTuckerTensorTrain.{probe,apply,entries}_derivatives` (**did not exist**) + `UT3Tangent.*_derivatives`.
- **3b-6′c** — transpose `𝒥ᵀ` + corewise: the 5 map-style transpose branches → `d`-prefixed (order-slice is
  `[:, :s_size]`, order at supercore axis 1) + 2 scan-style flag-flips + 2 more `reverse_utt` fixes;
  `UT3Tangent.*_derivatives_transpose` + `UniformTuckerTensorTrain.*_corewise_derivatives_transpose`. **All
  hardcoded `get_backend(False)` and `ragged_ops.reverse_tt` in `probe_derivatives.py` are now polymorphic.**
- **3b-6′d** — the mask-strict + garbage-robust hardening (`TestUT3DerivativeHardening`).

Earlier this session's context also produced the two extra unroll-trap discoveries beyond `build_input_jets`
(the `reverse_tt` in the four reversers), which the plan had mislabeled as "already uniform-aware".

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
- **3b-5 — the two geometries** (`UniformManifoldGeometry`/`UniformCorewiseGeometry` + `UNIFORM_MANIFOLD`/
  `UNIFORM_COREWISE`): mirror the ragged `MANIFOLD`/`COREWISE` method-for-method, behind the per-element
  `.all()` safe-mode preconditions. `project_ambient` is `UniformTuckerTensorTrain`-only (dense → ragged);
  new backend `corewise_retract` for the `d`-leading additive add. `COREWISE.base` masks verified against
  §6.3 (`up=down=tucker_edge_mask`, `left=right=tt_edge_mask`, no slice).
- **3b-6 — tangent + corewise probing** (`𝒥` / `𝒥ᵀ`), per-element verified vs ragged + mask-strict +
  garbage-robust + jit-clean:
  - **3b-6a** — the 18 `d`-prefixed uniform `WKC` grouped-block contractions in `backend/contractions.py`,
    oracle-tested per `W`/`K`/`C` combo. **3b-6b** — forward `𝒥`: fixed `compute_detas`/`assemble_tangent_zs`
    + new `backend/ubv_sampling` (mask-once + pack/unpack) + `UT3Tangent.probe`/`apply`/`entries`.
  - **3b-6c** — transpose `𝒥ᵀ`: fixed the four transpose branches + made `_apply_transpose_adjoint` /
    `_onehot_vectors` / `_entry_xis` polymorphic (the last fixed a latent jit *unroll* in entries) +
    `UT3Tangent.{probe,apply,entries}_transpose` + the corewise
    `UniformTuckerTensorTrain.{apply,entries,probe}_corewise_transpose` (the §6.3 substitution). Verified by
    the adjoint identity `⟨r,𝒥V⟩=⟨𝒥ᵀr,V⟩`. **3b-6d** — the mask-strict + garbage-robust hardening
    (`tests/test_uniform_probing.py`).
  - **The probe-derivative (jet) version — slice 3b-6′ — is now DONE** (this session; see above).

## Next steps (finishing increment 3b)

The full slicing + design lives in **`dev/uniform_fix_plan.md`** (the living plan); status here.
1. **3b-7 — sweep + cleanup.** Tests/doctests final sweep; **delete `OLD_uniform*.py` + the `if False:`
   graveyard** once functionality is confirmed preserved (the standing caution). Relax the now-cosmetic
   `Sequence`-only type hints on the polymorphic `apply_tangent`/`entries_tangent` (+ transposes, + the
   `probe_derivatives.py` jet fns) to `Union` (runtime already polymorphic — not a blocker).
2. Then make the optimizers/fitting work on the uniform layer (speed is its whole point), and the
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
