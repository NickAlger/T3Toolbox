# T3Toolbox — current handoff

_Updated 2026-07-12 (evening). Prior history: `dev/archive/handoff_2026-07-12_1.0_complete.md`
(the 1.0 completion: R1–R7, the R4 doc pass, the docs user/dev split S1–S5, the cordon/ETT/
literature morning)._

## Where we are — 2026.0.0 SHIPPED ✅

**2026.0.0 is live on PyPI (2026-07-13) — `pip install t3toolbox`** (+ the `[jax]` extra).
REL-1 → REL-4 all done; the full release history is archived at
`dev/archive/release_plan_2026-07-13.md`. The ship (REL-4): CHANGELOG `[Unreleased]` →
`[2026.0.0]`; install recipe flipped to `pip install t3toolbox`; gates green (593 tests /
40,215 subtests, docs `-W`, doctests, wheel + `twine check` + fresh-venv numpy-only smoke); commit
`21f7b6fb`, tag `v2026.0.0` → the trusted-publishing workflow (approved on the `pypi` environment)
→ published. **Verified against real PyPI**: numpy-only and `[jax]` fresh-venv installs, quickstart
smoke, and the full `getting_started.rst` doctest (64/64) against the installed package.

Loose ends: the **GitHub Release** for `v2026.0.0` — Nick to create via the web UI (notes = the
CHANGELOG `[2026.0.0]` section); **Zenodo DOI** still deferred (Nick, later); the dead `gh-pages`
branch can be deleted (optional).

## Active threads

- **Weighted layer (edge weights) — COMPLETE & SHIPPED, ragged + uniform (2026-07-15). Thread closed;
  committed, NOT pushed.** Diagonal weights on the internal edges, as a lightweight data format +
  `absorb` into cores. `T3Weights`/`UT3Weights` weight a **tensor**; `T3FrameWeights`/`UT3FrameWeights`
  are a **metric on a tangent's coordinates** (Grasedyck-Kramer). All four carry `absorb` /
  `weighted_norm`/`weighted_inner` / `reciprocal`/`sqrt` / `concatenate`/`kronecker`, plus
  `from_*svd` / `from_*weights` and ragged<->uniform conversions; frontend free functions are
  family-prefixed (`t3_`/`ut3_`/`fv_`/`ufv_absorb_weights`) and the whole surface is exported at the
  package root. Tests: `tests/test_weighted.py`, `tests/test_uniform_weighted.py`.
  - **Durable knowledge is now in the rendered docs** (build notes archived): user usage ->
    `docs/weighting.md`; design records -> **`docs/contributor/weighted_internals.md`** (the two-classes
    reasoning, the metric-on-variations change, the frame-like stack model + the two-level check, the
    uniform mirror's three traps, placement notes, and what's deferred); the testing lesson ->
    `docs/contributor/testing_strategy.md` ("Exercising a mask check is not testing it"); the naming rule
    -> `docs/naming_conventions.md`; the `ut3_norm`/`ut3_inner` gap -> `docs/contributor/deferred_and_rejected.md`.
    Build records: `dev/archive/weighted_layer_design.md` (ragged), `dev/archive/uniform_weighting_design.md`
    (uniform; §8 = its decision log). **The rendered docs are authoritative where the archives disagree.**
  - **Side-fixes that landed with it** (all committed): `common.prefix_mask` extracted (~8 duplicates
    across 4 modules); `require_concrete_masks` moved to `common` (it is the mask-representation contract,
    not a `ut3_` family member); a stale CI `--ignore` for the S4-deleted parked module removed; and
    `docs/batching_and_stacking.md`'s "weighted is parked" line corrected.
  - **Known gap, surfaced but NOT fixed:** `ufv_masking` still does not call `require_concrete_masks`, so
    the frame/variation masks are unguarded against being passed as traced jit args (they fail with jax's
    cryptic `ConcretizationTypeError` instead of the actionable message). Pre-existing; unrelated to
    weighting; cheap to fix.
  - **Deferred (reachable from the primitives):** weighted `+`/`-`/scale/`⊙` as operations + an optional
    thin container; the Grasedyck-Kramer `SingularValueRegularizer` — **both layers now have every
    primitive it needs**, so it is the natural next consumer.

- **Regularization framework — COMPLETE & SHIPPED (S1–S5, 2026-07-14).** Identity (Tikhonov)
  regularization on the fitting objective `min ½‖ω⊙(S(X)−y)‖² + ρ(X)`, composing with every optimizer /
  kind / geometry / representation: `regularizer=IdentityRegularizer(λ)` on any optimizer, ragged + uniform,
  backend-homed (`backend/regularization.py`). Plus the **`obj = misfit + reg` display split** (`verbose=` +
  `stats['history']`/`['diagnostics']`; also fixed a latent `(unwt …)` mislabel bug). Commits: S1–S4 + the
  split **pushed** (`d9700056`, `54d752d2`); **S5 docs + this cleanup uncommitted → next: commit.**
  - **Durable knowledge now in the rendered docs** (design note archived): user usage →
    `fitting_and_optimization.md` §4.9 (+ the §5 "rank is the primary regularizer" note); contributor design
    decisions + the `v_X`/`value` derivations + uniform mask-safety + stochastic scaling + the deferred items
    (Grasedyck–Kramer seam, base-point-as-tangent public op, `already_left_orthogonal` amortization) →
    `docs/contributor/fitting_internals.md` ("Regularization" + "What's deferred"). Full build record:
    `dev/archive/regularization_design.md`.
  - Worked example: `examples/fit_hilbert_regularized.py` (Option A — Hilbert denoising, fit ~0.30, λ by
    validation). Small open optimization: masked-last-core `point_norm_sq` (uniform, cheaper + ragged-consistent).

- **`use_jit=True` auto-convert (silent-drop bug) — DONE + PUSHED to `main` (2026-07-14, `3ce68fea`).**
  Fixes: `use_jit=True` with numpy inputs used to silently run eager (the flag looked accepted but did
  nothing — meaningless "jit" benchmarks). Root cause was the `_maybe_jit` / `xwhile` guard requiring all
  inputs to already be jax. **Key technical finding:** it's *not* a jax limitation — a jit tracer *is* a
  `jnp.ndarray`, so our type-inference dispatch routes to `jnp` during tracing; forcing jit on numpy
  inputs runs correctly and (with x64) matches eager bit-for-bit. The guard's real job was preventing a
  silent **float32** downgrade (jax's default), not a crash. Nick's call: **auto-convert** — requesting
  jit is opting into jax-world precision. Fix (`_prepare_jit_inputs` in `backend/optimizers.py`): when
  `use_jit=True`, `jnp.asarray` `x0` + `problem.sample`/`data` (masks/weight left alone), so both jit
  mechanisms engage; returns a **jax-backed** result (float32 unless x64); **raises** if jax absent
  (the one un-honorable case). New `common.tree_to_jax`. Verified end-to-end ragged + uniform (masks stay
  numpy) + no-jax raise. Tests: backend `test_use_jit_requires_jax` + jax-backed assertions in
  `test_newton_cg_recovers_to_high_accuracy`; frontend `test_newton_cg_use_jit_returns_jax`. Docs:
  `fitting_and_optimization.md` §4.5, frontend module docstring, CLAUDE.md shipped-surface. Two refinements
  (Nick, same session): (a) **`use_jit` promoted to an explicit frontend kwarg** on `newton_cg`/`mc_sgd`/
  `adam` (was implicit via `**kwargs` — justified singling-out: it's the only kwarg that changes the
  return type/precision); (b) a **3-part precision doctest** in `optimizers.newton_cg` — numpy float64
  (~1e-10) vs jit float32 (~1e-7) vs jit + `jax_enable_x64` (float64 restored, ~1e-10). Verifiable via the
  dtype contrast + straddle-1e-8 booleans (raw floats aren't bit-reproducible). **x64 leak avoided**:
  `jax.experimental.enable_x64` is gone in jax 0.10, so the doctest uses the global `jax.config.update`
  but captures dtype/err as plain Python values *while x64 is on*, then restores `x64=False` BEFORE
  asserting — a green run guarantees no leak into the single-process `--doctest-modules` sweep (verified:
  full sweep 169 passed; also green on the jax 0.4.30 compat-floor env). **Next: commit.**

- **Newton-CG warm-start reference overrides — DONE + PUSHED to `main` (2026-07-14, `649edb62`).**
  `newton_cg` now takes three optional kwargs so a warm-start continuation loop isn't hurt by a
  misleadingly-small initial `‖g0‖`: `g0norm_newton` / `g0norm_cg` override the reference norm the
  Newton stop (`‖g‖ ≤ gtol_rel·‖g0‖`) and the CG forcing term (`η = min(0.5, (‖g‖/‖g0‖)**power)`) are
  relative to (chained fallback: `g0norm_newton` also feeds CG unless `g0norm_cg` is given; `g0norm_cg`
  alone touches only CG; neither → the computed initial norm as before), and `cg_forcing_power`
  (default `0.5`) trades CG iters per Newton step for fewer Newton steps (raise it when the manifold
  retraction is expensive vs a Hessian-apply). **Backend-only change** (`backend/optimizers.py`); the
  frontend forwards via `**kwargs`, uniform inherits it for free. `NewtonInfo.g0norm` now reports the
  effective Newton reference. Tests: `test_optimizers.py::test_newton_cg_g0norm_and_forcing_overrides`
  (four fallback cases + power direction, reconstructing each ref from the reported η) +
  `test_optimizers_frontend.py::test_newton_cg_g0norm_kwargs_forward`. Docs: `fitting_and_optimization.md`
  §5, cross-ref in `rank_continuation.md`, CLAUDE.md shipped-surface. Suites green (57 opt/display/dispatch
  + doctests). **Next: commit** (message per §Workflow); nothing else open on this thread.

- **Newton-CG diagnostic display — DONE + MERGED to `main` (2026-07-13).**
  `optimizers.newton_cg(..., verbose=True)` prints a per-iteration block (objective/gradient, CG stats,
  line search, ρ, wall time) + a per-`(mode, order)` relative-error table (`‖r_ij‖/‖y_ij‖`), with an
  optional `val_sample`/`val_data` validation column; records also returned in `stats['diagnostics']`.
  **Backend-owned** (anti-drift): a raw-`.data` user gets the identical display via
  `backend.optimizer_display.make_newton_display` + `newton_cg(callback=...)`. Works on ragged **and**
  uniform (the `block_sumsq` reduction is dual-path; validation auto-packed). Table layout follows the
  kind's axes (plain probe: mode cols; probe_derivatives: mode rows × order cols, train|val cells).
  Example `examples/fit_probe_display.py` shows both layouts. Full suite green (619 tests). Design record
  + slice list: `dev/newton_display_plan.md`; merged fast-forward (commits `53aab004`..`545653ce`).
  Thread closed.

- **Per-mode residual weighting — DONE + MERGED to `main` (2026-07-13).**
  The fitting layer's residual weight `ω` generalized from a per-order vector to an `ω[mode, order]`
  matrix; **per-mode weighting** added to the probe models (probe is the only kind with a per-mode
  axis — apply/entries stay order-only). `probe_model(weight=(d,))`,
  `probe_derivatives_model(weight=(d,order+1))`, topt threads it, uniform mirror is compile-once
  (nested-tuple aux). New example `examples/fit_per_mode_weight_probes.py`; docs §4.6 rewritten.
  Design record + the full slice list: `dev/per_mode_weighting_plan.md`. Full suite green, docs `-W`
  clean; merged fast-forward (commits `02972a86`..`1dcd84ce`). Thread closed.

- **The toolbox reference paper** (independent): scope + curation in `dev/paper_scope.md`.
  Next: walk the groups starting at Group 6 (`docs/symmetric_probe_derivatives.tex` is nearly a
  drop-in chapter). Paper-grade material queued there from the archive sweep: the two-spaces
  geometry picture; the apply/entries sweep-level scatter derivations.

## Backlog (not scheduled)

- **Base-point-as-tangent as a public library op** (Nick, 2026-07-14) — representing a base point `X` as a
  gauged tangent `v_X` within its own tangent space is broadly useful; expose it as a first-class op
  (frontend `T3Tangent`/`UT3Tangent` factory + backend helper) with the direct construction (last TT
  variation `= P_last`, else zero; already gauged). Reg's `_manifold_point_tangent` /
  `uniform_manifold_ops`'s closure is the current internal impl to promote/share.
  (Now also in `docs/contributor/fitting_internals.md` "What's deferred"; full context:
  `dev/archive/regularization_design.md` §11b.)
- **Default-path doctest pass** for undocumented public functions (Nick wants this).
- **`core_shapes` (property, strips stack) vs `get_core_shapes` (static, includes stack)**
  inconsistency — verified live 2026-07-12; a code decision for Nick.
- **Zenodo DOI** — Nick, at a later date.
- Delete the dead `gh-pages` branch (optional; Pages deploys from artifacts now).
- Per-test seeding → `pytest -n auto`; trimming `test_dispatch` jit time (deferred niceties).

## Post-1.0 (1.1) threads

- The Goal-1 **`fit(...)` facade** (auto geometry/optimizer/ranks/`x0` + rank continuation —
  "standard user, no fiddling").
- **Weighted-layer revival/redesign** (currently parked + cordoned with warnings).

## Standing constraints

The durable rules live where they belong: project-wide conventions and gotchas in **CLAUDE.md**;
contributor-facing conventions and decision records in the rendered **Contributor guide**
(`docs/contributor/` — naming rules, refactoring methodology, testing strategy, the
deferred/rejected ledger). Two operational ones worth repeating here: the docs build must stay
at **zero warnings** (`sphinx -W` in CI), and doctest outputs are **run-and-pasted, never
hand-written**.
