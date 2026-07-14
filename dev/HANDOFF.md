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

- **Regularization framework — S1 DONE (2026-07-14, uncommitted); design in `dev/regularization_design.md`.**
  Adding regularization to the fitting objective `min ½‖ω⊙(S(X)−y)‖² + ρ(X)`. **S1 (ragged identity,
  Newton-CG) is implemented + full suite green (627 tests):** new `backend/regularization.py`
  (`Regularizer` + `IdentityRegularizer`, re-exported from `bopt` and `optimizers`); `GeometryOps` gains
  `point_norm_sq` + `point_tangent` (manifold `v_X` via `tv_project_t3_onto_tangent_space`; corewise
  cores-as-tangent); `regularizer` field on `Problem`/`LocalModel` folded into
  `objective`/`gradient`/`gn_quadratic`/`hvp` + `Problem.objective`; `least_squares_problem(regularizer=)`;
  frontend `newton_cg`/`gradient_descent` `regularizer=` (+ uniform→`NotImplementedError` guard, S3).
  Verified: FD total gradient (1e-8/1e-11), manifold ridge shrinks ‖x‖, corewise weight-decay, jit-clean.
  **S1b DONE (uncommitted):** frontend `GaussNewtonModel` reg parity — `regularizer=` on all six model
  factories (uniform x → `NotImplementedError`), folded into `objective_value`/`gradient`/`gn_quadratic`/
  `gn_hessian`/`evaluate` (delegates to the backend `Regularizer`); jax pytree reg registration fixed
  (carries `regularizer` as aux, else dropped on jit round-trip). Frontend == backend LocalModel verified;
  `test_fitting.py::test_regularizer`; full suite green (628).
  **S2 DONE (uncommitted):** `mc_sgd`/`adam` `regularizer=` with the **(batch/n) scaling** —
  `_minibatch_step_problem` scales the reg by `min(batch,n)/n` for the per-step kernel (generic
  `_ScaledRegularizer` wrapper), full-batch stop keeps the full reg; verified scale=batch/n, mc_sgd+adam
  shrink, jit-clean.
  **S3 DONE (uncommitted):** uniform twin (optimizer path) — `point_norm_sq`/`point_tangent` on the uniform
  `GeometryOps` (manifold: `utv_project_ut3_onto_tangent_space` + orthogonalize→project→inner, mask-aware,
  no hand-mask logic; corewise: cores + masked `inner`); `uniform_least_squares_problem(regularizer=)`;
  frontend uniform guard dropped. **Verified uniform reg == ragged reg exactly (~1e-11)** + garbage-robust
  (1e6 padding, §7). Key finding: the uniform layer masks by *multiply*, so NaN padding isn't invariant
  (pre-existing) — use **large finite** garbage. **`point_tangent` refactored (2026-07-14, per Nick):**
  from `tv_project_t3_onto_tangent_space(frame, base)` (correct but roundabout — projects the base point
  onto its *own* tangent, contractions collapse to `P_last`) to the **direct construction** (last TT
  variation `= P_last`, else zero; already gauged — no projection). Verified element-wise-identical to the
  projection (ragged + uniform); all reg tests green.
  **S3b DONE (uncommitted):** uniform `UniformGaussNewtonModel` reg parity (roll-your-own) — `regularizer=`
  on all six model factories (`_reject_uniform_regularizer` guard removed); folded into
  `objective_value`/`gradient`/`gn_quadratic`/`gn_hessian`/`evaluate` via `_ubgeom` (backend uniform
  `GeometryOps` at rank) + `_reg_tangent`; pytree carries reg. `test_fitting.py::test_uniform_regularizer`
  (identities + uniform==ragged objective, both geometries). Full suite green (633).
  **S4 DONE (uncommitted):** `examples/fit_hilbert_regularized.py` (**Option A** — chosen after regime
  exploration; rationale block in `regularization_design.md` §10 S4). Fit a Hilbert tensor 16³ (spectral
  decay) at rank (3,3,3) from 400 noisy applies + 400 validation; unreg overfits slightly (recovery ~0.356,
  above the t3svd floor 0.0077), reg traces the U-curve (‖X‖ 4.12→3.13), λ from held-out validation → mean
  0.306 (~1.16×, val picks optimum 3/6, near-optimal rest). Showcases the `obj = misfit + reg` split. Pointer
  in `docs/fitting_and_optimization.md` §6. (Superseded an ill-posed exact-rank-2 draft — dramatic 2.9× but
  poor final fit ~0.6; Nick chose the good-fit/modest-gain framing.)
  **Misfit/reg display split DONE + COMMITTED (`d9700056`, unpushed):** `LocalModel.misfit`/`.regularization`
  props; `NewtonInfo` carries both → `stats['history']`/`['diagnostics']`; `verbose=` shows `obj = misfit +
  reg` (Format A); fixed a latent `(unwt …)` mislabel bug (compared vs total, now vs misfit). Tests + doctest.
  **Next: S5** (docs — a new § in `fitting_and_optimization.md` + CLAUDE.md shipped-surface). Also noted:
  masked-last-core `point_norm_sq` optimization.
  Design note + full slice plan: **`dev/regularization_design.md`**.
  Framework: a `Regularizer` is an objective term folded into the local GN model
  (`objective`/`gradient`/`gn_quadratic`/`hvp`) + `Problem.objective`, so it composes with every
  optimizer/kind/geometry/representation. Decisions locked: **D1** identity in the geometry's own tangent
  metric (`H_R = λ·project`; manifold = HS-ridge, corewise = weight-decay + makes the gauge-singular
  corewise Newton `H` PD); **D2** true objective regularization (adds `g_R` — forced by compose-with-all-
  optimizers); **D3** `X_ref=0`, `λ` scalar, `Regularizer` a protocol so Grasedyck–Kramer (inverse-
  unfolding-σ weighting) drops in later. Nick's key concern captured: **uniform garbage-safety** — the
  `ρ` reduction must sum only masked content; route everything through mask-aware primitives
  (`inner`/`project`/masked norm), never raw supercores; test with garbage/NaN-padding robustness + exact
  output masks (not just dense-vs-ragged). **Interface subtleties resolved + verified (2026-07-14):**
  (a) the attachment point is a single gauged tangent term `v_X` = last TT variation `= P_last` (already
  gauged; `dense(v_X)=X`; `‖v_X‖_coord=‖X‖_HS`; the naive sum-of-frame-core-norms is WRONG, 42291.8 vs
  42280.8); `H_R=λI` is the EXACT Riemannian Hessian (grad fully tangent → no curvature term). (b) the
  manifold retraction (`t3svd`) always emits **left-orthogonal** cores, so the line-search norm
  `value(x)=‖x.tt_cores[-1]‖²` is one core norm — no re-orthogonalization, no waste on rejected candidates.
  Noted follow-up (separate from reg, §11a): `already_left_orthogonal=True` after retract to skip the next
  step's re-orthogonalization. **Backend-user equality (§5a):** reg is fully backend-homed (`Regularizer` +
  `IdentityRegularizer` in `backend/regularization.py`; `point_norm_sq`/`point_tangent`/`value` on
  `GeometryOps`; `regularizer=` on `bopt.least_squares_problem`; folding in `LocalModel`), so a raw-`.data`
  user regularizes with the same one kwarg. The `value` left-orthogonal precondition is check-free in the
  backend the house way — and the checker tools already exist (`t3_orthogonality_residual` /
  `t3_left_orthogonalize`, public in `backend/t3_orthogonalization.__all__`), so no inequality and no new
  checker. Queued: an `examples/` demo (S4). **Next: implement slice S1 (ragged identity, Newton-CG).**
  Nothing committed yet (design note only).

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
  (`dev/regularization_design.md` §11b.)
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
