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

- **Uniform weighting layer — S0–S4 DONE (2026-07-15, committed, NOT pushed). NEXT: S5 (docs sweep) —
  then the thread closes. Plan: `dev/uniform_weighting_design.md`** (§8 = the settled decisions; §6 = the
  slices). The uniform mirror is **functionally complete**: `UT3Weights` + `UT3FrameWeights`, both with
  absorb / weighted_norm / weighted_inner / reciprocal / sqrt / concatenate / kronecker, plus
  `from_ut3svd` / `from_ut3weights` and the ragged↔uniform conversions, backend + frontend + pytrees.
  - **S4 was half built and half deliberately dropped (§8.7).** The wiring (classes, methods,
    `absorb_weights`, the `UT3Tangent` methods) landed inside S1/S3. The clause "+ dispatch inference
    (ragged vs uniform from the arg)" is **not built, by decision** (Nick, 2026-07-15): the **module IS the
    dispatch** — the user picks the layer they work in, and the conversion hooks are how they switch. So the
    weighted surface is parallel and module-scoped (`tucker_tensor_train.absorb_weights` vs
    `uniform_tucker_tensor_train.absorb_weights`), matching how the root exposes ragged/uniform side by side
    under distinct names. The optimizers' `isinstance` dispatch is not a counterexample (`newton_cg` is a
    single entry point with no module to dispatch through). Loose end noted in §8.7: the weighted surface is
    not re-exported at the package root at all, unlike every other frontend class.
  - **S3 (done) — the tangent metric.** `UT3FrameWeights` is **frame-like** (stack `C`), and the S0 model
    pays off exactly as designed: the `C`→`K+C` lift is **free** via the right-aligned `'...'`, which works
    only because `C` is innermost. Verified bit-identical to a `K`-tiled metric (0.0), and all-ones recovers
    `corewise_norm`. **Placement note (forced, not preference):** `utv_weighted_norm`/`_inner` live in
    `utv_operations`, not beside `ufv_absorb_weights` as ragged does — they need `utv_corewise_inner`, and
    `utv_operations` already imports `ufv_operations`, so the ragged placement would be a circular import.
  - **S1/S2 (done) — the uniform base-point layer ships**: `UT3Weights` (weight supercores + a `UT3Masks`
    holder; no `shape` field) + `absorb_weights` / `weighted_norm` / `weighted_inner` / `reciprocal` /
    `sqrt` / `concatenate` / `kronecker` / `from_ut3svd` + the ragged↔uniform conversions, backend
    (`ut3_operations` / `ut3_linalg` / `ut3_conversions`) + frontend + pytree. Matches the ragged oracle
    exactly (absorb 0.0, norm/inner ~1e-12). Tests: `tests/test_uniform_weighted.py`; user doc:
    `docs/weighting.md` §"On the uniform layer". **Three things a future reader should not re-derive:**
    (a) **`absorb` needs no masking** — it is a pointwise scale along edge axes, not a reduction, so it is
    garbage-transparent (the pre-review plan claimed the opposite); as a result `ut3_operations` does not
    import the masking layer at all, and the wall fell out rather than being imposed. (b) **`reciprocal`
    must guard the padding** (`1/0 = inf` → `0*inf = nan` poisons masked reductions; the GK metric *is* a
    reciprocal) but deliberately does **not** guard real-slot zeros. (c) **Uniform adds a mask-equality
    precondition** ragged gets free from shapes. **Mutation testing earned its keep twice**: it caught that
    the S1 mismatch test only perturbed the tucker mask (a `weights_consistent` ignoring the tt mask
    passed), and it confirmed S2's gappy-mask tests kill both plausible-but-wrong prefix masks. Worth
    repeating on any further mask-carrying op.
  - **Two side-fixes** (committed separately, both prompted by the wall): `common.prefix_mask` extracted
    (~8 duplicates across 4 modules), and `require_concrete_masks` moved to `common` — it was already
    documented as "infrastructure, unprefixed by design" but lived in `ut3_masking`, and **`ufv_masking`
    still does not use it, so the frame/variation masks are unguarded against traced masks** (pre-existing,
    not fixed). Plus a stale CI `--ignore` for the S4-deleted parked module.
  - **S0 (done) — fixed the ragged frame-weight stack model**, which the design review found wrong. It had to
    go first: the ragged layer is this build's equivalence oracle, so mirroring the bug would have had the
    oracle certify it. `T3FrameWeights` is **frame-like** (carries the frame stack `C`), but
    `fv_weights_consistent` demanded the weight match the *variations'* full `K+C` stack, so it **rejected**
    the canonical Grasedyck-Kramer weight (`from_t3weights(from_t3svd(x))` is `C`-stacked) the moment there
    was a `K`-stack of tangents at that frame. **No numbers were ever wrong** — the predicate is purely
    diagnostic (in no enforcement path), and a `C`-only weight already computed bit-identically to a
    `K`-tiled one (verified, max diff `0.0`; the leading `'...'` lifts `C` over `K` because `C` is innermost).
    Root cause: at the metric-on-variations design change the implementation moved the *stack* to the
    variations when only the *absorption target* should have moved — where an object **batches** and what it
    **acts on** are different questions. Fix as shipped: the **trailing-stack rule** in
    `fv_weights_consistent` (`check_fv_pair`'s existing frame<->variations rule, with the weight playing the
    frame's role; stays blind to the frame, non-breaking) + the new **`check_fw_pair(frame, weights)`** guard
    (`weights.stack_shape == frame.stack_shape` exactly, + ranks vs the frame's variation holes) wired into
    `T3Tangent.weighted_norm`/`weighted_inner`/`absorb_weights` + docs (`weighting.md` §Batching, the class
    docstring, `batching_and_stacking.md` — which also had a stale "weighted is parked" line) + tests (the
    helper built only `K+C` weights, which is why nothing caught it; `test_frame_like_stack` +
    `test_tangent_rejects_non_frame_stack` added, and the former was **verified to fail against the pre-S0
    predicate**). `dev/weighted_layer_design.md` §4/§6 marked superseded (they still described the abandoned
    absorb-into-the-frame design + its doubled-rank `tv_to_t3` norm path). Gates: full suite 654 passed /
    40,304 subtests; docs `-W` clean.
  - **Also settled at review (§8)**: mask-guarded `reciprocal`/`sqrt` (`1/0 = inf` -> `0*inf = nan` poisons
    the GK path — the plan had missed it), `kronecker` ships unpaired (there is no `ut3_mult`), a
    masking/weighting **conceptual wall** (weighting never calls masking; break shared code into a neutral
    subfunction) — nearly free, since `absorb` is **garbage-transparent** (pointwise, not a reduction: the
    pre-review plan's "absorb must mask on entry" was wrong), and `UT3Weights` carries no `shape` field.
  - **S5 next (the last slice)** — a docs sweep. `docs/weighting.md` already carries the uniform mirror;
    what remains is promoting the durable rationale into the contributor guide and archiving the design
    note (the reg-thread pattern), including: **mutation testing caught a hole in my own tests twice**
    (S1's tucker-only mask perturbation; S3's mismatch test that widened supercores and so was rejected on
    SHAPE, never reaching the mask check) — the lesson is that a mask check must be isolated, not merely
    exercised, and it belongs in `docs/contributor/testing_strategy.md` beside the phantom-rank story. The
    original mirror scope, for reference:
  `UT3Weights` / `UT3FrameWeights` (weight supercores + boolean masks — reuse `UT3Masks` /
  `UT3VariationsMasks`; a weight's edges are the object's edges) + masked `absorb` / `weighted_norm`/`inner`
  / `concatenate` / `kronecker` / conversions / `from_t3svd` / `from_t3weights`. **Lever:** the ragged layer
  is the equivalence oracle (`to_ragged(op_uniform(to_uniform(x))) == op_ragged(x)`). **Hard part:** the
  masks — mask on entry (garbage don't-care), and `concat`/`kron` go **gappy** (masks concatenate/Kronecker
  per `docs/uniform_masks_vs_ranks.md`); `absorb`/`norm`/`inner` keep the mask. Slices S1–S5 + a watch-list
  (gappy masks, mask-on-entry, host-numpy masks, variable-rank-per-stack tests) in the plan. Ragged layer
  fully shipped (`T3Weights`/`T3FrameWeights` + all ops + `from_t3svd`/`from_t3weights`, backend+frontend;
  `docs/weighting.md`, `docs/frame_variations.md`).

- **Weighted tensor-network layer — SHIPPED (S1–S5, 2026-07-15). Plan/record: `dev/weighted_layer_design.md`;
  user doc: `docs/weighting.md`.** Diagonal edge-weights, lightly (no heavy wrapper). **`T3Weights`**
  (tucker[d], tt[d+1] = `t3svd` sval format) weights a **`TuckerTensorTrain` as a tensor** — `absorb_weights`
  / `weighted_norm`/`weighted_inner` / `concatenate`↔`+` / `kronecker`↔`⊙` (Kronecker-of-weights verified
  1.2e-15) / `from_t3svd`. **`T3FrameWeights`** (up/down/left/right, each len d) is a **metric on a tangent's
  coordinates** (Grasedyck–Kramer preconditioner) — `T3Tangent.weighted_norm/weighted_inner`, absorbed into
  the **variation** cores (frame orthonormal). Commits `358860bb`/`059124f1`/`99dcb8b5`/`80354977` + docs.
  **Key design change during build (see design-note header + §6):** the tangent weight is the
  **metric-on-variations** (`d` natural edges; the frame's `d+1`-th left/right cores are base-point padding),
  NOT the tensor-weighting we first sketched — cleaner, `O(ranks)`, and what GK needs. Old parked `wt3_*`
  layer retired (S4). **Deferred (reachable):** weighted `+`/`⊙`/scale ops + optional thin container; the
  uniform mirror (weights carry boolean masks); the GK `SingularValueRegularizer`.
  **Nick reviews the whole thing 2026-07-15** (built autonomously per his request). Follow-ups after review:
  promote the durable rationale into the rendered contributor guide + archive the design note (reg-thread pattern).

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
