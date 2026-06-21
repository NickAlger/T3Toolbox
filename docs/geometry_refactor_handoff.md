# Geometry refactor — handoff (break point, 2026-06-19)

*Resume-here note for the `geometry-refactor` branch. Everything below is committed and pushed; the
working tree is clean. Read this first, then the linked docs for detail.*

> **SUPERSEDED (2026-06-21): G3 + derivative fitting (D1–D4) have since landed and the branch is being
> merged to `main`.** This note is kept as the 2026-06-19 break-point record (the "How to verify" /
> "State caveats" sections are 2026-06-19 snapshots — e.g. the suite has since moved to the `t3toolbox`
> env, 310 passed). Only "What's next" below is updated. Current source of truth: `CLAUDE.md` "Current
> state" + `docs/fitting_and_optimization.md`.

## Where we are

Branch `geometry-refactor` (not merged to `main`). The refactor introduces a Manopt-style **geometry
abstraction** (`ManifoldGeometry` / `CorewiseGeometry`), one geometry-generic `GaussNewtonModel`, and
`T3Tangent` as the universal tangent — so optimizers can be written **once, not per geometry**. Plan:
[`geometry_refactor_plan.md`](geometry_refactor_plan.md).

Done so far:

- **G1 — geometries + thin `T3Tangent`** ✅ (commit `3b64c997`). `ManifoldGeometry` / `CorewiseGeometry`
  singletons (`MANIFOLD` / `COREWISE` in `manifold.py`): `base`, `project`, `retract`, `project_ambient`,
  `transport`, and the HS `inner` / `norm`.
- **G2 — generic `GaussNewtonModel`** ✅ (commit `de8a51d1`). One model in `fitting.py` over `apply` /
  `entries` / `probe`, with `gradient` / `gn_hessian` / `jacobian` / `gn_quadratic` / `evaluate` and the
  `apply_model` / `entries_model` / `probe_model` factories.
- **Safe / unsafe mode (S1–S6)** ✅ — **the big arc this session.** See below.
- **MC-SGD prototype** ✅ — the apply-derivative example now fits via Manifold Cauchy SGD. See below.

## Done this session

### 1. Safe / unsafe mode (S1–S6) — the jit/recompile/OO predicament, dissolved

**Root insight:** the "same tangent space" guard on `T3Tangent` (`self.basis is other.basis`) was a
**numerical** property faked as **structural** via object identity. That single fake forced `T3Tangent`'s
basis to be jax **aux_data** (→ jit recompiled on every base change) and false-failed on a jit round-trip.

**The fix + the house-rule change.** Numericalize the guard (`safety.frames_equal`, an `is`-fast-path then
value compare). This let the basis become a pytree **leaf** and the whole `GaussNewtonModel` a registered
pytree → **`jit(lambda model, p: model.gn_hessian(p))` compiles once across all bases** (`traces=1`); you
jit the frontend matvec directly. Generalized to a **safe/unsafe-mode** discipline: numerical
**preconditions** are enforced in safe mode (an ambient `contextvars` tolerance, the one sanctioned
global), skipped under `safety.unsafe()` / any jax trace; correctness-neutral (the `assert`/`-O`
precedent). Two tolerances (numpy / jax, float32). inner/norm moved onto the geometries (HS vs Euclidean).
ORTH preconditions wired into the manifold projections/retraction (cached residuals). Minimal rank settled
empirically as **not** a precondition.

The two CLAUDE.md house rules were amended accordingly ("numerical → enforce in safe mode"; "one ambient
`safety` global"), and stale `basis-as-aux` references were fixed repo-wide.

- New module: `t3toolbox/safety.py` (+ `tests/test_safety.py`).
- Plan / catalog: [`safe_unsafe_mode_plan.md`](safe_unsafe_mode_plan.md),
  [`numerical_contract_catalog.md`](numerical_contract_catalog.md).
- Verified: **full suite 295 passed**, dispatch + doctests clean, both Hilbert examples reproduce.

### 2. MC-SGD prototype (Manifold Cauchy SGD)

Converted `examples/fit_hilbert_from_apply_derivatives.py` from slow full-batch Newton-CG to **MC-SGD**
(T4S §5.3.2): the tuning-free Cauchy step `α = ‖g‖²/‖Jg‖²` (= `model.gradient` + `model.gn_quadratic`),
minibatched over base points. **~8× faster to the noise floor** than Newton-CG. Finding: the optimizer is
robust, but the **batch-size + epoch-based-stopping heuristics are finicky at the toy `N_X=10` scale**
(likely a small-scale artifact; unproven at scale). Full write-up:
[`mcsgd_apply_derivatives.md`](mcsgd_apply_derivatives.md).

## What's next (updated 2026-06-21)

1. **G3 — `optimizers.py` ✅ DONE.** Four geometry-agnostic optimizers (`gradient_descent`, `mc_sgd`,
   `adam`, `newton_cg`), backend-first with a thin frontend adapter. MC-SGD's stopping window was made
   **absolute-iteration-based** (the batch=3 fix); the derivative `GaussNewtonModel`s
   (`{apply,entries,probe}_derivatives_model`) were built and apply/entries/probe derivative fitting wired
   end-to-end (D1–D4). `newton_cg` tolerates the singular corewise `H` (inexact/regularized inner CG); no
   `hessian_is_degenerate` hint was needed. See `docs/optimizers_plan.md`,
   `docs/fitting_and_optimization.md`, `docs/derivative_fitting_plan.md`.
2. **Merge `geometry-refactor` → `main` (2026-06-21).** Suite green (310 passed, 39k subtests); doctests
   swept clean for numpy 2.x. **Deferred past the merge to the 1.0 pass (NOT merge gates):** the Goal-1
   `fit(...)` facade, the broader *example pass* (`optimizers_plan.md` §10, PROPOSED), and the
   `entries_apply_probe.md` / `symmetric_probe_derivatives.tex` doc refreshes.

## How to verify (when resuming)

- Full suite (~110 s): `cd /home/nick/repos/T3Toolbox && PYTHONPATH=$PWD python -m pytest tests/ -q`
  (env: conda `tttt`). Last run: **295 passed.**
- Examples (`PYTHONPATH=$PWD python examples/<name>.py`): `fit_hilbert_tensor_newton_cg.py` (apply,
  Newton-CG), `fit_hilbert_from_apply_derivatives.py` (apply-derivatives, MC-SGD, ~35 s). Both reproduce.
- `examples/jax_jit_example.py` is **pre-existing stale** (imports a `t3toolbox.jax` subpackage that does
  not exist) — unrelated to this work, ignore.

## Pointers (read order when resuming G3)

1. [`geometry_refactor_plan.md`](geometry_refactor_plan.md) — the plan; §7 slices (G1/G2 done, G3 next),
   §8 open questions (incl. the MC-SGD note + Nick's optimizer-ideas placeholder).
2. [`mcsgd_apply_derivatives.md`](mcsgd_apply_derivatives.md) — the MC-SGD method, findings, open questions.
3. [`safe_unsafe_mode_plan.md`](safe_unsafe_mode_plan.md) + [`numerical_contract_catalog.md`](numerical_contract_catalog.md)
   — the safe-mode mechanism the optimizers rely on (no-recompile jit, the guards).
4. `CLAUDE.md` — "House philosophy" (the amended structural-vs-numerical rule) and "Current state" (the
   geometry + safe-mode bullet).

## State caveats

- This is a branch; `main` does not have any of it. Commit per logical chunk, push to `geometry-refactor`.
- The deferred uniform **tangent** layer, the weighted layer, and the symmetric-probe-derivatives **merge**
  (branch `probe-derivatives`) are unrelated and untouched here.
