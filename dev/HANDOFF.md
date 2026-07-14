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

- **Newton-CG warm-start reference overrides — DONE, on `main` (2026-07-14), uncommitted.**
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
