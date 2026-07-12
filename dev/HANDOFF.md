# T3Toolbox — current handoff

_Updated 2026-07-12 (evening). Prior history: `dev/archive/handoff_2026-07-12_1.0_complete.md`
(the 1.0 completion: R1–R7, the R4 doc pass, the docs user/dev split S1–S5, the cordon/ETT/
literature morning)._

## Where we are — RELEASE IN FLIGHT

**2026.0.0 is one tag away from PyPI.** Plan: `dev/release_plan.md` (all decisions settled).
REL-1 (metadata, banner off, CITATION.cff, badges) ✅, REL-2 (wheel smoke test — also fixed the
jax-less import print) ✅, REL-3 (release workflow, trusted publishing, Nick's PyPI setup) ✅,
and the **TestPyPI dry run is VERIFIED end-to-end** (`v2026.0.0-rc1` → workflow → fresh-venv
`pip install` from TestPyPI → silent import, correct math).

**→ Next: REL-4, the ship** — held until after tonight's (2026-07-12, 11:59pm) arXiv
announcement of the updated T4S preprint, so the docstring citations resolve publicly.
Sequence: CHANGELOG `[Unreleased]` → `[2026.0.0]`; install recipe flips to
`pip install t3toolbox` (+ document the `[jax]` extra); final gates; tag `v2026.0.0`; verify
from real PyPI; GitHub release; then refresh this file + CLAUDE.md to "1.0 SHIPPED".

## Active threads

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
