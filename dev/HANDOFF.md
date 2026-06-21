# T3Toolbox — current handoff

_Updated 2026-06-21._

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
- **Next:** the **uniform-layer fix** (the 1.0 centerpiece).

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
1. **Fix the uniform layer** — the 1.0 centerpiece. It now **subsumes the backend module reorg + the
   per-op polymorphism triage** (design: **`dev/uniform_fix_plan.md`**; reorg context: `dev/naming_review.md` §4). Natural entry points: the per-op
   polymorphism triage (*already-poly / make-poly / can't-or-shouldn't*; ragged/uniform **inferred**
   via `is_ndarray`) and the **`ut3_sampling` packing bug** (a prior stopgap). Parts A–E below.
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
