# T3Toolbox — current handoff

_Updated 2026-07-11 (second session of the day)._

## Where we are

**The doc pass (R4) is DONE (2026-07-11)** — on top of the naming pass + backend module reorg
(earlier today) and the uniform frontend U7 + frame rename (2026-07-10). Branch `main`, direct
commits. Full suite green (593 tests / 40,215 subtests, exit-code checked); **the docs build is at
ZERO warnings and CI now enforces `-W`**.

**R3 (README) is DONE too** (same session): README rewritten against the current library —
banner kept (off only at shipping), pitch + diagram, quickstart seeded from `getting_started.rst`
(snippets re-verified end-to-end; the claimed ranks/outputs are real), refreshed functionality
list (fitting/optimizers/uniform/safety included), docs + examples links.

**→ Next: R5 (test CI), then R6 (cleanup).** The Pages toggle is FLIPPED (Nick, 2026-07-11) and
the R4 commits are pushed — the new `docs` workflow deploys from `main`; the old `gh-pages`
branch is now dead weight (delete whenever, optional).

## Done this session (R4, the doc pass) — slices D1–D5

Plan of record (approved, all decisions settled, executed): **`dev/archive/docs_pass_plan.md`**.
Warning burn-down: 394 (baseline, fraction of the surface) → 62 (D1) → 40 (D2) → 7 (D3) → **0** (D4).

- **D1 — conf.py + toolchain (`9cce8503`):** autoapi now covers the **full validated surface**
  (10 frontend + 37 backend module pages; excluded: `OLD_*` + the parked weighted modules only —
  backend users are first-class, Nick's decision). **Verbatim source signatures**: a
  `verbatim_signature` jinja filter in `conf.py` (ast-based) + the custom autoapi templates render
  each function/method's signature exactly as written — **the trailing shape-contract comments
  included** (autoapi's regenerated signatures drop comments; this was the settled option-(i)
  spike). Version single-sourced from `pyproject.toml` via `tomllib` (2026.0.0 — docs builds are
  py3.11-only; zero constraint on the library). myst-parser enabled. Dropped
  `imported-members`/`private-members` (each object documented once; curated `__all__` publics).
  Template gotcha for posterity: the stock templates' **double blank lines are load-bearing**
  (trim_blocks eats one newline after an inline `{% endif %}` on the directive line).
- **D2 — docstrings (`a507823b`), the one code-touching slice:** module docstrings seeded into the
  **31 modules that had none** (content from `docs/naming_conventions.md`'s module map); the 14
  docstring RST breakages fixed (the ``None. typo ×3, a missing backtick in `t3svd` Returns,
  blank-line structure around lists/formulas, five first lines that wrapped mid-markup —
  autosummary truncates at the line break); the ambiguous `.sum_stack` xref qualified; PARKED
  notes on the two weighted stragglers; a `conf.py` `autoapi-skip-member` dedup for the deliberate
  `if jax_available:` rebindings in `backend/common.py`. Suite-gated.
- **D3a — information architecture (`f5edfa5a`):** `index.rst` rewritten as a compact landing;
  **`user_guide.rst`** (salvaged accurate background + fresh current-state sections: the real
  numpy/jax dispatch story, safe/unsafe, fitting, frontend-vs-backend, the CURRENT uniform
  minimal-rank contract); **`design_notes.rst`** (all 23 `docs/*.md` rendered, grouped);
  **`api_reference.rst`** (hand-written landing mirroring `__init__.__all__` + backend map by
  area). `modules.rst` deleted. The dead `t3toolbox.jax` namespace story is gone.
- **D3b — quickstart (`387eba0f`):** `getting_started.rst`, five run-verified examples (create/
  arithmetic/t3svd, sampling vs dense, tangent+retract at a minimal-rank point, a real Newton-CG
  fit to <1e-6, jax dispatch + jit); **passes `python -m doctest` clean** — outputs are real.
- **D4 — link audit (`9ddb6d23`):** out-of-tree links (dev/archive ×5, examples ×1, .tex ×1) →
  GitHub blob URLs; the dangling "why fit from derivatives" link removed by rewording (Nick:
  research-side rationale, not library docs). **Zero-warning build**; render sweep over all 23
  design-note pages clean.
- **D5 — docs CI:** `.github/workflows/build-sphinx-docs.yaml` replaced — pinned actions
  (checkout@v4 / setup-python@v5 / py3.11 / `docs/requirements.txt`), **`sphinx -W`** (warnings
  are errors), PRs build-only, pushes to `main` deploy via the official Pages artifact actions
  (`upload-pages-artifact@v3` + `deploy-pages@v4`; no more gh-pages branch commits, no more
  `@master` third-party actions). Needs Nick's Pages-source toggle (above); the old `gh-pages`
  branch can be deleted afterward (optional).
- Local docs build (recorded in the plan): install `docs/requirements.txt` into the `t3toolbox`
  env with `--only-binary=:all:`; then
  `python -m sphinx -W --keep-going -b html docs docs/_build/html` from the repo root.

**Decisions settled with Nick this session** (full statements in the archived plan): backend is a
first-class documented surface; all `docs/*.md` render as user pages; `2026.0.0` everywhere;
verbatim-signature spike approved (and it worked); reproducible-output rule is docstring/docs-page
scoped — `examples/` keep raw convergence floats; no "why fit from derivatives" note (research-
side); tomllib in conf.py confirmed harmless to library compat; no WIP banner on the docs site.

## Prior sessions (2026-07-10 .. 11) — naming pass, U7 frontend, frame rename

Summarized; details in the git log, `dev/archive/naming_pass_plan.md`,
`dev/archive/uniform_optimizers_plan.md`.
1. **Naming pass + backend module reorg DONE** — the family-prefix grammar, module map, curated
   `t3toolbox/__init__.py` (R2); catalog: `docs/naming_conventions.md`.
2. **Uniform layer CLOSED (R7)** — backend, optimizers, U7 frontend (ragged-vs-uniform inferred
   from `x0`), jit-compile-once.
3. **`basis`/`base` → `frame` rename DONE**; packaging fix (`pip install -e .`).

## Active thread: the T3Toolbox software reference paper (independent)

Scope + curation: **`dev/paper_scope.md`** (11 operation groups; two settled findings: the
canonical-Gaussian-tangent result, and minimal rank is a correctness precondition for nothing).
**Not the T4S paper** (that is the existing arXiv preprint, a historical algorithm reference).
Next on this thread: walk the groups starting at Group 6 (`docs/symmetric_probe_derivatives.tex`
is nearly a drop-in chapter).

## Next steps

1. **R5 test CI** (pytest + numpy 1.x/2.x matrix; wire doctests in — `getting_started.rst` and the
   module doctests are ready for it; the docs CI is done and separate).
2. **R6 cleanup:** the remaining `OLD_test_*.py` files (**`OLD_test_linalg.py` needs a real
   coverage audit** — there is no current `test_linalg.py`; the other six have modern
   counterparts), `.idea/` (uncomment the ready line in `.gitignore`), `t4s.pdf` untracked-status
   decision. `docs/make.bat` was committed in D1. `CHANGELOG.md` still to create (R1 leftover;
   pyproject links to it).
3. **→ 1.1:** the Goal-1 `fit(...)` facade; revive/redesign the weighted layer.

## The 1.0 roadmap — summary

- **R1** packaging correctness — mostly done; `CHANGELOG.md` still to create.
- **R2 — DONE (2026-07-11):** public API surface + naming/organization review.
- **R3 — DONE (2026-07-11):** README + quickstart (banner off only at shipping).
- **R4 — DONE (2026-07-11):** docs build + design rationale rendered as user docs; zero warnings,
  `-W` in CI; full frontend+backend API reference with verbatim source signatures.
- **R5** test CI (numpy matrix + doctests). No auto-formatter near the curated style.
- **R6** cleanup — `OLD_test_*` etc., delete only after confirming preserved.
- **R7 — DONE** (uniform layer + optimizers + U7 frontend).

## Don't-trip constraints (the maintainer's standing rules)

- **The docs build must stay at zero warnings** — CI runs `sphinx -W`; a new doc/docstring that
  warns fails the build. Run the local build before pushing doc-touching changes.
- **Naming: read [`docs/naming_conventions.md`](../docs/naming_conventions.md) before naming
  anything new**; user-over-convention; the semantic markers ("corewise", "numerically_") and the
  representation-encoding parameter names are load-bearing.
- **The uniform optimizer requires a minimal-rank base** (`uniform_minimal`; frontend calls it
  transparently; `uniform_least_squares_problem` rejects non-minimal x0).
- **The packedness-mirror convention** (U3.5) — don't "normalize" it to a flag.
- **A uniform op needs more than dense-vs-ragged** — exact output masks + garbage-robustness
  (`docs/testing_strategy.md`). Masks are host numpy (`np`), supercores `xnp`.
- Numerical test assertions use a **tolerance**; exact comparison only for structure.
- Never delete an `OLD_*` (or anything) until functionality is **confirmed preserved**.
- "DO NOT USE" banner stays until the literal moment of shipping (README only; the docs site
  carries no banner — Nick's call).
- **No automated tool rewrites the code style** (esp. the shape comments). No `manifold.py` rename.
- **Weighted layer is out of scope** until its post-1.0 revival — no renames, no deletions, only
  reference-fixes required to keep it importable.
- Research caveats are not user-facing. Notes are preserved/relocated, never lost.
- Doctest outputs in docstrings/docs pages are **run-and-pasted, never hand-written**
  (`docs/doctest_style.md`); `examples/` scripts are exempt (raw convergence floats are the point).
