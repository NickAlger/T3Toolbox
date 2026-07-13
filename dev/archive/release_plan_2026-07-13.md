# Release plan — T3Toolbox 2026.0.0 to PyPI

_Drafted 2026-07-12. **DRAFT — for Nick's review.** Completes R1 (packaging correctness) and
ships 1.0. Precedent: the gated-slice pattern (`dev/archive/docs_pass_plan.md`,
`dev/archive/docs_split_plan.md`)._

## Decisions recorded (Nick, 2026-07-12)

1. **PyPI name**: `t3toolbox` is free (404 on the simple index — definitive; the name is already
   in normalized form). Claimed via the pending-publisher registration in REL-3.
2. **Wheel smoke test**: approved (REL-2).
3. **Trusted publishing**: approved; Nick gets the click-by-click walkthrough (REL-3).
4. **Dependency floors + `requires-python`**: approved — `numpy>=1.22`, `jax>=0.4.30` in the
   extra, `requires-python = ">=3.9"` (3.8 was claimed but never tested, and is EOL).
5. **arXiv**: updated version submitted; announcement scheduled 11:59pm tonight; URL unchanged.
   **Sequencing consequence: tag/publish AFTER the announcement**, so the docstring citations
   (local-`t4s.pdf` numbering) resolve publicly from the first release minute.
6. **Banner**: comes off NOW (Nick's call — the owner of the only-at-shipping rule relaxed it),
   in REL-1 rather than the ship commit. The CLAUDE.md sentence noting the banner goes with it.
7. **Classifiers**: approved (REL-1).
8. **`CITATION.cff`**: approved (REL-1) — cites the software + the T4S arXiv paper.
9. **Badges** (tests + docs): approved (REL-1). **CONTRIBUTING pointer** in the README to the
   rendered Contributor guide: approved (REL-1). **Zenodo: deferred** — Nick will do it later.
   **Blake is on board** with going public.

Sequencing note: the install recipe flips to `pip install t3toolbox` only in the ship commit
(REL-4) — the README must never instruct a command that 404s.

## REL-1 — metadata + front-matter prep (one commit, can land now)

- `pyproject.toml`: `dependencies = ["numpy>=1.22"]`; `jax = ["jax>=0.4.30"]`;
  `requires-python = ">=3.9"`; classifiers →
  `Development Status :: 4 - Beta` (or 5 — Nick's call at review),
  `Intended Audience :: Science/Research`,
  `Topic :: Scientific/Engineering :: Mathematics`,
  `Operating System :: OS Independent`,
  `Programming Language :: Python :: 3` + explicit `:: 3.9` … `:: 3.13`.
  (License stays as the SPDX `license = "MIT"` — no license classifier; modern setuptools
  deprecates carrying both.)
- **Banner off**: delete README line 1; update the CLAUDE.md sentence that references it.
- `CITATION.cff`: software metadata (authors Alger + Christierson, repo, license, version) with
  `preferred-citation` = the T4S arXiv paper (arXiv:2603.21141) — GitHub renders the "Cite this
  repository" button from it.
- README: CI badges (tests + docs workflows) at the top; a one-line contributing pointer to
  https://nickalger.github.io/T3Toolbox/contributor_guide.html.
- Gates: docs `-W` build (README isn't rendered by Sphinx, but CLAUDE/CITATION touches ride
  along); suite untouched → CI on push.

## REL-2 — wheel smoke test (verification, no commit unless it finds bugs)

The truth check nothing has done yet: everything so far ran via `PYTHONPATH`, never from an
installed artifact.

1. `python -m build` (sdist + wheel) in the `t3toolbox` env (pure-Python — the known CC/CXX trap
   only bites native builds; `pip install build --only-binary=:all:` if needed).
2. `twine check dist/*` (metadata renders on PyPI).
3. Inspect the wheel's file list: all `t3toolbox*` subpackages present, no strays (tests/, dev/,
   docs/, examples/ must NOT be in the wheel; the sdist may carry more — check what
   `packages.find` + default sdist rules produce).
4. **Fresh venv, numpy-only**: install the wheel, `import t3toolbox`, run the quickstart's
   numpy examples — this also validates the jax-optional claim (CI never tests jax-absent).
5. **Fresh venv, with `[jax]` extra**: install `dist/*.whl[jax]`, run the full
   `docs/getting_started.rst` doctest against the installed package.
6. Any failure → fix, fold into REL-1's commit or its own, re-run.

## REL-3 — the release workflow + PyPI trusted publishing (one commit + Nick's clicks)

**The workflow** (`.github/workflows/release.yaml`, mine to write): triggers on tag push
`v20*`; job 1 builds sdist+wheel + `twine check` + uploads as artifact; job 2 publishes to PyPI
via `pypa/gh-action-pypi-publish@release/v1` with **trusted publishing** (OIDC — no tokens
anywhere), gated on the `pypi` GitHub environment. Optional job 0 (first run only): publish to
TestPyPI from a `v*-rc*` tag for the dry run.

**Nick's click-by-click** (one-time, ~5 minutes):

1. Log in / create your account on https://pypi.org (enable 2FA — required for new publishers).
2. Account → **Publishing** → "Add a new **pending** publisher" (pending because the project
   doesn't exist yet — this simultaneously reserves the name for your first trusted publish):
   - PyPI project name: `t3toolbox`
   - Owner: `NickAlger` · Repository: `T3Toolbox`
   - Workflow name: `release.yaml`
   - Environment name: `pypi`
3. Repeat on https://test.pypi.org with environment name `testpypi` (for the dry run).
4. In GitHub: repo → Settings → Environments → create `pypi` (and `testpypi`). Optionally add
   yourself as a required reviewer on `pypi` — then every publish waits for your explicit click.

**Dry run**: tag `v2026.0.0-rc1` → workflow publishes to TestPyPI → in a fresh venv:
`pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ t3toolbox`
→ import + quickstart. Green = the pipeline is proven end-to-end.

## REL-4 — the ship (one commit + one tag, AFTER tonight's arXiv announcement)

- `CHANGELOG.md`: `[Unreleased]` → `[2026.0.0] — 2026-07-XX` (content already drafted).
- Install recipe flip in README + `docs/index.rst` + anywhere else that says clone-then-pip:
  `pip install t3toolbox` primary, `pip install "t3toolbox[jax]"` for the jax extra
  (finally documenting the extra), from-source retained as the development install.
- Final gates: full suite, docs `-W`, and a rebuild-the-wheel sanity pass.
- Commit, `git tag v2026.0.0`, push with tags → the workflow publishes → **verify from a clean
  venv against real PyPI** → GitHub Release created from the tag with the CHANGELOG section as
  notes.

## Post-release

- HANDOFF + CLAUDE.md: 1.0 SHIPPED; current-state pointers refreshed; this plan → `dev/archive/`.
- Zenodo DOI: deferred (Nick, later).
- Watch the first `pip install` bug reports; the 1.1 threads (the `fit(...)` facade, the
  weighted revival) pick up from the handoff.

## Open items — RESOLVED

- **Development Status classifier: `5 - Production/Stable`** (Nick, 2026-07-12, after the
  stability-promise-vs-roadmap discussion: the classifier vouches for what exists — correct,
  dense-verified, API stabilized by the naming pass; planned work is additive. The would-we-break
  test: no — the house pattern is add-alongside).
- Ship timing: any time after tonight's arXiv announcement (11:59pm).
