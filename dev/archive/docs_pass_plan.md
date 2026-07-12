# R4 docs-pass plan — Sphinx build, rendered design docs, full API reference

_Drafted 2026-07-11. Approved by Nick the same day (all decisions settled)._
_**EXECUTED 2026-07-11** — slices D1 (`9cce8503`), D2 (`a507823b`), D3a (`f5edfa5a`),
D3b (`387eba0f`), D4 (`9ddb6d23`), D5+D6 (docs CI + wrap-up). Warning burn-down:
394 → 62 → 40 → 7 → **0**, with `-W` now enforced in CI. Archived per the handoff ritual;
current state lives in `dev/HANDOFF.md`._
_Precedent: `dev/archive/naming_pass_plan.md` (slices, one commit each, gated)._

## Scope

**R4 only**: make the Sphinx build correct and complete, render every `docs/` document as a user
page, produce a full API reference over the validated frontend **and backend** surface, and fix the
docs CI that publishes the site. **Out of scope:** R3 README/quickstart (deliberately *after* R4 —
doing the docs first situates us for the README), R5 test CI, R6 cleanup (except two trivial riders
noted below).

## Decisions already made (Nick, 2026-07-11)

1. **(a) Full validated surface in the API reference — backend is first-class.** Many users will
   only ever use the backend; the reference must cover it completely.
2. **(b) Every document in `docs/` becomes a rendered user page.**
3. **(c) Version is `2026.0.0` consistently everywhere** (kills `conf.py`'s stale `release = '0.1'`).
4. **(d) R3 (README) happens after R4.**
5. **(e) Reproducible output is a docstring-example rule, not an `examples/` rule** (Nick,
   2026-07-11 follow-up): the `examples/` scripts keep printing raw floating-point convergence
   behavior — no seeding/reproducibility requirement there. The doctest reproducibility rule
   applies to examples inside docstrings and docs pages, and even there may be *rarely,
   intentionally* broken when raw numbers illustrate a key point.
6. **(f) No "why fit from derivatives" note.** That rationale is Nick's ongoing research (uses the
   library, not part of it). Fitting-from-derivatives is documented as a *feature*; the dangling
   link in `entries_apply_probe.md` is resolved by rewording the sentence to drop it (D4).
7. **(g) No WIP banner on the docs site** — the README banner comes off soon anyway; don't add it
   elsewhere.

## Current state (verified by a local build, 2026-07-11)

Build command used: `cd docs && <tttt-env-python> -m sphinx -b html . /tmp/t3docs_build`
(Sphinx 7.4.7 / sphinx-autoapi 3.6.1 / pydata-sphinx-theme 0.16.1 — currently only installed in the
old `tttt` env). Result: **exit 0, 394 warnings** — and the docs CI publishes this broken output to
nickalger.github.io on every push to `main`. The defects:

1. **`autoapi_ignore` excludes most of the library**: `*backend*.py`, `*anifold*.py`, `*niform*.py`,
   `*frame_variation*` — so manifold, tangents, everything uniform, and the entire backend are
   missing from the reference. Only ragged-frontend fragments render.
2. **`docs/index.rst` is stale beyond patching**: links to 10 module pages that no longer exist
   (pre-rename names: `orthogonalization`, `t3svd`, `uniform`, `linalg`, …); documents a
   **`t3toolbox.jax.*` namespace that was removed** when dispatch became inferred-from-array-types;
   contains three duplicated, half-truncated "Jax versions of most numerical functions are suitable
   for" blocks (an old broken edit); every code example uses pre-naming-pass API paths.
3. **No `myst_parser`** — all 23 `docs/*.md` design documents are invisible to Sphinx.
4. **~380 "more than one target found" warnings**: the curated `__init__.py` re-exports (and the
   backend's internal `from common import *`) make autoapi document objects twice.
5. **Stragglers**: `modules.rst` still titled "TuckerTensorTrainTools"; dead `autodoc2_packages`
   config (extension not loaded); `release = '0.1'`; CI actions pinned to `@master`
   (`sphinx-notes/pages`, `ad-m/github-push-action`), deploying even on PR builds.
6. One **dangling doc link**: `docs/entries_apply_probe.md` §4 links
   `derivative_order_information_and_conditioning.md` ("why fit from derivatives") — the file exists
   nowhere in the repo (never written, or lives in the research repo). Also referenced from
   `dev/archive/derivatives_mirror_plan.md`.

_Stale-note correction (for CLAUDE.md in D6): `docs/_build` is **not** committed — it is gitignored
and only on local disk. That worry is obsolete._

Facts gathered for sizing:

- **Module inventory**: 12 frontend modules + 39 backend modules. The unvalidated surface is exactly
  the parked weighted layer, which is cleanly isolated in **two modules**
  (`t3toolbox/weighted_tucker_tensor_train.py`, `backend/wt3_operations.py`) plus two parked
  stragglers inside live modules (`fv_operations.absorb_weights_into_tangent_cores`, the unexported
  `wt3_squash_tails` copy in `t3_operations.py`).
- **31 of 50 modules lack a module docstring** (inventory in the appendix) — with the backend
  becoming a first-class rendered reference, each module page's lede comes from its docstring.
- **Link audit inventory** (all `](...)` targets across `docs/*.md`): same-directory `.md` links
  (myst resolves these automatically); 6 links into `../dev/archive/*.md`; 1 into `../examples/`;
  1 to `symmetric_probe_derivatives.tex`; the 1 dangling link above.
- **No `$`-math anywhere in the md docs** (unicode math throughout) — no math extension needed.
- Suite baseline confirmed green today: 593 tests / 40,215 subtests, exit 0.

## Target site architecture

```
index (landing)          — what T3 is (keep the good diagram prose), install, version
User guide               — quickstart (current frontend API); T3 background (salvaged from old
                           index); numpy/jax dispatch as it actually works now (inferred from array
                           types, constructors' use_jax, jit/pytree story, safe mode vs jit);
                           batching & stacking on-ramp; safe/unsafe mode; uniform overview
                           (with the CURRENT minimal-rank contract — frontend calls uniform_minimal
                           transparently); fitting & optimizers overview; frontend-vs-backend
                           (the razor: who should use which)
Design & internals       — all 23 docs/*.md, grouped:
                           · Conventions & style: naming_conventions, signature_style,
                             doctest_style, testing_strategy
                           · Core design: batching_and_stacking, entries_apply_probe, transposes,
                             fitting_and_optimization, numerical_contract_catalog,
                             rank_continuation, probing_section6_notes,
                             ambient_derivative_transpose_note
                           · T3-SVD: t3svd_design_rationale, t3svd_minimal_ranks, t3svd_verification
                           · Uniform: the 9 uniform_* docs (equivalence contract first)
API reference            — hand-written landing page mirroring the curated __init__ surface,
                           then autoapi: frontend modules; backend modules grouped per the
                           naming_conventions module map. Weighted + OLD excluded.
```

The two `.tex` files (`symmetric_probe_derivatives.tex`, `ttm_t3m_ht_note.tex`) stay as repo source
(they are toolbox-paper material, not renderable pages); links to them become GitHub URLs.

## Slices

Each slice = one commit (D3 may be two), gated as stated. Docs-only slices gate on a clean local
build + nav click-through; the one code-touching slice (D2) gates on the full test suite.

### D1 — toolchain + `conf.py` scope fix

- Add **`myst-parser`** to `docs/requirements.txt` (with sensible version floors for the four deps)
  and install the docs toolchain into the **`t3toolbox` env** via
  `pip install --only-binary=:all: -r docs/requirements.txt` (pure-Python wheels; `--only-binary`
  sidesteps the env's known CC/CXX trap). Canonical local build command becomes
  `cd docs && <t3toolbox-env-python> -m sphinx -b html . _build/html`.
- **`autoapi_ignore` → `["*OLD*", "*weighted_tucker_tensor_train*", "*wt3_*"]`** — i.e. exclude
  exactly the unvalidated surface; manifold, uniform, frame-variations, and all of backend come in.
- Enable `myst_parser` (`myst_heading_anchors = 3` so md→md#heading links resolve). Transient
  "document not in any toctree" warnings expected until D3 wires the toctree — noted, accepted.
- **Version**: source `release`/`version` from `pyproject.toml` via `tomllib` (docs builds are
  py3.11 both locally and in CI, which we control) → `2026.0.0` everywhere, single-sourced.
  (Fallback option if you prefer zero cleverness: a third hardcode with the keep-in-sync comment.)
- **Kill the duplicate-target warnings**: drop `imported-members` from `autoapi_options`, so every
  object is documented once, in its defining module (this also stops `linalg` re-documenting
  `common`'s star-imported members). The curated package-level surface is instead presented by the
  hand-written API landing page (D3). Remaining options tuned empirically in-slice: keep
  `members`, `undoc-members`, `show-inheritance`, `show-module-summary`; keep the operator dunders
  (`__add__` etc. on the frozen dataclasses) visible while suppressing `__repr__`/pytree plumbing
  noise (the custom `_templates/autoapi` templates are the lever if options alone can't).
- Remove the dead `autodoc2_packages` block and the misleading commented-out
  `autoapi_prepare_jinja_env` assignment line (the function itself is live — autoapi picks it up by
  name; keep it).
- Rider: commit the stray `docs/make.bat` (Windows twin of the tracked Makefile).
- **Gate**: local build; warnings should collapse from 394 to roughly the stale-index 10 + the
  transient toctree ones; record the count in the commit message.

### D2 — module docstrings (the one code-touching slice)

Seed the **31 missing module docstrings** (appendix) with 1–4 lines each: what the module holds,
its family prefix, and (where apt) a pointer to the governing design doc — content lifted from the
module map in `docs/naming_conventions.md`, not invented. These become the autoapi module-page
ledes and double as in-editor orientation for backend users. While in there: add a one-line
"parked pending the weighted-layer redesign" note to the two parked stragglers' docstrings
(`absorb_weights_into_tangent_cores`, `wt3_squash_tails`), since they will now render.
**Gate**: full test suite (docstrings are inert, but the gate is cheap insurance).

### D3 — information architecture: new landing page + toctree (may split into D3a/D3b)

- **Rewrite `docs/index.rst`** per the sitemap above; **delete `modules.rst`** (superseded; the
  stale title dies with it). Salvage the good prose (tensor diagrams, minimal-rank background,
  the batching/stacking section — already current); replace the dead `t3toolbox.jax` story with
  the real dispatch story; fix the stale uniform minimal-rank caveat to the current contract.
- **Hand-written API reference landing page** mirroring `t3toolbox/__init__.py`'s `__all__`
  (classes / geometries / models / optimizers / safety), each entry linking to its canonical
  autoapi page; backend section grouped per the naming-conventions module map.
- **Every executable example on the docs pages rewritten against the current API, run, and the
  real output pasted** (per `docs/doctest_style.md` — never hand-written output). This also preps
  R5's doctest wiring. Scope note (decision (e)): this governs docs-page/docstring examples only —
  the `examples/` scripts are untouched by R4 and keep their raw convergence printouts.
- If split: D3a = structure/toctree/design-docs wiring; D3b = examples + dispatch story.
- **Gate**: clean build; all 23 md docs reachable from the nav; sidebar sane in the pydata theme.

### D4 — design-doc rendering + link audit

- Per-doc render check of all 23 md files (heading hierarchy for the sidebar, code fences, tables,
  unicode math passthrough). Rendering fixes only — **no content rewrites** (the docs are current
  as of the naming pass; deeper edits are R3+ territory).
- Execute the link audit: same-dir md links stay (myst-native); `../dev/archive/*.md` (6),
  `../examples/*.py` (1), and the `.tex` link (1) → **GitHub blob URLs** (historical/source
  references, correct as external links); the **dangling
  `derivative_order_information_and_conditioning.md`** link is resolved by **rewording the
  sentence to drop it** (decision (f) — the rationale is research-side, not library docs).
- **Gate**: **zero-warning build** — this is the slice where the count reaches 0.

### D5 — docs CI modernization

- Replace the `@master` actions with a pinned modern flow: `actions/checkout@v4`,
  `actions/setup-python@v5` (3.11), `pip install -r docs/requirements.txt`,
  `sphinx-build **-W** ...` (fail-on-warning — affordable now that D4 reached zero).
- **PRs build only; deploys happen only on push to `main`.** Deploy mechanism (decision (3),
  agreed): the official Pages actions (`upload-pages-artifact` + `deploy-pages`) — build output is
  served from a workflow artifact, no `gh-pages` branch commits at all. Sequence: (1) the new
  workflow lands with the deploy job; (2) **Nick flips one repo setting** — GitHub → Settings →
  Pages → "Build and deployment" → Source: **"GitHub Actions"** (currently "Deploy from a
  branch"); (3) next push to `main` deploys; the URL stays `nickalger.github.io/T3Toolbox/`.
  The old `gh-pages` branch then serves nothing and can be deleted whenever (optional).
- **Gate**: green Actions run; live site spot-checked.

### D6 — wrap-up

Refresh `dev/HANDOFF.md` (R4 done → R3 next); fix CLAUDE.md's stale R4 notes (the "committed
`_build`" claim, the broken-build description); move this plan to `dev/archive/` dated.

## Open decisions — ALL SETTLED 2026-07-11 (2, 3, 5 → (f), D5, (g) above; 1 and 4 below)

1. **Rendered signatures lose the shape comments — the most consequential question here.**
   sphinx-autoapi rebuilds signatures from the AST, so the trailing `#` shape contracts — which are
   *the real type* in this codebase — vanish from the API pages. For a backend-first reference this
   matters. Options: **(i)** timeboxed spike: extend the repo's existing custom autoapi templates
   (`docs/_templates/autoapi/`) to pull each function's *verbatim source signature* — comments,
   alignment and all — into its docs page as a literal code block (autoapi's parsed objects carry
   file paths and line numbers, so the template can slice the real `def ...` lines out of the
   source); **(ii)** a prominent "view source on GitHub" link per object/module page; **(iii)**
   accept the loss. **Recommend (i) with (ii) as the fallback**, spiked inside D1 since it shapes
   the templates. ("Spike" = a small bounded experiment to test feasibility before committing;
   "timeboxed" = fixed budget of ~1–2 h, then fall back rather than sink more time.)
   _Status: **SETTLED — Nick strongly agrees with (i), the timeboxed spike** (2026-07-11)._
2. **`conf.py` version sourcing**: `tomllib` read of `pyproject.toml` (recommended) vs a third
   hardcode. **Compatibility clarification (Nick's question):** `conf.py` executes *only* when
   Sphinx builds the docs — on the maintainer machine and the CI runner, both py3.11. It is not
   part of the installed package and never runs for library users, so tomllib here places **zero
   constraint on the library's Python support** — `requires-python` and the library code are
   untouched, legacy old-Python consumers unaffected. _Status: **SETTLED — tomllib confirmed**
   (2026-07-11)._

## Risks / watch items

- **Build time**: `autoapi_own_page_level = 'method'` over ~48 rendered modules means thousands of
  pages. If the build gets slow, raise to `'function'` or `'class'` — a rendering knob, no content
  loss. Watch in D1.
- **myst strictness**: 23 organically-grown md docs may surface parser surprises (odd nesting,
  bare-HTML fragments). D4 absorbs these; content stays untouched.
- **`autoapi_options` tuning** may take a few iterations to get the dunder-visibility balance
  right; bounded by the custom templates as the escape hatch.
- The docs use the **local `t4s.pdf` numbering** in docstring citations; arXiv catches up by
  release (per CLAUDE.md) — nothing to do, just don't "fix" citations to arXiv numbering.

## Acceptance criteria (R4 done means)

- `sphinx-build` exits 0 with **0 warnings**, and CI enforces `-W`.
- Every validated module renders: all 12 frontend modules except `weighted_tucker_tensor_train`,
  all 39 backend modules except `wt3_operations`; weighted + `OLD_*` absent from the reference.
- All 23 `docs/*.md` render and are reachable from the nav; no dangling internal links.
- Every executable example in the docs runs against the current API with pasted-real output.
- Version reads `2026.0.0` everywhere (pyproject, `__init__` fallback, docs).
- PRs build the docs without deploying; `main` pushes deploy; the live site is correct.
- Full suite green after D2 (the only code-touching slice).

## Appendix — the 31 modules missing a module docstring (D2 inventory)

Frontend (4): `corewise`, `frame_variations_format`, `manifold`, `uniform_frame_variations_format`.

Backend (27): `apply`, `common`, `contractions`, `entries`, `fv_conversions`, `fv_operations`,
`linalg`, `probing`, `ranks`, `sampling_derivatives`, `stacking`, `t3_linalg`, `t3_operations`,
`t3_orthogonalization`, `t3_svd`, `tt_orthogonalization`, `tv_operations`, `ufv_conversions`,
`ufv_masking`, `ut3_conversions`, `ut3_linalg`, `ut3_masking`, `ut3_operations`,
`ut3_orthogonalization`, `ut3_sampling`, `ut3_svd`, `wt3_operations`†.

† `wt3_operations` is excluded from the rendered reference but gets a one-line "parked" docstring
anyway (in-editor orientation; zero-cost).

_(The 19 that already have docstrings: `__init__` ×2, `backend/fitting`, `backend/optimizers`,
`t3_constructors`, `t3_conversions`, `tt_operations`, `ufv_operations`, `uniform_fitting`,
`ut3_constructors`, `utv_operations`, `utv_sampling`, `fitting`, `optimizers`, `safety`,
`tucker_tensor_train`, `uniform_manifold`, `uniform_tucker_tensor_train`,
`weighted_tucker_tensor_train`.)_
