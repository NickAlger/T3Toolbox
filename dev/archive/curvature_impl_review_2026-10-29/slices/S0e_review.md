# S0e review (reviewer) — GaussNewtonModel -> LocalModel; least_squares_problem; newton_cg explicit keywords

Base: main @ 3e5bc92, 28 files uncommitted (`git diff --stat`: +503 / -180). Read: the handoff, CLAUDE.md
(including the orchestrator's on-disk edit, which now names `fitting.LocalModel`), plan §C.9 (keywords),
§C.13, §C.16, §G, §H S0e, §I (I-4 / I-18b / I-26 / I-30), §K, §L.2 / L.6, and the cited docs.

## 1. Rename completeness — PASS (one MINOR on m37)

- `grep -rIl GaussNewtonModel --exclude-dir=.git --exclude-dir=dev .` returns only: CLAUDE.md (2, both
  "renamed from" pointers, the orchestrator's), CHANGELOG (history + the new entry), release_notes (the
  new upgrade note + the 2026.1.0 history with a "(renamed ... 2026.3.0)" pointer), parameters_not_closures
  (the `UniformGaussNewtonModel` history), oracle_sweeps/README (the change record), and
  test_optimizers_frontend (asserts the old name is gone). No alias anywhere; `hasattr(fitting,
  'GaussNewtonModel')` is False (tested). No stray `_setup` caller (tests/test_fitting.py's `_setup` is a
  local test helper, unrelated).
- Class docstring (`fitting.py:188-191`): "the local model ... the curvature actions there (the
  Gauss-Newton `H = JᵀJ`: gn_hessian / gn_quadratic)". It no longer calls itself Gauss-Newton and promises
  nothing about `hessian='exact'`. Same for the backend `LocalModel` docstring. Good.
- `sweep_models.py` docstring change is recorded in `tests/oracle_sweeps/README.md`. Good.
- m37 qualified names: `fitting_and_optimization.md:243-262` rewritten exactly as the plan asks;
  numerical_contracts, batching_and_stacking, api_reference (fully-qualified `:py:class:` target),
  naming_conventions all qualified. See MINOR-1 for the pages the checklist missed.

## 2. Constructor (§C.16 + I-30) — PASS

- `optimizers.least_squares_problem(geometry, kind, sample, data, x0, order=None, weight=None,
  regularizer=None, chunk_size='auto', batch=None) -> LeastSquaresSetup` matches the C.16 signature
  (`optimizers.py:196-207`). Body is base `_setup` byte-for-byte except the two `return` lines.
- `LeastSquaresSetup` is `@dc.dataclass(frozen=True, eq=False)` with `problem`, `x0_cores`, `rewrap`. Not
  iterable, not indexable (tested: unpacking and `[0]` raise `TypeError`, so an old-style
  `problem, init, rewrap = ...` breaks loudly). `eq=False` gives identity eq/hash; I checked this against
  the `ExplicitEquality` rule (`backend/common.py:523`): that mixin is scoped to the array-carrying
  *mathematical* classes and its error message points to `allclose` / `corewise_equal`, which a result
  record does not have, so plain identity is the right choice here, and the comment says why.
- In `optimizers.__all__` (both names); not at the root (tested). `_setup` removed, not aliased (tested).
  All four optimizers call it (tested with a `mock.patch.object(..., wraps=...)` spy, call_count 4).
  `_geometry_ops` untouched and private. `ragged_backend_geometry` not in `fitting.__all__` (tested).
- Doctest: seeded, run-and-paste style, shows structure (shapes, type names, `stop_reason`), one
  `np.allclose`-style bool, the structural gotcha with `+IGNORE_EXCEPTION_DETAIL`. Matches
  doctest_style.md. API page entry present (`api_reference.rst:84-87`).
- **Adjudication of the C.16 correction — the implementer is right, the plan text was imprecise.** Base
  `_setup` (3e5bc92 `optimizers.py:206-208`) calls `uf.uniform_minimal(x0, ...)` unconditionally on the
  uniform branch, and `_ragged_minimal` only under `_is_manifold_geometry` on the ragged branch. Verified
  empirically on a non-minimal x0 (tucker (2,3,2), tt (1,2,1,1); minimal tucker (2,2,1)):
  UNIFORM_MANIFOLD -> reduced, **UNIFORM_COREWISE -> reduced**, MANIFOLD -> reduced, COREWISE -> untouched.
  So C.16's ".x0_cores ... reduced to minimal ranks on manifold geometries" holds only for ragged x0; the
  docstring ("on a manifold geometry a ragged x0 is reduced ..., and a uniform x0 always is") is correct.
  The plan text should be amended (orchestrator), not the code.

## 3. Keywords (§C.13) — PASS

- `newton_cg` has `max_newton=30, gtol_rel=1e-8, cg_maxiter=200, c_armijo=1e-4,
  on_line_search_failure='stop', g0_reference=None, g0norm_newton=None, g0norm_cg=None,
  cg_forcing_power=0.5` — identical to the backend defaults (`backend/optimizers.py:784-797`; also checked
  by `test_newton_cg_keywords_mirror_backend` via `inspect.signature`). No `**kwargs`. The S4-S6
  keywords (`hessian`, `curvature_damping`, `step_control`, `tr_radius0`) are correctly absent.
- Positional order of the first 15 parameters is unchanged from base, so positional callers are
  unaffected; every option the backend accepts is now explicit, so "every valid call is unchanged" holds.
- Every key T3Polynomial passes (checked in /home/claude/t3polynomial @ 388090d: `order`, `weight`,
  `max_newton`, `verbose`, `val_sample`, `val_data`, `g0norm_newton`, `gtol_rel`, `use_jit`, `callback`,
  `chunk_size`, plus `g0_reference` after §K) is present.
- **The plan's "a misspelt option now raises TypeError instead of being forwarded" was wrong; the
  implementer is right.** Reproduced on an archive of 3e5bc92: `newton_cg(..., max_newtn=3)` raised
  `TypeError: newton_cg() got an unexpected keyword argument 'max_newtn'. Did you mean 'max_newton'?`
  from the backend signature, *after* `_setup` had run (spy count 1). Now it raises at the frontend call
  and the problem is never built (tested with a mock that asserts it is not called). The CHANGELOG wording
  is accurate in substance; NIT-2 is a one-word precision fix.
- `gradient_descent` / `mc_sgd` / `adam` keep `**kwargs`. That matches the plan's scoping (C.13 and the
  §G CHANGELOG line name only `newton_cg`), and the CHANGELOG says so explicitly. Acceptable.

## 4. `docs/conf.py` post-transform — the right tool (NIT-3 only)

- Root cause confirmed in autoapi 3.8.1 itself: `_astroid_utils._resolve_annotation` ends with
  "Sphinx is capable of linking anything in the same module without needing a fully qualified path" and
  strips `annotation.root().name + "."`. On a per-object page there is no `py:module` context, and
  Sphinx's searchmode 1 (`python/__init__.py:942`, `searchmode = 1 if node.hasattr('refspecific')`)
  never uses `py:class` without a module, so it falls into the fuzzy `endswith('.LocalModel')` search.
- Reproduced: built `sphinx -W --keep-going` (Sphinx 9.2.0, autoapi 3.8.1, the CI-equivalent latest
  toolchain) on two scratch copies of the working tree. **With** the transform: `build succeeded`, 0
  warnings. **Without**: exactly 7 "more than one target found for cross-reference 'LocalModel'"
  (6 factories + `Problem.local_model`), build fails. HTML diff of the two trees: exactly the 6 factory
  pages differ, each by one link, `../backend/optimizers/LocalModel.html#t3toolbox.backend.optimizers.LocalModel`
  -> `LocalModel.html#t3toolbox.fitting.LocalModel`. `Problem.local_model.html` is identical (the fuzzy
  pick happened to be right there). No other HTML file differs.
- Simpler alternatives, checked and rejected:
  - fully qualified (or string / `from __future__ import annotations`) return annotations: autoapi strips
    the same-module prefix from *whatever* spelling resolves to this module, so the source cannot fix it;
  - an autoapi option: there is none that disables the strip;
  - the templates (`.. py:currentmodule::` on own pages): the directives use `obj.id` (fully qualified),
    and with a current module Sphinx prefixes it again (`t3toolbox.fitting.t3toolbox.fitting.apply_model`)
    -- so the templates would have to switch to module-relative names across function/method/class,
    which changes every object directive in the site to fix seven field types. More blast radius, not less.
  - `suppress_warnings = ['ref.python']`: silences -W but keeps the six links pointing at the backend.
- Logic checked: `hasattr('refspecific')` is exactly Sphinx's own fuzzy-search trigger; it skips refs that
  already carry a module (so module pages, where `py:module` is set, are untouched); inner-first
  `reversed(findall(desc))` makes a nested member's refs take the member's scope before the enclosing
  class sees them; it only sets `py:module` / `py:class`, never text or ids. Robustness: `env.domains['py']`
  is typed API in Sphinx 9 (`_DomainsContainer.__getitem__`) and a dict in Sphinx 7; `PythonDomain.modules`,
  `SphinxPostTransform` and `Node.findall` exist across `sphinx>=7` / its docutils floor. Fine.

## Findings

### BLOCKER — none. MAJOR — none.

### MINOR-1 — m37 is incomplete on two contributor pages the C.13 checklist did not list
The checklist enumerated the `GaussNewtonModel` sites, but m37 is broader: "every doc that mentions both
[LocalModels] uses the qualified forms". Two pages now mention both with a bare backend `LocalModel`:
- `docs/contributor/precompute_and_caching.md:24` -- "`LocalModel.geom_aux` / the frontend models'
  `geometry_aux`" (after the rename a bare `LocalModel.geom_aux` reads as the frontend class, which has
  no `geom_aux`), and `:63` -- "the backend `LocalModel` and both frontend models pass their stored
  companion" ("both frontend models" has been stale since the 2026.2.0 merge; there is one).
  Fix: `backend.optimizers.LocalModel.geom_aux` / `fitting.LocalModel`'s `geometry_aux`; and "the
  backend `backend.optimizers.LocalModel` and the frontend `fitting.LocalModel` pass their stored companion".
- `docs/contributor/fitting_internals.md` now names `fitting.LocalModel` (:42) but keeps "the backend
  `LocalModel`" (:52) and a bare "`Problem`/`LocalModel`/optimizers/kinds" (:121). Low ambiguity in
  context; qualify both as `backend.optimizers.LocalModel` for consistency with the rule.
(`fitting_and_optimization.md:208,210`, the file-labelled structure diagram, is fine as is: each bare
name sits in its file's row. `deferred_and_rejected.md:81` "the pytree `LocalModel`" mentions only one
class; optional `backend.optimizers.` prefix.)

### NIT-1 — a new docstring cross-reference that silently does not link
`optimizers.py:185` (`LeastSquaresSetup` docstring) uses a bare `:py:func:`least_squares_problem``. On a
class's own autoapi page Sphinx's exact-match search (searchmode 0) has `py:class` but no module, so it
does not resolve -- no warning, just an unlinked literal (checked in the built HTML; the bare
`:py:class:`LeastSquaresSetup`` inside the *function* page does link, through the function's prefix).
Fix: `:py:func:`~t3toolbox.optimizers.least_squares_problem``. (Same pre-existing failure, not S0e's: the
three bare `:py:func:` factory refs in the `fitting.LocalModel` class docstring (`fitting.py:193-194`),
`apply_model` / `entries_model` / `probe_model`; fixing them the same way in this rename commit is cheap.)
*Follow-up, explicitly not for S0e:* I built a variant of the transform without the `refspecific`
condition (so every module-less py ref inside an object gets the object's module, which is what Sphinx
does on a module page). It stays `-W` clean and resolves the bare roles that are dead today on **~90
own pages** site-wide (sampled: `fitting.LocalModel`'s factory refs, `LeastSquaresSetup`'s
`least_squares_problem`, `T3Tangent`'s `MANIFOLD.inner`; all resolve to the right target). That is a
site-wide docs fix with a big HTML diff -- its own docs commit, not this rename.

### NIT-2 — CHANGELOG / test docstring wording
`CHANGELOG.md:67` and `tests/test_optimizers_frontend.py:439`: "the backend loop rejected it after the
setup work" -- it was the backend *call's signature* (`TypeError: newton_cg() got an unexpected keyword
argument ...`), before any iteration. Suggest "the backend call rejected it, after the setup work".

### NIT-3 — conf.py comment: name the upstream cause
Add one clause to the comment block (`docs/conf.py:145-154`) pointing at autoapi's
`_astroid_utils._resolve_annotation` same-module strip and at Sphinx's `hasattr('refspecific')` fuzzy
trigger, so a later maintainer knows the transform can go when autoapi stops stripping (or sets a module
on own pages). Optional.

### NIT-4 — pin the corrected C.16 statement with a test
`test_x0_cores_minimal_on_manifold_only` (`tests/test_optimizers_frontend.py:389`) covers ragged only, so
the docstring's corrected claim "a uniform x0 always is [reduced]" (the point where the plan was wrong) is
untested at the constructor. Add the uniform twin (non-minimal `UniformTuckerTensorTrain.from_t3(x0)` on
`UNIFORM_COREWISE` comes back at minimal ranks -- verified true above) and rename the test
`test_ragged_x0_cores_minimal_on_manifold_only`.

### NIT-5 — release_notes reflow
`docs/release_notes.md:62-63`: the inserted pointer leaves a ragged source line ("above) now serves both
representations," then the old line). Reflow the paragraph.

### Observations (not S0e findings)
- `docs/user_guide.rst:351-357` (pre-existing text, now about `LocalModel`) says the model does
  "retraction at a base point" and that the four optimizers "drive it". Today the frontend model has no
  `retract` (S6 adds it, C.13) and the optimizers drive the *backend* `LocalModel`. Becomes true at S6
  for the first half; worth a pass at S13.
- The new "Upgrading from 2026.2.0" section has only the S0e items, as the brief asked. §G also lists
  S0d's items for it (`g0_reference='warm_start'`, dropping the `history[0]['gnorm']` bookkeeping,
  `stats['newton_steps']`); S0d did not add them. Track for S13.
- Plan C.16's return comment ("reduced to minimal ranks on manifold geometries") should be amended to
  the implementer's (correct) wording.

## 5. House style / scope — PASS
- No formatter churn: every hunk is the rename, the constructor, the explicit keywords, or their docs.
  The `newton_cg` signature regroup is the signature_style.md blank-line-grouped form, with comment
  columns aligned within each group (the long `on_line_search_failure` name breaks its group's column
  exactly as in the backend signature it mirrors).
- No behaviour change beyond the CHANGELOG lines: `least_squares_problem` is base `_setup` with two
  return statements changed; the optimizers pass the same arguments; the message strings that named the
  class (`require_unstacked_for_regularizer(..., 'LocalModel')`, the `_require_unstacked` text) are
  rename-only. The touched example's comment change also fixes a stale claim ("no GaussNewtonModel
  yet" -- `apply_derivatives_model` exists).
- Release notes "Upgrading from 2026.2.0": rename + constructor only, per §G / the brief.

## 6. Gates (run by me, last)
- Touched test files (`test_optimizers_frontend`, `test_fitting`, `backend/test_optimizers`,
  `test_sharing`, `test_dispatch`, `test_oracle_sweep`), `-n 2`: 207 passed, 1 skipped, 13,158 subtests.
- `pytest --doctest-modules t3toolbox`: **197 passed**.
- `python -m doctest` on all 44 `docs/*.md` + `docs/contributor/*.md` pages (minus doctest_style.md) and
  `docs/getting_started.rst`: 0 failures.
- `examples/fit_hilbert_from_apply_derivatives.py` (the one touched example): exit 0 (6.5 s).
- `sphinx -W --keep-going` (Sphinx 9.2.0 / autoapi 3.8.1): clean with the transform; 7 warnings without.
- Full suite, `-n 2`, two halves: 283 + 646 = **929 passed, 1 skipped, 3,256 + 40,002 = 43,258
  subtests** -- identical to the implementer's counts.
- Not checked: the numpy 1.x generation (no such env here); the new doctest prints only shapes, type
  names, a stop reason and one bool, so it is generation-independent by construction.
- `T3TOOLBOX_SLOW_TESTS=1 pytest tests/test_oracle_sweep.py -n 2`: **5 passed, 12,790 subtests** (the
  warnings are the expected adam-on-MANIFOLD RuntimeWarning and a jax float64 UserWarning) -- matches.
- Working tree left as found: no files of mine in the repo (doc builds ran on scratch copies; pytest ran
  with `-p no:cacheprovider`). `CLAUDE.md` shows as modified -- that is the orchestrator's edit, not mine.

## Must-fix before commit (small, doc-only, no re-review needed)
1. MINOR-1: qualify the backend `LocalModel` in `docs/contributor/precompute_and_caching.md:24,63` (and
   drop the stale "both frontend models"), and in `docs/contributor/fitting_internals.md:52,121`.
2. NIT-1 (the S0e-introduced part): `optimizers.py:185` -> `:py:func:`~t3toolbox.optimizers.least_squares_problem``.

Recommended, optional: NIT-2 (CHANGELOG "backend loop" -> "backend call"), NIT-3 (conf.py comment cites
the upstream cause), NIT-4 (uniform twin of the x0-reduction test), NIT-5 (release-notes reflow), the
three pre-existing `fitting.LocalModel` factory refs. Orchestrator: amend plan C.16's x0_cores comment.

VERDICT: APPROVE
