# S0e handoff — GaussNewtonModel -> LocalModel; optimizers.least_squares_problem; newton_cg explicit kw. UNCOMMITTED.

**Files (28).** `fitting.py` (rename; docstrings "local model", qualified targets, realigned); `__init__.py`; `optimizers.py`
(`LeastSquaresSetup` frozen eq=False + `least_squares_problem`, both in `__all__`; `_setup` gone; 4 optimizers call it;
`newton_cg` explicit kw, signature regrouped); docstrings in `backend/{optimizers,fitting}.py`, `manifold.py`,
`shared_geometry.py`. Tests: `test_fitting`, `test_dispatch` (3 `*_gauss_newton_model*` methods -> `*_local_model*`),
`backend/test_optimizers`, `test_sharing`, `test_oracle_sweep`, `oracle_sweeps/sweep_models.py` docstring + its README,
`test_optimizers_frontend` (+8 tests, `TestLeastSquaresProblemAndKeywords`). Example `fit_hilbert_from_apply_derivatives`.
Docs: fitting_and_optimization (m37 qualified + constructor), naming_conventions, release_notes ("Upgrading from 2026.2.0":
rename + constructor only), api_reference, user_guide, numerical_contracts, batching_and_stacking, contributor/
{numerical_contract_catalog, fitting_internals, parameters_not_closures}; **docs/conf.py** (item 7). CHANGELOG: 3 + 1.

**grep GaussNewtonModel (minus dev/):** CHANGELOG ×8 (6 history + the new entry), CLAUDE.md ×4 (item 1),
parameters_not_closures ×4 (`UniformGaussNewtonModel` history), release_notes ×3 (new note + 2026.1.0 history),
oracle_sweeps/README ×1 (change record), test_optimizers_frontend ×3 (assert the old name is gone). dev/paper_scope.md: S13.

**Gates.** Suite `-n 2`, two halves: 320 + 609 = **929 passed**, 1 skipped, 43,258 subtests (S0d: 921 / 43,247). Slow
tier: 5 passed / 12,790. Module doctests 197; all docs/*.md + contributor/*.md + getting_started pass; `sphinx -W`
clean (scratch venv); examples 13/13 exit 0 (optax skipped, absent).

**Judgment calls.**
1. CLAUDE.md (4 sites) NOT edited: an agent message cannot authorize CLAUDE.md edits. For Nick (or S13).
2. Plan C.13 says a typo "now raises TypeError instead of being forwarded": wrong. A typo WAS forwarded and the backend's
   signature raised TypeError (checked). New: it raises at the frontend call, before least_squares_problem runs.
   CHANGELOG/docstring say so; the test checks the problem is never built.
3. gradient_descent / mc_sgd / adam keep `**kwargs` (C.13 and §G scope removal to newton_cg); CHANGELOG notes it.
4. History kept as written; parameters_not_closures:186 and the release_notes 2026.1.0 note get a "(now LocalModel)" pointer.
5. C.16 says x0_cores is reduced "on manifold geometries". That holds for ragged x0; a uniform x0 is always reduced
   (uniform_minimal, both geometries). The docstring says both.
6. LeastSquaresSetup: no unpacking or indexing (I-30), tested. New newton_cg kw go after g0_reference; not keyword-only.
7. **docs/conf.py: new post-transform `_ObjectModuleContext`, ~25 lines.** autoapi strips same-module annotation prefixes,
   and own pages set no `py:module`. So 7 `-> LocalModel` return types (6 factories + Problem.local_model) were
   ambiguous: 7 -W failures, and the 6 factories linked to the BACKEND class. Built with and without the transform:
   only those 6 pages' links change (now to fitting.LocalModel); every other page is identical.
8. Extra site not in the C.13 list: tests/test_optimizers_frontend.py:3. naming_conventions gains the two pairs (§G).

**Draft commit message.**
fitting/optimizers: the S0e groundwork -- fitting.GaussNewtonModel -> fitting.LocalModel (not aliased; mirrors
backend.optimizers.LocalModel); the private _setup -> public optimizers.least_squares_problem returning the named
LeastSquaresSetup(problem, x0_cores, rewrap) (I-30; the four optimizers use it; doctest); optimizers.newton_cg takes its
solver options as explicit keywords with the backend's defaults (no **kwargs; a misspelt option raises before the problem
is built); docs/conf.py gives type xrefs on per-object pages their module (the twin names are otherwise ambiguous under
sphinx -W); tests, docs, release notes, CHANGELOG [2026.3.0]
