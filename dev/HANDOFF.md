# T3Toolbox — current handoff

_Updated 2026-10-29, at the 2026.3.0 release._ The previous handoff (the exact-Hessian thread through S13, the
2026-08-22 whole-library review thread, the 2026.2 recon backlog) is archived verbatim at
`dev/archive/handoff_2026-10-29_pre-2026.3.0.md`; older ones are `dev/archive/handoff_2026-08-21_optimization_layer.md`,
`handoff_2026-08-20_pre-2026.1.0.md` and `handoff_2026-07-12_1.0_complete.md`. What each release contains is the
CHANGELOG; why it is the way it is, `docs/`. This file is where-we-are + what's next, nothing else.

## Where we are

**2026.3.0 SHIPPED to PyPI (2026-10-29)** — tag `v2026.3.0` at `ad578f6` (rc1 dry-run on TestPyPI first), published
via trusted publishing, verified from clean venvs (numpy-only: import, GN + exact fits, the jax-absent warn-and-run;
`[jax]`: the quickstart 81/81 against the installed package). The GitHub Release object is created by Nick (the session
cannot create releases). 2026.2.0 shipped 2026-08-25, 2026.1.0 2026-08-20, 2026.0.0 2026-07-13. Release prep is done
(2026-10-29): the three version sites bumped, the CHANGELOG stamped `[2026.3.0] — 2026-10-29`, the release notes'
"Upgrading from 2026.2.0" complete, the wheel smoke test passed (`twine check` PASSED; the wheel holds the 53 package
modules and nothing else; a numpy-only py3.11 venv runs the quickstart, an exact-Hessian fit to 1.5e-12 and the
jax-absent `use_jit` warning; a `[jax]` venv runs `getting_started.rst` 81/81 against the installed wheel), and the
release gates are green on the final tree: the full suite (1,170 tests / 46,364 subtests) and the serial slow tier (66 /
13,173) on py3.13, 225 module doctests, `getting_started.rst` and every doc page on all three environments below, and
`sphinx -W` (renders 2026.3.0); the floor's full suite was last run at S13 (since then the library changed only in
docstrings and annotations). What remains: the `v2026.3.0-rc1` tag → TestPyPI install check → the `v2026.3.0` tag →
approve the `pypi` environment → a clean-venv check against real PyPI → the GitHub Release from the CHANGELOG section.

**The exact-Hessian arc (2026-10-25 → 29) — DONE, shipped in 2026.3.0.** `newton_cg(..., hessian='exact')`: the
Gauss-Newton term plus the Weingarten curvature under a Steihaug–Toint trust region, for every sampling kind on every
geometry × representation, with the zero-tensor Newton reference, the operand-scaled gauge check, clean uniform
padding and recurrence jet kernels. Slices S0c–S13, the close-out review S13b, the math note (S15a, final pass S15b with
every T4S citation verified against arXiv v2), each with an implementer and an independent-reviewer record; S14 updated
T3Polynomial downstream (in its own repo). Rulings on 2026-10-29: the reorganization R **cancelled**; **I-39** (the S3
memory gate: the two-hole's TT-assembly peak 1.25–1.61× the Gauss-Newton one) accepted; **I-40** (the operand-scaled
gauge check's finite-difference limit) accepted as documented. The durable record is `docs/fitting_and_optimization.md`
§4.12, `docs/contributor/curvature_internals.md`, `docs/contributor/verification_records.md`, the ledger
`docs/contributor/deferred_and_rejected.md` and the math note `docs/exact_hessian_two_hole.tex` (+pdf). The build record
is archived: the plan `dev/archive/riemannian_newton_curvature_impl_plan_2026-10-29_shipped.md` (its design plan
`..._plan_2026-10-27_implemented.md`), the slice records and the deferred table in
`dev/archive/curvature_impl_review_2026-10-29/slices/` (`DEFERRED.md`), the experiments in `.../shared_exp/`, the S0b
downstream report in `.../s0b/`, and the cancelled R scope `dev/archive/reorg_scope_2026-10-29_cancelled.md`.

## Release gates (run all before a tag)

- **Full suite:** `PYTHONPATH=$PWD <env-python> -m pytest tests/ -q -n auto` (per-test seeded, order-independent).
- **Slow tier — REQUIRED, run SERIALLY:** `T3TOOLBOX_SLOW_TESTS=1 PYTHONPATH=$PWD <env-python> -m pytest
  tests/test_oracle_sweep.py tests/backend/test_uniform_curvature.py -q -p no:cacheprovider` (the full optimizer matrix,
  the exact sweep's seeds × layouts × order 3, the full-length uniform exact runs).
- **Doctests + docs:** `pytest --doctest-modules t3toolbox`; `docs/getting_started.rst` and every `docs/*.md` +
  `docs/contributor/*.md` page (except `doctest_style.md`); `cd docs && sphinx-build -W -q -b html . <out>`.
- **Three local environments:** (1) the system py3.13 / numpy 2.5 / jax 0.12 — runs AHEAD of anything CI sees;
  (2) the **CI current-leg twin py3.11 / numpy 2.4 / jax 0.10.2** (what the leg's `numpy>=2` + `jax` resolve to on
  py3.11; BLAS-dependent stop reasons and step counts have differed from (1)); (3) the **floor py3.11 / numpy 1.26 /
  jax 0.4.30** — the full suite, plus `docs/chunking.md` by hand (the CI floor leg skips the doc doctests when its test
  step fails, and the measured chunk costs depend on the jax release). Venvs do not survive the container: `uv venv -p
  3.11 <dir>` then `uv pip install "numpy<2" jax==0.4.30 jaxlib==0.4.30 pytest pytest-subtests pytest-xdist` (floor) or
  `"numpy>=2,<2.5" jax==0.10.2 ...` (twin).
- **Wheel smoke** (`dev/archive/release_plan_2026-07-13.md` REL-2): build, `twine check`, the wheel's file list, a
  numpy-only venv, a `[jax]` venv running `getting_started.rst` against the installed package. A release is: bump
  `pyproject.toml` / `CITATION.cff` / the `__init__` fallback, stamp the CHANGELOG, smoke, gates, rc tag → TestPyPI →
  tag → approve `pypi` → clean-venv check → GitHub Release (REL-1/REL-3 are one-time work, done).

## Next up — the post-2026.3.0 backlog

**From the exact-Hessian arc** (each also in the ledger `docs/contributor/deferred_and_rejected.md`):
- **float32 end to end.** A float32 fit comes back float64: 38 default-dtype constructors in eight backend modules sit
  upstream of the jet kernels (which preserve float32 since 2026.3.0). A ready follow-up patch is
  `dev/archive/curvature_impl_review_2026-10-29/slices/float32_end_to_end_followup.patch` (checked 2026-10-29 over the
  six kinds × both layouts × manifold / corewise / shared × both Hessians; float64 bit for bit). Its own slice: each scan
  init must take the dtype of ALL its scan's operands, which wants a mixed-dtype jax test per site.
- **Legacy RNG** (backlog from the cancelled R): move the random constructors off the global `np.random` stream to
  explicit `Generator` / jax-key arguments (the E2 and O2 flakes were this).
- **I-36 — the Newton-schedule A/B** (Gauss-Newton vs exact as the continuation default; Nick leans exact with the warm
  start). Empirical, after implementation; the per-call override hooks exist (and T3Polynomial's per-level
  `newton_kwargs` schedule). Inputs so far: `dev/archive/curvature_impl_review_2026-10-29/shared_exp/shared_vs_unshared_exact.md`
  and `.../nonlinearity_vs_tied_factors.md` (numbers in `verification_records.md`); missing are GPU timings and the
  Hessian-apply cost against the per-iteration retraction and frame build.
- **The S9 tied-projection clip.** The uniform tied *projection* (`ufv_share_tucker_variations`, Gauss-Newton code)
  clips at the padded dims and does not mask padded `s_g` levels (17 % off ragged at a near-singular point the
  equivalence contract exempts; not robust to garbage in a hand-built companion). Needs its own bit-identity run.
- **The leftover-chunk overshoot.** On current XLA a remainder chunk is allocated on top of the full one, so
  `max_chunk_size_within` can overshoot its budget ~2×; `_wchunked_sum` doubles the peak at a near-full remainder.
- **The S7b follow-ups:** carry the operand scale through `normalized` / `sum_tangents` / `unstack_tangents`; a
  `UT3Tangent.__truediv__`; a faster `corewise_stack_norm` (safe-mode `±` is 4–6× slower since S7b; the optimizers run
  on `.data` and are unaffected).
- Smaller: `fv_share_tucker_variations`'s mixed-backend inference; optional `_blm.hvp` routing for the frontend exact
  `hessian(p)`; the `docs/conf.py` xref variant for the dead `:py:func:` links; the six
  `u_*_corewise_transpose_vs_ragged` oracle-harness checks (excluded since Phase D, repairable); `docs/pad_safe_svd.tex`
  still cites the pre-archive `dev/review_2026-08-22/` path (fix + PDF rebuild). Downstream, not ours: T3Polynomial's
  `x01` script has a pre-existing `NameError` (`K`), noted at S14.

**Carried forward:**
- The Goal-1 **`fit(...)` facade** — auto geometry / optimizer / ranks / `x0` + rank continuation and validation
  ("standard user, no fiddling"); the largest single item, deferred. The Newton-schedule hooks are its seam.
- The **Grasedyck–Kramer `SingularValueRegularizer`** and weighted `+` / `−` / scale / `⊙`; **base-point-as-tangent**
  as a public op (promote the regularizer's private `_manifold_point_tangent`); **sharded fitting**
  (`dev/sharded_fitting_plan.md`); **data-shaped residual weighting** (deferred 2026-09-03 — pick up from
  `dev/data_shaped_residual_weight_reviews.md`, not the plan's v1).
- The 2026.2 recon backlog, still open at last look (re-verify before picking one up; the archived handoff has the
  detail): frontend access to `edge_condition_numbers` / `t3_tie_tucker_factors`; uniform `share`; `UT3Weights` /
  `UT3FrameWeights` `reverse` / `stack`; `make_newton_display`'s backend uniform validation packing; the frontend-only
  `chunk_size='auto'` derivation; the `shared_geometry.py` signature pass; mechanical signature items (17 `regularizer:
  typ.Any`, the `(order+1)+W` tuple comma, `T3Weights`' `len=d+1`); `t3_stack`'s docstring; `core_shapes` vs
  `get_core_shapes`; the R2-12 / R2-13 review leftovers; a default-path doctest pass; trimming `test_dispatch` jit time.
- The **toolbox reference paper** (independent of releases): `dev/paper_scope.md`. The standalone grouped-contractions
  library: PARKED. Zenodo DOI (Nick, later); the dead `gh-pages` branch (optional). Sweep candidates: the DONE plans
  `dev/newton_display_plan.md` and `dev/per_mode_weighting_plan.md`.

## Standing open questions

- **`dev/OPEN_QUESTION_extension_surface.md`** — what extension the fitting layer is meant to support (opened
  2026-08-21; 2026.3.0 added the sixth `SamplingKind` operation and the four exact-mode geometry members to its
  surface). Standing, not a thread: it stays in `dev/` until answered.

## Standing constraints

Project-wide conventions live in **CLAUDE.md**; contributor conventions and decision records in the rendered
**Contributor guide** (`docs/contributor/`). Operational ones worth repeating: the docs build stays at **zero
warnings** (`sphinx -W` in CI); doctest outputs are **run-and-pasted, never hand-written**, and CI runs every
`docs/*.md` / `docs/contributor/*.md` page; a markdown doctest needs a **blank line before the closing fence**; a doctest
must not pin a stop reason or step count at the resolution floor (BLAS-dependent — assert a fact with a wide margin);
subTest params must be primitives under xdist.
