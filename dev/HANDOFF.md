# T3Toolbox — current handoff

_Updated 2026-08-20, just after the 2026.1.0 release. The per-thread history of everything that went
into 2026.0.0 and 2026.1.0 is archived at `dev/archive/handoff_2026-08-20_pre-2026.1.0.md` (and, for
1.0, `dev/archive/handoff_2026-07-12_1.0_complete.md`); what each release contains is the CHANGELOG,
and why it is the way it is is `docs/`. This file is where-we-are + what's next, nothing else._

## Newest thread — the eager-jax scan-body sweep (2026-08-21)

**Tiers 1 and 2 are done and pushed** (`f0099c4c`, `48cdd13c`); **tier 3 is planned but not started**
and wants its own session.

The defect: `lax.scan` keys its trace/compile cache on the IDENTITY of the body it is handed, so a
body defined inside its caller is a fresh object every call — it re-traces, re-compiles, and leaves
LLVM JIT mappings behind until `use_jit=True` runs abort on Linux's `vm.max_map_count`. It is also
why the jit path measured ~40x SLOWER than numpy: nearly all of its per-iteration time was
recompilation. Found from a downstream consumer, diagnosed here.

On the `probe_derivatives` Newton-CG path: **compiles per Newton iteration 19 → 1**, mappings
877 → 294. `common.jax_map` also had to change — `jax.lax.map` builds a fresh wrapper lambda per
call and so can never hit the cache however stable the body is.

- **Durable principles**: `docs/contributor/scan_body_principles.md` (in the Contributor guide).
- **Catalogue** of all 54 sites, with the measurement recipe: `dev/scan_body_sweep.md`.
- **Plan**, incl. the full tier-3 specification: `dev/scan_body_plan.md`.

**Tier 3 — `_cg_solve`, the last per-iteration recompile.** Not a hoist; a small refactor Nick has
wanted for a while. `cond` captures `tol2`, which changes every Newton iteration, and a stable body
reading a changed Python value silently gets the cached jaxpr with the OLD value — so this is a
correctness constraint, not a performance preference: `tol2`/`maxiter` must be carried in the loop
state. `body` captures `hvp`, a bound method of a per-iteration `LocalModel`. An `lru_cache` on it is
recorded as REJECTED (never hits, and pins every `LocalModel`'s arrays). All viable routes need
`LocalModel` to become a jax pytree — prove that first. Templates already in-house: `SamplingKind`'s
`identity`-tuple `__eq__`/`__hash__`, `common.ValueHashedMasks`, and `UniformGaussNewtonModel`'s
pytree registration. Target agreed with Nick: **once per fitting run is acceptable**. Oracle:
`tests/backend/test_optimizers.py` already covers eager-vs-jit agreement.

Two known non-defects left deliberately: `optimizers.py:395` jits a fresh `step` per `mc_sgd`/`adam`
*call* (one compile per optimizer invocation, which coincides with a rank change — accepted), and
`xmap` bodies dispatch per-element rather than per-operation, which differs from the old whole-tree
rule only for mixed numpy/jax sequences that nothing in the library produces (documented in the
principles page).

## Where we are — 2026.1.0 SHIPPED ✅

**Live on PyPI, 2026-08-20 — `pip install t3toolbox`** (2026.0.0 was 2026-07-13). Tag `v2026.1.0`,
published by the release workflow via OIDC trusted publishing after a `v2026.1.0-rc1` TestPyPI dry run.

**Verified against the published artifact**, not a local build: a clean venv with `pip install
t3toolbox` (numpy only, jax genuinely absent — the only test the jax-optional claim ever gets, since
CI always has jax) runs arithmetic, T3-SVD, the three sampling ops, a Newton-CG manifold fit to
1.8e-12, shared factors, weighting and the uniform layer; a clean venv with `pip install
"t3toolbox[jax]"` runs the full `getting_started.rst` doctest 81/81.

Gates at the tag: **726 tests / 41,976 subtests**, 193 module doctests, doc doctests on both numpy
generations, compat floor green, `sphinx -W` clean, wheel with no strays (`t3toolbox` + dist-info only).

**What 2026.1.0 added** (detail in the CHANGELOG): shared Tucker factors (SF-T3) ragged + uniform; the
grouped-einsum interpreter `contract` replacing the ~104 named contractions (breaking); the weighted
edge-layer, ragged + uniform; regularization; per-mode residual weighting; the Newton-CG diagnostic
display; chunking; the recurrence/scan jets; and rank continuation for shared factors.

Loose ends: **Zenodo DOI** still deferred (Nick, later); the dead `gh-pages` branch can be deleted
(optional). GitHub Releases exist for both tags (`v2026.0.0` 2026-07-13, `v2026.1.0` 2026-08-20) --
an older handoff claimed 2026.0.0's was missing; it was not.

## Standing open questions

**None.**

## Next up — the 2026.2 backlog

Most of this came out of the **convention reconnaissance run 2026-08-20** (eight parallel audits of the
post-2026.0.0 work against the project's own rules: the razor, uniform parity, dispatch, stacking, safe
mode, signature style, naming, plus a catch-all). The findings that were correctness bugs were fixed
before the release; what follows is what was deliberately left.

**Backend/frontend razor — additive gaps, none breaking**
- `utv_project_ut3_onto_tangent_space` has no `shared_data=`, though every ragged twin and every
  sibling uniform op does — so a raw-`.data` user doing a shared uniform `project_ambient` (hence
  `transport`) must compose the tie by hand. One-line fix.
- `chunk_size` is not threaded through `ut3_probe_corewise_derivatives_transpose` (hard-coded 100).
  This is the wrong side to drop it on: chunking only engages on uniform+jax, so a ragged `chunk_size`
  is inert sugar and a missing uniform one is real.
- No frontend access to `edge_condition_numbers` (the continuation diagnostic that tells "guard fired"
  from "structure maximal") or to `t3_tie_tucker_factors` — five error messages point users at the
  backend for the latter.
- `from_t3svd` / `from_ut3svd` discard the SVD'd train, so weighting a point takes **two** SVDs; the
  documented workaround is the tell. A backend constructor returning `(train, weights)` fixes it.
  *(Hit this personally during the release smoke test — it and the `sharing=` variant are the two ways
  that API is easy to misuse.)*
- `make_newton_display` silently mis-scores uniform **validation** data unless the caller packs it
  first; the training sample is packed internally, so there is no cue.
- The whole `chunk_size='auto'` derivation is frontend-only (five separate pieces of knowledge).

**Uniform parity**
- No uniform `share` — the rank-changing quasi-optimal projection. The cheap tie repair now exists
  (`ut3_tie_tucker_factors`, added 2026.1.0), which is what drift needs; `share` is a bigger job on the
  masked layer and nothing yet needs it.
- `UT3Weights` / `UT3FrameWeights` lack `reverse` / `stack` (Nick: deferred deliberately, there was a
  subtlety; low priority). `docs/weighting.md`'s "the same operations" claim should be trimmed or the
  methods added.

**Signature shape comments** (the trailing `#` comment IS the type contract, so these are real)
- The standard-form jet wrappers lost their contracts beside their fully-documented `_trs` twins:
  `compute_tau_tilde_jets` / `compute_sigma_tilde_jets` are public, take five array args, and carry
  **no annotations and no shape comments**; both `assemble_*_variation_jets` are the same story, and
  two declare `-> NDArray` where the ragged / `chunk_size=None` path returns a `len=d` tuple.
- `shared_geometry.py`'s ops surface never got a signature pass: 9 methods with unannotated args, 9
  with no return annotation, `shared_data=` wanting `Optional[SharedFrameData]` at four sites.
- Mechanical: 16 sites write `(order+1)+W+…` without the tuple comma; 18 `regularizer: typ.Any` should
  name `Regularizer`; the ragged `T3Weights` family comments omit that `tt_weights` is `len=d+1` while
  `tucker_weights` is `len=d` — precisely the inexpressible part.

**Testing**
- The **uniform weighted layer has zero positive jit coverage** in `test_dispatch` (it appears only in
  the negative traced-mask test). All of it was verified working by hand during the audit — this is a
  missing test, not a bug.
- The uniform **tied-tangent** ops are tested unstacked only (no `C`/`K` matrix), while the ragged twin
  has the K-over-C broadcast test. The companion carries `C` on every array, so the stacked path is
  real and untested.
- ~25 new public ops have no `test_dispatch` entry (full list in the archived recon notes).

**Small**
- `get_backend` raises a bare `NameError: name 'jnp' is not defined` when jax is requested but absent.
  Found during the release smoke test — it is exactly the user who pip-installed without the `[jax]`
  extra, so the message should say so.
- `t3_stack` has no docstring at all; `bfit_pd` (a lazy-import shim in `optimizers.py`) should be
  `_bfit_pd`; the `check_{fv,ufv,fw,ufw}_pair` export policy is inconsistent (one of four exported).

**Carried forward from before**
- The Goal-1 **`fit(...)` facade** — auto geometry/optimizer/ranks/`x0` + rank continuation and
  validation ("standard user, no fiddling"). The largest single item.
- **Grasedyck–Kramer `SingularValueRegularizer`** and weighted `+` / `−` / scale / `⊙` operations —
  both layers now have every primitive these need.
- **Base-point-as-tangent** as a public op (representing `X` as a gauged tangent in its own tangent
  space); the regularizer's private `_manifold_point_tangent` is the impl to promote.
- **Sharded fitting** — the `shard_map` boundary + `psum` in the optimizer; design and the genuine
  decisions are in `dev/sharded_fitting_plan.md`.
- Default-path **doctest pass** for undocumented public functions; `core_shapes` (property, strips
  stack) vs `get_core_shapes` (static, includes stack) inconsistency; per-test seeding → `pytest -n
  auto`; trimming `test_dispatch` jit time.
- The **toolbox reference paper** (independent of releases): scope and curation in `dev/paper_scope.md`.
- **Standalone grouped-contractions library** — PARKED; revisit when a grouped contraction is wanted in
  another project (the einx comparison lives in the research repo).

## Standing constraints

The durable rules live where they belong: project-wide conventions and gotchas in **CLAUDE.md**;
contributor-facing conventions and decision records in the rendered **Contributor guide**
(`docs/contributor/`). Four operational ones worth repeating:

- the docs build must stay at **zero warnings** (`sphinx -W` in CI);
- doctest outputs are **run-and-pasted, never hand-written** — and CI now runs them over **every**
  `docs/*.md` and `docs/contributor/*.md` page, not just the module docstrings and the quickstart;
- a markdown doctest needs a **blank line before the closing fence**, or doctest swallows the fence
  into the expected output (this bites once per page);
- release checklist of record: `dev/archive/release_plan_2026-07-13.md` — 2026.1.0 followed it, and
  REL-1/REL-3 (metadata, the workflow, trusted publishing, the GitHub environments) are one-time work
  that is already done. A release is now: bump the three version sites (`pyproject.toml`,
  `CITATION.cff`, `__init__`'s PYTHONPATH fallback), stamp the CHANGELOG, wheel smoke test, gates, tag
  → approve the `pypi` environment → verify from a clean venv → GitHub Release.
