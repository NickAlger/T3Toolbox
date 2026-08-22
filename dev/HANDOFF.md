# T3Toolbox — current handoff

_Updated 2026-08-21._ _Previously updated 2026-08-20, just after the 2026.1.0 release. The per-thread history of everything that went
into 2026.0.0 and 2026.1.0 is archived at `dev/archive/handoff_2026-08-20_pre-2026.1.0.md` (and, for
1.0, `dev/archive/handoff_2026-07-12_1.0_complete.md`); what each release contains is the CHANGELOG,
and why it is the way it is is `docs/`. This file is where-we-are + what's next, nothing else._

## Newest thread — the optimization-layer restructuring (2026-08-21)

**On branch `optimization-layer-value-typed`, NOT pushed, NOT released.** Nine commits, every gate
green at each: full suite **751 passed / 42,011 subtests**, 192 module doctests, doc pages clean,
`sphinx -W` clean.

    8ef7c118  backend geometries become value-typed classes
    b38b0439  give the frame conventions one home each
    a0b268bd  reject a regularized fit of a STACKED point
    927cfc64  reject a STACKED point in the optimizers
    3a31dd2f  SamplingKind becomes value-typed classes
    981a4b46  move the jit boundary -- 1 compile per Newton iteration -> 0
    843de628  one GaussNewtonModel for both representations
    604149f2  docs: record the restructuring, close the sweep
    cb36ef9f  act on the four-lane review

### How this started, and what it really was

It began as tier 3 of the scan-body sweep (archived:
`dev/archive/scan_body_plan_2026-08-21_complete.md`), which had got the `probe_derivatives` Newton-CG
path from 19 compiles per Newton iteration to 1. The last one was `_cg_solve`, and the plan said to
defunctionalize its `while_loop` body.

That turned out to be the wrong altitude. The real defect was that **every axis object in the layer was
a bag of closures**, and every jax cache keys on identity — so a rebuilt-but-identical geometry or kind
was always a new cache key, which is the normal case in a fitting loop. The same workaround had been
independently invented three times (`SamplingKind.identity`, `UniformGaussNewtonModel`'s four shadow
fields, a memoized factory in the chunked assembly). Fixing the cause instead: **parameters are fields,
behaviour is methods**, value identity from `common.ValueHashedFields`.

**The durable record is [`docs/contributor/parameters_not_closures.md`](../docs/contributor/parameters_not_closures.md)** —
rationale, the sharpened backend rule, measurements, rejected alternatives. Read that, not this.

### Two real bugs found on the way, both live in shipped 2026.1.0

- **A derived kind silently reused its parent's compiled program.** `dc.replace` copied the `identity`
  tuple, so `dc.replace(APPLY, forward=<other math>) == APPLY`; jit returned 115.302888 where eager
  gave 28.825722. Unrepresentable now (a variant is a subclass).
- **A regularized fit of a stacked point mis-weighted silently.** The misfit keeps the stack `C`, every
  regularizer scalar collapses it, so the whole-stack total was added to each element. Now raises.
  Separately: no optimizer ever supported stacked points (`float()` on a shape-`C` objective) — that
  now raises at the entry instead of from inside the loop.

### Results

| | before | after |
|---|---|---|
| compiles / Newton iteration (uniform `probe_derivatives`) | 1 | **0** |
| `mc_sgd` / `adam` step kernel | one compile per optimizer *call* | once per shape signature |
| CG tolerance staleness | freshness load-bearing for correctness | unrepresentable |
| `backend/optimizers.py` T3-specific imports | 3 | **0** |

Numerics were held to bit-identical against the pre-refactor tree throughout — geometry surfaces
(124,692 values), corewise transposes (24,882), kind surfaces (19,426), frontend model surfaces
(8,160) — with two documented exceptions, both jitted `newton_cg`, agreeing to 1e-12 / 1e-15 relative
on the fitted **tensor** (XLA fuses a larger compiled region differently).

### Reviewed (2026-08-21)

Four independent review lanes plus an `examples/` sweep, each required to ship a reproduction with its
finding. Result: **two silent wrong answers**, both fixed in `cb36ef9f`.

- `SharedGeometry.__eq__`/`__hash__` keyed on a hardcoded class name, so a subclass collided with the
  shipped wrapper in the jit cache. **Live in 2026.1.0**, and exactly the failure the decision record
  called unrepresentable — the record overclaimed, and now says so.
- `_weight_matrix` aliased the caller's array, so a weight sweep reusing one buffer desynced the cache
  key from the compiled program. Sharpened by this refactor (the weight became part of the value
  identity); now copied and frozen.

Plus seven regressions of mine, all documentation or ergonomics: a second `__all__` silently rebinding
the first, `has_block_sumsq` defaulting the wrong way, `weight=` renamed gratuitously, six removed
uniform builders documented nowhere, three new geometry protocol members undocumented, a "supply the
five operations" recipe that needed nine and pointed at private bases, and stale cross-references.

Positive evidence worth keeping: 250 untested combinations run against independent oracles found
nothing; compile-once re-measured across 48 configurations, not the 1 I had checked; all 14 `examples/`
scripts byte-identical against the pre-refactor tree — **they are not in CI**, which is a real gap.

Three things the review found and we chose not to fix — all in
`docs/contributor/deferred_and_rejected.md`: a construction-time guard for "parameter is not a field"
(the design's one sharp edge — see *Honest limits* in the decision record), a kind/geometry rank-pairing
guard, and `d=1` on the uniform layer (pre-existing, in `ut3_svd`).

### Next

1. **Nick reviews the branch**, then merge to `main`. Nothing is pushed.
2. **Release**: bump `pyproject.toml` (`2026.2.0` — the scheme has no major slot, and 2026.1.0 itself
   shipped breaking changes), retitle the CHANGELOG's `[Unreleased]`, follow
   `dev/archive/release_plan_2026-07-13.md`. The upgrade notes are already written in
   `docs/release_notes.md`.
3. **Follow-ups** are logged in `docs/contributor/deferred_and_rejected.md`: stacked optimization, the
   seven open-coded sharing normalizations in the SVD layer, a mixin for the repeated uniform-geometry
   methods, and jitting the outer Newton loop (its prerequisite — a pytree `LocalModel` — now exists;
   what it fights is the host-side `callback` display).

### Two verification lessons worth keeping

- **Breadth is not coverage when the cases share a degeneracy.** A frame-shape derivation passed 21
  structures including stacks, and was wrong; all 21 had `nD == nU`.
- **Compare invariants, not representations.** An optimizer diff showed 1.75 absolute and looked like a
  regression; it was comparing gauge-dependent cores (`U → UQ`, `G → Qᵀ G` leaves the tensor unchanged).

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
