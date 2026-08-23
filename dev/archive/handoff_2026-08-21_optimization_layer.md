# Archived 2026-08-22: the optimization-layer restructuring thread, as it stood in dev/HANDOFF.md before the whole-library review

## Newest thread — the optimization-layer restructuring (2026-08-21)

**MERGED into local `main`, NOT pushed, NOT released.** The branch
`optimization-layer-value-typed` is pushed and is the reviewable unit; `main` carries the merge plus the
`release: 2026.2.0` commit and is **15 commits ahead of `origin/main`**. Nick is doing an in-depth review
before cutting the release (2026-08-22) — so nothing on `main` has left this machine, and there is no
tag. Thirteen commits on the branch, every gate green at each: full suite **751 passed / 42,011 subtests**, 192 module doctests, doc pages clean,
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
scripts byte-identical against the pre-refactor tree. They were **not in CI**, which is how a refactor of
the whole optimization layer could land without one of them being executed; `tests.yaml` now has an
`examples` job that runs all fourteen. It is a second job with no `needs:`, so it runs in parallel with
the test matrix and takes about as long (~5-6 min locally), costing compute minutes rather than waiting.

Three things the review found and we chose not to fix — all in
`docs/contributor/deferred_and_rejected.md`: a construction-time guard for "parameter is not a field"
(the design's one sharp edge — see *Honest limits* in the decision record), a kind/geometry rank-pairing
guard, and `d=1` on the uniform layer (pre-existing, in `ut3_svd`).

### Next — release prep is DONE; the remaining steps are one-way

Nick's in-depth review comes first (2026-08-22). Everything below it is already done and verified:

- `main` = branch merge + `release: 2026.2.0`. Version bumped in **both** places it lives
  (`pyproject.toml` and the fallback in `t3toolbox/__init__.py` — the second is the easy one to miss);
  CHANGELOG `[Unreleased]` retitled `[2026.2.0] — 2026-08-22`. `docs/release_notes.md` needed no edit:
  it pulls the changelog through its `{include}`, and its upgrade section is already titled
  "Upgrading from 2026.1.0".
- Verified against the **built artifact**, per `dev/archive/release_plan_2026-07-13.md` REL-2:
  `twine check` PASSED on both; wheel is `t3toolbox` (53) + dist-info (5) with no strays; a fresh
  numpy-only venv fits to 9.6e-10 ragged / 4.5e-11 uniform with jax genuinely absent and `use_jit=True`
  raising cleanly; a fresh `[jax]` venv (jax 0.10.2 / numpy 2.4.6) runs `getting_started` 81/81.
  The numpy-only leg matters more than usual: CI always installs jax, so it is exercised only here, and
  this release adds code with jax-absent branches.
- Final gate on merged `main`: 755 passed / 42,011 subtests, 192 module doctests, quickstart and every
  doc page clean.

What remains is irreversible and deliberately not done — a PyPI version cannot be reused:

    git push origin main
    git tag -a v2026.2.0 -m "..." && git push origin v2026.2.0     # triggers TestPyPI then PyPI

REL-3 (trusted publishing) was configured for 2026.1.0, so there are no clicks this time.

**Push `main` before tagging.** The new `examples` CI job has never run on a real runner — the workflow
triggers on push to `main`, so that push is its first execution. Better to see it red there than during
a release.
3. **Follow-ups** are logged in `docs/contributor/deferred_and_rejected.md`: stacked optimization, the
   seven open-coded sharing normalizations in the SVD layer, a mixin for the repeated uniform-geometry
   methods, and jitting the outer Newton loop (its prerequisite — a pytree `LocalModel` — now exists;
   what it fights is the host-side `callback` display).

### Two verification lessons worth keeping

- **Breadth is not coverage when the cases share a degeneracy.** A frame-shape derivation passed 21
  structures including stacks, and was wrong; all 21 had `nD == nU`.
- **Compare invariants, not representations.** An optimizer diff showed 1.75 absolute and looked like a
  regression; it was comparing gauge-dependent cores (`U → UQ`, `G → Qᵀ G` leaves the tensor unchanged).

