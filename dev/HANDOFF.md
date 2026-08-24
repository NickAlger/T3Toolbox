# T3Toolbox — current handoff

_Updated 2026-08-22._ _Previously updated 2026-08-21 and 2026-08-20, just after the 2026.1.0 release. The per-thread history of everything that went
into 2026.0.0 and 2026.1.0 is archived at `dev/archive/handoff_2026-08-20_pre-2026.1.0.md` (and, for
1.0, `dev/archive/handoff_2026-07-12_1.0_complete.md`); what each release contains is the CHANGELOG,
and why it is the way it is is `docs/`. This file is where-we-are + what's next, nothing else._

## Newest thread — the whole-library pre-release review (2026-08-22)

**Where it stands:** `main` carries the 2026.2.0 release commit, the review commits, and the S1b
pad-safe-SVD slices (all **pushed 2026-08-23**, CI green), plus the **review-continuation commits of
2026-08-24** (below; push pending their gate); **not tagged**. Every S and C cluster is fixed; the E
list is being worked through in clusters. Before tagging: Nick's one-way steps (wheel `twine check`,
the numpy-only venv smoke, tag).

**Review continuation (2026-08-24) — the E list, in clusters.** Triage: the E findings grouped into
six work clusters (session record; `findings_compact.md` is the frozen Phase-A list). Done and
committed: **cluster 6** (the design rulings: Armijo exhaustion `on_line_search_failure='stop'|'accept'`
with honest `alpha` + `ls_failed`; per-element ragged geometry `inner`/`point_norm_sq`;
`sum_over_probes` default parity; `from_ut3svd(n=, r=)`; `shared_data=` on the uniform projection; GD
documented eager-only), **cluster 1** (obscure errors -> structural guards: str-leaf tree recursion,
`tree_zip` mismatch, `truncated_svd` min>max, uniform scalar-only `*` / layout-checked `stack` /
host-numpy `minimal_ranks`, weighted rank/stack consistency, `COREWISE.retract` frame-kind guard,
`stack_tangents` mixed-K, `SharedGeometry` layer checks), **cluster 2** (`__rmul__`/`__truediv__`;
`use_orthogonalization` parity; `check_fw_pair` export; uniform `sum_stack(axis=)` DEFERRED +
documented in the ledger; the kinds' unread protocol args documented), plus the **citation policy**
(CITATION.cff cites the software; T4S under `references`). Lesson recorded: the R3-6 guard's blast
radius included a test that deliberately swept min>max combos -- ~4000 failing subTests turned the
suite into a 30-min reporting crawl (diagnosed with a SIGABRT faulthandler dump); targeted `-k` runs
had deselected exactly that test, so a guard-adding commit should run the full file of any test that
sweeps the guarded parameter.

**Cluster 3 — DONE (2026-08-24), the equality + jit/value-hash contract.** Nick's ruling (after an
ecosystem survey, the T3Frame `eq=False` archaeology, and a greenfield analysis): **`==` raises** a
directive TypeError (identity-True fast path) and **`hash` raises** on ALL twelve runtime
array-carrying classes -- there is no cross-library convention to lean on (numpy elementwise,
torch/TF identity-or-elementwise, sympy structural), so the user must say what they mean. The named
checks: **`allclose`** (mathematical, per-stack-element bool array -- `C` for trains/frames/weights,
`K+C` for tangents/variations; formed as `norm(A - B) <= atol + rtol*max(norm A, norm B)` via our own
subtraction/orthogonalized norm, stable when A ~ B, which optimization makes routine; default rtol =
`safety.comparison_rtol`, jax-aware) and **`corewise_equal`** (representational, bitwise, single
bool, never raises). Frames: `allclose` = same BASE POINT (gauge-invariant); the same-tangent-space
question stays `safety.frames_equal`. The T3Frame identity-`eq=False` was archaeology-confirmed
obsolete (it served the pre-S4 aux_data guard). Also in the cluster: the ValueHashed mixins'
hash/eq now derive from ONE content key (dtype-strict, length-safe; jax fields raise); the mask
holders and geometry/kind `from_point` sites store defensive READ-ONLY mask copies; `shared(...)`
keys identity + jit aux on the canonical partition (label-spelling-independent, subclass-preserving
round-trip); the `stack(unstack(x))`-under-jit E-items were verified already fixed (Phase C's
static-axes moveaxis) and pinned by a dispatch regression test. Commits `707653d8` (allclose family),
`6385de36` (the ==/hash flip), `e475449c` (3b-3e); new `tests/test_equality_semantics.py` +
`TestAuxKeyHygiene`; gate: full suite 849 passed / 29,294 subtests + module and doc-page doctests,
2026-08-24. CHANGELOG carries the breaking-change entry.

**Cluster 5 -- DONE (2026-08-24), the docstring/stale-text sweep** (taken before cluster 4 at Nick's
request). Three commits: `c6657acb` (the text sweep: every `dev/` path in shipped docstrings/comments
retargeted to its durable `docs/` counterpart per Nick's rule -- archive refs only while a thread is
open; retired letters/terms -- tv_operations' V/G -> K/C, `base-inner` -> `frame-inner` library-wide,
the 3b-*/uniform-fix build-slice tags dropped; T3Base/T3Variation -> real class names; dead
conditionals, rRi bond normalization, `jax.linear_transpose` claims trimmed to the adjoint identity
the tests actually enforce -- ruling: an adjoint-identity check suffices, and `test_probe_derivatives`
has real ones; api_reference's module-level names; the contract catalog's 'Sign-off questions'
replaced with the recorded resolutions), `cbd552c5` (code stragglers: `chunk_size` threaded through
both frontends' `probe_corewise_derivatives_transpose`, no-host-pull `'auto'` resolution +
`DEFAULT_CHUNK_SIZE`, jit-safe `compute_minimal_ranks`, a real oracle for the vacuous
`test_riemannian_gradient`, the `to_jax` doctest x64 leak), and the notation commit (see below).
Deferred, recorded here: R2-13's minor backend ergonomics (t3_inner_product list concat, ndarray-axis
`t3_sum`), R2-12's stacking malformed-tree sub-items (already in the backlog), and the no-test items
(R2-9, R8-10, R4-15) -> Phase D.

**The derivative-order letter ruling (R6-7, Nick 2026-08-24; recorded in
`docs/naming_conventions.md` §"Index letters"):** the scalar is spelled `order` (no letter); the
order/jet axis is lowercase `t` with `r`,`s` the binomial-split axes (the `trs` tensor); `K` is the
tangent stack ONLY. The math note `symmetric_probe_derivatives.tex` renamed to match: max order
$K \to m$ ($J^m$ jets), running $k \to t$, binomial $j \to r$ -- unifying the derivation sections
with the note's own trs/adjoint sections, which already used t/r/s; mode dummies stayed $j$ (a
$j \to r$ there would collide with rank subscripts $r_i$). Builds warning-free.

**Cluster 4 -- DONE (2026-08-24), safety internals.** One of its three queued items was already
closed (the `set_default_safety` validation + thread-scope docs landed with cluster 1; only its
Phase-D test remains). The two real items, both ruled by Nick:
- **H5-5** (commit `9f5555ad`): the uniform same-frame guard masks the frame supercores
  (`ufv_apply_frame_masks`) before `safety.frames_equal_or_skip` -- padding is don't-care, the
  tangent space depends only on real content. `safety.frames_equal` stays representation-agnostic;
  its docstring states the caller-passes-real-content contract (Nick's question "should safety.py
  mask?" -- answered as a layering argument: safety takes plain array trees; ragged data IS real
  content; the mask-supercore pairing is the uniform layer's knowledge).
- **H6-12**: `_inside_jax_trace` is the committed-array probe ALONE -- `jax.core.trace_state_clean`
  does not exist in jax 0.10 (every check on jax inputs paid a raise/catch), and the docstring now
  says what the probe delivers: jit-only detection; under grad/vmap with concrete operands the
  checks just run, harmlessly. **The vmap-over-UT3 investigation Nick requested** (his concern:
  batch axes landing before the mode axis): verified 2026-08-24 that a stacked UT3 cannot be
  vmap-ed with ANY in_axes -- axis 0 slices the mode axis d (validate rejects), axis 1 fails too
  because the rank masks are static aux with per-element stack axes and vmap cannot slice aux.
  Every route errors loudly at validation; nothing convention-violating comes out. Recorded in
  `docs/batching_and_stacking.md` §7 and `docs/contributor/uniform_pytree_composition.md`
  (repro: scratchpad `vmap_ut3_probe.py`). Batching uniform objects = the native C stack.

**Next (queued):** Phase D test hardening (incl. the R4-15 untested-name list and
`set_default_safety`'s test), the R2-12 deferred sub-items, and the 2026.2.0 tag steps
(twine check, numpy-only venv smoke, tag).

**What happened.** Nick asked for an in-depth review of the whole library (bugs first, doc/code mismatches
second) before cutting 2026.2.0, because the 2026.1/2.0 work had kept turning up pre-existing bugs. The
record is `dev/review_2026-08-22/` — read `findings.md` (the ledger: 186 findings from 19 lanes, clustered
into 14 silent-wrong-answer and 13 crash clusters, the inline verification verdicts, and the Phase C status
table mapping every cluster to its commit). Method, for next time: Phase A was one Workflow of 19 lanes
(10 region reads, 7 bug-class hunters, 2 oracle sweeps), every finding with a repro script; it cost ~5.4M
tokens, which made the planned adversarial-verification workflow unaffordable — verification was done
inline by re-running each lane's repro (nothing refuted; one finding, S14, was worse than claimed), and
fixes were made in the main session with the repro turned into the regression test. Budget lesson:
discovery is the only phase that needs the big fan-out; verification and fixing are cheaper in-session.

**Rulings Nick made (so they are not re-litigated):** exact products (`*`, `t3_mult`) keep the Kronecker
structure on boundary bonds, only truncating `t3m` canonicalizes them; a per-mode residual weight may carry
*more* rows than modes (continuation schemes), fewer is an error; `d = 1` is a supported *degenerate* case
(a vector); jax requested but absent → numpy plus a one-time warning, never an error, `use_jit` included;
`adam` on a manifold warns (gauge-dependent moments) and a ragged manifold `x0` is reduced to minimal ranks
on entry; geometry/kind strings are case-insensitive; the `rank_adjustment_sweep` is deliberately TT-only
(preserves, never creates, Tucker orthonormality) — docs now say so; the uniform forward norm never uses
the raw zipper by default (precision), its *derivative* does (through a `custom_jvp`); the uniform frame
must support non-minimal ranks, structural (done, S1a) and numerical (S1b, open) by orthonormal completion.

**S1b — FIXED (2026-08-23), the last S item.** The uniform frame of a *numerically* rank-deficient
train (the zero-padded `resize` warm start, `x + x`) used to lose tangent directions: the SVD's
null-space completion could land in **padded** slots, which the masks erase. Nick resolved the design
externally and delivered the **pad-safe SVD packet** (Method D, "sketch–project" — exact,
tolerance-free, one jit compile across mask patterns; the packet + verification record live in
`dev/review_2026-08-22/repros/S1b/`, incl. `packet/`). Implemented in three slices:

- `backend.linalg.pad_safe_svd` (the primitive; symmetric `min(n, m)` contract, Frobenius `c = 4‖A‖_F`
  with the load-bearing 4× margin, fixed Haar sketch) + `tests/backend/test_linalg.py` + a
  `test_dispatch` jit bucket;
- the uniform frame sweep: masks threaded through `t3_orthogonal_representations`'s uniform path (the
  evolved per-site recurrences, host-precomputed), the pad-safe scan step in `tt_orthogonalization`,
  mask-aware `down/up_orthogonalize_*_supercores`;
- `ut3svd`'s own sweep (unshared + SF-T3 shared), so the t3svd-gauge frame
  (`ut3svd_orthogonal_representations`) and retraction are covered too.

All six S1b cases report **0 lost directions** on both frame paths (`s1b_cases.py`, the
`TestPadSafeFrame` regression class). **The one behavioural consequence:** the uniform manifold frame
is now gauge-EQUIVALENT to ragged, no longer bit-identical (the old bit-equality was a LAPACK
accident); every gauge-invariant quantity still matches ragged exactly. Contract statement:
`docs/uniform_equivalence_contract.md` §"Gauge-carrying operations"; CHANGELOG `[2026.2.0]`
Fixed/Changed/Added. The standing question `dev/OPEN_QUESTION_uniform_rank_deficient_frame.md` is
resolved and archived (`dev/archive/OPEN_QUESTION_uniform_rank_deficient_frame_2026-08-23_resolved.md`).

**Also deferred, by the budget plan:** the E list in the ledger (`findings_compact.md`), and the test-hardening
phase (Phase D: the uniform prongs missing in `test_uniform_frame_variations_format` / the `_CONFIGS` matrix in
`test_uniform_tucker_tensor_train`, direct tests for the 13 `*_from_sweep` hooks, `test_dispatch` entries for
~25 ops, promoting the oracle sweeps). Nick's standing rule for any further agent work: Sonnet at medium
effort for bounded verify/test tasks, no maps in the prompt, a `+Nk` hard ceiling; Fable only for discovery.

**Gates on the finished tree (2026-08-23, post-S1b):** full suite **813 passed / 42,708 subtests**
(7:20, on the final tree incl. `TestPadSafeFrame` and the `pad_safe_svd` suite in
`tests/backend/test_linalg.py`; the slice-2b tree had separately passed 808/42,696); module doctests green; quickstart + every doc-page
doctest green (the CI command, both checked also on numpy 1.22 for the new module); `sphinx -W`
clean; the uniform continuation, uniform probe-derivatives, and shared-factors examples pass end to
end. The knowledge record is `docs/pad_safe_svd.tex` (+pdf), linked from the docstrings. The
previous release gates (wheel `twine check`, the numpy-only venv smoke) have **not** been re-run
since the review commits and must be, per `dev/archive/release_plan_2026-07-13.md` REL-2, before
tagging.

### The optimization-layer restructuring (2026-08-21) — merged, in 2026.2.0

Recorded in `docs/contributor/parameters_not_closures.md` and the CHANGELOG; the per-thread detail that used
to live here is in `dev/archive/handoff_2026-08-21_optimization_layer.md`.

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
- ~~`from_t3svd` / `from_ut3svd` discard the SVD'd train~~ — closed 2026-08-22 by `t3svd_orthogonal_representations`
  (one SVD, and the frame in the gauge the σ's belong to; review S14).
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
- ~~`get_backend` raises a bare `NameError` when jax is requested but absent~~ — closed 2026-08-22: the
  library-wide jax-absent policy (`common.jax_or_warn`: numpy + a one-time warning; review C10).
- `t3_stack` has no docstring at all; the `check_{fv,ufv,fw,ufw}_pair` export policy is inconsistent (one
  of four exported). (`bfit_pd` → `_bfit_pd` done 2026-08-22.)

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
