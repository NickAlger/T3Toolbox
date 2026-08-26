# Whole-library review, 2026-08-22 — consolidated findings (Phase A, unverified)

_Status: Phase A (find) complete; Phase B (adversarial verification) NOT yet run. Every item below is a
lane's claim with a reproduction script; about half of the planning-stage candidates turned out real, so
treat these as strong candidates, not verdicts, until Phase B._

**How this was produced.** 19 parallel lanes (10 region reads, 7 bug-class hunters, 2 oracle sweeps) over
the whole library at `657f6001`, each required to ship a reproduction per finding. Raw data:
`findings_raw.json` (all 186 entries, full text), `findings_compact.md` (one line each), `coverage.md`
(what each lane checked and did not), `lanes/<lane>.md` (the 19 narrative reports), `repros/<lane>/`
(the scripts; run with `PYTHONPATH=<repo>` from outside the repo). Classes: **S** silent wrong answer ·
**C** crash / obscure error on a supported path · **D** doc ≠ code · **E** ergonomic / style / dead code.

**Counts:** 186 findings — S 21 (20 confirmed) · C 39 · D 64 · E 62. Many are the same defect found by
several lanes (which is itself evidence); clustered below into **14 S clusters** and **13 C clusters**.

---

## S — silent wrong answers (clustered)

| # | cluster | members | where | reach |
|---|---|---|---|---|
| S1 | **Uniform orthogonal frame is wrong on non-minimal or rank-deficient input.** Two root causes: (a) the frame masks come from `ranks.compute_orthogonal_representation_ranks`, a right-then-left recurrence, but the sweep actually run is left-then-right — on any non-minimal TT rank the left/down masks are too small (the map said this function had no caller; it is called at `ufv_conversions.py:107`); (b) the right/down uniform sweep is not rank-aware at zero singular values, so a numerically rank-deficient train (zero-padded `resize` warm start, `x + x`) gets zero slices inside real mask slots. Consequence: safe mode raises "requires an orthogonal frame" on a frame the library built; under `unsafe()`/jit every manifold op (project, project_ambient, transport, inner/norm, retract) silently works in a smaller tangent space; the **documented uniform rank-continuation loop** (`docs/rank_continuation.md` "On the uniform layer") optimizes in the wrong space. Ragged twin is fine. | R3-1, H5-2, H6-1, O1-2, H4-2 | `ranks.py:549`, `ufv_conversions.py:60,107` | public: `UNIFORM_MANIFOLD.frame(from_t3(x))` for non-minimal `x`; the resize loop |
| S2 | **`ut3_squash_tails` sums padded boundary-bond slots into the real slot.** So uniform `+`, `-`, `squash_tails`, `sum_stack`, and through `__sub__` also `UT3Frame.allclose/is_consistent`, read garbage — violating the equivalence contract's "padding is don't-care". Reachable without raw construction: `UNIFORM_COREWISE` `randn`/`retract` and `adam(UNIFORM_COREWISE, …)` leave nonzero values exactly there (R8-3: the corewise gradient is nonzero in padded boundary slots, though documented "clean-padded"). | R8-1, H5-1 (+D R8-3) | `ut3_operations.py:82` | public: `adam(UNIFORM_COREWISE,…)` result then `x_fit + y` |
| S3 | **`t3m` with boundary TT bonds ≠ 1** (allowed by `validate`; produced by `segment`, `resize`, `from_tensor_train`, constructor): `inplace_fused` (the default) never caps the last bond (`max_tt_ranks` violated, 3–6× worse error); `swap` is wrong when both boundary bonds are equal and ≠ 1 and raises a raw einsum error when they differ; `swap` silently ignores a numpy-array `max_tt_ranks` (`isinstance(…, typ.Sequence)`). `form_then_round` is correct. `docs/t3m_methods.md:56` claims "boundary bonds stay 1". | R1-1, R2-1, R1-2, R2-2 (+D R1-10) | `t3_linalg.py:337,536,555,573` | public: `x.t3m(y, max_tt_ranks=…)` |
| S4 | **`rank_adjustment_sweep('right_to_left')` is not right-orthogonal** — only the TT cores are; the Tucker cores are not (the non-orthogonal factor is pushed up). Five doc sites promise it, and the documented chain `…('right_to_left').t3svd(…, assume_orthogonal=True)` truncates wrongly (0.216 vs 0.165 at identical ranks). | H2-2 | `t3_svd.py:301` | public, documented recipe |
| S5 | **`entries_ambient_transpose` returns zero factors for negative indices** (one-hots built with `arange(N) == idx`); the forward and the other two transposes wrap numpy-style, so the adjoint identity fails silently. Under jax, out-of-range indices clamp in the forward. | R5-1 | `entries.py:105` | public |
| S6 | **`Problem.objective` / `Problem.local_model` discard `data=` when `sample` is omitted** (`if sample is None: sample, data = self.sample, self.data`) — scores the training data silently; `sample=` alone crashes with a bare TypeError. | H2-1 | `backend/optimizers.py:178,197` | backend (documented entry point) |
| S7 | **Regularized `Problem.objective` at a raw (non-left-orthogonal) point** — characterized: backend-only. `LocalModel.objective` and the frontend model go through the frame; all four optimizers are unaffected (iteration-0 values come from the `LocalModel`, later points are retraction outputs, left-orthogonal to 1e-15). Fix is cheap either way. | R7-1 | `geometry.py:71` | backend only |
| S8 | **Backend probe kinds silently truncate an over-long per-mode weight** (`m > d` rows → first `d` used; `m < d` → IndexError at `sumsq`); the frontend rejects both. | R7-6 (+E H2-5) | `backend/fitting.py:257` | backend only |
| S9 | **Shared companion built on a `UT3Frame.to_t3frame()` leaf gives a wrong tied projection** (32% off, not in the tied tangent space): `fv_shared_frame_data` re-sweeps with fresh SVDs whose sign/rotation choices differ from the uniform construction's. The companion *spectrum* is right, so nothing flags it. This cross-layer route is the one `UniformManifoldGeometry.project_ambient`'s docstring recommends for dense gradients. Proposed fix builds the centers gauge-consistently from the stored cores only (no fresh SVD). | R9-1 (+D-susp R9-8) | `sharing.py:477` | public, docstring-recommended route |
| S10 | **`UT3Variations.sum_stack(axis=-1)` / `UT3Tangent.sum_tangents(axis=-1)` hit the mode axis** (array axis `1+axis`): silently sums over modes when `K == d`, obscure error otherwise; ragged twin follows numpy semantics. | R9-3 | `ufv_operations.py:195`, `utv_operations.py:169` | public |
| S11 | **`jax.grad` through uniform `norm()` / `ut3_weighted_norm` / `ut3_weighted_inner` (default orthogonalized path) is all-NaN on any train with rank slack** — the SVD-based uniform orthogonalization at padded zero singular values. Ragged, `use_orthogonalization=False`, and the tangent-level ops are finite. | R10-1 | `ut3_linalg.py:204` | public (autodiff users) |
| S12 | **A structural tangent mismatch is only caught by the numerical same-frame guard**; under `unsafe()`/jit, `a + b` of tangents at frames of different rank structure broadcasts silently (or raises a raw numpy error). Contradicts "structural problems always error". Uniform twin is fine (mask equality is structural). | H6-4 | `manifold.py:357`, `fitting.py:108` | user error, but documented to raise |
| S13 | **Structural checks written as `assert`** (~29 sites: `__mul__`, `t3_add`, `t3_inner_product`, `t3_mult`, `t3_sum`, `t3_sum_stack`, `xcat/xappend/xprepend`, `corewise.*`): empty-message `AssertionError` normally; under `python -O` a wrong-shaped `x * ndarray` silently broadcasts and corewise ops silently truncate. | R2-6, H4-6, R1-4, R1-11, H3-6 | `tucker_tensor_train.py:2023`, `corewise.py:62`, … | public under `-O` |
| S14 | _(suspected)_ **Stacked `t3_orthogonal_representations` rotates the t3svd Tucker factors**, so the `docs/weighting.md` recipe (`T3FrameWeights.from_t3weights(T3Weights.from_t3svd(x))` paired with a re-orthogonalized frame) applies σ-weights in the wrong gauge for `C ≠ ()` (and for uniform). Needs confirmation. | H4-9 | `docs/weighting.md:79` | documented recipe |

## C — crashes / obscure errors on supported paths (clustered)

| # | cluster | members | where |
|---|---|---|---|
| C1 | **Ragged `d = 1`:** `MANIFOLD.project_ambient(frame, T3)`, `project_ambient(…, method='t3svd')`, `MANIFOLD.transport` raise `IndexError` (`tt_zipper_left_to_right` on an empty tuple); `dense_probe` at d=1 with `W ≠ ()` raises an einsum error. Everything else works at d=1; the ledger says ragged d=1 is fine. | R5-3, R4-2, H3-2, H4-3, O1-1, R5-4 | `tv_operations.py:402`, `tt_operations.py:180`, `probing.py:1542` |
| C2 | **Uniform `d = 1` — root cause found:** `_tt_squash_tails_uniform` concatenates `[G0, mid, Gf]`, duplicating the single core, so `squash_tails`, `+`, `-`, `sum_stack`, `norm`, `inner`, weighted norm/inner, `t3svd`, `rank_adjustment_sweep`, `UT3Frame.from_ut3`, `is_left/right_orthogonal` (empty-array max) all crash. The ledger attributes d=1 to `ut3svd_supercores` only; the scope is the whole uniform layer. | R8-2, H5-3, R3-4, R10-3 | `tt_operations.py:114`, `ut3_orthogonalization.py:63`, `ut3_svd.py:280` |
| C3 | **Uniform `K`-stacked tangent:** `UNIFORM_MANIFOLD.transport(v, frame)` and `project_ambient(frame, K-stacked UT3)` raise a reshape error (frame's `C` masks applied to `K+C` variations); ragged twins accept `K`. | R9-2, H3-1, O1-3 | `utv_operations.py:446,574` |
| C4 | **`stacking.stack` is not jittable** (`xnp.arange` as `moveaxis` source → tracer): every frontend `stack`/`stack_tangents`/`stack_frame` fails under `jit`; `unstack` works. | R2-5, H4-4 (+E R1-17, R10-5) | `stacking.py:261` |
| C5 | **Slack-padded minimal `x0` crashes every uniform manifold optimizer at step 2**: `ut3svd` slices output to `max(raw ranks)`, so `utv_retract` returns smaller padded dims than the frame (its docstring says otherwise) and the loop-invariant masks desync. Reachable via the documented `from_t3(x, n=, r=)` "force a larger pad". | H5-4 | `ut3_svd.py:111`, `utv_operations.py:326` |
| C6 | **`adam` on ragged `MANIFOLD` with a non-minimal `x0` crashes at step 2** (moment trees allocated at `x0`'s shapes; the first retraction drops the redundant rank). The other three optimizers rebuild per step and run; the uniform path reduces `x0` via `uniform_minimal`. | O2-1 | `backend/optimizers.py:483` |
| C7 | **GAUGE precondition uses the *relative* tolerance as an *absolute* bound on an unnormalized residual** (`max|UᵀV|` scales with the tangent): `MANIFOLD.norm/inner` falsely reject a correctly gauged tangent of norm ≳1e8 (numpy) — and on jax float32 eager already at ‖p‖ ≈ 54, which breaks the documented jax-but-not-jit safe-mode use; conversely a tiny ungauged tangent passes. ORTH/TIED residuals are relative by construction. | H6-2, H4-1 | `manifold.py:547,1316`, `uniform_manifold.py:1113` |
| C8 | **Python-float residual crashes the tangent/corewise `apply`/`entries` transposes** (`compute_sigma_hat` does `c[..., None]`); the ambient twin's doctest passes a bare float. | R5-2 | `probing.py:834` |
| C9 | **Validation display:** `val_data` without `val_sample` → TypeError deep in the kind; `val_sample` alone silently ignored (no column, no error). Both frontend and backend. | R7-3, H2-3 | `optimizer_display.py:215`, `optimizers.py:394` |
| C10 | **jax-absent paths:** `randn(use_jax=True)` is a second bare `NameError` site; `to_jax()` / `load(use_jax=True)` / `tree_to_jax` silently return numpy. | H4-7, R2-11 | `common.py:102,405` |
| C11 | **Uniform jit inputs:** numpy-integer entries in a uniform `shape` tuple become traced leaves (`_is_static_leaf` accepts only Python `int`) → `newton_cg(use_jit=True)` dies with `TracerArrayConversionError`; jax-array masks pass validation but are unhashable as a cache key. | H1-1, H1-2 | `common.py:608`, `uniform_tucker_tensor_train.py:161` |
| C12 | **`uniform_least_squares_problem`:** any non-`'manifold'` geometry string (e.g. `'Manifold'`) silently builds corewise; `order=None` for a derivative kind builds a Problem that fails on first use. | R9-4, R9-5 | `uniform_fitting.py:437,440` |
| C13 | **Structural-validation gaps and wrong messages:** `check_fv_pair` accepts a d-mismatched pair (four ops then return partial sums); `T3Frame.validate` rank-mismatch message prints stack dims; `entries()` wrong-length list → `AttributeError`; `sum_stack` bad axis → bare assert; `share(rtol=)` on a stacked T3 fails deep; uniform `+`/`inner` with different padded `N` → raw numpy errors (docstring says the frontend enforces it); `GaussNewtonModel` factories do no sample/residual validation; `set_default_safety(None, None)` accepted then TypeError; `has_numerically_minimal_ranks()` raises on a stacked train. | R4-1, R4-8/H6-5, R1-3/R5-5, R1-11, R1-14, R8-5, H6-11, H6-8, H4-5 | various |

## D — doc ≠ code (64; the substantive ones)

These change what a reader *believes* the library does (the rest are typos/stale names, listed in
`findings_compact.md`):

- **`truncated_svd` documents a per-singular-value `rtol`/`atol` rule; the code implements a tail-Frobenius rule** (R2-3, R3-3 — `linalg.py:81`; propagates to every directional/pair wrapper and `t3svd`).
- `rank_adjustment_sweep` "compose both directions for guaranteed minimal ranks" is false when a Tucker rank exceeds its mode size (R3-2).
- `t3_norm(use_orthogonalization=False)` still orthogonalizes, twice (R2-4).
- Chunking: the kept (`sum_over_probes=False`) chunk path does not bound memory; the summed path is memory-inert when `W // chunk_size == 2` — incl. the default 100 for `W ∈ [200, 299]` (R6-1, R6-2); numpy-on-uniform does engage the loop contrary to `chunking.md:61` (R6-3).
- Minibatch λ scales by the *nominal* `batch`, not the draw's size, when `draw=` is given (R7-2).
- `adam` on `MANIFOLD` is gauge-dependent: numpy-eager, jax-eager, jit, ragged and uniform give different iterates (O2-2) — a design caveat nobody states.
- `numerical_contracts.md`: "retract stays a valid first-order retraction on a non-minimal frame" is false at rank-deficient frames (O1-4); `shared(...).frame(x)` listed as a TIED precondition but the code ties silently (H7-1, H6-3, R7-4, R7-5).
- Uniform `+` squashes tails, ragged `+` does not → the `+`↔`concatenate` duality does not hold for uniform (R8-4, R10-2).
- `compute_mu` shape comment says `left_tt_cores len=d-1`; all callers pass `d` and `d-1` silently truncates (R5-8).
- `fitting_and_optimization.md` §3 lists `norm` on the Geometry protocol (doesn't exist) and omits four required members (R7-8).
- `entries_apply_probe.md` §7 names `TuckerTensorTrain.{entries,apply}_transpose`, which don't exist (R5-10); `transposes.md:37` mis-describes the corewise transpose (R5-9).
- `rank_continuation.md:179` uniform loop calls `resize` with the wrong arity (R1-9).
- Paper equation/figure numbers cited in code/docs are off by one/two vs the local `t4s.pdf` (R4-3).
- The GAUGE/same-frame/identity docstrings in `manifold.py` and the `T3Frame` `eq=False` comment still describe the retired identity design (R4-5, H6-10, R4-6, H1-4); `H1-3`: "all instances of a geometry are interchangeable" is false (a pytree-reconstructed `ManifoldGeometry` is rejected by the fitting factories).
- `CLAUDE.md` Current-state still names the removed `GeometryOps` protocol and calls SF-T3 "slices 8–13 pending review" (H7-6).

## E — 62 ergonomic/style/dead-code items

Listed in `findings_compact.md`; notable: no `__rmul__` on `TuckerTensorTrain` (`2.0 * x` fails; uniform and tangent have it); ragged `t3_weighted_norm/inner` lack `use_orthogonalization`; `sum_over_probes` defaults differ between the six ragged `*_transpose_from_sweep` and their uniform twins; the six `Uniform*Kind.forward/transpose` accept but never read `sample`/`frame_data`; `ValueHashedMasks` bool-vs-int8 masks compare equal but hash differently; `tree_contains_jax` recurses forever on a `str` leaf; `safety._inside_jax_trace`'s primary detector does not exist in jax 0.10; `frames_equal` compares padding too; `test_manifold.py:886 test_riemannian_gradient` compares an expression with itself (vacuous).

## Coverage and what was not checked

Per lane in `coverage.md`. Oracle sweeps: O1 ran the sampling/tangent matrix (the only failures were the
d=1 and K-stack crashes above and the non-minimal uniform frame); O2 ran the fitting matrix (only the adam
non-minimal crash; every objective/gradient/Hessian/adjoint check at asymmetric shapes passed, ragged ==
uniform, jit == eager). The derivative-jet math (R6) verified standard forms == `_trs` twins == dense
oracles == finite differences at asymmetric shapes and orders 0–4: no numerical finding.

## Next

Phase B: adversarial verification (two independent refuters per S/C cluster, one confirmer per substantive
D), then triage with Nick under the agreed gate (confirmed S + C block the tag; D fixed; E deferred).

---

## Phase B — verification verdicts (inline, 2026-08-22)

Done in the main session by re-running each lane's repro and reading the code; multi-lane clusters
(S1, S2, S3, S13, C1, C2, C3, C4, C7, C9, C13) were treated as independently confirmed by construction
and only spot-checked. Every single-lane S/C item and every substantive D item was re-run.

| item | verdict | note |
|---|---|---|
| S1 | holds (spot-checked) | both root causes reproduce: mask recurrence left ranks (1,2,2,1) vs actual (1,2,3,1); zero-padded `resize` start → uniform frame residual 1.0 vs ragged 7e-16 |
| S2 | holds (spot-checked) | R8-3 also holds: the corewise transposes leave nonzero values in padded boundary-bond slots (864/… sweep rows), which is what feeds S2 |
| S4 | holds | `is_right_orthogonal()` False after the sweep on generic input; documented chain truncates worse (0.216 vs 0.165) |
| S5 | holds | ambient transpose zero for negative indices; jax forward silently clamps out-of-range (extra D) |
| S6 | holds | `objective(x, data=other)` returns the training value |
| S7 | holds, backend-only | already characterized |
| S8 | holds, backend-only | |
| S9 | holds | 32–33% off on the `to_t3frame()` leaf route, which `UniformManifoldGeometry.project_ambient`'s docstring recommends for dense gradients |
| S10 | holds | |
| S11 | holds | NaN grads on tight and padded trains; finite with `use_orthogonalization=False` |
| S12 | holds | broadcastable-hole case: `a + b` under `unsafe()` returns a tangent with no error |
| S13 | holds | |
| **S14** | **holds, sharper than claimed** | the frame's Tucker basis is rotated relative to the t3svd singular basis **unstacked too** (the default up-orthogonalization re-SVDs an already-orthonormal factor: degenerate spectrum → arbitrary rotation). `t3_orthogonal_representations(xs, already_left_orthogonal=True)` preserves the gauge to 1e-15 (stacked and unstacked). The `docs/weighting.md` recipe omits the flag, so the σ-weights land on rotated coordinates. Fix is documentation + a guarded helper. |
| C5 | holds | slack pad (n=5,r=4 over real 3,2) crashes `newton_cg`/`mc_sgd`/`adam`/backend GD; COREWISE fine |
| C6 | holds | adam only; the other three run |
| C7 | holds | numpy false failure at ‖p‖≈5e7; jax float32 eager fails on a modest `randn((4,5,6),…)` problem (`gauge_residual` 1.2e-5 > 1e-5) |
| C8, C9, C10, C12 | hold | |
| D `truncated_svd` rule | holds | spectrum [1, .08×4], rtol .1: doc rule keeps 1, code keeps 4 (tail-Frobenius); `t3svd` parsimony bound in the docs is false |
| D λ vs `draw` | holds | factor 0.5 used for a 5-element draw with `batch=50` |
| D adam gauge-dependence | holds | MANIFOLD 5.7e-3 numpy-vs-jax, 6.9e-3 ragged-vs-uniform; COREWISE and newton_cg agree to 1e-15 |
| D two-chunk memory | holds | W=200, chunk 100: 39.6 MiB vs dense 42.8 (expected ~21) |
| D `t3_norm(use_orthogonalization=False)` | holds | orthogonalizes twice |
| D retract on non-minimal frame | holds | FD ladder flat at 1.1e-1 on a `share()`-entered point; fine on the `n0 > r0 r1` case and on `shared_manifold.retract` |
| D `n_i > N_i` sweep claim, interchangeable-geometry claim | hold | |
| R7-13 SharedGeometry subclass | refuted (as in planning) | an unregistered subclass is a leaf; as a jit arg it raises a TypeError, no silent base-class swap |

Nothing in the S/C list was refuted. One lane claim was imprecise (S14 "stacked only"), in the direction of being worse.

---

## Phase C — fixes landed (2026-08-22, same day)

Every commit carries the cluster id; `git log --oneline 657f6001..` lists them. Tier 1 + Tier 2, all
in the ledger's order of severity:

| cluster | commit subject (abridged) | status |
|---|---|---|
| S2, C2 | uniform squash_tails masks on entry; d = 1 degenerates to the vector case | fixed |
| S3 | t3m canonicalizes boundary TT bonds; ndarray max_tt_ranks honored | fixed |
| S5, C8 | entries ambient transpose wraps negative indices; float residuals | fixed |
| S6, S7 | Problem takes (sample, data) together; point_norm_sq exact at any ragged point | fixed |
| S8 | per-mode weight may carry more rows than modes (ruled intended; short rows error) | changed |
| S10, S12, S13 | uniform sum_stack axes; structural tangent check unconditional; 30 asserts → errors | fixed |
| C1, C4 | ragged d = 1 projection/transport/dense_probe; stack() under jit | fixed |
| C6, C9, C12 | ragged manifold x0 reduced to minimal (adam warns); val args; string validation | fixed |
| C10 | jax requested but absent → numpy + warning, library-wide (incl. use_jit) | changed |
| S14 | t3svd_orthogonal_representations (+ uniform twin): the frame in the T3-SVD gauge | fixed |
| C13 | nine validation gaps / wrong messages | fixed |
| known D list | CITATION, CLAUDE.md, README/user guide, API reference, chunking/sharing/weighting docs, … | fixed |
| S1a | frame-mask rank recurrence follows the sweep (non-minimal structural ranks) | fixed |
| C7 | gauge residual relative (whole-tangent norm), both layers | fixed |
| C5 | utv_retract re-pads to the frame dims (new ut3_pad_ranks) | fixed |
| S9 | shared companion centers from the zipper of the stored chains | fixed |
| S4 | rank_adjustment_sweep: orthogonality stated conditionally; n_i > N_i guard | fixed (reclassified D + small fix) |
| C3 | uniform transport / project_ambient of a K-stacked tangent | fixed |
| S11 | ut3_norm / ut3_inner with a custom_jvp (no SVD in the derivative) | fixed |
| substantive D | truncation rule, Geometry protocol, contracts rows, transposes, chunking regimes, … | fixed |

**S1b — FIXED (2026-08-23).** The uniform frame on a *numerically* rank-deficient train (the
zero-padded `resize` warm start, `x + x`): the SVD's null-space completion could land in the
**padded** bond/mode slots, which the masks then erase. Fixed by Nick's pad-safe SVD design
(Method D, "sketch–project"; packet + verification: `repros/S1b/packet/`): the new
`backend.linalg.pad_safe_svd` replaces every kept-basis SVD in the uniform sweeps (frame sweep AND
`ut3svd`'s, unshared + shared), with the mask recurrences threaded per step. 0 lost directions on
all six cases, both frame paths; regression class `TestPadSafeFrame`; behavioural note (frame now
gauge-equivalent, not bit-identical, to ragged) in the CHANGELOG and
`docs/uniform_equivalence_contract.md`.
Also open: the E list and the test-hardening phase (Phase D), deferred to later sessions per the budget plan.
