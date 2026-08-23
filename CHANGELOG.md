# Changelog

All notable changes to T3Toolbox are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/); versions are `YYYY.MINOR.PATCH`.

## [2026.2.0] — 2026-08-22

### Changed — breaking

- **The optimization layer's axis objects are classes, not records of closures.** A geometry and a
  sampling kind are now frozen dataclasses whose *parameters are fields*; their behaviour is methods.
  This is what makes them compare and hash **by value**, which matters because both ride as jax
  `aux_data` and therefore form part of the compilation cache key. Rationale, measurements and the
  rejected alternatives: [`contributor/parameters_not_closures.md`](contributor/parameters_not_closures.md).
  - **New `backend/geometry.py`** with `ManifoldGeometryOps` / `CorewiseGeometryOps` and their
    `Uniform*` twins (the uniform ones carry the fixed rank they were built at; use `from_point`).
    Removed: `backend.optimizers.GeometryOps`, `MANIFOLD_OPS`, `COREWISE_OPS`, `shared_geometry_ops`,
    and `backend.uniform_fitting.uniform_{manifold,corewise,geometry}_ops`.
  - **Sharing is a `groups` field**, not a wrapper — one class and one code path for shared and
    unshared. The frontend `shared_manifold` / `shared_corewise` constructors are unchanged.
  - **`SamplingKind` is a class hierarchy** (`ApplyKind`, `EntriesKind`, `ProbeKind`, the three
    `*DerivativesKind`, and their `Uniform*` twins). The `*_kind` constructor spellings still work.
    Removed: the `identity` field and its `__eq__`/`__hash__`, and the six uniform builder functions
    `uniform_{apply,entries,probe}_kind` / `uniform_{apply,entries,probe}_derivatives_kind` (use
    `Uniform*Kind.from_point(x0_data, **parameters)`, or the unchanged `uniform_sampling_kind` /
    `uniform_derivatives_kind` dispatchers). A custom kind is now a **subclass**; deriving one with
    `dataclasses.replace` no longer type-checks, and that is the point — see below. `ScalarOutputKind`
    and `ProbeOutputKind` are public, so a new operator can inherit the `‖·‖²` reductions.
  - **`fitting.UniformGaussNewtonModel` is removed.** One `GaussNewtonModel` serves both
    representations; the factories still dispatch on `x`. It is deliberately not aliased.
  - `optimizer_display` gates its error table on `kind.has_block_sumsq` rather than on a `block_sumsq`
    field being `None`. `block_sumsq` is optional; the flag is declared by whoever implements it.
  - **A custom backend geometry needs three more members**: `stack_shape(x_cores)`, `base_point(frame)`,
    and a `precompute` that is a *callable* returning `None` rather than `None` itself. The `Geometry`
    protocol in `backend/optimizers.py` lists the full surface.

- **Optimizing a STACKED point raises `NotImplementedError`.** It never worked -- `newton_cg` and
  `gradient_descent` failed on `float()` of a shape-`C` objective, `mc_sgd` on a broadcast, and `adam`
  once its loss logging first fired -- but it failed obscurely, from deep inside the loop. Building the
  local model on a stacked point *is* supported and still works, so rolling your own loop over a stacked
  problem is the route. Stacked optimization is possible future work.

### Fixed

_The items below came out of the 2026-08-22 whole-library review (`dev/review_2026-08-22/`)._

- **Uniform `+`, `-`, `squash_tails` and `sum_stack` read padding garbage.** `ut3_squash_tails` summed
  the leading/trailing TT bond over *every* slot, including the padded ones the equivalence contract
  declares don't-care -- and a corewise gradient step (`adam(UNIFORM_COREWISE, …)`, `UNIFORM_COREWISE.retract`)
  leaves exactly those slots nonzero, so `x_fit + y` was silently wrong while `to_dense`/`norm` (which mask
  on entry) were right. It now masks on entry.
- **`t3m` with boundary TT ranks `r0, rd != 1`.** `inplace_fused` (the default) never SVD'd the last bond,
  so an unsquashed trailing bond survived as `rd_x · rd_y` and escaped `max_tt_ranks`; `swap` contracted the
  two operands' trailing bonds against each other (silently wrong when equal, a raw einsum error when not)
  and ignored a per-position `max_tt_ranks` given as a numpy array. Every `t3m` method now canonicalizes
  both operands' boundary bonds to 1 on entry -- the rule, and why the exact `*`/`t3_mult` deliberately do
  not, is in [`t3m_methods.md`](t3m_methods.md).
- **`entries_ambient_transpose` returned zero factors for a negative index.** Its one-hots were built with
  `arange(N) == index`, which matches nothing for `-1`; the forward `entries` and the tangent/corewise
  transposes wrap numpy-style, so the adjoint identity silently failed. All four now agree (`eye(N)[index]`).
  Documented alongside: an out-of-range index raises under numpy but clamps under jax (gather semantics).
- **The documented singular-value-metric recipe applied the σ's in the wrong gauge.** `docs/weighting.md`
  paired `T3FrameWeights.from_t3weights(T3Weights.from_t3svd(x))` with a frame from
  `t3_orthogonal_representations(x)`, which re-SVDs the already-orthonormal Tucker factors of a T3-SVD
  result -- a degenerate spectrum, so the frame's Tucker basis comes back rotated by an arbitrary orthogonal
  matrix relative to the singular basis, and per-coordinate σ-weights then weight the wrong directions
  (unstacked too; the planning measurement said "stacked only"). New **`t3svd_orthogonal_representations(x, **t3svd_kwargs)`**
  (+ the `ut3svd_` twin, both exported at the root) returns `(frame, variations, T3Weights)` from ONE SVD
  with the frame built `already_left_orthogonal=True`, so its Tucker basis is the singular basis; the recipe
  and the `from_t3svd` docstring now say so.
- **The shared-factor companion was wrong on a frame converted between layers.** `fv_shared_frame_data`
  recomputed the centers by re-running the SVD sweep that built the frame, which reproduces the
  construction's sign choices only when it is the same SVD on the same arrays; on a `UT3Frame.to_t3frame()`
  leaf -- the route `UniformManifoldGeometry.project_ambient` recommends for dense gradients -- the tied
  projection was silently 30% off (the group spectrum, being gauge-invariant, looked right). The centers
  are now the zipper of the stored chains (`H_i = L_i Z_{i+1}`, no SVD), exact and gauge-consistent for a
  frame built by either layer; the uniform twin masks first, and the "frames packed from ragged are not
  guaranteed" caveat is gone.
- **`jax.grad` through uniform `norm()` / `inner()` / `ut3_weighted_norm` / `ut3_weighted_inner` was all-NaN on
  any train with rank slack** -- the default orthogonalized path differentiated through the SVD, whose JVP
  has `1/(σᵢ²−σⱼ²)` terms and a padded train has several exactly-zero σ's. New backend twins
  `ut3_norm` / `ut3_inner` (closing a ledger item) keep the precise orthogonalized VALUE and carry a
  `custom_jvp` with the exact multilinear derivative (`2⟨T, dT⟩`, via the zipper with the orthogonalized
  side held fixed -- no SVD in the derivative path); the frontend and the weighted ops route through them.
- **`backend.optimizers.Problem.objective(x, data=…)` / `.local_model(x, data=…)` ignored `data`** when
  `sample` was omitted, silently scoring the training data; `sample=` alone crashed with a bare `TypeError`.
  The pair now goes together or not at all (a structural error otherwise).
- **The regularized objective was wrong at a raw point on the ragged manifold.** `ManifoldGeometryOps.point_norm_sq`
  read `‖last TT core‖²`, exact only for a left-orthogonal point (measured 3 where the truth was 1400 on a
  `randn` point). Backend-only: the optimizers evaluate the regularizer through the frame or at retraction
  outputs, so no optimizer result or `stats` row was affected. It now left-orthogonalizes first (no `W`
  factor, negligible beside the misfit it sits next to), matching the uniform twin.
- **`UT3Variations.sum_stack(axis=-1)` / `UT3Tangent.sum_tangents(axis=-1)` summed over the MODE axis** (the
  raw `1 + axis` sent `-1` to array axis 0), silently when `K == d` and with an obscure shape error
  otherwise. Negative axes now count from the last stack axis, as in numpy and the ragged twin; out of
  range raises.
- **A structural tangent mismatch was caught only by the numerical same-frame guard**, which is skipped
  under `safety.unsafe()` / a jax trace -- so `a + b` for tangents at frames of different rank structure
  broadcast silently there. The core-shape comparison now runs unconditionally (structural problems always
  error), in `T3Tangent` and the fitting model's trial-tangent guard.
- **30 structural checks were written as `assert`** (`x * ndarray` shape, stack-axis ranges, core-tuple
  lengths, `corewise` operand types, `dense_probe` vector shapes, …): a bare `AssertionError` normally, and
  no check at all under `python -O`, where a wrong-shaped `x * ndarray` broadcast silently. All are
  `ValueError` / `TypeError` with a message now.
- **The gauge precondition compared an absolute residual against the relative tolerance.** `gauge_residual`
  was `max|UᵀV|`, which scales with the tangent, while `MANIFOLD.inner`/`norm` threshold it at
  `rtol` (1e-9 numpy, 1e-5 jax): a correctly gauged tangent of norm ≳1e8 failed in numpy, and on jax's
  default float32 one of norm ≳50 -- the documented jax-eager safe-mode use -- while a tiny ungauged
  tangent passed. The residual is now relative (each gram divided by its variation core's norm, per stack
  element), on both layers.
- **The tangent/corewise `apply`/`entries` transposes rejected a bare Python float residual** (the natural
  unstacked case; the ambient twin's own doctest passes `1.7`).

- **`d = 1` on the ragged manifold:** `MANIFOLD.project_ambient(frame, <T3>)`, `project_ambient(…, method='t3svd')`
  and `MANIFOLD.transport` raised an `IndexError` (an empty TT chain has no zipper); `dense_probe` raised on a
  one-mode tensor with a probe stack. A one-mode T3 is a vector; these now degenerate to that case like
  everything else.
- **`UNIFORM_MANIFOLD.transport` / `project_ambient` crashed on a `K`-stacked tangent or gradient** (the
  frame's `C`-stack gauge masks were handed to `K+C` variations and failed in a reshape); the ragged twins
  accepted `K`. The masks are now broadcast over `K` (new `ufv_masking.ufv_variation_masks_over_stack`, the
  same rule the sampling transposes already used).
- **`stack()` was not jittable** (`TuckerTensorTrain.stack`, `T3Frame.stack`, `T3Tangent.stack_tangents`, the
  uniform twins): `moveaxis` was given an `arange` source, which is a tracer under `jit`.
- **`adam` on a ragged manifold with a non-minimal-rank `x0` crashed at step 2** (its moment trees were
  allocated at `x0`'s core shapes; the first retraction drops the redundant rank). A ragged manifold `x0` is now
  reduced to minimal ranks on entry, mirroring the uniform path's `uniform_minimal`; a minimal start is untouched.
- **A slack-padded `x0` crashed every uniform manifold optimizer at step 2.** `UniformTuckerTensorTrain.from_t3(x, n=, r=)`
  (the documented "force a larger pad") passes the minimal-rank gate, but `utv_retract` returned its point at
  the max rank `ut3svd` kept rather than at the frame's padded dims (its docstring promised the latter), so
  the loop-invariant masks no longer fit. The retraction now re-pads (new `backend.ut3_operations.ut3_pad_ranks`).
- **`newton_cg(verbose=True, val_data=…)` without `val_sample` failed deep inside the kind**, and `val_sample`
  alone was silently ignored; both now raise at the entry (`make_newton_display` too).
- **`uniform_least_squares_problem`** accepted any non-`'manifold'` geometry string as corewise (so `'Manifold'`
  silently fit on the wrong geometry), and built a `Problem` for a derivative kind without `order=` that failed
  on first use. Geometry and kind strings are validated (case-insensitively, see below) and `order` is required up front.

- **Structural validation gaps and wrong messages** (each a deep or misleading failure before):
  `check_fv_pair` accepted a frame and variations with different numbers of cores (four ops then returned
  partial sums); `T3Frame.validate`'s Tucker-rank message printed stack dimensions on a stacked frame;
  `entries()` with a wrong-length *list* index raised `AttributeError` instead of the documented
  `ValueError`; `share(rtol=…)` on a stacked T3 failed inside `truncated_svd`; uniform `+` / `inner` on
  operands with different padded widths `N` raised raw numpy errors; the six `GaussNewtonModel` factories
  did no validation of the sample / residual against the point; `set_default_safety(None, None)` was
  accepted and then every check crashed (it now means *unsafe by default*, the one script-level way to
  get it; a lone `None` is rejected); `has_numerically_minimal_ranks()` raised on a stacked train (now a
  per-element bool array like the other checkers).

### Changed

- **jax requested but not installed: run on numpy and warn, never raise.** `TuckerTensorTrain.randn(use_jax=True)`
  / `UniformTuckerTensorTrain.randn(use_jax=True)` used to die with a bare `NameError`, `to_jax()` /
  `load(use_jax=True)` silently returned numpy, and the optimizers' `use_jit=True` raised. The policy is now one
  rule (`backend.common.jax_or_warn`): the request runs on numpy with a one-time `RuntimeWarning` naming the
  `t3toolbox[jax]` extra -- a project developed without jax on one machine and deployed with it on another runs
  unchanged on both. (`use_jit=True` with jax *present* still auto-converts to jax, as in 2026.1.0.)
- **Geometry and kind strings are case-insensitive** on the frontend optimizers and `uniform_least_squares_problem`
  (`'Manifold'`, `'PROBE_DERIVATIVES'`, …).
- **`adam` on a manifold geometry warns**: its per-coordinate moments live in the gauge-dependent core
  coordinates, so its iterates depend on the backend / representation (measured ~1e-2); `COREWISE` is the
  intended geometry, `mc_sgd` / `newton_cg` the gauge-invariant manifold optimizers.
- **A per-mode residual weight may have more rows than modes.** The extra rows are ignored (intended: one
  weight rides through a continuation scheme that adds modes), now on both layers -- the frontend used to
  reject it while the backend truncated silently. Fewer rows than modes is a structural error on both.

### Fixed (continued)

- **`d = 1` on the uniform layer.** The uniform boundary-bond squash duplicated the single core, which took
  down every squashing op (`+`, `-`, `norm`, `inner`, `t3svd`, `rank_adjustment_sweep`, `UT3Frame.from_ut3`,
  `is_left_orthogonal`). A one-mode T3 now degenerates to the vector case on both layers, as intended.
- **A regularized fit of a stacked point silently mis-weighted, and now raises.** The data misfit keeps
  the frame stack `C` (one value per element) while every regularizer scalar collapses it, so
  `objective = misfit + ρ` added the whole-stack regularization total to *each* element -- inflating the
  effective `λ` by about `|C|`, unevenly. The regularizer *gradient* was per-element correct, which is
  what made it easy to miss. No test combined `stack_shape` with a regularizer.

- **A derived sampling kind silently reused its parent's compiled program.** `dataclasses.replace`
  copied the hand-maintained `identity` tuple unchanged, so `dc.replace(APPLY, forward=<other math>)`
  compared equal to `APPLY`; under `jit` it returned `APPLY`'s answer (measured: 115.302888 where eager
  gave 28.825722). Unrepresentable now — a variant is a subclass, hence a distinct type.

### Documentation

- **`rank_adjustment_sweep`'s orthogonality is conditional**, and the docs said so unconditionally: the
  sweep re-SVDs the TT cores and bonds and *preserves* Tucker orthonormality but never creates it, so its
  result is right-/left-orthogonal in the full sense only when the input's Tucker cores were already
  orthonormal (any `t3svd` result) -- the documented precondition, now stated at every site. The one real
  gap behind "compose both directions for guaranteed minimal ranks" -- a Tucker rank above its mode size
  (`n_i > N_i`), which no TT-side step can reduce -- is closed by a Tucker up-SVD at the start of the sweep.

- A sweep of doc ↔ code mismatches found by the 2026-08-22 review: the README / user-guide core diagram
  and frame-core listings (`d` cores, not `d+1`; `up_tucker_cores`), the Tucker-rank tuple, the uniform
  mask form (canonical prefix vs gappy working form), the registered-pytree list (the uniform classes were
  missing), the paper section cited for probing, stale "slice"/"deferred" module docstrings on the uniform
  layer, the `UT3FrameMasks` value-hash comment, `chunking.md`'s `'auto'` default (it is the frontend's),
  `sharing.md`'s "closure state", six `dev/regularization_design.md` paths (now in `dev/archive/`), the
  `t3m_methods.md` boundary-bond sentence, the `weighting.md` recipes and "same operations" claim, the
  `GeometryOps` name in the project guide, `backend.geometry` and `corewise` in the API reference, and the
  `save`/`load` doctests, which now write into a temporary directory instead of the working directory.

### Performance

- **The inner CG compiles once per fitting run instead of once per Newton iteration** (measured
  1 → 0 compiles per iteration on the uniform `probe_derivatives` path, warm). `_cg_solve` is jitted as
  a whole function of `(local_model, rhs, tol, maxiter)`, so the cache key is the model's pytree
  treedef — value-based, thanks to the change above.
- **`mc_sgd` / `adam` per-step kernels compile once per shape signature**, process-wide, rather than
  once per optimizer call: the kernels are module-level functions and the jit wrapper is memoized on
  them, replacing a per-call closure that discarded its own cache.
- A user-defined kind or geometry now gets this for free, provided its parameters are dataclass fields.

## [2026.1.0] — 2026-08-20

### Added

- **Shared Tucker factors (SF-T3)** — optimize over Tucker tensor trains whose Tucker factors are
  constrained equal within user-specified groups of modes (the SF-ETT decomposition of Molozhavenko &
  Rakhuba (2026), generalized from one trailing block to an arbitrary partition; the partition is
  always user-provided). A `sharing` spec is one hashable group label per mode, e.g. `(0, 0, 1)`;
  a shared T3 is an ordinary `TuckerTensorTrain` whose group factors are equal (redundant storage —
  a compute-not-memory feature). Ragged layer and its full uniform mirror:
  - **The shared geometries** — `shared(MANIFOLD, sharing)` with shorthands `shared_manifold(sharing)`
    / `shared_corewise(sharing)`, exported from the package root: drop-in geometry wrappers for every
    optimizer and fitting model. Projections post-pass onto the tied tangent subspace *in the base
    geometry's own metric* (manifold: the tilted least-squares projection through a per-frame SVD
    companion; corewise: the per-group mean), and the manifold retraction goes through a tied
    doubled-rank embedding plus the grouped `t3svd`, so every iterate stays exactly tied (one array
    per group).
  - **Grouped `t3svd(sharing=)` / `rank_adjustment_sweep(..., sharing=)`** — the paper-faithful
    two-phase truncation (TT rounding, then simultaneous group Tucker SVDs on concatenated
    matricizations), reporting one spectrum `s_g` per group; `sharing=None` or an all-singleton
    partition is the existing sweep exactly.
  - **`TuckerTensorTrain.share(sharing, ...)`** — the quasi-optimal shared initializer (exact
    common-span rewrite + grouped truncation), and the checker method
    **`has_shared_tucker_factors(sharing, rtol=)`** (per-stack-element bool).
  - **Shared rank bookkeeping** — `get_minimal_ranks(..., sharing=)` /
    `backend.ranks.compute_minimal_ranks(sharing=)` with the group ceiling
    `n_g <= min(N_g, sum_i min(N_g, rL_i*rR_i))` (per-mode ceilings ADD across a group, so a shared
    rank may exceed an individual mode's `rL_i*rR_i` — the unshared reduction would clip it and untie
    the group), `manifold.manifold_dim(s, sharing=)` (one Stiefel term per group; validated against
    dense tied-tangent ranks), and `frame_has_minimal_ranks(..., sharing=)`.
  - **Shared rank continuation** — `continuation_ranks(sharing=)` /
    `backend.ranks.compute_continuation_ranks(sharing=)`: a group's Tucker edges are ONE edge — one
    `κ_g = s_g[0]/s_g[-1]` in the conditioning pool, one growth decision applied group-wide, one
    `max_grow` candidate — with the shared useless-rank removal (`κ_g` is never worse than the group's
    worst per-mode condition number, and can be far better on complementary spectra). Plus
    `resize(..., sharing=)` for the zero-padded warm start: the group factor is padded once (one array
    per group), the represented tensor unchanged. A freshly padded restart carries exactly-zero new
    spectrum levels (the tied Tucker channel is gated); the escape runs through the untied TT-variation
    channel within the first Newton steps — which is why full shared rank is a diagnostic, never an
    enforced precondition.
  - **Uniform mirror, grouped truncation family** — `UniformTuckerTensorTrain.t3svd(sharing=)` /
    `rank_adjustment_sweep(..., sharing=)` / `has_shared_tucker_factors(sharing, rtol=)`, backend
    `ut3svd(sharing=)` (the two-phase grouped sweep in scan/supercore form: TT-bond rounding scan with
    the Tucker steps skipped, centers collected by the polymorphic right sweep, per-group SVDs on
    statically-gathered concatenations — mask-only truncation, ONE group rank mask at every group
    mode), `ut3_rank_adjustment_sweep(sharing=)`, the grouped host recurrence
    `compute_raw_sweep_ranks(sharing=)` (verified == the ragged grouped output ranks over randomized
    structures/caps), the masked checkers `ut3_sharing_residual` / `ut3_tucker_factors_shared`, and a
    sharing-aware `uniform_minimal` — required: the per-mode reduction silently unties a shared
    uniform start (it can clip a group rank to unequal per-mode values). All verified under the
    uniform equivalence contract (== the ragged grouped ops on real parts, per stack element,
    varying-rank stacks included), with exact output-mask assertions, garbage robustness, and
    jit-clean dispatch.
  - **Uniform mirror, tied tangent machinery** — the uniform companion `ufv_shared_frame_data`
    (the identical polymorphic derivation on the frame's stored supercores — deliberately NOT
    re-masked: the companion's exactness rests on reproducing the construction's own sweep on the
    same arrays, and the padded rows of each `S_i^T` vanish because completion rows are orthogonal
    to the centers' row space), the tied post-passes `ufv_share_tucker_variations` /
    `ufv_share_tucker_variations_corewise` (mask-and-delegate to the polymorphic ragged solves), and
    `shared_data=` threading through `utv_orthogonal_gauge_projection` / `utv_to_ut3` (the TIED
    doubled-rank embedding: `Udot` at every group mode, the companion's centers replacing the down
    cores, the variation block rebuilt at the up width) / `utv_retract` (tied embedding + the grouped
    `ut3svd`). Verified gauge-invariantly against the ragged twins (dense tangents/points at
    machine precision; outputs exactly tied).
  - **Uniform mirror, shared geometries + fitting** — `shared(UNIFORM_MANIFOLD, sharing)` /
    `shared(UNIFORM_COREWISE, sharing)`: the `SharedGeometry` wrapper now takes the uniform
    singletons too (uniform points/frames/tangents in and out; same value-hashed identity), the
    backend factories take `sharing=` (`uniform_geometry_ops` and both singles — the closures
    capture the partition beside the masks and populate `precompute` with the uniform companion),
    `uniform_least_squares_problem(sharing=)` with a shared-minimal gate, and the frontend fitting
    models carry the companion as a `geometry_aux` leaf (`UniformGaussNewtonModel`), so the packed
    compile-once path holds for shared fits (one trace across rebuilt same-rank models). All four
    optimizers run shared on the uniform layer; deterministic trajectories match the ragged shared
    runs, and every iterate stays exactly tied.
  - **Docs**: the user page [`docs/sharing.md`](https://nickalger.github.io/T3Toolbox/sharing.html) (the format, the grouped
    truncation, *what the group spectrum is*, the two geometries, rank machinery, batching,
    uniform, scope — including sharing ≠ symmetry); the design record
    [`docs/contributor/sharing_internals.md`](https://nickalger.github.io/T3Toolbox/contributor/sharing_internals.html); a shared
    section in the CI-doctested getting-started tour; the sharing section of
    `docs/rank_continuation.md`; and the TIED precondition rows in `docs/numerical_contracts.md`.
  - **Weights × sharing** — the combination composes within the existing framework (absorbing keeps
    a tied T3 tied iff the group Tucker weight vectors are equal; TT-bond weights never touch the
    factors; `from_t3svd(x, sharing=…)` builds group-equal weights; the weight algebra preserves
    group-equality), so the only addition is the non-enforcing compatibility checker:
    `T3Weights.has_shared_tucker_weights(sharing, rtol=)` and the `UT3Weights` twin (masked
    content), with `t3_tucker_weights_sharing_residual` / `t3_tucker_weights_shared` (+ `ut3_*`) in
    `backend.sharing`. Nothing gates — absorbing group-unequal weights legitimately unties.
  - **Example**: `examples/fit_shared_factors_jetted_probes.py` — a groupwise-symmetric five-mode
    target (two Hilbert tensors coupled by a random matrix; two sharing groups of different mode
    sizes) fit from noisy probe-derivative jets, running the SAME rank-continuation fit shared vs
    unshared: tying the factors the target's symmetry justifies reaches ~35% lower true error with
    ~37% fewer parameters before overfitting.
  - **Entering the format is the geometry's job.** `shared(...).frame(x)` ties an untied `x` first,
    silently, by the per-group mean — so an untied initial guess is a non-event for the optimizers, and
    the slow drift a long run of low-precision first-order steps can produce is absorbed at the next
    frame. An already-tied point is a bitwise fixed point, so the ordinary path is unchanged. The
    uniform layer gets the same route without a round trip through ragged:
    **`ut3_tie_tucker_factors`**, the twin of `t3_tie_tucker_factors` (garbage-transparent, so it needs
    no masking; masks and TT cores untouched). The shared **corewise** retraction ties the *sum* rather
    than aliasing one mode's copy of it, which makes it total: `mean_i(U_i + V_i) = mean_i(U_i) +
    mean_i(V_i)`, so an untied tangent (always handled) and an untied base point both land on the
    shared set with nothing silently discarded. The TIED-tangent precondition is now one backend
    checker — `fv_tied_variations_residual` / `ufv_tied_variations_residual`, a single **global
    Frobenius** ratio **per stack element** — replacing two hand-rolled formulas that disagreed between
    the ragged and uniform paths and collapsed the whole stack into one scalar (so one untied stack
    element could hide behind many tied ones).
  - Backend surface in `backend.sharing` (`validate_sharing`, `t3_sharing_residual`,
    `t3_tucker_factors_shared`, `t3_tie_tucker_factors`, `SharedFrameData` +
    `fv_shared_frame_data`, the tied post-passes) and `backend.t3_svd.t3_share_tucker_factors`.
    Safe mode checks tied factors at shared entry points; full shared rank is a diagnostic, never a
    precondition (rank-continuation restarts legitimately sit below it).

- **The grouped-einsum interpreter `backend.contractions.contract`** — one general entry point for
  every grouped contraction: `contract('WCa,Caib,WCi->WCb', *operands, len_W=...)`. Standard einsum
  strings where an UPPERCASE letter is a *group* of zero or more axes (`W` probe stack, `K` tangent
  stack, `C` frame stack, …); group sizes are solved exactly from the operand ndims (identifiability
  is decided from the subscripts alone — a call site either always needs a `len_<G>` supplement or
  never does, and the error names precisely what is missing); groups expand into fresh single-axis
  letters and one ordinary einsum runs on the operands as given. **No reshape ever happens**, so
  every sub-axis of every group is shardable (compiler-verified across the whole library vocabulary)
  and fusing two groups is inexpressible. numpy keeps the greedy pairwise BLAS path (computed on the
  grouped string); jax gets a single fused einsum.

- **Weighting (edge weights)** — diagonal weights on the internal edges of a T3, as a lightweight data
  format plus `absorb` into cores, in **both** the ragged and uniform representations. Two classes per
  layer, because a tensor and a tangent have genuinely different edges:
  - **`T3Weights` / `UT3Weights`** weight a Tucker tensor train **as a tensor** (`tucker[d]`, `tt[d+1]` —
    exactly the shape `t3svd` returns, so the singular values *are* the canonical weight object).
  - **`T3FrameWeights` / `UT3FrameWeights`** are a **metric on a tangent's coordinates** (`up`/`down`/
    `left`/`right`, each `len=d`) — the Grasedyck–Kramer preconditioner — absorbed into the variation
    cores with the frame left orthonormal, so they are `O(ranks)`.
  - Operations on all four: `absorb_weights`, `weighted_norm` / `weighted_inner`, `reciprocal` / `sqrt`,
    and `concatenate` / `kronecker` (the `+` / `⊙` duality: ranks add / multiply). Constructors
    `from_t3svd` / `from_ut3svd` and `from_t3weights` / `from_ut3weights`, plus ragged↔uniform
    conversions for both weight types.
  - The frontend free functions carry the family prefix — `t3_absorb_weights`, `ut3_absorb_weights`,
    `fv_absorb_weights`, `ufv_absorb_weights` (+ `t3_`/`ut3_weighted_norm` / `_weighted_inner`) — and the
    whole surface is exported from the package root. Docs: `docs/weighting.md`; design records:
    `docs/contributor/weighted_internals.md`.

- **Regularization on the fitting objective** — `regularizer=` on every optimizer
  (`gradient_descent` / `mc_sgd` / `adam` / `newton_cg`) and on the six model factories, minimizing
  `½‖ω⊙(S(x) − y)‖² + ρ(x)`. Shipped implementation: `IdentityRegularizer(λ)` = `½λ‖x‖²` — a
  Hilbert–Schmidt ridge on `MANIFOLD`, weight decay on `COREWISE` — composing with every optimizer,
  sampling kind, geometry and representation (ragged **and** uniform), with `λ` auto-scaled by
  `batch/n` in the minibatch steps so the stochastic objective is an unbiased estimate of the full
  one. Extensible through the small `Regularizer` protocol in `backend/regularization.py`. Docs:
  `docs/fitting_and_optimization.md` §4.9; example `examples/fit_hilbert_regularized.py`
  (denoising, with λ chosen by held-out validation).

- **Per-mode residual weighting** — the fitting layer's residual weight `ω` generalizes from a
  per-order vector to an `ω[mode, order]` matrix, owned by the sampling kind: `probe_model(weight=)`
  takes a length-`d` per-mode vector and `probe_derivatives_model(weight=)` a `(d, order+1)` matrix
  (probe is the only kind with a per-mode axis, so `apply`/`entries` stay per-order). A bare vector
  is still read as per-order, so existing calls are unchanged. Docs:
  `docs/fitting_and_optimization.md` §4.6; example `examples/fit_per_mode_weight_probes.py`
  (inverse-noise weighting of a noisier mode).

- **Newton-CG diagnostics** — `newton_cg(..., verbose=True)` prints a per-iteration block (objective
  and gradient norms, CG statistics, line search, ρ, wall time) plus a per-`(mode, order)`
  relative-error table, with an optional held-out validation column via `val_sample=` / `val_data=`;
  the same records are returned in `stats['diagnostics']` and `stats['history']`, and a regularized
  run splits the objective as `obj = misfit + reg`. Backend-owned
  (`backend.optimizer_display.make_newton_display` + a `callback=` hook), so a raw-`.data` user gets
  the identical display; ragged and uniform. Example `examples/fit_probe_display.py`.

- **Newton-CG warm-start controls** — `g0norm_newton` / `g0norm_cg` pin the reference `‖g₀‖` that the
  Newton stopping test and the CG forcing term are measured against (the computed initial norm is
  misleadingly small after a rank-continuation warm start; `g0norm_newton` also feeds CG unless
  `g0norm_cg` is given), and `cg_forcing_power` (default `0.5`) trades CG iterations per Newton step
  against the number of Newton steps. `NewtonInfo.g0norm` reports the effective reference. Docs:
  `docs/fitting_and_optimization.md` §5.

- **`chunk_size` for the probe-derivative transpose** — the `𝒥ᵀ` variation assemblies are computed in
  chunks over the probe stack `W`, trading recompute for peak memory: `chunk_size` (default `100`,
  `None` = dense) threads through the whole transpose chain and both `T3Tangent` /
  `UT3Tangent.probe_derivatives_transpose`, and the optimizers accept `chunk_size='auto'` (the
  default), which sizes it from `x0`'s shapes. The sizing helpers
  `backend.sampling_derivatives.estimate_chunk_size` / `max_chunk_size_within` are eager (outside
  `jit`), measure the per-row cost by lowering the real kernel rather than estimating it
  analytically, and take an `n_shards=` for per-device sizing and a `stack_shape=` for a batch of base
  points (the frame stack `C` multiplies the assembly but not an absolute byte budget, so omitting it
  on a stacked frame would make `max_chunk_size_within` return a chunk up to `prod(C)` times too
  large). Docs: `docs/chunking.md`.

- **Recurrence/scan jets are now the standard derivative path** — the pushthrough, combine and
  variation-assembly jets are computed by affine two-term recurrences and order-scans rather than by
  contracting the dense binomial `trs` tensor. The lean forms own the canonical names
  (`compute_mu_jets`, `compute_eta_jets`, `compute_{sigma,tau}_jets`, `compute_deta_jets`, the tilde
  twins, and `assemble_{tt,tucker}_variation_jets`); the dense forms remain public as the `*_trs`
  reference twins (numerically equal, and the test oracle). Under `jit` on the uniform layer this is
  a large memory win at unchanged accuracy — measured at rank 128, `W`=32000: ~14–28× less XLA
  temporary for the `eta` jets and 168 GB → 2.63 GB (64×) for the TT-core variation assembly.

### Changed

- **BREAKING: the backend `optimizers.GeometryOps` protocol** gains an optional `precompute` slot
  (`frame -> geometry aux`, `None` for the existing geometries), and `project`/`retract` take a third
  argument (`(frame, variations, aux=None)`). `Problem.local_model` builds the aux once per Newton
  step (a `LocalModel.geom_aux` leaf field beside `sweep`) and passes it to `project`/`retract`; the
  frontend `GaussNewtonModel` mirrors it as a `geometry_aux` leaf. Migration for custom `GeometryOps`:
  accept (and ignore) `aux=None` in `project`/`retract`; the shared geometries use the slot to compute
  their per-frame SVD companion once per local model instead of once per CG matvec.

- **The `Regularizer` protocol threads the geometry aux**: `gradient`/`hessian`/`quadratic` gain an
  `aux=None` parameter (the per-frame geometry companion, e.g. the SF-T3 `SharedFrameData`), and
  every model/`LocalModel` call site passes its stored companion — closing the one seam where a
  regularized *shared* fit rebuilt the companion per CG matvec. Custom `Regularizer`
  implementations: accept (and may ignore) `aux=None`. Design record:
  `docs/contributor/precompute_and_caching.md` (the precompute/caching principle and audit).

- `backend/common.py` gains `prefix_mask` (the boolean prefix indicator shared by every uniform prefix
  structure) and now hosts `require_concrete_masks`, which moved from `backend/ut3_masking.py` — it is
  infrastructure for the uniform *mask-representation contract*, not part of any one object family.

- **`use_jit=True` now converts its inputs to jax instead of silently running eager.** Previously the
  flag was accepted but ignored unless `x0` / `sample` / `data` were already jax arrays, so a "jit"
  run could be an eager one (and a meaningless benchmark). Requesting jit is now taken as opting into
  jax-world precision: `x0`, `problem.sample` and `problem.data` are moved onto jax, the result comes
  back **jax-backed** (float32 unless `jax_enable_x64` is set), and the call **raises** if jax is not
  installed. `use_jit` is also now an explicit keyword on `newton_cg` / `mc_sgd` / `adam` rather than
  arriving through `**kwargs`. Docs: `docs/fitting_and_optimization.md` §4.5.

### Fixed

- **A `GaussNewtonModel` built from a parameterized sampling kind recompiled on every rebuild.** The
  model carries its `SamplingKind` as jax pytree `aux_data`, and jax keys its compilation cache on the
  aux; the derivative kinds and the weighted probe kind are built per model out of fresh closures, which
  under dataclass field equality never compare equal. So the documented "roll your own optimizer"
  pattern — rebuild the model at each outer point, `jit` a function of it — paid a full recompile every
  step (measured: 3 traces for 3 rebuilds, against 1 for the `apply`/`entries`/`probe` singletons).
  `SamplingKind` now compares on a value `identity` (name, order, residual weight, `chunk_size`) rather
  than on its closures, so a rebuilt kind with the same parameters is the same cache key while a
  genuinely different one still gets its own compilation. The uniform model already did this through its
  value-hashed aux; the ragged twin now matches, and both are pinned by regression tests.
- `backend.fv_conversions` ignored `squash_tails=False` — the parameter was shadowed inside the
  function, so the tails were squashed regardless of what the caller asked for.
- The uniform frame/variation entry points (`ufv_apply_frame_masks`, `ufv_apply_variations_masks`,
  `ut3frame_to_t3frame`, `ut3variations_to_t3variations`) accepted **traced** masks and failed deep
  inside jax with a `TracerArrayConversionError`; they now call `require_concrete_masks` and raise the
  actionable message, like the plain-layer entry points already did.

### Removed

- **BREAKING: the ~104 named contraction functions in `backend.contractions`**
  (`WCa_Caib_WCi_to_WCb`-style), replaced by the `contract` interpreter above. Migration is
  mechanical: the function name is the subscripts string (`X_Y_to_Z(a, b)` →
  `contract('X,Y->Z', a, b)`), and the trailing `n_probe` / `n_frame` arguments become
  `len_W=` / `len_C=` keywords. Numerically identical (each named function was verified equal to
  its `contract` call over an empty/single/multi-axis block-shape matrix before removal).
- The old parked weighted layer (`weighted_tucker_tensor_train.py`, `backend/wt3_operations.py`, the
  `wt3_` prefix, and the broken `absorb_weights_into_tangent_cores`), superseded by the above.

## [2026.0.0] — 2026-07-13

The first public release — the initial public surface:

- The **Tucker tensor train (T3) format** — arithmetic with dense-tensor semantics,
  orthogonalization, minimal ranks, T3-SVD, save/load, batching on every operation.
- The three **sampling operations** (`entries` / `apply` / `probe`), their symmetric
  directional derivatives, and the ambient/corewise/tangent transposes.
- The **fixed-rank T3 manifold**: orthogonal frame + gauged variations, tangent vectors,
  gauge projections, retraction, and the `MANIFOLD` / `COREWISE` geometries.
- **Least-squares fitting** from any sampling operation or its derivatives (Gauss-Newton
  models) with four optimizers (`gradient_descent`, `mc_sgd`, `adam`, `newton_cg`).
- The **uniform layer**: supercores + boolean rank masks mirroring the whole stack, for
  `jax.lax.scan` vectorization and compile-once `jit` (optimizers included).
- **NumPy / JAX** backends with dispatch inferred from input array types; **safe mode**
  for numerical-precondition checking.
