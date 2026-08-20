# Changelog

All notable changes to T3Toolbox are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/); versions are `YYYY.MINOR.PATCH`.

## [Unreleased]

### Added

- **Shared Tucker factors (SF-T3)** — optimize over Tucker tensor trains whose Tucker factors are
  constrained equal within user-specified groups of modes (the SF-ETT decomposition of Molozhavenko &
  Rakhuba (2026), generalized from one trailing block to an arbitrary partition; the partition is
  always user-provided). A `sharing` spec is one hashable group label per mode, e.g. `(0, 0, 1)`;
  a shared T3 is an ordinary `TuckerTensorTrain` whose group factors are equal (redundant storage —
  a compute-not-memory feature). Ragged layer (the uniform mirror is in progress):
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
    `ufv_mean_tucker_variations` (mask-and-delegate to the polymorphic ragged solves), and
    `shared_data=` threading through `utv_orthogonal_gauge_projection` / `utv_to_ut3` (the TIED
    doubled-rank embedding: `Udot` at every group mode, the companion's centers replacing the down
    cores, the variation block rebuilt at the up width) / `utv_retract` (tied embedding + the grouped
    `ut3svd`). Verified gauge-invariantly against the ragged twins (dense tangents/points at
    machine precision; outputs exactly tied) — the shared uniform geometries and fitting path land
    next.
  - Backend surface in `backend.sharing` (`validate_sharing`, `t3_sharing_residual`,
    `t3_tucker_factors_shared`, `t3_share_tucker_cores`, `T3SharedFrameData` +
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

### Changed

- **BREAKING: the backend `optimizers.GeometryOps` protocol** gains an optional `precompute` slot
  (`frame -> geometry aux`, `None` for the existing geometries), and `project`/`retract` take a third
  argument (`(frame, variations, aux=None)`). `Problem.local_model` builds the aux once per Newton
  step (a `LocalModel.geom_aux` leaf field beside `sweep`) and passes it to `project`/`retract`; the
  frontend `GaussNewtonModel` mirrors it as a `geometry_aux` leaf. Migration for custom `GeometryOps`:
  accept (and ignore) `aux=None` in `project`/`retract`; the shared geometries use the slot to compute
  their per-frame SVD companion once per local model instead of once per CG matvec.

- `backend/common.py` gains `prefix_mask` (the boolean prefix indicator shared by every uniform prefix
  structure) and now hosts `require_concrete_masks`, which moved from `backend/ut3_masking.py` — it is
  infrastructure for the uniform *mask-representation contract*, not part of any one object family.

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
