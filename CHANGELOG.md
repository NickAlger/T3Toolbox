# Changelog

All notable changes to T3Toolbox are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/); versions are `YYYY.MINOR.PATCH`.

## [Unreleased]

### Added

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
