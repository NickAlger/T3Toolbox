# The uniform equivalence contract

> The governing correctness principle for the uniform T3 layer, and the most important of the uniform
> design notes: what `UniformTuckerTensorTrain` *is* (a faster representation of ragged T3s, nothing
> more), and what "correct" means for every uniform operation. (How the contract drives the test
> strategy and the masking schedule is contributor material:
> [`contributor/testing_strategy.md`](contributor/testing_strategy.md),
> [`contributor/uniform_internals.md`](contributor/uniform_internals.md).)

---

## The contract

> A `UniformTuckerTensorTrain` is a faster representation of (a stack of) ragged Tucker tensor trains.
> For every operation with a ragged counterpart,
>
> ```
> to_uniform → op_uniform → to_ragged   ==   op_ragged
> ```
>
> **on the masked ("real") parts only.** The padded ("garbage") parts are explicitly don't-care.

That is the whole thing. The uniform layer exists for GPU efficiency and `jit` (no loop unrolling); it
introduces **no new semantics**. Anything a uniform op produces in the real blocks must match what the
ragged op would have produced; anything in the padding is free.

It holds element-wise in the stacked/variety case too (`docs/uniform_ranks_and_varieties.md`): a
varying-rank stack has no single ragged T3, but `to_uniform(tree of ragged T3s) → op → to_ragged` still
equals the tree of per-element ragged results. So the contract and the determinantal-variety view are
fully compatible — "faster ragged" is exactly "a batch of points moved the same way the ragged op would
move each one."

The contract holds on **non-minimal-rank input too**: a uniform op does whatever the ragged op does
there (see the contributor notes for how that case is pinned down and tested).

## The vector I/O boundary — packedness mirror

The sampling ops (`probe` / `apply` / `entries` and their derivatives) take mode-**vectors** across the
ragged/uniform seam, and `probe` returns them too. These can be carried two ways, and the uniform ops
**infer the input's packedness and mirror it in the output** (the same "infer, don't flag" rule as
numpy/jax and ragged/uniform dispatch):

- **ragged in → ragged out** — a `len=d` sequence of real-width vectors. Here the contract holds as
  written: `op_uniform(ragged) == op_ragged`.
- **packed in → packed out** — one supercore-shaped `(d,)+…+(N,)` array (each mode zero-padded to the
  common width `N`). Here the contract holds *after unpacking*: `unpack(op_uniform(packed)) == op_ragged`,
  equivalently `op_uniform(packed) == pack(op_ragged)`.

So the contract is not weakened — it attaches to whichever form crossed the boundary. The packed form is
the **inner-loop path** (the optimizer keeps probe residuals packed end-to-end — no per-matvec
unpack/repack, and `d` stays a single scan-able axis rather than a Python list); the ragged form is the
drop-in "faster ragged" for one-off/interactive use. The user-facing ops (frontend methods included, by
delegation) mirror; the fitting **split-seam** `*_from_sweep` hooks are packed-only (the loop's natural
mode). A backend user gets full parity via the mirroring ops plus the exposed `pack_vectors` /
`unpack_vectors` / `pack_if_ragged` helpers.

**The padding convention is a prefix**, and it is *not* obvious (internal rank edge-vectors scatter; only
the mode/shape padding is a prefix): real data sits in `[0:Ni]`, zeros in `[Ni:N]`, keyed on the `shape`
ints — no mask needed. `pack`/`unpack` encode exactly this, so packed reductions (e.g. `sumsq_over_probes`
over a packed array) are correct because the padding is inert zeros.

**The fill must be finite.** "Garbage don't-care" implicitly assumes finite garbage: masking works by
multiplication, and `0 × NaN = NaN` — padding filled with `NaN`/`inf` ("to be safe") poisons masked
reductions and breaks correctness. Zeros are the robust fill, and packed vectors always travel with
their shape information (the fill is never used to infer shape).

## Honest scope and limits

- **Only user-facing ops with ragged twins are bound by the contract.** The masking/padding plumbing
  (`ut3_make_masks`, `ut3_apply_masks`, pack/unpack) is uniform-only — it is the machinery that
  *makes* the contract hold, not something the contract constrains.
- **The uniform layer mirrors only the shape-stable subset of ragged ops.** `rtol`/`atol` truncation has
  no uniform counterpart (data-dependent shapes would break uniformity/`jit`); uniform truncates via
  **max-rank masks**. So the SVD form of the contract is `ut3svd(max-rank mask) == t3svd(max_ranks)`, not
  the `rtol`/`atol` path. This is a deliberate narrowing, not a gap (`docs/uniform_ranks_and_varieties.md`).
  Relatedly, rank-changing ops (e.g. `ut3svd`) move to the **minimal *structural* ranks** (computable
  from shape + rank structure → static, jit-safe; they only ever shrink), never the **numerical** rank
  (value-dependent, would break `jit` — it is the forbidden `rtol=0`).
- **The contract says nothing about the garbage** — only the masked real parts are meaningful.
- **Orthogonalization at reduced numerical rank completes the prefix with *arbitrary* orthonormal
  vectors.** When a slot's structural rank exceeds its numerical rank, the extra prefix columns are
  legitimate-but-arbitrary orthonormal directions (deterministic per run, not meaningful data) —
  visible in outputs, and expected.
- **The shared-factor companion is guaranteed only for frames the uniform layer built.**
  `ufv_shared_frame_data` re-sweeps the frame's supercores *as stored* (padding included), so it is
  exact for a frame from `ut3_orthogonal_representations`; a frame packed from a ragged one via
  `t3frame_to_ut3frame` is **not** guaranteed (the padded re-sweep can flip sign gauges against the
  unpadded construction). See `docs/contributor/sharing_internals.md`.

See also `docs/uniform_ranks_and_varieties.md`, `docs/uniform_supercore_layout.md`,
`docs/uniform_masks_vs_ranks.md`, `docs/contributor/uniform_pytree_composition.md`.
