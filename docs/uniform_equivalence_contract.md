# The uniform equivalence contract

> The governing correctness principle for the uniform T3 layer, and the most important of the uniform
> design notes. It says what `UniformTuckerTensorTrain` *is* (a faster representation of ragged T3s,
> nothing more), what "correct" means for every uniform operation, and — as a direct consequence — how
> the layer is tested and how the masking is scheduled. As with the other notes, this is the reasoning
> and its honest limits, not a rulebook.

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

## Consequence 1 — it *is* the test strategy

The verified layers check correctness against **dense** ground truth. The uniform layer checks against
**ragged** ground truth, via the round trip:

- convert a ragged T3 (or a stack/tree of them) to uniform, run the uniform op, convert back, and
  compare to the ragged op on the same input — comparing **real parts only**. The cleanest comparisons
  discard the padding automatically: compare `to_dense()`s, or convert back with `ut3_to_t3` and compare
  cores. Property checks (orthonormality, etc.) are made on the real blocks.

Every uniform op — `+`, `·`, `inner`, `norm`, orthogonalizations, `ut3svd`, `entries`/`apply`/`probe`,
`to_dense` — gets the same one-line test against its ragged twin. This is exhaustive and uniform across
the layer.

## Consequence 2 — it resolves the masking schedule

A recurring stuck-point is "apply masks once up front, or re-mask along the way?" The contract dissolves
it: **it prescribes a result, not a schedule.** Any masking schedule whose real parts match ragged is
correct, and the round-trip test certifies it — there is no need to prove the schedule correct a priori.

The schedule we adopt, as the simplest one that satisfies the contract:

- **mask on entry** — zero the garbage so it cannot flow into a real result through a contraction; and
- **re-mask after any step that writes garbage back into the padding** — chiefly an SVD, whose padded
  columns get filled with arbitrary orthonormal completions.

Between those points the data stays put (no compaction — see `docs/uniform_masks_vs_ranks.md`). Whether
this schedule is exactly right in a given op is decided by the round-trip test, not by hand.

## The not-minimal-ranks case

The one genuinely subtle spot. Orthogonalizing a non-minimal core can change the realized rank (an `n×N`
Tucker core with `n>N` has only `N` nonzero singular values). The contract pins the answer down without
guesswork: **do whatever the ragged op does** — keep the rank with an arbitrary orthonormal completion,
or reduce it and update the mask — determined by reading/running the ragged path, then verified by the
round trip. (This is the behavior that was mid-refactor in the copied-in code.)

## Honest scope and limits

- **Only user-facing ops with ragged twins are bound by the contract.** The masking/padding plumbing
  (`make_uniform_masks`, `apply_masks_to_cores`, pack/unpack) is uniform-only — it is the machinery that
  *makes* the contract hold, not something the contract constrains.
- **The uniform layer mirrors only the shape-stable subset of ragged ops.** `rtol`/`atol` truncation has
  no uniform counterpart (data-dependent shapes would break uniformity/`jit`); uniform truncates via
  **max-rank masks**. So the SVD form of the contract is `ut3svd(max-rank mask) == t3svd(max_ranks)`, not
  the `rtol`/`atol` path. This is a deliberate narrowing, not a gap (`docs/uniform_ranks_and_varieties.md`).
- **The contract says nothing about the garbage.** Tests must compare real parts only (via `to_dense` or
  convert-back); a test that compared raw padded supercores would fail for no real reason.

See also `docs/uniform_ranks_and_varieties.md`, `docs/uniform_supercore_layout.md`,
`docs/uniform_masks_vs_ranks.md`, `docs/uniform_pytree_composition.md`, and the running
`docs/uniform_port_plan.md`.
