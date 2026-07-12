# Uniform layer — implementation internals

Contributor-facing design records for the uniform layer that user docs reference but do not carry.
(Started during the docs user/dev split, S1; the S2/S3 extractions and the P9 polymorphism-lenses
section land here too.)

## The equivalence contract as test strategy

(Excised from the user-facing [`../uniform_equivalence_contract.md`](../uniform_equivalence_contract.md);
the fuller treatment is [`testing_strategy.md`](testing_strategy.md).)

The verified layers check correctness against **dense** ground truth. The uniform layer checks
against **ragged** ground truth, via the round trip: convert a ragged T3 (or a stack/tree of them)
to uniform, run the uniform op, convert back, and compare to the ragged op on the same input —
comparing **real parts only**. The cleanest comparisons discard the padding automatically: compare
`to_dense()`s, or convert back with `ut3_to_t3` and compare cores. Property checks
(orthonormality, etc.) are made on the real blocks. Every uniform op — `+`, `·`, `inner`, `norm`,
orthogonalizations, `ut3svd`, `entries`/`apply`/`probe`, `to_dense` — gets the same one-line test
against its ragged twin: exhaustive and uniform across the layer. (A test that compared raw padded
supercores would fail for no real reason — compare real parts only.)

## The equivalence contract resolves the masking schedule

A recurring stuck-point was "apply masks once up front, or re-mask along the way?" The contract
dissolves it: **it prescribes a result, not a schedule.** Any masking schedule whose real parts
match ragged is correct, and the round-trip test certifies it — there is no need to prove the
schedule correct a priori. The schedule adopted, as the simplest that satisfies the contract:

- **mask on entry** — zero the garbage so it cannot flow into a real result through a contraction;
- **re-mask after any step that writes garbage back into the padding** — chiefly an SVD, whose
  padded columns get filled with arbitrary orthonormal completions.

Between those points the data stays put (no compaction — `../uniform_masks_vs_ranks.md`). Whether
the schedule is exactly right in a given op is decided by the round-trip test, not by hand.

## The not-minimal-ranks case

The one genuinely subtle spot. Orthogonalizing a non-minimal core can change the realized rank (an
`n×N` Tucker core with `n>N` has only `N` nonzero singular values). The contract pins the answer
down without guesswork: **do whatever the ragged op does** — keep the rank with an arbitrary
orthonormal completion, or reduce it and update the mask — determined by reading/running the
ragged path, then verified by the round trip.

## Supercore layout: the polymorphic straddle, and the rejected alternative

(Excised from the user-facing [`../uniform_supercore_layout.md`](../uniform_supercore_layout.md).)

**The polymorphic backends straddle the two layouts.** Functions meant to accept both a ragged
tuple and a uniform supercore (e.g. the shared TT-orthogonalization sweeps) rely on
`supercore[ii]` and `cores[ii]` both yielding "core `ii`." That works precisely *because* `d`
leads in the uniform case — but it means the two input shapes differ by the inserted leading axis,
and a polymorphic function must not assume stack-leading.

**The alternative we rejected:** lay uniform objects out stack-leading,
`stack_shape + (d,) + (per-core axes)`, to match the rest of the library. This reads more
consistently, but `lax.scan` only scans axis 0, so every sweep would either `moveaxis` `d` to the
front (data movement, on every pass) or forgo scanning. Since sweeps are central and the uniform
layer exists for GPU/JIT efficiency, paying a `moveaxis` per sweep defeats the purpose. The
consistency win was not worth the performance and compilation cost.

## Future preconditions on varying-`C` tangent stacks (directive)

(Excised from the user-facing [`../uniform_ranks_and_varieties.md`](../uniform_ranks_and_varieties.md).)
If some specific op ever turns out to *genuinely* need uniform `C` rank, add a **narrow
non-enforced precondition there** (the orthogonal / gauged pattern), never a blanket
construction-time restriction; shape consistency stays the only structural hard error.
*(Decided 2026-06-30, after an earlier draft of the varieties note over-restricted tangents to
uniform `C`.)*

## Optimizer design rule: masks as loop-invariant state

(The design rule behind the library's own uniform optimizers — MC-SGD, Newton-CG, … — executed in
the U1–U6 uniform-optimizers build; the user-facing recipe is `../uniform_backend_jit_recipe.md`.)

The backend optimization functions are **designed** so the masks are **loop-invariant state,
recomputed only at rank-continuation stage boundaries** (where a recompile is correct and rare),
while the per-step jitted kernel is pure supercore work with masks as fixed constants. The
frontend's value-hashed mask holders (`uniform_pytree_composition.md`) give the OO path this
cache-stability automatically; the backend gets it by **object reuse**. (If the finest separation
is ever wanted, `backend/fv_conversions.t3_orthogonal_representations` already returns *just the
cores*, mask-free, so a kernel could orthogonalize supercores inside and attach held masks
outside — but the bundled `ut3_orthogonal_representations` inside a close-over kernel is already
recompile-free, so this is optional.)
