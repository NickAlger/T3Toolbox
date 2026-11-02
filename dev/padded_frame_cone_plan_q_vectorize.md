# Design question: vectorize `ufv_padded_frame` with masks?

**Question** (maintainer, via the coordinator, 2026-11-02). Is there a *simple* polymorphic, masked design that
vectorizes the padded-frame construction over a uniform stack, including stacks with varying ranks?

- **If yes:** adopt it for all stacks.
- **If no:** drop vectorization for uniform- and nonuniform-rank stacks alike, and keep the host loop.

The precedent to measure against is `t3_orthogonal_representations` (`backend/fv_conversions.py:123-267`). Probes,
read-only, are in the session scratchpad: `probe_vec_cost.py`.

**Verdict under the rule: drop it.** A masked design exists. It is jit-clean and contains no data-dependent control
flow, but it is not simple:

- The precedent threads one structure; this construction threads three: the template, the old point and the hint.
- It needs two new masked primitives and masked rewrites of five steps.
- At realistic sizes it is slower than the host loop, not faster.
- No library consumer stacks it: the optimizers are unstacked.

So the host loop stays for every uniform stack, and O-5 moves from "deferred" to "rejected".

## Findings

**(a) Feasibility: yes, and jit-clean.**

- Every count can come from host masks: per-element stores from `compute_orthogonal_representation_ranks` on `(d,)+C` arrays, as in the precedent's `:191-215`.
- Permutations are constant index arrays built from host masks.
- Householder completion has no branches; the sign fixes are elementwise.
- The only branches are static ones on padded shapes.
- Masks stay `np` and supercores `xnp`.

Two corrections to the orchestrator's sketch:

- **(1) `pad_safe_svd` is not needed inside `pinned_leading_columns`.**
  - The guarantee comes from the final pad-safe QR. Move pad rows to the trailing pivots and order the columns `[old real | new real | pads]`. Householder then keeps every output column orthonormal, orthogonal to the pins, and bitwise zero on padded rows, whatever the seeds are.
  - So a plain batched SVD, with its padded rows zeroed, is enough to supply the leading directions. Its σ > 0 vectors are exactly real-supported; its σ ≈ 0 vectors are arbitrary anyway.
- **(2) No output scatter is needed.** With the QR input in the column order above, its output columns `[s_e, r_e)` already sit in the output's prefix slots. The per-element work is the *input* permutation (one `take_along_axis` per site).

**(b) Size: two new masked primitives and five masked step rewrites.**

- **New primitives:**
  - a pad-safe QR (`pinned_completion`), about 30 lines;
  - masked `pinned_leading_columns`, about 15 lines.
- **Masked rewrites of five steps:**
  - the Tucker step, including a group's column concatenation;
  - the pinned left sweep (the right sweep is its mirror);
  - the down step, with its "new Tucker slots" restriction as a per-element mask;
  - the old-block embedding: pad to the template's sizes, then a `where` copy;
  - the hint-free completion: nested Haar draws are fine (draw `kU_max` rows, element `e` takes the first `kU_e`), but the outside-first coordinates become per-element one-hot constants, plus the in-block completion.
- **Reused:** the hint's own down-, left- and right-orthogonalizations have masked twins, used as the precedent uses them (`row_mask` / `col_mask` / `out_mask`, `pad_masks`). The zippers are plain contractions on zero padding.
- **Mask bookkeeping:**
  - three store recurrences (template, old, hint);
  - a product mask for every reshaped unfolding: `(rL ⊗ nU)` and `(nU ⊗ rR)` for the sweeps, `(rL ⊗ rR)` for the down step. These are gappy when flattened.
  - a new-slot mask per site.
- **Estimate:** 250–400 lines on top of the ragged function, plus a test for each masked step: bitwise-zero padding, NaN padding, and equality with the ragged result up to rounding and gauge.
- **The host loop:** about 50 lines in the `ufv_zero_point_frame` pattern (`ufv_operations.py:302-354`): unpack, call the ragged function, repack, check the masks.

**(c) Readability suffers.**

- The precedent stays readable because a 45-line preamble computes one structure's masks and four existing masked helpers do the work.
- Here every pinned step takes four or five masks: row product, pin count, new slots, hint columns and output.
- The eight steps of the ragged algorithm (Appendix B is about 80 lines of dense numpy) would each gain a mask preamble and an `if uniform` branch.
- The mathematics, "leading directions of the content that is new relative to the old point", would be buried under slot arithmetic.

**(d) Correctness traps.**

- **Degenerate hints.**
  - The masked route completes from different arbitrary bases than LAPACK does, so the two layers differ when a hint is degenerate.
  - The frames are still valid, but their directions differ.
  - T4(a)'s bitwise equality would weaken to "valid, and the same subspaces when the hint is non-degenerate".
  - The host loop gives bitwise equality for free.
- **Ties at a cut.** The same issue, already accepted (R5).
- **Gappy hint masks.** A hint from `utv_to_ut3` has gappy doubled-rank masks such as `[1 1 0 | 1 0 0]`.
  - The first orthogonalizations must take them as input masks; the hint's chain recurrence runs on mask sums.
  - The variation-mask gauge shift stays inside `tangent_to_t3`.
- **Boundary bonds.** The precedent's tail squash and its `ttm_sq` prefix-1 boundary masks have to be reproduced.
- **Sharing.** Group ranks are equal within an element but vary across elements.
  - The concatenation of members' normal content then carries gappy column masks.
  - That is workable but is one more mask.
- **`C` only.** `K` would be a larger stack, as on the ragged layer.

None of these blocks the masked design; together they are the "complicated" part.

**(e) Performance: the loop wins at realistic sizes.**

Measured with numpy on CPU, |C| = 4, real blocks with ranks 30–50 and padded to the maximum:

| site | host loop, LAPACK on real blocks | batched `pad_safe_svd` | one element: LAPACK vs pad-safe |
|---|---|---|---|
| sweep, `(rL·nU) × zr` | 46 ms | 491 ms | 27 ms vs 87 ms |
| Tucker step, `w × (zl·zr)` | 81 ms | 372 ms | 28 ms vs 76 ms |

- **The leaner route doesn't fix it.** Plain SVD plus a pad-safe QR removes the roughly 3× pad-safe factor but not the padding inflation: every element pays the largest element's size.
- **The real consumer is unstacked.** `newton_cg` runs with |C| = 1, where there is nothing to vectorize.
- **The construction is cheap anyway.** It runs once per pass, 4–11 passes per level, which is small next to a level's hundreds of Hessian applications.
- **jit doesn't change this.** It would compile once per mask pattern, i.e. once per level, for a function called a handful of times.

**(f) Naming and placement.**

- **A polymorphic design** would have one unadorned `fv_padded_frame` that dispatches with `is_ndarray`, and no `ufv_` twin. That follows the naming rule's "Polymorphism: the ragged name is the polymorphic name" (`naming_conventions.md:53-61`). It would live in `backend/padded_frames.py` and import the uniform orthogonalization helpers, which keeps the import graph acyclic.
- **With the host loop** the pair stays `fv_padded_frame` / `ufv_padded_frame` in `padded_frames.py`. Here the `u` prefix means exactly what the doc says it signals: uniform-specific logic (unpacking through masks).

## Recommendation

Keep the host loop for all uniform stacks, with uniform and varying ranks treated alike, and record vectorization as
rejected.

The construction's natural precedent is `ufv_zero_point_frame`, the zero-tensor, hint-free case of this same
construction. It is a host-side, eager unpack/ragged/repack, and it does not even accept stacks.
`t3_orthogonal_representations` is the right precedent for a sweep the optimizers run inside jit at every iteration.
This construction runs eagerly a handful of times per level.

If a stacked consumer ever appears, the cheap route is not masks. Group the stack's elements by structure (host
`np.unique` over the per-element stores) and run the already `C`-vectorized ragged constructor once per group. That
fully vectorizes a uniform-rank stack and costs about 15 lines. It does use rank integers, so it is outside the
maintainer's masked principle; it is only a note for the future, not a recommendation now.

## What the plan would change

- **§3.9, `ufv_padded_frame` bullet.** "Vectorizing the uniform-rank special case is deferred (O-5)" becomes "a
  host loop by decision, for uniform- and varying-rank stacks alike". Give the reasons: three interacting structures,
  slower at realistic sizes, an unstacked consumer, and bitwise equality with the ragged construction for free.
- **O-5.** Move it from the open questions to `deferred_and_rejected.md` "Rejected": "a masked polymorphic
  `fv_padded_frame` (feasible and jit-clean, but two new masked primitives, five masked steps, three structures'
  masks, and slower than the host loop at ranks 30–50); the cheap route for a future stacked consumer is grouping by
  structure."
- **The internals page.** Add one paragraph with the measurement above.
- **Unchanged:** T4(a) (bitwise real blocks on identical inputs), T4(d) (per-element ranks equal per-element calls),
  and the R7 cost note.
