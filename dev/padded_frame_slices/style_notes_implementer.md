# Backend style, read off the gold standard (implementer, 2026-11-02)

**Traced.**
- `TuckerTensorTrain`: `zeros` and `randn` (`t3_zeros`, `t3_corewise_randn`), `__add__` (`t3_add`), `to_dense`,
  `resize` (`tucker_change_core_shapes`, `tt_change_core_shapes`), `entries` (`t3_entries`), `t3svd`,
  `get_minimal_ranks` and `continuation_ranks` (`ranks.*`), `down_orthogonalize_tucker_cores`.
- `UniformTuckerTensorTrain`: `zeros` and `randn` (`ut3_*`), `__add__`, `entries` (`ut3_entries` → `t3_entries`),
  `t3svd`, `down_orthogonalize_tucker_cores`, `minimal_ranks`, `from_t3` / `to_t3`, `validate` / `_from_data`.

**1. Inputs and outputs are immutable basic values.**
- The arguments are arrays, tuples of arrays, ints and `Sequence[int]`. Uniform functions also take a static
  `shape` int tuple and host bool masks.
- Each argument carries an aligned shape comment.
- The return is a fresh tuple, one comment per element. Nothing returns a dict or a record of shapes, and nothing
  takes an object only to read its shapes.

**2. Structure travels as integers when there are no arrays.**
- `t3_zeros(shape, tucker_ranks, tt_ranks, stack_shape, use_jax)` takes the ranks themselves.
- Rank arithmetic is a pure integer function (`compute_minimal_ranks`, `compute_continuation_ranks`) that the caller
  composes.
- A constructor with no array inputs carries `use_jax` and picks the output type.

**3. When arrays are given, structure is derived inline from their shapes.**
- `tucker_change_core_shapes` reads `old_tucker_ranks = [B.shape[-2] for B in tucker_cores]` in the body.
- `t3_add` reads `vsx = tucker_cores_x[0].shape[:-2]`.
- No helper object holds these values.
- Mismatches raise `ValueError` where they are found.

**4. Dispatch is inferred.**
- `use_jax = tree_contains_jax(inputs)`, then `xnp, xmap, xscan = get_backend(is_uniform, use_jax)`.
- `is_uniform = is_ndarray(x[0])` lets one function serve both layers (`t3_entries`, the zippers).

**5. The uniform layer shares the ragged code by masking.**
- `ut3_entries` calls `ut3_apply_masks(data)` and then `entries.t3_entries(masked, index)`.
- `ut3_down_orthogonalize_tucker_cores` masks, calls the shared supercore kernel with host masks, and computes the
  new masks on the host (`np`, `prefix_mask`).
- Masks are `np`; supercores are `xnp`.
- Layers cross only via `t3_to_ut3` / `ut3_to_t3`.

**6. The frontend validates, delegates, and rewraps.**
- A frontend method checks structure (shape, stack, padded `N`), plus safe-mode numerics where the operation
  requires them.
- It then calls exactly one backend function on `self.data` and rewraps the result: `TuckerTensorTrain(*...)`, or
  `_from_data(...)` for uniform.

**What this means for `fv_padded_frame`.**
- The target structure arrives as `shape, tucker_ranks, tt_ranks` (plus `dtype`, `use_jax` when nothing else carries
  them), not as a template frame.
- The four stores of the target frame come from `ranks.compute_orthogonal_representation_ranks`.
- The old stores are read inline from `old_frame`'s cores.
- `_frame_stores`'s dict goes away. Each body computes the counts it needs, inline, as integers.
