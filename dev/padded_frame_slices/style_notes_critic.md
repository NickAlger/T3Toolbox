# Backend style: the gold standard (critic's notes)

The precedent is the methods of `TuckerTensorTrain` and `UniformTuckerTensorTrain` and the backend functions they call. I traced:

| method | backend function(s) |
|---|---|
| `zeros`, `randn` | `t3_zeros`, `t3_corewise_randn`; `ut3_zeros` |
| `+` | `t3_add`, `ut3_add` |
| `inner`, `norm` | `t3_inner_product`, `ut3_norm` |
| `to_dense`, `from_t3` | `t3_to_dense`, `t3_to_ut3` |
| `entries`, `probe` | `t3_entries`, `ut3_entries`, `t3_probe` |
| `left_orthogonalize_tt_cores` | `tt_left_orthogonalize`, `ut3_left_orthogonalize_tt_cores` |
| `down_svd_tucker_core`, `t3svd` | `t3_down_svd_tucker_core`, `ut3svd` |
| `resize` | `tucker_change_core_shapes`, `tt_change_core_shapes` |
| `get_minimal_ranks`, `get_core_shapes` | `compute_minimal_ranks`, `t3_core_shapes` |
| `reverse`, `squash_tails` | `tt_reverse`, `ut3_reverse`, `ut3_squash_tails` |

## The rule

1. **Pure functions of basic immutable values.**
   - Inputs: arrays, tuples of arrays, ints, static int tuples.
   - On the uniform layer, `UT3Data = (tucker_supercore, tt_supercore, shape, (tucker_mask, tt_mask))` with host bool masks.
   - Outputs are the same kinds of value. There are no objects, dicts, dataclasses or options in the math.
2. **Structure travels as explicit integers.** `Sequence[int]` with `# len=d` / `# len=d+1` comments: `t3_zeros(shape, tucker_ranks, tt_ranks, stack_shape, use_jax)` and `tucker_change_core_shapes(cores, new_shape, new_tucker_ranks)`. No argument is an array-carrying object read only for its shapes.
3. **Bodies read structure off the arrays they actually use, inline, then follow the math.**
   - `[B.shape[-2] for B in tucker_cores]`, `cores[0].shape[:-2]`, `tk.shape[0]`.
   - Uniform: ranks from `mask.sum(-1)`, output masks from `prefix_mask(ranks, size)`.
   - No helper dictionaries of stores.
4. **The backend is inferred from the arrays.**
   - `tree_contains_jax` → `get_backend`.
   - A `use_jax` flag only where no array is input (constructors).
   - Polymorphic functions dispatch on `is_ndarray(x[0])` and keep the ragged name.
5. **Uniform functions: concrete masks, mask, compute.** They call `require_concrete_masks`, mask, then usually delegate to the ragged kernel.
6. **Frontend methods only validate and rewrap.** They check shape, stack and lengths (plus safe-mode residuals), call one backend function on `self.data`, and rewrap.

## Review checklist

For each new backend function:

- [ ] Arguments and returns are basic values.
- [ ] Structure is passed as `Sequence[int]` with `len=` comments.
- [ ] No argument is used only for its shapes. That is the ruled-out "template frame" of `fv_zero_point_frame` and `_frame_stores`.
- [ ] Structure is derived from the arrays actually used.
- [ ] The backend is inferred from arrays.
- [ ] The wrappers are thin.

Only these modules count as precedent. My r1–r5 acceptance of the template pattern is withdrawn.
