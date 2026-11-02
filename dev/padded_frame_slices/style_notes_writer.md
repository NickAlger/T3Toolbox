# Backend style — the gold standard (writer's notes, 2026-11-02)

Traced from `tucker_tensor_train.py` and `uniform_tucker_tensor_train.py`, the only allowable precedent.

**Ragged traces:**

- `zeros` → `t3_zeros`; `randn` → `t3_corewise_randn`
- `resize` → `tucker_change_core_shapes` + `tt_change_core_shapes`
- `__add__` → `t3_add`; `__mul__` → `t3_scale`; `norm` → `t3_norm`
- `to_dense` → `t3_to_dense`
- `apply` → `t3_apply`; `entries` → `t3_entries`
- `t3svd` → `t3_svd.t3svd`
- `down_orthogonalize_tucker_cores` → `t3_down_orthogonalize_tucker_cores`
- `left_orthogonalize_tt_cores` → `tt_left_orthogonalize`
- `get_minimal_ranks` → `compute_minimal_ranks`; `continuation_ranks` → `compute_continuation_ranks`

**Uniform traces:**

- `zeros` → `ut3_zeros`; `randn` → `ut3_corewise_randn`
- `from_t3` / `to_t3` → `t3_to_ut3` / `ut3_to_t3`
- `__add__` → `ut3_add`; `__mul__` → `ut3_scale`; `norm` / `inner` → `ut3_norm` / `ut3_inner`
- `to_dense` → `ut3_to_dense`
- `apply` / `entries` / `probe` → `ut3_apply` / `ut3_entries` / `ut3_probe`
- `t3svd` → `ut3svd`
- `left_orthogonalize_tt_cores` → `ut3_left_orthogonalize_tt_cores`

**The style.**

1. **Pure mathematical functions on immutable basic types:** arrays, tuples of arrays (`(tucker_cores, tt_cores)`),
   ints, static int tuples (`shape`), and host bool masks inside `UT3Data` `(tk_sc, tt_sc, shape, masks)`. No objects, no
   state. No argument is passed only to have its shapes read.
2. **Structure travels as `Sequence[int]`** with a shape comment on its own line: `shape # len=d`,
   `tucker_ranks # len=d`, `tt_ranks # len=d+1` (`t3_zeros`, `compute_minimal_ranks`, `tt_change_core_shapes`). Derived
   structure comes from the int-in, int-out functions of `ranks.py`.
3. **Bodies derive structure from array shapes inline, then follow the math.** `tucker_change_core_shapes`: `old_shape =
   [B.shape[-1] for B in tucker_cores]`, `stack_shape = tucker_cores[0].shape[:-2]`, deltas in a comprehension, then
   `pad_or_truncate`. There is no dict of shapes and no structure-bundling helper; integer checks sit inline.
4. **Dispatch is inferred from the arrays** (`tree_contains_jax` → `get_backend`). The only flag is on array-free
   constructors: `t3_zeros(…, stack_shape=(), use_jax=False)`, "no array inputs, so the flag chooses the output type".
5. **Uniform functions take the masks as data:** `require_concrete_masks` first, and gathers go through the masks
   (`ut3_to_t3`).
6. **Frontend methods only validate and rewrap.** Structural checks always, numerical ones in safe mode, one or two
   backend calls, then `TuckerTensorTrain(tuple(...), tuple(...))`. There is no math in the frontend.
7. **Returns are `Tuple[...]`**, one element per line, each with a shape comment.

**Consequences for the padded frame.**

- `fv_padded_frame(shape, tucker_ranks, tt_ranks, …)` derives its stores with
  `compute_orthogonal_representation_ranks`.
- `padded_structures` returns ints.
- The uniform member builds from ints and masks.
- `fv_zero_point_frame`'s template pattern and `_frame_stores` are not precedent.
