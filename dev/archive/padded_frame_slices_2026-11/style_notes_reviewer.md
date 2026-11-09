# Backend style: notes from the gold standard (reviewer, 2026-11-02)

## Traced

- **Ragged:** `zeros` and `randn` → `t3_zeros` and `t3_corewise_randn`; `resize` → `tucker_change_core_shapes` and
  `tt_change_core_shapes`; `__add__` → `t3_add`; `to_dense` → `t3_to_dense`; `reverse` → `tt_reverse`;
  `left_orthogonalize_tt_cores` → `tt_left_orthogonalize`; `t3svd` → `t3_svd.t3svd`; `get_minimal_ranks` →
  `compute_minimal_ranks`; `entries` → `t3_entries`; `inner` → `t3_inner_product`.
- **Uniform:** `zeros` and `randn` → `ut3_zeros` and `ut3_corewise_randn`; `from_t3` and `to_t3` → `t3_to_ut3` and
  `ut3_to_t3`; `to_dense` → `ut3_to_dense`; `__add__` → `ut3_add` and `ut3_squash_tails`; `reverse` →
  `ut3_reverse`; `left_orthogonalize_tt_cores` → `ut3_left_orthogonalize_tt_cores`; `t3svd` → `ut3svd`;
  `entries` → `ut3_entries`; `minimal_ranks` → `compute_minimal_ranks`.

## What the style is

1. **A backend function is a mathematical map.** It takes the mathematical objects it acts on and returns new ones.
   The data are tuples of arrays: a T3 is `(tucker_cores, tt_cores)`; a UT3 is `(tk, tt, shape, (masks))`.
2. **Structure is passed as structure.** When a function needs a target structure, it takes integer tuples:
   `shape`, `tucker_ranks`, `tt_ranks`, `stack_shape`, or `N=`, `n=`, `r=`. It never takes an object whose values it
   ignores and whose shapes it reads. `resize` is the model: the cores in, the new ranks in, the resized cores out.
   Constructors without array inputs add `use_jax`.
3. **Inputs and outputs are immutable basic types:** arrays, tuples of arrays, ints, static int tuples, host bool
   masks, and scalar options (`rtol`, `sharing` labels). The backend returns no dicts and no mutable containers.
   Derived quantities come from their own functions (`compute_minimal_ranks` → int tuples).
4. **Shapes are read from the data operated on** (e.g. old ranks from `tucker_cores`). They are locals, not
   interface.
5. **The signature reads as the math:** one shape-commented argument per line; dispatch inferred from the arrays.

## Consequences for the padded frame

- **`fv_padded_frame(frame=template, …)` breaks rule 2.** The map is (old frame of `X_old`, target structure,
  hint) → frame, so the target should be integer tuples. The candidates are the four stores `(nU, nD, rL, rR)`, or
  `(shape, tucker_ranks, tt_ranks)` with the stores derived by `ranks.compute_orthogonal_representation_ranks`.
  The dtype should come from the old frame or the hint.
- **`_frame_stores` returning a dict breaks rule 3.** It should be replaced by a function returning int tuples,
  or the stores should be read locally where they are used.
- **Style checks in my reviews.** Each review gets a style section checking every new function against rules 1–5.
  Only these two modules and their backend functions count as precedent.
