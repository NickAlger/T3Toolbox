# Backend separation refactor

## Goal

Honor the library's frontend/backend split: **all non-trivial math and structural-array logic lives in
the purely-functional backend**; the validated frontend modules (`tucker_tensor_train.py`,
`basis_variations_format.py`, `manifold.py`) are thin wrappers that delegate and repack.

This refactor moves the logic that has crept into the frontend back where it belongs. Every move is
**behavior-preserving**: each frontend method keeps its public name and signature but becomes a thin
delegate, so the existing test suite (150 passing) is the regression guard.

## Exceptions kept in the frontend (by decision)

- **`.validate()` methods** — `TuckerTensorTrain.validate`, `T3Basis.validate`, `T3Variations.validate`,
  `T3Tangent.validate`, **and `check_bv_pair`** (the third validator in `T3Tangent.validate`). Their
  code is a deliberate, readable specification of each class's structural invariants.
- **Borderline items** — `allclose` (T3Basis / T3Variations / T3Tangent; only a tolerance comparison is
  inline, norms are delegated), `T3Tangent.norm` (mirrors `corewise_norm`), `bv_to_t3` (orchestrates
  backend calls), `TuckerTensorTrain.from_canonical` (shape-derivation guard), `*.size` / `*.data_size`.
- **`tucker_tensor_train.py` is already clean** — every method delegates to the `ragged_*`/`t3_*`
  backends. The only exception touched here is the shared stack-sum axis-normalization (item 12).

## Backend homes

| module | gets |
|---|---|
| `backend/ranks.py` (existing) | manifold-dimension formula; basis minimal-rank logic |
| `backend/orthogonal_representations.py` (existing) | basis orthogonality + consistency residuals |
| `backend/tangent_operations.py` (existing) | tangent gauge residual |
| `backend/bv_operations.py` (**new**) | T3Variations constructors + vector unflatten |
| `corewise.py` (existing) | stack-aware corewise scale + stack-aware corewise sum |
| `backend/common.py` (existing) | core-family `.npz` serializer |

## Moves

| # | frontend (current) | → backend fn | frontend becomes |
|---|---|---|---|
| 1 | `T3Basis.is_orthogonal` (bvf) — einsum grams + max-abs-dev | `orthogonal_representations.basis_orthogonality_residual(data) -> float` | `residual(self.data) <= atol` |
| 2 | `T3Tangent.is_gauged` (manifold) — einsum gauge residual | `tangent_operations.gauge_residual(basis_data, var_data) -> float` | `residual(...) <= atol` |
| 3 | `T3Basis.is_consistent` (bvf) — dense reconstruct + norm | `orthogonal_representations.basis_consistency_residual(data) -> float` (relative norm) | `residual(self.data) <= rtol` |
| 4 | `manifold_dim` (manifold) — dim formula loops | `ranks.compute_manifold_dim(shape, tucker_ranks, tt_ranks) -> int` (calls `compute_minimal_ranks` directly, no frontend hop) | delegate |
| 5 | `T3Basis.has_minimal_ranks` (bvf) — rank equality + L/R, U/D redundancy | `ranks.basis_has_minimal_ranks(shape, up, down, left, right) -> bool` | delegate |
| 6 | `T3Variations.from_vector` (bvf) — offset-arithmetic unflatten | `bv_operations.variations_from_vector(flat, variation_shapes, stack_shape) -> (V, H)` | delegate + wrap + `to_jax` |
| 7 | `T3Variations.zeros` (bvf) | `bv_operations.zeros_variations(variation_shapes, stack_shape, use_jax) -> (V, H)` | delegate + wrap |
| 8 | `T3Variations.randn` (bvf) | `bv_operations.randn_variations(...) -> (V, H)` | delegate + wrap |
| 9 | `T3Variations.unit` (bvf) — zeros + indexed set | `bv_operations.unit_variations(variation_shapes, index, stack_shape, use_jax) -> (V, H)` | delegate + wrap |
| 10 | `T3Tangent.zeros` / `randn` (manifold) — build cores inline | (via #7/#8) | delegate to `T3Variations.zeros`/`randn` (randn keeps the existing gauge-projection call) |
| 11 | `T3Tangent.normalized` (manifold) — per-stack reshape/broadcast | `corewise.corewise_stack_scale(X, s) -> X` (pads `s` over each leaf's trailing axes) | `corewise_stack_scale(var.data, 1.0/self.norm())` + wrap |
| 12 | stack-sum axis-normalization: `TuckerTensorTrain.sum_stack_corewise`, `T3Variations.sum_stack`, `T3Tangent.sum_tangents` | `corewise.corewise_stack_sum(X, axis, n_stack) -> X` (normalizes `None`/int/neg axes, then `corewise_sum`) | delegate + wrap |
| 13 | `save` / `load`: `T3Basis`, `T3Variations`, `T3Tangent` — `'f%d_%d'` npz | `common.save_core_families(file, families)`, `common.load_core_families(file) -> families` | delegate (frontend repacks into the class + `to_jax`) |

Note on #13: keeps the existing `'f%d_%d'` key scheme, so saved files stay compatible.
`TuckerTensorTrain.save`/`load` use a different (pre-existing) scheme and are **not** touched.

## Cross-class reuse rules (do NOT borrow the Tucker backends)

- ❌ `t3_from_vector`, `t3_zeros`, `t3_corewise_randn` are `(shape, ranks)`-driven Tucker+TT layouts — they
  do **not** fit the variations' direct 2-family shape-list layout. Items 6–9 get their own `bv_operations`
  functions instead.
- ❌ `t3_sum` / `t3_sum_stack` are **dense, rank-growing** — variations/tangent need the *corewise* sum
  (item 12 stays on `corewise_sum`).
- ✅ Safe (layout-agnostic, already used correctly): `t3_to_vector`, `reverse_tt`, `corewise_*`.

## Follow-ups (deferred — address AFTER this refactor is finished)

These were surfaced while doing the refactor but are **out of scope** here (this refactor is strictly
behavior-preserving — code is moved as-is). Do not fix them as part of the backend-separation commit.

- **(A) numpy/jax dispatch drift — raw `np.*` calls. ✅ RESOLVED.** The principle (in `CLAUDE.md`):
  infer the backend from the inputs — `use_jax = tree_contains_jax(inputs)`,
  `xnp, _, _ = get_backend(False, use_jax)`, then `xnp.*` — **if any input arg contains jax → jax,
  else numpy**; and **every numpy/jax call site needs a `tests/test_dispatch.py` case** (reference:
  `backend/probing.py`). What was done:
  - the residual/checker backends — `orthogonal_representations.basis_orthogonality_residual` /
    `basis_consistency_residual` and `tangent_operations.gauge_residual` — rewritten to dispatch via
    `xnp` and be **jit-able** (collect per-core scalar deviations → `xnp.stack` → `xnp.max`; no
    `float()`/Python `max()` that would coerce to numpy or break jit). The frontend checkers
    (`is_orthogonal`/`is_consistent`/`is_gauged`) wrap the threshold in `bool()` to keep a clean
    Python-`bool` contract. Added to the `test_dispatch` jit bucket; numpy ≡ jax verified.
  - static shape products `np.prod(...)` → `math.prod(...)` (house style) in
    `bv_operations.variations_from_vector`, `*.size` (`TuckerTensorTrain`/`T3Basis`/`T3Variations`/
    `T3Tangent`).
  - `t3_linalg` inner-product zipper: `np.ones` → `xnp.ones` (added a `TuckerTensorTrain.inner`
    dispatch test, previously uncovered).
  - **Left numpy by design:** `common.save_core_families`/`load_core_families` (file I/O — a saved
    `.npz` is always concrete numpy), `ranks.py` `np.minimum` (integer rank arithmetic, no array
    data), `t3_operations` `np.broadcast_shapes` (tuple shape arithmetic), `bv_operations.unit_variations`
    `np.zeros` (pure constructor that builds numpy then `to_jax`s per the constructor pattern).

- **(B) `sum`-axis normalization convention mismatch.** The new `corewise.corewise_stack_sum` (item 12)
  normalizes a negative `axis` as `(axis + n_stack)` and leaves out-of-range axes to error in the
  reduction — matching `TuckerTensorTrain.sum_stack_corewise`'s existing convention. The pre-refactor
  `T3Tangent.sum_tangents` used `axis % k`, which **silently wraps** out-of-range axes. Identical for
  all valid axes; differs only on invalid input. The refactor adopts the `corewise_stack_sum` (no-wrap)
  convention library-wide. **Discuss after the refactor:** confirm we want the no-wrap behavior
  everywhere (recommended — an out-of-range stack axis is a structural error and should raise).

## Validation

- Full suite (`tests/`, 150) must stay green — these are behavior-preserving moves.
- Spot-check each new backend fn's result against the pre-refactor frontend result for one stacked +
  one unstacked case (residuals, dim, constructors, normalized, sums, save/load round-trip).

## Execution

One commit. Order within it: new `bv_operations.py` + the `ranks`/`orth_reps`/`tangent_operations`/
`corewise`/`common` backend additions first, then rewire each frontend method to delegate, then run the
suite.
