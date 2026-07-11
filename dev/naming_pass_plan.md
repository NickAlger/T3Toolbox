# Naming pass + backend module reorg — execution plan (Slice 0 artifact)

_Written 2026-07-10. Status: **EXECUTED 2026-07-11** (all slices, suite-gated per sub-slice, full
suite green throughout: 593 tests / 40,215 subtests). Execution deltas vs the tables below:
sampling landed as **option C** (by-type files `apply`/`entries`/`probing` + one
`sampling_derivatives.py`; the c1–c4 *function* renames applied as written); F1 landed as Nick's
**tt/utt** family (`tt_operations.py` + `tt_orthogonalization.py`, with the polymorphic merges
`tt_reverse` and `tt_squash_tails`; `utt_` reserved, no members); F7 became moot under option C
(uniform sampling keeps plain+derivatives together); everything weighted untouched. The user-facing
catalog is **`docs/naming_conventions.md`**. Companion decisions log: `dev/naming_review.md`._

## 0. Scope decisions (settled with Nick, 2026-07-10; amended 2026-07-11)

0. **Governing principle (Nick, 2026-07-11): the renaming exists to help the user** — orient in the
   codebase, find the function they want, and understand what it takes and returns. **When helping
   the user conflicts with a convention, the user wins, every time.** Apply judgment, not dogma.
   Corollary: parameter names may deliberately encode representation (`xx` = ragged collection,
   `uxx` = uniform collection; `x` = t3 data, `data` = ut3 data) — this is information for the
   user, not drift; keep it.
1. **File-level backend reorg: YES** (the §4 family×op-kind matrix). **Polymorphism triage: DROPPED**
   — the uniform layer shipped closed with explicit `ut3_`/`ufv_` twins, all verified;
   `ut3_sampling` is repaired and load-bearing (the §4 note calling it a buggy stopgap to eliminate
   is stale). The one already-polymorphic module (`backend/orthogonalization.py`, chain-level) stays
   polymorphic; nothing else is merged. Recorded as a possible post-1.0 evolution only.
2. **Function-name normalization to the §1 prefix grammar: YES**, folded into the reorg
   module-by-module (one import churn for backend users, not two).
3. `has_jax` → **`jax_available`** (the module-level "jax is importable" flag; the ambiguous
   *predicate* `has_jax` is already gone — `contains_jax`/`tree_contains_jax` are the survivors).
4. **`base-inner` → `frame-inner`** (docs-only): fold into Slice 4.
5. `Sequence`→`Union` hint relaxation: **deferred** (out of scope).
6. **Weighted layer: FULLY untouched — no exceptions** (Nick, 2026-07-11: it may be revived in a
   later version). `wt3_operations.py`, `weighted_tucker_tensor_train.py`, the parked
   `absorb_weights_into_tangent_cores`, **and** the apparently-dead `wt3_squash_tails` copy inside
   `t3_operations.py` all stay exactly as they are. All `w*`-prefixed names are out of scope for
   every rename in this pass.
7. `OLD_orthogonalization.py`: **deleted** (Nick confirmed superseded). Remaining `OLD_test_*.py`
   files stay for R6.
8. **No deprecation shims/aliases** — pre-release ("DO NOT USE" banner is still up), clean breaks.
9. **`dev/archive/` and `OLD_*` files are never rewritten** by any rename (historical records).

---

## A. Cross-class method matrix — dispositions

Extracted by AST over the seven parallel pairs (script: scratchpad `extract_matrix.py`; rerun at
execution). Geometries (`ManifoldGeometry`↔`UniformManifoldGeometry`,
`CorewiseGeometry`↔`UniformCorewiseGeometry`) and the GN models are **already fully name-aligned**.

### A1. Renames to execute (Slice 1)

| Where | Old → New | Why |
|---|---|---|
| `TuckerTensorTrain` | `.squash()` → `.squash_tails()` | Locked §3. Uniform + weighted twins already say `squash_tails`; ragged is the lone holdout — and ragged's own `to_dense(squash_tails=...)` kwarg already uses the full name. |
| `backend/common.py` (+31 refs) | `has_jax` → `jax_available` | Kills the "environment has jax" vs "object contains jax arrays" confusability. |

~~`stack(uxx)` → `stack(xx)`~~ **withdrawn** (2026-07-11): `uxx` deliberately tells the user the
stacked objects are uniform — representation-encoding parameter names are a feature (§0.0), kept
library-wide.

### A2. Same-op-different-name pairs that are OK BY DESIGN (do NOT merge)

The uniform classes encode the **operand type** in converter names: `*t3*` = cross-layer (ragged)
converters, `*ut3*` = same-layer. So `T3Frame.from_t3/to_t3` ↔ `UT3Frame.from_ut3/to_ut3`,
`T3Tangent.to_t3` ↔ `UT3Tangent.to_ut3`, plus the cross-layer `from_t3frame`/`to_t3frame`,
`from_t3variations`, `from_t3tangent` family. Force-merging would create real ambiguity
(`UniformTuckerTensorTrain.from_t3` already means "from *ragged*"). Document the convention in the
doc pass instead. Likewise `UniformGaussNewtonModel.kind` (lazy value-hashed rebuild) is deliberate
U7b design, not drift.

### A3. Gaps registry — INTENDED behavior (Nick, 2026-07-11)

These asymmetries are **intended for this version — several intended generally** (design, not
TODOs). Recorded so the doc pass documents them as deliberate; nothing here is filled or "fixed".

- `UT3Tangent` lacks `save`/`load`, `to_vector`/`from_vector`, `size`/`data_size` (ragged has them).
- `T3Variations` lacks `structure`; `UT3Frame` lacks `size`/`data_size`.
- Signature capability gaps (representation-inherent; document, don't unify):
  ragged `sum_stack(axis)` vs uniform `sum_stack()`; ragged `to_dense(squash_tails=)` vs uniform
  `to_dense()`; ragged `left/right_orthogonalize_tt_cores(return_variation_cores=)` vs uniform
  without; ragged `t3svd(rtol, atol, ...)` vs uniform mask-based; ragged
  `project_ambient(..., method=)` vs uniform without; constructor signatures differ (masks/shape).
- `has_minimal_ranks`/`minimal_ranks`: `cached_property` (ragged) vs `property` (uniform) — leave.
- Backend cross-layer mirror asymmetries recorded: `t3_sum(x, axis)` ↔ `ut3_full_sum(data)`
  (uniform supports all-modes only — name is capability-honest, keep); `t3_norm(use_orthogonalization=)`
  ↔ `ut3_norm_orthogonalized` (same reason, keep).

### A4. Same-looking, different-math watchlist (verify-before-merge, and markers that must survive)

Near-twin names whose *math* differs — the rename pass treats these as guarded: never unify, never
drop the distinguishing marker, and any future "align these two" proposal starts by reading both
bodies + docstrings.

- **`sum_stack` vs `sum_stack_corewise`** (and backend `t3_sum_stack` vs `corewise_sum`): genuine
  tensor sum of the represented stack (**rank-growing** — the S-fold `t3_add`) vs elementwise sum of
  the core arrays (**rank-preserving**, NOT the tensor sum in general — valid where cores carry
  additive structure, e.g. stacked variations). The docstrings already cross-warn; keep it that way.
- **`t3_sum(x, axis)` vs `t3_sum_stack(x, axis)`**: near-identical names, different objects summed —
  modes of the *represented tensor* vs *stack elements*.
- **`retract` vs `corewise_retract`** (uniform tv module): manifold retraction vs additive core
  update — different geometry, both legitimately "retract". Ragged `tangent_operations` has only the
  manifold one (the corewise retraction is served by the frontend geometry) — an asymmetry to keep.
- **The inner-product family**: `t3_inner_product_t3`/`ut3_inner_product` (Hilbert–Schmidt, of the
  *represented tensors*) vs `corewise_dot`/`ufv_corewise_inner` (Euclidean *coordinate* metric) vs
  `MANIFOLD.inner` (HS via gauged variations, precondition-checked). "corewise" is a **semantic
  marker** (coordinate-space math, never HS) — it is never added or dropped by any rename.
- **`fv_to_t3` vs `tangent_to_t3`** (→ `tv_to_t3`): indexed *single term* vs efficient *sum of all
  terms* — the locked grammar's own founding example (§1).
- **`t3_norm(use_orthogonalization=)` vs `ut3_norm_orthogonalized`**: same math, different
  capability surface; kept unmerged (A3).
- **`has_minimal_ranks` vs `has_numerically_minimal_ranks`**: structural integer arithmetic vs an
  SVD — the catalog's own naming split; preserved everywhere.
- **`t3_corewise_randn` (and `ut3_randn` → `ut3_corewise_randn`)** vs `MANIFOLD.randn`: iid core
  entries vs a genuine tangent-space Gaussian — "corewise" is load-bearing here too (F15).

---

## B. Module-cut map (current → target)

Op-kind columns: constructors · conversions · operations · linalg · orthogonalization · svd ·
sampling · sampling-derivatives · masking (uniform-only). Families: `t3`/`tv`/`fv` (+`u`). Infra
stays unprefixed outside the matrix.

### Unchanged (infra + already-conforming)

`common.py` (keep as the one infra catch-all — ⚑F9), `contractions.py`, `stacking.py`, `linalg.py`,
`ranks.py`, `fitting.py`, `uniform_fitting.py`, `ut3_constructors.py`, `ut3_conversions.py`,
`ufv_conversions.py`, `wt3_operations.py` (frozen).

### Ragged side

> **⛔ SAMPLING GROUPING ON HOLD (Nick, 2026-07-11): no sampling file moves without sign-off.**
> Nick prefers grouping by **sampling type** (entries / apply / probing / derivatives) over the
> by-family cut below — the t3 and tv algorithms per type are closely connected and presented
> together in the paper. Leading candidate ("option C", matching his phrasing): keep
> **`apply.py` / `entries.py` / `probing.py`** (each holding its type's t3 + tv + dense ops,
> transposes, sweeps — i.e. tangent-apply/entries + their sweeps move OUT of `probing.py` into
> their type files) **+ `sampling_derivatives.py`** (`probe_derivatives.py` renamed honestly — all
> jets stay together since the jet machinery is shared across types). The **function renames in
> §C.c1–c4 stand regardless of grouping** (the `t3_`/`tv_`/`dense_` prefixes carry the family axis
> within whatever file layout wins — under by-type they become *more* load-bearing, not less).
> Uniform sampling files (`ut3_sampling`, `ufv_sampling`→`utv_sampling`): default is keep the
> family cut (small files) and document the asymmetry; open to by-type mirroring. The four
> by-family rows below are retained as the alternative until the discussion settles:

| Current | Target (by-family alternative — on hold) | Notes |
|---|---|---|
| `apply.py`, `entries.py`, `probing.py` (t3 part) | **`t3_sampling.py`** (new) | The §4 sampling cut. `apply.py`/`entries.py` dissolve (2 functions each). |
| `probing.py` (tangent part — the bulk) | **`tv_sampling.py`** (new) | |
| `probe_derivatives.py` (t3 part) | **`t3_sampling_derivatives.py`** (new) | |
| `probe_derivatives.py` (tangent part) | **`tv_sampling_derivatives.py`** (new) | |
| `tangent_operations.py` | **`tv_operations.py`** | rename + function prefixes |
| `orthogonal_representations.py` | **dissolved** | `orthogonal_representations` → `fv_conversions.py` (mirrors uniform: `ut3_orthogonal_representations` lives in `ufv_conversions`); the two frame residuals → `fv_operations.py` (mirrors `ufv_frame_orthogonality_residual` in `ufv_operations`). |
| `fv_conversions.py`, `fv_operations.py` | keep names | function renames inside; parked `absorb_weights_into_tangent_cores` untouched in place. |
| `orthogonalization.py` | keep (polymorphic chain infra, unprefixed = correct per grammar) | functions → `*_tt_chain` to kill the 3-way collision (⚑F1). |
| `t3_operations.py` | **3-way split**: `t3_constructors.py` + `t3_conversions.py` + `t3_operations.py` (slimmed) | conforms ragged to the uniform template (⚑F8). |
| `dense_t3svd.py` | **merged into `t3_svd.py`** | dense→t3 SVDs are the t3 family's from-dense entry points (⚑F5). |
| `t3_linalg.py`, `t3_orthogonalization.py`, `t3_svd.py` | keep names | function renames inside. |

### Uniform side

| Current | Target | Notes |
|---|---|---|
| `ut3_sampling.py` | `ut3_sampling.py` + **`ut3_sampling_derivatives.py`** (new, 6 fns) | column symmetry with ragged (⚑F7). |
| `ufv_sampling.py` | **`utv_sampling.py`** + **`utv_sampling_derivatives.py`** | tangent sampling is the `tv` family; `ut3tangent_` prefix is off-grammar → `utv_`. |
| `ufv_tangent_operations.py` | **`utv_operations.py`** | mirror of `tv_operations.py`. |
| `ut3_operations.py`, `ut3_linalg.py`, `ut3_masking.py`, `ut3_orthogonalization.py`, `ut3_svd.py`, `ufv_operations.py`, `ufv_masking.py` | keep names | function renames inside. |

### Backend geometry name collision (§4)

`backend/optimizers.py`: `MANIFOLD`/`COREWISE` (`GeometryOps` singletons) → **`MANIFOLD_OPS`** /
**`COREWISE_OPS`**, in place (no new module — ⚑F10). Frontend `MANIFOLD`/`COREWISE` singletons keep
their names (no-prefix rule is frontend-only).

### Frontend

**No frontend file renames** (standing rule: no `manifold.py` rename). `corewise.py`, `safety.py`
untouched. Frontend method names unchanged except A1.

---

## C. Token inventory (old → new), by target module

“✓” = already conforming, unchanged. Names not listed anywhere = unchanged. Internal helpers not in
`__all__` keep their names unless listed. Every rename is a whole-word scripted substitution over
`t3toolbox/ tests/ examples/ docs/ dev/*.py CLAUDE.md` (excluding `OLD_*`, `dev/archive/`).

> **Note on c1–c4:** the *file* groupings shown are the on-hold by-family alternative (§B banner);
> the **function renames themselves stand under either grouping** — review them as rename tables,
> not as file assignments.

### c1. `t3_sampling.py` (← `apply.py` + `entries.py` + `probing.py` t3-part)

| Old | New |
|---|---|
| `tucker_tensor_train_apply` | `t3_apply` |
| `tucker_tensor_train_apply_ambient_transpose` | `t3_apply_ambient_transpose` |
| `tucker_tensor_train_entries` | `t3_entries` |
| `tucker_tensor_train_entries_ambient_transpose` | `t3_entries_ambient_transpose` |
| `probe_t3` | `t3_probe` |
| `probe_ambient_transpose` | `t3_probe_ambient_transpose` |
| `apply_corewise_transpose` | `t3_apply_corewise_transpose` (exact `ut3_` mirror) |
| `entries_corewise_transpose` | `t3_entries_corewise_transpose` |
| `probe_corewise_transpose` | `t3_probe_corewise_transpose` |
| `probe_dense` | `dense_probe` (⚑F4: `dense` = operand-representation prefix) |
| `compute_xis` / `compute_mus` / `compute_nus` / `compute_etas` | `compute_xi` / `compute_mu` / `compute_nu` / `compute_eta` (singular alignment; unprefixed chain helpers) |
| `assemble_zs` | `assemble_z` |

### c2. `tv_sampling.py` (← `probing.py` tangent-part)

| Old | New |
|---|---|
| `probe_tangent` / `apply_tangent` / `entries_tangent` | `tv_probe` / `tv_apply` / `tv_entries` |
| `probe_tangent_transpose` / `apply_tangent_transpose` / `entries_tangent_transpose` | `tv_probe_transpose` / `tv_apply_transpose` / `tv_entries_transpose` |
| `precompute_{apply,entries,probe}_frame_sweep` | `tv_precompute_{...}_frame_sweep` (⚑F3) |
| `{apply,entries,probe}_jacobian_from_sweep` | `tv_{...}_jacobian_from_sweep` |
| `{apply,entries,probe}_transpose_from_sweep` | `tv_{...}_transpose_from_sweep` |
| `compute_dxis` / `compute_sigmas` / `compute_taus` / `compute_detas` / `compute_sigma_hats` | singular: `compute_dxi` / `compute_sigma` / `compute_tau` / `compute_deta` / `compute_sigma_hat` |
| `compute_{deta,tau,sigma,dxi}_tildes` | `compute_{deta,tau,sigma,dxi}_tilde` |
| `assemble_tangent_zs` | `assemble_tangent_z` |
| `assemble_tucker_variations` / `assemble_tt_variations` | ✓ (“variations” is the object name, not a plural) |

### c3. `t3_sampling_derivatives.py` (← `probe_derivatives.py` t3-part)

| Old | New |
|---|---|
| `probe_derivatives_t3` / `apply_derivatives_t3` / `entries_derivatives_t3` | `t3_probe_derivatives` / `t3_apply_derivatives` / `t3_entries_derivatives` |
| `{probe,apply,entries}_corewise_derivatives_transpose` | `t3_{...}_corewise_derivatives_transpose` (exact `ut3_` mirror) |
| `{probe,apply,entries}_derivatives_dense` | `dense_{probe,apply,entries}_derivatives` |
| helpers `check_perturbation_vectors`, `check_perturbation_index`, `build_input_jets`, `binomial_combine_tensor`, `compute_{mu,nu,eta}_jets`, `assemble_z_jets` | ✓ (already singular-jets style) |

### c4. `tv_sampling_derivatives.py` (← `probe_derivatives.py` tangent-part)

| Old | New |
|---|---|
| `{probe,apply,entries}_tangent_derivatives` | `tv_{...}_derivatives` |
| `{probe,apply,entries}_tangent_derivatives_transpose` | `tv_{...}_derivatives_transpose` |
| `precompute_{...}_frame_sweep_jets` | `tv_precompute_{...}_frame_sweep_jets` |
| `{...}_jacobian_derivatives_from_sweep` | `tv_{...}_jacobian_derivatives_from_sweep` |
| `{...}_transpose_derivatives_from_sweep` | `tv_{...}_transpose_derivatives_from_sweep` |
| helpers `compute_{sigma,tau,deta}_jets`, `assemble_tangent_z_jets`, `compute_{deta,tau,sigma,dxi}_tilde_jets`, `assemble_{tucker,tt}_variation_jets`, `compute_sigma_hat_jets` | ✓ |

### c5. `tv_operations.py` (← `tangent_operations.py`)

| Old | New |
|---|---|
| `tangent_to_dense` / `tangent_to_t3` | `tv_to_dense` / `tv_to_t3` (locked §1) |
| `orthogonal_gauge_projection` / `oblique_gauge_projection` | `tv_orthogonal_gauge_projection` / `tv_oblique_gauge_projection` |
| `gauge_residual` / `retract` | `tv_gauge_residual` / `tv_retract` |
| `project_t3_onto_tangent_space` / `project_dense_onto_tangent_space` | `tv_project_t3_onto_tangent_space` / `tv_project_dense_onto_tangent_space` (⚑F2: long but regular) |
| `{stack,unstack}_tangent_stack`, `{stack,unstack}_frame_stack` | `tv_{stack,unstack}_tangent_stack`, `tv_{stack,unstack}_frame_stack` |
| `tt_zipper_left_to_right` / `tt_zipper_right_to_left` | ✓ (chain-level helpers — ⚑F1 convention) |

### c6. `fv_conversions.py` + `fv_operations.py`

| Old | New |
|---|---|
| `orthogonal_representations` (from dissolved module) | `t3_orthogonal_representations` (operand prefix; matches the frontend function and the uniform `ut3_orthogonal_representations`) → lands in `fv_conversions.py` |
| `frame_orthogonality_residual` / `frame_consistency_residual` | `fv_frame_orthogonality_residual` / `fv_frame_consistency_residual` → land in `fv_operations.py` |
| `fv_to_t3` | ✓ |
| `zeros_variations` / `randn_variations` / `unit_variations` / `variations_from_vector` | `fv_variations_zeros` / `fv_variations_randn` / `fv_variations_unit` / `fv_variations_from_vector` (⚑F6: `fv_<object>_<op>` micro-grammar) |
| `reverse_frame` | `fv_frame_reverse` (⚑F6) |
| `absorb_weights_into_tangent_cores` | **untouched** (parked, weighted) |

### c7. `t3_constructors.py` / `t3_conversions.py` / `t3_operations.py` (3-way split, ⚑F8)

| Old (in `t3_operations.py`) | New | Target module |
|---|---|---|
| `t3_zeros`, `t3_ones`, `t3_corewise_randn` | ✓ | `t3_constructors.py` |
| `to_dense` | `t3_to_dense` (largest blast radius of the pass) | `t3_conversions.py` |
| `t3_to_dense_chain`, `t3_to_vector`, `t3_from_vector`, `t3_to_tensor_train` (add to `__all__`), `t3_from_tensor_train`, `t3_from_canonical` | ✓ | `t3_conversions.py` |
| `absorb_tucker_into_tt` | `t3_absorb_tucker_into_tt` | `t3_operations.py` |
| `broadcast_t3_to_common_stack` | `t3_broadcast_to_common_stack` | `t3_operations.py` |
| `squash_tt_tails` (+ uniform twin) | **`tt_squash_tails`** (merged polymorphic — F1) | `t3_operations.py` (or the chain module — settle at execution) |
| `reverse_tt` (+ `reverse_utt`) | **`tt_reverse`** (merged polymorphic — F1) | ditto |
| `change_tucker_core_shapes` / `change_tt_core_shapes` | `tucker_change_core_shapes` / `tt_change_core_shapes` (F1) | `t3_operations.py` |
| — | **add `t3_squash_tails(data)`** (t3-level wrapper mirroring `ut3_squash_tails`; the razor — verify at execution what the frontend `.squash` body needs) | `t3_operations.py` |
| `t3_segment`, `t3_concatenate`, `t3_unstack`, `t3_stack` (add to `__all__`), `t3_core_shapes`, `t3_sum` | ✓ | `t3_operations.py` |
| `wt3_squash_tails` (unexported copy) | **untouched** (weighted = fully out of scope, §0.6) | stays in `t3_operations.py` |

### c8. `t3_linalg.py`, `t3_orthogonalization.py`, `t3_svd.py`, `orthogonalization.py`

| Old | New |
|---|---|
| `t3_inner_product_t3` | `t3_inner_product` (kills the suffix; exact `ut3_inner_product` mirror) |
| `t3_add`, `t3_sum_stack`, `t3_scale`, `t3_norm`, `t3_mult`, `t3m_*` (brand), `t3_plus_scalar` | ✓ |
| `left_orthogonalize_t3` / `right_orthogonalize_t3` | `t3_left_orthogonalize` / `t3_right_orthogonalize` (prefix, never suffix) |
| `up_orthogonalize_tt_cores` / `down_orthogonalize_tucker_cores` (t3-level) | `t3_up_orthogonalize_tt_cores` / `t3_down_orthogonalize_tucker_cores` |
| `{down,left,right}_svd_{tucker,tt}_core`, `up_svd_tt_core`, `down_svd_tt_core`, `orthogonalize_relative_to_{tucker,tt}_core` | `t3_`-prefixed; add the missing ones to `__all__` (they’re consumed by the frontend) |
| `t3_orthogonality_residual`, `t3svd` (brand) | ✓ |
| `rank_adjustment_sweep` | `t3_rank_adjustment_sweep` (mirror `ut3_rank_adjustment_sweep`) |
| `tucker_svd_dense` / `ttsvd_dense` / `t3svd_dense` (← `dense_t3svd.py`) | `dense_tucker_svd` / `dense_ttsvd` / `dense_t3svd` (⚑F4/F5) → `t3_svd.py` |
| `orthogonalization.left_orthogonalize_tt_cores` / `right_orthogonalize_tt_cores` (polymorphic chain) | `tt_left_orthogonalize` / `tt_right_orthogonalize` (F1 — kills the 3-way collision with the t3-level and ut3-level same-named ops; placement prefix-vs-suffix pending Nick) |

### c9. Uniform modules

| Old | New |
|---|---|
| `ut3tangent_*` (30 fns, `ufv_sampling.py`) | `utv_*` (same tails) → `utv_sampling.py` / `utv_sampling_derivatives.py` |
| `tangent_to_ut3` | `utv_to_ut3` |
| `retract` / `corewise_retract` / `gauge_residual` (uniform tv-level) | `utv_retract` / `utv_corewise_retract` / `utv_gauge_residual` |
| `orthogonal_gauge_projection` / `oblique_gauge_projection` (uniform) | `utv_orthogonal_gauge_projection` / `utv_oblique_gauge_projection` |
| `project_ut3_onto_tangent_space` | `utv_project_ut3_onto_tangent_space` |
| `{stack,unstack}_{tangent,frame}_stack` (uniform), `sum_tangent_stack` | `utv_`-prefixed |
| `ufv_corewise_inner` | `utv_corewise_inner` (⚑F11 — it’s the tangent coordinate metric; the `GeometryOps.inner` seam consumer updates with it) |
| `reverse_utt` | **merged into polymorphic `tt_reverse`** (F1 — one-line same-math twin) |
| `uniform_squash_tt_tails` | **merged into polymorphic `tt_squash_tails`** (F1) |
| `ut3_squash_tails`, `ut3_reverse` | ✓ (data-level, stay — they wrap the chain ops with mask/shape handling; NOT duplicates of them) |
| `pack_vectors`, `unpack_vectors`, `is_packed`, `pack_if_ragged` | ✓ (packedness infra) |
| `make_uniform_masks` / `apply_masks_to_cores` | `ut3_make_masks` / `ut3_apply_masks` (⚑F12); `require_concrete_masks` ✓ (guard) |
| `make_frame_masks` / `apply_frame_masks` / `apply_variations_masks` | `ufv_make_frame_masks` / `ufv_apply_frame_masks` / `ufv_apply_variations_masks` (⚑F12) |
| `{up,down,left,right}_orthogonalize_*_cores` (ut3-level) | `ut3_`-prefixed (mirror c8); `*_supercores` chain-level ✓ |
| `uniform_t3_svd` | `ut3svd_supercores` (chain-level worker under `ut3svd` — ⚑F13) |
| `ut3_randn` | `ut3_corewise_randn` (mirror `t3_corewise_randn`; the name is honest — iid core entries, not a manifold Gaussian) |
| `ufv_reverse_frame` / `ufv_reverse_variations` | `ufv_frame_reverse` / `ufv_variations_reverse` (only if ⚑F6 adopted; else ✓) |
| everything else in `ut3_*`/`ufv_*` | ✓ |

### c10. `optimizers.py` (backend) + `common.py`

| Old | New |
|---|---|
| `MANIFOLD` / `COREWISE` (backend `GeometryOps` singletons) | `MANIFOLD_OPS` / `COREWISE_OPS` (⚑F10) |
| `has_jax` | `jax_available` (31 refs incl. frontend + safety.py) |
| everything else | ✓ |

---

## D. Execution slices

Every sub-slice: scripted whole-word renames → full suite (`593 tests / 40,215 subtests`, ~6 min) →
grep audit (zero stragglers of every old token) → one commit. File moves via `git mv` so history
follows. Doctest outputs re-run, never hand-edited. Scope of rewrites: `t3toolbox/`, `tests/`,
`examples/`, `docs/*.md`, `dev/*.py`, `dev/*.md` (current only), `CLAUDE.md` — **never** `OLD_*`,
`dev/archive/`, `t4s.pdf`-adjacent TeX.

- **Slice 1 — frontend + flag** (small): A1 renames (`squash`→`squash_tails` incl. doctests;
  `uxx`→`xx`; `has_jax`→`jax_available`). 1–2 commits.
- **Slice 2a — the sampling cut** (biggest): **ON HOLD pending the grouping decision (§B banner)**.
  Function renames per c1–c4 + whichever file grouping Nick picks; consumers:
  `tucker_tensor_train.py`, `manifold.py`, `frame_variations_format.py`, `backend/fitting.py`,
  tests, examples.
- **Slice 2b — t3 family** (c7–c8): the 3-way split, dense merge, orthogonalization renames, chain
  collision fix, dead-duplicate deletion.
- **Slice 2c — ragged tv/fv** (c5–c6): `tv_operations.py`, dissolve `orthogonal_representations.py`.
- **Slice 2d — uniform mirrors** (c9): `utv_*` renames + the two sampling splits.
- **Slice 2e — geometry singletons + `__all__` hygiene** (c10 + stragglers).
- **Slice 3 — public API curation.** Populate `t3toolbox/__init__.py`; proposed export list:
  `TuckerTensorTrain`, `UniformTuckerTensorTrain`, `T3Frame`, `T3Variations`, `T3Tangent`,
  `UT3Frame`, `UT3Variations`, `UT3Tangent`, `MANIFOLD`, `COREWISE`, `UNIFORM_MANIFOLD`,
  `UNIFORM_COREWISE`, `GaussNewtonModel`, `UniformGaussNewtonModel`, the six `fitting.*_model`
  factories, the four optimizers, `safety` (module) + `safe`/`unsafe`, `__version__`.
  `backend/__init__.py` stays **empty** (backend users import submodules explicitly —
  `from t3toolbox.backend import t3_sampling`); ⚑F14.
- **Slice 4 — docs/notes reconciliation.** CLAUDE.md (module names, the `probing.py` references,
  `fv_operations` parked note), `docs/*.md` cross-refs, `docs/signature_style.md` reference-module
  pointer (`backend/probing.py` → `backend/tv_sampling.py`), HANDOFF + naming_review closure, and
  the **`base-inner`→`frame-inner`** prose rename (placeholder-protected, whitespace-normalized
  audit, word-level review — the hyphen-boundary trap from the frame rename applies directly).

Methodology (inherited from the frame rename, scaled to token renames): exhaustive per-token
inventory first (done — this doc); longest-token-first substitution order so substring collisions
can’t fire (e.g. `probe_t3` before any bare-`probe` handling — note `probe_t3` is a substring-safe
whole word, but `retract`/`gauge_residual` renames MUST be word-bounded and module-scoped since the
uniform twins share the bare names until 2d lands); import-cycle check after each file move; the
word-level semantic review reserved for Slice 4’s prose renames.

---

## E. Flagged decisions for Nick (⚑) — each with a recommendation

- **F1 — REVISED (Nick's tt/utt proposal, 2026-07-11): chains get a proper family, `tt`** (bare
  tensor-train chain), analogous to `t3`/`ut3`. Ragged-and-polymorphic share the `tt` name (the
  general plan: with polymorphism you don't need both); `utt` is reserved for uniform-*only* chain
  ops. Investigation result: the chain sweeps are already merged — ONE polymorphic SVD-based
  implementation in `orthogonalization.py`; the t3/ut3-level same-named functions are thin wrappers
  adding Tucker handling resp. mask bookkeeping (no duplicate math anywhere). Concretely:
  `left/right_orthogonalize_tt_cores` (polymorphic chain) → **`tt_left_orthogonalize` /
  `tt_right_orthogonalize`**; **merge the two remaining one-line same-math chain twins**
  polymorphically: `reverse_tt`+`reverse_utt` → **`tt_reverse`**, `squash_tt_tails`+
  `uniform_squash_tt_tails` → **`tt_squash_tails`** (the sweep already dispatches reverse
  internally — this simplifies it; the ONE sanctioned twin-merge, chain-level and trivial). `utt_`
  then has no members (kept in the grammar, unused). `tt_zipper_*` ✓ already conforms;
  `change_{tucker,tt}_core_shapes` → `tucker_change_core_shapes`/`tt_change_core_shapes`; the
  uniform supercore-PAIR workers (`up_orthogonalize_tt_supercores`,
  `down_orthogonalize_tucker_supercores` — they take BOTH supercores, so not `tt` ops) keep their
  descriptive `*_supercores` names inside `ut3_orthogonalization`. **Open sub-question — placement:**
  prefix (`tt_left_orthogonalize`) vs Nick's written form (`left_orthogonalize_tt`). *Recommend
  prefix*: one rule everywhere ("leading token = operand type") and `tt_`-autocomplete surfaces all
  chain ops together — but this is a user-help judgment, Nick decides.
- **F2 — projection names.** `tv_project_t3_onto_tangent_space` / `tv_project_dense_onto_tangent_space`
  / `utv_project_ut3_onto_tangent_space` — long but regular (locked: accept long morphemes).
  *Recommend yes.*
- **F3 — sweep machinery gets `tv_`/`utv_`** even though the precompute’s operand is a frame — the
  sweep exists only as the tangent-jacobian factor. *Recommend yes.*
- **F4 — `dense_` operand prefix** for dense-operand reference ops (`dense_probe`,
  `dense_t3svd`, …). *Recommend yes* — it IS the primary operand representation.
- **F5 — merge `dense_t3svd.py` into `t3_svd.py`.** From-dense construction is what a t3-family user
  looks for under “svd”. *Recommend yes.*
- **F6 — `fv_<object>_<op>` micro-grammar** (`fv_variations_zeros`, `fv_frame_reverse`,
  `ufv_variations_reverse`). Current order is mixed even within uniform. *Recommend yes*; the
  cheaper alternative is prefix-add-only with mixed order.
- **F7 — split sampling-derivatives modules on the uniform side too** (`ut3_sampling_derivatives.py`,
  `utv_sampling_derivatives.py`) for column symmetry, despite small files. *Recommend yes.*
- **F8 — 3-way split of `t3_operations.py`** into constructors/conversions/operations, conforming to
  the `ut3_` template. *Recommend yes.*
- **F9 — keep `common.py` as the single infra catch-all** (no split). *Recommend yes* — splitting is
  churn without user benefit.
- **F10 — backend geometry singletons renamed in place** (`MANIFOLD_OPS`/`COREWISE_OPS` in
  `backend/optimizers.py`), no new `geometry.py` module. *Recommend yes* — `GeometryOps` itself is
  already unambiguous.
- **F11 — `ufv_corewise_inner` → `utv_corewise_inner`** (tangent coordinate metric, not a bare
  variations op). *Recommend yes.*
- **F12 — masking function prefixes** (`ut3_make_masks`, `ufv_make_frame_masks`, …). *Recommend yes*,
  low confidence on exact tails — happy to keep current tails with prefixes only.
- **F13 — `uniform_t3_svd` → `ut3svd_supercores`.** *Mild recommend*; alternative
  `uniform_t3svd_chain`.
- **F14 — `backend/__init__.py` stays empty** (explicit submodule imports are the blessed backend
  style; re-exporting ~40 modules’ names flat would defeat the module namespacing). *Recommend yes.*
- **F15 — `ut3_randn` → `ut3_corewise_randn`** (mirror `t3_corewise_randn`). *Recommend yes*;
  alternative is renaming ragged to `t3_randn`, but “corewise” honestly says it’s not a manifold
  Gaussian.

Anything not flagged above I treat as settled by the locked grammar + your five decisions. If any ⚑
lands differently, only that table row changes — the slices don’t.
