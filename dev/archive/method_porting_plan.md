# TuckerTensorTrain → T3Basis / T3Variations / T3Tangent — porting plan (working doc)

Working notes for the audit (task #16): which `TuckerTensorTrain` methods should the validated classes
get? **Process:** walk `TuckerTensorTrain`'s methods in small groups, discuss & decide per group,
**implement nothing** until the whole plan is locked. Uncommitted working doc — delete once implemented.

## Governing principles
- **Module nesting `T3 → BV → Manifold`** — higher layers depend on lower; never push a manifold concept
  down (e.g. no `manifold_dimension` on `TuckerTensorTrain`).
- **Structural, not numerical** ranks/checks (no `rtol`/`atol`, no `t3svd` inside a "fast check").
- **Match ops to the type's algebra** — `T3Basis` = manifold point (frame + base point); `T3Variations`
  = thin split-agnostic **corewise** container (its ops are corewise, faithful to tangent ops only under
  orthogonal+gauge); `T3Tangent` = full tangent vector.
- **T3-only**; HT (Hierarchical Tucker) is a *documentation lens*, not a generalization target.

## Locked decisions
### Group 1 — metadata
- `size` (= `np.prod(shape)`, dense element count) and `data_size` (= stored core entries, "size on disk")
  on all three, consistent with `TuckerTensorTrain`; `T3Tangent.data_size` = basis + variations.
- `minimal_ranks` (structural) on `T3Basis` (+ `T3Tangent` delegate). Not `T3Variations`.
- dimension: `T3Tangent.tangent_space_dimension = manifold_dim((shape, up_ranks, left_ranks))` (Manifold
  layer — no circular dep). **NOT** on `T3Basis` (BV layer; would need an upward `bvf`→`manifold` import,
  violating the nesting) nor `TuckerTensorTrain`. (`manifold_dim` already reduces to minimal ranks + is
  gauge-quotiented.)
- Skip: `core_shapes` parity, unified `ranks`, `structure`-on-Variations.

### Group 2 — validate
- Add `T3Tangent.validate()` = `basis.validate()` + `variations.validate()` + `check_bv_pair`;
  `__post_init__` runs the full `validate()` (cheap, structural-only → jit/pytree-safe).

### Group 3 — to_dense / to_t3 (base-point reconstruction)
- `T3Basis.to_t3()` = base point as a `TuckerTensorTrain` at natural ranks (canonical-center
  reconstruction); `T3Basis.to_dense()` = `to_t3().to_dense()`. Documented: equals the base point for a
  consistent basis (e.g. from `orthogonal_representations`); a canonical-center form otherwise. NO
  consistency check.
- `T3Variations`: no `to_dense`/`to_t3` (only meaningful with a basis).

### Group 4 — structural surgery
- Port **`reverse()`** to all three. Use case: reverse a T3, then reverse its derived objects without
  recomputing. `T3Basis.reverse` must **swap L↔R** (new left = reverse+bond-transpose of old right; new
  right from old left; reverse+transpose `D`; reorder `U`) — the redundant L/R store makes it cheap;
  reuse the `reverse_tt` bond-transpose. Verify gauge/orthogonality preserved; test: `reverse` commutes
  with `to_dense` (mode axes flipped).
- Do NOT port `segment` / `concatenate` / `squash` / `resize` (plain-T3 representation surgery; breaks
  the frame/base-point/tangent structure — manifold uses `retract`/gauge instead).

### Group 5 — backend convert
- Add `to_jax`, `to_numpy`, `copy` (methods) + `contains_jax` (cached_property) to all three, mirroring
  `TuckerTensorTrain`. Trivial: map over cores; `T3Tangent` delegates to basis + variations.

### Group 6 — unstack / stack
- Nothing to port. All classes already have stacking appropriate to their structure (`T3Tangent`'s
  two-axis `*_tangents`/`*_basis` is the correct richer version; a bare `unstack()` there is ambiguous).
  Doc note only (see #4).

### Group 7 — constructors
- `T3Basis.random_orthogonal(structure, stack_shape=())` — orthogonal rep of a random T3 (genuine random
  base point; NOT iid cores). B-only; no bare `randn` on `T3Basis`.
- `T3Basis.from_t3(x)` — orthogonal representation (basis) of a given T3 (general primitive;
  `random_orthogonal` = `from_t3(TuckerTensorTrain.randn(...))`).
- `T3Variations.zeros(variation_shapes, stack_shape=())`, `randn(...)` — **explicit variation-shape spec**
  (low-level corewise, but what users expect for core shapes). No `ones`; no `zeros` on `T3Basis`.
- `_like` family (derive shapes/stack from an object): `T3Variations.zeros_like`/`randn_like`,
  `T3Tangent.zeros_like`/`randn_like` (same base), `T3Basis.random_orthogonal_like`.
- `T3Tangent.random_orthogonal(structure, apply_gauge_projection=…)` — fully random tangent (random base
  + random direction), optional orthogonal gauge projection. (Name may clash with "gauge-orthogonal" —
  confirm.)
- Skip `from_dense` (drags in `t3svd`).
- `unit(index)` on both `T3Variations` (`unit(variation_shapes, index, stack_shape=())`) and `T3Tangent`
  (`unit(base, index)`), with `index = (use_tt_coordinate, i, within_index)` — **bundled** (matches
  `bv_to_t3`). It's the standard basis of the variation-core space → an overcomplete, non-ambient-
  orthogonal generating set of the tangent space (document loudly). **No gauge option** (a projected unit
  isn't a unit; user gauges it themselves).
- Index bundling: keep all coordinate indices bundled. `bv_to_t3` already is; sweep for any other
  unbundled coordinate indices during implementation and bundle them.

### Group 8 — conversions + serialization
- `to_vector`/`from_vector` on `T3Variations` + `T3Tangent` only (optimization DOF; `T3Tangent.to_vector`
  = its variations, basis is the fixed point; `from_vector(flat, base)`). NOT `T3Basis` (not a free
  vector space — use `save`/`load`).
- `save`/`load` on all three.
- Skip `from_canonical`/`from_tensor_train`/`to_tensor_train` (compose via `from_t3`/`to_t3`).

### Group 9 — inner / norm
- Nothing ported. Skip `T3Variations.inner`/`norm` (Nick's call — keep the basis-less container from
  carrying an inner/norm whose meaning is conditional: corewise = HS only under orthogonal+gauge). Skip
  `T3Basis.inner`/`norm` (ambient op on points; compose via `to_t3`). `T3Tangent` already has them
  (corewise + HS-faithfulness warning).

### Group 10 — reductions
- `T3Tangent.sum_tangents(axis=None)` — corewise sum over the tangent stack `K` (batch of tangents → one;
  = tensor sum by linearity, ranks unchanged). `T3Variations.sum_stack(axis=None)` — corewise stack sum
  (named `sum_stack`; it is corewise — `T3Variations` has no tensor sum). Nothing on `T3Basis`.
- Skip `sum` (tensor-mode reduction) and the rank-growing tensor `sum_stack`.

### Group 11 — SVD / orthogonalization family
- Port none. These are the low-level primitives that *build* orthogonal reps (behind `t3svd`,
  `orthogonal_representations`, `retract`). `T3Basis` is already their orthogonal output; `T3Tangent` has
  `retract` + gauge projections as the structure-preserving analogues. Center-sliding (DMRG) is moot
  (`T3Basis` stores all center forms; T3-only, no sweep machinery).

### Group 12 — evaluation (`entries`/`apply`/`probe` + transposes)
- `T3Tangent` already has the full suite (the earlier slice work). `T3Variations`: skip (not a tensor;
  corewise-only). `T3Basis`: skip `apply`/`entries`/`probe` — compose via `to_t3()` (Group 3). Nothing new.

### Group 13 — t3svd / t3svd_dense
- Port none. Rank-truncation / dense→T3 construction machinery; `retract` is the manifold analogue
  (wraps `t3svd`), `from_dense` already skipped. Compose via `to_t3().t3svd()` for the rare need.

### Group 14 — arithmetic dunders
- `T3Variations`: add `__add__`/`__sub__`/`__mul__`/`__neg__` (corewise vector-space ops; sibling of
  `sum_stack`). `T3Basis`: none (manifold point, not a vector space). `T3Tangent` already has them.

## Doc-pass (#4) items accumulated
- `T3Basis` docs: adopt the HT framing (T3 = linear-tree HT; frame + root/center; tangent = per-node
  gauge variations; left/right = where the center sits).
- Fix stale `check_t3_base` See Also in `T3Basis` (the function does not exist).
- Fix `TuckerTensorTrain.minimal_ranks` docstring (it implies *numerical* minimality "smallest… same
  dense tensor… via T3-SVD"; the implementation is *structural*).
- Make structural-not-numerical explicit wherever ranks are documented.
- Document `T3Tangent.randn(apply_gauge_projection=True)` = isotropic Gaussian on the tangent space (==
  randn dense then orthogonal-project), via the corewise=ambient isometry; holds for orthogonal
  (minimal-rank) bases. (Verified: isometry relerr ~4e-14.)
- Strengthen the `T3Tangent.inner`/`norm` warning: for a non-orthogonal or non-gauged tangent, get the
  exact Hilbert–Schmidt value via `to_t3().norm()` / `to_t3().inner(other.to_t3())` (escape hatch, no new API).
- `T3Variations` docstring: state plainly it is the corewise container (its ops are corewise on the
  variation cores; faithful to tangent-space ops only under orthogonal+gauge).
- Doc note: why `T3Tangent` splits stacking into `unstack_tangents`/`unstack_basis` (+`stack_*`) — two
  stacks (`K`+`C`) — vs plain `unstack`/`stack` on the others.
- Fix `TuckerTensorTrain.resize` docstring (stale copy of `to_jax`'s: "Convert core arrays… to Jax arrays").

## Remaining groups to walk
**None — walk complete (all 14 groups decided).**

## Net additions by class (implementation worklist)

**T3Basis:** `to_t3`, `to_dense` (G3); `size`, `data_size`, `minimal_ranks` (G1);
`reverse` (G4, L↔R swap); `to_jax`/`to_numpy`/`copy`/`contains_jax` (G5); `random_orthogonal`, `from_t3`,
`random_orthogonal_like` (G7); `save`/`load` (G8).

**T3Variations:** `size`, `data_size` (G1); `reverse` (G4); `to_jax`/`to_numpy`/`copy`/`contains_jax`
(G5); `zeros`, `randn`, `unit`, `zeros_like`, `randn_like` (G7); `to_vector`/`from_vector`, `save`/`load`
(G8); `sum_stack` (corewise, G10); `__add__`/`__sub__`/`__mul__`/`__neg__` (G14). [all corewise]

**T3Tangent:** `size`, `data_size`, `minimal_ranks` (delegate), `tangent_space_dimension` (direct) (G1);
`validate` + `__post_init__` runs it (G2); `reverse` (G4); `to_jax`/`to_numpy`/`copy`/`contains_jax` (G5);
`random_orthogonal` (+gauge), `unit`, `zeros_like`, `randn_like` (G7); `to_vector`/`from_vector`,
`save`/`load` (G8); `sum_tangents` (G10).

**Cross-cutting:** bundle coordinate indices (sweep); `_like` constructors derive structure from an object.

**Not ported (reasons in the groups):** structural surgery except `reverse`; SVD/orthogonalization family;
`t3svd`/`t3svd_dense`; `sum`/tensor-`sum_stack`; `ones`; `from_canonical`/`from`/`to_tensor_train`;
evaluation, `inner`/`norm`, `to_dense`/`apply` on the wrong types.

## New methods (beyond TuckerTensorTrain ports)

Orthogonal projection onto the tangent space is the **core primitive**; several of these are it applied
to different inputs.
- **F — `manifold.project_dense_onto_tangent(Z_dense, basis)`** — orthogonal projection of a *dense*
  tensor onto the tangent space; module function (outside `T3Tangent`), returns a `T3Tangent`. [LOCKED.
  **Direct dense-contraction algorithm (verified correct):** for each variation slot, contract `Z` with
  the surrounding orthonormal frame (mixed-canonical: `L` before, `R` after, `U`; the `bv_to_t3` adjoint)
  → ungauged variation, then `orthogonal_gauge_projection`. Requires an orthogonal basis. Mirror
  `project`'s environment-contraction conventions; verify vs an independent `T_pM` projector
  (`P=QQᵀ` from gauge-projected `randn` tangents) and the `project(t3svd_dense(Z), basis)` baseline.]
- **`T3Tangent.transport(new_base)`** — projection-based vector transport (`= project(self.to_t3(),
  new_base)`). [LOCKED]
- **Riemannian gradient** (`manifold.riemannian_gradient(euclidean_grad, basis)`) — projection of a
  Euclidean gradient (dense→F, or T3→`project`) onto the tangent space. [LOCKED]
- **`__repr__`** on all four classes (class + shape/ranks/stack; never core dumps — confirmed none exist
  today). [LOCKED]
- **`T3Basis.orthogonalize()`** (`= from_t3(self.to_t3())`, re-orthogonalize a drifted basis) and
  **`T3Basis.is_consistent()`** (opt-in, expensive: do the L/R/center reconstructions agree?). [LOCKED]
- **`T3Tangent.normalized()`** (unit-norm direction). [LOCKED] — stack `__getitem__` **rejected**: `x[i]`
  reads as dense indexing / `.entries()`, so it would confuse users; use `unstack_*` for stack elements.
- **`allclose(other, rtol, atol)`** — semantic **norm-of-difference** `‖self−other‖ ≤ atol+rtol·‖other‖`:
  `T3Tangent` = `(self−other).norm()` (HS); `T3Basis` = `(self.to_t3()−other.to_t3()).norm()` (chordal,
  gauge-invariant); `T3Variations` = corewise diff-norm. Document the semantics. No representation-level
  variant (use `np.allclose` on `.data` for that). [LOCKED]
- **Future (deferred):** exp / log / geodesic distance — no closed form on this manifold; `retract` +
  chordal distance substitute. Revisit if a use case needs true geodesics.
