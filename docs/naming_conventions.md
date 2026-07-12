# Naming conventions

How things are named in T3Toolbox, so you can predict a name you have never seen — and find the
function you want from the name alone. The conventions exist to help the reader; where a
convention would have hurt more than helped, we broke it deliberately, and the exceptions are
cataloged at the end. (Prefix grammar and module map settled in the 2026-07 naming pass;
decisions log: `dev/archive/naming_review.md`.)

## The two layers

- **Frontend** (`t3toolbox/*.py`): thin object-oriented classes over frozen dataclasses.
  **Methods carry no family prefix — the class is the namespace** (`x.probe(...)`,
  `v.apply_derivatives(...)`).
- **Backend** (`t3toolbox/backend/*.py`): stateless functions on raw `.data` tuples /
  supercores. **Public backend functions carry a family prefix** naming their primary operand's
  representation. Backend users import submodules explicitly
  (`from t3toolbox.backend import probing`); nothing is re-exported flat.

## The family prefix grammar

`[w][u]<family>_` — modifiers stack in order **weighted → uniform → family**.

| Prefix | Operand | Example |
|---|---|---|
| `t3_` | ragged Tucker tensor train data `(tucker_cores, tt_cores)` | `t3_probe`, `t3_to_dense`, `t3_inner_product` |
| `ut3_` | uniform T3 data `(tucker_supercore, tt_supercore, shape, masks)` | `ut3_probe`, `ut3_squash_tails` |
| `tv_` | tangent vector: a `(frame, variations)` pair *treated as the sum of its terms* | `tv_apply`, `tv_retract`, `tv_to_t3` |
| `utv_` | uniform tangent vector | `utv_apply`, `utv_retract`, `utv_to_ut3` |
| `fv_` | frame / variations data as *indexed pieces* (coarse: frame-only and variations-only ops share it) | `fv_to_t3` (single indexed term), `fv_variations_zeros`, `fv_frame_reverse` |
| `ufv_` | uniform frame / variations | `ufv_apply_frame_masks`, `ufv_variations_reverse` |
| `tt_` | a bare TT core chain (no Tucker, no masks) | `tt_reverse`, `tt_squash_tails`, `tt_left_orthogonalize` |
| `dense_` | a dense (full) tensor operand | `dense_probe`, `dense_t3svd` |
| *(none)* | representation-agnostic infrastructure | `common`, `contractions`, `stacking`, `linalg`, `ranks` |

`w` (weighted) exists only on `t3`/`tv`; the weighted layer is parked pending a redesign.
The founding disambiguation example: `fv_to_t3(index, frame, variations)` materializes ONE
indexed term of the pair, while `tv_to_t3(frame, variations)` is the efficient SUM of all
terms — distinct math on the same data, told apart by the prefix alone.

## Polymorphism: the ragged name is the polymorphic name

Many backend ops accept ragged **or** uniform operands and dispatch internally
(`is_ndarray` on the input — a tuple of cores is ragged, one stacked supercore is uniform).
Such a function lives **unadorned in the base family**; there is no separate `u`-twin. The
`tt_` chain family is fully polymorphic (`tt_reverse`, `tt_squash_tails`,
`tt_left_orthogonalize` serve both layers — `utt_` is reserved but has no members); `t3_apply`
and `t3_entries` likewise take either representation. A `u`-prefix therefore signals
**uniform-specific** logic (masks, supercore packing), not merely "works on uniform".

## The module map (family × operation kind)

Backend modules tile a matrix of (family) × (operation kind: constructors · conversions ·
operations · linalg · orthogonalization · svd · sampling · masking):

| | constructors | conversions | operations | linalg | orthogonalization | svd | masking |
|---|---|---|---|---|---|---|---|
| **t3** | `t3_constructors` | `t3_conversions` | `t3_operations` | `t3_linalg` | `t3_orthogonalization` | `t3_svd` | — |
| **ut3** | `ut3_constructors` | `ut3_conversions` | `ut3_operations` | `ut3_linalg` | `ut3_orthogonalization` | `ut3_svd` | `ut3_masking` |
| **tv / utv** | — | — | `tv_operations` / `utv_operations` | — | — | — | — |
| **fv / ufv** | — | `fv_conversions` / `ufv_conversions` | `fv_operations` / `ufv_operations` | — | — | — | `ufv_masking` |
| **tt** | — | — | `tt_operations` | — | `tt_orthogonalization` | — | — |

Outside the matrix: the unprefixed infrastructure (`common`, `contractions`, `stacking`,
`linalg`, `ranks`) and the fitting/optimization stack (`fitting`, `optimizers`,
`uniform_fitting`), which is organized by role, not by operand family.

**Sampling is the deliberate exception** — the *ragged* sampling modules are grouped by
**sampling type**, not by family: `apply.py`, `entries.py`, `probing.py` each hold their type's
t3 + tv + dense ops, transposes, and frame sweeps, plus one `sampling_derivatives.py` for all
the jet (derivative) machinery. Why: for one sampling type the t3 evaluation, the tangent
𝒥/𝒥ᵀ, and the sweeps form a single connected algorithm (presented together in the paper), and
the containment **probe ⊃ apply ⊃ entries** gives the three files parallel structure —
`apply.py` and `entries.py` import the general machinery from `probing.py`, never the reverse.
The family axis survives inside each file through the function prefixes (`t3_apply`,
`tv_apply`, `dense_apply_derivatives`, ...).

The *uniform* sampling modules (`ut3_sampling.py`, `utv_sampling.py`) are instead grouped by
**object type** — deliberately asymmetric. The uniform sampling functions are not independent
mathematical algorithms: each is a thin wrapper (mask once, pack, delegate to the shared
polymorphic ragged machinery, re-mask) whose few real operations are tied to its *object's*
structure, not to the sampling math. Grouping wrappers by the object they wrap is what helps
the reader; the algorithm story lives in the ragged type files they delegate to.

## Converters encode the operand type

Within the **uniform** classes, `t3`-flavored names mean "the ragged layer":
`UniformTuckerTensorTrain.from_t3(x, ...)` converts a *ragged* T3 to uniform, and
`UT3Frame.from_t3frame` / `UT3Tangent.to_t3tangent` are the cross-layer converters, while
`from_ut3`/`to_ut3` are same-layer. On the ragged classes there is no such ambiguity, so
`T3Frame.from_t3`/`to_t3` are same-layer. Do not "unify" these — the names carry the types.

Some frontend modules also export a **same-named function pair** with the backend: e.g.
`frame_variations_format.t3_orthogonal_representations` (returns `T3Frame`/`T3Variations`
objects) wraps `backend.fv_conversions.t3_orthogonal_representations` (returns raw data), and
likewise `ut3_orthogonal_representations`. Same name = same operation; the module tells you
which layer you are holding.

## Parameter names carry representation too

- `x` — ragged T3 data; `data` — uniform UT3 data (supercores + shape + masks).
- `xx` — a collection of ragged objects; `uxx` — a collection of uniform objects.
- `frame`/`variations` (frontend objects) vs `frame_data`/`variations_data` (raw tuples).
- `ww` — probe/apply vectors; `pp` — perturbation vectors; `index` — entry indices.
- Batch blocks are single capital letters — `C` (frame stack), `W` (probe stack), `K` (tangent
  stack), frame-inner order `W+K+C` — and body locals encode their axis layout as a name suffix
  (`mu_WCa`, `C_aib`); see `docs/batching_and_stacking.md` and `docs/contributor/signature_style.md`.

## Semantic markers (never added or dropped by renames)

- **`corewise`** — coordinate-space math on the core arrays, never Hilbert–Schmidt / tensor
  space. `sum_stack` (true tensor sum; rank-growing) vs `sum_stack_corewise`
  (elementwise core sum; rank-preserving, NOT the tensor sum); `corewise_dot` /
  `utv_corewise_inner` (coordinate metric) vs `t3_inner_product` / `MANIFOLD.inner` (HS);
  `t3_corewise_randn` (iid core entries) vs `MANIFOLD.randn` (tangent-space Gaussian).
- **`numerically_`** — an SVD-grade numerical check, vs bare structural integer arithmetic
  (`has_minimal_ranks` vs `has_numerically_minimal_ranks`).
- **Capability-honest tails** — `ut3_full_sum` (all modes only, vs ragged `t3_sum(axis=...)`),
  `ut3_norm_orthogonalized` (always orthogonalizes). The uniform twin is not renamed to match
  the ragged name when it genuinely supports less.

## Cataloged exceptions (deliberate)

- **Sampling modules are grouped by type**, not family (above).
- **Brand names**: `t3svd`, `t3m` (and compounds like `t3m_swap`) are algorithm names, exempt
  from prefix-position rules.
- **Chain helpers keep descriptive names**: `tt_zipper_left_to_right`,
  `tucker_change_core_shapes`, and the uniform supercore-pair workers
  (`up_orthogonalize_tt_supercores`, `ut3svd_supercores`) — the name states the operand
  better than a bare family prefix would.
- **Fitting/optimization is role-named**: `uniform_minimal`, `uniform_least_squares_problem`,
  the `uniform_*_kind` seam builders, `pack_sample`/`pack_data`, and the packedness utilities
  (`is_packed`, `pack_if_ragged`) are fitting infrastructure, not family data ops.
- **`MANIFOLD` / `COREWISE`** are the frontend geometry singletons; the backend `GeometryOps`
  singletons are `MANIFOLD_OPS` / `COREWISE_OPS`.
- **Frontend methods stay unprefixed** even when a same-named backend function exists at a
  different level (e.g. the method `x.rank_adjustment_sweep()` vs backend
  `t3_rank_adjustment_sweep`); the class namespace disambiguates.
- **`require_concrete_masks`** is a jit guard (infrastructure), unprefixed by design.
