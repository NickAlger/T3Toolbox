# Naming conventions

How things are named in T3Toolbox, so you can predict a name you have never seen — and find the
function you want from the name alone. The conventions exist to help the reader; where a
convention would have hurt more than helped, we broke it deliberately, and the exceptions are
cataloged at the end. (The rules contributors must follow when *naming new things* — including
which of these conventions are rename-invariant — are in
[`contributor/naming_rules.md`](contributor/naming_rules.md).)

## The two layers

- **Frontend** (`t3toolbox/*.py`): thin object-oriented classes over frozen dataclasses.
  **Methods carry no family prefix — the class is the namespace** (`x.probe(...)`,
  `v.apply_derivatives(...)`).
- **Backend** (`t3toolbox/backend/*.py`): stateless functions on raw `.data` tuples /
  supercores. **Public backend functions carry a family prefix** naming their primary operand's
  representation. Backend users import submodules explicitly
  (`from t3toolbox.backend import probing`); nothing is re-exported flat.

## The family prefix grammar

`[u]<family>_` — the **uniform** modifier precedes the family.

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
| *(none)* | representation-agnostic infrastructure | `common`, `contractions`, `stacking`, `linalg`, `ranks`, `regularization`, `optimizer_display` |

The **weighted** layer has no prefix of its own: it names its operations inside the existing
families (`t3_absorb_weights` / `ut3_absorb_weights` / `fv_absorb_weights` / `ufv_absorb_weights`,
see below); the old `wt3_*` prefix layer was retired.

> **`weight=` on the optimizers is a different thing entirely.** The fitting layer's `weight=`
> ([`fitting_and_optimization.md`](fitting_and_optimization.md) §4.6) is the **residual weight `ω`**
> in the objective `½‖ω⊙r‖²` — a per-`(mode, order)` scaling of the *measurements*. It has nothing to
> do with the weighted layer's edge weights, which are diagonal factors on a tensor's internal edges
> ([`weighting.md`](weighting.md)). The collision is unfortunate but `weight=` shipped in 2026.0.0,
> so it stays; the plural `weights` and the `*Weights` classes always mean edge weights. **Shared Tucker factors** work the same way —
`backend/sharing.py` is one module, but each function inside it carries the family prefix of the data
it operates on (`t3_sharing_residual`, `ut3_tucker_factors_shared`, `fv_shared_frame_data`,
`ufv_share_tucker_variations`). A *feature* is not a family.
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
`T3Frame.from_t3`/`to_t3` are same-layer. The names carry the types.

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

## Semantic markers (load-bearing — you can trust them)

- **`corewise`** — coordinate-space math on the core arrays, never Hilbert–Schmidt / tensor
  space. `sum_stack` (true tensor sum; rank-growing) vs `sum_stack_corewise`
  (elementwise core sum; rank-preserving, NOT the tensor sum); `corewise_dot` /
  `utv_corewise_inner` (coordinate metric) vs `t3_inner_product` / `MANIFOLD.inner` (HS);
  `t3_corewise_randn` (iid core entries) vs `MANIFOLD.randn` (tangent-space Gaussian).
- **`numerically_`** — an SVD-grade numerical check, vs bare structural integer arithmetic
  (`has_minimal_ranks` vs `has_numerically_minimal_ranks`).
- **Checkers** — a non-enforcing predicate that reports a property, returning a verdict per stack
  element. Checkers never raise; safe mode raises at the *call site* that uses them. **The two tiers
  spell it differently, and that is deliberate:**
  - **Frontend: `is_` / `has_`, as a method or property on the object itself.** A checker asks a
    question *about that object*, so it belongs on it — `frame.is_orthogonal()`, `t.is_gauged()`,
    `x.has_shared_tucker_factors(sharing)`, `W.has_shared_tucker_weights(sharing)`. One that takes a
    tolerance or a spec argument is a **method**; a pure derived property (`has_minimal_ranks`) is a
    **property**.
  - **Backend: `<family>_<subject>_<predicate>`, adjective last.** The family prefix must come first
    (that is the whole grammar), so `t3_is_...` would read badly and bury the subject. Hence
    `t3_tucker_factors_shared`, `ut3_tucker_weights_shared`, `fv_weights_consistent`,
    `ranks.frame_has_minimal_ranks`. The paired *residual* — the continuous quantity the boolean
    thresholds — takes the same stem with `_residual` (`t3_sharing_residual`,
    `fv_tied_variations_residual`), and both keep the stack.
- **`_trs` suffix** (sampling-derivative jets) — the *dense binomial-tensor* implementation, kept as a
  reference twin of the standard recurrence/scan form that owns the bare name. `compute_mu_jets` (the
  standard, memory-lean, call-site-wired form) vs `compute_mu_jets_trs` (contracts the `trs` binomial
  tensor as a dense einsum operand: numerically equal to tolerance, higher peak memory, occasionally
  faster in tiny/memory-abundant regimes). The bare name is always the recommended default; `_trs` is
  opt-in. (`trs` = the binomial convolution tensor `trs[t,r,s] = C(t,r)·[r+s=t]`, the module's own term.)
- **Capability-honest tails** — `ut3_full_sum` (all modes only, vs ragged `t3_sum(axis=...)`),
  `ut3_norm_orthogonalized` (always orthogonalizes). The uniform twin is not renamed to match
  the ragged name when it genuinely supports less.

## Why "frame", not "basis"

The orthogonal representation of a tangent space (`T3Frame`, the `fv_`/`ufv_` families) is
deliberately **not** called a basis. "Basis" implies minimality — a linearly independent spanning
set — and the orthogonal representation does **not** require minimal ranks: at a non-minimal-rank
point it is *overcomplete*, and the library supports that on purpose (see the numerical-contracts
material: minimal rank is a precondition for essentially nothing). "Frame" is the idiomatic term in
manifold optimization for exactly this kind of possibly-redundant orthogonal spanning system. (The
T4S paper names the object neither way, so there is no paper conflict to worry about.)

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
- **`MANIFOLD` / `COREWISE`** are the frontend geometry singletons; the backend twins are the
  classes `ManifoldGeometryOps` / `CorewiseGeometryOps` in `backend/geometry.py` (with
  `Uniform`-prefixed versions carrying the fixed rank). The `Ops` suffix distinguishes the
  backend class from the frontend class of the same concept.
- **Frontend methods stay unprefixed** even when a same-named backend function exists at a
  different level (e.g. the method `x.rank_adjustment_sweep()` vs backend
  `t3_rank_adjustment_sweep`); the class namespace disambiguates.
- **Frontend *free* functions DO carry the family prefix** — they have no class namespace, so the
  prefix does that job instead. It is the same rule as the bullet above, not an exception to it:
  whatever disambiguates, disambiguates. Examples: `t3_orthogonal_representations` /
  `ut3_orthogonal_representations`, and the weighted layer's `t3_absorb_weights` /
  `ut3_absorb_weights` / `fv_absorb_weights` / `ufv_absorb_weights` (+ `t3_`/`ut3_weighted_norm`,
  `_weighted_inner`).
  - **Sharing a name with the backend twin is fine, and intended** — `t3_orthogonal_representations`
    exists in *both* `frame_variations_format` (objects) and `backend/fv_conversions` (raw `.data`).
    Same operation, two levels; the module namespace tells them apart, and the backend is never
    re-exported flat, so they never meet.
  - **The prefix is what makes the root export possible.** Ragged/uniform × tensor/tangent gives
    **four** distinct "absorb the weights" operations; unprefixed they would all be `absorb_weights`
    and could not coexist in `t3toolbox/__init__.py` (the second import would silently shadow the
    first). The weighted free functions were briefly unprefixed and hit exactly this; prefixing them
    was the fix (2026-07-15). The *classes* never had the problem — `T3Weights` / `UT3Weights` /
    `T3FrameWeights` / `UT3FrameWeights` are already distinct.
- **`require_concrete_masks`** is a jit guard (infrastructure), unprefixed by design — and lives in
  `backend/common.py` for the same reason, beside the `ValueHashedMasks` mixin and `prefix_mask`: those
  three are the uniform **mask-representation** contract (concrete host arrays · value-based hash/eq ·
  the canonical prefix form), which every uniform object's masks obey — plain, frame, variations, and
  weights. They are not part of any one object's `ut3_*`/`ufv_*` family.

## Intended asymmetries (uniform vs ragged)

Where a uniform class or method offers **less surface** than its ragged twin, the gap is a design
decision, not an unfinished port — do not read these as TODOs:

- **`UT3Tangent` has no `save`/`load`/`to_vector`/`from_vector`** (all four exist on `T3Tangent`).
  A varying-rank uniform stack has no single flat-vector signature to reconstruct from; jax-native
  optimization works on the pytree directly, and flat-vector interop (e.g. scipy) routes through
  the ragged layer.
- **`UniformTuckerTensorTrain.sum_stack()` takes no `axis`** (ragged: `sum_stack(axis=None)`).
- **`UniformTuckerTensorTrain.to_dense()` takes no `squash_tails`** (ragged has it).
- **`UniformTuckerTensorTrain.t3svd(...)` has no `rtol`/`atol`** — uniform truncation is
  structural (max-rank masks), never numerical; per-element numerical truncation could give
  different ranks per stack element, which the uniform shape cannot represent (see
  [`uniform_equivalence_contract.md`](uniform_equivalence_contract.md)).

The capability-honest-tails rule (above) is the same principle at the function level.
