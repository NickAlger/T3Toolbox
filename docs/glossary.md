# Glossary

One entry per concept, under one canonical name in bold, with the other names the docs and the API use for it
("also: ..."). Each definition is short; the link goes to the page that explains it fully (or, for a term the docs
use without defining, to where it matters most). Entries are alphabetical within each section, so related terms stay
together and a known term is still quick to find; to land on an entry from an alias, use the browser's find or the
docs search. Symbols that mean different things on
different pages are collected at the end, in [Symbols](#symbols).

## Tensor formats and layers

- **absorb** (`t3_absorb_weights`, `ut3_absorb_weights`, `fv_absorb_weights`, `ufv_absorb_weights`;
  `T3Tangent.absorb_weights`). Contract diagonal edge weights into the cores (tensor weights) or into the variation
  cores only (a tangent metric), keeping all shapes. [Edge weighting](weighting.md)
- **batch block** (also: block; the `C` / `W` / `K` blocks). One of the three independent batches a computation can
  carry: the frame stack `C`, the probe stack `W`, the tangent stack `K`. Each is zero or more leading axes.
  [The three batch blocks](batching_and_stacking.md#2-the-three-batch-blocks-c-w-k)
- **boundary bonds** (also: tails; `r0`, `rd`). The first and last TT ranks. Usually 1; otherwise the train's ends are
  vectors of ones. `squash_tails` makes both 1 without changing the represented tensor. [User guide](user_guide.rst)
- **central core** (also: central Tucker core; `C` in the `t3m` page). The `n0 × … × n(d-1)` core tensor of the Tucker
  decomposition, which a T3 stores as a tensor train. Never called a "Tucker core" here.
  [What there is to truncate](t3m_methods.md#what-there-is-to-truncate)
- **core** (also: core array). A small tensor contracted with its neighbours to form the represented tensor: Tucker
  cores, TT cores, frame cores, variation cores; on the uniform layer, a slice of a supercore. [User guide](user_guide.rst)
- **CP factors** (also: canonical factors; `from_canonical`). A length-`d` tuple of `(R, Ni)` factor matrices for the
  rank-`R` sum `Σ_k ⊗_i f_i[k]`; the ambient transposes return them. [Transposes](transposes.md)
- **`d`-leading layout** (also: mode axis first). A supercore puts the core index `d` before the stack,
  `(d,) + stack_shape + …`, so a sweep over cores is a `lax.scan` over axis 0. Ragged objects are stack-leading.
  [Why `d` leads](uniform_supercore_layout.md#why-d-leads)
- **determinantal variety** (also: bounded-rank variety, `V_{≤n,≤r}`). The tensors whose ranks are at most given
  bounds. A uniform stack is a batch of points in it; each element's masks name its nominal stratum.
  [A batch in a variety](uniform_ranks_and_varieties.md#the-real-question-a-batch-on-a-manifold-or-a-batch-in-a-variety)
- **edge** (also: internal edge; Tucker edge, TT bond). A contracted index of the T3 network: `d` Tucker edges (Tucker
  core to TT core, size `ni`) and `d+1` TT bonds (between TT cores, size `ri`). Cutting a Tucker edge gives a
  matricization, cutting a TT bond an unfolding. Masks and weights live on edges.
  [`T3Weights`](weighting.md#t3weights--a-weighted-tensor)
- **edge mask** (`tucker_edge_mask`, `tt_edge_mask`; `UT3Masks`; also: rank mask, mask). Per edge and stack element, a
  host boolean vector over the padded width: slot `j` is real iff the mask is `True` there. The rank is `mask.sum(-1)`.
  [What a mask means](uniform_masks_vs_ranks.md#what-a-mask-means)
- **edge weights** (`T3Weights`, `UT3Weights`; also: weighted layer, `weights`). One diagonal per Tucker edge (`d`) and
  per TT bond (`d+1`), the layout `t3svd`'s singular values come in. Unrelated to the residual weight `weight=`.
  [Edge weighting](weighting.md)
- **equivalence contract** (also: uniform equivalence contract). `to_uniform → op → to_ragged` equals the ragged op on
  the real slots; padding is don't-care. Gauge-carrying ops agree only up to gauge. A small number of operations are
  exceptions, most notably the alignment, whose two layers agree only if its fixed-point iteration runs to convergence
  and does not get stuck in a local optimum; the library keeps exceptions to those with a good reason.
  [The contract](uniform_equivalence_contract.md#the-contract)
- **frame stack `C`** (`stack_shape`, `frame_stack_shape`; also: core stack, frame/core stack, base-point stack). A
  batch of whole T3s or base points, on every core of an object. It *is* `stack_shape`.
  [The three batch blocks](batching_and_stacking.md#2-the-three-batch-blocks-c-w-k)
- **frame-inner** (order `W + K + C`). The library-wide order of the batch axes before the core axes: the frame stack
  `C` innermost, `W` outermost, so a frame broadcasts over its tangents and probes for free.
  [The frame-inner convention](batching_and_stacking.md#3-the-frame-inner-convention-and-why)
- **gappy mask** (also: working form; non-prefix mask). A mask whose real slots are not a prefix. On the uniform layer
  `+` concatenates rank masks (and a weights' `kronecker` Kroneckers weight masks), leaving gaps; arithmetic never
  compacts them, while `ut3svd` and frame compaction return prefix masks. Code selects real slots through the mask.
  [Canonical form vs working form](uniform_masks_vs_ranks.md#canonical-form-vs-working-form)
- **group ceiling**. The largest useful rank of a sharing group:
  `n_g ≤ min(N_g, Σ_{i∈g} min(N_g, r_i·r_{i+1}))`, so per-mode Tucker ceilings add across a group.
  [Rank machinery](sharing.md#rank-machinery-minimal-ranks-dimension-continuation)
- **group spectrum** (`s_g`). The singular values of a sharing group's concatenated matricizations
  `[T_(i1) | … | T_(ik)]`; `t3svd(sharing=…)` reports it at every mode of the group.
  [What the group spectrum is](sharing.md#what-the-group-spectrum-is)
- **grouped T3-SVD** (`t3svd(sharing=…)`; also: two-phase truncation). Round the TT bonds first, then truncate each
  sharing group with one SVD of its members' concatenated centers.
  [Grouped truncation](sharing.md#grouped-truncation-t3svdsharing)
- **Hadamard product** (`t3m`, `x * y`; also: elementwise product, `⊙`). The elementwise product of represented
  tensors (ragged layer only); ranks multiply. `x * y` is exact; `t3m` can also truncate, by `'form_then_round'`,
  `'inplace_fused'` (default) or `'swap'`. [`t3m` methods](t3m_methods.md)
- **heterogeneous stack**. One T3 whose cores carry different but broadcastable stacks, e.g. frame cores `C` and
  variation cores `K + C`. Allowed in the backend; frontend classes require one `stack_shape`.
  [Heterogeneous tuples](batching_and_stacking.md#5-heterogeneous-but-broadcastable-tuples-and-the-backendfrontend-split)
- **Hilbert–Schmidt** (also: HS, Frobenius; `inner`, `norm`). The inner product and norm of the represented dense
  tensors, `Σ X[i]·Y[i]`. On tangents it is `MANIFOLD.inner`/`norm`; the corewise metric equals it only on an
  orthonormal frame with gauged variations.
  [The geometries](numerical_contracts.md#the-geometries-the-semantic-innernorm-live-here)
- **left-orthogonal form / right-orthogonal form** (`is_left_orthogonal`, `is_right_orthogonal`; `t3svd(...,
  assume_orthogonal=True)`; also: orthogonality center, center remainder). A T3 with every Tucker core
  down-orthogonal and every TT core but the last left-orthogonal (left form) or but the first right-orthogonal (right
  form); that last or first core is the orthogonality center. `t3svd` returns left form; `assume_orthogonal=True`
  asserts right form.
  [How to minimize](t3svd_minimal_ranks.md#how-to-minimize-rank_adjustment_sweep)
- **mask holder** (`UT3Masks`, `UT3FrameMasks`, `UT3VariationsMasks`). The frozen frontend object that carries a
  uniform object's masks as jax `aux_data`, hashed by mask content.
  [The decision](contributor/uniform_pytree_composition.md#the-decision-a-small-structure-holder-value-hashed-by-mask-content)
- **matricization** (`X_(i)`; also: mode-`i` matricization). The matrix formed by cutting a Tucker–TT edge: mode `i`'s
  index as the rows, all other modes as the columns (Kolda & Bader's usage). For a tensor of shape `N0 × N1 × N2`:
  `N0 × (N1N2)`, `N1 × (N0N2)`, `N2 × (N0N1)`. Its rank is the minimal Tucker rank `ni`. Compare **unfolding**.
  [Notation](contributor/t3svd_verification.md#notation)
- **minimal ranks** (`has_minimal_ranks`, `minimal_ranks`; also: structurally minimal). Ranks with no structural
  redundancy, judged by integer arithmetic: `ni ≤ Ni`, `ni ≤ ri·r(i+1)`, `ri ≤ ni·r(i+1)`, `r(i+1) ≤ ri·ni`. No
  operation needs them for correctness. The fitting layer reduces `x0` itself (on the uniform layer always, since that
  path needs a minimal-rank frame; on the ragged layer on manifold geometries).
  [What minimal ranks means](t3svd_minimal_ranks.md#what-minimal-ranks-means)
- **mode** (also: index, physical index, leg; `d` = number of modes, the tensor's order). One index of the represented
  tensor. "Order" also means derivative order; see **jet**. [Notation](entries_apply_probe.md#notation)
- **mode size** (`Ni`; `shape`). The length of mode `i` of the represented tensor, `shape = (N0, …, N(d-1))`. On the
  uniform layer `N` is the padded width instead. [User guide](user_guide.rst)
- **numerically minimal** (`has_numerically_minimal_ranks`; marker `numerically_`). No stored rank is numerically
  redundant: an SVD-grade check, unlike the structural `has_minimal_ranks`.
  [Minimal ranks](numerical_contracts.md#minimal-ranks-the-settled-verdict)
- **orthogonal core** (left-, right-, down-, up-orthogonal; also: outer-orthogonal). A core with orthonormal rows or
  columns in one reshape (API "orthogonal" means orthonormal). TT core: *left*, its `(ri·ni) × r(i+1)` unfolding has
  orthonormal columns; *right*, its `ri × (ni·r(i+1))` unfolding has orthonormal rows; *up* (also outer), its
  `ni × (ri·r(i+1))` matricization has orthonormal rows (`up_orthogonalize_tt_cores`). Tucker core: *down*,
  `B Bᵀ = I` (`down_orthogonalize_tucker_cores`). The adjective names the side of the cut edge whose indices the
  orthonormality sums over. **Clash:** the frame pages name these by store instead (`O` "down-orthogonal", `U` "up
  orthogonal"); a rename of the stores is being assessed.
  [Four rank stores](frame_variations.md#four-rank-stores--and-why-excess-rank-is-inert)
- **oversample** (`t3m(method='swap', oversample=k)`). Relax every truncation criterion by `k` during the swap merge,
  then truncate once at the targets. Unrelated to the alignment's `oversampling`.
  [Oversampling](t3m_methods.md#oversampling-methodswap-only)
- **packed vectors** (`pack_vectors`, `unpack_vectors`, `pack_if_ragged`; also: packedness mirror). Mode vectors as one
  `(d,) + … + (N,)` zero-padded array instead of a length-`d` list; uniform ops return the form they were given.
  [The vector I/O boundary](uniform_equivalence_contract.md#the-vector-io-boundary--packedness-mirror)
- **pad-safe SVD** (`linalg.pad_safe_svd`). An SVD of a zero-padded matrix whose leading triplets are exactly the
  economy SVD of the real block, bitwise zero on padding, with no rank tolerance. Uniform sweeps use it.
  [The full-rank qualifier](uniform_equivalence_contract.md#the-full-rank-qualifier-measured-2026-08-27)
- **padded widths** (`n`, `r`, `N`; frames `nU`, `nD`, `rL`, `rR`; also: nominal rank, a level's widths, padded `N`).
  The common sizes a uniform object's cores are padded to: Tucker `n`, TT `r`, mode `N ≥ max Ni`.
  [The layout](uniform_supercore_layout.md#the-layout)
- **prefix mask** (`common.prefix_mask`; also: canonical form). A mask whose real slots are `[0, rank)`. `ut3svd` and
  SVD-based orthogonalization produce it; it is an output property, never an input assumption.
  [Canonical form vs working form](uniform_masks_vs_ranks.md#canonical-form-vs-working-form)
- **probe stack `W`** (also: sample stack, vec/index stack). A batch of probe-vector sets or entry indices. It lives on
  `ww` / `index` / `pp` only, never on the cores. [The three batch blocks](batching_and_stacking.md#2-the-three-batch-blocks-c-w-k)
- **ragged layer** (also: ragged representation). The default representation: tuples of per-core arrays, each with its
  own true shape. [Glossary of batching](batching_and_stacking.md#glossary)
- **rank adjustment sweep** (`rank_adjustment_sweep(direction)`). One lossless directional sweep that drops
  structurally redundant ranks; it reaches minimal ranks when the input is orthogonal in the opposite direction.
  [How to minimize](t3svd_minimal_ranks.md#how-to-minimize-rank_adjustment_sweep)
- **ranks** (`ranks`, `structure`, `uniform_structure`). `ranks = (tucker_ranks, tt_ranks)`;
  `structure = (shape, tucker_ranks, tt_ranks, stack_shape)`; uniform `uniform_structure = (d, N, n, r, stack_shape)`.
  [User guide](user_guide.rst)
- **raw sweep ranks** (`ranks.compute_raw_sweep_ranks`). The ranks `t3svd` returns under hard caps: no truncation or a
  tolerance gives minimal ranks, but a cap can orphan the rank before it.
  [Non-minimal ranks](t3svd_minimal_ranks.md#why-the-truncating-sweep-can-leave-non-minimal-ranks)
- **real slot / padding** (also: junk slot, the padded-frame code's word; padded slot, garbage, don't-care). On the
  uniform layer a slot is real iff its mask is `True`; every other entry is padding, whose values are ignored but
  must be finite (`0·NaN = NaN`). Not the new slots of a zero-padded point, which are real.
  [The contract](uniform_equivalence_contract.md#the-contract)
- **represented tensor** (also: dense tensor, the tensor a T3 represents). The `N0 × … × N(d-1)` array the cores
  contract to. Unless a name says `corewise`, every operation is defined on it, and it is never formed.
  [User guide](user_guide.rst)
- **resize** (`resize`). Change a T3's shape and ranks: zero-pad cores to grow (the represented tensor is unchanged),
  truncate cores to shrink. Growing ranks this way gives rank continuation's warm start. [How to use it](rank_continuation.md#how-to-use-it)
- **shape mask** (`shape_mask`). The `(d, N)` mask of real entries along each padded mode axis, rebuilt on the host
  from the static `shape` ints rather than stored. Always a prefix: the shape never changes under arithmetic.
  [Canonical form vs working form](uniform_masks_vs_ranks.md#canonical-form-vs-working-form)
- **share** (`x.share(sharing)`). Project a T3 onto the shared-factor format: each group's factor is the dominant
  subspace of its concatenated matricizations, then grouped T3-SVD rounding.
  [The format](sharing.md#the-format-an-ordinary-t3-with-tied-factors)
- **shared Tucker factors** (SF-T3; also: shared-factor format; SF-ETT, SF-Tucker in the literature). T3s whose Tucker
  cores are equal within each sharing group. Not a new class: an ordinary `TuckerTensorTrain` with tied factors.
  [Shared Tucker factors](sharing.md)
- **sharing group** (also: group; partition). The modes with one sharing label; the groups partition the modes, and a
  group's modes need equal sizes. Not a `contract` group. [Shared Tucker factors](sharing.md)
- **sharing spec** (`sharing=`). A length-`d` tuple of hashable labels, one per mode; equal labels share one Tucker
  factor. `None` means unshared. [Shared Tucker factors](sharing.md)
- **squash tails** (`squash_tails`). Make the boundary bonds 1 by summing them into the end cores; the represented
  tensor is unchanged. [User guide](user_guide.rst)
- **stack** (also: stacking, batching). Three meanings: (1) the leading `stack_shape` axes on one object's cores;
  (2) `backend/stacking.py`'s conversion between a tree of objects and one stacked object; (3) the uniform supercore.
  [Three meanings](batching_and_stacking.md#1-stacking-means-three-different-things)
- **stack_shape**. The leading batch axes shared by all of an object's cores, `core.shape = stack_shape + (…)`; `()`
  when unstacked. It is the frame stack `C`. [Glossary of batching](batching_and_stacking.md#glossary)
- **stratum**. The fixed-rank manifold a point lies on inside the bounded-rank variety. A uniform element's masks name
  its nominal stratum; zero content (a zero-padded point, a numerical deficiency) can put the tensor on a lower one.
  [The representation](uniform_ranks_and_varieties.md#the-representation-and-what-the-mask-actually-means)
- **supercore** (Tucker supercore, TT supercore). The `d` padded cores of one family stacked on a leading axis:
  Tucker `(d,) + stack_shape + (n, N)`, TT `(d,) + stack_shape + (r, n, r)`. [The layout](uniform_supercore_layout.md#the-layout)
- **sweep** (left-to-right, right-to-left). A pass along the core chain that carries an accumulated contraction or
  factorization from one core to the next, written as a scan: orthogonalization sweeps, the T3-SVD sweep, the frame
  sweep, the rank adjustment sweep. (Test "sweeps" are parameter grids.)
  [Paper ↔ code map](entries_apply_probe.md#8-paper--code-map-probing-t4s-6)
- **T3** (`TuckerTensorTrain`; also: Tucker tensor train, extended tensor train, ETT). A Tucker decomposition whose
  central core is stored as a tensor train: `d` Tucker cores and `d` TT cores. [User guide](user_guide.rst)
- **T3-SVD** (`t3svd`, uniform `ut3svd`, dense `t3svd_dense` / `dense_t3svd`; also: implicit T3-SVD, T4S Algorithm
  10; the dense one is Algorithm 9). Orthogonalize, then one left-to-right truncating sweep of SVDs on the cores;
  returns a left-orthogonal T3 and the Tucker and TT singular values. `ut3svd` takes rank caps only.
  [T3-SVD ranks](t3svd_minimal_ranks.md)
- **tangent stack `K`** (`tangent_stack_shape`). A batch of tangent vectors at one frame; it lives on the variation
  cores only, so a tangent's stack is `K + C`. `K` means nothing else.
  [The three batch blocks](batching_and_stacking.md#2-the-three-batch-blocks-c-w-k)
- **tied** (`TIED` in the contract tables; `has_shared_tucker_factors`, `t3_tie_tucker_factors`; also: untied). The
  Tucker factors of each sharing group are numerically equal. Per-mode operations that treat group modes differently untie them.
  [The format](sharing.md#the-format-an-ordinary-t3-with-tied-factors)
- **truncation spec** (`max_tucker_ranks`, `max_tt_ranks`, `rtol`, `atol`). Rank caps (an int or a per-position
  sequence) and per-step tolerances, shared by `t3svd` and `t3m`.
  [The truncation spec](t3m_methods.md#the-truncation-spec-shared-with-t3svd)
- **TT core** (`tt_cores`; `Gi`). The `d` order-3 cores of shape `stack_shape + (ri, ni, r(i+1))` whose train is the
  central core. [User guide](user_guide.rst)
- **TT rank** (`tt_ranks`, `ri`, length `d+1`; also: bond rank, bond). The size of TT bond `i`: between TT cores
  `i-1` and `i`, with `r0` and `rd` the boundary bonds. [User guide](user_guide.rst)
- **Tucker core** (`tucker_cores`; `Bi`, `Ui` in a frame; also: Tucker factor, Tucker basis, leaf). The `d` matrices
  of shape `stack_shape + (ni, Ni)` mapping mode `i` to its Tucker index. [User guide](user_guide.rst)
- **Tucker rank** (`tucker_ranks`, `ni`, length `d`). The size of the edge between Tucker core `i` and TT core `i`.
  [User guide](user_guide.rst)
- **unfolding** (`X^⟨k⟩`; also: TT unfolding). The matrix formed by cutting a TT–TT edge: the ordered modes split,
  everything left of the split as the rows, everything right as the columns (TT usage). For `N0 × N1 × N2`:
  `1 × (N0N1N2)`, `N0 × (N1N2)`, `(N0N1) × N2`, `(N0N1N2) × 1`, the two vector-shaped ones included or not by
  convention. Its rank is the minimal TT rank of that bond. Compare **matricization**.
  [Notation](contributor/t3svd_verification.md#notation)
- **uniform layer** (`UniformTuckerTensorTrain`, `UT3Frame`, `UT3Variations`, `UT3Tangent`; also: UT3, uniform
  representation). The padded mirror of the ragged layer: supercores plus host edge masks, so operations run under
  `lax.scan` and compile once. Raw `.data = (tucker_supercore, tt_supercore, shape, masks)`.
  [Equivalence contract](uniform_equivalence_contract.md)
- **varying-rank stack**. A uniform stack whose elements have different ranks: allowed along `C`, never along `K`.
  [The tangent layer](uniform_ranks_and_varieties.md#the-tangent-layer-keeps-the-varying-rank-stack--uniform-rank-is-a-k-stack-property-not-a-c-stack-one)
- **zipper** (`tt_zipper_left_to_right`, `tt_zipper_right_to_left`; dotted `tt_zipper_tangent_*`). The `d+1` partial
  contractions of two TT chains over their shared mode indices, accumulated in one direction; it recomputes frame
  centers exactly, by matrix products only. [The `S_i` machinery](contributor/sharing_internals.md#the-s_i-machinery-recompute-by-zipper-never-retain-never-re-svd)

## Frames, tangents and representations

- **aligned padded frame** (`fv_aligned_padded_frame`, `ufv_aligned_padded_frame`; also: alignment, aligned start). The
  padded frame refined by a fixed-point iteration so its new directions capture a hint: each pass projects the hint
  onto the frame's tangent space and gives every store the leading new directions of its content.
  [The alignment](aligned_padded_frame.md#the-alignment)
- **alignment pass** (`T3Tangent.align_padded`, `tv_align_padded_frame`; also: pass, alignment step). One update of the
  stores named in `update` (letters `U`, `O`, `L`, `R`); the others are transported. Returns the new frame and its
  movement. [The alignment](aligned_padded_frame.md#the-alignment)
- **ambient projection** (`project_ambient`). Project an ambient tensor (a T3 or a dense array) onto the tangent
  space at a frame; applied to an ambient gradient it gives the Riemannian gradient.
  [The geometries](numerical_contracts.md#the-geometries-the-semantic-innernorm-live-here)
- **base point** (also: point). The T3 at which a tangent space is attached; the frame encodes it. In the
  derivative examples, "base points X" are sample locations instead.
  [A tangent is a sum of terms](frame_variations.md#a-tangent-is-a-sum-of-d-terms)
- **base-point tangent** (`v_X`; `fv_base_point_tangent`). The point written as a gauged tangent at its own frame: all
  variations zero except the last TT variation, so `‖X‖² = ‖P_last‖²`.
  [`d` cores, `d+1` bonds](frame_variations.md#d-cores-d1-bonds--and-the-extra-ld-r0)
- **block layout** (split `(Na, Rb)`). The uniform alignment's internal layout: every rank slot is `[old block | new
  block]`, each padded to one width with its own mask; `(Na, Rb)` are the old block's Tucker and bond widths.
  [The uniform layer](aligned_padded_frame.md#the-uniform-layer)
- **captured hint energy** (`E = ‖Π_{T(F)} hint‖²`). How much of the hint a frame's tangent space holds. The
  alignment's fixed point is stationary for it but not a maximum: a saddle (proved for matrices; on T3s, stationarity
  and the upward single-store rotations are measured). [The alignment](aligned_padded_frame.md#the-alignment)
- **carrier** (`left[d-1]`, `right[0]`; also: `Ld` / `R0`, frame-internal padding cores, base-point padding). The last
  left and first right frame cores. They hold the base point, not an edge's frame; a padded frame keeps their old
  content only. [`d` cores, `d+1` bonds](frame_variations.md#d-cores-d1-bonds--and-the-extra-ld-r0)
- **compaction** (`ufv_compact_frame`). Rewrite a block-layout frame at a level's widths, real slots first with prefix
  masks, so Newton reuses its compiled programs. [The uniform layer](aligned_padded_frame.md#the-uniform-layer)
- **completion** (also: orthonormal completion). The orthonormal content of a store's new block, orthogonal to the
  old block. Also: the orthonormal vectors an SVD supplies at zero singular values.
  [The padded frame](aligned_padded_frame.md#the-padded-frame)
- **cone merit** (`m_c(F)`). The share of the hint's normal part that the new directions represent on one side of
  every edge. Measured fixed points of the alignment are strict local maxima of it, one of many.
  [The alignment](aligned_padded_frame.md#the-alignment)
- **`COREWISE`** (`CorewiseGeometry`, `CorewiseGeometryOps`). The Euclidean geometry of the raw cores: corewise frame,
  identity projection, additive retraction. Its Gauss-Newton Hessian is gauge-singular.
  [Choosing a geometry](fitting_and_optimization.md#24-choosing-a-geometry)
- **corewise frame** (`(U, G, G, G)`; `t3_corewise_frame`). The non-orthonormal frame whose down, left and right cores
  are the TT cores, so a tangent is a raw core perturbation. `COREWISE` uses it.
  [Corewise transpose](transposes.md#corewise--xapply_corewise_transposec-ww-sum_over_probes--instance)
- **doubled-rank representation** (`T3Tangent.to_t3`, `tv_to_t3`; `include_shift`). The tangent as one T3 of about
  twice the frame's ranks; `include_shift=True` adds the base point. [Frame-variations](frame_variations.md)
- **empty frame** (`fv_empty_frame`). The frame of a rank-0 T3: every rank axis has width 0 except the boundary bonds,
  so every slot of a padded frame built from it is a completion. [The zero point](aligned_padded_frame.md#the-zero-point)
- **fixed point** (of the alignment). Every store's completion spans the leading new directions of its own content,
  given the others, so a pass moves nothing. Stationary for the captured hint energy but a saddle of it, per store a
  minimum (proved for matrices); measured to be a strict local maximum of the cone merit.
  [The alignment](aligned_padded_frame.md#the-alignment)
- **frame** (`T3Frame`, `UT3Frame`; also: orthogonal frame, orthonormal frame). The four stores of orthonormal frame
  cores that encode a base point and span its tangent space. Not a basis: at a non-minimal point it is overcomplete.
  [Frame-variations](frame_variations.md)
- **frame cores** (up `Ui`, down `Oi`, left `Li`, right `Ri`; `up_tucker_cores`, `down_tt_cores`, `left_tt_cores`,
  `right_tt_cores`; also: `P`/`Q` for left/right, `D` or "outer" for down). Up: orthonormal Tucker cores. Down:
  TT-shaped `(rL, nD, rR)`, the TT core of the Tucker-variation terms (`Vi` varies in the complement of `Ui`'s rows).
  Left/right: the orthogonal TT environments. [Frame-variations](frame_variations.md#a-tangent-is-a-sum-of-d-terms)
- **frame-variations format** (`frame_variations_format`, `fv_` / `tv_`; also: frame-variations representation,
  (frame, variations) pair). A tangent stored as a frame plus variations; `fv_` treats the pair as indexed pieces,
  `tv_` as the sum of its terms. [Frame-variations](frame_variations.md)
- **gauge conditions** (`is_gauged`, `gauge_residual`; also: gauged). `Ui Viᵀ = 0` for every `i` (code layout), and
  each `Li` left-orthogonal against `Hi` for `i < d-1` (T4S eqs. 47–48). They make the variations coordinates.
  [Gauge conditions](frame_variations.md#gauge-conditions-make-the-variations-coordinates)
- **gauge freedom** (also: gauge, up to gauge). The non-uniqueness of a representation: any invertible change on an
  internal edge, e.g. column signs or rotations at a singular-value tie. Compare gauge-invariant quantities.
  [Gauge-carrying operations](uniform_equivalence_contract.md#gauge-carrying-operations)
- **gauge projection** (`Π`; also: gauge projector). Map raw variations to gauged ones; two kinds. *Orthogonal*
  (`MANIFOLD.project`, `tv_orthogonal_gauge_projection`): the coordinate-orthogonal projection onto the gauged
  subspace; it keeps the gauged part and changes the represented tangent, and applied after a bare transpose it gives
  the Riemannian gradient. *Oblique* (`MANIFOLD.project_oblique`, `tv_oblique_gauge_projection`; also: oblique
  projection): not orthogonal; it enforces the gauge while preserving the represented tangent, compensating through
  the down and right cores. (An ambient tensor goes through **ambient projection** instead.)
  [Notes](entries_apply_probe.md#4-the-riemannian--tangent-versions-and-their-transposes)
- **geometry** (`MANIFOLD`, `COREWISE`, `UNIFORM_MANIFOLD`, `UNIFORM_COREWISE`; backend `*GeometryOps`; protocol
  `optimizers.Geometry`). Where optimization happens: the frame, the projection `Π`, the retraction and the metric,
  bundled. [Choosing a geometry](fitting_and_optimization.md#24-choosing-a-geometry)
- **geometry aux** (`precompute`, `geometry_aux`, `geom_aux`; also: companion). Per-frame data a geometry computes
  once per local model and hands to `project`/`retract`; `None` for the plain geometries.
  [The pieces](fitting_and_optimization.md#3-the-pieces-structure)
- **Grasedyck–Kramer metric** (also: Grasedyck–Kramer preconditioner, inverse-σ weighting). Tangent-coordinate weights
  `1/σ` from the base point's singular values, penalizing poorly determined directions. [Edge weighting](weighting.md)
- **hint**. A tensor whose projection onto the frame's tangent space the alignment makes large: a T3, a dense array, or
  anything the `project` callable accepts. `newton_cg` uses the misfit gradient. [The alignment](aligned_padded_frame.md#the-alignment)
- **hint-free start** (`T3Frame.padded`, `UT3Frame.padded`; `max_passes=0`; also: seeded start, hint-free padded
  frame). The padded frame by a data-independent convention: seed rows completed against the old block by a
  sign-fixed QR. [The padded frame](aligned_padded_frame.md#the-padded-frame)
- **manifold** (also: fixed-rank T3 manifold, `M`, `T_xM` its tangent space). The set of T3s with fixed minimal ranks,
  an embedded submanifold of the dense tensor space. [User guide](user_guide.rst)
- **`MANIFOLD`** (`ManifoldGeometry`, `ManifoldGeometryOps`). The Riemannian geometry: orthonormal gauged frame, HS
  metric, gauge projection, T3-SVD retraction. [Choosing a geometry](fitting_and_optimization.md#24-choosing-a-geometry)
- **manifold dimension** (`manifold_dim`, `tangent_space_dimension`). The dimension of the fixed-rank (or shared)
  manifold at given ranks; correct for any ranks. [Rank machinery](sharing.md#rank-machinery-minimal-ranks-dimension-continuation)
- **movement** (`start_movements`, `cone_movements`). How far a pass turned the new directions: the largest over stores
  of `‖g_old − P⁺P⁺ᵀ g_old‖_F`; 0 at a fixed point. `ufv_aligned_padded_frame` returns a `(max_passes,)+C` array, NaN
  after the last pass; the `stats` keys hold one float per pass run. [The schedule](aligned_padded_frame.md#the-schedule)
- **old block / new block** (also: A-block / P-block, old slots / new slots). In each store of a padded frame, the
  leading slots holding `x_old`'s frame bitwise, and the trailing slots the rank increase adds.
  [The padded frame](aligned_padded_frame.md#the-padded-frame)
- **orthogonal representation** (`t3_orthogonal_representations`, `ut3_…`; `t3svd_orthogonal_representations`). A point
  factored into an orthonormal frame plus gauged variations (which include the base-point term). The `t3svd_` form
  uses the T3-SVD gauge and also returns the singular values. [Frame-variations](frame_variations.md)
- **padded frame** (`fv_padded_frame`). A frame of `x0` (= `x_old` zero-padded) that keeps `x_old`'s frame bitwise in
  the old block of every store and fills the new block with an orthonormal completion.
  [The padded frame](aligned_padded_frame.md#the-padded-frame)
- **pinned completion** (`linalg.pinned_completion`; masked `pad_safe_pinned_completion`). The `k` columns that
  continue a fixed orthonormal block to an orthonormal set, from seeds in order: the sign-fixed Householder QR of
  `[pinned | seeds]`, with no tolerance. The hint-free start's completion.
  [Why a QR](contributor/padded_frame_internals.md#the-start-the-pad-safe-pinned-completion-and-why-a-qr)
- **representation dependence**. The hint-free start depends on how `x_old`'s frame is represented, slot signs and
  bases at singular-value ties included, so layers and BLAS builds differ; the aligned frame does not, up to `tol`.
  [Representation dependence](aligned_padded_frame.md#representation-dependence)
- **retraction** (`retract`). Map a base point plus a tangent step back onto the set. `MANIFOLD`: truncate the shifted
  doubled-rank T3 to the frame's ranks by the implicit T3-SVD. `COREWISE`: add the step to the cores.
  [Choosing a geometry](fitting_and_optimization.md#24-choosing-a-geometry)
- **same-frame** (`safety.frames_equal`; also: same tangent space, SF in the contract tables). Two tangents share a
  tangent space iff their frames are numerically equal (identity is a fast path). Checked in safe mode.
  [`T3Tangent`](numerical_contracts.md#t3tangent)
- **same-mask precondition**. Uniform tangent add/subtract and weight absorption require equal masks, a structural
  host-side check, since padding would hide a mismatch. [Tangent masks](uniform_masks_vs_ranks.md#tangent-variation-masks-a-different-algebra)
- **schedule** (`oversampling`, `oversampled_passes`, `max_passes`, `tol`, `alternate`; mirrored by `padded_*`,
  `cone_*`). Oversampled passes, one truncating pass, then two-colour pairs until both movements of a pair are below
  `tol` (0.05) or `max_passes` (10) is reached. [The schedule](aligned_padded_frame.md#the-schedule)
- **seed rows** (`padded_seed`, `cone_seed`, `ZERO_FRAME_SEED`; also: keyed Gaussian stream, slot-indexed seeds). New
  slot `s` of a store is seeded by row `s` of a Gaussian stream keyed by seed, store and mode (or a group's first mode).
  [The padded frame](aligned_padded_frame.md#the-padded-frame)
- **shared geometry** (`shared(geometry, sharing)`, `shared_manifold`, `shared_corewise`; `SharedGeometry`). Any of the
  four geometries restricted to tied Tucker factors: projections land on the tied tangent subspace.
  [Optimizing over shared factors](sharing.md#optimizing-over-shared-factors)
- **slack** (Tucker slack, TT slack; also: excess rank). Rank requested beyond what the network supports, carried as
  differences between the stores' ranks. Geometrically inert; the first retraction collapses it.
  [Four rank stores](frame_variations.md#four-rank-stores--and-why-excess-rank-is-inert)
- **store** (`up_ranks`, `down_ranks`, `left_ranks`, `right_ranks`; also: rank store, orthogonal-representation
  ranks). One of the frame's four core families, each orthonormal in its own matricization or unfolding. Its ranks are
  set by the orthogonalization sweep (up, left, right over the left chain, then down); the differences are the slack.
  [Four rank stores](frame_variations.md#four-rank-stores--and-why-excess-rank-is-inert)
- **store transport**. In an alignment pass, a store not named in `update` keeps its completion, re-expressed in the
  new coordinates. Unrelated to vector transport. [Internals](contributor/padded_frame_internals.md#the-step-as-message-passing)
- **structurally zero** (also: zero point). A T3 with some Tucker rank or interior TT rank 0, hence the zero tensor.
  Its frame is the empty frame. [The zero point](aligned_padded_frame.md#the-zero-point)
- **T3-SVD gauge**. The frame whose Tucker bases are the T3-SVD's singular bases, as `t3svd_orthogonal_representations`
  returns it with its singular values. [`T3FrameWeights`](weighting.md#t3frameweights--a-metric-on-a-tangent)
- **tangent** (`T3Tangent`, `UT3Tangent`, `tv_`; also: tangent vector). A (frame, variations) pair representing a sum
  of `2d` terms, each with one variation core in place of a frame core (T4S eq. 46).
  [A tangent is a sum of terms](frame_variations.md#a-tangent-is-a-sum-of-d-terms)
- **tangent cone**. At a point of lower rank than the target ranks (the zero tensor, a zero-padded point), the
  first-order directions into the tensors of at most those ranks: a union of linear spaces, not one tangent space.
  The cone merit and the cone reference are named for it. [The math note](padded_frame_cone.pdf)
- **tangent metric** (`T3FrameWeights`, `UT3FrameWeights`). Diagonal weights on a tangent's coordinates (`up`, `down`,
  `left`, `right`, each length `d`), absorbed into the variations but batched with the frame stack `C`.
  [`T3FrameWeights`](weighting.md#t3frameweights--a-metric-on-a-tangent)
- **tied projection** (`Π_sh`; `SharedGeometry.project`). The projection onto the tied tangent subspace in the
  geometry's own metric: a tilted post-pass on the manifold, the per-group mean on corewise.
  [Optimizing over shared factors](sharing.md#optimizing-over-shared-factors)
- **two-colour pass** (`'UL'`, `'RO'`; `alternate=True`). A pass updating the up and left stores, then one updating
  right and down, the other colour transported; avoids the period-2 cycle of simultaneous updates.
  [The schedule](aligned_padded_frame.md#the-schedule)
- **variation masks** (`ufv_variation_masks`; also: gauge-shifted masks). A uniform tangent's variation masks, derived
  from the frame's as `(up, down, left[:-1], right[1:])`; tangent `+`/`−`/scale leave them unchanged.
  [Tangent masks](uniform_masks_vs_ranks.md#tangent-variation-masks-a-different-algebra)
- **variations** (`T3Variations`, `UT3Variations`; `tucker_variations` `Vi`, `tt_variations` `Hi`; also: variation
  cores, `δU`/`δG`). The cores that fill the frame's holes and encode a tangent's direction. Gauged, they are
  orthonormal coordinates. [Frame-variations](frame_variations.md)
- **vector transport** (`MANIFOLD.transport`). Move a tangent to the tangent space at another frame by projection.
  [The geometries](numerical_contracts.md#the-geometries-the-semantic-innernorm-live-here)
- **zero-point frame** (`fv_zero_point_frame`; also: canonical zero-point frame). The fixed frame at the zero tensor:
  seeded Haar (uniformly random orthonormal) Tucker cores, one per sharing group, and identity-prefix TT cores. The
  `'zero'` reference uses it.
  [The zero-point frame](contributor/curvature_internals.md#the-canonical-zero-point-frame-the-default-newton-reference)

## Fitting and sampling

- **ambient transpose** (`*_ambient_transpose`). The adjoint of a sampling operation on the full tensor space,
  returned as CP factors; static and frame-free. [Transposes](transposes.md)
- **apply** (`apply`, `t3_apply`, `tv_apply`). The multilinear form `X(w0, …, w(d-1))`: every mode contracted with a
  vector, one scalar per sample. [The three operations](entries_apply_probe.md#1-the-three-operations-precisely)
- **assembly** (also: variation-gradient assembly). The transpose stage that forms the variation gradients from the
  edge-variable jets; its memory is linear in `|W|`, which `chunk_size` bounds. [Where the memory goes](chunking.md#where-the-memory-goes)
- **back-projection**. A sampling adjoint applied to a residual, e.g. `c · (w0 ⊗ … ⊗ w(d-1))` for `apply`.
  [Transposes](transposes.md)
- **bare Jacobian** (`𝒥`, `𝒥ᵀ`; the tangent `entries`/`apply`/`probe` and their transposes; also: `𝒥⁽ˢ⁾`,
  "Riemannian Jacobian" in older pages). The tangent sampling map applied to variations, with no gauge projection.
  [Notes](entries_apply_probe.md#4-the-riemannian--tangent-versions-and-their-transposes)
- **`chunk_size`** (`'auto'`, `estimate_chunk_size`, `max_chunk_size_within`; also: chunking, W-chunking). Compute the
  probe-derivative transpose's assembly in slices of that many probes: the same numbers, a lower peak memory on the
  uniform + jax path. [What `chunk_size` does](chunking.md#what-chunk_size-does)
- **containment** (`probe ⊃ apply ⊃ entries`). `apply` is `probe` with the free mode contracted too; `entries` is
  `apply` with one-hot vectors. [How they relate](entries_apply_probe.md#2-how-they-relate-the-containment-hierarchy)
- **corewise transpose** (`*_corewise_transpose`). The gradient of a sampling operation with respect to a T3's own
  cores, as a raw `(tucker_grads, tt_grads)` tuple; for Adam or L-BFGS. [Transposes](transposes.md)
- **data** (`data=`; `y`, also `b`). The observed measurements, laid out like the forward `S(x)`. Not the `.data` raw
  tuple. [Drive a library optimizer](fitting_and_optimization.md#21-drive-a-library-optimizer-the-common-case)
- **draw** (`draw(rng)`, `flat_draw`; also: minibatch). A user function `rng → (sample_B, data_B)` called each step by
  `mc_sgd`/`adam`; the default draws `batch` measurements uniformly from the flattened `W`.
  [Custom minibatching](fitting_and_optimization.md#23-custom-minibatching--the-draw)
- **edge variables** (`xi`, `mu`, `nu`, `eta`; tangent `sigma`, `tau`, `dxi`, `deta`; adjoint `*_tilde`). The partial
  contractions of a sample through the network, one per edge: `ξ` up the Tucker cores, `μ`/`ν` along the left/right
  TT sweeps, `η` combined at each free mode. [Paper ↔ code map](entries_apply_probe.md#8-paper--code-map-probing-t4s-6)
- **entries** (`entries`, `index`). Read off `X[i0, …, i(d-1)]` at multi-indices `index` of shape `(d,) + W`; a scalar
  per index. Fitting from entries is tensor completion. [The three operations](entries_apply_probe.md#1-the-three-operations-precisely)
- **frame sweep** (`sweep`, `SamplingKind.precompute`, `*_from_sweep`; lean `(xi, mu)`, full `(xi, mu, nu, eta)`). The
  frame's edge variables for a sample, computed once per local model and reused by every `J`/`Jᵀ`.
  [Frame-sweep reuse](fitting_and_optimization.md#43-frame-sweep-reuse-inner-iterations-are-cheap)
- **Jacobian** (`J = 𝒥 ∘ Π`, `Jᵀ = Π ∘ 𝒥ᵀ`). The Riemannian forward and adjoint of a fit: the bare Jacobian composed
  with the gauge projection. `Jᵀr` is the Gauss-Newton gradient.
  [The one idea](fitting_and_optimization.md#1-the-one-idea-a-fit-factors-into-four-orthogonal-axes)
- **jet** (order axis `t`; `order`). The order-stacked derivatives `(y⁽⁰⁾, …, y⁽ᵒʳᵈᵉʳ⁾)` on a leading axis of length
  `order + 1`; order 0 is the plain operation. The maximum order is spelled `order`, never `K`.
  [Index letters](naming_conventions.md#index-letters-stacks-vs-derivative-order)
- **measurement space**. Where residuals live: one scalar per sample for `entries`/`apply`, `d` vectors per sample for
  `probe`, with the plain sum-of-products inner product. [Measurement spaces](entries_apply_probe.md#5-measurement-spaces-for-fitting)
- **misfit**. The data term of the objective, `½‖ω ⊙ (S(x) − y)‖²`. [Regularization](fitting_and_optimization.md#49-regularization-an-optional-objective-term)
- **multilinear extension** (also: Taylor data, grid corner). `entries_derivatives` returns the Taylor data of the
  represented tensor's multilinear extension at the grid corner `index`: the jet of `apply(e_index + sP)`.
  [Symmetric directional derivatives](entries_apply_probe.md#symmetric-directional-derivatives-a-fourth-dimension)
- **objective** (`f`). The misfit plus the optional regularizer, `½‖ω ⊙ r‖² + ρ(x)`; the history splits it as
  `obj = misfit + reg`. [Regularization](fitting_and_optimization.md#49-regularization-an-optional-objective-term)
- **one-hole / two-hole sweep** (`corewise_hessian`, `t3_<kind>_corewise_hessian`). One-hole: leave one core out (the
  corewise transpose). Two-hole: its derivative along a core direction, a second core replaced by its direction; the
  exact Hessian's sampling term. [Math note](exact_hessian_two_hole.pdf)
- **perturbation vectors** (`pp`; also: directions `P`, perturbation directions). The `d` vectors along which the
  derivatives are taken, one direction per sample, on the same stack `W` as `ww` or `index`.
  [Symmetric directional derivatives](entries_apply_probe.md#symmetric-directional-derivatives-a-fourth-dimension)
- **probe** (`probe`, `t3_probe`, `tv_probe`). Contract all modes but one, for each mode in turn: `d` vectors per
  sample. "A probe" also means one sample row. [The three operations](entries_apply_probe.md#1-the-three-operations-precisely)
- **probe vectors** (`ww`, `w_i`; also: test vectors, measurement vectors, probe rows). A length-`d` list of arrays
  `W + (Ni,)` that `apply` and `probe` contract with. [Notation](entries_apply_probe.md#notation)
- **pushthrough / combine** (jet kernels). Pushthrough carries a jet across one core in a sweep (the `μ`, `ν` jets);
  combine forms a free mode's output from left and right jets (`η`). Both are binomial jet products.
  [Math note](https://github.com/NickAlger/T3Toolbox/blob/main/docs/symmetric_probe_derivatives.tex)
- **regularizer** (`regularizer=`, `ρ(x)`; `IdentityRegularizer(λ)`, `Regularizer`). An optional objective term.
  `IdentityRegularizer` is `½λ‖x‖²`: an HS ridge on `MANIFOLD`, weight decay on `COREWISE`; scaled by `batch/n` in
  minibatch steps. [Regularization](fitting_and_optimization.md#49-regularization-an-optional-objective-term)
- **residual** (`r`). `r = S(x) − y`, unweighted; a list of `d` arrays for probe kinds, jets for derivative kinds. Not a
  checker's `_residual`. [Measurement spaces](entries_apply_probe.md#5-measurement-spaces-for-fitting)
- **residual weight** (`weight=`, `ω`; per-order, per-mode). A per-`(mode, order)` scaling in the misfit `½‖ω ⊙ r‖²`; a
  bare vector is per-order. Unrelated to edge weights (`weights`, `*Weights`).
  [The residual weight](fitting_and_optimization.md#46-the-residual-weight-ω-is-a-per-order--per-mode-matrix-owned-by-the-kind)
- **sample** (`sample=`; also: measurement). The measurement inputs: `ww` or `index`, or `(ww, pp)` / `(index, pp)` for
  derivative kinds. Their stack is `W`. [Drive a library optimizer](fitting_and_optimization.md#21-drive-a-library-optimizer-the-common-case)
- **sampling kind** (`kind`, `'apply'` … `'probe_derivatives'`; `SamplingKind`; also: kind). What a fit measures. A
  kind implements six operations: `precompute`, `forward` (bare `𝒥`), `transpose` (bare `𝒥ᵀ`), `point_forward`
  (`S(x)`), `take` (a minibatch gather) and `corewise_hessian` (exact Hessian only), plus norm reductions.
  [The pieces](fitting_and_optimization.md#3-the-pieces-structure)
- **sampling operations** (`entries`, `apply`, `probe`). The three ways to evaluate the represented tensor without
  forming it, by contraction with vectors in all or all-but-one modes. [Entries, applies, probes](entries_apply_probe.md)
- **`sum_over_probes`**. Transpose flag. `False` (default) keeps `W` as an output stack; `True` sums over it, giving
  the Gauss-Newton gradient `Jᵀr`. [Transposes and `sum_over_probes`](batching_and_stacking.md#11-transposes-adjoints-and-sum_over_probes)
- **symmetric directional derivatives** (`*_derivatives`, `'*_derivatives'` kinds). `y⁽ᵗ⁾ = dᵗ/dsᵗ op(X + sP)` at
  `s = 0`, for `t = 0..order`, in one repeated direction `P`; returned as a jet.
  [Symmetric directional derivatives](entries_apply_probe.md#symmetric-directional-derivatives-a-fourth-dimension)
- **tangent transpose** (`T3Tangent.*_transpose`; also: `𝒥ᵀ`). Back-project a residual into a tangent at a frame,
  without gauge projection. [Transposes](transposes.md)
- **`trs`** (also: binomial convolution tensor; `_trs` twins; banded recurrence, order scan). `trs[t,r,s] =
  C(t,r)·[r+s=t]`, the jet product rule. Shipped kernels never contract it (a banded recurrence or an order scan);
  dense `_trs` twins live in the tests. [Semantic markers](naming_conventions.md#semantic-markers-load-bearing--you-can-trust-them)

## Optimization

- **`adam`**. Adam over the cores, for the corewise geometry; on `MANIFOLD` it runs but warns, since its moments are
  gauge-dependent. [Drive a library optimizer](fitting_and_optimization.md#21-drive-a-library-optimizer-the-common-case)
- **Armijo line search** (`step_control='line_search'`, `c_armijo`, `on_line_search_failure`). Halve the step until
  `f(retract(αp)) ≤ f + c_armijo·α·⟨g, p⟩`, for up to 40 trials in `newton_cg` and 50 in `gradient_descent`. `'stop'`
  ends the run when all fail; `'accept'` takes the last trial.
  [Practical guidance](fitting_and_optimization.md#5-practical-guidance-from-the-field)
- **Cauchy step**. The step length `α = ‖g‖² / ‖𝒥g‖²` that minimizes the Gauss-Newton quadratic along `−g`;
  `gradient_descent` starts from it and `mc_sgd` takes it. [Drive a library optimizer](fitting_and_optimization.md#21-drive-a-library-optimizer-the-common-case)
- **cone reference** (`g0_reference='cone'`; `cone_*` options). `‖Π_{T(F)} g(0)‖`: the misfit-gradient norm at the
  zero tensor of `x0`'s ranks, on the frame aligned with that gradient. `cone_max_passes=0` uses the hint-free start
  there, a different frame from `'zero'`'s, so the two values differ. [In the optimizers](aligned_padded_frame.md#in-the-optimizers)
- **`continuation_ranks`** (`compute_continuation_ranks`; `tau`, `n_chunk`, `max_grow`, `kappa_guard`; also:
  condition-number continuation). Propose the next ranks from the iterate's spectra, growing edges with `κi` at least a
  factor `tau` below the worst; the current ranks mean stop. [The idea](rank_continuation.md#the-idea-section-541)
- **curvature damping** (`curvature_damping`, `ε`; `lift_eps`). The lift's damping relative to `‖X‖_HS` (default
  `1e-6`): directions with lift singular value well above it get the exact curvature, smaller ones are damped toward
  Gauss-Newton. A solver option, not a geometry field. [The exact Hessian](fitting_and_optimization.md#412-the-exact-hessian-and-the-trust-region)
- **curvature term** (`curvature_hessian`; also: Weingarten term). The exact Hessian's addition to Gauss-Newton,
  `Π Lᵀ(K_{Aᵀω²r} − K_{grad f}) L Π`: the residual's normal part paired with the manifold's second fundamental form.
  It can be negative. [The exact Hessian](fitting_and_optimization.md#412-the-exact-hessian-and-the-trust-region)
- **edge condition number** (`κi`, `ranks.edge_condition_numbers`; group `κ_g`). For an edge of structural rank r,
  the ratio σ₁/σ_r of the matrix cut at that edge (a matricization for a Tucker–TT edge, an unfolding for a TT–TT
  edge), over its first r singular values only: the cut matrix has at most r nonzero ones, so the ordinary condition
  number would be infinite. It is a property of the representation, not of the tensor: the same tensor carried at a
  larger r has σ_r = 0 and an infinite edge condition number.
  [`ranks.edge_condition_numbers`](rank_continuation.md#the-idea-section-541)
- **exact Hessian** (`hessian='exact'`; also: exact Riemannian Hessian, exact model). Gauss-Newton plus the curvature
  term; indefinite away from a minimizer, so it defaults to the trust region, and gives a superlinear tail when the
  residual does not vanish. [The exact Hessian](fitting_and_optimization.md#412-the-exact-hessian-and-the-trust-region)
- **flat slot** (also: maximal-rank slot). A slot at its rank ceiling (Tucker rank = mode size, or TT rank = product of
  its neighbours); the curvature term has no component there. At `d = 1` every slot is flat.
  [The exact Hessian](fitting_and_optimization.md#412-the-exact-hessian-and-the-trust-region)
- **forcing term** (`η`; `cg_forcing_power`). CG stops at relative residual `η = min(0.5, (‖g‖/‖g0‖)^power)`; the
  default power is 0.5, and 1.0 makes the exact Hessian's tail q-quadratic (the error squares each step).
  [Practical guidance](fitting_and_optimization.md#5-practical-guidance-from-the-field)
- **g0 reference** (`g0_reference`: `None`, `'zero'`, `'warm_start'`, `'cone'`; `g0norm_newton`, `g0norm_cg`; also:
  Newton reference, `‖g0‖`). The gradient norm the relative stop and forcing term divide by. `None` is the geometry's
  own: `'zero'` on manifold geometries, `'warm_start'` on corewise ones. Explicit `g0norm_*` values win.
  [Practical guidance](fitting_and_optimization.md#5-practical-guidance-from-the-field)
- **Gauss-Newton model** (`hessian='gauss_newton'`; `gn_hessian`, `gn_quadratic`). `m(p) = c + gᵀp + ½pᵀHp` with
  `H = Π𝒥ᵀω²𝒥Π` (plus the regularizer's); exact on the tangent space, positive semidefinite.
  [The Gauss-Newton model](fitting_and_optimization.md#42-the-gauss-newton-model-is-exact-not-an-approximation)
- **`gradient_descent`**. Steepest descent with a Cauchy initial step and an Armijo line search; eager only, the
  reference optimizer. [Drive a library optimizer](fitting_and_optimization.md#21-drive-a-library-optimizer-the-common-case)
- **`gtol_rel`** (also: relative stop). `newton_cg` and `gradient_descent` stop when `‖g‖ ≤ gtol_rel · ‖g0‖` (default
  `1e-8`); there is no `gtol`. [Practical guidance](fitting_and_optimization.md#5-practical-guidance-from-the-field)
- **history** (`stats['history']`, `NewtonInfo`). One record per Newton iteration: objective split, gradient norm,
  CG and step-control outcome, reference kinds, trust-region and curvature fields.
  [Drive a library optimizer](fitting_and_optimization.md#21-drive-a-library-optimizer-the-common-case)
- **ladder** (also: continuation ladder). The sequence of levels of one rank-continuation run.
  [Rank continuation](rank_continuation.md)
- **`least_squares_problem`** (`LeastSquaresSetup`: `.problem`, `.x0_cores`, `.rewrap`). Build the backend `Problem`
  from the frontend vocabulary, as the four optimizers do. [The pieces](fitting_and_optimization.md#3-the-pieces-structure)
- **level** (also: rank level, continuation level). One fixed-rank fit within rank continuation, at one rank set.
  [Rank continuation](rank_continuation.md)
- **lift** (`L`; `tv_corewise_lift`, `LiftFactors`, `fv_lift_factors`; `lift_sigma_min`). Map gauged frame coordinates
  to raw core directions of the base point, `θ̂ = L v`, through damped pseudo-inverses; `lift_sigma_min` is the
  smallest singular value it inverts. [The exact Hessian](fitting_and_optimization.md#412-the-exact-hessian-and-the-trust-region)
- **local model** (`fitting.LocalModel`, `backend.optimizers.LocalModel`; formerly `GaussNewtonModel`). The objective's
  model at a base point, over one geometry: value, gradient, Hessian actions and retraction, with the frame sweep
  computed once. Built by the `*_model` factories. [Build the local model](fitting_and_optimization.md#22-build-the-local-model-roll-your-own-optimizer)
- **`mc_sgd`** (also: Manifold Cauchy SGD, MC-SGD). Each step draws a minibatch, takes the Cauchy step along the
  gradient, and retracts; tuning-free (T4S §5.3.2). [Drive a library optimizer](fitting_and_optimization.md#21-drive-a-library-optimizer-the-common-case)
- **`newton_cg`** (also: inexact Riemannian Newton-CG). Each step builds the local model, solves `Hp = −g` by CG to the
  forcing tolerance, then a line search or trust region. [Drive a library optimizer](fitting_and_optimization.md#21-drive-a-library-optimizer-the-common-case)
- **oversampled ranks** (`oversampled_ranks`; also: working ranks). The ranks plus `p` on every Tucker rank and
  interior bond, made structurally minimal and never below the input: the alignment's ranks before truncation.
  [The schedule](aligned_padded_frame.md#the-schedule)
- **padded start** (`newton_cg(padded_from=x_old)`, `padded_*` options, `start_frame=`; backend `padded_start`).
  Iteration 0's frame: the aligned padded frame of `x_old` with the misfit gradient as hint, or a given `start_frame`.
  [In the optimizers](aligned_padded_frame.md#in-the-optimizers)
- **`Problem`** (backend `optimizers.Problem`). The least-squares oracle `(geometry, kind, sample, data)`;
  `local_model(x)` linearizes at a point. [The pieces](fitting_and_optimization.md#3-the-pieces-structure)
- **rank continuation**. Fit at small ranks, grow the ranks, re-fit warm-started, and stop when more rank stops
  helping. The library proposes the ranks; the user drives the loop. [Rank continuation](rank_continuation.md)
- **stacked point** (also: stacked optimization). A point with a frame stack `C`: the local model accepts it; the four
  optimizers raise `NotImplementedError`. [Stacked points](fitting_and_optimization.md#410-stacked-points-build-the-model-not-the-fit)
- **stats** (`newton_steps`, `newton_accepted`, `stop_reason`, `history`, `losses`; `start_frame`, `start_movements`,
  `cone_movements`). The dict an optimizer returns. `newton_steps` counts CG solves, `newton_accepted` steps taken.
  [Drive a library optimizer](fitting_and_optimization.md#21-drive-a-library-optimizer-the-common-case)
- **step control** (`step_control`: `'line_search'`, `'trust_region'`). How a Newton step is accepted: Armijo
  backtracking (Gauss-Newton default) or a trust region (exact default); either works with either Hessian.
  [The exact Hessian](fitting_and_optimization.md#412-the-exact-hessian-and-the-trust-region)
- **stop reason** (`'gradient'`, `'max_newton'`, `'line_search'`, `'objective_floor'`, `'trust_region_floor'`). Why
  `newton_cg` stopped. `'objective_floor'` is not a failure: the predicted change fell below the objective's
  floating-point resolution. [Drive a library optimizer](fitting_and_optimization.md#21-drive-a-library-optimizer-the-common-case)
- **tied lift** (`fv_tied_lift_factors`). The lift on a shared point: each group's coordinates lift to one raw factor
  direction; a per-mode lift is not a tied direction. [Optimizing over shared factors](sharing.md#optimizing-over-shared-factors)
- **trust region** (Steihaug–Toint CG; `tr_radius0` = `Δ0`; ratio `ρ`). CG inside `‖p‖ ≤ Δ`, then one trial
  retraction accepted iff `ρ = Δf / (gᵀp + ½pᵀHp) > 0.1`; `Δ0` defaults to the Gauss-Newton Cauchy length.
  [The exact Hessian](fitting_and_optimization.md#412-the-exact-hessian-and-the-trust-region)
- **uniform continuation**. Growing every Tucker rank and TT bond together; the baseline the condition-number scheme
  improves on. Unrelated to the uniform layer. [Why not grow every rank together?](rank_continuation.md#why-not-just-grow-every-rank-together)
- **useless-rank removal**. Make proposed ranks non-degenerate from shapes and ranks alone. (The *uniform bump*, a
  separate step, is the fallback growth when every edge is comparably conditioned.)
  [The idea](rank_continuation.md#the-idea-section-541)
- **verbose display** (`verbose=True`, `callback`, `make_newton_display`, `val_sample`/`val_data`). A per-iteration
  block with a per-`(mode, order)` relative-error table, train and validation; `callback(NewtonInfo)` replaces it.
  [Drive a library optimizer](fitting_and_optimization.md#21-drive-a-library-optimizer-the-common-case)
- **warm start** (also: zero-padded start, padded restart, continuation restart, zero-padded rank increase). Starting
  a level from the previous level's point zero-padded to the new ranks by `resize`: the same tensor, numerically rank
  deficient. (The `'warm_start'` g0 reference, `‖g(x0)‖`, is named after it.) [How to use it](rank_continuation.md#how-to-use-it)
- **`x0`** (also: initial guess, start point). The starting point; its class picks the layer, and the geometry must
  match. A zero start is fine on the manifold, not on `COREWISE`.
  [A zero start](fitting_and_optimization.md#411-a-zero-start-makes-the-ragged-and-uniform-layers-diverge)

## Conventions and testing

- **`aux_data`** (also: static structure; vs pytree leaf). The static part of a jax pytree, part of the jit cache key:
  masks, geometries, kinds. Leaves are traced data: cores, sweeps, samples, a tangent's frame.
  [jax pytrees and `vmap`](batching_and_stacking.md#7-jax-pytrees-and-vmap--how-batching-meets-autodiff)
- **backend** (`t3toolbox.backend.*`). Stateless functions on raw `.data` tuples (plus classes that only bind
  parameters), imported by submodule and never re-exported; a first-class surface that can do everything the
  frontend does. [Naming conventions](naming_conventions.md#the-two-layers)
- **backend razor** (also: the razor). Put logic in the backend when it is nontrivial and easy to get wrong, judged
  from a user who works only on raw `.data`; trivial logic may stay in the frontend.
  [Backend-first](contributor/fitting_internals.md#backend-first-the-razor)
- **backend rule** (also: plain data, basic types; parameters are fields, behaviour is methods). Backend functions
  implement the math on plain data; backend classes only bind parameters and name roles, and every line of math in a
  method is also a standalone function. [The backend rule](contributor/parameters_not_closures.md#the-backend-rule-this-replaces)
- **body-local suffix** (`mu_WCa`, `C_aib`). A local variable's name ends in its axis layout: capitals for groups,
  lowercase for single axes, a leading `d` for the supercore or derivative axis.
  [Shape-notation legend](batching_and_stacking.md#shape-notation-legend-how-to-read-the-shapes-in-the-code)
- **capability-honest tails** (`ut3_full_sum`, `ut3_norm_orthogonalized`). A uniform twin keeps a different name when
  it supports less than the ragged one. [Semantic markers](naming_conventions.md#semantic-markers-load-bearing--you-can-trust-them)
- **checker** (`is_*` / `has_*`; backend `<family>_<subject>_<predicate>`; paired `_residual`). A non-enforcing
  predicate returning a verdict per stack element; safe mode raises at the call site that uses it.
  [Semantic markers](naming_conventions.md#semantic-markers-load-bearing--you-can-trust-them)
- **`contract`** (`backend.contractions.contract`; also: grouped einsum, grouped contraction, `len_<G>=`). An einsum
  whose uppercase letters each stand for a group of zero or more axes, solved from the operand shapes; `len_<G>=`
  pins a split the string cannot. No reshape ever happens. [Grouped contractions](grouped_contractions.md)
- **converters** (`from_t3`, `to_t3`, `from_ut3`, `to_ut3`, `from_t3frame`, `to_t3tangent`, …). On uniform classes `t3`
  names the ragged layer and `ut3` the same layer; on ragged classes `from_t3`/`to_t3` are same-layer.
  [Converters](naming_conventions.md#converters-encode-the-operand-type)
- **corewise** (marker; `COREWISE`, `corewise_dot`, `sum_stack_corewise`). Coordinate-space math on the core arrays,
  never Hilbert–Schmidt. `corewise_dot`/`corewise_norm` collapse every axis, stacks included.
  [Semantic markers](naming_conventions.md#semantic-markers-load-bearing--you-can-trust-them)
- **`.data`** (also: raw tuple). The raw tuple a frontend object wraps: `(tucker_cores, tt_cores)`; a frame's
  `(up, down, left, right)`; `(tucker_variations, tt_variations)`; uniform `(supercores…, shape, masks)`.
  [Naming conventions](naming_conventions.md#parameter-names-carry-representation-too)
- **dispatch** (`get_backend` → `xnp`, `xmap`, `xscan`; `tree_contains_jax`; also: inferred dispatch). numpy vs jax
  is inferred from the input arrays at the lowest level, and ragged vs uniform from the operand types; no `use_jax`
  on operations. [NumPy and JAX](user_guide.rst)
- **dispatch test** (`tests/test_dispatch.py`; dispatch rows). Numerical tests run on numpy only; this file jit-compiles
  each public op (a stray `np.*` on a tracer raises) and checks compile counts. [Testing strategy](contributor/testing_strategy.md)
- **doctest** (also: reproducible example). Docstring and doc-page examples are run in CI, so they must reproduce
  exactly; they teach, coverage belongs in `tests/`. [Doctest style](contributor/doctest_style.md)
- **eager jax**. Running jax operations outside `jit`: each new structure compiles op by op, slowly and with many
  memory mappings. With jax inputs, pass `use_jit=True`. [Jit or eager](aligned_padded_frame.md#jit-or-eager)
- **equality is explicit** (`allclose`, `corewise_equal`; `ExplicitEquality`). `==` raises on the runtime classes;
  `allclose` compares what objects represent, per stack element; `corewise_equal` compares stored bits.
  [Equality is explicit](numerical_contracts.md#equality-is-explicit)
- **exact output-mask assertion / garbage-padded inputs** (also: tautology trap). The two uniform test tools: assert
  output masks against an independent derivation, and fill padding with large finite garbage and require the same
  result. [The two tools](contributor/testing_strategy.md#the-two-tools-that-close-it)
- **family prefix** (`t3_`, `ut3_`, `tv_`, `utv_`, `fv_`, `ufv_`, `tt_`, `dense_`; brand names `t3svd`, `t3m`). The
  prefix naming a backend function's primary operand; `u` marks uniform-specific logic.
  [The family prefix grammar](naming_conventions.md#the-family-prefix-grammar)
- **frame-as-leaf**. A tangent's frame is a pytree leaf, not `aux_data`, so a new frame each Newton step does not
  recompile; the same-frame guard is numerical. [Decision history](contributor/batching_internals.md#decision-history)
- **frontend** (`import t3toolbox`). Frozen dataclasses holding only the cores, whose methods delegate to the backend
  and re-wrap; they validate structure on construction. [Naming conventions](naming_conventions.md#the-two-layers)
- **`HISTORICAL`** (`tests/test_contractions_interpreter.py`). The frozen table giving, for every library `contract`
  string, which `len_*` supplement it requires; a new string must be added by hand.
  [The evidence structure](contributor/contractions_internals.md#9-the-evidence-structure-and-the-one-maintenance-duty)
- **HOST tag** (`HOST bool, static`). A shape-comment tag: the array must be host numpy, never jax or traced, because
  it is static structure (the masks). [The decode table](reading_signatures.md#the-decode-table)
- **import-graph test** (`tests/test_import_graph.py`). Checks that the backend import graph is acyclic and that
  `backend/optimizers.py` imports only `common`, `regularization` and `corewise`.
  [The optimizer seam](contributor/padded_frame_internals.md#the-optimizer-seam-t3-free)
- **intended asymmetries**. Places where the uniform layer deliberately offers less (no `to_vector`, no `rtol`/`atol`
  truncation, …); not TODOs. [Intended asymmetries](naming_conventions.md#intended-asymmetries-uniform-vs-ragged)
- **jax-absent policy** (`common.jax_or_warn`). A request for jax (`use_jax=True`, `to_jax`, `use_jit=True`) without
  jax installed runs on numpy and warns; it never raises. [jit composes](fitting_and_optimization.md#45-jit-composes-for-free-the-draw-stays-outside-the-compiled-kernel)
- **mask-once** (also: mask on entry; re-mask on exit). Zero the padding once on entry, then run mask-free, since
  `0 × finite garbage = 0`; re-mask after a step that writes into padding.
  [The mask-once mechanism](contributor/uniform_polymorphism.md#the-mask-once-mechanism-and-why-it-works)
- **mask-representation contract** (`prefix_mask`, `ValueHashedMasks`, `require_concrete_masks`). Every uniform mask is
  a concrete host array, value-hashed, with prefix as its canonical form. [Cataloged exceptions](naming_conventions.md#cataloged-exceptions-deliberate)
- **named contraction functions** (`WCa_Caib_WCi_to_WCb`, retired). The ~104 functions whose names spelled their
  subscripts, replaced in 2026.1.0 by `contract`. [Where it came from](grouped_contractions.md#5-where-it-came-from-motivation-in-brief)
- **numerical contracts**. The op-by-op statement of what each operation requires and guarantees, and what safe mode
  checks. [Numerical contracts](numerical_contracts.md)
- **options are not parameters**. A value that changes what is computed is a hashed field of a geometry or kind; one
  that changes only how a solver proceeds stays a keyword argument. [Options are not parameters](contributor/parameters_not_closures.md#options-are-not-parameters)
- **oracle sweeps** (`tests/test_oracle_sweep.py`, `tests/oracle_sweeps/`; `T3TOOLBOX_SLOW_TESTS=1`). Dense
  ground-truth checks of fitting and Hessian quantities across kinds, geometries and layers, in an always-on and a
  slow tier. [Testing strategy](contributor/testing_strategy.md)
- **parameters are fields, not closures** (`ValueHashedFields`). Geometries, kinds and local models are frozen
  dataclasses whose parameters are fields, so they hash and compare by value as jit cache keys.
  [Parameters are fields](contributor/parameters_not_closures.md)
- **phantom rank**. A too-permissive mask that claims rank where the supercore is zero; dense comparisons pass, so only
  exact output-mask assertions catch it. [The blind spot](contributor/testing_strategy.md#the-blind-spot-too-permissive-masks-phantom-rank)
- **polymorphic op** (also: the ragged name is the polymorphic name). A backend function that takes ragged or uniform
  operands under its unprefixed ragged name; ops returning a T3-like object get a `ut3_`/`utv_` twin instead.
  [Polymorphism](naming_conventions.md#polymorphism-the-ragged-name-is-the-polymorphic-name)
- **precondition / caveat**. A precondition is needed for a correct result and is checked in safe mode; a caveat only
  governs what a result means and is never checked. [The classification rule](numerical_contracts.md#the-classification-rule)
- **safe mode / unsafe mode** (`t3toolbox.safe()`, `t3toolbox.unsafe()`, `SafetyTolerances`). An ambient setting;
  safe (the default) checks numerical preconditions and raises, unsafe and any jax trace skip them. The numbers are
  identical. [Numerical contracts](numerical_contracts.md)
- **scan-body principles**. Prefer module-level bodies for `xscan`/`xmap`/`xwhile`, deriving what they need from their
  arguments (or a memoized specialization); a value that changes between calls rides in the loop state, or a cached
  program silently reuses the old one. [The principles](contributor/scan_body_principles.md#the-principles)
- **seam**. A boundary where two parts of the library hand data across in a fixed form: the ragged/uniform seam
  (`t3_to_ut3`, `ut3_to_t3`), the split `precompute` → `*_from_sweep` seam, the optimizer seam (geometry members).
  [The optimizer seam](contributor/padded_frame_internals.md#the-optimizer-seam-t3-free)
- **semantic markers** (`corewise`, `numerically_`, `_trs`, `_hessian`, `_tangent`, `_from_sweep`, `lift`). Name parts
  that separate same-looking near-twins with different math; `corewise` and `numerically_` are rename-invariant.
  [Semantic markers](naming_conventions.md#semantic-markers-load-bearing--you-can-trust-them)
- **shape comment** (`# len=d, elm_shape=…`, `shape=`). The trailing comment on each signature line, stating what
  Python types cannot: shapes, lengths, constraints. [Reading signatures](reading_signatures.md)
- **stack matrix**. The standard parametrization of uniform tests: no stack, `C`, `K`, `K + C`, forced padding,
  multi-axis stacks, varying rank across `C`. [The stack matrix](contributor/testing_strategy.md#the-stack-matrix)
- **structural vs numerical**. Structural problems (shapes, ranks, masks, options) always raise, in both modes and
  under jit; numerical preconditions are checked only in safe mode. [User guide](user_guide.rst)
- **T4S**. The paper Alger, Christierson, Chen & Ghattas (2026), "Tucker Tensor Train Taylor Series"
  (arXiv:2603.21141); citations use the arXiv numbering. [arXiv](https://arxiv.org/abs/2603.21141)
- **twin** (uniform twin, backend twin, frontend twin, `_trs` / test twin, dotted twin). The same operation at another
  layer or level, or a reference implementation of it. [Naming conventions](naming_conventions.md)
- **`use_jax`**. The flag on pure constructors with no array input (`randn`, `zeros`, `ones`, `load` and backend
  twins), choosing the output type; operations infer instead. [NumPy and JAX](user_guide.rst)
- **`use_jit`**. Opt in to jit on `mc_sgd`, `adam`, `newton_cg` (its inner CG and alignments) and the alignment
  loops: numpy inputs move to jax (float32 unless x64) and the result is jax-backed. Without jax it warns and runs eager.
  [jit composes](fitting_and_optimization.md#45-jit-composes-for-free-the-draw-stays-outside-the-compiled-kernel)
- **value-hashed** (`ValueHashedFields`, `ValueHashedMasks`; jit cache key). Hashing and equality by content, so a
  rebuilt but identical object is the same jit cache key and does not recompile.
  [Uniform jit recipe](uniform_backend_jit_recipe.md)

## Symbols

Uppercase `C`, `W`, `K` are reserved for the batch blocks in shape comments and contraction strings; lowercase
letters are single axes. The letters below also carry other meanings on some pages.

| symbol | meanings, and the pages that use each |
|---|---|
| `C` | the frame stack (batching, shape comments, every backend signature); the central core (`t3m_methods.md`); the binomial coefficient `C(t,r)` in `trs` (`naming_conventions.md`) |
| `N` | mode size `Ni` (user guide, ragged `shape`); the padded mode width (uniform pages and classes); the normal part `N` of a gradient (the exact-Hessian and padded-frame math notes); sample counts `N_X`, `N_P` (derivative examples) |
| `n` | Tucker rank `ni`; the padded Tucker width (uniform); the measurement count in `batch/n` (the `mc_sgd`/`adam` regularizer scaling) |
| `r` | TT rank `ri`; the padded TT width (uniform); the residual `r` (fitting pages) |
| `K` | the tangent stack (batching, naming conventions, shape comments); the derivative order in `fitting_and_optimization.md` §2.1, `backend/sampling_derivatives.py`'s module comment and the apply-derivative examples, where `order` is meant; the curvature operators `K_{Aᵀω²r}`, `K_{grad f}` (§4.12, `LocalModel` docstrings) |
| `W` | the probe stack (batching, shape comments); a `T3Weights` variable (`weighting.md`); the Khatri–Rao factor `Wi` (`t3m_methods.md`); the up-matricization `Wi` (shared-factor math note); an integer variable in a `chunking.md` example |
| `P` | the left frame core in the T4S paper's notation (`batching_and_stacking.md` §8, `entries_apply_probe.md` §8, `LocalModel.frame` as `(U, O, P, Q)`), where the user guide writes `L`; the derivative direction (`pp`); a completion `P⁺` (`aligned_padded_frame.md` movement, the padded-frame note); the tangent projector `P_X` / `P_F`; the core parameter space (`CorewiseGeometry`) |
| `L` | the left frame cores `Li` (user guide, `frame_variations.md`); the lift `L` (§4.12, curvature pages) |
| `H` | the TT variations `Hi` (frame pages); the Hessian `H` (fitting pages); the centers `Hi = Li Ti` (`curvature_internals.md`, CHANGELOG 2026.3.0); `H_R`, the regularizer's Hessian |
| `S` | the sampling operator `S(x)` (fitting pages); the hint `S` (padded-frame note); the frame factors `Si` (`sharing.md`, `curvature_internals.md`); `S` in the T/S transforms (`curvature_internals.md`) |
| `O`, `D` | the down frame core: `O` in the user guide, `frame_variations.md`, the math notes and the `update` letters; `D` in the `T3Frame` docstring; "outer" in the minimal-rank list |
| `κ` | the edge condition number `κi` and group `κ_g` (rank continuation, sharing); the display ratio `κ = pKp/(pHp − pKp)` (§4.12) |
| `ρ` | the regularizer `ρ(x)`; the trust-region ratio of actual to predicted change |
| `η`, `σ` | `η`: the central edge variable and the CG forcing term; `σ`: singular values, a tangent edge variable, and `lift_sigma_min` in the display |
| `Π` | the gauge projection on variations (`MANIFOLD.project`); `Π_{T(F)}`, the ambient projection onto a frame's tangent space (aligned-frame pages) |
| `X`, `x` | the T3 iterate; the sample location of a derivative measurement (derivative examples, "probe vectors X" in derivative-model docstrings) |
| `T` | the frame factors `Ti` (exact-Hessian note); the tangent space `T(F)` (padded-frame pages); the represented tensor (shared-factor note) |
| `d` | the number of modes; the leading supercore axis (uniform layer); a leading `d` in a local name's suffix is the supercore or derivative axis |
| `m` | the maximum derivative order in `symmetric_probe_derivatives.tex` (the code's `order`); the cone merit `m_c`; the Gauss-Newton model `m(p)` |
| `ε` | the lift damping (`curvature_damping`); machine epsilon in the `objective_floor` test; `adam`'s `eps` |
