R5-1    S conf t3toolbox/backend/entries.py:105 — entries_ambient_transpose silently returns zero factors for negative indices (siblings wrap)
R7-6    S conf t3toolbox/backend/fitting.py:257 — Backend probe kinds silently truncate a per-mode weight whose mode dimension exceeds d
R7-1    S conf t3toolbox/backend/geometry.py:71 — Regularized Problem.objective at a raw ragged-manifold point: characterized (backend-only; no optimizer affected)
H2-1    S conf t3toolbox/backend/optimizers.py:197 — Problem.objective/local_model silently discard data= when sample is omitted
R3-1    S conf t3toolbox/backend/ranks.py:549 — compute_orthogonal_representation_ranks wrong on non-minimal input; uniform frame masks built from it make UT3Frame.from_ut3 / UNIFORM_MANIF
R9-1    S conf t3toolbox/backend/sharing.py:477 — Shared companion built on a UT3Frame.to_t3frame() leaf gives a silently wrong tied projection
R1-1    S conf t3toolbox/backend/t3_linalg.py:337 — t3m(method='inplace_fused') ignores max_tt_ranks on the last bond when rd != 1
R2-1    S conf t3toolbox/backend/t3_linalg.py:555 — t3m_swap silently wrong when both inputs have equal boundary TT ranks != 1 (crashes when they differ)
R2-2    S conf t3toolbox/backend/t3_linalg.py:573 — t3m_swap silently ignores per-position max_tt_ranks given as a numpy array
H2-2    S conf t3toolbox/backend/t3_svd.py:301 — rank_adjustment_sweep('right_to_left') is documented as right-orthogonal but leaves the Tucker cores non-orthogonal
H6-1    S conf t3toolbox/backend/ufv_conversions.py:60 — UNIFORM_MANIFOLD.frame(x) returns a non-orthonormal frame when a TT core's right unfolding is rank-deficient
H4-2    S conf t3toolbox/backend/ufv_conversions.py:60 — Uniform orthogonal frame is NOT orthogonal for numerically rank-deficient trains (zero-padded resize warm starts, x + x); ragged twin is ort
H5-2    S conf t3toolbox/backend/ufv_conversions.py:107 — ut3_orthogonal_representations on a non-minimal-rank point: frame masks from a right-then-left rank recurrence do not match the left-then-ri
R9-3    S conf t3toolbox/backend/ufv_operations.py:195 — Negative axis in UT3Variations.sum_stack / sum_tangents hits the mode axis (silent when K == d)
R10-1   S conf t3toolbox/backend/ut3_linalg.py:204 — NaN jax.grad through uniform weighted norm/inner (default orthogonalized path) and plain ux.norm() on any padded train
R8-1    S conf t3toolbox/backend/ut3_operations.py:82 — ut3_squash_tails sums padded boundary-bond slots into the real slot; +, -, squash_tails, sum_stack are not garbage-robust and the corewise o
H5-1    S conf t3toolbox/backend/ut3_operations.py:82 — ut3_squash_tails sums the raw (unmasked) boundary bonds: squash_tails, +, -, sum_stack, UT3Frame.allclose/is_consistent read padding garbage
H6-4    S conf t3toolbox/manifold.py:357 — A structural tangent mismatch is only caught by the numerical same-frame guard; under unsafe()/jit it silently broadcasts
R2-6    S conf t3toolbox/tucker_tensor_train.py:2023 — Structural checks via assert: empty-message AssertionError normally, silent wrong result or obscure error under python -O (TuckerTensorTrain
O1-2    S conf t3toolbox/uniform_frame_variations_format.py:959 — ut3_orthogonal_representations / UT3Frame.from_ut3 / UNIFORM_MANIFOLD.frame return a non-orthogonal frame for non-minimal-rank UT3s
H4-9    S susp docs/weighting.md:79 — Stacked t3_orthogonal_representations rotates the t3svd Tucker factors, so T3FrameWeights.from_t3weights(T3Weights.from_t3svd(x)) is applied
H4-7    C conf t3toolbox/backend/common.py:405 — jax absent: randn(use_jax=True) is a bare NameError (beyond get_backend); to_jax()/load(use_jax=True) silently return numpy
H1-1    C conf t3toolbox/backend/common.py:608 — numpy-integer shape entries on a uniform point crash newton_cg(use_jit=True) with a deep TracerArrayConversionError
R7-3    C conf t3toolbox/backend/optimizer_display.py:215 — make_newton_display / newton_cg(verbose=True) with val_data but no val_sample: TypeError deep in the kind
H2-3    C conf t3toolbox/backend/optimizer_display.py:215 — val_sample without val_data is silently ignored; val_data without val_sample raises a bare TypeError
O2-1    C conf t3toolbox/backend/optimizers.py:483 — adam on a ragged MANIFOLD x0 with non-minimal ranks crashes at step 2 with a bare broadcast error
R5-2    C conf t3toolbox/backend/probing.py:834 — Python-float residual crashes tangent/corewise apply+entries transposes in compute_sigma_hat
R5-4    C conf t3toolbox/backend/probing.py:1542 — dense_probe crashes at d=1 with a non-empty probe stack W
R2-5    C conf t3toolbox/backend/stacking.py:261 — stacking.stack (TuckerTensorTrain.stack, T3Frame.stack, T3Tangent.stack_*) raises ConcretizationTypeError under jit
H4-4    C conf t3toolbox/backend/stacking.py:261 — stacking.stack (T3/T3Frame/T3Variations/T3Tangent/UT3 stack) is not jittable: xnp.arange used as static moveaxis source
R1-11   C conf t3toolbox/backend/t3_linalg.py:117 — sum_stack with an out-of-range stack axis raises a bare AssertionError
R1-2    C conf t3toolbox/backend/t3_linalg.py:536 — t3m(method='swap') crashes or mis-structures the result with non-1 boundary TT bonds
R8-2    C conf t3toolbox/backend/tt_operations.py:114 — d=1 breaks squash_tails, +, -, sum_stack, norm(), inner(), t3svd, rank_adjustment_sweep, left/right orthogonalize and is_left/right_orthogon
H5-3    C conf t3toolbox/backend/tt_operations.py:114 — d = 1 uniform: _tt_squash_tails_uniform duplicates the single core, crashing squash_tails, +, -, sum_stack, norm, inner, t3svd, rank_adjustm
H3-2    C conf t3toolbox/backend/tt_operations.py:180 — MANIFOLD.project_ambient(frame, T3 grad) and MANIFOLD.transport crash for d=1 with IndexError in tt_zipper_left_to_right (empty core tuple)
H4-3    C conf t3toolbox/backend/tt_operations.py:180 — d=1: MANIFOLD.project_ambient(frame, T3) and MANIFOLD.transport crash with IndexError in tt_zipper_left_to_right
R10-3   C conf t3toolbox/backend/tt_orthogonalization.py:62 — Uniform d=1 also crashes plain norm/inner and the weighted norm/inner (tt_left_orthogonalize), broader than the ledger's ut3svd_supercores e
R5-3    C conf t3toolbox/backend/tv_operations.py:402 — MANIFOLD.project_ambient(frame, t3) and transport crash at d=1 (IndexError in tt_zipper)
R4-2    C conf t3toolbox/backend/tv_operations.py:402 — d=1 crashes tv_project_t3_onto_tangent_space -> MANIFOLD.project_ambient(frame, T3), project_ambient(method='t3svd'), MANIFOLD.transport
O1-1    C conf t3toolbox/backend/tv_operations.py:402 — d=1: MANIFOLD.project_ambient(frame, T3) and MANIFOLD.transport raise IndexError
R9-4    C conf t3toolbox/backend/uniform_fitting.py:437 — uniform_least_squares_problem silently routes any non-'manifold' geometry string to corewise
R9-5    C conf t3toolbox/backend/uniform_fitting.py:440 — order=None for a derivative kind builds a Problem that fails deep on first use
H5-4    C conf t3toolbox/backend/ut3_svd.py:111 — Minimal-rank but slack-padded x0 passes uniform_minimal yet every uniform manifold optimizer crashes: ut3svd/utv_retract shrink the padded d
R3-4    C conf t3toolbox/backend/ut3_svd.py:280 — d=1 breaks every uniform SVD/frame/orthogonality call, not only uniform fits
O1-3    C conf t3toolbox/backend/utv_operations.py:446 — UNIFORM_MANIFOLD.transport / project_ambient crash for a K-stacked tangent or K-stacked UT3 gradient
R9-2    C conf t3toolbox/backend/utv_operations.py:574 — UniformManifoldGeometry.project_ambient/transport crash on a K-stacked tangent (ragged twins accept K)
H3-1    C conf t3toolbox/backend/utv_operations.py:574 — UNIFORM_MANIFOLD.transport / project_ambient crash for a tangent-stacked (K != ()) UT3Tangent; the ragged twin handles the same K+C-vs-C bro
H6-11   C conf t3toolbox/fitting.py:480 — GaussNewtonModel constructors do no structural validation of the sample/residual against the point; mismatches fail deep with obscure errors
R4-8    C conf t3toolbox/frame_variations_format.py:499 — T3Frame.validate Tucker-rank-mismatch message prints stack dimensions instead of the ranks
H6-5    C conf t3toolbox/frame_variations_format.py:499 — T3Frame.validate Tucker-rank mismatch message prints the wrong numbers for stacked frames
R4-1    C conf t3toolbox/frame_variations_format.py:1122 — check_fv_pair accepts a d-mismatched (frame, variations) pair; downstream ops return silent partial results or opaque errors
H6-2    C conf t3toolbox/manifold.py:547 — GAUGE precondition thresholds an absolute gauge residual at the relative tolerance (scale-dependent false failures and false passes)
H4-1    C conf t3toolbox/manifold.py:1316 — Safe-mode GAUGE check uses the relative tolerance as an absolute bound on an unnormalized residual: false failures for large tangents and fo
H6-8    C conf t3toolbox/safety.py:107 — set_default_safety(None, None) is accepted and then every precondition check crashes with a TypeError
H4-5    C conf t3toolbox/tucker_tensor_train.py:721 — TuckerTensorTrain.has_numerically_minimal_ranks() raises on a stacked train
R5-5    C conf t3toolbox/tucker_tensor_train.py:3380 — TuckerTensorTrain.entries wrong-length list index raises AttributeError, not the documented ValueError
R1-3    C conf t3toolbox/tucker_tensor_train.py:3380 — entries() with a wrong-length list index raises AttributeError instead of ValueError
R1-14   C conf t3toolbox/tucker_tensor_train.py:4264 — share(rtol=/atol=) on a stacked T3 fails deep in truncated_svd instead of at the frontend check
H1-2    C conf t3toolbox/uniform_tucker_tensor_train.py:161 — jax-array masks pass UT3Masks validation but cannot be a jit cache key (unhashable ArrayImpl / obscure pytree-metadata error)
R8-5    C conf t3toolbox/uniform_tucker_tensor_train.py:258 — + / inner with different padded N fail with raw numpy errors; ut3_add docstring claims the frontend enforces N
H7-6    D conf CLAUDE.md:455 — CLAUDE.md Current-state still names the removed GeometryOps protocol and calls SF-T3 'slices 8–13 pending review'
R6-3    D conf docs/chunking.md:61 — docs/chunking.md says NumPy falls back to the dense assembly regardless of chunk_size; on uniform NumPy arrays the chunk loop engages
R5-10   D conf docs/entries_apply_probe.md:220 — entries_apply_probe.md §7 names TuckerTensorTrain.{entries,apply}_transpose, which do not exist
R7-8    D conf docs/fitting_and_optimization.md:228 — docs §3 under-specifies the backend Geometry protocol (lists `norm`, omits four required members); structure block stale
H7-7    D conf docs/getting_started.rst:4 — getting_started.rst says 'five short examples'; the page has six sections
R4-14   D conf docs/numerical_contracts.md:46 — docs/numerical_contracts.md T3Tangent table omits weighted_inner (SF), weighted_norm and absorb_weights
H7-1    D conf docs/numerical_contracts.md:86 — numerical_contracts.md lists shared(...).frame(x) as a TIED precondition; the code ties silently
H6-3    D conf docs/numerical_contracts.md:86 — docs/numerical_contracts.md lists shared(...).frame(x) as a TIED precondition; the code ties an untied x silently
O1-4    D conf docs/numerical_contracts.md:120 — 'retract stays a valid first-order retraction on a non-minimal frame' is false at numerically rank-deficient frames (the normal output of x.
R1-9    D conf docs/rank_continuation.md:179 — docs/rank_continuation.md uniform-layer loop calls resize with the wrong arity
R1-10   D conf docs/t3m_methods.md:56 — docs/t3m_methods.md says 'the boundary bonds stay 1' under max_tt_ranks; false for inplace_fused and swap
R5-9    D conf docs/transposes.md:37 — transposes.md: corewise transpose is the pullback J^T A, not a projection of the ambient back-projection
H7-3    D conf docs/user_guide.rst:37 — user_guide.rst Tucker-rank tuple typo (n0, r1, ..., n(d-1))
H7-4    D conf docs/user_guide.rst:139 — user_guide.rst manifold section lists frame/variation families with d+1 cores (U0..Ud, L0..Ld, V0..Vd, H0..Hd) and the field name tucker_cor
H7-2    D conf docs/user_guide.rst:388 — user_guide.rst cites 'Section 5 of our paper' for the probing algorithms; local t4s.pdf has them in §4/§6
R10-2   D conf docs/weighting.md:46 — docs/weighting.md `+` <-> `concatenate` duality does not hold for the uniform frontend `+` (it squashes tails); `UT3Weights.concatenate` is 
R5-6    D conf t3toolbox/backend/apply.py:182 — tv_apply_transpose / T3Tangent.apply_transpose docstrings describe the retired scatter implementation
R4-10   D conf t3toolbox/backend/fv_conversions.py:52 — fv_conversions.py signature comments label the TT-core slots down_tucker_cores / right_tucker_cores
R3-3    D conf t3toolbox/backend/linalg.py:81 — Documented truncation rule 'sigma < max(atol, rtol*sigma1)' / parsimony bound 'r_hat <= #{sigma >= tau}' contradicts the code's tail-Frobeni
R2-3    D conf t3toolbox/backend/linalg.py:81 — linalg.truncated_svd (and all directional/pair wrappers) document a per-singular-value rtol/atol rule; code implements a tail-Frobenius rule
R2-10   D conf t3toolbox/backend/linalg.py:349 — Copy-paste errors in linalg.py / t3_orthogonalization.py / t3_operations.py / stacking.py signature comments and docstrings
R7-2    D conf t3toolbox/backend/optimizers.py:237 — `batch` is NOT ignored when `draw` is given: regularizer λ scales by the nominal batch, not the draw's size
R7-10   D conf t3toolbox/backend/optimizers.py:385 — gradient_descent has no use_jit; backend docstring cites an unshipped plan slice, frontend never says eager-only
O2-2    D conf t3toolbox/backend/optimizers.py:455 — adam on MANIFOLD is gauge-dependent: numpy-eager, jax-eager, jit, ragged and uniform give different iterates
R5-8    D conf t3toolbox/backend/probing.py:220 — compute_mu shape comment says left_tt_cores len=d-1; all callers pass d, and d-1 silently truncates
R6-4    D conf t3toolbox/backend/sampling_derivatives.py:49 — Stale provenance text: the 'project note' write-up now exists (docs/symmetric_probe_derivatives.tex); tex cites a moved file
R6-5    D conf t3toolbox/backend/sampling_derivatives.py:1640 — compute_dxi_tilde_jets docstring says output is at the order<=1 leg; it returns the full (order+1,) order axis
R6-2    D conf t3toolbox/backend/sampling_derivatives.py:1922 — Summed chunk path is memory-inert when W // chunk_size == 2 (incl. the default chunk_size=100 for W in [200,299])
R6-1    D conf t3toolbox/backend/sampling_derivatives.py:1967 — Kept (sum_over_probes=False) chunk path does not bound memory; docstring/doc page claim a sequential lax.map does
R2-4    D conf t3toolbox/backend/t3_linalg.py:232 — t3_norm(use_orthogonalization=False) still orthogonalizes (twice); frontend norm docstring promises zippering without orthogonalization
R3-2    D conf t3toolbox/backend/t3_svd.py:306 — rank_adjustment_sweep 'compose both directions for guaranteed minimal ranks' is false when a Tucker rank exceeds its mode size (n_i > N_i), 
R3-5    D conf t3toolbox/backend/t3_svd.py:451 — dense_tucker_svd / dense_ttsvd docstrings describe a nonexistent xnp parameter and 'See Also' nonexistent function names
H3-9    D conf t3toolbox/backend/tv_operations.py:498 — backend/tv_operations.py documents the stacks with the retired letters V (tangent stack) and G (frame stack), overloading the Tucker-variati
R8-8    D conf t3toolbox/backend/ut3_linalg.py:51 — ut3_add docstring says shape is combined 'via OR'; it is passed through unchanged
R8-7    D conf t3toolbox/backend/ut3_orthogonalization.py:137 — up_orthogonalize_tt_supercores docstring cites a nonexistent module backend/orthogonal_representations.py
H7-5    D conf t3toolbox/backend/ut3_orthogonalization.py:137 — Rendered docstring cites a nonexistent module backend/orthogonal_representations.py
R8-3    D conf t3toolbox/backend/ut3_sampling.py:101 — Corewise transposes are documented as 'clean-padded' but the gradient is nonzero in the padded boundary-bond slots
O2-3    D conf t3toolbox/fitting.py:334 — GaussNewtonModel.jacobian return comment and residual field comment omit the uniform PACKED layout
R7-12   D conf t3toolbox/fitting.py:576 — apply_derivatives_model signature annotates ragged-only while the body dispatches on a uniform x
R4-6    D conf t3toolbox/frame_variations_format.py:40 — T3Frame eq=False comment says 'so it can be jax aux_data' (it is a leaf); batching doc says T3Tangent is eq=False/identity-hashed (it is eq=
H1-4    D conf t3toolbox/frame_variations_format.py:40 — Stale 'T3Frame as jax aux_data' comments: T3Frame is a pytree leaf everywhere
R4-4    D conf t3toolbox/frame_variations_format.py:734 — tt_variations shape comment says (rLi, nUi, rRi); the right bond is rR(i+1) -- the class's own doctest shows it
R4-9    D conf t3toolbox/frame_variations_format.py:1142 — Frontend fv_to_t3 docstring: wrong exception type, wrong index range, stale parameter names
R4-5    D conf t3toolbox/manifold.py:114 — manifold.py docstrings still say same-frame means the same T3Frame object (identity); the check is numerical
H6-10   D conf t3toolbox/manifold.py:371 — T3Tangent.__add__/__sub__ docstrings still describe the retired identity guard ('same T3Frame object')
R4-3    D conf t3toolbox/manifold.py:556 — Paper equation/figure numbers cited in code and docs are off by one (equations) and two (figure) vs the local t4s.pdf
H1-3    D conf t3toolbox/manifold.py:1534 — 'all instances of a geometry are interchangeable' is false: a pytree-reconstructed ManifoldGeometry is rejected by the fitting factories, Sh
R4-7    D conf t3toolbox/safety.py:107 — safety.set_default_safety docstring promises a 'script-level default'; it is per-context (threads see the module default, scoped blocks disc
H6-9    D conf t3toolbox/safety.py:107 — set_default_safety's 'script-level default' is not inherited by threads
R7-4    D conf t3toolbox/shared_geometry.py:85 — Three places claim a safe-mode tied-factors check at SharedGeometry.frame; the code ties silently
R7-5    D conf t3toolbox/shared_geometry.py:173 — 'An untied initial guess is a non-event for the optimizers' — the backend geometries the optimizers use do not tie at frame, and the corewis
R1-8    D conf t3toolbox/tucker_tensor_train.py:101 — Assorted docstring notation errors in the class docstring and the svd/sum/randn method docs
R1-16   D conf t3toolbox/tucker_tensor_train.py:721 — has_numerically_minimal_ranks raises on a structurally-minimal stacked T3; docstring promises a bool
R1-6    D conf t3toolbox/tucker_tensor_train.py:1090 — reverse() docstring claims tt_ranks=(1, r(d-1), ..., r1, 1); code returns (rd, ..., r0)
R1-5    D conf t3toolbox/tucker_tensor_train.py:1857 — __add__ and inner raise NotImplementedError on stack-shape mismatch; docstrings say ValueError
R1-4    D conf t3toolbox/tucker_tensor_train.py:2023 — __mul__ with a wrong-shape ndarray uses a bare assert; docstring promises ValueError
H3-4    D conf t3toolbox/tucker_tensor_train.py:3294 — entries/apply docstrings call the stack order 'base-inner' and use idx_stack_shape+t3_stack_shape / vec_stack_shape+t3_stack_shape instead o
R5-7    D conf t3toolbox/tucker_tensor_train.py:3483 — TuckerTensorTrain.probe return shape comment/docstring omit the frame stack C (and use an undefined letter X)
R1-7    D conf t3toolbox/tucker_tensor_train.py:3483 — probe() return shape comment uses an undefined block letter X and drops the frame stack C
H3-3    D conf t3toolbox/tucker_tensor_train.py:3483 — TuckerTensorTrain.probe return shape-comment says elm_shape=X+W+(Ni,) (unknown letter X, frame-outer order); code returns W+C+(Ni,) and the 
R9-6    D conf t3toolbox/uniform_frame_variations_format.py:9 — Module docstring cites 'no save' as a deliberate asymmetry; the real UT3Variations asymmetries are undocumented
R9-7    D conf t3toolbox/uniform_manifold.py:254 — Stale 'forthcoming / lands later' docstrings, a phantom xnp parameter, and a dev/archive path
R8-4    D conf t3toolbox/uniform_tucker_tensor_train.py:264 — Uniform + squashes the boundary bonds but ragged + does not, so W_A.concatenate(W_B) pairs with A+B in ragged and raises in uniform
R9-8    D susp t3toolbox/backend/sharing.py:877 — 'Masking the frame first breaks the companion pairing' not reproduced; frame-padding garbage does corrupt it
H7-9    E conf CHANGELOG.md:59 — CHANGELOG describes _cg_solve's jitted signature without its use_jit argument
H7-8    E conf docs/api_reference.rst:35 — api_reference.rst omits several public __all__ names beyond the known geometry/corewise gap
H7-12   E conf docs/sharing.md:22 — 'the paper' in sharing.md refers to the SF-ETT paper, not T4S, without saying so; shared_geometry docstring omits the uniform bases
R2-11   E conf t3toolbox/backend/common.py:311 — jax-absent: NameError surfaces through the constructors and tree_to_jax, and to_jax silently returns numpy
R2-8    E conf t3toolbox/backend/common.py:413 — tree_contains_jax / tree_to_jax / stacking.tree_depth recurse forever on a str leaf
H1-6    E conf t3toolbox/backend/common.py:509 — ValueHashedMasks eq/hash contract: bool vs int8 masks with the same content compare equal but hash differently
R2-7    E conf t3toolbox/backend/common.py:542 — ValueHashedFields/ValueHashedMasks hash-eq edge cases: jax-array field unhashable, dtype-only mask difference breaks hash/eq contract, mask-
H2-5    E conf t3toolbox/backend/fitting.py:266 — Backend omega normalizer never validates the mode dimension: over-long weight silently truncated, short weight IndexError
R4-13   E conf t3toolbox/backend/fv_conversions.py:176 — fv_conversions.py: inversely-named local aliases and an ignored is_uniform flag (behaviour correct)
H3-5    E conf t3toolbox/backend/geometry.py:216 — Ragged backend ManifoldGeometryOps.inner / CorewiseGeometryOps.inner / point_norm_sq collapse the frame stack C to a scalar while the unifor
H1-5    E conf t3toolbox/backend/geometry.py:339 — ValueHashedFields caches its key; uniform geometries/kinds alias the point's writeable mask arrays, so in-place mutation after first hash go
R3-6    E conf t3toolbox/backend/linalg.py:207 — truncated_svd lets min_rank silently override max_rank
R7-14   E conf t3toolbox/backend/optimizers.py:159 — LocalModel.retract return annotation says Tangent for a returned point
R7-9    E conf t3toolbox/backend/optimizers.py:452 — mc_sgd stats['losses'] are the EMA-smoothed check values, undocumented
R7-7    E conf t3toolbox/backend/optimizers.py:691 — Armijo exhaustion accepts the last trial silently; recorded alpha is one halving past the step taken
R5-11   E conf t3toolbox/backend/probing.py:258 — Minor: dead conditionals, duplicate line, notation drift in the signature-style reference, stale K letter, undocumented jax clamp
R3-7    E conf t3toolbox/backend/ranks.py:135 — np.minimum in ranks.py: eager jax fine, two of three functions fail under jit, no traced caller in-library (not worse than known)
R6-7    E conf t3toolbox/backend/sampling_derivatives.py:141 — 'K' denotes both the derivative order and the tangent stack within the same docstrings
R6-6    E conf t3toolbox/backend/sampling_derivatives.py:814 — Three dead conditionals: 'tt_reverse if <cond> else tt_reverse'
R6-8    E conf t3toolbox/backend/sampling_derivatives.py:2197 — Docstrings claim jax.linear_transpose verification that no test performs; tangent-derivative tests use only one palindromic, nD==nU structur
R10-5   E conf t3toolbox/backend/stacking.py:452 — T3Weights/T3FrameWeights.stack(unstack()) fails under jit (inherited from stacking.basic_ragged_stack; TuckerTensorTrain too)
R2-12   E conf t3toolbox/backend/stacking.py:513 — stacking helpers on malformed trees: tree_zip truncates, stack ignores extra leaves, unstack negative axes use the first leaf's ndim, empty 
R2-13   E conf t3toolbox/backend/t3_linalg.py:179 — Minor backend ergonomics: t3_inner_product list/tuple family concat, t3_sum with ndarray axis, truncated_svd min_rank/max_rank corners
R10-8   E conf t3toolbox/backend/t3_operations.py:328 — HANDOFF backlog item on T3Weights len=d/len=d+1 shape comments is stale for the class; only the backend parameter comments still omit the le
R2-9    E conf t3toolbox/backend/t3_orthogonalization.py:89 — t3_right_orthogonalize and the single-core SVD / relative-to orthogonalization steps are correct but have no test
R4-12   E conf t3toolbox/backend/tv_operations.py:497 — tv_operations.py uses the retired stack letters G (frame) / V (tangent) and 'base-inner'; the K/C code is correct
H2-9    E conf t3toolbox/backend/uniform_fitting.py:94 — The six Uniform*Kind.forward/transpose accept but never read their sample and frame_data arguments
R9-10   E conf t3toolbox/backend/ut3_sampling.py:236 — chunk_size absent from BOTH frontends' probe_corewise_derivatives_transpose; results identical
R8-10   E conf t3toolbox/backend/ut3_svd.py:233 — ut3svd_supercores is public, has no caller and no test; verified correct
R9-11   E conf t3toolbox/backend/utv_operations.py:539 — utv_project_ut3_onto_tangent_space lacks shared_data=; the frontend shared project_ambient/transport do tie
H7-11   E conf t3toolbox/backend/utv_sampling.py:15 — More stale '3b' build-slice docstrings: utv_sampling says the transpose 'lands in 3b-6c' although it is in the same module
H2-4    E conf t3toolbox/backend/utv_sampling.py:194 — sum_over_probes defaults False on the six ragged tv_*_transpose_from_sweep but True on their six utv_* twins
H4-6    E conf t3toolbox/corewise.py:62 — 29 assert-guarded structural checks vanish under python -O: corewise ops silently truncate, T3 * wrong-shape array silently broadcasts
R10-7   E conf t3toolbox/frame_variations_format.py:30 — Export asymmetry: check_ufw_pair is in __all__, check_fw_pair (named as the public guard in two docs) is not
H6-6    E conf t3toolbox/frame_variations_format.py:514 — Retired class names and a wrong class name in validation messages/docstrings (T3Base, T3Variation, T3Frame in T3Variations)
R4-11   E conf t3toolbox/frame_variations_format.py:1125 — Stale class names and nonexistent See-Also targets in frame_variations_format.py messages/docstrings
H3-8    E conf t3toolbox/manifold.py:1479 — COREWISE.retract accepts a MANIFOLD-frame tangent unchecked: raw broadcast error on a slack frame (nU != nD), silently wrong-typed arithmeti
R7-11   E conf t3toolbox/optimizers.py:135 — _resolve_chunk_size pulls device arrays to host just to read shapes, and duplicates the magic 100
O2-4    E conf t3toolbox/optimizers.py:212 — gradient_descent has no use_jit and its docstring does not say so
H6-12   E conf t3toolbox/safety.py:137 — safety._inside_jax_trace: the primary detector (jax.core.trace_state_clean) does not exist in jax 0.10.2; docstring overstates what the prob
H5-5    E conf t3toolbox/safety.py:227 — safety.frames_equal compares raw frame supercores including padding, so real-identical frames with different padding are 'different tangent 
H2-8    E conf t3toolbox/shared_geometry.py:146 — Frontend SharedGeometry identity keys on raw labels while the backend geometry keys on the canonical partition
R9-9    E conf t3toolbox/shared_geometry.py:313 — Wrong-layer frame to a ragged geometry gives an obscure unpack error
R1-13   E conf t3toolbox/tucker_tensor_train.py:56 — TuckerTensorTrain and T3Weights are eq=True dataclasses over array fields: == raises, unhashable
R1-17   E conf t3toolbox/tucker_tensor_train.py:1249 — TuckerTensorTrain.stack(x.unstack()) does not trace under jax.jit
R1-12   E conf t3toolbox/tucker_tensor_train.py:1934 — TuckerTensorTrain has no __rmul__/__radd__/__truediv__; UniformTuckerTensorTrain and T3Tangent do support scalar-left multiplication
H4-8    E conf t3toolbox/tucker_tensor_train.py:1934 — TuckerTensorTrain has no __rmul__: 2.0 * x raises while x * 2.0 works; the uniform and tangent classes define it
H3-6    E conf t3toolbox/tucker_tensor_train.py:2023 — Obscure errors on rejected stack mixes: bare AssertionError for T3 * dense with mismatched stacks; numpy inhomogeneous-shape error from stac
R1-15   E conf t3toolbox/tucker_tensor_train.py:4793 — t3_absorb_weights with shape-inconsistent weights fails inside einsum rather than with a structural error
R10-6   E conf t3toolbox/tucker_tensor_train.py:4793 — Ragged tensor-weight ops silently broadcast a mis-stacked T3Weights (uniform rejects); t3_weighted_inner then fails with a bare AssertionErr
R10-9   E conf t3toolbox/tucker_tensor_train.py:4800 — Minor signature asymmetries across the weighted twins (use_orthogonalization, return annotations, Optional)
H2-6    E conf t3toolbox/tucker_tensor_train.py:4800 — Ragged frontend t3_weighted_norm/t3_weighted_inner lack the use_orthogonalization kwarg their backend and uniform twins have
H7-10   E conf t3toolbox/uniform_frame_variations_format.py:1049 — Code comment cites dev/uniform_fix_plan.md, which lives in dev/archive/
H3-7    E conf t3toolbox/uniform_tucker_tensor_train.py:246 — UniformTuckerTensorTrain * UniformTuckerTensorTrain raises an obscure numpy dimension error instead of a TypeError (scalar-only __mul__)
H2-7    E conf t3toolbox/uniform_tucker_tensor_train.py:269 — UniformTuckerTensorTrain.sum_stack() has no axis argument (ragged sum_stack(axis=None) does)
H6-7    E conf t3toolbox/uniform_tucker_tensor_train.py:615 — dev/ paths in shipped messages and docstrings (known class; full enumeration)
R8-9    E conf t3toolbox/uniform_tucker_tensor_train.py:656 — minimal_ranks returns jax arrays on a jax-backed UT3, against the host-numpy rule for rank metadata
R8-6    E conf t3toolbox/uniform_tucker_tensor_train.py:775 — UniformTuckerTensorTrain.stack of leaves with different padded sizes dies inside stacking.stack with an obscure numpy error; the need for a 
R10-4   E conf t3toolbox/uniform_tucker_tensor_train.py:1158 — from_ut3svd shrinks the pad, so the documented GK metric route is rejected against any padded-above train's frame; only a ragged detour work
R4-15   E conf tests/test_manifold.py:738 — Untested-but-correct surface in scope, and the d=1 gap in the project_ambient/transport tests
O1-5    E conf tests/test_manifold.py:886 — tests/test_manifold.py test_riemannian_gradient compares an expression with itself (vacuous)
R7-13   E susp t3toolbox/shared_geometry.py:390 — SharedGeometry pytree unflatten reconstructs the base class, dropping a subclass
