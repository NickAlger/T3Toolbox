# Glossary candidates A: user-facing docs

Scout A of three for the 2026.5.0 glossary page. Partition: every top-level `docs/*.md` and
`docs/*.rst` page (not `docs/contributor/`), `README.md`, `CHANGELOG.md` (terminology only) and
`docs/release_notes.md`. All pages were read in full; the CHANGELOG was read for vocabulary.

Columns: **Term (aliases)**, **Where defined / best explained** (file § heading or line),
**Suggested definition** (doc wording quoted where the doc has a definition), **Link to** (the page a
glossary entry should point at), **Related**. Terms marked † are used in the partition without any
definition. Inconsistent usage is flagged inline and collected in the notes at the end.

233 terms in 11 sections.

---

## 1. The format

| Term (aliases) | Where defined / best explained | Suggested definition | Link to | Related |
|---|---|---|---|---|
| Tucker tensor train (T3; extended tensor train, ETT) | `user_guide.rst` "Tucker tensor trains"; `README.md` l.6–27 | "The composition of a Tucker decomposition with a tensor train decomposition of the central core." Tucker factors `Bᵢ` (shape `(nᵢ, Nᵢ)`) attached to a chain of TT cores `Gᵢ` (shape `(rᵢ, nᵢ, rᵢ₊₁)`). | `user_guide.rst` | TT core, Tucker core, represented tensor |
| represented (dense) tensor | `user_guide.rst` "Tucker tensor trains represent dense tensors" | The `N₀×…×N_{d−1}` array obtained by contracting all cores. Unless stated otherwise every operation is defined on this tensor, which is never formed. | `user_guide.rst` | corewise, gauge freedom |
| mode; order `d` | `entries_apply_probe.md` "Notation" | A mode is one tensor index; `d` (the tensor's order) is the number of modes. **Overload:** "order" also means derivative order (`order`, the jet axis). | `entries_apply_probe.md` | mode size, derivative order |
| mode size (`Nᵢ`; ambient dimension, physical dimension) | `user_guide.rst` "Tucker tensor trains" | The length of mode `i` of the represented tensor. `shape = (N₀,…,N_{d−1})`. | `user_guide.rst` | shape_mask, padded `N` |
| Tucker core (`Bᵢ`, `Uᵢ`; Tucker factor, Tucker basis) | `user_guide.rst` "Tucker tensor trains"; `sharing.md` | The `(nᵢ, Nᵢ)` matrix that maps mode `i`'s Tucker rank leg to its physical index. **Inconsistent:** "Tucker core" (README, user guide) and "Tucker factor" (sharing, weighting, t3m) name the same object, while "central Tucker core" (user guide) names something else. | `user_guide.rst` | central core, up core |
| TT core (`Gᵢ`) | `user_guide.rst` "Tucker tensor trains" | The 3-tensor `(rᵢ, nᵢ, rᵢ₊₁)` of the tensor-train chain that represents the Tucker core tensor. | `user_guide.rst` | TT rank, supercore |
| central (Tucker) core (`C` in `t3m_methods.md`) | `user_guide.rst` l.7–9; `t3m_methods.md` "What there is to truncate" | The `n₀×…×n_{d−1}` core tensor of the Tucker decomposition, stored as a tensor train. Not the same as a "Tucker core". | `user_guide.rst` | Tucker core |
| Tucker rank (`nᵢ`) | `user_guide.rst` "Tucker tensor trains" | The size of the edge between Tucker core `Bᵢ` and TT core `Gᵢ`; `tucker_ranks` has length `d`. | `user_guide.rst` | minimal ranks |
| TT rank (`rᵢ`; TT bond, bond rank, `rL`/`rR`) | `user_guide.rst`; `t3svd_minimal_ranks.md` (`rL_i = r_i`, `rR_i = r_{i+1}`) | The size of the edge between adjacent TT cores; `tt_ranks` has length `d+1` with `r₀`, `r_d` the boundary bonds. | `user_guide.rst` | boundary bonds |
| boundary bonds (tails; `r₀`, `r_d`) | `user_guide.rst` l.40–42; `t3m_methods.md` "Boundary bonds" | The first and last TT ranks. Usually 1; when not, the "1"s in the diagram are vectors of ones. | `t3m_methods.md` | squash_tails |
| squash_tails | `t3m_methods.md` "Boundary bonds" | Canonicalize a train's boundary bonds to 1 by summing them into the end cores (represented tensor unchanged). | `t3m_methods.md` | boundary bonds |
| corewise (marker) | `user_guide.rst` l.85–87; `naming_conventions.md` "Semantic markers" | "Coordinate-space math on the core arrays, never Hilbert–Schmidt / tensor space." Operations named `corewise` treat the cores as raw coordinates. | `naming_conventions.md` | COREWISE, Hilbert–Schmidt |
| Hilbert–Schmidt (HS) inner product / norm † | used throughout (`numerical_contracts.md`, `frame_variations.md`) | The Frobenius inner product / norm of the represented dense tensors, `⟨X, Y⟩ = Σ X[i]·Y[i]`. Never defined in the docs. | `user_guide.rst` (needs a sentence) | corewise, MANIFOLD |
| minimal ranks | `user_guide.rst` "Minimal rank conditions"; `t3svd_minimal_ranks.md` "What minimal ranks means" | Ranks with no redundancy: `nᵢ ≤ Nᵢ`, `nᵢ ≤ rᵢ·rᵢ₊₁`, `rᵢ ≤ nᵢ·rᵢ₊₁`, `rᵢ₊₁ ≤ rᵢ·nᵢ`; equal to the matricization ranks of a generic tensor with that structure. | `t3svd_minimal_ranks.md` | structural vs numerical minimal |
| structural vs numerical minimal ranks | `numerical_contracts.md` "Minimal ranks: the settled verdict"; `naming_conventions.md` (`numerically_`) | `has_minimal_ranks`: the ranks equal the structural minimum (integer arithmetic). `has_numerically_minimal_ranks`: no stored rank is numerically redundant (SVD-grade). | `numerical_contracts.md` | checker |
| maximal rank (flat slot) | `fitting_and_optimization.md` §4.12 "Where the curvature vanishes" | A slot whose rank equals its ceiling (a Tucker rank equal to its mode size, a TT rank equal to the product of its neighbours). The representable set is flat there. | `fitting_and_optimization.md` §4.12 | curvature term |
| matricization / unfolding | `user_guide.rst` "Minimal rank conditions"; `t3svd_minimal_ranks.md` | The matrix obtained by splitting the modes into two groups at one edge of the network; its rank is that edge's minimal rank. | `t3svd_minimal_ranks.md` | minimal ranks |
| T3-SVD (`t3svd`; implicit T3-SVD, Algorithm 10) | `t3svd_minimal_ranks.md` intro; `README.md` | A left-to-right sweep of SVDs that truncates (or losslessly reduces) a T3; returns a left-orthogonal train and the Tucker and TT singular values. "Implicit" is never explained. | `t3svd_minimal_ranks.md` | rank_adjustment_sweep, truncation spec |
| rank_adjustment_sweep | `t3svd_minimal_ranks.md` "How to minimize" | A lossless directional sweep that drops structurally redundant ranks. One sweep reaches minimal ranks only if the input is already orthogonal in the opposite direction. | `t3svd_minimal_ranks.md` | minimal ranks |
| truncation spec (`max_tucker_ranks`, `max_tt_ranks`, `rtol`, `atol`) | `t3m_methods.md` "The truncation spec" | Rank caps (int or per-position sequence) and per-step tolerances shared by `t3svd` and `t3m`. Tolerances need unstacked input; uniform truncation takes caps only. | `t3m_methods.md` | T3-SVD |
| left- / right- / down-orthogonal | `frame_variations.md` "Four rank stores" (table) | A core whose `(rᵢ·nᵢ)×rᵢ₊₁` unfolding has orthonormal columns (left), whose `rᵢ×(nᵢ·rᵢ₊₁)` unfolding has orthonormal rows (right), or whose `nᵢ×(rᵢ·rᵢ₊₁)` unfolding has orthonormal rows (down). | `frame_variations.md` | frame cores |
| sweep (left-to-right, right-to-left, central) † | `entries_apply_probe.md` §1, §8; `uniform_supercore_layout.md` | A pass over the cores in mode order that carries an accumulated contraction from one core to the next. Used everywhere, never defined as a term. | `entries_apply_probe.md` §8 | edge variables, zipper |
| zipper † | `uniform_equivalence_contract.md` l.144; `naming_conventions.md` (`tt_zipper_left_to_right`) | Undefined in user docs. Appears to mean the sweep that contracts two TT chains against each other edge by edge. | (contributor docs) | sweep |
| `t3m` (elementwise product) and its methods | `t3m_methods.md` | The elementwise (Hadamard) product with optional truncation, by `'form_then_round'` (form, then T3-SVD), `'inplace_fused'` (truncate during one fused sweep; default) or `'swap'` (swap-based chain merge, for `r ≫ d`). `x * y` is the exact product. | `t3m_methods.md` | oversample (t3m) |
| oversample (`t3m`, `k`) | `t3m_methods.md` "Oversampling" | For `method='swap'`: relax every truncation criterion by `k` during the merge, then run one T3-SVD at the exact targets. **Collides** with the alignment's `oversampling` (§8), an unrelated knob. | `t3m_methods.md` | oversampling (alignment) |
| joint truncation | `t3m_methods.md` "The quality guarantee" | Truncating Tucker ranks weighted by the central TT environment (as T3-SVD does), not by an unweighted per-mode SVD. | `t3m_methods.md` | T3-SVD |
| TTM † | `t3m_methods.md` l.16 | "the TTM generalization": never expanded. | `t3m_methods.md` | t3m |
| resize | `rank_continuation.md` "How to use it" | Zero-pad a train's cores up to larger ranks (or shapes); the represented tensor is unchanged. | `rank_continuation.md` | zero-padded warm start |
| CP factors (canonical factors; `from_canonical`) | `transposes.md` "Ambient" | A tuple of `d` factor matrices `stack_shape + (R, Nᵢ)` representing a rank-`R` CP tensor `Σ_k ⊗ᵢ f_i[k]`; `from_canonical` converts it to a T3. | `transposes.md` | ambient transpose |

## 2. The manifold, frames and tangents

| Term (aliases) | Where defined / best explained | Suggested definition | Link to | Related |
|---|---|---|---|---|
| fixed-rank (T3) manifold (`M_{n,r}`) | `user_guide.rst` "T3 manifold"; `uniform_ranks_and_varieties.md` | Under the minimal-rank conditions, the set of T3s with fixed ranks is an embedded submanifold of `ℝ^{N₀×…×N_{d−1}}`. | `user_guide.rst` | tangent, retraction |
| base point (point) | `frame_variations.md` "A tangent is a sum of d terms" | The T3 at which a tangent space is attached; the frame encodes it. **Overload:** "point" also means a sample point of the jets (`entries_apply_probe.md` §4, "the points `X` and directions `P`") and a downstream input sample ("10 points × 100 directions"). | `frame_variations.md` | frame |
| frame (`T3Frame`; orthogonal frame, orthonormal frame) | `frame_variations.md`; `getting_started.rst` "Tangents and retraction"; `naming_conventions.md` "Why frame, not basis" | The orthonormal cores `(up, down, left, right)` that encode the base point and give the tangent space its coordinates; computed once per base point. Called a frame, not a basis, because at a non-minimal point it is overcomplete. | `frame_variations.md` | variations, store |
| frame cores: up (`Uᵢ`), down (`Oᵢ`), left (`Lᵢ`/`Pᵢ`), right (`Rᵢ`/`Qᵢ`) | `frame_variations.md` table; `user_guide.rst` "T3 manifold"; `batching_and_stacking.md` §8 | Up = orthonormal Tucker factors; left/right = left-/right-orthogonal TT environments; down = the down-orthogonal core whose mode leg holds the Tucker complement. **Inconsistent symbols:** `L`/`R` (user guide) vs `P`/`Q` (paper map, probing tables); `Oᵢ` is called "down" and "outer" in different places (`batching_and_stacking.md` §8 says "not outer"). | `frame_variations.md` | variations |
| variations (`T3Variations`; tucker variation `Vᵢ`/`δUᵢ`, tt variation `Hᵢ`/`δGᵢ`) | `frame_variations.md`; `user_guide.rst` "T3 manifold" | The cores that define a tangent vector relative to the frame: one Tucker variation and one TT variation per mode. Under the gauge conditions they are orthonormal coordinates of the tangent space. **Overload:** `Hᵢ` is also the "center" of a frame in the lift (CHANGELOG 2026.3.0, `H_i = L_i Z_{i+1}`). | `frame_variations.md` | gauge conditions |
| tangent (tangent vector; `T3Tangent`) | `frame_variations.md` "A tangent is a sum of d terms"; `api_reference.rst` | One (frame, variations) pair, representing the sum of `d` tensor-train terms, each with one variation core in place of a frame core. | `frame_variations.md` | frame, variations |
| gauge conditions (gauged) | `frame_variations.md` "Gauge conditions make the variations coordinates"; `numerical_contracts.md` table | `Uᵢᵀ Vᵢ = 0` and each `Lᵢ` left-orthogonal against all but the last `Hᵢ` (paper eqs. (47)–(48)). On an orthonormal frame they make corewise linear algebra on variations equal HS linear algebra on tangents. | `frame_variations.md` | gauge projector |
| gauge (as freedom) | `uniform_equivalence_contract.md` "Gauge-carrying operations"; `weighting.md` "The gauge matters" | The non-uniqueness of a representation: column signs, rotations within equal-σ clusters, null-space completions. **Overload:** distinct from the gauge *conditions* above; "the T3-SVD gauge" is a third use (a frame whose Tucker basis is the singular basis). | `uniform_equivalence_contract.md` | up to gauge |
| gauge projector (`Π`; `MANIFOLD.project`, orthogonal gauge projection) | `entries_apply_probe.md` §4, §8; `fitting_and_optimization.md` §1 | The projection of raw variations onto gauged variations; the Riemannian operators factor as `J = 𝒥∘Π`, `Jᵀ = Π∘𝒥ᵀ`. **Overload:** `Π_{T(F)}` / `P_F` in `aligned_padded_frame.md` is the ambient projection onto the tangent space, a different map. | `entries_apply_probe.md` §4 | Riemannian Jacobian |
| oblique projection (`project_oblique`) † | `weighting.md` l.104; `user_guide.rst` lit. [5] | Re-gauges ungauged variations without changing the tangent they represent. Used, never defined. | (needs a sentence in `frame_variations.md`) | gauge projector |
| ambient projection (`project_ambient`) † | `numerical_contracts.md` tables; `CHANGELOG.md` | Projects an ambient tensor (a T3 or dense array) onto the tangent space at a frame. Listed, never defined. | `numerical_contracts.md` | hint, alignment |
| retraction (`retract`) | `getting_started.rst` "Tangents and retraction"; `fitting_and_optimization.md` §2.4 | Maps a base point plus a tangent step back onto the manifold; on `MANIFOLD` by the "implicit truncated T3-SVD" of the doubled-rank embedding, on `COREWISE` by adding the step to the cores. | `fitting_and_optimization.md` §2.4 | doubled-rank embedding |
| vector transport (`transport`) † | `numerical_contracts.md` tables; `user_guide.rst` l.156 | Moves a tangent to another frame's tangent space. Listed, never defined. | `numerical_contracts.md` | tangent |
| doubled-rank embedding (`to_t3`, `include_shift`) | `frame_variations.md` l.45, "Where to look"; `entries_apply_probe.md` §8 | The exact representation of a tangent (optionally plus its base point) as a T3 of twice the frame's ranks. | `frame_variations.md` | retraction |
| orthogonal representation (`t3_orthogonal_representations`) | `getting_started.rst` "Tangents and retraction"; `user_guide.rst` "T3 manifold" | Factors a point into an orthonormal frame plus gauged variations. `t3svd_orthogonal_representations` does the same in the T3-SVD gauge and also returns the singular values. | `getting_started.rst` | frame |
| store (rank store; four rank stores: up, down, left, right) | `frame_variations.md` "Four rank stores"; `aligned_padded_frame.md` "The padded frame" | One family of frame cores and its rank tuple (`up_ranks`, `down_ranks`, `left_ranks`, `right_ranks`). Each core is orthonormal in its own unfolding, so each store's ranks are capped separately. | `frame_variations.md` | slack |
| slack (excess rank; "excess rank is inert") | `frame_variations.md` "Four rank stores"; `numerical_contracts.md` "Why excess rank is inert" | Rank requested beyond what the network supports is carried as differences between the four stores; the tangent space and HS identities are unchanged. | `frame_variations.md` | minimal ranks |
| frame-internal padding cores (`Ld`/`R0`; carriers `left[d-1]`, `right[0]`) | `frame_variations.md` "d cores, d+1 bonds"; `aligned_padded_frame.md` l.24 | The last left core and first right core, unused by the tangent terms; they hold the base point. **Inconsistent:** `frame_variations.md` calls them `Ld`/`R0` although its own indexing runs `L0…L(d−1)`; `aligned_padded_frame.md` calls them "carriers" `left[d-1]`, `right[0]`, the only use of that word. | `frame_variations.md` | base point |
| `manifold_dim` (tangent-space dimension) | `getting_started.rst` "Shared Tucker factors"; `sharing.md` "Dimension" | The dimension of the fixed-rank (or shared) manifold at given ranks; used to test whether a continuation level is underdetermined. | `rank_continuation.md` | rank continuation |
| geometry (`MANIFOLD`, `COREWISE`, uniform twins) | `fitting_and_optimization.md` §2.4, §4.1; `api_reference.rst` "Geometries" | The object that fixes where optimization happens: `MANIFOLD` = Hilbert–Schmidt metric with the gauge projector; `COREWISE` = Euclidean metric on raw cores, no projector. `UNIFORM_MANIFOLD` / `UNIFORM_COREWISE` are the uniform twins. **Overload:** `weighting.md` says "two objects, one per geometry", meaning tensor vs tangent. | `fitting_and_optimization.md` §2.4 | shared wrapper |
| Geometry protocol (geometry members) | `fitting_and_optimization.md` §3 "Geometry" | The members a backend geometry provides: `frame`, `base_point`, `stack_shape`, `project`, `retract`, `inner`, `point_norm_sq`, `point_tangent`, `precompute`, plus optional ones (`zero_point_frame`, `with_frame`, `require_frame_structure`, `aligned_padded_frame`) and four exact-mode ones. | `fitting_and_optimization.md` §3 | geometry aux |
| geometry aux (`precompute`, `geom_aux`, `geometry_aux`) | `fitting_and_optimization.md` §3; `release_notes.md` "Upgrading from 2026.0.0" | Per-frame data a geometry computes once per local model and hands to `project`/`retract`; `None` for the plain geometries, the shared companion for shared ones. | `fitting_and_optimization.md` §3 | companion |

## 3. Sampling, transposes and jets

| Term (aliases) | Where defined / best explained | Suggested definition | Link to | Related |
|---|---|---|---|---|
| sampling operations | `entries_apply_probe.md` "TL;DR"; `user_guide.rst` | The three ways to evaluate the represented tensor without forming it: `entries`, `apply`, `probe`. | `entries_apply_probe.md` | containment |
| `entries` | `entries_apply_probe.md` §1 | Read off `X[i₀,…,i_{d−1}]` at given multi-indices (point evaluation), computed by slicing Tucker fibers. | `entries_apply_probe.md` | index |
| `apply` | `entries_apply_probe.md` §1 | The multilinear form `X(w₀,…,w_{d−1})`: contract every mode with a vector, one scalar per sample. | `entries_apply_probe.md` | probe vectors |
| `probe` | `entries_apply_probe.md` §1; `user_guide.rst` diagram | Contract all modes but one, for each mode in turn: `d` vectors, the `i`-th being `X(w₀,…,·,…,w_{d−1})`. **Overload:** "probe" also means one sample row of `W` ("`|W|` = number of probes", `chunking.md`) and the vectors fed to `apply` ("unit-norm probe rows", README). | `entries_apply_probe.md` | probe stack |
| probe vectors (`ww`; test vectors, measurement vectors, probe rows) | `entries_apply_probe.md` "Notation"; `naming_conventions.md` "Parameter names" | A length-`d` list of arrays `W + (Nᵢ,)`, one vector per mode per sample, used by `apply` and `probe`. Four names in the docs for one thing. | `entries_apply_probe.md` | sample |
| `index` | `entries_apply_probe.md` §1 | An int array `(d,) + W` of multi-indices for `entries`. | `entries_apply_probe.md` | entries |
| perturbation directions (`pp`; directions `P`) | `entries_apply_probe.md` "Symmetric directional derivatives"; `fitting_and_optimization.md` §2.1 | The direction vectors of the jets, one list per sample, paired with `ww` or `index`. | `entries_apply_probe.md` | jet |
| fiber extraction | `entries_apply_probe.md` §2 | `probe` with one-hot vectors: the mode-`m` fiber through an index. Not a named API. | `entries_apply_probe.md` | probe |
| containment (`probe ⊃ apply ⊃ entries`) | `entries_apply_probe.md` §2 | `apply` is `probe` with the free index also contracted; `entries` is `apply` with one-hot vectors. | `entries_apply_probe.md` | sampling operations |
| measurement space | `entries_apply_probe.md` §5 | Where residuals live: one scalar per sample for `apply`/`entries`, `d` vectors per sample for `probe`, with the plain sum-of-products inner product. | `entries_apply_probe.md` | residual |
| edge variables (`ξ`, `μ`, `ν`, `η`; `σ`, `τ`, `δξ`, `δη`; frame edge variables) | `entries_apply_probe.md` §8; `fitting_and_optimization.md` §4.3 | The per-edge intermediates of the probing sweeps: `ξᵢ = Uᵢᵀwᵢ`, left partial products `μ`, right `ν`, central `η`; the tangent versions `σ`, `τ`, `δη`. Hat = base point, `δ` = perturbation, tilde = adjoint. **Overload:** `η` is also the CG forcing term, `σ` also singular values and `lift_sigma_min`. | `entries_apply_probe.md` §8 | frame sweep |
| frame sweep (`precompute`, `*_from_sweep` hooks) | `fitting_and_optimization.md` §4.3 | The `W`-scaling part of the Jacobian (the frame edge variables), which depends only on the frame and the sample vectors; computed once per frame and reused by every `J`/`Jᵀ`. | `fitting_and_optimization.md` §4.3 | local model |
| assembly (variation gradient assembly) | `chunking.md` "Where the memory goes" | The transpose stage that forms frame-pairwise products of edge-variable jets into variation gradients; its per-row cost is super-linear in the ranks. | `chunking.md` | chunk_size |
| Riemannian Jacobian (`𝒥`, bare Jacobian; `J`) | `entries_apply_probe.md` §4 "Bare 𝒥 / 𝒥ᵀ, no gauge projector" | `𝒥` is the tangent sampling operation applied to variations, with no projector; the full Riemannian operators are `J = 𝒥∘Π` and `Jᵀ = Π∘𝒥ᵀ`. | `entries_apply_probe.md` §4 | gauge projector |
| transposes: ambient / corewise / tangent | `transposes.md` "TL;DR" | Ambient: the literal adjoint on the full tensor space, returned as CP factors (static, frame-free). Corewise: the pullback of a residual through the core parametrization, a raw `(tucker_grads, tt_grads)` tuple. Tangent: the ambient back-projection projected onto the gauged tangent space, a `T3Tangent`. | `transposes.md` | sum_over_probes |
| back-projection | `transposes.md` "TL;DR" | The adjoint of a sampling operation applied to a residual, e.g. `c·(w₀⊗…⊗w_{d−1})` for `apply`. | `transposes.md` | ambient transpose |
| `sum_over_probes` | `batching_and_stacking.md` §11 | Transpose flag. `False` (default, primary) keeps `W` as an output stack, one result per probe; `True` sums over `W`, giving the Gauss-Newton gradient `Jᵀr`. | `batching_and_stacking.md` §11 | probe stack |
| jet (symmetric directional derivatives; `*_derivatives`) | `entries_apply_probe.md` "Symmetric directional derivatives"; `README.md` | `y⁽ᵗ⁾ = dᵗ/dsᵗ [op(X + sP)]|₀` for `t = 0..order`, stacked on a leading order axis; order 0 is the plain operation. Here `X` is a sample point, not the tensor. | `entries_apply_probe.md` | order axis |
| order axis (jet axis; derivative `order`; `t`) | `naming_conventions.md` "Index letters: stacks vs derivative order" | The leading axis of length `order + 1` on jet arrays. The maximum order is spelled `order` and has no capital letter. **Violated** in `fitting_and_optimization.md` §2.1, which writes the order as `K` (`jets` of shape `(K+1,)+W`). | `naming_conventions.md` | K |
| tensor degree † | `entries_apply_probe.md` l.180 ("orders ≥ tensor degree vanish") | Undefined. Presumably `d`, the degree of the polynomial `w ↦ X(w,…,w)`. | `entries_apply_probe.md` | jet |
| `trs` (binomial convolution tensor; `_trs` twins) | `naming_conventions.md` "Semantic markers" | `trs[t,r,s] = C(t,r)·[r+s=t]`, the derivation's binomial tensor. `_trs` names its dense reference twins, which live only in the tests. | `naming_conventions.md` | banded recurrence |
| banded recurrence / order scan | `naming_conventions.md` "_trs suffix"; `CHANGELOG.md` 2026.3.0 | How the shipped jet kernels combine orders without forming `trs`: an affine factor gives a banded two-term recurrence; a full convolution is a scan over orders. | `naming_conventions.md` | trs |
| two-hole sweep (two-hole) | `transposes.md` "Corewise" (l.122–129); `chunking.md` | A hand-written sweep with two cores replaced at a time; it computes the derivative of a corewise transpose along raw core directions, exactly and symmetrically. It is the exact Hessian's sampling term. | `transposes.md` | exact Hessian |

## 4. Batching and contractions

| Term (aliases) | Where defined / best explained | Suggested definition | Link to | Related |
|---|---|---|---|---|
| `stack_shape` (stacking, batching) | `batching_and_stacking.md` "Glossary" | "The leading batch axes shared by all of an object's cores; *is* the frame/core stack `C`." | `batching_and_stacking.md` | C |
| "stack" (three meanings) | `batching_and_stacking.md` §1 | (1) `stack_shape` axes on one object; (2) `backend/stacking.py`'s tree ↔ stacked-object conversion; (3) the uniform supercore. **A fourth sense:** "uniform stack" (§5, all cores share one `stack_shape`), which also collides with "uniform layer". | `batching_and_stacking.md` §1 | supercore |
| batch blocks (`C`, `W`, `K`) | `batching_and_stacking.md` §2; `reading_signatures.md` | Three independent batches on different arrays: `C` frame/core stack (every core), `W` probe stack (`ww` / `index` only), `K` tangent stack (variation cores only). | `batching_and_stacking.md` §2 | frame-inner |
| `C` (frame stack; core stack; frame/core stack; base-point stack) | `batching_and_stacking.md` "Glossary" | "A batch of T3s / base points, on every core (`= stack_shape`)." **Letter collision:** `C` is also the central core in `t3m_methods.md` and the binomial `C(t,r)`. | `batching_and_stacking.md` | stack_shape |
| `W` (probe stack; sample stack; vec/index stack) | `batching_and_stacking.md` "Glossary"; `entries_apply_probe.md` "Notation" | "A batch of probe-vector sets, on `ww` only" (or of multi-indices for `entries`). **Letter collision:** `W` is the `T3Weights` variable in `weighting.md`, the Khatri-Rao factor `Wᵢ` in `t3m_methods.md` and an integer in `chunking.md`. | `batching_and_stacking.md` | sum_over_probes |
| `K` (tangent stack) | `batching_and_stacking.md` "Glossary"; `naming_conventions.md` | "A batch of tangent vectors at one frame, on the variation cores only." Reserved by a 2026-08-24 ruling. **Collisions:** derivative order in `fitting_and_optimization.md` §2.1; the curvature operators `K_{Aᵀω²r}`, `K_{grad f}` in §4.12. | `batching_and_stacking.md` | order axis |
| frame-inner (ordering) | `batching_and_stacking.md` §3 | "`C` innermost, `W`/`K` outermost (`W + K + C`)", so a `C`-stacked frame broadcasts over its `K` tangents for free under right-aligned broadcasting. **Misused** in `uniform_supercore_layout.md` l.21 as a name for "stack leads the core axes". | `batching_and_stacking.md` §3 | stack-leading |
| heterogeneous stack | `batching_and_stacking.md` "Glossary", §5 | "One T3 whose cores have different-but-broadcastable stacks (frame `C`, variation `K+C`)." Allowed in the backend; frontend classes require one `stack_shape`. | `batching_and_stacking.md` §5 | validate |
| split recovered (`frame_stack_shape`, `tangent_stack_shape`) | `batching_and_stacking.md` §6 | The `K`/`C` split of a tangent is derived from the (frame, variations) pair, not stored: `C` is the trailing suffix of the variations' stack. | `batching_and_stacking.md` §6 | check_fv_pair |
| `stack_tangents` / `stack_frame` (two-axis stack/unstack) | `batching_and_stacking.md` §6 | `unstack_tangents`/`stack_tangents` peel the `K` stack (tangents sharing one frame); `unstack_frame`/`stack_frame` peel the `C` stack (tangents at distinct bases). | `batching_and_stacking.md` §6 | K, C |
| grouped einsum (`contract`; grouped-contraction dialect) | `grouped_contractions.md` §2 | An einsum whose uppercase letters each stand for a group of zero or more axes, sizes solved from the operand shapes; one ordinary einsum then runs with no reshape. | `grouped_contractions.md` | len_<G> |
| group (uppercase letter) | `grouped_contractions.md` §2 | In a `contract` subscript, a run of zero or more axes named by one uppercase letter; the interpreter gives the letters no meaning. | `grouped_contractions.md` | batch blocks |
| `len_<G>=` (supplement) | `grouped_contractions.md` "When the shapes cannot pin a split" | The number of axes in group `G`, required exactly when the subscripts alone cannot determine the split. | `grouped_contractions.md` | contract |
| named contraction functions (`WCa_Caib_WCi_to_WCb`) | `grouped_contractions.md` §5; `release_notes.md` "Upgrading from 2026.0.0" | The ~104 pre-2026.1.0 functions whose names spelled their subscripts; removed and replaced by `contract`. | `grouped_contractions.md` §5 | contract |
| shape comments (shape contract; `len=`, `elm_shape=`) | `reading_signatures.md` | The trailing `#` comments in signatures that state tuple lengths, element shapes and roles; "the type annotation the language cannot express." | `reading_signatures.md` | HOST bool, static |
| `HOST bool, static` (tag) | `reading_signatures.md` "The decode table" | The array must be host numpy, never jax or traced, because it is static structure (the uniform masks). | `reading_signatures.md` | edge masks |
| body-local suffix (`mu_WCa`) | `batching_and_stacking.md` "Shape-notation legend" | A local variable name ending in its axis layout: capitals for groups, lowercase for single axes, a leading `d` for the supercore/derivative axis. | `batching_and_stacking.md` | grouped einsum |
| sharding (shard over `W`; `shard_map`) | `chunking.md` "Sharding over W"; `grouped_contractions.md` §4 | Splitting an axis across devices. Any group axis of `contract` shards freely; chunking composes with sharding only through `shard_map`. | `chunking.md` | chunk_size |

## 5. The uniform layer

| Term (aliases) | Where defined / best explained | Suggested definition | Link to | Related |
|---|---|---|---|---|
| ragged layer (ragged representation) | `batching_and_stacking.md` "Glossary" | The default representation: tuples of per-mode arrays of differing shapes. | `batching_and_stacking.md` | uniform layer |
| uniform layer (`UniformTuckerTensorTrain`, `UT3Frame`, …; UT3) | `user_guide.rst` "Uniform Tucker tensor trains"; `uniform_equivalence_contract.md` | A padded mirror of the ragged layer: cores zero-padded to common sizes and stacked into supercores, with boolean masks, so operations run under `lax.scan` and compile once. **Overload of "uniform":** also "uniform continuation" (all ranks grown together), "uniform bump", "uniform stack". | `uniform_equivalence_contract.md` | supercore |
| supercore (Tucker supercore, TT supercore) | `user_guide.rst` "Uniform Tucker tensor trains"; `uniform_supercore_layout.md` | The `d` padded cores stacked on a leading axis: Tucker `(d,) + stack_shape + (n, N)`, TT `(d,) + stack_shape + (r, n, r)`. | `uniform_supercore_layout.md` | d-leading |
| `d`-leading layout | `uniform_supercore_layout.md` | The core index comes first in a supercore and the stack after it, so a sweep over cores is a `lax.scan` over axis 0. Ragged objects are stack-leading. | `uniform_supercore_layout.md` | supercore |
| padded widths (`n`, `r`, `N`; nominal rank; level's widths) | `uniform_ranks_and_varieties.md` "The representation"; `rank_continuation.md` l.246 | The common padded sizes of a uniform object: Tucker width `n`, TT width `r`, mode width `N = max Nᵢ`. "Nominal rank", "padded `N`" and "the level's widths" are used for these without definition. | `uniform_ranks_and_varieties.md` | edge masks |
| edge masks (rank masks; Tucker edge masks `(d, n)`, TT edge masks `(d+1, r)`) | `user_guide.rst` "Uniform Tucker tensor trains"; `uniform_masks_vs_ranks.md` "What a mask means" | Per-edge boolean vectors over the padded size marking the real slots; the integer rank is `mask.sum(-1)`. Equivalently a diagonal projector onto the real subspace. | `uniform_masks_vs_ranks.md` | real slot |
| `shape_mask` † | `uniform_ranks_and_varieties.md` l.71; `uniform_masks_vs_ranks.md` l.70 | The mask of real entries along each padded mode axis; never gappy. Not in the documented `.data` layout, and `uniform_equivalence_contract.md` says shape padding is "keyed on the shape ints — no mask needed". | `uniform_masks_vs_ranks.md` | packed vectors |
| real slot vs padding (padded slot; garbage; don't-care) | `uniform_equivalence_contract.md` "The contract"; `uniform_masks_vs_ranks.md` | A real slot is a masked-in coordinate whose content is meaningful; padding is everything else, explicitly don't-care but required to be finite. The user docs never say "junk slot". | `uniform_equivalence_contract.md` | edge masks |
| prefix mask (canonical form) vs gappy mask (working form) | `uniform_masks_vs_ranks.md` "Canonical form vs working form" | Canonical: real slots form a prefix `(1,…,1,0,…,0)`, as T3-SVD produces. Working: any pattern, e.g. after `+` (masks concatenate) or `×` (masks Kronecker); data is not compacted. | `uniform_masks_vs_ranks.md` | ut3svd |
| equivalence contract | `uniform_equivalence_contract.md` "The contract" | "`to_uniform → op_uniform → to_ragged == op_ragged` on the masked ('real') parts only." | `uniform_equivalence_contract.md` | up to gauge |
| up to gauge (gauge-carrying operations; full-rank qualifier) | `uniform_equivalence_contract.md` "Gauge-carrying operations" | Orthogonalization, `t3svd` and frames match across layers only up to the SVD's gauge freedom, and gauge-invariant quantities match exactly only at a numerically full-rank point. At a rank-deficient point (every continuation restart) the two layers legitimately diverge. | `uniform_equivalence_contract.md` | pad-safe SVD |
| pad-safe SVD (`pad_safe_svd`) | `uniform_equivalence_contract.md` "The full-rank qualifier"; `CHANGELOG.md` 2026.1.0 | A mask-aware SVD of a zero-padded matrix whose leading triplets are exactly the economy SVD of the real block, with completion directions kept off the padding. | `uniform_equivalence_contract.md` (+ `pad_safe_svd.pdf`) | pinned completion |
| packed vs ragged vectors (packedness mirror; `pack`/`unpack`) | `uniform_equivalence_contract.md` "The vector I/O boundary" | Mode vectors cross the ragged/uniform boundary as a `len=d` list (ragged) or one `(d,)+…+(N,)` zero-padded array (packed); uniform ops return the form they were given. | `uniform_equivalence_contract.md` | seam |
| seam (ragged/uniform seam; split-seam hooks; seam builders) † | `uniform_equivalence_contract.md` l.67, l.112; `naming_conventions.md` l.209 | Used for the ragged/uniform boundary and for the split `*_from_sweep` fitting hooks, never defined. | `uniform_equivalence_contract.md` | packed vectors |
| determinantal variety (bounded-rank variety; `V_{≤n,≤r}`) | `uniform_ranks_and_varieties.md` "The real question" | The set of tensors with ranks at most `(n, r)`; a padded uniform stack is a batch of points in it, the mask naming each point's stratum. | `uniform_ranks_and_varieties.md` | stratum |
| stratum | `uniform_ranks_and_varieties.md` "The representation" | The fixed-rank manifold a point lies on inside the variety; the mask names it. Also used for "the lower-rank stratum" a padded restart sits on. | `uniform_ranks_and_varieties.md` | variety |
| varying-rank stack | `uniform_ranks_and_varieties.md` | A uniform stack whose elements have different ranks (allowed along `C`, never along `K`). | `uniform_ranks_and_varieties.md` | C, K |
| `uniform_minimal` | `user_guide.rst` note l.273–283; `sharing.md` "On the uniform layer" | Reduces a uniform start to minimal (or shared-minimal) ranks; the uniform fitting path requires a minimal-rank start because its masks are loop-invariant. | `user_guide.rst` | minimal ranks |
| same-mask precondition | `uniform_masks_vs_ranks.md` "Tangent (variation) masks" | Uniform variation add/sub requires equal `shape` and masks (a structural, host-side check), since padding would hide a mismatch. | `uniform_masks_vs_ranks.md` | same-frame |
| gauge-shifted masks † | `uniform_ranks_and_varieties.md` l.104; `CHANGELOG.md` 2026.3.0 | The variation masks derived from the frame's masks; used, never defined. | `uniform_ranks_and_varieties.md` | edge masks |
| value-hashed masks (mask holder, `UT3Masks`) | `user_guide.rst` "NumPy and JAX"; `uniform_backend_jit_recipe.md` | Mask containers that hash and compare by value, so a rebuilt but identical mask set is the same jit cache key. | `uniform_backend_jit_recipe.md` | aux_data |

## 6. Fitting and optimization

| Term (aliases) | Where defined / best explained | Suggested definition | Link to | Related |
|---|---|---|---|---|
| misfit (least-squares misfit) | `fitting_and_optimization.md` intro | `½‖S(x) − y‖²` (weighted: `½‖ω⊙(S(x) − y)‖²`) for a sampling operator `S`. **Inconsistent data symbol:** `y` here, `b` in `entries_apply_probe.md` §4 and `getting_started.rst`. | `fitting_and_optimization.md` | objective |
| objective | `fitting_and_optimization.md` §4.9 | Misfit plus the optional regularization term `ρ(x)`; `stats['history']` records `objective`, `misfit`, `regularization`. | `fitting_and_optimization.md` §4.9 | regularizer |
| residual (`r`) | `fitting_and_optimization.md` §2.2 | `r = S(x) − y`, raw (unweighted). | `fitting_and_optimization.md` | measurement space |
| `data` / `sample` | `fitting_and_optimization.md` §2.1 (bullets) | `sample` is the measurement spec (`ww`, `index`, or `(ww, pp)` / `(index, pp)`); `data` is the observed `y`, raw. | `fitting_and_optimization.md` §2.1 | kind |
| sampling kind (kind; `SamplingKind`, `ApplyKind`, `ProbeKind`, …) | `fitting_and_optimization.md` §1, §3 | The measurement axis of a fit: the bare `𝒥`, `𝒥ᵀ`, point forward `S(x)` and `‖·‖²` reduction for one sampling operation; selected by a string (`'apply'`, `'probe_derivatives'`, …). Custom kinds are subclasses. | `fitting_and_optimization.md` §3 | geometry |
| draw (`draw(rng)`, `flat_draw`) | `fitting_and_optimization.md` §2.3, §4.4 | A user function returning a random minibatch `(sample_B, data_B)`; the default `flat_draw` picks a uniform random subset of the flattened `W`. Runs outside the compiled kernel. | `fitting_and_optimization.md` §2.3 | minibatch |
| optimizers (`gradient_descent`, `mc_sgd`, `adam`, `newton_cg`) | `fitting_and_optimization.md` §2.1 bullets | Gradient descent with Cauchy steps and Armijo; Manifold Cauchy SGD; corewise Adam; inexact Riemannian Newton-CG. | `fitting_and_optimization.md` | Newton-CG |
| Manifold Cauchy SGD (`mc_sgd`, MC-SGD) | `fitting_and_optimization.md` §2.1 | "Tuning-free stochastic" — the only description; the method itself is not explained. | `fitting_and_optimization.md` | Cauchy step |
| Cauchy step | `fitting_and_optimization.md` §2.2 (code) | The step length `α = gᵀg / ‖Jg‖²` that minimizes the quadratic model along `−g`. | `fitting_and_optimization.md` §2.2 | trust radius |
| Armijo line search † | `fitting_and_optimization.md` §2.1, §5 | Backtracking (40 halvings in `newton_cg`) until sufficient decrease; on failure the default `on_line_search_failure='stop'` ends the run. The Armijo condition itself is not stated. | `fitting_and_optimization.md` §5 | step_control |
| local model (`fitting.LocalModel`, `backend.optimizers.LocalModel`; formerly `GaussNewtonModel`) | `fitting_and_optimization.md` §2.2, §3; `user_guide.rst` "Fitting and optimization" | The quadratic model of the objective at a base point — gradient, Hessian actions, retraction — over one geometry; `T3Tangent` in and out on the frontend, raw cores on the backend. | `fitting_and_optimization.md` §2.2 | Problem |
| Gauss-Newton model (`m(p)`; GN normal operator) | `fitting_and_optimization.md` §1, §4.2 | `m(p) = c + gᵀp + ½ pᵀ(JᵀJ)p`. Exact on the affine tangent space because the sampling operations are linear in the tensor; not exact on the manifold when the residual is nonzero. | `fitting_and_optimization.md` §4.2 | exact Hessian |
| `Problem` / `least_squares_problem` / `LeastSquaresSetup` | `fitting_and_optimization.md` §3 | `Problem(geom, kind, sample, data)` is the backend oracle that builds local models; `optimizers.least_squares_problem` builds it from frontend vocabulary and returns `.problem`, `.x0_cores`, `.rewrap`. | `fitting_and_optimization.md` §3 | local model |
| residual weight (`ω`, `weight=`; per-order, per-mode) | `fitting_and_optimization.md` §4.6 | A per-`(mode, order)` matrix in the objective `½‖ω⊙r‖²`; a bare vector is per-order, a column per-mode (probe only). **Collides** with edge weights; the docs fix it as "`weight` = residual, `weights` / `*Weights` = edge". | `fitting_and_optimization.md` §4.6 | edge weights |
| regularizer (`ρ(x)`; `IdentityRegularizer`, Tikhonov, HS ridge, weight decay) | `fitting_and_optimization.md` §4.9 | An optional objective term. `IdentityRegularizer(λ)` is `½λ‖x‖²`: an HS ridge on `MANIFOLD`, weight decay on `COREWISE`. **Symbol collision:** `ρ` is also the actual-over-predicted ratio. | `fitting_and_optimization.md` §4.9 | objective |
| Newton-CG (inexact Riemannian Newton-CG) | `fitting_and_optimization.md` §2.1, §4.12 | Each Newton step solves the local model approximately by conjugate gradients, then retracts. | `fitting_and_optimization.md` | forcing term |
| forcing term / forcing power (`η`, `cg_forcing_power`) | `fitting_and_optimization.md` §5 | CG stops at relative residual `η = min(0.5, (‖g‖/‖g₀‖)^cg_forcing_power)`; a larger power spends more CG per Newton step. | `fitting_and_optimization.md` §5 | g0 reference |
| relative stop (`gtol_rel`) | `fitting_and_optimization.md` §5 | Newton stops when `‖g‖ ≤ gtol_rel·‖g₀‖`. | `fitting_and_optimization.md` §5 | g0 reference |
| g0 reference (`g0_reference`: `'zero'`, `'warm_start'`, `'cone'`; Newton reference; `‖g₀‖`; `cg_reference`) | `fitting_and_optimization.md` §5 (table) | The gradient norm the relative stop and forcing term divide by: at the zero tensor of `x0`'s ranks (default on manifold geometries), at the start `x0`, or at the zero tensor on the aligned frame. `g0norm_newton`/`g0norm_cg` override it. | `fitting_and_optimization.md` §5 | cone reference |
| canonical zero-point frame (`fv_zero_point_frame`, `ZERO_FRAME_SEED`) | `fitting_and_optimization.md` §5 "What exactly the reference is" | The fixed frame at `X = 0` where the zero reference is evaluated: seeded Haar-random Tucker factors (one per sharing group) and identity-prefix TT cores. Not coordinate-aligned, the same on both layers, nested across levels. | `fitting_and_optimization.md` §5 | zero reference |
| cone reference (`g0_reference='cone'`, `cone_*` options) | `aligned_padded_frame.md` "In the optimizers"; `fitting_and_optimization.md` §5 | `‖Π_{T(F)} g(0)‖` at the frame `F` aligned with the misfit gradient at the zero tensor. Why it is called "cone" is not explained in the user docs (only in `padded_frame_cone.pdf`). | `aligned_padded_frame.md` | cone merit |
| warm start / cold start / zero start | `rank_continuation.md` "How to use it"; `fitting_and_optimization.md` §4.11 | Warm: the previous level's point zero-padded by `resize`. Cold: the zero tensor of the level's ranks. Zero start: starting a fit at the zero tensor, fine on the manifold, not on `COREWISE`. | `rank_continuation.md` | restart |
| zero-padded restart (padded restart; continuation restart) | `numerical_contracts.md` "Minimal ranks" (`retract` row); `uniform_equivalence_contract.md` | A warm start at grown ranks; numerically rank-deficient (the new directions have zero singular values), so `retract` is not first-order accurate there and the two layers diverge. | `uniform_equivalence_contract.md` | warm start |
| exact Hessian (`hessian='exact'`; exact Riemannian Hessian, exact model) | `fitting_and_optimization.md` §4.12 | The Gauss-Newton term plus the manifold's curvature: `H p = Π𝒥ᵀω²𝒥Π p + Π Lᵀ(K_{Aᵀω²r} − K_{grad f}) L Π p (+ H_R p)`. Gives a superlinear tail when the residual does not vanish. | `fitting_and_optimization.md` §4.12 | curvature term |
| curvature term (Weingarten term) | `fitting_and_optimization.md` §4.12 | The residual's normal part paired with the manifold's second fundamental form (Absil, Mahony & Trumpf 2013); `curvature_hessian`/`curvature_quadratic` give it alone. It can be negative. | `fitting_and_optimization.md` §4.12 | lift |
| lift (`L`, `tv_corewise_lift`, `LiftFactors`) | `fitting_and_optimization.md` §4.12; `naming_conventions.md` "lift" | Maps gauged frame coordinates to raw core directions (`θ̂ = L v`) through clipped pseudo-inverses of the bond and Tucker spectra. **Overload:** "lift" also means extending an atomic operation over stacks (`batching_and_stacking.md` §11), a lifted tensor (`sharing.md` face 4) and moving to the tangent setting (`entries_apply_probe.md`). | `fitting_and_optimization.md` §4.12 | curvature_damping |
| `curvature_damping` (`ε`, `lift_eps`) | `fitting_and_optimization.md` §4.12 API table | The lift's damping, relative to `‖X‖_HS` (default `1e-6`): directions with lift singular value `σ ≫ ε` get the exact curvature, those with `σ ≲ ε` are damped toward Gauss-Newton. | `fitting_and_optimization.md` §4.12 | lift_sigma_min |
| `lift_sigma_min` (`σ` in the display) | `fitting_and_optimization.md` §4.12 "Reading the display" | The smallest singular value the lift inverts; the quadratic model's validity radius is about `σ`. | `fitting_and_optimization.md` §4.12 | κ display |
| display `κ` (`pKp`) | `fitting_and_optimization.md` §4.12 "Reading the display" | `κ = pKp / (pHp − pKp)`: the curvature term's curvature along the step relative to Gauss-Newton's. **Symbol collision:** `κ` is also the edge condition number and `κ_g`. | `fitting_and_optimization.md` §4.12 | edge condition number |
| trust region (Steihaug–Toint; `step_control`, `tr_radius0`, `Δ`) | `fitting_and_optimization.md` §4.12 "The API" | CG inside `‖p‖ ≤ Δ`, stopping at the boundary on nonpositive curvature; one trial retraction accepted iff `ρ > 0.1`; `Δ` shrinks to `¼‖p‖` when `ρ < 0.25`, doubles (capped at `10⁴·Δ0`) when `ρ > 0.75` on a boundary step. `Δ0` defaults to the Cauchy length `‖g‖³/⟨g, H_GN g⟩`. | `fitting_and_optimization.md` §4.12 | Cauchy step |
| actual-over-predicted ratio (`ρ`) | `fitting_and_optimization.md` §2.1 bullets, §4.12 | `ρ = Δf / (gᵀp + ½pᵀHp)`. Collides with the regularizer `ρ(x)`. | `fitting_and_optimization.md` §4.12 | trust region |
| q-quadratic † | `fitting_and_optimization.md` §4.12 "The forcing power" | Quadratic convergence of the error sequence (`e_{k+1} ≤ c·e_k²`); used, not defined. | `fitting_and_optimization.md` | forcing power |
| `stop_reason` (`'gradient'`, `'max_newton'`, `'line_search'`, `'objective_floor'`, `'trust_region_floor'`) | `fitting_and_optimization.md` §2.1 stats table | Why `newton_cg` stopped. `objective_floor` is "not a failure": a rejected trial predicted a change below the objective's floating-point resolution. `trust_region_floor`: `Δ < 10⁻¹²·Δ0`. | `fitting_and_optimization.md` §2.1 | stats |
| `newton_steps` / `newton_accepted` (formerly `newton`) | `fitting_and_optimization.md` §2.1 stats table | CG solves (including rejected trust-region trials) / steps actually taken. | `fitting_and_optimization.md` | history |
| history (`stats['history']`, `NewtonInfo`) | `fitting_and_optimization.md` §2.1 stats table | One scalar row per Newton iteration with the objective split, gradient norm, CG outcome, reference kinds, trust-region and curvature fields. | `fitting_and_optimization.md` | verbose display |
| verbose display (live diagnostics; `val_sample`/`val_data`) | `fitting_and_optimization.md` §2.1 "Live diagnostics" | `verbose=True` prints a per-iteration block plus a per-`(mode, order)` relative-error table; validation data adds a column. | `fitting_and_optimization.md` | history |
| stacked point (stacked optimization) | `fitting_and_optimization.md` §4.10 | A point with a `C` stack; the local model supports it, the four optimizers raise `NotImplementedError`. | `fitting_and_optimization.md` §4.10 | C |
| halo | `fitting_and_optimization.md` §5 | The large content a fit puts in the null space the data does not constrain (e.g. the non-symmetric part when only symmetric data exist). | `fitting_and_optimization.md` §5 | — |
| `chunk_size` (chunking; `'auto'`) | `chunking.md` "What chunk_size does" | Computes the probe-derivative transpose's assembly (and its two-hole) in slices of that many probes; bit-identical result, lower peak memory, active only on the uniform + jax path. `'auto'` lets the optimizers size it. | `chunking.md` | assembly |
| balance policy (`estimate_chunk_size`) | `chunking.md` "Choosing chunk_size" | Choose the chunk so the assembly's peak is comparable to the edge-variable jets already resident (total ≈ 2× the floor); measured from the compiled program. | `chunking.md` | max_chunk_size_within |
| `max_chunk_size_within` (`target_bytes`) | `chunking.md` "Choosing chunk_size" | The largest chunk whose assembly peak stays under an absolute byte budget. | `chunking.md` | chunk_size |

## 7. Rank continuation

| Term (aliases) | Where defined / best explained | Suggested definition | Link to | Related |
|---|---|---|---|---|
| rank continuation | `rank_continuation.md` intro | Solve at small ranks, grow the ranks, re-solve warm-started, and stop when more rank stops helping (or the next model is underdetermined). | `rank_continuation.md` | level |
| level (continuation level) | `rank_continuation.md` (loop code, l.96, l.135) | One fixed-rank fit within a continuation. Never defined. **Overload:** "levels" of a spectrum (`s_g`'s new levels) and "one level up" (API layer). | `rank_continuation.md` | ladder |
| ladder † | `rank_continuation.md` l.118; `aligned_padded_frame.md` "What the measurements showed"; `release_notes.md` l.29 | The sequence of continuation levels of one run. Never defined. | `rank_continuation.md` | level |
| uniform continuation | `rank_continuation.md` "Why not just grow every rank together?" | Growing every rank to the same level `r`, as opposed to per-edge growth. Unrelated to the uniform layer. | `rank_continuation.md` | uniform layer |
| edge condition number (`κᵢ`, `edge_condition_numbers`) | `rank_continuation.md` "The idea" | `κᵢ = σ₁ / σ_lastᵢ`: the ratio of the largest to the smallest retained singular value on an edge. | `rank_continuation.md` | tau |
| `tau` | `rank_continuation.md` "Parameters" | Grow only edges whose `κᵢ` is at least a factor `tau` below the worst edge's (default 10). | `rank_continuation.md` | edge condition number |
| useless-rank removal (uniform bump) | `rank_continuation.md` "The idea" | Making proposed ranks non-degenerate from shape and ranks alone; the uniform bump is the fallback growth when every edge is comparably conditioned. | `rank_continuation.md` | minimal ranks |
| `n_chunk` / `max_grow` / `kappa_guard` | `rank_continuation.md` "Parameters" | Growth per chosen edge per round; cap on edges grown per round; absolute conditioning limit above which an edge never grows (default `1e12`). | `rank_continuation.md` | continuation_ranks |
| `continuation_ranks` (`compute_continuation_ranks`) | `rank_continuation.md` "How to use it" | Proposes the next ranks from the current iterate's T3-SVD spectra; returns the current ranks when continuation should stop. Do not feed it uniform spectra. | `rank_continuation.md` | resize |
| `oversampled_ranks` | `CHANGELOG.md` 2026.4.0 Added | `+p` on every rank, structurally minimal, never below the input. Not mentioned in any user page. | `CHANGELOG.md` | oversampling |

## 8. The padded and aligned frames

| Term (aliases) | Where defined / best explained | Suggested definition | Link to | Related |
|---|---|---|---|---|
| padded frame (`T3Frame.padded`, `fv_padded_frame`) | `aligned_padded_frame.md` "The padded frame" | A frame for `x0` (= `x_old` zero-padded) that keeps `x_old`'s frame bitwise in the leading slots of every store and fills the new slots with an orthonormal completion. **Overload of "padded":** uniform padding, zero-padded warm start, frame-internal padding cores. | `aligned_padded_frame.md` | store, completion |
| old block / new block (old slots / new slots) | `aligned_padded_frame.md` "The padded frame", "The uniform layer"; `numerical_contracts.md` "The padded and aligned frames" | In each store, the leading slots holding `x_old`'s frame and the trailing slots added by the rank increase. The user docs do not use "A-block" / "P-block". | `aligned_padded_frame.md` | completion |
| completion (orthonormal completion) | `aligned_padded_frame.md` "The padded frame"; `numerical_contracts.md` | Orthonormal columns for the new slots, orthogonal to the old block. **Overload:** null-space completion of an SVD (`uniform_equivalence_contract.md`), "completion directions" (`uniform_masks_vs_ranks.md`), tensor completion (application). | `aligned_padded_frame.md` | pinned completion |
| hint-free start (hint-free padded frame) | `aligned_padded_frame.md` "The padded frame" | The data-independent completion: new slot `s` is seeded by row `s` of a keyed Gaussian stream and completed against the old block by a sign-fixed QR. | `aligned_padded_frame.md` | seed |
| keyed Gaussian stream / slot-indexed seeds (`padded_seed`, `cone_seed`) | `aligned_padded_frame.md` "The padded frame" | The deterministic random rows used to seed new slots, indexed by slot, so successive starts never reuse a seed. | `aligned_padded_frame.md` | hint-free start |
| alignment (aligned padded frame; `fv_aligned_padded_frame`) | `aligned_padded_frame.md` "The alignment" | A fixed-point iteration that, each pass, projects the hint onto the current frame's tangent space and replaces each store's new slots by the leading new directions of that projection's content, given the other stores. Old slots stay fixed. | `aligned_padded_frame.md` | hint, pass |
| hint | `aligned_padded_frame.md` "The alignment" | "A tensor whose projection onto the frame's tangent space should be large." `newton_cg` uses the misfit gradient. | `aligned_padded_frame.md` | alignment |
| `project` (custom projection) | `aligned_padded_frame.md` "Jit or eager"; `fitting_and_optimization.md` §3 | The callable that projects the hint onto a frame's tangent space; must be traceable and module-level under uniform jit. | `aligned_padded_frame.md` | ambient projection |
| pass (alignment pass; `T3Tangent.align_padded`, `update`) | `aligned_padded_frame.md` "The alignment"; `CHANGELOG.md` 2026.4.0 | One update of the selected stores' new slots; stores not in `update` are transported. | `aligned_padded_frame.md` | schedule |
| captured hint energy | `aligned_padded_frame.md` "What the fixed point satisfies" | `‖Π_{T(F)} hint‖²`; the fixed point is a stationary point of it, generally not a maximum. | `aligned_padded_frame.md` | cone merit |
| cone merit | `aligned_padded_frame.md` "What the fixed point satisfies" | "The share of the hint's normal part that the new directions represent on one side of every edge." Measured fixed points are strict local maxima of it, one of many. | `aligned_padded_frame.md` | cone reference |
| movement (movement record; `start_movements`, `cone_movements`) | `aligned_padded_frame.md` "The schedule", "The uniform layer" | How far a pass turned the new directions: the largest over the stores of `‖g_old − P⁺P⁺ᵀ g_old‖_F`; 0 at a fixed point. The uniform record is a fixed-length array, NaN after the last pass. | `aligned_padded_frame.md` | tol |
| schedule (alignment schedule) | `aligned_padded_frame.md` "The schedule" | Oversampled passes, one truncating pass, then two-colour pairs until both movements of a pair are below `tol` or `max_passes` is reached. | `aligned_padded_frame.md` | pass |
| oversampling / oversampled passes (`oversampling`, `oversampled_passes`) | `aligned_padded_frame.md` "The schedule" | Passes at the target ranks plus `oversampling` (default 5), run `oversampled_passes` times (default 2) before truncating. Unrelated to `t3m`'s `oversample`. | `aligned_padded_frame.md` | truncating pass |
| truncating pass | `aligned_padded_frame.md` "The schedule" | The single pass that cuts the oversampled frame back to the target ranks; it never stops the loop. | `aligned_padded_frame.md` | schedule |
| two-colour pairs (`'UL'`, `'RO'`; `alternate`) | `aligned_padded_frame.md` "The schedule" | A `'UL'` pass (up and left stores) then an `'RO'` pass (right and down), the other colour transported; avoids the period-2 cycling of simultaneous updates. | `aligned_padded_frame.md` | pass |
| `tol` / `max_passes` / zero passes | `aligned_padded_frame.md` "The schedule" | Stop threshold on movement (default 0.05); cap on all passes (default 10); `max_passes=0` returns the hint-free start. In practice every measured run hit the 10-pass cap. | `aligned_padded_frame.md` | movement |
| `start_frame` / `padded_from` | `aligned_padded_frame.md` "In the optimizers"; `release_notes.md` "Upgrading from 2026.3.0" | `newton_cg` keywords: `start_frame=F` linearizes iteration 0 at a given frame; `padded_from=x_old` builds the aligned frame (options `padded_*`). | `aligned_padded_frame.md` | padded frame |
| structurally zero point | `aligned_padded_frame.md` "The zero point" | A T3 with some Tucker rank or interior TT rank 0, hence the zero tensor; its frame has no old slots. | `aligned_padded_frame.md` | zero-point frame |
| representation dependence | `aligned_padded_frame.md` "Representation dependence" | The hint-free start depends on how `x_old`'s frame is represented (slot signs, bases at singular-value ties); the aligned frame does not, up to `tol`. | `aligned_padded_frame.md` | gauge |
| block layout (split `(Na, Rb)`) | `aligned_padded_frame.md` "The uniform layer" | The uniform loop's internal layout: every rank slot is `[old block | new block]`, each padded to one width with its own mask; `(Na, Rb)` are the old block's Tucker and bond widths. | `aligned_padded_frame.md` | old block |
| compacted frame (`ufv_compact_frame`) | `aligned_padded_frame.md` "The uniform layer" | The block-layout frame rewritten to the level's widths with real slots first and prefix masks, so Newton starts without recompiling. | `aligned_padded_frame.md` | prefix mask |
| eager jax (eager compilation cost) | `aligned_padded_frame.md` "Jit or eager" | Running jax ops outside `jit`: each new structure compiles its ops one by one, slow and memory-map hungry. "With jax inputs, pass `use_jit=True`." | `aligned_padded_frame.md` | use_jit |
| pinned completion (`pinned_completion`, `pinned_leading_columns`, `pad_safe_pinned_completion`) † | `CHANGELOG.md` 2026.4.0 Added only | Backend completion helpers; never defined in user docs. | (contributor docs) | completion |
| physical rows † | `aligned_padded_frame.md` l.115 | "Tucker completions are seeded in physical rows, which have no sign gauge" — presumably rows indexed by the mode size `Nᵢ`. Undefined. | `aligned_padded_frame.md` | hint-free start |

## 9. Shared Tucker factors

| Term (aliases) | Where defined / best explained | Suggested definition | Link to | Related |
|---|---|---|---|---|
| shared Tucker factors (SF-T3; shared-factor format; SF-Tucker, SF-ETT) | `sharing.md` intro | T3s whose Tucker factors are constrained equal within user-chosen groups of modes. SF-Tucker and SF-ETT are the published predecessors; T3Toolbox allows any partition. **Abbreviation clash:** `SF` means "same-frame" in `numerical_contracts.md`. | `sharing.md` | sharing spec |
| sharing spec (`sharing`) | `sharing.md` intro | "A length-`d` tuple of hashable labels, one per mode; modes with equal labels share one factor." | `sharing.md` | sharing group |
| sharing group (group; partition) | `sharing.md` intro | The set of modes with one label; the groups partition the modes. Group modes need equal mode sizes. **Overload:** "group" also means a `contract` axis group. | `sharing.md` | sharing spec |
| tied / untied (`TIED`, `has_shared_tucker_factors`, `t3_tie_tucker_factors`) | `sharing.md` "The format"; `numerical_contracts.md` table | Tied: the Tucker factors are numerically equal within each group (one array per group). Untying happens when a per-mode operation clips or weights group modes differently. | `sharing.md` | share |
| `share()` (entering the format) | `sharing.md` "The format" | Projects every group mode onto the dominant left singular subspace of the group's concatenated matricizations (the quasi-optimal shared basis), then rounds with the grouped T3-SVD. | `sharing.md` | grouped T3-SVD |
| grouped T3-SVD (two-phase truncation) | `sharing.md` "Grouped truncation" | Round the TT bonds with Tucker steps skipped, collect every mode's center from that same tensor, then truncate each group with one SVD of its concatenated centers. | `sharing.md` | group spectrum |
| group spectrum (`s_g`) | `sharing.md` "What the group spectrum is" | The singular values of the concatenated matricizations `[T_(i₁) | … | T_(iₖ)]`; `√k`-inflated relative to a single-mode spectrum. | `sharing.md` | κ_g |
| `κ_g` (group condition number) | `sharing.md` face 3; `rank_continuation.md` "Shared Tucker factors" | `s_{g,1}/s_{g,n_g}`, the conditioning of the tied-factor subproblem; one entry in the continuation pool per group. | `sharing.md` | edge condition number |
| group ceiling | `sharing.md` "Rank machinery"; `t3svd_minimal_ranks.md` "With shared Tucker factors" | `n_g ≤ min(N_g, Σ_{i∈g} min(N_g, r_{i−1}·r_i))`: per-mode Tucker ceilings add across a group. | `sharing.md` | minimal ranks |
| shared geometry wrappers (`shared(geometry, sharing)`, `shared_manifold`, `shared_corewise`) | `sharing.md` "Optimizing over shared factors" | Wrap any of the four geometries so every optimizer stays on the tied set: a tilted least-squares projection on the manifold, the per-group mean on corewise. | `sharing.md` | tied projection |
| companion (shared-frame companion; per-frame companion; SVD companion; `SharedFrameData`) | `sharing.md` "Batching"; `fitting_and_optimization.md` §3 | The per-frame data the shared manifold wrapper derives — "the group centers and stacked-`S` SVD" — computed once per local model as the geometry aux. | `sharing.md` | geometry aux |
| `S_i` factors / centers † | `sharing.md` "Optimizing over shared factors"; `CHANGELOG.md` 2026.3.0 | `S_i` are "part of the orthogonal frame's own factorization"; centers are the per-mode center cores. Neither is defined in the user docs. | `sharing.md` | companion |
| tied lift | `sharing.md` "The exact Hessian" | The lift on a shared manifold: each group's coordinates lift to one raw factor direction `U̇ = M_g⁺[V_{i₁}; …]`. | `sharing.md` | lift |
| gated channel / escape | `sharing.md` "Continuation" | After a padded shared restart the new `s_g` levels are zero, so the tied Tucker channel cannot move ("gated"); the TT variations carry the "escape" in the first steps. Informal terms. | `sharing.md` | zero-padded restart |
| Stiefel term † | `sharing.md` "Dimension" | `n_g(N_g − n_g)`, the dimension contributed by one orthonormal factor modulo rotation; the term of art is not explained. | `sharing.md` | manifold_dim |

## 10. Edge weighting

| Term (aliases) | Where defined / best explained | Suggested definition | Link to | Related |
|---|---|---|---|---|
| edge weights (weighted layer; `weights`) | `weighting.md` intro | Diagonal matrices imagined on the internal edges of a T3, stored as one vector per edge. | `weighting.md` | residual weight |
| `T3Weights` | `weighting.md` "T3Weights — a weighted tensor" | One diagonal per Tucker edge (`d`) and per TT bond (`d+1`), the shape `t3svd` returns; weights a tensor. | `weighting.md` | absorb |
| `T3FrameWeights` (tangent metric) | `weighting.md` "T3FrameWeights — a metric on a tangent" | A diagonal metric on a tangent's coordinates: `d` weights in each of the `up`, `down`, `left`, `right` families. Batches like the frame (`C`, never `K`). | `weighting.md` | Grasedyck–Kramer |
| absorb (`*_absorb_weights`) | `weighting.md` | Fold the diagonals into the cores (tensor) or variation cores (tangent); shape-preserving. Absorbed tangents are not gauged. | `weighting.md` | weighted norm |
| `+` / `⊙` duality (`concatenate`, `kronecker`) | `weighting.md` "The + / ⊙ duality" | Adding trains concatenates their edge weights (ranks add); the Hadamard product Kroneckers them (ranks multiply). | `weighting.md` | gappy mask |
| Grasedyck–Kramer preconditioner (singular-value metric) | `weighting.md` intro; `frame_variations.md` "Weights are a diagonal metric" | Inverse singular values on the coordinate edges, penalizing the poorly-informed tangent directions. | `weighting.md` | T3FrameWeights |

## 11. Conventions

| Term (aliases) | Where defined / best explained | Suggested definition | Link to | Related |
|---|---|---|---|---|
| frontend / backend (`.data`) | `user_guide.rst` "Frontend and backend"; `api_reference.rst` | Frontend: frozen dataclasses re-exported at the root, methods delegating to the backend. Backend: stateless functions on raw `.data` tuples, imported by submodule, a first-class surface. | `user_guide.rst` | family prefix |
| family prefix (`t3_`, `ut3_`, `tv_`, `utv_`, `fv_`, `ufv_`, `tt_`, `dense_`) | `naming_conventions.md` "The family prefix grammar" | The prefix naming a backend function's primary operand. `tv_` treats a (frame, variations) pair as the sum of its terms; `fv_` as indexed pieces. | `naming_conventions.md` | polymorphism |
| polymorphic (ragged name is polymorphic; `u`-prefix) | `naming_conventions.md` "Polymorphism" | A backend op that accepts ragged or uniform operands lives unprefixed in the base family; a `u` prefix signals uniform-specific logic. | `naming_conventions.md` | family prefix |
| checker (`is_`/`has_`; `_residual`) | `naming_conventions.md` "Semantic markers" | "A non-enforcing predicate that reports a property, returning a verdict per stack element." Safe mode raises at the call site; the paired residual is the continuous quantity. | `naming_conventions.md` | safe mode |
| `_hessian` / `_tangent` (dotted twins) / `_from_sweep` suffixes | `naming_conventions.md` "Semantic markers" | `_hessian`: derivative of the same-stem corewise transpose along raw core directions. `_tangent`: a kernel's directional derivative ("dotted twin"). `_from_sweep`: reuses a precomputed frame sweep. | `naming_conventions.md` | two-hole sweep |
| capability-honest tails | `naming_conventions.md` "Semantic markers" | A uniform twin keeps a different name when it supports less (e.g. `ut3_full_sum`). | `naming_conventions.md` | intended asymmetries |
| brand names (`t3svd`, `t3m`) | `naming_conventions.md` "Cataloged exceptions" | Algorithm names exempt from the prefix grammar. | `naming_conventions.md` | — |
| razor (backend razor) † | `fitting_and_optimization.md` §3 diagram; `naming_conventions.md` l.162 | Used twice with different glosses ("a raw-.data user runs the SAME check-free code"; "the product-rule bookkeeping is easy to get wrong"); never defined. | (needs a definition or removal) | frontend/backend |
| dispatch (numpy/jax inferred) | `user_guide.rst` "NumPy and JAX" | "Dispatch is inferred from the input array types at the lowest level": numpy in, numpy out; jax in, jax out. | `user_guide.rst` | use_jax |
| `use_jax` | `user_guide.rst` "NumPy and JAX" | The only backend flag, on constructors with no array inputs (`randn`, `zeros`, `ones`, `load`). | `user_guide.rst` | dispatch |
| `use_jit` | `fitting_and_optimization.md` §4.5; `aligned_padded_frame.md` "Jit or eager" | Jits the optimizers' per-step kernel (and the alignment); converts numpy inputs to jax and returns jax results (float32 unless x64). Without jax installed it warns and runs eager (since 2026.2.0). | `fitting_and_optimization.md` §4.5 | eager jax |
| pytree leaf vs `aux_data` (static structure) | `user_guide.rst` "NumPy and JAX"; `batching_and_stacking.md` §7 | Leaves are traced arrays (cores, a tangent's frame); `aux_data` is static structure (masks, geometries, kinds) that is part of the jit cache key. | `batching_and_stacking.md` §7 | value-hashed |
| jit cache key / compile once / recompile | `uniform_masks_vs_ranks.md` "What the masks do"; `uniform_backend_jit_recipe.md` | A compiled program is reused while the static structure is value-equal; a recompile happens exactly when ranks change. | `uniform_backend_jit_recipe.md` | aux_data |
| parameters-not-closures (parameters as dataclass fields) | `fitting_and_optimization.md` §3 "Writing your own"; `CHANGELOG.md` 2026.2.0 | Geometries and kinds are frozen dataclasses whose parameters are fields, so they hash and compare by value and form correct jit cache keys; a variant is a subclass. | `fitting_and_optimization.md` §3 | aux_data |
| safe mode / unsafe mode (safety mode; `t3toolbox.unsafe()`) | `user_guide.rst` "Safe and unsafe mode"; `numerical_contracts.md` | Safe mode (default) checks genuine numerical preconditions and raises; `with t3toolbox.unsafe():` (and any jax trace) skips them. Results are identical either way. | `numerical_contracts.md` | precondition |
| structural vs numerical (errors) | `user_guide.rst` "Safe and unsafe mode" | Structural problems (shapes, ranks, stack shapes) always error; numerical preconditions are checked only in safe mode. | `user_guide.rst` | safe mode |
| precondition vs caveat | `numerical_contracts.md` "The classification rule" | Precondition: without it the result is wrong (checked). Caveat: the result is correct, the property only governs what it means (never checked). | `numerical_contracts.md` | numerical contracts |
| numerical contracts | `numerical_contracts.md` | The op-by-op statement of what each operation requires and what it guarantees. **Overload of "contract":** equivalence contract, shape contract, mask-representation contract. | `numerical_contracts.md` | safe mode |
| same-frame (`SF`; `frames_equal`), `ORTH`, `GAUGE`, `TIED` | `numerical_contracts.md` "The numerical properties" | The four checked properties: two tangents share a tangent space (numerical frame equality); the frame is orthonormal; the variations are gauged; shared factors are tied. | `numerical_contracts.md` | checker |
| gauge-check scale (operand scale) | `numerical_contracts.md` "The gauge-check scale" | The gauge residual is relative; a tangent built by arithmetic is measured against its operands' scale, a computed one against its own norm. | `numerical_contracts.md` | GAUGE |
| explicit equality (`allclose`, `corewise_equal`) | `numerical_contracts.md` "Equality is explicit" | `==` is undefined on runtime classes; `allclose` compares what objects represent (per stack element), `corewise_equal` compares stored bits. | `numerical_contracts.md` | — |
| intended asymmetries (uniform vs ragged) | `naming_conventions.md` "Intended asymmetries" | Places where the uniform layer deliberately offers less (no `to_vector`, no `rtol`/`atol`, …); not TODOs. | `naming_conventions.md` | uniform layer |
| T4S (the T4S paper) † | `frame_variations.md` l.58; `batching_and_stacking.md` §8; `entries_apply_probe.md` §8 | Abbreviation for Alger, Christierson, Chen & Ghattas (2026), "Tucker Tensor Train Taylor Series". Never expanded. `uniform_ranks_and_varieties.md` l.38 says the library is "named for" it, which is misleading. | `index.rst` | — |

---

## Notes

### Used without definition anywhere in the partition (†)

Hilbert–Schmidt; sweep (as a term); zipper; TTM; tensor degree; oblique projection (`project_oblique`);
ambient projection (`project_ambient`); vector transport (`transport`); implicit (in "implicit T3-SVD");
seam / split-seam / seam builders; `shape_mask` (status unclear); gauge-shifted masks; padded widths
("nominal rank", "padded `N`", "the level's widths"); Armijo; q-quadratic; Manifold Cauchy SGD (named,
not explained); level and ladder; pinned completion; physical rows; `S_i` factors and centers; Stiefel
term; razor; T4S (never expanded); "cone" (why the reference and merit are called that is only in
`padded_frame_cone.pdf`); Haar (term of art); Khatri-Rao (term of art); second fundamental form
(cited, not explained); "gated" / "escape" (informal); "points × directions" (downstream T3Polynomial
vocabulary in `aligned_padded_frame.md` and the CHANGELOG); Darcy (downstream problem name).

### Brief terms not found in this partition

- **A-block / P-block**: absent. The user docs say "old block / new block" (`aligned_padded_frame.md`)
  and "old slots / new slots". If the contributor docs use A/P, the glossary should list both.
- **HISTORICAL contraction strings**: absent. The only related user-facing term is the removed
  "named contraction functions" (`grouped_contractions.md` §5, `release_notes.md`).
- **junk slot**: absent. The user docs say "padding", "padded slots", "garbage" and "don't-care".
- **pinned completion**: CHANGELOG only (backend function names), no definition.
- **carrier**: one use (`aligned_padded_frame.md` l.24), no definition.

### Inconsistencies

1. **Tucker core vs Tucker factor vs central core.** `Bᵢ` is a "Tucker core" in README and
   `user_guide.rst`, a "Tucker factor" in `sharing.md`, `weighting.md` and `t3m_methods.md`, while
   "central Tucker core" (`user_guide.rst` l.9) and `C` in `t3m_methods.md` mean the Tucker core
   *tensor*. Pick one name for `Bᵢ`.
2. **Frame-core symbols.** Left/right TT frame cores are `L`/`R` in `user_guide.rst` and
   `frame_variations.md` but `P`/`Q` in the paper maps (`batching_and_stacking.md` §8,
   `entries_apply_probe.md` §8). `Oᵢ` is "down" and "outer" (`frame_variations.md` table), and
   `batching_and_stacking.md` §8 says it is "not outer"; `user_guide.rst` "Minimal rank conditions"
   says "Outer TT core unfoldings". Variations are `V`/`H` and `δU`/`δG`.
3. **`Hᵢ` overload.** TT variation (`frame_variations.md`) and frame "center" (`H_i = L_i Z_{i+1}`,
   CHANGELOG 2026.3.0).
4. **Padding cores indexed two ways.** `frame_variations.md` says each family is `L0…L(d−1)` but calls
   the unused left core `Ld`; `aligned_padded_frame.md` calls it `left[d-1]` and names both "carriers".
   `reading_signatures.md`'s example has `left_tt_cores: len=d-1`.
5. **`K` reused.** `naming_conventions.md` reserves `K` for the tangent stack, yet
   `fitting_and_optimization.md` §2.1 uses `K` as the derivative order (`jets` shape `(K+1,)+W`), and
   §4.12 uses `K_{Aᵀω²r}` for curvature operators.
6. **Other letter collisions.** `W` (probe stack; `T3Weights` variable in `weighting.md`; Khatri-Rao
   `Wᵢ` in `t3m_methods.md`; an integer in `chunking.md`); `C` (frame stack; central core; binomial
   `C(t,r)`); `ρ` (regularizer; actual/predicted ratio); `κ` (edge condition number; `κ_g`; display
   `κ = pKp/(pHp−pKp)`); `η` (edge variable; forcing term); `σ` (singular values; edge variable;
   `lift_sigma_min`); `Π` (gauge projector on variations; `Π_{T(F)}` ambient tangent projection).
7. **`SF`** means "same-frame" in `numerical_contracts.md` and "shared factors" (SF-T3) everywhere
   else, including in the same page's sharing table.
8. **"uniform"** has four senses: the uniform layer; "uniform continuation" (all ranks grown together);
   the "uniform bump"; a "uniform stack" (all cores share one `stack_shape`, `batching_and_stacking.md`
   §5).
9. **"padded"/"padding"** has four senses: uniform zero-padding to common widths; the zero-padded warm
   start; the padded frame; the frame-internal padding cores `Ld`/`R0`.
10. **"probe"** is an operation, a probe-vector set, and one sample row ("`|W|` = number of probes");
    README and `getting_started.rst` call `apply`'s inputs "probe rows". `W` is called "probe stack",
    "probe / sample stack", "vec/index stack" and "sample stack"; the vectors are "probe vectors",
    "test vectors" and "measurement vectors".
11. **Data symbol.** `y` in `fitting_and_optimization.md`, `b` in `entries_apply_probe.md` §4 and
    `getting_started.rst`. In `entries_apply_probe.md` "Symmetric directional derivatives", `X` is a
    sample point, whereas everywhere else it is the tensor.
12. **"frame-inner" misused.** `uniform_supercore_layout.md` l.21 calls "stack leads the core axes"
    the frame-inner convention; frame-inner is the `W + K + C` ordering of the blocks.
13. **Orthogonal vs orthonormal.** API names say "orthogonal" (`is_orthogonal`,
    `t3_orthogonal_representations`, "orthogonal frame"); prose says "orthonormal frame". They mean the
    same thing here.
14. **"gauge"** means the gauge conditions, the representation's gauge freedom, and "the T3-SVD gauge"
    (a specific frame choice).
15. **"lift"** means the curvature lift `L`, lifting an atomic operation over stacks
    (`batching_and_stacking.md` §11), a "lifted tensor" (`sharing.md` face 4), and moving to the
    tangent setting (`entries_apply_probe.md`).
16. **"completion"** means the padded frame's orthonormal completion, an SVD's null-space completion,
    the padded "completion directions" (`uniform_masks_vs_ranks.md`), and tensor completion.
17. **"geometry"** means the `Geometry` objects, but `weighting.md` intro uses it for tensor vs
    tangent ("two objects, one per geometry").
18. **`oversample` vs `oversampling`.** `t3m(method='swap', oversample=k)` and the alignment's
    `oversampling=5` are unrelated knobs with near-identical names.
19. **`use_jit` without jax.** `release_notes.md` "Upgrading from 2026.0.0" (and CHANGELOG 2026.1.0)
    say the call raises; `fitting_and_optimization.md` §4.5 and CHANGELOG 2026.2.0 say it warns and runs
    eager. The 2026.0.0 upgrade note is stale.
20. **`shape_mask`** is described in `uniform_ranks_and_varieties.md` and `uniform_masks_vs_ranks.md`,
    but is absent from the documented `.data` layout, and `uniform_equivalence_contract.md` says shape
    padding needs "no mask".
21. **Index base.** `user_guide.rst` mostly indexes from 0 (`N0…N(d−1)`) but writes
    `R^(N1 x … x Nd)` in "T3 manifold" and `N1 x N2 x N3` in the probing diagram. The minimal-TT-rank
    line reads "(N*...*Ni) x (N(i+1)*...*N(d-1))", which looks garbled (should be
    `(N0⋯N(i−1)) × (Ni⋯N(d−1))` for bond `rᵢ`).
22. **"level"** means a continuation level, a spectrum level (new levels of `s_g`), and an API layer
    ("one level up", "highest-level to lowest").
23. **T4S** is never expanded, and `uniform_ranks_and_varieties.md` l.38 says the library is "named
    for" T4S.
24. **"razor"** is used twice with different glosses and defined nowhere.
25. `entries_apply_probe.md` intro points users at `CLAUDE.md` for "the terse version", a contributor
    file outside the published docs.
