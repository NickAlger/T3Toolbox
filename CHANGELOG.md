# Changelog

All notable changes to T3Toolbox are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/); versions are `YYYY.MINOR.PATCH`.

## [2026.3.0] — unreleased

### Changed — breaking

- **Uniform transposes return clean padding.** Every uniform transpose wrapper that holds the frame's or
  the point's masks now re-masks its output, so the padded slots come back as exact zeros instead of the
  raw sweep's garbage (the boundary-bond squash broadcasts the gradient into the padded boundary slots;
  measured up to ~3e12 on a moderate case). That covers the six uniform kinds' `transpose`
  (`backend.uniform_fitting.Uniform*Kind`), the public `utv_{probe,apply,entries}_transpose` and
  `utv_*_derivatives_transpose` (and so `UT3Tangent.*_transpose`), and `ut3_*_corewise_transpose` /
  `ut3_*_corewise_derivatives_transpose` (and so `UniformTuckerTensorTrain.*_corewise_*transpose`). This
  is numerical hygiene, not a change of numerics: the real block is **bit-identical** (one multiply by a
  host mask, which under jit is a folded constant), so every result that honours the equivalence
  contract's "padding is don't-care" is unchanged, and a new consumer -- the CG state included -- no
  longer has to remember to mask on entry. It breaks only code that read the padding. The bare
  `utv_*_transpose_from_sweep` seam halves stay bare by contract; their callers, the kinds, re-mask.
- **The default Newton reference is the zero tensor of the level's ranks, on the manifold geometries.**
  `newton_cg`'s relative stop `‖g‖ ≤ gtol_rel·‖g₀‖` and its CG forcing term, and `gradient_descent`'s
  stop, used the warm-start `‖g(x0)‖` as `‖g₀‖`. After a zero-padded continuation restart that gradient
  is deflated, so the stop was unreachable at the default `gtol_rel` and every level ran to its cap. Now
  `g0_reference=None` means the geometry's own reference: on `MANIFOLD` / `UNIFORM_MANIFOLD` and their
  shared wrappers, the gradient norm at the zero tensor of `x0`'s ranks; on the corewise geometries (where
  the zero cores are a critical point) the warm start, as before. The reference is evaluated at a
  canonical zero-point frame (below), so it is a deterministic convention good to an O(1) factor, the same
  on the ragged and uniform layers, and nested across continuation levels. It moves every manifold stop
  by an O(1) factor relative to 2026.2.0, **zero starts included** (the old reference sat at the start's
  LAPACK completion). Pass `g0_reference='warm_start'` to keep the old reference; the explicit
  `g0norm_newton` / `g0norm_cg` overrides still win. The downstream measurement behind the change
  (T3Polynomial, 30 Newton iterations per continuation level):

  | level | dof | `‖g₀‖` at the zero tensor | warm start / zero |
  |---|---|---|---|
  | 1 | 417 | 10.0 | 1.03 |
  | 3 | 1617 | 14.4 | 0.37 |
  | 5 | 4609 | 17.7 | 0.21 |
  | 7 | 10545 | 22.2 | 0.13 |
  | 9 | 17917 | 27.3 | 0.053 |

  The zero reference also tightens the CG forcing term (~2× more CG work per step at high rank there, for
  the same accuracy). Continuation loops that carried `stats['history'][0]['gnorm']` into `g0norm_newton`
  can drop that bookkeeping.
- **`newton_cg`'s `stats['newton']` is replaced** by `stats['newton_steps']` (the same count: CG solves),
  `stats['newton_accepted']` (steps taken) and `stats['stop_reason']` (`'gradient'` / `'max_newton'` /
  `'line_search'`). It is removed, not aliased: it overloaded "solves" and "steps".
- **A non-positive `g0norm_newton` / `g0norm_cg` raises `ValueError`.** `0.0` used to crash with a
  `ZeroDivisionError` in the forcing term.
- **Display only, numerics unchanged:** the `verbose=True` converged line names its reference --
  `converged (‖g‖ ≤ gtol·‖g₀‖, g₀ = zero tensor)` / `warm start` / `override`. A hand-built `NewtonInfo`
  row (no reference kind) prints exactly as before.
- **`fitting.GaussNewtonModel` is renamed `fitting.LocalModel`** (also `t3toolbox.LocalModel`), mirroring
  the backend class it wraps, `backend.optimizers.LocalModel`. It is **not aliased** -- the old name is
  gone, as `UniformGaussNewtonModel` went in 2026.2.0 -- so a breaking release breaks loudly once rather
  than carrying two names. The six `*_model` factories keep their names, and the rename leaves the class's
  methods and fields as they were (the exact mode adds to them, under Added).
- **The private `optimizers._setup` is removed**, replaced by the public `optimizers.least_squares_problem`
  (below), which takes the same arguments (plus `hessian`, the mode the problem will be solved in) and
  returns a named `LeastSquaresSetup` rather than a 3-tuple. Not aliased.
- **`optimizers.newton_cg` no longer takes `**kwargs`.** Its options are explicit keywords --
  `max_newton`, `gtol_rel`, `cg_maxiter`, `c_armijo`, `on_line_search_failure`, `g0norm_newton`,
  `g0norm_cg` and `cg_forcing_power`, beside the existing ones, and the exact-Hessian options `hessian`,
  `curvature_damping`, `step_control` and `tr_radius0` (below) -- with the backend's defaults, so every
  option is in its signature and its docs. A misspelt option raises `TypeError` at the call, before the
  problem is built (it used to be forwarded, and the backend call rejected it after the setup work).
  Every valid call is unchanged. (`gradient_descent` / `mc_sgd` / `adam` still forward `**kwargs`.)
- **Every production jet kernel is now a recurrence or an order scan; the binomial tensor is test-only**
  (`backend.sampling_derivatives`; performance and memory, numerics equal to ~1e-15 relative -- not bitwise,
  the summation order changed). 2026.1.0's move off the dense `trs[t,r,s] = C(t,r)·[r+s=t]` contraction had
  stopped partway: the apply/entries-derivative forward and transpose, the probe transpose's `dx̃i` combine
  and the TT gradient assembly (on every branch -- the W-chunked form only wrapped the dense kernel) still
  contracted `trs` or a slice of it, and the order scans applied `(order+1)²` weight slices. Now an affine
  factor gives a banded two-term recurrence, and a full convolution is an order scan whose binomial weights
  come from a shift recurrence on a carried jet. Measured (CPU, float32 `memory_analysis`, per W-row, uniform,
  unchunked): the TT assembly 4-8× less scratch, `dx̃i` 13-15×, the exact Hessian's two-hole TT assembly
  5-14×; the jitted uniform probe-derivative transpose 4.3× and the two-hole 6.4× less temporary (d=4,
  r=16, order 3, W=2000). Time: the ragged-numpy GN probe-derivative transpose 2.3× faster there, the
  apply-derivative transpose 2×; small jitted uniform problems pay the sequential scans' latency (~1.6× at
  d=5, r=8, W=300). Breaking, for backend callers of the jet kernels:
  - **`trs` arguments are gone.** `compute_mu_jets` / `compute_nu_jets` take `order` in its place (their
    input jets stop at order 1); `compute_eta_jets`, `compute_{sigma,tau,deta}_jets`,
    `compute_{tau,sigma}_tilde_jets`, `compute_dxi_tilde_jets`, `compute_sigma_hat_jets`,
    `assemble_tt_variation_jets`, `compute_{tau,sigma}_tilde_tangent_jets`, `compute_dxi_tilde_tangent_jets`
    and `assemble_tt_tangent_variation_jets` drop it (the order is read off their jets).
  - **`compute_dxi_tilde_jets` / `compute_dxi_tilde_tangent_jets` return only the two rows that are read**:
    `(2,)+W+K+C+(nOi,)` (rows `u = 0, 1`; row 1 is zero at order 0), not `(order+1,)+...`;
    `assemble_tucker_variation_jets` / `assemble_tucker_tangent_variation_jets` take that two-row jet.
  - **Removed** (not aliased): `binomial_combine_tensor` and the dense `*_trs` twins
    (`compute_{mu,nu,eta}_jets_trs`, `compute_{sigma,tau,deta}_jets_trs`, `compute_{tau,sigma}_tilde_jets_trs`,
    `assemble_tt_variation_jets_trs`) -- they are the test suite's reference forms now (`tests/_jet_trs_twins.py`);
    `assemble_tucker_variation_jets_trs` (which contracted no `trs`) is private -- pass `chunk_size=None` for
    the unchunked assembly.
  - **`estimate_chunk_size` / `max_chunk_size_within` measure the running program**: the per-row cost is
    now the slope in `chunk_size` of the compiled chunked transpose (and, for `hessian='exact'`, of the
    two-hole) -- the assemblies plus the chunk's slices of their input jets -- where it was one dense kernel
    lowered alone. The answers grow with the smaller cost: rank 128, order 5, a 16 MiB budget gives 195 (was
    2); the balanced policy now admits the whole `W` in the usual case (the assemblies cost less per row
    than the resident jets), so `chunk_size='auto'` fits chunk less or not at all. The measured cost is the
    jax release's own (figures above: current jax): jaxlib 0.4.30 packs the chunk loop into memory the
    earlier stages free, about one jet slice per row (~5-15× less), so its budget chunks are that much larger.

- **`SamplingKind` has a sixth operation, `corewise_hessian`**, needed only for the exact Hessian
  (`hessian='exact'`). A kind written against the five-operation recipe keeps working with the default
  Gauss-Newton model; `Problem.local_model(..., hessian='exact')` rejects it up front with a structural
  `ValueError` naming the missing operation (the base method carries the stdlib `abc.abstractmethod`
  marker, which the check reads; the class is not an `ABC`, so instantiation is unaffected).
- **`NewtonInfo.cg_truncated` excludes trust-region boundary stops**: it is now `(not cg_converged) and (not
  cg_ok) and (not cg_boundary)` -- a Steihaug stop on nonpositive curvature is a boundary step, not a
  truncation -- and `cg_converged` is `False` on a boundary stop, so a row's CG outcome is exactly one of
  converged / boundary / truncated / the cap. With the default line search there is no boundary, so both are
  unchanged there.
- **Trust-region rows have no line-search fields**: with `step_control='trust_region'` (below) a
  `NewtonInfo` row has `ls_steps = ls_failed = None` and `alpha = 1.0` (filter history rows on
  `step_control`); `newton_accepted` also excludes rejected trust-region trials.
- **The backend's private CG `_cg_solve` returns a 5-tuple** `(p, iters, resid², ok, boundary)` and takes a
  trailing `radius` (default `inf`: the plain CG, bit for bit). Internal, listed for scripts that called it.
- **Display only, numerics unchanged:** a `verbose=True` / `make_newton_display` run's last row names its stop
  when that is not the gradient test (`| stop: max_newton`, `line_search`, `objective_floor`,
  `trust_region_floor`); the Gauss-Newton line-search rows are otherwise byte-identical.

### Fixed

- **The safe-mode gauge check no longer rejects a near-cancelling difference of gauged tangents.**
  `MANIFOLD.norm` / `inner` (and the uniform twins) measured `a - b` against its own norm, so the
  finite-difference check of a Hessian, `MANIFOLD.norm(H_fd - hessian(p))` with the two agreeing to `1e-8`,
  raised on each operand's `1e-16` roundoff gauge violation. A tangent built by `+` / `-` / scalar `*` /
  `stack_tangents` is now measured against its operands' scale (`max(‖a‖, ‖b‖)`, `|c|·‖a‖`, per stack
  element, recursively) as a floor under its own norm, so the check is never stricter than before; a tangent
  computed directly (a projection, a gradient) keeps the self-relative check, and one ungauged in its own
  right still fails when checked itself. The scale is recorded only in safe mode, eagerly, and is not jax
  pytree state, so a jit round-trip measures self-relative again. Backend: `tv_gauge_residual` /
  `utv_gauge_residual` take `scale=` (default `None`: unchanged, bitwise). `docs/numerical_contracts.md`.
- **Jet outputs keep their inputs' dtype.** A float32 jet pipeline (`*_derivatives` forwards, transposes and
  the two-hole) now stays float32 under `jax_enable_x64` and on numpy alike. The sweeps' scan inits (the
  order-0 ones, the zero boundary jets, the adjoint zeros, the inner-scan accumulators, the entries one-hots)
  were built in the default dtype and promoted every float32 jet to float64 -- 2026.2.0 did the same. The
  plain `probe` / `apply` / `entries` ops still build default-dtype inits and still promote (recorded for a
  follow-up).
- **Uniform tangent stacks reach the local model.** A uniform kind's `forward` (hence a Gauss-Newton or exact
  `hvp`), the uniform geometries' `project` and `inner`, and the uniform `fitting.LocalModel`'s
  actions took a `K`-stacked trial tangent's coordinates (a stack of tangents at one frame) with the frame's
  `C`-stacked masks and failed in a reshape; the masks are now broadcast over `K`, as the ragged layer and the
  uniform transposes already did. The uniform geometries' `inner` keeps the whole `K + C` stack per element, as
  the ragged twin (it kept only `|C|` leading axes -- `K`'s, on a stacked pair -- unreachable until now).
- **A uniform `fitting.LocalModel` keeps its static structure out of the traced leaves.** Its jax flatten
  traced the frame sweep's mode sizes (Python ints) as leaves, so under `jit` they became tracers; it now
  partitions them into the treedef like the backend `LocalModel` (`backend.common.partition_static`).

### Added

- **`g0_reference`** on `newton_cg` and `gradient_descent` (frontend and backend): `None` (the geometry's
  own), `'zero'` or `'warm_start'`, case-insensitive. An explicit `'zero'` on a geometry with no zero-point
  frame (a corewise one) is a `ValueError`; a vanishing zero reference (e.g. all-zero data) falls back to
  the warm start with a `RuntimeWarning` by default and raises when `'zero'` was asked for explicitly.
- **The canonical zero-point frame**: `backend.fv_operations.fv_zero_point_frame(frame, groups, seed)` and
  its uniform twin `backend.ufv_operations.ufv_zero_point_frame`, with the fixed seed
  `fv_operations.ZERO_FRAME_SEED = 0`. Seeded Haar Tucker factors (one per sharing group, from a stream
  keyed by the group's first mode) and identity-prefix TT cores: rank-independent, nested, and the same
  real blocks on both layers. A Haar completion because a coordinate-aligned one gives a reference of
  exactly 0 on data with no energy on its leading coordinates.
- **`Geometry.zero_point_frame(frame)`**, an **optional**, duck-typed member: the four backend geometries
  define it (`None` on the corewise ones). A custom geometry without it keeps working, with the warm-start
  reference as its default.
- **`NewtonInfo` fields** `g0_reference` (the Newton-stop reference: `'zero'` / `'warm_start'` /
  `'override'`), `cg_reference` (the CG-forcing one), `step_control` (`'line_search'`) and `stop_reason`
  (on the last row), all defaulting to `None`; they ride in `stats['history']`.
- **`optimizers.least_squares_problem(geometry, kind, sample, data, x0, order, weight, regularizer,
  chunk_size, batch, hessian)`**, the frontend twin of `backend.optimizers.least_squares_problem`: it builds the
  backend `Problem` from the frontend vocabulary (a geometry singleton or `SharedGeometry`, a kind name,
  `order` / `weight`, a ragged or uniform `x0`) -- the constructor the four optimizers use, now public
  for scripts that build local models or drive a backend optimizer directly. `hessian` (`'gauss_newton'`
  default, or `'exact'`) is the mode the problem will be solved in: it does not fix the mode, it sizes
  `chunk_size='auto'` for the exact Hessian's two-hole too (`newton_cg` passes its own). It returns
  **`optimizers.LeastSquaresSetup`**, a frozen dataclass with `.problem`, `.x0_cores` (the backend
  starting cores, reduced to minimal ranks where the optimizers reduce them) and `.rewrap` (backend cores
  -> the frontend tensor). Module-level only, not re-exported at the root.
- **The lift from frame coordinates to raw core variations** (backend; a building block of the exact
  Hessian): `backend.linalg.filtered_reciprocal` and
  `damped_pseudo_inverse` (the clipped, optionally Tikhonov-damped reciprocal of a spectrum, and the
  pseudo-inverse built on it); `backend.fv_operations.fv_center_factors` (a frame's centers `H_i`, Tucker
  factors `S_iᵀ` and bond factors `T_i`, recomputed by zipper), `LiftFactors` and
  `fv_lift_factors(frame, curvature_damping)` (damping relative to `‖X‖_HS`);
  `backend.tv_operations.tv_corewise_lift` / `tv_corewise_lift_transpose`, exact at zero damping on every
  numerically full-rank orthogonal frame, minimal or not (at a rank-deficient point the clipped directions
  drop out). `fv_shared_frame_data` now takes its centers from
  `fv_center_factors` (bit-identical).
- **The inner product's derivatives in the cores** (backend; the exact Hessian's T3-to-T3 term):
  `backend.t3_linalg.t3_inner_product_corewise_transpose` (the gradient of `θ ↦ ⟨z, Φ(θ)⟩`),
  `t3_precompute_inner_product_sweep` and `t3_inner_product_corewise_hessian[_from_sweep]` (its Hessian
  applied to a stack of directions, by a two-hole sweep), and the tangent zippers
  `backend.tt_operations.tt_zipper_tangent_left_to_right` / `_right_to_left`.
  `tv_project_t3_onto_tangent_space` now gauges that shared kernel (bit-identical).
- **The probe-derivative transpose's derivative in the cores** (backend; the exact Hessian's sampling term
  for `probe_derivatives`):
  `backend.sampling_derivatives.t3_probe_corewise_derivatives_hessian` and its `_from_sweep` variant -- the
  core-coordinate Hessian of `θ ↦ ⟨z̃, S(Φ(θ))⟩` applied to a stack of directions (`S` the probe-derivative
  jets, `z̃` fixed residual jets, `ω²` already applied), the derivative of
  `t3_probe_corewise_derivatives_transpose`, by a hand-written two-hole sweep; exact and symmetric, ragged or
  on uniform supercores, its assemblies W-chunked by `chunk_size` like the transpose's. Its pieces are public
  beside the Gauss-Newton kernels they differentiate: `compute_deta_tilde_tangent_jets`,
  `compute_tau_tilde_tangent_jets` / `compute_sigma_tilde_tangent_jets` (the dotted adjoint sweeps),
  `compute_dxi_tilde_tangent_jets`, `assemble_tucker_tangent_variation_jets` and
  `assemble_tt_tangent_variation_jets` (factored by the product rule). `estimate_chunk_size` and
  `max_chunk_size_within` take the solver's mode vocabulary, `hessian='gauss_newton'` (the default, the
  transpose-only answer) or `'exact'` (case-insensitive; anything else raises `ValueError`); `'exact'` sizes the
  chunk so that the two-hole, whose chunk costs ~1.3-2.2× the transpose's per row, also fits.
- **The exact Riemannian Hessian in the backend local model** (driven by the backend `newton_cg`'s
  `hessian='exact'`, below; the frontend `fitting.LocalModel` twin is further below):
  `backend.optimizers.Problem.local_model(x, sample, data,
  hessian='gauss_newton' | 'exact', curvature_damping=None)`. An exact model's `hvp` is
  `Π[𝒥ᵀω²𝒥Πp + Lᵀ(K_{Aᵀω²r} − K_{grad f})LΠp] (+ H_R p)` -- the Gauss-Newton term plus the Weingarten
  curvature, through the lift `L`, the kind's `corewise_hessian` and the inner product's two-hole at the misfit
  gradient -- equal to the second derivative of the objective along the T3-SVD retraction; on a corewise
  geometry it is the plain Hessian of `f ∘ Φ`. New on `backend.optimizers.LocalModel`: `curvature` (a
  `CurvatureState`: the model's one misfit gradient and the geometry's per-frame curvature data, pytree leaves,
  so a rebuilt model or a new damping does not recompile), the derived `is_exact`, `curvature_hvp`,
  `quadratic` and `curvature_quadratic`; `gradient`, `gn_quadratic`, `jacobian` and `retract` are unchanged
  (an exact model's gradient is the Gauss-Newton one, bit for bit). The geometries gain the four exact-mode
  members `curvature_precompute` / `lift` / `lift_transpose` / `tangential_curvature` (optional for a custom
  geometry: only `hessian='exact'` checks for them), with `backend.geometry.ManifoldCurvatureData` and
  `DEFAULT_CURVATURE_DAMPING = 1e-6` (relative to `‖X‖_HS`; `curvature_damping` raises with the Gauss-Newton
  model and on a corewise geometry, where nothing would read it). `ProbeDerivativesKind` implements
  `corewise_hessian`; the other kinds follow (the shared manifold's tied lift and the uniform layer: below).
- **Exact Newton with a trust region in the backend `newton_cg`**: `backend.optimizers.newton_cg(...,
  hessian='gauss_newton' | 'exact', curvature_damping=None, step_control=None, tr_radius0=None)` (explicit
  keywords, the same on the frontend `optimizers.newton_cg`, below). `hessian='exact'` solves with the exact Riemannian
  Hessian above -- the superlinear Newton tail where Gauss-Newton is linear on a nonzero-residual fit
  (q-quadratic with `cg_forcing_power=1.0`; the default stays `0.5` in both modes) -- and `curvature_damping`
  (exact only; a value with Gauss-Newton raises) sets the lift's damping relative to `‖X‖_HS`.
  `step_control` is `'line_search'` (Armijo along the retraction; the Gauss-Newton default, unchanged bit for
  bit) or `'trust_region'` (the exact default; either combines with either Hessian): Steihaug-Toint CG in
  `‖p‖ ≤ Δ` -- it steps to the boundary on nonpositive curvature or when it would leave the region -- then one
  trial `retract(p)`, accepted iff `ρ = Δf / (gᵀp + ½pᵀHp) > 0.1`; `Δ ← ¼‖p‖` when `ρ < 0.25`, `Δ ← min(2Δ,
  10⁴Δ0)` when `ρ > 0.75` on a boundary step; `Δ0 = tr_radius0`, by default the Gauss-Newton Cauchy length
  `‖g‖³/⟨g, H_GN g⟩` (`tr_radius0` with the line search raises, as does `on_line_search_failure='accept'`
  with the trust region). A rejected trial keeps `x` and its local model and counts toward `max_newton`. Two
  new `stop_reason` values, checked only after a rejection: **`'objective_floor'`** -- the predicted change is
  below the objective's floating-point resolution, `|pred| ≤ 10·ε·(‖ω⊙y‖·‖ω⊙r‖ + |f|)` with `ε` the residual
  dtype's machine epsilon: the attainable accuracy is reached, **not a failure** -- and
  **`'trust_region_floor'`** (`Δ < 10⁻¹²·Δ0`). New `NewtonInfo` fields, all defaulting to `None`: `hessian`,
  `tr_radius`, `tr_accepted`, `cg_boundary`, `pKp` (the curvature term's part of `pHp`), `lift_sigma_min`,
  `lift_eps`; `step_control` gains `'trust_region'`. The display's `step` segment reads `TR Δ … ✓/✗` on a
  trust-region row, CG's symbol `∂` on a boundary stop, and an exact row appends `κ σ ε` (`κ = pKp/(pHp −
  pKp)`, the smallest lift singular value, the lift damping). A curvature-data object may carry the
  diagnostics `sigma_min` / `eps` (`backend.geometry.ManifoldCurvatureData` gains the `eps` property).
- **The exact Hessian on the frontend.** The six `fitting.*_model` factories take `hessian='gauss_newton' |
  'exact'` (case-insensitive) and `curvature_damping=None`: `hessian='exact'` builds the exact-mode
  `fitting.LocalModel`, whose new `curvature` field (the backend `CurvatureState`, a jax leaf) holds the model's
  one misfit transpose and the geometry's curvature data. The mode is derived, not stored: the read-only
  property `is_exact`. New methods, in both modes: `hessian(p)` and `quadratic(p)` (the mode's Hessian action and
  form -- `gn_hessian` / `gn_quadratic` on a Gauss-Newton model, plus the curvature term on an exact one) and
  `retract(p)` (`geometry.retract(p)` at the model's frame, with the shared manifold's stored companion -- the
  twin of `backend.optimizers.LocalModel.retract`); on an exact model only (a Gauss-Newton one raises),
  `curvature_hessian(p)` / `curvature_quadratic(p)`, the curvature term alone. `evaluate(p)` is mode-aware (the
  Gauss-Newton value is unchanged) and an exact model's `gradient` is the Gauss-Newton one, bit for bit. The
  rules are the backend's: `curvature_damping` with Gauss-Newton or on a corewise geometry raises, and a kind
  without `corewise_hessian` (every kind but `probe_derivatives` so far) raises a structural error at
  construction. The frontend `optimizers.newton_cg` takes
  `hessian`, `curvature_damping`, `step_control` and `tr_radius0` with the backend's defaults, and passes
  `hessian` to `least_squares_problem` so `chunk_size='auto'` is sized for the exact mode. The frontend run is
  the backend run, bit for bit.
- **The exact Hessian on the shared manifold** (`shared_manifold(...)` / `ManifoldGeometryOps` with `groups`;
  ragged): `hessian='exact'` now runs there too -- `newton_cg`, the `*_model` factories and
  `Problem.local_model`. The shared set is the tied
  parametrization `Φ(Tθ)` (`T` copies each group's factor to its modes), so the curvature term goes through
  a **tied lift**: each sharing group's coordinates lift to ONE raw factor direction,
  `U̇_g = M_g^{+ε}[V_i₁; …]`, solved against the same stacked-`S` SVD the tied projection uses (the group
  spectrum `s_g`, damped by the same `curvature_damping`), and its transpose sums the group's copies; the
  normal part is taken against the shared (tied) misfit gradient, and `hvp`'s outer projection ties the
  result. At a zero-padded shared restart the new `s_g` levels filter to exactly 0 -- the gated tied
  channel gets no curvature, the TT variations carry the escape -- and `lift_sigma_min` reads each
  group's `s_g` in place of its modes' own spectra. The shared retraction is second order as measured
  (O(h²) finite differences), not proven. New: `backend.sharing.fv_tied_lift_factors(frame,
  curvature_damping, shared_data)`, the lift factors of a shared point; `fv_operations.LiftFactors` gains
  `shared_data` / `tied_filters` (default `None` / `()`, pytree leaves -- the partition stays static), with
  which `tv_corewise_lift` / `tv_corewise_lift_transpose` tie the group slots. The tied projection and
  retraction now share the filtered solve with the lift (bit-identical). The shared corewise geometry's
  exact Hessian (the per-group mean of the plain Hessian) was already live.
- **The exact Hessian on the uniform layer** (`UNIFORM_MANIFOLD` / `UNIFORM_COREWISE`, unshared; ragged and
  uniform now both run `newton_cg(hessian='exact')`, the `*_model` factories' `hessian='exact'` and
  `Problem.local_model(hessian='exact')` for `probe_derivatives`, `use_jit` included). Per stack element the
  uniform exact model is the ragged one (the equivalence contract): every padded slot of a direction is masked
  before the second derivative sees it -- a direction's padding multiplies into real slots there, so the
  masks are load-bearing, not hygiene -- and every output is re-masked. From a numerically full-rank start a
  uniform exact run is the ragged run (counts and objective row by row); from a zero-padded restart the two
  layers' frames legitimately differ (`docs/uniform_equivalence_contract.md`) and each converges on its own.
  New backend functions, each the uniform twin of the ragged one it names:
  `backend.ufv_operations.ufv_lift_factors` (the lift factors on supercores: one batched damped pseudo-inverse
  per family, each block clipped at its REAL dimensions, the padded singular values exactly 0, the last TT
  slot's pass-through as one more bond slot), `backend.utv_operations.utv_corewise_lift` /
  `utv_corewise_lift_transpose`, `backend.ut3_linalg.ut3_inner_product_corewise_transpose` /
  `ut3_precompute_inner_product_sweep` / `ut3_inner_product_corewise_hessian[_from_sweep]` (the T3-to-T3
  term; the ragged `t3_precompute_inner_product_sweep` / `t3_inner_product_corewise_hessian_from_sweep` are now
  polymorphic over supercores, `d`-prefixed contractions) and
  `backend.ut3_sampling.ut3_probe_corewise_derivatives_hessian[_from_sweep]` (theta and the direction masked
  with theta's masks, the output re-masked). `UniformProbeDerivativesKind.corewise_hessian` replaces its interim
  stub (it masks with the masks of the FRAME it is handed, a strict subset of the point's at TT slack), and the
  uniform geometries gain the four exact-mode members (the curvature data's `sigma_min` reads real slots only).
  A shared uniform geometry raises `NotImplementedError` for `hessian='exact'` until its tied lift lands; the
  other kinds' uniform twins keep their stub.
- **`backend.geometry.UniformManifoldGeometryOps.from_frame(frame_data, plain_masks, sharing)`**: the uniform
  manifold geometry from a frame already in hand -- its variation masks are the frame's own, gauge-shifted --
  the same geometry as `from_point` without re-orthogonalizing. The uniform `fitting.LocalModel` builds its
  backend geometry this way (one orthogonalization fewer per regularized or exact model).

## [2026.2.0] — 2026-08-25

### Changed — breaking

- **The optimization layer's axis objects are classes, not records of closures.** A geometry and a
  sampling kind are now frozen dataclasses whose *parameters are fields*; their behaviour is methods.
  This is what makes them compare and hash **by value**, which matters because both ride as jax
  `aux_data` and therefore form part of the compilation cache key. Rationale, measurements and the
  rejected alternatives: [`contributor/parameters_not_closures.md`](contributor/parameters_not_closures.md).
  - **New `backend/geometry.py`** with `ManifoldGeometryOps` / `CorewiseGeometryOps` and their
    `Uniform*` twins (the uniform ones carry the fixed rank they were built at; use `from_point`).
    Removed: `backend.optimizers.GeometryOps`, `MANIFOLD_OPS`, `COREWISE_OPS`, `shared_geometry_ops`,
    and `backend.uniform_fitting.uniform_{manifold,corewise,geometry}_ops`.
  - **Sharing is a `groups` field**, not a wrapper — one class and one code path for shared and
    unshared. The frontend `shared_manifold` / `shared_corewise` constructors are unchanged.
  - **`SamplingKind` is a class hierarchy** (`ApplyKind`, `EntriesKind`, `ProbeKind`, the three
    `*DerivativesKind`, and their `Uniform*` twins). The `*_kind` constructor spellings still work.
    Removed: the `identity` field and its `__eq__`/`__hash__`, and the six uniform builder functions
    `uniform_{apply,entries,probe}_kind` / `uniform_{apply,entries,probe}_derivatives_kind` (use
    `Uniform*Kind.from_point(x0_data, **parameters)`, or the unchanged `uniform_sampling_kind` /
    `uniform_derivatives_kind` dispatchers). A custom kind is now a **subclass**; deriving one with
    `dataclasses.replace` no longer type-checks, and that is the point — see below. `ScalarOutputKind`
    and `ProbeOutputKind` are public, so a new operator can inherit the `‖·‖²` reductions.
  - **`fitting.UniformGaussNewtonModel` is removed.** One `GaussNewtonModel` serves both
    representations; the factories still dispatch on `x`. It is deliberately not aliased.
  - `optimizer_display` gates its error table on `kind.has_block_sumsq` rather than on a `block_sumsq`
    field being `None`. `block_sumsq` is optional; the flag is declared by whoever implements it.
  - **A custom backend geometry needs three more members**: `stack_shape(x_cores)`, `base_point(frame)`,
    and a `precompute` that is a *callable* returning `None` rather than `None` itself. The `Geometry`
    protocol in `backend/optimizers.py` lists the full surface.

- **Optimizing a STACKED point raises `NotImplementedError`.** It never worked -- `newton_cg` and
  `gradient_descent` failed on `float()` of a shape-`C` objective, `mc_sgd` on a broadcast, and `adam`
  once its loss logging first fired -- but it failed obscurely, from deep inside the loop. Building the
  local model on a stacked point *is* supported and still works, so rolling your own loop over a stacked
  problem is the route. Stacked optimization is possible future work.

### Fixed

_The items below came out of the 2026-08-22 whole-library review (`dev/review_2026-08-22/`)._

- **The safe-mode trace detector is the committed-array probe alone** (review H6-12). The old primary
  detector, `jax.core.trace_state_clean`, does not exist in jax 0.10 -- every check on jax inputs paid a
  raise/catch before falling back -- and its docstring promised any-transform detection the probe never
  delivered. Behavior under jit is unchanged (checks skip); under `grad`/`vmap` with concrete operands
  the checks run, as they always effectively did -- harmless and now documented. The batching docs also
  record why a stacked `UniformTuckerTensorTrain` cannot be `vmap`-ed at all (masks are static aux; use
  the native `C` stack).

- **The uniform same-frame guard compares real (masked) content only** (review H5-5). Two frames
  identical up to the don't-care padding now count as the SAME frame -- the tangent space depends only
  on the real content -- so `UT3Tangent` arithmetic / `stack_tangents` between a frame and a re-padded
  copy of it works instead of raising "different tangent spaces". `safety.frames_equal` itself stays
  representation-agnostic; its contract is now explicit: callers pass real content, and the uniform
  call sites mask their supercores first. (This also makes `UT3Frame`'s documented "frames_equal on
  the masked supercores" true.)

- **The review's oracle sweeps are permanent tests** (Phase D): `tests/test_oracle_sweep.py` runs the
  O1 sampling/tangent sweep (dense einsum + exact combinatorial jet oracles + adjoint identities,
  ragged and uniform incl. varying-rank stacks) and the O2 fitting sweep (polynomial-interpolation
  oracle vs `GaussNewtonModel`; optimizer trajectories vs hand recomputation, ragged == uniform) from
  verbatim copies of the review scripts in `tests/oracle_sweeps/`. Two tiers: always-on (full O1 + full
  O2 models + a pairwise-covering optimizer subset) and `T3TOOLBOX_SLOW_TESTS=1` for the full optimizer
  matrix -- a required release-gate step, so nothing the review matrix covered stops being enforced.

- **Small ergonomics from the review's E list** (R9-10, R7-11, R3-7): ``chunk_size`` now threads
  through both frontends' ``probe_corewise_derivatives_transpose`` (and the uniform backend wrapper) --
  it existed only on the ragged backend and the tangent-level transposes; ``chunk_size='auto'``
  resolution reads supercore shapes without pulling the arrays to host, and the fallback chunk size is
  the single ``backend.sampling_derivatives.DEFAULT_CHUNK_SIZE`` constant; the minimal-rank recurrence
  in ``backend.ranks`` uses the dispatched backend throughout (three stray ``np.minimum`` calls made it
  jit-hostile).

- **jit cache-key hygiene on the value-hashed aux family** (reviews H1-5, H1-6, R2-7, H2-8, R7-13).
  The mask holders' `hash` and `==` now derive from one content key (they could disagree on dtype and
  silently zip-truncate on length); a jax-array field in a `ValueHashedFields` aux raises a directive
  TypeError (device arrays belong in leaves); the aux objects (mask holders, uniform geometries and
  kinds) store defensive **read-only copies** of their mask arrays, so mutating a caller's mask in
  place can no longer stale a cached cache key -- in-place writes to stored masks now raise;
  `shared(...)` geometries key their identity (and jit aux) on the **canonical partition**, so label
  spellings of the same partition share one compiled program, and a `SharedGeometry` subclass now
  survives the jit round-trip as itself. The `stack(unstack(x))`-under-jit findings were verified
  already fixed by the review's static-axes `moveaxis` and are pinned by a regression test.

- **Obscure errors on malformed inputs became structural guards** (the review's E-cluster sweep,
  2026-08-24: R2-8, R2-12, R2-13, R3-6, H3-7, R8-6, R8-9, R1-15, R10-6, H3-8, R9-9). Str leaves no
  longer recurse the tree helpers forever; `tree_zip` rejects structure mismatches instead of
  silently truncating; `truncated_svd(min_rank > max_rank)` raises (min used to silently win);
  `t3_inner_product` takes list/tuple mixes and `t3_sum_stack` an ndarray axis; uniform `*` is
  scalar-only with a clear TypeError, uniform `stack` validates the static layout, and uniform
  `minimal_ranks` stays host numpy on jax-backed trains; the ragged tensor-weight ops check
  rank/stack consistency (one-metric broadcast still allowed); `COREWISE.retract` rejects a
  detectable manifold-frame tangent; `SharedGeometry` names a wrong-layer point/frame. One
  regression test per finding.

- **The uniform frame at a numerically rank-deficient point loses no tangent directions** (review
  S1b -- the last open review item). Every kept-basis SVD in the uniform orthogonalization and
  T3-SVD sweeps now goes through the new mask-aware `backend.linalg.pad_safe_svd`, so the
  `sigma ~= 0` completion columns stay OFF the padding instead of landing in padded slots that the
  masks then erase. Before the fix, a zero-padded `resize` warm start -- the rank-continuation
  restart -- produced a uniform frame whose masked real block was genuinely rank-deficient (60/60
  trials), silently losing exactly the tangent directions continuation needs to grow into. Affects
  `UT3Frame.from_ut3` / `ut3_orthogonal_representations`, the per-core `ut3_*_orthogonalize_*` ops,
  and `ut3svd` / `ut3_rank_adjustment_sweep` / `ut3svd_orthogonal_representations` (unshared and
  SF-T3 shared). Regression tests: `TestPadSafeFrame` in `tests/test_uniform_frame_variations_format.py`.

- **Uniform `+`, `-`, `squash_tails` and `sum_stack` read padding garbage.** `ut3_squash_tails` summed
  the leading/trailing TT bond over *every* slot, including the padded ones the equivalence contract
  declares don't-care -- and a corewise gradient step (`adam(UNIFORM_COREWISE, …)`, `UNIFORM_COREWISE.retract`)
  leaves exactly those slots nonzero, so `x_fit + y` was silently wrong while `to_dense`/`norm` (which mask
  on entry) were right. It now masks on entry.
- **`t3m` with boundary TT ranks `r0, rd != 1`.** `inplace_fused` (the default) never SVD'd the last bond,
  so an unsquashed trailing bond survived as `rd_x · rd_y` and escaped `max_tt_ranks`; `swap` contracted the
  two operands' trailing bonds against each other (silently wrong when equal, a raw einsum error when not)
  and ignored a per-position `max_tt_ranks` given as a numpy array. Every `t3m` method now canonicalizes
  both operands' boundary bonds to 1 on entry -- the rule, and why the exact `*`/`t3_mult` deliberately do
  not, is in [`t3m_methods.md`](t3m_methods.md).
- **`entries_ambient_transpose` returned zero factors for a negative index.** Its one-hots were built with
  `arange(N) == index`, which matches nothing for `-1`; the forward `entries` and the tangent/corewise
  transposes wrap numpy-style, so the adjoint identity silently failed. All four now agree (`eye(N)[index]`).
  Documented alongside: an out-of-range index raises under numpy but clamps under jax (gather semantics).
- **The documented singular-value-metric recipe applied the σ's in the wrong gauge.** `docs/weighting.md`
  paired `T3FrameWeights.from_t3weights(T3Weights.from_t3svd(x))` with a frame from
  `t3_orthogonal_representations(x)`, which re-SVDs the already-orthonormal Tucker factors of a T3-SVD
  result -- a degenerate spectrum, so the frame's Tucker basis comes back rotated by an arbitrary orthogonal
  matrix relative to the singular basis, and per-coordinate σ-weights then weight the wrong directions
  (unstacked too; the planning measurement said "stacked only"). New **`t3svd_orthogonal_representations(x, **t3svd_kwargs)`**
  (+ the `ut3svd_` twin, both exported at the root) returns `(frame, variations, T3Weights)` from ONE SVD
  with the frame built `already_left_orthogonal=True`, so its Tucker basis is the singular basis; the recipe
  and the `from_t3svd` docstring now say so.
- **The shared-factor companion was wrong on a frame converted between layers.** `fv_shared_frame_data`
  recomputed the centers by re-running the SVD sweep that built the frame, which reproduces the
  construction's sign choices only when it is the same SVD on the same arrays; on a `UT3Frame.to_t3frame()`
  leaf -- the route `UniformManifoldGeometry.project_ambient` recommends for dense gradients -- the tied
  projection was silently 30% off (the group spectrum, being gauge-invariant, looked right). The centers
  are now the zipper of the stored chains (`H_i = L_i Z_{i+1}`, no SVD), exact and gauge-consistent for a
  frame built by either layer; the uniform twin masks first, and the "frames packed from ragged are not
  guaranteed" caveat is gone.
- **`jax.grad` through uniform `norm()` / `inner()` / `ut3_weighted_norm` / `ut3_weighted_inner` was all-NaN on
  any train with rank slack** -- the default orthogonalized path differentiated through the SVD, whose JVP
  has `1/(σᵢ²−σⱼ²)` terms and a padded train has several exactly-zero σ's. New backend twins
  `ut3_norm` / `ut3_inner` (closing a ledger item) keep the precise orthogonalized VALUE and carry a
  `custom_jvp` with the exact multilinear derivative (`2⟨T, dT⟩`, via the zipper with the orthogonalized
  side held fixed -- no SVD in the derivative path); the frontend and the weighted ops route through them.
- **`backend.optimizers.Problem.objective(x, data=…)` / `.local_model(x, data=…)` ignored `data`** when
  `sample` was omitted, silently scoring the training data; `sample=` alone crashed with a bare `TypeError`.
  The pair now goes together or not at all (a structural error otherwise).
- **The regularized objective was wrong at a raw point on the ragged manifold.** `ManifoldGeometryOps.point_norm_sq`
  read `‖last TT core‖²`, exact only for a left-orthogonal point (measured 3 where the truth was 1400 on a
  `randn` point). Backend-only: the optimizers evaluate the regularizer through the frame or at retraction
  outputs, so no optimizer result or `stats` row was affected. It now left-orthogonalizes first (no `W`
  factor, negligible beside the misfit it sits next to), matching the uniform twin.
- **`UT3Variations.sum_stack(axis=-1)` / `UT3Tangent.sum_tangents(axis=-1)` summed over the MODE axis** (the
  raw `1 + axis` sent `-1` to array axis 0), silently when `K == d` and with an obscure shape error
  otherwise. Negative axes now count from the last stack axis, as in numpy and the ragged twin; out of
  range raises.
- **A structural tangent mismatch was caught only by the numerical same-frame guard**, which is skipped
  under `safety.unsafe()` / a jax trace -- so `a + b` for tangents at frames of different rank structure
  broadcast silently there. The core-shape comparison now runs unconditionally (structural problems always
  error), in `T3Tangent` and the fitting model's trial-tangent guard.
- **30 structural checks were written as `assert`** (`x * ndarray` shape, stack-axis ranges, core-tuple
  lengths, `corewise` operand types, `dense_probe` vector shapes, …): a bare `AssertionError` normally, and
  no check at all under `python -O`, where a wrong-shaped `x * ndarray` broadcast silently. All are
  `ValueError` / `TypeError` with a message now.
- **The gauge precondition compared an absolute residual against the relative tolerance.** `gauge_residual`
  was `max|UᵀV|`, which scales with the tangent, while `MANIFOLD.inner`/`norm` threshold it at
  `rtol` (1e-9 numpy, 1e-5 jax): a correctly gauged tangent of norm ≳1e8 failed in numpy, and on jax's
  default float32 one of norm ≳50 -- the documented jax-eager safe-mode use -- while a tiny ungauged
  tangent passed. The residual is now relative (each gram divided by its variation core's norm, per stack
  element), on both layers.
- **The tangent/corewise `apply`/`entries` transposes rejected a bare Python float residual** (the natural
  unstacked case; the ambient twin's own doctest passes `1.7`).

- **`d = 1` on the ragged manifold:** `MANIFOLD.project_ambient(frame, <T3>)`, `project_ambient(…, method='t3svd')`
  and `MANIFOLD.transport` raised an `IndexError` (an empty TT chain has no zipper); `dense_probe` raised on a
  one-mode tensor with a probe stack. A one-mode T3 is a vector; these now degenerate to that case like
  everything else.
- **`UNIFORM_MANIFOLD.transport` / `project_ambient` crashed on a `K`-stacked tangent or gradient** (the
  frame's `C`-stack gauge masks were handed to `K+C` variations and failed in a reshape); the ragged twins
  accepted `K`. The masks are now broadcast over `K` (new `ufv_masking.ufv_variation_masks_over_stack`, the
  same rule the sampling transposes already used).
- **`stack()` was not jittable** (`TuckerTensorTrain.stack`, `T3Frame.stack`, `T3Tangent.stack_tangents`, the
  uniform twins): `moveaxis` was given an `arange` source, which is a tracer under `jit`.
- **`adam` on a ragged manifold with a non-minimal-rank `x0` crashed at step 2** (its moment trees were
  allocated at `x0`'s core shapes; the first retraction drops the redundant rank). A ragged manifold `x0` is now
  reduced to minimal ranks on entry, mirroring the uniform path's `uniform_minimal`; a minimal start is untouched.
- **A slack-padded `x0` crashed every uniform manifold optimizer at step 2.** `UniformTuckerTensorTrain.from_t3(x, n=, r=)`
  (the documented "force a larger pad") passes the minimal-rank gate, but `utv_retract` returned its point at
  the max rank `ut3svd` kept rather than at the frame's padded dims (its docstring promised the latter), so
  the loop-invariant masks no longer fit. The retraction now re-pads (new `backend.ut3_operations.ut3_pad_ranks`).
- **`newton_cg(verbose=True, val_data=…)` without `val_sample` failed deep inside the kind**, and `val_sample`
  alone was silently ignored; both now raise at the entry (`make_newton_display` too).
- **`uniform_least_squares_problem`** accepted any non-`'manifold'` geometry string as corewise (so `'Manifold'`
  silently fit on the wrong geometry), and built a `Problem` for a derivative kind without `order=` that failed
  on first use. Geometry and kind strings are validated (case-insensitively, see below) and `order` is required up front.

- **Structural validation gaps and wrong messages** (each a deep or misleading failure before):
  `check_fv_pair` accepted a frame and variations with different numbers of cores (four ops then returned
  partial sums); `T3Frame.validate`'s Tucker-rank message printed stack dimensions on a stacked frame;
  `entries()` with a wrong-length *list* index raised `AttributeError` instead of the documented
  `ValueError`; `share(rtol=…)` on a stacked T3 failed inside `truncated_svd`; uniform `+` / `inner` on
  operands with different padded widths `N` raised raw numpy errors; the six `GaussNewtonModel` factories
  did no validation of the sample / residual against the point; `set_default_safety(None, None)` was
  accepted and then every check crashed (it now means *unsafe by default*, the one script-level way to
  get it; a lone `None` is rejected); `has_numerically_minimal_ranks()` raised on a stacked train (now a
  per-element bool array like the other checkers).

### Changed

- **`==` and `hash` are now explicit-only on every runtime array-carrying class** (review R1-13
  extended; ruled 2026-08-24). `TuckerTensorTrain`, `T3Weights`, `T3Frame`, `T3Variations`,
  `T3FrameWeights`, `T3Tangent` and their six uniform twins refuse `==` with a TypeError naming the
  named checks (`x == x` stays True via the identity fast path) and are **unhashable** -- the
  equality of a floating-point tensor representation is not a well-posed operator question, so the
  user says which equality they mean: `a.allclose(b[, rtol=, atol=])` for what the objects
  REPRESENT (numerical, per stack element; gauge/padding-invariant for trains; the BASE POINT for
  frames, with `safety.frames_equal` answering the same-FRAME question), `a.corewise_equal(b)` for
  the bitwise representation, `a is b` for identity. Previously `==` raised numpy's ambiguous-truth
  error (`T3Frame`: silently compared identity -- a vestige of the pre-safety-redesign aux_data era,
  per its own history). Value-based hash/eq remains exactly where jit needs it: the aux_data family
  (mask holders, geometries, kinds). Rule + rationale: `backend.common.ExplicitEquality`. The
  pre-existing `allclose` methods (`T3Tangent` / `UT3Tangent` / `UT3Frame`) are unified into the
  family: default `rtol` is now the ambient safety default (same value on numpy float64), the
  reference is the symmetric `max(||a||, ||b||)` (was `||other||`), and the tangent twins now CHECK
  the same-frame precondition they used to assume.

- **Citing T3Toolbox means citing the software, not the T4S paper.** `CITATION.cff`'s
  `preferred-citation` used to redirect citations to the research preprint the library grew out of;
  the library is a general-purpose T3 toolkit and is cited as software. The paper moved to the CFF
  `references` list (cite it when building on its research contributions); README / docs wording
  aligned.

- **The six `utv_*_transpose_from_sweep` hooks now default `sum_over_probes=False`, matching their
  ragged twins** (review H2-4). Each says it "shares" the ragged function, but the uniform defaults
  summed the probe stack `W` while ragged kept it -- so a raw-`.data` user porting ragged split-seam
  code to uniform silently got a different result shape. No shipped path changes (both frontends
  already defaulted `False`; the fitting kinds pass `True` explicitly). Breaking only for raw-backend
  callers that relied on the old uniform default: pass `sum_over_probes=True` explicitly.

- **A failed Armijo line search stops `newton_cg` / `gradient_descent` instead of silently accepting
  an ascent** (review R7-7). Exhausting every halving (40 / 50) used to accept the last trial anyway --
  measured objective *increases* accepted at the precision floor, ~2×40 objective evaluations per
  stagnant iteration -- and the recorded `alpha` was one halving past the step actually taken, so that
  row's `delta_f` / `rho` / `step_rel` described a step never tried. Now: the default
  `on_line_search_failure='stop'` rejects the step and terminates; `'accept'` opts into the old
  take-the-step behavior as a deliberate escape mode (it can bump a stagnant iterate loose);
  `NewtonInfo` / `stats['history']` gain `ls_failed`, `gradient_descent`'s stats gain
  `'line_search_failed'`, and `alpha` now records the last trial actually evaluated.

- **The ragged backend geometries' `inner` / `point_norm_sq` are per-element over the frame stack,
  matching the uniform twins and the frontend** (review H3-5). `ManifoldGeometryOps.inner`,
  `CorewiseGeometryOps.inner` / `point_norm_sq`, and `t3_left_orthogonal_norm_sq` used to collapse the
  stack `C` to a scalar while `UniformManifoldGeometryOps.inner` and `MANIFOLD.inner` returned shape
  `C` -- a silent cross-layer asymmetry for a raw-`.data` user composing a stacked loop. Unstacked
  behavior is unchanged (a 0-d scalar); shipped optimizer paths reject stacked points up front and are
  unaffected.

- **The uniform manifold frame is gauge-equivalent to ragged, no longer bit-identical.** The old
  bit-equality was an accident of both layers feeding the same matrices to the same LAPACK kernel;
  the pad-safe sweep (previous section) computes the same subspaces and singular values through a
  different -- still fully deterministic -- factorization route, so gauge-carrying cores agree with
  ragged only up to the SVD's inherent gauge freedom (column signs, rotations within equal-`sigma`
  clusters, the choice of null-space completion). Every gauge-invariant quantity -- objectives,
  gradient norms, optimizer trajectories, represented tensors -- still matches ragged exactly.
  The contract statement for gauge-carrying operations is now in `docs/uniform_equivalence_contract.md`.

- **jax requested but not installed: run on numpy and warn, never raise.** `TuckerTensorTrain.randn(use_jax=True)`
  / `UniformTuckerTensorTrain.randn(use_jax=True)` used to die with a bare `NameError`, `to_jax()` /
  `load(use_jax=True)` silently returned numpy, and the optimizers' `use_jit=True` raised. The policy is now one
  rule (`backend.common.jax_or_warn`): the request runs on numpy with a one-time `RuntimeWarning` naming the
  `t3toolbox[jax]` extra -- a project developed without jax on one machine and deployed with it on another runs
  unchanged on both. (`use_jit=True` with jax *present* still auto-converts to jax, as in 2026.1.0.)
- **Geometry and kind strings are case-insensitive** on the frontend optimizers and `uniform_least_squares_problem`
  (`'Manifold'`, `'PROBE_DERIVATIVES'`, …).
- **`adam` on a manifold geometry warns**: its per-coordinate moments live in the gauge-dependent core
  coordinates, so its iterates depend on the backend / representation (measured ~1e-2); `COREWISE` is the
  intended geometry, `mc_sgd` / `newton_cg` the gauge-invariant manifold optimizers.
- **A per-mode residual weight may have more rows than modes.** The extra rows are ignored (intended: one
  weight rides through a continuation scheme that adds modes), now on both layers -- the frontend used to
  reject it while the backend truncated silently. Fewer rows than modes is a structural error on both.

### Fixed (continued)

- **`d = 1` on the uniform layer.** The uniform boundary-bond squash duplicated the single core, which took
  down every squashing op (`+`, `-`, `norm`, `inner`, `t3svd`, `rank_adjustment_sweep`, `UT3Frame.from_ut3`,
  `is_left_orthogonal`). A one-mode T3 now degenerates to the vector case on both layers, as intended.
- **A regularized fit of a stacked point silently mis-weighted, and now raises.** The data misfit keeps
  the frame stack `C` (one value per element) while every regularizer scalar collapses it, so
  `objective = misfit + ρ` added the whole-stack regularization total to *each* element -- inflating the
  effective `λ` by about `|C|`, unevenly. The regularizer *gradient* was per-element correct, which is
  what made it easy to miss. No test combined `stack_shape` with a regularizer.

- **A derived sampling kind silently reused its parent's compiled program.** `dataclasses.replace`
  copied the hand-maintained `identity` tuple unchanged, so `dc.replace(APPLY, forward=<other math>)`
  compared equal to `APPLY`; under `jit` it returned `APPLY`'s answer (measured: 115.302888 where eager
  gave 28.825722). Unrepresentable now — a variant is a subclass, hence a distinct type.

### Documentation

- **`rank_adjustment_sweep`'s orthogonality is conditional**, and the docs said so unconditionally: the
  sweep re-SVDs the TT cores and bonds and *preserves* Tucker orthonormality but never creates it, so its
  result is right-/left-orthogonal in the full sense only when the input's Tucker cores were already
  orthonormal (any `t3svd` result) -- the documented precondition, now stated at every site. The one real
  gap behind "compose both directions for guaranteed minimal ranks" -- a Tucker rank above its mode size
  (`n_i > N_i`), which no TT-side step can reduce -- is closed by a Tucker up-SVD at the start of the sweep.

- A sweep of doc ↔ code mismatches found by the 2026-08-22 review: the README / user-guide core diagram
  and frame-core listings (`d` cores, not `d+1`; `up_tucker_cores`), the Tucker-rank tuple, the uniform
  mask form (canonical prefix vs gappy working form), the registered-pytree list (the uniform classes were
  missing), the paper section cited for probing, stale "slice"/"deferred" module docstrings on the uniform
  layer, the `UT3FrameMasks` value-hash comment, `chunking.md`'s `'auto'` default (it is the frontend's),
  `sharing.md`'s "closure state", six `dev/regularization_design.md` paths (now in `dev/archive/`), the
  `t3m_methods.md` boundary-bond sentence, the `weighting.md` recipes and "same operations" claim, the
  `GeometryOps` name in the project guide, `backend.geometry` and `corewise` in the API reference, and the
  `save`/`load` doctests, which now write into a temporary directory instead of the working directory;
  and, found by the same review: `truncated_svd`'s documented truncation rule (it is the tail-Frobenius
  TT-SVD rule, not a per-singular-value cut), the `Geometry` protocol's member list, the `shared(…).frame`
  contract row and the rank-deficient-frame retraction caveat in `numerical_contracts.md`, the corewise
  transpose's description in `transposes.md` (a pullback, not a projection), the λ-vs-`draw` scaling
  note, the chunking memory caveats (two-chunk and kept-path regimes), the `resize` call in the uniform
  continuation loop, the retired identity-guard wording on `T3Tangent`, and `t3_norm(use_orthogonalization=False)`,
  which orthogonalized anyway (it now honors the flag; stack-shape mismatches in `+`/`inner` are
  `ValueError`s, as documented).

### Performance

- **The inner CG compiles once per fitting run instead of once per Newton iteration** (measured
  1 → 0 compiles per iteration on the uniform `probe_derivatives` path, warm). `_cg_solve(local_model, rhs, tol, maxiter, use_jit)` dispatches
  to one function compiled over `(local_model, rhs, tol, maxiter)`, so the cache key is the model's pytree
  treedef — value-based, thanks to the change above.
- **`mc_sgd` / `adam` per-step kernels compile once per shape signature**, process-wide, rather than
  once per optimizer call: the kernels are module-level functions and the jit wrapper is memoized on
  them, replacing a per-call closure that discarded its own cache.
- A user-defined kind or geometry now gets this for free, provided its parameters are dataclass fields.

## [2026.1.0] — 2026-08-20

### Added

- **The explicit equality checks**: `allclose(other, rtol=, atol=)` and `corewise_equal(other)` on
  all twelve runtime classes (see the `==` entry under Changed), `corewise.corewise_equal` (the
  bitwise tree comparison underneath), and `safety.comparison_rtol` (the ambient jax-aware default
  tolerance for pure comparisons; mode-agnostic, like `frames_equal`). Train `allclose` compares the
  represented TENSORS by `norm(a - b)` -- the difference train through the orthogonalized norm,
  numerically stable exactly in the `a ~= b` optimization-residual case.

- **Scalar-left multiplication and scalar division on the point/tangent classes** (review R1-12 /
  H4-8): `TuckerTensorTrain` gains `__rmul__` (`2.0 * x` used to raise while `x * 2.0` worked), and
  `TuckerTensorTrain` / `UniformTuckerTensorTrain` / `T3Tangent` all gain scalar-only `__truediv__`
  (`x / 2`). The ragged weighted frontends `t3_weighted_norm` / `t3_weighted_inner` gain the
  `use_orthogonalization` kwarg their backend and uniform twins always had (review H2-6), and
  `check_fw_pair` is exported as the docs already said (review R10-7). Per-axis uniform
  `sum_stack(axis=)` is deferred-and-documented (review H2-7; `contributor/deferred_and_rejected.md`).

- **`backend.linalg.pad_safe_svd`** -- the mask-aware SVD of a zero-padded matrix: the first
  `min(n, m)` triplets are exactly the economy SVD of the unpadded real block, **bitwise** zero on
  the padding, with no rank tolerance anywhere and one jit compile across mask patterns (masks are
  runtime data). Batched, backend-agnostic, symmetric in `(n, m)` (wide real blocks need no
  caller-side transpose), deterministic (fixed Haar sketch per width). The algorithm is Method D
  ("sketch-project") of the pad-safe SVD design packet, `dev/review_2026-08-22/repros/S1b/packet/`:
  pads are permuted to the trailing QR pivots (Householder reflectors then never touch a padded
  row), one small augmented SVD pins the surplus, and the right factor is rebuilt from
  `A^T U = V diag(ss)`. The uniform sweeps use it wherever an SVD produces kept basis columns.
  The complete derivation note -- problem, contract, algorithm, measurements, and the
  alternatives considered -- is `docs/pad_safe_svd.tex` (+pdf).

- **Shared Tucker factors (SF-T3)** — optimize over Tucker tensor trains whose Tucker factors are
  constrained equal within user-specified groups of modes (the SF-ETT decomposition of Molozhavenko &
  Rakhuba (2026), generalized from one trailing block to an arbitrary partition; the partition is
  always user-provided). A `sharing` spec is one hashable group label per mode, e.g. `(0, 0, 1)`;
  a shared T3 is an ordinary `TuckerTensorTrain` whose group factors are equal (redundant storage —
  a compute-not-memory feature). Ragged layer and its full uniform mirror:
  - **The shared geometries** — `shared(MANIFOLD, sharing)` with shorthands `shared_manifold(sharing)`
    / `shared_corewise(sharing)`, exported from the package root: drop-in geometry wrappers for every
    optimizer and fitting model. Projections post-pass onto the tied tangent subspace *in the base
    geometry's own metric* (manifold: the tilted least-squares projection through a per-frame SVD
    companion; corewise: the per-group mean), and the manifold retraction goes through a tied
    doubled-rank embedding plus the grouped `t3svd`, so every iterate stays exactly tied (one array
    per group).
  - **Grouped `t3svd(sharing=)` / `rank_adjustment_sweep(..., sharing=)`** — the paper-faithful
    two-phase truncation (TT rounding, then simultaneous group Tucker SVDs on concatenated
    matricizations), reporting one spectrum `s_g` per group; `sharing=None` or an all-singleton
    partition is the existing sweep exactly.
  - **`TuckerTensorTrain.share(sharing, ...)`** — the quasi-optimal shared initializer (exact
    common-span rewrite + grouped truncation), and the checker method
    **`has_shared_tucker_factors(sharing, rtol=)`** (per-stack-element bool).
  - **Shared rank bookkeeping** — `get_minimal_ranks(..., sharing=)` /
    `backend.ranks.compute_minimal_ranks(sharing=)` with the group ceiling
    `n_g <= min(N_g, sum_i min(N_g, rL_i*rR_i))` (per-mode ceilings ADD across a group, so a shared
    rank may exceed an individual mode's `rL_i*rR_i` — the unshared reduction would clip it and untie
    the group), `manifold.manifold_dim(s, sharing=)` (one Stiefel term per group; validated against
    dense tied-tangent ranks), and `frame_has_minimal_ranks(..., sharing=)`.
  - **Shared rank continuation** — `continuation_ranks(sharing=)` /
    `backend.ranks.compute_continuation_ranks(sharing=)`: a group's Tucker edges are ONE edge — one
    `κ_g = s_g[0]/s_g[-1]` in the conditioning pool, one growth decision applied group-wide, one
    `max_grow` candidate — with the shared useless-rank removal (`κ_g` is never worse than the group's
    worst per-mode condition number, and can be far better on complementary spectra). Plus
    `resize(..., sharing=)` for the zero-padded warm start: the group factor is padded once (one array
    per group), the represented tensor unchanged. A freshly padded restart carries exactly-zero new
    spectrum levels (the tied Tucker channel is gated); the escape runs through the untied TT-variation
    channel within the first Newton steps — which is why full shared rank is a diagnostic, never an
    enforced precondition.
  - **Uniform mirror, grouped truncation family** — `UniformTuckerTensorTrain.t3svd(sharing=)` /
    `rank_adjustment_sweep(..., sharing=)` / `has_shared_tucker_factors(sharing, rtol=)`, backend
    `ut3svd(sharing=)` (the two-phase grouped sweep in scan/supercore form: TT-bond rounding scan with
    the Tucker steps skipped, centers collected by the polymorphic right sweep, per-group SVDs on
    statically-gathered concatenations — mask-only truncation, ONE group rank mask at every group
    mode), `ut3_rank_adjustment_sweep(sharing=)`, the grouped host recurrence
    `compute_raw_sweep_ranks(sharing=)` (verified == the ragged grouped output ranks over randomized
    structures/caps), the masked checkers `ut3_sharing_residual` / `ut3_tucker_factors_shared`, and a
    sharing-aware `uniform_minimal` — required: the per-mode reduction silently unties a shared
    uniform start (it can clip a group rank to unequal per-mode values). All verified under the
    uniform equivalence contract (== the ragged grouped ops on real parts, per stack element,
    varying-rank stacks included), with exact output-mask assertions, garbage robustness, and
    jit-clean dispatch.
  - **Uniform mirror, tied tangent machinery** — the uniform companion `ufv_shared_frame_data`
    (the identical polymorphic derivation on the frame's stored supercores — deliberately NOT
    re-masked: the companion's exactness rests on reproducing the construction's own sweep on the
    same arrays, and the padded rows of each `S_i^T` vanish because completion rows are orthogonal
    to the centers' row space), the tied post-passes `ufv_share_tucker_variations` /
    `ufv_share_tucker_variations_corewise` (mask-and-delegate to the polymorphic ragged solves), and
    `shared_data=` threading through `utv_orthogonal_gauge_projection` / `utv_to_ut3` (the TIED
    doubled-rank embedding: `Udot` at every group mode, the companion's centers replacing the down
    cores, the variation block rebuilt at the up width) / `utv_retract` (tied embedding + the grouped
    `ut3svd`). Verified gauge-invariantly against the ragged twins (dense tangents/points at
    machine precision; outputs exactly tied).
  - **Uniform mirror, shared geometries + fitting** — `shared(UNIFORM_MANIFOLD, sharing)` /
    `shared(UNIFORM_COREWISE, sharing)`: the `SharedGeometry` wrapper now takes the uniform
    singletons too (uniform points/frames/tangents in and out; same value-hashed identity), the
    backend factories take `sharing=` (`uniform_geometry_ops` and both singles — the closures
    capture the partition beside the masks and populate `precompute` with the uniform companion),
    `uniform_least_squares_problem(sharing=)` with a shared-minimal gate, and the frontend fitting
    models carry the companion as a `geometry_aux` leaf (`UniformGaussNewtonModel`), so the packed
    compile-once path holds for shared fits (one trace across rebuilt same-rank models). All four
    optimizers run shared on the uniform layer; deterministic trajectories match the ragged shared
    runs, and every iterate stays exactly tied.
  - **Docs**: the user page [`docs/sharing.md`](https://nickalger.github.io/T3Toolbox/sharing.html) (the format, the grouped
    truncation, *what the group spectrum is*, the two geometries, rank machinery, batching,
    uniform, scope — including sharing ≠ symmetry); the design record
    [`docs/contributor/sharing_internals.md`](https://nickalger.github.io/T3Toolbox/contributor/sharing_internals.html); a shared
    section in the CI-doctested getting-started tour; the sharing section of
    `docs/rank_continuation.md`; and the TIED precondition rows in `docs/numerical_contracts.md`.
  - **Weights × sharing** — the combination composes within the existing framework (absorbing keeps
    a tied T3 tied iff the group Tucker weight vectors are equal; TT-bond weights never touch the
    factors; `from_t3svd(x, sharing=…)` builds group-equal weights; the weight algebra preserves
    group-equality), so the only addition is the non-enforcing compatibility checker:
    `T3Weights.has_shared_tucker_weights(sharing, rtol=)` and the `UT3Weights` twin (masked
    content), with `t3_tucker_weights_sharing_residual` / `t3_tucker_weights_shared` (+ `ut3_*`) in
    `backend.sharing`. Nothing gates — absorbing group-unequal weights legitimately unties.
  - **Example**: `examples/fit_shared_factors_jetted_probes.py` — a groupwise-symmetric five-mode
    target (two Hilbert tensors coupled by a random matrix; two sharing groups of different mode
    sizes) fit from noisy probe-derivative jets, running the SAME rank-continuation fit shared vs
    unshared: tying the factors the target's symmetry justifies reaches ~35% lower true error with
    ~37% fewer parameters before overfitting.
  - **Entering the format is the geometry's job.** `shared(...).frame(x)` ties an untied `x` first,
    silently, by the per-group mean — so an untied initial guess is a non-event for the optimizers, and
    the slow drift a long run of low-precision first-order steps can produce is absorbed at the next
    frame. An already-tied point is a bitwise fixed point, so the ordinary path is unchanged. The
    uniform layer gets the same route without a round trip through ragged:
    **`ut3_tie_tucker_factors`**, the twin of `t3_tie_tucker_factors` (garbage-transparent, so it needs
    no masking; masks and TT cores untouched). The shared **corewise** retraction ties the *sum* rather
    than aliasing one mode's copy of it, which makes it total: `mean_i(U_i + V_i) = mean_i(U_i) +
    mean_i(V_i)`, so an untied tangent (always handled) and an untied base point both land on the
    shared set with nothing silently discarded. The TIED-tangent precondition is now one backend
    checker — `fv_tied_variations_residual` / `ufv_tied_variations_residual`, a single **global
    Frobenius** ratio **per stack element** — replacing two hand-rolled formulas that disagreed between
    the ragged and uniform paths and collapsed the whole stack into one scalar (so one untied stack
    element could hide behind many tied ones).
  - Backend surface in `backend.sharing` (`validate_sharing`, `t3_sharing_residual`,
    `t3_tucker_factors_shared`, `t3_tie_tucker_factors`, `SharedFrameData` +
    `fv_shared_frame_data`, the tied post-passes) and `backend.t3_svd.t3_share_tucker_factors`.
    Safe mode checks tied factors at shared entry points; full shared rank is a diagnostic, never a
    precondition (rank-continuation restarts legitimately sit below it).

- **The grouped-einsum interpreter `backend.contractions.contract`** — one general entry point for
  every grouped contraction: `contract('WCa,Caib,WCi->WCb', *operands, len_W=...)`. Standard einsum
  strings where an UPPERCASE letter is a *group* of zero or more axes (`W` probe stack, `K` tangent
  stack, `C` frame stack, …); group sizes are solved exactly from the operand ndims (identifiability
  is decided from the subscripts alone — a call site either always needs a `len_<G>` supplement or
  never does, and the error names precisely what is missing); groups expand into fresh single-axis
  letters and one ordinary einsum runs on the operands as given. **No reshape ever happens**, so
  every sub-axis of every group is shardable (compiler-verified across the whole library vocabulary)
  and fusing two groups is inexpressible. numpy keeps the greedy pairwise BLAS path (computed on the
  grouped string); jax gets a single fused einsum.

- **Weighting (edge weights)** — diagonal weights on the internal edges of a T3, as a lightweight data
  format plus `absorb` into cores, in **both** the ragged and uniform representations. Two classes per
  layer, because a tensor and a tangent have genuinely different edges:
  - **`T3Weights` / `UT3Weights`** weight a Tucker tensor train **as a tensor** (`tucker[d]`, `tt[d+1]` —
    exactly the shape `t3svd` returns, so the singular values *are* the canonical weight object).
  - **`T3FrameWeights` / `UT3FrameWeights`** are a **metric on a tangent's coordinates** (`up`/`down`/
    `left`/`right`, each `len=d`) — the Grasedyck–Kramer preconditioner — absorbed into the variation
    cores with the frame left orthonormal, so they are `O(ranks)`.
  - Operations on all four: `absorb_weights`, `weighted_norm` / `weighted_inner`, `reciprocal` / `sqrt`,
    and `concatenate` / `kronecker` (the `+` / `⊙` duality: ranks add / multiply). Constructors
    `from_t3svd` / `from_ut3svd` and `from_t3weights` / `from_ut3weights`, plus ragged↔uniform
    conversions for both weight types.
  - The frontend free functions carry the family prefix — `t3_absorb_weights`, `ut3_absorb_weights`,
    `fv_absorb_weights`, `ufv_absorb_weights` (+ `t3_`/`ut3_weighted_norm` / `_weighted_inner`) — and the
    whole surface is exported from the package root. Docs: `docs/weighting.md`; design records:
    `docs/contributor/weighted_internals.md`.

- **Regularization on the fitting objective** — `regularizer=` on every optimizer
  (`gradient_descent` / `mc_sgd` / `adam` / `newton_cg`) and on the six model factories, minimizing
  `½‖ω⊙(S(x) − y)‖² + ρ(x)`. Shipped implementation: `IdentityRegularizer(λ)` = `½λ‖x‖²` — a
  Hilbert–Schmidt ridge on `MANIFOLD`, weight decay on `COREWISE` — composing with every optimizer,
  sampling kind, geometry and representation (ragged **and** uniform), with `λ` auto-scaled by
  `batch/n` in the minibatch steps so the stochastic objective is an unbiased estimate of the full
  one. Extensible through the small `Regularizer` protocol in `backend/regularization.py`. Docs:
  `docs/fitting_and_optimization.md` §4.9; example `examples/fit_hilbert_regularized.py`
  (denoising, with λ chosen by held-out validation).

- **Per-mode residual weighting** — the fitting layer's residual weight `ω` generalizes from a
  per-order vector to an `ω[mode, order]` matrix, owned by the sampling kind: `probe_model(weight=)`
  takes a length-`d` per-mode vector and `probe_derivatives_model(weight=)` a `(d, order+1)` matrix
  (probe is the only kind with a per-mode axis, so `apply`/`entries` stay per-order). A bare vector
  is still read as per-order, so existing calls are unchanged. Docs:
  `docs/fitting_and_optimization.md` §4.6; example `examples/fit_per_mode_weight_probes.py`
  (inverse-noise weighting of a noisier mode).

- **Newton-CG diagnostics** — `newton_cg(..., verbose=True)` prints a per-iteration block (objective
  and gradient norms, CG statistics, line search, ρ, wall time) plus a per-`(mode, order)`
  relative-error table, with an optional held-out validation column via `val_sample=` / `val_data=`;
  the same records are returned in `stats['diagnostics']` and `stats['history']`, and a regularized
  run splits the objective as `obj = misfit + reg`. Backend-owned
  (`backend.optimizer_display.make_newton_display` + a `callback=` hook), so a raw-`.data` user gets
  the identical display; ragged and uniform. Example `examples/fit_probe_display.py`.

- **Newton-CG warm-start controls** — `g0norm_newton` / `g0norm_cg` pin the reference `‖g₀‖` that the
  Newton stopping test and the CG forcing term are measured against (the computed initial norm is
  misleadingly small after a rank-continuation warm start; `g0norm_newton` also feeds CG unless
  `g0norm_cg` is given), and `cg_forcing_power` (default `0.5`) trades CG iterations per Newton step
  against the number of Newton steps. `NewtonInfo.g0norm` reports the effective reference. Docs:
  `docs/fitting_and_optimization.md` §5.

- **`chunk_size` for the probe-derivative transpose** — the `𝒥ᵀ` variation assemblies are computed in
  chunks over the probe stack `W`, trading recompute for peak memory: `chunk_size` (default `100`,
  `None` = dense) threads through the whole transpose chain and both `T3Tangent` /
  `UT3Tangent.probe_derivatives_transpose`, and the optimizers accept `chunk_size='auto'` (the
  default), which sizes it from `x0`'s shapes. The sizing helpers
  `backend.sampling_derivatives.estimate_chunk_size` / `max_chunk_size_within` are eager (outside
  `jit`), measure the per-row cost by lowering the real kernel rather than estimating it
  analytically, and take an `n_shards=` for per-device sizing and a `stack_shape=` for a batch of base
  points (the frame stack `C` multiplies the assembly but not an absolute byte budget, so omitting it
  on a stacked frame would make `max_chunk_size_within` return a chunk up to `prod(C)` times too
  large). Docs: `docs/chunking.md`.

- **Recurrence/scan jets are now the standard derivative path** — the pushthrough, combine and
  variation-assembly jets are computed by affine two-term recurrences and order-scans rather than by
  contracting the dense binomial `trs` tensor. The lean forms own the canonical names
  (`compute_mu_jets`, `compute_eta_jets`, `compute_{sigma,tau}_jets`, `compute_deta_jets`, the tilde
  twins, and `assemble_{tt,tucker}_variation_jets`); the dense forms remain public as the `*_trs`
  reference twins (numerically equal, and the test oracle). Under `jit` on the uniform layer this is
  a large memory win at unchanged accuracy — measured at rank 128, `W`=32000: ~14–28× less XLA
  temporary for the `eta` jets and 168 GB → 2.63 GB (64×) for the TT-core variation assembly.
  *(Not all of them, as shipped: several kernels kept dense `trs` contractions until 2026.3.0 -- see there.)*

### Changed

- **BREAKING: the backend `optimizers.GeometryOps` protocol** gains an optional `precompute` slot
  (`frame -> geometry aux`, `None` for the existing geometries), and `project`/`retract` take a third
  argument (`(frame, variations, aux=None)`). `Problem.local_model` builds the aux once per Newton
  step (a `LocalModel.geom_aux` leaf field beside `sweep`) and passes it to `project`/`retract`; the
  frontend `GaussNewtonModel` mirrors it as a `geometry_aux` leaf. Migration for custom `GeometryOps`:
  accept (and ignore) `aux=None` in `project`/`retract`; the shared geometries use the slot to compute
  their per-frame SVD companion once per local model instead of once per CG matvec.

- **The `Regularizer` protocol threads the geometry aux**: `gradient`/`hessian`/`quadratic` gain an
  `aux=None` parameter (the per-frame geometry companion, e.g. the SF-T3 `SharedFrameData`), and
  every model/`LocalModel` call site passes its stored companion — closing the one seam where a
  regularized *shared* fit rebuilt the companion per CG matvec. Custom `Regularizer`
  implementations: accept (and may ignore) `aux=None`. Design record:
  `docs/contributor/precompute_and_caching.md` (the precompute/caching principle and audit).

- `backend/common.py` gains `prefix_mask` (the boolean prefix indicator shared by every uniform prefix
  structure) and now hosts `require_concrete_masks`, which moved from `backend/ut3_masking.py` — it is
  infrastructure for the uniform *mask-representation contract*, not part of any one object family.

- **`use_jit=True` now converts its inputs to jax instead of silently running eager.** Previously the
  flag was accepted but ignored unless `x0` / `sample` / `data` were already jax arrays, so a "jit"
  run could be an eager one (and a meaningless benchmark). Requesting jit is now taken as opting into
  jax-world precision: `x0`, `problem.sample` and `problem.data` are moved onto jax, the result comes
  back **jax-backed** (float32 unless `jax_enable_x64` is set), and the call **raises** if jax is not
  installed. `use_jit` is also now an explicit keyword on `newton_cg` / `mc_sgd` / `adam` rather than
  arriving through `**kwargs`. Docs: `docs/fitting_and_optimization.md` §4.5.

### Fixed

- **A `GaussNewtonModel` built from a parameterized sampling kind recompiled on every rebuild.** The
  model carries its `SamplingKind` as jax pytree `aux_data`, and jax keys its compilation cache on the
  aux; the derivative kinds and the weighted probe kind are built per model out of fresh closures, which
  under dataclass field equality never compare equal. So the documented "roll your own optimizer"
  pattern — rebuild the model at each outer point, `jit` a function of it — paid a full recompile every
  step (measured: 3 traces for 3 rebuilds, against 1 for the `apply`/`entries`/`probe` singletons).
  `SamplingKind` now compares on a value `identity` (name, order, residual weight, `chunk_size`) rather
  than on its closures, so a rebuilt kind with the same parameters is the same cache key while a
  genuinely different one still gets its own compilation. The uniform model already did this through its
  value-hashed aux; the ragged twin now matches, and both are pinned by regression tests.
- `backend.fv_conversions` ignored `squash_tails=False` — the parameter was shadowed inside the
  function, so the tails were squashed regardless of what the caller asked for.
- The uniform frame/variation entry points (`ufv_apply_frame_masks`, `ufv_apply_variations_masks`,
  `ut3frame_to_t3frame`, `ut3variations_to_t3variations`) accepted **traced** masks and failed deep
  inside jax with a `TracerArrayConversionError`; they now call `require_concrete_masks` and raise the
  actionable message, like the plain-layer entry points already did.

### Removed

- **BREAKING: the ~104 named contraction functions in `backend.contractions`**
  (`WCa_Caib_WCi_to_WCb`-style), replaced by the `contract` interpreter above. Migration is
  mechanical: the function name is the subscripts string (`X_Y_to_Z(a, b)` →
  `contract('X,Y->Z', a, b)`), and the trailing `n_probe` / `n_frame` arguments become
  `len_W=` / `len_C=` keywords. Numerically identical (each named function was verified equal to
  its `contract` call over an empty/single/multi-axis block-shape matrix before removal).
- The old parked weighted layer (`weighted_tucker_tensor_train.py`, `backend/wt3_operations.py`, the
  `wt3_` prefix, and the broken `absorb_weights_into_tangent_cores`), superseded by the above.

## [2026.0.0] — 2026-07-13

The first public release — the initial public surface:

- The **Tucker tensor train (T3) format** — arithmetic with dense-tensor semantics,
  orthogonalization, minimal ranks, T3-SVD, save/load, batching on every operation.
- The three **sampling operations** (`entries` / `apply` / `probe`), their symmetric
  directional derivatives, and the ambient/corewise/tangent transposes.
- The **fixed-rank T3 manifold**: orthogonal frame + gauged variations, tangent vectors,
  gauge projections, retraction, and the `MANIFOLD` / `COREWISE` geometries.
- **Least-squares fitting** from any sampling operation or its derivatives (Gauss-Newton
  models) with four optimizers (`gradient_descent`, `mc_sgd`, `adam`, `newton_cg`).
- The **uniform layer**: supercores + boolean rank masks mirroring the whole stack, for
  `jax.lax.scan` vectorization and compile-once `jit` (optimizers included).
- **NumPy / JAX** backends with dispatch inferred from input array types; **safe mode**
  for numerical-precondition checking.
