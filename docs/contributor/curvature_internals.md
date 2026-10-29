# The exact Hessian — curvature internals

The design record behind `hessian='exact'` (2026.3.0): how the exact Riemannian Hessian is assembled from
the library's own sweeps, the rules each piece obeys, and the measurements the design rests on. The user
story — when to use it, the API, reading the display — is
[`../fitting_and_optimization.md`](../fitting_and_optimization.md) §4.12; the mathematics in full (the
derivations, the two-hole sweep as algorithms, the verification) is the math note
[`../exact_hessian_two_hole.pdf`](../exact_hessian_two_hole.pdf) (source `docs/exact_hessian_two_hole.tex`).
Read this before touching the lift, a two-hole kernel, the curvature companions or the trust region.

## The formula, and where each piece lives

At a point `X = Φ(θ)` with orthogonal frame `(U, O, L, R)` (`T3Frame.data`), gauged coordinates
`v = (V_i, H_i)` and the gauge projection `Π`, the exact Riemannian Hessian of
`f(X) = ½‖ω ⊙ (A X − y)‖²` is

```
H p = Π 𝒥ᵀω²𝒥 Π p  +  Π Lᵀ (K_{Aᵀω²r} − K_{grad f}) L Π p  +  H_R p
```

the Gauss-Newton term plus the Weingarten term (Absil, Mahony & Trumpf 2013, "An extrinsic look at the
Riemannian Hessian"): the residual's normal part paired with the second fundamental form, written as a
Lagrangian Hessian `f∘Φ − ⟨grad f, Φ⟩` pulled back through the lift. Reference: T4S (arXiv:2603.21141)
App. A for the frame and gauge, Psenka–Boumal (arXiv:2011.13395) for the second-order retraction.

| math | code | notes |
|---|---|---|
| frame factors `T_i = ⟨L_i, H_i⟩`, `S_iᵀ = ⟨O_i, H_i⟩`, centers `H_i = L_i T_i` | `fv_operations.fv_center_factors` | `T_i` by zipper on demand, never stored on the frame (I-1); `σ(S_i)` is the mode-`i` Tucker spectrum, `σ(T_i)` the bond spectrum; `‖X‖² = Σσ(S_0)²` |
| the lift `L`: `U̇_i = (S_iᵀ)^{+ε} V_i`, `Ġ_i = H_i ×₃ T_i^{+ε}` (`i ≤ d−2`), `Ġ_{d−1} = H_{d−1}` | `tv_operations.tv_corewise_lift` / `_transpose`; factors `fv_lift_factors` | raw core directions of the base point `θ = (U, L)` |
| the filter `f(σ) = σ/(σ²+ε²)`, clipped, `f(0) = 0` | `linalg.filtered_reciprocal`, `damped_pseudo_inverse` | `ε = curvature_damping·‖X‖_HS`; `eps=None` is the tied solve's filter, bit for bit |
| `K_c θ̂ = d/ds ∇_θ⟨c, S(Φ(θ + sθ̂))⟩` | `t3_<kind>_corewise[_derivatives]_hessian[_from_sweep]`; `SamplingKind.corewise_hessian` | the two-hole sweep, raw `K + C` |
| `Z = grad f` as a doubled-rank T3; `K_Z θ̂` | `tv_to_t3(frame, g)`; `t3_linalg.t3_inner_product_corewise_hessian[_from_sweep]` | the geometry's `tangential_curvature` |
| the model | `backend.optimizers.LocalModel.hvp` / `curvature_hvp`; `Problem.local_model(hessian='exact')` | `hvp` projects `𝒥ᵀz + y` once |
| corewise exact | `GN + K_{Aᵀω²r}` | no lift, no gradient term: the plain Hessian of `f ∘ Φ` |
| regularizer | `H_R = λΠ` | already exact (`X` lies in its own tangent space); the curvature uses the **misfit** gradient only |

The curvature body (`LocalModel.curvature_hvp`) is four calls, each a geometry or kind member: `θ̂ =
geom.lift(frame, Πp, data)`, `k = kind.corewise_hessian(θ̂, residual, sample, frame, sweep, base_point)`,
`k −= geom.tangential_curvature(frame, θ̂, data)` (`None` on corewise), `y = geom.lift_transpose(frame,
k, data)`. Everything T3-specific is behind those members, so `backend/optimizers.py` still imports no
T3 module.

## The lift and its one-sided exactness

The lift turns gauged frame coordinates into raw core directions of the base point, so that a *corewise*
second derivative can act on a *tangent* direction. It is a pseudo-inverse, and it is exact on one side:
`tv_to_dense(t3_corewise_frame(U, L), L v) == tv_to_dense(frame, v)` for every tangent `v` — at zero
damping, on every frame `t3_orthogonal_representations` builds, minimal or not. On a Tucker-slack frame
`S_iᵀ` is wide and on a TT-slack frame `T_i` is tall, and the pinv is still an exact one-sided inverse
there (F2, F3: 6e-15 / 9e-16); only a hand-built frame with a *wide* `T_i` would give a least-squares lift
(a caveat, never reached by the library). The last TT slot passes through undamped. The same factors serve
both directions, so `Lᵀ_ε K L_ε` stays symmetric for every `ε`.

```python
>>> import numpy as np
>>> import t3toolbox.tucker_tensor_train as t3
>>> import t3toolbox.backend.fv_conversions as fvc
>>> import t3toolbox.backend.fv_operations as fvo
>>> import t3toolbox.backend.tv_operations as tvo
>>> np.random.seed(0)
>>> x = t3.TuckerTensorTrain.randn((5, 6, 7), (2, 3, 2), (1, 2, 2, 1))
>>> frame, _ = fvc.t3_orthogonal_representations(x.data)
>>> v = tvo.tv_orthogonal_gauge_projection(frame, fvo.fv_variations_randn(fvo.fv_variation_shapes(frame), ()))
>>> lift = fvo.fv_lift_factors(frame, 0.0)                 # undamped
>>> theta_hat = tvo.tv_corewise_lift(frame, v, lift)       # raw directions of the base point (U, L)
>>> U, _, L, _ = frame
>>> a = tvo.tv_to_dense(fvc.t3_corewise_frame((U, L)), theta_hat)
>>> b = tvo.tv_to_dense(frame, v)
>>> bool(np.linalg.norm(a - b) < 1e-12 * np.linalg.norm(b))     # the lift represents the tangent
True

```

**The damping is a solver parameter, not a geometry field** (I-22; the "options are not parameters" rule
of [`parameters_not_closures.md`](parameters_not_closures.md)). It reaches the computation only through the
traced `LiftFactors.eps` leaf — so a new damping never recompiles — and the default `1e-6` lives in
`backend/geometry.py` (`DEFAULT_CURVATURE_DAMPING`), which resolves `None`. At a zero-padded restart the new
singular values are roundoff (~1e-17); the relative clip zeroes them, which is what makes restarts need no
special case (E2, F19; the trust region bounds the rest).

## The two-hole sweep

`K_c θ̂` is the directional derivative of the corewise transpose: the one-hole sweep leaves one core
out and contracts everything else; its derivative along `θ̂` leaves one core out *and* replaces one other
core by its direction — two holes. Each kernel is hand-written as the **product rule of the Gauss-Newton
scan bodies it differentiates**, never as autodiff and never through the T3-SVD (the jvp and the
polynomial finite difference are oracles only).

**The block-sum oracle** (F5) is the reference every two-hole is tested against, and lives only in
`tests/backend/_curvature_oracles.py` (I-15): embed `(θ, θ̂)` into a doubled-rank train `Y` whose
`i`-th diagonal block carries `θ̂_i`, run the *one-hole* corewise transpose at `Y`, and read back the
blocks — equal to `K_c θ̂` exactly, and to a central finite difference of the one-hole to the FD's
resolution. A third, independent oracle is the jvp: `jax.jvp` of the shipped corewise transpose along `θ̂`
(`tests/test_dispatch.py`, `TestCurvatureJvpOracle`, float64, 1e-12). The whole Hessian has a fourth, dense
one: the extrinsic Weingarten form through the parametrization's exact Jacobians
(`tests/oracle_sweeps/sweep_exact.py`, every kind × geometry × representation × sharing).

**The two-index-jet rule** (the derivative kinds). Every edge quantity is multilinear in the cores and the
sample vectors. The jet axis `t` (length `order + 1`) is the sample-direction Taylor index; the core
direction is a second, *independent* index truncated at first order, for which Leibniz needs no binomials,
`(fgh)˙ = ḟgh + fġh + fgḣ`. So each existing three-factor `t`-convolution spawns three dotted terms with
the same order wiring. The dotted input jet is `build_input_jets(compute_xi(U̇, ww), compute_xi(U̇, pp))`,
affine in `t` like `xi`. **Layout: companion arrays** (I-10) — undotted jets are `(order+1,)+W+C+(…)`,
dotted ones `(order+1,)+W+K+C+(…)` — not a stacked length-2 axis, which would broadcast every undotted
jet over `K`.

**The dotted kernels are recurrences** (I-33, Nick 2026-10-27, superseding the earlier "hold the
Gauss-Newton kernels' standard"). The hard rule for every production kernel: banded recurrences where a
factor is affine, order scans whose binomial weights come from a shift recurrence where the convolution is
full — never a dense binomial (`trs`) tensor, which lives only in the `_trs` test twins
(`tests/_jet_trs_twins.py` for the Gauss-Newton kernels, `tests/backend/_curvature_oracles.py` for the
two-holes). The dotted adjoint step is the product rule of `_adj_tilde_step`: its **prop** half (three
dotted pushthroughs, each a two-term reverse recurrence in one GEMM; shared since S12 as
`_adj_prop_tangent`) and its **src** half (three dotted sources, each an inner order scan). `dx̃i` and its
dot return only the two rows the Tucker assembly reads. Measured: equal to the `_trs` twins to ≤ 5.8e-16
over orders 0–4 (bar 1e-14).

**The factored assembly** (F33). The TT gradient assembly is trilinear; its derivative has three terms
per group. Forming the undotted **pair intermediate** and its dotted twin once, then
`dG = Ṗ ⊗ σ̃ + P ⊗ σ̃̇`, needs fewer `W`-summed final contractions: 6 instead of 9 for the probe-derivative
assembly (1.4–2.0× faster on that term, equal to 3e-15), probe 9 → 4, apply / entries 3 → 2 (S10/S11). The
assemblies are `W`-chunked through `_wchunked_sum` like the transpose's (probe derivatives only; the other
kinds' transposes have no chunked path, so neither do their two-holes).

**The T/S transforms** (F16). At the *manifold* frame the corewise sweep and adjoint state are cheap
transforms of the manifold ones (`ν_c = Tν`, `η_c = Sη`, `σ̃_c = Tσ̃`, `dx̃i_c = S dx̃i`, apply's `σ̂_c =
Tσ̂`; ≤ 9.5e-16) — at the corewise frame they are not (`ν_c = Tν` fails by 6.2). The shipped kernels do
not use them: the default is **recompute** (I-11, the house low-memory preference) — `xi` / `mu` are
reused from the model sweep, `ν` is re-swept through `G` (1–3 % of the two-hole), no corewise `η` or `dx̃i`
is formed. The transforms are what a **store** variant (~+60 % of the jet sweep for ~10–15 % of the
two-hole) would use; it is deferred ([`deferred_and_rejected.md`](deferred_and_rejected.md)).

**The T3-to-T3 term.** `K_{grad f}` is the same second derivative for `θ ↦ ⟨Z, Φ(θ)⟩` at the misfit
gradient `Z` (a doubled-rank T3, `tv_to_t3`): the ungauged kernel `t3_inner_product_corewise_transpose`
(moved from `tv_operations` to `t3_linalg`; `tv_project_t3_onto_tangent_space` gauges it, bit-identical),
its precomputed sweep, and the two-hole through the dotted zippers `tt_zipper_tangent_*`. Its cost is
`m`-independent.

## Model-level rules

**Misfit gradient once** (M2). Exact mode costs no second transpose: `Problem.local_model` computes
`g = Π 𝒥ᵀω²r` once, stores it in `CurvatureState.misfit_gradient`, and both `LocalModel.gradient` (plus
`g_R`) and the geometry's `curvature_precompute` read it. So an exact model's gradient is the
Gauss-Newton one **bit for bit**, and the per-Newton-step overhead is the `Z` build, one T3-to-T3 sweep and
`2d − 1` small SVDs (the lift's `d` Tucker and `d − 1` bond pseudo-inverses).

**The companion pytree rule** (M3). `LiftFactors`, `ManifoldCurvatureData` and `CurvatureState` are
registered pytrees (each under `if jax_available:` in its defining module) whose flatten runs
`partition_static`: arrays are leaves, masks / shape ints / Python bools go into a value-hashed
`StaticSkeleton` aux. One registration serves both layers — the backend `LocalModel` and the frontend
model just carry the companion as a child — so a rebuilt exact model, a new frame or a new damping is one
jit cache key, and a uniform companion's host masks are never traced. The mode is **derived**, never
stored (`is_exact = curvature is not None`, I-3, B1); the term is present iff `geometry_data is not None`.

**θ̂ masking from the FRAME** (uniform; m29, F18, F23, F25). On a uniform supercore a direction's padding
multiplies *into* real slots in the second derivative, so the masks are load-bearing (an unmasked `θ̂`
corrupts the real block by 2.6e3), not hygiene. The uniform kinds' `corewise_hessian` masks `θ` and `θ̂`
with the masks of the **frame it is handed** (`frame_data[5]`'s up / left masks) — not `self.masks`, which
is a strict superset at TT slack — and re-masks the output (the S0c rule: every uniform transpose returns
clean padding). Garbage here means **finite** garbage: NaN leaks library-wide because masks multiply
([`testing_strategy.md`](testing_strategy.md)). The uniform lift factors clip each block at its REAL
dimensions; padded singular values are exactly 0.

**Flat slots** (I-12). The curvature term is the second fundamental form of the representable set. A slot
at maximal rank (Tucker rank = mode size, or a TT rank = the product of its neighbours) puts no constraint
on its unfolding, so the set is locally open in that slot's directions and the term has no component
there; at `d = 1` every slot is maximal at once (the set is open in `ℝ^N`, E4, E5). The formula needs no
special case — the filtered pinvs and the two-hole sum give `Lᵀ K_N L = 0` numerically. The manifold
`d = 1` geometry's `curvature_precompute` returns `None` purely as a cost shortcut.

## The canonical zero-point frame (the default Newton reference)

The default Newton reference (`g0_reference='zero'` on the manifolds) is `‖Π𝒥ᵀ(−ω²y)‖` at the zero tensor
of `x0`'s ranks. At `X = 0` every orthonormal completion is a valid frame and the gradient norm depends on
the choice, so the library fixes one (I-29): `fv_zero_point_frame(frame, groups, seed=ZERO_FRAME_SEED)`.
Per sharing group `g` (singletons are their own group), with `k_g` its first mode index:

```
G   = np.random.RandomState([seed, k_g]).standard_normal((n_g, N_g))
Q, R = np.linalg.qr(G.T)
U_g = (Q * sign(diag(R))).T                  # replicated over the group's modes
```

and identity-prefix TT cores with `L[d−1] = R[0] = 0`. Three properties, each by construction:

- **Rank independence.** Each group draws from its own keyed stream, so a factor never depends on another
  mode's rank or size (bit-identical when they change).
- **Nesting.** QR column `j` depends only on draw rows `≤ j`, so level `r`'s factor is exactly the
  leading-row prefix of level `r + 1`'s — the per-level references along a continuation ladder are taken
  on nested subspaces.
- **Layer independence.** The uniform twin `ufv_zero_point_frame` unpacks, builds the same real blocks and
  packs back (equal to ≤ 3.9e-16), so ragged and uniform runs share one reference; with per-layer LAPACK
  completions 6 of 16 ragged/uniform pairs drifted, with one reference 0 of 16 (F30), and the 44 O2 pairs
  agree to 3.5e-12 (F37).

```python
>>> def zero_frame(tucker_ranks, tt_ranks):
...     f, _ = fvc.t3_orthogonal_representations(t3.TuckerTensorTrain.randn((6, 7, 8), tucker_ranks, tt_ranks).data)
...     return fvo.fv_zero_point_frame(f)
>>> lo, hi = zero_frame((3, 2, 2), (1, 2, 2, 1)), zero_frame((4, 2, 3), (1, 3, 3, 1))
>>> lo[0][0].shape, hi[0][0].shape
((3, 6), (4, 6))
>>> bool(np.array_equal(hi[0][0][:3], lo[0][0]))     # nested: rank 3 is the prefix of rank 4
True
>>> bool(np.array_equal(hi[0][1], lo[0][1]))         # rank-independent: mode 1 unchanged by modes 0 and 2
True
>>> float(np.abs(t3.TuckerTensorTrain(lo[0], lo[2]).to_dense()).max())   # the base point is the zero tensor
0.0

```

**Why Haar, and only for `U`.** The Tucker factor is the only part of the frame that meets the data. Any
**coordinate-aligned** completion — LAPACK's of a zero core, or an identity prefix — gives a reference of
**exactly 0** whenever the data have no energy on the leading coordinates (F32); a Haar draw never does
(seeds 0–5 gave 9.5–24 on such data). The TT cores act on rank indices, which carry no data orientation,
so identity prefixes are exact and simple. The reference is therefore a **convention**: a deterministic
data-gradient scale at these ranks, good to an O(1) factor (another seed, or a rotation of the problem,
moved it by up to ~1.7×; P1b, E9), which is harmless for a relative stop. `RandomState`'s legacy stream is
frozen (NEP 19), so `G` is bit-identical across numpy generations; the QR is reproducible to ~1e-15 across
BLAS builds. The corewise geometries have no zero-point frame (the zero cores are a critical point, the
reference would be exactly 0 — P1a, E8), so their default is the warm start, and the member is optional
and duck-typed for custom geometries (m34). Alternatives considered: [`deferred_and_rejected.md`](deferred_and_rejected.md).

## Shared factors

The shared set is the tied parametrization `Φ(Tθ)` (`T` copies each group's factor to its modes), so its
Hessian is the same formula through a **tied lift** (`sharing.fv_tied_lift_factors`): each group's
coordinates lift to ONE raw factor direction, `U̇_g = M_g^{+ε}[V_{i₁}; …]`, against the stacked-`S` SVD the
tied projection already uses (the two share `_tied_filtered_solve`, bit-identical), the transpose sums the
group's copies, and the normal part is taken against the **tied** misfit gradient. A per-mode lift
represents the same tangent but not a tied direction; the two coincide iff every `S_iᵀ` is square, so the
tests use **group-ceiling** structures (a group rank above some member's own ceiling `rL·rR`), where the
per-mode lift is off by 5.6–19 in the quadratic form.

- **`s_g` in place of the group modes' own spectra.** The tied lift inverts `M_g`, never a group mode's
  `S_iᵀ`, so `lift_sigma_min` = min(bond spectra, untied modes' Tucker spectra, each group's `s_g`). A group
  mode's own spectrum can be tiny at a perfectly regular shared point (a mode barely using a basis row that
  another member uses fully).
- **Padded restarts.** A zero-padded shared restart's new `s_g` levels are zero and filter to exactly 0:
  the gated tied channel gets no curvature, as it gets no first-order motion; the TT variations carry the
  escape. The group part of the lift stays exact there (a tied tangent's group coordinates lie in the range
  of `M_g`; 8.8e-16).
- **The uniform padded `s_g` mask** (S9). The uniform companion holds padded slices (row splits in
  multiples of the padded `nD`), and its padded `s_g` levels are not guaranteed exact zeros, so each group's
  damped filter is zeroed on them by the frame's **group rank mask**, not only by the relative clip, and
  clipped at the group's REAL dimensions (`eps·max(Σ nD_i, nU)` × the host real-level mask). In a
  300-structure survey 1357 of 1362 padded levels were exactly 0 and the rest ≤ 3.3e-17·s₁ — the mask is
  what makes that a guarantee, and what keeps a hand-built companion's garbage out. `_lift_sigma_min` reads
  real levels only (else every varying-rank point would report 0).
- The shared corewise geometry needs no lift: its exact Hessian is the per-group mean of the plain one
  (`project_mean(K θ̂)`). The shared retraction's second order is **measured** (O(h²) finite differences,
  F14), not proven.

## The operand-scaled gauge check (I-38, S7b)

`MANIFOLD.norm` / `inner` check that their operands are gauged, relative to a scale. A tangent computed
directly (a projection, a gradient) is measured against its own coordinate norm. A tangent built by
arithmetic carries its operands' scale — `max(‖a‖, ‖b‖)` for `a ± b`, `|c|·‖a‖` for scalar `*` and `/`,
per `K + C` element, recursively, through `stack_tangents` — as a **floor** under its own norm,
`max(own, scale)`: never stricter than the self-relative check (replacing it would false-fail chains of
sums). The motivating case is the finite-difference check of a Hessian, `M.norm(H_fd − hessian(p))`, whose
operands agree to 1e-8 but each carry a 1e-16-relative gauge violation. The scale rides the cached
`gauge_residual` slot, never a dataclass field (the classes hold only their cores), is recorded only in safe
mode and eagerly, and is not pytree state (a jit round-trip measures self-relative again). Backend:
`tv_gauge_residual(frame, variations, scale=None)` / `utv_gauge_residual`.

Recorded limits, both accepted: **F1** — dividing an FD quotient by `2h` divides the recorded scale too, so
`M.norm(H_fd − hessian(p))` certifies the gauge only to `1e-9·‖grad‖/(2h)` (≈ 1.6e6·‖Hp‖ at `h = 1e-5`); to
certify a computed tangent's gauge, check that tangent (documented in
[`../numerical_contracts.md`](../numerical_contracts.md)). **F2** — a safe-mode leaf `±` now computes both
operand norms eagerly (55 → 314 µs at 3.2k dof); the optimizers run on backend `.data` and are unaffected.
Follow-ups (not built): carry the scale through `normalized` / `sum_tangents` / `unstack_tangents`, and a
`UT3Tangent.__truediv__` (ragged has `/`, uniform does not).

## The trust region (Steihaug–Toint)

One CG body serves both step controls (I-5): the state carries `radius` and `boundary`; with
`radius = inf` it is the plain CG bit for bit on eager numpy. Steihaug stops on the boundary on
nonpositive curvature or when the next iterate would leave the region; a boundary stop is not a
truncation (`cg_truncated` excludes it). One trial `retract(p)` per row, accepted on `ρ > 0.1`; the radius
rules and constants (0.1 / 0.25 / 0.75, ¼‖p‖, 2Δ, `Δ_max = 10⁴Δ0`, `Δ0` the GN Cauchy length) ship as
module constants (O-2). A descent-guard step with scale < 1 counts as a boundary step for the 2Δ rule.
`pHp` is `gn_quadratic(p) + curvature_quadratic(p)` with ONE two-hole (`== lm.quadratic(p)`). The floors
are checked only after a rejection (an accepted step never stops the run): `objective_floor` when
`|pred| ≤ 10·ε·(‖ω⊙y‖·‖ω⊙r‖ + |f|)` with `ε = np.finfo(residual dtype).eps` (host-side, dtype-only — I-25,
m30), then `trust_region_floor` (`Δ < 10⁻¹²·Δ0`). A rejected trial keeps `x` and its local model, so the
row *after* it — at the same point, since each row describes `x` *before* its step — re-uses the rejected
row's error tables; the display reuses them only when that row was rejected **and** `info.lm is` its model
(a callback shared across continuation levels never prints a stale table).

## Facts the design rests on

Numbers from the writer/critic verification scripts and the slice measurements (the evidence behind each
row is summarized in [`verification_records.md`](verification_records.md); the living guarantees are
`tests/backend/test_curvature.py`, `tests/backend/test_uniform_curvature.py`, the exact oracle sweep and
the dispatch tests).

| # | claim | measured |
|---|---|---|
| E1 | exact matches FD to 5–6 digits at a small bond σ ∈ {1e-2, 1e-3, 1e-4}; the validity radius is ~σ and the Hessian is indefinite there | the FD tests draw points with `σ_min > 1e-3` by predicate |
| E2 | zero-padded `apply` restart, d = 3: exact + TR converges where GN + Armijo needs many more steps; `ε ∈ {0, 1e-6‖X‖}` agree, `ε = 1e-2‖X‖` is slower, a Δ-coupled ε fails | **one unseeded draw** (the legacy stream was never seeded): its 13 / 69 vs GN 54 is irreproducible; the seeded rows are F19 / F27 |
| E3 | float32 cancellation in the curvature term is benign | — |
| E4, E5 | `N ≡ 0` at `d = 1` and at maximal rank | T20 (no special case) |
| E6 | uniform `MANIFOLD` at `d = 1` already raised in the optimizers | unchanged |
| E7 | the old uniform corewise GN Hessian left 2.7e5 in padded boundary slots | fixed by S0c (clean padding) |
| E8 | the corewise zero reference is identically 0 | the warm-start default there |
| E9 | the zero reference moves ≤ 1.6–1.7× under a mode rotation; a nonzero start's `‖g‖` is invariant | "good to an O(1) factor" |
| E10 | per-layer zero references broke the O2 ragged/uniform pairing (5.2e-4 / 3.9e-3) | the canonical frame (F31, F37) |
| F13 | the probe-derivative two-hole (orders 0–3, `K ∈ {(2,), (2,3)}`, `C ∈ {(), (2,)}`, d ∈ {4, 3, 1}) = the block sum | ≤ 1.0e-15 (S3: 4.4e-16; S12 apply/entries derivatives ≤ 7e-16; an independent dense oracle 9.7e-16) |
| F14 | shared, sharing `(0,0,1)`: the tied lift satisfies `DΦ[Lv] = Ev`; the exact form matches FD along the shared retraction at O(h²) | lift 5.3e-16 (S7 through `Problem.local_model`: 4.7e-16); FD ratios 4.0 / 4.25 |
| F15 | θ and θ̂ masked on entry → real block exact; an unmasked θ̂ corrupts it | 5.5e-17; unmasked 2.6e3 |
| F16 | the T/S transforms hold at the manifold frame, fail at the corewise one | ≤ 9.5e-16; 6.2 |
| F17 | `‖X‖² = Σσ(S_0)² = Σσ(T_i)²` | ≤ 1.4e-15 |
| F18 | the masking rule on a real `UniformTuckerTensorTrain` | 5.5e-17; mutation 2.6e3 |
| F19 | probe derivatives, order 2, ω = `[1, .5, .25]`, restart rank 2 → 3, 5 % noise: GN + Armijo 39 steps / 130 hvps; exact + TR 8 / 23 at power 1.0, 10 / 28 at 0.5, 0 rejected | seeded; the plan's restart template |
| F21 | the TR floor rule on both degenerate cases → `objective_floor` | T15 |
| F22, F29 | T16's rates (plain `apply`; a GN warm-up of 15 steps with `g0_reference='warm_start'`): GN ratios 0.56–0.68; exact power 1.0 `e_{k+1}/e_k² ≤ 0.92`; power 0.5 `e_{k+1}/e_k^{1.5}` 0.43–0.98 in `[1e-6, 1e-1]` | S12 through the library: 0.917 with the warm-start warm-up (the zero-reference warm-up moves it to 0.941 — drift of the warm-up's end point, not the kernels); T16 itself is re-based on probe derivatives (S5) |
| F23 | the uniform frame's up / left masks ⊆ the point's; a strict subset at TT slack | why the kinds read the frame's masks |
| F24 | every step with `|pred| > 1e3·res` has `|1 − ρ| ≤ 1e-3` | T16's step criterion (n15) |
| F25 | the frame argument reaches the real extent (m29); numpy float32 → float64 residual (m30) | — |
| F26 | g₀ interplay on F19's problem: zero 7.01e4 vs warm start 2.07e4; GN 36 vs 39 steps; exact + TR 9/27 vs 10/28 (power 0.5), 7/24 vs 8/23 (1.0); COREWISE zero reference 0.0; `g0norm_newton=0.0` used to raise `ZeroDivisionError` | — |
| F27 | E2's three apply restarts under each reference: warm/zero 1.9e-2, 1.6e-14, 3.1e-1; noise-free (B) the zero reference declares the start converged; above target (A) GN never reaches 1e-8 | S11: the library's exact `newton_cg` reproduces the critic harness row for row (A 18/1/117 at 1e-8, 17/1/92 at 1e-3; B 4/1/12 `objective_floor`; C 11/1/53, 10/1/33) |
| F28, F38 | full exact + TR from non-minimal points, no reduction: Tucker slack 20 outer / 4 rejected / 78 CG iterations (one Hessian-apply each; the per-step `pᵀHp` is extra) → 2.6e-9; TT slack 34 / 9 / 272 → 2.2e-12; the same counts from the minimal-rank tensor; `ε = 0` ≡ `1e-6‖X‖` | reproduced **exactly** by the library on plain `apply` (S12); `apply_derivatives` order 1: 17/4/56, 31/10/240 |
| F30 | per-layer LAPACK zero references diverge 0.56–1.93×; 6 / 16 pairs drift; with one reference 0 / 16 | — |
| F31, F37 | the canonical frame is valid at d = 1, 3, 4 and shared; layer-independent to ≤ 3.9e-16; rank-independent, nested (error 0.0 over r = 1…7); 44 O2 pairs agree to 3.5e-12 | T26 |
| F32 | on data with no energy on the leading coordinates, aligned completions give exactly 0; Haar seeds 0–5 give 9.5–24 | T26 (e) |
| F33 | the factored assembly is 1.4–2.0× faster on its term, equal to 3e-15; exact / GN cost ≈ 2.4–3.0× | after the recurrence conversion (S3b) the ratio rose to 2.8–3.5× because Gauss-Newton got 2.3× faster; S10–S12: probe 2.8–2.9×, apply 3.1–3.2×, entries 3.1–3.3×, apply derivatives 3.0–3.9× |
| F34 | S0c: uniform transposes' padded output reached 3e12; re-masked it is 0, the real block bit-identical, overhead 0.08–0.61 % | S0c review: ~1.1 % eager on a small transpose including the host mask build, ~0.13 % large, 0 under jit |
| F35, F36 | every new grouped string equals `np.einsum` on the generic multi-axis point; the import DAG is acyclic; `AXIS_SIZE` extended by `l`, `n`, `x`, `z` | `tests/test_contractions_interpreter.py` |

Memory and compile time (S3, S8, S12): the two-hole's TT-assembly kernel takes 2.2–2.8× the Gauss-Newton
assembly's per-row scratch after S3b (the running chunked program, which also holds the chunk's jet slices,
~1.3–2.2× on current jax); unchunked, the exact two-holes peak at 1.85–2.1× the GN transpose (jax uniform f32);
with `estimate_chunk_size(hessian='exact')` the probe-derivative two-hole's chunked assembly peaks at
1.25–1.61× the GN assembly at its own default chunk (each balanced against its own jet floor). The exact
uniform CG compiles once per run, ~15 s vs ~9.5 s for Gauss-Newton (CPU x64, small problems).
