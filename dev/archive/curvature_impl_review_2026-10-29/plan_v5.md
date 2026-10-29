# Implementation plan — the exact Riemannian Hessian (curvature term) for `newton_cg`

**v5, 2026-10-26. This version folds in Nick's round-5 rulings, which reshape the plan from first principles for a
BREAKING release, together with the critic's six pre-checks (`critic_prechecks_v5.md`, P1–P5 and the rename note).
Nothing is implemented.**

- **Repos:** `/home/claude/t3toolbox` @ `920d776` (`main`); downstream consumer `/home/claude/t3polynomial` @ `388090d`.
  Both are read-only here. Updating T3Polynomial is **in scope** for the release (§K).
- **Design source:** `dev/riemannian_newton_curvature_plan.md` ("DP"). The zero-reference g0 plan
  `dev/zero_reference_g0_plan.md` ("G0P") is folded in (C.15).
- **[V] claims:** read-only scripts under `SCR/checks/`, with
  `SCR = /tmp/claude-0/-home-claude/29728d3c-c433-5028-9de9-1af7ead3c123/scratchpad/plan_review`. Run them as
  `cd /home/claude/t3toolbox && python -I -B SCR/checks/<script>`.
- **[E] claims:** the critic's experiments, `SCR/check_*.py`.
- **References:** file:line at `920d776` (T3Polynomial at `388090d`).

**Target** (Nick's decision, DP status banner): **level 2, the exact Riemannian Hessian, by a hand-written two-hole
sweep**:

```
H p = Π 𝒥ᵀω²𝒥 Π p  +  Π Lᵀ (K_{Aᵀω²r} − K_{grad f}) L Π p  +  λ Π p
```

- `L` is the lift from gauged frame coordinates to raw core variations.
- `K_Z` is the two-hole sweep: the directional derivative of a corewise transpose.
- `grad f` is the **misfit** Riemannian gradient, held as a doubled-rank T3.
- Companions:
  - Steihaug–Toint CG plus a trust region;
  - a damped lift;
  - minimal diagnostics;
  - the breaking-release groundwork: the zero reference, the stats keys, the public problem constructor, the
    `LocalModel` rename and uniform output hygiene.

**Primary target: `probe_derivatives`**, the only kind T3Polynomial fits. It must work on all three backends and with
sharing. It is built first and is pre-verified end to end (F13, F14, F18, F19).

---

## CHANGES (v4 → v5)

v4 was signed off by the critic (`critic_signoff_v4.md`, AGREE). v5 is not a patch on v4. It restructures the plan
around Nick's round-5 rulings and makes the "straightforward route" wherever v4 had bent to the letter of a convention.
§L explains, case by case, why each route keeps the convention's intent. §I now carries **only** genuinely open items.

### Nick's rulings (all SETTLED; §I lists them in one table)

| v4 item | ruling | what v5 does | where |
|---|---|---|---|
| I-1, I-3, I-5, I-6, I-8, I-9, I-10, I-11, I-15, I-16, I-17, I-19, I-20, I-21, I-24, I-25 | agreed / OK as is | carried forward unchanged | §I table |
| I-2 | `hessian='exact'` | `hessian: str = 'gauss_newton'`, either value, case-insensitive | C.9, C.13 |
| I-4 | **rename** frontend `GaussNewtonModel` → `LocalModel`, mirroring the backend class it wraps | `fitting.LocalModel` has a `hessian` mode field. Its methods are `gradient`, `hessian(p)`, `quadratic(p)`, `objective_value`, `evaluate`, `retract`, the explicit `gn_hessian` / `gn_quadratic` and `curvature_hessian` / `curvature_quadratic`. The factories keep their names. **Not aliased**, as in the 2026.2.0 precedent for `UniformGaussNewtonModel` (CHANGELOG `[2026.2.0]`). The rename is its own mechanical slice, S0e | C.13, H, K |
| I-7 | `cg_forcing_power: float = 0.5` everywhere; no mode-dependent default | `0.5` on both layers in both modes. The docs say that `1.0` gives the q-quadratic tail with the exact Hessian; measured in F29 | C.9, G, T16 |
| I-12 | OK; state the reasoning | B, "flat slots" paragraph: at d = 1, and at every slot whose rank is maximal, the representable set is flat, so N has no component there. The formula needs no special case. d = 1 is special only because every slot is maximal at once | B, C.7 |
| I-13 | keep only the cheap always-on diagnostics; **defer** the normal-space sector norms | Kept: Δ, accept, the CG boundary symbol, pKp/κ, real-slot σ_min (with `s_g` when shared), ε, stop_reason. The sector-norm idea goes to `deferred_and_rejected.md` as one paragraph (G gives its text). Rank continuation keeps using the t3svd unfolding condition numbers | C.14, G |
| I-14 | `corewise_hessian` is a **required** sixth `SamplingKind` operation; no ClassVar flag, no MRO shadow | It has the same `NotImplementedError` guidance as the other operations (`backend/fitting.py:330-364`). The early structural check means `type(kind).corewise_hessian is SamplingKind.corewise_hessian`. It is spelled with the stdlib `@abc.abstractmethod` marker, so `optimizers.py` stays import-free and the window stub is also rejected early; T23 asserts the verbatim identity form too. During S4–S12, `_UniformKind` has a one-line **raising** override that names its slice. T23 carries a mutation check. The OPEN_QUESTION note records "six operations" | C.6, T23, G |
| I-18 (a) | fold in the zero-reference g0 plan | `newton_cg(g0_reference=…)`. The library computes the reference itself, and the explicit overrides win. Two adjustments come from the critic's pre-checks P1a–P1c (below) | C.15 |
| I-18 (b) | a **public** backend-problem constructor for research scripts | The frontend `optimizers.least_squares_problem(geometry, kind, sample, data, x0, …) -> (problem, x0_cores, rewrap)`, the frontend twin of `backend.optimizers.least_squares_problem`. `_setup` is **removed** (renamed; not aliased). `_geometry_ops` stays private, and nothing downstream calls it any more | C.16, K |
| I-22 | damping is a **solver parameter**, not a geometry field | `lift_damping` → `curvature_damping`. It flows `newton_cg(hessian='exact', curvature_damping=None)` → `Problem.local_model(x, hessian, curvature_damping)` → `geom.curvature_precompute(frame, misfit_gradient, aux, curvature_damping)`. `None` means the geometry's own default (`DEFAULT_CURVATURE_DAMPING = 1e-6`, in `geometry.py`). A non-None value where it cannot apply raises. **No field on the geometry classes** | C.7, C.9, C.13, §L |
| I-23 | agree; never force a minimal frame. Add a **full** `newton_cg` end-to-end test on a Tucker-slack and a TT-slack frame | New T25, measured in F28. On the TT-slack frame, exact+TR converges in 34 steps while GN+Armijo does not converge in 80 | T25 |

### First-principles proposals adopted

| proposal | what v5 does | where |
|---|---|---|
| **Uniform output hygiene first** | New slice **S0c**: every uniform transpose wrapper that holds the frame's masks re-masks its output. That is the six uniform kinds' `transpose`, the six public `utv_*_transpose` and the six `ut3_*_corewise_*transpose`. Measured in F34: today's padded output reaches 4.1e7 (COREWISE, small) to 3.0e12; after re-masking it is 0. The real block is **bit-identical**, and the overhead is 0.08–0.6 %. The two-hole's θ̂ mask then becomes defence in depth, and it stays, with T17/T24 | C.0, H |
| **Production kernels hold the GN kernels' standard** (P5 rephrasing) | The dotted kernels use the same form as the GN kernel they differentiate: a bidiagonal recurrence where the input jet is affine, an order scan where the convolution is full, and otherwise the GN counterpart's form (dense `trs` in `compute_dxi_tilde_jets` and in the per-chunk TT assembly). The dense-`trs` **prototype** two-hole is not that standard. Cost is re-based on it (F33): **exact/GN = 2.73× (d = 4, r = 16, W = 2000, order 3) and 2.98× (d = 5, r = 8, W = 300, order 2)**, against 3.1–4.1× for the prototype at the same sizes across runs (`c35`, `c35b`). Most of the saving comes from a product-rule **factored TT assembly** (2.0× on that term, 3e-15) | C.4, E |
| **Grouped `contract` strings** for every new kernel that carries W/K/C, including the lift (v4 had `'...'`) | Every string is listed in D. Each is checked against `np.einsum` on the generic multi-axis point, with its HISTORICAL supplement (F35) | C.2–C.4, D |
| **jax-absent discipline** | Every new pytree registration sits under `if jax_available:`. `test_jax_absent.py` **constructs** each new class in the jax-hidden subprocess and runs `newton_cg(hessian='exact', use_jit=True)` (P4). String annotations are dropped in favour of direct imports. The import DAG is stated and verified acyclic, with 17 new edges (F35) | C.2, C.9, T28, §M |
| **Cleaner stats / history** | `stats['newton_steps']`, `stats['newton_accepted']` and `stats['stop_reason']` replace the overloaded `stats['newton']`. One `step` display segment renders `ls k …` (line search; byte-identical to today) or `TR Δ … ✓/✗`. History rows carry `step_control`, so consumers filter by mode rather than by `None` | C.9, C.14, K |
| **Explicit `step_control` / `tr_radius0`** on both `newton_cg` layers, and also `g0_reference`, `hessian` and `curvature_damping` | Explicit keywords; no `**kwargs` pass-through for the new options | C.9, C.13 |
| **New section "Downstream (T3Polynomial) update"** | Every edit with file:line; estimate a half-day | §K |
| **Not adopted (Nick):** bundling the kind-operation arguments into a `LocalContext` | Kept positional, as in v4 | — |

### The critic's v5 pre-checks (`critic_prechecks_v5.md`), and how each is handled

| # | finding | v5 resolution | evidence |
|---|---|---|---|
| **P1a** | On COREWISE the zero reference is identically 0: the map is multilinear in the cores. Today's fallback (`backend/optimizers.py:716`) would silently use 1.0 | **`g0_reference=None` means the geometry's own reference**, mirroring `curvature_damping=None`. The manifold geometries define a zero-point frame, so `None` resolves to `'zero'`. The corewise geometries define none (`zero_point_frame` returns `None`), so it resolves to `'warm_start'`. An **explicit** `'zero'` on a corewise geometry is a structural `ValueError` (the n16 no-effect rule). The resolved kind is recorded in `NewtonInfo.g0_reference`. Also fixed: today an explicit `g0norm_newton=0.0` crashes with `ZeroDivisionError` in the forcing term [V `c29`]. Non-positive overrides now raise | C.15, T26; F26 |
| **P1b** | The zero reference is not intrinsic. It moves up to 1.6× under an isometry of the problem, because the frame at X = 0 is LAPACK's fixed completion | Wording softened everywhere (C.15, G): the zero reference is a **deterministic convention for the data-gradient scale at these ranks, good to an O(1) factor**, not a measurement of an intrinsic quantity. That is harmless for a relative stop. New fact F32: any *coordinate-aligned* completion, LAPACK's included, **collapses to exactly 0** when the data has no energy on the prefix coordinates. The canonical frame below avoids that | C.15; F32 |
| **P1c** | The default breaks the O2 ragged-vs-uniform pairing: zero references differ by layer by 8–33 %, and the paired losses drift to 5.2e-4 / 3.9e-3 against a 1e-7 threshold | **Root-caused and fixed by construction, not by pinning the tests.** The zero reference is evaluated at a **canonical zero-point frame** built from the frame's *structure*: a seeded Haar Tucker completion (host numpy, one draw per sharing group), identity-prefix TT cores, and `L[-1] = R[0] = 0`. Both layers build the same real blocks, so the reference is **layer-independent** to 3.9e-16. All **44 O2 pairs**, re-run with each layer's own reference (d ∈ {3, 2}, unshared and shared, every kind and weight spec), agree to **2.6e-9** (threshold 1e-7) [V `c37b`]. The byte-close O2 copies need **no edit**. Audit of every other ragged-vs-uniform pairing: C.15 | C.15, T26; F30, F31 |
| **P2** | the rename is clean; blast-radius list | Adopted as the S0e checklist (C.13) | — |
| **P3** | T3Polynomial passes `g0norm_newton=None` to *mean* warm start. Under v5, `None` keeps its "no override" meaning, so that call would silently flip to the zero reference | §K changes every such call site to `g0_reference='warm_start'`: `t3polynomial/fitting.py:195-197`, `:338`, `:498` and `tests/test_fitting_selection.py:89`. It also rewrites the test that would otherwise test nothing (`:91`). A sentinel default that would make an explicit `None` raise was considered and rejected: it would make `None` mean something other than "unset" | §K |
| **P4** | jax-absent: construct, do not just import | T28 | — |
| **P5** | "banded recurrences only" contradicts today's production code | Restated as the house rule (`deferred_and_rejected.md:85-96`): the dotted kernels follow the GN counterparts' forms (above) | C.4, E |

### New evidence this round

| # | finding | script |
|---|---|---|
| F26 | g0 interplay on F19's problem (probe_derivatives, restart rank 2 → 3). Zero reference 7.01e4 vs warm start 2.07e4 (ratio 0.30). GN+Armijo: 36 vs 39 steps at `gtol_rel` 1e-8; 13 vs 15 at 1e-3. Exact+TR at power 0.5: 9/27 vs 10/28 (outer/hvps); at power 1.0: 7/24 vs 8/23. 0 rejections throughout. COREWISE: zero reference 0.0; today's `g0norm_newton=0.0` raises `ZeroDivisionError` | `c29` |
| F27 | E2's three apply restarts under each reference, at forcing power 1.0 and 0.5. Warm/zero ratios are 1.9e-2, 1.6e-14 and 3.1e-1. **Noise-free (B):** the zero reference correctly declares the start converged (0 steps). Under the warm start the start already sits at the floor (`‖g‖ = 2.7e-10`); the run makes 3–4 outer steps with a rejection and ends unconverged. **Above target (A):** GN+Armijo never reaches 1e-8 under either reference. At `gtol_rel` 1e-3 the zero reference stops it in 31 steps, while the warm start exhausts 80. Exact+TR converges under both, in 17–23 outer steps | `c29b`, `c29c` |
| F28 | I-23: full exact+TR from non-minimal points, with no reduction. Tucker slack (frame down `[2,4,2]` vs up `[4,4,4]`): 20 outer, 4 rejected, 78 hvps, converged; GN 25. TT slack (frame left `[1,4,9,1]` vs right `[1,4,4,1]`): **34 outer, 9 rejected, 272 hvps, converged to 2.2e-12; GN+Armijo does not converge in 80** (1.1e-7). ε = 0 and ε = 1e-6‖X‖ are identical | `c30` |
| F29 | I-7 on T16's problem: power **1.0 → q-quadratic** (`e_{k+1}/e_k² ≤ 0.92`); power **0.5 → superlinear of order ≈ 1.5** (`e_{k+1}/e_k^{1.5}` 0.43–0.98 in `[1e-6, 1e-1]`). Seed 2 at power 0.5 reaches the resolution floor at `e = 1.5e-7`, and the m22 rule stops it `objective_floor` at its first rejection (`|pred|/res = 3.9e-3`) | `c31`, `c31b` |
| F30 | Per-layer LAPACK zero references diverge: ragged vs uniform differ by 0.56–1.93×, and 6 of 16 seeded `newton_cg` pairs drift (corroborates P1c). With **one** reference on both layers, 0 of 16 drift | `c32`, `c32b`, `c32c`, `c32d` |
| F31 | The canonical zero-point frame is valid (orthogonality residual 0, base point 0, Π idempotent) at d = 1, 3, 4 and shared. Ragged == packed uniform to ≤ 3.9e-16. The 44 O2 pairs agree to 2.6e-9 | `c33`, `c33b`, `c36`, `c37`, `c37b` |
| F32 | On data with no energy on the leading coordinates, the zero reference collapses to **exactly 0** under both the LAPACK and the identity-prefix completions. The seeded Haar completion gives 1.6e1 (seeds 0–5: 9.5–24). On generic data, the Haar seeds 0–5 give 4.4e1–7.0e1 against 7.0e1 (LAPACK): the O(1) spread of P1b | `c38` |
| F33 | Cost re-based on the house-standard forms (P5). Profile of the prototype at (d = 4, r = 16, W = 2000, order 3): TT assembly **55 %** of the two-hole; dotted forward 14 %; dotted adjoint sweeps 15 %; undotted tildes 7 %; dotted dx̃i 7 %; ν re-sweep 1 %. The **factored** TT assembly (pair intermediates plus the product rule; 6 W-summed finals instead of 9) is 2.0× faster on that term and equal to 3e-15. The dotted adjoint step written as the product rule of the production `_adj_tilde_step` (prop as a recurrence, src as an order scan) equals the dense prototype to 3.7e-16. The whole standard-form two-hole equals the prototype to 2.6e-15. **Exact/GN: 2.73× (d = 4, r = 16, W = 2000, o = 3); 2.98× (d = 5, r = 8, W = 300, o = 2)** (numpy CPU; the dotted adjoint is timed as three production `_adj_tilde_step` calls, a proxy for the fused step; run-to-run noise ~10 %) | `c35`, `c35b`, `c35c`, `c35d` |
| F34 | S0c: the uniform kind transpose's padded output is 4.1e7 (COREWISE) / 3.2e5 (MANIFOLD) on a small case, and 3.0e12 / 4.4e8 on a larger one. After re-masking it is 0; the real block is bit-identical; the re-mask costs 0.08–0.61 % of the transpose | `c39` |
| F35 | 25 new grouped strings (lift, frame factors, tangent zipper, two-hole pieces, factored assembly) equal `np.einsum` on W = (2,3), K = (2,2), C = (2,3); each supplement is `''` or `'C'`. The backend import graph plus the 17 planned edges is **acyclic**; `optimizers.py` imports only `common` and `regularization` (plus the top-level `corewise`) | `c34` |

**Round history.** The CHANGES of rounds 1–4 are in `SCR/plan_v2.md` … `SCR/plan_v4.md`. Every fix they record
(M1–M4, m1–m30, n1–n17) is carried forward here. Where round 5 overrode one, the table above says so: m23 (shadow flag
→ raising override), m24 (geometry field → solver parameter), m25 (no public change → public constructor), and the
v4 I-7 mode-dependent default.

---

## Verified facts this plan rests on

F1–F25 are unchanged from v4. F26–F35 are new; see the table above.

| # | claim | script | result |
|---|---|---|---|
| F1 | `T_i := ⟨L_i, H_i⟩ = tt_zipper_right_to_left(L, R)[i+1]` for `i ≤ d−2`. Centers: `H_i = L_i T_i`. `S_iᵀ := ⟨O_i, H_i⟩ = V_i^X U_iᵀ`. σ(S_i) is the mode-i Tucker spectrum; σ(T_i) is the bond-(i+1) TT spectrum | `c1` | ≤ 3e-12; spectra = dense unfoldings |
| F2 | Lift identity `tv_to_dense(t3_corewise_frame(U, L), Lv) == tv_to_dense(frame, v)`, gauged and ungauged, minimal frame | `c2`, `c3`, `c4` | 6.0e-15, 1.2e-15 |
| F3 | The same holds on non-minimal frames, Tucker slack (`Sᵀ` wide) and TT slack (`T_1` tall). The pinv is an exact one-sided inverse on every `t3_orthogonal_representations` frame | `c7`, `c8`, `c12` | 9.4e-16, 9.2e-16 |
| F4 | Zero-padded `resize`: the guarded pinv is finite. At X = 0 the lift is 0, except the undamped last TT slot | `c7` | finite; 0.0 |
| F5 | **Block-sum oracle:** `K_c θ̂ = Bᵀ t3_*_corewise_transpose(c, sample, tv_to_t3(t3_corewise_frame(θ), θ̂))`, vs central FD of the one-hole; symmetric | `c2`, `c9` | 1.1e-9 (FD-limited); ⟨a,Kb⟩ = ⟨b,Ka⟩ to 13 digits |
| F6 | Exact = GN + ⟨θ̂,K_{Aᵀr}θ̂⟩ − ⟨θ̂,K_{grad f}θ̂⟩, vs FD along `tv_retract` | `c4` | 252.85288420 vs 252.85288182 |
| F7 | `K_N L p_vert = 0` | `c4`, `c10` | 1.3e-11 vs 1.3e4 |
| F8 | Under the orthogonal Π, `hvp(p_vert) ≠ 0` (DP §4.6 is wrong) | `c10` | 0.61 |
| F9 | apply two-hole from existing kernels; the lean sweep is identical at both frames | `c5` | 4.7e-16 |
| F10 | probe two-hole and T3–T3 two-hole, C = (2,) | `c6` | ≤ 4.4e-16 |
| F11 | apply_derivatives two-hole (order 2), C = (2,) | `c13` | ≤ 2.5e-16 |
| F12 | `len_*` sufficiency of the jet strings | `c11`, `c17` | §D |
| **F13** | **probe_derivatives two-hole**, the v3 recipe: xi/mu reused from the model sweep; ν re-swept through G; no corewise η or dx̃i; orders 0–3; K ∈ {(2,), (2,3)}; C ∈ {(), (2,)}; d ∈ {4, 3, 1} | `c22`, `c14`; critic `check_c14_extra.py`, `check_c14_d1.py` | ≤ 1.0e-15 |
| **F14** | **Shared exact**, sharing `(0,0,1)`, order 2. The tied lift satisfies `DΦ[Lv] = Ev`. Exact quadratic vs FD along the **shared** retraction | `c15` | lift 5.3e-16; FD O(h²) |
| **F15** | Padding: θ and θ̂ masked on entry → real block exact. Padded output is don't-care. An unmasked θ̂ corrupts the real block | `c16`, `c19`, critic `check_c16_boundary.py`, `check_garbage_theta_hat.py` | 5.5e-17; unmasked 2.6e3 |
| F16 | At the **manifold** frame the corewise sweep and adjoint state are T/S transforms of the manifold ones (`ν_c = Tν`, …). At the corewise frame `ν_c = Tν` fails | `c18`, `c26` | ≤ 9.5e-16; 6.2 |
| F17 | `‖X‖² = Σσ(S_0)² = Σσ(T_i)²` | `c18` | ≤ 1.4e-15 |
| **F18** | M4 on a real `UniformTuckerTensorTrain`: θ, θ̂ masked with the frame's `(um, lm)` | `c19` | 5.5e-17; mutation 2.6e3 |
| **F19** | **E2 on probe_derivatives** (order 2, ω = `[1, .5, .25]`, restart rank 2 → 3, target 4, 5 % noise): GN+Armijo 39 steps / 130 hvps; exact+TR **8 / 23, 0 rejected** (at power 1.0; at 0.5: 10 / 28, F26) | `c24`, `c29` | — |
| F20 | prototype cost (dense trs): 3.0–3.4×. **Superseded by F33** | `c25` | — |
| **F21** | TR floor rule (m22) on both degenerate cases → `objective_floor` | `c21` | — |
| **F22** | T16 calibration (target `(4,4,4)`, rank-2 fit, W = 200, 1 % noise): GN ratios 0.56–0.68; exact (power 1.0) `e_{k+1}/e_k² ≤ 0.92` in `[1e-8, 1e-1]` | `c20`, `c31` | — |
| **F23** | Uniform frame up/left masks ⊆ point masks; strict subset at TT slack | `c23` | — |
| **F24** | T16: every step with `|pred| > 1e3·res` has `|1−ρ| ≤ 1e-3` | `c27` | — |
| **F25** | The frame argument makes the real extent reachable (m29); numpy float32 → float64 residual (m30) | `c28` | — |
| F26–F35 | the table in CHANGES above | `c29`–`c39` | — |

### The critic's experiments [E] (kept)

| # | claim | script |
|---|---|---|
| E1 | Exact matches FD to 5–6 digits at a small bond σ ∈ {1e-2, 1e-3, 1e-4}. The validity radius is ~σ, and the Hessian is indefinite there | `check_small_sigma*.py` |
| **E2** | Zero-padded restart, apply, d = 3: ε ∈ {0, 1e-6‖X‖} gives **13 / 69, 0 rejected**, vs GN+Armijo 54 and GN+TR 52 / 352. ε = 1e-2‖X‖: 17 / 173. Δ-coupled ε fails | `check_restart_tr*.py` |
| E3 | float32 cancellation is benign | `check_float32_cancellation.py` |
| E4, E5 | `N ≡ 0` at d = 1 and at maximal rank | `check_d1_normal.py`, `check_full_rank_N0.py` |
| E6 | Uniform MANIFOLD d = 1 optimizers already raise | — |
| E7 | Today's uniform COREWISE GN Hessian leaves 2.7e5 in θ's padded boundary slots | `check_onehole_boundary_leak.py` |
| E8 (P1a) | COREWISE zero reference ≡ 0 (ragged and uniform) | `check_zero_reference_geometries.py` |
| E9 (P1b) | The zero reference moves ≤ 1.6× under a mode rotation; a nonzero start's ‖g‖ is invariant | `check_zero_reference_invariance.py` |
| E10 (P1c) | Per-layer zero references break the O2 pairing (5.2e-4 / 3.9e-3) | `check_zero_reference_layers.py`, `check_zero_reference_o2_pairs.py` |

---

## A. Scope and non-goals

**In scope**, in build order (H):

0. **Breaking-release groundwork** (S0c–S0e), which is independent of the curvature:
   - uniform transpose output hygiene;
   - the zero reference (`g0_reference`) and the canonical zero-point frame;
   - the stats keys `newton_steps` / `newton_accepted` / `stop_reason`;
   - non-positive `g0norm_*` overrides raise;
   - the public frontend `least_squares_problem` (which replaces `_setup`);
   - the frontend `GaussNewtonModel` → `LocalModel` rename.
1. The lift `L`, its adjoint `Lᵀ`, and the damped filter: ragged, uniform and shared, over C and K stacks.
2. The T3–T3 two-hole `K_{grad f}` (`t3_linalg` / `ut3_linalg`).
3. Two-hole corewise sweeps `K_c θ̂` for all six kinds, **`probe_derivatives` first**. Each is a standalone backend
   function plus the **required** `SamplingKind.corewise_hessian` (sixth operation).
4. `LocalModel` exact mode (`hessian='exact'`):
   - compiles once;
   - one misfit transpose per Newton step;
   - static-safe companions;
   - `curvature_damping` as a solver parameter.
5. Steihaug–Toint CG plus a trust region in `newton_cg`, the TR-aware `step` display segment, and `stop_reason`.
6. Frontend: `LocalModel` exact mode, the factories, and `optimizers.newton_cg` wiring; `_bgeom → from_frame`.
7. Minimal diagnostics (I-13).
8. Tests, docs, CHANGELOG, the research-repo measurement plan (S0a / S0b), and **the T3Polynomial update (§K)**.

**Non-goals.**

- No AD through the T3-SVD (L3). The jvp and polynomial-FD routes are oracles only.
- No preconditioner, LM damping or NL2SOL β.
- **No normal-space sector norms**: deferred (I-13; ledger paragraph in G).
- No curvature for `gradient_descent` / `mc_sgd` / `adam`. `gradient_descent` does get `g0_reference` (C.15).
- No stacked *optimization* (`_require_unstacked`, `optimizers.py:269-298`); the C/K stacks live at the model level.
- No nonlinear-residual Hessian term (T3Polynomial's `Σ rᵢ∇²rᵢ`).
- No `fit(...)` facade.
- No objective-plateau stop and no η floor. G0P §5 recommends both; they are recorded as open (J-12), not built.
- No change to how a zero **start** chooses its frame. Only the *reference* uses the canonical frame (J-13).
- No `LocalContext` bundling of the kind-operation arguments (Nick: a later rethink).
- No change to rank continuation itself.

---

## B. Math-to-code map

Notation: `X = Φ(θ)`; frame `(U, O, L, R)` = `T3Frame.data`; gauged coordinates `v = (V_i, H_i)`;
`Π = tv_orthogonal_gauge_projection` (`tv_operations.py:83-137`).

| math | code (new unless cited) | shapes (per C element) |
|---|---|---|
| raw gauge `θ = (U, L)` | `geom.base_point(frame)` (`geometry.py:183-189`, `:274-276`, `:378-380`, `:483-485`) | — |
| corewise frame of θ | `t3_corewise_frame` (`fv_conversions.py:30-41`); `ut3_corewise_frame` (`ufv_conversions.py:122-137`) | — |
| real extent of θ (uniform) | `(frame_data[5][0], frame_data[5][2]) = (um, lm)`, ⊆ the point masks (F23) | host bool |
| `T_i` | `tt_zipper_right_to_left(L, R)[i+1]` (`tt_operations.py:192-199`), `i ≤ d−2` | `(rL_{i+1}, rR_{i+1})`, `rR ≤ rL` |
| `H_i = L_i T_i`, `S_iᵀ = ⟨O_i, H_i⟩` | `fv_center_factors` (extracted from `sharing.py:546-554`) | `(rL_i,nU_i,rR_{i+1})`, `(nD_i,nU_i)` |
| `‖X‖_HS` | `√Σσ(S_0)²` (F17) | — |
| filter `f(σ) = σ/(σ²+ε²)`, clipped, `f(0) = 0` | `linalg.filtered_reciprocal`, `linalg.damped_pseudo_inverse` | — |
| `L`: `U̇_i = (S_iᵀ)^{+ε}V_i`, `Ġ_i = H_i ×₃ T_i^{+ε}` (`i ≤ d−2`), `Ġ_{d−1} = H_{d−1}` | `tv_corewise_lift` | `(nD,N)→(nU,N)`; `(rL,nU,rR)→(rL,nU,rL')` |
| `Lᵀ` | `tv_corewise_lift_transpose` | inverse shapes |
| `K_c θ̂ = d/ds ∇_θ⟨c, S(Φ(θ+sθ̂))⟩` | `t3_{probe,apply,entries}_corewise[_derivatives]_hessian[_from_sweep]`; `SamplingKind.corewise_hessian` | raw `K+C` |
| `Z = grad f` | `tv_to_t3(frame, g_misfit)` / `utv_to_ut3` | doubled rank |
| `∇_θ⟨Z, Φ(θ)⟩`; `K_Z θ̂` | `t3_inner_product_corewise_transpose`; `t3_inner_product_corewise_hessian[_from_sweep]` | raw `C`; `K+C` |
| Riemannian Hessian | `LocalModel.hvp` (exact) = `Π[𝒥ᵀ𝒥Πp + Lᵀ K_N LΠp] + H_R p` | `K+C` |
| COREWISE exact | `GN_corewise + K_{Aᵀr}` (no lift, no gradient term) | — |
| shared lift | `_tied_filtered_solve` (refactor of `sharing.py:644-664`) | — |
| identity regularizer | `H_R = λΠ` is already exact; the curvature uses the **misfit** gradient only | — |
| **zero reference** | `‖Π 𝒥ᵀ(−ω²y)‖` at `geom.zero_point_frame(frame)` (C.15) | scalar |

**Lift exactness.** On every frame from `t3_orthogonal_representations` the lift is exact (F2, F3).

**Gauge laws** (F7, F8): `K_N L p_vert = 0`, `hvp(p) = hvp(Πp)`, `Π hvp = hvp`.

**Flat slots (I-12, the reasoning).** The normal-space term is the second fundamental form of the representable set.
Where that set is *flat* there is no normal direction to curve into, so `N` has no component there:

- **A slot at maximal rank** — a Tucker rank `n_i = N_i`, or a TT rank equal to the product of its neighbours — puts no
  constraint on its unfolding. Locally the representable set is an open subset of a linear space in that slot's
  directions.
- **At d = 1 every slot is maximal at once.** A one-mode T3 is a vector in a rank-`n` subspace, and the set is open in
  `ℝ^N`. So `N ≡ 0` (E4).
- **At maximal rank in general**, the same holds slot by slot (E5).

The formula needs **no special case**: the lift's filtered pinvs and the two-hole sum give `Lᵀ K_N L = 0` there
numerically, and T20 asserts it. The only special handling is a *shortcut*. The MANIFOLD d = 1 geometry returns
`curvature_precompute = None`, so it skips work that would compute zero. That is a cost decision, not a correctness
one: d = 1 is special only because every slot is maximal at once.

---

## C. File-by-file changes

Signature conventions:

- one argument per line;
- three aligned columns;
- one return element per line;
- `len=d, elm_shape=…`;
- one-shot argument order, with the sweep appended (n1).

Signatures are expanded in full at implementation (n5). Docstrings cite the local `t4s.pdf` App. A and external
references (Absil–Mahony–Trumpf 2013; Psenka–Boumal arXiv:2011.13395; Steihaug 1983; Nocedal–Wright Alg. 4.1, 7.2).
They contain no `dev/` paths. **Annotations name the classes directly**: no string annotations, because the DAG is
acyclic (§M).

### C.0 Uniform output hygiene (slice S0c; precedes all curvature work)

**What.** Re-mask the output of every uniform transpose wrapper that holds the frame's or point's masks:

| wrapper | file:line | masks used |
|---|---|---|
| the six uniform kinds' `transpose` | `uniform_fitting.py:104`, `:124`, `:151`, `:173`, `:195`, `:224` | `ufv_variation_masks_over_stack(frame_data[5], C, K_new)` |
| `utv_{probe,apply,entries}_transpose`, `utv_*_derivatives_transpose` (public; they *attach* masks today but leave garbage in padding) | `utv_sampling.py:432-555` | the masks they already attach (`_gauge_masks_over_Knew`, `:419-429`) |
| `ut3_*_corewise_transpose`, `ut3_*_corewise_derivatives_transpose` | `ut3_sampling.py:107-256` | the point masks they already mask θ with (`ut3_apply_masks(data)`), via `ut3_apply_masks` on the output |

**What stays bare.** The `utv_*_transpose_from_sweep` seam functions (`utv_sampling.py:191-223`, `:321-365`) stay bare
by contract. Their docstrings already say "the caller's project attaches the variation masks". Their only callers are
the kinds, which now re-mask.

**Why first.** The curvature's θ̂ masking (M4) closes one instance of a general gap:

- today's uniform COREWISE GN Hessian puts 2.7e5 into padded boundary slots [E7];
- the kind transpose's raw padded output reaches 3.0e12 on a moderate case [V `c39`].

The house contract ("garbage don't-care, consumers mask on entry") makes this harmless today. It also means every
*new* consumer has to remember to mask. Re-masking at the producer makes the CG state clean by construction. The
two-hole's θ̂ mask (C.10) then becomes defence in depth, and it stays, with T17/T24 and the mutation check.

**Equivalence.**

- A re-mask is one multiply by a host-numpy mask. The real block is **bit-identical** (F34), so
  `to_uniform → op → to_ragged == op_ragged` is unchanged and every existing uniform test passes unchanged.
- Overhead is 0.08–0.61 % of the transpose (F34). Under jit the masks are constants, so the multiply fuses.
- T27 pins the exact output masks and the bitwise real block (`testing_strategy.md` §masks).

### C.1 `backend/linalg.py` — the filtered reciprocal

```python
def filtered_reciprocal(
        s:      NDArray,                     # singular values, shape=C+(k,), descending
        eps:    typ.Optional[NDArray],       # damping, shape=C (broadcast over k); None -> exact clipped 1/s
        rcond:  typ.Optional[float] = None,  # relative clip vs s[..., :1]; None -> dtype eps * k
) -> NDArray:                                # f(s), shape=C+(k,); 0 where clipped

def damped_pseudo_inverse(
        M:      NDArray,                     # shape=C+(m, n)
        eps:    typ.Optional[NDArray],       # shape=C; None -> clipped pinv
        rcond:  typ.Optional[float] = None,
) -> typ.Tuple[
        NDArray,                             # M^{+eps}, shape=C+(n, m)
        NDArray,                             # s, shape=C+(min(m, n),)
]:
```

- The guard is a branch-free double `where`.
- `eps=None` reproduces `_tied_solve`'s filter bit-for-bit (`sharing.py:658-659`).
- Clip at ε = 0: a zero-padded slot's σ comes back as ~1e-17, and the clip is what handles restarts (E2).
- It is a thin batched SVD with no kept basis, so it is pad-transparent and is not `pad_safe_svd`.
- Dispatch is inferred.

### C.2 `backend/fv_operations.py` — frame factors, lift factors, the zero-point frame

```python
def fv_center_factors(
        frame:  typ.Tuple[
            typ.Sequence[NDArray],  # up_tucker_cores. len=d, elm_shape=C+(nUi, Ni)
            typ.Sequence[NDArray],  # down_tt_cores.   len=d, elm_shape=C+(rLi, nDi, rR(i+1))
            typ.Sequence[NDArray],  # left_tt_cores.   len=d, elm_shape=C+(rLi, nUi, rL(i+1))
            typ.Sequence[NDArray],  # right_tt_cores.  len=d, elm_shape=C+(rRi, nUi, rR(i+1))
        ],
) -> typ.Tuple[
    typ.Tuple[NDArray, ...],        # centers H_i = L_i T_i.  len=d,   elm_shape=C+(rLi, nUi, rR(i+1))
    typ.Tuple[NDArray, ...],        # tucker factors S_i^T.   len=d,   elm_shape=C+(nDi, nUi)
    typ.Tuple[NDArray, ...],        # bond factors T_i.       len=d-1, elm_shape=C+(rL(i+1), rR(i+1))
]:
```

- These are steps 1–2 of `fv_shared_frame_data` (`sharing.py:546-554`), extracted. `fv_shared_frame_data` calls it,
  and its existing doctest pins bit-identity.
- The strings are grouped: `'Caib,Cbc->Caic'` (center) and `'Caxb,Caub->Cxu'` (`Sᵀ`), both identifiable (F35).

```python
@dc.dataclass(frozen=True, eq=False)       # identity hash/eq: array fields (SharedFrameData precedent, sharing.py:445)
class LiftFactors:
    tucker_pinvs:  typ.Tuple[NDArray, ...]  # (S_i^T)^{+eps}. len=d,   elm_shape=C+(nUi, nDi)
    bond_pinvs:    typ.Tuple[NDArray, ...]  # T_i^{+eps}.     len=d-1, elm_shape=C+(rR(i+1), rL(i+1))
    tucker_svals:  typ.Tuple[NDArray, ...]  # σ(S_i), elm_shape=C+(nDi,)
    bond_svals:    typ.Tuple[NDArray, ...]  # σ(T_i), elm_shape=C+(rR(i+1),)
    eps:           NDArray                  # absolute damping used, shape=C -- a traced LEAF

def fv_lift_factors(
        frame:              typ.Tuple,      # (U, O, L, R) orthogonal frame, stack C
        curvature_damping:  float,          # RELATIVE: eps = curvature_damping * ‖X‖_HS, ‖X‖² = Σ σ(S_0)² (F17)
        rcond:              typ.Optional[float] = None,
) -> LiftFactors:
```

- **Registration**, in this module, at the bottom:
  `if jax_available: jax.tree_util.register_pytree_node(LiftFactors, …)`. Arrays are leaves; there is no aux
  (precedent `sharing.py:1081-1091`).
- **I-1 (settled):** derive the factors by zipper, never store them on the frame, never re-SVD H/O/R.

**The canonical zero-point frame** (C.15; P1b/P1c, F31, F32):

```python
ZERO_FRAME_SEED = 0                       # the Haar draw's seed: a fixed CONVENTION, not a tunable

def fv_zero_point_frame(
        frame:   typ.Tuple,                         # a frame whose STRUCTURE (core shapes) to copy; stack C = ()
        groups:  typ.Tuple[typ.Tuple[int, ...], ...] = (),   # sharing groups: ONE up factor per group
        seed:    int = ZERO_FRAME_SEED,
) -> typ.Tuple[                                     # an orthogonal frame AT THE ZERO TENSOR, same core shapes
    typ.Tuple[NDArray, ...],                        # up:    seeded Haar rows (sign-fixed QR), one draw per group
    typ.Tuple[NDArray, ...],                        # down:  identity-prefix orthonormal cores
    typ.Tuple[NDArray, ...],                        # left:  identity-prefix, left-orthogonal; L[d-1] = 0
    typ.Tuple[NDArray, ...],                        # right: identity-prefix, right-orthogonal; R[0] = 0
]:
```

- **Construction.** Host numpy, `np.random.RandomState(seed)`, whose legacy stream is frozen across numpy 1.22 ↔ 2.x
  (NEP 19). Each up factor is the sign-fixed QR of an `N_i × n_i` standard-normal draw, which is a Haar `n_i`-frame (`c38`
  drew `N × N` and kept `n` columns; both are Haar). Draws come in mode order, one per group of the geometry's
  canonical `groups` (singletons are their own group), at the group's first mode. The result is cast to the template's dtype and backend
  (`xnp.asarray`, inferred).
- **Why Haar for U only:**
  - The Tucker factor is the only part of the frame that meets the data directly.
  - A coordinate-aligned U collapses the reference to 0 when the data has no energy on its prefix coordinates (F32).
  - The TT cores act on rank indices, which carry no data orientation.
  - Identity-prefix TT cores keep the construction simple and exactly orthogonal.
- **Validity:** orthogonality residual 0, base point 0, Π idempotent, at d = 1, 3, 4 and shared (F31).
- **Stacked templates:** `C ≠ ()` raises `NotImplementedError` naming the optimizers' unstacked contract
  (`_require_unstacked`). The function exists for the reference, which is unstacked by construction.

### C.3 Lift and lift adjoint (`tv_operations.py`); T3–T3 (`t3_linalg.py`)

```python
def tv_corewise_lift(
        frame:          typ.Tuple,              # (U, O, L, R), stack C
        variations:     typ.Tuple[
            typ.Sequence[NDArray],              # tucker_variations. len=d, elm_shape=K+C+(nDi, Ni)
            typ.Sequence[NDArray],              # tt_variations.     len=d, elm_shape=K+C+(rLi, nUi, rR(i+1))
        ],
        lift_factors:   fv_operations.LiftFactors,   # stack C  (a direct annotation: no cycle, §M)
) -> typ.Tuple[
    typ.Tuple[NDArray, ...],                    # Udot. len=d, elm_shape=K+C+(nUi, Ni)
    typ.Tuple[NDArray, ...],                    # Gdot. len=d, elm_shape=K+C+(rLi, nUi, rL(i+1))
]:

def tv_corewise_lift_transpose(
        frame:            typ.Tuple,
        core_variations:  typ.Tuple[typ.Sequence[NDArray], typ.Sequence[NDArray]],   # raw, K+C
        lift_factors:     fv_operations.LiftFactors,
) -> typ.Tuple[
    typ.Tuple[NDArray, ...],                    # UNGAUGED frame coords, elm_shape=K+C+(nDi, Ni)
    typ.Tuple[NDArray, ...],                    # elm_shape=K+C+(rLi, nUi, rR(i+1))
]:
```

- **Grouped strings** (F35):

  | direction | slot | string |
  |---|---|---|
  | lift | Tucker | `'Cux,KCxn->KCun'` |
  | lift | TT | `'KCaib,Cbc->KCaic'` |
  | lift transpose | Tucker | `'Cux,KCun->KCxn'` |
  | lift transpose | TT | `'KCaic,Cbc->KCaib'` |

  All are identifiable without a supplement: the factors are C-only and pin C.
- The last TT slot passes through undamped. The same factors are used in both directions, which keeps `Lᵀ_ε K L_ε`
  symmetric for every ε.

**T3–T3 kernels, moved to `t3_linalg.py` (m8).**

- The rebasis and variations map bodies (`tv_operations.py:333-362`) and the ungauged core of
  `tv_project_t3_onto_tangent_space` (`:388-415`) become the private kernel `_inner_product_frame_transpose(frame, z)`.
- `tv_project_t3_onto_tangent_space` then becomes `tv_orthogonal_gauge_projection(frame, t3_linalg._inner_product_frame_transpose(frame, x), shared_data)`.
  This is bit-identical, and the existing tests pin it.

```python
def t3_inner_product_corewise_transpose(z, core_pair):        # ∇_θ⟨Z, Φ(θ)⟩, raw, stack C
def t3_precompute_inner_product_sweep(z, core_pair):          # (BU, G2, ZL, ZR, env), stack C
def t3_inner_product_corewise_hessian(z, core_pair, core_directions):            # raw K+C
def t3_inner_product_corewise_hessian_from_sweep(z, core_pair, core_directions, sweep):
```

- **The dotted zippers.** New `tt_zipper_tangent_left_to_right` / `_right_to_left` (`tt_operations.py`), with the
  closure-free module-level scan body `_tt_zipper_tangent_step` and three grouped terms:
  - `'KCij,Ciak,Cjal->KCkl'` (carry);
  - `'Cij,KCiak,Cjal->KCkl'` (G2-dot);
  - `'Cij,Ciak,KCjal->KCkl'` (direction core).
- **The pieces:** `BUd = 'Czn,KCun->KCzu'`, `G2d = 'Cazb,KCzu->KCaub'`, the `dG` term `'KCazb,Czu->KCaub'` (F35).
- The uniform twins live in `ut3_linalg.py`, with d-prefixed strings.

### C.4 `backend/sampling_derivatives.py` — the `probe_derivatives` two-hole (PRIMARY; F13, F33)

```python
def t3_probe_corewise_derivatives_hessian(
        ztildes:          typ.Sequence[NDArray],  # residual jets (ω² applied), len=d, elm_shape=(order+1,)+W+C+(Ni,)
        ww:               typ.Sequence[NDArray],  # len=d, elm_shape=W+(Ni,)
        pp:               typ.Sequence[NDArray],  # len=d, elm_shape=W+(Ni,)
        core_pair:        typ.Tuple,              # θ, stack C
        core_directions:  typ.Tuple,              # θ̂, elm_shape=K+C+(...)
        order:            int,
        chunk_size:       typ.Optional[int] = DEFAULT_CHUNK_SIZE,
) -> typ.Tuple[
    typ.Tuple[NDArray, ...],                      # raw dU, elm_shape=K+C+(nUi, Ni)        (W summed)
    typ.Tuple[NDArray, ...],                      # raw dG, elm_shape=K+C+(rLi, nUi, rL(i+1))
]:

def t3_probe_corewise_derivatives_hessian_from_sweep(
        ztildes, ww, pp, core_pair, core_directions, order,
        frame_sweep:      typ.Tuple,              # the model's (xi, mu, nu, eta) jets; only xi, mu are reused (F9)
        chunk_size:       typ.Optional[int] = DEFAULT_CHUNK_SIZE,
) -> typ.Tuple[typ.Tuple[NDArray, ...], typ.Tuple[NDArray, ...]]:
```

**The production standard (P5, settled as the house applies it).** Each dotted kernel takes the **form of the GN
kernel it differentiates** (`deferred_and_rejected.md:85-96`):

- a fused bidiagonal recurrence where the input jet is affine (`compute_mu_jets`, `compute_sigma_jets`);
- an order scan where the convolution is genuinely full (`compute_eta_jets`, `compute_deta_jets`,
  `_adj_sweep_scanned`);
- otherwise the GN counterpart's form: dense `trs` inside `compute_dxi_tilde_jets` (`:1632-1655`), and dense `trs` per
  W-chunk in `assemble_tt_variation_jets` (`:1942-1990` → `_tt_variation_jets_trs_core`, `:1748-1772`).

The `_trs` dense forms exist **only as test references**, in `tests/backend/_curvature_oracles.py`. They are never
called in production. This mirrors `tests/test_jet_recurrence.py`'s use of the existing `*_trs` twins.

The composition, step by step (F13 is the math; F33 the form):

1. **Undotted state at the corewise frame (recompute default, I-11):**
   - `xi_j`, `mu_j` reused from the model sweep;
   - `nu_j = compute_nu_jets(G, xi_j, trs)`, the recurrence (1–3 % of the two-hole, F33);
   - no corewise η and no corewise dx̃i (m26);
   - the undotted tildes from the production `compute_deta_tilde_jets` (`:1440`), `compute_tau_tilde_jets` (`:1605`)
     and `compute_sigma_tilde_jets` (`:1611`). These are order scans.
2. **Dotted forward:** the existing production kernels:
   - `compute_xi` → `build_input_jets`;
   - `compute_sigma_jets(Gdot, G, G, …)` (`:1002`, banded);
   - `compute_tau_jets` (`:1033`, banded);
   - `compute_deta_jets` (`:919`, order scan).
3. **Dotted adjoint-up lift:** `'tWCo,KCio->tWKCi'(z̃, Udot)`, a new grouped string, `'C'`.
4. **Dotted adjoint sweeps.** A new module-level scan body `_adj_tilde_tangent_step`, in the **GN form**: the product
   rule of `_adj_tilde_step` (`:1518-1560`).
   - **prop:** the three dotted pushthroughs (carry-dot, core-dot, xi-dot), each a **two-term reverse recurrence**
     fused into one GEMM, exactly as `_adj_tilde_step`'s prop.
   - **src:** the three dotted sources (edge-dot, core-dot, deta-dot), each an **inner order scan** over the edge order
     `r`, with `_adj_tilde_step`'s `W·r²` peak.
   - K placement follows v4's six-term table (carry / core / xid / edge / core / detd).
   - `_adj_tilde_step` cannot be called as is, because it infers `nf` from a C-only `P` (`:1530`). The tangent step
     takes `n_frame` explicitly.
   - F33 verified the form: the product rule written as three calls of the production `_adj_tilde_step` (K squeezed)
     equals the dense prototype to 3.7e-16.
   - Ragged and uniform scan-body variants follow `:1563-1582`.
5. **Dotted dx̃i:** the six-term product rule of `_dxi_tilde_jets_step` (`:1619-1629`), in its form (dense `trs`, as the
   GN kernel). Uniform d-prefixed twin.
6. **Assemblies.**
   - **Tucker:** `'tWKCa,tWCo->KCao'(etad, z̃) + 'uWKCa,uWo->KCao'(dxtd, w_jets)`.
   - **TT, factored by the product rule (F33).** For each of the three trilinear groups (σ̃, τ̃ and the deta source),
     form the undotted pair intermediate and its dotted twin once. Example, σ̃:
     `P_t = 'trs,rWCa,sWCi->tWCai'(trs, μ, ξ)`, and `Ṗ` from the two dotted pairs `'trs,rWKCa,sWCi->tWKCai'` /
     `'trs,rWCa,sWKCi->tWKCai'`. Then
     `dG = 'tWKCai,tWCb->KCaib'(Ṗ, σ̃) + 'tWCai,tWKCb->KCaib'(P, σ̃̇)`.
     - That is 6 W-summed final contractions instead of 9.
     - It is 2.0× faster on the assembly at (r = 16, W = 2000, order 3) and equal to 3e-15.
     - The binomial convolution stays dense `trs`, as in the GN assembly.
     - All strings are in D; each is `''` or `'C'` (F35).
   - **Chunking.** W-chunked through `_wchunked_sum` (`:1908-1930`), with module-level chunk functions
     `_tt_tangent_chunk_summed` / `_tucker_tangent_chunk_summed`.
   - **Memory.** S3 measures the peak with `memory_analysis`. If it exceeds 1.5× the GN assembly,
     `estimate_chunk_size` gains a `hessian=` argument.

**Padding.** The function is mask-free by contract: it assumes masked θ and θ̂ (F15). Masking is the uniform wrapper's
job (C.10).

### C.5 Plain probe, apply/entries, apply/entries derivatives (S10–S12)

As in v4 (F9–F11):

- Each dotted kernel takes the form of its GN counterpart (P5).
- Each has grouped strings.
- No corewise η or dx̃i.
- ν is re-swept through G.

Until each kind's slice lands, its `corewise_hessian` is the base `NotImplementedError` (ragged), or the `_UniformKind`
raising override (uniform).

### C.6 `backend/fitting.py` / `uniform_fitting.py` — `corewise_hessian`, the required sixth operation (I-14)

On `SamplingKind` (`fitting.py:305-412`), beside the five `NotImplementedError` operations (`:330-364`):

```python
    def corewise_hessian(
            self,
            core_directions,   # θ̂ = (Udot, Gdot) at base_point, elm_shape=K+C+(...)
            residual,          # RAW r; ω² applied HERE, as in transpose
            sample,
            frame,             # this model's frame -- the slot forward/transpose take; uniform reads frame[5] (m29)
            sweep,             # this model's precompute(frame, sample); its xi/mu half is reused (F9)
            base_point,        # geom.base_point(frame) (I-24)
    ):                         # -> raw (dU, dG) over K+C; MASK-FREE by contract (masking is the uniform wrapper's job)
        """The directional derivative, along the core direction θ̂, of ``transpose``'s corewise form
        ``θ ↦ ∇_θ⟨ω²r, S(Φ(θ))⟩``, at θ = base_point. Needed only for ``hessian='exact'``."""
        raise NotImplementedError(
            "%s does not implement corewise_hessian(): the directional derivative of the corewise transpose, which "
            "hessian='exact' needs. Subclass a concrete kind (ApplyKind, ProbeKind, the *DerivativesKind) to inherit "
            "it, or implement it for a new operator; a kind without it still works with hessian='gauss_newton'."
            % type(self).__name__)
```

- **The contract** is defined through an existing member, the corewise form of `transpose`. Its argument order extends
  `forward(v, sample, frame, sweep)` / `transpose(r, sample, frame, sweep)` (`:338-350`), with the geometry-derived
  point appended.
- **Required, not optional.** The `SamplingKind` docstring and the `uniform_fitting.py:60` comment are updated to name
  six operations. `has_block_sumsq` remains the house's only capability flag (it gates a display, not a mode); no
  `has_corewise_hessian` is added.
- **The structural check**, in `Problem.local_model(hessian='exact')`. Its *meaning* is Nick's identity check:
  "the method the kind would run **is** the base class's unimplemented one".
  - **The spelling.** The base method is decorated `@abc.abstractmethod`. `SamplingKind` stays a plain dataclass with
    no `ABCMeta`, so the decorator only sets the stdlib's `__isabstractmethod__` marker and changes nothing about
    instantiation [V: a frozen dataclass subclass that omits the method instantiates; its `type(k).op` carries the
    marker, an overriding subclass's does not]. The check is then
    `getattr(type(kind).corewise_hessian, '__isabstractmethod__', False)`. For every kind that does not re-declare
    a marked stub it agrees exactly with `type(kind).corewise_hessian is SamplingKind.corewise_hessian`.
  - **Why the marker and not the identity check verbatim.** (1) The verbatim form needs `backend/optimizers.py` to
    import `backend.fitting`, its first T3-specific import (§L). (2) The window stub below is a *different* function
    object: the verbatim form would let a uniform kind in the window **pass** the early check and fail only at its
    first curvature call. The marker form rejects it structurally too. Same meaning, earlier error.
  - **Where the identity form appears verbatim:** T23 asserts it, alongside the marker check.
  - **On error** it raises `ValueError("<Kind> does not implement corewise_hessian(), which hessian='exact' needs; it
    still works with hessian='gauss_newton'")`.
  - GN never consults it, so a third-party five-operation kind keeps working for GN.
- **The S4–S12 window.** `_UniformKind` (`uniform_fitting.py:69-91`) gains a one-line raising override:

  ```python
      @abc.abstractmethod                    # the early check rejects it structurally (same marker as the base)
      def corewise_hessian(self, *args):     # until this class's masked override lands: S8 / S10 / S11 / S12
          raise NotImplementedError("%s.corewise_hessian: the masked uniform two-hole lands in slice %s"
                                    % (type(self).__name__, _UNIFORM_HESSIAN_SLICE[type(self).__name__]))
  ```

  - Each uniform class overrides it in its slice. The `_UniformKind` stub and the `_UNIFORM_HESSIAN_SLICE` table are
    deleted in S12, with the last override.
  - Because `_UniformKind` precedes the ragged kind in the MRO (`[UniformProbeDerivativesKind, _UniformKind,
    ProbeDerivativesKind, …]`, verified), a uniform kind in the window **cannot** inherit the ragged method and run it
    unmasked. The stub raises if called, and its marker makes the early check reject the kind before any matvec.
  - **T23, with a mutation check:** delete the stub, the uniform kind inherits the ragged implementation, the early
    check no longer raises, and the test fails.
  - No release ships inside the window (H).
- **`ProbeDerivativesKind`** (`:627-652`) implements it in S3/S4, passing `chunk_size=self.chunk_size`.
- **`dev/OPEN_QUESTION_extension_surface.md`** §"What to watch" (`:130-137`) records: "2026.3.0 adds a **sixth required
  operation**, `corewise_hessian`. Kinds without it keep working for Gauss–Newton and fail early, with a structural
  error, for `hessian='exact'`." The note stays open.

### C.7 `backend/geometry.py` — the protocol members and the damping default

**New protocol members** (`backend/optimizers.py:60-98` lists them):

```python
    def zero_point_frame(               # REQUIRED (the g0_reference default path uses it; C.15)
            self,
            frame,                      # a frame of this geometry: the STRUCTURE to copy
    ):                                  # -> the canonical frame at the zero tensor, or None if this geometry has none
    def curvature_precompute(           # hessian='exact' only
            self,
            frame,
            misfit_gradient,            # Π 𝒥ᵀω²r (tied when shared) -- the model's ONE misfit transpose
            aux,                        # geom.precompute(frame)
            curvature_damping,          # RELATIVE float, or None -> this geometry's default
    ):                                  # -> per-frame curvature data; None iff the term VANISHES here
    def lift(self, frame, variations, curvature_data):                    # -> raw θ̂, K+C
    def lift_transpose(self, frame, core_variations, curvature_data):     # -> UNGAUGED frame coords, K+C
    def tangential_curvature(self, frame, core_directions, curvature_data):  # -> raw K_{grad f} θ̂, or None (COREWISE)
```

**The damping default** (I-22; module constant here, never in `optimizers.py`):

```python
DEFAULT_CURVATURE_DAMPING = 1e-6     # relative to ‖X‖_HS; E2, F19
```

**`ManifoldGeometryOps`** (`:146-236`) and **`UniformManifoldGeometryOps`** (`:314-422`):

- `zero_point_frame(frame)` returns `fv_zero_point_frame(frame, self.groups)`. The uniform twin is
  `ufv_zero_point_frame(frame_data, self.groups)` (C.10).
- `curvature_precompute`:
  1. `cd = DEFAULT_CURVATURE_DAMPING if curvature_damping is None else float(curvature_damping)`.
  2. At **d = 1** it returns `None`. That is a cost shortcut; the reasoning is in B's "flat slots".
  3. Otherwise it returns `ManifoldCurvatureData(lift_factors, z, z_sweep, sigma_min)`:
     - `lift_factors = fv_lift_factors(frame, cd)`;
     - `z = tv_to_t3(frame, misfit_gradient)`;
     - `z_sweep = t3_precompute_inner_product_sweep(z, base_point)`;
     - `sigma_min` is the real-slot minimum, with `s_g` included when `groups` is set (n12).
- **No new field**, so `with_sharing` and `from_point` are unchanged.

**`CorewiseGeometryOps`** (`:239-308`) and its uniform twin:

- `zero_point_frame` returns `None`: zero cores are a critical point of every corewise objective (P1a, E8).
- `curvature_precompute(…, curvature_damping)` returns `()`. A non-None `curvature_damping` raises
  `ValueError("curvature_damping has no effect on a corewise geometry (there is no lift to damp)")` (n16).
- `lift` / `lift_transpose` are the identity; `tangential_curvature` returns `None`.

**`ManifoldCurvatureData`.**

- `@dc.dataclass(frozen=True, eq=False)`.
- Registered under `if jax_available:` at the bottom of `geometry.py`, via `partition_static((lift_factors, z, z_sweep, sigma_min))`,
  with the value-hashed `StaticSkeleton` as aux (M3).

### C.8 `UniformManifoldGeometryOps.from_frame` (S6a, unchanged from v4)

- `from_frame(frame_data, plain_masks, sharing)` builds `var_masks = ufv_variation_masks(frame_data[5])` without
  re-orthogonalising.
- This is the X13 performance fix, in its own commit, asserting identical masks to `from_point`.
- No damping argument (I-22).

### C.9 `backend/optimizers.py` — `CurvatureState`, `LocalModel`, `Problem`, Steihaug, the trust region, `newton_cg`

```python
@dc.dataclass(frozen=True, eq=False)
class CurvatureState:
    """The per-model curvature companion (exact mode): a LEAF beside sweep / geom_aux."""
    misfit_gradient:  typ.Any   # Π 𝒥ᵀω²r -- computed ONCE per Newton step; LocalModel.gradient reads it (M2)
    geometry_data:    typ.Any   # geom.curvature_precompute(...): None => the curvature term vanishes here
```

- It is registered under `if jax_available:`, inside the existing `optimizers.py:773` block, via
  `partition_static((misfit_gradient, geometry_data))`.
- **`LocalModel`** (`:105-161`) gains `curvature: typ.Any = None`.
  - Exact mode is `curvature is not None`.
  - The term is present iff `curvature.geometry_data is not None`.
  - Both are static at trace time.
  - `_flatten_local_model` (`:799-801`) adds `m.curvature` to the `partition_static` tuple, and the unflatten passes
    it by keyword.

```python
    @property
    def gradient(self):                # exact: curvature.misfit_gradient (+ g_R); bit-identical to the GN path
    def hvp(self, p):                  # Π[𝒥ᵀ𝒥Πp + Lᵀ K_N LΠp] + H_R p   (GN mode: unchanged)
    def curvature_hvp(self, p):        # Π Lᵀ K_N L Π p   (raises if not exact)
    def quadratic(self, p):            # gn_quadratic(p) + curvature_quadratic(p)   (GN mode: = gn_quadratic)
    def curvature_quadratic(self, p):  # geom.inner(Πp, Lᵀ K_N LΠp)  (keeps K+C)
    # gn_quadratic / jacobian / retract: UNCHANGED (X1; T3Polynomial d07 relies on gn_quadratic = ‖Jp‖²)
```

The curvature body is the same as v4:

1. `th = geom.lift(frame, Πp, cd)`.
2. `K = kind.corewise_hessian(th, residual, sample, frame, sweep, geom.base_point(frame))`.
3. `K −= geom.tangential_curvature(frame, th, cd)`, unless that returns `None`.
4. `y = geom.lift_transpose(frame, K, cd)`.

`hvp` projects `𝒥ᵀz + y` once.

**`Problem.local_model(x_cores, sample=None, data=None, hessian='gauss_newton', curvature_damping=None)`**
(`:190-201`):

1. `hessian = str(hessian).lower()`, validated against `{'gauss_newton', 'exact'}` (n10).
2. **A non-None `curvature_damping` with `hessian='gauss_newton'`** raises
   `ValueError("curvature_damping is only meaningful with hessian='exact'")` (n16).
3. The GN build is unchanged.
4. Exact mode:
   - the C.6 structural check;
   - `g = geom.project(frame, kind.transpose(…), aux)` — the only misfit transpose;
   - `cd = geom.curvature_precompute(frame, g, aux, curvature_damping)` — the damping is forwarded as an opaque float
     or `None`, never interpreted here;
   - `curvature = CurvatureState(g, cd)`.

**Steihaug–Toint in the existing CG body** (I-5, settled).

- The state grows to `(p, r, d, rs, i, ok, lm, tol2, maxiter, radius, boundary)`.
- `boundary0 = rs0 < 0.0`. All branches use `where`; there is no inf arithmetic; the body is closure-free.
- `radius = inf` is bit-identical to today on eager numpy, and equal to tolerance under jit.
- `_cg_core(local_model, rhs, tol, maxiter, use_jit, radius)` is jitted as `_jitted(_cg_core, (4,))` and returns
  `(p, i, rs, ok, boundary)`. Blast radius: `optimizers.py:730`, `tests/backend/test_optimizers.py:218`, `:223`,
  `tests/test_dispatch.py:595`.

**`newton_cg`** (`:646-770`). New explicit keywords (no `**kwargs` anywhere on this layer):

```python
        hessian:            str = 'gauss_newton',             # or 'exact' (case-insensitive)
        curvature_damping:  typ.Optional[float] = None,       # exact mode only; None -> the geometry's default (I-22)
        step_control:       typ.Optional[str] = None,         # None -> 'line_search' (GN) / 'trust_region' (exact)
        tr_radius0:         typ.Optional[float] = None,       # None -> GN Cauchy length ‖g‖³ / gᵀH_GN g
        g0_reference:       typ.Optional[str] = None,         # None -> the geometry's own ('zero' on manifolds) (C.15)
        cg_forcing_power:   float = 0.5,                      # UNCHANGED default, both modes (I-7)
```

- `step_control` is validated against `{'line_search', 'trust_region'}`. Both combine with either `hessian`.
- `tr_radius0` with `step_control='line_search'` raises (n16 rule).
- The `g0norm_newton` / `g0norm_cg` overrides keep their names and their chained fallback. C.15 gives their
  interplay with `g0_reference`.

**The loop.**

1. `lm = problem.local_model(x, hessian=hessian, curvature_damping=curvature_damping)`. It is rebuilt only after an
   accepted step.
2. At iteration 0, resolve the references from `lm` (C.15).
3. `Δ0 = tr_radius0`, or else `‖g‖³ / max(gn_quadratic(g), tiny)`, falling back to `‖g‖` when not finite.
   `Δ_max = 1e4·Δ0`.
4. `p, i, rs, ok, hit = _cg_solve(..., radius)`. `radius = inf` in line-search mode.
5. Descent guard: in TR mode `p = −g·min(1, Δ/‖g‖)`.
6. `pHp = lm.quadratic(p)`. In exact mode, also `pKp = lm.curvature_quadratic(p)`.
7. **Line search** (`step_control='line_search'`): unchanged, with ρ taken from `lm.quadratic`.
8. **TR acceptance:**
   - one trial at α = 1;
   - `ρ = Δf/pred`, with `pred = slope + ½pHp < 0`; accept iff `ρ > 0.1`;
   - if `ρ < 0.25` or ρ is not finite, `Δ ← ¼‖p‖`;
   - if `ρ > 0.75` and the step hit the boundary, `Δ ← min(2Δ, Δ_max)`.
9. **Floor** (m22, F21), checked only on the rejection branch:
   - **`objective_floor`** iff `|pred| ≤ 10·ε·(‖ω⊙y‖·‖ω⊙r‖ + |f|)`, with `ε = _resolution_eps(lm.residual)` (m30);
   - else **`trust_region_floor`** iff `Δ < 1e-12·Δ0`.
   - An accepted step is never a stop condition.

**Stats and history (the breaking cleanup).**

- `stats = {'losses', 'newton_steps', 'newton_accepted', 'stop_reason', 'history'}`.
  - **`stats['newton']` is removed**, not aliased. It overloaded "CG solves" and "steps".
  - `newton_steps` counts CG solves (= history rows that took a step).
  - `newton_accepted` counts accepted steps (= `newton_steps` in line-search mode, minus Armijo-exhausted rejections
    under `'stop'`).
- `stop_reason` ∈ `'gradient' | 'max_newton' | 'line_search' | 'trust_region_floor' | 'objective_floor'`.
  - `'line_search'` is today's Armijo-exhausted `'stop'`.
  - The final converged row carries it too.
- Rejected TR steps count toward `max_newton` and emit a row with `tr_accepted=False` and the unchanged `x_cores`.
  `losses[k] == history[k]['objective']`.

**`NewtonInfo`** (`:602-635`). New fields, appended, all defaulting to `None`:

| field | meaning |
|---|---|
| `hessian`, `step_control` | the mode strings, documented in the history-row docstring (n4) |
| `g0_reference` | `'zero'` / `'warm_start'` / `'override'` |
| `tr_radius`, `tr_accepted`, `cg_boundary` | trust-region bookkeeping |
| `pKp`, `lift_sigma_min`, `lift_eps` | exact-mode diagnostics |
| `stop_reason` | as in stats |

- `cg_truncated = (not cg_converged) and (not cg_ok) and (not cg_boundary)`.
- In TR mode, `ls_steps = ls_failed = None` and `alpha = 1.0`. Consumers filter on `step_control`.

**`_resolution_eps(residual)`** — host-side and dtype-only: `np.finfo(dtype).eps`, maximised over the residual's
floating leaves (I-25).

### C.10 Uniform mirror

- **`ufv_lift_factors`** (`ufv_operations.py`):
  - mask once, then the polymorphic zipper and d-prefixed grouped strings;
  - padded σ = 0 gives f = 0;
  - real-slot `sigma_min` via host masks, with `s_g` included when shared.
- **`utv_corewise_lift` / `_transpose`:** mask on entry. The output carries the frame's masks and is re-masked.
- **`ut3_inner_product_corewise_*`** (`ut3_linalg.py`): mask Z and the direction θ̂ first.
- **Sampling wrappers** (`ut3_sampling.py`), e.g. `ut3_probe_corewise_derivatives_hessian` (M4):
  1. mask θ with `data`'s masks;
  2. **mask θ̂ with the same masks** (`ut3_apply_masks((θ̂_U, θ̂_G, data[2], data[3]))`; a K+C direction broadcasts
     against C masks, F23);
  3. pack;
  4. delegate to the polymorphic kernel;
  5. **re-mask the output**, which is the S0c hygiene applied to the new wrapper.
- **Kinds:** `UniformProbeDerivativesKind.corewise_hessian(v_sc, r, sample, frame_data, sweep, base_sc)`.
  - The masks come from the **frame argument**, `(um, lm) = frame_data[5][0], frame_data[5][2]` (m29, F25). They do
    not come from the sweep, which holds bare supercores, nor from `self.masks`, which is a superset at TT slack (F23).
  - This override replaces the `_UniformKind` stub for that class (S8).
- **`ufv_zero_point_frame(frame_data, groups)`.** It unpacks the real blocks
  (`ufv_conversions.ut3frame_to_t3frame`, `:141`), calls `fv_zero_point_frame`, and packs back with the frame's own
  padded sizes (`t3frame_to_ut3frame`, `:197`).
  - The output masks are the template's.
  - Its real blocks equal the ragged construction to ≤ 3.9e-16 (F31).
  - Host-side and eager only, because it unpacks to ragged shapes. It is called once per `newton_cg`, before the loop
    (C.15). It is never traced; under jit the reference is resolved eagerly, like today's `computed_g0norm`.

### C.11 Shared factors

Unchanged from v4:

- `_tied_filtered_solve[_transpose]` (bit-identical at `eps=None`);
- the tied lift `U̇_g = M_g^{+ε}[V_{i₁};…]`, replicated over the group (F14);
- the adjoint sums the group and then applies `Π_sh`;
- **N_sh** uses the tied misfit gradient;
- padded restarts: the new `s_g` levels are 0, so f = 0;
- shared COREWISE uses `project_mean(K_{Aᵀr}θ̂)`;
- `lift_sigma_min` includes `min(s_g)`;
- the shared retraction's second order is measured (F14), not proven.

**New:** `fv_zero_point_frame(frame, groups)` draws **one** Haar factor per sharing group and replicates it over the
group's modes. The zero point is therefore exactly tied (`t3_sharing_residual` guards the zero/zero case,
`sharing.py:255-259`). A ragged shared zero frame packed into the uniform layout reproduces the shared reference to
1.8e-16 (`c33b`). With the per-group Haar draw, every shared O2 pair holds (`c37b`).

### C.12 Safe mode — preconditions vs caveats

**No new enforced preconditions.**

- The frontend `hessian`, `curvature_hessian`, `quadratic` and `curvature_quadratic` take `_require_at_frame`
  (`fitting.py:101-118`).
- Every new error is **structural** (the stated house line):
  - an unknown `hessian` / `step_control` / `g0_reference` string;
  - a no-effect option (`curvature_damping` with GN or on corewise; `tr_radius0` with the line search;
    `g0_reference='zero'` on a corewise geometry);
  - a non-positive `g0norm_*`;
  - a kind without `corewise_hessian` in exact mode.

**Caveats** (documented, never enforced):

1. Exact at ε = 0 on numerically full-rank points; damped toward GN for σ ≲ ε.
2. A hand-built wide `T_i` gives a least-squares lift.
3. Shared: exact where `s_g > 0`.
4. N ≡ 0 on flat slots (B).
5. `objective_floor` is a resolution limit, not a convergence certificate.
6. **New:** the zero reference is a convention: a deterministic data-gradient scale at these ranks, good to an O(1)
   factor (P1b, F32).

### C.13 Frontend (`t3toolbox/fitting.py`, `t3toolbox/optimizers.py`, `t3toolbox/manifold.py`)

**The rename (I-4, slice S0e — mechanical, before any curvature).** `fitting.GaussNewtonModel` → **`fitting.LocalModel`**.

- **Not aliased**, following the 2026.2.0 precedent: "`fitting.UniformGaussNewtonModel` is removed… It is
  deliberately not aliased."
- Same name, two levels: the backend `LocalModel` is not re-exported (`t3toolbox/__init__.py:39-48`), so
  `t3toolbox.LocalModel` collides with nothing (`naming_conventions.md` §"Sharing a name with the backend twin is
  fine").
- **Checklist (critic P2):**
  - `t3toolbox/fitting.py` (27);
  - `tests/test_fitting.py` (10);
  - `tests/test_dispatch.py` (8);
  - `CHANGELOG.md` (6; history entries stay as written, plus the new entry);
  - backend `optimizers.py` docstrings (5);
  - `docs/fitting_and_optimization.md` (4);
  - `docs/contributor/parameters_not_closures.md` (4);
  - `CLAUDE.md` (4);
  - `tests/backend/test_optimizers.py` (3);
  - `examples/fit_hilbert_from_apply_derivatives.py` (3);
  - `t3toolbox/__init__.py` (2: import and `__all__`);
  - `docs/release_notes.md` (2);
  - one each in `tests/test_sharing.py`, `tests/test_oracle_sweep.py` (docstring),
    `tests/oracle_sweeps/sweep_models.py` (docstring only; a byte-close copy, so the change is recorded in that
    directory's README beside its one existing local change), `shared_geometry.py`, `manifold.py`, `backend/fitting.py`,
    `docs/user_guide.rst`, `docs/numerical_contracts.md`, `docs/contributor/numerical_contract_catalog.md`,
    `docs/contributor/fitting_internals.md`, `docs/batching_and_stacking.md`, `docs/api_reference.rst`,
    `dev/paper_scope.md`.

**`fitting.LocalModel`, exact mode (S6).**

- A field `hessian: str = 'gauss_newton'` (the mode, a static aux) and a field `curvature: typ.Any = None` (a leaf).
  The registration (`:733`, `:748-754`) adds both, with `curvature`'s `partition_static` flatten keeping statics
  static (M3). The positional unflatten is updated.
- **Methods:**

  | method | meaning |
  |---|---|
  | `gradient` | GN: `_wrap(transpose)` (unchanged). Exact: `_tangent_at(frame, curvature.misfit_gradient)` + `g_R` — bitwise equal (m27) |
  | `hessian(p)` | the **mode-dependent** Hessian action: `gn_hessian(p)` in GN mode; `gn_hessian(p) + curvature_hessian(p)` in exact mode |
  | `quadratic(p)` | likewise: `gn_quadratic` or `gn_quadratic + curvature_quadratic` |
  | `evaluate(p)` | `c + gᵀp + ½ quadratic(p)` (mode-aware; identical in GN mode) |
  | `objective_value`, `jacobian`, `gn_hessian`, `gn_quadratic` | unchanged |
  | `curvature_hessian(p)`, `curvature_quadratic(p)` | exact mode only; they raise in GN mode |
  | `retract(p)` | **new on the model**, mirroring the backend `LocalModel.retract`: `geometry.retract(p)` at this frame (the shared geometry passes `geometry_aux`). The same T3-SVD retraction in both modes: the retraction belongs to the geometry, and its second order (F6, F14) is what makes the exact model's ρ meaningful. Nick's list marks `retract` "mode-dependent"; v5 reads that as "used by the mode-dependent step control". If a mode-specific retraction was meant, that is O-7 |

- The factories (`:534`, `:554`, `:574`, `:633`, `:684`, `:707`) and `_uniform_model` (`:454-488`) keep their names.
  They gain `hessian: str = 'gauss_newton'` and `curvature_damping: typ.Optional[float] = None`.
  - One helper, `_with_curvature(model, hessian, curvature_damping)`, builds the backend geometry
    (`ragged_backend_geometry`, or `from_frame` on uniform) and calls `geom.curvature_precompute(…, curvature_damping)`.
  - The no-effect rule (n16), with the same messages as the backend: `_with_curvature` raises for a non-None
    `curvature_damping` with `hessian='gauss_newton'`; `curvature_precompute` raises on a corewise geometry.
  - Downstream payoff: `probe_derivatives_model(..., hessian='exact').hessian(p)` replaces `x07`'s FD Hessian
    (`x07_hessian_free_newton.py:102-106`).

**`optimizers.newton_cg`** (`:318-439`) — explicit keywords, mirroring the backend:
`hessian`, `curvature_damping`, `step_control`, `tr_radius0`, `g0_reference`, `g0norm_newton`, `g0norm_cg`,
`cg_forcing_power`, `max_newton`, `gtol_rel`, `cg_maxiter`, `c_armijo`, `on_line_search_failure`.

- These sit beside the existing explicit ones (`order`, `weight`, `verbose`, `val_sample`, `val_data`, `callback`,
  `use_jit`, `regularizer`, `chunk_size`). Every key T3Polynomial passes today is among them (§K).
- **`**kwargs` is removed.** A misspelt option now raises `TypeError` instead of being forwarded.
- `gradient_descent` gains `g0_reference` (C.15).

**`manifold.py:1381-1386`:** the retract docstring gains "second order on a minimal, numerically full-rank frame".

### C.14 Diagnostics (`backend/optimizer_display.py`)

`_format_header` (`:121-144`). The `ls %d…` segment (`:140-141`) becomes a single **`step`** segment:

- line search: `"ls %d%s α %.2e ‖Δx‖/‖x‖ %.1e"` — byte-identical to today's text;
- trust region: `"TR Δ %.2e %s ‖Δx‖/‖x‖ %.1e"`, where `%s` is `✓` (accepted) or `✗` (rejected).

Other changes:

- **The CG symbol:** `✓`, `∂` (boundary), `⌇` (truncated) or `⋯` (maxiter).
- **Exact rows** append `" | κ %s σ %s ε %s"`: κ = `pKp/(pHp − pKp)` guarded; σ = `lift_sigma_min`; ε = `lift_eps`.
  All use `_cell` semantics (`:63-67`).
- **The converged line** names its reference when `info.g0_reference` is set: `"converged (‖g‖ ≤ gtol·‖g₀‖, g₀ =
  zero tensor)"`, or `warm start` / `override` (G0P §4.1). With `g0_reference=None` (a hand-built `NewtonInfo`, as in
  the `format_newton_iter` doctest at `:171`) the text is unchanged.
- **`stop_reason`** appears on the final row.
- **Rejected TR rows** reuse the previous row's error tables (n14).
- **GN output:** the line-search `step` segment is byte-identical. The converged-line suffix is the only GN-visible
  text change, and it is listed in the CHANGELOG.

**I-13 (settled): these are the only diagnostics.** The normal-space sector norms are deferred (G gives the ledger
paragraph). Rank-continuation decisions keep using the t3svd unfolding condition numbers.

### C.15 The zero reference (I-18a, folded from G0P; fixes P1a–P1c)

**The API** (both `newton_cg` layers; `gradient_descent` too):

```python
        g0_reference:   typ.Optional[str] = None,     # None -> geom's own; 'zero' | 'warm_start'
        g0norm_newton:  typ.Optional[float] = None,   # explicit Newton-stop reference (wins); must be > 0
        g0norm_cg:      typ.Optional[float] = None,   # explicit CG-forcing reference (wins for CG); must be > 0
```

**Resolution**, in order, once, at iteration 0 (after `_prepare_jit_inputs`, on the level's own `Problem`, reusing iteration 0's local model `lm0`):

1. `ref = g0_reference`, lower-cased, or, if it is `None`, the **geometry's own**: `'zero'` if
   `geom.zero_point_frame(frame0)` is not `None` (the manifold geometries), else `'warm_start'` (the corewise
   geometries; P1a). An explicit `'zero'` on a geometry with no zero-point frame raises (n16 rule).
2. **The zero reference** (only when `ref == 'zero'` and the overrides do not already fix both references). Let
   `F0 = geom.zero_point_frame(lm0.frame)`, where `lm0` is iteration 0's local model, which the loop builds anyway. The
   call copies only the frame's structure, so there is no extra orthogonalisation. The zero tensor at x0's ranks has
   x0's frame structure, because frame ranks are structural. Then:
   - `sw = problem.kind.precompute(F0, sample)`;
   - `r0 = problem.kind.point_forward(zeros, sample) − data`, with `zeros = corewise_zeros_like(x0_cores)`. This is the same
     residual formula `Problem.local_model` uses, so a custom kind with `S(0) ≠ 0` is handled (for the built-in kinds
     `r0 = −data`);
   - `g = geom.project(F0, problem.kind.transpose(r0, sample, F0, sw), aux=geom.precompute(F0))`;
   - `zero_ref = √geom.inner(g, g)`. With a regularizer, `g_R(0) = 0` (G0P §5), so only the misfit enters.
   - This is one sweep plus one transpose, the cost of one Newton gradient.
   - On the level's `Problem`, so it reuses the resolved `chunk_size`: no second `'auto'` compile (G0P §3).
   - Its dtype and device follow `x0`'s converted ones.
3. **The warm-start reference:** `‖g(x0)‖`, today's `computed_g0norm`. It is read from iteration 0 as today. The `> 0
   else 1.0` fallback (`:716`) is kept **only** for this kind: a zero warm-start gradient means x0 is stationary, and
   the stop fires immediately anyway. (For `ref == 'warm_start'` only.)
4. **The overrides win**, with the existing chained fallback:
   - `newton_ref = g0norm_newton` if given, else the resolved reference;
   - `cg_ref = g0norm_cg` if given, else `newton_ref`.

   So `g0norm_newton` alone sets **both** tests, and `g0norm_cg` alone sets only CG (unchanged; G0P §5 bullet 2).
   `NewtonInfo.g0_reference` records `'override'` when `g0norm_newton` is given, else the resolved kind. A
   non-positive override raises `ValueError`. It used to crash with `ZeroDivisionError` in the forcing term [V `c29`].
5. **A zero `zero_ref`** (the data gradient vanishes at these ranks; for example, the data is orthogonal to every
   rank-r direction) raises `ValueError("the zero-tensor reference is 0 …; pass g0_reference='warm_start' or
   g0norm_newton")`. It is **never** silently replaced by 1.0. The canonical frame makes this non-generic (F32).

**The forcing term.**

- `η = min(0.5, (‖g‖/cg_ref)^cg_forcing_power)` is unchanged.
- The default power stays 0.5 (I-7). G0P §0 measured that the zero reference tightens η about 2× at high rank in
  continuation (1.9× wall time at a fixed cap). That trade is accepted and documented, and the η floor / plateau stop
  remain open (J-12).
- On F19's restart the zero reference *reduced* the step count (F26). On E2, it turned an 80-step non-convergence into
  31 steps at `gtol_rel` 1e-3 (F27).

**Why a canonical frame (P1b/P1c, F30–F32).** At X = 0 every orthonormal completion is a valid frame, and the
gradient's norm depends on which one is chosen. That is the G0P §3 "trap".

- LAPACK's completion differs between the ragged and uniform layers, so per-layer references differ by up to 1.9×.
  That breaks trajectory equivalence (6/16 pairs drift; the O2 sweep drifts to 3.9e-3).
- A coordinate-aligned completion, LAPACK's included, gives **exactly 0** on data with no energy on its prefix
  coordinates (F32).
- `zero_point_frame` fixes both problems:
  - It is built from the structure alone, with the same real blocks on both layers, so the reference is
    layer-independent (≤ 3.9e-16) and the O2 pairs hold at 2.6e-9.
  - It is a seeded Haar draw for U, so it is generic.
- Seeds 0–5 vary the reference by an O(1) factor (F32), which is the honest content of P1b. The seed is a fixed
  convention (`ZERO_FRAME_SEED = 0`), not a parameter.
- **The start's frame is untouched.** A zero x0 still takes `geom.frame(x0)`'s LAPACK completion for its first step,
  so numerics change only through the reference (J-13).

**The G0P wording, corrected (P1b).** G0P §4.2 calls the zero reference "the problem's intrinsic data-gradient scale".
v5's docs say "a deterministic data-gradient scale at the level's ranks, fixed by convention and good to an O(1)
factor; comparable across levels and layers; harmless for a relative stop".

**Audit of ragged-vs-uniform pairings** (P1c). Every test or example that compares a ragged trajectory against a
uniform one:

| site | why it stays equal |
|---|---|
| `tests/oracle_sweeps/sweep_optimizers.py:136-145` (O2, both tiers) | the canonical reference is layer-independent; all 44 pairs at ≤ 2.6e-9 [V `c37b`]. No edit to the byte-close copies |
| `tests/test_optimizers_frontend.py` U7a: `:211` (gradient_descent), `:238` (verbose display), `:260` (regularizer) | same reference on both layers. `gradient_descent` gains `g0_reference` with the same default and construction |
| `tests/backend/test_uniform_fitting.py:470-480` (`newton_cg_matches_ragged`, `gradient_descent_matches_ragged`) | same |
| `tests/test_sharing.py:1281` (`test_trajectories_match_ragged_and_recovery_stays_tied`, shared ragged vs uniform) | one Haar draw per group on both layers; the shared O2 pairs in `c37b` hold |
| examples `fit_hilbert_uniform_newton_cg.py:15-16` (a docstring pairing, not executed against each other) | n/a |

T26 adds the pinned pair check: one ragged and one uniform `newton_cg` from the same nonzero start, `newton_losses`
equal to 1e-7, the O2 threshold (observed worst over 44 pairs: 2.6e-9).

**Zero starts change too: not a no-op, as G0P §4.2 had expected.** The reference is now taken at the canonical frame,
not at the start's LAPACK frame, so even from a zero start it moves by an O(1) factor (F32). The S0d gate:

- every test asserting `info.g0norm == history[0]['gnorm']` is re-pinned to `'warm_start'` or re-derived;
- recovery tests asserting accuracy at the default `gtol_rel` are run and checked; a reference that differs by an O(1)
  factor shifts the stop by at most one or two steps.

**Tests with nonzero starts whose numbers change.** They are audited, and pinned to `g0_reference='warm_start'` where
the test is *about* the old semantics:

- `tests/backend/test_optimizers.py:262-310` (the override test: its baseline asserts `info.g0norm == first gnorm`
  from a **zero** start, so it now pins `'warm_start'` for the baseline and adds a `'zero'` row);
- `tests/test_optimizers_frontend.py:79-90` (override forwarding; unaffected because overrides win);
- `:307-356` (`test_nonminimal_x0_handled_transparently`, `test_validation_args_go_together`, `test_kind_string_is_case_insensitive`: `randn` / `x_true` starts, `max_newton=1`; the asserts are structural or compare two identical calls, so they are unaffected).

### C.16 The public backend-problem constructor (I-18b)

**The name:** `t3toolbox.optimizers.least_squares_problem`.

- It is the frontend twin of `t3toolbox.backend.optimizers.least_squares_problem`: same name, same operation, one level
  up (`naming_conventions.md` §"Sharing a name with the backend twin is fine, and intended").
- It sits in the role-named fitting family ("`uniform_least_squares_problem`… are fitting infrastructure").
- It is in `optimizers.__all__`. It is **not** re-exported at the root, which keeps the root namespace clean
  (`naming_conventions.md` §"The prefix is what makes the root export possible").

```python
def least_squares_problem(
        geometry,                                  # frontend geometry: MANIFOLD / COREWISE / UNIFORM_* / a SharedGeometry
        kind:        str,                          # 'apply' / 'entries' / 'probe' (+ '_derivatives'), case-insensitive
        sample:      typ.Any,
        data:        typ.Any,
        x0:          Point,                        # ragged or uniform: picks the representation (as today's _setup)
        order:       typ.Optional[int] = None,
        weight:      typ.Optional[typ.Any] = None,
        regularizer: typ.Any = None,
        chunk_size:  typ.Any = 'auto',
        batch:       typ.Optional[int] = None,     # minibatch W for 'auto' (mc_sgd / adam)
) -> typ.Tuple[
        bopt.Problem,                              # the backend Problem: .local_model(...), .objective(...), .geom
        typ.Any,                                   # x0 as backend cores (ragged .data or the bare supercore pair),
                                                   #   reduced to minimal ranks on manifold geometries
        typ.Callable,                              # rewrap: backend cores -> the frontend tensor of x0's representation
]:
```

- It is today's `_setup` (`t3toolbox/optimizers.py:174-229`), renamed and documented, with a seeded doctest that
  builds a problem and takes `local_model(x).gradient`.
- **`_setup` is removed**, not aliased. All four optimizers call `least_squares_problem` (`:258`, `:282`, `:313`,
  `:424`). Rationale: in a breaking release a private alias would keep a second spelling alive for a downstream that
  is being updated anyway (§K).
- **`_geometry_ops` stays private.** It is an internal adapter whose only callers are `least_squares_problem` and the
  error messages. Downstream stops calling it, because `least_squares_problem(...)[0].geom` gives the same backend
  geometry and its two uses build a whole problem anyway (§K: `fitting.py:108`, `d07:64`).
- **`ragged_backend_geometry`** stays out of `fitting.__all__`. It has no remaining downstream caller.

---

## D. Batching contract (frame-inner `W + K + C`) and the grouped strings

| function | inputs (stacks) | output | machinery |
|---|---|---|---|
| `filtered_reciprocal`, `damped_pseudo_inverse` | s C+(k); M C+(m,n); eps C | C+… | elementwise; batched SVD |
| `fv_center_factors`, `fv_lift_factors` | frame C | C | grouped `contract`; zipper scan |
| `fv_zero_point_frame` / `ufv_zero_point_frame` | template frame, C = () | C = () | host numpy construction |
| `tv_corewise_lift[_transpose]` | frame / factors C; variations K+C | K+C | grouped `contract` (C pins itself) |
| `t3_*inner_product*` | Z, θ, sweep C; θ̂ K+C | K+C | grouped `contract`; tangent-zipper scan |
| `t3_probe_corewise_derivatives_hessian_from_sweep` | jets `(o+1,)+W+C`; dotted `(o+1,)+W+K+C`; z̃ `(o+1,)+W+C+(N,)` | K+C (W summed, chunked) | grouped `contract` with `trs`; GN-form recurrences / scans |
| `ut3_*_corewise_*hessian` | θ C masks; θ̂ `(d,)+K+C+(...)`, masked with θ's masks | K+C, re-masked | `ut3_apply_masks`, then delegate |
| uniform transposes after S0c | as today | as today, **re-masked** | one host-mask multiply |
| `LocalModel.hvp / curvature_hvp / curvature_quadratic` | p K+C | K+C | composition; `geom.inner` → `corewise_stack_dot` |

**`'...'` einsums** remain only for per-element operations on one opaque batch: the filter's elementwise `where`, the
batched SVD, and the existing `xnp.einsum` in `fv_frame_orthogonality_residual`. Everything that carries W / K / C
is a grouped `contract`. The '...' string would hide what the batch is; the grouped string documents it and is
searchable.

**New `HISTORICAL` rows** (`tests/test_contractions_interpreter.py:70`). The v4 table's 20 jet rows are kept
(supplements by analog, verified sufficient in `c17`). The 25 rows verified in F35 (`c34`, against `np.einsum` on
W = (2,3), K = (2,2), C = (2,3)) are added:

```
lift / Lᵀ            'Cux,KCxn->KCun': ''   'KCaib,Cbc->KCaic': ''   'Cux,KCun->KCxn': ''   'KCaic,Cbc->KCaib': ''
frame factors        'Caib,Cbc->Caic': ''   'Caxb,Caub->Cxu': ''
tangent zipper       'KCij,Ciak,Cjal->KCkl': ''   'Cij,KCiak,Cjal->KCkl': ''   'Cij,Ciak,KCjal->KCkl': ''
T3–T3 two-hole       'Czn,KCun->KCzu': ''   'Cazb,KCzu->KCaub': ''   'KCazb,Czu->KCaub': ''
factored TT assembly 'trs,rWCa,sWCi->tWCai': ''   'trs,rWKCa,sWCi->tWKCai': 'C'   'trs,rWCa,sWKCi->tWKCai': 'C'
                     'tWKCai,tWCb->KCaib': 'C'   'tWCai,tWKCb->KCaib': 'C'
                     'trs,tWCa,rWCi->sWCai': ''   'trs,tWKCa,rWCi->sWKCai': 'C'   'trs,tWCa,rWKCi->sWKCai': 'C'
                     'trs,rWCa,tWCi->sWCai': ''   'trs,rWKCa,tWCi->sWKCai': 'C'   'trs,rWCa,tWKCi->sWKCai': 'C'
                     'sWKCai,sWCb->KCaib': 'C'   'sWCai,sWKCb->KCaib': 'C'
```

- The uniform d-prefixed twins use the same letters (S8).
- The scan-form kernels record their own strings in the commit where they appear (`TestVocabularyCompleteness`,
  `:297-310`); it cannot under-count.
- The W-chunked tangent assembly engages on uniform + jax only, with ≥ 3 chunks; the `shard_map` + `psum` recipe
  applies unchanged (`chunking.md`).

---

## E. Costs and memory

Measured where marked [M]. Final numbers come from S3, S8 and S0b, recorded in `verification_records.md`.

| operation | cost | notes |
|---|---|---|
| **probe_derivatives exact / GN** | **[M] 2.73× (d = 4, r = 16, W = 2000, order 3); 2.98× (d = 5, r = 8, W = 300, order 2)** — house-standard forms (F33) | the prototype's 3.0–4.1× across runs (F20, `c35`, `c35b`) is superseded: its dense dotted adjoint and 9-term assembly are not the standard. S3 re-measures on the implemented kernels; the gate is ≤ 3.5× at both sizes |
| two-hole profile (standard forms, larger size) | the TT assembly halves (F33); the dotted forward and adjoint sweeps (~15 % each) are already GN-form | the remaining dominant terms mirror their GN counterparts |
| **vs the round-5 expectation (≈ 2–2.5×)** | the measured 2.73–2.98× sits above it | the expectation discounted c25's prototype. That is right in direction, but the prototype's excess was mostly its 9-term assembly, not its trs forms: the standard-form dotted adjoint is about as fast as the dense one at these sizes (`c35`: scanned/dense 1.1–2.0× *slower* at small r, near parity at r = 24). The irreducible part is that the dotted forward plus the dotted adjoint cost ≈ 1.2× a GN transpose (`c35b`). S3 measures on the fused kernel |
| apply exact / GN | ≈ 2× (N-dominated) – 3.5× (r-dominated) | estimate |
| probe exact / GN | ≈ 3–4× | estimate |
| `K_{grad f}` (T3–T3, doubled rank) | m-independent `O(d[Nn·n_Z + n²r² + nr³])` | — |
| lift / `Lᵀ` / factors | m-independent; `d` small SVDs per Newton step | — |
| per Newton step | + Z build + T3–T3 sweep; **no** extra transpose (M2) | — |
| zero reference (C.15) | one frame sweep + one transpose, **once per `newton_cg` call** | the cost of one gradient |
| S0c re-mask | **[M] 0.08–0.61 % of a uniform transpose** (F34) | constant-folded masks under jit |

**Memory.**

- **Recompute default:** no new resident W-scaling state. The transients are the dotted edge variables
  (`(o+1)·W·K·r` each, with K = 1 in CG), the re-swept ν (`(o+1)·W·r` per core), and the factored assembly's pair
  intermediates (`(o+1)·W·K·r·n` per core, inside a W chunk).
- **Store variant** (I-11, deferred): about +60 % of the jet sweep, for a ~10–15 % saving on the two-hole.
- **Compile time** for the exact uniform_jax CG body is measured in S8; the fallback is `remat`.

---

## F. Tests (numpy-only numerics; jax in `test_dispatch` / the jvp oracle / the jax-absent subprocess)

New file `tests/backend/test_curvature.py`.

- Test pattern:
  - `subTest` over structures × stacks: none, C, K, K+C, multi-axis C, W and K;
  - per-test seeding.
- Structures:
  - minimal;
  - Tucker slack `(6,6,6),(4,4,4),(1,2,2,1)` and TT slack `(1,9,9,1)`;
  - d ∈ {1, 2, 4};
  - rank-1 bonds;
  - a zero-padded `resize`;
  - the zero point;
  - maximal rank.
- **Helper `tests/backend/_curvature_oracles.py`:**
  - the block-sum oracle (F5);
  - the dense Lagrangian Hessian;
  - the **dense-`trs` reference twins**, `c22.two_hole_v3` rewritten seeded. They exist only here (P5): the production
    kernels are pinned to them at 1e-15.

T1–T24 are unchanged from v4 except as noted. T25–T29 are new.

| test | asserts | oracle |
|---|---|---|
| T1 factors | `T_i == zipper`; `S_iᵀ == V_iU_iᵀ`; spectra; `Σσ(S_0)² = ‖X‖²` | F1, F17 |
| T2 lift identity | minimal and both slack kinds, gauged and ungauged | F2, F3 |
| T3 lift adjoint | `⟨Lv, y⟩ == ⟨v, Lᵀy⟩` per K+C element, several ε | — |
| T4 damping | ε → 0 continuity; finite at a padded restart; zero-point lift | F4 |
| T5 two-holes | every `*_corewise_[derivatives_]hessian` + T3–T3 vs block-sum and FD of the one-hole; probe_derivatives orders 0–3, K = (2,3), d = 1; **the production (GN-form) kernels == the `_trs` reference twins** | F5, F13, F33 |
| T6 symmetry | `⟨a,Kb⟩ = ⟨b,Ka⟩`; `⟨p, hvp q⟩ = ⟨q, hvp p⟩` | — |
| T7 retraction FD | `⟨p, hvp p⟩` vs `d²/dt² f(tv_retract(tp))` | F6 |
| T8 gauge laws | `K_N L p_vert = 0`; `hvp(p) = hvp(Πp)`; `Π hvp = hvp` | F7, F8 |
| T9 dense Lagrangian | hvp vs FD Hessian of `f∘Φ − ⟨grad f, Φ⟩` through L | DP §4.6 |
| T10 regularized | exact + `IdentityRegularizer(λ)` vs FD along `tv_retract` | — |
| T11 weights | ω[mode,order]: the two-hole uses `ω²r` | block-sum |
| T12 shared | tied lift; adjoint; exact vs FD along the shared retraction; padded shared restart | F14 |
| T13 corewise exact | `GN + K_{Aᵀr}` vs block-sum and the FD Hessian of `θ ↦ f(Φ(θ))` | — |
| T14 Steihaug unit | ToyModel: indefinite → boundary; small Δ → boundary; Δ = ∞ bit-identical; `cg_truncated` excludes boundary stops | — |
| T15 TR loop | rejection keeps x; radius rules incl. Δ_max; descent guard; `losses[k] == history[k]['objective']`; **`newton_steps` / `newton_accepted` / `stop_reason`**; floor (a)–(e) as v4 (m22, m30) | F21, F25 |
| T16 rate | T16's own problem (F22). **Power 1.0** (passed explicitly): ≥ 3 points with `e_{k+1} ≤ 2·e_k²`; ρ asserted only on steps with `|pred| > 1e3·res`. **Power 0.5** (the default): `e_{k+1} ≤ 2·e_k^{1.5}` in `[1e-6, 1e-1]` (observed ≤ 0.98, F29); `stop_reason ∈ {'gradient', 'objective_floor'}`. GN control: ratios in `(0.3, 0.9)`. Constants re-pinned in S5 | F22, F29 |
| T17 uniform | per element vs ragged; exact output masks; garbage-padded θ, samples, θ̂ (incl. the boundary-bond slots on `UNIFORM_COREWISE`); mutation (θ̂ mask removed → fails) | F15, F18 |
| T18 frontend | exact `fitting.LocalModel` == backend `LocalModel`; exact `gradient` == GN `gradient` bitwise; `hessian(p)` / `quadratic(p)` mode-dependent; `retract` == geometry retract; `_bgeom` via `from_frame` == via `from_point`; `curvature_damping` reaches `fv_lift_factors` (factors equal to a direct call at that value); a non-None value raises with GN and on any corewise geometry, shared included (n16) | backend |
| T19 display | `test_exact_mode_header`: the `step` segment in both forms (LS byte-identical to today; `TR Δ … ✓/✗`), ∂, κ zero denominator, σ —, the converged-line reference name, `stop_reason`; rejected rows do not call `point_forward` | — |
| T20 flat slots | maximal rank and ragged d = 1: `‖Lᵀ K_N LΠp‖ ≤ 1e-12·‖Lᵀ K_{Aᵀr} LΠp‖` with **no special case** (the d = 1 shortcut bypassed); the MANIFOLD d = 1 model has `geometry_data is None` and `hvp ==` GN | E4, E5 |
| T21 restart regime | exact `newton_cg` from a zero-padded `resize` (E2 row 1): `stop_reason == 'gradient'`, ≤ 2 rejections; ragged + uniform | E2 |
| T22 shared restart | the same on `shared_manifold((0,0,1))`, probe_derivatives order 2 | F19 |
| **T23 required sixth operation** | a `SamplingKind` subclass with five operations: GN works; `local_model(hessian='exact')` raises the structural error naming it; for it and for every built-in ragged kind, Nick's identity form `type(k).corewise_hessian is SamplingKind.corewise_hessian` agrees with the marker check (asserted verbatim). **In the window, per uniform kind until its override lands:** the structural error, and the mutation check (delete the `_UniformKind` stub → the kind inherits the ragged method → the test fails). After the override: it reads the masks from `frame` (m29), with a TT-slack frame whose `frame[5]` ⊊ `self.masks` | F23, F25 |
| T24 θ̂ masking unit | garbage in θ̂'s padded slots == zeroed padding (real block bitwise), every edge padded; masked θ == the sweep's `(mb[0], mb[2])` | F18, F25 |
| **T25 non-minimal end to end (I-23)** | full `newton_cg(hessian='exact')` with **no minimal reduction** (the backend `Problem`, and the frontend with a slack x0 on COREWISE-free paths) from the Tucker-slack and TT-slack points of F28: converges (`stop_reason == 'gradient'`); the frame ranks are those of `t3_orthogonal_representations` throughout; on TT slack GN+Armijo at the same budget does not converge (the documented contrast); ε = 0 vs 1e-6 identical. Ragged; uniform twin on the frontend path | F28 |
| **T26 zero reference** | (a) resolution: `None` → `'zero'` on MANIFOLD and shared manifold, `'warm_start'` on COREWISE (ragged and uniform); an explicit `'zero'` on corewise raises; (b) **layer independence**: the ragged and uniform references from the same x0 agree to 1e-14, and one ragged/uniform `newton_cg` pair from a nonzero start has `newton_losses` equal to 1e-7 (observed ≤ 2.6e-9, F31); (c) the canonical frame is valid (orthogonality residual ≤ 1e-14, base point 0, Π idempotent) at d = 1, 3, 4 and shared, and tied when shared; (d) **non-collapse**: data with no energy on the prefix coordinates gives a positive reference (F32); (e) overrides win (`g0norm_newton` sets both, `g0norm_cg` sets CG only; `NewtonInfo.g0_reference == 'override'`); non-positive overrides raise; a zero `zero_ref` raises; (f) the warm-start path reproduces today's `computed_g0norm` bitwise; (g) regularized: the reference equals the unregularized one | F26, F30–F32 |
| **T27 S0c output hygiene** | per uniform transpose wrapper (kinds, `utv_*_transpose`, `ut3_*_corewise_*transpose`): exact output masks; padded output == 0; real block **bitwise** equal to the pre-S0c value (frozen in the test); garbage-padded inputs | F34 |
| **T28 jax-absent** (`tests/test_jax_absent.py`) | in the jax-hidden subprocess: **construct** `LiftFactors` (via `fv_lift_factors`), `ManifoldCurvatureData` (via `curvature_precompute`), `CurvatureState` and an exact `fitting.LocalModel` (via `probe_derivatives_model(..., hessian='exact')`); call its `hessian(p)`; run `newton_cg(..., hessian='exact', use_jit=True)` to completion with the jax-absent warning (P4) | — |
| **T29 stats / display contract** | `stats` has `newton_steps`, `newton_accepted`, `stop_reason` and **no** `newton`; history rows carry `step_control`; LS rows have `ls_steps` ints, TR rows `None` | — |

**Jax** (`tests/test_dispatch.py`):

- jit every new op, ragged and uniform;
- `test_exact_local_model_rebuilt_each_step_is_one_jit_cache_key` (`:556`) also covers "ε changes every Newton step
  with one compile" (ε is a leaf), and "a different `curvature_damping` does not recompile" (it reaches only the ε leaf);
- `test_jit_uniform_exact_local_model` (M3);
- `test_cg_radius_is_not_stale`;
- the probe_derivatives exact jit with chunking engaged;
- `tests/test_curvature_jvp_oracle.py` (jax-gated, float64);
- `zero_point_frame` is host-side and eager by design (C.10). `test_dispatch` checks jax-in → jax-out for it and that
  `newton_cg(use_jit=True)` resolves the reference eagerly; it is not jit-compiled.

**Oracle sweeps.** New `tests/oracle_sweeps/sweep_exact.py` in both tiers. The existing O2 files need **no edit** for
the g0 default (C.15 audit). Only `sweep_models.py`'s docstring changes, for the rename (C.13).

**Research-repo measurement** (never copied in):

- **S0a** (F19 is its first point; F26 adds the reference interplay);
- **S0b**, after S6: T3Polynomial `e01`/`o01`/`d05`/`s05`, plus `x06`/`x07` as FD cross-checks.

**Budget.** T16/T21/T22/T25 run in ≤ 3 s each, or they move to the slow tier.

---

## G. Docs, CHANGELOG, doctests

**User docs.**

- **`docs/fitting_and_optimization.md`:**
  - §4.2 qualifier (m14);
  - **new §4.12 "The exact Hessian and the trust region"**: when it helps, the API (`hessian`, `step_control`,
    `tr_radius0`, `curvature_damping`), the **forcing power** ("`0.5` is the default in both modes; with
    `hessian='exact'`, `cg_forcing_power=1.0` gives the q-quadratic tail, measured here [F29]"), cost, restarts,
    reading Δ/κ/σ/ε, and the flat-slot / d = 1 caveat;
  - **§5 rewritten for the reference:** `g0_reference`, the overrides, the corewise default, "a convention, good to an
    O(1) factor";
  - the stats-key table;
  - the LocalModel method table;
  - `:464-472` replaced (G0P §4.3).
- **`docs/rank_continuation.md`:** the per-level zero reference is now the **default** (`:76-97`, `:103-109`,
  `:172-182` rewritten; the "reuse level-0's g₀" advice dropped); exact mode at restarts needs nothing special (E2,
  F19).
- **`docs/sharing.md`:** exact on shared geometries; the continuation section loses the manual g₀ tracking.
- `docs/chunking.md`, `docs/numerical_contracts.md` (caveats 1–6), `docs/naming_conventions.md`
  (`corewise_hessian`; `least_squares_problem` as a same-named frontend/backend pair; `fitting.LocalModel` /
  `backend.optimizers.LocalModel` likewise), `docs/transposes.md`.
- `docs/api_reference.rst`, `docs/user_guide.rst`: the rename and `optimizers.least_squares_problem`.

**Examples.**

- `fit_shared_factors_jetted_probes.py:121-130` and `tests/test_sharing.py:1077-1102` drop the manual level-0 g₀
  tracking.
- The two uniform continuation examples (`fit_hilbert_uniform_newton_cg.py:147`,
  `fit_hilbert_uniform_probe_derivatives_newton_cg.py:165`) need no code change; their prose is updated.
- `fit_varied_rank_tensor_newton_cg.py` (hand-rolled) gets a comment that it does not inherit library fixes.
- Every `stats['newton']` → `stats['newton_steps']`: `fit_hilbert_tensor_newton_cg.py:302`,
  `fit_hilbert_uniform_newton_cg.py:158`, `fit_hilbert_uniform_probe_derivatives_newton_cg.py:177`. The hand-rolled
  `fit_varied_rank_tensor_newton_cg.py:207/255/291` uses its own dict and is renamed for consistency.

**Contributor docs.**

- **New `docs/contributor/curvature_internals.md`** (in the `contributor_guide.rst` toctree):
  - the lift and its one-sided exactness;
  - the block-sum oracle;
  - the two-index-jet rule;
  - **"the dotted kernels hold the GN kernels' standard"** (P5, F33);
  - the factored assembly;
  - the T/S transforms;
  - the companion pytree rule;
  - misfit-gradient-once;
  - θ̂ masking;
  - the flat-slot reasoning (I-12);
  - the canonical zero-point frame and why it is Haar (F30–F32);
  - E1–E10 and F13–F35.
- **`deferred_and_rejected.md`:**
  - Rejected: L3; jvp/FD as shipped implementations; a `LocalContext` argument bundle (for now); a `has_corewise_hessian`
    flag (superseded by the required operation).
  - Deferred: storing curvature state (I-11), preconditioning / LM / NL2SOL β, per-mode ε, reusing shared centers
    (I-21), the curvature-carrying-regularizer hook (m19), the η floor and the objective-plateau stop (J-12), and the
    **normal-space sector norms**, as this paragraph:

    > **Normal-space sector norms — deferred** (I-13, Nick 2026-10-26). The curvature term's normal-space operand Z
    > (the misfit gradient's ambient lift) splits into 2^d sectors `E_S = ⊗_k (U_k^⊥ if k ∈ S else U_k)`. The tangent
    > space sees `∅` (the core) and the singletons `{k}` (the Tucker slots). So the normal part decomposes as
    > `‖N_core‖² = ‖P_∅Z‖² − ‖H^grad‖²`, `‖N_k‖² = ‖P_{k}Z‖² − ‖V_k^grad‖²`, and `‖N_{≥2}‖²` as the remainder. Per
    > sector, they would say where the data pull leaves the manifold: a rank-continuation signal and a curvature
    > diagnostic. Two routes: an exact Gram route, `O(m²dN)`, for apply/entries; and unbiased rank-one Gaussian
    > probing, `E[⟨Z, P_S(⊗y_j)⟩²] = ‖P_S Z‖²`, for every kind. Deferred because the always-on diagnostics (Δ, ρ, κ,
    > σ_min) cover the solver's needs, and rank continuation already reads the t3svd unfolding condition numbers.
- `verification_records.md` (E2, F19, F26–F35; the S0a/S0b, S3/S8 numbers).
- `numerical_contract_catalog.md` (the caveats).
- `scan_body_principles.md:190`, `parameters_not_closures.md:111`, and a new paragraph in `parameters_not_closures.md`
  (§L.3's "options are not parameters").
- `dev/OPEN_QUESTION_extension_surface.md`: "six operations" (C.6); do not archive.

**CHANGELOG `## [2026.3.0]`**, created in S0c:

- **Changed — breaking:**
  - `fitting.GaussNewtonModel` → `fitting.LocalModel` (not aliased);
  - `stats['newton']` → `newton_steps` / `newton_accepted` / `stop_reason`;
  - the default Newton reference is the zero tensor of the level's ranks on manifold geometries (`g0_reference`), with
    G0P §0's table;
  - `_setup` removed, replaced by the public `optimizers.least_squares_problem`;
  - `optimizers.newton_cg` no longer takes `**kwargs`;
  - `SamplingKind` has a sixth required operation for `hessian='exact'` (five-operation kinds still work for GN);
  - uniform transposes return clean padding;
  - the converged display line names its reference;
  - non-positive `g0norm_*` raise.
- **Added:** `hessian='exact'`, the trust region, `curvature_damping`, `step_control` / `tr_radius0`, the backend
  functions, the `NewtonInfo` fields, `fitting.LocalModel.{hessian, quadratic, curvature_*, retract}`,
  `Geometry.zero_point_frame`.
- **`docs/release_notes.md` "Upgrading from 2026.2.0":**
  - pass `g0_reference='warm_start'` to keep today's reference;
  - delete `history[0]['gnorm']` bookkeeping;
  - rename `GaussNewtonModel`;
  - read `stats['newton_steps']`;
  - `_setup` → `least_squares_problem`;
  - custom geometries gain `zero_point_frame` (return `None` to opt out) and, for exact mode only, the four curvature
    members;
  - custom kinds gain `corewise_hessian` for exact mode only.

**`dev/` routing.**

- The DP and G0P go to `dev/archive/` with resolution banners (S13).
- `dev/curvature_oracles/` goes into T7/T9 and is archived.
- `dev/HANDOFF.md` and the CLAUDE.md Current-state pointer are refreshed (the version, the rename, the stats keys).

**Gates:** `sphinx -W`; module and page doctests on both numpy generations.

---

## H. Ordered slices

Each slice is one commit and independently green (`pytest tests/ -n auto`, doctests, `sphinx -W`).

**The order principle.** The breaking-release groundwork lands first (S0c–S0e), as small, reviewable, mechanical or
self-contained commits. Then T3Polynomial's path. By S9, probe_derivatives runs exact end to end on ragged, uniform and
uniform_jax, both unshared and shared.

| slice | content | gate tests |
|---|---|---|
| **S0a** (research repo) | E2 extended to probe_derivatives (F19, F26); ε, the TR constants, Δ0 | numbers → `verification_records.md`. If S0a has not finished by S4, ship `DEFAULT_CURVATURE_DAMPING = 1e-6` and record that |
| **S0c** | uniform output hygiene (C.0); CHANGELOG `[2026.3.0]` header | **T27**; every existing uniform test unchanged |
| **S0d** | the zero reference: `fv_zero_point_frame` / `ufv_zero_point_frame`, `Geometry.zero_point_frame` (four geometries), `g0_reference` on `newton_cg` + `gradient_descent` (both layers), override validation, the stats-key rename (`newton_steps` / `newton_accepted` / `stop_reason`), `NewtonInfo.g0_reference` / `step_control` / `stop_reason`, the converged-line text; examples' `stats['newton']`; docs §5 / rank_continuation / sharing | **T26**, **T29**; the O2 sweep unchanged and green (both tiers); `test_optimizers.py:262` re-pinned (C.15) |
| **S0e** | the frontend rename `GaussNewtonModel` → `LocalModel` (the C.13 checklist); `_setup` → public `optimizers.least_squares_problem` (+ doctest); `optimizers.newton_cg` explicit keywords (no `**kwargs`) | the existing suite, renamed; a doctest; `TypeError` on a misspelt option |
| S1 | `filtered_reciprocal` / `damped_pseudo_inverse`; `fv_center_factors` (+ the `fv_shared_frame_data` refactor, bit-identical); `LiftFactors` (+ `if jax_available:` registration), `fv_lift_factors`; `tv_corewise_lift[_transpose]` | T1–T4; the sharing tests unchanged |
| S2 | move `_inner_product_frame_transpose` to `t3_linalg` (bit-identical); `t3_inner_product_corewise_transpose`; `tt_zipper_tangent_*`; the T3–T3 two-hole | T5 (T3–T3), T6, T20 (backend half) |
| **S3** | **the probe_derivatives two-hole, ragged, in the GN forms** (C.4): the `_trs` reference twins in the test helper; `_adj_tilde_tangent_step` (recurrence + order scan); the dotted dx̃i; the **factored** TT assembly, chunked; HISTORICAL rows; memory and **cost re-measure (gate ≤ 3.5×)** | T5 (orders 0–3, K = (2,3), d = 1; production == `_trs` twins), T6, T11 |
| S4 | the protocols: `SamplingKind.corewise_hessian` (sixth operation, `@abc.abstractmethod` marker) + `ProbeDerivativesKind`'s implementation + the **`_UniformKind` raising stub**; the Geometry exact members and `DEFAULT_CURVATURE_DAMPING`; `ManifoldCurvatureData` / `CurvatureState` (+ registrations); backend `LocalModel` exact mode; `Problem.local_model(hessian, curvature_damping)` | T7–T10, T13, T20, **T23**; dispatch: exact hvp jit and one cache key |
| S5 | Steihaug body and state; `_cg_core` / `_cg_solve` 5-tuple (and its 3 call sites); `step_control` / `tr_radius0`; the TR loop, floor rule and dtype ε; TR bookkeeping; the `NewtonInfo` TR / exact fields; the `cg_truncated` redefinition; the **`step` display segment**; `hessian` / `curvature_damping` on backend `newton_cg` | T14–T16 (constants re-pinned here), T19, T21 (ragged), **T25** (ragged); `test_cg_radius_is_not_stale`; existing GN tests and O2 unchanged |
| S6a | `UniformManifoldGeometryOps.from_frame`; `_bgeom` → `from_frame` (GN path) | T18 (mask equality); uniform GN tests unchanged |
| S6 | frontend exact mode: `fitting.LocalModel.hessian` / `curvature` fields, the new methods, all 7 construction sites, the pytree; `optimizers.newton_cg` exact keywords | T18; **S0b can start** (ragged) |
| S7 | shared (ragged): `_tied_filtered_solve[_transpose]`, tied lift, `s_g` in `sigma_min`, shared corewise | T12, T22 |
| S8 | uniform: `ufv_lift_factors`, `utv_corewise_lift[_transpose]`, `ut3_inner_product_*`, `ut3_probe_corewise_derivatives_hessian` (θ, θ̂ masked; output re-masked), `UniformProbeDerivativesKind.corewise_hessian` (replaces the stub), the uniform geometries' exact members | T17, T24, T20, T21 / T25 (uniform); jit; chunked jit; compile time |
| S9 | shared (uniform) | T12 uniform twin + T17 rules. **T3Polynomial path complete** |
| S10 | plain probe (ragged + uniform override) | T5, T6, T11, T17, T23 |
| S11 | apply + entries | T5, T6, T17, T23 |
| S12 | apply/entries derivatives; **delete the `_UniformKind` stub and `_UNIFORM_HESSIAN_SLICE`** | T5, T6, T11, T17, T23 (window cases removed) |
| S13 | docs, the final CHANGELOG, release notes, doctests, `sweep_exact.py`, the jvp oracle, **T28** (jax-absent construction), the `dev/` archive, HANDOFF / CLAUDE.md | the full release gate incl. `T3TOOLBOX_SLOW_TESTS=1` |
| **S14** (T3Polynomial repo) | §K, against the release candidate | T3Polynomial's suite |

- S0c–S0e change behaviour only as the CHANGELOG lists. S1–S3 are pure additions. S4 is the first curvature
  consumer change, and GN stays the default throughout.
- **No release before S13.** T3Polynomial (S14) updates against the tagged release candidate.

---

## I. Decisions

### Settled (Nick, round 5) — recorded, not reopened

| # | decision | settled as |
|---|---|---|
| I-1 | S/T source | zipper on demand, once per model |
| I-2 | mode API | `hessian='gauss_newton' \| 'exact'`, case-insensitive |
| I-3 | mode encoding | `curvature is not None`; the term is present iff `geometry_data is not None` |
| I-4 | frontend name | **`fitting.LocalModel`**, mirroring the backend class; not aliased; factories keep their names |
| I-5 | Steihaug | one CG body; Δ = ∞ ≡ today |
| I-6 | TR policy | TR by default in exact mode; ρ > 0.1; ¼‖p‖ / 2Δ; Δ0 = GN Cauchy; Δ_max = 1e4·Δ0; floor only on rejection (`objective_floor`, then `trust_region_floor`) |
| I-7 | forcing power | **`cg_forcing_power: float = 0.5` everywhere**; 1.0 documented for the q-quadratic tail (F29) |
| I-8 | ε | relative to ‖X‖_HS, `DEFAULT_CURVATURE_DAMPING = 1e-6` |
| I-9 | default mode | GN |
| I-10 | jet layout | companion arrays |
| I-11 | store vs recompute | recompute |
| I-12 | d = 1 / flat slots | no special case in the formula; flat slots carry no normal component (B); the MANIFOLD d = 1 `None` is a cost shortcut |
| I-13 | diagnostics | Δ, accept, boundary, pKp/κ, real-slot σ_min (+ `s_g`), ε, stop_reason; **sector norms deferred** (ledger paragraph, G) |
| I-14 | kind surface | **`corewise_hessian` is the required sixth operation**; no flag, no shadow; early structural check; `_UniformKind` raising stub in S4–S12; OPEN_QUESTION says "six" |
| I-15 | block-sum oracle | test helper only |
| I-16 | S0 timing | S0a before S4; S0b after S6 |
| I-17 | TR bookkeeping | rejected steps count toward `max_newton`; one `losses` entry per row; **`newton_steps` / `newton_accepted` / `stop_reason`** |
| I-18a | g0 reference | **`g0_reference`**, default the geometry's own (`'zero'` on manifolds, `'warm_start'` on corewise); overrides win; canonical zero-point frame (C.15) |
| I-18b | public constructor | **`optimizers.least_squares_problem`**; `_setup` removed; `_geometry_ops` private |
| I-19 | `stop_reason` | five values; `objective_floor` is not a failure |
| I-20 | slice order | groundwork S0c–S0e, then T3Polynomial's path S1–S9, other kinds S10–S12, one release at S13, downstream S14 |
| I-21 | shared centers | not reused |
| I-22 | damping | **a solver parameter, `curvature_damping`**, flowing newton_cg → local_model → curvature_precompute; default in the geometry module; no field (§L) |
| I-23 | uniform θ̂ masks; non-minimal frames | the frame's `(um, lm)`; never force a minimal frame; **T25** end-to-end on Tucker and TT slack |
| I-24 | `corewise_hessian`'s point | keep `base_point` |
| I-25 | floor ε | `np.finfo(residual dtype).eps` |

### Genuinely open

| # | item | how it closes |
|---|---|---|
| O-1 | `DEFAULT_CURVATURE_DAMPING` final value | S0a / S0b on T3Polynomial's fits. Two problems (E2, F19) agree on 1e-6 ≡ 0 so far |
| O-2 | the TR constants (0.1 / 0.25 / 0.75, ¼, 2×, Δ_max = 1e4·Δ0) | S0a; they ship as module constants |
| O-3 | T16 / T25 constants | re-pinned run-and-paste in S5 on the implemented code |
| O-4 | the S3 cost gate (≤ 3.5×) and the memory gate (≤ 1.5× the GN assembly) | S3 measurements |
| O-5 | the jax float32 `objective_floor` rejection bound (T15(e)) | pinned at implementation (no jax in the review env) |
| O-6 | `ZERO_FRAME_SEED` | fixed at 0 by convention; listed so that Nick can veto the Haar draw in favour of another *generic* deterministic completion. A coordinate-aligned one is ruled out by F32 |
| O-7 | `retract` "mode-dependent" (Nick's I-4 list) | v5 keeps one retraction for both modes (C.13); if a mode-specific retraction was meant, Nick says which |

---

## J. Risks and open questions

1. **Indefinite, ill-conditioned CG.** The TR bounds the step, not the CG count. S0b measures it; the follow-up is a
   preconditioner or LM. E2 / F19 / F28 saw 3–8 hvps per Newton step.
2. **ε** — O-1.
3. **The GN-form two-hole rewrite** (S3) is pinned to the `_trs` reference twins at 1e-15.
4. **S3 assembly memory**: the factored form's pair intermediates add `(o+1)·W·K·r·n` per core inside a chunk; measured
   in S3.
5. **float32** is benign (E3).
6. **The shared retraction order** is measured, not proven (F14).
7. **Compile time** (S8); the fallback is `remat`.
8. **`_bgeom → from_frame`** is isolated in S6a.
9. **The forcing term vs the zero reference** (G0P §0): about 2× CG work at high rank in continuation at a fixed cap.
   Accepted (I-7); the docs show how to trade it with `cg_forcing_power`.
10. **The zero reference is a convention** (P1b). Documented as such; layer-independent by construction. It moves
    zero-start runs by an O(1) factor in the reference, so their stop point by a step or two (C.15).
11. **Out of scope:** subsampled exact Newton (mc_sgd).
12. **Open:** an objective-plateau stop and an η floor (G0P §0, §5, §6). Deferred and recorded; the TR's
    `objective_floor` covers the resolution limit, not a slow plateau.
13. **Open:** whether a zero **start** should also use the canonical frame (it would make zero-start trajectories
    layer-independent too, but it changes today's zero-start numerics). Not done here.
14. **Open:** `tr_radius0` across continuation levels; the default recomputes Δ0 per level.

---

## K. Downstream (T3Polynomial) update — slice S14, about a half-day

All references are at `/home/claude/t3polynomial` @ `388090d`. The update is made against the 2026.3.0 release
candidate. Each edit is mechanical unless marked.

**K.1 `t3polynomial/fitting.py` — the reference machinery becomes library-native.**

| line(s) | today | after |
|---|---|---|
| `:86-112` | `_zero_gradient_norm(...)`: `topt._geometry_ops` + `bopt.least_squares_problem` + `local_model(zeros).gradient` | **deleted**. The library computes the reference (`g0_reference` default `'zero'` on the manifold geometries T3Polynomial uses), on the same level, in the same layer, at a layer-independent frame |
| `:158-164` | `WARM_START_REFERENCE_WARNING` | kept: the downstream policy still warns on the warm start |
| `:180-188` | `_newton_fit` docstring: "the zero reference is structural here… `g0norm_newton=None`" | reworded: the zero reference is the **library default**; the warm start is `g0_reference='warm_start'` |
| `:189-197` | `if 'g0norm_newton' not in kwargs: … _zero_gradient_norm(...) … setdefault('gtol_rel', 1e-3)` / `elif kwargs['g0norm_newton'] is None: warn; drop it` | `if kwargs.get('g0_reference') == 'warm_start': warnings.warn(WARM_START_REFERENCE_WARNING, …)` `else: kwargs.setdefault('gtol_rel', 1e-3)` (the warm-start path keeps the library's own `gtol_rel`, as today). **The semantics collision (P3)**: `g0norm_newton=None` no longer means "warm start". It now means "no override" → the zero default. So every caller that passes `None` to *mean* warm start is changed (below) |
| `:199`, `:208` | `topt.newton_cg(...)` | unchanged calls (they now pass `g0_reference` through `kwargs`) |
| `:338-341` | `extra = dict(g0norm_newton=None) if warm_start else dict(g0norm_newton=_zero_gradient_norm(...), gtol_rel=gtol_ref)` | `extra = dict(g0_reference='warm_start') if warm_start else dict(gtol_rel=gtol_ref)` |
| `:353`, `:364` | `newton=stats['newton']`, `{stats['newton']:>6}` | `newton=stats['newton_steps']`, `{stats['newton_steps']:>6}`. The record key `newton` is kept for the downstream's own level records (x02 / o01 / e01 / n01 read `rec['newton']`) |
| `:498-501` | as `:338-341` (adaptive) | as `:338-341` |
| `:513`, `:525` | `stats['newton']` | `stats['newton_steps']` |
| `:28`, `:55`, `:183-188`, `:287-288` | docstrings mentioning `g0norm_newton=None` / `_zero_gradient_norm` / `ls_steps` | reworded (`g0_reference`; `ls_steps` is `None` on TR rows) |

**K.2 `tests/test_fitting_selection.py:55-91`** (`test_gtol_ref_default_and_zero_reference_is_structural`).

- `:61`: import `_newton_fit` only (`_zero_gradient_norm` is gone).
- `:83`: `g0z = _zero_gradient_norm(...)` becomes the library's own reference, read from the run:
  `st['history'][0]['g0norm']` with `st['history'][0]['g0_reference'] == 'zero'`.
- `:86`, `:91`: `st['newton']` → `st['newton_steps']`.
- `:89`: `g0norm_newton=None` → **`g0_reference='warm_start'`**. With the old spelling, `None` would mean "no
  override" and the test would compare the zero reference against itself, testing nothing (P3).
- `:91`: keeps `assertGreaterEqual(st_warm['newton_steps'], st['newton_steps'])`, and adds
  `assertEqual(st_warm['history'][0]['g0_reference'], 'warm_start')`, so the test cannot pass vacuously.
- `:67`: `rec_loose[-1]['newton']` is the downstream's own record key, unchanged.

**K.3 Scripts.**

| script:line | edit |
|---|---|
| `scripts/x06_hessian_fd_probe.py:59-61`, `:65-113` | no class name appears (the model comes from `tfit.probe_derivatives_model`); `.gn_hessian` / `.gn_quadratic` / `.gradient` / `.objective_value` are unchanged. **Optional**: add an `exact` column, `probe_derivatives_model(..., hessian='exact').hessian(p)`, beside the FD Hessian, as the cross-check S0b wants |
| `scripts/x07_hessian_free_newton.py:72-74`, `:95`, `:102-106`, `:117` | `hess(p)` (central FD of the projected gradient along the retraction) can become `model_at(X, hessian='exact').hessian(p)`. Kept as an **FD oracle** (S0b's cross-check); a flag selects it. `m.gn_quadratic` (`:117`) is unchanged |
| `scripts/d07_curvature_dispersion.py:63-65` | `problem_at(...)`: `bopt.least_squares_problem(topt._geometry_ops(t3m.MANIFOLD, shape), bfit.probe_derivatives_kind(order, weight), sample, data)` → `topt.least_squares_problem(t3m.MANIFOLD, 'probe_derivatives', sample, data, X, order=order, weight=weight, chunk_size=None)[0]`. It takes the point `X` (it already has it at `:118`) in place of `shape`; `:119` and `:140` pass `X`. `lm.gradient / .gn_quadratic / .hvp` (`:120-163`) are unchanged |
| `scripts/d08_profile_ops.py:113-114` | `topt._setup(geom, 'probe_derivatives', sample, dat, x0, order, weight, None, chunk_size=None)` → `topt.least_squares_problem(geom, 'probe_derivatives', sample, dat, x0, order=order, weight=weight, chunk_size=None)` (keywords, with the same values) |
| `scripts/d05_darcy_fit_assess.py:232` | `ls = [... if r.get('ls_steps') is not None]` — already TR-safe. Made explicit: `if r.get('step_control', 'line_search') == 'line_search'`. `:237` `cg_truncated` keeps its meaning (boundary stops now excluded; noted in the report). `:272` `newton_per_level=[int(r['newton']) …]` reads the downstream record key: unchanged |
| `scripts/b01_backend_bench.py:275-278` | `callback=lambda info: steps.append(info.ls_steps)`, histogram over non-`None`: already TR-safe. The `None` filter at `:276` stays, and the printed count uses LS rows only. `:152`, `:154`, `:318` read the downstream record key `newton`: unchanged |
| `scripts/x02`, `o01`, `e01`, `n01` (`rec['newton']`) | the downstream record key: unchanged |
| `dev/t3toolbox-upstream-notes.md:695`, `:718`, `:788` | `GaussNewtonModel` → `fitting.LocalModel` (notes only) |

**K.4 Estimate.** About a half-day:

- K.1 + K.2: ~2 h, including running the suite;
- K.3: ~1 h;
- an S0b smoke run on `e01` at one level: ~1 h.

No algorithmic change downstream. The zero reference it computed by hand is now the library default, at a frame that
is also layer-independent, so `uniform` / `uniform_jax` levels now reference the same number as `ragged`. Today they
all use the ragged reference, so this is no regression.

---

## L. How each straightforward route keeps the conventions' intent

**L.1 Damping as a solver parameter (I-22) vs "parameters are FIELDS" and "optimizers.py imports no T3 module".**

- **What the field rule is for.** `parameters_not_closures.md` targets objects that **ride as jit `aux_data`**, whose
  hash/eq *is* the compilation cache key: a parameter sealed in a closure made rebuilt-but-identical objects new cache
  keys and silently miscompiled. The rule is about **value identity of aux objects**, not about where every number
  lives.
- **Why the damping is not such a parameter.** It is not a property of the manifold. Changing it does not change the
  set being optimized over (unlike `groups`, which changes the manifold), so it is not geometry identity. It is an
  optimization option, like an LM λ or `cg_forcing_power`, which are already plain solver arguments.
- **How it travels.** It reaches the computation only through the traced `LiftFactors.eps` leaf. So it never enters
  any aux, cannot go stale, and cannot cause a recompile (T-dispatch asserts this).
- **The import rule.** `backend/optimizers.py` stays T3-free: a float passing through imports nothing. The only
  T3-specific thing, the default 1e-6, lives in `backend/geometry.py`, which resolves `None`.

**L.2 The public constructor (I-18b) vs "don't widen the API in a feature PR".**

- **The intent** is to avoid accidental, unreviewed surface.
- This release is *the* breaking release, and the surface is deliberate:
  - a downstream already depends on the private spellings (`_setup`, `_geometry_ops`);
  - the backend razor asks what a backend-level user needs — here a `Problem` from frontend vocabulary — and a private
    underscore function is the worst of both: used, but unpromised.
- Promoting it with a name, a doctest and a CHANGELOG line **reduces** accidental surface. Removing `_setup` rather
  than aliasing it follows the 2026.2.0 rename precedent.

**L.3 The required sixth operation (I-14) vs a capability flag.**

- **The flag pattern's intent** (`has_block_sumsq`) is to let an *optional display* degrade gracefully.
- `corewise_hessian` is not optional for the feature that uses it. It is the derivative of `transpose`, part of
  what a kind *is* once exact Newton exists. Making it a required operation, with the same `NotImplementedError`
  guidance as the other five, puts the extension contract in **one** place: the base class.
- The early check is the structural-error rule ("structural problems always error"), and GN users are unaffected.
- The `@abc.abstractmethod` marker is the stdlib's own name for "this method is unimplemented". It keeps
  `optimizers.py` import-free without inventing a new flag.

**L.4 Grouped strings for the lift (v4 used `'...'`).** The `'...'` einsum is the house's one-batch-block machinery,
and the lift does broadcast over K for free. The grouped string says *which* blocks are there (`KC` vs `C`), is
searchable by `TestVocabularyCompleteness`, and costs nothing (F35: identifiable with no supplement). That is the
intent behind both machineries: say what the batch is.

**L.5 GN-form kernels vs "banded recurrences only" (P5).** The intent of Nick's rule is that **small-scale dense timings
must not choose production forms**. The house already applies that rule as "the right form per convolution type"
(`deferred_and_rejected.md:85-96`). Holding each dotted kernel to its GN counterpart's form satisfies the intent
exactly. Demanding more would make the derivative faster than the operator it differentiates, with no evidence it is
needed.

**L.6 No alias for the rename.** The 2026.2.0 precedent ("deliberately not aliased") exists so that a breaking release
breaks loudly once, rather than carrying two names indefinitely. The downstream update (§K) is in scope.

---

## M. Import DAG (new and changed edges; verified acyclic, F35)

New edges (→ means "imports"):

```
fv_operations    → linalg                         (filtered pinvs)
ufv_operations   → linalg, fv_operations, ufv_conversions   (ufv_lift_factors; ufv_zero_point_frame)
sharing          → fv_operations, linalg          (fv_center_factors; filtered_reciprocal)
tv_operations    → fv_operations, t3_linalg       (LiftFactors annotation; _inner_product_frame_transpose)
t3_linalg        → fv_conversions                 (t3_corewise_frame)
ut3_linalg       → ufv_conversions
geometry         → t3_linalg, ut3_linalg, linalg  (z sweep; DEFAULT_CURVATURE_DAMPING lives here)
ut3_sampling     → ut3_masking                    (already present)
utv_sampling     → ufv_masking                    (already present)
uniform_fitting  → ufv_masking (present), ufv_operations (not needed by the design; included in the check as headroom)
```

- **Unchanged:** `optimizers → {common, regularization}` and the top-level `corewise`. No T3 module.
- `fitting` (frontend) → backend `geometry`, `optimizers`, `uniform_fitting`: as today.
- `optimizers` (frontend) → `fitting`: as today.
- **No cycle.** `fv_operations` imports neither `tv_operations` nor `sharing`; `tv_operations` does not import
  `fv_operations` today. Both modules already have `from __future__ import annotations` (`tv_operations.py:12`,
  `fv_operations.py:11`), but no string annotations are needed: direct imports suffice.
- The graph scan (`c34`) parses every backend module's real import lines (doctest lines excluded) plus these 17 edges
  and finds it acyclic.

**Pytree registrations, each under `if jax_available:` in its defining module:**

| class | module | precedent |
|---|---|---|
| `LiftFactors` | `fv_operations.py` | `sharing.py:1081` |
| `ManifoldCurvatureData` | `geometry.py` | `optimizers.py:773` |
| `CurvatureState` | `optimizers.py` (the existing block) | — |
| frontend `LocalModel`'s `curvature` child | `fitting.py:733` (the existing block) | `manifold.py:1572` |

T28 constructs each one in the jax-hidden subprocess.

---

## Conflicts between the design plan (DP) / G0P and the code — consolidated

| # | source says | code reality | resolution |
|---|---|---|---|
| X1 | DP §4.3: curvature into `gn_quadratic` and Cauchy on it | the same method serves both; T3Polynomial relies on `gn_quadratic = ‖Jp‖²` | new `quadratic`; `gn_*` keeps its GN meaning |
| X2 | DP §4.1/§4.3: S/T from the frame construction | the construction discards them | zipper inside the curvature leaf |
| X3 | DP §4.2: the T3 projection at `(U,G,G,G)` is ∇⟨Z,Φ⟩ | it gauges at the end | ungauged kernel moved to `t3_linalg` |
| X4 | DP §4.2: frame sweep reused | true for xi/mu; probe ν runs through R | re-sweep ν through G |
| X5 | DP §4.2 route 3 "NumPy backend" | `xnp`-agnostic | all backends |
| X6 | DP §4.7: jvp first | the two-hole is the deliverable | block-sum + jvp oracles |
| X7 | DP §4.6: hvp of pure gauge = 0 | false under orthogonal Π (F8) | the three coordinate laws |
| X8 | DP §4.5: 2–2.5× | probe_derivatives **2.73–2.98× in GN forms** (F33) | re-measured S3/S8/S0b |
| X9 | DP §4.1: S from the "centre-core SVD" | a contraction | contraction; SVD only to filter |
| X10 | DP §4.4: ε ~ ‖N‖/λ_min | needs an eigensolve | ε relative to ‖X‖ (E2) |
| X11 | DP §4.3: residual raw | `LocalModel.gradient` includes `g_R` | misfit gradient carried in `CurvatureState` |
| X12 | DP §4.1: non-minimal → slack carries no curvature | the pinv is exact one-sided (F3); full runs converge (F28) | caveat only for a hand-built wide `T` |
| X13 | (frontend) | `_bgeom` re-orthogonalises per uniform model build | `from_frame`, S6a |
| X14 | DP §4.4: "phase-in" via damping at restarts | the clip + TR handle restarts (E2) | wording removed |
| X15 | (uniform) | the one-hole wrappers mask only θ, and leave garbage in output | S0c re-masks every uniform transpose output; the two-holes mask θ and θ̂ |
| **X16** | G0P §3–§4: compute the zero reference "in the same layer/geometry as the level's fit"; §4.2 "intrinsic data-gradient scale" | same-layer LAPACK frames differ across layers (P1c, F30), collapse on aligned data (F32), and the scale is convention-dependent (P1b) | a canonical, layer-independent, generic zero-point frame (C.15); the wording softened |
| **X17** | G0P §4.2: the default `'zero'` for every geometry | zero on corewise is exactly 0 (P1a) | the geometry chooses (`zero_point_frame` → `None` on corewise ⇒ warm start) |
| **X18** | G0P §4.1: a string spelling `g0norm_newton='zero'` | a float-or-string parameter overloads one argument | a separate `g0_reference` selector; `g0norm_*` stay float overrides |
