# Implementation plan — the exact Riemannian Hessian (curvature term) for `newton_cg`

**v1, 2026-10-25. Writer draft for critic review; nothing implemented.** Repo `/home/claude/t3toolbox` @ `920d776`
(`main`). Design source: `dev/riemannian_newton_curvature_plan.md` ("DP"); its §2 math is taken as settled and
re-verified below. Every claim marked **[V]** was checked by a read-only script under
`SCR/checks/` (`SCR = /tmp/claude-0/-home-claude/29728d3c-c433-5028-9de9-1af7ead3c123/scratchpad/plan_review`),
run as `cd /home/claude/t3toolbox && python -I SCR/checks/<script>` (the scripts insert the repo path themselves;
the env has numpy only — no jax). File:line references are to `920d776`.

Target (Nick's decision, DP status banner): **level 2, the exact Riemannian Hessian, computed by a hand-written
two-hole sweep**:

```
H p = Π 𝒥ᵀω²𝒥 Π p  +  Π Lᵀ (K_{Aᵀω²r} − K_{grad f}) L Π p  +  λ Π p
```

`L` = lift from gauged frame coordinates to raw core variations; `K_Z` = the two-hole sweep (the directional
derivative of a corewise transpose); `grad f` = the *misfit* Riemannian gradient as a doubled-rank T3. Companions:
Steihaug–Toint CG + trust region, a damped lift, minimal diagnostics.

---

## Verified facts this plan rests on (summary of `SCR/checks/`)

| # | claim | script | result |
|---|---|---|---|
| F1 | `T_i := ⟨L_i, H_i⟩ = tt_zipper_right_to_left(L, R)[i+1]`; centers `H_i = L_i T_i`; `S_iᵀ := einsum('axb,aub->xu', O_i, H_i)` equals `V_i^X U_iᵀ`; σ(S_i) = mode-`i` Tucker singular values of X, σ(T_i) = bond-`(i+1)` TT singular values; `T_{d−1}` factor `ZZ[d] = ones(1,1)` | `c1_identities.py` | all ≤ 3e-12; spectra match dense unfoldings |
| F2 | Lift identity `DΦ[Lv] = E v` (`tv_to_dense(t3_corewise_frame(U,L), Lv) == tv_to_dense(frame, v)`), gauged **and** ungauged `v`, minimal frame | `c3_debug.py`, `c4_hessian_oracle.py` | ≤ 3.3e-15 per slot; 7.1e-16 total |
| F3 | Same identity on **non-minimal** frames from `t3_orthogonal_representations`: Tucker-slack (`n=4 > rL·rR=2`, `Sᵀ` is 2×4, wide) and TT-slack (`T_1` is 9×4, tall) — pinv is an exact right/left inverse | `c7_nonminimal_padded.py`, `c8_nonminimal_debug.py`, `c12_nonminimal_tt.py` | 9.4e-16, 9.2e-16 |
| F4 | Zero-padded `resize` warm start: guarded pinv finite; at X = 0 the lift is identically 0 except the undamped last TT slot | `c7_nonminimal_padded.py` | finite; 0.0 |
| F5 | **Block-sum oracle** (zero new code): `K_c θ̂ = Bᵀ t3_*_corewise_transpose(c, sample, tv_to_t3(t3_corewise_frame(θ), θ̂))`, where `Bᵀ` sums the gradient blocks in which `G` sits (R-, L- and O-blocks of the doubled TT core; U-block of the doubled Tucker core); vs central FD of the corewise transpose | `c2_lift_blocksum.py`, `c9_blocksum_jets.py` | apply 9.4e-10, entries 1.2e-10, apply/entries/probe_derivatives ≤ 2.4e-10 (FD-limited); ⟨a,Kb⟩=⟨b,Ka⟩ to 13 digits |
| F6 | **Exact Hessian = GN + ⟨θ̂,K_{Aᵀr}θ̂⟩ − ⟨θ̂,K_{grad f}θ̂⟩** vs FD of f along `tv_retract` (minimal, TT-SVD-conditioned point, d = 4) | `c4_hessian_oracle.py` | 252.85288420 vs 252.85288182 (h = 2.5e-3, O(h²)); GN 285.03, level-1 200.01 |
| F7 | Lagrangian null space: raw vertical directions satisfy `K_N L p_vert = 0` | `c4_hessian_oracle.py`, `c10_vertical.py` | 1.3e-11 vs `‖K_{Aᵀr} L p_vert‖ = 1.3e4` |
| F8 | **DP §4.6 wording is wrong under the orthogonal Π:** a frame-coordinate vertical `p_vert` has `E p_vert = 0` but `‖Π p_vert‖/‖p_vert‖ = 0.61` and `‖E Π p_vert‖ = 5.3` — so "hvp of a pure-gauge direction is zero" is false (already false for today's GN hvp) | `c10_vertical.py` | — |
| F9 | Apply two-hole built **from existing kernels only** (`compute_dxi`, `compute_sigma`, `compute_tau`, `compute_sigma_hat`) + 3-term assemblies; the lean sweep `(xi, mu)` is identical at the manifold frame and at `t3_corewise_frame(U, L)` | `c5_apply_twohole.py` | 4.7e-16 vs F5; sweep `allclose` True |
| F10 | Probe two-hole from existing kernels (`compute_sigma/tau/deta`, two `_sigma_step` calls per dotted tilde step, `compute_deta` ×2 for the dotted `dxi_tilde`) + ONE new contraction; T3–T3 two-hole by dotted zippers; both with C = (2,), K = 3 | `c6_probe_t3t3_twohole.py` | ≤ 4.4e-16 vs F5 |
| F11 | apply_derivatives two-hole (two-index jet, order 2): existing `build_input_jets`, `tv_precompute_apply_frame_sweep_jets`, `compute_sigma_jets` + ONE new 3-term adjoint-wired scan step + dotted assemblies; C = (2,), K = 2 | `c13_apply_jets_twohole.py` | ≤ 2.5e-16 vs F5 |
| F12 | `len_*` supplements of the new grouped strings (one sufficient choice each) | `c11_contract_supplements.py` | table in §D |

---

## CHANGES

First version — nothing to list. (Later versions: what changed, what I pushed back on, with doc citations.)

---

## A. Scope and non-goals

**In scope.**

1. The lift `L` and its adjoint `Lᵀ`, plus the damped variant. Ragged and uniform; C and K stacks; shared factors.
2. Two-hole corewise sweeps `K_c θ̂` for all six sampling kinds (apply, entries, probe and the three `*_derivatives`).
   Each is shipped as a standalone backend function plus a `SamplingKind.corewise_hessian` method.
3. The T3–T3 two-hole `K_{grad f}`, next to `tv_project_t3_onto_tangent_space`.
4. `LocalModel` exact mode (`hessian='exact'`), compile-once.
5. Steihaug–Toint CG plus a trust region in `newton_cg`.
6. Frontend `GaussNewtonModel` / factories / `optimizers.newton_cg` wiring.
7. Minimal diagnostics: TR radius, accept flag, CG boundary hit, curvature part of `pᵀHp`, `σ_min` of the lift factors.
8. Tests, docs, CHANGELOG.

**Non-goals.**

- **No AD through the T3-SVD (L3).** Rejected in DP §3 and §4.5 (SVD second derivatives are NaN at `t = 0` and do not
  survive `pad_safe_svd`); recorded in `deferred_and_rejected.md`.
- **The JAX-jvp and polynomial-FD routes (DP §4.2 routes 1–2) are not shipped.** The jvp route is a jax-gated test
  oracle only. The FD route is superseded as an oracle by the exact block-sum oracle (F5).
- **No preconditioner, no LM damping, no NL2SOL-style β switching.** DP §4.3 and §5 leave these open; the trust region
  is the shipped mechanism, and the rest is deferred with a measurement hook (§J).
- **No `N`-slot-norm diagnostic** (DP §4.4). It needs ‖Aᵀr‖-sized work; deferred.
- **No curvature for `gradient_descent` / `mc_sgd` / `adam`.** Their Cauchy steps stay on `gn_quadratic`
  (`optimizers.py:357`, `:424`).
- **No stacked optimization.** `_require_unstacked` (`optimizers.py:269-298`) is unchanged. Stacked C and K-stacked
  `p` are supported at the `LocalModel` / `GaussNewtonModel` level, which is the documented stacked-point path
  (`fitting_and_optimization.md` §4.10).
- **No nonlinear-residual Hessian term.** Downstream (T3Polynomial) adds its own `Σ rᵢ∇²rᵢ`; DP §5 last bullet.
- **No exact Hessian for uniform `d = 1`.** It raises structurally, mirroring `utv_to_ut3` (`utv_operations.py:301-307`).
  Ragged `d = 1` is supported (decision I-12).
- **No `fit(...)` facade and no change to rank-continuation logic.**

---

## B. Math-to-code map

Notation: point `X = Φ(θ)`; frame `(U, O, L, R)` = `(up, down, left, right)` = `T3Frame.data`; gauged coordinates
`v = (V_i, H_i)`; `Π` = `tv_orthogonal_gauge_projection` (`tv_operations.py:83-137`).

| math | code (new unless cited) | shapes (per C element) |
|---|---|---|
| raw gauge `θ = (U, L)` | `geom.base_point(frame) = (frame[0], frame[2])` (`geometry.py:183-189`, `:274-276`, `:378-380`, `:483-485`) | — |
| corewise frame of θ | `t3_corewise_frame(θ) = (U, L, L, L)` (`fv_conversions.py:30-41`); uniform `ut3_corewise_frame` (`ufv_conversions.py:122-137`) | — |
| `T_i` (bond factor) | `ZZ = tt_zipper_right_to_left(L, R)`, `T_i = ZZ[i+1]` (`tt_operations.py:192-199`) | `(rL_{i+1}, rR_{i+1})`; `rR ≤ rL` by construction (R is the right-orth of the L chain, `fv_conversions.py:240-249`) |
| centers `H_i^X = L_i T_i` | `sharing.py:547-548` (refactored into `fv_center_factors`) | `(rL_i, nU_i, rR_{i+1})` |
| `S_iᵀ = ⟨O_i, H_i^X⟩` | `sharing.py:552` (same refactor) | `(nD_i, nU_i)`; `nD ≤ nU` by construction (`t3_orthogonalization.py:102-125`) |
| damped inverse `σ ↦ σ/(σ²+ε²)` with a clip | `linalg.filtered_reciprocal`, `linalg.damped_pseudo_inverse` | — |
| `L: V_i ↦ U̇_i = (S_iᵀ)^{+ε} V_i` | `tv_corewise_lift` | `(nD,N) → (nU,N)` |
| `L: H_i ↦ Ġ_i = H_i ×₃ T_i^{+ε}` (`i < d−1`), `Ġ_{d−1} = H_{d−1}` | `tv_corewise_lift` | `(rL,nU,rR) → (rL,nU,rL_{i+1})` |
| `Lᵀ` (Frobenius adjoint): `Y_U ↦ (S_iᵀ)^{+ε ᵀ} Y_U`, `Y_G ↦ Y_G ×₃ (T_i^{+ε})ᵀ` | `tv_corewise_lift_transpose` | inverse shapes |
| `K_c θ̂ = d/ds ∇_θ⟨c, S(Φ(θ+sθ̂))⟩` | `t3_{apply,entries,probe}_corewise_hessian[_from_sweep]`, `t3_*_corewise_derivatives_hessian[_from_sweep]` | raw `K+C` |
| `⟨θ̂, K_c θ̂⟩ = ⟨c, S(D²Φ[θ̂,θ̂])⟩`, `D²Φ = Σ_{a≠b}` (ordered pairs) | normalisation verified by F6 (no extra ½ or 2) | — |
| `Z = grad f` as a T3 | `tv_to_t3(frame, g_misfit)` (`tv_operations.py:199-330`) | doubled rank |
| `∇_θ⟨Z, Φ(θ)⟩` (one-hole) | `t3_inner_product_corewise_transpose`: the ungauged core of `tv_project_t3_onto_tangent_space` (`:365-419`) at `t3_corewise_frame(θ)` (verified, `c2`) | raw `C` |
| `K_Z θ̂` (two-hole) | `t3_inner_product_corewise_hessian[_from_sweep]` (dotted zippers, F10) | raw `K+C` |
| `N = Aᵀω²r − grad f` | `K_N = K_{Aᵀω²r} − K_{grad f}`; never formed | — |
| Riemannian Hessian | `LocalModel.hvp` (exact mode) = `Π[𝒥ᵀ𝒥Πp + Lᵀ K_N L Πp] + H_R p` | `K+C` |
| COREWISE exact Hessian | `GN_corewise + K_{Aᵀω²r}`, no lift, no grad term (DP §4.3) | — |
| shared lift | the companion's stacked-S SVD with the damped filter (`_tied_solve`, `sharing.py:644-664`, refactored) | — |
| identity regularizer | `H_R = λΠ` is already exact (`fitting_internals.md` §Regularization); the curvature uses the **misfit** gradient only | — |

**Why the lift is exact even on non-minimal frames (refines DP §4.1).** On every frame built by
`t3_orthogonal_representations`, two facts hold by construction:

- `S_iᵀ` is wide-or-square (`nD ≤ nU`), so its pseudo-inverse is a right inverse: `Sᵀ(Sᵀ)⁺ = I`.
- `T_i` is tall-or-square (`rR ≤ rL`), so its pseudo-inverse is a left inverse: `T⁺T = I`.

Whenever the point is numerically full-rank (rank `Sᵀ = nD`, rank `T = rR`), the slot identities are therefore exact
(F3). A wide `T` (`rL < rR`) can only arise from a hand-built frame, for example `T3Frame.random_orthogonal`
(`frame_variations.md` "Four rank stores"). There `L⁺` is the least-squares lift, which is a caveat (§C.12).

**Lift-independence and the vertical space.** `⟨𝔚(ξ,N),η⟩ = ⟨N, D²Φ[ξ̂,η̂]⟩` holds for any lifts. It is the Hessian of
the Lagrangian `f∘Φ − ⟨grad f, Φ⟩` at a critical point of the Lagrangian, so raw vertical directions lie in `ker K_N`
(F7). The correct coordinate-level statements are:

1. `K_N L p_vert = 0`.
2. `hvp(p) = hvp(Πp)`.
3. `Π hvp(p) = hvp(p)`.

"`hvp(p_vert) = 0`" is not one of them (F8).

---

## C. File-by-file changes

Signatures follow `docs/contributor/signature_style.md`:

- one argument per line;
- name · type · `#` shape, in grouped aligned columns;
- one return element per line;
- `len=d, elm_shape=...` micro-grammar.

Docstrings cite the local `t4s.pdf` Appendix A for the frame, and external references (Absil–Mahony–Trumpf 2013;
Psenka–Boumal arXiv:2011.13395; Steihaug 1983; Nocedal–Wright Alg. 4.1, 7.2). Shipped text contains no `dev/` paths
(HANDOFF cluster-5 ruling).

### C.1 `backend/linalg.py` — the filtered reciprocal

```python
def filtered_reciprocal(
        s:      NDArray,                  # singular values, shape=C+(k,) (descending, as returned by svd)
        eps:    typ.Optional[NDArray],    # damping; shape=C (broadcast over k); None -> exact clipped 1/s
        rcond:  typ.Optional[float] = None,  # relative clip vs s[...,:1]; None -> dtype eps * k
) -> NDArray:                             # f(s), shape=C+(k,): s/(s²+eps²) (or 1/s), 0 where clipped
```

- A double-`where` guard, branch-free:
  `keep = s > rcond·s_max`; `f = where(keep, s / where(keep, s² + eps², 1), 0)`.
- `eps=None` reproduces `_tied_solve`'s `where(keep, 1/where(keep, s, 1), 0)` bit-for-bit (`sharing.py:658-659`),
  so the shared projection is unchanged.
- Why a clip even at ε = 0: a zero-padded slot's singular value comes back as ~1e-17, not 0. Without the clip, 1/σ is
  1e17. With ε > 0 the clip is redundant but harmless.

```python
def damped_pseudo_inverse(
        M:      NDArray,                  # shape=C+(m, n)
        eps:    typ.Optional[NDArray],    # shape=C; None -> clipped pinv
        rcond:  typ.Optional[float] = None,
) -> typ.Tuple[
        NDArray,                          # M^{+eps}, shape=C+(n, m)
        NDArray,                          # s, shape=C+(min(m,n),)  -- for diagnostics
]:
```

- This is a thin batched SVD; no basis is kept.
- It is pad-transparent: masked padding gives zero rows/columns, hence zero σ, hence f = 0. So it uses plain
  `xnp.linalg.svd`, not `pad_safe_svd`. `pad_safe_svd` is only needed when a *kept basis* must be a mask prefix
  (`uniform_svd_prefix_orthogonalization.md`).
- Dispatch is inferred from the inputs.

### C.2 `backend/fv_operations.py` — frame factors and lift factors (the S_i / T_i decision)

```python
def fv_center_factors(
        frame:  typ.Tuple[
            typ.Sequence[NDArray],  # up_tucker_cores. len=d, elm_shape=C+(nUi, Ni)
            typ.Sequence[NDArray],  # down_tt_cores.   len=d, elm_shape=C+(rLi, nDi, rR(i+1))
            typ.Sequence[NDArray],  # left_tt_cores.   len=d, elm_shape=C+(rLi, nUi, rL(i+1))
            typ.Sequence[NDArray],  # right_tt_cores.  len=d, elm_shape=C+(rRi, nUi, rR(i+1))
        ],
) -> typ.Tuple[
    typ.Tuple[NDArray, ...],        # centers H_i = L_i T_i.   len=d,   elm_shape=C+(rLi, nUi, rR(i+1))
    typ.Tuple[NDArray, ...],        # tucker factors S_i^T.    len=d,   elm_shape=C+(nDi, nUi)
    typ.Tuple[NDArray, ...],        # bond factors T_i.        len=d-1, elm_shape=C+(rL(i+1), rR(i+1))
]:
```

- The function is steps 1–2 of `fv_shared_frame_data` (`sharing.py:543-554`), extracted.
- `fv_shared_frame_data` is rewritten to call it. That rewrite is bit-identical, and its existing doctest pins it.
- Steps: GEMM only, using the zipper `tt_zipper_right_to_left(L, R)` and `'...'` einsums.
- **Import check before writing:** `sharing.py` must not already be imported by `fv_operations.py`. It is not today
  (`fv_operations.py:13-21`). If a cycle appears, the function goes in `sharing.py` with an `fv_` prefix instead
  (the feature-is-not-a-family rule, `naming_conventions.md` §grammar).

```python
@dc.dataclass(frozen=True, eq=False)    # identity hash/eq: array fields (the SharedFrameData precedent, sharing.py:445)
class LiftFactors:
    tucker_pinvs:  typ.Tuple[NDArray, ...]   # (S_i^T)^{+eps}. len=d,   elm_shape=C+(nUi, nDi)
    bond_pinvs:    typ.Tuple[NDArray, ...]   # T_i^{+eps}.     len=d-1, elm_shape=C+(rR(i+1), rL(i+1))
    tucker_svals:  typ.Tuple[NDArray, ...]   # σ(S_i) = mode-i Tucker singular values of X.  elm_shape=C+(nDi,)
    bond_svals:    typ.Tuple[NDArray, ...]   # σ(T_i) = bond-(i+1) TT singular values of X. elm_shape=C+(rR(i+1),)
    eps:           NDArray                   # the absolute damping actually used, shape=C

def fv_lift_factors(
        frame:          typ.Tuple,     # (U, O, L, R) orthogonal frame, stack C
        lift_damping:   float,         # RELATIVE damping; eps = lift_damping * ‖X‖_HS per C element
        rcond:          typ.Optional[float] = None,
) -> LiftFactors:
```

- `‖X‖²_HS` is computed as `t3_left_orthogonal_norm_sq((U, L))` (`geometry.py:73-85`). That is exact because the
  frame is left-orthogonal.
- The result is registered as a jax pytree with arrays as leaves and no aux, like `sharing.py:1081-1091`.
- **Decision I-1 (recommended): derive S/T on demand by the zipper.**
  - This follows `sharing_internals.md` §"The S_i machinery", decisions 1–2: "derived on demand, never stored";
    "recomputed by the ZIPPER... not by a re-sweep and not by a re-SVD".
  - It also follows `precompute_and_caching.md` "Examined and left alone": no state on frames.
  - It does not widen the `t3_orthogonal_representations` return contract, which is used across the library.
  - The SVDs here are of the small factors themselves, to damp-invert them. They never re-derive H, O or R, so there
    is no sign hazard.
- **Conflict with DP §4.1/§4.3.** The DP says "keep two small matrices... currently discarded" and "`S_i`, `T_i` ride on
  `LocalModel` as pytree leaves". They do ride as leaves (inside the curvature companion, §C.8), but they come from
  the zipper, not the frame construction.
- The DP also says S_i comes "from the centre-core SVD `H_i = O_i S_iᵀ`". The code gets `S_iᵀ` by a contraction
  (`sharing.py:552`); the only SVD is the damping SVD.

### C.3 `backend/tv_operations.py` — lift, lift adjoint, T3–T3 one-hole and two-hole

```python
def tv_corewise_lift(
        frame:        typ.Tuple,                 # (U, O, L, R), stack C
        variations:   typ.Tuple[
            typ.Sequence[NDArray],               # tucker_variations. len=d, elm_shape=K+C+(nDi, Ni)
            typ.Sequence[NDArray],               # tt_variations.     len=d, elm_shape=K+C+(rLi, nUi, rR(i+1))
        ],
        lift_factors: 'fv_operations.LiftFactors',  # = fv_lift_factors(frame, ...), stack C
) -> typ.Tuple[
    typ.Tuple[NDArray, ...],                     # Udot. len=d, elm_shape=K+C+(nUi, Ni)
    typ.Tuple[NDArray, ...],                     # Gdot. len=d, elm_shape=K+C+(rLi, nUi, rL(i+1))
]:
```

- The raw core variations at `base_point(frame)` satisfy `DΦ[Lv] = Ev`.
- It is exact at ε = 0 on any frame from `t3_orthogonal_representations` at a numerically full-rank point, minimal
  or not (F2, F3).
- The two einsums are `'...ux,...xn->...un'` and `'...aib,...bc->...aic'`. They use the `'...'` machinery: C is
  innermost and the factors broadcast over K for free.
- The last TT slot passes through undamped.

```python
def tv_corewise_lift_transpose(
        frame:            typ.Tuple,             # (U, O, L, R), stack C
        core_variations:  typ.Tuple[
            typ.Sequence[NDArray],               # len=d, elm_shape=K+C+(nUi, Ni)
            typ.Sequence[NDArray],               # len=d, elm_shape=K+C+(rLi, nUi, rL(i+1))
        ],
        lift_factors:     'fv_operations.LiftFactors',
) -> typ.Tuple[
    typ.Tuple[NDArray, ...],                     # len=d, elm_shape=K+C+(nDi, Ni)   (UNGAUGED frame coords)
    typ.Tuple[NDArray, ...],                     # len=d, elm_shape=K+C+(rLi, nUi, rR(i+1))
]:
```

- This is the Frobenius adjoint, built from the same factors, so `Lᵀ_ε K L_ε` stays symmetric for every ε.

**T3–T3 refactor.** The body of `tv_project_t3_onto_tangent_space` (`:388-415`) becomes a private
`_project_t3_ungauged(frame, x)`, keeping the d = 1 branch at `:402-407`. The public function becomes
`tv_orthogonal_gauge_projection(frame, _project_t3_ungauged(frame, x), shared_data)`, bit-identical. Then:

```python
def t3_inner_product_corewise_transpose(
        z:          typ.Tuple[typ.Sequence[NDArray], typ.Sequence[NDArray]],  # the fixed T3 Z, stack C
        core_pair:  typ.Tuple[typ.Sequence[NDArray], typ.Sequence[NDArray]],  # θ = (tucker_cores, tt_cores), stack C
) -> typ.Tuple[
    typ.Tuple[NDArray, ...],    # ∇_U ⟨Z, Φ(θ)⟩, shapes of tucker_cores
    typ.Tuple[NDArray, ...],    # ∇_G ⟨Z, Φ(θ)⟩, shapes of tt_cores
]:
```

- Body: `return _project_t3_ungauged(t3_corewise_frame(core_pair), z)`.
- This is the corewise transpose of `t3_inner_product` (`t3_linalg.py:176`), by the §6.3 substitution — the
  corewise-trio precedent (CLAUDE.md razor corollary).
- **Conflict with DP §4.2:** the DP assumed `tv_project_t3_onto_tangent_space` at `(U,G,G,G)` *is* this gradient. It is
  not, because it ends with the gauge projection (`:417-419`). Hence the refactor.

```python
def t3_precompute_inner_product_sweep(
        z:          typ.Tuple,     # Z, stack C (e.g. tv_to_t3(frame, g) -- doubled rank)
        core_pair:  typ.Tuple,     # θ, stack C
) -> typ.Tuple[
    typ.Tuple[NDArray, ...],       # BU_i = B_Z U_iᵀ.         len=d,   elm_shape=C+(nZi, nUi)
    typ.Tuple[NDArray, ...],       # G2_i = G_Z ×₂ BU_i.      len=d,   elm_shape=C+(rZi, nUi, rZ(i+1))
    typ.Tuple[NDArray, ...],       # left zippers ZL.         len=d,   elm_shape=C+(rZi, rLi)
    typ.Tuple[NDArray, ...],       # right zippers ZR.        len=d,   elm_shape=C+(rZ(i+1), rL(i+1))
    typ.Tuple[NDArray, ...],       # env_i = ZL G_Z ZR.       len=d,   elm_shape=C+(rLi, nZi, rL(i+1))
]:

def t3_inner_product_corewise_hessian_from_sweep(
        core_directions:  typ.Tuple,   # θ̂ = (Udot, Gdot), stack K+C
        z:                typ.Tuple,   # Z, stack C
        core_pair:        typ.Tuple,   # θ, stack C
        sweep:            typ.Tuple,   # = t3_precompute_inner_product_sweep(z, core_pair)
) -> typ.Tuple[
    typ.Tuple[NDArray, ...],           # (K_Z θ̂)_U, elm_shape=K+C+(nUi, Ni)
    typ.Tuple[NDArray, ...],           # (K_Z θ̂)_G, elm_shape=K+C+(rLi, nUi, rL(i+1))
]:

def t3_inner_product_corewise_hessian(z, core_pair, core_directions):   # one-shot: sweep + from_sweep
```

Per-matvec work:

- `BUd = B_Z U̇ᵀ`, by reusing `_project_t3_rebasis_step` (`:333-343`), which is linear in `U`.
- `G2d`.
- The dotted zippers, by a new polymorphic `tt_zipper_tangent_left_to_right(coresA, coresB, dcoresA, dcoresB,
  zippers)` and its right-to-left mirror in `tt_operations.py`. The scan body `_tt_zipper_tangent_step(carry, x)`
  is module-level and closure-free, with three terms on `'...ij,...iak,...jal->...kl'` (F10).
- `envd = ZLd G_Z ZR + ZL G_Z ZRd`.
- Assembly: `dG = envd ×_i BU + env ×_i BUd` and `dU = (envd·G + env·Ġ) B_Z`.
- d = 1: zippers are absent, so only the `BUd` / `Ġ` terms remain. This needs a dedicated branch and test.
- Machinery: `'...'` throughout; θ̂ is K+C, everything else is C.
- Naming: the tangent-zipper helpers keep descriptive chain names (`naming_conventions.md` §"Cataloged exceptions").

### C.4 `backend/apply.py`, `backend/entries.py` — plain two-holes (lean sweep reused)

```python
def t3_apply_corewise_hessian_from_sweep(
        core_directions:  typ.Tuple,               # θ̂ = (Udot, Gdot), elm_shape=K+C+(...)
        c:                NDArray,                 # residual (ω² already applied), shape=W+C
        ww:               typ.Sequence[NDArray],   # apply vectors, len=d, elm_shape=W+(Ni,)
        core_pair:        typ.Tuple,               # θ, stack C
        frame_sweep:      typ.Tuple[
            typ.Sequence[NDArray],                 # xis. len=d, elm_shape=W+C+(nUi,)
            typ.Sequence[NDArray],                 # mus. len=d, elm_shape=W+C+(rLi,)
        ],                                         # = tv_precompute_apply_frame_sweep(any frame with up=θ_U, left=θ_G)
) -> typ.Tuple[
    typ.Tuple[NDArray, ...],                       # elm_shape=K+C+(nUi, Ni)   (W summed)
    typ.Tuple[NDArray, ...],                       # elm_shape=K+C+(rLi, nUi, rL(i+1))
]:
def t3_apply_corewise_hessian(c, ww, core_pair, core_directions):   # one-shot
```

Body, in order (F9):

1. `sh = compute_sigma_hat(G, xis, c)` (`probing.py:812-838`), the residual-seeded adjoint through θ's own TT cores.
2. `xid = compute_dxi(Udot, ww)` (`:336-360`).
3. `mud = compute_sigma(Gdot, G, G, xis, xid, mus)` (`:390-424`).
4. `shd = compute_tau(Gdot, G, G, xis, xid, sh)` (`:427-459`; `sh` sits in the `nus` slot).
5. Three-term assemblies:
   - `dG = 'WKCa,WCi,WCb->KCaib'(mud,xi,sh) + 'WCa,WKCi,WCb->KCaib'(mu,xid,sh) + 'WCa,WCi,WKCb->KCaib'(mu,xi,shd)`.
   - `dxh = 'WKCa,Caib,WCb->WKCi'(mud,G,sh) + 'WCa,KCaib,WCb->WKCi'(mu,Gdot,sh) + 'WCa,Caib,WKCb->WKCi'(mu,G,shd)`.
   - `dU = 'Wo,WKCa->KCao'(w, dxh)`.

**Sweep reuse.** `(xi, mu)` depend only on `(frame[0], frame[2]) = base_point`. The model's existing sweep is
therefore the corewise one, whichever geometry built it (F9). Only `sh` is per-model. It is recomputed per matvec by
default (decision I-11), which matches the adjoint-state store-vs-recompute preference (`fitting_internals.md`
§"Low-memory transpose").

**Entries.** Identical, except:

- `xis` come from the entries sweep;
- `xid = _entry_xis(Udot, index)` (`entries.py:264-293`; works for K+C because the W axes are moved to the front);
- `dU` uses `_onehot_vectors` (`:296-309`), as in `tv_entries_transpose_from_sweep` (`:216-236`).

The shared tail is a private `_apply_corewise_hessian_adjoint(...)`, the analog of `_apply_transpose_adjoint`
(`apply.py:357-400`), with ragged per-core maps (module-level map bodies) and a d-prefixed uniform branch (§C.9).

### C.5 `backend/probing.py` — probe two-hole

```python
def t3_probe_corewise_hessian_from_sweep(
        core_directions:  typ.Tuple,               # θ̂, elm_shape=K+C+(...)
        ztildes:          typ.Sequence[NDArray],   # residual (ω² applied), len=d, elm_shape=W+C+(Ni,)
        ww:               typ.Sequence[NDArray],   # len=d, elm_shape=W+(Ni,)
        core_pair:        typ.Tuple,               # θ, stack C
        frame_sweep:      typ.Tuple,               # the model's (xis, mus, nus, etas); only xis/mus reused
) -> typ.Tuple[typ.Tuple[NDArray, ...], typ.Tuple[NDArray, ...]]:   # K+C raw (W summed)
def t3_probe_corewise_hessian(ztildes, ww, core_pair, core_directions):   # one-shot
```

**Conflict with DP §4.2** ("frame-dependent parts stay precomputed"). True for apply/entries, but **not** for probe:
the manifold sweep's `nu`/`eta` run through R/O, while the corewise sweep needs them through θ's `G = L`. So per
matvec (recompute default, I-11):

1. `nus_c = compute_nu(G, xis)`, `etas_c = compute_eta(G, mus, nus_c)`.
2. The undotted tildes at the corewise frame: `compute_deta_tilde(U, z)`, `compute_tau_tilde`,
   `compute_sigma_tilde` (`:856-947`).
3. The dotted forward: `xid = compute_dxi`; `mud = compute_sigma(Gdot,G,G,…)`; `nud = compute_tau(Gdot,G,G,…,nus_c)`;
   `etad = compute_deta(Gdot,G,G,mus,nus_c,mud,nud)` (`:478-521`).
4. `detd = 'WCo,KCio->WKCi'(z, Udot)` — the one new string.
5. The dotted tildes, by a new module-level scan body. Its carry update is two existing `_sigma_step` calls (F10):

```python
def _tau_tilde_tangent_step(
        carry:  NDArray,       # tau_tilde-dot: W+K+C+(rLi,)
        x:      typ.Tuple,     # (G, Gdot, xi, xid, tau_tilde, deta_tilde, deta_tilde_dot, mu, mu_dot), one core
) -> typ.Tuple[NDArray, typ.Tuple[NDArray]]:
    # d/ds of  tau' = tau·G(xi) + mu·G(deta)   (the two terms of _tau_tilde_step, probing.py:885-898):
    #   _sigma_step(carry, G, G, Gdot, xi, xid, tau)  +  _sigma_step(mu_dot, G, G, Gdot, deta, deta_dot, mu)
```

   `compute_tau_tilde_tangent` / `compute_sigma_tilde_tangent` (by reversal, the `compute_sigma_tilde` pattern,
   `:927-947`).
6. `dxi_tilde_dot = compute_deta(Gdot,G,G,tau_tilde,nus_c,tau_tilde_dot,nud) + compute_deta(Gdot,G,G,mus,sigma_tilde,
   mud,sigma_tilde_dot)`. This reuses `compute_deta`'s three placements: K on the swept carry, on the core, and on the
   other edge.
7. Assemblies:
   - Tucker: `dU = 'WCo,WKCa->KCao'(z, etad) + 'Wo,WKCa->KCao'(w, dxi_tilde_dot)`. The first string is new.
   - TT: 9 terms, the product rule of the three `assemble_tt_variations` terms (`:1117-1160`) over their three legs.
     All 9 strings already exist (`'WKCi,WCa,WCj->KCiaj'`, `'WCi,WKCa,WCj->KCiaj'`, `'WCi,WCa,WKCj->KCiaj'`).
   - Both live in a private map body `_assemble_tt_variations_tangent_core` with `_summed` variants
     (`n_probe` rides as a map operand, the `:1105-1114` pattern).

### C.6 `backend/sampling_derivatives.py` — the derivative kinds (two-index jets)

**How the two-index jet composes.**

- Every edge quantity is multilinear in `(cores, sample vectors)`.
- The existing jet axis `t` (length `order+1`) is the sample-direction Taylor index, combined by the binomial tensor
  `trs` (`:240-255`).
- The core-direction derivative is a second, *independent* index truncated at `s ≤ 1`. For it, Leibniz needs no
  binomials: `(fgh)˙ = ḟgh + fġh + fgḣ`.
- So each existing `t`-convolution term with three factors spawns three dotted terms with **the same `trs` wiring**.
- Cores have `t`-order 0, as does `Ġ`.
- The dotted input jet is `build_input_jets(compute_xi(Udot, ww), compute_xi(Udot, pp))` (`:258-277`), affine in `t`
  just like `xi`.
- **Layout (decision I-10): companion arrays, not a stacked length-2 axis.**
  - Undotted quantities are `(order+1,)+W+C+(...)`; dotted ones are `(order+1,)+W+K+C+(...)`.
  - Stacking them would force broadcasting every undotted jet over K, which costs memory, and the K-placement of each
    term differs.
  - The `docs/batching_and_stacking.md` §3 rule (frame data carries C only) points the same way.

**Apply/entries derivatives** (F11). Precompute: none (I-11). Per matvec:

- `sh_j = compute_sigma_hat_jets(G, xi_j, c, trs)` (`:2237-2268`).
- `xid_j`.
- `mud_j = compute_sigma_jets(Gdot, G, G, xi_j, xid_j, mu_j, trs)` (`:1002-1030`, existing).
- **New scan body**, following the trs-sliced form of `_sigma_hat_jets_step` (`:2226-2234`):

```python
def _sigma_hat_tangent_jets_step(
        carry:  NDArray,       # sigma_hat-dot jet: (order+1,)+W+K+C+(rR(i+1),)
        data:   typ.Tuple,     # (G_rev, Gdot_rev, xi_jet, xid_jet, sigma_hat_jet, trs_xi), one core
) -> typ.Tuple[NDArray, typ.Tuple[NDArray]]:
    # 'trs,tWKCa,Caib,rWCi->sWKCb'(carry, G, xi) + 'trs,tWCa,KCaib,rWCi->sWKCb'(sh, Gdot, xi)
    #   + 'trs,tWCa,Caib,rWKCi->sWKCb'(sh, G, xid)
```

  The dotted assemblies are 3 + 3 terms (strings in §D). `dU` uses the existing `'uWKCa,uWo->KCao'`.
  Public functions: `t3_{apply,entries}_corewise_derivatives_hessian[_from_sweep]`.

**Probe derivatives** (the hardest slice, S5b; no numerical pre-verification beyond F5's oracle):

- The undotted tildes come from `compute_{deta,tau,sigma}_tilde_jets` (`:1440`, `:1605`, `:1611`) at the corewise
  frame, plus the recomputed corewise `nu_jets`/`eta_jets`.
- The dotted tilde recurrence cannot reuse `_adj_tilde_step` (`:1518-1560`): it infers `nf = len(P.shape[:-3])`
  from the core, so a K-carrying `Ġ` breaks it, and its `prop` assumes K on the carry. **New:**
  `_adj_tilde_tangent_step` — six adjoint-wired terms, each one of the dotted factors:
  - `carry_dot·G·xi` (K on carry);
  - `carry·Ġ·xi` (K on core, `len_C`);
  - `carry·G·xid` (K on xi);
  - `edge·G·deta_dot` (K on deta);
  - `edge·Ġ·deta` (K on core);
  - `edge_dot·G·deta` (K on edge).

  `prop` keeps the fused two-term reverse recurrence; `src` keeps the inner order-scan with `W·r²` peak.
  Ragged/uniform scan-body variants follow the `_adj_sweep_scanned_step_{uniform,ragged}` pattern (`:1563-1582`).
- **New** `_dxi_tilde_tangent_jets_step`: six terms, the product rule of `_dxi_tilde_jets_step`'s two adjoint-hooked
  terms.
- Dotted assemblies:
  - TT: 9 terms over the three `_tt_variation_jets_trs_core` strings.
  - Tucker: 2 + 2 terms.
- These are W-chunked through `_wchunked_sum` (`:1908-1930`) with new **module-level** chunk functions
  `_tt_tangent_chunk_summed` / `_tucker_tangent_chunk_summed`. They are stable objects for the memoized
  `_wchunked_sum_step` factory (`:1866-1885`) and honour the kind's `chunk_size`.
- Peak per W row: each dotted term has the shape of a GN assembly term, so the summed peak is ~1× (sequential) to ~3×
  (if XLA keeps the three partials live) the GN assembly. Slice S5b **measures** it with `memory_analysis`. If it is
  above 1.5×, `estimate_chunk_size` gains `hessian='exact'` (lowering the tangent assembly kernel instead) and the
  frontend `_resolve_chunk_size` (`optimizers.py:145-166`) passes it.

### C.7 `backend/fitting.py` — the `SamplingKind` protocol

On `SamplingKind` (`:305-412`):

```python
    has_corewise_hessian:  typ.ClassVar[bool] = False   # set True by whoever IMPLEMENTS corewise_hessian (the
    #                                                     has_block_sumsq pattern, :326)

    def corewise_hessian(
            self,
            core_directions,   # θ̂ = (Udot, Gdot) at base_point, elm_shape=K+C+(...)
            residual,          # RAW r = S(x) − data; ω² is applied HERE, exactly as in transpose (:553-555)
            sample,
            base_point,        # (tucker_cores, tt_cores) = geom.base_point(frame), stack C
            sweep,             # this model's precompute(frame, sample); its xi/mu half is reused
    ):                         # -> raw (dU, dG), K+C: d/ds ∇_θ⟨ω²r, S(Φ(θ+sθ̂))⟩ |_{s=0}
        raise NotImplementedError(...)   # message names hessian='exact' and has_corewise_hessian
```

- All six ragged kinds implement it (`:485-652`), delegating to §C.4–C.6 with `self._apply_weight(r, 2)`; probe
  derivatives pass `chunk_size=self.chunk_size`.
- The kinds stay geometry-agnostic: they receive the point, never the geometry.
- This **widens the open `SamplingKind` surface** — the standing question in `dev/OPEN_QUESTION_extension_surface.md`
  ("what to watch"). It is one optional method plus a capability flag; a custom kind without it still works in GN
  mode. Decision I-14.

### C.8 `backend/geometry.py` and the `Geometry` protocol (`optimizers.py:60-98`)

These are the exact-mode members, needed only when `hessian='exact'`; GN users and custom GN-only geometries are
unaffected:

```python
    def curvature_precompute(self, frame, misfit_gradient, aux=None, lift_damping=...):   # -> companion | None
    def lift(self, frame, variations, curvature):                  # frame coords -> raw θ̂ at base_point
    def lift_transpose(self, frame, core_variations, curvature):    # raw -> frame coords (ungauged)
    def tangential_curvature(self, frame, core_directions, curvature):  # raw K_{grad f} θ̂, or None
    def corewise_inner(self, a, b):                                 # raw-coordinate dot, per element over K+C
```

**`ManifoldGeometryOps`** (`:146-236`):

- `curvature_precompute` returns the tuple `(fv_lift_factors(frame, lift_damping), Z, ip_sweep)`, where
  `Z = tv_to_t3(frame, misfit_gradient)` and `ip_sweep = t3_precompute_inner_product_sweep(Z, base_point)`.
- With `groups`, the lift factors also carry the per-group damped tied factors (§C.11), and Z uses the plain
  (untied) embedding, which represents the same ambient tensor.
- `lift` / `lift_transpose` delegate to §C.3.
- `tangential_curvature` → `t3_inner_product_corewise_hessian_from_sweep`.
- `corewise_inner` → `cw.corewise_stack_dot(a, b, a[0][0].ndim - 2)`.

**`CorewiseGeometryOps`** (`:239-308`):

- `curvature_precompute` returns `None`; `lift` / `lift_transpose` are the identity; `tangential_curvature` returns
  `None` (the exact corewise Hessian is GN + `K_{Aᵀr}`, DP §4.3); `corewise_inner = inner`.
- `lift_damping` is ignored.

All are methods over fields, with the math in public functions (`parameters_not_closures.md`). Hash/eq are unchanged:
no new fields.

### C.9 `backend/optimizers.py` — `LocalModel`, `Problem`, Steihaug CG, trust region

**`LocalModel`** (`:105-161`) gains one field, appended with a default:

```python
    curvature:   typ.Any = None   # (geom_curvature, None) -- the per-model curvature companion; None = GN model.
                                  # A LEAF beside sweep/geom_aux: it holds arrays, so it flows as traced data.
```

- **Decision I-3:** the mode is *derived* from `curvature is not None`. There is no separate static mode field
  ("prefer minimal dataclasses — derive, don't store", CLAUDE.md Workflow).
- The pytree structure differs between modes (None is an empty pytree), so GN and exact compile separately, each once.
- The tuple's second slot is reserved for a kind-side stored sweep if I-11 flips to "store"; it is `None` today.

Methods:

```python
    def hvp(self, p):                 # Π[𝒥ᵀ𝒥Πp + Lᵀ K_N L Πp] + H_R p      (curvature term iff self.curvature)
    def curvature_hvp(self, p):       # Π Lᵀ (K_{Aᵀω²r} − K_{grad f}) L Π p  (raises if curvature is None)
    def quadratic(self, p):           # gn_quadratic(p) + curvature_quadratic(p)   -- the model actually solved
    def curvature_quadratic(self, p): # ⟨LΠp, K_N LΠp⟩ via geom.corewise_inner (keeps K+C); no Lᵀ, no Π
    # gn_quadratic / gradient / jacobian / retract: UNCHANGED (gn_* keeps meaning Gauss-Newton)
```

- **Conflict with DP §4.3** ("add `⟨p, Lᵀ(K − K')L p⟩` to `gn_quadratic`"). That would silently move the Cauchy steps
  of `_mc_sgd_step` (`:357`) and `gradient_descent` (`:424`) and misname them. Instead a new `quadratic` is added,
  and `newton_cg` switches `:737` to it. In GN mode `quadratic is gn_quadratic`, numerically.
- The curvature body inside `hvp` / `curvature_hvp`:

```python
    geom_curv, _ = self.curvature
    base = self.geom.base_point(self.frame)
    theta_hat = self.geom.lift(self.frame, Pp, geom_curv)
    K = self.kind.corewise_hessian(theta_hat, self.residual, self.sample, base, self.sweep)
    Kt = self.geom.tangential_curvature(self.frame, theta_hat, geom_curv)
    if Kt is not None:                       # static (Python None) -- resolved at trace time
        K = cw.corewise_sub(K, Kt)
    y = self.geom.lift_transpose(self.frame, K, geom_curv)
```

- `hvp` projects `𝒥ᵀ z + y` once (`Π` linear; the shared post-pass is included via `aux`).

**`Problem.local_model`** (`:190-201`):

```python
    def local_model(self, x_cores, sample=None, data=None,
                    hessian: str = 'gauss_newton',      # 'gauss_newton' | 'exact'
                    lift_damping: float = LIFT_DAMPING_DEFAULT) -> LocalModel:
```

In exact mode, after the GN build:

1. Structural check `kind.has_corewise_hessian`.
2. `g_misfit = geom.project(frame, kind.transpose(residual, ...), aux)`. This is the **misfit** gradient: not
   `LocalModel.gradient`, which adds `g_R` at `:138-139` (that would inject a spurious `−K_{λX}`; checklist item 11).
   It costs one extra transpose per Newton step; caching it on the frozen model was rejected because the model is a
   jit argument.
3. `geom_curv = geom.curvature_precompute(frame, g_misfit, aux, lift_damping)`.
4. `dc.replace(lm, curvature=(geom_curv, None))`.

`least_squares_problem` is unchanged.

**Pytree** (`:799-805`): `partition_static((m.sample, m.frame, m.sweep, m.residual, m.geom_aux, m.curvature))`. This
works for `None`, plain tuples and registered dataclasses (the `geom_aux` precedent). Unflatten passes `curvature`
by keyword.

**Steihaug–Toint in the existing body** (decision I-5: one body, `radius = inf` ≡ today, bit-for-bit). The state grows
to `(p, r, d, rs, i, ok, lm, tol2, maxiter, radius, boundary)`. `_cg_cond` is unchanged (`:540-545`; it still reads
`ok`).

```python
def _cg_body(state):
    p, r, d, rs, i, ok, lm, tol2, maxiter, radius, _boundary = state
    xnp, _, _ = get_backend(False, tree_contains_jax(r))
    inner = lm.geom.inner
    Hd = lm.hvp(d)
    dHd = inner(d, Hd)
    pos = dHd > 0.0
    alpha_cg = xnp.where(pos, rs / xnp.where(pos, dHd, 1.0), 0.0)   # the :561 expression, untouched
    finite = xnp.isfinite(radius)
    rad2 = xnp.where(finite, radius * radius, 0.0)                    # no inf arithmetic anywhere
    pp, pd, dd = inner(p, p), inner(p, d), inner(d, d)
    over = finite & pos & (pp + 2.0 * alpha_cg * pd + alpha_cg * alpha_cg * dd >= rad2)
    hit = finite & ((~pos) | over)                                    # step to the boundary
    disc = xnp.maximum(pd * pd + dd * (rad2 - pp), 0.0)
    tau = xnp.where(hit, (-pd + xnp.sqrt(xnp.where(hit, disc, 0.0))) / xnp.where(dd > 0.0, dd, 1.0), 0.0)
    alpha = xnp.where(hit, tau, alpha_cg)
    p = cw.corewise_add(p, cw.corewise_scale(d, alpha))
    r = cw.corewise_sub(r, cw.corewise_scale(Hd, alpha))
    rs_new = inner(r, r)
    keep = pos & ~hit
    beta = xnp.where(keep, rs_new / rs, 0.0)
    d = cw.corewise_add(r, cw.corewise_scale(d, beta))
    return (p, r, d, rs_new, i + 1, keep, lm, tol2, maxiter, radius, hit)
```

- Bit-identity at `radius = inf`: `hit = over = False`, `alpha = alpha_cg`, `keep = pos`, so p/r/d/rs are unchanged
  to the bit. The three extra dots cost ~nothing next to `hvp`.
- The body is module-level and closure-free, and the radius rides in the state — the `:527-539` rule and
  `scan_body_principles.md` §5: the radius changes every TR iteration, so it can never be a closure.
- `_cg_core(local_model, rhs, tol, maxiter, use_jit, radius)` puts `radius` last. `_jitted(_cg_core, (4,))` keeps
  `use_jit` as the only static argument; `radius` is traced, `xnp.asarray(radius, rs0.dtype)`.
- It returns `(p, i, rs, ok, boundary)`, and `_cg_solve(local_model, rhs, tol, maxiter, use_jit, radius=math.inf)`
  likewise.
- **Blast radius:** the three call sites that destructure four values change in the same commit:
  - `tests/backend/test_optimizers.py:218` and `:223`;
  - `tests/test_dispatch.py:595`;
  - `optimizers.py:730`.

  The `dev/archive/review_2026-08-22/repros/*` scripts are archived and not updated.

**`newton_cg`** (`:646-770`) new keywords, appended:

```python
        hessian:          str = 'gauss_newton',              # 'exact': the Riemannian (manifold) / Euclidean (corewise) Hessian
        lift_damping:     float = LIFT_DAMPING_DEFAULT,      # exact only: eps = lift_damping * ‖X‖_HS
        step_control:     typ.Optional[str] = None,          # None -> 'line_search' (GN) / 'trust_region' (exact)
        tr_radius0:       typ.Optional[float] = None,        # None -> the GN Cauchy step length ‖g‖³ / gᵀH_GN g
```

- `cg_forcing_power` changes from `float = 0.5` to `typ.Optional[float] = None`. It resolves to 0.5 for GN (as today)
  and 1.0 for exact (DP §4.3: η = O(‖g‖) for quadratic convergence). This is non-breaking for every existing call.
  Decision I-7.
- The `g0norm_newton` / `g0norm_cg` chained fallback (`:717-718`) is untouched. The forcing term stays relative to
  `cg_ref`, so it composes with the queued zero-reference plan (`dev/zero_reference_g0_plan.md` §4), which changes
  only what `cg_ref` defaults to.

Loop changes:

1. `lm = problem.local_model(x, hessian=hessian, lift_damping=lift_damping)` — **only when the last step was accepted**
   (a rejected TR step keeps x, lm, g, gnorm: same frame, no new sweep).
2. `radius = inf` unless `step_control == 'trust_region'`. Then `radius = tr_radius0 or ‖g‖³/max(lm.gn_quadratic(g),
   tiny)`, computed at iteration 0 (one forward), falling back to `‖g‖` if not finite and positive.
3. `p, cg_i, cg_rs, cg_ok, cg_hit = _cg_solve(lm, neg_g, cg_tol, cg_maxiter, use_jit, radius)`.
4. The descent guard (`:734-736`) stays. Steihaug iterates decrease the model monotonically, so it only fires on
   p = 0.
5. `pHp = float(lm.quadratic(p))`; `pKp = float(lm.curvature_quadratic(p))` in exact mode.
6. **Line search** (unchanged code path, `:738-751`), with `predicted = α·slope + ½α²·pHp`. The model's pHp makes ρ
   consistent with the model actually solved.
7. **Trust region:**
   - Single trial `x_trial = lm.retract(p)` (α = 1), `f_new = objective(x_trial)`.
   - `predicted = slope + ½·pHp`; `ρ = (f_new − f)/predicted` if `predicted < 0`, else NaN.
   - `accept = isfinite(ρ) and ρ > _TR_ACCEPT (0.1)`.
   - If ρ is not finite or `ρ < 0.25`: `radius = 0.25·‖p‖`. This uses ‖p‖, not Δ: an interior step would otherwise
     re-solve to the same p.
   - Else if `ρ > 0.75 and cg_hit`: `radius = 2·radius`. Nocedal–Wright Alg. 4.1; the constants are module-level
     (I-6).
   - If `radius < _TR_MIN_REL·max(‖x‖, 1)` the loop stops: the TR analog of `ls_failed` under `'stop'`;
     `on_line_search_failure` is documented as line-search-only.
8. The Armijo-along-retraction path remains the GN default and the opt-in fallback for exact
   (`step_control='line_search'`). Every combination is legal; exact + line search is safe because Δ = inf
   truncates on nonpositive curvature (today's behaviour) and the descent guard catches p = 0.

**`NewtonInfo`** (`:602-635`), appended, all `Optional[...] = None`:

| field | meaning |
|---|---|
| `hessian` | `'gauss_newton'` / `'exact'` |
| `tr_radius` | Δ used for this solve |
| `tr_accepted` | whether the TR step was accepted |
| `cg_boundary` | CG stopped on the TR boundary |
| `pKp` | curvature part of `pHp` |
| `lift_sigma_min` | min over all `tucker_svals` / `bond_svals` (host read of the companion) |
| `lift_eps` | the ε used |

- `cg_truncated` keeps its meaning: nonpositive curvature with Δ = inf. `pHp` / `rho` docstrings change from "GN" to
  "the model solved".
- `_NEWTON_SCALAR_FIELDS` (`:638`) grows automatically; `test_newton_cg_diagnostics` compares sets, so it passes
  unchanged.
- In TR mode `alpha = 1.0` and `ls_steps = ls_failed = None`.

### C.10 Uniform mirror (`ufv_*`, `utv_*`, `ut3_*`, `uniform_fitting.py`, geometry twins)

- **`ufv_lift_factors(frame_data, lift_damping)`** (`ufv_operations.py`):
  - Mask once (`ufv_apply_frame_masks`, `ufv_masking.py:64`), then delegate to the polymorphic zipper on supercores
    (as `utv_project_ut3_onto_tangent_space` does at `utv_operations.py:582-583`).
  - `S_iᵀ` and `T_i` are d-vectorized `'d...'` einsums; padded rows/cols are 0, hence σ = 0 and f = 0.
  - ε per C element uses the masked `ufv_base_point_tangent` norm (`geometry.py:105-123`).
  - Uniform `T` is the square-bond `(d,)+C+(r,r)`.
- **`utv_corewise_lift` / `utv_corewise_lift_transpose`** (`utv_operations.py`):
  - Mask the input variations on entry. The output carries the **corewise-frame variation masks**
    `ufv_variation_masks((tk, tk, tt, tt))` (= `UniformCorewiseGeometryOps.var_masks`, `geometry.py:458-464`).
  - The transpose output carries the frame's gauge-shifted masks over K (`ufv_variation_masks_over_stack`,
    `ufv_masking.py:170`).
  - Masks are host `np`; supercores are `xnp` (CLAUDE.md dispatch exception).
- **`ut3_inner_product_corewise_transpose` / `ut3_precompute_inner_product_sweep` /
  `ut3_inner_product_corewise_hessian_from_sweep`:**
  - Z = `utv_to_ut3(frame_data, g)` is **masked first** (`ut3_apply_masks`, as at `utv_operations.py:574`). Its gappy
    `[Q|P]` TT masks and honest-zero boundaries are consumed through the mask, never a prefix slice
    (`testing_strategy.md` "phantom rank").
  - `d < 2` raises with the `utv_to_ut3` message (I-12).
- **Sampling** (`ut3_sampling.py`, object-type grouping, `naming_conventions.md` §module map): `ut3_*_corewise_hessian`
  wrappers that mask once, pack `ww`/`pp`, and delegate to the polymorphic ragged-type functions of §C.4–C.6.
  - Those functions gain d-prefixed uniform branches: maps vectorized over `d` (no Python loop — the unroll trap,
    `sampling_derivatives.py:271-277`); sweeps via `xscan`.
  - New uniform strings are in §D.
- **Kinds** (`uniform_fitting.py:69-233`): `corewise_hessian(v_sc, r, sample, base_sc, sweep)` overrides.
  - The point is `(base_sc[0], base_sc[1], self.shape, self.masks)` (the `_point` shim, `:83-86`; precedent
    `fitting.py:311`).
  - The sweep is the uniform layout `(mb, packed, shape, psweep)` (`utv_sampling.py:124-154`; jets `:238-281`).
- **Geometries:** the `UniformManifoldGeometryOps` / `UniformCorewiseGeometryOps` exact members. `corewise_inner` is
  the masked `utv_corewise_inner` (`utv_operations.py:410-435`) over the corewise var masks.
- **New `UniformManifoldGeometryOps.from_frame(frame_data, plain_masks, sharing)`.** It builds `var_masks =
  ufv_variation_masks(frame_data[5])` **without orthogonalizing**, the same derivation the kinds use
  (`uniform_fitting.py:88-91`).
  - **Conflict found:** `GaussNewtonModel._bgeom` (`fitting.py:311-314`) calls `from_point`, which runs
    `ut3_orthogonal_representations`. That is a full re-orthogonalization per model build, and it must not run inside
    a jit trace. The exact-mode frontend uses `from_frame`, and `_bgeom` switches to it as well (bit-identical masks,
    asserted in a test).
- **Varying-rank C** (`uniform_ranks_and_varieties.md` "tangent layer keeps the varying-rank stack"):
  - ε, the factors, the clip and the masks are all per element. A lower-rank element's padded slots have σ = 0, hence
    f = 0, so no coupling.
  - Masked-out directions lie in `ker(L)` and `ker(Lᵀ)`, so the CG reductions are unpolluted.

### C.11 Shared factors (`backend/sharing.py`, `ManifoldGeometryOps.groups`, `SharedGeometry`)

- **Refactor** `_tied_solve` (`sharing.py:644-664`) into `_tied_filtered_solve(shared_data, gi, V, s_filter)` plus
  its adjoint `_tied_filtered_solve_transpose(shared_data, gi, Y, s_filter)`.
  - The filter is `linalg.filtered_reciprocal(s_g, eps, rcond)` (§C.1).
  - `eps=None` is today's clipped pinv, bit-identical; the existing tied-projection tests pin it.
- **Shared lift on group modes:**
  - `U̇_g = M_g^{+ε} [V_{i₁}; …]`, the same `U̇_g` placed at every member `i ∈ g` (Φ_shared = Φ∘T, T duplicates the
    factor).
  - Singleton modes use the per-mode lift; TT slots are unchanged (untied).
- **Adjoint:**
  - `Y_g = Σ_{i∈g} Y_U_i` (Tᵀ);
  - `[V_out_{i₁}; …] = U_M diag(f(s_g)) Vt_M Y_g`, split by `row_splits`;
  - then `Π_sh` in `hvp`'s project via `aux`.
- **Gradient:** `N_sh` uses the tied misfit gradient `Π_sh 𝒥ᵀω²r`, which `geom.project(..., aux)` already returns
  (DP §4.3). `D²Φ_shared = D²Φ[T·,T·]` follows automatically.
- **Padded restarts** (`sharing_internals.md` §"Padded restarts"): new `s_g` levels are exactly zero, so the filter
  is 0. The tied channel gets no curvature in the new directions, exactly as it gets no first-order motion; the
  escape through the TT variations is untouched (those slots use the per-bond `T` lift, damped near zero).
- **Shared COREWISE:** the curvature is `project_mean(K_{Aᵀr} θ̂)`, the same post-pass and metric as its GN Hessian
  (`sharing_internals.md` "One principle, two formulas"). No lift; no formula swap.
- **Oracle caveat:** the shared retraction is *not proven* second order (the DP proves only the unshared T3-SVD).
  Shared tests therefore use the block-sum oracle and a dense Lagrangian Hessian built from the tied parameterisation
  (§F), not FD along the shared retraction.

### C.12 Safe mode — preconditions vs caveats (`numerical_contracts.md` classification rule)

**No new enforced preconditions.** On the frontend:

- `hessian(p)` / `curvature_hessian(p)` / `quadratic(p)` / `curvature_quadratic(p)` take the existing same-frame guard
  `_require_at_frame` (`fitting.py:101-118`).
- Orthogonality stays a construction invariant of the model (`numerical_contract_catalog.md`, GaussNewtonModel row).

New **caveats**, documented and never enforced:

1. "Exact" means ε = 0 at a numerically full-rank point. Otherwise the curvature is damped toward GN in the
   small-σ directions. Zero-padded restarts and the zero start live there by construction
   (`rank_continuation.md`; `sharing.md` "full shared rank is a diagnostic").
2. On a hand-built frame with a wide `T_i` (`rL < rR`), `L⁺` is the least-squares lift.
3. Shared: exact when `s_g > 0`.

The backend stays check-free.

### C.13 Frontend

**`t3toolbox/fitting.py` `GaussNewtonModel`** (`:184-403`):

- Field `curvature: typ.Any = None`, appended. The pytree registration (`:748-754`) gains it as a leaf; the
  positional unflatten is updated.
- New methods:
  - `hessian(p)` — the model's Hessian: GN, plus curvature when built exact;
  - `curvature_hessian(p)`;
  - `quadratic(p)`;
  - `curvature_quadratic(p)`.
- `evaluate(p)` becomes mode-aware (`c + gᵀΠp + ½quadratic(p)`). Its docstring states the GN-mode identity
  `= ½‖r + 𝒥Πp‖²` and that the exact mode is the second-order model of `f∘R`. `gn_hessian` / `gn_quadratic` are
  unchanged.
- The name stays `GaussNewtonModel` (decision I-4).
- Factories (`:534`, `:554`, `:574`, `:633`, `:684`, `:707`) and `_uniform_model` (`:454-488`) gain
  `hessian='gauss_newton'` and `lift_damping=LIFT_DAMPING_DEFAULT`. They share a helper
  `_with_curvature(model, hessian, lift_damping)` that calls the backend geometry (via `ragged_backend_geometry` /
  `from_frame`) and the kind; all 7 construction sites are covered.
- **`t3toolbox/optimizers.py` `newton_cg`** (`:318-439`): explicit `hessian=` and `lift_damping=`. `step_control` /
  `tr_radius0` ride `**kwargs` to the backend, documented in the docstring. `_setup` is unchanged.
- **`t3toolbox/manifold.py`:** the retract docstring at `:1381-1386` adds "second order on a minimal, numerically
  full-rank frame". The matching sentence goes in the `numerical_contracts.md` retract row.

### C.14 Diagnostics (`backend/optimizer_display.py`)

`_format_header` (`:121-144`) appends ` | TR Δ %.2e %s κ %.2f σ %.1e` (`✓` accepted / `✗` rejected; κ = `pKp/(pHp −
pKp)`) **only when `info.hessian == 'exact'` or `info.tr_radius is not None`**. The CG symbol gains `∂` for a
boundary stop.

GN-mode output is byte-identical, so the `format_newton_iter` doctests (`:160-187`) are unchanged. Any new exact-mode
doctest is run-and-pasted (`doctest_style.md`).

---

## D. Batching contract (frame-inner `W + K + C`)

| function | inputs (stacks) | output | machinery |
|---|---|---|---|
| `filtered_reciprocal`, `damped_pseudo_inverse` | `s`: C+(k); `M`: C+(m,n); `eps`: C | C+… | elementwise + batched SVD over C |
| `fv_center_factors`, `fv_lift_factors` | frame C | C | `'...'` einsums, zipper scan |
| `tv_corewise_lift`, `…_transpose` | frame C, factors C, variations K+C | K+C | `'...'` (C innermost → factors broadcast over K free) |
| `t3_precompute_inner_product_sweep` | Z C, θ C | C | `'...'` |
| `t3_inner_product_corewise_hessian_from_sweep` | θ̂ K+C; Z, θ, sweep C | K+C | `'...'`, tangent zipper scan |
| `t3_apply/entries_corewise_hessian_from_sweep` | θ̂ K+C; c W+C; ww W; θ C; sweep W+C | K+C (W summed) | grouped `contract` (independent blocks: W on samples, K on θ̂, C on cores) |
| `t3_probe_corewise_hessian_from_sweep` | θ̂ K+C; z̃ len d W+C+(N,); ww W | K+C | grouped `contract` |
| `t3_*_corewise_derivatives_hessian_from_sweep` | jets `(o+1,)+W+C`; dotted `(o+1,)+W+K+C`; input jets `(2,)+W[+K]+C` | K+C | grouped `contract` with `trs`; probe: chunked W |
| `LocalModel.hvp / curvature_hvp` | p K+C (CG: K = ()), model C | K+C | composition |
| `LocalModel.curvature_quadratic` | p K+C | K+C (0-d unstacked) | `geom.corewise_inner` → `corewise_stack_dot(a, b, n_stack=len(K+C))` (not `corewise_dot`, which collapses every axis) |
| Steihaug `‖p+τd‖` | unstacked (optimizer) | 0-d | `geom.inner` (per element; equals HS on gauged) |

**Rules applied.**

- Per-frame data (factors, ε, Z, the inner-product sweep) carry **C only** and are never tiled to K+C
  (`batching_and_stacking.md` §3; `weighted_internals.md` "batches with the FRAME").
- The residual carries W+C and **no K**. K enters only through θ̂.
- K placement follows the `_sigma_step` precedent (`probing.py:363-377`): on the carry (self-inferred), on the
  variation core (`len_C` supplied from the C-only frame core), or on the dotted input (self-inferred).

**New grouped strings** (must enter `HISTORICAL`, `tests/test_contractions_interpreter.py:70`; the AST sweep
`scan_vocabulary`, `:200-211`, forces it). The supplement shown is one sufficient choice from `c11`. The implementer
decides each by hand (`contractions_internals.md` §9), preferring the existing analog's letter — e.g. the analog
`'WCi,KCio->WKCo'` records `'C'`:

```
'WKCa,WCi,WCb->KCaib' : W        'WCa,WKCi,WCb->KCaib' : W        'WCo,KCio->WKCi' : W|C      'WCo,WKCa->KCao' : W
'trs,tWCa,KCaib,rWCi->sWKCb' : W  'trs,tWCa,Caib,rWKCi->sWKCb' : ''
'trs,rWKCa,sWCi,tWCb->KCaib' : W  'trs,rWCa,sWKCi,tWCb->KCaib' : W
'trs,rWKCa,Caib,tWCb->sWKCi' : '' 'trs,rWCa,KCaib,tWCb->sWKCi' : W  'trs,rWCa,Caib,tWKCb->sWKCi' : ''
'dWKCa,dWCi,dWCb->dKCaib' : W     'dWCa,dWKCi,dWCb->dKCaib' : W    'dWCo,dKCio->dWKCi' : W     'dWCo,dWKCa->dKCao' : W
+ the probe-derivative tangent strings, decided in S5b (one HISTORICAL row each).
```

**W-chunking.** Only the probe-derivatives tangent assembly is chunked, consistent with `docs/chunking.md` ("transpose-only
concept"; the r² blow-up lives in the assembly).

- It engages on uniform + jax only, and needs ≥ 3 chunks for the bound.
- It honours the kind's `chunk_size` and the `'auto'` estimator, subject to the S5b measurement (§C.6).
- Sharding: the two-hole is a map-reduce over W like the transpose, so the `shard_map` + `psum` recipe applies
  unchanged. No dynamic slice on a sharded W is introduced beyond the existing chunk loop.

---

## E. Costs and memory (estimates; measured in S0/S7 and recorded in `verification_records.md`)

Per sample, leading order: `N` = mode size, `n` = Tucker rank, `r` = bond. Following `docs/fitting_and_optimization.md`
§4.7, these are regime-qualified, not promises.

| operation | flops / sample | notes |
|---|---|---|
| GN matvec (apply), given sweep | `2dNn + 6dnr²` | the DP's `2d(Nn+3nr²)` |
| apply two-hole, recompute default | `2dNn + 15dnr²` | `sh` (3), `ẋi`, `μ̇` (3), `ṡ` (3), `dĠ` (3), `dẋh` (3), `dU̇` |
| exact matvec / GN matvec, apply | **≈ 2× (N-dominated) to ≈ 3.5× (r-dominated)** | vs the DP's "2–2.5×" (`:65`, `:491`); store-mode (I-11) saves the `sh` sweep (~ −10%) |
| probe two-hole / probe GN matvec | ≈ 3–4× | recomputes `ν_c`, `η_c` and three tildes (~5 sweeps), plus dotted forward (3+3+3), tildes (6+6), `dx̃i` (6), assemblies (9+4) |
| jets | × (o+1) banded parts, × (o+1)² full convolutions | the probe-derivatives `src` order-scan dominates |
| `K_{grad f}` (T3–T3 at doubled rank 2r) | `O(d[Nn·n_Z + n²r² + n r³])` | m-independent |
| lift / `Lᵀ` | `O(d[n² N + n r³])` | m-independent |
| per Newton step | + `d` small SVDs, + 1 transpose (misfit gradient), + Z build | — |

**Memory.**

- Recompute default: no new W-scaling *resident* state. Transients per matvec are the dotted edge variables
  (`d·W·K·r` each, K = 1 in CG) — for apply ~3 extra `μ`-sized arrays, for probe ~9.
- Store mode (I-11 alternative):
  - apply/entries add `sh` (+1 `μ`-sized array, ~ +40% of the lean sweep);
  - probe adds `ν_c`, `η_c` and the three tildes (~ +1.25× the full probe sweep).
- The curvature companion (factors + Z sweep) is m-independent and `O(d r²)`.

---

## F. Tests (numpy-only numerics; jax only in dispatch / the jvp oracle)

New file `tests/backend/test_curvature.py`. Each test class follows the existing pattern:

- `subTest` over structures × stacks;
- the stack matrix: none, C, K, K+C, multi-axis C, multi-axis W;
- primitive subTest params (xdist);
- per-test seeding;
- `np.allclose` / relative-error asserts; never `==` on runtime classes.

**Structures** include:

- minimal well-conditioned points (TT-SVD core, the DP oracle pitfall);
- non-minimal frames: Tucker slack `(6,6,6),(4,4,4),(1,2,2,1)` and TT slack `(1,9,9,1)`;
- d = 1, d = 2, d = 4;
- rank-1 bonds;
- a zero-padded `resize`;
- the zero point.

| test | asserts | oracle |
|---|---|---|
| T1 factors | `T_i == zipper`, `S_iᵀ == V_i U_iᵀ`, spectra == dense unfolding spectra | F1 |
| T2 lift identity | `tv_to_dense(t3_corewise_frame(base), L v) == tv_to_dense(frame, v)`, gauged + ungauged, minimal + both slack kinds, ε = 0 | F2, F3 |
| T3 lift adjoint | `⟨Lv, y⟩_raw == ⟨v, Lᵀy⟩` per K+C element (`corewise_stack_dot`), several ε | — |
| T4 damping | ε → 0 continuity; ε = 0 finite at padded restart; zero point → lift 0 except last slot; `curvature_hvp ≡ 0` at the zero start | F4 |
| T5 two-holes | every `t3_*_corewise_[derivatives_]hessian` and `t3_inner_product_corewise_hessian` vs **block-sum** (exact) and vs central FD of the one-hole (~1e-9) | F5 |
| T6 symmetry | `⟨a, K b⟩ == ⟨b, K a⟩`; `⟨p, hvp q⟩ == ⟨q, hvp p⟩` (exact mode) | — |
| T7 retraction FD | `⟨p, hvp p⟩` vs `d²/dt² f(tv_retract(tp))`, Richardson over h ∈ {1e-2, 5e-3, 2.5e-3}; unshared only, at minimal TT-SVD-conditioned frames | F6 |
| T8 gauge laws | `K_N L p_vert = 0` (raw, ε = 0) with the generators of `tests/backend/test_tv_operations.py:121-132`; `hvp(p) == hvp(Πp)`; `Π hvp(p) == hvp(p)` | F7, F8 |
| T9 dense Lagrangian | small dense: `hvp` vs the FD Hessian of `θ ↦ f(Φ(θ)) − ⟨grad f, Φ(θ)⟩` pushed through `L` (the DP §4.6 Oracle 1); level-1 control | — |
| T10 regularized | exact hvp with `IdentityRegularizer(λ)` vs FD of `f + ρ` along `tv_retract`; the curvature uses the misfit gradient | — |
| T11 weights | ω per mode (probe) / per order (derivatives): two-hole uses `ω²r` | block-sum with `_apply_weight(r,2)` |
| T12 shared | tied lift identity; adjoint; exact hvp vs dense Lagrangian Hessian of `Φ∘T`; padded shared restart → finite, no curvature in gated channel | — |
| T13 corewise exact | `COREWISE` exact hvp == `GN + K_{Aᵀr}` (block-sum); FD of `θ ↦ f(Φ(θ))` Hessian | — |
| T14 Steihaug unit | ToyModel (`test_optimizers.py:201-225` pattern): indefinite diagonal → boundary step with `‖p‖ = Δ`; PD + small Δ → boundary; Δ = inf → **bit-identical** to the 4-tuple results (the updated existing test) | — |
| T15 TR loop | rejection keeps x and reuses the model (counter on `local_model` calls); radius update rules; collapse stop; history fields present | — |
| T16 convergence rate | seeded small apply fit with nonzero residual (d = 3, `(6,6,6)`, ranks 2, W = 200, 1% noise; start = 15 GN steps): exact tail q-quadratic (`log‖g_{k+1}‖ / log‖g_k‖ ≥ 1.6` over the last two steps), `|1 − ρ| < 0.05`; GN control linear (`‖g_{k+1}‖/‖g_k‖ ∈ (0.05, 0.95)`) | — |
| T17 uniform | per element vs ragged over `_CONFIGS` (incl. varying-rank C), from nonzero full-rank starts (`uniform_equivalence_contract.md` full-rank qualifier); **exact output masks** derived non-circularly (lift: corewise var masks; `Lᵀ`, hvp: frame var masks over K); **garbage-padded inputs** (`1e6·(1−mask)`) in frame supercores and θ̂; isolated rejections (d = 1 exact; kind without `has_corewise_hessian`) with a mutation check (`testing_strategy.md` "Checklist for a new uniform op") | ragged |
| T18 frontend | `GaussNewtonModel(hessian='exact')` matches `LocalModel` bit-for-bit (extends `test_matches_backend_local_model`, `tests/test_fitting.py:595`); `_bgeom` via `from_frame` == via `from_point` masks; same-frame guard on the new methods | backend |

**Jax.**

- `tests/test_dispatch.py`: jit each new op (lift, `Lᵀ`, all `*_corewise_hessian`, T3–T3, exact `LocalModel.hvp`,
  ragged + uniform). A stray `np.*` on a tracer raises.
- New `test_exact_local_model_rebuilt_each_step_is_one_jit_cache_key`, mirroring `:556`.
- New `test_cg_radius_is_not_stale`, mirroring `:581`: two radii, one compile, distinct results.
- New `tests/test_curvature_jvp_oracle.py`, `skipUnless(jax)`, float64 via the x64 context: the two-hole equals
  `jax.jvp` of the corewise transpose for all six kinds and T3–T3 (DP route 2 as oracle only).

**Oracle sweeps** (`tests/test_oracle_sweep.py`):

- Always-on O2 model sweep: add exact-mode model rows (hvp vs block-sum/dense).
- Slow tier (`T3TOOLBOX_SLOW_TESTS=1`, a required release gate): add `hessian='exact'` to the optimizer matrix.
- The existing GN rows must pass **unchanged** — the bit-identity guarantee of §C.9.

**Contractions.** HISTORICAL rows (§D). `TestVocabularyOracle` then checks every new string over the W×K×C shape
matrix automatically.

**Budget.** The suite is ~7 min at `-n auto` today. The new always-on tests are small (d ≤ 4, ranks ≤ 3); T16 ≤ 3 s or
it moves to the slow tier deliberately.

---

## G. Docs, CHANGELOG, doctests

**User docs.**

- `docs/fitting_and_optimization.md`, new §4.12 "The exact Hessian and the trust region":
  - when it helps: ρ far from 1, a linear tail — the DP §2.4 rate `≈ |1 − ρ_tr|` stated as guidance;
  - the API, cost, damping and the continuation phase-in;
  - how to read Δ/κ/σ in the display;
  - a seeded doctest printing structure/booleans only (e.g. "exact converged in fewer Newton steps: True"), run and
    pasted, with a blank line before the closing fence;
  - plus a §5 guidance bullet and §2.1 kwargs.
- `docs/rank_continuation.md`:
  - padded restarts are automatically GN in the new directions (f(0) = 0);
  - optionally carry `tr_radius0` across levels;
  - the `g0norm` pin interacts with the forcing term (power 1.0 in exact).
- `docs/chunking.md`: one paragraph — the exact-mode probe-derivatives tangent assembly uses the same `chunk_size`
  (plus the S5b measurement).
- `docs/numerical_contracts.md`: exact-Hessian rows (caveats 1–3 of §C.12; no preconditions); the retract row's
  second-order sentence.
- `docs/naming_conventions.md`: `corewise_hessian` as a `corewise` example; `gn_*` = Gauss–Newton only; the new
  `tt_zipper_tangent_*` in cataloged exceptions.
- `docs/transposes.md`: a short pointer section "Differentiating a corewise transpose: the corewise Hessian".
- `docs/sharing.md`: one paragraph on the exact Hessian on shared geometries (tied lift; s_g zeros gate curvature).

**Contributor docs.**

- New `docs/contributor/curvature_internals.md`, added to the `contributor_guide.rst` toctree:
  - DP §2 promoted: the three forms, the Lagrangian, the T3-SVD second-order proof;
  - the lift and its exactness on non-minimal frames;
  - the block-sum oracle;
  - the two-index-jet composition rule;
  - decisions I-1…I-15 as settled;
  - measurements.
- `docs/contributor/deferred_and_rejected.md`:
  - L3 (rejected);
  - jvp/FD routes as shipped implementations (rejected; oracles);
  - storing curvature sweeps (deferred);
  - preconditioning / LM / NL2SOL β (deferred);
  - slot norms of N (deferred);
  - per-mode ε (deferred).
- `docs/contributor/verification_records.md`: S0 numbers and the cost/memory measurements.
- `numerical_contract_catalog.md`: the caveat entries.

**Release notes.**

- `CHANGELOG.md` `[Unreleased]`:
  - Added: `hessian='exact'`, trust region, the new backend functions, `NewtonInfo` fields.
  - Changed: `cg_forcing_power` default `None` (resolves to 0.5 as before in GN), `GaussNewtonModel.evaluate`
    mode-aware (identical in GN).
  - Internal: `_cg_solve` returns 5 values.
- `docs/release_notes.md` upgrade note: custom `Geometry` / `SamplingKind` need the exact-mode members **only for
  `hessian='exact'`**.

**`dev/` routing** (CLAUDE.md handoff ritual):

- `dev/riemannian_newton_curvature_plan.md` → `dev/archive/riemannian_newton_curvature_plan_2026-10-25.md` with a
  resolution banner, once S12 lands.
- `dev/curvature_oracles/` → ported into T7/T9, then archived.
- `dev/HANDOFF.md` and the CLAUDE.md Current-state pointer refreshed.

**Gates.** `sphinx -W`; module and page doctests on both numpy generations.

---

## H. Ordered slices (each one commit, independently green: `pytest tests/ -n auto` + doctests + `sphinx -W`)

| slice | content | gate tests |
|---|---|---|
| **S0** (measurement, research repo — not a library commit; decision I-16) | block-sum route (zero new code) wired into a scratch Steihaug loop on the Hilbert levels + one T3Polynomial level: `1 − ρ_tr`, tail rate, σ_min(S/T), ‖N‖, ε sweep {0, 1e-8, 1e-6, 1e-4, 1e-2}, TR constants, Δ0 policy, forcing power | numbers → `verification_records.md` |
| S1 | `linalg.filtered_reciprocal` / `damped_pseudo_inverse`; `fv_center_factors` (+ `fv_shared_frame_data` refactor, bit-identical); `LiftFactors`, `fv_lift_factors`; `tv_corewise_lift[_transpose]` (ragged) | T1–T4; existing sharing tests unchanged |
| S2 | `_project_t3_ungauged` refactor (bit-identical); `t3_inner_product_corewise_transpose`; `tt_zipper_tangent_*`; `t3_precompute_inner_product_sweep`; `t3_inner_product_corewise_hessian[_from_sweep]` (ragged, incl. d = 1) | T5 (T3–T3), T6 |
| S3 | apply + entries two-holes (ragged) + HISTORICAL rows | T5, T6 |
| S4 | probe two-hole (ragged) | T5, T6 |
| S5a | apply/entries derivatives two-holes (ragged) | T5, T6, T11 |
| S5b | probe derivatives two-hole + chunked tangent assembly + memory measurement (ragged; chunk path exercised on uniform in S9) | T5, T6, T11 |
| S6 | protocols: `SamplingKind.corewise_hessian` + flag (6 kinds); Geometry exact members (Manifold/Corewise); `LocalModel.curvature`, `hvp`/`quadratic`/`curvature_*`; `Problem.local_model(hessian, lift_damping)`; pytree | T7–T10, T13; dispatch jit of exact hvp + one-cache-key |
| S7 | Steihaug body/state, `_cg_core/_cg_solve` 5-tuple (+3 call sites), TR loop, `NewtonInfo` fields, display, `cg_forcing_power=None` | T14–T16; `test_cg_radius_is_not_stale`; existing GN optimizer tests + O2 unchanged |
| S8 | frontend: `GaussNewtonModel.curvature` + methods + 7 construction sites + pytree; `optimizers.newton_cg` kwargs; `_bgeom` → `from_frame` | T18 |
| S9 | uniform: d-prefixed branches of all new polymorphic kernels; `ufv_lift_factors`, `utv_corewise_lift[_transpose]`, uniform T3–T3; uniform kinds/geometries exact members; `from_frame` | T17; dispatch (uniform) |
| S10 | shared (ragged): `_tied_filtered_solve[_transpose]` refactor (bit-identical), tied lift, shared corewise | T12 |
| S11 | shared (uniform) | T12 uniform twin + T17 rules |
| S12 | docs, CHANGELOG, release notes, doctests, oracle-sweep rows (always-on + slow), jvp oracle test, `dev/` archive + HANDOFF/CLAUDE.md | full release gate incl. `T3TOOLBOX_SLOW_TESTS=1` |

S1–S5 are pure additions: no consumer changes, and every existing test is untouched. S6 is the first consumer change,
and is still GN-default.

---

## I. Decisions for Nick (my recommendation · the alternative)

1. **Source of S_i/T_i.**
   - Rec: derive on demand by zipper inside `curvature_precompute`, once per model (`sharing_internals.md`
     decisions 1–2).
   - Alt: return them from `t3_orthogonal_representations` (widens a hot return contract; construction byproducts
     are not reproducible across layer conversions — review S9).
2. **Mode API.**
   - Rec: `hessian='gauss_newton' | 'exact'` on `newton_cg`, `local_model` and the factories. "exact" is right on
     both geometries; "riemannian" is wrong on COREWISE.
   - Alt: `curvature: bool = False`, or `hessian='newton'`.
3. **Mode encoding in `LocalModel`.**
   - Rec: derived from `curvature is not None` (one leaf field; no redundant static).
   - Alt: an explicit static `hessian` field in aux.
4. **Frontend name.**
   - Rec: keep `GaussNewtonModel` (shipped public API since 2026.0.0) and add `hessian`/`curvature_hessian`/
     `quadratic`/`curvature_quadratic`; `evaluate` becomes mode-aware.
   - Alt: rename to `LocalModel` / `NewtonModel` with a deprecated alias; or keep `evaluate` GN-only.
5. **Steihaug.**
   - Rec: one CG body with Δ = inf ≡ today, bit-for-bit.
   - Alt: a separate `_steihaug_body` (two bodies to keep in sync).
6. **TR policy.**
   - Rec: TR default in exact mode; accept ρ > 0.1; Δ ← ¼‖p‖ when ρ < ¼; Δ ← 2Δ when ρ > ¾ on the boundary;
     Δ0 = GN Cauchy length; constants module-level, not kwargs.
   - Alt: Armijo default for exact too, or exposing the constants.
7. **Forcing power.**
   - Rec: default `None` → 0.5 (GN) / 1.0 (exact).
   - Alt: keep 0.5 everywhere and document.
8. **Damping ε.**
   - Rec: one scalar relative to ‖X‖_HS per C element (σ(S), σ(T) are unfolding singular values of X, all ≤ ‖X‖);
     provisional default `LIFT_DAMPING_DEFAULT = 1e-6`, **fixed by S0**.
   - Alt: per-mode/per-bond scales (σ_max of each factor), or the DP's `‖N‖/λ_min(H_GN)` (needs an eigensolve —
     **conflicts** with "nothing new per outer iteration", DP §4.5).
   - The tension (J-2): large ε gives a clean phase-in at restarts but kills curvature exactly in the small-σ
     directions that drive GN's linear tail.
9. **Default Hessian.**
   - Rec: GN stays the default for 2026.3; revisit after S0 + one release of field use.
   - Alt: exact by default once measured.
10. **Two-index jet layout.**
    - Rec: companion arrays (dotted quantities carry K, undotted do not).
    - Alt: a stacked length-2 core-order axis (forces K-broadcast of all undotted jets).
11. **Store vs recompute** of residual-dependent adjoint state for `K_{Aᵀr}`.
    - Rec: **recompute per matvec** (house low-memory preference, `fitting_internals.md`; +~10% apply, +~25% probe
      per matvec, zero new resident W-memory; simpler protocol — one kind method).
    - Alt: store per model in the reserved `curvature[1]` slot (faster matvec, +40% / +125% W-memory).
12. **Uniform d = 1 exact.**
    - Rec: structural `NotImplementedError` (mirrors `utv_to_ut3`); ragged d = 1 supported.
    - Alt: a special-case T3–T3 path for d = 1 uniform.
13. **Diagnostics.**
    - Rec: Δ, accept, boundary, `pKp`, σ_min, ε only; defer N-slot norms (a rank-continuation signal, DP §4.4).
    - Alt: include slot norms now.
14. **SamplingKind surface.**
    - Rec: one optional method + `has_corewise_hessian` flag; note it against `OPEN_QUESTION_extension_surface.md`
      (it widens the open surface; the question stays open).
    - Alt: close the surface (exact mode only for the six built-ins via a registry).
15. **Block-sum oracle.**
    - Rec: a test helper only (`tests/backend/_curvature_oracles.py`).
    - Alt: a public reference function (it is ~8× the cost and doubles the rank — a trap as API).
16. **S0 before S6.**
    - Rec: yes, in the research repo (`CLAUDE.md` "Research experiments → a separate research repo"); it fixes I-6,
      I-7, I-8 with numbers.
    - Alt: implement first, tune later (risks baking in guessed defaults and doctests).

---

## J. Risks and open questions

1. **Indefinite, ill-conditioned CG.** Downstream saw unpreconditioned CG stall on the full Hessian
   (`dev/zero_reference_g0_plan.md` §6). The TR bounds the steps but not the CG iteration count. Mitigation: S0
   measures CG iterations per Newton step. If it is ≫ the GN count, the follow-up (deferred) is a GN-CG flexible
   preconditioner or LM `H + μΠ`.
2. **ε tension.** See I-8. A wrong default either wastes the curvature in the tail or destabilises restarts. Hence
   S0, and a knob documented per continuation level.
3. **Probe-derivatives kernel complexity** (S5b, the six-term adjoint-wired jet steps). There is no pre-verification
   beyond the oracle. Mitigation: the block-sum oracle is exact, so every term placement is checkable to 1e-15 before
   the slice is green; the `_trs` dense reference twin is optional.
4. **Memory of the S5b tangent assembly** may exceed the GN assembly. It is measured; the estimator is extended if
   needed.
5. **float32 under jax.**
   - `K_{Aᵀr}` and `K_{grad f}` are each large (~‖Aᵀr‖/σ) while their difference is ~‖N‖/σ.
   - Early on `grad f` is large, but the gauged lift keeps us off the vertical space where the cancellation is total
     (F7: 1.3e4 vs 1e-11). Late, `grad f → 0`.
   - Expected relative noise ~1e-7·‖grad f‖/‖N‖. S9 measures it with the dispatch smoke tests at float32.
6. **Shared retraction order.** Second order is unproven, so a TR on shared geometries uses a possibly first-order
   retraction: ρ ≠ 1 even with an exact model. This degrades to superlinear at worst; documented, measured in S0.
7. **Compile time.** The exact CG body is larger (two-holes inside `lax.while_loop`). It compiles once per run, but
   first-iteration latency grows. Measure in S9 and record.
8. **`_bgeom` → `from_frame` switch (S8).** It touches the GN path. It is bit-identical by construction (same mask
   derivation as the kinds) and asserted by T18; any mismatch blocks the slice.
9. **Open — `hessian` on the stochastic optimizers?** A subsampled Newton (mc_sgd with exact curvature) is plausible
   later; out of scope.
10. **Open — interaction with the zero-reference g0 plan** (queued, `dev/zero_reference_g0_plan.md`). The forcing
    term is reference-relative, so both compose. Its addendum's objective-decrement stop is still wanted in exact mode
    (the tail terminates faster, but continuation still wants a decrement stop).
11. **Open — `tr_radius0` across continuation levels.** A zero-padded restart changes the scale. Rec: recompute the
    Cauchy Δ0 per level (default) and let users pass the previous level's final Δ.

---

## Conflicts between the design plan (DP) and the code — consolidated

| # | DP says | code reality | resolution here |
|---|---|---|---|
| X1 | §4.3: add curvature to `gn_quadratic` *and* keep GD/mc_sgd Cauchy on it | the same method serves both (`optimizers.py:357`, `:424`, `:737`) | new `quadratic`; `gn_*` keeps GN meaning |
| X2 | §4.1/§4.3: keep S_i/T_i from the frame construction; ride on `LocalModel` | the construction discards them; caching docs forbid frame state | zipper on demand inside the curvature leaf (I-1) |
| X3 | §4.2: `tv_project_t3_onto_tangent_space` at `(U,G,G,G)` is ∇⟨Z,Φ⟩ | it gauge-projects at the end (`:417-419`) | `_project_t3_ungauged` refactor |
| X4 | §4.2: frame sweep reused | true for apply/entries (F9); probe `ν`/`η` run through R/O | recompute corewise `ν`/`η` for probe kinds |
| X5 | §4.2 route 3 "NumPy backend" | the two-hole is `xnp`-agnostic | the only shipped route, all backends incl. jit |
| X6 | §4.7: jvp route first | Nick: two-hole is the deliverable; jax not in this env | block-sum oracle (numpy, exact) + jvp as a jax-gated oracle |
| X7 | §4.6: "hvp of a pure-gauge direction is zero" | false under orthogonal Π (F8) | test `K_N L p_vert = 0`, `hvp(p) = hvp(Πp)`, `Π hvp = hvp` |
| X8 | §4.5: 2–2.5× | estimate 2–3.5× (apply), 3–4× (probe) | measure; regime-qualified |
| X9 | §4.1: S_i from "the centre-core SVD" | `S_iᵀ = ⟨O_i, H_i⟩` by contraction (`sharing.py:552`) | contraction; SVD only to damp-invert |
| X10 | §4.4: ε ~ ‖N‖/λ_min(H_GN) | needs an eigensolve per step | ε relative to ‖X‖; DP scale as an S0 diagnostic |
| X11 | §4.3: residual raw, apply ω² | `LocalModel.gradient` includes `g_R` (`:138-139`) | curvature uses the misfit gradient only |
| X12 | §4.1: non-minimal → "slack directions carry no curvature" | slack is harmless: the pinv is an exact one-sided inverse on every `t3_orthogonal_representations` frame (F3) | caveat only for hand-built frames with wide `T` |
| X13 | (frontend) | `_bgeom` re-orthogonalizes per uniform model build (`fitting.py:311-314`) | `UniformManifoldGeometryOps.from_frame` |
