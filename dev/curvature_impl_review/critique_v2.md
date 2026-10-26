# Critique of `plan_v2.md` (round 2)

Critic review of the v2 plan. Repo `/home/claude/t3toolbox` @ `920d776` and T3Polynomial `/home/claude/t3polynomial`
@ `388090d`, both read-only. All runs used `python -I -B`, so no `__pycache__` was written. "SCR" means the
`plan_review` scratch directory.

**Bottom line.** v2 resolves all three round-1 MAJORs and nearly every MINOR. I verified each fix against the code
and re-ran `c1`, `c2` and `c14`–`c18`. I accept both pushbacks (m18 → I-21, and I-4). No invented names: every
cited function, file:line and doc anchor exists, in both repos.

The new probe_derivatives-first structure is sound and the pre-verification is real. I extended it to order 0,
multi-axis K and d = 1; all pass at ≤ 4e-16.

**One new MAJOR, M4.** In the actual uniform layout, the padding-transparency claim (F15) does not hold:

- garbage in the **direction θ̂** contaminates the real block of the two-hole;
- on `UNIFORM_COREWISE`, θ̂ is the raw CG direction, which does carry such garbage.

The plan never says θ̂ is masked on entry. The fix is a few lines of plan text plus one test. There are also seven
new MINORs and five NITs. Verdict: REVISE, with a short list.

---

## 0. What I verified this round

### Round-1 fixes, checked against the code

| v1 | v2 resolution | verified |
|---|---|---|
| M1 display crash | TR rows replace the `ls %d…` segment; `_cell` guards (`optimizer_display.py:63-67`); κ guarded; GN byte-identical | ✓ C.14. A rejected row's ρ = NaN formats as `nan` with `%.2f`, so there is no crash. |
| M2 double transpose | `CurvatureState.misfit_gradient`; `LocalModel.gradient` and frontend `gradient` read it | ✓ C.9, C.13; one detail in m27 |
| M3 traced host masks | `CurvatureState` / `ManifoldCurvatureData` flatten through `partition_static` (`common.py:698-724`) | ✓ Traced `partition_static`: masks and shape ints are static (`_is_static_leaf`, `:686-695`); `None` and nested registered dataclasses become dynamic leaves and jax recurses into them; `()` round-trips. The frontend registration (`fitting.py:748-754`) just passes the companion through. |
| m1 facts | `c1` no longer compares `T_{d−1}`; `c2` is fixed **and deterministic** | ✓ Two runs both gave 6.04e-15 / 1.20e-15. |
| m7 import cycle | ‖X‖² = Σσ(S₀)² (F17) | ✓ `c18`: 1.4e-15 / 4.7e-16 |
| m8 `t3_linalg` | kernel moved down | ✓ I built the import graph: `t3_linalg` does not reach `tv_operations`, `ut3_linalg` does not reach `utv_operations`, and `fv_operations` and `sharing` are mutually unreachable. Acyclic. |
| m18 pushback | keep the zipper | **Accepted.** `SharedFrameData` holds centers for nontrivial groups only (`sharing.py:472`) and no bond factors at all, while the lift needs `T_i` for every bond, so the zipper runs regardless. The saving would be d small GEMMs. |
| I-4 pushback | keep `GaussNewtonModel` | **Accepted** as the recommendation. T3Polynomial uses it heavily (`scripts/x06_hessian_fd_probe.py:61`, `x07_hessian_free_newton.py:102-106`). Still Nick's call. |

### The new scripts, re-run

| check | plan claim | my result |
|---|---|---|
| `c14` probe_derivatives two-hole | ≤ 3.6e-16, orders 0–3 | ✓ orders 1–3 ≤ 3.6e-16. Order 0 is **not run by `c14`**. I ran it, plus extras (`SCR/check_c14_extra.py`, `check_c14_d1.py`): order 0 2.4e-16 / 3.9e-16 (C = (), (2,)); K = (2,3), C = (2,) 2.9e-16; **d = 1** 3.8e-17 / 0.0 (non-trivial references, ‖ref‖ ~ 7–29) |
| `c15` shared exact | lift 5.3e-16; FD O(h²) | ✓ errors to exact 1.1e-4, 2.8e-5, 6.9e-6, 2.0e-6 (×4 per halving) |
| `c16` padding transparency | 0.0 / 0.0 | ✓ only for **interior** padding (`c16` keeps boundary bonds at 1). See M4. |
| `c17` HISTORICAL | 20 new, one sufficient letter each | ✓ every chosen letter is in `c17`'s sufficient set; the analog rows exist (`tests/test_contractions_interpreter.py:87`, `:90`, `:108`, `:151`, `:175-181`) |
| `c18` T/S transforms, F17 | ≤ 9.5e-16 | ✓ |

### New experiments (all in SCR)

**`check_c16_boundary.py`.** `c16` with the boundary bonds padded to r, as the uniform supercore actually does: the
tt edge mask row 0 is `[1,0]` in a `(d, r, n, r)` supercore. I confirmed this on a real
`UniformTuckerTensorTrain`.

- Real block: 1.4e-16.
- Padded output: **1.3e3, not 0**.

**`check_onehole_boundary_leak.py`.** The **existing** one-hole `t3_probe_corewise_derivatives_transpose` behaves the
same way:

- interior padding: 0.0;
- padded boundary bonds: padded output **2.2e2**.

Garbage in padded output is therefore normal library behaviour ("garbage don't-care", consumers mask), and not a
defect by itself.

**`check_garbage_theta_hat.py`.** Clean θ, samples and residual, with garbage only in θ̂'s padded slots:

- full padding garbage: real-block error **1.5e3**;
- garbage **only in the padded boundary-bond slots**: **1.6e3**.

**Live garbage on today's code.** Today's uniform `COREWISE` GN Hessian, `probe_derivatives_model(UNIFORM_COREWISE,
…).gn_hessian(p)`, leaves **2.7e5 / 3.9e5** in exactly those boundary slots. The `MANIFOLD` twin leaves 0.0.

**`check_objective_floor.py`.** E2's zero-residual over-ranked restart, run through the v2 floor rules:

- f = 1.1e-22, ‖g‖ = 2.7e-10, Δ0 = 5.8e-13, ‖X‖_HS = 95.
- The floor `1e-12·max(‖X‖, Δ0)` = 9.5e-11 is **above Δ0**, so the run stops after its first, accepted (ρ ≈ 1.04),
  step.
- |pred| = 8.0e-23, against the rule's `16·ε·|f|` = 3.9e-37, so it is labelled **`trust_region_floor`**, not the
  plan's claimed `objective_floor`.
- f's actual evaluation resolution `ε·‖y‖·‖r‖` is 5.4e-24.

**T3Polynomial surface.** All of the plan's downstream citations are accurate:

- `t3polynomial/fitting.py:86-112`, `:189-194`;
- `d05:232`, `b01:275`, `d07:64`, `d08:113`;
- `e01:255`, `o01:329` (`chunk_size=None`).

`stats['newton']` is consumed downstream as the per-level Newton count (`t3polynomial/fitting.py:353`, `:513`;
`tests/test_fitting_selection.py:67`, `:86`, `:91`). Under v2's bookkeeping (CG solves, rejections included,
≤ `max_newton`) those assertions keep their meaning. T3Polynomial's own `stop_reason` lives in its level records
(`fitting.py:486-551`), not in the toolbox `stats`, so there is no collision.

---

## 1. Razor and parameters-as-fields

- **OK.** Every new piece of math is a public backend function. `CurvatureState` is generic and lives in
  `backend/optimizers.py`; `ManifoldCurvatureData` lives in `geometry.py`. Leaves and statics are partitioned
  correctly.
- **m24 — MINOR: `lift_damping` is a manifold parameter threaded through the geometry-agnostic layer.**
  - It appears in the generic `Geometry.curvature_precompute(…, lift_damping)` protocol (C.7), which `COREWISE`
    ignores and every custom geometry must accept. It also appears in `Problem.local_model(…, lift_damping)` and in
    backend `newton_cg(lift_damping=DEFAULT_LIFT_DAMPING)` (C.9).
  - The plan never says where `DEFAULT_LIFT_DAMPING` is defined. Importing it into `backend/optimizers.py` from a
    T3 module breaks the stated invariant "`backend/optimizers.py` now imports no T3-specific module"
    (CLAUDE.md, the 2026-08-21 restructure).
  - Under `docs/contributor/parameters_not_closures.md` ("parameters are FIELDS"), the relative damping is a
    **field** of `ManifoldGeometryOps` and its uniform twin (`lift_damping: float = 1e-6`, value-hashed like
    `groups`). That is fine for compile-once: it is constant per run. The absolute ε that changes every Newton step
    stays the `LiftFactors.eps` leaf (C.2), which is what keeps "not stale, no recompile".
  - Then `curvature_precompute(frame, misfit_gradient, aux)` and `Problem.local_model(x, …, hessian=)` carry no
    damping. The frontend `newton_cg(lift_damping=)` / factories set it with `dc.replace(geom, lift_damping=…)`. An
    optional `_setup` kwarg is non-breaking for `d08`.
  - **Fix:** adopt the field, or list it as **I-22** for Nick with the import-invariant cost stated. If adopted,
    `test_lift_damping_is_not_stale…` becomes "absolute ε changes across Newton steps with one compile". The
    existing one-cache-key test already covers that.

## 2. Ragged vs uniform

- **M4 — MAJOR: in the uniform layout, the two-hole is not padding-transparent in its direction argument, and the
  plan never masks θ̂.**
  - **The plan's claim.** C.10: "**Mask-on-entry suffices: the two-hole is padding-transparent** (F15: 0.0
    real-block error, exactly 0 in padded output)". F15 says "**exactly 0** in padded output slots".
  - **What `c16` actually tested.** It pads only interior bonds and leaves the boundary bonds at 1. The uniform
    supercore pads the boundary bonds too (§0).
  - **What I measured.**
    - With boundary padding, the padded output is 1.3e3. That is harmless: consumers mask, and the existing one-hole
      does the same at 2.2e2.
    - Garbage in θ̂'s padded boundary-bond slots corrupts the **real block** at 1.6e3 relative. The ones-initialised
      boundary carries (`probing.py:238`, `tt_operations.py:187`) contract the padded rows of the first and last
      cores into real slots.
  - **Where the plan masks.** C.10's uniform kinds use `(base_sc[0], base_sc[1], self.shape, self.masks)` as the
    **point** (θ). The existing wrapper being mirrored, `ut3_probe_corewise_derivatives_transpose`
    (`ut3_sampling.py:239-262`), masks only the point; it has no direction to mask. Nothing says θ̂ = `v_sc` is
    masked.
  - **Where garbage enters.** On `MANIFOLD`, θ̂ is the lift output, which the plan does re-mask (n7). On
    `UNIFORM_COREWISE` the lift is the identity, so θ̂ is the raw CG direction. CG directions are built from `−g`
    and `Hd`, and the uniform corewise transpose and GN Hessian leave 2.7e5 in those boundary slots today (§0).
    `geom.inner` masks (`utv_operations.py:410-436`), so CG's scalars are unaffected, but the curvature term would
    be silently wrong.
  - **Fix:**
    1. The uniform `corewise_hessian` (kinds and `ut3_*_corewise_*hessian`) masks **both θ and θ̂ on entry**. θ̂
       takes θ's own core masks, the `_point` masks applied to `v_sc`, mirroring how the existing forwards mask
       their variations (`ufv_apply_variations_masks`).
    2. Restate F15: the real block is exact for clean padding, and padded **output** is garbage don't-care (not 0)
       under padded boundary bonds.
    3. Make T17's garbage case explicitly include θ̂'s boundary-bond slots on `UNIFORM_COREWISE`, plus a mutation
       check that removing the θ̂ mask fails it (`testing_strategy.md:143`).
  - **n13 — NIT: state the minimal-point assumption.** For θ the kind uses its plain point masks, while m9 says the
    frame's `(um, lm)`. These agree at the minimal points that uniform fitting requires (`uniform_minimal`,
    `uniform_fitting.py:347-374`). State that assumption, or take θ's masks from the frame as the lift does.
- **m23 — MINOR: the uniform kinds inherit the ragged capability before their override exists.**
  - The uniform kinds subclass the ragged ones with `_UniformKind` first, e.g. `UniformProbeDerivativesKind(
    _UniformKind, bfit.ProbeDerivativesKind)` (`uniform_fitting.py:207`).
  - From S4, when `ProbeDerivativesKind.has_corewise_hessian = True` and its ragged method land, the uniform twin
    **inherits both**. Its masking and packing override only arrives in S8 (S10–S12 for the others).
  - In between, a uniform exact model would call the ragged method on supercores with no masking. That is exactly
    the M4 failure, on a green main, with no test exercising it.
  - **Fix:** in S4, set `has_corewise_hessian = False` on `_UniformKind`, which shadows the ragged True through MRO.
    Each uniform class sets it True in the slice that adds its override. Add an isolated-rejection test: a uniform
    kind without its override raises the structural error.
- **OK.** Uniform lift masks from `frame_data[5]` (m9); real-slot `sigma_min`; Z masked first; `d`-prefixed
  branches; the varying-rank per-element argument.

## 3. Safe mode

- **OK.** No new enforced preconditions. The caveats are classified correctly, including caveat 4 (d = 1 and
  maximal-rank, where N ≡ 0, E4/E5). The backend stays check-free.

## 4. Batching (W + K + C)

- **OK.** The batching table is right. I re-checked the K placement in `c14` against `_sigma_step` precedent, and
  multi-axis K = (2,3) passes (§0). The `len_*` letters are all sufficient (`c17`).

## 5. Naming

- **n10 — NIT: case-insensitive mode strings.** `hessian=` and `step_control=` should be validated
  case-insensitively, per the ruling "geometry/kind strings are case-insensitive" (`dev/HANDOFF.md:150-157`;
  precedents `t3toolbox/optimizers.py:199`, `backend/optimizers.py:699`).
- **OK.** `DEFAULT_LIFT_DAMPING`, the `t3_linalg` / `ut3_linalg` placement, the `_trs` twins, and the
  `tt_zipper_tangent_*` cataloged exception.

## 6. Signature style

- **OK.** The n1 argument rule is applied. Some signatures remain abbreviated (C.4's `_from_sweep`, C.7) and must be
  expanded at implementation (n5, already noted).

## 7. Dispatch

- **OK.** Dispatch is inferred. The radius is in the CG state. `boundary0 = rs0 < 0.0` is dtype-stable.
  `_jitted(_cg_core, (4,))` is unchanged.

## 8. Jet machinery

- **m26 — MINOR: the recompute default does unnecessary work.**
  - `c14.two_hole` computes `eta_j` (inside `tv_precompute_probe_frame_sweep_jets(cf, …)`) and `dxts =
    compute_dxi_tilde_jets(...)` (`c14:57`, `:61`), and **never uses them**. The derivative of the corewise
    transpose needs only the **dotted** η̇ and dx̃i̇; that is the whole of `c14`'s assembly.
  - So C.4 step 1's "`eta_j = compute_eta_jets(G, …)` recomputed through G" and the dx̃i tilde should be dropped.
  - F16 also applies to the recompute default, not only to store mode. The model sweep already stores the
    manifold-frame `ν` (probe sweeps are full: `fitting_internals.md:16-28`). So `ν_c = T_i ν_i` (F16, 2.6e-16) is
    one `(r×r)·(W·r)` GEMM per core instead of a `W·n·r²` re-sweep through G. The companion already holds `T_i`.
  - **Fix:** drop η_c and dx̃i_c; use `ν_c = Tν`. Update E's probe and probe_derivatives cost rows and the I-11
    framing ("~10% re-sweep" largely disappears).
- **OK.** The two-index-jet rule. The 6-term `_adj_tilde_tangent_step`. `_trs` reference twins landing first, then
  the scan form tested equal to them. The chunked tangent assembly with S3's memory measurement.
- **n9 — NIT: F13 overstates its coverage.** It cites orders 0–3, but `c14` runs 1–3. Order 0, multi-axis K and
  d = 1 pass (§0). Add those configurations to `c14`, or cite my scripts.

## 9. Geometry / SamplingKind / LocalModel protocols

- **OK.** `CurvatureState` (misfit gradient plus geometry data). The "term present iff `geometry_data is not None`"
  static rule. `COREWISE` returning `()`. MANIFOLD d = 1 returning `None`. `corewise_inner` dropped. No reserved
  slot.
- **m24** (§1) bears on the protocol: `curvature_precompute` should not carry `lift_damping`.
- **m23** (§2) is the capability-flag inheritance rule.

## 10. Shared factors

- **OK.** F14 reproduces: the tied lift is the companion's clipped solve, and the adjoint `U_M diag(f) Vt_M Σ_i Y_i`
  is exactly the transpose of replicate ∘ `Vt_Mᵀ diag(f) U_Mᵀ`. The second order of the shared retraction is
  measured, honestly stated, at O(h²).
- **n12 — NIT: shared diagnostics.** On a shared geometry, `lift_sigma_min` should include the group spectra `s_g`.
  Those are what gate the tied lift (C.11), not the per-mode σ(S_i).

## 11. Regularizer, ω, g0

- **OK.** The misfit-only gradient and the m19 hook record.
- **OK.** T3Polynomial's zero-referenced `g0norm_newton` + `gtol_rel=1e-3` composes unchanged
  (`t3polynomial/fitting.py:189-194`).
- **OK.** The `cg_forcing_power` doc sites are listed (m5).

## 12. Tests

- **M4 → T17**, **m23 →** an isolated-rejection test.
- **m28 — MINOR: T16's constants are borrowed from a different problem.**
  - T16 fixes `C = 10` and the GN control band `(0.05, 0.95)` from E2's numbers, but T16 runs a different problem:
    `(6,6,6)`, ranks 2, W = 200, 1% noise, after 15 GN steps.
  - At 1% noise GN's linear rate ≈ ‖N‖/(σλ) may already be below 0.05, which would fail the GN control.
  - **Fix:** run T16's own problem and pin its constants from that run. The run-and-paste rule
    (`doctest_style.md`) applies to test thresholds in spirit.
- **n11 — NIT: test labels.** CHANGES m15 says "F-T20…T24", but the table has T19–T22 plus the dispatch tests.
  Fix the labels.
- **OK.** T20 (open structures, the non-circular cancellation), T21/T22 (restarts), the dispatch additions, and the
  `sweep_exact.py` routing (m6).

## 13. Docs, doctests, CHANGELOG routing

- **m25 — MINOR (I-18): `ragged_backend_geometry` is not public yet.**
  - `fitting.ragged_backend_geometry` is called "the public equivalent" of `topt._geometry_ops`, but it is not in
    `t3toolbox/fitting.py`'s `__all__` (`:86-88`), and the API page does not list it.
  - Documenting it in §2.2 is a public-surface widening: add it to `__all__` plus a CHANGELOG "Added" line.
  - The alternative is to state `_geometry_ops`' stability for downstream. Either way, the decision should say
    which.
- **OK.** Everything else in G: the §4.2 qualifier, §4.12, `curvature_internals.md`, `deferred_and_rejected.md`,
  `verification_records.md` (E2 now), the OPEN_QUESTION append, the per-slice CHANGELOG.

## 14. Math-to-code map

- **OK, verified.** Everything is as in round 1, plus:
  - F14 shared;
  - F16 T/S identities;
  - F17 ε from σ(S₀);
  - the d = 1 embedding layout (`[U; U̇]`, `[Ġ | G]` along n), which the ragged two-hole handles (§0).

## 15. Everything else

- **m22 — MINOR: the TR floor misclassifies its own motivating case and can cut off productive steps.**
  - E2's zero-residual restart is labelled `trust_region_floor`, not `'objective_floor'` (§0). `16·ε·|f|` is the
    wrong resolution for an f that is itself a cancellation result: f's absolute error is ~`ε·‖y‖·‖r‖`, here
    5.4e-24, against |pred| = 8.0e-23.
  - Because the floor is `1e-12·max(‖X‖_HS, Δ0)`, it can sit above Δ0 whenever ‖g‖ ≪ ‖X‖. Here it stopped the run
    after one accepted ρ ≈ 1 step.
  - **Fix:**
    - Label with `|pred| ≤ c·ε·(f + ½‖ω⊙y‖²)`, or `c·ε·‖y‖·‖r‖`, with c chosen from a measurement.
    - Check the floor only on the **rejection** branch, i.e. after Δ was shrunk, so a small but productive Δ0 is
      never a stop condition.
    - T15 asserts both on E2's degenerate case.
- **m27 — MINOR: use `_tangent_at`, not `_wrap`, for the exact frontend gradient.**
  - The frontend exact `gradient` "wraps `curvature.misfit_gradient`" (C.13). `_wrap` re-projects
    (`fitting.py:293-294`: `self._project(_tangent_at(...))`), so the exact gradient would be Π(Π(raw)). That is not
    bit-identical to the GN `gradient` = Π(raw), contradicting T18's bitwise assertion, and it costs a projection.
  - **Fix:** use `_tangent_at(self.frame, misfit_gradient)` plus `g_R` as today.
- **n14 — NIT: wasted display work on rejected rows.** On a rejected TR row the verbose display recomputes the
  per-(mode, order) train and validation table for an unchanged x: a validation forward per rejection
  (`optimizer_display.py:194-240`). Reuse the previous row's table, or skip it on rejected rows.
- **OK.**
  - TR bookkeeping I-17.
  - `stop_reason` I-19, apart from m22.
  - `losses[k] == history[k]['objective']`.
  - GN `stats['newton']` semantics unchanged, so T3Polynomial's per-level counts and `assertLess(st['newton'], 40)`
    stay valid.
  - `ls_steps=None` handled by `d05:232` and in the release note for `b01:275`.
  - The slice order I-20: a single release at S13, plus m23's flag rule for the S4–S8 window.

---

## Responses to the decisions

| # | verdict |
|---|---|
| I-1 | Agree |
| I-2 | Defer |
| I-3 | Agree |
| I-4 | **Accept the pushback** (keep the name); Nick's call |
| I-5 | Agree |
| I-6 | Agree, with m22 |
| I-7 | Agree |
| I-8 | Agree (E2); with m24 for *where* the parameter lives |
| I-9, I-10 | Agree |
| I-11 | Defer. Note m26: F16 makes the recompute default cheaper too. |
| I-12 | **Agree** (revised policy verified: d = 1 N ≡ 0; corewise d = 1 two-hole works) |
| I-13 | Agree, plus n12 |
| I-14, I-15, I-16 | Agree |
| I-17 | Agree |
| I-18 | Agree in substance; m25 (`__all__`) |
| I-19 | Agree, with m22's criterion fix |
| I-20 | **Agree**, with m23 |
| I-21 | **Accept the pushback** (withdraw m18) |
| I-22 (new) | `lift_damping` as a geometry field (m24). Recommend the field; Nick's call. |

X1–X14 are all correct. X14 matches E2.

---

## What must change for AGREE (prioritized)

1. **M4.** The uniform `corewise_hessian` masks θ̂ (and θ) on entry. Restate F15 and C.10: real block exact, padded
   output is garbage don't-care. T17 garbage includes θ̂'s boundary slots on `UNIFORM_COREWISE`, with a mutation
   check.
2. **m23.** `_UniformKind.has_corewise_hessian = False` in S4, flipped per uniform override, with a rejection test.
3. **m22.** Fix the floor label criterion, and check the floor only after a rejection.
4. **m27.** `_tangent_at`, not `_wrap`, for the exact frontend gradient.
5. **m24.** `lift_damping` as a geometry field, or listed as I-22, with `DEFAULT_LIFT_DAMPING`'s home stated against
   the optimizers-module import invariant.
6. **m25.** `ragged_backend_geometry`'s public status decided (`__all__` + CHANGELOG, or not).
7. **m26.** Drop η_c and dx̃i_c; ν_c = Tν; update costs.
8. **m28.** Measure T16's own constants.
9. **NITs** n9–n14 are noted for implementation.

VERDICT: REVISE
