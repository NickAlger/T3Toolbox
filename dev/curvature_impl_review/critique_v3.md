# Critique of `plan_v3.md` (round 3)

Critic review of the v3 plan. Repos `/home/claude/t3toolbox` @ `920d776` and `/home/claude/t3polynomial` @ `388090d`,
both read-only; all runs used `python -I -B`. "SCR" means the `plan_review` scratch directory.

**Bottom line.** v3 fixes M4 and every round-2 MINOR, and I verified each fix. Both new pushbacks hold up:

- **ν_c = Tν.** I accept it. `c26` confirms the transform is frame-specific (6.3e-16 at the manifold frame, 6.2 at the
  corewise frame), so a geometry-agnostic kind would need a new geometry→kind channel to save about 2.5 % of the
  two-hole. That is below the `precompute_and_caching.md:50-52` threshold.
- **I-18 (no public-surface change).** I accept it.

I re-ran every new script (`c19`–`c26`) and all reproduce.

No BLOCKER or MAJOR remains. Two new MINORs:

- **m29** — the uniform kind cannot reach the masks the plan says it uses;
- **m30** — the objective-floor resolution must use the dtype's ε.

Each is a one-line plan edit. Three NITs. Under the brief's rule (every MINOR fixed or deferred to Nick) the verdict
is REVISE. Once those two edits are in, I have nothing left that blocks AGREE.

---

## 0. Verification of the v3 claims

| claim | script | my run |
|---|---|---|
| M4 fix: θ and θ̂ masked, real layout, boundary bonds padded | `c19` | ✓ 5.5e-17; mutation 2.6e3 (tt mask rows 0 and d `[1 0 0]`) |
| v3 recipe (no corewise η / dx̃i; ν re-swept) | `c22` | ✓ ≤ 1.0e-15: orders 0–3, K = (2,) and (2,3), C = () and (2,), d = 4/3/1 |
| frame masks ⊆ point masks; strict at TT slack; K+C broadcast | `c23` | ✓ every line as claimed |
| ν_c = Tν is frame-specific | `c26` | ✓ 6.3e-16 manifold vs 6.2e0 corewise |
| floor rule on both degenerate cases | `c21` | ✓ (A) 3 accepted steps, then `objective_floor` at the rejection (`|pred|/res` 1.5), f = 7.7e-25; (B) `objective_floor` at the first rejection (4.9e-4) |
| T16 calibration | `c20` | ✓ GN ratios 0.56–0.68. Exact constants in `[1e-8,1e-1]`: seed 1 0.82/0.60/0.45, seed 2 0.92/0.71/0.61. ρ on those steps 1.00/1.00/1.26 and 1.00/1.00/1.00 |
| F19 probe_derivatives restart | `c24` | ✓ exactly: GN 39 / 130 hvps; exact 8 / 23, 0 rejected, for ε = 0 and 1e-6‖X‖; ε = 1e-2‖X‖ 9 / 36 |
| F20 cost | `c25` | ✓ exact/GN 3.2× and 3.0×; ν re-sweep 2.6 % / 2.5 %; undotted tildes **11.4 % / 10.4 %** (plan: 13–15 %; timing noise, n17) |
| m23 MRO shadow | inline | ✓ `[UniformProbeDerivativesKind, _UniformKind, ProbeDerivativesKind, _WwPpSample, …]`; `_UniformKind` is a dataclass, so a `ClassVar` is not a field |
| m24 field | code | ✓ `with_sharing` uses `dc.replace`; the only positional construction (`tests/backend/test_geometry.py:149`, 4 args) is unaffected by an appended defaulted field; no doctest prints a geometry repr; floats value-hash (`common.py:597-635`) |
| m27 bitwise gradient | code | ✓ the frontend and backend projections call the same `tv_/utv_orthogonal_gauge_projection` (`manifold.py:1307`, `uniform_manifold.py:1101`, `shared_geometry.py:240-247` vs `geometry.py:202-209`, `:389-398`). `_tangent_at` (`fitting.py:121-130`) adds no projection |

## 1–14. Section checks (only what changed since round 2)

- **Razor (§1).** OK. m24 is adopted as a field (I-22): `backend/optimizers.py` keeps its T3-free imports
  (`:30-40`), and the absolute ε stays a leaf.
- **Ragged/uniform (§2).** The M4 fix is right (θ and θ̂ masked, output re-masked; F15 restated; T17 and T24). The
  m23 shadow is right. One gap in the kind override, m29 below.
- **Safe mode (§3).** OK. Caveat 5 (`objective_floor` is a resolution limit, not a certificate) is correctly a caveat.
- **Batching (§4).** OK. The new D row for the `ut3_*_corewise_*hessian` wrappers is right: C-only masks broadcast
  over K (F23).
- **Naming (§5).** OK. n10 is applied (case-insensitive mode strings).
- **Signatures and dispatch (§6–7).** OK.
- **Jets (§8).** m26 is applied: the dropped η and dx̃i are verified by `c22`. The pushback is accepted (above).
- **Protocols (§9).** OK, except m29.
- **Shared (§10).** OK; n12 is applied (`s_g` in `sigma_min`).
- **Regularizer, ω, g0 (§11).** OK. The `|f|` term in the floor resolution covers ρ.
- **Tests (§12).** T15(a–d), T16 recalibrated, T17, T23 and T24 are all good. See n15.
- **Docs (§13).** OK. `curvature_internals.md` gains the θ̂-masking rule.
- **Math (§14).** OK; X15 is a correct addition.

## 15. New findings

- **m29 — MINOR: the uniform kind cannot reach the masks the plan says it uses.**
  - C.10 says the uniform `corewise_hessian(v_sc, r, sample, base_sc, sweep)` takes
    `(um, lm) = frame_data[5][0], frame_data[5][2]` "from the masked frame `mb` in the sweep".
  - The uniform jet sweep is `(mb, packed_ww, packed_pp, shape, psweep)` (`utv_sampling.py:254-266`), and its `mb`
    is `ufv_apply_frame_masks(frame_data)`. That is **four supercores and no masks**: I checked, and no bool arrays
    are present.
  - `base_sc` is a bare pair (`geometry.py:378-380`, `:483-485`). The C.6 signature passes no frame.
  - So `(um, lm)` is unreachable as written. An implementer would likely fall back to `self.masks`, which is the
    set the plan itself rejects at TT slack (I-23).
  - **Fix:** give `corewise_hessian` the frame, as `forward` / `transpose` already have it (`uniform_fitting.py:220-226`
    take `frame_data`; `LocalModel` holds `self.frame`). For example:
    `corewise_hessian(core_directions, residual, sample, frame, sweep, base_point)`. The uniform kind reads
    `frame[5]`; the ragged kind ignores it.
  - With a frame argument, `base_point` can also be dropped: both geometries' `base_point` are `(frame[0], frame[2])`.
    That is the writer's or Nick's call, but it should be stated.
  - Update C.6, C.9's call line, C.10 and T23/T24.
- **m30 — MINOR: the objective-floor resolution must use the run's dtype.**
  - `res = ε_mach·(‖ω⊙y‖·‖ω⊙r‖ + |f|)` does not say which ε.
  - `newton_cg(use_jit=True)` auto-converts to jax **float32** unless x64 is on (`fitting_and_optimization.md`
    §4.5). There, f's resolution is about 1e-7 relative. With float64's ε (2.2e-16) the test never fires.
  - At a float32 floor the loop then keeps rejecting, quartering Δ about 20 times (log₄ 1e12), with a CG solve each
    time, until `trust_region_floor`. That label is "failure", for what is a resolution limit.
  - T3Polynomial is unaffected, because its `uniform_jax` enables x64 (`t3polynomial/fitting.py:154-160`), but the
    library default user is affected.
  - **Fix:** `ε = finfo(residual dtype).eps`, the dtype-aware precedent of `safety.py:89-90`. Add a T15 case (e)
    with float32 numpy inputs: the floor is classified `objective_floor` in O(1) rejections.
- **n15 — NIT: the T16 ρ bound is too tight.**
  - The plan says its constants are "≥ 2× the observed values". The ρ bound is not: `|1−ρ| < 0.3` against an
    observed 0.26 (`c20` seed 1).
  - The 1.26 is on the step e = 3.8e-6 → 6.4e-12, where pred is near f's resolution, so ρ is noise there.
  - **Fix:** assert ρ only on steps with `|pred| > 1e3·res`, or widen to 0.5. S5 re-pins anyway.
- **n16 — NIT: `lift_damping` with Gauss-Newton is unspecified.** With `hessian='gauss_newton'` a non-None
  `lift_damping` is silently a no-op, while COREWISE + non-None raises. Make the two consistent: either raise with
  "only meaningful with hessian='exact'", or accept it everywhere and document it.
- **n17 — NIT: F20's tilde share.** F20 says the undotted tildes are 13–15 % of the two-hole; my re-run of `c25` gave
  10.4–11.4 %. State it as ~10–15 % (timing), or cite S3's measurement.

---

## Decisions

All I-1…I-23 verdicts stand as recorded in the plan's §I.

| decision | verdict |
|---|---|
| I-22 (field) | Agree |
| I-23 (frame masks) | Agree, conditional on m29 making them reachable |
| I-11 (ν pushback) | Accept; the store variant keeps the transform |
| I-18 | Accept |

## What must change for AGREE

1. **m29.** Pass the frame to `SamplingKind.corewise_hessian` (or otherwise make `frame_data[5]` reachable), and
   update C.6, C.9, C.10 and T23/T24 to match.
2. **m30.** Use a dtype-derived ε in the objective-floor resolution, plus a float32 T15 case.
3. n15–n17 can be fixed at implementation; noting them in the plan is enough. Alternatively, m29 and m30 may be
   listed in §I as deferred to Nick, which also satisfies the rule.

VERDICT: REVISE
