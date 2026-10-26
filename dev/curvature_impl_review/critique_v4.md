# Critique of `plan_v4.md` (round 4, sign-off)

Critic review of the final candidate. Repos `/home/claude/t3toolbox` @ `920d776` and `/home/claude/t3polynomial` @
`388090d`, both read-only; all runs used `python -I -B`. "SCR" means the `plan_review` scratch directory.

**Bottom line.** v4 applies round 3's two MINORs (m29, m30) and three NITs (n15–n17) correctly. I verified each edit
against the code and re-ran `c27` and `c28`. I diffed the plan body against v3: the only changes are these five items
and their test, slice and decision entries; nothing else moved. No BLOCKER, MAJOR or MINOR remains open. Every open
choice is either settled or listed in §I for Nick.

**VERDICT: AGREE.**

---

## 0. Verification of the v4 edits

| item | v4 change | verified |
|---|---|---|
| **m29** — the frame's masks are reachable | `corewise_hessian(core_directions, residual, sample, frame, sweep, base_point)`; the uniform kind reads `frame[5]` (C.6, C.9 call line `:687`, C.10, T23, T24) | ✓ (1) |
| **m30** — dtype ε | `ε = _resolution_eps(lm.residual)` = `np.finfo(dtype).eps`, maximised over floating leaves; host-side, T3-agnostic (C.9 step 8; I-25; T15(e)) | ✓ (2) |
| **n15** — T16 ρ bound | ρ asserted only where `|pred| > 1e3·res`; bound `|1−ρ| < 0.1` | ✓ (3) |
| **n16** — `lift_damping` no-op | a non-None value is a structural `ValueError` with `hessian='gauss_newton'` and on any corewise geometry (C.13, T18) | ✓ (4) |
| **n17** — tilde share | "~10–15 %" (F20, E, I-11) | ✓ matches my re-run (10.4–11.4 %) and the writer's 13–15 % |

**(1) m29.**

- The new argument order extends `forward(variations, sample, frame, sweep)` and `transpose(residual, sample, frame,
  sweep)` (`backend/fitting.py:338`, `:345`).
- `c28` re-run: the sweep's `mb` holds no masks in all four cases (minimal and TT slack, on manifold and corewise
  uniform frames). In each case `ut3_apply_masks(θ, frame (um, lm)) == (mb[0], mb[2])` bitwise, so θ, θ̂ and the reused
  `xi`/`mu` share one real extent.
- **I-24, keeping `base_point`.** Accepted. It is redundant with `frame`, but the cited rationale is the geometry's own
  docstring (`geometry.py:186-187`: "What a frame *is* belongs to the geometry, so consumers … ask for the point rather
  than indexing the tuple"), and the alternative is stated for Nick.

**(2) m30.** I checked the numpy upcast myself: float32 cores and samples give a **float64** residual for both
`APPLY.point_forward` and `probe_derivatives_kind(2).point_forward`. So ε = 2.2e-16 is correct on numpy, and float32
arises only on jax without x64, as the plan states.

The float32 end-to-end in T15(e) is jax-gated and unmeasured in this env. The plan says so and pins the rejection
bound at implementation. That is acceptable: CI has jax, and the classification itself is a host-side formula
verified on numpy by `c21`.

**(3) n15.** `c27` re-run:

- Seed 1: steps 0–5 have `|pred|/res ≥ 1.1e4` and ρ = 1.000. The outlier (ρ = 1.26) is at `|pred|/res = 5.3e-2` and is
  excluded. The run ends with `objective_floor` at `8.0e-14`.
- Seed 2: steps with `|pred|/res ≥ 9.5e6` have ρ ∈ [0.999, 1.001]. The outlier (5.8e5) is at 1.9e-7 and is excluded.
  The run ends with `gradient`.
- In `[1e-8, 1e-1]`, each seed keeps ≥ 2 ρ-asserted steps and the 3 quadratic-constant points.
- `stop_reason ∈ {gradient, objective_floor}` matches the two seeds.

**(4) n16.** This is consistent with the residual-weight precedent (`t3toolbox/optimizers.py:202-204`). It is
non-breaking: the default is `None`, and T3Polynomial's `_setup` call (`d08:113`) passes no `lift_damping`.

## Sections 1–15

Unchanged from round 3 except as follows.

| section | status |
|---|---|
| Razor and fields (§1) | OK. The protocol carries no damping; `backend/optimizers.py` stays T3-free (`_resolution_eps` is dtype-only). |
| Ragged/uniform (§2) | OK. M4 (θ, θ̂ masked; output re-masked) plus m29 (masks reachable through `frame`) plus m23 (shadow flag). |
| Safe mode (§3) | OK. No new preconditions; caveats 1–5. |
| Batching (§4) | OK. |
| Naming (§5) | OK. `_resolution_eps` is a private helper; `DEFAULT_LIFT_DAMPING`. |
| Signature style (§6) | OK. Implementation expands every signature (n5). |
| Dispatch (§7) | OK. ε is host-side; nothing new is traced. |
| Jets (§8) | OK (`c22`). |
| Protocols (§9) | OK. `corewise_hessian`'s order mirrors `forward`/`transpose` with the point appended (I-24). |
| Shared (§10) | OK. |
| Regularizer, ω, g0 (§11) | OK. |
| Tests (§12) | OK: T15(e), the T16 ρ rule, T18 n16, T23/T24 frame-mask cases. |
| Docs, routing (§13) | OK. |
| Math map (§14) | OK. |
| Other (§15) | OK. |

## Open for Nick (from §I, not blockers)

| decision | question |
|---|---|
| I-2 | the mode API name |
| I-4 | keep `GaussNewtonModel` |
| I-11 | store vs recompute |
| I-18 | promote `ragged_backend_geometry` |
| I-22 | the damping field |
| I-24 | keep `base_point` |
| I-25 | the dtype ε |

Plus the S0a/S0b measurements that confirm `DEFAULT_LIFT_DAMPING = 1e-6`, the TR constants, and the cost and memory
numbers.

## What must change for AGREE

Nothing.

At implementation:

- pin the jax-gated T15(e) rejection bound;
- re-pin T16's constants run-and-paste (S5);
- measure the S3 assembly memory and the S8 compile time, as the plan states.

VERDICT: AGREE
