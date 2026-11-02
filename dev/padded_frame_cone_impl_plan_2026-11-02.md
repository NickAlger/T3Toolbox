# Implementation plan — the padded frame, its subspace iteration, and the cone reference

> **STATUS: round 3 (writer, revised after the round-2 review `dev/padded_frame_cone_plan_review_r2.md`; round 1:
> `dev/padded_frame_cone_plan_review_r1.md`), 2026-11-02. Nothing is implemented.** A writer/critic plan in the
> curvature-arc format (`dev/archive/riemannian_newton_curvature_impl_plan_2026-10-29_shipped.md`): the
> critic reviews this file, the writer revises it in place and records each finding in the CHANGES table,
> until the critic agrees; Nick and the orchestrating session then adjudicate the decisions table (§9).
>
> - **Repos:** T3Toolbox @ `efe5e3d` (`main`, 2026.3.0 shipped); downstream T3Polynomial @ `66f0af1`.
> - **Amends:** `dev/archive/zero_reference_g0_plan_2026-10-27_implemented.md` (the zero reference) and the
>   2026.3.0 documentation of it ("a convention good to an O(1) factor" — false on importance-ordered bases).
> - **Motivation and measurements:** T3Polynomial `dev/zero_reference_cone_2026_11_02.md`,
>   `t3polynomial/cone.py`, `dev/t3toolbox-upstream-notes.md` §2026-11-02.
> - **[P] claims** are the writer's read-only probes (the facts table P1–P14); every one was run against `main` @ `efe5e3d` with
>   the `t3toolbox` env. The probe scripts live in the writer's session scratchpad (ephemeral); Appendix A states
>   each probe precisely enough to re-run, and Appendix B is the reference prototype they share.

**Target.** An additive feature (Nick's decision 2: **no existing default changes**):

1. a **structural, δ-free constructor of the orthonormal frame at a zero-padded rank increase**,
   `fv_padded_frame` (+ the uniform twin `ufv_padded_frame`): the old block embedded, the new slots
   completed either hint-free (the generalization of `fv_zero_point_frame`, which is its `s = 0` instance,
   bit for bit) or from a **hint** tangent by a *pinned T3-SVD* — leading directions of the hint's normal
   content, the old block pinned, every count structural;
2. the **subspace iteration** that drives it (`padded_start`): the gradient at the current frame is the
   next hint; run at oversampled ranks, truncated by a final construction;
3. the **optimizer seam**: `Problem.local_model(..., frame=)`, `newton_cg(..., start_frame=, padded_from=)`
   and a standalone `padded_start` (backend and frontend);
4. **`g0_reference='cone'`**: the norm of the projection of the zero-tensor gradient onto the tangent cone at
   zero (a best rank-`r` approximation of it), estimated — exactly for matrices — by the frame gradient at the
   iteration's fixed point; independent of any chosen completion, unlike the default Haar `'zero'` reference,
   which stays the default.

---

## CHANGES

| round | finding | change | where |
|---|---|---|---|
| 2 | **M1** the cone reference was defined as `sup_F ‖P_{T_F} G‖`, which is the rank-`2r` value, not the cone projection | defined as the norm of the projection onto the tangent cone at zero (a best rank-`r` approximation), computed at the subspace iteration's fixed point; the matrix case restated (the aligned completion gives `P_{TC}G` exactly; the supremum `‖G_{2r}‖` is attained at a staggered completion, which is not a fixed point); every "sup" / "samples" / "supremum" phrasing removed; T8(h) tests the distinction | Target 4, §0, §2.2, §2.7, §3.7, §7 (fitting §5 row, `deferred_and_rejected`, the tex paragraph, CHANGELOG), I-21 |
| 2 | **M2** `NewtonInfo.start_frame` added a key to every default run's history (decision 2) | field dropped; `stats['start_frame']` and, with `padded_from`, `stats['padded_start_gnorms']` only on runs that use a start frame; G-bit compares whole `stats` dicts | §1, §3.1, §3.6, §5.5, §6 T7(a)(b) and G-bit, §7, I-2, I-18 |
| 2 | **M3** "identical on the two layers" is false at a padded restart (the layers' old frames differ by a sign gauge); T5(f) could not pass | chose option (b): the uniform `padded_structures` builds the old frame through the ragged path, so the old block and the hint-free start are bitwise the ragged ones and the frames agree to rounding; claims restated as "the same frame up to rounding" with the near-tie caveat; cross-layer tests made gauge-invariant (T5(f), T8(c), T9(e)), bitwise only on identical inputs (T4(a), new T4(e)) | §3.3, §3.4, §4.8, §6, §7, I-28, R11 |
| 2 | **M4** nothing tested the TT sweeps or the down step against ground truth | T3(h): the critic's dense oracle per step (Tucker, `L`, `R`, `O`) at `d = 3, 4`, `C = ()` and `(2,)`, unshared and shared | §6, §8 S3 |
| 2 | **m5** `fv_operations ↔ tv_operations` import cycle | `tv_to_t3` imported inside the hinted helper; every new edge listed; G-acyclic gate in S3 | §1, §3.2, §4.4, §6, §8, I-32 |
| 2 | **m6** the docs quoted old/Haar ratios as cone/Haar | cone/Haar quoted (15–76×: 71, 76, 41, 26, 17, 15); the §0 table gains that column | §0, §7, CHANGELOG, release notes |
| 2 | **m7** `3e-4` was equated with the library's pre-2026.3.0 `1e-3` | stated as measured against T3Polynomial's own coordinate-aligned reference; no user-doc tolerance recommendation until S11 (new O-9) | §0, §3.7, §7, §11, O-9 |
| 2 | **m8** the unconverged-level result was softened, and the release notes advised `padded_from` at every level | "neutral to worse (up to 4 more Newton steps measured)"; advice limited to levels whose predecessor converged; the loop sketch passes it conditionally; S11 measures R1 at scale | §7, R1, I-27, §11 |
| 2 | **m9** the `K`-stack rejection rested on a false premise | a `K`-stacked hint is lifted to `K + C` outputs by broadcasting (the library's lift), tested in T3(g) | §3.2, §3.9, §4.2, §4.8, §5.2–§5.3, I-5, R7 |
| 2 | **m10** a caller-supplied frame needs ORTH | ORTH documented as a precondition of `local_model(frame=)` / `start_frame`, checked by the frontend in safe mode, and added to `numerical_contracts.md` | §3.6, §3.8, §5.5, §7, T9(d) |
| 2 | **m11** a shared group's old block must be tied | the old block is read from the group's first mode; TIED is a documented precondition with a frontend safe-mode check | §2.9, §3.2, §3.6, §3.8, §4.2, T9(d) |
| 2 | **m12** the hint-frame condition was mis-stated and mis-classified | restated (the hint frame contains the old frame's subspaces: any padded frame of the same old frame, any frame of `X_old` when the old ranks are minimal) and classified as a caveat; the frontend row dropped; `padded_structures` now requires minimal old ranks (structural) | §2.3, §3.2, §3.4, §3.8, I-30 |
| 2 | **m13** the oversampled structure could fall below the target | oversampled ranks = the elementwise maximum of `x0`'s ranks and `minimal(r + p, ρ + p)`; T5(h) | §3.4, §4.9, I-29 |
| 2 | **m14** no dispatch row for the uniform twin | T10 gains `ufv_padded_frame`, the uniform members and `padded_start` on both layers | §6 |
| 2 | **m15** with `old_ranks=None` the returned frame is the zero tensor's | semantics stated (the cone frame at zero; a start frame for a zero `x0` only); T9(c) restricted; the frontend stack contract and annotation | §3.5, §5.5, §5.7, §6, I-19 |
| 2 | **m16** per-pass monotonicity is not a property | P13 labelled an observation; T5(b) is a final-vs-first seeded regression | P13, §6 |
| 2 | **n17–n31** NITs | n17 no `dev/` reference in shipped text; n18 seed "tunable through `PaddedFrameOptions(seed=)`"; n19 only the cone `gnorm` may be reused as a reference (R2); n20 `_prepare_jit_frame` via `partition_static` / `rebuild_static`, the no-compile reason restated; n21 the `_cg_core` cache counter; n22 float sign literals (T1 float32 row); n23 `aux=` on every projection; n24 the same structural error for `start_frame` with a non-minimal `x0`, and the uniform padding check reads real slots only; n25 T7(g) (zero counts = the default run) and T3(i) (`d = 1`); n26 the display shows nothing; n27 the measured downstream range quoted; n28 2026.4.0 settled (I-33); n29 no corewise members (I-31); n30 the tie case (§4.1, R5); n31 the doctest margin | §3–§10 as listed |
| 3 | **m32** the uniform old-rank slice assumes prefix masks; a gappy `x0` (e.g. `uz + ua`, masks `[1 0 1 0]`) gathers the zero block, silently under `unsafe()` | `padded_structures` requires a uniform `x0`'s rank masks to be prefix masks (structural, both modes; the error names the remedy, `from_t3` of a ragged `resize`). This is stricter than the suggested inclusion check, which the probe's masks pass (`[1 0 0 0] ⊆ [1 0 1 0]`). The ragged layout is stated as numerical; the zero-padding message names the prefix layout. T5(g) gains the probe's case under `safe()` and `unsafe()` | §3.4, §3.6, §3.8, §6 T5(g), §7 (`numerical_contracts.md`), I-34 |
| 3 | **m33 / n34** the function-level import vs placement; the cited precedents were wrong | presented as a judgment call for Nick with both options. Recommended: **(C)**, a new module `backend/padded_frames.py` with downward imports only, the house's placement answer (`utv_operations.py:769-776`). The alternative **(A)**, the function-level import, is recorded as an exception if chosen. The precedent is corrected (`_bfit_pd` and `_jitted` are lazy / optional imports, not cycle breakers). G-acyclic now also asserts no function-level import. The body follows (C) | §1, §3.1, §3.2, §4.4, §5.2–§5.3, §6, §7 (`naming_conventions.md`, `api_reference.rst`, `frame_variations.md`), §8, §12, I-32 |
| 3 | **n35** two places still stated the old hint condition | "containing the old frame's subspaces (§2.3)" | §4.4, §5.4 |
| 3 | **n36** minimal old ranks are necessary, not sufficient | added "and `X_old` has full rank at them (generic, not structural)" | §2.3 |
| 3 | **n37** "so … at or slightly above" did not follow | the T3 bracket restated: at least the projection onto `F*`'s own rank-`r` space; possibly above the cone value through rank-`2r` directions; measured 0.99–1.04× | §2.7, §7 (tex paragraph) |
| 3 | **n38** user docs said "computed as"; 15–76× is a lower bound | "exact for matrices; for T3, the frame-gradient norm at the fixed point, within a few percent of the quasi-optimal value"; "at least 15–76×" everywhere user-facing | Target 4, §0, §3.7, §7 (fitting §5, `numerical_contracts.md`, release notes, CHANGELOG), I-25 |
| 3 | **n39** `stats['start_frame']` holds a label | renamed `stats['start_frame_source']` (a label) | §3.1, §3.6, §5.5–§5.6, §6 T7, §7, I-18, §12 |
| 4 | **n40** only the mask layout is structural | §3.4's uniform heading: "the mask layout is checked structurally; the content layout is the shared numerical precondition"; the zero-padding sentence reads "the real slots past the old prefix, up to `x0`'s ranks" | §3.4 |
| 4 | **n41** an AST scan sees function-level imports too | G-acyclic: the graph is built from every import node and must be acyclic, and the backend has no function-level library import | §3.2, §6, §12, I-32 |
| 4 | **n42** the naming line was placed in the "Outside the matrix" list | moved beside the `sharing` sentence (`naming_conventions.md:45-48`) | §1, §7 |
| 4 | **n43** S5 did not name the prefix-mask check | S5 lists the uniform prefix-mask check (I-34) | §8 |

---

## Probed facts this plan rests on (P1–P14)

All probes: numpy, ragged, `apply` kind on `MANIFOLD`, unit-norm Gaussian sample rows. "Continuation
case `cN`" = seed `N`, a target of norm 1 at the NEW ranks, a level fitted to `gtol_rel=1e-10` at the OLD
ranks (`newton_cg(..., g0_reference='warm_start')`), then zero-padded with `resize` to the new ranks:

| case | shape | old (Tucker / TT) | new (Tucker / TT) | `|W|` |
|---|---|---|---|---|
| c11 | (7,8,9,6) | (2,3,3,2) / (1,2,3,2,1) | (3,4,4,3) / (1,3,4,3,1) | 900 |
| c12 | (6,7,8,9,5) | (2,2,3,2,2) / (1,2,3,3,2,1) | (3,3,4,3,3) / (1,3,4,4,3,1) | 1500 |
| c13 | (8,8,8,8) | (2,2,2,2) / (1,2,2,2,1) | (4,4,4,4) / (1,4,4,4,1) | 1500 |
| c14 | (6,7,8,9,5) | (2,2,2,2,2) / (1,2,2,2,2,1) | (3,3,3,3,3) / (1,3,3,3,3,1) | 1500 |
| c15 | (9,9,9,9) | (2,3,3,2) / (1,2,3,2,1) | (4,5,5,4) / (1,4,5,4,1) | 2500 |
| c16 | (10,10,10,10) | (3,3,3,3) / (1,3,3,3,1) | (5,5,5,5) / (1,5,5,5,1) | 3000 |

"First step" = one Gauss-Newton step at the given frame (CG to `η = 0.5`, Armijo), then the library's
`newton_cg` to `‖g‖ ≤ 1e-9`; reported as `f1/f0` (objective after the first step over before) and the total
Newton count. The constructions are the reference prototype of Appendix B.

| # | claim | result |
|---|---|---|
| P1 | the constructed frame is orthonormal and its base point is exactly `X_old`, hint-free and hinted, every pass | orthogonality residual ≤ 2.2e-15; `(U,L)` and `(U,R)` reconstruct `X_old` to ≤ 1.6e-15 relative (c11, c13, d = 4) |
| P2 | at `s = 0` the hint-free construction **is** `fv_zero_point_frame` | bitwise equal, unstacked (the TT completion must enumerate each core in its own unfolding order — `R` in `(n, b)` order; a reversal-based `R` fails, measured) |
| P3 | the doubled-rank hint `Z = tv_to_t3(F', v)` at **any** frame `F'` of `X_old` contains `X_old`'s interfaces, so the overlaps `Ω^L, Ω^R` of the old chains with `Z`'s orthogonalized chains have orthonormal rows | `‖ΩΩᵀ − I‖_max ≤ 3.3e-15` with `F'` the library frame of the padded point (old block LAPACK-rotated) or a padded frame; `‖AAᵀ − I‖ ≤ 1.0e-15` for the old Tucker block in the hint's coordinates |
| P4 | the hinted construction beats the library completion on the first step, from the hint-free (Haar) start, at `p = 5`, 4 passes | `f1/f0` / Newton: library 0.860/9, 0.800/9, 0.785/9, 0.679/8, 0.696/8, 0.778/9 (c11–c16); padded 0.345/6, 0.325/7, 0.294/7, 0.149/6, 0.242/7, 0.262/7 |
| P5 | oversampling matters from the Haar start (sequential-Tucker variant, 4 passes) | `f1/f0` at `p = 0 / 2 / 4`: c11 0.243/0.168/0.144, c12 0.699/0.388/0.338, c13 0.307/0.238/0.289, c14 0.488/0.179/0.160, c15 0.241/0.250/0.240 |
| P6 | the Tucker step can be HOSVD-style (each mode's full hint environment) instead of sequential (inside the left sweep) at no measured cost — which makes it group-ready and the two TT sweeps mirror images | `f1/f0`/Newton (p = 5, 4 passes, Haar start; re-run 2026-11-02): sequential 0.135/5, 0.311/7, 0.286/8, 0.451/7, 0.241/7, 0.260/7; HOSVD-style 0.345/6, 0.325/7, 0.294/7, 0.149/6, 0.242/7, 0.262/7 (c11–c16) — neither dominates, Newton totals within one step. At `p = 0` the HOSVD-style step gives 0.246, 0.699, 0.448, 0.471, 0.241, 0.257 |
| P7 | the brief's Tucker shortcut F4 — "the leading right singular vectors of the hint's Tucker variation `V_i`" — fails at the final truncation | `V_i ⊥ U'_i` (the hint frame's WHOLE Tucker factor), so after oversampled passes, whose `U'` already spans the good directions, `V_i` carries no information: `f1/f0` 0.396 / 0.526 / 0.771 vs 0.330 / 0.292 / 0.145 with the full normal part (c11, c13, c14). The identity behind F4 is true but different: the leading left singular vectors of the hint's mode-`i` unfolding projected off `U'_i` (not `U_old`) are `V_i`'s leading right singular vectors |
| P8 | the two-sided projection (pins on the row side AND the old interface removed on the column side) beats pins alone | first-step `f1/f0`, an earlier HOSVD-style variant, 4 passes from the Haar start, `p = 0`: two-sided 0.55 / 0.73 / 0.55 / 0.39 vs one-sided 0.66 / 0.74 / 0.65 / 0.64 (c11–c14); §2.4 gives the reason (old-coupled content is captured by every completion, so spending new slots on it is waste) |
| P9 | the δ-limit of the retraction frame (`frame(R(X_old + t·v))`, `t → 0`) is **not** the target: its `U`, `L` coincide with a sequential pinned sweep (projector differences 1e-7…2e-5 at `t = 1e-5, 1e-7`), but its `R`, `O` carry `O(t)` rotations of the old block (the dense first-order coefficient `(Y − X)/t` reproduces the ε-frame to 1e-7; the pinned right chain differs at every bond left of the last) | and it is beaten on the first step: ε-limit 2 passes from the library frame `f1/f0` 0.588, 0.458, 0.403, 0.285, 0.388 (c11–c15) vs P4's padded values |
| P10 | "rotation corrections" (removing old-coupled content from the carries) are inert in the chosen construction | identical results with and without, at a converged and at an unconverged old level (to 3 digits in `f1/f0` and Newton counts); not part of the design |
| P11 | from an **unconverged** old level the padded start is neutral to slightly worse | old level stopped after 1 / 3 Newton steps: library vs padded `f1/f0`/Newton c11 0.114/19 vs 0.154/22, 0.147/17 vs 0.395/17; c13 0.193/16 vs 0.287/20, 0.288/15 vs 0.358/14; c14 0.096/24 vs 0.135/23, 0.171/18 vs 0.269/17 |
| P12 | the cone reference (s = 0, HOSVD-style construction) against the dense-frame reference `‖P_{T_F} G‖`, `F` = the frame of the dense `T3SVD_r(G)` with zero carriers, two problems × up to four ranks | Haar `'zero'` reference 0.20–0.45× the dense value; cone `p = 5`, 10 passes 1.001–1.041×; `p = 5`, 4 passes 0.989–1.016×; `p = 0`, 10 passes 0.920–0.996× (stalls at low rank without oversampling) |
| P13 | an observation, not a property: the captured norm per pass increased on every probed case (from a staggered completion one pass lowers it, §2.2) | e.g. c13, `p = 4`: 0.147, 0.167, 0.184, 0.189 (at the oversampled structure); the final target-rank value is not comparable to the oversampled ones |
| P14 | **the matrix oracle** (`d = 2`, Tucker ranks = TT rank, `entries` with every entry observed, so the ambient gradient is the residual matrix `G`): the T3 frame tangent space is the matrix one, and the iteration converges to the tangent-cone projection | at a Haar frame the library's `‖P_T G‖` equals `(‖G‖² − ‖P_U^⊥ G P_V^⊥‖²)^{1/2}` to 1e-15; `s = 0`, `r = 3`, 30 passes, `p = 0`: 13.190905958272921 vs `‖G_r‖` = 13.19090595827292; `s = 1 → r = 3`, 40 passes: 10.99395790646683 vs `(‖P_{T_X}G‖² + ‖N_{r−s}‖²)^{1/2}` = 10.993957906466825, where the library frame of the padded point captures 6.45 |

**Downstream evidence** (T3Polynomial `dev/zero_reference_cone_2026_11_02.md`; Darcy o01 and e01 cells, the
real use case — importance-ordered inputs, `probe_derivatives`, the exact Hessian): §0.

---

## 0. Why

**The zero reference is a frame projection, not an O(1) convention.** At the zero tensor every core is zero,
so the measured gradient is the ambient gradient `G = 𝒥ᵀω²(S(0) − y)` projected onto the tangent space of
whatever completion the frame carries. 2026.3.0 fixed that completion to seeded Haar Tucker factors and
documented the result as "a deterministic data-gradient scale, good to an O(1) factor". On a homogenized,
importance-ordered basis (active subspace, Hermite, KL — the common surrogate case) a Haar frame is a random
projection capturing ~`r/N` per mode, and the claim fails by one to two orders of magnitude (downstream record §1;
"old" is T3Polynomial's own coordinate-aligned zero reference, "dense" the norm of the truncated T3-SVD of `G`,
the quasi-optimal rank-`r` approximation — for matrices the cone value itself):

| Darcy o01, level (rank) | old LAPACK frame | library Haar (seed 0) | Haar seeds 1–4 | old / Haar | dense `‖T3SVD_r G‖` | dense / Haar |
|---|---|---|---|---|---|---|
| L0 (1) | 2.70 | 0.207 | 0.22–0.47 | 13.1 | 14.7 | 71 |
| L1 (6) | 15.18 | 0.622 | 0.76–1.65 | 24.4 | 47.3 | 76 |
| L2 (11) | 19.13 | 1.368 | 1.61–2.86 | 14.0 | 55.9 | 41 |
| L3 (16) | 22.32 | 2.329 | 2.93–4.48 | 9.6 | 60.7 | 26 |
| L4 (21) | 26.46 | 3.716 | 4.26–7.04 | 7.1 | 64.4 | 17 |
| L5 (23) | 29.04 | 4.327 | 5.04–8.54 | 6.7 | 66.1 | 15 |

(e01, 16 exchangeable inputs: old/Haar 2.1–6.7.) The reference feeds the stop and the CG forcing term, so a
`‖g₀‖` at least 15–76× below the cone value demanded that much tighter an absolute stop while CG sat at its `η = 0.5` cap.
The quantity that does not depend on a chosen completion is the norm of the projection of `G` onto the **tangent
cone** at zero — at zero the cone is the rank-`r` set itself, so the projection is a best rank-`r` approximation of
`G` (`‖G_r‖` for matrices; §2.7 says what is computed for T3). It is **not** the largest frame projection
`sup_F ‖P_{T_F} G‖`, which is larger (§2.2). The downstream iteration's value sat 0.2–2.4 % above the dense frame
value at ranks ≥ 6 and 8.6 % above it at the e01 rank-3 cell (record §2). Downstream, a relative tolerance of
`3e-4` against it behaved like T3Polynomial's own coordinate-aligned reference at `1e-3` (record §4) — not the
library's pre-2026.3.0 warm-start reference; the library recommendation waits for S11 (§3.7).

**The frame of a zero-padded restart is a choice the library makes badly.** A `resize` warm start is rank
deficient; its frame's new slots are LAPACK's completion on the ragged layer and the pad-safe Haar sketch's
on the uniform layer — different, data-blind, and backend dependent (Darcy level 1, same absolute stop,
Newton / CG, record §3):

| start | ragged exact | uniform_jax exact | uniform_jax GN (f after step 1) |
|---|---|---|---|
| W1 padded, library completion | 21 / 181 | **44 / 469** | cap |
| W2 padded, Haar completion (ε) | 18 / 208 | 18 / 208 | cap |
| W3 padded, gradient-informed, 1 pass (ε) | **13 / 136** | **13 / 136** | 47 / 548 |
| W4 padded, 2 blended passes (ε) | 15 / 233 | 15 / 233 | cap |
| W6 padded, Tucker cone completion, 1 / 3 passes (δ-free) | 20 / 315, 16 / 188 | | |
| C1 exact zeros, library completion | 32 / 294 | 32 / 294 | cap (8.70) |
| C7 exact zero, cone Tucker frame (δ-free) | **19 / 247** | | |

The gradient-informed completion is the robust winner and removes the backend split — but downstream could
only set it by injecting `ε`-content into the point ("a frame smuggled in as content", Nick): the TT frame
cores are derived from the core content, so a completion cannot be expressed without content, and content at
`1e-8` made the exact Hessian's curvature term reject first steps. The δ-free Tucker-only version recovered
about half the gain (W6). Measured on the orchestrator's Darcy level-1 probe: the captured `‖P_T ∇f‖` goes
15.3 (LAPACK) → 22 → 28 → 32 → 33 over passes, 8.7 → 15 → 22 → 28 → 32 from a seeded Haar start, and an
oversampled `+5` iteration reaches 37 at rank `r` after 4 passes; Newton from the enriched frame took 13–14
iterations against 21 (ragged) / 44 (uniform) from the library's completions, at the same absolute stop.

**What this plan adds**, all opt-in: a constructor that builds such a frame **structurally** (counts from the
rank recurrence, no threshold, no `ε`, the retraction and `t3_orthogonal_representations` untouched — Nick's
decision 4), the iteration that improves it, a seam to start `newton_cg` from it, and the cone reference. The
small-problem measurements P4–P5 reproduce the downstream pattern (the first Newton step removes 65–85 % of
the objective against 14–32 % from the library completion; 1–3 Newton steps saved per level), and P14 shows the
iteration reaching the tangent-cone value exactly in the matrix case.

---

## 1. Inventory — every site touched (`main` @ `efe5e3d`)

**New code**

| file | what | where |
|---|---|---|
| `backend/linalg.py` | `pinned_completion`, `pinned_leading_columns` (the two factorization primitives, §4.1) | after `damped_pseudo_inverse` (`:778-825`); `__all__` `:20-33` |
| `backend/padded_frames.py` (**new**) | `fv_padded_frame`, `ufv_padded_frame` (its `__all__`) and the private helpers `_empty_old_frame`, `_embed`, `_outside_first`, `_hint_coordinates`, `_tucker_completion`, `_pinned_left_sweep`, `_down_completion` (§4.2–§4.8). Module-level imports, all downward: `common`, `contractions`, `linalg`, `tt_operations`, `tt_orthogonalization`, `t3_orthogonalization`, `tv_operations` (`tv_to_t3`), `fv_operations` (`ZERO_FRAME_SEED`, `fv_variation_shapes`, and the private `_haar_rows` / `_identity_prefix` — cross-module private use has house precedent, e.g. `geometry`'s `ufv_operations._real_group_levels`), `ufv_conversions`, `ufv_operations` (`ufv_stack` / `ufv_unstack`), `ufv_masking` | new module (§3.2; the placement is judgment call I-32); imported by `geometry` only |
| `backend/fv_operations.py`, `backend/ufv_operations.py` | `fv_zero_point_frame` (`:288-430`), `ufv_zero_point_frame` (`:302-354`) | a "See also" docstring line each; no code change |
| `backend/geometry.py` | three optional members on the two **manifold** geometry classes: `padded_frame`, `padded_structures` (the uniform one builds the old frame through the ragged path), `require_frame_structure` (§3.4); new imports `ranks`, `t3_constructors`, `ut3_constructors`, `ut3_conversions` | `ManifoldGeometryOps` after `zero_point_frame` (`:326-333`); `UniformManifoldGeometryOps` after `:706-713`; imports `:58-72`; the corewise classes (`:392`, `:774`) unchanged |
| `backend/optimizers.py` | `PaddedFrameOptions`, `PaddedStart`, `padded_start`, the constants `PADDED_START_PASSES` / `CONE_REFERENCE_PASSES` / `PADDED_FRAME_OVERSAMPLE`, `_cone_reference_gnorm` (§3.5, §3.7) | after `least_squares_problem` (`:434-444`); `__all__` `:46-59` |
| `t3toolbox/optimizers.py` | frontend `padded_start` and the `PaddedFrameOptions` re-export (§5.7) | after `least_squares_problem` (`:197-330`); `__all__` `:57-66` |
| `docs/contributor/padded_frame_internals.md` | the design record (§7) | new; `docs/contributor_guide.rst` toctree |
| `tests/backend/test_padded_frame.py`, `tests/backend/test_padded_start.py` | T1–T6 (§6) | new |

**Changed code** (every change additive; behaviour without the new keywords — `stats` included — is bit-identical, T7(a), G-bit)

| file:line | today | change |
|---|---|---|
| `backend/optimizers.py:65-131` | `Geometry` protocol docstring: one optional member (`zero_point_frame`) | documents the three new optional members (§3.4) |
| `backend/optimizers.py:346-416` | `Problem.local_model(x_cores, sample, data, hessian, curvature_damping)`; `:405` `frame = self.geom.frame(x_cores)` | new keyword `frame=None` (preconditions ORTH and same tensor, documented); when given, `geom.require_frame_structure(x_cores, frame)` then use it (§3.6) |
| `backend/optimizers.py:516-532` | `_prepare_jit_inputs(use_jit, x0, problem)` | unchanged; a sibling `_prepare_jit_frame(use_jit, frame)` converts a given start frame's dynamic leaves (`common.partition_static` / `rebuild_static`, `:698-743`; masks and `shape` stay host) |
| `backend/optimizers.py:604` | `_G0_REFERENCES = ('zero', 'warm_start')` | `+ 'cone'`; the message in `_check_g0_reference` (`:607-618`) names it |
| `backend/optimizers.py:650-699` | `_resolve_g0_reference(problem, frame, x0, g0_reference, compute, who, hint)` | a `cone_options` argument and a `'cone'` branch (§3.7); `'zero'` / `None` / `'warm_start'` paths byte-identical |
| `backend/optimizers.py:702-763` | `gradient_descent(..., g0_reference)`; resolution `:742-745` | `+ cone_options=None`, passed to the resolution |
| `backend/optimizers.py:1123-1422` | `newton_cg(problem, x0, ...)`; iteration-0 model `:1299`; reference `:1307-1319`; the returned `stats` `:1421` | `+ start_frame`, `padded_from`, `start_options`, `cone_options` (§3.6); iteration 0 builds `problem.local_model(x0, frame=start)` when a start frame is in play; `stats` gains `'start_frame_source'` (and `'padded_start_gnorms'`) **only on such runs**; `NewtonInfo` (`:958-1028`) and the history rows unchanged |
| `backend/optimizer_display.py:121` | `_REFERENCE_NAMES = {'zero': 'zero tensor', 'warm_start': ..., 'override': ...}` | `+ 'cone': 'tangent cone'` (the converged line reads `g₀ = tangent cone`) |
| `t3toolbox/optimizers.py:420-611` | frontend `newton_cg` (explicit keywords, `:596-606` forwards) | the four new keywords (frontend types), forwarded; the minimal-`x0` structural check and the safe-mode checks — zero padding, ORTH, same tensor, TIED (§3.6) |
| `t3toolbox/optimizers.py:333-360` | frontend `gradient_descent` (forwards `**kwargs`) | docstring only: `g0_reference='cone'` and `cone_options` reach the backend through `**kwargs` |
| `tests/test_contractions_interpreter.py:74` | `HISTORICAL` (249 strings) | `+` the new strings of §4.10, each `''` (fully determined; decided by hand). `AXIS_SIZE` (`:49-54`) unchanged: every new string uses letters already sized there, chosen with distinct sizes within each string |
| `tests/test_optimizers_frontend.py:448-490` | keyword mirror + forwarding | the mirror covers the new keywords automatically; the forwarding test's `opts` gains `padded_from` / `start_options` / `cone_options` (`start_frame` asserted separately, forwarded as `.data`) |
| `tests/backend/test_zero_reference.py` | the 2026.3.0 zero-reference suite | `+` the cone cases T8 |
| `tests/backend/test_optimizers.py` | the `newton_cg` suite; default `set(stats)` pinned at `:477`, `:860` | `+` T7; the pins stay unchanged |
| `tests/test_dispatch.py`, `tests/test_jax_absent.py` | jit / jax-absent sweeps | `+` T10, T11 |

**Reused, not changed** (decision 4: the retraction and the orthogonal representation are not touched):
`tv_operations.tv_to_t3` (`:205-336`, the hint as a doubled-rank T3), `tv_retract` (`:853-892`),
`fv_conversions.t3_orthogonal_representations` (`:123-267`), `t3_orthogonalization.t3_down_orthogonalize_tucker_cores`
(`:144-175`), `tt_orthogonalization.tt_left_orthogonalize` / `tt_right_orthogonalize` (`:58-139`),
`tt_operations.tt_zipper_left_to_right` / `_right_to_left` (`:173-202`) and `tt_reverse` (`:31-44`), `fv_operations.fv_zero_point_frame`
(`:288-430`, untouched — the `s = 0` hint-free instance is tested equal to it bit for bit) with `_haar_rows`
(`:274-285`) and `_identity_prefix` (`:262-271`), `ufv_conversions.ut3frame_to_t3frame` / `t3frame_to_ut3frame` /
`ut3variations_to_t3variations` (`:141-305`), `ut3_masking.ut3_make_masks` (`:25-49`), `ut3_constructors.ut3_zeros`
(`:110-121`), `t3_constructors.t3_zeros` (`:20-41`), `ranks.compute_orthogonal_representation_ranks` (`:501-568`),
`linalg.pad_or_truncate` (`:36-77`), `ranks.compute_minimal_ranks` (`:51`), `ut3_conversions.ut3_to_t3` (`:125`),
`common.partition_static` / `rebuild_static` (`:698-743`), `contractions.contract`.

**Docs** (§7 has the text): `docs/fitting_and_optimization.md` §2.1 (the `stats` table `:96-104`), §3 (the `Geometry` members `:269-283`),
§4.11 (`:501-512`), §4.12 ("Restarts need nothing special" `:601-608`), §5 (the reference bullet `:651-706`); `docs/rank_continuation.md` (`:103-146`, `:187-196`, `:198-231`);
`docs/numerical_contracts.md` (`:192-203`); `docs/frame_variations.md` (a section between `:68-112` and `:114`);
`docs/uniform_equivalence_contract.md` (`:47-79`); `docs/sharing.md` (`:184-209`); `docs/api_reference.rst`
(`:72-102`); `docs/release_notes.md` (a new "Upgrading from 2026.3.0" head); `CHANGELOG.md` (a new top section);
`docs/contributor/curvature_internals.md` (`:171-222`, `:307`); `docs/contributor/deferred_and_rejected.md`
(`:54-62`, `:227-229`); `docs/contributor/verification_records.md` (`:91-102` + a section);
`docs/contributor/sharing_internals.md` (`:98-122`); `docs/exact_hessian_two_hole.tex` (`:904-940`; `:937` the O(1)
sentence); `docs/naming_conventions.md` (`:45-48`, beside the `sharing` example — option C of I-32); new
`docs/contributor/padded_frame_internals.md`.

---

## 2. Mathematics

### 2.1 Setting

`X_old` is a T3 at ranks `(s, σ)` with an orthonormal frame `F_old = (U°, O°, L°, R°)` whose four rank stores
are `(nU°, nD°, rL°, rR°)`; the target ranks `(r, ρ) ≥ (s, σ)` give the template stores `(nU, nD, rL, rR)` by
the recurrence of `ranks.compute_orthogonal_representation_ranks` (`nU_i = min(r_i, N_i)`,
`rL_{k+1} = min(rL_k nU_k, ρ_{k+1})`, `rR_k = min(rL_k, nU_k rR_{k+1})`, `nD_i = min(nU_i, rL_i rR_{i+1})`). Every
step of the recurrence is monotone in its inputs, so `(r, ρ) ≥ (s, σ)` implies **old ≤ new in all four
stores** — the embedding below always exists. (`s = 0` — no old block — is the zero tensor; its "old stores"
are 0 except the boundary bonds.)

A **padded frame** of `X_old` at `(r, ρ)` is an orthonormal frame `F = (U, O, L, R)` with the template's core
shapes such that

- **the old block is embedded**: `U_i[:nU°_i] = U°_i`; `L_k[:rL°_k, :nU°_k, :rL°_{k+1}] = L°_k` with zeros in the
  rest of those columns; likewise `R_k` (old rows), `O_i` (old down rows);
- **the carriers carry only the old content**: `L_{d-1} = embed(L°_{d-1})`, `R_0 = embed(R°_0)` (zero on every
  new index);
- **the new slots are orthonormal completions**: rows `nU°_i..nU_i-1` of `U_i` orthonormal and orthogonal to
  `U°_i`; the new columns of `L_k` orthonormal and orthogonal to the embedded old ones in the
  `(rL_k·nU_k)` unfolding; the new rows of `R_k` in the `(nU_k·rR_{k+1})` unfolding; the new down rows of `O_i`
  in the `(rL_i·rR_{i+1})` bond space.

Then both base-point reconstructions equal `X_old` exactly: every contraction path that enters a new index
(a new bond index, or a new Tucker slot — which only new columns use) ends at a carrier whose new slots are
zero (P1: ≤ 1.6e-15). Orthonormality holds core by core. Note what is **not** required: the `L` and `R` chains
need not be the left/right canonical chains of one common tensor beyond `X_old`. The tangent machinery needs
only per-core orthonormality — in the sum-of-terms picture the cross terms `⟨term(H_i), term(H_j)⟩` vanish by
the gauge `L_iᵀH_i = 0` and the orthonormality of `L_{<i}` / `R_{>j}`; `term(V_i)` is orthogonal to every
`H`-term by `V_i ⊥ U_i`; each term is an isometry of its coordinates — so `MANIFOLD.inner`, `project` and
the gauge are exact on such a frame, and the retraction (the T3-SVD of the doubled-rank representation, which
takes any frame) is defined there, with the first-order caveat every rank-deficient frame has (§2.2). Base-point
consistency is the one cross-chain property, and it concerns only the carriers: `fv_frame_consistency_residual`
(the `(U, L)` vs `(U, R)` reconstructions) is ≤ 1.6e-15 on every probed padded frame (P1).

### 2.2 The tangent cone, frame tangent spaces, and what a completion buys

At a rank-deficient point `X ∈ M_{≤(r,ρ)}` there is no tangent space, only the tangent cone `TC_X`. Every
completion `Q` gives a **frame tangent space** `T_Q` (the span of the `2d` terms at the frame), and

- `T_Q` does not depend on the carriers (`L_{d-1}` and `R_0` appear in no term — `docs/frame_variations.md`), so
  "the base reset to zero" changes the base point, never the tangent space;
- `T_Q ⊇ T_X M_s` for **every** completion: the old frame's variations are among the padded frame's (a Tucker
  variation along any `U_i^⊥` direction paired with the old down rows; every TT variation with old bonds);
- every element of the cone lies in the `T_Q` of a completion aligned with it, but the union of the `T_Q` is
  **larger** than the cone: a frame tangent space holds normal parts (relative to `X`) of rank up to
  `2(r − s)`, the cone only `r − s`. The retraction truncates the excess, which is why it is not first-order
  accurate at a padded point (`docs/numerical_contracts.md`).

**The matrix case, exactly.** For `X = U_X S V_Xᵀ` of rank `s`, a completion `(U_new, V_new)` with `k = r − s`
columns each, and `N = P_{U_X}^⊥ G P_{V_X}^⊥` the **normal part** of `G`:

- `P_{T_Q} G = G − P_U^⊥ G P_V^⊥` with `U = [U_X, U_new]`, so `‖P_{T_Q}G‖² = ‖G‖² − ‖P_{U_new}^⊥ N P_{V_new}^⊥‖²`.
- The cone is `TC_X = T_X M_s ⊕ {normal parts of rank ≤ k}`, and the projection onto it is
  `P_{TC} G = P_{T_X M_s} G + N_k` (Eckart–Young on `N`), with `‖P_{TC}G‖² = ‖P_{T_X M_s}G‖² + ‖N_k‖²`.
- At the **aligned** completion — `U_new`, `V_new` the top-`k` singular subspaces of `N` — the frame gradient is
  the cone projection exactly: `P_{T_Q} G = P_{TC} G` (the normal part of `P_{T_Q}G` is
  `N − P_{U_new}^⊥ N P_{V_new}^⊥ = N_k`). The frame gradient there is the cone's steepest-descent direction.
- The aligned completion does **not** maximize the frame projection. Since `P_{U_new}^⊥ N P_{V_new}^⊥ = N − (rank ≤ 2k)`,
  `sup_Q ‖P_{T_Q}G‖² = ‖P_{T_X M_s}G‖² + ‖N_{2k}‖²`, attained by a **staggered** completion (`U_new` the top-`k` left
  directions, `V_new` the next `k` right ones). Smallest example, `s = 0`, `r = 1`: the aligned frame `(u_1, v_1)`
  captures `σ_1`; the staggered `(u_1, v_2)` captures `(σ_1² + σ_2²)^{1/2}`. So "the largest frame projection" is
  not the cone value and is not what this plan computes.

Two consequences carry over to T3:

1. **New directions must come from the normal part, two-sided.** The block `P_{U_X}^⊥ G P_{V_X}` ("old-coupled
   content": a new left direction paired with `X`'s own right interface) is captured by `T_Q` for every
   completion, so a completion that spends slots on it wastes them (P8).
2. **Power iteration on the completion finds the aligned completion.** Given the frame gradient `v = P_{T_Q}G`,
   its normal part is `N' = P_{U_new}N + N P_{V_new} − P_{U_new} N P_{V_new}`, whose column space is
   `span(U_new) + span(N V_new)` (and symmetrically for the rows): taking its leading `k` directions is a block
   power step, and its **attracting** fixed point is the aligned completion (P14). The staggered maximizer is not
   a fixed point: there the frame gradient's normal part is `N_{2k}`, whose leading `k` directions are the aligned
   ones, so one pass moves it to the aligned completion.

### 2.3 The pinned T3-SVD of a hint

A **hint** is a tangent `v` at a frame `F'` of `X_old` (any structure at or above the old frame's; typically the
Riemannian gradient at the current padded frame). The construction reads `v` only as a tensor,
`Z = tv_to_t3(F', v)` (the doubled-rank T3). When `F'` contains the old frame's subspaces — every padded frame of
the same old frame does, and so does any frame of `X_old` when the old ranks are minimal (§3.4 requires them to
be) and `X_old` has full rank at them (generic, not structural) — `Z`'s orthogonalized chains contain `X_old`'s
interfaces (P3) and every step below is the stated projection exactly. For any other hint the output is still an orthonormal padded frame of `X_old` (the two-sided
removals are contractions by overlaps of orthonormal sets, and the pins are never touched); the condition only
decides *which* directions are chosen — a caveat, not a precondition.

For each unfolding the frame needs, the new slots are the **leading directions of the hint's normal content
there, with the old block pinned**:

| slot | candidate space (rows) | the hint's content (columns) | two-sided: removed from the columns | pins |
|---|---|---|---|---|
| Tucker `U_i` (a group: one factor) | `ℝ^{N_i}` | `Z`'s mode-`i` unfolding against its own left and right chains at `i` (the mixed-canonical center; HOSVD-style, the full environment) | `X_old`'s mode-`i` row space (the old down rows, in `Z`'s coordinates) | `U°_i` |
| left bond `L_k` (k ≤ d−2) | `(rL_k · nU_k)`: the chosen left basis at bond `k` ⊗ the chosen Tucker basis at mode `k` | `Z` projected onto the left basis built so far and onto `U_k`, against `Z`'s right-orthonormal chain at bond `k+1` (TT-SVD order) | `X_old`'s right interface at bond `k+1` | `L°_k` (embedded) |
| right bond `R_k` (k ≥ 1) | `(nU_k · rR_{k+1})` | the mirror image (the left sweep run on the reversed problem) | `X_old`'s left interface at bond `k` | `R°_k` (embedded) |
| down `O_i` | `(rL_i · rR_{i+1})` | `Z`'s one-hole environment at mode `i` in the final `L`, `R`, `U` bases, restricted to the **new** Tucker slots | — (the new-slot restriction is the two-sided cut: `X_old`'s center lives on the old slots) | `O°_i` (embedded) |

Every step is one primitive (§4.1): **`pinned_leading_columns(P, A, k)`** = the leading `k` left singular
directions of `(I − PPᵀ)A`, completed by Householder against `P` when `(I − PPᵀ)A` has rank `< k`. Pinning means
the old directions are **held fixed** — they are not re-estimated from the hint, so nothing of `X_old` moves.

- **Counts are structural**: `k = nU_i − nU°_i`, `rL_{k+1} − rL°_{k+1}`, `rR_k − rR°_k`, `nD_i − nD°_i`, read off
  the two rank sets; no singular value is compared with anything.
- **A degenerate hint is completed, not thresholded**: if the hint has fewer directions than slots (or none —
  `v = 0`), the trailing columns come from the Householder process against the pins and the leading ones —
  a statement about the hint (it carries no information there), not about the data.
- **The Tucker step is per mode, the TT steps are sweeps**, both chosen for the measured reasons in P6: the
  HOSVD-style Tucker step (each mode's full hint environment) is as good as the sequential one inside the
  left sweep, makes a sharing group's step one SVD of the group's concatenated normal contents
  (`docs/sharing.md` §"What the group spectrum is", face 1), and lets the `L` and `R` sweeps be exact mirror
  images (one code path, run on the reversed problem — `fv_frame_reverse` and `tt_reverse` are exact).
- **`R` is not derived from `L`.** In `t3_orthogonal_representations` the right chain is the right sweep of the
  point's left chain. At a padded point the point is `X_old`, which carries no information in the new slots,
  so the library's `R` completion is LAPACK's — exactly the arbitrariness this feature removes. The pinned `R`
  is the right-to-left pinned TT-SVD of the hint: the same principle, mirrored.
- **`O` from the new Tucker slots.** `O_i`'s rows define where a Tucker variation (`V_i ⊥ U_i`) can live. The
  down rows of a frame are the row space of the point's center; for the limit point `X_old + t·W` that row
  space is `X_old`'s (pinned) plus the new rows of `W`'s center on the **new** Tucker slots. (The alternative — the hint's content orthogonal to the new Tucker
  factor, "where a variation would capture most" — was mixed in an earlier variant, better on one probe and
  worse on another, and has no limit reading; not adopted.)

### 2.4 Why not the δ-limit, and why two-sided

The obvious definition — "the frame of the retraction `R(X_old + t·v)` as `t → 0`" (the downstream
`ε`-enrichment, computed exactly) — is the wrong target (P9). Its `U` and `L` coincide with a sequential pinned
sweep, but its `R` and `O` contain the `O(t)` rotations of the **old** directions that the sequential T3-SVD
induces (old-coupled content at a later bond, routed back through the chain). Those rotations are tangent
content (§2.2 item 1), and the pinned construction excludes them by definition; it is also better on every
measured first step (P4 vs P9). The construction is therefore stated directly (§2.3), not as a limit.

**Two-sided** at the TT bonds follows from §2.2: at bond `k+1`, the hint's content in (a new left direction) ⊗
(`X_old`'s right interface) is captured by every completion — through `H_k` when the left direction is
gauge-orthogonal to `L_k`, through the later holes otherwise — so new columns spent on it are wasted. The
column-side projection removes it; P8 measures the difference. (Removing the same content from the sweep's
*carry* as well — the "rotation corrections" — is provably inert: the next step's column projection removes it
again; P10.)

### 2.5 Hint-free completions (the generalization of `fv_zero_point_frame`)

With no hint the new slots are a fixed convention:

- **Tucker**: the keyed Haar stream of `fv_zero_point_frame` (`RandomState([seed, k_g])`, `k_g` the group's first
  mode), its first `k = nU_g − nU°_g` draw rows as seeds, made orthonormal against `U°_g` by
  `pinned_completion` — **Haar-distributed in the complement of `U°`** (the projection of a Gaussian onto a
  subspace independent of it is Gaussian there). At `s = 0` the pins are empty and this is
  `fv_zero_point_frame`'s draw and sign-fixed QR, bit for bit (P2). Nested in `k` and rank-independent across
  modes, as there.
- **TT and down cores**: the coordinate directions of the slot's unfolding **outside** the old block, in the
  core's own unfolding order (`L`: `(a, n)` row-major; `R`: `(n, b)`; `O`: `(a, b)` — the orders
  `_identity_prefix` uses), exactly orthonormal and orthogonal to the embedded old block (disjoint supports);
  when those run out (a bond that grows while its neighbours do not — a TT-only increase at the first core is
  the common case), the remainder is the Householder completion of the old columns **inside** the old block,
  seeded by a keyed Gaussian (`RandomState([seed, i, c])`; §4.3 says why not coordinate seeds). At `s = 0` every
  coordinate is outside: the identity prefix, bit for bit (P2).

Why the asymmetry is kept from 2026.3.0: the Tucker factor is the only part of the frame that meets the data,
and a coordinate-aligned factor gives an exactly-zero hint on data with no energy on the leading
coordinates (`docs/contributor/curvature_internals.md` F32) — from which no iteration can recover; the TT
cores act on rank indices.

### 2.6 The subspace iteration

`F_{k+1} = Pin(F_k, P_{T_{F_k}} G)`: the gradient at the current frame is the next hint. At `s = 0` this is the
frame form of **unit-step Riemannian gradient descent on `½‖Y − G‖²`**: the Riemannian gradient at `Y` is
`P_{T_Y}(Y − G) = Y − P_{T_Y} G` (the base point lies in its own tangent space, `v_X`), so the unit step lands
on `P_{T_Y} G` and the retraction truncates it — `Y_{k+1} = T(P_{T_{Y_k}} G)`; since `T_Y` depends only on `Y`'s
frame, it is an iteration on frames, and the pinned construction plays the truncation's role (a T3-SVD-type
truncation of the hint: the HOSVD-style Tucker step and the TT-SVD sweeps). Each pass costs one frame sweep and one transpose (`G` is never
formed: `P_{T_F} G` is the problem's own gradient at the frame). As HOOI is for Tucker, it is a block subspace
iteration — each mode's new basis is the leading subspace of `G` applied to the other modes' current bases —
with the rate set by the gaps of `G`'s unfoldings (`σ_{k+1}/σ_k` per pass), improved by **oversampling** as in
randomized SVD: iterate at the oversampled ranks (`+p`, minimal, never below the target — §3.4), then **truncate by a final
construction** at the target ranks from the last oversampled gradient. P5 and P12 measure both effects
(without oversampling the iteration can stall at low rank — a saddle — as downstream e01 `r = 3` did). At
`s > 0` the same iteration is the power iteration of §2.2 on the completion, `X_old` fixed. In the matrix case it
converges to the tangent-cone value exactly, at `s = 0` and `s > 0` (P14).

The **start** is the hint-free padded frame at the oversampled ranks — deterministic, layer-independent, and
generic (§2.5); not the library frame of the padded point, whose completion is backend dependent (the 21/44
split of §0). The measured price of Haar over a coordinate-aligned start on importance-ordered data is about one
pass (downstream: 8.7 → 15 vs 15.3 → 22).

### 2.7 The cone reference

**What it is.** At the zero tensor the tangent cone of the rank-`r` set `M_{≤r}` is `M_{≤r}` itself (a cone is its
own tangent cone at its apex), so the projection of the ambient gradient `G` onto the cone is a **best rank-`r`
approximation** of `G`, and the cone reference is its norm. For matrices that is `‖G_r‖`.

**What is computed.** `g0_reference='cone'` returns `‖P_{T_{F*}} G‖`, the frame-gradient norm at the fixed point
`F*` of the subspace iteration (§2.6) run at the zero tensor of `x0`'s ranks from the hint-free start, oversampled
and truncated. Its relation to the cone value:

- **matrices:** `F*` is the aligned (top-`r`) completion and `P_{T_{F*}}G = P_{TC}G` exactly (§2.2), so the
  computed number is `‖G_r‖` (P14: to 1e-15);
- **T3:** a best approximation in `M_{≤r}` is not computable exactly, and the iteration is HOOI-like, so the
  computed number is an estimate of the cone value, bracketed as follows. It is **at least** the norm of `G`'s
  projection onto `F*`'s own rank-`r` tensor space (the tensors with `F*`'s Tucker factors and left chain and a free
  last core — a subspace of `T_{F*}`; that projection is a HOOI-like quasi-optimal rank-`r` approximation). It
  **can exceed** the true cone value, because `T_{F*}` also holds rank-`2r` directions. It **measured** 0.99–1.04× the
  frame value `‖P_{T_F}G‖` at `F` = the frame of `T3SVD_r(G)` (P12, 4–10 passes; 1.00–1.04× at the default 10), and
  downstream 0.2–2.4 % above that value at ranks ≥ 6 and 8.6 % above it at the e01 rank-3 cell.

**What it is not.** It is not `sup_F ‖P_{T_F}G‖`: that supremum is attained by staggered completions and is the
norm of a best rank-`2r` approximation for matrices (§2.2). It is not an average over frames either — a seed
average estimates `E_F ‖P_{T_F}G‖`, a third number. And the default `'zero'` value at one Haar frame is neither
a sample of it nor a bound on it (a random frame can capture more than the aligned one, by the staggered
mechanism): it is a different, completion-dependent quantity. What makes the cone reference independent of the
seed is that the iteration converges to the data's fixed point; the seed only starts it (P12: the Haar `'zero'`
value is 0.20–0.45× of it on the probed cells).

It is **metric-dependent** like every projection (Frobenius here; covariant under a Kronecker metric, not under
Gauss–Newton's); the metric-invariant stop would be the Newton decrement (O-4). A vanishing cone reference
(all-zero data) is a structural error for the explicit request (§3.7).

### 2.8 Why no tolerance, no per-element count, and full determinism

The pad-safe SVD's design record rejected "Method B, post-hoc masked completion" (a rank tolerance `τ` and a
per-element number of replacements) and "Method C, masked noise" (`σ ≲ η` erased; randomness breaks
determinism) (`docs/pad_safe_svd.tex` App. A). The construction has neither defect:

- every count is a difference of two structural ranks (§2.3), identical for every stack element, so stacked
  inputs vectorize with static shapes and nothing is decided per element;
- the leading directions are taken in order from an SVD and **completed** by a Householder QR against the pins;
  no singular value is compared with a threshold, and a zero hint is a legal input;
- no content is injected anywhere: the new slots carry zero content in the point (the base point is `X_old`
  exactly), so the exact Hessian's lift clips the new directions' zero singular values as at any padded
  restart (F10 of the brief; `docs/fitting_and_optimization.md` §4.12) — the downstream `ε` failure mode cannot
  occur;
- it is deterministic: the hint-free start is a keyed seeded draw (host numpy, frozen legacy stream), and the
  hint is a deterministic function of the data. The non-unique cases — the Householder completion of a
  rank-deficient hint, and a tie at a cut — are deterministic for a given input and LAPACK, as LAPACK's own
  completions are.

It sits in the lineage the pad-safe record cites approvingly: the BUG integrator's augmented basis (the basis
is gauge), Vermeylen–Vandereycken–Absil's rank increase along the normal part of the tangent cone appended as
new core columns, and DMRG's structured subspace expansion.

### 2.9 Sharing

A group's Tucker factor is one array. Its old block is read from the group's first mode, `U°_{min(g)}` — the
precondition is **TIED** (the members' old factors numerically equal, `numerical_contracts.md` §"The shared
surface"), checked by the frontend in safe mode (§3.6); with untied old factors every other member's old block
would differ and the frame would not represent `X_old`. The Tucker step concatenates the members' normal contents
(each compressed to its column space first, §4.5) and takes one pinned leading set against `U°_g` (sharing face 1);
the hint-free start draws one Haar seed per group keyed `min(g)` (the `fv_zero_point_frame` rule). The TT and down
cores are untied, as in every shared frame. At a zero-padded shared restart the new `s_g` levels are still exactly
zero (the new slots carry no content) and the tied Tucker channel is still gated
(`docs/contributor/sharing_internals.md` §"Padded restarts"); what changes is that the TT channel's first-order
mass in the new up-slots now pairs with data-informed Tucker directions instead of the arbitrary completion — the
escape route that section describes, now aimed.

---

## 3. Design

### 3.1 The pieces and the data flow

```
 backend/linalg.py         pinned_completion, pinned_leading_columns          (factorizations; no T3 semantics)
          │
 backend/padded_frames.py  fv_padded_frame(frame, old_frame, hint, groups, seed)   ── the constructor (ragged; C, and K + C for a K-stacked hint)
          │                          ▲ reused, all imported downward at module level: tv_to_t3,
          │                            t3_down_orthogonalize_tucker_cores, tt_*_orthogonalize, tt_zipper_*, tt_reverse,
          │                            contract, _haar_rows, _identity_prefix, pad_or_truncate
   (same module)            ufv_padded_frame(...)          ── unpack real blocks → fv_padded_frame → repack (host loop over the stack)
          │
 backend/geometry.py       ManifoldGeometryOps / UniformManifoldGeometryOps only (optional, duck-typed members):
                             padded_frame(...)            → the layer's constructor with the geometry's groups
                             padded_structures(...)       → (old_frame, target, oversampled, oversampled_geometry)
                             require_frame_structure(x, frame)
          │
 backend/optimizers.py     padded_start(problem, x_cores, old_ranks, options, final_gradient) → PaddedStart
   (T3-free)               Problem.local_model(..., frame=)     newton_cg(..., start_frame, padded_from, start_options,
                           _cone_reference_gnorm (g0_reference='cone')                     cone_options)
          │
 t3toolbox/optimizers.py   newton_cg(... the same keywords, frontend frames ...)    padded_start(geometry, kind, ..., x0, old_ranks)
                           PaddedFrameOptions (re-export)     safe-mode checks (zero padding, same tensor, ORTH, TIED)
```

**`newton_cg(..., padded_from=old_ranks)`, iteration 0:**

```
x0 (the padded point: X_old zero-padded to x0's ranks)
 └─ padded_start(problem, x0, old_ranks, start_options, final_gradient=False)
     ├─ geom.padded_structures(x0, old_ranks, p) ── old_frame: the frame of x0 sliced to old_ranks (on the uniform
     │                                              layer through the ragged path, §3.3); target: a frame of x0's
     │                                              structure; oversampled template + its geometry (§3.4)
     ├─ r0 = kind.point_forward(x0, sample) − data            (once: the point never moves)
     ├─ F = geom_os.padded_frame(oversampled, old_frame, None, None, seed)      (the hint-free start)
     ├─ repeat passes − 1 times:  g = geom_os.project(F, kind.transpose(r0, sample, F, kind.precompute(F, sample)),
     │                                                aux=geom_os.precompute(F))
     │                            F = geom_os.padded_frame(oversampled, old_frame, F, g, seed)
     └─ last pass: g at F (as above), then F = geom.padded_frame(target, old_frame, F, g, seed)   (the truncation)
 └─ lm0 = problem.local_model(x0, frame=F) → require_frame_structure(x0, F) → the model at F
 stats: 'start_frame_source' = 'padded' and 'padded_start_gnorms' = the per-pass norms — present only on such runs.
 The frame is rebuilt from the retracted point after the first accepted step, as today: the start frame
 influences iteration 0 only.
```

### 3.2 The constructor `fv_padded_frame`

**Inputs.** A **template** frame at the new ranks (only its core shapes and dtype are read — the
`fv_zero_point_frame` convention), the **old frame** (an orthonormal frame of `X_old`, or `None` for the zero
tensor), an optional **hint** `(hint_frame, hint_variations)`, the sharing `groups`, and the `seed`.

**Output.** A padded frame of `X_old` (§2.1) with the template's core shapes and dtype and the backend (numpy/jax)
inferred from the array inputs; stack `C`, or `K + C` for a `K`-stacked hint (§3.9).

**Structural** (checked always, both modes; each a `ValueError` naming the offending store and mode):

- the template is admissible (`fv_zero_point_frame`'s check, `:379-388`);
- the old frame's four stores are ≤ the template's, mode by mode;
- the stacks agree: the old frame's and the hint frame's stack is the template's `C`; the hint variations' stack
  is `K + C` (`K = ()` is the common case; the split is recovered from the leading axes, the library's rule —
  `docs/batching_and_stacking.md` §6);
- the hint frame's stores are ≥ the old frame's and its mode sizes are the template's (its ranks are otherwise
  free: the driver's final construction takes the **oversampled** frame as the hint of a **target** template);
  the hint variations have the hint frame's variation shapes (`fv_variation_shapes`);
- a sharing group's members have one `(rank, size)` in the template, the old frame and the hint frame; `d` agrees
  everywhere.

**Numerical preconditions** (documented, never checked in the backend; the frontend checks what it builds): the old
frame is orthonormal (ORTH), and with `groups` its Tucker factors are tied (TIED; the group's old block is read from
`U°_{min(g)}`). **Caveat** (not a precondition): the hint frame should contain the old frame's subspaces (§2.3); with
any other hint the output is still a valid padded frame of `X_old`.

**Two paths.** `hint is None` → the hint-free completions (§4.3). Otherwise the hinted construction (§4.4–§4.7).
`old_frame is None` → the empty old block (`_empty_old_frame`), so `s = 0` is not a separate code path; the
`s = 0`, hint-free, unstacked call **is** `fv_zero_point_frame` (tested bitwise).

**Placement and imports** (judgment call I-32; the recommended option). The constructor needs `tv_to_t3`, and
`tv_operations` imports `fv_operations` at module level (`tv_operations.py:23`) — so the constructor cannot sit in
`fv_operations` with a module-level import, and `tv_to_t3` cannot move below `fv_operations` (it calls
`sharing.fv_tied_ambient_directions`, `tv_operations.py:244-245`, and `sharing` imports `fv_operations`). The house's
recorded answer to a would-be cycle is **placement** — "named for its operand, placed for its imports"
(`utv_operations.py:769-776`; a module may host several families, as `sharing` hosts four). So both twins and their
helpers live in a new topic module, **`backend/padded_frames.py`**, whose imports all point downward (the §1 row);
nothing imports it but `geometry`. The alternative — keep `fv_padded_frame` in `fv_operations` beside
`fv_zero_point_frame` and import `tv_to_t3` inside the function — works at runtime, but it would be the backend's
first function-level import of a library module, an upward edge a module-level scan would miss; the
lazy and optional imports elsewhere (`t3toolbox/optimizers.py`'s `_bfit_pd`, `_jitted`'s `import jax`) defer cost or
keep a dependency optional, they do not break cycles, so they are no precedent for it. `geometry` gains
`padded_frames`, `ranks`, `t3_constructors`, `ut3_constructors`, `ut3_conversions` (none imports `geometry`). Gate
G-acyclic (S3): the backend's import graph, built by an AST scan from every import node (module- and
function-level), is acyclic, and the backend has no function-level import of a library module.

**Traceability.** All shapes come from the template and the old frame; the Haar draws and the coordinate index sets
are host numpy constants; every array operation is `xnp` / `contract` / `xnp.linalg`. So `fv_padded_frame` is
`jax.jit`-traceable (T10) — a bonus, not a requirement: its consumers call it eagerly.

### 3.3 The uniform twin `ufv_padded_frame`, and the two layers

The `ufv_zero_point_frame` pattern (`ufv_operations.py:302-354`): require concrete masks; per stack element (a host
loop over `C`, or `K + C` for a `K`-stacked hint), unpack the real blocks of the template, the old frame, the hint
frame (`ut3frame_to_t3frame`) and the hint variations (`ut3variations_to_t3variations`), run `fv_padded_frame`,
repack at the **template's** padded sizes (`t3frame_to_ut3frame(..., N=, nU=, nD=, rL=, rR=)`), require the repacked
masks to equal the template's (prefix masks — a structural error otherwise), restack (`ufv_stack`). The output masks
are the template's; the padding is exactly zero; garbage in any input's padding is never read. On **identical
inputs** the real blocks are the ragged construction's bit for bit (the same code on the same arrays). No
`pad_safe_svd` is involved: every factorization runs on unpacked real blocks.

**Across the layers** the inputs must be made identical too, and two of them would not be by default: the two
layers' frames of the same point differ by a sign gauge even at a generic point — `t3_orthogonal_representations`
(LAPACK) against `ut3_orthogonal_representations` (the pad-safe SVD), equal in absolute value only
(`uniform_equivalence_contract.md` §"Gauge-carrying operations"; the critic's probe measured entrywise differences
of order 1 with equal magnitudes to 2.7e-12). So the uniform `padded_structures` builds the **old frame through the
ragged path** (§3.4): per element, unpack the old point's real blocks, `t3_orthogonal_representations`, repack. The
old block and the hint-free start are then bitwise the ragged ones; the hints are the two layers' gradients, equal
to rounding; and the padded frames agree to rounding — the same subspaces, hence the same first step (§6 T5(f), T9(e)
test this gauge-invariantly). A near-tie at a cut (`σ_k ≈ σ_{k+1}` in a hint unfolding) can still split them, as for
any truncation. Host-side and eager by design (it reads the masks), like `ufv_zero_point_frame`.

### 3.4 Three optional geometry members

Looked up duck-typed (the `zero_point_frame` precedent, `backend/optimizers.py:78-85`), defined on the two manifold
geometries only. A geometry without them — the corewise ones (a corewise frame is the point itself: there is no
completion to choose) and any custom geometry — keeps working; `padded_from`, `start_frame`, `local_model(frame=)`
and `g0_reference='cone'` on it are structural errors naming the missing member.

| member | `ManifoldGeometryOps` | `UniformManifoldGeometryOps` |
|---|---|---|
| `padded_frame(frame, old_frame, hint_frame, hint_tangent, seed)` | `fv_padded_frame(frame, old_frame, (hint_frame, hint_tangent) or None, self.groups, seed)` | `ufv_padded_frame(...)`; the hint tangent is the bare supercore pair `project` returns, given masks from `hint_frame` (`ufv_masking.ufv_variation_masks_over_stack(hint_frame[5], C, K + C)`) |
| `padded_structures(x_cores, old_ranks, oversample)` | `old_frame = t3_orthogonal_representations(slice(x_cores, old_ranks))[0]` (`None` when `old_ranks is None`); `target = t3_orthogonal_representations(x_cores)[0]` (structure only); `oversampled = t3_orthogonal_representations(t3_zeros(shape, *os_ranks, C))[0]`; returns `(old_frame, target, oversampled, self)` | the old frame **through the ragged path**: per stack element, `ut3_conversions.ut3_to_t3` of the supercores under the old-rank masks (`ut3_make_masks(old ranks, n, r)`), `t3_orthogonal_representations`, `t3frame_to_ut3frame` at the target frame's padded sizes, then `ufv_stack`; `target = self.frame(x_sc)`; the oversampled template **and its own geometry** from `ut3_zeros(shape, *os_ranks, C)` and `UniformManifoldGeometryOps.from_point(..., sharing)` (the geometry holds variation masks for one structure) |
| `require_frame_structure(x_cores, frame)` | the frame's core shapes equal those of `x_cores`' orthogonal representation (`compute_orthogonal_representation_ranks`) + `C` | the frame's variation masks equal `self.var_masks` and its supercore shapes the expected padded ones |

`padded_structures`' structural checks (both modes): `old_ranks` has lengths `d` / `d + 1` (or `(d,)+C` /
`(d+1,)+C` int arrays on the uniform layer); on the uniform layer, `x_cores`' rank masks are prefix masks (below);
`old_ranks ≤ x_cores`' ranks elementwise; `old_ranks` structurally
**minimal** (`ranks.compute_minimal_ranks(shape, *old_ranks, sharing) == old_ranks`; the shared notion with sharing) —
an old frame with slack carries LAPACK completion directions that would be pinned as "old" and that the hint's chains
need not contain (§2.3).

The **oversampled ranks** are the elementwise maximum of `x_cores`' ranks and
`ranks.compute_minimal_ranks(shape, r + p, (1, ρ_1 + p, …, ρ_{d-1} + p, 1), sharing)` (`backend/ranks.py:51`): `+p`
on every Tucker rank and interior bond, reduced to the structurally minimal ranks (the shared notion with sharing, so a
group keeps one rank), and never below the target — so `old ≤ target ≤ oversampled` in all four stores always, and
`p = 0` is the target exactly, minimal `x_cores` or not. Per stack element on the uniform layer.

The slice of `x_cores` at the old ranks assumes the **prefix layout** `resize` produces: `X_old` in the leading slots,
zeros after them (`[..., :s_i, :]`, `[..., :σ_k, :s_k, :σ_{k+1}]`; uniform: the same supercores under the old-rank
prefix masks).

- **Uniform layer: the mask layout is checked structurally; the content layout is the shared numerical
  precondition.** Uniform rank masks may legally be gappy after `+` or `×`
  (`docs/uniform_masks_vs_ranks.md` §"Masks change under arithmetic"), and a prefix gather then reads the wrong
  slots. Example (the round-2 critic's probe): `x0 = uz + ua` with `uz` a rank-1 zero tensor and `ua` a rank-1
  tensor, both packed with `n = r = 2`, has masks `[1 0 1 0]`; old ranks 1 gather slot 0, the zero block — `‖X_old‖ = 0`
  against `‖x0‖ = 3.23`.
  - So with `old_ranks` given, `padded_structures` requires `x_cores`' rank masks to be **prefix masks**, per
    stack element and per edge (`mask == prefix_mask(mask.sum(-1), size)`, host numpy). This implies that the old-rank
    prefix lies inside the real slots. Inclusion alone would not be enough: the probe's old prefix `[1 0 0 0]` lies
    inside `[1 0 1 0]`.
  - Otherwise it raises a structural `ValueError` in both modes, under `safety.unsafe()` too, naming the remedy:
    pack the padded point in prefix layout, as `UniformTuckerTensorTrain.from_t3` of a ragged `resize` does.
  - The realistic path never meets it: there is no uniform `resize`, and `from_t3` produces prefix masks.
- **Ragged layer: the layout is a numerical precondition.** It has no masks, so the layout cannot be read from
  structure. A ragged `z + a` is indistinguishable from a `resize`, so the layout is numerical there and left to the
  zero-padding check.
- **Zeros outside the old block are numerical on both layers.** That `x0` is exactly zero outside the old block — on
  the uniform layer, in the real slots past the old prefix, up to `x0`'s ranks — is a
  numerical precondition, checked by the frontend in safe mode (§3.6). The error message says what is meant: "x0 is
  not a zero-padding, in the prefix layout `resize` produces, of a point at padded_from's ranks".

Why members and not a T3-specific loop: the loop (§3.5) is generic — residual once, gradient per pass, one
constructor call per pass — and only these three facts are layer- and representation-specific. Keeping the loop in
`backend/optimizers.py` keeps that module T3-free (CLAUDE.md, the 2026.2.0 restructure), and every line of math the
members bind is a standalone function.

### 3.5 The driver `padded_start` (backend, T3-free)

```
padded_start(problem, x_cores, old_ranks=None, options=None, final_gradient=True) -> PaddedStart
```

- `old_ranks` given: `x_cores` is `X_old` zero-padded to its ranks; the residual is taken at `x_cores` (the point
  never moves) and the returned frame is a padded frame **of `x_cores`'s tensor**.
- `old_ranks=None`: the **zero tensor of `x_cores`' ranks** (`s = 0`); the residual is taken at
  `corewise_zeros_like(x_cores)` (`_zero_reference_gnorm`'s formula) and the returned frame is the **cone frame at
  the zero tensor** — its base point is 0 whatever `x_cores` holds, so it is a valid `start_frame` only for a zero
  `x0`.
- The pass count, oversampling and seed come from `options` (`PaddedFrameOptions`, §5.5); `None` fields resolve to
  the module constants (`PADDED_START_PASSES = 4` here; the cone resolution passes `CONE_REFERENCE_PASSES = 10`;
  `PADDED_FRAME_OVERSAMPLE = 5`); the seed's `None` is resolved by the geometry (`fv_operations.ZERO_FRAME_SEED`).
- `passes ≥ 1` gradients are evaluated: passes `1 … passes − 1` rebuild the frame at the oversampled structure, the
  last one at the target structure (§3.1). Every projection passes `aux=geom.precompute(F)` (as
  `_zero_reference_gnorm` does), so a shared geometry builds its companion once per pass.
- The misfit gradient only: the regularizer is not consulted (its gradient `λ v_X` lies in `T_X M_s` and is removed
  by the two-sided steps), as in the zero reference.
- Returns `PaddedStart(frame, gnorms, gnorm)`: the frame at `x_cores`' structure; the captured `‖P_T ∇f‖` per pass
  (at that pass's structure — the oversampled values are not comparable to the target one); and the captured norm
  at the returned frame when `final_gradient=True` (one more sweep and transpose; `newton_cg` passes `False` because
  iteration 0 computes it anyway). With `old_ranks=None` that `gnorm` **is** the cone reference; with `old_ranks`
  given it is the deflated warm-start gradient at the padded frame and must not be used as a Newton reference.
- Stacked `x_cores` (`C ≠ ()`) are supported: `gnorms` and `gnorm` are then per element (`geom.inner` is
  per-element). The optimizers stay unstacked (§3.9).
- Raises (structural) on a geometry without the members: "padded starts need a manifold geometry".

### 3.6 The optimizer seam

**`Problem.local_model(..., frame=None)`.** When `frame` is given it replaces `geom.frame(x_cores)` for the sweep,
the geometry aux, the gradient, the curvature, the retraction and the regularizer's base point. Documented
preconditions: `frame` is **orthonormal** (ORTH — `numerical_contracts.md` §`fitting.LocalModel` calls it a
construction invariant of `geometry.frame(x)`; a caller-supplied frame makes it a precondition) and represents the
**same tensor** as `x_cores` (the residual is computed at `x_cores`, everything frame-dependent at `frame`).
Structural check always: `geom.require_frame_structure(x_cores, frame)` (a geometry without the member raises a
structural `ValueError` naming it, as the exact-mode members do). The numerical properties are not checked in the
backend (house rule); the frontend checks them.

**`newton_cg` (backend):** four keywords.

- `start_frame` — iteration 0's frame (an orthonormal frame of `x0`'s tensor at `x0`'s structure).
- `padded_from` — `X_old`'s `(tucker_ranks, tt_ranks)`: iteration 0's frame is
  `padded_start(problem, x0, padded_from, start_options, final_gradient=False).frame`.
- `start_options` — the `PaddedFrameOptions` of `padded_from` (structural error without `padded_from`).
- `cone_options` — the `PaddedFrameOptions` of `g0_reference='cone'` (structural error with any other reference).
- `start_frame` and `padded_from` together: structural error. Neither: the run is today's run, `stats` included,
  bit for bit (decision 2; T7(a), G-bit).
- **Record** (only when used — default runs' `stats` keep exactly today's keys, `tests/backend/test_optimizers.py:477`,
  `:860`): `stats['start_frame_source'] = 'given' | 'padded'`, and with `padded_from` also
  `stats['padded_start_gnorms']` (the per-pass captured norms). History rows are unchanged (no new `NewtonInfo`
  field).
- Under `use_jit`, `_prepare_jit_inputs` converts `x0` / `sample` / `data` first; a given `start_frame` is converted by
  `_prepare_jit_frame(use_jit, frame)` = `common.partition_static` → `tree_to_jax` on the dynamic leaves →
  `rebuild_static` (the house split of a raw frame tuple: a uniform frame's masks and `shape` stay host); a
  `padded_from` frame is built eagerly from the converted inputs and is therefore jax already. The start frame adds
  no compile: it enters the CG as the local model's leaf, and a uniform frame's masks — value-hashed static aux —
  equal the template's.
- The start frame influences **iteration 0 only**: the local model is rebuilt from the retracted point after the
  first accepted step (unchanged code); a rejected trust-region trial keeps the model, hence the start frame.
- The reference templates (`'zero'`, `'cone'`) read iteration 0's frame for its structure only, so with a start
  frame the default `'zero'` value is today's bit for bit.

**Frontend `newton_cg`:** the same four keywords; `start_frame` is a `T3Frame` / `UT3Frame` (its `.data` is passed);
`start_options` / `cone_options` are `PaddedFrameOptions`. Frontend-only checks:

- **structural (both modes):** `padded_from` or `start_frame` with an `x0` that is not structurally minimal is an
  error naming `TuckerTensorTrain.get_minimal_ranks` (the frontend would otherwise reduce `x0` — `_ragged_minimal` /
  `uniform_minimal` — destroying the padding layout, or making a given frame's structure stale and failing late in
  `require_frame_structure`); a `start_frame` of the wrong representation (ragged for a uniform `x0`) is an error;
  any of the new keywords on a corewise geometry is an error.
- **numerical, safe mode only** (`safety.checks_active` on the operands; skipped under `safety.unsafe()` and under a
  trace):
  - with `padded_from`: the part of `x0` outside the old ranks is exactly zero — on the uniform layer only `x0`'s
    **real** slots are read (padding beyond `x0`'s own masks is legal garbage; with the prefix masks §3.4 requires,
    the real slots outside the old prefix are exactly the padded ones) — `ValueError`: "x0 is not a zero-padding, in
    the prefix layout `resize` produces, of a point at padded_from's ranks";
  - with `start_frame`: ORTH (`start_frame.is_orthogonal()` at the safety tolerance — the cached
    `orthogonality_residual`) and the same tensor (`start_frame.to_t3()` / `.to_ut3()` equals `x0` at the safety
    tolerance, the analog of `fitting._require_at_frame`'s numerical leg, `:108-126`);
  - on a shared geometry: TIED for `x0` (`x0.has_shared_tucker_factors(sharing)`) with `padded_from`, and for a
    `start_frame`'s base point.

### 3.7 `g0_reference='cone'`

`_G0_REFERENCES` gains `'cone'`; `_resolve_g0_reference` resolves it pre-loop, on the level's own problem:

1. the geometry must have `padded_structures` — else a structural `ValueError` (corewise and custom geometries;
   the message names the member);
2. computed iff `g0norm_newton is None` (the `'zero'` rule, m32 of the curvature plan);
3. `_cone_reference_gnorm(problem, x0, cone_options)` = `padded_start(problem, x0, None, options with passes
   defaulting to CONE_REFERENCE_PASSES, final_gradient=True).gnorm` — §2.7 says what this number is;
4. vanishing (`== 0`, all-zero data): `ValueError` — `'cone'` is always explicit, so the `'zero'` default's
   fallback-with-warning does not apply;
5. recorded as `g0_reference='cone'` (and `cg_reference='cone'` unless `g0norm_cg` overrides) — the existing
   per-row fields, whose values only change on runs that ask for `'cone'`; the display reads `g₀ = tangent cone`.

The default (`g0_reference=None`) is unchanged — `'zero'` on the manifold geometries (decision 2). **No tolerance
recommendation ships with it yet.** The cone value is larger than the Haar `'zero'` value (at least 15–76× on the Darcy
surrogate, §0), so a relative stop against it is looser at the same `gtol_rel`; downstream, `gtol_rel=3e-4` against
the cone behaved like T3Polynomial's **own** coordinate-aligned zero reference at `1e-3` (not the library's
pre-2026.3.0 warm-start reference). S11 measures it against the library's own references before the docs recommend a
value. `gradient_descent` resolves through the same function and gains `cone_options`.

### 3.8 Structural vs numerical — where each decision lives

| decision | kind | where | both modes / safe only |
|---|---|---|---|
| how many new directions per slot | structural (rank differences) | `fv_padded_frame` | — (no decision is made) |
| which directions (hint) | numerical content (the hint's orientation) | `pinned_leading_columns` | — (deterministic computation, no check) |
| which directions (hint-free) | convention (seeded, keyed) | `_tucker_completion`, `_outside_first` | — |
| old stores ≤ template stores; stacks (`C`, `K + C`); hint-frame stores ≥ old; groups; `d` | structural | `fv_padded_frame` / `ufv_padded_frame` | both |
| `old_ranks ≤ x0`'s ranks and structurally minimal; their lengths | structural | `padded_structures` | both |
| a uniform `x0` with `old_ranks` has prefix rank masks (the layout the old-rank slice reads) | structural | `padded_structures` | both |
| a ragged `x0` holds `X_old` in the prefix layout (no masks to read) | numerical precondition (the zero-padding check covers it) | frontend | safe only |
| an explicit frame at `x`'s structure | structural | `geom.require_frame_structure` via `Problem.local_model` | both |
| `start_frame` with `padded_from`; `start_options` without `padded_from`; `cone_options` without `'cone'`; invalid `PaddedFrameOptions`; a geometry without the members | structural | `newton_cg`, `PaddedFrameOptions.__post_init__`, `padded_start` | both |
| `padded_from` / `start_frame` with a non-minimal frontend `x0`; a frame of the wrong representation | structural | frontend `newton_cg` / `padded_start` | both |
| `x0`'s padding exactly zero (real slots only) | numerical precondition | frontend | safe only |
| a start frame is orthonormal (ORTH) and represents `x0` | numerical precondition | frontend | safe only |
| tied old factors (TIED) on a shared geometry | numerical precondition | frontend | safe only |
| the hint frame contains the old frame's subspaces | caveat (decides directions, never validity) | documented on `fv_padded_frame` | — |
| the cone reference vanishes | runtime (explicit request impossible) | `_resolve_g0_reference` | both |

### 3.9 Stacking — what is committed and why

- **`fv_padded_frame`: the frame stack `C`, vectorized.** A ragged stack has one rank structure, so every count and
  index set is shared and every operation is a `C`-grouped `contract`, a batched `xnp.linalg.svd` / `qr`, or an
  elementwise op; the Haar seeds broadcast over `C`. At `s = 0` the hint-free result is `fv_zero_point_frame`'s
  frame broadcast over `C` (`fv_zero_point_frame` itself keeps refusing stacks — unchanged).
- **A `K`-stacked hint is lifted, not reduced**: hint variations with stack `K + C` at a `C` hint frame give a
  `K + C`-stacked output, one padded frame per hint (frame-inner: `C` stays innermost). The template, the old frame
  and the hint frame are broadcast to `K + C` (`xnp.broadcast_to`) and the `C`-vectorized code runs with
  `C := K + C` — the library's lift of an atom over a stack (`docs/batching_and_stacking.md` §11). A reduction of
  the `K` hints to one (their sum, their joint range) would be a policy choice and is left to the caller.
- **`ufv_padded_frame`: `C` (or `K + C`) by a host loop over the elements**, varying ranks included (the variety):
  the unpack produces per-element ragged blocks of different shapes, which no single vectorized call can take.
  `O(|C|)` eager iterations — acceptable for a once-per-level construction; vectorizing the uniform-rank special
  case is deferred (O-5).
- **`padded_start` (backend and frontend): `C`**, following the constructor and the kinds (`Problem.local_model`
  already supports a stacked point): per-element `gnorms` / `gnorm` (arrays of shape `C`; floats unstacked);
  `old_ranks` as per-element `(d,)+C` / `(d+1,)+C` int arrays on a varying-rank uniform stack.
- **The optimizers stay unstacked** (`_require_unstacked`, unchanged); `newton_cg(padded_from=...)` therefore runs
  `padded_start` unstacked.

---

## 4. Algorithms, step by step

### 4.0 Notation and conventions for every step

- Every array carries the frame stack `C` as **leading** axes (`C = ()` for one frame). The old frame `(U°, O°, L°, R°)`
  has stores `(s, sD, sL, sR)`; the template `(nU, nD, rL, rR)`. The **counts** are `kU_i = nU_i − s_i`,
  `kD_i = nD_i − sD_i`, `kL_k = rL_k − sL_k`, `kR_k = rR_k − sR_k` (Python ints, ≥ 0 by §3.2's check).
- `_embed(A, shape)` = `linalg.pad_or_truncate(A, [(0, 0)]*|C| + [(0, t − a) …])`: the old block in the leading
  slots, zeros elsewhere. A pad, not a contraction.
- `_unfold(A, m)` = `A.reshape(C + (∏ first m non-stack axes, ∏ the rest))`. Used **only** to form an SVD/QR input
  and to fold a factor back; it merges adjacent **non-stack** axes and keeps `C` leading (the sharding rule:
  no stack axis is ever merged or moved). Every other contraction is a `contract(...)` call with the `C`
  group; a one-line `swapaxes` or a plain `concatenate` is used where that is all that is needed.
- New `contract` subscripts are listed with their HISTORICAL decision in §4.10; letters are chosen so the
  sizes in `AXIS_SIZE` are distinct within each string (a transposed axis is then a shape error in the oracle
  tests). Two existing strings are reused verbatim.
- Static branches only: every `if` is on a count or a shape (Python ints), never on array values — so every
  step is `jax.jit`-traceable and vectorizes over `C`.

### 4.1 The primitives (`backend/linalg.py`)

`pinned_completion(pinned, seeds, k)` — `pinned` `C+(n, s)` orthonormal columns, `seeds` `C+(n, m)`:

1. `k == 0` → return `zeros C+(n, 0)` (no factorization). `s + k > n` → structural `ValueError`.
2. `S = seeds[..., :k]` if `m ≥ k`, else `concatenate([seeds, zeros C+(n, k − m)], -1)` (missing seeds are zero
   columns; Householder completes them deterministically — probed: orthonormal and `⊥ pinned` to 2e-16 numpy,
   1e-7 jax float32).
3. `Q, R = xnp.linalg.qr(concatenate([pinned, S], -1))` — reduced Householder QR, batched over `C`.
4. `sg = xnp.where(xnp.diagonal(R, axis1=-2, axis2=-1)[..., s:s + k] >= 0, 1.0, −1.0).astype(Q.dtype)` — the
   `_haar_rows` sign rule, as float literals cast to `Q`'s dtype (an int `where` would promote a float32 `Q`).
5. return `Q[..., :, s:s + k] * sg[..., None, :]`.

Output: orthonormal columns orthogonal to `span(pinned)` for **every** input (the Householder `Q` is orthonormal and
its first `s` columns span `pinned`'s columns), whatever the seeds' rank — no threshold. With `s = 0` and Gaussian
seeds it is `_haar_rows`' recipe exactly (transposed).

`pinned_leading_columns(pinned, A, k)` — the leading `k` left singular directions of `(I − PPᵀ)A`, completed:

1. `k == 0` → `zeros C+(n, 0)`; `A` with zero columns → `pinned_completion(pinned, A, k)` (all seeds zero).
2. `PtA = contract('Cia,Cib->Cab', pinned, A)`; `Aperp = A − contract('Cia,Cab->Cib', pinned, PtA)`.
3. `Uq, _, _ = xnp.linalg.svd(Aperp, full_matrices=False)` (`C+(n, min(n, m))`, descending).
4. return `pinned_completion(pinned, Uq, k)`.

When `Aperp` has `≥ k` nonzero singular values the result spans its leading-`k` left singular subspace (the QR of
an orthonormal `[P | U_k]` returns `U_k` up to column signs); otherwise the trailing columns are the Householder
completion — the hint has no information there, and nothing is thresholded. The other LAPACK-dependent case is a
tie at the cut (`σ_k = σ_{k+1}`), where any basis of the tied subspace is a leading set. Neither primitive sees
padding: the uniform layer unpacks first (§4.8), so `pad_safe_svd` is not involved anywhere in this feature.

### 4.2 Set-up: validation, the empty old frame

1. Read the template's stores and dtype; `xnp` from `tree_contains_jax((frame, old_frame, hint))`.
2. `old_frame is None` → `_empty_old_frame(frame)`: `U°_i` `C+(0, N_i)`; `O°_i`, `L°_i`, `R°_i` with zero-size middle
   axes and bond stores `sL = sR = (1, 0, …, 0, 1)` — the stores of the zero tensor. No separate `s = 0` path.
3. The structural checks of §3.2 (each a `ValueError` naming store and mode).
4. A `K`-stacked hint (`hint_variations`' stack is `K + C` with `K ≠ ()`, recovered by comparing its leading axes
   with `hint_frame`'s stack): broadcast the template's shapes, the old frame and the hint frame to `K + C`
   (`xnp.broadcast_to`) and continue with `C := K + C` (§3.9). The result has stack `K + C`.
5. Group keys `k_g = min(g)` (the `fv_zero_point_frame` rule), the per-group members, and the group's old block
   `U°_g := U°_{k_g}` (TIED is the caller's precondition, §3.2).

### 4.3 The hint-free completion

**Tucker, per group `g`** (one factor for all members):

- `s_g == 0`: `U_g = xnp.asarray(_haar_rows(nU_g, N_g, k_g, seed), dtype=dtype)`, broadcast over `C` —
  `fv_zero_point_frame`'s own call, so the `s = 0` result is bitwise its result.
- `s_g > 0`: `G = RandomState([seed, k_g]).standard_normal((kU_g, N_g))` (host; the same keyed stream, its first
  `kU_g` rows, so the completion is nested in `kU_g`); `new = pinned_completion(swapaxes(U°_g), broadcast(Gᵀ),
  kU_g)`; `U_g = concatenate([U°_g, swapaxes(new)], -2)`. Haar-distributed in `U°_g`'s complement.

**TT and down cores** — `_outside_first`, per core, in the core's own unfolding order (`L`: `(a, n)` rows a-major,
new **columns**; `R`: `(n, b)` columns n-major, new **rows**; `O`: `(a, b)` a-major, new **down rows**):

1. The coordinate directions of the unfolding **outside** the old block (`{a < sL_k, n < s_k}` for `L_k`, and
   likewise), in index order; the first `t = min(count, #outside)` become unit vectors — host constants (an index
   set), exactly orthonormal and orthogonal to the embedded old block (disjoint support).
2. If `t < count` (a bond grows while its neighbours do not — e.g. a TT-only increase at the first core): the
   remaining `count − t` come from **inside** the old block: `pinned_completion(_unfold(old core, 2), G_in,
   count − t)` in the old block's own coordinates, with `G_in = RandomState([seed, i, c]).standard_normal(…)`
   (`c = 1, 2, 3` for left / right / down; a three-word key, disjoint from the Tucker streams' two-word keys), then
   `_embed` into the core's block. Seeded Gaussian rather than coordinate seeds so the completion never meets a
   seed lying in the old block's span (a coordinate-aligned old core would make a coordinate seed degenerate).
3. Assemble: `[embedded old | outside units | embedded in-block]` along the new axis, then fold back.
4. The carriers: `L_{d−1} = _embed(L°_{d−1})`, `R_0 = _embed(R°_0)`.

At `s = 0` every coordinate is outside and step 1 produces exactly `_identity_prefix`'s arrays (ones and zeros; P2),
and the carriers are zero.

### 4.4 The hint in its own coordinates

(`padded_frames` imports `tv_operations` at module level; §3.2.)

```
Z          = tv_operations.tv_to_t3(hint_frame, hint_variations)            # the doubled-rank T3 of the hint (stack C)
W, Ztt     = t3_orthogonalization.t3_down_orthogonalize_tucker_cores(Z)     # W_i  C+(w_i, N_i) orthonormal rows; Ztt_i C+(z_i, w_i, z_{i+1})
ZL         = tt_orthogonalization.tt_left_orthogonalize(Ztt)               # C+(zl_i, w_i, zl_{i+1}), left-orthonormal (but the last)
ZR         = tt_orthogonalization.tt_right_orthogonalize(Ztt)              # C+(zr_i, w_i, zr_{i+1}), right-orthonormal (but the first)
A_i        = contract('Czn,Cun->Czu', U°_i, W_i)                           # C+(s_i, w_i)       existing string
Lh_i       = contract('Cazb,Czu->Caub', L°_i, A_i)                         # C+(sL_i, w_i, sL_{i+1})   existing string
Rh_i       = contract('Cazb,Czu->Caub', R°_i, A_i)                         # C+(sR_i, w_i, sR_{i+1})
ΩL         = tt_operations.tt_zipper_left_to_right(Lh, ZL)                 # ΩL_k C+(sL_k, zl_k)  X_old's left interface in ZL's coordinates
ΩR         = tt_operations.tt_zipper_right_to_left(Rh, ZR)                 # ΩR_k C+(sR_k, zr_k)
ZZ         = tt_operations.tt_zipper_right_to_left(ZL, ZR)                 # ZZ_k C+(zl_k, zr_k)  the gauge between the two chains
```

When the hint frame contains the old frame's subspaces (§2.3), `Z`'s chains contain `X_old`'s interfaces: `A_i`, `ΩL_k`, `ΩR_k` have
orthonormal rows (P3; also for a zero hint, whose doubled-rank cores still carry the frame blocks). All reused
functions are stack-aware.

### 4.5 The Tucker step (per group; HOSVD-style, two-sided)

Per mode `i` (every member of every group):

```
Ht_i = contract('Cxjb,Cbz->Cjxz', ZL[i], ZZ[i+1])            # C+(w_i, zl_i, zr_{i+1}): the hint's mixed-canonical center, Tucker leg first
E_i  = contract('Caib,Cax,Cbz->Cixz', O°_i, ΩL[i], ΩR[i+1])   # C+(sD_i, zl_i, zr_{i+1}): X_old's down rows there (orthonormal rows)
T_i  = contract('Cjxz,Cixz->Cji', Ht_i, E_i)
N_i  = Ht_i − contract('Cji,Cixz->Cjxz', T_i, E_i)            # column side: old-coupled content removed
M_i  = N_i.reshape(C + (w_i, zl_i · zr_{i+1}))               # SVD input (two trailing rank axes merged; C leading)
Uq, sq, _ = xnp.linalg.svd(M_i, full_matrices=False)          # compression: exact for the Gram, M_i M_iᵀ = (Uq sq)(Uq sq)ᵀ
Y_i  = contract('Cjn,Cjc->Cnc', W_i, Uq * sq[..., None, :])   # C+(N_i, c_i): the normal content, ambient
```

Per group `g = (i_1, …, i_p)`:

```
Y_g   = concatenate([Y_{i_1}, …, Y_{i_p}], -1)                # the group's concatenated content (docs/sharing.md, face 1)
new_g = pinned_leading_columns(swapaxes(U°_g), Y_g, kU_g)     # row side: pinned off U°_g
U_g   = concatenate([U°_g, swapaxes(new_g)], -2)               # old rows bitwise, then the new rows; replicated over members
B_i   = contract('Czn,Cun->Czu', U_i, W_i)                     # C+(nU_i, w_i): the new Tucker basis in the hint's coordinates
```

Ambient coordinates (rather than the prototype's `w`-coordinates) make the step independent of `w_i ≥ nU_i` and make
sharing the same code; for a singleton group it equals the prototype's step in exact arithmetic. `B_i`'s first
`s_i` rows are `A_i`. (Why the full environment and not the hint's Tucker variation `V_i`: P7.)

### 4.6 The pinned left sweep, and the mirrored right sweep

`_pinned_left_sweep(chain, Ω, B, old_cores, stores)` — `chain` right-orthonormal (but its first core), `Ω` the old
far-side interfaces in `chain`'s coordinates:

```
Cv = xnp.ones(C + (1, 1))                                               # the hint on L_{<k} ⊗ chain_{≥k}
for k in 0 … d−2:
    Mb  = contract('Cxa,Caib,Cni->Cxnb', Cv, chain[k], B[k])            # C+(rL_k, nU_k, zr_{k+1})
    Mp  = Mb − contract('Cxnb,Czb,Czc->Cxnc', Mb, Ω[k+1], Ω[k+1])       # column side: X_old's right interface at bond k+1 removed
    P   = _unfold(_embed(old_cores[k], (rL_k, nU_k, sL_{k+1})), 2)      # C+(rL_k·nU_k, sL_{k+1}): the pins
    new = pinned_leading_columns(P, _unfold(Mp, 2), kL_{k+1})            # row side: pinned off the old columns
    L_k = concatenate([P, new], -1).reshape(C + (rL_k, nU_k, rL_{k+1}))
    Cv  = contract('Cxnc,Cxnb->Ccb', L_k, Mb)                            # C+(rL_{k+1}, zr_{k+1})
L_{d−1} = _embed(old_cores[d−1], (rL_{d−1}, nU_{d−1}, 1))                # the carrier: the old content only
```

- **`L`** = `_pinned_left_sweep(ZR, ΩR, B, L°, (rL, nU, sL))`. Because `ZR`'s tail is right-orthonormal, the SVD of
  the `(rL_k·nU_k) × zr_{k+1}` matrix is the SVD of the full unfolding (the TT-SVD argument).
- **`R`** = `tt_reverse(_pinned_left_sweep(tt_reverse(ZL), ΩL[::-1], B[::-1], tt_reverse(R°), reversed stores))`:
  the reversed left-orthonormal chain is right-orthonormal, `ΩL[::-1][k'+1]` is the left interface at the mirrored
  bond, and the reversed sweep's carrier is `R_0 = _embed(R°_0)`. One code path; `tt_reverse` is exact and
  stack-aware (`tt_operations.py:31`).
- The carry uses `Mb`, not `Mp`: the "rotation corrections" that would carry `Mp` are inert (P10) — the next step's
  column projection removes the same content again.

### 4.7 The down completion

```
G2_i = contract('Cazb,Cuz->Caub', Ztt[i], B[i])                         # C+(z_i, nU_i, z_{i+1}): the hint TT in the new Tucker basis
ΛL   = tt_operations.tt_zipper_left_to_right(G2, L)                      # ΛL_i C+(z_i, rL_i)
ΛR   = tt_operations.tt_zipper_right_to_left(G2, R)                      # ΛR_i C+(z_i, rR_i)
for i in 0 … d−1:
    env = contract('Cxa,Cxzc,Ccb->Cabz', ΛL[i], G2_i, ΛR[i+1])          # C+(rL_i, rR_{i+1}, nU_i): the hint's center in the final bases
    Anew = env[..., s_i:].reshape(C + (rL_i · rR_{i+1}, kU_i))           # the NEW Tucker slots only
    P   = _unfold(swapaxes(_embed(O°_i, (rL_i, sD_i, rR_{i+1})), -2, -1), 2)   # C+(rL_i·rR_{i+1}, sD_i)
    new = pinned_leading_columns(P, Anew, kD_i)
    O_i = swapaxes(concatenate([P, new], -1).reshape(C + (rL_i, rR_{i+1}, nD_i)), -2, -1)
```

The restriction to the new Tucker slots is the down step's two-sided cut (§2.3): `X_old`'s center lives on the old
slots. `kD_i > kU_i` (old down rank limited by its bonds) is handled by the completion.

### 4.8 The uniform twin

`ufv_padded_frame` (host, eager; `require_concrete_masks` first):

1. Per element of the flattened stack (`ufv_unstack`; for a `K`-stacked hint the frames' elements are repeated over
   `K`): unpack the template, the old frame and the hint frame (`ufv_conversions.ut3frame_to_t3frame`) and the hint
   variations (`ut3variations_to_t3variations`) — real blocks only, so padding garbage is never read and each element
   may have its own ranks.
2. `fv_padded_frame(...)` on the element (the same groups and seed).
3. Pack at the **template's** padded sizes (`t3frame_to_ut3frame(..., N=, nU=, nD=, rL=, rR=)`); restack
   (`ufv_stack`).
4. The packed masks must equal the template's (prefix masks; a structural `ValueError` otherwise, the
   `ufv_zero_point_frame` check); return `packed[:5] + (template_masks,)`.

On identical inputs the real blocks are the ragged construction's **bit for bit**. Across the layers the inputs are
made identical by building the uniform old frame through the ragged path (§3.3, §3.4); the hints then differ at
rounding only.

### 4.9 Oversampling, truncation, and the pass loop

Exactly §3.5's loop: the residual once; the hint-free frame at the oversampled structure; `passes − 1` hinted
rebuilds at the oversampled structure; the last rebuild at the target structure (the truncation: the template is
the target, the hint the oversampled frame and its gradient). The oversampled gradient is projected and normed
with the **oversampled** geometry (the uniform geometry holds its variation masks); the final gradient with the
target one; every projection receives `aux=geom.precompute(F)`. `oversample = 0` makes the two structures equal
(the maximum rule of §3.4). With `passes = 1` the result is the one-pass enrichment from the hint-free frame (the
downstream W3 analogue, δ-free).

### 4.10 The new contraction strings (HISTORICAL entries)

All have the single group `C` and are fully determined from the operand ndims, so each HISTORICAL entry is `''`
(decided by hand: every operand's lowercase count is fixed, so `|C| = ndim − #lowercase` is solved from any
operand). `test_contractions_sharding.py` picks them up automatically (AST scan).

| # | string | where | letters (AXIS_SIZE) |
|---|---|---|---|
| 1 | `'Cia,Cib->Cab'` | `pinned_leading_columns`: `PᵀA` | i3 a2 b4 |
| 2 | `'Cia,Cab->Cib'` | `pinned_leading_columns`: `P(PᵀA)` | i3 a2 b4 |
| 3 | `'Cxjb,Cbz->Cjxz'` | Tucker: the hint's center, Tucker leg first | x7 j2 b4 z5 |
| 4 | `'Caib,Cax,Cbz->Cixz'` | Tucker: `X_old`'s down rows in the hint environment | a2 i3 b4 x7 z5 |
| 5 | `'Cjxz,Cixz->Cji'` | Tucker: overlap with them | j2 x7 z5 i3 |
| 6 | `'Cji,Cixz->Cjxz'` | Tucker: removal | j2 i3 x7 z5 |
| 7 | `'Cjn,Cjc->Cnc'` | Tucker: compressed content to ambient | j2 n6 c3 |
| 8 | `'Cxa,Caib,Cni->Cxnb'` | sweep: carry × hint core × Tucker basis | x7 a2 i3 b4 n6 |
| 9 | `'Cxnb,Czb,Czc->Cxnc'` | sweep: projector onto the old far interface | x7 n6 b4 z5 c3 |
| 10 | `'Cxnc,Cxnb->Ccb'` | sweep: carry update | x7 n6 c3 b4 |
| 11 | `'Cazb,Cuz->Caub'` | down: hint TT in the new Tucker basis | a2 z5 b4 u3 |
| 12 | `'Cxa,Cxzc,Ccb->Cabz'` | down: the environment, unfolding order | x7 a2 z5 c3 b4 |

Reused verbatim (already frozen): `'Czn,Cun->Czu'` (`A_i`, `B_i`), `'Cazb,Czu->Caub'` (`Lh`, `Rh`). No `d`-prefixed
(uniform supercore) strings: the uniform twin unpacks.

---

## 5. API

Every signature in house style; docstrings follow the module's existing ones (an Examples section with a doctest
for every public function, as listed). `C` is the frame stack.

### 5.1 `backend/linalg.py` (both in `__all__`)

```python
def pinned_completion(
        pinned:  NDArray,   # C+(n, s): orthonormal columns, held fixed (s may be 0)
        seeds:   NDArray,   # C+(n, m): directions to complete from, in order; m < k pads with zero columns
        k:       int,       # completion columns; s + k <= n (structural)
) -> NDArray:               # C+(n, k): orthonormal columns orthogonal to span(pinned); sign-fixed
    """The ``k`` columns that continue ``pinned`` to an orthonormal set, taken from ``seeds`` in order: the
    sign-fixed Householder QR of ``[pinned | seeds[:k]]``, columns ``s..s+k-1``. Valid for any seeds (a seed in
    the span of the earlier columns is completed deterministically); no threshold. ``s = 0`` with Gaussian
    seeds is the Haar recipe of ``fv_operations._haar_rows``."""

def pinned_leading_columns(
        pinned:  NDArray,   # C+(n, s): orthonormal columns, held fixed
        A:       NDArray,   # C+(n, m): the content whose leading directions are wanted
        k:       int,       # s + k <= n (structural)
) -> NDArray:               # C+(n, k): the leading k left singular directions of (I - P P^T) A, completed against P
    """The pinned truncated SVD: the leading ``k`` left singular directions of ``A`` projected off
    ``span(pinned)``, completed by :py:func:`pinned_completion` when the projection has rank ``< k``. The
    count ``k`` is the caller's (structural); nothing is compared with a tolerance."""
```

Doctests: orthonormality and `⊥ pinned` (`< 1e-14`), including all-zero seeds; a planted leading direction is
recovered (`|⟨u, e⟩| > 1 − 1e-12`).

### 5.2 `backend/padded_frames.py`: `fv_padded_frame` (in the module's `__all__`)

```python
def fv_padded_frame(
        frame:      typ.Tuple[
            typ.Sequence[NDArray],  # up_tucker_cores. len=d, elm_shape=C+(nUi, Ni)
            typ.Sequence[NDArray],  # down_tt_cores.   len=d, elm_shape=C+(rLi, nDi, rR(i+1))
            typ.Sequence[NDArray],  # left_tt_cores.   len=d, elm_shape=C+(rLi, nUi, rL(i+1))
            typ.Sequence[NDArray],  # right_tt_cores.  len=d, elm_shape=C+(rRi, nUi, rR(i+1))
        ],                                                    # the TEMPLATE: only its shapes and dtype are read
        old_frame:  typ.Optional[typ.Tuple] = None,           # (U, O, L, R) of X_old, stack C, stores <= the template's; None = the zero tensor
        hint:       typ.Optional[typ.Tuple[typ.Tuple, typ.Tuple]] = None,  # (hint_frame, hint_variations): stack C frame, K+C variations; None = hint-free
        groups:     typ.Tuple[typ.Tuple[int, ...], ...] = (),  # sharing partition (canonical_groups); () = unshared
        seed:       int = ZERO_FRAME_SEED,                     # the hint-free draws' seed: a convention; tunable through PaddedFrameOptions(seed=)
) -> typ.Tuple[
    typ.Tuple[NDArray, ...],  # up:    U°_i in the leading rows, then the completion.  len=d, elm_shape=K+C+(nUi, Ni)
    typ.Tuple[NDArray, ...],  # down:  O°_i embedded, then the completion.             len=d, elm_shape=K+C+(rLi, nDi, rR(i+1))
    typ.Tuple[NDArray, ...],  # left:  L°_k embedded, then the completion; L[d-1] = L°[d-1] embedded (the carrier)
    typ.Tuple[NDArray, ...],  # right: R°_k embedded, then the completion; R[0] = R°[0] embedded (the carrier)
]:
```

Docstring sections: what a padded frame is (§2.1); the two completions (§2.3, §2.5) with the counts; "structural,
δ-free: no threshold, no content in the point"; the numerical preconditions (ORTH for the old frame; TIED with
`groups`, the old block read from each group's first mode) and the caveat on the hint frame (§2.3), all unchecked
here; stacking (`C` vectorized; a `K`-stacked hint lifted to `K + C` outputs); backend inferred; jit-traceable;
`s = 0` hint-free = `fv_zero_point_frame`. Doctests: (a) the `s = 0` identity (`np.array_equal` on all `4d` cores);
(b) a `resize`d point: orthonormality residual `< 1e-14`, `(U, L)` reconstructs `X_old` (`np.allclose`), the old
block embedded bitwise (`np.array_equal(U[i][:s_i], U°_i)`); (c) a hinted call from the library gradient: still (b).
`fv_zero_point_frame` is **not** changed (its docstring gains one "See also" line).

### 5.3 `backend/padded_frames.py`: `ufv_padded_frame` (in the module's `__all__`)

```python
def ufv_padded_frame(
        frame_data:      typ.Tuple,                                  # uniform frame .data (up, down, left, right, shape, masks): the TEMPLATE
        old_frame_data:  typ.Optional[typ.Tuple] = None,             # uniform frame .data of X_old (same stack C), or None = the zero tensor
        hint:            typ.Optional[typ.Tuple[typ.Tuple, typ.Tuple]] = None,  # (hint_frame_data, hint_variations_data), uniform .data; stack C frame, K+C variations
        groups:          typ.Tuple[typ.Tuple[int, ...], ...] = (),   # sharing partition; () = unshared
        seed:            int = fv_operations.ZERO_FRAME_SEED,        # as fv_padded_frame
) -> typ.Tuple:                                                      # uniform frame .data: the template's masks (broadcast over K), padding exactly zero
```

Host-side and eager (it reads masks). Doctest: the real blocks equal `fv_padded_frame`'s on the unpacked inputs
(`np.array_equal`), and `out[5] is frame_data[5]` (unstacked hint).

### 5.4 Geometry members (`backend/geometry.py`)

On `ManifoldGeometryOps` and `UniformManifoldGeometryOps` only (the corewise geometries do not define them, §3.4):

```python
    def padded_frame(
            self,
            frame:         typ.Tuple,                  # the template: a frame at the structure to build (shapes read)
            old_frame:     typ.Optional[typ.Tuple],    # a frame of X_old (from padded_structures), or None = the zero tensor
            hint_frame:    typ.Optional[typ.Tuple],    # the hint's frame (containing the old frame's subspaces, §2.3; any structure >= the old), or None
            hint_tangent:  typ.Optional[typ.Any],      # the hint: a tangent at hint_frame as project returns it, or None
            seed:          typ.Optional[int] = None,   # None = fv_operations.ZERO_FRAME_SEED
    ) -> typ.Tuple:                                    # the padded frame (this layer's frame type)

    def padded_structures(
            self,
            x_cores:     typ.Tuple,                    # X_old zero-padded to its ranks (old_ranks given), or any point (old_ranks=None)
            old_ranks:   typ.Optional[typ.Tuple],      # (tucker_ranks, tt_ranks) of X_old, structurally minimal; uniform: (d,)+C / (d+1,)+C int arrays allowed; None = the zero tensor
            oversample:  int,                          # p >= 0: +p on every Tucker rank and interior bond, minimal, never below x_cores' ranks
    ) -> typ.Tuple[
        typ.Optional[typ.Tuple],  # old_frame:   the frame of X_old (uniform: built through the ragged path); None when old_ranks is None
        typ.Tuple,                # target:      a frame with x_cores' structure (shapes only)
        typ.Tuple,                # oversampled: a frame at the oversampled ranks (shapes only)
        typ.Any,                  # the geometry of the oversampled structure (self on the ragged layer)
    ]:

    def require_frame_structure(
            self,
            x_cores:  typ.Tuple,   # the point
            frame:    typ.Tuple,   # a frame claimed to be at x_cores' structure (this layer's frame type)
    ) -> None:                     # structural ValueError naming the mismatching store; both modes
```

`padded_structures` raises (structural) on old ranks of the wrong length, above `x_cores`' ranks, or not
structurally minimal (§3.4). The `Geometry` protocol docstring (`backend/optimizers.py:78-85`) lists the three as
optional members beside `zero_point_frame`, with what their absence means.

### 5.5 `backend/optimizers.py`

```python
PADDED_START_PASSES     = 4   # newton_cg(padded_from=...) / padded_start: gradient evaluations (P4, downstream)
CONE_REFERENCE_PASSES   = 10  # g0_reference='cone' (P12; downstream within 2.4 % of the dense frame value at ranks >= 6)
PADDED_FRAME_OVERSAMPLE = 5   # the oversampling p of both (P5, P12; downstream e01 r = 3 stalled at p = 0)

@dc.dataclass(frozen=True)
class PaddedFrameOptions:
    """Solver options of the padded-frame iteration (``padded_start``, ``newton_cg(start_options=,
    cone_options=)``); a ``None`` field means the call site's default. Structural validation in
    ``__post_init__``: ``passes >= 1``, ``oversample >= 0``, ``0 <= seed < 2**32`` (integers)."""
    passes:      typ.Optional[int] = None   # gradient evaluations; None = PADDED_START_PASSES / CONE_REFERENCE_PASSES
    oversample:  typ.Optional[int] = None   # p; None = PADDED_FRAME_OVERSAMPLE
    seed:        typ.Optional[int] = None   # the hint-free start's seed; None = fv_operations.ZERO_FRAME_SEED (resolved by the geometry)

@dc.dataclass(frozen=True, eq=False)
class PaddedStart:
    """The result of :py:func:`padded_start`."""
    frame:   typ.Any                        # the padded frame at x_cores' structure; with old_ranks=None the cone frame at zero
    gnorms:  typ.Tuple[typ.Any, ...]        # ‖P_T ∇f‖ at each pass's frame (oversampled structure: not comparable to gnorm); shape-C arrays when stacked
    gnorm:   typ.Optional[typ.Any]          # ‖P_T ∇f‖ at the returned frame (None when final_gradient=False); the cone reference iff old_ranks is None

def padded_start(
        problem:         Problem,                                # the level's Problem (manifold geometry)
        x_cores:         Cores,                                  # X_old zero-padded to its ranks; with old_ranks=None any point (its ranks)
        old_ranks:       typ.Optional[typ.Tuple] = None,         # (tucker_ranks, tt_ranks) of X_old; None = the zero tensor (the cone)
        options:         typ.Optional[PaddedFrameOptions] = None,  # None = all defaults (PADDED_START_PASSES passes)
        final_gradient:  bool = True,                            # evaluate gnorm at the returned frame (one more sweep + transpose)
) -> PaddedStart:
```

Docstring: §2.6 in brief; what `gnorm` is in each mode (§3.5); cost (`passes (+1)` frame sweeps and transposes,
one residual); the misfit only; stacked `C` supported, per-element norms; raises on a geometry without the members.
Doctest: on a seeded `apply` problem at a `resize`d point, `st.frame` is a frame of the point (base point
`allclose`), `len(st.gnorms) == 4`, and `st.gnorm > 1.2 * g_lib` with `g_lib` the library frame's gradient norm (a
wide margin, so the doctest is stable across numpy generations).

`Problem.local_model` gains one keyword (last):

```python
            frame:              typ.Optional[typ.Any] = None,   # an ORTHONORMAL frame of x_cores' TENSOR at its structure (e.g. a padded frame); None = geom.frame(x_cores)
```

`newton_cg` gains four keywords, after `g0norm_cg` (the reference block):

```python
        start_frame:            typ.Optional[typ.Any] = None,   # iteration 0's frame: an orthonormal frame of x0's tensor at x0's structure; None = geom.frame(x0)
        padded_from:            typ.Optional[typ.Tuple] = None, # X_old's (tucker_ranks, tt_ranks): iteration 0's frame = padded_start(...).frame
        start_options:          typ.Optional[PaddedFrameOptions] = None,  # padded_from only: passes / oversample / seed
        cone_options:           typ.Optional[PaddedFrameOptions] = None,  # g0_reference='cone' only: passes / oversample / seed
```

`stats` gains `'start_frame_source'` (a label: `'given'` / `'padded'`) and, with `padded_from`, `'padded_start_gnorms'` — **only on
runs that use them**; `NewtonInfo` and the history rows are unchanged. `gradient_descent` gains `cone_options` (same
line, after `g0_reference`). `_G0_REFERENCES = ('zero', 'warm_start', 'cone')`; `_resolve_g0_reference(...,
cone_options=None)` returns `('cone', value)`; `_cone_reference_gnorm(problem, x0, cone_options) -> float` is
private; private `_prepare_jit_frame(use_jit, frame)` uses `common.partition_static` / `rebuild_static` around
`tree_to_jax`. New public names in `__all__`: `PaddedFrameOptions`, `PaddedStart`, `padded_start` (the constants are
module-level, like `ZERO_FRAME_SEED`).

### 5.6 `backend/optimizer_display.py`

`_REFERENCE_NAMES['cone'] = 'tangent cone'`. A padded start is not visible in the display at all (the display prints
no `stats` keys); it shows only through `stats['start_frame_source']`.

### 5.7 Frontend `t3toolbox/optimizers.py`

```python
def padded_start(
        geometry,                    # ragged/uniform MANIFOLD (or a shared wrapper); a corewise geometry raises
        kind:        str,            # 'apply' / 'entries' / 'probe' (+ '_derivatives')
        sample:      typ.Any,        # as newton_cg
        data:        typ.Any,        # as newton_cg
        x0:          Point,          # X_old zero-padded (old_ranks given; structurally minimal), or any point (old_ranks=None); stack C allowed
        old_ranks:   typ.Optional[typ.Tuple[typ.Any, typ.Any]] = None,   # X_old's (tucker_ranks, tt_ranks): int sequences, or (d,)+C / (d+1,)+C int arrays for a varying-rank uniform stack; None = the zero tensor of x0's ranks

        *,
        order:       typ.Optional[int] = None,       # derivative kinds
        weight:      typ.Optional[typ.Any] = None,   # residual weight ω
        chunk_size:  typ.Any = 'auto',               # as newton_cg
        options:     typ.Optional[bopt.PaddedFrameOptions] = None,
) -> typ.Tuple[
    typ.Any,   # T3Frame / UT3Frame: old_ranks given -> a padded frame of x0's tensor at x0's structure;
               #                     old_ranks=None  -> the cone frame at the zero tensor of x0's ranks (a start_frame for a zero x0 only)
    dict,      # {'gnorms': [...], 'gnorm': ...}: floats unstacked, shape-C arrays stacked; 'gnorm' is the cone reference iff old_ranks is None
]:
```

`newton_cg` mirrors the four keywords (`start_frame` a `T3Frame` / `UT3Frame`, forwarded as `.data`); `gradient_descent`
reaches `cone_options` through its `**kwargs` (docstring line only); `PaddedFrameOptions` is re-exported in `__all__`.
Not re-exported at the package root (as `least_squares_problem`). The frontend's checks are §3.6's. There is **no
frontend wrapper of the constructor**: `fv_padded_frame` is a backend primitive; the frontend surface is the start
it serves.

---

## 6. Tests

Numpy-only unless the test is about jax (house convention). New files `tests/backend/test_padded_frame.py` (T1–T4)
and `tests/backend/test_padded_start.py` (T5–T6); the rest extend the named files. "Valid frame" below means:
`fv_frame_orthogonality_residual ≤ 1e-14` (float64), both base-point reconstructions `allclose` to `X_old`
(rtol 1e-13), the old blocks present **bitwise** (`np.array_equal` on the leading slots), the template's shapes.
**Cross-layer comparisons are gauge-invariant** unless the inputs are identical by construction: ambient projectors
per family (`U_iᵀU_i`; the dense left/right interface projectors at each bond, small `d`), the dense `P_{T_F} Y` of a
fixed random `Y`, gradient norms and objectives — never gauge-carrying cores entrywise
(`uniform_equivalence_contract.md`).

| # | file | asserts |
|---|---|---|
| T1 | `test_linalg.py` | (a) `pinned_completion`: orthonormal and `⊥ pinned` for random, rank-deficient (duplicated columns, seeds inside `span(pinned)`), all-zero and too-few (`m < k`) seeds; `k = 0` → `(n, 0)` without a factorization; `s + k > n` → `ValueError`; float32 in → float32 out. (b) `s = 0` with the keyed Gaussian seeds equals `_haar_rows(...).T` **bitwise**. (c) `pinned_leading_columns`: a planted `A = P X + U_t diag(σ) V_tᵀ` with `U_t ⊥ P` and a gap returns `span(U_t[:, :k])` (projector distance `< 1e-12`); rank-deficient `A` is completed (still orthonormal, `⊥ P`). (d) `C = (2,)` and `(2, 3)` stacks equal the per-element results (`allclose` 1e-14). |
| T2 | `test_padded_frame.py` | hint-free `fv_padded_frame`: (a) `old_frame=None` equals `fv_zero_point_frame` **bitwise** — `d = 1, 3, 4`, unshared and `groups=((0, 1), (2,))`; (b) a valid frame at `resize`d points (Tucker-only, TT-only and joint increases); (c) the in-block cases (TT-only increase at bond 1; Tucker-only increase at saturated bonds) give valid frames and the structural counts, deterministically (two calls bitwise equal); (d) the Tucker completion is nested (`kU → kU + 1` extends the factor by one row, bitwise); (e) stacked `C` equals per-element calls. |
| T3 | `test_padded_frame.py` | hinted `fv_padded_frame`: (a) valid frames for a hint at the library frame of the padded point, at a padded frame, and a **zero** hint; (b) the internal coordinates (`_hint_coordinates`): `A_i`, `ΩL_k`, `ΩR_k` have orthonormal rows (`≤ 1e-13`); (c) **matrix fixed point** (`d = 2`, Tucker ranks = TT rank): with new slots aligned with the top singular subspaces of the normal part `N` and the hint `P_T G` at that frame, the constructor returns the same subspaces (projector distance `< 1e-10`); (d) the truncation (oversampled hint frame, target template) gives a valid target-structure frame; (e) sharing: tied `U` (`np.array_equal` across members), and the group's new rows span the top eigenvectors of the summed projected Grams `Σ_m P⊥ W_mᵀ M_m M_mᵀ W_m P⊥` (projector oracle, `< 1e-10`); (f) stacked `C` equals per-element calls (`allclose` 1e-12); (g) the `K` lift: a hint with variations stacked `K = (2,)` gives a `K + C` frame equal to the per-hint calls (`allclose` 1e-12); structural errors — a hint frame smaller than the old frame, a template smaller than the old frame, mismatched `d` / shape / stack — each a `ValueError` naming the store; (h) **dense oracle per step** (the only test that sees a wrong-subspace bug in the sweeps or the down step): with `Z` the dense hint, `Φ_k` / `Ψ_k` the dense left/right bases of the constructed frame and `Φ°`, `Ψ°` the old ones, the new directions equal (projector distance `< 1e-9`, on generic data with gaps and `≥ k` hint directions) — Tucker `i`: the leading left singular vectors of `P_{U°_i}^⊥ Z_(i) (I − E_dᵀE_d)`, `E_d[m] = Σ_ab O°_i[a,m,b] Φ°_i[a] ⊗ Ψ°_{i+1}[b]`; `L_k`: of `(I − PPᵀ) M (I − Ψ°_{k+1}ᵀΨ°_{k+1})`, `M = (Φ_k ⊗ U_k)ᵀ Z_{(0..k ∣ k+1..)}`, `P` the embedded `L°_k`; `R_k`: the mirror image (the content in `U_k ⊗ Ψ_{k+1}`, columns projected off `Φ°_k`, pinned off the embedded `R°_k`); `O_i`: the leading bond-space directions of `Z`'s coordinates in `Φ_i ⊗ U_i ⊗ Ψ_{i+1}` on the new Tucker slots, pinned off the embedded `O°_i` — at `C = ()` and `(2,)`, unshared and shared, `d = 3` and `4`; (i) a `d = 1` hinted call (empty sweeps, zero-size `Ω`) gives a valid frame. |
| T4 | `test_padded_frame.py` | `ufv_padded_frame`: (a) on identical inputs the real blocks equal `fv_padded_frame`'s **bitwise** (hint-free and hinted, `s = 0` and `s > 0`, shared); `ufv_padded_frame(..., old=None, hint=None)` equals `ufv_zero_point_frame`; (b) `out[5] is template[5]`, padding exactly zero; (c) NaN-filled padding in every input leaves the output unchanged; (d) a stack with per-element ranks equals per-element calls; (e) the uniform `padded_structures` old frame and hint-free start equal the ragged ones **bitwise** in their real blocks (they are built through the ragged path, §3.3), numpy. |
| T5 | `test_padded_start.py` | backend `padded_start`: (a) **matrix oracle** (`d = 2`, `entries` kind with every entry observed, so the ambient gradient is the residual matrix): at `s = 0`, `gnorm → ‖G_r‖` and at `s > 0`, `gnorm → (‖P_{T_X}G‖² + ‖N_{r−s}‖²)^{1/2}` to rtol 1e-8 with `passes=40, oversample=0` on a planted gap; (b) a seeded regression on c11–c13: the final captured norm exceeds the first pass's by a stated margin (the per-pass sequences of P13 are recorded as observations — the captured norm is not monotone in general, §2.2); (c) the frame is a frame of `x_cores` at its structure (`require_frame_structure` passes, base point `allclose`); with `old_ranks=None` its base point is 0; (d) `passes=1`, `oversample=0`, `final_gradient=False` (`gnorm is None`); (e) stacked `C`: per-element `gnorms` equal the unstacked runs; (f) ragged vs uniform, **gauge-invariant**: `gnorms` and `gnorm` equal to rtol 1e-10, the frames' per-family ambient projectors and their `P_{T_F} Y` equal to 1e-10 (at `s = 0` and `s > 0`); (g) structural errors: corewise geometry, a geometry stub without the members, `old_ranks` above `x_cores`' ranks, of the wrong length, or not minimal; a uniform `x0` with gappy rank masks — the `uz + ua` case of §3.4 (masks `[1 0 1 0]`, old ranks 1) — raises under `safety.safe()` **and** `safety.unsafe()` (it is structural), where the prefix gather would have taken the zero block; (h) the oversampled structure at `p = 0` equals the target for a non-minimal `x_cores` (the maximum rule). |
| T6 | `test_padded_start.py` | `Problem.local_model(frame=)`: (a) `frame=geom.frame(x)` gives a model equal **bitwise** to the default (misfit, gradient, `hvp` on a fixed direction), GN and exact; (b) at a padded frame: the misfit is bitwise the default's (the residual does not see the frame); `lm.retract` of the zero tangent returns `X_old`; (c) a wrong-structure frame → `ValueError` under `safety.safe()` and `safety.unsafe()`; a corewise geometry → `ValueError` naming `require_frame_structure`; (d) exact mode at a padded frame: builds, `lift_sigma_min` is at the clip (the new slots' zero singular values), `hvp` symmetric (the existing symmetry check, 1e-12). |
| T7 | `tests/backend/test_optimizers.py` | `newton_cg` seam: (a) the default `set(stats)` is unchanged (the existing pins at `:477`, `:860` stay as they are); `newton_cg(x0)` and `newton_cg(x0, start_frame=geom.frame(x0))` have equal histories in every field (the seam adds no arithmetic) and the second's `stats` differ only by `'start_frame_source': 'given'`; (b) `padded_from`: iteration 0's `info.lm.frame` equals `padded_start(...).frame` bitwise (same options); `stats['start_frame_source'] == 'padded'`, `stats['padded_start_gnorms']` has `passes` entries; the run converges (`stop_reason='gradient'`); (c) structural errors: `start_frame` with `padded_from`, `start_options` without `padded_from`, `cone_options` without `'cone'`, an invalid `PaddedFrameOptions`, `padded_from` / `start_frame` on corewise; (d) from the first accepted step on, `info.lm.frame` equals `geom.frame(info.x_cores)` bitwise (iteration 0 only); every row at `x0` (rejected trust-region trials included) carries the start frame; (e) `use_jit=True` on numpy inputs with a numpy `start_frame`: runs, the frame's arrays are jax and a uniform frame's masks stay numpy; `bopt._jitted(bopt._cg_core, (4,))._cache_size()` grows by no more than the default run's (the `test_dispatch.py:1194-1207` pattern); (f) shared ragged and uniform geometries: `padded_from` runs, the start frame is tied; (g) `padded_from` equal to `x0`'s own ranks (zero counts everywhere): ragged — the start frame is the library frame bitwise and the history equals the default run's; uniform — the old frame comes through the ragged path, so the run equals the default one gauge-invariantly (objectives rtol 1e-10). |
| T8 | `tests/backend/test_zero_reference.py` | `g0_reference='cone'`: (a) `d = 2` oracle: equals `‖G_r‖` (the cone value, **not** `‖G_{2r}‖`) to rtol 1e-8 at `PaddedFrameOptions(passes=40, oversample=0)`, to 2 % at the defaults; (b) T3: within `[0.95, 1.10]×` the dense-frame value `‖P_{T_F}G‖` (`F` = frame of `dense_t3svd(G)`, zero carriers) on the two P12 problems; (c) ragged == uniform to rtol 1e-10; (d) the default is unchanged (`'zero'` recorded); `'cone'` on corewise → `ValueError`; all-zero data → `ValueError`; with `g0norm_newton` it is not computed (spy on `padded_start`); (e) rows record `g0_reference='cone'`, `cg_reference='cone'`; the converged display line reads `g₀ = tangent cone`; (f) `gradient_descent(g0_reference='cone', cone_options=...)`; (g) seeds 0–3 give cone references within 2 % of each other on a gapped problem; (h) the staggered completion (§2.2) is not a fixed point: at `d = 2`, `s = 0`, a frame with `U = u_1`, `V = v_2` captures `‖G_2‖` (the frame-projection supremum at `r = 1`), and one constructor pass from it returns the aligned frame (projector `< 1e-12`), which captures `‖G_1‖`. |
| T9 | `tests/test_optimizers_frontend.py` | (a) the keyword mirror (`:448`) covers the four keywords with the backend defaults (automatic); (b) the forwarding test's `opts` (`:468`) gains `padded_from`, `start_options`, `cone_options` (and `g0_reference='cone'` in a second call); a `start_frame` is forwarded as its `.data` (separate assertion); (c) frontend `padded_start`, ragged and uniform: returns `T3Frame` / `UT3Frame`; with `old_ranks` given `frame.to_t3()` / `.to_ut3()` equals `x0` (`allclose`), with `old_ranks=None` it is zero; a `C`-stacked `x0` gives a stacked frame and shape-`C` norms; (d) safe mode: nonzero padding with `padded_from` → `ValueError`, skipped under `safety.unsafe()`; garbage in a uniform `x0`'s padding beyond its own masks is **not** flagged; a `start_frame` of another tensor → `ValueError`; a non-orthonormal `start_frame` → `ValueError` (ORTH); untied `x0` on a shared geometry with `padded_from` → `ValueError` (TIED); all safe-only; structural (both modes): `padded_from` or `start_frame` with a non-minimal `x0`, a ragged frame with a uniform `x0`; (e) **the equivalence test**: ragged vs uniform `newton_cg(padded_from=...)` on the `x01`-style restart give per-iteration objectives equal to rtol 1e-8 over 10 iterations (the library start: 5.6e-1, `uniform_equivalence_contract.md`). |
| T10 | `tests/test_dispatch.py` | numpy in → numpy out, jax in → jax out for the two primitives, `fv_padded_frame`, `ufv_padded_frame` (jax supercores out, masks numpy — the `ufv_zero_point_frame` rows at `:1898-1912`), the uniform geometry members `padded_frame` / `padded_structures`, and `padded_start` on both layers; `jax.jit(fv_padded_frame, static_argnames=('groups', 'seed'))` (hint-free and hinted) equals eager (rtol 1e-5 float32, 1e-12 x64). |
| T11 | `tests/test_jax_absent.py` | with jax hidden: the new functions import and run on numpy; `newton_cg(padded_from=..., use_jit=True)` warns once and runs eager. |
| T12 | `tests/test_contractions_interpreter.py` | `HISTORICAL` gains the twelve strings of §4.10 (each `''`); `TestVocabularyCompleteness` would fail without them; `test_contractions_sharding.py` covers them automatically. |
| T13 | `tests/backend/test_sharing.py` | the end-to-end shared continuation test, rerun with `padded_from` at each level, reaches the same true shared ranks; `frame_has_minimal_ranks` unaffected (diagnostic only). |
| T14 | doctests | every new docstring example (§5) and every new `>>>` block in the amended docs (§7) runs in the doctest suite. |

**Measurement gates** (not unit tests; recorded in `docs/contributor/verification_records.md` at the slice that
enables them): G-P4 — through the library API, the padded start reproduces P4 (c11–c16: `f1/f0` below the
library's on ≥ 5 of 6 and the Newton total ≤ the library's on every case); G-P12 — the cone reference reproduces
P12's range; G-bit — three seeded default runs (ragged GN line search, ragged exact trust region, uniform GN) on
`main` and on the branch in one session give **equal `stats` dicts**: the same keys, every history field bitwise
except `wall_time` (a golden file is not used: bitwise results are platform-specific); G-acyclic (S3) — the backend's
import graph, built by an AST scan from every import node (module- and function-level), has no cycle, and the backend
has no function-level import of a library module — both hold today, and the gate keeps holding for later features
(under option A of I-32 the second assertion lists the one recorded exception).

---

## 7. Docs and CHANGELOG

Every page below is amended in slice 10 (the API pages' docstrings land with their code). Text is given in full
where it is new; for amendments, the replacement or the added paragraph. Shipped text cites no `dev/` file.

**`docs/fitting_and_optimization.md`**

- §2.1, the `stats` table: two rows — `start_frame_source`: "the label `'given'` / `'padded'` — present only when `newton_cg` was
  given `start_frame` or `padded_from` (§5)"; `padded_start_gnorms`: "with `padded_from` only: the captured
  gradient norm at each pass of the padded start".
- §3, the `Geometry` bullet (`:269-283`): after the `zero_point_frame` sentence — "Three more are optional, for the
  padded start (§5): `padded_frame`, `padded_structures` and `require_frame_structure`. The manifold geometries
  define them; a geometry without them — the corewise ones, whose frame is the point itself, and any custom
  geometry — cannot take `padded_from` / `start_frame` / `g0_reference='cone'` (a structural error names the
  member)."
- §4.11, a closing paragraph: "**From a zero-padded restart the divergence can be removed.**
  `newton_cg(..., padded_from=old_ranks)` builds iteration 0's frame from the data; the uniform layer builds its
  old block through the ragged construction, so the two layers start from the same frame up to rounding and take
  the same first step (a near-tie in the gradient's spectrum can still separate them, as for any truncation). From
  a zero start, pass the frame `optimizers.padded_start(..., x0)` returns as `start_frame` (§5)."
- §4.12, "Restarts need nothing special": add "A padded start frame (§5) changes none of this: its new slots carry
  no content, so the lift clips them exactly as at the library's completion."
- §5, the reference bullet:
  - the table gains `| 'cone' | the norm of the projection of the gradient at the zero tensor of x0's ranks onto
    the tangent cone there — a best rank-r approximation of it: exact for matrices; for T3, the frame-gradient
    norm at the fixed point of a subspace iteration, within a few percent of the quasi-optimal value (≈ 11
    sweeps and transposes per run; explicit only) |`;
  - "*What exactly the reference is*" — replace "good to an O(1) factor (another seed, or a rotation of the
    problem, moved it by up to ~1.7× in the measured cases) — harmless for a relative stop, but not an intrinsic
    property of the problem" by: "a **frame projection**: the gradient at the zero tensor projected onto one
    completion's tangent space. On generic data different seeds and rotations move it by up to ~1.7×. On
    importance-ordered bases (active subspaces, Hermite, KL — the common surrogate case) a Haar completion is a
    random projection that captures about `r/N` per mode, and the reference can be far below the cone value
    (`g0_reference='cone'`): **at least 15–76×** below it over six continuation levels of a measured Darcy surrogate
    (71×, 76×, 41×, 26×, 17×, 15×). A relative stop then asks for that much more accuracy, and CG sits at its
    forcing cap." Keep the three convention properties (they remain true of `'zero'`). No tolerance is
    recommended for `'cone'` in this release (§3.7 of the plan; the docs say "the cone value is larger, so retune
    `gtol_rel` when switching").
  - a new bullet after it: "**Padded starts.** A zero-padded `resize` is rank deficient: its frame's new slots are
    a completion the library chooses (LAPACK's on the ragged layer, a pad-safe sketch on the uniform one),
    unrelated to the data. `newton_cg(..., padded_from=(old_tucker_ranks, old_tt_ranks))` replaces it, for
    iteration 0 only, by the **padded frame**: the old frame kept, the new slots the leading directions of the
    gradient's content that is new relative to the old point, found by a few passes of subspace iteration
    (`start_options=PaddedFrameOptions(passes, oversample, seed)`; defaults 4 and 5). The point is not changed —
    no content is injected. Measured on six small continuation cases whose previous level had **converged**: the
    first Newton step removed 65–85 % of the objective against 14–32 % from the library completion, and each
    level took 1–3 fewer Newton steps. After a level that **stopped early** it was neutral to worse (up to 4 more
    Newton steps measured), so use it after converged levels. `optimizers.padded_start(...)` returns the frame
    itself (to pass as `start_frame`; with `old_ranks=None` it is the cone frame at the zero tensor, a start frame
    for a zero `x0`)." With a doctest: a `resize`, then `newton_cg(..., padded_from=...)`, printing
    `st['start_frame_source']` → `'padded'`.
- §6 Pointers gains `contributor/padded_frame_internals.md`.

**`docs/rank_continuation.md`**

- The reference box (`:103-121`): replace "good to an O(1) factor — harmless for a relative stop" by "a frame
  projection — on importance-ordered inputs it can be one to two orders of magnitude below the cone value
  (`fitting_and_optimization.md` §5), which `g0_reference='cone'` computes". Add: "A **padded start** —
  `newton_cg(..., padded_from=(old_tucker, old_tt))` with the ranks before `resize` — takes iteration 0 in the
  data's directions instead of an arbitrary completion; measured 1–3 Newton steps fewer per level when the
  previous level converged, neutral to worse when it stopped early."
- The loop sketch (`:132-137`): pass `padded_from` only after a converged level —
  `padded_from=old_ranks if stats['stop_reason'] == 'gradient' else None`, with `old_ranks = (X.tucker_ranks,
  X.tt_ranks)` read before the `resize`.
- "Shared Tucker factors" (`:164-196`) and "On the uniform layer" (`:198-231`): one sentence each — the padded frame
  is tied (one factor per group, from the group's concatenated gradient content; the old factors must be tied); on
  the uniform layer the old block is built through the ragged construction, so ragged and uniform continuations
  start each level from the same frame up to rounding.

**`docs/numerical_contracts.md`** — the row at `:203`: "good to an O(1) factor (… ~1.7× …)" becomes "a frame
projection (seed-dependent by ~1.7× on generic data; on importance-ordered data at least 15–76× below the cone value,
`g0_reference='cone'`)". New rows in "The optimizers' relative stop": `'cone'` (structural: a geometry without
`padded_structures`, a vanishing value); "`newton_cg(start_frame=, padded_from=)` / `padded_start`" — structural:
frame structure (`require_frame_structure`), mutual exclusion, option validity, old ranks above `x0`'s or not
minimal, a uniform `x0` whose rank masks are not prefix masks (gappy after `+` / `×`), `padded_from` / `start_frame` on
a non-minimal frontend `x0`, a corewise geometry; numerical, safe mode
only: `x0` exactly zero outside the old ranks (real slots), the start frame **ORTH** and representing `x0`, **TIED**
`x0` / start frame on a shared geometry. The `fitting.LocalModel` section gains: "a frame supplied by the caller
(`Problem.local_model(frame=)`, `newton_cg(start_frame=)`) makes ORTH a precondition rather than a construction
invariant".

**`docs/frame_variations.md`** — a new section after "Four rank stores" (`:68-112`):

> ## Padded frames: at a rank-deficient point the frame is a choice
>
> A zero-padded point (a `resize` warm start, or the zero tensor) has fewer independent directions than its
> stores. Its frame keeps the old block — the old point's frame, embedded in the leading slots, with the carriers
> `L_{d−1}`, `R_0` holding only the old content — and **completes** the new slots with any orthonormal directions:
> every completion is a valid frame (per-core orthonormality and the gauge are all the tangent machinery needs),
> they all reconstruct the same point, and their tangent spaces differ. The library's sweep picks a completion
> without looking at the data; `backend.padded_frames.fv_padded_frame` picks one structurally — counts from the
> rank stores, no threshold — either by a seeded convention (`fv_zero_point_frame` is its zero-tensor case) or from
> a hint tangent, taking the leading directions of the hint's content that is new relative to the old point
> (`contributor/padded_frame_internals.md`).

**`docs/uniform_equivalence_contract.md`** — after the full-rank qualifier's measured table (`:61-63`): "**Padded
starts make the choice once, for both layers.** `newton_cg(padded_from=...)` builds iteration 0's frame by one
ragged construction — on the uniform layer from real blocks unpacked from the supercores, with the old frame
itself computed through the ragged path — so both layers start from the same frame up to rounding and take the same
first step; the iterate is then numerically full rank and the element-wise contract applies (measured: per-iteration
objectives agree to `<1e-8` over 10 iterations, `tests/test_optimizers_frontend.py`). A near-tie in the gradient's
spectrum at a cut can still separate them, as for any truncation." The "Consequence for testing" paragraph gains:
"…or start both layers from the same padded frame, and compare gauge-invariant quantities."

**`docs/sharing.md`** — "Rank machinery … Continuation" (`:197-211`): "With `newton_cg(padded_from=…)` the new group
levels are completed from the group's concatenated gradient content (face 1 above) instead of arbitrarily — still
exactly zero in the point, so the tied channel stays gated, but the TT channel's escape starts in the data's
directions. The old factors must be tied (TIED; safe mode checks it)."

**`docs/api_reference.rst`** — "Fitting and optimization": add `optimizers.padded_start` and
`optimizers.PaddedFrameOptions` (module-level, not at the root), and the `newton_cg` keywords sentence; the backend
surface's frame-variations area: `padded_frames.fv_padded_frame`, `padded_frames.ufv_padded_frame`,
`linalg.pinned_completion`, `linalg.pinned_leading_columns`.

**`docs/naming_conventions.md`** — beside the "Shared Tucker factors" sentence (`:45-48`, the `sharing` example of a
feature module whose functions carry their data's prefix): add "Likewise `backend/padded_frames.py` hosts the `fv_` /
`ufv_` padded-frame constructors: named for their operands, placed for their imports (`tv_to_t3` sits above
`fv_operations`)." (Not in the "Outside the matrix" list, which names infrastructure and the fitting stack only. Under
option A of I-32 this line is not needed.)

**`docs/contributor/padded_frame_internals.md`** (new; in `contributor_guide.rst`'s toctree after
`curvature_internals`): the design record — §2 of this plan (the cone and the frame tangent spaces, why the aligned
completion and not the supremum, the pinned T3-SVD, two-sidedness, the δ-limit and why it is not the target, the
hint-free conventions, GD = subspace iteration, the cone reference and what it is not), §4's algorithm with the
contraction table, the cross-layer construction of §3.3, the placement (I-32), §9's decisions with their reasons, and the measurements
P1–P14.

**`docs/contributor/curvature_internals.md`** — "The canonical zero-point frame" (`:171-222`): the "good to an O(1)
factor" sentence (`:217`) is corrected as in `fitting_and_optimization.md`, with a pointer to the cone reference; the
table row E9 (`:307`): "good to an O(1) factor on generic data; not on importance-ordered data (2026.4.0,
`padded_frame_internals.md`)".

**`docs/contributor/deferred_and_rejected.md`** — I-29's "A possible later refinement … averaging the reference over
a few seeds" (`:60-62`) → "superseded by the cone reference (2026.4.0): a seed average estimates the mean frame
projection, a completion-dependent number; the cone reference is the projection onto the tangent cone, computed at
the subspace iteration's fixed point". The deferred "zero **start** using the canonical zero-point frame"
(`:227-229`) → "available opt-in: `start_frame=` (2026.4.0); the default zero start is unchanged". New "Rejected"
entries: the δ-limit frame as the definition (P9); the largest frame projection `sup_F ‖P_{T_F}G‖` as the reference
(it is the rank-`2r` value, attained at staggered completions that the iteration does not converge to); the Tucker
shortcut from the hint's `V_i` alone (P7); carrying the projected content in the sweep ("rotation corrections",
P10); coordinate seeds for the in-block completion; ε-content enrichment (downstream); a frontend wrapper of the
constructor; padded-start members on the corewise geometries; a numerical rank decision anywhere (decision 4). New
"Deferred": `'cone'` or `padded_from` as defaults (O-1, O-2); a vectorized uniform twin for uniform-rank stacks
(O-5); a tolerance recommendation for `'cone'` (pending the downstream measurement).

**`docs/contributor/verification_records.md`** — a new section "The padded frame and the cone reference
(2026.4.0)": P1–P14, the measurement gates G-P4, G-P12, G-bit, G-acyclic, and the downstream acceptance numbers. The
"zero reference" section's E9 line gains the cone/Haar ratios of §0.

**`docs/contributor/sharing_internals.md`** — "Padded restarts" (`:98-122`): one paragraph — the padded frame aims
the escape (the TT channel's first-order mass pairs with data-informed tied directions); full shared rank stays a
diagnostic; TIED old factors are a precondition of the padded start.

**`docs/exact_hessian_two_hole.tex`** — the zero-reference subsection (`:904-940`): `:937`'s "good to an $O(1)$
factor" is corrected, and a paragraph defines the cone reference: the norm of the projection of $G$ onto the
tangent cone of $\mathcal M_{\le r}$ at $0$ (a best rank-$r$ approximation; $\|G_r\|$ for matrices), estimated by
$\|P_{T_{F^*}}G\|$ at the fixed point $F^*$ of the subspace iteration — equal to it for matrices (the aligned
completion); for T3 at least the norm of $G$'s projection onto $F^*$'s own rank-$r$ tensor space, possibly above the
cone value through the rank-$2r$ directions of $T_{F^*}$, and measured within a few percent of the value at the
T3-SVD frame — and **not** $\sup_F\|P_{T_F}G\|$, which those rank-$2r$ directions make larger ($\|G_{2r}\|$ for
matrices). The PDF is rebuilt at release.

**`docs/release_notes.md`** — a new head "Upgrading from 2026.3.0": "Nothing breaks: every new behaviour is opt-in,
and without the new keywords every run — its `stats` included — is bit-identical to 2026.3.0. Two additions are
worth trying. In a continuation loop, pass `padded_from=(old_tucker_ranks, old_tt_ranks)` (the ranks before
`resize`) to `newton_cg` at a level whose predecessor converged: measured 1–3 fewer Newton steps per level on small
cases, and neutral to worse after a level that stopped at its cap. If your inputs are importance-ordered and your
stops are relative, `g0_reference='cone'` gives a reference that does not depend on a chosen completion; the
default zero reference measured at least 15–76× below it on a Darcy surrogate. The cone value is the larger one, so retune
`gtol_rel` when switching. The docs' earlier claim that the zero reference is good to an O(1) factor was wrong on
such data and is corrected."

**`CHANGELOG.md`** — a new top section:

```markdown
## [2026.4.0] — unreleased

The padded start: a structural, data-informed frame for a rank-deficient point, and the Newton reference it makes
possible. Every addition is opt-in; without the new keywords every run, its `stats` included, is bit-identical to
2026.3.0.

### Added

- **`newton_cg(..., padded_from=old_ranks)`** starts a zero-padded continuation level from the **padded frame**:
  the old point's frame kept, the new slots completed by the leading directions of the gradient's content that is
  new relative to the old point (a few passes of subspace iteration at oversampled ranks, then truncated). The
  point is unchanged — no content is injected — and every count is structural (no threshold). On six small
  continuation cases whose previous level had converged, the first Newton step removed 65–85 % of the objective
  against 14–32 % from the library's completion and each level took 1–3 fewer Newton steps; after a level that
  stopped early it was neutral to worse. The ragged and uniform layers start each level from the same frame up to
  rounding. Options: `start_options=PaddedFrameOptions(passes=4, oversample=5, seed=0)`. Such runs record
  `stats['start_frame_source']` and `stats['padded_start_gnorms']`.
- **`newton_cg(..., start_frame=frame)`**: iteration 0's frame given explicitly (an orthonormal frame of `x0`'s
  tensor at its structure); and **`optimizers.padded_start(...)`**, which returns the padded frame (with
  `old_ranks=None`, the cone frame at the zero tensor) and the captured gradient norms.
- **`g0_reference='cone'`** (`newton_cg`, `gradient_descent`): the Newton reference as the norm of the projection
  of the zero-tensor gradient onto the **tangent cone** at zero — a best rank-`r` approximation of it: exact for
  matrices; for T3, the frame-gradient norm at the fixed point of a subspace iteration, within a few percent of the
  quasi-optimal value (≈ 11 sweeps and transposes; `cone_options=`).
  Unlike the default `'zero'` reference it does not depend on a chosen completion. Explicit only; the default is
  unchanged.
- Backend: `fv_padded_frame` / `ufv_padded_frame` (the constructor; `fv_zero_point_frame` is its zero-tensor,
  hint-free case, bit for bit), `linalg.pinned_completion` / `pinned_leading_columns`,
  `backend.optimizers.padded_start` / `PaddedStart` / `PaddedFrameOptions`, `Problem.local_model(frame=)`, and the
  optional manifold-geometry members `padded_frame`, `padded_structures`, `require_frame_structure`.

### Fixed (documentation)

- The zero reference was documented as "good to an O(1) factor". That holds across seeds on generic data, but on
  importance-ordered bases a Haar completion captures about `r/N` per mode, and on a Darcy surrogate the reference
  measured at least 15–76× below the tangent-cone value. The docs now describe it as a frame projection and point to
  `g0_reference='cone'`.
```

---

## 8. Slices

One commit per slice (files staged by name), the full suite green at each; review stops after S3 (the
construction), S7 (the optimizer seam) and S10 (the docs) — Nick's "discussion between steps".

| slice | content | files | gate |
|---|---|---|---|
| S1 | `pinned_completion`, `pinned_leading_columns`; HISTORICAL strings 1–2 | `backend/linalg.py`, `tests/backend/test_linalg.py`, `tests/test_contractions_interpreter.py` | T1, T12 (partial), T10 (primitives) |
| S2 | the new module; `fv_padded_frame`, hint-free path: `_empty_old_frame`, `_embed`, `_outside_first`, the in-block fallback, validation | `backend/padded_frames.py` (new), `tests/backend/test_padded_frame.py` | T2 (the `s = 0` bitwise identity first) |
| S3 | the hinted path: `_hint_coordinates`, `_tucker_completion` (groups), `_pinned_left_sweep` (both chains), `_down_completion`, the `K` lift; HISTORICAL 3–12 | `backend/padded_frames.py`, tests | T3 (incl. the dense oracle T3(h)), T10 (jit), T12, G-acyclic; P1/P3 re-measured through the function. **Review stop.** |
| S4 | `ufv_padded_frame` | `backend/padded_frames.py`, tests | T4(a)–(d), T10 (uniform rows) |
| S5 | the three geometry members on the two manifold classes (the uniform old frame through the ragged path; the oversampled maximum rule; the minimal-old-ranks check; the uniform prefix-mask check, I-34); the protocol docstring | `backend/geometry.py`, `backend/optimizers.py:65-131` | T4(e), T5(g)–(h), member rows of T10, G-acyclic |
| S6 | `PaddedFrameOptions`, `PaddedStart`, `padded_start`, the constants; `Problem.local_model(frame=)` | `backend/optimizers.py`, `tests/backend/test_padded_start.py` | T5 (matrix oracle), T6 |
| S7 | `newton_cg(start_frame=, padded_from=, start_options=)`, the `stats` keys (only when used), `_prepare_jit_frame` | `backend/optimizers.py`, `tests/backend/test_optimizers.py` | T7, T11 (partial); G-bit (equal `stats` dicts); G-P4. **Review stop.** |
| S8 | `g0_reference='cone'`, `cone_options` (both optimizers), `_cone_reference_gnorm`, the display label | `backend/optimizers.py`, `backend/optimizer_display.py`, `tests/backend/test_zero_reference.py` | T8; G-P12 |
| S9 | frontend `padded_start`, the four `newton_cg` keywords, safe-mode checks, the re-export | `t3toolbox/optimizers.py`, `tests/test_optimizers_frontend.py`, `tests/test_jax_absent.py`, `tests/test_dispatch.py` | T9 (incl. the equivalence test), T10, T11, T13 |
| S10 | every doc of §7, the internals page, CHANGELOG, release notes, the math-note paragraph | `docs/…`, `CHANGELOG.md` | T14 (doctests), the Sphinx build. **Review stop.** |
| S11 | downstream acceptance in T3Polynomial (§11) — measurement, no library change | T3Polynomial `scripts/`, `dev/` | the §11 numbers recorded in `verification_records.md` and the CHANGELOG |
| S12 | release 2026.4.0 | `pyproject`, tag | the release checklist |

---

## 9. Decisions

Nick's binding decisions are I-1–I-5. The rest are the writer's judgment calls, each with its reason; the critic and
Nick should weigh the ones marked **(weigh)**. Round 2 changed I-5, I-18, I-21 and added I-28–I-33; round 3 made I-32 an
explicit judgment call for Nick, renamed the `stats` key in I-18, and added I-34.

| I | decision | settled as |
|---|---|---|
| I-1 | surface | both: the standalone `padded_start` (backend + frontend) and the `newton_cg` keywords (Nick 1) |
| I-2 | defaults | nothing changes without a new keyword: `g0_reference` default `'zero'`, `fv_zero_point_frame` untouched, `newton_cg` bit-identical **including `stats` and history** (Nick 2; G-bit, T7(a)) |
| I-3 | options | module constants + one frozen `PaddedFrameOptions(passes, oversample, seed)` carried by two keywords (`start_options`, `cone_options`) — the two call sites share the three knobs (Nick 3 left the form open; the critic agrees) |
| I-4 | structural only | counts from the stores; no threshold; no content in the point; the retraction and the orthogonal representation untouched; the pinned T3-SVD is a new function on the hint (Nick 4) |
| I-5 | stacking | `fv_padded_frame` vectorized over `C`; a `K`-stacked hint **lifted** to `K + C` outputs (one padded frame per hint, by broadcasting — the library's lift, not a reduction); `ufv_padded_frame` a host loop (varying ranks allowed); `padded_start` over `C` (backend and frontend); the optimizers unstacked (Nick 5; §3.9) |
| I-6 | the construction | HOSVD-style two-sided Tucker step + mirrored pinned TT sweeps + the down step from the new Tucker slots (design (d), Appendix B) — measured best or equal (P4, P6), group-ready, one sweep code path; verified per step against a dense oracle by the critic (≤ 4.4e-11), and T3(h) makes that a test |
| I-7 | the δ-limit | not the definition: it carries `O(t)` rotations of the old block in `R`, `O` and measured worse (P9) |
| I-8 | the down step | the hint's center in the final bases, restricted to the new Tucker slots, pinned off `O°` (§2.3); at `s = 0` it gives the down rows of the truncated hint's center, the frame P12's dense reference uses |
| I-9 | rotation corrections | dropped: inert at converged and unconverged levels (P10) |
| I-10 | Tucker coordinates | ambient, each member's normal content compressed, a group concatenated (sharing face 1); equal to the prototype's `w`-coordinate step whenever the hint has `≥ kU` directions, and free of `w_i ≥ nU_i` |
| I-11 | hint-free conventions | Tucker: Haar in `U°`'s complement from the keyed stream's first `kU` rows (nested; `s = 0` = `_haar_rows`); TT/D: outside-first coordinates (`s = 0` = identity prefixes), then an in-block completion from a seeded Gaussian keyed `[seed, i, c]` |
| I-12 | the driver's start | the hint-free frame at the oversampled structure, not the library frame of the padded point: layer-independent and deterministic; its measured cost is about one pass |
| I-13 | passes | `passes` = gradient evaluations, the last construction truncates; defaults 4 (start), 10 (cone), `p = 5` (P4, P5, P12, downstream) — provisional until S11 **(weigh)** |
| I-14 | the residual | once per call, at the padded point (or at zero); the misfit only — the regularizer's gradient is tangent to the old stratum and removed by the two-sided steps |
| I-15 | where the loop lives | `backend/optimizers.py` (T3-free) with three optional duck-typed geometry members; a geometry without them keeps working |
| I-16 | the oversampled geometry | returned by `padded_structures` (the uniform geometry holds variation masks for one structure) |
| I-17 | `local_model(frame=)` | preconditions ORTH and "same tensor"; structural check always (`require_frame_structure`), the numerical ones in frontend safe mode only (house rule) |
| I-18 | recording the start frame | `stats['start_frame_source']` (a label: `'given'` / `'padded'`) (and `stats['padded_start_gnorms']` with `padded_from`) **only on runs that use a start frame**; no history field — default runs keep today's `stats` exactly (decision 2) |
| I-19 | the zero-tensor start | not a `padded_from` value: `padded_start(old_ranks=None)` + `start_frame=` (two calls; the returned frame is the cone frame at zero, a start frame for a zero `x0` only) |
| I-20 | frontend `padded_from` / `start_frame` | require a structurally minimal `x0` (the frontend's minimal reduction would destroy the padding layout or make a given frame stale) |
| I-21 | `'cone'` | the norm of the projection onto the tangent cone at zero (a best rank-`r` approximation), computed at the subspace iteration's fixed point — **not** the supremum of frame projections; explicit only; computed iff `g0norm_newton is None`; vanishing raises; recorded `'cone'`; `cone_options` errors without it; no tolerance recommendation until S11 |
| I-22 | no frontend constructor | `fv_padded_frame` stays a backend primitive; the frontend surface is `padded_start` (razor: the frontend wraps what users call) |
| I-23 | no `pad_safe_svd` | the uniform twin unpacks real blocks, so no SVD ever sees padding |
| I-24 | contractions | twelve new `C`-group strings, letters with distinct `AXIS_SIZE` sizes; two existing strings reused; reshapes only for SVD inputs, `C` leading |
| I-25 | the O(1) claim | corrected in every doc that makes it, with the cone/Haar ratios (at least 15–76×: the dense `‖T3SVD_r G‖` is a lower bound on the cone value); CHANGELOG "Fixed (documentation)" |
| I-26 | `fv_zero_point_frame` | untouched; its equality with the `s = 0` hint-free constructor is a test (T2(a)), not a delegation — the least risky way to keep decision 2 |
| I-27 | unconverged levels | documented as neutral to worse (P11: up to 4 more Newton steps); the docs advise the padded start after converged levels; no automatic switch (that would be a numerical decision) |
| I-28 | the two layers | the uniform `padded_structures` builds the old frame **through the ragged path**, so both layers start from the same frame up to rounding (the default frames differ by a sign gauge); cross-layer tests stay gauge-invariant anyway **(weigh)** |
| I-29 | oversampled ranks | the elementwise maximum of `x0`'s ranks and `minimal(r + p, ρ + p)`, so `old ≤ target ≤ oversampled` always and `p = 0` is the target |
| I-30 | old ranks | must be ≤ `x0`'s and structurally minimal (structural): an old frame's slack directions are LAPACK completions that would be pinned as "old" **(weigh)** |
| I-31 | corewise geometries | no padded-start members: a corewise frame is the point itself; every new keyword on corewise is a structural error |
| I-32 | placement and imports — **judgment call for Nick** | two sound options. **(C, recommended)** a new topic module `backend/padded_frames.py` hosting both twins and their helpers, all imports downward — the house's recorded answer to a would-be cycle ("named for its operand, placed for its imports", `utv_operations.py:769-776`); costs a module-map line and two cross-module private helper uses (house precedent). **(A)** keep `fv_padded_frame` in `fv_operations` beside `fv_zero_point_frame` and import `tv_to_t3` inside the function — smallest change, but the backend's first function-level import of a library module, an upward edge a module-level scan would miss; if chosen, it is recorded as the one deliberate exception (a comment at the import, a line on the internals page, and G-acyclic's no-function-level-import assertion listing that edge). The cited `_bfit_pd` / `_jitted` imports are lazy / optional-dependency imports, not cycle breakers. Moving `tv_to_t3` below `fv_operations` is not an option (it calls `sharing`, which imports `fv_operations`) **(weigh)** |
| I-33 | version | 2026.4.0 (new public API; the house numbering increments the minor per release) |
| I-34 | the uniform padded layout | with `old_ranks`, a uniform `x0` must have **prefix** rank masks (structural, both modes): the old-rank slice reads the prefix, and gappy masks (legal after `+` / `×`) would make it read the wrong slots. Prefix masks imply that the old prefix lies inside the real slots; inclusion alone is not enough (the round-2 probe's `[1 0 0 0] ⊆ [1 0 1 0]` gathers the zero block). On the ragged layer the layout is a numerical precondition (no masks), covered by the safe-mode zero-padding check |

---

## 10. Risks and open questions

**Risks**

| # | risk | mitigation |
|---|---|---|
| R1 | From an **unconverged** old level the padded start is neutral to worse (P11: `f1/f0` worse on all six cases; Newton totals +3, 0, +4, −1, −1, −1): the old block is not a critical point, so the gradient's old-coupled content — which the two-sided steps discard — is where the progress is. Downstream ladders often end levels at the cap. | Opt-in; the docs advise it after converged levels only and the loop sketch passes it conditionally (§7); no automatic switch (I-27). The general recommendation waits for S11's ladders. |
| R2 | **Cost.** `padded_from` adds `passes` frame sweeps and transposes (4) per level; `'cone'` adds 11 per run, against 1 for `'zero'`. On the measured Darcy levels a level's CG did hundreds of Hessian-applies, so this is small — but it is measured at toy scale only. | Measure at scale in S11 (the "measure at scale" rule); `PaddedFrameOptions(passes=…)` lowers it. To reuse a cone value, pass `g0norm_newton=padded_start(..., old_ranks=None).gnorm` — only that `gnorm` is the cone value; a `padded_from` start's `gnorm` is the deflated warm-start gradient and must never be used as a reference. |
| R3 | **Memory at oversampled ranks.** The problem's `chunk_size='auto'` is resolved at the target ranks; the oversampled passes run at `r + p`, so a `probe_derivatives` transpose sized near the memory limit can exceed it. | Documented on `PaddedFrameOptions.oversample`; `oversample=0` removes it; S11 runs the uniform `probe_derivatives` Darcy cell on GPU. Re-resolving the chunk at the oversampled ranks is deferred (O-7). |
| R4 | **Eager jax passes.** Under `use_jit` the passes run eagerly on jax arrays (per-op dispatch, no compile). | Small problems pay dispatch, large ones are dominated by the transposes; S11 measures. Jitting the pass body is a later option (it is a pure function of the frame). |
| R5 | **LAPACK-dependent choices**: a degenerate hint (fewer directions than slots, or zero) and a tie at a cut (`σ_k = σ_{k+1}`) leave the choice to the Householder completion or to the SVD's basis of the tied subspace, which can differ across LAPACK builds and devices. | Always a valid frame (T1(a), T3(a)); deterministic per backend; the hint carries no information that separates the candidates, so any choice is as good. Stated in the docstrings. |
| R6 | **The cone value is an iteration output** — 0.99–1.04× the dense-frame value (P12), dependent on `passes`/`p`, and metric-dependent like every projection. Relative stops are comparable across levels only with fixed options. | Defaults fixed and documented; `cone_options` is explicit; the metric-invariant stop (the Newton decrement) stays an open question (O-4). |
| R7 | **The `K` lift multiplies the work by `|K|`** (one padded frame per hint), and the uniform twin loops over `K + C` on the host. | No library consumer passes `K`; the lift exists for completeness of the stacking contract (decision 5). |
| R8 | **Exact Hessian at a padded frame**: new slots aimed at the data make the first steps larger in exactly the directions the lift clips. | The trust region bounds them; T6(d) and T7(b) run exact mode from padded frames; S11 measures exact-mode ladders. No content is injected, so the downstream ε failure (first-step rejections) cannot recur by that mechanism. |
| R9 | **Two near-duplicate zero-tensor constructions** (`fv_zero_point_frame` and the `s = 0` hint-free `fv_padded_frame`) could drift. | T2(a) and T4(a) pin bitwise equality; the docstrings cross-reference. |
| R10 | **The in-block Gaussian keys** `[seed, i, c]` are a new reproducibility surface. | `RandomState`'s legacy stream is frozen (NEP 19), as for the Tucker draws; T2(c) pins determinism. |
| R11 | **Cross-layer agreement is to rounding, not bitwise**: the two layers' gradients differ at rounding, and a near-tie at a cut can separate their padded frames. | Stated in the docs (§7); every cross-layer test is gauge-invariant with rounding tolerances (§6). |

**Open questions** (none blocks the implementation)

| # | question | writer's position |
|---|---|---|
| O-1 | Should `'cone'` become the default reference on the manifold geometries (with a new default `gtol_rel`)? | Not in this release (decision 2). Revisit after S11's ladders on the three downstream cells. |
| O-2 | Should a continuation loop use `padded_from` by default (downstream, or a future canonical continuation function — T3Polynomial `dev/OPEN_QUESTION_canonical_rank_continuation_2026_11_02.md`, parked)? | Downstream decides per S11, given R1; the library keeps it opt-in. |
| O-3 | Oversampling as a fixed `+5` or relative to the rank (`⌈r/4⌉`)? | Fixed, as in randomized SVD practice; revisit with S11's high-rank levels. |
| O-4 | The metric-invariant stop (the Newton decrement) instead of any gradient-norm reference. | Out of scope; recorded. |
| O-5 | Vectorize `ufv_padded_frame` for uniform-rank stacks. | Deferred until a stacked consumer exists. |
| O-6 | A frontend `T3Frame.padded(...)`-style wrapper of the constructor. | Not now (I-22). |
| O-7 | Re-resolve `chunk_size` at the oversampled ranks (R3). | Deferred; needs the problem to rebuild its kind per structure. |
| O-8 | `start_frame` for `gradient_descent` (its first step is also a frame choice). | Not now: the brief scopes the seam to `newton_cg`; trivial to add later through the same `local_model(frame=)`. |
| O-9 | Which `gtol_rel` to recommend with `'cone'`. | None until S11 measures the cone against the library's own references (§3.7). |

---

## 11. Downstream (T3Polynomial)

The library change is what the downstream investigation asked for (T3Polynomial
`dev/t3toolbox-upstream-notes.md` §2026-11-02; `dev/zero_reference_cone_2026_11_02.md`). Acceptance runs in S11, in
the `t3toolbox` env with the library branch installed (`pip install .`; pure Python, no native build).

**Code changes downstream** (T3Polynomial, after the library lands; not part of this plan's library slices):

- `t3polynomial/fitting.py`: the continuation loop records the level's ranks before the `resize` at `:612` and,
  when that level converged, passes `padded_from=(old_tucker, old_tt)` to the next level's `_newton_fit` (`:569`;
  `_newton_fit` forwards `**kwargs`, `:176-245`) — opt-in through the per-level `newton_kwargs` hook, so existing
  scripts are unchanged. The sibling path at `:372`/`:381` likewise.
- The `g0norm_newton=None` guard (`:215`) stays; `gtol_rel` defaults stay (`1e-3` against `'zero'`); a cone ladder
  passes `g0_reference='cone', gtol_rel=3e-4` per level — the downstream ladder setting, measured against
  T3Polynomial's own coordinate-aligned zero reference at `1e-3`, not a library recommendation.
- `t3polynomial/cone.py` (the prototype) retires: `cone_reference` → `g0_reference='cone'` or
  `optimizers.padded_start(..., old_ranks=None).gnorm`; `tucker_cone_completion` → `padded_from` (now the TT
  completion too, δ-free); `cone_frame_zero_start` → `padded_start` + `start_frame`.

**Acceptance measurements** (recorded in `verification_records.md` and quoted in the CHANGELOG):

1. Darcy o01, level 1 restart (the §0 table): `padded_from` on ragged and `uniform_jax`, exact and GN. Expect the
   same Newton/CG counts on the two layers up to rounding effects (the 21-vs-44 split gone) and counts at or below
   W6's (20/315, 16/188) and near W3's ε-version (13/136).
2. The g05 ladder (Darcy, 6 levels) with `g0_reference='cone', gtol_rel=3e-4`, with and without `padded_from`
   (conditional on the previous level converging, and unconditional, to measure R1 at scale): Newton/CG totals,
   levels converged, final test p50 against the four rows of the downstream g05 table; and the library's own
   references (`'zero'`, `'warm_start'`) at their tolerances beside `'cone'` at 1e-3, 3e-4 and 1e-4 — the input to a
   tolerance recommendation (O-9).
3. e01 (16 exchangeable inputs) and the eigen cell: the same pair of runs, to check transfer beyond
   importance-ordered inputs.
4. The few-points-many-directions layouts Nick uses (10×100 laptop; 100×5000 on the GPU pod, `runpod` notes in the
   research repo) for the cost question R2–R4 — wall time per level with and without the padded start.

---

## 12. Pitfalls checklist

| pitfall | how this plan handles it | where |
|---|---|---|
| **Named contractions** (coordinator) | Every nontrivial contraction is a `contract(...)` call with the `C` group — the hint re-basis, the Tucker environment and its projection, the sweep's carry/projection/update, the down environments, the primitive's projection; twelve new strings with HISTORICAL `''`, two reused. Plain `xnp.linalg.qr/svd` only for factorizations; `concatenate`, `swapaxes`, `pad_or_truncate` where that is all that is needed. | §4.0–§4.10 |
| **Sharding safety** (coordinator) | No reshape inside contractions; reshapes only to form SVD/QR inputs (merging adjacent non-stack rank axes) and to fold factors back; `C` always leading; no stack axis moved or merged. `test_contractions_sharding.py` sweeps the new strings automatically. | §4.0, §4.10 |
| batching / stacking (C/W/K, frame-inner) | `C` vectorized in the ragged constructor and the driver; `W` never appears (the constructor sees no samples; the kinds keep their own `W` handling); a `K`-stacked hint is lifted to `K + C` outputs by broadcasting (frame-inner: `C` innermost), never reduced; optimizers unstacked. | §3.9, I-5 |
| ragged / uniform split | `ufv_padded_frame` unpacks, runs the ragged code, repacks at the template's padded sizes; uniform masks stay host `np`, supercores `xnp`; the oversampled uniform geometry is built for its own masks; the uniform old frame comes through the ragged path, so the layers start from the same frame up to rounding. | §3.3, §3.4, §4.8, I-28 |
| `pad_safe_svd` at masked SVDs | none needed: no SVD sees padded data (the twin unpacks); stated in the docstrings and the internals page. | §3.3, I-23 |
| numpy/jax dispatch | `xnp` from the array inputs (`tree_contains_jax`); host draws converted with `xnp.asarray(..., dtype)`; float sign literals cast to `Q`'s dtype; the result follows the template's dtype and the inputs' backend. | §3.2, §4.1, T10 |
| jittability | the constructor and primitives are traceable (static shapes and counts, host constants, no value-dependent branches); the uniform twin and the driver are eager by design (masks, Python loop); a start frame enters the CG as a local-model leaf, its uniform masks value-equal to the template's (no recompile). | §3.2, §3.6, T7(e), T10 |
| numerical vs structural | structural checks always (shapes, counts, stacks, option validity, minimal old ranks and frontend `x0`, mutual exclusion, corewise); numerical ones (zero padding on real slots, ORTH, same tensor, TIED) only in frontend safe mode, skipped under `unsafe()` and traces; the hint-frame condition is a caveat. | §3.8 |
| no numerical rank decision | counts are store differences; leading directions are taken in order and completed; no `σ` is compared with a tolerance anywhere. | §2.8, §4.1 |
| comment and signature style | one argument per line, aligned name · type · `#` shape contract, one return element per line with `len=d, elm_shape=C+(...)`. | §5 |
| backend/frontend razor | math in backend standalone functions; the frontend adds types, the minimal-rank and safe-mode checks, and the re-export; no frontend constructor (I-22). | §3.4, §5.7 |
| sharing | one tied Tucker factor per group from the concatenated normal content, the old block read from the group's first mode under TIED; hint-free draws keyed by `min(g)`; TT/down untied; `frame_has_minimal_ranks` stays a diagnostic. | §2.9, §4.5, T3(e), T13 |
| safe mode | as "numerical vs structural"; the backend never runs a numerical check. | §3.6 |
| naming (`docs/naming_conventions.md`) | `fv_` / `ufv_` prefixes for the frame constructions, hosted by the topic module `padded_frames` (named for the operand, placed for the imports; the module map's "outside the matrix" list gains it, §7); unprefixed `linalg` primitives (as `pad_safe_svd`); `require_` for a raising structural member; options are a frozen dataclass passed as a keyword (options are not parameters) | §3.2, §5, §7 |
| import acyclicity | the padded-frame code lives in a new module whose imports all point downward (option C of I-32); G-acyclic in S3: the backend import graph from every import node (AST scan, module- and function-level) is acyclic, and the backend has no function-level library import | §3.2, §6, I-32 |
| doctests | every public function has an Examples section; amended docs carry `>>>` blocks; doctest margins wide enough for both numpy generations; T14. | §5, §7 |
| CHANGELOG | a new section in the Keep-a-Changelog format of `[2026.3.0]`: Added + Fixed (documentation); no `dev/` citations in shipped text. | §7 |
| equivalence-contract obligations | bitwise real blocks on identical inputs (T4(a)), bitwise uniform old frame and hint-free start (T4(e)); gauge-invariant cross-layer comparisons of `padded_start` (T5(f)), the cone reference (T8(c)) and the padded-restart trajectory (T9(e)); the contract doc's new paragraph. | §6, §7 |
| bit-identical defaults | the default path executes the same statements; no new history field; `stats` keys only when used; T7(a); G-bit compares whole `stats` dicts. | §3.6, §6, I-2, I-18 |
| options are not parameters | `PaddedFrameOptions` is a solver option passed per call, never a geometry or kind field; constants are module-level defaults. | §5.5 |

---

## Appendix A. Probe protocol (to re-run the [P] claims)

The probes used the `t3toolbox` env (`/home/nick/miniconda3/envs/t3toolbox/bin/python`, `PYTHONPATH` at the repo)
on `main` @ `efe5e3d`. Continuation case set-up: `X_true = TuckerTensorTrain.randn(shape, new ranks)` normalized; `W`
unit-norm Gaussian probe rows per mode; `data = X_true.apply(ww)`; `X_old` fitted from `TuckerTensorTrain.randn(shape,
old ranks)` by `newton_cg(..., g0_reference='warm_start', gtol_rel=1e-10, max_newton=60)`; `X_pad =
X_old.resize(shape, new ranks)`. The gradient of a frame `F`: `geom.project(F, kind.transpose(r0, ww, F,
kind.precompute(F, ww)))` with `r0 = kind.point_forward(X_pad.data, ww) − data`. "First step": one GN step (CG to
`η = 0.5`, Armijo) at the given frame, `f1/f0`, then `newton_cg` from the stepped point to `‖g‖ ≤ 1e-9`, Newton
steps counted including the first. Passes and oversampling as §4.9. The cone probe: `G = −Σ_w data_w ⊗_i ww_i[w]`
dense, the dense-frame value at `dense_t3svd(G)`'s frame with zero carriers.

## Appendix B. The reference prototype (numpy, unstacked, unshared)

What produced P1–P14, with the inert correction branch removed (P10). The library version differs as §4 says:
`C`-stacked, `contract(...)` strings, the Tucker step in ambient coordinates with groups, the in-block fallback.

```python
def pinned_complement(P, S, k):                      # = linalg.pinned_completion (unstacked)
    n, s = P.shape; m = S.shape[1]
    if m < k: S = np.concatenate([S, np.zeros((n, k - m))], axis=1)
    Q, R = np.linalg.qr(np.concatenate([P, S[:, :k]], axis=1))
    sg = np.sign(np.diag(R)[s:s + k]); sg[sg == 0] = 1.0
    return Q[:, s:s + k] * sg

def pinned_leading(P, A, k):                         # = linalg.pinned_leading_columns (unstacked)
    U, _, _ = np.linalg.svd(A - P @ (P.T @ A), full_matrices=False)
    return pinned_complement(P, U[:, :k], k)

def embed(a, shape):
    out = np.zeros(shape); out[tuple(slice(0, n) for n in a.shape)] = a; return out

def stores(f):                                        # (nU, nD, rL, rR) of a frame
    U, O, L, R = f
    return ([u.shape[0] for u in U], [o.shape[1] for o in O],
            [l.shape[0] for l in L] + [L[-1].shape[2]], [r.shape[0] for r in R] + [R[-1].shape[2]])

def left_sweep(chainR, Om, B, oL, rL, nU, srL):      # §4.6: pinned left TT-SVD of the hint, two-sided
    d = len(chainR); L = []; Cv = np.ones((1, 1))
    for k in range(d - 1):
        Mb = np.einsum('xa,anb,mn->xmb', Cv, chainR[k], B[k])                       # (rL_k, nU_k, zr_{k+1})
        M2 = Mb.reshape(rL[k] * nU[k], -1)
        Mp = M2 - (M2 @ Om[k + 1].T) @ Om[k + 1]                                     # off X_old's right interface
        P = embed(oL[k], (rL[k], nU[k], srL[k + 1])).reshape(rL[k] * nU[k], srL[k + 1])
        Lk = np.concatenate([P, pinned_leading(P, Mp, rL[k + 1] - srL[k + 1])], axis=1)
        L.append(Lk.reshape(rL[k], nU[k], rL[k + 1]))
        Cv = Lk.T @ M2
    L.append(embed(oL[d - 1], (rL[d - 1], nU[d - 1], rL[d])))
    return L

def padded_frame(template, old, hint):                # design (d): §4.4-§4.7
    nU, nD, rL, rR = stores(template); oU, oO, oL, oR = old; snU, snD, srL, srR = stores(old)
    d = len(nU)
    Z = tv_to_t3(*hint)
    W, Ztt = t3_down_orthogonalize_tucker_cores(Z)
    ZR, ZL = tt_right_orthogonalize(Ztt), tt_left_orthogonalize(Ztt)
    A = [oU[i] @ W[i].T for i in range(d)]
    OmR = tt_zipper_right_to_left([np.einsum('anb,nm->amb', oR[i], A[i]) for i in range(d)], ZR)
    OmL = tt_zipper_left_to_right([np.einsum('anb,nm->amb', oL[i], A[i]) for i in range(d)], ZL)
    ZZ = tt_zipper_right_to_left(ZL, ZR)
    B = []
    for k in range(d):                                                               # §4.5 (w-coordinates)
        H = np.einsum('aib,bc->aic', ZL[k], ZZ[k + 1]); zl, w, zr = H.shape
        M = np.moveaxis(H, 1, 0).reshape(w, zl * zr)
        E = np.einsum('acb,ax,bz->cxz', oO[k], OmL[k], OmR[k + 1]).reshape(snD[k], zl * zr)
        M = M - (M @ E.T) @ E                                                         # two-sided: column side
        B.append(np.concatenate([A[k], pinned_leading(A[k].T, M, nU[k] - snU[k]).T], axis=0))
    U = [np.concatenate([oU[i], B[i][snU[i]:] @ W[i]], axis=0) for i in range(d)]   # old rows bitwise
    L = left_sweep(ZR, OmR, B, oL, rL, nU, srL)
    rev = lambda cs: [np.transpose(c, (2, 1, 0)) for c in cs[::-1]]
    R = rev(left_sweep(rev(ZL), OmL[::-1], B[::-1], rev(oR), rR[::-1], nU[::-1], srR[::-1]))
    G2 = [np.einsum('azb,uz->aub', Ztt[i], B[i]) for i in range(d)]
    LL, RR = tt_zipper_left_to_right(G2, L), tt_zipper_right_to_left(G2, R)
    O = []
    for i in range(d):                                                               # §4.7
        env = np.einsum('xa,xub,bc->auc', LL[i], G2[i], RR[i + 1])
        Anew = np.transpose(env[:, snU[i]:, :], (0, 2, 1)).reshape(rL[i] * rR[i + 1], nU[i] - snU[i])
        P = np.transpose(embed(oO[i], (rL[i], snD[i], rR[i + 1])), (0, 2, 1)).reshape(rL[i] * rR[i + 1], snD[i])
        Oi = np.concatenate([P, pinned_leading(P, Anew, nD[i] - snD[i])], axis=1)
        O.append(np.transpose(Oi.reshape(rL[i], rR[i + 1], nD[i]), (0, 2, 1)))
    return tuple(U), tuple(O), tuple(L), tuple(R)

def padded_start(grad, template, os_template, old, passes, hint_free_start):        # §4.9
    F = hint_free_start(os_template, old)
    for j in range(passes):
        F = padded_frame(template if j == passes - 1 else os_template, old, (F, grad(F)))
    return F
```

(The prototype's `U` for the measured runs was `B[i] @ W[i]`, equal to the line above to rounding; `old` for the
zero tensor is the empty frame of §4.2.)
