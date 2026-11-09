# Implementation plan — the padded frame, its subspace iteration, and the cone reference

> **STATUS: round 12, part 1 (writer; the fixes of the critic's review `dev/padded_frame_cone_plan_review_r8.md` that do
> not wait on Nick: MINOR 1 and N1–N4. MINOR 2 and the twin set (J1) are held for his decision. Round 11 was the
> frontend/backend split, Nick's ruling after he approved rounds 9–10. Round 10 fixed the critic's round-6 review
> `dev/padded_frame_cone_plan_review_r6.md` of the round-9 backend-style amendment, with Nick's decisions on I-32,
> `dtype` and the `with_ranks` test; round 8 closed the critic's round-5 approval `…_review_r5.md`; earlier reviews
> `…_review_r1.md` … `…_review_r4.md`; the critic's notes `…_plan_q_old_point.md`, `…_plan_q_vectorize.md`),
> 2026-11-02. Implementation paused at the slice reached (see `dev/padded_frame_slices/`); it resumes with the
> corrective slice S2′ (§8).** A writer/critic plan in the
> curvature-arc format (`dev/archive/riemannian_newton_curvature_impl_plan_2026-10-29_shipped.md`): the
> critic reviews this file, the writer revises it in place and records each finding in the CHANGES table,
> until the critic agrees; Nick and the orchestrating session then adjudicate the decisions table (§9).
>
> - **Repos:** T3Toolbox @ `efe5e3d` (`main`, 2026.3.0 shipped); downstream T3Polynomial @ `66f0af1`.
> - **Amends:** `dev/archive/zero_reference_g0_plan_2026-10-27_implemented.md` (the zero reference) and the
>   2026.3.0 documentation of it ("a convention good to an O(1) factor" — false on importance-ordered bases).
> - **Motivation and measurements:** T3Polynomial `dev/zero_reference_cone_2026_11_02.md`,
>   `t3polynomial/cone.py`, `dev/t3toolbox-upstream-notes.md` §2026-11-02.
> - **[P] claims** are the writer's read-only probes (the facts table P1–P17); every one was run against `main` @ `efe5e3d` with
>   the `t3toolbox` env. The probe scripts live in the writer's session scratchpad (ephemeral); Appendix A states
>   each probe precisely enough to re-run, and Appendix B is the reference prototype they share.

**Target.** An additive feature (Nick's decision 2: **no existing default changes**):

1. a **structural, δ-free constructor of the orthonormal frame at a zero-padded rank increase**,
   `fv_padded_frame` (the uniform layer converts explicitly at its one call site — no twin): the old block embedded, the new slots
   completed either hint-free (the generalization of `fv_zero_point_frame`, which is its `s = 0` instance — the
   coordinate cores exactly, the Haar factors to rounding) or from a **hint** — any T3 — by a *pinned T3-SVD*:
   leading directions of the hint's normal content, the old block pinned, every count structural;
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
| 5 | **Nick 1 — the hint is a T3, not a tangent** (the construction reads only the cores of `tv_to_t3`'s output) | `fv_padded_frame(frame, old_frame, hint, groups, seed)` with `hint` a ragged T3 `.data` (template's mode sizes, any ranks, stack `C`; structural checks on consistency, mode sizes and stack only); `ufv_padded_frame` with a uniform T3 `.data` unpacked by `ut3_to_t3` (gappy masks fine, e.g. `utv_to_ut3`'s `rL + rR` bond padding); geometry `padded_frame(frame, old_frame, hint, seed)` plus a new member `tangent_to_t3` (chosen over folding: one hint type, a reusable layer fact, a T3-free driver); the driver embeds each pass's gradient; the `K` lift deleted (one hint per frame; combining hints is the caller's reduction); §2.3's caveat restated for a T3 with a verified exactness recipe (`hint + X_old`, P16) and the cost scaling; the hint's tails squashed (a T3 sum has boundary bonds 2 — found by the new probe, P16); every shape downstream of `Z` in the hint's own ranks; a new test with an external T3 hint (T3(j)), the gradient tests kept through the new call shape; I-32 re-examined — the upward import is gone, placement is now a topical choice, recommended beside the zero-point frames | Target, §1, §2.3, §2.6, §3.1–§3.5, §3.8–§3.9, §4.2, §4.4, §4.5, §4.8–§4.9, §5.2–§5.4, §6 T3/T4/T10, §7, §8, I-5, I-15, I-32, I-35, R7, O-6, P15–P16, App. B |
| 5 | **Nick 2 — a standalone math note after the code** | slice S12 (after the docs S10 and the downstream acceptance S11; release now S13) writes `docs/padded_frame_cone.tex` + `.pdf`, mirroring `exact_hessian_two_hole.tex` (committed source and PDF, `latexmk`, the `\code` macro, a resolving-commit header, linked by markdown from the user and contributor docs). §7 gives its 12-section outline, the eight algorithm boxes with the implementing functions and files, the evidence it collects, its links, and its gate: a clean build plus `tests/test_math_note_references.py` (every cited function resolves, every quoted subscripts string is in the library vocabulary) | §1, §7, §8, I-36 |
| 6 | **Nick 1 — bitwise vs tolerance** | §6's preamble states the rule: bitwise only for structural facts (old block copy, zero carriers and padding, masks, coordinate completions, the default-path claims G-bit / T6(a) / T7(a) — the latter two classed with G-bit as invariance-of-the-default-path claims); stated tolerances (`1e-14`, `1e-13`, `1e-14`, the safety rtol) for everything computed, including the `s = 0` identity's Haar factors and the uniform twin's real blocks; frame claims through assembled projectors plus a dense `P_{T_F} Y`. Every "bit for bit" relabelled | Target, §1, §2.5, §3.2, §3.3, §3.6, §4.3, §4.8, §5.2–§5.3, §6, I-26, I-38, R9, P2 |
| 6 | **Nick 2 — frame equivalence; revert I-28** | each layer builds its old frame natively; cross-layer agreement is "the same frame up to gauge and rounding", tested gauge-invariantly; scoped to subspaces (`frames_equal` for coordinates untouched). **Consequence, flagged:** the in-block seeds had to change from a seeded Gaussian to coordinates. A Gaussian in the core's coordinates is not covariant under the layers' sign gauge — new probe P17: a projector difference of 1.4, against 0.0 for coordinate seeds — so the claim would have been false for the hint-free start whenever the in-block completion runs. I-11 is updated, and T2(f) tests the covariance | §3.3, §3.4, §2.5, §4.3, §6 T2(f)/T4(e)/T5(f), §7 (`uniform_equivalence_contract.md`), I-11, I-28, R10, P17 |
| 6 | **Nick 3 — `padded_from` takes the old point** (the critic's note, section by section) | `newton_cg(x0, padded_from=x_old)`, `padded_start(problem, x_cores, x_old, …)`, `padded_structures(x_cores, x_old, working_ranks)`; the uniform `x_old` is its full `.data` with its own masks (gappy fine). I-34 and the prefix-layout requirement are removed. The zero-padding check becomes `x0.allclose(x_old)` at `effective_rtol` (the difference's norm through orthogonalization, never the inner-product formula). I-30 becomes the frontend's reduction of `x_old` plus a backend caveat; I-20 keeps its `start_frame` half. New guard: `x_old`'s frame stores ≤ `x0`'s (plus layer / shape / stack / padded `N`). TIED moves to `x_old`. T5(g), T7(b)/(g), T9(c)/(d) and T13 are updated | §2.3, §3.1–§3.8, §5.4–§5.7, §6, §7, I-20, I-30, I-34, I-37 |
| 6 | **Nick 4 — acceptance rewritten** | S11 runs on the laptop, autonomously: restart arms on Darcy o01 level 1 and e01 rank 7, both layers, both Hessians, `padded_from` vs the default; one Darcy ladder pair, `'zero'` 1e-3 vs `'cone'` 1e-4 / 3e-4. Runpod is manual and after the release: one 20 × 1000 ladder pair with per-level pass costs. 100 × 5000 is deferred. The release is decoupled from the GPU run. Each measurement names the decision it feeds. The downstream code changes are T3Polynomial's business, not slices | §8, §11, I-39, R1–R4 |
| 6 | **Nick 5 — the working ranks as a pure rank function** | `ranks.oversampled_ranks(shape, tucker, tt, p, sharing)` (+ `TuckerTensorTrain.oversampled_ranks` beside `continuation_ranks`) is the only home of the rule; `padded_start` / `padded_structures` take `working_ranks=None` (= the target) and compute no ranks; the convenience paths fill it through a geometry member `oversampled_ranks` with `p = 5`; `oversample` left the options; R3 renamed (memory `chunk_size` vs rank `p`) | §1, §2.6, §3.4, §3.5, §4.9, §5.4–§5.5, §6 T5(h), I-29, R3 |
| 6 | **Nick 6 — vectorization dropped** | the uniform construction (since item 7 the geometry member, not a twin) stays a host loop for all stacks by decision; O-5 → the "Rejected" ledger with the critic's reasons (250–400 masked lines vs ~50, readability, slower on CPU, diverging degenerate-hint completions) and the grouping-by-structure note | §3.9, §7 (`deferred_and_rejected.md`, the internals page), I-5, O-5 |
| 6 | **Nick 7 (addendum) — no `ufv_padded_frame` twin** | §3.3 and §5.3 rewritten. `UniformManifoldGeometryOps.padded_frame` converts explicitly: it unpacks with `ut3frame_to_t3frame` / `ut3_to_t3`, calls `fv_padded_frame`, and repacks with `t3frame_to_ut3frame` at the template's padded sizes and prefix masks, in a host loop, with a one-line docstring. `fv_padded_frame`'s docstring gives the three-call recipe for raw uniform `.data`. T4 and T10's uniform rows target the member: real blocks within `1e-14`, masks equal the template's, garbage never read. The ledger gets the exception to the twin symmetry and the `ufv_zero_point_frame` consistency question, and the naming doc's "Intended asymmetries" gets a bullet. No pushback: the razor holds (the math is standalone, the glue a callable backend method plus a documented recipe), and the equivalence tests keep their content | §1, §3.1–§3.4, §3.8–§3.9, §4.8, §5.2–§5.3, §6, §7, §8 (S4/S5), I-5, I-23, I-32, I-41, §12 |
| 6 | **Nick's decisions** | I-13 4 / 10 / 5 settled. I-32: `fv_operations` / `ufv_operations`, helpers nearby. I-3: **plain keywords** (`padded_passes`, `padded_seed`, `cone_passes`, `cone_seed`; `passes`, `seed`, `working_ranks` on `padded_start`), `PaddedFrameOptions` removed, `PaddedStart` kept. O-1: non-breaking, stated in the CHANGELOG and release notes; G-nonbreaking added; the frontend forwards a new keyword only when set, so the pre-existing forwarding test passes unmodified. O-2: T3Polynomial adopts `padded_from`, and T3Toolbox's continuation examples are unchanged. O-3 fixed oversampling. O-4 closed. O-6 not now. O-7 deferred. O-8: a `dev/` note plus a ledger entry. O-9: `gtol_rel=1e-4` with `'cone'`, with the ≈ 3e-4-of-the-old-reference note | §3.6–§3.7, §5.5, §5.7, §6, §7, §8, §9, §10 |
| 7 | **A** the six `newton_cg` keywords were inserted mid-signature, which shifts later positional parameters (contradicting "no breaking changes") | appended **keyword-only** after the last parameter (backend: after `callback`; frontend: after `tr_radius0`) behind a `*`; `gradient_descent`'s two likewise; stated in the CHANGELOG; the mirror test is unaffected | §1, §3.6, §5.5, §5.7, §7 (CHANGELOG), I-40 |
| 7 | **B** the shipped `1e-4` advice compared with "the pre-2026.3.0 `1e-3` regime" (the library's old stop was against the warm start) | user docs state `gtol_rel=1e-4` and point to the verification records, with no comparison; the comparison stays in §3.7 and O-9, labelled as T3Polynomial's reference | §3.7, §7 (fitting §5, release notes), O-9 |
| 7 | **C** the exceptions to cross-layer agreement were listed five different ways, and "which the first hinted pass replaces" overstated | one canonical list E1–E4 in §3.3 (near-tie; degenerate hint; old frames differing by more than signs; an in-block coordinate seed in the old span), referenced by R5 and R10, quoted in the contract-doc paragraph and the internals page, pointed to from fitting §4.11 and `rank_continuation.md`; "whose effect the hinted passes damp but, at the default four, do not remove"; the rarity of E4 cited from the critic's probe | §3.3, §4.3, §7, R5, R10 |
| 7 | **n1–n7** | n1: "each layer builds its **old** frame natively" in the three shipped sentences. n2: the backend's equal-padded-`N` check is dropped (the frontend keeps it for `allclose`), and the pre-existing `UniformTuckerTensorTrain.allclose` docstring mismatch becomes a separate ledger item. n3: G-nonbreaking also runs main's `tests/backend/test_*.py`. n4: the `working_ranks` example covers a uniform `x0` (`ranks.oversampled_ranks`). n5: T4(a) gains a non-minimal `x_old` / template row. n6: a code comment at the conditional forwarding. n7: a §3.8 caveat row for E4 | §3.4, §3.6, §3.8, §5.4, §5.7, §6, §7, I-37 |
| 8 | **n8–n10** (review r5, approved) | n8: slice S5 no longer lists the dropped padded-`N` guard. n9: §3.8's frontend structural row gains the uniform padded-`N` check that the frontend's same-tensor `allclose` needs. n10: §1, the release notes and the pitfalls table now say "the pre-existing tests (frontend and backend) pass unmodified" | §1, §3.8, §7, §8, §12 |
| 9 | **The backend style** (Nick: a style problem paused the implementation; option C — fix the construction's boundary first) | the gold standard studied and traced (`dev/padded_frame_slices/style_notes_writer.md`). `fv_padded_frame(shape, tucker_ranks, tt_ranks, old_frame, hint, groups, seed, stack_shape, dtype, use_jax)`: the structure as three int sequences; the stores derived inside by `ranks.compute_orthogonal_representation_ranks`; array shapes read inline; inline integer checks (no dict helper, no `_frame_stores`); dtype and backend from the arrays, or the array-free-constructor convention; the frame tuple returned; no template frame | §3.2, §5.2, §4.0, §4.2 |
| 9 | **Integer structures in the members** | `padded_structures` returns `(shape, target ranks, working ranks)` — integers only, never a frame or an orthogonalized zero tensor. `padded_frame(shape, tucker, tt, x_old, hint, seed, …)` builds the old frame from the old point itself. A new member, `with_ranks(tucker, tt)` (ragged: `self`), builds the uniform geometry at other ranks from ints and masks (`ut3_make_masks`, `ufv_make_frame_masks`, `ufv_variation_masks`). The uniform conversion repacks with masks built from the integer stores | §3.1, §3.3, §3.4, §4.8, §5.4 |
| 9 | **`fv_zero_point_frame`** | unchanged this release; its template pattern is ruled wrong and goes to a later audit arc (ledger, I-26); not a precedent | §3.2, §7, I-26 |
| 9 | **The style rule in §12** | the rule verbatim, the gold standard as the only allowable precedent, "precedent outside the gold standard is not justification", and a pitfalls row | §12, I-42 |
| 9 | **Flagged: I-32 reopened** | deriving the stores inside needs `ranks`, and `fv_operations → ranks → sharing → fv_operations` is a cycle. Options: (A) a new module `backend/padded_frames.py` (recommended); (B) a function-level import (contradicts G-acyclic); (C) the recurrence inline (duplicates). For Nick | §3.2, §5.2, I-32 |
| 10 | **MINOR 1** (review r6): the template survived in normative text | every trace removed; the structure is `(shape, tucker_ranks, tt_ranks)` plus the stores `ranks.compute_orthogonal_representation_ranks` derives. §4.2 builds the empty old frame from integers; §5.2's raw-uniform recipe builds the old frame natively and takes integer ranks; T4(a) calls the member with `x_old=None`; T4(b) compares the masks with the native sweep's and the geometry's `var_masks`; the §6 preamble, §3.6 and the three pitfalls rows name the geometry's masks and padded sizes. Also swept: §2, §3.3 step 1, §3.8, §3.9, §4.0 (the reading rule replaced by a structure bullet), §4.9, T3(d)/(g), I-35 and Appendix B (the prototype takes the target stores as integers; the math is unchanged) | §2, §3.3, §3.6, §3.8, §3.9, §4.0, §4.2, §4.9, §5.2, §6, §12, I-35, App. B |
| 10 | **I-32 settled: option A** (Nick; overrides the critic's option D) | `fv_padded_frame` and its private helpers go in a new module, `backend/padded_frames.py`. Its imports all point downward (`ranks`; `fv_operations` for `ZERO_FRAME_SEED`, `_haar_rows`, `_identity_prefix`; …), and only `geometry` imports it (AST check, acyclic). The justification is the import graph alone; the `utv_operations` citation is dropped (r6 NIT). `compute_orthogonal_representation_ranks` stays in `ranks`. D is recorded as not taken; B and C are rejected. A new ledger entry records `ranks`' dependence on `sharing` as a code smell for the later drift-fix arc; nothing changes now | §1, §3.1, §3.2, §5.2, §7, §12, I-32 |
| 10 | **Slice S2′** (the handoff's corrective slice) | new row. S2′ creates `backend/padded_frames.py` and moves the S2/S3 code into it with the round-9 signature. `fv_operations` returns to its committed pre-arc form (`efe5e3d`); the new module imports its `_haar_rows` / `_identity_prefix`. S4's uncommitted member is brought to the integer form. S3's oracle tests carry over | §8 |
| 10 | **J1: no `dtype`** (Nick) | dropped from `fv_padded_frame` and the geometry member. The dtype follows the arrays; an array-free call gives the backend's default float (float64 under numpy), as `t3_zeros`. There is no cast; §3.5 states the one consequence: the cone's passes run in float64 on float32 numpy data | §3.2, §3.4, §3.5, §4.2, §5.2, §5.4, §12, I-43 |
| 10 | **J2: §12's scope** (the maintainer's relaxation) | the rule binds every backend function of the feature, including everything the cone path calls. In the driver and geometry (optimizer-layer) code it is the goal: that layer passes points | §12, I-44 |
| 10 | **J3: `with_ranks`** (Nick) | §3.4 says why its hand-built masks are safe. New test T5(i): `with_ranks` of a geometry's own ranks reproduces its `masks` and `var_masks` exactly. `with_ranks` is added to §1's inventory and slice S5, and I-16 now names it as the working geometry's source (round 9 had changed §3.4 and §5.4 only) | §1, §3.4, §6, §8, I-16, I-45 |
| 10 | **NIT** (review r6): §3.1 | the cone path is drawn, with the array-free call's `stack_shape` and `use_jax` taken from `x0`. The stack needed a source in a T3-free driver: `padded_structures` now also returns `x_cores`' stack `C` (integers) | §3.1, §3.4, §5.4 |
| 11 | **The frontend/backend split** (Nick) | every new public backend function or keyword gets a frontend twin that validates, calls the backend and rewraps, and carries the description, the doctests and the validation. The backend keeps a one-line docstring, its shape comments and the terse guards that the gold standard keeps (`tucker_change_core_shapes`, `t3_add`, `t3_to_ut3`, `t3svd`). Stated in §12 with Nick's words, in §5's preamble, and as I-46 | §5, §12, I-46 |
| 11 | **Nick's correction: doctests without a twin stay** | a backend function without a twin keeps its docstring, doctests and checks; nothing is deleted or converted to a test. This replaces "its doctests become tests". The linalg pair keep their S1 doctests | §5.1, §5.8, §12, T14 |
| 11 | **The twin map** (new §5.8) | twins: `T3Frame.padded` (supersedes O-6 / I-22); `padded_frame(x0, x_old)` on `ManifoldGeometry`, `UniformManifoldGeometry` and `SharedGeometry`; the existing `T3Tangent.to_t3` / `UT3Tangent.to_ut3`; `TuckerTensorTrain.oversampled_ranks`; the frontend `padded_start`; `frame=` on the six `fitting` model factories; the frontend `newton_cg` / `gradient_descent` keywords. No twin, each with its reason: the linalg pair, four driver members, `UT3Frame`, private pieces and constants. The new signatures and what each twin checks are in §5.8 | §5.8, I-22, I-47, O-6 |
| 11 | **Validation moved** | §3.2's checks split into three kept guards and the twin's checks. `padded_structures`' checks move to the frontend callers through one `_require_padded_pair` helper per layer. T3(g) and T5(g) now test the kept guards; the moved checks are tested in the new T15 | §3.2, §3.4, §5.4, §5.8, §6 |
| 11 | **Doctests and descriptions moved** | `fv_padded_frame` → `T3Frame.padded`; the uniform member's description and the raw-`.data` recipe → `UniformManifoldGeometry.padded_frame` (I-41); `ranks.oversampled_ranks` → `TuckerTensorTrain.oversampled_ranks`; the backend `padded_start` → the frontend one; `Problem.local_model(frame=)` → the factories' `frame=`. T14 lists the moves | §5.2–§5.5, §5.7, T14, I-41 |
| 11 | **Slices** | S2′ grows: `T3Frame.padded`, and `TuckerTensorTrain.oversampled_ranks` (moved from S9). S4 gains the uniform geometry twin, S5 the ragged and shared ones, and S9 the factories' `frame=`. The frontend `padded_start` stays in S9 and carries the backend's description from there | §8 |
| 11 | **Files and docs; two inconsistencies fixed** | §1's table gains the frontend sites, the `fitting` factories and T15's files. §7's api_reference and CHANGELOG list the twins. The ledger drops "a frontend wrapper of the constructor". Two inconsistencies are fixed: the member lists in §1, the CHANGELOG and the `fitting_and_optimization.md` text gain `with_ranks` ("six", a round-9 omission), and §5.2 no longer keeps S2's "See also" line in `fv_zero_point_frame` (round 10 returns `fv_operations` to its pre-arc form). A stale `ranks.py` line range in §1 is corrected | §1, §5.2, §7 |
| 12 | **MINOR 1** (review r8): the uniform rank twin | `UniformTuckerTensorTrain.oversampled_ranks(p, sharing)` mirrors `minimal_ranks`: per element, host numpy, one doctest. `ranks.oversampled_ranks` accepts per-element int arrays, as `compute_minimal_ranks` does; S2′ widens the committed S1 version. Frontend `padded_start`'s `working_ranks` becomes `x0.oversampled_ranks(5)` on either layer. Updated: T5(h), T14 and T15 | §1, §5.4, §5.7, §5.8, §6, §7, §8, I-29, I-47 |
| 12 | **N1**: the reason for no `UT3Frame.padded` | a uniform frame is useful only at the padded sizes and masks of the point it serves, and only a point or a geometry supplies them. A standalone `UT3Frame.padded` would take five padded sizes or default them, a template in disguise | §5.8, I-47 |
| 12 | **N2**: S2′ edits committed code only | S4's uncommitted work is stashed while S2′'s gate runs. S4 brings its own member to the integer form when it lands | §8 |
| 12 | **N3**: the optional groups guard, taken | a fourth kept guard in `fv_padded_frame`: a sharing group whose members differ in `(Tucker rank, size)`, the analog of `t3svd`'s `validate_sharing`. "Three kept guards" becomes four throughout | §3.2, §5.2, §5.8, §8, §12, I-22, T3(g) |
| 12 | **N4**: where the raw-`.data` recipe lives | it is backend-user documentation. It lives in the internals page ("The uniform conversion") and is cited by `UniformManifoldGeometryOps.padded_frame`'s docstring; any frontend twin links to it. This is independent of the pending twin-set decision | §3.3, §5.2, §5.3, §5.8, §7, I-41 |
| 12 | **Held** for Nick's decision (part 2) | MINOR 2, the shared twin's tying, and J1: whether to drop the three geometry twins and defer the factories' `frame=` | — |

---

## Probed facts this plan rests on (P1–P17)

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
| P2 | at `s = 0` the hint-free construction **is** `fv_zero_point_frame` | bitwise equal in one process, unstacked (the tests compare the Haar factors at `1e-14`, §6) (the TT completion must enumerate each core in its own unfolding order — `R` in `(n, b)` order; a reversal-based `R` fails, measured) |
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
| P15 | the T3-hint call shape changes nothing for gradient hints, and an external T3 hint gives a valid padded frame (round 5) | the prototype taking the embedded gradient `tv_to_t3(F, g)` as its hint equals the round-1 prototype to 1.2e-15 (c13). A random external T3 hint (Tucker (5, 6, 5, 6), TT (1, 4, 6, 4, 1)) gives orthogonality 8.9e-16 and the base point `X_old` to 4.0e-16 relative (c13). The prototype's `w`-coordinate Tucker step needs `w_i ≥ nU_i` (a lower-rank external hint fails there); the library's ambient step (§4.5) has no such limit |
| P16 | adding `X_old` to an arbitrary hint makes every step exact, once the sum's boundary bonds are squashed (round 5) | matrix case (`d = 2`, `s = 1 → r = 3`, a random rank-6 T3 hint `H`): with `H + X_old`, tails squashed by `tt_operations.tt_squash_tails`, the new slots equal the top-2 singular subspaces of `P_u^⊥ H P_v^⊥` to 2.6e-15. With the raw `H` they differ (projector distances 0.45 / 1.1 — the caveat case, still a valid frame). Unsquashed, the T3 sum's boundary bonds 2 broke the zippers' ones-boundary (an overlap Gram of 1.59 instead of 1) — hence the squash in §4.4 |
| P17 | the in-block completion must use coordinate seeds for the hint-free start to be gauge-covariant (round 6) | the old block filling a `(2 × 3)` unfolding, bond 2 grown from 2 to 3 (a choice among four in-block directions), under a sign gauge of the old frame (one left-basis vector and one Tucker row flipped): the new bond-2 interface projector moves by **1.4** with seeded-Gaussian seeds and by **0.0** with coordinate seeds. A sign flip of an old bond vector alone leaves outside-first starts and 4-pass padded starts unchanged: gradient norms equal to 4e-15 (c11, c13) |

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
library's pre-2026.3.0 warm-start reference. The library recommends `gtol_rel=1e-4` with `'cone'` (O-9), stated without
a comparison; S11 records its cost (§3.7).

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
| `backend/padded_frames.py` (**new**; I-32, Nick, round 10) | `fv_padded_frame(shape, tucker_ranks, tt_ranks, …)` (public, in the module's `__all__`) and its private helpers `_empty_old_frame`, `_embed`, `_outside_first`, `_hint_coordinates`, `_tucker_completion`, `_pinned_left_sweep`, `_down_completion` (§4.2–§4.7). Imports, all downward: `ranks` (the store recurrence), `fv_operations` (`ZERO_FRAME_SEED`, `_haar_rows`, `_identity_prefix`), `t3_orthogonalization`, `tt_orthogonalization`, `tt_operations`, `contractions`, `linalg`, `common`. Only `geometry` imports it | new file |
| `backend/fv_operations.py` | — | **unchanged** from `efe5e3d` (S2′ returns it to that form); the new module imports its `ZERO_FRAME_SEED`, `_haar_rows` and `_identity_prefix` |
| `backend/ufv_operations.py` | — | **unchanged**: no uniform twin (item 7); the uniform conversion lives in `UniformManifoldGeometryOps.padded_frame` |
| `backend/geometry.py` | six optional members on the two **manifold** geometry classes: `tangent_to_t3`, `padded_frame` (the old frame built natively from `x_old`; the uniform one from its own `.data` and masks), `padded_structures` (integers only), `with_ranks` (ragged: `self`; uniform: from ints and masks), `oversampled_ranks` (delegates to the rank function), `require_frame_structure` (§3.4); new imports `ranks`, `padded_frames`, `ut3_conversions` | `ManifoldGeometryOps` after `zero_point_frame` (`:326-333`); `UniformManifoldGeometryOps` after `:706-713`; imports `:58-72`; the corewise classes (`:392`, `:774`) unchanged |
| `backend/ranks.py` | `oversampled_ranks(shape, tucker_ranks, tt_ranks, p, sharing=None)` — the one home of the working-rank rule (§3.5); round 11: a one-line docstring, its doctests moved to the twin; round 12: per-element int arrays accepted, as `compute_minimal_ranks` | beside `compute_minimal_ranks` (`:51`); `__all__` |
| `backend/optimizers.py` | `PaddedStart`, `padded_start`, the constants `PADDED_START_PASSES` / `CONE_REFERENCE_PASSES` / `PADDED_FRAME_OVERSAMPLE`, `_cone_reference_gnorm` (§3.5, §3.7) | after `least_squares_problem` (`:434-444`); `__all__` `:46-59` |
| `t3toolbox/optimizers.py` | frontend `padded_start` (§5.7): the twin of the backend `padded_start`, with its description and doctest | after `least_squares_problem` (`:197-330`); `__all__` `:57-66` |
| `t3toolbox/tucker_tensor_train.py` | `TuckerTensorTrain.oversampled_ranks(p, sharing=None)` (§5.4): the twin of `ranks.oversampled_ranks`, with its docstring and doctests | beside `continuation_ranks` (`:4398`) |
| `t3toolbox/uniform_tucker_tensor_train.py` | `UniformTuckerTensorTrain.oversampled_ranks(p, sharing=None)` (§5.4, round 12): the uniform twin, per element | beside `minimal_ranks` (`:713`) |
| `t3toolbox/frame_variations_format.py` | `T3Frame.padded(...)`: the twin of `fv_padded_frame`, with its description, doctests and checks (§5.8) | beside `random_orthogonal` (`:595-615`) |
| `t3toolbox/manifold.py`, `t3toolbox/uniform_manifold.py`, `t3toolbox/shared_geometry.py` | `padded_frame(x0, x_old, hint, seed)` on `ManifoldGeometry`, `UniformManifoldGeometry` and `SharedGeometry`: the twins of the geometry member (§5.8); the private `_require_padded_pair` (ragged and uniform) | after each class's `frame` (`manifold.py:1294-1300`, `uniform_manifold.py:1068-1074`, `shared_geometry.py:171-198`) |
| `docs/contributor/padded_frame_internals.md` | the design record (§7) | new; `docs/contributor_guide.rst` toctree |
| `tests/backend/test_padded_frame.py`, `tests/backend/test_padded_start.py` | T1–T6 (§6) | new |
| `docs/padded_frame_cone.tex` + `.pdf`, `tests/test_math_note_references.py` | the math note and its reference gate (§7, slice S12) | new |

**Changed code** (every change additive; behaviour without the new keywords — `stats` included — is bit-identical, T7(a), G-bit; the pre-existing tests (frontend and backend) pass unmodified, G-nonbreaking)

| file:line | today | change |
|---|---|---|
| `backend/optimizers.py:65-131` | `Geometry` protocol docstring: one optional member (`zero_point_frame`) | documents the six new optional members (§3.4) |
| `backend/optimizers.py:346-416` | `Problem.local_model(x_cores, sample, data, hessian, curvature_damping)`; `:405` `frame = self.geom.frame(x_cores)` | new keyword `frame=None` (preconditions ORTH and same tensor, documented); when given, `geom.require_frame_structure(x_cores, frame)` then use it (§3.6) |
| `backend/optimizers.py:516-532` | `_prepare_jit_inputs(use_jit, x0, problem)` | unchanged; a sibling `_prepare_jit_frame(use_jit, frame)` converts a given start frame's dynamic leaves (`common.partition_static` / `rebuild_static`, `:698-743`; masks and `shape` stay host) |
| `backend/optimizers.py:604` | `_G0_REFERENCES = ('zero', 'warm_start')` | `+ 'cone'`; the message in `_check_g0_reference` (`:607-618`) names it |
| `backend/optimizers.py:650-699` | `_resolve_g0_reference(problem, frame, x0, g0_reference, compute, who, hint)` | `cone_passes` / `cone_seed` arguments and a `'cone'` branch (§3.7); `'zero'` / `None` / `'warm_start'` paths byte-identical |
| `backend/optimizers.py:702-763` | `gradient_descent(..., g0_reference)`; resolution `:742-745` | `+ cone_passes=None, cone_seed=None`, appended keyword-only, passed to the resolution |
| `backend/optimizers.py:1123-1422` | `newton_cg(problem, x0, ...)`; iteration-0 model `:1299`; reference `:1307-1319`; the returned `stats` `:1421` | `+ start_frame`, `padded_from` (the old point), `padded_passes`, `padded_seed`, `cone_passes`, `cone_seed`, **appended keyword-only after `callback`** (§3.6); iteration 0 builds `problem.local_model(x0, frame=start)` when a start frame is in play; `stats` gains `'start_frame_source'` (and `'padded_start_gnorms'`) **only on such runs**; `NewtonInfo` (`:958-1028`) and the history rows unchanged |
| `backend/optimizer_display.py:121` | `_REFERENCE_NAMES = {'zero': 'zero tensor', 'warm_start': ..., 'override': ...}` | `+ 'cone': 'tangent cone'` (the converged line reads `g₀ = tangent cone`) |
| `t3toolbox/optimizers.py:420-611` | frontend `newton_cg` (explicit keywords, `:596-606` forwards) | the six new keywords (frontend types), **appended keyword-only after `tr_radius0`** and **forwarded only when set**; the structural reduction of `x_old`; the start-frame structure check against the reduced `x0`; the safe-mode checks — `x0` equals `x_old`, ORTH and same tensor for a start frame, TIED (§3.6) |
| `t3toolbox/optimizers.py:333-360` | frontend `gradient_descent` (forwards `**kwargs`) | docstring only: `g0_reference='cone'`, `cone_passes` and `cone_seed` reach the backend through `**kwargs` |
| `tests/test_contractions_interpreter.py:74` | `HISTORICAL` (249 strings) | `+` the new strings of §4.10, each `''` (fully determined; decided by hand). `AXIS_SIZE` (`:49-54`) unchanged: every new string uses letters already sized there, chosen with distinct sizes within each string |
| `tests/test_optimizers_frontend.py:448-490` | keyword mirror + forwarding | **unmodified** (G-nonbreaking): the mirror covers the new keywords automatically, and the forwarding test passes because a default call forwards no new keyword; new test functions cover the new keywords (T9) |
| `tests/backend/test_zero_reference.py` | the 2026.3.0 zero-reference suite | `+` the cone cases T8 |
| `tests/backend/test_optimizers.py` | the `newton_cg` suite; default `set(stats)` pinned at `:477`, `:860` | `+` T7; the pins stay unchanged |
| `tests/test_dispatch.py`, `tests/test_jax_absent.py` | jit / jax-absent sweeps | `+` T10, T11 |
| `t3toolbox/fitting.py:520-530`, `:599-636`, `:682-990` | the six model factories build `frame = geometry.frame(x)` (`_ragged_frame`, `_uniform_model`) | `+ frame=None`, appended keyword-only on each: the twin of `Problem.local_model(frame=)` (§5.8); a given frame replaces `geometry.frame(x)` after the structural check and, in safe mode, ORTH and the same tensor |
| `tests/test_frame_variations_format.py`, `tests/test_manifold.py`, `tests/test_uniform_manifold.py`, `tests/test_sharing.py`, `tests/test_tucker_tensor_train.py`, `tests/test_uniform_tucker_tensor_train.py`, `tests/test_fitting.py` | the frontend suites | `+` T15 (new test functions only) |

**Reused, not changed** (decision 4: the retraction and the orthogonal representation are not touched):
`tv_operations.tv_to_t3` (`:205-336`, the hint as a doubled-rank T3), `tv_retract` (`:853-892`),
`fv_conversions.t3_orthogonal_representations` (`:123-267`), `t3_orthogonalization.t3_down_orthogonalize_tucker_cores`
(`:144-175`), `tt_orthogonalization.tt_left_orthogonalize` / `tt_right_orthogonalize` (`:58-139`),
`tt_operations.tt_zipper_left_to_right` / `_right_to_left` (`:173-202`) and `tt_reverse` (`:31-44`), `fv_operations.fv_zero_point_frame`
(`:288-430`, untouched — the `s = 0` hint-free instance is tested equal to it: coordinates bitwise, Haar factors to
`1e-14`) with `_haar_rows`
(`:274-285`) and `_identity_prefix` (`:262-271`), `ufv_conversions.ut3frame_to_t3frame` / `t3frame_to_ut3frame` /
`ut3variations_to_t3variations` (`:141-305`), `ut3_masking.ut3_make_masks` (`:25-49`), `ut3_constructors.ut3_zeros`
(`:110-121`), `t3_constructors.t3_zeros` (`:20-41`), `ranks.compute_orthogonal_representation_ranks` (`:579-646`),
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
sentence); `docs/naming_conventions.md` (`:45-48`, the topic-module sentence, I-32); new
`docs/contributor/padded_frame_internals.md`.

---

## 2. Mathematics

### 2.1 Setting

`X_old` is a T3 at ranks `(s, σ)` with an orthonormal frame `F_old = (U°, O°, L°, R°)` whose four rank stores
are `(nU°, nD°, rL°, rR°)`; the target ranks `(r, ρ) ≥ (s, σ)` give the target stores `(nU, nD, rL, rR)` by
the recurrence of `ranks.compute_orthogonal_representation_ranks` (`nU_i = min(r_i, N_i)`,
`rL_{k+1} = min(rL_k nU_k, ρ_{k+1})`, `rR_k = min(rL_k, nU_k rR_{k+1})`, `nD_i = min(nU_i, rL_i rR_{i+1})`). Every
step of the recurrence is monotone in its inputs, so `(r, ρ) ≥ (s, σ)` implies **old ≤ new in all four
stores** — the embedding below always exists. (`s = 0` — no old block — is the zero tensor; its "old stores"
are 0 except the boundary bonds.)

A **padded frame** of `X_old` at `(r, ρ)` is an orthonormal frame `F = (U, O, L, R)` with the target stores'
core shapes such that

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

A **hint** is a T3 `Z` — any Tucker tensor train at the mode sizes `shape`, of any ranks (stack `C`). The
construction reads nothing but `Z`'s cores, so it is a function of an ambient tensor; no frame or tangent type is
involved. That is the honest type, and it is general: the hint can be the gradient at the current padded frame (the
driver's choice, §2.6), another level's search direction, a gradient from another data set or metric, a sum of such
directions, or the data tensor itself. The driver's hint is the gradient `v` at a padded frame `F'`, embedded as the
doubled-rank T3 `tv_to_t3(F', v)` (Tucker ranks `nU + nD`, TT ranks `rL + rR` of `F'`) by the geometry
(`tangent_to_t3`, §3.4).

**Exactness.** When `Z`'s orthogonalized chains contain `X_old`'s interfaces, every step below is the stated
projection exactly (the column-side overlaps `Ω` then have orthonormal rows). Three cases:

- **Gradient hints:** a gradient embedded at a frame containing the old frame's subspaces satisfies the condition
  (P3). Every padded frame of the same old frame contains them, and so does any frame of `X_old` when the old ranks
  are minimal (the frontend reduces `x_old` structurally, §3.6) and `X_old` has full rank at them (generic, not
  structural).
- **Any other T3 hint:** the condition is obtained by **adding `X_old` to it** — the T3 sum `Z + X_old`, rank-additive.
  This changes no normal content: `X_old`'s Tucker rows lie in `U°`, its bond content lies on the old interfaces, and
  its center sits on the old Tucker slots, so every step removes it. What it changes is that the sum's chains contain
  `X_old`'s interfaces.
- **Otherwise:** the output is still an orthonormal padded frame of `X_old`. The pins are never touched, and the
  column-side removals `I − ΩᵀΩ` (with `‖Ω‖ ≤ 1`, overlaps of orthonormal sets) are then contractions rather than
  projections. The condition only decides *which* directions are chosen — a caveat, not a precondition.

**Cost scales with the hint's ranks.** That covers the down-orthogonalization (`N_i × w_i`), the chain
orthogonalizations at the hint's TT ranks, the zippers (old × hint and hint × hint) and the sweeps' SVDs of
`(rL·nU) × zr` matrices. A gradient embeds at the doubled frame ranks; an arbitrary high-rank hint costs more, and a
caller may compress it first (`t3svd` to a few times the target ranks), at the price of whatever directions that
truncation drops.

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
  `v = 0`), the trailing columns are the SVD's remaining left singular directions of the projected content (noise-level
  singular values: their orientation is not determined by the content and is roundoff- and backend-dependent),
  re-orthogonalized against the pins and the leading ones by the Householder QR — always a valid completion, a
  statement about the hint (it carries no information there), not about the data (S1 review, finding 1; R5/E1).
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
  `fv_zero_point_frame`'s draw and sign-fixed QR — the same computation (P2 measured it bitwise in one process; the
  tests compare at 1e-14, §6). Nested in `k` and rank-independent across modes, as there.
- **TT and down cores**: the coordinate directions of the slot's unfolding **outside** the old block, in the
  core's own unfolding order (`L`: `(a, n)` row-major; `R`: `(n, b)`; `O`: `(a, b)` — the orders
  `_identity_prefix` uses), exactly orthonormal and orthogonal to the embedded old block (disjoint supports);
  when those run out (a bond that grows while its neighbours do not — a TT-only increase at the first core is
  the common case), the remainder is the Householder completion of the old columns **inside** the old block,
  seeded by that block's coordinate directions in index order. The seeds are coordinates rather than a Gaussian so
  that the completion is covariant under a sign gauge of the old frame (§3.3, §4.3; P17). At `s = 0` every
  coordinate is outside, and the result is exactly the identity prefix — a structural fact (P2).

Why the asymmetry is kept from 2026.3.0: the Tucker factor is the only part of the frame that meets the data,
and a coordinate-aligned factor gives an exactly-zero hint on data with no energy on the leading
coordinates (`docs/contributor/curvature_internals.md` F32) — from which no iteration can recover; the TT
cores act on rank indices.

### 2.6 The subspace iteration

`F_{k+1} = Pin(F_k, tangent_to_t3(F_k, P_{T_{F_k}} G))`: the gradient at the current frame, embedded as a T3, is the
next hint. At `s = 0` this is the
frame form of **unit-step Riemannian gradient descent on `½‖Y − G‖²`**: the Riemannian gradient at `Y` is
`P_{T_Y}(Y − G) = Y − P_{T_Y} G` (the base point lies in its own tangent space, `v_X`), so the unit step lands
on `P_{T_Y} G` and the retraction truncates it — `Y_{k+1} = T(P_{T_{Y_k}} G)`; since `T_Y` depends only on `Y`'s
frame, it is an iteration on frames, and the pinned construction plays the truncation's role (a T3-SVD-type
truncation of the hint: the HOSVD-style Tucker step and the TT-SVD sweeps). Each pass costs one frame sweep and one transpose (`G` is never
formed: `P_{T_F} G` is the problem's own gradient at the frame). As HOOI is for Tucker, it is a block subspace
iteration — each mode's new basis is the leading subspace of `G` applied to the other modes' current bases —
with the rate set by the gaps of `G`'s unfoldings (`σ_{k+1}/σ_k` per pass), improved by **oversampling** as in
randomized SVD: iterate at the working ranks (`ranks.oversampled_ranks`: `+p`, minimal, never below the target — §3.5), then **truncate by a final
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
 backend/ranks.py           oversampled_ranks(shape, tucker_ranks, tt_ranks, p, sharing)   ── THE "+p, minimal, never below" rule
 backend/linalg.py          pinned_completion, pinned_leading_columns          (factorizations; no T3 semantics)
          │
 backend/padded_frames.py   fv_padded_frame(shape, tucker_ranks, tt_ranks, old_frame, hint, groups, seed)  ── the constructor (ragged, stack C);
   (new, I-32)              fv_zero_point_frame (fv_operations) is its s = 0, hint-free case      hint = any ragged T3 (cores)
          │                          ▲ reused (module-level, all downward): ranks.compute_orthogonal_representation_ranks,
          │                            t3_down_orthogonalize_tucker_cores, tt_*_orthogonalize, tt_zipper_*, tt_reverse,
          │                            tt_squash_tails, contract, fv_operations._haar_rows / _identity_prefix, pad_or_truncate
          │                 (no uniform twin: the uniform geometry member converts explicitly)
          │
 backend/geometry.py        ManifoldGeometryOps / UniformManifoldGeometryOps only (optional, duck-typed members):
                              tangent_to_t3(frame, tangent)            → the tangent as this layer's T3 (tv_to_t3 / utv_to_ut3)
                              padded_frame(shape, tucker, tt, x_old, hint, seed, stack_shape, use_jax)
                                                                       → fv_padded_frame with the geometry's groups
                                                                         (uniform: unpack → fv_padded_frame → repack, host loop)
                              padded_structures(x, x_old, working_ranks) → (shape, C, target ranks, working ranks): ints only
                              with_ranks(tucker, tt)                   → this geometry at those ranks (uniform: from ints and masks)
                              oversampled_ranks(x, p)                  → ranks.oversampled_ranks with the geometry's sharing
                              require_frame_structure(x, frame)
          │
 backend/optimizers.py      padded_start(problem, x_cores, x_old, working_ranks, passes, seed, final_gradient) → PaddedStart
   (T3-free)                Problem.local_model(..., frame=)
                            newton_cg(..., start_frame, padded_from, padded_passes, padded_seed, cone_passes, cone_seed)
                            _cone_reference_gnorm (g0_reference='cone')
          │
 t3toolbox/optimizers.py    newton_cg(... the same keywords; padded_from = the old point ...)
                            padded_start(geometry, kind, ..., x0, x_old, working_ranks=, passes=, seed=)
                            safe-mode checks (x0 equals x_old, ORTH and same tensor for start_frame, TIED x_old)
 t3toolbox/tucker_tensor_train.py   TuckerTensorTrain.oversampled_ranks(p, sharing)   (beside continuation_ranks)
```

**`newton_cg(x0, padded_from=x_old)`, iteration 0:**

```
x0 (the new level's start: the same tensor as x_old at higher ranks, e.g. x_old.resize(...))
 └─ working = geom.oversampled_ranks(x0, PADDED_FRAME_OVERSAMPLE)          (the rank function; the only rank arithmetic)
 └─ padded_start(problem, x0, x_old, working, passes, seed, final_gradient=False)
     ├─ shape, C, target, working = geom.padded_structures(x0, x_old, working) (integers only)
     ├─ geom_w = geom.with_ranks(*working)                                       (uniform: built from ints and masks)
     ├─ r0 = kind.point_forward(x0, sample) − data             (once: the point never moves)
     ├─ F = geom_w.padded_frame(shape, *working, x_old, None, seed)            (the hint-free start; x_old's frame built inside)
     ├─ repeat passes − 1 times:  g = geom_w.project(F, kind.transpose(r0, sample, F, kind.precompute(F, sample)),
     │                                               aux=geom_w.precompute(F))
     │                            F = geom_w.padded_frame(shape, *working, x_old, geom_w.tangent_to_t3(F, g), seed)
     └─ last pass: g at F (as above), then F = geom.padded_frame(shape, *target, x_old, geom_w.tangent_to_t3(F, g), seed)
                                                                                    (the truncation)
 └─ lm0 = problem.local_model(x0, frame=F) → require_frame_structure(x0, F) → the model at F
 stats: 'start_frame_source' = 'padded' and 'padded_start_gnorms' = the per-pass norms — present only on such runs.
 The frame is rebuilt from the retracted point after the first accepted step, as today: the start frame
 influences iteration 0 only.
```

**`g0_reference='cone'`** (§3.7) — the feature's one array-free construction:

```
x0 (any point; only its structure, stack and backend are read)
 └─ padded_start(problem, x0, None, geom.oversampled_ranks(x0, PADDED_FRAME_OVERSAMPLE), cone_passes, cone_seed)
     ├─ shape, C, target, working = geom.padded_structures(x0, None, working)
     ├─ geom_w = geom.with_ranks(*working)
     ├─ r0 = kind.point_forward(corewise_zeros_like(x0), sample) − data      (the zero tensor of x0's ranks, s = 0)
     ├─ F = geom_w.padded_frame(shape, *working, None, None, seed,
     │                          stack_shape=C, use_jax=tree_contains_jax(x0)) (array-free: the default float, I-43)
     └─ the passes as above with x_old = None; final_gradient=True, and .gnorm is the cone reference
```

### 3.2 The constructor `fv_padded_frame`

**Style (round 9).** The constructor is written in the gold-standard backend style (§12). It is a **pure
mathematical function** on immutable basic types. The structure travels as **three int sequences**, `shape`,
`tucker_ranks` and `tt_ranks`, as in `t3_zeros`, `compute_minimal_ranks` and `tt_change_core_shapes`. There is **no
template frame**: no argument is passed to have its shapes read. There is **no dict-of-shapes helper**: integer checks
sit inline, as in `tucker_change_core_shapes`. `fv_zero_point_frame`'s template-frame signature (commit 3e5bc922) is
**not** a precedent — the maintainer has ruled it wrong — and is **not changed in this release**; its defect is
recorded for a later audit arc (§7's ledger).

**Inputs.**

- **The target structure:** `shape` (`len=d`), `tucker_ranks` (`len=d`), `tt_ranks` (`len=d+1`).
- **The old frame:** an orthonormal frame of `X_old`, or `None` for the zero tensor.
- **The hint** (optional): a ragged T3 `.data` `(tucker_cores, tt_cores)` at the mode sizes `shape`, of any ranks,
  stack `C` (§2.3).
- **The sharing `groups` and the `seed`.**
- **For array-free calls only** (no old frame and no hint — the `s = 0` hint-free frame): `stack_shape=()` and
  `use_jax=False`, by the convention of the array-free constructors (`t3_zeros(…, stack_shape=(), use_jax=False)`:
  "no array inputs, so the flag chooses the output type"). There is **no `dtype`** (Nick, round 10; I-43): the dtype
  follows the arrays, and an array-free call returns the backend's default float (float64 under numpy), as `t3_zeros`
  does.

**Derived inside.** The four stores come from the rank recurrence, the one function the orthogonal representation and
the uniform frame masks already use:

```
nU, nD, rL, rR = ranks.compute_orthogonal_representation_ranks(shape, tucker_ranks, tt_ranks)
```

The old frame's stores are read inline from its arrays (`[U.shape[-2] for U in old_frame[0]]`, …), and so are the
hint's ranks and mode sizes (`[B.shape[-1] for B in hint[0]]`, …). The stack is `old_frame[0][0].shape[:-2]` or the
hint's. The backend is `tree_contains_jax((old_frame, hint))`, and the dtype is `xnp.result_type` of the array
inputs. With an array input present, the array-free parameters must keep their defaults (a structural error
otherwise: the arrays decide).

**Output.** A padded frame of `X_old` (§2.1): the frame tuple `(up, down, left, right)` with the derived stores, stack
`C`, and the inputs' dtype and backend (array-free: the backend's default float).

**Structural** (both modes). Round 11 splits these checks (§5.8, §12). The backend keeps four **kept guards**, terse
and inline:

- a count below zero (an old store above the derived one);
- an old frame / hint stack mismatch;
- an array-free argument passed with an array (above);
- since round 12 (review r8 N3), a sharing group whose members differ in `(Tucker rank, size)`. This is the analog of
  `t3svd`'s kept `validate_sharing`: the group concatenation would otherwise fail obscurely.

The rest move to the twin `T3Frame.padded`, with explicit messages:

- `len(shape) == len(tucker_ranks) == len(tt_ranks) − 1`, and `tt_ranks` has boundary 1 *(twin)*;
- the old frame's four stores are ≤ the derived stores, mode by mode *(kept as the count guard; the twin's message
  names the store and mode)*;
- the old frame's and the hint's stacks agree *(kept; the twin as well)*;
- the hint is a consistent T3: `d` cores each; adjacent TT bonds agree; each TT core's middle axis equals its Tucker
  core's rank *(twin: a `TuckerTensorTrain` hint is consistent by construction)*. Its mode sizes are `shape`, and its
  ranks are not compared with anything. A hint with another
  stack (several hints stacked) is an error that names the caller's reduction: combine the hints into one T3 first,
  for example by a T3 sum. Boundary bonds may exceed 1 (a T3 sum has boundary bonds 2): the hint's tails are squashed
  first (§4.4);
- a sharing group's members have equal `(tucker rank, size)` in `(tucker_ranks, shape)` *(kept, round 12; the twin's
  message is explicit)*, and the old frame's group factors equal stores *(twin)*.

**Numerical preconditions** (never checked in the backend; documented on the twin, which checks ORTH and TIED in safe
mode, §5.8): the old frame is orthonormal (ORTH), and with `groups` its Tucker factors are tied (TIED; the group's old
block is read from `U°_{min(g)}`). **Caveats** (not preconditions): the hint's chains should contain `X_old`'s interfaces for the steps
to be exact projections — true of the driver's gradient hints, and obtainable for any hint by adding `X_old` to it
(§2.3); with any other hint the output is still a valid padded frame of `X_old`. Numerically zero singular values of
`X_old`, and in the backend its structural slack, give old-frame directions that are completions, not data — they are
pinned as old like the rest (the frontend reduces `x_old` structurally, §3.6).

**Two paths.** `hint is None` → the hint-free completions (§4.3). Otherwise the hinted construction (§4.4–§4.7),
which starts from the hint's cores. `old_frame is None` → the empty old block (`_empty_old_frame`, built from `shape`
and the boundary ranks), so `s = 0` is not a separate code path. The `s = 0`, hint-free, unstacked, array-free call
`fv_padded_frame(shape, tucker_ranks, tt_ranks)` reproduces `fv_zero_point_frame` on the frame of
`t3_zeros(shape, tucker_ranks, tt_ranks)` — the test builds that frame, for `fv_zero_point_frame`'s sake only. The
coordinate cores agree exactly and the Haar factors to 1e-14 (§6's rule).

**Placement and imports — I-32 (Nick, round 10: a new module).** The constructor and its private helpers live in a
new module, **`backend/padded_frames.py`**. The reason is the import graph alone. The constructor derives its stores
with `ranks.compute_orthogonal_representation_ranks`, and `fv_operations` cannot import `ranks`: `ranks` imports
`sharing` (`ranks.py:16`), which imports `fv_operations` (`sharing.py:34`), so the edge would close the cycle
`fv_operations → ranks → sharing → fv_operations`.

The new module's imports all point downward: `ranks`, `fv_operations` (`ZERO_FRAME_SEED` and the private
`_haar_rows`, `_identity_prefix`), `t3_orthogonalization`, `tt_orthogonalization`, `tt_operations`, `contractions`,
`linalg`, `common`. Only `geometry` imports it. The writer's AST check of the pre-arc graph plus these edges finds no
cycle (round 10). `fv_operations` returns to its committed pre-arc form (S2′, §8). The recurrence stays in `ranks`.
That `ranks` depends on `sharing` at all is a code smell; it is recorded for the later drift-fix arc (§7's ledger)
and not changed here.

Not taken:

- **(B)** a function-level import of `ranks` inside `fv_operations`. It breaks G-acyclic's "no function-level import
  of a library module".
- **(C)** the recurrence re-derived inline. That would be a third copy (`fv_conversions.py:206-214` already
  re-derives it) of a recurrence with a documented order-sensitivity history (`ranks.py`'s docstring, review S1).
- **(D)** the recurrence moved down to `fv_conversions`, with `ranks` keeping an alias (the critic's r6
  recommendation). Nick keeps it in `ranks`.

`docs/naming_conventions.md` gains one sentence beside the `sharing` example (`:45-48`): `padded_frames` is a topic
module whose function carries its family prefix. `geometry` gains `ranks`, `padded_frames` and `ut3_conversions` (the
uniform member's `ut3_to_t3`); none of them imports `geometry`. Gate G-acyclic (S3, rerun in S2′): the backend's
import graph, built by an AST scan from every import node (module- and function-level), is acyclic, and the backend
has no function-level import of a library module.

**Traceability.** All shapes come from the three int sequences, the old frame and the hint; the Haar draws and the
coordinate index sets are host numpy constants; every array operation is `xnp` / `contract` / `xnp.linalg`. So
`fv_padded_frame` is `jax.jit`-traceable with the int sequences static (T10) — a bonus, not a requirement: its
consumers call it eagerly.

### 3.3 The uniform layer: no twin, an explicit conversion; and the two layers

**No `ufv_padded_frame`** (Nick, round 6, item 7). The construction is ragged by nature — it works on unpacked real
blocks — and rank continuation is a ragged-layer activity (`rank_continuation.md` §"On the uniform layer": there is
no uniform `resize`). A `ufv_` name would hide a host round trip behind a name that suggests a native uniform op. So
the conversion happens **explicitly at its one call site**, `UniformManifoldGeometryOps.padded_frame`, whose
docstring says in one line that the frame is built through the ragged constructor, host-side and eager. The member
requires concrete masks, then for each element of a host loop over `C`:

1. **Unpack.** The old frame — built **natively** from `x_old`'s full `.data` by `ut3_orthogonal_representations`
   — by `ufv_conversions.ut3frame_to_t3frame`, and the hint — a uniform T3 `.data` `(tucker_supercore, tt_supercore,
   shape, masks)` — by `ut3_conversions.ut3_to_t3`. The target structure is the element's integer ranks (the member's
   arguments). `ut3_to_t3` returns one ragged T3 per element and gathers through the masks (boolean index), so it
   handles gappy hint masks. The driver's hints come from `utv_to_ut3`, which pads every bond to `rL + rR` with its
   own concatenated masks (the appended boundary slots marked real, zero in the supercore); the gather takes exactly
   the real blocks.
2. **Construct.** Run `fv_padded_frame(shape, tucker_e, tt_e, old_frame_e, hint_e, self.groups, seed)`.
3. **Repack.** Repack at **this geometry's** frame padded sizes with `ufv_conversions.t3frame_to_ut3frame(..., N=, nU=,
   nD=, rL=, rR=)`. The masks are built from the integer stores, `ufv_make_frame_masks(*compute_orthogonal_representation_ranks(shape,
   tucker, tt), nU, nD, rL, rR)` — the calls `ut3_orthogonal_representations` itself makes — and their variation
   masks must equal the geometry's `var_masks` (a structural error otherwise).
4. **Restack.** Restack with `ufv_stack`.

The padding is exactly zero, and garbage in any input's padding is never read. On **identical inputs** the real
blocks equal the ragged construction's to rounding (the same code on the same arrays; §6 tests them at 1e-14 through
the member). No `pad_safe_svd` is involved: every factorization runs on unpacked real blocks.

The same three calls are the recipe a raw uniform `.data` user follows outside a geometry. The recipe is backend-user
documentation, so it lives in `docs/contributor/padded_frame_internals.md` ("The uniform conversion"), cited by the
member's docstring (round 12; §5.2). The host loop covers uniform- and varying-rank stacks alike (§3.9).

The razor is satisfied:

- the math is the standalone `fv_padded_frame`;
- the conversion glue is a backend method a raw-data user can call on any `UniformManifoldGeometryOps` (it reads only
  `self.groups`);
- the glue is documented as a recipe.

So no backend user is left "one fiddly step short".

The ledger records this as a documented exception to the "every ragged op has a uniform twin" symmetry, for frame
construction at degenerate points. `ufv_zero_point_frame` — the same kind of wrapper — is left untouched this
release, with the consistency question noted for later (§7).

**Across the layers: gauge equivalence** (I-28 reverted, round 6). Each layer builds its old frame **natively** from
`x_old`. The ragged layer uses `t3_orthogonal_representations(x_old)`. The uniform layer uses
`ut3_orthogonal_representations(x_old.data)` — the full `.data` with `x_old`'s own masks, which also handles gappy masks.
The two differ by a sign gauge even at a generic point (LAPACK against the pad-safe SVD, equal in absolute value
only: `uniform_equivalence_contract.md` §"Gauge-carrying operations"; the critic measured entrywise differences of order 1
with equal magnitudes to 2.7e-12). Every step of the construction is covariant under such a gauge:

- the Tucker hint-free completion depends on `span(U°)` and an ambient Haar draw only;
- the coordinate completions — outside-first, and the in-block completion with **coordinate seeds** — are mapped by a
  sign gauge onto signed copies of themselves. A seeded Gaussian in the core's coordinates is not: probe P17 measured a
  projector difference of 1.4 under a sign gauge, against 0.0 for coordinate seeds. That is why I-11 reverts to
  coordinate seeds;
- every hinted step reads spans and the hint as a tensor.

So the two layers' padded frames agree **up to gauge and rounding** — the same subspaces, hence the same tangent
space and the same first step. §6 tests this gauge-invariantly: assembled projectors and a dense `P_{T_F} Y`.

**The canonical list of exceptions** — the one list, stated here and referenced (R5, R10) or quoted (the contract
doc, the internals page) everywhere else. Each is a case where a choice is not determined by spans:

- **E1 — a near-tie at a cut** (`σ_k ≈ σ_{k+1}` in a hint unfolding);
- **E2 — a degenerate hint** (fewer directions than slots, or zero);
- **E3 — old frames that differ by more than signs** (a tied or numerically deficient `x_old`: a rotation within a
  tied singular cluster);
- **E4 — an in-block coordinate seed in the old columns' span.** This happens with a coordinate-aligned old core. It
  is rare on library-produced frames: the critic's r4 probe found first in-block seed residuals of 0.05–0.8, and core
  0 has no in-block room for a minimal `x_old`. Its effect on the hint-free start is damped by the hinted passes but,
  at the default four, not removed.

This is a statement about **subspaces** only. Tangent **coordinates** live in one frame's gauge, and the existing
entrywise same-frame guard for them (`safety.frames_equal`, `numerical_contracts.md` §"Equality is explicit") is
correct and untouched: two gauge-equivalent frames are different frames for coordinates.

### 3.4 Six optional geometry members

Looked up duck-typed (the `zero_point_frame` precedent, `backend/optimizers.py:78-85`), defined on the two manifold
geometries only. A geometry without them — the corewise ones (a corewise frame is the point itself: there is no
completion to choose) and any custom geometry — keeps working; `padded_from`, `start_frame`, `local_model(frame=)`
and `g0_reference='cone'` on it are structural errors naming the missing member.

**Round 9 — structure as integers.** The members pass the structure as int sequences, like the gold-standard backend
(§12):

- `padded_structures` returns **integer structures only** — never a frame, never an orthogonalized zero tensor;
- `padded_frame` takes the target `(shape, tucker_ranks, tt_ranks)` and the **old point**, and builds the old frame
  itself;
- the uniform members build from `(shape, ranks)` integers and masks.

The round-6 table, which returned a target frame and a working template from `t3_orthogonal_representations(t3_zeros(...))`
and a working geometry from `from_point(ut3_zeros(...))`, is withdrawn.

**Why a separate `tangent_to_t3` member.** The driver's hint is the gradient embedded as a T3, and the embedding is a
layer fact: `tv_to_t3` on the ragged layer, `utv_to_ut3` with the frame's variation masks on the uniform one. There
were two ways to supply it:

- **A small embedding member (chosen).** `padded_frame` keeps one hint type, the T3 a user passes. The embedding is a
  reusable geometric fact ("the tangent as an ambient T3") rather than a padded-frame detail, and the driver composes
  two members and stays T3-free.
- **Folding the embedding into a "compute the pass's hint" member.** That member would duplicate the embedding, or
  would make `padded_frame` accept tangents again.

No `shared_data` is passed: the untied embedding represents the same tensor as the tied one ("an exact rewrite",
`tv_operations.py:230-236`), and the construction reads the hint only as a tensor.

`oversampled_ranks` is a one-line adapter around the rank function (§3.5), so the T3-free driver fills
`working_ranks` without rank arithmetic of its own. `with_ranks` gives the geometry at another structure, which the
uniform layer needs because its geometry holds masks; it mirrors the existing `with_sharing`. It is needed:
`from_point` and `from_frame` take arrays, and building a zero point only to read its masks is the ruled-out pattern.

`from_point`'s docstring warns against building masks by hand (`geometry.py:600-603`), and the uniform `with_ranks`
does build them. Here that is safe, for three reasons:

- its masks come from the recurrence and the mask builder that `ut3_orthogonal_representations` itself calls
  (`ufv_conversions.py:111-115`), at the padded sizes that sweep produces (the min rules in the table below);
- the conversion's frames are prefix by construction;
- at the target, the integer-built masks are checked against the geometry's (§3.3 step 3).

T5(i) guards the rest (Nick, round 10): `with_ranks` of a geometry's own ranks must reproduce its `masks` and
`var_masks` exactly, so any drift from the sweep fails a test.

| member | `ManifoldGeometryOps` | `UniformManifoldGeometryOps` |
|---|---|---|
| `tangent_to_t3(frame, tangent)` | `tv_operations.tv_to_t3(frame, tangent)` — the doubled-rank T3 cores (Tucker `nU + nD`, TT `rL + rR`) | `utv_operations.utv_to_ut3(frame, (tangent[0], tangent[1], shape, var masks))` with the variation masks of `frame` (`ufv_masking.ufv_variation_masks(frame[5])`) — a uniform T3 `.data`, bonds padded to `rL + rR` with its own concatenated masks |
| `padded_frame(shape, tucker_ranks, tt_ranks, x_old, hint, seed, stack_shape, use_jax)` | `fv_padded_frame(shape, tucker_ranks, tt_ranks, old_frame, hint, self.groups, seed, …)` with `old_frame = t3_orthogonal_representations(x_old)[0]` (`None` when `x_old is None`); the array-free arguments are passed only when `x_old` and `hint` are both `None` | the explicit conversion of §3.3 / §4.8, from **ints and masks**. Per element of a host loop over `C`: the old frame **natively** from `x_old`'s full `.data` (`ut3_orthogonal_representations`, then `ut3frame_to_t3frame`); the hint by `ut3_to_t3`; `fv_padded_frame(shape, tucker_e, tt_e, …, self.groups, seed)`; repack with `t3frame_to_ut3frame` at **this geometry's** frame padded sizes; the frame masks built from the integer stores with `ufv_make_frame_masks(*compute_orthogonal_representation_ranks(shape, tucker, tt), nU, nD, rL, rR)` (the calls `ut3_orthogonal_representations` makes, `ufv_conversions.py:110-119`), which must equal the geometry's. Docstring: "built through the ragged constructor, host-side and eager" |
| `padded_structures(x_cores, x_old, working_ranks)` | returns `(shape, stack_shape, target_ranks, working_ranks)` — **integers only**: `shape`, the stack `C` and the target ranks from `x_cores`' core shapes, inline (`[B.shape[-1] for B in tucker_cores]`, …); `working_ranks` validated (or the target when `None`) | the same from the geometry's held `shape` and masks (`masks[0].sum(-1)`, `masks[1].sum(-1)`; per-element arrays on a varying-rank stack); `C` from `x_cores`' supercores |
| `with_ranks(tucker_ranks, tt_ranks)` | `self` (the ragged geometry holds no structure) | a new geometry at those ranks built from ints and masks: `masks = ut3_make_masks(tucker_ranks, tt_ranks, n, r)` with `n`, `r` the larger of this geometry's padded sizes and the ranks' maxima; `var_masks = ufv_variation_masks(ufv_make_frame_masks(*compute_orthogonal_representation_ranks(shape, tucker_ranks, tt_ranks), nU, nD, rL, rR))` with the frame padded sizes by the orthogonal sweep's min rules on `(n, r, N)` (`nU = min(n, N)`, `rL = rR = r`, `nD = min(nU, r²)`) — no orthogonalized zero tensor |
| `oversampled_ranks(x_cores, p)` | `ranks.oversampled_ranks(shape, *target ranks, p, sharing)` — unstacked | the same, from the held masks (unstacked) |
| `require_frame_structure(x_cores, frame)` | the frame's core shapes equal the stores `compute_orthogonal_representation_ranks` gives for `x_cores`' ranks, plus `C` | the frame's variation masks equal `self.var_masks`, and its supercore shapes the geometry's frame padded sizes |

**`padded_structures`' structural checks** (both modes). Round 11 moves them to the frontend callers — the geometry
twins, the frontend `padded_start` and `newton_cg(padded_from=)` — through one helper per layer,
`_require_padded_pair` (§5.8), with explicit messages. The member keeps none: the constructor's count guard catches
the one violation that would otherwise be silent. The checks:

- **Layer, shape and stack.** `x_old` is of `x_cores`' layer — the ragged cores pair, or the uniform full `.data`; a
  bare supercore pair is rejected, because the old frame needs `x_old`'s own masks — with the same shape and stack.
  (No padded-`N` requirement in the backend: the old frame unpacks through `x_old`'s own masks and the repack uses the
  geometry's sizes. The frontend checks an equal padded `N`, because its same-tensor check subtracts — §3.6.)
- **Stores.** `x_old`'s frame stores — `compute_orthogonal_representation_ranks` of its ranks, which are read from its
  core shapes or its masks — are ≤ `x_cores`', per store and stack element.
- **Working ranks.** `working_ranks`, when given, has lengths `d` / `d + 1` (or per-element `(d,)+C` / `(d+1,)+C` int
  arrays on a varying-rank uniform stack), and its stores are ≥ the target's.

There is no slice of `x_cores` and so no layout requirement: the old frame comes from the old point itself, inside
`padded_frame`. It is recomputed at each pass, a pure function of `x_old`: one orthogonalization at the old ranks per
pass, small next to the pass's transpose. That `x_cores` and `x_old` represent the **same tensor** is a numerical
precondition, checked by the frontend in safe mode (§3.6). The ranks of `x_old` need not be minimal in the backend; the
docstring carries the slack caveat of §3.2, and the frontend reduces `x_old` first.

Why members and not a T3-specific loop: the loop (§3.5) is generic — residual once, gradient per pass, one
constructor call per pass — and only these facts are layer- and representation-specific. Keeping the loop in
`backend/optimizers.py` keeps that module T3-free (CLAUDE.md, the 2026.2.0 restructure), and every line of math the
members bind is a standalone function.

### 3.5 The working ranks, and the driver `padded_start` (backend, T3-free)

**The rank function.** `ranks.oversampled_ranks(shape, tucker_ranks, tt_ranks, p, sharing=None)` is the only place the
oversampling rule lives. It takes integer tuples and returns the elementwise maximum of the input and
`compute_minimal_ranks(shape, r + p, (1, ρ_1 + p, …, ρ_{d−1} + p, 1), sharing)`: `+p` on every Tucker rank and interior
bond, reduced to structurally minimal ranks (the shared notion with sharing, so a group keeps one rank).

- For a minimal input the output is minimal, and never below the input.
- A non-minimal input keeps its slack, so the working structure never undercuts the target.
- `old ≤ target ≤ working` holds in all four stores, and `p = 0` returns the input.

Its frontend sibling `TuckerTensorTrain.oversampled_ranks(p, sharing=None)` sits beside `continuation_ranks`.
`padded_start` and `padded_structures` compute no ranks.

```
padded_start(problem, x_cores, x_old=None, working_ranks=None, passes=PADDED_START_PASSES, seed=None,
             final_gradient=True) -> PaddedStart
```

- **The old point.**
  - `x_old` given: `x_cores` is the new level's point, the same tensor as `x_old` at higher ranks (e.g. its `resize`).
    The residual is taken at `x_cores` — the point never moves — and the returned frame is a padded frame **of that
    tensor**.
  - `x_old=None`: the **zero tensor of `x_cores`' ranks** (`s = 0`). The residual is taken at
    `corewise_zeros_like(x_cores)` (`_zero_reference_gnorm`'s formula) and the returned frame is the **cone frame
    at the zero tensor**. Its base point is 0 whatever `x_cores` holds, so it is a valid `start_frame` only for a
    zero `x0`.

    Its first construction is the feature's only **array-free** call. `stack_shape` comes from
    `padded_structures`, and `use_jax` from `tree_contains_jax(x_cores)`: the optimizer layer reads its point
    (§12). There is no dtype (I-43), so the frame comes out in the backend's default float, as `t3_zeros`'s
    output does. On float32 data under numpy, or under jax with x64 enabled, the passes therefore run in float64
    by promotion. The cone value is unaffected. The docstring tells a caller who reuses such a frame as a float32
    `start_frame` to cast it first.
- **Working ranks, passes, seed.**
  - `working_ranks=None` means the target ranks, i.e. no oversampling. The convenience paths —
    `newton_cg(padded_from=…)` and the `'cone'` resolution — pass `geom.oversampled_ranks(x0,
    PADDED_FRAME_OVERSAMPLE)` (`p = 5`).
  - `passes` is the number of gradient evaluations: `PADDED_START_PASSES = 4` here, `CONE_REFERENCE_PASSES = 10` from
    the cone resolution.
  - `seed=None` is resolved by the geometry to `fv_operations.ZERO_FRAME_SEED`.
  - Plain keywords throughout: there is no options object (I-3).
- **The passes.**
  - `passes ≥ 1` gradients are evaluated. Each is embedded as a T3 by `geom_w.tangent_to_t3(F, g)` and handed to
    the constructor as the hint.
  - Passes `1 … passes − 1` rebuild the frame at the working structure, and the last one at the target structure
    (§3.1).
  - Every projection passes `aux=geom.precompute(F)` (as `_zero_reference_gnorm` does), so a shared geometry builds
    its companion once per pass.
  - Only the misfit gradient is used. The regularizer is not consulted: its gradient `λ v_X` lies in `T_X M_s` and
    the two-sided steps remove it, as in the zero reference.
- **The result.** `PaddedStart(frame, gnorms, gnorm)`:
  - `frame`: the frame at `x_cores`' structure;
  - `gnorms`: the captured `‖P_T ∇f‖` per pass, at that pass's structure (the working-structure values are not
    comparable to the target one);
  - `gnorm`: the captured norm at the returned frame when `final_gradient=True` — one more sweep and transpose;
    `newton_cg` passes `False`, because iteration 0 computes it anyway.

  With `x_old=None` that `gnorm` **is** the cone reference. With `x_old` given it is the deflated warm-start
  gradient at the padded frame and must not be used as a Newton reference.
- **Stacks and errors.**
  - Stacked `x_cores` (`C ≠ ()`) are supported: `gnorms` and `gnorm` are then per element (`geom.inner` is
    per-element), and `working_ranks` per element on a varying-rank uniform stack. The optimizers stay unstacked
    (§3.9).
  - It raises (structural) on a geometry without the members: "padded starts need a manifold geometry".

### 3.6 The optimizer seam

**`Problem.local_model(..., frame=None)`.** When `frame` is given it replaces `geom.frame(x_cores)` for the sweep,
the geometry aux, the gradient, the curvature, the retraction and the regularizer's base point.

- **Preconditions (documented).** `frame` is **orthonormal** — ORTH; `numerical_contracts.md` §`fitting.LocalModel`
  calls it a construction invariant of `geometry.frame(x)`, and a caller-supplied frame makes it a precondition. It
  also represents the **same tensor** as `x_cores`: the residual is computed at `x_cores`, everything
  frame-dependent at `frame`.
- **Checked always.** `geom.require_frame_structure(x_cores, frame)`, structurally. A geometry without the member
  raises a structural `ValueError` naming it, as the exact-mode members do.
- **Not checked in the backend.** The numerical properties (house rule); the frontend checks them.

**`newton_cg` (backend): six keywords, appended keyword-only** after its last parameter (`callback`), behind a `*` —
so no existing positional call changes meaning (O-1; MINOR A of review r4).

- **The keywords.**
  - `start_frame`: iteration 0's frame — an orthonormal frame of `x0`'s tensor at `x0`'s structure.
  - `padded_from`: the **old point** `x_old` — on the ragged layer the cores pair, on the uniform layer the full
    `.data`. Iteration 0's frame is `padded_start(problem, x0, x_old, geom.oversampled_ranks(x0,
    PADDED_FRAME_OVERSAMPLE), padded_passes, padded_seed, final_gradient=False).frame`.
  - `padded_passes`, `padded_seed`: the start's passes and seed. `None` gives `PADDED_START_PASSES` / the
    geometry's default seed; either one without `padded_from` is a structural error.
  - `cone_passes`, `cone_seed`: the same for `g0_reference='cone'` (`None` gives `CONE_REFERENCE_PASSES` / the
    default seed); either one with any other reference is a structural error. The values are validated
    structurally: `passes ≥ 1`, and `0 ≤ seed < 2**32`, both integers.
- **Exclusion.** `start_frame` and `padded_from` together are a structural error. With neither, the run is today's
  run, `stats` included, unchanged (decision 2; T7(a), G-bit).
- **Record** (only when used — default runs' `stats` keep exactly today's keys, `tests/backend/test_optimizers.py:477`,
  `:860`). `stats['start_frame_source']` is a label, `'given'` or `'padded'`; with `padded_from` there is also
  `stats['padded_start_gnorms']`, the per-pass captured norms. History rows are unchanged (no new `NewtonInfo`
  field).
- **Under `use_jit`.**
  - `_prepare_jit_inputs` converts `x0`, `sample` and `data` first.
  - `padded_from` gets the same conversion as `x0`, the uniform `.data` through `common.partition_static`.
  - A given `start_frame` is converted by `_prepare_jit_frame(use_jit, frame)`: `common.partition_static`, then
    `tree_to_jax` on the dynamic leaves, then `rebuild_static` (a uniform frame's masks and `shape` stay host).
  - A `padded_from` frame is built eagerly from the converted inputs.
  - The start frame adds no compile: it enters the CG as the local model's leaf, and a uniform frame's masks —
    value-hashed static aux — equal the problem geometry's frame masks.
- **Scope.** The start frame influences **iteration 0 only**: the local model is rebuilt from the retracted point
  after the first accepted step (unchanged code); a rejected trust-region trial keeps the model, hence the start
  frame. The `'zero'` reference reads iteration 0's frame for its structure only (unchanged code: `fv_zero_point_frame`
  is untouched, I-26), so with a start frame the default `'zero'` value is today's, unchanged. The `'cone'` reference
  is resolved from `x0` before the loop (§3.7) and does not read the frame.

**Frontend `newton_cg`.**

- **The keywords and their forwarding.** The six keywords are mirrored, appended keyword-only after `tr_radius0`. `padded_from` takes a `TuckerTensorTrain` /
  `UniformTuckerTensorTrain` of `x0`'s layer, and `start_frame` a `T3Frame` / `UT3Frame`; their data are passed
  through. A new keyword is **forwarded only when it is set**, so a default call reaches the backend exactly as today
  — the pre-existing forwarding test passes unmodified (G-nonbreaking).
- **Frontend-only work.**
  - **Reduction of `x_old`.** `x_old` is reduced structurally before its frame is built (`_ragged_minimal` /
    `uniform_minimal`, sharing-aware), which removes the slack that would be pinned as old (formerly I-30). This is
    a no-op on a continuation result: `newton_cg` returns minimal points. `x0` is reduced as usual by
    `least_squares_problem`; there is no layout requirement on it any more.
  - **Structural, both modes.** A `start_frame` must have the structure of the reduced `x0`; the error says "build it
    at `x0`'s minimal structure" (I-20's surviving half). Also errors: a `padded_from` or `start_frame` of the wrong
    layer; a uniform `x_old` with a different padded `N`, which `allclose`'s subtraction requires; any of the new
    keywords on a corewise geometry.
  - **Numerical, safe mode only** (`safety.checks_active` on the operands; skipped under `safety.unsafe()` and under
    a trace):
    - with `padded_from`, **the same tensor**: `x0.allclose(x_old, rtol=safety.effective_rtol(x0.data,
      x_old.data))`. The norm of the difference is taken **through orthogonalization**, as `allclose` does, and never
      by the inner-product formula `‖a‖² + ‖b‖² − 2⟨a, b⟩`, which bottoms out near 3e-8 in float64 (the critic's
      measurement) and would raise false errors at the numpy rtol of 1e-9. The message: "x0 and padded_from are not
      the same tensor (‖x0 − x_old‖ = … relative)";
    - on a shared geometry, **TIED for `x_old`** (`x_old.has_shared_tucker_factors(sharing)`): the group's old block
      is read from `x_old`'s frame;
    - with `start_frame`: ORTH (`start_frame.is_orthogonal()` at the safety tolerance — the cached
      `orthogonality_residual`) and the same tensor (`start_frame.to_t3()` / `.to_ut3()` `allclose` `x0` at the
      safety rtol, the analog of `fitting._require_at_frame`'s numerical leg, `:108-126`); on a shared geometry, TIED
      for the start frame's base point.

### 3.7 `g0_reference='cone'`

`_G0_REFERENCES` gains `'cone'`; `_resolve_g0_reference` resolves it before the loop, on the level's own problem:

1. The geometry must have `padded_structures` and `oversampled_ranks`. Otherwise this is a structural `ValueError`
   (corewise and custom geometries); the message names the member.
2. The reference is computed iff `g0norm_newton is None` (the `'zero'` rule, m32 of the curvature plan).
3. `_cone_reference_gnorm(problem, x0, cone_passes, cone_seed)` is `padded_start(problem, x0, None,
   geom.oversampled_ranks(x0, PADDED_FRAME_OVERSAMPLE), cone_passes or CONE_REFERENCE_PASSES, cone_seed,
   final_gradient=True).gnorm`. §2.7 says what this number is.
4. A vanishing value (`== 0`, all-zero data) is a `ValueError`. `'cone'` is always explicit, so the `'zero'`
   default's fallback-with-warning does not apply.
5. It is recorded as `g0_reference='cone'` (and `cg_reference='cone'` unless `g0norm_cg` overrides). These are the
   existing per-row fields; their values change only on runs that ask for `'cone'`. The display reads
   `g₀ = tangent cone`.

The default (`g0_reference=None`) is unchanged — `'zero'` on the manifold geometries (decision 2; O-1: this release is
non-breaking).

**The tolerance (O-9, settled by Nick).** The docs recommend `gtol_rel=1e-4` with `'cone'`.

- For the plan's record only, not shipped text: the cone value is 2.3–5.4× **T3Polynomial's** coordinate-aligned zero
  reference on the Darcy surrogate (§0's table). So `1e-4` against the cone is about `3e-4` of that downstream
  reference, about 3× tighter than T3Polynomial's `1e-3` regime. It is not comparable with the library's own
  pre-2026.3.0 stop, which was against the warm start (CHANGELOG `[2026.3.0]`; round 1's m7). The user docs state the
  recommendation without the comparison.
- The S11 ladder pair runs `1e-4` and `3e-4`, so the cost of the recommendation is on record.

`gradient_descent` resolves through the same function and gains `cone_passes` and `cone_seed`.

### 3.8 Structural vs numerical — where each decision lives

| decision | kind | where | both modes / safe only |
|---|---|---|---|
| how many new directions per slot | structural (rank differences) | `fv_padded_frame` | — (no decision is made) |
| which directions (hint) | numerical content (the hint's orientation) | `pinned_leading_columns` | — (deterministic computation, no check) |
| which directions (hint-free) | convention (seeded Haar for Tucker, coordinates for TT / down) | `_tucker_completion`, `_outside_first` | — |
| old stores ≤ the target stores derived from `(shape, tucker_ranks, tt_ranks)`; stacks (`C` everywhere, the hint included); the hint a consistent T3 at the mode sizes `shape`; groups; `d` | structural | the twin `T3Frame.padded`, explicit; the backend keeps the count and stack guards (§3.2) and the uniform member's repack-mask check (round 11) | both |
| `x_old` of `x0`'s layer, shape, stack (uniform: full `.data`); `x_old`'s frame stores ≤ `x0`'s; `working_ranks` lengths and ≥ the target | structural | the frontend `_require_padded_pair` (round 11; formerly `padded_structures`) | both |
| an explicit frame at `x`'s structure | structural | `geom.require_frame_structure` via `Problem.local_model` | both |
| `start_frame` with `padded_from`; `padded_passes` / `padded_seed` without `padded_from`; `cone_passes` / `cone_seed` without `'cone'`; invalid passes / seed values; a geometry without the members | structural | `newton_cg`, `padded_start` | both |
| `start_frame` at the reduced `x0`'s structure; a frame or old point of the wrong layer; a uniform `x_old` with a different padded `N` (the same-tensor check's `allclose` subtracts); a corewise geometry | structural | frontend `newton_cg` / `padded_start` | both |
| `x0` and `x_old` are the same tensor | numerical precondition | frontend (`allclose` at `effective_rtol`) | safe only |
| a start frame is orthonormal (ORTH) and represents `x0` | numerical precondition | frontend | safe only |
| tied factors (TIED) for `x_old` / a start frame's base point on a shared geometry | numerical precondition | frontend | safe only |
| the hint's chains contain `X_old`'s interfaces (true of gradient hints; obtainable by adding `X_old` to any hint) | caveat (decides directions, never validity) | documented on `T3Frame.padded` (round 11) | — |
| an in-block coordinate seed lies in the old columns' span (a coordinate-aligned old core; exception E4) | caveat (decides directions, never validity; backend-dependent) | documented on `T3Frame.padded` (round 11) | — |
| `x_old`'s numerically zero singular values (and, in the backend, its structural slack) give arbitrary old directions | caveat (the frontend's reduction removes the slack) | documented on the frontend `padded_start`, `newton_cg(padded_from=)` and the geometry twins (round 11) | — |
| the cone reference vanishes | runtime (explicit request impossible) | `_resolve_g0_reference` | both |

### 3.9 Stacking — what is committed and why

- **`fv_padded_frame`: the frame stack `C`, vectorized.** A ragged stack has one rank structure, so every count and
  index set is shared and every operation is a `C`-grouped `contract`, a batched `xnp.linalg.svd` / `qr`, or an
  elementwise op; the Haar seeds broadcast over `C`. At `s = 0` the hint-free result reproduces
  `fv_zero_point_frame`'s frame broadcast over `C` (`fv_zero_point_frame` itself keeps refusing stacks — unchanged).
- **One hint per frame.** The hint is a T3 with the frame's stack `C`. Several hints — a tangent stack, or
  directions of different origin — are combined by the caller into one T3 (a T3 sum, say) before the call; that
  reduction is the caller's policy.
- **The uniform member: `C` by a host loop over the elements, by decision** — for uniform- and varying-rank stacks
  alike (Nick, round 6; the masked vectorization is rejected, §7's ledger entry):
  - it follows the `ufv_zero_point_frame` precedent (the zero-tensor, hint-free case of this construction, host-side
    and eager);
  - a masked design would thread three structures (the target structure, the old point, the hint), adding two
    masked primitives and five masked steps — 250–400 lines against about 50;
  - it measured slower than the loop on CPU at ranks 30–50;
  - its degenerate-hint completions would diverge from the ragged ones;
  - the one consumer, `newton_cg`, is unstacked.

  If a stacked consumer appears, the cheap route is grouping the elements by structure (about 15 lines).
- **`padded_start` (backend and frontend): `C`**, following the constructor and the kinds (`Problem.local_model`
  already supports a stacked point): per-element `gnorms` / `gnorm` (arrays of shape `C`; floats unstacked);
  `working_ranks` as per-element `(d,)+C` / `(d+1,)+C` int arrays on a varying-rank uniform stack.
- **The optimizers stay unstacked** (`_require_unstacked`, unchanged); `newton_cg(padded_from=...)` therefore runs
  `padded_start` unstacked.

---

## 4. Algorithms, step by step

### 4.0 Notation and conventions for every step

- **The target structure** is the three int sequences `(shape, tucker_ranks, tt_ranks)` and the four stores
  `(nU, nD, rL, rR)` that `ranks.compute_orthogonal_representation_ranks` derives from them (§3.2 and §5.2 are
  normative).
- Every array carries the frame stack `C` as **leading** axes (`C = ()` for one frame). The old frame `(U°, O°, L°, R°)`
  has stores `(s, sD, sL, sR)`; the target `(nU, nD, rL, rR)`. The **counts** are `kU_i = nU_i − s_i`,
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
an orthonormal `[P | U_k]` returns `U_k` up to column signs); otherwise the trailing columns are the SVD's noise-level
left singular vectors of `Aperp`, re-orthogonalized against `P` by the QR — valid, but their orientation is decided by
roundoff (it changes when content inside `span(P)` changes, and across backends), so projector-equality tests must
use hints with at least `k` directions per slot; the hint has no information there, and nothing is thresholded. The other LAPACK-dependent case is a
tie at the cut (`σ_k = σ_{k+1}`), where any basis of the tied subspace is a leading set. Neither primitive sees
padding: the uniform layer unpacks first (§4.8), so `pad_safe_svd` is not involved anywhere in this feature.

### 4.2 Set-up: validation, the empty old frame

1. Derive the stores from the three int sequences with `ranks.compute_orthogonal_representation_ranks`. Take `xnp`
   and the dtype from the array inputs (`tree_contains_jax((old_frame, hint))`, `xnp.result_type`); on an array-free
   call, take `xnp` from `use_jax` and the backend's default float (float64 under numpy), as `t3_zeros` does. There is
   no `dtype` argument (I-43).
2. `old_frame is None` → `_empty_old_frame(shape, C, xnp, dtype)`, with step 1's dtype, built from integers:
   `U°_i` `C+(0, N_i)`; `O°_i`, `L°_i`, `R°_i` with zero-size middle axes and bond stores `sL = sR = (1, 0, …, 0, 1)`
   — the stores of the zero tensor. No separate `s = 0` path.
3. The structural checks of §3.2 (each a `ValueError` naming store and mode), the hint's included: a consistent T3,
   the mode sizes `shape`, stack `C` (any other stack names the caller's reduction, §3.9).
4. Group keys `k_g = min(g)` (the `fv_zero_point_frame` rule), the per-group members, and the group's old block
   `U°_g := U°_{k_g}` (TIED is the caller's precondition, §3.2).

### 4.3 The hint-free completion

**Tucker, per group `g`** (one factor for all members):

- `s_g == 0`: `U_g = xnp.asarray(_haar_rows(nU_g, N_g, k_g, seed), dtype=dtype)`, broadcast over `C` —
  `fv_zero_point_frame`'s own call (the same computation; §6 compares the Haar factors at 1e-14, the library's own
  "reproducible to ~1e-15" statement for them).
- `s_g > 0`: `G = RandomState([seed, k_g]).standard_normal((kU_g, N_g))` (host; the same keyed stream, its first
  `kU_g` rows, so the completion is nested in `kU_g`); `new = pinned_completion(swapaxes(U°_g), broadcast(Gᵀ),
  kU_g)`; `U_g = concatenate([U°_g, swapaxes(new)], -2)`. Haar-distributed in `U°_g`'s complement.

**TT and down cores** — `_outside_first`, per core, in the core's own unfolding order (`L`: `(a, n)` rows a-major,
new **columns**; `R`: `(n, b)` columns n-major, new **rows**; `O`: `(a, b)` a-major, new **down rows**):

1. The coordinate directions of the unfolding **outside** the old block (`{a < sL_k, n < s_k}` for `L_k`, and
   likewise), in index order; the first `t = min(count, #outside)` become unit vectors — host constants (an index
   set), exactly orthonormal and orthogonal to the embedded old block (disjoint support).
2. If `t < count` (a bond grows while its neighbours do not — e.g. a TT-only increase at the first core), the
   remaining `count − t` come from **inside** the old block.
   - They are `pinned_completion(_unfold(old core, 2), E_in, count − t)` in the old block's own coordinates, with
     `E_in` the block's **coordinate directions in index order** as seeds, then `_embed`ded into the core's block.
   - **Coordinate seeds, not a seeded Gaussian** (round 6, forced by the gauge-equivalence decision of §3.3). A sign
     gauge of the old frame maps coordinate seeds onto signed copies, so the completion's span is covariant. A fixed
     Gaussian in the core's coordinates is not: P17 measured a projector difference of 1.4 against 0.0.
   - The price is the case round 1 chose the Gaussian for: a coordinate seed lying in the old columns' span (a
     coordinate-aligned old core). Householder then completes it deterministically for a given backend. It affects
     only the hint-free start, whose effect the hinted passes damp but, at the default four, do not remove — so it
     is exception E4 of §3.3's canonical list. It is rare on library-produced frames (R5).
3. Assemble: `[embedded old | outside units | embedded in-block]` along the new axis, then fold back.
4. The carriers: `L_{d−1} = _embed(L°_{d−1})`, `R_0 = _embed(R°_0)`.

At `s = 0` every coordinate is outside and step 1 produces exactly `_identity_prefix`'s arrays (ones and zeros; P2),
and the carriers are zero.

### 4.4 The hint in its own coordinates

The hint `Z = (Z_U, Z_G)` is given: Tucker cores `Z_U[i]` `C+(m_i, N_i)` and TT cores `Z_G[i]` `C+(z_i, m_i, z_{i+1})` at
its own ranks `(m, z)` (for a gradient embedded at a frame, `m = nU + nD` and `z = rL + rR` of that frame — nothing
below assumes it). The construction starts from the given cores: the TT tails are squashed to boundary bonds 1 (an
exact rewrite; a T3 sum has boundary bonds 2, which would break the zippers' ones-boundary — P16), then
down-orthogonalized:

```
Z_G        = tt_operations.tt_squash_tails(Z_G)                            # boundary bonds 1, the same tensor
W, Ztt     = t3_orthogonalization.t3_down_orthogonalize_tucker_cores((Z_U, Z_G))     # W_i  C+(w_i, N_i), w_i = min(m_i, N_i), orthonormal rows; Ztt_i C+(z_i, w_i, z_{i+1})
ZL         = tt_orthogonalization.tt_left_orthogonalize(Ztt)               # C+(zl_i, w_i, zl_{i+1}), left-orthonormal (but the last)
ZR         = tt_orthogonalization.tt_right_orthogonalize(Ztt)              # C+(zr_i, w_i, zr_{i+1}), right-orthonormal (but the first)
A_i        = contract('Czn,Cun->Czu', U°_i, W_i)                           # C+(s_i, w_i)       existing string
Lh_i       = contract('Cazb,Czu->Caub', L°_i, A_i)                         # C+(sL_i, w_i, sL_{i+1})   existing string
Rh_i       = contract('Cazb,Czu->Caub', R°_i, A_i)                         # C+(sR_i, w_i, sR_{i+1})
ΩL         = tt_operations.tt_zipper_left_to_right(Lh, ZL)                 # ΩL_k C+(sL_k, zl_k)  X_old's left interface in ZL's coordinates
ΩR         = tt_operations.tt_zipper_right_to_left(Rh, ZR)                 # ΩR_k C+(sR_k, zr_k)
ZZ         = tt_operations.tt_zipper_right_to_left(ZL, ZR)                 # ZZ_k C+(zl_k, zr_k)  the gauge between the two chains
```

The bonds `zl`, `zr` are the chains' own (at most the hint's `z`); every shape below is in the hint's ranks
`(w, zl, zr)`, never in a hint frame's stores. When `Z`'s chains contain `X_old`'s interfaces (§2.3: a gradient
embedded at a frame containing the old frame's subspaces — P3, also for a zero gradient, whose doubled-rank cores
still carry the frame blocks — or any hint with `X_old` added), `A_i`, `ΩL_k`, `ΩR_k` have orthonormal rows; for
any other hint they are contractions (`‖·‖ ≤ 1`, overlaps of orthonormal sets) and the steps remain valid. All
reused functions are stack-aware.

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
`s_i` rows are `A_i`. (Why the full environment and not, for a gradient hint, its Tucker variation `V_i` alone: P7.)

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

### 4.8 The uniform layer's conversion (`UniformManifoldGeometryOps.padded_frame`)

The member (host, eager; `require_concrete_masks` first) does what §3.3 lists:

1. Per element of the flattened stack: `ufv_unstack` for the frames, while `ut3_conversions.ut3_to_t3` returns the
   hint per element.
   - Build the old frame natively from `x_old`'s full `.data` (`ut3_orthogonal_representations`), unpack it with
     `ufv_conversions.ut3frame_to_t3frame`, and unpack the hint — a uniform T3 `.data` — with `ut3_to_t3`; the
     target structure is the element's integer ranks.
   - `ut3_to_t3` gathers through the masks, gappy ones included; the driver's hints from `utv_to_ut3` pad every bond
     to `rL + rR` with concatenated masks.
   - Only real blocks are read, so padding garbage is never read, and each element may have its own ranks.
2. Run `fv_padded_frame(shape, tucker_e, tt_e, …)` on the element, with the geometry's groups and the seed.
3. Pack at **this geometry's** frame padded sizes (`t3frame_to_ut3frame(..., N=, nU=, nD=, rL=, rR=)`) and restack
   (`ufv_stack`).
4. Check that the packed masks equal those built from the integer stores (`ufv_make_frame_masks`) and the geometry's
   (a structural `ValueError` otherwise), and return the frame with those masks.

On identical inputs the real blocks match the ragged construction's to rounding (the same code on the same arrays;
tested at `1e-14`). Across the layers each builds its old frame natively, and the padded frames agree up to gauge and
rounding (§3.3).

### 4.9 Oversampling, truncation, and the pass loop

Exactly §3.5's loop:

- the residual once;
- the hint-free frame at the **working** structure — `working_ranks`, which the convenience paths fill with
  `ranks.oversampled_ranks(·, p = 5)` through `geom.oversampled_ranks`;
- `passes − 1` hinted rebuilds at the working structure;
- the last rebuild at the target structure — the truncation: `padded_frame` is called with the target ranks, and
  the hint is the working frame's gradient embedded as a T3 by `geom_w.tangent_to_t3`.

The working-structure gradient is projected, normed and embedded with the **working** geometry (the uniform geometry
holds its variation masks), and the final gradient with the target one. Every projection receives
`aux=geom.precompute(F)`. `working_ranks=None` (or `p = 0` in the rank function) makes the two structures equal. With
`passes = 1` the result is the one-pass enrichment from the hint-free frame (the downstream W3 analogue, δ-free).

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
(uniform supercore) strings: the uniform member unpacks.

---

## 5. API

Every signature in house style. `C` is the frame stack. **Round 11 (Nick): the frontend/backend split.** A backend
function with a frontend twin keeps a one-line docstring, its shape comments and the terse guards of §12; its twin
carries the description, the doctests and the validation. A backend function without a twin keeps its docstring and
doctests. §5.8 maps every new function to its twin, or says why it has none.

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
recovered (`|⟨u, e⟩| > 1 − 1e-12`). **No frontend twin** (§5.8), so the docstrings, the doctests and the structural
checks stay as S1 committed them (Nick, round 11: nothing is deleted from a function without a twin).

### 5.2 `fv_padded_frame` (`backend/padded_frames.py`, in its `__all__`; I-32, §3.2)

```python
def fv_padded_frame(
        shape:         typ.Sequence[int],                   # len=d, the mode sizes (N0,...,N(d-1))
        tucker_ranks:  typ.Sequence[int],                   # len=d, the target Tucker ranks
        tt_ranks:      typ.Sequence[int],                   # len=d+1, the target TT ranks (boundary 1)
        old_frame:     typ.Optional[typ.Tuple[
            typ.Sequence[NDArray],  # up_tucker_cores. len=d, elm_shape=C+(nU°i, Ni)
            typ.Sequence[NDArray],  # down_tt_cores.   len=d, elm_shape=C+(rL°i, nD°i, rR°(i+1))
            typ.Sequence[NDArray],  # left_tt_cores.   len=d, elm_shape=C+(rL°i, nU°i, rL°(i+1))
            typ.Sequence[NDArray],  # right_tt_cores.  len=d, elm_shape=C+(rR°i, nU°i, rR°(i+1))
        ]] = None,                                          # the orthonormal frame of X_old; None = the zero tensor
        hint:          typ.Optional[typ.Tuple[
            typ.Sequence[NDArray],  # tucker_cores. len=d, elm_shape=C+(mi, Ni)       -- the hint's own Tucker ranks m
            typ.Sequence[NDArray],  # tt_cores.     len=d, elm_shape=C+(zi, mi, z(i+1)) -- its own TT ranks z
        ]] = None,                                          # any ragged T3 at the mode sizes `shape`; None = hint-free
        groups:        typ.Tuple[typ.Tuple[int, ...], ...] = (),  # sharing partition (canonical_groups); () = unshared
        seed:          int = ZERO_FRAME_SEED,               # the hint-free draws' seed: a convention, tunable per call

        stack_shape:   typ.Tuple[int, ...] = (),            # array-free calls only (no old_frame, no hint): the stack C
        use_jax:       bool = False,                        # array-free calls only: no array inputs, so the flag chooses the output type
) -> typ.Tuple[
    typ.Tuple[NDArray, ...],  # up:    U°_i in the leading rows, then the completion.  len=d, elm_shape=C+(nUi, Ni)
    typ.Tuple[NDArray, ...],  # down:  O°_i embedded, then the completion.             len=d, elm_shape=C+(rLi, nDi, rR(i+1))
    typ.Tuple[NDArray, ...],  # left:  L°_k embedded, then the completion; L[d-1] = L°[d-1] embedded (the carrier)
    typ.Tuple[NDArray, ...],  # right: R°_k embedded, then the completion; R[0] = R°[0] embedded (the carrier)
]:                            # stores (nU, nD, rL, rR) = ranks.compute_orthogonal_representation_ranks(shape, tucker_ranks, tt_ranks)
```

The body begins in the gold-standard way:

- the stores come from `compute_orthogonal_representation_ranks`;
- the old frame's stores, the hint's ranks and mode sizes, and the stack are read inline from array shapes;
- only the four kept guards of §3.2 sit inline (terse; §12), with no helper and no dict;
- `use_jax = tree_contains_jax((old_frame, hint))`, and `xnp` is chosen from it, unless the call is array-free.

**Docstring (round 11):** one line, naming the twin — `'''The frame of X_old at a zero-padded rank increase: the
old block embedded, the new slots completed hint-free or from the hint (frontend twin: T3Frame.padded).'''`. The
description and the doctests below move to the frontend twin **`T3Frame.padded`** (§5.8), rewritten in frontend
types; the structural checks of §3.2 move with them, except the four kept guards.

The twin's docstring sections:

- what a padded frame is (§2.1), and the two completions (§2.3, §2.5) with the counts;
- "structural, δ-free: no threshold, no content in the point";
- the hint as an ambient T3, with examples (a gradient via the geometry's `tangent_to_t3`, another direction, a sum of
  hints), and the cost scaling with the hint's ranks;
- the numerical preconditions — ORTH for the old frame; TIED with `sharing`, the old block read from each group's
  first mode — which the twin checks in safe mode, and the caveat on the hint (§2.3: exact when its chains contain
  `X_old`'s interfaces; add `X_old` to an arbitrary hint to make it so), which is documented, not checked;
- stacking: `C` vectorized; one hint per frame;
- backend inferred from the arrays, or the array-free convention; jit-traceable with the int sequences static; the
  array-free `s = 0` call reproduces `fv_zero_point_frame` on the frame of `t3_zeros(shape, tucker_ranks, tt_ranks)`.

**Where the recipe for raw uniform `.data` lives** (no uniform twin, item 7; round 12, review r8 N4). It is
backend-user documentation, read by someone holding raw `.data`, so it does not move to a frontend twin. It lives in
`docs/contributor/padded_frame_internals.md`, in the section "The uniform conversion". The backend member
`UniformManifoldGeometryOps.padded_frame` cites it from its docstring, whatever that docstring's length after the
twin-set decision. Any frontend twin links to it rather than carrying it. Per stack element:

  1. take the element's target ranks as integers (e.g. from the new point's masks, `masks[0].sum(-1)`,
     `masks[1].sum(-1)`);
  2. build the old point's frame natively (`ut3_orthogonal_representations`) and unpack it
     (`ufv_conversions.ut3frame_to_t3frame`);
  3. unpack the hint (`ut3_conversions.ut3_to_t3`);
  4. call `fv_padded_frame(shape, tucker_e, tt_e, …)`;
  5. repack with `ufv_conversions.t3frame_to_ut3frame(..., N=, nU=, nD=, rL=, rR=)` at the wanted padded sizes, and
     build the masks from the integer stores with `ufv_make_frame_masks`.

  `UniformManifoldGeometryOps.padded_frame` does exactly this.

The twin's doctests (moved from the backend, in frontend types):

- (a) the `s = 0` identity — `T3Frame.padded(shape, tucker_ranks, tt_ranks)` against `fv_zero_point_frame` (imported
  from the backend: it has no twin) on the frame of `TuckerTensorTrain.zeros(shape, tucker_ranks, tt_ranks)`: the
  coordinate cores and zero carriers bitwise (`np.array_equal`), the Haar factors to `1e-14`;
- (b) a `resize`d point with `old_frame=T3Frame.from_t3(x_old)`: `is_orthogonal()`, `to_dense()` equals `X_old`'s
  within `1e-13` relative, the old block embedded as a copy, bitwise (`np.array_equal(U[i][:s_i], U°_i)`) — a
  structural fact;
- (c) a hint embedded from a gradient (`MANIFOLD.randn(frame).to_t3()`): still (b);
- (d) an arbitrary T3 hint (`TuckerTensorTrain.randn(...)`): still (b).

`fv_zero_point_frame` is **not** changed: S2′ returns `fv_operations` to its pre-arc form, so S2's "See also" line in
its docstring goes too (round 10).

### 5.3 No uniform twin (item 7)

There is no `ufv_padded_frame`. The uniform geometry's `padded_frame` member converts explicitly (§3.3, §4.8), and
its docstring says in one line: "built through the ragged constructor `fv_padded_frame`, host-side and eager (it
reads the masks)", and cites the raw-`.data` recipe in the internals page (round 12, §5.2). `ufv_operations` is
unchanged.

### 5.4 Geometry members (`backend/geometry.py`), and the rank function

On `ManifoldGeometryOps` and `UniformManifoldGeometryOps` only (the corewise geometries do not define them, §3.4):

```python
    def tangent_to_t3(
            self,
            frame:    typ.Tuple,   # a frame of this layer
            tangent:  typ.Any,     # a tangent at frame, as project returns it (ragged variations / bare variation supercore pair)
    ) -> typ.Tuple:                # this layer's T3 .data of the same tensor: ragged (tucker_cores, tt_cores) via tv_to_t3; uniform UT3 .data via utv_to_ut3

    def padded_frame(
            self,
            shape:         typ.Sequence[int],          # len=d, the mode sizes
            tucker_ranks:  typ.Union[
                typ.Sequence[int],  # (n0,...,n(d-1)), the target Tucker ranks
                NDArray,            # dtype=int, shape=(d,) + stack_shape (uniform varying-rank stacks; host)
            ],
            tt_ranks:      typ.Union[
                typ.Sequence[int],  # (1,r1,...,r(d-1),1), the target TT ranks
                NDArray,            # dtype=int, shape=(d+1,) + stack_shape (uniform varying-rank stacks; host)
            ],
            x_old:         typ.Optional[typ.Tuple],    # the old POINT (ragged cores / uniform full .data); its frame is built here; None = the zero tensor
            hint:          typ.Optional[typ.Tuple],    # this layer's T3 .data (any ranks, stack C), e.g. tangent_to_t3(F, g); or None
            seed:          typ.Optional[int] = None,   # None = fv_operations.ZERO_FRAME_SEED

            stack_shape:   typ.Tuple[int, ...] = (),   # array-free calls only (x_old and hint both None): the stack C
            use_jax:       bool = False,               # array-free calls only: the output type
    ) -> typ.Tuple:                                    # the padded frame (this layer's frame type; uniform: this geometry's frame masks)

    def padded_structures(
            self,
            x_cores:        typ.Tuple,                  # the point the start is for (the new level's x0, or any point for the cone)
            x_old:          typ.Optional[typ.Tuple],    # the old point: ragged cores pair / uniform full .data (its own masks); None = the zero tensor
            working_ranks:  typ.Optional[typ.Tuple[
                typ.Union[typ.Sequence[int], NDArray],  # tucker: (n0,...,n(d-1)), or dtype=int, shape=(d,) + stack_shape
                typ.Union[typ.Sequence[int], NDArray],  # tt:     (1,r1,...,1),    or dtype=int, shape=(d+1,) + stack_shape
            ]] = None,                                  # the ranks to iterate at; None = x_cores' ranks
    ) -> typ.Tuple[
        typ.Tuple[int, ...],      # shape.          len=d
        typ.Tuple[int, ...],      # stack_shape.    C, x_cores' stack (for the array-free padded_frame call)
        typ.Tuple[
            typ.Union[typ.Sequence[int], NDArray],  # target tucker ranks: len=d, or dtype=int, shape=(d,) + stack_shape
            typ.Union[typ.Sequence[int], NDArray],  # target tt ranks:     len=d+1, or dtype=int, shape=(d+1,) + stack_shape
        ],                        # read from x_cores (uniform: from the held masks)
        typ.Tuple[
            typ.Union[typ.Sequence[int], NDArray],  # working tucker ranks (as the target's)
            typ.Union[typ.Sequence[int], NDArray],  # working tt ranks
        ],                        # the target when working_ranks is None
    ]:                            # INTEGERS ONLY — never a frame, never an orthogonalized zero tensor

    def with_ranks(
            self,
            tucker_ranks:  typ.Union[
                typ.Sequence[int],  # (n0,...,n(d-1))
                NDArray,            # dtype=int, shape=(d,) + stack_shape (uniform varying-rank stacks; host)
            ],
            tt_ranks:      typ.Union[
                typ.Sequence[int],  # (1,r1,...,r(d-1),1)
                NDArray,            # dtype=int, shape=(d+1,) + stack_shape (uniform varying-rank stacks; host)
            ],
    ) -> typ.Any:                                      # this geometry at those ranks: self (ragged); a uniform geometry built from ints and masks

    def oversampled_ranks(
            self,
            x_cores:  typ.Tuple,   # an unstacked point of this layer
            p:        int,         # the rank increment, p >= 0
    ) -> typ.Tuple[
        typ.Tuple[int, ...],       # tucker ranks
        typ.Tuple[int, ...],       # tt ranks
    ]:                             # ranks.oversampled_ranks(shape, *x_cores' ranks, p, the geometry's sharing)

    def require_frame_structure(
            self,
            x_cores:  typ.Tuple,   # the point
            frame:    typ.Tuple,   # a frame claimed to be at x_cores' structure (this layer's frame type)
    ) -> None:                     # structural ValueError naming the mismatching store; both modes
```

**Round 11.** `padded_structures`' checks — an `x_old` of another layer, shape or stack (or a bare uniform supercore
pair), `x_old`'s stores above `x_cores`', `working_ranks` of the wrong length or below the target (§3.4) — move to
the frontend callers, with explicit messages (§5.8). The member keeps none: the one violation that would be silent
(stores above the target) is the constructor's kept count guard. The uniform `padded_frame` keeps its mask check (a
kept guard, §12: a mask drift would be silent). Docstrings are one line each: `padded_frame`'s twins are the
frontend geometries' `padded_frame`, and `tangent_to_t3`'s are the existing `T3Tangent.to_t3` / `UT3Tangent.to_ut3`.
The other four have no twin (§5.8) and keep their docstrings. The `Geometry` protocol docstring
(`backend/optimizers.py:78-85`) lists the six as optional members beside `zero_point_frame`, with what their absence
means.

**`backend/ranks.py`** (in `__all__`):

```python
def oversampled_ranks(
        shape:         typ.Sequence[int],             # (N0, ..., N(d-1))
        tucker_ranks:  typ.Union[
            typ.Sequence[int],  # (n0, ..., n(d-1))
            NDArray,            # dtype=int, shape=(d,) + stack_shape (per element; round 12)
        ],
        tt_ranks:      typ.Union[
            typ.Sequence[int],  # (1, r1, ..., r(d-1), 1)
            NDArray,            # dtype=int, shape=(d+1,) + stack_shape (per element; round 12)
        ],
        p:             int,                           # the rank increment, p >= 0 (structural)
        sharing:       typ.Optional[typ.Sequence] = None,  # len=d group labels; None = unshared
) -> typ.Tuple[
    typ.Union[typ.Tuple[int, ...], NDArray],  # tucker ranks: max(input, minimal(+p)); minimal for a minimal input
    typ.Union[typ.Tuple[int, ...], NDArray],  # tt ranks:     the same; boundary bonds stay 1
]:                                            # sequences in -> int tuples out; arrays in -> host int arrays out
```

**Round 12 (review r8, MINOR 1).** The rank function also accepts per-element int arrays, `(d,)+C` and `(d+1,)+C`, as
`compute_minimal_ranks` does, and returns host int arrays then. The `+p` and the elementwise maximum are array
operations, and `compute_minimal_ranks` already takes the arrays. The committed S1 version accepts sequences only
("reduce per element"); S2′ widens it.

The one home of "+p on every Tucker rank and interior bond, structurally minimal, never below the input" (§3.5).
**Round 11:** a one-line docstring; it keeps its terse `p` and length checks (as `compute_minimal_ranks` does). Its
description and doctests (S1 committed them in the backend) move to the frontend twin: `p = 0` returns a minimal
input unchanged; the output is never below the input; with sharing a group keeps one rank. The twin checks `p` (a
non-negative int) and `sharing` with explicit messages, and it moves from S9 to S2′. It sits beside
`TuckerTensorTrain.continuation_ranks` (`tucker_tensor_train.py:4398`):

```python
    def oversampled_ranks(
            self:     'TuckerTensorTrain',
            p:        int,                            # the rank increment, p >= 0
            sharing:  typ.Sequence = None,            # len=d group labels (None = unshared)
    ) -> Tuple[
        Tuple[int, ...],  # new Tucker ranks
        Tuple[int, ...],  # new TT ranks
    ]:
```

Its uniform twin (round 12, review r8 MINOR 1) mirrors `minimal_ranks` (`uniform_tucker_tensor_train.py:713`): per
element, host numpy, beside it. It makes the same explicit checks and has one doctest: a varying-rank stack's result,
element by element, equals the ragged twin on each unstacked element.

```python
    def oversampled_ranks(
            self:     'UniformTuckerTensorTrain',
            p:        int,                            # the rank increment, p >= 0
            sharing:  typ.Sequence = None,            # len=d group labels (None = unshared)
    ) -> Tuple[
        NDArray,  # new Tucker ranks. dtype=int, shape=(d,) + stack_shape (HOST numpy)
        NDArray,  # new TT ranks.     dtype=int, shape=(d+1,) + stack_shape (HOST numpy)
    ]:                # ranks.oversampled_ranks(self.shape, self.tucker_ranks, self.tt_ranks, p, sharing)
```

### 5.5 `backend/optimizers.py`

```python
PADDED_START_PASSES     = 4   # newton_cg(padded_from=...) / padded_start: gradient evaluations (I-13; P4, downstream)
CONE_REFERENCE_PASSES   = 10  # g0_reference='cone' (I-13; P12; downstream within 2.4 % of the dense frame value at ranks >= 6)
PADDED_FRAME_OVERSAMPLE = 5   # the p the convenience paths pass to geom.oversampled_ranks (I-13; P5, P12; e01 r = 3 stalled at p = 0)

@dc.dataclass(frozen=True, eq=False)
class PaddedStart:
    """The result of :py:func:`padded_start`."""
    frame:   typ.Any                        # the padded frame at x_cores' structure; with x_old=None the cone frame at zero
    gnorms:  typ.Tuple[typ.Any, ...]        # ‖P_T ∇f‖ at each pass's frame (working structure: not comparable to gnorm); shape-C arrays when stacked
    gnorm:   typ.Optional[typ.Any]          # ‖P_T ∇f‖ at the returned frame (None when final_gradient=False); the cone reference iff x_old is None

def padded_start(
        problem:         Problem,                         # the level's Problem (manifold geometry)
        x_cores:         Cores,                           # the point the start is for: the new level's x0 (the same tensor as x_old), or any point (x_old=None)
        x_old:           typ.Optional[typ.Any] = None,    # the old point: ragged cores pair / uniform full .data; None = the zero tensor (the cone)
        working_ranks:   typ.Optional[typ.Tuple] = None,  # ranks to iterate at; None = x_cores' ranks (no oversampling); fill with geom.oversampled_ranks
        passes:          int = PADDED_START_PASSES,       # gradient evaluations (>= 1); the last construction truncates to x_cores' structure
        seed:            typ.Optional[int] = None,        # the hint-free start's seed; None = fv_operations.ZERO_FRAME_SEED (via the geometry)
        final_gradient:  bool = True,                     # evaluate gnorm at the returned frame (one more sweep + transpose)
) -> PaddedStart:
```

**Docstring (round 11): one line.** It keeps one guard, the members check ("padded starts need a manifold geometry"):
without it, duck typing would fail obscurely. Its description and doctest move to the frontend twin (§5.7). The
description covers: §2.6 in brief; what `gnorm` is in each mode (§3.5); the caveat on `x_old` (§3.2: numerically zero
singular values and structural slack give arbitrary old directions); cost (`passes (+1)` frame sweeps and
transposes, one residual); the misfit only; stacked `C` supported, per-element norms; the members requirement.
The doctest, written on the twin in its types (`frame, info = padded_start(...)`): on a seeded `apply` problem with
`x0 = x_old.resize(...)`, `frame` is a frame of `x0` (the base point's dense difference below 1e-13 relative),
`len(info['gnorms']) == 4`, and `info['gnorm'] > 1.2 * g_lib` with `g_lib` the library frame's gradient norm (a wide
margin, so the doctest is stable across numpy generations).

`Problem.local_model` gains one keyword (last). Its description moves to the frontend twin, the `frame=` keyword of
the `fitting` model factories (§5.8). The backend keeps the parameter comment and the `require_frame_structure` guard:

```python
            frame:              typ.Optional[typ.Any] = None,   # an ORTHONORMAL frame of x_cores' TENSOR at its structure (e.g. a padded frame); None = geom.frame(x_cores)
```

`newton_cg` gains six keywords, **appended keyword-only** after its last parameter `callback` (a `*` before the block),
so no existing positional call changes meaning (I-40):

```python
        callback:               typ.Optional[typ.Callable] = None,  # (existing, unchanged — the last positional-or-keyword parameter)

        *,                                                     # the new block is keyword-only
        start_frame:            typ.Optional[typ.Any] = None,   # iteration 0's frame: an orthonormal frame of x0's tensor at x0's structure; None = geom.frame(x0)
        padded_from:            typ.Optional[typ.Any] = None,   # the OLD POINT x_old (ragged cores / uniform full .data), the same tensor as x0: iteration 0's frame = padded_start(...).frame
        padded_passes:          typ.Optional[int] = None,       # padded_from only: passes; None = PADDED_START_PASSES
        padded_seed:            typ.Optional[int] = None,       # padded_from only: the hint-free start's seed; None = the geometry's default
        cone_passes:            typ.Optional[int] = None,       # g0_reference='cone' only: passes; None = CONE_REFERENCE_PASSES
        cone_seed:              typ.Optional[int] = None,       # g0_reference='cone' only: the seed; None = the geometry's default
```

- **`stats`.** It gains `'start_frame_source'` (a label: `'given'` / `'padded'`) and, with `padded_from`,
  `'padded_start_gnorms'` — **only on runs that use them**. `NewtonInfo` and the history rows are unchanged.
- **`gradient_descent`** gains `cone_passes` and `cone_seed`, appended keyword-only after its last parameter
  (`g0_reference`), for the same reason.
- **Private changes.**
  - `_G0_REFERENCES = ('zero', 'warm_start', 'cone')`.
  - `_resolve_g0_reference(..., cone_passes=None, cone_seed=None)` returns `('cone', value)`.
  - `_cone_reference_gnorm(problem, x0, cone_passes, cone_seed) -> float`.
  - `_prepare_jit_frame(use_jit, frame)` uses `common.partition_static` / `rebuild_static` around `tree_to_jax`; the
    same conversion is applied to `padded_from`.
- **Public names.** New in `__all__`: `PaddedStart` and `padded_start`. The constants are module-level, like
  `ZERO_FRAME_SEED`. There is no options class (I-3).

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
        x0:          Point,          # the point the start is for (the new level's start), stack C allowed
        x_old:       typ.Optional[Point] = None,   # the old point (same layer, the same tensor as x0); None = the zero tensor of x0's ranks (the cone)

        *,
        order:          typ.Optional[int] = None,       # derivative kinds
        weight:         typ.Optional[typ.Any] = None,   # residual weight ω
        chunk_size:     typ.Any = 'auto',               # as newton_cg (a memory chunk of the probe-derivative assembly, not a rank)
        working_ranks:  typ.Optional[typ.Tuple] = None, # ranks to iterate at — x0.oversampled_ranks(5) on either layer (round 12); None = x0's ranks
        passes:         int = bopt.PADDED_START_PASSES,
        seed:           typ.Optional[int] = None,
) -> typ.Tuple[
    typ.Any,   # T3Frame / UT3Frame: x_old given -> a padded frame of x0's tensor at x0's (reduced) structure;
               #                     x_old=None  -> the cone frame at the zero tensor of x0's ranks (a start_frame for a zero x0 only)
    dict,      # {'gnorms': [...], 'gnorm': ...}: floats unstacked, shape-C arrays stacked; 'gnorm' is the cone reference iff x_old is None
]:
```

- **Reductions and checks.** `x_old` is reduced structurally first (`_ragged_minimal` / `uniform_minimal`,
  sharing-aware) and `x0` as `least_squares_problem` always does. The safe-mode checks are §3.6's: same tensor, TIED.
- **`newton_cg`** mirrors the six keywords, **appended keyword-only after its last parameter `tr_radius0`** (a `*`
  before the block, as in the backend): `padded_from` takes a `TuckerTensorTrain` / `UniformTuckerTensorTrain`,
  forwarded as `.data` after the reduction; `start_frame` takes a `T3Frame` / `UT3Frame`, forwarded as `.data`. They
  are **forwarded only when set**, with a one-line code comment at the forwarding ("forwarded only when set:
  G-nonbreaking") so that a later cleanup does not turn it into always-forwarding and break the pre-existing
  forwarding test.
- **`gradient_descent`** reaches `cone_passes` / `cone_seed` through its `**kwargs` (docstring line only).
- **Exports.** Neither the frontend `padded_start` nor anything else new is re-exported at the package root (as
  `least_squares_problem`).
- **The twin of the backend `padded_start`** (round 11): it carries the description and the doctest moved from §5.5,
  in frontend types (`frame, info = padded_start(...)`), and it runs the pair checks through the shared helper of
  §5.8.
- **The constructor's twin** is `T3Frame.padded` (§5.8; Nick, round 11). It supersedes O-6 and I-22's "no frontend
  constructor".


### 5.8 Frontend twins (Nick, round 11)

**The rule** (§12 states it in full). Every new public backend function or keyword that a frontend user would call has
a frontend twin: a method of the frontend class it acts on, or a standalone function in the frontend module. The twin
is written as the gold standard writes `TuckerTensorTrain.resize`, `__add__` and `t3svd`: validate, make one or two
backend calls, rewrap. It carries the detailed description, the doctests and the argument validation — explicit
messages that name the fix, and the safe-mode numerical checks. Its backend function keeps a one-line docstring, its
shape comments and only the terse guards of §12. A backend function **without** a twin keeps its docstring and
doctests: nothing is deleted or converted (Nick's correction).

**The map.**

| backend (new) | frontend twin | the twin's backend call and rewrap | moves to the twin | the backend keeps |
|---|---|---|---|---|
| `padded_frames.fv_padded_frame` | **`T3Frame.padded`** (static; `frame_variations_format.py`, beside `random_orthogonal`) | `T3Frame(*fv_padded_frame(shape, tucker_ranks, tt_ranks, old_frame.data, hint.data, canonical_groups(sharing, shape), seed, stack_shape, use_jax))`, with `None` passed through | §5.2's description and doctests (a)–(d); §3.2's structural checks, explicit; ORTH and TIED of `old_frame` in safe mode | one line; the four guards of §3.2 |
| geometry member `padded_frame` | **`ManifoldGeometry.padded_frame`**, **`UniformManifoldGeometry.padded_frame`**, **`SharedGeometry.padded_frame`**: the padded frame **at `x0`** (the analog of `frame(x0)`), built from `x_old` | ragged: `T3Frame.padded(x0.shape, x0.tucker_ranks, x0.tt_ranks, old_frame=self.frame(x_old), hint=hint, seed=seed)`; uniform: `UniformManifoldGeometryOps.from_point(x0.data).padded_frame(...)`, rewrapped as a `UT3Frame`; shared: `x_old` tied first (as `SharedGeometry.frame` ties), then the base layer's call with the sharing | the uniform conversion's description (the raw-`.data` recipe stays backend documentation, §5.2); the pair checks moved from `padded_structures` (layer, shape, stack, stores); the same tensor in safe mode; one doctest each on the ragged and the uniform twin | one line; the uniform mask check (§3.3 step 3) |
| geometry member `tangent_to_t3` | the existing **`T3Tangent.to_t3`** / **`UT3Tangent.to_ut3`** (the same backend calls) | — | — | one line |
| `ranks.oversampled_ranks` | **`TuckerTensorTrain.oversampled_ranks`** and, per element, **`UniformTuckerTensorTrain.oversampled_ranks`** (round 12) (§5.4) | `ranks.oversampled_ranks(self.shape, self.tucker_ranks, self.tt_ranks, p, sharing)` | the description and the three doctests (ragged); one new doctest (uniform); `p` and `sharing`, explicit | one line; its terse `p` and length checks |
| `optimizers.padded_start`, `PaddedStart` | the frontend **`optimizers.padded_start`** (§5.7) | `PaddedStart` becomes `(T3Frame / UT3Frame, {'gnorms', 'gnorm'})` | §5.5's description and doctest; the pair checks; the reductions and safe-mode checks (already there) | one line; the members guard |
| `Problem.local_model(frame=)` | **`frame=None`** on the six `fitting` model factories (`apply_model`, `entries_model`, `probe_model` and the three `*_derivatives_model`), through `_ragged_frame` / `_uniform_model` | the given frame replaces `geometry.frame(x)`; `LocalModel` unchanged | the keyword's description; structure (explicit); ORTH and same tensor (safe mode, as `_require_at_frame`, `fitting.py:108-126`) | the parameter comment; `require_frame_structure` |
| `newton_cg`'s six keywords, `gradient_descent`'s two | the frontend **`newton_cg`** / **`gradient_descent`** (§3.6, §5.7) | forwarded only when set | their descriptions; §3.6's frontend checks | parameter comments; the keyword-contract guards (§12) |

**No twin, and why.** These are internal pieces; each keeps its docstring and doctests.

- **`linalg.pinned_completion`, `linalg.pinned_leading_columns`** — factorization primitives with no T3 semantics.
  The frontend has no linear-algebra layer: `pad_safe_svd` and `damped_pseudo_inverse` have no twin either. Their S1
  docstrings, doctests and structural checks stay.
- **The members `padded_structures`, `with_ranks`, `oversampled_ranks` and `require_frame_structure`** — the T3-free
  driver's plumbing, like the existing `zero_point_frame` member, which has no frontend twin either.
  - The frontend reads integers off its own objects (`structure`), so `padded_structures` needs no twin; its checks
    move to the frontend callers.
  - The frontend geometries are stateless singletons that hold no structure to re-rank, so `with_ranks` needs none.
  - The rank rule's twins are `TuckerTensorTrain.oversampled_ranks` and `UniformTuckerTensorTrain.oversampled_ranks`,
    which cover the `oversampled_ranks` member.
  - `require_frame_structure` is a kept guard; its explicit counterpart is the frontend check on `start_frame` and
    `frame=`.
- **`UT3Frame`** gets no `padded` method (round 12, review r8 N1). A uniform frame is useful only at the padded sizes
  and masks of the point it serves (§3.3 step 3), and only a point or a geometry supplies those. A standalone
  `UT3Frame.padded` would have to take five padded sizes as integers, or default them to sizes that need not fit the
  point; either way it would be a template in disguise. The uniform routes go through a point (§5.8's map).
- **Private pieces and constants**: `_cone_reference_gnorm`, `_prepare_jit_frame`, the `'cone'` branch of
  `_resolve_g0_reference`, `_REFERENCE_NAMES['cone']`, and the three module constants (the frontend uses them as
  defaults).

**The new frontend signatures.**

```python
# t3toolbox/frame_variations_format.py, class T3Frame
    @staticmethod
    def padded(
            shape:         typ.Sequence[int],                            # len=d, the mode sizes
            tucker_ranks:  typ.Sequence[int],                            # len=d, the target Tucker ranks
            tt_ranks:      typ.Sequence[int],                            # len=d+1, the target TT ranks (boundary 1)
            old_frame:     typ.Optional['T3Frame'] = None,               # an orthonormal frame of X_old; None = the zero tensor
            hint:          typ.Optional['t3.TuckerTensorTrain'] = None,  # any T3 at mode sizes `shape`, stack C; None = hint-free
            sharing:       typ.Optional[typ.Sequence] = None,            # len=d group labels (None = unshared)
            seed:          int = fv_operations.ZERO_FRAME_SEED,          # the hint-free draws' seed
            stack_shape:   typ.Tuple[int, ...] = (),                     # array-free calls only (no old_frame, no hint): the stack C
            use_jax:       bool = False,                                 # array-free calls only
    ) -> 'T3Frame':                                                      # a padded frame of X_old at the target structure

# t3toolbox/manifold.py, class ManifoldGeometry; likewise uniform_manifold.UniformManifoldGeometry
# (UniformTuckerTensorTrain in, UT3Frame out) and shared_geometry.SharedGeometry over either
    def padded_frame(
            self,
            x0:     t3.TuckerTensorTrain,                        # the point the frame is for: X_old's tensor at the target structure
            x_old:  t3.TuckerTensorTrain,                        # the old point: the same tensor at lower ranks
            hint:   typ.Optional[t3.TuckerTensorTrain] = None,   # any T3 at x0's mode sizes and stack; None = hint-free
            seed:   typ.Optional[int] = None,                    # None = fv_operations.ZERO_FRAME_SEED
    ) -> bvf.T3Frame:                                            # a padded frame of x0's tensor at x0's structure

# t3toolbox/fitting.py, each of the six model factories, appended keyword-only
            *,
            frame:  typ.Optional[typ.Any] = None,   # T3Frame / UT3Frame: an orthonormal frame of x's tensor at x's structure; None = geometry.frame(x)
```

**What the twins check** (explicit messages that name the fix; numerical checks only when `safety.checks_active`).

- **`T3Frame.padded`.**
  - Structural: lengths and the boundary TT ranks; `old_frame` of mode sizes `shape`; its stores within the
    target's (from `ranks.compute_orthogonal_representation_ranks`); `hint` a `TuckerTensorTrain` — consistent by
    construction (`TuckerTensorTrain.validate`) — of mode sizes `shape` and of `old_frame`'s stack, the message naming
    the caller's reduction for any other stack; the array-free arguments only when there is no array;
    `canonical_groups(sharing, shape)`, and equal `(Tucker rank, size)` within each group.
  - Safe mode: `old_frame` ORTH (`is_orthogonal` at `safety.effective_rtol`); with `sharing`, TIED — the old group
    factors equal.
  - A wrong-type `old_frame` or `hint` is a `TypeError`.
- **The geometry twins.** The pair checks — layer, shape, stack, and `x_old`'s stores within `x0`'s; on the uniform
  layer also an equal padded `N`, which `allclose`'s subtraction needs (§3.6) — live in one private helper per layer,
  `_require_padded_pair` in `manifold.py` / `uniform_manifold.py`. The frontend
  `padded_start` and `newton_cg(padded_from=)` use the same helper. In safe mode the twins also check the same
  tensor (`x0.allclose(x_old)` at `safety.effective_rtol`, §3.6's measurement). The shared twin does not check TIED:
  it ties `x_old` first, silently, as `SharedGeometry.frame` ties its point. `x_old` is not reduced here: the
  docstring carries §3.2's slack caveat, and the optimizers reduce `x_old` (§3.6).
- **`frame=` on the factories.** The structure of `x` (explicit message); ORTH (`_require_orthogonal_frame`); the
  same tensor (`frame.to_t3()` / `.to_ut3()` `allclose` `x`, safe mode).

**Doctests that move** (T14). `fv_padded_frame`'s four move to `T3Frame.padded`. `ranks.oversampled_ranks`' move to
`TuckerTensorTrain.oversampled_ranks`. The backend `padded_start`'s one moves to the frontend `padded_start`. There
is one new doctest on `UniformTuckerTensorTrain.oversampled_ranks` (round 12), and one on each manifold geometry's `padded_frame`: the frame is orthonormal and `to_dense()` equals `x0`'s
within `1e-13` relative. The linalg pair keep theirs.

**Slices** (§8). Each construction-layer twin lands with the code it documents: `T3Frame.padded` and the two
`oversampled_ranks` twins in S2′, `UniformManifoldGeometry.padded_frame` in S4, `ManifoldGeometry` /
`SharedGeometry.padded_frame` in S5. The optimizer layer's twins — the frontend `padded_start`, the `newton_cg`
keywords, the factories' `frame=` — land in S9, the frontend slice, as planned.

---

## 6. Tests

Numpy-only unless the test is about jax (house convention). New files: `tests/backend/test_padded_frame.py` (T1–T4)
and `tests/backend/test_padded_start.py` (T5–T6); the rest extend the named files **by additions only** (G-nonbreaking).

**Bitwise vs tolerance** (Nick's rule, round 6):

- **Bitwise (`np.array_equal`) only for structural facts.** These are the embedded old block (a copy), zero carriers,
  zero padding, masks (equal to the geometry's frame masks), and the coordinate completions of the
  hint-free start (outside-first unit vectors, identity prefixes). Two invariance-of-the-default-path claims belong
  here too: the G-bit gate that default runs are unchanged, and T6(a) / T7(a), that passing the library's own frame
  explicitly takes the same code path. They are the G-bit kind of claim, so they stay bitwise.
- **Explicit stated tolerances for everything that passed through floating-point arithmetic** (never `np.allclose`
  defaults):
  - orthogonality residual `≤ 1e-14` (float64);
  - relative reconstruction `≤ 1e-13`;
  - computed arrays, such as Haar factors or the uniform member's real blocks against the ragged construction:
    `≤ 1e-14` absolute;
  - the safety rtol where a safe-mode check is exercised.

  This covers the `s = 0` identity with `fv_zero_point_frame`, whose Haar factors the library itself calls
  "reproducible to ~1e-15 across BLAS builds", not bitwise.
- **Frame claims are gauge-invariant.** They use assembled projectors, plus the dense `P_{T_F} Y` of a fixed random
  `Y`:
  - `U_iᵀU_i` per Tucker factor;
  - per bond, the column-space projector of the embedded chain unfolding — the dense left interface `Φ_k Φ_kᵀ` and
    right interface `Ψ_k Ψ_kᵀ`;
  - per mode, the down core's row-space projector in the bond space (`O_i` as an `(rL·rR) × nD` matrix).

"Valid frame" below means: orthogonality residual `≤ 1e-14`; both base-point reconstructions within `1e-13` relative of
`X_old`; the old block present bitwise as a copy; the shapes of the target stores
(`ranks.compute_orthogonal_representation_ranks(shape, tucker_ranks, tt_ranks)`).

| # | file | asserts |
|---|---|---|
| T1 | `test_linalg.py` | (a) `pinned_completion`: orthonormal and `⊥ pinned` (`≤ 1e-14`) for random, rank-deficient (duplicated columns, seeds inside `span(pinned)`), all-zero and too-few (`m < k`) seeds; `k = 0` → `(n, 0)` without a factorization; `s + k > n` → `ValueError`; float32 in → float32 out. (b) `s = 0` with the keyed Gaussian seeds equals `_haar_rows(...).T` to `1e-14`. (c) `pinned_leading_columns`: a planted `A = P X + U_t diag(σ) V_tᵀ` with `U_t ⊥ P` and a gap returns `span(U_t[:, :k])` (projector distance `< 1e-12`); rank-deficient `A` is completed (still orthonormal, `⊥ P`, `≤ 1e-14`). (d) `C = (2,)` and `(2, 3)` stacks equal the per-element results to `1e-14`. |
| T2 | `test_padded_frame.py` | hint-free `fv_padded_frame`: (a) `old_frame=None` reproduces `fv_zero_point_frame` — the coordinate cores and zero carriers **bitwise**, the Haar Tucker factors to `1e-14` — `d = 1, 3, 4`, unshared and `groups=((0, 1), (2,))`; (b) a valid frame at `resize`d points (Tucker-only, TT-only and joint increases); (c) the in-block cases (TT-only increase at bond 1; Tucker-only increase at saturated bonds) give valid frames and the structural counts; the in-block columns come from coordinate seeds; two calls agree to `1e-15`; (d) the Tucker completion is nested (`kU → kU + 1`: the leading rows agree to `1e-14`); (e) stacked `C` equals per-element calls to `1e-14`; (f) **sign-gauge covariance** (the property §3.3's cross-layer claim rests on): two sign-gauge-equivalent old frames give hint-free frames with equal assembled projectors (`≤ 1e-12`), an in-block case included (P17). |
| T3 | `test_padded_frame.py` | hinted `fv_padded_frame` (the hint a T3; the gradient hints embedded by `tv_to_t3`, the driver's call shape): (a) valid frames for a gradient hint embedded at the library frame of the padded point, at a padded frame, and a **zero** gradient; (b) the internal coordinates (`_hint_coordinates`): for gradient hints `A_i`, `ΩL_k`, `ΩR_k` have orthonormal rows (`≤ 1e-13`), for an arbitrary T3 hint norms `≤ 1 + 1e-13`; (c) **matrix fixed point** (`d = 2`, Tucker ranks = TT rank): with new slots aligned with the top singular subspaces of the normal part `N` and the hint `P_T G` at that frame, the constructor returns the same subspaces (projector distance `< 1e-10`); (d) the truncation (a hint embedded at the working frame, the target ranks) gives a valid target-structure frame; (e) sharing: one `U` per group (the members' factors bitwise equal — one array, a structural fact), and the group's new rows span the top eigenvectors of the summed projected Grams `Σ_m P⊥ W_mᵀ M_m M_mᵀ W_m P⊥` (projector oracle, `< 1e-10`); (f) stacked `C` equals per-element calls (`≤ 1e-12`); (g) the backend's **kept guards** (round 11, §12) — target ranks whose stores fall below the old frame's (a negative count), an old frame / hint stack mismatch, an array-free argument passed with an array, and (round 12) a sharing group whose members differ in `(Tucker rank, size)` — each a terse `ValueError`; the other structural errors (a hint of other mode sizes or another stack, mismatched `d` / shape) move to the twin's test, T15(b); (h) **dense oracle per step** (the only test that sees a wrong-subspace bug in the sweeps or the down step): with `Z` the dense hint, `Φ_k` / `Ψ_k` the dense left/right bases of the constructed frame and `Φ°`, `Ψ°` the old ones, the new directions equal (projector distance `< 1e-9`, on generic data with gaps and `≥ k` hint directions) — Tucker `i`: the leading left singular vectors of `P_{U°_i}^⊥ Z_(i) (I − E_dᵀE_d)`, `E_d[m] = Σ_ab O°_i[a,m,b] Φ°_i[a] ⊗ Ψ°_{i+1}[b]`; `L_k`: of `(I − PPᵀ) M (I − Ψ°_{k+1}ᵀΨ°_{k+1})`, `M = (Φ_k ⊗ U_k)ᵀ Z_{(0..k ∣ k+1..)}`, `P` the embedded `L°_k`; `R_k`: the mirror image (the content in `U_k ⊗ Ψ_{k+1}`, columns projected off `Φ°_k`, pinned off the embedded `R°_k`); `O_i`: the leading bond-space directions of `Z`'s coordinates in `Φ_i ⊗ U_i ⊗ Ψ_{i+1}` on the new Tucker slots, pinned off the embedded `O°_i` — at `C = ()` and `(2,)`, unshared and shared, `d = 3` and `4`; (i) a `d = 1` hinted call (empty sweeps, zero-size `Ω`) gives a valid frame; (j) **an arbitrary external T3 hint** `H` (a `TuckerTensorTrain.randn` of ranks unrelated to the frame — some below the target's Tucker ranks — not a gradient; and a hint built by `+`, whose boundary bonds 2 the constructor squashes): the result is a valid padded frame at `s > 0`; at `s = 0`, and at `s > 0` with the augmented hint `H + X_old` (§2.3), its new directions capture `H`'s normal content — the dense per-step oracle of (h) with `Z = H` holds to `< 1e-9`, and in the matrix case (`d = 2`) the new slots are the top-`k` singular subspaces of `P_{U°}^⊥ H P_{V°}^⊥` (projector `< 1e-10`); with the raw `H` at `s > 0` the frame is valid and the captured norm `‖P_{T_F} H‖` is at least the library frame's. |
| T4 | `test_padded_frame.py` | the uniform conversion, through **`UniformManifoldGeometryOps.padded_frame`** (no twin, item 7): (a) on identical inputs the real blocks equal `fv_padded_frame`'s to `1e-14` (hint-free and hinted, `s = 0` and `s > 0`, shared; the uniform hint a UT3 — from `from_t3` of a ragged hint, and from `utv_to_ut3` of a gradient, whose bond padding to `rL + rR` comes with its own concatenated masks — unpacked by `ut3_to_t3`; and a **non-minimal** `x_old` and target, which the contract requires to agree too); the member with `x_old=None, hint=None` (array-free) reproduces `ufv_zero_point_frame` on the native frame of `ut3_zeros` at the target ranks — the test builds that frame, for `ufv_zero_point_frame`'s sake only (coordinates and masks bitwise, the Haar factors to `1e-14`); (b) the masks: `out[5]` equals the frame masks of the native sweep at the target (`ut3_orthogonal_representations`), and `ufv_variation_masks(out[5])` equals the geometry's `var_masks` (`np.array_equal`, structural); the padding is exactly zero (bitwise); (c) NaN-filled padding in every input (the hint's included) leaves the real blocks NaN-free and equal to the clean run's to `1e-14`; (d) a stack with per-element ranks equals per-element calls to `1e-14`; (e) the **native** uniform old frame (`ut3_orthogonal_representations(x_old.data)`) is gauge-equivalent to the ragged one — per-family projectors equal to `1e-12` — and a gappy `x_old` (from uniform `+`) gives the same projectors as its prefix-layout twin. |
| T5 | `test_padded_start.py` | backend `padded_start` and the rank function: (a) **matrix oracle** (`d = 2`, `entries` kind with every entry observed, so the ambient gradient is the residual matrix): at `s = 0`, `gnorm → ‖G_r‖` and at `s > 0`, `gnorm → (‖P_{T_X}G‖² + ‖N_{r−s}‖²)^{1/2}` to rtol `1e-8` with `passes=40` and `working_ranks=None` on a planted gap; (b) a seeded regression on c11–c13: the final captured norm exceeds the first pass's by a stated margin (the per-pass sequences of P13 are recorded as observations); (c) the frame is a frame of `x_cores` at its structure (`require_frame_structure` passes, base point within `1e-13` relative); with `x_old=None` its base point is exactly 0 (zero carriers — structural); (d) `passes=1`, `working_ranks=None`, `final_gradient=False` (`gnorm is None`); (e) stacked `C`: per-element `gnorms` equal the unstacked runs to rtol `1e-12`; (f) ragged vs uniform, **gauge-invariant**: `gnorms` and `gnorm` equal to rtol `1e-10`, the frames' per-family projectors and their `P_{T_F} Y` equal to `1e-10` (at `s = 0` and `s > 0`, an in-block case included); (g) the kept members guard — a corewise geometry and a geometry stub without the members raise (terse); `padded_structures`' former checks (an `x_old` of another layer, shape or stack, a bare uniform supercore pair, `x_old`'s stores above `x_cores`', `working_ranks` of the wrong length or below the target) move to T15(b) — and acceptance: a **gappy** uniform `x_old` (from uniform `+`) is accepted and gives the same padded frame, gauge-invariantly, as its prefix-layout twin; (h) `ranks.oversampled_ranks`: per-element int arrays (a varying-rank stack) give, element by element, the per-element calls' results (round 12); `p = 0` returns a minimal input unchanged; the output is never below the input (a non-minimal input keeps its slack); with sharing a group keeps one rank; the frontend `TuckerTensorTrain.oversampled_ranks` and the geometry member agree with it; (i) **`with_ranks`** (Nick, round 10): for uniform geometries built by `from_point` — unstacked, uniform- and varying-rank stacks, a non-minimal point, padded sizes above the ranks, unshared and shared (the groups carried over) — `geom.with_ranks(*the ranks read from geom.masks)` reproduces `geom.masks` and `geom.var_masks` exactly (`np.array_equal`, structural), so a drift of the hand-built masks from the sweep fails; the ragged `with_ranks` returns `self`. |
| T6 | `test_padded_start.py` | `Problem.local_model(frame=)`: (a) `frame=geom.frame(x)` gives a model equal **bitwise** to the default (misfit, gradient, `hvp` on a fixed direction), GN and exact — the default-path claim; (b) at a padded frame: the misfit is bitwise the default's (the residual does not see the frame); `lm.retract` of the zero tangent returns `X_old` (within `1e-13` relative); (c) a wrong-structure frame → `ValueError` under `safety.safe()` and `safety.unsafe()`; a corewise geometry → `ValueError` naming `require_frame_structure`; (d) exact mode at a padded frame: builds, `lift_sigma_min` is at the clip (the new slots' zero singular values), `hvp` symmetric (the existing symmetry check, `1e-12`). |
| T7 | `tests/backend/test_optimizers.py` (new test functions) | `newton_cg` seam: (a) the default `set(stats)` is unchanged (the existing pins at `:477`, `:860` stay as they are); `newton_cg(x0)` and `newton_cg(x0, start_frame=geom.frame(x0))` have bitwise equal histories (the default-path claim) and the second's `stats` differ only by `'start_frame_source': 'given'`; (b) `padded_from=x_old` (`x0 = x_old.resize(...)`): iteration 0's `info.lm.frame` equals `padded_start(problem, x0, x_old, geom.oversampled_ranks(x0, 5)).frame` to `1e-14`; `stats['start_frame_source'] == 'padded'`, `stats['padded_start_gnorms']` has `PADDED_START_PASSES` entries; the run converges (`stop_reason='gradient'`); (c) structural errors: `start_frame` with `padded_from`, `padded_passes` / `padded_seed` without `padded_from`, `cone_passes` / `cone_seed` without `'cone'`, passes `< 1`, a seed outside `[0, 2**32)`, `padded_from` / `start_frame` on corewise; (d) from the first accepted step on, `info.lm.frame` equals `geom.frame(info.x_cores)` to `1e-14` (iteration 0 only); every row at `x0` (rejected trust-region trials included) carries the start frame; (e) `use_jit=True` on numpy inputs with a numpy `start_frame` and a numpy `padded_from`: runs, the converted arrays are jax and uniform masks stay numpy; `bopt._jitted(bopt._cg_core, (4,))._cache_size()` grows by no more than the default run's (the `test_dispatch.py:1194-1207` pattern); (f) shared ragged and uniform geometries: `padded_from` runs, the start frame is tied; (g) `padded_from=x0` itself (zero counts everywhere): the start frame equals the library frame to `1e-14` on each layer (both build `x0`'s frame natively) and the run's objectives equal the default run's to rtol `1e-12`. |
| T8 | `tests/backend/test_zero_reference.py` (new test functions) | `g0_reference='cone'`: (a) `d = 2` oracle: `padded_start(problem, x0, None, working_ranks=None, passes=40).gnorm` equals `‖G_r‖` (the cone value, **not** `‖G_{2r}‖`) to rtol `1e-8`, and the `'cone'` resolution at its defaults (working ranks `oversampled_ranks(x0, 5)`, 10 passes) to 2 %; (b) T3: within `[0.95, 1.10]×` the dense-frame value `‖P_{T_F}G‖` (`F` = frame of `dense_t3svd(G)`, zero carriers) on the two P12 problems; (c) ragged == uniform to rtol `1e-10`; (d) the default is unchanged (`'zero'` recorded); `'cone'` on corewise → `ValueError`; all-zero data → `ValueError`; with `g0norm_newton` it is not computed (spy on `padded_start`); (e) rows record `g0_reference='cone'`, `cg_reference='cone'`; the converged display line reads `g₀ = tangent cone`; (f) `gradient_descent(g0_reference='cone', cone_passes=..., cone_seed=...)`; (g) seeds 0–3 give cone references within 2 % of each other on a gapped problem; (h) the staggered completion (§2.2) is not a fixed point: at `d = 2`, `s = 0`, a frame with `U = u_1`, `V = v_2` captures `‖G_2‖` (to rtol `1e-12`), and one constructor pass from it returns the aligned frame (projector `< 1e-12`), which captures `‖G_1‖`. |
| T9 | `tests/test_optimizers_frontend.py` (new test functions; the pre-existing ones unmodified) | (a) the keyword mirror (`:448`) covers the six keywords with the backend defaults (automatic, unmodified); (b) the pre-existing forwarding test (`:465`) passes **unmodified** (a default call forwards no new keyword); a new test asserts that each new keyword, when set, reaches the backend with its value, `padded_from` and `start_frame` as their reduced `.data`; (c) frontend `padded_start`, ragged and uniform: returns `T3Frame` / `UT3Frame`; with `x_old` given `frame.to_t3()` / `.to_ut3()` is within `1e-13` relative of `x0`, with `x_old=None` it is zero; a `C`-stacked `x0` gives a stacked frame and shape-`C` norms; (d) safe mode: `x0` and `x_old` different tensors → `ValueError` naming the relative difference, skipped under `safety.unsafe()`; a `start_frame` of another tensor → `ValueError`; a non-orthonormal `start_frame` → `ValueError` (ORTH); untied `x_old` on a shared geometry → `ValueError` (TIED); all safe-only; structural (both modes): a `start_frame` not at the reduced `x0`'s structure ("build it at `x0`'s minimal structure"), a frame or old point of the wrong layer, a uniform `x_old` with another padded `N`; accepted: a non-minimal `x0` and a non-minimal `x_old` with `padded_from` (both reduced, the run converges); (e) **the equivalence test**: ragged vs uniform `newton_cg(padded_from=x_old)` on the `x01`-style restart give per-iteration objectives equal to rtol `1e-8` over 10 iterations (the library start: 5.6e-1, `uniform_equivalence_contract.md`). |
| T10 | `tests/test_dispatch.py` (new rows) | numpy in → numpy out, jax in → jax out for the two primitives, `fv_padded_frame`, the uniform member `padded_frame` (jax supercores out, masks numpy — the `ufv_zero_point_frame` rows at `:1898-1912`), the geometry members `tangent_to_t3` (ragged cores / uniform UT3 `.data`, masks numpy), `padded_frame` and `padded_structures` on both layers, and `padded_start` on both layers; `jax.jit(fv_padded_frame, static_argnames=('groups', 'seed'))` (hint-free and hinted) equals eager (rtol `1e-5` float32, `1e-12` x64). |
| T11 | `tests/test_jax_absent.py` (new rows) | with jax hidden: the new functions import and run on numpy; `newton_cg(padded_from=..., use_jit=True)` warns once and runs eager. |
| T12 | `tests/test_contractions_interpreter.py` | `HISTORICAL` gains the twelve strings of §4.10 (each `''`) — the one edit to a pre-existing test file, required by `TestVocabularyCompleteness`; `test_contractions_sharding.py` covers them automatically. |
| T13 | `tests/backend/test_sharing.py` (new test function) | the end-to-end shared continuation test's ladder, rerun with `padded_from=` the previous level's result at each level, reaches the same true shared ranks; `frame_has_minimal_ranks` unaffected (diagnostic only). |
| T14 | doctests | every new docstring example (§5, §5.8) and every new `>>>` block in the amended docs (§7) runs in the doctest suite. Round 11 moves doctests between layers. `fv_padded_frame`'s four move to `T3Frame.padded`, `ranks.oversampled_ranks`' three to `TuckerTensorTrain.oversampled_ranks`, and the backend `padded_start`'s one to the frontend `padded_start`. New doctests: one on `UniformTuckerTensorTrain.oversampled_ranks` (round 12) and one on each manifold geometry's `padded_frame`. The linalg pair keep theirs (no twin), as do the members without a twin. |
| T15 | `tests/test_frame_variations_format.py`, `tests/test_manifold.py`, `tests/test_uniform_manifold.py`, `tests/test_sharing.py`, `tests/test_tucker_tensor_train.py`, `tests/test_uniform_tucker_tensor_train.py`, `tests/test_fitting.py` (new test functions) | **the frontend twins** (§5.8): (a) each twin equals its backend call **bitwise** on the same inputs — the rewrap adds no arithmetic. This covers `T3Frame.padded(...).data` against `fv_padded_frame`, a geometry twin against its backend member, `TuckerTensorTrain.oversampled_ranks` and `UniformTuckerTensorTrain.oversampled_ranks` against the rank function, and `apply_model(..., frame=F)`'s misfit and gradient against `Problem.local_model(..., frame=F.data)`'s. (b) The structural errors moved from T3(g) and T5(g), each a `ValueError` with the explicit message, in both modes: lengths and boundary ranks; an `old_frame` of other mode sizes or with stores above the target; a hint of other mode sizes or another stack (the message names the reduction); malformed `sharing`; `x0` / `x_old` of another layer, shape or stack, or `x_old`'s stores above `x0`'s; `working_ranks` of the wrong length or below the target; a `frame=` of the wrong structure or layer; a negative or non-integer `p`. A wrong-type `old_frame` or hint is a `TypeError`. (c) The safe-mode checks raise and are skipped under `safety.unsafe()`: ORTH (a perturbed `old_frame`, a perturbed `frame=`), TIED (`T3Frame.padded` with `sharing` and untied old group factors), the same tensor (`x0` ≠ `x_old`; a `frame=` of another tensor). (d) The shared geometry's twin ties an untied `x_old` silently, as `SharedGeometry.frame` does. |

**Gates** (not unit tests; recorded in `docs/contributor/verification_records.md` at the slice that enables them):

- **G-P4** — through the library API, the padded start reproduces P4: on c11–c16, `f1/f0` is below the library's on
  ≥ 5 of 6 cases and the Newton total is ≤ the library's on every case.
- **G-P12** — the cone reference reproduces P12's range.
- **G-bit** — three seeded default runs (ragged GN line search, ragged exact trust region, uniform GN), run on `main`
  and on the branch in one session, give **equal `stats` dicts**: the same keys, every history field bitwise except
  `wall_time`. No golden file: bitwise results are platform-specific.
- **G-acyclic (S3)** — the backend's import graph, built by an AST scan from every import node, is acyclic, and the
  backend has no function-level import of a library module.
- **G-nonbreaking (O-1; S9 and S13)** — the pre-existing test files, `tests/test_*.py` **and `tests/backend/test_*.py`**
  as of `main` @ `efe5e3d`, are run unmodified against the branch and pass. The backend is documented public API, and none of
  those files pins anything the plan changes: the `g0_reference` message test matches only `'g0_reference must be'`, and no
  test pins an `__all__`. They are checked out from `main` into a temporary directory. The
  one expected exception is `test_contractions_interpreter.py`'s vocabulary-completeness check, which by design must
  list the new library strings (T12). Together with G-bit this makes the release's "no breaking changes" statement
  (§7) a tested fact.

---

## 7. Docs and CHANGELOG

Every page below is amended in slice 10, except the math note (slice 12); the API pages' docstrings land with their
code. Text is given in full where it is new; for amendments, the replacement or the added paragraph. Shipped text
cites no `dev/` file. **The release is non-breaking (O-1):** §7's CHANGELOG and release-notes texts say so explicitly,
and G-bit plus G-nonbreaking (§6) make it a tested fact.

**`docs/fitting_and_optimization.md`**

- **§2.1, the `stats` table.** Two rows:
  - `start_frame_source`: "the label `'given'` / `'padded'` — present only when `newton_cg` was given `start_frame`
    or `padded_from` (§5)";
  - `padded_start_gnorms`: "with `padded_from` only: the captured gradient norm at each pass of the padded start".
- **§3, the `Geometry` bullet** (`:269-283`). After the `zero_point_frame` sentence: "Six more are optional, for the
  padded start (§5): `tangent_to_t3`, `padded_frame`, `padded_structures`, `with_ranks`, `oversampled_ranks` and
  `require_frame_structure`. The manifold geometries define them; a geometry without them — the corewise ones, whose
  frame is the point itself, and any custom geometry — cannot take `padded_from` / `start_frame` /
  `g0_reference='cone'` (a structural error names the member)."
- **§4.11, a closing paragraph.** "**From a zero-padded restart the divergence can be removed.** `newton_cg(...,
  padded_from=x_old)` builds iteration 0's frame from the data. Each layer builds its **old** frame natively, and
  the two padded frames agree up to gauge and rounding — the same subspaces, so the same first step — except in the
  four cases listed in `uniform_equivalence_contract.md`. From a zero start, pass the frame
  `optimizers.padded_start(..., x0)` returns as `start_frame` (§5)."
- **§4.12, "Restarts need nothing special".** Add: "A padded start frame (§5) changes none of this: its new slots
  carry no content, so the lift clips them exactly as at the library's completion."
- **§5, the reference bullet.**
  - **The table** gains `| 'cone' | the norm of the projection of the gradient at the zero tensor of x0's ranks onto
    the tangent cone there — a best rank-r approximation of it: exact for matrices; for T3, the frame-gradient norm
    at the fixed point of a subspace iteration, within a few percent of the quasi-optimal value (≈ 11 sweeps and
    transposes per run; explicit only) |`.
  - **"*What exactly the reference is*".** Replace "good to an O(1) factor (another seed, or a rotation of the
    problem, moved it by up to ~1.7× in the measured cases) — harmless for a relative stop, but not an intrinsic
    property of the problem" by: "a **frame projection**: the gradient at the zero tensor projected onto one
    completion's tangent space. On generic data different seeds and rotations move it by up to ~1.7×. On
    importance-ordered bases (active subspaces, Hermite, KL — the common surrogate case) a Haar completion is a
    random projection that captures about `r/N` per mode, and the reference can be far below the cone value
    (`g0_reference='cone'`): **at least 15–76×** below it over six continuation levels of a measured Darcy surrogate
    (71×, 76×, 41×, 26×, 17×, 15×). A relative stop then asks for that much more accuracy, and CG sits at its
    forcing cap. With `'cone'`, use `gtol_rel=1e-4`; its cost is in the verification records."
    Keep the three convention properties (they remain true of `'zero'`).
  - **A new bullet after it.** "**Padded starts.** A zero-padded `resize` is rank deficient: its frame's new slots are
    a completion the library chooses (LAPACK's on the ragged layer, a pad-safe sketch on the uniform one),
    unrelated to the data. `newton_cg(x0, padded_from=x_old)` replaces it, for iteration 0 only, by the **padded
    frame**. Here `x_old` is the previous level's result, the same tensor as `x0` at lower ranks. The old point's
    frame is kept, and the new slots are the leading directions of the gradient's content that is new relative to
    it, found by a few passes of subspace iteration (`padded_passes=`, `padded_seed=`; defaults 4 passes, working
    ranks oversampled by 5). The point is not changed — no content is injected — and no rank bookkeeping is needed.
    Measured on six small continuation cases whose previous level had **converged**: the first Newton step removed
    65–85 % of the objective against 14–32 % from the library completion, and each level took 1–3 fewer Newton
    steps. After a level that **stopped early** it was neutral to worse (up to 4 more Newton steps measured), so use
    it after converged levels. `optimizers.padded_start(...)` returns the frame itself, to pass as `start_frame`;
    with `x_old=None` it is the cone frame at the zero tensor, a start frame for a zero `x0`. The mathematics: the
    math note [`padded_frame_cone.pdf`](padded_frame_cone.pdf)." With a doctest: `x0 = x_old.resize(...)`, then
    `newton_cg(..., x0, padded_from=x_old)`, printing `st['start_frame_source']` → `'padded'`.
- **§6 Pointers** gains `contributor/padded_frame_internals.md` and the math note.

**`docs/rank_continuation.md`** — prose only. The continuation examples and the loop sketch are unchanged this release
(O-2); T3Polynomial adopts `padded_from` in its own ladder.

- **The reference box** (`:103-121`). Replace "good to an O(1) factor — harmless for a relative stop" by "a frame
  projection — on importance-ordered inputs it can be one to two orders of magnitude below the cone value
  (`fitting_and_optimization.md` §5), which `g0_reference='cone'` computes".
- **A paragraph after it.** "A **padded start** — `newton_cg(X0, padded_from=X)` with `X` the previous level's
  result — takes iteration 0 in the data's directions instead of an arbitrary completion. It measured 1–3 fewer
  Newton steps per level when the previous level converged, and neutral to worse when it stopped early
  (`fitting_and_optimization.md` §5)."
- **"Shared Tucker factors" (`:164-196`) and "On the uniform layer" (`:198-231`).** One sentence each. The padded
  frame is tied (one factor per group, from the group's concatenated gradient content; the old point's factors must
  be tied). Each layer builds the old point's frame natively, so ragged and uniform continuations start each level
  from the same frame up to gauge and rounding (the exceptions are listed in `uniform_equivalence_contract.md`).

**`docs/numerical_contracts.md`**

- **The row at `:203`.** "good to an O(1) factor (… ~1.7× …)" becomes "a frame projection (seed-dependent by ~1.7× on
  generic data; on importance-ordered data at least 15–76× below the cone value, `g0_reference='cone'`)".
- **New rows in "The optimizers' relative stop".**
  - `'cone'` — structural: a geometry without `padded_structures`, a vanishing value.
  - "`newton_cg(start_frame=, padded_from=)` / `padded_start`" — structural:
    - frame structure (`require_frame_structure`, against the reduced `x0`);
    - mutual exclusion and keyword validity;
    - an old point of another layer, shape or stack, or with frame stores above `x0`'s (the frontend also requires an
      equal uniform padded `N`, for its same-tensor check);
    - working ranks below the target;
    - a corewise geometry.

    Numerical, safe mode only:
    - `x0` and `padded_from` the same tensor (`allclose` at `effective_rtol`, the difference's norm through
      orthogonalization);
    - the start frame **ORTH** and representing `x0`;
    - **TIED** `padded_from` / start frame on a shared geometry.
- **The `fitting.LocalModel` section** gains: "a frame supplied by the caller (`Problem.local_model(frame=)`,
  `newton_cg(start_frame=)`) makes ORTH a precondition rather than a construction invariant".

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
> a hint, any Tucker tensor train (a gradient embedded as one, another direction, a sum of directions), taking the
> leading directions of the hint's content that is new relative to the old point
> (`contributor/padded_frame_internals.md`; the mathematics: the math note `padded_frame_cone.pdf`).

**`docs/uniform_equivalence_contract.md`** — after the full-rank qualifier's measured table (`:61-63`):

> **Padded starts make the same choice on both layers, up to gauge.** With `newton_cg(padded_from=...)`, each layer
> builds the **old** point's frame natively: LAPACK on the ragged layer, the pad-safe SVD on the uniform one, and the
> two differ by a sign gauge. The padded construction itself runs on unpacked real blocks in both cases, and every
> step of it is covariant under that gauge. So the two padded frames span the same subspaces up to rounding and take
> the same first step; the iterate is then numerically full rank, and the element-wise contract applies. Measured:
> per-iteration objectives agree to `<1e-8` over 10 iterations (`tests/test_optimizers_frontend.py`).
>
> The exceptions — cases where a choice is not determined by spans — are:
>
> - a near-tie at a cut;
> - a degenerate hint;
> - old frames that differ by more than signs (a tied or numerically deficient old point);
> - an in-block coordinate seed in the old columns' span, rare on library-produced frames.
>
> This is a statement about subspaces. Tangent *coordinates* still require the same frame entrywise, and
> `safety.frames_equal` keeps guarding that, unchanged.

The "Consequence for testing" paragraph gains: "…or start both layers from a padded start, and compare
gauge-invariant quantities (projectors, dense tangent-space projections, objectives)."

**`docs/sharing.md`** — "Rank machinery … Continuation" (`:197-211`): "With `newton_cg(padded_from=…)` the new group
levels are completed from the group's concatenated gradient content (face 1 above) instead of arbitrarily. They are
still exactly zero in the point, so the tied channel stays gated, but the TT channel's escape starts in the data's
directions. The old point's factors must be tied (TIED; safe mode checks it)."

**`docs/api_reference.rst`**

- **"Fitting and optimization".** Add `optimizers.padded_start` (module-level, not at the root), the `newton_cg`
  keywords sentence, `TuckerTensorTrain.oversampled_ranks` beside `continuation_ranks`,
  `UniformTuckerTensorTrain.oversampled_ranks` beside `minimal_ranks` (round 12), and the model factories' `frame=`.
- **The frame and geometry areas** (round 11). Add `T3Frame.padded` and the `padded_frame` methods of
  `ManifoldGeometry`, `UniformManifoldGeometry` and `SharedGeometry`.
- **The backend surface's frame-variations and rank areas.** Add `padded_frames.fv_padded_frame` (a new module),
  `linalg.pinned_completion`, `linalg.pinned_leading_columns` and
  `ranks.oversampled_ranks`.

**`docs/naming_conventions.md`** — "Intended asymmetries (uniform vs ragged)" gains a bullet: "**No `ufv_padded_frame`.**
Frame construction at a degenerate point is ragged by nature (it works on unpacked real blocks), and rank
continuation is a ragged-layer activity; the uniform geometry's `padded_frame` member converts explicitly (built
through `fv_padded_frame`, host-side and eager), and `contributor/padded_frame_internals.md` gives the recipe for raw
uniform `.data`. `ufv_zero_point_frame`, the same kind of wrapper, remains for now." And beside the `sharing` example
(`:45-48`) one sentence: "`backend/padded_frames.py` is likewise a topic module: `fv_padded_frame` carries the
family prefix of the frame it builds."

**`docs/contributor/padded_frame_internals.md`** (new; in `contributor_guide.rst`'s toctree after
`curvature_internals`): the design record.

- **From §2 of this plan:** the cone and the frame tangent spaces; why the aligned completion and not the supremum; the
  pinned T3-SVD of a T3 hint; two-sidedness; the δ-limit and why it is not the target; the hint-free conventions and
  why their seeds are coordinates (gauge covariance); GD = subspace iteration; the cone reference and what it is not.
- §4's algorithm with the contraction table.
- The cross-layer gauge equivalence of §3.3, with its canonical exception list E1–E4 quoted.
- The placement (I-32).
- "The uniform conversion": the uniform layer without a twin (item 7), with the **raw-`.data` recipe** (§5.2; cited by
  `UniformManifoldGeometryOps.padded_frame`'s docstring, round 12), and the rejected masked vectorization of the uniform construction, with its measurement (the critic's note: the host loop at
  46 / 81 ms against 491 / 372 ms batched pad-safe, at `|C| = 4` and ranks 30–50).
- §9's decisions with their reasons.
- The measurements P1–P17.

Its opening paragraph links the math note (`../padded_frame_cone.pdf`, source `docs/padded_frame_cone.tex`), as
`curvature_internals.md:8` links the 2026.3.0 note.

**`docs/contributor/curvature_internals.md`** — "The canonical zero-point frame" (`:171-222`):

- the "good to an O(1) factor" sentence (`:217`) is corrected as in `fitting_and_optimization.md`, with a pointer to
  the cone reference;
- the table row E9 (`:307`) reads "good to an O(1) factor on generic data; not on importance-ordered data (2026.4.0,
  `padded_frame_internals.md`)".

**`docs/contributor/deferred_and_rejected.md`**

- **I-29's "A possible later refinement … averaging the reference over a few seeds" (`:60-62`).** Becomes:
  "superseded by the cone reference (2026.4.0): a seed average estimates the mean frame projection, a
  completion-dependent number; the cone reference is the projection onto the tangent cone, computed at the subspace
  iteration's fixed point".
- **The deferred "zero *start* using the canonical zero-point frame" (`:227-229`).** Becomes: "available opt-in:
  `start_frame=` (2026.4.0); the default zero start is unchanged".
- **New "Rejected" entries.**
  - The δ-limit frame as the definition (P9).
  - The largest frame projection `sup_F ‖P_{T_F}G‖` as the reference: it is the rank-`2r` value, attained at
    staggered completions that the iteration does not converge to.
  - The Tucker shortcut from the hint's `V_i` alone (P7).
  - Carrying the projected content in the sweep ("rotation corrections", P10).
  - Seeded-Gaussian seeds for the in-block completion: not covariant under the old frame's gauge (P17); coordinate
    seeds are used.
  - A ragged round trip for the uniform old frame. Considered in rounds 2–5 to make the layers bitwise equal;
    replaced by native frames and gauge equivalence.
  - Old ranks as the `padded_from` argument: replaced by the old point, which removes the layout, prefix-mask and
    minimal-old-ranks requirements.
  - An options object for the padded-start knobs: plain keywords instead.
  - **A `ufv_padded_frame` twin** (item 7). The construction is ragged by nature and rank continuation is a ragged-layer
    activity; a `ufv_` name would hide a host round trip behind a name that suggests a native uniform op. The "every
    ragged op has a uniform twin" symmetry gains this documented exception for frame construction at degenerate
    points; the uniform geometry's `padded_frame` member converts explicitly.
  - **A masked, vectorized uniform construction** (O-5, the critic's study). It is feasible and jit-clean, but needs two
    new masked primitives, five masked steps and three structures' masks: 250–400 lines against about 50. It
    reads worse, measured slower than the host loop on CPU at ranks 30–50, and its degenerate-hint completions
    diverge from the ragged ones. The cheap route for a future stacked consumer is grouping the elements by
    structure, about 15 lines.
  - ε-content enrichment (downstream).
  - Padded-start members on the corewise geometries.
  - A numerical rank decision anywhere (decision 4).
- **New "Deferred" entries.**
  - `'cone'` or `padded_from` as library defaults (O-1).
  - Re-resolving `chunk_size` at the working ranks (O-7).
  - **An audit arc for `fv_zero_point_frame`** (round 9): its template-frame signature ("a frame read only for its
    shapes", commit 3e5bc922) is ruled wrong by the maintainer; it stays unchanged in this release, and the audit
    moves it to the gold-standard style (structure as int sequences), together with `ufv_zero_point_frame`.
  - **`ranks` imports `sharing`** (`ranks.py:16`; `sharing` imports `fv_operations`, `sharing.py:34`) — a code smell
    (Nick, round 10): the integer rank arithmetic depends on the sharing module, so `fv_operations` cannot use the
    rank recurrence without a cycle, which is why `fv_padded_frame` lives in `backend/padded_frames.py` (I-32).
    Recorded for the later drift-fix arc; not changed here.
  - **A pre-existing docstring mismatch, recorded and not changed here:** `UniformTuckerTensorTrain.allclose` says
    padded widths need not match, but it raises on a different padded `N` (it works on different `n` / `r`; the r4
    critic's probe). Fix the docstring or re-pad in a later change.
  - **`ufv_zero_point_frame`'s consistency with item 7.** It is the same kind of host-side unpack/ragged/repack
    wrapper and is left untouched this release; whether it should also become an explicit conversion inside the
    uniform geometry's `zero_point_frame` member is a later question.
  - **A start frame for `gradient_descent` and the other manifold methods**, through the same
    `local_model(frame=)` seam (O-8). A short `dev/` note — `dev/start_frame_other_methods_2026_11.md`, written in
    S10 — records the seam and what each method would need.

**`docs/contributor/verification_records.md`** — a new section "The padded frame and the cone reference
(2026.4.0)": P1–P17, the gates G-P4, G-P12, G-bit, G-acyclic, G-nonbreaking, and the S11 measurements (the laptop
arms and ladder pair at release; the GPU ladder pair when it lands). The "zero reference" section's E9 line gains the
cone/Haar ratios of §0.

**`docs/contributor/sharing_internals.md`** — "Padded restarts" (`:98-122`): one paragraph — the padded frame aims
the escape (the TT channel's first-order mass pairs with data-informed tied directions); full shared rank stays a
diagnostic; TIED old-point factors are a precondition of the padded start.

**`docs/exact_hessian_two_hole.tex`** — the zero-reference subsection (`:904-940`): `:937`'s "good to an $O(1)$
factor" is corrected, and a paragraph defines the cone reference: the norm of the projection of $G$ onto the
tangent cone of $\mathcal M_{\le r}$ at $0$ (a best rank-$r$ approximation; $\|G_r\|$ for matrices), estimated by
$\|P_{T_{F^*}}G\|$ at the fixed point $F^*$ of the subspace iteration — equal to it for matrices (the aligned
completion); for T3 at least the norm of $G$'s projection onto $F^*$'s own rank-$r$ tensor space, possibly above the
cone value through the rank-$2r$ directions of $T_{F^*}$, and measured within a few percent of the value at the
T3-SVD frame — and **not** $\sup_F\|P_{T_F}G\|$, which those rank-$2r$ directions make larger ($\|G_{2r}\|$ for
matrices), with a forward reference to the new math note `padded_frame_cone.pdf`. The PDF is rebuilt at release.

**`docs/padded_frame_cone.tex` + `.pdf` — the math note (new; written in slice S12, after the code and the
docs).** It mirrors `docs/exact_hessian_two_hole.tex`, the 2026.3.0 note:

- **Form.** Both the source and the built PDF are committed in `docs/`. A header comment names it the mathematical
  companion of the design record (`docs/contributor/padded_frame_internals.md`), gives the build command
  (`latexmk -pdf padded_frame_cone.tex`, pdflatex) and the packages (amsmath, amssymb, amsthm, mathtools, algorithm,
  algpseudocode, booktabs, tikz, url, hyperref, geometry — all standard), and records the commit at which every code
  reference resolves. Code names use the same `\code` macro (typewriter, breakable at `_` and `.`).
- **Placement in the docs.** The note is linked, not built, by Sphinx: markdown links to the `.pdf`, as
  `fitting_and_optimization.md:531` and `contributor/curvature_internals.md:8` link the 2026.3.0 note.

Title: "Padded frames, the tangent cone, and the cone reference". Outline:

1. **Setup and notation.** T3 frames `(U, O, L, R)`; the four rank stores and their recurrence; frame tangent spaces
   (the `2d` terms, the gauge); the carriers; a padded point and its old block.
2. **The tangent cone at a rank-deficient point.** The matrix case (Schneider–Uschmajew): `TC_X = T_X M_s ⊕ {normal
   parts of rank ≤ r − s}` and `P_{TC}G = P_{T_X}G + N_k`; what carries over to T3 (Kutschan) and what is used.
3. **Frame tangent spaces versus the cone.**
   - Every cone element lies in the tangent space of a completion aligned with it.
   - The union of the frame tangent spaces is larger: their normal parts reach rank `2k`.
   - The supremum `‖P_{T_X}G‖² + ‖N_{2k}‖²` is attained at a staggered completion.
   - At the aligned completion `P_{T_Q}G = P_{TC}G`.
   - The `r = 1` example.
4. **Padded frames.**
   - The definition: the old block embedded, the carriers, the completions.
   - Validity: per-core orthonormality and the gauge suffice, and the base point is `X_old` exactly.
   - Why every completion is a valid frame.
5. **The pinned T3-SVD of a hint.**
   - The hint as a T3.
   - Per slot (Tucker, left bonds, right bonds, down): the candidate space, the hint's content there, the two-sided
     removal and the pins.
   - Exactness when the hint's chains contain `X_old`'s interfaces (gradient hints; `H + X_old`, tails squashed),
     and the contraction caveat otherwise.
   - Counts are structural; a degenerate hint is completed; determinism and its two LAPACK-dependent cases.
6. **The hint-free construction.**
   - `s = 0`: the canonical zero-point frame.
   - `s > 0`: Haar draws in the complement, outside-first coordinates, the in-block completion from coordinate seeds
     (covariant under a sign gauge of the old frame — why the two layers agree up to gauge).
   - Nesting, rank independence, layer independence.
7. **The subspace iteration.**
   - At `s = 0`: unit-step Riemannian gradient descent on `½‖Y − G‖²` (HOOI-like).
   - At `s > 0`: the block power iteration on the completion.
   - The aligned completion is the attracting fixed point; the staggered maximizer is not one.
   - Oversampling and truncation (the randomized-SVD analogy), and the start.
8. **The cone reference.**
   - What it is: the projection onto `M_{≤r}` at its apex.
   - What is computed: exact for matrices; for T3, bracketed (§2.7 of this plan).
   - What it is not: the supremum, a seed average, the Haar value.
   - Its metric dependence.
9. **Shared Tucker factors.** The group concatenation (face 1), the TIED old block, padded restarts and the gated
   tied channel.
10. **Algorithms.** One box each, in the `algorithm` / `algpseudocode` style of the 2026.3.0 note. Each box cites
    the implementing functions by qualified name and file, with the `contract` subscripts of its steps as
    comments. They become real names once S1–S9 have landed — the reason the note is written after the code.

    | box | content | implementing functions | file |
    |---|---|---|---|
    | 1 | the pinned primitives | `linalg.pinned_completion`, `linalg.pinned_leading_columns` | `backend/linalg.py` |
    | 2 | the hint-free construction | `padded_frames.fv_padded_frame` (`hint=None`), `padded_frames._outside_first`, `fv_operations._haar_rows` | `backend/padded_frames.py`, `backend/fv_operations.py` |
    | 3 | the hint's coordinates and the Tucker step | `padded_frames._hint_coordinates`, `padded_frames._tucker_completion`, `tt_operations.tt_squash_tails` | `backend/padded_frames.py`, `backend/tt_operations.py` |
    | 4 | the pinned left sweep and the mirrored right sweep | `padded_frames._pinned_left_sweep`, `tt_operations.tt_reverse` | `backend/padded_frames.py`, `backend/tt_operations.py` |
    | 5 | the down step | `padded_frames._down_completion` | `backend/padded_frames.py` |
    | 6 | the uniform conversion (remark box: no twin) | `geometry.UniformManifoldGeometryOps.padded_frame`, `ufv_conversions.ut3frame_to_t3frame`, `ut3_conversions.ut3_to_t3`, `ufv_conversions.t3frame_to_ut3frame` | `backend/geometry.py`, `backend/ufv_conversions.py`, `backend/ut3_conversions.py` |
    | 7 | the pass loop, with oversampling and truncation | `optimizers.padded_start`, `ranks.oversampled_ranks`, and the geometry members `tangent_to_t3`, `padded_structures`, `padded_frame`, `oversampled_ranks` | `backend/optimizers.py`, `backend/ranks.py`, `backend/geometry.py` |
    | 8 | the cone reference | `optimizers._cone_reference_gnorm`, `optimizers.newton_cg` (`g0_reference='cone'`) | `backend/optimizers.py` |

11. **Numerical evidence.**
    - The plan's probes P1–P16: frame validity; the first-step and Newton tables; oversampling; the cone against
      the dense-frame value; the matrix oracle; the T3-hint and augmentation checks.
    - The critic's per-step dense oracle (≤ 4.4e-11).
    - The downstream record: cone/Haar at least 15–76×, the restart table, the ladders.
    - S11's acceptance numbers.
    - The library tests that pin these: T3(h), T3(j), T5(a), T8.
12. **References.** Schneider & Uschmajew (2015); Kutschan (2018); Olikier, Gallivan & Absil; Vermeylen,
    Vandereycken & Absil; De Lathauwer, De Moor & Vandewalle (HOOI, 2000); Halko, Martinsson & Tropp (2011);
    Ceruti, Kusch & Lubich (the BUG integrator); Hubig et al. (DMRG subspace expansion); T4S (arXiv:2603.21141v2,
    cited as the 2026.3.0 note cites it).

**Gate (S12):**

- `latexmk -pdf` builds with no errors and no undefined references (checked in the log), and the PDF is committed.
- A new permanent test, `tests/test_math_note_references.py`, scans the note. Every `\code{<module>.<name>}` must
  resolve to an attribute of `t3toolbox.backend.<module>` or `t3toolbox.<module>`. Every quoted subscripts string
  (`\texttt{'…->…'}`) must occur in the library's contraction vocabulary, by the same AST scan
  `tests/test_contractions_interpreter.py` uses. This is the reference analog of G-acyclic: a later rename breaks
  it loudly.

**Links** (where the user docs point to it):

- `docs/fitting_and_optimization.md` §5 — the reference bullet and the padded-starts bullet end with "The
  mathematics: the math note [`padded_frame_cone.pdf`](padded_frame_cone.pdf)".
- `docs/rank_continuation.md` — the padded-start paragraph.
- `docs/frame_variations.md` — the new section's pointer.
- `docs/contributor/padded_frame_internals.md` — its opening paragraph, as `curvature_internals.md:8` does.
- `CHANGELOG.md` — the `[2026.4.0]` introduction, as `[2026.3.0]`'s names its note.
- `docs/release_notes.md` — the new head.
- `CLAUDE.md` — if the release adds a summary line there (2026.3.0 did, `:480-486`), it names the note.

**`docs/release_notes.md`** — a new head "Upgrading from 2026.3.0":

> **No breaking changes.** Every new behaviour is opt-in, and without the new keywords every run, its `stats`
> included, is bit-identical to 2026.3.0. The pre-existing tests (frontend and backend) pass unmodified.
>
> Two additions are worth trying:
>
> - **The padded start.** In a continuation loop, pass the previous level's result: `newton_cg(X0,
>   padded_from=X)`, with `X0 = X.resize(...)`. Do it at a level whose predecessor converged. It measured 1–3 fewer
>   Newton steps per level on small cases, and neutral to worse after a level that stopped at its cap.
> - **The cone reference.** If your inputs are importance-ordered and your stops are relative, `g0_reference='cone'`
>   with `gtol_rel=1e-4` gives a reference that does not depend on a chosen completion. The default zero reference
>   measured at least 15–76× below it on a Darcy surrogate. The cost of `1e-4` is in the verification records.
>
> The docs' earlier claim that the zero reference is good to an O(1) factor was wrong on such data and is corrected.
> The mathematics is in the math note `padded_frame_cone.pdf`.

**`CHANGELOG.md`** — a new top section:

```markdown
## [2026.4.0] — unreleased

The padded start: a structural, data-informed frame for a rank-deficient point, and the Newton reference it makes
possible. **No breaking changes**: every addition is opt-in, and without the new keywords every run, its `stats`
included, is bit-identical to 2026.3.0. The mathematics — the tangent cone, the pinned T3-SVD of a hint, the
subspace iteration and the cone reference, with the algorithms as implemented — is written up in the math note
`docs/padded_frame_cone.pdf` (source `docs/padded_frame_cone.tex`).

### Added

- **`newton_cg(x0, padded_from=x_old)`** starts a continuation level from the **padded frame**. `x_old` is the
  previous level's result, the same tensor as `x0` at lower ranks, so no rank bookkeeping is needed. Its frame is
  kept, and the new slots are completed by the leading directions of the gradient's content that is new relative to
  it: a few passes of subspace iteration at oversampled working ranks, then a truncation. The point is unchanged — no
  content is injected — and every count is structural, with no threshold.

  On six small continuation cases whose previous level had converged, the first Newton step removed 65–85 % of the
  objective, against 14–32 % from the library's completion, and each level took 1–3 fewer Newton steps. After a level
  that stopped early it was neutral to worse. The ragged and uniform layers start each level from the same frame up
  to gauge and rounding.

  Keywords: `padded_passes=` (default 4) and `padded_seed=`. Such runs record `stats['start_frame_source']` and
  `stats['padded_start_gnorms']`. Every new keyword of `newton_cg` and `gradient_descent` is keyword-only, appended at
  the end of the signature, so existing positional calls are unaffected.
- **`newton_cg(..., start_frame=frame)`**: iteration 0's frame given explicitly, an orthonormal frame of `x0`'s
  tensor at its structure. **`optimizers.padded_start(...)`** returns the padded frame (with `x_old=None`, the cone
  frame at the zero tensor) and the captured gradient norms.
- **`g0_reference='cone'`** (`newton_cg`, `gradient_descent`): the Newton reference as the norm of the projection
  of the zero-tensor gradient onto the **tangent cone** at zero, a best rank-`r` approximation of it. It is exact for
  matrices; for T3 it is the frame-gradient norm at the fixed point of a subspace iteration, within a few percent of
  the quasi-optimal value (≈ 11 sweeps and transposes; `cone_passes=`, `cone_seed=`). Unlike the default `'zero'`
  reference it does not depend on a chosen completion. It is explicit only, the default is unchanged, and
  `gtol_rel=1e-4` is recommended with it.
- **`TuckerTensorTrain.oversampled_ranks(p)`**, **`UniformTuckerTensorTrain.oversampled_ranks(p)`** (per stack
  element) and **`ranks.oversampled_ranks`**: the ranks `+p`, structurally minimal, never below the input. These are
  the working ranks of a padded start.
- **`T3Frame.padded(shape, tucker_ranks, tt_ranks, old_frame=…, hint=…)`** builds a padded frame directly.
  **`padded_frame(x0, x_old)`** on `MANIFOLD`, `UNIFORM_MANIFOLD` and a shared geometry gives the padded frame at a
  new level's start. The `fitting` model factories accept **`frame=`**, an orthonormal frame of `x` supplied by the
  caller.
- **Backend.**
  - `fv_padded_frame` — the constructor, documented on its frontend twin `T3Frame.padded` (the uniform layer converts through the geometry member). Its hint is any Tucker tensor train at the frame's
    mode sizes: a gradient embedded as one, another direction, or a sum of directions. `fv_zero_point_frame` is its
    zero-tensor, hint-free case.
  - `linalg.pinned_completion` / `pinned_leading_columns`.
  - `backend.optimizers.padded_start` / `PaddedStart`, and `Problem.local_model(frame=)`.
  - The optional manifold-geometry members `tangent_to_t3`, `padded_frame`, `padded_structures`, `with_ranks`,
    `oversampled_ranks` and `require_frame_structure`.

### Fixed (documentation)

- The zero reference was documented as "good to an O(1) factor". That holds across seeds on generic data, but on
  importance-ordered bases a Haar completion captures about `r/N` per mode, and on a Darcy surrogate the reference
  measured at least 15–76× below the tangent-cone value. The docs now describe it as a frame projection and point to
  `g0_reference='cone'`.
```

---

## 8. Slices

One commit per slice (files staged by name), the full suite green at each; review stops after S3 (the
construction), S7 (the optimizer seam), S10 (the docs) and S12 (the math note) — Nick's "discussion between steps".

| slice | content | files | gate |
|---|---|---|---|
| S1 | `pinned_completion`, `pinned_leading_columns`; HISTORICAL strings 1–2; `ranks.oversampled_ranks` | `backend/linalg.py`, `backend/ranks.py`, `tests/backend/test_linalg.py`, `tests/test_contractions_interpreter.py` | T1, T5(h) (the rank function's rows), T12 (partial), T10 (primitives) |
| S2 | `fv_padded_frame`, hint-free path: `_empty_old_frame`, `_embed`, `_outside_first`, the in-block completion (coordinate seeds), validation | `backend/fv_operations.py` (moved to `padded_frames` by S2′), `tests/backend/test_padded_frame.py` | T2 (incl. the gauge covariance T2(f)) |
| S3 | the hinted path: `_hint_coordinates` (the tail squash), `_tucker_completion` (groups), `_pinned_left_sweep` (both chains), `_down_completion`; HISTORICAL 3–12 | `backend/fv_operations.py` (moved to `padded_frames` by S2′), tests | T3 (incl. the dense oracle T3(h) and the external hint T3(j)), T10 (jit), T12, G-acyclic; P1/P3 re-measured through the function. **Review stop.** |
| S2′ | **corrective** (rounds 9–12; the handoff's option C): create `backend/padded_frames.py` and move the S2/S3 code into it, in the round-9 signature `fv_padded_frame(shape, tucker_ranks, tt_ranks, old_frame=None, hint=None, groups=(), seed=ZERO_FRAME_SEED, stack_shape=(), use_jax=False)` — the stores from `ranks.compute_orthogonal_representation_ranks`, no template frame, `_frame_stores` deleted, no `dtype`; `fv_operations` returns to its committed pre-arc form, its `ZERO_FRAME_SEED`, `_haar_rows` and `_identity_prefix` imported by the new module; **round 11:** the frontend twins of S1–S3's public functions land here. `T3Frame.padded` takes `fv_padded_frame`'s description, doctests and checks, except the four kept guards. `TuckerTensorTrain.oversampled_ranks`, moved from S9, takes `ranks.oversampled_ranks`' docstring and doctests. The linalg pair stay as committed (no twin). **Round 12:** `ranks.oversampled_ranks` widens to per-element arrays, and `UniformTuckerTensorTrain.oversampled_ranks` lands. S2′ edits **committed code only** (review r8 N2). S4's uncommitted work is stashed while S2′'s gate runs, and S4 brings its own member to §3.3's integer form when it lands | `backend/padded_frames.py` (new), `backend/fv_operations.py`, `backend/ranks.py`, `t3toolbox/frame_variations_format.py`, `t3toolbox/tucker_tensor_train.py`, `t3toolbox/uniform_tucker_tensor_train.py`, `tests/backend/test_padded_frame.py`, `tests/backend/test_padded_start.py` (T5(h)), `tests/test_frame_variations_format.py`, `tests/test_tucker_tensor_train.py`, `tests/test_uniform_tucker_tensor_train.py` | the gate runs with S4's uncommitted work stashed; T2 and T3 on the new signature (S3's oracle tests carry over; only the calls change); T5(h); `git diff efe5e3d -- t3toolbox/backend/fv_operations.py` empty; G-acyclic rerun; T15 (these twins' rows); T14 (the moved doctests run on the twins) |
| S4 | the uniform conversion in `UniformManifoldGeometryOps.padded_frame` (unpack → `fv_padded_frame` → repack, host loop; no `ufv_` twin) — S4 restores its stashed working-tree member and brings it to §3.3's integer form against S2′'s committed constructor (round 12) — and its frontend twin `UniformManifoldGeometry.padded_frame` with the uniform `_require_padded_pair` (round 11) | `backend/geometry.py`, `t3toolbox/uniform_manifold.py`, tests, `tests/test_uniform_manifold.py` | T4(a)–(d), T10 (uniform rows), T15 (the uniform twin) |
| S5 | the remaining geometry members on the two manifold classes (the ragged `padded_frame`, `tangent_to_t3`, `padded_structures`, `with_ranks`, `oversampled_ranks`, `require_frame_structure`), and the twins `ManifoldGeometry.padded_frame` and `SharedGeometry.padded_frame` with the ragged `_require_padded_pair` (round 11): the native old frame from `x_old` (uniform: its own `.data` and masks); the structural checks (layer, shape, stack, `x_old`'s stores ≤ `x0`'s, working ranks ≥ target), in the frontend `_require_padded_pair` since round 11 (the member keeps none); `oversampled_ranks` delegating to the rank function; the protocol docstring | `backend/geometry.py`, `backend/optimizers.py:65-131`, `t3toolbox/manifold.py`, `t3toolbox/shared_geometry.py`, `tests/test_manifold.py`, `tests/test_sharing.py` | T4(e), T5(g), T5(i), member rows of T10, T15 (the ragged and shared twins), G-acyclic |
| S6 | `PaddedStart`, `padded_start` (plain keywords, `working_ranks`; a one-line docstring — its description and doctest land with the frontend twin in S9), the constants; `Problem.local_model(frame=)` | `backend/optimizers.py`, `tests/backend/test_padded_start.py` | T5 (matrix oracle), T6 |
| S7 | `newton_cg(start_frame=, padded_from=x_old, padded_passes=, padded_seed=)`, the working ranks through `geom.oversampled_ranks`, the `stats` keys (only when used), `_prepare_jit_frame` and the `padded_from` conversion | `backend/optimizers.py`, `tests/backend/test_optimizers.py` | T7, T11 (partial); G-bit (equal `stats` dicts); G-P4. **Review stop.** |
| S8 | `g0_reference='cone'` with `cone_passes` / `cone_seed` (both optimizers), `_cone_reference_gnorm`, the display label | `backend/optimizers.py`, `backend/optimizer_display.py`, `tests/backend/test_zero_reference.py` | T8; G-P12 |
| S9 | frontend `padded_start` (with the backend's description and doctest, round 11), the six `newton_cg` keywords (forwarded only when set), the reduction of `x_old`, the safe-mode checks, the model factories' `frame=` (round 11; `TuckerTensorTrain.oversampled_ranks` moved to S2′) | `t3toolbox/optimizers.py`, `t3toolbox/fitting.py`, `tests/test_fitting.py`, new test functions in `tests/test_optimizers_frontend.py`, `tests/test_jax_absent.py`, `tests/test_dispatch.py` | T9 (incl. the equivalence test), T10, T11, T13; **G-nonbreaking** |
| S10 | every doc of §7 except the math note: the internals page, CHANGELOG, release notes, the 2026.3.0 note's cone paragraph, the ledger entries, the `dev/` note for O-8 | `docs/…`, `CHANGELOG.md`, `dev/start_frame_other_methods_2026_11.md` | T14 (doctests), the Sphinx build. **Review stop.** |
| S11 | **acceptance on the laptop** (§11): the restart arms and the ladder pair — measurement, no library change | T3Polynomial `scripts/`, `dev/` (its own process) | the numbers recorded in `verification_records.md`; the decisions they feed taken (§11) |
| S12 | **the math note** `docs/padded_frame_cone.tex` + `.pdf` (§7: outline, eight algorithm boxes citing the implemented functions and files, the numerical evidence incl. S11's numbers), its links from the user and contributor docs, and the reference test | `docs/padded_frame_cone.tex`, `docs/padded_frame_cone.pdf`, `tests/test_math_note_references.py`, the linking docs | `latexmk -pdf` clean (no errors, no undefined references); `tests/test_math_note_references.py` passes. **Review stop.** |
| S13 | release 2026.4.0 — **decoupled from the GPU run** | `pyproject`, tag | the release checklist; the library tests; S11's laptop measurements; G-nonbreaking rerun |
| after | the Runpod 20 × 1000 ladder pair (§11), manual | T3Polynomial / the research repo's runpod notes | not a release gate: it informs O-9's advice text and the I-13 defaults at scale, which may land in a later docs update |

---

## 9. Decisions

Nick's binding decisions are I-1–I-5. Since round 6 Nick has also settled I-3, I-13 and the open questions (§10);
in round 10 I-32 and I-43–I-45; and in round 11 I-22 and I-46. The remaining rows are the writer's judgment calls; the critic and Nick should
weigh the ones marked **(weigh)**. History:

- round 2 changed I-5, I-18 and I-21, and added I-28–I-33;
- round 3 made I-32 a judgment call and added I-34;
- round 5 rewrote I-5 and I-32, and added I-35 and I-36;
- round 6 reverted I-28, removed I-34, rewrote I-3, I-11, I-20, I-29 and I-30, and added I-37–I-41;
- round 7 made every new keyword keyword-only (I-40) and dropped the backend's padded-`N` guard (I-37);
- round 9 rewrote I-26, reopened I-32 (the `ranks` import cycle) and added I-42 (the backend style);
- round 10 settled I-32 (Nick: a new module), rewrote I-16 (the working geometry comes from `with_ranks`), and added
  I-43–I-45 (Nick's calls on `dtype`, §12's scope and the `with_ranks` test);
- round 11 rewrote I-22 and I-41 and added I-46 and I-47 (the frontend/backend split).

| I | decision | settled as |
|---|---|---|
| I-1 | surface | both: the standalone `padded_start` (backend + frontend) and the `newton_cg` keywords (Nick 1) |
| I-2 | defaults | nothing changes without a new keyword: `g0_reference` default `'zero'`, `fv_zero_point_frame` untouched, `newton_cg` bit-identical **including `stats` and history** (Nick 2; G-bit, T7(a)); the release is non-breaking (I-40) |
| I-3 | the knobs (Nick, round 6) | **plain keywords**, no options object: `newton_cg(padded_passes=, padded_seed=, cone_passes=, cone_seed=)`, `gradient_descent(cone_passes=, cone_seed=)`, `padded_start(working_ranks=, passes=, seed=)`; module constants for the defaults; the oversampling `p` is a module default that the convenience paths pass to the rank function (per-call `p`: `padded_start(working_ranks=x0.oversampled_ranks(p))`); `PaddedStart` stays as the result type |
| I-4 | structural only | counts from the stores; no threshold; no content in the point; the retraction and the orthogonal representation untouched; the pinned T3-SVD is a new function on the hint (Nick 4) |
| I-5 | stacking | `C` only: `fv_padded_frame` vectorized over `C`; the uniform member (`UniformManifoldGeometryOps.padded_frame`) a host loop for all stacks (Nick, round 6; masked vectorization rejected — §3.9, §7's ledger); `padded_start` over `C` (backend and frontend); one T3 hint per frame — combining several hints is the caller's reduction; the optimizers unstacked (Nick 5) |
| I-6 | the construction | HOSVD-style two-sided Tucker step + mirrored pinned TT sweeps + the down step from the new Tucker slots (design (d), Appendix B) — measured best or equal (P4, P6), group-ready, one sweep code path; verified per step against a dense oracle by the critic (≤ 4.4e-11), and T3(h) makes that a test |
| I-7 | the δ-limit | not the definition: it carries `O(t)` rotations of the old block in `R`, `O` and measured worse (P9) |
| I-8 | the down step | the hint's center in the final bases, restricted to the new Tucker slots, pinned off `O°` (§2.3); at `s = 0` it gives the down rows of the truncated hint's center |
| I-9 | rotation corrections | dropped: inert at converged and unconverged levels (P10) |
| I-10 | Tucker coordinates | ambient, each member's normal content compressed, a group concatenated (sharing face 1); equal to the prototype's `w`-coordinate step whenever the hint has `≥ kU` directions, and free of `w_i ≥ nU_i` (an external hint may be low-rank) |
| I-11 | hint-free conventions (round 6) | Tucker: Haar in `U°`'s complement from the keyed stream's first `kU` rows (nested; `s = 0` = `_haar_rows`); TT / down: outside-first coordinates (`s = 0` = identity prefixes), then an in-block completion from **coordinate seeds** in index order — covariant under a sign gauge of the old frame, which §3.3's gauge equivalence needs (a seeded Gaussian is not: P17); the coordinate seed's degenerate case affects only the start (R5) **(weigh)** |
| I-12 | the driver's start | the hint-free frame at the working structure, not the library frame of the padded point: deterministic and gauge-covariant; its measured cost is about one pass |
| I-13 | passes and oversampling (Nick, round 6) | `passes` = gradient evaluations, the last construction truncates; **4 (start), 10 (cone), `p = 5`**; the Runpod run checks them at scale (§11) |
| I-14 | the residual | once per call, at `x0` (or at zero); the misfit only — the regularizer's gradient is tangent to the old stratum and removed by the two-sided steps |
| I-15 | where the loop lives | `backend/optimizers.py` (T3-free) with five optional duck-typed geometry members (`tangent_to_t3` embeds each pass's gradient as the T3 hint; `oversampled_ranks` adapts the rank function); a geometry without them keeps working |
| I-16 | the working geometry | `with_ranks(*working)` (round 9; the uniform geometry holds variation masks for one structure, so the working structure needs its own) |
| I-17 | `local_model(frame=)` | preconditions ORTH and "same tensor"; structural check always (`require_frame_structure`), the numerical ones in frontend safe mode only (house rule) |
| I-18 | recording the start frame | `stats['start_frame_source']` (a label: `'given'` / `'padded'`) and `stats['padded_start_gnorms']` with `padded_from` — **only on runs that use a start frame**; no history field (decision 2) |
| I-19 | the zero-tensor start | `padded_start(x_old=None)` + `start_frame=` (two calls; the returned frame is the cone frame at zero, a start frame for a zero `x0` only) |
| I-20 | frontend `start_frame` (round 6) | a structure check against the **reduced** `x0` ("build it at `x0`'s minimal structure"); the `padded_from` half is gone with the old ranks (I-37) |
| I-21 | `'cone'` | the norm of the projection onto the tangent cone at zero (a best rank-`r` approximation), computed at the subspace iteration's fixed point — **not** the supremum of frame projections; explicit only; computed iff `g0norm_newton is None`; vanishing raises; recorded `'cone'`; `cone_passes` / `cone_seed` error without it; **`gtol_rel=1e-4` recommended** (O-9, Nick) |
| I-22 | the constructor's frontend twin (Nick, round 11) | **`T3Frame.padded`** (§5.8). It carries the description, the doctests and the checks; `fv_padded_frame` keeps one line and the four kept guards. This supersedes "no frontend constructor" (O-6) |
| I-23 | no `pad_safe_svd` | the uniform member unpacks real blocks, so no SVD of the construction ever sees padding |
| I-24 | contractions | twelve new `C`-group strings, letters with distinct `AXIS_SIZE` sizes; two existing strings reused; reshapes only for SVD inputs, `C` leading |
| I-25 | the O(1) claim | corrected in every doc that makes it, with the cone/Haar ratios (at least 15–76×: the dense `‖T3SVD_r G‖` is a lower bound on the cone value); CHANGELOG "Fixed (documentation)" |
| I-26 | `fv_zero_point_frame` | **not changed in this release**: its template-frame signature (commit 3e5bc922) is ruled wrong by the maintainer (round 9) and goes to a later audit arc (§7's ledger); it is not a precedent. The array-free `s = 0` constructor call reproduces it — coordinates bitwise, the Haar factors to `1e-14` (I-38) |
| I-27 | unconverged levels | documented as neutral to worse (P11: up to 4 more Newton steps); the docs advise the padded start after converged levels; no automatic switch (that would be a numerical decision) |
| I-28 | the two layers — **reverted** (Nick, round 6) | each layer builds its old frame **natively** from `x_old` (no ragged round trip); cross-layer agreement is gauge equivalence, "the same frame up to gauge and rounding", tested with projectors and a dense probe; a statement about subspaces only — `frames_equal` for tangent coordinates is untouched |
| I-29 | working ranks (round 6) | the rule lives in one rank function, `ranks.oversampled_ranks(shape, tucker, tt, p, sharing)` (elementwise max of the input and `minimal(+p)`; frontend twins `TuckerTensorTrain.oversampled_ranks` and, per element, `UniformTuckerTensorTrain.oversampled_ranks` (round 12)); `padded_start` / `padded_structures` take explicit `working_ranks=None` (= the target) and compute no ranks |
| I-30 | the old point's slack (round 6) | the frontend reduces `x_old` structurally (`_ragged_minimal` / `uniform_minimal`, sharing-aware) before its frame; the backend accepts any `x_old` with a docstring caveat (numerically zero singular values and structural slack give arbitrary old directions) |
| I-31 | corewise geometries | no padded-start members: a corewise frame is the point itself; every new keyword on corewise is a structural error |
| I-32 | placement (Nick, round 10) | **a new module `backend/padded_frames.py`** (option A) holding `fv_padded_frame` and its private helpers. All its imports point downward (`ranks`; `fv_operations`'s `ZERO_FRAME_SEED`, `_haar_rows`, `_identity_prefix`; the orthogonalizations, `tt_operations`, `contractions`, `linalg`), and only `geometry` imports it. The justification is the import graph alone: `fv_operations` cannot import `ranks` (`ranks → sharing → fv_operations`). `compute_orthogonal_representation_ranks` stays in `ranks`; `ranks`' dependence on `sharing` is a code smell for the later drift-fix arc (ledger). Not taken: (B) a function-level import (it breaks G-acyclic); (C) the recurrence inline (a third copy); (D) the recurrence moved down behind an alias (the critic's r6 recommendation). `fv_operations` returns to its pre-arc form in S2′. No uniform twin (I-41); G-acyclic checks the result |
| I-33 | version | 2026.4.0 (new public API; the house numbering increments the minor per release) |
| I-34 | — | **removed** (round 6): the prefix-layout requirement disappears with the old point (I-37) |
| I-35 | the hint's type (Nick, round 5) | a **T3** at the mode sizes `shape`, any ranks, stack `C`; the driver embeds each pass's gradient through `tangent_to_t3`; tails squashed (P16); exact when its chains contain `X_old`'s interfaces (gradient hints, or any hint plus `X_old`), valid otherwise; cost scales with the hint's ranks; the `K` lift withdrawn |
| I-36 | the math note (Nick, round 5) | `docs/padded_frame_cone.tex` + `.pdf`, written in S12 after the code, mirroring the 2026.3.0 note; a clean build plus a permanent reference test |
| I-37 | `padded_from` takes the old **point** (Nick, round 6; the critic's note) | `newton_cg(x0, padded_from=x_old)`, `padded_start(problem, x_cores, x_old)`; uniform: `x_old` as its full `.data`, its frame built from its own masks (gappy masks fine); backend guards: same layer / shape / stack, `x_old`'s frame stores ≤ `x0`'s (the frontend adds an equal uniform padded `N` for its same-tensor check); safe mode: `x0.allclose(x_old)` at `effective_rtol` (the difference's norm through orthogonalization, never the inner-product formula), TIED `x_old`; no layout, prefix or minimal-old-ranks requirement |
| I-38 | bitwise vs tolerance (Nick, round 6) | bitwise only for structural facts (old block copy, zero carriers / padding, masks, coordinate completions, the default-path claims G-bit / T6(a) / T7(a)); stated tolerances for everything computed; gauge-invariant projector and dense-probe checks for frame claims (§6's preamble) |
| I-39 | acceptance (Nick, round 6) | laptop, autonomous: the restart arms and one ladder pair (the release gate); Runpod, manual: one 20 × 1000 ladder pair after the release (not a gate); 100 × 5000 deferred to the paper; each measurement tied to the decision it feeds (§11) |
| I-40 | non-breaking release (O-1, Nick, round 6) | stated in the CHANGELOG and the release notes; every new keyword is **appended keyword-only** at the end of its signature (backend and frontend `newton_cg`, `gradient_descent`), so no positional call changes meaning (review r4, MINOR A — the alternative, grouping them with the reference block, would shift later positional parameters); the frontend forwards a new keyword only when it is set; G-bit plus G-nonbreaking (the pre-existing frontend and backend tests pass unmodified) |
| I-41 | no uniform twin (Nick, round 6, item 7) | no `ufv_padded_frame`: the construction is ragged by nature and rank continuation is a ragged-layer activity; a `ufv_` name would hide a host round trip. `UniformManifoldGeometryOps.padded_frame` converts explicitly (unpack → `fv_padded_frame` → repack, host loop, one-line docstring); the recipe for raw uniform `.data` is backend-user documentation: it lives in the internals page, cited by the member's docstring (round 12); the uniform equivalence tests target the member; the ledger records the exception to the twin symmetry, and `ufv_zero_point_frame` stays this release with the consistency question noted. No serious problem found: the razor holds (the math is standalone, the glue is a callable backend method plus a documented recipe), and the equivalence tests keep their content |
| I-42 | the backend style (Nick, round 9) | the gold standard — `tucker_tensor_train.py`, `uniform_tucker_tensor_train.py` and the backend functions they trace to — is the only allowable precedent: pure mathematical functions on immutable basic types; structure as `Sequence[int]`; inline shape reading and integer checks; dispatch from the arrays. `fv_padded_frame(shape, tucker_ranks, tt_ranks, …)` derives its stores inside; no template frame; no dict helper; `padded_structures` returns integers only; the uniform members build from ints and masks (§3.2, §3.4, §5.2, §5.4, §12) |
| I-43 | `dtype` (Nick, round 10) | **none**: no `dtype` argument anywhere in the feature. The dtype follows the arrays, and an array-free call returns the backend's default float (float64 under numpy), as the gold-standard constructors do (`t3_zeros`, `ut3_zeros`, `t3_corewise_randn`). The accepted consequence: on float32 data under numpy, or jax with x64, the cone's passes run in float64; the cone value is unaffected (§3.5) |
| I-44 | §12's scope (the maintainer's relaxation, round 10) | the style rule **binds every backend function of the feature**, including everything the cone path calls. In the driver and geometry (optimizer-layer) code it is **the goal**: that layer passes points, its genuine inputs. `padded_start` reads `x_cores`' structure and stack through `padded_structures` and its backend through `tree_contains_jax`, and takes the cone's residual at `corewise_zeros_like(x_cores)`; the members never read a frame only for its shapes |
| I-45 | `with_ranks`' hand-built masks (Nick, round 10) | kept: `from_point` and `from_frame` need arrays, and a zero point built only to read its masks is the ruled-out pattern. §3.4 gives the reason it is safe, and T5(i) guards it: `with_ranks` of a geometry's own ranks reproduces its `masks` and `var_masks` exactly |
| I-46 | the frontend/backend split (Nick, round 11) | every new public backend function or keyword has a frontend twin. The twin validates (explicit messages; the safe-mode numerical checks), makes one or two backend calls and rewraps, and it carries the description and the doctests. The backend keeps a one-line docstring, its shape comments and the terse guards of §12, as `tucker_change_core_shapes`, `t3_add`, `t3_to_ut3` and `t3svd` do. A function without a twin keeps its docstring and doctests (Nick's correction) |
| I-47 | the twin map (round 11) | §5.8. Twins: `T3Frame.padded`; `padded_frame(x0, x_old)` on the three manifold geometries; the existing `T3Tangent.to_t3` / `UT3Tangent.to_ut3`; `TuckerTensorTrain.oversampled_ranks` and `UniformTuckerTensorTrain.oversampled_ranks` (round 12); the frontend `padded_start`; `frame=` on the six model factories; the frontend `newton_cg` / `gradient_descent` keywords. No twin: the linalg pair; the members `padded_structures`, `with_ranks`, `oversampled_ranks`, `require_frame_structure`; `UT3Frame` (a uniform frame needs the padded sizes of the point it serves; round 12); private pieces and constants. **(weigh)**: the three geometry twins and the factories' `frame=` are the largest new frontend surface |

---

## 10. Risks and open questions

**Risks**

| # | risk | mitigation |
|---|---|---|
| R1 | From an **unconverged** old level the padded start is neutral to worse (P11: `f1/f0` worse on all six cases; Newton totals +3, 0, +4, −1, −1, −1): the old block is not a critical point, so the gradient's old-coupled content — which the two-sided steps discard — is where the progress is. Downstream ladders often end levels at the cap. | Opt-in; the docs advise it after converged levels only; no automatic switch (I-27). S11's restart arms decide whether T3Polynomial's ladder enables it. |
| R2 | **Cost of the extra passes.** `padded_from` adds `passes` frame sweeps and transposes (4) per level; `'cone'` adds 11 per run, against 1 for `'zero'`. On the measured Darcy levels a level's CG did hundreds of Hessian-applies, so this is small — but it is measured at toy scale only. | The Runpod 20 × 1000 ladder pair records the per-level cost (§11); `padded_passes=` / `cone_passes=` lower it. To reuse a cone value, pass `g0norm_newton=padded_start(..., x_old=None).gnorm` — only that `gnorm` is the cone value; a `padded_from` start's `gnorm` is the deflated warm-start gradient and must never be used as a reference. |
| R3 | **Memory at the working (oversampled) ranks.** The problem's memory chunk — the `chunk_size` keyword of the probe-derivative assembly, unrelated to the rank increment `p` of `oversampled_ranks` — is resolved at the target ranks; the passes run at the working ranks, so a `probe_derivatives` transpose sized near the memory limit can exceed it. | Documented on `padded_start(working_ranks=)`; `padded_start` with `working_ranks=None` removes it; the Runpod run watches peak memory. Re-resolving the chunk at the working ranks is deferred (O-7). |
| R4 | **Eager jax passes.** Under `use_jit` the passes run eagerly on jax arrays (per-op dispatch, no compile). | Small problems pay dispatch, large ones are dominated by the transposes; the Runpod run measures. Jitting the pass body is a later option. |
| R5 | **LAPACK-dependent choices**: a degenerate hint (fewer directions than slots, or zero), a tie at a cut (`σ_k = σ_{k+1}`), and an in-block coordinate seed lying in the old columns' span (a coordinate-aligned old core) leave the choice to the Householder completion or to the SVD's basis of a tied subspace, which can differ across LAPACK builds and devices. | Always a valid frame (T1(a), T3(a)); deterministic per backend; the hint carries no information that separates the candidates, so any choice is as good. The coordinate-seed case is rare on library-produced frames: the critic's probe found first in-block seed residuals of 0.05–0.8, and core 0 has no in-block room for a minimal `x_old`. It affects the hint-free start, whose effect the hinted passes damp but, at the default four, do not remove. These are exceptions E1, E2 and E4 of §3.3's canonical list. Stated in the docstrings. |
| R6 | **The cone value is an iteration output** — 0.99–1.04× the dense-frame value (P12), dependent on the passes and the working ranks, and metric-dependent like every projection. Relative stops are comparable across levels only with fixed settings. | Defaults fixed and documented; `cone_passes` / `cone_seed` are explicit; the gradient-norm reference stop stays (O-4 closed). |
| R7 | **An arbitrary T3 hint can be expensive**: the cost scales with the hint's ranks (§2.3), and a high-rank external hint (the data tensor, a sum of many directions) makes the down-orthogonalization, the chain orthogonalizations, the zippers and the sweep SVDs larger than a gradient's doubled frame ranks. | Documented on `fv_padded_frame`; the caller may compress the hint first (`t3svd` to a few times the target ranks). The driver's hints are gradients at the doubled frame ranks. |
| R8 | **Exact Hessian at a padded frame**: new slots aimed at the data make the first steps larger in exactly the directions the lift clips. | The trust region bounds them; T6(d) and T7(b) run exact mode from padded frames; S11's restart arms run both Hessians. No content is injected, so the downstream ε failure (first-step rejections) cannot recur by that mechanism. |
| R9 | **Two near-duplicate zero-tensor constructions** (`fv_zero_point_frame` and the `s = 0` hint-free `fv_padded_frame`) could drift. | T2(a) and T4(a) pin their equality (coordinates bitwise, Haar factors to `1e-14`); the docstrings cross-reference. |
| R10 | **Cross-layer agreement is gauge equivalence to rounding**: the two layers' old frames differ by a sign gauge and their gradients by rounding; the exceptions are §3.3's canonical list E1–E4 (a near-tie at a cut; a degenerate hint; old frames that differ by more than signs; an in-block coordinate seed in the old columns' span). | Every step is gauge-covariant under signs (§3.3; T2(f) tests the hint-free start, T5(f) the whole start); the docs state the list; every cross-layer test is gauge-invariant with rounding tolerances (§6). |

**Open questions — settled by Nick (round 6)**

| # | question | settled as |
|---|---|---|
| O-1 | Should `'cone'` become the default reference? | **No**: defaults unchanged; this release is **non-breaking** — stated in §7's CHANGELOG and release notes, gated by G-bit and G-nonbreaking (I-40). |
| O-2 | Should continuation use `padded_from` by default? | **T3Polynomial adopts `padded_from` in its ladder** (its own process, §11); T3Toolbox's continuation examples are unchanged this release. |
| O-3 | Fixed `+5` or rank-relative oversampling? | **Fixed** (as in randomized SVD), via the rank function with `p = 5`. |
| O-4 | A metric-invariant stop (the Newton decrement)? | **Closed**: the gradient-norm reference stop stays. |
| O-5 | Vectorize the uniform construction? | **Rejected** — moved to the ledger with the critic's reasons (§3.9, §7: 250–400 masked lines against about 50, readability, slower on CPU, diverging degenerate-hint completions); grouping by structure (about 15 lines) is the cheap route if a stacked consumer appears. |
| O-6 | A frontend wrapper of the constructor? | **Superseded** (Nick, round 11): `T3Frame.padded` (I-22, §5.8). |
| O-7 | Re-resolve `chunk_size` at the working ranks (R3)? | **Deferred** (ledger). |
| O-8 | `start_frame` for `gradient_descent` and the other manifold methods? | **Not now**: a `dev/` note (`dev/start_frame_other_methods_2026_11.md`, S10) and a "Deferred" ledger entry — the same `local_model(frame=)` seam serves them. |
| O-9 | Which `gtol_rel` with `'cone'`? | **`1e-4`**, stated in the user docs without a comparison (review r4, MINOR B). For the record: about `3e-4` of **T3Polynomial's** coordinate-aligned reference (the cone value is 2.3–5.4× it on the Darcy surrogate), i.e. about 3× tighter than T3Polynomial's `1e-3` regime — not the library's pre-2026.3.0 warm-start stop. S11's ladder pair runs `1e-4` and `3e-4` so the cost is on record; the Runpod run may refine the advice text after the release. |

---

## 11. Downstream (T3Polynomial) and acceptance

The library change is what the downstream investigation asked for (T3Polynomial
`dev/t3toolbox-upstream-notes.md` §2026-11-02; `dev/zero_reference_cone_2026_11_02.md`). Measurements run in the
`t3toolbox` env with the library branch installed (`pip install .`; pure Python, no native build).

**Downstream code changes are T3Polynomial's business** under its lighter process — not slices of this plan. For
reference, what it will do:

- its ladder passes `padded_from=X` (the previous level's result) to the next level's `_newton_fit` when that level
  converged (O-2; `_newton_fit` forwards `**kwargs`, `fitting.py:176-245`). There is no rank bookkeeping, since the
  old point replaces the old ranks;
- its `'cone'` runs pass `g0_reference='cone'` with the tolerance S11 settles;
- `t3polynomial/cone.py`, the prototype, retires:
  - `cone_reference` → `g0_reference='cone'`, or `optimizers.padded_start(..., x_old=None).gnorm`;
  - `tucker_cone_completion` → `padded_from` (now the TT completion too, δ-free);
  - `cone_frame_zero_start` → `padded_start` + `start_frame`.

**S11 — acceptance on the laptop** (autonomous; the release gate together with the library tests). Each measurement
is stated with the decision it feeds:

| # | measurement | arms | decision it feeds |
|---|---|---|---|
| A1 | **restart arms**, Darcy o01 level 1 (the §0 restart) and **e01 at rank 7** | both layers (ragged; `uniform_jax` on CPU) × both Hessians (Gauss–Newton, exact) × {`padded_from`, the library default}, at the same absolute stop; Newton / CG counts, first-step `f1/f0`, wall time | whether `padded_from` ships enabled in T3Polynomial's ladder (R1), and the cross-layer claim (equal counts up to rounding effects; the 21-vs-44 split gone) |
| A2 | **one Darcy o01 ladder pair** | `'zero'` at `gtol_rel=1e-3` against `'cone'` at `1e-4` and at `3e-4`; Gauss–Newton over the full ladder, or the exact Hessian restricted to the first three levels if the full exact ladder is too slow on the laptop; Newton / CG totals, levels converged, final test p50 | the tolerance (O-9): the cost of the recommended `1e-4` on record beside `3e-4` |

**After the release — Runpod, MANUAL** (not a release gate; the release is decoupled from the GPU run). The procedure:

1. Nick starts the pod.
2. The session prepares **one** queue script, launches it, and collects on a DONE marker.
3. The session rsyncs the results back and tells Nick when to stop the pod.

The procedure follows the maintainer-local research repo's Runpod notes; auto-stop is unconfirmed, so the stop
is Nick's.

| # | measurement | arms | decision it feeds |
|---|---|---|---|
| G1 | **one 20 × 1000 Darcy ladder pair** | the library default against `'cone'` + `padded_from` after converged levels; per-level wall time and peak memory of the extra passes recorded (R2–R4) | whether the I-13 defaults (4 / 10 / `p = 5`) survive at scale, and O-9's advice text — either may land in a docs update after the release |

The 100 × 5000 layout is deferred to the paper.

---

## 12. Pitfalls checklist

**The backend style rule** (Nick, round 9 — verbatim): backend functions are PURE MATHEMATICAL FUNCTIONS whose inputs
and outputs are immutable objects of basic types (arrays, tuples of arrays, ints, static int tuples, host bool masks),
structure travels as `Sequence[int]` with `len=d` shape comments, bodies derive structure from array shapes inline and
follow the math, dispatch is inferred from arrays, frontend methods only validate and rewrap.

**The gold standard is the only allowable precedent:** `t3toolbox/tucker_tensor_train.py`,
`t3toolbox/uniform_tucker_tensor_train.py`, and the backend functions they trace to (the traces:
`dev/padded_frame_slices/style_notes_writer.md`). **Precedent outside the gold standard is not justification.** That
excludes the geometry/optimizer layer, and `fv_zero_point_frame`, whose "template frame read only for its shapes"
pattern (commit 3e5bc922) the maintainer has ruled wrong, together with the dict-of-shapes helper `_frame_stores`.

**Scope** (round 10, the maintainer's relaxation; I-44). The rule **binds every backend function of the feature**:
the constructor and its helpers in `padded_frames`, the `linalg` primitives, `ranks.oversampled_ranks`, and everything
the cone path calls. The cone path builds its first frame by the array-free `fv_padded_frame`, not by
`fv_zero_point_frame`. In the driver and geometry code — the optimizer layer: `padded_start`, the seam in
`newton_cg`, `_cone_reference_gnorm`, the geometry members — the rule is **the goal**. That layer passes points, its
genuine inputs:

- `padded_start` reads `x_cores`' structure and stack through `padded_structures`, and its backend through
  `tree_contains_jax`;
- the cone path takes its residual at `corewise_zeros_like(x_cores)`;
- the members take and return integers, points, frames and geometries, and never read a frame only for its shapes.

**The frontend/backend split** (Nick, round 11; I-46). In Nick's words: "There should be frontend versions of most of
these functions as either methods of the appropriate classes or as standalone functions. Most of the docstrings
(detailed description and doctests) should be moved from the backend functions to their frontend twins. Most of the
in-function argument validation should be moved from the backend to the frontend." And: "Leave doctests that cannot
be moved. Do not delete doctests from the backend if there is no frontend twin to move them to." The gold standard
shows the split: `TuckerTensorTrain.resize` carries the full docstring, the doctests and the checks, while
`tucker_change_core_shapes` has a one-line docstring and shape comments; `__add__` validates with explicit messages,
then makes one backend call and rewraps.

- **A backend function with a twin keeps** what the gold-standard backend keeps. This is a one-line docstring and its
  shape comments (`tucker_change_core_shapes`, `t3_add`, `t3_to_ut3`), and dispatch from the arrays. It also keeps
  terse structural guards on its own static arguments and on array agreement, but only where a violation would
  otherwise be silent or obscure:
  - `t3_add` keeps its stack-mismatch check, `t3svd` its `validate_sharing`, and the uniform functions
    `require_concrete_masks`; `tucker_change_core_shapes` and `t3_to_ut3` keep none;
  - it keeps no safe-mode numerical check, no remedy in a message and no doctest.

  The kept guards here:
  - the constructor's four (a count below zero, an old frame / hint stack mismatch, an array-free argument with an
    array, a sharing group with unequal `(Tucker rank, size)`; §3.2);
  - the uniform member's mask check (§3.3);
  - `require_frame_structure`;
  - the backend `padded_start`'s members guard;
  - `newton_cg`'s keyword-contract guards (`start_frame` with `padded_from`, a `padded_*` keyword without
    `padded_from`, a `cone_*` keyword with another reference, `passes ≥ 1`, the seed range; §3.6), whose
    violations would otherwise be silently ignored;
  - `ranks.oversampled_ranks`' `p` and length checks, as `compute_minimal_ranks` keeps its own.
- **Its twin carries** the detailed description, all its doctests, and the validation: explicit messages that name the
  fix (as `__add__` and `t3svd` do), and the safe-mode numerical checks through `safety` (ORTH, TIED, the same
  tensor). It is written as validate → one or two backend calls → rewrap, with no math (§5.8).
- **A backend function without a twin** (§5.8: the linalg pair, the driver members, private pieces) keeps its
  docstring, its doctests and its checks. Nothing is deleted, and nothing is converted to a test.

| pitfall | how this plan handles it | where |
|---|---|---|
| **backend style — the gold standard** (Nick, round 9; scope and `dtype`, round 10) | `fv_padded_frame(shape, tucker_ranks, tt_ranks, old_frame, hint, groups, seed, stack_shape, use_jax)` takes the structure as three int sequences, derives the four stores with `ranks.compute_orthogonal_representation_ranks`, reads array shapes inline, checks integers inline (no helper, no dict), infers dispatch from its arrays (the array-free-constructor convention otherwise), and returns the frame tuple; no template frame anywhere; `padded_structures` returns integers only; the uniform members build from ints and masks (`ut3_make_masks`, `ufv_make_frame_masks`, `ufv_variation_masks`), never from an orthogonalized zero tensor; `fv_zero_point_frame` is unchanged this release and its defect goes to a later audit arc; no `dtype` (I-43); the rule binds the feature's backend functions and is the goal in the optimizer layer (I-44) | §3.2, §3.4, §5.2, §5.4, I-43, I-44 |
| **Named contractions** (coordinator) | Every nontrivial contraction is a `contract(...)` call with the `C` group — the hint re-basis, the Tucker environment and its projection, the sweep's carry/projection/update, the down environments, the primitive's projection; twelve new strings with HISTORICAL `''`, two reused. Plain `xnp.linalg.qr/svd` only for factorizations; `concatenate`, `swapaxes`, `pad_or_truncate` where that is all that is needed. | §4.0–§4.10 |
| **Sharding safety** (coordinator) | No reshape inside contractions; reshapes only to form SVD/QR inputs (merging adjacent non-stack rank axes) and to fold factors back; `C` always leading; no stack axis moved or merged. `test_contractions_sharding.py` sweeps the new strings automatically. | §4.0, §4.10 |
| **bitwise vs tolerance** (Nick, round 6) | bitwise only for structural facts (the old block as a copy, zero carriers and padding, masks, coordinate completions, the default-path claims G-bit / T6(a) / T7(a)); stated tolerances (`1e-14` orthogonality, `1e-13` relative reconstruction, `1e-14` computed arrays, the safety rtol in safe-mode tests; never `np.allclose` defaults) for anything computed; frame claims through assembled projectors (`U_iᵀU_i`, each bond's interface projector, each down core's row-space projector) and a dense `P_{T_F} Y`; no "bit for bit" claim for a computed result anywhere in the plan | §6, I-38 |
| batching / stacking (C/W/K, frame-inner) | `C` vectorized in the ragged constructor and the driver; `W` never appears (the constructor sees no samples; the kinds keep their own `W` handling); `K` never appears either — the hint is one T3 per frame, and several hints are combined by the caller; the uniform member a host loop for all stacks by decision; optimizers unstacked | §3.9, I-5 |
| ragged / uniform split | no uniform twin (I-41): `UniformManifoldGeometryOps.padded_frame` unpacks, runs `fv_padded_frame`, repacks at the geometry's frame padded sizes, host loop; uniform masks stay host `np`, supercores `xnp`; the working uniform geometry is built for its own masks; each layer builds the old frame **natively** (the uniform one from `x_old`'s own `.data` and masks), and the layers agree up to gauge and rounding | §3.3, §3.4, §4.8, I-28, I-41 |
| `pad_safe_svd` at masked SVDs | none needed: no SVD of the construction sees padded data (the uniform member unpacks); the native uniform old frame comes from `ut3_orthogonal_representations`, which uses `pad_safe_svd` itself | §3.3, I-23 |
| numpy/jax dispatch | `xnp` from the array inputs (`tree_contains_jax`); host draws converted with `xnp.asarray(..., dtype)`; float sign literals cast to `Q`'s dtype; the result follows the array inputs' dtype and backend — array-free, the backend's default float, as `t3_zeros` (no `dtype`, I-43); `padded_from` converted like `x0` under `use_jit` | §3.2, §3.6, §4.1, T10 |
| jittability | the constructor and primitives are traceable (static shapes and counts, host constants, no value-dependent branches); the uniform member and the driver are eager by design (masks, Python loop); a start frame enters the CG as a local-model leaf, its uniform masks value-equal to the problem geometry's frame masks (no recompile) | §3.2, §3.6, T7(e), T10 |
| numerical vs structural | structural checks always (shapes, counts, stacks, the old point's layer / shape / stack and stores (frontend: also the uniform padded `N`), working ranks, keyword validity, mutual exclusion, the reduced-`x0` structure of a start frame, corewise); numerical ones (`x0` equals `x_old`, ORTH and same tensor for a start frame, TIED) only in frontend safe mode, skipped under `unsafe()` and traces; the hint's chains and `x_old`'s zero singular values are caveats | §3.8 |
| no numerical rank decision | counts are store differences; leading directions are taken in order and completed; no `σ` is compared with a tolerance anywhere | §2.8, §4.1 |
| comment and signature style | one argument per line, aligned name · type · `#` shape contract, one return element per line with `len=d, elm_shape=C+(...)` | §5 |
| backend/frontend razor (Nick, round 11) | math in backend standalone functions and one rank function, each with a one-line docstring, its shape comments and only the kept guards above; every public piece has a frontend twin (§5.8) that validates (explicit messages, safe mode), calls the backend and rewraps, and carries the description and the doctests; the frontend also adds the structural reduction of `x_old` (§3.6); a function without a twin keeps its docstring and doctests | §3.4, §5.7, §5.8, I-22, I-46, I-47 |
| sharing | one tied Tucker factor per group from the concatenated normal content, the old block read from the group's first mode under TIED (checked on `x_old`); hint-free draws keyed by `min(g)`; TT/down untied; the rank function's shared notion keeps a group's rank equal | §2.9, §4.5, T3(e), T13 |
| safe mode | as "numerical vs structural"; the same-tensor check uses `allclose` (the difference's norm through orthogonalization) at `effective_rtol`, never the inner-product formula; the backend never runs a numerical check | §3.6 |
| naming (`docs/naming_conventions.md`) | the `fv_` prefix for the constructor in the topic module `padded_frames` — a feature module whose function carries its family prefix, as in `sharing` (I-32; one sentence in the naming page); unprefixed `linalg` primitives (as `pad_safe_svd`) and `ranks.oversampled_ranks` (as `compute_minimal_ranks`'s neighbour); `require_` for a raising structural member; plain keywords named per call site (`padded_passes`, `cone_seed`, …) | §3.2, §5, I-3, I-32 |
| import acyclicity | the new module `padded_frames` imports only downward (`ranks`, `fv_operations`, …; no `tv_to_t3` since round 5), and only `geometry` imports it; `fv_operations` cannot import `ranks` (`ranks → sharing → fv_operations`, I-32); G-acyclic in S3: the backend import graph from every import node (AST scan, module- and function-level) is acyclic, and the backend has no function-level library import | §3.2, §6, I-32 |
| doctests | a twinned function's doctests live on its twin (T14 lists the moves); a function without a twin keeps its own (Nick, round 11); amended docs carry `>>>` blocks; doctest margins wide enough for both numpy generations | §5, §5.8, §7, T14 |
| CHANGELOG | a new section in the Keep-a-Changelog format of `[2026.3.0]`: "No breaking changes", Added, Fixed (documentation); no `dev/` citations in shipped text | §7 |
| equivalence-contract obligations | through the uniform member, on identical inputs the real blocks match the ragged construction to `1e-14`, the masks equal the native sweep's and match the geometry's `var_masks`, and garbage padding is never read (T4); the native uniform old frame is gauge-equivalent to the ragged one (T4(e)); gauge-invariant cross-layer comparisons of `padded_start` (T5(f)), the cone reference (T8(c)) and the padded-restart trajectory (T9(e)); the contract doc's new paragraph, scoped to subspaces (`frames_equal` for coordinates untouched) | §6, §7 |
| bit-identical defaults, non-breaking | the default path executes the same statements; no new history field; `stats` keys only when used; the frontend forwards a new keyword only when set; T7(a); G-bit compares whole `stats` dicts; G-nonbreaking runs the pre-existing tests (frontend and backend) unmodified | §3.6, §6, I-2, I-18, I-40 |
| options are not parameters | the padded-start knobs are per-call keywords (never a geometry or kind field); the defaults are module constants; `p` reaches the rank function from the call site | §5.5, I-3 |

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

What produced P1–P16, with the inert correction branch removed (P10) and, since round 5, the hint taken as a T3 (the
prototype embedded the gradient inside the constructor; P15 checks that the numbers are unchanged). Two limits of the
prototype that the library removes: its `w`-coordinate Tucker step needs `w_i ≥ nU_i` (the library's step is
ambient, §4.5), and it does not squash a hint's boundary bonds (the library does, §4.4; P16).
The library version differs as §4 says:
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

def padded_frame(target, old, Z):                     # design (d): §4.4-§4.7; Z = the hint, any T3 (tucker_cores, tt_cores)
    nU, nD, rL, rR = target                           # the target stores (ints): compute_orthogonal_representation_ranks
    oU, oO, oL, oR = old; snU, snD, srL, srR = stores(old)
    d = len(nU)
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

def padded_start(grad, target, working, old, passes, hint_free_start):              # §4.9; target, working: store tuples
    F = hint_free_start(working, old)
    for j in range(passes):
        Z = tv_to_t3(F, grad(F))                                                     # the geometry's tangent_to_t3
        F = padded_frame(target if j == passes - 1 else working, old, Z)
    return F
```

(The prototype's `U` for the measured runs was `B[i] @ W[i]`, equal to the line above to rounding; `old` for the
zero tensor is the empty frame of §4.2.)
