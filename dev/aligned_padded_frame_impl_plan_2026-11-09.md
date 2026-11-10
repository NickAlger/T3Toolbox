# Aligned padded frame — implementation plan (fresh plan, 2026-11-09)

> **Status: APPROVED BY THE CRITIC (reviews r4 and r6, 2026-11-09) after five writer rounds
> (`dev/aligned_padded_frame_plan_review_r{1,...,6}.md`). Rounds 4–5 (CHANGES 4.1–5.3) folded in the Newton
> measurement (§0, §2.3, §11) and zero alignment passes (D-38). Rounds 6–7 (no critic round, factual or Nick's
> decision): the stores skip rule, the text cycle case, the higher-rank measurement (CHANGES 6.x); jit under user
> control and the pair stop rule (D-39, CHANGES 7.x). Round 8 (Nick's decisions and as-built deviations, no critic
> round): Gaussian slot-indexed seeds (D-41, slice N8b) and D-40, D-42–D-48 (CHANGES 8.x). Round 9: the verified
> fixed-point statement (D-50), slot-sign dependence (D-49); N9 done, N10 in progress (CHANGES 9.x).** Then implementation
> slice by slice (§8) with an implementer/reviewer pair at extra-high effort. Every agent first reads a handful of
> `TuckerTensorTrain` / `UniformTuckerTensorTrain` methods and the backend functions they trace to
> (`docs/contributor/signature_style.md`, `naming_rules.md`). Those modules are the only allowable precedent for code shape.
>
> **Supersedes** `dev/archive/padded_frame_cone_impl_plan_2026-11-02_superseded.md` and its reviews r1–r8. §9 and
> Appendix B list what survives from it. The mathematics is Nick's note "Completing frames by truncating A + εS,
> with projection-only access" (https://claude.ai/artifact/EvfyxaZKGuKUL5MdC6UfjD), specialized to T3s.
>
> **Reference prototypes** (research repo, read-only for the library; `P/` below means
> `nicks_research_experiments/tensor/aligned_padded_frame/`):
>
> | role | prototype |
> |---|---|
> | ragged step | `P/core_form2.py`; `P/cycle/core_form3.py` adds `update=` |
> | ragged loop | `P/ragged_loop.py` |
> | uniform step | `P/uniform_form2.py` |
> | uniform start | `P/uniform_start.py` (`method='gs'`) |
> | uniform loop and compaction | `P/uniform_loop.py` |
> | dense per-edge reference | `P/dense_oracle.py`, plus `P/cycle/cycle_schemes.py` for two-colour and transport |
> | scatter into uniform layouts | `P/uniform_testing.py` |
>
> `P/README.md` holds every measurement; the Darcy runs are `T3Polynomial/scripts/g06*.py`. These files are
> references for the MATHEMATICS and the verification only. They are not references for code shape: they use for
> loops, raw einsums, slicing and closure scan bodies.
>
> **For the maintainer.** None of D-1–D-19 is reopened. Nick CONFIRMED the three realizations below (round 4).
> New since then: D-38 (zero alignment passes as a documented spelling) and §0's measured motivation.
> (1) **D-8's "standalone loop with class dispatch" becomes one twin per layer**:
> `frame_variations_format.fv_aligned_padded_frame` and `uniform_frame_variations_format.ufv_aligned_padded_frame`.
> Each dispatches on the hint's class (§3.6, D-35). One cross-layer function is not possible in `manifold.py`,
> because `uniform_manifold` imports `manifold`. It would need a new frontend module that imports both layers. D-15
> (no polymorphism between layers) also favours one twin per layer.
> (2) **`newton_cg` gains fourteen keywords** (D-33): `start_frame`, `padded_from`, and two plain groups of six. The
> `*_alternate` knobs are new, so every knob stays settable. A schedule record would be smaller, but it is the
> `PaddedFrameOptions` you rejected (superseded I-3).
> (3) **No frontend `padded_start` twin** (D-35, following D-8's list). This costs one use case: the aligned frame
> of a problem's gradient WITHOUT running Newton, e.g. for your own optimizer. That case goes through the public
> backend `padded_start` (§3.5, with a doctest of the route `least_squares_problem` → `padded_start` → rewrap).
> Confirmed: no twin.

## CHANGES

| round | # | source | what changed | where |
|---|---|---|---|---|
| 0 | — | session | first draft | — |
| 1 | 1 | critic M1; orchestrator (a) | **The seam is T3-free again.** Two optional, duck-typed backend geometry members carry the layer-specific work (the `zero_point_frame` precedent; D-9 dropped only the FRONTEND twins): `with_frame(frame)` and `aligned_padded_frame(x0, x_old, project, hint, …)`. `backend/optimizers.py` imports nothing new. On the uniform layer `x_old` is the full `.data`, so the old frame is built with its own masks. "No new member" removed | §1, §3.2, §3.5, §5.5, D-25 |
| 1 | 2 | writer #2; orchestrator #2 | `project` in the optimizer layer is the module-level `_misfit_gradient_at(frame, hint)` with `hint = (problem, residual)`. No closure | §3.5, §5.6, D-26 |
| 1 | 3 | critic M2; orchestrator (1) | `'cone'` uses the residual at the ZERO tensor, `point_forward(0) − data`, as `'zero'` does. The driver computes the residual at the point the loop is for. Tests added: `cone_max_passes=0` with `cone_oversampling=0` equals `'zero'`; the value does not depend on `x0`'s values | §2.5, §3.5, T14(d) |
| 1 | 4 | critic M3; writer #3; orchestrator (b) | No `NewtonInfo` field. New `stats` keys appear only on runs that use a new keyword. G-bit means the whole `stats` dict is bit-identical without them. The stale `stats['newton']` reference is fixed. No display change | §3.5, §5.6, T14(c), D-27 |
| 1 | 5 | critic M4; orchestrator (2) | **A jit boundary for the uniform step.** One module-level kernel takes its statics explicitly: the masks and shape through `common.partition_static` (value-hashed), and the split and `update` as static arguments. It is jitted once through a memoized wrapper. Scan bodies are module-level and closure-free. Per-level program count stated. T13 rewritten as a compile-count test | §3.1(6), §3.4, §4.4, §3.5, T13, D-34 |
| 1 | 6 | critic M5; orchestrator (3) | Three never-prototyped paths are marked **not verified**, with T14 rows: the uniform kinds' gradient at block-layout frames; the tied projection under sharing; `hessian='exact'` from a supplied start frame | §2.3, §3.4, §10, T14(e)–(g) |
| 1 | 7 | critic MINOR 6 | The uniform factorization works on the PROJECTED content `Y⊥ = (I − ĀĀᵀ)Y`, with a guarded scaling for `Y⊥ = 0`. Transported stores likewise | §3.4, §4.4 |
| 1 | 8 | critic MINOR 7; writer #9; orchestrator | `pad_safe_pinned_completion(pinned, pinned_mask, seeds, seed_mask, row_mask)`: per-element counts as masks, static output width `m`, inputs masked at entry, and the real-support argument. `pad_safe_svd`'s permutation is now described correctly: computed at run time from masks that may be traced. The new primitive permutes on the host | §3.4, §5.2, T7, D-30 |
| 1 | 9 | critic MINOR 8, 9, 11 | Every signature written out in house style in §5, and §1 is now a name list. Uniform arguments are named `frame_data` / `variations_data`. Public entries take `sharing` labels, not `groups`. The uniform step takes target ranks (default: the frame's own) and derives its counts and P-block widths. `level_widths` names the loop's compaction widths | §1, §5, D-22 |
| 1 | 10 | critic MINOR 10; writer #6; orchestrator | D-22 resolved: `fv_aligned_padded_frame` (ragged) and `ufv_aligned_padded_frame` (uniform) | §1, §5, D-22 |
| 1 | 11 | critic MINOR 12 | Transport plus truncation in one pass is a structural `ValueError` in both steps. `update` must be a non-empty set of `U`, `L`, `R`, `O` without repeats | §2.2, §4.1, T3 |
| 1 | 12 | critic MINOR 13; writer #12 | Loop edge cases fixed. The truncating pass counts against the cap. No oversampled phase when working = target. `max_passes = 0` is legal only when working = target. The stop rule is stated exactly. Movements come back as a tuple (one `C` array per pass) in both layers | §2.3, §4.3, §4.6, T6, D-36 |
| 1 | 13 | critic MINOR 14; writer #5; orchestrator #5 | **No `None` for the zero tensor in the frame functions.** It is the EMPTY frame, built by `fv_empty_frame(shape, stack_shape, use_jax)` / `ufv_empty_frame(shape, N, stack_shape, use_jax)`. The parameters that were "read only when `old_frame` is None" are gone. The step's old ranks are plain ints. Frontend spelling: a rank-0 `x_old`. The two caveats are handled structurally: the uniform orthogonalization raises on it, and `compute_minimal_ranks` maps the last boundary rank to 0 | §2.4, §3.3, §3.6, §5, D-21, D-29 |
| 1 | 14 | critic MINOR 15; orchestrator #4 | The `PaddedStart` class is gone; `padded_start` returns `(frame, movements)` | §5.6, D-28 |
| 1 | 15 | critic MINOR 16; orchestrator | Fourteen keywords on `newton_cg`: `padded_alternate` and `cone_alternate` added. Option defaults are `None` (= the D-6 value), so an option set where its group is not engaged raises (house rule). `gradient_descent` gets the six cone keywords. `None` seeds mean `ZERO_FRAME_SEED` | §1, §5.6, D-6, D-33 |
| 1 | 16 | critic MINOR 17 | The frontend's validation and conversion list is written out (layer, shape, stack, `N`; stores; same tensor; ties; reduction; `use_jit` conversion; corewise; a no-op level) | §3.6 |
| 1 | 17 | critic MINOR 18; writer #16 | The frontend loop is one twin per layer, each beside its frame class. This fixes the import cycle; reasons in D-35. The `N` check, the cross-layer hint error and `project`'s calling convention are stated. The frontend `padded_start` twin is dropped: it is not in D-8, and `newton_cg(padded_from=)` and the loop twins cover it | §3.6, §5.7, D-35 |
| 1 | 18 | critic MINOR 19 | Integer definitions. `Na` and `Rb` are the old frame's PADDED widths. The level widths come from `ufv_frame_widths(N, n, r) = (min(n, N), min(n, N, r²), r, r)`: checked by the writer on 60 random structures; not a prototype claim; pinned by T17. The member reads the level widths off its own `var_masks` | §3.4, §5.3, §5.5, T17 |
| 1 | 19 | critic MINOR 20 | The group precondition: equal A-block extents within a group is a structural check; tied values are a frontend safe-mode check | §3.3, §3.4, §3.6 |
| 1 | 20 | critic MINOR 21; writer #7 | `ranks.oversampled_ranks` widened to `(d,)+C` HOST int arrays, as `compute_minimal_ranks` does | §5.4, T19 |
| 1 | 21 | critic MINOR 22, 24; writer #13 | Tests rewritten per D-24 (bitwise only for structural facts): T2 one-sided float32 bound; the cycle test moved to N3 with a ported transport reference; T5 against S2 outputs frozen before deletion, and Tucker factors only to 1e-14; isolated per-family rejection tests with a mutation check; exact output masks for the start and the step; finite garbage written into the clean layout; `d = 1`; every test file named | §6, §8 |
| 1 | 22 | critic MINOR 23; writer #9 | Grouped `contract('…')` strings are listed with the engine's letters (Appendix C; all within `AXIS_SIZE`). HISTORICAL lives in `tests/test_contractions_interpreter.py`. N1 removes S3's ten strings | §3.1(4), §8, App. C |
| 1 | 23 | critic MINOR 25; writer #11; orchestrator (c) | Claims corrected. Every g06 arm used the SIMULTANEOUS step; with p = 5 for 2 passes, two-colour differs from it only from the fourth pass on. N10 runs the shipped default. "≤ 0.5 %" corrected to 1.4 % at pass 4 (0.5 % at pass 6). The §0 headline is qualified. "Monotone" is scoped to its measurement. A script column is added to the facts table | §0, §2.3, D-32 |
| 1 | 24 | critic MINOR 25 | Docs list extended: `api_reference.rst`, `uniform_svd_prefix_orthogonalization.md`, the `'cone'` rows, `rank_continuation.md`'s uniform section, `verification_records.md` | §7 |
| 1 | 25 | critic MINOR 26 | Slice table corrected (files, test files, N2/N3/N6 test moves, N7 adds `geometry.py`) | §8 |
| 1 | 26 | writer #8 | `xscan` has no `reverse`. Reverse sweeps run on reversed inputs, as the library's right sweeps do (`tt_reverse`, `rev_Q`) | §4.1, §4.4 |
| 1 | 27 | writer #14 | The zero-tensor start's dtype is stated: the backend default float, as `t3_zeros`. So `'cone'` runs in float64 on float32 numpy data | §2.5, §10 |
| 1 | 28 | writer #1; orchestrator | `Problem.local_model(frame=)`: a T3-free structural check (exact comparison with `geom.frame(x)`: every core shape, `shape`, all four mask families with boundary entries). Safe-mode numerical checks stay in the frontend | §3.5, D-37 |
| 1 | 29 | critic NIT 27–37 | Haar seeds through `pinned_completion` (no private import). Old-frame check worded. Unclear sentence deleted. Gnorm not duplicated. Doc sentences added (`'warm_start'` with a start frame; the regularizer). T6/T12 cases defined. Seed defaults written out. Scripts per fact | §2.3, §3.2, §3.3, §6, §7 |
| 1 | 30 | writer | Import DAG (§3.7) and a permanent import-graph test (T18; J-5) | §3.7, T18 |
| 2 | 1 | critic r2 MAJOR 1; orchestrator | **D-37 replaced.** `local_model(frame=)` no longer orthogonalizes `x` to read structure. It calls a third optional member, `require_frame_structure(x_cores, frame)`, which uses integers and held masks only. Ragged: the frame's core shapes against `compute_orthogonal_representation_ranks` of `x`'s ranks, plus `C` and `N`. Uniform: variation masks against `var_masks`, the two boundary entries, `shape`, and the widths. Absent member: `ValueError` naming it. The frontend `start_frame` check uses it too | §3.5, §3.6, §3.7, §5.5, §5.6, §7, T14(a), D-25, D-37 |
| 2 | 2 | critic r2 MINOR 2 | A rank-0 `padded_from` has a backend spelling: both members accept rank-0 `x_old` data. The uniform member detects it structurally (empty Tucker masks and interior bond masks) and builds `ufv_empty_frame`; the ragged member's `self.frame` already handles it. `None` stays only for the cone (`padded_start`, the members). New test T14(h) | §3.5, §3.6, T14(h) |
| 2 | 3 | critic r2 MINOR 3 | The uniform loop's "nothing to complete" return is `ufv_compact_frame(old_frame_data, level_widths)`, keeping its return contract; T6 / T12 edge cases | §2.3, §4.6, T6, T12 |
| 2 | 4 | critic r2 MINOR 4 | The kernel's host data is stated. Every host-derived mask (per-group A-masks, row masks, one-hots) is precomputed as host bool. The group index tables travel as nested tuples of Python ints (static under `partition_static`); integer numpy arrays would be traced. The uniform module has its own memoized jit getter (`_jitted` is private to `backend/optimizers.py`) | §3.4, D-34 |
| 2 | 5 | critic r2 MINOR 5 | T14(g) split. (g1) At a numerically full-rank `x` with a gauge-rotated frame: the doctest's FD criterion, plus equality with the default frame's quadratic. (g2) At the padded `x0`: finite, and the objective decreases. FD is invalid at a zero-padded point | T14 |
| 2 | 6 | critic r2 MINOR 6 | `cone_*` options set while `g0norm_newton` is given raise (D-33's rule). `g0_reference='cone'` itself with `g0norm_newton` stays legal, as `'zero'` does | §3.6, T14(d), D-33 |
| 2 | 7 | critic r2 MINOR 7 | §5.7 written out in full: `UT3Frame.padded`, the frontend `ufv_aligned_padded_frame`, `UT3Tangent.align_padded`, and the frontend `newton_cg` / `gradient_descent` keyword blocks | §5.7 |
| 2 | 8 | critic r2 MINOR 8, "the four calls" | The backend `padded_start` is PUBLIC (in `__all__`), with a doctest of the `least_squares_problem` route. D-35's sentence corrected; the one use case of a frontend twin flagged to Nick. `_misfit_gradient_at` stays private | §3.5, §5.6, D-35, For the maintainer (3) |
| 2 | 9 | critic r2 MINOR 9 | T14(d): `'cone'` with `cone_oversampling=0, cone_max_passes=0` equals `'zero'` to ≤ 1e-12 relative (the Tucker factors agree to 1e-14, not bitwise) | T14(d) |
| 3 | 1 | critic r3 MINOR; orchestrator | **One rule for the zero old point, in all three places.** A tensor is STRUCTURALLY ZERO when any Tucker rank or any interior TT rank is 0. The critic checked that a partially-zero-rank `x_old` (Tucker `(0,2,2)`, TT `(1,0,2,1)`) has non-empty stores: the ragged member would pin arbitrary directions, and the uniform member would not recognise it. Now: the frontend converts any structurally-zero `x_old` to the canonical rank-0 tensor first; both backend members apply the same test and build the empty frame; T14(h) and T15 get the partial case | §3.5, §3.6, T14(h), T15, D-21 |
| 3 | 2 | critic r3 NIT | §5.7's frontend return annotations written as `typ.Tuple[typ.Union[t3.TuckerTensorTrain, ut3.UniformTuckerTensorTrain], dict]`, the current form (the module's `Point` alias) | §5.7 |
| 4 | 1 | orchestrator r4 (the Newton measurement) | The g06 Newton measurement added as a table to §2.3, with the plain reading, and as a baseline table in §11. Acceptance keeps the SHIPPED default; the 20×1000 run now has three arms (library / hint-free / default aligned) | §2.3, §10, §11 |
| 4 | 2 | orchestrator r4 | §0 rewritten to what was measured. The established value is the data-independent seeded start frame plus the data-aware cone reference; the alignment's Newton value is unproven at laptop scale and is to be measured at 20×1000. The design and defaults are unchanged | §0, §10 |
| 4 | 3 | orchestrator r4 | **Zero alignment passes** is a supported, documented spelling (D-38). `max_passes = 0` returns the hint-free start at the TARGET ranks, with oversampling ignored, no projection and empty movements. So `padded_max_passes=0` is the hint-free start, and `cone_max_passes=0` is the reference at the hint-free frame (= `'zero'` to rounding). Edge cases rechecked (`max_passes = 0 / 1`, working = / ≠ target, no-op level). The r1 rule "`max_passes = 0` only when working = target" is withdrawn | §2.3, §5.1, §7, T6, T14(d), D-36, D-38 |
| 5 | 1 | critic r5 MINOR 1 | The level-2 reading is stated exactly. The hint-free start and every aligned start with oversampling: 49 → 21–33. The no-oversampling start: 42 and 29. The noise is sourced (about 5 iterations, the 2026-11-02 investigation's "Noise calibration"), and the table's seed spreads reach 13. §0's basin bullet carries the exception (one 4-pass aligned run ends in the other basin) | §0, §2.3, §11 |
| 5 | 2 | critic r5 MINOR 2; orchestrator | Zero passes with other schedule options set RAISES in `newton_cg`, `gradient_descent` and `padded_start`, by D-33's rule. The message names the fix: remove the option or set a positive pass count. `*_seed` is read and allowed. The loops (literal defaults) document the unread values instead. §2.3's stale "only other place" sentence is replaced. D-38 and T14(d) updated | §2.3, T14(d), D-38 |
| 5 | 3 | critic r5 NIT 3–5 | §11's duplicated bullet removed, and the no-oversampling row added to its baseline table. T14(d) compares the uniform iteration-0 frame with the COMPACTED start. §0 now calls the cone reference "the design's second deliverable", with its evidence cited (the 2026-11-02 investigation; D-5, D-7), not "established" | §0, §11, T14(d) |
| 6 | 1 | orchestrator (from N3) | The oversampled phase is skipped when the working and target STORES match (`compute_orthogonal_representation_ranks` of each), not the ranks: the `+p` ranks can reduce to the target's stores near the boundaries. As implemented in N3; the reviewer verified on 40,000 random structures that the two rules never disagree where it matters. The uniform loop (N6) uses the same rule | §2.3, §4.3, §4.6, T6 |
| 6 | 2 | orchestrator (from N3) | The cycle case is a Python module of float literals (`tests/backend/_padded_frame_cycle_case.py`, checked bitwise against the source), because the repository forbids binary fixtures (`*.npz` is git-ignored). N3's review-stop clarification: T16 dispatch rows for every public backend function, host loops included | T4, T16, §8 N3, App. A |
| 6 | 3 | orchestrator (the higher-rank Darcy measurement) | Added to §2.3 and §11. Capture fractions at 1→6 … 16→21. Newton at 11 → 16: the aligned start with ≥ 10 passes cuts Newton by about 40 % and CG by about 2.4× against both other starts; 4 passes do not, which supports the cap of 10. §0 and §10 updated for consistency: the alignment's Newton value appears with rank on the laptop ladder | §0, §2.3, §10, §11 |
| 7 | 1 | Nick 2026-11-10 (via the orchestrator) | **Jit is under the user's control (D-39).** Backend functions stay eager; `use_jit=False` on both loops, the members, `padded_start` and the frontend twins. Ragged: a host loop around a memoized jitted step (static: old ranks, output ranks, `sharing`, `update`). Uniform: the whole loop as one `jax.jit` (`fori_loop`, a truncating call, a `(UL, RO)` `while_loop`, the compaction; static cap; `tol` and the hint traced; `project` traceable). "One jitted kernel" becomes "a jittable kernel, jitted only under `use_jit`". The measured reason (932 eager compilations, ~22 s at d = 4, against one ~2 s compile) is recorded in §2.3. `newton_cg`'s `use_jit` governs both CG and the alignment | §2.3, §3.1(6), §3.4, §3.5, §4.3, §4.4, §4.6, §5.1, §5.3, §5.5, §5.6, §5.7, D-20, D-34, D-39 |
| 7 | 2 | writer, from (2) | **The pair stop rule.** A `while_loop` body of one `'UL'` then one `'RO'` pass cannot stop between them without a parity `cond`. So both layers stop only after a `(UL, RO)` pair, and an odd remainder runs one final `'UL'` under a cond on the STOP FLAG (not parity), keeping `max_passes` reachable. N3's per-pass rule changes in the follow-up slice N3b, so eager, jitted and cross-layer runs agree pass for pass. The uniform movements become a fixed-length NaN-padded array in both modes | §2.3, §8 N3b, T6, T12, D-36 |
| 7 | 3 | writer, from (3) | `gradient_descent` has no `use_jit` (eager only, by design), so there is nothing to forward and its cone reference runs eagerly. This departs from the request's "likewise"; stated in §3.5, §7 and D-39 | §3.5, §7, D-39 |
| 7 | 4 | Nick 2026-11-10 | Docs: a "jit or eager" user-doc section (the rule, the figures). Tests: T6 / T12 eager = jit; T13 compile counts (ragged jitted step ≤ 4 programs; uniform whole loop ONCE per structure, `tol` traced); T14(i), the problem's gradient inside the uniform whole-loop jit (**not verified** before it). Slices: N3b (ragged `use_jit`, pair rule), N6 (uniform whole-loop jit) | §6, §7, §8 |
| 8 | 1 | Nick 2026-11-10 (D-41) | **Gaussian seeds everywhere, slot-indexed**: every store's start seeds are rows `w..w+k−1` of the keyed stream `(seed, store kind, group's first mode)`, of the ragged unfolding's length, host-generated and scattered; pinned completion as before. The coordinate convention is deleted. The zero-point start has random cores, so `'cone'` at zero passes is no longer `'zero'`. Slice N8b; T5 restated, T20 added, T8 / T14(d) amended; the Darcy hint-free baselines marked for re-recording at N10 | §2.3, §2.4, §2.5, §3.2, §3.4, §4.2, §4.5, §5.1–§5.3, §6, §8, §10, D-38, D-41 |
| 8 | 2 | Nick 2026-11-10 (D-40, D-42, D-47, D-48) | Recorded: the eager-jax risk is documentation only, with the memory-mapping finding (8,000–9,500 mappings per eager alignment; ~7 levels to `vm.max_map_count`) in §2.3, §7, §10. T14(f)'s root cause is basis choice at singular-value ties, documented in N9 with the numbers; canonical bases are a later audit item. The `functools.partial` binding is accepted in this instance only. N9's docstring requirements are listed in §7 | §2.3, §2.4, §6 T14(f), §7, §10, D-40, D-42, D-47, D-48 |
| 8 | 3 | session, as built (D-43–D-46) | `newton_cg` passes a structurally-zero `padded_from` through unconverted. Compaction pads first when the level is wider. A NaN movement never stops the loop (the record cannot tell it from an unrun pass). T15 in one file. Stacked tolerances 1e-13. The scatter helper renamed. Seven private helpers shared between the two modules (§3.2). The library's whole-loop jit first call is ~9 s | §3.2, §3.4, §3.6, §4.5, §5.3, §6, §7, D-43–D-46 |
| 9 | 1 | reviewer-verified 2026-11-10 (J-51; D-50) | The fixed point's characterization corrected to the VERIFIED statement. Each store's update maximizes its own content's capture given the others (Gauss–Seidel, not ascent of `E`). A fixed point is a stationary point of `E` but generally a saddle: a minimum per store, lowered by paired rotations; proved for matrices, measured on T3s. A strict local maximum of the cone merit is observed on small trees. The docs' unified wording is added verbatim | §2.2, §7, D-48, D-50 |
| 9 | 2 | Nick 2026-11-10 (D-49) | The Gaussian core seeds are not sign-covariant: a slot sign flip changes the completion's span (0.987 for one bond); the starts differ across layers (~1, even at generic points) and possibly across BLAS builds; Tucker is unaffected. Accept and document; sign canonicalization joins the canonical-bases audit item. "Representation" now includes slot signs. Re-measured washout on the Gaussian convention added. T8 compares starts on scattered copies of one ragged frame | §2.4, §6 T8, D-42, D-49 |
| 9 | 3 | orchestrator | T20's pivot bound is a property of the chosen seeds, not a guarantee. D-41 says the zero TENSOR's start changes (`fv_zero_point_frame` unchanged). §8: N9 done (cc7d6fc6), N10 in progress | §6 T20, §8, D-41 |
| 2 | 10 | critic r2 NIT 10–15 | Return tuples expanded one element per line. `_frame_theta` named correctly. The loop defaults are named once (`ALIGN_*` in `padded_frames.py`) and referenced everywhere; `padded_start` takes `None` and forwards only what is set. T18 builds the graph on module files. "`oversampled_passes` is then ignored" added. D-18 gains a bracketed gloss | §2.3, §3.5, §5, T18, D-18 |

## 0. Why, in one page

At a rank increase in rank continuation, the converged point `X_old` is zero-padded to the new ranks. Its frame at
the new ranks is the old frame, which must stay exactly as it is, plus a **completion**: new orthonormal directions
in every store that the data do not determine. Today Newton starts from the frame `t3_orthogonal_representations`
gives the zero-padded point, whose completion is whatever LAPACK's QR leaves at a numerically rank-deficient point.
The relative stop is referenced to one seeded Haar frame at the zero tensor (2026.3.0).

**What was measured** (laptop Darcy ladder, exact Hessian, the same absolute stop for every start, two start seeds;
table in §2.3):

- **At level 2 a generic seeded start roughly halves the Newton count.** The hint-free start and every aligned
  start WITH oversampling cut the library padded frame's 49 Newton iterations to 21–33. The no-oversampling
  4-pass start gave 42 and 29.
  - Noise: the 2026-11-02 investigation put single-trajectory noise at about 5 iterations on the e01 cell
    (`T3Polynomial/dev/zero_reference_cone_2026_11_02.md`, "Noise calibration"). The seed spreads in this table
    reach 13 (row (f), level 2: 42 vs 29).
  - At level 1 the counts are within that spread.
- **The alignment captures far more of the gradient**: 2–5× the frame-gradient norm of the hint-free frame (level
  1: 8.4 → 36.4 after four passes, 40.3 after forty).
- **It did not reduce Newton counts further at levels 1–2** beyond the seed spread. At the higher-rank level 11 → 16
  (§2.3), the aligned start with 10 or more passes cut Newton iterations by about 40 % and CG iterations by about
  2.4× against both the library and the hint-free starts; 4 passes did not.
- **The start decides the basin.** At level 1 the aligned starts end at objective 3.3096 / test error 0.180, the
  others at 3.2927 / 0.190. One of the two 4-pass aligned runs ends in the other basin.

**So the feature delivers two things:**

1. **Established by the table: a data-independent, seeded generic start frame** at a rank increase. This is the
   hint-free padded frame, i.e. zero alignment passes (`newton_cg(padded_from=x_old, padded_max_passes=0)`, D-38).
2. **The design's second deliverable: a data-aware reference** gradient norm at the zero tensor,
   `g0_reference='cone'`, like-for-like with the gradient norms the stop compares it to (D-5, D-7).
   - Its motivation is the 2026-11-02 zero-reference investigation (`T3Polynomial/dev/zero_reference_cone_2026_11_02.md`).
   - The Newton table does not compare it with `'zero'`: every arm used the cone as its absolute stop.

The alignment's further value in Newton counts **appears with rank** on the laptop ladder: none at levels 1–2,
about 40 % at 11 → 16 with the default cap. It is to be measured at 20×1000 (§11). The design ships it whole and
with its defaults (D-6, D-7). It is cheap (≤ 10 % of a gradient per pass; 45 ms against a 1.1 s gradient at
11 → 16), and it is what the cone reference is.

**The alignment.** The completion is aligned with the gradient by a fixed-point loop

    F  <-  align(F, project(F, hint)).

- `project(F, hint)` is the orthogonal projection of one fixed tensor `S` onto the tangent space at the current
  padded frame `F`. `S` is the gradient at `X_old`, or any tensor.
- `align` replaces every completion by the leading new directions of that tangent vector, with the old blocks held
  fixed.

The loop starts from the seeded hint-free padded frame. It runs a few passes with the completion oversampled, then
passes at the target ranks, updating the two edge directions on alternate passes. It runs once per rank set, at the
outset, in two places:

- the Newton start, `newton_cg(..., padded_from=x_old)`;
- the reference at the zero tensor, `g0_reference='cone'`.

Zero passes is a supported spelling: the start itself (D-38).

**What changed since the superseded plan.**

- The hint is no longer embedded as a doubled-rank T3 and processed by a dense-style construction (width-`2R`
  factorizations; 22 s per step at `N = 64`, ranks 41 → 46). The step is the note's core form on the frame and its
  variation cores: 72 ms, same output to 1e-14.
- The uniform layer gets a truly uniform twin in a **block layout**.
- The loop updates the two edge directions on alternate passes, because the simultaneous update cycles on Darcy.

Release policy, style and the optimizer seam carry over from the superseded plan (§9).

## 1. What ships (names; the signatures are in §5)

| layer | module | new or changed |
|---|---|---|
| backend, ragged | `backend/padded_frames.py` (new) | `fv_empty_frame` (the frame of the zero tensor), `fv_padded_frame` (the start), `tv_align_padded_frame` (one step), `fv_aligned_padded_frame` (the loop) |
| backend, uniform | `backend/uniform_padded_frames.py` (new) | `ufv_empty_frame`, `ufv_padded_frame`, `utv_align_padded_frame`, `ufv_compact_frame`, `ufv_aligned_padded_frame` |
| backend, shared modules | `backend/linalg.py` | `+ pad_safe_pinned_completion` |
| | `backend/ufv_masking.py` | `+ ufv_frame_widths` |
| | `backend/ranks.py` | `oversampled_ranks` widened to per-element arrays (S1 function, committed, never released) |
| backend, geometry | `backend/geometry.py` | optional members `with_frame`, `aligned_padded_frame` and `require_frame_structure` on `ManifoldGeometryOps` and `UniformManifoldGeometryOps` (sharing through their `groups` field). The corewise geometries do not define them |
| backend, optimizers | `backend/optimizers.py` | `Problem.local_model(..., frame=None)`; the private module-level `_misfit_gradient_at`; the PUBLIC `padded_start` (in `__all__`); fourteen keyword-only `newton_cg` keywords; `g0_reference='cone'` with six keywords on `newton_cg` and `gradient_descent` |
| frontend | `tucker_tensor_train.py`, `uniform_tucker_tensor_train.py` | `oversampled_ranks` (method) |
| | `frame_variations_format.py`, `uniform_frame_variations_format.py` | `T3Frame.padded`, `UT3Frame.padded` (static); `fv_aligned_padded_frame`, `ufv_aligned_padded_frame` (the loop twins) |
| | `manifold.py`, `uniform_manifold.py` | `T3Tangent.align_padded`, `UT3Tangent.align_padded` |
| | `optimizers.py` | the new `newton_cg` / `gradient_descent` keywords, forwarded only when set |

Release: **2026.4.0, additive and non-breaking.** Every default is unchanged (`g0_reference` stays `'zero'`;
`fv_zero_point_frame` is untouched), and every new keyword is appended keyword-only.

## 2. Mathematics (the note, specialized to T3s)

### 2.1 Frames, blocks, directed edges

A T3 is a tree: Tucker leaves `l_i` (physical size `N_i`) and TT cores `c_i` in a chain. One frame core per DIRECTED
edge gives exactly the library's four stores:

| directed edge | store | core | output slot |
|---|---|---|---|
| `l_i -> c_i` | `up[i]` | `U_i  C+(nU_i, N_i)` | the Tucker index |
| `c_i -> l_i` | `down[i]` | `O_i  C+(rL_i, nD_i, rR_{i+1})` | the down index |
| `c_i -> c_{i+1}` | `left[i]` | `L_i  C+(rL_i, nU_i, rL_{i+1})`; for `i ≤ d-2` (`L_{d-1}` is the carrier of the base point) | bond `i+1` |
| `c_i -> c_{i-1}` | `right[i]` | `R_i  C+(rR_i, nU_i, rR_{i+1})`; for `i ≥ 1` (`R_0` is the carrier) | bond `i` |

**A padded frame** of `X_old` at stores `(nU, nD, rL, rR)` has two parts:

- **The A-block.** The old frame, bitwise, in the LEADING slots of every store. Its widths `nU°, nD°, rL°, rR°` are
  `ranks.compute_orthogonal_representation_ranks(shape, old ranks)`. A frame's own `(up, left)` widths reproduce its
  stores; this held on 4000 rank sets (`P/README.md`, "Findings" 3).
- **The P-block.** A completion `P` in the trailing slots, of width `k = new − old` per directed edge.

An old frame core is zero unless every input slot is in its A-part. The two chains reconstruct `X_old` through the
carriers, which hold the old content only.

**The zero tensor** has the empty old frame: stores `(0,…,0), (0,…,0), (1,0,…,0,1), (1,0,…,0,1)`, i.e. zero-width
rank axes except the boundary bonds. It is the orthogonal representation of the rank-0 T3, Tucker ranks `(0,)*d` and
TT ranks `(1,)+(0,)*(d-1)+(1,)` (checked by the critic and the writer: `t3_orthogonal_representations` of
`TuckerTensorTrain.zeros(shape, (0,)*d, (1,0,…,0,1))` returns exactly these stores).

**Variations** `(V, H)` at the frame are the library's gauged variations (`U_i V_iᵀ = 0`, `L_iᵀ H_i = 0`). The step
does not depend on the gauge: ungauged variations give the same frames, to 5e-15 (`P/check_cases.py`).

### 2.2 One alignment step (the note's §10, unrooted message passing)

Given a padded frame `F` and a tangent vector `Z` at `F` with variation cores `(V, H)`, the step replaces every
completion by the leading left singular directions of `Z`'s normal content at that edge, with the old block pinned.
Two sweeps over directed edges.

**Step 1: far-side messages, old frames** (`Phi`, `Xi` per directed edge `a -> b`, from the far leaves inward):

- `Phi[a->b] = P_{a->b}ᵀ Z_far U'_{b->a}` (size `k × R`).
- `Xi Xiᵀ = P_{a->b}ᵀ Z_far (I − U'_{b->a} U'_{b->a}ᵀ) Z_farᵀ P_{a->b}`: a square root of a Gram matrix; any factor
  will do. Only `Xi Xiᵀ` enters the left singular subspace below.
- `Z_far` is the part of `Z` on `b`'s side.
- Order:
  1. all `down` edges (leaves have no other neighbour);
  2. the `left` edges from the far end;
  3. the `right` edges from the near end;
  4. all `up` edges.

**Step 2: new completions, new frames.** Order: all `up` edges; the `left` sweep `i = 0 .. d-2`; the `right` sweep
`i = d-1 .. 1`; all `down` edges. At edge `a -> b` the incoming slots of `a` are already in their NEW coordinates:
the old block unchanged, the completion block multiplied by `o[y->a] = P⁺_{y->a}ᵀ P_{y->a}`. Then:

    g_old = the old completion P_{a->b} in the new coordinates                 (N_a x k_ab)
    g_new = Z_near P_{b->a} = the node term of a on the far completion
            + sum over a's other neighbours z of (F[a->z] restricted) x_z D[z->a]
    Y     = [ g_old Phi_P + g_new | g_old Xi ]          Phi = [Phi_A | Phi_P]
    P⁺    = the leading k⁺ left singular directions of (I − ĀĀᵀ) Y, completed against Ā   (pinned_leading_columns)
    o[a->b] = P⁺ᵀ g_old,    D[a->b] = [Ā | P⁺]ᵀ g_new,    movement_e = ‖ g_old − P⁺ (P⁺ᵀ g_old) ‖_F

`Y` is exactly the edge's normal content (the note's identity `Y = Q (L J) Θᵀ`). So nothing wider than `old + 2k` is
factorized, and the tangent is never embedded as a doubled-rank T3.

**What a store's update does, and what the fixed point satisfies** (D-50; J-51, reviewer-verified 2026-11-10).

- Each store's update maximizes `‖P⁺ᵀ Y‖²` for its OWN content `Y`, given the other stores' completions. The passes
  are alternating, Gauss–Seidel-type updates of these per-store problems. They are NOT an ascent method for the
  captured frame energy `E = ‖Π_{T(F)} g‖²`.
- A fixed point is a stationary point of `E` but generally a SADDLE of it:
  - per store it is a minimum: rotating one store's completion alone never lowers `E`, and generically raises it at
    second order;
  - a paired rotation of two stores lowers it.

  This is proved for matrices (`docs/padded_frame_cone.tex`, Proposition 1) and measured on T3s in both layers.
- A strict local maximum of the cone merit is OBSERVED on small trees, one of many; it is not proved.
- The docs' unified wording, used verbatim in the user doc, the docstrings and the math note:

  > At a fixed point, every store's completion spans the leading new directions of its own content, given the
  > other stores' completions, so one more pass moves nothing. The completions are then a stationary point of the
  > captured hint energy ‖Π_{T(F)} hint‖² (proved for matrices, measured on T3s) but generally not a maximum of it:
  > rotating one store's completion alone never lowers that energy and generically raises it at second order. In
  > measurements on small trees, they are a strict local maximum of the cone merit, one of many.

The boundary bonds have old rank 1 and no completion. The messages of the nonexistent edges past the ends are
therefore zero-size arrays, and EVERY core runs the same body: both sweeps are folds with an initial carry (scans).

**Verified against the dense per-edge update** (`P/check_step.py`, `P/check_cases.py`, `P/check_v2.py`): one step
agrees to 3e-14 on all four stores for:

- d = 2..5, uneven increments, non-minimal ranks;
- a tangent part 10× the normal part;
- truncating steps, sharing groups, the zero old point, stacks.

Frames are orthonormal to 2e-15, and the old block is bitwise untouched. Float32 errors follow `u·max(1, s)`, and the
step is scale-invariant in the hint (`P/check_numerics.py`). **Not verified at `d = 1`**: the prototypes ran
d = 2..5. At `d = 1` the sweeps are empty scans; T1 and T9 add it.

**Truncation.** `k⁺ < k` (oversampled → target) is the same step with fewer selected directions.

**Transport and two-colour.**

- A store that is NOT updated in a pass is **transported**: its content is `Y = g_old` alone, the old completion
  re-expressed in the new candidate space.
- A pass that updates only `up` and `left` (toward the root at `c_{d-1}`), followed by a pass that updates only `right`
  and `down`, is block coordinate ascent with a projection between the halves.
- **A pass may not both transport and truncate.** A transported store can lose directions arbitrarily in its
  truncated neighbours' coordinates: σ ≈ 1e-17 where 2 must be kept (`P/README.md`, "Two-colour passes", "Not
  matched"). Both steps therefore raise `ValueError` when a store not in `update` has an output count below its
  input count. The loop never asks for it.

**Degenerate content.** When `Y` has fewer than `k⁺` nonzero singular values, the trailing directions are the
primitive's completion: valid, orthogonal to the old block, and arbitrary (roundoff- and backend-dependent). This is
accepted (D-12). Such directions carry no content and are dropped at truncation. A zero hint therefore moves the
completion, unlike the superseded dense form.

### 2.3 The loop and its facts

Let `working = ranks.oversampled_ranks(shape, target, p, sharing)`.

**The skip rule compares STORES, not ranks.** `n_over = oversampled_passes` if the working STORES differ from the
target STORES (`compute_orthogonal_representation_ranks` of each) for any stack element; else `0`. Near the
boundaries the `+p` ranks can reduce to the target's stores, and an oversampled pass would then be a wasted
projection. "Working = target" below means equal stores. The N3 implementer did it so; the reviewer verified on
40,000 random structures that the stores rule and the ranks rule never disagree where it matters. The uniform loop
(N6) uses the same rule.

With `oversampling = 0` or maximal ranks there is no oversampled phase,
and `oversampled_passes` is then ignored.

The loops ignore an unread value in three places, all stated in the user doc:

- this case;
- the D-36 clamp below;
- zero passes (D-38): `oversampling`, `oversampled_passes`, `tol` and `alternate` are not read.

The loops cannot tell a set value from a default, because their defaults are the `ALIGN_*` constants. The
optimizer-level keywords default to `None` and can: there, zero passes together with any other explicitly set option
of its group, except `*_seed`, raises (D-33, D-38).

    F_0 = the start (§2.4) at the working ranks
    pass it = 0 .. n_over-1  : F <- step(F, project(F, hint)) at the working ranks, update 'ULRO'
    pass it = n_over         : the same, OUTPUT AT THE TARGET RANKS (truncating when working ≠ target), 'ULRO'
    then, alternate=True:  PAIRS of passes ('UL' then 'RO') while a whole pair fits under max_passes;
                           stop after a pair when both of its movements are below tol;
                           if the cap leaves one pass slot and the loop has not stopped: one final 'UL' pass
          alternate=False: single 'ULRO' passes while under max_passes; stop after a pass whose movement is below tol
    return F, (movement of every pass)

**The pair rule (r7, D-39).** It makes the uniform whole-loop jit a `while_loop` whose body is one `'UL'` pass then
one `'RO'` pass, with no `lax.cond` on parity, and the ragged loop follows it too, so the layers agree pass for pass.
It changes the r1 rule (a stop test after every pass, the first pair being the truncating pass and the first `'UL'`)
in two ways:

- the stop is tested only on `(UL, RO)` pairs;
- an odd remainder is used by one final `'UL'` pass, run only if the loop has not stopped. This is a test on the stop
  flag, not on parity, so `max_passes` passes are still reached when nothing stops early.

N3's ragged loop is adjusted in its follow-up slice. With the defaults (p = 5 for 2 passes, cap 10), the passes are:
2 oversampled, 1 truncating, 3 pairs, and the final `'UL'`.

The movement of a pass is its maximum over edges, and the stop compares the maximum over the stack too (D-13).
When the target stores equal the old stores, there is nothing to complete: the loop runs no pass. The ragged loop
returns the old frame and an empty tuple of movements. The uniform loop returns
`ufv_compact_frame(old_frame_data, level_widths)`, so its return contract (level widths, prefix masks) holds, and
movements all NaN. **The uniform movements are a fixed-length `(max_passes,)+C` array, NaN after the last pass, in
both modes**, because the jitted loop needs a static shape (D-36, D-39).

**The cap counts the truncating pass.** If `oversampled_passes ≥ max_passes ≥ 1`, then `n_over = max_passes − 1`,
so the last pass truncates. This departs from the prototypes, which ran an extra truncating pass.

**Zero passes (D-38).** `max_passes = 0` returns the start (§2.4) built AT THE TARGET RANKS: the seeded hint-free
padded frame.

- At the loop level, `oversampling` and `oversampled_passes` are then ignored, since there is nothing to truncate,
  and `tol` and `alternate` are not read. The seed is read: it keys the start's Gaussian streams (D-41).
- Through `newton_cg`, `gradient_descent` and `padded_start`, `padded_max_passes=0` / `cone_max_passes=0` (or
  `max_passes=0`) together with an explicitly set `*_oversampling`, `*_oversampled_passes`, `*_tol` or `*_alternate`
  is a structural error. The message names the fix: remove the option, or set a positive pass count.
- `project` is never called and `hint` is never read; movements are empty (ragged: `()`; uniform: shape `(0,)+C`).
- The uniform loop compacts the start to `level_widths` as usual.
- With target stores equal to the old stores, it is the no-op return below, which is the same thing.

`max_passes = 0` is never an error. Checked against the other edge cases:

- `max_passes = 1` with working ≠ target runs one truncating pass from the working start (`n_over = 0`);
- `max_passes = 1` with working = target runs one `'ULRO'` pass at the target;
- `max_passes = 2` with working = target runs the `'ULRO'` pass, then the final `'UL'` (one slot left, no pair);
- the stop rule is never reached at `max_passes ≤ 2`.

**`project(frame, hint) -> variations`** must be the orthogonal projection of ONE fixed tensor onto the tangent space
at `frame`. Library choices: `tv_project_dense_onto_tangent_space`, `tv_project_t3_onto_tangent_space`,
`utv_project_ut3_onto_tangent_space`, or the optimizer layer's `_misfit_gradient_at` (residual computed once). `hint`
is handed to `project` unread.

**Facts the defaults rest on.** Every g06 arm (`PART=loop | zero | newton`) used the SIMULTANEOUS step
(`core_form2.align_stores`, all stores every pass). With p = 5 for 2 passes the two-colour default differs from it
only from the fourth pass on. Acceptance (N10) runs the shipped default.

| fact | script |
|---|---|
| The simultaneous update **cycles**: period 2 on Darcy level 1 (from about pass 400) and on a 243-number case. The stationary point (merit 0.22855) lies above the cycle (0.22712) and is not an attractor of that map | `P/cycle/cycle_schemes.py`, `P/cycle/cycle_variants.py`, `T3Polynomial/scripts/g06c_cycle_repro.py`, `g06d_cycle_minimal.py` |
| Two-colour, measured on the 243-number case with p = 0 (dense): converges with no merit decrease (0 of 299 passes). Not measured after a truncating pass, nor pass by pass on Darcy | `P/cycle/cycle_schemes.py` |
| Two-colour on the Darcy level-1 gradient (seed 0, p = 5 for 2; sampled passes): 35.9 vs 36.4 at pass 4 (−1.4 %); 37.0 vs 37.2 at pass 6 (−0.5 %). Its movement falls (5e-4 at pass 150) where the simultaneous update sticks at 1e-2 | `P/cycle/darcy_schemes.py` |
| Inertia (a proximal weight on the old completion) slows Darcy badly; rejected | `P/cycle/darcy_schemes.py` |
| Darcy, simultaneous step: the movement falls below 0.05 only after 18–37 passes. Passes 8–10 give about 93 % of the 40-pass value. The long-run value depends on the seed (36.1 / 40.5 / 41.7 at level 1) | `T3Polynomial/scripts/g06_aligned_frame.py` (`PART=loop`), `g06b_movement_diag.py` |
| Toy family, simultaneous, p = 5 for 2 passes then target passes: 97 % of the converged merit in 4 passes, 99.7 % in 6. Oversampling alone plateaus at 88 %. Two-colour on the same family: 97.6 % / 99.7 % | `P/check_schedule.py`, `P/cycle/toy_schemes.py` |
| At the zero tensor, simultaneous step: 95 % (rank 6) / 99 % (rank 11) of the converged frame-gradient norm in 10 passes. Converged values agree with the 2026-11-02 investigation (48.9 / 56.7). **Two-colour at the zero tensor: not measured** | `g06_aligned_frame.py` (`PART=zero`) |
| Per pass, the step costs ≤ 10 % of a gradient: 15 / 105 ms at level 1, 24–55 / 500–800 ms at level 2 (ragged numpy) | `g06_aligned_frame.py` |
| The movement tracks the true frame movement within 15 % from 1 down to 1e-15. The shortcut `sqrt(k − ‖o‖²)` floors at 1e-8 | `P/check_v2.py` |
| At d = 24 with a near-tied hint spectrum, the defaults do not converge in 10 passes in either layer | `P/check_uniform_loop.py` |
| Uniform loop = ragged loop on generic hints: same pass counts, movements ≤ 2e-14, spans ≤ 3.7e-14. With rank-deficient content during oversampling, the layers' movements differ by 2e-2 (those completions are a choice); they agree to 3e-15 from the truncating pass on | `P/check_uniform_loop.py`, `P/check_uniform_groups.py` |
| Newton from each start on the Darcy ladder: the table below | `g06_aligned_frame.py` (`PART=newton`) |
| **Eager jax is costly on first call; jit is under the user's control (D-39).** The ragged step's first EAGER call on jax arrays at d = 4 issues 932 op-by-op compilations (einsum 290, dynamic_slice 172, moveaxis 98, concatenate 84, reshape 48, subtract 42), about 22 s, then 0.16 s on the second call. Jitted, it is one ~2 s compile. The uniform whole loop under one `jax.jit` compiles in about 6 s at d = 6 and at d = 24 (prototype); the library's whole-loop jit first call is about 9 s at d = 6 and 24 (N6). Each eager-jax alignment at a new structure compiles hundreds of executables and adds 8,000–9,500 process memory mappings; an eager-jax rank continuation aligning at every level could pass `vm.max_map_count` (65,530) after about 7 levels and abort (`LLVM ERROR: Unable to allocate section memory!`, the failure `scan_body_principles.md` records) | measured at N3, N6 and N8 (session); `P/check_uniform_loop.py` part E |

So: the pass cap is the effective stop, and the tolerance is an early exit. The `'cone'` reference is "the value
after the default schedule", deterministic given the seed (D-7).

**The Newton measurement** (`T3Polynomial/scripts/g06_aligned_frame.py`, `PART=newton`; data under
`T3Polynomial/data/g0_reference/`).

- Setup: laptop Darcy o01 ladder, exact Hessian with the trust region, the same absolute stop for every start
  (`g0norm` = the cone reference, `gtol 3e-4`), levels 1 → 6 and 6 → 11, two start seeds. Injection of the start
  frame was verified on every run.
- The hint-free start in these runs is the COORDINATE convention (pre-D-41). Every aligned arm started from it. The
  Gaussian slot-indexed start (D-41) is the same in expectation; the hint-free rows are re-recorded at N10.
- Every aligned arm used the SIMULTANEOUS step, with fixed pass budgets and no movement stop. With p = 5 for 2
  passes, two-colour differs from it only from the fourth pass on.
- Cells: frame-gradient norm `‖P_F G‖` at the start · Newton iterations / CG iterations · final objective · test
  error. Values are given per seed, separated by `/` or `,`; a single value is one run or two runs that agree to the
  digits shown, and `—` was not reported.

| start | level 1 (1 → 6) | level 2 (6 → 11) |
|---|---|---|
| (a) the library's padded frame (`t3_orthogonal_representations` of the zero-padded point; today's default) | 15.3 · 21/181 · 3.2927 · 0.190 | 5.47 · **49/1480** · 1.6706 · 0.1170 |
| (b) the hint-free padded frame (`padded_max_passes=0`; COORDINATE convention, superseded by D-41; to be re-recorded at N10) | 8.4 / 7.9 · 16/254, 19/197 · 3.2927 · 0.190 | 4.35 / 4.54 · 28/714, 24/687 · 1.6716 · 0.1168 / 0.1170 |
| (c) aligned, p = 5 for 2 passes then target, 4 passes in all | 36.4 / 35.2 · 17/286, 11/109 · 3.3096 / 3.2927 · 0.180 / 0.190 | 15.5 · 22/648, 28/1031 · 1.6728 · 0.1143 |
| (d) as (c), 6 passes | 37.2 / 36.5 · 15/175, 20/271 · — · — | 16.1 / 16.3 · 27/578, 30/665 · — · — |
| (e) as (c), 10 passes (the default cap) | 37.6 / 37.7 · 20/273, 25/276 · 3.3095 · 0.181 | 16.6 / 16.7 · 28/1360, 33/873 · — · — |
| (f) p = 0, 4 passes | 32.5 / 29.8 · 14/246, 24/504 · — · — | 13.1 / 12.5 · 42/1151, 29/758 · — · — |
| (g) as (c), 40 passes | 40.3 · 16/181, 18/273 · 3.3096 · 0.180 | 16.8 · 21/729, 24/801 · 1.6732 · 0.1155 |

The reading, stated plainly:

- At level 2, the hint-free start (b) and every aligned start WITH oversampling (c, d, e, g) cut the library padded
  frame's 49 Newton iterations to 21–33. The aligned start without oversampling (f, 4 passes) gave 42 and 29.
- Noise: the 2026-11-02 investigation put single-trajectory noise at about 5 iterations on the e01 cell
  (`T3Polynomial/dev/zero_reference_cone_2026_11_02.md`, "Noise calibration": 17 vs 22 at identical frames). The seed
  spreads in this table reach 13 (row (f), level 2).
- The alignment passes do not reduce Newton counts further beyond that spread, at either level.
- The start decides the basin. At level 1 the aligned starts end at 3.3096 / 0.180 instead of 3.2927 / 0.190; one of
  the two (c) runs ends in the other basin.
- The aligned start captures 2–5× the frame-gradient norm of the hint-free one.

**The higher-rank Darcy measurement** (simultaneous step; starts from the o01 ladder's converged plain-start
checkpoints).

Capture fraction `‖P_F g‖ / ‖grad f‖` at the start (seed 1 within .01 of these):

| start | 1 → 6 | 6 → 11 | 11 → 16 | 16 → 21 |
|---|---|---|---|---|
| library padded frame | .189 | .117 | .106 | .101 |
| hint-free padded frame (coordinate convention; re-record at N10) | .104 | .093 | .094 | .093 |
| aligned, p = 5 for 2, 4 passes | .449 | .330 | .231 | .183 |
| aligned, 10 passes | .464 | .354 | .270 | .217 |
| aligned, 40 passes | .497 | .357 | .273 | .227 |

At 16 → 21 the loop had not settled by pass 40 (movement 0.07–0.15).

Newton at 11 → 16 (exact Hessian, the same stop; two seeds; cells Newton / CG iterations):

| library | hint-free | aligned, 4 passes | aligned, 10 passes | aligned, 40 passes |
|---|---|---|---|---|
| 36/1657 | 32/1860, 32/1551 | 24/1517, 25/1246 | 21/699, 22/538 | 22/700, 22/703 |

- Final objective 0.9680–0.9689 and test error 0.0815–0.0839 across all arms.
- Per pass at 11 → 16, the gradient costs about 1.1 s (2.0 s oversampled) and the step 45 ms.
- Newton at 16 → 21 was out of reach on the laptop.

The reading, stated plainly:

- The generic completions capture a steady ~9–10 % of the gradient at every level. The aligned fraction falls with
  rank (50 % → 23 %), but with 10 or more passes it stays 2.3–4.8× the generic one.
- At 11 → 16 the aligned start with 10 or more passes cuts Newton iterations by about 40 % and CG iterations by
  about 2.4×, against both the library and the hint-free starts. At levels 1–2 it had not.
- 4 passes are not enough there (Newton 24 / 25, CG barely reduced), which supports the cap of 10 (D-6, D-7).

So the alignment's value in Newton counts appears with rank on the laptop ladder: none at levels 1–2, about 40 % at
11 → 16. It remains to be measured at 20×1000 (§11). The defaults (D-6) are unchanged.

### 2.4 The start: Gaussian seeds everywhere, slot-indexed (D-41, both layers)

**Seeds.** Every store's completion seeds, for the Tucker factors and the TT/down cores alike, are rows of a keyed
legacy `RandomState` stream. `RandomState` is used because it is version-stable.

- **Key:** `(seed, store kind, the sharing group's first mode)`. For the TT/down stores the "group" is the mode
  itself.
- **Slot-indexed:** completion slot `s` of a store uses stream row `s`. A start at old width `w` draws rows
  `w .. w+k−1`, so a start-from-start chain never reuses a row.
- **Row length:** the ragged unfolding's length. That is `N_g` for a Tucker factor, and the unfolding's real input
  slots for a TT/down core (`rL_i·nU_i` for a left core, …).
- Rows are generated on the HOST at the ragged size, per group and per stack element.

**Completion.** `pinned_completion(old block, seedsᵀ, k)` for every store, as before: the sign-fixed Householder
completion in seed order. The old block is `U°ᵀ` for a Tucker factor, or the old core's unfolding columns, zero
outside the A-part. The coordinate-seed convention is deleted: its outside/inside branch and `_coordinate_seed_order`
in the ragged layer, the unit-vector scatter in the uniform one.

**Why (the floor CI job, numpy 1.26 / OpenBLAS).**

- The old Tucker stream restarted at row 0 on every call. A start-from-start chain therefore reused a seed lying in
  the current span: pivot 1e-16, with the completed column decided by BLAS rounding (0.349 between the layers on CI,
  1e-16 locally).
- The cores' outside-row coordinate seeds were chain-safe, but their inside fallback (no new rows) had the same
  structural hole on chains and on coordinate-structured points.
- Nick went further than the minimal fix, for two reasons. **Isotropy:** a uniformly random subspace of the
  complement is the natural no-information completion. **Conditioning:** pivots are bounded away from zero with high
  probability, and completions are continuous in the old frame off a null set.

**At the zero tensor** the start has random orthonormal TT/down cores (no identity prefix) and differently keyed
Tucker factors. It is therefore NOT `fv_zero_point_frame`'s frame: the `'cone'` value changes, and `'cone'` at zero
passes is a different valid convention from `'zero'` (D-12 stands; D-38 amended).

**The convention is basis-dependent.** A TT/down seed row is expressed in the slot coordinates of its neighbours'
completion blocks, so the completed span depends on the neighbours' bases. The uniform twin must therefore reproduce
the same Gram–Schmidt bases, not just the spans: the same host seed rows, scattered into the block layout, and a
pad-safe Householder completion (§3.4). The seed rows agree bitwise through the scatter (structural); the completions
agree to rounding (T8) where the two layers' old frames have the same slot signs. Across layers, the old frames' signs can differ (D-49), and then so do the starts.

**The start is a function of the old frame's REPRESENTATION, including slot signs** (D-42, D-49), because seeds
live in its bases.
Representations of one point in different bases give different starts, all valid (D-12). This happens at
singular-value ties, e.g. a `t3svd` output or a shared point (`share()` gives orthonormal Tucker factors). The aligned
loop's output is representation-independent up to its stop tolerance.

**Slot signs (D-49).** The Gaussian core seeds are not sign-covariant. A slot sign flip, which is arbitrary in an
unfixed LAPACK SVD, flips a core unfolding's rows: the old block flips with them, the seeds do not, so the
completion's span changes (0.987 for one bond).
- The ragged and uniform starts of even a GENERIC point differ by about 1.
- A hint-free start can differ across BLAS builds.
- Tucker completions are unaffected.

Accepted and documented. There is no local row-gauge rule and no representation-level sign canonicalization now;
the latter joins the canonical-bases audit item.

Measured washout:

| convention | starts differ in span | aligned frames agree, default tol | aligned frames agree, tol 1e-10 |
|---|---|---|---|
| coordinate (T14(f), N7) | 0.27–1.00 on `t3svd` / shared points; 1e-15 on generic ones | 2e-5 … 5e-3 | 3e-14 … 5e-10 |
| Gaussian (re-measured after N8b) | 0.95–1.00 | 7e-4 … 3e-2, equal pass counts | 8e-13 … 6e-11 |

Canonical bases (and signs) at ties are a later audit item.

### 2.5 The reference at the zero tensor

`g0_reference='cone'` runs the loop at the zero tensor of `x0`'s ranks: the old frame is the empty frame, and the
residual is `r0 = point_forward(0) − data`, exactly as `_zero_reference_gnorm` builds it (the regularizer is excluded
for the same reason). It reports the **frame-gradient norm** `‖P_{T_F} 𝒥ᵀ r0‖` at the returned frame, at the cost of
one more projection. This is like-for-like with the gradient norms the stopping test compares it to (D-5).

- It does not depend on `x0`'s values, only on its ranks.
- The default `'zero'` (one Haar frame, `fv_zero_point_frame`) is unchanged. Since D-41 the cone's start is the
  Gaussian slot-indexed frame, not `fv_zero_point_frame`'s, so `'cone'` at zero passes is not `'zero'`.
- The empty frame is array-free, so it takes the backend's default float (float64 numpy; jax's default). The cone
  therefore runs in float64 on float32 numpy data (superseded J1).

## 3. Design

### 3.1 The rules every slice is held to

1. **Backend functions are pure mathematical functions on immutable basic types**: arrays, tuples of arrays, ints,
   static int tuples, host bool masks.
   - No object is passed to be read for its shapes, and there is no dict of shapes.
   - Structure travels as `Sequence[int]` with `len=` shape comments.
   - Dispatch is inferred from the arrays. The array-free constructors (`fv_empty_frame`, `ufv_empty_frame`) keep a
     `use_jax` flag, as `t3_zeros` does.
   - The gold standard is `TuckerTensorTrain`, `UniformTuckerTensorTrain` and the backend they call. Precedent
     outside them is not justification.
   - The optimizer layer (geometry members, `Problem`, the drivers) is the documented relaxation: it may read a
     point's or its own masks' shapes, as `frame(x)` and `stack_shape(x)` do.
2. **Structural, not numerical.** Counts come from integers and masks. There is no rank tolerance and no epsilon
   content. The stopping tolerance and the oversampling are the user's tunables, not rank decisions.
3. **The uniform layer.**
   - Masks are the only statement of which slots are real. Inputs may be gappy, and junk may be (finite) garbage.
   - Never assume where junk sits in an input. A function may return its own layout (junk-last) with masks.
   - Truly uniform: no per-mode arrays of different shapes, and no Python loop over modes in the mathematics. Maps
     are batched over the leading mode axis; sweeps are scans.
   - Factorizations: `pad_safe_svd` in the step; `pad_safe_pinned_completion` in the start only (§3.4); a plain
     batched thin SVD in `root`, which is safe because its output `V·diag(s)` is zero on every null direction.
4. **Library code shape.**
   - Scans: `xscan`, the ragged host loop or `lax.scan` on supercores. Reverse sweeps run on reversed inputs.
   - Grouped `contract('…')` strings with the engine's letters, each recorded in HISTORICAL in
     `tests/test_contractions_interpreter.py` (Appendix C). No raw einsum, and no `@` on stacked operands. The
     prototypes' einsums and matmuls are the specification of each contraction.
   - Shape annotations on every signature.
   - One-line docstrings plus shape comments in the backend; descriptions, doctests and validation in the frontend
     twins.
   - Sharing safety: no reshape across group sub-axes (reshapes touch core axes only).
5. **`project` is a required argument in the backend loops; `hint` is opaque.** The frontend twins dispatch on
   classes. The optimizer layer's `project` is a module-level function with explicit data (D-26).
6. **Compile budget** (Nick).
   - Recompiling after a rank change is fine; a small fixed number inside a loop is tolerable; repeated recompiles
     inside a loop never.
   - Every scan body is module-level and closure-free. Statics come from operand shapes (e.g. `Kb = o.shape[-1]`,
     `Rb = rT − Kb`) or from an `lru_cache` factory keyed on hashable ints and bools (`scan_body_principles.md`
     §2, §4), never on arrays.
   - **Jit is under the user's control (D-39), as everywhere in the library.** Backend functions stay eager, and a
     `use_jit: bool = False` keyword on the two loops selects jit (the `newton_cg` / `mc_sgd` / `adam` precedent).
     Nothing jits unless asked.
     - Ragged, `use_jit=True` with jax inputs: the HOST loop calls a memoized module-level jitted step (an
       `lru_cache`d getter in `padded_frames.py`, the `_jitted` pattern). Its static arguments are the old ranks,
       the output ranks, `sharing` and `update`, so a level compiles two to four step programs and nothing
       recompiles across passes. The ragged loop is never jitted as a whole: its per-mode Python loops would unroll.
     - Uniform, `use_jit=True`: the WHOLE loop is one `jax.jit` (§3.4).
     - The uniform step is a JITTABLE module-level kernel with explicit statics (§3.4, D-34), jitted only under
       `use_jit` (inside the whole-loop jit).
     - The reason is measured (§2.3): an eager first step at d = 4 is about 22 s of op-by-op compilation against
       one ~2 s compile jitted.
   - The uniform loop returns its frame at the level's padded widths, so the first Newton step compiles nothing new.

### 3.2 Modules and layering

- **`backend/padded_frames.py` (new).** The ragged empty frame, start, step and loop. Imports downward only:
  `common`, `contractions`, `linalg` (`pinned_completion`, `pinned_leading_columns`, `pad_or_truncate`), `ranks`,
  `sharing` (`canonical_groups`), `fv_operations` (`ZERO_FRAME_SEED`, a public constant; no private helper). The
  `ranks → sharing` dependence is a smell for the later audit arc, not touched here.
- **`backend/uniform_padded_frames.py` (new).** The uniform empty frame, start, step, compaction and loop. Imports
  `common`, `contractions`, `linalg`, `ranks`, `sharing`, `ufv_masking`, `fv_operations` (`ZERO_FRAME_SEED`), and
  `padded_frames` for the shared private math helpers (D-46): seven private functions of `padded_frames`, including
  the slot-indexed seed rows (D-41), are imported by the uniform module. Accepted as a sibling-module convenience; they
  stay private (no `__all__` entry).
- **`backend/linalg.py`**: `+ pad_safe_pinned_completion` beside `pad_safe_svd`. `pinned_completion` and
  `pinned_leading_columns` (S1, committed) are unchanged.
- **`backend/ufv_masking.py`**: `+ ufv_frame_widths(N, n, r)`, the integer rule for the padded widths of
  `ut3_orthogonal_representations`' frame.
- **`backend/ranks.py`**: `oversampled_ranks` accepts `(d,)+C` HOST int arrays and returns arrays for them.
- **`backend/fv_operations.py`** returns to its pre-arc form (`efe5e3d`). The S2/S3 `fv_padded_frame` and its helpers
  are **deleted** (never released; D-3). `fv_zero_point_frame` is unchanged (D-19).
- **`backend/geometry.py`**: the members `with_frame`, `aligned_padded_frame` and `require_frame_structure` on the
  two manifold geometries (D-25). The uncommitted S4 member `padded_frame` is discarded.
- **`backend/optimizers.py`**:
  - `Problem.local_model(frame=)`;
  - the private module-level `_misfit_gradient_at`, and the public `padded_start` (added to `__all__`);
  - the `newton_cg` / `gradient_descent` keywords;
  - `'cone'` in `_check_g0_reference` and `_resolve_g0_reference`.

  **It imports no new module** and stays T3-free: every T3-specific operation is reached through the geometry
  (`curvature_internals.md`, "Everything T3-specific is behind those members").
- **`backend/optimizer_display.py`**: unchanged. Nothing new per iteration; the start's record is in `stats`.
- Frontend: §3.6.

### 3.3 The ragged layer

**Stores from ranks.** `stores(shape, tucker, tt) = compute_orthogonal_representation_ranks(...)`. The zero tensor is
the empty frame (`fv_empty_frame`), never `None`.

**The old frame** is given as a frame, because it is used: its cores are the pinned blocks. Its stores are read off
its shapes, and so are `shape`, the stack `C`, dtype and backend. Structural check: the old frame's `(down, right)`
widths must equal those its `(up, left)` widths imply through `compute_orthogonal_representation_ranks`. Otherwise it
is not an orthogonal representation's layout, and the call raises `ValueError`.

**Target stores** must be ≥ the old stores per edge, with boundary bonds 1 (structural).

**Stacks.** The frame stack `C` is on every core; the movement has shape `C`; the loop stops on its maximum over the
stack (D-13).

**Sharing groups.**

- One completion per group, from the members' contents side by side: `Y_g = [Y_{i_1} | Y_{i_2} | …]`. The factors
  stay tied bitwise: one array per group.
- Structural: equal old and target Tucker stores within each group.
- The old factors themselves must be tied. The step reads the group's first mode. That is a numerical precondition,
  checked in safe mode by the frontend twins (as `t3svd(sharing=)` does).

### 3.4 The uniform layer: block layout

**Layout.** Every rank slot of a padded frame is `[A-block | P-block]`, each padded to ONE uniform width with its own
mask:

- Tucker slots `[Na | Ka]`; bond slots `[Rb | Kb]`.
- `Na` is the old frame's padded Tucker width: `max(U°.shape[-2], O°.shape[-2])`. `Rb` is its padded bond width:
  `max(L°.shape[-1], R°.shape[-1])`. The narrower old supercores get junk slots appended (mask `False`); nothing
  moves.
- `Ka` and `Kb` are the maxima of the completion counts (`kU, kD` and `kL, kR`) over edges and the stack.
- The A-block is the old frame's supercores AS GIVEN, with gappy masks allowed. Junk is masked to zero once at entry.
  The P-block is appended, as under `+`.
- The ONLY positional fact is the split `(Na, Rb)`: two static ints shared by all modes. "Restrict this slot to its
  completion part" is therefore one static slice of the whole supercore.
- Frame data is the library's `(U, O, L, R, shape, (mU, mD, mL, mR))`, with `nU = nD = Na + Ka` and
  `rL = rR = Rb + Kb`.
- The empty frame (`ufv_empty_frame`) has `Na = 0`, `Rb = 1`, and boundary-bond masks `True` at bonds `0` and `d`.

**Step** (`utv_align_padded_frame`).

- Maps over modes are one batched `'dC…'` contraction each, plus one batched factorization.
- The edge factorization is `pad_safe_svd(M, row_mask, col_mask)` with:
  - `M = [Ā | Y⊥ / (4‖Y⊥‖_F)]` and `Y⊥ = (I − ĀĀᵀ) Y` (two contractions);
  - the guarded scaling: `Y⊥ = 0` gives the zero block, never `0/0`;
  - row mask = the outer product of the incoming slots' output masks;
  - column mask = `[the edge's A-block mask | ones]`.
- Its triplets come out in this order: the pins (σ = 1); the content by singular value (σ ≤ 1/4, because
  `Y⊥ ⟂ Ā`); then the real-supported completions. All are mutually orthogonal, with pads bitwise zero
  (`pad_safe_svd`'s contract).
- The `k¹` triplets after the pins are picked by a host one-hot at offset = the pin count (the A-block mask sum),
  restricted to the output P-mask.
- The sweeps are `xscan` (`lax.scan` under jax) with module-level, shape-uniform bodies and zero carries at the ends.
  Stacks ride as `C` after the mode axis.
- Groups use host tables, built from `canonical_groups(sharing, shape)`:
  - `members (G, m)` padded with the first mode, a validity mask, `first (G,)` and `group_of (d,)`;
  - the members' `Y⊥` side by side;
  - one factorization batched over `(G,)+C`, scattered back;
  - the old block from the first mode, so the output is tied bitwise;
  - structural: equal A-block up masks within each group.
- Transported stores use `Y⊥ = (I − ĀĀᵀ) g_old`.
- The output P-block widths are the maxima of the output counts (≤ the input widths; structural).
- **The kernel and its statics (D-34, D-39).** The public function prepares all host data on the host, in forms `partition_static`
  keeps static:
  - every host-derived mask as a host BOOL array: the output P-masks (the counts' prefixes), the per-group A-masks,
    the edge row masks, and the one-hot selections;
  - the group index tables (`members`, `valid`, `first`, `group_of`) as nested tuples of Python ints and bools. An
    integer numpy array is traced by `partition_static` (`common._is_static_leaf`), so no host integer array crosses
    the boundary.

  `partition_static` then splits `(frame_data, variations_data, host data)`. The function calls ONE **jittable**
  module-level kernel `_utv_align_kernel(dynamic, skeleton, split, update)`, jitted only under `use_jit` (D-39).
  Inside the kernel the tuples become `np` index arrays at trace time (host constants for the gathers).
  - The public step function, and the uniform loop with `use_jit=False`, call the kernel EAGERLY (numpy, or eager
    jax with module-level, closure-free scan bodies: each scan compiles once per body and shape).
  - The loop with `use_jit=True` calls it inside the whole-loop jit below. Its statics are fixed at trace time:
    `skeleton` (a value-hashed `StaticSkeleton`), `split` (an int tuple) and `update` (a str).
- **The whole-loop jit** (`ufv_aligned_padded_frame(..., use_jit=True)`, D-39; prototyped in `P/uniform_loop.py`:
  about 6 s of compile at d = 6 and 24). One module-level `_ufv_aligned_loop_kernel`, under an `lru_cache`d
  `jax.jit` getter in `uniform_padded_frames.py`. `optimizers._jitted` is private to `backend/optimizers.py`, which
  this module must not import.
  - **Static arguments:** the old frame's skeleton (masks, `shape`); the hint's static part via `partition_static`;
    `project` (a module-level function, hashed by identity); the per-element counts as host bool masks; the split;
    `level_widths`; `sharing`; `oversampling`, `oversampled_passes`, `max_passes` (the cap is static); `alternate`;
    `seed`.
  - **Traced:** the old frame's supercores, the hint's arrays (and registered pytrees, such as the `Problem` in
    `_misfit_gradient_at`'s hint), and `tol` (a value that may change between calls belongs in the state;
    `scan_body_principles.md` §5).
  - **Body:**
    1. the start;
    2. a `lax.fori_loop` for the oversampled passes, at the working shapes;
    3. one truncating call;
    4. a `lax.while_loop` for the target phase. Its body runs one `'UL'` pass then one `'RO'` pass, so there is no
       `lax.cond` on parity, and its state carries the pass counter. The stop test is on the iteration's pair.
       `alternate=False` uses a one-pass `'ULRO'` body;
    5. the final `'UL'` pass for an odd remainder, under `lax.cond` on the stop flag (§2.3);
    6. the compaction.
  - **Returns** the frame and the `(max_passes,)+C` movements, NaN after the last pass.
  - **`project` must be traceable** under `use_jit=True`. The library's projections are traceable; the problem's
    gradient (`_misfit_gradient_at`) is designed to be, and is **not verified** inside a whole-loop jit (the
    prototype jitted the loop with `utv_project_ut3_onto_tangent_space` only); T13 / T14 test it.
  - The eager and jitted paths are tested for equality (same pass counts, frames to 1e-12; T12, T13).

**Start** (`ufv_padded_frame`).

- The seeds are the ragged convention's (D-41), built on the HOST at the ragged size: the slot-indexed stream rows,
  per group (Tucker) or mode (TT/down) and per stack element. They are then scattered into the block layout, onto
  the REAL rows of each unfolding in their ragged row-major order (a cumsum of the masks) and onto the P-block slots
  of the counts. The scatter is a host gather of constants, so the seeds agree bitwise with the ragged start's
  (structural).
- Each store is one batched `pad_safe_pinned_completion`, a map with no sweep: the A-block as the pinned columns,
  the scattered seeds, and `seed_mask` = the P-block's count prefix. The coordinate unit-vector placement is deleted.
- It reproduces the ragged start to rounding (T8), with junk exactly zero.
- No scan, so it runs eagerly (jax's per-operation caches).

**`pad_safe_pinned_completion`** (D-11, D-30) is a sign-fixed Householder QR with host permutations:

- **Mechanics.** Static gathers built on the host from host masks:
  - rows: real rows first;
  - columns: `[real pinned | selected seeds (seed_mask) in order | the rest]`.

  The selected columns are scattered back to their seed positions, and zeros are written elsewhere. By contrast,
  `pad_safe_svd` computes its row permutation at RUN TIME by an `argsort` of a key built from the masks, which may
  be traced. The new primitive's masks are construction-time structure, so its permutations are host constants.
- **Contract.** Inputs are masked at entry (junk entries anything). Per element,
  `#seed_mask ≤ #row_mask − #pinned_mask` (structural).
- **Why the real support is exact.** The junk rows are zero and trail. Every Householder reflector therefore
  vanishes on them, so every selected column is bitwise zero there. No content decides a rank, every returned column
  is real by count, and Householder is deterministic: the two properties the SVD-only note asks for.
- **Why a QR here and nowhere else.** The start's semantics IS the column order of the seeds: the Gram–Schmidt
  completion of the Gaussian rows in slot order. Those bases are what the ragged convention produces, and what the
  neighbouring stores' seeds are expressed in (§2.4). The SVD-only rule of
  `docs/contributor/uniform_svd_prefix_orthogonalization.md` concerns orthogonalization sweeps, where the content
  decides the rank. A seeded completion is not one.

**Loop** (`ufv_aligned_padded_frame`).

- The schedule of §2.3, as a host loop by default, or as one jitted program under `use_jit=True` (D-20, D-39).
  `project` is called at block-layout frames.
- The library's `utv_project_ut3_onto_tangent_space` accepts gappy block-layout frames, verified at 5e-16
  (`P/check_uniform_loop.py`). It requires the hint's padded `N` to equal the frame's.
- **Not verified:** the uniform kinds' `precompute` / `transpose` at block-layout frames (the uniform
  `padded_start` path; §10, T14(e)).
- **At exit, compaction** (`ufv_compact_frame`):
  - gather real slots first on every rank axis: `argsort(~mask, stable)`, one `take_along_axis` per axis, with the
    same index array for every core axis an edge's mask governs;
  - zero the junk;
  - return the masks `ufv_make_frame_masks` builds from the integer stores at `level_widths`;
  - an axis whose level width exceeds the input's padded width is zero-padded first (D-44).

  Reason: `utv_retract` pads its output to the FRAME's padded widths, and the optimizer's geometry holds the level's
  prefix masks.

**Level widths.** `ufv_frame_widths(N, n, r) = (min(n, N), min(n, N, r²), r, r)` are the `(nU, nD, rL, rR)` widths
`ut3_orthogonal_representations` gives a point padded to `(N, n, r)`. The writer checked this on 60 random structures
(`n` up to `max + 2`, `N` up to `max + 1`); it is not a prototype claim, and T17 pins it. The geometry member does not
use the rule: it reads its own `var_masks` widths. The frontend twin uses it.

**Minimal ranks.** The uniform optimizer requires a minimal-rank start, and the frontend reduces `x_old` as it
reduces `x0`. Non-minimal inputs to the uniform functions are supported (`P/check_uniform_loop.py`); the backend loop
always takes `level_widths` explicitly.

### 3.5 The optimizer seam (T3-free)

**Geometry members** (`backend/geometry.py`). All three are optional and duck-typed (`getattr`, as
`zero_point_frame`). They are defined on `ManifoldGeometryOps` and `UniformManifoldGeometryOps`, and sharing comes
through their `groups`.

- `with_frame(frame)` returns the geometry at that frame's structure. Ragged: `self`, since the ragged geometry holds
  no rank structure. Uniform: `from_frame(frame_data, (frame_data[5][0], frame_data[5][2]))` with this geometry's
  `groups`. The plain masks are the frame's own up and left masks, gappy at block-layout frames, as
  `uniform_fitting._frame_theta` reads them.
- `aligned_padded_frame(x0, x_old, project, hint, oversampling, oversampled_passes, max_passes, tol, alternate, seed,
  use_jit)` runs the layer's loop, passing `use_jit` through.
  - Target ranks: from `x0`'s core shapes (ragged) or `self.masks` (uniform).
  - Old frame from `x_old`'s own data. One rule in both members: `x_old` is **structurally zero** when ANY Tucker
    rank or ANY interior TT rank is 0, read off the core shapes (ragged) or the mask sums (uniform). The test is per
    stack element; a stack mixing zero and nonzero elements is a structural error (the members serve the unstacked
    optimizers). A structurally-zero `x_old` gives the empty frame, read off `x_old`:
    `fv_empty_frame(shape, C, use_jax)` (ragged) or `ufv_empty_frame(shape, N, C, use_jax)` (uniform). Never its
    orthogonal representation: at partially-zero ranks (Tucker `(0,2,2)`, TT `(1,0,2,1)`) that has non-empty
    stores, so arbitrary directions would be pinned (critic r3, checked); and the uniform orthogonalization raises at
    rank 0 (checked).
    - Otherwise, ragged: `self.frame(x_old)`; uniform (`x_old` the full `.data`, its own masks):
      `ut3_orthogonal_representations(x_old)[0]`.
  - `x_old=None` means the zero tensor, for the cone only: the member builds the empty frame from `x0`'s structure
    (shape, stack, backend, and on the uniform layer `N`).
  - Sharing labels from `self.groups`.
  - Uniform: compaction to the level widths, read off `self.var_masks`.
  - Unset options take the `ALIGN_*` defaults of `padded_frames.py` (§5.1).
  - Returns `(frame, movements)`.
- `require_frame_structure(x_cores, frame)` raises `ValueError`, naming the mismatch and the fix, unless `frame` has
  exactly the structure the geometry's frame of `x_cores` would have. It uses integers and held masks only; it never
  orthogonalizes `x_cores` (D-37).
  - Ragged: `x`'s shape, ranks and stack `C` are read off its core shapes. `frame`'s core shapes must equal
    `C + …` of `compute_orthogonal_representation_ranks(shape, ranks)`, store by store.
  - Uniform:
    - `ufv_variation_masks(frame[5])` equals `self.var_masks`, array by array;
    - the two boundary entries, `frame[5][2][d]` and `frame[5][3][0]`, equal `prefix_mask(1, width)`;
    - `frame[4] == self.shape`;
    - each supercore's rank widths equal its masks' widths, and its `N` equals `x`'s.

    This is D-10's CHECK-AND-REJECT over all four mask families, boundary entries included.

**`_misfit_gradient_at(frame, hint)`** (module-level; `hint = (problem, residual)`):

    geom  = problem.geom.with_frame(frame)
    sweep = problem.kind.precompute(frame, problem.sample)
    return geom.project(frame, problem.kind.transpose(residual, problem.sample, frame, sweep), aux=geom.precompute(frame))

- It omits the regularizer's gradient, which is harmless: `X_old` lies in every padded frame's tangent space, and the
  step uses the normal content.
- On the uniform layer it returns the bare variation pair; the step reads `[0]`, `[1]` and masks them with the
  frame's own masks.
- **Not verified:** the uniform kinds at block-layout frames (gappy, wider than the level), and the tied projection
  (`geom.precompute` → `ufv_shared_frame_data` / `fv_shared_frame_data`) at padded frames. The prototypes used
  untied projections only. T14(e), T14(f).

**`padded_start(problem, x0, x_old, …) -> (frame, movements)`** is PUBLIC (`__all__`).

- The residual at the loop's point: `x0` when `x_old` is given, the zero tensor (`corewise_zeros_like(x0)`) when it
  is `None`.
- Then `problem.geom.aligned_padded_frame(x0, x_old, _misfit_gradient_at, (problem, residual), …)`.
- A geometry without the member raises `ValueError` naming it (corewise geometries; custom geometries).
- Every option defaults to `None` and is forwarded only when set, so the member's `ALIGN_*` / `ZERO_FRAME_SEED`
  defaults apply. The optimizer layer cannot import `padded_frames`.
- `use_jit: bool = False` is forwarded to the member and so to the loop (D-39). On the uniform layer with
  `use_jit=True`, `_misfit_gradient_at` runs inside the whole-loop jit: the `Problem` is a registered pytree, so its
  arrays are traced and its kind and geometry are static aux. **Not verified** in the prototypes; T14(i).
- `max_passes=0` (D-38) returns the hint-free start at the target. The residual is still formed, one forward, so
  there is one code path; it is never projected.
- Its doctest shows the route for a frontend user who wants a problem's aligned frame without running Newton:
  `optimizers.least_squares_problem(...)` → `.problem`, `.x0_cores` → `padded_start(problem, x0_cores,
  x_old.data)` → rewrap the frame (`T3Frame(*frame)` / `UT3Frame`).

**`Problem.local_model(x, …, frame=None)`.** With `frame`:

- The structural check is `self.geom.require_frame_structure(x, frame)` (D-37). A geometry without the member
  raises `ValueError` naming it.
- Then it linearizes at `frame` instead of `geom.frame(x)`.
- Safe-mode numerical checks (orthonormality, `frame` reconstructs `x`) are the frontend's (house rule).

**`newton_cg`.**

- `padded_from=x_old` → `start_frame = padded_start(problem, x0, x_old, padded_*, use_jit=use_jit)[0]`.
- **One flag:** `newton_cg`'s existing `use_jit` governs both the CG core and the alignment (the padded start and the
  cone), as its keyword doc states. No new keyword.
- `gradient_descent` has no `use_jit` (eager only, by design: "the reference implementation of the backend-first
  stack"), so its cone reference runs the loop eagerly.
- `start_frame=F` → the iteration-0 model is `problem.local_model(x0, frame=F, …)`; a rejected trust-region trial
  keeps it, as today.
- `'cone'` → `_resolve_g0_reference`:
  - runs `padded_start(problem, x0, None, cone_*, use_jit=use_jit)`;
  - reports `sqrt(inner(g, g))` with `g = _misfit_gradient_at(F, (problem, r0))` and `inner` from
    `problem.geom.with_frame(F)`;
  - it is not computed when `g0norm_newton` is given (as `'zero'`). `g0_reference='cone'` itself stays legal then,
    as `'zero'` does; an explicitly set `cone_*` option is a structural error (D-33);
  - an explicit `'cone'` whose value is 0 raises (as explicit `'zero'`).
- **Stats** (D-27). With no new keyword, the whole `stats` dict is bit-identical (the five keys pinned at
  `tests/backend/test_optimizers.py:477, :860`). Otherwise:
  - `stats['start_frame'] = 'given' | 'padded'`;
  - `stats['start_movements']` (padded only; a tuple of floats, one per pass run; the uniform array's NaN tail
    dropped);
  - `stats['cone_movements']` (`'cone'` only);
  - history rows carry `g0_reference='cone'` (an existing field, a new value, only when requested).
- `NewtonInfo` is unchanged. Iteration 0's `gnorm` is the start's frame-gradient norm.

**Compile accounting per level (jax inputs; all fixed, none repeated; D-39).** T13 counts it.

- `use_jit=False` (default): everything eager. jax's per-operation caches fill on the first pass, at a cost: an
  eager first step at d = 4 is about 22 s, 932 op-by-op compilations (§2.3). Module-level scan bodies compile once
  per body and shape. Nothing repeats across passes or levels of the same structure.
- `use_jit=True`, ragged: two to four jitted step programs per level (`'ULRO'` working, `'ULRO'` truncating, `'UL'`,
  `'RO'`), about 2 s each at d = 4. The same again for `'cone'` (other old ranks). The projection runs eagerly in
  the host loop.
- `use_jit=True`, uniform: ONE whole-loop program per level (about 6 s at d = 6 and 24), containing the start, every
  pass, the projections and the compaction. One more for `'cone'` (other A-block widths: `Na = 0`, `Rb = 1`).
- Either way the compacted start frame compiles nothing new for the first Newton step. Gradient-based loops are out
  of scope (D-2).

### 3.6 Frontend twins (D-8, D-9, D-35)

Each twin validates with explicit messages, makes one or two backend calls, and rewraps. It carries the description,
doctests and safe-mode checks. Its backend function keeps a one-line docstring and shape comments. Backend functions
without a twin keep their doctests: `fv_empty_frame`, `ufv_empty_frame`, `pad_safe_pinned_completion`,
`ufv_compact_frame`, `ufv_frame_widths`, the members, `padded_start` (the frontend route, §3.5).

**`padded_from=<rank-0 tensor>`** (a zero start with an aligned frame) reaches the backend as rank-0 data, not
`None`. In the backend `newton_cg`, `padded_from=None` means "off". Both members accept rank-0 data, and apply the
structurally-zero test of §3.5.

**The zero old point** is a rank-0 tensor (D-21): `TuckerTensorTrain.zeros(shape, (0,)*d, (1,)+(0,)*(d−1)+(1,),
stack_shape)`, or `UniformTuckerTensorTrain.zeros(shape, 0, (1,)+(0,)*(d−1)+(1,), stack_shape)`.

**One rule, applied first.** Every twin and the frontend `newton_cg(padded_from=)` apply the §3.5 test: an `x_old`
with any Tucker rank or interior TT rank 0 is STRUCTURALLY ZERO (e.g. Tucker `(0,2,2)`, TT `(1,0,2,1)`). For such an
`x_old`:

- the reduction to minimal ranks is skipped, because `compute_minimal_ranks` maps the last boundary rank to 0 at
  rank 0 (checked);
- the frame twins use the empty frame directly (built from `x_old`'s shape, stack, backend, and on the uniform layer
  `N`), because `ut3_orthogonal_representations` raises at rank 0 (checked);
- `newton_cg` passes the structurally-zero `x_old` through AS GIVEN, not converted (D-43, as built in N8). The
  backend members apply the same test and build the empty frame themselves (§3.5).

**Validation and conversion** (`T3Frame.padded`, `UT3Frame.padded`, the two loop twins, `optimizers.newton_cg`):

- `x_old` (or a start frame) of `x0`'s layer, with the same `shape`, stack and padded `N` (uniform). A structural
  error names the conversion.
- `x_old`'s stores ≤ the target's per edge (structural; the backend repeats it).
- `x_old` is reduced to minimal ranks first, sharing-aware: the optimizers' `_ragged_minimal` and
  `uniform_fitting.uniform_minimal`. The frame twins call the same three methods (`has_minimal_ranks`, `t3svd`,
  `rank_adjustment_sweep`).
- A start frame's structure is checked by the backend member `require_frame_structure` (structural, both modes).
- Safe mode: `x0.allclose(x_old)` (the zero padding), tied factors under `sharing` (TIED), an orthonormal start frame
  that reconstructs `x0`.
- `use_jit=True` converts `start_frame` / `padded_from` together with `x0` (`_prepare_jit_inputs`).
- `start_frame`, `padded_from` or `'cone'` on a corewise geometry is a structural error.
- `start_frame` together with `padded_from` is a structural error.
- An option of the `padded_*` group set without `padded_from`, or of the `cone_*` group set without
  `g0_reference='cone'` or with `g0norm_newton` given (the cone is then not computed), is a structural error.
- `x0`'s stores equal to `x_old`'s: there is nothing to complete. The backend loops return an empty tuple of
  movements without a pass (the ragged one the old frame, the uniform one its compaction to the level widths). This
  is documented on the twins.

**The loop twins' `project` and hint.**

- `project=None` picks the projection from the hint's class:
  - ragged twin: `TuckerTensorTrain` → `tv_project_t3_onto_tangent_space`; `numpy`/jax array →
    `tv_project_dense_onto_tangent_space`;
  - uniform twin: `UniformTuckerTensorTrain` → `utv_project_ut3_onto_tangent_space`. The hint must have `x_old`'s
    padded `N`, else a structural error worded as `UniformTuckerTensorTrain.__add__`'s, naming `from_t3(..., N=)`.
- A hint of the other layer is a structural error naming the conversion; any other hint without `project` is a
  structural error.
- A user `project` receives raw data: the frame's `.data` (gappy block-layout frames on the uniform layer). It
  returns raw variation data, `(V, H)` / `(V, H, …)`.
- The default projections are untied, with or without `sharing`. The step ties through its group update.

### 3.7 Import DAG (new edges; acyclic; T18)

    padded_frames          -> common, contractions, linalg, ranks, sharing, fv_operations
    uniform_padded_frames  -> common, contractions, linalg, ranks, sharing, ufv_masking, fv_operations, padded_frames
    geometry               -> padded_frames, uniform_padded_frames, ranks   (ufv_conversions, ufv_masking already imported)
    optimizers             -> (nothing new: T3-free)
    frontend: frame_variations_format -> backend.padded_frames, backend.tv_operations
              uniform_frame_variations_format -> backend.uniform_padded_frames, backend.padded_frames (the ALIGN_*
                                                 defaults), backend.utv_operations, backend.ufv_masking
              manifold -> backend.padded_frames;  uniform_manifold -> backend.uniform_padded_frames

No cycle: `fv_operations`, `ranks`, `sharing`, `linalg` and `ufv_masking` import neither new module, and no frontend
module gains an import of a sibling frontend module. The graph is defined on module files, not packages (T18).

## 4. Algorithms, step by step (library form)

### 4.1 Ragged step `tv_align_padded_frame`

**Inputs.** The frame `(U, O, L, R)` (stack `C`), the variations `(V, H)` at it, the old ranks, the output ranks,
`sharing` and `update`.

**Host structure:**

- `s°` = stores(shape, old ranks); `s¹` = stores(shape, output ranks); `s` = the frame's own stores, read off its
  shapes.
- Structural checks:
  - `s° ≤ s¹ ≤ s` per edge, with boundary bonds 1 in all three;
  - `update` a non-empty set of `{U, L, R, O}` without repeats;
  - no store outside `update` with `s¹ < s` (transport plus truncation);
  - equal `s°` / `s¹` Tucker stores within each group.
- `k = s − s°`, `k¹ = s¹ − s°`.

**Common shapes:**

- Per-core blocks are sliced at the old widths while the scan inputs are built. On the ragged layer `xscan` is a host
  loop, so the per-core widths may also ride the inputs as Python ints.
- Reverse sweeps run on reversed input tuples, with the outputs reversed back.
- Bodies are module-level.
- `root(M)` is a factor `Xi` with `Xi Xiᵀ = Mᵀ M`, from the thin SVD: `Xi = Vᵀ_hᵀ · diag(s)`.
- `unf_out`, `unf_in`, `unf_mid` unfold a core with the last / first / middle axis as columns (core axes only).

The `contract` strings are Appendix C's, by label.

1. **Down messages** (`xmap`, body `_down_message`).
   - `VP_i = V_i[nD°_i:, :]`
   - `PhiO_i = contract(c1, VP_i, U_i)`
   - `XiO_i = root(VP_iᵀ − contract(c2, U_i, PhiO_i))`
2. **Left messages** (`xscan`, body `_left_message_step`).
   - Runs over `j = d−1 .. 1` (reversed inputs).
   - Carry `(PhiL, XiL)` of the edge `c_j → c_{j+1}`, initially `C+(0, 1)`, `C+(0, 0)`.
   - Inputs per `j`: `H_P = H_j[rL°_j:, :, :]`, `βO = O_j[rL°_j:, nD°_j:, :]`, `βL = L_j[rL°_j:, :, rL°_{j+1}:]`,
     `R_j`, `PhiO_j`, `XiO_j`.
   - Body:
     - `Vb = H_P + contract(c3, βO, PhiO) + contract(c4, βL, PhiL)`
     - `rows = [unf_in(contract(c5, βO, XiO)), unf_in(contract(c6, βL, XiL))]`
     - `PhiL' = contract(c7, Vb, R_j)`
     - `XiL' = root(concat_rows([unf_in(Vb − contract(c8, PhiL', R_j))] + rows))`
3. **Right messages** (`xscan`, body `_right_message_step`).
   - Runs over `j = 0 .. d−2`.
   - Carry `(PhiR, XiR)` of the edge `c_j → c_{j−1}`, initially `C+(0, 1)`, `C+(0, 0)`.
   - Inputs: `H_j[:, :, rR°_{j+1}:]`, `βO = O_j[:, nD°_j:, rR°_{j+1}:]`, `βR = R_j[rR°_j:, :, rR°_{j+1}:]`, `L_j`,
     `PhiO_j`, `XiO_j`.
   - Body:
     - `Vb = H_P + contract(c9, βO, PhiO) + contract(c10, βR, PhiR)`
     - `rows`: `unf_out(contract(c11, …))`, `unf_out(contract(c12, …))`
     - `PhiR' = contract(c13, Vb, L_j)`
     - `XiR' = root([unf_out(Vb − contract(c14, PhiR', L_j))] + rows)`
4. **Up messages** (`xmap`).
   - Inputs per mode: `H_i[:, nU°_i:, :]`, `βL = L_i[:, nU°_i:, rL°_{i+1}:]`, `βR = R_i[rR°_i:, nU°_i:, :]`, `O_i`, and
     the messages `PhiL_i, XiL_i` (edge `c_i → c_{i+1}`) and `PhiR_i, XiR_i` (edge `c_i → c_{i−1}`).
   - `Vb = H_P + contract(c15, βL, PhiL_i) + contract(c16, βR, PhiR_i)`
   - `rows`: `unf_mid(contract(c17, βL, XiL_i))`, `unf_mid(contract(c18, βR, XiR_i))`
   - `PhiU_i = contract(c19, Vb, O_i)`
   - `XiU_i = root([unf_mid(Vb − contract(c20, PhiU_i, O_i))] + rows)`
5. **Up edges** (`xmap` over groups; host member lists from `canonical_groups`).
   - From the group's first mode `f`: `Ā = U_f[:nU°, :]ᵀ`, `g_old = U_f[nU°:, :]ᵀ`.
   - For each member `i`: `g_new_i = V_i[nD°_i:, :]ᵀ` and `Φ_P,i = PhiU_i[:, nD°_i:]`.
   - Content, the members side by side: `Y = [contract(c21, g_old, Φ_P,i) + g_new_i | contract(c21, g_old, XiU_i)]`.
     When `'U' ∉ update`, `Y = g_old`.
   - `P⁺ = pinned_leading_columns(Ā, Y, k¹_U)`.
   - `U⁺ = [U°_f ; P⁺ᵀ]`, one array for every member.
   - `o_U = contract(c22, P⁺, g_old)`.
   - `DU_i = contract(c24, U⁺, g_new_i)`.
   - `movement = ‖g_old − contract(c23, P⁺, o_U)‖_F`.
6. **Left edges** (`xscan`, body `_left_edge_step`).
   - Runs over `i = 0 .. d−2`.
   - Carry `(o_L, D_L)` of the edge `c_{i−1} → c_i`, initially `C+(0, 0)`, `C+(1, 0)`.
   - `NEW_L(X)` keeps `X[:rL°_i]` and maps the rest by `contract(c8, o_L, X[rL°_i:])`. `NEW_T(X)` is the same on the
     Tucker axis with `o_U,i` (`c25`).
   - `M = NEW_T(NEW_L(H_i[:, :, rR°_{i+1}:])) + NEW_L(contract(c26, O_i[:, nD°_i:, rR°_{i+1}:], DU_i)) + NEW_T(contract(c27, R_i[rR°_i:, :, rR°_{i+1}:], D_L))`
   - `g_new = unf_out(M)`; `g_old = unf_out(NEW_T(NEW_L(L_i[:, :, rL°_{i+1}:])))`.
   - `Ā = unf_out(pad_or_truncate(L_i[:rL°_i, :nU°_i, :rL°_{i+1}] → (rL¹_i, nU¹_i, rL°_{i+1})))`.
   - `Y = [contract(c21, g_old, PhiL_i[:, rR°_{i+1}:]) + g_new | contract(c21, g_old, XiL_i)]`, or `g_old` when
     `'L' ∉ update`.
   - `P⁺ = pinned_leading_columns(Ā, Y, k¹_{L,i+1})`; `L⁺_i = fold([Ā | P⁺])`.
   - New carry: `o_L' = contract(c22, P⁺, g_old)`, `D_L' = contract(c22, [Ā|P⁺], g_new)`.
7. **Right edges** (`xscan` over `i = d−1 .. 1`, reversed inputs; carry `(o_R, D_R)`, initially `C+(0, 0)`,
   `C+(1, 0)`).
   - `M = NEW_R(NEW_T(H_i[rL°_i:, :, :])) + NEW_R(contract(c28, O_i[rL°_i:, nD°_i:, :], DU_i)) + NEW_T(contract(c29, L_i[rL°_i:, :, rL°_{i+1}:], D_R))`,
     with `NEW_R` on the last axis (`c14`).
   - `g_new = unf_in(M)`; `g_old = unf_in(NEW_R(NEW_T(R_i[rR°_i:, :, :])))`.
   - `Ā = unf_in(pad(R_i[:rR°_i, :nU°_i, :rR°_{i+1}] → (rR°_i, nU¹_i, rR¹_{i+1})))`.
   - Content from `PhiR_i[:, rL°_i:]` and `XiR_i`; `P⁺` with `k¹_{R,i}`; `R⁺_i` = fold.
8. **Down edges** (`xmap`), with `o_L,i`, `D_L,i` (edge `c_{i−1} → c_i`) and `o_R,i+1`, `D_R,i+1`
   (edge `c_{i+1} → c_i`).
   - `M = NEW_R(NEW_L(H_i[:, nU°_i:, :])) + NEW_L(contract(c30, L_i[:, nU°_i:, rL°_{i+1}:], D_R,i+1)) + NEW_R(contract(c31, R_i[rR°_i:, nU°_i:, :], D_L,i))`
   - `g_new = unf_mid(M)`; `g_old = unf_mid(NEW_R(NEW_L(O_i[:, nD°_i:, :])))`.
   - `Ā = unf_mid(pad(O_i[:rL°_i, :nD°_i, :rR°_{i+1}] → (rL¹_i, nD°_i, rR¹_{i+1})))`.
   - Content from `PhiO_i[:, nU°_i:]` and `XiO_i`; `P⁺` with `k¹_{D,i}`; `O⁺_i` = fold.
9. **Carriers.** `L⁺_{d−1} = pad(L_{d−1}[:rL°_{d−1}, :nU°_{d−1}, :])` and `R⁺_0 = pad(R_0[:, :nU°_0, :rR°_1])`, to the
   output widths.
10. **Movement** = the maximum over all edges, shape `C`.

### 4.2 Ragged start `fv_padded_frame`

The integer signature, with the D-41 seed convention (N8b; it replaces S2's coordinate convention):

- Every store, the same rule: seeds = rows `w .. w+k−1` of the keyed stream `(seed, store kind, the group's first
  mode)`, each row of the ragged unfolding's length; then `pinned_completion(old block, seedsᵀ, k)`.
  - Tucker, per group: old block `U°ᵀ`, row length `N_g`; the result is replicated over the group's members, so
    they stay tied.
  - TT and down cores, per mode: the old core's unfolding columns, zero outside the A-part.
- Carriers: the old ones, zero-padded.

The seed rows are host bookkeeping: a private module-level helper, shared with the uniform start (D-46).
`_coordinate_seed_order` and the outside/inside branch are deleted.

### 4.3 Ragged loop `fv_aligned_padded_frame`

§2.3 verbatim, as a host loop.

1. `working = ranks.oversampled_ranks(shape, target, p, sharing)`. The oversampled phase is skipped when the working
   and target STORES match (§2.3); then the start is built at the target.
2. The start at the working ranks.
3. Every pass: one call of `project`, then one step. Old ranks come from the old frame's `(up, left)` widths. With
   `use_jit=True` and jax inputs, the step is the memoized jitted step (static: old ranks, output ranks, `sharing`,
   `update`); the host loop and `project` stay eager (D-39).
4. Returns the movements as a tuple (each `C`).

### 4.4 Uniform step `utv_align_padded_frame`

`P/uniform_form2.uniform_align2` in the §3.4 layout, steps 1–10 as in §4.1, with these differences:

- **Masks once at entry.** The frame by its masks and the shape prefix mask; `V` by `(mD, mN)`; `H` by
  `(mL[:-1], mU, mR[1:])`.
- **Slicing.** The P-parts are static slices at the split.
- **Maps.** One `'dC…'` contraction each (Appendix C, d-twins).
- **Sweeps.** `xscan` with module-level bodies over the (reversed for reverse sweeps) supercores.
  - Message carries: `C+(Kb, rT)` and `C+(Kb, Kb)`, zero at the ends.
  - Edge carries: `C+(Kb1, Kb)` and `C+(rT1, Kb)`.
  - Per-mode mask slices and one-hots ride the scan inputs as data (`pad_safe_svd` accepts traced masks).
  - The split comes from carry shapes (`Kb = o.shape[-1]`, `Rb = rT − Kb`).
- **Factorization.** `pad_safe_svd` on `[Ā | guard(Y⊥)]`, then the one-hot selection (`contract(c21, U_s, sel)`).
  Transported stores use `Y⊥ = (I − ĀĀᵀ) g_old`.
- **`NEW`** is a block operation at the split.
- **Groups** via host tables.
- **Carriers.** The A-blocks of `L_{d−1}` and `R_0`, masked and zero-padded to the output widths.
- **Output masks.** The A-block masks as given, followed by `prefix_mask(k¹, Ka1 | Kb1)`.
- **Kernel.** All of the above runs inside `_utv_align_kernel`: jittable, and jitted only under `use_jit` (D-34,
  D-39).

### 4.5 Uniform start `ufv_padded_frame`, compaction `ufv_compact_frame`, empty frame `ufv_empty_frame`

**The start:** `P/uniform_start.uniform_start(method='gs')`.

- `_old_block_layout` pads the old frame to `(Na, Rb)` and masks it.
- The seeds as in §3.4 (D-41): the ragged stream rows, built on the host per group / mode and per stack element,
  scattered.
- Tucker: `pad_safe_pinned_completion(U°_firstᵀ, A-mask, scattered seeds, prefix(kU_first), shape mask)` batched
  over `(G,)+C`, scattered by `group_of`.
- TT / down: `pad_safe_pinned_completion(the A-block's unfolding columns, A-masks, scattered seeds, prefix(k),
  the unfolding's real-row mask)` batched over the modes and `C`. The outside unit-vector placement and the inside
  sub-grid completion (the prototype's `offset=t`) are deleted.

**The compaction:** `P/uniform_loop.compact`. Structural: every real count ≤ its level width. When a level width
exceeds the input frame's padded width, the axis is zero-padded (junk, mask `False`) before the gather (D-44).

**The empty frame:**

| part | shape or value |
|---|---|
| `U` | `(d,)+C+(0, N)` |
| `O`, `L`, `R` | `(d,)+C+(1, 0, 1)` |
| up and down masks | `(d,)+C+(0,)` |
| left and right masks | `(d+1,)+C+(1,)`, `True` at bonds `0` and `d` only |
| `N` | `max(shape)` by default |

### 4.6 Uniform loop `ufv_aligned_padded_frame`

§2.3 as a host loop (D-20).

1. The working ranks per stack element (`ranks.oversampled_ranks` on `(d,)+C` arrays). The oversampled phase is
   skipped when the working and target STORES match for every element: the §2.3 rule, the same as the ragged loop.
2. The start at the working counts.
3. Every pass: `project(frame_data, hint)` at the block-layout frame, then `utv_align_padded_frame`.
4. `ufv_compact_frame(·, level_widths)` at exit. When there is nothing to complete (target stores = old stores), the
   loop skips steps 2–3 and returns `ufv_compact_frame(old_frame_data, level_widths)`.
5. Returns `(frame_data, movements)`, with movements a `(max_passes,)+C` array, NaN after the last pass.
6. With `use_jit=True`, steps 2–4 are the one jitted `_ufv_aligned_loop_kernel` of §3.4 (D-39). With
   `use_jit=False` they are a host loop calling the kernel eagerly. Both follow the same pair schedule (§2.3), so
   their pass counts agree.

### 4.7 The geometry members and the drivers

As §3.5. The ragged member's old frame is `self.frame(x_old)`, i.e. `t3_orthogonal_representations`. The uniform
member's is `ufv_conversions.ut3_orthogonal_representations(x_old)[0]`.

## 5. API (signatures in the house style; the twins' docstrings carry the descriptions)

### 5.1 `backend/padded_frames.py`

```python
FrameData = typ.Tuple[typ.Sequence[NDArray], typ.Sequence[NDArray], typ.Sequence[NDArray], typ.Sequence[NDArray]]
#   (up, down, left, right), each len=d, elm_shapes C+(nUi, Ni), C+(rLi, nDi, rR(i+1)), C+(rLi, nUi, rL(i+1)), C+(rRi, nUi, rR(i+1))
VariationsData = typ.Tuple[typ.Sequence[NDArray], typ.Sequence[NDArray]]
#   (tucker_variations, tt_variations), each len=d, elm_shapes C+(nDi, Ni), C+(rLi, nUi, rR(i+1))

ALIGN_OVERSAMPLING = 5          # the loops' defaults (D-6), named once; the members and the frontend twins reference them
ALIGN_OVERSAMPLED_PASSES = 2
ALIGN_MAX_PASSES = 10
ALIGN_TOL = 0.05
ALIGN_ALTERNATE = True

def fv_empty_frame(
        shape:        typ.Sequence[int],        # len=d, the mode sizes (N0, ..., N(d-1))
        stack_shape:  typ.Sequence[int] = (),   # C
        use_jax:      bool = False,             # array-free constructor: the flag chooses the output type
) -> typ.Tuple[                                 # the frame of the zero tensor (default float), b = (1, 0, ..., 0, 1):
    typ.Tuple[NDArray, ...],    # up.    len=d, elm_shape=C+(0, Ni)
    typ.Tuple[NDArray, ...],    # down.  len=d, elm_shape=C+(bi, 0, b(i+1))
    typ.Tuple[NDArray, ...],    # left.  len=d, elm_shape=C+(bi, 0, b(i+1))
    typ.Tuple[NDArray, ...],    # right. len=d, elm_shape=C+(bi, 0, b(i+1))
]:

def fv_padded_frame(
        old_frame:     FrameData,                          # the old frame, stack C (fv_empty_frame for the zero tensor)
        tucker_ranks:  typ.Sequence[int],                  # len=d, target Tucker ranks (stores >= the old frame's)
        tt_ranks:      typ.Sequence[int],                  # len=d+1, target TT ranks (boundary 1)

        sharing:       typ.Optional[typ.Sequence] = None,  # len=d group labels; None = unshared
        seed:          int = ZERO_FRAME_SEED,              # keys every store's slot-indexed Gaussian seed stream (D-41)
) -> typ.Tuple[                 # (nU, nD, rL, rR) = compute_orthogonal_representation_ranks(shape, tucker_ranks, tt_ranks)
    typ.Tuple[NDArray, ...],    # up.    len=d, elm_shape=C+(nUi, Ni): the old rows bitwise, then the seeded completion (D-41)
    typ.Tuple[NDArray, ...],    # down.  len=d, elm_shape=C+(rLi, nDi, rR(i+1)): the old block bitwise, then seeded
    typ.Tuple[NDArray, ...],    # left.  len=d, elm_shape=C+(rLi, nUi, rL(i+1)); left[d-1] = the old carrier, zero-padded
    typ.Tuple[NDArray, ...],    # right. len=d, elm_shape=C+(rRi, nUi, rR(i+1)); right[0] = the old carrier, zero-padded
]:

def tv_align_padded_frame(
        frame:             FrameData,                                # a padded frame: old blocks in the leading slots, stack C
        variations:        VariationsData,                           # a tangent vector at `frame` (gauged or not), stack C
        old_tucker_ranks:  typ.Sequence[int],                        # len=d, the old blocks' extents; the zero tensor: (0,)*d
        old_tt_ranks:      typ.Sequence[int],                        # len=d+1; the zero tensor: (1, 0, ..., 0, 1)

        tucker_ranks:      typ.Optional[typ.Sequence[int]] = None,   # len=d, output ranks; None = the frame's own
        tt_ranks:          typ.Optional[typ.Sequence[int]] = None,   # len=d+1, output ranks; None = the frame's own

        sharing:           typ.Optional[typ.Sequence] = None,        # len=d group labels; None = unshared
        update:            str = 'ULRO',                             # stores updated: 'U' up, 'L' left, 'R' right, 'O' down; the rest transported
) -> typ.Tuple[
    FrameData,   # the frame at the output ranks: old blocks bitwise, completions aligned
    NDArray,     # movement, shape=C: max over directed edges of ‖g_old − P⁺P⁺ᵀ g_old‖_F
]:

def fv_aligned_padded_frame(
        old_frame:           FrameData,                          # the old frame, stack C (fv_empty_frame for the zero tensor)
        tucker_ranks:        typ.Sequence[int],                  # len=d, target
        tt_ranks:            typ.Sequence[int],                  # len=d+1, target (boundary 1)
        hint:                typ.Any,                            # opaque: handed to `project` unread
        project:             typ.Callable,                       # project(frame, hint) -> VariationsData: ONE fixed tensor projected at `frame`

        sharing:             typ.Optional[typ.Sequence] = None,  # len=d group labels; None = unshared
        oversampling:        int   = ALIGN_OVERSAMPLING,         # p: working ranks = ranks.oversampled_ranks(shape, target, p, sharing)
        oversampled_passes:  int   = ALIGN_OVERSAMPLED_PASSES,   # passes at the working ranks before the truncating pass
        max_passes:          int   = ALIGN_MAX_PASSES,           # passes in all, the truncating pass included; 0 = the hint-free start at the target (D-38)
        tol:                 float = ALIGN_TOL,                  # early exit on the movement
        alternate:           bool  = ALIGN_ALTERNATE,            # two-colour passes after the truncating pass
        seed:                int   = ZERO_FRAME_SEED,            # the start's seed streams (D-41)
        use_jit:             bool  = False,                      # jax inputs: the host loop calls a memoized jitted step (D-39); else eager
) -> typ.Tuple[
    FrameData,                 # the aligned padded frame at the target ranks
    typ.Tuple[NDArray, ...],   # movements, one per pass, each shape=C
]:
```

### 5.2 `backend/linalg.py`

```python
def pad_safe_pinned_completion(
        pinned:       NDArray,   # C+(n, s): orthonormal on its real columns; junk entries anything (masked at entry)
        pinned_mask:  NDArray,   # HOST bool, static, broadcastable to C+(s,): True = a real pinned column
        seeds:        NDArray,   # C+(n, m): seed columns, completed in column order; junk entries anything
        seed_mask:    NDArray,   # HOST bool, static, broadcastable to C+(m,): True = complete this seed (per-element counts)
        row_mask:     NDArray,   # HOST bool, static, broadcastable to C+(n,): True = a real row
) -> NDArray:                    # C+(n, m): column j = seed j completed against the real pins and the earlier selected seeds
                                 #   (the uniform start's one factorization, every store, D-41)
                                 #   (Gram–Schmidt order, the pinned_completion sign rule) where seed_mask, else 0; bitwise 0 on
                                 #   junk rows. Per element (structural): sum(seed_mask) <= sum(row_mask) - sum(pinned_mask)
```

### 5.3 `backend/uniform_padded_frames.py` and `backend/ufv_masking.py`

```python
UFrameData = typ.Tuple[NDArray, NDArray, NDArray, NDArray, typ.Tuple[int, ...], typ.Tuple[NDArray, NDArray, NDArray, NDArray]]
#   up (d,)+C+(nU, N), down (d,)+C+(rL, nD, rR), left (d,)+C+(rL, nU, rL), right (d,)+C+(rR, nU, rR), shape (len=d),
#   masks HOST bool (up (d,)+C+(nU,), down (d,)+C+(nD,), left (d+1,)+C+(rL,), right (d+1,)+C+(rR,)); block layout: nU = nD, rL = rR
UVariationsData = typ.Tuple[NDArray, NDArray, typ.Tuple[int, ...], typ.Tuple[NDArray, NDArray, NDArray, NDArray]]
#   V (d,)+C+(nD, N), H (d,)+C+(rL, nU, rR), shape, variation masks; only V, H are read (masked with the frame's own masks)

def ufv_empty_frame(
        shape:        typ.Sequence[int],        # len=d, the real mode sizes
        N:            typ.Optional[int] = None, # the padded mode size; None = max(shape)
        stack_shape:  typ.Sequence[int] = (),   # C
        use_jax:      bool = False,             # array-free constructor: chooses the supercore type (masks are always host)
) -> typ.Tuple[                                 # the frame of the zero tensor (Na = 0, Rb = 1; default float):
    NDArray,               # up.    shape=(d,)+C+(0, N)
    NDArray,               # down.  shape=(d,)+C+(1, 0, 1)
    NDArray,               # left.  shape=(d,)+C+(1, 0, 1)
    NDArray,               # right. shape=(d,)+C+(1, 0, 1)
    typ.Tuple[int, ...],   # shape, len=d
    typ.Tuple[NDArray, NDArray, NDArray, NDArray],   # masks, HOST bool: up/down (d,)+C+(0,); left/right (d+1,)+C+(1,), True at bonds 0 and d only
]:

def ufv_padded_frame(
        old_frame_data:  UFrameData,                                  # the old frame at its own widths, any mask pattern, stack C
        tucker_ranks:    typ.Union[typ.Sequence[int], NDArray],       # len=d, or HOST int (d,)+C (per stack element)
        tt_ranks:        typ.Union[typ.Sequence[int], NDArray],       # len=d+1, or HOST int (d+1,)+C

        sharing:         typ.Optional[typ.Sequence] = None,           # len=d group labels; None = unshared
        seed:            int = ZERO_FRAME_SEED,                       # the start's seed streams (D-41), as the ragged start
) -> typ.Tuple[
    UFrameData,            # block layout: [A-block | P-block] per slot, old frame as given (junk zeroed), completions real-first
                           #   (D-41 seeds from the host at the ragged size, scattered; pad-safe pinned completion per store)
    typ.Tuple[int, int],   # split (Na, Rb): the static A-block widths
]:

def utv_align_padded_frame(
        frame_data:       UFrameData,                                        # block layout, any mask pattern in either block
        variations_data:  UVariationsData,                                   # a tangent vector at `frame_data`
        split:            typ.Tuple[int, int],                               # (Na, Rb), static

        tucker_ranks:     typ.Optional[typ.Union[typ.Sequence[int], NDArray]] = None,   # len=d or HOST int (d,)+C; None = the frame's own
        tt_ranks:         typ.Optional[typ.Union[typ.Sequence[int], NDArray]] = None,   # len=d+1 or HOST int (d+1,)+C; None = the frame's own

        sharing:          typ.Optional[typ.Sequence] = None,                 # len=d group labels; None = unshared
        update:           str = 'ULRO',                                      # static; as tv_align_padded_frame
) -> typ.Tuple[
    UFrameData,   # P-blocks at the output widths (the counts' maxima), real-first; A-blocks as given
    NDArray,      # movement, shape=C
]:

def ufv_compact_frame(
        frame_data:    UFrameData,                     # any layout and mask pattern
        level_widths:  typ.Tuple[int, int, int, int],  # (WU, WD, WL, WR), each >= every real count of its store; wider than the input: padded first
) -> UFrameData:                                       # junk-last, junk zeroed; masks = ufv_make_frame_masks of the counts at level_widths

def ufv_aligned_padded_frame(
        old_frame_data:      UFrameData,                              # the old frame, stack C (ufv_empty_frame for the zero tensor)
        tucker_ranks:        typ.Union[typ.Sequence[int], NDArray],   # len=d or HOST int (d,)+C, target
        tt_ranks:            typ.Union[typ.Sequence[int], NDArray],   # len=d+1 or HOST int (d+1,)+C, target
        hint:                typ.Any,                                 # opaque: handed to `project` unread
        project:             typ.Callable,                            # project(frame_data, hint) -> UVariationsData (or the bare (V, H) pair); traceable and module-level under use_jit
        level_widths:        typ.Tuple[int, int, int, int],           # (WU, WD, WL, WR): the compaction widths (the level's frame)

        sharing:             typ.Optional[typ.Sequence] = None,       # len=d group labels; None = unshared
        oversampling:        int   = ALIGN_OVERSAMPLING,              # as fv_aligned_padded_frame (constants from padded_frames)
        oversampled_passes:  int   = ALIGN_OVERSAMPLED_PASSES,
        max_passes:          int   = ALIGN_MAX_PASSES,                # static under use_jit (the movements' length)
        tol:                 float = ALIGN_TOL,                       # traced under use_jit
        alternate:           bool  = ALIGN_ALTERNATE,
        seed:                int   = ZERO_FRAME_SEED,
        use_jit:             bool  = False,                           # the WHOLE loop as one jax.jit (D-39); else a host loop, eager kernel
) -> typ.Tuple[
    UFrameData,   # the aligned frame compacted to level_widths, prefix masks
    NDArray,      # movements, shape=(max_passes,)+C, NaN after the last pass (both modes)
]:

# backend/ufv_masking.py
def ufv_frame_widths(
        N:  int,   # the point's padded mode size
        n:  int,   # its padded Tucker width
        r:  int,   # its padded TT width
) -> typ.Tuple[
    int,   # WU = min(n, N)
    int,   # WD = min(n, N, r*r)
    int,   # WL = r
    int,   # WR = r
]:
```

### 5.4 `backend/ranks.py` (changed)

```python
def oversampled_ranks(
        shape:         typ.Sequence[int],                               # len=d
        tucker_ranks:  typ.Union[typ.Sequence[int], NDArray],           # len=d, or HOST int (d,)+C
        tt_ranks:      typ.Union[typ.Sequence[int], NDArray],           # len=d+1, or HOST int (d+1,)+C
        p:             int,                                             # the rank increment, p >= 0 (structural)
        sharing:       typ.Optional[typ.Sequence] = None,               # len=d group labels; None = unshared
) -> typ.Tuple[
    typ.Union[typ.Tuple[int, ...], NDArray],   # tucker ranks: int tuple for sequences, HOST int (d,)+C for arrays
    typ.Union[typ.Tuple[int, ...], NDArray],   # tt ranks:     int tuple for sequences, HOST int (d+1,)+C for arrays
]:
```

### 5.5 `backend/geometry.py` (members on `ManifoldGeometryOps` and `UniformManifoldGeometryOps`)

```python
    def with_frame(
            self,
            frame:  typ.Tuple,    # a frame of this layer at any structure (uniform: frame .data, any widths and masks)
    ) -> 'ManifoldGeometryOps':   # ragged: self; uniform: from_frame(frame, (frame[5][0], frame[5][2])) with self.groups

    def aligned_padded_frame(
            self,
            x0:                  typ.Tuple,                 # the point at the target ranks (ragged cores; uniform bare pair at self's masks)
            x_old:               typ.Optional[typ.Tuple],   # the old point at minimal ranks (ragged cores; uniform full .data); None = the zero tensor
            project:             typ.Callable,              # project(frame, hint) -> variations
            hint:                typ.Any,                   # opaque

            oversampling:        int   = ALIGN_OVERSAMPLING,          # padded_frames' named defaults
            oversampled_passes:  int   = ALIGN_OVERSAMPLED_PASSES,
            max_passes:          int   = ALIGN_MAX_PASSES,
            tol:                 float = ALIGN_TOL,
            alternate:           bool  = ALIGN_ALTERNATE,
            seed:                int   = ZERO_FRAME_SEED,
            use_jit:             bool  = False,                       # passed to the loop (D-39)
    ) -> typ.Tuple[
        typ.Tuple,   # the aligned frame of x0 (uniform: at the widths of self.var_masks, prefix masks)
        typ.Any,     # movements: ragged a tuple (each C); uniform (max_passes,)+C, NaN after the last pass
    ]:

    def require_frame_structure(
            self,
            x_cores:  typ.Tuple,   # the point (ragged cores; uniform bare pair at self's masks)
            frame:    typ.Tuple,   # a candidate frame of x_cores (ragged frame; uniform frame .data)
    ) -> None:                     # raises ValueError unless `frame` has the structure of this geometry's frame at x_cores
                                   #   (integers and held masks only; never orthogonalizes; §3.5)
```

### 5.6 `backend/optimizers.py`

```python
class Problem:
    def local_model(
            self,
            x_cores:            Cores,                       # the point to linearize at
            sample:             typ.Any = None,
            data:               typ.Any = None,
            hessian:            str = 'gauss_newton',
            curvature_damping:  typ.Optional[float] = None,
            frame:              typ.Any = None,              # NEW: a frame of x_cores to linearize at; checked by geom.require_frame_structure
    ) -> LocalModel:

def _misfit_gradient_at(
        frame:  typ.Any,      # a frame of the problem's geometry, any structure (ragged frame / uniform frame .data)
        hint:   typ.Tuple,    # (problem, residual): the problem, and its residual at the loop's point (computed once)
) -> typ.Any:                 # Π 𝒥ᵀ residual at `frame` (ragged variations / uniform bare pair); no regularizer term

def padded_start(                                       # PUBLIC (__all__); doctest: the least_squares_problem route (§3.5)
        problem:             'Problem',
        x0:                  Cores,                     # the zero-padded point at the target ranks (minimal); uniform: bare pair
        x_old:               typ.Optional[typ.Any],     # the old point at minimal ranks (ragged cores / uniform full .data, rank 0 allowed); None = the zero tensor (the cone)

        oversampling:        typ.Optional[int]   = None,   # None = the member's default (ALIGN_OVERSAMPLING = 5); forwarded only when set
        oversampled_passes:  typ.Optional[int]   = None,   # None = 2
        max_passes:          typ.Optional[int]   = None,   # None = 10
        tol:                 typ.Optional[float] = None,   # None = 0.05
        alternate:           typ.Optional[bool]  = None,   # None = True
        seed:                typ.Optional[int]   = None,   # None = ZERO_FRAME_SEED
        use_jit:             bool = False,                 # forwarded to the loop (D-39); newton_cg passes its own use_jit
) -> typ.Tuple[
    typ.Any,   # the aligned frame of x0 (ragged frame / uniform frame .data at the level's widths)
    typ.Any,   # movements, as the member returns them (ragged tuple; uniform NaN-padded array)
]:

def newton_cg(
        problem, x0, ..., use_jit=False, callback=None,        # unchanged; use_jit now also governs the alignment (D-39)
        *,
        start_frame:                typ.Any = None,             # a frame of x0 for iteration 0 (structurally checked)
        padded_from:                typ.Any = None,             # the old point: start_frame = padded_start(problem, x0, padded_from, padded_*)

        padded_oversampling:        typ.Optional[int]   = None, # None = 5    (padded_from only, else a structural error)
        padded_oversampled_passes:  typ.Optional[int]   = None, # None = 2
        padded_max_passes:          typ.Optional[int]   = None, # None = 10
        padded_tol:                 typ.Optional[float] = None, # None = 0.05
        padded_alternate:           typ.Optional[bool]  = None, # None = True
        padded_seed:                typ.Optional[int]   = None, # None = ZERO_FRAME_SEED

        cone_oversampling:          typ.Optional[int]   = None, # None = 5    (g0_reference='cone' without g0norm_newton only, else a structural error)
        cone_oversampled_passes:    typ.Optional[int]   = None, # None = 2
        cone_max_passes:            typ.Optional[int]   = None, # None = 10
        cone_tol:                   typ.Optional[float] = None, # None = 0.05
        cone_alternate:             typ.Optional[bool]  = None, # None = True
        cone_seed:                  typ.Optional[int]   = None, # None = ZERO_FRAME_SEED
) -> typ.Tuple[Tangent, dict]:
# g0_reference: None | 'zero' | 'warm_start' | 'cone' (NEW). gradient_descent gains the six cone_* keywords, keyword-only, appended.
```

### 5.7 Frontend twins

```python
# tucker_tensor_train.py / uniform_tucker_tensor_train.py
    def oversampled_ranks(
            self,
            p:        int,                                 # the rank increment, p >= 0
            sharing:  typ.Optional[typ.Sequence] = None,   # len=d group labels; None = unshared
    ) -> typ.Tuple[
        typ.Tuple[int, ...],   # tucker ranks, len=d   (uniform: HOST int (d,)+stack_shape)
        typ.Tuple[int, ...],   # tt ranks, len=d+1     (uniform: HOST int (d+1,)+stack_shape)
    ]:

# frame_variations_format.py
    @staticmethod
    def padded(                                            # T3Frame.padded
            x_old:         't3.TuckerTensorTrain',         # the old point (reduced to minimal ranks here); rank 0 = the zero tensor
            tucker_ranks:  typ.Sequence[int],              # len=d, target
            tt_ranks:      typ.Sequence[int],              # len=d+1, target (boundary 1)

            sharing:       typ.Optional[typ.Sequence] = None,   # len=d group labels; None = unshared
            seed:          int = ZERO_FRAME_SEED,
    ) -> 'T3Frame':                                        # the seeded padded frame (old frame in the leading slots)

def fv_aligned_padded_frame(                               # frame_variations_format.fv_aligned_padded_frame
        x_old:               't3.TuckerTensorTrain',       # the old point; rank 0 = the zero tensor
        tucker_ranks:        typ.Sequence[int],            # len=d, target
        tt_ranks:            typ.Sequence[int],            # len=d+1, target
        hint:                typ.Any,                      # TuckerTensorTrain / dense array, or anything with `project`
        project:             typ.Optional[typ.Callable] = None,   # project(frame.data, hint) -> variations data; None = by the hint's class

        sharing:             typ.Optional[typ.Sequence] = None,   # len=d group labels; None = unshared
        oversampling:        int   = ALIGN_OVERSAMPLING,          # the backend's named defaults
        oversampled_passes:  int   = ALIGN_OVERSAMPLED_PASSES,
        max_passes:          int   = ALIGN_MAX_PASSES,
        tol:                 float = ALIGN_TOL,
        alternate:           bool  = ALIGN_ALTERNATE,
        seed:                int   = ZERO_FRAME_SEED,
        use_jit:             bool  = False,                # jax inputs: a memoized jitted step in the host loop (D-39)
) -> typ.Tuple[
    'T3Frame',                 # the aligned padded frame at the target ranks
    typ.Tuple[NDArray, ...],   # movements, one per pass
]:

# uniform_frame_variations_format.py
    @staticmethod
    def padded(                                            # UT3Frame.padded
            x_old:         'ut3.UniformTuckerTensorTrain',              # the old point (reduced to minimal ranks here); rank 0 = the zero tensor
            tucker_ranks:  typ.Union[typ.Sequence[int], NDArray],       # len=d, or HOST int (d,)+C, target
            tt_ranks:      typ.Union[typ.Sequence[int], NDArray],       # len=d+1, or HOST int (d+1,)+C, target

            sharing:       typ.Optional[typ.Sequence] = None,           # len=d group labels; None = unshared
            seed:          int = ZERO_FRAME_SEED,
    ) -> typ.Tuple[
        'UT3Frame',            # the seeded padded frame in block layout
        typ.Tuple[int, int],   # split (Na, Rb), for UT3Tangent.align_padded
    ]:

def ufv_aligned_padded_frame(                              # uniform_frame_variations_format.ufv_aligned_padded_frame
        x_old:               'ut3.UniformTuckerTensorTrain',              # the old point; rank 0 = the zero tensor
        tucker_ranks:        typ.Union[typ.Sequence[int], NDArray],       # len=d, or HOST int (d,)+C, target
        tt_ranks:            typ.Union[typ.Sequence[int], NDArray],       # len=d+1, or HOST int (d+1,)+C, target
        hint:                typ.Any,                                     # UniformTuckerTensorTrain (x_old's padded N), or anything with `project`
        project:             typ.Optional[typ.Callable] = None,           # project(frame_data, hint) -> variations data; None = by the hint's class

        level_widths:        typ.Optional[typ.Tuple[int, int, int, int]] = None,   # (WU, WD, WL, WR); None = ufv_frame_widths(x_old.N, max target Tucker rank, max target TT rank)

        sharing:             typ.Optional[typ.Sequence] = None,           # len=d group labels; None = unshared
        oversampling:        int   = ALIGN_OVERSAMPLING,
        oversampled_passes:  int   = ALIGN_OVERSAMPLED_PASSES,
        max_passes:          int   = ALIGN_MAX_PASSES,
        tol:                 float = ALIGN_TOL,
        alternate:           bool  = ALIGN_ALTERNATE,
        seed:                int   = ZERO_FRAME_SEED,
        use_jit:             bool  = False,                               # the whole loop as one jax.jit; `project` must then be traceable (D-39)
) -> typ.Tuple[
    'UT3Frame',   # the aligned frame at level_widths, prefix masks (the level's frame structure for that padding)
    NDArray,      # movements, shape=(max_passes,)+C, NaN after the last pass
]:

# manifold.py
    def align_padded(                                      # T3Tangent.align_padded
            self,
            old_tucker_ranks:  typ.Sequence[int],                       # len=d; the zero tensor: (0,)*d
            old_tt_ranks:      typ.Sequence[int],                       # len=d+1; the zero tensor: (1, 0, ..., 0, 1)

            tucker_ranks:      typ.Optional[typ.Sequence[int]] = None,  # len=d, output; None = the frame's own
            tt_ranks:          typ.Optional[typ.Sequence[int]] = None,  # len=d+1, output; None = the frame's own

            sharing:           typ.Optional[typ.Sequence] = None,
            update:            str = 'ULRO',
    ) -> typ.Tuple[
        'bvf.T3Frame',   # the aligned frame
        NDArray,         # movement, shape = frame stack C
    ]:

# uniform_manifold.py
    def align_padded(                                      # UT3Tangent.align_padded
            self,
            split:         typ.Tuple[int, int],                                       # (Na, Rb), from UT3Frame.padded

            tucker_ranks:  typ.Optional[typ.Union[typ.Sequence[int], NDArray]] = None,   # len=d or HOST int (d,)+C, output; None = the frame's own
            tt_ranks:      typ.Optional[typ.Union[typ.Sequence[int], NDArray]] = None,   # len=d+1 or HOST int (d+1,)+C; None = the frame's own

            sharing:       typ.Optional[typ.Sequence] = None,
            update:        str = 'ULRO',
    ) -> typ.Tuple[
        'ubv.UT3Frame',   # the aligned frame, block layout, P-blocks real-first
        NDArray,          # movement, shape = frame stack C
    ]:

# optimizers.py: the new keywords, appended keyword-only after `tr_radius0` (newton_cg) / `g0_reference` (gradient_descent)
def newton_cg(
        geometry, kind, sample, data, x0, ..., tr_radius0=None,   # unchanged
        *,
        start_frame:                typ.Optional[typ.Union['bvf.T3Frame', 'ubv.UT3Frame']] = None,           # a frame of x0 (after its reduction) for iteration 0
        padded_from:                typ.Optional[typ.Union['t3.TuckerTensorTrain', 'ut3.UniformTuckerTensorTrain']] = None,   # the old point; rank 0 = the zero tensor

        padded_oversampling:        typ.Optional[int]   = None,   # None = 5; padded_from only
        padded_oversampled_passes:  typ.Optional[int]   = None,   # None = 2
        padded_max_passes:          typ.Optional[int]   = None,   # None = 10
        padded_tol:                 typ.Optional[float] = None,   # None = 0.05
        padded_alternate:           typ.Optional[bool]  = None,   # None = True
        padded_seed:                typ.Optional[int]   = None,   # None = ZERO_FRAME_SEED

        cone_oversampling:          typ.Optional[int]   = None,   # None = 5; g0_reference='cone' without g0norm_newton only
        cone_oversampled_passes:    typ.Optional[int]   = None,   # None = 2
        cone_max_passes:            typ.Optional[int]   = None,   # None = 10
        cone_tol:                   typ.Optional[float] = None,   # None = 0.05
        cone_alternate:             typ.Optional[bool]  = None,   # None = True
        cone_seed:                  typ.Optional[int]   = None,   # None = ZERO_FRAME_SEED
) -> typ.Tuple[
    typ.Union['t3.TuckerTensorTrain', 'ut3.UniformTuckerTensorTrain'],   # x_opt (the module's `Point` alias)
    dict,                                                                # stats
]:

```

All set options are forwarded to the backend; unset ones are not passed (G-bit). The frontend `gradient_descent`
keeps its signature (checked: it ends in `**kwargs`, forwarded to the backend). `g0_reference='cone'` and the six
`cone_*` options reach the backend `gradient_descent` through that `**kwargs`. There they are explicit, keyword-only
parameters appended after `g0_reference`, as in §5.6, and the frontend docstring documents them.

## 6. Tests

**Helpers.**

- `tests/backend/_padded_frame_oracles.py`: the dense per-edge reference. It ports `P/dense_oracle.py`
  (`dense_frames`, `dense_pass`, `tangent_basis`, `dense_project`, `piece_basis`, `distances`) and
  `P/cycle/cycle_schemes.select_pass` (two-colour and transport).
- `tests/backend/_uniform_padded_frame_testing.py`: `P/uniform_testing.py`'s scatter. It puts a ragged case into block
  layout with gappy masks in BOTH blocks and FINITE garbage in every junk entry. The garbage is written into a copy of
  the clean array's own layout, so equal layouts make bitwise invariance meaningful (`testing_strategy.md`).

**Measures.** Frames are compared by the largest principal-angle sine per directed edge.

**Inputs.** Generic hints have ≥ `k` directions per slot in every projector-equality test (J-2), and movements away
from `tol` at the decisive pass.

**Criteria** follow D-24. Bitwise only for structural facts: the old block (copied), junk (masked or constructed),
masks, tied factors (one array), placed 0/1 seeds, gathers. Tolerances otherwise.

| # | file | test | pass criterion |
|---|---|---|---|
| T1 | `tests/backend/test_padded_frame.py` | `tv_align_padded_frame` vs the dense update. Cases: d = 1, 2, 3, 5; uneven increments; non-minimal ranks; tangent part 10×; truncating; the empty old frame; groups; stack C | ≤ 1e-9 on all four stores; orthonormal ≤ 1e-12; old block bitwise; both chains reconstruct X_old ≤ 1e-12 |
| T2 | same | Ungauged variations; hint scale 1e-14 … 1e14; float32 | ≤ 1e-9 vs gauged; same frames ≤ 1e-9; float32 error ≤ 30·u·max(1, s) (measured ≈ 5u at s = 0) |
| T3 | same | Degenerate hints (zero, rank one, an information-free increase); structural rejections, each isolated: transport plus truncation, a bad `update`, old stores above the output stores, unequal group stores | valid frames (orthonormal, pinned, reconstructs); each a `ValueError` |
| T4 | same (N3) | Two-colour = transport of the other stores (vs `select_pass`). The cycle regression on the 243-number case, stored as a Python module of float literals, `tests/backend/_padded_frame_cycle_case.py` (`S`, `X_old`'s cores, target ranks, seed; checked bitwise against the source `g06c_cycle_case.npz`). The repository forbids binary fixtures: `*.npz` is git-ignored, being platform- and version-dependent. all-stores cycles (movement > 0.05 at pass 200); alternating converges (movement < 1e-8) | ≤ 1e-9 per pass; as stated |
| T5 | same | (D-41, N8b; replaces the S2-parity and `fv_zero_point_frame` rows) `fv_padded_frame` against an INDEPENDENT reference per store: the sign-fixed numpy QR of `[old block | stream rows w..w+k−1]`, with the stream rows regenerated in the test from the documented key `(seed, store kind, group's first mode)`, at the slot indices and the ragged unfolding length. Cases: old widths 0 (the empty frame), generic and structured; groups (tied output, one array); stacks. Plus the convention properties: slot `s` uses row `s`; the zero-tensor start has random orthonormal cores and differs from `fv_zero_point_frame`. N1's S2-parity record is superseded | ≤ 1e-14; tied factors one array; seed rows bitwise (structural) |
| T6 | same | `fv_aligned_padded_frame` vs the dense loop with the same schedule (two-colour via `select_pass`; p = 0 and p = 5; 40 passes). Edge cases: working stores = target stores, incl. `+p` ranks that reduce to the target's stores (no oversampled phase); `oversampled_passes ≥ max_passes` (the last pass truncates); `max_passes = 0` with p = 0 and p = 5 (the start at the TARGET ranks, equal to `fv_padded_frame(old_frame, target)` exactly since it is the same call; `project` never called, checked with a raising `project`; empty movements); `max_passes = 1` with working ≠ target (one truncating pass) and with working = target; target stores = old stores (no pass, empty movements, the old frame); the PAIR stop rule (stop only after a `(UL, RO)` pair; an odd remainder runs one final `'UL'` only when not stopped); `max_passes = 2` at working = target. **Eager vs `use_jit=True`** on jax inputs (the memoized jitted step): same pass counts, frames ≤ 1e-12 | ≤ 1e-9 per pass; pass counts as stated; movements a tuple of `C` arrays; eager = jit as stated |
| T7 | `tests/backend/test_linalg.py` | `pad_safe_pinned_completion` vs `pinned_completion` on the real block; interior pads in rows, pins and seeds; per-element `seed_mask` in a stack; offset masks (columns `t..t+k−1`); count rejection | ≤ 1e-14 on the real block (same sign rule); junk rows and unselected columns exactly 0 |
| T8 | `tests/backend/test_uniform_padded_frame.py` | `ufv_padded_frame` vs `fv_padded_frame` through the scatter (8 cases incl. the empty frame, four old widths, groups); the scattered seed rows equal the ragged ones bitwise (D-41, structural); the completions are compared on old frames built by scattering ONE ragged frame, so the slot signs agree (D-49). Exact output masks vs the independently derived `[A-mask as given | prefix(counts)]`. `ufv_empty_frame` = the scatter of `fv_empty_frame`. jit vs numpy | ≤ 1e-12; masks exact; junk exactly 0; jit ≤ 1e-12 |
| T9 | same | `utv_align_padded_frame` vs `tv_align_padded_frame` (the T1 list incl. d = 1; two-colour; truncating). Exact output masks. Garbage robustness (clean vs dirty) | ≤ 1e-9; masks exact; clean == dirty bitwise (equal layouts) |
| T10 | same | Stacks: C = (3,) with different ranks per element, C = (2, 2); stacked = per element (step and start in N5, loop in N6) | ≤ 1e-13 (D-45) |
| T11 | same | `ufv_compact_frame`: masks; real entries moved; junk zero; `utv_retract` of the zero tangent at the level returns X_old; `from_frame` variation masks == `from_point`'s | masks exact; gathers bitwise; ≤ 1e-12 |
| T12 | same | `ufv_aligned_padded_frame` vs `fv_aligned_padded_frame` (defaults; tol 1e-10; the empty frame; non-minimal; p = 0; d = 2; alternate on/off), generic hints with movements away from `tol`. The no-op level (target stores = old stores) returns `ufv_compact_frame(old_frame_data, level_widths)`: prefix masks at `level_widths`, all-NaN movements. **Eager vs `use_jit=True`** (the whole-loop jit) on the same cases, incl. the pair rule's odd-remainder final pass and an early stop | same pass counts; movements ≤ 1e-12 (NaN at the same positions); spans ≤ 1e-9; the no-op's masks exact; eager = jit: same pass counts, frames ≤ 1e-12 |
| T13 | same + `tests/backend/test_padded_frame.py` + `tests/test_dispatch.py` | **Compile counts** (`_cache_size()` on the memoized wrappers, as the mask-pattern test; d ≥ 3). (a) Ragged `use_jit=True`, ≥ 6 passes at fixed structure: the jitted step's cache holds at most four programs after the loop, and a second loop at the same structure adds none. (b) Uniform `use_jit=True`: the whole-loop jit compiles ONCE per structure. A second call at the same structure, with another `tol` (traced) or another hint of the same shapes, adds nothing; another `max_passes` (static) adds one. (c) Uniform `use_jit=False` on jax: no scan body recompiles across passes (module-level bodies). First-call times at d = 6 and 24, eager and jitted (informational, beside §2.3's figures) | cache sizes as stated |
| T14 | `tests/backend/test_padded_start.py` | **Optimizer seam:** (a) `require_frame_structure` (through `local_model(frame=)` and the frontend `start_frame`) rejects, one family at a time with every other byte identical: each ragged store's shape; each uniform variation-mask family (up, down, left, right); each boundary entry (`left[d]`, `right[0]`); `shape`; a padded width. Plus a mutation check per family (deleting that comparison in the member turns the test red). It accepts the aligned frame and `geom.frame(x)`. A geometry without the member raises. (b) `newton_cg(padded_from=)` ≡ `newton_cg(start_frame=padded_start(...)[0])`. (c) G-bit: `set(stats)` and history keys unchanged without new keywords; the new keys only with them; a before/after equality of a seeded run recorded in the N7 handoff. (d) `'cone'`: deterministic given the seed; `cone_max_passes=0` (any `cone_oversampling`) equals the frame-gradient norm at the Gaussian hint-free zero-tensor start (`fv_padded_frame(fv_empty_frame(...), ...)`), NOT `'zero'` (D-41); `padded_max_passes=0` starts Newton from the hint-free padded frame (the iteration-0 frame equals `fv_padded_frame(old_frame, target)` on the ragged layer and `ufv_compact_frame(ufv_padded_frame(...)[0], level_widths)` on the uniform one, ≤ 1e-14; no projection runs); zero passes with an explicitly set `*_oversampling` / `*_oversampled_passes` / `*_tol` / `*_alternate` raises in `newton_cg`, `gradient_descent` and `padded_start`, one option at a time (`*_seed` allowed); independent of `x0`'s values; corewise / no member / vanishing explicit / options without their group / `cone_*` with `g0norm_newton` raise; `'cone'` with `g0norm_newton` alone skips the loop; `gradient_descent` too. (e) **not verified before this test:** `padded_start` uniform vs ragged on `probe_derivatives`, d ≥ 3, by gauge-invariant quantities (per-pass `‖P_F g‖`, spans). (f) **not verified before this test:** a shared problem on both layers, one pass vs a dense tied-projection reference. Root cause of the N7 cross-layer difference (D-42, not a bug): at singular-value ties (`share()` and `t3svd` give orthonormal Tucker factors) the two layers' orthogonal representations pick different bases, and the hint-free start reads its seeds in those bases. So the layers' STARTS differ in span (0.27–1.00 on such points, 1e-15 on generic ones), while the aligned loop's outputs agree to its tolerance (2e-5 … 5e-3 at the default tol, 3e-14 … 5e-10 at tol 1e-10, measured on the coordinate convention; re-measured on the Gaussian one: starts 0.95–1.00, outputs 7e-4 … 3e-2 at the default tol and 8e-13 … 6e-11 at tol 1e-10, D-49). The row compares the loop's OUTPUT at tol 1e-10, and the starts only on generic points. (g1) **not verified before this test:** at a numerically full-rank `x` with a gauge-rotated frame `F` of `x` (same structure, accepted by the member), `local_model(x, frame=F, hessian='exact')` passes the `Problem.local_model` doctest's FD criterion along the retraction from `F`, and its quadratic of a matched direction equals the default frame's. (g2) **not verified before this test:** at the zero-padded `x0`, `newton_cg(hessian='exact', start_frame=F)` is finite and decreases the objective (FD is invalid there: the retraction is not first-order accurate at a numerically rank-deficient point). (h) `newton_cg(padded_from=<rank-0 tensor>)` and a partially-zero-rank `padded_from` (Tucker `(0,2,2)`, TT `(1,0,2,1)`), on both layers, through the frontend AND directly through the backend members. Both give the empty frame's start (equal to the canonical rank-0 case, ≤ 1e-14); a stack mixing zero and nonzero elements raises in the members. (i) **not verified before this test:** `newton_cg(..., use_jit=True, padded_from=...)` and `g0_reference='cone'` on the UNIFORM layer run `_misfit_gradient_at` inside the whole-loop jit (the problem's gradient traced). Equal to `use_jit=False` (same pass counts, the iteration-0 frame ≤ 1e-12 relative to its scale, the cone value ≤ 1e-12 relative); one flag governs CG and the alignment | as stated; (d) `'cone'` at zero passes = the hint-free start's frame-gradient norm ≤ 1e-12 relative; (e) ≤ 1e-9 at tol 1e-10, ≤ 0.2 at the default stop; (f) ≤ 1e-9; (g1) ≤ 1e-4 relative to FD, matched quadratics ≤ 1e-10 relative; (g2) as stated |
| T15 | ONE file, `tests/test_aligned_padded_frame_frontend.py` (D-45) | Twins: each equals its backend call (no added arithmetic); hint-class dispatch; the `N` check; cross-layer hints; the rank-0 spelling; the frame twins' empty frame for a structurally-zero `x_old` (partially-zero ranks included); `newton_cg(padded_from=)` passing a structurally-zero `x_old` through unconverted (D-43); every §3.6 validation message; safe-mode checks (ORTH, TIED, same tensor) raise and are skipped under `safety.unsafe()`; doctests | array_equal; as stated |
| T16 | `tests/test_dispatch.py` | Rows for every new PUBLIC backend function, host loops included (`fv_aligned_padded_frame`, `ufv_aligned_padded_frame`, `padded_start`): numpy/jax in → out, ragged/uniform, stacked; uniform masks stay host | all rows |
| T17 | `tests/backend/test_uniform_padded_frame.py` | `ufv_frame_widths(N, n, r)` == the widths of `ut3_orthogonal_representations` over a sweep (n > N, r² < n, d = 1..4) | exact |
| T18 | `tests/test_import_graph.py` (new; J-5) | The graph is built on module FILES, not packages, so `optimizer_display`'s `from t3toolbox import corewise` does not route through `t3toolbox/__init__`. The backend import graph is acyclic; `backend/optimizers.py` imports only `common`, `regularization` and `corewise` | exact |
| T19 | `tests/backend/test_padded_start.py` | `ranks.oversampled_ranks` on `(d,)+C` arrays = the per-element calls; sequences unchanged | exact |
| T20 | `tests/backend/test_padded_frame.py`, `tests/backend/test_uniform_padded_frame.py` (N8b) | **Start-from-start chains** (D-41), both layers: `F1 = start(old)`, then `F2 = start(F1 as the old frame, at higher ranks)`, and a third step. On generic points and on structured ones (identity-prefix / zero-point cores, a `t3svd` output, a shared point). Cases include a store with NO new unfolding rows (the completion lies inside the old block's sub-grid: the old convention's hole). Checks: no stream row reused across the chain; every pinned-QR pivot bounded away from zero (min `|R_jj|` over the selected columns ≥ 1e-3: a property of the chosen seeds, checked deterministically, not a guarantee); orthonormal ≤ 1e-12; old blocks bitwise; the two layers agree through the scatter | as stated; layers ≤ 1e-12 |

## 7. Docs and CHANGELOG

**New pages.**

- `docs/aligned_padded_frame.md` (user doc) covers:
  - what the aligned start is, and when to use `padded_from` and `'cone'` (`gtol_rel=1e-4` with `'cone'`, D-24);
  - the schedule and its knobs, with zero passes (`padded_max_passes=0`, `cone_max_passes=0`) documented as the
    cheap generic start / reference, and the measurements: the generic frame carried the level-2 gain, and the
    alignment's gain appeared at 11 → 16 (D-38);
  - **a "jit or eager" section** (D-39):
    - the rule: nothing jits unless asked, and `use_jit=True` on the loops (or `newton_cg`'s own `use_jit`, which
      also governs the alignment) selects it;
    - what each layer does under it: the ragged host loop with a jitted step; the uniform whole loop as one program,
      whose `project` must then be traceable (the library's projections and the problem's gradient are);
    - the figures: an eager first step at d = 4 is ~22 s of op-by-op compilation (932 compilations) and 0.16 s
      after; jitted it is one ~2 s compile; the uniform whole loop's first call is ~9 s at d = 6 and 24;
    - **the memory-mapping hazard** (D-40). Each eager-jax alignment at a new structure compiles hundreds of
      executables and adds 8,000–9,500 process memory mappings. An eager-jax rank continuation that aligns at every
      level could pass `vm.max_map_count` (65,530) after about 7 levels and abort with `LLVM ERROR: Unable to allocate
      section memory!`. The doc RECOMMENDS `use_jit=True` for jax inputs, which compiles 2–4 programs per level
      (ragged) or 1 (uniform). It is DOCUMENTATION ONLY: the loops do not warn (Nick);
    - `gradient_descent` (eager only) runs its cone reference eagerly;
  - what the measurements showed (§2.3, with the simultaneous / two-colour distinction; the hint-free baselines
    re-recorded at N10 on the Gaussian convention);
  - **the start's convention and its representation dependence** (D-41, D-42): Gaussian slot-indexed seeds (isotropic,
    well conditioned, chain-safe). The start is a function of the old frame's representation, so different bases at
    singular-value ties (`t3svd` output, shared points) give different, equally valid starts, while the aligned
    loop's output is representation-independent up to its tolerance (§2.4's numbers);
  - the uniform movement record: NaN after the last pass, so it cannot tell a NaN movement from an unrun pass; a NaN
    movement never stops the loop (D-44);
  - the zero point as a rank-0 tensor;
  - the uniform block layout as a user sees it;
  - "store" defined in one clause on first use, linking `docs/frame_variations.md` § "Four rank stores" (Nick: that
    definition stands as is);
  - one sentence: `g0_reference='warm_start'` with a start frame is `‖g‖` at that frame;
  - one sentence on the regularizer (§3.5).
- **Docstrings (N9, Nick's requirements, D-48).** The frontend aligned-loop docstrings (both layers) carry real content;
  the list below is a floor, not a ceiling. Frontend docstrings may be much more detailed than backend ones. Be
  creative: the reader should understand the function, see it by example, and be forewarned about the gotchas.
  - The PURPOSE: bias a frame's completion toward a hint.
  - A BRIEF method summary: a fixed-point iteration on projections. Each pass fills the missing basis at every store
    from the hint's projection, given the other stores' current completions; two-colour `UL` / `RO` ordering after
    the truncation.
  - WHAT the fixed point satisfies, in the VERIFIED form (D-50, §2.2), using the docs' unified wording verbatim. Each
    store's update maximizes its own content's capture given the others (alternating Gauss–Seidel updates, not
    ascent of `E`). The fixed point is a stationary point of `E = ‖Π_{T(F)} hint‖²` but generally a saddle: a minimum
    per store, lowered by paired rotations. Proved for matrices; measured on T3s. A strict local maximum of the cone
    merit is observed on small trees, not proved. Demonstrate what holds.
  - PROPERTIES: many fixed points; no global guarantee; the representation dependence of the hint-free start,
    including slot signs (D-42, D-49); the loop output representation-independent to `tol`.
  - An OVERVIEW of the parameters: oversampling and oversampled passes, `max_passes`, `tol`, `alternate` (the
    two-colour switch), `use_jit`, `max_passes=0`.
  - A LINK to the user doc.
  - DOCTESTS illustrating the properties, platform-stable (structural facts or inequalities, never printed floats):
    (a) the fixed-point property at convergence (one more pass moves nothing; each store's completion equals the
    leading directions of its content); (b) oversampling on vs off at the SAME small pass count, the oversampled run
    capturing more.
  - "Store" is linked to `docs/frame_variations.md` on first use (also in the `padded_frames.py` module docstring).
  - The uniform whole-loop kernel's docstring and the N9 scan-body note state that its `functools.partial` binding
    is accepted IN THIS INSTANCE ONLY and sets no precedent (D-47).
- `docs/contributor/padded_frame_internals.md` covers:
  - the message passing in library terms;
  - the block layout, the empty frame, the compaction and why;
  - the pad-safe pinned completion and why a QR;
  - the jittable kernels, the memoized jit getters, the whole-loop jit's static/traced split, and the compile
    accounting (D-39);
  - the T3-free seam: the three members, and why the structure check works from integers and held masks;
  - the route to a problem's aligned frame without Newton (`least_squares_problem` → `padded_start`).
- `docs/aligned_padded_frame.tex` (+pdf): the standalone math note, written after the code. It is Nick's note
  specialized to T3s, with algorithm boxes citing the implemented functions.

**Existing pages.**

- `docs/rank_continuation.md`: a section pointing to the new user doc. Rewrite "On the uniform layer": continuation
  stays ragged, and the frame twins exist on both layers.
- `docs/fitting_and_optimization.md`: §5's `g0_reference` table gains `'cone'`; a note on `padded_from` /
  `start_frame`; §3's `Geometry` members gain the two optional members.
- `docs/numerical_contracts.md`:
  - the padded-frame contracts: old block bitwise, completions ⟂, degenerate content arbitrary, no transport with
    truncation;
  - TIED for `x_old` under sharing;
  - the `'cone'` row (structural errors; the convention caveat).
- `docs/contributor/uniform_svd_prefix_orthogonalization.md`: "There is no QR in any orthogonalization path" gains
  the one seeded completion and why it is not an orthogonalization.
- `docs/contributor/verification_records.md`: the §2.3 facts and the acceptance numbers.
- `docs/api_reference.rst`: the new backend modules.
- The `Geometry` protocol docstring in `backend/optimizers.py`: the two optional members.

**Records.**

- CHANGELOG `[2026.4.0]`: added / changed (none) / removed (none released); release notes.
- `dev/padded_frame_cone_judgment_calls_2026_11.md` gains:
  - J-7: the style pause;
  - J-8: the cycle and two-colour;
  - J-9: the QR in the uniform start;
  - J-10: the empty frame and rank-0 spelling.

## 8. Slices

Each slice runs the same way:

1. The implementer builds.
2. The reviewer reviews, style section first: every input and output a basic immutable type; no template objects;
   scans with module-level bodies; `contract` strings; annotations.
3. The session checks the diff and the full suite, commits, and starts the next slice.

Review stops (Nick + session) after N3, N6, N8 and N10.

| slice | content | files | gates |
|---|---|---|---|
| N1 | **Cleanup and the ragged start.** (Done, 3f9679ef: S2 was verified bitwise live from git rather than frozen; that S2-parity record is superseded by D-41.) Freeze S2's outputs on T5's cases FIRST. Then: `fv_operations.py` back to `efe5e3d`; delete the S2/S3 tests and dispatch rows (keep S1's T10 row); remove S3's ten HISTORICAL strings (keep S1's two); refresh `test_padded_start.py`'s docstring; discard the S4 working tree. Create `backend/padded_frames.py` with `fv_empty_frame` and `fv_padded_frame`; widen `ranks.oversampled_ranks`; add `tests/test_import_graph.py` | `backend/fv_operations.py`, `backend/padded_frames.py`, `backend/ranks.py`, `tests/backend/test_padded_frame.py`, `tests/backend/test_padded_start.py`, `tests/test_dispatch.py`, `tests/test_contractions_interpreter.py`, `tests/test_import_graph.py`, `tests/data/` | full suite green; `git diff efe5e3d -- t3toolbox/backend/fv_operations.py` empty; T5, T16 rows, T18, T19 |
| N2 | `tv_align_padded_frame`; the Appendix C ragged strings in HISTORICAL; the dense reference module | `backend/padded_frames.py`, `tests/backend/_padded_frame_oracles.py`, `tests/backend/test_padded_frame.py`, `tests/test_contractions_interpreter.py` | T1–T3; the contraction suite |
| N3 | `fv_aligned_padded_frame`; the cycle case. **Review stop.** Review-stop clarification: dispatch rows (T16) are required for every public backend function, host loops included, so the loop gets its T16 row | `backend/padded_frames.py`, `tests/backend/test_padded_frame.py`, `tests/backend/_padded_frame_cycle_case.py`, `tests/test_dispatch.py` | T4, T6, T16 row |
| N3b | **N3 follow-up (D-39):** `use_jit` on `fv_aligned_padded_frame` (the memoized jitted step, static on the old and output ranks, `sharing`, `update`; the host loop and `project` eager); the PAIR stop rule of §2.3 replaces the per-pass rule, so the layers agree pass for pass | `backend/padded_frames.py`, `tests/backend/test_padded_frame.py`, `tests/test_dispatch.py` | T6 (pair rule; eager = jit), T13(a), T16 row with `use_jit` |
| N4 | `pad_safe_pinned_completion`, `ufv_frame_widths`, `ufv_empty_frame`, `ufv_padded_frame`; the scatter helper | `backend/linalg.py`, `backend/ufv_masking.py`, `backend/uniform_padded_frames.py`, `tests/backend/test_linalg.py`, `tests/backend/_uniform_padded_frame_testing.py`, `tests/backend/test_uniform_padded_frame.py` | T7, T8, T17 |
| N5 | `utv_align_padded_frame` (block layout, the JITTABLE kernel with explicit statics, eager by default, scans, `pad_safe_svd`, groups, stacks, `update`); the uniform strings | `backend/uniform_padded_frames.py`, `tests/backend/test_uniform_padded_frame.py`, `tests/test_contractions_interpreter.py` | T9, T10 (step, start) |
| N6 | `ufv_compact_frame`, `ufv_aligned_padded_frame` with `use_jit` (the host loop calling the kernel eagerly by default; under `use_jit=True` the whole-loop jit: `fori_loop`, the truncating call, the `(UL, RO)` `while_loop`, the final conditional `'UL'`, the compaction; NaN-padded movements). **Review stop** | same, `tests/test_dispatch.py` | T10 (loop), T11, T12 (eager = jit), T13(b, c), T16 rows |
| N7 | The seam: geometry members (`use_jit` passed through); `Problem.local_model(frame=)`; `_misfit_gradient_at`; `padded_start` (`use_jit`); the `newton_cg` / `gradient_descent` keywords, `newton_cg`'s `use_jit` forwarded to the alignment; `'cone'` | `backend/geometry.py`, `backend/optimizers.py`, `tests/backend/test_padded_start.py`, `tests/backend/test_optimizers.py`, `tests/test_dispatch.py` | T14 (incl. (i)), T16 rows, T18, G-bit |
| N8 | Frontend twins and docstring moves. **Review stop** | `tucker_tensor_train.py`, `uniform_tucker_tensor_train.py`, `frame_variations_format.py`, `uniform_frame_variations_format.py`, `manifold.py`, `uniform_manifold.py`, `optimizers.py`, and T15's files | T15, doctests |
| N8b | **D-41: Gaussian seeds everywhere, slot-indexed** (before N9). The ragged start's seed rows from the keyed stream `(seed, store kind, group's first mode)`, slot-indexed, of the ragged unfolding's length; delete `_coordinate_seed_order` and the outside/inside branch. The uniform start: the same host rows scattered, one `pad_safe_pinned_completion` per store; delete the unit-vector scatter. T5 restated (independent pinned-QR reference); T8 (seed scatter bitwise); T20 (chains, both layers, the no-new-rows case, pivots bounded away from zero); T14(d) (`'cone'` at zero passes = the hint-free start, not `'zero'`); the dispatch rows and doctests that print starts | `backend/padded_frames.py`, `backend/uniform_padded_frames.py`, `tests/backend/test_padded_frame.py`, `tests/backend/test_uniform_padded_frame.py`, `tests/backend/_uniform_padded_frame_testing.py`, `tests/backend/test_padded_start.py`, `tests/test_dispatch.py`, `tests/test_aligned_padded_frame_frontend.py` | T5, T8, T14(d), T20; the full suite on both environments (the floor job is where the bug surfaced) |
| N9 (DONE, cc7d6fc6) | Docs (incl. §7's jit-or-eager section, the representation-dependence statement and Nick's docstring requirements, D-48), CHANGELOG, release notes, judgment calls, the math note | `docs/`, `CHANGELOG.md`, `dev/` | doctests in docs; `sphinx -W` |
| N10 (IN PROGRESS) | Acceptance (§11), with the SHIPPED default schedule. **Review stop** | T3Polynomial (its own process) | the table of §11 filled |

## 9. Decisions

| # | decision | by / when |
|---|---|---|
| D-1 | The hinted path is the note's core form (frame + variation cores in; width old + 2k), replacing the dense construction | Nick 2026-11-09 |
| D-2 | Scope: the uniform layer IS in this release; NEWTON ONLY gets the aligned start; gradient-based loops later | Nick 2026-11-09 |
| D-3 | The committed S2/S3 hinted construction is deleted (never released); S1's primitives and the hint-free start stay; S4 discarded | Nick 2026-11-09 |
| D-4 | API shape: start / step / loop; `project` + `hint`; `project` REQUIRED in the backend (no shape-based dispatch), class dispatch in the frontend | Nick 2026-11-09 |
| D-5 | `g0_reference='cone'` reports the frame-gradient norm at the aligned frame | Nick 2026-11-09 |
| D-6 | Defaults: oversampling 5, oversampled passes 2, cap 10, tolerance 0.05, all user-settable kwargs; start and cone each have their own keyword group with the same defaults. The groups are realized by D-33 | Nick 2026-11-09 (groups: session) |
| D-7 | The cap is the effective stop; the tolerance an early exit; the cone reference = "the value after the schedule", deterministic given the seed | Nick 2026-11-09 |
| D-8 | Frontend twins: `T3Frame.padded`, tangent-level step, standalone loop with class dispatch, `oversampled_ranks` on both tensor classes; docstrings / doctests / validation in the twins; backend functions without twins keep their doctests. The loop is realized per layer (D-35) | Nick 2026-11-09 |
| D-9 | Drop the three FRONTEND geometry twins; defer `frame=` on the six fitting factories | Nick 2026-11-09 |
| D-10 | Uniform geometry: CHECK-AND-REJECT a start frame whose masks differ from the level's (the loop returns matching masks) | Nick 2026-11-09 |
| D-11 | The start keeps the committed convention in both layers; its uniform twin is a pad-safe Householder pinned completion (the only QR; justified in §3.4) | Nick 2026-11-09 |
| D-12 | Degenerate content: arbitrary completions accepted where the hint carries no information | Nick 2026-11-09 |
| D-13 | A stacked loop stops on the maximum movement over the stack | Nick 2026-11-09 |
| D-14 | Two-colour updates after the truncating pass (default on) | Nick 2026-11-09 |
| D-15 | No polymorphism between layers; a uniform version only because it is truly uniform (block layout); otherwise users convert UT3 → T3 → UT3 | Nick 2026-11-09 |
| D-16 | Uniform padding for rank increases: one uniform old width and one uniform completion width, masks for the smaller edges | Nick 2026-11-09 |
| D-17 | Masks are the only truth in uniform inputs; our functions may return junk-last; the loop compacts at exit | Nick 2026-11-09 |
| D-18 | Library code: scans, named contractions [i.e. grouped `contract('…')` strings, §3.1(4)], shape annotations; no raw QR where order matters | Nick 2026-11-09 |
| D-19 | Backend = pure functions on immutable basic types; the gold standard is the only precedent; `fv_zero_point_frame` unchanged this release (audit arc later) | Nick 2026-11-02 |
| D-20 | **RESOLVED: host loops by default** (amended by D-39). With `use_jit=False` both loops are host loops calling eager steps: `project` is an arbitrary callable, and the stop branches on the movement on the host. Under `use_jit=True` the ragged loop stays a host loop around a jitted step, and the uniform loop becomes one jitted program (D-39) | session; writer and critic r1; orchestrator r1; amended r7 |
| D-21 | **RESOLVED: the zero old point** is the empty frame in the backend (D-29), and a rank-0 tensor in the frontend: structural, carrying shape, stack and backend. ONE rule everywhere (r3): a tensor is structurally zero when any Tucker rank or interior TT rank is 0. The frontend converts such an `x_old` to the canonical rank-0 tensor first, and both members apply the same test. The twins skip the reduction (`compute_minimal_ranks` maps the last boundary rank to 0) and the uniform orthogonalization (it raises on rank 0). `None` remains only in the T3-free optimizer layer for the cone (`padded_start(x_old=None)`, the member), which cannot build the empty frame itself. A rank-0 `padded_from` reaches the backend as rank-0 data; both members accept it, and the uniform member builds `ufv_empty_frame` (r2) | writer and critic r1; orchestrator r1; critic r2 |
| D-22 | **RESOLVED:** `fv_aligned_padded_frame` (ragged) and `ufv_aligned_padded_frame` (uniform), as the prefix grammar requires | writer and critic r1; orchestrator r1 |
| D-23 | **RESOLVED:** `backend/padded_frames.py` (Nick, round 10) and `backend/uniform_padded_frames.py` (the per-layer module pairs `t3_svd` / `ut3_svd`); no new frontend module (D-35) | writer and critic r1 |
| D-24 | Carried over from the superseded plan: non-breaking additive release; no `dtype`; `padded_from` takes the old point (reduced structurally to minimal ranks); keyword-only appended parameters; gauge-equivalent cross-layer claims, bitwise only for structural facts; numerical tolerance checks for numerical tests; `compute_orthogonal_representation_ranks` stays in `ranks`; `gtol_rel=1e-4` with `'cone'` in the docs | Nick 2026-11-02 … 11-09 |
| D-25 | Backend geometry members are how the T3-free optimizer layer reaches layer-specific operations: `with_frame`, `aligned_padded_frame` and `require_frame_structure` (r2), optional and duck-typed, absent on corewise. D-9 dropped only the frontend twins. `backend/optimizers.py` imports nothing new | orchestrator r1, r2 (critic r1 M1, r2 M1) |
| D-26 | The optimizer layer's `project` is the module-level `_misfit_gradient_at(frame, hint)` with `hint = (problem, residual)`, the residual at the loop's point: `x0`, or the zero tensor for `'cone'`. No closure | orchestrator r1 (critic M2) |
| D-27 | G-bit: without a new keyword the WHOLE `stats` dict is bit-identical. New keys only with a new keyword. No `NewtonInfo` field. No display change | orchestrator r1 (critic M3) |
| D-28 | Results are tuples (`padded_start -> (frame, movements)`); no result classes in the backend | orchestrator r1 |
| D-29 | No `None` for the zero tensor in the frame functions: the empty frame, from `fv_empty_frame(shape, stack_shape, use_jax)` / `ufv_empty_frame(shape, N, stack_shape, use_jax)` (default dtype policy, as `t3_zeros`) | orchestrator r1 |
| D-30 | `pad_safe_pinned_completion` takes per-element counts as masks, with a static output width; host permutations | orchestrator r1 |
| D-31 | Test criteria per D-24: bitwise only for structural facts | orchestrator r1 |
| D-32 | Every g06 arm used the simultaneous step. No two-colour rerun now; acceptance N10 runs the shipped default | orchestrator r1 |
| D-33 | `newton_cg`: `start_frame`, `padded_from`, and two plain groups of six (`*_oversampling`, `*_oversampled_passes`, `*_max_passes`, `*_tol`, `*_alternate`, `*_seed`): fourteen keywords. `gradient_descent` gets the cone group. Option defaults are `None` = the D-6 value, so an option set without its group engaged raises (house rule). That includes `cone_*` set with `g0norm_newton` given (the cone is then not computed); `'cone'` itself stays legal there, as `'zero'` does (r2). A schedule record would be smaller but is the rejected `PaddedFrameOptions` (superseded I-3) | writer r1, for review; orchestrator r1 (every knob settable) |
| D-34 | The uniform step is a JITTABLE module-level kernel with explicit statics, jitted only under `use_jit` (r7, D-39): masks and shape via `partition_static`; every host-derived mask precomputed as host bool and the group tables as tuples of Python ints (never integer numpy arrays, which would be traced); split and `update` static; the module's own `lru_cache`d jit getter; module-level, closure-free scan bodies | orchestrator r1 (critic M4); amended r7 |
| D-35 | The frontend loop: one twin per layer beside its frame class (`frame_variations_format.fv_aligned_padded_frame`, `uniform_frame_variations_format.ufv_aligned_padded_frame`), each dispatching on the hint's class. Reasons: `uniform_manifold` imports `manifold` (no cycle-free cross-layer home without a new module); D-15; the house's per-layer frontend free-function pairs (`t3_` / `ut3_orthogonal_representations`). The frontend `padded_start` twin is dropped: it is not in D-8, and `newton_cg(padded_from=)` is the user path. The one case it would serve, a problem's aligned frame without running Newton, goes through the PUBLIC backend `padded_start` (doctest: the `least_squares_problem` route). Flagged to Nick | writer r1; critic r2 agrees; correction r2 |
| D-36 | The loop cap counts the truncating pass. Movements: the ragged loop returns a tuple of `C` arrays; the uniform loop returns a fixed-length `(max_passes,)+C` array, NaN after the last pass, in both modes (r7: the whole-loop jit needs a static shape). (r1 also had "`max_passes = 0` only when working = target"; superseded by D-38) | writer r1 (critic MINOR 13); amended r7 |
| D-38 | **Zero alignment passes is a supported, documented spelling.** `max_passes = 0` returns the seeded hint-free padded frame at the TARGET ranks: no projection, empty movements; the loops do not read their other schedule options then.<br>Through the optimizers: `newton_cg(padded_from=x_old, padded_max_passes=0)` starts Newton from it. `g0_reference='cone', cone_max_passes=0` gives the reference at the hint-free frame of the zero tensor. That frame was `fv_zero_point_frame`'s (= `'zero'` to rounding) under the coordinate convention; since D-41 it is the Gaussian slot-indexed frame, so `'cone'` at zero passes is NOT `'zero'` (r8).<br>By D-33's rule (an option nothing reads is a mistake), zero passes with an explicitly set `*_oversampling`, `*_oversampled_passes`, `*_tol` or `*_alternate` RAISES in `newton_cg`, `gradient_descent` and `padded_start`. The message names the fix: remove the option or set a positive pass count. `*_seed` is read and allowed.<br>Reason: the Newton measurement (§2.3) found the generic start frame carries the level-2 Newton gain, and the alignment adds nothing measurable beyond the seed spread at laptop scale. The defaults (D-6) are unchanged | orchestrator r4, r5 (from the Newton measurement; critic r5) |
| D-39 | **Jit is under the user's control, as everywhere in the library.** Backend functions stay eager; `use_jit: bool = False` on `fv_aligned_padded_frame` and `ufv_aligned_padded_frame` (and their frontend twins, the members and `padded_start`) selects jit, the `newton_cg` / `mc_sgd` / `adam` precedent.<br>Ragged: the host loop calls a memoized module-level jitted step (static: old ranks, output ranks, `sharing`, `update`); two to four step programs per level, nothing recompiles across passes; the loop itself is never jitted (its per-mode Python loops would unroll).<br>Uniform: the WHOLE loop is one `jax.jit` (`fori_loop` for the oversampled passes; one truncating call; a `while_loop` whose body is a `'UL'` pass then an `'RO'` pass, the pass counter in the state; the compaction). The cap is static; `tol` and the hint's arrays are traced; `project` must be traceable.<br>Both loops follow the PAIR stop rule (§2.3) so eager, jitted and cross-layer runs agree pass for pass.<br>`newton_cg`'s existing `use_jit` governs both the CG core and the alignment; `gradient_descent` has no `use_jit` (eager by design), so its cone runs eagerly.<br>Reason (measured, §2.3): an eager first ragged step at d = 4 is 932 op-by-op compilations, about 22 s (0.16 s after); jitted, one ~2 s compile; the uniform whole-loop jit compiles in about 6 s at d = 6 and 24 | Nick 2026-11-10 |
| D-40 | **Eager-jax risk is DOCUMENTATION ONLY, no warning.** The user doc's "jit or eager" section states the costs: ~22 s of op-by-op compilation on the first eager step at d = 4, and 8,000–9,500 memory mappings per eager alignment at a new structure, so about 7 levels to `vm.max_map_count`. It recommends `use_jit=True` for jax inputs. The loops do not warn | Nick 2026-11-10 (third review stop) |
| D-41 | **Gaussian seeds everywhere, slot-indexed.** The hint-free start's completion seeds for EVERY store (Tucker and TT/down) are rows of a keyed legacy `RandomState` stream, key `(seed, store kind, sharing group's first mode)`. Slot `s` uses stream row `s`, so a start at old width `w` draws rows `w..w+k−1` and start-from-start chains never reuse a row. Rows have the ragged unfolding's length and are generated on the host per group and stack element, then scattered; pinned completion as before.<br>Reason: the floor CI job (numpy 1.26 / OpenBLAS) found that the Tucker stream restarted at row 0 per call. A chained start reused a seed lying in the current span (pivot 1e-16, the column decided by BLAS rounding: 0.349 between the layers on CI). The cores' inside fallback had the same hole on chains and coordinate-structured points.<br>Nick's further reasons: isotropy (a uniformly random subspace of the complement is the natural no-information completion) and conditioning (pivots bounded away from zero w.h.p.; completions continuous in the old frame off a null set).<br>Consequences: the zero tensor's start changes (random orthonormal cores; `fv_zero_point_frame` itself is unchanged), the `'cone'` value changes (D-12 stands; D-38 amended); `_coordinate_seed_order`, the outside/inside branch and the uniform unit-vector scatter are deleted; T5 restated, T20 added; N1's S2-parity record superseded; the Darcy hint-free baselines re-recorded at N10. Slice N8b | Nick 2026-11-10 |
| D-42 | **The hint-free start is a function of the old frame's representation, including slot signs (D-49); document, no code change.** T14(f)'s cross-layer difference was not a bug. At singular-value ties (`t3svd` outputs, shared points) the layers' orthogonal representations choose different bases, and the start's seeds are read in those bases: on the coordinate convention, starts differ by 0.27–1.00 in span while the loop's outputs agree to 2e-5 … 5e-3 at the default tol and 3e-14 … 5e-10 at tol 1e-10. Re-measured on the Gaussian convention (D-49): starts 0.95–1.00, aligned frames 7e-4 … 3e-2 at the default tol (equal pass counts), 8e-13 … 6e-11 at tol 1e-10. All starts are valid (D-12). Documented in N9; canonical bases at ties are a later audit item | Nick 2026-11-10 |
| D-43 | `newton_cg(padded_from=)` passes a structurally-zero `x_old` through as given (not converted to the canonical rank-0 tensor); the backend members' structural-zero test handles it. The frame twins build the empty frame directly | N8 (session), accepted |
| D-44 | Uniform details as built in N6. `ufv_compact_frame` zero-pads an axis first when the level is wider than the input frame. A NaN movement never stops the loop, and the NaN-padded record cannot tell a NaN movement from an unrun pass (documented in N9) | N3b, N6 (session), accepted |
| D-45 | Test logistics as built. T15 in ONE file, `tests/test_aligned_padded_frame_frontend.py`. Stacked-vs-per-element tolerances are 1e-13. The scatter helper is `tests/backend/_uniform_padded_frame_testing.py`. Small test data are Python modules of literals (`*.npz` is git-ignored), without a general rule | N4–N8 (session), accepted |
| D-46 | `uniform_padded_frames` imports seven private math helpers from `padded_frames` (the seed rows among them); accepted as a sibling-module convenience, recorded in §3.2 | N5 (session), accepted |
| D-47 | The uniform whole-loop jit binds its statics with one `functools.partial` inside the single jit. Accepted IN THIS INSTANCE ONLY; it sets NO precedent. The kernel docstring and the N9 scan-body note say so | Nick 2026-11-10 (second review stop) |
| D-48 | N9's docstring requirements (§7): the purpose, a brief method, what the fixed point satisfies (in D-50's verified form and the docs' unified wording), the properties (many fixed points; representation dependence incl. slot signs), a parameter overview, a link to the user doc. Platform-stable doctests of the fixed-point property and of oversampling on/off at equal pass counts. Frontend docstrings may be much richer than backend ones; "store" is linked to `docs/frame_variations.md` on first use | Nick 2026-11-10 |
| D-49 | **The Gaussian core seeds are not sign-covariant; accept and document.** A slot sign flip (arbitrary in an unfixed LAPACK SVD) flips a core unfolding's rows. The old block flips with them; the seeds do not, so the completion's span changes (0.987 for one bond). The ragged and uniform starts of a generic point differ by ~1, and a hint-free start can differ across BLAS builds; Tucker completions are unaffected. RULING: accept and document. No local row-gauge rule, and no representation-level sign canonicalization now (that joins the canonical-bases audit item). D-42's "representation" means "representation including slot signs". Re-measured washout on the Gaussian convention: starts differ 0.95–1.00 in span; aligned frames agree to 7e-4 … 3e-2 at the default tol with equal pass counts, and to 8e-13 … 6e-11 at tol 1e-10 | Nick 2026-11-10 |
| D-50 | **What the fixed point satisfies (verified, J-51).** Each store's update maximizes `‖P⁺ᵀY‖²` for its own content given the others: alternating Gauss–Seidel updates, NOT ascent of the captured frame energy `E = ‖Π_{T(F)} g‖²`. A fixed point is a stationary point of `E` but generally a SADDLE: per store a minimum (rotating one store's completion alone never lowers `E`, generically raises it at second order); a paired rotation lowers it. Proved for matrices (`docs/padded_frame_cone.tex`, Proposition 1); measured on T3s in both layers. A strict local maximum of the cone merit is OBSERVED on small trees, not proved. The docs use the unified wording of §2.2 verbatim; §7 and D-48 corrected | reviewer-verified 2026-11-10 (J-51) |
| D-37 | `Problem.local_model(frame=)` (and the frontend `start_frame`) checks structure through the member `require_frame_structure(x_cores, frame)`: integers and held masks only, never an orthogonalization. Ragged: the frame's core shapes against `compute_orthogonal_representation_ranks` of `x`'s ranks, plus `C` and `N`. Uniform: the frame's variation masks against `var_masks`, the boundary entries, `shape`, the widths. The r1 text (compare with `geom.frame(x)`) is withdrawn: it orthogonalized to read structure, the pattern the arc was paused for | orchestrator r2 (critic r2 M1) |

## 10. Risks and open questions

- **The alignment's Newton value is established only at one higher-rank laptop level.** At levels 1–2 the
  alignment passes bought no Newton iterations beyond the generic start frame. At 11 → 16, with 10 or more passes,
  they cut about 40 % (§2.3). 20×1000 is to measure it at scale (§11). The loop is a start-frame heuristic, not a
  solver, slow and seed-dependent on real problems; the defaults are a budget. The start also decides the basin
  (§2.3, level 1), in both directions.
- **Not verified, with tests added:**
  - the uniform kinds' `precompute` / `transpose` at block-layout frames (T14(e));
  - the tied projection under sharing at padded frames, both layers (T14(f));
  - `hessian='exact'` from a supplied start frame (T14(g));
  - `d = 1` (T1, T9);
  - the `ufv_frame_widths` rule beyond the writer's 60-structure check (T17);
  - the jit boundary's compile count (T13).
- **`chunk_size` at block widths.** The uniform `probe_derivatives` kind's chunk is sized at the level's widths;
  block-layout passes are wider. Memory is not measured (superseded O-7). Acceptance logs peak memory on one level.
- **Cross-layer identity** rests on the start reproducing bases, not spans. Degenerate seeds (an inside seed in the
  old span, up to roundoff) may differ across backends: the same caveat as the ragged start. The `'cone'` value
  agrees across layers only when the oversampled content has full rank (§2.3); otherwise it is layer-dependent by
  D-12.
- **dtype.** An empty-frame start uses the default float, so `'cone'` runs in float64 on float32 numpy data.
- **Eager jax** (D-40): ~22 s of first-call compilation per new structure, and 8,000–9,500 memory mappings per eager
  alignment: an eager-jax continuation could abort after about 7 levels (`vm.max_map_count`). Mitigation: documentation
  recommending `use_jit=True`; no warning.
- **Representation dependence** (D-42, D-49): the hint-free start depends on the old frame's bases at singular-value
  ties and on its slot signs (even at generic points: cross-layer and cross-BLAS starts can differ by about 1).
  Valid by D-12, and washed out by the loop to its tolerance; canonical bases at ties are a later audit item.
- **The seed convention changed after N1–N8** (D-41): every printed start, doctest and baseline from the coordinate
  convention is re-derived in N8b / re-recorded at N10.
- **GPU** has not been measured anywhere.
- **Large d.** Near-tied spectra converge slowly; the cap applies.

## 11. Downstream and acceptance

**Baseline: the prototype Newton measurements** (simultaneous step; full tables in §2.3). Newton / CG iterations
per start seed:

| start | level 1 (1 → 6) | level 2 (6 → 11) | 11 → 16 |
|---|---|---|---|
| library padded frame (today) | 21/181 | 49/1480 | 36/1657 |
| hint-free padded frame (`padded_max_passes=0`; coordinate convention, re-record at N10) | 16/254, 19/197 | 28/714, 24/687 | 32/1860, 32/1551 |
| aligned, 4 passes | 17/286, 11/109 | 22/648, 28/1031 | 24/1517, 25/1246 |
| aligned without oversampling, 4 passes | 14/246, 24/504 | 42/1151, 29/758 | — |
| aligned, 10 passes (the default cap) | 20/273, 25/276 | 28/1360, 33/873 | 21/699, 22/538 |
| aligned, 40 passes | 16/181, 18/273 | 21/729, 24/801 | 22/700, 22/703 |

The capture fraction `‖P_F g‖ / ‖grad f‖`, levels 1 → 6 / 6 → 11 / 11 → 16 / 16 → 21:

- library .189/.117/.106/.101;
- hint-free .104/.093/.094/.093;
- aligned at 10 passes .464/.354/.270/.217.

**Reading.**

- At level 2, the hint-free start and the aligned starts with oversampling cut 49 to 21–33; without oversampling,
  42 and 29.
- The alignment adds nothing measurable beyond the seed spread (up to 13) at levels 1–2.
- At 11 → 16, with 10 or more passes, it cuts Newton by about 40 % and CG by about 2.4× against both other starts;
  4 passes are not enough.
- Newton at 16 → 21 was out of reach on the laptop.

**Acceptance (N10) runs the SHIPPED library code with the SHIPPED default schedule (two-colour):**

- T3Polynomial adopts `padded_from` and `g0_reference='cone'` in its ladder (its own process; `t3polynomial/cone.py`
  retired).
- Laptop: the Darcy o01 ladder, levels 1 → 6 and 6 → 11.
  - Starts: library; hint-free (`padded_max_passes=0`); aligned with the default schedule (10 passes), and at 4 and
    40 passes; aligned without oversampling.
  - Measure: Newton and CG counts, objective, test error; two seeds; compile logging and peak memory on one level.
  - The baseline above goes beside them.
- Runpod, 20×1000, decoupled from the release. It measures the alignment's value in Newton counts at scale, the
  open question of §0: library vs hint-free (`padded_max_passes=0`) vs the default aligned start.

## 12. Pitfalls checklist (for every implementer and reviewer)

- [ ] Every backend input and output is an array, tuple of arrays, int, static int tuple, host bool mask, or a frozen
  dataclass whose parameters are fields. No result classes.
- [ ] Nothing orthogonalized to read its structure, anywhere (structure checks from integers and held masks:
  `require_frame_structure`).
- [ ] The kernel's host data static: bool masks and tuples of Python ints, never integer numpy arrays.
- [ ] No object read for its shapes outside the optimizer layer; no dict of shapes; structure as `Sequence[int]` with
  `len=` comments.
- [ ] No `None` for the zero tensor in a frame function: use the empty frame.
- [ ] `project` is required in the backend loops; `hint` is never inspected; the optimizer layer's `project` is
  module-level with explicit data.
- [ ] `backend/optimizers.py` imports nothing new (T18).
- [ ] Scans for the sweeps (`xscan`; reversed inputs for reverse sweeps). Every scan body is module-level and
  closure-free; statics come from shapes or an `lru_cache` factory on ints and bools. Batched maps over the mode
  axis in the uniform layer; no Python loop over modes in uniform mathematics.
- [ ] Grouped `contract('…')` strings from Appendix C, each in HISTORICAL; no raw einsum; no `@` on stacked operands.
- [ ] A shape annotation on every signature line; `frame_data` / `variations_data` for uniform raw tuples; `sharing`
  labels on public entries.
- [ ] Uniform inputs are read through masks only. Junk is masked once at entry; the split is the only positional
  fact; output junk-last with masks.
- [ ] `pad_safe_svd` on `[Ā | guard(Y⊥)]` for the step (the projected content); `pad_safe_pinned_completion` for the
  start; no other QR. `root` is the plain thin SVD.
- [ ] Nothing jits unless asked: `use_jit=False` by default. The uniform step is a jittable module-level kernel with
  explicit statics; the ragged step's jit is a memoized module-level wrapper; the uniform whole-loop jit traces `tol`
  and the hint's arrays and holds the cap static (D-39).
- [ ] Both loops use the pair stop rule (§2.3).
- [ ] Structural counts and checks only; no rank tolerance; degenerate content left arbitrary.
- [ ] Start seeds from the keyed, slot-indexed stream (D-41): slot `s` uses row `s`, never a restart at row 0; the
  uniform start scatters the ragged host rows.
- [ ] Eager-jax costs are documented, never warned about (D-40).
- [ ] The old block bitwise in the output; carriers hold the old content only; groups tied bitwise.
- [ ] Transport plus truncation raises.
- [ ] The uniform loop returns the frame at the level's widths (compaction).
- [ ] The cone residual is at the zero tensor.
- [ ] Defaults unchanged; new keywords appended keyword-only; `stats` bit-identical without them.
- [ ] Docstrings, doctests and validation in the frontend twins; backend one-liners; untwinned doctests stay.

## Appendix A. Prototype → library map

| prototype | library |
|---|---|
| `core_form2.align_stores`; `cycle/core_form3.py` `update=` | `padded_frames.tv_align_padded_frame` |
| S2 hint-free path (`fv_operations.fv_padded_frame(template, old)`) + `core_form.start_frame` | `padded_frames.fv_padded_frame(old_frame, tucker_ranks, tt_ranks, ...)` |
| `core_form.stores_of_ranks(shape, None, None)`; `uniform_start._old_block_layout(None)` | `fv_empty_frame`, `ufv_empty_frame` |
| `ragged_loop.py` | `padded_frames.fv_aligned_padded_frame` |
| `uniform_start.uniform_start(method='gs')`, `gs_complete` | `uniform_padded_frames.ufv_padded_frame`, `linalg.pad_safe_pinned_completion` |
| `uniform_form2.uniform_align2` | `uniform_padded_frames.utv_align_padded_frame` |
| `uniform_loop.uniform_aligned_padded_frame(alternate=True)`, `compact` | `uniform_padded_frames.ufv_aligned_padded_frame`, `ufv_compact_frame` |
| `dense_oracle.py`, `cycle/cycle_schemes.select_pass` | `tests/backend/_padded_frame_oracles.py` |
| `uniform_testing.py` | `tests/backend/_uniform_padded_frame_testing.py` |
| `g06c_cycle_case.npz` | `tests/backend/_padded_frame_cycle_case.py` (float literals; checked bitwise against the source) |

## Appendix B. From the superseded plan

**Kept:**

- §2.2 of the old plan (tangent cone, frame tangent spaces, aligned vs staggered completions), as background for the
  math note;
- S1 (`pinned_completion`, `pinned_leading_columns`, `oversampled_ranks`), as committed, with `oversampled_ranks`
  widened;
- the optimizer seam through geometry members (its §3.4 reasoning: they keep `backend/optimizers.py` T3-free);
- the structural reduction of `x_old`, the keyword-only rule, the CHANGELOG / non-breaking gates, the acceptance
  protocol (§11);
- the ledger of deferred items: O-7 `chunk_size` at working ranks; O-8 start frames for other methods;
  `fv_zero_point_frame`'s template signature for the audit arc.

**Dropped:**

- the hint as a T3, the dense pinned T3-SVD construction and its exactness caveat, `tangent_to_t3`;
- the frontend geometry twins and the `padded_frame` / `padded_structures` / `with_ranks` members;
- the fixed pass counts, and "no uniform twin".

## Appendix C. Grouped contraction strings (engine letters; all in `AXIS_SIZE`)

**Conventions.**

- One group `C` on every operand, with fixed lowercase counts, so `C` is solved from any operand: every supplement
  is `''`.
- Letters: `a`, `b`, `i`, `k`, `n`, `u`, `x`, `z` (sizes 2, 4, 3, 2, 6, 3, 7, 5 in `AXIS_SIZE`). No extension is
  needed.
- The uniform map sites use the `d`-prefixed twin (`'dC…'`), with `d` the leading mode or group axis. The scan-body
  sites use the plain string.
- Three strings already exist and are reused: `'Cia,Cib->Cab'`, `'Cia,Cab->Cib'` (S1) and `'Caxb,Caub->Cxu'`
  (2026.3.0).
- The prototypes' einsum or matmul each string replaces is named, so the implementer translates one by one.

| label | string | operands (ragged §4.1) | prototype | d-twin (uniform map) |
|---|---|---|---|---|
| c1 | `Ckn,Cun->Cku` | `PhiO = VP · Uᵀ` over the physical index | `'...kn,...un->...ku'` | yes |
| c2 | `Cun,Cku->Cnk` | `Uᵀ PhiOᵀ` (the XiO argument) | `T(U) @ T(PhiO)` | yes |
| c3 | `Cxkb,Ckn->Cxnb` | `βO ×_down PhiO` (left msg) | `'...xkb,...kn->...xnb'` | — |
| c4 | `Cxnk,Ckb->Cxnb` | `βL ×_out PhiL` | `'...xnk,...kb->...xnb'` | — |
| c5 | `Cxkb,Ckz->Cxzb` | `βO ×_down XiO` | `'...xkb,...km->...xmb'` | — |
| c6 | `Cxnk,Ckz->Cxnz` | `βL ×_out XiL` | `'...xnk,...km->...xnm'` | — |
| c7 | `Cxnb,Canb->Cxa` | `PhiL' = ⟨Vb, R⟩` | `'...xnb,...ynb->...xy'` | — |
| c8 | `Cxa,Canb->Cxnb` | `PhiL' ×₁ R`; also `NEW` on the first axis | `'...xy,...ynb->...xnb'`, `_new(-3)` | yes (down edges) |
| c9 | `Cakx,Ckn->Canx` | `βO ×_down PhiO` (right msg) | `'...akx,...kn->...anx'` | — |
| c10 | `Cknx,Cka->Canx` | `βR ×_in PhiR` | `'...knx,...ka->...anx'` | — |
| c11 | `Cakx,Ckz->Cazx` | `βO ×_down XiO` | `'...akx,...km->...amx'` | — |
| c12 | `Cknx,Ckz->Cznx` | `βR ×_in XiR` | `'...knx,...km->...mnx'` | — |
| c13 | `Canx,Canb->Cxb` | `PhiR' = ⟨Vb, L⟩` | `'...anx,...any->...xy'` | — |
| c14 | `Cxb,Canb->Canx` | `PhiR' ×₃ L`; also `NEW` on the last axis | `'...xy,...any->...anx'`, `_new(-1)` | yes (down edges) |
| c15 | `Caxk,Ckb->Caxb` | `βL ×_out PhiL` (up msg) | `'...axk,...kb->...axb'` | yes |
| c16 | `Ckxb,Cka->Caxb` | `βR ×_in PhiR` | `'...kxb,...ka->...axb'` | yes |
| c17 | `Caxk,Ckz->Caxz` | `βL ×_out XiL` | `'...axk,...km->...axm'` | yes |
| c18 | `Ckxb,Ckz->Czxb` | `βR ×_in XiR` | `'...kxb,...km->...mxb'` | yes |
| c19 | `Caxb,Caub->Cxu` (exists) | `PhiU = ⟨Vb, O⟩` | `'...axb,...amb->...xm'` | yes |
| c20 | `Cxu,Caub->Caxb` | `PhiU ×₂ O` | `'...xm,...amb->...axb'` | yes |
| c21 | `Cik,Ckb->Cib` | `g_old Φ_P`, `g_old Ξ`, the one-hot selection `U_s sel` | `g_old @ PhiP`, `Us @ sel` | yes |
| c22 | `Cia,Cib->Cab` (exists) | `o = P⁺ᵀ g_old`, `D = [Ā|P⁺]ᵀ g_new`, `ĀᵀY` | `_T(P) @ g_old` | yes |
| c23 | `Cia,Cab->Cib` (exists) | `P⁺ o`, `Ā(ĀᵀY)` | `P @ o` | yes |
| c24 | `Cun,Cnk->Cuk` | `DU = U⁺ g_new` | `U_new @ gn` | yes |
| c25 | `Cxn,Canb->Caxb` | `NEW` on the Tucker axis | `_new(-2)` | — |
| c26 | `Cakx,Cnk->Canx` | `O_P ×_down DU` (left edge) | `'...akx,...nk->...anx'` | — |
| c27 | `Cknx,Cak->Canx` | `R_P ×_in D_L` | `'...knx,...ak->...anx'` | — |
| c28 | `Cxkb,Cnk->Cxnb` | `O_P ×_down DU` (right edge) | `'...xkb,...nk->...xnb'` | — |
| c29 | `Cxnk,Cbk->Cxnb` | `L_P ×_out D_R` | `'...xnk,...bk->...xnb'` | — |
| c30 | `Caxk,Cbk->Caxb` | `L_P ×_out D_R` (down edge) | `'...axk,...bk->...axb'` | yes |
| c31 | `Ckxb,Cak->Caxb` | `R_P ×_in D_L` (down edge) | `'...kxb,...ak->...axb'` | yes |

**Not contractions** (allowed as they are):

- the batched thin SVD in `root`, and `pad_safe_svd`;
- elementwise mask multiplies and the movement's sum of squares;
- slices, `pad_or_truncate`, concatenation;
- unfold / fold reshapes and `moveaxis` on core axes;
- host gathers (group tables, compaction).

New HISTORICAL rows: c1–c31 minus the three existing ones (28), plus the d-twins marked "yes" (16 rows: c1, c2, c8,
c14–c24, c30, c31), minus `'dCaxb,dCaub->dCxu'`, which already exists (checked): 15 new d-rows.
