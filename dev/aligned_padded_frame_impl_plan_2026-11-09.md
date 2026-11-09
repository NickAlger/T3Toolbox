# Aligned padded frame — implementation plan (fresh plan, 2026-11-09)

> **Status: APPROVED BY THE CRITIC (review r4, 2026-11-09) after three writer rounds
> (`dev/aligned_padded_frame_plan_review_r{1,2,3,4}.md`); awaiting Nick's and the session's review.** Then implementation
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
> **For the maintainer.** None of D-1–D-19 is reopened. Two places where the plan realizes a ruling in a form you
> should confirm:
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
> backend `padded_start` (§3.5, with a doctest of the route `least_squares_problem` → `padded_start` → rewrap). Say
> if you want the twin.

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
| 2 | 10 | critic r2 NIT 10–15 | Return tuples expanded one element per line. `_frame_theta` named correctly. The loop defaults are named once (`ALIGN_*` in `padded_frames.py`) and referenced everywhere; `padded_start` takes `None` and forwards only what is set. T18 builds the graph on module files. "`oversampled_passes` is then ignored" added. D-18 gains a bracketed gloss | §2.3, §3.5, §5, T18, D-18 |

## 0. Why, in one page

At a rank increase in rank continuation, the converged point `X_old` is zero-padded to the new ranks. Its frame at
the new ranks is the old frame, which must stay exactly as it is, plus a **completion**: new orthonormal directions
in every store that the data do not determine. Today these are filled without looking at the data:

- `t3_orthogonal_representations` fills them with whatever LAPACK's QR leaves;
- the 2026.3.0 zero-point frame fills them with a seeded Haar draw.

The first Newton steps at the new level are then spent discovering directions the gradient already knows.

This feature **aligns the completion with the gradient** by a fixed-point loop

    F  <-  align(F, project(F, hint)).

- `project(F, hint)` is the orthogonal projection of one fixed tensor `S` onto the tangent space at the current padded
  frame `F`. `S` is the gradient at `X_old`, or any tensor.
- `align` replaces every completion by the leading new directions of that tangent vector, with the old blocks held
  fixed.

The loop runs a few passes with the completion oversampled. It then runs passes at the target ranks, updating the
two edge directions on alternate passes. It runs once per rank set, at the outset, in two places:

- the Newton start, `newton_cg(..., padded_from=x_old)`;
- a data-aware reference gradient norm at the zero tensor, `g0_reference='cone'`.

**Measured on the laptop Darcy ladder, level 1** (frame-gradient norm `‖P_F G‖`; converged value about 40.5):

| start | `‖P_F G‖` |
|---|---|
| hint-free start | 8.4 |
| the library's own padded frame | 15.3 |
| aligned start, four passes, simultaneous step (the g06 measurement) | 36.4 |
| aligned start, four passes, two-colour default (`P/cycle/darcy_schemes.py`, seed 0) | 35.9 |

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

Let `working = ranks.oversampled_ranks(shape, target, p, sharing)`, and `n_over = oversampled_passes` if working ≠
target (for any stack element), else `0`. With `oversampling = 0` or maximal ranks there is no oversampled phase,
and `oversampled_passes` is then ignored. The D-36 clamp below is the only other place a set value is not honoured;
the user doc states both.

    F_0 = the start (§2.4) at the working ranks
    pass it = 0 .. n_over-1  : F <- step(F, project(F, hint)) at the working ranks, update 'ULRO'
    pass it = n_over         : the same, OUTPUT AT THE TARGET RANKS (truncating when working ≠ target), 'ULRO'
    pass it > n_over         : alternate=True:  'UL' when (it − n_over) is odd, 'RO' when even
                               alternate=False: 'ULRO'
    stop after pass it when
        alternate=True:  it > n_over and the movements of passes it−1 and it are both below tol
                         (at it = n_over+1 the pair is the truncating pass and the first 'UL' pass)
        alternate=False: it ≥ n_over and the movement of pass it is below tol
        or it + 1 = max_passes
    return F, (movement of every pass)

The movement of a pass is its maximum over edges, and the stop compares the maximum over the stack too (D-13).
When the target stores equal the old stores, there is nothing to complete. The loop returns an empty tuple of
movements, without a pass. The ragged loop returns the old frame. The uniform loop returns
`ufv_compact_frame(old_frame_data, level_widths)`, so its return contract (level widths, prefix masks) holds.

**The cap counts the truncating pass.** If `oversampled_passes ≥ max_passes`, then `n_over = max_passes − 1`, so the
last pass truncates. This departs from the prototypes, which ran an extra truncating pass. `max_passes ≥ 1` is
required when working ≠ target. `max_passes = 0` is legal only when working = target, and returns the start.

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
| Newton from the aligned start (Darcy ladder): **pending**, the measurement is running (simultaneous step, fixed pass budgets) | `g06_aligned_frame.py` (`PART=newton`) |

So: the pass cap is the effective stop, and the tolerance is an early exit. The `'cone'` reference is "the value
after the default schedule", deterministic given the seed (D-7).

### 2.4 The start (the committed convention, both layers)

**Tucker factors.** Per sharing group `g`, with key `min(g)`:

1. Take the first `k` rows of `RandomState([seed, min(g)]).standard_normal((k, N_g))` as seeds.
2. Complete them against the old factor: `pinned_completion(U°ᵀ, seedsᵀ, k)`.

With no old rows, this IS the Haar recipe of `fv_zero_point_frame` (the `pinned_completion` docstring says so).

**TT and down cores.**

1. The coordinate directions of the slot's unfolding OUTSIDE the old block, in the core's row-major order. These are
   placed directly as exact 0/1 values; they are orthogonal to the old block, which is zero there.
2. Then the Householder completion INSIDE the old block, seeded by the block's own coordinates in index order.

**At the zero tensor** this is `fv_zero_point_frame`'s frame: TT and down cores identical (constructed 0/1 values),
Tucker factors to 1e-14 (S2, commit `2562264a`). So the cone reference and the `'zero'` reference share their start.

**The convention is basis-dependent.** A TT seed names a coordinate of its neighbours' completion slots. The uniform
twin must therefore reproduce the same Gram–Schmidt bases, not just the spans (§3.4). Agreement across layers:
≤ 3.4e-16 (`P/check_uniform_start.py`).

### 2.5 The reference at the zero tensor

`g0_reference='cone'` runs the loop at the zero tensor of `x0`'s ranks: the old frame is the empty frame, and the
residual is `r0 = point_forward(0) − data`, exactly as `_zero_reference_gnorm` builds it (the regularizer is excluded
for the same reason). It reports the **frame-gradient norm** `‖P_{T_F} 𝒥ᵀ r0‖` at the returned frame, at the cost of
one more projection. This is like-for-like with the gradient norms the stopping test compares it to (D-5).

- It does not depend on `x0`'s values, only on its ranks.
- The default `'zero'` (one Haar frame) is unchanged.
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
   - The uniform step runs through one module-level jitted kernel with explicit statics (§3.4, D-34).
   - The uniform loop returns its frame at the level's padded widths, so the first Newton step compiles nothing new.

### 3.2 Modules and layering

- **`backend/padded_frames.py` (new).** The ragged empty frame, start, step and loop. Imports downward only:
  `common`, `contractions`, `linalg` (`pinned_completion`, `pinned_leading_columns`, `pad_or_truncate`), `ranks`,
  `sharing` (`canonical_groups`), `fv_operations` (`ZERO_FRAME_SEED`, a public constant; no private helper). The
  `ranks → sharing` dependence is a smell for the later audit arc, not touched here.
- **`backend/uniform_padded_frames.py` (new).** The uniform empty frame, start, step, compaction and loop. Imports
  `common`, `contractions`, `linalg`, `ranks`, `sharing`, `ufv_masking`, `fv_operations` (`ZERO_FRAME_SEED`), and
  `padded_frames` for shared host bookkeeping only (the Haar seed rows and the coordinate-seed order).
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
- **Jit boundary (D-34).** The public function prepares all host data on the host, in forms `partition_static`
  keeps static:
  - every host-derived mask as a host BOOL array: the output P-masks (the counts' prefixes), the per-group A-masks,
    the edge row masks, and the one-hot selections;
  - the group index tables (`members`, `valid`, `first`, `group_of`) as nested tuples of Python ints and bools. An
    integer numpy array is traced by `partition_static` (`common._is_static_leaf`), so no host integer array crosses
    the boundary.

  `partition_static` then splits `(frame_data, variations_data, host data)`, and the function calls ONE module-level
  kernel `_utv_align_kernel(dynamic, skeleton, split, update)`. Inside the kernel the tuples become `np` index
  arrays at trace time (host constants for the gathers).
  - On numpy the kernel runs directly.
  - On jax it runs through `_jitted_align_kernel()`, an `lru_cache`d module-level getter in
    `uniform_padded_frames.py` that returns `jax.jit(_utv_align_kernel, static_argnums=(1, 2, 3))`.
    `optimizers._jitted` is private to `backend/optimizers.py`, which this module must not import.
  - The static arguments are `skeleton` (a value-hashed `StaticSkeleton`), `split` (an int tuple) and `update` (a
    str). A pass at an already-seen structure therefore compiles nothing.

**Start** (`ufv_padded_frame`).

- The seeds are the ragged convention's, built on the host:
  - Haar seed rows per group;
  - coordinate seeds at the REAL rows of each unfolding, in their ragged row-major order (a cumsum of the masks),
    outside the A-block first, then inside.
- Each store is one batched completion: a map, no sweep.
  - Outside unit vectors are placed directly (exact 0/1).
  - The Tucker store and the inside completions use `pad_safe_pinned_completion`.
- It reproduces the ragged start to 3.4e-16, with junk exactly zero (`P/check_uniform_start.py`).
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
- **Why a QR here and nowhere else.** The start's semantics IS the column order of the seeds. Gram–Schmidt bases are
  what the ragged convention produces, and what downstream coordinate seeds depend on (§2.4). The SVD-only rule of
  `docs/contributor/uniform_svd_prefix_orthogonalization.md` concerns orthogonalization sweeps, where the content
  decides the rank. A seeded completion is not one.

**Loop** (`ufv_aligned_padded_frame`).

- The ragged schedule (§2.3), as a host loop (D-20). `project` is called at block-layout frames.
- The library's `utv_project_ut3_onto_tangent_space` accepts gappy block-layout frames, verified at 5e-16
  (`P/check_uniform_loop.py`). It requires the hint's padded `N` to equal the frame's.
- **Not verified:** the uniform kinds' `precompute` / `transpose` at block-layout frames (the uniform
  `padded_start` path; §10, T14(e)).
- **At exit, compaction** (`ufv_compact_frame`):
  - gather real slots first on every rank axis: `argsort(~mask, stable)`, one `take_along_axis` per axis, with the
    same index array for every core axis an edge's mask governs;
  - zero the junk;
  - return the masks `ufv_make_frame_masks` builds from the integer stores at `level_widths`.

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
- `aligned_padded_frame(x0, x_old, project, hint, oversampling, oversampled_passes, max_passes, tol, alternate, seed)`
  runs the layer's loop.
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
- Its doctest shows the route for a frontend user who wants a problem's aligned frame without running Newton:
  `optimizers.least_squares_problem(...)` → `.problem`, `.x0_cores` → `padded_start(problem, x0_cores,
  x_old.data)` → rewrap the frame (`T3Frame(*frame)` / `UT3Frame`).

**`Problem.local_model(x, …, frame=None)`.** With `frame`:

- The structural check is `self.geom.require_frame_structure(x, frame)` (D-37). A geometry without the member
  raises `ValueError` naming it.
- Then it linearizes at `frame` instead of `geom.frame(x)`.
- Safe-mode numerical checks (orthonormality, `frame` reconstructs `x`) are the frontend's (house rule).

**`newton_cg`.**

- `padded_from=x_old` → `start_frame = padded_start(problem, x0, x_old, padded_*)[0]`.
- `start_frame=F` → the iteration-0 model is `problem.local_model(x0, frame=F, …)`; a rejected trust-region trial
  keeps it, as today.
- `'cone'` → `_resolve_g0_reference`:
  - runs `padded_start(problem, x0, None, cone_*)`;
  - reports `sqrt(inner(g, g))` with `g = _misfit_gradient_at(F, (problem, r0))` and `inner` from
    `problem.geom.with_frame(F)`;
  - it is not computed when `g0norm_newton` is given (as `'zero'`). `g0_reference='cone'` itself stays legal then,
    as `'zero'` does; an explicitly set `cone_*` option is a structural error (D-33);
  - an explicit `'cone'` whose value is 0 raises (as explicit `'zero'`).
- **Stats** (D-27). With no new keyword, the whole `stats` dict is bit-identical (the five keys pinned at
  `tests/backend/test_optimizers.py:477, :860`). Otherwise:
  - `stats['start_frame'] = 'given' | 'padded'`;
  - `stats['start_movements']` (padded only; a tuple of floats);
  - `stats['cone_movements']` (`'cone'` only);
  - history rows carry `g0_reference='cone'` (an existing field, a new value, only when requested).
- `NewtonInfo` is unchanged. Iteration 0's `gnorm` is the start's frame-gradient norm.

**Compile accounting per level (host loops, uniform jax; all fixed, none repeated).** Not measured in the library;
T13 counts it.

- Step-kernel programs:
  - `'ULRO'` at the working structure;
  - `'ULRO'` truncating;
  - `'UL'` and `'RO'` at the target structure;
  - the same four again for `'cone'`, whose A-block widths differ (`Na = 0`, `Rb = 1`).
- Gradient programs (eager kind/geometry code) at the working and target block widths, and again for `'cone'`.
- The start and the compaction: eager, no scan.
- The compacted start frame compiles nothing new for the first Newton step. Gradient-based loops are out of scope
  (D-2).

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

**One rule, applied first.** Every twin and the frontend `newton_cg(padded_from=)` apply the §3.5 test. Any
STRUCTURALLY-ZERO `x_old` (any Tucker rank or interior TT rank 0, e.g. Tucker `(0,2,2)`, TT `(1,0,2,1)`) is
converted to the canonical rank-0 tensor of the same shape, stack and backend (uniform: and `N`) BEFORE anything
else. Then:

- the reduction to minimal ranks is skipped, because `compute_minimal_ranks` maps the last boundary rank to 0 at
  rank 0 (checked);
- the frame twins use the empty frame directly, because `ut3_orthogonal_representations` raises at rank 0
  (checked);
- `newton_cg` passes the canonical rank-0 data to the backend, whose members apply the same test.

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

The S2 mathematics, unchanged, with the integer signature:

- Tucker rows: `pinned_completion(U°ᵀ, Haar seed rowsᵀ, k¹_U)` per group (`RandomState([seed, min(g)])`).
- TT and down cores: the outside coordinate directions in unfolding order, placed directly; then
  `pinned_completion(old block, inside coordinate seeds, remaining)` on the old block's sub-grid.
- Carriers: the old ones, zero-padded.

Seeds and coordinate orders are host bookkeeping (`_haar_seed_rows`, `_coordinate_seed_order`, private,
module-level), shared with the uniform start.

### 4.3 Ragged loop `fv_aligned_padded_frame`

§2.3 verbatim, as a host loop.

1. `working = ranks.oversampled_ranks(shape, target, p, sharing)`.
2. The start at the working ranks.
3. Every pass: one call of `project`, then one step. Old ranks come from the old frame's `(up, left)` widths.
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
- **Jit boundary.** All of the above runs inside `_utv_align_kernel` (D-34).

### 4.5 Uniform start `ufv_padded_frame`, compaction `ufv_compact_frame`, empty frame `ufv_empty_frame`

**The start:** `P/uniform_start.uniform_start(method='gs')`.

- `_old_block_layout` pads the old frame to `(Na, Rb)` and masks it.
- The seeds as in §3.4.
- Tucker: `pad_safe_pinned_completion(U°_firstᵀ, A-mask, Haar seeds, prefix(kU_first), shape mask)` batched over
  `(G,)+C`, scattered by `group_of`.
- TT / down:
  - outside units placed directly;
  - inside: `pad_safe_pinned_completion` on the A-block sub-grid, with `seed_mask` selecting columns `t .. t+k_in−1`
    (the prototype's `offset=t`, now a mask).

**The compaction:** `P/uniform_loop.compact`. Structural: every real count ≤ its level width.

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

1. The working ranks per stack element (`ranks.oversampled_ranks` on `(d,)+C` arrays).
2. The start at the working counts.
3. Every pass: `project(frame_data, hint)` at the block-layout frame, then `utv_align_padded_frame`.
4. `ufv_compact_frame(·, level_widths)` at exit. When there is nothing to complete (target stores = old stores), the
   loop skips steps 2–3 and returns `ufv_compact_frame(old_frame_data, level_widths)`.
5. Returns `(frame_data, movements)`, movements as a tuple (each `C`).

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
        seed:          int = ZERO_FRAME_SEED,              # the keyed Haar stream of the Tucker completions
) -> typ.Tuple[                 # (nU, nD, rL, rR) = compute_orthogonal_representation_ranks(shape, tucker_ranks, tt_ranks)
    typ.Tuple[NDArray, ...],    # up.    len=d, elm_shape=C+(nUi, Ni): the old rows bitwise, then the Haar completion
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
        max_passes:          int   = ALIGN_MAX_PASSES,           # passes in all, the truncating pass included; 0 only when working = target
        tol:                 float = ALIGN_TOL,                  # early exit on the movement
        alternate:           bool  = ALIGN_ALTERNATE,            # two-colour passes after the truncating pass
        seed:                int   = ZERO_FRAME_SEED,            # the start's Haar stream
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
        seed:            int = ZERO_FRAME_SEED,                       # the keyed Haar stream
) -> typ.Tuple[
    UFrameData,            # block layout: [A-block | P-block] per slot, old frame as given (junk zeroed), completions real-first
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
        level_widths:  typ.Tuple[int, int, int, int],  # (WU, WD, WL, WR), each >= every real count of its store
) -> UFrameData:                                       # junk-last, junk zeroed; masks = ufv_make_frame_masks of the counts at level_widths

def ufv_aligned_padded_frame(
        old_frame_data:      UFrameData,                              # the old frame, stack C (ufv_empty_frame for the zero tensor)
        tucker_ranks:        typ.Union[typ.Sequence[int], NDArray],   # len=d or HOST int (d,)+C, target
        tt_ranks:            typ.Union[typ.Sequence[int], NDArray],   # len=d+1 or HOST int (d+1,)+C, target
        hint:                typ.Any,                                 # opaque: handed to `project` unread
        project:             typ.Callable,                            # project(frame_data, hint) -> UVariationsData (or the bare (V, H) pair)
        level_widths:        typ.Tuple[int, int, int, int],           # (WU, WD, WL, WR): the compaction widths (the level's frame)

        sharing:             typ.Optional[typ.Sequence] = None,       # len=d group labels; None = unshared
        oversampling:        int   = ALIGN_OVERSAMPLING,              # as fv_aligned_padded_frame (constants from padded_frames)
        oversampled_passes:  int   = ALIGN_OVERSAMPLED_PASSES,
        max_passes:          int   = ALIGN_MAX_PASSES,
        tol:                 float = ALIGN_TOL,
        alternate:           bool  = ALIGN_ALTERNATE,
        seed:                int   = ZERO_FRAME_SEED,
) -> typ.Tuple[
    UFrameData,                # the aligned frame compacted to level_widths, prefix masks
    typ.Tuple[NDArray, ...],   # movements, one per pass, each shape=C
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
    ) -> typ.Tuple[
        typ.Tuple,                 # the aligned frame of x0 (uniform: at the widths of self.var_masks, prefix masks)
        typ.Tuple[NDArray, ...],   # movements, one per pass
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
) -> typ.Tuple[
    typ.Any,                   # the aligned frame of x0 (ragged frame / uniform frame .data at the level's widths)
    typ.Tuple[NDArray, ...],   # movements, one per pass
]:

def newton_cg(
        problem, x0, ..., callback=None,                       # unchanged
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
) -> typ.Tuple[
    'UT3Frame',                # the aligned frame at level_widths, prefix masks (the level's frame structure for that padding)
    typ.Tuple[NDArray, ...],   # movements, one per pass, each shape=C
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
- `tests/backend/_uniform_padded_scatter.py`: `P/uniform_testing.py`'s scatter. It puts a ragged case into block
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
| T4 | same (N3) | Two-colour = transport of the other stores (vs `select_pass`). The cycle regression on `tests/data/padded_frame_cycle_case.npz` (`S`, `X_old`'s cores, target ranks, seed; from `g06c_cycle_case.npz`): all-stores cycles (movement > 0.05 at pass 200); alternating converges (movement < 1e-8) | ≤ 1e-9 per pass; as stated |
| T5 | same | `fv_padded_frame` vs S2 outputs frozen in N1 before deletion (`tests/data/padded_frame_s2_reference.npz`, the S2 tests' cases); vs `fv_zero_point_frame` at the empty frame (unstacked); convention properties | ≤ 1e-14; TT/down equal (placed 0/1), Tucker ≤ 1e-14; outside seeds placed exactly |
| T6 | same | `fv_aligned_padded_frame` vs the dense loop with the same schedule (two-colour via `select_pass`; p = 0 and p = 5; 40 passes). Edge cases: working = target (no oversampled phase); `oversampled_passes ≥ max_passes` (the last pass truncates); `max_passes = 0` (start; raises when working ≠ target); target stores = old stores (no pass, empty movements, the old frame); the stop rule incl. the truncating-pass pair | ≤ 1e-9 per pass; pass counts as stated; movements a tuple of `C` arrays |
| T7 | `tests/backend/test_linalg.py` | `pad_safe_pinned_completion` vs `pinned_completion` on the real block; interior pads in rows, pins and seeds; per-element `seed_mask` in a stack; offset masks (columns `t..t+k−1`); count rejection | ≤ 1e-14 on the real block (same sign rule); junk rows and unselected columns exactly 0 |
| T8 | `tests/backend/test_uniform_padded_frame.py` | `ufv_padded_frame` vs `fv_padded_frame` through the scatter (8 cases incl. the empty frame, four old widths, groups). Exact output masks vs the independently derived `[A-mask as given | prefix(counts)]`. `ufv_empty_frame` = the scatter of `fv_empty_frame`. jit vs numpy | ≤ 1e-12; masks exact; junk exactly 0; jit ≤ 1e-12 |
| T9 | same | `utv_align_padded_frame` vs `tv_align_padded_frame` (the T1 list incl. d = 1; two-colour; truncating). Exact output masks. Garbage robustness (clean vs dirty) | ≤ 1e-9; masks exact; clean == dirty bitwise (equal layouts) |
| T10 | same | Stacks: C = (3,) with different ranks per element, C = (2, 2); stacked = per element (step and start in N5, loop in N6) | ≤ 1e-14 |
| T11 | same | `ufv_compact_frame`: masks; real entries moved; junk zero; `utv_retract` of the zero tangent at the level returns X_old; `from_frame` variation masks == `from_point`'s | masks exact; gathers bitwise; ≤ 1e-12 |
| T12 | same | `ufv_aligned_padded_frame` vs `fv_aligned_padded_frame` (defaults; tol 1e-10; the empty frame; non-minimal; p = 0; d = 2; alternate on/off), generic hints with movements away from `tol`. The no-op level (target stores = old stores) returns `ufv_compact_frame(old_frame_data, level_widths)`: prefix masks at `level_widths`, empty movements | same pass counts; movements ≤ 1e-12; spans ≤ 1e-9; the no-op's masks exact |
| T13 | same + `tests/test_dispatch.py` | **Compile count:** an eager-jax uniform loop, d ≥ 3, ≥ 6 passes at fixed structure. After the first pass of each step variant, the step kernel's jit cache (`_cache_size()`, as the mask-pattern test) does not grow; a second loop at the same structure adds nothing. Jitted step and compaction equal numpy. First-call time at d = 6 and 24 (informational) | cache sizes as stated; ≤ 1e-12 |
| T14 | `tests/backend/test_padded_start.py` | **Optimizer seam:** (a) `require_frame_structure` (through `local_model(frame=)` and the frontend `start_frame`) rejects, one family at a time with every other byte identical: each ragged store's shape; each uniform variation-mask family (up, down, left, right); each boundary entry (`left[d]`, `right[0]`); `shape`; a padded width. Plus a mutation check per family (deleting that comparison in the member turns the test red). It accepts the aligned frame and `geom.frame(x)`. A geometry without the member raises. (b) `newton_cg(padded_from=)` ≡ `newton_cg(start_frame=padded_start(...)[0])`. (c) G-bit: `set(stats)` and history keys unchanged without new keywords; the new keys only with them; a before/after equality of a seeded run recorded in the N7 handoff. (d) `'cone'`: deterministic given the seed; `cone_oversampling=0, cone_max_passes=0` equals `'zero'`; independent of `x0`'s values; corewise / no member / vanishing explicit / options without their group / `cone_*` with `g0norm_newton` raise; `'cone'` with `g0norm_newton` alone skips the loop; `gradient_descent` too. (e) **not verified before this test:** `padded_start` uniform vs ragged on `probe_derivatives`, d ≥ 3, by gauge-invariant quantities (per-pass `‖P_F g‖`, spans). (f) **not verified before this test:** a shared problem on both layers, one pass vs a dense tied-projection reference. (g1) **not verified before this test:** at a numerically full-rank `x` with a gauge-rotated frame `F` of `x` (same structure, accepted by the member), `local_model(x, frame=F, hessian='exact')` passes the `Problem.local_model` doctest's FD criterion along the retraction from `F`, and its quadratic of a matched direction equals the default frame's. (g2) **not verified before this test:** at the zero-padded `x0`, `newton_cg(hessian='exact', start_frame=F)` is finite and decreases the objective (FD is invalid there: the retraction is not first-order accurate at a numerically rank-deficient point). (h) `newton_cg(padded_from=<rank-0 tensor>)` and a partially-zero-rank `padded_from` (Tucker `(0,2,2)`, TT `(1,0,2,1)`), on both layers, through the frontend AND directly through the backend members. Both give the empty frame's start (equal to the canonical rank-0 case, ≤ 1e-14); a stack mixing zero and nonzero elements raises in the members | as stated; (d) `'cone'` = `'zero'` ≤ 1e-12 relative; (e) ≤ 1e-9 at tol 1e-10, ≤ 0.2 at the default stop; (f) ≤ 1e-9; (g1) ≤ 1e-4 relative to FD, matched quadratics ≤ 1e-10 relative; (g2) as stated |
| T15 | frontend files: `test_frame_variations_format.py`, `test_uniform_frame_variations_format.py`, `test_manifold.py`, `test_uniform_manifold.py`, `test_tucker_tensor_train.py`, `test_uniform_tucker_tensor_train.py`, `test_optimizers_frontend.py` | Twins: each equals its backend call (no added arithmetic); hint-class dispatch; the `N` check; cross-layer hints; the rank-0 spelling and the conversion of a partially-zero-rank `x_old` to it; every §3.6 validation message; safe-mode checks (ORTH, TIED, same tensor) raise and are skipped under `safety.unsafe()`; doctests | array_equal; as stated |
| T16 | `tests/test_dispatch.py` | Rows for every new backend function (numpy/jax in → out, ragged/uniform, stacked; uniform masks stay host) | all rows |
| T17 | `tests/backend/test_uniform_padded_frame.py` | `ufv_frame_widths(N, n, r)` == the widths of `ut3_orthogonal_representations` over a sweep (n > N, r² < n, d = 1..4) | exact |
| T18 | `tests/test_import_graph.py` (new; J-5) | The graph is built on module FILES, not packages, so `optimizer_display`'s `from t3toolbox import corewise` does not route through `t3toolbox/__init__`. The backend import graph is acyclic; `backend/optimizers.py` imports only `common`, `regularization` and `corewise` | exact |
| T19 | `tests/backend/test_padded_start.py` | `ranks.oversampled_ranks` on `(d,)+C` arrays = the per-element calls; sequences unchanged | exact |

## 7. Docs and CHANGELOG

**New pages.**

- `docs/aligned_padded_frame.md` (user doc) covers:
  - what the aligned start is, and when to use `padded_from` and `'cone'` (`gtol_rel=1e-4` with `'cone'`, D-24);
  - the schedule and its knobs;
  - what the measurements showed (§2.3, with the simultaneous / two-colour distinction);
  - the zero point as a rank-0 tensor;
  - the uniform block layout as a user sees it;
  - one sentence: `g0_reference='warm_start'` with a start frame is `‖g‖` at that frame;
  - one sentence on the regularizer (§3.5).
- `docs/contributor/padded_frame_internals.md` covers:
  - the message passing in library terms;
  - the block layout, the empty frame, the compaction and why;
  - the pad-safe pinned completion and why a QR;
  - the jit boundary and the compile accounting;
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
| N1 | **Cleanup and the ragged start.** Freeze S2's outputs on T5's cases into `tests/data/padded_frame_s2_reference.npz` FIRST. Then: `fv_operations.py` back to `efe5e3d`; delete the S2/S3 tests and dispatch rows (keep S1's T10 row); remove S3's ten HISTORICAL strings (keep S1's two); refresh `test_padded_start.py`'s docstring; discard the S4 working tree. Create `backend/padded_frames.py` with `fv_empty_frame` and `fv_padded_frame`; widen `ranks.oversampled_ranks`; add `tests/test_import_graph.py` | `backend/fv_operations.py`, `backend/padded_frames.py`, `backend/ranks.py`, `tests/backend/test_padded_frame.py`, `tests/backend/test_padded_start.py`, `tests/test_dispatch.py`, `tests/test_contractions_interpreter.py`, `tests/test_import_graph.py`, `tests/data/` | full suite green; `git diff efe5e3d -- t3toolbox/backend/fv_operations.py` empty; T5, T16 rows, T18, T19 |
| N2 | `tv_align_padded_frame`; the Appendix C ragged strings in HISTORICAL; the dense reference module | `backend/padded_frames.py`, `tests/backend/_padded_frame_oracles.py`, `tests/backend/test_padded_frame.py`, `tests/test_contractions_interpreter.py` | T1–T3; the contraction suite |
| N3 | `fv_aligned_padded_frame`; the cycle case. **Review stop** | `backend/padded_frames.py`, `tests/backend/test_padded_frame.py`, `tests/data/padded_frame_cycle_case.npz` | T4, T6 |
| N4 | `pad_safe_pinned_completion`, `ufv_frame_widths`, `ufv_empty_frame`, `ufv_padded_frame`; the scatter helper | `backend/linalg.py`, `backend/ufv_masking.py`, `backend/uniform_padded_frames.py`, `tests/backend/test_linalg.py`, `tests/backend/_uniform_padded_scatter.py`, `tests/backend/test_uniform_padded_frame.py` | T7, T8, T17 |
| N5 | `utv_align_padded_frame` (block layout, the kernel and its jit boundary, scans, `pad_safe_svd`, groups, stacks, `update`); the uniform strings | `backend/uniform_padded_frames.py`, `tests/backend/test_uniform_padded_frame.py`, `tests/test_contractions_interpreter.py` | T9, T10 (step, start) |
| N6 | `ufv_compact_frame`, `ufv_aligned_padded_frame`. **Review stop** | same, `tests/test_dispatch.py` | T10 (loop), T11–T13, T16 rows |
| N7 | The seam: geometry members; `Problem.local_model(frame=)`; `_misfit_gradient_at`; `padded_start`; the `newton_cg` / `gradient_descent` keywords; `'cone'` | `backend/geometry.py`, `backend/optimizers.py`, `tests/backend/test_padded_start.py`, `tests/backend/test_optimizers.py`, `tests/test_dispatch.py` | T14, T16 rows, T18, G-bit |
| N8 | Frontend twins and docstring moves. **Review stop** | `tucker_tensor_train.py`, `uniform_tucker_tensor_train.py`, `frame_variations_format.py`, `uniform_frame_variations_format.py`, `manifold.py`, `uniform_manifold.py`, `optimizers.py`, and T15's files | T15, doctests |
| N9 | Docs, CHANGELOG, release notes, judgment calls, the math note | `docs/`, `CHANGELOG.md`, `dev/` | doctests in docs; `sphinx -W` |
| N10 | Acceptance (§11), with the SHIPPED default schedule. **Review stop** | T3Polynomial (its own process) | the table of §11 filled |

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
| D-20 | **RESOLVED: host loops.** `project` is an arbitrary callable, the stop branches on the movement on the host, and a whole-loop jit buys about 6 s of compile for a loop that runs once per level. The condition is the uniform step's jit boundary (D-34) | session; writer and critic r1; orchestrator r1 |
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
| D-34 | The uniform step's jit boundary: one module-level kernel; masks and shape via `partition_static`; every host-derived mask precomputed as host bool and the group tables as tuples of Python ints (never integer numpy arrays, which would be traced); split and `update` static arguments; the module's own `lru_cache`d jit getter; module-level, closure-free scan bodies | orchestrator r1 (critic M4) |
| D-35 | The frontend loop: one twin per layer beside its frame class (`frame_variations_format.fv_aligned_padded_frame`, `uniform_frame_variations_format.ufv_aligned_padded_frame`), each dispatching on the hint's class. Reasons: `uniform_manifold` imports `manifold` (no cycle-free cross-layer home without a new module); D-15; the house's per-layer frontend free-function pairs (`t3_` / `ut3_orthogonal_representations`). The frontend `padded_start` twin is dropped: it is not in D-8, and `newton_cg(padded_from=)` is the user path. The one case it would serve, a problem's aligned frame without running Newton, goes through the PUBLIC backend `padded_start` (doctest: the `least_squares_problem` route). Flagged to Nick | writer r1; critic r2 agrees; correction r2 |
| D-36 | The loop cap counts the truncating pass; `max_passes = 0` is legal only when working = target; movements are a tuple of `C` arrays in both layers | writer r1 (critic MINOR 13) |
| D-37 | `Problem.local_model(frame=)` (and the frontend `start_frame`) checks structure through the member `require_frame_structure(x_cores, frame)`: integers and held masks only, never an orthogonalization. Ragged: the frame's core shapes against `compute_orthogonal_representation_ranks` of `x`'s ranks, plus `C` and `N`. Uniform: the frame's variation masks against `var_masks`, the boundary entries, `shape`, the widths. The r1 text (compare with `geom.frame(x)`) is withdrawn: it orthogonalized to read structure, the pattern the arc was paused for | orchestrator r2 (critic r2 M1) |

## 10. Risks and open questions

- **Convergence is slow and seed-dependent on real problems.** The loop is a start-frame heuristic, not a solver;
  the defaults are a budget. Whether more passes buy Newton iterations is being measured (§11), with the
  simultaneous step.
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
- **GPU** has not been measured anywhere.
- **Large d.** Near-tied spectra converge slowly; the cap applies.

## 11. Downstream and acceptance

- T3Polynomial adopts `padded_from` and `g0_reference='cone'` in its ladder (its own process; `t3polynomial/cone.py`
  retired).
- Laptop: the Darcy o01 ladder, levels 1 → 6 and 6 → 11.
  - Starts: library, hint-free, and aligned with the SHIPPED default schedule (two-colour) at 4 / 6 / 10 passes.
    Also aligned without oversampling, and aligned at 40 passes.
  - Measure: Newton and CG counts, objective, test error; two seeds; compile logging and peak memory on one level.
  - The g06 Newton numbers (simultaneous step) go beside them when they arrive.
- Runpod: one manual 20×1000 pair (library vs aligned), decoupled from the release.

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
- [ ] The uniform step runs through its module-level jitted kernel with explicit statics.
- [ ] Structural counts and checks only; no rank tolerance; degenerate content left arbitrary.
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
| `uniform_testing.py` | `tests/backend/_uniform_padded_scatter.py` |
| `g06c_cycle_case.npz` | `tests/data/padded_frame_cycle_case.npz` |

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
