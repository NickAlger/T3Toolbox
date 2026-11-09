# Aligned padded frame — implementation plan (fresh plan, 2026-11-09)

> **Status: DRAFT 0, written by the orchestrating session from the research prototypes and the maintainer's
> decisions.** Next: a writer/critic pair (Opus 5.5, extra-high effort) iterates on this document until they agree;
> then Nick and the session review; then implementation, slice by slice, with an implementer/reviewer pair.
> Every one of those agents first reads a handful of `TuckerTensorTrain` methods and the backend functions they
> trace to (`docs/contributor/signature_style.md`, `naming_rules.md`); those modules and `UniformTuckerTensorTrain`
> are the only allowable precedent for code shape.
>
> **Supersedes** `dev/archive/padded_frame_cone_impl_plan_2026-11-02_superseded.md` (and its reviews r1–r8, archived
> beside it). What survives from it is listed in §9 and Appendix B; the mathematics is now Nick's note
> "Completing frames by truncating A + εS, with projection-only access" (https://claude.ai/artifact/EvfyxaZKGuKUL5MdC6UfjD),
> specialized to T3s.
>
> **Reference prototypes** (research repo, read-only for the library): `nicks_research_experiments/tensor/aligned_padded_frame/`
> — ragged step `core_form2.py` (+ `cycle/core_form3.py` for `update=`), ragged loop `ragged_loop.py`,
> uniform step `uniform_form2.py`, uniform start `uniform_start.py`, uniform loop + compaction `uniform_loop.py`,
> dense per-edge reference `dense_oracle.py`, the `README.md` with every measurement. Measurements on Darcy:
> `T3Polynomial/scripts/g06*.py`. These are references for the MATHEMATICS and the verification; they are NOT
> references for code shape (for loops, raw einsums, slicing).

## CHANGES

| round | by | what |
|---|---|---|
| 0 | session | first draft |

## 0. Why, in one page

At a rank increase in rank continuation the converged point `X_old` is zero-padded to the new ranks. Its frame at
the new ranks is the old frame (which must stay exactly as it is) plus a **completion**: new orthonormal directions
in every store that the data does not determine. `t3_orthogonal_representations` fills them with whatever LAPACK's
QR leaves, the 2026.3.0 zero-point frame fills them with a seeded Haar draw; neither looks at the data, and the first
Newton steps at the new level are spent discovering directions the gradient already knows.

This feature **aligns the completion with the gradient**: a fixed-point loop

    F  <-  align(F, project(F, hint)),

where `project(F, hint)` is the orthogonal projection of one fixed tensor `S` (the gradient at `X_old`, or any
tensor) onto the tangent space at the current padded frame `F`, and `align` replaces every completion by the
leading new directions of that tangent vector, old blocks held fixed. The loop runs a few passes with the completion
oversampled, then passes at the target ranks, updating the two edge directions on alternate passes. It is used once
per rank set, at the outset, for the Newton start (`newton_cg(..., padded_from=x_old)`) and for a data-aware reference
gradient norm at the zero tensor (`g0_reference='cone'`). Measured on the laptop Darcy ladder, the hint-free start
captures 8.4 of the gradient's 40.5 at level 1, the library's own padded frame 15.3, and the aligned start 36.4
after four passes.

What changed since the superseded plan: the hint is no longer embedded as a doubled-rank T3 and processed by a
dense-style construction (width `2R` factorizations, 22 s per step at `N = 64`, ranks 41 → 46); the step is the
note's core form on the frame and its variation cores (72 ms, same output to 1e-14). The uniform layer gets a
truly uniform twin in a **block layout**. The loop updates the two edge directions on alternate passes, because the
simultaneous update was found to cycle on Darcy. Everything the superseded plan decided about release policy,
style and the optimizer seam carries over (§9).

## 1. What ships (public surface)

Backend, ragged (`backend/padded_frames.py`, new):
- `fv_padded_frame(shape, tucker_ranks, tt_ranks, old_frame=None, groups=(), seed=...)` — the seeded hint-free padded frame (the START).
- `tv_align_padded_frame(frame, variations, old_tucker_ranks, old_tt_ranks, tucker_ranks=None, tt_ranks=None, groups=(), update='ULRO')` — ONE alignment step; returns `(frame, movement)`.
- `aligned_padded_frame(shape, tucker_ranks, tt_ranks, hint, project, old_frame=None, groups=(), ...)` — the LOOP; returns `(frame, movements)`.

Backend, uniform (`backend/uniform_padded_frames.py`, new; `backend/linalg.py` one primitive):
- `linalg.pad_safe_pinned_completion(pinned, pinned_mask, seeds, row_mask, k)` — the uniform twin of `pinned_completion`.
- `ufv_padded_frame(shape, tucker_ranks, tt_ranks, old_frame=None, groups=(), seed=..., ...)` — the start in block layout; returns `(frame_data, split)`.
- `utv_align_padded_frame(frame_data, variations_data, split, new_counts, out_widths=None, groups=(), update='ULRO')` — the step in block layout.
- `ufv_aligned_padded_frame(shape, tucker_ranks, tt_ranks, hint, project, out_widths, old_frame=None, groups=(), ...)` — the loop; returns the frame COMPACTED to `out_widths` (junk-last, prefix masks).
- `ufv_compact_frame(frame_data, widths)` — the compaction.

Optimizer layer (`backend/optimizers.py`, frontend `optimizers.py`):
- `Problem.local_model(..., frame=None)` — linearize at a supplied frame (checked, never trusted).
- `padded_start(problem, x0, x_old, ...)` — the driver: the aligned frame of the zero-padded point, with the problem's gradient as the projection.
- `newton_cg(..., start_frame=None, padded_from=None, padded_oversampling=5, padded_oversampled_passes=2, padded_max_passes=10, padded_tol=0.05, padded_seed=None, cone_oversampling=5, cone_oversampled_passes=2, cone_max_passes=10, cone_tol=0.05, cone_seed=None)`; `g0_reference='cone'` on `newton_cg` and `gradient_descent`.

Frontend twins (`frame_variations_format.py`, `uniform_frame_variations_format.py`, `manifold.py`, `uniform_manifold.py`, `tucker_tensor_train.py`, `uniform_tucker_tensor_train.py`, `optimizers.py`):
- `T3Frame.padded(...)`, `UT3Frame.padded(...)` (static) — the start.
- `T3Tangent.align_padded(...)`, `UT3Tangent.align_padded(...)` — the step on a tangent (frame + variations).
- `manifold.aligned_padded_frame(x_old, tucker_ranks, tt_ranks, hint, project=None, ...)` — the loop, dispatching on the class of `x_old` (ragged / uniform) and of `hint` (`TuckerTensorTrain`, `UniformTuckerTensorTrain`, dense array; anything else needs `project`).
- `TuckerTensorTrain.oversampled_ranks(p, sharing=None)`, `UniformTuckerTensorTrain.oversampled_ranks(p, sharing=None)`.
- frontend `optimizers.padded_start`, and the new `newton_cg` / `gradient_descent` keywords, forwarded only when set.

Release: **2026.4.0, additive and non-breaking** — every default unchanged (`g0_reference` stays `'zero'`,
`fv_zero_point_frame` untouched), every new keyword appended keyword-only.

## 2. Mathematics (the note, specialized to T3s)

### 2.1 Frames, blocks, directed edges

A T3 is a tree: Tucker leaves `l_i` (physical size `N_i`) and TT cores `c_i` in a chain. One frame core per
DIRECTED edge is exactly the library's four stores:

| directed edge | store | core | output slot |
|---|---|---|---|
| `l_i -> c_i` | `up[i]`    | `U_i  C+(nU_i, N_i)` | the Tucker index |
| `c_i -> l_i` | `down[i]`  | `O_i  C+(rL_i, nD_i, rR_{i+1})` | the down index |
| `c_i -> c_{i+1}` | `left[i]` | `L_i  C+(rL_i, nU_i, rL_{i+1})` (`i <= d-2`; `L_{d-1}` is the carrier of the base point) | bond `i+1` |
| `c_i -> c_{i-1}` | `right[i]` | `R_i  C+(rR_i, nU_i, rR_{i+1})` (`i >= 1`; `R_0` is the carrier) | bond `i` |

A **padded frame** of `X_old` at ranks `(nU, nD, rL, rR)` holds the old frame bitwise in the LEADING slots of every
store (the **A-block**, widths `nU°, nD°, rL°, rR°` = `ranks.compute_orthogonal_representation_ranks(shape, old
ranks)`; a frame's own `(up, left)` widths reproduce its stores, verified on 4000 rank sets) and a **completion**
`P` in the trailing slots (the **P-block**, widths `k = new − old` per directed edge). An old frame core is zero
unless every input slot is in its A-part. The two chains reconstruct `X_old` through the carriers, which hold the
old content only. Variations `(V, H)` at the frame are the library's gauged variations (`U_i V_i^T = 0`,
`L_i^T H_i = 0`); the step does not depend on the gauge (verified: ungauged variations give the same frames).

### 2.2 One alignment step (the note's §10, unrooted message passing)

Given a padded frame `F` and a tangent vector `Z` at `F` with variation cores `(V, H)`, replace every completion
by the leading left singular directions of `Z`'s normal content at that edge, with the old block pinned. Two
sweeps over directed edges:

**Step 1 — far-side messages, old frames** (`Phi`, `Xi` per directed edge `a -> b`; from the far leaves inward):
`Phi[a->b] = P_{a->b}^T Z_far U'_{b->a}` (`k x R`) and `Xi Xi^T = P_{a->b}^T Z_far (I − U'_{b->a} U'_{b->a}^T) Z_far^T P_{a->b}`
(a square root of a Gram matrix; any factor). `Z_far` is the part of `Z` on `b`'s side. Order: all `down` edges
(leaves have no other neighbour); the `left` edges from the far end (`c_j -> c_{j+1}` for `j = d-2 .. 0`, each from
`O_{j+1}`, `L_{j+1}` and the messages of the next edge); the `right` edges from the near end; all `up` edges.

**Step 2 — new completions, new frames** (in message order: all `up`; the `left` sweep `i = 0 .. d-2`; the `right`
sweep `i = d-1 .. 1`; all `down`). At `a -> b`, with the incoming slots of `a` already in their NEW coordinates
(the old block unchanged, the completion block multiplied by `o[y->a] = P⁺_{y->a}^T P_{y->a}`):

    g_old = the old completion P_{a->b} in the new coordinates           (N_a x k_ab)
    g_new = Z_near P_{b->a} = the node term of a on the far completion
            + sum over a's other neighbours z of (F[a->z] restricted) x_z D[z->a]
    Y     = [ g_old Phi_P + g_new | g_old Xi ]                          (N_a x (k_ba + m)),  Phi = [Phi_A | Phi_P]
    P⁺    = pinned_leading_columns(Ā, Y, k⁺)       the leading k⁺ left singular directions of (I − ĀĀ^T) Y, completed against Ā
    o[a->b] = P⁺^T g_old,    D[a->b] = [Ā | P⁺]^T g_new,    movement_e = ‖ g_old − P⁺ (P⁺^T g_old) ‖_F

`Y` is the edge's normal content exactly (the note's identity `Y = Q (L J) Θ^T`), so nothing wider than
`old + 2k` is factorized and the tangent is never embedded as a doubled-rank T3. The boundary bonds have old rank 1
and no completion, so the messages of the nonexistent edges past the ends are zero-size arrays and EVERY core runs
the same body: both sweeps are folds with an initial carry (scans). **Verified:** one step equals the dense per-edge
update to 3e-14 on all four stores (d = 2..5, uneven increments, non-minimal ranks, a tangent part 10x the normal
part, truncating steps, sharing groups, the zero old point, stacks); frames are orthonormal to 2e-15, the old block
bitwise untouched; float32 errors follow `u · max(1, s)`; the step is scale-invariant in the hint.

**Truncation.** `k⁺ < k` (oversampled → target) is the same step with fewer selected directions.

**Transport and two-colour.** A store that is NOT updated in a pass is **transported**: its content is `Y = g_old`
alone (the old completion re-expressed in the new candidate space). A pass that updates only `up` and `left`
(toward the root at `c_{d-1}`), followed by a pass that updates only `right` and `down`, is block coordinate ascent
with a projection between the halves. A pass may not both transport and truncate (a transported store would lose
directions arbitrarily) — the loop never does.

**Degenerate content.** When `Y` has fewer than `k⁺` nonzero singular values the trailing directions are the
primitive's Householder completion: valid, orthogonal to the old block, arbitrary (roundoff- and backend-dependent).
This is accepted (D-12): such directions carry no content and are dropped at truncation; a zero hint therefore
moves the completion (unlike the superseded dense form).

### 2.3 The loop and its facts

    F_0 = fv_padded_frame(shape, working ranks, old_frame, groups, seed)        working = oversampled_ranks(target, p)
    passes 1 .. m          : F <- step(F, project(F, hint))  at the working ranks           (all stores)
    pass  m+1              : F <- step(F, project(F, hint))  TRUNCATING to the target ranks  (all stores)
    passes m+2, m+3, ...   : alternate  update='UL', 'RO', ...  at the target ranks
    stop: movement (max over edges, and over the stack) below tol on the last two passes (one of each colour),
          or max_passes passes in all;  with alternate=False: movement below tol on the last pass.
    return F, movements

`project(frame, hint) -> variations` must be the orthogonal projection of ONE fixed tensor onto the tangent space
at `frame` (the library's `tv_project_dense_onto_tangent_space`, `tv_project_t3_onto_tangent_space`, or the
problem's gradient with the residual computed once). `hint` is handed to it unread.

Measured facts the defaults rest on (details in the prototype README and `T3Polynomial/scripts/g06*`):

| fact | where |
|---|---|
| The simultaneous update (all stores every pass) **cycles**: period 2 on Darcy level 1 and on a 243-number case; the stationary point lies above the cycle and is not an attractor of that map | `cycle/README.md` |
| Two-colour after the truncating pass converges monotonically (merit never decreased), no parameter; costs ≤ 0.5 % early on Darcy (35.9 vs 36.4 at pass 4); as fast as the simultaneous update on the toy family | same; `toy_schemes.py` |
| Inertia (proximal weight on the old completion) slows Darcy badly — rejected | `darcy_schemes.py` |
| On Darcy the movement falls below 0.05 only after 18–37 passes; the value at 8–10 passes is ~93 % of the 40-pass value; the long-run value is seed-dependent (36.1 / 40.5 / 41.7 at level 1) | `g06_aligned_frame.py` |
| Toy family: p = 5 for 2 passes then target passes reaches 97 % of the converged merit in 4 passes, 99.7 % in 6; oversampling alone plateaus at 88 % | `check_schedule.py` |
| At the zero tensor: 95 % (rank 6) / 99 % (rank 11) of the converged frame-gradient norm in 10 passes; converged values agree with the 2026-11-02 investigation (48.9 / 56.7) | `g06` |
| Per pass the step is ≤ 10 % of a gradient (15 / 105 ms at level 1, 24–55 / 500–800 ms at level 2) | `g06` |
| The movement measure tracks the true frame movement within 15 % from 1 down to 1e-15; the shortcut `sqrt(k − ‖o‖²)` floors at 1e-8 | `check_v2.py` |
| At d = 24 with a near-tied hint spectrum the defaults do not converge in 10 passes | uniform README |

So: the pass cap is the effective stop; the tolerance is an early exit; the `'cone'` reference is "the value after the
default schedule", deterministic given the seed (D-7).

### 2.4 The start (the committed convention, both layers)

Tucker factors: the keyed Haar stream of `fv_zero_point_frame` (`RandomState([seed, min(group)])`), the first `k`
draw rows as seeds, made orthonormal against the old factor (`pinned_completion`). TT and down cores: the
coordinate directions of the slot's unfolding OUTSIDE the old block in the core's row-major order, then the
Householder completion INSIDE the old block seeded by the block's own coordinates in index order. At the zero
tensor this is exactly `fv_zero_point_frame`'s frame (Haar Tucker factors, identity-prefix TT cores), so the cone
reference and the `'zero'` reference share their start. The convention is **basis-dependent** (a TT seed names a
coordinate of its neighbours' completion slots), which is why the uniform twin must reproduce the same Gram–Schmidt
bases, not just the spans (§3.4; verified ≤ 3.4e-16 across layers).

### 2.5 The reference at the zero tensor

`g0_reference='cone'` runs the loop at the zero tensor of `x0`'s ranks (`old_frame=None`) with the problem's
gradient, then reports the **frame-gradient norm** `‖P_{T_F} G‖` at the returned frame (one more projection). It is
like-for-like with the gradient norms the stopping test compares it to (D-5). The default `'zero'` (one Haar frame)
is unchanged.

## 3. Design

### 3.1 The rules every slice is held to

1. **Backend functions are pure mathematical functions on immutable basic types**: arrays, tuples of arrays, ints,
   static int tuples, host bool masks. No object is passed to be read for its shapes; no dict of shapes; structure
   travels as `Sequence[int]` with `len=` shape comments; dispatch is inferred from the arrays. Gold standard:
   `TuckerTensorTrain`, `UniformTuckerTensorTrain` and the backend they call. Precedent outside them is not justification.
2. **Structural, not numerical**: counts from integers and masks; no rank tolerance, no epsilon content. The
   stopping tolerance and the oversampling are the user's tunables, not rank decisions.
3. **The uniform layer**: masks are the only statement of which slots are real; inputs may be gappy and junk may be
   garbage; never assume where junk sits in an input; a function may return its own layout (junk-last) with masks.
   Truly uniform: no per-mode arrays of different shapes, no Python loop over modes in the mathematics — maps are
   batched over the leading mode axis, sweeps are scans. Factorizations are `pad_safe_svd` (step) and the new
   `pad_safe_pinned_completion` (start only, §3.4).
4. **Library code shape**: scans (`xscan` on the ragged tuple, `lax.scan` on supercores), named contractions through
   `backend/contractions.py` with HISTORICAL entries (never raw einsums), shape annotations on every signature, one-line
   docstrings + shape comments in the backend, descriptions / doctests / validation in the frontend twins, sharding
   safety (the engine never reshapes across group sub-axes).
5. **`project` is a required argument in the backend**; `hint` is opaque. The frontend dispatches on classes.
6. **Compile budget** (Nick): recompiling after a rank change is fine; a small fixed number inside a loop is
   tolerable; repeated recompiles inside a loop never. The uniform loop returns its frame at the level's padded widths
   so the first Newton step compiles nothing new.

### 3.2 Modules and layering

- `backend/padded_frames.py` (new): the ragged start, step and loop. Imports downward only: `linalg`, `ranks`,
  `tv_operations` (nothing), `fv_operations` (`ZERO_FRAME_SEED`, `_haar_rows`, `_identity_prefix`), `contractions`.
  `ranks` still imports `sharing`, which imports `fv_operations` — the `ranks → sharing` dependence is a smell for
  the later audit arc (ledger), not touched here.
- `backend/uniform_padded_frames.py` (new): the uniform start, step, loop and compaction. Imports `linalg`,
  `ranks`, `ufv_masking`, `ut3_masking`, `contractions`, `padded_frames` (shared host bookkeeping only).
- `backend/linalg.py`: `+ pad_safe_pinned_completion` beside `pad_safe_svd`; `pinned_completion` and
  `pinned_leading_columns` (S1, shipped) unchanged.
- `backend/fv_operations.py`: returns to its pre-arc form (`efe5e3d`) — the S2/S3 `fv_padded_frame` and its helpers
  are **deleted** (never released; D-3); `fv_zero_point_frame` unchanged.
- `backend/optimizers.py`: `Problem.local_model(frame=)`, `padded_start`, `newton_cg` keywords, `'cone'` in
  `_resolve_g0_reference`; `backend/optimizer_display.py`: the start's gradient norm and pass count in the record.
- `backend/geometry.py`: no new member (the geometry twins are dropped, D-9); the uncommitted S4 member is discarded.
- Frontend: §5.6.

### 3.3 The ragged layer

Stores come from ranks: `stores(shape, tucker, tt) = compute_orthogonal_representation_ranks(...)`; `None, None`
for the old ranks means the zero tensor (empty stores `(0,)*d, (0,)*d, (1,0,..,0,1), (1,0,..,0,1)`). The old frame
is given as a frame (it is used: its cores are the pinned blocks); the frontend builds it from `x_old` reduced
structurally to minimal ranks (`compute_minimal_ranks`) — the backend requires `old_frame`'s stores to equal
`stores(shape, old ranks)` (structural check). Stacks: a frame stack `C` on every core; the movement has shape `C`;
the loop stops on its maximum over the stack (D-13). Sharing groups: one completion per group from the members'
contents side by side (`Y_g = [Y_{i_1} | Y_{i_2} | ...]`), factors stay tied bitwise.

### 3.4 The uniform layer: block layout

Every rank slot of a padded frame is `[A-block | P-block]`, each padded to ONE uniform width with its own mask:
Tucker slots `[Na | Ka]`, bond slots `[Rb | Kb]`, `Na = max(nU°, nD°)`, `Rb = max(rL°, rR°)` over edges and the
stack, `Ka = max kU, kD`, `Kb = max kL, kR`. The A-block is the old frame's supercores AS GIVEN (gappy masks allowed,
nothing moved; junk masked to zero once at entry); the P-block is appended, as under `+`. The ONLY positional fact
is the split `(Na, Rb)`, two static integers shared by all modes, so "restrict this slot to its completion part" is
one static slice of the whole supercore. Frame data is the library's `(U, O, L, R, shape, (mU, mD, mL, mR))` with
`nU = nD = Na + Ka`, `rL = rR = Rb + Kb`.

- **Step**: maps over modes are one batched contraction and one batched factorization (`pad_safe_svd` on
  `[Ā | Y / (4‖Y‖_F)]`, row mask = the outer product of the incoming slots' masks, column mask = `[A-mask | ones]`;
  its triplets come out pins first (σ = 1), content by singular value (σ ≤ 1/4), then real-supported completions,
  all mutually orthogonal, pads bitwise zero; the `k⁺` triplets after the pins are selected by a static one-hot
  at the pin count = mask sum); the sweeps are `lax.scan` with a shape-uniform body and zero carries at the ends;
  stacks ride as `C` after the mode axis; groups use host tables (`members (G, m)` padded, validity mask, `first`,
  `group_of`) and one factorization batched over groups; `update` is static (two-colour = two more compiled step
  programs); the truncating pass writes into narrower P-blocks (`out_widths`).
- **Start**: the same seeds as the ragged convention (Haar rows per group; coordinate seeds at the REAL rows of each
  unfolding in their ragged row-major order via a cumsum of the masks, outside first then inside), one batched
  `pad_safe_pinned_completion` per store (a map, no sweep). Reproduces the ragged start to 3.4e-16, bitwise equal
  under jit, junk exactly zero.
- **Loop**: the ragged schedule; `project` is called at block-layout frames (the library's
  `utv_project_ut3_onto_tangent_space` accepts gappy block-layout frames; it requires the hint's padded `N` to equal
  the frame's); under jax the whole loop can be one jitted program (`fori_loop` + `while_loop`; about 6 s to compile,
  flat in `d`) — the library's host loop is also acceptable (D-20). At exit **compaction**: `ufv_compact_frame`
  gathers real slots first on every rank axis (`argsort(~mask, stable)`, one `take_along_axis` per axis, the same
  index array for every core axis an edge's mask governs), zeroes junk, and returns prefix masks equal to
  `ufv_make_frame_masks` of the integer stores at the given `out_widths`. Reason: `utv_retract` pads its output to the
  FRAME's padded widths, and the optimizer's geometry holds the prefix masks of the level.
- **`pad_safe_pinned_completion`** (D-11): a Householder QR of `[pinned real columns | seeds | pinned junk columns]`
  with the real rows permuted to the leading pivot positions (static host permutations — the mechanism
  `pad_safe_svd` uses internally), sign-fixed like `pinned_completion`; the selected columns are bitwise zero on junk
  rows. Why a QR here and nowhere else: the start's semantics IS the column order of the seeds (Gram–Schmidt bases are
  what the ragged convention produces and what downstream coordinate seeds depend on); the seeds are independent of
  the old block by construction (outside coordinates) or deliberately completed in index order (inside). The SVD-only
  rule of `docs/contributor/uniform_svd_prefix_orthogonalization.md` concerns orthogonalization sweeps, where the
  content decides the rank; it does not apply to a seeded completion.
- **Minimal ranks**: the uniform optimizer requires a minimal-rank start; `padded_start` reduces `x_old` structurally
  as the ragged path does. Non-minimal inputs to the uniform functions are supported (verified) but the level's widths
  must then be given explicitly (`out_widths`).

### 3.5 The optimizer seam

- `Problem.local_model(x, frame=F)`: linearize at the supplied frame. Structural checks (a `ValueError` naming the fix):
  `F`'s stores equal the geometry's frame stores at `x`'s ranks (ragged: shapes; uniform: the masks equal the
  geometry's `var_masks`/frame masks — CHECK-AND-REJECT, D-10); safe mode: orthonormality and that `F` reconstructs `x`.
- `padded_start(problem, x0, x_old, ...)`: the residual at `x0` once; `project(frame, hint=None)` = the problem's
  gradient at `frame` (`geom_F.project(frame, kind.transpose(residual, sample, frame, kind.precompute(frame, sample)))`,
  where `geom_F` is the problem's geometry for the ragged layer and `UniformManifoldGeometryOps.from_frame(frame,
  plain masks of the frame's ranks)` for the uniform layer — one geometry per phase, both value-hashed); the old
  frame = `geom.frame(x_old reduced)`; runs `aligned_padded_frame` / `ufv_aligned_padded_frame` with the problem's
  groups; returns `PaddedStart(frame, gnorm_start, movements)`, `gnorm_start` the frame-gradient norm at the returned
  frame (one more projection, reused by the first local model when possible).
- `newton_cg`: `padded_from=x_old` → `start_frame = padded_start(...)`; `start_frame=F` → the iteration-0 model is
  `problem.local_model(x0, frame=F)`; `g0_reference='cone'` → `_resolve_g0_reference` runs the loop at the zero tensor
  of `x0`'s ranks and reports `‖P_{T_F} G‖` there; `stats['newton']` / `NewtonInfo` gain `start_frame` (`'padded' |
  'given' | None`), `start_gnorm`, `padded_passes`, `cone_passes`. Keyword-only, appended after the last parameter;
  forwarded by the frontend only when set.
- Compile accounting per level (host loops; all fixed, none repeated): the alignment's step and gradient programs at
  the oversampled structure and at the target structure (shared with the optimizer when old ranks are uniform);
  the compacted start frame compiles nothing new for the first Newton step. Gradient-based loops are out of scope (D-2).

### 3.6 Frontend twins (D-8, D-9)

Each twin validates with explicit messages, makes one or two backend calls, rewraps; it carries the description,
doctests and safe-mode checks; its backend function keeps a one-line docstring and shape comments. Backend functions
without a twin (`pad_safe_pinned_completion`, `ufv_compact_frame`, the step internals) keep their doctests.
Dropped: geometry members and twins; `frame=` on the six fitting factories (deferred; the seam is `newton_cg(start_frame=)`).

## 4. Algorithms, step by step (library form)

### 4.1 Ragged step `tv_align_padded_frame`

Inputs: frame `(U, O, L, R)` (stack `C` on every core), variations `(V, H)` at it, old stores `s°`, output stores
`s¹` (`<=` the frame's own, `>= s°`), groups, `update`. Current stores from the frame's shapes (the frame is an input).

1. **Down messages** (map over modes, batched contraction): `VP = V[i][..., nD°:, :]`; `PhiO[i] = <VP, U_i>` over the
   physical index (`C+(kD, nU)`); `XiO[i] = root(VP^T − U_i^T PhiO^T)` (`root(M)`: a factor `Xi` with `Xi Xi^T = M^T M`,
   from the thin SVD of `M`).
2. **Left messages** (`xscan`, reverse, over cores `j = d-1 .. 1`; carry `(PhiL, XiL)` of the edge `c_j -> c_{j+1}`,
   initial zero-size `C+(0, 1)`, `C+(0, 0)`): with `beta_O = O_j[rL°_j:, nD°_j:, :]`, `beta_L = L_j[rL°_j:, :, rL°_{j+1}:]`:
   `Vb = H_j[rL°_j:, :, :] + beta_O x_down PhiO[j] + beta_L x_out PhiL`; `rows = [unf(beta_O x_down XiO[j]), unf(beta_L x_out XiL)]`;
   `PhiL[j-1] = <Vb, R_j>` over (tucker, right); `XiL[j-1] = root([unf(Vb − PhiL[j-1] x R_j); rows])`.
3. **Right messages**: the mirror (`xscan` forward over `j = 0 .. d-2`, carry of `c_j -> c_{j-1}`).
4. **Up messages** (map): `Vb = H_i[:, nU°:, :] + L_i[:, nU°:, rL°_{i+1}:] x_out PhiL[i] + R_i[rR°_i:, nU°:, :] x_out PhiR[i]`;
   `PhiU[i] = <Vb, O_i>` over (left, right); `XiU[i] = root([...])`.
5. **Up edges** (map over groups): `Ā = U_i[:nU°]^T`, `g_old = U_i[nU°:]^T`, per member `g_new = V_i[nD°:]^T`,
   `Phi_P = PhiU[i][:, nD°:]`; `Y = [g_old Phi_P + g_new | g_old XiU]` (members side by side);
   `P⁺ = pinned_leading_columns(Ā, Y, kU⁺)` (or `Y = g_old` if `'U' not in update`); `U⁺ = [U°; P⁺^T]`;
   `oU = P⁺^T g_old`; `DU_i = U⁺ g_new_i`; movement.
6. **Left edges** (`xscan` forward `i = 0 .. d-2`, carry `(oL, DL)`, initial zero-size `C+(0,0)`, `C+(1,0)`):
   `NEW_L` / `NEW_T` = old block unchanged, completion block times `oL` / `oU_i`;
   `M = NEW_T(NEW_L(H_i[:, :, rR°_{i+1}:])) + NEW_L(O_i[:, nD°:, rR°_{i+1}:] x_down DU_i) + NEW_T(R_i[rR°_i:, :, rR°_{i+1}:] x_out DL)`;
   `g_new = unf_out(M)`, `g_old = unf_out(NEW_T(NEW_L(L_i[:, :, rL°_{i+1}:])))`, `Ā = unf_out(embed(L_i[:rL°, :nU°, :rL°_{i+1}]))`;
   `P⁺ = pinned_leading_columns(Ā, [g_old PhiL_P + g_new | g_old XiL], kL⁺)`; `L⁺_i = fold([Ā | P⁺])`;
   `oL = P⁺^T g_old`, `DL = [Ā | P⁺]^T g_new`.
7. **Right edges**: the mirror (`xscan` reverse `i = d-1 .. 1`).
8. **Down edges** (map): `M = NEW_R(NEW_L(H_i[:, nU°:, :])) + NEW_L(L_i[:, nU°:, rL°_{i+1}:] x_out DR_{i+1}) + NEW_R(R_i[rR°_i:, nU°:, :] x_out DL_{i-1})`;
   `Ā = unf_mid(embed(O_i[:rL°, :nD°, :rR°_{i+1}]))`; `P⁺ = pinned_leading_columns(Ā, [g_old PhiO_P + g_new | g_old XiO], kD⁺)`.
9. **Carriers**: `L⁺_{d-1} = embed(L_{d-1}[:rL°, :nU°, :])`, `R⁺_0 = embed(R_0[:, :nU°, :rR°_1])`.
10. **Movement** = max over edges (per stack element).

All contractions are named strings in `backend/contractions.py` (new HISTORICAL entries; the block letters `C`
for the stack, the slot letters per the engine's rules). The prototype's einsums (`core_form2.py`) are the
specification of each contraction; the implementer translates them one by one.

### 4.2 Ragged start `fv_padded_frame`

The S2 construction with its integer-structure signature (its mathematics is unchanged): Tucker rows =
`pinned_completion(U°^T, Haar seeds, kU)` per group; TT and down cores: the outside coordinate directions in
unfolding order, then `pinned_completion(old block, inside coordinate seeds, remaining)`; at the zero tensor the
identity prefix bitwise (`fv_zero_point_frame`).

### 4.3 Ragged loop `aligned_padded_frame`

§2.3 verbatim; `working = oversampled_ranks(shape, tucker, tt, p, sharing)`; every pass one call of `project`
then one step; the stop as stated. Host loop.

### 4.4 Uniform step `utv_align_padded_frame`

`uniform_form2.uniform_align2`, with the §3.4 layout; steps 1–10 as above with: slices at the static split; masks
multiplied once at entry; `root` per batched SVD; the factorization `pad_safe_svd([Ā | Y/(4‖Y‖_F)], row_mask, col_mask)`
and the one-hot selection; `NEW` as a block operation; groups via host tables; the carriers masked to their A-block
inputs; `out_widths` for the truncating pass.

### 4.5 Uniform start `ufv_padded_frame`, loop `ufv_aligned_padded_frame`, compaction `ufv_compact_frame`

`uniform_start.uniform_start(method='gs')`, `uniform_loop.uniform_aligned_padded_frame(alternate=True)` and
`uniform_loop.compact`, with the signatures of §5.

## 5. API (signatures in the house style; the twins' docstrings carry the descriptions)

### 5.1 `backend/padded_frames.py`

```python
Frame = typ.Tuple[typ.Sequence[NDArray], typ.Sequence[NDArray], typ.Sequence[NDArray], typ.Sequence[NDArray]]
#        up C+(nU_i, N_i)   down C+(rL_i, nD_i, rR_{i+1})   left C+(rL_i, nU_i, rL_{i+1})   right C+(rR_i, nU_i, rR_{i+1})
Variations = typ.Tuple[typ.Sequence[NDArray], typ.Sequence[NDArray]]   # tucker C+(nD_i, N_i), tt C+(rL_i, nU_i, rR_{i+1})

def fv_padded_frame(
        shape:         typ.Sequence[int],                # len=d, mode sizes (N0, ..., N(d-1))
        tucker_ranks:  typ.Sequence[int],                # len=d, target Tucker ranks
        tt_ranks:      typ.Sequence[int],                # len=d+1, target TT ranks (boundary 1)
        old_frame:     typ.Optional[Frame] = None,       # the orthonormal frame of X_old at ITS ranks (minimal); None = the zero tensor
        groups:        typ.Tuple[typ.Tuple[int, ...], ...] = (),   # sharing partition (canonical_groups); () = unshared
        seed:          int = ZERO_FRAME_SEED,            # the keyed Haar stream of the Tucker completions
        stack_shape:   typ.Tuple[int, ...] = (),         # C; read only when old_frame is None
        use_jax:       bool = False,                     # read only when old_frame is None
) -> Frame:                                              # the padded frame: old frame bitwise in the leading slots, seeded completions

def tv_align_padded_frame(
        frame:             Frame,                        # a padded frame of X_old (old blocks in the leading slots)
        variations:        Variations,                   # a tangent vector at `frame` (gauged or not)
        old_tucker_ranks:  typ.Optional[typ.Sequence[int]],   # len=d; None, None = the zero tensor
        old_tt_ranks:      typ.Optional[typ.Sequence[int]],   # len=d+1
        tucker_ranks:      typ.Optional[typ.Sequence[int]] = None,   # output ranks, <= the frame's; None = the frame's own
        tt_ranks:          typ.Optional[typ.Sequence[int]] = None,
        groups:            typ.Tuple[typ.Tuple[int, ...], ...] = (),
        update:            str = 'ULRO',                 # STATIC: the stores replaced from the tangent ('U' up, 'L' left, 'R' right, 'O' down); the rest transported
) -> typ.Tuple[
    Frame,          # the frame at the output ranks: old blocks untouched, completions aligned
    NDArray,        # movement, shape = C: max over edges of ‖g_old − P⁺ P⁺ᵀ g_old‖_F
]:

def aligned_padded_frame(
        shape:               typ.Sequence[int],          # len=d
        tucker_ranks:        typ.Sequence[int],          # len=d, target
        tt_ranks:            typ.Sequence[int],          # len=d+1, target
        hint:                typ.Any,                    # opaque: handed to `project` unread
        project:             typ.Callable,               # project(frame, hint) -> Variations: the projection of ONE fixed tensor at `frame`
        old_frame:           typ.Optional[Frame] = None, # None = the zero tensor
        groups:              typ.Tuple[typ.Tuple[int, ...], ...] = (),
        oversampling:        int = 5,                    # p: working ranks = oversampled_ranks(target, p)
        oversampled_passes:  int = 2,                    # passes at the working ranks before the truncating pass
        max_passes:          int = 10,                   # passes in all (the effective stop)
        tol:                 float = 0.05,               # early exit on the movement (two consecutive passes when alternating)
        alternate:           bool = True,                # two-colour passes after the truncating pass
        seed:                int = ZERO_FRAME_SEED,
        stack_shape:         typ.Tuple[int, ...] = (),   # read only when old_frame is None
        use_jax:             bool = False,               # read only when old_frame is None
) -> typ.Tuple[
    Frame,                      # the aligned padded frame at the target ranks
    typ.Tuple[NDArray, ...],    # the movement of every pass, each shape = C
]:
```

### 5.2 `backend/linalg.py`

```python
def pad_safe_pinned_completion(
        pinned:       NDArray,   # C+(n, s_pad): orthonormal real columns, junk columns anything
        pinned_mask:  NDArray,   # HOST bool, broadcastable to C+(s_pad,): True = real pinned column
        seeds:        NDArray,   # C+(n, m): seed columns, in the order they are to be completed
        row_mask:     NDArray,   # HOST bool, broadcastable to C+(n,): True = real row
        k:            int,       # columns returned (static); k <= real rows − real pins
) -> NDArray:                    # C+(n, k): Householder-completed seeds, orthonormal, orthogonal to the real pins, bitwise zero on junk rows;
                                 #   Gram–Schmidt order of the seeds (the ragged pinned_completion's bases), sign-fixed the same way
```

### 5.3 `backend/uniform_padded_frames.py`

```python
UFrame = typ.Tuple[NDArray, NDArray, NDArray, NDArray, typ.Tuple[int, ...], typ.Tuple[NDArray, NDArray, NDArray, NDArray]]
#   U (d,)+C+(nT, N)  O/L/R (d,)+C+(rT, nT, rT)  shape  (mU (d,)+C+(nT,), mD (d,)+C+(nT,), mL (d+1,)+C+(rT,), mR (d+1,)+C+(rT,)) HOST bool

def ufv_padded_frame(
        shape:         typ.Sequence[int],                          # len=d
        tucker_ranks:  typ.Union[typ.Sequence[int], NDArray],      # len=d, or HOST int (d,)+C (per stack element)
        tt_ranks:      typ.Union[typ.Sequence[int], NDArray],      # len=d+1, or HOST int (d+1,)+C
        old_frame:     typ.Optional[UFrame] = None,                # at its own widths, any mask pattern; None = the zero tensor
        groups:        typ.Tuple[typ.Tuple[int, ...], ...] = (),
        seed:          int = ZERO_FRAME_SEED,
        completion_widths: typ.Optional[typ.Tuple[int, int]] = None,   # (Ka, Kb); None = the maxima of the counts
        stack_shape:   typ.Tuple[int, ...] = (),                   # read only when old_frame is None
        N:             typ.Optional[int] = None,                   # padded mode size; read only when old_frame is None (default max(shape))
        use_jax:       bool = False,                               # read only when old_frame is None
) -> typ.Tuple[
    UFrame,                     # block layout: [A-block | P-block] per slot, old frame as given, completions real-first
    typ.Tuple[int, int],        # split (Na, Rb): the static A-block widths
]:

def utv_align_padded_frame(
        frame:        UFrame,                                      # block layout, any mask pattern in either block
        variations:   typ.Tuple[NDArray, NDArray, ...],            # V (d,)+C+(nT, N), H (d,)+C+(rT, nT, rT) at `frame`
        split:        typ.Tuple[int, int],                         # (Na, Rb), static
        new_counts:   typ.Tuple[NDArray, NDArray, NDArray, NDArray],   # HOST int: kU, kD (d,)+C; kL, kR (d+1,)+C — the output completion counts
        out_widths:   typ.Optional[typ.Tuple[int, int]] = None,    # (Ka1, Kb1) output P-block widths; None = the input's
        groups:       typ.Tuple[typ.Tuple[int, ...], ...] = (),
        update:       str = 'ULRO',                                # STATIC
) -> typ.Tuple[UFrame, NDArray]:                                   # (frame, movement (C))

def ufv_compact_frame(
        frame:   UFrame,
        widths:  typ.Tuple[int, int, int, int],   # (WU, WD, WL, WR): the target padded widths (the level's), >= every real count
) -> UFrame:                                      # junk-last on every rank axis (one take_along_axis per axis, the same index array for every
                                                  #   core axis an edge's mask governs), junk zeroed, masks = ufv_make_frame_masks of the counts

def ufv_aligned_padded_frame(
        shape, tucker_ranks, tt_ranks, hint, project,              # as the ragged loop; tucker/tt may be (d,)+C / (d+1,)+C
        out_widths:    typ.Tuple[int, int, int, int],              # the level's padded widths for the compaction
        old_frame:     typ.Optional[UFrame] = None,
        groups=(), oversampling=5, oversampled_passes=2, max_passes=10, tol=0.05, alternate=True, seed=ZERO_FRAME_SEED,
        stack_shape=(), N=None, use_jax=False,
) -> typ.Tuple[UFrame, NDArray]:                                   # (compacted frame with prefix masks, movements (max_passes,)+C, nan after the stop)
```

### 5.4 `backend/optimizers.py`

```python
class Problem:
    def local_model(self, x_cores, sample=None, data=None, hessian='gauss_newton', curvature_damping=None,
                    frame=None):            # NEW: linearize at this frame of x_cores (structurally checked against the geometry; safe mode: orthonormal, reconstructs x)

@dataclass(frozen=True)
class PaddedStart(ValueHashedFields):
    frame:       typ.Any       # the aligned padded frame (ragged Frame / uniform UFrame at the level's widths)
    gnorm:       float         # ‖P_{T_frame} g‖ at it
    movements:   typ.Tuple     # the loop's movements

def padded_start(
        problem:   'Problem',
        x0:        Cores,                    # the zero-padded point at the new ranks (minimal)
        x_old:     typ.Optional[Cores],      # the old point; None = the zero tensor (the cone frame)
        oversampling: int = 5, oversampled_passes: int = 2, max_passes: int = 10, tol: float = 0.05,
        alternate: bool = True, seed: typ.Optional[int] = None,
) -> PaddedStart

def newton_cg(problem, x0, ..., callback=None,
              start_frame=None,            # a frame of x0 for iteration 0 (checked)
              padded_from=None,            # the old point: start_frame = padded_start(problem, x0, padded_from, padded_*)
              padded_oversampling=5, padded_oversampled_passes=2, padded_max_passes=10, padded_tol=0.05, padded_seed=None,
              cone_oversampling=5, cone_oversampled_passes=2, cone_max_passes=10, cone_tol=0.05, cone_seed=None)
# g0_reference: 'zero' (default, unchanged) | 'warm_start' | 'cone' (NEW) | None
```

### 5.5 `backend/ranks.py` (shipped, S1): `oversampled_ranks(shape, tucker_ranks, tt_ranks, p, sharing=None)`.

### 5.6 Frontend twins

```python
T3Frame.padded(x_old: Optional[TuckerTensorTrain], tucker_ranks, tt_ranks, sharing=None, seed=...) -> T3Frame      # static
UT3Frame.padded(x_old: Optional[UniformTuckerTensorTrain], tucker_ranks, tt_ranks, sharing=None, seed=..., completion_widths=None) -> (UT3Frame, split)
T3Tangent.align_padded(self, old_tucker_ranks, old_tt_ranks, tucker_ranks=None, tt_ranks=None, sharing=None, update='ULRO') -> (T3Frame, movement)
UT3Tangent.align_padded(self, split, new_counts, out_widths=None, sharing=None, update='ULRO') -> (UT3Frame, movement)
manifold.aligned_padded_frame(x_old, tucker_ranks, tt_ranks, hint, project=None, sharing=None, oversampling=5,
                              oversampled_passes=2, max_passes=10, tol=0.05, alternate=True, seed=...) -> (T3Frame | UT3Frame, movements)
TuckerTensorTrain.oversampled_ranks(self, p, sharing=None) -> (tucker, tt);  UniformTuckerTensorTrain.oversampled_ranks(self, p, sharing=None)
optimizers.padded_start(problem, x0, x_old, ...) -> PaddedStart (frontend types);  optimizers.newton_cg(..., the new keywords)
```

`manifold.aligned_padded_frame` picks `project` from the hint's class when `project is None`:
`TuckerTensorTrain` → `tv_project_t3_onto_tangent_space`, `UniformTuckerTensorTrain` → `utv_project_ut3_onto_tangent_space`,
`numpy.ndarray`/jax array → `tv_project_dense_onto_tangent_space` (ragged `x_old` only); any other hint without
`project` is a structural error naming the fix. `x_old=None` needs `shape`-bearing arguments (a `zeros` tensor is the
simplest spelling: the twin accepts a `TuckerTensorTrain` of zeros and detects the zero case structurally? — NO:
it takes `x_old=None` plus `shape=`; decide in review, D-21).

## 6. Tests

A dense per-edge reference (port of `dense_oracle.py`: `dense_frames`, `dense_pass`, `tangent_basis`, `dense_project`,
`piece_basis`, `distances`) lives in `tests/backend/padded_frame_reference.py`. Every uniform test input is built by
scattering a ragged case into block layout with gappy masks in BOTH blocks and random garbage in every junk entry
(`uniform_testing.py`'s recipe). Frames are compared by the largest principal-angle sine per directed edge.

| # | test | pass criterion |
|---|---|---|
| T1 | `tv_align_padded_frame` vs the dense update: d = 2, 3, 5; uneven increments; non-minimal ranks; tangent part 10×; truncating; zero old point; groups; stack C | ≤ 1e-9 on all four stores; orthonormal ≤ 1e-12; old block bitwise; both chains reconstruct X_old |
| T2 | ungauged variations give the same frames; scale 1e-14 … 1e14; float32 error ∝ u·max(1, s) | ≤ 1e-9; same; within 3× of u·s |
| T3 | degenerate hints (zero, rank one, information-free increase): valid frames | orthonormal, pinned, reconstructs |
| T4 | two-colour pass = transport of the other stores; the cycle regression: `tests/data/padded_frame_cycle_case.npz` (243 numbers) — all-stores cycles (movement > 0.05 at pass 200), alternating converges (movement < 1e-8) | as stated |
| T5 | `fv_padded_frame` = the shipped S2 construction on the same inputs (bitwise), = `fv_zero_point_frame` at the zero tensor (bitwise) | bitwise |
| T6 | `aligned_padded_frame` vs the dense loop (p = 0, 40 passes) and vs the reference loop with the reference step | ≤ 1e-9 per pass |
| T7 | `pad_safe_pinned_completion`: vs `pinned_completion` on the real block, pads interior, rows and columns | bitwise zero on junk rows, real block equal up to the sign rule |
| T8 | `ufv_padded_frame` vs `fv_padded_frame` through the scatter (8 cases incl. zero point, four different old widths, groups) | ≤ 1e-12; under jit bitwise equal to numpy; junk zero |
| T9 | `utv_align_padded_frame` vs `tv_align_padded_frame` (the T1 list; two-colour; truncating with `out_widths`) | ≤ 1e-9 |
| T10 | stacks: C = (3,) with different ranks per element, C = (2, 2): stacked = per element | step/start bitwise; loop ≤ 1e-12 |
| T11 | `ufv_compact_frame`: masks == `ufv_make_frame_masks`; real entries bitwise; `utv_retract` of the zero tangent at the level's shapes returns X_old; `from_frame` variation masks == `from_point`'s | as stated |
| T12 | `ufv_aligned_padded_frame` vs `aligned_padded_frame` (defaults; tol 1e-10; zero point; non-minimal; p = 0; d = 2; alternating) | same pass counts; movements ≤ 1e-12; spans ≤ 1e-9 |
| T13 | jit: the uniform step, start and loop under `jax.jit` with `lax.scan` (d = 5); first-call time recorded at d = 6 and 24 (informational) | equal to numpy ≤ 1e-12 |
| T14 | optimizer seam: `local_model(frame=)` rejects a frame of wrong stores/masks (structural error), accepts the start frame; `newton_cg(padded_from=)` and `start_frame=` reproduce each other; `stats` fields; `g0_reference='cone'` deterministic given the seed; defaults unchanged → bit-identical results and `stats` without the new keywords (G-bit) | as stated |
| T15 | frontend twins: dispatch on classes; validation messages; doctests | as stated |
| T16 | dispatch table rows (`tests/test_dispatch.py`) for every new backend function (numpy/jax, ragged/uniform, stacked) | all rows |

## 7. Docs and CHANGELOG

- `docs/aligned_padded_frame.md` (new user doc): what the aligned start is, when to use `padded_from` and `'cone'`,
  the schedule and its knobs, what the measurements showed (facts table), the uniform block layout as seen by a user.
- `docs/rank_continuation.md`: a section pointing to it; `docs/numerical_contracts.md`: the padded-frame contracts
  (old block bitwise, completions ⟂, degenerate content arbitrary, two-colour precondition).
- `docs/contributor/padded_frame_internals.md` (new): the message passing in library terms, the block layout, the
  compaction and why, the pad-safe pinned completion and why a QR, compile accounting.
- `docs/padded_frame_cone.tex` (+pdf): the standalone math note after the code — Nick's note specialized to T3s,
  algorithm boxes citing the implemented functions.
- CHANGELOG `[2026.4.0]`: added / changed(none) / removed(none released); release notes.
- `dev/padded_frame_cone_judgment_calls_2026_11.md`: J-7 (the style pause), J-8 (the cycle and two-colour), J-9 (QR in the uniform start).

## 8. Slices

Each slice: implementer builds, reviewer reviews (style section first: every input/output a basic immutable type;
no template objects; scans; named contractions; annotations), the session checks the diff and the full suite, commits,
starts the next. Review stops (Nick + session) after N3, N6, N8, N10.

| slice | content | files | gates |
|---|---|---|---|
| N1 | **Cleanup**: `fv_operations.py` back to `efe5e3d`; delete S2/S3 tests and dispatch rows; discard the S4 working tree; create `backend/padded_frames.py` with `fv_padded_frame` (integer signature, S2 mathematics) and its tests (T5, T16) | `backend/fv_operations.py`, `backend/padded_frames.py`, `tests/backend/test_padded_frame.py`, `tests/test_dispatch.py` | full suite green; `git diff efe5e3d -- backend/fv_operations.py` empty |
| N2 | `tv_align_padded_frame` (scans, named contractions, HISTORICAL entries), the dense reference module, T1–T4 | `backend/padded_frames.py`, `backend/contractions.py`, `tests/backend/padded_frame_reference.py`, `tests/data/padded_frame_cycle_case.npz` | T1–T4 |
| N3 | `aligned_padded_frame`, T6. **Review stop.** | `backend/padded_frames.py` | T6 |
| N4 | `pad_safe_pinned_completion`, `ufv_padded_frame`, T7, T8 | `backend/linalg.py`, `backend/uniform_padded_frames.py` | T7, T8 |
| N5 | `utv_align_padded_frame` (block layout, scans, `pad_safe_svd`, groups, stacks, `update`, `out_widths`), T9, T10 | `backend/uniform_padded_frames.py` | T9, T10 |
| N6 | `ufv_compact_frame`, `ufv_aligned_padded_frame`, T11–T13. **Review stop.** | same | T11–T13 |
| N7 | optimizer seam: `local_model(frame=)`, `padded_start`, `newton_cg` keywords, `'cone'`, display, T14 | `backend/optimizers.py`, `backend/optimizer_display.py` | T14, G-bit |
| N8 | frontend twins + docstring moves, T15. **Review stop.** | frontend modules | T15, doctests |
| N9 | docs, CHANGELOG, release notes, judgment calls; the math note | `docs/`, `CHANGELOG.md`, `dev/` | doctests in docs |
| N10 | acceptance (§11). **Review stop.** | T3Polynomial (its own process) | the table of §11 filled |

## 9. Decisions

| # | decision | by / when |
|---|---|---|
| D-1 | The hinted path is the note's core form (frame + variation cores in; width old + 2k), replacing the dense construction | Nick 2026-11-09 |
| D-2 | Scope: the uniform layer IS in this release; NEWTON ONLY gets the aligned start; gradient-based loops later | Nick 2026-11-09 |
| D-3 | The committed S2/S3 hinted construction is deleted (never released); S1's primitives and the hint-free start stay; S4 discarded | Nick 2026-11-09 |
| D-4 | API shape: start / step / loop; `project` + `hint`; `project` REQUIRED in the backend (no shape-based dispatch), class dispatch in the frontend | Nick 2026-11-09 |
| D-5 | `g0_reference='cone'` reports the frame-gradient norm at the aligned frame | Nick 2026-11-09 |
| D-6 | Defaults: oversampling 5, oversampled passes 2, cap 10, tolerance 0.05, all user-settable kwargs; start and cone each have their own keyword group with the same defaults | Nick 2026-11-09 (groups: session) |
| D-7 | The cap is the effective stop; the tolerance an early exit; the cone reference = "the value after the schedule", deterministic given the seed | Nick 2026-11-09 |
| D-8 | Frontend twins: `T3Frame.padded`, tangent-level step, standalone loop with class dispatch, `oversampled_ranks` on both tensor classes; docstrings / doctests / validation in the twins; backend functions without twins keep their doctests | Nick 2026-11-09 |
| D-9 | Drop the three geometry twins; defer `frame=` on the six fitting factories | Nick 2026-11-09 |
| D-10 | Uniform geometry: CHECK-AND-REJECT a start frame whose masks differ from the level's (the loop returns matching masks) | Nick 2026-11-09 |
| D-11 | The start keeps the committed convention in both layers; its uniform twin is a pad-safe Householder pinned completion (the only QR; justified in §3.4) | Nick 2026-11-09 |
| D-12 | Degenerate content: arbitrary completions accepted where the hint carries no information | Nick 2026-11-09 |
| D-13 | A stacked loop stops on the maximum movement over the stack | Nick 2026-11-09 |
| D-14 | Two-colour updates after the truncating pass (default on) | Nick 2026-11-09 |
| D-15 | No polymorphism between layers; a uniform version only because it is truly uniform (block layout); otherwise users convert UT3 → T3 → UT3 | Nick 2026-11-09 |
| D-16 | Uniform padding for rank increases: one uniform old width and one uniform completion width, masks for the smaller edges | Nick 2026-11-09 |
| D-17 | Masks are the only truth in uniform inputs; our functions may return junk-last; the loop compacts at exit | Nick 2026-11-09 |
| D-18 | Library code: scans, named contractions, shape annotations; no raw QR where order matters | Nick 2026-11-09 |
| D-19 | Backend = pure functions on immutable basic types; the gold standard is the only precedent; `fv_zero_point_frame` unchanged this release (audit arc later) | Nick 2026-11-02 |
| D-20 | The loops are host loops in the library (the whole-loop jit of the prototype is not required) | session, for review |
| D-21 | The frontend loop's spelling of the zero old point (`x_old=None` + `shape=`) | open, for review |
| D-22 | Naming: `aligned_padded_frame` (ragged) and `ufv_aligned_padded_frame` (uniform); alternative `fv_aligned_padded_frame` for symmetry | open, for review |
| D-23 | New module names `padded_frames.py`, `uniform_padded_frames.py` | session, for review |
| D-24 | Carried over from the superseded plan: non-breaking additive release; no `dtype`; `padded_from` takes the old point (reduced structurally to minimal ranks); keyword-only appended parameters; gauge-equivalent cross-layer claims, bitwise only for structural facts; numerical tolerance checks for numerical tests; `compute_orthogonal_representation_ranks` stays in `ranks`; `gtol_rel=1e-4` with `'cone'` in the docs | Nick 2026-11-02 … 11-09 |

## 10. Risks and open questions

- **Convergence is slow and seed-dependent on real problems**; the loop is a start-frame heuristic, not a solver.
  Documented; the defaults are a budget. Whether more passes buy Newton iterations is being measured (§11).
- **Compile count** per level: the alignment adds step and gradient programs at up to two structures; oversampling 0
  removes one. To be measured with `JAX_LOG_COMPILES` in acceptance.
- **Cross-layer identity** rests on the start reproducing bases, not spans; degenerate seeds (an inside seed in the
  old span, up to roundoff) may differ across backends — the same caveat as the shipped ragged start.
- **Tied projections through `shared_data`** were not exercised in the prototypes; T9/T12 with groups cover the
  untied path only. The frontend twin for shared geometry must be checked in N8.
- **GPU** not measured anywhere.
- **Large d**: near-tied spectra converge slowly; the cap applies.
- **The uniform optimizer requires minimal ranks**; `padded_start` reduces `x_old`; a user calling the backend
  functions with non-minimal ranks must pass `out_widths`.

## 11. Downstream and acceptance

- T3Polynomial adopts `padded_from` and `g0_reference='cone'` in its ladder (its own process; `t3polynomial/cone.py` retired).
- Laptop: the Darcy o01 ladder, levels 1 → 6 and 6 → 11, starts: library, hint-free, aligned at 4 / 6 / 10 passes,
  aligned without oversampling, aligned at 40 passes; Newton and CG counts, objective, test error, two seeds; compile
  logging on one level. (The Newton part of this table is running as this draft is written.)
- Runpod: one manual 20×1000 pair (library vs aligned), decoupled from the release.

## 12. Pitfalls checklist (for every implementer and reviewer)

- [ ] Every backend input/output an array, tuple of arrays, int, static int tuple, host bool mask, or a frozen dataclass whose parameters are fields.
- [ ] No object read for its shapes; no dict of shapes; structure as `Sequence[int]` with `len=` comments.
- [ ] `project` required in the backend; `hint` never inspected.
- [ ] Scans for the sweeps (`xscan` / `lax.scan`); batched maps over the mode axis in the uniform layer; no Python loop over modes in uniform mathematics.
- [ ] Named contractions with HISTORICAL entries; no raw einsum.
- [ ] Shape annotation on every signature line.
- [ ] Uniform inputs read through masks only; junk masked once at entry; the split is the only positional fact; output junk-last with masks.
- [ ] `pad_safe_svd` for the step; `pad_safe_pinned_completion` for the start; no other QR.
- [ ] Structural counts and checks only; no rank tolerance; degenerate content left arbitrary.
- [ ] The old block bitwise in the output; carriers hold the old content only.
- [ ] A pass never both transports and truncates.
- [ ] The uniform loop returns the frame at the level's padded widths (compaction).
- [ ] Defaults unchanged; new keywords appended keyword-only; `stats` without the keywords bit-identical.
- [ ] Docstrings, doctests and validation in the frontend twins; backend one-liners; untwinned doctests stay.
- [ ] Compile accounting: nothing per-step enters a jit cache key.

## Appendix A. Prototype → library map

| prototype | library |
|---|---|
| `core_form2.align_stores`, `tv_align_padded_frame`; `cycle/core_form3.py` `update=` | `padded_frames.tv_align_padded_frame` |
| S2 hint-free path (`fv_operations.fv_padded_frame(template, old)`) + `core_form.start_frame` | `padded_frames.fv_padded_frame(shape, ranks, old_frame, ...)` |
| `ragged_loop.py` | `padded_frames.aligned_padded_frame` |
| `uniform_start.uniform_start(method='gs')`, `gs_complete` | `uniform_padded_frames.ufv_padded_frame`, `linalg.pad_safe_pinned_completion` |
| `uniform_form2.uniform_align2` | `uniform_padded_frames.utv_align_padded_frame` |
| `uniform_loop.uniform_aligned_padded_frame(alternate=True)`, `compact` | `uniform_padded_frames.ufv_aligned_padded_frame`, `ufv_compact_frame` |
| `dense_oracle.py` | `tests/backend/padded_frame_reference.py` |
| `g06c_cycle_case.npz` | `tests/data/padded_frame_cycle_case.npz` |

## Appendix B. From the superseded plan

Kept: §2.2 of the old plan (tangent cone, frame tangent spaces, aligned vs staggered completions) as background for
the math note; S1 (`pinned_completion`, `pinned_leading_columns`, `oversampled_ranks`) as shipped; the optimizer seam
(§3.6), the structural reduction of `x_old`, the keyword-only rule, the CHANGELOG/non-breaking gates, the acceptance
protocol (§11), the ledger of deferred items (O-7 `chunk_size` at working ranks; O-8 start frames for other
methods; `fv_zero_point_frame`'s template signature for the audit arc). Dropped: the hint as a T3, the dense pinned
T3-SVD construction and its exactness caveat, `tangent_to_t3`, the geometry members and their twins, the fixed pass
counts, "no uniform twin".
