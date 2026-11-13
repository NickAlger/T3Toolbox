# The aligned padded frame

When rank continuation raises the ranks of a fitted point `x_old`, the next level starts Newton from `x0`: `x_old`
zero-padded to the new ranks. `x0` is the same tensor, but its orthonormal frame has to say something about the new
slots, and the Newton step from a frame depends on those new directions. This page is about choosing them well.

Two tools come out of it:

- **`newton_cg(padded_from=x_old)`** builds iteration 0's frame by *aligning* the new directions with the misfit
  gradient, instead of taking whatever `x0`'s own orthogonalization produces;
- **`g0_reference='cone'`** measures the reference gradient norm `‖g₀‖` of the relative stopping test at the zero
  tensor, on a frame aligned with the data gradient there.

The frame twins behind them are public: `T3Frame.padded` / `UT3Frame.padded` (the hint-free start),
`fv_aligned_padded_frame` / `ufv_aligned_padded_frame` (the alignment with any hint), and `T3Tangent.align_padded` /
`UT3Tangent.align_padded` (one pass).

## The padded frame

A frame has four **stores**: the families of frame cores (down, up, left, right), each named for the side of its
edges that its cores occupy and each one rank axis per mode or bond (see
[Four rank stores](frame_variations.md#four-rank-stores--and-why-excess-rank-is-inert)). At a rank
increase every store gains slots. A **padded frame** keeps `x_old`'s own frame in the leading slots of every store,
bitwise, and fills the new slots with an orthonormal completion. Whatever the completion, the frame represents `x0`
(the carriers `left[d-1]` and `right[0]` keep the old content only).

**The hint-free start** (`T3Frame.padded`, `newton_cg(..., padded_max_passes=0)`) fills the new slots by a
data-independent convention. In every store, new slot `s` is seeded by row `s` of a keyed Gaussian stream, and the
seeds are completed against the old block by a sign-fixed QR. It is cheap and isotropic (a random subspace of the old
block's complement) and well conditioned. Because it is slot-indexed, successive starts never reuse a seed.

## The alignment

`fv_aligned_padded_frame(x_old, tucker_ranks, tt_ranks, hint)` refines the start toward a **hint**: a tensor whose
projection onto the frame's tangent space should be large. For `newton_cg` the hint is the misfit gradient.

**The method is a fixed-point iteration on projections.** Each pass projects the hint onto the current frame's tangent
space and, at every store, replaces the new slots by the leading new directions of that projection's content there,
given the other stores' current completions. The old slots stay fixed.

**What the fixed point satisfies.** At a fixed point, every store's completion spans the leading new directions of its
own content, given the other stores' completions, so one more pass moves nothing. The completions are then a stationary
point of the captured hint energy `‖Π_{T(F)} hint‖²` (proved for matrices, measured on T3s) but generally not a maximum
of it: rotating one store's completion alone never lowers that energy and generically raises it at second order. In
measurements on small trees, they are a strict local maximum of the cone merit, one of many. Updating any single store
alone moves nothing either (the `fv_aligned_padded_frame` docstring shows both). The cone merit is the share of the
hint's normal part that the new directions represent on one side of every edge; its best value does not depend on which
side.

**What it does not guarantee.** In the measurements the merit has many strict local maxima. An excursion of the hint
across several edges is captured only if every edge it crosses is aligned with it, so a configuration that ignores it
cannot see it at second order. The maximum reached depends on the start and the schedule; oversampling usually helps,
because wider completions see those excursions through each edge. Where the hint has no information at an edge, the
remaining directions there are an arbitrary valid completion.

### The schedule

1. **Oversampled passes** (`oversampling`, default 5; `oversampled_passes`, default 2): passes at the target ranks
   plus `oversampling`. They are skipped when those wider ranks give the target's stores.
2. **One truncating pass** to the target ranks.
3. **Two-colour pairs** (`alternate=True`, the default): a `('D', 'L')` pass (down and left stores), then a
   `('U', 'R')` pass (up and right), the other colour transported. A simultaneous update can cycle with period 2;
   the pairs do not. `alternate=False` runs simultaneous passes.
4. **The stop:** after a pair whose two movements are both below `tol` (default 0.05). The truncating pass never
   stops the loop, and a NaN movement never does. `max_passes` (default 10) caps all passes, the truncating one
   included. An odd remainder runs one final `('D', 'L')` pass. With the defaults: 2 oversampled, 1 truncating, 3
   pairs and a final `('D', 'L')`.

The **movement** of a pass is how far the new directions turned: the largest, over the stores, of
`‖g_old − P⁺P⁺ᵀ g_old‖_F`. It is 0 at a fixed point.

**Zero passes** (`max_passes=0`, `padded_max_passes=0`, `cone_max_passes=0`) return the hint-free start: no
projection is computed, and the other schedule options are not read. On the optimizers, setting another option of
the same group together with zero passes is an error, except the seed (`padded_seed`, `cone_seed`), which the
hint-free start uses.

When the target ranks add nothing to `x_old`'s structure, the frame is `x_old`'s and no pass runs.

### In the optimizers

- `newton_cg(..., padded_from=x_old)` aligns with the misfit gradient at `x0` (options `padded_*`).
  `start_frame=F` gives a frame directly; it is checked structurally and, in safe mode, for orthonormality and for
  representing `x0`.
- `g0_reference='cone'` computes `‖Π_{T(F)} g(0)‖` at the frame `F` aligned with the misfit gradient at the zero
  tensor (options `cone_*`). It is like-for-like with the gradient norms the stop compares. A relative tolerance near
  `gtol_rel=1e-4` is the natural scale with it. `cone_max_passes=0` evaluates it at the zero tensor's hint-free
  start, which is a different frame from `'zero'`'s (`fv_zero_point_frame`), so the two references differ.
- `g0_reference='warm_start'` with a start frame is `‖g‖` at that frame.
- The alignment's hint is the misfit gradient alone; the regularizer's gradient is left out. For the library's
  `IdentityRegularizer` this changes nothing: its gradient is a multiple of `x0`, which lies in every padded frame's
  tangent space, and the step uses only the normal content.
- `gradient_descent` takes `'cone'` and the `cone_*` options too, and runs the alignment eagerly.
- Both need a manifold geometry. `x_old` must be `x0`'s layer, shape, stack and padded `N`. It is reduced to
  minimal ranks first, and must have tied factors under sharing (safe mode).

### The zero point

A **structurally zero** `x_old` -- any Tucker rank or interior TT rank 0, e.g.
`TuckerTensorTrain.zeros(shape, (0,)*d, (1,)+(0,)*(d-1)+(1,))` -- is the zero tensor. Its frame is empty (no old
slots), and every completion is the convention's.

## Representation dependence

The hint-free start is a function of `x_old`'s frame **as represented**. Its seeds are read in the old frame's bases,
and Gaussian seeds do not flip with a slot's sign. Two representations of the same frame -- different bases at a
singular-value tie (a `t3svd` output, a `share()`d point), or a flipped slot sign, which an unfixed LAPACK SVD chooses
freely -- give different, equally valid starts. In practice:

- the ragged and the uniform starts of the same point differ: the two layers' orthogonalizations pick different
  slot signs. Measured span distances were 0.95 to 1.00 on generic points, `t3svd` outputs and shared points alike;
- a hint-free start can differ across BLAS builds;
- **the aligned frame does not depend on the representation**, up to `tol`. From those starts, the two layers' loops
  ended within 7e-4 to 3e-2 of each other at the default `tol`, with the same pass counts, and within 8e-13 to 6e-11
  at `tol=1e-10`, with pass counts within 2 (d = 4, nine cases).

Compare aligned frames across layers or machines, not hint-free starts. Tucker completions are seeded in physical
rows, which have no sign gauge.

## The uniform layer

The uniform loop works internally in **block layout**: every rank slot is `[old block | new block]`, each padded to
one width with its own mask. The old block is `x_old`'s frame as its masks give it (gappy masks are fine); the split
`(Na, Rb)` is the old block's Tucker and bond widths. `UT3Frame.padded` returns this layout and the split, which
`UT3Tangent.align_padded` takes.

`ufv_aligned_padded_frame` returns the frame **compacted** to the level's widths: real slots first, with the prefix
masks of the frame `ut3_orthogonal_representations` gives a point padded to that level. Newton can start from it
without compiling anything new.

Its **movement record** is a fixed-length `(max_passes,)+C` array, NaN after the last pass, because the jitted loop
needs a static shape. It cannot tell a NaN movement from a pass that never ran.

## Jit or eager

Nothing jits unless asked. `use_jit=True` on the loop twins -- or `newton_cg`'s own `use_jit`, which governs the
alignment as well as CG -- selects it:

- **ragged**: the host loop calls a memoized jitted step. A level compiles two to four step programs (the oversampled
  `('D', 'U', 'L', 'R')`, the truncating `('D', 'U', 'L', 'R')`, `('D', 'L')` and `('U', 'R')`), about 2 s each at
  d = 4, and nothing recompiles across passes. The projection runs eagerly;
- **uniform**: the WHOLE loop -- the start, every pass, the projections, the compaction -- is one program, compiled once
  per structure: about 9 s at d = 6 and at d = 24 (first call), then 0.02–0.08 s. `tol` and the hint's arrays are
  traced, so new values do not recompile. A custom `project` must then be traceable and module-level. The library's
  projections and the optimizers' misfit gradient are.

**Eager jax is expensive.** On jax arrays without `use_jit`, every new structure compiles its operations one by one.
The first eager step at d = 4 issued 932 compilations, about 22 s, and 0.16 s after that. Each eager alignment at a new
structure also adds 8,000–9,500 process memory mappings, one per compiled executable section. An eager-jax rank
continuation that aligns at every level can therefore pass Linux's `vm.max_map_count` (65,530) after about seven
levels and abort with `LLVM ERROR: Unable to allocate section memory!`. **With jax inputs, pass `use_jit=True`.** The
loops do not warn. numpy inputs need nothing; they are fast and eager.

## What the measurements showed

**The shipped code on full rank-continuation ladders** (A100, exact Hessian, `gtol_rel=1e-3`, the Darcy ladder of
the downstream T3Polynomial project, 2026-11-10). On 10 points × 100 directions (six levels), the Newton totals were
254 for the library's own start frame with the `'zero'` reference, 119 with `'cone'`, 110 for the hint-free padded
start with `'cone'` and 115 for the aligned start with `'cone'`; CG totals 10,175 / 2,610 / 2,576 / 1,839. The
reference, not the start, halves Newton: `'cone'` was 20–100× larger than `'zero'` there, so the same `gtol_rel` stops
at a larger `‖g‖`, with best test errors within 4 % (p50 0.0607–0.0631). Within `'cone'`, the aligned start took about
30 % fewer CG iterations than either other start, mostly at levels 3–5. At 20 points × 1000 directions (eleven
levels, the same reference), the aligned start took 166 Newton / 4,130 CG against the hint-free start's 191 / 4,719,
with identical test error; a library-start run from 2026-11-01 (zero-type reference, another card) took 167 / 4,688.
**Every alignment stopped on the 10-pass cap, never on `tol=0.05`** (last movements 0.27–0.43): in practice
`padded_max_passes` is the knob that sets the alignment's cost.

**Earlier, from a laptop Darcy ladder** (exact Hessian, the same absolute stop for every start, two start seeds; the
prototype, with an earlier hint-free convention):

- **A generic seeded start roughly halves the Newton count at level 2.** The hint-free start and every aligned start
  with oversampling cut the library padded frame's 49 Newton iterations to 21–33.
- **Its Newton value appears with rank.** At levels 1–2 it did not reduce Newton counts beyond the seed spread. At the
  higher-rank level 11 → 16, the aligned start with 10 or more passes cut Newton iterations by about 40 % and CG
  iterations by about 2.4×; 4 passes did not.
- **It is cheap**: under 10 % of a gradient per pass (45 ms against a 1.1 s gradient at 11 → 16).

**The alignment captures far more of the gradient.** Re-recorded with the shipped code, the fraction
`‖P_F ∇f‖ / ‖∇f‖` at levels 1 → 6 / 6 → 11 / 11 → 16 was 0.06–0.07 / 0.09 / 0.09 for the hint-free start and
0.45–0.46 / 0.35 / 0.26 for the default alignment: 3–8× more.

The mathematics, with the algorithms as implemented, is written up in the math note `docs/padded_frame_cone.pdf`
(source `docs/padded_frame_cone.tex`).
