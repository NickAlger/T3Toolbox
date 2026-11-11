# Padded and aligned frames: internals

The user-facing story is [`../aligned_padded_frame.md`](../aligned_padded_frame.md); the mathematics is the math
note [`../padded_frame_cone.tex`](../padded_frame_cone.tex) (+pdf). This page is for whoever changes
`backend/padded_frames.py`, `backend/uniform_padded_frames.py` or the optimizer seam that uses them.

## The step as message passing

A T3 is a tree: Tucker leaves `l_i` and TT cores `c_i` in a chain. Each **directed** edge has one frame core, and the
four directions are the library's four stores:

| directed edge | store | output slot |
|---|---|---|
| `l_i → c_i` | `up[i]` | the Tucker index |
| `c_i → l_i` | `down[i]` | the down index |
| `c_i → c_(i+1)` | `left[i]`, `i ≤ d-2` | bond `i+1` |
| `c_i → c_(i-1)` | `right[i]`, `i ≥ 1` | bond `i` |

`left[d-1]` and `right[0]` are the carriers of the base point. They hold the old content only and are never
completed. A padded frame has the old frame bitwise in the leading slots of each store (the **A-block**) and the
completion in the trailing ones (the **P-block**).

`tv_align_padded_frame` replaces every completion by the leading new directions of a tangent vector's normal content
at that edge, with the A-block pinned. It does this without ever forming the doubled-rank tangent, in two
message-passing stages:

1. **Far-side messages, at the old frames** (`_up_message`, `_left_message_step`, `_right_message_step`,
   `_down_message`). On each directed edge `a → b`, `Φ = Pᵀ Z_far U′` (what the far side sends through the old far
   frame) and `Ξ`, a square root of the rest (`_root`, from a thin SVD; only `ΞΞᵀ` is used). The order is the down
   edges, the left messages from the far end, the right messages from the near end, then the up edges.
2. **New completions, in message order** (`_down_edge`, `_left_edge_step`, `_right_edge_step`, `_up_edge`). At
   `a → b`, `a`'s incoming slots are first re-expressed in their new coordinates (`_new_coordinates`: the old block
   kept, the completion block multiplied by `o = P⁺ᵀP`). Then
   `Y = [g_old Φ_P + g_new | g_old Ξ]` (`_edge_content`), and the new completion `P⁺` is the leading directions of
   `(I − ĀĀᵀ)Y`, completed against `Ā` (`linalg.pinned_leading_columns`). The edge emits `o` and
   `D = [Ā | P⁺]ᵀ g_new` for the edges after it, and its movement `‖g_old − P⁺o‖_F`.

`Y` is the edge's normal content exactly (the note's `Y = Q(LJ)Θᵀ` identity), so nothing wider than `old + 2k` is
factorized. The messages past the chain's ends are zero-size arrays, so every core runs the same body, and the
sweeps are scans with module-level bodies ([`scan_body_principles.md`](scan_body_principles.md)).

**Sharing.** A sharing group has one Tucker factor. Its up edge stacks the members' contents side by side in one `Y`
(`_down_edge`), so the group gets one completion.

**Transport.** A store not in `update` gets `Y = g_old`: its old completion re-expressed in the new candidate space.
Two-colour passes (`'UL'`, then `'RO'`) are alternating (Gauss–Seidel) updates of the two colours, with a projection
in between. A pass may not both transport and truncate: a transported store can lose directions arbitrarily in a
truncated neighbour's coordinates, so the step raises.

**What a store's update maximizes, and what the loop converges to.** Each updated store takes the leading directions
of its own content matrix `Y` given the others, so it maximizes `‖P⁺ᵀY‖²`, the captured energy of that content. `Y`
depends on the store's previous completion, so this is a self-consistency condition, not the maximizer of one fixed
objective. In particular the loop is not coordinate ascent of the joint captured energy `E = ‖Π_{T(F)} hint‖²`: at the
fixed point `E` is a saddle, and per store a minimum.

At a fixed point, every store's completion spans the leading new directions of its own content, given the other stores'
completions, so one more pass moves nothing. The completions are then a stationary point of the captured hint energy
`‖Π_{T(F)} hint‖²` (proved for matrices, measured on T3s) but generally not a maximum of it: rotating one store's
completion alone never lowers that energy and generically raises it at second order. In measurements on small trees,
they are a strict local maximum of the cone merit, one of many.

The cone merit is the share of the hint's normal part that the new directions represent on one side of every edge; its
best value does not depend on which side. The math note (§5) has the matrix proof and the measurements.

## The uniform layer: block layout

Every rank slot of a uniform padded frame is `[A-block | P-block]`, each padded to one width with its own mask. The
only positional fact is the static split `(Na, Rb)`, the A-block's Tucker and bond widths. The A-block is the old
frame's supercores AS GIVEN, with whatever gappy mask pattern they have; junk is zeroed once, at entry. Nothing
assumes junk-last on input.

- **The empty frame** (`fv_empty_frame` / `ufv_empty_frame`, `Na = 0`, `Rb = 1`) is the orthogonal representation
  of the rank-0 T3: zero-width rank axes except the boundary bonds. The zero point then needs no special case: every
  slot is a completion, and the cone reference and a structurally zero `padded_from` run the same code.
- **The edge** (`_u_edge`, `_pad_safe_basis`) is the ragged edge's `pinned_leading_columns` in padded form: one
  `linalg.pad_safe_svd` of `[Ā | Y⊥/(4‖Y⊥‖_F)]`. The pins have singular value 1, and the scaled content's are at
  most 1/4, so the pins lead. A host one-hot (`_selection`) then picks the `k⁺` triplets after the pin count. Every
  count comes from the masks. No rank is decided from content: degenerate content is completed by `pad_safe_svd`.
- **The compaction** (`ufv_compact_frame`) gathers the real slots first at a level's widths, with prefix masks. The
  loop's output then has the structure that `ut3_orthogonal_representations` gives a point padded to that level, so
  Newton reuses its compiled programs. If the level is wider than the input frame, the frame is zero-padded first.

## The start: the pad-safe pinned completion, and why a QR

The hint-free start completes each store's A-block with Gaussian seeds: slot `s` gets row `s` of the keyed legacy
`RandomState([seed, store tag, key])` stream (`_seed_rows`; tags up 0, down 1, left 2, right 3; the key is the
mode, or a sharing group's first mode). The rows are drawn on the host at the ragged unfolding's length. The ragged
layer completes them with `linalg.pinned_completion`, the sign-fixed Householder QR of `[pins | seeds]`. The uniform
layer places the same rows at the real rows (the flat slot order is the ragged row-major order) and calls
`linalg.pad_safe_pinned_completion`.

A QR, not an SVD, because the convention is defined by the seeds' ORDER (Gram–Schmidt: column `j` is seed `j` made
orthonormal against the pins and the earlier seeds), and an SVD orders by singular value. The padded form gets the
ragged order from static host permutations built from the masks: real rows first; columns ordered as real pins, the
selected seeds in order, then the rest. The junk rows are zero and trail, so no reflector of a selected column
touches them, and the result is bitwise zero there.

The start is a function of the old frame's **representation**, slot signs included. A seed row is read in the
neighbours' slot coordinates, and a Gaussian row does not flip when a slot's sign does. The two layers'
orthogonalizations pick different signs and different bases at ties, so their starts differ (generically by about
1.0 in span). This was accepted (D-42, D-49): the aligned output is representation-independent to `tol`.
Representation-level canonical bases are a separate audit item. The tests compare the layers' starts per layer, each
against its own old frame (`tests/backend/test_padded_frame.py`, T5; `TestStartChains`).

## Jit: kernels, getters, and compile accounting

Nothing jits unless asked (D-39).

- **The kernels are jittable and module-level.** The ragged step is `tv_align_padded_frame` itself. The uniform step
  is `_utv_align_kernel(dynamic, skeleton, split, update)`; its host-side preparation (masks, row masks, pin counts,
  selections, group tables) goes into the `StaticSkeleton` through `partition_static`.
- **The memoized getters.** `_jitted(fn, static_argnums)` is an `lru_cache` around `jax.jit`, so one wrapper per
  function persists across calls and its programs are reused.
  - **Ragged** (`fv_aligned_padded_frame(use_jit=True)`): the host loop calls `_jitted(tv_align_padded_frame,
    (2, …, 7))`, with the old ranks, output ranks, `sharing` and `update` as statics. A level compiles at most four
    step programs: the oversampled `'ULRO'`, the truncating `'ULRO'`, `'UL'` and `'RO'`. The projection runs eagerly
    between steps.
  - **Uniform** (`ufv_aligned_padded_frame(use_jit=True)`): the WHOLE loop is
    `_jitted(_ufv_aligned_loop_kernel, (2, 3, 4, 5))`. The statics are the skeleton (structure and masks), `project`,
    the hashable `schedule` (working and target ranks, sharing, seed, `n_over`, `max_passes`, `alternate`, level
    widths) and `use_jit`. The traced part is the old frame's and the hint's arrays, and `tol`. Inside: a `fori_loop`
    of oversampled passes, the truncating pass, a `while_loop` of `(UL, RO)` pairs, a `cond` for the final `'UL'`,
    and the compaction. One compile per structure; new values of `tol` or of the arrays do not recompile.
  - The loop bodies are bound with `functools.partial` inside that one jit. This is accepted in this instance only
    and sets no precedent (D-47; [`scan_body_principles.md`](scan_body_principles.md)).
- **The record.** The jitted loop needs a static output shape, so the uniform movements are `(max_passes,)+C`, NaN
  after the last pass, in both modes. The stop uses `np.max` / `xnp.max`, so a NaN never stops the loop.
- **Eager jax.** Each op compiles on first use. The first eager ragged step at d = 4 issued 932 compilations. Each
  eager-jax alignment at a new structure adds 8,000–9,500 process memory mappings, and `vm.max_map_count` is
  65,530. The tests release them with `jax.clear_caches()` (`_release_jax_programs` in `tests/test_dispatch.py`).
  The loops do not warn (D-40); the user doc recommends `use_jit=True` for jax.

## The optimizer seam (T3-free)

`backend/optimizers.py` imports no T3 module for this feature. Three optional geometry members carry the
layer-specific work. They are defined on `ManifoldGeometryOps` and `UniformManifoldGeometryOps`, with sharing
through their `groups` field:

- `with_frame(frame)`: the geometry at a frame's structure. The ragged geometry holds no structure and returns
  itself. The uniform one takes the frame's up and left masks as its variation masks.
- `require_frame_structure(x_cores, frame)`: a structural check for `local_model(frame=)` and `start_frame`. It
  works from integers (the frame's core shapes against `compute_orthogonal_representation_ranks` of `x`'s ranks,
  plus `C` and `N`) and from held masks (the uniform variation masks), and never orthogonalizes `x`. Reading the
  structure from an orthogonalization would cost a sweep and could decide ranks from content.
- `aligned_padded_frame(x0, x_old, project, hint, ...)`: the empty frame when `x_old` is `None` or structurally zero
  (any Tucker or interior TT rank 0), else `self.frame(x_old)`; then the layer's loop. The optimizers pass
  `_misfit_gradient_at` as `project` and `(problem, residual)` as the hint, so the misfit gradient at each frame is
  the hint's projection.

`_cone_reference` calls the member at the zero tensor (`x_old=None`) with the zero point's residual. It then
evaluates the frame-gradient norm with `with_frame(frame).inner`.

**A problem's aligned frame without Newton.** Build the problem (`least_squares_problem`, the public constructor)
and call `backend.optimizers.padded_start(problem, x0, x_old, ...)`. It returns the member's frame and movements;
`Problem.local_model(x0, frame=F)` linearizes there. That is the route for a custom optimizer, and the one
`newton_cg(padded_from=)` takes internally.

## Where the guarantees live

- `tests/backend/test_padded_frame.py`: the ragged start (T5 convention reference), step (T1–T4 against dense
  oracles) and loop (T6, the pair rule).
- `tests/backend/test_uniform_padded_frame.py`: the uniform twins against the ragged ones per element (T8–T12, T17),
  the chains (T20).
- `tests/backend/test_padded_start.py`: the seam (T14, T19).
- `tests/test_aligned_padded_frame_frontend.py`: the frontend twins (T15).
- `tests/test_dispatch.py`: dispatch rows for every public function, and the jit equivalence and compile counts.
