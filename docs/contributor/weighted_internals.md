# Weighted layer — implementation internals

Design records for the weighted layer (`docs/weighting.md` is the user-facing account). Two builds
produced it: the ragged layer and then its uniform mirror, and most of what is worth preserving is
*why a tempting alternative is wrong*, plus three corrections the builds made to their own plans.

---

## Two classes, because a tensor and a tangent have different edges

`T3Weights` pairs with a `TuckerTensorTrain`; `T3FrameWeights` pairs with a tangent. They are not the
same object wearing two hats:

- A **T3 as a network** has exactly two internal edge families — Tucker-rank edges `nᵢ` (`d` of them)
  and TT-bond edges `rᵢ` (`d+1`, ends trivial). So a weight is `(tucker[d], tt[d+1])` — which is
  *exactly* the shape `t3svd` returns, so the singular values are literally a `T3Weights`.
- A **tangent** is not one network. It is a sum of `d` diagrams, each with one variation core, and the
  edges *left* of the variation are left-orthogonal (`rL`) while those *right* are right-orthogonal
  (`rR`). Those four rank families (`nU`/`nD`/`rL`/`rR`) are not a labelling choice — they are *which
  side of the variation an edge is on*, and the shapes forbid mixing (`rLₖ ≠ rRₖ` in general; a left
  diagonal will not fit a right bond).

Forcing one class would need `down=None, left=right` for the plain case — uglier than two honest types.

## The tangent weight is a METRIC (the design change that caught us out)

The first design absorbed the tangent weights into the **frame** cores (`up→U, down→O, left→P,
right→Q`, variations untouched). It shipped as the opposite: a **metric on the variation
coordinates**, absorbed into `V`/`H`, with the frame left orthonormal.

Why: the frame's `d+1`-th left/right cores are base-point padding — "not really part of the frame"
(`docs/frame_variations.md`) — so there are only `d` *natural* tangent edges, matching the `d`
variation cores. And the metric reading is what Grasedyck–Kramer actually needs.

The abandoned design is worth knowing about because it explains a piece of machinery that **does not
exist**: absorbing into the frame breaks orthonormality, which would force the weighted norm through
the doubled-rank `tv_to_t3` embedding. Under the metric design the frame stays orthonormal, so the
weighted norm is just the corewise norm of the weight-scaled variations — `O(ranks)`, no doubled
tensor, no `tv_weighted_*` twins. If you find a note describing that path, it is pre-change.

## A weight batches with the FRAME but is absorbed into the VARIATIONS

**This is the subtlety, and it is the one that actually bit.** The two questions are independent:

| question | answer for `T3FrameWeights` |
|---|---|
| what does it *act on*? | the **variations** (`down`→V, `up`/`left`/`right`→H) |
| what does it *batch with*? | the **frame** — stack `C`, never the variations' `K + C` |

A metric is one-per-base-point, shared by all `K` tangent vectors at that frame. At the design change,
the implementation moved the *absorption target* to the variations — correctly — and moved the *stack*
along with it, which was wrong. Nothing computed the wrong number (the leading `'...'` broadcasts `C`
over `K + C` for free, and a `C`-only weight was verified bit-identical to a `K`-tiled one), but
`fv_weights_consistent` demanded the weight match the variations' full `K + C` stack, and so **rejected
the canonical Grasedyck–Kramer weight** — `from_t3weights(from_t3svd(x))` is `C`-stacked — the moment
there was a `K`-stack of tangents. The predicate contradicted the operation it described.

It survived because the tests only ever built `K+C` weights, so the broken case was never constructed,
and because `is_consistent_with` is purely diagnostic (in no enforcement path) — a lying predicate, not
a wrong result. Fixed 2026-07-15; the lesson is in the table above.

### The two-level check that came out of it

Pairing a weight with variations and pairing it with a frame are different questions, so they get
different checks:

- **`fv_weights_consistent(variations, weights)` — the trailing rule, blind to the frame.** The
  weight's stack must be the *trailing* part of the variation stack. This is exactly the rule
  `check_fv_pair` already applies to (frame, variations), with the weight playing the frame's role. It
  stays blind on purpose: the variations are intentionally blind to their frame, and the standalone
  `fv_absorb_weights(variations, W)` has no frame to consult.
- **`check_fw_pair(frame, weights)` — the exact stack, where the frame is present.** A `K+C` weight
  passes the trailing rule too (it reads as `C_w = K+C`: that many base points, one tangent each — a
  coherent object, just not *this* frame's metric), so only here is there enough information to reject
  it. Wired into `T3Tangent.weighted_norm`/`weighted_inner`/`absorb_weights`.

**Rejected:** checking the weight against the *frame* in the backend predicate
(`fv_weights_consistent(frame, weights)`). It invents a second pairing pattern where the library
already has one, and couples the predicate to a frame the standalone absorb does not have.

## The uniform mirror

A uniform weight is "the ragged weight, padded for performance." It **carries the edge masks** of the
object it weights (a weight's edges *are* the tensor's edges) and has **no `shape` field** — weights
live only on internal edges; external/physical-mode weights are out of scope.

### `absorb` is garbage-transparent — it needs no masking

The pre-build plan asserted that "absorb and norm/inner are reductions, so garbage padding must be
zeroed first." True of norm/inner; **false of absorb**, and the claim is load-bearing enough that a
reader would implement a pointless entry-mask against it.

`absorb` is a *pointwise scale along each edge axis*, not a reduction: output slot `i` depends only on
the core's slot `i` and the weight's slot `i`, so garbage can never mix into a real slot. Garbage in
the padding gives garbage in the padding, which the equivalence contract declares don't-care. The
reductions that follow (`weighted_norm`/`_inner`) reduce to the *existing* plain uniform norm/inner,
which mask their own inputs. Same for `concatenate`/`kronecker`: supercores are copied or
outer-product'd, never reduced.

### The masking/weighting wall, and why it cost nothing

Masking and weighting are kept apart in code even where the mechanics coincide, because they are
opposite kinds of thing: masks are boolean **structure** (static aux, non-differentiable, define rank,
value-neutral); weights are float **parameters** (traced leaves, differentiable, scale the contraction,
value-affecting). This is the same distinction that
[`uniform_rank_masks_rationale.md`](uniform_rank_masks_rationale.md) invokes to reject "a float mask
doubling as edge weights."

> **Weighting code never calls a `*_apply_masks` / `*_make_masks` function.** Where the mechanics
> genuinely coincide, the shared core is broken out into a **neutral** subfunction that each side calls.

Because `absorb` needs no masking, the wall falls out rather than being imposed —
`backend/ut3_operations.py` does not import the masking layer at all. The rule forced exactly two
extractions, both of which improved the layer independently of weighting:

- **`common.prefix_mask(ranks, pad)`** — `arange(pad) < ranks[..., None]` was written out in ~8 places
  across four modules, including a private `_prefix_mask` the mask builders themselves did not use.
- **`common.require_concrete_masks`** — already documented as "a jit guard (infrastructure), unprefixed
  by design", yet living in `ut3_masking`; `ufv_masking` never used it, so the home was accidental. It
  belongs beside `ValueHashedMasks` and `prefix_mask`: those three are the uniform
  **mask-representation contract** (concrete host arrays · value-based hash/eq · canonical prefix form),
  obeyed by every uniform object's masks rather than owned by any one family.

*(Known gap, surfaced by that move and not fixed: `ufv_masking` still does not call
`require_concrete_masks`, so the frame/variation masks are unguarded against being passed as traced jit
args — they fail with jax's cryptic `ConcretizationTypeError` instead of the actionable message.)*

### `reciprocal` must guard the padding — and this is why weights carry masks

A canonical weight's padding is **zero**, and `1/0 = inf`. Masking downstream works by multiplication,
so `0 × inf = nan` then poisons every masked reduction — the exact hazard
[`../uniform_equivalence_contract.md`](../uniform_equivalence_contract.md) names ("the fill must be
finite"). This is not a corner case: **the Grasedyck–Kramer metric *is* `from_ut3svd(x).reciprocal()`**,
so the headline path runs straight through it.

The fix is the standard double-`where`: neutralize the padding to `1`, apply `fn`, then overwrite the
padding with `0`. Both halves earn their keep — the inner one keeps the **gradient** finite (a `nan`
from a dead branch still propagates backward through a single `where`), and it neutralizes large-finite
garbage padding too, so no separate entry-mask is needed. `sqrt` uses the same helper (`sqrt` is fine at
`0` but not differentiable there, and garbage padding can be negative).

**Real-slot zeros are deliberately NOT guarded.** A genuinely zero singular value gives `inf`, exactly
as in ragged. It is real data, not a padding artifact; silently clamping it would hide a rank-deficient
point.

This is also the argument that a uniform weight **must** carry masks: a maskless weight cannot implement
`reciprocal` correctly.

### `concatenate` / `kronecker`: the mask is just another weight

The pre-build plan billed these as "the hard ones" and "the most error-prone code in the build."
**Wrong** — and worth correcting, because it sends a reader looking for a difficulty that is not there.

Treat the mask as another weight that happens to hold 0s and 1s. What is needed is

```
weight_AB * mask_AB  ==  combine(weight_A * mask_A, weight_B * mask_B)
```

and it is satisfied by combining the weights and the masks **the same way**, because both concatenation
and the Kronecker product commute with elementwise multiply (`kron(a∘p, b∘q) = kron(a,b) ∘ kron(p,q)` —
the mixed-product property; nothing special to booleans). Each op is one operation applied twice, with
no mask cleverness at all. That closure *is* the argument for boolean masks over integer ranks: an
integer rank cannot even express the strided result.

The genuine risks are ordering/axis mistakes:

1. **`kron` must be a last-axis outer product broadcasting the shared `(d,)+stack` prefix — NOT
   `np.kron`**, which would Kronecker the mode/stack axes too. The ragged build hit exactly this.
2. **A-major must agree with any core-combine that pairs with it.** In uniform there is none yet — there
   is no `ut3_mult`, only `ut3_add` — so `ut3_kronecker_weights` ships verified against the ragged oracle
   but *unpartnered*. If `ut3_mult` is ever built, its index pairing must match.

The output masks *are* gappy (concat leaves a hole wherever an input had rank slack; kron's real set
`{a*nB + b : mask_A[a] and mask_B[b]}` is strided over the **padded** width, so even two prefix inputs
give holes). That is a description of the result, not a difficulty: it costs the combines nothing, and
only obliges *consumers* to read the mask instead of slicing a prefix. It cannot be flattened — slot
`a*nB + b` with `b >= rank_B` holds `wA[a] * <padding>`, so a prefix mask of rank `rA·rB` would claim
padding as real data (phantom rank).

**Consuming a gappy mask needs no `argwhere`** — boolean indexing (`w[m]`) selects the real slots in
ascending order directly, and works on numpy, on jax eagerly, and even under `jit` (the masks are
closed-over host constants, so jax resolves the shape at trace time; `NonConcreteBooleanIndexError`
fires only for a *traced* mask, which `require_concrete_masks` forbids). The point was never
"gather vs slice" — it is that a rank mask may be gappy, so the real slots must be selected *through
the mask* rather than as a prefix. (The physical `shape` genuinely *is* a contiguous prefix, and does
slice.)

### The one precondition uniform adds

Ragged catches a weight/object rank mismatch as a loud einsum shape error (a length-`n` weight vector
against a rank-`n` core). Uniform pads both to a common `(n, r)`, so a mismatched **mask** is invisible
to the shapes and would silently corrupt: a weight whose mask calls slot `i` padding carries a canonical
zero there, so absorbing it **zeroes a real slot**. Hence an explicit structural predicate
(`ut3_weights_consistent` / `ufv_weights_consistent`) with frontend enforcement — the same precondition
uniform adds to variation add/sub ([`../uniform_masks_vs_ranks.md`](../uniform_masks_vs_ranks.md)).

## Placement notes (both forced, not preference)

- **`utv_weighted_norm` / `utv_weighted_inner` live in `utv_operations`**, not beside
  `ufv_absorb_weights` as their ragged twins do (`fv_weighted_norm` is in `fv_operations`). They need
  `utv_corewise_inner`, and `utv_operations` already imports `ufv_operations` — the ragged placement
  would be a circular import. Ragged has no such constraint: its `fv_` layer reaches a standalone
  `corewise` module.
- **The frontend free functions carry the family prefix** (`t3_absorb_weights` / `ut3_absorb_weights` /
  `fv_absorb_weights` / `ufv_absorb_weights`). Ragged/uniform × tensor/tangent gives *four* distinct
  "absorb the weights" operations; unprefixed they would all be `absorb_weights` and could not coexist
  at the package root. See [`../naming_conventions.md`](../naming_conventions.md) — the rule is that a
  free function has no class namespace, so the prefix does that job.

## No ragged/uniform dispatch — the module IS the dispatch

The weighted free functions are deliberately **parallel and module-scoped**: nothing infers the layer
from the argument. The user controls dispatch by deciding which layer they work in, and the conversion
hooks are how they switch.

`optimizers.py`'s `isinstance(x0, UniformTuckerTensorTrain)` is not a counterexample: `newton_cg` is a
single entry point a user calls with whatever `x0` they have, so it has *no module* to dispatch through.
`absorb_weights` does.

## What's deferred

- **Weighted `+` / `−` / scale / `⊙` as operations**, and an optional thin container for operator sugar
  (`a + b`, `a.norm()`). Guaranteed reachable: every weighted op reduces to a move the substrate already
  provides — add → direct-sum cores + `concatenate` weights (+ tail-squash); `⊙` → Kronecker-combine
  cores + `kronecker` weights; inner/norm/apply/entries/probe → `absorb` then the plain op. So each is a
  *pure addition* (a new free function), never a refactor.
- **The Grasedyck–Kramer `SingularValueRegularizer`** — the intended consumer. If `M = W²` then
  `⟨x, Mx⟩ = ‖absorb(x, W)‖²` and `Mx = absorb(x, W²)`, so absorb-into-tangent is exactly its primitive;
  it builds `W` from the frame's singular values and calls absorb. Both layers now have everything it
  needs. See [`fitting_internals.md`](fitting_internals.md).
- **External / physical-mode weights** (on the `Nᵢ` output legs) — out of scope, deliberately. The
  refactor made clear the external edges are of a different character than the internal ones; the old
  code's `shape_weights` family was dropped.
- **Backend `ut3_norm` / `ut3_inner` twins** — see [`deferred_and_rejected.md`](deferred_and_rejected.md).

## Build records

`dev/archive/weighted_layer_design.md` (ragged) and `dev/archive/uniform_weighting_design.md` (the
uniform mirror; §8 is its decision log). Both are historical — where they disagree with this note or
with `docs/weighting.md`, they are wrong.
