# Weighted tensor-network layer — design note

_Started 2026-07-14. Redesign of the parked weighted layer. Scope: a lightweight **edge-weight
representation** (diagonal weights on the internal edges of a T3 / a frame-variations tangent), the
ability to **absorb** weights into cores, and enough weighted linear algebra (`norm`, `inner`,
`concatenate`, `kronecker`) that the weighted objects form a closed algebra — while leaving a clean seam
for the deferred **Grasedyck–Kramer singular-value regularizer** (which consumes this layer), and NOT
rebuilding the heavy `WeightedTuckerTensorTrain` wrapper. Design agreed interactively with Nick; **all
decisions settled 2026-07-14** (§9). **SHIPPED 2026-07-15** — slices S1–S5 done (commits `358860bb`,
`059124f1`, `99dcb8b5`, `80354977`, + docs). **One design change during build:** the tangent weight is a
**metric on the variation coordinates** (Approach-1, `d`-each, weight `V`/`H`, frame orthonormal), NOT the
tensor-weighting-of-all-edges we first sketched — because the frame's `d+1`-th left/right cores are
base-point padding ("not really part of the frame"), so there are only `d` natural tangent edges, and the
metric interpretation is what Grasedyck–Kramer needs (§6 revised). Deferred: weighted `+`/`⊙` operations,
the uniform mirror, the GK regularizer._

## 0. Resume from a fresh context (read this first)

The idea: it is sometimes useful to imagine **diagonal matrices inserted between neighbouring cores** of
a Tucker-tensor-train network. Putting e.g. inverse singular values there makes the norm of the weighted
network penalise the "less-informed" rank directions (Grasedyck–Kramer preconditioning). We are reviving
this — but *lightly*: the parked layer (`weighted_tucker_tensor_train.py`, `backend/wt3_operations.py`,
and a broken `absorb_weights_into_tangent_cores` in `fv_operations.py`) had the right **data format**
(`EdgeVectors`: per-edge diagonal vectors) but wrapped it in a heavy `WeightedTuckerTensorTrain` class
that mirrored the whole `TuckerTensorTrain` API. Nick's call: keep **weight classes**, drop the wrapper,
and combine weights with objects via **functions**.

**Two weight classes**, because a `TuckerTensorTrain` and a tangent have genuinely different edge
structure (§2). **Everything else is a function of `(object, weights)`** (§4). The linear algebra we can't
yet foresee is *guaranteed reachable* because it is a property of the **data format + `absorb`**, not of a
wrapper (§5). Verified this session: the Kronecker law for Hadamard products (§3, rel err 1.2e-15) and
that `tv_to_t3` is a purely structural assembly usable on weighted (non-orthonormal) cores (§6).

## 1. Goal & scope

- **Build now:** the two weight classes + `absorb` + `is_consistent_with` + `from_t3svd` +
  `weighted_norm` + `weighted_inner` + `concatenate` + `kronecker`. Ragged first.
- **Explicitly deferred (but architecturally reachable — §5):** weighted `+`/`−`/scale/`⊙` (Hadamard) as
  free functions; an optional thin `WeightedT3`/`WeightedTangent` container for operator sugar; the
  **uniform** mirror (must reconcile with the boolean rank masks — the uniform layer deliberately chose
  boolean prefix masks over float edge-weights); the **Grasedyck–Kramer `SingularValueRegularizer`**
  (consumes this layer via `absorb`; see the reg design note §11).
- **Out of scope:** external/physical-mode weights (weights on the `Nᵢ` output legs). The refactor has
  made clear the external edges are of a *different character* than the internal edges and should be
  treated separately; the old code's `shape_weights` family is dropped (§2, §8).

## 2. The math — edges, and why two classes

A T3 as a tensor network has exactly two kinds of **internal** edges:
- **Tucker-rank edges** `nᵢ` — between Tucker factor `Uᵢ` and TT core `Gᵢ` (d of them).
- **TT-bond edges** `rᵢ` — between neighbouring TT cores (d+1, with `r₀=r_d=1` trivial).

So a **plain-`TuckerTensorTrain` weight is `(tucker[d], tt[d+1])`** — one diagonal vector per edge. This
is exactly the old `EdgeVectors` *and* exactly the shape `t3svd` returns (`tucker_svals[d]`,
`tt_svals[d+1]`) — the unifying fact of §7.

A **frame / tangent is richer.** A frame-variations pair does **not** represent one T3 — it represents a
family of **d T3s**, each a frame surrounding one variation core, and the tangent is their **sum**
(`Σᵢ termᵢ`). For `termᵢ`, the edges *left* of the variation are left-orthogonal (`P`, bonds `rL`) and the
edges *right* are right-orthogonal (`Q`, bonds `rR`); above/below are the Tucker up/down cores. So the
four rank families are not a labelling choice — they are *which side of the variation an edge is on*, and
the **shapes forbid mixing** (`rLₖ ≠ rRₖ` in general; a left diagonal literally will not fit on a right
bond). The four families:

| family | rank | weights (frame + variation cores) |
|---|---|---|
| **up** | `nU` (retained Tucker) | `U`'s up-leg, `P`/`Q`'s physical leg, `H`'s physical leg |
| **down** | `nD` (Tucker complement) | `O`'s physical leg, `V`'s row-leg |
| **left** | `rL` (left TT bonds) | `P`'s bonds, `O`'s left bond, `H`'s left bond |
| **right** | `rR` (right TT bonds) | `Q`'s bonds, `O`'s right bond, `H`'s right bond |

**Decision: two classes** — `T3Weights` `(tucker[d], tt[d+1])` and `T3FrameWeights`
`(up, down, left, right)`. A plain T3 genuinely has no "down" (no complement) and no left/right split (one
gauge); forcing it into four families would need `down=None, left=right`, which is uglier than two honest
types.

## 3. The concatenate / kronecker duality (verified)

The two ways to combine two *weighted tensors* map onto the two elementary operations on the *per-edge
diagonals*:

| tensor op | cores | ranks | **weights** |
|---|---|---|---|
| `A + B` | block-diagonal (direct sum) | **add** | **`concatenate`** (`[wᴬ ; wᴮ]` = diag of `D_A ⊕ D_B`) |
| `A ⊙ B` (Hadamard) | Kronecker on internal legs | **multiply** | **`kronecker`** (`wᴬ ⊗ wᴮ` = diag of `D_A ⊗ D_B`) |

`concatenate ↔ ⊕ ↔ +` and `kronecker ↔ ⊗ ↔ ⊙`. **Verified numerically (2026-07-14, rel err 1.2e-15):**
for the Hadamard product (product ranks multiply; represents the elementwise product of the represented
tensors), **both** weight families Kronecker-multiply edge-by-edge — `sᶜ = sᴬ ⊗ sᴮ`, `tᶜ = tᴬ ⊗ tᴮ` —
using per-edge `np.kron`. The product cores are `Uᶜ[(α,β),x] = Uᴬ[α,x]Uᴮ[β,x]` (physical output shared,
Tucker rank Kronecker) and `Gᶜ[(a,b),(i,j),(a',b')] = Gᴬ[a,i,a']Gᴮ[b,j,b']` (all three internal legs
Kronecker). **Consistency requirement:** `kronecker` on the weights and the Hadamard core-combine must use
the *same* A-major index pairing, i.e. literally `np.kron(wᴬ, wᴮ)` — mismatched ordering silently
corrupts (hit exactly this mid-verification). Boundaries are free for `⊙` (`kron` of two length-1 vectors
is length-1, so trivial bonds stay trivial); `+` needs a tail-squash afterward (`1+1=2` on the boundary
bonds → the old `squash_tails`, which lives in the future `weighted_add`, not in `concatenate` itself).

This duality is the strongest signal the lightweight per-edge-vector class is the right abstraction: the
weighted objects are a **closed algebra** under `+`, `⊙`, scaling, `inner`, `norm`, all built from
`concatenate`/`kronecker`/`absorb`.

## 4. The classes and functions

**Classes (frozen dataclasses, registered pytrees; mirror the existing `T3Frame`/`EdgeVectors` style):**
- **`T3Weights`** — `(tucker_weights [len d, (nᵢ,)], tt_weights [len d+1, (rᵢ,)])`. Pairs with
  `TuckerTensorTrain`.
- **`T3FrameWeights`** — `(up, down, left, right)`, each a per-edge diagonal sequence. Pairs with
  `T3Frame` / `T3Tangent`.

Both carry: `validate` (internal shape consistency, structural → always raises), `ranks`/`stack_shape`,
`stack`/`unstack`/`reverse`, elementwise (`reciprocal`, `sqrt`, scale), **`concatenate`**, **`kronecker`**.

**Functions (combine weights with objects — free functions):**
- **`absorb`** — contract the diagonals into the cores (side-conventions decided inline; backend/frontend split §4a).
  - T3: `absorb(x: TuckerTensorTrain, W: T3Weights) -> TuckerTensorTrain` (shape-preserving; the
    represented tensor is the fully-weighted network). **TT weights absorbed leftward** (boundary `r₀`
    rightward into `G₀`); **Tucker weights into the Tucker cores.**
  - Frame/tangent: **SUPERSEDED — see the header + §6.** *This bullet records the original sketch: absorb
    into the four **frame** cores (up→`U`, down→`O`, left→`P`, right→`Q`), variations untouched, with the
    weighted frame deliberately non-orthonormal. **That is not what shipped.*** The design changed mid-build
    to the **metric-on-variations** (Approach-1): `absorb(variations, W: T3FrameWeights) -> T3Variations`
    weights the **variation** cores — **down→`V`, up/left/right→`H`** — and the **frame is left orthonormal
    and untouched**. Reason: the frame's `d+1`-th left/right cores are base-point padding ("not really part
    of the frame"), so there are only `d` natural tangent edges, and the metric reading is what
    Grasedyck–Kramer needs. The shipped code (`fv_absorb_weights`) and `docs/weighting.md` are correct; this
    §4 bullet and §6 below were left stale and are corrected here (2026-07-15).
- **`is_consistent_with(W, object) -> bool`** — non-raising shape predicate (mirrors `is_orthogonal`).
  `absorb` also validates and raises on mismatch (structural).
- **`from_t3svd`** — build a `T3Weights` from the `t3svd` singular values (§7), **the unmodified σ's** — the
  user inverts/sqrts as needed (inverse-σ weighting = `from_t3svd(x).reciprocal()`).
- **`weighted_norm` / `weighted_inner`** — `absorb`, then the plain norm/inner (operands must share physical
  **shape**; ranks/weights may differ).
  - T3: `weighted_norm(x, W) = t3_norm(absorb(x, W))`; `weighted_inner(xᴬ,Wᴬ,xᴮ,Wᴮ) = t3_inner(absorb …, absorb …)`.
  - Tangent: via the doubled-rank T3 path (§6).

## 4a. Backend/frontend split (decided 2026-07-14)

Governing principle (the razor): **"weight ⊗ object" operations are backend, co-located with the
object's other ops; "weight-alone" transforms are also backend helpers** (not frontend-inline) — because
even the "trivial" ones (`concatenate`/`kronecker`) hide non-obvious axis bookkeeping (*which* axis over a
multi-dim stack prefix; *which* of the four `U/D/L/R` families), exactly the thing a raw-`.data` user
should not have to reconstruct (razor: trivial-*but*-non-obvious → backend).

```
FRONTEND
  T3Weights        (tucker_tensor_train.py)      (tucker[d], tt[d+1])
  T3FrameWeights   (frame_variations_format.py)  (up, down, left, right)
    · validate / ranks / stack / unstack / reverse / reciprocal / sqrt / concatenate / kronecker
      (concatenate/kronecker/… delegate to the backend), from_t3svd classmethod, registered pytrees
  free fns: absorb, weighted_norm, weighted_inner, is_consistent   (dispatch on type → backend → re-wrap)

BACKEND
  t3_operations.py :  t3_absorb_weights · t3_concatenate_weights · t3_kronecker_weights
                      t3_weighted_norm · t3_weighted_inner · t3_weights_consistent
  fv_operations.py :  fv_absorb_weights (weights the 4 frame cores) ·
                      fv_concatenate_weights · fv_kronecker_weights · fv_weights_consistent
  tv_operations.py :  tv_weighted_norm · tv_weighted_inner   (= fv_absorb_weights → tv_to_t3 → t3_norm/inner)
  RETIRE backend/wt3_operations.py  and  the wt3_ prefix
```

- **Frame absorb is `fv_`** (it acts on the frame alone — the weights pair with the *frame*), so it sits
  with the frame ops; a future weighted `fv_to_t3` (one-term converter) can call it with **no `tv_`
  back-import**. Only the tangent `weighted_norm`/`inner` are `tv_` (they need the variations via
  `tv_to_t3`). Import direction stays `tv_ → fv_ → t3_`, acyclic.
- **Prefix follows the module** (`t3_`/`fv_`/`tv_`), not a weighting-specific prefix — the concept lives
  distributed across the object modules + the two classes, maximally applying the "most relevant module"
  rule. `T3Weights`/`T3FrameWeights` + the `*_absorb_weights` / `*_weighted_*` names keep it discoverable.

## 4b. Stacking / batching (mirror the paired class; all machinery-1)

Batching is the library's most error-prone area ([`docs/batching_and_stacking.md`](docs/batching_and_stacking.md)).
The governing principle here (Nick): **mirror the batching of `T3Weights` on `TuckerTensorTrain`, and of
`T3FrameWeights` on `T3Frame`** — copy the paired class's `stack_shape` derivation, `validate`, and
`stack`/`unstack`, rather than inventing conventions.

**The key fact: the whole weighted layer is *machinery-1* — a single shared broadcast prefix (a leading
`'...'` einsum), never the grouped-block `contractions.py` machinery.** Because a weight *always shares
exactly its object's `C` (frame/core) stack, and nothing else*: there is **no `W`** (weights never touch
probe vectors) and **no independent second block**, so one leading `'...'` handles every op. (Confirmed
with Nick 2026-07-14.)

- **Weights carry `C` only — never `K`.** `T3FrameWeights` mirrors `T3Frame`, which is `C`-only (the
  *variations* carry `K+C`). A `K`-batch of tangents at one frame shares the **one** frame-weight;
  `tv_to_t3`'s frame-inner broadcast lifts the weighted frame `C → K+C` for free (the same `bcast2`/`bcast3`
  it already applies to the unweighted frame). So the weights ride along; we never manufacture a `K` axis.
- **Shapes:** `T3Weights` — `tucker_weights` `C+(nᵢ,)`, `tt_weights` `C+(rᵢ,)`, `stack_shape =
  tucker_weights[0].shape[:-1]`. `T3FrameWeights` — four families `C+(rank,)`, `stack_shape` from
  `up[0].shape[:-1]`. `validate` requires a **uniform `C`** (the frontend-dataclass asymmetry, batching doc
  §5) — same as `TuckerTensorTrain`/`T3Frame`.
- **`absorb`** is `'...i,...iaj->...iaj'`-style (leading `'...'`, weight and core share `C`). The weighted
  frame from `fv_absorb_weights` stays at `C` (lazy); `tv_to_t3` materializes the `C→K+C` broadcast when it
  builds the validated doubled-rank `TuckerTensorTrain` (batching doc §5: bare-tuple → lazy; validated-class
  → materialize).
- **Stacked `kronecker` gotcha:** it is a **last-axis outer-product-then-reshape that broadcasts the shared
  `C`** — `(wᴬ[...,:,None] * wᴮ[...,None,:]).reshape(C + (rᴬ·rᴮ,))` — **not** `np.kron`, which would
  Kronecker the `C` axes too. This "which axis, broadcast `C`" knowledge is exactly why the
  `*_kronecker_weights` helpers are backend, not frontend-inline.
- **Verification discipline:** test every op against the dense oracle with a **non-trivial, non-square,
  multi-axis stack** (e.g. `stack_shape=(2,3)`) — that is what actually exposes an axis transposition or a
  `kron`-over-the-wrong-axis; a scalar/no-stack test would not.

## 5. What we are NOT building — and the guarantee we are not locked out

**No `WeightedTuckerTensorTrain` / `WeightedTangent` wrapper as the substrate.** The old design inverted
the architecture: it made the *container* the substrate and hung every operation off it as a method
(669 lines, a full arithmetic mirror). We invert it back: **the substrate is the weight classes + `absorb`;
everything else is a function of `(object, weights)`.**

**Guarantee (Nick's central concern — supporting weighted linear algebra later):** every weighted
operation reduces to one of two moves the substrate already provides —

| weighted op | reduces to | needs |
|---|---|---|
| scale / negate | scale `x0` (or a weight) | — |
| **add / sub** | direct-sum cores + **`concatenate`** weights (+ tail-squash) | `concatenate` |
| **inner / norm** | `absorb` both → plain `inner`/`norm` | `absorb` |
| **Hadamard `⊙`** | Kronecker-combine cores + **`kronecker`** weights | `kronecker` |
| apply / entries / probe | `absorb` → plain op | `absorb` |

So each future op is a **pure addition** (a new free function), never a refactor of the classes. If
operator ergonomics (`a + b`, `a.norm()`) are later wanted, a *thin* `WeightedT3` container that forwards
to these functions is a ~20-line optional add — sugar over the substrate, not the substrate. Decide it
then; zero cost now.

## 6. Weighted-tangent norm — SUPERSEDED (the doubled-rank T3 path was NOT what shipped)

> **⚠️ This whole section records the abandoned first design.** It assumed weights are absorbed into the
> **frame** cores (`U/O/P/Q`), which breaks orthonormality and therefore forces the doubled-rank `tv_to_t3`
> detour below. The build instead adopted the **metric-on-variations** (see the header + §4): weights go into
> `V`/`H`, the **frame stays orthonormal**, and so the weighted norm is simply the **corewise coordinate
> norm of the weight-absorbed variations** — `corewise_stack_norm(fv_absorb_weights(variations, W))`, which
> is `O(ranks)` and needs no `tv_to_t3`, no doubled-rank tensor, and no orthonormality repair. The shipped
> functions are `fv_weighted_norm` / `fv_weighted_inner` in `fv_operations` (**not** `tv_operations` — the
> `tv_*` twins in §4a were never needed). `docs/weighting.md` + `docs/frame_variations.md` are the correct
> account. Kept only as the record of the road not taken. *(Corrected 2026-07-15; the §4b stacking analysis
> below is still accurate and, notably, states the **correct** stack rule — "weights carry `C` only, never
> `K`; `T3FrameWeights` mirrors `T3Frame`" — which the implementation then got wrong by moving the stack to
> the variations along with the absorption target. Fixed in the uniform-weighting build's S0;
> `dev/uniform_weighting_design.md` §8.6.)*

Absorbing weights into a tangent breaks the frame's orthonormality, so the cheap `Σ‖variationᵢ‖²`
shortcut and `MANIFOLD.norm`/`inner` are **invalid** (and would trip the safe-mode orthogonality
precondition). The correct, and still **O(ranks)**, path (Nick):

```
weighted_norm(tangent, W)   = t3_norm ( tv_to_t3( absorb(tangent, W), include_shift=False ) )
weighted_inner(A,Wᴬ,B,Wᴮ)   = t3_inner( tv_to_t3(absorb(A,Wᴬ)), tv_to_t3(absorb(B,Wᴮ)) )
```

`T3Tangent.to_t3` (→ `tv_operations.tv_to_t3`, the doubled-rank embedding of Appendix A.3.1, eqs 50–53,
Fig 20) is **already shipped and purely structural** — Tucker cores become `[Uᵢ; Vᵢ]` (concatenate on the
rank axis), TT cores form the block-bidiagonal embedding, built entirely from `concatenate`/`broadcast_to`,
with **no orthogonalization and no orthonormality assumption** (verified by reading `tv_to_t3`). So it runs
verbatim on the weighted non-orthonormal cores.

Two consequences: (a) the frame `absorb` weights **only the four frame cores** `U/O/P/Q` (each its own
family; `V`/`H` untouched — §4), and `tv_to_t3` then assembles the correctly weighted doubled tensor
because the up/down and left/right blocks sit on opposite sides of each doubled edge; (b) the doubled-rank
T3's weights are the **concatenations** of the frame families
(`[up;down]` per Tucker edge, `[left;right]` per TT bond) — a nice internal consistency, and a free **test
oracle** (weighted-tangent-norm-via-this-path must equal the dense norm of the weighted tangent tensor).

## 7. Relationship to `t3svd` (the canonical weight)

`t3svd`'s singular-value output `(tucker_svals[d], tt_svals[d+1])` **is** a `T3Weights` — same shape. So
`from_t3svd(x)` hands back the natural weight object, and inverse-σ (Grasedyck–Kramer) weighting is
`from_t3svd(x).reciprocal()`. This is the elegant core: singular values are the canonical weight, and the
data format was chosen to make that literal.

## 8. Relationship to the Grasedyck–Kramer regularizer (this layer is its primitive)

The deferred `SingularValueRegularizer` (reg design note §11) is a **consumer**, not a competitor. If
`M = W²`, then `⟨x, Mx⟩ = ‖absorb(x, W)‖²`, `Mx = absorb(x, W²)` — so **absorb-into-tangent is exactly the
primitive the GK regularizer needs**; the regularizer just builds `W` from the frame's singular values and
calls `absorb`. Reviving this layer directly enables the GK reg. Build the primitive here; leave the
regularizer as the next consumer.

## 9. Decisions (settled 2026-07-14) + the one deferred question

1. **`absorb` side-convention — DECIDED (§4/§4b).** T3: TT weights leftward (`r₀` rightward into `G₀`),
   Tucker weights into the Tucker cores. Frame: up→`U`, down→`O`, left→`P`, right→`Q` (variations untouched).
2. **`from_t3svd` / weight convention — DECIDED.** `from_t3svd` returns the **unmodified singular values**;
   the caller reciprocates/sqrts. The weight is the diagonal you *insert*, and `weighted_norm` squares it
   (so `diag(1/σ)` penalises by `1/σ²`) — documented on `weighted_norm`.
3. **Naming — DECIDED (§4a).** Frontend `T3Weights` / `T3FrameWeights`. Backend prefix follows the module
   (`t3_`/`fv_`/`tv_`); `*_absorb_weights` / `*_{concatenate,kronecker}_weights` / `*_weighted_{norm,inner}`
   / `*_weights_consistent`. `wt3_` retired.
4. **`weighted_inner` operand compatibility — DECIDED.** Operands must share physical **shape** (same
   ambient tensor space); ranks/weights may differ. Structural check.
5. **Uniform mirror — DEFERRED, but the shape is decided.** The **uniform weight objects carry boolean
   masks** (like `UniformTuckerTensorTrain`: masks as static `aux_data`, value-based hash/eq), and the masks
   are checked for compatibility when operating with uniform objects — i.e. a uniform weight is "the ragged
   weight, padded for performance." Reconciling the float edge-weights with the boolean prefix masks is the
   real work (`dev/archive/uniform_fix_plan.md` cautions on it); scheduled after the ragged layer.

## 10. Verification strategy

Against dense ground truth throughout (house rule):
- **T3 `absorb`**: `absorb(x, W).to_dense()` == the hand-einsum weighted network (the old class doctest's
  `'ix,jy,kz,aib,bjc,ckd,i,j,k,a,b,c,d->xyz'` is the d=3 oracle).
- **`concatenate` / `kronecker`**: build `A+B` / `A⊙B` from cores, check `absorb(combined).to_dense()`
  equals dense `+` / `⊙` of the two weighted tensors (the `⊙` check is the §3 verification, promoted to a
  test).
- **`from_t3svd`**: `absorb(x, from_t3svd(x)?)` and the `.reciprocal()` round-trip (with the §9.2
  convention pinned).
- **Weighted tangent norm**: the §6 oracle (via-`to_t3` == dense norm of the weighted tangent tensor).
- **jit dispatch**: infer jax from inputs (no `use_jax` threading — the parked code's old pattern is
  dropped); add the absorb/norm paths to `test_dispatch`.

## 11. Implementation slices (ragged first)

1. **S1 — `T3Weights` + T3 `absorb` + `is_consistent_with` + `from_t3svd`.** The data format + the core
   bridge. Backend `t3_*` on raw tuples; frontend class + free functions. Mine the old
   `contract_edge_vectors_into_t3` for the axis bookkeeping (drop `use_jax`; infer dispatch). Tests vs the
   dense oracle.
2. **S2 — T3 weighted algebra: `weighted_norm`, `weighted_inner`, `concatenate`, `kronecker`.** The §3
   duality tests (promote the `⊙` verification). Elementwise (`reciprocal`, `sqrt`).
3. **S3 — `T3FrameWeights` + `fv_absorb_weights` + `tv_weighted_norm`/`tv_weighted_inner`.** Rewrite the
   parked `absorb_weights_into_tangent_cores` → `fv_absorb_weights` (correct `(U,O,P,Q)` order, **frame-only**:
   up→`U`, down→`O`, left→`P`, right→`Q`; variations untouched). Norm/inner via the §6 `tv_to_t3` path in
   `tv_operations` (raw tuples, no `T3Tangent` re-wrap). Tests vs the §6 oracle.
4. **S4 — retire the old layer.** Delete `weighted_tucker_tensor_train.py`, `backend/wt3_operations.py`,
   the parked `absorb_weights_into_tangent_cores`, and the duplicate `wt3_squash_tails` in `t3_operations.py`
   (once the new layer subsumes them). Remove the `conf.py` autoapi ignores and the cordon warnings.
5. **S5 — docs.** User-facing (a weighting section) + contributor rationale (the concat/kron duality, the
   doubled-rank norm path, the two-classes reasoning).

**Deferred slices (reachable, not scheduled):** weighted `+`/`−`/scale/`⊙` free functions (+ tail-squash) →
optional thin container; the uniform mirror; the GK `SingularValueRegularizer`.

## 12. Retirement / mining map (old → new)

- `EdgeVectors` (tucker[d], tt[d+1]) → **`T3Weights`** (same format; the right idea). Keep `concatenate`,
  `reverse`, `stack`/`unstack`; add `kronecker`, `reciprocal`, `sqrt`; drop `use_jax`.
- `WeightedTuckerTensorTrain` (the heavy wrapper) → **deleted**; replaced by free functions + (later)
  optional thin container.
- `contract_edge_vectors_into_t3` → **`absorb`** (T3). Mine the axis einsums.
- `absorb_weights_into_tangent_cores` (broken: old `(U,P,Q,O)` order, 5-vs-4 unpack bug, `shape_weights`) →
  **`fv_absorb_weights`** in `fv_operations` (correct `(U,O,P,Q)` order, four families, **frame-only**, no
  external weights). Mine the per-core einsums.
- `wt3_squash_tails` (two copies) → folded into the future `weighted_add`; both copies deleted.
- `wt3_apply`/`wt3_entries`/`wt3_probe`/`wt3_inner_product`/`wt3_add`/`wt3_sub` → not ported as such;
  reachable as `absorb` + plain op / the deferred weighted arithmetic.
