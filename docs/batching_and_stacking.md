# Batching and stacking in T3Toolbox — the complete reference

> Batch/stack axes — `stack_shape`, the `C`/`W`/`K` blocks, `K`-stacked (tangent-stacked)
> tangents, `vmap`/`jit` over these objects — are the most subtle part of the library. This file
> is the source of truth for *what the conventions are* and *why*; the "Start here" section is
> enough for most use. (Rules for *extending* the batching machinery, and the decision history,
> live in [`contributor/batching_internals.md`](contributor/batching_internals.md).)

---

## Start here — the mental model in one screen

*New to stacking here? Read this section; it is enough to **use** the library. The numbered
sections below are the full reference.*

**Stacking = leading batch axes on the cores, so one object holds many.** Every `TuckerTensorTrain` /
`T3Frame` / `T3Variations` stores its cores as `core.shape == stack_shape + (tensor/rank axes)`. With
`stack_shape = (2, 3)` you carry a `2×3` grid of T3s/bases/variations in one object, and operations
vectorize over them.

**There are three kinds of batch — three "blocks" — that batch different things, on different arrays.**
Keeping them straight is the whole game; conflating them is the #1 source of confusion:

- **`C` — the frame/core stack.** A batch of whole T3s / base points on the manifold. Lives on **every
  core** (it *is* `stack_shape`). Mnemonic: on every **C**ore.
- **`W` — the probe stack.** A batch of probe-vector sets. Lives on the **probe vectors `ww` only**, not
  the cores. Mnemonic: the **w** vectors.
- **`K` — the tangent stack.** A batch of tangent vectors attached at the *same* base point. Lives on
  the **variation cores only**. Mnemonic: the **k** tangent vectors at each frame.

One object can carry more than one block at once. Order is always **frame-inner**: `W + K + C + (axes)`
(`C` innermost, adjacent to the indices; `W`/`K` outermost). Why that order? See §3 — it is exactly
what lets a shared frame broadcast over its batch for free.

### One concrete example (the picture that makes it click)

Take `d = 3` modes, a batch of **2 base points** (`C = (2,)`), with **3 tangent vectors at each**
(`K = (3,)`), probed by **4 probe-sets** (`W = (4,)`). Then every array's shape is:

| Array | Blocks | Shape |
|---|---|---|
| frame Tucker core `Uᵢ` (`T3Frame`) | `C` | `(2,) + (nᵢ, Nᵢ)` |
| frame TT core `Pᵢ` (`T3Frame`) | `C` | `(2,) + (rᵢ, nᵢ, rᵢ₊₁)` |
| variation core `δUᵢ` (`T3Variations`) | `K + C` | `(3, 2) + (nᵢ, Nᵢ)` |
| probe vector `wᵢ` (`ww`) | `W` | `(4,) + (Nᵢ,)` |
| forward probe `zᵢ` (`tangent.probe(ww)`) | `W + K + C` | `(4, 3, 2) + (Nᵢ,)` |

Read it as: *2 base points, each with its own frame; 3 tangent vectors attached at each; probe every
one against 4 probe-sets → `4 × 3 × 2` probe results per mode.* The frame (`C`) is **shared**
across the `K` tangents at it — never copied; frame-inner broadcasting (§3) handles that for free.

> ⚠️ The word **"stack" itself means three different things** in this codebase (`stack_shape`; the
> tree↔object machinery in `stacking.py`; the uniform supercore representation). When confused, first
> decide *which*. See §1.

### Shape-notation legend (how to read the shapes in the code)

The codebase annotates shapes in trailing comments and encodes them in names. The whole scheme:

- A core's shape is written **`stack_shape + (tensor/rank axes)`**, e.g. `Bi.shape = stack_shape + (nᵢ, Nᵢ)`.
- **Capital letters are grouped batch blocks**, each standing for **zero or more** axes: **`C`** frame/core
  stack, **`W`** probe stack, **`K`** tangent stack. An empty block contributes no axes (so the same code
  handles no-stack and stacked).
- **lowercase letters are single axes** — tensor modes / ranks (`n`, `N`, `r`) or contraction legs
  (`a`, `i`, `b`, `o`, `j`).
- **Order is frame-inner: `W + K + C + (axes)`** (probe outer, tangent middle, frame inner).
- **Body locals suffix their layout:** `mu_WCa` is an array with axes `W + C + (a,)`.
- **Grouped contractions are einsum strings:** `contract('WCa,Caib,WCi->WCb', ...)` reads as a
  per-operand block+leg signature (operand 1 = `W+C+(a,)`; operand 2 = a `C`-only core with legs
  `a,i,b`; operand 3 = `W+C+(i,)`; output frame-inner `W+C+(b,)`).
- A leading **`d`** in a subscripts term (`dWCa`) is the uniform layer's supercore/derivative axis
  (see `docs/uniform_supercore_layout.md`).

### Glossary

| Term | Meaning |
|---|---|
| **`stack_shape`** | the leading batch axes shared by all of an object's cores; *is* the frame/core stack `C`. |
| **frame / core stack `C`** | a batch of T3s / base points, on every core (`= stack_shape`). |
| **probe stack `W`** | a batch of probe-vector sets, on `ww` only. |
| **tangent stack `K`** | a batch of tangent vectors at one frame, on the variation cores only; `T3Tangent.tangent_stack_shape`. |
| **frame-inner** | the ordering rule: `C` innermost, `W`/`K` outermost (`W + K + C`). |
| **`frame_stack_shape` / `tangent_stack_shape`** | a `T3Tangent`'s `C` and `K` parts, *derived* from the (frame, variations) pairing (§6), not stored. |
| **heterogeneous stack** | one T3 whose cores have different-but-broadcastable stacks (frame `C`, variation `K+C`). First-class in the backend (§5). |
| **"the split is recovered"** | `C`/`K`/`W` lengths are read off operand shapes wherever the shapes can pin them; `contract` demands an explicit `len_<G>=` exactly when they cannot (§4). The tangent `K`/`C` split is derived from the (frame, variations) pairing (§6). |
| **`sum_over_probes`** | transpose flag (§11): `False` (default, **primary**) keeps the probe stack `W` as an output stack — one tangent/tensor per probe; `True` sums `W` (`= Σ_W` of `False`) for the optimization `Jᵀr`. |
| **ragged / uniform / weighted** | the three representations. **ragged** (tuples of arrays) is the default; **uniform** (supercores + masks) is the jit/GPU mirror of the whole stack (`docs/uniform_equivalence_contract.md`); **weighted** is diagonal edge weights on the internal edges — a data format + `absorb` into cores, not a separate object layer (`docs/weighting.md`). |

> The block letters `W`/`K`/`C` are deliberately disjoint from the core/variation symbols
> (`U`,`P`,`Q`,`O`,`G`,`H`,`B`) and from `Jᵀ`/tensor/`T3`, so a shape comment is never ambiguous.

---

## 0. TL;DR

- The word **"stack"** means **three different things** in this codebase. Keep them apart (§1).
- There are **three distinct batch *blocks*** — `C` (frame/core stack), `W` (probe stack), `K` (tangent
  stack). They batch *different things* and live on *different subsets* of the operands (§2).
- Library-wide convention is **frame-inner**: the core stack `C` is **innermost** (adjacent to the
  tensor indices); the extra stacks `W`, `K` are **outermost**. Orders: `W+C`, `K+C`, `W+K+C` (§3).
- There are **two machineries** for batch axes: a leading `...` in einsum (for *one* broadcastable
  prefix) and the **grouped-einsum interpreter** `contractions.contract` (for *two* independent
  blocks on different operand subsets) (§4).
- A T3 may have **heterogeneous-but-broadcastable** core stacks (e.g. frame cores `C`, one variation
  core `K+C`). This is *first-class* in the backend; `t3_broadcast_to_common_stack` materializes it
  when a uniform-stack object is required (§5).
- Backend functions accept raw tuples and tolerate heterogeneous stacks; **frontend dataclasses
  (`validate`) require a uniform stack**. That asymmetry explains where broadcasting is lazy vs
  materialized (§5).
- The `K`/`C` split is **not stored** — it is recovered from the `(frame, variations)` pairing (§6).
- `jax` pytree registration makes `T3Tangent` `vmap`/`jit`-able; **the frame and variations are both
  leaves** (the same-frame guard is numerical, not identity), so the frame flows as traced data with no
  per-frame recompile (§7).

---

## 1. "Stacking" means three different things

1. **`stack_shape` — leading batch axes on ONE object's cores.** Every core of a `TuckerTensorTrain`
   / `T3Frame` / `T3Variations` is `core.shape == stack_shape + (tensor/rank axes)`. A single leading
   `'...'` in an einsum rides these axes along for free. **This is the common case** and the meaning
   of "stacked" 95% of the time. Caveat: a plain `'...'` carries exactly **one** shared/broadcast
   prefix.

2. **`backend/stacking.py` — converting a Python *tree of separate objects* ↔ *one stacked object*.**
   `stack(tree, axes)` / `unstack(S, axes)` / `tree_zip` / `apply_func_to_leaf_subtrees`. This is its
   own tree machinery — **not** jax pytrees, and not the same as meaning (1). It is what
   `T3Tangent.unstack_*` / `stack_*`, `T3Frame.stack`, etc. are built on.

3. **The uniform supercore (`ut3_*`, `ufv_*`, `uniform_*`) — a separate representation.** One stacked
   supercore array + boolean rank masks, built for `jax.lax.scan` vectorization, GPU efficiency, and
   compile-once `jit` — a faster mirror of the ragged layer, optimizers included. See
   `docs/uniform_equivalence_contract.md` and the other `uniform_*` notes.

When something is confusing, first ask **which of these three** you are dealing with.

---

## 2. The three batch *blocks*: `C`, `W`, `K`

Within meaning (1), there are **three semantically distinct things one might batch**. They are not
interchangeable, and a single object can carry more than one at once.

| Block | Name | Batches… | Lives on… | Appears in |
|------|------|----------|-----------|------------|
| **`C`** | frame / core stack | a batch of **T3 objects / base points** | **the cores** (every core) | everywhere; `= stack_shape` of a `TuckerTensorTrain`/`T3Frame` |
| **`W`** | probe stack | a batch of **probe vectors** | **the probe vectors `ww` only** (not the cores) | probing (`probe_*`), `apply`/`entries` (there `W` = the vec/index batch) |
| **`K`** | tangent stack | a batch of **tangent vectors sharing one frame** | **the variations only** (not the frame cores) | `T3Tangent` (`tangent_stack_shape`) |

Key facts:

- **`C` is shared across `W` and `K`.** A probe stack `W` of probe vectors is applied to a `C`-batch
  of T3s → output carries both. A `K`-batch of tangents shares one `C`-batch of base points.
- **`W` and `K` live on *different operands* than `C`.** `W` is only on `ww`; `K` is only on the
  variation cores; `C` is only on the cores. This is exactly why a single `'...'` is not enough (§4):
  `'...'` broadcasts a *shared* prefix, but `W`/`K` are present on a *subset* of operands.
- In the **transpose** of probing, the probe stack and the tangent stack coincide: `K == W` (each
  probe residual becomes one tangent) — when `sum_over_probes=False`; setting it `True` sums `W` away.
  See §11 for the full story on transposes and `sum_over_probes`.

**Concrete shapes (frame-inner, see §3):**
- a T3 core: `C + (rL, n, rR)` (tt core) or `C + (n, N)` (tucker core)
- a probe vector `ww[i]`: `W + (N_i,)`
- a `T3Variations` core: `K + C + (rL, n, rR)` or `K + C + (n, N)`
- a forward probe output `zz[i]`: `W + C + (N_i,)`  (= `t3_probe` output `W + C`)
- a transpose-probe non-summed output: a `T3Tangent` whose `K == W`, i.e. variations `K + C`

---

## 3. The frame-inner convention (and *why*)

**Rule (library-wide): order batch axes by how core-bound they are. The frame/core stack `C` is
INNERMOST (adjacent to the indices). Extra stacks — probe `W`, tangent `K` — are OUTERMOST.**

So the canonical orderings are:
- variation cores: `K + C + core`
- forward probe / `apply` / `entries`: `W + C`
- full probe of a `K`-stacked tangent: `W + K + C`
- transpose non-summed: `K + C` (with `K == W`)

### Why this exact order — the broadcasting mechanism

This is **the** reason for the convention, and the thing most worth internalizing.

The bulk of the tangent/T3 operations combine a **frame** (carrying only `C`) with a **variation**
(carrying `K + C`, or `W + C`) using plain `'...'` einsums (`to_dense`, the gauge projections,
`tv_project_t3_onto_tangent_space`, corewise linalg). numpy/jax `'...'` broadcasting is **right-aligned**.

- With **frame-inner** (`C` innermost), `C` is the **trailing suffix** of `K + C`. So a `C`-stacked frame
  core broadcasts cleanly under a `K + C`-stacked variation core: the trailing `C` axes align, and the
  leading `K` axes are replicated **for free** — exactly the semantics "the one base point is shared
  by all `K` tangent vectors at it." Same for `W + C`.
- With frame-*outer* (`C + K`), the trailing axes would be `K`, `C` would be the prefix, and the free
  right-aligned broadcast would mismatch. You would have to insert explicit transposes/reshapes at
  every boundary.

The **custom grouped-block contractions** (§4) are *flops-neutral* to block order (they reshape each
block to one flat axis regardless), so they are made to follow the same frame-inner order purely for
**one consistent layout with no boundary-transpose copies**. That is the whole justification —
broadcasting determines the order; the contractions just conform.

> Mnemonic: **"a frame broadcasts over its batch only when the frame axes are on the inside."**

---

## 4. The two machineries for batch axes

### (a) One broadcastable prefix → a leading `'...'` in einsum

If all operands share **one** batch prefix — or one operand's prefix is a frame that *broadcasts*
under another's via frame-inner (§3) — a leading `'...'` handles it for free. This covers `to_dense`,
the gauge projections, `project_*`, `corewise_*`, `tv_to_dense/_t3`, orthogonalization, etc.

This is why most of the library "just works" with stacking: you write the einsum with `'...'` and
negative axes, and `C`/`K`/`W` ride along.

### (b) Two independent blocks on DIFFERENT operand subsets → `contractions.contract`

When **two** independent batch blocks live on **different subsets** of the operands, a single `'...'`
**cannot** express it: right-aligned broadcasting would force the two blocks to align, but they are on
different operands and must stay independent. **The canonical case is probing:** the core/frame stack
`C` (on the cores) and the probe stack `W` (on the probe vectors only).

So probing is built on the **grouped-einsum interpreter** `backend.contractions.contract`:

- The subscripts are a standard einsum string with one extension: an **UPPERCASE letter is a GROUP
  of zero or more axes**, lowercase letters are single axes — e.g.
  `contract('WCa,Caib,WCi->WCb', mu, P, xi)`. An empty group contributes no axes, so the *same
  call* handles no-stack / one-stack / both-stack.
- Each group's axis count is **solved from the operand ndims** (a small exact linear system).
  Whether the ndims *can* pin a split is a property of the subscripts alone — so a call site either
  never needs help or always does, and in the latter case `contract` raises, naming exactly the
  `len_<G>=` keyword to pass (e.g. `contract('WCo,WCa->Cao', z, eta, len_W=1)`). Groups that always
  travel together (`'WKCi,Cio->WKCo'` — nothing separates `W` from `K` anywhere) need no split at
  all: only their combined total matters, and any supplied split is verified but changes nothing.
- The groups then expand into ordinary single-axis letters and **one einsum runs on the operands
  exactly as given — no reshape, no data movement**.
- **Output order is frame-inner:** `W` (and `K`) outer, `C` inner. E.g. `'...->WCb'` returns
  `W_shape + C_shape + (b,)`.

A *third* independent block (forward-probing a `K`-stacked tangent — `W` probes, `K` tangents, `C`
frame) is just a third capital letter in the string: `contract('WKCa,Caib,WCi->WKCb', ...)`.

**Decision rule:** if your two batches are on the *same* operands → `'...'`. If they are on *different*
operand subsets and must remain independent → `contract`. (Full usage guide, with examples and the
guarantees spelled out: [`grouped_contractions`](grouped_contractions.md).)

**Sharding for multi-GPU: any axis of any group shards freely.** `contract` never reshapes, so every
group sub-axis is an honest einsum axis; sharding a batch/free axis costs zero collectives, and
sharding a summed axis costs exactly the all-reduce the mathematics requires. (This is checked by
compiling every subscripts string in the library under virtual devices —
`tests/test_contractions_sharding.py`; the history of why it is checked, not just argued, is
contributor material: [`contributor/batching_internals.md`](contributor/batching_internals.md).)

---

## 5. Heterogeneous-but-broadcastable tuples, and the backend/frontend split

A subtle but pervasive pattern: a single T3's cores may have **different (but broadcastable) stacks**.
The archetype is a **tangent term**: the frame cores carry `C`, but one variation core carries `K + C`.
Per §3 this is a *deliberate, valid* layout — frame-inner makes `C` the suffix of `K + C`, so the
operation broadcasts the frame over `K`.

- **The `'...'`-einsum ops handle this for free** (gauge, `project_*`). They never materialize
  the broadcast — they keep the frame at `C` and let `'...'` replicate it. This is the *efficient*
  pattern (no `|K|` copies of the frame).
- **The one reshape-based primitive, `to_dense`, is broadcast-aware** via
  **`t3_broadcast_to_common_stack`** (`backend/t3_operations.py`): compute `np.broadcast_shapes` of all
  core stacks, `broadcast_to` each core up, then contract. No-op for a uniform-stack T3.

### Backend tuples vs frontend dataclasses — the validate asymmetry

This is *the* thing that decides where broadcasting is lazy vs materialized:

- **Backend functions take raw `(tucker_cores, tt_cores)` tuples and tolerate heterogeneous stacks.**
  Heterogeneous-but-broadcastable tuples are first-class there.
- **Frontend dataclasses validate a UNIFORM stack.** `TuckerTensorTrain.validate()` (and
  `T3Frame`/`T3Variations`) *hard-require* every core to share one `stack_shape`. So **any builder that
  produces a class instance must materialize the broadcast to uniform**:
  - `tv_to_t3` → must broadcast frame→`K+C` before concatenating the doubled-rank cores (its output
    *is* a validated `TuckerTensorTrain`; `[U_i ; V_i]` is one array).
  - frontend `bvf.fv_to_t3` → must `t3_broadcast_to_common_stack` the mixed-stack term before wrapping
    in `TuckerTensorTrain`.
  - whereas `to_dense` produces a *bare array*, so it can broadcast lazily inside the contraction.

> Rule of thumb: **bare-array output → broadcast lazily; validated-class-instance output → materialize
> the broadcast to a uniform stack.**

---

## 6. The `K`/`C` split is recovered, not stored

`T3Tangent` bundles a `T3Frame` (stack `C`) with a `T3Variations` (stack `K + C`). The split is
**derived**, never stored as a field (minimal dataclasses):

- `frame_stack_shape` (`C`) `= frame.stack_shape`
- `stack_shape` (`K + C`) `= variations.stack_shape`
- `tangent_stack_shape` (`K`) `=` the part of `variations.stack_shape` that *exceeds* `frame.stack_shape`
  — i.e. `C` is the **trailing suffix** of `K + C`, and `K` is the leading remainder.

`check_fv_pair` enforces exactly this: *`frame.stack_shape` must be the trailing (inner) suffix of
`variations.stack_shape`.* (It does **not** require equality — that was the pre-`K`-stack invariant.)

**`T3FrameWeights` is frame-like, and follows the same rule.** A tangent metric
([`weighting.md`](weighting.md)) is one metric *per base point*, so it carries `C` — not `K + C` — even
though it is *absorbed into* the variations. Where it batches and what it acts on are different questions;
conflating them is what put an over-strict `K + C` stack check on the weight predicate until it was fixed.
So pairing a weight with variations uses the trailing-suffix rule (`fv_weights_consistent`, blind to the
frame like the variations themselves), and `check_fw_pair` — the weight↔frame analog of `check_fv_pair` —
enforces the exact `weights.stack_shape == frame.stack_shape` wherever the frame is present.

Because the split is not recoverable from a *bare tree of objects*, the two-axis stack/unstack (§ below)
must name which stack it peels.

### Two-axis stack/unstack on `T3Tangent`

A single monolithic `stack`/`unstack` cannot faithfully invert two stacks (the `K`/`C` split isn't in
a bare tree). So `T3Tangent` has **two explicit pairs**, each peeling **one named stack**:

- `unstack_tangents` / `stack_tangents` — peel the **tangent stack `K`**. Yields a `K`-shaped tree of
  tangents that **share the frame** (same `T3Frame` *object* → same tangent space → mutually
  linalg-compatible). "For each vector within the frame." `stack_tangents` **guards** that all leaves
  share the same frame (the same-frame check, as in `inner`/`+`).
- `unstack_frame` / `stack_frame` — peel the **frame stack `C`**. Yields a `C`-shaped tree of
  single-base-point tangents at **distinct** bases (distinct tangent spaces). "For each frame."
  `stack_frame` places `C` **innermost** (variation stack → `K + C`), which needs *interior-axis*
  stacking that the component `T3Variations.stack` can't do — hence it is a real op, not user-assembled.

`T3Frame`/`T3Variations` keep their single plain `stack`/`unstack` (each is a single-stack object).
Backend functionals (`tv_unstack_tangent_stack`/`tv_stack_tangent_stack`/`tv_unstack_frame_stack`/
`tv_stack_frame_stack` in `tv_operations.py`) do the array/tree work; the `T3Tangent` methods are
thin wrappers doing the compatibility checks.

---

## 7. jax pytrees and `vmap` — how batching meets autodiff

The frozen dataclasses are registered jax pytrees (`if jax_available:` at the end of each module), so they
can be `jit`/`vmap`/`grad`-ed:

- `TuckerTensorTrain`, `T3Frame`, `T3Variations`: leaves = the cores (`x.data`), no aux.
- **`T3Tangent`: the frame AND the variations are leaves** (no aux). Both the fixed frame and the moving
  tangent vector flow as traced data. Consequences:
  - **`vmap` over a `T3Tangent` maps over *both* the frame and variation leaves.** For the
    batch-of-tangents-sharing-one-frame picture (`vmap`-over-`K`, frame fixed), close the frame over and map
    a function of the variations: `vmap(lambda v: f(T3Tangent(frame, v)))`. The `K`-stacked forward
    probe does not use `vmap` internally — it uses the genuine 3-block (`W`,`K`,`C`) contractions (§4):
    consistent with the `contract` interpreter (the blessed mechanism for independent blocks), no
    Python `K` loop on the numpy path, and low-level einsums fold into XLA at least as well as a `vmap`
    (which can add layout/transpose churn over a long function).
  - **The same-frame guard is NUMERICAL (`safety.frames_equal`), not object identity.** Two tangents
    share a tangent space iff their frame cores are *value-equal* (with an `is`-identity fast path); the
    check is safe-mode + eager-only (skips under `safety.unsafe()` / any trace). So it survives a jit
    round-trip (a reconstructed, value-equal frame passes instead of false-failing) and `inner`/`+` jit
    cleanly.
  - **Payoff: NO per-frame recompile.** Because the frame is a leaf (traced data, not a compile-time aux
    constant), a tangent or `GaussNewtonModel` crossing a `jit` boundary compiles **once across all
    bases** (`traces=1`) — you jit the frontend matvec directly. To grad *w.r.t. the variations only*,
    close the frame over; grad-w.r.t.-the-frame is also available (the frame is a leaf).
  - `T3Frame` / `T3Tangent` are `@dataclass(frozen=True, eq=False)`: value hash/eq on ndarray fields is
    ambiguous, so they stay identity-hashed — but the frame is a **leaf-bearing pytree node**, not
    `aux_data`.

- **The uniform layer does not `vmap` — by construction, and loudly.** A stacked
  `UniformTuckerTensorTrain` cannot be `vmap`-ed over its stack with *any* `in_axes`: axis 0 slices the
  **mode axis** `d` (the supercore layout is `(d,) + stack_shape + (...)`) and `validate` rejects the
  wreckage; axis 1 — the actual stack axis — *also* fails, because the rank masks ride as static
  `aux_data` with their own per-element stack axes and `vmap` cannot slice aux. Every route errors at
  validation; nothing convention-violating comes out. This is not a gap to fix: batching over uniform
  objects is what the native `C` stack with packed contractions is *for*. The one legitimate `vmap`
  pattern is at the raw-supercore level — map over the stack axis of the bare arrays and construct an
  unstacked object inside with the (shared) unstacked masks. Ragged objects `vmap` fine with default
  axes (their cores carry the stack leading).

`vmap`/`jit` invocation across the ops is checked cheaply in `tests/test_dispatch.py` (a `jit`-compile
of an op proves no hidden numpy — a stray `np.*` on a tracer raises).

---

## 8. Decoding the names (so you can read the code)

The naming scheme encodes axis layout — once you know it, the einsums read themselves.

- **Body locals** suffix the axis layout: `C_aib`, `mu_WCa`, `B0_b_j_c`, `xi_WCp`. Capitals = grouped
  index blocks (`C`/`W`/`K`), lowercase = single axes, a leading `d` = a stacked/derivative (uniform
  supercore) axis. `apply`/`entries` use the same `W` (vec/index stack) and `C` (core stack) as
  everywhere else.
- **Grouped contractions** are `contract('WCa,Caib,WCi->WCb', ...)` calls: read each term as a
  grouped-block einsum signature; output is frame-inner (`W` outer, `C` inner).
- **Paper ↔ code** (Appendix A of the T4S paper): `U`=up_tucker, `P`=left_tt, `Q`=right_tt,
  `O`=down_tt (called `down_tt_cores`, not "outer"); `δU`=tucker_variations (`V`), `δG`=tt_variations
  (`H`). The block letters `W`/`K`/`C` are **deliberately disjoint** from these core/variation symbols,
  so `V` here means only the Tucker variation and `G` only the TT core — **no overload**.

---

## 9. Gotchas (the ones that have actually bitten)

- **`corewise_dot` / `corewise_norm` collapse EVERY axis** (stacks included) to a scalar. To keep the
  stack (vectorized linalg), use `corewise.corewise_stack_dot(X, Y, n_stack)`.
- **Validity checkers and residuals are per-stack-element.** `is_orthogonal`, `is_gauged`, `is_consistent`,
  `is_left/right_orthogonal`, `allclose`, and the `*_residual` properties reduce over the *non-stack* axes
  and return an array of shape `stack_shape` — **a scalar when unstacked, an array when stacked** (stack
  elements can differ: e.g. `frame.is_orthogonal()` on a `(2,)` stack → `[ True False]`). Reduce with
  `.all()` for one verdict; the safe-mode preconditions do exactly this at the call site, so they require
  **all** stack elements to pass (see `docs/numerical_contracts.md`). **One asymmetry worth knowing:**
  the *structural* `has_minimal_ranks` is a **scalar in the ragged layer** (one core shape ⇒ ranks are
  shared across the stack, so there is nothing per-element to report) but **per-element in the uniform
  layer** (ranks vary per stack element — the determinantal variety, `docs/uniform_ranks_and_varieties.md`);
  the *numerical* checkers are per-element in both layers.
- **`to_dense` and the doubled-rank `to_t3` are not symmetric about broadcasting** — `to_dense`
  broadcasts lazily (bare array out), `to_t3` materializes (validated class out). See §5.
- **`apply`/`entries` are `W+C`** (vec/index stack OUTER, T3 stack INNER), like everything else.
- **Canonical core-tuple orderings (frontend takes precedence):** `TuckerTensorTrain.data =
  (tucker_cores, tt_cores)`; `T3Frame.data = (up, down, left, right)`; `T3Variations.data =
  (tucker_variations, tt_variations)`; `(frame, variations)` pairs are frame-first. Pass `.data`
  straight through; do not reorder.

---

## 10. Where to look (file/function map)

| Concern | Look at |
|---|---|
| The `'...'`-einsum ops (broadcast a frame over `K`/`W`) | `backend/t3_conversions.py` (`t3_to_dense`) + `backend/t3_operations.py` (`t3_broadcast_to_common_stack`), `backend/tv_operations.py` (`tv_to_dense/_t3`, gauge, `project_*`) |
| The grouped-einsum interpreter (`W`/`K`/`C` group blocks) | `backend/contractions.py` (`contract`) |
| Probing (2-block `W`,`C`; 3-block `W`,`K`,`C` for a tangent-stacked tangent — forward + transpose) | `backend/probing.py`, `manifold.py` (`T3Tangent.probe`/`probe_transpose`) |
| Tree ↔ stacked-object (meaning 2) | `backend/stacking.py` |
| Two-axis stack/unstack | `manifold.py` (`unstack_tangents`/`_frame`, `stack_*`) + `tv_operations.py` (`*_stack` backend fns) |
| The `K`/`C` split + bv-pair check | `frame_variations_format.py` (`check_fv_pair`), `manifold.py` (`frame_stack_shape`/`tangent_stack_shape`/`stack_shape`) |
| jax pytree registration + `vmap`/`jit` | bottom of `frame_variations_format.py` & `manifold.py`; `tests/test_dispatch.py` |
| The validate/uniform-stack requirement | `tucker_tensor_train.py` (`validate`), `frame_variations_format.py` (`T3Frame.validate`/`T3Variations.validate`) |

---

## 11. Transposes, adjoints, and `sum_over_probes`

Every transpose in the library — `probe_transpose`, and the `apply`/`entries` adjoints on both
`T3Tangent` and `TuckerTensorTrain` — takes a `sum_over_probes` flag. This is the one
place where the probe stack `W` does something subtle, so here is the whole story.

### The mental model

The **atomic** operation is single-probe: one set of vectors `ww`, one residual. Everything batched is
that atom **lifted over the stacks** `W`/`K`/`C` (§2). The transpose lifts the same way:

- **`sum_over_probes=False` (default) is the primary transpose.** It lifts the atomic adjoint over `W`,
  so `W` stays a passthrough **stacking** axis — *one tangent (or tensor) per probe*. This is the plain
  Jacobian-transpose; it is well-defined on its own, with no reference to summing. Here `W` behaves
  exactly like `C` and `K`: unstack → transpose each → restack.
- **`sum_over_probes=True` is a derived convenience** that additionally **contracts** `W`:
  ```
  sum_over_probes=True   ==   (sum over the W axes of   sum_over_probes=False)
  ```
  It exists because, in optimization, the `W` probes are the *outputs of one operator on one shared
  input* — the forward broadcasts that single input across all `W` probes, and **the transpose of a
  broadcast is a sum**. Summing the per-probe contributions gives the one gradient / back-projection
  `Jᵀr`.

The trap (which has caught the authors): do **not** read `True` as "the real transpose" and `False` as
a special case. It is the reverse — `False` is the fundamental object, `True` is `Σ_W` applied on top.

### Which mode do I want?

- **Using `J` and `Jᵀ` as standalone linear operators**, or you want one output per probe → **`False`**
  (default).
- **Assembling a gradient `g = Jᵀr`, or a Gauss-Newton Hessian apply `JᵀJ v`** in an optimizer (one
  residual *vector* over the `W` data points → one gradient) → **`True`**.

Either way the two agree by the invariant above, so when in doubt use `False` and sum the `W` axes
yourself once you actually need the gradient.

### Shape contract

The residual lives in the **forward's output space**; the transpose maps it back. `False` keeps `W` as
an output stack; `True` sums `W` away; `K` and `C` always pass through.

| transpose | residual in | `sum=False` out | `sum=True` out |
|---|---|---|---|
| `T3Tangent.probe_transpose` | `ztildes[i]`: `W + K + C + (Nᵢ,)` | tangent stack `W + K`, frame `C` | tangent stack `K`, frame `C` |
| `T3Tangent.apply_transpose` / `entries_transpose` | `c`: `W + C` | tangent stack `W`, frame `C` | tangent stack `()`, frame `C` |
| `TuckerTensorTrain.apply_ambient_transpose` / `entries_ambient_transpose` | `c`: `W + C` | CP factors, stack `W + C` (CP rank 1) | CP factors, stack `C` (`W` becomes the CP rank) |
| `TuckerTensorTrain.apply_corewise_transpose` / `entries_corewise_transpose` | `c`: `W + C` | core grads, stack `W + C` | core grads, stack `C` |

The `apply`/`entries` adjoints accept a `K` block in the residual too — it is optional, so a residual
of shape `W + C` and one of shape `W + K + C` are both valid. Their forward outputs
are `W + K + C` (tangent) and `W + C` (plain), so the residual-in column is just "the forward output,
adjoint-mapped back."
