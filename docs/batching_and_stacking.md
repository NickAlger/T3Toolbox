# Batching and stacking in T3Toolbox — the complete reference

> **If you (human or AI) are about to touch anything with batch/stack axes — `stack_shape`,
> `contractions.py`, probing, `K`-stacked (tangent-stacked) tangents, `to_dense`, `vmap`/`jit` over
> these objects — read this first.** It is the single most error-prone part of the library, and the
> subtleties below were learned the hard way. This file is the source of truth for *why* the
> conventions are what they are; `CLAUDE.md` has the terse version and points here.

---

## Start here — the mental model in one screen

*New to stacking here? Read this section; it is enough to **use** the library. The numbered sections
below are the full reference for people **editing** it.*

**Stacking = leading batch axes on the cores, so one object holds many.** Every `TuckerTensorTrain` /
`T3Basis` / `T3Variations` stores its cores as `core.shape == stack_shape + (tensor/rank axes)`. With
`stack_shape = (2, 3)` you carry a `2×3` grid of T3s/bases/variations in one object, and operations
vectorize over them.

**There are three kinds of batch — three "blocks" — that batch different things, on different arrays.**
Keeping them straight is the whole game; conflating them is the #1 source of confusion:

- **`C` — the base/core stack.** A batch of whole T3s / base points on the manifold. Lives on **every
  core** (it *is* `stack_shape`). Mnemonic: on every **C**ore.
- **`W` — the probe stack.** A batch of probe-vector sets. Lives on the **probe vectors `ww` only**, not
  the cores. Mnemonic: the **w** vectors.
- **`K` — the tangent stack.** A batch of tangent vectors attached at the *same* base point. Lives on
  the **variation cores only**. Mnemonic: the **k** tangent vectors at each base.

One object can carry more than one block at once. Order is always **base-inner**: `W + K + C + (axes)`
(`C` innermost, adjacent to the indices; `W`/`K` outermost). Why that order? See §3 — it is exactly
what lets a shared base broadcast over its batch for free.

### One concrete example (the picture that makes it click)

Take `d = 3` modes, a batch of **2 base points** (`C = (2,)`), with **3 tangent vectors at each**
(`K = (3,)`), probed by **4 probe-sets** (`W = (4,)`). Then every array's shape is:

| Array | Blocks | Shape |
|---|---|---|
| base Tucker core `Uᵢ` (`T3Basis`) | `C` | `(2,) + (nᵢ, Nᵢ)` |
| base TT core `Pᵢ` (`T3Basis`) | `C` | `(2,) + (rᵢ, nᵢ, rᵢ₊₁)` |
| variation core `δUᵢ` (`T3Variations`) | `K + C` | `(3, 2) + (nᵢ, Nᵢ)` |
| probe vector `wᵢ` (`ww`) | `W` | `(4,) + (Nᵢ,)` |
| forward probe `zᵢ` (`tangent.probe(ww)`) | `W + K + C` | `(4, 3, 2) + (Nᵢ,)` |

Read it as: *2 base points, each with its own frame; 3 tangent vectors attached at each; probe every
one against 4 probe-sets → `4 × 3 × 2` probe results per mode.* The base frame (`C`) is **shared**
across the `K` tangents at it — never copied; base-inner broadcasting (§3) handles that for free.

> ⚠️ The word **"stack" itself means three different things** in this codebase (`stack_shape`; the
> tree↔object machinery in `stacking.py`; the deferred uniform supercore). When confused, first decide
> *which*. See §1.

### Shape-notation legend (how to read the shapes in the code)

The codebase annotates shapes in trailing comments and encodes them in names. The whole scheme:

- A core's shape is written **`stack_shape + (tensor/rank axes)`**, e.g. `Bi.shape = stack_shape + (nᵢ, Nᵢ)`.
- **Capital letters are grouped batch blocks**, each standing for **zero or more** axes: **`C`** base/core
  stack, **`W`** probe stack, **`K`** tangent stack. An empty block contributes no axes (so the same code
  handles no-stack and stacked).
- **lowercase letters are single axes** — tensor modes / ranks (`n`, `N`, `r`) or contraction legs
  (`a`, `i`, `b`, `o`, `j`).
- **Order is base-inner: `W + K + C + (axes)`** (probe outer, tangent middle, base inner).
- **Body locals suffix their layout:** `mu_WCa` is an array with axes `W + C + (a,)`.
- **Contraction functions are `inputs_to_output`:** `WCa_Caib_WCi_to_WCb` reads as a per-operand
  block+leg signature (operand 1 = `W+C+(a,)`; operand 2 = a `C`-only core with legs `a,i,b`; operand
  3 = `W+C+(i,)`; output base-inner `W+C+(b,)`).
- A leading **`d`** on a contraction name (`dWCa_…`) is the deferred uniform layer's supercore/derivative
  axis (ignore unless repairing the uniform layer).

### Glossary

| Term | Meaning |
|---|---|
| **`stack_shape`** | the leading batch axes shared by all of an object's cores; *is* the base/core stack `C`. |
| **base / core stack `C`** | a batch of T3s / base points, on every core (`= stack_shape`). (Was `G` before the W/K/C rename.) |
| **probe stack `W`** | a batch of probe-vector sets, on `ww` only. (Was `F`.) |
| **tangent stack `K`** | a batch of tangent vectors at one base, on the variation cores only; `T3Tangent.tangent_stack_shape`. (Was `V`.) |
| **base-inner** | the ordering rule: `C` innermost, `W`/`K` outermost (`W + K + C`). |
| **`base_stack_shape` / `tangent_stack_shape`** | a `T3Tangent`'s `C` and `K` parts, *derived* from the (basis, variations) pairing (§6), not stored. |
| **heterogeneous stack** | one T3 whose cores have different-but-broadcastable stacks (base `C`, variation `K+C`). First-class in the backend (§5). |
| **"the split is recovered"** | `C`/`K`/`W` lengths are read off operand shapes, never threaded as parameters (§4, §6). |
| **`sum_over_probes`** | transpose flag (§11): `False` (default, **primary**) keeps the probe stack `W` as an output stack — one tangent/tensor per probe; `True` sums `W` (`= Σ_W` of `False`) for the optimization `Jᵀr`. |
| **ragged / uniform / weighted** | the three representations. Only **ragged** (tuples of arrays) is fully working; uniform (supercore) and weighted are deferred. |

> **Why these letters?** `W`/`K`/`C` are deliberately disjoint from the core/variation symbols
> (`U`,`P`,`Q`,`O`,`G`,`H`,`B`) and from `Jᵀ`/tensor/`T3`, so a shape comment is never ambiguous.
> (Before this rename they were `F`/`V`/`G`, which clashed with the TT-core `G` and the
> Tucker-variation `V`; `apply`/`entries`/`probe_dense` additionally drifted to `X`/`V`/`I`/`K`/`Z`.)

---

## 0. TL;DR

- The word **"stack"** means **three different things** in this codebase. Keep them apart (§1).
- There are **three distinct batch *blocks*** — `C` (base/core stack), `W` (probe stack), `K` (tangent
  stack). They batch *different things* and live on *different subsets* of the operands (§2).
- Library-wide convention is **base-inner**: the core stack `C` is **innermost** (adjacent to the
  tensor indices); the extra stacks `W`, `K` are **outermost**. Orders: `W+C`, `K+C`, `W+K+C` (§3).
- There are **two machineries** for batch axes: a leading `...` in einsum (for *one* broadcastable
  prefix) and the **named grouped-block contractions** in `contractions.py` (for *two* independent
  blocks on different operand subsets) (§4).
- A T3 may have **heterogeneous-but-broadcastable** core stacks (e.g. base cores `C`, one variation
  core `K+C`). This is *first-class* in the backend; `broadcast_t3_to_common_stack` materializes it
  when a uniform-stack object is required (§5).
- Backend functions accept raw tuples and tolerate heterogeneous stacks; **frontend dataclasses
  (`validate`) require a uniform stack**. That asymmetry explains where broadcasting is lazy vs
  materialized (§5).
- The `K`/`C` split is **not stored** — it is recovered from the `(basis, variations)` pairing (§6).
- `jax` pytree registration makes `T3Tangent` `vmap`/`jit`-able; **the basis is `aux_data`**, so
  `vmap` over a tangent maps over the `K` stack with the base fixed (§7).

---

## 1. "Stacking" means three different things

1. **`stack_shape` — leading batch axes on ONE object's cores.** Every core of a `TuckerTensorTrain`
   / `T3Basis` / `T3Variations` is `core.shape == stack_shape + (tensor/rank axes)`. A single leading
   `'...'` in an einsum rides these axes along for free. **This is the common case** and the meaning
   of "stacked" 95% of the time. Caveat: a plain `'...'` carries exactly **one** shared/broadcast
   prefix.

2. **`backend/stacking.py` — converting a Python *tree of separate objects* ↔ *one stacked object*.**
   `stack(tree, axes)` / `unstack(S, axes)` / `tree_zip` / `apply_func_to_leaf_subtrees`. This is its
   own tree machinery — **not** jax pytrees, and not the same as meaning (1). It is what
   `T3Tangent.unstack_*` / `stack_*`, `T3Basis.stack`, etc. are built on.

3. **The uniform supercore (`ut3_*`, `ubv_*`, `uniform_*`) — a separate representation, currently
   deferred/broken.** One stacked supercore array + masks for `jax.lax.scan`. Ignore unless you are
   specifically repairing the uniform layer.

When something is confusing, first ask **which of these three** you are dealing with.

---

## 2. The three batch *blocks*: `C`, `W`, `K`

Within meaning (1), there are **three semantically distinct things one might batch**. They are not
interchangeable, and a single object can carry more than one at once.

| Block | Name | Batches… | Lives on… | Appears in |
|------|------|----------|-----------|------------|
| **`C`** | base / core stack | a batch of **T3 objects / base points** | **the cores** (every core) | everywhere; `= stack_shape` of a `TuckerTensorTrain`/`T3Basis` |
| **`W`** | probe stack | a batch of **probe vectors** | **the probe vectors `ww` only** (not the cores) | probing (`probe_*`), `apply`/`entries` (there `W` = the vec/index batch) |
| **`K`** | tangent stack | a batch of **tangent vectors sharing one base** | **the variations only** (not the basis cores) | `T3Tangent` (`tangent_stack_shape`) |

Key facts:

- **`C` is shared across `W` and `K`.** A probe stack `W` of probe vectors is applied to a `C`-batch
  of T3s → output carries both. A `K`-batch of tangents shares one `C`-batch of base points.
- **`W` and `K` live on *different operands* than `C`.** `W` is only on `ww`; `K` is only on the
  variation cores; `C` is only on the cores. This is exactly why a single `'...'` is not enough (§4):
  `'...'` broadcasts a *shared* prefix, but `W`/`K` are present on a *subset* of operands.
- In the **transpose** of probing, the probe stack and the tangent stack coincide: `K == W` (each
  probe residual becomes one tangent) — when `sum_over_probes=False`; setting it `True` sums `W` away.
  See §11 for the full story on transposes and `sum_over_probes`.

**Concrete shapes (base-inner, see §3):**
- a T3 core: `C + (rL, n, rR)` (tt core) or `C + (n, N)` (tucker core)
- a probe vector `ww[i]`: `W + (N_i,)`
- a `T3Variations` core: `K + C + (rL, n, rR)` or `K + C + (n, N)`
- a forward probe output `zz[i]`: `W + C + (N_i,)`  (= `probe_t3` output `W + C`)
- a transpose-probe non-summed output: a `T3Tangent` whose `K == W`, i.e. variations `K + C`

---

## 3. The base-inner convention (and *why*)

**Rule (library-wide): order batch axes by how core-bound they are. The base/core stack `C` is
INNERMOST (adjacent to the indices). Extra stacks — probe `W`, tangent `K` — are OUTERMOST.**

So the canonical orderings are:
- variation cores: `K + C + core`
- forward probe / `apply` / `entries`: `W + C`
- full probe of a `K`-stacked tangent: `W + K + C`
- transpose non-summed: `K + C` (with `K == W`)

### Why this exact order — the broadcasting mechanism

This is **the** reason for the convention, and the thing most worth internalizing.

The bulk of the tangent/T3 operations combine a **base** (carrying only `C`) with a **variation**
(carrying `K + C`, or `W + C`) using plain `'...'` einsums (`to_dense`, the gauge projections,
`project_t3_onto_tangent_space`, corewise linalg). numpy/jax `'...'` broadcasting is **right-aligned**.

- With **base-inner** (`C` innermost), `C` is the **trailing suffix** of `K + C`. So a `C`-stacked base
  core broadcasts cleanly under a `K + C`-stacked variation core: the trailing `C` axes align, and the
  leading `K` axes are replicated **for free** — exactly the semantics "the one base point is shared
  by all `K` tangent vectors at it." Same for `W + C`.
- With base-*outer* (`C + K`), the trailing axes would be `K`, `C` would be the prefix, and the free
  right-aligned broadcast would mismatch. You would have to insert explicit transposes/reshapes at
  every boundary.

The **custom grouped-block contractions** (§4) are *flops-neutral* to block order (they reshape each
block to one flat axis regardless), so they are made to follow the same base-inner order purely for
**one consistent layout with no boundary-transpose copies**. That is the whole justification —
broadcasting determines the order; the contractions just conform.

> Mnemonic: **"a base broadcasts over its batch only when the base axes are on the inside."**

---

## 4. The two machineries for batch axes

### (a) One broadcastable prefix → a leading `'...'` in einsum

If all operands share **one** batch prefix — or one operand's prefix is a base that *broadcasts*
under another's via base-inner (§3) — a leading `'...'` handles it for free. This covers `to_dense`,
the gauge projections, `project_*`, `corewise_*`, `tangent_to_dense/_t3`, orthogonalization, etc.

This is why most of the library "just works" with stacking: you write the einsum with `'...'` and
negative axes, and `C`/`K`/`W` ride along.

### (b) Two independent blocks on DIFFERENT operand subsets → `contractions.py`

When **two** independent batch blocks live on **different subsets** of the operands, a single `'...'`
**cannot** express it: right-aligned broadcasting would force the two blocks to align, but they are on
different operands and must stay independent. **The canonical case is probing:** the core/base stack
`C` (on the cores) and the probe stack `W` (on the probe vectors only).

So probing is built on the **named grouped-block contraction toolkit** in `backend/contractions.py`:

- Each function is named `inputs_to_output` with **one capital letter per grouped block** and
  lowercase letters for single axes — e.g. `WCa_Caib_WCi_to_WCb`, `WCo_WCa_to_Cao`.
- Each grouped block (a capital) is reshaped to **one flat axis** of size `math.prod(shape)` (which is
  **`1` when the block is empty**, so the *same code* handles no-stack / one-stack / both-stack — the
  empty case collapses to a length-1 axis).
- The flattened operands are `einsum`'d with the capitals as ordinary indices, then the result is
  reshaped back to the original block shapes.
- **Output order is base-inner too:** `W` (and `K`) outer, `C` inner. E.g. `WCa_..._to_WCb` returns
  `W_shape + C_shape + (b,)`.

When you need a *third* private block (e.g. forward-probing a `K`-stacked tangent — `W` probes, `K`
tangents, `C` base, all independent), you need a **3-block** contraction (`W`, `K`, `C`). These exist
as of slice 5c (the bottom of `contractions.py`, base-inner output `W + K + C`), used by both the
forward tangent probe's perturbation sweep and the transpose (`probe_transpose` accepting `K`-stacked
residuals: the adjoint sweep reuses the forward's contractions, the assembly adds 10 outer-product
builders in keep-`W`/sum-`W` forms). **The split is recovered from operand shapes, never passed in:**
whichever stack has a *pure* operand pins its length — a `C`-only base core pins `len(C)` (forward), a
`W`-only probe vector pins `len(W)` (transpose tucker-assemble) — and the rest self-infer the remainder.
Only a contraction with *no* pure operand for the needed split takes an int count (`{W+C, K+C, W+C}` or
`{W+C, W+K+C}` is underdetermined by axis counts alone): the forward's variation-core-only ones take
`n_base`; the transpose's `tt`-assemble takes `n_probe`. Each is recomputed at the lowest level that
holds a suitable operand (the sweep `_func`, or `probe_tangent_transpose` for `tt`-assemble, which has
no pure operand of its own), the same precedent as the original `n_probe`. Each reduces to the
corresponding 2-block contraction when `K` is empty. (The earlier plan to defer this in favour of
map-over-`K` was reversed — see §7.)

**Decision rule:** if your two batches are on the *same* operands → `'...'`. If they are on *different*
operand subsets and must remain independent → a grouped-block contraction.

---

## 5. Heterogeneous-but-broadcastable tuples, and the backend/frontend split

A subtle but pervasive pattern: a single T3's cores may have **different (but broadcastable) stacks**.
The archetype is a **tangent term**: the base cores carry `C`, but one variation core carries `K + C`.
Per §3 this is a *deliberate, valid* layout — base-inner makes `C` the suffix of `K + C`, so the
operation broadcasts the base over `K`.

- **The `'...'`-einsum ops already handle this for free** (gauge, `project_*`). They never materialize
  the broadcast — they keep the base at `C` and let `'...'` replicate it. This is the *efficient*
  pattern (no `|K|` copies of the base).
- **The one reshape-based primitive, `to_dense`, did NOT** (it read a single `vs` from the first core
  and hard-reshaped every core to it). It is now made broadcast-aware via
  **`broadcast_t3_to_common_stack`** (`backend/t3_operations.py`): compute `np.broadcast_shapes` of all
  core stacks, `broadcast_to` each core up, then contract. No-op for a uniform-stack T3.

### Backend tuples vs frontend dataclasses — the validate asymmetry

This is *the* thing that decides where broadcasting is lazy vs materialized:

- **Backend functions take raw `(tucker_cores, tt_cores)` tuples and tolerate heterogeneous stacks.**
  Heterogeneous-but-broadcastable tuples are first-class there.
- **Frontend dataclasses validate a UNIFORM stack.** `TuckerTensorTrain.validate()` (and
  `T3Basis`/`T3Variations`) *hard-require* every core to share one `stack_shape`. So **any builder that
  produces a class instance must materialize the broadcast to uniform**:
  - `tangent_to_t3` → must broadcast base→`K+C` before concatenating the doubled-rank cores (its output
    *is* a validated `TuckerTensorTrain`; `[U_i ; V_i]` is one array).
  - frontend `bvf.bv_to_t3` → must `broadcast_t3_to_common_stack` the mixed-stack term before wrapping
    in `TuckerTensorTrain`.
  - whereas `to_dense` produces a *bare array*, so it can broadcast lazily inside the contraction.

> Rule of thumb: **bare-array output → broadcast lazily; validated-class-instance output → materialize
> the broadcast to a uniform stack.**

---

## 6. The `K`/`C` split is recovered, not stored

`T3Tangent` bundles a `T3Basis` (stack `C`) with a `T3Variations` (stack `K + C`). The split is
**derived**, never stored as a field (minimal dataclasses):

- `base_stack_shape` (`C`) `= basis.stack_shape`
- `stack_shape` (`K + C`) `= variations.stack_shape`
- `tangent_stack_shape` (`K`) `=` the part of `variations.stack_shape` that *exceeds* `basis.stack_shape`
  — i.e. `C` is the **trailing suffix** of `K + C`, and `K` is the leading remainder.

`check_bv_pair` enforces exactly this: *`base.stack_shape` must be the trailing (inner) suffix of
`variations.stack_shape`.* (It does **not** require equality — that was the pre-`K`-stack invariant.)

Because the split is not recoverable from a *bare tree of objects*, the two-axis stack/unstack (§ below)
must name which stack it peels.

### Two-axis stack/unstack on `T3Tangent`

A single monolithic `stack`/`unstack` cannot faithfully invert two stacks (the `K`/`C` split isn't in
a bare tree). So `T3Tangent` has **two explicit pairs**, each peeling **one named stack**:

- `unstack_tangents` / `stack_tangents` — peel the **tangent stack `K`**. Yields a `K`-shaped tree of
  tangents that **share the base** (same `T3Basis` *object* → same tangent space → mutually
  linalg-compatible). "For each vector within the basis." `stack_tangents` **guards** that all leaves
  share the same basis object (structural identity, as in `inner`/`+`).
- `unstack_basis` / `stack_basis` — peel the **base stack `C`**. Yields a `C`-shaped tree of
  single-base-point tangents at **distinct** bases (distinct tangent spaces). "For each basis."
  `stack_basis` places `C` **innermost** (variation stack → `K + C`), which needs *interior-axis*
  stacking that the component `T3Variations.stack` can't do — hence it is a real op, not user-assembled.

`T3Basis`/`T3Variations` keep their single plain `stack`/`unstack` (each is a single-stack object).
Backend functionals (`unstack_tangent_stack`/`stack_tangent_stack`/`unstack_base_stack`/
`stack_base_stack` in `tangent_operations.py`) do the array/tree work; the `T3Tangent` methods are
thin wrappers doing the compatibility checks.

---

## 7. jax pytrees and `vmap` — how batching meets autodiff

The frozen dataclasses are registered jax pytrees (`if has_jax:` at the end of each module), so they
can be `jit`/`vmap`/`grad`-ed:

- `TuckerTensorTrain`, `T3Basis`, `T3Variations`: leaves = the cores (`x.data`), no aux.
- **`T3Tangent`: the BASIS is `aux_data`; only the variations are leaves.** The fixed frame is static;
  the moving tangent vector (the variations) is what you differentiate/optimize/`vmap`. Consequences:
  - **`vmap` over a `T3Tangent` maps over the variations = the `K` stack, basis fixed** — i.e.
    `vmap`-over-`K` *is* the batch-of-tangents-sharing-one-base picture, a clean jax route in general.
    For 5c (forward-probe a `K`-stacked tangent) we deliberately did **not** use it: the probe instead
    grew genuine 3-block (`W`,`K`,`C`) contractions (§4). Rationale — consistency with the
    `contractions.py` toolkit (the blessed mechanism for independent blocks), no Python `K` loop on the
    numpy path, and low-level einsums fold into XLA at least as well as a `vmap` (which can add
    layout/transpose churn over a long function). The `vmap`-over-`K` picture still holds for other
    ops; it just isn't how probing batches.
  - **The same-tangent-space identity guard survives `jit`.** `aux_data` is stored in the treedef and
    reconstructed *by reference*, so two tangents built from the same basis object keep
    `a.basis is b.basis` after flatten/unflatten — `inner`/`+` jit cleanly. It still fires for
    genuinely different bases.
  - **Trade-off:** the basis is a `jit` compile-time constant keyed by object identity. **Hold the
    basis object stable for cache hits; a new base point recompiles.** This fits the usage (heavy
    fixed-base inner solves / batch ops at one base). To differentiate *w.r.t. the basis*, drop to the
    backend functions on the raw cores.
  - `T3Basis` is `@dataclass(frozen=True, eq=False)` so it can serve as `aux_data` (it holds arrays →
    value hash/eq is impossible/ambiguous; identity hash matches the "same object" semantics).

`vmap`/`jit` invocation across the ops is checked cheaply in `tests/test_dispatch.py` (a `jit`-compile
of an op proves no hidden numpy — a stray `np.*` on a tracer raises).

---

## 8. Decoding the names (so you can read the code)

The naming scheme encodes axis layout — once you know it, the einsums read themselves.

- **Body locals** suffix the axis layout: `C_aib`, `mu_WCa`, `B0_b_j_c`, `xi_WCp`. Capitals = grouped
  index blocks (`C`/`W`/`K`), lowercase = single axes, a leading `d` = a stacked/derivative (uniform
  supercore) axis. `apply`/`entries` now use the same `W` (vec/index stack) and `C` (core stack) as
  everywhere else — they previously used a private `X`/`V`/`I` scheme (so `mu_VXa` is now `mu_WCa`).
- **Contraction functions** are `inputs_to_output`, e.g. `WCa_Caib_WCi_to_WCb`: read each token as a
  grouped-block einsum signature; output is base-inner (`W` outer, `C` inner).
- **Paper ↔ code** (Appendix A of the T4S paper): `U`=up_tucker, `P`=left_tt, `Q`=right_tt,
  `O`=down_tt (called `down_tt_cores`, not "outer"); `δU`=tucker_variations (`V`), `δG`=tt_variations
  (`H`). The block letters `W`/`K`/`C` are **deliberately disjoint** from these core/variation symbols,
  so `V` here means only the Tucker variation and `G` only the TT core — **no overload**. (Removing that
  overload — old block letters `F`/`V`/`G` clashing with TT-core `G` and Tucker-variation `V` — was the
  motivation for the rename.)

---

## 9. Gotchas (the ones that have actually bitten)

- **`corewise_dot` / `corewise_norm` collapse EVERY axis** (stacks included) to a scalar. To keep the
  stack (vectorized linalg), use `corewise.corewise_stack_dot(X, Y, n_stack)`.
- **`to_dense` and the doubled-rank `to_t3` are not symmetric about broadcasting** — `to_dense`
  broadcasts lazily (bare array out), `to_t3` materializes (validated class out). See §5.
- **`apply`/`entries` are now `W+C`** (vec/index stack OUTER, T3 stack INNER) as of slice 5b — older
  code/notes may say `G+F` or `F+G`.
- **Canonical core-tuple orderings (frontend takes precedence):** `TuckerTensorTrain.data =
  (tucker_cores, tt_cores)`; `T3Basis.data = (up, down, left, right)`; `T3Variations.data =
  (tucker_variations, tt_variations)`; `(basis, variations)` pairs are basis-first. Pass `.data`
  straight through; do not reorder.
- **Tests are RNG-order sensitive** (one global seed at import) — a bug we hit. New numerical tests are
  numpy-only (jax is covered by `test_dispatch`); see `CLAUDE.md`.
- **Stacked arrays blow up fast.** In tests keep stack dims 1–2 and core dims small.
- The **deferred uniform/weighted layers** still thread `use_jax` (old pattern) and use meaning (3)
  stacking — don't take them as a model.

---

## 10. Where to look (file/function map)

| Concern | Look at |
|---|---|
| The `'...'`-einsum ops (broadcast a base over `K`/`W`) | `backend/t3_operations.py` (`to_dense`, `broadcast_t3_to_common_stack`), `backend/tangent_operations.py` (`tangent_to_dense/_t3`, gauge, `project_*`) |
| The grouped-block contraction toolkit (`W`/`K`/`C` blocks) | `backend/contractions.py` |
| Probing (2-block `W`,`C`; 3-block `W`,`K`,`C` for a tangent-stacked tangent — forward + transpose) | `backend/probing.py`, `manifold.py` (`T3Tangent.probe`/`probe_transpose`) |
| Tree ↔ stacked-object (meaning 2) | `backend/stacking.py` |
| Two-axis stack/unstack | `manifold.py` (`unstack_tangents`/`_basis`, `stack_*`) + `tangent_operations.py` (`*_stack` backend fns) |
| The `K`/`C` split + bv-pair check | `basis_variations_format.py` (`check_bv_pair`), `manifold.py` (`base_stack_shape`/`tangent_stack_shape`/`stack_shape`) |
| jax pytree registration + `vmap`/`jit` | bottom of `basis_variations_format.py` & `manifold.py`; `tests/test_dispatch.py` |
| The validate/uniform-stack requirement | `tucker_tensor_train.py` (`validate`), `basis_variations_format.py` (`T3Basis.validate`/`T3Variations.validate`) |

---

## 11. Transposes, adjoints, and `sum_over_probes`

Every transpose in the library — `probe_transpose`, and the `apply_transpose`/`entries_transpose`
adjoints on both `T3Tangent` and `TuckerTensorTrain` — takes a `sum_over_probes` flag. This is the one
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
| `T3Tangent.probe_transpose` | `ztildes[i]`: `W + K + C + (Nᵢ,)` | tangent stack `W + K`, base `C` | tangent stack `K`, base `C` |
| `T3Tangent.apply_transpose` / `entries_transpose` | `c`: `W + C` | tangent stack `W`, base `C` | tangent stack `()`, base `C` |
| `TuckerTensorTrain.apply_transpose` / `entries_transpose` | `c`: `W + C` | T3 `stack_shape = W + C` | T3 `stack_shape = C` |

The `apply`/`entries` adjoints currently take a residual with no `K` block (`K`-stacked residuals are a
deferred `probe_transpose`-style extension; see `docs/apply_entries_handoff.md`). Their forward outputs
are `W + K + C` (tangent) and `W + C` (plain), so the residual-in column is just "the forward output,
adjoint-mapped back."

---

*Maintenance note: when you change a stacking convention, this file and `CLAUDE.md` are part of the
blast radius — update both. The conventions here are deliberate; if you find yourself wanting to break
base-inner, re-read §3 first.*
