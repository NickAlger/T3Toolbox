# Batching and stacking in T3Toolbox — the complete reference

> **If you (human or AI) are about to touch anything with batch/stack axes — `stack_shape`,
> `contractions.py`, probing, V-stacked tangents, `to_dense`, `vmap`/`jit` over these objects — read
> this first.** It is the single most error-prone part of the library, and the subtleties below were
> learned the hard way. This file is the source of truth for *why* the conventions are what they are;
> `CLAUDE.md` has the terse version and points here.

---

## 0. TL;DR

- The word **"stack"** means **three different things** in this codebase. Keep them apart (§1).
- There are **three distinct batch *blocks*** — `G` (base/core stack), `F` (probe stack), `V` (tangent
  stack). They batch *different things* and live on *different subsets* of the operands (§2).
- Library-wide convention is **base-inner**: the core stack `G` is **innermost** (adjacent to the
  tensor indices); the extra stacks `F`, `V` are **outermost**. Orders: `F+G`, `V+G`, `F+V+G` (§3).
- There are **two machineries** for batch axes: a leading `...` in einsum (for *one* broadcastable
  prefix) and the **named grouped-block contractions** in `contractions.py` (for *two* independent
  blocks on different operand subsets) (§4).
- A T3 may have **heterogeneous-but-broadcastable** core stacks (e.g. base cores `G`, one variation
  core `V+G`). This is *first-class* in the backend; `broadcast_t3_to_common_stack` materializes it
  when a uniform-stack object is required (§5).
- Backend functions accept raw tuples and tolerate heterogeneous stacks; **frontend dataclasses
  (`validate`) require a uniform stack**. That asymmetry explains where broadcasting is lazy vs
  materialized (§5).
- The `V`/`G` split is **not stored** — it is recovered from the `(basis, variations)` pairing (§6).
- `jax` pytree registration makes `T3Tangent` `vmap`/`jit`-able; **the basis is `aux_data`**, so
  `vmap` over a tangent maps over the `V` stack with the base fixed (§7).

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

## 2. The three batch *blocks*: `G`, `F`, `V`

Within meaning (1), there are **three semantically distinct things one might batch**. They are not
interchangeable, and a single object can carry more than one at once.

| Block | Name | Batches… | Lives on… | Appears in |
|------|------|----------|-----------|------------|
| **`G`** | base / core stack | a batch of **T3 objects / base points** | **the cores** (every core) | everywhere; `= stack_shape` of a `TuckerTensorTrain`/`T3Basis` |
| **`F`** | probe stack | a batch of **probe vectors** | **the probe vectors `ww` only** (not the cores) | probing (`probe_*`), `apply`/`entries` (there `F` = the vec/index batch) |
| **`V`** | tangent stack | a batch of **tangent vectors sharing one base** | **the variations only** (not the basis cores) | `T3Tangent` (`tangent_stack_shape`) |

Key facts:

- **`G` is shared across `F` and `V`.** A probe stack `F` of probe vectors is applied to a `G`-batch
  of T3s → output carries both. A `V`-batch of tangents shares one `G`-batch of base points.
- **`F` and `V` live on *different operands* than `G`.** `F` is only on `ww`; `V` is only on the
  variation cores; `G` is only on the cores. This is exactly why a single `'...'` is not enough (§4):
  `'...'` broadcasts a *shared* prefix, but `F`/`V` are present on a *subset* of operands.
- In the **transpose** of probing, the probe stack and the tangent stack coincide: `V == F` (each
  probe residual becomes one tangent).

**Concrete shapes (base-inner, see §3):**
- a T3 core: `G + (rL, n, rR)` (tt core) or `G + (n, N)` (tucker core)
- a probe vector `ww[i]`: `F + (N_i,)`
- a `T3Variations` core: `V + G + (rL, n, rR)` or `V + G + (n, N)`
- a forward probe output `zz[i]`: `F + G + (N_i,)`  (= `probe_t3` output `F + G`)
- a transpose-probe non-summed output: a `T3Tangent` whose `V == F`, i.e. variations `V + G`

---

## 3. The base-inner convention (and *why*)

**Rule (library-wide): order batch axes by how core-bound they are. The base/core stack `G` is
INNERMOST (adjacent to the indices). Extra stacks — probe `F`, tangent `V` — are OUTERMOST.**

So the canonical orderings are:
- variation cores: `V + G + core`
- forward probe / `apply` / `entries`: `F + G`
- full probe of a `V`-stacked tangent: `F + V + G`
- transpose non-summed: `V + G` (with `V == F`)

### Why this exact order — the broadcasting mechanism

This is **the** reason for the convention, and the thing most worth internalizing.

The bulk of the tangent/T3 operations combine a **base** (carrying only `G`) with a **variation**
(carrying `V + G`, or `F + G`) using plain `'...'` einsums (`to_dense`, the gauge projections,
`project_t3_onto_tangent_space`, corewise linalg). numpy/jax `'...'` broadcasting is **right-aligned**.

- With **base-inner** (`G` innermost), `G` is the **trailing suffix** of `V + G`. So a `G`-stacked base
  core broadcasts cleanly under a `V + G`-stacked variation core: the trailing `G` axes align, and the
  leading `V` axes are replicated **for free** — exactly the semantics "the one base point is shared
  by all `V` tangent vectors at it." Same for `F + G`.
- With base-*outer* (`G + V`), the trailing axes would be `V`, `G` would be the prefix, and the free
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
negative axes, and `G`/`V`/`F` ride along.

### (b) Two independent blocks on DIFFERENT operand subsets → `contractions.py`

When **two** independent batch blocks live on **different subsets** of the operands, a single `'...'`
**cannot** express it: right-aligned broadcasting would force the two blocks to align, but they are on
different operands and must stay independent. **The canonical case is probing:** the core/base stack
`G` (on the cores) and the probe stack `F` (on the probe vectors only).

So probing is built on the **named grouped-block contraction toolkit** in `backend/contractions.py`:

- Each function is named `inputs_to_output` with **one capital letter per grouped block** and
  lowercase letters for single axes — e.g. `FGa_Gaib_FGi_to_FGb`, `FGo_FGa_to_Gao`.
- Each grouped block (a capital) is reshaped to **one flat axis** of size `math.prod(shape)` (which is
  **`1` when the block is empty**, so the *same code* handles no-stack / one-stack / both-stack — the
  empty case collapses to a length-1 axis).
- The flattened operands are `einsum`'d with the capitals as ordinary indices, then the result is
  reshaped back to the original block shapes.
- **Output order is base-inner too:** `F` (and `V`) outer, `G` inner. E.g. `FGa_..._to_FGb` returns
  `F_shape + G_shape + (b,)`.

When you need a *third* private block (e.g. forward-probing a `V`-stacked tangent — `F` probes, `V`
tangents, `G` base, all independent), you need a **3-block** contraction (`F`, `V`, `G`). The library
currently has 1- and 2-block contractions; 3-block is **deferred** in favour of map-over-`V` (§7).

**Decision rule:** if your two batches are on the *same* operands → `'...'`. If they are on *different*
operand subsets and must remain independent → a grouped-block contraction.

---

## 5. Heterogeneous-but-broadcastable tuples, and the backend/frontend split

A subtle but pervasive pattern: a single T3's cores may have **different (but broadcastable) stacks**.
The archetype is a **tangent term**: the base cores carry `G`, but one variation core carries `V + G`.
Per §3 this is a *deliberate, valid* layout — base-inner makes `G` the suffix of `V + G`, so the
operation broadcasts the base over `V`.

- **The `'...'`-einsum ops already handle this for free** (gauge, `project_*`). They never materialize
  the broadcast — they keep the base at `G` and let `'...'` replicate it. This is the *efficient*
  pattern (no `|V|` copies of the base).
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
  - `tangent_to_t3` → must broadcast base→`V+G` before concatenating the doubled-rank cores (its output
    *is* a validated `TuckerTensorTrain`; `[U_i ; V_i]` is one array).
  - frontend `bvf.bv_to_t3` → must `broadcast_t3_to_common_stack` the mixed-stack term before wrapping
    in `TuckerTensorTrain`.
  - whereas `to_dense` produces a *bare array*, so it can broadcast lazily inside the contraction.

> Rule of thumb: **bare-array output → broadcast lazily; validated-class-instance output → materialize
> the broadcast to a uniform stack.**

---

## 6. The `V`/`G` split is recovered, not stored

`T3Tangent` bundles a `T3Basis` (stack `G`) with a `T3Variations` (stack `V + G`). The split is
**derived**, never stored as a field (minimal dataclasses):

- `base_stack_shape` (`G`) `= basis.stack_shape`
- `stack_shape` (`V + G`) `= variations.stack_shape`
- `tangent_stack_shape` (`V`) `=` the part of `variations.stack_shape` that *exceeds* `basis.stack_shape`
  — i.e. `G` is the **trailing suffix** of `V + G`, and `V` is the leading remainder.

`check_bv_pair` enforces exactly this: *`base.stack_shape` must be the trailing (inner) suffix of
`variations.stack_shape`.* (It does **not** require equality — that was the pre-`V`-stack invariant.)

Because the split is not recoverable from a *bare tree of objects*, the two-axis stack/unstack (§ below)
must name which stack it peels.

### Two-axis stack/unstack on `T3Tangent`

A single monolithic `stack`/`unstack` cannot faithfully invert two stacks (the `V`/`G` split isn't in
a bare tree). So `T3Tangent` has **two explicit pairs**, each peeling **one named stack**:

- `unstack_tangents` / `stack_tangents` — peel the **tangent stack `V`**. Yields a `V`-shaped tree of
  tangents that **share the base** (same `T3Basis` *object* → same tangent space → mutually
  linalg-compatible). "For each vector within the basis." `stack_tangents` **guards** that all leaves
  share the same basis object (structural identity, as in `inner`/`+`).
- `unstack_basis` / `stack_basis` — peel the **base stack `G`**. Yields a `G`-shaped tree of
  single-base-point tangents at **distinct** bases (distinct tangent spaces). "For each basis."
  `stack_basis` places `G` **innermost** (variation stack → `V + G`), which needs *interior-axis*
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
  - **`vmap` over a `T3Tangent` maps over the variations = the `V` stack, basis fixed** — i.e.
    `vmap`-over-`V` *is* the batch-of-tangents-sharing-one-base picture. This is the clean jax route for
    5c (forward-probe a `V`-stacked tangent): `vmap` the 2-block probe over `V` (basis fixed as aux),
    numpy path loops. Defer true 3-block (`F`,`V`,`G`) contractions until there's a perf need.
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

- **Body locals** suffix the axis layout: `G_aib`, `mu_FGa`, `B0_b_j_c`, `xi_IXp`. Capitals = grouped
  index blocks (`G`/`F`/`V`/`X`/`I`...), lowercase = single axes, a leading `d` = a stacked/derivative
  axis. In `apply`/`entries` the body uses `X` for the core stack `G` and `V`/`I` for the vec/index
  stack `F` (so `mu_VXa` = `F + G + a`).
- **Contraction functions** are `inputs_to_output`, e.g. `FGa_Gaib_FGi_to_FGb`: read each token as a
  grouped-block einsum signature; output is base-inner (`F` outer, `G` inner).
- **Paper ↔ code** (Appendix A of the T4S paper): `U`=up_tucker, `P`=left_tt, `Q`=right_tt,
  `O`=down_tt (called `down_tt_cores`, not "outer"); `δU`=tucker_variations (`V`), `δG`=tt_variations
  (`H`). Note `V` is overloaded: the paper's variation symbol vs this doc's **tangent stack `V`** —
  context disambiguates.

---

## 9. Gotchas (the ones that have actually bitten)

- **`corewise_dot` / `corewise_norm` collapse EVERY axis** (stacks included) to a scalar. To keep the
  stack (vectorized linalg), use `corewise.corewise_stack_dot(X, Y, n_stack)`.
- **`to_dense` and the doubled-rank `to_t3` are not symmetric about broadcasting** — `to_dense`
  broadcasts lazily (bare array out), `to_t3` materializes (validated class out). See §5.
- **`apply`/`entries` are now `F+G`** (vec/index stack OUTER, T3 stack INNER) as of slice 5b — older
  code/notes may say `G+F`.
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
| The `'...'`-einsum ops (broadcast a base over `V`/`F`) | `backend/t3_operations.py` (`to_dense`, `broadcast_t3_to_common_stack`), `backend/tangent_operations.py` (`tangent_to_dense/_t3`, gauge, `project_*`) |
| The grouped-block contraction toolkit (`F`/`G`/`V` blocks) | `backend/contractions.py` |
| Probing (the canonical 2-block `F`,`G` case) | `backend/probing.py`, `manifold.py` (`T3Tangent.probe`/`probe_transpose`) |
| Tree ↔ stacked-object (meaning 2) | `backend/stacking.py` |
| Two-axis stack/unstack | `manifold.py` (`unstack_tangents`/`_basis`, `stack_*`) + `tangent_operations.py` (`*_stack` backend fns) |
| The `V`/`G` split + bv-pair check | `basis_variations_format.py` (`check_bv_pair`), `manifold.py` (`base_stack_shape`/`tangent_stack_shape`/`stack_shape`) |
| jax pytree registration + `vmap`/`jit` | bottom of `basis_variations_format.py` & `manifold.py`; `tests/test_dispatch.py` |
| The validate/uniform-stack requirement | `tucker_tensor_train.py` (`validate`), `basis_variations_format.py` (`T3Basis.validate`/`T3Variations.validate`) |

---

*Maintenance note: when you change a stacking convention, this file and `CLAUDE.md` are part of the
blast radius — update both. The conventions here are deliberate; if you find yourself wanting to break
base-inner, re-read §3 first.*
