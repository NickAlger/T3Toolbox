# UniformTuckerTensorTrain port plan (living document)

> A running, function-by-function record of how the uniform T3 layer is being rebuilt: for each piece,
> whether we **share** the existing ragged backend (polymorphism "just works"), **rewrite** it
> uniform-specifically (a real structural difference or a vectorization win), or leave it **trivial**
> (inline one-liner), plus the reasoning. This is the implementation checklist; the *why* behind the
> cross-cutting design decisions lives in the four design notes (see below). Updated as we walk through
> the functions; entries marked ✅ are decided, ⬜ are pending.

**Hybrid principle** (per `CLAUDE.md`): share where polymorphism just works; rewrite where there's a
legitimate structural difference, where polymorphism would need unnatural branching, or where splitting
yields a performance gain (typically: ragged *maps over the mode index `d`*, uniform *folds `d` into a
batched array op / einsum*).

**Design notes (the "why"):** [`uniform_ranks_and_varieties.md`](uniform_ranks_and_varieties.md),
[`uniform_supercore_layout.md`](uniform_supercore_layout.md), [`uniform_masks_vs_ranks.md`](uniform_masks_vs_ranks.md),
[`uniform_pytree_composition.md`](uniform_pytree_composition.md).

**Legend:** SHARE = use the polymorphic ragged backend (dispatch on `is_ndarray`); REWRITE = uniform-specific;
TRIVIAL = inline one-liner; SWEEP = sequential over `d` (`xscan`/`lax.scan`); BATCH = vectorized over `d`(+stack).

---

## Cross-cutting repairs (apply to every uniform module as we touch it)

- **Imports:** flatten stale `t3toolbox.backend.{uniform_tucker_tensor_train,tucker_tensor_train}.*`
  paths to `t3toolbox.backend.*` (broken in `ut3_conversions`, `ut3_linalg`, `ut3_svd`, and the frontend).
- **Dispatch:** drop `use_jax` threading; infer from inputs (`tree_contains_jax`/`is_ndarray`). Keep
  `use_jax` only on pure constructors with no array inputs (`make_uniform_masks`, `uniform_randn/zeros`).
- **Structural errors:** `assert` → `ValueError`/`RuntimeError` with messages (structural-vs-numerical).
- **Frontend:** clean OO class (methods, not module-level `ut3_*` functions); masks follow core dtype.
- **Docstrings:** repair the mangled `import … >>>` lines and stale `t3.t3_corewise_randn` references
  (use `t3.TuckerTensorTrain.randn`); examples must be run and pasted (`docs/doctest_style.md`).

---

## Data structure ✅

`UniformTuckerTensorTrain = tucker_supercore + tt_supercore + masks`-holder.
- `d`-leading supercores `(d,)+stack_shape+(per-core)`; shape fixed across stack, ranks may vary (variety).
- Boolean masks (closure under add/multiply); masks bundled in an `eq=False` identity-hashed holder
  (static `aux_data`), supercores are dynamic children.
- Repairs: `tt_supercore` comment `(d+1,)`→`(d,)`; `assert`→`ValueError` in `validate`.

## Conversions + masking ✅ — REWRITE (uniform-specific; no ragged twin)

- `t3_to_ut3`, `ut3_to_t3`, `make_uniform_masks`, `apply_masks_to_cores`: inherently uniform (pad/stack/
  mask). Just repair imports + dispatch.
- `ut3_to_t3` returns a **tree** of ragged T3s (one per stack element), never auto-stacks — honest for a
  varying-rank stack (`uniform_ranks_and_varieties.md`); caller stacks if uniform-rank.
- `make_uniform_masks` is a pure constructor (keeps `use_jax`); simplify the recursive `_func1` to a
  vectorized `arange(pad) < ranks[..., None]`.

## to_dense ✅ — SHARE the contraction, wrap it

- Factor the chain contraction (`t3_operations.to_dense` body, ≈ lines 78–98) into a representation-agnostic
  helper `_t3_chain_contract` (already works on tuple *or* supercore: it `zip`s/indexes cores and uses
  `'...'` for the stack).
- ragged `to_dense` = `broadcast_to_common_stack` → helper. uniform `to_dense` = `apply_masks` → helper →
  **static prefix-slice** to `shape` (`shape_mask` is always prefix; `shape` is concrete → jit-friendly,
  no `argwhere`). Replaces the current round-trip via `ut3_to_t3`. Not perf-critical (inspection/tests only).

## Linear algebra ✅

- **`__mul__` (scale) / `__neg__`** — TRIVIAL, uniform-specific: scale the last TT supercore slice;
  `neg = ·(−1)`. No masking (rank-preserving). Ragged twin scales last Tucker core; both one-liners → don't share.
- **`__add__` / `__sub__`** — REWRITE, uniform-specific (BATCH over `d`). Direct sum: `xnp.block`
  block-diagonal TT supercore + concat Tucker supercore along the rank axis (all `d` at once), and merge
  masks (`tucker/tt` concat, `shape` OR) — the "add → concatenate masks" algebra. `x`,`y` need not share
  padding `n`/`r`; only `N`,`d`,`stack_shape` match (structural `ValueError`). `squash_tails` after. Repair
  existing `ut3_add`; `sub = add(x, -y)`.

---

## Pending ⬜

- ⬜ **Orthogonalization** (Tucker-core: BATCH/REWRITE; TT left/right: SWEEP/SHARE) — next.
- ⬜ **inner / norm** (uses orthogonalization)
- ⬜ **sum / sum_stack**
- ⬜ **ut3svd** (mask-based truncation; no rtol/atol)
- ⬜ **entries / apply / probe** (+ transposes?)
- ⬜ **stack / unstack**
- ⬜ **constructors** (zeros/ones/randn; from_canonical/from_tensor_train; from_vector/to_vector) **+ IO** (save/load)
- ⬜ **frontend class assembly** (methods, properties, validate, repr, copy, to_jax/to_numpy) + tests/doctests
- ⬜ **jax-wiring** (pytree registration: identity-hashed mask holder; resolve aux hashability)
