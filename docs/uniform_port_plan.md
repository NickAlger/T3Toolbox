# UniformTuckerTensorTrain port plan (living document)

> A running, function-by-function record of how the uniform T3 layer is being rebuilt: for each piece,
> whether we **share** the existing ragged backend (polymorphism "just works"), **rewrite** it
> uniform-specifically (a real structural difference or a vectorization win), or leave it **trivial**
> (inline one-liner), plus the reasoning. This is the implementation checklist; the *why* behind the
> cross-cutting design decisions lives in the four design notes (see below). Updated as we walk through
> the functions; entries marked ✅ are decided, ⬜ are pending.

**Governing contract** ([`uniform_equivalence_contract.md`](uniform_equivalence_contract.md)): the
uniform layer is a *faster ragged layer* — `to_uniform → op_uniform → to_ragged == op_ragged` on the
real (masked) parts; garbage in padding is don't-care. This defines correctness for every op below and
**the test strategy**: round-trip each op against its ragged twin (compare `to_dense`s or convert back),
real parts only. Masking schedule: mask on entry + re-mask after any SVD; certified by the round-trip
test, not proven a priori.

**Hybrid principle** (per `CLAUDE.md`): share where polymorphism just works; rewrite where there's a
legitimate structural difference, where polymorphism would need unnatural branching, or where splitting
yields a performance gain (typically: ragged *maps over the mode index `d`*, uniform *folds `d` into a
batched array op / einsum*).

**Design notes (the "why"):** [`uniform_equivalence_contract.md`](uniform_equivalence_contract.md)
(**governing**), [`uniform_ranks_and_varieties.md`](uniform_ranks_and_varieties.md),
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

## Orthogonalization ✅ — split verdict (the cleanest hybrid example)

- **Tucker-core orthogonalization** (keystone `down_orthogonalize_tucker_cores` / `up_orthogonalize_tt_cores`)
  — REWRITE, uniform-specific, **BATCH**. Core-local (each core's SVD pushes its remainder into the
  same-index neighbor; no inter-core dependency), so uniform does one batched `xnp.linalg.svd` over the
  `(d,)+stack` axes vs. ragged's `xmap` over the tuple — a real GPU win. Existing
  `up_orthogonalize_uniform_tucker_cores` / `down_orthogonalize_uniform_tt_cores`.
- **TT left/right orthogonalization** — SHARE, **SWEEP** (already done). Sequential (each core's `R`
  pushes into the next), so it must scan; `orthogonalization.py` is already polymorphic (`is_ndarray`
  dispatch, `xscan`→`lax.scan`, uniform-aware `left_svd_pair`). Uniform frontend already calls it.
- Repairs: fix the `self.apply_masks_to_cores` bug (re-mask via `ut3_masking` per the masking schedule);
  align flipped method names to the keystone; fix docstrings calling nonexistent methods.
- **Verify (not-minimal ranks):** match whatever ragged orthogonalization does to a non-minimal core
  (keep-with-completion vs reduce-and-update-mask) — read/run the ragged path, certify by round-trip.
  This is the spot that was mid-refactor when copied in.

---

## inner / norm ✅ — uniform orchestration reusing decided pieces

- REWRITE (uniform-specific orchestration): re-mask → optional orthogonalize (the decided split) →
  absorb Tucker into TT → zipper. `norm` fast path = `‖last TT core‖_F` when orthogonalized, else
  `inner(x, x)`.
- The zipper (carry-`M` contraction `'...ab,...aoc,...bod->...cd'`) is a SWEEP → uniform `xscan`; the
  ragged twin isn't factored, so leave the uniform zipper its own scan.
- **Factor a shared primitive** `absorb_tucker_into_tt(tucker_cores, tt_cores)` (the per-core `G·U`
  batched einsum) — reused by `to_dense` and `inner`, polymorphic over tuple/supercore.
- Test: round-trip `ut3.inner(ux, uy)` vs `x.inner(y)`.

## sum / sum_stack ✅ (partial-`sum` deferred)

- **`sum_stack`** (genuine tensor sum over the stack) — REWRITE, uniform-specific: direct sum over the
  stack (block-diagonal TT, concat Tucker ranks, squash) — `add` folded over the whole stack axis at
  once. **Not** `supercore.sum(stack_axes)` (that's corewise — the commented-out `ut3_sum_stack` had
  this wrong).
- **`sum` over physical axes — full (all axes → scalar)**: reuse `apply` with `ones(Ni)` vectors (a
  sweep we build for `apply`); masking makes `ones(N)` sum only the real columns.
- **`sum_stack_corewise`**: corewise op; defer to `corewise.py` (don't reimplement in `ut3_*`).
- **Partial `sum`** (subset of physical axes): DEFERRED — see "Deferred" below.
- Test: round-trip `sum_stack` / full-`sum` vs ragged.

## ut3svd ✅ — REWRITE (the defining uniform op)

- Uniform-specific mask-truncation sweep (scan over `d`, pad factors back, multiply by **prefix** masks),
  reusing the shared orthogonalization (batched Tucker rewrite + shared TT sweep) and the polymorphic
  `linalg` SVD primitives. Truncation = **max-rank masks only, no `rtol`/`atol`** (per the contract);
  per-stack-element max-ranks allowed (the variety / rank-sweep payoff).
- **Output ranks shrink to the minimal STRUCTURAL ranks** (`compute_minimal_ranks(structure, max_ranks)`)
  and the **padded supercore shrinks to match** (`n' = max minimal tucker rank`, `r' = max minimal tt
  rank`, over modes + stack) — dropping the current "pad back to original `n` for consistency" hack.
  jit-safe: the new size comes from the *static* structure/aux, not from values. Explicitly **not**
  numerical-zero dropping (data-dependent → the forbidden `rtol=0`). Ranks only ever shrink
  (minimal ≤ original), never grow.
- Works for **non-minimal** inputs (the old "only minimal" comment was a caution, not a target); verify
  by round-trip vs `t3.t3svd(max_ranks)`. The T3-specific wrinkle (Tucker rank shifts as a side-effect of
  TT-rank reduction — absent in pure TT-SVD) is expected.

## Pending ⬜

- ⬜ **entries / apply / probe** (+ transposes?) — next.
- ⬜ **stack / unstack**
- ⬜ **constructors** (zeros/ones/randn; from_canonical/from_tensor_train; from_vector/to_vector) **+ IO** (save/load)
- ⬜ **frontend class assembly** (methods, properties, validate, repr, copy, to_jax/to_numpy) + tests/doctests
- ⬜ **jax-wiring** (pytree registration: identity-hashed mask holder; resolve aux hashability)

## Deferred (return later — wanted)

- **Partial `sum`** (sum a *subset* of physical axes → a smaller uniform T3). A structural reduction:
  contract the summed Tucker cores with `ones`, fold the resulting bond-matrices into neighbors, and
  rebuild a smaller (`d' < d`) supercore + masks — a sweep that re-shapes. Self-contained; addable
  later without touching the rest of the layer. (Full-sum is already covered via `apply`-ones.)
