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
  `use_jax` only on the **supercore** pure constructors (`uniform_randn/zeros/ones`). **Masks are ALWAYS
  numpy** — all mask logic uses `np`, not `xnp` (host static structure, required for jit; see
  `uniform_pytree_composition.md`). So `make_uniform_masks` and the mask builders take **no `use_jax`**
  (they always emit numpy), and `supercores → xnp; masks → np` is the rule. Don't "fix" mask `np.*` to
  `xnp`.
- **Structural errors:** `assert` → `ValueError`/`RuntimeError` with messages (structural-vs-numerical).
- **Frontend = thin wrappers (backend/frontend razor):** every frontend operation must be reproducible
  on the raw `.data` tuple `(supercore, supercore, (3 masks))` via a backend `ut3_*` function
  (`.data → .data`, or `→ dense`/scalar); the method just calls it and re-wraps via `_from_data`. All
  nontrivial logic (mask/rank recomputation, squash boundary masks, `to_dense` slicing, orthogonalization
  recurrences) lives in the backend — only the OO-class/`UT3Masks` construction stays frontend-side.
  Exception: genuinely trivial one-liners a user would write faster than find. Clean OO class (methods,
  not free functions); masks are always **numpy** (host structure; `np` not `xnp` — see the Dispatch
  bullet above and `uniform_pytree_composition.md`), never following the supercore dtype.
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
- `make_uniform_masks` is a pure constructor that **always emits numpy masks** (no `use_jax` — masks
  are host structure); simplify the recursive `_func1` to a vectorized `np.arange(pad) < ranks[..., None]`.

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

## entries / apply / probe (+ transposes) ✅

- **`entries`** — SHARE (done): `entries.tucker_tensor_train_entries` already dispatches
  `is_uniform = is_ndarray(x[0])`. Frontend re-masks; round-trip test.
- **`apply`** — SHARE after a one-liner: `apply.tucker_tensor_train_apply` hardcodes
  `get_backend(False, …)` ("uniform NOT IMPLEMENTED YET"); change to `is_uniform = is_ndarray(x[0])`
  like `entries`.
- **`probe`** — SHARE + verify: `probing.probe_t3` already has `is_uniform` branches (einsum-over-`d`
  for uniform). Verify the uniform path by round-trip; repair imports/`use_jax`.
- **Transposes** (`apply_transpose` / `entries_transpose`) — IN SCOPE (core feature, not deferred).
  REWRITE the assembly to build a uniform supercore (+ masks), vectorized over `d`. Needs new
  **`d`-folded named contractions** (the outer-product builders have no `d`-sibling; only 3 `d`-variants
  exist today) — audit each transpose against its active block pattern and add the missing ones,
  oracle-tested in `tests/backend/test_contractions.py`. `probe_transpose` is tangent → deferred.
- **Contraction naming = batch-group type signature** (library-wide; add to `batching_and_stacking.md`):
  every call site uses a named contraction matching its active groups. If a fewer-group contraction
  subsumes it via a **shared aligned prefix** that flattens in, add the full-group name that
  **delegates** (reshape → call → reshape) and call that. A group on **some operands only** (e.g. `d`
  on cores but not probes) is a genuine new implementation. Both get a name + a test (delegating ones:
  thin/transitive).

## stack / unstack ✅ — SHARE generic machinery + thin uniform wrapper

- `stacking.py` (`stack`/`unstack`/`apply_func_to_leaf_subtrees`/`get_first_leaf`/`tree_zip`) is already
  generic (takes `axes`) → SHARE. `ut3_stack`/`ut3_unstack` are thin uniform wrappers handling
  (1) the `d`-leading axis offset (`axes = range(1, 1+len(stack))`); (2) `shape_mask` is shared (no
  stack) — replicated onto each leaf on unstack, must-match across leaves on stack.
- Backend works on the raw 5-tuple; frontend reconstructs the `UT3Masks` holder per leaf.
- Variety-OK: leaves share padded `n,r` with per-leaf rank masks; `stack` requires matching padded `n,r`.
- Repair: imports/`use_jax`, holder reconstruction. Test: round-trip + per-leaf vs `t3_to_ut3(x.unstack()[i])`.

## constructors + IO ✅

- **`zeros` / `ones` / `randn`** — REWRITE, uniform-specific, pure constructors (keep `use_jax`). Build
  supercores + masks directly (derive padded `n=max(tucker_ranks)`, `r=max(tt_ranks)`, `N=max(shape)`),
  apply masks. `tucker_ranks`/`tt_ranks` may be **per-stack-element arrays** (variety / rank batches),
  which direct construction handles and a ragged round-trip cannot.
- **`from_canonical` / `from_tensor_train` / `to_tensor_train`** — reuse the verified **ragged** ops via
  round-trip (`t3_from_canonical` etc. ∘ `t3_to_ut3` / `ut3_to_t3`); one-time, not perf-critical, ranks
  uniform so the round-trip is faithful.
- **`save` / `load`** — SHARE `common.save_core_families` / `load_core_families` on the 5 arrays
  (2 supercores + 3 masks); thin uniform wrapper; `load` keeps `use_jax`.
- **`to_vector` / `from_vector`** — **DROPPED** (see Deferred): uniform `to_vector` (real entries) just
  equals ragged `to_vector`, and both optimization paths bypass it.

## frontend class assembly ✅

- Wire decided backends into a clean OO class (methods, not free `ut3_*` functions — un-jumble), mirroring
  `TuckerTensorTrain`, respecting the composition `UT3 = tucker_supercore + tt_supercore + masks`-holder.
- **`.data` is nested, mirroring the fields:** `(tucker_supercore, tt_supercore, (shape_mask,
  tucker_edge_mask, tt_edge_mask))` — supercores flat, the three masks grouped as a sub-tuple (the
  holder's raw arrays). `.supercores = .data[:2]`, `.masks = .data[2]`. **Backend `ut3_*` functions take
  this nested layout** (supercore-only ops take `.data[:2]`; mask-using ops unpack `.data[2]`) — update
  signatures from the old flat 5-tuple. Raw arrays (razor); holder is frontend-only.
- Properties (`d,n,N,r,stack_shape,uniform_structure,shape,tucker_ranks,tt_ranks,structure`) all derived.
  `validate` → structural `ValueError`s. `to_jax`/`to_numpy` convert the **supercores only** — masks
  stay numpy (host structure; jit-required, see `uniform_pytree_composition.md`). `copy` trivial.
  `repr` = structure summary. Repair `reverse`/`squash_tails`/`apply_masks`.

## jax-wiring (in progress — pytree registered; mask `np` refactor + guard + tests pending)

- Register pytree: children `=(tucker_supercore, tt_supercore)`, aux `=` the `eq=False` `UT3Masks` holder
  (identity hash/eq → contents never hashed → solves aux-hashability; the `T3Basis`↔`T3Tangent` pattern). **Done.**
- **Masks are host numpy, computed with `np`** (jit-required): the `xnp → np` mask refactor (builders,
  rank recurrences, `+`/`×`, `int(mask.sum())` extraction; `to_jax`/`make_uniform_masks` keep/emit numpy).
- **Tracer guard** (a traced mask → clear structural error leading with the close-over fix) + the
  **`HOST bool, static`** signature-comment contract on mask args (`docs/signature_style.md`).
- Coverage à la `test_dispatch`: jit each uniform op (a stray `np.*` on a *traced supercore* raises → no
  hidden numpy; mask `np.*` is on concrete host arrays, so it's fine and intentional); jax-in → jax-out.
  Plus a **right/wrong functional doctest** (close-over works; masks-as-traced-args trips the guard) so
  the no-frontend user is covered. Detailed ordered build plan: `docs/uniform_slice_handoff.md` slice 7.

---

## Implementation order (suggested slices — each independently testable by round-trip vs ragged)

1. **Foundation — ✅ DONE.** imports/flatten; clean `UT3Masks` holder + `UniformTuckerTensorTrain`
   (`validate`→`ValueError`, properties, `repr`, `to_dense`, `apply_masks`, `reverse`, `squash_tails`,
   `unstack`/`stack`, `to_jax`/`to_numpy`/`copy`); conversions (`t3_to_ut3`/`ut3_to_t3`, nested `.data`,
   inferred dispatch); masking (`make_uniform_masks` vectorized; `apply_masks` nested-layout);
   `to_dense` = mask → shared `t3_to_dense_chain` (factored out of ragged `to_dense`) → static
   prefix-slice. Tests `tests/test_uniform_tucker_tensor_train.py` (13, incl. a varying-rank-stack
   variety test) + full ragged regression green. **jax pytree registration deferred to slice 7** (the
   class works in numpy; `to_jax`/`to_numpy` convert dtype but it is not yet a registered pytree).
2. **Orthogonalization — ✅ DONE.** `down_orthogonalize_tucker_cores` / `up_orthogonalize_tt_cores` =
   batched-SVD rewrites (`ut3_orthogonalization`, renamed to the keystone names, dispatch migrated, fixed
   a hidden `np.einsum`); `left`/`right_orthogonalize_tt_cores` SHARE the polymorphic
   `orthogonalization.py` sweep (fixed its `if xs[0]:` → `len(xs[0]) > 0` so the uniform-supercore path
   works — was never exercised before). Re-mask on entry suffices (`R = ss·Vᵀ` zeroes padded slots).
   Ranks shrink to the structural minimum the SVD yields, with masks recomputed via per-op rules
   (`min(shape, n)`, `min(n, rL·rR)`, and the L→R / R→L bond recurrences). Verified vs ragged on
   minimal, **non-minimal** (tucker 8→5; tt `(1,40,40,1)→(1,5,25,1)`/`(1,25,5,1)`), and stacked. Full
   ragged regression still green.
3. **Linear algebra — ✅ DONE.** `__mul__`/`__neg__` (scale last Tucker slice), `__add__`/`__sub__`
   (`ut3_add`: block-diagonal TT + concat Tucker/masks, then squash; `x`,`y` may differ in `n`/`r`),
   `sum_stack` (`ut3_sum_stack`: super-diagonal block merge of the whole stack via three identities, then
   squash), `inner`/`norm` (frontend orthogonalizes via the slice-2 methods, then a backend mask → squash
   → `absorb_tucker_into_tt` → scan-zipper; `norm` fast path = ‖last TT core‖). Factored the shared
   `absorb_tucker_into_tt` in `t3_operations` (now also opens `to_dense`'s chain). Verified vs dense
   (scale/neg/add/sub ~1e-16; inner/norm both orth paths ~1e-15; sum_stack ~1e-16), unstacked + stacked.
4. **Sampling — ✅ DONE.** `entries`/`apply`/`probe`/`sum` (full). Backend `ut3_sampling` wrappers
   re-mask, then call the SHARED `entries`/`apply`/`probing.probe_t3` on the masked supercores (vectors
   zero-padded to `N`; probe results sliced back to the real shape; `entries` needs no packing). Fixed
   `apply.tucker_tensor_train_apply`'s hardcoded `is_uniform=False` → `is_ndarray(x[0])` (now a real
   `lax.scan`, like `entries`). Full `sum` = `apply` with all-ones; partial `sum` raises NotImplementedError
   (deferred). Verified vs keystone/dense (entries/apply/probe exact; sum ~1e-15), incl. stacked T3s and
   W-stacked vectors/probes; ragged + jax regression green.
5. **ut3svd — ✅ DONE.** Mask-truncation T3-SVD: `ut3svd(data, max_*_ranks)` caps by `min(current, max)`,
   takes the minimal STRUCTURAL ranks (`compute_minimal_ranks`), builds prefix truncation masks, runs the
   sweep (`uniform_t3_svd`: orthogonalize + scan of per-edge SVDs, pad factors back, multiply by masks),
   then shrinks the padded supercore to those ranks. No `rtol`/`atol` (rejected; would be data-dependent);
   per-stack-element `max_*_ranks` allowed (the variety). Extracted `down_orthogonalize_tucker_supercores`
   (supercore-level, reused by the sweep). Verified vs `t3svd`: tensor matches (≤3e-15) on no-truncation
   (ranks match ragged), truncation, non-minimal, stacked, and per-stack-element caps. **Note:** under
   truncation, uniform yields ranks ≤ ragged's (it applies the full `rL·rR` structural bound that ragged's
   sweep-order leaves behind) — same tensor, tidier; the contract is on the represented tensor.
6. **Transposes** — `apply_transpose`/`entries_transpose`; audit + add `d`-folded contractions (oracle-tested).
7. **jax-wiring** — pytree registration + `test_dispatch` coverage.
8. **Constructors + IO** — `zeros`/`ones`/`randn`; `from_canonical`/`from_tensor_train`/`to_tensor_train`
   (ragged round-trip); `save`/`load`.
9. **`t3m`** (elementwise multiply + truncation) — depends on `ut3svd`. Batched multiply (fold `d` into
   `t3_mult`'s einsums); **masks combine by Kronecker** (the ⊗ side — `ut3_add` is the ⊕/concat side;
   needs a small Kronecker-mask helper). Truncation by **max-rank masks only** (no `rtol`/`atol`;
   per-stack-element caps OK), shrinking to minimal structural ranks. **Two methods, `inplace_fused` the
   default:** `form_then_round` (full product → `ut3svd`) and a max-rank **`inplace_fused`** sweep that
   caps each bond per-step and **never materializes the full product** — the memory-critical path, since
   `t3m` ranks grow multiplicatively (`n_x·n_y`, `r_x·r_y`). No `swap`/`oversample` (those recover
   `rtol`/`t3svd` quality and fight the `d`-leading scan layout).

Deferred (wanted): partial `sum`; `to_vector`/`from_vector` (intentionally omitted — route via ragged + jax-pytree).

## Deferred (return later — wanted)

- **Partial `sum`** (sum a *subset* of physical axes → a smaller uniform T3). A structural reduction:
  contract the summed Tucker cores with `ones`, fold the resulting bond-matrices into neighbors, and
  rebuild a smaller (`d' < d`) supercore + masks — a sweep that re-shapes. Self-contained; addable
  later without touching the rest of the layer. (Full-sum is already covered via `apply`-ones.)
- **`to_vector` / `from_vector`** — the **ragged-signature** version is **permanently ruled out** (not
  merely deferred). Decisive reason: a **varying-rank stack has no single flat-vector reconstruction
  signature** — the ragged `from_vector(vec, shape, ranks)` contract cannot encode per-element ranks; and
  for a uniform-rank stack the real-entries vector is *identical* to ragged's anyway. Both interop paths
  bypass it: JAX-native optimization operates on the **pytree directly** (supercores = differentiable
  children; padding has zero gradient and stays inert); flat-vector/scipy interop routes through ragged —
  `ut3_to_t3` → `TuckerTensorTrain.to_vector` — clean **only for a uniform-rank stack** (a varying-rank
  `ut3_to_t3` returns a *tree*: pick one stratum first, and flat-vector optimization across strata is
  ill-posed regardless). If a concrete need surfaces it would be a **new, distinct, mask-carrying method**
  (real entries + masks passed separately), never the ragged `to_vector` contract.
- **Property checkers — placement.** `has_minimal_ranks` belongs on `UniformTuckerTensorTrain` (a
  structural-rank property, mirroring the keystone). `is_orthogonal`/`is_gauged` are *frame* properties of
  the deferred uniform **basis** layer (`UT3Basis`), **not** of plain UT3. When built, all property
  checkers follow the equivalence-contract behavior: evaluate on the **real (masked) sub-blocks**, **per
  stack element at the realized rank** (`mask.sum(-1)`), and **report a bool, never raise** (numerical
  property → non-enforcing, per the structural-vs-numerical guard).
- **`t3m`** — promoted to planned slice #9 (above); the `method=` open question is resolved there
  (both `form_then_round` and a default max-rank `inplace_fused` sweep).

---

## Validation (docs stress-tested)

These notes were validated by 5 fresh **context-free** agents (no access to the design conversation),
each posed one design question neutrally: variety/ranks, masks-vs-integer-ranks, `to_vector` keep/drop,
and two **generalization** tests on un-walked ops (`is_orthogonal`, `t3m`). **All five reached the
documented decisions** — the two generalization ops were *derived* from the principles — confirming the
notes stand on their own. The precision/scope fixes they suggested are folded in above; the one genuinely
*undecided* point they surfaced — the `t3m` `method=` question — is now resolved (slice #9: both
`form_then_round` and a default max-rank `inplace_fused` sweep).
