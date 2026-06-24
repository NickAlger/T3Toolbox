# Uniform-layer fix — design & plan (the 1.0 centerpiece)

_In progress (2026-06-21). Planning. Subsumes the backend module reorg + the per-op polymorphism
triage (see `naming_review.md` §4). Goal: make the uniform layer run, polymorphic-where-right, fast,
and tested — mirroring the ragged layer._

## Plan slices (A–E, with triage + reorg threaded through)
1. **Triage survey** ("understand"): catalog uniform-relevant ops, classify each by the lenses below,
   assess current state (runs / stubbed / buggy — esp. the `ut3_sampling` packing bug). Output: per-op
   plan + target module layout.
2. **Sampling polymorphic** (probe/apply/entries) — acts-on ops; hooks already in `probing.py`; fix the
   packing bug. Unblocks fitting/optimizers (C).
3. **Uniform frame–variations + tangent layer** (`ufv_`/`utv_`) → OO-frontend + functional-backend
   mirroring ragged (B).
4. **Optimizers/fitting on uniform** (C — the speed payoff).
5. **Derivative probing on uniform** (D). Tests/docstrings/doctests woven throughout (E).

## Polymorphism triage — LENSES, not hard rules
Apply with judgment; no loophole-hunting. Agents *gather state + surface reasoning*; **humans classify**;
edge cases get flagged, not auto-resolved.

- **Lens (can it?): what does it RETURN.** Returns vectors / scalars / a dense array (a *read-out*) →
  **polymorphic** via mask-once. Returns a T3-like object (T3 / frame / tangent; restructuring) → **lean
  uniform-specific** (masks entangle with the result's structure). Canonical: `inner`/`norm` *touch* the
  T3 but *return a scalar* → polymorphic. (Cruder phrasing: acts-*on*-data vs acts-*to*-self.)
- **Lens (how, if polymorphic): scan vs map.** **Scans over cores** (carry state along the TT) → one
  polymorphic scan (`xscan`). **Maps over cores** → NOT a polymorphic `xmap`; the uniform path is a single
  **einsum over the leading core axis** (vectorized; perf).
- **Mechanism — mask-once:** for read-outs, apply masks to the supercores *once upfront* (zero the
  garbage), then run the core algorithm **mask-free** → same code ragged/uniform. Cheaper than masking
  data in-flight. Acts-to ops can't (they mask running intermediates) → uniform-specific.
- **Why it works:** supercore leading axis = core index, so `supercore[i]` ≈ ragged `cores[i]` — per-core
  code "just works."
- **Dispatch:** ragged/uniform is *inferred* at the lowest level (`is_ndarray(up_tucker_cores)`), like
  numpy/jax. A polymorphic op needs **no `ut3_` twin**.
- **Verify:** the equivalence contract — `to_uniform(x) → op → to_ragged == op_ragged(x)` on *real* parts
  (garbage don't-care).

## Packed vectors (the sampling I/O)
- Uniform mode-vectors are **packed** — one stacked, length-`N` array, not a ragged list — and **stay
  packed** through the computation. (Mismatch here is the likely `ut3_sampling` bug: list vs packed.)
- **Real entries sit in a contiguous PREFIX `[0:nᵢ]`** — RESOLVED against the code: the mode/shape mask is
  `np.arange(N) < nᵢ` (`ut3_masking.py:72`) and is passed through every op untouched; it does **not** scatter
  (only the *rank* masks scatter, via concat/Kronecker). So: **pack** `packed[:nᵢ] = ragged`, `packed[nᵢ:] = 0`;
  **unpack** `ragged = packed[:nᵢ]`. Round-trips; the supercore's real mode-entries are the same prefix.
- **Fill = zeros by default; MUST be finite.** NaN/inf breaks correctness (`0 × NaN = NaN` poisons the
  masked reduction — mask-once relies on `0 × garbage = 0`). Padding values are don't-care for correctness;
  zeros are the robust choice (correct even if the cores' mode-mask is absent). Packed vectors always
  travel with their **mask** — the fill is never a reliable shape encoding (entries can be 0 by coincidence,
  and the real entries are scattered, not a prefix).
- **Canonical axis order** (leading→trailing): `[tensor-mode d]` → `[derivative order]` → `[stacks W,K,C]`
  (base-inner) → `[mode-data axis]`. Uniform array = ragged element-shape with `d` prepended (so
  `uniform[i]` = the ragged element → polymorphism). Trailing data axis is `N` (uniform) / `nᵢ` (ragged) —
  the one shape difference pack/unpack reconcile.
- **pack/unpack = standalone pure functions** (`ut3_operations.pack_vectors`/`unpack_vectors`) taking the
  **shape int tuple `(n₀,…,n_{d-1})`** (+ `N` from the supercore mode axis), **not** a mask — the mode shape
  is a prefix, fully captured by the ints. Frontend surfaces **one helper, not per-class methods** (vectors
  are external data, not owned by any uniform object). *(Resolved: the mode/shape mask does NOT scatter —
  verified `np.arange(N) < shape` + pass-through; only rank masks scatter. See the shape-tuple decision below.)*

## Shape: store the int tuple, not a `shape_mask` (DECIDED — verified in code)
**Replace `shape_mask` (bool `(d,N)`) with the shape int tuple `(n₀,…,n_{d-1})`**, promoted OUT of `masks`
(so `masks` = the rank masks `(tucker_edge_mask, tt_edge_mask)`; `shape` is its own static field). `N` is
recovered from the supercore mode axis — nothing lost.

### Slice 2 — DECIDED & IN PROGRESS (2026-06-23)
- **`.data` representation: (A) 4-arity flat** — `.data = (tk_sc, tt_sc, shape, (tkm, ttm))`. `shape` is a
  sibling dataclass field (it is genuinely **not a mask**: no stack axis, value-hashable, prefix-enforced
  by construction); `UT3Masks` shrinks to the two rank masks; pytree aux = `(shape, masks)`. Every
  mask-using backend op's unpack changes `a, b, masks = x` → `a, b, shape, masks = x` (mechanical).
  `make_uniform_masks(...)` returns `(tkm, ttm)`; `apply_masks_to_cores` reconstructs the boolean
  `shape_mask` on the **host** (`np.arange(N) < np.array(shape)[:,None]`) for its einsum; `uniform_t3_svd`'s
  `rank_truncation_masks` becomes the 2-tuple `(tkm, ttm)`; `ut3_to_t3`/`ut3_to_dense` drop the
  over-engineered `np.argwhere` on the shape mask for a prefix slice `[:Ni]`.
- **Scope: the plain `ut3_` layer ONLY** — `UniformTuckerTensorTrain` + the 8 `ut3_*` backend modules +
  `tests/test_uniform_tucker_tensor_train.py` (~9 files, suite-gated 49/49). The bv/manifold layer's
  `shape_mask` (`ubv_masking`, `ubv_conversions`, `uniform_basis_variations_format`, `uniform_manifold`)
  is **independent and broken**; it is rebuilt directly on the int-tuple convention in Slice 3 — not
  migrated twice. (Verified: `ut3_masking` is consumed only within the `ut3_` layer; the bv layer has its
  own `ubv_masking`; the only non-prefix `shape_mask`s anywhere are two `np.random.choice` doctests in the
  bv layer.)
- **jit value-hashing is a PARTIAL win (honesty caveat).** Promoting `shape` to a value-hashable int tuple
  removes it from the retrace triggers and kills the `int(mask.sum())`-on-a-tracer trap — a strict
  improvement. But the rank masks stay bool arrays in the `eq=False` identity-hashed `UT3Masks`, so a
  fresh-but-identical holder still retraces; "same shape → same compiled program" only fully lands once the
  rank masks are also value-keyed (`tobytes()`/frozen tuples). **Deferred** — revisit only if profiling
  shows retrace cost. The doc's jit claim should be softened to match.

- **Why (code):** `shape_mask = np.arange(N) < shape[:,None]` is always a contiguous prefix and is passed
  through every algebraic op untouched (only *ranks* scatter via concat/Kronecker — `ut3_svd` keeps
  `shape_mask`, comments it "unused"). Most consumers already `.sum(axis=-1)` it back to ints. It's a
  redundant encoding of the tuple, and structurally special already (no stack axis — "shared across the stack").
- **Wins:** deletes the `.sum()` round-trips; mode-masking becomes a static slice / host-reconstructed boolean;
  and it **enforces the prefix invariant structurally** (you can't write a scattered shape with ints) — today's
  doctests build scattered `np.random.choice` shape masks, an unenforced footgun.
- **jit (the priority — uniform is mostly jitted): strictly better.**
  - Int tuple = **value-hashable static aux_data** → value-based jit cache key (same shape → same compiled
    program). The current bool-mask holder is `eq=False` *identity*-hashed → can over-retrace; the tuple fixes
    that for shape.
  - No `int(mask.sum())` on a possibly-traced array — `shape` is already concrete ints (sidesteps the
    "jnp-on-a-mask → tracer → `int()` breaks" trap).
  - `N = supercore.shape[mode_axis]` is a static int under jit; retracing-on-shape-change is correct and unchanged.
  - **Discipline:** where the boolean is still needed (`apply_masks_to_cores` einsum), reconstruct with **`np`**
    (host) from the static ints — `np.arange(N) < np.array(shape)[:,None]` → a numpy constant folded into the
    program. NEVER `jnp` in the trace (the existing "masks → `np`" rule, applied to on-demand reconstruction).
- **Migration (~12 files):** `.sum()`-sites collapse to direct access; ~3 sites reconstruct the boolean / use
  `range(nᵢ)`; update `.data`/pytree/`validate`/type-annotations; promote `shape` out of `masks`. Verify
  (grep + equivalence-contract tests) that no path relies on a non-prefix `shape_mask`.

## Triage survey findings (2026-06-21) — three-way split
- **Plain uniform layer (`ut3_*`): SOLID, tested (49/49).** No stubs. Leave alone. `shape_mask`→tuple
  migration confirmed safe (all consumers prefix-compatible; the `ut3_to_t3` `argwhere` is over-engineered).
- **Ragged sampling polymorphism: MOSTLY ALREADY DONE.** `probe_t3` + helper chain, `apply`/`entries`,
  `probe_tangent`, `probe_tangent_transpose` already dispatch on `is_uniform`. **Gaps:** `apply_tangent_transpose`
  / `entries_tangent_transpose` route through `_apply_transpose_adjoint` (probing.py:1034–1061, ragged-only) →
  the 3 `*_corewise_transpose` inherit it; and `apply_tangent`/`entries_tangent` signatures still require a
  `Sequence`.
- **Uniform tangent/frame layer: THE BROKEN CENTERPIECE.** (1) 3 broken imports in
  `uniform_basis_variations_format.py` (lines 10/11/17 — module can't import; **fix first**). (2)
  `uniform_manifold.py` imports `OLD_uniform`, uses OLD types, 4/7 functions `if False:` stubs. (3) `ubv_to_ut3`
  stubbed. (4) No `UT3Tangent` class; 10+ tangent ops have no uniform version; zero tangent-layer tests.
  The `UT3Basis`/`UT3Variations` classes themselves run once imports are fixed.

## Decisions from the triage
- **Sampling is strict; packing lives at the boundary.** A sampling op assumes matching operand types
  (uniform cores ↔ packed vectors; ragged cores ↔ ragged vectors); a **mismatch ERRORS** (structural
  house-rule; no silent auto-pack). `pack_vectors`/`unpack_vectors` are explicit boundary helpers. The
  `ut3_sampling` "bug" = packing done *inside* the op — move it out; the op becomes fully polymorphic. (Perf:
  pack once at the boundary, same principle as mask-once.)
- **norm/inner can be polymorphic on BOTH algorithms.** Zipper path: read-out scan, mask-once → polymorphic.
  Orthogonalize-for-stability path: also polymorphic, by an **exactness** argument — every orthogonalization
  step is an *exact* factorization contracted along an *internal* edge, so `step(T3).to_dense() ==
  T3.to_dense()` identically (`Q·(R·C_{i+1}) = (QR)·C_{i+1}`). Hence the padded-slot garbage **provably
  contracts to zero** (else `to_dense` would change), with no mask maintenance and no isolation hand-waving.
  **SVD is fine** (library SVD-only convention preserved): `Q=U`, `R=SVᵀ` is an exact factorization;
  triangularity is irrelevant. **The one caveat is exactness ⇒ NO TRUNCATION:** the read-out orthogonalization
  must be **shape-preserving** (keep all singular values, incl. the padded zeros) and touch no masks — dropping
  a *nonzero* σ changes `to_dense` (breaks it); dropping the *zero/padded* σ is exact but shrinks ranks →
  changes masks → uniform-specific bookkeeping a read-out shouldn't do. So norm/inner need a shape-preserving
  exact orthogonalization, **distinct from the mask-recomputing `left_orthogonalize_tt_cores`**. (Correctness
  is a proof modulo roundoff → a smoke test suffices.) Lens: *orthogonalization returning a T3 →
  uniform-specific; inside a scalar read-out → polymorphic.*

## Agreed slices (next)
1. **Fix the 3 imports** in `uniform_basis_variations_format.py` — ✅ **DONE (2026-06-21)**. Module imports;
   no regression (uniform suite 49/49). Unblocked its doctests, which now RUN and surface concrete Slice-3
   staleness (the "make broken code run" items): `NameError: basis_left_mask` (docstring example);
   `TypeError: basic_uniform_stack() got an unexpected 'use_jax'` (stale API call in
   `ut3_orthogonal_representations` — 26/43 of its doctest failures); scattered-`shape_mask` doctests
   (`np.random.choice`). 35 doctest failures total = the broken-tangent-layer state, now visible.
2. **`shape_mask` → shape int tuple** — ✅ **DONE (2026-06-23).** Plain `ut3_` layer migrated to the
   4-arity-flat `.data = (tk_sc, tt_sc, shape, (tkm, ttm))`; `UT3Masks` shrunk to the two rank masks;
   `shape` a value-hashed pytree-aux field. **The one real wrinkle:** the int-tuple `shape` is a
   `Sequence`, which `stacking.py`'s `isinstance(x, Sequence)` leaf-predicate would recurse into — so
   `ut3_stack`/`ut3_unstack` and the frontend `unstack` use a **dynamic leaf template**
   `ut3_leaf_structure(d) = (None, None, (None,)*d, (None, None))` + a manual first-leaf drill
   (`_first_data_leaf`) to keep `shape` out of the tree walkers. `ut3_save`/`load` serialize `shape` as a
   third family. Full suite green (327 / 39 198 subtests); jax dispatch + doctests green. **Open jit
   caveat carried forward:** value-hashing is shape-only (rank masks stay identity-hashed) — see the
   "Slice 2 DECIDED" note above.
2.5. **Cleanup before the rebuild — ✅ DONE (2026-06-23).** Removed the ambiguous
   `UniformTuckerTensorTrain.{from_canonical, from_tensor_train, to_tensor_train}` frontend methods + their
   backend twins (`ut3_from_canonical`/`ut3_from_tensor_train`/`ut3_to_tensor_train`) + dead imports. They
   took *ragged* CP/TT data and round-tripped through `TuckerTensorTrain` (ragged-vs-uniform input
   ambiguity); users compose `t3_to_ut3` / `ut3_to_t3` explicitly. Tests/doctests updated; the array-in
   jit-dispatch check now exercises `t3_to_ut3`. (Per the razor: trivial-and-obvious round-trips don't earn
   a backend home.)

**Slice 3 is split into 3a + 3b** (decided 2026-06-23) — the old "rebuild the tangent layer" is really two
stacked layers (frame/variations foundation, then tangent/manifold on top); the user wants the foundation
solid first. **Naming decision: DEFER the rename.** 3a keeps the current `UT3Basis`/`UT3Variations`/`ubv_`
names (consistent with ragged `T3Basis` *now*); the global `T3Basis→T3Frame` + `bv_→fv_` + `ubv_→ufv_`
rename (naming_review.md §2) is its own later mechanical, suite+doctest-gated pass over the whole library.
Rationale: one kind of change per slice; keep the working ragged layer stable; don't block the rebuild.

3a. **Frame/variations foundation (the bulk of the rebuild).** Rebuild `UT3Basis` + `UT3Variations`
   **directly in the target shape** (rebuild-in-place: the layer is *broken*, unlike Slice 2's *solid* plain
   layer) — mirror the ragged `T3Basis`/`T3Variations` method-for-method, on:
   - the **int-tuple `shape`** + the **plain-layer pytree composition** (a `UFV*Masks`-style identity-hashed
     aux holder for the rank masks, `shape` value-hashed aux, supercores as the only children — today the
     bv masks are wrongly pytree *children* with no registration). Drops the 9-tuple/7-tuple `.data` to
     supercores + shape + rank-mask holder.
   - the missing ~50 methods (`to_t3`/`to_dense`, `orthogonalize`, geometry hooks, vector conversions,
     linalg, `reverse`, save/load, `from_t3`/`random_orthogonal`, repr…) + `validate` + tests/doctests.
   - the `ubv_*` backend (`ubv_masking`/`ubv_conversions`) migrated to the int-tuple + 5→fewer-mask layout.
3b. **Tangent + manifold.** `UT3Tangent` + `uniform_manifold` rebuilt off the new types (drop the
   `OLD_uniform` import + ~600 lines of `if False:` dead code); un-stub retract/project/transport + the
   geometry; un-stub `ubv_to_ut3`; derivative probing (ragged was built polymorphism-ready); tests.
4. **Close the ragged-poly gaps** (`_apply_transpose_adjoint` polymorphism + the `Sequence` signatures) so
   optimizers/derivatives run on uniform.

**Open verifications:** a norm/inner orthogonalize-path smoke test (correctness is a proof modulo roundoff,
not an open question); confirm no path relies on a non-prefix `shape_mask`.
