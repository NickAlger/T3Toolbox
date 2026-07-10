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
- **jit value-hashing — now FULLY RESOLVED (2026-06-23).** Slice 2 promoted `shape` to a value-hashable
  int tuple but left the rank masks `eq=False` *identity*-hashed (a fresh-but-identical holder still
  retraced — the deferred "partial win" caveat). That deferral is now retired: the rank masks are
  **value-hashed by content** via the `common.ValueHashedMasks` mixin (cached `tobytes` hash + `array_equal`
  eq + `is` fast-path), so a rebuilt-but-identical holder is the *same* jit cache key — "same structure →
  same compiled program" fully lands. This was forced by the optimization use case (the orthogonal frame is
  rebuilt every iteration → fresh holders → identity hashing would recompile *every step*, dwarfing the
  per-step compute). We hash **mask bytes** (not a rank count) because masks scatter off canonical form
  (add = concat → gappy, non-prefix). Empirically: identity → 5 compiles / 5 iters; value → 1. Regression
  test `tests/test_dispatch.py::test_mask_rebuild_does_not_recompile`. Docs updated
  (`docs/uniform_pytree_composition.md`, CLAUDE.md).

### CONSIDERED & REJECTED (2026-06-24): a maskless uniform tangent layer
_(Canonical user-facing write-up: [`docs/uniform_rank_masks_rationale.md`](../docs/uniform_rank_masks_rationale.md).)_
- **The idea:** drop masks from the bv/fv/tangent layer entirely — inflate the frame (SVD orthonormal
  completion) + zero-extend variations to the padded size `R`, work on pure supercores. Motivated by the
  minimal-rank catalog (`docs/numerical_contract_catalog.md`, empirically verified): **no tangent op needs
  minimal rank** (inner/norm need only orthogonal+gauged — it corrects the stale CLAUDE.md "+minimal";
  project/transport/gauge need orthogonal; retract is a soft caveat). The catalog's "non-minimal base vs
  dense oracle" tests *are* inflation tests, so `op(inflate(t)).to_dense() == op(t).to_dense()` is
  essentially pre-verified. Payoff would have been huge: no masks → no static jit-key structure → the whole
  recompile saga evaporates for the tangent layer.
- **Why REJECTED:** the claim holds for **operations on a given tangent**, but **optimization computes new
  tangents (gradients)**. The masks zero the variation in the padded slots — that is what **prevents the
  gradient from growing rank into the padding**. That pin is a **core feature**, not bookkeeping: it lets
  you fit at specific per-edge ranks and grow them under control (`examples/fit_varied_rank_tensor_newton_cg.py`,
  `docs/rank_continuation.md`). Maskless would force uniform rank `R` everywhere → either too low (leave fit
  on the table) or too high (overfit). **Simulating varied ranks within uniform is the point of the layer.**
- **Consequence:** masks stay; **value-hashing is the right recompile fix**; SVD orthogonalization
  (`docs/uniform_svd_prefix_orthogonalization.md`) keeps the masks a deterministic prefix → loop-invariant
  at fixed rank → no recompile within a continuation stage; rank continuation recompiles only at stage
  boundaries (rank changes — correct, rare). Increment 3a proceeds **with** masks, as planned.

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
   third family. Full suite green (327 / 39 198 subtests); jax dispatch + doctests green. _(The Slice-2
   "value-hashing is shape-only" caveat is now **resolved** — rank masks are value-hashed via
   `ValueHashedMasks`; see the "Slice 2 DECIDED" note above.)_
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
   - the **int-tuple `shape`** + the **plain-layer pytree composition** (a `ValueHashedMasks` aux holder for
     the rank masks — value-hashed by content, NOT identity; `shape` value-hashed aux; supercores as the
     only children — today the bv masks are wrongly pytree *children* with no registration). Drops the
     9-tuple/7-tuple `.data` to supercores + shape + rank-mask holder. _(Increment 1 ✅ DONE for `UT3Basis`
     + `UT3BasisMasks`; value-hashing applied across all holders 2026-06-23.)_
   - the missing ~50 methods (`to_t3`/`to_dense`, `orthogonalize`, geometry hooks, vector conversions,
     linalg, `reverse`, save/load, `from_t3`/`random_orthogonal`, repr…) + `validate` + tests/doctests.
   - the `ubv_*` backend (`ubv_masking`/`ubv_conversions`) migrated to the int-tuple + 5→fewer-mask layout.
3b. **Tangent + manifold.** `UT3Tangent` + `uniform_manifold` rebuilt off the new types (drop the
   `OLD_uniform` import + ~600 lines of `if False:` dead code); un-stub retract/project/transport + the
   geometry; un-stub `ubv_to_ut3`; derivative probing (ragged was built polymorphism-ready); tests.
   - **DESIGN CONSTRAINT — jit/recompile (decided 2026-06-24; doc: `docs/uniform_backend_jit_recipe.md`).**
     The backend optimization functions (uniform MC-SGD / Newton-CG / fitting) MUST be structured so the
     masks are **loop-invariant state held across a rank-stage and reused**, with only the supercores (+
     minibatch) traced — recompiling only at rank-continuation stage boundaries (correct + rare). Two valid
     shapes: (a) jit the whole per-step kernel, close over the base masks → `ut3_orthogonal_representations`
     inside re-derives the frame masks as **constant-folded constants** (not a jit cache key → no recompile;
     empirically 1 compile); (b) hold the masks separate and reuse the same objects across iterations (the
     value-hashing idea made explicit at the backend level). The **anti-pattern** to avoid: running
     `ut3_orthogonal_representations` outside a jit and passing its fresh-object masks into a separate jit
     (traced → rejected, or a fresh closure each step → recompile). `ut3_orthogonal_representations` itself
     is correct as-is; this is a constraint on how the optimizer *calls* it.
4. **Close the ragged-poly gaps** (`_apply_transpose_adjoint` polymorphism + the `Sequence` signatures) so
   optimizers/derivatives run on uniform.

**Open verifications:** a norm/inner orthogonalize-path smoke test (correctness is a proof modulo roundoff,
not an open question); confirm no path relies on a non-prefix `shape_mask`.

---

# Increment 2c — detailed plan (2026-06-29)

_The bv-layer foundation buildout: the converters + the `UT3Basis`/`UT3Variations` method buildout,
mirroring ragged `T3Basis`/`T3Variations` **only where the uniform structure earns it**. Designed with
Nick 2026-06-29; PENDING his review before we start. Done in slices 2c-A … 2c-G, one at a time._

## Guiding principle (the triage lens)
- **Does the method use the uniform structure nontrivially?** If there is no computational disadvantage to
  doing it in the ragged layer then converting, *consider not mirroring it*. If it is naturally done
  differently in uniform — a vectorized supercore op (einsum / slice / `[::-1]` over the leading `d` axis)
  where ragged loops core-by-core — it **earns its keep**.
- **Do uniform things directly**, not via ragged methods + conversions, wherever reasonable. (Canonical:
  `bv_to_ut3` must substitute a *slice of the variation supercore into the frame supercore* — NOT convert
  to ragged, call `bv_to_t3`, and convert back, as the old `if False` stub did.)
- Not a hard rule; applied within reason.

Outcome of applying it: nearly everything earns its keep (uniform realizes it as a vectorized supercore
op). The only outright **drop** is `to_vector`/`from_vector`.

## Settled design decisions (2026-06-29)
- **Drop `size`/`data_size` from the uniform bv layer.** Precedent: the plain `UniformTuckerTensorTrain`
  already omits both (`data_size` is ambiguous under padding + mixed bool/float; `size = prod(shape)` is
  well-defined but the plain layer still omits it). If the uniform tangent (3b) needs a footprint metric,
  define it there as padded-supercore entry count.
- **Drop `to_vector`/`from_vector` from the uniform layer.** Their purpose is interop with external
  flat-vector optimizers (scipy etc.), which cannot exploit the uniform structure; that user is better
  served by the flat layer.
- **Checkers return per-stack-element results.** The determinantal-variety premise is that ranks vary per
  stack element, so "is element `[i,j]` orthogonal / minimal" is the fact, not a collapsed bool. Rule:
  - residual properties reduce **only over non-stack axes** → shape `stack_shape` (scalar when unstacked);
  - predicates (`is_orthogonal`, `has_minimal_ranks`, `is_consistent`, `allclose`) → bool arrays of shape
    `stack_shape`;
  - safety preconditions that must branch add `.all()` **at the call site** (the checker stays per-element).
  This requires **revisiting the verified ragged checkers AND `TuckerTensorTrain`** for the same
  semantics — a wide-blast-radius API change (doctests print `True`; safety does `bool(...)`). Done **LAST**
  (slice 2c-G), full-suite gated, with the exact ragged return-type change pinned down first. New uniform
  checkers are per-element from the start.

## Converter inventory + homes
**Placement rule (Nick's):** a converter **tied to one class** lives with it — **method** if `to_`,
**staticmethod** if `from_`; a converter **combining two peer classes** is a **standalone function**.
Placement follows logical dependency: `TuckerTensorTrain` stands alone and does not know
`UniformTuckerTensorTrain`; the uniform classes import and know the ragged ones, so **all ragged↔uniform
converters live in the uniform modules**. The backend `.data`-level functions (`ut3_conversions`,
`ubv_conversions`) are unaffected — this governs only the **frontend** wrapper.

### Within-layer (frame/variations/tangent ↔ tensor, one representation)
| Operation | Ragged | Uniform | Home |
|---|---|---|---|
| Orthogonal rep: tensor → (frame, variations) | `t3_orthogonal_representations` | `ut3_orthogonal_representations` ✅ | **standalone** (produces two peers) |
| Frame → its base point | `T3Basis.to_t3` | `UT3Basis.to_ut3` | **method** |
| Tensor → its frame (convenience = orth-rep[0]) | `T3Basis.from_t3` | `UT3Basis.from_ut3` | **staticmethod** |
| Single-variation substitution → tensor | `bv_to_t3` | ~~`ubv_to_ut3`~~ **DROPPED** (see Refinements round 2) | — |
| Whole tangent → tensor (doubled-rank efficient formula) | `T3Tangent.to_t3` | `UT3Tangent.to_ut3` (3b) | **method** |
| → dense | `*.to_dense` | `*.to_dense` | **method** |

~~`bv_to_ut3` (genuinely uniform): …~~ **DROPPED** — see "Refinements (round 2)" below. (The substituted
left/right subchains are differently-shaped supercores glued by the variation; there is no clean single
uniform supercore op, which is why the old `if False` stub went via ragged. Low importance; dropped for
now, maybe permanently.)

### Cross-layer (ragged ↔ uniform) — ALL migrated to methods/staticmethods on the uniform class
These were always intended to be methods; the existing module-func forms are retired here.
| Pair | uniform→ragged (method) | ragged→uniform (staticmethod) | Retires |
|---|---|---|---|
| tensor | `UniformTuckerTensorTrain.to_t3` | `.from_t3` | `ut3_to_t3` / `t3_to_ut3` |
| frame | `UT3Basis.to_t3basis` | `.from_t3basis` | `ut3basis_to_t3basis` (module func) |
| variations | `UT3Variations.to_t3variations` | `.from_t3variations` | — (new) |
| tangent (3b) | `UT3Tangent.to_t3tangent` | `.from_t3tangent` | — (new) |

## Method disposition (the `UT3Basis`/`UT3Variations` buildout)
| Ragged method(s) | Verdict | Uniform-native realization |
|---|---|---|
| `unstack` / `stack` | port | tree machinery over supercores+masks (mirror `ut3_operations`); dynamic-leaf-template |
| `to_ut3` / `to_dense` / `from_ut3` | port | direct supercore restructure; `to_dense = to_ut3().to_dense()` |
| `to_jax`/`to_numpy`/`copy`/`contains_jax`/`__repr__` | port | supercores convert; masks stay `np` |
| `size` / `data_size` | **drop** | (see decisions) |
| `__add__`/`__sub__`/`__mul__`/`__neg__`/`sum_stack`/`allclose` (Variations) | port | `corewise_*` on supercore arrays + carry masks; vectorized |
| `zeros`/`randn`/`unit`/`zeros_like`/`randn_like` (Variations) | port | build padded supercores + masks directly |
| `reverse` | port | `[::-1].swapaxes` + L/R swap (mirror `ut3_reverse`) |
| `orthogonalize` / `random_orthogonal(_like)` | port | via `ut3_orthogonal_representations` — uniform-native |
| checkers (`is_orthogonal`/`orthogonality_residual`, minimal-rank family, `is_consistent`, `allclose`) | port | masked-supercore einsum vs the **mask-prefix identity**, reduced over non-stack axes (per-element) |
| `to_vector` / `from_vector` | **drop** | external flat-optimizer interop → use the flat layer |
| `save` / `load` | port | mirror plain-UT3 3-family (supercores + shape + masks) |

## The slices (done one at a time)
- **2c-A — Cross-layer converters → methods (B1 tensor, B2 frame, B3 variations).** Add the missing
  backend twins (`t3basis_to_ut3basis`, `ut3variations_to_t3variations`, `t3variations_to_ut3variations`;
  `ut3basis_to_t3basis` exists). Move the frontend wrappers to method/staticmethod form on the uniform
  classes; **retire** `t3_to_ut3`/`ut3_to_t3`/`ut3basis_to_t3basis` module funcs and update all call sites +
  tests/doctests. Pad policy mirrors `t3_to_ut3` (max-over-modes, optional overrides). *First — the glue used
  to test the rest; touches plain-layer consumers (grep), so full-suite gate.*
- **2c-B — Within-layer frame → tensor + dtype utils.** `UT3Basis.to_ut3`/`to_dense`/`from_ut3`, and the
  dtype/structural utilities (`to_jax`/`to_numpy`/`copy`/`contains_jax`/`__repr__`) on both classes.
  (`ubv_to_ut3` dropped — see Refinements round 2.)
- **2c-C — `unstack`/`stack`.** New backend `ubv_operations` mirroring `ut3_operations` (dynamic-leaf
  template + first-leaf drill); frontend wrappers on both classes.
- **2c-D — `UT3Variations` vector space + constructors.** `__add__`/`__sub__`/`__mul__`/`__rmul__`/`__neg__`/
  `sum_stack`/`allclose`; `zeros`/`randn`/`unit`/`zeros_like`/`randn_like` — all direct on supercores.
  **Masking semantics (decided 2026-06-29; user doc: the "tangent vector-space ops" section of
  `docs/uniform_masks_vs_ranks.md`):** the variation algebra is a *fixed-rank vector space* (corewise),
  NOT the tensor ⊕/⊗ (concat/Kronecker) — so add/sub keep the mask (require an explicit **same-mask + same-
  shape structural precondition**, since uniform padding hides the mismatch ragged would catch as a shape
  error); scalar mul/neg leave the mask; `sum_stack` ORs the mask over the summed stack axes (no-op for a
  same-mask `K`-stack). Masks never change → no aux_data churn / recompile / `check_ubv_pair` breakage.
  Constructors fill supercores completely with optional all-True-default masks; `*_like(x)` takes
  shape+masks+stack from `x` (so `zeros_like(basis)` is the zero tangent carrying the base's gauge masks).
- **2c-E — `reverse` / `orthogonalize` / `random_orthogonal(_like)`** — direct uniform (`reverse` needs a
  backend `ubv_reverse`; `orthogonalize`/`random_orthogonal` reuse `ut3_orthogonal_representations`).
- **2c-F — `save`/`load`** (both classes). *(Independent; may move earlier if convenient.)*
- **2c-G — Checkers + per-element-semantics revisit** (ragged `T3Basis`/`T3Variations` + `TuckerTensorTrain`
  + plain `UniformTuckerTensorTrain`). **Last**, full-suite gated, exact ragged return-type change pinned
  down first (see decisions).

_Dependencies: 2c-A first (glue). `orthogonalize` (2c-E) needs `to_ut3` (2c-B). 2c-G last. The rest are
largely independent._

_After 2c → **3b** (UT3Tangent + uniform_manifold): the tangent converters (A5 `UT3Tangent.to_ut3`,
B4 `from_t3tangent`/`to_t3tangent`), geometry, derivative probing._

## Refinements (2026-06-29, round 2) — stacking/masking pre-mortem
A careful pass over the two historically-tricky areas (stacking, masking), grounded in the code, settled
these. They AMEND the slices above.

- **`ubv_to_ut3` DROPPED.** Substituting a variation yields a left subchain supercore + a differently-shaped
  right subchain supercore glued by the variation — not a clean single uniform supercore op (the reason the
  old `if False` stub went via ragged). Low importance; dropped for now, maybe permanently. (Correct name
  would have been `ubv_to_ut3`, moot.) 2c-B loses it.

- **Stacking is mirrored from the ragged layer EXACTLY; fix every stacking issue as encountered, never
  defer.** The ragged model (verified, full of subtle pitfalls we already navigated):
  - `T3Basis` carries **one** stack `C` (base points), on every core (+ masks, in uniform).
  - `T3Variations` carries **one** stack and is **split-agnostic** — it does NOT know/store any `K`/`C`
    partition; `stack_shape` is just its whole leading stack (supercores AND masks ride it together).
  - `check_bv_pair` only ties them: `base.stack_shape` (`C`) must be the **inner/trailing suffix** of
    `variations.stack_shape`, plus the holes match. No split is computed here.
  - `T3Tangent` **infers** the split, never stores it: `C = basis.stack_shape`, `K = variations.stack_shape`
    minus that suffix, full `= K + C` (base-inner: variation cores stack `K + C + (core,)`).
  - **Resolved my earlier Model-A/B "where do the variation masks live" fork: there is no fork.** The
    variation carries one stack including its masks (split-agnostic); nothing is "constant over K" from the
    variation's view. The uniform layer mirrors this verbatim.

- **`check_ubv_pair` `K`-stacking BUG → fix in 2c (not 3b).** Today it demands `base.uniform_structure ==
  variations.uniform_structure`, which includes `stack_shape`, so it *forbids* any tangent-stack excess.
  Fix to mirror ragged `check_bv_pair`: (1) the stack-free structure `(d,N,nU,nD,rL,rR)` matches; (2) `C` is
  the inner suffix of the variation stack; (3) rank-structure (mask) match — since masks carry the stack,
  the base masks (over `C`) equal the variation masks (over `K+C`) **broadcast over the excess `K`**, with
  the gauge shifts (`basis_left[:-1]`, `basis_right[1:]`). This is the uniform analog of ragged's exact
  `variation_shapes` match. **Add `K≠()` tests wherever the type permits** (`UT3Variations` ops,
  `check_ubv_pair`); frames are `C`-only so no `K` there.

- **Checkers (2c-G) — orthogonality principle CLARIFIED.** Each masked supercore slice IS a hypothetical
  ragged core (slice by mask). Orthogonality = whether *that* ragged core is orthonormal in its sense — e.g.
  left TT core `tt_sc[i]` masked to `(rLi, nUi, rL(i+1))`, require its `(rLi·nUi, rL(i+1))` left-unfolding
  has orthonormal columns. This is exactly `ut3_orthogonality_residual`'s masked-Gram-vs-`diag(outgoing_mask)`
  (the per-element equivalence oracle). Work = extend to the 4 basis senses (U/O/L/R), per-element reduction
  (keep `stack_shape`), correct *outgoing* mask index per sense (left core `i` → `basis_left_mask[i+1]`).
  **2c-G goes ragged-FIRST**: build/fix the per-element semantics on `TuckerTensorTrain`/`T3Basis`/
  `T3Variations` (they become the oracle), then the uniform checkers test against them.

- **Constructors (2c-D) — fill-complete + optional masks.** `zeros`/`randn`/`unit` always fill the supercores
  completely; the user must supply the (padded) shape; masks are **optional** and default to all-True arrays
  of the appropriate shapes (full rank). `*_like(x)` derives shape + masks + stack from an object. (`unit`'s
  index must land in a real slot — trivially true under the all-True default.)

- **`reverse` (2c-E) — oracle.** Left/right swap under reversal; correctness lens is that `reverse` commutes
  with conversion: `to_t3basis(B.reverse()) == to_t3basis(B).reverse()` (and at the tangent level once 3b
  lands, `x.reverse().to_t3tangent() == x.to_t3tangent().reverse()`).

- **Oracle gaps acknowledged:** per-element checkers need the ragged per-element version first (hence 2c-G
  ragged-first); `K`-bugs are only caught by `K≠()` tests (hence add them as we build, per the
  fix-stacking-now rule).

---

# Increment 2c-G — per-element checker semantics (detailed plan, 2026-06-29)

_The last 2c slice, and the only one that edits the **verified** layer. Designed with Nick; PENDING final
review. Done **ragged-first** (the ragged per-element checkers are the oracle for the uniform ones),
full-suite gated after each sub-slice._

## Design rules (agreed 2026-06-29)
- **Residuals + bool predicates return shape `stack_shape`** (scalar when unstacked): reduce over the
  **non-stack axes only** (today they `max` over *everything*, incl. the stack, → one scalar).
- **Safety preconditions reduce with `.all()` at the call site** (Option A): `safety.require` stays a
  scalar gate; each precondition becomes `safety.require(checker(...).all(), msg)` ("require **all** stack
  elements pass"). The `and` of two checks becomes `chk1.all() and chk2.all()`. ~6 sites in `manifold.py`
  (`_require_orthogonal_frame`, `inner`, `norm`). The checker itself never collapses — only the
  precondition's *consumption* adds `.all()`. (`is_orthogonal` is per-`C`; `is_gauged` is per-`K+C`; `.all()`
  reduces whatever shape, which is why call-site reduction is clean.)
- **Granularity is per representation (the "honest" rule), NOT a forced uniform contract — PENDING Nick:**
  - **Numerical** checkers (orthogonality / consistency / gauge / `allclose`) vary per stack element →
    **per-element in BOTH ragged and uniform.**
  - **Structural** `has_minimal_ranks`: ragged ranks are **shared across the stack** (one core shape) → it
    is genuinely one answer → **ragged stays SCALAR (unchanged)**; uniform ranks vary per element →
    **uniform is per-element.** _(Alternative considered: broadcast ragged structural to `stack_shape` for a
    single contract; rejected as churn-for-its-own-sake on verified code that has no per-element variation.
    Flag for Nick.)_
  - `has_numerically_minimal_ranks` combines a numerical (per-element) and a structural (ragged-scalar)
    check → **per-element** (`is_orthogonal(atol) & has_minimal_ranks`; numpy broadcasts the scalar).
- **Not bool checkers → unchanged in 2c-G:** `minimal_ranks` (returns the *ranks*, already arrays in
  uniform / tuples in ragged), `tangent_space_dimension` / `manifold_dim` (return an int). The uniform
  per-element *dimension* (varying ranks → varying dim) is deferred to 3b if needed.
- **Doctests (per checker family):** (a) print the return **shape**; (b) the `.all()` summary; (c) a
  **mixed stack** `stack_shape=(2,)` with element 0 orthogonal + element 1 deliberately perturbed, printed
  **whole** (`[ True False]`) — the doctest that *teaches* the per-element semantics, built by `stack`-ing
  a good frame + a perturbed one; (d) **scalar-vs-array** (same checker unstacked → scalar, stacked →
  array). One "structure" doctest per checker family, not per method.

## The checker inventory (current → new)
| Class | checker | now | 2c-G |
|---|---|---|---|
| **T3Basis** | `orthogonality_residual` | scalar float | `(C,)` array |
| | `is_orthogonal`, `is_consistent`, `allclose` | bool | `(C,)` bool array |
| | `has_minimal_ranks` | bool | **scalar (unchanged)** |
| | `has_numerically_minimal_ranks` | bool | `(C,)` bool array |
| **T3Variations** | `allclose` | bool | `(K+C,)` bool array |
| **T3Tangent** | `is_orthogonal` (→basis), `allclose` | bool | per-element |
| | `gauge_residual` | scalar float | `(K+C,)` array |
| | `is_gauged` | bool | `(K+C,)` bool array |
| | `has_minimal_ranks` (→basis) | bool | scalar (unchanged) |
| | `has_numerically_minimal_ranks` | bool | per-element |
| **TuckerTensorTrain** | `is_left_orthogonal`, `is_right_orthogonal` | bool | `stack_shape` bool array |
| | `has_minimal_ranks` | bool | scalar (unchanged) |
| | `has_numerically_minimal_ranks` | bool | per-element |
| **UniformTuckerTensorTrain** | `is_left_orthogonal`, `is_right_orthogonal` | bool | `stack_shape` array |
| | `has_minimal_ranks` | bool (np.all) | **per-element** (ranks vary) |
| **UT3Basis** (new, G3) | `orthogonality_residual` (masked-Gram, 4 senses), `is_orthogonal`, `is_consistent`, `allclose`, `has_numerically_minimal_ranks` | — | per-element |
| | `minimal_ranks`, `has_minimal_ranks` | — | per-element (structural, varies) |
| **UT3Variations** (G3) | `allclose` (2c-D shipped it collapsed) | scalar | per-element |

## Backend residual changes (reduce over non-stack axes → keep `stack_shape`)
- `orthogonal_representations.basis_orthogonality_residual`, `.basis_consistency_residual`
- `tangent_operations.gauge_residual`
- `t3_orthogonalization.t3_orthogonality_residual` (TuckerTensorTrain)
- `ut3_orthogonalization.ut3_orthogonality_residual` (plain UT3 — already masked; just change the final
  `max` to keep `stack_shape`)
Each infers its stack rank from the cores and `max`-reduces only the per-core/non-stack axes.

## The uniform-native checkers (G3)
`UT3Basis` orthogonality: each masked supercore slice IS a hypothetical ragged core; require *its* Gram ==
`diag(outgoing_mask)`, per stack element — the masked-Gram pattern of `ut3_orthogonality_residual`,
extended to the **four** basis senses (up `U`, down/outer `O`, left `L`, right `R`), with the correct
*outgoing* mask per sense (left core `i` → `basis_left_mask[i+1]`). Oracle: the ragged per-element
`is_orthogonal` via `to_t3basis` (unstack + per-element).

## Sub-slices
- **2c-G1 — ragged numerical residual checkers + safety `.all()`.** THE risky one (verified code + safety).
  Backend residuals → `stack_shape`; T3Basis/T3Variations/T3Tangent/TuckerTensorTrain numerical predicates
  → per-element; `manifold.py` precondition `.all()` edits; the per-element doctests. **Grep ALL consumers
  of the residuals/predicates first** (wide blast radius). Full-suite gated.
- **2c-G2 — `has_numerically_minimal_ranks` → per-element** (the numerical×structural combiner) across the
  classes; ragged `has_minimal_ranks` left scalar (pending the granularity decision). Lower risk (no safety
  dependency — minimal rank is not a precondition, per `numerical_contract_catalog.md`).
- **2c-G3 — uniform-native checkers** (UT3Basis/UT3Variations/UniformTTT), per-element from the start,
  tested against the ragged oracle. New code (no verified-code risk).

## Open decision for Nick
**Ragged `has_minimal_ranks` (structural): stay scalar (honest — ragged ranks are shared) or broadcast to
`stack_shape` (one contract across all checkers)?** Recommendation: **stay scalar** (less verified-code
churn, honest granularity; numerical checkers carry the per-element story).

---

# Increment 3b — uniform tangent + manifold layer (detailed plan, 2026-06-29)

_The final stretch of the 1.0 centerpiece: rebuild the uniform tangent + manifold layer off the new
`UT3Basis`/`UT3Variations` (2c is done), mirroring ragged `manifold.py` + `backend/tangent_operations.py`.
Designed with Nick 2026-06-29; PENDING his review before we start. The current `uniform_manifold.py` is a
graveyard (imports `OLD_uniform`, dead `if False:` blocks, doesn't import) — a full rebuild._

## The three layers we mirror
1. **`UT3Tangent`** (mirror `T3Tangent`) — a frozen dataclass bundling `(UT3Basis, UT3Variations)`; the
   `K`/`C` split is **inferred** from the pair, never stored (the split-agnostic stacking of 2c).
2. **`UniformManifoldGeometry` / `UniformCorewiseGeometry`** (mirror `ManifoldGeometry` / `CorewiseGeometry`)
   — stateless bundles of `base`/`randn`/`project`/`inner`/`norm`/`retract` (+ `project_ambient`/`transport`
   on the manifold), with the per-element safe-mode preconditions (`.all()`, from 2c-G).
3. **backend uniform tangent ops** (mirror `tangent_operations.py`) — the algorithms. **The real work.**

## Guiding principle (reaffirmed for 3b)
- Mirror where uniform earns it; **do uniform things directly** (vectorized supercore ops), not via
  ragged-then-convert. Drop only `size`/`data_size` and `to_vector`/`from_vector` (already dropped in 2c).
- **Scan vs map** (the backend reuse rule):
  - **Scan-based** ops (TT sweeps: `tt_zipper_*`, the `compute_sigmas`-style edge-variable carries) →
    **reuse the ragged code polymorphically** by dispatching on `is_uniform` + **mask-once** the supercores
    first (padded bonds contract to zero; mask-free inside the sweep). Exception: if a sweep must *update*
    masks mid-stream, don't reuse — but the manifold sweeps contract the fixed orthonormal frame, so
    mask-once-before suffices.
  - **Map-based** ops (gauge projection, the doubled-rank build, `assemble_*`) → a **vectorized einsum over
    the leading `d` axis**, NOT an `xmap` loop and NOT a bare `d...` einsum: for the tangent maps it must
    treat `W`/`K`/`C` as **separate blocks** (`d`-prefixed grouped-block contractions).
- **The `W`/`K`/`C` multi-block stacking is the hard, untrusted part** — neither the OLD code nor the
  current uniform-tangent probing handles the full combination. **Build it ourselves and test every combo**
  (`W` / `K` / `W+K` / `+C`) against the ragged grouped-block oracle. Do NOT trust OLD stacking or assume a
  single `d...` prefix is enough.

## Triage — case-by-case

### Layer 1 — `UT3Tangent` (mirror `T3Tangent`)
Mirror nearly everything, delegating to `UT3Basis`/`UT3Variations`:
| group | verdict |
|---|---|
| structure props (`d`/`shape`/`base_stack_shape`/`tangent_stack_shape`/`stack_shape`/`structure`/`data`) | **mirror** — infer `K`/`C` from the pair |
| `to_jax`/`to_numpy`/`copy`/`contains_jax`/`__repr__` | **mirror** (delegate) |
| `size` / `data_size`, `to_vector` / `from_vector` | **drop** (2c decisions) |
| vector-space (`__add__`/`__sub__`/`__mul__`/`__neg__`), `_check_same_tangent_space`, `sum_tangents`, `normalized`, `zeros`/`unit`/`zeros_like` | **mirror** (delegate to `UT3Variations` + a uniform `safety.frames_equal`) |
| `corewise_inner`/`corewise_norm`, `allclose` | **mirror, uniform-direct** — supercores are `d`-leading, so reduce over the non-stack axes (like `UT3Variations.allclose`), NOT `corewise_stack_*` |
| `is_orthogonal`/`minimal_ranks`/`has_minimal_ranks`/`has_numerically_minimal_ranks` | **mirror** (delegate to `UT3Basis`; per-element from 2c-G) |
| `tangent_space_dimension` | **mirror, per-element** — `int` unstacked, `array(int)` shape `stack_shape` stacked (`manifold_dim` made rank-array-aware) |
| `gauge_residual`/`is_gauged` | **mirror** — needs the uniform `gauge_residual` backend |
| `reverse`, `unstack_*`/`stack_*` | **mirror** (the commute-with-conversion lens; reuse `ubv_operations`) |
| `to_dense`/`to_ut3` (doubled-rank), cross-layer `to_t3tangent`/`from_t3tangent` | **mirror** (see backend) |
| `probe`/`apply`/`entries` + derivatives + transposes | **mirror by wiring** — but see the probing-tangent gap below (NOT free) |
| `save`/`load` | **mirror** (basis+variations families, like 2c-F) |

**Prerequisite:** `check_ubv_pair` still **forbids `K`** (confirmed — it compares `uniform_structure`, which
includes `stack_shape`). Must become the ragged-style suffix check + the mask broadcast-over-`K`.

### Layer 2 — the geometries
**Mirror all.** `MANIFOLD`: `base`, `randn`, `random_orthogonal`, `randn_like`, `project`/`project_oblique`,
`inner`/`norm` (+ per-element `.all()` preconditions), `retract`, `project_ambient`, `transport`. `COREWISE`:
`base` (the `(U,G,G,G)` frame), `randn`, `project` (identity), `inner`/`norm`, `retract` (additive).
`project_ambient(basis, dense)` does **not** get an efficient native uniform path (see below).

### Layer 3 — backend algorithms (the meaty layer) + reuse map
| # | op | structure | uniform approach | source |
|---|---|---|---|---|
| (a) | `tangent_to_ut3` (doubled-rank) — **keystone** | per-core block-bidiagonal (`[U;V]` tucker concat; 2×2 zero-padded TT blocks; first/mid/last differ) | vectorize the mid-block over `d`; **masks double (concat)** | **port OLD `uniform_tangent_to_uniform_t3` + the mask-concat rule** |
| (b) | `tangent_to_dense` | sum of `2d` `bv_to_t3` terms (ragged) | **= `to_ut3()(doubled).to_dense()`** (no `bv_to_ut3`) | trivial from (a) |
| (c) | `retract` | `tangent_to_t3(shift)` → `t3svd` truncate to base ranks | `tangent_to_ut3(shift)` → `ut3.t3svd` truncate to the **base masks**, prefix-slice | OLD skeleton (sub `ut3.t3svd` for vapor `ut3_svd_masked`) |
| (d) | `orthogonal_gauge_projection` | per-core **map** of `'...'` einsums | vectorized `d`-axis einsum + `[:-1]` boundary + mask-once | **port OLD `ut3_orthogonal_gauge_projection_for_loop`** (identical einsums) |
| (e) | `oblique_gauge_projection` | **sequential** (compensate through down/right) | sequential; vectorize the independent parts | ragged template (no OLD) |
| (g) | `project_t3_onto_tangent_space` | TT **zippers** (scan) + per-core map + gauge | reuse zippers (mask-once + polymorphic) + `d`-axis map + uniform gauge | ragged template (no OLD) |
| (h) | `gauge_residual` | per-core gauge inner products, max | vectorized `d`-axis einsum + masking (per-element, like G1) | ragged template (no OLD) |
| (f) | `project_dense_onto_tangent_space` | left sweep + per-slot reductions | **DROP the native uniform path** (perf layer; users shouldn't work dense here). Provide the cross-layer hooks (`to_t3basis` + `from_t3tangent`) so dense ground-truth checks go via ragged. | — |
| — | `tt_zipper_*`, stack/unstack tangent helpers | scans / tree ops | make polymorphic (`is_uniform` + mask-once) / reuse `ubv_operations` | reuse |

## OLD-code mining findings (folded in)
- **OLD masks are boolean prefix masks** (not float edge-weights) → the supercore einsums port; only the
  **mask packaging** differs (OLD flat 5-tuple → the new `UT3*Masks` holders).
- **Fully implemented + portable:** (a) doubled-rank `tangent→ut3` (`uniform_manifold.py:29`) and (d)
  orthogonal gauge (`:565`, the for-loop — identical einsums to ragged).
- **The keystone artifact — the doubled-rank mask-concat rule:** `tucker_mask = concat(up, down)`;
  `tt_mask = concat(left_ext, right_ext)` with `left_ext = [left, ones]`, `right_ext = [ones, right]`
  (boundary edges padded full). The doubled TT bond is `rL+rR`. **This is the single most valuable thing the
  OLD code gives us** — the ragged code never had to think about it.
- **(c) retract skeleton** (`:272`) is correct (doubled-shift → truncate → prefix slice) but calls vapor
  helpers. **Vapor to ignore:** `ut3_svd_masked`, `ut3_orthogonal_gauge_projection_using_map`,
  `pack_uniform_tensor_train`.
- **(e), (g), (h)** have no OLD uniform code → from ragged templates. **(f)** OLD skeleton is debug-only.
- **OLD stacking is not trusted** (stack-naive concats) — use OLD for the *unstacked* algorithm + the
  mask-concat rule; **derive + test the `K`/`C`/`W` stacking ourselves.**

## The probing-tangent gap (diagnosed)
`probing.py` is uniform-polymorphic for the **plain `UniformTTT`** (real `d`-prefixed `WC` contractions,
`dCio_dWo_to_dWCi` …), but the **uniform *tangent* path is incomplete**:
- **scan-style** tangent fns are fine: `compute_sigmas` runs `xscan` over `d`, and its step `_sigma_step`
  uses the proper `WKC` grouped-block contractions.
- **map-style** tangent fns are **broken for `C≠()` AND `K≠()`** (see "Validation hardening" — NOT just a
  base `C` stack, and **NOT `compute_dxis`**, which is fine): the broken ones are `compute_detas`,
  `assemble_tangent_zs`, `compute_dxi_tildes`, `assemble_tucker_variations`, `assemble_tt_variations`. They
  fall back to raw `d...` einsums (e.g. `einsum('d...i,diaj->d...aj', sigmas, right_tt_cores)` —
  `right_tt_cores` is `diaj`, **no `...`**, so it assumes `C=()`, and the single `...` conflates `W` with
  `K`). `compute_deta_tildes` is broken *differently* (a wrong grouped contraction → needs a shared-`C`
  `dWCo_dCio_to_dWCi`). There are **no `d`-prefixed uniform `WKC` contractions** in `contractions.py`.
- It is **untested** (no `UT3Tangent` exists yet).
**Fix (a real sub-slice):** build the `d`-prefixed uniform `WKC` grouped-block contractions in
`contractions.py` (mirror the ragged `WKC` family, leading-`d` batch) and rewrite the map-style uniform
tangent branches to use them; test all stacking combos against the ragged `WKC` oracle.

## Other design points
- **jit/recompile constraint** (`docs/uniform_backend_jit_recipe.md`): the geometry/optimizer must hold
  masks loop-invariant and trace only supercores; at fixed rank the doubled→truncate masks are
  deterministic, so the kernel compiles once. Shapes the `retract`/geometry design.
- **Masking through conversions:** unlike the 2c ops, `to_ut3` is **doubled-rank** — masks change (double
  via concat), then `retract` truncates back. The boundary `ones`-padding in the mask-concat is the
  error-prone spot.
- **`K` everywhere:** variations carry `K+C`; einsums ride it (or the `d`-prefixed grouped-block
  contractions handle it); add `K≠()` tests throughout (fix-stacking-now).

## Sub-slices (dependency-ordered)

> **STATUS (2026-07-01): increment 3b is DONE — 3b-0 … 3b-5 + 3b-6 (6a–6d) + 3b-6′ (6′a–6′d) + 3b-7.** The
> uniform tangent *backend*, the *two geometries* (`UNIFORM_MANIFOLD`/`UNIFORM_COREWISE`), *tangent + corewise
> probing* (`𝒥`/`𝒥ᵀ`), and the *derivative (jet) probing* (`probe_derivatives`: forward + transpose + corewise,
> both the plain `UniformTuckerTensorTrain` and `UT3Tangent` layers) are all per-element verified vs ragged +
> adjoint-identity + mask-strict + garbage-robust + jit-clean. **3b-7 sweep+cleanup** done too: the
> sampling/derivative surface is doctested to the reference-module standard; `OLD_uniform*.py` deleted
> (signed off); `Sequence`→`Union` hint relaxation deferred to R2. **Next: optimizers/fitting on the uniform
> layer.** Live status: `dev/HANDOFF.md`.
>
> **3b-6′ refinements vs the plan below (recorded for the record):** (1) only the *map-style* jet fns needed
> `d`-prefixed contractions; the *scan-style* ones (`_apply_derivatives_*_from_*`, `_adj_sweep`,
> `compute_sigma_hat_jets`) needed only a dispatch-flag flip (the `xscan` strips `d`, the ragged `trs_*` runs
> per-slice). (2) `_init_jet` needed NO change (the scan carry has no `d` axis). (3) Beyond `build_input_jets`,
> the **`reverse_tt` in the four reversers** (`compute_nu_jets`/`compute_tau_jets`/`compute_sigma_tilde_jets`/
> `compute_sigma_hat_jets`) was a second unroll-trap class — the plan mislabeled these as "already
> uniform-aware"; fixed via `uniform_ops.reverse_utt`. (4) Chose **Approach B** (d-prefixed vectorized einsums,
> matching the plain-probing reference module) over xmap-over-`d`, so the whole probing/derivatives surface is
> consistent + fully vectorized over `d`; the forward-jet map fns (`compute_eta_jets`/`assemble_z_jets`/
> `compute_deta_jets`/`assemble_tangent_z_jets`) were converted to `d`-prefixed too.

- **3b-0** — `check_ubv_pair` `K`-fix (prerequisite; small): suffix check on `C` + mask broadcast-over-`K`.
- **3b-1** — `UT3Tangent` skeleton: bundle + `K`/`C` inference, vector-space ops, `stack`/`unstack`,
  constructors, dtype/copy/repr, `reverse`. No backend math yet.
- **3b-2** — **keystone**: backend `tangent_to_ut3` (doubled-rank, port (a) + mask-concat) +
  `UT3Tangent.to_ut3`/`to_dense` (b) + `retract` (c). Verify per-element vs the ragged doubled-rank `to_t3`
  and `retract` (equivalence contract).
- **3b-3** — gauge: `orthogonal_gauge_projection` (port (d)) + `oblique_gauge_projection` (e) +
  `gauge_residual`/`is_gauged` (h).
- **3b-4** — `project_t3_onto_tangent_space` (g) only (drop (f); reuse zippers via mask-once). Cross-layer
  `to_t3tangent`/`from_t3tangent`.
- **3b-5** — the two geometries (`UniformManifoldGeometry`/`UniformCorewiseGeometry`), wired with the jit
  constraint + per-element preconditions.
- **3b-6** — the `d`-prefixed uniform `WKC` grouped-block contractions + fix the map-style uniform tangent
  probing + wire `UT3Tangent.probe`/`apply`/`entries` (+ derivatives). Plus the old Slice-4 ragged-poly gaps
  (`_apply_transpose_adjoint`, the `Sequence` signatures).
- **3b-7** — tests/doctests sweep (equivalence contract, `K≠()`); **delete `OLD_uniform*` + the `if False:`
  graveyard** once functionality is confirmed preserved.

## Testing strategy
Per slice: the **equivalence contract** `to_ragged(uniform_op(to_uniform(x))) == ragged_op(x)` per stack
element (the tangent oracle is `to_t3tangent` / the doubled-rank `to_t3`). **Include `K≠()`** (tangent
stack) tests everywhere the type permits. For the `WKC` contractions: test every `W`/`K`/`W+K`/`+C` combo
against the ragged grouped-block result. jax dispatch in `test_dispatch` (masks stay host; jit the kernel).

## Validation hardening (2026-06-29) — pins from the cold-resume validation pass

_Three fresh agents read the 3b plan with NO prior context and reproduced the intended decisions; these
are the underspecifications + one error they surfaced. **These pins SUPERSEDE the prose above where they
conflict.** Read this subsection before starting any 3b slice._

### Corrections to the prose above
- **`compute_dxis` is NOT broken** (error above): it dispatches to the proper `d`-prefixed
  `dCio_dWo_to_dWCi` and handles `C` and `K`. The broken map-style uniform tangent functions are exactly:
  `compute_detas`, `assemble_tangent_zs`, `compute_dxi_tildes`, `assemble_tucker_variations`,
  `assemble_tt_variations`, and `compute_deta_tildes`.
- **The probing diagnosis is `C≠()` AND `K≠()`**, not just "a base `C` stack": the raw-`d...` einsums give
  the cores no stack-subscript home and the single `...` conflates `W` with `K`.
- **`compute_deta_tildes` is a *wrong-contraction* case, not a raw-`d...` one**: it calls the outer-product
  `dCio_dWo_to_dWCi` where it needs a **shared-`C`** twin. So 3b-6 must also add a 2-block
  **`dWCo_dCio_to_dWCi`** (mirror ragged `WCo_Cio_to_WCi`) — "mirror the `WKC` family" does not cover it.

### Keystone (3b-2) — the load-bearing pins
- **Signature:** `tangent_to_ut3(basis_data, variations_data, include_shift=False)` (model on ragged
  `tangent_operations.tangent_to_t3`). Doubled masks from the variation's length-`d`
  `variations_left/right_mask` + the basis/variation `up`/`down` masks.
- **The doubled-rank mask rule (EXACT — the #1 correctness trap):**
  - `tucker_mask = concat([up_mask, down_mask], axis=-1)` -> `(d,)+stack+(nU+nD,)`.
  - `left_ext  = concat([variations_left_mask,  ones((1,)+stack+(rL,))], axis=0)` -> `(d+1,)+stack+(rL,)`.
  - `right_ext = concat([ones((1,)+stack+(rR,)), variations_right_mask],  axis=0)` -> `(d+1,)+stack+(rR,)`.
  - `tt_mask = concat([left_ext, right_ext], axis=-1)` -> `(d+1,)+stack+(rL+rR,)`.
  - **TRAP:** the appended boundary slot is **`ones` (FULL)**, NOT `basis_left_mask[d]`/`basis_right_mask[0]`
    (those hold the real boundary rank ~1 -> WRONG). The interior source is the **length-`d`
    `variations_*` masks** (= `basis_*_mask[:-1]` / `[1:]`).
- **Stacking discipline (do NOT trust OLD):** OLD uses positive `axis=1/2/3` + stack-free `Z` zeros —
  correct only for `stack=()`. Use **negative axes** (`-1/-2/-3`) and **broadcast each base supercore
  (stack `C`) up to `K+C`** (mirror ragged `tangent_to_t3`'s `bcast2`/`bcast3`); shape the `Z` zero-blocks
  on `K+C`. Test `C≠()`, `K≠()`, `K+C` (x `include_shift ∈ {False,True}`).
- **`include_shift`:** OLD and ragged fold the last-core shift into *different* zero-block arrangements;
  pick ONE convention and verify dense-equivalence in **both** modes (don't assume the two references agree
  structurally).
- **Oracle (until 3b-4):** `to_t3tangent`/`from_t3tangent` do NOT exist at 3b-2. Build the ragged oracle
  from the **2c converters**: `bvf.T3Tangent(B.to_t3basis(), V.to_t3variations()).to_t3(include_shift)`
  (then `.to_dense()`, or `tangent_operations.tangent_to_dense`). Cleanest comparison is **dense**
  (stack- and mask-aware), ~1e-13, per stack element. _(This corrects the "Testing strategy" line, which
  names `to_t3tangent` — that converter is a 3b-4 deliverable.)_

### Retract (3b-2 (c)) — the `K`-stack gap
- `retract` = `tangent_to_ut3(shift)` -> `ut3.t3svd(max_tucker_ranks, max_tt_ranks)` truncate ->
  prefix-slice supercores to base padded dims (`nU`, `rL`, `N`).
- **`K`-stack:** base ranks are `(d,)+C` / `(d+1,)+C` but the shifted ut3 is `K+C`, so they won't
  right-align — **reshape base ranks to `(d,)+(1,)*|K|+C`** before `t3svd`. (The plan's retract item was
  silent on `K`.)
- The retracted UT3 carries the **base masks** (truncation targets the base ranks).

### Backend reuse (scan vs map) — clarifications
- **Editing verified `tangent_operations.py` in place is SANCTIONED** for the scan reuse (make the
  `tt_zipper_*` + `reverse_tt` dispatch on `is_uniform`; mask-once the supercores first). Precedent:
  `orthogonal_representations`, `probing` are already inferred-`is_uniform`. Wide blast radius -> **grep
  all consumers + full-suite gate.**
- **Mask-once happens in the *calling* tangent function**, before the contraction/sweep; the `d`-prefixed
  contractions and the reused zippers stay **mask-agnostic** (padded bonds contract to zero because the
  operands were zeroed upfront).
- **"Map -> vectorized `d`-axis einsum" is a PERF rule, not correctness:** a correct-but-looped first cut
  is acceptable, then optimized. Get the equivalence-contract green first, optimize after.

### 3b-0 `check_ubv_pair` `K`-fix — the exact mask comparison
1. Compare the **stack-free** structure `(d,N,nU,nD,rL,rR)` (slice `uniform_structure[:6]`), not the full
   7-tuple (whose trailing `stack_shape` is what wrongly rejects `K`).
2. Require `base.stack_shape` to be the **trailing suffix** of `variations.stack_shape`
   (`var_stack[len(var_stack)-len(base_stack):] == base_stack`).
3. Rank-mask match **broadcast over the excess `K`** (keep the gauge shifts `basis_left_mask[:-1]` /
   `basis_right_mask[1:]`): `np.array_equal(np.broadcast_to(base_mask_reshaped_with_K_axes, var_mask.shape),
   var_mask)` — i.e. the variation mask is constant along `K` and equals the base mask. Add `K≠()` tests
   (the only doctest uses `stack_shape=()`).

### Slice gating + drops
- **Slice gating:** each slice is gated on the **prior slice's equivalence-contract tests being green**
  (a resumer must re-verify the keystone before building the gauge layer on it).
- **`project_dense` fallback ordering:** the go-via-ragged dense path needs **`from_t3tangent`**, which
  only lands in **3b-4** — so the dense-ground-truth round-trip is unavailable until then (`to_t3basis`
  already exists from 2c-A; the in-the-middle step is ragged `MANIFOLD.project_ambient(basis, dense)`).
- **`size`/`data_size` is a *conditional* drop** (define a padded-supercore-entry footprint in 3b if the
  tangent needs one), not a hard removal.

### `d`-prefixed contraction inventory to build (3b-6)
Mirror each ragged `WKC` with a leading shared `d` (strip `d`, derive block shapes, flatten each block
keeping `d`, `_grouped_einsum`). Per broken function:
- `compute_detas`: `dWKCa_dCaib_dWCb_to_dWKCi`, `dWCa_dKCaib_dWCb_to_dWKCi` (n_base), `dWCa_dCaib_dWKCb_to_dWKCi`.
- `assemble_tangent_zs`: `dWKCi_dCio_to_dWKCo`, `dWCi_dKCio_to_dWKCo` (n_base).
- `compute_deta_tildes`: `dWCo_dCio_to_dWCi` (shared-`C`, no `K`).
- `compute_dxi_tildes`: `dWKCa_dCaib_dWCb_to_dWKCi`, `dWCa_dCaib_dWKCb_to_dWKCi`.
- `assemble_tucker_variations`: `dWKCo_dWCa_to_{dWKCao,dKCao}` (n_probe), `dWo_dWKCa_to_{dWKCao,dKCao}`.
- `assemble_tt_variations`: `dWCi_dWCa_dWKCj` / `dWKCi_dWCa_dWCj` / `dWCi_dWKCa_dWCj` -> `d{WKCiaj,KCiaj}` (n_probe).
**Oracle:** per leading-`d` index, the `d`-prefixed contraction must equal the ragged `WKC` applied to
`operand[i]`; test all `W`/`K`/`W+K`/`+C` combos. (Ragged `WKC` family + the existing `d`-prefixed plain
2-block contractions: `backend/contractions.py` ~lines 315–461 (`dCio_dWo_to_dWCi`, …) and ~759–1357.)

## Varying ranks across the `C` (base) stack — decided 2026-06-30

A uniform tangent stack **supports varying ranks across the base (`C`) stack** (a rank sweep / batch of
models at different ranks). Uniform rank is required only *within* one tangent space — i.e. **across `K`
(the tangent stack), where it is automatic** (one shared base; `check_ubv_pair` enforces masks constant
along `K`). A `C`-stack is a tangent to the **product** of (possibly different) fixed-rank manifolds, so
its per-element tangent-space dimensions may differ. Full reasoning: the revised tangent section of
`docs/uniform_ranks_and_varieties.md`. (This overturns that doc's earlier over-restriction to uniform-`C`;
no committed code assumed uniform-`C` — 2c + 3b-1a are already per-element, verified by experiment:
a varying-rank `C`-stack tangent gives per-element dims matching the ragged models.)

**Consequences for the remaining slices:**
- **Do NOT add a uniform-`C` precondition.** Implement every backend op (gauge, retract, projection,
  later the optimizers/CG) **per-element-mask-aware** — vectorize over `C` with each element applying its
  own mask. The subtle spot is a batched Gauss-Newton/CG over a varying-rank `C`: the masked-out
  directions must sit in `ker(J)` so they add zero to the CG reductions and don't pollute the active
  per-element solve (holds iff the gauge projection + `J`/`Jᵀ` respect the masks).
- **Varying-`C` is a first-class equivalence-contract test case** from 3b-2 on: alongside `()` / `C` / `K`
  / `K+C`, include a **varying-rank `C` stack** (build it via the 2c heterogeneous `UT3Basis.stack`, as in
  `test_stack_heterogeneous_ranks`) and check each element against its own ragged model.
- If a specific op *empirically* needs uniform `C`, add a **narrow non-enforced precondition there**
  (orthogonal/gauged pattern), never a blanket restriction. "If we hit problems later, do the deep
  thinking then" (the agreed posture).

## Slice 3b-4c — test-hardening pass for the UT3 tangent layer (decided 2026-06-30)

**Why.** Every correctness test in `test_uniform_manifold.py` is a dense/numerical comparison on
**clean-padding** inputs (everything comes from `ut3_orthogonal_representations` / `from_t3`). That
combination is **blind to too-permissive masks** (phantom rank): on clean padding an over-claimed rank
just keeps zeros, so the dense is still right. This is the exact class of the phantom-boundary bug that
only the paper caught (the `ones`-vs-`zeros` doubled-mask boundary). The hardening pass closes this blind
spot plus the stacking / masking / boundary gaps.

**Two complementary tools** do most of the work:
- **Exact output-mask assertions** — catch phantom rank / wrong masks directly. Non-tautological: derive
  the expected mask from a *different* source than the impl (the base ranks + the paper rule), not from
  the impl's own construction.
- **Garbage-padded *inputs*** — catch ops that fail to mask-once and silently depend on clean padding;
  for `tangent_to_ut3` (which builds from raw supercores) the garbage flows into the output's
  input-derived padding, so it also exposes a too-permissive mask there.

> Tempting-but-wrong: corrupting an *output's* padding using the object's **own** mask and re-densifying
> is **tautological** (`to_dense` applies that same mask, so it always passes). Non-circularity requires
> the expected real-region to come from an independent source -> exact-mask assertions.

**Shared helpers** (in the test module):
- `_corrupt_padding(obj, scale=1e3)` — add `scale*(1 - apply_masks(ones_like(sc)))` to each supercore of a
  `UT3Variations` / `UniformTuckerTensorTrain` / `UT3Tangent` (corrupts the masked-out region per the
  object's own -- correct, for a valid input -- mask). Used on **inputs**.
- `_expected_doubled_masks(basis)` — independently build the doubled `(tucker_edge_mask, tt_edge_mask)`
  from the **base ranks** + the paper rule: tucker = prefix-pair `[up_prefix | down_prefix]`; tt =
  `[Q-block | P-block]` honest boundaries (`Q0=0, Qi=right_rank_i`; `Pd=0, Pi=left_rank_i`).
- `_PAD_FORCED` — padding strictly above the real max ranks so **every** core has a padded region.

**Additions (priority order):**
- **(A) Mask-once / garbage-input robustness** (likely to find bugs): corrupt each op's input padding, run
  it, assert the result matches the clean-input result (which matches ragged). Ops: `tangent_to_ut3`
  (both shifts), `retract`, `orthogonal_gauge_projection`, `oblique_gauge_projection`,
  `project_ut3_onto_tangent_space`, `gauge_residual`. Prime suspects: the **`squash_tt_tails` drop in
  `project`** and the boundary handling.
- **(B) Exact output-mask assertions** (the strictness fix): `tangent_to_ut3` masks ==
  `_expected_doubled_masks` (full arrays, unstacked/C/K+C/varying-C/forced-pad); `retract` masks == the
  base plain-UT3 masks per element; `orthogonal`/`oblique` gauge output masks == input masks; `project`
  output masks == base gauge masks; stack/unstack/sum round-tripped masks == original (exact array).
- **(C) Orthogonal gauge with a `K` stack** (+ `gauge_residual` with `K`) — currently only oblique has `K`.
- **(D) Multi-axis stacks** `C=(2,3)` / `K=(2,3)` across doubled-rank, retract, both gauges, project,
  stack/unstack — targets off-by-ones in the negative-axis / `1+n_K` / suffix arithmetic.
- **(E) Forced-larger padding** woven into a representative subset of the dense + (A) + (B) tests, so
  masking is exercised on every core (not just the sub-max-rank ones).

**Logistics:** all in `tests/test_uniform_manifold.py`; numpy-only correctness + existing jit coverage;
full-suite gate. **No impl changes unless a test finds a bug -> then fix the impl + note it** (do not
relax the test). An **independent adversarial audit agent** runs alongside (cold read of the impl, try to
construct a passing-but-wrong case); reconcile its findings with this plan. A durable testing-strategy
note for future developers follows once the pass is green.

---

# Increment 3b-6 — uniform tangent probing (`probing.py`) — detailed plan (2026-06-30)

_Designed with Nick 2026-06-30 (all four decisions resolved below); PENDING nothing — approved, starting
3b-6a. **Sequencing decision: probing and `probe_derivatives.py` are split into two slices.** 3b-6 does
plain probing (`probing.py`); the jet/derivative version (`probe_derivatives.py`) is the follow-on slice
**3b-6′**, built by mirroring this one (order-0 of every jet contraction reproduces the plain one we verify
here -- a built-in correctness anchor). 3b-7 (cleanup) is unchanged._

## Goal
Make `probing.py`'s **tangent** path run for uniform `UT3Tangent`: the forward Riemannian Jacobian `𝒥`
(`probe`/`apply`/`entries` on `UT3Tangent`) and its transpose `𝒥ᵀ`, verified per stack element against the
ragged path. Written **derivatives-aware** so the conventions extend cleanly to 3b-6′.

## Verified diagnosis (read the code, 2026-06-30)
- **Scan-style tangent fns already work** (`compute_sigmas`/`compute_taus`/**`compute_dxis`**): they
  `is_ndarray`-dispatch + `xscan` with proper `WKC` grouped-block steps. (`compute_dxis` is NOT broken --
  corrects an earlier plan line.)
- **Six map-style tangent fns are broken**: they have `if is_uniform:` branches, but those use raw `d...`
  einsums with the cores written `diaj` (no stack -> assume `C=()`) and a single `...` conflating `W` with
  `K`. Confirmed: `compute_detas` does `einsum('d...i,diaj->d...aj', sigmas, right_tt_cores)` where the
  ragged branch correctly calls `WKCa_Caib_WCb_to_WKCi(sigma, Q, nu)` + two siblings. The six:
  **`compute_detas`, `assemble_tangent_zs`** (forward) and **`compute_deta_tildes`, `compute_dxi_tildes`,
  `assemble_tucker_variations`, `assemble_tt_variations`** (transpose).
- **`compute_deta_tildes` is the wrong-contraction special case**: its uniform branch calls the
  outer-product `dCio_dWo_to_dWCi` (its comment even asserts "ztildes carry no separate T3 stack C" --
  false for a `C`-stacked tangent); it needs the **shared-`C`** `dWCo_dCio_to_dWCi`.
- **Two ragged-poly gaps** block the apply/entries transposes for uniform: `_apply_transpose_adjoint` is
  ragged-only (`get_backend(False)`), and `apply_tangent`/`entries_tangent` signatures still demand a
  `Sequence`.
- **The `d`-prefix recipe is already proven**: `dCio_dWo_to_dWCi` is exactly the `d`-prefixed
  `Cio_Wo_to_WCi` (same `_grouped_einsum` block-flattening). So 6a applies a proven pattern to the `WKC`
  family -- not new machinery.

## Resolved decisions (2026-06-30)
1. **Scope = tangent + corewise transpose flavors; ambient deferred.** 3b-6 does the tangent `𝒥`/`𝒥ᵀ` for
   all three ops (probe/apply/entries) AND the corewise transpose (it falls out of the same
   `_apply_transpose_adjoint` polymorphism fix). The **ambient transpose is deferred** (rarely used; the
   ambient *derivative* transpose is already a documented deferral).
2. **Derivatives-aware naming.** Name the contractions so the jet family is a clean superset
   (`trs_·d·WKC` with the order axis); note "order-0 == this plain contraction" on each -- the 3b-6′ anchor.
3. **Test home = a new `tests/test_uniform_probing.py`** (mirror `tests/test_probing.py` for the uniform
   tangent path), not growing `test_uniform_manifold.py`.
4. **Granularity = 6a/b/c/d** (contraction foundation isolated + proven first).

## Sub-slices (dependency-ordered, each suite-gated)
- **3b-6a -- the `d`-prefixed uniform `WKC` contractions** (`contractions.py`). Built + tested in ISOLATION
  (no `probing.py` changes). Each = the ragged `WKC` einsum with `d` prepended to every operand + output,
  routed through `_grouped_einsum` (capital blocks flatten; `d` rides as a leading batch). Inventory
  (ragged twin -> consumer):
  - `dWKCa_dCaib_dWCb_to_dWKCi` (`WKCa_Caib_WCb_to_WKCi`) -- compute_detas, compute_dxi_tildes
  - `dWCa_dKCaib_dWCb_to_dWKCi` (`WCa_KCaib_WCb_to_WKCi`) -- compute_detas
  - `dWCa_dCaib_dWKCb_to_dWKCi` (`WCa_Caib_WKCb_to_WKCi`) -- compute_detas, compute_dxi_tildes
  - `dWKCi_dCio_to_dWKCo`, `dWCi_dKCio_to_dWKCo` (`WKCi_Cio_to_WKCo`, `WCi_KCio_to_WKCo`) -- assemble_tangent_zs
  - `dWCo_dCio_to_dWCi` shared-C (`WCo_Cio_to_WCi`) -- compute_deta_tildes
  - `dWKCo_dWCa_to_{dWKCao,dKCao}`, `dWo_dWKCa_to_{dWKCao,dKCao}` (the `WKCo_WCa_*`/`Wo_WKCa_*` quartet) -- assemble_tucker_variations
  - the three `d·WKCiaj`/`d·KCiaj` builders (`WCi_WCa_WKCj_to_*` family) -- assemble_tt_variations
  ~16 contractions (some shared). **Test (the strict part):** per leading-`d` index the result equals the
  ragged `WKC` contraction applied to `operand[i]`, over EVERY `W`/`K`/`W+K`/`+C` combo (the multi-block
  stacking is the historically-untrusted part).
- **3b-6b -- forward `𝒥`.** Rewrite `compute_detas`/`assemble_tangent_zs` uniform branches to the new
  contractions -> `probe_tangent`/`apply_tangent`/`entries_tangent` dispatch for uniform. Wire
  `UT3Tangent.probe`/`apply`/`entries` via a `ut3_sampling`-style tangent helper (mask-once basis+variations
  -> `pack_vectors` -> `probing.probe_tangent` -> `unpack_vectors`). Test forward equivalence per element.
- **3b-6c -- transpose `𝒥ᵀ` + corewise.** Rewrite the four broken transpose branches (incl.
  `dWCo_dCio_to_dWCi` for `compute_deta_tildes`); make `_apply_transpose_adjoint` polymorphic + relax the
  `Sequence` signatures (old Slice-4 gaps). Wire `UT3Tangent.{probe,apply,entries}_transpose` (tangent) +
  the corewise transposes. Test `⟨r,𝒥V⟩ = ⟨𝒥ᵀr,V⟩` per element + dense, both `sum_over_probes` modes.
- **3b-6d -- harden + dispatch.** Mask-strict (exact output masks where a tangent is returned) +
  garbage-padded-input robustness + varying-`C` + jit dispatch, per `docs/testing_strategy.md`.

## Follow-on: 3b-6′ -- uniform `probe_derivatives.py` (detailed plan, 2026-06-30 cont.)

_The jet/derivative version of 3b-6 (`probe_derivatives.py` is the "jetted" probing: same computational
structure + a leading derivative-order axis via a binomial jet-product). Split from 3b-6 (agreed with Nick):
build 3b-6 first as the verified foundation, then mirror it here. **Written for a future agent/session with
little context** -- read the "Lessons from 3b-6" subsection below FIRST; it is the map to the traps._

### Scope -- BOTH layers (the correction: not just the tangent)
`probe_derivatives` must work for **both**:
1. the **plain `UniformTuckerTensorTrain`** -- the base derivative sampling. **It has NO `*_derivatives`
   methods today** (grep: the ragged `TuckerTensorTrain.{probe,apply,entries}_derivatives` at
   `tucker_tensor_train.py` 3735 / 3774 / 3809 have no uniform counterpart). Add the 3 methods (+ the plain
   backend wrappers in `ut3_sampling`, mirroring `ut3_probe`/etc. with a `pp` perturbation-vector pack and
   the leading order axis on the output).
2. the **`UT3Tangent`** -- the Riemannian `𝒥`/`𝒥ᵀ` derivatives + the corewise derivative transposes (mirror
   ragged `T3Tangent.{probe,apply,entries}_derivatives` + `*_derivatives_transpose`, and
   `TuckerTensorTrain.*_corewise_derivatives_transpose`). Backend wrappers in `ubv_sampling` /
   `ut3_sampling`. **The ambient derivative transpose stays DEFERRED** (`docs/ambient_derivative_transpose_note.md`).

Both layers are broken the same way and get fixed by the same contraction work -- do them together per op.

### Verified state (2026-06-30) -- what's uniform-ready, what's broken
- **Scan-style jet fns ARE uniform-aware** (`is_ndarray` + `xscan`): `compute_mu_jets` / `compute_nu_jets`
  / `compute_eta_jets` / `assemble_z_jets` / `compute_sigma_jets` / `compute_tau_jets` (probe_derivatives.py
  271 / 318 / 339 / 618 / 535 / 591). Leave them; they use the `trs_*` contractions inside the scan step.
- **Map-style jet fns hardcode `get_backend(False)` -> BROKEN for uniform** (the exact analog of the plain
  probing map-style break): `_apply_derivatives_t3_from_xi_jets` (726), `_apply_derivatives_from_jets`
  (774), `compute_deta_tilde_jets` (960), `_adj_sweep` (975; the reverse-sweep helper behind
  `compute_tau_tilde_jets`/`compute_sigma_tilde_jets`), `compute_dxi_tilde_jets` (1029),
  `assemble_tucker_variation_jets` (1055), `assemble_tt_variation_jets` (1086), `compute_sigma_hat_jets`
  (1207), `_apply_derivatives_transpose_from_jets` (1233). Rewrite each uniform branch with the
  `d`-prefixed jet contractions + `is_ndarray` dispatch.
- **The UNROLL TRAP -- `build_input_jets` (line 244)** does `tuple(xnp.stack([xi, dxi], axis=0) for xi, dxi
  in zip(xis, dxis))` -- iterating the supercore `d` axis (a Python loop). This is EXACTLY the
  `_entry_xis` bug 3b-6c fixed: for uniform it "works" by ragged-emulation but **unrolls under jit** and
  returns ragged tuples. Fix: for uniform (`xis` a supercore `(d,)+W+C+(nU,)`) vectorize as
  `xnp.stack([xis, dxis], axis=1)` -> `(d,)+(2,)+W+C+(nU,)` (order axis at axis 1, AFTER the leading `d`).
  `_init_jet` (247) builds `mu_0` with an explicit order axis -- for uniform it needs the `d`-leading
  layout too (order axis inserted after `d`). Also audit the plain entries `probe_derivatives_t3` (109),
  `apply_derivatives_t3` (742), `entries_derivatives_t3` (853) for the same. `check_perturbation_*` (184 /
  202) are host-side shape checks on the *ragged input vectors* (pre-pack) -- fine, not a trap.

### Sub-slices (mirror 3b-6a/b/c/d)
- **3b-6′a -- the `d`-prefixed jet contractions** in `contractions.py`. Each is a ragged `trs_*` `WKC`
  contraction (`contractions.py` ~1374-1785: `trs_rWKCa_Caib_sWCi_to_tWKCb`, the `tWKCi_Cio_to_tWKCo`
  family, the `trs_..._to_WKCaib`/`KCaib` assemble family, `tWCa_tWKCo_to_{WKCao,KCao}`, ...) with `d`
  prepended to every operand + output. **`d` AND the order axes (`r,s,t,u`) ride as leading batches**; the
  `W`/`K`/`C` block-flattening is unchanged (same recipe as 3b-6a). Test with the **existing `_check_jet3`
  harness** (`tests/backend/test_contractions.py` 631) + a `d`-prefixed variant (mirror `_check_3group_d`).
  **The order-0 anchor:** each jet contraction at order 0 reduces to the plain `dWKC` contraction verified
  in 3b-6a -- assert it (the ragged `WCa_WCi_WKCb_to_WKCaib` docstring already notes "the order-0 strip of
  `trs_rWCa_uWCi_tWKCb_to_WKCaib`"). Also need the plain-jet (`trs_·WC`, no K) `d`-prefixed twins for the
  base-point (plain UniformTTT) derivative path.
- **3b-6′b -- forward `𝒥` derivatives.** Fix `build_input_jets` + `_init_jet` (the unroll trap) + the
  forward map-style jet fns (`_apply_derivatives_*_from_*`, `assemble_z_jets` is already OK). Wire
  `UniformTuckerTensorTrain.{probe,apply,entries}_derivatives` AND `UT3Tangent.{...}_derivatives` (backend
  wrappers: mask-once + pack `ww` **and `pp`** + call + unpack; the output carries the leading order axis).
- **3b-6′c -- transpose `𝒥ᵀ` derivatives + corewise.** Fix `_adj_sweep` / `compute_*_tilde_jets` /
  `assemble_*_variation_jets` / `compute_sigma_hat_jets` / `_apply_derivatives_transpose_from_jets`. Wire
  the `*_derivatives_transpose` (tangent) + `*_corewise_derivatives_transpose` (the §6.3 substitution, as in
  3b-6c). Gauge masks over `K_new`: reuse `ubv_sampling._gauge_masks_over_Knew` (it already exists).
- **3b-6′d -- harden.** Same as 3b-6d (`tests/test_uniform_probing.py` helpers extend cleanly): the adjoint
  identity **per order**, per-element vs ragged over `_CONFIGS` + varying-`C`, garbage-robustness, exact
  masks, forced-pad, jit.

### Lessons from 3b-6 -- carry these into 3b-6′ (read first)
1. **Hunt the unroll trap FIRST.** Any `for ... in enumerate(cores)` / `zip(cores)` / `cores[k]` that walks
   a uniform **supercore's leading `d` axis** silently "works" by ragged-emulation but (a) **unrolls under
   jit** (compile time superlinear in `d`) and (b) returns **ragged tuples** that break downstream
   supercore-expecting code. 3b-6c hit this in `_entry_xis` (forward entries looked fine -- even passed a
   jit test -- because unrolling is numerically correct). In `probe_derivatives` the known instance is
   `build_input_jets` (244); grep `enumerate(` / `get_backend(False` / `[k]` and audit each. Fix by
   vectorizing over `d`, inserting the order axis at **axis 1** (after `d`).
2. **The `d`-prefix recipe generalizes cleanly.** Take the ragged contraction's `_grouped_einsum` string,
   prepend `d` to every operand + output; `d` (and, for jets, the order axes `r,s,t,u`) ride as leading
   batches; the `W`/`K`/`C` front-counted slices shift by `+1` per leading batch axis. The reshape gains a
   leading `d_shape` (+ order shapes). See the 3b-6a `dWKC` family (`contractions.py`, the "d-prefixed
   uniform WKC contractions" section) as the worked template; the ragged `trs_*` family is right above it.
3. **Order 0 is a free correctness anchor.** Every jet contraction at order 0 == the plain `dWKC` one you
   already verified in 3b-6a. Assert it -- it catches a whole class of order-axis bookkeeping slips for
   nearly free, on top of the full `np.einsum` oracle.
4. **The map/scan split is the whole diagnosis.** Scan-style fns (`xscan` over `d`, `trs_*` grouped-block
   step) were already made uniform-aware upstream; only the **map-style** fns (a per-core map -> a
   vectorized `d`-axis einsum) are broken. This held for plain probing (3b-6b/c) and holds for the jets.
5. **The frontend boundary is a fixed pattern.** `ubv_sampling` (tangent) / `ut3_sampling` (plain):
   **mask-once** the supercores, **pack** the vectors (`ww` -- and for derivatives **also `pp`**), call the
   now-polymorphic backend (`is_uniform` inferred from the masked supercores being bare ndarrays),
   **unpack** the probe output. Transposes return a tangent -> attach masks via
   `ubv_sampling._gauge_masks_over_Knew` (gauge masks over the new tangent stack `K_new` = `W+K` if
   `sum_over_probes=False`, `K` if `True`). Corewise transposes return **raw gradient supercores** (no mask
   holder; clean-padded). `n_probe` for a packed uniform `ww` is `ww[0].ndim - 1` (a `ww[0]` d-slice) or
   `ww.ndim - 2` inside a `d`-prefixed map fn; the order axis does not change it.
6. **`_onehot_vectors` / `_entry_xis` are already polymorphic** (3b-6c): the jet entries/entries-transpose
   derivative paths reuse them -- don't re-break them. `_entry_xis` is a **vectorized fiber-slice gather**
   (advanced indexing on axis 0 + axis -1, broadcasting to `(d,)+W`), NOT a one-hot contraction (which would
   re-introduce the `N` factor entries exists to avoid).
7. **Test spine: adjoint identity + per-element-vs-ragged + hardening.** `⟨r, 𝒥V⟩ = ⟨𝒥ᵀr, V⟩` (now summed
   appropriately per order) is the strongest, cheapest transpose check; add per-element-vs-ragged and the
   garbage/exact-mask hardening (`docs/testing_strategy.md`). Reuse `tests/test_uniform_probing.py`'s
   helpers (`_corrupt`, `_expected_gauge_masks`, `_full_unstack`, the `_CONFIGS` matrix); the order axis
   just adds one leading axis to the reshapes.
8. **`xnp` often goes unused** in a rewritten uniform branch (the `d`-prefixed contractions infer their own
   backend from the arrays). Leave the `xnp, xmap, xscan = get_backend(...)` unpack; don't chase the lint.
