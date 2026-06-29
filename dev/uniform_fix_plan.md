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
