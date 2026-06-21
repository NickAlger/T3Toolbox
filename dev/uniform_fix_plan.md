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

## Next
Run the **triage survey** (apply the lenses to the actual uniform/ragged code; assess current state).
