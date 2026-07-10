# Composing UniformTuckerTensorTrain: dynamic supercores + a static structure holder

> A design-philosophy note for `UniformTuckerTensorTrain`. It records *why* the class is composed as
> two supercore fields (dynamic, differentiable) plus one small *structure holder* field (static,
> non-differentiable), rather than as a flat record — and why that holder is **value-hashed by its mask
> content** (so a rebuilt-but-identical holder is the same `jit` cache key). As with the other notes,
> this is an honest accounting of the choice and its costs, not a
> rule to follow on faith. It pairs with `docs/uniform_masks_vs_ranks.md` (why the structure is stored
> as boolean masks) and `docs/uniform_ranks_and_varieties.md` (what the object represents).

---

## The forcing constraint: masks are static, and static lives in pytree `aux_data`

The supercores are data we differentiate, `jit`, and `vmap` through. The masks are *structure* — the
variety stratum plus ambient dimensions (`docs/uniform_ranks_and_varieties.md`) — and we explicitly do
**not** want gradients flowing through them. In a JAX pytree that distinction is exactly children vs.
`aux_data`:

- **children** are the dynamic leaves: traced by abstract shape/dtype, never hashed by content.
- **`aux_data`** is the static part, carried in the *treedef* — and the treedef is `jit`'s **cache
  key**, so `aux_data` must be **hashable**, and treedef equality compares it with `==`.

So masks belong in `aux_data`. But a raw array — numpy *or* jax — cannot be `aux_data`: it is
unhashable, and `array == array` returns an *array*, not a single bool. (The current broken UT3 code
puts a bare tuple of three mask arrays in `aux_data`; that does not survive `jit`.) Something has to
make the static masks hashable.

## The decision: a small structure holder, value-hashed by mask content

```python
@dataclass(frozen=True, eq=False)   # eq=False suppresses the dataclass __eq__ (it would do the
class UT3Masks(ValueHashedMasks):   # array-ambiguous `array == array`); the mixin supplies VALUE-based
    tucker_edge_mask                # __hash__/__eq__ over mask content (common.ValueHashedMasks)
    tt_edge_mask
    # exposes the rank metadata (tucker_ranks, tt_ranks, …) derived from the masks

@dataclass(frozen=True)
class UniformTuckerTensorTrain:
    tucker_supercore        # dynamic  -> pytree child
    tt_supercore            # dynamic  -> pytree child
    shape: Tuple[int, ...]  # static   -> pytree aux_data (a value-hashable int tuple — the physical mode
                            #             dims are a contiguous prefix, so an int tuple suffices, no mask)
    masks: UT3Masks         # static   -> pytree aux_data (value-hashed by content)
```

The holder hashes and compares by **mask content** — `common.ValueHashedMasks`: a cached `tobytes` hash +
`np.array_equal` equality, with an `is`-identity fast path. `eq=False` is still required (it suppresses
the dataclass-generated `__eq__`, which would compute the array-ambiguous `array == array`), so the
mixin's value-based methods stand.

**Why value, not identity.** The holder rides in `aux_data`, which *is* `jit`'s cache key. If it hashed by
*object identity* (the bare `eq=False` default), a rebuilt-but-array-identical holder would be a **new**
key → recompile. That is fatal for the layer's whole purpose: in a manifold-optimization loop the
orthogonal frame is rebuilt every iteration (`ut3_orthogonal_representations` produces fresh mask holders,
but at fixed rank the *structure is unchanged*), so identity hashing would recompile **every step**, and
recompilation dwarfs the per-step compute — the exact opposite of why the uniform layer exists. Value
hashing makes the cache key reflect the rank **structure**: identical structure → cache hit (compile
once); a genuinely different structure → recompile (correct). A regression test pins this contract
(`tests/test_dispatch.py::test_mask_rebuild_does_not_recompile`).

The holder is deliberately **not** itself a registered pytree: as opaque `aux_data` its masks are never
flattened into children, so they cannot be traced or differentiated — precisely the guarantee we want. It
is also genuinely meaningful, not just a hashability wrapper: it *is* the rank-structure descriptor
(`tucker_ranks`, `tt_ranks`, … derive from it), so the dataclass stays minimal (two rank masks; `shape`
and the sizes are not stored redundantly). The same mixin is reused by the frame/variation holders
(`UT3BasisMasks`, and the upcoming `UT3VariationsMasks`).

## Consequences worth knowing

- **Masks are numpy (host), not jax — and this is load-bearing, not a free dtype choice.** The masks must
  be concrete host arrays because their *values* are read (`int(mask.sum())`), recomputed into aux, and now
  **value-hashed** (`tobytes`) as the jit cache key — all of which a tracer breaks. See *Masks are numpy
  (host) — the jit story* below.
- **The backend is unaffected (raw arrays).** The holder is a frontend/pytree concern; backend functions
  take **raw arrays** in a layout that mirrors the fields — `.data = (tucker_supercore, tt_supercore,
  shape, (tucker_edge_mask, tt_edge_mask))`, supercores flat, then the static `shape` int tuple, then the
  two rank masks grouped as a sub-tuple (supercore-only ops take `.data[:2]`; `shape` is `.data[2]`;
  mask-using ops unpack `.data[3]`). A user on raw `.data` is never forced through the holder — consistent
  with the backend/frontend razor.

## Masks are numpy (host) — the jit story

The masks are `aux_data`, i.e. *static structure*. The hard-won point (it cost a debugging session —
see `dev/archive/uniform_slice_handoff.md`) is that **the masks must be stored *and computed* as numpy (host)
arrays, even when the supercores are jax.** This is required for jit correctness; it is not a
backend-agnosticism slip.

**Why jax masks break under jit.** Inside a jit trace, *every* `jnp` op returns a tracer — even on a
concrete constant. So if the masks are jax arrays, any mask op inside a traced function is a tracer:

- `int(mask.sum())` — the host-int shape/rank extraction (e.g. `ut3_to_dense`'s static prefix-slice,
  `t3svd`, the `.shape` property) → `ConcretizationTypeError`: you cannot pull a Python `int` from a
  tracer.
- mask *recomputation* — the rank recurrences in orthogonalization/svd and the `+`/`×` concat/Kronecker
  → the new masks are tracers, which would then **leak into the output's `aux_data`**. (Value hashing now
  makes this fail fast — `tobytes` on a tracer raises at hash time — instead of silently producing an
  object whose masks are escaped tracers; `require_concrete_masks` guards it explicitly regardless.)

Numpy masks avoid both: numpy ops run on the host and are never staged into the jaxpr, so
`int(mask.sum())` is a real host int and recomputed masks stay concrete. **So all mask logic uses `np`,
not `xnp`:** the structure is resolved on the host; only the data (supercores) flows through `xnp`.

**Under jit this costs nothing — it is optimal.** Because the mask logic is numpy on concrete inputs,
the whole structural computation (rank recurrences, prefix masks, shape extraction, concat/Kronecker)
runs **once at trace/compile time on the host** and folds into the compiled program as constants. Per
call, only the *data* computation runs on the device, with the masks already resident as compile-time
device constants. That is the ideal split — structure at compile (host), data at runtime (device) —
with **zero per-call host↔device mask transfer**.

**Eager (non-jit) GPU is the only place a transfer happens, and it is small.** Run eagerly, the masking
multiply (`supercore * mask`) moves the (boolean, KB-scale) masks host→device per op — a latency-bound
~µs cost. It is not a new bottleneck: eager jax is already dominated by per-op Python *dispatch*
overhead (tens of µs/op — the reason to jit), which dwarfs the mask transfer; and terminal ops
(`to_dense`, scalar `inner`/`norm`, measurement results) pull a larger result back to host anyway. The
only way to avoid even this is device masks — which break jit. That is a bad trade: sacrificing the
entire jit path (uniform's whole purpose) to save µs on the off-purpose eager path (eager work is the
ragged layer's job).

**Deferred future option (do NOT build speculatively):** if eager-GPU uniform is ever *profiled* as a
real hot path, cache a device copy of the masks (`jax.device_put` once, reuse the handle) so repeated
eager ops don't re-transfer. Local optimization, not a design change, and explicitly deferred — the
standing guidance is "if you care about performance, jit," at which point the masks are free.

### How to jit a uniform op — purely functional, or via the frontend

The rule is just *keep the masks static, trace the supercores* — and the **purely functional path needs
no class** (the OO-averse backend user is fully supported):

- **Functional (raw `.data`).** Close over the masks (host constants) and trace only the supercores —
  the standard jax idiom:

  ```python
  masks = (shape_mask, tucker_edge_mask, tt_edge_mask)        # HOST bool, static
  dense = jax.jit(lambda tk, tt: ut3_to_dense((tk, tt, masks)))(tucker_sc, tt_sc)
  ```

  The closed-over numpy masks become compile-time device constants; the supercores are the traced args.
  (`static_argnums`/`static_argnames` is *not* the route — the masks are arrays, hence unhashable, so
  they can't be marked static without a wrapper; close-over is the clean move.)
- **Frontend.** If you do use `UniformTuckerTensorTrain`, `jax.jit` over it works with nothing to think
  about: the masks ride as `aux_data` (static), the supercores are children (traced).

**The misuse, and the guard.** Passing the masks *among the traced args* (e.g. `jax.jit(ut3_to_dense)(data)`,
which traces every leaf of the tuple, masks included) makes them tracers → the host-int extraction and
mask recomputation fail. The backend **guards** this: a traced mask raises a clear, actionable error
("uniform masks must be concrete host arrays … close over the masks and trace only the supercores")
instead of jax's cryptic `ConcretizationTypeError`. So the failure is self-explaining; the right and
wrong forms are shown as a doctest.

### Why `T3Basis` (jax arrays) is fine as aux, but masks aren't

`T3Tangent`'s `T3Basis` may hold jax arrays as aux (its arrays are used only as data; in the current code
it actually flows as a pytree *leaf* via a numerical same-frame guard — see `manifold.py`). The UT3 masks
cannot be jax — for reasons that are about *what the op does with the aux*, not the dtype label:

- **`T3Basis` is aux used only as *data*.** Its cores feed contractions (`einsum`), whose tracer results
  flow on as data. It is never read as a host value (its *structure* is its array shape, which jax tracks
  statically — no `int(basis…)`), and a tangent op at a fixed base **reuses the same input basis**
  unchanged as the output aux. Nothing forces it concrete → jax is fine.
- **The masks' *values* ARE host-readable static structure — read, recomputed, AND now hashed.** A uniform
  op (a) pulls the real shape/ranks as host Python ints — `int(mask.sum())` — to slice the padded supercore
  (the mask *values*, not the array shape, say which slot is real); (b) **recomputes** the masks (ranks
  change under `+`/`×`/svd) into the output's *static* aux; and (c) the holder now **value-hashes** the
  masks (`tobytes`) as the `jit` cache key. All three demand concreteness → host numpy (a jax mask gives a
  tracer on `.sum()`/`.tobytes()`, and recomputed jax masks would leak into aux). Value hashing actually
  *strengthens* this: a stray tracer mask now fails fast at hash time rather than leaking silently.

One line: **basis = aux-used-as-data** (jax fine); **masks = aux whose values are host-readable structure
and are recomputed** (must be numpy). (The basis *would* hit the same wall if you re-orthogonalized a base
inside jit and wrapped a fresh tangent around it — which is why the convention is to hold the basis object
stable; uniform can't sidestep it, because changing the structure *is* the op.)

> **⚠️ Maintainers (human or AI): the `np.*` in the uniform mask code is INTENTIONAL.** Historically a
> bare `np.` was a tell that code wasn't backend-agnostic (should be `xnp`). **That heuristic does not
> apply to mask logic.** Masks are host structure and MUST be numpy. Do **not** "fix" mask `np.*` to
> `xnp`/jax for consistency — it silently breaks jit (tracer-leak into `aux_data`, `int()`
> concretization errors). The rule across the uniform layer is: **supercores (data) → `xnp`; masks
> (structure) → `np`.** (`make_uniform_masks` and the other mask builders therefore always emit numpy,
> with no `use_jax` flag; `to_jax`/`to_numpy` convert the supercores only, never the masks.)

## Honest costs

- **One extra class and a composition step**, versus a flat five-field record. Mild, and offset by the
  holder being a real object, but real.
- **Value hashing costs a small content hash/eq per `jit` call** (the price of *not* recompiling). The
  holder hashes its mask bytes (`tobytes`, cached on the frozen object) and compares with
  `np.array_equal` (with an `is`-identity fast path). That is `O(mask size)` = rank-sized — far smaller
  than the supercores, and negligible next to the op, let alone the recompile it prevents. We key on the
  **mask bytes**, not an integer rank *count*: off canonical form the real slots scatter (add = concat of
  two prefix masks → a gappy, non-prefix pattern), so a rank count cannot key it (see
  `uniform_masks_vs_ranks.md`); the bytes key both canonical and gappy forms. *(Optional further
  optimization: have the optimizer reuse the same `masks` object across iterations at fixed rank, so even
  the `array_equal` short-circuits on `is` — gravy, not required for correctness.)* This **retires** the
  earlier identity-hashing caveat (the original design deferred value hashing; the optimization-loop
  recompile it caused is exactly why we no longer defer it).

## Scope note

The pytree registration and hashing are *jax-wiring*, implemented later. This note fixes the **class
shape** now — `supercores + structure holder` — because that decision sets the constructor signature,
the `.data` layout, and every derived property, all of which the rest of the build depends on.
