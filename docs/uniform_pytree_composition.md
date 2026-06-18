# Composing UniformTuckerTensorTrain: dynamic supercores + a static structure holder

> A design-philosophy note for `UniformTuckerTensorTrain`. It records *why* the class is composed as
> two supercore fields (dynamic, differentiable) plus one small *structure holder* field (static,
> non-differentiable), rather than as a flat five-field record — and why that holder is an `eq=False`
> dataclass. As with the other notes, this is an honest accounting of the choice and its costs, not a
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

## The decision: a small `eq=False` structure holder

```python
@dataclass(frozen=True, eq=False)   # eq=False -> identity __hash__/__eq__, so this array-holding
class UT3Masks:                     # object is valid jax aux_data (value hash/eq is impossible)
    shape_mask
    tucker_edge_mask
    tt_edge_mask
    # exposes d, N, n, r, stack_shape, shape, tucker_ranks, tt_ranks — all derived from the masks

@dataclass(frozen=True)
class UniformTuckerTensorTrain:
    tucker_supercore     # dynamic  -> pytree child
    tt_supercore         # dynamic  -> pytree child
    masks: UT3Masks      # static   -> pytree aux_data
```

`eq=False` leaves Python's default **identity** `__hash__`/`__eq__` in place, so the holder is hashable
by `id()` and its array contents are never inspected. `jit` then caches on the holder's *object
identity*: reuse the same `masks` object → cache hit; a structurally different one → recompile. This is
the **exact** mechanism the verified code already uses — `T3Basis` is `@dataclass(frozen=True, eq=False)`
and rides as `T3Tangent`'s `aux_data` for the same reason. So `UniformTuckerTensorTrain = supercores +
masks` is the same "dynamic data + static frame" decomposition as `T3Tangent = variations + basis`, not
a new pattern.

The holder is deliberately **not** itself a registered pytree: as opaque `aux_data` its masks are never
flattened into children, so they cannot be traced or differentiated — which is precisely the guarantee
we want. It is also genuinely meaningful, not just a hashability wrapper: it *is* the structure
descriptor, and everything (`d`, `N`, `n`, `r`, `stack_shape`, `shape`, `tucker_ranks`, `tt_ranks`) is
derived from it, so the dataclass stays minimal (three masks; no stored redundant sizes).

## Consequences worth knowing

- **Masks are numpy (host), not jax — and this is load-bearing, not a free dtype choice.** An earlier
  version of this note claimed the dtype "dissolves" because identity hashing never inspects contents.
  That is wrong: the issue is not hashing (identity hashing *is* dtype-agnostic) but the **mask
  computations**. See *Masks are numpy (host) — the jit story* below.
- **The backend is unaffected (raw arrays).** The holder is a frontend/pytree concern; backend functions
  take **raw arrays** in a layout that mirrors the fields — `.data = (tucker_supercore, tt_supercore,
  (shape_mask, tucker_edge_mask, tt_edge_mask))`, supercores flat and the three masks grouped as a
  sub-tuple (supercore-only ops take `.data[:2]`; mask-using ops unpack `.data[2]`). A user on raw
  `.data` is never forced through the holder — consistent with the backend/frontend razor.

## Masks are numpy (host) — the jit story

The masks are `aux_data`, i.e. *static structure*. The hard-won point (it cost a debugging session —
see `uniform_slice_handoff.md`) is that **the masks must be stored *and computed* as numpy (host)
arrays, even when the supercores are jax.** This is required for jit correctness; it is not a
backend-agnosticism slip.

**Why jax masks break under jit.** Inside a jit trace, *every* `jnp` op returns a tracer — even on a
concrete constant. So if the masks are jax arrays, any mask op inside a traced function is a tracer:

- `int(mask.sum())` — the host-int shape/rank extraction (e.g. `ut3_to_dense`'s static prefix-slice,
  `t3svd`, the `.shape` property) → `ConcretizationTypeError`: you cannot pull a Python `int` from a
  tracer.
- mask *recomputation* — the rank recurrences in orthogonalization/svd and the `+`/`×` concat/Kronecker
  → the new masks are tracers, which then **leak into the output's `aux_data`**. There is no error at
  flatten time (the holder is identity-hashed and never inspects its contents), so the returned object
  *looks* fine but its masks are escaped tracers — silently invalid.

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
- **Identity-keyed caching is unforgiving.** Two UT3s with *equal* structure built as *separate* objects
  are distinct `aux_data`, so each triggers its own `jit` compilation — you must hold the `masks` object
  stable to get cache hits (the same caveat `T3Tangent` carries for its base point). The alternative is
  a **content-hashed** holder — hash the rank/mask bytes so equal rebuilds (e.g. an optimization loop
  `ux_{i+1} = op(ux_i)` at fixed structure) hit the cache. With masks now numpy regardless (above), this
  alternative no longer carries the "pins masks to numpy" cost the original design feared — so it is a
  viable **future enhancement** (hash the canonical-form ranks, or the mask bytes for gappy working
  forms; see `uniform_masks_vs_ranks.md` for why a rank *count* alone is an insufficient key off
  canonical form). For now we keep identity hashing, matching the established `T3Tangent` pattern, and
  defer value hashing as an optional improvement.

## Scope note

The pytree registration and hashing are *jax-wiring*, implemented later. This note fixes the **class
shape** now — `supercores + structure holder` — because that decision sets the constructor signature,
the `.data` layout, and every derived property, all of which the rest of the build depends on.
