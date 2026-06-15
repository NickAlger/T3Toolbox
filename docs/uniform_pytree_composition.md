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

- **The dtype debate dissolves.** Because identity hashing never inspects mask contents, the masks may
  be numpy or jax — the cache mechanism does not care. So masks can simply follow the supercore dtype
  on construction, with no impact here.
- **The backend is unaffected.** The holder is a frontend/pytree concern. Backend functions still take
  raw arrays (`(tucker_supercore, tt_supercore)` and the three mask arrays), so a user working on raw
  `.data` is not forced through the holder — consistent with the backend/frontend razor.

## Honest costs

- **One extra class and a composition step**, versus a flat five-field record. Mild, and offset by the
  holder being a real object, but real.
- **Identity-keyed caching is unforgiving.** Two UT3s with *equal* structure built as *separate* objects
  are distinct `aux_data`, so each triggers its own `jit` compilation — you must hold the `masks` object
  stable to get cache hits (the same caveat `T3Tangent` carries for its base point). The rejected
  alternative — a flat five-field UT3 with a *content*-hashable mask adapter — would cache across equal
  rebuilds, but at the price of pinning masks to numpy (host-side byte hashing) and re-hashing the mask
  bytes at every flatten boundary. We preferred matching the established identity-hashing pattern and
  keeping the masks dtype-free.

## Scope note

The pytree registration and hashing are *jax-wiring*, implemented later. This note fixes the **class
shape** now — `supercores + structure holder` — because that decision sets the constructor signature,
the `.data` layout, and every derived property, all of which the rest of the build depends on.
