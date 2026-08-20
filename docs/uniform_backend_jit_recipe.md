# Jitting uniform T3s: hold masks fixed, trace only supercores

> A backend-usage note for the uniform layer. A backend user who bypasses the OO frontend and works on raw
> `.data` tuples needs to jit an optimization loop **without recompiling every step** — which matters
> because accelerating this kind of fitting is the whole point of the uniform layer. This records the recipe
> and why it works. Pairs with `contributor/uniform_pytree_composition.md` (the frontend value-hashing story),
> `contributor/uniform_svd_prefix_orthogonalization.md`, and `contributor/uniform_rank_masks_rationale.md`.

## The constraint

A uniform `.data` tuple carries **masks** (static structure, host numpy). They cannot be jit-*traced*: a
traced mask breaks host-int extraction (`int(mask.sum())`) and is rejected by `require_concrete_masks`. So
under jit the masks must be **static** — closed over, not passed as traced args.

## The recipe (the `optimizers.py` pattern, uniform-adapted)

Jit the **whole per-step kernel once**, closing over the loop-invariant masks; trace only the supercores
(+ minibatch):

```python
>>> import numpy as np, jax
>>> import t3toolbox.tucker_tensor_train as t3
>>> import t3toolbox.uniform_tucker_tensor_train as ut3
>>> from t3toolbox.backend.ufv_conversions import ut3_orthogonal_representations
>>> from t3toolbox.backend.utv_operations import utv_retract
>>> np.random.seed(0)
>>> x = ut3.UniformTuckerTensorTrain.from_t3(
...         t3.TuckerTensorTrain.randn((4, 5, 6), (2, 3, 2), (1, 2, 2, 1)))
>>> shape, masks = x.shape, x.masks.data       # LOOP-INVARIANT at fixed rank -- built once
>>> traces = []                                # counts how many times the body is TRACED
>>> @jax.jit
... def step(supercores):
...     traces.append(1)                                             # runs at TRACE time only
...     data = (supercores[0], supercores[1], shape, masks)          # masks/shape CLOSED OVER (constants)
...     frame_data, var_data = ut3_orthogonal_representations(data)  # re-orthogonalize INSIDE the trace
...     var_data = (0.9 * var_data[0], 0.9 * var_data[1]) + var_data[2:]  # stand-in gradient step
...     return utv_retract(frame_data, var_data)[:2]                 # retract -> new supercores
>>> supercores = tuple(jax.numpy.asarray(s) for s in x.supercores)
>>> for _ in range(3):                         # the supercores CHANGE every step
...     supercores = step(supercores)
>>> print(len(traces))                         # compiled ONCE, despite re-orthogonalizing each step
1

```

- The frame masks are **closed over** → the kernel compiles once; no recompile.
- `ut3_orthogonal_representations` re-derives the **frame/variation masks inside the trace** from the
  *concrete* frame ranks → host-numpy constants, **constant-folded** into the compiled program. They are
  **not** part of the jit cache key (the key is the closure identity + the supercores' avals), so
  re-deriving them every step is free — it happens once, at trace time — and causes no recompile.
  Empirically: re-orthogonalizing every step with *changing* supercores → **1 compile**.

## Why the masks are safe to hold fixed

At fixed rank the masks **never change**: SVD-based orthogonalization yields a *deterministic prefix*
(`contributor/uniform_svd_prefix_orthogonalization.md`), and the uniform retraction (max-rank-only, no numerical
truncation) preserves the padded ranks. So the mask objects are genuinely loop-invariant -- reuse them.

## The anti-pattern (what NOT to do)

Running `ut3_orthogonal_representations` **outside** a jit and passing its (fresh-object) output masks
**into** a separate jit -- either traced (rejected by the guard) or closed over by a **freshly-built
closure each iteration** (a new cache key → recompile *every* step). The fix is this recipe: hold the masks
fixed and reuse them, or jit the whole step so the frame masks stay internal constants.

The library's own uniform optimizers follow this same recipe internally (masks as loop-invariant
state, recomputed only at rank-continuation boundaries); the design record is
[`contributor/uniform_internals.md`](contributor/uniform_internals.md).

## The frontend path, the raw `.data` layout, and what the mask design costs you

- **Via the frontend, there is nothing to think about**: `jax.jit` over a
  `UniformTuckerTensorTrain` (or a function taking one) just works — the masks ride as static
  pytree `aux_data`, the supercores are traced children. Because the mask holders hash by
  **value**, rebuilding the frame each optimization step does *not* recompile: identical rank
  structure → same cache key.
- **On raw data**, the backend layout mirrors the class fields:
  ``.data = (tucker_supercore, tt_supercore, shape, (tucker_edge_mask, tt_edge_mask))`` —
  supercores first, then the static `shape` int tuple, then the two rank masks as a sub-tuple
  (supercore-only ops take `.data[:2]`; mask-using ops unpack `.data[3]`).
- **Misuse is guarded, and the error explains the fix**: passing the masks among the *traced*
  arguments (e.g. `jax.jit(ut3_to_dense)(data)`, which traces every leaf) raises a clear
  "uniform masks must be concrete host arrays … close over the masks and trace only the
  supercores" instead of jax's cryptic `ConcretizationTypeError`.
- **The performance story**: under jit the host-numpy masks fold into the compiled program as
  device constants — zero per-call transfer, structure resolved at compile time. Run *eagerly* on
  GPU, the masking multiply moves the KB-scale masks host→device per op (~µs, dominated by eager
  dispatch overhead anyway) — if you care about that, jit, at which point the masks are free.

Why the composition works this way (aux-data vs children, value hashing, why host numpy is
load-bearing): [`contributor/uniform_pytree_composition.md`](contributor/uniform_pytree_composition.md).
