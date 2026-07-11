# Uniform backend jit recipe: hold masks fixed, trace only supercores

> A backend-usage note for the uniform layer. A backend user who bypasses the OO frontend and works on raw
> `.data` tuples needs to jit an optimization loop **without recompiling every step** — which matters
> because accelerating this kind of fitting is the whole point of the uniform layer. This records the recipe
> and why it works. Pairs with `uniform_pytree_composition.md` (the frontend value-hashing story),
> `uniform_svd_prefix_orthogonalization.md`, and `uniform_rank_masks_rationale.md`.

## The constraint

A uniform `.data` tuple carries **masks** (static structure, host numpy). They cannot be jit-*traced*: a
traced mask breaks host-int extraction (`int(mask.sum())`) and is rejected by `require_concrete_masks`. So
under jit the masks must be **static** — closed over, not passed as traced args.

## The recipe (the `optimizers.py` pattern, uniform-adapted)

Jit the **whole per-step kernel once**, closing over the loop-invariant masks; trace only the supercores
(+ minibatch):

```python
shape, masks = frame.shape, frame.masks_data            # LOOP-INVARIANT at fixed rank -- built once
@jax.jit
def step(supercores, minibatch):
    data = (supercores[0], supercores[1], shape, masks)         # masks/shape CLOSED OVER (constants)
    frame_data, var_data = ut3_orthogonal_representations(data) # re-orthogonalize INSIDE the trace
    ...  # apply tangent, gradient, retract
    return new_supercores
for it: supercores = step(supercores, draw())
```

- The frame masks are **closed over** → the kernel compiles once; no recompile.
- `ut3_orthogonal_representations` re-derives the **frame/variation masks inside the trace** from the
  *concrete* frame ranks → host-numpy constants, **constant-folded** into the compiled program. They are
  **not** part of the jit cache key (the key is the closure identity + the supercores' avals), so
  re-deriving them every step is free — it happens once, at trace time — and causes no recompile.
  Empirically: re-orthogonalizing every step with *changing* supercores → **1 compile**.

## Why the masks are safe to hold fixed

At fixed rank the masks **never change**: SVD-based orthogonalization yields a *deterministic prefix*
(`uniform_svd_prefix_orthogonalization.md`), and the uniform retraction (max-rank-only, no numerical
truncation) preserves the padded ranks. So the mask objects are genuinely loop-invariant -- reuse them.

## The anti-pattern (what NOT to do)

Running `ut3_orthogonal_representations` **outside** a jit and passing its (fresh-object) output masks
**into** a separate jit -- either traced (rejected by the guard) or closed over by a **freshly-built
closure each iteration** (a new cache key → recompile *every* step). The fix is this recipe: hold the masks
fixed and reuse them, or jit the whole step so the frame masks stay internal constants.

## Optimizer design rule (for the uniform manifold / optimizer -- 3b and beyond)

The backend optimization functions (MC-SGD, Newton-CG, …) should be **designed** so the masks are
**loop-invariant state, recomputed only at rank-continuation stage boundaries** (where a recompile is
correct and rare), while the per-step jitted kernel is pure supercore work with masks as fixed constants.
The frontend's value-hashed mask holders (`uniform_pytree_composition.md`) give the OO path this
cache-stability automatically; the backend gets it by **object reuse**. (If the finest separation is ever
wanted, `backend/fv_conversions.t3_orthogonal_representations` already returns *just the cores*,
mask-free, so a kernel could orthogonalize supercores inside and attach held masks outside -- but the
bundled `ut3_orthogonal_representations` inside a close-over kernel is already recompile-free, so this is
optional.)
