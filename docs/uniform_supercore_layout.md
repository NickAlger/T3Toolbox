# Why the core index leads the supercore (outside the stack)

> Why a uniform supercore is laid out as `(d,) + stack_shape + (per-core axes)` — the mode/core
> index `d` first, the stack *after* it — even though that departs from the stack-leading
> convention used everywhere else in the library. If you index supercores directly, this is the
> axis order you are working with.

---

## The layout

A uniform supercore stacks the `d` cores of a Tucker tensor train onto a single leading array axis:

```
tucker_supercore : (d,) + stack_shape + (n, N)
tt_supercore     : (d,) + stack_shape + (r, n, r)
```

So axis `0` is the **core/mode index**, the batch (`stack_shape`) sits at axes `1 .. len(stack_shape)`,
and the per-core tensor axes come last. This is a deliberate departure from the rest of T3Toolbox,
where `stack_shape` is the *leading* prefix of every core (the "frame-inner" convention in
`docs/batching_and_stacking.md`). Inside a uniform object the stack is *not* leading; `d` is.

## Why `d` leads

Three reasons, in roughly descending importance:

1. **Sweep operations compile to real scans.** Much of T3 algebra is a left-to-right sweep over the
   cores (the SVD sweep, `entries`, the orthogonalization passes). Under JAX, `jax.lax.scan` scans
   **axis 0** of its inputs. Putting the core index first means a sweep over cores *is* a scan over
   axis 0 — it compiles to a single rolled loop instead of unrolling `d` Python iterations. If the
   stack led and `d` were buried in the middle, every sweep would need a `moveaxis` (data movement) to
   expose `d`, or could not use `lax.scan` at all. This is the decisive reason, and it is specific to
   the uniform layer (the ragged layer sweeps over a Python tuple, so the question never arises there).

2. **Per-core locality.** With `d` first, each core occupies a contiguous block of memory. A sweep
   touches one block at a time, which keeps the per-core work local — friendlier to caches and to GPU
   memory access than a layout that interleaves the core index among the batch/tensor axes.

3. **It is what stacking naturally produces.** `xnp.stack(cores)` puts the stacked axis first, so
   `d`-leading is the zero-friction result of building a supercore from a tuple of cores.

A complementary point: several uniform ops avoid the sweep entirely by folding `d` into an einsum as
just another index (the vectorization win that motivates the uniform layer — e.g. batched SVD over all
cores at once). For those, `d`'s *position* is irrelevant to correctness; keeping it as a named leading
axis simply keeps the einsum ops and the scan ops on the same convention.

## The cost to keep in mind

**Two conventions in one library.** Ragged objects are stack-leading; uniform objects are
`d`-leading. Anyone reasoning across both must remember that, inside a uniform object, the stack axes
are offset by one (they start at axis 1, not 0). This is a real cognitive load and a plausible source
of axis bugs — the same family of mistakes catalogued in `docs/batching_and_stacking.md`. The
stacking helpers already bake in the offset (`ut3_stack`/`unstack` use `axes = range(1, 1+len(stack))`),
which contains the damage but does not erase the asymmetry.

(The implementation-side consequences — the polymorphic-backend straddle, and the rejected
stack-leading alternative — are recorded in
[`contributor/uniform_internals.md`](contributor/uniform_internals.md).)

See also `docs/uniform_masks_vs_ranks.md` (why the rank metadata is stored as boolean masks) and
`docs/uniform_ranks_and_varieties.md` (what a stacked uniform T3 represents).
