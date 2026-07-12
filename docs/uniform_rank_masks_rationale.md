# Why uniform tangent vectors carry rank masks (and why "just inflate to uniform rank" fails)

> Design rationale. The uniform layer pads every core to common sizes and records each edge's *real* rank
> with a boolean mask. A natural question — since no manifold operation needs *minimal* rank — is whether
> the masks can be dropped, working purely on the padded supercores. They cannot, and the reason is the
> whole point of the layer: **the masks are how variable ranks are controlled during optimization.** This
> note records that decision and the reasoning. Pairs with `uniform_ranks_and_varieties.md` (what the
> object represents), `uniform_masks_vs_ranks.md` (why boolean masks rather than integer ranks),
> `numerical_contract_catalog.md` (the minimal-rank audit), `contributor/uniform_svd_prefix_orthogonalization.md`, and
> `contributor/uniform_pytree_composition.md`.

## What the masks do

A uniform Tucker tensor train pads each edge to a common size `R` — so a whole *stack* of T3s can live in
one array, `lax.scan`-friendly — and marks slots `[0, rank)` real with a prefix mask. The real rank may
differ **per edge and per stack element**: that is the *determinantal variety* — fixed shape, ranks that
vary (`uniform_ranks_and_varieties.md`). In the tangent / optimization layer the masks **zero the
variation in the padded slots**, so a gradient step cannot put content beyond the real rank.

## Why variable ranks matter

Real data rarely wants one rank everywhere. A good T3 fit may need a small rank on one edge and a large
one on another; forcing a single rank either **underfits** (rank too low — accuracy left on the table) or
**overfits** (rank too high — fitting noise, wasting compute). *Rank continuation* exploits exactly this:
start low and grow ranks edge-by-edge under a conditioning criterion
(`docs/rank_continuation.md`, `examples/fit_varied_rank_tensor_newton_cg.py`). All of it presumes you can
hold ranks at chosen, possibly-varied values — i.e. that the optimizer **respects the rank structure**.

## The tempting simplification — and why it fails

The minimal-rank audit (`numerical_contract_catalog.md`, empirically verified against the dense oracle)
found that **no manifold operation requires minimal rank**: `inner`/`norm` need only an orthonormal,
gauged frame; `project`/`transport`/gauge need only orthonormality; `retract` is valid on a non-minimal
frame. Equivalently, for a **given** tangent vector, inflating its ranks to the padded size (orthonormally
completing the frame, zero-extending the variations) leaves every operation's result unchanged:

```
op(inflate(tangent)).to_dense()  ==  op(tangent).to_dense()
```

So why not drop the masks entirely, work on the padded supercores, and treat every slot as real? It would
be a large simplification — no static structure in the jit cache key, so the per-step-recompile problem
would vanish for the tangent layer.

**Because optimization does not operate on a fixed tangent — it computes a new one (the gradient) every
step.** At a rank-deficient frame the projected gradient generally **has content in the padded
("completion") directions** — moving there *increases* the rank. The masks are exactly what zero that
content. Drop them and the gradient grows every edge toward the common padded rank `R`: you lose rank
control and overfit — the precise failure the variable-rank feature exists to prevent.

So the maskless equivalence is genuine, but only for **operations on a fixed tangent**. The masks are not
bookkeeping; they are a **functional rank constraint** that makes variable-rank fitting work. Keeping them
is the deliberate choice.

## The cost of keeping masks, and how it is paid

Masks are static structure, so they ride in jax `aux_data` and become part of the `jit` cache key — which
*risks a recompile* every time the frame is rebuilt (i.e. every optimization step). Two design choices
remove that cost:

- **SVD-based orthogonalization** places the real content in the leading slots, so the masks come out a
  **deterministic prefix** — bit-identical every step at fixed rank
  (`contributor/uniform_svd_prefix_orthogonalization.md`).
- **Value-hashed mask holders** key the `jit` cache on mask *content*, so a rebuilt-but-identical holder
  is a cache hit (`contributor/uniform_pytree_composition.md`).

Together: **no recompile within a fixed-rank stage; a recompile only when the ranks genuinely change**
(e.g. a rank-continuation step) — which is correct, and rare.
