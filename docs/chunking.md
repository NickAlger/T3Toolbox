# Memory chunking of the derivative transpose (`chunk_size`)

The probe-derivative **transpose** `𝒥ᵀ` (back-projecting residual jets into a variation gradient —
`tv_probe_derivatives_transpose`, `T3Tangent.probe_derivatives_transpose`, their uniform twins, and
the fitting `𝒥ᵀr` step) has one stage whose peak memory can dwarf everything else: the **variation
gradient assembly**. The `chunk_size` parameter bounds that peak. This note explains the memory model,
how to set `chunk_size`, and how it composes with multi-device sharding.

## Why the assembly is the bottleneck

Every stage of the transpose is linear in the sample-stack size `|W|` (the number of probes), but they
fall into two very different per-`W`-row costs:

- **The edge-variable jets** (`mu`, `nu`, `sigma_tilde`, `tau_tilde`, `deta_tilde`, …) — resident
  arrays with per-row cost `~ (order+1)·K·r`. This is the necessary floor: it *is* the data.
- **The assembly intermediate** — the only stage whose per-row cost is **super-linear in the ranks**:
  it forms frame-pairwise products (`mu ⊗ xi`, `xi ⊗ nu`, `mu ⊗ nu`), each `~ (order+1)²·(rank·rank)`.
  That `(order+1)²·r²` factor is what turns a few-GB pipeline into hundreds of GB.

So the assembly is the only thing that runs out of memory, and `chunk_size` is the only knob that
touches it. (The *forward* `𝒥` assembles via a plain lift with no `r²` blow-up, so it needs no
chunking — `chunk_size` is a transpose-only concept.)

## What `chunk_size` does

The assembly's peak is exactly linear in `|W|`, so it is computed in **slices of `chunk_size` probes**
and the partial results are combined — **added** when `W` is summed (`sum_over_probes=True`, the
Gauss–Newton gradient) or **concatenated** when `W` is kept (`sum_over_probes=False`). The peak drops
from `|W|·(per-row)` to `chunk_size·(per-row)`, and the result is bit-for-bit the dense assembly (an
exact reorganization, not an approximation).

```python
# same numbers, bounded memory:
JTr_dense   = T3Tangent.probe_derivatives_transpose(r, ww, pp, frame, order, chunk_size=None)
JTr_chunked = T3Tangent.probe_derivatives_transpose(r, ww, pp, frame, order, chunk_size=512)
# JTr_chunked == JTr_dense, but the chunked run peaks at ~512/|W| of the memory
```

Semantics:

- **`chunk_size = None`** (or any value `≥ |W|`) → the dense assembly, no chunking.
- **`chunk_size = <int>`** → chunk into slices of that many probes. The default is a small, safe fixed
  value.
- Chunking engages **only on the uniform + JAX path** (a stacked supercore under `jit`), because only
  there does the sequential `lax.scan`/`lax.map` actually force the intermediates to be freed one slice
  at a time. On the ragged path or with NumPy, arrays are freed eagerly and the call falls back to the
  dense assembly regardless of `chunk_size`.

### A fixed `chunk_size` bounds the chunk *count*, not the bytes

Peak memory is `chunk_size · (per-row bytes)`, and the per-row bytes grow like `(order+1)²·r²` (TT
gradient) or `(order+1)·nU·N` (Tucker gradient, the term that dominates when the ambient dimension
`N ≫ n`). So a single fixed default is only "safe for moderate problems": comfortable at `r = 128`,
but heavier at large ranks or large `N`. To actually **bound the memory**, choose `chunk_size` from the
problem shapes (next section).

## Choosing `chunk_size`

The recommended policy is **balance**: pick `chunk_size` so the assembly's peak is comparable to the
edge-variable memory that is *already* necessarily resident. Then the assembly is never the tallest
pole — if the rest of the pipeline fits, so does the assembly (total peak `≈ 2×` the necessary floor).
This is device-agnostic: it needs no knowledge of the device's memory, only the problem shapes.

**The estimator.** A helper that computes a memory-balanced `chunk_size` from the problem shapes
(`d`, `order`, ranks, `|W|`, `dtype`) — measuring the assembly's true per-row cost rather than guessing
it — is provided as a separate, eager (outside-`jit`) function. Call it once and pass the integer it
returns as `chunk_size`. See the estimator's own documentation for the exact signature and the optional
"largest chunk that fits an absolute byte budget" policy.

Until then, a practical manual rule: the assembly's per-row cost is roughly `(order+1)²·r²·8` bytes (a
few MB per probe at `r = 128`, order 5); pick `chunk_size ≈ (your per-device memory budget) / (that)`.

## Sharding over `W`

For very large `|W|` the sample stack is often **sharded across devices**. Chunking and sharding
compose — but *only* through `shard_map`, and the reason is worth understanding.

`chunk_size` chunks **within a device**; sharding splits **across devices**. The transpose is a
map-reduce over `W` (independent per-probe work, then a sum), which is exactly the `shard_map` shape:

Only the `W`-carrying arrays go through `shard_map`; the replicated frame and the scalar `order` are
closed over (a scalar cannot take a `PartitionSpec`):

```python
import jax
from functools import partial
from jax.sharding import PartitionSpec as P

@partial(jax.shard_map, mesh=mesh,
         in_specs=(P('w', None),   # ztildes: W sharded
                   P('w', None),   # ww:      W sharded
                   P('w', None)),  # pp:      W sharded
         out_specs=P())            # summed gradient: replicated
def jtr(ztildes, ww, pp):
    dU, dG = tv_probe_derivatives_transpose(          # frame (replicated), order: closed over
        ztildes, ww, pp, frame, order, sum_over_probes=True)   # chunk_size chunks the LOCAL shard
    return jax.tree.map(lambda a: jax.lax.psum(a, 'w'), (dU, dG))   # finish the W-sum across shards
```

Inside `shard_map` the body sees each device's **local** shard as an ordinary array, so `chunk_size`
chunks the local `W` with no cross-device traffic, and the `add`-reducer for `sum_over_probes=True`
produces a per-shard partial that a single cheap `psum` finishes. (For `sum_over_probes=False` the
output keeps `W`; use `out_specs=P('w', ...)` and drop the `psum`.) You only need to know **which
inputs carry `W`** — the residual jets, `ww`, `pp`; the frame does not.

### Pitfall: do not feed a globally-sharded `W` array to a plain `jit`

If instead you shard `W` with the automatic partitioner and call the transpose under a plain `jit`
(no `shard_map`), the per-chunk **dynamic** slice on the sharded axis forces XLA to **all-gather** the
whole `W` axis onto every device — silently undoing the sharding (per-device memory jumps back to the
*full* `|W|`) and adding communication. This is because the chunk loop's slice offset is a runtime
value, so the partitioner cannot prove which shard each window hits and conservatively replicates. Use
`shard_map` (above) to shard `W`; it is the only way to get per-shard chunking without the all-gather.
