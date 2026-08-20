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
>>> import numpy as np
>>> import t3toolbox.tucker_tensor_train as t3
>>> import t3toolbox.frame_variations_format as bvf
>>> import t3toolbox.manifold as t3m
>>> np.random.seed(0)
>>> shape, tucker_ranks, tt_ranks, order = (12, 12, 12), (6, 6, 6), (1, 6, 6, 1), 2
>>> frame, _ = bvf.t3_orthogonal_representations(t3.TuckerTensorTrain.randn(shape, tucker_ranks, tt_ranks))
>>> ww = [np.random.randn(512, N) for N in shape]              # |W| = 512 probes
>>> pp = [np.random.randn(512, N) for N in shape]              # the perturbation directions
>>> r  = [np.random.randn(order + 1, 512, N) for N in shape]   # residual jets, (order+1)+W+(Ni,)

>>> # same numbers, bounded memory:
>>> JTr_dense   = t3m.T3Tangent.probe_derivatives_transpose(r, ww, pp, frame, order,
...                                                         sum_over_probes=True, chunk_size=None)
>>> JTr_chunked = t3m.T3Tangent.probe_derivatives_transpose(r, ww, pp, frame, order,
...                                                         sum_over_probes=True, chunk_size=128)
>>> float((JTr_chunked - JTr_dense).corewise_norm())   # identical numbers (memory drops on uniform+jax)
0.0

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

**The estimator.** Rather than guess, call `estimate_chunk_size` once (eagerly, outside `jit`) with the
same shapes you used to build the problem, and pass the integer it returns as `chunk_size`. It measures
the assembly's true per-row cost with XLA's own scratch accounting (`memory_analysis`) — no ~20×-off
analytic formula — and returns the largest chunk whose peak stays comparable to the resident jets.
(The measurement lowers a real kernel, so **`estimate_chunk_size` and `max_chunk_size_within` need jax
installed** — as does chunking itself. The value they return is machine-dependent.)

```python
>>> from t3toolbox.backend.sampling_derivatives import estimate_chunk_size, max_chunk_size_within
>>> cs = estimate_chunk_size(
...     mode_shapes=shape,           # ambient dims (as passed to TuckerTensorTrain.randn)
...     tucker_ranks=tucker_ranks,
...     tt_ranks=tt_ranks,           # the d+1 TT bonds
...     order=order,
...     n_probes=512,                # number of probes
...     n_tangent=1,                 # tangent-stack size K (a batch of tangents at one frame)
...     stack_shape=(),              # frame stack C (a batch of base points); () = unstacked
...     n_shards=1,                  # W split across this many devices -> sizes the LOCAL shard
...     dtype=np.float32,            # float64 under x64
... )
>>> bool(isinstance(cs, int) and 1 <= cs <= 512)     # a usable chunk, never larger than |W|
True
>>> JTr = t3m.T3Tangent.probe_derivatives_transpose(r, ww, pp, frame, order,
...                                                 sum_over_probes=True, chunk_size=cs)
>>> float((JTr - JTr_dense).corewise_norm())         # still the exact dense assembly
0.0

```

It sizes the **larger** of the TT-core (`r²` legs) and Tucker (`nO`, `N` legs) gradients, so it stays
safe when `N ≫ n`; and when the whole assembly already fits the balance it returns `chunk_size = |W|`
(the transpose then runs dense, no chunking overhead). The first call compiles (~2 s); results are cached
by shape.

If instead you want to fill a known device — the "use my whole GPU" policy — use
`max_chunk_size_within(..., target_bytes=...)`, which returns the largest `chunk_size` whose assembly
peak stays under an absolute byte cap (e.g. a fraction of device memory):

```python
>>> shapes = dict(mode_shapes=shape, tucker_ranks=tucker_ranks, tt_ranks=tt_ranks, order=order)
>>> tight = max_chunk_size_within(**shapes, n_probes=512, target_bytes=2**20)   # cap the assembly at 1 MiB
>>> loose = max_chunk_size_within(**shapes, n_probes=512, target_bytes=2**24)   # ... at 16 MiB
>>> bool(1 <= tight <= loose <= 512)      # a bigger budget never chunks smaller; both capped by |W|
True

```

**If the frame is a batch of base points, pass `stack_shape`** (the frame stack `C`). Every
edge-variable jet carries `C`, so the assembly costs `prod(C)` times more — while a `target_bytes`
budget does not scale at all. Omitting it on a stacked frame therefore makes
`max_chunk_size_within` return a chunk up to `prod(C)` times too large, which is the
out-of-memory that policy exists to prevent:

```python
>>> free    = max_chunk_size_within(**shapes, n_probes=8192, target_bytes=2**24)
>>> stacked = max_chunk_size_within(**shapes, n_probes=8192, target_bytes=2**24, stack_shape=(4,))
>>> round(free / stacked)              # the same budget buys a 4x smaller chunk on a 4-point batch
4

```

`estimate_chunk_size` is only mildly sensitive to it (the jet floor it balances against scales too,
so the two sides largely cancel — not exactly, because `ww`/`pp` carry no `C`).

For sharded `W`, pass `n_shards` (or read it from the input array's `.sharding` eagerly) so the returned
chunk sizes each device's shard; combine with the `shard_map` recipe below.

```python
>>> bool(estimate_chunk_size(**shapes, n_probes=512, n_shards=4) <= cs)   # sizes the LOCAL shard
True

```

**In fitting you get this for free.** The optimizers (`newton_cg`, `mc_sgd`, `adam`, `gradient_descent`)
and `probe_derivatives_kind` take a `chunk_size` argument defaulting to `'auto'`: for a uniform
`probe_derivatives` fit they call `estimate_chunk_size` once with the shapes read off `x0` (and the
minibatch size for the minibatch optimizers), so a large-`|W|` fit stops OOMing with no action from you.
Pass an `int` or `None` to override; ragged and non-`probe_derivatives` fits ignore it (nothing to chunk).

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
