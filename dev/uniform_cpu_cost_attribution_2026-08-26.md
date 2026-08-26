# Where the uniform layer's CPU cost sits — a per-operation attribution

_2026-08-26. Measured against v2026.2.0 (`v2026.2.0-1-gf816b9a4`) in the `t3toolbox` env (py3.11,
numpy 2.x, jax 0.10), CPU only._

`dev/bench_uniform_vs_ragged.py` already establishes the end-to-end wall-clock picture on synthetic
problems, and says plainly that both uniform wins are a GPU story. This note is narrower: it
attributes the CPU gap to individual T3 operations and then to a single contraction, using a real
PDE fitting problem rather than a synthetic one. It records what was tried and what came out; it
proposes nothing.

## Setup

A degree-4 polynomial surrogate of a Darcy parameter-to-observable map (the T3Polynomial project):
`'probe_derivatives'`, shape `(23, 23, 23, 23, 24)`, `order = 3`, 216 probe rows, 100,224 scalar
measurements, per-`(mode, order)` block weights, uniform tucker/tt ranks stepped by rank
continuation. Backends compared are `'ragged'` (numpy) and `'uniform'` (eager, **not** jit) —
`use_jit` is never set, so nothing here is about XLA kernels for the arithmetic.

## 1. Per-operation timings (uniform / ragged ratio)

Matched structure, real data, median of 3 after a warm call:

| rank | probe_derivatives | frame | model build | **gn_hessian** | retract |
|---|---|---|---|---|---|
| 4 | 1.4x | 5.7x | 2.5x | **1.2x** | 5.6x |
| 6 | 1.3x | 6.0x | 2.0x | **1.9x** | 6.4x |
| 8 | 1.7x | 5.1x | 1.7x | **2.3x** | 4.4x |
| 10 | 1.5x | 5.3x | 1.9x | **3.4x** | 4.1x |

Absolute values at rank 10 (ms): `gn_hessian` 128.1 -> 430.7, `retract` 7.5 -> 30.6, `frame`
1.6 -> 8.4, `probe_derivatives` 14.0 -> 20.9, `model` 15.5 -> 29.3. `from_t3` / `to_t3` round-trip
is negligible (~0.7 ms / ~0.1 ms, once per continuation level).

`frame` and `retract` carry large but roughly *constant* ratios and are small in absolute terms.
`gn_hessian` is both the dominant absolute cost and the only ratio that grows with rank. Since CG
calls it once per iteration, it sets the pace of the whole solve.

## 2. It is not CG iteration counts

| rank | cg total ragged / uniform | ms per CG iteration |
|---|---|---|
| 4 | 53 / 68 | 52.1 -> 87.6 |
| 6 | 123 / 107 | 56.9 -> 190.9 |
| 8 | 129 / 117 | 93.7 -> 345.9 |

Uniform does *fewer* CG iterations at ranks 6 and 8. From a realistic continuation warm start
(rank-6 level entered from the converged rank-5 iterate, `max_newton=30`): ragged 25.7 s / 382 CG /
67.4 ms per CG; uniform 73.9 s / 379 CG / 195.0 ms per CG. Passing `val_sample` / `val_data` to
`newton_cg` changes nothing measurable (uniform 69.4 s without vs 72.1 s with).

## 3. The dominant contraction

Instrumenting `contractions.contract` and attributing by call site, over `gn_hessian` x3 at rank 8:

| | share of `contract()` time | calls | per call |
|---|---|---|---|
| uniform `sampling_derivatives.py:1821` | **51 %** (0.444 s of 0.87 s) | 9 | 49 ms |
| ragged `sampling_derivatives.py:1771` | 14.7 % (0.044 s of 0.30 s) | 15 | 2.9 ms |

Both are the `t_det` term of the TT variation-jet assembly — `assemble_tt_variation_jets_trs`
(`is_uniform` branch) and `_tt_variation_jets_trs_core` respectively. Note this is *not*
`tv_probe_derivatives_transpose`, which was the first guess and does not carry the time.

```
ragged   'trs,rWCa,tWKCi,sWCb->KCaib'    shapes [(4,4,4), (4,216,1), (4,216,8), (4,216,8)]
uniform  'trs,drWCa,dtWKCi,dsWCb->dKCaib' shapes [(4,4,4), (5,4,100,8), (5,4,100,8), (5,4,100,8)]
```

The true `tt_ranks` are `(1, 8, 8, 8, 8, 1)`. The ragged operand carries `a`-extent **1** on the
boundary bond; the uniform supercore pads every bond to the uniform max **8** and masks afterwards.
Counting the contracted work per `gn_hessian`:

- ragged: 5 calls (one per mode) x `r*W*a` = 4*216*1 = **4,320 units**
- uniform: 3 calls (one per W chunk) x `d*r*W*a` = 5*4*100*8 = **48,000 units**

Predicted ratio **11.1x**, measured **10.1x** (0.444 / 0.044). Because a TT's boundary bonds are
rank 1 by construction, the padding factor on those modes is exactly `r`, which is consistent with
the `gn_hessian` ratio growing 1.2 -> 1.9 -> 2.3 -> 3.4 across ranks 4/6/8/10 rather than staying
flat. On CPU the `d`-batching also turns several small BLAS-friendly einsums into one large
batched-matmul (`bmm_einsum`) over largely-padding operands: uniform issues ~6x fewer einsum calls,
each ~50x dearer (16 ms vs 0.3 ms at rank 6).

## 4. `chunk_size='auto'` compiles XLA on the numpy path

`optimizers._resolve_chunk_size` -> `sampling_derivatives.estimate_chunk_size` ->
`_assembly_per_row_bytes` -> `_temp`, which is

```python
jax.jit(f).lower(*args).compile().memory_analysis().temp_size_in_bytes
```

Four XLA compilations run per Newton solve purely to read a memory estimate, on a backend whose
arithmetic never uses jit. Cost is roughly flat in rank (0.75 s at rank 1 to 0.97 s at rank 8, ~0.8 s
typical) and is paid again at every continuation level, since the shapes change with the ranks.
`jax_log_compiles` confirms the four `jit(<lambda>)` compilations and their shapes.

Its weight relative to the solve:

| rank | ragged | uniform (`auto`) | uniform (`chunk_size=None`) |
|---|---|---|---|
| 4 | 1.77 s | 4.11 s | **1.88 s** |
| 6 | 3.62 s | 9.36 s | 8.66 s |
| 8 | 5.38 s | 16.06 s | 12.11 s |

At rank 4 the memory probe accounts for essentially the entire ragged/uniform gap. Above that the
padded-contraction cost of section 3 dominates and the probe becomes a minor constant. The resolved
chunk size also falls as ranks grow (216, 216, 216, 192, 158, 129, 87, 76 for ranks 1..10 at
`n_probe = 216`), so the uniform assembly is progressively more fragmented at exactly the ranks
where each call is dearest. The ragged layer ignores `chunk_size` and never enters this path.

## 5. A measurement caveat worth recording

An initial end-to-end continuation run reported a 15x ragged/uniform gap at rank 6 (21 s vs 309 s).
That number does not reproduce: controlled runs from the identical warm start give 2.9x (25.7 s vs
73.9 s), and the level-by-level times in that run were themselves unstable — the same rank-2 level
took 15 s, 25 s and 48 s across three attempts of bit-identical work. The original figure was
contaminated by machine contention. The defensible CPU figures are the controlled ones above:
roughly 2-3x at working ranks, trending toward ~4x by rank 10, essentially all of it in
`gn_hessian`. Validation misfits from those runs are exact and reproducible; only the timings were
affected.

## Scripts

The measurements were driven by throwaway scripts in the T3Polynomial session scratchpad (per-op
timing, CG accounting, `contract` call-site attribution via a wrapper on
`contractions.contract`, and `estimate_chunk_size` scaling). Nothing was added to the T3Toolbox
test suite or benchmark set.
