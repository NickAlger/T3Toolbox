# Execution plans

> **Status after the review of 2026-11-14: this document describes the design before the review.** Read [DESIGN_AFTER_REVIEW.md](DESIGN_AFTER_REVIEW.md) first. Where the two disagree, that one wins. This document has not been rewritten.
>
> What no longer holds here:
>
> - **The `Plan` and its fields** `samples`, `assembly`, `path`, `orders`, `sites`, `executor`. The plan is now `SamplePlan(hold, scans, scan_sites, vectorize_sites)`: nested scans over axes of the sample stack, with chunk sizes as array shapes.
> - **The executor and the host loop.** The loop over samples is a scan inside the compiled program.
> - **`orders` and `path`.** There is no option for jet orders or for association order.
> - **"The plan is passed to every call and bound to nothing."** `LocalModel` carries it, with a default.
> - **"The plan needs the drivers."** Scans around today's kernels come first (stage 1).
> - **The policies `fill` and `balance`.** The library's `estimate_chunk_size` and `max_chunk_size_within` are extended instead. `fill` as written sizes for one operation only and underpredicts (findings D9, D10).
> - **`chunk_size` as a field of the sampling kind.** It is removed in stage 2.
> - **The words "block", "level" and "keep".** They are now "chunk", "scan" and "hold".
>
> What still stands: the design constraints, the reasoning on keeping against chunking, and the small-scale measurements with their stated limits.
>
> Known errors, from the reviewers: the per-block kernel is described as returning a core-sized result only, which predates per-sample outputs (D3). `chunk_size='auto'` is resolved in the frontend, not by the backend optimizers (F14). "Sequential map for combines and assemblies; vectorized for lifts" describes the jet kernels only; the plain kernels vectorize combines and assemblies over sites on the uniform layout (F14).

Added 2026-11-14, against T3Toolbox commit `3dc15d9`. Companion to [PROPOSAL.md](PROPOSAL.md), [WORKED_EXAMPLE.md](WORKED_EXAMPLE.md) and [EVALUATOR.md](EVALUATOR.md); see [README.md](README.md) for status and how to resume.

The same rules can be executed in many ways, and the best way depends on problem size, hardware and scale. The proposal is to separate what is computed from how it is executed, so that performance choices are options of a few drivers instead of properties of each kernel.

This document covers how: an immutable `Plan` that sets block sizes, association order and the executor, and the policies that choose one. What is computed, kept and handed back is in [EVALUATOR.md](EVALUATOR.md).

A runnable mockup, `scripts/plan_mockup.py`, reproduces the library's objective, gradient and Gauss-Newton product to about 4e-15 under five combinations of plan and keep set. Its results are in EVALUATOR.md.

Status: the maintainer agreed with the broad shape and decided that the plan is passed to every call and carries no `keep` field. The other details are open; see "Open questions" at the end.

## Design constraints

These come from the maintainer and shape everything below.

- **A framework, not a fixed approach.** The library serves users with different problems and hardware. The goal is to expose performance choices cleanly, not to lock in one set.
- **No decisions from small-scale tests.** The target is very large problems, sharded across many GPUs. Nothing measured on a small machine may set a default or constrain the design.
- **Simple, with as little state as possible.**
- **Numerical operations are tested to tolerance.** Bitwise tests are for structural properties. Rounding-level changes from reordering are allowed as long as the algorithm stays numerically stable.

Two consequences follow directly. Different plans may differ at rounding level, so "every plan gives the same answer to tolerance" is a valid and strong test. And the default plan must reproduce today's hand-made choices, so that nothing is decided by the measurements in this document.

## The decisions

The performance choices reduce to three kinds.

1. **Block size along a batch axis.** Scan, einsum with an extra index, and chunking are one knob: how many slices of an axis are live at once (one, all, or a block).
2. **Association order** within one rule: which two operands are contracted first.
3. **What to keep** between calls, against recomputing it.

| Axis or choice | Today | Alternatives |
| --- | --- | --- |
| Samples `W` | Chunked only in the probe-derivative transpose assembly and the two-hole, and only effective on uniform cores with jax | Block any operation, since each is a map or a map-reduce over `W`. This also works with ragged cores |
| Orders | Scan for products of two full jets; a fused pair for an affine factor | Vectorized |
| Sites, in per-site maps | Sequential map for combines and assemblies; vectorized for lifts | Either, per operation |
| Association | Fixed from the subscript string on numpy; chosen by jax from the sizes on jax | Size-aware, explicit, or measured |
| Kept components | The frame sweep, lean or full, fixed per sampling kind | Nothing, only the inputs, or more |

Sweeps along the chain are sequential by nature and offer no choice of this kind.

Further choices that fit the same framework:

- **Where the sample loop runs**: in the compiled graph, on the host, across devices, across hosts.
- **Reduction order and accumulator precision** when summing partial results over very many blocks.
- **Padding the last block** so one trace serves any number of samples.
- **Running several sweeps in one scan**, such as left and right, or several grades.
- **Float precision.**

## The plan and the drivers

A `Plan` is a frozen, hashable value with a default for every field. It is passed explicitly to each operation. There is no global configuration.

```python
@dataclass(frozen=True)
class Plan:
    samples:  int = None            # outer block; None = all samples at once
    assembly: int = None            # inner block for the gradient assemblies; None = the outer block
    path:     str = 'carry_first'   # association in a sweep step; or 'input_first'
    orders:   str = 'sequential'    # jet products: 'sequential' | 'vectorized'
    sites:    str = 'sequential'    # per-site maps: 'sequential' | 'vectorized'
    executor: str = 'host_loop'     # where the sample loop runs: 'host_loop' | 'scan' | 'shard_map' | ...
```

The plan says how to execute. It does not say what to compute or what to keep. Those are `want` and `have`, arguments of the call, described in [EVALUATOR.md](EVALUATOR.md).

Each option lives in exactly one driver, and the term lists never mention any of them.

| Option | Driver that honours it |
| --- | --- |
| `samples`, `executor` | `over_samples`, the one sample loop, written once as the outermost wrapper |
| `assembly`, `sites` | `sitewise`, the per-site driver |
| `orders` | the jet product kernel |
| `path` | `sweep` and `contract` |

Three structural rules make this scale:

- **The per-block kernel is a pure function** of one block of samples, the cores and the plan. It returns a core-sized partial result and never depends on the total number of samples.
- **The sample loop is an interchangeable executor.** A compiled scan, a host loop streaming from disk, `shard_map` across devices and a multi-host run are the same map-reduce. Scaling strategy is a choice of executor, independent of the kernels.
- **Only core-sized reductions cross devices.** The library's sharding test already asserts this, and every plan must preserve it.

Because a plan is hashable, it can be part of the key of the memoized step-body factories that the scan-body principles require.

## Two levels of sample blocking

Sample blocking is nested at two levels: one outer block for a whole gradient or Hessian product, and an optional finer block inside it for the assemblies. Block sizes per sub-operation beyond that add knobs without adding capability.

The two levels do different jobs:

- **The inner block bounds scratch inside one stage.** The stage's output still exists for all samples.
- **The outer block bounds the lifetime of what stages hand to each other.** The edge variables for a block die when the block ends. Nothing sized by `W` then exists except the data and whatever the plan keeps.

With `B` the outer block and `b` the inner one, peak memory is:

```
peak = kept + data + B * (inter-stage bytes per sample) + b * (assembly scratch per sample)
```

Today's library is `B = W` with `b = chunk_size`.

| | Outer only | Per-stage only (today) | Nested |
| --- | --- | --- | --- |
| Inter-stage arrays for all samples | no | yes; this is today's memory floor | no, when `B < W` |
| Jacobian fused with its transpose | free: composition inside the block | not possible | free |
| Sweeps run at their own best batch | no; they use the assembly's block | yes | yes |
| Chunk logic | one wrapper; kernels stay plain | inside each hungry kernel | one wrapper and one inner loop |
| Memory model to calibrate | one line for the whole operation | one per stage | two numbers |
| Ragged cores, numpy, shards, minibatches | handled uniformly | uniform cores with jax only | handled uniformly |

Why two levels and not a block per sub-operation: stages consume each other's outputs, so block sizes only make sense nested. A stage can subdivide the block it receives only if it reduces over `W`. The assemblies do; the sweeps do not.

The case for the inner level is the maintainer's scenario: the assembly needs a small block for memory while the sweeps would run better on a larger one. Sweeps also have scratch of their own, which is better treated by association order than by a block of their own.

## Keeping against blocking

What can be kept, and by which operation, is in [EVALUATOR.md](EVALUATOR.md). This section is about its cost in memory.

Peak memory is roughly:

```
peak = data + (kept bytes per sample) * W + (transient bytes per sample) * B
```

Blocking shrinks only the last term. With everything kept, that term is contraction scratch, the direction-dependent components, and `J p`, which is as large as the data.

One Gauss-Newton product in the mockup, with `W` = 4096. The totals are kept plus peak during the call, in MB. The 13.8 MB of data is not counted.

| Kept | Kept (MB) | Total, one block | Total, blocks of 512 | Total, blocks of 128 |
| --- | --- | --- | --- | --- |
| `FRAME` | 2.97 | 13.84 | 4.57 | 3.87 |
| `INPUTS` | 0.75 | 13.78 | 2.61 | 1.56 |
| `NOTHING` | 0.00 | 13.78 | 1.95 | 0.85 |

- **With one block, keeping is free.** The components exist for all samples either way.
- **With blocks, the kept arrays are a floor** that no block size can go below.
- **The two pull against each other.** Here the frame sweep costs about 0.8 KB per sample kept, and a product about 2.8 KB per sample of block. Keeping the frame for all samples costs the memory of about 1,100 more samples of block. Where memory forces many small blocks, keeping costs more than a whole block. Where a few large blocks fit, it costs little.
- **How high the floor is depends on the sampling kind.** For probes the kept frame sweep is a fraction of the data, roughly ranks over mode sizes; it is 22% here. For entries and apply the data per sample is small, and the kept `xi` and `mu` can be far larger than it. With streamed data nothing sized by `W` can be kept.
- **Kept components need not live where the blocks run.** They are a per-sample input like the data, so they can sit in host memory and be fed a block at a time, at the cost of transfer. This is not tested.

## Policies

The plan and the drivers are mechanism and are neutral. A policy is a plain function from a problem to a `Plan`, and users can supply their own. An automatic mode is a policy, not a property of the kernels.

| Policy | What it does | Needs |
| --- | --- | --- |
| `current` | Reproduces today's hand-made choices. The default | nothing |
| `balance` | The library's existing rule, generalized: scratch comparable to what is already resident | shapes only |
| `fill(fraction)` | Sizes the blocks to a fraction of a memory budget, spending memory in a stated priority order | a memory measurement on the user's install |
| `measure` | Times the few discrete candidates on one small block and picks the fastest that fits. Opt-in | trial runs |

The model follows FFTW: make a plan, then execute with it. The only state is a memo of measurements keyed on the problem's signature, as `estimate_chunk_size` has today. The chosen plan should be recorded with the run, so a performance result can be reproduced.

What to keep is the caller's decision and an input to the policy. `fill(..., keep=INPUTS)` sizes the blocks for that keep set, and says so when the budget cannot hold it.

### "Choose the batch that fills memory"

As a null hypothesis, this is a good sizing rule and an incomplete policy.

In its favour:

- **It depends only on memory**, a hard constraint with a clean model: peak is affine in the block size.
- **It needs no timing**, so it imports no machine's quirks.
- **Memory can be read from the compiled program without running it**, as `_chunk_slope` already does through XLA's memory analysis.

Against it as the whole answer:

- **It does not say how to spend the memory.** Memory buys a larger outer block, a larger assembly block, vectorized orders, kept components, or a batch of tangents. Filling needs a priority order among them, and writing that order down makes it a policy.
- **"Available" is not a clean number.** jax preallocates most of a GPU by default, other allocations compete, and the chunking documentation notes that one program's scratch differs 5 to 15 times between jaxlib releases.
- **The loss is asymmetric.** A little under the optimum costs a little time; a little over is a crash. So it becomes "fill a fraction".
- **Block size is a static shape.** A size that tracks free memory changes between runs and defeats compile caching unless sizes are quantized.
- **It is silent on the discrete choices**: scan against vectorize, and contraction order.

So sizes come from an explicit memory budget measured on the user's own install, discrete choices come from a preset or the user, and timed selection is opt-in.

The budget rule also becomes necessary at scale. The existing balance rule anchors to the resident jets. Once nothing is kept there is no such anchor, and a per-device budget is the only thing left to size against.

## What the library does today

Each of these choices is currently made once, inside the kernels. Locations are at commit `3dc15d9`.

- **Contraction order.** jax gets one einsum, jitted or not, and chooses a path from the sizes. numpy gets a forced pairwise path, generated by a greedy rule in `_pairwise_path` (`backend/contractions.py`) from the subscript string alone and cached, so it is the same at every size. The split is numpy against jax, not eager against jit.
- **Scans.** Products of two full jets scan over the order. Combines and assemblies map sequentially over sites. The memory benefit exists only on uniform cores with jax, where the loops are real scans; on the ragged path they unroll.
- **Chunking.** Only the probe-derivative transpose assembly and the two-hole are chunked, with `DEFAULT_CHUNK_SIZE = 100` (`backend/sampling_derivatives.py`), and only effectively on uniform cores with jax.
- **Automatic sizing.** The optimizers default to `chunk_size='auto'`, which calls `estimate_chunk_size`. It reads XLA's memory analysis of the compiled chunked transpose at two chunk sizes (`_chunk_slope`) and applies the balance rule. This is the existing precedent for a memory-measured policy. See `docs/chunking.md`.
- **Keeping.** The frame sweep is kept, lean or full depending on the sampling kind. In exact mode only core-sized curvature state is kept; the per-sample residual-grade state is recomputed on every product. See "What can be kept, by operation".
- **Timing.** `newton_cg(timing=True)` records seconds per phase (`PhaseSeconds`, `NewtonTiming` in `backend/optimizers.py`), which a measured policy could read.

## Small-scale measurements, and their limits

These were run on a 2-core CPU with jax 0.12 and numpy 2.5. The numbers below are the run recorded in `scripts/timings_2core_cpu_3dc15d9.txt`. An earlier run of the same scripts differed by up to about 70% in individual cells, and the ranking among the small cases changed between the two runs.

They show that the best choice can go either way, and that this machine cannot rank close alternatives. Both are arguments for a framework. The numbers must not set defaults, and GPU behaviour is untested.

The eta jet combine under jit, with the order axis and the site axis each sequential or vectorized (`scripts/eta_axes.py`). Each cell is time and compiled scratch:

| Size | Both sequential (library) | Order sequential, sites vectorized | Order vectorized, sites sequential | Both vectorized |
| --- | --- | --- | --- | --- |
| d=8, r=8, n=8, order 5, W=256 | 1.6 ms, 0.3 MB | 1.7 ms, 2 MB | 1.5 ms, 1 MB | 1.5 ms, 8 MB |
| d=8, r=32, n=16, order 5, W=4096 | 84 ms, 20 MB | 176 ms, 148 MB | 139 ms, 99 MB | 343 ms, 792 MB |
| d=8, r=96, n=32, order 5, W=1024 | 181 ms, 21 MB | 293 ms, 156 MB | 287 ms, 146 MB | 545 ms, 1170 MB |
| d=24, r=16, n=8, order 2, W=2048 | 24 ms, 2.5 MB | 27 ms, 56 MB | 23 ms, 6 MB | 64 ms, 153 MB |

One sweep step in eager numpy, by which pair is contracted first (`scripts/order_numpy.py`):

| Size | `mu` with `G` first (the library's fixed path) | `G` with `xi` first |
| --- | --- | --- |
| W=4096, r=64, n=8 | 15 ms | 52 ms |
| W=4096, r=32, n=32 | 41 ms | 7 ms |
| W=4096, r=8, n=64 | 12 ms | 0.9 ms |
| W=4096, r=8, n=256 | 93 ms | 3 ms |

A left sweep run in sample blocks (`scripts/sweep_blocks.py`). At the two larger sizes, blocks of about a thousand samples ran within about 25% of all samples at once, in either direction, and blocks of 16 cost 1.4 and 4.5 times as much. At the smallest size, a 4 ms sweep, any blocking cost about 1.5 times as much. These thresholds are specific to the test machine.

## Open questions

- **Priority for spending memory** in the `fill` policy. The current lean is the outer block before keeping.
- **Whether a `measure` policy is wanted at all** at first, given that it makes the plan machine-dependent.
- **Which executors come first**, and how streamed data enters the sample loop.
- **Reduction order and accumulator precision** at very large numbers of blocks.
- **Measurements on target hardware.** `eta_axes.py` and `sweep_blocks.py` need a GPU run before any default beyond `current` is chosen.
- **The numpy contraction path.** Whether to make it size-aware independently of the rest.

Questions about the dependency graph, `want` and `have` are in [EVALUATOR.md](EVALUATOR.md).

## Relation to the graded-messages refactor

The two fit together but are separable.

- **The plan needs the drivers.** With hand-written kernels, each option would have to be threaded through dozens of functions. With `sweep`, `sitewise`, the jet product and one sample loop, each option has one home. This is the practical reason to do the refactor first, or at least its driver layer.
- **Parts can land without the refactor.** A size-aware numpy contraction path and a composed `gn_product` hook on the sampling kind are independent of the naming and jet changes.
- **The plan-invariance test** (every plan agrees to tolerance) is a new kind of test that the refactor makes cheap: one operation, many plans.
