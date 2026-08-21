# Scan-body sweep — the plan

*Companion to [`scan_body_sweep.md`](scan_body_sweep.md) (the catalogue) and
`docs/contributor/scan_body_principles.md` (the durable principles). Written 2026-08-21 after
workshopping eight conversions with Nick. Ephemeral: archive when the sweep is done.*

**Line numbers drift as edits land — identify a site by its BODY NAME and caller, not by line.**

## Where we are

**TIER 1 IS COMPLETE** (2026-08-21). Every jax-reachable scan/map body in the library is now a
closure-free module-level function, plus the ragged-only ones for consistency. Done in four passes:
eight workshopped with Nick, then three parallel agents partitioned by file.

Measured on the `probe_derivatives` Newton-CG path (uniform+jax, fixed ranks):

| after | compiles / Newton iteration | new mappings / iteration |
|---|---|---|
| start | 19 | 877 |
| the eight workshopped sites | 6 | 504 |
| **tier 1 complete** | **1** | **294** |

The single remaining per-iteration recompile is `optimizers.py:568` — Tier 3. Three inner scans
(`_accumulate`, `_deta_jets_core._step`, `_src_step`) still show as fresh objects but fire **once**,
on the cold trace only: their cached outer bodies never re-run them. That is the designed outcome,
not residual work.

Gates, all green at completion: full suite **726 passed / 41,976 subtests**; doc doctests over 43
pages clean; `sphinx -W` clean. Numerics were verified per agent against the pre-change tree and are
**bit-identical (0.0)** throughout — jointly some 5,600 array values across ragged/uniform ×
numpy/jax, degenerate branches (order-0, `W=()`, `C=()`, multi-axis stacks), both `sum_over_probes`
settings, and the chunked `W > chunk_size` path.

**Verification lesson for future passes:** with concurrent editors, `git stash` is unsafe (one shared
stack, and `HEAD` is the wrong baseline once earlier conversions have landed). Snapshot the package
to a scratch tree and diff against that instead. All three agents reached this independently.

## What the workshop settled

These are decided; they should not be re-litigated per site.

- **Naming**: `_<caller>_step`, e.g. `compute_mu` → `_mu_step`. Where a file already has a
  module-level helper of that name, disambiguate by role (`_sigma_step` was taken, so the sweep
  wrapper became `_sigma_sweep_step`). `probing.py` had eleven bodies called `_func`; hoisting forces
  distinct names, which is an improvement in itself.
- **Category A means literally verbatim.** No `get_backend` line is added to a body that has no
  `xnp` use, purely for cross-site uniformity.
- **Dispatch inside a body** is the house pattern used from within:
  `xnp, _, _ = get_backend(<literal>, tree_contains_jax((<every argument>)))`. Inspect *all*
  arguments — a subset loses the fixed-point property and risks picking numpy while a jax leaf is
  present. The `is_uniform` literal is inert when only `xnp` is taken; ~25 existing call sites
  already pass a literal there.
- **`trs` becomes an extra operand**, not a rebuilt constant: broadcast to `(d,)+trs_r.shape` on the
  uniform path, `(trs_r,)*d` on ragged. This avoids both the per-iteration `binomial_combine_tensor`
  rebuild and the unstated "trs is always the standard binomial tensor" assumption. (Superseded
  later if the planned move away from `trs` in the non-`*_trs` functions lands.)
- **A captured `xscan` (the ragged/uniform flag) becomes two named module-level bodies** over a
  shared core — `_X_step_uniform` / `_X_step_ragged` — selected by the `if` the caller already has.
  Not an `lru_cache` factory: the key would be a lone bool, and two named bodies also give a natural
  seam for future uniform-path performance work (the memory-vs-time tradeoff that moved
  `compute_eta_jets` off a super-einsum in the first place).
- **The memoized-factory pattern of principle 4 stays unused for now.** It appears nowhere in
  `t3toolbox` today, and the house answer to "fresh object, stable value" is value-based identity on
  a frozen dataclass (`SamplingKind`, `ValueHashedMasks`). Reserve the factory for a case that
  genuinely earns it; the strongest remaining candidate is `(n_probe, sum_over_probes)` in the
  chunked assemblers.
- **`jax_map` seam**: a scan-based map with a weak-keyed adapter cache. The adapter must hold `f`
  by **weakref**, not capture it — a `WeakKeyDictionary` holds values strongly, so an adapter that
  captures its own key makes every entry immortal and pins each throwaway body's arrays. Measured:
  the capturing variant leaked all five 8 MB test arrays; the weakref variant released all five.
- **Seam and `xmap` hoist land together.** Neither does anything alone — verified both directions.

## Two mechanism facts that drive sequencing

1. **`lax.map` never caches** (it rebuilds its wrapper lambda every call), so an `xmap` body hoist is
   inert until the seam change is in. `lax.scan` does cache on body identity.
2. **For `scan`, a stable outer body subsumes a fresh inner one** — a cache hit means the outer
   body's Python never runs, so the inner closure is never constructed. This is why several
   category-D *inner* scans cost nothing. It does **not** apply when the outer is a `map`.

## Tier 1 — the mechanical sweep (do first)

Every one of these follows a recipe already proven above. Suggested order is by traffic.

| body / caller | file | cat | recipe |
|---|---|---|---|
| `compute_sigma_jets` `_func` | sampling_derivatives | B | `order`/`s_size` from carry axis 0, `tvec` rebuilt inline, `xnp` by dispatch — the closest analogue to `_mu_jets_step` |
| `t3_apply` `_func` | apply | A | verbatim |
| `_apply_from_xis` `_func` | apply | A | verbatim; wraps `_sigma_step`, needs a distinct name from `probing.py`'s wrapper (different `ys`) |
| `t3_entries` `_func` | entries | B | `xnp` by dispatch; `n_idx = ind.ndim` (**verified** at index stacks `(5,)`, `(5,3)`, `()`) |
| `tt_zipper_left_to_right` `_func` | tt_operations | B | `xnp` only |
| `_ut3svd_shared_supercores` `_tt_step` | ut3_svd | B | sibling of `_ut3svd_step`; identical derivations, distinct body (2 `xs` vs 4) → `_ut3svd_shared_step` |
| `ut3_inner_product` `_push` | ut3_linalg | B | `xnp` only; **keep the `(0,)` int `ys`** — the caller unpacks that arity |
| `utv_oblique_gauge_projection` `_tt_step` | utv_operations | B | `xnp` only; mixed `C` / `K+C` stacks rely on `'...'` broadcasting — do not tidy the einsums |
| `compute_deta_jets` `_func` + `_step` | sampling_derivatives | C + D | exactly the `eta_jets` recipe: two named bodies, `trs_r` as an operand, inner subsumed |
| `_adj_sweep_scanned` `_step` + `_adj_tilde_step`'s `_src_step` | sampling_derivatives | B\* + D | two named bodies; `order`/`s_size` from carry, `svec` rebuilt inline, `trs_r` as operand; inner subsumed. Also **trim `_adj_tilde_step`'s 11-parameter signature** once the derivations move inside — one edit spanning both sites |
| `_apply_derivatives_t3_from_xi_jets` `_func` | sampling_derivatives | B | `s_size` from carry, `trs_push` as operand |
| `_apply_derivatives_from_jets` `_func` | sampling_derivatives | B | `trs_push` as operand; thin wrapper over module-level `_sigma_jet_step` |
| `compute_sigma_hat_jets` `_step` | sampling_derivatives | B | `s_size` from carry, `trs_xi` as operand |
| `compute_mu_jets_trs`, `compute_sigma_jets_trs`, `_adj_sweep` bodies | sampling_derivatives | B | `*_trs` reference forms — public via `__all__` but no in-library caller. Lowest priority of tier 1 |

**Not in tier 1**: the ~25 ragged-path `xmap` bodies. `get_backend` returns `ragged_map` there
regardless of `use_jax`, so no `lax` primitive is ever built and the defect cannot occur. Converting
them is consistency-only. **Open question for Nick: do we do them at all?**

## Tier 2 — `_wchunked_reduce` — **DONE** (2026-08-21)

Split into `_wchunked_sum` (the scan path) and `_wchunked_concat` (the list path, which never had the
defect); `out_w_axis` dropped, `n_probe` and `sum_over_probes` pinned by construction, so
`assemble_one` collapsed into four module-level chunk functions (`_{tucker,tt}_chunk_{summed,kept}`).

**Operands as `xs` turned out to be impossible.** `lax.scan` only scans the LEADING axis, so
operands-as-`xs` needs the chunk axis at position 0 — a `moveaxis` of every operand, which is exactly
the transpose the original design rejected for duplicating the N-large residual. Reshaping is free;
the transpose that must follow it is not. So the operands ride the **carry** instead (loop-invariant
pass-through), and the scan body comes from an `lru_cache` factory keyed on
`(chunk_fn, cs, w_axes)` — the first use of principle 4's memoized factory in the library, at the
site the plan predicted would earn it. `cs` must be a key rather than a capture because it is the
`dynamic_slice` size (necessarily static) *and* user-tunable, so a stable body closing over it would
go stale (principle 5). Body captures are `chunk_fn, cs, jax, w_axes` — all hashable non-arrays.

**Nick's probe-axis flattening landed too**, for its own sake rather than to feed `xs`: `_flatten_w`
merges the probe axes into one super-axis, making `n_probe == 1` an invariant of the chunked path.
That **removed a restriction** — both call sites previously bailed with `n_probe != 1 -> dense()`, so
multi-axis probe stacks got no chunking at all. Only applied on the summed path, where reducing the
stack makes the merge exact; a kept stack would need unflattening, so the concat path keeps the old
bailout.

Results:

| check | result |
|---|---|
| numerics, 32 cases (2 assemblers x 2 probe ranks x numpy/jax x both flags x dense/chunked) | 28 bit-identical; the 4 that differ are the multi-axis summed cases that now chunk instead of falling back to dense — agreeing with dense to ~1e-13 re-association roundoff |
| compiles / mappings over 5 repeat calls of the chunked summed assembly | **0 / 0** (was 2 guaranteed misses per call) |
| peak XLA temp, before vs after | **identical**: 6.75 MiB chunked vs 52.68 MiB dense. Ops-in-carry costs nothing — XLA treats loop-carried invariants like constants, so no padding was ever needed |
| tests | `test_jet_recurrence` + `test_probe_derivatives` + `test_dispatch` + `test_chunk_size_estimator`: 60 passed / 785 subtests |

## Tier 3 — `optimizers.py` `_cg_solve`: the `LocalModel` refactor

Nick has wanted a refactor here for a while; this is the forcing function. **Do after tiers 1-2.**

The captures: `cond` holds `maxiter` and `tol2`; `body` holds `hvp`, `inner`, `xnp`.

**The freshness is currently load-bearing for correctness.** `tol2 = (eta·gnorm)²` is recomputed
every Newton iteration, and a stable `while_loop` body that reads a changed Python value gets the
**cached jaxpr with the old value, silently**. Demonstrated: threshold 10 → 20 → 3 returned
10, 10, 10; carried in the state it returned 10, 20, 3. So a naive hoist here is a wrong-answer bug,
not a missed optimization. `tol2` and `maxiter` must move into the loop state — not optional.

**Rejected, and worth recording as rejected**: an `lru_cache` keyed on `hvp`. It type-checks (bound
methods are hashable) and is actively harmful — `lm.hvp` is a new object on every attribute access
so the cache never hits, and it would pin every `LocalModel`'s frame/sweep/residual arrays for the
process lifetime. That is the leaking variant from the weakref measurement.

**Recompile target: once per fitting run is acceptable** (Nick, 2026-08-21). That is what
`GeometryOps` hashing by closure identity gives for free — stable within a run, different across
runs, which is the correct granularity since the uniform closures capture different masks. Tightening
further would mean giving `GeometryOps` value-based identity, a materially bigger change, and is not
required.

**The prerequisite for every viable option** — defunctionalize, factory-plus-state, or jit
`_cg_solve` whole — is the same: **`LocalModel` as a jax pytree**, arrays as leaves, `geom`/`kind` as
stably-hashing aux. First task of this tier is to prove that feasible.

The house has a working template: `SamplingKind` is a frozen dataclass with an `identity` tuple and
custom `__eq__`/`__hash__` added for exactly this reason ("one recompile per outer step"),
`common.ValueHashedMasks` is the value-based hash mixin, and `UniformGaussNewtonModel` is already a
registered pytree that keeps closures out of `aux_data` and rebuilds its kind lazily.

Only bites at `use_jit=True`, but there it is plausibly the most expensive recompile in the library —
the traced graph contains the whole Hessian apply. `tests/backend/test_optimizers.py` already covers
eager-vs-jit agreement, which gives this tier a real oracle.

## Verification protocol

Per conversion:

1. `co_freevars` of the new body is `()` (or exactly the intended factory key).
2. Numerics against the **pre-change tree**, via `git stash` of the touched files, on both backends
   and any degenerate branch (order-0, empty stack, `W=()`). Bit-identical is the standard met so
   far — accept nothing looser without a reason.
3. Compile/mapping count on a workload that actually reaches the site (see the caveat above).

Harness gotchas already paid for, so nobody pays twice: seed `np.random` before building random
tensors or the two sides differ; `x.to_t3()` returns a tuple for stacked tensors; `T3Variations` is
not iterable; `T3Tangent.corewise_norm()` returns an **array** when the tangent carries a probe
stack; the transposes are classmethod-style, `probe_transpose(ztildes, ww, frame)`.

Batch the full suite (`~6 min`, baseline 726 passed / 41,976 subtests) at the end of a group of
conversions rather than per edit. Docs gates: `python -m doctest` over `docs/*.md` +
`docs/contributor/*.md`, and `sphinx -W` (~10 min).

## Answered by Nick (2026-08-21)

1. **Ragged-only sites: convert them**, for consistency. Added to tier 1 (~25 sites, zero defect
   risk, no new decisions).
2. **The staleness finding is now in `scan_body_principles.md`** as principle 5 (the old principle 5
   became 6). It is a correctness rule: a value that can change between calls must be carried in the
   state, never closed over by a stable body.
3. **`ut3_conversions.py:110`**: bind the lambda to a name.
4. **`optimizers.py:395` — no action.** `_maybe_jit` jits a `step` created once per `mc_sgd`/`adam`
   *call* and reused across all `max_iter` iterations, so it costs one compile per optimizer
   invocation. In rank continuation that coincides with a rank change, where shapes force a
   recompile anyway. Nick's standard: recompiling per gradient step at fixed rank is unacceptable;
   per Newton step is undesirable but acceptable for now absent a straightforward fix; per rank
   change is forced. This is the third case.
5. **`_adj_sweep_scanned` is no longer experimental** — drop "EXPERIMENTAL" from its docstring as
   part of the tier-1 edit that touches it.
