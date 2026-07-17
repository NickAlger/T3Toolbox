# Sharded fitting (postponed slice "D") — design notes

_Postponed 2026-07-16 by Nick (wants to think about the scope more). This captures the design points
raised so the thread can be resumed cold. Follow-on to the `chunk_size` work (slices A/B/C, shipped);
the estimator already takes `n_shards` and `docs/chunking.md` ships the manual `shard_map` recipe._

## The goal

Let a fitting user with a very large minibatch `|W|` (too big for one device even chunked) shard `W`
across devices and get a correct, per-device-chunked Gauss-Newton solve — by establishing a `shard_map`
boundary + `psum` inside the optimizer, so the user passes a `mesh` and it "just works".

## The key finding (grounds everything)

The Newton step's **only** W-reductions are two `SamplingKind` ops:
- **`kind.sumsq`** — the objective `½‖ω⊙r‖²` and the Gauss-Newton quadratic `‖𝒥Πp‖²`.
- **`kind.transpose`** — the gradient `Π𝒥ᵀr` and the GN matvec `Π𝒥ᵀ𝒥Πp`.

Everything else is either a **per-probe map** (`kind.forward`, `kind.point_forward` — output carries `W`,
no cross-probe mixing) or **replicated** state (`x_cores`, the frame `(U,O,P,Q)`, the CG carry, the line
search). So the whole step data-parallelizes cleanly: shard `W`, run the step per device, `psum` those
two reductions → every device computes the identical replicated update. (`LocalModel` in
`backend/optimizers.py` is where `sumsq`/`transpose` are called: `misfit`, `gradient`, `gn_quadratic`,
`hvp`.)

## The boundary

Wrap the per-step kernel (what `_maybe_jit` currently jits in `backend/optimizers.py`) in `shard_map`
over the W mesh axis:
- **`in_specs`**: `sample` / `data` / `residual` → `W`-sharded; `frame` / `x_cores` / CG-carry → replicated.
- **`out_specs`**: replicated (the update).
- **`psum`** at the two reduction points: `sumsq` / `transpose` produce per-shard partials; a `psum`
  over the mesh axis finishes them. With those psummed, the CG `while_loop` + line search run *inside*
  `shard_map` (single-device code per shard) on identical reduced values → a consistent update. The
  per-probe `forward` runs on each device's local `W` shard. Cost: one all-reduce per matvec (the
  standard distributed-CG price). `chunk_size='auto'` inside `shard_map` reads the *local* W → per-shard
  chunking for free (this is exactly why the estimator is eager + takes `n_shards`).

## Genuine decisions (unresolved — Nick to steer)

1. **Scope — `newton_cg` only, or all four optimizers?** `newton_cg` is the workhorse and the
   full-batch case where sharded `W` actually bites; `mc_sgd`/`adam` minibatches are usually small enough
   that single-device chunking suffices. _Leaning: `newton_cg` first._
2. **API — a `mesh=` (+ `w_axis='w'`) parameter** on the optimizer. When given, the step shards over that
   axis; absent → unchanged. _Leaning: this, over a global/context var (explicit; composes with `use_jit`)._
3. **The `psum` plumbing — thread an optional `axis_name` into `sumsq`/`transpose`** (the only two ops
   that must `psum`, and only when a mesh is active), scoped so non-sharded calls stay byte-identical.
   _Leaning: thread it (explicit) over a context var._

## Effort / testing

Genuinely multi-file and design-sensitive (unlike A/B/C it changes the optimizer + `SamplingKind`
contract): a `shard_map` wrapper parallel to `_maybe_jit`, psum-aware `sumsq`/`transpose`, pytree
`in_specs`/`out_specs` derivation, and **2–4-device fake-CPU-mesh tests** (`XLA_FLAGS=
--xla_force_host_platform_device_count=N`) asserting the sharded fit matches the single-device fit and
that the HLO has no operand `all-gather` (only the `psum` all-reduces). The all-gather trap (dynamic
slice on a GSPMD-sharded axis) is why `shard_map`, not plain `jit` on sharded inputs, is the path — see
`docs/chunking.md` "Sharding over W".

## Smaller first step (alternative to the full `mesh=`)

A tested standalone **sharded-transpose building block** (`shard_map` `tv_probe_derivatives_transpose` +
`psum`, on a fake mesh) + the end-to-end recipe in the docs, deferring the optimizer `mesh=` integration.
Lower risk, delivers the memory-critical piece, leaves the optimizer contract untouched.
