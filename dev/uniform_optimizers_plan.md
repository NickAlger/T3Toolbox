# Optimizers/fitting on the uniform layer — design & plan (the speed payoff)

_Planning (2026-07-06). The next increment after 3b (uniform tangent + probing + jets, all DONE). Goal:
make the geometry-generic Gauss-Newton fitting stack (`fitting.py`, `optimizers.py`) run on **uniform
supercores** — the whole reason the uniform layer exists is speed (`lax.scan` over the `d` core axis
replacing the ragged Python loop). Mirrors the ragged fitting layer; reuses its optimizer algorithms
verbatim. Governing constraint: `docs/uniform_backend_jit_recipe.md` (hold masks fixed, trace only
supercores). Design settled with Nick in the 2026-07-06 session._

_**Progress:** U1 DONE (2026-07-06) — `ubv_corewise_inner` + `GeometryOps.inner` seam; ragged
non-regression byte-identical, uniform-equivalence + garbage-robust + stacking verified, jit-clean.
U2 DONE (2026-07-06) — `backend/uniform_fitting.py` `uniform_{manifold,corewise}_ops` factories (bare
supercore pairs, masks closed over); verified == the frontend geometry `.data` path + mask
loop-invariance across points. Next: U3 (uniform `SamplingKind` builders)._

## The key architectural fact (why this is mostly wiring, not new math)

The four optimizer algorithms (`gradient_descent` / `mc_sgd` / `adam` / `newton_cg`, in
`backend/optimizers.py`) are written **purely against three pluggable seams** and never mention
`TuckerTensorTrain` / ragged cores directly:

- **`GeometryOps(base, project, retract, inner)`** — the per-(layer, geometry) vector-space + chart ops.
- **`SamplingKind(precompute, forward, transpose, sumsq, w_axes, point_forward, n_measurements, take)`** —
  the per-(layer, sampling-op) `𝒥` / `𝒥ᵀ` primitives.
- **`corewise.*`** vector arithmetic (`add` / `scale` / `sub` / `zeros_like` / `map` / `neg`) — structural,
  recurse over tuple structure onto array leaves.

So we do **not** rewrite the optimizers. We supply **uniform implementations of the three seams** and feed
them in. Readiness of each seam (confirmed by recon, 2026-07-06):

- **Geometry** — the raw-tuple twins (`base`/`project`/`retract`/gauge) already exist in
  `backend/ubv_tangent_operations.py`. A uniform `GeometryOps` is a mechanical wrap. **New:** the `inner`
  field (see Decision D2).
- **Sampling** — the `precompute → forward/transpose` primitives in `backend/probing.py` +
  `backend/probe_derivatives.py` are **already `is_uniform`-polymorphic**; they run on uniform data the
  moment they are fed the masked supercore 4-tuple base + packed probe vectors. Only the boundary work
  (mask-once base; pack/unpack probe vectors) must be lifted onto the seam (Decision D4).
- **corewise ops** — the structural ops work verbatim on a supercore pair (a 2-leaf tree); only the
  **reductions** (`dot`/`norm`) are mask-sensitive and become `GeometryOps.inner` (Decision D2).

## Architecture: supercores are the traced state, masks are closed-over (Arch-B1)

Falls directly out of the jit recipe. The optimize loop must trace only the supercores and hold the masks
fixed (a traced mask breaks `int(mask.sum())` and is rejected by the concreteness guard).

```
optimizer state  =  (tucker_supercore, tt_supercore)      ← plain arrays, TRACED
loop-invariant   =  (shape, base_masks)                   ← host numpy, CLOSED OVER
```

At **fixed rank** (fixed-rank fitting) *every* mask in the problem — frame masks, variation masks — is
loop-invariant and derivable from the fixed `base_masks`. So:

- A **uniform `Problem`** captures `(shape, base_masks)` once at construction; its `geom` and `kind` are
  **closures over those masks**, built by a factory (`uniform_manifold_ops(shape, base_masks)`,
  `uniform_corewise_ops(...)`). *(Structural difference from ragged: the ragged `MANIFOLD`/`COREWISE` are
  stateless module singletons; the uniform ones are factory-built to capture masks. Unavoidable, fine.)*
- **Tangents in the optimizer are bare supercore pairs** `(tkv_sc, ttv_sc)`. The geometry/kind closures
  **attach the held masks at the boundary** (inside `project` / `inner` / `forward` / `transpose`) and
  strip them on output — so the optimizer never sees a mask. `corewise.*` structural ops then work verbatim.
- Each `geom.base(supercores)` re-attaches the held masks and calls `ut3_orthogonal_representations`; the
  frame masks it re-derives **inside the trace** constant-fold to device constants → the "1 compile"
  behavior the recipe records. jit closes over the base masks → compiles once; no per-step recompile.

The four optimizer bodies are then reused **verbatim** — the only edit to `backend/optimizers.py` is the
`corewise_dot` → `geom.inner` swap (Decision D2).

## Settled design decisions

- **D1 — Reuse the optimizer bodies (don't fork).** The pluggable seams are the whole point; the bodies
  stay in `backend/optimizers.py` untouched apart from the `inner` swap. ✔ agreed.

- **D2 — A masked + stacked coordinate inner product; `GeometryOps.inner`.** `corewise_dot` sums the *whole*
  supercore including don't-care padding, so it is only correct if the padding is zero. Rather than force
  clean padding by construction (a fragile invariant — the phantom-rank blindness
  `docs/testing_strategy.md` warns of), we build the **honest reduction**: mask both variation supercores,
  then sum over the leading `d` axis + core dims, **keeping the `K+C` stack** (one scalar per stack element).
  - **New backend primitive `ubv_corewise_inner(va_data, vb_data, n_stack)`** — masking
    (`ubv_masking.apply_variations_masks`) + the `d`-leading stack reduction. This is the raw-tuple backend
    twin of the frontend `UT3Tangent.corewise_inner`; the reduction currently sits **inline in the frontend**
    (`uniform_manifold.py:89` `_variations_stack_dot`). We **lift it to the backend and delegate the frontend
    method to it** — one implementation, closing a thin-frontend gap (backend/frontend razor).
  - **`GeometryOps` gains an `inner` field** (the check-free coordinate inner product the frontend
    `Geometry.inner` wraps with its HS/gauge checks). ragged `inner = cw.corewise_dot` (byte-identical);
    uniform `inner = ubv_corewise_inner` closed over the variation masks + the problem's `n_stack`.
  - **Bodies edit:** swap the ~6 `cw.corewise_dot(a, b)` sites → `geom.inner(a, b)` (and thread the inner
    into `_cg_solve`). Norms derive as `inner(x, x) ** 0.5` (the bodies already do this — no separate `norm`
    field needed). **Drop** the earlier COREWISE-project-masking idea entirely; COREWISE.project stays a
    true identity, matching ragged. The fragile clean-padding invariant disappears — reductions are robustly
    masked. ✔ agreed (Nick's redirect; naming `inner` to mirror the frontend).
  - **Semantics note:** `GeometryOps.inner` is the *coordinate* dot for both geometries. On the manifold
    (orthonormal + gauged) it equals Hilbert-Schmidt numerically; on corewise it is the intended Euclidean
    coordinate metric. Both frontend geometries are "checks + coordinate dot" differing only in *which*
    checks, so the check-free backend twin is the same function for both — standard backend-is-check-free.

  - **The `inner` is built stack-ready now** (the `n_stack` parameter), even though increment-1 exercises
    only `n_stack=0`. This is what batched fitting (D3 follow-on) needs — one primitive serves both.

- **D3 — Scope increment-1 to single-tensor fits (`C=()`, `K=()`); defer batched fitting.** The uniform
  speedup for a single tensor is `lax.scan` over `d` (no stacking needed). With `n_stack=0` the masked inner
  collapses to a scalar, **numerically equal to the ragged optimizer** on the same problem — a clean
  equivalence bar. Batched fitting (a real `C` stack, one solve over many tensors) needs per-element step
  lengths (the stack-keeping `inner` with `n_stack=|C|`) and per-element line-search/convergence logic —
  an explicit follow-on, unblocked by the stack-ready `inner`. ✔ agreed.

- **D4 — Expose the uniform sampling split-seam as backend functions.** The uniform `precompute` /
  `forward` / `transpose` are polymorphic, but the boundary work (mask base → 4-tuple; pack/unpack probe
  vectors to width `N`) is fused inside the `ubv_sampling`/`ut3_sampling` wrappers. Per the razor, add thin
  `ubv_sampling` functions exposing the split so a raw-`.data` user can wire their own uniform optimizer
  without re-deriving the packing. Probe vectors are loop-invariant → **pack once** at `Problem`
  construction, not per step. ✔ agreed.

- **D5 — Module placement.** Uniform `SamplingKind` builders in a new **`backend/uniform_fitting.py`** (twin
  of `backend/fitting.py`); the uniform `GeometryOps` factories alongside them (or in
  `ubv_tangent_operations.py`). `backend/optimizers.py` algorithm bodies stay shared. ✔ agreed.

## Slices (mirroring the 3b granularity — each independently reviewable + tested vs a ground truth)

Critical path is **U1–U5** (the layer *works*); **U6** is the payoff (speed); **U7** is polish.

- **U1 — the masked/stacked coordinate inner + the `GeometryOps.inner` seam.**
  - `ubv_corewise_inner(va_data, vb_data, n_stack)` (mask + `d`-leading stack reduction); refactor the
    frontend `_variations_stack_dot` / `UT3Tangent.corewise_inner` to delegate to it.
  - Add the `inner` field to `GeometryOps`; update the two ragged singletons (`MANIFOLD`/`COREWISE`) to
    `inner = cw.corewise_dot`; swap the `corewise_dot` sites in `backend/optimizers.py` to `geom.inner`.
  - **Test:** ragged fitting/optimizer/manifold suite **byte-identical** (grep-all-consumers non-regression);
    `ubv_corewise_inner == UT3Tangent.corewise_inner`; garbage-padded-robust (masked → ignores garbage);
    `== cw.corewise_dot` on a clean single-tensor (`n_stack=0`). Self-contained first slice, 3b-mold.

- **U2 — uniform `GeometryOps` factories** (`base` / `project` / `retract` / `inner`, masks closed over),
  wrapping `ubv_tangent_operations.{orthogonal_gauge_projection, retract, corewise_retract}` + a `base`
  builder + `ubv_corewise_inner`. Bare-supercore-pair in/out convention (masks attached internally).
  - **Test:** backend twin `==` the already-verified frontend `UNIFORM_MANIFOLD` / `UNIFORM_COREWISE`
    `.data` path.

- **U3 — uniform `SamplingKind`** — plain apply/entries/probe, split-seam exposed (D4): mask-once base,
  pack `ww` once, wrap the polymorphic `probing.precompute_*` / `*_from_sweep`.
  - **Test:** uniform-equivalence (`to_ragged` == ragged `SamplingKind`) + adjoint identity
    `⟨r, 𝒥v⟩ = ⟨𝒥ᵀr, v⟩` + mask-strict + garbage-robust.

- **U3′ — jet `SamplingKind`** (apply/entries/probe `_derivatives`) over `probe_derivatives.*`. Same tests.

- **U4 — `LocalModel` / `Problem` wiring + `uniform_least_squares_problem` factory** (captures
  `(shape, base_masks)`, packs `ww`, infers `n_stack` from x0's stack shape = 0 for increment-1).
  - **Test:** `LocalModel.{gradient, gn_quadratic, hvp, objective}` on uniform `==` ragged equivalents;
    gradient gauged (manifold) / raw (corewise).

- **U5 — run all four optimizers eager.** Reuse the bodies. Fit a known tensor from apply/entries/probe
  (+ derivatives); Newton-CG on manifold, adam on corewise, etc.
  - **Test:** `x_opt` **numerically equals** the ragged optimizer's result on the same problem; convergence.

- **U6 — the jit path (the payoff).** Wire `use_jit` for uniform (the mask-closed-over kernel). Prove the
  per-step kernel **compiles once** across iterations (compile-count assertion, per the recipe). Measure
  ragged-vs-uniform wall-clock on a nontrivial fit.
  - **Test:** `test_dispatch`-style jit-clean (a stray `np.*` on a tracer raises) + compile-once + a timing
    sanity check.

- **U7 — frontend surface + docs.** Extend `fitting.py` (`apply_model` &c. accept `UniformTuckerTensorTrain`
  + the uniform geometries) and `optimizers.py` (accept uniform `x0` + geometry), or add uniform factory
  functions. Doctests to the reference-module standard; refresh `dev/HANDOFF.md`; sweep this plan's
  superseded notes to `dev/archive/`.

## Deferred / follow-on

- **Batched fitting** (a real `C` base stack — one solve over many tensors): needs the stack-keeping `inner`
  (`n_stack=|C|`, already built) + per-element step-length / line-search / convergence logic. Unblocked by
  U1's stack-ready primitive; scoped out of increment-1 (D3).
- **Ragged stacked `inner`**: ragged `GeometryOps.inner = cw.corewise_dot` (full collapse) is kept for
  byte-identical non-regression; a ragged `corewise_stack_dot`-based `inner` is only needed if ragged
  batched fitting is ever wanted (unlikely — uniform is the batched path).

## Don't-trip constraints (carried from `dev/HANDOFF.md`)
- A uniform op needs more than dense-vs-ragged — also **exact output masks + garbage-robustness**
  (`docs/testing_strategy.md`). Masks are host numpy (`np`, not `xnp`); supercores are `xnp`.
- Changing a backend convention (here: adding a `GeometryOps` field) has a **wide blast radius — grep ALL
  consumers**; run the full fitting + manifold + optimizer suite, not just the touched tests.
- No automated tool rewrites the code style (esp. the shape comments). No `manifold.py` rename.
