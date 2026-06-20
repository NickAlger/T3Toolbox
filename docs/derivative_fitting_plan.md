# Fitting from derivative data — plan (through the pilot)

*Design settled with Nick (2026-06-20), branch `geometry-refactor`. This plan covers the infrastructure
for fitting Tucker tensor trains from **derivative** sampling data (`{apply,entries,probe}_derivatives`)
with the existing library optimizers, plus **one pilot example** (apply-derivatives + MC-SGD). It
refines [`optimizers_plan.md`](optimizers_plan.md) §9 (the derivative `SamplingKind`s) and resolves the
"base-jet caching" open thread in [`probe_derivatives_handoff.md`](probe_derivatives_handoff.md) for the
fitting path. **Explicitly out of scope:** the high-level `fit(...)` facade, rank-continuation helper,
and the broader example pass / re-pointing of existing examples (those come after the pilot — see §12).*

## 0. Resume from a fresh context (read this first)

The design is decided; do not re-derive it. The optimizer stack (G3.1–G3.4 of `optimizers_plan.md`) is
done: four backend optimizers (`gradient_descent`/`mc_sgd`/`adam`/`newton_cg`) consuming a `Problem`
oracle built from a `SamplingKind` (operator) × `GeometryOps` (manifold/corewise). Regular probing
already has the **base-sweep reuse** we need (`precompute_base_sweep` + `*_from_sweep`); the derivative
backend computes the *same* sweep jetted but doesn't expose the split. This plan adds that split, builds
derivative `SamplingKind`s on top, makes the oracle layout-agnostic with a user-supplyable minibatch
draw, adds the frontend, and ships the pilot. Build §8 in order.

## 1. Goals & scope

Fit `min_x ½‖S(x) − y‖²` where `S` is a **derivative** sampling operator (the symmetric directional
derivative jets `d^t/ds^t` of apply/entries/probe), reusing the existing optimizers **unchanged**. Three
standing goals (Nick's), used to judge every choice:

1. **Ease of use** for the standard user who just wants a good fit (← the facade closes this; deferred).
2. **Flexibility** for the researcher customizing the fit (custom minibatching, custom normalization,
   roll-your-own optimizer).
3. **Performance** when wanted (jit, on-device minibatching, base-sweep reuse).

**In scope (this plan):** the backend base-sweep-jets split; derivative `SamplingKind`s; the
layout-agnostic `Problem` + minibatch `draw`; frontend derivative model factories + optimizer adapter;
the pilot example + tests.

**Out of scope (deferred, §12):** the `fit(...)` facade, rank-continuation/validation helper,
geometry/optimizer/`x0` defaults; refactoring or adding *other* examples; order-slicing; multi-source
fits; the on-device draw *implementation* (the path is designed and kept open, but the pilot runs the
host/numpy default).

## 2. Decisions (locked this session)

1. **The fit factors into four orthogonal axes** — *kind* (what you measure: `𝒥`, `𝒥ᵀ`, `S`, the
   `‖·‖²` reduction), *geometry* (where you optimize: `base`, gauge `Π`, `retract`), *draw* (how you
   subsample), *optimizer* (how you step). No combinatorial code: 4 optimizers × 2 geometries × 3 kinds
   × any draw.
2. **`SamplingKind` is operator-only**, parameterized for derivatives by `(order, weight)` — a
   *constructor*, not a singleton. The kind owns only the minimal layout needed to build the **default**
   draw; a user-supplied draw needs none of it.
3. **Minibatching = a user-supplied `draw` function**, optional, defaulting to **flat** (a random subset
   across the whole sample stack `W`). `W` stays a grouped collection of axes (no single-axis forcing).
   The user writes `draw` by indexing their own arrays, so slice-on-X / slice-on-P / flat are all just
   different index expressions they control. **Order-slicing is NOT a draw** (order is an output-only
   axis: the forward computes the whole jet jointly, so subsetting orders is output-masking, not
   input-subsetting) — deferred, kept reachable.
4. **Normalization is a per-order residual weight `ω`, owned by the kind** (not folded into the
   forward + pre-normalized data). The objective is `½‖ω ⊙ (S(x) − y)‖²`: `forward` and `point_forward`
   stay **raw**, `ω` enters only `sumsq` and `transpose`. The user passes **raw data + a weight vector**.
   This removes the silent footgun where a custom draw returning raw `data_B` would mismatch a
   forward that divides by the scale. `ω` is **created outside the optimization** (the user's choice of
   normalization — per-order RMS, a physical length scale, etc.; default `ω = 1`).
5. **Base-sweep reuse for derivatives is the mechanical mirror of regular probing's
   `precompute_base_sweep`/`*_from_sweep`** — not a deferred feature. The derivative forwards/transposes
   already compute the base sweep `(xi,mu,nu,eta)_jets` as a cleanly separated block and delegate to
   `*_from_jets` tails; we lift the base block into a `precompute_base_sweep_jets` and add `*_from_sweep`
   wrappers. The derivative kind's `precompute` then returns a *real* sweep, reused across
   `gradient`/`hvp`/`gn_quadratic` exactly like the regular kinds.
6. **Add frontend derivative model factories** (`{apply,entries,probe}_derivatives_model`) — `~3 lines`
   each, since `GaussNewtonModel` is already kind-generic.
7. **jit/on-device:** the `draw` runs *outside* the compiled per-step kernel; the kernel
   `step(cores, sample_B, data_B)` compiles **once** (fixed batch size → constant shapes) and is reused.
   Resident jax data + a jax draw keep the minibatch on-device (only a random key crosses per step);
   numpy data + numpy draw run eager (the existing `use_jit` silent fallback). The pilot uses the host
   numpy default; the device path is documented, not built.

## 3. Architecture & data flow

```
SamplingKind (operator)      GeometryOps (chart)         draw (subsample)      optimizer (step)
  forward   = 𝒥 v              base    = frame(x)          draw(rng) ->          mc_sgd / adam /
  transpose = 𝒥ᵀ (ω²⊙r)        project = Π (gauge)           (sample_B,            newton_cg /
  point_fwd = S(x)  (raw)       retract = chart step          data_B)             gradient_descent
  sumsq     = ½‖ω⊙·‖²
  precompute= base sweep
```

A stochastic step: `sample_B, data_B = draw(rng)` → `lm = problem.local_model(cores, sample_B, data_B)`
(builds `base = geom.base(cores)`, the base sweep once, residual `r = S(cores) − data_B`) →
`g = lm.gradient` (= `Π 𝒥ᵀ(ω²⊙r)`), Cauchy `α = ‖g‖²/lm.gn_quadratic(g)` → `cores = lm.retract(−α g)`.
The full-batch stopping loss uses `problem.objective(cores)` (the stored full `sample`/`data`).

## 4. Backend: the base-sweep-jets split (`backend/probe_derivatives.py`)

Mirror `probing.precompute_base_sweep` / `*_from_sweep`. The base sweep is the jetted twin of the regular
`(xis, mus, nus, etas)` — but **per-kind, not shared** (a divergence from plain probing, discovered in D1):
apply/entries are the all-modes special case whose forward **and** adjoint-state transpose collapse to a
single left pass, so they use only `(xi, mu)` — the right (`nu`) sweep and central (`eta`) combine never
enter the Lagrangian. Only *probe* (one free mode) needs all four. (In plain probing the apply transpose
*does* use `nu`/`eta` — a fuller, non-adjoint-state assembly — hence it shares `precompute_base_sweep`
with probe. The derivative apply transpose is leaner, hence the split.) Verified by NaN-poisoning + a
~12× speedup of the lean precompute over the full one — and it runs **per step** in MC-SGD, so this is a
real saving, not amortized.

```python
def precompute_apply_base_sweep_jets(base, ww, pp, order)    -> (xi_jets, mu_jets)                  # lean
def precompute_entries_base_sweep_jets(base, index, pp, order) -> (xi_jets, mu_jets)                # lean (index-based xi)
def precompute_probe_base_sweep_jets(base, ww, pp, order)    -> (xi_jets, mu_jets, nu_jets, eta_jets)  # full
```

`forward` / `transpose` from-sweep wrappers (consume the sweep, compute only the variation jets, call the
existing `*_from_jets` tails):

```python
apply_jacobian_derivatives_from_sweep(variation, ww, pp, base, sweep, order)            -> z_jets   # (order+1)+W+K+C
probe_jacobian_derivatives_from_sweep(variation, ww, pp, base, sweep, order)            -> z_jets   # per-mode, +(Ni,)
entries_jacobian_derivatives_from_sweep(variation, index, pp, base, sweep, order)       -> z_jets

apply_transpose_derivatives_from_sweep(c, ww, pp, base, sweep, order, sum_over_probes)  -> (dU,dG)
probe_transpose_derivatives_from_sweep(ztildes, ww, pp, base, sweep, order, sum_over_probes)
entries_transpose_derivatives_from_sweep(c, index, pp, base, sweep, order, sum_over_probes)
```

- The base cores `(U,O,P,Q)` are still passed to the from-sweep functions (the variation sweep references
  `Q,O` for sigma, `P,O` for tau, etc.) — exactly as regular `*_from_sweep` takes `base` *and* the sweep.
- `w_jets = build_input_jets(ww, pp)` (the ambient input jets for the `dU` assembly in the transpose) is
  base-independent and cheap; compute it inside the from-sweep transpose (don't bloat the sweep tuple,
  matching the regular four-tuple).
- **Reimplement the existing public `*_tangent_derivatives` / `*_tangent_derivatives_transpose` as thin
  wrappers** = `from_sweep(precompute(...), ...)`. One code path; the existing tests then validate the
  composition for free.

**Verification:** `from_sweep ∘ precompute == the old monolithic function`, bit-identical (~1e-15) across
`W/K/C/order`, both `sum_over_probes`. Re-run `test_probe_derivatives`, `test_contractions`,
`test_dispatch`.

## 5. Derivative `SamplingKind`s (`backend/fitting.py`)

Constructors alongside `APPLY`/`ENTRIES`/`PROBE`:

```python
apply_derivatives_kind(order, weight=None)   -> SamplingKind   # sample = (ww, pp)
entries_derivatives_kind(order, weight=None) -> SamplingKind   # sample = (index, pp)
probe_derivatives_kind(order, weight=None)   -> SamplingKind   # sample = (ww, pp)
```

`weight` `ω` is per order, shape `(order+1,)`, broadcast over `W` (`+ Ni` for probe); `None` → `ω = 1`.
Kind fields:

- `precompute` = the per-kind `precompute_{apply,entries,probe}_base_sweep_jets`, order bound; **a real
  base sweep, reused across `gradient`/`hvp`/`gn_quadratic`** (apply/entries lean `(xi, mu)`, probe full).
- `forward` = `*_jacobian_derivatives_from_sweep` (raw `𝒥 v`); unpacks `sample → (ww, pp)`.
- `transpose` = `*_transpose_derivatives_from_sweep` on `ω² ⊙ residual`, `sum_over_probes=True`.
- `point_forward` = `*_derivatives_t3` (raw `S(x)`).
- `sumsq` = `½‖ω ⊙ out‖²` reduction: weight `out` by `ω` (broadcast over the leading order axis), then
  sum over **order + W (+ Ni)** keeping `C`. The order axis is "just another leading axis" → reuse
  `sumsq_over_samples(ω⊙out, n_w+1)` / `sumsq_over_probes(ω⊙out, n_w+1)`.
- `w_axes` = `n_w` (real count of `W` axes; the order axis is handled inside `sumsq` via `+1`).
- **Default-draw layout helpers** (used *only* to build the flat default; a custom draw ignores them):
  `n_measurements(sample)` (= `prod(W)`) and `gather_flat(sample, data, idx)` (flatten the `W` axes of
  `ww`/`pp`/`index` *and* the `W` axes of `data` — which sit *after* the leading order axis — and gather).

**The weighting math** (settled): objective `f = ½‖ω⊙r‖²`, `r = S(x) − y` (raw); gradient
`Π 𝒥ᵀ(ω²⊙r)`; GN Hessian `Π 𝒥ᵀ diag(ω²) 𝒥 Π`; `gn_quadratic(p) = ‖ω⊙𝒥Πp‖² = sumsq(𝒥Πp)`. So `ω` lives
in `sumsq` (factor `ω`, squared by the reduction) and `transpose` (factor `ω²` on the residual) — nowhere
else.

## 6. The oracle: layout-agnostic `Problem` + `draw` (`backend/optimizers.py`)

Make `Problem` layout-agnostic (this also kills the existing `kind.name == 'entries'` dispatch and
prepares for the uniform layer later):

- `point_forward` moves **onto the kind** (drop the `_POINT_FORWARD` dict).
- `local_model(x_cores, sample=None, data=None)` — `None` ⇒ the stored full `(sample, data)`; a minibatch
  passes explicit `(sample_B, data_B)`. (Replaces the old `sample_idx` + internal `_slice_*`.)
- `objective(x_cores, sample=None, data=None)` — the full-batch loss for the stopping rule.
- `n_samples` / `_slice_sample` / `_slice_data` / `_sample_data` are **removed** from `Problem`.

The **draw**:

- `flat_draw(kind, sample, data, batch) -> draw` — the built-in default; `draw(rng)` does
  `idx = rng.choice(kind.n_measurements(sample), batch, replace=False)` then
  `kind.gather_flat(sample, data, idx)` → `(sample_B, data_B)`.
- A user passes any `draw(rng) -> (sample_B, data_B)` (they index their own arrays; the library needs no
  layout knowledge of theirs). Returns a **fixed-size** batch (so jit compiles once).
- *On-device variant (documented, not built):* a draw that uses `jax.random` on resident jax data,
  producing device sub-arrays; the optimizer threads a per-step key. Host numpy is the pilot default.

## 7. The optimizers (`backend/optimizers.py`)

- `mc_sgd(problem, x0, *, draw=None, batch=..., ...)` and `adam(..., draw=None, batch=..., ...)`: if
  `draw is None`, build `flat_draw(problem.kind, problem.sample, problem.data, batch)`. Each step:
  `sample_B, data_B = draw(rng)` → `lm = problem.local_model(cores, sample_B, data_B)` → step. Stopping
  uses `problem.objective(cores)` (full). (Removes the internal `rng.choice` + slicing.)
- `gradient_descent`, `newton_cg`: unchanged (full-batch; no `draw`). The derivative kind's `precompute`
  gives `newton_cg`'s inner CG the base-sweep reuse automatically.
- `use_jit`: jit the kernel `step(cores, sample_B, data_B)` (minibatch arrays are the traced inputs);
  fires when `x0`/`sample_B`/`data_B` are all jax; fixed batch ⇒ one compile. `draw` stays outside.

## 8. Frontend (`fitting.py` + `optimizers.py`)

- `fitting.py`: `apply_derivatives_model(geometry, x, ww, pp, order, residual, weight=None)` (+ entries/
  probe). Binds the derivative kind, `sample = (ww, pp)` (or `(index, pp)`), raw `residual`. **No
  `GaussNewtonModel` changes** — it already drives any kind through `forward/transpose/sumsq/w_axes/
  precompute`. Add doctests.
- `optimizers.py` adapter: accept the derivative kinds. Likely `mc_sgd(geometry, kind, sample, data, x0,
  *, order=None, weight=None, draw=None, batch=...)` where a derivative `kind` builds the parameterized
  `SamplingKind` from `(order, weight)` and `sample = (ww, pp)`; re-wrap the result `TuckerTensorTrain`.
  (Exact signature finalized in implementation.)

## 9. Pilot example

`examples/fit_hilbert_from_apply_derivatives_topt.py` (name TBD) — the apply-derivatives Hilbert fit via
the **library** `topt.mc_sgd`, mirroring `fit_hilbert_from_apply_derivatives.py` (same problem config,
same per-order RMS normalization now expressed as the residual weight `ω`, same example-level rank
continuation). The optimizer call replaces the inline `manifold_cauchy_sgd`. **Cross-check:** recovers
the tensor to the same ballpark as the inline reference (true error at the noise floor). The existing
inline example is **left unchanged** (refactoring/re-pointing examples is out of scope — §12).

## 10. Testing & verification

- **Backend split:** `from_sweep ∘ precompute == monolithic` (~1e-15, all `W/K/C/order`, both
  `sum_over_probes`); existing derivative tests still green.
- **Oracle == frontend:** derivative `Problem`/`LocalModel` `gradient`/`gn_quadratic` ==
  `fitting.apply_derivatives_model` (the existing oracle==frontend pattern), both geometries.
- **Weighting:** `ω`-weighted gradient matches `jax.grad` / finite-difference of `½‖ω⊙r‖²` at a point.
- **Recovery:** the pilot recovers the Hilbert tensor (true error ≲ noise floor).
- **jit dispatch:** the derivative kind through `topt.mc_sgd` compiles + recovers with jax inputs (no
  hidden numpy); add to `tests/test_dispatch.py`.
- Tests in `tests/backend/test_optimizers.py`, `tests/test_optimizers_frontend.py`,
  `tests/test_probe_derivatives.py` (the split), `tests/test_dispatch.py`.

## 11. Build slices

1. **D1 — backend split. ✅ DONE.** Per-kind base-sweep precompute (apply/entries lean `(xi,mu)`, probe
   full) + the `*_from_sweep` forward/transpose wrappers; the public monolithic functions are thin
   wrappers; verified identical + tests.
2. **D2 — derivative kinds + layout-agnostic oracle. ✅ DONE.** `{apply,entries,probe}_derivatives_kind(
   order, weight)` in `backend/fitting.py` (the `ω` residual weight in `sumsq`/`transpose`; `point_forward`
   + the default-draw `n_measurements`/`take` on the kind). `Problem` is now **layout-agnostic**:
   `point_forward` moved onto the kind, `local_model`/`objective` take an explicit minibatch `(sample,
   data)`, the `kind.name == 'entries'` slicing dispatch is gone. `flat_draw(problem, batch)` builds the
   default draw; `mc_sgd`/`adam` take an optional `draw=None` and feed `(sample_B, data_B)` to the jitted
   step. **No separate `derivative_least_squares_problem` factory** — `least_squares_problem(geom, kind,
   sample, data)` is fully generic now (the kind carries everything). Verified: corewise gradient ==
   `jax.grad` and a finite difference of `½‖ω⊙r‖²`; `gn_quadratic == pᵀHp`; the flat draw flattens a
   multi-axis `W`; `mc_sgd` recovers. Tests: `tests/backend/test_optimizers.py::test_derivative_kinds`.
3. **D3 — frontend (NEXT).** `fitting.py` derivative model factories (`{apply,entries,probe}_derivatives_
   model`, + doctests); `optimizers.py` adapter for the derivative kinds (order/weight + the `(ww,pp)`
   sample + the optional `draw`).
4. **D4 — pilot + tests.** The pilot example; oracle==frontend, weighting, recovery, jit-dispatch tests;
   full suite green.

   *(Detours this session, all committed: the `_grouped_einsum` BLAS-path fix in `contractions.py`
   (11–19× on the derivative forward/transpose); the low-memory K-aware adjoint-state regular apply/
   entries transpose + a latent `rR_d≠1` seed bugfix. None change the D-plan; they speed/repair the
   foundation it sits on.)*

## 12. Deferred (explicitly NOT in this plan)

- **Goal-1 facade** — `fit(kind, sample, data, ranks=...)` picking geometry + optimizer + geometry-aware
  `x0`, with a rank-continuation/validation helper. This is what actually delivers "standard user, no
  fiddling"; build it after the pilot. (Covers the geometry/optimizer/`x0` footguns from the §2/§3
  assessment.)
- **The broader example pass** — `optimizers_plan.md` §10 (re-point apply→`newton_cg`,
  probes→`adam`, apply-derivatives→`mc_sgd`; the build-your-own track; bridges). Not this plan.
- **Order-slicing** (output-masking minibatch; likely an outer continuation loop) and the order/
  polynomial-degree **continuation** ideas — research, later.
- **Multi-source fits / per-sample gradients** (SVRG-style) — reachable at the backend level
  (`sum_over_probes=False`, summing two local models), not packaged.
- **On-device draw implementation** — designed and kept open (§6); the pilot runs host numpy.
