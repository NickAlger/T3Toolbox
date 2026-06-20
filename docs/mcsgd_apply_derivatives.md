# MC-SGD for apply-derivative fitting — prototype notes

*Worklog + findings from prototyping **Manifold Cauchy SGD (MC-SGD)** on the apply-derivative fitting
example (2026-06-19). Status: **prototyped, working, promising; auxiliary heuristics finicky at the toy
scale.** The optimizer lives inline in [`examples/fit_hilbert_from_apply_derivatives.py`](../examples/fit_hilbert_from_apply_derivatives.py);
the forward-looking design notes are in [`geometry_refactor_plan.md`](geometry_refactor_plan.md) §8 (G3).*

## TL;DR

- We replaced the slow full-batch Riemannian Newton-CG in the apply-derivative example with **MC-SGD**
  (T4S paper, Section 5.3.2). It fits the Hilbert tensor from apply-derivative jets **to the noise floor**,
  **~8× faster** than Newton-CG on that demo, and **tuning-free** (the step size is the Cauchy step, not a
  schedule).
- **MC = "Manifold Cauchy", not "Monte-Carlo."** The defining feature is a Cauchy step size from one extra
  Jacobian-vector product; the Monte-Carlo aspect (minibatching) is secondary.
- The **core optimizer is robust**; the finickiness we hit is entirely in the **auxiliary heuristics**
  — minibatch size and the stopping criterion — and is **most likely a small-scale artifact** of the toy
  problem (`N_X = 10` base points). Unproven at scale → future research.
- **Flat `(X, P)`-pair minibatching is far more robust** than the original X-only scheme at this scale: it
  stays near the noise floor across seeds where X-only intermittently fails badly (§3, added 2026-06-19).
  Likely the better default for the library MC-SGD.

## 1. What MC-SGD is (T4S §5.3.2)

A first-order stochastic Riemannian optimizer for fixed-rank least squares `min_X ½‖J(X) − b‖²` on the
Tucker-tensor-train manifold `ℳ_{n,r}`. Per iteration `k`:

1. **Minibatch.** Draw a fresh minibatch `B` of the training samples.
2. **Gauged stochastic gradient.** `g = Π Jᵀr` on `B` — the transpose-Jacobian of the minibatch residual
   `r`, then the gauge projection `Π` (`MANIFOLD.project`). A `T3Tangent` at the current frame.
3. **Cauchy step size.** `α = ‖g‖² / ‖J g‖²` — the **exact 1D minimizer of the local Gauss-Newton
   quadratic model along `−g`** (equivalently, a manifold Gauss-Newton-CG step truncated after one CG
   iteration). Costs **one extra forward Jacobian-vector product** `J g`; **no learning-rate schedule**.
4. **Step + retract.** `X ← retract(−α g)` (the implicit truncated T3-SVD retraction).
5. **Stop** the fixed-rank stage when an exponentially-smoothed loss stops decreasing; then increase rank.

`‖·‖` are Hilbert–Schmidt norms (the metric); since `g` is gauged at an orthonormal frame, `‖g‖²` is just
the coordinate `corewise_inner(g, g)`, and `‖J g‖²` is the measurement-space norm of the forward
`J g`. The guard caps `α` when `‖J g‖ ≈ 0`.

### It is exactly the library's existing hooks

The geometry refactor already exposes the Cauchy step on `fitting.GaussNewtonModel`:

```python
g     = model.gradient                                  # = Π Jᵀr
alpha = g.corewise_inner(g) / model.gn_quadratic(g)     # = ‖g‖² / ‖J g‖²   (one forward J·v)
X     = MANIFOLD.retract((-alpha) * g)
```

`gn_quadratic(p) = ‖J Π p‖²` (added in the safe-mode arc precisely for "cheap Cauchy / line-search step
lengths") **is** the Cauchy denominator. The example **inlines** this against the apply-derivative forward
/ transpose closures, because there is **no derivative `GaussNewtonModel` yet** (an `apply_derivatives_model`
in `fitting.py` would let apply/entries/probe reuse the same loop — a G3 item).

### What is inherited from the example (orthogonal to the optimizer)

Unit-norm probe vectors, **per-order RMS normalization** of the derivative jets (the essential
conditioning step), rank continuation by **increase-by-1** with **zero-padded warm starts**, and
**validation-selected** best rank — all unchanged from the Newton-CG version. Only the optimizer changed.

## 2. What we did

- **Converted** `examples/fit_hilbert_from_apply_derivatives.py` from full-batch Newton-CG to MC-SGD
  (inline prototype). This also removed the slow example that bottlenecked iteration, and removed
  redundancy (the apply-only example already demonstrates Newton-CG).
- **Minibatched over base points `X`** (a few `X`, with **all** their directions `P`) — base points are
  the scarce/expensive resource, directions the cheap one, so this mirrors how data arrives and keeps each
  sample a complete local jet. (The `(N_P, N_X)` sample stack: minibatching = selecting a subset of the
  `N_X` axis.) Unbiased: `(N_X/|B|)·g_B` estimates the full gradient.
- **Head-to-head** against Newton-CG on the same problem (a throwaway comparison, for curiosity).

## 3. What we found

### MC-SGD works on apply-derivatives, and is much faster

Head-to-head on the same Hilbert demo (`SHAPE=(12,12,12,12)`, order 4, `N_X=10` train base points × 30
directions, 1% noise, `SEED=0`):

| method | wall-clock | best level | true error | note |
|---|---|---|---|---|
| **MC-SGD** (minibatch 2 base pts) | **35 s** | 5 | **1.21 %** | sits at the 1% noise floor |
| Newton-CG (full batch) | 4 m 41 s | 4 | 0.61 % | averages a bit below the floor |

So **~8× faster for ~2× the error** — the expected first-order/stochastic-vs-Gauss-Newton tradeoff, plus
tuning-free. Both are essentially "at the noise floor." The final MC-SGD rank-continuation table is clean
and monotone (true error 0.21 → 0.066 → 0.016 → 0.013 → 0.012 with rank; validation correctly picks
level 5).

### Minibatch size — the paper's 10% rule degenerates at small `N_X`

The paper's rule of thumb is `|B| ≈ 10%` of the samples. With only `N_X = 10` base points that rounds to a
**single** base point, whose gradient is too noisy. A sweep (default stopping, `SEED=0`):

| batch size | result |
|---|---|
| **1** (the 10% rule) | single noisy base point → noisy loss → stopping trigger-happy; needed a patience hack |
| **2** | **clean table, ~35 s, the paper's default stopping just works** ✅ |
| **3** | *worse* — a **catastrophic early stop at level 1** (12 iters, true error 0.99) |

So we floored the batch at **2** (`min(N_X, max(2, N_X//10))` — keeps the 10% rule for larger sets,
floors it for small ones). Counter-intuitively a *bigger* batch (3) was worse — see the stopping section.

### Stopping criterion — full-batch signal is right, but the window matters

We stop a fixed-rank stage with the paper's smoothed-loss test, but evaluate it on the **full-batch loss
once per epoch**, not the minibatch loss: a single base point's minibatch loss has too much
base-point-to-base-point variance to detect the plateau (it false-triggers early). One full forward (no
transpose) per epoch is cheap, and the deterministic full-batch loss is a clean signal.

**But the full-batch smoothing alone does not fix batch=3.** Instrumenting batch=3 at level 1 (from a zero
start) showed the *deterministic* full-batch loss **genuinely bounces up early**:

```
epoch  iter   raw_full_loss   smoothed   Δs(lag-3 epochs)
  2     6      4.5300e-01     4.5737e-01
  3     9      4.9159e-01     4.7900e-01    ← real bounce UP (0.453 → 0.492)
  4    12      4.6932e-01     4.7288e-01    +8.0e-03  ← STOP fires (smoothed still above epoch 1)
  ...           (if not stopped, it descends steadily: epoch 24 → true 0.295)
```

The bounce is in the true loss (not minibatch noise), so smoothing can't suppress it. Two things conspire:

1. **The stopping window is epoch-based** (`lag = C_t · n_s/|B|`), so a **bigger batch shrinks the epoch**:
   at batch=3, `iters/epoch = round(10/3) = 3`, so `lag = 3 epochs ≈ 9–12 iterations` — the stop can fire
   by iter 12, **before the method clears its initial transient**. At batch=2, `iters/epoch = 5`, the
   window is 15 iters **and** the batch-2 transient happens to be monotone, so it never trips.
2. **A genuine early non-monotonicity** in the Cauchy iterate (the curvature estimate from a fresh
   minibatch wobbles in the first few steps from a zero start).

**Key point:** batch=3 is **not a worse method** — when *prevented* from stopping early it converges
**faster per epoch** than batch=2. The failure is purely in the **stopping rule**, because the lag is tied
to `n_x/n_x_batch` (epoch size) rather than absolute iterations.

### Flat `(X, P)`-pair minibatches are far more robust than X-only (at this scale)

A natural question — the nested-data minibatch policy (§4) — is whether to keep base points whole. The
reference example draws a few base points `X` with *all* their directions `P`; the alternative is to view
the data as a flat list of `N_P · N_X` individual `(X, P)` **pairs** and draw a minibatch of those **at
random**, mixing freely across base points. Hypothesis (Nick's): the extra mixing decorrelates the
per-sample contributions and lowers the stochastic-gradient variance. The variant lives in
[`examples/fit_hilbert_from_apply_derivatives_flat.py`](../examples/fit_hilbert_from_apply_derivatives_flat.py)
— identical problem and optimizer, only the minibatch draw changes; `apply_derivative_operator` is reused
**verbatim** (it already treats the sample stack `W` as arbitrary leading axes, so flattening
`(N_P, N_X) → (N_P·N_X,)` is a pure reshape that "just works"). We **match the per-step sample count**
(60 pairs = the X-only example's 2 base points × 30 directions), so the epoch length is identical and the
*only* difference is the minibatch composition.

**It is not a marginal win — it fixes the robustness.** Five seeds, validation-picked rank, true relative
error (noise floor 1e-2):

| seed | flat `(X,P)` pairs | X-only |
|---|---|---|
| 0 | 6.99e-03 | 1.21e-02 |
| 1 | 8.66e-03 | 2.04e-02 |
| 2 | 1.97e-02 | **7.52e-01** ⚠ |
| 3 | 1.09e-02 | 1.09e-02 |
| 4 | 1.12e-02 | **1.34e-01** ⚠ |

Flat batching sits near the noise floor on **all five** seeds (worst 1.97e-2); X-only is fine on three but
**fails badly on two** (seed 2 ≈ no fit, seed 4 = bad underfit) — the same early-stop / noisy-gradient
failure modes diagnosed above as the small-`N_X` finickiness. Mechanism: a 60-pair flat minibatch spreads
across ~all 10 base points (each contributing a random subset of its directions), so no step is hostage to
the 2 base points that happened to be drawn; the X-only scheme hands the whole gradient to 2 base points
per step, so its Cauchy curvature estimate **and** its stopping signal both wobble. Flat was occasionally
slower (seed 2: 82 s vs 48 s) — but because it kept productively iterating rather than stopping early at a
bad point.

**Caveat (as everywhere here): `N_X = 10`, 5 seeds — suggestive, not conclusive.** But it points the same
way as the rest of the small-scale story: the finickiness is base-point-clustering variance, and mixing
across base points removes it.

## 4. Open questions / future work

- **Robustness at scale (the central unknown — future research).** Both finicky behaviours trace directly
  to `N_X = 10` being tiny: at scale a minibatch can be small-*fraction* yet large-*absolute* (clean
  gradient), and an epoch is large in absolute iterations (robust stopping window). The paper's MC-SGD is
  robust at scale **on probe fitting** — but whether **apply-derivative** MC-SGD is robust at scale is
  **unproven**. Hunch: it works fine at scale; not established.
- **Make the stopping window absolute-iteration-based** (or add a minimum-iterations guard), not
  epoch-based, so it **decouples from batch size**. This is the concrete fix for the batch=3 failure mode;
  validate it when MC-SGD moves into the library.
- **Promote to `optimizers.py` (G3).** Lift the validated loop into a geometry-agnostic MC-SGD optimizer
  (it needs only `gradient` + `gn_quadratic` + `retract`, which both geometries supply), with the
  no-recompile jit story from the safe-mode arc.
- **Derivative `GaussNewtonModel`.** An `apply_derivatives_model` (and entries/probe variants) in
  `fitting.py` would let the optimizer use `model.gradient` / `model.gn_quadratic` instead of inline
  closures — and `apply` / `entries` / `probe` would get MC-SGD for free.
- **Minibatch policy for nested data — flat pairs win (here).** X-only (all `P` per `X`) matched the cost
  model, but flattening to random `(X, P)` pairs is **substantially more robust** at this scale (§3, the
  5-seed table) and is the recommended default for the library MC-SGD. Still open **at scale**: whether the
  gap persists when `N_X` is large (X-only's clustering variance should shrink on its own), and whether any
  regime wants whole-base-point jets kept together (e.g. correlated per-base-point noise).

## 5. Pointers

- Optimizer + example: [`examples/fit_hilbert_from_apply_derivatives.py`](../examples/fit_hilbert_from_apply_derivatives.py)
  (`manifold_cauchy_sgd`).
- Flat `(X, P)`-pair batching variant (the §3 robustness experiment):
  [`examples/fit_hilbert_from_apply_derivatives_flat.py`](../examples/fit_hilbert_from_apply_derivatives_flat.py).
- Method reference: **T4S paper, Section 5.3.2 "Manifold Cauchy SGD"** (eqs. 26–27); rank continuation
  Section 5.4; the Cauchy step's GN-quadratic identity in the same section.
- G3 design notes / open items: [`geometry_refactor_plan.md`](geometry_refactor_plan.md) §8.
- The Cauchy hooks: `fitting.GaussNewtonModel.gradient` / `.gn_quadratic` (added in the safe-mode arc,
  [`safe_unsafe_mode_plan.md`](safe_unsafe_mode_plan.md)).
