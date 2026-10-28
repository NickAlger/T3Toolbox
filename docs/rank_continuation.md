# Rank continuation — choosing which ranks to grow

> **If you are fitting a Tucker tensor train and don't know its ranks ahead of time, read this.**
> Rank continuation grows the ranks gradually during optimization — solve at small ranks, grow, re-solve
> warm-started — and stops when more rank stops helping. The question this doc answers is *which* ranks
> to grow: instead of growing every rank together ("uniform"), grow the rank on each edge by an amount
> the **data** tells you it can use, read off the singular values of the current iterate's unfoldings.

Math reference: **Section 5.4** ("Rank continuation") of Alger, Christierson, Chen & Ghattas (2026),
*"Tucker Tensor Train Taylor Series"* (arXiv:2603.21141) — §5.4.1 chooses the new ranks, §5.4.2 is the
warm start. (As elsewhere, the local `t4s.pdf` numbering is newer than the public arXiv.)

Worked, runnable example: [`examples/fit_varied_rank_tensor_newton_cg.py`](https://github.com/NickAlger/T3Toolbox/blob/main/examples/fit_varied_rank_tensor_newton_cg.py).

---

## Why not just grow every rank together?

A Tucker tensor train has many ranks — one Tucker rank `nᵢ` per mode and one TT bond `rᵢ` between cores.
Different edges often carry very different amounts of information. **Uniform** continuation (grow them
all to the same level `r`) then has to choose between two bad options: stop early and *under-fit* the
edges that need a high rank, or push `r` up and *over-fit* the edges that only needed a low one — wasting
degrees of freedom (hence data) and inviting overfitting. The example makes this concrete: a target with
true ranks `(8,2,5,3)` / bonds `(1,8,8,3,1)` is represented essentially exactly with ~370 degrees of
freedom, but the smallest *uniform* ranks that contain it (`8` everywhere) need ~1216 — 3.3× more.

## The idea (Section 5.4.1)

After each fixed-rank fit, look at the singular values of the current iterate's matrix unfoldings (from
the implicit T3-SVD). For each edge form its **condition number** `κᵢ = σ₁ / σ_lastᵢ` — the ratio of the
largest to the smallest *retained* singular value:

* **small `κᵢ`** → that edge is well conditioned: its rank is being used efficiently and there is room
  for more;
* **large `κᵢ`** → that edge's rank is already "used up" (the last singular value it kept is tiny);
  adding more there would just fit noise.

So **grow only the well-conditioned edges** — those whose `κᵢ` is at least a factor `tau` below the worst
edge's. This brings all edges toward comparable conditioning, spending new ranks where they help. The
proposed ranks are then made non-degenerate ("useless-rank removal", which only uses shape and ranks, no
linear algebra), with a uniform bump as a fallback when every edge is already comparably conditioned (and
to get off the ground from rank 1).

## How to use it

The library provides the **rank-proposal step**; you drive the continuation loop (which optimizer, how to
split validation data, when to stop). The proposal is one method on `TuckerTensorTrain`:

```python
>>> import numpy as np
>>> import t3toolbox.tucker_tensor_train as t3
>>> np.random.seed(0)
>>> i, j, k = np.meshgrid(np.arange(8), np.arange(7), np.arange(6), indexing='ij')
>>> T = 1.0 / (1.0 + 0.05*i + 1.0*j + 8.0*k)     # smooth, very different decay rate per mode
>>> X, _, _ = t3.TuckerTensorTrain.t3svd_dense(T, max_tucker_ranks=3, max_tt_ranks=3)
>>> print(X.tucker_ranks, X.tt_ranks)
(3, 3, 3) (1, 3, 3, 1)
>>> new_tucker_ranks, new_tt_ranks = X.continuation_ranks()        # the ranks to grow to next
>>> print(new_tucker_ranks, new_tt_ranks)      # only mode 1 is well conditioned enough to grow
(3, 4, 3) (1, 3, 3, 1)

```

and you warm-start the next fit by zero-padding the converged cores up to those ranks (§5.4.2) with
`resize` (this does not change the represented tensor):

```python
>>> X0 = X.resize(X.shape, new_tucker_ranks, new_tt_ranks)         # warm start at the grown ranks
>>> print(X0.tucker_ranks, X0.tt_ranks)
(3, 4, 3) (1, 3, 3, 1)
>>> print(bool(np.allclose(X0.to_dense(), X.to_dense())))          # zero padding: the same tensor
True

```

A complete continuation loop — solve, propose, warm-start, repeat — looks like this (see the example for
the full version with a real Newton-CG fit):

```python
import t3toolbox.tucker_tensor_train as t3
import t3toolbox.manifold as t3m

X = t3.TuckerTensorTrain.zeros(shape, (1,) * d, (1,) * (d + 1))   # start from rank 1
records = []
while True:
    X = fixed_rank_fit(X, train_data)                      # your optimizer (e.g. Riemannian Newton-CG)
    records.append((X, validation_error(X)))               # held-out error, for model selection

    new_tucker, new_tt = X.continuation_ranks()            # Section 5.4.1: which ranks to grow
    if (new_tucker, new_tt) == (X.tucker_ranks, X.tt_ranks):
        break                                              # nothing grew -> stop (maximal, or too ill-conditioned)
    if n_train / t3m.manifold_dim((shape, new_tucker, new_tt)) < tau_data:
        break                                              # next model would be underdetermined -> stop
    X = X.resize(shape, new_tucker, new_tt)                # zero-padded warm start

X_best = min(records, key=lambda r: r[1])[0]               # select the level with smallest validation error
```

Two standard outer-loop choices live in *your* loop, not the library: **terminate** when the next model
would become underdetermined (training data over manifold dimension falls below a threshold, `tau_data`
above), and **select** the rank level that minimized a held-out validation error.

> **The inner `newton_cg` fit needs no reference bookkeeping.** Each `fixed_rank_fit` after the first is
> warm-started (zero-padded from the converged lower-rank iterate), so its **initial** gradient is
> already small. `newton_cg`'s stopping tests are *relative*, and on a manifold geometry their default
> reference (`g0_reference=None`, resolved to `'zero'`) is the gradient norm at the **zero tensor of the
> level's own ranks**, not that deflated warm-start gradient. Each level therefore gets a reference at its
> own scale — it grows with the ranks — with nothing to carry between levels. (Before 2026.3.0 the
> reference was the warm start, and the advice was to pin `g0norm_newton` to the first stage's `‖g‖`;
> pass `g0_reference='warm_start'` for that old behaviour.)
>
> The reference is a **convention**, fixed so that it is reproducible: it is evaluated at a canonical
> zero-point frame with seeded Haar-random Tucker factors and identity-prefix TT cores. That makes it a
> deterministic data-gradient scale at the level's ranks, good to an O(1) factor — harmless for a
> relative stop. It is Haar rather than coordinate-aligned because an aligned completion gives exactly 0
> on data with no energy on its leading coordinates. It is identical on the ragged and uniform layers,
> and **nested across levels** (a mode's level-`r` factor is the leading block of its level-`r + 1`
> factor), so the per-level references are consistent along the ladder. Details, and the explicit
> `g0norm_newton` / `g0norm_cg` overrides (which still win), in
> [`fitting_and_optimization.md`](fitting_and_optimization.md) §5. Optionally raise `cg_forcing_power`
> above `0.5` to spend more CG per Newton step when the retraction is expensive.

**The per-level schedule is yours too.** Every Newton knob is a per-call `newton_cg` keyword, so the loop
can give each level its own: `hessian='exact'` (the exact Riemannian Hessian with its trust region,
[`fitting_and_optimization.md`](fitting_and_optimization.md) §4.12), `cg_forcing_power`, `max_newton`,
`gtol_rel`, the reference. The exact mode needs **nothing special at a zero-padded restart**: the restart's
new directions have zero singular values, the lift's clipped pseudo-inverse drops them from the curvature
term (the Hessian is Gauss-Newton there, and the gradient still moves along them), and the trust region
bounds the step — on a measured probe-derivative restart it took 10 steps where Gauss-Newton took 39 (from
a zero-padded restart it can also settle in a different local minimum than Gauss-Newton would, §4.12). A
sketch, with the schedule a function of the level (which schedule is the best default is still an open,
empirical question):

```python
def schedule(level):                                       # your choice, per level
    return dict(hessian='gauss_newton', max_newton=20) if level == 0 else dict(hessian='exact', max_newton=30)

for level in range(max_levels):
    X, stats = optimizers.newton_cg(t3m.MANIFOLD, kind, sample, data, X, **schedule(level))
    ...                                                    # continuation_ranks / resize as above
```

A **cold start** stays available at every level: pass the zero tensor of the level's ranks
(`TuckerTensorTrain.zeros(shape, new_tucker, new_tt)`) as `x0` instead of the `resize` warm start, in
either mode. The default reference does not change — it depends only on the level's ranks, not on `x0` —
so a cold and a warm start at the same level are measured against the same `‖g₀‖`.

`continuation_ranks` returns the **current** ranks unchanged when continuation should stop — either the
structure is already maximal, or every edge is too ill conditioned to grow safely (see `kappa_guard`
below). Test for that as the loop does above.

## Parameters

All are optional with sensible defaults; the only one you are likely to touch is `tau`.

| parameter | what it does |
|---|---|
| `tau` | Sensitivity of the "well-conditioned" test. **A knob you may need to experiment with**: smaller grows more edges per round, larger grows fewer. The default `10.0` is the paper's value; some targets need a smaller one to differentiate edges (the example uses `3.0`). |
| `n_chunk` | How much to grow each chosen edge per round (default `1`). |
| `max_grow` | Cap on how many edges grow per round. `None` (default) grows all eligible edges at once; `max_grow=1` grows **one edge at a time** (the single best-conditioned edge with room) — pair with `tau=1.0` for the most conservative, finest-grained continuation. |
| `kappa_guard` | An absolute conditioning safety net (default `1e12`): never grow an edge this ill-conditioned. It should fire only on genuine near-degeneracy. |
| `rtol`, `atol` | Passed to the internal T3-SVD. By default the current ranks are read from the iterate's structural ranks; set these to continue from the numerical rank at a tolerance instead. |

## Shared Tucker factors

With a sharing partition ([`sharing.md`](sharing.md)), a group's Tucker edges are **one edge**: the
group modes carry one spectrum (`s_g`, from the grouped `t3svd`), contribute one condition number
`κ_g = s_{g,1}/s_{g,n_g}` to the pool (exactly the conditioning of the tied-factor subproblem — the
`√k` scale inflation of `s_g` cancels in every ratio, so group and singleton edges compete fairly),
receive one growth decision applied group-wide, count as **one** `max_grow` candidate, and are
cleaned by the shared useless-rank removal (the group ceiling). The loop is unchanged — thread the
spec through both calls:

```python
>>> ii, jj, kk = np.meshgrid(np.arange(7), np.arange(7), np.arange(6), indexing='ij')
>>> Ts = 1.0 / (1.0 + 0.1*ii + 2.0*jj + 5.0*kk)                # modes 0 and 1 have equal size
>>> Xd, _, _ = t3.TuckerTensorTrain.t3svd_dense(Ts, max_tucker_ranks=3, max_tt_ranks=3)
>>> sharing = (0, 0, 1)
>>> Xs = Xd.share(sharing, max_tucker_ranks=3, max_tt_ranks=3)  # a point with modes 0,1 tied
>>> print(Xs.continuation_ranks())               # unshared: mode 0 is too ill conditioned to grow ...
((3, 4, 4), (1, 3, 4, 1))
>>> new_tucker, new_tt = Xs.continuation_ranks(sharing=sharing)
>>> Xs = Xs.resize(Xs.shape, new_tucker, new_tt, sharing=sharing)   # the group factor padded ONCE, tied warm start
>>> print(new_tucker, new_tt)                    # ... shared: one edge, so modes 0 and 1 grow together
(4, 4, 4) (1, 3, 4, 1)
>>> print(bool(np.all(Xs.has_shared_tucker_factors(sharing))), Xs.data[0][0] is Xs.data[0][1])
True True

```

A freshly padded shared point has exactly-zero new spectrum levels (the tied Tucker channel is
momentarily gated; the escape runs through the TT variations within the first steps), which is why
full shared rank is never enforced anywhere. The reference above matters doubly here: with the old
warm-start reference the fit stalled at the target level and continuation over-grew. The default zero
reference is taken at the zero tensor of the **group** ranks, in the shared geometry (one Haar factor per
group, so the zero point is exactly tied), and needs no pin.

## On the uniform layer

**Rank continuation is a ragged-layer activity, deliberately.** There is no uniform `resize` or
`continuation_ranks`, and that is not an oversight: continuation *changes* the ranks, while a uniform
object's whole point is a fixed padded shape with the masks held loop-invariant. A uniform T3 is also
inherently a stack, and `continuation_ranks` already refuses a stacked ragged point for the same reason
— different stack elements would continue to different ranks.

The shipped pattern is therefore a round trip: keep the cheap bookkeeping on the ragged point, and drop
into the uniform layer only for the fit, one continuation level at a time. (The zero-padded warm start
is numerically rank-deficient, and the uniform frame handles that safely: the sweep's pad-safe SVDs
keep every tangent direction — see `uniform_equivalence_contract.md`, "Gauge-carrying operations".)

```python
x = x0                                        # ragged
for level in range(max_levels):
    ux = UniformTuckerTensorTrain.from_t3(x)              # pack for the fit
    ux, stats = newton_cg(UNIFORM_MANIFOLD, kind, sample, data, ux)   # zero reference at this level's ranks
    x = ux.to_t3()                                        # back to ragged for the bookkeeping
    new_tucker, new_tt = x.continuation_ranks()
    if (new_tucker, new_tt) == (x.tucker_ranks, x.tt_ranks):
        break                                             # nothing left to grow
    x = x.resize(x.shape, new_tucker, new_tt)
```

`examples/fit_hilbert_uniform_newton_cg.py` runs this end to end. Each level's reference is computed on
the uniform layer itself, and equals the ragged one (the canonical zero-point frame has the same real
blocks on both layers), so the round trip does not change where a level stops.

> **Do not feed `ut3svd` singular values to `compute_continuation_ranks`.** It infers the current
> ranks from `len(s[i])`, and uniform spectra are padded to the *nominal* rank — so every length is
> the padded width, the padding zeros drive `kappa_i` to infinity, and the call **type-checks and
> returns silently wrong ranks**. In a measured case it proposed `((2,2,2),(1,2,2,1))` where the truth
> was `((3,3,3),(1,3,3,1))` — a spurious "stop continuing". Convert with `.to_t3()` first, as above.

## Working on raw `.data` tuples (no frontend)

`continuation_ranks` is a thin wrapper: it computes the iterate's T3-SVD and forwards the singular values
to the backend. If you bypass the OO frontend, call the backend directly with singular values you already
have (e.g. from `t3svd`):

```python
>>> import t3toolbox.backend.ranks as ranks
>>> _, tucker_singular_values, tt_singular_values = X.t3svd()      # or singular values you already have
>>> new_tucker, new_tt = ranks.compute_continuation_ranks(X.shape, tucker_singular_values, tt_singular_values)
>>> print((new_tucker, new_tt) == X.continuation_ranks())          # the frontend is exactly this call
True
>>> kappa_tucker, kappa_tt = ranks.edge_condition_numbers(tucker_singular_values, tt_singular_values)
>>> print(len(kappa_tucker), len(kappa_tt))                        # d Tucker edges, d+1 TT bonds
3 4
>>> print(float(kappa_tt[0]), float(kappa_tt[-1]))                 # the length-1 boundary bonds are 1.0
1.0 1.0

```

`ranks.edge_condition_numbers(tucker_singular_values, tt_singular_values)` returns the per-edge condition
numbers themselves, if you want to inspect or report which edges are well conditioned.
