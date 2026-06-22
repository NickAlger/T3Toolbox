# Rank continuation — choosing which ranks to grow

> **If you are fitting a Tucker tensor train and don't know its ranks ahead of time, read this.**
> Rank continuation grows the ranks gradually during optimization — solve at small ranks, grow, re-solve
> warm-started — and stops when more rank stops helping. The question this doc answers is *which* ranks
> to grow: instead of growing every rank together ("uniform"), grow the rank on each edge by an amount
> the **data** tells you it can use, read off the singular values of the current iterate's unfoldings.

Math reference: **Section 5.4** ("Rank continuation") of Alger, Christierson, Chen & Ghattas (2026),
*"Tucker Tensor Train Taylor Series"* (arXiv:2603.21141) — §5.4.1 chooses the new ranks, §5.4.2 is the
warm start. (As elsewhere, the local `t4s.pdf` numbering is newer than the public arXiv.)

Worked, runnable example: [`examples/fit_varied_rank_tensor_newton_cg.py`](../examples/fit_varied_rank_tensor_newton_cg.py).

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
new_tucker_ranks, new_tt_ranks = X.continuation_ranks()        # the ranks to grow to next
```

and you warm-start the next fit by zero-padding the converged cores up to those ranks (§5.4.2) with
`resize` (this does not change the represented tensor):

```python
X0 = X.resize(X.shape, new_tucker_ranks, new_tt_ranks)         # warm start at the grown ranks
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

## Working on raw `.data` tuples (no frontend)

`continuation_ranks` is a thin wrapper: it computes the iterate's T3-SVD and forwards the singular values
to the backend. If you bypass the OO frontend, call the backend directly with singular values you already
have (e.g. from `t3svd`):

```python
import t3toolbox.backend.ranks as ranks
new_tucker, new_tt = ranks.compute_continuation_ranks(shape, tucker_singular_values, tt_singular_values)
```

`ranks.edge_condition_numbers(tucker_singular_values, tt_singular_values)` returns the per-edge condition
numbers themselves, if you want to inspect or report which edges are well conditioned.
