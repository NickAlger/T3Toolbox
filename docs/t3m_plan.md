# T3M — elementwise multiplication of Tucker tensor trains

## Goal

Add the **elementwise (Hadamard) product** `C = A ⊙ B` of two Tucker tensor trains, with optional rank
truncation, via **three interchangeable algorithms** covering complementary cost regimes, all sharing a
single truncation spec. Plus a small upgrade to `t3svd` (accept a scalar max-rank).

This is the "TTM algorithm for `t3_mult`" item from the `CLAUDE.md` TODO. TTM = the *Tensor Train
Multiplication* algorithm of **Michailidis, Fenton & Kiffner, arXiv:2410.19747** (a separate, well-known
paper — *not* the T4S paper). We generalize it to T3 and, after a complexity analysis (below), pair it
with the cheaper in-place method that is actually the right default for general `d`.

### What exists today

`backend/t3_linalg.py::t3_mult` forms the **full** product and does **no truncation** (its docstring says
so): Tucker factor `W_i = einsum('...io,...jo->...ijo', Bx, By)` (rank `n_x·n_y`), central core
`G_i = einsum('...aib,...ujv->...auijbv', Gx, Gy)` (bond `r_x·r_y`). It is stack-aware. `__mul__` (T3×T3)
calls it. This plan keeps that combination math and adds truncation + two memory-efficient algorithms.

## The math

Write a T3 as `T[x] = Σ_n (∏_i U_i[n_i, x_i]) · C[n]`, with `C` the Tucker core in TT form (`C = TT(G)`).
The product splits into **two truncatable rank families**:

- **Tucker factors (the COPY over `x_i`):** `W_i = U^A_i ⊙ U^B_i` (row-wise Khatri–Rao), rank
  `n^A_i·n^B_i` → truncate to `ñ_i`.
- **Central core-TT (the bonds):** `G̃_i = G^A_i ⊗ G^B_i` (Kronecker), bond `r^A_i·r^B_i` → truncate
  to `r̃_i`.

So a T3 product has **Tucker ranks** *and* **TT bonds** to truncate, both blown up (`n²`, `r²`).

## The three methods

Notation: `d` = #modes, `r` = TT bond, `n` = Tucker rank, `N` = mode dim. Headline scalings (constants
depend on whether `n` or `r` dominates):

| method | algorithm | compute | memory | sweet spot |
|---|---|---|---|---|
| **(a) `form_then_round`** | `t3_mult` (form full product) → `t3svd` round | `O(d·r⁶)` | `O(d·r⁴·n²)` (whole product) | tiny `r`; **parallel** form; reference/oracle |
| **(b) `inplace_fused`** | fused L→R sweep + R→L cleanup; joint per-site truncation | `O(d·r⁴)` | `O(r̃·n²·r²)` (one site) | **large `d`** — the default |
| **(c) `swap`** | concatenate central TTs + swap + contract + truncate | `O(d²·r³)` | `O(r̃²)` | `r ≫ d` (huge bonds, few modes) |

**Crossover** is `d` vs `r`: (b) and (c) both avoid forming the full product (so they beat (a) on memory
everywhere); between them, (b) wins for `d ≳ r` and (c) wins for `r ≫ d`. (a)'s only virtue is that its
forming step is **embarrassingly parallel** (no sweep) — kept for that and as the test oracle.

### (a) `t3m_form_then_round(x, y, ...)`
`t3svd(t3_mult(x, y), max_tucker_ranks, max_tt_ranks, rtol, atol)`. If no truncation is requested,
**return the full product directly** (skip the round) — this is the exact `__mul__` path. Trivial; reuses
`t3_mult` + `t3svd` verbatim. Stack-aware (with max-rank).

### (b) `t3m_inplace_fused(x, y, ...)` — the workhorse
A **fused left-to-right sweep** that builds each product core on the fly and truncates as it goes, never
materializing the `r²`-bond / `n²`-rank product:

- Maintain a carry `R` (the left environment; keeps the `A` and `B` bonds **separate**, in canonical
  form).
- At mode `i`: contract `R` with `G^A_i`, `G^B_i`; form `W_i = U^A_i ⊙ U^B_i`; apply the **joint
  per-site truncation** (Tucker rank `ñ_i` weighted by the canonical environment, then TT bond `r̃_i`)
  — reusing the same per-core SVD logic `t3svd` uses (`down_svd_tt_core` / `left_svd_tt_core`); emit
  the output Tucker factor `Ũ_i` and central core; carry the remainder forward.
- A **right-to-left cleanup sweep** (`truncated_svd`) for quasi-optimal compression.

The `r²` bond appears only transiently at the active site and is truncated immediately. Memory
`O(r̃·n²·r²)` (one site), compute `O(d·r⁴)`. **Joint** truncation (see Decisions). Short-circuits to the
parallel form when no truncation is requested.

### (c) `t3m_swap(x, y, ...)` — the `r ≫ d` method
The T3 generalization of TTM. Concatenate the two central TTs (`A`'s cores, then `B`'s **reversed**), with
each core's Tucker factor riding along on its `n_i` leg. Iterate:

- **swap** (a `truncated_svd` of a contracted core pair; candidate reuse: `left_svd_pair`/`right_svd_pair`)
  to bring same-mode cores adjacent — Tucker factors follow for free;
- **contract** when same-mode cores meet: COPY-merge the Tucker factors (`W_i` + joint SVD truncation)
  and fuse the central cores.

Mixed-canonical gauge maintained via SVD (no QR). `O(d²·r³)` (the `d(d−1)/2` swaps), memory `O(r̃²)`.
Most complex. Same short-circuit when no truncation.

## Settled decisions

1. **Truncation is JOINT, not separate.** The Tucker-rank truncation is **weighted by the central TT**
   (the canonical environment), exactly as `t3svd`'s per-site SVDs do — *not* an unweighted per-mode
   `W_i` SVD. Rationale: (a) rounds with `t3svd` (weighted / ST-HOSVD quality); if (b)/(c) truncated the
   Tucker rank unweighted (HOSVD), they would **compress *worse* than (a)** for the same tolerance,
   which is backwards for the "smart" methods. All three therefore reuse `t3svd`'s per-core truncation
   logic; (b)/(c) only add the on-the-fly product formation around it.

2. **Truncation spec = `t3svd`'s, with one typing upgrade.**
   - `max_tucker_ranks`: `int | Sequence[int]` (len `d`) — **scalar broadcasts** to all modes.
   - `max_tt_ranks`: `int | Sequence[int]` (len `d+1`) — scalar broadcasts (boundary `r_0=r_d=1` stay
     `1` under the cap).
   - `rtol`, `atol`: `float` (scalar only). **Per-step** (total error `≤ √(2d−1)·rtol`).
   - **`rtol`/`atol` require unstacked input** — they raise on stacked (different slices → different
     ranks → ragged), mirroring `truncated_svd`/`t3svd`. **Max-rank truncation is stacking-OK** (uniform
     shapes).
   - *Not* adding `rtol: Sequence[float]`: a single `rtol` feeds both the Tucker and TT SVDs, so a
     sequence has no unambiguous target; the clean fine-grained form would be separate
     `rtol_tucker`/`rtol_tt` (a deliberate, bigger change), and per-mode tolerances are a niche need.
     Deferred.

3. **SVD everywhere — no QR.** For numerical robustness (slower is acceptable). `truncated_svd`
   (full SVD then slice) is the single primitive; calling it with no truncation gives the
   orthogonalization sweeps for free.

4. **`__mul__` (T3×T3) uses method (a) with no truncation** → the **exact** full product, **stack-aware
   by default**. (Plain `*` stays exact and works on stacked T3s.) `t3m()`'s default method is **(b)**.

5. **(b)/(c) short-circuit to the parallel form** (= the exact `t3_mult` product) when no truncation is
   requested, so `t3m(method=...)` with no tolerances is fast and exact regardless of method.

## API

**Frontend** (`tucker_tensor_train.py`):
```python
def t3m(self,
        other,                       # TuckerTensorTrain, same shape & stack_shape
        method='inplace_fused',      # 'form_then_round' | 'inplace_fused' | 'swap'
        max_tucker_ranks=None,       # int | Sequence[int] (len d)
        max_tt_ranks=None,           # int | Sequence[int] (len d+1)
        rtol=None,                   # float; requires unstacked
        atol=None,                   # float; requires unstacked
        ) -> 'TuckerTensorTrain'
```
`__mul__`'s T3×T3 branch → method (a), no truncation (unchanged result vs today; just routed through the
new code path).

**Backend** (in `t3_linalg.py`, beside `t3_mult`): `t3m_form_then_round`, `t3m_inplace_fused`,
`t3m_swap`, each `(x, y, max_tucker_ranks, max_tt_ranks, rtol, atol) -> (tucker_cores, tt_cores)`.

**Shared helper** `normalize_max_ranks(spec, length)` (`None→None`; `int→[int]*length`; `Sequence`→list
with a length check), in a shared module (`ranks.py`), used by `t3svd` *and* the three T3M backends.

## Reuse (no new SVD machinery)
- `truncated_svd` (linalg.py) — the one SVD primitive.
- `t3svd` (backend/t3_svd.py) — the rounder for (a) and the per-site truncation pattern for (b)/(c).
- `down_svd_tt_core` / `left_svd_tt_core` / `right_svd_tt_core` (t3_orthogonalization.py) — per-core
  joint truncation steps.
- `left_svd_pair` / `right_svd_pair` (linalg.py) — candidate for the (c) swap.
- The Khatri–Rao / Kronecker einsums already in `t3_mult`.

## Phases (commit per phase; each verified against the dense oracle)

- **Phase 0 — `t3svd` upgrade. ✅ DONE (`f157adac`).** `ranks.normalize_max_ranks` threaded through
  `t3svd` + `dense_t3svd` + frontend wrappers so a **scalar max-rank** works everywhere. (`rtol`/`atol`
  already worked — verified.) Test: `test_t3svd_scalar_max_ranks`.
- **Phase 1 — scaffold + (a). ✅ DONE (`e64705d7`).** `TuckerTensorTrain.t3m()` frontend + validator
  (`rtol`/`atol` ⊥ stacking, shape/stack checks), `backend.t3_linalg.t3m_form_then_round`, `__mul__`
  routed through (a), and `tests/test_t3m.py` (reusable dense-oracle harness).
- **Phase 2 — (b) `inplace_fused`. ✅ DONE (`f6deae43`).** `t3m_inplace_fused`: right-orthogonalize
  the two central TTs separately (Kronecker is then right-canonical, unformed), single L→R fused sweep
  with separate `(r_x, r_y)` carry, joint per-site truncation. **No cleanup sweep needed** — the right
  side being right-canonical makes each site's truncation optimal. `t3m` default flipped to it.
- **Phase 3 — (c) `swap`. ⬜ TODO.** See `docs/t3m_handoff.md`. The hard part is **gauge-managed
  truncating swaps**.
- **Phase 4 — docs + tests. ⬜ TODO.** `t3m` doc polish + method-selection guidance; finish the
  `CLAUDE.md` `t3_mult` TODO; a cross-method joint-quality test (b/c ≤ a ranks); `test_dispatch` jit
  cases for the t3m methods (max-rank ⇒ static shapes).

> **Status:** Phases 0–2 live and tested (full suite green). (a) and (b) work; (b) is the default. (c)
> and the doc/test polish remain — handoff in `docs/t3m_handoff.md`.

## Testing strategy (oracle = the dense product)

- **No truncation:** all three `.to_dense()` == `A.to_dense() * B.to_dense()` to machine precision, and
  the three agree with each other.
- **With truncation:** `‖method − exact_product‖ ≤` truncation bound (dropped singular-value mass × the
  `√(2d−1)` factor for `rtol`), as in the dense-projection truncation test.
- **Joint-quality check (guards Decision 1):** for the same `rtol`, (b)/(c) must not keep larger ranks
  than (a) — i.e. they compress at least as well as `form_then_round`.
- **Stacking:** stacked + max-rank → per-slice vs the oracle; stacked + rtol/atol → raises.
- **Dispatch (`test_dispatch`):** jit each method with **max-rank** (static shapes); rtol/atol stay
  eager. Plus a numpy==jax smoke check.

## Deferred / out of scope
- `rtol_tucker`/`rtol_tt` (graded tolerances) and a global error budget (`÷√(2d−1)`) — only if a concrete
  need arises.
- Uniform-layer (`ut3_*`) and weighted-layer support.
- A method-auto-selection heuristic (pick (b) vs (c) from `d` vs `r`).
