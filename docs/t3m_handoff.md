# T3M handoff — resuming at Phase 3 (swap) + Phase 4 (docs/tests)

> **✅ COMPLETED & SUPERSEDED (commit `022216a5` + slices 2–4).** All of T3M is done: methods
> (a)/(b)/(c) live and tested. Method (c) was built per **`docs/t3m_swap_plan.md`** (the concrete plan
> that updated the sketch below — explicit-center gauge + `oversample`/`t3svd`-cleanup for the Tucker
> leaf-frame tension), with the theory in **`docs/ttm_t3m_ht_note.tex`**. This file is kept for history;
> the Phase-3 sketch below is the *pre-implementation* idea and was refined by `t3m_swap_plan.md`.

Read `docs/t3m_plan.md` first (the full design + all settled decisions). This file is the concrete
"how to resume" note. Phases 0–2 are done, committed, and tested; (c) and the doc/test polish remain.

## State (commits)

- `f157adac` Phase 0 — `t3svd` scalar max-rank (`ranks.normalize_max_ranks`).
- `e64705d7` Phase 1 — `t3m()` scaffold + method (a) + `tests/test_t3m.py` harness.
- `f6deae43` Phase 2 — method (b) `inplace_fused` (the default).

Full suite green. `*` (T3×T3) → method (a) exact. `t3m()` default → (b). `'swap'` currently raises
`NotImplementedError`.

## Where the code lives

- Backend methods: `t3toolbox/backend/t3_linalg.py` — `t3m_form_then_round`, `t3m_inplace_fused`
  (study this one; (c) reuses the same primitives). Each is `(x, y, max_tucker_ranks, max_tt_ranks,
  rtol, atol) -> (tucker_cores, tt_cores)`; `x = (tucker_cores, tt_cores)`.
- Frontend dispatcher: `TuckerTensorTrain.t3m` in `tucker_tensor_train.py` — the `backend = {...}.get(method)`
  dict (add `'swap': ragged_linalg.t3m_swap`) and the validator (shape/stack + rtol/atol⊥stacking,
  already done).
- Tests: `tests/test_t3m.py` — reusable harness: `check_exact`, `check_sweep_exact` (generous ranks
  exercise the truncating path), `check_truncated` (≤ 2× the dense-`t3svd` reference). **To wire up
  (c): add `t3m_swap` to `__all__` + the dispatcher dict, then a `test_swap` that calls all three
  checks** (copy `test_inplace_fused`).

## Settled decisions (do not relitigate — see plan for the why)

- **Joint truncation** (Tucker rank weighted by the central TT, like `t3svd` per-site) — so (c) never
  compresses worse than (a). `check_truncated`'s "≤ 2× the optimal reference" is the guard.
- **Spec** = `t3svd`'s: `max_*_ranks` scalar-or-sequence (use `ranks.normalize_max_ranks(spec, len)`),
  per-step `rtol`/`atol`; rtol/atol require unstacked (frontend raises); max-rank is stacking-OK.
- **SVD everywhere** via `linalg.truncated_svd` (full-SVD-then-slice; no QR). No-truncation ⇒
  short-circuit to `t3_mult` (exact).

## Phase 3 — `t3m_swap` (method (c))

The T3 generalization of TTM, for `r ≫ d` (`O(d²·r³)`, memory `O(r̃²)`). Algorithm:

1. **Concatenate** the two central TTs into a length-`2d` chain: `G^A_0 … G^A_{d-1}` then `G^B`
   **reversed** (`reverse_tt` from `t3_operations` transposes the bonds). Same-mode cores then sit
   `~d` apart; the reversal makes the mode-`(d-1)` cores adjacent so swap-count is `d(d−1)/2`. Keep
   each core's **Tucker factor attached** to its `n_i` (tucker) leg, plus its mode index — represent a
   chain element as e.g. `(core[l,n,r], U[n,N], mode)`.
2. **Swap** to bring a same-mode pair adjacent: contract the shared bond, swap the two tucker legs,
   reshape, `truncated_svd` → two cores with swapped legs/factors. The factors/modes ride along.
3. **Contract** when same-mode cores meet: COPY-merge `W_i = U^A_i ⊙ U^B_i`, then the **joint** Tucker
   + TT truncation (same per-site logic as (b) — copy that block: orthonormalize `W`, fold the
   remainder in, weighted Tucker SVD, then the TT-bond SVD).
4. Iterate (swaps + contracts) until `d` cores remain = the product.

**The hard part is the gauge.** For the swap truncations to be *optimal* (and not lose accuracy or
blow intermediate bonds), keep the chain in **mixed-canonical gauge centered at the swap pair** (the
TTM paper: "centred at the rightmost site of the SWAP pair"); after a contraction, move the gauge
(via `truncated_svd` with no truncation, since we're SVD-everywhere). Without correct gauge the
truncating swaps are wrong — this is why (c) is more involved than (b), and why a no-truncation "swap"
(which would just be a worse (a)) is not acceptable.

Reuse from (b) in `t3_linalg.py`: the per-site joint-truncation block (Tucker orthonormalize + weighted
SVD + TT SVD), `linalg.truncated_svd`, `orth.right_orthogonalize_tt_cores` (for initial gauging),
`t3_operations.reverse_tt`. Verify with `check_exact` + `check_sweep_exact` (must reproduce the product
exactly with generous ranks) + `check_truncated`.

If gauge management proves too fiddly for a first pass, a *correct but non-optimal* fallback is exact
(non-truncating) swaps + joint truncation only at the contracts — but note this loses (c)'s memory
advantage, so it's only a stepping stone, not shippable as-is.

## Phase 4 — doc/test polish

- `t3m` docstring already covers the three methods + the rtol/atol⊥stacking warning; add a one-line
  **method-selection guidance** (b default; (a) for tiny bonds / parallel; (c) for `r≫d`).
- **Cross-method joint-quality test**: for the same `rtol`, (b)/(c) keep ranks `≤` (a)'s (+ small
  slack) on a graded-spectrum product. (This makes Decision 1 explicit; `check_truncated` covers it
  indirectly today.)
- **`test_dispatch.py`**: jit each t3m method with **max-rank** (static shapes ⇒ jit-able); rtol/atol
  stay eager. None of the t3m methods are in `test_dispatch` yet.
- Finish the `CLAUDE.md` `t3_mult`/T3M note (mark (c) done when it lands).

## Stray repo files (leave alone)
`TTM.pdf`, `T3M_handdrawn_diagrams.jpeg` are reference-only and intentionally untracked (do **not**
`git add` them). Pre-existing strays (`.idea/`, `t4s.pdf`, `t3_test_file*.npz`, `docs/conf_OLD.py`,
`docs/make.bat`) are unrelated cleanup-backlog items.
