# T3M method (c) `swap` — concrete implementation plan

This is the actionable build plan for the swap method. It **supersedes the Phase-3
section of `docs/t3m_handoff.md`** and refines method (c) in `docs/t3m_plan.md`. The
*theory* (why this is the right generalization, the leaf-frame tension, the HT view,
why oversample+cleanup is forced) lives in `docs/ttm_t3m_ht_note.tex`. Read the plan
+ handoff for the spec/decisions; this file says exactly what to write.

The algorithm below is **already prototype-verified** against the dense oracle
(`/tmp/proto_swap.py` in the session it was developed): exact path machine-precision
on all `STRUCTURES`, stacked included; rtol respects tolerance; cap-2 within the ≤2×
bar. The code blocks here are the backend-adapted (xnp, `'...'`) form of that verified
prototype, so transcribe them rather than re-deriving.

## Settled design decisions (recap — do not relitigate)

1. **Swap skeleton = TTM.** Concatenate `[G^A_0..G^A_{d-1}] + reverse_tt([G^B])`,
   reverse makes mode `d-1` adjacent (seam), `d(d-1)/2` swaps. The *only* change from
   TTM is the merge: copy-tensor → **Khatri–Rao** `W_i = U^A_i ⊙_row U^B_i` + joint
   weighted-Tucker truncation (same per-site logic as `t3svd`/method (b)).
2. **Gauge = mixed-canonical centered at the active core.** Everything surrounding the
   core being truncated is orthogonal toward it, so every swap/merge SVD is locally
   optimal. Tracked **explicitly** via a `center` index (shuttle → bubble-with-center →
   contract). This is what makes the no-truncation path machine-precise.
3. **Pre-down-orthogonalize** both inputs' Tucker factors (`down_orthogonalize_tucker_cores`,
   = t3svd's first step / HT "orthogonalize leaves toward root"). Free quality win
   (1.52×→1.21× at cap-2), exact path preserved.
4. **Oversample + cleanup** is the principled resolution of the leaf-frame tension
   (see §"Oversample semantics"). Default `oversample=1` (no cleanup; aggressive corner).
5. **Joint truncation** (Tucker weighted by the canonical environment) — never compresses
   worse than method (a) for tight tolerances. Per-step rtol/atol; rtol/atol require
   unstacked (frontend already raises); max-rank is stacking-OK.
6. **SVD everywhere** via `linalg.truncated_svd` (full-SVD-then-slice; no QR). No
   truncation requested ⇒ short-circuit to `t3_mult` (exact).

## The algorithm (precise)

Chain element = `[G, U, mode]`: central core `G` (stack+`(rL,n,rR)`), Tucker factor
`U` (stack+`(n,N)`), original mode index. Maintain an integer `center`.

**Invariant after processing mode `i`:** `m_i` is the orthogonality center; cores
left of it (`a_0..a_{i-1}`) are left-orthogonal, cores right of it
(`m_{i+1}..m_{d-1}`, remaining `b`'s) are right-orthogonal.

Chain at the start of mode `i` (length `d+1+i`): `[a_0..a_i]` (idx `0..i`),
merged block `[m_{i+1}..m_{d-1}]` (idx `i+1..d-1`), `[b_i..b_0]` (idx `d..d+i`);
`b_i` sits at idx `d`.

Driver per mode `i` (from `d-1` downto `0`):
- if `i < d-1`: `move_center → d-1` (right end of block); then bubble `b_i` left to
  idx `i+1` via swaps at `(d-1,d),(d-2,d-1),…,(i+1,i+2)` (push='left', center follows `b_i`).
- `contract(idx i, idx i+1)` → `m_i` (Tucker-truncated, center).

For `i=d-1` there is no block/bubble: contract the seam pair directly.

## Backend helpers (transcribe; xnp inferred from inputs)

```python
def _swap(eL, eR, push, max_rank, rtol, atol):
    """Swap adjacent chain elements (G,U,mode); push='left'|'right' = where the center lands."""
    GL, UL, mL = eL                       # GL: stack+(a, nL, b);  UL: stack+(nL, NL)
    GR, UR, mR = eR                       # GR: stack+(b, nR, c);  UR: stack+(nR, NR)
    xnp, _, _ = get_backend(False, is_jax_ndarray(GL) or is_jax_ndarray(GR))
    stack = GL.shape[:-3]
    a, nL, b = GL.shape[-3:]
    _, nR, c = GR.shape[-3:]
    T = xnp.einsum('...anb,...bmc->...anmc', GL, GR)          # stack+(a, nL, nR, c)
    T = xnp.moveaxis(T, -3, -2)                               # stack+(a, nR, nL, c)
    M = T.reshape(stack + (a * nR, nL * c))
    U, s, Vt = linalg.truncated_svd(M, max_rank=max_rank, rtol=rtol, atol=atol)
    rp = s.shape[-1]
    if push == 'right':
        newGL = U.reshape(stack + (a, nR, rp))                # left-orth, physical nR
        newGR = (s[..., :, None] * Vt).reshape(stack + (rp, nL, c))
    else:                                                     # push left -> center at left core
        newGL = (U * s[..., None, :]).reshape(stack + (a, nR, rp))
        newGR = Vt.reshape(stack + (rp, nL, c))               # right-orth, physical nL
    return [newGL, UR, mR], [newGR, UL, mL]                   # Tucker factors / modes swap


def _contract(eL, eR, max_tucker_rank, rtol, atol):
    """Merge same-mode pair (a_i,b_i)->m_i. KR Tucker factor, weighted-env Tucker truncation. Center=m_i."""
    Ga, Ua, mi = eL                       # Ga: stack+(aL, nA, b);  Ua: stack+(nA, N)
    Gb, Ub, _  = eR                       # Gb: stack+(b, nB, c);   Ub: stack+(nB, N)
    xnp, _, _ = get_backend(False, is_jax_ndarray(Ga) or is_jax_ndarray(Gb))
    stack = Ga.shape[:-3]
    aL, nA, b = Ga.shape[-3:]
    _, nB, c = Gb.shape[-3:]
    N = Ua.shape[-1]
    P = nA * nB
    site = xnp.einsum('...anb,...bmc->...anmc', Ga, Gb).reshape(stack + (aL, P, c))
    W = xnp.einsum('...nx,...mx->...nmx', Ua, Ub).reshape(stack + (P, N))   # Khatri-Rao
    Uw, sw, Vt = linalg.truncated_svd(W)                      # lossless orthonormalize of W
    Utilde_full = Vt                                          # [k, N]
    Mw = Uw * sw[..., None, :]                                # [P, k]  -> folded into the site
    site = xnp.einsum('...apc,...pk->...akc', site, Mw)       # [aL, k, c]
    k = site.shape[-2]
    env = xnp.moveaxis(site, -2, -3).reshape(stack + (k, aL * c))           # weighted env (both bonds orth)
    Qt, st, Vt2 = linalg.truncated_svd(env, max_rank=max_tucker_rank, rtol=rtol, atol=atol)
    ntil = st.shape[-1]
    Utilde = xnp.einsum('...kr,...kN->...rN', Qt, Utilde_full)              # [ntil, N]  new Tucker factor
    site = (st[..., :, None] * Vt2).reshape(stack + (ntil, aL, c))
    site = xnp.moveaxis(site, -3, -2)                         # [aL, ntil, c]  = m_i central core (center)
    return [site, Utilde, mi]


def _move_center(chain, frm, to):
    """Move the orthogonality center from index frm to to (no truncation)."""
    if to > frm:
        for p in range(frm, to):
            nGL, nGR, _ = linalg.left_svd_pair(chain[p][0], chain[p + 1][0])
            chain[p][0], chain[p + 1][0] = nGL, nGR
    elif to < frm:
        for p in range(frm, to, -1):
            nGL, nGR, _ = linalg.right_svd_pair(chain[p - 1][0], chain[p][0])
            chain[p - 1][0], chain[p][0] = nGL, nGR
    return to
```

Notes: `_swap`/`_contract` reshape on a stack prefix and use `'...'` einsums, so they
are stack-aware exactly like the rest of the backend. `truncated_svd`/`*_svd_pair`
already infer jax-ness internally; the helpers only need `xnp` for einsum/moveaxis.

## Oversample semantics (coherent for max-rank AND tol together)

`oversample = k ≥ 1` (numeric; default `1`). The rule is uniform: **relax every active
criterion by `k` in-process; apply the exact criteria once at cleanup.**

- in-process Tucker cap (at contracts, per-mode): `⌈k · mtr[i]⌉` (None stays None)
- in-process TT bond cap (at swaps, **uniform** = `⌈k · max(mrr)⌉`; None stays None)
- in-process tolerances: `rtol/k`, `atol/k` (smaller ⇒ keeps more)
- cleanup (when it runs): a single `t3svd` round at the **exact** `(mtr, mrr, rtol, atol)`.

In-process swaps use a **uniform** bond cap (not per-position) — the swap process does
not preserve per-position bond identity. Per-position TT caps are recovered at cleanup.
(Per-position *Tucker* caps are honored in-process: each contract knows its mode.)

**Cleanup runs iff** `oversample > 1` **or** `max_tt_ranks` was given as an explicit
(non-scalar) sequence. So the fast path — `oversample=1` with scalar/None caps and/or
rtol — does no cleanup; a per-position TT sequence triggers a final round to honor it
exactly; any oversampling triggers the final round that makes the oversampling pay off.

Why this is correct: in-process truncation is centered (locally optimal) and only
bounds memory; the decisive truncation is the cleanup round, run when every Tucker leg
is already compressed and with `O(d)` (not `O(d²)`) steps. Empirically `k≈3` recovers
the optimal error; memory stays `O((k·r̃)²)`, still ≪ the `O(r²)` full product for `r̃≪r`.

## Driver

```python
def t3m_swap(x, y, max_tucker_ranks=None, max_tt_ranks=None, rtol=None, atol=None, oversample=1):
    if max_tucker_ranks is None and max_tt_ranks is None and rtol is None and atol is None:
        return t3_mult(x, y)                                  # exact short-circuit (oversample irrelevant)
    if oversample < 1:
        raise ValueError('oversample must be >= 1, got %r' % (oversample,))

    Ux, Gx = x
    Uy, Gy = y
    d = len(Ux)
    use_jax = is_jax_ndarray(x) or is_jax_ndarray(y)
    xnp, _, _ = get_backend(False, use_jax)

    mtr = ranks.normalize_max_ranks(max_tucker_ranks, d)        # exact targets
    mrr = ranks.normalize_max_ranks(max_tt_ranks, d + 1)
    osc = lambda c: None if c is None else int(math.ceil(oversample * c))
    mtr_in = tuple(osc(c) for c in mtr)
    nn = [r for r in mrr if r is not None]
    mrr_in = osc(max(nn)) if nn else None                      # uniform in-process bond cap
    rtol_in = None if rtol is None else rtol / oversample
    atol_in = None if atol is None else atol / oversample

    # leaves -> root, then build/gauge the chain
    Ux, Gx = ragged_orth.down_orthogonalize_tucker_cores((Ux, Gx))
    Uy, Gy = ragged_orth.down_orthogonalize_tucker_cores((Uy, Gy))
    Gx_o = orth.left_orthogonalize_tt_cores(list(Gx))
    Gy_o = orth.right_orthogonalize_tt_cores(t3_ops.reverse_tt(list(Gy)))
    Uy_rev = list(Uy[::-1])
    chain = [[Gx_o[i], Ux[i], i] for i in range(d)] \
          + [[Gy_o[j], Uy_rev[j], d - 1 - j] for j in range(d)]
    center = d - 1                                              # seam pair = (d-1, d)

    for i in range(d - 1, -1, -1):
        if i < d - 1:
            center = _move_center(chain, center, d - 1)
            for p in range(d - 1, i, -1):
                chain[p], chain[p + 1] = _swap(chain[p], chain[p + 1], 'left', mrr_in, rtol_in, atol_in)
                center = p
        chain = chain[:i] + [_contract(chain[i], chain[i + 1], mtr_in[i], rtol_in, atol_in)] + chain[i + 2:]
        center = i

    tucker_cores = tuple(e[1] for e in chain)
    tt_cores = tuple(e[0] for e in chain)

    need_cleanup = (oversample > 1) or (
        isinstance(max_tt_ranks, typ.Sequence) and not isinstance(max_tt_ranks, (int, np.integer)))
    if need_cleanup:
        rounded, _, _ = ragged_t3svd.t3svd(
            (tucker_cores, tt_cores),
            max_tt_ranks=max_tt_ranks, max_tucker_ranks=max_tucker_ranks, rtol=rtol, atol=atol)
        return rounded
    return tucker_cores, tt_cores
```

Add `t3m_swap` (and the three `_`-helpers stay private, not in `__all__`) — append
`'t3m_swap'` to `t3_linalg.py`'s `__all__`.

## Frontend (`tucker_tensor_train.py::t3m`)

- Add parameter `oversample: float = 1` to the `t3m` signature (one-per-line, aligned
  shape/role comment), document it in the docstring (applies to `'swap'` only; default
  1 = no oversampling; suggest `2` as a good modest value for better quality at a small
  memory cost; `t3svd`-quality as `oversample → ∞`).
- Register `'swap': ragged_linalg.t3m_swap` in the backend dict (remove the
  `NotImplementedError` path for `'swap'`).
- Validation: `if oversample < 1: raise ValueError`; `if oversample != 1 and method != 'swap':
  raise ValueError('oversample only applies to method="swap"')`. Keep the existing
  shape/stack and rtol/atol⊥stacking checks.
- Dispatch the call with `oversample` only for swap, e.g.:
  ```python
  kw = dict(max_tucker_ranks=max_tucker_ranks, max_tt_ranks=max_tt_ranks, rtol=rtol, atol=atol)
  if method == 'swap':
      kw['oversample'] = oversample
  return TuckerTensorTrain(*backend(self.data, other.data, **kw))
  ```
- Method-selection guidance line (Phase 4): (b) default; (a) tiny bonds / parallel /
  oracle; (c) for `r≫d`, with `oversample≈2` if you want near-(a) quality.

## Tests (`tests/test_t3m.py`, `tests/test_dispatch.py`)

- `test_swap`: `check_exact('swap')`, `check_sweep_exact('swap')`, `check_truncated('swap')`
  (copy `test_inplace_fused`).
- **Update `test_validation`**: the `assertRaises(NotImplementedError)` for `method='swap'`
  must go (swap is implemented now). Add `assertRaises(ValueError)` for `oversample=0.5`
  and for `oversample=2, method='inplace_fused'`.
- `test_swap_oversample`: on the cap-2 case, assert `relerr(oversample=3) < relerr(oversample=1)`
  and `relerr(oversample=3) ≤ 1.1 × dense-t3svd reference`; assert oversample preserves the
  exact path (`check_exact` with oversample=2 via a small helper, or a direct call).
- `test_swap_per_position_tt`: non-uniform `max_tt_ranks=(1,2,3,2,1)` on the d=4 structure
  with `oversample=1` → output `tt_ranks` ≤ the sequence per-position (cleanup honors it).
- **Stacked + oversample + max-rank**: per-slice vs oracle (cleanup must be stacking-safe);
  reuse the stacked harness with `max_tucker_ranks=2, max_tt_ranks=2, oversample=2`.
- Cross-method joint-quality (Phase 4): for a graded-spectrum product, same `rtol`,
  `(c, oversample=2)` keeps ranks ≤ `(a)` + small slack (Decision 5 made explicit).
- `test_dispatch.py`: jit `t3m_swap` with **max-rank** (static shapes) including
  `oversample=2` (cleanup is `t3svd`-on-static-shapes, jit-able); a numpy==jax smoke
  check. rtol/atol stay eager. The driver's Python loops over `d`/swaps are static under
  trace (shapes known), so this jits.

## Edge cases / gotchas to verify

- `d == 1`: no swaps, one contract at the seam (`center=0`, contract idx `0,1`). Prototype
  passed `((10,),(3,),(1,1))` at machine precision — keep that structure in the sweep test.
- Boundary bonds: `m_0.left` and `m_{d-1}.right` must come out `1`. (They do — the swaps
  re-thread the seam; verified.) `to_dense` already squashes tails, but the cleanup
  `t3svd` also squashes — fine either way.
- `oversample=1` + scalar caps: uniform in-process cap == exact, **no cleanup**, fast path.
- `oversample=1` + rtol: per-step rtol in-process, no cleanup → accumulation ~`d·rtol`
  (documented; use `oversample>1` for tighter quality with rtol).
- Stacked: `rtol/atol` already forbidden by the frontend; max-rank + cleanup is
  stacking-safe (`t3svd` with max-rank works stacked).
- Run the **full** t3m + adjacent suites after wiring (the frontend dict + validator change
  ripples): `test_t3m`, `test_tucker_tensor_train`, `test_dispatch`.

## Doc updates after it lands

- `docs/t3m_plan.md`: rewrite method (c) from "truncate per-site during swaps" to the
  oversample+cleanup version; mark Phase 3 done; note the `oversample` knob + default.
- `CLAUDE.md` "Open questions / TODO" → T3M bullet: mark (c) `t3m_swap` done; note the
  three methods all live + tested; point to `ttm_t3m_ht_note.tex` for the theory.
- `docs/t3m_handoff.md`: mark Phase 3 complete (or note it's superseded by this file).

## Commit slices (verify tests green before each)

1. Backend `t3m_swap` + helpers + `__all__`; frontend dispatch/validation/docstring;
   `test_swap` (exact/sweep/truncated) + `test_validation` update. Run full suite.
2. Oversample tests (`test_swap_oversample`, `test_swap_per_position_tt`, stacked+oversample)
   + cross-method joint-quality test.
3. `test_dispatch` jit cases for `t3m_swap`.
4. Doc updates (`t3m_plan.md`, `CLAUDE.md`, handoff) + method-selection guidance line.

## Acceptance criteria

- No-truncation / generous-rank: `‖swap − exact‖/‖exact‖ < 1e-10`, all structures,
  stacked included.
- Truncated (cap or rtol): within the dense-`t3svd` reference bound (`check_truncated`'s ≤2×);
  with `oversample≈3`, within ~1.1×.
- `oversample=1` scalar path does no cleanup; per-position TT sequence and any
  `oversample>1` do exactly one `t3svd` cleanup.
- jit dispatch (max-rank) green in `test_dispatch`; full suite green.
