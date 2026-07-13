# Newton-CG fitting diagnostics / display — plan

> **STATUS: DONE (2026-07-13), branch `feat/newton-cg-display`.** All six slices shipped + green
> (D1 loop plumbing + NewtonInfo/callback/history; D2 `block_sumsq`; D3 backend `optimizer_display`;
> D4 frontend `verbose=`/`callback=`; D5 `examples/fit_probe_display.py` + doctest; D6 uniform mirror —
> `block_sumsq_over_probes` made dual-path so the uniform kinds inherit it, validation auto-packed).
> `newton_cg(..., verbose=True, val_sample=, val_data=)` on both ragged and uniform layers; backend users
> get the identical display via `optimizer_display.make_newton_display`. Kept below as the design record.

*Design settled with Nick (2026-07-13), branch off `main`. Adds an optional per-iteration **diagnostic
display** to the Newton-CG fitting loop (objective/gradient, CG stats, line-search, and a per-`(mode,
order)` relative-error table), plus a **stored per-iteration history** in the returned `stats`. The whole
capability lives in the **backend** so a raw-`.data` user gets the identical display; the frontend adds
only a `verbose=True` convenience switch.*

## 0. Resume from a fresh context (read this first)

The design is decided; do not re-derive it. The Newton-CG loop is `backend/optimizers.py::newton_cg`
(the frontend `optimizers.py::newton_cg` is a thin adapter). Each Newton iteration already computes the
objective `f`, `gnorm`, the forcing term `η`, `slope`, runs `_cg_solve` (inner CG), and Armijo-backtracks
— but it **throws away** the CG iteration count / success / achieved residual (they live inside
`_cg_solve`'s `xwhile` state and only `p` is returned) and the line-search step count. No per-`(mode,
order)` reduction exists (`kind.sumsq` collapses everything). This plan adds the missing plumbing, a
kind-level per-block reduction, and a backend display module. **Anti-drift is a first-class goal (Nick):
the display is a backend capability, not a frontend one.** Build §5 in order.

## 1. The architecture (backend-owned display)

The display needs only backend objects — the `kind` (block reduction + `point_forward` on validation),
the `LocalModel`'s residual, `x_cores`, `geom.inner`, and the raw `val_sample`/`val_data`. **None of it is
frontend-specific**, so it all lives in the backend; the frontend `verbose=True` is a one-liner over the
backend builder. Three cleanliness rules:

- **Isolate the side effect.** The formatter is a **pure function returning a string**; printing is behind
  an injectable `print_fn=print`. The pure algorithm modules stay pure; the string formatter is testable
  without capturing stdout.
- **`block_sumsq` is a `SamplingKind` field** (the `sumsq` sibling) — the per-`(mode, order)` layout
  knowledge stays with the kind, in the backend. No frontend import anywhere in the display path.
- **The callback is host-side** — it reads the concrete residual each Newton iteration. Compatible with
  `newton_cg` (only the *inner CG* jits under `use_jit`; the outer loop + callback are already on the
  host). Documented: a display callback is incompatible with any hypothetical fully-jitted *outer* loop.

## 2. Decisions (locked this session)

1. **Backend placement.** `block_sumsq` → `backend/fitting.py` (kind field). A new
   **`backend/optimizer_display.py`** owns the formatter + the callback builder. `backend/optimizers.py`
   gains the generic `callback` hook + richer `stats`. The frontend adds only `verbose=`.
2. **Table = per-`(mode, order)` relative error**, scaled by the **data norm** (`‖r_ij‖ / ‖y_ij‖`) — the
   honest per-block recovery error, independent of `ω` (Decision B): for derivatives it auto-normalizes
   the order-magnitude spread. `‖y_ij‖ = 0` blocks render as `—`.

2a. **Layout rule — the dataset axis fills a spare grid dimension.** Three axes compete for the 2-D grid:
   **mode**, **order**, **dataset** (train/val). A flat table shows two; the third goes into cells. Pick
   by how many *data* axes the kind has:
   - **2 data axes** — `probe_derivatives` (mode × order): **mode rows × order cols**, dataset in
     `train|val` cells (flattening mode·order into columns gets too wide).
   - **1 data axis** — the dataset axis is free, so **dataset = rows, the data axis = cols**: plain
     `probe` → mode cols; `apply`/`entries_derivatives` → order cols. (Compact; train/val clearly
     labelled instead of crammed in cells.)
   - **0 data axes** — plain `apply`/`entries` (scalar): no table, one line
     `rel err  train 5.2e-4 | val 6.1e-4`.
   **The stored matrices stay canonical `(n_mode, n_order)`** whatever the rendering — layout is cosmetic,
   never leaks into `stats['diagnostics']` or a user's scripting. The `rows=… cols=…` legend states the
   orientation per layout, so the cross-kind switch is self-describing.
3. **`block_sumsq` is UNWEIGHTED** (raw `‖r_ij‖²`, no `ω`) — so the table is the true relative error *and*
   the unweighted objective `½Σ block_sumsq(r)` falls out for free (shown alongside the weighted
   `lm.objective` when `ω ≠ 1`).
4. **Format `%.1e`** — Python pads the exponent to a signed 2-digit field, so every cell is exactly 7
   chars (`5.2e-04`, `1.0e+00`, `8.1e-16`) → column-aligned with no extra work.
5. **Train + validation, validation opt-in.** Default = train only. When the user passes
   `val_sample`/`val_data`, the callback evaluates **one** extra `point_forward` on the val sample per
   Newton step (no transpose/sweep — trivial next to CG) and shows **one table with `train|val` cells**.
   The val data is closed over by the backend callback builder — it never enters `newton_cg`'s core.
6. **Store diagnostics.** `newton_cg` always returns `stats['history']` (per-iteration scalar dicts).
   The display callback additionally records the per-block error matrices; the frontend merges them into
   the returned `stats` (`losses` stays for backward compat).
7. **API: both** — `verbose=True` (built-in formatter) *and* a power-user `callback=` hook, on the
   backend (`make_newton_display` + raw `callback`) and the frontend (`verbose` flag + pass-through
   `callback`).
8. **Ragged first; uniform is REQUIRED, not optional.** Ship ragged, then mirror `block_sumsq` for the
   packed uniform kinds + pack `val` in the uniform frontend path. Tracked by an un-missable TODO + a
   skipped uniform test (§5, §7) so it cannot quietly slip.

## 3. The per-iteration display

One header line + the table, per Newton iteration:

```
iter  3 | obj 5.23e-04 (unwt 4.9e-04)  ‖g‖ 1.44e-03 (2.1e-02·g₀)  | CG 14/200 tol 3.0e-04 resid 8.7e-05 ✓ | ls 2 α 2.5e-01 ‖Δx‖/‖x‖ 3e-02 | Δf -1.2e-04 ρ 0.98 | 0.31s
  rel err (train|val)   rows=mode  cols=order
              ord0             ord1             ord2             ord3
    m0  5.2e-04|6.1e-04  1.0e-02|1.1e-02  3.3e-02|3.5e-02  8.1e-02|8.4e-02
    m1  ...
```

Header fields (all cheap from loop state):
- **(0)** iteration number.
- **(1)** `obj` = weighted `½‖ω⊙r‖²` (`= lm.objective`); `(unwt …)` shown only when `ω ≠ 1`; `‖g‖` and
  `‖g‖/‖g₀‖`.
- **(2)** CG: `iters/maxiter`, `tol` (= `η·‖g‖`), achieved `resid` (`‖Hp+g‖`), and `✓` (converged) /
  `⋯` (hit maxiter) / `⌇` (truncated on nonpositive curvature).
- **(3)** line-search steps + accepted `α`.
- **(5 / additions)** `‖Δx‖/‖x‖` (relative step), `Δf` (actual decrease), `ρ` (actual/predicted reduction
  — the GN-model-trust diagnostic, `predicted = −(α·slope + ½α²·pᵀHp)`), elapsed wall-time.

The table layout follows the axis rule (Decision 2a) — the `train|val` cells above are the
**2-data-axis** `probe_derivatives` case. The **1-data-axis** kinds put the dataset in rows instead:

```
# plain probe (mode only)                    # apply / entries derivatives (order only)
  rel err   rows=train/val  cols=mode          rel err   rows=train/val  cols=order
             m0       m1       m2                          ord0     ord1     ord2     ord3
    train  5.2e-04  1.0e-02  3.3e-02             train   5.2e-04  2.7e-02  9.1e-02  1.8e-01
    val    6.1e-04  1.1e-02  3.5e-02             val     6.1e-04  2.9e-02  9.4e-02  1.9e-01
```

and plain `apply`/`entries` (scalar) is the one-liner `rel err  train 5.2e-04 | val 6.1e-04`. The
`block_sumsq` matrix shapes are `probe_derivatives → d×(order+1)`, plain `probe → d×1`,
`apply/entries_derivatives → 1×(order+1)`, plain `apply/entries → 1×1`; the formatter dispatches the
layout off that shape (+ whether `val` is present). (Assumes the frame/core stack `C = ()` — a
single-tensor fit; `block_sumsq` sums any `C` in with `W`.)

## 4. Data structures

- **`NewtonInfo`** (frozen dataclass, `backend/optimizers.py`) — passed to `callback(info)` every
  iteration, so a custom callback sees everything: `iteration`, `x_cores`, `lm` (LocalModel — residual /
  sample / frame / objective), `objective`, `gnorm`, `g0norm`, `forcing_eta`, `cg_tol`, `cg_iters`,
  `cg_converged`, `cg_resid`, `cg_truncated`, `ls_steps`, `alpha`, `slope`, `pHp`, `delta_f`,
  `step_rel`, `wall_time`.
- **`stats['history']`** — a list of the scalar subset of `NewtonInfo` (JSON-ish dict, no arrays), always
  returned. When the display callback is used, its records (adding `train_err`, `val_err` matrices) are
  merged into the returned stats (`stats['diagnostics']`).

## 5. Implementation slices (ragged first)

**D1 — backend loop plumbing** (`backend/optimizers.py`). `_cg_solve` returns `(p, cg_info)` (iters,
converged, resid, truncated) — read from the `xwhile` final state (no algorithm change). Capture the
line-search step count + accepted `α`. Compute `pHp` (`= lm.gn_quadratic(p)`, one forward — for `ρ`),
`delta_f`, `step_rel`, `wall_time`. Assemble `NewtonInfo`; add `callback=None` (call per iter) and the
always-on scalar `history` in `stats`. Tests: `stats['history']` fields present + a recording callback
sees each iter; CG `converged`/`truncated`/`iters` correct on a tiny problem (incl. a corewise
gauge-singular `H` → truncated).

**D2 — `block_sumsq`** (`backend/fitting.py`). Add `block_sumsq_over_samples` / `block_sumsq_over_probes`
(the `sumsq_*` siblings that keep mode+order), wire a `block_sumsq` field on every kind (ragged), returning
the 2-D `(n_mode, n_order)` **unweighted** matrix. Tests: dense oracle — `block_sumsq(r)` per block equals
a hand `np.sum(r_ij**2)`; `block_sumsq(r).sum() == unweighted ‖r‖²`; every kind's matrix shape matches its
`ω`-shape.

**D3 — backend display module** (`backend/optimizer_display.py`, new). `format_newton_iter(info,
data_norms, val=…, fmt) -> str` (pure); `make_newton_display(problem, val_sample=None, val_data=None,
print_fn=print, fmt='%.1e', record=True) -> (callback, records)` — precompute `block_sumsq(data)` once,
close over it + val. Tests: the formatter's string is column-aligned (assert equal cell widths, incl. a
`1.0e+00` next to a `5.2e-04`); the `train|val` layout; `—` for a zero-norm block; a full
`make_newton_display` + `newton_cg` run records the right matrices.

**D4 — frontend** (`optimizers.py`). `topt.newton_cg(..., verbose=False, val_sample=None, val_data=None,
callback=None)`: build the backend display callback when `verbose` (pack `val` if uniform — but that path
is D6), thread `callback`, merge `records` into the returned `stats`. Test: a `verbose=True` ragged run
prints (capture) + returns `stats['diagnostics']`; a custom `callback=` is honored.

**D5 — example (both layouts) + doctest.** A dedicated **`examples/fit_probe_display.py`** that fits one
synthetic tensor **both ways**, back to back, each `verbose=True` with a train/val split, so a single run
prints **both table layouts**:
  1. **plain `probe`** (`probe_model` / `newton_cg` kind `'probe'`) — the 1-data-axis layout (mode cols,
     train/val rows);
  2. **`probe_derivatives`** — the 2-data-axis layout (mode rows, order cols, `train|val` cells).
The maintainer runs it to eyeball both displays (the acceptance check for the whole feature). It doubles
as living documentation of the two renderings. Plus a small module doctest on the pure `format_newton_iter`
capturing a couple of aligned rows (fixed inputs; asserts the alignment/structure, not raw optimizer
floats — the doctest convention). Runs clean; numbers pasted from a real run, never hand-written.

**D6 — uniform mirror (REQUIRED follow-up, do NOT drop).** `block_sumsq` for the packed uniform kinds
(mode axis 0, order axis 1, padded free-mode summed like `sumsq_over_probes`'s packed path); pack `val`
in the uniform frontend path. Flip the skipped uniform test (§7) to active. Tracked in
`dev/HANDOFF.md` until done.

## 6. Backend-user surface (the anti-drift check)

A raw-`.data` user gets the identical display with no frontend:

```python
import t3toolbox.backend.optimizers as bopt
import t3toolbox.backend.optimizer_display as bdisp
cb, records = bdisp.make_newton_display(problem, val_sample=vs, val_data=vd)
x, stats = bopt.newton_cg(problem, x0, callback=cb)   # prints each iter; records == stats history
```

## 7. Risks / watch-list

- **Don't forget uniform (D6).** The whole reason for a skipped-but-present uniform test + a HANDOFF line:
  the ragged/uniform mirror is load-bearing and the maintainer explicitly flagged this.
- **`block_sumsq` blast radius**: a new kind field touches every kind builder (singletons + derivative
  builders + uniform `dc.replace`). Grep all `SamplingKind(` / `dc.replace(` sites.
- **jit**: the callback is host-only; `use_jit=True` (inner CG) still works (callback reads the concrete
  residual). A `test_dispatch` note, not a new jit path.
- **Alignment robustness**: `%.1e` is 7 chars for exponents in `[-99, 99]`; a 3-digit exponent (never for
  fitting rel-errors) would be 8. The formatter right-pads to the actual max cell width defensively.
- **Backward compat**: `newton_cg` return stays `(x, stats)` with `losses`/`newton` intact; `history` /
  `diagnostics` are additive.

## 8. Out of scope

- Display for the *other* optimizers (`gradient_descent`/`mc_sgd`/`adam`) — the `callback` hook could
  extend there later; this plan is Newton-CG only.
- A plotting/TUI layer — text only.
- Per-`C`-stack tables (batched fits) — the display assumes a single tensor (`C = ()`).
