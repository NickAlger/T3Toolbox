# Design question: pass the old point instead of the old ranks?

**Question** (maintainer, via the coordinator, 2026-11-02): can `newton_cg(x0, padded_from=x_old)` /
`padded_start(problem, x_cores, x_old)` replace the old ranks, and so remove three requirements:

- I-30: old ranks structurally minimal;
- I-34: prefix masks on a uniform `x0`;
- I-20 plus the safe-mode zero-padding check: `x0` holds `X_old` in `resize`'s layout, and the frontend `x0` is minimal.

Plan read at round 5 (1964 lines). Probe: `probe_oldpoint.py` in the session scratchpad, read-only, `t3toolbox` env.

**Answer: yes, adopt it.** Only the old-rank slice needs the layout. With the point passed, what is left is:

- one structural store check;
- one tensor-level safe-mode check, the twin of the existing `start_frame` check;
- a caveat that the frontend's existing reduction already handles.

## Findings

**(1) Nothing beyond the slice needs the layout or old-rank minimality.**

- **Unaffected by the slice:**
  - The construction (§4) reads only the old frame, the template's stores, the hint and the seed.
  - The "outside-first" coordinates are slots of the *frame* relative to the old frame's stores, not slots of `x0`.
  - The driver (§3.5) needs `x0` only for the residual.
  - The seam (§3.6) needs the frame and `x0` to represent the same tensor, not to share a layout.
- **I-30's real concern is `x_old`'s frame, not `x0`.**
  - `compute_orthogonal_representation_ranks` keeps `up_i = min(n_i, N_i)` even when `n_i` exceeds what the bonds
    realize (`ranks.py:501-568`; `down_i = min(up_i, left_i·right_{i+1})` is then smaller than `up_i`).
  - So a non-minimal `x_old` has slack rows in `U°`, and they would be pinned as old.
  - A structural reduction of `x_old` removes that slack.
  - Numerically zero directions of a minimal `x_old` stay arbitrary either way. That is already a caveat (n36).
- **What the change breaks, all small:**
  - **(a) The uniform old frame.** `geom.frame(x_old)` is wrong on the uniform layer: it applies the geometry's held masks, which describe `x0`'s structure
    (`geometry.py:644-650`). The old frame must come from `x_old`'s own `.data`, masks included. So the backend
    uniform `padded_from` is a full UT3 `.data`, unlike the bare supercore pair `x0` is.
  - **(b) A new structural check.** `old ≤ target` is no longer implied by a slice. Add an explicit check that `x_old`'s frame stores are ≤
    `x0`'s, per store and stack element.
  - **(c) Padded width on the uniform layer.** The same-tensor check needs equal padded `N` there: `__sub__` raises otherwise
    (`uniform_tucker_tensor_train.py:319-322`), whatever `allclose`'s docstring says.
  - **(d) jit conversion.** Under `use_jit`, `x_old` gets the same jax conversion as `x0` (`partition_static` for the uniform `.data`).

**(2) The same-tensor check: precise and cheap if it uses the existing `allclose`.**

- **The check.** `TuckerTensorTrain.allclose` / `UniformTuckerTensorTrain.allclose` form the difference as a T3 and take its norm
  through orthogonalization: `‖x0 − x_old‖ ≤ rtol·max(‖x0‖, ‖x_old‖)`, per stack element.
- **Tolerance.** Use `safety.effective_rtol`: 1e-9 for numpy, 1e-5 for jax.
- **Measured** at `d = 5`, ranks ≤ 8:

  | `x0` | dtype | orthogonalized difference | inner-product formula `‖a‖²+‖b‖²−2⟨a,b⟩` |
  |---|---|---|---|
  | `resize(x_old)` | float64 | 2.6e-15 | 3.3e-8 |
  | `resize(x_old)`, then the frontend's structural reduction | float64 | 2.7e-14 | 0 (lucky cancellation) |
  | `resize(x_old)`, then the frontend's structural reduction | float32 | 2.1e-7 | 1.7e-4 |

  - The inner-product formula reaches 3.3e-8 in float64, above the numpy rtol, so it would raise false errors. Never use it.
  - float32 *numpy* inputs get the numpy rtol of 1e-9 and would fail. That is the existing float32 backlog and applies to every safe-mode check.
- **Cost.** One orthogonalization sweep at the summed ranks plus two norms, about one frame sweep. It runs once per call and only in safe mode. The uniform layer takes the same route with masks.

**(3) Sharing and stacks.**

- **Sharing.**
  - The TIED check moves from `x0` to `x_old`, because the group's old block is read from `x_old`'s frame at `U°_{min(g)}`.
  - The frontend reductions take `sharing=` and keep groups tied (`uniform_fitting.py:435-458`).
- **Varying-rank uniform stack.** Per element, `ut3_to_t3(x_old.data)` gathers through `x_old`'s own masks; then the ragged path runs as now.
  - Gappy masks are fine: on a `[1 1 0 1 1 0]` `x_old` the gather recovers it to 1.5e-15.
  - I-28's bitwise cross-layer old frame survives with no prefix requirement.
  - A native `ut3_orthogonal_representations(x_old.data)` also works (it outputs prefix frame masks), at the cost of a gauge.

**(4) A non-minimal `x_old`: reduce it in the frontend, accept it in the backend.**

- **Frontend.** Reduce with the existing `_ragged_minimal` / `uniform_minimal`. This is a no-op on a continuation result, which `newton_cg` returns minimal.
- **Backend.** Accept any `x_old`, with a docstring caveat: the frame is always valid, but structural slack and numerical zeros are completed arbitrarily and pinned as old.
- **Not inside `geometry`.** The uniform reduction lives in `uniform_fitting`, which imports `geometry`, so calling it from there would create a cycle.
- **Gauge.**
  - The reduction changes the cores, not the tensor.
  - At full numerical rank, the frame's subspaces are intrinsic to the tensor: Tucker spans, interface spaces, down row spaces.
  - So the padded frame's tangent space and the first step are unchanged; only the coordinates differ.
  - Deficient directions depend on the path either way.

**(5) The oversampled-ranks rule still works.**

- **Monotonicity.** Minimal ranks and the store recurrence are monotone: 3000 of 3000 random pairs `r ≤ r'` in the probe, including a shared group.
- **Reduced `x0`.** The target is minimal, and `max(x0, minimal(r+p)) = minimal(r+p) ≥ target`.
- **Non-minimal backend `x0`.** The maximum still keeps `target ≤ oversampled`.
- **`old ≤ target`.** It follows from monotonicity whenever `x_old`'s ranks are ≤ `x0`'s and both are reduced. The check in (1b) catches a wrong `x_old`.

**(6) Other routes considered, all weaker.**

- **Keep `old_ranks`, gather through `x0`'s own masks.** Read the first `s` real slots instead of the prefix. This drops I-34 only; the layout and zero-padding precondition remain.
- **`newton_cg(x_old, padded_to=…)`.** The uniform problem's geometry is built at `x0`'s structure before `newton_cg` runs, so `x0` must exist anyway.
- **Pass the previous level's frame.** This exposes optimizer internals and gains nothing over passing the point.

## Recommendation

Adopt the old-point API. Everything the three requirements protected is then covered by:

- **structural, both modes:** same layer, shape (and padded `N` on the uniform layer) and stack; `x_old`'s frame stores ≤ `x0`'s;
- **numerical, safe mode:** `x0.allclose(x_old)`; TIED `x_old` on a shared geometry;
- **caveat:** `x_old`'s numerically zero directions, and in the backend its structural slack, are completed arbitrarily and pinned as old. The frontend reduces `x_old` structurally.

I-34 disappears. I-20 keeps only its `start_frame` half: a structure check against the reduced `x0`, whose error says
"build it at `x0`'s minimal structure". I-30 becomes a frontend reduction plus a backend caveat.

## What the plan would change

- **API** (§3.5, §3.6, §5.4–§5.7).
  - `padded_from=x_old`: on the frontend, a T3 object of `x0`'s layer; on the ragged backend, the cores pair; on the uniform backend, the full `.data`.
  - `padded_start(problem, x_cores, x_old=None, …)`.
  - `padded_structures(x_cores, x_old, oversample)`.
- **§3.4.**
  - The old frame becomes `t3_orthogonal_representations(x_old)` on the ragged layer. On the uniform layer it goes through the ragged path with `x_old`'s own masks, never `self.frame(x_old)`.
  - Drop the slice paragraph, the prefix-mask check and the minimal-old-ranks check.
  - Add the store check.
- **§3.6, §3.8, `numerical_contracts.md`.**
  - The zero-padding check and its message are replaced by `x0.allclose(x_old)`.
  - The frontend may reduce `x0` under `padded_from`.
- **Decisions.** Drop I-34. Rewrite I-30 (frontend reduction plus backend caveat) and I-20 (`start_frame` only). I-28 keeps its substance, now through `x_old`'s own masks.
- **Tests.**
  - T5(g): a gappy `x_old` is accepted and gives the same frame as its prefix-layout twin; old stores above the template's, or a layer or `N` mismatch, raise.
  - T7(b)/(g): `padded_from=x_old`, and `padded_from=x0`.
  - T9(d): `x0 ≠ x_old` raises in safe mode and is skipped under `unsafe()`.
  - A frontend run with a non-minimal `x0` and `padded_from`.
  - T13 passes `x_old`.
- **Docs.**
  - The continuation loop becomes `padded_from = X if stats['stop_reason'] == 'gradient' else None`, with `X` the previous level's result. No rank bookkeeping.
  - The `padded_from` / `x_old` docstrings carry the caveat.
