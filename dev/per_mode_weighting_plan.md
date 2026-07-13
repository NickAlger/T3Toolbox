# Per-mode residual weighting — plan

*Design settled with Nick (2026-07-13), branch `main`. Adds **per-mode** residual weighting to the
probe fitting models, generalizing the existing **per-order** weight `ω` on the derivative models from a
vector to a `(mode, order)` **matrix**. Fully backward compatible. Covers backend + frontend + uniform
mirror + the topt optimizer adapter + one worked example + docs.*

## 0. Resume from a fresh context (read this first)

The design is decided; do not re-derive it. Today the only residual weighting is a **per-order** vector
`ω` of length `order+1`, owned by the three *derivative* sampling kinds (`{apply,entries,probe}_
derivatives_kind` in `backend/fitting.py`), built by `_make_order_weight(weight, order)` and injected in
exactly two hooks: the objective/quadratic reduction `sumsq` (`×ω`) and the gradient/transpose (`×ω²`),
so the objective is `½‖ω ⊙ (S(x) − y)‖²` while `forward`/`point_forward`/`data` stay **raw**. This
preserves `H = 𝒥ᵀω²𝒥` (symmetric PSD) and `pᵀHp = ‖ω⊙Jp‖²`. See `dev/archive/derivative_fitting_plan.md`
§2.4 and `docs/fitting_and_optimization.md` §4.6.

Per-mode weighting is the **same diagonal-residual-scaling idea with one more index axis**. It slots into
the *same two hooks*; no new math, no new invariant. The whole change is: (a) `ω` becomes a matrix
`ω[mode, order]`; (b) plain `probe_model`/`probe_kind` gain a weight (the order-0 special case of the same
machinery); (c) apply/entries stay order-only (they have no mode axis). Build §5 in order.

## 1. The one structural fact that shapes everything

Only **probe** produces a per-mode residual. The residual layouts:

| kind | residual | order axis | **mode axis** |
|---|---|---|---|
| apply / entries (plain)       | one array `W+C`               | ✗ | **✗** |
| apply / entries (derivatives) | `(order+1)+W+C`               | ✓ | **✗** |
| probe (plain)                 | list of `d` arrays `W+C+(Nᵢ,)`         | ✗ | **✓** |
| probe (derivatives)           | list of `d` arrays `(order+1)+W+C+(Nᵢ,)` | ✓ | **✓** |

apply/entries contract *every* mode into a scalar — there is no per-mode residual to weight (the order-`t`
symmetric derivative is `Σ_{|S|=t} T(p on S, x elsewhere)`, a sum over mode-*subsets*, not a per-mode
decomposition). So:

> **Per-mode weighting is a probe-only feature.** It is well-defined for `probe_model` and
> `probe_derivatives_model`; it has no meaning for apply/entries.

## 2. Decisions (locked this session)

1. **Mode weighting is probe-only.** `probe_model` (per-mode) and `probe_derivatives_model`
   (mode × order). apply/entries derivative models stay **order-only**; a genuine per-mode weight handed
   to them (mode dim > 1) is a **structural error** (hard, both modes) with a message pointing to probe.
   Plain `apply_model`/`entries_model` gain **no** weight parameter (a single scalar residual has nothing
   to weight — a global weight is a no-op for least-squares).

2. **The weight is a matrix `ω[mode, order]`, conceptual shape `(d, order+1)`**, applied as
   `½‖ω ⊙ r‖²` (`ω` is the amplitude weight: `×ω` in `sumsq`, `×ω²` in `transpose`; `forward`/`data` raw).
   Interpretation follows plain NumPy **right-aligned broadcasting**, so the axis order is `(…, mode,
   order)` with **order innermost / most-important** (for models that *have* an order axis):
   - bare 1-D `(order+1,)` or `(1, order+1)` → **per-order** (row), broadcast over modes — *this is
     exactly today's behavior → fully backward compatible*;
   - `(d, 1)` → **per-mode** (column), broadcast over orders;
   - `(d, order+1)` → the full independent matrix ("an arbitrary combination").

   Nick's importance model: axes are ordered by how much they can break the fit, most-important
   **innermost** (rightmost). A hypothetical future `other` axis (less important than mode) slots in
   **outermost** (`other × mode × order`); because new axes go on the left and numpy right-aligns, every
   existing input keeps its meaning — this is the forward-compat guarantee, and it dictates Decision 3.

3. **The bare-vector rule (not an exception) — Decision C.** A bare 1-D vector always binds to the
   **innermost (most-important) axis the model has**. Derivative models have `(mode, order)`, so bare =
   order; **plain probe has only `(mode,)`**, so bare `(d,)` = **per-mode**. Same rule, different
   innermost axis — no special case. Consequently **plain probe accepts a 1-D `(d,)` only and *rejects*
   `(d, 1)`**: a 2-D input asserts an order axis plain probe does not have, and — the real reason — a
   `(d, 1)` accepted today would, against a future `(other, mode)` tensor, silently right-align to
   "per-*other*, broadcast over mode" (the opposite of per-mode). Rejecting it keeps `(d,)` the single,
   forward-stable spelling. (`(d, 1)` stays valid for `probe_derivatives`, where the order axis is real.)
   Internally a `(d,)` is canonicalized to the 2-D `(d, 1)` working form so `_make_weight` stays one code
   path; the rejection is a frontend `ndim==1` validation, not a change to the machinery.

4. **Inputs:** numpy/jax arrays or (nested) sequences; normalized with `np.asarray(..., float)` at the
   boundary. The weight is **host-numpy static structure** (like `ω` today and the uniform masks), not
   traced data — it folds into the compiled program as a device constant. **Shape mismatches are
   structural → hard error, always** (`o ∉ {1, order+1}` or `m ∉ {1, d}`).

5. **Invariants unchanged.** `H = 𝒥ᵀ ω² 𝒥` symmetric PSD; `pᵀHp = ‖ω⊙Jp‖²`; the gauge projection `Π`
   is applied *after* the weighted transpose, untouched. Mode weighting is a diagonal residual scaling —
   the existing order-weight proof carries over verbatim with `ω` a matrix.

6. **In scope:** `backend/fitting.py`, `backend/uniform_fitting.py`, `fitting.py`
   (`UniformGaussNewtonModel` aux included), the **topt** adapter `optimizers.py` +
   `uniform_least_squares_problem`, tests, one example, docs.

7. **Example: honest, empirically verified.** A probe fit where per-mode scales vary widely; per-mode
   weighting improves the recovered tensor (Frobenius error of `T_fit − T_true`). Construction finalized
   by experiment — I will report real numbers, and if plain Frobenius does not show it I will make the
   effect real (relative per-mode measurement noise / rank-limited fit — the PDE forward/reverse
   scaling), never fake it. See §7.

## 3. Weight semantics — the canonical form

Normalize any input to a 2-D `ω[m, o]` with `m ∈ {1, d}`, `o ∈ {1, order+1}`, then apply per layout:

- **ragged probe (list of `d`):** element `i` scaled by the order-vector `ω[i if m>1 else 0, :]` along
  its order axis (axis 0 per element; length `o`). Plain probe has `o = 1` → a per-mode **scalar**;
  derivatives have `o = order+1`. The *same* reshape `(o,)+(1,)*(x[i].ndim−1)` covers both (for plain
  probe the leading `1` broadcasts harmlessly over a `W` axis).
- **apply/entries array (`(order+1)+W+C`, no mode axis):** require `m = 1` (else the structural error of
  Decision 1); place the order-vector at the order axis. This is today's `_make_order_weight` array path.
- **packed uniform probe (`(d,)+(order+1,)+W+C+(N,)` for derivatives, `(d,)+W+C+(N,)` for plain):** place
  `ω`'s mode axis at `axis 0` and order axis at `axis 1` (derivatives) / no order axis (plain), 1s
  elsewhere; `m = 1` broadcasts over `d`.

**Backward-compat guarantee:** `weight=(order+1,)` on any derivative model → normalizes to `(1, order+1)`
→ byte-identical behavior to today. A regression test pins this.

## 4. `_make_order_weight` → matrix-aware helper (`backend/fitting.py`)

Split the current `_make_order_weight(weight, order, order_axis=0)` into:

- **`_weight_matrix(weight, order, bare)`** → `np.ndarray (m, o)` or `None`. `bare ∈ {'order','mode'}`
  controls the 1-D interpretation (`'order'` → `(1, o)` row; `'mode'` → `(m, 1)` col). A 2-D array passes
  through **idempotently** when `bare='order'` (so the uniform layer can re-feed a normalized matrix);
  when `bare='mode'` (plain probe) a 2-D input is **rejected** (Decision 3 — plain probe has no order
  axis). Validates `o ∈ {1, order+1}` here (order is known); `m ∈ {1, d}` is validated in the frontend
  (§6, where `d` is known) and otherwise falls out of broadcasting.
- **`_make_weight(w2d, order_axis=0, mode_axis=None)`** → `apply_w(x, power)`. `None → identity`. Handles
  the three layouts of §3 (list branch indexes `w2d` rows per mode; array branch places the non-unit
  axes at `mode_axis`/`order_axis`, either possibly `None`). Numpy `**` on the host → device-constant on
  the jax path, exactly like today.

Per-kind wiring:

| builder | `bare` | branch / axes |
|---|---|---|
| `probe_kind(weight=None)` **NEW** (plain) | `'mode'` | ragged list, `o=1` |
| `apply_derivatives_kind(order, weight)`   | `'order'` | array, `order_axis=0`, `mode_axis=None` |
| `entries_derivatives_kind(order, weight)` | `'order'` | array, `order_axis=0`, `mode_axis=None` |
| `probe_derivatives_kind(order, weight)`   | `'order'` | ragged list (mode = list index) |
| `uniform_probe_kind(x0, weight=None)` **NEW** | `'mode'` | packed, `mode_axis=0`, no order |
| `uniform_apply/entries_derivatives_kind`  | `'order'` | packed array, `order_axis=1`, `mode_axis=None` |
| `uniform_probe_derivatives_kind`          | `'order'` | packed, `mode_axis=0`, `order_axis=1` |

`APPLY`/`ENTRIES`/`PROBE` stay module singletons `= *_kind()` (unweighted) — backward compatible for
direct importers. `probe_kind()` with `weight=None` returns the same behavior as the `PROBE` singleton.

## 5. Implementation slices (incremental, reviewable)

**S1 — backend ragged** (`backend/fitting.py`). Add `_weight_matrix` + generalize `_make_weight`; add
`probe_kind(weight=None)` (`PROBE = probe_kind()`); route the 3 derivative kinds through the matrix path;
`o` validation in the kinds. Tests: dense oracle that the objective is `½Σ_{i,t}(ω[i,t] r_{i,t})²` and the
gradient is `𝒥ᵀ(ω²r)` for a **full matrix** on probe_derivatives; plain-probe per-mode; the backward-compat
regression; the apply/entries **reject-per-mode** structural error.

**S2 — frontend ragged** (`fitting.py`). `probe_model(..., weight=None)` builds `fb.probe_kind(weight)`;
matrix weight in `probe_derivatives_model`; order-only guard (`m==1`) in
`apply/entries_derivatives_model`; **no** weight on `apply/entries_model`. Validate `m ∈ {1, d}` with a
clear message. Update the `# ω, (order+1,)` signature shape-comments → `# ω[mode,order], (d,order+1)
broadcast (§4.6)` and the docstrings. Tests: frontend↔backend oracle for each weighted case.

**S3 — uniform mirror** (`backend/uniform_fitting.py` + `UniformGaussNewtonModel`).
`uniform_probe_kind(x0, weight=None)` + `uniform_sampling_kind(name, x0, weight=None)`; matrix weight in
the uniform derivative kinds (packed axes per §3). `_uniform_model` threads weight for **plain probe** too
(today it only threads for the `order`-kinds). `UniformGaussNewtonModel.weight` aux becomes a **hashable
nested tuple** (store the *normalized* 2-D matrix → tuple-of-tuples → stable jit cache key); `.kind`
rebuild passes it back (idempotent `_weight_matrix`). Tests: `TestUniformGaussNewtonModel` matrix +
plain-probe cases vs the backend `LocalModel`; **`tests/test_dispatch.py`** — jit a probe_derivatives
matrix-weight op and a plain-probe mode-weight op (proves the matrix folds as a numpy constant, no tracer
leak); the uniform mask-strict / garbage-padding robustness check (weight is applied on real parts
post-forward, so padding stays don't-care — assert it).

**S4 — topt adapter** (`optimizers.py` + `uniform_least_squares_problem`). `_setup(...)` builds
`bfit.probe_kind(weight)` for plain probe when `weight` is given (error if weight given for plain
apply/entries); `uniform_least_squares_problem` threads weight through the plain-probe kind path. Backend
`optimizers.py` `least_squares_problem`/`LocalModel` are kind-generic → unchanged. Test: a plain-probe
per-mode-weighted `newton_cg`/`lbfgs` run that descends, ragged + uniform.

**S5 — example** (`examples/fit_*_per_mode_weight.py`). §7. Verify empirically before committing.

**S6 — docs** (`docs/fitting_and_optimization.md`). §8.

Run after each backend-touching slice: `test_fitting` + `test_optimizers_frontend` +
`backend/test_uniform_fitting` + `backend/test_optimizers` + `test_dispatch` (the wide blast radius of a
kind-signature change per CLAUDE.md).

## 6. Validation placement (house philosophy: structural → hard error, both modes)

- **`o ∈ {1, order+1}`** — in the kind builders (`order` known there).
- **`m ∈ {1, d}`** — in the frontend factories (`d = len(ww)` / `len(x's modes)` known there), with a
  message naming the two legal shapes.
- **per-mode weight to apply/entries derivatives** (`m > 1`) — structural error, raised where the array
  branch sees `mode_axis=None` with `m>1`, and pre-empted in the frontend with the friendlier message.
- **2-D weight to plain probe** (`ndim != 1`, e.g. `(d, 1)`) — structural error (Decision 3), raised in
  `probe_model` / the topt plain-probe path / `uniform_probe_kind`; message: "plain probe takes a 1-D
  per-mode weight `(d,)`; it has no order axis".
- All of the above are **structural** (shape/consistency) → raised unconditionally, independent of
  safe/unsafe mode, matching the `TuckerTensorTrain`/`T3Frame` `__post_init__` precedent.

## 7. The example (honesty-first; construction finalized by experiment)

Goal: a **simple synthetic tensor** where per-mode weighting demonstrably helps, measured by the
Frobenius error of the recovered tensor. Primary construction to try (matches Nick's PDE forward/reverse
intuition — "modes with widely varying scalings"):

1. `T_true` = a known low-rank T3 (e.g. dims `(12,12,12)`, small ranks).
2. Probe per mode → targets `y_i = probe_i(T_true)`; the modes carry **widely varying scale** `s_i`
   (built into `T_true` or applied to the probe design).
3. Make the weighting *matter* (weighting only changes the minimizer under noise or rank-starvation): add
   **per-mode relative measurement noise** `σ_i = ρ·s_i` (the realistic PDE case) and/or fit at a rank
   below `T_true`.
4. Fit twice with `probe_model` + manifold Newton-CG (or the scipy L-BFGS bridge): **unweighted** vs
   **per-mode** `ω_i = 1/s_i` (inverse-scale = homoscedastic GLS, provably lower-variance by
   Gauss–Markov).
5. Report `‖T_fit − T_true‖_F / ‖T_true‖_F` for both, plus per-mode relative errors.

I will **run it and paste the real numbers**. If inverse-scale + relative noise does not move the plain
Frobenius metric enough to be a clean teaching example, I will either strengthen the construction or
report a complementary balanced/per-mode metric alongside — but the headline claim will be whatever the
numbers actually support. The example is a **probe** fit (mode weighting is probe-only); it isolates the
mode effect (no order weighting), keeping the story clean. (The docs will *also* show the `(mode, order)`
matrix form on `probe_derivatives`.)

## 8. Docs (`docs/fitting_and_optimization.md`)

- **§4.6** retitle → "Residual weighting is a per-order / per-mode **matrix** `ω`". Cover: the
  `ω[mode, order]` `(d, order+1)` convention; the importance ordering + right-aligned broadcasting
  (row = order, column = mode, matrix = both); the **probe-only** nature of mode weighting; the plain
  `probe_model` bare-`(d,)` exception; a worked snippet + a pointer to the new example.
- The `weight` bullet (~line 60) and the factory list (~line 78) — note the matrix + `probe_model`'s new
  `weight`.
- Sweep the code shape-comments (`# ω, (order+1,)`) across `fitting.py` / `backend/fitting.py` /
  `backend/uniform_fitting.py` / `optimizers.py` to the matrix contract.

## 9. Risks / watch-list

- **Uniform jit cache key.** The `weight` aux goes matrix → must stay a hashable **nested tuple** or
  "compile-once" breaks. Store the normalized 2-D form; covered by a `test_dispatch` recompile check.
- **Blast radius.** `_make_weight` feeds 2 ragged + 3 uniform derivative kinds + the 2 new plain-probe
  kinds + the packed `order_axis`/`mode_axis` logic. Grep all consumers; run the full fitting + dispatch
  + uniform suites, not just touched files.
- **Backward compat.** `(order+1,)` behavior must be byte-identical — pinned by a regression assert.
- **Plain-probe rejects `(d, 1)`** (Decision 3) — a deliberate, forward-compat-preserving rule, not an
  oversight; a regression test asserts both the `(d,)` accept and the `(d, 1)` reject so a future refactor
  can't silently loosen it.
- **Example honesty** (§7) — flagged; numbers reported, effect made real not narrated.

## 10. Explicitly out of scope

- The `fit(...)` facade / auto-`ω` (1.1). The user supplies `ω` (per-mode RMS, `1/s_i`, a physical
  scale…); default `ω = 1`.
- Reviving the parked **weighted tensor-network** layer (unrelated to this residual weight).
- A per-mode weight for apply/entries (mathematically undefined — §1).
- Weighting axes beyond `(mode, order)` (the hypothetical `other` axis stays hypothetical).
