# Data-shaped residual weighting — plan

> **STATUS: DEFERRED (Nick, 2026-09-03) — do not implement from this v1.** Two independent reviews
> found v1's "single mechanism, clean cut" wrong against its own decisions in two places (a shipped
> example breaks; a silent positional mis-bind downstream) and its breaking surface much larger than
> assumed. The consolidated findings and the revised recommendation — a SEPARATE kwarg for the
> data-shaped weight, the `(mode, order)` matrix left exactly as it is, the effective weight their
> product, shipped as a non-breaking `### Added` — are in
> `dev/data_shaped_residual_weight_reviews.md` §1–§3. Pick up there; §3–§10 below remain the
> reference for the canonical form, the leaf plumbing, validation, tests and docs, re-cut per the
> reviews.
>
> *v1 status line, kept for the record:* PLANNED (2026-09-03), nothing implemented. Design settled with
> Nick (T3Polynomial's eigenproblem detour, 2026-09-03; survey record in T3Polynomial
> `dev/t3toolbox-upstream-notes.md`, 2026-09-03 section). Decisions: **single mechanism, clean cut**
> (breaking at the backend surface, non-breaking at the frontend); the residual weight becomes a
> **traced leaf** shaped like the data; the `ω[mode, order]` matrix survives as frontend sugar.

*Generalizes the residual weight `ω` from the `(mode, order)` matrix of
`dev/per_mode_weighting_plan.md` to any array **broadcastable to the data blocks** — per row (a
(sample, ray) measurement), per order, per mode, per component, all through one elementwise
mechanism. The matrix is the small special case and keeps its spelling. Covers backend ragged +
uniform, the optimizer layer (minibatch, jit), the frontend, tests, docs, CHANGELOG.*

## 0. Resume from a fresh context (read this first)

The design is decided; do not re-derive it. Today `ω` is a **field of the sampling kind**
(`backend/fitting.py`: `ProbeKind.weight`, `_DerivativesKind.weight`), canonicalized to a host-numpy
2-D matrix by `_weight_matrix` (L199–234), turned into `apply_w(x, power) = x·ω**power` by
`_make_weight` (L237–285) and applied in exactly two hooks: `sumsq` (`×ω`, L423 / L437) and
`transpose` (`×ω²`, ragged L554 / L595 / L618 / L645, uniform L152 / L175 / L197 / L226). Because a
kind is a value-hashed frozen dataclass whose fields ARE its jit cache key (`ValueHashedFields`,
`common.py` L597–675; `parameters_not_closures.md`), the weight is static structure that folds into
the compiled program as a device constant (`docs/fitting_and_optimization.md` §4.6, L311–312). A
data-sized weight cannot live there: numpy fields hash by `tobytes()` per model rebuild, a jax field
raises `TypeError`, and `dev/OPEN_QUESTION_extension_surface.md` §1 already names "aux → leaf" for
changing weights as a genuine design change. `precompute_and_caching.md` (L437–442) files ω at
*per-problem* scope ("Problem fields") — the docs are ahead of the code, and this plan brings the code
to them.

**The whole change**: (a) the weight leaves the kind and becomes a **leaf** on `Problem` /
`LocalModel` / `GaussNewtonModel`, converted like `sample` / `data`, gathered by the minibatch
`take`, partitioned as dynamic in the pytrees; (b) the kind keeps only **layout** knowledge — where
the order / mode / W axes sit in its residual — and its `sumsq` / `transpose` take the weight as an
operand; (c) the frontend expands the `(mode, order)` matrix into a small broadcastable leaf, so
existing callers see the same signatures and, on numpy, bit-identical numbers. Build §5 in order.

## 1. The structural facts that shape everything

**Residual layouts** (`backend/fitting.py` L63, L75, L107, L121; `fitting.py` L519–531):

| kind | residual | order axis | mode axis | W axes |
|---|---|---|---|---|
| apply / entries (plain) | one array `W+C` | ✗ | ✗ | leading |
| apply / entries (derivatives) | `(order+1,)+W+C` | 0 | ✗ | after order |
| probe (plain) | list of `d` arrays `W+C+(Nᵢ,)` | ✗ | list index | leading |
| probe (derivatives) | list of `d` arrays `(order+1,)+W+C+(Nᵢ,)` | 0 | list index | after order |
| uniform probe (plain / derivatives) | packed `(d,)+W+C+(N,)` / `(d,)+(order+1,)+W+C+(N,)` | — / 1 | 0 | after (d, order) |

**Only W is data stacking at the optimization layer.** `C` is the T3's own `stack_shape` (a batch
of base points, on every core; `batching_and_stacking.md` §1–3): `Problem.local_model` and
`GaussNewtonModel` evaluate a stacked point as a batch of objectives, the library optimizers refuse it
(`require_unstacked_for_regularizer` / the 2026.2.0 "stacked optimization raises"). `K`, the tangent
stack, lives on variation cores and appears only inside the frame-inner transposes; the GN model's
residual never carries it. **Therefore the weight is shaped like the DATA — no C, no K — and the
library inserts the C axes itself at evaluation** (§3). This replaces any "equal ndim to the
residual" rule with the principle: *weights are data-shaped; C and K belong to the tensor side and
are broadcast.*

**The two hooks stay the only hooks.** `½‖ω⊙r‖²` → `sumsq` scales by `ω`, `transpose` (the gradient
`𝒥ᵀω²r` and the Hessian action `𝒥ᵀω²𝒥p`) by `ω²`; `forward` / `point_forward` / `data` stay raw, so a
custom `draw` returning raw `data_B` can never silently break the residual (§4.6's rationale is
unchanged). `H = 𝒥ᵀω²𝒥` stays symmetric PSD and `pᵀHp = ‖ω⊙Jp‖²`.

**The display is unweighted by design.** `block_sumsq*` (`backend/fitting.py` L98–103, L390–393) and
the per-iteration relative-error tables never see ω; the header's `(unwt …)` annotation
(`optimizer_display.py` L126–130) shows the weighted misfit next to them. Unchanged here.

## 2. Decisions (locked 2026-09-03)

1. **Single mechanism, clean cut.** One weight object, the data-shaped one. The backend kinds
   **lose** their `weight` field and the `weight=` parameter of `probe_kind`, `apply_derivatives_kind`,
   `entries_derivatives_kind`, `probe_derivatives_kind`, `uniform_sampling_kind`,
   `uniform_derivatives_kind`, `Uniform*Kind.from_point(weight=)`; `kind.weight` and
   `kind._apply_weight` go. This is **`Changed — breaking`** at the backend surface (a first-class
   documented layer, `docs/api_reference.rst` L4–8) with an upgrade paragraph in `release_notes.md`.
   No deprecation shim: a shim keeps the static path alive and is a second mechanism.
2. **Non-breaking at the frontend.** `probe_model(weight=)`, `apply/entries/probe_derivatives_model(weight=)`,
   `newton_cg` / `mc_sgd` / `adam` / `lbfgs` `weight=` keep their signatures and accept everything they
   accept today with the same rules (§6); numpy results are **bit-identical** for the matrix forms
   (gate test, §7.1). The data-shaped forms are additions.
3. **The weight is a leaf** at per-problem scope: `Problem.weight`, `LocalModel.weight`,
   `GaussNewtonModel.weight`, `None` by default. Converted to jax alongside `sample` / `data` under
   `use_jit` (`_prepare_jit_inputs`); required to be jax when present for `_jit_engaged`; dynamic in
   `_flatten_problem` / `_flatten_local_model` / the `GaussNewtonModel` registration. Consequence,
   documented as `Changed`: same-shape weights of different values now **share** one compilation, and
   ω² is computed on the traced path (bit-identical on numpy / x64; ulp-level on float32 jit, which no
   contract promises against).
4. **`draw(rng)` may return a third element.** `draw(rng) → (sample_B, data_B)` stays valid for an
   unweighted problem; a weighted problem needs `(sample_B, data_B, weight_B)` and the default
   `flat_draw` provides it. A weighted problem given a 2-tuple draw is a **structural error** (the
   weight cannot be sliced for it) with a message naming the third element. `Changed` (contract widened).
5. **Canonical dtype is float, always.** `np.asarray(w, dtype=float)` on the ragged path (numpy
   float64), the jax float dtype `sample` / `data` receive under jit. A **bool** input is accepted and
   converted (a 0/1 mask is a legitimate weight); the conversion is stated in the signature comment.
   Why it matters: `_is_static_leaf` (`common.py` L686–695) classifies numpy bool arrays as **static**
   (masks are bool), so an unconverted mask weight would be hashed into the cache key and baked as a
   constant instead of traced. NaN / inf entries are rejected (finite-fill rule of the uniform contract).
6. **All six kinds take a weight.** Plain `apply` / `entries` gain a per-row weight under the same
   mechanism (weighted least squares on scalar measurements; the earlier "a global weight is a no-op"
   reason no longer applies once the weight can vary per row). A scalar weight is accepted and is a
   no-op — no special case.
7. **The `(mode, order)` matrix is frontend sugar**, expanded once at model-build time into the small
   broadcastable leaf the kind's layout dictates (§3), with every existing rule preserved: bare vector
   binds to order for derivative kinds and to mode for plain probe; plain probe rejects 2-D; `o ∈ {1,
   order+1}`; `m ∈ {1} ∪ [d, ∞)` with extra rows truncated; apply/entries reject `m > 1`.
8. **Uniform mirror**: the weight is packed like data (`pack_weight` beside `pack_data`, zero fill);
   padded slots must be finite (they multiply zero residual padding, so any finite garbage is inert).

## 3. Weight semantics — the canonical form and the broadcast contract

**Canonical (backend) form** — per kind, the data's layout with **no C**, float dtype:

| kind | canonical weight |
|---|---|
| apply / entries (plain) | one array broadcastable to `W` |
| apply / entries (derivatives) | one array broadcastable to `(order+1,)+W` |
| probe (plain) | list of `d` arrays, element `i` broadcastable to `W+(Nᵢ,)` |
| probe (derivatives) | list of `d` arrays, element `i` broadcastable to `(order+1,)+W+(Nᵢ,)` |
| uniform probe | one packed array broadcastable to `(d,)+W+(N,)` / `(d,)+(order+1,)+W+(N,)` |

with **ndim equal to the data block's ndim** (so that broadcasting is unambiguous: a W-carrying
weight against a probe block must not slide into the `Nᵢ` slot) and, per W axis, size `∈ {1, W_k}`
(a weight whose W block is neither all-singleton nor exactly W is `broadcast_to` at canonicalization —
materialized, documented — so the flat minibatch gather has one rule: gather if the W block is full,
keep if it is singleton).

**Evaluation with a stacked point.** `apply_weight(x, w, power, n_lead, n_w)` (module-level, replacing
`_make_weight`'s closure): `x` is the residual `(lead)+W+C+(trail)`, `w` the canonical weight
`(lead)+W+(trail)`; insert `x.ndim − w.ndim` unit axes into `w` after position `n_lead + n_w`, multiply by
`w**power`. `n_lead` is 1 for the derivative kinds (order axis) / 2 for the packed uniform derivative
probe (d, order) / 1 for the packed uniform plain probe (d) / 0 otherwise — a kind ClassVar, the
successor of `_order_axis` / `_mode_axis`. `w=None` → `x` unchanged (the identity path stays the
fast path).

**Sugar expansion** (frontend / `optimizers._setup`, where `d`, `order`, `n_w` are known):
`ω[mode, order]` `(m, o)` → ragged probe: list of `d` arrays `ω[i or 0].reshape((o,) + (1,)*n_w + (1,))`
(derivatives; plain probe: `ω[i].reshape((1,)*n_w + (1,))`); apply/entries derivatives: one array
`ω[0].reshape((o,) + (1,)*n_w)`; uniform packed probe: `ω[:d].reshape((d, o) + (1,)*n_w + (1,))`
(plain: `(d,) + (1,)*n_w + (1,)`). These are literally the reshapes `_make_weight` builds today
(L270, L275–284), now materialized once as a tiny leaf instead of per call. `o = 1` broadcasts over
the order axis; `m = 1` over modes; rows beyond `d` are truncated as today.

**Uniform packing** — `pack_weight(name, weight, N)` (`backend/uniform_fitting.py`, beside
`pack_data`): probe kinds only. Each element's trailing axis is `Nᵢ` → `pack_vectors` (zero fill) or `1`
→ kept; the `d` elements are stacked on a new leading axis, broadcasting each to a common shape first
where the elements disagree in their singleton pattern (materializes; documented). apply/entries:
unchanged. NaN/inf anywhere → structural error at canonicalization, before packing.

## 4. Where the weight lives — the leaf plumbing

- `backend/optimizers.py`: `Problem(geom, kind, sample, data, regularizer=None, weight=None)` and
  `LocalModel(..., regularizer=None, geom_aux=None, weight=None)` — **appended with defaults**, so
  positional construction keeps working. `least_squares_problem(geom, kind, sample, data,
  regularizer=None, weight=None)`. `Problem._sample_and_data` → `_batch(sample, data, weight, where)`:
  all three given or none (a weighted problem with `sample`+`data` but no `weight` is the structural
  "2-tuple draw" error of Decision 4). `local_model` / `objective` pass the weight to `kind.sumsq` /
  `LocalModel`. `flat_draw` returns `kind.take(sample, data, idx, weight)` → a 3-tuple iff the problem
  is weighted. `_minibatch_step_problem` / `_mc_sgd_step` / `_adam_step` gain `weight_B` (module-level,
  closure-free, as now — the jit wrapper memoized on them stays stable). `_prepare_jit_inputs`
  converts `weight` with `tree_to_jax`; `_jit_engaged` requires it jax when present.
  `_flatten_problem` / `_flatten_local_model`: the weight joins the `partition_static` dynamic tuple;
  `_unflatten_*` rebuild positionally (append at the END of the tuple and the constructor call).
- `fitting.py` (frontend): `GaussNewtonModel.weight: typ.Any = None` (appended); `objective_value`,
  `gradient`, `gn_quadratic`, `gn_hessian`, `evaluate` pass it to the kind; the pytree registration's
  positional lambdas gain the leaf (surprise §9.6 — edit both lambdas together and add the
  `test_dispatch` leaf check). `_uniform_model(..., weight)` packs it via `pack_weight` and stores it on
  the model instead of on the kind.
- `backend/fitting.py`: kinds lose the field; `sumsq(out, n_w, weight=None)`, `transpose(r, sample,
  frame, sweep, weight=None)`; `take(sample, data, idx, weight=None)` returns `(sample_B, data_B)` or
  `(sample_B, data_B, weight_B)` (the gather applied to every element of a probe list; singleton W kept).
  New module-level `canonical_residual_weight(weight, data, kind, n_w)` (validates ndim/broadcast/W
  rule/finite, canonicalizes dtype, materializes the mixed-W case) and `expand_weight_matrix(w2d, kind,
  n_w)` (the sugar, §3).
- `optimizers.py` (frontend `_setup`, L201–204 region): the weight no longer builds a weighted kind; it
  is expanded / canonicalized against the data and handed to `least_squares_problem(weight=)`.
- `backend/uniform_fitting.py`: `uniform_least_squares_problem(..., weight, ...)` keeps its `weight`
  parameter (now: sugar or data-shaped, packed and routed to the `Problem`); `_ptake_*` gain the
  weight gather (packed W positions: probe data axis 1, derivative probe axis 2, scalar data 0 / 1).

## 5. Implementation slices (incremental, reviewable; gates after each)

**S0 — inventory** (`refactoring_methodology.md` "Before you start"). Grep every reference to
`weight` / `_apply_weight` / `_make_weight` / `_weight_matrix` / `_canonical_weight` / `.weight` /
`weight=` across code, tests, docstrings, docs prose, `CLAUDE.md`, `dev/`, `examples/`; both import
forms. Distinguish residual-weight hits from **edge-weight** hits (`T3Weights`, `weights`,
`weighted_norm`, `absorb_weights`, `docs/weighting.md`, `weighted_internals.md`) — the latter are out of
scope and must not be touched (`naming_conventions.md` L40–45: plural = edge weights). The inventory
is the sweep plan.

**S1 — backend ragged core** (`backend/fitting.py`). `apply_weight` module function; kinds lose
`weight` + `__post_init__` canonicalization + `_apply_weight`; `n_lead` ClassVar; `sumsq` / `transpose`
take `weight`; `take` gathers it; `canonical_residual_weight` + `expand_weight_matrix`. Tests (S1):
`TestResidualWeighting` rewritten to hand the model a weight (its explicit-indexing oracles are
non-circular and stay); a data-shaped oracle (per-row, per-component, explicit indexing); the
duplication identity; `expand_weight_matrix` equals the documented reshapes; rejections (§6).

**S2 — backend optimizers** (`backend/optimizers.py`). §4 plumbing; `flat_draw` 3-tuple;
`_minibatch_step_problem` / steps; pytrees; jit inputs. Tests: a weighted `Problem` minibatch step
equals a hand computation on the drawn subset; a 2-tuple custom draw on a weighted problem raises;
an unweighted problem with a 2-tuple draw is unchanged (bit-identical); pytree round trip.

**S3 — frontend** (`fitting.py`, `optimizers.py`). `_canonical_weight` → sugar + data-shaped
acceptance; `GaussNewtonModel.weight` + registration; the six factories thread the weight to the
model; `_setup` routes to the problem. Tests: the frontend↔backend oracle per weighted case;
**the sugar bit-identity gate** (§7.1); the `test_dispatch` cache tests updated (§7.5).

**S4 — uniform mirror** (`backend/uniform_fitting.py`, `fitting._uniform_model`). `pack_weight`,
`_ptake_*` weight gather, uniform kinds lose `weight`, `uniform_least_squares_problem` routes it.
Tests: dense-vs-ragged with a data-shaped weight; exact masks; **garbage-padded weight**
robustness; jit compile-once; `uniform_equivalence_contract` items.

**S5 — examples**: `examples/fit_per_mode_weight_probes.py` runs unchanged (sugar); add one short
per-row example only if it earns its place (a heteroscedastic-noise probe fit with `ω = 1/σ_row`) —
verify empirically before committing (`per_mode_weighting_plan.md` §7's honesty rule).

**S6 — docs + CHANGELOG** (§8), then the `sphinx -W` gate.

**S7 — downstream check** (not in this repo): T3Polynomial passes `weight=(d+1, J+1)` through
`fitting._newton_fit`; its `tests/test_fitting_recovery.py` and the jets identity tests must be
green on the new library with no source change (the sugar path), then its `per_block_weights` may
return data-shaped arrays for the schemes that motivated this (open-issues 29/30 there).

Gates after every backend-touching slice (the per-mode plan's list, still right): `test_fitting`,
`test_optimizers_frontend`, `backend/test_uniform_fitting`, `backend/test_optimizers`,
`test_dispatch`, `test_optimizer_display`, plus the module doctests. `set -o pipefail`; never edit
tracked files while a gate runs (`refactoring_methodology.md` "Gates").

## 6. Validation placement (structural → hard error, both modes)

- **Sugar rules** (unchanged, now in `expand_weight_matrix` / the frontend): `o ∈ {1, order+1}`;
  `m ∈ {1} ∪ [d, ∞)` (`1 < m < d` rejected; `m > d` truncated); apply/entries reject `m > 1`; plain
  probe rejects 2-D. Messages verbatim from today where they exist.
- **Data-shaped rules** (`canonical_residual_weight`, called where the data is known — the frontend
  factories, `_setup`, `least_squares_problem`): ndim must equal the data block's; every non-W axis
  size `∈ {1, data size}`; every W axis size `∈ {1, W_k}` (mixed → `broadcast_to`); a probe weight
  must be a length-`d` list (or one array applied to every mode — accept, expand); entries finite;
  dtype → float (bool converted silently, documented).
- **Draw contract**: a weighted problem receiving a 2-tuple minibatch → `ValueError` naming
  `weight_B`; an unweighted problem receiving a 3-tuple → `ValueError` (a weight the problem does not
  know about is a programming error, not a feature).
- Backend leniency policy as today: the kinds validate what they can see (`n_lead`, broadcasting at
  multiply time raises numpy's own error); the frontend pre-empts with the friendlier message.

## 7. Tests the conventions demand

1. **Sugar bit-identity** (the gate for Decision 2): for all six kinds × ragged / uniform, with a
   matrix weight, `objective_value`, `gradient`, `gn_quadratic(p)`, `gn_hessian(p)` are
   `np.array_equal` to the values computed by the current commit's code path. Implement as a
   fixture of reference values captured from `main` before S1 (seeded, small), checked exactly on
   numpy; `allclose` on float32 jit.
2. **Duplication identity**: weight `√2` on measurement `j` equals duplicating `j` in `(sample, data)`
   — objective, gradient, GN quadratic and Hessian action (`allclose`: the W-sum order differs).
3. **Explicit-indexing oracles** for per-row, per-component and full data-shaped weights on
   `probe_derivatives` (both geometries), per-row on plain `apply` (WLS), pattern of
   `TestResidualWeighting`: the objective is a hand sum, `𝒥ᵀ(ω²r)` equals the unweighted gradient
   on the explicitly scaled residual, `pᵀHp = ‖ω⊙Jp‖²`.
4. **Ragged / uniform equivalence** with a data-shaped weight on the gauge-invariant quantities;
   **garbage-padded** weight slots (large finite) give identical results; NaN / inf rejected.
5. **jit cache**: same-shape different-valued weights → **one** trace (the existing
   `test_dispatch` assertion "different residual weights must not share a compilation" at L684–685
   flips to "must share"); `_leaves_all_jax` covers the new leaf; a rebuilt model with the same kind
   parameters still compiles once (the existing test).
6. **Minibatch**: `flat_draw` on a weighted problem gathers ω consistently (the drawn objective equals
   a hand computation); custom 2-tuple / 3-tuple mismatches raise; `_ScaledRegularizer` scales by
   count, not weight mass — documented, tested unchanged.
7. **Stacked point C** through `GaussNewtonModel` with a data-shaped weight (C inserted, not
   broadcast against): `test_optimizers.py` L720–745 pattern.
8. **Rejections**, each isolated (`testing_strategy.md` L500–533): wrong ndim; non-broadcastable
   trailing axis; bad W size; non-finite; wrong list length; draw-tuple mismatches.
9. **Display**: `block_sumsq` unweighted; the `(unwt …)` annotation fires with a data-shaped weight.
10. **Doctests**: one block in `probe_derivatives_model` (a per-row weight), one in §4.6; the
    existing blocks unchanged (`doctest_style.md`: one example per observable behaviour, no
    cross-product).

## 8. Docs and CHANGELOG

- `docs/fitting_and_optimization.md` §4.6: retitle ("the residual weight is data-shaped; the
  `(mode, order)` matrix is sugar"), the canonical-form table of §3, the draw-contract widening in
  §4.4 / §2.3, and **remove** the "host-numpy static structure … device constant" sentence (now a
  leaf; compile shared across weights). `_prepare_jit_inputs`'s docstring likewise.
- `docs/api_reference.rst`: the kinds no longer list `weight`; `least_squares_problem(weight=)`.
- `docs/contributor/fitting_internals.md` (the weight hooks), `precompute_and_caching.md` (the scope
  ladder entry is now true), `parameters_not_closures.md` (a note that the residual weight is the
  one per-problem *leaf* the kinds consume — parameters are still fields), `numerical_contract_catalog.md`
  (the finite-weight rule; ulp-level float32 statement), `naming_conventions.md` (unchanged rule,
  add `pack_weight`).
- `dev/OPEN_QUESTION_extension_surface.md` §1: resolved by this plan (iterate-dependent weights are
  now leaves; the snapshot trap disappears because the leaf is converted, not copied into a key).
- `CHANGELOG.md`: `### Changed — breaking` (kinds lose `weight`; the constructors' parameter; the
  draw contract for weighted problems) + `### Added` (data-shaped weights, per-row on plain
  apply/entries, `pack_weight`, `least_squares_problem(weight=)`) + `### Changed` (leaf; compile
  sharing; float32 ulps). `docs/release_notes.md`: an "Upgrading from 2026.2.0" paragraph: *pass
  the weight to the model / problem, not the kind* — one-line migration per constructor.
- Verification record for the duplication identity if it is stated numerically in the docs.

## 9. Risks / watch-list

1. **The pytree lambdas are positional** (`fitting.py` L748–753; `backend/optimizers.py` L791–805):
   adding a leaf in one and not the other is a silent wrong answer. Change both in one commit; the
   round-trip test (§7.5) is the guard.
2. **`_is_static_leaf` and bool** (Decision 5): canonicalize before the leaf is ever partitioned.
3. **The W-gather with singleton W** — `_flat_gather` reshapes the W block to one axis; a singleton
   block must be skipped, not gathered (it would index out of range). One helper, tested.
4. **Materialization** in the mixed-singleton W case and in `pack_weight`'s cross-mode broadcast —
   memory equal to the data; documented, never silent (log at `verbose`? no — a docstring sentence).
5. **`chunk_size`** (`docs/chunking.md`): the chunked `𝒥ᵀ` on the uniform derivative probe must
   chunk the weight's W block along with the residual (or apply ω² before chunking — it is applied
   to `r` before `transpose`, so nothing changes; confirm by the chunked-vs-dense bit-identity test).
6. **Downstream**: T3Polynomial's `_newton_fit` passes `weight` positionally? (check: it is a keyword
   in `newton_cg`) — S7.
7. **Float32 jit**: ω² traced vs host-folded — ulps; state it, do not chase it.

## 10. Explicitly out of scope

- Iterate-dependent weight *schemes* (IRLS, singular-value weights) — they become possible (the leaf
  changes per step without recompiling) but no scheme ships here.
- Any weighting inside the display's relative-error tables (unweighted by design).
- Edge weights (`T3Weights` etc.) — a different object; untouched.
- Per-order minibatch slicing and per-sample gradients (`fitting_internals.md` L112–117, still deferred).
