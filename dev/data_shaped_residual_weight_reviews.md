# Data-shaped residual weighting — review record and revised recommendation

> **STATUS: DEFERRED (Nick, 2026-09-03).** The plan `dev/data_shaped_residual_weight_plan.md` (v1,
> "single mechanism, clean cut") was reviewed by two independent agents with identical briefs and no
> shared context; their findings are consolidated in §1, the design consequence in §2, the revised
> recommendation for when this is picked up in §3, and the two reports verbatim in §4–§5. Nothing is
> implemented. Pick up here, not from the plan's v1 decisions.

*Context: T3Polynomial's eigen example (its `dev/eigenproblem_design_2026_09_02.md` §8) needed per-row
and per-component control of the residual weight ω. Nick's proposal: weights that are an exact
structural match for the data blocks (broadcasting makes the (mode, order) matrix the small special
case). The plan chose ONE mechanism (the matrix becomes sugar for a traced leaf) with a breaking cut
at the backend surface. The reviews found the breaking surface larger than the plan assumed and two
of its rules wrong against its own locked decisions.*

## 1. Consolidated findings

Severities as the reviewers gave them; "both" = found independently by both. Key claims were verified
against the code (the topt example, the recorded rejection of return-arity polymorphism, the
model-inside-jit recipe, `tree_to_jax(None)`, the bit-exactness warning, T3Polynomial's positional
backend call).

**Blockers (both):**

1. **Decision 4 (a weighted problem with a 2-tuple `draw` is an error) breaks a shipped frontend
   path.** `examples/fit_hilbert_from_apply_derivatives_topt.py` (L103–108, L166–168) passes
   `weight=omega` AND a custom 2-tuple `draw`; `docs/fitting_and_optimization.md` §2.3/§4.4 document
   the same. Contradicts Decision 2 (non-breaking frontend). Fix under the v1 design: a weight whose
   W block is all-singleton (every sugar weight) needs no slicing — `Problem._batch` reuses it; the
   error fires only for a full-W weight with a 2-tuple draw.
2. **S7's "T3Polynomial green with no source change" is false, and the failure is silent.**
   `t3polynomial/fitting.py:84` calls `bfit.probe_derivatives_kind(jet_order, weight)` positionally;
   after the field removal the matrix lands in `chunk_size`, and the ragged path never inspects
   `chunk_size` for list inputs (`sampling_derivatives.assemble_tucker_variation_jets`), so
   `_zero_gradient_norm` would return the UNWEIGHTED norm without raising. Fix: the kind's
   `__post_init__` rejects a non-int `chunk_size` naming the removed parameter; patch downstream;
   CHANGELOG names the positional shift.
3. **A traced weight must survive canonicalization** (B: BLOCKER; A: CONSIDER). The documented
   "build the model inside your own `jax.jit`" recipe (`fitting.py` L46–58) means a weight can arrive
   as a tracer; `np.asarray(w, dtype=float)` snapshots a jax weight to a host constant — the trap
   `dev/OPEN_QUESTION_extension_surface.md` §1 names, which the plan claims to remove — and raises on
   a tracer. Fix: canonicalize with the input's backend (`get_backend`); shape checks only under a
   trace; the finite check frontend-only and host-only (it is a VALUE check, not "structural, both
   modes" — both reviewers).

**Should-fix (both):**

4. **Sugar vs data-shaped classification is unstated and shapes collide**: a per-row `(W, 1)` vs the
   rejected `(d, 1)` when `W = d`; `(1, order+1)` vs `(1, W)`; a single `(order+1, W)` array read as
   `ω[mode, order]` when `order+1 ≥ d`. Rule proposed: probe kinds — one array is always sugar,
   data-shaped is always a length-`d` list; scalar kinds — 1-D is sugar, ≥2-D data-shaped (drops the
   undocumented `(1, order+1)` spelling). (Dissolves under the separate-kwarg design, §2.)
5. **Decision 6's scalar weight contradicts the equal-ndim rule** — canonicalize 0-d to all-singleton.
6. **Validate against the data-block shape derived from (kind, sample)**, never against `data` or
   the residual: the frontend factories hold a residual that carries C, and a backend `Problem.data`
   may carry C too (`tests/backend/test_optimizers.py` L735–739).
7. **`tree_to_jax(None)` raises** — `_prepare_jit_inputs` must skip a `None` weight; `_jit_engaged`
   treats `None` as absent; `partition_static` round-trips `None` (precedent `geom_aux`).
8. **Test inventory incomplete** — every test that builds a weighted KIND directly must be rewritten:
   `tests/test_fitting.py` L392, L442–520, L521–553 (`TestPerModeWeightRowCount`);
   `tests/backend/test_uniform_fitting.py` L256–258 (`_weight_matrix`/`_make_weight`), L270, L419–442;
   `tests/backend/test_optimizers.py` L153–160, L724–727, L760–763, L772 (positional `chunk_size`),
   L782; `tests/test_dispatch.py` L444–456 (asserts ω is a host constant), L651–685 (the
   "must not share a compilation" assertion FLIPS), L721–758; `tests/test_optimizers_frontend.py`
   L183–190 (plain apply `weight=[1.0]` rejection flips under Decision 6). `tests/oracle_sweeps/` go
   through the frontend and survive via sugar.
9. **Docs inventory** beyond §8: `fitting_and_optimization.md` L21, L114–117, L168 (draw contract),
   L213 (`*_derivatives_kind(order, weight)`), L238 (`Problem(...)`), L306–314; `optimizers.py` L273
   and `backend/optimizers.py` L447, L492 draw shape-comments; `uniform_fitting.py` L5–8, L60–61,
   L83 and `_UniformKind.from_point`'s "order / weight / chunk" docstring; `CLAUDE.md` L409–414;
   `parameters_not_closures.md` and `OPEN_QUESTION_extension_surface.md` L39–41 ("a kind at order 2
   with weight ω is a distinct compiled program" becomes false); stale comments in `test_dispatch.py`;
   `release_notes.md` L38–40 left as history.
10. **Simplification (both, independently):** apply the weight at the ~9 model call sites
    (`backend/optimizers.py` L122, L136, L147, L154, L213; `fitting.py` L324, L336, L367, L383, L399)
    and leave the kinds' `sumsq` / `transpose` signatures untouched — a user's custom kind gets
    weighting for free. A goes further: every residual is `(lead)+W+K+C+(trail)` with
    `trail ∈ {0, 1}`, so one two-valued `n_trail` ClassVar on the existing `ScalarOutputKind` /
    `ProbeOutputKind` bases replaces `n_lead` AND `_order_axis` / `_mode_axis` / `_bare_binds_to`;
    one module-level `gather_weight(weight, idx, n_w, n_trail)` replaces edits to all twelve
    `_take_*` / `_ptake_*`; `apply_weight(x, w, power, n_trail)` needs no `n_w`.
11. **Naming** — `apply_weight` collides with the `apply` sampling op; `scale_residual` /
    `weight_residual`.

**One reviewer, endorsed:**

12. (B) **Return-arity polymorphism** of `take`/`flat_draw` (2-tuple iff unweighted) contradicts a
    recorded rejection (`deferred_and_rejected.md` L20–26). Always return a 3-tuple internally with
    `weight_B=None`; normalize user draws at the step callers.
13. (A, verified) **`jnp.take` on a size-1 axis with out-of-range indices returns NaN silently**
    (numpy raises). Any singleton-W skip in the gather must be pinned on the jax path in
    `test_dispatch`, or `mc_sgd(use_jit=True)` with a sugar weight would NaN silently on a regression.
14. (A) Build `GaussNewtonModel`'s pytree flatten/unflatten from `dc.fields` with a declared aux set;
    the positional lambdas (`fitting.py` L748–753) are a silent-wrong-answer hazard.
15. (A) `uniform_least_squares_problem(weight=)` accepting "sugar or data-shaped" is a second
    mechanism at the backend surface — expand in the frontend `_setup` (which knows `d`, `order`,
    `n_w`, the representation) and hand the backend canonical weights only.
16. (B) The §7.1 bit-identity fixture "captured from `main`" is platform-bound
    (`test_uniform_fitting.py` L449 warns against bit-exactness across BLAS); use today's
    reshape-and-multiply as an in-test oracle on the same machine.
17. (B) **Mixed backends**: `np_residual * jax_weight` silently returns a jax array — state "the
    weight follows the data's backend; a mismatch is not an error but flips dispatch" or reject in
    `_batch`. Canonical dtype should follow the data's dtype, not always float64.
18. (B) Recompiles on a weight SHAPE change and on `None ↔ present` (treedef change) — document in
    §4.5; `w**2` per matvec is a data-sized elementwise op (negligible; a second `w²` leaf if ever
    profiled).
19. (A) The §1 claim that the GN residual "never carries K" is wrong (`GaussNewtonModel.jacobian(p)`
    with a K-stacked tangent yields `W+K+C`); the C-insertion rule still works because K and C are
    contiguous after W — state it and add a K-stacked case to §7.7.
20. (A) NIT: the uniform sugar reshape `ω[:d].reshape((d, o)+…)` fails for `m = 1` → `(min(m, d), o)`;
    `probe_kind()` becomes `PROBE` (decide on the alias); archive `per_mode_weighting_plan.md` per the
    CLAUDE.md routing rule once this lands.

**Declined / deferred:** (A) materialize mixed-singleton W at gather time via a `broadcast_to` view
rather than at canonicalization — cheaper, numerically identical, touches a locked decision; noted.

**Examples:** with (1) fixed, all five `weight=` examples run unchanged through the sugar path
(`fit_per_mode_weight_probes.py`, `fit_probe_display.py`, `fit_hilbert_uniform_probe_derivatives_
newton_cg.py`, `fit_shared_factors_jetted_probes.py`, `fit_hilbert_from_apply_derivatives_topt.py`);
no example prose becomes wrong. List all five as the S5 regression gate.

## 2. What the reviews change in the design

The single mechanism's real breaking surface is not "two constructors and a field": it is every
direct backend construction of a weighted kind (tests, T3Polynomial, a silent positional mis-bind),
a shipped example through the frontend, the dispatch test and two doc statements about the device
constant, and the numpy bit-identity of existing callers (their code path changes). Its one concrete
benefit — compile sharing across weights — is needed only for the weight that changes, the data-shaped
one.

**Nick's question (2026-09-03): what dissolves if the data-shaped weight is a SEPARATE kwarg?**
Two readings:

- *Split at the API only, one mechanism underneath* (matrix expanded into the same leaf): dissolves
  the classification family (4, 5, the dropped `(1, order+1)` spelling) and 15. Both blockers, 3, 6,
  7, 8, 9 and the breaking entry remain — they come from the mechanism, not the API.
- *Split at the API and keep the matrix where it is* — the matrix stays a static kind field exactly
  as today, the new kwarg (`residual_weight=`, name TBD; never `weights`) is the only leaf, and the
  effective weight is their PRODUCT, one extra elementwise multiply. Dissolves by construction: both
  blockers (existing callers never hold a leaf; kind constructors keep their signature), the breaking
  entry and upgrade paragraph, the dispatch-test flip, the doc statements, existing callers'
  bit-identity and the float32-ulp question, and 13 (a data-shaped weight can be required to carry
  the full W block, so the gather is a plain gather and no singleton rule exists). What remains is
  the new path's own work: the leaf on `Problem` / `LocalModel` / `GaussNewtonModel` and their pytree
  entries, the gather in the draw (a leaf with a 2-tuple draw is then simply an error on a NEW API),
  backend-aware canonicalization (3), the finite check host-side, validation against (kind, sample)
  (6), the `None` skip (7), the C-axis insertion, the mixed-backend rule (17), the K statement (19),
  and the simplifications (10, 14) which still apply.

## 3. Revised recommendation (for pickup)

Adopt the **separate kwarg + product** design and ship it as a non-breaking `### Added`:

- `weight=` unchanged (the `(mode, order)` matrix, static on the kind, host-numpy, as documented).
- New `residual_weight=` (final name per `naming_conventions.md`; singular) on the six frontend
  factories, `newton_cg` / `mc_sgd` / `adam` / `lbfgs`, `least_squares_problem`,
  `uniform_least_squares_problem`: per kind a data-shaped operand (§3 of the plan) with **full W**
  required (no singleton W; other axes may be 1), float, ndim equal to the data block's, validated
  against (kind, sample); carried as a leaf on `Problem` / `LocalModel` / `GaussNewtonModel`
  (appended fields with `None` defaults; pytree from `dc.fields`), converted like sample/data under
  jit (skip `None`), gathered by `flat_draw` (3-tuple always internally; a `residual_weight`ed problem
  with a custom 2-tuple draw → structural error naming the third element), packed by `pack_weight`
  beside `pack_data` (zero fill; finite garbage inert).
- Applied at the nine model call sites, product with the kind's static ω, via one module function
  (`scale_residual(x, w, power, n_trail)`) with `n_trail ∈ {0, 1}` on the two output base classes; the
  kinds' protocol untouched.
- Tests: the duplication identity (√2 on a row ≡ the row duplicated) on objective / gradient /
  gn_quadratic / gn_hessian; explicit-indexing oracles per-row and per-component for probe_derivatives
  (both geometries) and per-row WLS on plain apply/entries (Decision 6 kept: the plain kinds gain the
  new kwarg only); ragged/uniform equivalence with garbage-padded weight slots; one compile for
  same-shape different-valued leaves; `None ↔ present` recompiles once; a traced leaf inside a user
  `jax.jit`; the C-stacked and K-stacked cases; the draw-tuple mismatches; every existing weight test
  UNCHANGED and green (the gate that the matrix path did not move).
- Docs: §4.6 gains a "data-shaped residual weight" subsection and the product statement; the draw
  contract's optional third element; `precompute_and_caching.md`'s scope-ladder entry becomes true
  for the new kwarg; `OPEN_QUESTION_extension_surface.md` §1 resolved for iterate-dependent weights
  via the new kwarg (the snapshot trap stays documented for the static matrix, where it is correct).
- CHANGELOG: `### Added` only.

Re-cut the plan's §5 slices on that basis; S0 (inventory) and the gate list stand.

## 4. Reviewer A — report (verbatim)

<details><summary>expand</summary>

# Review: `dev/data_shaped_residual_weight_plan.md` (reviewer A)

Read: the plan, the per-mode precedent, all listed contributor/user docs, the six code modules, the tests, every `weight=` example, and T3Polynomial's caller. Two empirical checks were run read-only (below). Severity key: BLOCKER / SHOULD-FIX / CONSIDER / NIT; each tagged **wrong** / **underspecified** / **taste**.

## 1. Simplicity

**C1 (CONSIDER, taste but substantive) — the whole layout question collapses to one bit, and that removes `n_lead`, the per-kind `take` edits, and the kind-protocol change.** Every residual layout in the plan's §1 table has the form `(lead)+W+K+C+(trail)` with `trail ∈ {0, 1}` (a free-mode axis for probe kinds, none for scalar kinds), and the canonical weight is `(lead)+W+(trail)` with equal ndim. So the W block of a weight starts at `w.ndim − n_trail − n_w`, and the extra `K+C` axes are inserted at `w.ndim − n_trail`. Nothing needs `n_lead ∈ {0,1,2}`: `n_trail` is a two-valued ClassVar on the *existing* `ScalarOutputKind` / `ProbeOutputKind` bases (`backend/fitting.py:416,430`), inherited by all uniform kinds, so `_order_axis` / `_mode_axis` / `_bare_binds_to` (`:399-401`, `uniform_fitting.py:135-136,208-209`) all go. Consequences: (a) `gather_weight(weight, idx, n_w, n_trail)` is one module function (list → per element; singleton block → skip) — none of the six `_take_*` / six `_ptake_*` need touching; `flat_draw` calls `kind.take` then `gather_weight`. (b) `apply_weight(x, w, power, n_trail)` needs no `n_w`, so `transpose` (which is not handed `n_w`) needs no `w_axes(sample)` call. (c) You can then leave the kinds' `sumsq` / `transpose` signatures **unchanged** and apply the weight in the nine model call sites (`backend/optimizers.py:122,136,147,154,213`; `fitting.py:324,336,367,383,399`) — the kind protocol a user subclasses is untouched, a user kind gets weighting for free, and the six `self._apply_weight(r, 2)` wrappers become zero edits. This is not among the locked decisions, and it is strictly less surface than §4.

**C2 (CONSIDER) — materialize mixed-singleton W at gather time, not canonicalization.** `broadcast_to` is a view; `np.take` on the view materializes only the drawn rows, and under jit the draw runs outside the kernel. Materializing on the `Problem` costs a data-sized copy that the full-batch optimizers never use. Locked, so only flagging that the alternative is numerically identical and cheaper.

**NIT** — `apply_weight` reads as "the apply kind's weight"; `scale_residual` or `weight_residual` avoids the collision with the `apply` op.

## 2. Correctness

**The math survives** — `sumsq(ω⊙·)` and `transpose(ω²·)` are the only hooks; with the insertion rule above, `K` (from a K-stacked `p` in `hvp`/`gn_hessian`: `sampling_derivatives.py:2134` documents `(order+1,)+W+K+C+(Ni,)`) and `C` are contiguous after W, so one rule covers both. Chunked `𝒥ᵀ`: ω² is applied to `r` *before* `tv_probe_transpose_derivatives_from_sweep` (`backend/fitting.py:644-646`), and the chunker slices the already-weighted `ztildes` (`sampling_derivatives.py:1733-1745`) — unaffected; the existing `test_chunk_size_does_not_change_the_gradient` (`tests/backend/test_optimizers.py:772`) covers it once it takes a weight. Display, `_ScaledRegularizer`, `SharedGeometry` (weight enters before `Π`) — unaffected. Numpy bit-identity is plausible: the per-element ops are the same square-then-multiply; only the reshape moves.

**B1 (BLOCKER, plan is wrong) — Decision 4 as written breaks a shipped frontend path, contradicting Decision 2.** `examples/fit_hilbert_from_apply_derivatives_topt.py:103-108,166-168` passes `weight=omega` **and** a custom 2-tuple `draw`; so does the documented contract at `docs/fitting_and_optimization.md:21,158-168`. Under "a weighted problem given a 2-tuple draw is a structural error" that call raises. Fix: apply the plan's own gather rule at the draw boundary — a weight whose W block is all-singleton (every sugar form) needs no slicing, so `Problem._batch` reuses `self.weight` when the draw omits it; only a full-W weight with a 2-tuple draw is the error. That keeps every existing `weight= + draw=` call working and makes the error fire exactly where slicing is impossible.

**B2 (BLOCKER, plan is wrong) — S7's "green with no source change" is false, and the failure is silent.** `t3polynomial/fitting.py:84` builds `bfit.probe_derivatives_kind(jet_order, weight)` *positionally*. After the field removal `ProbeDerivativesKind`'s positional order is `(order, chunk_size)`, so `weight` lands in `chunk_size`. On the ragged path `assemble_tucker_variation_jets` (`sampling_derivatives.py:1733`) returns `dense()` when `etas` is a list **before** ever inspecting `chunk_size`, so nothing raises: `_zero_gradient_norm` silently returns the *unweighted* `‖g₀‖`. Fix: `ProbeDerivativesKind.__post_init__` rejects a non-`int`/`None` `chunk_size` with a message naming the removed `weight` parameter (structural, jit-safe); S7 patches L84; CHANGELOG names the positional shift.

**S1 (SHOULD-FIX, underspecified with real collisions) — how an input is classified as sugar vs data-shaped is never stated, and shapes collide.** (a) Plain probe: §2.7 "rejects 2-D" vs §6 "one array applied to every mode — accept": a per-row `(W, 1)` is 2-D, and is indistinguishable from the rejected `(d, 1)` when `W == d`. (b) apply/entries derivatives with `n_w = 1`: today's silently-accepted 2-D `(1, order+1)` sugar (`fitting.py:163-164`) collides with a data-shaped `(1, W)` when `W == order+1`. (c) probe_derivatives: a list of `(order+1, W)` arrays (trailing 1 forgotten) passes `np.asarray` → `(d, order+1, W)`... fine, but a single `(order+1, W)` array is read as `ω[mode, order]` whenever `order+1 ≥ d` (`fitting.py:171`). Recommended rule: probe kinds — one array is always sugar, data-shaped is always a length-`d` list (drop the "one array for every mode" convenience; `[w]*d` is one keystroke); scalar kinds — 1-D is sugar, ≥2-D is data-shaped (drops the undocumented `(1, order+1)` spelling; no example or doc uses it — a one-line `Changed — breaking`).

**S2 (SHOULD-FIX, inconsistent)** — Decision 6 accepts a scalar; §3/§6 require `ndim == data ndim`. Resolve by canonicalizing a 0-d input to `(1,)*ndim`.

**S3 (SHOULD-FIX, underspecified)** — `canonical_residual_weight(weight, data, kind, n_w)` takes `data`, but the frontend factories have only the residual, which carries `C`. Validate against the data-block *shape* derived from the sample + kind (exactly what `_check_ragged_sample` computes, `fitting.py:491-531`), so `Problem` (data) and model (residual) paths share one validator.

**S4 (SHOULD-FIX, test targeting)** — verified: `jnp.take` on a size-1 axis with out-of-range indices returns **NaN silently** (numpy raises `IndexError`). The singleton-skip in the gather (§9.3) must therefore be pinned on the **jax** path in `test_dispatch`, or `mc_sgd(use_jit=True)` with a sugar weight would NaN silently if the skip regresses.

**NIT** — §3's uniform sugar `ω[:d].reshape((d, o)+…)` fails for `m = 1`; it is `(min(m, d), o)`.

## 3. Consistency with conventions

- **S5 (SHOULD-FIX, contradicts Decision 1)** — `uniform_least_squares_problem(weight=)` "sugar or data-shaped" (§4) is a second mechanism at the backend surface. `_setup` knows `d`, `order`, `n_w`, and the representation; expand there and hand the backend only canonical weights.
- **C3 (CONSIDER)** — the NaN/inf rejection is a *value* check; `fitting_internals.md` §"Backend-first" keeps the backend free of numerical checks, and a `bool(np.isfinite(...).all())` on a traced weight (a raw-`.data` user building a `Problem` inside jit) raises `ConcretizationTypeError`. Put it in the frontend only, host-numpy only; the uniform contract's "fill must be finite" is stated, not enforced, so this matches precedent.
- Value-hashed kinds / leaf-vs-aux / `precompute_and_caching.md`'s scope ladder: consistent. Worth stating in `parameters_not_closures.md` that the plan resolves `OPEN_QUESTION_extension_surface.md` §1 by keeping the weight **off the kind** (data-side), not by making the kind a pytree as that note feared.
- Numerical contracts: `GaussNewtonModel` gains no precondition; the float32 ulp statement belongs in `numerical_contract_catalog.md` as the plan says. `deferred_and_rejected.md`: nothing contradicted.
- **NIT** — `_prepare_jit_inputs` / `_jit_engaged` must special-case `None` (`tree_to_jax(None)` → `jnp.asarray(None)` raises; `tree_contains_jax(None)` is `False`).

## 4. Unforeseen consequences

- **Dispatch**: `partition_static` classifies a 0-d/singleton float array as dynamic (`common.py:686-695`) — correct; a `None` weight flows through `partition_static` as an empty pytree node, so `None↔present` changes the treedef and recompiles once (correct). A numpy weight against jax data promotes correctly under x64-off; under x64-on a hand-built `Problem` with float32 jax data and a float64 numpy weight promotes the residual to float64 — same as today, mention in the docstring. **C6**: `np.asarray(w, dtype=float)` on a jax weight is a device→host→device round trip and fails on a tracer; canonicalize with the input's `xnp`.
- **JIT**: `_jitted` stays memoized on the module-level steps; `weight_B=None` vs array gives distinct treedefs, hence distinct compiles — right. Sugar leaves have a fixed tiny shape regardless of `m > d` truncation, so compile-once across continuation holds. `lax.while_loop` carries the weight inside `lm` as a loop invariant — free.
- **Positional pytree lambdas** (§9.1): **C4** — build `GaussNewtonModel`'s flatten/unflatten from `dc.fields` with a declared aux set; kills the hazard rather than guarding it.
- **Stacking**: a sugar leaf `(o,1,…,1)` against a `W+C` residual reproduces today's `reshape((o,)+(1,)*(x.ndim−1))` exactly — the `TestWeightCombinations` stacked test (`tests/backend/test_optimizers.py:720-745`) is the right gate.
- Display, stopping criteria, regularizer scaling, sharing: nothing new (per-row weights change `‖g₀‖`'s scale exactly as sugar does today).

## 5. Examples

With **B1** fixed, all five run unchanged through the sugar path: `fit_per_mode_weight_probes.py:88-91`, `fit_probe_display.py:78-84`, `fit_hilbert_uniform_probe_derivatives_newton_cg.py:144-167`, `fit_shared_factors_jetted_probes.py:130-131`, `fit_hilbert_from_apply_derivatives_topt.py:166-168`. Without it, the last one breaks. No example prose becomes wrong; `fit_per_mode_weight_probes.py:20-21` ("apply/entries have no per-mode axis") stays true. The plan's S5 lists only one example — list all five as the regression gate.

## 6. Tests / docs

**S6 (SHOULD-FIX, incomplete inventory)** — tests that construct weighted *kinds* directly must be rewritten, beyond the two the plan names: `tests/backend/test_uniform_fitting.py:256-258` (`_weight_matrix`/`_make_weight`), `:419-442` (`factory(O, WEIGHT)`); `tests/backend/test_optimizers.py:726-727,782` (`bfit.probe_kind(omega)`, `probe_derivatives_kind(order, omega)`) and `:772` (positional `chunk_size`); `tests/test_fitting.py:480-504,521-546` (`TestPerModeWeightRowCount` uses `bfit.probe_kind(weight=)`); `tests/test_dispatch.py:444-456` (asserts the weight is a host constant — now a traced leaf; assert `tree_leaves(model)` contains it instead — `_leaves_all_jax(hp)` at `:757` checks the *output*, not the model); `tests/test_optimizers_frontend.py:187-189` (plain apply `weight=[1.0]` is now an accepted per-row broadcast — the assertion flips). Add: the S1 collision rejections (isolated, per `testing_strategy.md`'s rule), the B2 `chunk_size` guard, and the S4 jax-path singleton gather.

**Docs** the plan omits: `fitting_and_optimization.md:21,168` (draw contract), `:213` (`*_derivatives_kind(order, weight)`), `:238` (`Problem(...)`); `optimizers.py:273` and `backend/optimizers.py:447,492` draw shape-comments; `uniform_fitting.py:5-8,60-61,83` and `_UniformKind.from_point`'s "order / weight / chunk" docstring; `CLAUDE.md:409-413`; `release_notes.md:38-40` (superseded by the new upgrade paragraph — leave as history, do not edit). Doctests: `fitting_and_optimization.md:76-86,346-359` all use sugar and stay green. Verification record: only if the duplication identity is stated numerically, as the plan says.

</details>

## 5. Reviewer B — report (verbatim)

<details><summary>expand</summary>

# Review: `dev/data_shaped_residual_weight_plan.md` (reviewer B)

Verdict up front: the leaf design is sound and the two-hook math survives. There are four places where the plan as written contradicts its own locked decisions or a documented promise, and several underspecified seams that will bite at implementation time. Findings are tagged **[wrong]**, **[underspecified]**, or **[taste]**.

## 1. Simplicity

**S1 — SHOULD-FIX [taste, but it shrinks the breaking surface].** The plan threads `weight` through every kind's `sumsq` / `transpose` (12 method signatures across two layers, `backend/fitting.py:422-440,553-646`; `uniform_fitting.py:151-227`). The kinds only contribute *layout* (`n_lead`); the multiply itself is layout-generic. Alternative: keep the kind protocol untouched, and have the ~9 consumer sites (`LocalModel.misfit/gradient/gn_quadratic/hvp`, `Problem.objective`, `GaussNewtonModel.objective_value/gradient/gn_quadratic/gn_hessian/evaluate`) call one module-level `apply_weight(x, w, power, kind.n_lead, n_w)` *before* the unchanged `kind.sumsq(out, n_w)` / `kind.transpose(r, ...)`. Same math, one function, no signature change for custom kinds, and `dev/OPEN_QUESTION_extension_surface.md` §"price" gets no wider.

**S2 — SHOULD-FIX [wrong].** The `take`/`flat_draw` "2-tuple iff unweighted, 3-tuple iff weighted" is return-arity polymorphism, which `docs/contributor/deferred_and_rejected.md:20-26` rejects as a locked decision for exactly the reasons that apply here (generic consumption, shape-comment contract). Return `(sample_B, data_B, weight_B)` always with `weight_B=None`; accept a user 2-tuple by normalizing `*draw(rng)` in `_mc_sgd_step`/`_adam_step` callers.

**S3 — CONSIDER [taste].** The mixed-singleton-W materialization at canonicalization (§3, risk 4) exists only so `_flat_gather` (`backend/fitting.py:152-162`) has one rule. A gather that handles singleton axes directly needs no materialization: `ix = np.unravel_index(idx, W)`, index each W axis with `ix_k` if `W_k > 1` else `0`. Verified: `w[:, ip, 0, :]` on a `(o, NP, 1, Ni)` weight equals `broadcast_to(...).reshape(o,-1,Ni)[:, idx]` bit-for-bit, and is jit-safe. The W rule then collapses to plain broadcasting semantics, with an all-singleton fast path that returns `w` unchanged (which S2/§4.2 below depends on).

**S4 — NIT.** `n_lead` is the minimal layout fact; keep it. `pack_weight`'s cross-mode broadcast (§3) could instead reject mixed per-mode singleton patterns (one clear structural error) rather than document a materialization; either is fine. `apply_weight` as a name reads as "the apply kind's weight" in a module whose other `apply` is a sampling op; `scale_by_weight` avoids it.

## 2. Correctness

The two hooks are the only places: `evaluate` (`fitting.py:389-403`) composes `c + ⟨g,Πp⟩ + ½·sumsq(Jp)` and becomes `½‖ω⊙(r+Jp)‖²` once `sumsq` is weighted; `hvp`/`gn_hessian` get ω² via `transpose`; the line search reads `Problem.objective` (`backend/optimizers.py:203-216`, `743`) which goes through `sumsq`; the display reads unweighted `block_sumsq` and compares to `info.misfit` (`optimizer_display.py:226-232`), so `(unwt …)` fires for any nontrivial weight; `_ScaledRegularizer` (`regularization.py:99-118`) is weight-blind; chunking slices the already-weighted `r` along W (`sampling_derivatives.py:1908-1930`), so bit-identity holds; `SharedGeometry.project/precompute` act on tangents only. The C-insertion rule (§3) checks out for all seven layouts, including apply/entries with a stacked point (`x = W+C`, `w = W`, insert after `n_w`).

**C1 — SHOULD-FIX [wrong].** §1 says the GN residual "never carries K". `GaussNewtonModel.jacobian(p)` with a K-stacked tangent (`t3m.COREWISE.randn(frame, stack_shape=(3,))` is legal) yields `W+K+C` and feeds `sumsq`/`transpose` (`probing.py:1163`, `sampling_derivatives.py:2133` accept `W+K+C`). The insertion rule happens to be correct anyway because K sits between W and C and the inserted unit axes cover `K+C` jointly; state that, and add a K-stacked case to §7.7.

**C2 — SHOULD-FIX [underspecified].** `canonical_residual_weight(weight, data, kind, n_w)` validates "ndim equal to the data block's" against `data`. But (a) the frontend factories receive a *residual* that carries C (`_check_ragged_sample`, `fitting.py:519-531`), and (b) the backend `Problem.data` may itself carry C (`tests/backend/test_optimizers.py:735-739` builds `stacked_data` with a C axis for `least_squares_problem`). So `data.ndim` is not the data-block ndim at either call site. The expected canonical shape is fully determined by `(kind, sample)` — `n_lead` from the kind, W from `kind.w_axes(sample)`/sample shapes, `Ni` from `sample[i].shape[-1]` — so validate against that, never against `data`.

**C3 — SHOULD-FIX [wrong, internally inconsistent].** Decision 6 accepts "a scalar weight … no special case", but §3 requires ndim equal to the data block. A 0-d array violates the rule. Either the sugar path reshapes scalars to the all-singleton canonical form, or the ndim rule admits 0-d. Related: `_setup` (`optimizers.py:201-203`) and `uniform_sampling_kind` (`uniform_fitting.py:252-256`) currently *reject* weights on plain apply/entries, and `tests/test_optimizers_frontend.py:183-190` asserts the rejection; both flip under Decision 6 and neither is in §7.

## 3. Consistency with conventions

**K1 — BLOCKER [wrong].** Decision 5's `np.asarray(w, dtype=float)` plus the finite check contradicts §8's claim that "the snapshot trap disappears" and §10's "iterate-dependent weights become possible". A jax weight passed to `least_squares_problem` or a `*_model` factory is pulled to host and snapshotted — the exact trap `OPEN_QUESTION_extension_surface.md:103-107` describes — and a *traced* weight (a model built inside a user `jax.jit`, the documented recipe at `fitting.py:48-57`) raises on `np.asarray`. Canonicalize with the inferred backend (`get_backend(False, is_jax_ndarray(w))`; bool → `astype(xnp.result_type(float))`, which does not warn under x64-off), do shape checks only (jit-safe), and make the finite check host-only/eager-only, as the house structural-vs-numerical rule requires — a value check cannot be "structural, both modes" on a tracer.

**K2 — SHOULD-FIX [underspecified].** The sugar-vs-data-shaped disambiguation at the frontend is not stated. It is unambiguous by ndim for probe (sugar ≤ 2-D, data ≥ 2-D with a trailing `Ni`) and probe_derivatives, but for plain apply/entries with `n_w ∈ {1,2}` a 1-D/2-D array is *both* a candidate matrix and a candidate data-shaped array. Rule to write down: plain apply/entries have no sugar beyond the scalar; any array there is data-shaped.

**K3 — SHOULD-FIX [wrong].** The precedent `dev/per_mode_weighting_plan.md` and the plan's own §6 say "structural → hard error, both modes". `NaN/inf rejection` is a value property; call it what it is and scope it to eager numpy inputs (see K1).

**K4 — CONSIDER.** `docs/contributor/parameters_not_closures.md` and `OPEN_QUESTION_extension_surface.md:39-41` both state that a kind "at order 2 with weight ω is a distinct compiled program". That becomes false; §8 lists the former but not the latter. `tests/test_dispatch.py:444-445, 518-521, 756-757` carry now-stale comments about the ω aux; update with the assertion flips.

**K5 — NIT.** `probe_kind` (`backend/fitting.py:661`) becomes `ProbeKind` with zero fields, i.e. `probe_kind()` is `PROBE`; decide whether the alias survives. `dev/per_mode_weighting_plan.md` should move to `dev/archive/` per the CLAUDE.md routing rule once this lands.

## 4. Unforeseen consequences

**U1 — BLOCKER [wrong; conflicts with Decision 2].** Decision 4 makes "a weighted problem given a 2-tuple draw" a structural error. `examples/fit_hilbert_from_apply_derivatives_topt.py:103-108,165-168` passes `weight=omega` *and* a custom 2-tuple `draw` to `topt.mc_sgd`. That is a shipped, documented example (`docs/fitting_and_optimization.md:497-498`) and it breaks — through the frontend, which Decision 2 promises is non-breaking. Fix, and it costs nothing: when `Problem.weight`'s W block is all-singleton (every sugar weight, hence every existing user), a 2-tuple draw needs no slicing; `_batch` should reuse `self.weight` and raise only when the stored weight actually carries W. Combined with S2/S3 this is one predicate, not a special case.

**U2 — BLOCKER [wrong].** §5 S7 and risk 6 say T3Polynomial "must be green with no source change". `t3polynomial/fitting.py:84` calls `bfit.probe_derivatives_kind(jet_order, weight)` — the backend constructor, which loses its `weight` parameter. S7 needs a one-line downstream change (`least_squares_problem(..., weight=weight)`), and the plan should say so.

**U3 — SHOULD-FIX [underspecified].** `tree_to_jax(None)` raises `ValueError` (verified); `_prepare_jit_inputs` (`backend/optimizers.py:301-317`) must skip a `None` weight, and `_jit_engaged` must treat `None` as "not present". `partition_static` with a `None` leaf round-trips fine (precedent: `geom_aux`).

**U4 — CONSIDER [underspecified].** Mixed backends: `np_residual * jax_weight` silently returns a jax array (verified), so a jax weight on numpy data flips the dispatch of everything downstream of `sumsq`/`transpose` without error. State the rule ("the weight follows the data's backend; a mismatch is not an error but the result is jax") or reject the mismatch in `_batch`. Also dtype: a float64 weight on float32 numpy data upcasts the residual; today's constant does the same, but the canonical dtype should probably follow the data's dtype rather than always `float`.

**U5 — CONSIDER.** Recompiles: a weight shape change, and `None ↔ present` (treedef change), each recompile; the module-level `_jitted` wrappers stay stable. Document both in §4.5. Memory: `w**2` is recomputed per matvec as a data-sized elementwise op; negligible next to the contractions, but a second `w²` leaf would remove it if it ever shows in a profile.

## 5. Examples

Unchanged via sugar: `fit_per_mode_weight_probes.py:88-91`, `fit_shared_factors_jetted_probes.py:130-131`, `fit_probe_display.py:82-83`, `fit_hilbert_uniform_probe_derivatives_newton_cg.py:165-167` (uniform, per-order). Their prose stays true (mode weighting is still a probe-only *concept*; only the mechanism generalizes). **Breaks under the plan as written:** `fit_hilbert_from_apply_derivatives_topt.py` (U1). Prose that becomes wrong: `fit_per_mode_weight_probes.py:20-21` ("apply/entries … have no per-mode axis to weight") is still true, but `fit_probe_display.py:12-13` and `fit_hilbert_from_apply_derivatives_topt.py:11-18` describe ω only as per-order; fine as-is, optionally a pointer to the new form. `docs/fitting_and_optimization.md:114-117, 213, 306-314` and `CLAUDE.md:409-414` describe ω as the matrix owned by the kind and must change (both pages are doctested).

## 6. Tests and docs

**T1 — SHOULD-FIX [wrong].** §7.1's "fixture of reference values captured from `main`" checked with `np.array_equal` is platform-captured bit-exactness, which `tests/backend/test_uniform_fitting.py` itself warns against ("never rely on bit-exactness … different hardware / BLAS"). Instead copy today's `_make_weight` reshape-and-multiply into the test as the oracle and compare `array_equal` on the same machine; that is the actual claim.

**T2 — SHOULD-FIX [underspecified].** The plan's test list omits the concrete rewrite inventory of tests that construct weighted kinds directly, all of which stop compiling: `tests/test_fitting.py:392, 442-520, 541-553`; `tests/backend/test_uniform_fitting.py:256-258, 270, 442` (uses the private `_weight_matrix`/`_make_weight`); `tests/backend/test_optimizers.py:153-160, 724-726, 760-763, 782` (note `probe_derivatives_kind(order, None, chunk_size)` positional); `tests/test_dispatch.py:651-685, 721-758`; `tests/test_optimizers_frontend.py:183-190`. `tests/oracle_sweeps/` go through the frontend only and survive via sugar — confirm the "verbatim" property is intact.

**T3 — SHOULD-FIX.** Missing tests for the plan's own motivation and edge cases: a model built inside a user `jax.jit` with a *traced* data-shaped weight (K1); a backend `Problem` whose `data` carries C together with a data-shaped weight (C2); a K-stacked tangent through `gn_quadratic` with a weight (C1); a custom 2-tuple draw on a sugar-weighted problem still running (U1); `_leaves_all_jax` applied to the *model's* flattened leaves, not just outputs (`test_dispatch.py:75-84` checks outputs only); the plain-apply per-row WLS oracle plus the flipped rejection test.

**T4 — NIT.** `docs/fitting_and_optimization.md` §4.6 and `getting_started`-adjacent pages are CI-doctested; §8 should name the `sphinx -W` and page-doctest runs as gates for S6, and `dev/OPEN_QUESTION_extension_surface.md` §1 should be marked resolved only if K1 is fixed (otherwise the trap it names still exists).

**Summary of what to change in the plan before implementing:** U1 and K1 are the two places the plan is wrong against its own decisions; U2 is a wrong downstream claim; S2 conflicts with a recorded rejection; C2/K2/U3 are gaps an implementer would have to guess at.

</details>
