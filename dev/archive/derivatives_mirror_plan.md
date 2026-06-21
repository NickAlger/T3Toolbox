# Mirroring the probing surface with derivative versions — plan & handoff

**Branch:** `probe-derivatives`. **Status:** planning complete (incl. the stacking correction below);
Slice 1 starting.
**Companion docs:** `docs/probe_derivatives_handoff.md` (the *done* probe-derivative work),
`docs/symmetric_probe_derivatives.tex` (the math), `docs/entries_apply_probe.md` (the three sampling
ops), `docs/batching_and_stacking.md` (**read first for anything with stack axes**),
`docs/transposes.md` (ambient/corewise/tangent taxonomy),
`docs/derivative_order_information_and_conditioning.md` (why fit from derivatives — deferred thought
experiment).

## Goal & priority

Symmetric directional derivatives (`d^t/ds^t` of a sampling op along one repeated direction `P`) are a
**core library feature**, mirrored across the **entire** probing surface — every `probe`/`apply`/
`entries` function and method gets a `*_derivatives` analog, in **both backend and frontend**, with
**full stacking** (same as probing). Not a probe-only add-on. The fitting example is *nice-to-have*;
the Hessian-conditioning experiment is a *thought experiment* for later — neither gates this work.

The existing `backend/probe_derivatives.py` covers only the **probe** column, is **single-tangent (no
`K` stack)**, and has **no frontend**. This plan fills the grid *and* retrofits `K`-stacking.

## Stacking — the axes (READ THIS; it is the error-prone part)

A derivative probe has **four** kinds of axis. Mirrors probing exactly, plus the convolved order axis:

| axis | batches | lives on | notes |
|---|---|---|---|
| **`order`** `t` (size `order+1`) | the derivative orders `0..order` | every edge-variable jet | **convolved** by the binomial `trs` (not a passive rider); placed **outermost** |
| **`W`** sample stack | a batch of paired `(X, P)` samples | input vectors `ww`/`pp` **only** | `X`=`ww` (points) and `P`=`pp` (directions) **share one stack** — each sample is a paired `(x,p)`. This **is** probing's probe stack `W`. Already supported. |
| **`K`** tangent stack | a batch of `T3Tangent` vectors at one shared base | variation cores **only** | what regular `probe_tangent` supports. **To be added** (was wrongly deferred). |
| **`C`** base stack | a batch of T3 base points | every core (`= stack_shape`) | already supported |

Layout (base-inner, mirroring `probe_tangent`'s `W+K+C`): **`order + W + K + C + (Nᵢ,)`**. Base jets
(`xi, mu, nu, eta`) carry `order + W + C` (no `K`); variation jets (`sigma, tau, deta, dxi`) and
variation cores (`dG, dU`) carry the `K`. Euclidean plain-T3 forwards have **no `K`** (no tangent),
only `order + W + C`.

**Naming:** use **`W`** for the sample stack (NOT `S` — the contractions already use `W`; fix the
`S→W` prose drift in `probe_derivatives.py`). Rename the order-count local from `K` to **`order`** (it
collided with the tangent stack `K`); keep lowercase `t,r,s,u` for order indices.

## The full mirror grid

Derivative analog of each probing variant. ✓=done, ⟳=exists-but-needs-`K`-retrofit, **NEW**=to build.
Backend functions all live in `backend/probe_derivatives.py`:

| op | fwd Euclidean (`TTT`) | fwd Riemannian (`T3Tangent`) | transpose: tangent | transpose: corewise | transpose: ambient |
|---|---|---|---|---|---|
| **probe** | `probe_derivatives_t3` ✅ | `probe_tangent_derivatives` ✅ | `probe_tangent_derivatives_transpose` ✅ | `probe_corewise_derivatives_transpose` ✅ | ⬜ DEFERRED |
| **apply** | `apply_derivatives_t3` ✅ | `apply_tangent_derivatives` ✅ | `apply_tangent_derivatives_transpose` ✅ | `apply_corewise_derivatives_transpose` ✅ | ⬜ DEFERRED |
| **entries** | `entries_derivatives_t3` ✅ | `entries_tangent_derivatives` ✅ | `entries_tangent_derivatives_transpose` ✅ | `entries_corewise_derivatives_transpose` ✅ | ⬜ DEFERRED |

All ✅ cells exist in the backend **and** the frontend (`TuckerTensorTrain`/`T3Tangent`), full `W+K+C`,
verified to ~1e-16. The ambient column is **deferred** — see
[`docs/ambient_derivative_transpose_note.md`](ambient_derivative_transpose_note.md).

Dense oracles: `probe_derivatives_dense` ✓; `apply_derivatives_dense`, `entries_derivatives_dense` **NEW**.
Frontend (currently **zero** derivative methods): `TuckerTensorTrain` + `T3Tangent` methods for every
cell above, `(ww, pp, order)` / `(index, pp, order)` signatures.

## Locked design decisions

1. **Tangent transpose algorithm = adjoint-state (approach A).** The apply/entries tangent transpose
   is the jet-ified adjoint-state Lagrangian with the per-mode residual gone: the scalar residual jet
   `ρ` **seeds one propagation sweep at the terminal bond**. Reuses every existing `trs` contraction
   for the 2-block case; `K` adds the order-threaded 3-block versions (below). (Rejected the "direct
   multinomial scatter" needing a new 4-leg `trsq`.)
2. **Full `W + K + C` stacking, mirroring probing** (not `vmap`/map-over-`K`). The perturbation sweeps
   + assembly use **order-threaded 3-block contractions** — the existing 2-block jet contractions
   (`trs_rWCa_Caib_sWCi_to_tWCb`) with `K` added exactly as `WKCa_Caib_WCi_to_WKCb` extends
   `WCa_Caib_WCi_to_WCb`. Split self-infers (`C`-only core pins `len(C)`, `W+C` jet pins `len(W)`,
   `K`=remainder; variation-core-only terms take `n_base`). **These are missing from `contractions.py`
   — to be written.** Each reduces to the 2-block version when `K=()`.
3. **Naming = `W` (sample) / `order` (count) / parallel `*_derivatives`** (see Stacking + grid).
4. **Module = keep all `*_derivatives*` in `backend/probe_derivatives.py`** (don't re-scatter).

## The order-threaded 3-block contractions (new in `contractions.py`)

Forward perturbation sweep + assembly (8): sigma/tau pushthrough
`trs_rWKCa_Caib_sWCi_to_tWKCb` (K on sigma), `trs_rWCa_KCaib_sWCi_to_tWKCb` (K on dG, takes
`n_base`), `trs_rWCa_Caib_sWKCi_to_tWKCb` (K on dxi); deta combine `trs_rWKCa_Caib_sWCb_to_tWKCi`,
`trs_rWCa_KCaib_sWCb_to_tWKCi` (`n_base`), `trs_rWCa_Caib_sWKCb_to_tWKCi`; lift `tWKCi_Cio_to_tWKCo`,
`tWCi_KCio_to_tWKCo` (`n_base`, order a passive broadcast — no `trs`). Transpose adds the order-threaded
3-block adjoint-sweep + assembly contractions (Slice 2; count TBD, mirrors the non-order probe
transpose's "10 outer-product builders"). All get dense/loop-oracle tests in `backend/test_contractions.py`.

## Frontend design decision (locked): SEPARATE methods, not kwargs on probe/apply/entries

The derivative ops get their **own** frontend methods (parallel to the backend), **not** optional
`perturbation`/`order` kwargs on the existing `probe`/`apply`/`entries`. Rationale (discussed): the
derivative output gains a leading `order` axis, so a unified method would have **return-rank
polymorphism** (output rank depends on whether a kwarg was passed) — which fights the house
shape-comment-as-contract style, makes generic consumption harder, and is gnarlier still for the
transposes (residual *input* rank becomes context-dependent). A named `*_derivatives` method also
self-documents an otherwise-buried advanced feature. The frontend *is* allowed to differ from the
backend, but here matching it is the better call. (The X/P stack consistency check is needed either
way, so it isn't a differentiator.)

**Frontend method names** (forwards take `(ww, pp, order)` / `(index, pp, order)`, all required):
- `TuckerTensorTrain`: `probe_derivatives`, `apply_derivatives`, `entries_derivatives` (forward);
  `{probe,apply,entries}_corewise_derivatives_transpose`. (`*_ambient_derivatives_transpose` deferred.)
- `T3Tangent`: `probe_derivatives`, `apply_derivatives`, `entries_derivatives` (forward);
  `{probe,apply,entries}_derivatives_transpose` (tangent; static, returns `T3Tangent`).
- **Consistency check (structural → hard error):** the forwards validate that `pp`'s stack matches
  `ww`'s (or `index`'s) `W`.

## Slicing

- **Slice 1 (in progress): K-retrofit the derivative FORWARD + apply/entries forward.** The 8
  order-threaded 3-block contractions; make `compute_{sigma,tau,deta}_jets` + `assemble_tangent_z_jets`
  `K`-aware (⟳ → `probe_tangent_derivatives` gains `K`); add `apply_derivatives_t3` (Euclidean, `W+C`),
  `apply_tangent_derivatives` (Riemannian, `W+K+C`, reuse the `K`-aware sigma sweep terminal + bond
  sum), and the `entries` forward siblings; dense oracles `apply_derivatives_dense` /
  `entries_derivatives_dense`; tests incl. `K`-stacked cases. *Implement+verify apply first.*
- **Slice 2: tangent transposes with `K`.** Order-threaded 3-block adjoint contractions;
  `probe_tangent_derivatives_transpose` `K`-retrofit; `apply/entries_tangent_derivatives_transpose`
  (adjoint-state, seeded sweep); tests (`jax.linear_transpose` + adjoint identity + `Lᵀ` cross-check).
- **Slice 3 (separate methods — see decision above): corewise wrappers + frontend hookup.**
  - **3a: backend corewise derivative wrappers** — `{probe,apply,entries}_corewise_derivatives_transpose`
    (the `P,Q,O→G` substitution into the tangent derivative transpose). Verify vs `jax.grad` of the
    forward w.r.t. the cores.
  - **3b: `T3Tangent` frontend** — 3 forwards + 3 tangent `*_derivatives_transpose` + doctests + tests.
  - **3c: `TuckerTensorTrain` frontend** — 3 forwards + 3 corewise transposes + doctests + tests.
- ~~**Slice 4: ambient derivative transposes**~~ — **DEFERRED (not needed).** The base-free adjoint's
  back-projection is inherently exponential-rank (elementary-symmetric combinations of `wⱼ`/`pⱼ`),
  unlike the clean rank-`d`/rank-1 non-derivative ambient; least-used flavor; no use case. Full analysis
  + implementation sketch in [`docs/ambient_derivative_transpose_note.md`](ambient_derivative_transpose_note.md).

## Algorithm notes

**Forward apply (Riemannian):** `mu_jets` (via `P`, base, no `K`) → `K`-aware sigma-jet sweep to its
**terminal carry**, sum the terminal bond → `order + W + K + C`. Reuses `_sigma_jet_step` (factor out
of `compute_sigma_jets`). **Forward apply (Euclidean):** terminal `mu`-jet via `G`, bond summed (`W+C`).

**Transpose apply (Riemannian), adjoint-state:** base `xi/mu` sweep; **`sigma_hat` sweep** — propagation
-only adjoint via `Q`, seeded at the terminal bond by `ρ` (the residual carries `K`, so `sigma_hat`
carries `K`); `dxi_hat = mu ⋆ O ⋆ sigma_hat`; assemble `dG_tilde = mu ⊗ xi ⊗ sigma_hat`,
`dU_tilde = dxi_hat ⊗ w_jet`. **Index caveat:** `dG_tilde_i` pairs with `sigma_hat_i` = adjoint of the
*after-step-i* carry; `sigma_hat_d` is the `ρ` seed — pin with the oracle, not the hand-derivation.

**Entries:** apply-derivatives with one-hot base vectors + general direction `P` (Taylor data at grid
corner `index`). Base `xi` order-0 from fiber slicing `U_i[…,index_i]` (reuse `probing._entry_xis`),
order-1 from `U_i p_i`; `dU_tilde` scatter lands on the indexed rows.

## Verification strategy

- **Forward:** dense all-modes subset-expansion oracle (`apply/entries_derivatives_dense`:
  `y^(t)=t!·Σ_{|S|=t} T(p in S, w else)`, all modes contracted) — checked per `(W,K,C,order)` element.
  Plus containment cross-check `apply^(t) = ⟨z_m^(t), w_m⟩ + t·⟨z_m^(t-1), p_m⟩` from the probe-derivative
  jets. Plus order-0 == `apply_tangent`.
- **Transpose:** `jax.linear_transpose` of the (variation-linear) forward + dense adjoint identity
  `⟨ρ, J v⟩ = ⟨Jᵀ ρ, v⟩` (`sum_over_probes=True`), to ~1e-16 across `W×K×C×order`, both
  `sum_over_probes`. Plus `apply_transpose(ρ) == probe_transpose(Lᵀρ)` (`Lᵀ ρ` builds a probe-residual
  jet `ztilde_m^(t) = (1/d)[ρ^(t) w_m + (t+1) ρ^(t+1) p_m]`).
- Tests: `tests/test_probe_derivatives.py`, `backend/test_contractions.py`; jit dispatch in
  `tests/test_dispatch.py`.

## Doc-update todos (as slices land)

- `docs/entries_apply_probe.md` §4 table is **stale** (says "probe has no ambient transpose", omits
  corewise) — refresh to the three-flavor grid + a derivative/jet dimension.
- `docs/symmetric_probe_derivatives.tex` — add the apply/entries specialization and the `K` stack
  (its §8 "Remaining" defers both; both now in scope).
- `docs/probe_derivatives_handoff.md` + CLAUDE.md "Current state" — keep in sync; reverse the
  "tangent stack `K` deferred" line.

## Status / progress log

- [x] Slice 1: 8 order-threaded 3-block fwd contractions — in `contractions.py`, tested in `backend/test_contractions.py` (`_check_jet3`, 8 tests; explicit-einsum + `K=()` reduction + multi-axis `W`).
- [x] Slice 1: `K`-retrofit `compute_{sigma,tau,deta}_jets` + `assemble_tangent_z_jets` + `_sigma_jet_step` factored + `K`→`order` rename — `probe_tangent_derivatives` now full `W+K+C`, verified vs dense oracle.
- [x] Slice 1: `apply_derivatives_t3` + `apply_tangent_derivatives` + `apply_derivatives_dense` — dense oracle + containment cross-check + order-0 == `apply_tangent`.
- [x] Slice 1: `entries_derivatives_t3` + `entries_tangent_derivatives` (fiber-slice base via `probing._entry_xis` + general `P`) + `entries_derivatives_dense` — dense oracle + order-0 == `entries_tangent`.
- [x] Slice 1: tests folded into `tests/test_probe_derivatives.py` (`K`-stacked forward, apply, entries; 11 tests) + `backend/test_contractions.py` + `tests/test_dispatch.py` (jit: K-stacked fwd, apply/entries, new 3-block contractions). **Full suite green (242).**
- [x] Slice 1: `S→W` prose fix (module header + docstrings) + module header rewritten for `W/K/C/order`. **Deferred to Slice 2 pass:** adding explicit `K` to the per-arg variation-jet shape comments (`W+C`→`W+K+C`) in `compute_{sigma,tau,deta}_jets`/`assemble_tangent_z_jets` (header already states it; done when the transpose retrofit touches every function).
- [x] Slice 2a: **probe transpose `K`-retrofit.** 15 order-threaded 3-block ADJOINT contractions (5 sweeps + lift wrapper, 10 gradient-assembly outer products via `_assemble_dG_jet3`/`_assemble_dU_eta`/`_assemble_dU_dxi`); `probe_tangent_derivatives_transpose` now full `W+K+C` (contraction swaps + the `uWKCa_uWo` self-pin dropping `n_probe`). Verified: adjoint identity + `sum_over_probes` consistency + `jax.linear_transpose` (~1e-16, all `W/K/C/order`). Tests in `backend/test_contractions.py` (`_check_jet3` generalized for `trs_sub`/`n_probe`; 15 methods), `test_probe_derivatives.py` (K-transpose adjoint identity), `test_dispatch.py` (jit). **Full suite green (258).**
- [x] Slice 2b: **apply/entries adjoint-state tangent transposes.** `compute_sigma_hat_jets` (the `ρ`-seeded propagation-only sweep via `Q`, no `deta_tilde` source) + `apply/entries_tangent_derivatives_transpose` (reuse the Slice-2a contractions; `dG=mu⊗xi⊗sigma_hat`, `dU=dxi_hat⊗w_jet`, `dxi_hat=mu*O*sigma_hat`; ~half the probe transpose). Full `W+K+C`. Verified vs adjoint identity + `sum_over_probes` consistency + `jax.linear_transpose` (~1e-16, all stacks). Tests in `test_probe_derivatives.py` + jit in `test_dispatch.py`. **Full suite green (259).**
- [x] Slice 2: per-arg variation-jet/tilde `K` shape-comments updated (`W+C`→`W+K+C` for `sigma/tau/deta/dxi` jets + `*_tilde` + Riemannian `z_jets`/`ztildes`); base jets (`mu/nu/xi/eta`) and Euclidean stay `W+C`; stale "single tangent" docstring fixed.
- [x] Slice 3a: backend corewise derivative wrappers (`{probe,apply,entries}_corewise_derivatives_transpose`) — verified vs `jax.grad` + numpy finite-difference test. Committed `4136e6a3`.
- [x] Slice 3b: `T3Tangent` frontend — `{probe,apply,entries}_derivatives` + `{…}_derivatives_transpose` (static, → `T3Tangent`) + doctests + `test_manifold` test. Shared `check_perturbation_{vectors,index}` (backend, jit-safe) for the X/P stack guard.
- [x] Slice 3c: `TuckerTensorTrain` frontend — `{probe,apply,entries}_derivatives` + `{…}_corewise_derivatives_transpose` + doctests + `test_tucker_tensor_train` test. **Full suite green (262); all doctests pass.**
- [x] ~~Slice 4: ambient transposes~~ — **DEFERRED (not needed)**; analysis in `docs/ambient_derivative_transpose_note.md`.
- [x] **Doc refresh** — `entries_apply_probe.md` (§4 grid), `symmetric_probe_derivatives.tex` (apply/entries +
  `K` stack), CLAUDE.md "Current state", `probe_derivatives_handoff.md`. Done before the review.
- [x] **`/code-review high` pass + agreed fixes (committed `4ac6205b`).** Tier 1: (#1) deleted the dead 2-block
  transpose contractions in `contractions.py` — superseded by the 3-block `W+K+C` versions; kept
  `tWCo_Cio_to_tWCi` (still delegated to); removed 14 stale `__all__` names. (#3) added the numpy
  `optimize=True` branch to `_assemble_dG_jet3`. (#4) fixed 6 variation-core shape comments (`C+`→`K+C+`).
  Tier 2: consolidated `_w_jets`→`build_input_jets` (deleted the helper; dropped the now-unused
  `xnp`/`get_backend` in the apply/entries transposes); added structural validation to the **6 transpose
  frontend methods** (`check_perturbation_{vectors,index}`), and extended `check_perturbation_index` with an
  optional ambient-shape (`Ni`) check wired through `self.shape`/`basis.shape` on every entries frontend
  (forward + transpose). **Left alone (Nick's call):** the `is_uniform` notes (uniform layer still WIP),
  unguarded invariants, `_assemble_dG_jet3`'s dict-dispatch style. Tests: `test_probe_derivatives` +
  `test_manifold` + `test_tucker_tensor_train` + `test_contractions` (173) and `test_dispatch` (8) all green.
- [x] **Derivative fitting example** — `examples/fit_hilbert_from_apply_derivatives.py`: fits a T3 to
  the Hilbert tensor from apply-**derivative** data (orders 0..d, per-order normalized + unit-norm probes),
  Riemannian Newton-CG + rank continuation. **Converges and recovers the tensor** (true err 5e-3 at rank
  4, val picks rank 4); exercises the frontend `apply_derivatives`/`apply_derivatives_transpose` end-to-end
  — the integration test passed, no issues uncovered. **Review now unblocked.**
- Deferred nice-to-have (Nick's call): the Hessian-conditioning experiment
  (`docs/derivative_order_information_and_conditioning.md`).
- **Future-work / low-priority cleanup:** `_assemble_dG_jet3`'s `{'A','I','B'}`-keyed dispatch (vs the
  codebase's explicit-per-contraction style) — revisit later.
- **Future-work / perf optimization (bidiagonal `trs`):** the pushthrough/assembly contractions convolve
  a full-order jet with the *input/perturbation* jet, which is capped at 2 orders (`s ∈ {0,1}`), so
  `trs[t,r,s]` is nonzero only at `r ∈ {t, t-1}` — **bidiagonal**. We currently compute them as a dense
  `(K+1)×(K+1)` einsum (`O(K²)`) where the math is `O(K)`. Exploiting it (two slice-contractions:
  `core·jet⁽ᵗ⁾` + `t·core·jet⁽ᵗ⁻¹⁾`) would cut the order-mixing by `~K/2` on those contractions (only the
  *combine* in probe is a genuine full `O(K²)` convolution). Measured (W=300, ranks 4): deriv apply fwd is
  ~32× regular at order 4, transpose ~78×. Not urgent (training cost secondary; `K ≤ d` small; trades away
  the clean uniform-`trs` abstraction) — log, don't block the merge.

## Session-end handoff (2026-06-18)

**Where we are.** The derivative grid is *complete and reviewed*: forwards + tangent/corewise transposes
for probe/apply/entries, full `W+K+C` stacking, backend + both frontends, all verified (~1e-16) and tested.
The `/code-review high` pass is done and every agreed fix is committed (`4ac6205b`). The branch
`probe-derivatives` is **green** (`test_probe_derivatives` + `test_manifold` + `test_tucker_tensor_train` +
`test_contractions` = 173, `test_dispatch` = 8) and **not yet merged to `main`**.

**Two open threads before/at merge:**

1. **DISCUSS — base-jet caching (a real feature Nick wants, for regular probing *and* derivatives).**
   In a Riemannian solve the base is *fixed* across the inner CG, but every `J`/`Jᵀ` call recomputes the
   base sweeps (`xi_jets`, `mu_jets`, `nu_jets` and their regular-probing analogues). Nick: "this is a
   feature I have wanted to add... for both regular probing AND derivatives. Let's talk about this." The
   shape of the design (a "Jacobian-operator-at-fixed-base" object / a reuse hook that precomputes the
   base sweeps once and exposes `apply`/`transpose` closures) is **undecided — lead with the reasoning,
   not a bare menu** (CLAUDE.md workflow). This is the next conversation to have.

2. **Merge `probe-derivatives` → `main`.** Nick said the review had to happen "until after we have added
   the fitting example" — both are now done, so the merge is unblocked once (1) is resolved or explicitly
   deferred. The example (`examples/fit_hilbert_from_apply_derivatives.py`) converges end-to-end.

**Deferred (logged, not blocking):** ambient derivative transposes (`docs/ambient_derivative_transpose_note.md`,
exponential-rank → not needed); the bidiagonal-`trs` perf optimization (above); the Hessian-conditioning
experiment (`docs/derivative_order_information_and_conditioning.md`); `_assemble_dG_jet3` dict-dispatch
restyle. Tangent stack `K` for the Euclidean/ambient transpose and the project-once gather opt are in
`symmetric_probe_derivatives.tex` §8.
