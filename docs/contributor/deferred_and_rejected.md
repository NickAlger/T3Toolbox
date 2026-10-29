# Deferred and rejected — the ledger

Deliberately-not-built features and considered-and-rejected designs, each with its *why* and a
source pointer, so settled calls are not re-litigated and deferred ideas are not lost. (Entries
that already have their own note or live in another internals doc appear here as one-line
cross-references.)

## Rejected

- **Truncation kwargs on dense tangent projection** (`project_dense_onto_tangent`). Rejected: with
  the library's full-SVD `truncated_svd`, the t3svd-then-project path's *first* unfolding SVD costs
  `O(Nᵈ·N₀)` regardless of any tolerance — truncation only shrinks steps that are not the
  bottleneck — while the contraction method's dominant step is `O(Nᵈ·n)` with `n ≤ N₀` the Tucker
  rank. Contraction therefore dominates everywhere; the literature agrees (projection is
  contractions; the SVD belongs to retraction/rounding). The `'t3svd'` method is kept solely as an
  **independent cross-check oracle** (it is a second implementation of the same map, which is what
  verified the contraction code). Source: the t3m-swap planning session,
  `dev/archive/t3m-swap-planning_2026-06-13.md` ("keep both methods, contraction as default, skip
  the truncation options").
- **A kwarg-unified derivative interface** (`probe(..., order=)` instead of separate
  `*_derivatives` methods). Rejected (locked decision): the derivative output gains a leading
  `order` axis, so a unified method would have **return-rank polymorphism** — output rank depending
  on whether a kwarg was passed — which fights the shape-comment-as-contract style, makes generic
  consumption harder, and is worse still for the transposes (the residual *input* rank becomes
  context-dependent). A named `*_derivatives` method also self-documents an otherwise-buried
  advanced feature. Source: `dev/archive/derivatives_mirror_plan.md` §"Frontend design decision".
- **Any global / ambient jit toggle** (context manager, module flag, contextvar). Rejected: the
  bar for global toggles is very high (`safety.py`'s contextvar is the one sanctioned global,
  justified as correctness-neutral); the jit choice must be **per-function, at the call site** —
  which is what the `use_jit` per-optimizer parameters and the close-over recipe deliver. Source:
  `dev/archive/jit_oo_handoff.md`.

- **The exact Hessian by automatic differentiation** (2026.3.0). Rejected as the shipped route, kept as an
  oracle. Differentiating *through the T3-SVD* (the "level 3" Hessian of `f ∘ R`) is expensive, numerically
  fragile at the near-degenerate points continuation lives at, and unnecessary: the T3-SVD retraction is
  second order, so the Riemannian Hessian is the Lagrangian form the hand-written two-hole sweeps compute
  exactly. A `jax.jvp` of the corewise transpose and a polynomial finite difference are the **test oracles**
  (`TestCurvatureJvpOracle` in `tests/test_dispatch.py`; the block-sum and dense oracles of
  `tests/backend/_curvature_oracles.py`), never a production path — the jvp would make the numpy path
  jax-dependent and give up the factored, chunked assemblies. Source: the implementation plan's non-goals and
  X6 (the plan, archived under `dev/archive/` with the 2026.3.0 release).
- **A capability flag for the exact Hessian (`has_corewise_hessian`)**. Rejected (I-14): `corewise_hessian`
  is not optional for the feature that uses it — it is the derivative of `transpose`, part of what a kind
  *is* once exact Newton exists — so it is a **required sixth operation** with the same `NotImplementedError`
  guidance as the other five, checked structurally and early (the stdlib `abc.abstractmethod` marker) by
  `Problem.local_model(hessian='exact')`. The house's one capability flag, `has_block_sumsq`, gates a display
  that can degrade gracefully, which the exact mode cannot.
- **A `LocalContext` bundle of the kind-operation arguments** (for now). `forward` / `transpose` /
  `corewise_hessian` keep their positional `(…, sample, frame, sweep[, base_point])` arguments; bundling them
  is a later rethink of the kind protocol (Nick, 2026-10-26), not part of the exact Hessian.
- **The damping as a geometry field.** Rejected (I-22): it changes how the solver approaches the problem,
  not the set optimized over, so it is a solver option threaded to the call that reads it
  ([`parameters_not_closures.md`](parameters_not_closures.md) §"Options are not parameters").
- **Other completions for the zero-reference frame** (I-29, adopted the seeded, keyed, nested Haar recipe —
  [`curvature_internals.md`](curvature_internals.md)). (a) A coordinate-aligned completion (an identity
  prefix, or LAPACK's completion of a zero core): rejected, because it gives a reference of **exactly 0**
  on data with no energy on the leading coordinates. (b) `'warm_start'` as the default with `'zero'` opt-in:
  rejected, because it keeps the continuation deflation the zero reference was introduced to remove (every
  warm-started level ran to its cap at the default `gtol_rel`). (c) Per-mode seeds: equivalent to the
  adopted per-group keying (a singleton group is its mode). A possible later refinement, not planned:
  averaging the reference over a few seeds (each draw is a one-sweep estimate of a frame-averaged norm; the
  O(1) seed spread is that estimate's sampling variance).

## Deferred (would be built if the need materializes)

- **Per-axis uniform `sum_stack(axis=)`** (review H2-7; ruled defer + document, 2026-08-24). Ragged
  `sum_stack` takes an axis subset; the uniform twin sums the whole stack only. The gap is the mask
  algebra: a subset sum multiplies ranks by the summed sizes per element while keeping the other
  stack axes' rank variation, so the S-fold concat/gappy-mask machinery of `+` must be applied along
  a kept/summed axis split. Reachable from the existing primitives; meanwhile the ragged detour
  (`to_t3()` → `sum_stack(axis)` → `from_t3`) or unstack/restack covers it. The `sum_stack`
  docstring states the asymmetry.

- **A guard that a sampling kind and a geometry were built at the same rank.** Both now carry `shape`
  and `masks` as comparable fields, but nothing compares them. Pairing a kind built at one rank with a
  geometry at another — reachable when the two ranks have identical *padded* shapes, so no shape error
  catches it — runs to completion and produces a wrong objective. `Problem.__post_init__` could compare
  them when both are present.

- ~~**`d = 1` on the uniform layer.**~~ **Resolved 2026-08-22.** The cause was not `ut3svd_supercores`
  but `_tt_squash_tails_uniform`, which concatenated `[first, middle, last]` and so duplicated the single
  core; every uniform op that squashes (`+`, `-`, `squash_tails`, `sum_stack`, `norm`, `t3svd`, the
  frame) failed downstream. `d = 1` is now a supported *degenerate* case on both layers: a one-mode T3
  is a vector in a rank-`n` subspace, rank is not meaningful, and every op reduces to the vector case
  (`tests/test_uniform_tucker_tensor_train.py::TestUniformGarbageAndDegenerate`).

- **Stacked optimization.** A stacked point carries a batch of base points `C`, so every objective is
  an array of shape `C` -- but the optimizers' convergence tests, Armijo line searches and plateau
  detection all reduce it with a Python `float()`. All four now raise `NotImplementedError` at the
  entry rather than failing from inside the loop. The Gauss-Newton *model* supports stacked points and
  returns a shape-`C` objective, so rolling your own loop works today; making the optimizers vectorize
  over `C` (and deciding what a per-element convergence test means when elements converge at different
  rates) is the real work. Low priority.
  Regularizing a stacked point is a separate, sharper problem: the misfit keeps `C` while every
  regularizer scalar collapses it, so it raises regardless of the optimizer
  ([`fitting_and_optimization.md`](../fitting_and_optimization.md) §4.10).

- **Consolidating the sharing-partition normalization at its remaining call sites.** `None`-or-
  all-singletons → "skip the shared path" is open-coded at seven sites in `backend/ranks.py`,
  `backend/t3_svd.py` and `backend/ut3_svd.py`, each normalizing to a `None` sentinel.
  `sharing.canonical_groups` is now the named form (it normalizes to `()`, which a geometry can store
  as a field and compare by value). The seven were verified duplicative, not wrong; converting them
  means touching the SVD layer, so it was left out of the optimization-layer work.

- **A shared base for the repeated uniform-geometry methods.** `_variations`, `n_stack`, `base_point`
  and `inner` are identical across `UniformManifoldGeometryOps` and `UniformCorewiseGeometryOps`
  (~20 lines). A mixin would remove the repetition; left alone because dataclass inheritance with
  defaulted fields is finicky enough that the duplication is currently the clearer of the two.

- **Jitting the outer Newton loop.** Per iteration the host does roughly a dozen device→host syncs,
  plus one per line-search backtrack (up to 40). Noise on CPU, not on a GPU. It is downstream of the
  pytree `LocalModel` that now exists, so the prerequisite is met; what it fights is the `callback`
  diagnostic display, which reads the concrete residual host-side (`jax.debug.callback` is the likely
  route). Its own decision, deliberately kept out of the jit-boundary work.

- **The bidiagonal-`trs` jet optimization — deferred, then BUILT; finished in 2026.3.0 (no production
  kernel contracts the binomial tensor).** The pushthrough/assembly contractions convolve a full-order jet
  with the input/perturbation jet, which is capped at 2 orders (`s ∈ {0,1}`), so `trs[t,r,s]` is nonzero
  only at `r ∈ {t, t−1}` — bidiagonal, `O(order)` where the dense `(order+1)×(order+1)` einsum is
  `O(order²)`. Deferred at design time (training cost secondary, `order ≤ d` small); the measurement that
  motivated it (W=300, ranks 4): derivative apply forward ~32× the plain op at order 4, transpose ~78×.
  It was built afterwards (2026.1.0) as the recurrence/scan forms in `backend/sampling_derivatives.py`,
  **but only partway**: the mu/nu/sigma/tau pushthroughs, the eta/deta combines and the tilde sweeps
  became recurrences and order scans, while 19 production contractions kept the dense `trs` operand or a
  slice of it — the apply/entries-derivative forward and transpose, the probe transpose's `dx̃i` combine,
  and the TT gradient assembly on every branch (the W-chunked form only wrapped the dense kernel) — and the
  order scans applied `(order+1)²` weight slices per step. 2026.3.0 finished it, the standing rule now:
  **production kernels are recurrences / scan-based convolutions and correlations; dense `trs` forms live
  only in test twins.** One factor affine → a banded recurrence over the shifted pair `[x^(t), t·x^(t−1)]`
  (forward) / `[x^(s), (s+1)·x^(s+1)]` (reverse), contracted against the input jet in one GEMM — this covers
  the pushthroughs, the adjoint propagation and the TT assembly's sigma~/tau~ groups; the two-row `dx̃i`
  correlation (which now returns only the two rows the Tucker assembly reads) is that pair met by a diagonal
  order scan, for memory. Both factors full → an order scan whose binomial weights come from a shift
  recurrence on a carried jet (`D^r x = r!·C(t,r)·x^(t−r)`, `D x = t·x^(t−1)`), never a weight matrix. The
  dense binomial forms survive only as the `*_trs` reference twins in the test suite
  (`tests/_jet_trs_twins.py`, `tests/backend/_curvature_oracles.py`), never imported by the package —
  the oracle in `tests/test_jet_recurrence.py` and `tests/backend/test_curvature.py`. Measured on the change
  (uniform, CPU float32, per W-row): TT assembly scratch 4–8× smaller, `dx̃i` 13–15×, the exact Hessian's
  two-hole assembly 5–14×. Source: `dev/archive/derivatives_mirror_plan.md` §"Future-work / perf
  optimization".
- **Small-problem latency of the scanned jet kernels — recorded, accepted (2026.3.0).** The order scans run
  per core (`xmap` over the cores of an inner order scan) where the dense forms were one einsum vectorized over
  the cores, so a small jitted problem pays the sequential steps: at d=5, r=8, order 2, W=300 the jitted
  `dx̃i` kernel went 1.08 → 3.88 ms (d·(order+1) = 15 steps; eta 0.52 → 0.86 ms), the GN probe transpose
  5.5 → 8.6 ms, while large problems got faster and leaner (the TT assembly 6× faster on numpy; 4–6× less
  temporary jitted). Not a reason for a dense contraction: `dx̃i`'s length-2 axis (its two rows) is already one
  fused GEMM, and the scan is what keeps the `(order+1)·W·r·n` slice of `mu O` off the peak (12.6 GB per core
  at r=128, W=32000, order 5). The lever, if latency ever matters, is a layout one: one order scan whose bodies
  are vectorized over the cores (`d·W·r·n` per slice) instead of a map over cores of scans.
- **Structured tangent stacks (`K`) — open idea** (Nick, 2026-06-12). When the `K` tangent vectors
  carry special structure — e.g. an orthonormal block within one tangent space `T_x M` — can
  `𝒥⁽ˢ⁾` be applied more cheaply than the general 3-group sweep? Plausible angles: shared
  sub-contractions across the block, a factored form of the perturbation edge variables, work that
  cancels under orthonormality. Not pursued; the general (arbitrary-`K`) path stays the default
  regardless. Source: `dev/archive/probing_section6_notes.md`.
- **Riemannian L-BFGS with vector transport** — see the deferred list in
  [`fitting_internals.md`](fitting_internals.md) (which also records why plain L-BFGS is
  deliberately *not* a library optimizer: the scipy bridge is the intended pattern).
- **The ambient derivative transpose** — has its own analysis-and-deferral note:
  [`ambient_derivative_transpose_note.md`](ambient_derivative_transpose_note.md)
  (intrinsically exponential-rank back-projection; no use case).
- **numpy 2-operand einsum: `optimize=False` is not BLAS** (unresolved; low priority — Nick,
  2026-07-15). `_grouped_einsum`'s docstring claims *"2-operand contractions are already BLAS, so they
  pass straight through on both."* For numpy that is **false**: passing through means `optimize=False`,
  which runs `c_einsum`, never BLAS. Forcing the path (`_pairwise_path` already returns the correct
  `('einsum_path', (0,1))` for 2 operands) buys BLAS but pays a fixed ~20–25µs dispatch overhead, so
  there is a **real crossover**, measured on `'WCi,Cio->WCo'`: **0.1× at W=1, 0.7× at W=32, 1.8× at
  W=64, 9.0× at W=2048.** It loses up to 10× below W≈64 and wins up to 9× above. So the short-circuit's
  *effect* is a defensible default for small contractions — a regime-dependent tradeoff, not a bug.
  **The library's operating point sits well above the crossover** (`W` indexes the samples being fitted,
  so it is large by construction; the core dims are tied to it, `r ≈ a ≈ i ≈ b` ranging 1..`sqrt(W)/2`
  for probing, 1..`cbrt(W)/2` for entries/apply). But **numpy is not the performance path** — jax is,
  and jax's einsum has good path selection — so the bar here is "not doing something dumb", not
  "optimal". Not free either: the BLAS path is **not bit-identical** (~1.7e-16, one ULP — different
  summation order). *(Not to be confused with the MULTI-operand case, where numpy's own optimizer picks
  a FLOP-tied non-BLAS path and `_pairwise_path` is 22× faster on the 4-operand `trs` contraction. That
  machinery is doing real work and must not be touched.)* Source: `dev/archive/contractions_sharding_plan.md` §8.
- ~~**Backend `ut3_norm` / `ut3_inner` twins**~~ **Built 2026-08-22** (with the review's S11 fix: their jax
  derivative is the exact multilinear rule, not a derivative through the SVD, which was NaN on every padded
  train). The original note, for the record: (wanted eventually; low priority — Nick, 2026-07-15).
  The ragged backend has self-contained `t3_norm(x, use_orthogonalization=True)` /
  `t3_inner_product(x, y, use_orthogonalization=True)`, which orthogonalize internally. The uniform
  backend exposes only the already-orthogonalized fast path `ut3_norm_orthogonalized` (plus a
  no-orth `ut3_inner_product`), leaving the *orthogonalize-then-norm* composition to the frontend
  (`UniformTuckerTensorTrain.norm`/`.inner`). So a raw-`.data` user must know to call
  `ut3_left_orthogonalize_tt_cores(ut3_down_orthogonalize_tucker_cores(x))` first — two functions,
  in that order — which the backend/frontend razor says belongs *in* the backend (non-obvious,
  easy to get wrong, and the ragged twin already provides it). **Unlike the gaps in
  [`../naming_conventions.md`](../naming_conventions.md) §"Intended asymmetries", this one is not
  a design decision — it is an unfinished port.** Noticed while writing `ut3_weighted_norm`
  (uniform weighting S1), which had to reproduce the chain; it is parked in that module's private
  `_ut3_left_orthogonalized` helper, so filling the gap means promoting that helper into real
  `ut3_norm`/`ut3_inner` and having the frontend + weighted twins delegate. Source:
  `weighted_internals.md` (S1), `backend/ut3_linalg.py`.

- **Normal-space sector norms — deferred** (I-13, Nick 2026-10-26). The curvature term's normal-space operand Z
  (the misfit gradient's ambient lift) splits into 2^d sectors `E_S = ⊗_k (U_k^⊥ if k ∈ S else U_k)`. The tangent
  space sees `∅` (the core) and the singletons `{k}` (the Tucker slots). So the normal part decomposes as
  `‖N_core‖² = ‖P_∅Z‖² − ‖H^grad‖²`, `‖N_k‖² = ‖P_{k}Z‖² − ‖V_k^grad‖²`, and `‖N_{≥2}‖²` as the remainder. Per
  sector, they would say where the data pull leaves the manifold: a rank-continuation signal and a curvature
  diagnostic. Two routes: an exact Gram route, `O(m²dN)`, for apply/entries; and unbiased rank-one Gaussian
  probing, `E[⟨Z, P_S(⊗y_j)⟩²] = ‖P_S Z‖²`, for every kind. Deferred because the always-on diagnostics (Δ, ρ, κ,
  σ_min) cover the solver's needs, and rank continuation already reads the t3svd unfolding condition numbers.
- **The Newton schedule — an open, empirical question** (I-36, Nick 2026-10-27). Gauss-Newton vs the exact
  Hessian as the continuation default, when to switch between them, the forcing power: to be settled by an
  A/B run over a battery of examples, after the implementation. Nick leans **exact with the warm start** (a
  warm-started level starts much closer to the quadratic basin than the cold starts S0b measured); still
  missing are GPU timings and the Hessian-apply cost weighed against the per-iteration retraction and frame
  construction. **Built now: the override hooks** — every schedule knob is a per-call `newton_cg` keyword, so a
  continuation loop (and a future `fit(...)` facade) can run a per-level schedule; cold / zero starts remain a
  user option (I-37). The inputs gathered so far (the experiment records, archived under `dev/archive/`; numbers
  in [`verification_records.md`](verification_records.md)): on the restart template Gauss-Newton's rate
  constant `ρ(H_GN⁻¹K)` at the minimizer is ≈ 0.7 shared or unshared, exact's gain is the same size either way
  (3.5–5× fewer outer steps, 5–6.6× fewer CG, 1.6–2× wall), and tying more factors leaves the rate flat;
  downstream (T3Polynomial) exact is superlinear in the tail (6 vs 30+ steps) but from zero-padded *cold*
  starts it can pick a worse basin and is radius-limited early. Thin: one setup, numpy, a handful of seeds.
- **Exact-Newton refinements** (2026.3.0; each reachable from the shipped pieces). *Storing* the
  residual-dependent curvature state per model instead of recomputing it per Hessian-apply (I-11: ~+60 % of the
  jet sweep resident for ~10–15 % of the two-hole; the T/S transforms of
  [`curvature_internals.md`](curvature_internals.md) are how); a **preconditioner** for the inner CG, LM damping
  or an NL2SOL-style β (the trust region bounds the step, not the CG count; 3–8 Hessian-applies per step were
  measured); a **per-mode** lift damping `ε`; reusing the shared companion's **centers** across the projection
  and the lift (I-21: not reused, recomputed by zipper); a **curvature-carrying regularizer hook** (m19: the
  curvature uses the misfit gradient because `∇ρ` is tangent for ridge and weight decay; a Grasedyck–Kramer
  prior would need an optional `Regularizer.normal_gradient`); an **η floor** and an **objective-plateau stop**
  (J-12: the trust region's `objective_floor` covers the resolution limit, not a slow plateau); `tr_radius0`
  carried across continuation levels (today each level recomputes `Δ0`); a zero **start** using the canonical
  zero-point frame too (J.13 / I-31: it would make zero-start trajectories layer-independent, but changes
  today's zero-start numerics — left as is, to investigate); routing the frontend exact `hessian(p)` through
  the backend `hvp` (one projection instead of two: bitwise equal to the backend instead of to rounding).
- **No W-chunked path for the apply / entries (and their derivative) transposes and two-holes** (2026.3.0).
  `chunk_size='auto'` resolves to `None` for the five kinds other than `probe_derivatives`, in both modes: their
  Gauss-Newton transposes have no chunked path, so neither do their exact two-holes, which peak at ~2× the
  unchunked transpose (jax uniform f32, measured 1.85–2.1×). The probe-derivative chunking pattern
  (`_wchunked_sum` over module-level chunk bodies) is the recipe if a large-`|W|` apply fit ever needs it.
- **Chunk-estimator edges** (backlog). `_wchunked_sum` roughly doubles the peak when `W` leaves a near-full
  remainder chunk (pre-existing, Gauss-Newton too); and on current XLA a leftover chunk (`W` not a multiple of
  `chunk_size`) is allocated on top of the full chunk, so `max_chunk_size_within` can overshoot its budget by up
  to ~2× (found on the jax-0.4.30 CI leg). The balanced policy is unaffected in the usual case (it returns `W`).
- **The uniform tied projection's clip** (S9, a follow-up slice). `ufv_share_tucker_variations` — the shared
  uniform manifold's Gauss-Newton `project` — still clips at the PADDED dims and does not mask padded `s_g`
  levels, unlike the tied lift (which clips at the REAL dims × the group rank mask). A level between the two
  clips (~1e-15 relative) is kept by the lift and the ragged layer but dropped by the projection — 17 % off
  ragged at a near-singular point, which the equivalence contract exempts — and the projection is not robust
  to garbage in a *hand-built* companion's padded levels (no library path builds one; at the default damping
  the Hessian cannot see the difference). Deferred because the fix touches Gauss-Newton code and needs its own
  bit-identity run.
- **The operand-scaled gauge check's follow-ups** (S7b; [`curvature_internals.md`](curvature_internals.md)).
  Carry the operand scale through `normalized` / `sum_tangents` / `unstack_tangents`; give `UT3Tangent` a
  `__truediv__` (ragged has `/`); and a faster `corewise_stack_norm` would remove most of the safe-mode `±`
  cost (a leaf add 55 → 314 µs at 3.2k dof).
- **Small dtype / dispatch leftovers** (backlog). **A float32 fit comes back float64** (numpy, or jax under
  x64) on every `MANIFOLD` fit and most others, though the jet kernels preserve float32 since 2026.3.0: 38
  default-dtype constructors in eight backend modules sit upstream of them -- the plain `probe` / `apply` /
  `entries` scan inits, `tt_squash_tails`'s tail vectors (so `t3_orthogonal_representations` builds the frame
  in float64), the TT zipper boundaries, the tangent-to-T3 block builders, the uniform T3-SVD and
  shared-retraction paddings, `to_dense`'s tail vectors. Passing the operands' dtype at each site keeps a
  float32 numpy fit float32 end to end (checked 2026-10-29 over the six kinds × both layouts × manifold /
  corewise / shared × both Hessians) and leaves float64 bit for bit; it is its own slice because each scan
  init must take the dtype of ALL its scan's operands (a jax carry may not change type under x64 with
  mixed-precision inputs), which wants a mixed-dtype jax test per site. Separately,
  `sharing.fv_share_tucker_variations` infers the redistribution's backend from the variations only but the
  solve's from all inputs (a mixed numpy / jax call returns numpy through a jax float32 solve — harmless,
  but the two should agree).
