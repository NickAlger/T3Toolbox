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

## Deferred (would be built if the need materializes)

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

- **The bidiagonal-`trs` jet optimization — deferred, then BUILT; now the standard form.** The
  pushthrough/assembly contractions convolve a full-order jet with the input/perturbation jet, which
  is capped at 2 orders (`s ∈ {0,1}`), so `trs[t,r,s]` is nonzero only at `r ∈ {t, t−1}` —
  bidiagonal, `O(K)` where the dense `(K+1)×(K+1)` einsum is `O(K²)`. Deferred at design time
  (training cost secondary, `K ≤ d` small); the measurement that motivated it (W=300, ranks 4):
  derivative apply forward ~32× the plain op at order 4, transpose ~78×. It was built afterwards as
  the recurrence/scan forms in `backend/sampling_derivatives.py` — `compute_mu_jets` is a fused
  two-term recurrence with no `trs` operand at all, and the genuinely-full convolutions
  (`eta`/`deta`/tilde) scan the summed order axis instead. Those are the **defaults** at every call
  site; the dense binomial forms survive as the `*_trs` reference twins (`compute_mu_jets_trs`,
  `compute_eta_jets_trs`, …), the oracle in `tests/test_jet_recurrence.py`. Source:
  `dev/archive/derivatives_mirror_plan.md` §"Future-work / perf optimization".
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
- **Backend `ut3_norm` / `ut3_inner` twins** (wanted eventually; low priority — Nick, 2026-07-15).
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
