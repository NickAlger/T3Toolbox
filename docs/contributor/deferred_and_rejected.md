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

- **The bidiagonal-`trs` jet optimization.** The pushthrough/assembly contractions convolve a
  full-order jet with the input/perturbation jet, which is capped at 2 orders (`s ∈ {0,1}`), so
  `trs[t,r,s]` is nonzero only at `r ∈ {t, t−1}` — bidiagonal. The code computes these as a dense
  `(K+1)×(K+1)` einsum (`O(K²)`) where the math is `O(K)`; exploiting it (two slice-contractions)
  would cut the order-mixing by `~K/2` on those contractions (only the *combine* in probe is a
  genuine full `O(K²)` convolution). Design-time measurement (W=300, ranks 4): derivative apply
  forward ~32× the plain op at order 4, transpose ~78×. Deliberately not built: training cost is
  secondary, `K ≤ d` is small, and it trades away the clean uniform-`trs` abstraction. Source:
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
