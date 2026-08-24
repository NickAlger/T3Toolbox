# OPEN QUESTION — the uniform frame at a numerically rank-deficient point (review S1b)

> **RESOLVED 2026-08-23.** Nick's pad-safe SVD packet (Method D, sketch–project) was implemented:
> `backend.linalg.pad_safe_svd` plus the mask-threaded uniform sweeps — the frame sweep AND
> `ut3svd`'s own sweep, unshared and SF-T3 shared. All six S1b cases report 0 lost directions on
> both frame paths; regression tests `tests/backend/test_linalg.py` and `TestPadSafeFrame` in
> `tests/test_uniform_frame_variations_format.py`. The uniform frame is now gauge-equivalent (no
> longer bit-identical) to ragged — recorded in `docs/uniform_equivalence_contract.md`
> §"Gauge-carrying operations" and the CHANGELOG. The measured answers to the note's open points:
> the ε·c accuracy worry was refuted (`s1b_c_study.py` — pins occupy bitwise-zero rows, the
> augmented core is bitwise block-diagonal), and the `n ≥ m` feasibility corner dissolved (the
> symmetric `min(n, m)` contract needs no runtime transpose — `s1b_sym_variant.py`).

_Opened 2026-08-23 (Nick + Claude, during the 2026.2.0 pre-release review). Unresolved: nothing here is
implemented. A standing question — not a thread; do not archive until resolved. Measurements and scripts:
`dev/review_2026-08-22/repros/S1b/` (see its README for the prior-art survey with sources)._

## The problem, in one paragraph

On the uniform layer, every orthogonalization step is an SVD of a zero-padded unfolding. A padded row and a
*real* row that happens to be zero are indistinguishable to the factorization, so for a singular value that is
exactly (or numerically) zero the left singular vector — an arbitrary basis vector of the null space — may
land in the padded coordinates. T3-SVD does not care (that column is multiplied by σ = 0 on its way into the
next core; the tensor stays exact, measured 60/60). The **frame** does: its basis vectors are the output, so
a completion in the padding is, after masking, a lost tangent direction (measured 60/60 on zero-padded
`resize` warm starts; the masked real block is genuinely rank-deficient, not merely skewed; it cascades
through the sweep; it has been so since the uniform frame existed). The case that triggers it is exactly the
rank-continuation warm start — structurally minimal ranks, numerically deficient in both families
(`repros/S1b/s1b_cases.py`) — which is the library's selling point: a zero-padded restart has almost no
gradient energy in the already-resolved eigenspace, which deflates the Newton-CG system; jitter would
re-introduce it. So this must work on the uniform layer for GPU continuation to be credible.

**The matrix model** (Nick's): rows of an unfolding are `k` nonzero, `z` numerically zero, `p` padded-zero. The
first `r = rank` left singular vectors are determined by the tensor and automatically live in the `k` rows
(`u = Mv/σ`). The remaining `n − r` are any orthonormal basis of the complement, which in the ragged layer
spans `ℝ^{k+z}` and in the uniform layer spans `ℝ^{k+z} ⊕ ℝ^p`. No SVD algorithm can keep them out of the `p`
rows without being told the partition: an adversary swaps a `z` row for a `p` row. The masks *are* the
partition and are available at every step. In floating point a roundoff-level σ is as unconstrained as an
exact zero. What the tensor case adds: four sites (Tucker down-orth, both TT sweeps, the down step), three
row-index types, interleaved padding wherever the row index is a Kronecker product of axes (`rL·n`,
`rL·rR`) — **but the mode index `N` is padded only as a suffix**, and `N` is the only large dimension; `n`,
`r` are small.

## What others do (details + sources in the S1b README)

Nobody asks the SVD for the completion. TeNPy's `MPS.enlarge_chi` zero-fills one bond and completes the
other with random rows Gram–Schmidt'd against the existing ones and QR'd (with a warning if the projection
vanishes). The rank-adaptive BUG integrator augments `(K, U₀)` and takes an orthonormal basis of its range by
QR, noting the result is independent of which basis. Vermeylen–Vandereycken–Absil's rank-adaptive TT
completion never pads: the new directions are the normal-cone component of the gradient, appended to the
cores — which is what a zero-padded start followed by one gradient step computes implicitly (the mathematical
reason the padded restart works). DMRG subspace expansion appends a residual-derived term and lets the next
SVD restore orthonormality; it is contrasted in the literature with density-matrix *noise*. PyTorch's
`linalg.svd` docs state outright that the trailing singular vectors "can be arbitrary bases".

## Candidate approaches

All of them need the real-row partition at each site, so the **plumbing cost is the same for all** (the masks
— or the recurrence-derived current masks — must reach the polymorphic sweep's uniform step functions and the
scan body). They differ in what happens at the SVD.

### A. The GSVD of `(Mᵀ, I_real)`, evaluated as one augmented SVD — designed, checked

Encode the partition in the second matrix: the GSVD of the pair `(Mᵀ, B)` with `BᵀB = P_real` diagonalizes
`M Mᵀ − λ P_real`. Because `P_real M = M` the pair **commutes**, and the generalized singular pairs sort `ℝ^R`
into `c > 0` (the tensor's singular vectors, untouched), `c = 0, s = 1` (the real complement = the completion)
and the common null space (the padding). The augmented matrix `[M | ε·B]` has Gram `M Mᵀ + ε²P_real` — the same
commuting pair folded into one operator — so its ordinary SVD has the same eigenvectors with singular values
`√(σ²+ε²), ε, 0`; and LAPACK computes the GSVD as a QR + CS decomposition of the stacked `[A; B]`, which *is*
this matrix. Checked on the walkthrough unfolding: completion identical (overlap 1.000), σ > 0 vectors exact
(`repros/S1b/s1b_gsvd.py`). In exact arithmetic ε is irrelevant to the subspaces (only the ordering), so there
is **no rank threshold**: a roundoff-level σ joins the ε-block, which is real-supported too.

- **Three small-row sites** (`R = rL·n`, `rL·rR`): use the full `B = I_real` (`R` extra columns; cheap). Exact
  and ε-independent.
- **The Tucker site** (`R = N`, suffix padding): `B = I_real` would be `O(N³)`. Nick's observation that `N` is
  special: the exact augmentation there *is* the unpadded SVD embedded (block form
  `[M_real M_realᵀ + ε²I, 0; 0, 0]`), whose σ > 0 half the padded SVD already delivers; only the completion is
  missing, and a **thin** `B = C` = the first `n` coordinate vectors masked by the shape mask always suffices
  (`min(n, N_i) − r ≥ n_i − r` directions survive projection). One SVD of `[M | ε·C]`, keep `n` columns,
  remainder `Uᵀ M`. Checked over 400 random cases incl. `N_i < n` and rank 0: orthonormal, real columns
  real-supported, tensor exact to 1e-15, σ > 0 vectors unperturbed (`repros/S1b/s1b_tucker_aug.py`).
  Cost `O(N n²)`. What the thin `C` gives up: `CCᵀ` is only a partial projector, so commutation is
  approximate — a genuine `σ < ε` can fall out of the first `n` columns, bounding the reconstruction error by
  ε (≈ `1e-10·σ_max`; zero in the continuation case, whose deficient σ's are exact zeros).

Pros: one SVD per site, same code shape at all four, batched, jit-safe, deterministic, no branch on "which
slots are zero", no per-element completion count. Cons: the SVD at each site sees `n + |B|` columns instead of
`n` — roughly 3–4× the SVD flops at the Tucker site (`N·(2n)²` vs `N·n²`), and `R + n` columns at the small sites.

### B. Round-trip through ragged (Nick's question: do A's costs pay against this?)

Convert the point to ragged, build the frame with the ragged sweep (whose completions are real by
construction), convert back. Exact. But the uniform layer exists for the jit/GPU path, where a ragged
round-trip is not expressible: the conversion produces data-dependent shapes, the sweep is `d` separate
SVDs per stack element on the host (`C·d` small kernels on GPU, each a sync), and the compile-once loop is
broken at every iteration (the frame is rebuilt after every retraction). Eagerly, on numpy, it is viable — but
eagerly the uniform layer is already slower than ragged, so the comparison only matters under jit, where B is
not available. Honest framing of the cost question, then: A's overhead is paid on every frame build; how large
is it *relative to an optimizer iteration*? The frame sweep is `W`-independent (`O(d(N n² + n r³))` plus the
SVDs), while the misfit/Jacobian sweeps scale with `W`; for any real fit the frame is a small fraction, so a
3–4× factor on its SVDs is likely invisible — **to be measured, not assumed** (the `uniform_vs_ragged` bench in
`dev/` is the place). A lazy variant: evaluate the augmented SVD only when a σ ≈ 0 is detected — under jit via
`lax.cond` on a traced predicate (both branches compiled, cost only when taken); this keeps the common-case
cost at 1× but re-introduces a threshold.

### C. "Dumb" noise on the non-padded rows (Nick's question: pros and cons)

Add a tiny perturbation `E` supported on the real rows before the SVD, so every real direction gets
`σ ≳ η > 0` and outranks the padding; the σ ≈ 0 real slots then get vectors determined by the noise, which are
real-supported because `E` is. Use `U` from the noisy SVD but the *clean* remainder `Uᵀ M` (exact projection,
and `range(M) ⊆ range(U)` whenever `σ ≫ η`).

Pros: keeps the matrix at `N × n` — **no extra SVD cost** (the one clear advantage over A); one line at each
site; same plumbing (the row mask is needed to confine `E`). Cons, in order of weight:
1. *First-order versus second-order perturbation.* The noisy Gram is `M Mᵀ + (M Eᵀ + E Mᵀ) + E Eᵀ`; the
   augmentation's is `M Mᵀ + E Eᵀ` with no cross term. So noise perturbs the kept singular vectors at
   `O(η/gap)` where A perturbs them at `O(η²/gap²)` (and not at all with the commuting `B`). Within a cluster of
   close σ's that rotation is gauge (harmless); between the kept block and the rest it is a genuine `O(η/σ_min)`
   change of the tangent space. With `η = 1e-10·σ_max` this is far below any tolerance, so numerically it is
   acceptable; it is just strictly less clean.
2. *Determinism.* Random `E` makes the frame non-reproducible and breaks every "same frame" comparison
   (`safety.frames_equal`, the jit-cache reasoning, the ragged-vs-uniform equivalence tests at 1e-15). A **fixed
   dither** instead of random noise fixes this — and the natural fixed pattern is `E = η·C`, the masked
   coordinate block of A, *added into* `M` rather than appended. That variant is exactly "A without the extra
   columns", paying item 1's first-order cross term for the column savings.
3. *Nothing else.* The completion directions being "noise-chosen" is don't-care (any real orthonormal
   completion is a valid gauge), and ragged-vs-uniform tests must compare subspaces / dense tangents rather
   than cores in either approach.

### D. Explicit completion post-pass (the first design, before the GSVD view)

Plain SVD, then replace the `σ ≤ τ·σ_max` columns by real-supported vectors orthogonal to the kept ones
(project `C`, orthonormalize). Exact (kept vectors untouched), `O(N n²)`, what TeNPy does by hand. Cons: a
rank threshold `τ` and a per-element number of replacements — under jit that is `where`/`take_along_axis`
bookkeeping rather than a branch, i.e. more code than A for the same result.

### E. Shrink the masks to the numerical rank — rejected

Treats the deficient point as lower-rank; continuation then cannot grow into the padding (Nick, 2026-08-22).

## Comparison

| | exact tensor | kept vectors | completion real | deterministic | threshold | SVD cost factor | jit/batched |
|---|---|---|---|---|---|---|---|
| A, full `B` (small sites) | yes | untouched (commutes) | yes | yes | none | `(R+n)/n` cols, R small | yes |
| A, thin `C` (Tucker site) | to ε | `O(ε²/gap²)` | yes | yes | ε only (no branch) | ~3–4× | yes |
| B, ragged round-trip | yes | untouched | yes | yes | none | 1× but unbatched, host | **no** |
| C, noise (fixed dither) | to η | `O(η/gap)` | yes | yes (fixed pattern) | η only | **1×** | yes |
| C, random noise | to η | `O(η/gap)` | yes | **no** | η | 1× | yes |
| D, post-pass | yes | untouched | yes | yes | τ + bookkeeping | ~2–3× | yes, more code |

## 2026-08-23 — RESOLUTION DIRECTION: Nick's pad-safe SVD packet (Method D, sketch–project)

Nick worked the problem through on paper (workshopped externally) and delivered a packet — spec
(`ALGORITHM.md`), 10-page derivation (`padded_svd.tex`: Methods A–D with error analysis), NumPy + JAX
implementations, pytest suite — stored at `dev/review_2026-08-22/repros/S1b/packet/`. **Method D
supersedes every option above**: exact, tolerance-free (no ε, η, or τ anywhere — every count comes from
masks or bitwise {0,1} indicators), `O(N M² + M³)` independent of pad counts, static shapes with masks
as traced runtime data (one jit compile across mask patterns, verified), arbitrary *interior* pads on
both sides — so ONE primitive serves all four sweep sites, including the Kronecker-interleaved
rank-product rows that the thin-`C` variant of option A special-cased around.

The algorithm, in one paragraph: sketch `Y = A_pad Ω` (pad rows bitwise zero for any `Ω`; fixed Haar
`Ω`), permute pad rows to the trailing pivots and QR — Householder reflectors then never touch pad
rows, so `Q` splits bitwise into data-supported columns and exact pad coordinate vectors (flagged by a
bitwise indicator `t`); one SVD of the small augmented core `[QᵀA_pad | c·diag(t)]` (option A's
augmentation, applied where it costs `M³` instead of `N·…`) pins the surplus at `σ = c = 4‖A‖_F` and
makes every kept left vector data-supported, including the σ = 0 completions; `V` is rebuilt from
`W = A_padᵀU = VΣ`, whose permuted QR *is* `V` up to column signs. The load-bearing pieces: the
pads-to-trailing-pivots permutation rule, the 4× margin in `c`, and Householder-QR semantics.

Verified here (2026-08-23): packet demos + all 7 tests green in the project env; and
`s1b_packet_integration.py` runs it on the REAL case-B warm-start unfoldings — the interior-pad TT
up-orth site loses a direction under today's plain SVD, `pad_safe_svd` recovers it, pads bitwise zero
(= the library's canonical clean-padding prefix form for free), σ's exactly the unpadded block's,
reconstruction 1e-15. The earlier cost worry is moot: it replaces one `N×M` SVD with tall-skinny
QR/matmuls + an `M×2M` SVD — same order, and the Chan-style big-QR-small-SVD split typically *beats*
a big SVD on GPU.

Open implementation questions (the remaining decisions):
1. **Where the primitive lives** — proposal: backend-agnostic `pad_safe_svd` in `backend/linalg.py`
   (xnp, masks as data), swapped into the uniform sweep sites (`ut3_orthogonalization` + the uniform
   branches of the TT sweeps). The frame, `ut3svd`, and retraction all route through those; retraction
   additionally gains a guarantee that a truncated prefix is real-supported.
2. **Feasibility corner** — the contract needs per-element `n ≥ m` (real rows ≥ real cols). Always true
   at minimal ranks; a structurally non-minimal *wide* slot (S1's supported case) needs the transposed
   call or the mask recurrence's existing `min` cap deciding which side gets the completion.
3. **`Ω` handling** — one Haar sketch per site width `M`, drawn host-side from a fixed seed and closed
   over (a jit constant; plays fine with the value-hashed cache keys).
4. **float32** — the uniform jax path defaults to f32; bitwise claims are precision-independent, test
   tolerances loosen to ~1e-5 (packet README §6).
5. **Packet nit** — the NumPy reference computes `c` from the spectral norm (`norm(A_pad, 2)`), the JAX
   version and the spec from Frobenius; port the Frobenius form (cheaper, and the documented margin).

## What would settle it

1. **A measurement** of the frame sweep's share of a uniform Newton-CG iteration at realistic `(N, n, r, W)`,
   numpy and jit — if the frame is a few percent, A's factor is noise and the cleanest option wins outright;
   if the frame is a large share (small `W`, large `N`), C-with-dither at 1× becomes attractive.
2. **The acceptance test** for any of them: the documented uniform continuation loop from a zero-padded
   `resize` start (`docs/rank_continuation.md`, `examples/fit_hilbert_uniform_newton_cg.py`) must produce the
   same objective trajectory as ragged (H4 measured 6.05e4 ragged vs 8.08e4 uniform after 4 Newton steps
   today), the frame must be orthonormal on the masked real block (`UT3Frame.is_orthogonal()` True) with no
   lost directions, and `repros/S1b/s1b_cases.py` case B must report 0 lost directions.
3. Nick's pencil-and-paper pass on whether a more elegant formulation exists than "the GSVD with the mask as
   the second matrix" — open.
