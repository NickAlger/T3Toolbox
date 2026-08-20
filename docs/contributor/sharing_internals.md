# Sharing internals: the design record

> Decision record for the shared-Tucker-factors (SF-T3) implementation — the `S_i` machinery and
> its measurements, the choices that were revised during design review, the tied embedding, the
> restart analysis, and the uniform-mirror lessons. The user-facing story is
> [`sharing.md`](../sharing.md); the math derivations live in the maintainers' working notes
> ([`../shared_t3_math.tex`](../shared_t3_math.tex), the derivation note). Papers: Peshekhonov, Arzhantsev & Rakhuba,
> "Training a Tucker Model With Shared Factors: a Riemannian Optimization Approach", AISTATS 2024,
> PMLR 238 ([link](https://proceedings.mlr.press/v238/peshekhonov24a.html)) — SF-Tucker; and
> Molozhavenko & Rakhuba, "Optimization on the extended tensor-train manifold with shared factors",
> Comput. Appl. Math. 45(6):221, 2026 ([doi](https://doi.org/10.1007/s40314-025-03605-0)) — SF-ETT,
> whose Algorithm 1 and Theorem 5 are implemented here (generalized to arbitrary partitions).
> Both are listed in full in the user guide's literature section.

## The `S_i` machinery: recompute by re-sweep, never retain, never re-SVD

The tied tangent subspace at a shared point is basis-free: per group, `V_i = S_iᵀ U̇` for one common
gauged ambient `U̇`, where `S_i = W2_i O2_iᵀ` is the small factor the frame construction itself
produces (`backend/t3_orthogonalization.py`, the `Cxi` local; `S_i S_iᵀ = Γ_i` and `W2_i = S_i O2_i`
exactly). Three decisions:

1. **Derived on demand, never stored.** `SharedFrameData` (the companion) is computed from a frame
   when needed — frames stay minimal dataclasses.
2. **Recomputed by RE-SWEEP, not by a zipper and not by a re-SVD.** The companion re-runs
   `tt_right_orthogonalize(left_tt_cores, return_variation_cores=True)` — the *same function on the
   same stored arrays* that produced the centers at construction, hence **bit-identical** (measured,
   float64 and jax float32). A re-SVD would face sign/degenerate-block hazards (symmetric tensors
   have repeated `Γ` eigenvalues); the stored-`O` pairing is immune. The zipper
   (`H_i = Z_i R_i`, GEMM-only) measured fine — flat ~5·eps32 even at `κ_TT ~ 1e6`, because the
   operands are orthogonal — and stays a documented alternative if the SVD-based re-sweep ever
   shows up in a GPU profile.
3. **The companion holds a thin SVD of the stacked `M_g = concat_i(S_iᵀ)` — never a Cholesky/Gram.**
   Measured in jax float32 on a graded spectrum (`s_min/s_max ≈ 1e-4`): the trailing spectrum level
   comes out at relative error **7.5e-5** from the SVD vs **3.1e-1** from a Gram-eigh (squaring
   destroys it). The SVD also gives the projection solve at sensitivity `κ_g` (not the normal
   equations' `κ_g²`) and a clipped pseudoinverse that is well-defined (minimum-norm) at the
   rank-deficient points continuation restarts visit by construction.

Selected measurements on file (design round):

| claim | measured |
|---|---|
| re-sweep recompute vs construction | bit-identical (float64 and jax float32) |
| zipper recompute vs construction | ≤ 4.5e-15 (float64), ≤ 3.2e-7 (float32), flat in `d`, `κ_TT` |
| group spectrum trailing level, float32 | SVD 7.5e-5 vs Gram-eigh 3.1e-1 |
| tied post-pass vs dense orthogonal projection | 1.6e-13 |
| multi-group dimension formula vs dense tangent rank | 44 == 44 (unshared reduction gives 42) |

## One principle, two formulas (the revised decision)

The original plan had one post-pass for both geometries. Wrong — the review revised it: each
geometry projects onto *its* tied subspace in *its* metric on *its* coordinates.

- **Manifold** (gauged, `S`-absorbed coordinates, where corewise = Hilbert–Schmidt): the tilted
  projection `U̇ = M_g⁺ [V_{i₁}; …; V_{iₖ}]`, redistributed as `V_i ← S_iᵀ U̇`.
- **Corewise** (raw factor copies, Euclidean metric, additive retraction): the tied subspace is
  `{δU_i all equal}` and the projection is the plain per-group **arithmetic mean**.

The Gram formula at a corewise frame is a *different, wrong* projection, and the mean at a manifold
frame projects onto the wrong subspace — both directions are pinned by an adversarial test (the
strongest form: gauged manifold coordinates can differ in *shape* across a group when the `nD_i`
differ, so the mean is not even defined on them). Means are computed in **drift form**
(`ref + mean(differences)`): an exactly-tied group is a bitwise fixed point, where the plain
`(ΣB_i)/k` already perturbs the last ulp at `k = 3`.

## Grouped `t3svd`: why two-phase, and what it implies

Firing group Tucker steps inside the interlaced sweep is impossible — the sweep's single moving
center never has all of a group's matricizations simultaneously `i`-orthogonal. Hence the
paper-faithful two-phase (TT-round → collect all centers of the same rounded tensor → all Tucker
SVDs at once → lossless left re-orthogonalization), with the dispatch anchor: `sharing=None` /
all-singleton = the literal unshared sweep, bit-identical. Implications kept on purpose:

- Under truncation the two-phase treats even singleton modes differently than the interlaced sweep
  (all Tucker steps see one TT-rounded tensor); lossless calls agree exactly. This also makes the
  reported spectra a mutually consistent family — what the continuation κ-comparison wants.
- The group truncation error is **bounded** by the `s_g` tail (the tail is the *sum* of the
  single-mode projection errors; equality holds only for singletons) — a math-note erratum found by
  a test (measured 4.19e2 vs tail 4.80e2).
- `share()` without caps is the lossless common-span rewrite; all rank *selection* happens in the
  grouped rounding. An earlier pre-truncation of the stacked-factor SVD caused a double-truncation
  spectrum-reporting artifact under caps (caught by the agreement test) and was removed.

## The tied retraction: an embedding, not a repair

The naive doubled factor `[U_g; V_i]` is **not tied** — the `V_i` differ across a group in value
and (when the `nD_i` differ) in shape, so nothing can be averaged afterwards. The shared retraction
builds the embedding tied (SF-ETT §5.2, generalized): recover `U̇` per group by the companion's
clipped solve (exact on tied coordinates — the solve residual doubles as the safe-mode tied-tangent
check), put `[U_g; U̇]` at every group mode (one array), and replace the down core `O_i` by the
companion's center `H_i` (the identity `S_i`-absorbed-`O_i = H_i` makes this an exact rewrite of
each Tucker term). Then the grouped `t3svd` truncates back to the frame's ranks.

## Padded restarts: which tangent channels are alive

Rank continuation warm-starts by zero-padding, which places the iterate *on* the lower-shared-rank
stratum: the fresh factor directions carry no core mass, the corresponding rows of every `S_i`
vanish, the new levels of `s_g` are **exactly zero**, and the tied Tucker channel admits no
first-order motion in the new directions (the clipped pinv correctly returns zero there). The
escape is carried by the **untied TT-variation channel** — first-order mass in the new up-slots,
paired with the deterministically-completed factor columns — measured to activate the new
directions within two Newton steps; `κ_g` is transiently large afterwards and the clipped solve
handles it. Consequences wired into the code: the projection solve is a clipped pseudoinverse
(mandatory), and **full shared rank is a diagnostic, never an enforced precondition** — enforcing
it would reject every restart. (ε-inflation of the new slots is a documented fallback only.)

The end-to-end continuation loop needs the documented warm-start guidance
([`rank_continuation.md`](../rank_continuation.md)): pin `g0norm_newton` across levels — without
the pin the fit stalls at the target level and continuation over-grows (reproduced, then fixed, in
the tests).

## The `GeometryOps.precompute` slot

Per-projection companion recompute is cheap full-batch but would put `2d` small SVDs on every CG
matvec. The sanctioned (breaking) protocol extension: `GeometryOps` gains an optional
`precompute: frame -> aux` slot; `Problem.local_model` computes the aux **once per Newton step**
(the `sweep` pattern) and passes it back to `project`/`retract` as `aux=`; the frontend models
mirror it as a `geometry_aux` **leaf** (it holds arrays — it must flow as traced data, never as jit
aux). The `SharedGeometry` wrapper itself is a zero-leaf pytree with value-based `__eq__`/`__hash__`
over `(base_name, sharing)` — the `ValueHashedMasks` precedent — so rebuilt-equal wrappers are one
jit cache key and shared fits compile once.

## The uniform mirror: delegation, and two hard-won lessons

Most of the uniform tangent layer is **mask-and-delegate**: the ragged companion/post-pass/solve
functions are fully polymorphic (`'...'`-einsums + mode-axis indexing + concatenate), so the uniform
twins mask where appropriate and call them on supercore slices. The grouped `ut3svd` is the
two-phase in scan/supercore form (TT-bond-only scan; centers via the polymorphic right sweep;
per-group SVDs on statically-gathered concatenations — `supercore[group]` + concatenate, no
segment-sums; masks multiply, nothing is sliced), sized by the grouped multi-pass
`compute_raw_sweep_ranks(sharing=)` recurrence — pinned equal to the ragged grouped output ranks
over randomized structures/caps *before* implementation. Two lessons that cost real debugging:

1. **The uniform companion must re-sweep the frame's supercores AS STORED — not re-masked.** The
   stored padding slots carry the construction's arbitrary orthonormal completions; masking them
   first changes the padded SVDs' sign choices and breaks the `⟨O_i, H_i⟩` pairing (one flipped
   bond column destroyed a group spectrum in test). The stored padding is harmless — completion
   rows are orthogonal to the centers' row space, so the padded `S_iᵀ` rows vanish to roundoff.
   Contract: companions belong to frames built by `ut3_orthogonal_representations`; a
   `t3frame_to_ut3frame`-packed *ragged* frame is not guaranteed (the padded re-sweep can flip
   signs against the unpadded construction).
2. **Cross-layer comparisons must be gauge-invariant and start shared-minimal.** Each layer's frame
   construction chooses its own SVD sign gauge (padding changes LAPACK's choices), so tests compare
   represented *dense* tangents/points and the invariant group spectrum — never raw coordinates.
   And trajectory comparisons need a tied, **nonzero**, **shared-minimal** start: a zero start has
   an arbitrary orthogonal completion in its frame, and a non-minimal start is transparently
   reduced (`uniform_minimal`) on the uniform path only — both legitimately diverge the layers.

Related uniform facts: `uniform_minimal` had to become sharing-aware (the per-mode reduction clips
a group rank the group ceiling admits — `(4,4,2) → (2,4,2)` on the canonical test structure —
structurally untying the group), and the tied embedding rebuilds the variation block at the **up**
width `nU` (the `U̇` slot), zero-padding singleton blocks.

## Smaller decisions on record

- **The checker is a method, not a root free function.** `has_shared_tucker_factors` follows the
  `has_minimal_ranks` checker grammar (a property of the T3, combined with a *spec*, not with
  another substantive object — the weights free-function precedent does not apply, and the uniform
  twin is a method too, so nothing collides). Root exports are the three geometry names only;
  `x.share(…)` is the frontend of `t3_share_tucker_factors`.
- **Spectra validation in continuation is exact `array_equal`** — the grouped `t3svd` assigns one
  `s_g` array per group, so unequal spectra at group modes means the input is not a shared spectrum
  family (raise, don't average).
- **`t3_share_tucker_factors` lives in `backend/t3_svd.py`**, not `sharing.py` (import direction:
  `t3_svd` imports `sharing`).
- **The regularizer used to recompute the companion** on its own `project` calls (2-arg path). That
  "accepted cost" note was mis-scoped — the recompute was once per CG **matvec**, not once per model
  — and it is **fixed**: `Regularizer.gradient`/`hessian`/`quadratic` now take `aux=` and the models
  pass their stored companion through ([`precompute_and_caching.md`](precompute_and_caching.md)).
  Unaffected either way: the regularizer's base-point tangent has zero Tucker variations (trivially
  tied), so `point_norm_sq`/`point_tangent` delegate unchanged.
