# Handoff: Shared Tucker factors (SF-T3) in T3Toolbox — v3

**Audience:** Claude Code, working in the T3Toolbox repository (github.com/NickAlger/T3Toolbox).
**Goal:** Optimize over Tucker tensor trains whose Tucker factors are constrained equal within
user-specified groups of modes ("shared factors"), following Molozhavenko & Rakhuba,
*Optimization on the extended tensor-train manifold with shared factors*, Comput. Appl. Math.
45:221 (2026) — their SF-ETT, generalized from one trailing shared block to an arbitrary
partition of modes into sharing groups. The partition is always user-provided; automatic
selection is permanently out of scope.

**Math source of truth: `dev/shared_t3_math.tex`** (updated alongside this v3 — the tilted
subspace, grouped rounding + bounds, dimension, continuation, restart analysis). This handoff
carries the *code-convention* forms and the implementation decisions. Read also the design-notes
pages named in v2 (`frame_variations`, `batching_and_stacking` — REQUIRED, `t3svd_minimal_ranks`,
`uniform_equivalence_contract`, `uniform_masks_vs_ranks`, `numerical_contracts`,
`naming_conventions`).

**v3 status: all decisions below are AGREED (Nick, 2026-08-19), after two design-review rounds:**
a full codebase/papers review (7 research agents + first-hand reads + numerical verification
scripts), and an S_i numerics round (float32/jax measurements). The v2→v3 changes are folded in
silently; the load-bearing ones are marked **[v3]**. Verification scripts that ground the claims
live in the session scratchpad and get promoted into permanent tests in slices 2–3
(`verify_sharing_claims.py`, `verify_padded_restart.py`, `s_recompute_stability*.py`).

---

## 0. Code conventions used throughout (read first)

Storage is the **row convention**: Tucker cores `B_i` have shape `stack + (n_i, N_i)` with
orthonormal ROWS on an orthogonal frame; tucker variations `V_i` have shape
`stack + (nD_i, N_i)` with `nD_i = min(n_i, rL_i·rR_i)` (the down rank — NOT `n_i` in general);
the gauge is `einsum('...ia,...ja->...ij', U_i, V_i) = 0`. The frame is
`(U, O, L, R) = (up_tucker, down_tt, left_tt, right_tt)`; `O_i` is the **up-orthonormalized
center TT core** (`(rL_i, nD_i, rR_i)`, a bond-space object — there is NO ambient complement
anywhere in the codebase, and none is needed). `H_i` denotes the mode-i center core of the
mixed-canonical chain (`x = L_0…L_{i-1}·H_i·R_{i+1}…`); `W2_i` its up-matricization
(`(n_i, rL_i·rR_i)`).

**S_i** := `W2_i @ O2_i^T`, shape `(n_i, nD_i)`, satisfying `S_i S_i^T = Γ_i = W2_i W2_i^T` and
`W2_i = S_i O2_i` exactly (`O2` = O's up-matricization). It is born inside
`t3_up_orthogonalize_tt_cores` (`backend/t3_orthogonalization.py`, the `Cxi = ssx*WTxi` local —
`Cxi = S_i^T`) / `up_orthogonalize_tt_supercores` (uniform), absorbed into the base point's own
tucker coordinates `V_i^base = S_i^T U_i`.

**The tied (shared) tangent subspace, basis-free:** at a shared point (tied `U_g`), a tangent is
tangent to the shared manifold iff its TT variations are unrestricted and, per group `g`, its
Tucker variations satisfy

```
V_i = S_i^T @ Udot          for one common gauged ambient direction Udot (n_g × N),
                            U_g @ Udot^T = 0,      for all i in g.
```

No complement basis, no tied-`O` requirement (v2's Step 0.2 dissolved — verified: the per-mode
SVD gauge freedom `(O_i, S_i) → (O_i Q, S_i Q)` cancels identically in everything below).

---

## 1. Agreed design decisions

1. **General partition.** `sharing` = a length-`d` tuple of hashable group labels, e.g.
   `(0, 1, 1, 2, 2, 2)`; `sharing=None` = fully unshared, existing behavior exactly. Modes in a
   group must have equal mode sizes (structural error otherwise); groups need not be adjacent.
   Canonical internal form: `groups: tuple[tuple[int, ...], ...]` — ALL modes covered
   (singletons included), inner tuples ascending, groups ordered by first mode; static/aux for
   jit.
2. **No frontend data-structure changes.** `TuckerTensorTrain`, `T3Frame`, `T3Variations`,
   `T3Tangent`, and the uniform classes stay as they are. A shared T3 is an ordinary T3 whose
   Tucker factors are equal within groups (redundant storage — the paper's Eq.-15 memory saving
   is deliberately forgone; document the compute-not-memory trade-off).
3. **Delivery:** `sharing=` kwargs on backend functions + frontend methods
   (`t3svd`, `t3_rank_adjustment_sweep`, `compute_minimal_ranks`, `compute_manifold_dim`,
   `compute_continuation_ranks`, uniform twins), one geometry wrapper
   `shared(base_geometry, sharing)` with factories `shared_manifold(sharing)` /
   `shared_corewise(sharing)`, a derived companion (`T3SharedFrameData`), and **one sanctioned
   protocol extension [v3]**: `GeometryOps` gains an optional `precompute` slot (§4.4;
   breaking-change release already planned for the contractions work).
4. **In scope for v1:** both geometries, the uniform layer, shared minimal ranks +
   `manifold_dim` + rank continuation, the user docs page, and the symmetric jetted-probes
   example.
5. **Inner products are untouched.** The tied subspace is a linear subspace; the restriction of
   either geometry's metric to it is itself. Never add `sharing` to `inner`/`norm`.
6. **[v3] The post-pass is geometry-specific — decision 7 of v2 is REVISED.** One principle —
   *orthogonally project onto the geometry's tied subspace in the geometry's own metric on its
   own coordinates* — two formulas:
   - **MANIFOLD** (gauged, S-absorbed coordinates; corewise = HS there): the tilted projection
     `Udot = M_g^+ Vstack`, `V_i ← S_i^T Udot` (§4.3).
   - **COREWISE** (raw core perturbations at the `(U,G,G,G)` frame; metric = Euclidean on raw
     entries; additive retraction): the tied subspace is `{δU_i all equal}` and the projection
     is the **plain per-group arithmetic mean**, assigned as ONE array per group. No `S`, no
     companion, no solve. (The Gram formula at a corewise frame would be a different, wrong
     projection; conversely the mean at a manifold frame projects onto the wrong subspace —
     both directions get an adversarial test, §7.7b.)
7. **[v3] Grouped `t3svd` is the paper-faithful two-phase algorithm** (Nick's decision), with
   the dispatch rule of §4.2. The v2 idea of firing group steps inside the interlaced sweep is
   dead — the sweep's single moving center makes simultaneous i-orthogonal `W2_i` impossible.
8. **[v3] Full shared rank is a DIAGNOSTIC, never an enforced precondition** — enforcing it
   would reject every rank-continuation restart (§4.8). The projection solve is a clipped
   pseudoinverse and is well-defined (min-norm) at degenerate points.

---

## 2. The S_i machinery **[v3 — replaces v2's Step 0]**

### 2.1 Recompute, don't retain

`T3SharedFrameData` is **derived from the frame on demand** — never carried inside a frame, never
cached statefully:

```
fv_shared_frame_data(frame_data, groups) -> T3SharedFrameData
  1. (_, H) = tt_right_orthogonalize(left_tt_cores, return_variation_cores=True)
       # the RE-SWEEP: the same function, on the same stored arrays, that produced the centers
       # at frame construction — measured BIT-IDENTICAL to construction (float64 AND jax
       # float32). No zipper (measured fine — flat ~5·eps32 even at kappa_TT ~ 1e6, orthogonal
       # operands, ||Z_i||_F = ||T|| — but the re-sweep is exact and settles the question;
       # the zipper stays documented as a GEMM-only alternative if GPU profiling ever wants it).
  2. per mode of each nontrivial group: S_i = W2_i @ O2_i^T against the frame's STORED O_i
       # one GEMM; no re-SVD, so no sign/degenerate-block hazards (symmetric tensors have
       # repeated Gamma eigenvalues — the stored-O pairing is immune).
  3. per nontrivial group: thin SVD of  M_g = vstack_i(S_i^T)   ((sum_i nD_i) × n_g)
```

### 2.2 The companion holds an SVD, not a Cholesky **[v3 — replaces v2 §4.3's chol]**

```python
@dataclass(frozen=True)
class T3SharedFrameData:            # backend/sharing.py; jax pytree: arrays leaves, groups aux
    groups:   tuple                 # static: the FULL canonical partition
    centers:  tuple                 # per nontrivial group: tuple of H_i  (frame stack C leading)
    svd_U:    tuple                 # per nontrivial group: U_M of thin SVD of M_g,  C+(sum nD, q)
    svd_s:    tuple                 # per nontrivial group: sigma(M_g) = s_g — THE group spectrum
    svd_Vt:   tuple                 # per nontrivial group: Vt_M,  C+(q, n_g)
```

Why the SVD (measured, jax float32, graded spectrum with `s_min/s_max ≈ 1e-4`):

| quantity | Gram-sum eigh | thin SVD of `M_g` |
|---|---|---|
| trailing `s_g` level rel. err (float32) | **3.1e-1** (destroyed — squaring) | **7.5e-5** |
| projection-solve sensitivity | `κ_g²` always (normal equations) | intrinsic LS (`κ_g` on the solution) |
| degenerate points | singular chol | clipped pinv, min-norm, well-defined |

One tiny batched SVD per group per frame buys the solve, the spectrum (the continuation/aptness
statistic — genuinely "free at frame construction" now, and trustworthy in float32), and
degeneracy tolerance. Never form `Σ S_i S_i^T`.

### 2.3 Threading: the `GeometryOps.precompute` slot **[v3]**

Per-projection recompute is ~`1/W` of a matvec (fine full-batch) but would put `2d` small SVDs
per CG matvec on GPU. Sanctioned protocol extension (backend `optimizers.GeometryOps`):

```
precompute: Optional[Callable] = None     # frame -> geometry aux (None for existing geometries)
project:    (frame, variations, aux=None) -> variations
retract:    (frame, variations, aux=None) -> x_cores
```

`Problem.local_model` computes `aux = geom.precompute(frame)` once per Newton step, stores it as
a **leaf** field beside `sweep`, and passes it to `project`/`retract`. Existing geometries take
and ignore `aux` (updated in place — the breaking release absorbs the signature change). The
frontend `GaussNewtonModel` mirrors with a `geometry_aux` leaf field; standalone frontend calls
(`geom.project(v)` outside a model) recompute internally via `fv_shared_frame_data`.

### 2.4 Safe-mode contracts (frontend-only, per the house convention)

- **Structural, always:** partition validity (`validate_sharing`), equal mode sizes AND equal
  Tucker ranks within groups (shape-level).
- **Numerical preconditions (safe mode, skipped under `unsafe()`/jit):**
  `t3_sharing_residual <= effective_rtol` at shared `frame(x)` entry, shared `retract` entry,
  and frontend `t3svd(sharing=)` entry; tied-variation-coordinates residual
  (`||Vstack − M_g Udot||/||Vstack||` from the retraction's own solve) at `retract` entry.
- **Diagnostics only (never enforced):** full shared rank / `s_{g,min}` (would reject
  continuation restarts, §4.8); `gauge_residual` after the post-pass (a test, not a runtime
  check — the post-pass preserves gauge identically, verified).
- Update `docs/numerical_contracts.md` + the contributor catalog per the house recipe.

### 2.5 Permanent invariant tests (promote the verification scripts)

`S_i S_i^T == Γ_i` and `W2_i == S_i O2_i` (per mode); re-sweep-recompute == construction
(exact equality); tied factors survive frame construction bit-identically; post-pass ==
dense orthogonal projection onto the tied subspace (measured 1.6e-13); post-pass idempotent,
gauge-preserving, fixed on exactly-tied input, recovers `Udot` exactly.

### Side-fix (own commit, before slice 1)

`fv_conversions.t3_orthogonal_representations` shadows its `squash_tails` parameter with a local
lambda — `squash_tails=False` silently still squashes. No caller passes `False` today; fix +
regression test. (Separately noted, not fixed here: the numpy ragged sweeps upcast float32 →
float64 after step 1 — benign, jax path is clean float32; backlog note.)

---

## 3. Stacking and broadcasting

As v2 §3, with two resolutions:

1. **The cross-stack rank rule (v2 §3.4) resolves to the existing behavior: tolerances REFUSE
   stacked input** (`truncated_svd` raises; only explicit caps batch). Grouped paths mirror
   this exactly — `rtol`/`atol` + nonempty stack raises; `max_*_ranks` batch. Nothing new to
   invent.
2. The companion follows the FRAME stack `C` (leading axes on `centers`/`svd_*`); the post-pass
   broadcasts `C` (companion) against `K+C` (variations) — the library-wide frame-inner
   right-aligned broadcasting gives this for free; use `contract(...)`/batched `xnp.linalg`
   where a plain `'...'` cannot express the two blocks. Sanity rule stands: every new function
   = (static Python loop over groups) × (batched array ops); a group loop never iterates a
   stack axis.

---

## 4. File-by-file changes (ragged layer)

### 4.1 `backend/sharing.py` (new) — slice 1

- `validate_sharing(sharing, shape) -> groups` — canonicalize + structural errors (length `d`,
  hashable labels, equal mode sizes per group).
- `t3_sharing_residual(x, sharing) -> NDArray` — per-stack-element max over groups/modes of
  `||B_i − B_ref||_F / ||B_ref||_F` (`B_ref` = the group's first factor; branch-free zero
  guard). Shape `stack_shape` (0-d unstacked), house residual convention.
- `t3_tucker_factors_shared(x, sharing, rtol=1e-9) -> bool array` — the boolean checker on the
  residual (check-free backend; frontend safe-mode sites pair the residual with
  `safety.effective_rtol`).
- `t3_share_tucker_cores(x, sharing)` — per-group arithmetic mean, ONE array assigned to every
  group mode. Drift repair for nearly-tied POINTS only; never for tangents/embeddings.
- (slice 2) `T3SharedFrameData` + `fv_shared_frame_data` per §2.

### 4.2 Grouped `t3svd` — `backend/t3_svd.py` (+ `t3_rank_adjustment_sweep`) — slice 3

**Dispatch [v3]:** `sharing=None` OR all-singleton partition → the literal existing interlaced
sweep (bit-identical regression anchor). Any partition with a real group → **two-phase for ALL
modes** (singletons included — the paper's Algorithm-1 structure; under truncation this treats
singletons differently than the unshared sweep does; expected, documented; the lossless case
agrees with everything).

**Two-phase algorithm** (input right-orthogonalized as today; `assume_orthogonal` semantics
unchanged; tied+orthonormal factors asserted/assigned at entry — the Tucker
down-orthogonalization is tie-preserving on tied input, verified bit-identical):

1. **TT rounding:** the L→R sweep of `t3_left_svd_tt_core` with `max_tt_ranks`/`rtol`/`atol`,
   NO Tucker steps. Output left-orthogonal at the TT caps.
2. **Collect:** `tt_right_orthogonalize(left_cores, return_variation_cores=True)` → every
   center `H_i` of the TT-rounded tensor, simultaneously, losslessly (the frame-construction
   machinery — nothing new).
3. **Tucker steps, simultaneous (HOSVD-style):** singletons — truncated SVD of `W2_i`;
   groups — ONE truncated SVD of `[W2_{i_1} | … | W2_{i_k}]` → `Y_g`, spectrum `s_g`
   (cross-group equal `max_tucker_ranks` validated; tolerance = the tail-Frobenius rule against
   the concatenated matrix's norm `√k‖T‖` — document). Apply `Y^T` to the factor ONCE per group
   (assign the same array to every group mode) and `Y` to every group core's up leg, on the
   phase-2 right-orthogonal chain. Report `s_g` as `ss_tucker[i]` for every `i ∈ g`.
4. **Restore the left-orthogonal output contract** (`point_norm_sq` relies on it): one plain
   lossless left-orthogonalization sweep. Rank behavior = the raw-sweep rule
   (`compute_raw_sweep_ranks` analog) — the "non-minimal caveat" is inherited from the unshared
   sweep, not worsened; retraction rank-stability keeps its existing minimal-frame caveat.

Bonus of the paper ordering: all Tucker spectra come from the SAME TT-rounded tensor — a
mutually consistent family (exact spectra of the iterate in the lossless call), which is what
the continuation κ-comparison wants.

**Rank upper bound** (state + prove in a comment, tail-energy form): every selected group rank
`≤ rank_ε([X_(i1)|…|X_(ik)])` of the ORIGINAL input, where `rank_ε` counts by tail Frobenius
energy (`docs/contributor/t3svd_verification.md`'s criterion — the per-σ phrasing elsewhere is
stale); TT bounds unchanged. Edge-monotonicity argument as in the unshared proof.

**`t3_rank_adjustment_sweep(x, direction, sharing=None)` [v3 — new requirement]:** the per-mode
lossless reduction would UNTIE groups (`n_g` may legitimately exceed a single mode's
`rL_i·rR_i`). Grouped variant: group Tucker steps use the structural rank of the concatenation
(min of dims — no tolerance, same as the per-mode lossless steps). Needed by `uniform_minimal`
(§5) and the frontend method.

### 4.3 Post-pass — `backend/sharing.py` + kwarg threading — slice 5

`fv_share_tucker_variations(variations_data, shared_data)` (fv_ prefix — variations-only op,
the `fv_absorb_weights` precedent): per nontrivial group, stacked form,

```
Vstack = vstack_i(V_i)                       # (K+C) + (sum nD, N)
Udot   = Vt_M^T @ diag(clip_pinv(s_g)) @ U_M^T @ Vstack     # clipped pinv of M_g; batched GEMMs
V_i    <- row-slices of  M_g @ Udot          # = S_i^T Udot; static row offsets from the nD_i
```

TT variations untouched; gauge preserved identically (verified); tied output assigned
consistently. Corewise twin `fv_mean_tucker_variations(variations_data, groups)` — per-group
mean, one array per group. Thread `shared_data=None` kwargs through
`tv_orthogonal_gauge_projection` / `tv_project_t3_onto_tangent_space` /
`tv_project_dense_onto_tangent_space` (default unchanged; the post-pass fires after the
existing gauge projection — `P_tied = P_tied ∘ P_gauge` since tied ⊂ gauged, verified vs the
dense projector to 1.6e-13).

### 4.4 `shared_geometry.py` (new) + integration gates — slice 6

Backend: `shared_ops(base_ops, groups) -> GeometryOps` — a closure factory (the
`uniform_geometry_ops` pattern): `frame` = delegate (+ safe-mode tied check at the frontend
layer); `precompute` = `fv_shared_frame_data` (MANIFOLD) / `None`-equivalent trivial aux
(COREWISE — the mean needs only `groups`); `project` = delegate then the geometry's post-pass;
`retract` = §4.5; `inner` = delegate. `point_norm_sq`/`point_tangent` delegate unchanged (the
regularizer's `v_X` has zero Tucker variations → trivially tied; `IdentityRegularizer` composes
for free).

Frontend: a `SharedGeometry` class wrapping MANIFOLD/COREWISE with the same method surface
(+ `project_ambient`/`transport` = delegate-then-post-pass at the target frame, MANIFOLD only;
`randn`/`randn_like` = delegate then project — a standard Gaussian on the tied subspace;
`project_oblique` = delegate + safe-mode tied check, documented). Value-based
`__eq__`/`__hash__` over `(base, groups)` (jit-aux stability — the `ValueHashedMasks`
precedent). **Integration gates [v3] — six identity-check sites must learn the wrapper:**
`optimizers._geometry_ops`, `optimizers._uniform_geometry_name`, `fitting._ragged_frame`,
`fitting._backend_geometry_ops`, `fitting._uniform_model`, `UniformGaussNewtonModel._ubgeom`.

### 4.5 Shared retraction **[v3 — replaces v2's Step 0.4 re-share, which is unsound]**

`tv_to_t3`'s doubled factors are `[U_i; V_i]` — for a tied tangent these differ across a group
in VALUE and (when `nD_i` differ) in SHAPE; there is nothing to mean-average. The shared
embedding mirrors the SF-ETT paper's §5.2 (tied by construction):

- per nontrivial group: recover `Udot` from the tied `V_i` by the companion's clipped pinv
  (exact on tied input; the solve residual is the safe-mode tied-tangent check); doubled factor
  = `vstack(U_g, Udot)` — ONE array per group, `2n_g` rows; the mode-i bottom core block is the
  **center `H_i`** (from the companion) in place of `O_i` — the identity
  `einsum(S_i, O_i) == H_i` is exact by construction (verified).
- singleton modes unchanged (`[U_i; V_i]` with `O_i`).
- then grouped `t3svd(sharing, max_tucker_ranks=frame up-ranks, max_tt_ranks=frame left-ranks)`.

Implemented as a `shared_data=` branch in `tv_to_t3`/`tv_retract` (kwarg, default None ⇒
existing). COREWISE retract needs none of this: additive, tied-in ⇒ tied-out exactly.

### 4.6 `t3_share_tucker_factors` initializer — `backend/sharing.py` — slice 4

Nonshared → shared quasi-optimal projection (math M5): per group, one tall QR of the stacked
factors, SVD of the small stacked matrix, `U_g = Q Y`; project every mode; then grouped
`t3svd`. Frontend method `TuckerTensorTrain.share(sharing, ...)`. Must agree with grouped
`t3svd` on already-shared input; stack-aware.

### 4.7 Shared minimal ranks + manifold dim — `backend/ranks.py` — slice 7

- `compute_minimal_ranks(shape, tucker_ranks, tt_ranks, sharing=None)` — **structural integer
  arithmetic only** (v2 §4.6 conflated structural/numerical; the numerical shared rank is the
  grouped `t3svd`'s `s_g` at a tolerance). Changes: group-wide reductions; the group ceiling
  `n_g ≤ min(N_g, Σ_{i∈g} min(N_g, rL_i·rR_i))` replacing the per-mode phase-3 cap at group
  modes, re-evaluated at every group-mode visit; single pass suffices (M7.5 lemma; idempotence
  = a property-based test, not a runtime check).
- `compute_manifold_dim(shape, tucker_ranks, tt_ranks, sharing=None)` — shared minimal-rank
  reduction FIRST (verified concretely: the unshared reduction clips `n_g` where the group
  ceiling admits it and under-counts — measured 42 vs the true 44), then the standard TT term
  (limits: subtract `Σ_{i=1}^{d-1} r_i²` — the SF-ETT paper's printed Thm 5 has an off-by-one)
  + one Stiefel term `n_g(N_g − n_g)` per GROUP. Frontend `manifold_dim(s, sharing=None)`
  (existing one-structure-tuple signature). Multi-group smoothness/dimension is OUR extension
  (both papers prove single-block only); the dense tangent-rank test (§7.9) is the empirical
  backstop — already validated once (44 == 44) in the verification script.
- `frame_has_minimal_ranks(..., sharing=)` where the shared geometry needs it.

### 4.8 Shared rank continuation — `backend/ranks.py` — slice 8

Policy identical to unshared (T4S **§5.4.1**; grow iff `κ < κ_max / τ` — v2's inequality was
inverted), with:

- the group contributes ONE `κ_g = s_{g,1}/s_{g,n_g}` to the pool (spectra: the group modes'
  `tucker_singular_values[i]` all carry `s_g` from the grouped lossless `t3svd`; validate equal
  within groups; dedupe per group);
- one growth decision per group, applied group-wide; `kappa_guard` applies to `κ_g`;
  `max_grow` counts a group as ONE candidate (`_grow_capped_edges` trials the group-wide
  increment, cleaned by shared `compute_minimal_ranks`); uniform-bump counts each group once;
- group-aware useless-rank removal = shared `compute_minimal_ranks` (§4.7);
- zero-padded restart: `resize` pads the group factor once, same array to every mode (verified:
  padding + frame construction preserve exact ties).

**The padded-restart escape [v3 — new analysis, verified P1–P4]:** at a freshly padded shared
point the new factor directions carry no core mass, so `S_i` has exactly-zero rows there,
`s_g`'s new levels are exactly 0, and the tied-Tucker channel is gated (the tied projection of
a generic gradient has exactly-zero content in the new directions). The escape survives through
the **untied TT-variation channel** (O(1) first-order mass in the new up-slots, paired with the
deterministically-completed factor columns — measured); the tied-Tucker channel reactivates at
step 2 with a transiently large `κ_g` (the clip-pinv handles it; small mass ⇒ large permitted
rotation, the same natural-gradient behavior the unshared S-absorbed coordinates have).
Consequences: clipped pinv mandatory (§2.2); full shared rank never a precondition (§2.4);
a dedicated restart-escape test (§7.14h); ε-inflation of the new slots stays a documented
fallback only.

Aptness diagnostic (`κ_g` vs `κ^loc_i`), termination/selection via sharing-aware
`manifold_dim`, and the host-side ragged layer note: as v2.

### 4.9 Minor touch points

- Exports (`t3toolbox/__init__.py`, slice 7) **[REVISED at slice 7, Nick 2026-08-19]:** root gains
  ONLY the three geometry names `shared`, `shared_manifold`, `shared_corewise` (frontend, from
  `shared_geometry.py`). The two backend functions v3 listed are NOT exported: the checker became
  the **method `TuckerTensorTrain.has_shared_tucker_factors(sharing, rtol=1e-9)`** (a property
  checker of the T3 — the `has_minimal_ranks` grammar — not a free function: it combines the T3
  with a *spec*, not another substantive object; the uniform twin will be a method on
  `UniformTuckerTensorTrain`, no collision), and `t3_share_tucker_factors` needs no free-function
  frontend because `x.share(...)` IS its frontend. Raw-`.data` users import `backend.sharing` /
  `backend.t3_svd` as always.
- Docs (slices 12–13): user page **`docs/sharing.md`** on the `weighting.md` template (tied
  tangent picture; grouped `t3svd` + the two-phase/singleton note; the two geometries and their
  different post-passes; Batching; Uniform; Scope: memory-vs-compute, **sharing ≠ symmetry**
  (SF-Tucker is explicit; our flagship example is a symmetric tensor — preempt the conflation),
  weights×sharing deferred, partition user-provided); **plus a dedicated subsection "What the
  group spectrum is" [added after Nick's slice-8 question — MUST be included]:** the four
  equivalent faces of `s_g` — (a) singular values of the concatenated matricization
  `[T_(i1)|…|T_(ik)]` (representation-independent; what SF-Tucker/SF-ETT compute — cite
  Peshekhonov, Arzhantsev & Rakhuba 2024 AND Molozhavenko & Rakhuba 2026); (b) `s_g² =
  eig(Σ Γ_i)` summed mode Grams; (c) the Jacobian spectrum of a gauged tied factor motion (why
  `κ_g` IS the tied subproblem conditioning); (d) an honest single-cut spectrum of the k-fold
  stacked/lifted tensor (mode-permuted copies along a new axis — the unshared cut intuition
  survives one level up). Plus: the `√k` scale (`Σs² = k‖T‖²` — cancels in ratios but breaks
  the per-mode norm invariant at group modes), elementwise domination `s_{g,j} ≥ σ_{i,j}`, the
  d=2 `[T|Tᵀ]` blend picture, under truncation all phase-3 spectra come from the SAME phase-1
  tensor (cleaner than the unshared moving-target sweep), and the `T3Weights.from_t3svd`
  caveat (grouped `ss_tucker` breaks its norm convention — name it in the deferred
  weights×sharing Scope note). A pointer paragraph in `docs/rank_continuation.md` (grouped
  edges = one edge; the sharing kwargs). Contributor record
  `docs/contributor/sharing_internals.md` (S_i story + measurements, decision-7 revision,
  embedding design, restart analysis); `getting_started.rst` snippet with `sharing=(0,0,0)`
  (CI-doctested); CHANGELOG (`### Added` + the `!` protocol note for `GeometryOps`).
- Example (slice 13): `examples/` — fit a symmetric target (the stock Hilbert tensor IS
  symmetric) to **jetted probes** (`probe_derivatives`, orders `0..d−1`, per-order ω) with
  `sharing=(0,…,0)`; compare shared-vs-unshared parameter counts (`manifold_dim(sharing=)`) and
  recovery; assert `t3_tucker_factors_shared` at the end; print the M7.4 degeneration
  (`κ_g` == per-mode `κ` exactly on symmetric data). Template:
  `fit_hilbert_uniform_probe_derivatives_newton_cg.py`; ragged `newton_cg` for readability,
  uniform variant noted.

---

## 5. Uniform layer (v1) — slices 9–11

As v2 §5, with the v3 corrections:

1. `ut3svd(sharing=)`: the two-phase structure inside the scan machinery — one scan collects
   the centers (the right-orthogonal gauge + accumulated carry already exposes each mode's
   unfolding), group SVDs on masked zero-padded concatenations (static shapes; zero columns ⇒
   zero singular values), a second pass applies. Rank selection stays **structural/mask-only**
   (no tolerances on uniform, per the equivalence contract); the grouped raw-sweep-ranks
   recurrence (host, static) sizes the shrink; ONE group rank mask assigned to every group mode.
2. Uniform companion: `S` slices as a small supercore-style array (mode axis leading), built in
   `up_orthogonalize_tt_supercores`'s twin recompute; per-group stacked SVD with masked padding.
   **No segment-sum machinery** — static per-group gathers (`supercore[np.asarray(group)]`) +
   `xnp.concatenate`, the house pattern.
3. Geometry: extend the factories — `uniform_geometry_ops(name, x0_data, sharing=None)` — the
   closures capture `groups` alongside the masks; the packed compile-once path
   (`UniformGaussNewtonModel`, `uniform_least_squares_problem`) threads `sharing` into its
   value-hashed aux (a nested tuple, like `weight`).
4. **`uniform_minimal` must be sharing-aware [v3]** — `optimizers._setup` runs it on every
   uniform `x0`; the per-mode reduction would silently untie a shared start. Route through the
   grouped `t3svd`/`rank_adjustment` (shared structural ranks).
5. S_i on the uniform layer: the `C_x_i` local of `up_orthogonalize_tt_supercores`
   (`backend/ut3_orthogonalization.py:147`) — the recompute mirrors §2.1 through the
   polymorphic `t3_orthogonal_representations` kernels.
6. Primary harness: the uniform equivalence contract (every shared uniform op == its ragged
   twin after conversion, truncation included, jitted included) + exact-mask asserts +
   garbage-robustness per `testing_strategy.md`; plus a compile-once shared uniform fit.

---

## 6. New public surface (summary)

```python
import t3toolbox as t3t

geom  = t3t.shared_manifold(sharing)          # = t3t.shared(t3t.MANIFOLD, sharing)
geomc = t3t.shared_corewise(sharing)          # = t3t.shared(t3t.COREWISE, sharing)
x_fit, stats = t3t.newton_cg(geom, 'apply', ww, b, x0, ...)   # optimizers unchanged
# a UniformTuckerTensorTrain x0 runs the packed/jit path, as today

y, ss_tucker, ss_tt = x.t3svd(sharing=sharing, ...)           # grouped truncation (s_g per group mode)
x0 = x.share(sharing, max_tucker_ranks=None, max_tt_ranks=None, rtol=None)   # backend: t3_share_tucker_factors
x.rank_adjustment_sweep(direction, sharing=sharing)
TuckerTensorTrain.get_minimal_ranks(shape, tucker_ranks, tt_ranks, sharing=sharing)  # staticmethod, as today
x.continuation_ranks(sharing=sharing, tau=10.0, n_chunk=1, kappa_guard=1e12, max_grow=None)
t3m.manifold_dim(s, sharing=sharing)                          # existing structure-tuple signature
x.has_shared_tucker_factors(sharing, rtol=...)                # method (slice-7 revision); backend:
                                                              #   sharing.t3_tucker_factors_shared / t3_sharing_residual
```

---

## 7. Test matrix

All numerical tests numpy-only per the house convention; jax/jit via `test_dispatch.py`
entries; stacked variants per §3 (`stack_shape=(3,)` and `(2,2)`). Changes vs v2 marked.

1. **S_i/companion invariants (permanent — promote the verification scripts):** §2.5 list.
2. **Singleton/None regression:** `t3svd(sharing=None)` and all-singleton partitions
   bit-identical to `t3svd()` (the dispatch rule); `shared(GEOM, singletons)` reproduces GEOM
   on the README fitting example, both geometries, both layers.
3. **Grouped SVD correctness:** dense concatenated-matricization subspace match; truncation
   error = the `s_g` tail; quasi-optimality with `C(d)`.
4. **Lossless shared-vs-unshared agreement** (dense allclose + same TT ranks); under truncation
   they are EXPECTED to differ (two-phase note).
5. **Rank-not-too-large:** exact-rank recovery; the tail-energy upper bound vs dense
   (small sizes); through `share` and the retraction (doubled input → exactly target ranks).
6. **`share` initializer:** agrees with grouped `t3svd` on shared input; exact recovery;
   `C(d)` bound.
7. **Projection vs dense reference:**
   (a) MANIFOLD: `shared(MANIFOLD).project_ambient` == dense orthogonal projection onto the
   tied tangent space (~1e-12; prototype already passing at 1.6e-13).
   (b) **[v3] geometry-separation adversarial test:** at a shared point with deliberately
   unequal group Grams, the manifold post-pass == the dense HS projection (and ≠ the mean);
   the corewise mean == the dense raw-Euclidean projection onto `{δU_i equal}` (and ≠ the Gram
   formula). Replaces v2's "identical result expected".
8. **Gradient consistency:** finite differences of `f(retract(x, t·ξ))` vs `inner(grad, ξ)`,
   both geometries, through `GaussNewtonModel`.
9. **Dimension cross-checks:** dense tied-basis rank == `manifold_dim(sharing=)` (validated
   once already: 44 == 44, on a structure where the group ceiling exceeds a per-mode ceiling);
   shared `compute_minimal_ranks` on constructed rank-deficient concatenations.
10. **Retraction axioms:** `retract(x, 0) == x`; `O(t²)` second-order agreement; output exactly
    tied (single array per group); both geometries.
11. **End-to-end recovery, iterates stay shared:** README example with a shared target,
    MANIFOLD ragged + COREWISE ragged + uniform packed jit; `t3_tucker_factors_shared` after
    every iteration; parameter count matches shared `manifold_dim`.
12. **Uniform equivalence contract** (+ exact masks + garbage robustness + compile-once).
13. **Stacking:** stacked variants of 3/5/7/10/11; the frame-stack-1 × tangent-stack-k
    broadcast; stack/unstack round-trips preserve exact ties.
14. **Rank continuation:** (a) all-singleton == existing exactly; (b) `s_g` spectrum identity
    vs dense (~1e-12); (c) symmetric degeneration (`s_g = √k·σ`, `κ_g == κ` exactly);
    (d) complementary-spectra sum behavior + the M7.3 bound; (e) group-aware removal incl. the
    `n_g > rL_i·rR_i` case + property-based idempotence; (f) padded restart: dense-equal,
    exactly tied; **(g) [v3] restart escape: from a zero-padded shared start, the fit activates
    the new group directions within two Newton steps** (P4's TT channel) and continuation
    proceeds; (h) end-to-end continuation to true shared ranks.

---

## 8. Commit sequence — STATUS 2026-08-19 (session 2): 0–8 DONE, one commit each

0. **DONE** (`2bad59de`) handoff v3 + `shared_t3_math.tex` errata (+ rebuilt pdf).
0'. **DONE** (`d15d4807`) `fix(fv_conversions)`: the `squash_tails` shadowing bug + regression test.
1. **DONE** (`74676bcd`) `backend/sharing.py`: `validate_sharing`, `t3_sharing_residual`,
   `t3_tucker_factors_shared`, `t3_share_tucker_cores` + `tests/test_sharing.py` + dispatch entry.
2. **DONE** (`e10dba36`) `T3SharedFrameData` + `fv_shared_frame_data` (re-sweep + stacked SVD) +
   the §2.5 permanent invariant tests (safe-mode wiring landed with slice 6, where the frontend
   entry points exist).
3. **DONE** (`cc339507`) grouped `t3svd` (two-phase + dispatch) + `t3_rank_adjustment_sweep(sharing=)`
   + rank-bound comment + frontend threading with safe-mode tied checks + tests 2–5.
4. **DONE** (`93df40a8`) `t3_share_tucker_factors` + `TuckerTensorTrain.share()` + test 6.
5. **DONE** (`6cbaaa85`) post-pass (`fv_share_tucker_variations` + `fv_mean_tucker_variations`)
   + `tv_*` kwarg threading + tests 7 (incl. stacked / K-over-C broadcast).
6. **DONE** (`0a41027e`, breaking `!`) `t3toolbox/shared_geometry.py` + backend
   `shared_geometry_ops` + the `GeometryOps.precompute` protocol extension + the tied
   embedding/retract + the integration gates + tests 8, 10, 11-ragged.
   (+ `89f0dced` per-class setUp seeding in `tests/test_sharing.py`; `e7ce4531` HANDOFF.)
7. **DONE** (`187ea0ae`) shared `compute_minimal_ranks(sharing=)` (group ceiling, single-pass) +
   `compute_manifold_dim(sharing=)` + `frame_has_minimal_ranks(sharing=)` + frontend threading
   (`manifold_dim(s, sharing=)`, `get_minimal_ranks(sharing=)`) + the
   `has_shared_tucker_factors` METHOD (the §4.9 export revision) + root exports
   (`shared`/`shared_manifold`/`shared_corewise`) + tests 9 (hand-worked in
   `tests/backend/test_ranks.py`, dense ground truths in `tests/test_sharing.py`) + CHANGELOG
   (the sharing `### Added` entry + the BREAKING `GeometryOps` note under `### Changed`).
   **Gate state: full suite 687 passed / 41,940 subtests; whole-package doctests green;
   compat-floor env green (touched tests + doctests + import); docs `-W` build green; NOT pushed.**
8. **DONE** (`50ba1e68`) shared rank continuation: `compute_continuation_ranks(sharing=)` (one
   `κ_g` per group; group-wide decisions; group = ONE `max_grow` candidate via the group-aware
   `_grow_capped_edges(sharing=)`; shared removal at all three cleanup sites; identical-spectrum
   validation with exact `array_equal` — the grouped t3svd assigns one `s_g` array per group) +
   frontend `continuation_ranks(sharing=)` (threads through the grouped t3svd, inheriting its
   safe-mode tied check) + `resize(..., sharing=)` (safe-mode tied check at entry; post-pass =
   `t3_share_tucker_cores`, exact on tied input via the drift-form mean — ONE array per group) +
   tests 14: synthetic growth-rule tests in `tests/backend/test_ranks.py`
   (TestSharedContinuationRanks — group-wide growth, κ_g as κ_max, guard on κ_g, group as one
   max_grow candidate, capped-group skip, fallback bumps group once, unequal-spectra rejection,
   all-singleton==unshared) and tensor tests in `tests/test_sharing.py` (TestSharedContinuation —
   14c symmetric degeneration `s_g = √3·σ` to 7e-16 + `κ_g == κ_loc`; 14d mediant bound + the
   complementary construction `κ_g = 1` vs per-mode `1e4`; 14f tied padded restart, dense-equal,
   new levels EXACTLY 0 in both the t3svd spectrum and the companion `svd_s`, untied-input
   rejection; 14g restart escape — new `s_g` level 0 → O(1) mass within two Newton steps; 14h
   end-to-end continuation loop from rank-1 zeros to exactly the target shared ranks, rel err
   1e-10, `g0norm_newton` pinned per the rank_continuation.md warm-start guidance). CHANGELOG
   extended. Docs pointer in `docs/rank_continuation.md` deferred to slice 12 with the rest.
9. **DONE** (`988d0777`) uniform grouped truncation family: backend `_ut3svd_shared_supercores`
   (the two-phase in scan/supercore form -- TT-bond-only scan, polymorphic center collection,
   per-group SVDs on static gathers + `xnp.concatenate`, lossless left re-orth; spectra masked to
   the FINAL masks = the ragged trim) + `ut3svd(sharing=)` / `_reduce_left_to_right(sharing=)` /
   `ut3_rank_adjustment_sweep(sharing=)` (partition reversed as `tuple(reversed(sharing))`) +
   `compute_raw_sweep_ranks(sharing=)` (the grouped MULTI-PASS recurrence: right-orth, capped
   phase-1 bonds, lossless right, group concat sizes `min(n_g, Σ rL·rR, cap)`, lossless left --
   pinned == ragged grouped t3svd output ranks over 88 randomized structures/caps BEFORE
   implementation) + `ut3_sharing_residual`/`ut3_tucker_factors_shared` (masked content;
   structurally-unequal group rank masks raise) + `uniform_minimal(x0, sharing=)` (the untie
   hazard closed: plain reduction clips (4,4,2)->(2,4,2) on the ceiling structure) + frontend
   `UniformTuckerTensorTrain.t3svd(sharing=)` / `rank_adjustment_sweep(sharing=)` /
   `has_shared_tucker_factors` with the safe-mode tied checks. Tests: TestUniformShared in
   test_sharing.py (contract vs ragged incl. forced padding + varying-rank stacks + per-element
   caps, exact masks == ragged ranks, garbage robustness, dispatch anchor bit-identical,
   adjustment/minimal equivalences, per-element checker verdicts) + jit entries in test_dispatch.
10. **DONE** (`335a4971`) uniform companion + tied tangent machinery: `ufv_shared_frame_data`
   (delegates to the POLYMORPHIC `fv_shared_frame_data` on the frame's stored supercores —
   **deliberately NOT re-masked**: the companion must reproduce the construction's own sweep on
   the SAME arrays; masking first changes the padded SVDs' sign gauge and breaks the `<O, H>`
   pairing — found by test, a flipped bond column destroyed a group spectrum; the padded `S_i^T`
   rows vanish anyway since completion rows ⊥ the centers' row space. Contract: frames from
   `ut3_orthogonal_representations`; a `t3frame_to_ut3frame`-packed ragged frame is NOT
   guaranteed) + `ufv_share_tucker_variations`/`ufv_mean_tucker_variations` (mask-and-delegate;
   the ragged post-passes are fully polymorphic) + `shared_data=` on
   `utv_orthogonal_gauge_projection` (post-pass after the gauge loops), `utv_to_ut3` (tied
   embedding: `Udot` at every group mode — garbage-immune through the companion's masked-clean
   `U_M` — centers replace the down cores, the variation block rebuilt at the UP width `nU` with
   singleton blocks zero-padded and block masks = the frame's up mask at group modes), and
   `utv_retract` (tied embedding + `ut3svd(sharing=groups_to_labels(...))`). Tests
   (TestUniformSharedTangent): **gauge-invariant comparisons only** — each layer builds its own
   frame (padded SVD sign gauges differ), so compare represented DENSE tangents/points + the
   invariant group spectrum, on SHARED-MINIMAL structures (at non-minimal ranks the two layers
   legitimately build different frames — the pre-existing reason uniform fitting requires
   `uniform_minimal`). Companion==ragged==dense spectra; tied projection/retraction == ragged at
   ~1e-15; threading == separate post-pass; idempotence; the corewise mean twin; variation-padding
   garbage robustness; the jit chain (companion + tied projection + tied retraction, masks/groups
   closed over) in test_dispatch.
11. **DONE** (`3fe13802`) shared uniform geometries + fitting gates: backend factories
   `uniform_manifold_ops`/`uniform_corewise_ops`/`uniform_geometry_ops` take `sharing=`
   (closures capture `groups` beside the masks; manifold populates `precompute` with
   `ufv_shared_frame_data`, project/retract thread `shared_data=aux`; corewise = the mean
   post-pass, no companion); `uniform_least_squares_problem(sharing=)` with a SHARED-minimal
   gate. `SharedGeometry` widened to the uniform singletons (`is_uniform`, 4-way `base_name`
   + pytree map, `_is_manifold_kind`; every method branches -- uniform tangent ops call the
   `utv_*`/`ufv_*` machinery directly with `um._ut3variations_from_data`/`_ut3_from_data`,
   the `t3m._require_orthogonal_frame` cross-module precedent; the uniform retract's
   tied-coordinates safe check compares against the MASKED variations -- raw padding garbage
   would false-fail it). Gates: `_uniform_geometry_name` returns `(name, sharing)`;
   `_setup` threads `uniform_minimal(x0, sharing=)` + the problem `sharing=`;
   `_geometry_ops`/`_ragged_frame`/`_backend_geometry_ops` reject layer mismatches;
   `_uniform_model` accepts the wrapper and computes `geometry_aux = precompute_aux(frame)`;
   `UniformGaussNewtonModel` gains the `geometry_aux` LEAF (pytree children now 5) + `_project`
   at all four sites + a sharing-aware `_ubgeom`. Tests (TestUniformSharedGeometry):
   deterministic gd trajectories == ragged shared (needs a tied NONZERO SHARED-MINIMAL start:
   zero starts have arbitrary frame completions, non-minimal starts are reduced on the uniform
   path only -- both legitimately diverge the layers); corewise match; newton_cg recovery at
   true ranks (3e-11) with every output tied; model invariants == ragged (objective, grad
   norms, quadratics, <g,Hg>, regularized -- the regularizer exercising the sharing-aware
   `_ubgeom`); layer-mismatch gates both directions; compile-once
   (test_jit_shared_uniform_gauss_newton_model: ONE trace across rebuilt models).
12. **DONE** (`6bc725a5`) the docs: `docs/sharing.md` (the format; grouped truncation + dispatch
   rule; the REQUIRED "What the group spectrum is" section — four faces, √k scale, elementwise
   domination, the d=2 blend, one-tensor spectra families; the two geometries / one principle two
   formulas; rank machinery incl. the group-ceiling `get_minimal_ranks` example; batching; uniform;
   Scope with sharing ≠ symmetry + the weights×sharing `from_t3svd` caveat) +
   `docs/contributor/sharing_internals.md` (S_i machinery + measurements table, decision-7
   revision, two-phase rationale + erratum, tied embedding, restart analysis, precompute slot, the
   uniform delegation + the two hard-won lessons, smaller decisions incl. the checker-is-a-method
   record) + the CI-doctested getting-started "Shared Tucker factors" section (54 -> 36 dims,
   shared fit to 1e-4, run on both envs) + the `rank_continuation.md` sharing section + the TIED
   precondition rows/section in `numerical_contracts.md` + a user-guide pointer + toctrees wired
   (design_notes + contributor_guide) + CHANGELOG doc pointers. Sphinx -W green.
13. **DONE** (`f5f17dfe`) the example, redesigned per Nick (2026-08-20):
   `examples/fit_shared_factors_jetted_probes.py`. Target = the GROUPWISE-symmetric
   `T[i,j,k,n,o] = Σ A[i,j,k,l] B[l,m] C[m,n,o]` (A, C Hilbert; B random) — symmetric within
   groups {0,1,2} and {3,4} (different sizes across groups), `sharing=(0,0,0,1,1)` mirroring it;
   showcases the arbitrary-partition generalization, not the degenerate all-modes case. Data =
   noisy probe-derivative jets (orders 0..2, per-order ω, unit-norm vectors). The comparison:
   the SAME adaptive continuation fit (continuation_ranks + resize warm starts + the
   g0norm_newton pin) twice, differing ONLY in geometry — shared_manifold vs MANIFOLD. Verdict
   (seed 0; effect verified robust across seeds 0–2 at two problem sizes before fixing
   constants): shared best-by-validation true error 3.44e-2 at DOF 99 vs unshared 5.32e-2 at
   DOF 156 — ~35% lower error, ~37% fewer parameters; both validation curves show the
   overfitting turn, shared later/lower. Includes the cor:sym demo (s_g = √3·σ to 9.6e-16) and
   the tied-iterates assertions. Runtime ~1.5 min. NOTE the honest cap, stated in the
   docstring: TT parameters are never tied, so the advantage scales like √(DOF ratio) — the
   dramatic 3.5× gap seen at MAX_NEWTON=25 was an under-convergence artifact; the fair-budget
   (MAX_NEWTON=30–40) gap is the real one.

**ALL SLICES 0–13 DONE — the feature is complete** (code, tests, docs, example).

### 8b. Implementation state — what exists where (for a fresh context)

- **`backend/sharing.py`** (all of): `validate_sharing(sharing, shape) -> groups`;
  `nontrivial_groups(groups)`; `groups_to_labels(groups)` (inverse of validate);
  `t3_sharing_residual(x, sharing)` (per-stack max relative factor deviation; inf on
  zero-ref/nonzero-other; structural rank/size errors always); `t3_tucker_factors_shared(x,
  sharing, rtol=1e-9)`; `t3_share_tucker_cores(x, sharing)` (mean in DRIFT form `ref +
  mean(diffs)` — plain `sum/k` perturbs the last ulp already at k=3; ONE array per group);
  `T3SharedFrameData(groups, row_splits, centers, svd_U, svd_s, svd_Vt)` (arrays leaves,
  statics aux; `eq=False`); `fv_shared_frame_data(frame_data, groups)` (re-sweep
  `tt_right_orthogonalize(left, return_variation_cores=True)` — bit-identical to construction
  — then `S_i^T = einsum('...axb,...aub->...xu', O_i, H_i)` against the STORED down cores,
  then per-group batched thin SVD of `M_g = concat(S_i^T)`); `fv_share_tucker_variations`
  (manifold post-pass via `_tied_solve` clip-pinv; dtype-aware rcond); `fv_tied_ambient_directions`
  (returns `Udot` per group — the retraction consumes it); `fv_mean_tucker_variations`
  (corewise post-pass, drift form).
- **`backend/t3_svd.py`**: `t3svd(..., sharing=None)` — dispatch: None/all-singleton → the
  literal existing sweep (bit-identical, tested); else `_t3svd_shared` (phase 1 TT-bond sweep
  with Tucker steps SKIPPED; phase 2 collect centers via one right sweep; phase 3 all Tucker
  SVDs at once — singletons per mode, groups on the concatenation, `Y` applied to the
  right-orthogonal chain, ONE factor array per group, `s_g` reported at every group mode;
  phase 4 lossless left re-orth restoring the left-orthogonal contract, reported TT spectra
  TRIMMED to final bond dims — continuation reads ranks off sval lengths). Helpers
  `_up_matricization`, `_reversed_groups`. `t3_rank_adjustment_sweep(..., sharing=)`
  ('right_to_left' via mode reversal + `_t3svd_shared` uncapped/assume_orthogonal).
  `t3_share_tucker_factors` lives HERE (not sharing.py — import cycle: t3_svd imports sharing):
  the exact common-span rewrite (one stacked-factor SVD per group; coefficients are the SVD's
  own row blocks, valid for arbitrary factors; NO pre-truncation) + grouped `t3svd` doing ALL
  selection.
- **`backend/tv_operations.py`**: `shared_data=None` kwargs on `tv_orthogonal_gauge_projection`
  (post-pass fires after the gauge loops), `tv_project_t3/dense_onto_tangent_space`
  (pass-through), `tv_to_t3` (the TIED embedding: substitute `Udot` into the group modes'
  variation slots and the companion's centers `H_i` into their down-core slots BEFORE the
  existing assembly — drop-in; then re-assign one doubled factor per group), `tv_retract`
  (tied embedding + grouped t3svd via `groups_to_labels`).
- **`backend/optimizers.py`** (BREAKING): `GeometryOps` gains `precompute` (frame -> aux);
  `project`/`retract` signatures are `(frame, var, aux=None)` everywhere (both ragged
  singletons + both uniform factories in `uniform_fitting.py` updated); `LocalModel.geom_aux`
  leaf field (built once in `Problem.local_model`, passed to project/retract);
  `shared_geometry_ops(base, groups)` factory beside the singletons (corewise variant re-ties
  structurally after `corewise_add`).
- **`t3toolbox/shared_geometry.py`**: `SharedGeometry` (value-based `__eq__`/`__hash__` over
  `(base_name, sharing)`; zero-leaf pytree; `groups(shape)` canonicalizes lazily — shape is
  unknown at construction), `shared`/`shared_manifold`/`shared_corewise`; methods: `frame`
  (safe-mode tied check), `shared_frame_data`, `precompute_aux` (companion / None),
  `project(v, shared_data=None)`, `project_oblique` (delegate; manifold only), `inner`/`norm`
  (delegate), `randn`/`randn_like` (delegate+project), `retract(p, shared_data=None)`
  (safe-mode: ORTH + tied factors + tied COORDINATES via one extra post-pass compare),
  `project_ambient`, `transport` (tied check on new_frame). NO full-shared-rank precondition
  anywhere (restarts sit on the stratum).
- **Gates**: `optimizers._geometry_ops(geometry, shape=None)` (the one call site passes
  `x0.shape`); `fitting._ragged_frame` accepts the wrapper; `fitting._backend_geometry_ops
  (geometry, shape)` (called from `_bgeom` with `self.frame.shape`); `_ragged_geometry_aux`
  helper; `GaussNewtonModel.geometry_aux` leaf (pytree children now 5) + `_project` helper
  used at all four project sites; all six model factories thread the aux. Uniform gates
  (`_uniform_geometry_name`, `_uniform_model`, `_ubgeom`) untouched — `SharedGeometry`
  rejects uniform bases at construction until slices 9–11.
- **Frontend `tucker_tensor_train.py`**: `t3svd(sharing=)` + `rank_adjustment_sweep(direction,
  sharing=)` (both with the safe-mode tied check: `safety.checks_active` + `effective_rtol` +
  residual `.all()`), `share(sharing, ...)` method; new imports `backend_sharing`, `safety`.
- **`backend/ranks.py` (slice 7)**: `compute_minimal_ranks(..., sharing=None, use_jax=False)` —
  dispatch None/all-singleton → the literal existing sweep; else within-group input-rank equality
  validated (structural ValueError — an unequal proposal is not a shared rank vector), then the
  single-pass shared sweep (phase 3 left-to-right, Tucker step BEFORE TT step at each core, the
  group ceiling `min(N_g, Σ min(N_g, rL·rR))` re-evaluated at every group-mode visit with the
  CURRENT bonds, assigned group-wide); dual-mode (sequence / stacked array) preserved — every new
  op is elementwise `xnp.minimum`/sum, so the uniform slices get the batched path for free.
  `compute_manifold_dim(..., sharing=None)` — shared reduction first, TT term unchanged, ONE
  Stiefel term per group. `frame_has_minimal_ranks(..., sharing=None)` — returns False (not an
  error) on untied up-ranks. `ranks.py` now imports `backend.sharing` (no cycle: sharing imports
  only `tt_orthogonalization` + `common`). Frontend: `manifold_dim(s, sharing=)`,
  `get_minimal_ranks(..., sharing=)` (the group-ceiling doctest pair `((2,4,2))` vs `((4,4,2))`),
  the `has_shared_tucker_factors` method (see the §4.9 export revision). Root exports: the three
  geometry names. Verified BEFORE implementation (scratch, promoted to tests): sweep == dense
  edge-cut ranks of tied T3s (48 hand+randomized structures), idempotent (200 trials),
  all-singleton == unshared (200 trials); dim formula == dense tied-tangent SVD rank (4
  structures incl. the group-ceiling case: shared 32 == dense 32, unshared formula says 36).
- **`backend/ranks.py` (slice 8)**: `compute_continuation_ranks(..., sharing=None)` — validates
  the group modes carry the IDENTICAL spectrum (exact `array_equal`; the grouped t3svd assigns one
  `s_g` array per group — unequal spectra raise), then the per-mode grow logic is group-consistent
  FOR FREE (equal spectra ⇒ equal kappas ⇒ equal verdicts), so only three things genuinely change:
  the three cleanup sites pass `sharing=` (shared removal), `max_grow` candidates are built one per
  GROUP (indexed by first mode; `_grow_capped_edges(sharing=)` trials the group-wide increment —
  REQUIRED, a single-mode trial would raise the within-group equality error), and the fallback bump
  stays tied for free. Frontend: `continuation_ranks(sharing=)` (threads t3svd(sharing) — safe-mode
  tied check inherited); `resize(..., sharing=)` (tied check at entry + `t3_share_tucker_cores`
  post-pass — exact on tied input, ONE array per group). The e2e loop needs the
  rank_continuation.md warm-start guidance (`g0norm_newton` pinned across levels): without it the
  fit stalls at the target level and continuation over-grows — reproduced, then fixed, in the 14h
  test.
- **Uniform layer (slice 9)**: `backend/ut3_svd.py` gains `_ut3svd_shared_supercores` (two-phase in
  scan/supercore form; tie insurance = a static `ref_index` gather; phase-3 loop over ALL groups
  incl. singletons with one SVD of the statically-gathered concatenation, `Y`/`ss` zero-padded to
  width `n` then masked by the group's cap mask; boundary spectra mirror ragged: `ss_tt0` before the
  scan, `ss_last` from the FINAL last core after phase 4, interior bonds from the scan's first `d-1`
  steps); `ut3svd(sharing=)` sizes/slices via `compute_raw_sweep_ranks(sharing=)` and masks the
  reported spectra by the FINAL masks (the ragged trim); `_reduce_left_to_right(sharing=)` uses
  shared-minimal masks + the shared sweep with `skip_orthogonalization=True`; reversal remaps the
  partition by `tuple(reversed(sharing))`. `backend/sharing.py` gains the masked uniform checkers
  (`ut3_sharing_residual` -- structurally-unequal group rank masks RAISE, so a per-mode-clipped
  "untied ranks" state is caught before any value comparison). `uniform_fitting.uniform_minimal`
  takes `sharing=` (inline shared-minimality check; routes through the grouped frontend calls).
  Frontend `UniformTuckerTensorTrain`: `t3svd(sharing=)` + `rank_adjustment_sweep(sharing=)` (safe
  mode: `checks_active(self.data[:2])` -- supercores only, masks are host) +
  `has_shared_tucker_factors`. NOT yet uniform: the companion, `utv_*` `shared_data=`, geometry
  factories, `SharedGeometry` uniform bases, the fitting gates (slices 10–11).
- **Tests**: `tests/test_sharing.py` — 10 classes (ValidateSharing, SharingCheckers,
  ShareTuckerCores, GroupedT3svd, ShareTuckerFactors, SharedPostPass, SharedFrameData,
  SharedGeometry, + slice 7's SharedMinimalRanksGroundTruth / SharedManifoldDimGroundTruth —
  dense edge-cut and tied-tangent-SVD ground truths), EVERY class seeds in `setUp` (suite-order
  RNG coupling bit once — module-level seeds run at import, before any test). Hand-worked shared
  rank/dim tuples live in `tests/backend/test_ranks.py` (SharedMinimalRanks / SharedManifoldDim /
  SharedFrameHasMinimalRanks — same structures as the dense ground truths, pure host arithmetic;
  incl. idempotence + all-singleton==unshared property sweeps, the isolated unequal-rank
  rejection, and stacked-array mode). jit entries: `test_dispatch.py::test_jit_backend`
  (residual/checker/mean/companion/tied projection), `test_jit_tucker_tensor_train` (grouped
  t3svd + adjustment), `test_jit_shared_geometry_fitting` (shared model matvec); ranks are
  host-only by doctrine — no dispatch entries. `test_frame_variations_format.py` has the
  squash_tails regression.

### 8c. Session-1 lessons not in the spec above (do not re-derive)

- **Tex erratum found by test:** the group truncation error is BOUNDED by the `s_g` tail (the
  tail = the SUM of single-mode projection errors; equality only for singletons). Fixed in
  `shared_t3_math.tex` ("Truncation error bound" remark) with the measured 4.19e2-vs-4.80e2.
- **`share` semantics:** without caps/rtol it is the LOSSLESS common-span rewrite (group rank
  `min(sum n_i, N)` — structural, consistent with unshared `t3svd`'s no-tolerance behavior);
  agreement with grouped `t3svd` holds at EQUAL truncation settings. The tex Algorithm-3
  pre-truncation (K-SVD) is recorded as an optional efficiency device — implementing it caused
  a double-truncation spectrum-reporting artifact under caps (caught by the agreement test).
- **Corewise facts:** zero cores are a critical point of the multilinear parametrization
  (corewise e2e tests start from a small TIED random point); gauged manifold coordinates can
  differ in SHAPE across a group (`nD_i = min(n, rL_i*rR_i)`) so the mean is not even defined
  on them — the strongest geometry-separation demonstration (in the tests).
- **Regularizer + shared:** `Regularizer.hessian/quadratic` call `geom.project(frame, p)`
  2-arg → the shared project recomputes the companion per reg call (correct; accepted cost;
  tiny vs a matvec). `point_tangent`'s `v_X` has zero Tucker variations → trivially tied.
- **The doctest gem:** `rank_adjustment_sweep` docstring shows the group ceiling live — caps
  `[1,1,2,1]` give shared `(3,3,2)` (kept: ceiling `1+2=3`) where the unshared sweep clips to
  `(1,2,2)` and unties.
- Verification-script results are promoted into `tests/test_sharing.py`; the float32
  measurements live in Appendix A below (the scratchpad scripts are gone with the session).

---

## 9. Explicitly deferred (record, do not implement)

- **Weights × sharing:** interaction undefined in v1; no API site takes both today — document
  in `docs/sharing.md` Scope (+ the deferred/rejected ledger) rather than erroring.
- **ε-inflation restart** (SALSA-flavored): documented fallback if the two-step escape (§4.8)
  proves slow in practice.
- **Zipper-based companion recompute** (GEMM-only): measured viable; switch only if GPU
  profiling of the precompute ever demands it.
- Automatic partition selection: permanently out of scope.

---

## Appendix: measurements on file (jax float32 unless noted; scripts in the session scratchpad)

- Re-sweep recompute vs construction: **bit-identical** (float64 and jax float32).
- Zipper recompute vs construction: ≤ 4.5e-15 (float64), ≤ 3.2e-07 (jax float32), flat in `d`
  and in `κ_TT ~ 1e6`; `‖Z_i‖_F/‖T‖ = 1.000` exactly (interior).
- Group spectrum, trailing level at `s_min/s_max ≈ 1e-4`: SVD of `M_g` **7.5e-5**, Gram-eigh
  **3.1e-1** (float32).
- Norm-of-difference pathology (context, `‖c‖/‖a‖ = 1e-3`): zipper inner product **1.3e-2**
  vs orthogonalized **6.3e-6** — the squaring mechanism; absent from the linear H-recompute.
- Tied post-pass == dense orthogonal projection: **1.6e-13** (float64).
- Multi-group dimension formula == dense tied-basis rank: **44 == 44**, on a structure where
  the unshared reduction would clip to 42.
- Padded restart: `S_i` new rows exactly 0; tied `Udot` new rows exactly 0; TT-variation mass
  in the new up-slots O(1).
