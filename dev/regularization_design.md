# Regularization framework — design note

_Started 2026-07-14. Design-only; no code yet. Scope: ship **identity (Tikhonov) regularization** now,
through a framework that composes with every optimizer / sampling kind / geometry / representation and
leaves a clean seam for **Grasedyck–Kramer-style** (inverse-unfolding-singular-value) regularization
later._

## 0. Resume from a fresh context (read this first)

We are adding regularization to the fitting objective: `min_X  ½‖ω⊙(S(X)−y)‖² + ρ(X)`. The organizing
idea: **a `Regularizer` is a term added to the objective, folded into the local Gauss-Newton model's
`objective`/`gradient`/`gn_quadratic`/`hvp` and into `Problem.objective`.** Because every optimizer and
every kind only ever touch those surfaces, regularization composes with all of them for free. Identity
is `ρ = ½λ‖·‖²` **in the geometry's own tangent metric** (→ `H_R = λ·project`); Grasedyck–Kramer is the
same interface with a different (frame-dependent) metric operator `M`. Decisions locked in §2. Two
subtleties, both **resolved & verified** (§4/§4a, 2026-07-14): (a) the attachment point `X` is a *single
gauged tangent term* `v_X` (last TT variation `= P_last`), so the tangent isometry applies to it and
`point_norm_sq/point_tangent` are trivial — the naive "sum of frame-core norms" is WRONG; (b) the manifold
retraction always emits **left-orthogonal** cores, so the line-search norm `value(x) = ‖x.tt_cores[-1]‖²`
is one core norm, no re-orthogonalization. The one genuinely hard *open* part is the **uniform layer**
(§7): the objective term is a *reduction*, so it must sum only over masked (real) supercore entries —
garbage padding must not contribute. Slice plan in §10 (S1 ragged → S2 stochastic → S3 uniform → S4
`examples/` demo → S5 docs).

## 1. Goal & scope

- **Now:** identity regularization, `ρ(X) = ½λ‖X − X_ref‖²` (default `X_ref = 0`), `λ` a scalar.
- **Framework:** must compose cleanly with the four optimizers (`gradient_descent`/`mc_sgd`/`adam`/
  `newton_cg`), all sampling kinds (`apply`/`entries`/`probe` + `_derivatives`), both geometries
  (`MANIFOLD`/`COREWISE`), and both representations (ragged + uniform). Independent of the residual
  weight `ω`.
- **Later (NOT now, but the seam is designed in):** Grasedyck–Kramer regularization — weight cores by the
  inverse of the singular values of their matrix unfoldings. This is the same `ρ = ½λ⟨·, M·⟩` with a
  frame-dependent SPD `M` built from the unfolding SVDs; it drops in as a new `Regularizer`
  implementation with **no** signature churn.

## 2. Decisions (locked this session)

1. **D1 — identity in the geometry's own tangent metric.** `ρ` contributes `H_R = λ·Π` (= `λ·project`;
   for a gauged tangent, `H_R p = λp`). On the **manifold** the tangent coordinate metric *is* the
   Hilbert–Schmidt metric (orthogonal frame + gauged variations, per CLAUDE.md), so this is exactly the
   physically-meaningful ambient ridge `½λ‖X−X_ref‖²_HS`. On **corewise** the same principle is core
   weight-decay `½λ Σ‖core_i‖²` — and, as a minor bonus, it makes the otherwise **gauge-singular corewise
   Newton Hessian strictly PD** (better-conditioned; `fitting.py:282` flags the singularity — though CG
   already converges on it since `g = 𝒥ᵀr ∈ range(H)`, so this is conditioning, not a hard fix). The trade-off accepted: the *tensor-space
   meaning* differs by geometry (HS-ridge vs weight-decay); the payoff is a uniformly clean `H_R = λI`
   that reuses `project`, and each geometry regularizing in its own natural metric.
2. **D2 — true objective regularization** (adds `g_R` to the gradient), **not** Newton-Hessian-only
   damping. Forced by the compose-with-*all*-optimizers goal: damping the Hessian alone is invisible to
   the objective/gradient, so first-order methods (`gradient_descent`/`mc_sgd`/`adam`) and even Newton's
   line search / ρ would be unregularized. To reach every optimizer, regularization **must be a term in
   the objective**, hence contributes to the gradient.
3. **D3 — `X_ref = 0` default; `λ` scalar now.** The `Regularizer` is a *protocol* (§4), so the `M`
   generalization is a new implementation, not a parameter on one class. `X_ref` as a settable reference
   is a later extension.
4. **Uniform mask-safety is a hard requirement** (§7), not an afterthought: the regularizer may only
   touch mask-aware primitives (`geom.inner`, `geom.project`, the mask-correct norm, mask-preserving
   `corewise` arithmetic) — **never raw supercores** — so garbage padding cannot contaminate `ρ`. Tested
   with garbage-padded-input robustness + exact output masks, not just dense-vs-ragged.

## 3. The math

Objective: `Φ(X) = ½‖ω⊙(S(X)−y)‖² + ρ(X)`, with `ρ(X) = ½λ⟨X−X_ref, M(X−X_ref)⟩` (`M = I` for identity;
`M = M(frame)` for Grasedyck–Kramer). At the linearization frame the GN-consistent contributions are:

| surface | data term (existing) | **+ regularizer term** |
|---|---|---|
| objective `c` | `½‖ω⊙r‖²` | `+ ρ(X) = ½λ⟨X−X_ref, M(X−X_ref)⟩` |
| gradient `g` | `Π 𝒥ᵀ(ω²r)` | `+ g_R = λ·Π M (X−X_ref)` |
| Hessian `H p` | `Π 𝒥ᵀ𝒥 Π p` | `+ H_R p = λ·Π M Π p`  (identity: `λ·project(p)`) |
| `gn_quadratic` `pᵀHp` | `‖𝒥Πp‖²` | `+ λ⟨p, M p⟩`  (identity: `λ·inner(p, p)`) |

`predicted reduction` / `evaluate(p)` inherit the reg terms because they are assembled from
`c`, `g`, `gn_quadratic`. Retraction curvature is dropped (GN-consistent, matching the data term).

## 4. The `Regularizer` interface

A protocol (small stateless objects carrying `λ` and, later, an `M`); geometry-agnostic — it receives the
geometry + frame/point and leans on their primitives:

```
Regularizer (protocol):
    value(geometry, x_cores)      -> scalar     # ρ(X) at ANY point (for Problem.objective / line search)
    gradient(geometry, frame)     -> tangent    # g_R = λ Π M (X − X_ref)   (a projected/masked tangent)
    hessian(geometry, frame, p)   -> tangent    # H_R p = λ Π M Π p
    quadratic(geometry, frame, p) -> scalar     # ⟨p, H_R p⟩  (cheap, for gn_quadratic)
```

`IdentityRegularizer(strength=λ)` is the concrete implementation:
- `value      = ½λ · geometry.point_norm_sq(x_cores)`
- `gradient   = λ · geometry.point_tangent(frame)`         (already gauged/projected — no `project` needed)
- `hessian    = λ · geometry.project(frame, p)`            (= `λp` for gauged `p`)
- `quadratic  = λ · geometry.inner(project(p), project(p))`

**Two new geometry primitives** (added to `GeometryOps` backend + the frontend geometries). Both are
verified constructions (2026-07-14 empirical check), **not** the naive "sum of frame-core norms" — that is
WRONG (the frame's orthogonal cores have norm `√rank`, not content: measured `42291.8` vs the true
`‖X‖²=42280.8`).

- `point_tangent(frame) -> tangent` (= `v_X`, the attachment point as a gauged tangent): **all variations
  zero except the last TT variation, set to the frame's last left_tt core `P_last`.** This is the key
  fact — the attachment point `X` is a *single* frame/variation term: `dense(v_X, no shift) = X`, and it
  is **already gauged** because the last TT variation is the one slot with no gauge condition. So
  `g_R = λ·v_X` needs no `project`. Corewise: the cores as a tangent (trivial).
- `point_norm_sq(frame) -> scalar`: `‖X‖²` in the coordinate metric. Manifold: `‖P_last‖²` (the last
  left_tt core) — because the frame is left-orthogonal, `‖X‖_HS = ‖P_last‖ = ‖v_X‖_coord` (all three
  verified equal). Trivial, no dense tensor. Corewise: `Σ‖core_i‖²`.

**Why the isometry applies (the shifted-vs-zero-centered subtlety, resolved).** The isometry
`corewise_inner(v,v') = ⟨dense(v),dense(v')⟩_HS` (Appendix A.3) holds on the **zero-centered** gauged
tangent space, not for the attachment point directly. Resolution: `X = dense(v_X)` with `v_X` a gauged
tangent, so the local model of `ρ(X+dense(p)) = ½λ‖v_X + p‖²_coord`; since `v_X + p` is gauged (sum of
gauged), the isometry extends bilinearly to exactly `½λ‖X + dense(p)‖²_HS`. Verified:
`⟨v_X,p⟩_coord = ⟨X,dense(p)⟩_HS`.

**`H_R = λI` is the EXACT Riemannian Hessian on the manifold, not just a GN approximation** — `grad =
X` is fully tangent (zero normal component: scaling preserves rank), so the Weingarten/curvature term
vanishes. Adding `λI` also makes the gauge-singular **corewise** Newton `H` strictly PD (a conditioning
improvement; CG already converged since `g ∈ range(H)`).

Grasedyck–Kramer later = a `SingularValueRegularizer` implementing the same four methods with `M(frame)`
from the unfolding SVDs; `point_norm_sq`/`point_tangent`/`project` compose with an added `M`-apply.

### 4a. Computing `value(x)` — the line-search norm (retraction is already left-orthogonal)

`value(x)` is a norm of a **T3** (a point), not a tangent, so it does not use the tangent isometry. Key
finding (verified): the manifold **retraction output is already left-orthogonal** — `MANIFOLD.retract` →
`tv_retract` → `t3svd`, whose contract is *"the result is always left-orthogonal."* So for a
left-orthogonal T3 the content is in the last TT core: `‖x‖_HS = ‖x.tt_cores[-1]‖` (verified
`206.142320 == 206.142320`). Both reg-evaluation sites are therefore already orthogonal and `value` never
re-orthogonalizes:
- **Manifold `value(x) = ½λ‖x.tt_cores[-1]‖²`** — one core norm. At the current-iterate frame this is the
  same `‖P_last‖²` as `point_norm_sq(frame)` (internal consistency). At a retracted line-search candidate,
  `x_trial` is left-orthogonal from `t3svd`. **No waste on rejected candidates** — the retraction (hence
  the left-orthogonalization) is already paid to form `x_trial` for the *data* term `‖S(x_trial)−y‖²`; the
  reg adds ~one core norm per candidate.
- **Corewise `value(x) = ½λ Σ‖core_i‖²`** — the raw coordinate norm (weight-decay); corewise retract is
  additive (non-orthogonal cores), and this needs no orthogonality.

**Precondition — handled the house way (backend check-free, checker tools provided).** "Read the last TT
core" is correct **only for a left-orthogonal input**. This is a *numerical precondition*, so (per the
project's structural-vs-numerical split) the **backend `value` is check-free** — it reads the last core,
correct for its call sites (the frame is orthogonal by construction; retracted candidates are
left-orthogonal from `t3svd`). Safe-mode enforcement is **frontend-only** (CLAUDE.md): the frontend `value`
routes a safe-mode `t3_orthogonality_residual` check (skipped under jit/unsafe) so a stray non-orthogonal
input errors loudly. **The backend user self-checks — and the tools already exist** (public in
`backend/t3_orthogonalization.__all__`): `t3_orthogonality_residual(cores)` to *check* left-orthogonal
form, `t3_left_orthogonalize(cores)` to *make* it so (then read the last core). So there is **no
backend/frontend inequality** here and **no new checker to build** — the reg feature just reuses the
existing orthogonality checker (the same one the manifold ops already lean on). Note `point_norm_sq(frame)`
has **no** precondition (a frame is orthogonal by construction); only `value(x_cores)` on an arbitrary
point carries the left-orthogonal assumption. (Corewise has no such precondition at all.) *The one open
sub-question — whether the **frontend** `value` should error in safe mode (default) or silently
orthogonalize-then-read — stays deferred; it does not touch the Newton-CG path or the backend contract.*

## 5. Where it threads (plumbing points)

- **`Problem`** (backend) gains a `regularizer` field (default `None`). **`Problem.objective(x)`** adds
  `regularizer.value(geom, x)` (so line search / plateau / actual-reduction ρ see it).
- **`LocalModel`** (backend) folds the reg terms into `objective`, `gradient`, `gn_quadratic`, `hvp`.
- **Frontend `GaussNewtonModel`** mirrors it; the model factories (`apply_model`/`probe_model`/… and the
  uniform `_uniform_model`) accept `regularizer=`.
- **Frontend `optimizers.*`** accept `regularizer=` and pass it into the `Problem` (`_setup`), exactly
  like `weight=` today. `newton_cg`/`gradient_descent` get it deterministically; `mc_sgd`/`adam` need the
  stochastic-scaling decision (§8.1).
- `ω` (residual weight, on the kind) and `λ`/`M` (the prior) are **orthogonal** — neither touches the
  other's machinery.

## 5a. Backend-user surface (the razor check — treat raw-`.data` users equally)

A raw-`.data` user must regularize with the **same one kwarg** as a frontend user. This holds iff
everything reg-specific is **backend-homed** (this is a hard requirement of the plan, not an afterthought):

- **`Regularizer` protocol + `IdentityRegularizer` live in the backend** — a new `backend/regularization.py`
  (check-free; depends only on the `GeometryOps` interface), re-exported from `backend/optimizers.py`. **No
  frontend import needed** to construct or use a regularizer. (The frontend also re-exports it for its
  users' convenience — symmetric, not privileged.)
- **`point_norm_sq` / `point_tangent` / `value` are methods on `GeometryOps`** (`MANIFOLD_OPS` /
  `COREWISE_OPS`), not only the frontend geometry singletons — so `LocalModel` can compute reg on raw
  tuples.
- **`regularizer=` is a param on `bopt.least_squares_problem`** (+ fields on `Problem` and `LocalModel`,
  default `None`). Attaching is one kwarg — the same ergonomics as `weight`-on-the-kind. Don't force a
  raw-data user to `dc.replace(problem, regularizer=…)` by hand (the "one fiddly step short" corollary).
- **The reg folding lives in `LocalModel`** (`objective`/`gradient`/`gn_quadratic`/`hvp`) — the backend
  twin of `GaussNewtonModel`. So a backend user **rolling their own optimizer** via `problem.local_model(x)`
  gets reg-aware `.gradient` / `.hvp` / `.gn_quadratic` for free, with zero reimplementation.

Backend workflow — identical shape to the frontend:

```
# backend (raw .data)                         # frontend (equivalent)
reg = bopt.IdentityRegularizer(1e-3)          reg = <same> (re-exported)
problem = bopt.least_squares_problem(         x, s = topt.newton_cg(MANIFOLD, 'probe', ww, data, x0,
    bopt.MANIFOLD_OPS, bfit.PROBE,                                  regularizer=reg)
    ww, data, regularizer=reg)
x, s = bopt.newton_cg(problem, x0)
```

**On the `value` precondition — no inequality, by design.** Safe-mode numerical preconditions are
**frontend-only** (CLAUDE.md: the backend is check-free by design; the backend user does their own checks).
That is *not* an inequality **as long as the backend provides the checker tools** — and it already does:
`t3_orthogonality_residual` (check left-orthogonal form) and `t3_left_orthogonalize` (make it so) are
public in `backend/t3_orthogonalization.__all__`. So a backend user guards `value` exactly as the frontend
safe-mode does, using the same function. The reg feature adds **no new checker** and needs none. The house
rule to honor: *whenever a backend op carries a numerical precondition, make sure the matching non-enforcing
checker is a public backend function* — here it already is. Everything else in the plan is symmetric.

## 6. Per-geometry specialization

| | manifold | corewise |
|---|---|---|
| tangent metric | HS (orthogonal + gauged) | Euclidean core entries |
| `ρ` (identity) | `½λ‖X−X_ref‖²_HS` | `½λ Σ‖core_i − ref_i‖²` |
| `H_R p` | `λp` (**exact** Riemannian Hessian, no curvature) | `λp` (project = id) |
| bonus | — | gauge-singular Newton `H` → strictly PD (conditioning; CG already converged) |
| `point_tangent` = `v_X` | last TT variation `= P_last`, else 0 (already gauged) | the cores as a tangent (trivial) |
| `point_norm_sq` | `‖P_last‖²` (last left_tt core) | `Σ‖core_i‖²` |
| `value(x)` (line search) | `‖x.tt_cores[-1]‖²` (x is left-orthogonal from retract) | `Σ‖core_i‖²` |

## 7. Uniform mask-safety (the critical section)

A stacked uniform supercore has **garbage** in the out-of-mask (padding) rank slots — don't-care values.
The risk is sharpest in the **objective term `ρ`, because it is a reduction**: a naive
`½λ·Σ(supercore²)` over the full supercore would sum garbage → not just wrong but **nondeterministic**
(don't-care entries). Same failure class as phantom-rank / doubled-boundary
(`docs/contributor/testing_strategy.md`). `g_R` and `H_R p` are *tangents*, so garbage there is inert
(every downstream reduction — `gn_quadratic`, the CG inner — is already masked), but their **output masks
must be exact** so nothing downstream trips.

**Design rule (guarantees safety by construction):** the regularizer routes *only* through mask-aware
primitives and never indexes raw supercores.

| contribution | expressed via | garbage-safe because |
|---|---|---|
| `ρ` | mask-correct norm — the uniform tangent/HS norms already sum only masked content (`uniform_manifold.py:103, 250, 260`) | reduction skips padding |
| `g_R` | `λ · geom.project(point_tangent)` | `project` zeros variation padding → exact mask |
| `H_R p` | `λ · geom.project(p)` | re-project keeps padding zeroed → exact mask |
| `gn_quadratic` | `λ · geom.inner(p, p)` (masked) | padding never summed |

So D1 (identity in the geometry's own tangent metric) is not merely elegant — it is what makes the
uniform layer *safe*, because it inherits the mask-correctness already built into `inner`/`project`/the
tangent norms. A regularizer that reached into `.tucker_supercore` directly would reintroduce the garbage
bug. **Interface consequence:** the regularizer is handed tangents + the geometry, never supercores.

## 8. Composition subtleties (decide per slice; none block the ragged Newton-CG slice)

1. **Stochastic optimizers (`mc_sgd`/`adam`).** `ρ` is deterministic; the data term is a minibatch **sum**
   over `batch` of `n` measurements, so `E[minibatch data-grad] ≈ (batch/n)·full data-grad`. To keep the
   reg at the right relative scale, add `(batch/n)·g_R` per step (or `(n/batch)·`data + `g_R`). Decide in
   the stochastic slice; note the chosen convention.
2. **`gn_quadratic` / `evaluate` must include the reg term** (`+ λ⟨p,Mp⟩`) or the line-search denominator
   and predicted-reduction ρ silently disagree with the true objective. Easy to forget.
3. **`ω`-independence** — asserted by a test (a reg + a nontrivial `ω` compose without cross-talk).
4. **Uniform mirror is REQUIRED, not optional** — tracked by an un-missable skipped test (§10).

## 9. Correctness spec & verification

- **Spec (uniform):** the equivalence contract per contribution — `to_ragged(reg_uniform(X)) ==
  reg_ragged(to_ragged(X))` for `value` (scalar), `gradient`, `hessian` (tangents).
- **Ragged correctness:** finite-difference the objective — `g_R ≈ ∂ρ/∂p`, `H_R ≈ ∂²ρ/∂p²` at the frame;
  and the composed `Φ` gradient/Hessian vs a dense-rebuild reference.
- **Uniform (beyond dense-vs-ragged, which is blind to garbage):**
  (i) **garbage-padded-input robustness** — fill out-of-mask padding with garbage (and a NaN variant),
  assert `ρ`/`g_R`/`H_R p` unchanged and finite (NaN-invariance proves no padding leaked into a
  reduction); (ii) **exact output masks** on `g_R` and `H_R p` (derived non-circularly).
- **jit dispatch:** add the reg-bearing model/optimizer path to `test_dispatch` (a stray `np.*` on a
  tracer raises).

## 10. Implementation slices (ragged first)

1. **S1 — ragged identity, Newton-CG. ✅ DONE (2026-07-14, uncommitted).** New `backend/regularization.py`
   (`Regularizer` + `IdentityRegularizer`, re-exported from `backend/optimizers.py` and `optimizers.py`);
   `GeometryOps.point_norm_sq` + `point_tangent` (manifold: `v_X` via `tv_project_t3_onto_tangent_space`,
   norm via last TT core; corewise: cores-as-tangent, `corewise_dot`); `regularizer` field on `Problem` +
   `LocalModel`, folded into `objective`/`gradient`/`gn_quadratic`/`hvp` + `Problem.objective`;
   `least_squares_problem(regularizer=)`; frontend `newton_cg`/`gradient_descent` `regularizer=` (+ uniform
   guard → `NotImplementedError`, S3). Tests: `test_identity_regularizer_contributions` (value + FD total
   gradient + `λ‖Πp‖²`, both geometries), `test_manifold_point_tangent_is_vX` (`dense(v_X)=X`, `‖X‖=‖P_last‖`),
   `test_regularized_newton_cg_shrinks` (λ=0 recovers, ridge shrinks monotonically), frontend
   `test_newton_cg_regularizer` (+ ω-compose + uniform guard), + a reg subTest in `test_jit_paths_recover`.
   Full suite green (627). **NOTE: frontend `GaussNewtonModel` reg wiring deferred to S1b** (the frontend
   *optimizer* path works fully via the backend; the roll-your-own frontend model is a smaller surface).
1b. **S1b — frontend `GaussNewtonModel` reg parity. ✅ DONE (2026-07-14, uncommitted).** `regularizer=` on
   all six model factories (`apply`/`entries`/`probe` + the three `*_derivatives`; uniform x →
   `NotImplementedError`); `regularizer` field on `GaussNewtonModel` folded into
   `objective_value`/`gradient`/`gn_quadratic`/`gn_hessian`/`evaluate` (delegates to the backend
   `Regularizer` via `_backend_geometry_ops(self.geometry)`, wrapping the raw tangent as a `T3Tangent`).
   **jax pytree registration updated to carry `regularizer` as static aux** (else it drops on a jit
   round-trip — fixed + verified). Test `test_fitting.py::test_regularizer`: the two-form identity and
   `gn_quadratic==pᵀHp` hold with reg on all kinds/geometries, objective gains ρ, uniform rejected.
   Verified the frontend model == the backend `LocalModel` with reg (obj/grad/gn_quadratic/hvp) and the
   pytree round-trip preserves the regularizer. Full suite green (628).
2. **S2 — stochastic optimizers.** `mc_sgd`/`adam` `regularizer=` with the (batch/n) scaling (§8.1).
3. **S3 — uniform twin.** Uniform `point_norm_sq`/`point_tangent`/`value` (read `P_last` mask-correctly);
   uniform `GaussNewtonModel` + optimizers `regularizer=`. Tests: equivalence-to-ragged +
   **garbage-robustness + exact masks** (§9). Guarded by a skipped test from S1 so it cannot silently slip.
4. **S4 — user-facing example** in `examples/` (Nick wants this): a runnable demo of identity
   regularization stabilizing an ill-posed fit — e.g. recover a low-rank tensor from *too few* / noisy
   probes where the unregularized Newton-CG fit is unstable, and `regularizer=IdentityRegularizer(λ)`
   recovers a sensible solution; show the λ-vs-error tradeoff. Follows the examples-first doctest
   convention; wire into the existing `examples/fit_*` family.
5. **S5 — docs.** `docs/fitting_and_optimization.md` new § (+ link the example); CLAUDE.md shipped-surface.

## 11. Grasedyck–Kramer seam (future, not now)

A `SingularValueRegularizer` implementing the same four `Regularizer` methods, with `M(frame)` a
diagonal/block operator built from the singular values of the cores' matrix unfoldings (penalize
directions with small σ — poorly-determined — more). Reuses `point_tangent`/`project` composed with an
`M`-apply on tangents. No change to `Problem`/`LocalModel`/optimizers/kinds — that's the payoff of
designing the protocol now. (Math reference to pin down when we get there: Grasedyck & Kramer, stable
ALS / preconditioned tensor optimization.)

## 11a. Follow-up (independent of the reg feature): `already_left_orthogonal` amortization

The manifold retraction (`t3svd`) always emits **left-orthogonal** cores, but the next Newton step's
`geom.frame(x) = t3_orthogonal_representations(x)` currently re-orthogonalizes from scratch.
`t3_orthogonal_representations` already takes `already_left_orthogonal=True`, which skips the
left-orthogonalization sweep. Passing it after a retraction speeds up **every** Newton step (not just
regularized ones), so file it as a **separate optimization**, not part of the reg work — but the reg
`value(x)` design (which relies on the retraction's left-orthogonality) makes it salient. Watch: the
optimizer would need to carry the "this point came from a retraction / is left-orthogonal" flag from
`retract` to the next `local_model`.

## 12. Risks / watch-list

- **`point_tangent` = `v_X`** — **RESOLVED** (§4, verified 2026-07-14): last TT variation `= P_last`, else
  zero; already gauged; `dense(v_X)=X`; `‖v_X‖_coord=‖X‖_HS`. Not the naive sum-of-frame-norms (that is
  wrong). S1 test confirms across structures/ranks + the left-orthogonal convention.
- **`value` non-orthogonal edge case** (§4a) — a raw non-left-orthogonal point reaching `value` reads the
  wrong "last core" norm. **Resolved the house way:** backend `value` is check-free; the precondition
  checker (`t3_orthogonality_residual`) + orthogonalizer (`t3_left_orthogonalize`) are already public
  backend tools, so backend and frontend users are equal. Only open sub-question: whether the *frontend*
  `value` errors (safe-mode) or auto-orthogonalizes — deferred, doesn't touch Newton-CG or the backend.
- **Uniform garbage** — the whole of §7; the NaN-padding test is the tripwire.
- **Stochastic scaling** (§8.1) — a wrong scale silently over/under-regularizes; pin with a test that the
  full-batch `mc_sgd` step matches the deterministic reg gradient.
- **`gn_quadratic`/`evaluate` forgetting the reg term** (§8.2) — silent ρ/line-search inconsistency.

## 13. Out of scope (for the identity feature)

- Grasedyck–Kramer / any non-identity `M` (seam only; §11).
- `X_ref ≠ 0` references (default 0; easy later extension).
- Non-quadratic regularizers (L1 / TV / entropy) — the protocol admits them, but not designed here.
- Auto-selecting `λ` (discrepancy principle, L-curve) — user supplies `λ`.
