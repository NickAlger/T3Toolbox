# `fitting.py` — least-squares operators with base-sweep reuse — plan & handoff

*Design locked with Nick (2026-06-18). This doc is the reviewable plan for the new fitting layer and the
first vertical slice (the **apply** exemplar). Build starts after Nick reviews this.*

## 1. Goal & separation of concerns

A new module pair — backend `backend/fitting.py` (pure functions on raw tuples) + frontend `fitting.py`
(thin dataclass frontend) — providing the **least-squares building blocks** for fitting a Tucker tensor
train from sampled data: the **objective**, **gradient**, **Jacobian apply** `J`, **Jacobian-transpose
apply** `Jᵀ`, and **Gauss-Newton Hessian apply** `JᵀJ`, for every sampling type
(`probe`/`apply`/`entries` × `plain`/`derivative` × `tangent`/`corewise`).

The split of concerns is deliberate:

> **Probing is about the Tucker tensor train. Fitting is about the gradients.** `probing.py` /
> `probe_derivatives.py` own the geometric sampling primitives (the base sweep + the `J`/`Jᵀ` assembles);
> `fitting.py` *composes* them into objective/gradient/Hessian, and owns the **reuse** of the expensive
> shared computation. Solvers (CG, Newton, L-BFGS) live elsewhere (`optimizers.py`), not here.

**The reuse target (the whole point).** Inside an inner CG / GN solve the **base is fixed**, but today
every `J`/`Jᵀ`/`JᵀJ` call recomputes the **base sweep** — the base-and-data-dependent edge variables
(`xis`, `mus`, `nus`, `etas`) that do *not* depend on the tangent direction or the residual. `fitting.py`
computes the base sweep **once per outer step** and threads it through every inner apply. Across `n` inner
iterations this turns `2n` base sweeps into `1`.

## 2. The base sweep, concretely (apply)

Grounding in the current `probing.py` (verified):

- Forward `apply_tangent` (723) computes `xis = compute_xis(U, ww)`, `mus = compute_mus(P, xis)`, then
  `_apply_from_xis(xis, dxis, mus, Q, O, var_tt)`. Uses **`xis, mus`** of the base sweep (it handles
  right/down inline via the variation).
- Transpose `apply_tangent_transpose` (817) computes the **full** `xis, mus, nus, etas`, then
  `_apply_transpose_assemble(c, ww, xis, mus, nus, etas, sum_over_probes)`.

So the reusable bundle is `base_sweep = (xis, mus, nus, etas)` (the transpose's superset); the forward
consumes the `xis, mus` subset. **`JᵀJ v` = transpose(jacobian(v))** reuses one `base_sweep` for both
halves. The inner assemble functions (`_apply_from_xis`, `_apply_transpose_assemble`) **already exist** as
separate pieces — the only thing the public entry points recompute is the four base-sweep lines.

## 3. Design invariants (the rules the code must hold)

1. **Frame vs. raw-core sweep — never cross them.** `F(x) = sample(x, ww)` is representation-independent,
   but `mus/nus/etas` from the **orthogonal frame** ≠ those from the **raw cores**, and the downstream
   tangent ops are defined on the *frame* sweep. So the objective is evaluated through the **same
   representation its downstream ops consume**: the **frame** for the tangent family, the **raw cores**
   for corewise. A raw-core sweep is *never* reused for a tangent op. (This crossing would be a silent
   *numerical* bug, not a shape error — hence the dataclass below bundles base + sweep so it's
   unrepresentable.)
2. **The gauge projector `Π` — the tangent family is `𝒥∘Π` / `Π∘𝒥ᵀ`, not bare.** `T3Tangent.probe`/
   `apply`/`entries` and their `*_transpose` are the **bare** single-sample Jacobian `𝒥` / `𝒥ᵀ` (no gauge
   projector — verified: "compose a gauge projection yourself"). The Riemannian least-squares Jacobian is
   `J = 𝒥 ∘ Π` and `Jᵀ = Π ∘ 𝒥ᵀ`, where `Π = T3Tangent.orthogonal_gauge_projection()` is the orthogonal
   projector onto gauged variations (`δÛ=(I−UUᵀ)δU`, `δĜ=(I−Pᴸ(Pᴸ)ᵀ)δG`; paper §6, §6.1, eqs 48–49). So
   the fitting layer **must** compose `Π`:
   - gradient `g = Π 𝒥ᵀ r`,  GN-normal operator `H = JᵀJ = Π 𝒥ᵀ 𝒥 Π` (symmetric, maps gauged→gauged).
   - `Π` needs **only orthogonality** (it uses the frame cores `U, Pᴸ`) — consistent with the frame base.
     It is **cheap and `W`-independent** (`O(core size)`, no sample stack), and it does **not** touch the
     base sweep — it just bookends the input/output, so the reuse story (§2 above) is unchanged.
   - **Matched pair — manifold ⟺ `Π`, corewise ⟺ NO `Π`.** The corewise operator is the bare `𝒥ᵀ𝒥`
     (the §6.3 substitution drops the gauge — cores are free variables). **Crossing either way gives
     silently-wrong numbers, not an error**: applying `Π` in the corewise case projects out real gradient
     components (and is anyway ill-defined — `Π` is only a projector on an *orthonormal* frame, which
     corewise's raw cores are not); omitting `Π` in the manifold case leaves the result off the gauged
     tangent space. So the choice is **structural, never a flag**: there is **no `apply_gauge` parameter**
     anywhere — manifold and corewise are *different functions*, and `Π` appears only in the manifold path.
     It is further protected by type: only the manifold family holds an orthonormal `T3Basis` for which `Π`
     is even defined; the corewise family holds raw `TuckerTensorTrain` cores.
3. **`sum_over_probes=True` throughout fitting.** The LSQ normal operator `JᵀJ = Σ_w J_wᵀ J_w` sums over
   samples, and the gradient `Jᵀr` sums the residuals. The non-summed (keep-`W`) `Jᵀ` stays in
   `manifold.py` for other uses; `fitting.py` always sums.
4. **Backend pure on basic types; frontend = frozen dataclass; backend functions are self-contained.**
   Matches the library razor: a backend user on raw `.data` tuples can do everything a frontend user can
   **without thinking hard** — in particular, **every `backend/fitting.py` function applies `Π` itself**
   (gauge-projects its inputs/outputs as needed), so a backend user can never forget the gauge and get a
   silently-wrong result. The frontend is a thin dataclass that delegates straight through. The **bare**
   `𝒥`/`𝒥ᵀ` (no `Π`) live in `probing.py` for users who *explicitly* want the un-projected Jacobian; the
   gauge is a *fitting/Riemannian* concern, so it is owned by the *fitting* layer, not pushed onto the
   caller. (Corewise fitting functions likewise own their "no `Π`" choice — the user doesn't decide it.)
5. **Structural → hard error, numerical → let through.** Same-base identity guard on the operator (like
   `T3Tangent` `+`/`inner`); stale/mismatched sweep is prevented structurally by the dataclass bundling,
   not by value-checking.

## 4. Frontend: `GaussNewtonModel` (the dataclass)

A frozen dataclass holding the base, the sample vectors, and the **residual** `r = F(base) − y`; the
expensive base sweep, the objective value, and the gradient are `@cached_property`s (idiomatic — frontend
dataclasses hold the minimal inputs, everything else is a cached_property delegating to the backend). The
caches *are* the reuse: each is computed on first touch and reused across the inner solve, GC'd when the
model leaves scope. **Not a persistent keyed cache with invalidation** — an immutable, lexically-scoped
value, exactly per Nick's spec.

The interface is **model-shaped** (value + gradient + GN-Hessian-action + the quadratic-model value) — the
exact surface an `optimizers.py` CG / trust-region / Newton solver consumes. "Model" (not "Operator")
answers the earlier objection that the object does more than apply the GN Hessian; raw `J`/`Jᵀ` are *not*
the model's job — they live in the backend (`§5`) for users who want them.

```python
@dataclass(frozen=True)
class GaussNewtonModel:
    base:     T3Basis              # the orthogonal frame (tangent family)
    ww:       Tuple[NDArray, ...]  # the sample vectors, len=d, elm_shape=W+(Ni,)
    residual: NDArray              # r = F(base) − y, shape W[+C]  (the data residual at the base)

    @cached_property
    def _base_sweep(self):                 # (xis, mus, nus, etas) -- computed ONCE, reused
        return fb.precompute_base_sweep(self.base.data, self.ww)

    @cached_property
    def objective_value(self) -> NDArray:  # c = ½‖r‖²        (the model constant; shape () or C)
    @cached_property
    def gradient(self) -> T3Tangent:       # g = Π 𝒥ᵀ r       (sum_over_probes=True), a gauged tangent at self.base

    def gn_hessian(self, p: T3Tangent) -> T3Tangent:  # H p = Π 𝒥ᵀ 𝒥 Π p   (H = JᵀJ, the GN Hessian)
    def evaluate(self, p: T3Tangent) -> NDArray:      # m(p) = c + ⟨g,p⟩ + ½‖𝒥 Π p‖²   (see §4a)
```

- `J = 𝒥 ∘ Π`, `Jᵀ = Π ∘ 𝒥ᵀ` (invariant §3.2): `gradient`/`gn_hessian` compose the gauge projector `Π`
  around the bare probe `𝒥`. `H = JᵀJ` is the **Gauss-Newton** Hessian (the `JᵀJ` term only — the true
  Hessian's second-order residual term is dropped; that is what "Gauss-Newton" / the class name means).
- `gn_hessian`/`evaluate` take a `T3Tangent` and **require `p.basis is self.base`** (identity guard);
  `gn_hessian`/`gradient` return a **gauged** `T3Tangent` at `self.base`.
- **Corewise** is the thin wrapper (`§6`): a `CorewiseGaussNewtonModel(x, …)` builds an internal
  `GaussNewtonModel` at the substituted base `(U, G, G, G)` and maps core-deltas ↔ variations, returning
  raw `(tucker_grads, tt_grads)` for the gradient/Hessian. Same machinery, different held type + return type.
- Factory: a module-level `fitting.gauss_newton_apply(base, ww, residual)` for now; `.gauss_newton(…)`
  methods on `T3Tangent`/`TuckerTensorTrain` can come later.

### 4a. The quadratic-model value `evaluate(p)` — the requested helper

The local Gauss-Newton model around the base is

```
m(p) = c + gᵀp + ½ pᵀ H p ,    c = ½‖r‖²,   g = Π𝒥ᵀr,   H = JᵀJ = Π𝒥ᵀ𝒥Π   (GN, not the full Hessian).
```

`evaluate(p)` returns it **reusing every precomputed piece**: the cached `c` (`objective_value`), the
cached `g` (`gradient`), and the cached base sweep (for **one forward apply** `𝒥(Πp)`). Two efficiency
levers: (i) the quadratic term needs only the **forward**, not a full Hessian apply, via

```
½ pᵀ H p = ½ pᵀ Π𝒥ᵀ𝒥Π p = ½‖𝒥 Π p‖²      (Π idempotent + self-adjoint)
```

and (ii) `Π p` is computed **once** and shared by both terms (and `⟨g,p⟩ = ⟨g, Πp⟩` since `g` is already
gauged). So the model value is **one `𝒥` apply + one `Π`** (+ a cheap tangent inner + the cached scalar),
versus the *two* `𝒥` applies a `gn_hessian` costs. All of this lives in the **self-contained backend**
`quadratic_model_value` (it applies `Π` itself, §3.4); the frontend method just delegates:

```python
def evaluate(self, p):                              # require p.basis is self.base
    return fb.quadratic_model_value(                # backend applies Π, shares Πp, one bare forward
        p.variations.data, self.ww, self.base.data, self._base_sweep,
        self.gradient.variations.data, self.objective_value)
```

- The inner term uses the **Riemannian (Hilbert-Schmidt) tangent inner product** — the correct `gᵀp`:
  since `g = Π𝒥ᵀr` is the HS adjoint of `J=𝒥∘Π`, `⟨g,p⟩ = ⟨r, 𝒥Πp⟩` (the adjoint identity we test), so
  `evaluate(p)` equals `½‖r + 𝒥(Πp)‖²` exactly (a strong oracle, §9).
- `½‖𝒥Πp‖²` sums over the sample stack `W` and **keeps the base stack `C`**; a `K`-stacked `p` rides
  through → one model value per trial step (batch-evaluate candidate steps for free). Uses the
  stack-preserving reductions, not `corewise_norm` (which would collapse `C`/`K`).
- If `Hp` is already in hand (e.g. mid-CG), `½⟨p, Hp⟩` reuses it instead; standalone, the forward-only form
  is cheaper.

## 5. Backend: `backend/fitting.py` + the tiny `probing.py` split

**`probing.py` additions (small, mechanical — name & expose the existing split):**

```python
def precompute_base_sweep(base, ww) -> base_sweep            # bundles compute_xis/mus/nus/etas
def apply_jacobian_from_sweep(variation, base, base_sweep) -> z    # = _apply_from_xis, sweep injected
def apply_transpose_from_sweep(c, ww, base, base_sweep, sum_over_probes) -> (dU, dG)  # = _apply_transpose_assemble
```

The existing public `apply_tangent` / `apply_tangent_transpose` are rewritten as
`precompute_base_sweep` + the `_from_sweep` call (behavior identical; the all-in-one API stays). The
`_from_sweep` functions are the **bare** `𝒥`/`𝒥ᵀ` (no `Π`); they are public so `fitting.py` composes
*public* probing functions, not privates (the razor: the split is a named capability with a docstring +
test).

**`backend/fitting.py` (apply, tangent)** — composes the bare `𝒥`/`𝒥ᵀ` with the gauge projector `Π`
(invariant §3.2). `Π` is the existing `tangent_operations.orthogonal_gauge_projection` (frame-only, cheap,
sweep-independent). **Every one of these applies `Π` internally** (invariant §3.4) — the caller never
gauges anything:

```python
def apply_jacobian(p_data, ww, base, base_sweep) -> z              # = 𝒥(Π p)        (Riemannian forward)
def apply_gradient(r, ww, base, base_sweep) -> variation_data      # = Π 𝒥ᵀ r        (sum_over_probes=True)
def apply_gn_hessian(p_data, ww, base, base_sweep) -> variation_data  # = Π 𝒥ᵀ 𝒥 Π p   (the GN normal op)
def quadratic_model_value(                                         # = c + ⟨g, Πp⟩ + ½‖𝒥 Π p‖²
        p_data, ww, base, base_sweep, gradient_data, objective_value) -> scalar
    # projects p itself; one bare forward; shares Πp across both terms
```

i.e. `backend/fitting.py` is the `Π`-composition + the `JᵀJ`/model-value reductions; everything heavy
(the base sweep, the `𝒥`/`𝒥ᵀ` assembles) is upstream. A backend user calls these on raw tuples and gets
the correct gauge-projected Riemannian result with no `Π` bookkeeping of their own. **Corewise** omits the
`Π` calls (bare `𝒥ᵀ𝒥`) and sweeps the raw cores — the §6 wrapper (and it, too, owns that choice).

## 6. Taming the combinatorial surface (the three collapses)

The `3 × 2 × 2` grid is mostly illusory — the axes are not independent, and the reuse target is the
most-shared piece:

- **(a) tangent → corewise = substitution wrapper.** Corewise is the §6.3 `P,Q,O → G` substitution
  (`apply_corewise_transpose` is *literally* `apply_tangent_transpose` at base `(U,G,G,G)`). Corewise is a
  thin wrapper that swaps frame→cores and returns core grads — but, per §3.2, it wraps the **bare
  `probing.py` `_from_sweep` primitives** (`𝒥`/`𝒥ᵀ` on the substituted cores), **not** the `Π`-applying
  `backend/fitting.py` tangent functions. *This is the exact spot the gauge mix-up would sneak in*: if the
  corewise wrapper reused `apply_gn_hessian` (which applies `Π`), it would gauge-project against the
  non-orthonormal `G` cores and silently corrupt the corewise gradient. So the shared code is the bare
  probing assembly, and `Π` is composed only one level up, in the manifold fitting functions.
- **(b) plain → derivative = one skeleton, two op-sets.** The fitting *composition*
  (`precompute → jacobian → transpose → gn_hessian`) is structurally identical; plain calls `probing.*`,
  derivative calls the jet-ified `probe_derivatives.*` (matching signatures, the jet rides an extra order
  axis). Same skeleton, swap which primitives it composes.
- **(c) probe/apply/entries share the base sweep.** `compute_xis/mus/nus/etas` are **identical for probe
  and apply**; they fork only at the cheap *assemble*. **Entries** reuses the same `mus/nus/etas`
  machinery with only the `xi` *seed* swapped (`_entry_xis`/one-hot). So the expensive, reusable target is
  written **once per plain/jet**, shared across all three sampling kinds.

**Style (locked):** explicit per-type thin functions for the public surface (house style: readable,
grep-able, no runtime dispatch). Each is a 1–3 line composition of existing primitives; promote to a
shared private helper only if the threading repeats verbatim more than a couple of times.

## 7. Locked naming & placement decisions

- Dataclass: **`GaussNewtonModel`** — model-shaped (value + gradient + GN-Hessian-action + quadratic-model
  value); "Model" (not "Operator") because it does more than apply the Hessian, and the Hessian is
  specifically `H = JᵀJ` (Gauss-Newton). Raw `J`/`Jᵀ` live in the backend, not on the model.
- Thin-layer style: **explicit per-type one-liners** (not an ops-record factory).
- Solvers: a single flat **`optimizers.py`** when we get there (folder only if it grows); kept out of
  `fitting.py`'s dependency graph; may start in `examples/`.

## 8. The apply exemplar — the first slice (build this)

Scope: **apply, tangent, plain** end-to-end, validated reusing the sweep. Steps:

1. **`probing.py` split** — add `precompute_base_sweep`, `apply_jacobian_from_sweep`,
   `apply_transpose_from_sweep` (the **bare** `𝒥`/`𝒥ᵀ`); rewrite `apply_tangent`/`apply_tangent_transpose`
   to use them; add to `__all__`. Verify the existing `test_manifold`/probing tests still pass (unchanged).
2. **`backend/fitting.py`** — `apply_jacobian` (`𝒥(Πp)`), `apply_gradient` (`Π𝒥ᵀr`,
   `sum_over_probes=True`), `apply_gn_hessian` (`Π𝒥ᵀ𝒥Πp`), `quadratic_model_value` (`c + ⟨g,Πp⟩ +
   ½‖𝒥Πp‖²`). **All apply `Π` internally** (§3.4) — no caller-side gauging. `Π` =
   `tangent_operations.orthogonal_gauge_projection`.
3. **`fitting.py`** — the `GaussNewtonModel` dataclass (fields `base, ww, residual`; cached `_base_sweep`,
   `objective_value`, `gradient`; methods `gn_hessian`, `evaluate`; same-base guard) + factory
   `fitting.gauss_newton_apply(base, ww, residual)`. Model-shaped surface (no public `J`/`Jᵀ`).
4. **Tests** (`tests/test_fitting.py`) — see §9.
5. **Doctest** on `GaussNewtonModel` (reproducible-example convention): build a model, show
   `gn_hessian == Π𝒥ᵀ𝒥Π p`, `evaluate == ½‖r + 𝒥Πp‖²`, and the gauged-output check.

The exemplar **does** include `objective_value` (`c`) and `gradient` (`g`) — they come from the held
`residual` (the linear-LSQ form: residual given directly). **Deferred** is only the *nonlinear-fit
factory* that computes `r = F(base) − y` by evaluating `F` **through the frame** (so its sweep is the
reusable one, invariant §3.1) — needs confirming the dense-tensor apply reduces from the same base sweep.

## 9. Verification

- **Exact dense-truth model check (the HEADLINE oracle).** Because the sampling forward is **linear in the
  ambient tensor**, `φ(T) = ½‖A·T − y‖²` is *exactly quadratic*, its GN Hessian `AᵀA` *is* the true ambient
  Hessian (the dropped second-order term is zero), and the GN model is the **exact** restriction of `φ` to
  the affine tangent space `x + T_xM` — not a 2nd-order approximation. So for a random tangent `p`:

  ```
  evaluate(p)  ==  ½‖ residual + apply_dense( (Π p).to_dense() ) ‖²          (~1e-13)
  ```

  (equivalently `φ(x_dense + dense(Πp))`, since `apply_dense` is linear and `residual = A·x_dense − y`),
  where `apply_dense` samples the dense tensor with `ww` and `Πp = p.orthogonal_gauge_projection()`. This
  single check exercises `objective_value` (at `p=0`), `gradient` (the cross term), and the GN Hessian (the
  quadratic term) against true dense ground truth, at once.
  - **Use the gauge-projected `Πp`** on the oracle side: `orthogonal_gauge_projection` *changes* the dense
    image (it "represents a DIFFERENT tangent vector" — only `oblique_gauge_projection` preserves it). With
    a **gauged** `p` (`Πp = p`) this is just `evaluate(p) == ½‖residual + apply_dense(p.to_dense())‖²`,
    matching the recipe directly; with an **un-gauged** `p` it additionally makes the check **sensitive to
    a missing `Π`** (`dense(Πp) ≠ dense(p)`) — run both.
- **Reuse-equivalence (the core correctness claim):** `*_from_sweep(precompute(base, ww))` ==
  the existing all-in-one `apply_tangent`/`apply_tangent_transpose` (the uniform-layer-style "faster path
  == slow path" test), ~1e-14.
- **`gn_hessian` == `Π 𝒥ᵀ(𝒥(Π v))`** computed the long way (gauge-project, two separate bare probing
  calls, gauge-project), ~1e-14.
- **Gauge correctness:** `gradient` and `gn_hessian(p)` outputs are **gauged** (`.is_gauged()`); the
  operator is **symmetric on gauged variations** (`⟨q, Hp⟩ == ⟨p, Hq⟩`); `evaluate(p) == ½‖r + 𝒥(Πp)‖²`
  (the projected-adjoint oracle) and `== c + ⟨g,p⟩ + ½⟨p, Hp⟩` (consistency of the two quadratic-term
  forms).
- **Adjoint identity** `⟨z, 𝒥(Πv)⟩ = ⟨Π𝒥ᵀz, v⟩` (sum over probes), reusing the model.
- **Razor / self-containment (§3.4):** the `backend/fitting.py` functions give the *same* result on a raw
  (un-gauged) `p` as on `Π p` — proving they project internally, so a backend user never has to gauge.
- **Same-base guard** raises on a tangent from a different `T3Basis` (structural error doctest).
- **jit** (`test_dispatch`): the model's `gn_hessian`/`evaluate` under `jax.jit` (cached_property + frozen
  dataclass pytree — confirm the sweep folds in; mirror the `T3Tangent` basis-as-aux pattern if needed).
- **Against the example:** drop the model into `examples/fit_hilbert_tensor_newton_cg.py`'s inner CG
  (replace the per-apply recompute) and confirm identical iterates + a wall-clock drop.

## 10. After the exemplar (fan-out order)

1. **probe + entries (plain, tangent)** — same skeleton; probe shares the apply sweep (different
   assemble), entries swaps the `xi` seed.
2. **derivative (probe/apply/entries, tangent)** — swap the op-set to `probe_derivatives.*` (jets); the
   `base_sweep` becomes `*_jets`.
3. **corewise (all)** — the substitution wrapper `CorewiseGaussNewtonModel` (bare `𝒥ᵀ𝒥`, no `Π`). Two
   no-Π oracles: (i) the **exact dense-truth** analog of §9 — the corewise linearization is the sum of
   single-core swaps `dense_lin = Σ_core T(cores, core→δcore)` (each a T3 → dense, summed; this *is*
   `dense(J_cw·δcores)`), so `evaluate(δcores) == ½‖residual + apply_dense(dense_lin)‖²` with **no** gauge
   projection (projecting would break the match); (ii) `gradient`/`gn_hessian` vs `jax.grad` /
   `jax.linear_transpose` of `cores ↦ ½‖F−y‖²`. The manifold dense-truth check (§9, *with* `Π`) and these
   (*without* `Π`) together pin the matched pair (§3.2).
4. **`.objective` + the Riemannian-fit-of-a-tensor mode** (frame-evaluated `F`), then `optimizers.py` +
   refit the Hilbert examples on top.

**Open/confirm during build:** does the dense-tensor apply (`TuckerTensorTrain.apply` backend) reduce
from the same `(xis, mus)` so `.objective` reuses the sweep? (flagged in §8). jit treatment of the
`cached_property` sweep on the frozen dataclass (§9).

**Known limitation (deferred, surfaced in Slice 1).** `apply_tangent_transpose` / `entries_tangent_transpose`
(and so `apply_transpose_from_sweep`) accept a residual of shape `W + C` only — they are **not K-aware**: a
tangent-stack `K` in the residual (`W + K + C`, the output space of a K-stacked forward `apply_tangent`) is
unsupported (the `c[..., None] * mu` scatter in `_apply_transpose_assemble` misaligns `K`; `probe_tangent_transpose`
*is* K-aware, apply/entries are not). It does **not** affect the fitting layer (which uses `K=()`). Revisit
if a K-stacked apply/entries transpose is ever needed (mirror the probe transpose's 3-block contractions).
