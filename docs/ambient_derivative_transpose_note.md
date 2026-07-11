# The ambient derivative transpose — analysis & deferral note

**Status: DEFERRED (not currently needed).** The symmetric-derivative work (branch `probe-derivatives`)
mirrors the probing surface with derivative versions for the **forward**, **tangent** (Riemannian), and
**corewise** (non-manifold core-gradient) flavors of `probe`/`apply`/`entries`. The **ambient** flavor —
the frame-free adjoint on the full tensor space, returning canonical (CP) factors — was the one cell of
the grid left unbuilt. This note records the math and the design analysis so it can be picked back up
if a use case arises. It is **not** scheduled work.

Companion docs: `docs/transposes.md` (the ambient/corewise/tangent taxonomy),
`docs/symmetric_probe_derivatives.tex` (the forward + tangent-transpose math),
`dev/archive/derivatives_mirror_plan.md` (the overall plan/handoff). Backend home if built:
`backend/sampling_derivatives.py`; the non-derivative versions to mirror are
`probing.t3_probe_ambient_transpose`, `apply.t3_apply_ambient_transpose`,
`entries.t3_entries_ambient_transpose`.

## 1. Recap: the non-derivative ambient transposes

The **ambient** transpose is the literal adjoint of a sampling op viewed as a **linear map on the full
tensor space** `T ↦ samples`. Its output is the back-projection tensor, returned as **CP factors** (the
natural type — a multilinear adjoint emits one vector per mode), realized via `from_canonical`:

- **apply** (`T ↦ ⟨T, w₀⊗…⊗w_{d-1}⟩`, a scalar): adjoint of residual `c` is `c·(w₀⊗…⊗w_{d-1})` —
  **rank-1** CP.
- **probe** (`T ↦ {⟨T, w₀⊗…⊗·⊗…⊗w_{d-1}⟩}ᵢ`, `d` vectors): adjoint of `{ztildeᵢ}` is
  `Σᵢ w₀⊗…⊗ztildeᵢ⊗…⊗w_{d-1}` (residual in slot `i`, probe vectors elsewhere) — **rank-`d`** CP.
- **entries** = apply with one-hot vectors `e_{index}`: a rank-1 one-hot scatter. Needs an explicit
  `shape` (the one-hots don't carry the ambient dims `Nᵢ`).

`sum_over_probes=True` folds the probe stack `W` into the CP rank (one rank-`d|W|` / `|W|` CP); `False`
keeps `W` as a stacking axis.

## 2. The derivative ambient transpose: the math

The derivative forward is the **jet** (in `s`) of the plain forward along `X + sP`. For apply, the
plain forward is `apply(T) = ⟨T, ⊗ⱼwⱼ⟩`, so

```
applyᵗ(T) = dᵗ/dsᵗ ⟨T, ⊗ⱼ(wⱼ + s pⱼ)⟩|₀ = ⟨T, Mₜ⟩,   Mₜ := t!·[sᵗ] ⊗ⱼ(wⱼ + s pⱼ).
```

`applyᵗ` is linear in `T`, with "matrix" `Mₜ`. So the ambient adjoint of the residual jet `{c⁽ᵗ⁾}` is

```
Tᵀ = Σₜ c⁽ᵗ⁾ Mₜ = Σₜ c⁽ᵗ⁾ · t! · [sᵗ] ⊗ⱼ(wⱼ + s pⱼ)          (apply)
```

and, the same way (residual in the free slot `i`, the off-modes `j≠i` perturbed),

```
Tᵀ = Σᵢ Σₜ rᵢ⁽ᵗ⁾ · t! · [sᵗ] ⊗_{j≠i}(wⱼ + s pⱼ)   placed in slot i   (probe)
```

**Reading.** `Tᵀ` is exactly the **jet of the non-derivative ambient back-projection, evaluated at the
perturbed probe vectors `wⱼ + s pⱼ`**, then contracted against the residual jet on the order axis. The
`[sᵗ]` coefficient of the product `⊗ⱼ(wⱼ+s pⱼ)` is a **binomial/multinomial jet convolution** of the `d`
per-mode 2-jets `[wⱼ, pⱼ]` (value at order 0, direction at order 1) — so the binomial tensor `trs`
appears in the outer products, as one would expect by analogy with the forward.

**entries.** Apply with one-hot in the frame slot and a general direction: vectors `(e_{index_j} + s pⱼ)`.
So `Tᵀ = Σₜ c⁽ᵗ⁾ t! [sᵗ] ⊗ⱼ(e_{index_j} + s pⱼ)` — the `w`-slots become one-hot scatters at `index`.
Unlike the non-derivative `entries_ambient_transpose`, **no explicit `shape` is needed**: `pp[j]` carries
`Nⱼ`.

## 3. The catch: the back-projection is inherently high-rank

The crucial fact. The coefficient

```
[sᵗ] ⊗ⱼ(wⱼ + s pⱼ) = Σ_{|S|=t} ⊗ⱼ (pⱼ if j∈S else wⱼ)
```

is an **elementary-symmetric** combination — a genuine sum of `C(d-1,t)` (probe) / `C(d,t)` (apply)
rank-1 terms, not a single one. Summing over `t`:

| op | CP rank of `Tᵀ` |
|---|---|
| **apply** | `2^d` |
| **probe** | `d · 2^{d-1}` |
| **entries** | `2^d` (one-hot factors) |

(`× |W|` more if `sum_over_probes=True` folds `W` into the rank.) This is a property of **the tensor
itself**, not of how we write it: unlike the clean **rank-1 / rank-`d`** non-derivative ambient
back-projections, the derivative one is exponentially higher rank. `d` is usually small (`2^{d-1} ≤ 32`
at `d=4`), so it is finite and computable — but the materialized object (and any `from_canonical` T3
built from it, whose ranks track the CP rank) is intrinsically large.

## 4. Two representations (neither removes the rank)

- **(A) Subset-expanded standard CP.** Enumerate `S` (which modes contribute `p` vs `w`); the residual
  enters at order `|S|` in the free slot, weighted by `|S|!`. Yields **plain CP factors that
  `from_canonical` already consumes** — *zero new machinery*, a ~10-line nested loop over `i` and the
  `2^{d-1}` subsets. Rank `d·2^{d-1}` is intrinsic anyway. **Recommended if built.**
- **(B) Jet-CP** (the "trs in the outer products" idea). Keep `d` per-mode 2-jet factors `[wⱼ,pⱼ]` + the
  residual jet, and convolve the orders via a `d`-fold multinomial `trs` (a `trsq…q` tensor) **only at
  reconstruction**. Compact *representation* (rank `d`/`1`), but needs a **new jet-CP type + a `d`-fold
  multinomial reconstruction**, and it **still materializes to the same `d·2^{d-1}` rank**. So (B)
  defers the blow-up without removing it.

## 5. Why deferred

- **Least-used flavor.** Derivative-fitting uses the **tangent** transpose (Riemannian) or **corewise**
  (Adam/L-BFGS), both done. The ambient one exists to form a *dense Euclidean gradient as a CP sum* —
  rarely the thing you want, and here that sum is exponential-rank rather than the clean rank-`d` the
  non-derivative version gives.
- **Output inherently less clean** than its non-derivative sibling — the one place the
  strict-generalization stays ugly regardless of representation.
- No concrete use case; added (if ever) for grid completeness only.

## 6. Implementation sketch if picked up (option A)

Mirror `probing.t3_probe_ambient_transpose` (which builds the rank-`d` CP with the residual on a diagonal
slot), but enumerate subsets:

- `apply_ambient_derivatives_transpose(c, ww, pp, order, sum_over_probes=False)` →
  CP rank `2^d`: for each subset `S ⊆ {0..d-1}`, a term with factor-`k` = `pp[k]` if `k∈S` else `ww[k]`,
  scaled (fold into one factor) by `c[|S|]·|S|!`. Only subsets with `|S| ≤ order` contribute (residual
  has orders `0..order`); above the tensor order the derivative vanishes anyway.
- `probe_ambient_derivatives_transpose(ztildes, ww, pp, order, sum_over_probes=False)` →
  CP rank `d·2^{d-1}`: for each mode `i` and each `S ⊆ {j≠i}`, factor-`i` = `ztildes[i][|S|]`, factor-`k`
  (`k≠i`) = `pp[k]` if `k∈S` else `ww[k]`, scaled by `|S|!`.
- `entries_ambient_derivatives_transpose(c, index, pp, order, sum_over_probes=False)` → as apply with
  `ww[k] = e_{index_k}` (one-hot, length `Nₖ` from `pp[k]`); no explicit `shape` arg.
- Frontend: `TuckerTensorTrain.{probe,apply,entries}_ambient_derivatives_transpose` (static), returning
  CP factors for `from_canonical`, paralleling the non-derivative `*_ambient_transpose` statics.
- Verify: the dense adjoint identity `⟨Tᵀ, X⟩_F == Σ ⟨residual, sample_derivatives(X)⟩` (`from_canonical`
  the factors, compare to the forward derivative of a random `X`). **Log the CP rank** (no silent caps).
- `sum_over_probes`: fold `W` into the rank exactly as the non-derivative versions do.
