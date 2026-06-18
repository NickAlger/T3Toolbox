# Symmetric probing derivatives — handoff

## What it is

Symmetric directional derivatives of **probing** and of **tangent-vector probing**:
`y_i^(k) = d^k/ds^k y_i(X + s P)|_0` for `k = 0..K`, the derivatives obtained by perturbing every
probe vector in the *same* direction `P`, repeated `k` times. Computed by one probing-style sweep
carrying an extra Taylor-jet axis. The motivating use case: fit a Tucker tensor train (or a tangent
vector) to measured derivative data.

## Status

Lives on branch **`probe-derivatives`** (7 commits), ready to merge to `main`. Full suite green (231).

- Backend: `t3toolbox/backend/probe_derivatives.py` + the `t`/`trs` contractions in
  `t3toolbox/backend/contractions.py`.
- Math note (full derivation): `docs/symmetric_probe_derivatives.tex` (compiles with pdflatex).
- Tests: `tests/test_probe_derivatives.py` + jit cases in `tests/test_dispatch.py`.

## What's implemented (forward + transpose, Euclidean + Riemannian, fully stacked)

| | function | stacks |
|---|---|---|
| Euclidean forward | `probe_derivatives_t3` | order `t` + sample `S` + base `C` |
| Riemannian forward | `probe_tangent_derivatives` | `t + S + C` (single tangent, no `K`) |
| Riemannian transpose | `probe_tangent_derivatives_transpose` | `t + S + C`, `sum_over_probes` |
| dense oracle | `probe_derivatives_dense` | unstacked (testing) |

Edge-variable jets: forward `compute_{mu,nu,eta}_jets` / `assemble_z_jets` (plain),
`compute_{sigma,tau,deta}_jets` / `assemble_tangent_z_jets` (variation sweep); transpose adjoints
`compute_{deta,tau,sigma,dxi}_tilde_jets` and gradient assembly `assemble_{tucker,tt}_variation_jets`.
Helpers `binomial_combine_tensor`, `build_input_jets`.

## The math (one paragraph; full derivation in the `.tex`)

`X + sP` is **affine** in `s`, so each input vector carries a trivial Taylor jet `(x, p, 0, …)`, and
every product of jets is a **binomial convolution** driven by `trs[t,r,s] = C(t,r)·[r+s=t]`. The
pushthrough, combine, and lift become single einsums against `trs` (the forward `t`-contractions). The
**transpose** is the jet-ified **adjoint-state Lagrangian** (t4s.pdf Thm 7): replace every contraction
by its `trs` version, and stationarity hands back the adjoint sweeps — the **same `trs` wired as its
transpose** (adjoint of a binomial convolution = binomial correlation: the multiplier's order summed,
the swept order freed) — plus the order-less gradient assembly, whose binomial-tensor **arity equals
the assembled core's internal-edge count** (Tucker 1-edge → plain order-sum `t`; TT 3-edge → `trs`;
an HT 4-tensor would need `trsq`).

## Conventions

- Order axis `t` **outermost**. Sample stack `S` (= probing's `W`, paired `(X,P)` samples) and base
  stack `C` (= `stack_shape` on the cores), **base-inner**: `S` outer, `C` inner. `sum_over_probes`
  sums `S` (the `J^T r` back-projection for fitting) or keeps it (`S` rides into the variation stack).
- Naming distinguishes the jet-ified functions from probing.py's unjetified ones: forward `*_jets`,
  transpose adjoints `*_tilde_jets`, gradient assembly `*_variation_jets`.
- Contractions named by index structure, `trs` operand carries the binomial tensor; the input-jet
  (`xi`) order leg is sliced to `{0,1}` since `X+sP` is affine.

## Verification (how to trust / re-check it)

- **Forward**: dense subset-expansion oracle — each action is multilinear of degree `d-1`, so
  `y_i^(k) = k!·Σ_{|S|=k} (contract with p_j for j∈S, x_j else)` is exact. (`probe_derivatives_dense`;
  for a tangent, the densified tangent is a dense tensor, so the same oracle applies.)
- **Transpose**: the dense **adjoint identity** `⟨r, J v⟩ = ⟨J^T r, v⟩` (`sum_over_probes=True`) and
  the exact **`jax.linear_transpose`** of the (linear-in-the-variation) forward. Both to ~1e-16 across
  `S × C × order`, both `sum_over_probes` settings.
- One gotcha for ad-hoc oracle checks: `jax.linear_transpose` returns the variation gradient, which is
  **order-less and `S`-summed** (shape `C + core`) — compare it against `sum_over_probes=True`, not
  `False` (which keeps `S`).

## Future work (Nick's roadmap)

1. **Jet-ified `apply` / `entries`** — the all-modes special cases of probing (probing leaves one mode
   free; `apply`/`entries` contract *every* mode). Reuses almost all of the machinery here; mimic
   probing.py's `apply_tangent` / `entries_tangent` (+ their transposes). Should be a thin layer over
   the jet sweeps (no central `nu`/`eta`, single terminal contraction).
2. **Frontend hookup** — methods on `TuckerTensorTrain` / `T3Tangent` (probe-derivatives forward +
   tangent transpose), wrapping the backend the way the existing `probe`/`apply`/`entries` methods do.
3. **Fitting example** — fit a T3 to a Hilbert tensor from its probe-derivative (or apply-derivative)
   data, mimicking `examples/fit_hilbert_tensor_newton_cg.py` (which fits from ordinary applies).
4. **Merge `probe-derivatives` → `main`.**

## Deferred (noted in the `.tex` §8 "Remaining")

- **Tangent stack `K`** (many tangent vectors at one base) — touches both forward and transpose; would
  reuse probing.py's three-block `W/K/C` contraction pattern.
- **Euclidean corewise / ambient transpose** — gradient of the plain-T3 probe derivatives w.r.t. the
  cores as independent variables, and the base-free adjoint (the derivative-map analogue of probing's
  three transpose flavours).
- **Project-once gather** — avoid recomputing the `X`-projection per repeated direction when one `X`
  is swept with many `P` (only matters in the large-ambient / small-rank regime; expose as an option).
