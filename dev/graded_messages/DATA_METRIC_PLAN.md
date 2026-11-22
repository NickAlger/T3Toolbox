# The data metric: a sub-arc between P16 and P17 (FIRST DRAFT, Claude, 2026-11-21)

Status: draft for critique by a writer/reviewer pair; the maintainer reads the refined plan before any build.
Decisions it implements: `TARGET_DECISIONS.md`, "The data metric: decisions 1 and 2 of the sampling audit".
Evidence: `SAMPLING_AUDIT.md`. Out of scope: decisions 3 and 4 (the corewise representation; the Hessians' names and
whether `jet_to_dual_jet` stays public), a full-operator `W`, nonlinear losses.

## 1. The goal, typed

Parameter space `X`, data space with `J: X -> Y'` (the sampled outputs are measurements), the metric-free transpose
`J^T: Y -> X'`, the data Riesz map `W: Y' -> Y`. The objective-free maps (the forwards, tangents, transposes and
second-order transposes, on `TuckerTensorTrain`, `T3Tangent`, their uniform twins and the backend) take and return
the types above and **contain no data metric**. The data metric is the objective's and lives in the fitting layer,
as one public function, used by the oracle and by anyone composing a fit by hand: `g = J^T (W r)`.

Two facts fix the representation (audit, verified):
- the pairing of `Y` with `Y'` is **slot by slot on Taylor coefficients**, `<c, a> = sum_k c[k] a[k]`; the jet flip
  in `jet_to_dual_jet` is the engine's internal representation of that pairing (a corner coefficient), not a metric;
- **the library's data coordinates are Taylor coefficients.** Derivative data enter only through
  `prepare_derivative_data`, which converts the data and the weight together. So no element of `Y` is ever held in
  derivative coordinates, and the audit's trap D4 (a `Y` element converts the opposite way to data) cannot arise.

## 2. What changes

**A. The weight becomes elementwise (decision 2).** A weight is an array (probe: one per mode, as `y` is) that
broadcasts to the shape of `y`: probe `F + W + (N_i,)` per mode, apply `F + W`, entries `W` (no `C`: one `y` per
stacked model since P16). It is `omega`, entering squared, as today. Today's block forms are broadcast cases: probe
`(d,) + F` becomes one `F + (1,)*len(W) + (1,)` array per mode; apply `F`; entries `()`. To settle: whether the block
forms stay accepted as input (a reading rule, unambiguous by rank?) or are converted once at the boundary.
Consequences to carry through:
- `jets.check_weight` and the engine's `weight` reading; `jet_to_dual_jet` multiplies elementwise;
- `LeastSquaresBatch.take` slices a weight's sample axes with the draw (when they are not of size 1), so a drawn
  batch keeps each sample's weight; the minibatch unbiasedness test with per-sample weights;
- `pack_batch` packs a per-mode probe weight with **zero** in the mode padding (the engine masks `y`; the weight
  must not reintroduce padding);
- `f_resolution`'s `|omega y|` with elementwise `omega`;
- the display: records stay per `(mode, order)` block (O-42), the "unwt" figure unchanged;
- `prepare_derivative_data`: the `t!` conversion of the weight, elementwise;
- the old-form translation (P16, deleted in P17): the old block weights map to the broadcast form.

**B. The public `W` (decision 1).** One function in the fitting layer (backend `fitting.py`, re-exported by
`t3toolbox.fitting`), name to be chosen (draft: `weigh(kind, order, r, weight) -> W r`): `r` of the type of `z` in
Taylor coordinates, returns `omega^2 * r` elementwise, slot order unchanged (an element of `Y` as the transposes take
it). The engine's seed becomes `flip(W r)`: `jet_to_dual_jet` is split into the public `W` and the private flip, one
implementation of the weighting.

**C. The transposes typed as `Y` (decision 1).** Every transpose and second-order transpose (backend: 18 functions;
frontend: the corewise, tangent and ambient transpose methods and the corewise Hessians, ragged and uniform): the
argument `residual` is renamed (draft: `c`) and documented as "an element of `Y`, paired with `z` slot by slot; for
the gradient of a weighted least-squares fit pass `W r`". No `weight` parameter. The frontend docstrings that call
the output "the Gauss-Newton `J^T r`" (`manifold.py:768,916`, `uniform_manifold.py:643`,
`tucker_tensor_train.py:3837,4003`) say `J^T c` and show `J^T (W r)`. The right/wrong adjoint doctests (finding A of
P13) stay, with the new name.

## 3. Slices (each design-first: a design note, the reviewer's critique, Claude's rulings, then the build)

- **M1. The elementwise weight in the engine** (A, engine side): `check_weight`, the engine's reading, the public `W`
  and the split of `jet_to_dual_jet` (B). Tests: `W` against `omega^2 r` elementwise for every kind and broadcast
  case; the engine's weighted `g`, `JtJp`, `Kp` against a dense oracle with a per-sample, per-entry weight;
  garbage-padded uniform with a padded weight.
- **M2. The typed transposes** (C): the rename, the docstrings, the frontend methods; the adjoint identity
  `<c, J x_dot> = <J^T c, x_dot>` typed; `J^T (W r)` equals the oracle's `g` (one implementation).
- **M3. The fitting layer** (A, fitting side): `LeastSquaresBatch` and `check_batch` with elementwise weights; `take`
  slicing weights; `pack_batch`; `f_resolution`; `prepare_derivative_data`; the display; the translation's mapping.
  Tests: per-sample-weighted fits against a dense oracle; unbiasedness by exact subset enumeration with per-sample
  weights; an entries fit with per-sample weights (impossible today).
- **Docs:** P19 (already the documentation slice) gains the typing and the data metric; this arc writes docstrings
  only.

## 4. Questions for the maintainer (to be sharpened by the critique)

1. The name of the public `W` (`weigh`, `apply_weight`, `data_riesz`, ...) and of the transposes' argument (`c`,
   `dual`, `y_dual`, ...).
2. Block weight forms: still accepted (read by rank) or converted at the boundary only?
3. `omega` (squared inside) as today, or `omega^2` given directly?
