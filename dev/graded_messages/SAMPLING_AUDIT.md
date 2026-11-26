# The sampling functions: an audit of their typing and placement (the pause after P16)

2026-11-21. A read-only audit for the maintainer's discussion at the pause after P16: how the sampling functions are
typed, and where they live. **Findings and options, not decisions.** Every claim is checked in the code at
`2802b490` (`graded-messages`); `file:line` refers to that tree. One light script was run (section 3.0).

The scheme used throughout (the maintainer's):

- `X` the parameter space (the cores, or a frame's variations); `R_X: X -> X'` its Riesz map, the geometry's metric
  (Euclidean on raw cores for `COREWISE`; Hilbert-Schmidt plus the gauge projection `Pi` for `MANIFOLD`).
- `Y'` the measurement space, where `z`, `y`, `r = z - y` and `J x_dot` live; `J: X -> Y'`. `Y` its dual;
  `J^T: Y -> X'` the metric-free (Banach) transpose; `W: Y' -> Y` the data Riesz map (a weight; later a loss
  Hessian). Then `f = 1/2 <r, W r>`, `g = J^T W r`, `H_gn = J^T W J: X -> X'`.
- The input space `V_i = R^{N_i}`: the vectors contracted into the tensor (`ww`, the input jets; the entries index).

---

## 1. Summary (one page)

**The flip is not a metric (Claude's correction holds).** `jet_to_dual_jet` = `W` followed by a change of
representation (every jet axis reversed), so that the engine can read the pairing `sum_k c[k] a[k]` as the corner
coefficient of a jet product. Checked: with the plain slot-by-slot pairing the public transposes satisfy the adjoint
identity to 1e-16 at jet shapes `(2,)` and `(1, 1)`, and the engine with a weight on the same route gives exactly
`J^T (omega^2 c)` (section 3.0). The only metric is the weight.

**But the maintainer's principle still flags the public transposes and Hessians, for a reason the correction does not
remove.** Their argument is named `residual` and documented "of the type of `z`" (`Y'`), and the frontend docstrings
call their output "the usual Gauss-Newton `J^T r`" or "the summed gradient `J^T r`". That is `J^T W r` with `W`
silently `I`. Read with the argument typed `Y`, the functions are correct and metric-free. Read as their names and
docstrings say, they are a `Y' -> X'` map with a hidden identity Riesz map. **The numbers are the same either way.
What has to change is the declared type of one argument, in 18 backend functions and 15 frontend methods.**

**The engine already supports a weighted transpose on this route.** The public functions are the only place it is
fixed to the identity (`probing.py:1065`, `:1574`; `sampling_variables.py:119-123`). Adding `weight=` is therefore
cheap. Whether it is right is the question.

**Six more findings matter to the design:**

1. Two public representations of `Y`. The transposes take `Y` plain. `jet_to_dual_jet` (exported by
   `t3toolbox.fitting`) returns `Y` turned, and nothing in the frontend consumes that form. Passing a dual jet to a
   transpose flips it twice and gives a wrong answer, silently (D3).
2. The variance is not encoded. `derivatives_to_taylor` converts `Y'` objects (data). A `Y` object in derivative
   coordinates converts the other way, by `taylor_to_derivatives`, and no function or document says so. A user
   computing a weighted gradient by hand gets it wrong with either converter unless they square the factorials
   themselves (D4).
3. The weight is block-scalar only: `(d,)+F` for probe, `F` for apply, `()` for entries (`jets.py:1290`). There is no
   per-sample weight, no per-element weight in `C`, and no metric within a mode. **For entries the only weight is one
   scalar for the whole objective.** "Almost all fits will be weighted" cannot be met today for per-measurement
   noise (D5).
4. For the probe kind the data space is the dual of the input space: `z_i ∈ V_i'`. `rescale_coordinates` already
   transforms probe data as covectors (it divides by the scales), and the loss then changes because `V_i'` carries
   the identity metric. So an input-space metric would show up as a data metric, and none exists. **No input-space
   weighting exists anywhere**, as Claude believed. The only way to get one is to scale the input vectors before
   fitting (D6).
5. On the parameter side, `T3Tangent` holds both `X` and `X'`. The tangent transposes return a cotangent typed as a
   `T3Tangent`. `MANIFOLD.project` is the explicit `R_X^{-1}` ("raw cotangent -> Riemannian gradient",
   `manifold.py:1177`), and `COREWISE.project` is the identity. Nothing is silent here except the type name (P1).
6. **"Act on a point or on a tangent" does not separate the corewise operations.** On the flat corewise manifold the
   frame is the point (P16). The candidate line "by object" therefore says nothing about where
   `*_corewise_{tangent,transpose,hessian}` belong (N2).

**On the corewise "Hessian".** It is `D(J(x)^T c)[x_dot]`, the second derivative of `phi_c = <c, z(x)>`, and it does
not depend on the objective (checked against a difference quotient of the transpose: 5e-9 at `h = 1e-4`). The name is
consistent with the library's existing `t3_inner_product_corewise_hessian` (Hessian of `theta -> <z, Phi(theta)>`,
`t3_linalg.py:857`). A rename should therefore cover both families, or neither (N3).

**Decide first (section 5):**

- the declared type of the transposes' data argument: `Y` (a covector, no weight) or `Y'` (a residual, a weight
  required);
- what a data metric is meant to cover: block-scalar `omega`, or per-sample, within-mode, or an operator for
  nonlinear losses.

The placement of the functions follows from those two answers.

---

## 2. Inventory

Columns: **Acts on** (P = a point, T = a tangent at a manifold frame, C = the corewise flat manifold, whose frame is
the point; A = the ambient space, no frame); **Type**; **Metric** (`W`/`R_X`/input: applied fully, partly, or as a
silent identity); **wt?** (whether a weight can be passed); **obj?** (whether it depends on the objective); **placement
under the three candidate lines** (L1 by object, L2 by objective dependence, L3 methods versus functions).
Each uniform twin is grouped with its ragged counterpart; "TT" means `TuckerTensorTrain` and "UTT"
`UniformTuckerTensorTrain`.

| # | Family: backend / frontend (ragged; uniform) | Acts on | Type | Metric | wt? | obj? | L1 / L2 / L3 |
|---|---|---|---|---|---|---|---|
| A | **Forwards at a point.** `t3_{probe,apply,entries}` (`probing:87`, `apply:54`, `entries:67`); `ut3_*` (`ut3_sampling:74-106`); `ut3_full_sum`. TT/UTT `.probe/.apply/.entries`, UTT `.sum` | P | `X -> Y'`, nonlinear (`z(x)`) | none | n/a | no | stay / stay / methods |
| A' | **Dense ground truth.** `dense_probe` (`probing:1677`), `dense_apply` (`apply:469`), `dense_entries` (`entries:418`); not in the frontend (O-27/28) | A (a dense `T`) | `E -> Y'`, linear | none | n/a | no | functions in every option |
| B | **Tangent forwards.** `tv_*` (`probing:557`, `apply:141`, `entries:120`); `utv_*` (`utv_sampling:87-125`). `T3Tangent`/`UT3Tangent` `.probe/.apply/.entries` | T | `J: X -> Y'` (bare `𝒥`; `LocalModel.jacobian` composes `Pi`) | none | n/a | no | stay / stay / methods |
| C | **Tangent transposes.** `tv_*_transpose` (`probing:1030`, `apply:169`, `entries:147`); `utv_*_transpose` (`utv_sampling:467-512`). Static `T3Tangent`/`UT3Tangent.*_transpose` (`manifold:752,906,947`; `uniform_manifold:632-694`) | T | **declared** `Y'` ("residual, of the type of z") `-> X'` returned **as a `T3Tangent`**; computed as `Y -> X'` | **`W` silently `I`** (held `Jp`, no weight, `probing:1064-1066`); `R_X` not applied (bare, documented) | no (the engine can: 3.0) | no | stay / stay (re-typed) / static constructors |
| D | **Corewise forwards** (new, P12/13). `t3_*_corewise_tangent` (`probing:1138`, `apply:311`, `entries:265`); `ut3_*` (`ut3_sampling:143,177,209`). TT/UTT `.*_corewise_tangent` (`tucker_tensor_train:4063-4115`) | C | `J: X -> Y'`, `X` = raw cores | none | n/a | no | **undecided** (N2) / stay / methods |
| E | **Corewise transposes.** `t3_*_corewise_transpose` (`probing:1178`, `apply:351`, `entries:304`); `ut3_*` (`ut3_sampling:158,191,225`). TT/UTT `.*_corewise_transpose` (`tucker_tensor_train:3818,3977,4034`; UTT `:486-542`) | C | as C, returning raw core pairs (`X'`) | `W` silently `I`; docs say "the summed gradient `J^T r`" (`tucker_tensor_train:3837,4003`; UTT `:498`) | no | no | undecided / stay (re-typed) / methods |
| F | **Corewise "Hessians".** `t3_*_corewise_hessian` (`probing:1537`, `apply:422`, `entries:372`); `ut3_*` (`ut3_sampling:479,550,584`). TT/UTT `.*_corewise_hessian` (`tucker_tensor_train:4126-4188`; UTT `:571-633`) | C | `(c ∈ Y, x_dot ∈ X) -> X'`: `D(J^T c)[x_dot]` | `W` silently `I` (held `r`, no weight, `probing:1574`) | no | **no** (any `c`) | undecided / stay / maintainer's lean: functions; Claude: methods, renamed |
| G | **Ambient transposes.** `t3_*_ambient_transpose` (`probing:1071`, `apply:82`, `entries:94`); static TT methods (`tucker_tensor_train:3759,3869,3929`); no uniform twin | A | `Y -> E'` (CP factors), identified with `E` through HS | `W` silently `I`; `E' = E` by HS (the maintainer's premise); no jets (O-43) | no | no | functions (frame-free) / stay / static |
| G' | **The same pattern on a T3 operand.** `t3_inner_product_corewise_{transpose,hessian}` (`t3_linalg:857`), `ut3_*` (`ut3_linalg:426`). Read by the geometry for `K_{grad f}` (`geometry:540`) | C | `(z ∈ E ≅ E', x_dot) -> X'` | HS | n/a | no | the naming precedent for F |
| H | **The engine.** `sampling_variables`, `HeldVariables`, `computed_names`, `frame_form`, `frame_shape` (`sampling_variables`); private steps (`sampling_steps`); the reference module (`sampling_definitions`) | P, T, C | every variable. `r, z, Jp ∈ Y'`; `w_tilde = flip(W r) ∈ Y` (turned); `g, JtJp, Kp ∈ X'`; `f, Jp_norm_sq` scalars | **`W` complete and explicit** (`f_step`, `w_tilde_step`, `Jp_norm_sq_step`, `sampling_steps:957-1060`); `weight=None` means `I` | yes | both | backend infrastructure in every option |
| I | **Dual jets.** `jet_to_dual_jet`, `dual_jet_to_jet` (`jets:298,423`); frontend `fitting.jet_to_dual_jet` (re-exported, `fitting:83`), `fitting.dual_jet_to_jet` (safe-mode zero check, `fitting:190`) | data | `W` then turn: `Y' -> Y_turned`; inverse `Y_turned -> Y'` | `W` explicit; argument required (`None` = `I`, stated) | yes | yes (`W`) | — / fitting / functions |
| I' | **Derivative data and the weight.** `derivatives_to_taylor`, `taylor_to_derivatives`, `derivative_weight_to_taylor_weight`, `prepare_derivative_data` (`jets:516-798`); frontend reach in `fitting` | data | change of basis on `Y'` (data), and on `omega` | converts `Y'` and `omega`; **no converter for a `Y` object** (D4) | yes | `prepare_*`: yes | — / split / functions |
| I'' | **Input jets and the checkers.** `affine_jet`, `index_to_one_hot`; `check_sample`, `check_data`, `check_weight` (`jets:814-1298`) | input / data | build `V_i`-valued jets; checkers | none (`check_weight`: shape of `omega`) | n/a | `check_weight`: yes | — / sampling (`check_weight` to fitting) / functions |
| J | **Data preparation.** `data_block_norms`, `equalizing_weight`, `rescale_direction`, `rescale_coordinates` (+ frontend checked twin), `t3_/ut3_unscale_coordinates`, `check_coordinate_scales`; TT/UTT `.unscale_coordinates` | input + data | changes of coordinates on `V_i` and `Y'`; a weight built from `Y'` norms | `data_block_norms` uses the identity norm on `Y'` (to build a `W`); `rescale_direction` is a weight change in disguise (D8) | — | yes (they shape the loss) | — / fitting / functions |
| K | **The oracle.** `LeastSquaresOracle.query` / `check_batch` / `take` / `n_measurements`, `LeastSquaresBatch` (`backend/fitting:758-1021`); the `Oracle` protocol (`optimizers:181`); `pack_sample`/`pack_data`/`pack_batch`/`uniform_minimal` | frame or point | `f, g = J^T W r, JtJp = J^T W J p, Jp_norm_sq = <Jp, W Jp>, Kp = D(J^T W r)[.]` (`X'`, before projection); `r, Jp ∈ Y'` | **`W = scale * diag(omega^2)` complete** (O-21; `fitting:918,935-941`); `r_block_sumsq`, `y_block_sumsq` identity, on purpose | yes, in the batch | yes | — / fitting / classes |
| L | **The local model and the optimizers.** Backend `LocalModel` (`optimizers:287`: `gradient`, `hvp`, `jacobian`, `gn_quadratic`, `curvature_hvp`, `residual`, `objective`), `local_model`, `point_objective`, `flat_draw`, the four optimizers; frontend `fitting.LocalModel` (`fitting:546-590`), `local_model`, the frontend optimizers; P17: `point_value_and_gradient` | frame | `X -> X` after `R_X^{-1}` (`Pi` composed: `optimizers:363-401`) | `W` from the oracle; `R_X` explicit through `geom.project` | yes (batch) | yes | — / fitting / functions + classes |
| M | **The display.** `relative_errors`, `format_newton_iter`, `make_newton_display` (`optimizer_display:56,413,555`) | data | per-block `‖r_b‖/‖y_b‖`; `obj_unweighted = 1/2 ‖r‖^2` | identity `W`, **deliberate and labelled** (`optimizer_display:27-30, 262-266`) | n/a | yes | — / fitting / functions |
| N | **The geometries** (parameter side). Frontend `MANIFOLD`/`COREWISE`/`UNIFORM_*`/`shared(...)`: `project`, `project_oblique`, `inner`, `norm`, `project_ambient`, `retract`. Backend `Geometry` (`optimizers:75`) and `*GeometryOps` (`geometry:289-1252`): `project`, `inner`, `point_norm_sq`, `point_tangent`, `curvature_precompute(frame, misfit_gradient, ...)`, `lift`, `lift_transpose`, `tangential_curvature` | T / C | `R_X^{-1}` (with `Pi`) `: X' -> X`; the metric | `R_X` explicit (`MANIFOLD.project` = `Pi`, HS on an orthonormal frame, checked; `COREWISE.project` = `I`; `shared` = mean per group, the induced metric's) | `T3FrameWeights` not pluggable (P2) | the curvature members: through `misfit_gradient` | — / geometry / classes |
| N' | **The regularizer.** `IdentityRegularizer` (`regularization:75`) | point | `1/2 λ ‖x‖^2_{R_X}` | uses `R_X` (HS on `MANIFOLD`, Euclidean on `COREWISE`) | n/a | yes | — / fitting / classes |
| O | **The legacy, deleted in P20.** `sampling_derivatives` (57 names); `probing` `compute_*`/`assemble_*` and `*_from_sweep`; `utv_`/`ut3_` `*_derivatives*`; the old kinds in `backend/fitting.py` (`SamplingKind.transpose` applied `omega^2`, `:608-621`) and `uniform_fitting.py` | — | — | the old layer weighted inside the kind and left the kernels unweighted, the same split as today | — | — | out of scope |
| Q | **The other weights**, not data-side: `T3Weights` (`t3_weighted_norm/inner`, absorb), `T3FrameWeights` (`fv_absorb_weights`, `T3Tangent.weighted_inner`) | P / T | metrics on `E` and on `X` | parameter-side | — | no | the input-space check: none is input-side |

Counts for the blast radius:

- Weightless transposes and Hessians: 18 in the backend (C, E, F, ragged and uniform, three kinds each) plus 3
  ambient; 15 frontend methods (TT 3+3, UTT 3+3, `T3Tangent` 3; and `UT3Tangent` 3) plus 3 static ambient ones.
- Keyword calls `residual=`: about 140 in `tests/` (`test_uniform_probing` 26, `test_frontend_sampling` 26,
  `test_dispatch` 24, `test_tucker_tensor_train` 20, ...) and 17 in module doctests.

---

## 3. Findings

### 3.0 What was run

`scratchpad/auditor/adj.py`, numpy, `x` of shape `(4, 5, 6)`, `W = 3`. At `order = (2,)` (input `affine_jet(ww, p)`)
and `(1, 1)` (`affine_jet(ww, p, q)`):

- `t3_probe_corewise_transpose` against `t3_probe_corewise_tangent` under the plain slot pairing
  `sum r * J x_dot`: relative error 8e-16 and 0.
- `tv_probe_transpose` against `tv_probe`: 6e-16 and 3e-16.
- The engine with `held=HeldVariables(Jp=r)` and `weight=omega` against `J^T(omega^2 r)`: 0 and 6e-15.
- `t3_probe_corewise_hessian(x, ww, c, x_dot)` against a central difference of the corewise transpose:
  5e-9 and 7e-9, at `h = 1e-4` (an `O(h^2)` error).

### 3.1 Data side

**D1. The transposes are declared `Y' -> X'`, and the conversion `Y' -> Y` is a silent identity.**

- The backend docstrings say "`residual` is a vector of the type of `z` ..., with no weight; it is turned inside"
  (`probing:1042,1191`, `apply:181,364`, `entries:158,316`).
- The frontend says "the usual Gauss-Newton `J^T r`" (`manifold:768,916`; `uniform_manifold:643`) and "the summed
  gradient `J^T r`" (`tucker_tensor_train:3837,4003`; `uniform_tucker_tensor_train:498`). Both are true only for
  `W = I`.
- `docs/entries_apply_probe.md:157,166` gives the Riemannian gradient as `MANIFOLD.project(op_transpose(r, ...))`,
  again with `W = I`. This page is stale and is rewritten in P19.

Claude's reading, the argument as a `Y` element `c`, is correct and metric-free (3.0). But no signature, name or
docstring says it. A user with a weight who follows the docstrings computes the unweighted gradient, with no error.

**D2. The corewise Hessians: the same, on the `r` route.**

They hold `r = residual` with no weight (`probing:1574`). The engine then forms `w_tilde = flip(I r)`. The
docstring and signature say "c, of the type of z ...; no weight" (`probing:1540`, `apply:425`, `entries:375`,
`ut3_sampling:553,587`), and the frontend says "`residual` is a vector of the type of `z`, no weight"
(`tucker_tensor_train:4139`). Here the docstring at least calls the argument `c`. The curvature term of a fit needs
`c = W r`, and the oracle supplies that from the batch (K).

**D3. Two public representations of `Y`.**

The engine's `Y` is turned: `w_tilde` (`HeldVariables.w_tilde`, "turned", `sampling_variables:899`;
`jet_to_dual_jet`'s result). The transposes' `Y` is plain. `t3toolbox.fitting` exports `jet_to_dual_jet` and its own
checked `dual_jet_to_jet` (`fitting:83,100,190`), but no frontend function takes a dual jet. Only tests use the pair
(`tests/test_jets.py`, `test_dispatch.py`, `test_jax_absent.py`). The name "dual jet" reads as "the covector", that
is, the transposes' argument. Passed to a transpose at a nonempty jet shape, it is flipped twice: a wrong answer, and
no error. At `order = ()` it is harmless, so a plain-sampling test would not catch it.

**D4. The variance of `Y` against `Y'` is not encoded.**

The kind-typed helpers convert objects "of the type of `z`" between derivative and Taylor coordinates:

- `derivatives_to_taylor` divides slot `k` by `k!`. That is correct for data, residuals and `Jp`, all in `Y'`.
- A `Y` element converts the other way. Write `D = diag(k!)`, `r_d = D r_t`, and an unweighted objective in
  derivative coordinates. Its gradient is `J_d^T r_d = J^T D r_d = J^T D^2 r_t`.
- So the correct call is `transpose(taylor_to_derivatives(r_d))`, the converter with the opposite name.
- Both natural calls are wrong: `transpose(derivatives_to_taylor(r_d))` misses `D^2`, and `transpose(r_d)` misses
  `D`.

The fitting layer is right because `prepare_derivative_data` carries the factorials in `omega` (`jets:701`). A
hand-rolled gradient with the public transposes is a trap. This is the strongest argument that `W` should not be
left implicit at the transposes, either through a weight or through an explicit, typed `Y`.

**D5. What the weight can express.**

`check_weight` accepts exactly `(d,)+F` (probe), `F` (apply) or `()` (entries) (`jets:1290-1292`). The engine's
steps reshape it as `F + (1,)*...` (`sampling_steps:970,1053`), so it cannot carry `W` or `C` axes. In consequence:

- There is no per-sample weight: heteroscedastic noise, per-measurement confidence, or a dropped sample.
- For entries, the "weight" is one scalar: it scales the whole objective and weights nothing.
- There is no metric within a mode for probe, such as quadrature weights or a mass matrix on `V_i`.
- `W` is diagonal and positive semidefinite. A zero weight is legitimate, since it drops a block. `W` is then not a
  Riesz isomorphism, and `dual_jet_to_jet` (`W^{-1}`) does not exist there. The frontend raises in safe mode
  (`fitting:272-281`), and the backend divides by zero by design (O-20).

Per-sample weights can be reached only by preparing the input. `apply` and `probe` are multilinear in `ww`, so
scaling every `w_j` by `omega_s^(1/d)` (apply), or by `omega_s^(1/(d-1))` (probe, `d >= 2`), and `y_s` by
`omega_s`, weights sample `s` by `omega_s`. Entries must first be rewritten as apply at `index_to_one_hot`. This is
not documented.

The display assumes the weight's shape is the table's shape ("A block is an entry of the weight",
`optimizer_display:27-30`; O-42). A per-sample weight would break that assumption.

**D6. The input space: no weighting anywhere.** Confirmed.

- `T3Weights` and `T3FrameWeights` are parameter-side (Q).
- `rescale_coordinates` and `rescale_direction` are changes of coordinates, not metrics.
- For the probe kind the output `z_i = X(w_0, ..., ., ..., w_{d-1})` is a covector on `V_i`, so `Y' = ⊕_i V_i'`
  (per sample and slot). `rescale_coordinates` treats it so: probe data are divided by the scales, the contragredient
  transform (`data_preparation:117`, docstring `:20-23`). The docstring's "which also changes the loss" is the
  identity metric on `V_i'` showing through.
- The probe ambient transpose places `residual_i` as a vector of `V_i` in a CP factor (`probing:1126-1130`), which
  identifies `V_i'` with `V_i` through the identity.
- An input metric `M_i` would therefore be, for probe, a data metric, `W_i = M_i^{-1}` on that mode. For apply and
  entries the data are scalars, and an input metric does not enter the loss.

**D7. The minibatch scale and the oracle are consistent.**

- O-21 puts the scale into the data inner product: `W_batch = (n/b) diag(omega^2)`.
- The oracle scales `f`, `g`, `JtJp`, `Jp_norm_sq`, `Kp` and `f_resolution`, and leaves `r` and `Jp` alone
  (`fitting:938-944`). Those are exactly the `W`-bearing outputs and the `Y'` ones.
- `f_resolution` uses `|omega y|` (`fitting:935`).
- This is the one place the scheme is applied in full.

A limit, noted in LOG: `y` cannot carry the frame stack `C`, so a stacked model shares one `y` (the P16 narrowing),
and the weight has no `C` either.

**D8. `rescale_direction` is a weight change.**

With exact arithmetic, multiplying the input and data slots by `length^k` gives the residual `L r`, so the objective
is `1/2 ‖omega L r‖^2`: the weight `omega_k length^k`. The docstring says "only the loss changes". So two public
routes set the same `W`, which is worth one sentence in P19's docs. Rescaling also improves the conditioning of the
input jets, so the two routes are not redundant in floating point.

**D9. Identity `W` that is deliberate.**

- `r_block_sumsq` and `y_block_sumsq` are unweighted (O-21), and the display's per-block ratio is invariant under a
  block-constant weight.
- `obj_unweighted = 1/2 ‖r‖^2` (`optimizer_display:262-266`) adds slots of different units, since slot `k` scales
  as `[z]/[s]^k`. It is labelled `unwt`. A minor point.

### 3.2 Parameter side

**P1. `T3Tangent` stands for both `X` and `X'`.**

- The tangent transposes return a `T3Tangent` holding the bare `𝒥^T c`, a cotangent in variation coordinates,
  ungauged ("Bare ... no gauge projector", `manifold:777`).
- `MANIFOLD.project` is documented as "raw cotangent -> Riemannian gradient" (`manifold:1177`): `Pi` on the
  orthonormal frame, which is `R_X^{-1}` restricted to the gauge. Safe mode checks orthogonality.
- `COREWISE.project` is the identity (`manifold:1387`): the Euclidean `R_X`, correct for that geometry.
- The backend uses plain tuples throughout, and `LocalModel` always projects (`optimizers:363-401`), so the fitting
  path is right. The type name is the only implicit part. A frontend user who takes `T3Tangent.probe_transpose(...)`
  as "the gradient" on `MANIFOLD` has an ungauged cotangent, as documented.

**P2. `R_X` is fixed per geometry, and `T3FrameWeights` is not pluggable.**

The Grasedyck-Kramer frame metric exists (`T3FrameWeights`, `weighted_inner`), but no `Geometry` takes it. A
preconditioned manifold is reachable only by writing a geometry. This is not a type error. It is the parameter-side
counterpart of "where weights live".

**P3. The regularizer depends on `R_X`.**

`IdentityRegularizer` is `1/2 λ ‖x‖^2` in the geometry's metric: HS on `MANIFOLD`, weight decay on `COREWISE`
(`regularization:75-96`). So the objective itself changes with the geometry. This is documented. Under L2 it is an
objective term and belongs to fitting, where it is.

**P4. The geometry's curvature members take the misfit gradient.**

`curvature_precompute(frame, misfit_gradient, ...)` (`optimizers:134`) is fed `g`, which is objective-dependent. But
with the covector typing, `K_{grad f}` is "the Weingarten map at a covector": `tangential_curvature` applies
`t3_inner_product_corewise_hessian` at that covector (`geometry:540`). That is exactly the shape of the corewise
"Hessian" (F). If F counts as free of the objective, so do these; if F moves to fitting, these are the same case.

### 3.3 Naming and placement

**N1. `residual` is the name of a role, not a type.**

- The engine holds the transposes' argument as `Jp` (`probing:1065`) and the Hessians' as `r` (`probing:1574`). Both
  are `Y'` roles, to which the engine then applies `W` (here `I`).
- The engine is consistent internally: `r`, `z` and `Jp` in `Y'`, and `w_tilde` in `Y`.
- The public functions borrow a `Y'` role name for a `Y` argument. The plan's N16 merged `ztildes`/`c` into
  `residual` (plan 2.4 table). Before that, `c` was the apply/entries name, closer to the `Y` reading.

**N2. The line "by object" collapses for corewise.**

Since P16 the corewise geometry's frame is the point `(B, G)`. So D, E and F act on the tangent space of the flat
manifold at `x`, which is the same object as the point's core space. The library holds them two ways:

- methods of TT/UTT taking raw core pairs (O-49: "a frontend corewise direction is the pair of core tuples only");
- `T3Tangent` at `COREWISE.frame(x) = (B, G, G, G)` (`manifold:1359`). There, `T3Tangent.probe` computes the same
  `J x_dot` as `x.probe_corewise_tangent`, through the four-store frame, with its sweeps through `G` duplicated
  (plan 2.4).

The second route is what `fitting.LocalModel` uses on `COREWISE` (`fitting.py` module docstring, line 20). The
check-in on `t3_corewise_frame` before P20 is this question.

**N3. "corewise_hessian" is a family name, not a one-off.**

`t3_inner_product_corewise_hessian` (`t3_linalg:857`; `ut3_linalg:426`) has the same meaning, the Hessian of
`theta -> <c, op(theta)>`, and the same pairing with `*_corewise_transpose`.

- Candidate names:
  - `*_corewise_transpose_tangent`, the directional derivative of the transpose. It fits the `_tangent` stem
    documented under O-48, and keeps "transpose" in the name, which is where the covector enters.
  - `*_corewise_second_adjoint`, the AD term.
  - `*_corewise_hessian` kept, with a docstring that leads with `phi_c`.
- A rename touches 12 sampling names, plus 4 `inner_product` names (and their `_from_sweep`) if the family is kept
  uniform.
- None of the sampling names is released (added in P12 and P13), so no deprecation is needed. The `inner_product`
  names were released in 2026.3.0.

**N4. The jets module mixes both sides of L2.**

`backend/jets.py` holds the objective-free input jets and sample checkers together with the objective-side `W`
(`jet_to_dual_jet`, `check_weight`, the weight converters). The frontend already routes all of it through
`t3toolbox.fitting`, except `affine_jet` and `index_to_one_hot`, which are top-level too.

---

## 4. Options for organizing

The first two choices are independent: the type of the data argument (4.1) and the reach of the data metric (4.2).
Placement (4.3) then follows.

### 4.1 The type of the data argument of the transposes and Hessians (C, E, F, G)

**T-a. Typed `Y`, no weight. The objective line, made explicit.**

- Rename `residual` to a `Y` name (`covector`, or `c`, the pre-N16 apply/entries name) in the 18+3 backend
  functions and 15+3 frontend methods.
- The docstrings lead with `phi_c(x) = <c, z(x)>` and its gradient.
- Fix the five "Gauss-Newton `J^T r`" sentences.
- Add one public, explicit data Riesz map in the fitting layer, plain and unturned: `weigh(kind, order, r, weight) ->
  W r` (`omega^2 * r`, per block). A hand-written weighted gradient is then `transpose(..., weigh(r, omega))`.
- Optionally add `covector_derivatives_to_taylor` for D4, or document the contragredient rule beside
  `derivatives_to_taylor`.
- Pros:
  - No objective in any sampling function.
  - Future-proof: a nonlinear loss supplies its own `W`, and the transposes do not change.
  - The engine is unchanged.
  - Satisfies the principle: the conversion `Y' -> Y` is a visible call, and the weight is its argument.
- Cons:
  - The weighted gradient takes two calls.
  - The maintainer asked for "first-class" weights, which this places beside the functions, not in them.
- Blast radius:
  - About 140 test and 17 doctest keyword calls `residual=` (positional calls are unaffected).
  - P13's surface: names of arguments only, nothing moves.
  - P17: none (no example calls a frontend transpose, plan P13).
  - P19: `transposes.md`, `entries_apply_probe.md` and `naming_conventions.md` are rewritten anyway.
  - P20: none.

**T-b. Typed `Y'` with a required weight. First-class weights in the functions.**

- Keep `residual` (`Y'`) and add `weight` (`(d,)+F | F | ()`) with **no default**, as `jet_to_dual_jet` already
  requires (`jets:339`): `None` must be written.
- Backend: one keyword passed to the engine (3.0 shows the route works).
- Pros:
  - One call gives `J^T W r`.
  - The principle holds by signature: no silent identity, because `None` is explicit.
  - Consistent with the oracle's batch.
- Cons:
  - Objective data (`omega`, least squares, diagonal) in every objective-free map.
  - The parameter's type is block-scalar `omega`. If 4.2 widens the metric later, every sampling signature changes
    again, the break the maintainer wants to avoid.
  - Uses outside fitting (a sketch, `J^T` of an arbitrary covector, CP back-projection) must pass `None` with no
    meaning to it.
  - The ambient transposes need jets-free weight shapes.
- Blast radius:
  - Every call of 33+6 functions: the roughly 91 positional transpose calls P13 counted, the about 140 keyword
    calls, every Hessian test (`test_curvature`, `test_uniform_curvature`, `test_frontend_sampling`, `test_dispatch`),
    and the module doctests.
  - P13's surface: one more argument on 36 public functions.
  - P17 to P20: as T-a.

**T-b'. As T-b, with `weight=None` defaulting to `I`.** Additive, nothing breaks. But the default *is* the silent
identity the principle names, so T-b' meets the request and not the principle.

**T-c. A typed data metric object.**

- The transposes take `c ∈ Y`, as in T-a.
- The fitting layer holds `W` as an object: a small frozen `DataMetric` protocol with `apply(kind, order, r) -> W r`
  (and `pairing`); `DiagonalWeight(omega)` is the least-squares one, and a future loss supplies its own.
- The batch's `weight` becomes a `DataMetric`, or stays an array wrapped by the oracle.
- Pros: the scheme encoded in types; nonlinear losses and wider metrics (4.2) fit without touching sampling.
- Cons: a new class (the razor: a raw-data user still calls `weigh` on arrays, so a function twin is needed); the
  `LeastSquaresBatch` and oracle shapes move again, after P14 and P16.
- Blast radius: T-a, plus `backend/fitting.py`, `check_batch`, the batch's pytree fields, the display's shape rule,
  and the P16 tests of the batch.

### 4.2 The reach of the data metric (independent of 4.1)

**R-0. Keep the block-scalar `omega`.**

- Document per-sample weighting by input scaling (D5) and its limits.
- Note that entries has only a scalar weight.

**R-1. Add a per-sample `W` axis.**

- The weight becomes `(d,)+F+W'` with `W'` broadcastable to `W`, or a separate `sample_weight` of shape `W`.
- Changes `check_weight`, the reshapes in `f_step`, `w_tilde_step` and `Jp_norm_sq_step` (`sampling_steps:957-1060`),
  `take` (subset the weight with the samples), `f_resolution`, and the display (the table is per block; a per-sample
  factor must be summed out).
- Entries then has a real weight.
- Medium. Engine-local, but it reopens O-42's display rule.

**R-2. An operator `W`** (within-mode mass matrices for probe; correlated slots; a nonlinear loss's Hessian).

- Needs T-a or T-c, since an operator cannot be a reshape of a weight.
- The engine's `w_tilde_step` would call `W` before the turn.
- Part 2 at the earliest.

### 4.3 Placement

**L1. By object** (point: base modules; tangent: tangent layer; corewise as a flat manifold).

- A and A' on the point classes, B and C in the tangent layer: as today.
- D, E and F: L1 gives no answer (N2). Two readings:
  - *L1-p.* The flat manifold's tangent space is the point's core space: stay on TT/UTT with raw pairs (O-49, as
    today).
  - *L1-t.* They are tangent operations of `COREWISE`: move to the tangent layer, taking a `T3Tangent` at the
    corewise frame. This reverses O-49 and keeps `t3_corewise_frame`. 12 frontend methods move; the
    backend is unchanged. P19 and P20's check-in would be redone.
- G: frame-free, so neither.
- Assessment: L1 is a good rule for A, B and C, and it adds nothing for corewise.

**L2. By objective dependence** (the maintainer's lean).

- Objective-free: A to G, I'', the engine as infrastructure.
- Objective-dependent: I (`W`), I' `prepare_*` and the weight converter, J, K, L, M, N', and `check_weight`.
- The current layout is already close to L2. The deltas:
  - (i) the re-typing of 4.1 (T-a or T-c);
  - (ii) `jets.py` split, or its weight half documented as fitting-side, with `jet_to_dual_jet` and
    `dual_jet_to_jet` made backend/engine-only or renamed `*_turned` (D3);
  - (iii) decide whether N's curvature members count as objective-free operators at a covector (P4). If F is
    objective-free, so are they, and nothing moves.
- Blast radius:
  - (ii): the frontend exports in `fitting:83,100` and `fitting.dual_jet_to_jet`, with their tests (`test_jets` 71
    references, `test_dispatch` 16, `test_jax_absent` 7).
  - (i): as in 4.1.

**L3. Methods versus functions.**

- *L3-maintainer.* The raw operations and their transposes are methods. F (the Hessians) become functions in
  `t3toolbox.fitting`, 6 frontend methods (TT 3, UTT 3), as `fitting.probe_corewise_hessian(x, ww, c, x_dot, ...)`.
  Their P13 tests and doctests move. The backend is unchanged (it has no methods). Consistency then wants the same
  for `inner_product_corewise_hessian`, which has no frontend method today.
- *L3-Claude.* F stays as methods, renamed (N3); the objective-dependent operations (K, L, M) stay functions and
  classes in `fitting`/`optimizers`, as today.
- Either way the ambient transposes stay static methods, frame-free, as the tangent transposes are static
  constructors (N16).

### 4.4 Combinations that hang together

- **Minimal and principled: T-a + R-0 + L2(ii) + L3-Claude.** Re-type and rename one argument, add `weigh`, fix the
  docstrings, make the turned representation engine-only, rename the Hessians if wanted. Small; mostly P19 text.
  Leaves R-1 for later.
- **First-class weights now: T-b + R-1.** The weight is in every sampling signature, and per-sample at once, so the
  signature is not reopened for that. A large test churn. The risk is R-2 later.
- **Typed for the future: T-c + R-1 + L2 + L3-Claude.** Most new structure, best for nonlinear losses. It touches P14
  and P16's batch again.

---

## 5. What the maintainer must decide first

1. **The type of the transposes' data argument: `Y` (a covector, no weight) or `Y'` (a residual plus a required
   weight).**
   - Everything else follows. T-a and T-c say `Y`; T-b says `Y'`.
   - The deciding question: should an objective-free map such as `J^T` carry the objective's metric?
   - Under the maintainer's own refinement (the dividing line is objective dependence), `Y` is the consistent answer,
     and "first-class" then means a first-class `W` function beside the transposes, not a parameter in them.
2. **The reach of `W`: block-scalar `omega`, per-sample, or an operator.**
   - The answer decides whether a `weight` parameter in sampling signatures can be stable (only if `W` stays an
     array of fixed meaning).
   - It also decides whether entries fits are weighted at all.
3. **Corewise: point methods with raw pairs (O-49), or tangent operations of `COREWISE`.** The line "by object" cannot
   decide it (N2), and P19 and P20's `t3_corewise_frame` check-in depends on it.
4. **Names**, after 1 to 3: the argument (`residual` -> ?); the Hessians (and whether `inner_product_corewise_hessian`
   follows); whether `jet_to_dual_jet` and `dual_jet_to_jet` stay public.

Scheduling note: the P13 surface is unreleased, so every option here is a break of unreleased names only, except
`inner_product_corewise_hessian` (released in 2026.3.0). Doing it before P17 keeps the examples (P17) and the pages
(P19) to one pass.
