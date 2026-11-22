# The data metric: a sub-arc between P16 and P17 (every question decided, 2026-11-22)

Status: Claude's first draft (2026-11-21), revised by the writer through seven review rounds (the answers:
`DATA_METRIC_PLAN_RESPONSE_r1.md`). The maintainer has decided every question (`TARGET_DECISIONS.md`, the notes of
2026-11-22 from "The data-metric plan's questions: the maintainer's first answers" to "The data-metric plan: Q3"); §8
records the decisions. The reviewer signed off this version (`DATA_METRIC_PLAN_REVIEW_r7.md`). It is ready for the
maintainer's final read, with one open question (§8: the constructor `blocks_to_weight`).
Implements: `TARGET_DECISIONS.md`, "The data metric: decisions 1 and 2 of the sampling audit", with O-21, and the
notation decision ("Notation for the jets and the data metric").
Evidence: `SAMPLING_AUDIT.md`; for derivative coordinates, `T3POLYNOMIAL_JET_USE.md`. `file:line` refers to
`aaee0b04` unless a slice's counts name another commit.
Out of scope: decision 3 (corewise as point methods or `COREWISE` tangent operations); decision 4 except the frontend
status of the dual-jet converters (decided, Q9): the names of the Hessians and of the backend converters; a
non-diagonal `M` (a covariance); nonlinear losses.

**In brief.**
- The transposes and second-order transposes take an element of `Y` (dual jets in slot coordinates), with no weight
  inside, and their argument `residual` becomes `c`.
- One public function, `c = data_metric(r, weight, minibatch_scale)`, applies `M: Y* -> Y` elementwise. A batch has
  it as a method, `batch.data_metric(r)`. Every place that applies `ω²` goes through it.
- **Every data-space object (`y`, `ω`, residuals) has the type of `z`**: each axis 1 or `z`'s size, never `K`.
  There is no block form; a constructor (proposed `blocks_to_weight`, §8's one open question) builds a weight from
  per-block values. Per-element data on a stack follow from the rule.
- Data are in Taylor coordinates everywhere, with one conversion at the boundary.
- Lightweight typing (§9): algebraic types in the signature comments, with `*` for duals; the dual-jet converters
  private in the frontend; transformation-law tests.
- Six slices, M0 to M5, before P17. M0 renames the jet group `J` to `P` and `J_in_i` to `F_in,i`.

## 1. The mathematics, typed

| Symbol | Space or map | In the code |
| --- | --- | --- |
| `X` | parameters: a frame's variations `(V, H)`, or raw cores on the corewise point | directions `p`, `ẋ` |
| `X*` | its dual | the blocks the transposes return; `g`, `JtJp`, `Kp` before the projection |
| `Y*` | the measurement space (`J` lands here) | `z`, `y`, `r = z - y`, `Jp` |
| `Y` | its predual, `Y** = Y` | the transposes' argument `c`; `M r` |
| `J: X -> Y*` | the tangent forwards (`J = dz(x)`, the upright `d` of the glossary) | `tv_*`, `*_corewise_tangent` |
| `J^T: Y -> X*` | the metric-free transposes: `<J^T c, ẋ> = <c, J ẋ>` | `tv_*_transpose`, `*_corewise_transpose`, `*_ambient_transpose` |
| `M: Y* -> Y` | the data metric: `M = γ · diag(ω²)` today, possibly a loss Hessian later | `data_metric` (§3) |
| `R_X: X -> X*` | the parameter metric; `R_X^{-1}` is `geometry.project` | unchanged here |

`*` is the algebraic dual (Q10). Concretely (§9.0), for apply `Y* = P` and `Y = P*`; for probe `⊕_i P⊗V_i*` and
`⊕_i P*⊗V_i`; for entries `ℝ` and `ℝ`. All of these are per sample.

**The pairing of `Y` with `Y*`** is slot by slot on Taylor coefficients: `<c, a> = Σ c·a` over the modes (probe), the
slots `k` of the box `F`, the samples `W` and the mode edge. There is one pairing per element of the stacks `K` and
`C`. It holds no metric. The flip in `jet_to_dual_jet` is how the engine represents this pairing (as the corner
coefficient of a jet product), not a metric (audit 3.0, verified to 1e-16).

**The objective and its outputs** (`f`, `g`, `JtJp`, `Jp_norm_sq`, `Kp` as the oracle returns them):

    f = ½ <M r, r>,   g = J^T M r,   JtJp = J^T M J p,   Jp_norm_sq = <M Jp, Jp>,   Kp = d(J^T c)[p] at c = M r.

**The minibatch scale is part of `M`** (O-21). `γ = 1` on a full batch, and `γ = |W|/b` on a draw of `b` of its `|W|`
samples.
- Every `M`-bearing output carries `γ`; the `Y*` outputs (`r`, `Jp`) do not. Hence `J^T(M r)` equals the oracle's `g`
  on every batch, drawn ones included.
- `f_resolution = eps · (√<M y, y> √(2f) + |f|)`, with `γ` in both `M` and `f`. This is today's value
  (`fitting.py:933-944`).
- The engine never sees `γ`; the oracle multiplies its outputs by it.

**The objective-free maps contain no `M`.** They are the forwards, tangents, transposes and second-order transposes (42
public functions, §5). `M` is the objective's: `data_metric`, the oracle, the optimizers. The engine's `weight` argument
is the oracle's channel into the engine; no objective-free public function passes one.

**Coordinates (Q7).** The library's data coordinates are Taylor coefficients, everywhere. A user with derivative data
converts once, at the boundary: data and weight together (`prepare_derivative_data`), and every stored copy stays in
Taylor coordinates. §4 gives the recipe for a data-relative weight. A user who composes an objective in other data
coordinates must still convert `Y` elements the contragredient way (audit D4):
- in derivative coordinates, slot `k` is `r_d[k] = k! r_t[k]`, with `k! = k_1!…k_m!`;
- the pairing then fixes `c_t[k] = k! c_d[k]`;
- so a `Y` element converts by `taylor_to_derivatives`, the converter with the opposite name.

The two hand-built examples do this, correctly (`examples/fit_hilbert_from_apply_derivatives.py:142-144`,
`..._flat.py:137-139`). Probe data under `rescale_coordinates` are the same case: they are covectors in `V_i*` (audit
D6).

## 2. The rule for data-space objects (Q5, Q8)

**Every data-space object has the type of `z`.** That covers the data `y`, the weight `ω`, the residual `r`, and any
other array of `z`'s type that a user passes.
- **Arrays.**
  - For probe, the object is `d` arrays: a tuple or list, or one array whose first axis is the mode. The number is
    exactly `d`; a mode axis of 1 is not broadcast (6.7's explicit point, kept).
  - For apply and entries, the object is one array.
- **Rank.** Each array has exactly the rank of `z`'s array.
- **Sizes.** Each axis has size 1 or `z`'s size.
- **Never `K`.** An object carrying a stack of directions raises by rank.

| Kind | Each array of `z` (packed probe: `N` for `N_i`) | Examples of `ω` |
| --- | --- | --- |
| probe | `F+W+C+(N_i,)` | per slot and mode: `F+(1,)*(len W+len C)+(1,)` per mode; per sample: `(1,)*m+W+(1,)*len C+(1,)`; per entry: `z`'s shape |
| apply | `F+W+C` | per slot: `F+(1,)*(len W+len C)`; per sample: `(1,)*m+W+(1,)*len C` |
| entries | `W+C` | one number: `(1,)*(len W+len C)`; per sample: `W+(1,)*len C` |

- **What a size of 1 means:** the same along that axis.
  - Shared data on a stack has size-1 `C` axes.
  - A per-sample weight has size-1 jet axes.
  - A weight constant over the edge has an edge of size 1.
  - With `C = ()` (an unstacked model), `y` keeps today's shape.
- **`ω`** enters squared and may be zero, which drops entries. `None` means ones.
- **No block form.** Today's `(d,)+F`, `F` and `()` weights are not accepted. `blocks_to_weight` builds a full-form
  weight from them (§4).
- **Per-element data on a stack follow from the rule** (Q8). `y` and `ω` may carry `C` at full size: one dataset
  per stacked model. This lifts P16's narrowing (one `y` shared across a stack, LOG:1240). It also removes `r_step`'s
  special alignment of a `y` without `C` (`sampling_steps.r_step`): `r = z - y` by broadcasting.
- **Nothing is read by rank, so there is no trap of reading.** A per-sample weight written as a bare `W` raises by
  rank (unless `m = 0` and `C = ()`, where it is the full form). One limit stays, documented in `check_weight` as
  today (`jets.py:1248-1251`): a shape check cannot see axes given in another order when their sizes coincide.
- **The engine's stack check still excludes `y` and `ω`** (`sampling_variables.py:1097`). For them a size-1 axis is
  meant; any other mismatch fails in broadcasting.

**This supersedes, by name:**
- 2.5 ("the exact shape only", `TARGET_DECISIONS.md:1326-1332`);
- 6.7's broadcasting decision (`:2307-2317`), keeping its mode-count point;
- this plan's earlier block-or-full reading rule.

**The rejection table under the rule** (`tests/test_jets.py:849-870`: `d = 3`, `W = (7,)`, `n = 3`, `C = ()`).
- Two of the 19 cases become legal: apply `(1,)` at `()` and entries `(1,)`. Both are a full form with a `W` axis of
  size 1.
- The other 17 raise: by the mode count, by rank, or by a size that is neither 1 nor `z`'s.
- The block forms that `test_check_weight_passes_the_exact_shape_and_none` accepts now raise. The exception is where
  block and full form coincide (`W = C = ()`). That test's table becomes the full forms.

**The checkers** gain the stack:
- `check_data(kind, order, sample, y, stack_shape=())`;
- `check_weight(kind, order, sample, weight, stack_shape=())`;
- `LeastSquaresOracle.check_batch(batch, shape, stack_shape=())`.

They check the rule against `z`'s shape, read from the sample as `check_data` reads it today, plus the frame stack
`C`. The default `()` is the unstacked model; a stacked `y` checked without its stack fails by rank, not silently.
Callers:
- `check_batch` (`fitting.py:958`);
- `test_data_preparation.py:100-101`, `test_dispatch.py:823`, `test_jax_absent.py:182`, `test_jets`;
- the frontend batch preparation.

## 3. The public `M`: `data_metric` (Q1, Q3)

```python
def data_metric(
        r:                typ.Any,                   # Y*: of the type of z (§2), no K. probe len=d (or packed), elm_shape=F+W+C+(Ni,); apply F+W+C; entries W+C
        weight:           typ.Optional[typ.Any],     # ω: of the type of z (§2); None = ones
        minibatch_scale:  typ.Optional[typ.Any],     # γ (O-21): |W|/b on a drawn batch; None = 1
) -> typ.Any:                                        # Y: γ * ω**2 * r elementwise; the structure of r (probe: a tuple for a tuple)
```

- **Elementwise; no `kind` or `order`.** The full-form `ω` has `r`'s rank, so it broadcasts positionally. The flip is
  the engine's business.
- **How arrays pair.** A tuple or list `r` is read element by element, with the weight as a tuple or along its first
  axis. An array `r` pairs with an array weight elementwise; that includes a packed probe, `(d,)+F+W+C+(N,)`. Each
  pair must have equal rank, so a `K`-stacked `r` raises. Products with a stack of directions go through the oracle.
  With `weight=None` there is no rank to compare, and the result is `γ r`.
- **Two edge cases for M1's design note.**
  - A packed `r` with a ragged tuple weight is not covered by the pairing above. Either it raises, or the tuple is
    read mode by mode.
  - A packed weight whose mode axis has size 1 would broadcast unchecked over the modes. This is a documented limit,
    not a hole: the checkers take the ragged form only, and `pack_batch` builds packed weights from checked ragged
    ones.
- **Both factors are required** (Q3), with no defaults. `None` written out means ones, or a scale of 1.
- **The dtype** is the lower floating precision of `r` and `ω`, as today.
- **The method:** `LeastSquaresBatch.data_metric(r)` is `data_metric(r, self.weight, self.minibatch_scale)`. That is
  the common call, and it cannot forget the scale.
- **Home: `backend/jets.py`**, with the jets helpers, in its `__all__`. It is re-exported by `t3toolbox.fitting`.
  - `jets` imports only `contractions` and `common`.
  - The engine (`sampling_steps` → `jets`) and the oracle (`backend/fitting.py` → `sampling_variables` →
    `sampling_steps` → `jets`) both reach it, with no cycle.
  - The method lives on the class in `backend/fitting.py`.

**One implementation of `ω²`.** `data_metric` is `γ × _omega_squared(r, ω) × r`. Its private core `_omega_squared` is
the one place where `ω²` is formed, in the lower precision of the two. The engine inserts `K` itself, with no `γ`:

| Site | Today | After |
| --- | --- | --- |
| the engine's `K` | (weights had no `C`) | `_with_K(ω, lens)`: `lens.K` size-1 axes after `F+W`, placed from the group lengths (engine-private) |
| `w_tilde_step`, also as `_Jp.w_tilde` (`sampling_steps.py:988`) | `jet_to_dual_jet` | `flip(data_metric(value, ω_K, None))` |
| `f_step` (`:1052-1053`) | own reshape of `ω_F` | `½ Σ r · data_metric(r, ω, None)`, keeping `C` |
| `Jp_norm_sq_step` (`:969-970`) | own reshape | `Σ Jp · data_metric(Jp, ω_K, None)`, keeping `K + C` |
| `_modes` (`:913-922`) | `asarray(weight)[i]` | per array, by §2 (a ragged tuple cannot be `asarray`-ed) |
| `r_step` | inserts `C` into `y` | `z - y` (`y` has `C` by §2) |
| `f_resolution` (`fitting.py:935`) | `ω² × y_block_sumsq` | `Σ y · data_metric(y, ω, None)`, per element of `C`, then `× γ` (§4) |
| the display's weighted block sums (new, §4) | | through `data_metric` |
| `jet_to_dual_jet` (`jets.py:298`) | `weight² · jet`, flipped | `flip ∘ data_metric(jet, ω, None)` |
| `dual_jet_to_jet` (`jets.py:498-507`) | divide by `_squared_weights` | flip, then divide by `_omega_squared` (`M^{-1}` at `γ = 1`) |
| frontend zero check (`fitting.py:275`) | `_squared_weights` | deleted with the frontend `dual_jet_to_jet` (Q9, §9.2) |

- **`jet_to_dual_jet` and `dual_jet_to_jet` keep their signatures,** `(kind, order, …)`, since the flip needs them.
  - Their weight is the full form.
  - Routed through `data_metric`, they raise on a `K`-stacked jet only when a weight is given, since then there is a
    rank to compare. With `weight=None` they flip a `K`-stacked jet, as today. The engine's own `K` path is
    `_with_K`.
  - D3 and their backend names stay with decision 4.

## 4. Through the fitting layer

**`blocks_to_weight(kind, order, blocks, y)`**, the constructor (Q5).
- It places per-block values on the jet axes (and the modes) of a full-form weight shaped like `y`, with size 1 on
  every other axis (`W`, `C`, the edge). The blocks are `(d,)+F` for probe (per mode and slot), `F` for apply (per
  slot), `()` for entries (one number).
- It returns a `d`-leading array for probe and one array otherwise. It reads only `y`'s rank, which includes `C`.
- It is how every block-shaped quantity becomes a weight:
  - `equalizing_weight(data_block_norms(…))`;
  - per-mode noise levels and per-slot weights;
  - the translation's and the bridge's old weights;
  - the pages' examples.
- **The name and signature** are the one open question (§8).
- `data_block_norms` and `equalizing_weight` stay block-shaped: they are per-block quantities, not weights.

**Derivative coordinates (Q7).** Taylor coordinates everywhere, one conversion at the boundary.
- **The conversion.** `prepare_derivative_data` converts the data and the weight together, at assembly. Every stored
  copy (data, validation data, anything compared with a model's output) stays in Taylor coordinates. Mixing in a
  stored derivative copy is the real risk, a primal one (`T3POLYNOMIAL_JET_USE.md` §3).
- **The recipe for a data-relative weight:** convert the data, then compute the weight, for instance
  `blocks_to_weight(kind, order, equalizing_weight(data_block_norms(kind, order, y_taylor)), y_taylor)`. The weight
  then needs no conversion.
- **`prepare_derivative_data(weight=None)`** still returns `k!` (ones on the derivatives, in Taylor coordinates), now
  in full form. Its docstring says that this is rarely what data with decaying jets want, and points to the recipe.
- **`equalizing_weight`'s relative floor depends on the coordinates.** Taylor coordinates widen the spread across
  slots by up to `n!`; one map in the study already spans 1.3e6 in derivative coordinates. The existing advice,
  "rescale the direction first, then equalize", covers it, and is stated beside the recipe.
- **Where it is documented:** `prepare_derivative_data`, `data_metric`, `derivatives_to_taylor` /
  `taylor_to_derivatives` (the D4 sentence), and the pages (P19).
- **The factorials** (`_taylor_weight`, `jets.py:255-263`).
  - Today they are applied with right-aligned broadcasting, which puts them on a full-form weight's last axes. Checked:
    an apply `ones((3, 3))` at `(2,)` returns `[[1,1,2]]*3` where it should be `[[1,1,1],[1,1,1],[2,2,2]]`.
  - In the full form the jet axes are the leading `m` axes of every per-mode array. So the conversion is `ω ×
    factorials` reshaped to `F+(1,)*(ndim-m)`, `_scale_slots`'s own alignment. A size-1 jet axis comes back with full
    `F`.
  - `derivative_weight_to_taylor_weight(kind, order, weight)` gains `kind` (the module is unreleased).
  - The weight comes back in the form it was given.

**Uniform packing**, for `y` and `ω` alike: one rule for any data-space object, in `pack_data` / `pack_batch`.
- Each axis of `F+W+C` becomes the join across the modes: full if any mode has it full.
- The edge is kept at 1 if every mode's edge is 1. Otherwise every edge is broadcast to `N_i` and padded to `N`.
  `pack_if_ragged` must never pad a size-1 edge: it would put the value on entry 0 only (the reviewer's run).
- The arrays are then stacked to `(d,)+F+W+C+(e,)`. An already packed object passes through.
- The engine masks `y` and `ω` on entry with the shape mask on the edge, only where the edge has size `N`. `'y'`
  becomes conditional and `'weight'` is added to `_UNIFORM_MASKS` (`sampling_variables.py:1366`). So a garbage-padded
  `y` or `ω` cannot make `nan · 0 = nan`.

**`take`** (`fitting.py:980-1021`), for `y` and `ω` alike. Their `W` axes start at `m` (an apply or entries array, or a
ragged probe mode), or at `1 + m` (a `d`-leading probe array).
1. If every `W` axis is 1, reshape them to one axis `(1,)`, with no gather.
2. Otherwise broadcast them to the batch's `W`, flatten, and gather `indices`.

`None` passes through. `b` is static, so this stays shape-static under jit.

**`f_resolution`** is `Σ y · data_metric(y, ω, None)` over every axis but `C`, so it is per element, like `f`. It is
computed from the unscaled batch, then `× γ`. A packed probe `y` is unpacked (`fitting.py:928-929`), and so is a packed
`ω` with an `N` edge.

**The display (Q4)** reports `‖ω⊙r_b‖/‖ω⊙y_b‖` per block `b` of the block shape `(d,)+F | F | ()`. That is the
table's shape, not a weight form.
- The numbers come from two new optional outputs, the weighted, unscaled block sums of squares: an engine variable for
  `r`, with its reference definition, and an oracle-local one for `y`.
- `r_block_sumsq`, `y_block_sumsq` and the `unwt` header figure stay plain (O-21).
- A weight constant within each block, such as any old block weight through `blocks_to_weight`, gives today's
  numbers. A block whose weight is zero throughout shows `—`.
- The text "a block is an entry of the weight, so the errors have the weight's shape" becomes "the block shape"
  (`optimizer_display.py:29-30, 165, 186, 423, 584`; `data_preparation.py:137`; `sampling_steps.py:1068`).

**The old forms.** P16's translation (deleted in P17) and the bridge convert their old block weights with
`blocks_to_weight`, so the translation's 19 pins stay. The page doctests that pass block weights to a batch
(`docs/fitting_and_optimization.md:428, 433`) do the same.

**jit and sharing.**
- The weight and `y` stay batch leaves (`fitting.py:782-788`).
- A new value does not recompile. A new shape does (size 1 to full on some axis), and P16's compile test gains that
  case.
- `LeastSquaresBatch.weight`'s annotation widens to a tuple for a ragged probe.
- Shared factors do not interact: the weight is on the data side, sharing on the parameter side.

## 5. The transposes typed (decision 1, Q2)

- **Renamed:** `residual` → `c`, in **42 functions**.
  - Backend, 21: `tv_*_transpose` 3, `utv_*_transpose` 3, `t3_*_corewise_transpose` 3, `ut3_*_corewise_transpose` 3,
    `t3_*_corewise_hessian` 3, `ut3_*_corewise_hessian` 3, `t3_*_ambient_transpose` 3.
  - Frontend, 21: TT 9, UTT 6, `T3Tangent` 3, `UT3Tangent` 3.
  - About 205 mentions in their bodies. No `weight` parameter is added.
- **Documented** as "an element of `Y`, paired with `z` slot by slot (§1). For the gradient of a weighted
  least-squares fit, pass `data_metric(r, weight, minibatch_scale)`, or `batch.data_metric(r)`."
- **Prose retyped.** Every "the Gauss-Newton `J^T r`" or "the summed gradient `J^T r`" becomes `J^T c`, with
  `J^T(M r)` shown (`manifold.py:768, 916`; `uniform_manifold.py:643`; `tucker_tensor_train.py:3837, 4003`;
  `uniform_tucker_tensor_train.py:498`). So does the residual wording around them (`manifold.py:760`;
  `sampling_variables.py:121-122, 872`).
- **Call sites.**
  - 123 `residual=` keyword calls in tests: `test_uniform_probing` 26, `test_frontend_sampling` 26, `test_dispatch` 24,
    `test_tucker_tensor_train` 20, `oracle_sweeps/o1_ragged` 9, `o1_uniform` 7, `test_manifold` 6, `test_fitting` 5.
  - 17 in module doctests.
  - Positional calls are unaffected. P13's right/wrong adjoint doctests stay, renamed.
- **The two examples** keep their code, which is correct, and gain a comment citing the contragredient rule (§1).

## 6. Also changed, and left alone

- **Changed (docstrings and comments):**
  - `jets.py`: the module text (`:51-56`, "no broadcast"), `check_data`, `check_weight`, `jet_to_dual_jet` and
    `dual_jet_to_jet`;
  - the engine's `y` and `weight` docs (`sampling_variables.py:1688-1692`) and the step comments;
  - `LeastSquaresBatch` (`fitting.py:764, 772`), and every text that says `y` has no `C` (the oracle's, the
    engine's);
  - the frontend `fitting` module text (`:45-47`);
  - `data_preparation`: `data_block_norms` and `equalizing_weight` stay block-shaped, and their examples go through
    `blocks_to_weight`.
- **Changed (tests).** At `b5a848d5`:
  - about 170 `weight=` lines in tests, some of them on the old kinds (deleted in P20);
  - 39 `LeastSquaresBatch(` calls;
  - 12 `check_weight`, 7 `check_data`, 20 `prepare_derivative_data`, 7 `derivative_weight_to_taylor_weight` and 73
    dual-jet converter calls;
  - the tests of stacked models that share a `y`, which gains size-1 `C` axes;
  - `test_jets`: the rejection and acceptance tables ruled as in §2, and the factorials;
  - `test_data_preparation`, `test_least_squares_oracle`, `test_fitting`, `test_optimizer_display`;
  - `test_dispatch` and `test_jax_absent`, which cover `data_metric` and `blocks_to_weight` (jit; numpy in, numpy
    out).
- **Left alone:**
  - the old kinds' `_apply_weight` / `_weight_matrix`, `*_derivatives_transpose` and `_from_sweep` (deleted in P20);
  - `t3_inner_product_corewise_*` (a T3 operand, not `Y`);
  - the engine's stack check (§2).
- **Docs.** P19, already the documentation slice, gains §1's typing, the data metric, the rule of §2 and Q7's recipe.
  This arc writes docstrings, plus M0's glossary and naming entries.

## 7. Slices

Each slice is design-first: a design note, the reviewer's critique, Claude's rulings, then the build. Its oracles are
independent of the code under test, and it ends with the full suite, the module doctests and the pages green. The law
tests are §9.3's; the types in comments are §9.1's.

| Slice | What changes | Tested against | Leaves green |
| --- | --- | --- | --- |
| **M0. The notation** (mechanical) | `J -> P` and `J_in_i -> F_in,i` (below); the glossary and `naming_conventions.md` entries | **gate: the full suite green with every number unchanged**; a string or `GroupLengths` still written with `J` raises, beside the retired `T` in `TestGroupLengths` (`test_jet_contract.py:827`) | everything, bit for bit |
| **M1. The new functions** (`jets`; additive) | `data_metric` and its private core; the engine's `_with_K`; `blocks_to_weight`; `LeastSquaresBatch.data_metric`; 9.2 (the frontend converters and O-20's check removed; the backend converters retyped); the law helper `tests/_typing.py` with §9.3's tables; `signature_style.md`'s grammar (9.1), in which its own new signatures are written | arrays written by hand per case, with `F`, `W`, `C` sizes **distinct and equal**; a `K`-stacked `r` raises when a weight is given (and gives `γ r` with `weight=None`); the laws (`M` maps the `P` law to the `P*` law with `ω` moved by its own) | everything; no caller switched yet |
| **M2. The switch to the rule** (one change; it may be built in parts on a branch, as P16 was) | the checkers with `stack_shape`; the engine (`r_step`, `f_step`, `Jp_norm_sq_step`, `w_tilde_step`, `_modes`, the masks for `y` and `ω`); the reference module, extended **independently** (its `_weight_squared`, `:655-697`; the independence test forbids importing `jets`); `jet_to_dual_jet` / `dual_jet_to_jet` through `data_metric`; the factorials and `prepare_derivative_data`, with Q7's docs; `check_batch`, `take`, `pack_batch`, `f_resolution`; stacked `y`; the translation, the bridge and the pages through `blocks_to_weight`; `tests/oracle_sweeps/oracle.py` extended with full-form weights by its own alignment; every test that passes a weight or a stacked `y` | the reference over the jet matrix (`f`, `g`, `JtJp`, `Jp_norm_sq`, `Kp`) with full-form `ω`, `y` with `C`, and `K`; the dense oracle on per-sample-weighted fits and an entries per-sample fit; **a sample with `ω = 0` gives the `f` and `g` of the batch without it**; **per-element stacked data give the separate fits**; unbiasedness by exact subset enumeration with a partly broadcast weight; garbage-padded `y` and `ω`; uniform against ragged; the factorials by hand with equal and distinct sizes; the ruled rejection table; the compile test; the laws on the engine; **every existing weighted test, its block weight through `blocks_to_weight`, reproduces its pre-M2 numbers to rounding**; **the reference module's full-form reading is built first and pinned against brute-force dense sums before the engine switches** (the two change in one slice, so their errors could correlate) | everything; the translation's 19 pins unchanged |
| **M3. The display** | the weighted ratio; the weighted block sums (engine, reference, oracle); the block-shape text | block-constant weights reproduce today's pinned display numbers; a zero-weighted outlier does not dominate; the reference | the optimizers and the display records |
| **M4. The typed transposes** | the rename and the docstrings of §5 (mechanical; 42 functions, 140 call sites); the example comments | the adjoint doctests renamed; on a **drawn batch**, `<J^T(batch.data_metric(r)), ẋ>` against a central difference of the dense misfit `½ γ Σ ω² r²` (`oracle.py`) along a random core direction; the laws on all 42 | everything |
| **M5. The types** (comments only) | §9.1's types in the ~100 public sampling signatures, backend and frontend; the type column of `sampling_variables.py`'s variable table | the laws on the functions no earlier slice tested (forwards, tangents, dense functions, `jets` helpers) | everything; no line of code changes |

**Why M2 is one change.** A weight, `y` and their consumers (engine, checkers, `take`, packing, the translation) must
agree on one representation. If one switches before the others, there is a silent wrong answer in between: a full-form
weight passed through an old `take` keeps the full batch's samples on a drawn batch. M1 is additive, so M2's switch
starts from tested functions.

**M0, the rename, in full.** It is pure and numerically neutral, and it goes first so that M1 and M2 start on the
renamed tree. Nothing it touches is released: none of it is in `v2026.5.0`. The counts are from an AST scan of the code
at `ffa4ad58`; docstrings and doctests change too, but are not counted.

| What | Count | Where |
| --- | --- | --- |
| the grammar: the parser's `'J'` | 5 lines | `contractions.py:578-629` |
| grouped string literals with `J` | 202 | `sampling_steps.py` 93, `test_jet_contract.py` 55, `_jet_contract_cases.py` 38, `test_dispatch.py` 8, `test_contractions_sharding.py` 5, two more files 3 |
| `jet_contract(` calls carrying them | 153 | |
| `GroupLengths(J=…)` / `lens.J` | 25 / 9 | the field `J` becomes `P` |
| body-local names with a `J` group suffix (`mu_JWCa` 54, `w_JWo` 42, `nu_JWCb` 41, `xi_JWCi` 37, the `_tilde` and `_dot` forms, …) | 377 uses of 21 names | `sampling_steps.py` 358, `data_preparation.py` 14, the rest in `ut3_operations`, `fitting` and two test files. CLAUDE.md's rule: the suffix spells the grouped subscripts, so the names follow the strings |
| the keyword `len_J` | 33 lines, 8 files | a public backend keyword of `ut3_operations.pack_vectors` and `pack_if_ragged`, used by `uniform_fitting.pack_sample` and the engine. It becomes `len_P` |
| `'J'` literals in the test machinery that reads groups out of strings | 22 | `_jet_contract_cases` 9, `test_jet_contract` 8, `test_contractions_interpreter` 4, `test_contractions_sharding` 1 |
| the size name `J_in_i` → `F_in,i` (Q12) | 180 lines, 15 uses of a local `J_in` | 163 in the library (signature comments `J_in_i+W+(Ni,)`, 56 in `sampling_definitions`, the engine's variable table), 17 in tests; no doc page |

- **Stays `J`:** the Jacobian (`J`, `Jp`, `JtJp`, `held_Jp`).
- **Unaffected:** no doc page, example or CLAUDE.md names a `J` group, `jet_contract` or `GroupLengths` (checked). No
  string uses `P` or `Q` as a group today (the reviewer's AST scan).
- **The durable record**, not deferred to P19, since it records a decision (the routing rule):
  - The glossary's Symbols table **extends** its crowded `P` row with the jet group; it does not add a second one.
  - It adds rows for `Q`, `M` (with the manifold collision), `Y` / `Y*` (with `*` the algebraic dual), `γ`, `ℓ`
    and `[G]`.
  - Its `F` mentions note both meanings, the box sizes and the frame (`P_F`, `T(F)`).
  - Its preamble's reserved letters gain `P` and `Q`.
  - `naming_conventions.md`, "Index letters", gains `P` and `Q` beside `C`, `W` and `K`, and `[G]`.

**Types in one comments-only slice (M5).** M1 writes the grammar and its own new signatures in it, and every existing
signature is typed in M5:
- the notation is reviewed once, as a whole;
- M4 stays mechanical;
- the ~50 forwards, tangents and helpers that no other slice touches need a slice anyway.

The cost: M5 edits the comments on the 42 signature lines M4 renamed.

## 8. The decisions (the maintainer, 2026-11-22)

| Q | Decided | Where |
| --- | --- | --- |
| 1 | the public `M` is `data_metric` | §3 |
| 2 | the transposes' argument is `c` | §5 |
| 3 | `c = data_metric(r, weight, minibatch_scale)`: elementwise, no `kind` or `order`, both factors required (`None` written out allowed), `r` without `K` (raises by rank); plus `LeastSquaresBatch.data_metric(r)`; with the jets helpers, re-exported by `t3toolbox.fitting` | §3 |
| 4 | the display shows the weighted ratio `‖ω⊙r_b‖/‖ω⊙y_b‖` | §4, M3 |
| 5, 8 | one rule: every data-space object has the type of `z`, each axis 1 or `z`'s size, never `K`; the full form only, with a constructor from blocks; per-element data on a stack follow | §2, §4 |
| 6 | the input is `ω` (squared inside) | §2 |
| 7 | Taylor coordinates everywhere, one conversion at the boundary, the recipe for a data-relative weight; documented in `prepare_derivative_data`, `data_metric` and the pages | §1, §4 |
| 9 | the dual-jet converters are frontend-private and backend-public ("the backend user needs this to work with held edge variables") | §9.2 |
| 10 | the dual symbol is `*` ("these are algebraic duals") | §1, §9 |
| 11 | types in backend and frontend signatures ("arguably more important in the backend") | §9.1 |
| 12 | `J_in_i` becomes `F_in,i`; `J` means only the Jacobian | M0 |

**The one open question for the maintainer's final read: the constructor (§4).** It is a new public name.
- **(a) `blocks_to_weight(kind, order, blocks, y)`** (recommended). It follows the converter grammar `X_to_Y`, as in
  `derivatives_to_taylor`. It returns a full-form weight shaped like `y`: the blocks on the jet axes (and the modes),
  size 1 elsewhere, as a `d`-leading array for probe and one array otherwise. Taking `y` gives it the rank, with `C`,
  and no stack argument is needed.
- (b) `weight_from_blocks(kind, order, blocks, y)`: the same function, with the name read as a constructor.
- (c) `block_weight(...)`: shorter, but it reads as "the weight of a block", the old form it replaces.

**Proposed by the writer, to be put to the maintainer in the design note that builds each:**
- the checkers' keyword `stack_shape=()`, the library's existing name for `C` (M2);
- `prepare_derivative_data(weight=None)` still returning `k!`, now in full form (M2);
- the names of M3's new engine variable and oracle output, the weighted block sums (M3). Engine variable names are
  the maintainer's.

## 9. Lightweight typing (folded in by the maintainer, 2026-11-22)

The jet/dual-jet and vector/covector distinctions are traps (audit D3, D4; finding A of P13). This section adds three
things:
- types in the signature comments (9.1);
- one frontend representation of `Y` (9.2);
- transformation-law tests (9.3).

Two further ideas are carried to Part 2: types in `jet_contract`'s strings, and types on the declared graph checked at
import. The letters are decided (9.0; M0 records them), and M1's design note settles the comment grammar before any
slice writes a type.

### 9.0 The notation, and the dictionary with §1

The maintainer's notation (`TARGET_DECISIONS.md`, "Notation for the jets and the data metric", and Q10). The dual
symbol is `*`, the algebraic dual. Everything is per sample: the sample stack `W` is a direct sum, and `K` and `C` are
stacks of separate elements.

| Symbol | Meaning |
| --- | --- |
| `P` | the jets on the box `F` (Taylor coefficients, product truncated to `F`): both the ring and its group of axes, the letter of the jet axes in contraction strings (was `J`; M0) |
| `P_in,i` | the input jets of mode `i`, on their own box `F_in,i` |
| `Q` | the dual-jet group in the engine's turned form; a letter from Part 2's typed grammar on, not used in this arc's code |
| `P*` | the dual of `P`, in **slot coordinates**: `<c, a> = Σ_k c_k a_k` |
| `F` | the box sizes `(n_1 + 1, …, n_m + 1)`, the sizes of `P`'s axes for a full jet. Shape comments write `F`; strings write the group `P` |
| `F_in,i` | the jet sizes of the input jets `w[i]`, read clipped to `min(F_in,i, F)` (was `J_in_i`; Q12, M0) |
| `m`, `n_j`, `k` | the number of jet variables (`len(order)`), the jet orders (`order = (n_1, …, n_m)`), a slot's multi-index in the box |
| `k!` | `k_1!…k_m!`, a slot's factor between derivatives and Taylor coefficients (no letter of its own) |
| `ℝ` | the real numbers (`R` is the right store) |
| `N_i`, `e` | the mode sizes; the edge size of a probe data-space array, `e ∈ {1, N_i}` |
| `V_i = ℝ^{N_i}`, `V_i*` | mode `i`'s input space and its dual |
| `E = ⊗_i V_i*`, `E* = ⊗_i V_i` | the tensor (a multilinear form on the inputs), and CP factors |
| `X`, `X*` | the parameters and their dual (§1) |
| `Y*`, `Y` | the measurements (`J: X -> Y*`), and their predual: the table below |
| `ω` | the weight, entering squared, of the type of `z` (§2) |
| `γ` | the minibatch scale (O-21): 1 on a full batch; on a draw of `b` samples, the batch's number of samples over `b`. Not `s`, which names the jet variables `s_j` |
| `M: Y* -> Y` | the data metric: `γ · diag(ω²)` today, possibly a loss Hessian later. It shares its letter with the manifold `M` of the glossary, a collision the maintainer accepts |
| `R_X: X -> X*` | the parameter metric; `R_X^{-1}` is `geometry.project` |
| `J`, `Jp`, `JtJp` | the Jacobian and its products (unchanged; this is why the jet group is no longer `J`) |
| `W`, `K`, `C` | the sample stack, the direction stack, the frame stack (unchanged) |
| `ℓ_j`, `Δ_i` | the jet law's scale per jet variable (`rescale_direction`'s `length`), and the input law's diagonal per mode (9.3). Not `λ`, the regularizer's strength |
| `[G]` | in a family of shapes, the group `G` is present only for some variables or settings (structural presence). Every group may also be empty (zero axes), in `contract` and `jet_contract` alike |

The new letters `γ` and `ℓ` were checked against the code, the glossary and the pages. `γ` is free, and `ℓ` appears
only as a local subscript in one `.tex` note. The accepted collision is `M` with the manifold. `*` means multiplication
only in Python code (`γ * ω**2 * r`), and a wildcard in names (`tv_*`).

| Kind | `Y*`: `z`, `y`, `r`, `Jp` | `Y`: `c`, `M r` |
| --- | --- | --- |
| apply | `P` | `P*` |
| probe | `⊕_i P⊗V_i*` | `⊕_i P*⊗V_i` |
| entries | `ℝ` | `ℝ` |

Finite dimensions give `P** = P`. §1's star marks the measurements as covectors of an abstract `Y`, and `P` names the
concrete jet space. So for apply `Y* = P` and `Y = P*`.

**Which pairing is meant where.** `*` is always the algebraic dual with the slot pairing. The ring structure of `P`
(the truncated product) enters in two places only:
- **The probe type.** Probe is apply with mode `i` left open: `Hom_P(P⊗V_i, P) = P⊗V_i*`. The `Hom` is `P`-linear, and
  the star is still the dual of `V_i`.
- **The engine's turned form.** The engine represents `P*` inside `P`: slot `k` of a dual jet holds `c_(n-k)`, and
  `<c, a>` is the corner coefficient of the product. `w_tilde` and the tilde variables are `Y` in this form (the group
  `Q` of Part 2).

### 9.1 The type in the signature comment (Q11: backend and frontend)

The grammar puts the type first: `# P*⊗V_i, len=d, elm_shape=F+W+[K]+C+(Ni,)`. That changes `signature_style.md`'s
micro-grammar ("`len` first", `docs/contributor/signature_style.md:74-89`), a durable contributor rule. So the doc
changes in M1, the first slice that writes types, not in P19.

| Function(s), ragged and uniform, backend and frontend | Type |
| --- | --- |
| `*_apply` | `∏_j P_in,j⊗V_j -> P` |
| `*_probe` | `∏_j P_in,j⊗V_j -> ⊕_i P⊗V_i*` |
| `*_entries` | `index^d -> ℝ`. An index is a basis vector, not an element of `V_i` |
| `dense_*` | `E × inputs -> Y*`, linear in `E` |
| the tangents `tv_*`, `*_corewise_tangent` | `X × inputs -> Y*`, linear in `X` |
| the transposes `tv_*_transpose`, `*_corewise_transpose` | `inputs × Y -> X*`, linear in `Y` |
| the second-order transposes `*_corewise_hessian` | `inputs × Y × X -> X*`, bilinear |
| the ambient transposes `*_ambient_transpose` | `inputs × Y -> E*` (CP factors `c·⊗w`). Read as a tensor (`from_canonical`), `E*` is identified with `E` by Hilbert-Schmidt: the one silent metric on the parameter side (audit G) |
| `data_metric`, `LeastSquaresBatch.data_metric` | `Y* -> Y`, `M = γ·diag(ω²)` |
| `jet_to_dual_jet` / `dual_jet_to_jet` (backend) | `Y* -> Y` in turned coordinates, and back (9.2) |
| `blocks_to_weight` | block values `->` a weight of the type of `z` |
| the oracle's outputs | `r`, `Jp`: `Y*`. `g`, `JtJp`, `Kp`: `X*`, before the projection. `f`, `Jp_norm_sq`, `f_resolution`: `ℝ`. The block sums: `ℝ` per block |

- **Scope:** about 100 public signatures, backend and frontend.
  - the 42 transposes and second-order transposes;
  - about 30 forwards and tangents;
  - the 3 dense functions;
  - about 12 `jets` helpers;
  - `data_metric`;
  - the oracle.
- **The engine's types** go in a type column of the variable table in `sampling_variables.py`'s module docstring, a
  page users reach. They do not go in `dev/graded_messages/SAMPLE_VARIABLES.md`, which is under `dev/` (the routing
  rule).
  - `z`, `r` and `Jp` are `Y*`.
  - `w_tilde` and every tilde variable are `Y` in turned form.
  - `g`, `JtJp` and `Kp` are `X*`.

### 9.2 One representation of `Y` in the frontend (Q9: decided (a))

- **Frontend: slot coordinates only.** `jet_to_dual_jet` and `dual_jet_to_jet` leave `t3toolbox.fitting` and its
  `__all__`, and the frontend `dual_jet_to_jet` is deleted. No frontend function takes or returns a turned form.
  `data_metric` returns slot coordinates.
- **Backend: both stay public in `backend/jets`,** retyped as the converters between slot coordinates and the engine's
  turned form. Their docstrings say: "for `sampling_variables`' `w_tilde` and held tilde variables; never for a
  transpose (D3)".
  - `w_tilde` and every tilde variable are public engine variables in turned form. They can be wanted, and
    `HeldVariables` holds them, marked "turned" (`sampling_variables.py:899-905`; the module text `:222`).
  - `sampling_definitions` takes a turned `w_tilde=` (`:86-89`).
- **This supersedes O-20** (`TARGET_DECISIONS.md:358`). O-20 made the frontend `dual_jet_to_jet` raise in safe mode,
  while the backend divides by zero. With the frontend function gone, the backend's documented division by zero is the
  only behaviour.
- It settles the frontend part of decision 4. The backend names stay with decision 4.

**Blast radius**, in lines naming either function, at `c3f37d34`:

| Where | Lines | Fate |
| --- | --- | --- |
| `backend/jets.py` | 26 | stays; docstrings retyped |
| `fitting.py` | 18 | the export, the frontend function, its doctests, the module text (`:45-47`) go |
| `sampling_steps.py` | 2 | stays (M2 routes it through `data_metric`) |
| `test_jets` | 71 | `TestDualJet` stays; `TestFrontendDualJetToJet` (O-20) goes |
| `test_dispatch`, `test_jax_absent` | 16, 7 | the frontend uses move to the backend names |
| examples, `docs/` | 0 | |

The transposes' prose "turned inside (a dual jet)", about 70 lines, is retyped in M4 anyway.

### 9.3 Transformation-law tests

A test helper, `tests/_typing.py`, checks functions against the laws of their declared types. It works on raw backend
arrays and costs nothing at run time. The reviewer verified both laws on 8 functions at orders `()`, `(2,)` and
`(1, 1)`, to 1e-15 or better, with the faults below caught at `O(1)` (`DATA_METRIC_PLAN_REVIEW_r3.md`).

**The jet law**, `s_j -> ℓ_j s_j`, with `ℓ^k = ∏_j ℓ_j^(k_j)`:

| Object | Law |
| --- | --- |
| a `P` coefficient `k`: `z`, `y`, `r`, `Jp`; the input jets on `F_in` | `ℓ^k` |
| a `P*` slot `k`: `c`, `M r` | `ℓ^-k` |
| a turned `P*` slot `k`: `w_tilde` | `ℓ^-(n-k)` |
| `ω` at slot `k` | `ℓ^-k`, so `f` is invariant |
| `X`, `X*`, `V_i`, the point | unchanged |

- It catches a missing, double or one-axis flip, and a jet/dual-jet swap.
- It needs `ℓ_j ≠ 1`, distinct per variable, and some `n_j ≥ 1`. At `order = ()` it is vacuous.
- It does not catch a Taylor/derivative mix: `k! a_k` scales as `a_k` does.

**The input law**, `w_i -> Δ_i w_i` (`Δ_i` diagonal and positive, `(N_i,)`), with every represented value fixed:

| Object | Law |
| --- | --- |
| the input jets `w_i` | `Δ_i w_i` |
| `X` on the mode axis: the Tucker cores `B_i` and their directions; the down store `D_i`; the variations `V_i` | `· Δ_i^-1` |
| the TT cores `G`, the stores `U`, `L`, `R`, the variations `H`, and their directions | unchanged |
| `X*` on the mode axis: the blocks `g_B`, `K_B`, `Ṽ_i` | `· Δ_i` |
| `X*` TT blocks | unchanged |
| probe `z_i`, `y_i`, `r_i`, `Jp_i` (`P⊗V_i*`) | `Δ_i^-1 ·` |
| probe `c_i`, `M r_i` (`P*⊗V_i`) | `Δ_i ·` |
| apply `z`, `c` | unchanged |
| entries `z`; entries `c` (by duality) | `/ ∏_i Δ_i[n_i]`; `· ∏_i Δ_i[n_i]` |
| the ambient transposes' CP factors (`E*`) | `Δ_i ·` per factor |
| `ω` | the inverse of its `Y*` law, so `f` is invariant. Probe with a full edge: `ω_i Δ_i`. A weight with an edge of size 1 follows only for `Δ_i` a multiple of the identity, which is audit D6: such a weight makes a mode's rescaling change the loss |

- It catches treating `c` as `V_i*` inside library code (a probe residual handled where `c` is meant).
- It also catches wrong co- or contravariance in anything that changes `V_i` coordinates.
- **Caveats:**
  - It holds for the metric-free maps only: forwards, tangents, transposes, second-order transposes and `M`. The
    transformed frame is not orthonormal, and Hilbert-Schmidt on `E` is not invariant under a non-orthogonal `Δ`. So it
    does not hold for `MANIFOLD.project`, `retract`, `inner`, `frame(x)` or `t3svd`.
  - On a shared geometry, `Δ_i` is equal within each sharing group.
  - On the uniform layout, `Δ` is arbitrary on the padding, which the masks zero.

**What no scaling law catches:** the probe transpose called with the residual and the sample swapped.
- Both arguments are `V_i`-typed (`c ∈ P*⊗V_i`, `w ∈ P_in⊗V_i`), so the input law moves them alike.
- The swapped function at `ℓ` is the original at `1/ℓ`, so it passes the jet law exactly.
- A swapped call is a valid transpose of another problem (LOG). The right/wrong adjoint doctests stay the only guard.

**Both laws are homogeneity tests: necessary, not sufficient.** A wrong constant, sign or binomial factor passes them.
The dense oracles stay the primary tests.

**The library's own instances.**
- `rescale_coordinates` / `t3_unscale_coordinates` are the input law, with `Δ` = the scales.
- `rescale_direction` is the jet law, one `ℓ` per variable (O-40). Audit D8 is the jet law applied to `ω`:
  `rescale_direction` leaves `ω` fixed, so the loss changes.
- M1 adds law tests for them.

**Where the laws run:** M1 for `data_metric`, the rescalings and the new functions; M2 for the engine and the
converters; M4 for the 42 transposes; M5 for the rest (§7).
