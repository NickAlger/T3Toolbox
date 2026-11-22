# The data metric: a sub-arc between P16 and P17 (revised after review round 3 and the notation decision, 2026-11-22)

Status: Claude's first draft (2026-11-21), revised by the writer after review rounds 1 and 3 (the answers:
`DATA_METRIC_PLAN_RESPONSE_r1.md`; round 2 signed off §§1-8). §9 folds in the maintainer's typing ideas, and the
whole plan uses his notation (9.0: the jet group `P`, the data metric `M`). The maintainer reads it before any build.
Implements: `TARGET_DECISIONS.md`, "The data metric: decisions 1 and 2 of the sampling audit", with O-21.
Evidence: `SAMPLING_AUDIT.md`. `file:line` refers to `aaee0b04`.
Out of scope: decision 3 (corewise as point methods or `COREWISE` tangent operations); decision 4 except the
frontend status of the dual-jet converters (question 9): the names of the Hessians and of the backend converters; an
non-diagonal `M` (a covariance); nonlinear losses; a weight or a `y` carrying the frame stack `C` (question 8).

**In brief.** The transposes and second-order transposes take an element of `Y` (dual jets in slot coordinates), with
no weight inside, and their argument `residual` becomes `c`. One public function, proposed `apply_data_metric(kind,
order, r, weight, minibatch_scale)`, applies `M: Y' -> Y`; every place that applies `omega^2` goes through it. The
weight becomes elementwise: today's block form, or the full shape of `y`, read by an exact rule. Lightweight typing
(section 9): algebraic types in the signature comments, the dual-jet converters private in the frontend, and
transformation-law tests. M1 also renames the jet group `J` to `P` in `jet_contract`. Five slices, M1 to M5, before
P17. Eleven questions for the maintainer in section 8.

## 1. The mathematics, typed

| Symbol | Space or map | In the code |
| --- | --- | --- |
| `X` | parameters: a frame's variations `(V, H)`, or raw cores on the corewise point | directions `p`, `ẋ` |
| `X'` | its dual | the blocks the transposes return; `g`, `JtJp`, `Kp` before the projection |
| `Y'` | the measurement space (`J` lands here) | `z`, `y`, `r = z - y`, `Jp` |
| `Y` | its predual, `Y'' = Y` | the transposes' argument `c`; `M r` |
| `J: X -> Y'` | the tangent forwards (`J = Dz(x)`) | `tv_*`, `*_corewise_tangent` |
| `J^T: Y -> X'` | the metric-free transposes: `<J^T c, ẋ> = <c, J ẋ>` | `tv_*_transpose`, `*_corewise_transpose`, `*_ambient_transpose` |
| `M: Y' -> Y` | the data metric: `M = s · diag(ω²)` today, possibly a loss Hessian later | today inside the engine and the oracle only |
| `R_X: X -> X'` | the parameter metric; `R_X^{-1}` is `geometry.project` | unchanged here |

Concretely (§9.0): `Y' = P` and `Y = P'` for apply, `⊕_i P⊗V_i'` and `⊕_i P'⊗V_i` for probe, `R` for entries,
per sample.

**The pairing of `Y` with `Y'`** is slot by slot on Taylor coefficients: `<c, a> = Σ c·a` over the modes (probe),
the slots `k` of the box `F`, the samples `W` and the mode edge, one pairing per element of the stacks `K` and `C`.
It holds no metric. The flip in `jet_to_dual_jet` is how the engine represents this pairing (as the corner
coefficient of a jet product), not a metric (audit 3.0, verified to 1e-16).

**The objective and its outputs** (`f`, `g`, `JtJp`, `Jp_norm_sq`, `Kp` as the oracle returns them):

    f = ½ <M r, r>,   g = J^T M r,   JtJp = J^T M J p,   Jp_norm_sq = <M Jp, Jp>,   Kp = D(J^T c)[p] at c = M r.

**The minibatch scale is part of `M`** (O-21): `s = 1` on a full batch and `s = n/b` on a drawn one, so every
`M`-bearing output carries `s` and the `Y'` outputs (`r`, `Jp`) do not. Hence `J^T(M r)` equals the oracle's `g` on
every batch, drawn ones included. `f_resolution = eps · (√<M y, y> √(2f) + |f|)` with `s` in both `M` and `f`, which
is today's value (`fitting.py:933-944`). The engine never sees `s`; the oracle multiplies its outputs by it.

**The objective-free maps contain no `M`.** They are the forwards, tangents, transposes and second-order transposes
(42 public functions, §5). `M` is the objective's: one public function (§3), the oracle, the optimizers. The engine's
`weight` argument is the oracle's channel into the engine; no objective-free public function passes one.

**Coordinates.** The library's data coordinates are Taylor coefficients: it never produces or consumes a `Y'` or `Y`
element in other coordinates, and derivative data enter only through `prepare_derivative_data`, which converts the
data and the weight together. **A user who composes an objective in other data coordinates must convert `Y`
elements the contragredient way.** In derivative coordinates `r_d = D r_t` with `D = diag(k_1!…k_m!)`; the pairing
fixes `c_t = D c_d`, so a `Y` element converts by `taylor_to_derivatives`, the converter with the opposite name (audit
D4). The two hand-built examples do this, correctly (`examples/fit_hilbert_from_apply_derivatives.py:142-144`,
`..._flat.py:137-139`). Probe data under `rescale_coordinates` are the same case: they are covectors of `V_i'`, audit
D6.

## 2. The weight's reading rule (decision 2)

`ω` is given, entering squared (as today; question 6). With `m = len(order)`, `F = (n_1+1, …, n_m+1)`, `W` the sample
axes and `N_i` the mode sizes:

| Kind | Arrays of `y` | Each array of `y` | Block form, per array | Full form, per array |
| --- | --- | --- | --- | --- |
| probe | `d`: a tuple or list, or one `d`-leading packed array | `F+W+(N_i,)` (packed: `N`) | `F` | `F+W+(e,)`, `e ∈ {1, N_i}` (packed: `{1, N}`) |
| apply | 1 | `F+W` | `F` | `F+W` |
| entries | 1 | `W` | `()` | `W` |

1. **Per array of `y`.** The probe weight is per mode: a tuple or list of `d` arrays, or one array whose first axis
   is the mode, read along that axis (as `jets._arrays` reads a packed `y`). **The number of modes is exactly `d`**,
   never broadcast. The apply and entries weights are one array.
2. **Two forms, told apart by rank.** Each array has rank `m` (the block form) or the rank of `y`'s array (the full
   form). Any other rank raises. Where the two ranks coincide (apply or entries with `W = ()`), the two forms are the
   same array. The probe full form always carries the edge axis, so its rank is above `m`.
3. **Sizes.** Each axis is 1 or the size of the axis of `y` it stands for. The block form stands for `F`, the leading
   `m` axes. The full form stands for `y`'s axes one by one.
4. **Meaning.** `ω` multiplies the residual entries it broadcasts over. The block form is constant over the samples
   and the edge. Today's forms are the block form: probe `(d,)+F`, apply `F`, entries `()`, read as today.
5. **Stacks.** On a value carrying stacks (`r`: `F+W+C(+N_i)`; `Jp`, or a value with `K`: `F+W+K+C(+N_i)`), a block
   form gets size-1 axes after `F` (as today, `sampling_steps.py:970`). A full form gets `value.ndim - weight.ndim`
   size-1 axes after `W`: before the edge (probe) or at the end (apply, entries). No group length is passed.
6. `None` is ones, as today. The weight never carries `K` or `C`.

**What can be written.** A per-sample apply weight is `(1,)*m + W`; a per-sample probe weight constant over the edge
is `(1,)*m + W + (1,)` per mode. **The trap**, documented beside the transposed-square trap (`jets.py:1248-1251`): a
per-sample weight written as a bare `W` with `len(W) == m` reads as the block form, and passes the check only if the
sizes coincide. Then it weights slots, not samples.

**The rejection table under the rule** (`tests/test_jets.py:849-870`, `D = 3`, `W = (7,)`, `n = 3`). Four cases
become legal, all of them a constant weight: probe `(D, 1)` ("a jet dimension of 1"), apply `(1,)` at `(n,)`, apply
`(1,)` at `()` (a full form with `W` of size 1), and entries `(1,)`. The other fifteen still raise, each for one of three
reasons:
- the mode count is exact (probe with 1, `n + 1` or `D + 1` rows, or no mode axis);
- the rank is neither `m` nor the full rank;
- a size is neither 1 nor `y`'s (apply `(1, n+1)` and `(D, n+1)`; probe `(D, 3, 2)` at `(1, 2)`).

**This supersedes two decisions, by name.** 2.5 ("the exact shape only", `TARGET_DECISIONS.md:1326-1332`) and 6.7's
"broadcasting decision (exact shape only)" (`:2307-2317`). Decision 2 itself made the block forms broadcast cases.
**6.7's explicit point is kept:** "a mode dimension of 1 is not broadcast" (rule 1). The engine's structural stack
check (`TARGET_DECISIONS.md:758-761`) still excludes the weight (`sampling_variables.py:1097`). For the weight, a
size-1 axis is the intended meaning, not a silent wrong answer.

**`check_weight(kind, order, sample, weight)`** checks rules 1 to 3 against the shapes `y` would have, read from the
sample as `check_data` reads them. It takes the sample, not `y`, because a batch's `y` may be `None`; `d` comes from
the sample. Callers change: `check_batch` (`fitting.py:958`), `test_data_preparation.py:100-101`,
`test_dispatch.py:823`, `test_jax_absent.py:182`, `test_jets`.

## 3. The public `M` (decision 1)

```python
def apply_data_metric(                          # name: question 1
        kind:             str,                  # 'probe' | 'apply' | 'entries'
        order:            typ.Tuple[int, ...],  # the jet shape; entries: ()
        r:                typ.Any,              # in Y': of the type of z. probe len=d, elm_shape=F+W+[K]+C+(Ni,); apply F+W+[K]+C; entries W+[K]+C
        weight:           typ.Optional[typ.Any],# by §2; None = ones
        minibatch_scale:  typ.Optional[typ.Any],# s of O-21: n/b on a drawn batch; None = 1
) -> typ.Any:                                   # in Y: s * weight**2 * r, slot order unchanged, the shapes of r. probe: a tuple
```

- **Both `weight` and `minibatch_scale` are required**, with `None` written out, as `jet_to_dual_jet` already
  requires its weight (`jets.py:339`). Every factor of `M` is visible in the call (question 3). A batch's call reads
  `apply_data_metric(oracle.kind, oracle.order, r, batch.weight, batch.minibatch_scale)`.
- Stacks are placed by rule 5. A packed probe `r` is read mode by mode and returns a tuple, like every function of
  `jets`. It acts on what it is given: a packed `r`'s padding times the weight's.
- **Home: `backend/jets.py`**, beside `jet_to_dual_jet`, in its `__all__`. `backend/fitting.py` imports
  `sampling_variables` → `sampling_steps` → `jets` (`fitting.py:37`, `sampling_variables.py:320`), and `jets` imports
  only `contractions` and `common`. So the engine and the oracle can both call it, with no cycle. Re-exported by
  `t3toolbox.fitting` under the same name (as the other `jets` helpers are, `fitting.py:78-87`). There is one backend
  home; not also `backend/fitting.py`.
- **One implementation of `ω²`.** A private helper in `jets` gives `ω²` aligned to a value by §2 (rules 1, 2, 5),
  in the lower floating precision of the two (as `_squared_weights` does today). `apply_data_metric` is
  `s × helper × r`. Every other site goes through the helper:

| Site | Today | After |
| --- | --- | --- |
| `w_tilde_step` (`sampling_steps.py:988`) | `jet_to_dual_jet` | `flip(helper × r)`, through `jet_to_dual_jet` |
| `f_step` (`:1052-1053`) | own reshape of `ω_F` | `½ Σ r · helper × r`, keeping `C` |
| `Jp_norm_sq_step` (`:969-970`) | own reshape | `Σ Jp · helper × Jp`, keeping `K + C` |
| `_modes` (`:913-922`) | `asarray(weight)[i]` | per array by rule 1 (a ragged tuple cannot be `asarray`-ed) |
| `f_resolution` (`fitting.py:935`) | `ω² × y_block_sumsq` | `Σ y · helper × y` (§4) |
| `dual_jet_to_jet` (`jets.py:498-507`) | divide by `_squared_weights` | `M^{-1} ∘ flip`: flip, then divide by the helper |
| frontend zero check (`fitting.py:275`) | `_squared_weights` | deleted with the frontend `dual_jet_to_jet` (9.2, Q9 (a)); else the helper |

- **`jet_to_dual_jet` keeps its signature and behaviour,** implemented as `flip ∘ apply_data_metric(…, weight, None)`.
  `dual_jet_to_jet` is `M^{-1} ∘ flip`. Their frontend status is 9.2 (Q9); D3 (a dual jet passed to a transpose is
  flipped twice) and their backend names stay with decision 4. Only their weight argument widens to §2.
- The engine keeps its `weight` input. Its seed is `flip(M r)` with `s` left to the oracle, as today. The oracle
  multiplies its `M`-bearing outputs by `s`. Both are exact, since every scaled output is linear in `M`.

## 4. Through the fitting layer

**Derivative weights** (`_taylor_weight`, `jets.py:255-263`). Today the factorials are multiplied with numpy's
right-aligned broadcasting, so a full-form weight gets them on its last axes. Checked: an apply `ones((3, 3))` at
`(2,)` returns `[[1,1,2]]*3`, and it should be `[[1,1,1],[1,1,1],[2,2,2]]`. Under §2 the jet axes are the
**leading** `m` axes of every per-mode array, in both forms. So the conversion is `ω × factorials_F` reshaped to
`F + (1,)*(ndim - m)`, per array, which is `_scale_slots`'s own alignment. A block form with a size-1 jet axis comes
back with full `F` by that broadcast. The weight comes back in the form it was given (a `d`-leading array or a
tuple).
- `derivative_weight_to_taylor_weight(kind, order, weight)`: it gains `kind` (the module's `(kind, order, …)`
  order). Without `kind` it cannot tell a probe's mode axis from a jet axis. The module is unreleased, so this is a
  free break.
- `prepare_derivative_data`: the same conversion. With `weight=None` it still returns the block `k!` weight.

**Uniform padding.** A per-mode full-form probe weight needs packing for the uniform layout. Edges differ, so the
`d` arrays cannot be stacked as given.
- `pack_batch` packs it this way. Each axis of `F + W` is broadcast to its full size if any mode has it full. The
  edge is kept at 1 if every mode's edge is 1. Otherwise every edge is broadcast to `N_i` and padded to `N`. Then
  the arrays are stacked to `(d,)+F+W+(e,)`.
- A block form, `None`, an apply or entries weight, and an already packed weight pass through. Never pad a size-1
  edge: `pack_if_ragged` would put the weight on entry 0 only (the reviewer's run).
- The engine masks a probe weight on entry with the shape mask on its edge: `_UNIFORM_MASKS['weight'] =
  (('N', None),)` (`sampling_variables.py:1366`), applied only to a full-form weight whose edge has size `N`. A
  garbage-padded weight then cannot make `nan · 0 = nan` in `f` or `g`, and `pack_batch`'s padding value no longer
  matters.

**`take`** (`fitting.py:980-1021`). A full-form weight's `W` axes start at `m` (an apply or entries array, or a
ragged probe mode) or at `1 + m` (a `d`-leading probe array). In order:
1. If every `W` axis of the weight is 1, reshape them to one axis `(1,)`, with no gather.
2. Otherwise broadcast them to the batch's `W`, flatten, and gather `indices`, as for `y`.

A block form and `None` pass through. `b` is static, so this stays shape-static under jit.

**`f_resolution`**: `Σ y · helper × y`, from the unscaled batch, then `× s` as now. A packed probe `y` is unpacked
(as now, `fitting.py:928-929`), and a packed full-form weight with an `N` edge is unpacked with it.

**The display** (question 4). With block weights, `‖r_b‖/‖y_b‖` per block was also the weighted per-block error. A
block-constant `ω` cancels from the ratio (audit D9). With elementwise weights it is not: a sample dropped with
`ω = 0` (an outlier, missing data) still dominates the unweighted table while the fit ignores it.
- Recommended: (b) **report `‖ω⊙r_b‖/‖ω⊙y_b‖`**, from two new optional oracle outputs: the weighted, unscaled block
  sums of squares (names in M3's design).
- For every block-form weight with no zero block this equals today's numbers. A block whose weight is zero
  throughout shows `—`.
- `r_block_sumsq` and `y_block_sumsq` stay plain (O-21 unchanged), and the header's `unwt` figure stays `½ Σ
  r_block_sumsq`.
- Either way, the text "a block is an entry of the weight, so the errors have the weight's shape" becomes "the block
  shape `(d,)+F | F | ()`": `optimizer_display.py:29-30, 165, 186, 423, 584`; `data_preparation.py:137`;
  `sampling_steps.py:1068`.

**jit and sharing.** The weight stays a batch leaf (`fitting.py:782-788`). A new value does not recompile; a new
shape does (block to full, or size 1 to full), and P16's compile test gains that case. `LeastSquaresBatch.weight`'s
annotation widens to a tuple for a ragged probe full form, which is a pytree subtree. Shared factors do not interact:
the weight is on the data side, sharing on the parameter side.

**P17's translation is unchanged.** Its outputs are block forms (`optimizers.py:177-197`), read as today. Its 19 pins
stay. P17's `point_value_and_gradient` composes `J^T(M r)` with the scale.

## 5. The transposes typed (decision 1)

- **Renamed:** `residual` → `c` (question 2), in **42 functions**.
  - Backend, 21: `tv_*_transpose` 3, `utv_*_transpose` 3, `t3_*_corewise_transpose` 3, `ut3_*_corewise_transpose` 3,
    `t3_*_corewise_hessian` 3, `ut3_*_corewise_hessian` 3, `t3_*_ambient_transpose` 3.
  - Frontend, 21: TT 9, UTT 6, `T3Tangent` 3, `UT3Tangent` 3.
  - About 205 mentions in their bodies. No `weight` parameter is added.
- **Documented** as "an element of `Y`, paired with `z` slot by slot (§1); for the gradient of a weighted
  least-squares fit pass `apply_data_metric(kind, order, r, weight, minibatch_scale)`."
- **Prose retyped:** every "the Gauss-Newton `J^T r`" / "the summed gradient `J^T r`" becomes `J^T c`, with `J^T(M r)`
  shown (`manifold.py:768, 916`; `uniform_manifold.py:643`; `tucker_tensor_train.py:3837, 4003`;
  `uniform_tucker_tensor_train.py:498`). So does the residual wording around them (`manifold.py:760` "the residuals
  live in the forward probe space"; `sampling_variables.py:121-122, 872`).
- **Call sites:** 123 `residual=` keyword calls in tests (`test_uniform_probing` 26, `test_frontend_sampling` 26,
  `test_dispatch` 24, `test_tucker_tensor_train` 20, `oracle_sweeps/o1_ragged` 9, `o1_uniform` 7, `test_manifold` 6,
  `test_fitting` 5) and 17 in module doctests. Positional calls are unaffected. P13's right/wrong adjoint doctests stay,
  renamed.
- **Coordinates (D4)**, one sentence each in `apply_data_metric`, `derivatives_to_taylor` and `taylor_to_derivatives`: a
  `Y` element converts the contragredient way (§1). The two examples keep their code, which is correct, and gain a
  comment citing the rule (question 7).

## 6. Also changed, and left alone

- **Changed (docstrings and comments):**
  - `jets.py` module text (`:51-56`, "no broadcast"), `check_weight`, `jet_to_dual_jet`, `dual_jet_to_jet`;
  - the engine's `weight` docs (`sampling_variables.py:1691-1692`) and the step comments;
  - `LeastSquaresBatch` (`fitting.py:764, 772`);
  - the frontend `fitting` module text (`:45-47`);
  - `data_preparation` (`data_block_norms` and `equalizing_weight` stay valid block-form producers).
- **Changed (tests):** `test_jets` (the rejection table ruled as §2; the factorial conversions), `test_data_preparation`,
  `test_least_squares_oracle`, `test_fitting`, `test_optimizer_display`. `test_dispatch` and `test_jax_absent` cover
  `apply_data_metric` (jit; numpy in, numpy out).
- **Left alone:**
  - the old kinds' `_apply_weight` / `_weight_matrix`, `*_derivatives_transpose` and `_from_sweep` (deleted in P20);
  - the bridge (block forms unchanged);
  - `t3_inner_product_corewise_*` (a T3 operand, not `Y`);
  - the translation (§4);
  - the engine's stack check (§2).
- **Docs:** P19, already the documentation slice, gains §1's typing and the data metric. This arc writes docstrings
  only.

## 7. Slices

Each slice is design-first (a design note, the reviewer's critique, Claude's rulings, then the build). Its oracles are
independent of the code under test, and it ends with the full suite, the module doctests and the pages green. The
law tests are §9.3's; types in comments are §9.1's.

| Slice | What changes | Tested against | Leaves green |
| --- | --- | --- | --- |
| **M1. The rule, `M` and the notation** (`jets`, `contractions`) | the `ω²` helper; `apply_data_metric`; `check_weight(kind, order, sample, weight)`; `jet_to_dual_jet` / `dual_jet_to_jet` through the helper; the factorial conversions per §4, with `kind`; D4 docstrings; 9.2 (the frontend converters and O-20's check removed; the backend converters retyped); the law helper `tests/_typing.py` with §9.3's full tables; `signature_style.md`'s grammar (9.1), and its own new signatures written in it; the notation work below (`J -> P`, the glossary, `naming_conventions.md`) | arrays written by hand per case, with `F` and `W` sizes **distinct and equal** (equal sizes hide the factorial bug), `K` and `C` non-empty; the ruled rejection table; `jet_to_dual_jet = flip ∘ M` and its inverse; **the laws**: `M` maps the `P` law to the `P'` law with `ω` moved by its own, and the converters map `P'` to the turned law; `rescale_coordinates` / `t3_unscale_coordinates` and `rescale_direction` as law instances; an old `J` string raises | everything: block forms read as before, the 19 translation pins unchanged |
| **M2. The engine and the reference** | `f_step`, `Jp_norm_sq_step`, `_modes`, the seed through the helper; `_UNIFORM_MASKS['weight']`; `sampling_definitions` extended to §2 **independently** (its `_weight_squared`, `:655-697`; the independence test forbids importing `jets`) | the reference over the jet matrix: `f`, `g`, `JtJp`, `Jp_norm_sq`, `Kp` with full-form weights, `K` and `C`; uniform against ragged; a garbage-padded weight; **a sample with `ω = 0` gives the `f` and `g` of the batch without it**; the laws on the seed, `g`, `JtJp`, `Kp` | the engine's existing by-definition tests (block weights) |
| **M3. The fitting layer** | `LeastSquaresBatch` docs, `check_batch`, `take`, `pack_batch`, `f_resolution`, the display (as ruled), `data_preparation` docs; `tests/oracle_sweeps/oracle.py` extended with a full-form weight written with its own alignment (not `jets`) | the extended dense oracle: per-sample-weighted fits; unbiasedness by exact subset enumeration with a **partly broadcast** per-sample weight; an entries fit with per-sample weights; the compile test (a new weight shape recompiles, a new value does not); the display pinned on block weights | the optimizers, the translation, the slow tier |
| **M4. The typed transposes** | the rename and the docstrings of §5 (mechanical; 42 functions, 140 call sites); the example comments | the adjoint doctests renamed; on a **drawn batch**, `<J^T(M r), ẋ>` against a central difference of the dense misfit `½ s Σ ω² r²` (`oracle.py`) along a random core direction; the laws on all 42 | everything |
| **M5. The types** (comments only) | §9.1's types in the ~100 public sampling signatures, backend and frontend; the type column of `sampling_variables.py`'s variable table | the laws on the functions no earlier slice tested (forwards, tangents, dense functions, `jets` helpers) | everything; no line of code changes |

**M1's notation work** (the maintainer's notation decision). `jet_contract` and `GroupLengths` are unreleased (not in
`v2026.5.0`), so this is a free rename.
- **The jet group `J` becomes `P` in `jet_contract`'s grammar.** The parser's `'J'` literals (`contractions.py:578-629`,
  5 lines) and the docstrings change.
- **The literal strings.** About 216 grouped strings contain `J`, in 173 `jet_contract(` call lines:

  | File | Strings |
  | --- | --- |
  | `sampling_steps.py` | 94 |
  | `test_jet_contract.py` | 56 |
  | `_jet_contract_cases.py` | 38 |
  | `contractions.py` | 11 |
  | `test_dispatch.py` | 8 |
  | `test_contractions_sharding.py` | 5 |
  | `test_sampling_variables.py` | 3 |
  | `test_sampling_apply.py` | 1 |

- **`GroupLengths`.** The field `J` becomes `P`. That is 32 constructions with `J=`, and 14 reads of `lens.J`
  (`contractions.py` 7, `sampling_steps.py` 3, tests 4).
- **Prose** that names the group `J` in the module texts (`contractions`, `sampling_steps`, `sampling_variables`).
- **Tested by:**
  - the suite, renamed and green with unchanged numbers (a pure rename);
  - a string still written with `J`, which must raise. Today an unknown group already raises: "group(s) [...] are
    not groups of a jet_contract string", checked.
- **The durable record.** It is not deferred to P19: it records a decision (the routing rule).
  - `docs/glossary.md`'s Symbols table gains `P`, `Q`, `F`, `M` (with the manifold collision), `Y` / `Y'` and
    `[G]`.
  - `docs/naming_conventions.md`, "Index letters", gains the group letters `P` and `Q` beside `C`, `W` and `K`,
    and `[G]`.
- No doc page or example names `jet_contract` or `GroupLengths` (checked).

**Types in one comments-only slice (M5), not per slice.** M1 writes the grammar and its own new signatures in it.
Every existing signature is typed in M5:
- the notation is reviewed once, as a whole, and a change to it after M1 touches one slice, not four;
- M4 stays mechanical (a rename and prose);
- the ~50 forwards, tangents and helpers no other slice touches would need a slice of their own anyway.

The cost: M5 edits the comments of the 42 signature lines M4 renamed. The alternative is per slice, with M4 also
typing the untouched forwards, which would roughly double M4.

## 8. Questions for the maintainer

1. **The name of the public `M`.**
   - (a) `apply_data_metric` (recommended). It names the operator as you named it, "the data metric". It stays right
     where a weight is zero or `M` becomes a loss Hessian. It is verb-object, like its neighbours
     `prepare_derivative_data` and `rescale_direction`. Its one blemish: "apply" is also a sampling kind, so
     `apply_data_metric('apply', …)` says "apply" twice.
   - (b) `data_riesz_map`, the earlier recommendation, from your "`W: Y' -> Y` is the Riesz map". A Riesz map is an
     isomorphism, which `M` is not where a weight is zero (audit D5), nor as a loss Hessian.
   - (c) `data_metric`: the noun, but it reads as returning `M`, not `M r`.
   - (d) `weigh`: short, and says less.
2. **The name of the transposes' argument.** `c` (recommended: the pre-N16 apply/entries name, the `phi_c` of O-22,
   already the Hessians' docstring name). Not `covector`: under your typing the argument is in `Y`, and the covectors
   are the measurements in `Y'`. Not `dual`: it collides with "dual jet" (D3).
3. **`M`'s signature.** (a) Arrays, with `weight` and `minibatch_scale` both required (recommended: every factor
   visible, the convention `jet_to_dual_jet` already follows, usable with no batch). (b) The batch,
   `(kind, order, r, batch)`. It cannot forget the scale, but it ties a `jets` function to the fitting layer's record
   type, which it cannot import. (c) A keyword `minibatch_scale=None`, the silent identity your principle names.
4. **The display.** (b) `‖ω⊙r_b‖/‖ω⊙y_b‖` (recommended; today's numbers for every block weight; a dropped sample
   stops dominating). (a) Keep the unweighted ratio and document that it no longer reflects the fit.
5. **The reading rule** (§2): the block and full forms, told apart by rank, superseding 2.5 / 6.7's exact shape while
   keeping the mode count exact (recommended: every existing weight stays valid, with no conversion). Or: the full form
   only, block forms converted once at the boundary (one form inside, but a break for every caller).
6. **`ω` or `ω²` as the input.** `ω` (recommended). `equalizing_weight` returns `1/norm` (`data_preparation.py:203`),
   `prepare_derivative_data` multiplies `ω` by `k!` (not `(k!)²`), a noise level is `ω = 1/σ`, a zero drops a block,
   and the display, the docs and P16's pinned translation all speak `ω`.
7. **Derivative coordinates (D4).** Document the contragredient rule in `apply_data_metric` and the two converters, and
   comment the two examples (recommended: the converter exists, `taylor_to_derivatives`, so one sentence is the whole
   fix). Or add a converter for `Y` elements (a second name for the same arithmetic; its name belongs with decision 4).
8. **A weight or `y` carrying `C`** (a stacked model with data per element, narrowed in P16, LOG:1240). Out of scope
   here, recommended: a CHANGELOG line, and a later arc if wanted. §2's rule extends to it by one more stack position.
9. **Where the dual-jet converters are public** (9.2).
   - (a) **Frontend-private, backend-public,** retyped as the converters to and from the engine's turned form
     (recommended: the razor, since `w_tilde` and the tilde variables are public in the backend).
   - (b) Private everywhere, with `w_tilde` also made engine-internal: a larger change, to the engine and the reference
     module.
   - (c) Unchanged, with decision 4 left open.

   (a) and (b) both supersede O-20.
10. **One dual symbol.** `'` (recommended: yours, as in `J: X -> Y'`), with `P'`, `V_i'`, `E'`, and the dictionary of
    9.0 tying `Y'`/`Y` to `P`/`P'`. Or `*` throughout, with §1 rewritten to match.
11. **The scope of 9.1.** Backend and frontend (recommended: backend users read the backend signatures, the razor's
    first reader). Or the frontend only (about half the comments).

## 9. Lightweight typing (folded in by the maintainer, 2026-11-22)

The jet/dual-jet and vector/covector distinctions are traps (audit D3, D4; finding A of P13). This section adds three
things:
- types in the signature comments (9.1);
- one frontend representation of `Y` (9.2);
- transformation-law tests (9.3).

Two further ideas are carried to Part 2: types in `jet_contract`'s strings, and types on the declared graph checked
at import. M1's design note settles the notation and the comment grammar before any slice writes a type.

### 9.0 The notation, and the dictionary with §1

The maintainer's notation (`TARGET_DECISIONS.md`, "Notation for the jets and the data metric", 2026-11-22). There is
one dual symbol, `'`, always the real dual (question 10). Everything is per sample: the sample stack `W` is a direct
sum, and `K` and `C` are stacks of separate elements.

| Symbol | Meaning |
| --- | --- |
| `P` | the jets on the box `F` (Taylor coefficients, product truncated to `F`): both the ring and its group of axes, the letter of the jet axes in contraction strings (was `J`; M1) |
| `P_in,i` | the input jets of mode `i`, on their own box `J_in_i` |
| `Q` | the dual-jet group in the engine's turned form; a letter from Part 2's typed grammar on, not used in this arc's code |
| `P'` | the dual of `P`, in **slot coordinates**: `<c, a> = Σ_k c_k a_k` |
| `F` | the box sizes `(n_1 + 1, …, n_m + 1)`, the sizes of `P`'s axes for a full jet. Shape comments write `F`; strings write the group `P` |
| `J_in_i` | the jet sizes of the input jets `w[i]` (read clipped to `min(J_in_i, F)`) |
| `V_i = R^{N_i}`, `V_i'` | mode `i`'s input space and its dual |
| `E = ⊗_i V_i'`, `E' = ⊗_i V_i` | the tensor (a multilinear form on the inputs), and CP factors |
| `X`, `X'` | the parameters and their dual (§1) |
| `Y'`, `Y` | the measurements (`J: X -> Y'`), and their predual: the table below |
| `M: Y' -> Y` | the data metric: `s · diag(ω²)` today, possibly a loss Hessian later. It shares its letter with the manifold `M` of the glossary, a collision the maintainer accepts (context tells them apart) |
| `R_X: X -> X'` | the parameter metric; `R_X^{-1}` is `geometry.project` |
| `J`, `Jp`, `JtJp` | the Jacobian and its products (unchanged; this is why the jet group is no longer `J`) |
| `W`, `K`, `C` | the sample stack, the direction stack, the frame stack (unchanged) |
| `[G]` | in a family of shapes, the group `G` is present only for some variables or settings (structural presence). Every group may also be empty (zero axes), in `contract` and `jet_contract` alike |

| Kind | `Y'`: `z`, `y`, `r`, `Jp` | `Y`: `c`, `M r` |
| --- | --- | --- |
| apply | `P` | `P'` |
| probe | `⊕_i P⊗V_i'` | `⊕_i P'⊗V_i` |
| entries | `R` | `R` |

Finite dimensions give `P'' = P`. §1's prime marks the measurements as covectors of an abstract `Y`, and `P` names the
concrete jet space. So for apply `Y' = P` and `Y = P'`.

**Which pairing is meant where.** `'` is always the real dual with the slot pairing. The ring structure of `P` (the
truncated product) enters in two places only:
- **The probe type.** Probe is apply with mode `i` left open: `Hom_P(P⊗V_i, P) = P⊗V_i'`. The `Hom` is
  `P`-linear, and the prime is still the real dual of `V_i`.
- **The engine's turned form.** The engine represents `P'` inside `P`: slot `k` of a dual jet holds `c_(n-k)`, and
  `<c, a>` is the corner coefficient of the product. `w_tilde` and the tilde variables are `Y` in this form (the group
  `Q` of Part 2).

### 9.1 The type in the signature comment

The grammar puts the type first: `# P'⊗V_i, len=d, elm_shape=F+W+[K]+C+(Ni,)`. That changes
`signature_style.md`'s micro-grammar ("`len` first", `docs/contributor/signature_style.md:74-89`), a durable
contributor rule. So the doc changes in M1, the first slice that writes types, not in P19.

| Function(s), ragged and uniform, backend and frontend | Type |
| --- | --- |
| `*_apply` | `∏_j P_in,j⊗V_j -> P` |
| `*_probe` | `∏_j P_in,j⊗V_j -> ⊕_i P⊗V_i'` |
| `*_entries` | `index^d -> R`. An index is a basis vector, not an element of `V_i` |
| `dense_*` | `E × inputs -> Y'`, linear in `E` |
| the tangents `tv_*`, `*_corewise_tangent` | `X × inputs -> Y'`, linear in `X` |
| the transposes `tv_*_transpose`, `*_corewise_transpose` | `inputs × Y -> X'`, linear in `Y` |
| the second-order transposes `*_corewise_hessian` | `inputs × Y × X -> X'`, bilinear |
| the ambient transposes `*_ambient_transpose` | `inputs × Y -> E'` (CP factors `c·⊗w`). Read as a tensor (`from_canonical`), `E'` is identified with `E` by Hilbert-Schmidt: the one silent metric on the parameter side (audit G) |
| `apply_data_metric` | `Y' -> Y`, `M = s·diag(ω²)` |
| `jet_to_dual_jet` / `dual_jet_to_jet` (backend) | `Y' -> Y` in turned coordinates, and back (9.2) |
| the oracle's outputs | `r`, `Jp`: `Y'`. `g`, `JtJp`, `Kp`: `X'`, before the projection. `f`, `Jp_norm_sq`, `f_resolution`: `R`. The block sums: `R` per block |

- **Scope** (question 11): about 100 public signatures, backend and frontend.
  - the 42 transposes and second-order transposes;
  - about 30 forwards and tangents;
  - the 3 dense functions;
  - about 12 `jets` helpers;
  - `M`;
  - the oracle.
- **The engine's types** go in a type column of the variable table in `sampling_variables.py`'s module docstring, a
  page users reach. Not in `dev/graded_messages/SAMPLE_VARIABLES.md`, which is under `dev/` (the routing rule).
  There, `z`, `r` and `Jp` are `Y'`; `w_tilde` and every tilde variable are `Y` in turned form; `g`, `JtJp` and `Kp`
  are `X'`.

### 9.2 One representation of `Y` in the frontend (question 9)

- **Frontend: slot coordinates only.** `jet_to_dual_jet` and `dual_jet_to_jet` leave `t3toolbox.fitting` and its
  `__all__`, and the frontend `dual_jet_to_jet` is deleted. No frontend function takes or returns a turned form.
  The public `M` returns slot coordinates.
- **Backend: both stay public in `backend/jets`,** retyped as the converters between slot coordinates and the
  engine's turned form. Their docstrings say: "for `sampling_variables`' `w_tilde` and held tilde variables; never for
  a transpose (D3)".
  - The reason: `w_tilde` and every tilde variable are public engine variables in turned form. They can be wanted,
    and `HeldVariables` holds them, marked "turned" (`sampling_variables.py:899-905`; the module text `:222`).
    `sampling_definitions` takes a turned `w_tilde=` (`:86-89`).
  - A backend user who holds or seeds one needs the converter, and a missing flip passes every shape check (the
    razor).
- **This supersedes O-20** (`TARGET_DECISIONS.md:358`): "raise in safe mode" in the frontend `dual_jet_to_jet`, the
  backend dividing by zero. With the frontend function gone, the backend's documented division by zero is the only
  behaviour.
- It settles the frontend part of decision 4. The backend names stay with decision 4.

**Blast radius**, in lines naming either function, at `c3f37d34`:

| Where | Lines | Fate |
| --- | --- | --- |
| `backend/jets.py` | 26 | stays; docstrings retyped |
| `fitting.py` | 18 | the export, the frontend function, its doctests, the module text (`:45-47`) go |
| `sampling_steps.py` | 2 | stays |
| `test_jets` | 71 | `TestDualJet` stays; `TestFrontendDualJetToJet` (O-20) goes |
| `test_dispatch`, `test_jax_absent` | 16, 7 | frontend uses go to the backend names |
| examples, `docs/` | 0 | |

The transposes' prose "turned inside (a dual jet)", about 70 lines, is retyped in M4 anyway.

### 9.3 Transformation-law tests

A test helper, `tests/_typing.py`, checks functions against the laws of their declared types. It works on raw backend
arrays and costs nothing at run time. The reviewer verified both laws on 8 functions at orders `()`, `(2,)` and
`(1, 1)`, to 1e-15 or better, with the faults below caught at `O(1)` (`DATA_METRIC_PLAN_REVIEW_r3.md`).

**The jet law**, `s_j -> λ_j s_j`, with `λ^k = ∏_j λ_j^(k_j)`:

| Object | Law |
| --- | --- |
| a `P` coefficient `k`: `z`, `y`, `r`, `Jp`; the input jets on `J_in` | `λ^k` |
| a `P'` slot `k`: `c`, `M r` | `λ^-k` |
| a turned `P'` slot `k`: `w_tilde` | `λ^-(n-k)` |
| `ω` at slot `k` | `λ^-k`, so `f` is invariant |
| `X`, `X'`, `V_i`, the point | unchanged |

- It catches a missing, double or one-axis flip, and a jet/dual-jet swap.
- It needs `λ_j ≠ 1`, distinct per variable, and some `n_j ≥ 1`. At `order = ()` it is vacuous.
- It does not catch a Taylor/derivative mix: `k! a_k` scales as `a_k` does.

**The input law**, `w_i -> Δ_i w_i` (`Δ_i` diagonal and positive, `(N_i,)`), with every represented value fixed:

| Object | Law |
| --- | --- |
| the input jets `w_i` | `Δ_i w_i` |
| `X` on the mode axis: the Tucker cores `B_i` and their directions; the down store `D_i`; the variations `V_i` | `· Δ_i^-1` |
| the TT cores `G`, the stores `U`, `L`, `R`, the variations `H`, and their directions | unchanged |
| `X'` on the mode axis: the blocks `g_B`, `K_B`, `Ṽ_i` | `· Δ_i` |
| `X'` TT blocks | unchanged |
| probe `z_i`, `y_i`, `r_i`, `Jp_i` (`P⊗V_i'`) | `Δ_i^-1 ·` |
| probe `c_i`, `M r_i` (`P'⊗V_i`) | `Δ_i ·` |
| apply `z`, `c` | unchanged |
| entries `z`; entries `c` (by duality) | `/ ∏_i Δ_i[n_i]`; `· ∏_i Δ_i[n_i]` |
| the ambient transposes' CP factors (`E'`) | `Δ_i ·` per factor |
| `ω` | the inverse of its `Y'` law, so `f` is invariant. Probe, full form with the edge: `ω_i Δ_i`. A block form follows only for `Δ_i = λ_i I`, which is audit D6: under a block weight, rescaling a mode changes the loss |

- It catches treating `c` as `V_i'` inside library code (a probe residual handled where `c` is meant).
- It also catches wrong co- or contravariance in anything that changes `V_i` coordinates.
- **Caveats:**
  - It holds for the metric-free maps only: forwards, tangents, transposes, second-order transposes and `M`. The
    transformed frame is not orthonormal, and Hilbert-Schmidt on `E` is not invariant under a non-orthogonal `Δ`. So
    it does not hold for `MANIFOLD.project`, `retract`, `inner`, `frame(x)` or `t3svd`.
  - On a shared geometry, `Δ_i` is equal within each sharing group.
  - On the uniform layout, `Δ` is arbitrary on the padding, which the masks zero.

**What no scaling law catches.** The probe transpose called with the residual and the sample swapped.
- Both arguments are `V_i`-typed (`c ∈ P'⊗V_i`, `w ∈ P_in⊗V_i`), so the input law moves them alike.
- The swapped function at `λ` is the original at `1/λ`, so it passes the jet law exactly.
- A swapped call is a valid transpose of another problem (LOG). The right/wrong adjoint doctests stay the only guard.

**Both laws are homogeneity tests: necessary, not sufficient.** A wrong constant, sign or binomial factor passes them.
The dense oracles stay the primary tests.

**The library's own instances.** `rescale_coordinates` / `t3_unscale_coordinates` are the input law with `Δ` = the
scales. `rescale_direction` is the jet law, one `λ` per variable (O-40). M1 adds law tests for them. Audit
D8 is the jet law applied to `ω`: `rescale_direction` leaves `ω` fixed, so the loss changes.

**Where the laws run:** M1 for `M`, the converters and the rescalings; M2 for the engine; M4 for the 42 transposes;
M5 for the rest (§7).
