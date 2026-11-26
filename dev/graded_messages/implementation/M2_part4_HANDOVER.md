# M2 part 4 handover: the backend fitting layer on the rule

Implementer (the third of M2), 2026-11-22, on `m2-wip` at `14029e05` (parts 1 to 3 committed). Built on
`M2_DESIGN_NOTE.md` (part 4), `M2_DESIGN_REVIEW.md`, decisions 13 to 17 of `M_ARC_DECISIONS.md` and the part-3 review.
Nothing committed.

**Required reading done** (BRIEF): `TuckerTensorTrain.probe` → `probing.t3_probe` → `sampling_variables.sampling_variables`
(`_closure_facts`, `_run`, `_site_body`/`_sweep_body`) → `sampling_steps` (`xi_step`, `mu_step`, `r_step`, `f_step`,
`w_tilde_step`, `_modes`, `_with_K`, `_weighted_sum`) → `jets.data_metric`; `TuckerTensorTrain.inner`/`norm` →
`t3_linalg.t3_inner_product`/`t3_norm`; `TuckerTensorTrain.t3svd` (its safe-mode sharing check) → `t3_svd.t3svd` and
`_t3svd_shared`.

**Scope.** The backend fitting layer reads `y` and the weight by the data-space rule (plan §2, §4): `f_resolution`
(decision 16: first), `check_batch` with `stack_shape`, `take`, `pack_data`/`pack_batch`; the backend tests of part 3's
red list. The frontend (`prepare_batch` passing `stack_shape`), the translation of the old forms, the pages, the
examples, the battery comparison and the full gate are part 5.

## 1. What changed

**`t3toolbox/backend/fitting.py`.**
- **`f_resolution`** (decision 13(c), 16): `eps * (sqrt(<y, M y>) * sqrt(2 f) + |f|)` with `<y, M y>` from the new private
  `_y_metric_y`: `y` broadcast to `z`'s shape (`F + W + C (+ (N_i,))`, `W` read off the sample, `C` the shape of `f`,
  `N_i` from `frame_shape`), then `sum y * data_metric(y, omega, None)` over everything but `C`; per element of `C`
  like `f`. A packed probe `y` or weight is unpacked first (`_arrays_of`: `unpack_vectors`, an edge of 1 stays 1).
  Unscaled, then `x gamma` once with the other scaled outputs (unchanged). The old block formula is gone;
  `y_block_sumsq` is unchanged (plain block sums: M3's), and it is now computed only when wanted.
- **`check_batch(batch, shape, stack_shape=())`**: passes `stack_shape` to `check_data` and `check_weight`. Doctest:
  a shared `y` with a size-1 `C` passes; a `C` axis of another size raises.
- **`take`**: `y` and the weight drawn alike by the new `_take_data_space` / `_take_W`: their `W` axes start at `m`
  (apply, entries, each array of a ragged probe object) or `1 + m` (a mode-leading probe array, packed or not); if
  every `W` axis is 1 they become one axis of 1 (no gather), else they are broadcast to the batch's `W`, flattened and
  gathered. Before, the weight passed through untouched (right for a block weight, a silent wrong answer for a full
  one: a per-sample weight would have kept all `n` samples against `b` drawn ones). A `W` part that is not `W` (each
  axis 1 or its size), or missing, raises a `ValueError` (see §5). New `_W` (the sample stack), `n_measurements` reads it.
  Doctest: a per-sample weight drawn with its samples; a weight constant over the samples keeps its axis of 1.
- `LeastSquaresBatch`: docstring (the type of `z` with `C`, per-element or shared data, `blocks_to_weight`) and the
  `weight` annotation widened to `typ.Any` (a tuple for a ragged probe). `LeastSquaresOracle`'s docstring: `f = 1/2 <M r,
  r>`, and the definition of `f_resolution`.

**`t3toolbox/backend/uniform_fitting.py`.** `pack_data` is the one rule for a probe data-space object: it calls
`ut3_operations._pack_data_space` (part 2's, which the engine already uses), so an edge of 1 is kept or broadcast,
never zero-padded (the old `pack_if_ragged` put an edge-1 value on entry 0 only). `pack_batch` packs the weight with
`pack_data` too (before: passed through). Docstrings and a doctest of the three edge cases.

**`t3toolbox/backend/optimizers.py`.** The `Oracle` protocol's docstring: `check_batch(batch, shape, stack_shape=())`,
which the frontend calls with the model's stack (correction (e); the frontend's call is part 5).

## 2. Tests

**New: `tests/test_oracle_data_space.py`** (10 tests, about 240 subtests; numpy only). Its anchors are independent of
the code under test: `expand` of `tests/oracle_sweeps/oracle.py` (`np.repeat`, no broadcasting), explicit index loops,
and `misfit_full` (dense).
- `TestCheckBatch`: `stack_shape` reaches both checkers. A shared `y` (size-1 `C`) and a per-element weight pass with it;
  then six cases must raise, three kinds each: a full-`C` `y` without it, a full-`C` weight without it, a stacked `y`
  against `C = ()`, a `y` and a weight with `C = 3` against `C = 2`, and an unstacked weight against `C = 2`.
- `TestTake`:
  - `y` and the weight against an explicit loop (each drawn element read at its unravelled `W` index, read at 0 on a
    size-1 axis). `W = (2, 3)`, the `W` patterns full on axis 0 only, on axis 1 only, all 1, and full; `y` full or
    constant over `W`; random size-1 axes elsewhere. Probe, apply and entries at `()` and `(1,)`; probe ragged and
    packed. Exact (`array_equal`, shapes included).
  - A mode-leading unpacked probe weight is drawn after its mode axis.
  - A block weight and a wrong `W` size raise at the draw.
- `TestPacking`:
  - `pack_batch` packs the weight by the rule of `y`, against a loop: the join of the modes' size-1 axes, an edge of 1
    kept, a full edge zero-padded, an edge of 1 broadcast to `N` when the edges are mixed. Cases: per entry, per slot,
    and mixed edges and patterns, with `y`'s patterns reversed against the weight's. The scale passes through, a
    packed object is kept (`assertIs`), and apply's and entries' are unchanged.
  - Finite garbage (`1e3`) in the packed padding of the sample, `y` and the weight changes nothing, uniform against
    ragged: `f`, `f_resolution`, `r_block_sumsq` and every real block of `g`, for three weight patterns.
- `TestFResolution`: against a dense `<y, M y>` per element of `C` (`y` and the weight expanded to `z`'s shape, summed
  by a loop over `C`). 120 subtests: three kinds, `()` and `(1,)`, `C = ()` and `(2,)`, four trials of random size-1
  patterns (one with `y` shared by `C` and constant over `W`, one unweighted), each on a manifold frame, a corewise
  point and the uniform point (packed). Tolerance `1e-12`.
- **The named oracles at the oracle**:
  - `TestAZeroWeightDeletesASample`: a per-sample weight with a zero at sample `j`, whose residual is checked to be
    nonzero there. `f`, `g` and `JtJp` equal those of the batch with `j` deleted (`1e-13`). Setting the weight at `j`
    to 1 adds exactly that sample's `1/2 |r_j|^2` to `f` (to `1e-9`), so the deletion is visible. Three kinds, `()`
    and `(1,)`, ragged frame and uniform frame (packed).
  - `TestStackedDataAreSeparateFits`: `C = (2,)`, with `y` (a stacked target's data) and the weight full in `C`, and
    different per element. The stacked query's `f`, `g`, `JtJp` and `f_resolution` equal the unstacked queries at each
    element's frame and data (`1e-13`). The other element's data give another `f`. Also uniform stacked point against
    ragged stacked point: `f`, `f_resolution`, the real blocks of `g`.
  - `TestUnbiasedOverEverySubset`: `W = (2, 3)`, the weight full on `W`'s axis 0 only (and per slot), `y` full or
    constant over `W`.
    - The full `f` equals `misfit_full` (dense).
    - `<g, p>` equals the dense central difference along a direction in one Tucker core. That difference is exact,
      since the misfit is quadratic along it.
    - Over all 15 subsets of 2 (`take`), the mean drawn `f` and `g` equal the full ones (`1e-13`).
    - Three kinds; probe and apply also along a line at `(1,)`.

**Converted** (only the form of `y` and the weight; no assertion or tolerance loosened):
- `tests/test_least_squares_oracle.py`:
  - The fixture's block weight is placed by `blocks_to_weight`; the random draws are unchanged.
  - The three `f_resolution` expectations now use a dense `<y, M y>` (`y_metric_y`: `expand`, a sum), in place of the
    old block formula. This is the definition change of decision 13(c), not a loosening: same tolerances.
  - `test_a_drawn_batch`'s `inner` expands the full-form weight.
  - The bridge to the old kinds uses the new `derivative_weight_to_taylor_weight(kind, order, weight)` on the placed
    weight.
- `tests/_fits.py` (the backend test helper):
  - `fit` and `derivative_fit` keep taking **one value per block** (the form the tests were written in) and place it
    with `blocks_to_weight` on `y` (`derivative_fit`: on the derivative data, before `prepare_derivative_data`).
  - New `shared_over(fit, C)`: size-1 `C` axes in `y` and the weight, for a stacked model with shared data.
  - New `at_element(fit, cc)`: the inverse, one element's data.
  - `C` is the last group before the probe edge, or the last group for apply and entries, in every form.
- `tests/backend/test_curvature.py`:
  - The four fixtures (`_pd_fit`, `_shared_fit`, `_plain_fit`, `_deriv_fit`) return `shared_over(fit, C)`.
  - The four per-element loops use `F.at_element(prob, cc)`.
  - `_w2r` multiplies by position (the weight is of the type of `z`).
- `tests/backend/test_uniform_curvature.py`: `_kind_setup` packs `shared_over(fit, (n,))` for a stacked uniform frame,
  while the per-element ragged models keep the fit as built.
- `tests/backend/test_optimizers.py`: the two stacked tests (`test_a_stacked_local_model_still_works`,
  `test_weighted_stacked_model_matches_the_per_element_models`) use `shared_over` for the stacked model.
- `tests/backend/test_optimizer_display.py`: `_batch` places its block weight with `blocks_to_weight`. The asserted
  ratios and the `(unwt …)` figure are unchanged (M3's).
- `tests/test_dispatch.py`: `test_jit_the_least_squares_oracle`'s weight is placed with `blocks_to_weight`.

## 3. Mutants

Script `scratchpad/m2impl3/mutants_p4.py`, on a scratch copy of `t3toolbox/`, `tests/` and `pyproject.toml` (deleted
after). Each mutant is applied alone and run against `test_oracle_data_space.py` and `test_least_squares_oracle.py` at
`-n 2`. Unmutated: 28 passed, 535 subtests. **14 of 14 killed.**

| Mutant | Killed by |
| --- | --- |
| `f_resolution`: `y` not broadcast to `z` (summed over its own shape) | `TestFResolution` (89) |
| `f_resolution`: summed over `C` too | `TestFResolution`, stacked = separate (71) |
| `f_resolution`: the weight ignored | `TestFResolution`, the converted oracle tests (113) |
| `f_resolution`: a packed weight not unpacked | `TestFResolution`, the garbage test, uniform tests (24) |
| `f_resolution`: the old block formula `weight**2 * y_block_sumsq` | 7 tests (115) |
| `take`: the weight passed through (the old `take`) | `TestTake`, the unbiasedness oracle (69) |
| `take`: always gathered, no axis of 1 kept | `TestTake` (35), by the drawn shape only: the numbers are the same (the plan's form, §5) |
| `take`: a mode-leading probe array read from `m` | `TestTake`, `TestTheBatch.test_take`, the drawn-batch tests (20) |
| `take`: gathered without broadcasting to `W` | `TestTake`, the unbiasedness oracle (39) |
| `take`: the type check dropped | `test_an_object_without_the_W_axes_raises` (2) |
| `pack_batch`: the weight not packed | both `TestPacking` tests (3) |
| `pack_data`: the old `pack_if_ragged` (an edge of 1 zero-padded) | `TestPacking`, `TestFResolution`, the zero-weight oracle, `TestTake` (8) |
| `check_batch`: `stack_shape` not passed to `check_weight` | `TestCheckBatch` (6) |
| `check_batch`: `stack_shape` not passed to `check_data` | `TestCheckBatch` (3) |

## 4. The runs

| Run | Result |
| --- | --- |
| `f_resolution` against a dense `<y, M y>`, by script before any test (`scratchpad/m2impl3/fres_check.py`): three kinds, `()`/`(2,)`, `C = ()`/`(2,)`, random size-1 patterns, frame, point and uniform | worst relative error 8.0e-16 |
| `test_least_squares_oracle.py` after conversion | 18 passed, 293 subtests |
| `test_oracle_data_space.py` | 10 passed, 242 subtests (one visibility check first written too strictly; rewritten to pin the exact term, §2) |
| module doctests of `backend/fitting`, `backend/uniform_fitting`, `backend/optimizers` | 11 passed |
| part 3's 123 red ids, after the library change and `_fits` (before the other conversions) | 43 red |
| the backend and dispatch ids among those 43, after the conversions | 2 red, then 1: `test_jit_the_least_squares_oracle` converted; `test_uniform_exact_newton_cg_use_jit` is the frontend translation |
| **the full suite, `pytest tests/ -v -n 2`** (`scratchpad/m2impl3/suite4_v.log`) | **69 failed, 1,940 passed, 5 skipped, 1 xfailed, 92,035 subtests passed, 36:15** (failed counts subtests); MemAvailable about 10.6 GB at the midpoint |

**The red, `M2_part4_EXPECTED_RED.txt`:** 23 test ids, each with its cause and route.
- Against part 3's 123: 23 still red, 0 new, 100 turned green (`m2/part4_green_since_part3.txt`).
- Every one of the 23 reaches an old form through the **frontend**:
  - `fitting.local_model` with a stacked `y` without `C`: 9;
  - `prepare_batch` → `check_batch` with a block weight: 2;
  - the translation in `t3toolbox/optimizers.py`: 9;
  - `_fits.old_model`, the old factories' shim: 1;
  - a frontend test's own `prepare_derivative_data` call: 1;
  - the optimizer sweep, through the translation: 1.
- No backend route is red. Part 3's `R` (the float32 `f_resolution` gate) is green.

**The tally against part 3's `-v` log** (`m0/tally.py`; kept as `m2/tally_part3_vs_part4.txt`): 110 nodes differ, the 100
red-to-green and the 10 new nodes of `test_oracle_data_space.py`. No file in `t3toolbox/` or `tests/` is newer than the
log. **Each of the 100 green nodes has exactly part 1's outcome and subtest count** (compared against
`scratchpad/m2impl/p1_suite_v.log`, the last all-green log), so no conversion dropped a comparison. Nothing is left
running.

## 5. Departures and choices

- **`take` raises on an object whose `W` part is not `W`** (missing, or an axis neither 1 nor its size). `take` is not a
  checker, but without this a block weight (no `W` axes) would be read with its jet axes in `W`'s place. On apply it
  would even be silently "repaired": a block `F` would become a legal per-slot `F + (1,)`, and a check would then pass
  on a drawn batch that refuses the full one. The check is structural, static and cheap. It is the house rule
  ("structural problems always error"), not a numerical precondition.
- **A drawn object constant over `W` keeps one axis of 1** (no gather), per plan §4 step 1. The gather would give the
  same numbers at `b` times the memory. `TestTake` pins the shape.
- **`f_resolution` broadcasts `y` to `z`'s shape** (decision 13(c)), with `z`'s shape read off the call: `F` from
  `order`, `W` from the sample, `C` as `f`'s shape (`f` is computed per element of `C`), `N_i` from `frame_shape`.
  The alternative, reading the shape off `r`, fails when `r` is held rather than computed. The memory is `z`'s size,
  which `r` already takes.
- **`_y_metric_y` and `_take_data_space` are private.** The backend user reaches them through `query('f_resolution')`
  and `take`, both public; there is no standalone capability to expose (the razor).
- **`pack_data` calls `ut3_operations._pack_data_space`**, a private function of another backend module. That is the
  design note's "a private helper … which `pack_data` adopts in part 4": one rule, one implementation, reached
  publicly through `pack_data`.
- **`_fits.fit` / `derivative_fit` keep the block vocabulary** and place it with `blocks_to_weight`. Every converted
  backend weight then goes "through `blocks_to_weight`" in one place, and about 25 call sites are unchanged, which
  keeps the diff audit small. A test that wants a per-sample or per-entry weight builds its `LeastSquaresBatch` itself,
  as `test_oracle_data_space.py` does.
- **`y_block_sumsq` and the display are untouched** (M3), except that `y_block_sumsq` is no longer computed when only
  `f_resolution` is wanted.

## 6. For part 5, and found outside the part

**Part 5's red list, by cause** (`M2_part4_EXPECTED_RED.txt`):
- `test_fitting.TestLocalModel` (8 tests) and `TestUniformBackendGeometryFromFrame`: a stacked model with one `y`
  without `C`, through `fitting.local_model`. The test's `y` needs size-1 `C` axes, the frontend's `prepare_batch`
  must pass `stack_shape` to `check_batch` (correction (e)), and the per-element references need the unstacked `y`.
  `F.shared_over` / `F.at_element` (now in `tests/_fits.py`) do exactly this.
- `_fits.old_model` (the old factories' shim, used by `test_the_old_model_shim_needs_one_y_over_the_stack`) still
  removes the stack from `y` (`_one_over_the_stack`). Under the rule it should keep size-1 `C` axes; its docstring says
  "the data cannot carry the point's stack", which is no longer true.
- The translation (`t3toolbox/optimizers.py:108`, `:196`) and `_old_draw`: the old block weights through
  `blocks_to_weight`. `old_vocabulary` in `tests/_fits.py` is the translation's independent oracle; it already goes
  through `fit` / `derivative_fit`, which now place blocks, so it may need no change.
- `test_fitting.TestResidualWeighting.test_plain_probe_per_mode` and `TestPerModeWeightRowCount`: a block weight into the
  frontend's `prepare_batch`.
- `test_optimizers_frontend.py:1587`: the test's own `prepare_derivative_data` call with a block weight.
- The sweep `TestO2OptimizerOracleSweep` goes through the translation.

**Also for part 5:**
- The compile test's shape change (plan §4): a weight going from size 1 to full on some axis recompiles; a new value
  does not.
- The battery `--compare` (the expectations of the design review's finding 2); the pages
  (`fitting_and_optimization.md:426-433`, `numerical_contracts.md:93-95`); the full gate with the slow tier.
- `y_block_sumsq` is still the plain block sum over `y`'s own shape, which for a `y` with size-1 axes undercounts against
  `r_block_sumsq`. That is M3's to fix (decision 13(c)), not part 5's.

**Found outside the part:**
- `t3toolbox/fitting.py:595` (`prepare_batch`'s docstring) still says `oracle.check_batch(batch, x.shape)`; it changes
  with the call in part 5. No doc page names the old signature (grep of `docs/*.md`).
- No library defect found.
