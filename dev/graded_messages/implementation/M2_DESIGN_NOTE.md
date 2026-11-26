# M2 design note: the switch to the rule

Implementer, 2026-11-22, at `85c6e54d` on `m2-wip`. Read only. Sources: plan §2 to §4, §6, 9.3 and the M2 row; the
notes of 2026-11-22 ("Q5, Q7, Q8", "Q3", "Defaults, and the block sums"); `M_ARC_DECISIONS.md`; M0/M1 records; plan
reviews r6/r7. Scratch: `scratchpad/m2impl/`.

## 1. The parts, bottom up

The reference first (gate 2), then each layer from the engine up, so that every part is reviewable against layers
already switched. The engine goes before `jets`: once `w_tilde_step` seeds with `flip(data_metric(…))` it no longer
calls `jet_to_dual_jet`, so switching the converters later cannot break it.

**Part 1. The independent oracles** (suite green).
- `sampling_definitions`: `_weight_squared` reads the full form by its own alignment (equal rank, positional; a
  rank check that raises), with no import of `jets`. Docstrings of `weight` and `y`.
- `tests/oracle_sweeps/oracle.py`: additive `weighted_residual_full` / `misfit_full`, the weight expanded by
  `np.repeat` along its size-1 axes (no broadcasting), Taylor coordinates. The old-form path stays for the sweeps.
- Pins against brute force (`np.ndindex` loops over the dense `z`, `Jp`): `w_tilde`, `f`, `Jp_norm_sq`; `g` through
  `<g, p> = d/dt f` (central difference along one variation, exact since `z` is linear there); `JtJp` through
  `<JtJp, q> = Σ ω² Jp Jq`. Weights per slot, per sample, per entry, per mode; sizes of `F`, `W` distinct and equal.
  The ring-based checks in `test_sampling_definitions` gain the full form.
- Engine-against-reference tests keep the block for the engine and pass `blocks_to_weight(…)` to the reference: the
  old engine then pins the new reading too.
- The pre-M2 battery (§2) is recorded here, before any library code changes.

**Part 2. The engine** (`sampling_steps`, `sampling_variables`).
- `r_step` is `z - y`; `f_step`, `Jp_norm_sq_step` sum `r · data_metric(r, ω, None)` keeping `C` (`K + C`);
  `w_tilde_step` (and `_Jp.w_tilde`) is `flip(data_metric(value, ω_K, None))`; `_modes` pairs per array (a tuple
  or a `d`-leading `ω`); `_with_K` inserts `lens.K` size-1 axes after `F + W` (after the mode axis for a
  `d`-leading array).
- Uniform: a ragged probe `y` or `ω` packed on entry by the data-space rule of plan §4 (a private helper in
  `ut3_operations`, which `pack_data` adopts in part 4); masking per decision 2.
- The structural check of decision 1. Engine docstrings and table rows.
- Tests: every engine test passing a weight moves to the full form. Against the reference, over the jet matrix,
  `ω` in the patterns above with `C` full and of size 1, `y` with `C`, and `K` (reference looped per element);
  uniform against ragged; garbage (finite) in the padding of `y` and `ω`; the laws (jet: `f`, `g`, `JtJp`,
  `Jp_norm_sq`, `Kp` invariant, `r` by `P`, `w_tilde` by `P_star_turned`; input: probe and apply, `D` and `V` moved);
  `ω = 0` equals deletion and stacked equals separate, at the engine; the old-kernels bridge with `blocks_to_weight`.
- Expected red: everything reaching the engine through the oracle with a block weight or with a stacked `y` without
  `C` (`test_least_squares_oracle`, `backend/test_optimizers`, `test_uniform_fitting`, the new-form parts of
  `test_curvature`/`test_uniform_curvature`, `test_optimizer_display`, `test_fitting`, `test_optimizers_frontend`,
  `test_sharing`, `test_dispatch`'s fit rows, weighted doctests, pages, examples, the sweeps).

**Part 3. `jets`.**
- `check_data(kind, order, sample, y, stack_shape=())`, `check_weight(kind, order, sample, weight, stack_shape=())`:
  the rule against `z`'s shape read from the sample plus `C`; mode count exact. The rejection table re-ruled (2
  legal, 17 raise).
- The converters through `data_metric` (a pairing helper factored out of it, shared with `dual_jet_to_jet`'s
  division by `_omega_squared`); `_squared_weights` deleted.
- `derivative_weight_to_taylor_weight(kind, order, weight)` and `prepare_derivative_data`: factorials on the leading
  jet axes, a size-1 jet axis returned full, the form returned as given; `weight=None` gives `k!` through
  `blocks_to_weight`; Q7's text in the three docstrings.
- `check_batch`'s one call to the checkers updated (no stack yet).
- Tests: factorials by hand (sizes distinct and equal); M1's bitwise bridge re-pinned against the old block
  formula written out in the test (else it compares `data_metric` with itself); converter laws in full form;
  `test_dispatch`/`test_jax_absent` jets rows.
- Expected red: part 2's, and the batch checks of weighted fits now raising.

**Part 4. The backend fitting layer.**
- `check_batch(batch, shape, stack_shape=())`; `take` (`y` and `ω`: no gather when every `W` axis is 1, else
  broadcast, flatten, gather); `pack_data`/`pack_batch` for `ω` and `y`; `f_resolution` per decision 3;
  `LeastSquaresBatch` annotation and docs.
- Tests: the oracle-level named oracles (§3), `check_batch` tables, `take` against index loops, packing against
  loops with garbage, `f_resolution` against a dense `<y, M y>`; the backend tests of part 2's red list.
- Expected red: the frontend, pages, examples.

**Part 5. The frontend and the old forms.**
- `prepare_batch` passes `stack_shape`; the translation and `_old_draw` through `blocks_to_weight`; the pages'
  doctests (`fitting_and_optimization.md:426-433`, `numerical_contracts.md:93-95`); stacked-`y` tests gain size-1 `C`.
- The compile test gains a shape change (size 1 to full) that recompiles.
- The battery comparison and the full gate (suite, doctests, pages, examples, Sphinx, slow tier).
- Expected red: none.

## 2. The gate, with the reviewer's two strengthenings

1. **Pre-M2 numbers.** `implementation/m2/battery.py`, after M0's: one script run on a `git archive` of `85c6e54d`
   (`--form old`: block weights, shared `y` without `C`) and on the final tree (`--form new`: the same blocks through
   `blocks_to_weight`, `y` with size-1 `C`). It covers every weighted engine output and oracle output (incl.
   `f_resolution` and a drawn batch), three kinds, manifold/corewise, ragged/uniform, orders `()` to `(2, 1)`,
   `K`, `C`, `prepare_derivative_data`, short weighted `newton_cg`/`mc_sgd` (new form and translated), and one jitted
   run. Expected: **bitwise** for `w_tilde`, `g`, `JtJp`, `Kp`, `r` (M1 showed the seed bitwise); **to rounding**
   (rtol 1e-13) for `f`, `Jp_norm_sq`, `f_resolution` (`r·(ω²r)` replaces `ω²(r·r)`). Beside it: the `-v`
   node/subtest tally against a pre-M2 log, and a diff audit that every converted test changed only its weight and
   `y` arguments, never an assertion or a tolerance. The translation's 19 pins pass unchanged.
2. **The reference first.** Part 1 is reviewed and signed off before any engine line changes; its pins use no
   library code but dense functions.

## 3. The named oracles

| Oracle | Where |
| --- | --- |
| a sample with `ω = 0` gives the `f` and `g` (and `JtJp`) of the batch without it | engine (part 2), oracle and uniform (part 4); three kinds |
| per-element stacked data give the separate fits | `C = (2,)`, `y` and `ω` full in `C`, against unstacked queries per element: engine (2), oracle, local model, ragged and uniform (4, 5) |
| unbiasedness by exact subset enumeration | all `C(5, 2)` draws through `take`; mean of `f`, `g` equals the full batch's (dense `misfit_full`); `W = (2, 3)` with `ω` full on one axis only, and a `y` constant in `W` (part 4) |
| garbage padding | finite garbage in packed `y`, `ω`, and the sample (2, 4) |
| the laws on the engine | part 2 |
| dense oracle, per-sample-weighted fits and an entries per-sample fit | `misfit_full` against the local model's objective and `evaluate(t p)` (part 5) |

## 4. Decisions needed (plan corrections; nothing decided here)

1. **The engine checks `y` (and `ω`) structurally.** Plan §2 keeps them out of the stack check because "any other
   mismatch fails in broadcasting". That holds for a size, not for a rank: with `r_step`'s alignment gone, a `y`
   without `C` on a stacked model is right-aligned, so at `order = ()` with `W = C = (2,)` its `W` axis lands on
   `C` and the fit is silently wrong (the brief's first trap). Recommend: one static check at entry, rank equal
   to `z`'s, each axis 1 or the size the setting gives (`F`, `W`, `C`, `N_i`; uniform `N`), exact mode count. Cheap,
   structural, and it makes every unconverted stacked test fail loudly.
2. **Uniform: mask `y` unconditionally** (plan §4 makes it conditional on an `N` edge). `r = z - y` with an edge-1
   `y` puts `-y` on the padding of `r`, and `f` sums it. Checked on today's engine (`scratchpad/m2impl/edge1_mask.py`):
   an edge-1 packed `y` gives `f = 32256.365047732736` (ragged 32256.365047732732); with `'y'` dropped from the masks,
   `32271.13`. The mask of an edge-1 `y` broadcasts it to `N` with zero padding, which is right. `ω` can stay
   conditional (`ω² · 0 = 0`).
3. **`f_resolution`'s `<y, M y>` is over `z`'s shape.** The literal `Σ y · data_metric(y, ω, None)` raises when `ω` is
   fuller than `y` (data_metric never grows its argument: a shared `y` with a per-element `ω`), and undercounts an
   axis where both are 1 and `z` is not. Recommend broadcasting `y` to `r`'s shape (computed for `f`, and held) and
   summing over all but `C`. The same applies to M3's `y_block_sumsq`.
4. **`check_data` takes a `d`-leading array `y`?** §2 allows one; today's checker refuses arrays, and §3 says the
   checkers take the ragged form. `check_weight` must accept one (`blocks_to_weight` returns it). Recommend: accept a
   `d`-leading array in both iff every mode's slice obeys the rule (a padded edge then fails unless every `N_i = N`).
5. **`check_batch`'s protocol** (O-29's optional member) gains `stack_shape`; recommend the frontend always passes it,
   and the tests' user oracles follow.
6. **Block-shape wording.** M2 makes false "`r_block_sumsq` has the weight's shape" (engine table, `sampling_steps`,
   the reference, `data_block_norms`). Recommend fixing it in the files M2 touches; the display's text stays M3's.
