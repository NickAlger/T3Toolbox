# P16 CI fix: `test_rounding_gate_at_float32`

2026-11-21. CI run 44988706632 failed `tests/test_dispatch.py::TestSwitchNewton::test_rounding_gate_at_float32` at its
floor assertion on both legs (current: 4.8e-4 relative off; floor: 7.3e-4), against a `1e-4` relative tolerance. It
passed on the laptop. Not a library bug: the test mixed the two routes D9 separates. Test-only fix.

## Cause

The switch's rounding gate floors each tracked row at
`floor = _SWITCH_ROUNDING · _TR_RESOLUTION_FACTOR · _objective_resolution(lm, reg_val)
       = 1000 · ε · (‖ω⊙y‖·‖ω⊙r‖ + |f|)`,
with `‖r‖` and `f` the **model's**: `_objective_resolution` queries the oracle's `f_resolution` at `lm.frame` with
`lm.held`, i.e. the **frame route** (`lm.misfit`).

The test rebuilt that floor by hand from `row['misfit']` and `row['objective']`. Since D9 those are the **point
route's** (`point_objective` at the raw cores, scored at the start and carried from each accepted trial). Same point
(the row's `x`), different route.

At float32 the two routes' residuals differ by an absolute `O(ε‖y‖)`, so their misfits differ by a relative
`O(2ε‖y‖/‖r‖)`. Here `ε‖y‖ ≈ 1.0e-4` (`‖y‖ ≈ 860`). On the next-to-last tracked row `‖r‖ ≈ 0.03`
(misfit ≈ 4.5e-4, floor ≈ 3.09e-3, the CI value), so the misfits can differ by up to a few 1e-3 and the hand floor by
half that. That is legitimate float32 variation of the *reported misfit*, not an error in the floor; the `1e-4`
tolerance just doesn't cover it. On the laptop the gap happened to be 6.6e-5. Other BLAS/CPUs round differently,
and CI landed at 4.8e-4 and 7.3e-4. On the last rows (misfit at the rounding level) the gap is O(1). The sibling
`test_exact_newton_cg_jit_compiles_the_cg_once_and_floors_at_float32` was fixed the same way in P16 part 2.

## Evidence

`evidence.py` (scratch: `/tmp/claude-1000/-home-nick-repos-T3Toolbox/352ad37b-5ac8-40a1-a326-a7fc3a369cb2/scratchpad/cifix/evidence.py`)
reruns the test's fit and records `info.lm.misfit` through the callback. Per tracked row it prints the recorded floor
against the hand formula on the point route (the old test) and on the frame route. The optional argument is a relative
perturbation of the start cores at float32 rounding level (`1e-7`), a stand-in for another machine's rounding.

| start | rows | worst \|floor/hand − 1\|, point route | worst, frame route |
|---|---|---|---|
| as in the test (laptop) | 11 | 6.6e-5 (row 9) | 7.7e-8 |
| perturbed, seed 1 | 12 | 0.20 (row 10); **5.9e-4 at row 9, floor 3.088e-3** (the CI picture) | 1.0e-7 |
| perturbed, seed 2 | 12 | 0.34 | 1.0e-7 |
| perturbed, seed 3 | 13 | 0.47 | 8.1e-8 |
| perturbed, seed 4 | 13 | 8.7e-3 | 1.1e-7 |

The frame route reproduces the recorded floor to ~1e-7 in every case; the point route misses by 1e-4 to 0.5 on the
last rows. Running the old test under these perturbed starts reproduces the CI failure (below).

## Fix (`tests/test_dispatch.py`, the test only)

- The run records `float(info.lm.misfit)` per row through `callback=`. The hand floor
  `1000·ε·(‖y‖·sqrt(2f) + |f|)` reads that `f` (the model's misfit; unregularized, so `f` is also the `|f|` term).
- The tolerance is derived from float32 rounding of the floor itself, given its inputs:
  - `‖y‖` is float64 in the test and float32 in the run. Rounding the data to float32 costs `≤ u` (`u = ε/2`). The
    float32 sum of `N` squares is off by `≤ (N−1)u + u` relative (recursive summation, worst case), and the square
    root halves that: `≤ (N/2 + 2)u`.
  - The float32 evaluation of `ε·(‖y‖·sqrt(2f) + |f|)` from those values: a few roundings, `≤ 8u`.
  - `tol = (N/2 + 16)·ε` covers both with room. `N = 600` here, so `tol ≈ 3.8e-5`; measured `≤ 1.2e-7`.
- The verdict margins (were ±1 %, sized for the old float64-vs-float32 `‖y‖` worry) are now `±tol`, the same bound.
- The switched-at check slices `history` (zipped with the model misfits) by row index, as before, and reads the
  model misfit there too.
- The docstring says which route the floor reads and why.

## Mutants (scratch copy of the tree, each against the fixed test)

Script: `mutants.sh` in the scratch folder above.

| mutant | fixed test |
|---|---|
| M1 floor without `_TR_RESOLUTION_FACTOR` (10× low) | killed (floor assertion, row 0) |
| M2 `_SWITCH_ROUNDING` 100 → 101 (1 % high) | killed (floor assertion) |
| M3 resolution ignores `f_resolution` (`ε\|f\|` only) | killed (floor assertion) |
| M5 `f_resolution` drops `\|f\|` | killed (floor assertion) |
| M7 floor 0.99× | killed (floor assertion) |
| M6 gate ignores the floor (always informative) | killed (verdict: `True is not false`) |
| M4 gate `min(e_gn, e_n) > floor` for `max` | survives here, as with the old test (this run has no row with one error each side of the floor). Killed by `test_optimizers.py::test_switch_sample` ("e_n at it: a hit"). |
| none | passes; also under six perturbed starts (seeds 1–6) |
| **old** test, perturbed starts seeds 1–4 | **fails, all four**, at the floor assertion with CI-sized gaps (1.8e-6, 8.0e-7, 3.3e-7, 4.7e-7 absolute on a 3.09e-3 floor). This is the CI failure, reproduced. |

## Runs

Laptop, `t3toolbox` env (py3.11, numpy 2.x, jax 0.10), `-n 2`:

| run | result |
|---|---|
| `tests/test_dispatch.py -k "SwitchNewton or float32"` | 6 passed, 22 subtests passed (95 s) |
| `tests/backend/test_optimizers.py` | 76 passed, 1 skipped, 354 subtests passed (101 s) |
| the fixed test alone | passed (20 s) |

Not run here: the full suite, the jax-0.4.30 / numpy-1.26 floor leg. CI will run both.
