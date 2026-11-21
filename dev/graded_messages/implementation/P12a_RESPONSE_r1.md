# P12a, response to review round 1

Implementer, 2026-11-20, answering `P12a_REVIEW_r1.md` and the coordinator's rulings on it. Nothing is committed.

## 1. (Must fix) The core sizes of the directions — done

`_check_stacks` (`backend/sampling_variables.py`) now takes `variations` and `core_directions`, and checks the
trailing core axes of every direction and of `Vc` by tuple equality against the sizes the stores give
(`setting.sizes`):

| Operand | Expected core axes |
| --- | --- |
| `V[i]`, `Vc[i]` | `(nU_i, N_i)` |
| `H[i]` | `(rL_i, nD_i, rR_(i+1))` |
| `D_dot[i]` | `(nD_i, N_i)` |
| `L_dot[i]` | `(rL_i, nD_i, rL_(i+1))` |

- On a corewise point the sizes are the point's own (`U = R = L`). On the uniform layout they are the supercores'
  padded sizes; a supercore iterates over `d`, so the same loop serves both layouts.
- A count of cores other than `d` raises too.
- The error is a `ValueError` naming the operand, e.g. "every direction must have the core sizes of the frame:
  L_dot[1] has shape (1, 3, 2), where the stores give the core axes (2, 3, 2)".
- The `K + C` prefix stays `_read_K`'s.

Both of the reviewer's reproductions now raise, `L_dot[1]` at `rL = 1` against 2 and `V[0]` at `nU = 1` against 3,
and so do the `D_dot`, `H` and `V` cases in `scratchpad/p12/checks/razor2.py`. The point's own cores are not checked
against each other: that is the frontend's `validate()`, as before.

**Test:** `TestTheCoreSizesOfTheDirections` in `tests/test_sampling_functions.py`.
- Each operand alone (`D_dot`, `L_dot`, and on a frame `V`, `H`, `Vc`), every core, every core axis of size more
  than 1, cut to size 1.
- On a manifold frame and a corewise point, ragged and uniform, through the engine directly (the twins' merged
  signatures are P12b's).
- Each raise names the operand and its index.
- The good call is run first, and every operand has at least one case.

## 2. (Should fix) Transposes with a residual carrying `K` — done

`TestAResidualWithAStack` in `tests/test_sampling_functions.py` covers the three kinds, `tv_*_transpose` and
`t3_*_corewise_transpose`, at `K = (2,)` and `(2, 3)`, with `sum_over_samples` `True` and `False`. Per element `k`:
- the call with `K` equals the call on that element's residual;
- at `True`, the element's own adjoint identity, `<J p, c_k> = <p, (J^T c)_k>`.

The reviewer's three missed mutants (each `t3_*_corewise_transpose` rerouted to `g` from a held `r`) are now killed
by this file, each in a scratch copy, deleted after.

## 3. The nit and the record

- **The bridge guard's coverage:** the handover no longer claims "every old call of the file". It says what the guard
  wraps, and that some derivative-kernel calls run outside it (they do not reach the engine today). The bridge
  docstring never made the claim.
- **For P20:** the handover records that `test_uniform_curvature`'s plain direction-mask control now runs on the old
  `*_from_sweep` kernel, that P20 deletes it with that kernel, and that P11's `TestKpAtTTSlack` is the engine's
  control.

## 4. Runs

- **Current env:** the edited engine and sampling test files, curvature, uniform curvature, uniform probing and
  manifold pass (341 passed, 2 skipped, 11,029 subtests).
- **`tests/test_sampling_functions.py`:** 15 passed, 240 subtests.
- **The floor (`tttt`):** the functions, variables and uniform-sampling files pass (45 passed).
- **The full gate at `-n 2`:** all six steps PASS (suite 1816 passed, 90,237 subtests); lowest MemAvailable 2556 MB.
