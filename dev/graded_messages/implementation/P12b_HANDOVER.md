# P12b: the uniform twins, the corewise tangent, the Hessian's flag — handover

The second commit of slice P12 of `PART1_PLAN.md`, built to the plan's P12 (and its section 2.4),
`P12_DESIGN_NOTE.md` and `P12_DESIGN_REVIEW.md`. Written by the implementer, 2026-11-20, against `graded-messages` at
`5905b733` (P12a committed). Nothing is committed.

## 1. What changed

**The ragged functions** (`backend/probing.py`, `apply.py`, `entries.py`):
- **New: `t3_{probe,apply,entries}_corewise_tangent(x, ww | index, core_directions, order=())`** (N43, O-48): `J x_dot`
  along `core_directions`, of the type of `z` with `K` after `W`.
  - It is one engine call wanting `Jp` on the corewise point (the raw route, P8). It has no `sum_over_samples`.
  - Its doctest shows Euler's identity `J x = 2d z`.
  - The docstring places it in the trio and points repeated calls at one point to the engine with `held=`.
- **The three corewise Hessians gain `sum_over_samples=True`.** `False` keeps `W` leading the blocks' stack, through
  the engine's `keep_W`.

**The uniform twins, merged signatures** (`backend/ut3_sampling.py`, `utv_sampling.py`; section 2.4's table):
- `ut3_probe(x, ww, order=())`, `ut3_apply(x, ww, order=())`, `ut3_entries(x, index)`;
- `ut3_*_corewise_tangent(x, ww | index, core_directions, order=())` (new, N44);
- `ut3_*_corewise_transpose(x, ww | index, residual, order=(), sum_over_samples=True)`;
- `ut3_*_corewise_hessian(x, ww | index, residual, core_directions, order=(), sum_over_samples=True)`;
- `utv_*(frame, variations, ww | index, order=())`;
- `utv_*_transpose(frame, ww | index, residual, order=(), sum_over_samples=True)`, still returning the
  `UT3Variations` `.data` with the gauge masks attached.

Here `x` is the uniform point's `.data` and `frame` the uniform frame's. How each one works:
- Each is one engine call: the engine packs a ragged sample (`len_J=len(order)`) and masks on entry and exit.
- The twins pack a ragged probe residual themselves (`pack_if_ragged(residual, N)`).
- **Probe outputs mirror the sample's form** (`z`, the tangent, `Jp`): a ragged sample gives a ragged tuple, a packed
  sample a packed array.
- The `utv_*` twins read `variations[:2]` and mask them with the frame's masks; the module docstring says so.

A `Vectors` alias (ragged or packed) keeps the signature columns aligned. `ut3_full_sum` is unchanged. The module
docstrings now describe the engine-backed twins.

**The call sites**:
- `uniform_tucker_tensor_train.py`: `probe` and the three corewise transposes;
- `uniform_manifold.py`: `UT3Tangent.probe`, `apply`, `entries` and the three static transposes;
- `backend/uniform_fitting.py`: probe's `point_forward`, its arguments swapped.

The frontend keeps its own form until P13 (`sum_over_samples=sum_over_probes`).

Tests that moved:
- `test_uniform_probing.py` (the six transposes' lambdas, by keyword);
- `test_uniform_curvature.py` (the by-reference table, through a `_by_old_order` adapter);
- `test_dispatch.py` (the two-hole test's uniform Hessian calls);
- the `ut3_probe_corewise_hessian` doctest.

## 2. Tests added (`tests/test_sampling_functions.py`, `tests/test_dispatch.py`)

- **`TestTheCorewiseTangent`**:
  - **the numpy dense oracle**: `J x_dot` equals `dense_*` of `T_dot`, the sum of the `2d` dense tensors with one core
    replaced by its direction, per element of `K = (2,)`. This runs at `()`, `(2,)` and `(1, 1)`, with input jets of
    mixed sizes per mode (a constant, a line and a full jet; for `(1, 1)`, `(2, 1)`, `(1, 2)` and `(2, 2)`), on
    `C = (2,)`.
  - **the substitution**: `tv_*` at `t3_corewise_frame(x)` is **bitwise** equal (the same steps run).
  - the adjoint pair with the corewise transpose, per `k`.
  - **the structural step count**: the closure for `Jp` on a corewise call with `core_directions` has each row once,
    no `_frame.*` row and no read of `U`, `R`, `V` or `H`. It is no longer than the four-store frame's closure for the
    same `Jp`, which is the record of the substitution's larger count.
- **`TestTheCorewiseHessianFlag`**: `sum_over_samples=False` keeps `W` first and sums to the default, at the three
  orders.
- **`TestAgainstAutomaticDifferentiation`** (O-22), skipped where jax is absent:
  - the tangent against `jax.jvp` of `t3_*`;
  - the Hessian against `jax.jvp` of `jax.grad` of `phi(x) = <c, t3_*(x)>`;
  - both at `()`, `(2,)`, `(1, 1)`, in float64 (x64 switched on and restored).
- **`TestTheUniformTwins`**: all 18 twins against their ragged functions.
  - The setup:
    - the point is padded beyond every size (`N = 6`, `n = r = 4`, `C = (2,)`);
    - **garbage (1e3)** fills every padded slot of the point, the frame, the directions (`K + C = (2, 2)`) and the
      variations (`K = (2,)`), and, in the packed form, the mode padding of the sample and of the probe residual.
  - What each comparison asserts:
    - the real block equals the ragged result, and the padding of every block is exactly zero, by an indicator built
      from the ragged blocks' shapes (not from any mask);
    - probe outputs keep the sample's form (a tuple of the ragged shapes, or a packed array with zero padding);
    - both `sum_over_samples` values are checked, at `()`, `(2,)` and `(1, 1)` (entries at `()`).
- **`test_the_signatures`** gains the tangents, the Hessians' flag and the 18 twins' parameter lists.
- **`TestDispatch.test_jit_the_sampling_functions`**: every ragged and uniform sampling function, the six tangents
  included, jitted with jax in and every output leaf jax, at `order = (2,)` (entries none). The uniform masks are
  closed over.

## 3. Verified by script, before the tests (`scratchpad/p12b/verify.py`)

The worst relative errors:

| Check | Worst error | At |
| --- | --- | --- |
| tangent against the dense oracle | 5.9e-16 | `()`, `(2,)`, `(1, 1)`; `K = (2,)`, `C = (2,)` |
| tangent against the substitution | 0 (bitwise) | — |
| Hessian unsummed against summed | 2.6e-16 | — |
| tangent against `jax.jvp` | 7.8e-16 | — |
| Hessian against `jax.jvp(jax.grad)` | 6.3e-16 | — |

## 4. Mutation pass (scratch copy, deleted)

15 mutants, all killed by `tests/test_sampling_functions.py`:
- a tangent dropping `order` (apply, probe);
- a ragged Hessian ignoring its flag;
- the uniform twins:
  - `ut3_probe` and `ut3_probe_corewise_tangent` not unpacking (the mirror);
  - `ut3_probe` dropping `order`;
  - a `ut3` Hessian and a `ut3` transpose ignoring the flag;
  - the `ut3` probe Hessian not packing a ragged residual;
  - `utv_probe` dropping `order` or not unpacking;
  - a `utv` transpose ignoring the flag;
  - a `ut3` tangent cutting the directions' `K`;
- the engine's masks of `D_dot`, and of `V`, removed (the garbage tests).

One mutant I wrote was equivalent (a no-op on the variations) and is not counted.

## 5. Environments

- **`t3toolbox`**: the affected files pass (424 passed, 6,373 subtests): uniform probing, uniform curvature, uniform
  manifold, uniform tensor train, uniform fitting, sampling functions, sweep reuse, curvature, uniform sampling.
- **The floor `tttt`**: `test_sampling_functions`, `test_uniform_probing` and `test_uniform_curvature` pass (114).
- **Module doctests**: every changed module passes on both environments (54 each).

## 6. Gate

The full gate at `-n 2` (logs in `scratchpad/gate_P12b`): **all PASS.**

| Step | Result |
| --- | --- |
| suite | 1824 passed, 5 skipped, 1 xfailed, 90,351 subtests, in 2063 s; peak 8.0 GB; **lowest MemAvailable 3925 MB** |
| doctest-modules | 327 |
| quickstart | pass |
| pages | 48 |
| examples | 14 |
| sphinx | 0 warnings |

## 7. Left for later

- the frontend methods `*_corewise_tangent` and `*_corewise_hessian` on both tensor-train classes (P13);
- the frontend's own sum flag (P13);
- the docs pages that name these functions in prose (P19);
- deleting the `_by_old_order` adapter in `test_uniform_curvature`, with the old-order tests, when P20 removes the
  `*_from_sweep` functions it pairs with.
