# P13a: the frontend sampling signatures — handover

The first commit of P13. It was built to `P13_DESIGN_NOTE.md`, `P13_DESIGN_REVIEW.md` and the coordinator's rulings:
- the split;
- the generalized directions guard;
- `np.asarray` in all five entries methods;
- a `d = 3` swap test;
- **finding A decided by the maintainer: `residual` stays positional, N16's order stands, and the change is
  documented by right-and-wrong doctests.**

Written by the implementer, 2026-11-21, against `graded-messages` at `df7422eb`. Nothing is committed.

## 1. What changed

**The four classes take the merged forms of section 2.5.** The rule is N16: the sample, what is applied, `order=()`
(probe and apply), then `sum_over_samples=True` where there is a sum. Each method is one backend call.

`TuckerTensorTrain`:
- `probe(ww, order=())`, `apply(ww, order=())` (`vecs` renamed), `entries(index)`;
- `k_corewise_transpose(ww, residual, order=(), sum_over_samples=True)`;
- the static ambient transposes `(ww, residual, sum_over_samples=False)`, and
  `entries_ambient_transpose(index, shape, residual, sum_over_samples=False)`;
- new: `k_corewise_tangent(ww, core_directions, order=())` and
  `k_corewise_hessian(ww, residual, core_directions, order=(), sum_over_samples=True)`, with entries taking no `order`.

`UniformTuckerTensorTrain` has the same twelve, without the ambient transposes, on supercores; probe outputs mirror
the sample's form.

`T3Tangent` and `UT3Tangent`:
- `probe` and `apply` gain `order`;
- the static transposes take the frame first: `(frame, ww | index, residual, order=(), sum_over_samples=True)`.

**The six `*_derivatives*` methods per class stay untouched (P13b).**

**What the frontend checks:**
- **The sample:** `jets.check_sample(kind, order, sample, shape)` in every method that takes one. It is structural and
  always on. On the uniform classes it runs on a ragged sample only (`ut3._check_sample`); a packed sample is the
  engine's.
- **The ambient transposes** read the shape off `ww`, or take it given (entries).
- **A direction** must be a pair of core sequences, or two supercores on the uniform class.
  `tucker_tensor_train._check_core_directions` raises a `TypeError` pointing to `y.data` (or `y.data[:2]`) for
  anything else: a `TuckerTensorTrain`, a `UniformTuckerTensorTrain`, a `T3Tangent`, three parts, or a pair of arrays
  on the ragged class.
- **An index** that is not an array goes through `np.asarray` (`_as_index`) in every entries method (a jax array is
  left alone). The two old ad hoc length checks (`entries`, `apply`) are replaced by `check_sample`; both raised
  `ValueError`, as it does.

**Finding A, documented as ruled.** The probe transposes are where an old positional call runs silently, because
the sample and the residual are both `d` vectors. Their docstrings now carry a right-and-wrong doctest at `d = 3`:
- frontend: `TuckerTensorTrain.probe_corewise_transpose` and `probe_ambient_transpose`;
- backend: `probing.t3_probe_corewise_transpose` and `t3_probe_ambient_transpose` (the latter had no examples).

Each checks the adjoint relation against the forward map at the **intended** sample (`probe_corewise_tangent`, or
`probe` / `dense_probe` of a second tensor): the right call holds to `1e-12`, and the old order does not. Measured, the
wrong order is off by a relative 2.4 (corewise) and 55 (ambient). There is no safe-mode check. The CHANGELOG upgrade
note is P19's.

**New doctests on `TuckerTensorTrain.probe`:**
- the shapes at `()`, at `(2,)` with `affine_jet(ww, pp)`, and at `(1, 1)` with `affine_jet(ww, pp, qq)`;
- the forgotten-`order` trap (O-44): wrong against right, with slot 1 compared.

The probe Hessian has a doctest by central difference, and the probe tangent one with a direction that changes one
core.

**Call sites.** Every frontend transpose, corewise transpose and ambient transpose call in the tests was rewritten
by a parser-based rewriter, which moved the arguments and mapped `sum_over_probes` to `sum_over_samples`. Each call
whose old default (`False`) differed from the new one (`True`) got an explicit `sum_over_samples=False`. The calls:
- `test_dispatch` 24;
- `test_uniform_probing` 24 + 1 by hand;
- `test_tucker_tensor_train` 20;
- `o1_uniform` 10;
- `o1_ragged` 9;
- `test_manifold` 6;
- `test_fitting` 2 + 1 by hand.

Tests and doctests pass `residual=` by keyword, readable but not required. The defining modules' doctests were
edited with their methods. No page or example calls a changed method: the docs' prose tables are P19's.

## 2. Tests (`tests/test_frontend_sampling.py`, new)

- **Each method against its backend function:** bitwise, at `()` and `(2,)`, for `TuckerTensorTrain` and `T3Tangent`
  (the transposes return a `T3Tangent`). The uniform classes are compared with the ragged ones on real parts, with
  a ragged and a packed sample, and the probe mirror is asserted.
- **The checks:**
  - every sample-taking method on the four classes raises on `d - 1` modes, a mismatched `W`, and too few axes for
    `(1, 1)`;
  - entries methods raise on `d - 1` rows and on a float index;
  - an integer `order` raises a `TypeError`;
  - no entries method has `order`;
  - a nested-list index equals the array, on the eight entries entry points;
  - the direction guard, for each wrong form on both classes, while a list pair and two supercores pass.
- **The Hessian methods:**
  - symmetry `<y, H z> = <z, H y>`;
  - `H y` equals the slope at 0 of an exact polynomial fit of degree `2d - 1` to the corewise transpose along
    `x + t y` on `2d` points (`1e-9`, at `()` and `(2,)`).
- **The sum flag:** `False` keeps `W` and sums to the default, on the corewise transposes, Hessians, tangent
  transposes and uniform corewise transposes.
- **The swap guard** (the review's request): at `d = 3`, `probe_corewise_transpose` and `probe_ambient_transpose`
  with `ww` and the residual swapped differ by more than 1e-2 relative.

## 3. Verified by script

- The right and wrong orders, measured before the doctests: corewise `1.1e-16` against `2.4`, ambient `3.5e-15`
  against `55`.
- The six new methods are bitwise their backend functions (the tests).
- The polynomial fit reproduces `H y` to `1e-9`.

## 4. Mutation pass (scratch copy, deleted)

12 mutants, all killed by `tests/test_frontend_sampling.py`:
- `probe` without `check_sample`, and dropping `order`;
- the tangent transpose's flag fixed;
- `entries` and `UT3Tangent.entries` without the list conversion;
- the guard off, and the uniform guard checking the ragged form;
- `UT3.probe` dropping `order`;
- the apply Hessian ignoring its flag;
- `check_sample` run on a packed sample;
- the entries tangent with the TT directions zeroed;
- the probe Hessian with a doubled residual.

## 5. Environments and gate

- **`t3toolbox`**: the gate's `quick` step over the ten affected files passed (512 tests, 39,791 subtests).
- **`tttt`**: the new file, `test_manifold` and `test_uniform_probing` pass (103).
- **Module doctests**: the five changed modules pass on both environments (133 each).
- **The full gate at `-n 2`**: below.

The full gate at `-n 2` (logs in `scratchpad/gate_P13a`) passed every step:
- **suite**: 1835 passed, 5 skipped, 1 xfailed, 90,480 subtests, in 1903 s; peak 8.1 GB; **lowest MemAvailable
  3851 MB**;
- **doctest-modules**: 331;
- **quickstart**: pass;
- **pages**: 48;
- **examples**: 14;
- **sphinx**: 0 warnings.

## 6. For P13b

- Remove the 24 `*_derivatives*` methods and move every caller: the 14 test files, the module and page doctests, and
  the 6 examples with their dense ground truth.
- Delete `chunking.md` and its links.
- Add the conversion test independent of `affine_jet` and `taylor_to_derivatives`, which survives P20.
