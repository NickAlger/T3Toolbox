# P4d: `jets` and `data_preparation` in the jet shape — handover

Slice P4d of `PART1_PLAN.md` (section 5.3, "P4d"; section 2.4). Written by the implementer, 2026-11-19, against
`graded-messages` at `5a102126` (P4c committed). Nothing is committed.

## 1. What changed

**The jet shape, read in one place.** Every function of `backend/jets.py` and `backend/data_preparation.py` that
takes `order` reads it with `contractions._read_order`, through one helper, `jets._kind_and_order(kind, order, name)`:
- checks the kind;
- reads the jet shape: a tuple only, of nonnegative integers (numpy integers normalized; an integer, a list, a bool,
  a float or a negative entry raises);
- raises a `ValueError` for the entries kind with any jet shape but `()`.

`derivative_weight_to_taylor_weight`, which has no kind, calls `contractions._read_order` directly. With `()`, the
entries kind returns its data unchanged (O-47): the conversions are the identity, `jet_to_dual_jet` is `weight**2`
times the data, and the default weight of `prepare_derivative_data` is a 0-d one.

**`t3toolbox/backend/jets.py`** (rewritten; 12 doctested functions and the module).
- *The dual jet.* `jet_to_dual_jet` weighs by `weight**2`, then flips every jet axis (`_flipped`), so slot `k` holds
  `weight[order - k]**2 jet[order - k]`. `dual_jet_to_jet` flips, then divides. The weight's shape is `(d,) + F` for
  probe, `F` for apply and `()` for entries.
- *Derivative data.* `_factorials(order)` is a host float64 array on the box, slot `k` holding `k_1! ... k_m!`.
  - `derivatives_to_taylor` divides slot `k` by it, and `taylor_to_derivatives` multiplies.
  - `derivative_weight_to_taylor_weight` multiplies the weight's trailing `m` axes by it. The weight goes the other
    way from the data, and the probe mode axis leads.
  - `prepare_derivative_data` with no weight returns the factorials: `F` for apply, `(d,) + F` for probe.
- *The input jets.* `affine_jet` and `index_to_one_hot` are kept from P4b, unchanged.
- *The checkers.* `check_sample`, `check_data` and `check_weight` take `(kind, order, ...)` and input jets.
  - `check_sample` checks that `ww` is a tuple or a list of `d` arrays, each one array. Each array has at least
    `m + 1` axes, and every jet axis has at least one slot. All arrays have one `W = shape[m:-1]`, and each has its
    mode size. For entries, `index` is one integer array `(d,) + W`.
  - `check_data` checks the data on the full box `F`, whatever the input jet sizes.
  - `check_weight` checks the exact shape.
  - The docstring of `check_sample` states the trap, with doctests. A shape cannot say how many leading axes are jet
    axes. So input jets passed with `order = ()` may read as one more sample axis and pass:
    `check_sample('probe', (), affine_jet(ww, pp), shape)` passes.
  - `check_data` states the same trap, with a doctest: the jets of a line at `(1,)`, checked with `order = ()`,
    pass as data with no jets.
  - `check_weight` states that the jet shape is taken as given: a forgotten one is caught only if the weight then has
    the wrong shape. Its other limit is stated too: a weight with its axes in another order but the right sizes
    passes.
- `check_perturbation_vectors` and `check_perturbation_index` are deleted from `jets` and its `__all__`. The
  originals stay in `sampling_derivatives`.
- The module docstring is rewritten: the four groups; a jet in several variables; the turn as a flip of every axis;
  the entries kind has no jets. Its examples use `(3,)`, and a two-variable `prepare_derivative_data('apply',
  (1, 2), ...)` whose weight is `[[1, 1, 2], [1, 1, 2]]`.

**`t3toolbox/backend/data_preparation.py`** (8 doctests).
- `data_block_norms(kind, order, y)` returns the norm of each slot `k`: shape `(d,) + F` for probe, `F` for apply,
  `()` for entries.
- `rescale_direction(kind, order, sample, y, length)`:
  - accepts the probe and apply kinds; it needs at least one jet variable, and raises with `()` and for entries;
  - takes `sample`, the input jets;
  - takes `length`: one number or 0-d array, or one per jet variable (a sequence or a 1-d array of `m`; anything
    else raises, naming both forms);
  - multiplies slot `k` of every input jet (by its own jet sizes) and of `y` (on the box) by
    `length_1**k_1 ... length_m**k_m` (`_slot_factors` and `_scale_jet`);
  - returns `(tuple of input jets, y)`;
  - takes a traced length, and keeps each array's own dtype.
- `rescale_coordinates(kind, order, sample, y, scales)`:
  - multiplies every slot of every input jet by its mode's scales (the last axis);
  - for probe, divides the data by the scales;
  - for apply, returns the data as given;
  - raises for entries.
- The module docstring and every docstring and shape comment are in `F` / `J` / `J_in_i` (N39). The module example
  is `line = affine_jet(ww, pp)` at `order = (2,)`.

**`t3toolbox/fitting.py`.**
- `affine_jet` and `index_to_one_hot` are imported from `backend.jets`. They are in `__all__` and named in the
  module docstring's section on preparing data (O-37).
- Its own `rescale_coordinates` reads the mode sizes off the last axis of each input jet, for
  `check_coordinate_scales`.
- Its own `dual_jet_to_jet` reads the kind and the jet shape (`_kind_and_order`) before its safe-mode check. So an
  integer order is a `TypeError` even with a zero weight. The function then makes the zero-divisor check and calls the
  backend.
- Their signatures, docstrings and doctests use the jet shape.

**`t3toolbox/__init__.py`.** `affine_jet` and `index_to_one_hot` are imported from `t3toolbox.fitting` and added to
`__all__`. They are the backend functions themselves.

**`t3toolbox/tucker_tensor_train.py`.** One doctest line in `unscale_coordinates` changed:
`fitting.rescale_coordinates('apply', 0, ...)` became `('apply', (), ...)`. See section 4.

## 2. Tests

**`tests/test_jets.py`**: 44 tests, 471 subtests. Rewritten over the jet shapes `()`, `(1,)`, `(3,)`, `(1, 2)` and
`(2, 0)`.

*The plan's list:*
- The pairing identity `sum_k omega_k^2 a_k b_k = [corner](jet_to_dual_jet(a) * b)`:
  - at `m = 1` and `m = 2`: `(1,)`, `(2,)`, `(5,)`, `(1, 2)`, `(2, 3)`, `(2, 0)`;
  - the product is computed in the test by explicit loops over every pair of multi-indices, untruncated, and its
    corner read off;
  - at `m = 1`, also by `numpy.polynomial.polymul`.
- The round trip, on a packed probe array too.
- `taylor_to_derivatives` against mixed partial derivatives of a known polynomial in two variables
  (`numpy.polynomial.polynomial.polyder` along each axis, at `(3, 2)`), for apply and probe. Also `exp(a s_1 + b s_2)`.
- `prepare_derivative_data` with no weight returns `k_1! ... k_m!`, for apply and for probe with its mode axis.
- The objective written on derivatives is reproduced at `(1, 2)`.
- The entries kind with `()` returns its data unchanged. With a jet shape it raises in every function, the frontend
  `dual_jet_to_jet` included.
- The checkers raise on each wrong form:
  - too few axes;
  - `W` disagreeing once the jet axes are read;
  - a jet axis of size 0;
  - data not of the full box (the input's sizes);
  - the old weight shapes, among them `(d, n + 1)` for `m = 2` and the jet axes exchanged.
- The trap of a forgotten jet shape, pinned: it passes `check_sample` and `check_data`, and all three checkers'
  docstrings state it.
- The frontend `dual_jet_to_jet` raises in safe mode at `m = 2` as at `m = 1`; divides in unsafe mode and under jit.

*Also:*
- every function reads the jet shape with the one reader;
- `affine_jet` and `index_to_one_hot` reachable from `t3toolbox.fitting` and `t3toolbox`, the backend objects;
- the validators of `pp` are gone.

**`tests/test_data_preparation.py`**: 29 tests, 143 subtests. Sections 1 to 3 rewritten; the uniform section and
the public-names test unchanged.
- `rescale_direction` gives the data of the same tensor at the new input jets (`dense_probe`, `dense_apply`):
  - at `(1,)`, `(3,)`, `(1, 2)` and `(2, 1)`;
  - on lines, planes and input jets of their own sizes per mode;
  - with one length, and one per variable (a tuple, a list, an array).
- Slot `k` is checked against `length_1**k_1 ... length_m**k_m`, written out.
- `rescale_coordinates` then `t3_unscale_coordinates` gives the same probe values at an input jet, at `(2,)` and
  `(1, 2)`.

**`tests/test_dispatch.py`.** `_jets_and_preparation_cases` covers `(3,)` and `(1, 2)`: every function, the probe
and apply types, packed probe arrays. `rescale_direction` is jitted with a traced length, and with one per variable.
The entries kind is covered with `()`. 274 subtests pass, run alone.
- Four other tests of the file called the old forms. The plan's count of 45 calls covered only the helper, and the
  first full gate found the four (section 3). They are moved to input jets and tuple orders:
  - `test_one_dispatch_over_all_the_inputs_of_a_call` gains a case where one of two lengths is a jax scalar;
  - `test_a_numpy_float64_array_with_a_jax_array_when_x64_is_off`;
  - `test_the_frontend_rescale_coordinates_under_jit`;
  - `test_the_checkers_run_under_jit`: `(2,)` for probe and apply, `()` for entries; the wrong forms are now a
    missing mode, data of the wrong shape and a weight of shape `(1,)`.

**`tests/test_jax_absent.py`.** `_JETS_SCRIPT` runs over `(2,)` and `(1, 2)` for probe and apply, and `()` for
entries, on input jets. It adds the frontend `rescale_coordinates` and the top-level names.

**`tests/test_import_graph.py`.** One line changed, a departure (section 4).

**`subTest` labels are builtins** (the coordinator's rule, `BRIEF.md`).
- Every `subTest(...)` P4d adds or changes was checked.
- The labels are `str`, `int`, `float`, `bool`, or tuples of Python ints: the jet shapes are literals. A label that
  may hold a numpy value is written `repr(...)` or `str(...)`: the lengths and the bad lengths of `rescale_direction`.
- The four test files were run once with `TestCase.subTest` patched to record any label that is not a builtin. It
  found none, over 662 subtests.
- `test_dispatch`'s jets test was checked by reading, not run that way: its labels are `function=name` (`str`),
  `dtype=np.dtype(dtype).name` (`str`) and `numpy_argument=k` (`int`).

**Mutation pass** (in a scratch copy of `t3toolbox/` + `tests/`, deleted afterwards). All 20 mutants were killed:

| Area | Mutant |
| --- | --- |
| The dual jet | a flip of the first jet axis only |
| Derivative data | factorials of the first variable only |
| Derivative data | the Taylor weight divided instead of multiplied |
| The kind check | the entries kind accepting a jet shape |
| `rescale_direction` | one length used for every variable |
| `rescale_direction` | the lengths reversed |
| `rescale_direction` | the exponent off by one |
| `rescale_direction` | the input jets not scaled |
| `rescale_direction` | the probe data not scaled |
| `rescale_direction` | `()` accepted |
| `rescale_coordinates` | the probe data not divided by the scales |
| `data_block_norms` | the norms over one axis too many |
| Frontend | `dual_jet_to_jet` reading the order after its check |
| Frontend | `rescale_coordinates` reading the mode sizes from the first axis |
| Exports | `affine_jet` dropped from `t3toolbox` |
| Exports | `index_to_one_hot` dropped from `t3toolbox.fitting` |
| Checkers | a jet axis of size 0 passing |
| Checkers | `W` read with the jet axes |
| Checkers | too few axes passing |
| Checkers | the probe data's mode size read off the data |

## 3. The gate

The whole gate, nothing else running:
`PATH=/home/nick/miniconda3/envs/t3toolbox/bin:$PATH dev/graded_messages/implementation/run_gate.sh --detach -n 3 -l <scratchpad>/implementer/gate_P4d`,
then `wait`. The environment was python 3.11.15, numpy 2.4.6 and jax 0.10.2.

**The first run.** Every step passed but the suite. The summary, copied:

    step                result counts                                                      seconds   peak_MB  minavl_MB
    suite               FAIL   6 failed, 1659 passed, 5 skipped, 1 xfailed, 1 warning, 69856 subtests passed   1396.0     10369       2004
    doctest-modules     PASS   312 passed                                                     47.5      1523      10554
    doctest-quickstart  PASS   1 file(s), 0 with failures, 0 failed example(s)                 2.5       277      11798
    doctest-pages       PASS   48 file(s), 0 with failures, 0 failed example(s)               14.0       366      11716
    examples            PASS   14 examples, 0 failed                                           379       640      11447
    sphinx              PASS   0 warning/error line(s)                                       996.7       636      11288

The six failures were all in `tests/test_dispatch.py`: three tests and three subtests of a fourth. The four tests
still called the old forms (section 2). Moving them to the new forms changed only that test file.

**After the fix:**
- `quick tests/test_dispatch.py`, `-n 3`: PASS, 98 passed, 746 subtests passed (704 s).
- The suite step again, `-n 3`:

      step                result counts                                                      seconds   peak_MB  minavl_MB
      suite               PASS   1662 passed, 5 skipped, 1 xfailed, 1 warning, 69880 subtests passed   1439.5     10396       2005

The other five steps were not rerun. Their tree differed from the final one only in `tests/test_dispatch.py`, which
none of them reads.

Against P4c's gate:
- The suite has 2 more tests (1,660 to 1,662): the jet shape read by every function, and the reach from
  `t3toolbox`. The subtests go from 69,649 to 69,880.
- The same 5 skips, 1 warning and 1 xfailed.
- The module doctests are the same 312.
- The suite's peak memory is 10.4 GB, with 2.0 GB available at its lowest.

No process, lock or file of the gate is left in the repository; the logs are in the scratchpad.

## 4. Departures and choices

1. **`tests/test_import_graph.py` changed, where the plan says unchanged.**
   - The test pinned `jets` to import only `common`. Point 1 of the brief has `jets` read the order with
     `contractions._read_order`, so `jets` now imports `contractions`.
   - The pin is now `{common, contractions}`, plus one assertion that `contractions` does not import `jets` (no
     cycle).
2. **`t3toolbox/tucker_tensor_train.py`.** The plan's search found no caller in pages or examples. It missed one
   module doctest: `TuckerTensorTrain.unscale_coordinates` calls `fitting.rescale_coordinates('apply', 0, ...)`. The
   order became `()`. The method itself is unchanged.
3. **`rescale_direction` with the entries kind now raises.** The old function scaled `pp` for entries. With no jets
   there is no direction, and section 2.4 names probe and apply only.
4. **`length` per variable.** It may be a sequence, or a 1-d array of `m` (numpy or jax, traced included). A 1-d
   array of length 1 at `m = 1` is accepted, the same as one number. Any other shape raises a `ValueError`.
5. **The frontend `dual_jet_to_jet` reads the jet shape first.** Without that, an integer order with a zero weight
   would raise the zero-divisor `ValueError` before the backend's `TypeError`.

## 5. The search for `pp` and integer orders (point 6)

I searched `t3toolbox/` and `tests/` for `pp`, `order == 0`, `order + 1`, `order: int`, `range(order` and
`(order,)`. That leaves out `sampling_derivatives` and the old frontend methods, which stay until P20 / P13. No
function written in P1 to P4 takes `pp` or reads an integer order. The remaining hits, and why each stays:

*In `t3toolbox/`:*
- **`pp` as the name of a direction in docstring examples:** `affine_jet(ww, pp)` and `ww + s pp`, in `jets`,
  `data_preparation`, `sampling_definitions`, `probing`, `apply` and `entries`. It is a variable of the example, not
  a parameter.
- **`order + 1` in comments of `contractions`** (lines 449, 610, 653, 661, 745, 747, 756). It is shorthand for the
  box of a tuple jet shape, elementwise. No integer is read.
- **`probing.py:2057`, `dense_probe(T, affine_jet(ww, pp), 2)`.** A doctest showing that an integer order raises.
- **`t3toolbox/fitting.py`, the old model layer:**
  - `_canonical_weight(order: int)` (lines 365 to 397);
  - `_check_ragged_sample(..., order=0)` with `pp` (lines 855 to 883);
  - `LocalModel.sample`'s comment (line 521);
  - `*_derivatives_model(geometry, x, ww, pp, order: int, ...)` (lines 1014 on).

  They stay until the frontend switch-over, which replaces the model factories.

*In `tests/`:*
- **Integer orders and `(ww, pp)` pairs as negative cases.** In `test_jets.py` and `test_data_preparation.py` they
  are refused with a `TypeError`. The old pair `(self.ww, self.pp)` is refused by `check_sample`.
- **`test_sampling_definitions.py`.** `self.pp` is an `affine_jet` direction. The bridge to the old kernels
  (`sampling_derivatives.t3_probe_derivatives(P.ww, P.pp, ..., n)`, `dense_entries_derivatives`) goes in P20.
- **`test_dispatch.py`.** The other `pp` hits are in tests of the old operations: `test_jit_probe_derivatives`, the
  two-hole and derivative-kind tests, `_probe_derivatives_problem_jax` and `test_jit_uniform_derivatives`. Those
  operations go when they are switched over. In the rewritten cases, `pp` and `qq` are `affine_jet` directions.
- **`test_jax_absent.py`.** `pp` and `qq` are `affine_jet` directions.

## 6. Found outside the slice

- The rendered docs (`docs/api_reference.rst` and the user pages) do not mention `affine_jet` or `index_to_one_hot`.
  This is P19's documentation slice. The Sphinx step of the gate builds with them in `t3toolbox.fitting.__all__`.
