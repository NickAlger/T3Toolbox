# P3: handover

Slice P3 of `PART1_PLAN.md`: jets, the checkers and data preparation, with their reach from the frontend. Written
2026-11-17 on `graded-messages` at `a38d1a3` (P2). Nothing is committed. The slice adds code and removes nothing.

**Brought up to date after review round 1** (`P3_REVIEW_r1.md`; the answers are in `P3_RESPONSE_r1.md`). What
changed in that round: one new public backend checker, `check_coordinate_scales` (decided by the coordinator),
which the two methods call in place of a private frontend helper; `check_sample` refuses a missing `pp` and names
the cause when a sample has the form of the other order; the frontend `dual_jet_to_jet` checks the divisor as the
backend computes it; `rescale_direction` takes a traced length; every function chooses numpy or jax once over all
its inputs; `equalizing_weight` is silent on all-zero norms; 0-dimensional inputs give arrays; the uniform twin
zeroes packed scales beyond each `Ni`; the docstrings state the ragged type and the trap of a packed probe array,
and the note of row 2.5 on the default weight; 14 more tests. **And after the coordinator's decision that
followed:** `t3toolbox.fitting.rescale_coordinates` is fitting's own function, which checks the scales and calls
the backend one; 2 more tests. **And after the maintainer's decision of 12:02:** the four functions whose result
depends on the type of the data take `kind`, in the order `(kind, order, ...)`, and the type is decided by `kind`,
never by whether the object is a tuple; a packed probe array with `kind='probe'` is then taken mode by mode and is
right. This file describes the tree after all three.

## 1. What changed

| File | Change |
| --- | --- |
| `t3toolbox/backend/jets.py` | **new**: 11 public functions (below), and the module doctest of 2.3 |
| `t3toolbox/backend/data_preparation.py` | **new**: 7 public functions (below), `check_coordinate_scales` among them |
| `t3toolbox/fitting.py` | nine backend functions imported and listed in `__all__`; its own `dual_jet_to_jet` and `rescale_coordinates`; one section added to the module docstring. Nothing else touched |
| `t3toolbox/tucker_tensor_train.py` | the method `TuckerTensorTrain.unscale_coordinates(scales)`; one import |
| `t3toolbox/uniform_tucker_tensor_train.py` | the method `UniformTuckerTensorTrain.unscale_coordinates(scales)`; one import |
| `tests/test_jets.py` | **new**, 42 tests |
| `tests/test_data_preparation.py` | **new**, 27 tests |
| `tests/test_dispatch.py` | six tests (and one helper that lists the cases) |
| `tests/test_import_graph.py` | three tests; the docstring |
| `tests/test_jax_absent.py` | one test and its script |

Nothing is deleted. The old validators (`sampling_derivatives.check_perturbation_vectors`,
`check_perturbation_index`), the old frontend `_canonical_weight` and `_check_ragged_sample`, and all derivative-mode
code stand as they were.

### The public surface, backend

`backend/jets.py`:

```python
def jet_to_dual_jet(kind, order, jet, weight) -> typ.Any            # weight**2 * jet, then the order axis turned
def dual_jet_to_jet(kind, order, dual_jet, weight) -> typ.Any       # turned back, then divided by weight**2; no check
def derivatives_to_taylor(kind, order, jets) -> typ.Any             # slot t divided by t!
def taylor_to_derivatives(kind, order, jets) -> typ.Any             # slot t multiplied by t!
def derivative_weight_to_taylor_weight(weight, order) -> NDArray    # slot t (the last axis) multiplied by t!
def prepare_derivative_data(kind, order, y, weight=None, validation=None) -> (y_taylor, weight_taylor, validation_taylor | None)
def check_perturbation_vectors(ww, pp) -> None                      # copied unchanged
def check_perturbation_index(index, pp, shape=None) -> None         # copied unchanged
def check_sample(kind, order, sample, shape) -> None
def check_data(kind, order, sample, y) -> None
def check_weight(kind, order, weight, d) -> None
```

`backend/data_preparation.py`:

```python
def data_block_norms(kind, order, y) -> NDArray                     # probe (d,)+T; apply/entries T
def equalizing_weight(block_norms, floor=1e-6) -> NDArray           # 1 / max(x, floor * max(x))
def rescale_direction(kind, order, sample, y, length) -> (sample, y)        # pp * length; slot t of y * length**t
def rescale_coordinates(kind, order, sample, y, scales) -> (sample, y)      # vectors * scales; probe y[i] / scales[i]
def t3_unscale_coordinates(x, scales) -> (tucker_cores, tt_cores)           # column j of Tucker core i * scales[i][j]
def ut3_unscale_coordinates(x, scales) -> (tucker_supercore, tt_supercore, shape, masks)
def check_coordinate_scales(scales, shape) -> None                          # new in round 1: see below
```

**`check_coordinate_scales`, the one name that is not in the plan** (decided by the coordinator after review
round 1, to be recorded for the maintainer):

```python
def check_coordinate_scales(
        scales:  typ.Sequence[NDArray],    # expected: len=d, elm_shape=(Ni,)
        shape:   typ.Sequence[int],        # the mode sizes N_i, len=d
) -> None:
```

Its docstring says: good `scales` are a tuple or a list with one array per mode, the `i`-th of shape `(N_i,)`, one
scale per coordinate, the form `rescale_coordinates` and the two unscale functions take. Refused, with a
`ValueError` naming what is wrong: a wrong count of arrays, an array of the wrong length for its mode, an array
that is not a vector, and anything that is not a tuple or a list (scales packed to `(d, N)` included). It is a
checker, not a guarantee: the three functions that take `scales` do not call it, and the docstring says what they
do with wrong `scales`, silently (a short list drops the modes it does not cover; a scale of length 1, or a number,
is broadcast). The two `unscale_coordinates` methods call it. Only the form is checked, never a value, so it runs
under `jit`; a zero scale is not refused, and the docstring says what it does (`rescale_coordinates` zeroes that
coordinate of every input and divides the probe data there by zero; `t3_unscale_coordinates` makes every entry of
the tensor with that coordinate zero).

The signatures are the plan's sketch (section 2.4), argument for argument, **except the first four**, which the
maintainer changed on 2026-11-17 (12:02): they take `kind` first and `order` second, the order of every other
function of the two modules that takes a kind. `weight` has no default in them (it had none; pass `None` for no
weight). `derivative_weight_to_taylor_weight(weight, order)` is unchanged: the order axis of a weight is its last
axis for every kind.

**`kind` says which type an object is of; the object is never looked at to decide it.** An object "of the type of
`z`" is, for probe, one array `T + W + ... + (Ni,)` per mode, and for apply and entries one array `T + W + ...`
with the order axis first. For `'probe'` the four functions (and `prepare_derivative_data`) go over the modes along
the first axis of the object and return a tuple, one array per mode. So a tuple, a list and an array packed for the
uniform layout, `(d,) + T + ... + (N,)`, are all taken mode by mode: **a packed array with `kind='probe'` gives
what `tuple(packed)` gives**, with no code for the uniform layout. **A mismatched type is not guarded against**: a
tuple given with `kind='apply'` is made into one array, which fails (a `ValueError` from numpy) where the modes
differ in size, and otherwise reads the mode axis as the order axis (an error from broadcasting, or, with no
weight, the modes turned and no error). An unknown kind raises. The weight is `(d,) + T` for probe and `T`
otherwise, `None` meaning ones. `scales` is `len=d`, `elm_shape=(Ni,)`.

### The public surface, frontend

```python
# t3toolbox/fitting.py: nine backend functions, the same objects, under their backend names
derivatives_to_taylor, taylor_to_derivatives, derivative_weight_to_taylor_weight, prepare_derivative_data,
jet_to_dual_jet, data_block_norms, equalizing_weight, rescale_direction, t3_unscale_coordinates

# t3toolbox/fitting.py: two functions of its own, with the backend's name and parameters
def dual_jet_to_jet(kind, order, dual_jet, weight) -> typ.Any    # the safe-mode check of the divisor, then
                                                             # backend_jets.dual_jet_to_jet
def rescale_coordinates(kind, order, sample, y, scales) -> (sample, y)      # check_coordinate_scales (always), then
                                                             # backend_preparation.rescale_coordinates
TuckerTensorTrain.unscale_coordinates(self, scales) -> TuckerTensorTrain                # over t3_unscale_coordinates
UniformTuckerTensorTrain.unscale_coordinates(self, scales) -> UniformTuckerTensorTrain  # over ut3_unscale_coordinates
```

The frontend `dual_jet_to_jet`: `if weight is not None and safety.checks_active(dual_jet, weight)`, take the
divisors from the private backend helper `jets._squared_weights` (`weight**2` in the precision of the computation,
the very arrays the backend divides by), count their entries that are exactly zero and `safety.require` that there
are none; then call the backend function. So it raises in safe mode, also for a weight whose square underflows,
and divides inside `safety.unsafe()` and under a jax trace (with the weight traced, and with the weight a concrete
array closed over: `checks_active` detects both). The two `unscale_coordinates` methods call the backend
`check_coordinate_scales`, then the backend function.

The frontend `rescale_coordinates`: for `'probe'` and `'apply'` it reads the mode sizes off the sample (the last
axis of each `ww[i]`) and calls the backend `check_coordinate_scales(scales, shape)`; then the backend
`rescale_coordinates` (which raises for any other kind). The check is structural, so it is made in both safety
modes and under `jit`. The checker is not re-exported from `fitting`.

### How the code is built

- **No safe mode in the backend.** Neither module imports `t3toolbox.safety`, reads a mode or raises on a number.
  The backend `dual_jet_to_jet` divides by `weight**2` whatever it is.
- **Checkers, not guarantees.** No backend function calls `check_sample`, `check_data`, `check_weight` or
  `check_coordinate_scales`, the functions of their own modules included (a test reads the syntax trees). `check_sample` calls the two validators of `pp`, as
  the plan says.
- **Dispatch** is read off the arrays, once per call, with `tree_contains_jax` over all the arrays the call is
  given (the modes of a probe tuple, the weight, the validation data, the length, the scales), then `get_backend`;
  no flag. One jax array makes every array the call computes a jax array. The uniform masks are passed through
  untouched, host numpy; the mask of the mode sizes in `ut3_unscale_coordinates` is host numpy.
- **No array in a default dtype.** The factorials, the powers of the length, the ones of the default weight and the
  packed scales are made in the dtype of the array they act on. Where two arrays of the user meet (a jet and a
  weight; vectors, data or cores and the scales) the result has the lower floating precision of the two, on jax if
  either is (the absorbing rule of 5.6), and on jax no more than jax's own floating dtype, so a numpy float64 array
  is never asked of jax in a dtype it lacks: see choice 4 below.
- **Signatures and names** in the house style: one argument per line with its shape comment; body locals carry the
  axis layout (`jet_TX`, `weight_T`, `scales_dCnN`).

## 2. The tests, and what each establishes

**The algebra was checked by a script before the tests were written** (not kept in the tree): the library's own
derivative-mode outputs at 2026.5.0 (`x.probe_derivatives`, `x.apply_derivatives`) are `t!` times the Taylor jets of
`dense_probe` / `dense_apply` and convert to them (5e-16); the objective of the old derivative-mode kind
(`bfit.probe_derivatives_kind(order, weight=om)` in a `Problem`) equals the Taylor objective after
`prepare_derivative_data` (2.0135458518093183 against 2.0135458518093077, 5e-15); the pairing identity, the round
trip, both rescalings against the dense functions, and the uniform twin against the ragged one. The comparison with
the old kinds is not in the suite, because that code goes in P16 and P20; the suite's tests take independent routes
that stay.

`tests/test_jets.py` (42 tests):

| Class | Tests | Establishes |
| --- | --- | --- |
| `TestDualJet` | 12 | `dual[t] == weight[order-t]**2 * jet[order-t]` slot by slot, three kinds, orders 0, 1, 3, sample stacks `()`, `(7,)`, `(2,3)` and extra axes `K + C`; one hand-computed case that tells a plain weight and a missing turn apart; no weight is ones; the round trip both ways to 1e-13; **the pairing identity**, the top coefficient taken by `numpy.polynomial.polymul` entry by entry; the backend inverse at a zero weight returns `inf`/`nan` in that slot, leaves the others untouched, and raises and warns nothing (run with warnings turned into errors); a backend function given a weight that `check_weight` rejects computes with it; the result has the lower precision of jet and weight; **`kind` decides the type**: with `kind='probe'` a list, a tuple and a packed array `(d,) + T + W + (N,)` give the same tuple, one array per mode, in the four functions, `prepare_derivative_data` and the frontend `dual_jet_to_jet`, with the values checked against the definition written out on the packed array, and one mode still gives a tuple of one; **a mismatched type is not guarded** (what the docstrings say is pinned: a tuple with `kind='apply'` raises where the modes differ in size or do not broadcast, and with no weight comes back with the modes turned and no error; an unknown kind raises); the four signatures are `(kind, order, ...)` with no default; a 0-dimensional input gives an array |
| `TestFrontendDualJetToJet` | 7 | it is the frontend's own function with the backend's parameters; nonzero weight: the backend's numbers exactly, in safe and in unsafe mode; a zero weight raises in safe mode (probe, apply, and the 0-d weight of order 0); divides in `safety.unsafe()` with the backend's numbers and no warning; divides under `jax.jit` with the weight traced and with the weight closed over, and raises on eager jax arrays; **a small nonzero weight (`1e-8`, `1e-150`) passes in safe mode** with the backend's numbers; **a weight whose square underflows is refused like a zero** (float32 `1e-30`; float64 `1e-30` against a float32 dual jet; `1e-200`) |
| `TestDerivativeData` | 8 | `exp(a s)` analytically; **against derivatives taken by `numpy.polynomial`** from an interpolating polynomial of `dense_apply` and `dense_probe` along `ww + s pp` (a route with no factorial of ours in it); slot by slot and the round trip; the weight conversion on the last axis for both weight shapes; **the objective written on derivatives is reproduced** for the three kinds with a weight and with none, and at order 2 and above it is shown to differ from ones on the Taylor coefficients; **no weight returns `t!`** exactly, `(order+1,)` for apply and entries and `(d, order+1)` for probe, and `check_weight` accepts it; a given weight and validation data are converted; dtype kept, unknown kind raises |
| `TestCheckers` | 14 | each checker passes every good input (three kinds, orders 0, 1, 3, three sample stacks) and raises, with the message checked, on each wrong form: a wrong count of vectors, a wrong mode size, a wrong sample stack, a packed array, a scalar in place of a vector, the wrong form for the order, `pp` of a wrong count, size or stack, an index of wrong `d`, of float dtype, or given as a list; `y` of a wrong count, size, stack, order axis, with a `C` axis, or a list for apply; **16 wrong weight shapes**, the old forms among them (a bare per-order vector, a mode dimension of 1, an order dimension of 1, the plain probe's `(d, 1)`); unknown kind and bad order; **no backend module refers to a checker** (`check_coordinate_scales` included); the two validators are textually the old ones; every new public function of `jets` has a doctest; **a missing `pp` is refused** at order 1 and above; the message names the cause when the sample has the form of the other order; the stated limit of `check_weight` (a transposed square probe weight passes) |
| `TestFrontendReach` | 1 | of the names of `t3toolbox.fitting`, nine are the backend objects; `dual_jet_to_jet` and `rescale_coordinates` are fitting's own, with the backend's parameters; all are in `__all__`; no checker is an attribute of `fitting` |

`tests/test_data_preparation.py` (27 tests):

| Class | Tests | Establishes |
| --- | --- | --- |
| `TestEqualizing` | 7 | **the block norms have the shape of the weight** at orders 0, 1, 3 for three kinds (and `check_weight` accepts them); each is `numpy.linalg.norm` of its block; the weight is `1 / max(x, floor * max(x))` entry by entry for three floors, with both branches met; **a zero block gets the floor weight**; the floor is relative (`weight(c x) = weight(x) / c`); the weighted blocks of real probe data have norm one; all-zero norms give `inf` with no warning, and a 0-dimensional input an array |
| `TestRescaleDirection` | 5 | the rescaled data are the data of the same dense tensor at the rescaled sample (three kinds, two orders, two lengths); the block norms scale by `length**t`; order 0 and an unknown kind raise; dtype kept; the length may be a 0-dimensional array |
| `TestRescaleCoordinates` | 11 | the rescaled data are the data of the tensor in the new coordinates (probe and apply, orders 0 and 2); **rescale then unscale gives back the same probe values in the original coordinates**; unscale is the dense tensor scaled along every mode, with `C` stacks; **the method equals the function**; **`check_coordinate_scales` passes good scales and raises on each of nine wrong forms**; it looks at no value (a zero scale passes and zeroes its slice of the tensor); the backend functions do not check `scales` (the two silent behaviours the docstrings state are pinned); **both methods raise on the nine forms, through the backend checker**; **`fitting.rescale_coordinates` raises on the nine forms in both safety modes and gives the backend's arrays exactly on good input**, reads the mode sizes off the sample, leaves the backend's own errors to the backend, and goes through the backend checker; entries raises; the lower precision where arrays meet |
| `TestUniformUnscaleCoordinates` | 4 | the three prongs of `testing_strategy.md`: **equal to the ragged function on real parts** over the `(C, force_pad)` matrix and a stack of varying ranks, without and **with garbage in every padded slot** (a test first shows the garbage helper touches no real entry and does change the array); **the output masks, shape and TT supercore are exactly the input's**, the masks host numpy bool; the mode padding comes out zero; packed scales with garbage beyond `Ni` and garbage in the supercores: real parts right, the mode padding zero, the rank padding multiplied like the rest (the docstring's statement); the method equals the function and the ragged method |

`tests/test_dispatch.py`: `test_jit_jets_and_data_preparation` (every array-returning function of the two modules,
probe type and apply type: numpy in gives numpy out; jit and eager on jax give jax; float32 stays float32 with x64
on; the numbers are numpy's; each argument in turn a numpy array held outside the trace);
`test_jit_ut3_unscale_coordinates` (jax supercores, the masks stay host numpy, both methods under jit);
`test_the_checkers_run_under_jit` (a good input passes on tracers, a wrong `pp`, `y`, weight or `scales` raises at
trace time); the cases of the first test include `rescale_direction` with **the length a traced argument**;
`test_one_dispatch_over_all_the_inputs_of_a_call` (twelve calls with one jax array among numpy ones: every computed
array is jax); `test_a_numpy_float64_array_with_a_jax_array_when_x64_is_off` (no warning, float32 jax out);
`test_the_frontend_rescale_coordinates_under_jit` (jax out with the backend's numbers; a wrong count, a wrong
length for a mode and a length-1 scale raise at trace time, in safe and in unsafe mode).

`tests/test_jax_absent.py`: `test_jets_the_checkers_and_data_preparation_run_without_jax` (both modules, the reach
from `fitting`, the frontend `dual_jet_to_jet` raising in safe mode and dividing in unsafe mode, both methods, with
warnings turned into errors).

`tests/test_import_graph.py`: `test_no_backend_module_imports_safety` (directly or through any backend module, in
any import form, also inside a function); `test_the_import_finder_sees_safety_in_both_import_forms`;
`test_the_jet_and_preparation_modules_import_downward` (`jets`: `common` only; `data_preparation`: `common`, `jets`,
`ut3_operations`).

**The tests were tested** (round 1 added 20 more mutations, the reviewer's uncaught one among them, all caught:
`P3_RESPONSE_r1.md`): 25 mutations of the code in a scratch copy (run with `python -B`, the tree untouched),
each caught: a plain weight for the squared one; no turn, in either direction; the factorials the wrong way; the
factorial off by one; the weight divided; no weight returning ones; the `errstate` removed; `check_weight` allowing
a dimension of 1, or a bare vector; the stack of `pp` unchecked; the highest precision for the lowest; an absolute
floor; the length to the power `t + 1`; the probe data multiplied; unscale dividing; the uniform scales padded with
ones; the uniform modes reversed; wrong axes in the block norms; the frontend never checking, and always checking;
either method without its check of `scales`; a backend module importing `safety`; a backend function calling a
checker.

## 3. The gate

Run through `dev/graded_messages/implementation/run_gate.sh`.

**The whole gate on the final tree** (after the `kind` change; `run_gate.sh --detach -n 2 -l /tmp/gate_P3_r3`, then
`run_gate.sh wait /tmp/gate_P3_r3`; exit 0). No file under `t3toolbox`, `tests`, `docs` or `examples` is newer than
the run's `PID` file (started 12:10:25; checked with `find -newer`):

    step                result counts                                                      seconds   peak_MB  minavl_MB
    suite               PASS   1580 passed, 5 skipped, 1 xfailed, 1 warning, 65875 subtests passed    871.2      3855       4068
    doctest-modules     PASS   304 passed                                                     25.3      1427       6271
    doctest-quickstart  PASS   1 file(s), 0 with failures, 0 failed example(s)                 1.5       223       7370
    doctest-pages       PASS   48 file(s), 0 with failures, 0 failed example(s)               11.8       461       7168
    examples            PASS   14 examples, 0 failed                                           182       368       7305
    sphinx              PASS   0 warning/error line(s)                                       183.6       718       6823

- No failure and no error in any step, so nothing to mark as in the baseline or new.
- The suite: 1,580 passed, which is P2's 1,501 and the 79 tests of this slice (42 in `test_jets.py`, 27 in
  `test_data_preparation.py`, 6 in `test_dispatch.py`, 3 in `test_import_graph.py`, 1 in `test_jax_absent.py`);
  65,875 subtests against 65,157. The 5 skipped are the slow tier, as in the baseline. The 1 expected failure is
  P1's marker for the contraction order deferred to Part 2 (row 5.5), untouched. The 1 warning is the baseline's
  intended one (Adam on the manifold geometry, `tests/test_oracle_sweep.py`).
- The module doctests: 304 passed, against 282 before the slice: 10 in `jets` (the module and nine functions), 8 in
  `data_preparation` (the module and seven functions), the frontend `dual_jet_to_jet` and `rescale_coordinates`,
  and the two methods.
- The suite of this run contains the slice's test files and `tests/test_fitting.py`, and the run contains the
  `doctest-modules` and `sphinx` steps, so those were not run a second time on their own after the `kind` change.
  Before the run, with the bytecode cache removed and `python -B`: the slice's four test files (80 passed, 547
  subtests), the six slice tests of `test_dispatch.py`, the module doctests of `jets`, `data_preparation` and
  `fitting` (24 passed), and seven mutations of the `kind` change in a scratch copy, all caught in the end.

**The runs of the steps before**, each on the tree of its time, for the record:
- after the frontend `rescale_coordinates` (before the `kind` change): `run_gate.sh quick` on the slice's five test
  files and `tests/test_fitting.py`: 220 passed, 1285 subtests (`/tmp/gate_P3_r2b_quick`); `doctest-modules` 304
  passed and `sphinx` 0 warnings (`/tmp/gate_P3_r2b_docs`);
- after review round 1: the whole gate, 1576 passed, 5 skipped, 1 xfailed, 65836 subtests; 303 module doctests;
  quickstart, 48 pages, 14 examples; sphinx 0 warnings (`/tmp/gate_P3_r2`). The session was interrupted once during
  that run; it was detached and went on, and the tree and the notes were checked after the resumption;
- the first handover: the slice's tests through `run_gate.sh quick` (280 passed, 1612 subtests,
  `/tmp/gate_P3_quick`); the sphinx step alone, which failed on four warnings from one docstring (issue 4 of
  section 5), fixed before the gate; the whole gate, 1562 passed, 5 skipped, 1 xfailed, 65775 subtests; 302
  module doctests; sphinx 0 warnings (`/tmp/gate_P3`).

## 4. Departures from the plan, and the choices it left open

**Departures: two, neither the implementer's.** (a) The public backend checker
`check_coordinate_scales(scales, shape)` is not in the plan's table of names (review finding 2; decided by the
coordinator on the rule that the backend gives its user a checker for each thing the frontend checks). (b) Four
signatures differ from the plan's sketch: `jet_to_dual_jet`, `dual_jet_to_jet`, `derivatives_to_taylor` and
`taylor_to_derivatives` take `(kind, order, ...)` (the maintainer, 2026-11-17, 12:02). Everything else, names,
signatures, argument order and modules, is the plan's. `dev/graded_messages/PART1_PLAN.md` and
`SAMPLE_VARIABLES.md` show as modified in the working tree; those edits are not mine.

**Choices that touch the public behaviour** (the reviewer's list; for the log):

- **(1) Superseded.** The first handover's choice 1 (the type of `z` read from whether the object is a tuple, in
  the four functions that had no `kind`) is gone: by the maintainer's decision of 12:02 the four take `kind`, and
  `kind` decides the type (section 1). The trap of the packed probe array went with it.
- **(4) The lowest precision wins.** Where two arrays of the user meet, the result has the lower floating
  precision (row 5.6 applied to public helpers the row does not name), by one private helper,
  `jets._lowest_precision_dtype(xnp, *arrays)`, which `data_preparation` imports; on jax it is capped at jax's own
  floating dtype. A weight converted alone keeps its dtype. The engine's seed, made through `jet_to_dual_jet`,
  follows 5.6 by this. The helper may belong in `common` when the engine needs it (P10).
- **(5) `prepare_derivative_data` accepts order 0**: it returns the data unchanged and ones in the shape of a
  weight, where the plan's comment says `order >= 1`. The default weight has the floating dtype of `y`.
- **(10) `ut3_unscale_coordinates` also takes scales already packed to `(d, N)`**, wider than the plan's "the same
  scales" (`ut3_operations.pack_if_ragged`). The scales are set to zero beyond each `Ni` from `shape`, ragged or
  packed, so the padding of the mode axis comes out zero whatever it held; the rank padding is multiplied like the
  rest; the masks are returned as given.
- **(11) The checkers take the ragged form only.** `check_sample` wants `ww` and `pp` a tuple or a list of arrays
  (a packed array is refused: check before packing) and `index` one integer array; `check_data` wants the probe
  data as a tuple or a list; `check_coordinate_scales` the ragged scales. **Consequence for P14:**
  `check_batch` is built from them, so a backend user who holds a packed batch has no checker for it; Claude
  decides at P14 whether that is wanted.

**The other choices the plan left to the implementer.**

2. **The backend `dual_jet_to_jet` is silent on numpy too.** The division is made inside
   `np.errstate(divide='ignore', invalid='ignore')`, so a zero divisor gives `inf` or `nan` with no
   `RuntimeWarning`, as on jax, and cannot become an error under a user's `-W error`. `equalizing_weight` on
   all-zero norms is silent in the same way and gives `inf`.
3. **The frontend check is exact, with no tolerance, and on the divisor.** It refuses an entry of `weight**2`, as
   the backend computes it, that is exactly zero: a zero weight, and a weight whose square underflows in the
   precision of the computation. A small weight is a legitimate weight and passes.
6. **`rescale_direction` at order 0 raises** a `ValueError` (a sample has no direction there);
   **`rescale_coordinates` for `'entries'` raises** (the plan: "probe and apply only"). Both, and an unknown kind,
   are what the functions need in order to run.
7. **What the rescalings return.** The inputs `ww` or `index` of `rescale_direction` are returned as given (the
   same object, also when the directions and the data come back on jax); vectors that are rescaled come back in
   tuples; the data of the apply kind under `rescale_coordinates` are returned as given. `length` is a number or a
   0-dimensional array and may be traced.
9. **0-dimensional results are arrays**, not numpy scalars: `data_block_norms` and `equalizing_weight` for apply
   and entries at order 0, and the jet functions on a 0-dimensional input.
11b. **What the checkers do not look at.** No value: an index out of range, a zero scale and a zero weight all
    pass (the last is the frontend `dual_jet_to_jet`'s check, in safe mode). `check_data` reads `W`, `d` and the
    mode sizes from the sample, which it takes to be good, and wants exact shapes with no stack of base points.
    `check_weight` cannot see a probe weight given transposed when `d == order + 1` (its docstring says so). The
    checkers of a sample, its data and its weight raise on an unknown kind and on an order that is not an integer
    `>= 0`.
12. **The two methods and `fitting.rescale_coordinates` check `scales` structurally, always**, through the backend
    `check_coordinate_scales` (the latter decided by the coordinator after review round 1).
13. **The uniform method takes the ragged `scales`**, the ones given to `rescale_coordinates`, whatever the padded
    mode size (the checker refuses packed ones).
14. **Where the tests live.** `tests/test_jets.py` (named by the plan) and `tests/test_data_preparation.py`. The
    tests of the frontend `dual_jet_to_jet` and of the reach from `fitting` are in `test_jets.py`.
15. **The import graph test also pins the imports of the two new modules.**

**New names.** One public: `check_coordinate_scales` (`backend/data_preparation.py`), decided by the coordinator
after review round 1; **to be recorded for the maintainer**. It is not re-exported from `t3toolbox.fitting`: the
plan's list of ten is unchanged, and two of the ten are now fitting's own functions that call the backend ones
(`dual_jet_to_jet`, `rescale_coordinates`); the frontend user is checked by those and by the methods. Private:
`jets._lowest_precision_dtype` and `jets._require_kind` (imported by `data_preparation`), `jets._squared_weights`
(called by the frontend `dual_jet_to_jet`), and module-private helpers.

## 5. Issues, and things outside the slice

1. **The probe type on the uniform layout: settled by the maintainer.** The four functions take `kind`; a packed
   probe array with `kind='probe'` is taken mode by mode and is right; the result is a tuple, which the caller
   stacks if it wants it packed. Callers of the four functions: none outside this slice (searched `t3toolbox`,
   `tests`, `docs`, `examples`, `README.md` and `/home/claude/t3polynomial` for the four names, the module's import
   forms and aliases). What is left unguarded is a mismatched type (section 1). The functions of
   `data_preparation` that take `kind` still document the ragged type for probe: `data_block_norms` and
   `rescale_direction` would also go over a packed array mode by mode, but `rescale_coordinates` divides mode
   `i`'s data by ragged scales of length `Ni`, which a packed array of width `N` does not broadcast against.
2. **Wrong `scales` at `fitting.rescale_coordinates`: decided and done.** The frontend function checks them
   (above), so a frontend user gets the error where the scales first come in.
3. **`docs/api_reference.rst`** lists the backend modules by hand and lacks `jets` and `data_preparation` (as it
   lacks `sampling_definitions`, noted at P2 for P19). Not touched.
4. **sphinx reads the "type" of a numpydoc parameter as a cross-reference.** `x: (tucker_supercore, tt_supercore,
   shape, masks)` in a Parameters section gave four "more than one target" warnings (`shape`, `masks`), which `-W`
   makes errors. The docstrings say `x: tuple` and spell the tuple in the text. Worth knowing for later slices.
5. **`CHANGELOG.md`** has no entry for the branch's additions; none was added (Part 1 is not released on its own).

**For P19** (review findings 12 and 13, left as instructed):
- The references to the re-exported functions render as plain text. In the built page of `t3toolbox.fitting` the
  names `derivatives_to_taylor()`, ..., `t3_unscale_coordinates()`, `jet_to_dual_jet()` are not links: autoapi
  does not document an imported name under the importing module. (`t3toolbox.fitting.rescale_coordinates()`, in the
  docstrings of the two `unscale_coordinates` methods, is fitting's own function since the decision above, so it
  has a page; whether the link resolves was not looked at.) Targets `t3toolbox.backend.jets.<name>` and
  `t3toolbox.backend.data_preparation.<name>` would link.
- A frontend reader has no teaching example that imports only `t3toolbox.fitting`: the doctests of the three cases,
  the weight trap, the probe kind and the bad-scaling walk are in the backend modules and import
  `t3toolbox.backend.jets`. P19 should carry a frontend example.
- `docs/api_reference.rst` (issue 3).
