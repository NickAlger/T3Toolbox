# P3: response to review round 1

To `P3_REVIEW_r1.md` (NOT YET; four should-fix, twelve minor), with the coordinator's decisions on findings 2 and
5 and on the packed probe array. Every finding is accepted; one part of finding 3 is answered differently from
its wording, and said so there. 2026-11-17.

## Should fix

**1. `check_sample` accepted `(ww, None)` at order 1 and above. Accepted.**
`check_sample` now raises before anything else when `order >= 1` and `pp is None`:
`at order 2 the sample of 'apply' needs the directions pp: it is the pair (ww, pp); got (ww, None)` (with `index`
for entries). Test: `TestCheckers.test_check_sample_refuses_a_missing_pp` (three kinds, two orders, the pair as a
tuple and as a list).

**2. The `scales` checker. Accepted, as decided by the coordinator.**
A public backend checker `check_coordinate_scales(scales, shape)` in `backend/data_preparation.py`, in `__all__`
(signature and docstring summary in the handover, section 1). It refuses anything that is not a tuple or a list
(one array per mode as a single array, or scales packed to `(d, N)`), a wrong count, an array of the wrong length
for its mode (a length of 1 among them), a wrong rank (a number, two axes), a list in place of an array. It looks
at no value: a zero, negative, infinite or `nan` scale passes, and the docstring says what a zero scale does
(`rescale_coordinates` zeroes that coordinate of the inputs and divides the probe data there by zero;
`t3_unscale_coordinates` zeroes every entry of the tensor with that coordinate).
- The two `unscale_coordinates` methods call it; the private `tucker_tensor_train._check_coordinate_scales` is
  gone.
- `rescale_coordinates`, `t3_unscale_coordinates` and `ut3_unscale_coordinates` stay check-free. Their docstrings
  name the checker and say what wrong `scales` silently do (a short list drops a mode; a length-1 scale is
  broadcast).
- Tests: `test_check_coordinate_scales_passes_good_scales_and_raises_on_each_wrong_form` (nine wrong forms);
  `test_check_coordinate_scales_looks_at_no_value`; `test_the_backend_functions_do_not_check_scales` (pins the
  two silent behaviours the docstrings state); `test_the_methods_raise_on_scales_of_the_wrong_form` (both classes,
  the nine forms, and with the backend checker patched to show the method goes through it);
  `test_no_backend_module_calls_a_checker` now covers the new name too; it runs under `jit`
  (`test_the_checkers_run_under_jit`) and without jax.
- **The new public name is `check_coordinate_scales`**, to be recorded for the maintainer.

**3. `rescale_direction` with a traced `length`. Accepted.**
No `float(...)` on the length: the powers are `xnp.asarray(length, dtype) ** xnp.arange(order + 1, dtype)`, and the
directions are multiplied by `xnp.asarray(length, dtype)`, each in the dtype of the array it multiplies. The
dispatch is one `tree_contains_jax((sample, y, length))`. The signature comment and the docstring say that
`length` is a number or a 0-dimensional array and may be traced.
- Tests: two cases in `test_jit_jets_and_data_preparation` with the length an array argument (traced under `jit`;
  closed over as numpy in the mixed sub-case); `test_one_dispatch_over_all_the_inputs_of_a_call` (numpy data, an
  eager jax length: the directions and the data come back jax); `test_the_length_may_be_a_zero_dimensional_array`
  on numpy.
- **One point answered differently from "every output is jax".** The inputs `ww` or `index` are not computed
  with, and `rescale_direction` returns them as the object given (handover choice 7; `index` is an integer array
  that nothing here should convert). So with numpy inputs and a jax length, the returned sample is
  `(the numpy ww as given, jax pp)`, and the data are jax. Everything the function computes is of one kind; the
  docstring says so. If the reviewer wants the pass-through converted as well, it is one line, at the cost of the
  "as given" promise.

**4. The uncaught mutation. Accepted.**
`TestFrontendDualJetToJet.test_a_small_nonzero_weight_passes_in_safe_mode`: a weight of `1e-8`, and of `1e-150`
(whose square is still a float64), in float64, passes the frontend check in safe mode and gives the backend's
numbers exactly, for probe and apply. The reviewer's mutant
(`n_zero = int(np.sum(np.abs(np.asarray(weight)) <= 1e-8))`) now fails that test; a mutant with a tolerance of
`1e-200` on the divisor fails it too.

## Minor

**5. A weight whose square underflows. Accepted (the coordinator's decision: the check is on the divisor).**
The frontend check is now on `weight**2` as the backend computes it, in the precision of the computation, still
exact. Both functions take it from one private backend helper, `jets._squared_weights(jet, weight, xnp)`: the
backend divides by its result and the frontend tests its result against zero, so "the divisor as the backend
computes it" holds by construction. The message reads `the squared weight is zero in 1 entry ...`. The docstrings
of both `dual_jet_to_jet` say it. Test: `test_a_weight_whose_square_underflows_is_refused_like_a_zero` (a float32
weight of `1e-30`; a float64 weight of `1e-30` against a float32 dual jet; `1e-200` in float64; one entry of one
mode's row for probe; and the float64 `1e-30` against a float64 dual jet passing). A mutant that checks the
weight in place of the divisor fails it.

**6. Two `check_sample` messages that did not name the cause. Accepted.**
Where the inputs fail and the sample looks like that of the other form, the message goes on:
`... At order 0 the sample is the inputs ww alone, with no pp; this sample looks like the pair (ww, pp)` (also for
`index`), and `... At order 1 the sample is the pair (ww, pp); this sample looks like ww alone` (two modes, `ww`
alone at order 1 and above). A wrong sample that does not look like the other form gets no such sentence. Test:
`test_check_sample_names_the_cause_when_the_form_is_that_of_the_other_order`.

**7. `equalizing_weight` on all-zero norms; 0-dimensional outputs. Accepted.**
The numpy division is inside `np.errstate(divide='ignore')`: `inf` in every entry, no warning, as on jax; stated in
the docstring. A 0-dimensional input returns a 0-dimensional array of the input's module. The same for
`jet_to_dual_jet`, `dual_jet_to_jet`, `derivatives_to_taylor`, `taylor_to_derivatives` and
`derivative_weight_to_taylor_weight` on a 0-dimensional numpy input. Tests:
`test_all_zero_norms_give_infinity_silently_and_a_number_gives_an_array` (with warnings as errors),
`test_a_zero_dimensional_input_gives_an_array`.

**8. The false sentence about mode padding with packed scales. Accepted, by making it true.**
`ut3_unscale_coordinates` now sets the scales to zero beyond each `Ni` from `shape` (a host numpy mask), whatever a
packed `scales` holds there. The docstring says exactly what holds: the mode padding comes out zero for ragged and
packed scales alike; the rank padding is multiplied like the rest (zero stays zero, garbage stays garbage); the
real entries depend on no padding. Test: `test_packed_scales_are_the_ragged_ones` now runs with garbage in the
packed scales beyond `Ni` and in every padded slot of the supercores, and asserts the zero mode padding and, within
each `Ni`, every entry (rank padding included) equal to the input's times its scale.

**9. Dispatch per array inside a probe tuple. Accepted.**
Every function of the two modules now chooses numpy or jax once, with `tree_contains_jax` over all the arrays of
the call, and hands `xnp` down: `jet_to_dual_jet` and `dual_jet_to_jet` (jet and weight), `derivatives_to_taylor`
and `taylor_to_derivatives` (all modes), `prepare_derivative_data` (data, weight and validation data together),
`data_block_norms`, `rescale_direction` (sample, data, length), `rescale_coordinates` (sample, data, scales),
`t3_unscale_coordinates` (Tucker cores and scales), `ut3_unscale_coordinates` (as before). Test:
`test_dispatch.py::test_one_dispatch_over_all_the_inputs_of_a_call`, twelve cases with one jax array among numpy
ones.

**10. A jax integer weight with a numpy float64 jet, x64 off. Fixed (one line, following row 5.6).**
What happened: the result dtype was the lowest floating precision among the arrays, here float64 from the numpy
jet alone (an integer weight has no floating dtype), and that dtype was then asked of jax, which has no float64
with x64 off: a warning, and float32 anyway. The same would happen for any numpy float64 array meeting a jax array
that does not lower the dtype. The fix is in `_lowest_precision_dtype`, which now takes `xnp`: on jax, jax's own
floating dtype (`xnp.result_type(float)`: float32 unless x64 is on) joins the candidates, so the result is no more
precise than jax can hold. That is row 5.6 (jax takes precedence; the lower precision wins). Test:
`test_dispatch.py::test_a_numpy_float64_array_with_a_jax_array_when_x64_is_off` (six calls with warnings as errors,
float32 jax out).

**11. The asterisk of row 2.5. Accepted.**
The module docstring of `jets` now says: with no weight given the weight is ones, the plain sum of squares of the
Taylor coefficient residuals, and that default may change in a later release. The `weight` entries of
`jet_to_dual_jet`, both `dual_jet_to_jet` and `check_weight` carry it. `prepare_derivative_data` says that its own
`None` (ones on the derivatives) is not the fit's default, which is ones on the Taylor coefficients and may change.

**12, 13. Cross-references that render as plain text; a frontend-only teaching example. Left for P19,** as
instructed, and listed in the handover under "For P19".

**14. The functions that take `kind` assume the ragged type. Accepted.**
One line each in `prepare_derivative_data`, `check_data` (`check_sample` said it already), `data_block_norms`,
`rescale_direction`, `rescale_coordinates`, and in both module docstrings.

**15. `_KINDS` and `_require_kind` duplicated. Accepted.** One copy, in `jets`; `data_preparation` imports
`_require_kind` beside `_lowest_precision_dtype`.

**16. `check_weight` and a transposed probe weight when `d == order + 1`. Accepted.** The docstring says a shape
check cannot see it and that the mode axis is the first. Test:
`test_check_weight_cannot_see_a_transposed_square_weight` pins the limit.

## The packed probe array (the handover's issue 1; the reviewer's question for P11)

As decided: no signature changes, and no guard that raises. The module docstring of `jets` has a paragraph "The
ragged type only" with the trap spelled out as measured (a packed array `(d,) + T + ...` passed whole is read as
one array: with a weight `(d,) + T` the modes come out reversed and the orders are not turned, with no weight the
modes reversed, and `derivatives_to_taylor` divides the mode axis by `t!`: a broadcast error unless
`d == order + 1`, a wrong answer if it is; `tuple(packed)` is right). Each of the four functions, and the frontend
`dual_jet_to_jet`, says it. `jet_to_dual_jet` has a doctest: `tuple(packed)` gives the right answer, the whole
array a wrong one without an error. Test: `TestDualJet.test_a_packed_probe_array_goes_in_mode_by_mode` (the four
functions through `tuple(packed)` equal the ragged call exactly; the whole array gives precisely the wrong result
the docstrings describe, so the description cannot drift).

## What was run after the changes

- The slice's test files and the module doctests of the touched modules, with the bytecode cache removed and
  `python -B`: 77 passed (532 subtests) in `test_jets.py`, `test_data_preparation.py`, `test_import_graph.py`,
  `test_jax_absent.py`; 117 doctests in the five touched modules; the five slice tests of `test_dispatch.py`
  (149 subtests).
- 20 mutations in a scratch copy (`python -B`, the tree untouched), all caught: the reviewer's mutant; the check
  on the weight in place of the divisor; a tiny tolerance on the divisor; `pp is None` accepted; each of the two
  hints removed; the dual jet dispatching on the weight alone; no jax cap on the dtype; a scalar out for a
  0-dimensional input; `equalizing_weight` warning, and returning a scalar; packed scales not zeroed beyond `Ni`;
  the length left out of the dispatch; `float(length)`; the count of `scales` unchecked; a length of 1 allowed;
  the rank unchecked; a backend function calling `check_coordinate_scales`; either method without its check.
- The whole gate on the final tree: in the handover, section 3.

## Not accepted

Nothing. One point is answered differently from its wording: the pass-through of `ww` / `index` in
`rescale_direction` (finding 3, above).

## Addendum: `fitting.rescale_coordinates` checks the scales (the coordinator's decision after this response)

The point left open in the handover (a frontend user who passed wrong `scales` to `fitting.rescale_coordinates`
got no error there, only later at `x.unscale_coordinates(scales)`) is decided: the frontend is where checks are
enforced, as with `dual_jet_to_jet`.

- **`t3toolbox.fitting.rescale_coordinates(kind, order, sample, y, scales)` is now fitting's own function**, with
  the backend's name and parameters. For `'probe'` and `'apply'` it reads the mode sizes off the sample (the last
  axis of each `ww[i]`), calls the backend `check_coordinate_scales(scales, shape)`, then the backend
  `rescale_coordinates`. For any other kind it calls the backend function, which raises. The checker is **not**
  re-exported: the plan's list stays ten names, of which nine are the backend objects and two
  (`dual_jet_to_jet`, `rescale_coordinates`) are fitting's own functions that call the backend ones.
- **Structural, so always.** Confirmed by test: the check raises in safe mode, inside `safety.unsafe()`, and under
  `jax.jit` (shapes are static), also `jit` inside `safety.unsafe()`.
- **Tests.** `test_data_preparation.py`,
  `test_the_frontend_rescale_coordinates_checks_the_scales_and_gives_the_backend_s_numbers`: the nine wrong forms
  (a wrong count, a wrong length for a mode, a length of 1 among them) raise for probe and apply at orders 0 and 2,
  in both safety modes; good input gives the backend's arrays exactly in both modes; scales right for another
  tensor are refused (the mode sizes come from the sample); `'entries'` and an unknown kind raise the backend's
  error; the function goes through the backend checker (patched). `test_dispatch.py`,
  `test_the_frontend_rescale_coordinates_under_jit`: jax out under `jit` with the backend's numbers, numpy in
  numpy out, and the three wrong forms raise at trace time, in both modes. `test_jets.py`,
  `test_the_names_of_t3toolbox_fitting`, now says: nine names are the backend objects; the two own functions have
  the backend's parameters and live in `t3toolbox.fitting`; no checker and no `ut3_unscale_coordinates` is an
  attribute of `fitting`. Five mutations of the new function in a scratch copy, all caught (no check; a check only
  in safe mode; a check skipped under a trace; the mode sizes read off the scales; wrong numbers).
- **Docstrings.** The function's own (with a doctest of the refused length-1 scale); the section on preparation in
  the module docstring of `t3toolbox.fitting`. The backend `rescale_coordinates` already names the checker.
- **Finding 3, the pass-through of `ww` / `index` in `rescale_direction`:** kept as it is, documented (decided).
- **Runs** (the bytecode cache removed first): in the handover, section 3.

## Addendum 2: the four functions take `kind` (the maintainer, 2026-11-17, 12:02)

"A kind argument is acceptable and sometimes even desirable for functions whose output differs based on kind."
This supersedes the section "The packed probe array" above and handover choice 1.

- **The four signatures**, in `backend/jets.py`:

  ```python
  def jet_to_dual_jet(kind, order, jet, weight) -> typ.Any
  def dual_jet_to_jet(kind, order, dual_jet, weight) -> typ.Any
  def derivatives_to_taylor(kind, order, jets) -> typ.Any
  def taylor_to_derivatives(kind, order, jets) -> typ.Any
  ```

  `weight` was a required argument and stays one (no default invented; `None` is passed for no weight).
  `derivative_weight_to_taylor_weight(weight, order)` is unchanged. The frontend's own
  `fitting.dual_jet_to_jet(kind, order, dual_jet, weight)` has the same signature; the re-exports follow.
- **The type is decided by `kind`, never by the object.** One private helper, `jets._arrays(kind, value)`, gives
  the arrays: for `'probe'` `[value[i] for i in range(len(value))]`, for the others `[value]`; `_of_type_of_z`
  returns a tuple for probe. `_is_per_mode` is gone.
- **A packed array.** With `kind='probe'` an array `(d,) + T + ... + (N,)` is taken mode by mode along its first
  axis and gives exactly what `tuple(packed)` and the ragged list give, returned as a tuple, one array per mode
  (checked on numpy and under `jit`). No code for the uniform layout. The trap is gone, and with it its
  description in the docstrings and the test that pinned the wrong result.
- **A mismatched type** is not guarded against, and the docstrings say what happens: a tuple given with
  `kind='apply'` is made into one array: a `ValueError` from numpy where the modes differ in size; where they do
  not, the mode axis is read as the order axis, which is an error from broadcasting or, with no weight, the modes
  turned and no error. An unknown kind raises a `ValueError` (the functions need the kind in order to run).
- **Callers changed:** `prepare_derivative_data` (through the private `_scale_orders(kind, ...)`),
  `_squared_weights(kind, ...)`, and the frontend `dual_jet_to_jet`; the doctests of the module, of the four
  functions and of the frontend function (re-run and pasted); `tests/test_jets.py` (60 calls),
  `tests/test_dispatch.py` (16 calls) and the script of `tests/test_jax_absent.py` (6 calls). No other caller
  exists: `t3toolbox`, `tests`, `docs`, `examples`, `README.md` and `/home/claude/t3polynomial` were searched for
  the four names, the import forms of the module and its aliases.
- **Tests.** `TestDualJet.test_kind_decides_the_type_and_a_packed_probe_array_is_taken_mode_by_mode` (a list, a
  tuple and a packed array give the same tuple in the four functions, `prepare_derivative_data` and the frontend
  function; the values against the definition written out on the packed array; one mode still gives a tuple of
  one); `test_an_object_of_the_other_type_than_kind_says_is_not_guarded_against` (pins what the docstrings say);
  `test_the_four_functions_take_kind_then_order`; four packed cases in `test_jit_jets_and_data_preparation`. Seven
  mutations in a scratch copy, all caught in the end (one, the return type taken from the count of arrays, was
  missed at first and is now caught by the one-mode case).
- **Runs:** in the handover, section 3.

