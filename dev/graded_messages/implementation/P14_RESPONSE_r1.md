# P14, response to review round 1

Implementer, 2026-11-21, answering `P14_REVIEW_r1.md` and the coordinator's rulings on it. Nothing is committed.

## 1. (Must fix) The packed probe `y`'s padding: done

`LeastSquaresOracle.query` now unpacks a packed probe `y` before reducing it. `y_block_sumsq`, and `f_resolution`
through it, read the real part only:
- it calls `ut3_operations.unpack_vectors(y, shape)`, with `shape` the uniform frame's static shape (`frame[4]` on a
  frame, `frame[2]` on a point, from `frame_form`), which is traceable;
- this matches the engine, which masks the packed `y` on its mode axis: the same real part;
- apply's and entries' `y` have no mode axis, so no padding, and are untouched.

**Test.** `TestQuery.test_the_uniform_layout_after_pack_batch` now puts garbage (1e3) in the mode padding of every
packed form:
- the probe and apply samples;
- the probe `y`.

It asserts `f`, `r_block_sumsq`, `y_block_sumsq` and `f_resolution` against ragged to 1e-12, and the real blocks of
`g`. With the fix reverted, 3 of its subtests fail.

## 2. (Should fix) `eps` pinned to the result dtype: done

`TestTheResolutionDtype`: on float32 inputs, `f` is float32 and `f_resolution` equals the formula with
`finfo(float32).eps` (to 1e-5 relative). The reviewer's float64-`eps` mutant now fails it.

## 3. (Should fix) Nothing calls `check_batch`, in every backend module: done

`tests/test_jets.py::test_no_backend_module_calls_a_checker` now asserts, for **every** backend module, that no call
in it is `check_batch` (as an attribute or a name). A planted `o.check_batch(b, s)` in `optimizers.py` fails it.

## 4. Runs

- `tests/test_least_squares_oracle.py` and `tests/test_jets.py` pass (59 tests, 534 subtests).
- The three fixes' mutants (revert the unpack, a float64 `eps`, a planted `check_batch` call) are each killed, in a
  scratch copy that has since been deleted.
- **The full gate at `-n 2`**: all six steps pass (suite 1852 passed); lowest MemAvailable 3479 MB.
