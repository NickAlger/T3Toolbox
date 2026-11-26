# P4b: the input jets and the dense functions — handover

Slice P4b of `PART1_PLAN.md` (section 5.3; the forms in section 2.4; entries 118 to 120). Written by the
implementer, 2026-11-19, against `graded-messages` at `45a92b1f` (P4a committed). Nothing is committed. **Updated after
review round 1** (`P4b_REVIEW_r1.md`, NOT YET on one should-fix finding; answered in `P4b_RESPONSE_r1.md`):
`index_to_one_hot` raises on a numpy index outside the grid.

## 1. What changed

**`t3toolbox/backend/jets.py`.**
- `affine_jet(ww, *directions)` and `index_to_one_hot(index, shape)` are new public names, in `__all__` under
  "The input jets".
- A private `_affine_jet_of_one_array` builds one input jet.
- The module docstring gains one paragraph, the group of the input jets, and says "Four groups" where it said
  three. The rest of the docstring is P4d's.
- `itertools` is imported.

**`affine_jet(ww, *directions)`:**
- slot `(0, ..., 0)` is `w`, slot `e_j` is `p_j`, and every other slot is a zero that is made, not `0 * w`;
- the dtype is the promotion of the point and the directions;
- it returns a tuple for a tuple or a list, and a packed array `(d,) + (2,) * m + W + (N,)` for a packed one;
- with no direction it returns `ww` itself;
- it checks only that each direction has the form and the shapes of `ww`.

**`index_to_one_hot(index, shape)`:**
- one-hot vectors `W + (Ni,)` per mode, as a tuple, built by comparing the index with an `arange`;
- numpy for a numpy index and jax for a jax index, in the default float dtype (`xnp.zeros(()).dtype`);
- a negative index wraps, as numpy's indexing does. **After review round 1:** a numpy index outside `[-N, N)`
  raises an `IndexError` naming the mode and the values; a jax index (possibly a tracer) is not checked and gives
  a zero vector out of range;
- it raises on a non-integer dtype or on a count of rows other than `len(shape)`.

**`t3toolbox/backend/probing.py`.**
- `dense_probe(T, ww, order=())` takes input jets, `J_in_i + W + (Ni,)`, and returns `len=d`, `F + W + C + (Ni,)`.
- `_dense_jet(T, ww, order, open_mode, name)` is rewritten:
  - it reads `order` with `contractions._read_order`, then checks the structure: the number of vectors, at least
    `m + 1` axes per input, jet axes of at least one slot, the same `W` once the jet axes are read, and each mode's
    length;
  - it contracts the tensor one mode at a time, keeping the partly contracted polynomial's coefficients by
    multi-index (a dict). Contracting mode `j` multiplies by the input jet `w_j(s)` through explicit loops over the
    pairs of multi-indices whose sum lies in the box, one einsum per pair (`_dense_contract_one_mode`);
  - input slots beyond the box are never read; the slots no product reaches are exact zeros;
  - `d = 1` with its mode open gives the tensor broadcast over `W`, as before;
  - it never calls `jet_contract` and shares nothing with the sweeps.
- `_dense_contract` (replaced by `_dense_contract_one_mode`) and `_check_pp_and_order` are deleted: private, and
  no other caller.

**`t3toolbox/backend/apply.py`:** `dense_apply(T, ww, order=())`, over `_dense_jet`.

**`t3toolbox/backend/entries.py`:** `dense_entries(T, index)`, indexing only; `pp` and `order` are gone (entry 120).
Its docstring says how to get the jets off the grid and shows it: `dense_apply` at
`affine_jet(index_to_one_hot(index, shape), pp)`.

**Docstrings (run and passed).** Each says what a jet of a dense tensor is in several variables:
- `dense_probe` has a doctest of a line (`affine_jet(ww, pp)`) and of a curve of three slots, and the gotcha that an
  integer `order` raises;
- `dense_apply` has a doctest of two directions at `(1, 2)`, checked against hand-written sums;
- `dense_entries` shows the recipe of entry 120.

**`t3toolbox/backend/sampling_definitions.py`.**
- `_dense_sample` builds the two-slot input jet itself, `xnp.stack([w, p])` per mode, and passes `(order,)`. The
  module still does not import `jets`; its independence test is unchanged and passes.
- `_z` above order 0 for entries goes through `_dense_sample`, the apply at the unit vectors the module already
  builds. That is what the old `dense_entries(T, index, pp, order)` computed.

- `_einsum` passes `optimize='greedy'` on the jax path too, as it does on numpy. This fixes the CI floor's hang,
  which Claude diagnosed: on jax 0.4.30 a bare `jnp.einsum` defaults to `'optimal'`, an exhaustive path search
  exponential in the operand count, and `_tensor` contracts a whole four-site network in one einsum. The
  docstring says why, so that nobody "restores" the bare call. It is the only change of the house convention, and
  the library's other bare `jnp.einsum` calls (two or three operands) stay.

**`t3toolbox/backend/data_preparation.py`:** the four calls with `pp` in its doctests become
`dense_probe(T, affine_jet(ww, pp), (order,))` and the apply twin (P4d rewrites the rest).

**Where the reader of `order` lives (point 1 of the brief for P4b).** The dense functions call
`contractions._read_order` directly: `probing` already imports `contractions`, and `apply` and `entries` reach it
through `probing._dense_jet`. `jets` does not re-expose it. No function of `jets` takes a jet shape before P4d, which
will import it from `contractions` (allowed: `contractions` imports only `common`). So there is one implementation,
and `test_import_graph.py` passes unchanged.

## 2. The tests

**`tests/test_sampling_definitions.py`.** `TestDenseFunctions` is rewritten (17 tests); `TestInputJets` is new (4
tests). The module docstring's item 8 is updated.

*No jets:*
- `dense_probe` against hand-written einsums, with `W`, `C` and no stack; `order=()` given or not;
- one mode: the tensor broadcast, and with jets zero above slot 0;
- `dense_apply` equals the probe contracted with the vector of its open mode;
- `dense_entries` is indexing: a negative index, a nested list, a non-finite entry elsewhere, `W = ()`.

*The jets against the definition.* `dense_probe` and `dense_apply` at input jets are compared with the calls with no
jets at the input vectors evaluated on a grid of roots of unity, coefficients recovered by an FFT. The grid's degree
bound is the sum of the inputs' degrees. Shapes are checked exactly and values to 1e-11, at:
- a line at `(1,)`, `(2,)`, `(3,)`, `(5,)`;
- two directions at `(1, 1)`, `(1, 2)`, `(2, 3)`;
- a curve of three slots at `(1,)`, `(2,)`, `(4,)`;
- modes of different input jet sizes, below, at and above the box, including a mode of size 1 on every axis and
  the jet matrix's `(3, 2)`, `(1, 1)`, `(2, 3)` at `(1, 2)`;
- `m = 3`, at `(1, 1, 1)`;
- with and without a `C` stack.

*Other jet properties:*
- NaN in every input slot beyond the box gives a result bit-equal to the inputs cut to the box;
- slot 2 is a sum of Taylor coefficients, not a derivative;
- slot 0 is the call with no jets, and the slots above the degree are exactly zero;
- the recipe of entry 120: `dense_apply(T, affine_jet(index_to_one_hot(index, shape), pp), (n,))` equals
  `sampling_derivatives.dense_entries_derivatives / t!` at each sample, at `n` = 1, 2, 4, and its slot 0 is
  `dense_entries`.

*Structure:*
- an integer, a list or a float as `order` raises `TypeError` naming the function; a negative entry raises
  `ValueError`; a numpy integer is accepted;
- wrong shapes raise: a mode size, sample stacks, an input without its jet axis, a jet axis of size 0, too few
  axes, too many index rows or vectors;
- the signatures are `(T, ww, order=())` and `(T, index)`, in `__all__`;
- numpy in, numpy out, float32 kept (at `(2,)` and `(1, 2)`); jax in, jax out.

*`TestInputJets`:*
- the slots of `affine_jet` at `m = 1, 2, 3`: slot 0 is the point, slot `e_j` the `j`-th direction, zero elsewhere,
  and the zero is not `0 * w` (a NaN point leaves the `(1, 1)` slot 0);
- no direction returns `ww` itself; a list gives a tuple;
- packed vectors give the packed array, equal to stacking the ragged result; dtype promotion; its three errors;
- `index_to_one_hot` is the rows of the identity (negative indices wrap); `dense_apply` at them equals
  `dense_entries` exactly; a non-integer dtype and a wrong count of rows raise. **After review round 1:** a numpy
  index outside `[-N, N)` raises an `IndexError` naming the mode and the values, the ends of `[-N, N)` pass, and a jax
  index out of range gives zeros.

**Other test files:**
- `tests/test_dispatch.py`: `test_jit_dense_sampling_functions` covers `dense_probe` and `dense_apply` at `()`, at a
  line at `(3,)`, and at inputs of jet sizes `(2, 3)`, `(1, 1)`, `(3, 2)` at `(1, 2)`, plus `dense_entries`. It runs
  float64 and float32, jit and eager, and each argument in turn closed over as numpy.
  `test_jit_input_jet_helpers` (new) covers `affine_jet` (a line, two directions, packed), `index_to_one_hot`
  (jax's default float), and a closed-over numpy direction.
- `tests/test_jax_absent.py`: the dense functions at a hand-built line; `affine_jet` (ragged and packed) and
  `index_to_one_hot` without jax.
- `tests/test_data_preparation.py` (its `dense_sample` helper and one call) and `tests/test_jets.py` (two calls): the
  calls with `pp`, through `affine_jet` (and `index_to_one_hot` for entries off the grid).

**Verified by script first** (my scratchpad, `check_dense.py`, not in the tree):
- The new functions against plain `np.einsum` at roots-of-unity nodes plus an FFT: 336 cases, `m = 1, 2, 3`, random
  input sizes per mode, `d` = 1 to 4, `W` and `C` stacks; worst relative error 3.8e-15.
- Against the P2 dense functions (from `git show HEAD`) on line inputs, probe and apply, and entries off the grid
  against the old `_dense_jet` at `eye(N)[index]`: largest absolute difference 2.5e-14.
- `order = ()` equals P2's order 0.

**Mutations** (a scratch copy of `t3toolbox/` and `tests/`, `python -B`, each mutant against the dense tests, and
the dispatch tests for the dispatch mutants; the copy deleted): **23 of 24 caught.** Caught:
- in `_dense_jet`: the top slot dropped, the contracted axis taken by mode number, the missing slots not zero,
  terms overwritten instead of added, no broadcast at `d = 1`;
- the checks: `order` not read through `_read_order`, the size-0 and `W` checks removed;
- dispatch: read off `T` only; a host-numpy einsum;
- `dense_apply` leaving mode 0 open; `dense_entries` without its `moveaxis`;
- in `affine_jet`: the zero slot as `0 * w`, every direction slot taken from the first direction, the packed jet
  axes put in front, a host-numpy stack, a tuple returned with no direction, the shape check dropped;
- in `index_to_one_hot`: no wrap, float32, host numpy, the dtype check dropped;
- `sampling_definitions`' two slots swapped.

**Not caught, and equivalent:** the input's slots not clipped to the box in the loop. The extra multi-indices all
give offsets beyond the box and are skipped by the check that follows before any slot is read, the same as in P4a.

**Runs** (single process, as the coordinator asked):
- `tests/test_sampling_definitions.py`: 65 passed, 4,103 subtests, 147 s.
- The two dispatch tests: 2 passed, 48 subtests, 23 s.
- `test_import_graph`, `test_jax_absent`, `test_jets`, `test_data_preparation`, `test_manifold` and
  `test_tucker_tensor_train`: 221 passed, 24,947 subtests, 83 s.
- The module doctests of `probing`, `apply`, `entries`, `jets`, `data_preparation`, `sampling_definitions`,
  `manifold` and `uniform_manifold`: 94 passed.

## 3. The gate

The whole gate on the final tree (the `_einsum` fix included), nothing else running:
`PATH=/home/nick/miniconda3/envs/t3toolbox/bin:$PATH dev/graded_messages/implementation/run_gate.sh --detach -n 3 -l <scratchpad>/implementer/gate_P4b`,
then `wait`. The summary, copied:

    step                result counts                                                      seconds   peak_MB  minavl_MB
    suite               PASS   1657 passed, 5 skipped, 1 xfailed, 1 warning, 68234 subtests passed   1492.2      9844       2361
    doctest-modules     PASS   312 passed                                                     48.9      1910      10074
    doctest-quickstart  PASS   1 file(s), 0 with failures, 0 failed example(s)                 2.7       284      11721
    doctest-pages       PASS   48 file(s), 0 with failures, 0 failed example(s)               14.3       364      11650
    examples            PASS   14 examples, 0 failed                                           386       639      11340
    sphinx              PASS   0 warning/error line(s)                                      1027.4       635      11185

**No failure.** Against P4a's gate:
- The suite has 10 more tests (1,647 to 1,657): `TestDenseFunctions` and `TestInputJets` have 21 where
  `TestDenseFunctions` had 12 (+9), and `test_jit_input_jet_helpers` is new (+1). The subtests go from 68,223 to
  68,234.
- The same 5 skips (the slow tier), the same 1 warning, the same 1 xfailed (row 5.5).
- The module doctests rise from 310 to 312: `affine_jet` and `index_to_one_hot`.
- The suite's peak memory is 9.8 GB, with 2.4 GB available at its lowest.

**After review round 1**, single process: `TestInputJets`, `TestDenseFunctions`, `test_jax_absent.py` and
`test_data_preparation.py` gave 54 passed; the two dispatch rows 2 passed; the module doctests of `jets`, `entries`,
`apply` and `probing` 21 passed. No full gate, as ruled.

The sphinx build was deleted after reading. No process, lock or file of the gate is left in the repository.

**The floor run, single process**, after the `_einsum` fix (the CI floor leg had hung on this file), on the
`tttt` env (python 3.9, numpy 1.22.4, jax 0.4.30):
`PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD timeout 1500 /home/nick/miniconda3/envs/tttt/bin/python -B -m pytest tests/test_sampling_definitions.py -p no:cacheprovider -q`

| Env | Result | Wall time |
| --- | --- | --- |
| `tttt` (the floor) | 65 passed | 107 s |
| `t3toolbox` (python 3.11, numpy 2.4.6, jax 0.10.2) | 65 passed, 4,103 subtests | 132 s |

The module doctests of `sampling_definitions.py` pass (37) after the fix.

## 4. Departures from the plan; choices

**Departures:** one change outside the plan's P4b, decided by Claude as minor: `sampling_definitions._einsum`
passes `optimize='greedy'` on jax (section 1; the cause of the CI floor's hang in
`TestKp::test_jax_against_finite_differences`). Otherwise none in what is built. Where the plan says "the private reading of `order`" is in `jets`, it is
`contractions._read_order`, as point 1 of the brief for P4b allowed (section 1 says why).

**Choices:**
1. **`_dense_jet` contracts one mode at a time.** It keeps the partial polynomial's coefficients by multi-index; the
   product is taken by loops over pairs of multi-indices whose sum is in the box. The cost per mode is the box's
   slots times the input's. The alternative, one einsum per combination of every mode's multi-indices (exponential
   in `d`), was not taken.
2. `order` is the third parameter of `dense_probe` and `dense_apply` with the default `()`; it can be given by
   position or by keyword.
3. `affine_jet` with no direction returns `ww` itself (the plan: "returns `ww` unchanged"). Its zero slots are made
   as zeros of the promoted dtype, so a non-finite point does not leak into them.
4. `index_to_one_hot` compares the index with an `arange`, rather than indexing an identity, so its memory is that
   of its output (`W x N`), not `N x N`. Its index semantics are numpy's, as in `dense_entries`:
   - a negative index wraps;
   - a numpy index outside `[-N, N)` raises an `IndexError`;
   - a jax index is not checked, and gives zeros out of range.

   The first version gave zeros for any out-of-range index. Review round 1 and Claude's ruling changed it: a zero
   vector would turn a bad index into plausible data.

   A non-array index is converted with `np.asarray`.
5. The tests of `affine_jet` and `index_to_one_hot` live in `tests/test_sampling_definitions.py` (`TestInputJets`),
   where the plan's P4b test list puts them. P4d's `tests/test_jets.py` may be their natural home later.
6. `dense_entries` keeps `xnp.asarray` on `T` and `index` (a nested list as an index still works).

## 5. Found outside the slice

1. The one-mode jets helper is the only jets helper in `probing`; P4c's reference module will call these dense
   functions with input jets of its own, as the plan says.
2. `test_dispatch.py`'s dense test now makes 14 jits per function and dtype, about 20 s in all.
