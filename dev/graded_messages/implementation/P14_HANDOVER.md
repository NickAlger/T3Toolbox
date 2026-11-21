# P14: the oracle and the batch — handover

Slice P14 of `PART1_PLAN.md` (additive), built to `P14_DESIGN_NOTE.md`, `P14_DESIGN_REVIEW.md` and the coordinator's
rulings 1 to 6. Written by the implementer, 2026-11-21, against `graded-messages` at `3769bad4`. Nothing is
committed.

**The maintainer has not yet seen `keep=` and `frame_form`.** Each is easy to rename:
- `keep` is one keyword with one use site (`LeastSquaresOracle.query`);
- `frame_form` is one function with two call sites (the engine and the oracle).

## 1. What changed

**The engine** (`backend/sampling_variables.py`):
- **`frame_form(frame) -> (layout, call)`** is public, in `__all__`, with a doctest. Its values are
  `'ragged' | 'uniform'` and `'manifold' | 'corewise'`.
  - It reads the layout first (`is_ndarray(frame[0])`), then the count. A count of four is a ragged frame or a uniform
    point.
  - It raises the engine's existing `ValueError` otherwise.
  - The engine now reads its call type through it (ruling 2).
- **`keep=None`** (ruling 1):
  - names must be `HeldVariables` slots: an unknown name raises a `ValueError`, and a bare string a `TypeError`;
  - when `keep` is given (even `()`), the return has one more element, a `HeldVariables`;
  - the record is the `held` passed in (its arrays as given) plus each kept name the closure computed. A kept name
    the call did not compute is absent: nothing is computed only to be kept. On the uniform layout the kept values
    are masked like outputs;
  - the filled slots come from the closure, so they are static under jit (asserted in `test_dispatch`). Documented in
    the docstring.

**The oracle and the batch** (`backend/fitting.py`, beside the old kinds, in `__all__`):
- **`LeastSquaresBatch`** is section 2.2's dataclass: `sample`, `y`, `weight`, `minibatch_scale`, `drawn_indices`. It is
  a registered pytree with every field a leaf, and `==` and hashing are refused (an array-carrying record).
- **`LeastSquaresOracle(kind, order)`** is value-hashed (`ValueHashedFields`).
  - **Construction:**
    - `order` is required for every kind (O-44), so a missing one raises the dataclass's `TypeError`;
    - a non-tuple order raises a `TypeError`;
    - the entries kind with jets raises a `ValueError` naming the apply kind at the one-hot jet (O-46);
    - an unknown kind raises.
  - **`query(frame, batch, want, Vc, variations, core_directions, held)`:**
    - one engine call with `keep=('xi', 'mu', 'nu', 'eta', 'r')`; it returns `(values, held)`;
    - **on a corewise point** (`frame_form`): `core_directions` if given, else `variations` renamed; both given must be
      the same direction, the same object or the same arrays leaf by leaf by identity (`_same_leaves`), else a
      `ValueError` (ruling 3);
    - **`y_block_sumsq`** is `data_block_norms(...) ** 2`;
    - **`f_resolution`** is `eps (|w y| sqrt(2 f) + f)` from the unscaled batch values, then times the scale once
      (ruling 6). `eps` is from `f`'s dtype, and asking for it adds `f` to the engine call;
    - **the scale (O-21)** multiplies `f`, `g`, `JtJp`, `Jp_norm_sq`, `Kp` and `f_resolution`, and nothing else.
  - **The other methods:**
    - `check_batch(batch, shape)` calls `check_sample`, `check_data` (when `y` is given) and `check_weight`, on the
      ragged form; `query` never calls it;
    - `n_measurements(batch)` is `prod(W)`, read after the jets (and the packed `d`), or after an index's mode axis;
    - **`take(batch, indices)`** is a draw (ruling 4): it gathers flat over `W` (the jets skipped; ragged or packed;
      probe `y` likewise), sets `drawn_indices = indices` and `minibatch_scale = n / b` times any existing scale. The
      docstring shows `dc.replace(sub, minibatch_scale=None)` for a plain subset.
  - `optional_outputs` is the six names (O-7).
  - The class docstring has a doctest of a query, and the trap of input jets read with `order=()`: the call passes,
    and fits the probes at `ww` and at `pp` instead of the line.
- **`pack_batch(kind, order, batch, N)`** (`backend/uniform_fitting.py`, in `__all__`):
  - the sample with `pack_if_ragged(..., len_J=len(order))` (zero coefficients, O-45), probe `y` to `N`;
  - the index, scalar data, weight, scale and indices pass through;
  - `pack_sample` and `pack_data` are unchanged (ruling 5);
  - doctest with mixed jet sizes.
- **The `Oracle` protocol** is in `backend/optimizers.py` as documentation, in `__all__`: `optional_outputs`,
  `query`, and `check_batch` described as optional (O-29). The import graph is unchanged: `optimizers` still imports
  nothing T3-specific.
- **Reachability:** `t3toolbox.fitting` re-exports `LeastSquaresOracle` and `LeastSquaresBatch` (the same objects)
  and lists them in `__all__`.

## 2. Tests

**`tests/test_least_squares_oracle.py`** (new, 14 tests):
- **`keep`:**
  - no extra element without it;
  - the record equals the closure's computed names on the hold list;
  - the passed `held` arrays come back as the same objects;
  - the `g` from them equals the direct one;
  - `r` is not kept when not computed;
  - `keep=()` gives an empty record;
  - unknown names and a string raise.
- **`frame_form`:** the four forms (and the two meanings of four elements), and bad inputs.
- **`query`:**
  - every output (`f`, `g`, `JtJp`, `Jp_norm_sq`, `Jp`, `r`, `r_block_sumsq`, `Kp`) bitwise equal to the engine, on a
    manifold frame and a corewise point, at `()`, `(2,)` and `(1, 1)` with mixed input jet sizes (entries `()`);
  - `held` equals the closure's computed names on the hold list;
  - `y_block_sumsq` and `f_resolution` by their formulas;
  - on the uniform layout after `pack_batch` (a padded point), equal to ragged on the real parts.
- **The corewise direction's three cases:**
  - `core_directions` given;
  - `variations` renamed;
  - both given as the same, including a rebuilt container of the same arrays;
  - equal values in other arrays raise;
  - on a manifold frame both forms pass through.
- **The scale:**
  - **every subset of size 2 of 6:** the mean of the drawn `f` and `g` equals the full one (1e-13);
  - **on a drawn batch:**
    - `r`, `Jp` and the block sums are plain, and the six scaled outputs are times the scale;
    - `f = 1/2 <r, r>`, `<Jp, r> = <p, g>` and `Jp_norm_sq = <Jp, Jp>` hold in the batch's weighted, scaled inner
      product;
    - `f_resolution` is from the unscaled values times the scale once.
- **`take`:**
  - `n_measurements`, the scale (and nested, multiplied), and `drawn_indices` as given;
  - the gathered sample and `y` on the jet-skipped axis;
  - packed equals packing the ragged take;
  - a two-axis `W` is flattened.
- **`check_batch`:** a good batch passes; it raises on the weight, the modes, `W`, a mode size, `y`, and the index's
  rows and `W`.
- **The records:**
  - the batch's pytree round trip, `None` fields included;
  - the oracle's value hash and equality;
  - `order` required, an int `order` refused, entries with jets refused, an unknown kind refused, the
    `optional_outputs`;
  - reachability from `t3toolbox.fitting`.
- **The bridge:** against `ProbeDerivativesKind`, `ApplyDerivativesKind` and `EntriesDerivativesKind` at order 2,
  through `affine_jet`, `taylor_to_derivatives` and `derivative_weight_to_taylor_weight`: `f`, `g`, `JtJp` and `Kp`
  to 1e-12.

**`tests/test_dispatch.py::test_jit_the_least_squares_oracle`:**
- `query` jitted on ragged and uniform (after `pack_batch`), jax out;
- `held` carried from one compiled call (`f`) into the next (`g`, `JtJp`), against eager;
- the `keep=` record's slots under jit equal the eager ones.

## 3. Verified by script, before the tests (`scratchpad/p14/verify.py`)

| Check | Worst error |
| --- | --- |
| subset unbiasedness of `f` and `g`, all three kinds | 2.7e-16 |
| the three identities on a drawn batch | 1.1e-15 |
| `f_resolution`'s formula | exact |

## 4. Mutation pass (scratch copy, deleted)

13 mutants, all killed by `tests/test_least_squares_oracle.py`:
- `keep`: dropping the held passed in; computing names only to keep them;
- `frame_form`: uniform four read as a frame;
- the scale: reaching `r`; missing `Kp`; `f_resolution` from the scaled `f` (the `s^1.5` error);
- `take`: ignoring an existing scale; gathering at axis 0 instead of after the jets;
- the corewise direction: both directions not compared; `variations` not renamed on a point;
- `check_batch` skipping the weight;
- `pack_batch` without `len_J`;
- entries jets allowed.

## 5. For P16 (recorded as ruled)

- **The regularizer's resolution term.** The optimizer adds `eps |rho|` to the oracle's `f_resolution` for a
  regularized fit (today's `|f|` is `misfit + rho`), and scales the regularizer under the minibatch rule. P17's
  `point_value_and_gradient` uses the same rule.
- **A test that kills a mutant carrying `held` across MC-SGD draws.** A held `r` belongs to its batch's `y`.
- **`pack_sample` and `pack_data`** get the plan's `(kind, order, ...)` forms when the old kinds go.
- **The display** reads `y_block_sumsq` once, and `r_block_sumsq` through `held`, as planned.

## 6. Environments and gate

- **`t3toolbox`**: the new file passes (14 tests, 61 subtests), as does the dispatch row.
- **`tttt`**: the new file, `test_sampling_variables`, `test_sampling_graph` and `test_import_graph` pass (63), and so
  do the module doctests of the four changed modules (10).
- **The full gate at `-n 2`**: below.

**The first full gate** (`scratchpad/gate_P14`): every step passed except one subtest of the suite,
`tests/test_jets.py::TestCheckers::test_no_backend_module_calls_a_checker[fitting.py]`.
- **Why it failed.** That P3 test forbade any backend module to refer to `check_sample`, `check_data` or
  `check_weight`. The plan's 2.2 builds `LeastSquaresOracle.check_batch` from them in `backend/fitting.py`.
- **The change.** The test now allows them in `fitting.py` **only inside `check_batch`** (by its AST, outside it
  none), requires exactly those three there, and asserts that nothing in `fitting.py` calls `check_batch`. Checkers,
  not guarantees, as before.
- **Verified.** `test_jets.py` passes on both environments. The full gate was re-run with the fix: below.

**The re-run full gate** at `-n 2` (logs in `scratchpad/gate_P14b`) passed every step:
- **suite**: 1851 passed, 5 skipped, 1 xfailed, 90,556 subtests, in 1881 s; peak 8.3 GB; **lowest MemAvailable
  3735 MB**;
- **doctest-modules**: 319;
- **quickstart**: pass;
- **pages**: 47;
- **examples**: 14;
- **sphinx**: 0 warnings.

## 7. Round 1 (`P14_RESPONSE_r1.md`)

- **The packed probe `y`'s mode padding.** `query` now unpacks a packed probe `y` (the frame's static shape) before
  reducing it, so `y_block_sumsq` and `f_resolution` read the real part only, as the engine does. The uniform test
  now puts garbage in the padding of every packed form.
- **The dtype of `eps`.** A float32 test pins `f_resolution`'s `eps` to the result dtype.
- **`check_batch`.** The P3 test now asserts that no backend module calls `check_batch`.

**The round-1 full gate** at `-n 2` (logs in `scratchpad/gate_P14r1`) passed every step:
- **suite**: 1852 passed, 5 skipped, 1 xfailed, 90,556 subtests, in 1867 s; peak 8.5 GB; **lowest MemAvailable
  3479 MB**;
- **doctest-modules**: 319;
- **quickstart**: pass;
- **pages**: 47;
- **examples**: 14;
- **sphinx**: 0 warnings.
