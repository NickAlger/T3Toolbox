# P5: Engine A — handover

Slice P5 of `PART1_PLAN.md`, built to the accepted design (`P5_DESIGN_SKETCH.md`), with the reviewer's conditions
(`P5_DESIGN_REVIEW.md`) and the orchestrator's rulings of 2026-11-20. Written by the implementer, 2026-11-20, against
`graded-messages` after the orchestrator's commit of the `subTest` guard. Nothing is committed.

## 1. What changed

**`t3toolbox/backend/sampling_steps.py`** (new, private, `__all__ = []`, no doctests):
- The steps of the plain grade, each one site's math, each term one literal `contractions.jet_contract(...)` call
  through the module attribute:

  | Step | String |
  | --- | --- |
  | `xi_step(B, w)` | `'JWo,Cio->JWCi'` |
  | `mu_step(mu, G, xi)` | `'JWCa,Caib,JWCi->JWCb'` |
  | `nu_step(nu, G, xi)` | `'Caib,JWCi,JWCb->JWCa'` |
  | `eta_step(mu, G, nu)` | `'JWCa,Caib,JWCb->JWCi'` |
  | `z_step(eta, B)` | `'JWCi,Cio->JWCo'` |

  All five are in `_jet_contract_cases.STRINGS`, with `C` written. Statics by keyword: `lens`, `order`, `form`.
- `xi_step` passes the leaf's jet shape, `_leaf_order(w, order)`: the input's jet sizes, read off the operand's static
  shape, clipped to `F`, minus one. Every other step passes `order`.
- The whole steps `r_step` (`z - y`, `y` broadcast over `C`), `f_step` (`1/2 <r, weight^2 r>`, keeping `C`) and
  `r_block_sumsq_step` (`(d,) + F`, unweighted).
- `start_ones`: ones in slot `(0, ..., 0)`, zero elsewhere, on the full box `F + W + C + (size,)`.

**`t3toolbox/backend/sampling_variables.py`** (new; `__all__ = ['sampling_variables', 'HeldVariables']`):
- **The graph:** the row type `Rule(name, step, reads, needs, carries_K)`, and `_PROBE_GRAPH`, 11 rows.
  - `z` has two routes: `eta`, `Vc` (needs `{'Vc'}`), then `eta_D`, `D`.
  - Every row has `carries_K=False`.
- **Declared tables:**
  - the namespaces `_STORES` and `_ARGUMENTS`; the variables are the names some row makes;
  - `_BOND_OFFSET = {'mu': 0, 'nu': 1}`, by letter;
  - `_SHAPES`, the shape table: name to `(index, jet sizes, size)`.
- **The module docstring** carries the graph table and the shape table, which a test compares with the data. It
  states:
  - the precedence (the earlier applicable row wins, most specific first);
  - that a held variable never changes the route;
  - the bond offsets;
  - the start sizing (by the end bond of the row's first read, its own store);
  - the memo rule (a body closes over the record, the box and the form, nothing per site).
- **`_closure(kind, want, held, present)`**: `lru_cache`d on hashables only, with `present` as a frozenset that
  includes the pseudo-argument `'_manifold'`. It is a depth-first search: a held name is a leaf, and the first row
  whose `needs` are present is taken. It raises naming the missing argument and the chain, for example "'r' needs the
  argument y, which was not given (needed through f -> r)".
- **`HeldVariables`**: a frozen dataclass with 8 slots, each with its shape comment: `xi`, `mu`, `nu`, `nu_L`, `eta`,
  `eta_D`, `z`, `r`; not `w`.
  - It reads like a dictionary: `__getitem__`, `__contains__` (filled slots only; an unknown name gives `False`),
    `keys`, `__iter__`, `__len__`.
  - `with_values` returns a new record; `slots()` lists every slot.
  - An unknown name raises: `TypeError` at construction and in `with_values`, `KeyError` at lookup.
  - `==` raises and the record is unhashable. It has its own message, since the library's `ExplicitEquality` message
    names `allclose`, which a record of slots does not have.
  - It is a pytree only `if jax_available`: the leaves are the filled slots, the aux the tuple of filled names.
- **The engine `sampling_variables`**, in the order of the plan's section 2.3. It checks:
  - an unknown kind; `_read_order`; an entries call with jets;
  - a kind not built (`NotImplementedError`);
  - `want` given as a string; unknown names; `held` of the wrong type;
  - the frame structure (`NotImplementedError` for the uniform layout and the corewise point);
  - every filled held slot against `_shape`, all of them, read or not.

  The layout: `form='loop'`. Then:
  - the closure;
  - `use_jax` over every array input;
  - the result dtype, by `jets._lowest_precision_dtype`, over the arrays the closure's rows read (held values
    included), and those arrays cast once;
  - the rows, run by `_run`: `'whole'` directly; `'site'` through `xmap` with a memoized `_site_body`; the sweeps
    through `xscan` with a memoized `_sweep_body` (the `'right'` sweep on `[::-1]` sequences);
  - the start joined in the one helper `_with_end`.

  Each row gets `lens` or `lens_no_K` by its `carries_K`. `'w'` returns the sample as given, not cast.

**Tests.**
- **`tests/test_sampling_variables.py`** (new, 9 tests; review round 1 added the dtype of held variables):
  - by definition, every name against `sampling_definitions`, with a random weight, over the 7 jet cases of the plan's
    matrix (`defs.JET_CASES`), on 3 problems; one of them, `tails_one_axis`, has `L` and `R` end bonds that differ at
    both ends;
  - `w` as given (read and unread, and not cast);
  - `z` through `Vc` against the reference, on a consistent frame from `t3_orthogonal_representations`;
  - the shapes against `_shape` and the declared jet sizes;
  - every `jet_contract` call recorded: count `7 d`; the leaf string given the clipped jet shape; every other string
    given `order`, with a result of `F`;
  - the result dtype in its simplest cases (step 4 of section 2.3; see section 4);
  - safe, unsafe and jit: the same numbers and errors.
- **`tests/test_sampling_identities.py`** (new, 5 tests):
  - `()` is slot zero of every jet case;
  - `(n_1, n_2)` restricted to `(k, 0)` is the call along the first direction, at `(1, 2)`, `(3, 1)` and `(2, 0)`;
  - every slot held in turn gives the recomputed result, bit for bit;
  - a held variable is used: `mu` doubled doubles `eta` and `eta_D` and leaves `nu` alone; a held `z` doubled; a held
    `r` gives `f` with no `y`;
  - `z` through `D` equals `z` with `Vc`, at `()`, `(3,)` and `(1, 2)`.
- **`tests/test_sampling_graph.py`** (new, 16 tests; review round 1 added the held check of a second, unread slot to
  one of them):
  - the docstring graph table equals the graph row for row, and the shape table equals `_SHAPES`;
  - every step name resolves to a function of `sampling_steps`;
  - the three namespaces are disjoint and cover every read;
  - the `HeldVariables` slots are exactly the computed variables another row reads;
  - the bond offsets, with a hand check of `eta[i]` from `mu[i]` and `nu[i+1]`;
  - `carries_K` is a field;
  - the precedence (both routes applicable takes the first; the fewer facts take the last);
  - a held `eta_D` with `Vc` does not change the route (closure and step counts);
  - step counts for 14 wants, derived by hand;
  - the end starts sized by their own store;
  - every check.
- **`tests/test_held_variables.py`** (new, 7 tests): the slots; reads like a dictionary; unknown names; `with_values`;
  equality refused; the pytree round trip; under `jit` in the engine.
- **`tests/test_dispatch.py`:** `test_jit_sampling_variables`, the engine jitted for `('f',)` at `()` and `(2,)`:
  jax in, jax out; the frame closed over as numpy; the numbers numpy's.
- **`tests/test_jax_absent.py`:** `_ENGINE_SCRIPT`, the engine and `HeldVariables` with no jax.
- **`tests/test_contractions_interpreter.py`**, option (b):
  - `scan_vocabulary` skips the first argument of a `jet_contract` call (`_jet_contract_literals`);
  - `TestJetLiteralsStayInJetContract`: (i) every library literal naming `J` is such an argument; (ii) every
    `jet_contract` literal is in `STRINGS`.
- **`tests/test_contractions_sharding.py`:** its duplicated `_scan_vocabulary` skips the same literals. They are swept
  through the case file by the class's jet test.

Verification: the mathematics is pinned against the reference module (P4c), which shares no code with the engine.
On first run, everything passed except two test mistakes of mine: a `w` type check (the sample is returned as given,
a list) and an `assertIs` on a held value that the engine casts (a new tuple). The floor env then found a real dtype
bug, fixed: numpy 1.x lifts `0.5 * float32 scalar` to float64, and `f` was a numpy scalar, not an array. `f_step` now
multiplies by `xnp.asarray(0.5, dtype=...)` and returns `xnp.asarray`. The four new files pass on both environments
(the floor: py3.9, numpy 1.22, jax 0.4.30), and so do the module doctests and `test_dispatch -k sampling_variables`.

**Mutation pass** (a scratch copy, deleted). 26 mutants, all killed:
- **The steps and tables:** a leaf not clipped; a leaf given the full order; `nu` reading `L`; `nu_L` reading `R`;
  `eta_D` reading `U`; the `z` routes swapped; the `nu` offset 0; the `mu` offset 1.
- **The sweeps:** the end start sized from `L` always; the start ones in every slot; the right sweep not turned back.
- **The sums:** `f`'s weight not squared; `f` without the half; `r` with the other sign; `r_block_sumsq` summed over
  the jet axes.
- **The engine's checks and closure:** the held-shape check removed; the closure ignoring held; `needs` ignored;
  unknown names accepted; the dtype always float64; `w` returning the cast sample (it first survived; the `w` test
  then gained calls where `w` is read).
- **`HeldVariables`:** `in` over the slots, not the filled ones; `with_values` in place.
- **The interpreter test:** the scan keeping the jet literals; a `J` string held in a variable; a literal not in
  `STRINGS`. (A first form of the last mutant was whitespace-equal and proved nothing; replaced.)

**`subTest` labels:** every label is a builtin (`str`, tuples of ints). The conftest guard ran with every test above.

## 2. The gate

The whole gate at `-n 3` (`--detach`, then `wait`), with nothing else running. The environment was python 3.11.15,
numpy 2.4.6 and jax 0.10.2. The summary, copied:

    step                result counts                                                      seconds   peak_MB  minavl_MB
    suite               PASS   1705 passed, 5 skipped, 1 xfailed, 1 warning, 70687 subtests passed   1777.1      8861       3335
    doctest-modules     PASS   314 passed                                                     47.2      1401      10579
    doctest-quickstart  PASS   1 file(s), 0 with failures, 0 failed example(s)                 2.7       284      11711
    doctest-pages       PASS   48 file(s), 0 with failures, 0 failed example(s)               14.1       369      11631
    examples            PASS   14 examples, 0 failed                                           382       639      11370
    sphinx              FAIL   18 warning/error line(s)                                     1009.0       635      11232

**The Sphinx failure, and its fix at its source** (the orchestrator's ruling, 2026-11-20).
- All 18 warnings were one kind: `more than one target found for cross-reference 'z'`. The two targets were
  `geometry.ManifoldCurvatureData.z` and the new `HeldVariables.z`.
- The source is numpy-style type fields written as prose: `y: of the type of z`, and `of the type of z` as a return
  type, in the docstrings of `jets`, `data_preparation` and `t3toolbox.fitting`. Sphinx splits a type field into
  cross-references at `of` and `or`, so `z` was one. Until P5 it resolved, silently, to `ManifoldCurvatureData.z`, a
  wrong link.
- A first fix, a skip hook in `docs/conf.py` that kept the `HeldVariables` fields off the rendered pages, is reverted:
  `docs/conf.py` is as it was.
- **The fix:** every such type field is now a real type, `tuple of NDArray or NDArray` (`..., or None` where it was),
  and the words "of the type of `z`" move to the description, as a new first line: "Of the type of ``z`` (a tuple of
  ``d`` arrays for probe, one array otherwise)." That is 22 entries:
  - `backend/jets.py`: 13;
  - `backend/data_preparation.py`: 5;
  - `t3toolbox/fitting.py`: 4.

  Two other changes follow the same rule:
  - **`sampling_variables`'s own `y`** is now `sequence of NDArray, optional`, "The data, of the type of ``z``".
  - **The search found one more family of prose types**, in `backend/sampling_definitions.py` (P4c): 31 return types
    `NDArray, or a tuple of d of them if ``i`` is None` (or `d+1`, or "for probe with ``i`` None"), which name `d`.
    No `d` is documented today, so they warned nothing, but they are the same mistake. Each is now `NDArray or tuple
    of NDArray`, with "One array; with ``i`` None, a tuple of ``d`` of them." (or ``d + 1``, or "for probe ...") as
    the description's first line.

  No other type field of the branch's docstrings names a variable. (`length: float, 0-dimensional NDArray, or ``m``
  of them`, in `data_preparation`, names `m` as a literal and is left.)

**After the fix:** the steps `sphinx` and `doctest-modules`, `-n 1`, on the final tree (after review round 1's
change too), with the `HeldVariables` fields rendered:

    step                result counts                                                      seconds   peak_MB  minavl_MB
    doctest-modules     PASS   314 passed                                                     45.9      2006       9949
    sphinx              PASS   0 warning/error line(s)                                      1022.7       635      11183

**In the built HTML:**
- No page links to the anchor `ManifoldCurvatureData.z` but that class's own page. The fixed entries, for example
  `jets/jet_to_dual_jet.html`, show "Of the type of `z`" as literal text, and their only links to
  `ManifoldCurvatureData.html` are the navigation sidebar's.
- `HeldVariables.html` has the anchors of its fields (`HeldVariables.z`, `.mu`, `.nu_L`, ...).

The suite was not rerun after the gate. The changes since then:
- docstrings;
- review round 1's engine change (step 4) and its two tests.

The tests these touch passed single-process:
- the four new files with the module doctests, 39, on both environments;
- the docstring tests of `test_jets`, `test_data_preparation` and `test_sampling_definitions`, with the module
  doctests of the changed modules, 195.

`P5_RESPONSE_r1.md` has the rest.

Against P4d's gate (the guard's run, 1,665 passed):
- The suite has 40 more tests (1,705):
  - the 36 of the four new files;
  - `test_jit_sampling_variables`;
  - the `jax_absent` engine script;
  - the two guards of option (b).
- The module doctests are 314 (2 new: the engine and `HeldVariables`).
- The suite's peak memory is 8.9 GB, with 3.3 GB available at its lowest.

No process, lock or file of the gate is left in the repository.

## 3. Departures and choices

1. **The row type has five fields**, `carries_K` added to the plan's four (ruling 2; the reviewer's request 2).
2. **`step` is `(driver, step_name)`**, the name a string resolved by `getattr` at run time (the test hook). A test
   checks that every name resolves.
3. **`z_step` is the `z_term` alone**, `z_step(eta, B)`. In `jet_steps_check.py`, `z_step(mu, L, nu, D)` fused the
   `eta_term`. Here `eta_D` and `eta` are nodes of their own (two routes), so `z_step` takes the trunk. It is the same
   name, so the 22 steps still stand.
4. **Who carries the call type.** `present` carries `'_manifold'` (ruling 1); no row needs it yet. Only `y` and `Vc`
   are read as facts in P5. `variations` and `core_directions` enter `present` when given, and nothing reads them.
5. **The held-shape check covers every filled slot**, read or not, with each shape from `_shape`, the function the
   tests also read. A held list is accepted like a tuple.
6. **The dtype of a held variable** (the orchestrator's ruling 2 of review round 1, replacing this departure's first
   form). Every output is in the result dtype. The counted inputs include every held variable a row reads or that is
   wanted, each cast once, so a wanted held variable is returned cast. A held variable neither read nor wanted is not
   counted and not returned. The engine's docstring states it, and `TestResultDtype` pins it.
7. **The result dtype** is computed in P5 (ruling 7) with `jets._lowest_precision_dtype`, imported privately as
   `data_preparation` does. `TestResultDtype` covers four simple cases, so that the rule is not untested; the matrix
   stays P10's.
8. **`f` is a 0-dimensional array** on every backend (the numpy 1.x fix above).
9. **`HeldVariables` does not use the `ExplicitEquality` mixin** (choice 6's point: a message naming the wrong
   remedies). It refuses `==` the same way, with its own message.
10. **The not-yet-built paths raise `NotImplementedError`**: the apply and entries kinds, the corewise point, the
    uniform layout. The entries kind with jets raises its `ValueError` first, as section 2.3 step 1 says.
11. **The engine's docstring** is in the house shape; its examples were run and passed as written.
12. **Docstrings outside the plan's file list** (section 2): the prose type fields of `jets`, `data_preparation`,
    `t3toolbox.fitting` and `sampling_definitions` are made real types. Only docstrings changed; `docs/conf.py` is as
    it was.

## 4. Found outside the slice

- The P5 entry's "`z` (through `D`, and with `Vc` read off the frame)" was read as ruling 8 settled it; nothing more.
- `jets._lowest_precision_dtype` now has two importers outside `jets` (`data_preparation`, `sampling_variables`). If a
  third appears, it may deserve a public home in `common`.
