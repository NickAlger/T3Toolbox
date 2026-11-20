# P10: Engine F, the stacks, the unsummed blocks and the dtype matrix — handover

Slice P10 of `PART1_PLAN.md`, built to `P10_DESIGN_NOTE.md`, `P10_DESIGN_REVIEW.md` and the coordinator's rulings at
the end of the plan's P10. Written by the implementer, 2026-11-20, against `graded-messages` at `9c8bab25` (P9b
committed). Nothing is committed.

## 1. What changed

**`t3toolbox/backend/sampling_steps.py`.**
- **The six readout steps take a static `keep_W: bool = False`**: `V_tilde_step`, `H_tilde_step`, `V_tilde_dot_step`,
  `H_tilde_dot_step`, `V_top_step`, `H_readout_step`. Each has two bodies, `if keep_W:` the readouts with `W` first in
  their output, else the summed ones, each literal the first argument of its `jet_contract` call (see section 3, item
  1). **Six new literals** (three `V`: `'JWKCi,JWo->WKCio'`, `'JWCi,JWKCo->WKCio'`, `'JWKCi,JWCo->WKCio'`; three `H`:
  `'->WKCaib'` with `K` on the first, second or third operand); `K` being a group, its placements are these same six.
  Entries' `V_scatter_step` had `keep_W` since P9b.

**`t3toolbox/backend/sampling_variables.py`.**
- **`sum_over_samples`** is read: `_Setting.keep_W = not sum_over_samples`. A site row whose variable is a block
  (`_is_block`: jets `'none'` in the shape table; `V_tilde`, `H_tilde`, `V_tilde_dot`, `H_tilde_dot` and the `_Jp.`
  twins) gets `keep_W`; no other row does. `_site_body` takes it as a static and it joins the memo key. `_shape` puts
  `W` first on a block when kept, so `g`, `JtJp`, `Kp` (and a held block) are `W + [K] + C` and the core.
  `f`, `r_block_sumsq` and `Jp_norm_sq` ignore the flag (sums).
- **`_check_stacks(setting, stores, sample, Vc)`**, removable (pending O-18): exact tuple equality (not broadcast
  compatibility) of `C` as the prefix of every core of every store (`D`, `U`, `L`, `R`; `D`, `L` on a corewise point)
  and of every `Vc[i]` (with the core's rank), and of `W` after the jet axes of every `w[i]` (not for entries' one
  index array). A `ValueError` names the operand, its shape and the expected stack. Called by one line after
  `_read_setting`. Its docstring states its cost and how to remove it (the function, its call, `TestTheStackCheck`).
- **Docstrings**: a paragraph "The stacks" (each read once, the layout `J + W + [K] + C`, the check, the unsummed
  blocks, the sums that ignore the flag); the memo key with `keep_W`; the block shapes; the engine's
  `sum_over_samples` parameter and its signature comment; the checks paragraph (the stacks are checked; the mode
  sizes, jet sizes, `y` and the weight stay the checkers').

**Tests.**
- **`tests/test_sampling_stacks.py`** (new, 11 tests):
  - **the stack matrix** (`TestTheStackMatrix`): the six configurations (no stack, `C = (2,)`, `K = (3,)`,
    `K + C`, `C = (2, 3)`, `K = (2, 3)`) with every `W` in `()`, `(3,)`, `(2, 2)` once per kind (54 calls), the jet
    shapes `()`, `(2,)`, `(1, 1)` cyclic (entries `()`), the call type alternating manifold/corewise; every name of
    the call, element `(k, c)` by element, against the UNSTACKED engine at frame element `c`, direction `k` and **`W`
    flattened to one axis** (so a multi-axis `W` is checked against a single-axis one, the reviewer's gap);
    `r_block_sumsq` (a sum over `C` too) against the oracles' sum; and **one maximal case per kind** (`K = (2, 3)`,
    `C = (2, 3)`, `W = (2, 2)`, `(1, 1)`; entries `()`). The tails are at bond `d`, `L` and `R` differing;
  - **a held stacked variable**: held `mu` and `r` with `C = (2, 3)`, `K = (2,)`, `W = (2, 2)`, `(1, 1)` give `z`,
    `g`, `Kp`, `JtJp` bit for bit; a held `mu` without its `C` raises;
  - **unsummed** (`sum_over_samples=False`) on `C`, a multi-axis `K` (`(2, 3)` at `(1, 1)`), a multi-axis `W` on a
    corewise apply call, and entries with `K`, `C` and `W = (2, 3)`: the shape `W + [K] + C + core`, the sum over `W`
    equal to the summed call, and the blocks of the first, the last and an off-diagonal sample equal to the call on
    that sample alone;
  - **the reference anchor** (thin, as ruled): probe at `(2,)`, `W = (2, 2)` with `C = (2, 3)` and with `K = (2, 3)`,
    every name; and the maximal stacks at `(2,)` on the outputs; element by element against `sampling_definitions`;
  - **the stack check** (`TestTheStackCheck`): a size-1 `W` in `w[2]`, a size-1 `C` in `U[1]`, `Vc[0]` without its
    `C`, a size-1 `C` in a corewise `G[2]`, each with its message; a consistent `C = (1,)`, `W = (1,)` passes;
  - **the dtype matrix**: every output of a call wanting nine names on numpy float64 and float32 and jax float32 and
    float64 (jax under `jax.jit`, float64 in `jax_in_double_precision`); float32 with float64 gives float32; a float64
    `y` nothing reads leaves a float32 call float32; jax with numpy gives jax; a held float64 `mu` that is read gives
    float32 in a float32 call, one neither read nor wanted changes nothing, a jax one unread makes the call jax; an
    int32 or int64 index does not count and entries' one-hot `'w'` is in the result dtype; `index_to_one_hot`'s
    float64 vectors given to apply count like any float array.
- **`tests/_jet_contract_cases.py`**: the six literals in `STRINGS` (38 strings), with their family's library shapes.
- **`tests/test_jet_contract.py`**, **`tests/test_contractions_sharding.py`**, **`tests/test_contractions_interpreter.py`**:
  the counts that follow: 38 strings, 18 readouts; the forms' followers 19 and cases checked 34, the slots not read
  12 + 12, the readout cases 14; the sharding count `(38 + 15 + 5 + 3) * 2 * 2`; the interpreter's strings derived
  from the case file 30 (24 before). Every new literal is checked against the reference, its sensitivity,
  its two forms and its sharding over `W` by the existing tests.

**Verified before the tests asserted it:** the matrix, the maximal cases and the anchor were run as a script first
(they are the test bodies): every comparison within `1e-12` (the anchor within `1e-10`), on every name.

**Single-process runs:** the sampling tests (stacks, entries, apply, graph, variables, corewise, identities), held,
the bridge, jax-absent and the import graph, with the two modules' doctests: 124 passed (12,926 subtests) current,
124 passed on the floor; the jet-contract and sharding files (their counts updated): 46 passed, 1 xfailed, current.
(`TestUnsummed` was strengthened during the mutation pass, below, and re-run: it passes.) After the switch to two
bodies (section 3, item 1): the interpreter, jet-contract and sharding files pass (69 passed, 1 xfailed, current), and
the stacks, interpreter, variables and entries files with the steps' doctests on the floor (57 passed).

**What the new tests cost** (`test_sampling_stacks.py`, single-process, current env): 30.4 s in all: the covering
design 9.7 s, the maximal cases 7.1 s, the reference anchor 5.1 s, the dtype matrix 6.7 s (4.8 s of it the jitted jax
columns), the unsummed tier 1.0 s, the held case 0.3 s. At `-n 2` that is about 15 s of the suite's wall time; the six
new literals add a few seconds more to the jet-contract and sharding files.

**`subTest` labels:** builtins (stacks and jet shapes passed as `repr`; the guard ran).

**Mutation pass** (scratch copies, deleted). 13 mutants, all killed:
- **The reviewer's:** a multi-axis `W` offset (`f` and `Jp_norm_sq` summing only the first `W` axis), and the `keep_W`
  grid broadcast the wrong way. The grid mutant first **survived**: on a square `W = (2, 2)` it transposes the samples,
  and the test looked only at the diagonal (the first and the last samples) and at the sum. `TestUnsummed` now uses
  `W = (2, 3)` for entries and also checks a sample off the diagonal; it kills the mutant.
- The mutants on the readout steps (the literal with `K` before `W`, the twins not block rows, `keep_W` never passed,
  the grid) were re-run on the two-body form: all killed, by `test_sampling_stacks.py` alone.
- **The others:** the gather moving one `W` axis; an unsummed literal with `K` before `W`; the twins not taken as
  block rows; the stack check broadcast-compatible (a size-1 axis accepted), without `Vc`, without `W`; `r` inserting
  one `C` axis (probe) or none (apply); the start values with the first `C` axis only; `keep_W` never passed.

## 2. The gate

The whole gate at `-n 2` (`--detach`, then `run_gate.sh wait`), with nothing else running. The environment was python
3.11.15, numpy 2.4.6 and jax 0.10.2. The summary, copied:

    step                result counts                                                      seconds   peak_MB  minavl_MB
    suite               PASS   1782 passed, 5 skipped, 1 xfailed, 1 warning, 87350 subtests passed   1870.5      8425       3810
    doctest-modules     PASS   314 passed                                                     45.6      1735      10342
    doctest-quickstart  PASS   1 file(s), 0 with failures, 0 failed example(s)                 2.6       255      11808
    doctest-pages       PASS   48 file(s), 0 with failures, 0 failed example(s)               14.1       363      11720
    examples            PASS   14 examples, 0 failed                                           394       636      11449
    sphinx              PASS   0 warning/error line(s)                                      1111.6       628       9909

**No failure.** Against P9b's committed state (1,771: P9b's gate counted 1,770, before round 1 added the one-hot `'w'`
test), the suite has 11 more tests, those of `test_sampling_stacks.py`.

**Memory:** the lowest MemAvailable is **3.81 GB** (4.14 GB at P9b); the peak RSS summed over the workers 8.4 GB.

**Time:** the suite took 1,870 s against 1,804 s at P9b's gate (+66 s at `-n 2`): the new file (30 s single-process)
and the six literals swept by the jet-contract, sharding and interpreter files.

**An earlier gate of this slice failed** (26 failures, all in `test_contractions_interpreter.py`): the first build
passed the readout literals by name (section 3, item 1). Fixed by the two bodies; the gate above is the re-run.

No process, lock or file of the gate is left in the repository.

**Status: P10 is COMPLETE** (built, verified, mutation pass done, full gate green); nothing is committed, and the tree is
as listed: `sampling_steps.py`, `sampling_variables.py`, `_jet_contract_cases.py`, `test_contractions_interpreter.py`,
`test_contractions_sharding.py`, `test_jet_contract.py` modified; `test_sampling_stacks.py` and this handover new.

## 3. Departures and choices

1. **Two bodies, not a choice of literal "in place".** The ruling read "a static `keep_W` choosing in place"; the first
   build named the two literals in a conditional at the step's top and passed the chosen one by name. The gate's
   `test_contractions_interpreter.py` then failed: its guards require every `J` literal of the library to be the
   first argument of a `jet_contract` call (so no such string reaches `contract` unswept, and the case file stays the
   list of the call sites). So each readout step has two bodies, `if keep_W:` and the summed one, as the library's
   old `_summed`/`_unsummed` pair did, the literals at their call sites. The interpreter's count of strings derived
   from the case file goes from 24 to 30 (the three new `H` readouts, two each).
2. **The jax column of the dtype matrix runs under `jax.jit`**, with no jets: eager jax dispatches each contraction
   on its own and took 23 to 72 s for the same check; jitted it is under 5 s, and the dtype rule is the same. The
   jax mixed case (jax with numpy) stays eager, on three names.
3. **`_is_block` reads the shape table** (no declared set); the twins are blocks through `_table_name`.
4. **The covering design alternates the call type** (manifold, corewise) as the note proposed; the maximal cases
   are manifold calls.
5. **`r_block_sumsq`** sums over `C` as well as `W` (the weight's shape), so its matrix comparison is with the sum
   of the oracles over `C`.
6. **For P11** (not built): `_read_setting`, `_read_K` and `_check_stacks` get a uniform branch; the unsummed blocks
   keep `W` unreshaped, so they belong in P11's sharding test; `keep_W` reaches `xmap` through the same memo key.
