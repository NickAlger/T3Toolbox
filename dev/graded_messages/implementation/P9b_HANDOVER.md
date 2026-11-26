# P9b: Engine E, the entries kind on both call types — handover

Slice P9b of `PART1_PLAN.md` (the second half of P9), built to `P9_DESIGN_NOTE.md`, `P9_DESIGN_REVIEW.md` and the
settled design at the end of the plan's P9. Written by the implementer, 2026-11-20, against `graded-messages` at
`81e1174a` (P9a committed). Nothing is committed.

## 1. What changed

**`t3toolbox/backend/sampling_steps.py`**, a new section "The entries kind's leaf":
- `xi_gather_step(B, index)`: the columns of a Tucker core (or of `V`, `D_dot`) at the grid points,
  `B[..., index]` moved to `W + [K] + C + (n,)`. A slice, no contraction (as the old `_entry_xis`).
- `V_scatter_step(eta, index, D, keep_W=False)`: the trunk's tilde added into the columns `index`, `[K] + C + (n, N)`;
  `D` is read for `N` only. `W x n` in memory, not the one-hot product's `W x N`. Duplicates add up: `np.add.at` on
  numpy, `.at[index].add` on jax, both with the `W`-shaped index array directly (no reshape). **`keep_W=True`** (for
  P10's `sum_over_samples=False`): one block per sample, `W + [K] + C + (n, N)`, written with a `.at[...].set` / an
  indexed assignment on `(grids of W, index)`; nothing collides there. The engine does not pass it yet.

**`t3toolbox/backend/sampling_variables.py`.**
- **The derivation** `_entries(rows)` over the apply rows, by one declared table:
  - `_ENTRIES_STEPS = {'xi_step': 'xi_gather_step', 'xi_dot_step': 'xi_gather_step', 'V_top_step': 'V_scatter_step'}`;
  - every read of `w` is a read of `index`; a step in `_ENTRIES_SIZED` (the scatter) also reads `D`;
  - `_ENTRIES_FULL` maps each short node to its full node (`_eta_tilde_top` -> `eta_tilde`, `_eta_D_tilde_top` ->
    `eta_D_tilde`, `_eta_D_tilde_dot_top` -> `eta_D_tilde_dot`): at `order = ()` the short form is the whole trunk,
    so the short-node rows go and their readers read the full node;
  - invariants raise `RuntimeError`: a row reading the input with no step in the table; a derived row depending on
    `w` or a short node.
- `_ENTRIES_ROWS = _entries(_APPLY_ROWS)`; `_ROWS['entries']`; `_JP_PATHS` for entries, through the full `eta_tilde`
  (`'w_tilde', 'nu_tilde', 'eta_tilde', 'V_tilde', 'H_tilde'`); the corewise point by the same `_corewise`
  derivation. Six graphs now: entries 46 / 33 rows.
- **The index**: `'index'` is among `_ARGUMENTS`; `_INTEGER_ARGUMENTS = {'index'}` is neither counted for the dtype
  nor cast (it goes into the values as `xnp.asarray`); the sample is bound to `index` for the entries kind, to `w`
  otherwise. `_read_setting` reads `W = shape(index)[1:]` and no jets for entries.
- **Shapes**: `_KIND_SHAPES['entries']` as apply's (`z`, `r`, `w_tilde`, `Jp` are `W + [K] + C`); `r_block_sumsq` is
  a number.
- **The zero names** are apply's four, with the same reason (ruling 1 of P9a, pending the maintainer).
- The `NotImplementedError` for an unbuilt kind is gone (all three are built); the one for the uniform layout stays.
- **`'w'` on an entries call** (round 1, the coordinator's ruling) is the one-hot vectors of its definition: the
  entries kind's own row `w` (`_ENTRIES_OWN_ROWS`, `w_one_hot_step` reading `index` and `D`:
  `jets.index_to_one_hot`, cast once to `D`'s dtype), made only when wanted, in the result dtype. For probe and apply
  `'w'` is the sample as given.
- **Docstrings**: a section "The entries kind" with the table of steps (tested against `_ENTRIES_STEPS`); the closing
  sentence (three kinds built); the engine's `kind`, `sample` (the integer index, never cast, not counted), `want`
  (`'w'` for entries), `y` (`W`), `weight` (`()`), `Vc`, the return; a doctest example (entries' shapes, and `z`
  equal to apply's at `index_to_one_hot`), output from a real run.

**Tests.**
- **`tests/test_sampling_entries.py`** (new, 11 tests):
  - by definition, every entries name on a manifold frame and on a corewise point, `K = ()` and element by element
    `K = (2,)`, three problems; a test that the problems repeat an index in a mode (the tails problem does, in every mode);
  - **against apply** at `index_to_one_hot(index, shape)`, every name, both call types, `K = (2,)`, to `1e-12`;
  - **the scatter**: against the one-hot product with an index repeated three times and one twice (and a check that
    the `+=` form differs from it); `keep_W` against the per-sample outer products (exact) and summed back to the
    summed block; the mode size read off `D` with a Tucker rank above it; the gather against `B[..., index]`;
  - **the index**: a float32 frame with an int64 index gives float32, a float64 frame with an int32 index float64,
    and `'w'` is the one-hot vectors of `index_to_one_hot` in the result dtype, on both call types, built only when
    wanted (round 1);
  - a zero name raises on a corewise call;
  - the identities per element of `K = (2,)`, both call types;
  - step counts for `('z', 'g')` on both call types and `('Kp',)` (gathers, scatters, the full `eta_step`).
- **`tests/test_sampling_apply.py`**: the names comparison covers entries (the reference's names but the four zero
  ones; on a corewise point the section-9 spellings but the frame's other bases).
- **`tests/test_sampling_graph.py`**: the entries rows are `_entries(_APPLY_ROWS)`; six graphs; a new test that the
  derivation follows the table row for row (the docstring's table is `_ENTRIES_STEPS`; every row's step, reads,
  needs, `carries_K` and start; no `w` and no short node on either entries graph; the `RuntimeError` for a row with
  no step); the zero names are not made by entries' graphs; the direction-masks test leaves `index` out; the
  `NotImplementedError` check for entries is gone.
- **`tests/bridge/test_old_kernels_bridge.py`**: `TestTheEntriesKind`:
  - with no jets: `t3_entries` and `EntriesKind.point_forward` (`z`), `tv_entries` and `.forward` (`Jp`),
    `tv_entries_transpose` and `.transpose` (`g`, and `JtJp` given `weight^2 Jp`), `t3_entries_corewise_hessian` and
    `.corewise_hessian` (`Kp`), `K = ()` and `(2,)`, three problems;
  - **entry 120's migration**: `t3_entries_derivatives` and `EntriesDerivativesKind` (`point_forward`, `forward`,
    `transpose`) against the APPLY kind at `affine_jet(index_to_one_hot(index, shape), pp)`, `n` = 1 to 3, by the
    `t!` conversions.
- **`tests/test_dispatch.py`**: `test_jit_sampling_variables_the_entries_kind` (the index a traced jax integer array
  with repeated indices; manifold `z`, `g`, `Kp`, `JtJp`; corewise `g`, `Kp`; tails at both ends).
- **`tests/test_jax_absent.py`**: entries' `z` and `g`.

**Verified by script before the tests asserted it:**
- every entries name against the reference, manifold and corewise, `K = ()` and `(2,)`, three problems: worst
  3.2e-15;
- entries against apply at the one-hot vectors: equal (0.0 on the tails problem, every name);
- the bridge (`z`, `Jp`, `g`, `JtJp`, `Kp`, old functions and `EntriesKind`; the entries derivatives against apply
  at the one-hot jet, `n` = 1 to 3): worst 1.8e-15.

**Single-process runs:** the changed and related files (the sampling tests, the bridge, the jet-contract and
sharding tests, jax-absent, the import graph, the contractions interpreter) with the two modules' doctests pass on
both environments: 180 passed, 1 xfailed (29,104 subtests) current; 180 passed, 1 xfailed on the floor. (The mode-size
test was added after the mutation pass, below; the gate ran it.)

**`subTest` labels:** builtins (the guard ran).

**Mutation pass** (scratch copies, deleted). 12 non-equivalent mutants, all killed:
- **The reviewer's two:** the gather reading the wrong mode's index (the index rows reversed); a scatter dropping
  duplicates, on numpy (`out[index] += eta`) and on jax (`.at[index].set`, killed by the jit test, whose index
  repeats).
- **The others:** the gather without the move of `W`; the entries `W` read as if the sample were input jets; the
  `keep_W` scatter wrong (killed by its step test); the entries path through a short node (killed at import); the
  scatter reading `D_dot` for its size; the entries zero names removed; a short node kept (killed at import); the index
  cast where it is put among the values; the scatter's mode size taken as `max(N, n)`.
  - That last one first **survived**: every problem has `n < N`. `test_the_mode_size_is_that_of_D` (a Tucker rank 5
    over a mode of size 2) now kills it.
- **One equivalent mutant, not counted:** the index left in the floating set. It is then counted for the dtype (which
  ignores integer arrays) and cast, but the next line puts the uncast index among the values, so nothing changes; the
  mutant "the index cast where it is put among the values" is its non-equivalent form.

## 2. The gate

The whole gate at **`-n 2`** (the new setting; `--detach`, then `run_gate.sh wait`), with nothing else running. The
environment was python 3.11.15, numpy 2.4.6 and jax 0.10.2. The summary, copied:

    step                result counts                                                      seconds   peak_MB  minavl_MB
    suite               PASS   1770 passed, 5 skipped, 1 xfailed, 1 warning, 81705 subtests passed   1804.2      8018       4138
    doctest-modules     PASS   314 passed                                                     46.0      2029      10030
    doctest-quickstart  PASS   1 file(s), 0 with failures, 0 failed example(s)                 2.5       276      11782
    doctest-pages       PASS   48 file(s), 0 with failures, 0 failed example(s)               14.3       365      11702
    examples            PASS   14 examples, 0 failed                                           378       639      11434
    sphinx              PASS   0 warning/error line(s)                                      1028.9       631      11304

**No failure.** Against P9a (1,755), the suite has 15 more tests: `test_sampling_entries.py` (11), the derivation test
of the graph (1), the bridge's entries kind (2), the jit of the entries kind (1).

**Memory:** **the lowest MemAvailable is 4.14 GB** at `-n 2` (1.56 GB at `-n 3` for P9a); the peak RSS summed over the
workers 8.0 GB. The suite took 1,804 s (1,367 s at `-n 3`).

No process, lock or file of the gate is left in the repository.

## 3. Departures and choices

1. **`'w'` on an entries call is the one-hot vectors** (ruled in round 1; `P9b_RESPONSE_r1.md`): a row of entries'
   graph, so the closure builds it only when wanted and `D` makes its dtype the result dtype.
2. **The scatter reads `D` for the mode size** (`_ENTRIES_SIZED`): a step makes shapes only from its operands, and
   `N` is not among them. On a corewise point `D` is `B`. The read is of a store, so it costs nothing.
3. **No reshape in the scatter**: `np.add.at` and `.at[].add` take the `W`-shaped index array as it is, so `W` is
   never flattened (the contraction layer's sharding rule).
4. **Entries' twin path goes through the full `eta_tilde`** (`_Jp.eta_tilde` exists for entries, not for probe or
   apply), since entries has no short nodes.
5. **For P11** (the reviewer's note): `xi_gather_step` reads `D[..., index]`; on a padded uniform `D` the index stays in
   the real columns (`index < N_i`), and the scatter writes only there; the masks are P11's.
