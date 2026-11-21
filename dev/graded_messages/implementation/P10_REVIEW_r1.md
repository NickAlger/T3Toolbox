SIGN OFF

# P10 review, round 1: Engine F (the stacks, the unsummed blocks, the dtype matrix)

Reviewer, 2026-11-20, on the working tree of `graded-messages` (the implementer's uncommitted changes on `9c8bab25`).
No finding. Built to my design review, the multi-axis-`W` gap closed by a genuine flattening. My own checks agree:
- `z` and `f` with a non-square `W` against a dense oracle;
- the unsummed `g`, `JtJp` and `Kp` per sample, off the diagonal included;
- the stack check, raising where it must and passing a consistent size-1 stack.

## The covering design really flattens `W`

`TestTheStackMatrix.oracle` calls the engine at frame element `c` and direction `k` on a sample whose `W` is reshaped
to one axis (`flat`, for the inputs, the index and `y`). `element` reshapes the stacked call's output the same way,
after indexing `k` and `c` at the positions the layout `J + W + [K] + C` says.

So a multi-axis `W` (`(2, 2)`) is compared with a single-axis one (`(4,)`): a different `GroupLengths.W`, different
expanded strings, different axis offsets in the sums, the gather and the scatter. A `W` offset bug cannot cancel. The
implementer's mutant (`f` and `Jp_norm_sq` summing only the first `W` axis) is killed by it, which shows it is not
cancelling.

Also present:
- **the maximal case per kind** (`K = (2, 3)`, `C = (2, 3)`, `W = (2, 2)`, `(1, 1)`; entries `()`);
- **a held stacked variable** (`mu` and `r` with `C = (2, 3)`, `K = (2,)`, `W = (2, 2)`, and a held `mu` missing its
  `C` raising);
- **the thin reference anchor.**

## My checks (`p10_check.py`; seed 0 on the current env, seed 1 on the floor)

The problem: `C = (2,)`, `K = (2,)`, and a **non-square `W = (2, 3)`**; probe at `(1, 1)` with mixed input jets, apply
at `(2,)`; a nonuniform weight; both directions.

1. **`z` and `f`, per frame element `c`, against my dense oracle.** These are the stacked call's outputs,
   `W = (2, 3)` kept as it is, against the roots-of-unity contraction of the dense tensor of `(D[c], L[c])`. `f[c]` is
   written from its formula. This checks the `W` and `C` layout and the sums against code that shares nothing with the
   engine.
2. **The unsummed blocks.** For `g`, `JtJp` and `Kp` with `sum_over_samples=False`, the block of sample `(a, b)`
   equals the summed call on that sample alone, at `(0, 0)`, `(1, 2)`, `(0, 2)` and `(1, 0)`: two off the diagonal on a
   non-square `W`, where the implementer's grid mutant first survived. Two more properties hold:
   - the blocks summed over `W` equal the summed call;
   - the shape is `W + K + C + core` (for `JtJp`'s `V` block, `(2, 3, 2, 2, 3, 3)`).
3. **The stack check.** A size-1 `C` on `U[1]`, a size-1 first `W` axis on `w[2]` (`(1, 3)` against `(2, 3)`), and
   `Vc[0]` without its `C` each raise a `ValueError`, naming the operand and the stack read off `D[0]` or `w[0]`. A
   consistent `C = (1,)`, `W = (1, 1)` passes. So it is exact equality, not broadcast compatibility.

Worst relative error: 4.1e-16 (current), 4.8e-16 (floor).

## The departure: two bodies per readout step

Agree. Each of the six readout steps has `if keep_W:` and the summed body, each literal the first argument of its own
`jet_contract` call. That is the house rule (a literal string at the call site, statically readable), and the
interpreter test's guard is right to require it. Literals held in names would hide the strings from the AST sweeps
that check identifiability and sharding.

The cost is some repetition in six steps, the same shape as the library's old `_summed`/`_unsummed` pairs. The
alternative, a name holding the chosen literal, saves lines but loses the guard. Two bodies is the right trade.

## The rest, confirmed

- **Six literals,** as derived in my design review: three `V`, three `H`, `'->WKC...'`. They are in `STRINGS` and swept
  by the jet-contract tests (reference, sensitivity, the two forms) and by the sharding test over `W`. The counts
  follow.
- **`keep_W` reaches only block rows** (`_is_block`, read off the shape table, the twins through `_table_name`), and
  joins the memo key of the site bodies. `f`, `r_block_sumsq` and `Jp_norm_sq` ignore it.
- **`_check_stacks` is removable as promised:** one function, one call (`sampling_variables.py`, the line after
  `_read_setting`), one test class, and one sentence in the docstring.
- **The dtype matrix** covers the four backends, jax under `jit` and float64 inside `jax_in_double_precision` (no
  global setting), with the cases of my design review: a read held float64 variable lowered, an unread one ignored, a
  jax one making the call jax, the integer index not counted, and entries' one-hot `'w'` in the result dtype.
  Running the jax column jitted (departure 2) is right: the rule is the same, and eager jax per contraction was 23 to
  72 s.

## What I ran (single process)

- **Current env:** `test_sampling_stacks`, entries, apply, graph, variables, `test_jet_contract`,
  `test_contractions_interpreter`, and the module doctests of `sampling_variables` and `sampling_steps`: 144 passed,
  1 xfailed, 30,932 subtests.
- **Floor env (`tttt`):** `test_sampling_stacks` and `test_contractions_interpreter`, 30 passed; `p10_check.py`, OK.

The implementer's 13 mutants cover my design review's list (the `W` offset, the grid, a literal with `K` before `W`,
the stack check made broadcast-compatible or missing an operand). My checks above test the same properties from
outside, so I made no further mutants.

The working tree is as I found it, apart from this file. Nothing of mine is left running.
