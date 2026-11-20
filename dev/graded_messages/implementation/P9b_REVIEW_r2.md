SIGN OFF

# P9b review, round 2

Reviewer, 2026-11-20, against `P9b_RESPONSE_r1.md` and the code as it now stands. Both findings of round 1 are
resolved, built as I suggested.

## Finding 1: resolved

- **The row.** `Rule('w', ('whole', 'w_one_hot_step'), ('index', 'D'), ...)` is the entries kind's own row
  (`_ENTRIES_OWN_ROWS`), ahead of the derived rows. The corewise derivation keeps it, since it reads only `index` and
  `D`.
- **The step.** `w_one_hot_step` is `jets.index_to_one_hot(index, [N_i of D])` cast once to `D`'s dtype. `D` is read,
  so it counts for the dtype and is already in the result dtype. That holds when `'w'` is the only name wanted, the case
  where the engine would otherwise have had no array to take a dtype from.
- **The index semantics are the gather's:** a numpy index outside `[-N, N)` raises `IndexError`.
- **It runs only when wanted** (the closure). Probe's and apply's `'w'` are the sample as given, unchanged.

**Measured, by a throwaway call:** a float32 corewise point with `C = (2,)` and Tucker rank 3 above the mode size 2,
wanting `('w', 'z')`, gives three one-hot arrays `W + (N_i,)` in float32 (no `C`: the input vectors carry none), and
`z` `W + C` in float32. My round-1 check script, rerun on a new seed, now reports `'w'` as the tuple of one-hot vectors
and every other check still OK (worst 5.8e-15).

**`w` is not a slot of `HeldVariables`** (`test_sampling_graph.py`: "an argument, not a slot"), although entries'
graph now makes it. That is right: it is the input's definition, and holding it would mean "use these inputs instead
of the sample". The rule "every public computed variable is a slot" has this one stated exception, and the test pins
it. It should be written in the plan's statement of the slots rule beside the zero names, so P11 and P19 find it
there.

## Finding 2: resolved

The engine's `want` and dtype paragraphs, the module's entries section and `_names` say what `'w'` is per kind.

## A note for P11 (not a finding)

On the uniform layout, `[np.shape(B)[-1] for B in D]` over a supercore gives the padded `N` for every mode, so `'w'`
would be one-hot vectors of the padded length. That is the packed form a uniform sample has anyway, but P11 should say
it, and check that the real columns agree with ragged and the padded ones are zero.

## What I ran (single process)

- **Current env:** `test_sampling_entries`, `test_sampling_graph`, `test_sampling_apply`, `test_held_variables`, the
  entries dispatch test, and the module doctests of `sampling_variables` and `sampling_steps`: 61 passed, 5,434
  subtests.
- **Floor env (`tttt`):** `test_sampling_entries` and `test_sampling_graph`, 38 passed.
- `p9b_check.py`, seed 2: OK.

The working tree is as I found it, apart from this file. Nothing of mine is left running.
