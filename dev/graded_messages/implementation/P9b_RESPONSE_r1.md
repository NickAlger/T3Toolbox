# P9b response, round 1

Implementer, 2026-11-20, to `P9b_REVIEW_r1.md`. Nothing is committed.

## Finding 1 (should fix): `'w'` on an entries call is the one-hot vectors — done, the reviewer's way

- **A row of entries' graph.** `_ENTRIES_OWN_ROWS = (Rule('w', ('whole', 'w_one_hot_step'), ('index', 'D'), ...),)`,
  and `_ENTRIES_ROWS = _ENTRIES_OWN_ROWS + _entries(_APPLY_ROWS)`. The derivation is unchanged. The corewise
  derivation keeps the row, since it reads only `index` and `D`.
- **`w_one_hot_step(index, D)`** (`sampling_steps.py`, a whole step) is `jets.index_to_one_hot(index, [N_i of D])`,
  cast once to `D`'s dtype. `D` is read, so it counts for the dtype and is already cast to the result dtype. That holds
  even when `'w'` is the only name wanted. It keeps `index_to_one_hot`'s index semantics: a numpy index outside
  `[-N_i, N_i)` raises an `IndexError`, as the gather does.
- **The engine.** `'w'` goes through the closure when the kind's graph makes it, so the step runs only when `'w'` is
  wanted. For probe and apply (no row) it is the sample as given, as before. `_names` adds `'w'` only where the
  graph does not make it.
- **Test** (`test_sampling_entries.py::TestTheIndex::test_w_is_the_one_hot_vectors`):
  - on both call types, with a float32 and a float64 frame, wanting `('w',)` alone and `('w', 'z')`: `'w'` equals
    `index_to_one_hot(index, shape)` in the frame's dtype, exactly;
  - the step does not run when `'w'` is not wanted, and runs once alone when it is;
  - an out-of-range numpy index raises `IndexError`;
  - apply's `'w'` is still the sample object.
- The index test no longer asserts that `'w'` is the index object.
- `test_sampling_graph.py` checks `_ENTRIES_ROWS = _ENTRIES_OWN_ROWS + _entries(_APPLY_ROWS)` (own rows: `['w']`).
  Its derivation test now compares against `_entries(_APPLY_ROWS)`.

## Finding 2 (minor): the docstrings — done

- The engine's `want` parameter: `'w'` is the input vectors, the sample as given for probe and apply, and for entries
  the one-hot vectors of its definition, made only when wanted, in the result dtype.
- The engine's dtype paragraph says the same.
- The module docstring's entries section says that `w` is the entries kind's own row (`w_one_hot_step`, reading
  `index` and `D`).
- `_names` says so too.
- The zero-names message lists the names, `'w'` included, and needs no change.

## Runs

Single-process, no gate (as asked):
- current env: `test_sampling_entries.py`, `test_sampling_graph.py`, `test_sampling_apply.py` (the names comparison)
  and `test_held_variables.py`, with the two modules' doctests: 60 passed (5,432 subtests);
- floor env: the entries and graph tests and the doctests, 40 passed;
- the jit tests of the apply and entries kinds and the jax-absent tests: 7 passed.
