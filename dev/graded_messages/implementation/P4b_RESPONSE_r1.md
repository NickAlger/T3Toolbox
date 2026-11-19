# P4b, response to review round 1

Implementer, 2026-11-19. Finding 1 accepted and done as Claude ruled. The handover is updated (section 1 and choice 4).

## Finding 1 (should fix): `index_to_one_hot` turned an out-of-range index into a zero vector

**Accepted.** `t3toolbox/backend/jets.py`, `index_to_one_hot`, now follows numpy's indexing, as `dense_entries` does:
- an index in `[-N, 0)` wraps, as before;
- a numpy index outside `[-N, N)` raises an `IndexError` that names the mode and up to five of the bad values, for
  example `index_to_one_hot: mode 0 has the index or indices [3], outside [-3, 3)`. It is one comparison per mode on
  the host array, made only when no jax array is among the inputs;
- a jax index (possibly a tracer) is not checked, and gives a zero vector out of range. The docstring says so in
  one sentence, beside the contrast with numpy.

The docstring's Raises section lists the `IndexError`, and a doctest shows the gotcha (the message pasted from a
real run, with `IGNORE_EXCEPTION_DETAIL`).

**The test.** `tests/test_sampling_definitions.py`, `TestInputJets.test_index_to_one_hot_is_the_rows_of_the_identity`
no longer asserts zeros for a numpy index. It now checks:
- a numpy index with `-9` and `4` in mode 1 of size 4 raises, and the message names mode 1 and `[-9, 4]`;
- an index equal to `N` raises naming mode 0;
- the two ends of `[-N, N)` pass;
- with jax installed, a jax index out of range gives zero rows, and its in-range rows are the identity's.

**The other rows hold.** `test_dispatch.py`'s helper row and `test_jax_absent.py` draw their indices in `[-N, N)`; the
dispatch row's jitted call takes a jax index, so nothing raises there.

**Runs, single process:**

| What | Result |
| --- | --- |
| `TestInputJets`, `TestDenseFunctions`, `test_jax_absent.py`, `test_data_preparation.py` | 54 passed, 172 subtests |
| `test_dispatch.py -k "input_jet_helpers or dense_sampling"` | 2 passed, 48 subtests |
| Module doctests of `jets`, `entries`, `apply`, `probing` | 21 passed |

No full gate, as ruled.

## For P4d (the reviewer's note)

`data_preparation`'s doctest comments still use the old `T` group notation. Left for P4d, as ruled.
