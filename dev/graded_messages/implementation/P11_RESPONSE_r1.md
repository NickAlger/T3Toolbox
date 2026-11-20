# P11 response, round 1

Implementer, 2026-11-20, to `P11_REVIEW_r1.md`, with the coordinator's rulings. Nothing is committed.

## Finding 1 (should fix): the entries index's leading `d`, on both layouts. Done.

- **The check.** Step 2 of `sampling_variables` raises a `ValueError` on an entries call whose index is not
  `(d,) + W`, with `d = len(frame[0])` (the number of Tucker cores, or the supercore's leading axis):
  "the entries kind's index has one row per mode, shape (d,) + W with d = 3; got shape (1, 3)".
- **Where it sits.** It runs on both layouts, before the uniform index is normalized. It is not part of the removable
  `_check_stacks`: it is the sample's length, which the engine reads the sample by.
- **The cases it closes:**
  - the uniform leading 1 that broadcast to `T[i, i, i]`;
  - the uniform leading 2 and 4 that raised bare broadcasting errors;
  - the ragged leading 4 that was silently cut (since P9b).
- **Test.** `test_the_index_has_one_row_per_mode_on_both_layouts` (`test_sampling_uniform.py`): leading 1, 2 and 4
  each raise the message, on ragged and on uniform. It passes on both environments.

## Finding 2 (should fix): the one-hot `'w'`'s layout is asserted. Done.

`test_one_hot_w_has_the_padded_length` now asserts that `'w'` is an `np.ndarray`: the packed form, not a tuple of `d`
padded vectors. That kills the reviewer's `one_hot_not_packed` mutant.

## Rulings 3 and 4 (minor): done

- **The jit sentence.** The module docstring now says "the masks and `shape` are host values, static", and that under
  `jit` the frame is passed inside its object, where both are `aux_data`, or closed over.
- **"Every output is masked".** It now reads "every output but a `'w'` the caller passed (returned as given, its
  padding included)".
- **The eager-compile dispatch test** collects events only while it listens. It clears its flag in a `finally` and
  unregisters the listener where the jax has `unregister_event_duration_listener`; jax 0.4.30 does not have it, and
  there the flag stops the collecting.

## Runs

Single-process, no gate (as ruled):
- `test_sampling_uniform.py`, `test_sampling_entries.py`, `test_sampling_graph.py` and `test_sampling_stacks.py`, with
  the doctests of `sampling_variables`, `ut3_operations` and `sampling_steps`:
  - current: 67 passed (6,582 subtests);
  - floor: 67 passed.
- The dispatch tests of the uniform layout, the eager second call and the entries kind: 3 passed on each environment.
