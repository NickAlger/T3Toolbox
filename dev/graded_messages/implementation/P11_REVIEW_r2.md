SIGN OFF: both round-1 findings are fixed and tested, the nits are done, and the slice is green on both envs.

# P11 review, round 2: Engine G (the uniform layout, sharding over `W`)

Reviewer, 2026-11-20, on `P11_RESPONSE_r1.md` and the updated tree (uncommitted, on top of `fcb20bcd`). Everything was
run single-process, one run at a time. The mutant copies have been deleted, and the working tree is untouched apart
from this file.

## Finding 1, the entries index's leading `d`: fixed

**The check.** Step 2 of the engine raises a `ValueError` naming `d` and the shape when `np.shape(index)[0] != d`, on
both layouts:
- `d = len(frame[0])`, which counts the Tucker cores on ragged and reads the supercore's leading axis on uniform;
- it sits before the uniform index is normalized, and outside the removable `_check_stacks`, as asked.

**My re-run of the round-1 cases.** Every one of these now raises the named `ValueError` on both layouts:
- leading 1 (the uniform broadcast to `T[i, i, i]`);
- leading 2;
- leading 4 (silently cut on ragged before);
- a jax index of leading 1;
- a scalar index.

**Mutant.** A new mutant removing the `d` comparison is killed: the new test fails in 6 subtests (leading 1, 2 and 4,
on each layout).

## Finding 2, the one-hot `'w'`'s layout: fixed

`assertIsInstance(w, np.ndarray)` is added. My round-1 surviving mutant, `one_hot_not_packed`, is now killed by
`test_one_hot_w_has_the_padded_length`.

## Nits: all done

- The docstring says the masks **and `shape`** are static under `jit`.
- "Every output but a `'w'` the caller passed" is reworded as asked.
- The `jax.monitoring` listener stops collecting in a `finally`, and is unregistered where the jax has the call. jax
  0.4.30 does not, and there the flag stops it.

## Runs

| Run | Current env (py3.11, numpy 2.4.6, jax 0.10.2) | Floor (py3.9, numpy 1.22, jax 0.4.30) |
| --- | --- | --- |
| The test files below | 67 passed, 6,586 subtests | 67 passed |
| Doctests of `sampling_variables`, `ut3_operations`, `sampling_steps` | 3 passed | 3 passed |

The test files: `test_sampling_uniform`, `test_sampling_entries`, `test_sampling_graph`, `test_sampling_stacks`, the
two uniform dispatch tests, and `TestTheEngineShardsOverW`.

**My `p11_check.py` on a new seed (2):**
- A: the TT-slack library point, with garbage everywhere; every name equals ragged; `z` and `f` match the dense oracle.
- B: the 98 holdable variables, each held with garbage padding.
- C: the index cases, all raising as described above.

All pass.

## Standing conclusions from round 1

- **The layering is sound and every layer is pinned.**
  - The sample's `N` mask, the start mask and the exit masks are redundant for the real parts, by construction.
  - Each is held alone by the first half of a permanent negative control.
  - A layer that is too strict is caught by the equivalence tests. A layer that is too permissive leaks only padding.
- **The departures stand agreed:**
  - every output masked on exit;
  - program size measured by jaxpr lines;
  - the masks and `shape` static under `jit`;
  - a negative index counts from `N_i`.
- **Mutation pass:** 26 mutants of my own over the two rounds, all killed.

P11 closes the additive phase. Ready to commit (gate as the coordinator rules).
