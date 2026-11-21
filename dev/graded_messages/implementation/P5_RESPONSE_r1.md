# P5, response to review round 1

Implementer, 2026-11-20, on `P5_REVIEW_r1.md` and the orchestrator's rulings of the same day. Nothing is committed.

## Finding 1 (should fix): the held-shape check pinned on one slot only

**Done.** `tests/test_sampling_graph.py`, `TestChecks.test_a_held_variable_of_a_wrong_form_raises`, has the reviewer's
case: `HeldVariables(mu=good, nu=bad)` wanting `('eta_D',)`.
- `bad` is the call's `nu_L`, so its bonds have the `L` store's sizes where `nu`'s are the `R` store's.
- `nu` is the second filled slot, and `eta_D` does not read it.
- The test pins the message `the held nu[1] has shape (3, 2); expected (3, 3)`.

The reviewer's mutant, `for name in held.keys()[:1]:`, is now caught. I ran it in a scratch copy, deleted afterwards.

## Finding 2 and ruling 2: the dtype of a held variable

**Done, as ruled.** Every output of a call is in the result dtype. The counted inputs include every held variable
that a row reads or that is wanted, and each is cast once. A wanted held variable is therefore returned cast, so the
returned tuple has one dtype. A held variable neither read nor wanted is not counted and not returned.

- **The engine** (`sampling_variables`, step 4): `held_counted = held_names & (read | set(want))`. Those values are
  counted for the dtype and cast. A held slot outside that set is not put among the computation's values at all.
- **The docstring** of `sampling_variables` states the rule where the dtype is stated:
  - every output in one dtype;
  - the counted inputs, the held variables read or wanted among them;
  - a held variable neither read nor wanted not counted;
  - `w` returned as given.

  The module docstring's "What a call does" says the same in one clause.
- **The test**, `TestResultDtype.test_a_held_variable_read_or_wanted_counts_and_is_cast` in
  `tests/test_sampling_variables.py`, pins four cases:

  | Held | Other inputs | Wanted | Result |
  | --- | --- | --- | --- |
  | float64 `mu` (read and wanted) | float32 | `mu`, `eta` | both float32; `mu` equals the cast copy bit for bit; `eta` equals the call with the float32 `mu` held, bit for bit (the rows read the cast value) |
  | float64 `mu`, neither read nor wanted | float32 | `nu` | float32 |
  | float32 `mu`, neither read nor wanted | float64 | `nu` | float64 |
  | float32 `mu`, wanted only | float64 | `nu`, `mu` | both float32 |

  Departure 3 of the handover (a read held value cast, an unread one as given) is replaced by this rule.

**Mutants** (a scratch copy, deleted). Four were run against the four new test files, and all four are caught:
- the held check on the first slot only;
- a read held value not cast;
- a wanted held variable not counted;
- every held variable counted.

## Finding 3: the documentation files

Settled by the orchestrator's ruling on the docstrings:
- `docs/conf.py` is reverted to the committed file.
- The prose type fields are real types. That is 22 entries in `jets`, `data_preparation` and `t3toolbox.fitting`,
  `sampling_variables`' `y`, and 31 return types in `sampling_definitions` that named `d`.

The handover's section 2 has the account and the rebuilt Sphinx.

## What I ran

- **Single-process, current env** (py3.11, numpy 2.4.6, jax 0.10.2): the four new test files and the module doctests
  of `sampling_variables` and `sampling_steps`, 39 passed, 772 subtests.
- **The floor** (`tttt`: py3.9, numpy 1.22, jax 0.4.30): the same, 39 passed.
- **The docstring change:** the module doctests of `jets`, `data_preparation`, `t3toolbox.fitting`,
  `sampling_definitions` and `sampling_variables`, with `test_jets`, `test_data_preparation` and
  `test_sampling_definitions` (its slow jets-two-ways, `Kp` and jax classes left out, since they read no docstring):
  195 passed.
- **Sphinx and the module doctests** through the gate script: the handover's section 2.
