SIGN OFF

# P4b review, round 2

Reviewer, 2026-11-19, on the working tree (uncommitted, on `45a92b1f`), against `P4b_RESPONSE_r1.md` and the code
as it now stands.

## Finding 1 of round 1: resolved

`index_to_one_hot` (`t3toolbox/backend/jets.py`) now does what Claude ruled:
- an index in `[-N, 0)` wraps;
- a numpy index outside `[-N, N)` raises `IndexError`, naming the mode and up to five bad values;
- a jax index, possibly traced, is not checked and gives a zero vector out of range.

The docstring says this beside the contrast with numpy, lists the `IndexError` under Raises, and shows it in a
doctest (`IGNORE_EXCEPTION_DETAIL`).

Two details I checked:
- **The check runs only when no input is jax.** It is a host comparison on a concrete array, so it never meets a
  tracer, and a jax index keeps the zero-vector behaviour the docstring states.
- **The integer-dtype and row-count checks still come first.** So a float index still raises `ValueError`, not
  `IndexError`.

**Mutating the bounds** (a scratch copy, `python -B`, deleted after; against `TestInputJets` and the module
doctests of `jets`, which pass unmutated: 16 passed):

| Mutant | Result |
| --- | --- |
| the upper bound `>= N` turned into `> N` | caught |
| the upper bound `>= N` turned into `>= N + 1` | caught |
| the lower bound `< -N` turned into `<= -N` | caught (both ends of `[-N, N)` must pass) |
| the check removed (`if False:`) | caught |
| the lower bound `< -N` turned into `< -N - 1` | **missed** (finding 1 below) |

## Finding

**1. Minor. The lower edge of the range is not pinned on its own.**
- *Place:* `tests/test_sampling_definitions.py`, `TestInputJets.test_index_to_one_hot_is_the_rows_of_the_identity`.
- *Evidence:* the mutant `index[i] < -N - 1` passes. Its one out-of-range-low value, `-9`, sits in the same mode as
  `4`, which already trips the upper check. With the mutant, an index of exactly `-N - 1` would wrap to `-1` and
  come back as a zero vector, the silent case the ruling removed.
- *Resolve:* one more assertion: an index of exactly `-N - 1`, alone, in a mode of size `N`, raises `IndexError`
  (for example `-5` in a mode of size 4).

  It does not hold up the sign-off. I would add it before the commit, since it is one line and guards the very
  behaviour this round was about.

## Nothing else changed

The rest of the slice is as I reviewed it in round 1. So these stand:
- the independent dense oracle (about 640 outputs, worst error 2e-15);
- the NaN check beyond the box;
- the recipe of entry 120;
- the packed `affine_jet`;
- the mutants of round 1 (5 caught, 3 equivalent).

The implementer re-ran the slice's tests, the dispatch rows and the module doctests (results in the response); my
own run of `TestInputJets` with the `jets` doctests passes.

The working tree is as I found it, apart from this file. The scratch copy is deleted, and nothing of mine is left
running. `data_preparation`'s old `T` notation remains a note for P4d.
