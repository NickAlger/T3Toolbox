# P4a, response to review round 1

Implementer, 2026-11-19. All four findings accepted, as ruled by Claude. The library code changes only in two
places: one docstring sentence and one signature's comment layout. The handover is updated.

## Finding 1 (should fix): the lone readout and the unreachable-corner fallback were never compiled under jit

**Accepted.** `tests/test_dispatch.py`, `test_jit_jet_contract` now ends with two rows, at the cost the review named.

**The two rows:**
- `'JWCi,Cio->WCo'`, jet sizes `(2,)` at `(3,)`: a lone readout whose corner its operand does not reach (the
  `zeros_like` branch);
- `'JWCa,JWCi,JWCb->Caib'`, jet sizes `(2,)`, `(2,)`, `(2,)` at `(6,)`: the pair fallback, with the looped coefficients
  zeroed.

**What each row checks.** Each runs at one jet shape, under `'loop'`, in float32. Each first checks that numpy gives
zeros. Then it is jitted with every operand traced, and once with each operand in turn closed over as numpy: 7 jits
in all. Each must give jax leaves, float32, and numpy's zeros exactly.

**Evidence.** The reviewer's three host-numpy mutants, in a scratch copy, run against `test_jit_jet_contract`:

- `__import__('numpy').zeros_like` for the lone operand's zeros: **caught** (2 subtests fail);
- the same for the fallback's looped coefficients: **caught** (3 fail);
- `numpy.asarray` on the fallback's empty pair slice: **caught** (3 fail).

The copy was deleted afterwards.

**Runs:**
- `tests/test_jet_contract.py` and `tests/test_dispatch.py -k jet_contract`, single process: 48 passed, 1 xfailed,
  13,199 subtests, 90 s.
- The module doctests of `contractions.py`: 3 passed.

## Finding 2 (minor): `n_contractions` is not independent evidence

**Accepted.** The docstring of `_jet_contract_cases.n_contractions` now calls it a regression pin of the rule as built. It says what it restates: the choice of carrier, pair and scanned partner; the skip of empty pair ranges; the one call of the zeros. It names P1's table in `test_one_variable_with_the_library_shapes_makes_the_contractions_of_today` as the independent pin. The docstring of `test_the_number_of_contractions_follows_the_rule` says the same. The helper is kept.

## Finding 3 (minor): `_jet_roles`' signature broke the column rule

**Accepted.** The comments are now in blank-line-delimited groups, each aligned (`signature_style.md`):
`subscripts` alone in the first group; `jet_sizes` and `order` in the second, which shares its long type's column. The
argument order is unchanged.

## Finding 4 (minor): the `J`-alone rule was not in the docstring

**Accepted.** `jet_contract`'s docstring now says, beside "A string must have `J` on some operand": "and no input term
may be `J` alone (give it a group beside `J`, such as `W` of zero axes)." The error message is unchanged.

## The equivalent mutant the reviewer found

The pair ordered by size instead of string order. Agreed that it is equivalent: the fold is symmetric, and the
scanned readout is the same sum. String order is kept so that the choice is one rule; handover choice 2 says so.
