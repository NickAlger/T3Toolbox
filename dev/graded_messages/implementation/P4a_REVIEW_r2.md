SIGN OFF

# P4a review, round 2

Reviewer, 2026-11-19, on the working tree of `graded-messages` (the implementer's uncommitted changes on `96196427`),
against `P4a_RESPONSE_r1.md` and the code as it now stands. All four findings of round 1 are resolved. I checked
each one in the files, not in the response's account of them.

## The four findings of round 1

**1. The lone readout and the unreachable-corner fallback under jit. Resolved.**

`tests/test_dispatch.py`, `test_jit_jet_contract` ends with two rows, at the cost I proposed:
- the lone readout `'JWCi,Cio->WCo'`, jet sizes `(2,)`, at `(3,)`;
- the pair fallback `'JWCa,JWCi,JWCb->Caib'`, jet sizes `(2,)` three times, at `(6,)`.

Each row:
- runs under `'loop'`, in float32;
- first asserts that numpy gives zeros, so the row is known to reach the branch;
- is jitted with every operand traced, and once with each operand in turn closed over as numpy;
- requires jax leaves, float32, and exact equality with numpy's result.

The lambdas bind the loop variables and are called in the same iteration, so no late-binding slip is possible.

I re-ran my three host-numpy mutants on a fresh scratch copy (`python -B`, `PYTHONDONTWRITEBYTECODE=1`, deleted
after), against `tests/test_jet_contract.py`, `TestJetContractDerivedStrings` and the two `jet_contract` tests of
`test_dispatch.py`. In round 1 all three passed. Now:

| Mutant | Result |
| --- | --- |
| `numpy.zeros_like` for the lone operand's zeros | caught (2 failed) |
| the same for the fallback's looped coefficients | caught (3 failed) |
| `numpy.asarray` on the fallback's empty pair slice | caught (3 failed) |

**2. `n_contractions` as evidence. Resolved.** Its docstring now calls it "a regression pin of the rule as built".
It names what it restates (the choice of carrier, pair and scanned partner, the skip of empty ranges, the one call
of the zeros). It points to P1's table as the independent pin. `test_the_number_of_contractions_follows_the_rule`
says the same.

**3. `_jet_roles`' signature. Resolved.** The comments are now in blank-line-delimited groups, each group aligned
(`signature_style.md`). The arguments are unchanged.

**4. The `J`-alone rule. Resolved.** The `jet_contract` docstring now says it, beside the rule on a string with no
`J`.

## Nothing else changed in the library

Apart from the docstring sentence and the signature layout, the code is the code I reviewed in round 1. So the
independent checks of round 1 stand:
- the roots-of-unity oracle over 1,296 calls per seed, worst error 2.3e-15;
- slots beyond the box filled with NaN leave every result unchanged;
- the edge cases, the counts against P1 and the jaxpr program sizes;
- 25 of 26 code mutations caught, the one missed being equivalent;
- the floor env's run of `tests/test_jet_contract.py`.

The module doctests of `contractions.py` pass on the tree as it stands (3 passed).

## Not rerun

The full gate. The change since the implementer's green gate is two test rows and three comment or docstring
edits. The rows run in the slice's dispatch test, which passed in the mutant runs above for everything but the
mutated branch. Before committing, Claude may want to run the floor leg's `test_dispatch.py -k jet_contract`: the
two rows add 7 jits to a test that is already the slowest there.

The working tree is as I found it, apart from this file. The scratch copy is deleted, and nothing of mine is left
running.
