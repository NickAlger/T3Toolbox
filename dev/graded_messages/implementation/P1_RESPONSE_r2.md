# P1, the implementer's answers to the review of round 2

To `P1_REVIEW_r2.md` (verdict SIGN OFF, two minor findings, both in the tests). 2026-11-17. Both accepted. Only
`tests/test_jet_contract.py` changed; the library is untouched since the gate of round 1.

**10. The start shift of a readout beyond the 41 strings was tested in float64 only. Accepted.**
`TestAgainstReference.test_more_jets_than_the_41_have` now runs each of its two strings (a pair with an affine
operand and a further full jet; four full jets) in float64 and in float32, at orders 1, 2, 4 under both forms,
and asserts the result's dtype as well as its value (1e-12 and 1e-5 against a float64 reference).
Evidence: the mutant that builds the start shift's zeros in the default dtype
(`xnp.zeros(carried[:n_slots - 1 - left].shape)` in place of `zeros_like`) passed everything before and now
fails 3 subtests (the float32 result of `'scan'` comes out float64). The file was restored and diffed.

**11. The expected-failure test would hide a broken helper. Accepted.**
New plain test `TestOrderInsideAStep.test_on_numpy_what_a_product_step_pairs_first_today`: for each of the 20
product strings it runs the same helper (`_first_pairs(string, np)`) and asserts that there is exactly one step
with all slots, three operands, one carried jet, one operand with no order group, and a first pair of two
positions; that for the four `nu` strings with an affine leaf the first pair is the core with the coefficient;
that for the other 16 it is the carried jet with the core; and it pins the count, 16 against 4. So a helper that
raised, returned nothing or misread the roles now fails a test that is not an expected failure. This test will
fail at the same moment the marker turns into an unexpected success (when Part 2 fixes the order); both
docstrings say so, and the marker's says to update the counts then.

**Runs.** `run_gate.sh -l /tmp/gate_P1_r3_quick quick tests/test_jet_contract.py
tests/test_contractions_interpreter.py tests/test_contractions_sharding.py`:

    step                result counts                                                      seconds   peak_MB  minavl_MB
    quick               PASS   52 passed, 1 xfailed, 15357 subtests passed                   120.2       791       6920

and the module doctests of `contractions.py`: 3 passed. `tests/test_dispatch.py` was not touched and not rerun.
The whole gate was not rerun (a test-only change; the last whole gate, on the tree before this change, is in
the handover).
