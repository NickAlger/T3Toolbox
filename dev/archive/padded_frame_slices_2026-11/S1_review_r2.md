# S1 review, round 2: checking the round-1 fixes

Reviewer, 2026-11-02. Scope: the round-1 findings only (`S1_review.md`), against the current working tree and the
handoff's Round 2 section.

## Fixes checked

1. **MINOR 1 (rank-deficient docstring): fixed.** `linalg.py:950-959` now says that columns beyond the content's
   rank come from the SVD's trailing singular vectors, orthogonalized by the QR. They are always a valid
   completion ⊥ `P`, deterministic for a given input and backend, but `A` does not determine them: roundoff-level
   changes, such as in-span content, can rotate them by O(1). That matches the round-1 probe. The forward note to
   S3-S5 is in the handoff: projector-equality tests need hints with ≥ k directions per slot, and rank-deficient
   hints may only be asserted valid.
2. **MINOR 2 (jax validity): fixed.** The new subtests in `test_pinned_primitives_dispatch` cover both primitives
   under jit with x64. The seeds are all-zero, too-few, in-span(P) and none, with `C = (2,)`, `s = 2`, `k = 3`. The
   test asserts a completion residual ≤ 1e-14 and makes no numpy comparison. Re-run: `-k pinned` gives 1 passed
   with 10 subtests.
3. **NIT (seed j's): fixed** at `linalg.py:867`.
4. **NIT (boundary bonds): fixed.** `ranks.py:203-204` has the sentence.
5. **NIT (trailer): left to the orchestrator**, as intended.

Doctests of `linalg.py` and `ranks.py`: 15 passed.

## New, non-blocking

- **NIT:** two of the edited docstring lines are not rewrapped. `linalg.py:956` is 162 characters and `ranks.py:204`
  is 142. The surrounding text wraps at about 120. Rewrap them before committing; this is whitespace only and needs
  no further review round.

VERDICT: APPROVE
