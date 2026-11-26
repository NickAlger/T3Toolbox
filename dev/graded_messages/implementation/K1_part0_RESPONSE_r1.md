# K1 part 0: response to review round 1 and the orchestrator's fix

Implementer (the second), 2026-11-26, to `K1_part0_REVIEW_r1.md` (SIGN OFF) and the orchestrator's four fixes. Only
these four hunks changed.

1. **Comment column, `docs/fitting_and_optimization.md:149-150`.** The two trailing comments are aligned again (one
   column, the one line 150 has).
2. **Blank line, `t3toolbox/backend/uniform_fitting.py`.** `pack_batch`'s docstring has its blank line before the
   doctest again.
3. **The untested branch of `_require_draw`.** `TestOracleSeam.test_optional_outputs_are_checked_where_needed` gains
   the case of a batch with `take` and a non-integer `n_measurements` (`_FloatCountBatch`, `60.0`), on `mc_sgd` and
   `adam`: it passes (1 passed, 8 subtests). The reviewer's mutant (the `isinstance(n, numbers.Integral)` check
   removed, in a scratch copy) now dies: 2 failed.
4. **`backend/fitting.py`'s module docstring.** The re-flow was not on purpose: a wrap script re-filled the whole
   paragraph. It is now as `prose_edits.md` §1.1 has it: the changed lines only, the line holding the new sentence's
   end split after "before any" so that every line after it is as at `HEAD`, byte for byte, and the closing `'''` on
   its own line. `prose_edits.md` §1.1 now states that split. The rule is noted for the parts to come: a sentence
   rewritten in a docstring leaves the unchanged lines as they are; a paragraph is re-wrapped only where an inserted
   sentence forces it.

Reruns: the test above; the module doctests of `backend/fitting.py` and `backend/uniform_fitting.py` (8 passed);
the page doctest of `docs/fitting_and_optimization.md` (passes). Nothing is left running.
