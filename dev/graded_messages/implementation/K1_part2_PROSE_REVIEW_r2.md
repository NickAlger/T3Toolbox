SIGN OFF

# K1 part 2, prose review, round 2

Reviewer 4 (Claude), 2026-11-26, 01:32 to 01:36 CT. I re-read the six points of round 1 in the revised
`k1/prose_edits.md` §3.

**SIGN OFF** on the six points as revised. Two minor line-layout defects remain, each introduced by a revision; the
build writes them as given below, and its review checks them.

## The six points

1. **The dtype rule (§3.9) is the definition.** I checked it against my own model of the rule, not the implementer's
   script, and against today's `data_metric` over 12 pairings.
   - The pairings: `r` int64, float32 or float64, against no weight, an int64, a float32 or a float64 weight.
   - The rule reproduces today's result dtype in all 12, the integer cases included. **The default-float clause is
     the case where no operand is floating**: integer `r` with no weight or an integer weight. Today's function gives
     the default float there (`np.dtype(xnp.result_type(float))`, float64 on numpy), and the clause says the same.
   - Integer weights do not count toward the lowest precision, today or in the rule.
   - Optional: today's `_lowest_precision_dtype` also caps the dtype at jax's own float on jax. If `_metric` reuses
     it, as it should, that is true and could be said in five words.
2. **The four `Rule` rows** are in the table's overflow form, checked against the real columns of
   `sampling_variables.py`:
   - in row `:385`, the `reads` column starts at 60, `_NONE` at 91 and the flag at 110;
   - in the four rows, the `reads` tuple starts at 60, and the continuation lines put `_NONE` at 91 and `False`/`True`
     at 110, as `:393-406` do;
   - the first lines are 97 and 98 columns.
3. The `HeldVariables` and `_check_stacks` after-texts are written in full, each re-filled from the changed line to the
   paragraph's end. The re-fill is forced.
4. "one array (a number for entries) ... never a tuple or a list": right. See defect A.
5. The steps' comment states the new rule ("each axis z's size or, on a stack axis, 1").
6. §3.11 names `test_block_sums.py::TestAgainstDenseSums::test_engine_and_oracle` and cites
   `test_optimizer_display.py:439`.

## Two minor defects introduced by the revision (the build writes them so)

A. **§3.2, `block_weight`:** the inserted "(a number for entries)" pushed a line to **137 columns** ("tuple or a list
   (a ``TypeError``). It sits on the leading axes ... so it needs no"). Re-filled, the three lines are 116, 114 and 89
   columns:

        slot), entries ``()`` (all of the data); one array (a number for entries), the same on both layouts, never a
        tuple or a list (a ``TypeError``). It sits on the leading axes of ``r`` (row ``i`` on the jet axes of mode
        ``i``), so it needs no placement against a stack of directions. ``None`` is ones.

B. **§3.9, `_metric`:** the inserted dtype sentence left a broken line, "``gamma``. It checks" followed by "nothing:
   its callers check ...". Re-filled, the four lines are 107, 95, 112 and 36 columns:

        ``gamma`` last, cast to that dtype (it does not raise it). ``None`` is ones for either weight and 1 for
        ``gamma``. It checks nothing: its callers check (the engine on entry, the public metrics of
        :py:mod:`t3toolbox.backend.jets`). Values in a packed ``r_X``'s padding come back multiplied by the weights;
        masking them is the engine's.'''

`_metric` is a new docstring, so no "before" is affected by either re-fill.

What part 2's build review must check first is the list at the end of `K1_part2_PROSE_REVIEW_r1.md`, plus A and B.
