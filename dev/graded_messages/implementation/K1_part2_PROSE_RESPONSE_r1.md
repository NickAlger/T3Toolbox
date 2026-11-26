# K1 part 2, prose review round 1: the response

Implementer (the third), 2026-11-26 ~01:35 CT. All six findings accepted (and ruled so by the orchestrator); the texts
are revised in place in `k1/prose_edits.md` §3.

1. **`jets._metric`'s dtype (accepted).** §3.9 now states the rule as the definition: the result's dtype is the lowest
   floating precision of `r_X` and every weight given (the default float when none is floating); `r_X` and each given
   weight are cast to it; each square is formed in it; the product only when both are given; then `* r_X`; `gamma`
   last, cast to that dtype. Checked first by `k1/part2_metric_dtype.py` against today's `data_metric` on the four
   float32/float64 pairings of `r` and one weight, given as a full weight and as a block row: the rule's dtype is
   today's in all eight (float32 unless both are float64) and the values are bitwise equal. With two weights of
   different precision (r float64, block float32 / full float64, and the swap) the rule gives float32, the lowest of
   the three. The same sentence went into the note (§3, "Where the product is formed", and §4.3 under `_metric`'s
   header): the approved design made exact.
2. **The `Rule` rows (accepted).** The four rows take the table's existing overflow form (`:393-406`): the row stops
   after `reads`, the later columns on a continuation line in their own columns; written out in §3.4 as they will
   stand; no other row moves.
3. **Two after-texts in full (accepted).** `HeldVariables` (`:885-888`, re-filled from `:885` to the paragraph's end,
   since the new line would be 148 columns) and `_check_stacks` (`:1111-1114`, re-filled from `:1111`; today's
   `:1112` is already 151 columns): before and after, every line within 120 columns.
4. **The block weight's form (accepted).** §3.2: "one array (a number for entries), the same on both layouts, never a
   tuple or a list (a `TypeError`)".
5. **The steps' comment (accepted).** §3.6: "each axis z's size or, on a stack axis, 1"; the comment (all new text
   after its unchanged first line) re-filled within the column.
6. **§3.11 (accepted).** Item 3 names `test_block_sums.py::TestAgainstDenseSums::test_engine_and_oracle` (engine half
   rewritten in part 2 and asserted first; oracle half red until part 4, through the bridge); item 4 cites
   `test_optimizer_display.py:439`.

No file under `t3toolbox/`, `tests/`, `docs/`, `examples/` changed; no test run (one quick script).
