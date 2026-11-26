# Response to `REVIEW_r2.md` (the writer, 2026-11-25)

The critic signed off on condition of N1 and N7. Revision 3 of the plan takes every new finding.

| # | Verdict | Where resolved |
|---|---|---|
| N1 | Accepted | §2.3: the mode axis of a full weight is exact in `probe_data_metric` (a tuple of exactly `d` arrays; a packed first axis of exactly `d`), the other axes "1 or `r`'s size". Listed in §13.3. |
| N2 | Accepted | The ragged `entries` methods stop calling `_as_index` in K3, the engine's ragged path converts each member; the function is deleted in K4 (§4.2, K3). |
| N3 | Accepted | `check_weight` is removed in K1 (§2.4, §3.1, §5); `index_to_one_hot` has its own row, changing in K3, its refusal naming `tuple(index)` (§3.3, §5). |
| N4 | Accepted | `widths` is normalized to a tuple of ints; the hand-built route is documented as trusted on the members' agreement; a foreign oracle passes `prepare_batch` unchanged and `fitting.pack_batch` raises for it (§4.3). |
| N5 | Accepted | `pack_data` goes in K1 with `_pack_data_space`; `check_data` is named as the one function edited twice; the K1 helpers refuse an array from K1; §5's sentence says what is still touched twice. |
| N6 | Accepted | `d = 3, 5, 8` (§6); `sum_over_probes` goes in K0 (K0's gate); the quote is given in full with its qualification and flagged for the maintainer, since the notes record it differently (§4.3). |
| N7 | Accepted | §13.3 lists the full weight's size-1 axes in the public metric and the dropped `DataSpace`; N1 removes the third. |
| N8 | Accepted | §4.4: the validation data, `N = x0.N`; the K6 row is gone. |
| N9 | Accepted, the first alternative | `ut3_unscale_coordinates` masks its output in K5 (it holds the masks), so it meets K7's test as it stands. |
| N10 | Accepted | §1.6 exempts the conversions; `form_of` reads the container only, and a 0-d array falls to the structural checks (§3.1, K2). |
| C2's two facts | Added | §13.7 states both. |
| E7 | Taken | K5's comparison with the tree before it is on values, not bytes. |
