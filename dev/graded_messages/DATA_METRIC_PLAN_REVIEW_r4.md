SIGN OFF

# Review of the revised `DATA_METRIC_PLAN.md` §9 and what follows from it, round 4

Reviewer, 2026-11-22. I checked the revision and `DATA_METRIC_PLAN_RESPONSE_r1.md`'s "Round 3" against the tree. I
also ran one script, `scratchpad/r2/amb.py`, on the ambient transposes under the input law; round 3's scripts cover
the rest.

All seven round-3 findings are resolved, and the tables and laws are exact. Four minor points remain. None needs
another round; the implementers can settle them in M1's design note.

## The round-3 findings

| # | Finding | Status |
| --- | --- | --- |
| 1 | 9.2's premise | **Resolved.** The converters become frontend-private and stay backend-public, on the razor's ground (cited: `sampling_variables.py:899-905`, the `HeldVariables` slots "turned", confirmed; `sampling_definitions.py:86-89`). O-20 is named as superseded, and the §3 zero-check row allows for both answers to Q9. The blast radius is given in lines: 26 / 18 / 2 / 71 / 16 / 7 / 0. That agrees with my reference counts. |
| 2 | The swap | **Resolved.** 9.3 says "what no scaling law catches", with the `λ -> 1/λ` argument. The catch list is correct, and so is the caveat that the laws are homogeneity tests. |
| 3 | The parameter-side law | **Resolved.** The 9.3 table matches what I derived and measured. Entries' law is included, with its `c` and `ω` by duality (correct). The caveats are kept: metric-free maps only, sharing groups, padding. The rescalings are named as law instances. I checked against the code that `t3_unscale_coordinates` multiplies column `j` of core `i` by `scales[i][j]`, which is the inverse of `B -> B Δ^-1`. |
| 4 | Vocabulary and table | **Resolved.** 9.0 fixes one symbol, `'`, with a dictionary (`Y' = P`, `Y = P'`, …), and §1 gains the matching line. No `*` dual remains in the plan (checked by grep). `P_in,i` is on its own box. Which pairing is meant is confined to two named places. The sign slip is fixed (`λ^-(n-k)`). The missing rows are added. |
| 5 | 9.1's scope and placement | **Resolved.** The grammar changes in M1, in `signature_style.md`. The engine's types go into `sampling_variables.py`'s own table, not into `dev/`. |
| 6 | Slicing | **Resolved,** with the comments-only M5 (assessed below). |
| 7 | Q9 to Q11 | **Resolved.** Each question is a real choice with a recommendation. |

## Spot checks

**The dictionary against §1.**
- `J: X -> Y'` lands in `P` (apply) and `⊕ P⊗V_i'` (probe).
- `c ∈ Y = P'`.
- `W: Y' -> Y` maps the `P` law to the `P'` law once `ω` moves by `λ^-k`: `ω² r -> λ^-2k λ^k ω² r = λ^-k ω² r`.

This is consistent with §§1 to 8, which keep `Y`/`Y'` and are tied to §9 by the dictionary.

**The input law on `W` and `ω`.**
- probe: `ω_i Δ_i`, `r_i Δ_i^-1`, so `W r_i -> Δ_i W r_i`, the `P'⊗V_i` law;
- apply: `ω` unchanged;
- entries: `ω · ∏ Δ_i[n_i]`.

All correct.

**The ambient transposes, by script.**
- Apply: each CP factor scales by `Δ_i`, as the table says.
- **Entries: not per factor.** Its factors are the one-hots `e_{n_i}`, which do not change, with `c` folded into
  factor 0 (`False` per factor). The law holds for the **tensor** they represent: `(⊗Δ_i)` applied to it (`True`).

See minor point 1.

**The rescalings.** `rescale_direction` is the jet law on the inputs and on `y` (`data_preparation.py:284-285`).
`rescale_coordinates` and `t3_unscale_coordinates` are the input law. Both confirmed in the code.

**M5's trade-off.** The argument is sound:
- the notation is reviewed once;
- M4 stays a rename;
- the ~50 functions that no slice touches need a slice anyway.

The cost, re-editing the comments on M4's 42 signature lines, is small and mechanical. M5 changes no line of code;
its new tests, the laws on the functions no earlier slice covers, are test code. That is consistent.

## Minor points (for M1's design note)

1. **The ambient row of the 9.3 table.** "`Δ_i ·` per factor" holds for apply and probe. For entries it holds for the
   CP tensor (`⊗_i Δ_i` applied to it), not factor by factor. The law helper must compare entries' ambient output as a
   tensor.
2. **The converters' type.** The two sections type them differently:
   - 9.1 types them `Y' -> Y` in turned coordinates: `flip ∘ W`, `weight` required.
   - 9.2 calls them converters "between slot coordinates and the engine's turned form": `Y -> Y(turned)`, which is
     the `weight=None` case.

   Both are true. State the first as the type, and the second as its `weight=None` reading. M1's law test then covers
   both: `r` with `ω` moved maps to the turned `P'` law, and `c` with `weight=None` maps to `λ^-(n-k)`.
3. **The tilde variables are not elements of `Y`.** 9.1 says `w_tilde` and every tilde variable are "`Y` in turned
   form". That is exact for `w_tilde`. The others are partial contractions: turned `P'` jet axes tensored with an edge
   or a bond space. Write "carry `Y`'s jet axes, turned". Their jet law is still `λ^-(n-k)` on those axes.
4. **Readability.**
   - The 9.1 table's rows leave out the point or frame argument (the forwards are polynomial in it). The header could
     say "at a point `x` (or a frame)".
   - M1 is now the largest slice: the rule, `W`, the converters, 9.2, the law helper and the grammar. Its design note
     may split off 9.2 and the grammar if review gets heavy.
   - The plan (467 lines) has no short summary at the top. Three lines saying what the arc does (types the transposes,
     makes the weight elementwise and `W` public, adds types and law tests) would help the maintainer before §1's
     mathematics. This is optional.
