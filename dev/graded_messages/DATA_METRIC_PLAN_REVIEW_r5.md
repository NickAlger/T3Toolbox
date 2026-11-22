NOT YET

# Review of the notation pass on `DATA_METRIC_PLAN.md`, round 5

Reviewer, 2026-11-22. I checked the plan as it stands in the working tree, after `ccf10b0a`, together with
`DATA_METRIC_PLAN_RESPONSE_r1.md`, "Round 5 (notation)".

The substitution of `M` for the metric is clean. Four findings remain, none blocking, but each changes what the
maintainer reads or what M1 builds:

1. The `J -> P` rename misses most of its own scope.
2. M1 has become too large.
3. The 9.0 table leaves three letter collisions inside the plan.
4. Q1 understates the main cost of its recommended name.

One short revision fixes all four.

## The checks asked for

**No `W` meaning the metric is left.** An identifier-boundary scan lists every `W` token: each is the sample stack.
The one exception is a quotation, at `:314` (finding 5).

**No `M` meaning anything else.**
- Every `M` token is the data metric. The manifold `M` appears only as the stated collision (`:380`).
- The slice names `M1` to `M5` are not the token `M`, but they now read oddly beside it, as in "M1 for `M`"
  (`:523`). This is minor and optional; the slice names can stay.

**§1 against §9.** The two sections are consistent:
- §1's table and equations (`f = ½<M r, r>`, `g = J^T M r`, …) and the scale paragraph (O-21) match 9.0, 9.1
  (`apply_data_metric: Y' -> Y`) and 9.3 (`M r` under the `P'` law and the `P'⊗V_i` law).

**No collision with the new letters.** No contraction string in `t3toolbox/` or `tests/` uses `P` or `Q` as a group
(AST scan). So the rename collides with nothing in `contract` or `jet_contract`.

---

## Findings

### 1. SHOULD FIX: the `J -> P` rename covers the strings but not the code that spells the group

**The house rule.** CLAUDE.md requires local names to encode the axis layout "in the name suffix, matching the
grouped-subscripts scheme". So renaming the group in the strings without renaming the locals leaves the engine
self-inconsistent: the strings would say `P` and the names around them `J`.

**AST counts at the working tree**, against the plan's grep counts:

| Item | Plan | Measured | Note |
| --- | --- | --- | --- |
| `jet_contract(` calls | 173 lines | 153 calls | the difference is docstring and doctest lines; fine as "about" |
| grouped strings with `J` | about 216 | 183 full einsum literals | |
| `GroupLengths(J=…)` | 32 | 25 in code | the rest in docstrings |
| `lens.J` | 14 | 9 in code | the rest in docstrings |
| body-local names with a `J` group suffix | **not counted** | **253** | `mu_JWCa` 54, `nu_JWCb` 41, `w_JWo` 40, `xi_JWCi` 37, `nu_JWKCb` 13, `eta_JWCi` 7, `jet_JX` 7, `top_JWKCi` 6, … (`held_Jp`, the Jacobian, excluded) |
| the keyword `len_J` | **not counted** | **33, in 8 files** | **on a public backend function**, `ut3_operations.pack_if_ragged(…, len_J=…)`; also `uniform_fitting.pack_sample` and `pack_batch`'s docstrings |
| `'J'` literals in tests | **not counted** | **22** | `_jet_contract_cases` 9, `test_jet_contract` 8, `test_contractions_interpreter` 4, `test_contractions_sharding` 1: machinery that reads the group out of strings |

**What would resolve it.**
- Add the three missing rows to M1's notation work, or state why any of them stays `J`.
- The `len_J` keyword in particular is a public backend name.
- `TestGroupLengths` already checks that the retired group `T` raises (`test_jet_contract.py:827`). Add `J` there.

### 2. SHOULD FIX: split the rename out of M1

M1 now carries all of this:
- the reading rule, `apply_data_metric`, `check_weight`;
- the converters and 9.2, with the O-20 deletions;
- the factorial conversions and the D4 docstrings;
- the law helper with the full tables;
- the comment grammar;
- the glossary and naming entries;
- the `J -> P` rename: about 450 identifiers and strings in roughly 15 files, most of them in `sampling_steps.py`,
  which M2 then edits for the weight.

**The cost.** The rename is pure, mechanical and numerically neutral. Mixed into M1's design work, it would bury M1's
real diff, and M2 would review against a moving `sampling_steps.py`.

**Recommendation.** Make it a slice **M0, before M1**:
- the rename (finding 1's full scope), the glossary and `naming_conventions.md` entries;
- the "old `J` raises" test;
- gate: the suite green with unchanged numbers.

M1 then starts on the renamed tree. This follows the maintainer's own working rule: slice big work into reviewable
units.

### 3. SHOULD FIX: the 9.0 table leaves three letter collisions inside the plan

The maintainer's notation decision exists to retire collisions, so the plan should not keep any of its own.

| Letter | First meaning | Second meaning |
| --- | --- | --- |
| `s` | the minibatch scale in `M = s·diag(ω²)` (`:30, :45, :380, :419`) | the jet variables `s_j` in the jet law (`:472`, and the library's own usage) |
| `R` | the real numbers (`:390, :413`; the oracle row's scalars) | the right store `R` of the frame (`:492`, the input-law table) |
| `D` | `diag(k_1!…k_m!)` in §1's Coordinates paragraph (`:57`) | the down store `D_i` (`:491`) |

**What would resolve it.**
- Write `ℝ` for the reals.
- Give the scale and the factorial diagonal letters free in the glossary. For the scale, the code's name
  `minibatch_scale` in prose, or `n/b` written out. For the factorial diagonal, for example `Φ`, or `k!` written out.
- Add the plan's other working symbols to 9.0, since the table is meant to be the reference: `ω`, `m`, `k`, `λ_j`,
  `Δ_i`, `N_i`, and the edge size `e`.

**Two smaller notes on the table.**
- `J_in_i` keeps the retired letter as a size name. That is defensible, like `F`, and the plan says so. `F_in,i`
  would parallel `F`; this is a choice for M0's design note, not a blocker.
- The glossary's Symbols table already has a crowded `P` row: the derivative direction `pp`, `P⁺`, `P_X` / `P_F`,
  the core parameter space, and T4S's left core. It also uses `F` for a frame (`P_F`, `T(F)`). M0 *extends* the `P`
  row and notes the frame `F`. Its preamble ("Uppercase `C`, `W`, `K` are reserved for the batch blocks") gains `P`
  and `Q`.

### 4. SHOULD FIX: Q1, `apply_data_metric`. The reasoning holds, but the blemish is understated

**What holds.**
- A Riesz map is an isomorphism. `M` is not one where a weight is zero, and a loss Hessian may be singular or
  indefinite. So `data_riesz_map` overclaims, as the plan says.
- Read strictly, "metric" also presumes positive-definiteness. The decisive reason for (a) is therefore that "the
  data metric" is the maintainer's chosen name for `M`, not that "metric" is more exact. Say so; it is the honest
  argument, and it still wins.

**What is understated.**
- The plan's own blemish is "apply" twice in `apply_data_metric('apply', …)`. The larger problem is the name grammar.
- A leading kind name is the library's pattern for kind-specific operations: `apply_transpose`,
  `apply_corewise_tangent`, `apply_ambient_transpose`, `t3_apply_*`.
- So `apply_data_metric` parses as "the apply kind's data metric", next to a probe or entries call that has no
  `probe_data_metric`.

**What would resolve it.**
- State this cost in Q1.
- Either keep (a) knowingly, or offer one name that avoids a leading kind word. `weigh` (d) avoids it; a verb-first
  alternative such as `data_metric_times`, or the noun form (c) with its stated objection, are others.
- The writer chooses the recommendation. The question must show the trade-off.

### 5. MINOR

- **`:314`** quotes the maintainer's old notation ("`W: Y' -> Y` is the Riesz map"). It is a quotation, but it is the
  one `W`-as-metric left in a plan that just retired it. Write: "from your typing of the Riesz map (then written
  `W`)".
- **`:9-10`** has a grammar slip: "the names of the Hessians and of the backend converters; an non-diagonal `M`". Make
  it "a non-diagonal `M`".

## Summary for the writer

1. Count and include the suffix locals, `len_J` and the test literals, or state why any of them stays `J` (finding 1).
2. Move the rename and the glossary entries into an M0 (finding 2).
3. Remove the `s`, `R` and `D` collisions, and complete the 9.0 table (finding 3).
4. Put the kind-prefix cost into Q1 (finding 4).
5. Fix the two minor slips (finding 5).
