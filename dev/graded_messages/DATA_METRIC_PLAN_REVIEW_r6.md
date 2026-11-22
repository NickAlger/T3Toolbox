SIGN OFF

# Review of the plan after the writer's round 6, round 6

Reviewer, 2026-11-22. I checked the plan at the working tree after `ffa4ad58` against "Round 6" of
`DATA_METRIC_PLAN_RESPONSE_r1.md`, with my own AST recount and greps.

All five round-5 findings are resolved. The counts reconcile. `γ`, `ℝ` and the slot-by-slot factorials introduce no
new collision in the code or the glossary. Four minor points remain; none needs another round, and each can be
settled in M0's or M1's design note.

## The round-5 findings

| # | Finding | Status |
| --- | --- | --- |
| 1 | The rename's scope | **Resolved.** M0's table covers the grammar, 202 strings, 153 calls, `GroupLengths` 25 / `lens.J` 9, the suffixed locals, `len_J` (public on `pack_vectors` and `pack_if_ragged`, unreleased) and the 22 test literals. It also says what stays `J`. |
| 2 | M0 before M1 | **Resolved.** A separate mechanical slice, gated on the suite unchanged, with M1 back to the rule and `M`. |
| 3 | Letters | **Resolved.** `γ` is the scale (`γ = \|W\|/b`, which also retires the second `n`). `ℝ` is the reals, and `R` is only the right store. `k!` is written slot by slot, with no letter. The test's `D = 3` reads `d = 3`, and §1's derivatives use the glossary's upright `d`. 9.0 gains `ω`, `γ`, `m`, `n_j`, `k`, `k!`, `ℝ`, `N_i`, `e`, `λ_j`, `Δ_i`. A grep finds no `s` scale, `R` reals or `D = diag` left. |
| 4 | Q1 | **Resolved,** as a trade-off table with the kind-prefix cost stated for (a). The recommendation moves to `data_metric`. See minor point 2 for one cell. |
| 5 | Minor | **Resolved.** The `W` quotation is gone; "a non-diagonal `M`". |

## The checks asked for

**The counts reconcile: 377 against 253.**
- My round-5 pattern required the group suffix right after the *first* underscore. It therefore missed every compound
  name: `mu_tilde_JWCa` 25, `nu_tilde_JWCb` 25, `mu_dot_JWKCa` 19, `nu_dot_JWKCb` 19, `xi_tilde_JWCi` 18, and so on.
- An AST scan over `Name` and `arg` nodes gives **23 names, 390 uses**:
  - `sampling_steps.py` 358, `data_preparation.py` 14, `ut3_operations.py` 8, `fitting.py` 2, tests 8;
  - minus the 13 `Name`/`arg` uses of `len_J`, which the plan counts in its own `len_J` row: **377**, exactly the
    plan's figure.
  - That leaves 22 names against the plan's 21, a naming detail.
- The plan's file list ("`sampling_steps.py` 358, `data_preparation.py` 14, the rest in `ut3_operations`, `fitting`
  and two test files") matches.

**No new collision.**
- `γ` appears nowhere in `t3toolbox/`, the pages, the glossary or the examples. As `gamma`, it appears only as a
  local in one `.tex` example (`padded_frame_cone.tex:267-272`), as the response says.
- `ℝ` and `k!` collide with nothing.
- The plan's derivative-coordinates line is right: `<c_d, r_d> = Σ c_d k! r_t` gives `c_t = k! c_d`, and
  `taylor_to_derivatives` multiplies by `k!`.
- One exception, minor point 1.

**M0's gate and scope.**
- The scope is sound. It is unreleased throughout, and no contraction string uses `P` or `Q` today, so the rename is
  collision-free.
- `GroupLengths`' positional construction is unaffected.
- The gate ("full suite green, every number unchanged; an old `J` raises beside the retired `T`") is right in kind.
  Minor point 3 is about its evidence.

**Q1's table is fair** in substance. (a)'s kind-prefix cost and (b)'s isomorphism point are stated plainly, and (d)'s
"reads wrong for a loss Hessian" is true. One cell leans: minor point 2.

**Readability.** The plan reads well:
- 537 lines;
- "In brief" at the top;
- one notation table, 9.0, that the rest of the plan follows;
- each slice with its oracle;
- eleven questions, each with a recommendation.

## Minor points

1. **`λ_j` collides with the regularizer's `λ`.**
   - 9.0 says "every letter here was checked against the code's letters and the glossary". But `λ` is the library's
     regularization strength: `IdentityRegularizer(λ)`, `½λ‖x‖²`, in the glossary (`:451-452`), in
     `fitting_and_optimization.md` (15 mentions) and in `regularization.py` (6).
   - It appears only in 9.3's test laws, so the cost is low.
   - Either pick a free letter for the jet-law scale, or note the collision in 9.0, as the plan does for `M`.
2. **Q1, option (c)'s "For" cell.**
   - The cell says the name reads "like `project(…)`". `project` is a verb, and `data_metric` is a noun.
   - The library's noun functions return the noun they name: `data_block_norms`, `equalizing_weight`, `affine_jet`.
     So `data_metric(…)` returning `M r` is the exception to that pattern, which is more than a "cold reading" risk.
   - Say so in the "Against" cell. The recommendation can stand: it is the maintainer's name with no kind word, and
     the required `r` makes the call unambiguous. But the maintainer should see the actual cost.
3. **M0's evidence of "a pure rename".**
   - "Every number unchanged" is asserted through tolerance-based tests. Those cannot show bit-for-bit equality.
   - The stronger and cheap evidence is mechanical: apply the rename by a script, and check that the diff, mapped
     back (`P -> J` in group positions and suffixes, `len_P -> len_J`), is empty apart from the new test and the docs.
   - Put that check in M0's design note.
4. **`J_in_i` is the last `J` of the jet notation.**
   - Whether it becomes `F_in,i` is left to M0's design note. That makes it a Claude ruling on a notation the
     maintainer decided himself.
   - It is cheap to ask, so add it as a one-line question (Q12) with a recommendation. Or record in 9.0 that `J_in_i`
     stays, and why: a size name, like `F`, and the input jets' sizes, not the box.
