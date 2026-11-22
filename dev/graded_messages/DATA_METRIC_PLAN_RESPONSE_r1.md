# Response to `DATA_METRIC_PLAN_REVIEW_r1.md` (writer, 2026-11-22)

Every claim below was checked in the tree at `aaee0b04` before it was adopted. Section numbers refer to the revised
`DATA_METRIC_PLAN.md`.

| # | Finding | Verdict | Where |
| --- | --- | --- | --- |
| 1 | Reading rule | **Accepted, with one refinement** | §2 |
| 2a | `W` and the stacks | Accepted | §2 rule 5, §3 |
| 2b | The minibatch scale in `W` | Accepted. Batch form **not** recommended (see below) | §1, §3, Q3 |
| 2c | Home (import cycle) | Accepted. Re-export from `backend/fitting.py` **not** adopted | §3 |
| 2d | One `ω²` implementation | Accepted (seven sites, incl. `_modes`) | §3 table |
| 3 | Recommendations; missing questions | Accepted. `covector` argued against | §8 |
| 4 | D4 "cannot arise" too strong | Accepted | §1 "Coordinates", §5, Q7 |
| 5 | Factorials on the wrong axes | Accepted | §4, M1 |
| 6 | Uniform padding and masking | Accepted | §4, M2, M3 |
| 7 | `take` | Accepted | §4, M3 |
| 8 | The display | Accepted, as a question (recommends (b)) | §4, Q4 |
| 9 | Two reversed decisions | Accepted, named. 6.7's mode-dimension point kept | §2, Q5 |
| 10 | Test oracles | Accepted | M2, M3 |
| 11 | Counts and call sites | Accepted (one count differs by one) | §5, §6 |
| 12 | Translation unchanged | Accepted (the draft's bullet dropped) | §4 |
| 13 | Do not pre-empt decision 4 | Accepted | §3 |
| 14 | jit, sharing, the engine's role | Accepted | §1, §4 |
| 15 | Slicing | Accepted, M4's oracle sharpened | §7 |

## Notes on the partial acceptances

**1, the refinement: the probe mode count stays exact.** The rule "each axis 1 or `y`'s size" does not apply to the
probe's mode axis. That axis is not an axis of any array of `y`; it is the tuple index. Reading a `(1,)+F` weight per
array with `jets._arrays` would yield one array, and a `zip` against `d` modes would silently drop modes. So rule 1 makes
the count exactly `d`. This also keeps the one point of 6.7 that the maintainer confirmed explicitly ("a mode dimension
of 1 is not broadcast", `TARGET_DECISIONS.md:2317`). Everything else in the reviewer's rule is adopted as written: block
or full form by rank; the probe full form always with the edge; stacks inserted before the edge or appended.

Ruled against the rule, the 19-case rejection table (`test_jets.py:849-870`): 4 cases become legal, all constant
weights; 15 still raise (§2).

**2b: arrays, both required, recommended over the batch form.**
- The scale is in `W` (O-21). M4's test runs on a drawn batch.
- The batch form `(kind, order, batch, r)` cannot live in `jets`: `jets` cannot import `LeastSquaresBatch` without
  creating the cycle of 2c. It would have to duck-type the record.
- `weight` and `minibatch_scale` are therefore both required (`None` written out), as `jet_to_dual_jet` already
  requires its weight (`jets.py:339`). Forgetting the scale is then a visible `None`, not a default.
- The batch form is offered as option (b) of Q3.

**2c: one backend home.** `W` is defined in `backend/jets.py`, which imports only `contractions` and `common`. The
engine's chain to it is `sampling_variables.py:320` → `sampling_steps.py:59` → `jets`, so there is no cycle. It is
re-exported by `t3toolbox.fitting`, as `jet_to_dual_jet` is. A second backend name in `backend/fitting.py` would add
an import path and no capability, so it is not added.

**3: `covector` is wrong under the maintainer's typing.**
- `J: X -> Y'` puts the measurements, and `r`, in `Y'`: those are the covectors.
- The maintainer's principle calls `W: Y' -> Y` a "co-vector to vector" conversion (LOG, the pause after P16).
- The transposes' argument is in `Y`, the vector side, so `covector` would mislabel it.
- Q2 recommends `c` and gives this reason. `dual` is rejected for the reviewer's reason (it collides with "dual jet").

**8: the recommendation keeps O-21 intact.**
- Option (b) is built from **new** weighted, unscaled block sums.
- `r_block_sumsq` and `y_block_sumsq` stay plain, as O-21 decided, and the `unwt` header figure is unchanged.
- One caveat added: a block whose weight is zero throughout gives `0/0`, and shows `—`.

**9: which decisions are reversed.** Decision 2 supersedes 2.5 and 6.7's "exact shape only". The engine's stack check
(`TARGET_DECISIONS.md:758-761`) is not reversed: the weight was already outside it (`sampling_variables.py:1097`,
verified), and the plan says so.

**11: one count.** The backend (21), frontend (21) and test keyword calls (123, by file as the reviewer listed) are
confirmed by an AST scan and a grep. Module doctests: 17 `residual=` lines by `grep` over `>>>`/`...` lines in
`t3toolbox/` (the reviewer has 18). The difference does not change the plan.

**15: M4's oracle.** "A dense gradient of `½<r, W r>`" is made concrete. On a drawn batch, `<J^T(W r), ẋ>` is compared
with a central difference of the dense misfit `½ s Σ ω² r²` (`oracle.py`) along a random core direction. That oracle is
independent of the engine. M2 also gains one more independent check: a sample with `ω = 0` gives the `f` and `g` of
the batch with that sample deleted.

## Checked and confirmed as stated

- 1: the engine appends `(1,)*` after `F` (`sampling_steps.py:970, 1053`; `jets.py:290`); `_modes` does
  `asarray(weight)[i]` (`:920`).
- 2c: the imports (`fitting.py:37`, `sampling_variables.py:320`, `sampling_steps.py:59`).
- 2d: the sites.
- 4: the two examples (`fit_hilbert_from_apply_derivatives.py:142-144`, `_flat.py:137-139`); their code is correct
  (`J_d^T = J^T D S`).
- 5: `_taylor_weight` is right-aligned, and `derivative_weight_to_taylor_weight` takes no `kind`. `jets.py` is not on
  `main` or `v2026.5.0`, so the signature change is free.
- 6: `pack_batch` passes the weight through today; `_UNIFORM_MASKS` has no `'weight'` (`sampling_variables.py:1359-1367`).
- 7: `take` passes the weight through (`fitting.py:1021`).
- 8: the display text.
- 10: `sampling_definitions._weight_squared` (`:655-667`); `oracle.weighted_residual` on `omega[mode, order]`; the
  independence test forbids `jets`.
- 12: the translation emits `(d,)`, `(d, n+1)`, `(n+1,)` (`optimizers.py:173-192`).

---

# Round 3: response to `DATA_METRIC_PLAN_REVIEW_r3.md` (writer, 2026-11-22)

All seven findings are accepted. Every code claim was checked at `c3f37d34`, and the two scaling-law claims were
checked by derivation. §9 is rewritten as 9.0 to 9.3. §§1, 3, 7 and 8 follow it.

| # | Finding | Verdict | Where |
| --- | --- | --- | --- |
| 1 | 9.2: the turned form stays public in the backend | Accepted. O-20 named as superseded; blast radius given | 9.2, §3 (the frontend zero check row and the converters' paragraph), out-of-scope line, M1, Q9 |
| 2 | No law catches the residual/sample swap | Accepted. What each law catches; both are homogeneity tests | 9.3 |
| 3 | The parameter-side input law | Accepted, with its caveats, entries' law, and the rescalings as law instances | 9.3 |
| 4 | One dictionary; missing rows; `J_in`; pairings; the sign | Accepted | 9.0, 9.1 table, §1 (one line), 9.3 |
| 5 | 9.1's scope and placement | Accepted | 9.1, M1 |
| 6 | Slicing | Accepted. **Decided: a comments-only M5** | §7 |
| 7 | Q9 to Q11 | Accepted | §8 |

**What was verified.**
- **1.** `HeldVariables` holds `w_tilde` and six tilde variables, each commented "turned"
  (`sampling_variables.py:899-905`). The module text says the tilde variables are stored turned (`:222`).
  `sampling_definitions` takes a turned `w_tilde=` (`:86-89`). Tests use `w_tilde` in 8 files. O-20 is
  `TARGET_DECISIONS.md:358`.
- **1, the blast radius.** Counted as lines naming either function: `jets.py` 26, `fitting.py` 18, `sampling_steps.py`
  2, `test_jets` 71, `test_dispatch` 16, `test_jax_absent` 7, `examples/` and `docs/` 0. The plan gives these line
  counts, not the reviewer's reference counts; they agree in substance.
- **2, the swap.** Write `F(w, c) = T(c, w)`. Then `F(λ^k w, λ^-k c) = T(μ^k c, μ^-k w)` with `μ = 1/λ`, which is
  `T(c, w)` by `T`'s own invariance. Under the input law both arguments carry `· Δ_i`, so the swapped call transforms
  as the original. The swap is an exact symmetry of both laws, as the reviewer measured.
- **3, entries and duality.** Entries' `z = Σ … B_i[:, n_i] …` gives `z / ∏ Δ_i[n_i]` under `B_i -> B_i Δ_i^-1`. By
  duality its `c` (and `ω`) carry `· ∏ Δ_i[n_i]`; the plan states this.
- **3, the rescalings.** `rescale_coordinates` (inputs times the scales, probe data divided by them) is the input law.
  `rescale_direction` (slot `k` times `length^k`) is the jet law.
- **4.** The input jets are read on `min(J_in_i, F)` (`sampling_variables.py:212, 228`).
- **5.** The grammar's "`len` first" is at `signature_style.md:74-77`. `SAMPLE_VARIABLES.md` is under `dev/`.

**Notes.**
- **4, the dictionary.** One symbol, `'`, is the plan's recommendation (Q10). The dictionary states `Y' = P` and
  `Y = P'` for apply. In finite dimensions `P'' = P`, so §1's prime (measurements as covectors of an abstract `Y`)
  and the concrete name `P` are consistent, not inverted. The ring structure of `P` is confined to two named places:
  `Hom_P` in the probe type, and the engine's turned form.
- **4, the entries row.** It is typed `index^d -> R`, with "an index is a basis vector, not an element of `V_i`". That
  is why its input law differs.
- **6, the decision.** Comments only in M5. M1 writes the grammar and its own new signatures in it, and the law helper
  with the full tables.
  - The notation is reviewed once, as a whole, and a later change touches one slice.
  - M4 stays a rename.
  - The ~50 untouched forwards and helpers would need a slice anyway.
  - The cost is that M5 re-edits the comments on M4's 42 signature lines.
  - The laws run in M1 (`W` under the laws, the converters, the rescalings), M2 (the engine), M4 (the 42 transposes)
    and M5 (the rest).
- **7.**
  - Q9 recommends (a), frontend-private and backend-public.
  - Q10 recommends `'`.
  - Q11 recommends backend and frontend.

---

# Round 5 (notation): the maintainer's notation applied (writer, 2026-11-22)

Source: `TARGET_DECISIONS.md`, "Notation for the jets and the data metric". I applied it occurrence by occurrence,
never by blind replacement. Afterwards every remaining `` `W` `` in the plan was checked: each is the sample stack.

**The data metric `W` -> `M`.**
- Every place that meant the metric is now `M`: §1's table and equations, the scale paragraph, §3's title, its text and
  its site table (`M^{-1} ∘ flip`), the engine paragraph, §4, §5, the M1 and M4 rows, Q1, Q3, 9.0, 9.1, 9.2 and 9.3's
  tables and caveats.
- The out-of-scope "operator `W` (a covariance)" now reads "non-diagonal `M`".
- Every `W` that means the sample stack is unchanged: `F+W`, `W = (7,)`, `len(W)`, the `W` axes in `take`, and so on.

**The jet group `J` -> `P`.**
- The plan used no `J` as the jet group. `J`, `Jp` and `JtJp` are the Jacobian throughout and stay. `J_in_i` stays the
  input jet sizes, a size name like `F`.
- The rename goes in M1's scope, under "M1's notation work" in §7. It is unreleased: neither `jet_contract` nor
  `GroupLengths` is in `v2026.5.0`.
- The counts, by grep at `ccf10b0a`:
  - about 216 grouped strings with `J`, in 173 `jet_contract(` call lines;
  - 32 `GroupLengths(` constructions and 14 `lens.J` reads;
  - 5 `'J'` literals in the parser (`contractions.py:578-629`);
  - none in `docs/` or `examples/`.
- Checked: an unknown group letter already raises in `jet_contract`. So a string left with `J` after the rename fails
  loudly, and M1 tests that it does.

**9.0.** It becomes "The notation, and the dictionary with §1", with the table the coordinator listed:
- `P`, as both the ring and its group, and `P_in,i`;
- `Q`, used from Part 2;
- `P'`, `F` (shape comments write `F`, strings the group `P`), `J_in_i`;
- `V_i` / `V_i'`, `E` / `E'`, `X` / `X'`, `Y'` / `Y`;
- `M`, with the accepted collision with the manifold `M`;
- `R_X`, the Jacobian `J`, the stacks `W`, `K`, `C`, and `[G]`.

The turned form is tied to `Q` in one line.

**The glossary and `naming_conventions.md`** entries are in M1, not P19: they are the durable record of a decision.
- The glossary's Symbols table gains `P`, `Q`, `F`, `M`, `Y` / `Y'` and `[G]`.
- "Index letters" gains `P`, `Q` and `[G]`.

**Do the names still read right?** One does not: `data_riesz_map`.
- A Riesz map is an isomorphism. `M` is not one where a weight is zero (audit D5), and not as a future loss Hessian.
  The decided name of `M` is "the data metric".
- Q1 now recommends **`apply_data_metric`**:
  - it names the operator as the maintainer named it;
  - it stays right in both cases above;
  - it is verb-object, like `prepare_derivative_data` and `rescale_direction`.
- Its stated blemish: "apply" is also a sampling kind, so `apply_data_metric('apply', …)` says it twice.
- `data_riesz_map` is option (b), with the reason it fell; `data_metric` and `weigh` are options (c) and (d).
- The plan's body uses the recommended name.

The other names still read right:
- `c` (Q2);
- `jet_to_dual_jet` / `dual_jet_to_jet`, whose names stay with decision 4;
- `check_weight`;
- `minibatch_scale`.

---

# Round 6: response to `DATA_METRIC_PLAN_REVIEW_r5.md` (writer, 2026-11-22)

All five findings are accepted. I re-measured the counts with my own AST scan at `ffa4ad58`.

| # | Finding | Verdict | Where |
| --- | --- | --- | --- |
| 1 | The rename's full scope | Accepted, re-counted | §7, "M0, the rename, in full" |
| 2 | M0 before M1 | Accepted | §7 table (new M0 row; M1 is back to "the rule and `M`"), "In brief", 9.0, §9's preamble |
| 3 | Letter collisions; complete 9.0 | Accepted | 9.0 table, §1, §2, §3, §4, 9.1 |
| 4 | Q1's trade-off | Accepted; the **recommendation changes** to `data_metric` | Q1, the body |
| 5 | Minor | Accepted | the scope line; Q1 |

**1, the counts.** My scan (code only; docstrings and doctests change too):

| Item | Count | Breakdown |
| --- | --- | --- |
| grouped string literals with `J` | 202 | `sampling_steps.py` 93, `test_jet_contract.py` 55, `_jet_contract_cases.py` 38, the rest 16 |
| `jet_contract` calls | 153 | |
| `GroupLengths(J=…)` | 25 | |
| `lens.J` | 9 | |
| `len_J` | 33 lines in 8 files | |
| `'J'` literals in tests | 22 | `_jet_contract_cases` 9, `test_jet_contract` 8, `test_contractions_interpreter` 4, `test_contractions_sharding` 1 |
| body-local names with a `J` group suffix | 377 uses of 21 names | `sampling_steps.py` 358 |

- The local-name count differs from the reviewer's 253. My pattern counts every `*_<groups with J><axes>` name, including the `_tilde` and `_dot` forms; the reviewer's 253 is the sum of the names they listed. The plan gives 377, with the main names named.
- `len_J` is a keyword of two public backend functions, `ut3_operations.pack_vectors` and `pack_if_ragged`. It is absent from `v2026.5.0`, so `len_P` is a free rename.
- An unknown group letter already raises in `jet_contract`, so M0's "old `J` raises" test sits beside the retired `T` in `TestGroupLengths` (`test_jet_contract.py:827`).
- No doc page, example or CLAUDE.md names a `J` group (grep).

**3, the letters.**
- **The minibatch scale is `γ`.** `γ` is free in the glossary's Symbols table and in the code, apart from a local in one `.tex` example; `β` is CG's, and `Φ` is used about a hundred times.
  - It replaces `s` in §1, §3's signature and text, §4, the M4 row, 9.0 and 9.1.
  - On a draw, `γ = |W|/b` (the glossary's `|W|`), which also retires the plan's second `n`.
- **`ℝ`** replaces the reals `R` in 9.0, 9.1 and §1's concrete line. `R` is the right store only.
- **The factorial diagonal `D` is gone.** §1 writes slot by slot, `r_d[k] = k! r_t[k]` and `c_t[k] = k! c_d[k]`, with `k! = k_1!…k_m!` listed in 9.0 as a factor with no letter.
- **Two more `D`s, found in passing:**
  - The rejection table's `D = 3` (the test file's constant) is now `d = 3`.
  - §1's derivative `Dz(x)` and `D(J^T c)` now use the glossary's upright `d`, which since 2026.5.0 is the derivative symbol and no longer `D`.
- **9.0 gains** `ω`, `γ`, `m`, `n_j`, `k`, `k!`, `ℝ`, `N_i`, `e`, `λ_j` and `Δ_i`, and a line saying the glossary's `P` row is extended, not duplicated.
- **M0's record:**
  - The glossary's `F` mentions note both meanings (the box sizes and the frame `P_F`, `T(F)`).
  - The preamble's reserved letters gain `P` and `Q`.
  - Whether `J_in_i` becomes `F_in,i` is left to M0's design note.

**4, Q1.** Q1 now states the trade-off as a table.
- "Data metric" is the maintainer's chosen name. It is not more exact than "Riesz map": read strictly, "metric" presumes positive-definiteness too.
- `apply_data_metric` parses, under the library's leading-kind-word grammar, as "the apply kind's data metric".
- The recommendation moves to **(c) `data_metric`**:
  - it uses the maintainer's name with no kind word in front;
  - it reads as the operator applied, `M(r)`;
  - its cold-reading risk ("build `M`") is answered by the required `r` and the docstring's first line;
  - a later typed metric object would be a class, `DataMetric`, told apart by case.
- (a) `apply_data_metric`, (b) `data_riesz_map` and (d) `weigh` stay as options, each with its cost. The body uses `data_metric`.

**5.** The `W` quotation is gone with the Q1 rewrite: (b) now reads "your earlier 'Riesz map'". "an non-diagonal" now reads "a non-diagonal".

**Not changed:** the slice names M0 to M5 beside the symbol `M` (the reviewer's optional point). The one phrase that read oddly, "M1 for `M`", now says "M1 for the data metric".

**Round 6 addendum** (the coordinator's three points from `DATA_METRIC_PLAN_REVIEW_r6.md`; no further review round):
- **The jet-rescaling letter is now `ℓ_j`**, replacing `λ_j`, which collides with the regularizer's `λ`.
  - `ℓ` is free in the code, the glossary and the pages, apart from a local subscript in one `.tex` note. It matches
    `rescale_direction`'s `length`.
  - 9.3's input-law cell now says "a multiple of the identity" in place of `λ_i I`.
  - 9.0's claim that "every letter was checked" is gone. It now says that the two new letters, `γ` and `ℓ`, were
    checked, and names `M` as the one accepted collision.
- **Q1 (c)'s "Against" cell** now says that the library's noun functions return what they name (`data_block_norms`,
  `equalizing_weight`), so a `data_metric` that returns `M r` is an exception. The "like `project`" comparison is
  dropped, since `project` is a verb. The recommendation is kept, with the reason stated: every option breaks one
  pattern, and this break is the mildest.
- **Q12 is added: `J_in_i`.**
  - (a) Rename it to `F_in,i` (recommended): 180 lines, all unreleased, mostly comments, plus 15 uses of a local
    `J_in`.
  - (b) Keep it.
  - (c) Another name.
- M0's bit-for-bit evidence is left to M0's design note, as instructed.

---

# Round 7 (decisions): every question decided, folded in (writer, 2026-11-22)

Sources: `TARGET_DECISIONS.md`, the notes of 2026-11-22 from "The data-metric plan's questions: the maintainer's first
answers" to "The data-metric plan: Q3", and `T3POLYNOMIAL_JET_USE.md`. I rewrote the plan around them, and §8 is now
a record of the decisions.

- **Q10, `*` for duals.** It replaces `'` throughout: `X*`, `Y*`, `P*`, `V_i*`, `E*`, `Y** = Y`, `P** = P`, and the
  maps `J: X -> Y*`, `J^T: Y -> X*`, `M: Y* -> Y`. Every `*` is inside backticks. 9.0 states that `*` means
  multiplication only in Python code (`γ * ω**2 * r`) and is a wildcard in names (`tv_*`). A grep finds no `X'`,
  `Y'`, `P'`, `E'` or `V_i'` left.
- **Q5 and Q8, the rule.** §2 is rewritten. Every data-space object has the type of `z`, each axis 1 or `z`'s size,
  never `K`; the probe mode count is exact. Consequences:
  - There is no block form. The rank-reading rule and its trap are gone.
  - `y` and `ω` carry `C`: size 1 for shared data, full for per-element data. `r_step`'s alignment of a `y` without
    `C` goes.
  - The checkers gain `stack_shape=()`.
  - The rejection table is re-ruled. Two of the 19 cases are legal (a size-1 `W` axis); 17 raise; the old accepted
    block forms raise.
  - Everything that depended on the block form now uses the full form: `take` and packing (one rule for `y` and `ω`),
    the factorials (leading jet axes), the display (block shape is only the table's shape), the translation, the
    bridge and two page doctests (through the constructor), and `f_resolution` (per element of `C`, a consequence of
    per-element `y`).
- **The constructor.** I propose `blocks_to_weight(kind, order, blocks, y)`: per-block values on the jet axes (and
  modes), size 1 elsewhere, shaped like `y`.
  - The name follows the converter grammar. The alternatives are `weight_from_blocks` and `block_weight`. §8 lists
    it as the one name to confirm, in M1's design note.
  - `data_block_norms` and `equalizing_weight` stay block-shaped, as per-block quantities.
- **Q7.** §1 and §4 give it: Taylor coordinates everywhere, one conversion at the boundary, every stored copy in
  Taylor coordinates, and the recipe (convert the data, then compute the weight). From the study, I added:
  - `prepare_derivative_data(weight=None)` keeps returning `k!` (now in full form), documented as rarely right for
    decaying jets;
  - `equalizing_weight`'s floor depends on the coordinates (Taylor widens the spread by up to `n!`), covered by
    "rescale the direction first";
  - the warning that the real risk is primal mixing of stored derivative copies.
- **Q3.** §3's signature is `data_metric(r, weight, minibatch_scale)`, plus `LeastSquaresBatch.data_metric(r)`.
  - Elementwise, with no `kind` or `order`. A tuple or list is read element by element, and a packed array
    elementwise.
  - Equal rank per pair, so a `K`-stacked `r` raises. With `weight=None` the result is `γ r`.
  - The single-implementation table is rebuilt. `_omega_squared` is the one place `ω²` is formed. The engine inserts
    `K` through a private `_with_K` (its group lengths), and its steps call `data_metric` with `γ = None`.
    `r_step` and the new display sums are new rows.
  - `jet_to_dual_jet` / `dual_jet_to_jet` keep their signatures but now reject a `K`-stacked jet, since the engine's
    `K` path is private.
- **Q1, Q2, Q4, Q6, Q9, Q11 and Q12** are recorded in §8 and applied. The plan's body uses `data_metric` and `c`.
  9.2 is marked decided. `J_in_i -> F_in,i` joins M0's table: 180 lines and 15 uses of a local `J_in`.
- **The slices** are re-cut, because the representation must switch atomically. A full-form weight through an
  unswitched `take` would keep the full batch's samples on a drawn batch, a silent wrong answer.
  - **M1** is additive: `data_metric`, `_with_K`, `blocks_to_weight`, the batch method, 9.2, the law helper and the
    grammar.
  - **M2** is the switch: checkers, engine, reference, converters, factorials, `prepare_derivative_data`, `take`,
    packing, `f_resolution`, stacked `y`, the translation, the bridge, the pages and the tests. It is tested by, among
    others, "a zero-weight sample = the batch without it" and "per-element stacked data = separate fits". It may be
    built in parts, as P16 was.
  - **M3** is the display, now separate.
  - **M0, M4 and M5** are as before.
- **The blast radius.** §6 adds counts at `b5a848d5`:
  - about 170 `weight=` lines in tests, some of them on old kinds;
  - 39 `LeastSquaresBatch(` calls in tests;
  - 12 `check_weight`, 7 `check_data`, 20 `prepare_derivative_data`, 7 `derivative_weight_to_taylor_weight` and 73
    converter lines in tests;
  - the stacked-`y` tests.

**Round 7 addendum** (the coordinator's six points from `DATA_METRIC_PLAN_REVIEW_r7.md`, which signed off; no further
round):
- **`blocks_to_weight` is §8's one open question:** (a) `blocks_to_weight(kind, order, blocks, y)`, recommended,
  (b) `weight_from_blocks`, (c) `block_weight`. "In brief" and the status line point to it.
- **The converters on `K`.** They raise on a `K`-stacked jet only when a weight is given; with `weight=None` they
  flip it as today. M1's test line now says the same.
- **Three details are listed in §8 as writer proposals,** to be put to the maintainer in the design note that builds
  each: `stack_shape=()`, `prepare_derivative_data(weight=None)` returning `k!`, and M3's names.
- **`data_metric`'s two edge cases** (a packed `r` with a tuple weight; a packed weight with a size-1 mode axis) are
  noted in §3 for M1.
- **M2's gate** gains the pre-M2 numbers on every weighted test through `blocks_to_weight`, and the reference's
  full-form reading pinned against brute-force dense sums first.
- The status line and §6's "calls" are fixed.
