# R5 handoff: the docs, the six notes, the glossary and work list, CHANGELOG and release notes (store rename, 2026.5.0)

Implementer's record for the reviewer and the session. Base: HEAD `f4bb6357` (R4b `f5ef5a8c` + the R4 review stop).
Nothing is committed. Scope (plan §7 row R5, §6, §4 items 1–2 and 7–9, §4A for the docs' and notes' blocks): every user
and contributor page, `CLAUDE.md`, the six `.tex` notes and the four committed PDFs, `docs/glossary.md`,
`dev/glossary_naming_decisions.md`, `CHANGELOG.md`, `docs/release_notes.md`, the untracked root
`release_notes_2026.5.0.md`. No file under `t3toolbox/`, `tests/` or `examples/` changed (K15: 0 NAME hunks against
HEAD; the battery is bitwise).

**How the work ran.** The session's implementer did the pages, `CLAUDE.md`, the CHANGELOG / release notes, the tools and
the gates. Five sub-agents (Opus, xhigh, briefed with the plan's sections, the sample edits and the inventory) converted,
by hand, (1) `exact_hessian_two_hole.tex`, (2) `symmetric_probe_derivatives.tex` + `ttm_t3m_ht_note.tex`, (3)
`shared_t3_math.tex`, (4) `padded_frame_cone.tex` + `pad_safe_svd.tex`, (5) the glossary and the work list. Every note
edit was made with the Edit tool, block by block; no script touched a page or a note. Their records (block dispositions,
contraction-size checks, judgement calls, questions) are condensed in §2 and Appendix B; the glossary agent's full record
is Appendix C. The inventory's new spans and new texts were then filled by a script that carries each block's old span
through a line alignment and copies the tree verbatim (K12 checks the copy); every unchanged block carries a reason.

**Result.** Every K-check is clean at stage R5 (§9): K0–K9 (+ the new docs half of K9), K10/K10b (37 in place, 23
re-laid out), K11, **K12 301 of 301 blocks filled**, K13, K14, K15 (0 hunks), K17; K16 lists 21 LaTeX width forms (20, plus
`pad_safe_svd`'s `n_{D,j}` from the Q5 ruling), each signed off by its contraction-size check (§2.2, Appendix B). Battery **bitwise**. Markdown doctests and
`getting_started.rst` pass in both environments. `sphinx -W`: §9. All four committed PDFs rebuilt; the two uncommitted
notes compiled to scratch for Nick. **Q1 (§7) is resolved by the session's ruling** (relayed by the reviewer): the
exact-Hessian note's frame centers become `\bar H_i` (12 tokens), so they no longer share `H_i` with the gauged
TT coordinate.

## 1. What changed, by file

**User pages.**

- `frame_variations.md`: the tangent diagram (DOC-FV-0014, in place, `U0..U2` → `D0..D2`); the V-term sentence; the
  symbol table (DOC-FV-0023: down-Tucker `Dᵢ`/`D`/`nD`, up-TT `Uᵢ`/`U`/`nU`, `L`, `R` without T4S's `P`/`Q`, `V`'s
  width `nU`, `H`'s `nD`); `D0…D(d-1)`; the base-point slot `L_last` (the code's name since R4a); the gauge `Dᵢ Vᵢᵀ = 0`
  (DOC-FV-0057, code layout, work-list E10 done); **§ "Four rank stores"**: the `down_ranks`/`up_ranks` sentence, then
  sample batch 3 edit 1's **Store names** paragraph and picture and **Letters** paragraph, plus one sentence that the
  words and `nU`/`nD` swapped in 2026.5.0 (with a link to the upgrade notes), and the **T4S notation** paragraph with
  Nick's line verbatim; the caps table (DOC-FV-0080: `Dᵢ` (down / Tucker), `Uᵢ` (up)); `down == up`; "the operative (up)
  ranks"; the weights' legs `down`↔`H.nD`, `up`↔`V.nU`.
- `user_guide.rst`: `:97` "TT core up-matricizations are full rank" (inv 3; work list :300–301); the tangent diagram
  (DOC-UG-0125, in place on raw text, TABs kept: `U`→`D` on the Tucker rows, `O`→`U` in the chain); the frame-core list
  (DOC-UG-0139, re-laid out: reordered to `(down, up, left, right)`, `down-orthogonal` / `up-orthogonal`, §4.3's backwards
  adjective fixed); the batch-shape listing (DOC-UG-0215: `D_i`, the variation `V_i`).
- `aligned_padded_frame.md`: the store definition (new order, named for the side its cores occupy); the colours
  `('D', 'L')` / `('U', 'R')` (L7, ×4); the jit paragraph's four step programs.
- `weighting.md`: the weight-family table (DOC-W-0069: `down` on `H`'s `nD` leg, `up` on `V`'s `nU` leg, rows in
  physical order); "(`up`→`V`, `down`/`left`/`right`→`H`)"; `down = up =` ×2 (one a doctest comment; output unchanged).
- `batching_and_stacking.md`: the shape table (DOC-BAS-0040: `Dᵢ`, `Lᵢ`, `Vᵢ`); the disjointness list
  `(D,U,L,R,B,G,V,H)`; `contract(..., mu, L, xi)` (the code's local since R2); `[D_i ; V_i]`; §8's paper ↔ code map
  rewritten with the code side in the new letters (`U` = `down_tucker_cores` (our `D`), …, "The paper's `U` is our `D`");
  `T3Frame.data = (down, up, left, right) = (D, U, L, R)`.
- `entries_apply_probe.md`: a T3's own Tucker cores `Bᵢ` at `:65, 85, 120–121` (§3.3, inv 3 C13); §8 keeps the paper's
  letters and gains the T4S line after its notation line.
- `numerical_contracts.md`: "(D/U/L/R conditions)"; the gauge grams `D Vᵀ`, `Lᵀ H`; the doctest variable `D` → `diff`
  (inv 3 §5.8 / C8, output unchanged); `down`/`up` in order.
- `chunking.md`: the Tucker gradient `(order+1)·nU·N` (the Tucker variation's rows; inv 3 §5.9); the `shard_map` code
  block's transpose outputs `dU, dG` → `V, H` (shape rule, as `manifold.py:1134` writes them; comment column kept).
- `fitting_and_optimization.md`: the corewise raw cores `(B,G,G,G)`. `transposes.md`: "the Section 6.3 substitution
  `(U, L, R) → G`" (tier D's hand rule), "left/right/up frames". `t3m_methods.md`: `B_i`, `B_i^x ⊙ B_i^y` (§3.3).
  `sharing.md`: the gauged tied ambient direction `Ḋ` (session ruling (b)), the corewise "`Ḃ_i` are equal", the tied
  lift's raw direction `Ḃ` (as `fv_tied_lift_factors`' `Bdot_g`), "the `up` family lives on the `nU` legs".
  `naming_conventions.md`: §12.14's bullet verbatim, with a link to Four rank stores.

**Contributor pages.**

- `padded_frame_internals.md`: the edge-store paragraph (a store's edges point away from its name) and table (`down[i]`
  on `l_i → c_i`, `up[i]` on `c_i → l_i`, "the middle index (`nU`)", as the note's Table 1); sample edit 9's AFTER
  (`F′`, the store-and-arrow order); `Ψ = [Ā | P⁺]ᵀ g_new` (D-5); the sharing edge (`l_i → c_i`, `_down_edge`); the
  colours and jit step programs in letter tuples (L7); the seed tags `down 0, up 1, left 2, right 3` (the 2026.4.0
  integers of the same physical stores); the uniform geometry's "down and left masks" (`geometry.py:815`).
- `weighted_internals.md`: `up=None`; the abandoned design `down→D, up→U, left→L, right→R`; `up`→V, `down`/`left`/`right`→H.
- `curvature_internals.md`: `(D, U, L, R)`; `⟨U_i, H_i⟩`; the lift `Ḃ_i`, base point `θ = (D, L)` (also in the doctest:
  `D, _, L, _ = frame`, output unchanged); `compute_xi(Ḃ, …)`; down/left masks ×2; the zero-point factor `D_g`; "only for
  `D`"; the tied lift `Ḃ_g`; `dΦ` (D-11).
- `sharing_internals.md`: `Ḋ` (×5, the gauged tied direction), `U2_i` (the up core's matricization), `U_i`/`R_i`, the
  corewise `{Ḃ_i all equal}`, `[D_g; V_i]`, `[D_g; Ḋ]`, "the up core `U_i`", "new Tucker slots", "**down** width `nD`
  (the `Ḋ` slot)" (as `utv_to_ut3`'s docstring).
- `numerical_contract_catalog.md` (D/U/L/R; `down`/`up`), `parameters_not_closures.md` (`(down, up, …)`, `(B,G,G,G)`, "the
  up rank", `B → BQ`), `precompute_and_caching.md` (`⟨U, H⟩`), `batching_internals.md` (the letter list),
  `deferred_and_rejected.md` (`sh^r x`, `sh x`; `mu U`; `D_k^⊥`/`D_k`), `fitting_internals.md` (`L_last` ×3),
  `testing_strategy.md` (`[R | L]` ×2, the doubled bond as §1.5 item 3).

**`CLAUDE.md`** (sample batch 3 edit 7, plan §6): the notation map carries Nick's T4S line and the code names, then the
one clause "Widths: `nD` the down store's (the Tucker factor's rows), `nU` the up store's — they swapped meaning in
2026.5.0; see `docs/frame_variations.md` § Four rank stores"; `(U, L, R) → G` (`:115`); `down/up/left/right` (`:158`); the
history list `D`/`U`/`L`/`R`/`B`/`G`/`V`/`H` (`:182`); the gotcha `T3Frame.data = (down, up, left, right) = (D, U, L, R)`
(the `U`/up core is `up_tt_cores`) without T4S's tuple; the letter list `(D U L R B G V H …)`.

**The notes** (§2): all six converted; four PDFs rebuilt (`exact_hessian_two_hole`, `shared_t3_math`,
`padded_frame_cone`, `pad_safe_svd`); `symmetric_probe_derivatives.pdf` and `ttm_t3m_ht_note.pdf` compiled to
`<scratch>/r5/SPD_TTHN/pdf/` for Nick (tag builds beside them in `oldpdf/`), not committed.

**Glossary and work list** (§4, Appendix C). **CHANGELOG `[2026.5.0]`, `docs/release_notes.md` "Upgrading from
2026.4.0", root `release_notes_2026.5.0.md`** (§5, Appendix D).

**Dev files.** `dev/store_rename_check.py` (§8); `dev/store_rename_keeps.txt` (+32 KEEP rows: 26 inv 3 §9 English idioms
in the docs, 6 glossary adjective/travel rows); `dev/store_rename_slices/k9_registered.tsv` (+32 clause-(c) rows (two for `CLAUDE.md:70–71` after the session's M1 rewrap), the
R5 rewrites); `dev/store_rename_slices/diagrams_inventory.md` (the 247 docs/notes entries filled; three `re-laid out`
fields set: DOC-FV-0057, DOC-UG-0139, DOC-UG-0215; the reasons of the 89 unchanged blocks).

## 2. The notes

### 2.1 Per note

**`exact_hessian_two_hole.tex`** (71 blocks; 64 changed, 7 unchanged with reasons; 1066 → 1071 lines). θ's raw Tucker
cores `U_i` → `B_i` (format paragraph, tikz fig:holes, eq:t3t3's `BU_i` → `\hat B_i`, `\dot{BU}_i` → `\dot{\hat B}_i`,
fig:zipper, alg:t3t3); the frame `(U_i, O_i, P_i, Q_i)` → `(D_i, U_i, L_i, R_i)` = `T3Frame.data` = (down, up, left,
right), `n^D_i` → `n^U_i`; the T4S paragraph after the frame definition, with "T4S, and this note before 2026.5.0, write a
T3's own Tucker cores $U_i$, our $B_i$"; the base point `θ = (D, L)`; the gauged coordinates `v = (V_i, H_i)`,
`V_i ∈ R^{n^U_i × N_i}`; eq:E, the chains `L_{<i}`, `R_{>i}`, the gauge `V_i D_i^T = 0`, `(L_i^{⟨2⟩})^T H_i^{⟨2⟩} = 0`;
`\mathrm d\Phi`, `\mathrm d^2\Phi`, `\mathrm d_\zeta P` (every one); `\mathcal R_X`, `\mathcal R_x` (the three listed sites) and
the table's C8 `\mathcal R(tp)` (J7); PB20's `R_k^{-1}` kept, ", PB20's notation" added; eq:factors (`T_i =
⟨L_{>i}, R_{>i}⟩`, the centers `\bar H_i = L_i ×_3 T_i` (Q1's ruling), `S_i^T = ⟨U_i, \bar H_i⟩ ∈ R^{n^U_i × n_i}`); the lift `\dot B_i = (S_i^T)^+ V_i`, `\dot G_i = H_i ×_3
T_i^+` (eq:lift, fig:lift relabelled whole: `L_{i−1}`, `R_{i+1}`, `R_{i+2}`, `U_i`, red `H_i`, `V_i`, `\dot B_i`,
`L_{>i}`), alg:lift; the one-hole outputs `\widetilde V_i`, `\widetilde H_i` (11 sites); `(B, G, G, G)`, "corewise
substitution `U, L, R → G`", `U = L = R = G`, the base point's corewise frame `(D, L, L, L)`; the shift macro
`\newcommand{\Sh}{\mathsf{sh}}` with `(\Sh x)^{(t)} = t x^{(t−1)}` and its adjoint `\Sh^\T`, `\Sh^r`, `(\Sh^\T)^r` (eq:band,
eq:conv, the correlations, eq:asmjet — wrapped in `gathered` to clear a 17.3pt overfull — and the fold); the tied lift
`\dot B_g`, `V_i`, `Y^B_i`, `Σ n^U_i` (alg:tied); the zero-point frame `D_g`, `U_i, L_i, R_i`, `L_d = R_1 = 0`; "Haar only
for `D`"; `frame_data[5]`'s "down and left masks". Unchanged: `P_X` (0145), an SVD's `V f(Σ) U^T` (0383), the block-sum
`B^T` (0554, 0558; Q2), Z's `(B_z, G_z)` (0776), `U_M`, `V_M` (0863), the PB20 entry (1053).

**`symmetric_probe_derivatives.tex`** (48 blocks; 26 changed). The T3's own `\mathbf U_i` → `\mathbf B_i` (§1–§4,
Algorithm 1); `D^t` → `\mathrm d^t` (its `\mathbf P^t` kept); the frame tuple `((\mathbf D_i),(\mathbf U_i),(\mathbf
L_i),(\mathbf R_i))` "(down-Tucker, up-, left-, and right-TT cores)"; eq:rbase/rdxi/rsigma/rtau/rdeta/rz, the jets and
eq:rdsigma in `D`, `U`, `L`, `R`, `V`, `H`; the transposes `\widetilde{\mathbf V}_i`, `\widetilde{\mathbf H}_i`
(SPD-0540, SPD-0631 included); `\partial_{\mathbf H_i}`, `\partial_{\mathbf V_i}`; the corewise sentence (SPD-0559, now two
lines): `(\mathbf B,\mathbf G,\mathbf G,\mathbf G)` (`\mathbf B\to\mathbf D`, `\mathbf G\to\mathbf U,\mathbf L,\mathbf R`)
and "The manifold transpose requires an orthogonal frame; the corewise one works for any `\mathbf B`, `\mathbf G`" (K4's
phrase gone); :491/:500 rewritten to the new picture ("the edge down to its Tucker core", "the edge up to its TT core");
`δ\mathbf G` → `\mathbf H` on 8 prose lines outside the blocks. T4S paragraph at :402–405 with "Like this note before, it
writes `\mathbf U_i` for a T3's own Tucker factors, our `\mathbf B_i`". Unchanged (22): the direction `\mathbf P`, the
bilinear map `B`, the Lagrangian `\mathcal L`, `(r_L, n, r_R)`.

**`ttm_t3m_ht_note.tex`** (15 blocks, all changed). Leaf frames `U_i` → `B_i`; operand superscripts `A, B, C` → `X, Y, Z`
(26/24/9, matching §4.8's counts), the operand words ("X-half", "Y-core"); the HT transfer tensors `B_t` → `T_t`; the
macros `\A, \B` removed and `\X, \Y, \Z` defined, `\C` kept as the T3's core tensor (eq:t3) only; the T4S paragraph
names what HT papers write (`U_t`, `B_t`) and what the note wrote before (`\mathcal A, \mathcal B, \mathcal C`, `A, B, C`).

**`shared_t3_math.tex`** (42 blocks; 36 changed). The T3's own factors `B_i`, `B_g` in §1–§4 and prop:corewise (`\dot
B_i` the raw directions); the tangent sections (§5 through rem:svdsolve, and prop:retract with rem:embedding as one range)
`D_i`, `D_g`, `\dot D`; `K_i := B_i W_i` (alg:init, :282–303 only; :316's `B_i` is the factor itself and stays); `Z` for
the tied coordinates (`\dot D = O_g Z`, `C_i = Z S_i`, the Gram `tr(Z(Σ Γ_i)Z^T)`; :575 too, inside the plan's range);
`n_{D,i}` → `n_{U,i}` (×4, K16); "Tucker leg" ×2, "new Tucker slots"; the notation sentence "In the tangent-space
statements … $D_i$, $D_g$, $\dot D$ … At a point in orthonormal representation $D_i$ is $B_i$; the corewise geometry …
keeps $B_i$"; the T4S paragraph with "Before T3Toolbox 2026.5.0 this note wrote $U$ for the factors (now $B$, and $D$ in
the tangent statements), $B_i$ for the block $K_i$ …, and $B$ for the tied coordinates $Z$". Unchanged: `P_sf` and its
stages, `Q_i` (a rotation), `R(\T, \xi)`, `O_g`. `K`, `Z` were free before the edit.

**`padded_frame_cone.tex`** (28 blocks; 7 changed). D-5: `F_{x\to t}`, `F'` for the edge frames, `\Psi_{a\to b}` for the
messages (0096, 0152, 0171, the message table 0280, 0292); Table 1 (0102) rewritten whole (`down[i]` `D_i ∈ R^{n_{D,i}
×N_i}` "the Tucker index"; `up[i]` `U_i ∈ R^{r_{L,i}×n_{U,i}×r_{R,i+1}}` "the middle index"; `L_i`, `R_i` middle leg
`n_{D,i}`); Algorithm 1 (0306) in the store-and-arrow words and the code's current function names (`_up_message` first,
`_down_message` last; `_down_edge` first, `_up_edge` last, checked by the functions' docstrings); prose: `(down, up, left,
right)`, the store sentence by Table 1 (J3), the T4S paragraph, the seed tags `down 0, up 1, left 2, right 3`, "a TT or
up seed row", the colours `('D', 'L')` / `('U', 'R')` ×4, the root sentence verbatim from §4 item 2. Unchanged (21): the
completions `P`, `P^+`, the matrix case's SVD `U, V` with `P, Q`, the residual function `R(θ)`, the edge count `L`.

**`pad_safe_svd.tex`**: "the down step" → "the TT up-orthogonalization" ×2 (it is
`ut3_orthogonalization.up_orthogonalize_tt_supercores` that calls `pad_safe_svd`). All 32 blocks unchanged (SVD/QR
factors, `B = Q^T A_pad`, pad counts and `P_real`, the BUG paper's `U_0`), each with its reason.

### 2.2 K16: every LaTeX width form, signed off by contraction sizes

Layouts are each note's own (not converted): the exact-Hessian and padded-frame notes write the Tucker factor `n × N`
with orthonormal rows (the code's `(n, N)`), the shared note `N × n` with orthonormal columns. Code shapes for reference:
`D_i (nD_i, N_i)`, `U_i (rL_i, nU_i, rR_(i+1))`, `L_i (rL_i, nD_i, rL_(i+1))`, `R_i (rR_i, nD_i, rR_(i+1))`, `V_i
(nU_i, N_i)`, `H_i (rL_i, nD_i, rR_(i+1))`. The exact-Hessian note writes the down width as plain `n_i`.

| # | note:line (R5 tree) | form | check (sizes on both sides of every contraction) | with the old label |
|---|---|---|---|---|
| 1 | padded_frame_cone:113 | `D_i ∈ R^{n_{D,i}×N_i}` | input `H = R^{N_i}`: `N_i` = `N_i`; output rows `n_{D,i}` meet `L_i`'s and `R_i`'s middle legs `n_{D,i}` (#3, #4); code `C+(nDi, Ni)` | `n_{U,i}` rows against `L_i`'s `n_{D,i}` leg: 4 vs 2 at mode 0 of the agent's measured `(5,5,5)` / `(4,4,4)` / `(1,2,2,1)` point (`nD = (4,4,4)`, `nU = (2,4,2)`) |
| 2 | padded_frame_cone:114 | `U_i ∈ R^{r_{L,i}×n_{U,i}×r_{R,i+1}}` | `L_{i−1}`'s output `r_{L,i}` = `U_i`'s first axis; `R_{i+1}`'s output `r_{R,i+1}` = its last axis; the middle leg `n_{U,i}` meets `V_i`'s rows (`(nU, N)`); `_down_message`'s `'Caxb,Caub->Cxu'` contracts both bonds → `(kD, nU)` | `n_{D,i}` (4) orthonormal middle directions in a `1×2` bond space; no match with `V_i`'s `nU` rows |
| 3 | padded_frame_cone:115 | `L_i ∈ R^{r_{L,i}×n_{D,i}×r_{L,i+1}}` | `r_{L,i}` from `L_{i−1}`, `n_{D,i}` from `D_i`, output `r_{L,i+1}` into `L_{i+1}`/`U_{i+1}`; code `C+(rLi, nDi, rL(i+1))` | `n_{U,i}` leg against `D_i`'s `n_{D,i}` rows |
| 4 | padded_frame_cone:116 | `R_i ∈ R^{r_{R,i}×n_{D,i}×r_{R,i+1}}` | `n_{D,i}` from `D_i`, `r_{R,i+1}` from `R_{i+1}`, output `r_{R,i}`; code `C+(rRi, nDi, rR(i+1))` | as #3 |
| 5 | exact_hessian:118 | `U_i ∈ R^{r^L_{i−1}×n^U_i×r^R_i}` (and `D_i`, `L_i`, `R_i` with plain `n_i`) | bonds meet their neighbours; eq:E's `U_i ×_i V_i` contracts `n^U_i` with `V_i`'s `n^U_i` rows; `×_j D_j` contracts `L_j`'s `n_j` with `D_j`'s `n_j` | `n^D_i` (now the down width `n_i`) makes `S_i^T = ⟨U_i, H_i⟩` square, contradicting "`S_i^T` wide" (:340) and the code's `(nUi, nDi)` |
| 6, 7 | exact_hessian:119 ×2 | `n^U_i ≤ n_i`; `n^U_i < n_i` is Tucker slack | the up core's mode leg is the rank of the up-matricization, ≤ the Tucker rank | "`n_i ≤ n_i`", vacuous |
| 8 | exact_hessian:125 | `V_i ∈ R^{n^U_i×N_i}` | `U_i ×_i V_i`: `n^U_i` = `n^U_i`; gauge `V_i D_i^T`: `N_i` = `N_i`; lift `(S_i^T)^+ V_i`: `(n_i×n^U_i)(n^U_i×N_i)` → `B_i`'s shape | both contractions pair `n^U_i` with `n_i` |
| 9 | exact_hessian:309 | `S_i^T = ⟨U_i, \bar H_i⟩_{1,3} ∈ R^{n^U_i×n_i}` | `\bar H_i = L_i ×_3 T_i` (`r^L_{i−1}×n_i×r^L_i` times `r^L_i×r^R_i` → `r^L_{i−1}×n_i×r^R_i`); indices 1, 3 contract `r^L_{i−1}`, `r^R_i`, leaving `n^U_i × n_i` (`fv_center_factors`' `(nUi, nDi)`); `\bar H_i = U_i ×_2 S_i^T` contracts `n^U_i` | square |
| 10 | exact_hessian:898 | `max(Σ_{i∈g} n^U_i, n_g)` | `M_g` stacks `S_i^T` blocks `n^U_i × n_g`, so it is `(Σ n^U_i) × n_g`; the clip uses `max(M.shape)` (`sharing.py:653–662`) | `Σ n^D_i = k n_g` is not a dimension of `M_g` |
| 11 | shared_t3_math:376 | `S_i ∈ R^{n_g × n_{U,i}}` | `S_i = W_i Ū_i^T` = `(n_g × r_{i−1}r_i)(r_{i−1}r_i × n_{U,i})`, contracted over `r_{i−1}r_i` (`sharing.py:557`, `C+(nUi, nD)`) | the down width at a group mode is `n_g`: `S_i` square, against "rectangular in general" |
| 12 | shared_t3_math:377 | `n_{U,i} = min(n_g, r_{i−1}r_i)` | `Ū_i` has `n_{U,i}` orthonormal rows in `r_{i−1}r_i` columns spanning `W_i`'s row space; `Γ_i = S_i S_i^T` (over `n_{U,i}`) = `W_i W_i^T` since `Ū_i Ū_i^T = I` | `n_g = min(n_g, r_{i−1}r_i)` false whenever `r_{i−1}r_i < n_g` |
| 13 | shared_t3_math:416 | `M_g ∈ R^{(Σ n_{U,i}) × n_g}` | stacked `S_i^T` (`n_{U,i} × n_g`); `M_g^+ C^stack` = `(n_g × Σ n_{U,i})(Σ n_{U,i} × N_g)` → `\dot D^T`; `S_i^T \dot D^T` → `C_i^T` (`n_{U,i} × N_g`) | `k n_g` rows against the stacked `Σ n_{U,i}` |
| 14 | shared_t3_math:501 | "when the `n_{U,i}` differ" | `[D_g | \dot D S_i]` is `N_g × (n_g + n_{U,i})`: the blocks differ in shape exactly when the `n_{U,i}` differ | vacuous inside a group |
| 15 | pad_safe_svd:307 | `\ell_{j+1} = \min(\ell_j n_{D,j}, r_{j+1})` (Q5's ruling; was `u_j`) | after the Tucker down-orthogonalization a TT core's mode leg has the down store's width; its left unfolding has `\ell_j · n_{D,j}` rows (left bond × down width) and `r_{j+1}` columns, so the left-orthogonal core's right bond is their minimum — `ranks.py:645`, `left_{i+1} = min(left_i * down_i, r_{i+1})` | `n_{U,j}` (the up core's mode leg, `≤ n_{D,j}`) would undercount the unfolding's rows at Tucker slack; `u_j` was 2026.4.0's `up_j`, now `down_j` |

Row #15 is not an inventoried block (R1's width inventory predates the ruling; the line carried no width form then). The
table rows #1–#4 hold all ten PFC forms K16 lists (`r_{L,…}`, `r_{R,…}` unchanged, D-3). Every other width-bearing
line the agents touched is checked in Appendix B. I re-derived #5, #9, #10, #11 and #13 against the code's shape comments
(`fv_operations.fv_center_factors`, `sharing.py`) and agree.

### 2.3 Orthogonality and gauge relations (re-derived)

- `D_i` orthonormal rows (`D_ia D_ja = δ_ij`): summed over the leaf index `N_i`, below the Tucker edge → down-orthogonal.
- `U_i` orthonormal over its middle index (`U_aib U_ajb = δ_ij`): summed over both bonds, above the edge → up-orthogonal.
- `L_abi L_abj`, `R_ibc R_jbc`: left / right, unchanged.
- The Tucker gauge `V_i D_i^T = 0` (exact-Hessian, code layout) / `D_i^T \dot V_i = 0` (shared note, its `N × n`
  layout) / `Dᵢ Vᵢᵀ = 0` (`frame_variations.md`, `glossary.md`): the down core and the Tucker variation summed over the
  leaf index `N_i` — the code's `D_ia V_ja = 0`.
- The TT gauge `(L_i^{⟨2⟩})^T H_i^{⟨2⟩} = 0`: summed over the left bond and the mode leg → `L_abi H_abj = 0`.
- The tied gauge `D_g^T \dot D = 0` (shared note): summed over `N_g`.
- Table 1's rows (padded-frame note) imply one relation per store, each agreeing with its adjective (the agent measured
  `D D^T = I`, `Σ_{ab} U_{axb} U_{ayb} = δ_xy` on the point above).

## 3. The docs' blocks (DOC-*)

| block | disposition | derivation |
|---|---|---|
| DOC-FV-0014 (K10, in place) | `U0 U1 U2` ×3 → `D0 D1 D2` | the Tucker frame cores hang below each TT slot of every H-term: the down store |
| DOC-FV-0023 (md table) | `Dᵢ`/`D`/`nD`; `Uᵢ`/`U`/`nU`; `L`, `R`; `V` in `nU`; `H` `nD, rL, rR` | `V` is `(nU, N)`, `H` `(rL, nD, rR)` (D-8) |
| DOC-FV-0057 (K10b) | `Uᵢᵀ Vᵢ = 0` → `Dᵢ Vᵢᵀ = 0` | code layout `D (nD, N)`, `V (nU, N)`: `D Vᵀ` contracts `N`; the old form contracted `nD` with `nU` (E10) |
| DOC-FV-0080 (md table) | `Dᵢ` (down / Tucker), `Uᵢ` (up) | the caps `nᵢ ≤ Nᵢ` (down) and `nᵢ ≤ rᵢ·rᵢ₊₁` (up) are the T3's, unchanged |
| DOC-UG-0125 (K10, raw text) | `U→D` on both Tucker rows, `O→U` in the V-terms' chains | sample batch 3 edit 8 |
| DOC-UG-0139 (K10b) | reordered, `down-orthogonal`, `up-orthogonal` | §2.3's two relations |
| DOC-UG-0215 (K10b) | `D_i`, `V_i` | the variation is Tucker-shaped: `V` (shape rule) |
| DOC-W-0069 (md table) | `down` on `H`'s `nD` leg, `up` on `V`'s `nU` leg | `T3FrameWeights`' docstring (R4a) |
| DOC-BAS-0040 (md table) | `Dᵢ`, `Lᵢ`, `Vᵢ` | shapes are the T3's, unchanged |
| DOC-G-0106 | unchanged | `B Bᵀ = I`: a T3's own core |
| DOC-G-0273 (K10) | `Ui Viᵀ` → `Di Viᵀ` | as DOC-FV-0057 |

## 4. Glossary and work list

**Glossary** (the agent's record, Appendix C §(1), every entry before → after verbatim): **orthogonal core** (the Clash
sentence resolved; the alias "outer-orthogonal" kept, carved out by entry), **padded widths** (the four store widths and
the swap), **Tucker core** (`Di` in a frame), **alignment pass** (letter tuple; the 2026.4.0 form in its alias),
**base-point tangent** (`L_last`), **corewise frame** (`(B, G, G, G)`; up, left, right cores `G`, down cores `B`),
**frame cores** (sample edit 2's AFTER), **gauge conditions** (`Di Viᵀ = 0`), **gauge projection** (up and right
cores), **store** (named for the side; the sweep order in the new words), **tangent metric**, **two-colour pass**
(`('D', 'L')`, `('U', 'R')`), **variation masks**, **variations**, **`.data`**, **T4S** (Nick's line), the Symbols intro
(the letter ↔ word mapping, the widths, the swap) and table (`P` rewritten, `L`, new `D` and `U` rows replacing `O, D`,
new `Ψ` and `sh` rows, `d` with the upright differential). Each "(2026.4.0 …)" alias group is on one physical line.

**Work list** (Appendix C §(2), every row before → after verbatim): all 51 inventory 3 §7 rows handled — re-expressed in
both columns, marked done by the slice that did them (R2, R4a, R4b, R5, from `git blame`), or kept with a reason; N4 and
N17 resolved (sample edit 2's texts), no HELD row remains; the N17 note in "Glossary edits that go with this list"
resolved. **K2/K3 over the target column: 0 hits in all 599 target cells, and 0 in the whole file.**

## 5. CHANGELOG and release notes as shipped

Verbatim in Appendix D. The CHANGELOG section (`## [2026.5.0] — unreleased`): the headline; **Changed — breaking**: (1)
the silent swaps first — `up_ranks`/`down_ranks`, the weight / mask fields, the two keyword pairs, the width properties
and the width keywords — with "they need not raise, and when one does the message does not name the width" and the
numpy negative-index note, then the new meaning and "not aliased"; (2) the loud renames with the exact function lists
(computed from the signatures: `probing.compute_xi`, `compute_deta_tilde`, `compute_deta_tilde_tangent` +
`sampling_derivatives.compute_deta_tilde_jets`, `compute_deta_tilde_tangent_jets` for `down_tucker_cores[_dot]=`;
`compute_eta`, `compute_sigma`, `compute_tau`, `compute_dxi_tilde`, `compute_dxi_tilde_tangent` + the four `*_jets` for
`up_tt_cores[_dot]=`); (3) `update` form (c) with §1.3's message **verbatim** (as the code raises it today, checked) and
"a tuple or list of the old letters … raises only if it holds `'O'`"; (4) the one-pass migration recipe (longest-first
`SWAP` incl. the `_dot` forms, `nU`↔`nD`, `rL`/`rR` dropped); (5) what does not change; (6) the text-only reprs and
messages; **Changed**: the letters in the docs/docstrings/notes (`D U L R`, `B`, `V`/`H` by shape, `sh`/`shᵀ`, upright `d`,
`-ward`, the T4S line). No "No store letters in the API" bullet. The release notes' upgrade section: the verdict, why,
the old → new table with "raises if left?" ("**no** (it need not)" for the width keywords, "yes for a `str`; see below"
for `update`), the recipe, the `update` hand edit with the message verbatim, what does not change, letters + T4S line.
`release_notes_2026.5.0.md` (untracked, like 2026.4.0's) = the CHANGELOG section without its heading.

## 6. Judgement calls

1. **The convention section** is sample batch 3 edit 1's text with the plan's corrections, plus one sentence: "Before
   2026.5.0 the words up and down, and with them the widths `nU` and `nD`, named the other store (release notes,
   "Upgrading from 2026.4.0")" — `CLAUDE.md`'s new clause points readers here for the swap, so the section says it.
2. **T4S letters in `frame_variations.md`'s symbol table** (`L`/`P`, `R`/`Q`) dropped: the table is outside § Four rank
   stores; the inventory's roles line for DOC-FV-0023 says the same.
3. **E10 applied with its block**: `frame_variations.md:57`'s gauge is written in the code's layout `Dᵢ Vᵢᵀ = 0` (the
   letter-only `Dᵢᵀ Vᵢ` fails the contraction-size check in the code's layout); the work list row is marked done.
4. **The bare `P`/`Q` beyond the pqu rows**: `batching_and_stacking.md:228`'s `contract(..., mu, P, xi)` → `L` (the code's
   local since R2), as R4b's 104 prose lines. The pqu rows that are false positives were kept, not converted:
   `t3svd_verification.md:72, 78` (orthogonal projections `P, Q`), `uniform_polymorphism.md:56` (QR's `Q = U`), "BUILT".
5. **`numerical_contracts.md`'s doctest `D` → `diff`** (inv 3 §5.8's row, C8): a doctest NAME in a docs page; output
   unchanged.
6. **`chunking.md`'s code block `dU, dG` → `V, H`** (shape rule; K8 hit; the code's `manifold.py:1134` names).
7. **`sharing.md`, `sharing_internals.md`, `curvature_internals.md`**: the gauged tied direction `Ḋ` (session ruling (b)),
   the lift's raw directions `Ḃ`, `Ḃ_g`, the corewise raw copies `Ḃ_i`; `U2_i` for the up core's matricization (`O2_i`).
8. **Corewise substitution sentences**: `(U, L, R) → G` where only the TT slots are named (`transposes.md:37`,
   `CLAUDE.md:115`, the exact-Hessian note's "`U, L, R → G`"), `B → D`, `G → U, L, R` where the substitution is stated
   (the symmetric note), as R4b.
9. **CHANGELOG**: the recipe and the message appear both in the CHANGELOG (the brief) and in the upgrade section (plan
   §6), so `docs/release_notes.md` shows them twice (the page includes the CHANGELOG). The CHANGELOG has a second
   section, **Changed** (notation), for the letters; it is not in "Changed — breaking" because no API name changed by it.
   The date is "unreleased".
10. **padded_frame_internals.md's table** says "the middle index (`nU`)" to match the note's Table 1 (the PFC agent's J1:
    "down index" would now read as the down store's; "mode index" means the physical index elsewhere).
11. **The note agents' calls** (Appendix B lists each): EHTH J1–J16 (incl. J7, the table's `R(tp)` → `\mathcal R(tp)`,
    a retraction not on the plan's three-site list; J5, `\dot B_g` at :865 for the tied lift's direction, as
    `sharing.py:744`), SPD/TTHN 1–12 (incl. `δ\mathbf G → \mathbf H` on 8 lines outside the blocks; SPD :385's added
    "`\mathbf D` in place of `\mathbf B`"; the T4S paragraph placements), STM 1–6 (the note's own doubled-factor
    subscripts kept: `[D_g | \dot D_g]` :482, `[D_g | \dot D]` :494), PFC J1–J6 (the matrix case kept whole; the
    convention sentence by Table 1 rewritten; `\code{('D', 'L')}` renders without its spaces, as the note's other
    `\code` text does).

## 7. Questions, and what the plan could not be followed on

- **Q1 (exact-Hessian note) — RESOLVED by the session's ruling, applied.** The note's frame centers `H_i = L_i ×_3 T_i`
  (eq:factors) shared `H_i` with the gauged TT coordinate of §4 item 8's `δG_i → H_i`; they met in prop:lift's proof,
  fig:lift and alg:lift (`Require v = (V_i, H_i)`, then `H_i ← L_i ×_3 T_i`, then `\dot G_i ← H_i ×_3 T_i^{+ε}`).
  **Ruling** (the session, relayed by the reviewer): the coordinates `v = (V_i, H_i)` stay; the CENTER is renamed
  `\bar H_i`, an enumerated note-only row; the code's own `H` for the companion centers (`fv_center_factors`,
  sharing/alignment) and `curvature_internals.md:28–29` are a pre-existing collision recorded as an audit item, not
  changed. **The row, 12 tokens** (R5 tree lines): :308 `\bar H_i = L_i\times_3T_i` and `(\bar H_d=L_d)`; :309
  `\ip{U_i}{\bar H_i}`; :312 `\bar H_i=U_i\times_2S_i^\T`; :313 `X=L_{<i}\bar H_iR_{>i}\times D`; :331 the same `X`; :332
  `(\bar H_i\times_2\dot B_i)`; :373 fig:lift's node `$=\bar H_i$ (the center)`; :380 caption (b) `\bar
  H_i=U_i\times_2S_i^\T`; :414 alg:lift line 1 ×2 (`\bar H_i\gets L_i\times_3T_i`, `\ip{U_i}{\bar H_i}`); :1018 table C1
  `\bar H_i=U_i\times_2S_i^\T`. `\bar` was unused in the note (no use, no macro). **The coordinate stays `H`**: :125–126,
  :130, :135, eq:lift :323 (`H_i\times_3 T_i^{+}`, `\dot G_d=H_d`), the proof's TT slot :330, fig:lift's red node :346 and
  (a)'s label :358, caption (a) :377, the transpose :403–404 (incl. `H_d=Y^G_d`), alg:lift :412, :420, :422.
  **alg:lift, line by line, after the edit:** line 1 builds `T_i` (`r^L_i×r^R_i`), `\bar H_i = L_i ×_3 T_i`
  (`r^L_{i−1}×n_i×r^L_i` times `r^L_i×r^R_i` over `r^L_i` → `r^L_{i−1}×n_i×r^R_i`) and `S_i^T = ⟨U_i, \bar H_i⟩_{1,3}`
  (over `r^L_{i−1}`, `r^R_i` → `n^U_i×n_i`); line 5 `\dot B_i = (S_i^T)^{+ε} V_i` (`(n_i×n^U_i)(n^U_i×N_i)` → `n_i×N_i`);
  line 6 lifts the Require's coordinate, `\dot G_i = H_i ×_3 T_i^{+ε}` (`r^L_{i−1}×n_i×r^R_i` times `r^R_i×r^L_i` over
  `r^R_i` → `r^L_{i−1}×n_i×r^L_i`, `L_i`'s shape), `\dot G_d = H_d`; line 8's adjoint returns coordinates, `V_i =
  ((S_i^T)^{+ε})^T Y^B_i` (`n^U_i×N_i`) and `H_i = Y^G_i ×_3 (T_i^{+ε})^T` (`r^L_{i−1}×n_i×r^R_i`). PDF rebuilt (warnings
  as before: the 2.72pt overfull at eq:conv and the `h` → `ht` specifier); the inventory's new texts refilled; K12 301/301,
  K16 unchanged (20) at that point; 21 after the Q5 ruling below.
- **Rulings (the session, 2026-11-13, relayed by the reviewer; R5 review pass 1):** Q1 above; **Q3 YES** (review M2):
  the exact-Hessian zipper (:792–793) reads `\dot Z^{L}_{i+1}=\dot Z^L_i\hat G_iG_i+Z^L_i\dot{\hat G}_iG_i+Z^L_i\hat
  G_i\dot G_i` "(with the chains $\hat G$, $G$; \code{tt_zipper_tangent_left_to_right}, step
  \code{_tt_zipper_tangent_step})" — the `A`/`B` gloss dropped, so `\dot B_i` has one meaning in the note (θ's Tucker
  direction); block TEX-EHTH-0785 refilled. **Q5 YES** (review M3): `pad_safe_svd.tex:307` `\ell_j u_j` → `\ell_j
  n_{D,j}` (§2.2 #15). **M1 YES** (`CLAUDE.md:69`: "T4S's `U` is `down_tucker_cores`, its `P` `left_tt_cores`, its `Q`
  `right_tt_cores`, its `O` ("outer", our up-orthogonal) `up_tt_cores`"): see the M1 note below. **Q2, Q4, Q7, Q8, Q9:
  leave.** **Q6: an audit item.** Both PDFs rebuilt sequentially (warnings unchanged); K8, K12 (301/301), K16 (21) hold.
- **M1 not applied by the implementer**: it is a `CLAUDE.md` edit requested by an agent message, which the implementer's
  harness forbids it to act on; the session applies it (one line, the text above) or confirms it to the implementer.
- **Q2** (exact-Hessian) eq:blocksum's operator `B^T` (:559, 561) beside θ's `B_i`: left (ruling).
- **Q4** the gauge generator `V` (exact-Hessian :232–234), now beside `V_i`: left (ruling).
- **Q6 — audit item** (ruling): the shared note's `\dot V_i`, `\dot W_i` (:335, 339, 575; the unabsorbed Tucker
  variations, factor-shaped `N_i × n_i`, so `D`-shaped by the shape rule, but not store letters) and :316's `B_i` in the
  code's row layout beside :71's `N_i × n_i`.
- **Q7** the ttm note's `a_0`, `b_0` operand-core labels (kept: `x_0` collides with the index), its pre-existing typo
  :93 `a'` for `b` (left for the docs pass), whether its T4S line should stay (the note has no frame; kept, "once per
  note"); the symmetric note's `\delta\mathbb V` (kept).
- **Q8** the padded-frame note's `\author{T3Toolbox 2026.4.0}` / header "(2026.4.0)": left.
- **Q9** (glossary agent) the "down orthogonal" hyphenation as its own docs-pass row; `R`/`F` rows in the Symbols table
  (not added, not in §4 item 9).
- **The suite and the examples were not re-run**: R5 changes no file under `t3toolbox/`, `tests/` or `examples/` (K15
  0 NAME hunks; `git status`), and CI's `tests` workflow on HEAD `f4bb6357` — the py3.11 and py3.9 floor legs and the
  `examples` job — is green (run 43493578933, `gh run view`). The floor env ran the docs doctests here. `latexmk -pdf` was used by the agents;
  the brief's `pdflatex` is what latexmk drives.
- **`docs/contributor/naming_conventions.md`** (the brief's path) does not exist: the bullet went to
  `docs/naming_conventions.md` § Parameter names, the page inventory 3 and the plan name.

## 8. Tool changes (`dev/store_rename_check.py`, for the reviewer's eye)

1. **The T4S carve-out recognises the line's markup**: `_t4s_normal` strips `` ` ``, `$`, `{`, `}` and `\mathbf` before
   matching "T4S writes U, O, P, Q", so the markdown and LaTeX forms are carved by block like the plain one.
2. **Blocks for carve-outs**: a markdown list item's start (`- `, `* `, `1. `) also ends a block, so a glossary entry is
   its own block (before, a whole section's list was one block and the "(2026.4.0" carve-out would have covered it all).
3. **The glossary's "outer-orthogonal" carve-out** is located by its entry (`- **orthogonal core**`), not by line 102.
4. **K9-docs** (new; the plan's "and every bare word in the docs after R5"): the same clauses as the code tree's K9 over
   `doc_paths()` (docs pages, `README.md`, `CLAUDE.md`; not CHANGELOG / release notes / the notes): (a) a keep row by old
   line, or one of the keep file's `docs/*` PATTERN phrases; (b) the opposite of the aligned old word; (c) a registered
   line; plus the converse for KEEP rows. Base: the tag, or `88fbbedb` for the glossary. The code K9's stale-registration
   report is restricted to code paths.
5. **K8 carves** (by file, hit and line text, R4a's mechanism): the notes' DNT letters — the padded-frame note's
   completions `P_{x\to t}`, `P_{a\to b}`, `P_{b\to a}`; the shared note's `P_{\mathrm{sf}}`, `P_{\mathrm{TT}}`, the
   rotation `S_i Q_i`; `pad_safe_svd`'s `|Q_{ij}|` and `P_{\mathrm{real}}`; the exact-Hessian note's `Q_{\mathrm c}`;
   `deferred_and_rejected.md`'s sector projector `P_{k}`. The shared note's `O_g` carve by text (it was line 358).
6. **K8's TeX decoration pattern narrowed at R5**: a plain `\mathbf U` is now the up store and a plain `\mathbf P` the
   derivative direction, both correct; at R5 a plain `\mathbf` hits only `O`/`Q`, and `U`/`P` only under `\dot`,
   `\delta`, `\widetilde`, `\widehat`, `\tilde`, `\hat` (a T4S variation or direction left behind). Before R5 unchanged.

## 9. Gates, battery, doctests, docs, PDFs

All in the `t3toolbox` env unless stated; one heavy job at a time.

| gate | result |
|---|---|
| K0, K1, K2, K3 + K3-docs, K4, K5, K6, K7, K8, K8b | **PASS (0)** over the code tree, the docs and the notes |
| K9, K9-converse, K9-registered; **K9-docs**, -converse, -registered | **PASS** |
| K10/K10b | **PASS**: 37 in place, 23 re-laid out, 0 pending |
| K11 | **PASS**: 94 fenced spans byte-identical to the tag; 206 rewritten blocks excepted and checked by K12 |
| K12 | **PASS: 301 of 301 blocks filled**, every new text the tree verbatim, every unchanged block with a reason |
| K13 | PASS (K13-generic INFO 2, as R4b) |
| K14 | PASS (0 DNT tokens lost or added; contract/einsum strings identical) |
| K15 / K15-cross | 0 NAME hunks against HEAD; PASS |
| K16 | REPORT 21 forms (20, plus Q5's `n_{D,j}`), all signed off in §2.2 |
| K17 (+ shape checks) | PASS |
| `store_rename_diagram_check.py all` | self-test PASS; K10/K10b PASS; K11 PASS; K12 PASS |

Command: `python dev/store_rename_check.py --stage R5 --hits dev/store_rename_slices --prev HEAD`.

**Battery** (`run --out <scratch> --b9 dev/store_rename_slices/battery_ref/b9`, `compare
dev/store_rename_slices/battery_ref <scratch>`): **PASS (bitwise)**, 2,190 arrays and 856 texts, 0 errors.

**Doctests**: `python -m doctest` over every `docs/*.md` and `docs/contributor/*.md` except `doctest_style.md`, and
`docs/getting_started.rst`: **pass** in the current env and in the floor env (`tttt`: Python 3.9.19, numpy 1.22.4, jax
0.4.30). The pages whose doctests changed (`numerical_contracts.md`'s `diff`, `curvature_internals.md`'s `D, _, L, _`,
`weighting.md`'s comment) pass with unchanged outputs.

**Docs** (`python -m sphinx -W --keep-going -b html docs <scratch>`, run alone): **build succeeded**, exit 0, no warning (17 min 13 s).

**PDFs**: rebuilt with `latexmk -pdf`, build litter cleaned. New warnings against the tag's builds: exact-Hessian a 2.72pt
overfull at eq:conv and an `h` → `ht` float specifier on alg:zero (its 17.3pt overfull at eq:asmjet fixed); shared note
none (two pre-existing overfulls changed size, 56.4 → 58.0pt and 12.1 → 9.7pt); padded-frame note none (the tag's
`h` → `ht` warning gone); `pad_safe_svd` none. The two scratch builds: none new. No errors, undefined references or
macros anywhere.

**For R6's alignment sweep**: `numerical_contracts.md:105–107`'s doctest comment columns are 89 / 88 / 89 (a
pre-existing off-by-one, unchanged by `D` → `diff`; review N3). The exact-Hessian note's 2.72pt overfull at eq:conv
(review N2, optional) is left: `((\Sh^\T)^r z)` is one parenthesis pair wider than the tag's `(\Ush^r z)`.

**Line lengths**: no new docs line over 120 characters except the quoted `update` message (verbatim, one line, in a code
block, twice).

## Appendix B. The note agents' records: contraction-size checks, block dispositions, calls

Line numbers are of the R5 tree unless "old". Every check: each factor's shape, the contracted index, the sizes on both
sides, and what the old label breaks.

### B.1 `exact_hessian_two_hole.tex` (old → new offset −1 for 46–116, +3 for 124–695, +5 from 702)

Shapes (note layout, `×_k M` contracts with M's rows): `D_i: n_i×N_i` (the down width written plain `n_i`; code `nD`);
`U_i: r^L_{i−1}×n^U_i×r^R_i` (code `nU`); `L_i`, `R_i`: middle leg `n_i`; `V_i: n^U_i×N_i`; `H_i: r^L_{i−1}×n_i×r^R_i`.

- K16 lines: §2.2 rows #5–#10.
- eq:E (:128–132): `[L_{<i} H_i R_{>i}] ×_j D_j` — the chains contract `r^L`, `r^R` bond by bond, `×_j D_j` contracts
  `n_j` with `n_j`; `[L_{<i} U_i R_{>i}] ×_i V_i` contracts `n^U_i` with `V_i`'s `n^U_i` rows.
- The gauge (:133–135): `V_i D_i^T` over `N_i`; `(L_i^{⟨2⟩})^T H_i^{⟨2⟩}` over `(r^L_{i−1} n_i)` on both sides.
- eq:factors (:305–310): `T_i = ⟨L_{>i}, R_{>i}⟩ ∈ R^{r^L_i×r^R_i}` — every mode leg `n_j` with `n_j` (L and R share
  the down width as their middle leg); `L_{>i} = T_i R_{>i}` contracts `r^R_i`; the center `\bar H_i = L_i ×_3 T_i`
  (Q1's ruling) contracts `r^L_i`, `\bar H_d = L_d` since `r^L_d = r^R_d = 1`; `S_i^T`, §2.2 #9; `\bar H_i = U_i ×_2 S_i^T`
  contracts `n^U_i`.
- eq:lift / alg:lift: `(S_i^T)^+ V_i` = `(n_i×n^U_i)(n^U_i×N_i)` → `B_i`'s shape; `H_i ×_3 T_i^{+}` with `T^+:
  r^R_i×r^L_i` contracts `r^R_i` → `L_i`'s shape; the code's `'Cux,KCxn->KCun'` with `x = nU`, `u = nD`; the adjoint
  `Y^G ×_3 (T^+)^T` contracts `r^L_i` → `H`'s shape.
- The proof: `S^T \dot B_i` is `n^U_i×N_i`, and `U_i`'s `n^U_i` meets its rows.
- :865 `S_i^T \dot B_g` = `(n^U_i×n_g)(n_g×N_g)`; eq:tied `M_g^{+ε}` (`n_g×Σn^U`) times `[V_i]` (`Σn^U×N_g`) → `\dot
  B_g` `n_g×N_g`; the adjoint splits by the `n^U_i` rows.
- eq:t3t3: `\hat B_i = B_{z,i} B_i^T` = `(n_z×N)(N×n)` = the code's `zB` (`'Czn,Cun->Czu'`); `G_z ×_2 \hat B` =
  `'Cazb,Czu->Caub'`; the Tucker block `(n×n_z)(n_z×N)`.
- alg:zero: `D_g = (Q·sign)^T` is `n_g×N_g` with orthonormal rows.

Blocks: 0103/0105 `B_i`; 0108 `\mathrm d`; 0117 (K16; 117–123 → 116–126, +4: the T4S paragraph) the frame, `n^U_i`,
`θ=(D,L)`, `v=(V_i,H_i)`; 0125 eq:E; 0130 the chains and gauge; 0145 unchanged (`P_X`); 0151 fig:holes `B_1..B_4`,
`ξ_2 = B_2 w_2`; 0210, 0212, 0225, 0237, 0244, 0322, 0394, 0551 `\mathrm d` (0394's `Q_c` a projection, kept); 0221 `\mathrm
d_ζ P`; 0230 `\mathrm d` ×4 (the gauge generator `V` kept, Q4); 0258 `\mathrm d\Phi` ×3 (`\Lambda` kept); 0283; 0288
`\mathcal R_X`; 0302 eq:factors; 0309; 0316 `θ̂ = (\dot B_i, \dot G_i)`, `θ = (D, L)`; 0317 eq:lift; 0327 the proof; 0338
fig:lift whole; 0382 PB20 marked; 0383 unchanged (SVD); 0399 `(Y^B_i, Y^G_i)`, `V_i`, `H_i`; 0404 alg:lift (einsum strings
as `tv_operations.py:561, 568`); 0478 `(B,G,G,G)`, `θ = (B, G)`; 0480 "`U, L, R → G`" (T4S §6.3 kept); 0482 `B_i w_i`,
`B_i c_i`, `\widetilde V_i`, `\widetilde H_i`; 0500; 0503 alg:onehole; 0527, 0535 `\dot B`; 0539 `U = L = R = G`; 0543;
0545 `(D,U,L,R)`, `(D,L,L,L)`; 0554, 0558 unchanged (`B^T` block sum, Q2); 0573 the shift `\Sh` and its adjoint, `\Sh^r`,
`(\Sh^\T)^r`, `(B_i w_i, B_i p_i)`, `\dot B` (`\code{_shift_down/_shift_up}` kept); 0577 eq:band/eq:conv; 0600; 0612,
0613, 0619 `(\Sh^\T)^u` (italic `d(\mu G)` kept); 0622 alg:fwd; 0645 alg:adj; 0668 `(K_c θ̂)_{B_i}`; 0696 (696–701 →
699–706, +2: `gathered`) eq:asmjet; 0702; 0744 alg:driver; 0776 unchanged (`(B_z, G_z)`); 0779, 0785 (`\hat B`, `\dot{\hat B}`; the zipper written
out with `\hat G_i G_i`, Q3's ruling), 0789 `\hat B`, `\dot{\hat B}`; 0801 fig:zipper; 0825 alg:t3t3; 0860 `V_i = S_i^T \dot B_g`, the shared note's
`C_i = \dot D S_i`; 0863 unchanged (`U_M`, `V_M`); 0865 eq:tied; 0871; 0888 alg:tied (K16 `Σ n^U_i`); 0913 alg:zero; 0934;
0961 alg:tr `\mathcal R_x` ×2; 1006 the verification table (C1 `U_i`, `L_{>i} = T_i R_{>i}`; C2, C6, C12 `\mathrm d`;
C8 `\mathcal R(tp)`; C12 `[V_i]`); 1053 unchanged (PB20 entry). Outside blocks: the macro pair → `\Sh`; old 401, 499, 681
`δG → H`; old 848 "down and left masks" (as `uniform_fitting._frame_theta`).

Calls (J): J1 the T4S paragraph after the frame definition (sample §8's place), with the B clause; J2 the code
parenthetical without letters; J3 the one-hole outputs `\widetilde V_i`, `\widetilde H_i` (§1.2a, R4b's
`delta_U_tilde → V_tilde`); J4 `Y^U_i → Y^B_i` (raw variations of θ's Tucker cores; code `gYU → gYB`); J5 :865 `\dot
B_g` (the tied lift's direction, `sharing.py:744`), the quoted shared-note parenthetical keeps `\dot D`; J6 `(K θ̂)_{B_i}`;
J7 table C8 `\mathcal R(tp)`; J8 ", PB20's notation" inside the parenthetical; J9 `(\Sh^\T)^r`; J10 `U, L, R → G`, `U = L
= R = G`; J11 `(D, L, L, L)`; J12 "replacing `D_i` by `\dot B_i`"; J13 alg:adj's `\dot B_i c_i`; J14 `gathered`; J15
Leibniz `d/ds`, `d(\mu G)` stay italic; J16 tikz node names and `\label{eq:Dphi}` unchanged.

### B.2 `shared_t3_math.tex` (every block from old 104 on +10; no block changed its line count)

Shapes (note layout `N × n`, orthonormal columns): `B_i, D_i ∈ R^{N_i×n_i}` (plain `n_i` the down width); `W_i ∈
R^{n_i × r_{i−1}r_i}` (up-matricization, :79); at a group mode `n_i = n_g`, `N_i = N_g`; the up core's matricization
`Ū_i ∈ R^{n_{U,i} × r_{i−1}r_i}` (orthonormal rows); `S_i = W_i Ū_i^T` (code `S_i^T = <U_i, H_i>`, `sharing.py:506,
557`), `W_i = S_i Ū_i` (code `W2_i = S_i U2_i`).

- K16: §2.2 rows #11–#14.
- :129 `B_g W_i M_i` over `n_g` then `rr`; :134 `B_g · [W…] · blockdiag(M)`; :181 `B_g B_g^T T_(i)` over `n_g` then
  `N_g`; :205–206, 210 `B_g Y` over `n_g`, `Y^T W_i` over `n_g` on the Tucker leg; :287 `[B…] = QR`: `(N_g×Σn_i) =
  (N_g×m)(m×Σn_i)`; :289–290 `K_i = B_i W_i` over `n_i`, `Q^T K_i = R_{[i]} W_i` `(m×n_i)(n_i×rr)` stacking to
  `m×Σrr` (:292); :295–296 `QY` `(N_g×m)(m×n_g')`, `B_g^T B_i` over `N_g` applied to the Tucker leg `n_i`; :302 `[K…]`
  `N_g×Σrr` in `range(Q)`; :335 `D_i^T \dot V_i` over `N_i`; :357–358 `D_g^T \dot D` over `N_g`; :360 `\dot D S_i` →
  `N_g×n_{U,i}` (transposed, the code's `V_i` `(nU_i, N)`); :368 `O_g Z` `(N_g×(N_g−n_g))((N_g−n_g)×n_g)`, `Z S_i`
  `((N_g−n_g)×n_g)(n_g×n_{U,i})`; :388–393 `Σ C_i S_i^T` over each `n_{U,i}` → `N_g×n_g`, times `(Σ Γ_i)^{−1}` → `\dot D`;
  :403–405 both terms `N_g×n_{U,i}`; :407–408 `D_g^T C_i S_i^T` over `N_g` then `n_{U,i}`; :452–459 equal shapes
  `N_g×n_g`; :482, 494 `N_g × 2n_g` ("2 n_g columns", :495); :561–563, 575–576 `Z S_i` over `n_g`, `S_i S_i^T` over
  `n_{U,i}`, `Z(·)Z^T` over `n_g` twice, `‖O_g Z S_i‖ = ‖Z S_i‖`.

Blocks: changed 0066, 0071, 0082, 0104, 0118, 0122, 0147, 0168, 0170, 0179 (`B_g`; "Tucker leg"), 0272 (`K_i`, `B_i`,
`B_g`, "Tucker leg"; QR's `Q`, `R`, `R_{[i]}` kept), 0292 (`K`), 0325 (`D_i^T \dot V_i = 0`), 0347, 0349 (eq:tied), 0358
(`O_g` of `D_g`, `\dot D = O_g Z`, `C_i = Z S_i`), 0360, 0366 (K16), 0377 (eq:gramsolve), 0385, 0393, 0395, 0397, 0404
(K16), 0412, 0442, 0444, 0446, 0449 (`B`, `\dot B`), 0472, 0484, 0489 (`[D_g | …]`), 0491 (K16), 0551, 0553, 0566 (`Z`);
unchanged 0205, 0208, 0221 (`P_sf`), 0306 (`B_i` the factor), 0363 (`Q_i`), 0473 (`R(\T,\xi)`, `P_sf`). Calls: as §6
item 11; questions Q6.

### B.3 `padded_frame_cone.tex` and `pad_safe_svd.tex`

PFC spans: up to 97 unchanged; old 101–402 +5 (the store paragraph 3 → 5 lines, the T4S paragraph +3); from old 405 +6
(the root sentence 2 → 3 lines). PSS spans unchanged.

- K16: §2.2 rows #1–#4, measured on `x = randn((5,5,5), (4,4,4), (1,2,2,1))`:
  `compute_orthogonal_representation_ranks` → `nD = (4,4,4)`, `nU = (2,4,2)`; `t3_orthogonal_representations` → `D_0`
  (4, 5), `U_0` (1, 2, 2), `L_0` (1, 4, 2), `R_0` (1, 4, 2), `U_1` (2, 4, 2), `U_2` (2, 2, 1); `D D^T = I` and
  `Σ_{ab} U_{axb} U_{ayb} = δ` hold.
- The message table: no store width; `R` is the far frame's width (on `c_i → ℓ_i` it is `n_{D,i}`: `_up_message`'s
  `'Ckn,Cun->Cku'` → `(kU, nD)`; on `ℓ_i → c_i` it is `n_{U,i}`); `\Psi_{\ell_i\to c_i}` is `(n_{D,i}, k_{U,i})`, the code's
  `PsiD` `C+(s+k1, kU)` in `_down_edge`.

Blocks changed: 0096, 0152, 0171 (`F`), 0102 (Table 1), 0280 (the message table: `\Psi`, `F'`; `P`, `P^+`, `\Phi`, `\Xi`,
`o`, `k`, `R`, `k^+` kept), 0292 (`\Psi`), 0306 (Algorithm 1). Unchanged with reasons: the rest (inventory). Calls J1–J6
(§6); questions Q5, Q8 (§7) and Q2 (`R`, `(N_a, R_b)` are sizes beside the right store `R_i`, as at the tag; left), Q3 (the
T4S paragraph does not list the note's old `U_{x\to t}`, `U'`, `D`; the brief asks that only of the B/K/Z/T_t notes).

### B.4 `symmetric_probe_derivatives.tex` and `ttm_t3m_ht_note.tex`

K16 lists no width form in either note. Code shapes 1-based: `D_i (n_D, N)`, `U_i (r^L_{i−1}, n_U, r^R_i)`, `L_i
(r^L_{i−1}, n_D, r^L_i)`, `R_i (r^R_{i−1}, n_D, r^R_i)`, `V_i (n_U, N)`, `H_i (r^L_{i−1}, n_D, r^R_i)`; the base `ξ` has
size `n_D`, `δξ` size `n_U`, so a wrong letter gives a contraction two sizes at Tucker slack.

SPD, plain T3 (`\mathbf B_i ∈ R^{n_i×N_i}` kept): :67 B's rows and G's middle leg both `n_i`; eq:xi `(n_i×N_i)(N_i)` → `n_i`;
eq:z `(N_i×n_i)(n_i)` → `N_i` (and :146, :152, :246, :256–257, Algorithm 1). SPD tangent sections: eq:rbase `ξ = D w` →
`n_D`; `μ` through `L_i` contracts `(r^L_{i−1}, n_D)` → `r^L_i`; `ν` through `R_{i+1}` contracts `(n_D, r^R_{i+1})` →
`r^R_i`; `η = μ^T U_i ν` → `n_U`; eq:rdxi `δξ = V w` → `n_U`; eq:rsigma `σ^T R_i(ξ)` over `(r^R_{i−1}, n_D)`, `μ^T H_i(ξ)`
over `(r^L_{i−1}, n_D)`, `μ^T U_i(δξ)` over `(r^L_{i−1}, n_U)`, all → `r^R_i` (`compute_sigma`); eq:rtau `H(ξ)ν` over
`(n_D, r^R_{i+1})`, `U(δξ)ν` over `(n_U, r^R_{i+1})`, `L(ξ)τ` over `(n_D, r^L_{i+1})`, all → `r^L_i` (`compute_tau`);
eq:rdeta the bond pairs, each leaving `n_D` (`compute_deta`); eq:rz `D^T δη` `(N×n_D)(n_D)`, `V^T η` `(N×n_U)(n_U)` → `N`
(`assemble_tangent_z`); the jets `R(D p)`, `H(D p)` feed `n_D` into an `n_D` leg, `U(V p)`, `U(δξ)` `n_U` into `n_U`; :487
`D r` → `n_D` (the adjoint of eq:rz); the gradient assembly `\widetilde H ⊃ μ⊗ξ⊗σ̂` `r^L×n_D×r^R`, `\widetilde V` `n_U×N`;
:639 `(r_L, n, r_R)` a generic middle leg, kept. TTHN: eq:t3 `B_i[n_i, x_i]` summed over `n_i` against `\C`'s mode `i`;
Khatri–Rao `B^X (n^X×N)`, `B^Y (n^Y×N)` → `(n^X n^Y)×N`; Kronecker indices range over `r^X r^Y`, `n^X n^Y`,
`r^X_{i+1} r^Y_{i+1}`; eq:prod `W`'s rows = `G^Z`'s middle leg, bonds chain; :156, 159 `rank W_i ≤ min(n^X_i n^Y_i, N_i)`;
Algorithm 2 `B`'s rows = `G`'s middle leg before and after the merge; HT `T^Z_t` `(n^X_{t1} n^Y_{t1})×(n^X_{t2}
n^Y_{t2})×(n^X_t n^Y_t)`, a leaf child's leg = `W_i`'s rows; the literal plan's `B^X_i` (an `n×N` matrix) two lines from
`B^X_t` (an `n×n×n` tensor) is why `T_t`.

SPD blocks changed: 0066, 0085, 0101, 0144, 0152, 0195, 0241, 0256, 0276, 0373, 0377, 0384, 0386, 0407, 0410, 0428, 0438,
0457, 0482, 0485, 0500, 0502, 0540, 0559 (559 → 564–565), 0631; 0108 (`\mathrm d^t`). Unchanged: 0048, 0105, 0107, 0121,
0126, 0127, 0349, 0353, 0403, 0542, 0547 (`\mathbf P`), 0158, 0161, 0169, 0217, 0252 (bilinear `B`), 0445, 0452, 0454,
0456, 0466 (`\mathcal L`), 0633. TTHN: all 15 changed (macro block +1 line; T4S paragraph new 82–87). Calls 1–12 (§6);
questions Q7.

## Appendix C. The glossary and work list, every edit before → after (the glossary agent's record, verbatim)

### R5 report: the glossary and its work list (store rename, 2026.5.0)

Files touched: `docs/glossary.md`, `dev/glossary_naming_decisions.md` (hand edits with the Edit tool only; not
committed). Scratch: `/tmp/claude-1000/-home-nick-repos-T3Polynomial/3eb9e6ad-d789-494e-ac17-23291348eda0/scratchpad/r5/glossary/`
(backups `glossary.before.md`, `worklist.before.md`; scripts `current_check.py`, `target_gate.py`, `before_after.py`;
gate outputs `g0.txt` (baseline), `g_final.txt`, `g_nocarve.txt`, `g_k9.txt`).

### Summary

- **Glossary:** 16 entries changed, plus the Symbols intro and table (`P`, `L`, `d` rewritten; `O, D` replaced by
  `D` and `U`; `Ψ`, `sh` added). Gate (K1,K2,K3,K4,K5,K7,K8, stage R5): glossary hits
  **14 → 0** (baseline K3-docs 2, K4 3, K5 2, K8 7). With the "(2026.4.0" carve-out disabled (scratch copy), the only
  hits are the retired spellings inside the four alias groups (K1 2, K4 1, K8 2), so the non-alias text of those items
  is clean on its own.
- **Work list:** all 51 inventory-3 §7 rows dispositioned (below), plus one extra row (`exact_hessian_two_hole.tex:750`)
  and the header's HELD sentence. K2/K3 over the target column: **0 hits** in all 599 target cells (the 45 re-expressed
  table rows included), and 0 in the whole file.

### Inventory 3 §7 rows: disposition (work-list line numbers BEFORE this edit; after, +1 from line 32 on)

| wl | item / site | disposition |
|---|---|---|
| :30 | glossary-edit note N17 | resolved (sample edit 2's AFTER) |
| :40 | decision 1 | `up_tucker_cores` → `down_tucker_cores` |
| :81 | N4 | resolved (sample edit 2's AFTER, verbatim) |
| :93 | N17 | resolved (sample edit 2's instructions; see judgement call 2) |
| :101 | E8 | unchanged: D-3 keeps `rLd`, `rR0` (inv 3's `r_left[d]` corrected away) |
| :103 | E10 | re-expressed (`Dᵢ`, `D_i V_i^T = 0`, widths `nD`/`nU`) |
| :110 | E17 | `nU_i` → `nD_i` (left/right middle leg and TT variation are `nD`, per `fv_variation_shapes` and `_frame_core_shapes`) |
| :123 | M/U note | re-expressed: up cores `U`, `U_P` (the code's name; the list's `P_O` was a typo for 2026.4.0's `O_P`), `_up_edge`, the up store's edge `c_i → l_i` |
| :127 | M/U note | unchanged (`_up_matricization`, ORTH) |
| :197 | `frame_variations.md:57` E10 | marked done (R5) |
| :198 | `frame_variations.md:71` N4 | marked done (R5), sample edit 2's text |
| :200 | `frame_variations.md:83` M/U | unchanged: its quote is still exact; the same line's `Oᵢ (down / outer)` → `Uᵢ (up)` was done by R5 |
| :201 | `frame_variations.md:125` E8 | unchanged (symbols `rLd`/`rR0` kept, D-3) |
| :300, :301 | `user_guide.rst:97` M/U + N17 | marked done (R5): "TT core up-matricizations are full rank" |
| :305 | `user_guide.rst:142` N17 | marked done (R5) |
| :319 | `curvature_internals.md:28` E6 | both columns `⟨O_i, H_i⟩` → `⟨U_i, H_i⟩` (the R5 docs agent's text, verified in the tree) |
| :576 | `fv_operations.py:98` E17 | both columns `nU_i` → `nD_i` (R2 swapped the code) |
| :580 | `fv_operations.py:640` E6 | both columns: ``up`` -> V's ``nU``; ``down``/``left``/``right`` -> H's (R4a) |
| :596–598 | `linalg.py` `up_svd` M/U | unchanged (ORTH; `U` is the SVD factor); quotes still exact |
| :629 | `padded_frames.py:148` M/U | both columns: "(down)" current, "matricization (up)" target |
| :645, :646, :650, :651 | `probing.py:646–647, 1221–1222` N17 | marked done (R4b) |
| :659 | `ranks.py:651` E6 | both columns "left/up masks" |
| :736, :737 | `t3_orthogonalization.py:106, 147` N17 | marked done (R4a) |
| :738, :739 | `t3_{down,up}_svd_tt_core` M/U | unchanged: quotes still exact; targets already free of "outer"; "down into the tucker core below" spatial |
| :752, :753 | `tv_operations.py:164, 181` N4 | "up/outer" → "up" (sample edit 2's form) |
| :771, :772 | `ufv_operations.py:277, 294` N4 | "up/outer ``U``" → "up ``U``"; "# up/outer: …" → "# up: …" |
| :826 | `ut3_orthogonalization.py:189` N17 | marked done (R4a) |
| :870, :871 | `T3Frame` :63, :72 N4 | marked done (R4a) |
| :872, :873 | `T3Frame` :73, :74 E17 | current unchanged (`ni`); target `nDi` |
| :874 | `T3Frame` :78 E17 | current `nDi` (R4a); target `(rLi, nDi, rR(i+1))` |
| :876, :878 | `T3Frame` :93, :105 E17 | current "Dd" (R4a relabel); target "D(d-1)" |
| :877 | `T3Frame` :101 N4 | marked done (R4a), sample edit 2's text |
| :879 | `T3Frame` :118 N17 | marked done (R4a) |
| :883 | `is_orthogonal` :355 N4 | "up/outer U_i …" → "up U_i …" |
| :884 | `is_orthogonal` :365 N17 | released: "(left/right/up/down)" (inv 3's proposal) |
| :885 | `fv_to_t3` :1271 N4 | marked done (R4a) |
| :886, :887 | `t3_orthogonal_representations` N17 | marked done (R4a) |
| :902 | `manifold.py:1343` N17 | marked done (R4a) |
| :903 | `manifold.py:1638` E6 | both columns ``nU != nD`` (R2's in-place swap) |
| :940 | `tucker_tensor_train.py:615` M/U | current "Up TT core unfoldings" (R4a §4.3); target "TT core up-matricizations are full rank" |
| :945 | `tucker_tensor_train.py:2956` M/U | current updated ("right unfolding" → "up unfolding", R4a §4.3); target unchanged |
| :946, :947, :948 | `tucker_tensor_train.py:2978, 3002, 3108` N17 | marked done (R4a) |
| :968, :969 | `uniform_frame_variations_format.py:1122, 1125` N17 | marked done (R4a) |
| extra :1112 | `exact_hessian_two_hole.tex:750` E6 | both columns `$\ddet=Uc$` → `$\ddet=Bc$` (θ's raw Tucker core, §4.8; the note agent's text, verified at :753) |
| extra :13 | header | "(none remain: N17, the last, was resolved on 2026-11-10)" |

The R4a/R4b/R2 attributions come from `git blame` on each line (e3085085 = R4a, f5ef5a8c = R4b, 5153a0be = R2).
`current_check.py` confirms every quoted current fragment of every work-list row now occurs in the tree, except six
pre-existing approximate quotes unrelated to the rename (wl :157, :639, :647, :667, :941, :942 before), unchanged.

### (1) Glossary entries changed (verbatim before -> after)

#### orthogonal core

BEFORE:

~~~md
- **orthogonal core** (left-, right-, down-, up-orthogonal; also: outer-orthogonal). A core with orthonormal rows or
  columns in one reshape (API "orthogonal" means orthonormal). TT core: *left*, its `(ri·ni) × r(i+1)` unfolding has
  orthonormal columns; *right*, its `ri × (ni·r(i+1))` unfolding has orthonormal rows; *up* (also outer), its
  `ni × (ri·r(i+1))` matricization has orthonormal rows (`up_orthogonalize_tt_cores`). Tucker core: *down*,
  `B Bᵀ = I` (`down_orthogonalize_tucker_cores`). The adjective names the side of the cut edge whose indices the
  orthonormality sums over. **Clash:** the frame pages name these by store instead (`O` "down-orthogonal", `U` "up
  orthogonal"); a rename of the stores is being assessed.
  [Four rank stores](frame_variations.md#four-rank-stores--and-why-excess-rank-is-inert)
~~~

AFTER:

~~~md
- **orthogonal core** (left-, right-, down-, up-orthogonal; also: outer-orthogonal). A core with orthonormal rows or
  columns in one reshape (API "orthogonal" means orthonormal). TT core: *left*, its `(ri·ni) × r(i+1)` unfolding has
  orthonormal columns; *right*, its `ri × (ni·r(i+1))` unfolding has orthonormal rows; *up* (also outer), its
  `ni × (ri·r(i+1))` matricization has orthonormal rows (`up_orthogonalize_tt_cores`). Tucker core: *down*,
  `B Bᵀ = I` (`down_orthogonalize_tucker_cores`). The adjective names the side of the cut edge whose indices the
  orthonormality sums over. The frame stores are named the same way: an up core is up-orthogonal, a down core
  down-orthogonal (2026.5.0; before, the store names were the other way round).
  [Four rank stores](frame_variations.md#four-rank-stores--and-why-excess-rank-is-inert)
~~~

#### padded widths

BEFORE:

~~~md
- **padded widths** (`n`, `r`, `N`; frames `nD`, `nU`, `rL`, `rR`; also: nominal rank, a level's widths, padded `N`).
  The common sizes a uniform object's cores are padded to: Tucker `n`, TT `r`, mode `N ≥ max Ni`.
  [The layout](uniform_supercore_layout.md#the-layout)
~~~

AFTER:

~~~md
- **padded widths** (`n`, `r`, `N`; frames `nD`, `nU`, `rL`, `rR`; also: nominal rank, a level's widths, padded `N`).
  The common sizes a uniform object's cores are padded to: Tucker `n`, TT `r`, mode `N ≥ max Ni`. A frame's stores
  each have their own: `nD` the down store's (the Tucker factor's rows), `nU` the up store's (the up core's mode leg),
  `rL`, `rR` the left and right stores' bonds. `nD` and `nU` swapped meaning in 2026.5.0.
  [The layout](uniform_supercore_layout.md#the-layout)
~~~

#### Tucker core

BEFORE:

~~~md
- **Tucker core** (`tucker_cores`; `Bi`, `Ui` in a frame; also: Tucker factor, Tucker basis, leaf). The `d` matrices
  of shape `stack_shape + (ni, Ni)` mapping mode `i` to its Tucker index. [User guide](user_guide.rst)
~~~

AFTER:

~~~md
- **Tucker core** (`tucker_cores`; `Bi`, `Di` in a frame; also: Tucker factor, Tucker basis, leaf). The `d` matrices
  of shape `stack_shape + (ni, Ni)` mapping mode `i` to its Tucker index. [User guide](user_guide.rst)
~~~

#### alignment pass

BEFORE:

~~~md
- **alignment pass** (`T3Tangent.align_padded`, `tv_align_padded_frame`; also: pass, alignment step). One update of the
  stores named in `update` (letters `U`, `O`, `L`, `R`); the others are transported. Returns the new frame and its
  movement. [The alignment](aligned_padded_frame.md#the-alignment)
~~~

AFTER:

~~~md
- **alignment pass** (`T3Tangent.align_padded`, `tv_align_padded_frame`; also: pass, alignment step). One update of the
  stores named in `update`, a tuple of the store letters `D`, `U`, `L`, `R` in any order; the others are transported.
  (2026.4.0: the letters `U`, `O`, `L`, `R`, with `U` the Tucker store, or a string of them.) Returns the new frame and
  its movement. [The alignment](aligned_padded_frame.md#the-alignment)
~~~

#### base-point tangent

BEFORE:

~~~md
- **base-point tangent** (`v_X`; `fv_base_point_tangent`). The point written as a gauged tangent at its own frame: all
  variations zero except the last TT variation, so `‖X‖² = ‖P_last‖²`.
  [`d` cores, `d+1` bonds](frame_variations.md#d-cores-d1-bonds--and-the-extra-ld-r0)
~~~

AFTER:

~~~md
- **base-point tangent** (`v_X`; `fv_base_point_tangent`). The point written as a gauged tangent at its own frame: all
  variations zero except the last TT variation, so `‖X‖² = ‖L_last‖²`.
  [`d` cores, `d+1` bonds](frame_variations.md#d-cores-d1-bonds--and-the-extra-ld-r0)
~~~

#### corewise frame

BEFORE:

~~~md
- **corewise frame** (`(U, G, G, G)`; `t3_corewise_frame`). The non-orthonormal frame whose down, left and right cores
  are the TT cores, so a tangent is a raw core perturbation. `COREWISE` uses it.
  [Corewise transpose](transposes.md#corewise--xapply_corewise_transposec-ww-sum_over_probes--instance)
~~~

AFTER:

~~~md
- **corewise frame** (`(B, G, G, G)`; `t3_corewise_frame`; also: (2026.4.0: `(U, G, G, G)`)). The non-orthonormal frame
  whose up, left and right cores are the TT cores `G` and whose down cores are the Tucker cores `B`, so a tangent is a
  raw core perturbation. `COREWISE` uses it.
  [Corewise transpose](transposes.md#corewise--xapply_corewise_transposec-ww-sum_over_probes--instance)
~~~

#### frame cores

BEFORE:

~~~md
- **frame cores** (up `Ui`, down `Oi`, left `Li`, right `Ri`; `down_tucker_cores`, `up_tt_cores`, `left_tt_cores`,
  `right_tt_cores`; also: `P`/`Q` for left/right, `D` or "outer" for down). Up: orthonormal Tucker cores. Down:
  TT-shaped `(rL, nU, rR)`, the TT core of the Tucker-variation terms (`Vi` varies in the complement of `Ui`'s rows).
  Left/right: the orthogonal TT environments. [Frame-variations](frame_variations.md#a-tangent-is-a-sum-of-d-terms)
~~~

AFTER:

~~~md
- **frame cores** (down `Di`, up `Ui`, left `Li`, right `Ri`; `down_tucker_cores`, `up_tt_cores`, `left_tt_cores`,
  `right_tt_cores`; also: "outer" for up, T4S's letters (see **T4S**),
  (2026.4.0, names reversed: up `Ui` (Tucker), down `Oi` or `Di`, `up_tucker_cores`, `down_tt_cores`)).
  Down: the orthonormal Tucker factors `(nD, N)`. Up: TT-shaped `(rL, nU, rR)`, the TT core of the Tucker-variation
  terms (`Vi` varies in the complement of `Di`'s rows). Left/right: the orthogonal TT environments. Each store is
  named for the side of its edge its cores occupy. [Frame-variations](frame_variations.md#a-tangent-is-a-sum-of-d-terms)
~~~

#### gauge conditions

BEFORE:

~~~md
- **gauge conditions** (`is_gauged`, `gauge_residual`; also: gauged). `Ui Viᵀ = 0` for every `i` (code layout), and
  each `Li` left-orthogonal against `Hi` for `i < d-1` (T4S eqs. 47–48). They make the variations coordinates.
  [Gauge conditions](frame_variations.md#gauge-conditions-make-the-variations-coordinates)
~~~

AFTER:

~~~md
- **gauge conditions** (`is_gauged`, `gauge_residual`; also: gauged). `Di Viᵀ = 0` for every `i` (code layout), and
  each `Li` left-orthogonal against `Hi` for `i < d-1` (T4S eqs. 47–48). They make the variations coordinates.
  [Gauge conditions](frame_variations.md#gauge-conditions-make-the-variations-coordinates)
~~~

#### gauge projection

BEFORE:

~~~md
- **gauge projection** (`Π`; also: gauge projector). Map raw variations to gauged ones; two kinds. *Orthogonal*
  (`MANIFOLD.project`, `tv_orthogonal_gauge_projection`): the coordinate-orthogonal projection onto the gauged
  subspace; it keeps the gauged part and changes the represented tangent, and applied after a bare transpose it gives
  the Riemannian gradient. *Oblique* (`MANIFOLD.project_oblique`, `tv_oblique_gauge_projection`; also: oblique
  projection): not orthogonal; it enforces the gauge while preserving the represented tangent, compensating through
  the down and right cores. (An ambient tensor goes through **ambient projection** instead.)
  [Notes](entries_apply_probe.md#4-the-riemannian--tangent-versions-and-their-transposes)
~~~

AFTER:

~~~md
- **gauge projection** (`Π`; also: gauge projector). Map raw variations to gauged ones; two kinds. *Orthogonal*
  (`MANIFOLD.project`, `tv_orthogonal_gauge_projection`): the coordinate-orthogonal projection onto the gauged
  subspace; it keeps the gauged part and changes the represented tangent, and applied after a bare transpose it gives
  the Riemannian gradient. *Oblique* (`MANIFOLD.project_oblique`, `tv_oblique_gauge_projection`; also: oblique
  projection): not orthogonal; it enforces the gauge while preserving the represented tangent, compensating through
  the up and right cores. (An ambient tensor goes through **ambient projection** instead.)
  [Notes](entries_apply_probe.md#4-the-riemannian--tangent-versions-and-their-transposes)
~~~

#### store

BEFORE:

~~~md
- **store** (`down_ranks`, `up_ranks`, `left_ranks`, `right_ranks`; also: rank store, orthogonal-representation
  ranks). One of the frame's four core families, each orthonormal in its own matricization or unfolding. Its ranks are
  set by the orthogonalization sweep (up, left, right over the left chain, then down); the differences are the slack.
  [Four rank stores](frame_variations.md#four-rank-stores--and-why-excess-rank-is-inert)
~~~

AFTER:

~~~md
- **store** (`down_ranks`, `up_ranks`, `left_ranks`, `right_ranks`; also: rank store, orthogonal-representation
  ranks, (2026.4.0, names reversed: `up_ranks` (Tucker), `down_ranks` (TT-shaped))). One of the frame's four core
  families, named for the side of its edge its cores occupy, with the TT chain drawn on top: left and right of every
  TT bond, up (the TT-shaped cores) above and down (the Tucker factors) below every Tucker edge. Each is orthonormal
  in its own matricization or unfolding, over the indices on its side. Its ranks are set by the orthogonalization
  sweep (down, left, right over the left chain, then up); the differences are the slack.
  [Four rank stores](frame_variations.md#four-rank-stores--and-why-excess-rank-is-inert)
~~~

#### tangent metric

BEFORE:

~~~md
- **tangent metric** (`T3FrameWeights`, `UT3FrameWeights`). Diagonal weights on a tangent's coordinates (`up`, `down`,
  `left`, `right`, each length `d`), absorbed into the variations but batched with the frame stack `C`.
  [`T3FrameWeights`](weighting.md#t3frameweights--a-metric-on-a-tangent)
~~~

AFTER:

~~~md
- **tangent metric** (`T3FrameWeights`, `UT3FrameWeights`). Diagonal weights on a tangent's coordinates (`down`, `up`,
  `left`, `right`, each length `d`), absorbed into the variations but batched with the frame stack `C`.
  [`T3FrameWeights`](weighting.md#t3frameweights--a-metric-on-a-tangent)
~~~

#### two-colour pass, variation masks, variations

BEFORE:

~~~md
- **two-colour pass** (`'UL'`, `'RO'`; `alternate=True`). A pass updating the up and left stores, then one updating
  right and down, the other colour transported; avoids the period-2 cycle of simultaneous updates.
  [The schedule](aligned_padded_frame.md#the-schedule)
- **variation masks** (`ufv_variation_masks`; also: gauge-shifted masks). A uniform tangent's variation masks, derived
  from the frame's as `(up, down, left[:-1], right[1:])`; tangent `+`/`−`/scale leave them unchanged.
  [Tangent masks](uniform_masks_vs_ranks.md#tangent-variation-masks-a-different-algebra)
- **variations** (`T3Variations`, `UT3Variations`; `tucker_variations` `Vi`, `tt_variations` `Hi`; also: variation
  cores, `δU`/`δG`). The cores that fill the frame's holes and encode a tangent's direction. Gauged, they are
  orthonormal coordinates. [Frame-variations](frame_variations.md)
~~~

AFTER:

~~~md
- **two-colour pass** (`('D', 'L')`, `('U', 'R')`; `alternate=True`). A pass updating the down and left stores, then
  one updating the up and right stores, the other colour transported; avoids the period-2 cycle of simultaneous updates.
  [The schedule](aligned_padded_frame.md#the-schedule)
- **variation masks** (`ufv_variation_masks`; also: gauge-shifted masks). A uniform tangent's variation masks, derived
  from the frame's as `(down, up, left[:-1], right[1:])`; tangent `+`/`−`/scale leave them unchanged.
  [Tangent masks](uniform_masks_vs_ranks.md#tangent-variation-masks-a-different-algebra)
- **variations** (`T3Variations`, `UT3Variations`; `tucker_variations` `Vi`, `tt_variations` `Hi`; also: variation
  cores, T4S's letters (see **T4S**)). The cores that fill the frame's holes and encode a tangent's direction. Gauged,
  they are orthonormal coordinates. [Frame-variations](frame_variations.md)
~~~

#### `.data`

BEFORE:

~~~md
- **`.data`** (also: raw tuple). The raw tuple a frontend object wraps: `(tucker_cores, tt_cores)`; a frame's
  `(up, down, left, right)`; `(tucker_variations, tt_variations)`; uniform `(supercores…, shape, masks)`.
  [Naming conventions](naming_conventions.md#parameter-names-carry-representation-too)
~~~

AFTER:

~~~md
- **`.data`** (also: raw tuple). The raw tuple a frontend object wraps: `(tucker_cores, tt_cores)`; a frame's
  `(down, up, left, right)`; `(tucker_variations, tt_variations)`; uniform `(supercores…, shape, masks)`.
  [Naming conventions](naming_conventions.md#parameter-names-carry-representation-too)
~~~

#### T4S

BEFORE:

~~~md
- **T4S**. The paper Alger, Christierson, Chen & Ghattas (2026), "Tucker Tensor Train Taylor Series"
  (arXiv:2603.21141); citations use the arXiv numbering. [arXiv](https://arxiv.org/abs/2603.21141)
~~~

AFTER:

~~~md
- **T4S**. The paper Alger, Christierson, Chen & Ghattas (2026), "Tucker Tensor Train Taylor Series"
  (arXiv:2603.21141); citations use the arXiv numbering. T4S writes U, O, P, Q for our D, U, L, R; its δU is our V
  (`tucker_variations`), its δG our H (`tt_variations`). [arXiv](https://arxiv.org/abs/2603.21141)
~~~

#### Uppercase `C`, `W`, `K` are reserved for

BEFORE:

~~~md
Uppercase `C`, `W`, `K` are reserved for the batch blocks in shape comments and contraction strings; lowercase
letters are single axes. The letters below also carry other meanings on some pages.
~~~

AFTER:

~~~md
Uppercase `C`, `W`, `K` are reserved for the batch blocks in shape comments and contraction strings; lowercase
letters are single axes. A frame's stores are written `D`, `U`, `L`, `R` (down, up, left, right), with widths `nD`
(the down store's: the Tucker factor's rows), `nU` (the up store's: the up core's mode leg), `rL` and `rR`; a T3's own
cores are `B` (Tucker) and `G` (TT). Before 2026.5.0 the words up and down, and with them `nU` and `nD`, named the
other store ([Four rank stores](frame_variations.md#four-rank-stores--and-why-excess-rank-is-inert)). The letters
below also carry other meanings on some pages.
~~~

#### Symbols row `P`, Symbols row `L`

BEFORE:

~~~md
| `P` | the left frame core in the T4S paper's notation (`batching_and_stacking.md` §8, `entries_apply_probe.md` §8, `LocalModel.frame` as `(U, O, P, Q)`), where the user guide writes `L`; the derivative direction (`pp`); a completion `P⁺` (`aligned_padded_frame.md` movement, the padded-frame note); the tangent projector `P_X` / `P_F`; the core parameter space (`CorewiseGeometry`) |
| `L` | the left frame cores `Li` (user guide, `frame_variations.md`); the lift `L` (§4.12, curvature pages) |
~~~

AFTER:

~~~md
| `P` | the derivative direction (`pp`); a completion `P⁺` (`aligned_padded_frame.md` movement, the padded-frame note); the tangent projector `P_X` / `P_F`; the core parameter space (`CorewiseGeometry`); T4S's left frame core, our `L` (see **T4S**; the paper ↔ code maps, `batching_and_stacking.md` §8 and `entries_apply_probe.md` §8, keep the paper's letters) |
| `L` | the left frame cores `Li` and the left store's letter (user guide, `frame_variations.md`, `update`); the lift `L` (§4.12, curvature pages) |
~~~

#### Symbols row `D`, Symbols row `U`

BEFORE:

~~~md
| `O`, `D` | the down frame core: `O` in the user guide, `frame_variations.md`, the math notes and the `update` letters; `D` in the `T3Frame` docstring; "outer" in the minimal-rank list |
~~~

AFTER:

~~~md
| `D` | the down frame core `Di`, a Tucker factor, and the down store's letter (frame pages, diagrams, equations, `update`). Before 2026.5.0 `D` also stood for the derivative (now the upright `d`), the forward jet shift (now `sh`), the padded-frame messages (now `Ψ`) and, in the `T3Frame` docstring, the TT-shaped core (now `U`) |
| `U` | the up frame core `Ui`, TT-shaped, and the up store's letter (frame pages, diagrams, equations, `update`); the SVD factor in `UΣVᵀ`. Before 2026.5.0 `U` was the Tucker frame core and its `update` letter (now `D`), the reverse jet shift (now `shᵀ`) and the padded-frame note's edge frame (now `F_{x→t}`) |
~~~

#### Symbols row `Ψ`

BEFORE:

~~~md
(none: new)
~~~

AFTER:

~~~md
| `Ψ` | the padded-frame messages and products (`Ψ_{a→b}` in the padded-frame note, `Ψ` in `contributor/padded_frame_internals.md`, `Psi` in code), beside the note's edge frame `F_{x→t}`; before 2026.5.0 written `D` |
~~~

#### Symbols row `d`, Symbols row `sh`

BEFORE:

~~~md
| `d` | the number of modes; the leading supercore axis (uniform layer); a leading `d` in a local name's suffix is the supercore or derivative axis |
~~~

AFTER:

~~~md
| `d` | the number of modes; the leading supercore axis (uniform layer); a leading `d` in a local name's suffix is the supercore or derivative axis; upright, the differential (`dΦ`, `d²Φ`: `curvature_internals.md`, the exact-Hessian and symmetric-probe notes), told apart from the mode count by context. Before 2026.5.0 the differential was `D` |
| `sh` | the jet shift: forward `(sh x)^(t) = t x^(t-1)` and its adjoint, reverse `(shᵀ x)^(s) = (s+1) x^(s+1)` (`backend/sampling_derivatives.py`, whose functions `_shift_down`, `_shift_up` keep their names; the exact-Hessian note's `\mathsf{sh}`); before 2026.5.0 written `D` and `U` |
~~~

### (2) Work-list rows re-expressed (verbatim before -> after; line numbers before / after)

#### wl:14-14 (before) / wl:14-14 (after)

BEFORE:

~~~md
marked **HELD** wait for Nick's decision on their item.
~~~

AFTER:

~~~md
marked **HELD** wait for Nick's decision on their item (none remain: N17, the last, was resolved on 2026-11-10).
~~~

#### wl:30-31 (before) / wl:30-32 (after)

BEFORE:

~~~md
- N17: once Nick decides (the adjectives stay; the store names may change), release or replace the HELD rows and
  update the glossary's **orthogonal core** and **frame cores** entries.
~~~

AFTER:

~~~md
- N17 (resolved 2026-11-10): the adjectives stayed and the store names flipped (the 2026.5.0 store rename,
  `dev/store_rename_plan_2026-11-10.md`). The HELD rows below are released in the new store words; the glossary's
  **orthogonal core** and **frame cores** entries were rewritten by the rename (R5).
~~~

#### wl:40-40 (before) / wl:41-41 (after)

BEFORE:

~~~md
   `tucker_cores` / `up_tucker_cores` / `tucker_variations`. Against it: the docs use "Tucker factor" more (25 vs 14 in
~~~

AFTER:

~~~md
   `tucker_cores` / `down_tucker_cores` / `tucker_variations`. Against it: the docs use "Tucker factor" more (25 vs 14 in
~~~

#### wl:81-81 (before) / wl:82-82 (after)

BEFORE:

~~~md
| N4 | NAMING | the down store and its cores are named **down, `O`** (not `D`, not "outer" as a name). The orthogonality *adjective* is N17, not N4 | `align_padded(update=...)` letters `'U'`, `'O'`, `'L'`, `'R'`; `T3Frame.data = (up, down, left, right)`. Not "outer TT bonds" (`tucker_tensor_train.py:962`), which are the boundary bonds and stay |
~~~

AFTER:

~~~md
| N4 | NAMING (**RESOLVED** 2026-11-10: the store rename, 2026.5.0) | a store is named for the side of its edge its cores occupy: the TT-shaped store and its cores are **up, `U`**, the Tucker store and its cores **down, `D`**; not "outer" as a name (T4S's word; T4S's `O` is our `U`, its `U` our `D`). The orthogonality *adjective* is N17, not N4 | `align_padded(update=...)` takes a tuple of the letters `'D'`, `'U'`, `'L'`, `'R'` (a `str` raises); `T3Frame.data = (down, up, left, right)`; widths `nD`, `nU`, `rL`, `rR`. Not "outer TT bonds" (`tucker_tensor_train.py:962`), which are the boundary bonds and stay |
~~~

#### wl:93-93 (before) / wl:94-94 (after)

BEFORE:

~~~md
| N17 | NAMING (**HELD**: Nick is deciding) | the orthogonality adjectives follow the **method convention** (rule a): a Tucker core with `B Bᵀ = I` is *down-orthogonal* (`down_orthogonalize_tucker_cores`), a TT core orthonormal in its `ni × (ri·r(i+1))` matricization is *up-orthogonal* (`up_orthogonalize_tt_cores`; alias outer-orthogonal). Nick's reasoning: left cores are left-orthogonal and right cores right-orthogonal, and the `O` cores' orthonormal index group lies upward of the Tucker–TT edge, so the method adjectives are right and the **store names** "up"/"down" are the problem; a store rename is being assessed. The HELD rows below assume the adjectives stay and the frame pages change; a store rename would supersede them | `fv_conversions.py:184` binds `up_orthogonalize_tucker_cores = t3_down_orthogonalize_tucker_cores`; `ranks.py:644` "Tucker down-orthogonalization: up_i = ..."; `tucker_tensor_train.py:3159` (`B Bᵀ = I`, "down orthogonal") vs `:3207` (TT, "up orthogonal"); `up_svd_tt_core` (`:2978`) and `:3108` contradict even the method convention |
~~~

AFTER:

~~~md
| N17 | NAMING (**RESOLVED** 2026-11-10: the adjectives stay, the stores flip) | the orthogonality adjectives follow the **method convention** (rule a): a Tucker core with `B Bᵀ = I` is *down-orthogonal* (`down_orthogonalize_tucker_cores`), a TT core orthonormal in its `ni × (ri·r(i+1))` matricization is *up-orthogonal* (`up_orthogonalize_tt_cores`; alias outer-orthogonal). Nick's reasoning: left cores are left-orthogonal and right cores right-orthogonal, and the TT-shaped cores' (then `O`, now `U`) orthonormal index group lies upward of the Tucker–TT edge, so the method adjectives are right and the **store names** "up"/"down" were the problem. The 2026.5.0 store rename swapped the store names to agree. The former HELD rows below are released in the new store words; rows the rename itself applied are marked done | At 2026.4.0: `fv_conversions.py:184` (the rename's one-way A1 aliases) binds `up_orthogonalize_tucker_cores = t3_down_orthogonalize_tucker_cores`; `ranks.py:644` "Tucker down-orthogonalization: up_i = ..."; `tucker_tensor_train.py:3159` (`B Bᵀ = I`, "down orthogonal") vs `:3207` (TT, "up orthogonal"); `up_svd_tt_core` (`:2978`) and `:3108` (the rename's §4.3 rows) contradict even the method convention |
~~~

#### wl:103-103 (before) / wl:104-104 (after)

BEFORE:

~~~md
| E10 | ERROR | the Tucker gauge condition is written `Uᵢᵀ Vᵢ = 0` | `Ui`, `Vi` are stored `(n, N)`; `tv_orthogonal_gauge_projection` states `U_i V_i^T = 0` |
~~~

AFTER:

~~~md
| E10 | ERROR | the Tucker gauge condition is written `Uᵢᵀ Vᵢ = 0` (2026.4.0's letter for the frame's Tucker core, now `Dᵢ`; the one site, `frame_variations.md:57`, was fixed by the store rename) | `Di`, `Vi` are stored `(nD, N)`, `(nU, N)`; `tv_orthogonal_gauge_projection` states `D_i V_i^T = 0` |
~~~

#### wl:110-110 (before) / wl:111-111 (after)

BEFORE:

~~~md
| E17 | ERROR | docstring typos in index ranges and shapes | `tucker_ranks = (n0, …, n(d-1))`; modes, cores and diagrams are 0-based; `fv_variation_shapes` gives a TT variation the shape `(rL_i, nU_i, rR_(i+1))` (it reads `right[i].shape[-1]`), and the left/right frame cores carry `nU_i` |
~~~

AFTER:

~~~md
| E17 | ERROR | docstring typos in index ranges and shapes | `tucker_ranks = (n0, …, n(d-1))`; modes, cores and diagrams are 0-based; `fv_variation_shapes` gives a TT variation the shape `(rL_i, nD_i, rR_(i+1))` (it reads `right[i].shape[-1]`), and the left/right frame cores carry `nD_i` |
~~~

#### wl:123-124 (before) / wl:124-125 (after)

BEFORE:

~~~md
the columns, and are called with the middle slot on the down cores `O` (`P_O`, the down edge's `g_new`, `g_old`,
`Abar`), which makes those matricizations; and the test
~~~

AFTER:

~~~md
the columns, and are called with the middle slot on the up cores `U` (`U_P`, and `_up_edge`'s `g_new`, `g_old`, `Abar`
on the up store's edge, `c_i → l_i`), which makes those matricizations; and the test
~~~

#### wl:197-198 (before) / wl:198-199 (after)

BEFORE:

~~~md
| 57 | § Gauge conditions make the variations *coordinates* | E10 | "`Uᵢᵀ Vᵢ = 0`" | "`Uᵢ Vᵢᵀ = 0`" |
| 71 | § Four rank stores — and why excess rank is inert | N4 | "down/outer" | "down" |
~~~

AFTER:

~~~md
| 57 | § Gauge conditions make the variations *coordinates* | E10 | done by the store rename (R5): "`Dᵢ Vᵢᵀ = 0`" | — |
| 71 | § Four rank stores — and why excess rank is inert | N4 | done by the store rename (R5): "`up_ranks` (the mode leg of each up core `Uᵢ`)" | — |
~~~

#### wl:300-301 (before) / wl:301-302 (after)

BEFORE:

~~~md
| 97 | § Minimal rank conditions | M/U | "Outer TT core unfoldings are full rank" | "... TT core matricizations are full rank" (the adjective is N17, HELD) |
| 97 | § Minimal rank conditions | N17 | "Outer TT core unfoldings are full rank" | HELD (N17): "Up (outer) TT core unfoldings are full rank" |
~~~

AFTER:

~~~md
| 97 | § Minimal rank conditions | M/U | done by the store rename (R5), with the N17 row below: "TT core up-matricizations are full rank" | — |
| 97 | § Minimal rank conditions | N17 | done by the store rename (R5): "TT core up-matricizations are full rank" | — |
~~~

#### wl:305-305 (before) / wl:306-306 (after)

BEFORE:

~~~md
| 142 | § T3 manifold | N17 | "down_tt_cores = (O0,...,O(d-1)), down-orthogonal" | HELD (N17): "..., up-orthogonal" |
~~~

AFTER:

~~~md
| 142 | § T3 manifold | N17 | done by the store rename (R5): "up_tt_cores = (U0,...,U(d-1)), up-orthogonal" | — |
~~~

#### wl:319-319 (before) / wl:320-320 (after)

BEFORE:

~~~md
| 28 | § The formula, and where each piece lives | E6 | ``` \| frame factors `T_i = ⟨L_i, H_i⟩`, `S_iᵀ = ⟨O_i, H_i⟩`, centers `H_i = L_i T_i` \| `fv_operations.fv_center_factors` \| `T_i` by zipper on demand, never stored on the frame (I-1); `σ(S_i)` is the mode-`i` Tucker spectrum, `σ(T_i)` the bond spectrum; `‖X‖² = Σσ(S_0)²` \| ``` | ``` \| frame factors `T_i = ⟨L_i, H_i⟩`, `S_iᵀ = ⟨O_i, H_i⟩`, centers `H_i = L_i T_i` \| `fv_operations.fv_center_factors` \| `T_i` by zipper on demand, never stored on the frame; `σ(S_i)` is the mode-`i` Tucker spectrum, `σ(T_i)` the bond spectrum; `‖X‖² = Σσ(S_0)²` \| ``` |
~~~

AFTER:

~~~md
| 28 | § The formula, and where each piece lives | E6 | ``` \| frame factors `T_i = ⟨L_i, H_i⟩`, `S_iᵀ = ⟨U_i, H_i⟩`, centers `H_i = L_i T_i` \| `fv_operations.fv_center_factors` \| `T_i` by zipper on demand, never stored on the frame (I-1); `σ(S_i)` is the mode-`i` Tucker spectrum, `σ(T_i)` the bond spectrum; `‖X‖² = Σσ(S_0)²` \| ``` | ``` \| frame factors `T_i = ⟨L_i, H_i⟩`, `S_iᵀ = ⟨U_i, H_i⟩`, centers `H_i = L_i T_i` \| `fv_operations.fv_center_factors` \| `T_i` by zipper on demand, never stored on the frame; `σ(S_i)` is the mode-`i` Tucker spectrum, `σ(T_i)` the bond spectrum; `‖X‖² = Σσ(S_0)²` \| ``` |
~~~

#### wl:576-576 (before) / wl:577-577 (after)

BEFORE:

~~~md
| 98 | `fv_variation_shapes` | E17 | "``(rL_i, nU_i, rR_i)`` per TT variation" | "``(rL_i, nU_i, rR_(i+1))`` per TT variation" (the next lines already explain the shift) |
~~~

AFTER:

~~~md
| 98 | `fv_variation_shapes` | E17 | "``(rL_i, nD_i, rR_i)`` per TT variation" | "``(rL_i, nD_i, rR_(i+1))`` per TT variation" (the next lines already explain the shift) |
~~~

#### wl:580-580 (before) / wl:581-581 (after)

BEFORE:

~~~md
| 640 | `fv_absorb_weights` | E6 | ``` Approach-1 / metric-on-variations): ``down`` -> V's ``nD`` leg; ``up``/``left``/``right`` -> H's ``` | ``` i.e. on the variations): ``down`` -> V's ``nD`` leg; ``up``/``left``/``right`` -> H's ``` |
~~~

AFTER:

~~~md
| 640 | `fv_absorb_weights` | E6 | ``` Approach-1 / metric-on-variations): ``up`` -> V's ``nU`` leg; ``down``/``left``/``right`` -> H's ``` | ``` i.e. on the variations): ``up`` -> V's ``nU`` leg; ``down``/``left``/``right`` -> H's ``` |
~~~

#### wl:629-629 (before) / wl:630-630 (after)

BEFORE:

~~~md
| 148 | `_seed_rows` | M/U | "# the seed length: the mode size (up) or the real rows of the core's unfolding" | "... or the real rows of the core's matricization (down) or unfolding (left, right)" |
~~~

AFTER:

~~~md
| 148 | `_seed_rows` | M/U | "# the seed length: the mode size (down) or the real rows of the core's unfolding" | "... or the real rows of the core's matricization (up) or unfolding (left, right)" |
~~~

#### wl:645-646 (before) / wl:646-647 (after)

BEFORE:

~~~md
| 646 | `tv_probe` | N17 | "up_tucker_supercore ... up orthogonal elements" | HELD (N17): "down-orthogonal elements" |
| 647 | `tv_probe` | N17 | "down_tt_supercore ... down orthogonal elements" | HELD (N17): "up-orthogonal elements" |
~~~

AFTER:

~~~md
| 646 | `tv_probe` | N17 | done by the store rename (R4b): "down_tucker_supercore ... down orthogonal elements" | — |
| 647 | `tv_probe` | N17 | done by the store rename (R4b): "up_tt_supercore ... up orthogonal elements" | — |
~~~

#### wl:650-651 (before) / wl:651-652 (after)

BEFORE:

~~~md
| 1221 | `tv_probe_transpose` | N17 | "up_tucker_supercore ... up orthogonal elements" | HELD (N17): "down-orthogonal elements" |
| 1222 | `tv_probe_transpose` | N17 | "down_tt_supercore ... down orthogonal elements" | HELD (N17): "up-orthogonal elements" |
~~~

AFTER:

~~~md
| 1221 | `tv_probe_transpose` | N17 | done by the store rename (R4b): "down_tucker_supercore ... down orthogonal elements" | — |
| 1222 | `tv_probe_transpose` | N17 | done by the store rename (R4b): "up_tt_supercore ... up orthogonal elements" | — |
~~~

#### wl:659-659 (before) / wl:660-660 (after)

BEFORE:

~~~md
| 651 | `compute_orthogonal_representation_ranks` | E6 | ``` non-minimal TT bond they do not, and the uniform frame's left/down masks were too small (review S1). ``` | ``` non-minimal TT bond they do not, and the uniform frame's left/down masks were too small. ``` |
~~~

AFTER:

~~~md
| 651 | `compute_orthogonal_representation_ranks` | E6 | ``` non-minimal TT bond they do not, and the uniform frame's left/up masks were too small (review S1). ``` | ``` non-minimal TT bond they do not, and the uniform frame's left/up masks were too small. ``` |
~~~

#### wl:736-737 (before) / wl:737-738 (after)

BEFORE:

~~~md
| 106 | `_up_orthogonalize_tt_cores_step` | N17 | "# Oaxb. outer-orthogonal tt core" | HELD (N17): keep, or "up-orthogonal tt core" |
| 147 | `_down_orthogonalize_tucker_cores_step` | N17 | "# new_Uxo.  up-orthogonal tucker core" | HELD (N17): "down-orthogonal tucker core" |
~~~

AFTER:

~~~md
| 106 | `_up_orthogonalize_tt_cores_step` | N17 | done by the store rename (R4a): "# Uaxb. up-orthogonal tt core" | — |
| 147 | `_down_orthogonalize_tucker_cores_step` | N17 | done by the store rename (R4a): "# new_Bxo.  down-orthogonal tucker core" | — |
~~~

#### wl:752-753 (before) / wl:753-754 (after)

BEFORE:

~~~md
| 164 | `tv_oblique_gauge_projection` | N4 | "down/outer" | "down" |
| 181 | `tv_oblique_gauge_projection` | N4 | "down/outer" | "down" |
~~~

AFTER:

~~~md
| 164 | `tv_oblique_gauge_projection` | N4 | "up/outer" (the rename's tier-E row swapped the word only) | "up" |
| 181 | `tv_oblique_gauge_projection` | N4 | "up/outer" (the rename's tier-E row swapped the word only) | "up" |
~~~

#### wl:771-772 (before) / wl:772-773 (after)

BEFORE:

~~~md
| 277 | `ufv_frame_orthogonality_residual` | N4 | "down/outer" | "down" |
| 294 | `dev` | N4 | "down/outer" | "down" |
~~~

AFTER:

~~~md
| 277 | `ufv_frame_orthogonality_residual` | N4 | "up/outer ``U``" (the rename swapped the word and the letter) | "up ``U``" |
| 294 | `dev` | N4 | "# up/outer: mode orthonormal over bonds" (the rename swapped the word only) | "# up: mode orthonormal over bonds" |
~~~

#### wl:826-826 (before) / wl:827-827 (after)

BEFORE:

~~~md
| 189 | `up_orthogonalize_tt_supercores` | N17 | "new_tt = down-orthogonal O" | HELD (N17): "new_tt = up-orthogonal O" |
~~~

AFTER:

~~~md
| 189 | `up_orthogonalize_tt_supercores` | N17 | done by the store rename (R4a): "new_tt = up-orthogonal U" | — |
~~~

#### wl:870-874 (before) / wl:871-875 (after)

BEFORE:

~~~md
| 63 | `T3Frame` | N4 | diagram cores `D0`, `D1`, `D2` | `O0`, `O1`, `O2` |
| 72 | `T3Frame` | N4 | `down_tt_cores = (D0, ..., D(d-1))` | `down_tt_cores = (O0, ..., O(d-1))` |
| 73 | `T3Frame` | E17 | "left_tt_cores = (L0, ..., L(d-1)), elm_shape=(rLi, ni, rL(i+1))" | "elm_shape=(rLi, nUi, rL(i+1))" |
| 74 | `T3Frame` | E17 | "right_tt_cores = (R0, ..., R(d-1)), elm_shape=(rRi, ni, rR(i+1))" | "elm_shape=(rRi, nUi, rR(i+1))" |
| 78 | `T3Frame` | E17 | "tt_variations = (H0, ..., H(d-1)), elm_shape=(rLi, nUi, rRi)" | "elm_shape=(rLi, nUi, rR(i+1))" (`fv_variation_shapes` reads `right[i].shape[-1]`) |
~~~

AFTER:

~~~md
| 63 | `T3Frame` | N4 | done by the store rename (R4a, §4A block): the TT-shaped diagram cores are `U0`, `U1`, `U2`, the Tucker ones `D0`, `D1`, `D2` | — |
| 72 | `T3Frame` | N4 | done by the store rename (R4a): `up_tt_cores = (U0, ..., U(d-1))` | — |
| 73 | `T3Frame` | E17 | "left_tt_cores = (L0, ..., L(d-1)), elm_shape=(rLi, ni, rL(i+1))" | "elm_shape=(rLi, nDi, rL(i+1))" |
| 74 | `T3Frame` | E17 | "right_tt_cores = (R0, ..., R(d-1)), elm_shape=(rRi, ni, rR(i+1))" | "elm_shape=(rRi, nDi, rR(i+1))" |
| 78 | `T3Frame` | E17 | "tt_variations = (H0, ..., H(d-1)), elm_shape=(rLi, nDi, rRi)" | "elm_shape=(rLi, nDi, rR(i+1))" (`fv_variation_shapes` reads `right[i].shape[-1]`) |
~~~

#### wl:876-879 (before) / wl:877-880 (after)

BEFORE:

~~~md
| 93 | `T3Frame` | E17 | diagram label "Ud" | "U(d-1)" |
| 101 | `T3Frame` | N4 | rank diagram core "D1" (`T3Frame`; the next line already says `nO1`) | "O1" |
| 105 | `T3Frame` | E17 | diagram label "Ud" | "U(d-1)" |
| 118 | `T3Frame` | N17 | "down_tt_cores = (O0,...,O(d-1)), outer-orthogonal  O_aib O_ajb = delta_ij" | HELD (N17): keep, or "up-orthogonal" |
~~~

AFTER:

~~~md
| 93 | `T3Frame` | E17 | diagram label "Dd" (the store rename's §4A block relabelled the Tucker core) | "D(d-1)" |
| 101 | `T3Frame` | N4 | done by the store rename (R4a, §4A block 98–108): "U1", next line `nU1` | — |
| 105 | `T3Frame` | E17 | diagram label "Dd" (the store rename's §4A block relabelled the Tucker core) | "D(d-1)" |
| 118 | `T3Frame` | N17 | done by the store rename (R4a): "up_tt_cores = (U0,...,U(d-1)), up-orthogonal:    U_aib U_ajb = delta_ij" | — |
~~~

#### wl:883-887 (before) / wl:884-888 (after)

BEFORE:

~~~md
| 355 | `is_orthogonal` | N4 | "down/outer D_i (all i):   ``einsum('...iaj,...ibj->...ab', D, D) = I``" | "down O_i (all i):   ``einsum('...iaj,...ibj->...ab', O, O) = I``" |
| 365 | `is_orthogonal` | N17 | "Orthogonal cores (left/right/outer/Tucker)" | HELD (N17): keep, or "(left/right/up/down)" |
| 1271 | `fv_to_t3` | N4 | `fv_to_t3` diagram core "D2" | "O2" |
| 1371 | `t3_orthogonal_representations` | N17 | "tucker_cores = (U0,U1, U2, U3), up orthogonal" | HELD (N17): "down-orthogonal" |
| 1372 | `t3_orthogonal_representations` | N17 | "down_tt_cores = (O0, O1, O2, O3), down orthogonal" | HELD (N17): "up-orthogonal" |
~~~

AFTER:

~~~md
| 355 | `is_orthogonal` | N4 | "up/outer U_i (all i):     ``einsum('...iaj,...ibj->...ab', U, U) = I``" (the rename swapped the word and the letters) | "up U_i (all i):     ``einsum('...iaj,...ibj->...ab', U, U) = I``" |
| 365 | `is_orthogonal` | N17 | "Orthogonal cores (left/right/outer/Tucker)" | "Orthogonal cores (left/right/up/down)" |
| 1271 | `fv_to_t3` | N4 | done by the store rename (R4a, §4A block): diagram core "U2" | — |
| 1371 | `t3_orthogonal_representations` | N17 | done by the store rename (R4a): "down_tucker_cores = (D0, D1, D2, D3), down orthogonal" | — |
| 1372 | `t3_orthogonal_representations` | N17 | done by the store rename (R4a): "up_tt_cores = (U0, U1, U2, U3), up orthogonal" | — |
~~~

#### wl:902-903 (before) / wl:903-904 (after)

BEFORE:

~~~md
| 1343 | `frame` | N17 | "(the left/right/outer-orthogonal representation)" | HELD (N17): keep, or "(the orthogonal representation)" |
| 1638 | `retract` | E6 | ``` slots hold different cores, e.g. a slack frame with ``nD != nU`` -- review H3-8); when the ``` | ``` slots hold different cores, e.g. a slack frame with ``nD != nU``); when the ``` |
~~~

AFTER:

~~~md
| 1343 | `frame` | N17 | done by the store rename (R4a): "(the left/right/up-orthogonal representation)" | — |
| 1638 | `retract` | E6 | ``` slots hold different cores, e.g. a slack frame with ``nU != nD`` -- review H3-8); when the ``` | ``` slots hold different cores, e.g. a slack frame with ``nU != nD``); when the ``` |
~~~

#### wl:940-940 (before) / wl:941-941 (after)

BEFORE:

~~~md
| 615 | `get_minimal_ranks` | M/U | "Down TT core unfoldings are full rank" | "... TT core matricizations are full rank" (the adjective is N17, HELD) |
~~~

AFTER:

~~~md
| 615 | `get_minimal_ranks` | M/U | "Up TT core unfoldings are full rank" (the store rename's §4.3 row turned "Down" into "Up") | "TT core up-matricizations are full rank" (as `user_guide.rst:97` reads after the rename) |
~~~

#### wl:945-948 (before) / wl:946-949 (after)

BEFORE:

~~~md
| 2956 | `up_svd_tt_core` | M/U | `up_svd_tt_core`: "Compute SVD of ith TT-core right unfolding and contract non-orthogonal factor down into the tucker core below." | as `t3_up_svd_tt_core` above |
| 2978 | `up_svd_tt_core` | N17 | `up_svd_tt_core` Returns: "with ith TT-core down orthogonal" | HELD (N17): "up-orthogonal" |
| 3002 | `up_svd_tt_core` | N17 | "# TT-core is down orthogonal" | HELD (N17): "# TT-core is up-orthogonal" |
| 3108 | `orthogonalize_relative_to_tt_core` | N17 | "# Core below G1 is up orthogon[al]" (a Tucker core, `B Bᵀ = I`) | HELD (N17): "down-orthogonal" |
~~~

AFTER:

~~~md
| 2956 | `up_svd_tt_core` | M/U | `up_svd_tt_core`: "Compute SVD of ith TT-core up unfolding and contract non-orthogonal factor down into the tucker core below." (the store rename's §4.3 row turned "right" into "up") | as `t3_up_svd_tt_core` above |
| 2978 | `up_svd_tt_core` | N17 | done by the store rename (R4a): `up_svd_tt_core` Returns: "with ith TT-core up orthogonal" | — |
| 3002 | `up_svd_tt_core` | N17 | done by the store rename (R4a): "# TT-core is up orthogonal" | — |
| 3108 | `orthogonalize_relative_to_tt_core` | N17 | done by the store rename (R4a): "# Core below G1 is down orthogonal" (a Tucker core, `B Bᵀ = I`) | — |
~~~

#### wl:968-969 (before) / wl:969-970 (after)

BEFORE:

~~~md
| 1122 | `ut3_orthogonal_representations` | N17 | "tucker_cores = (U0,U1, U2, U3), up orthogonal" | HELD (N17): "down-orthogonal" |
| 1125 | `ut3_orthogonal_representations` | N17 | "outer_tt_cores = (O0, O1, O2, O3), down orthogonal" | HELD (N17): "down_tt_cores = (O0, O1, O2, O3), up-orthogonal" (the name `outer_tt_cores` is also stale: N4) |
~~~

AFTER:

~~~md
| 1122 | `ut3_orthogonal_representations` | N17 | done by the store rename (R4a): "down_tucker_cores = (D0, D1, D2, D3), down orthogonal" | — |
| 1125 | `ut3_orthogonal_representations` | N17 | done by the store rename (R4a): "up_tt_cores = (U0, U1, U2, U3), up orthogonal" (also N4: the stale name `outer_tt_cores` is gone) | — |
~~~

#### wl:1112-1112 (before) / wl:1113-1113 (after)

BEFORE:

~~~md
| 750 | § The assembly, factored by the product rule | E6 | ``` \State undotted: $\nu$ re-swept through $G$; $\ddet=Uc$; $\tilde\tau$, $\tilde\sigma$ \Comment{GN kernels, recompute (I-11)} ``` | ``` \State undotted: $\nu$ re-swept through $G$; $\ddet=Uc$; $\tilde\tau$, $\tilde\sigma$ \Comment{GN kernels, recompute} ``` |
~~~

AFTER:

~~~md
| 750 | § The assembly, factored by the product rule | E6 | ``` \State undotted: $\nu$ re-swept through $G$; $\ddet=Bc$; $\tilde\tau$, $\tilde\sigma$ \Comment{GN kernels, recompute (I-11)} ``` | ``` \State undotted: $\nu$ re-swept through $G$; $\ddet=Bc$; $\tilde\tau$, $\tilde\sigma$ \Comment{GN kernels, recompute} ``` |
~~~

### (3) Checks

**K2/K3 over the work list's target column** (`target_gate.py`; K2 = `(?<![\\\w])nO`, `n_\{O`, `n\^O`,
`\b(?:n_up|n_down|r_left|r_right|k_up|k_down|m_up|m_down)\w*`; K3 = `` `'(?:ULRO|UL|RO)'` ``, `\('(?:down|up|left|right)'`):

- all 599 "replacement" cells of the work list by file: K2 **0**, K3 **0**;
- the 45 table rows this edit re-expressed: K2 **0**, K3 **0**;
- the whole file (items table, notes, every column): K2 **0**, K3 **0**.

**Gate tool**, `dev/store_rename_check.py --stage R5 --hits dev/store_rename_slices --only K1,K2,K3,K4,K5,K7,K8`:

- baseline: `docs/glossary.md` 14 hits (K3-docs 2: `'UL'`, `'RO'`; K4 3: `(U, G, G, G)`, `(U, O, P, Q)` ×2; K5 2:
  `(up, down, left` ×2; K8 7: `` `O` `` ×4, `P_last`, `Oi`, `δU`);
- after: **0** glossary hits. (Tree-wide, K8 still reports 2 hits, both in `docs/exact_hessian_two_hole.tex:397`
  (`Q_{\mathrm c}`), another agent's file.)
- carve-outs relied on: the four "(2026.4.0" items (alignment pass, corewise frame, frame cores, store; each read by
  hand, and re-gated with the carve-out disabled: only their alias groups hit), the T4S entry (the T4S line), and
  N17's alias "outer-orthogonal" in **orthogonal core**. No hit is left that I believe must stay.

**K9 (informational; not in the brief's gate).** `--only K9` reports K9-docs FAIL with 14 glossary lines. The glossary
has no `v2026.4.0` image, so no line can pass by alignment; each needs a keep or an R5 registration, in files I may
not touch (`dev/store_rename_keeps.txt`, `dev/store_rename_slices/k9_registered.tsv`):

- keeps (inventory 3 §5.2 "stays": adjectives or a travel direction), unchanged by me: `:79` (down-orthogonal),
  `:102`, `:104`, `:105` (the **orthogonal core** adjectives), `:417` (`ξ` up the Tucker cores), `:726` (the
  up-matricization `Wi`);
- my R5 rewrites (clause (c)/R5 registration): `:107`, `:108` (orthogonal core), `:121` (padded widths), `:713–715`
  (Symbols intro), `:731`, `:732` (the `D` and `U` rows).

Lines are as in the final file. The session should register these or tell me to reword.

### (4) Judgement calls

1. **N17 rows marked done without the hyphen.** The rename applied the substantive change of every HELD N17 row (the
   adjective now agrees with the method convention), but writes "down orthogonal" where the HELD target had
   "down-orthogonal" (`probing.py:646–647, 1221–1222`, `frame_variations_format.py:1372–1373`,
   `uniform_frame_variations_format.py:1122, 1125`, `tucker_tensor_train.py:2978, 3002, 3108`). Their siblings in the
   same blocks ("left orthogonal elements", "left orthogonal") are unhyphenated, so I marked them done rather than
   keep a hyphen-only row. See question 1.
2. **N17 item row.** Sample edit 2's instructions applied (status cell, last sentences, the two evidence
   parentheticals), plus three small additions so the record reads true after the rename: "At 2026.4.0:" opens the
   evidence cell (its quotes are 2026.4.0 text); "the `O` cores'" → "the TT-shaped cores' (then `O`, now `U`)"; "are
   the problem" → "were the problem".
3. **E10 item.** The claim keeps the 2026.4.0 formula and says it was 2026.4.0's letter, fixed by the rename at its one
   site; the verification note gives the stored shapes `(nD, N)`, `(nU, N)` (from the `T3Frame` docstring) instead
   of the generic `(n, N)`.
4. **Line numbers.** The work list's "line" column is left as of `daa1e23d` (its stated convention: re-grep); several
   targets moved (`T3Frame` rows +1, `user_guide.rst:142` → 140, the tex :750 → 753).
5. **Two rows outside inventory 3 §7**: `exact_hessian_two_hole.tex:750` (θ's raw Tucker core `U` → `B`, plan §4.8;
   matches the note as converted) and the header's HELD sentence.
6. **`:940` target** follows inventory 3 and the form R5 already applied in `user_guide.rst:97` ("TT core
   up-matricizations are full rank"), not a sibling-parallel "Up TT core matricizations".
7. **`:884`** released as inventory 3 proposes, "(left/right/up/down)", although the sentence cites T4S App. A.1,
   whose word is "outer".
8. **Glossary alias groups.** Besides sample edit 2's group in **frame cores**, I added "(2026.4.0 …)" groups to
   **corewise frame** (`(U, G, G, G)`), **store** (`up_ranks` (Tucker), `down_ranks` (TT-shaped)) and **alignment
   pass** (the 2026.4.0 letters `U`, `O`, `L`, `R`, in words). Each group sits on one physical line, and
   the four whole items were read by hand.
9. **Not listed as retired spellings:** the `update` strings `'UL'`, `'RO'`, `'ULRO'` (K3-docs is not carved out by
   "(2026.4.0", so quoting them would fail the gate; the alignment pass entry describes 2026.4.0's form in words) and
   `nO` (K2: none anywhere).
10. **Symbols.** The letter ↔ word mapping (plan §6) went into the Symbols intro paragraph (`D`, `U`, `L`, `R`; `nD`,
    `nU`, `rL`, `rR`; `B`, `G`), because the table has no `R` row. The `D`/`U` rows give the old meanings as "Before
    2026.5.0 …" rather than "(2026.4.0 …)": the table is one block for the carve-out, so that string would exempt the
    whole table from K1/K2/K4/K5/K8. The `U` row does not mention T4S (T4S letters only in the T4S entry and the `P`
    row). The `L` row gained "and the left store's letter (… `update`)", parallel to `D` and `U`.
11. **Other glossary rows found by the grep:** **base-point tangent** `P_last` → `L_last` (the PQU inventory's
    glossary:230 row, D-7; the code and `frame_variations.md:42` already read `L_last`); **padded widths** gained one
    sentence defining the four widths and the swap. Checked and left: **left-orthogonal form** (down-orthogonal, ADJ),
    **edge variables** (`ξ` up, travel), **captured hint energy** ("upward", unrelated), the `W` row
    (up-matricization, ADJ), the `H`/`S`/`T` rows, **lift**/**curvature term** (the lift `L`, separate entries).

### Questions

1. Hyphenation (call 1): add one typographic row to the 601-row pass ("down orthogonal" → "down-orthogonal" and the
   siblings), or leave as is?
2. The Symbols table has no `R` row (right core and store letter; QR's `R`; the retraction, now `\mathcal R` in the
   exact-Hessian note; the tests' RNG `R`) and no `F` row (the edge frame `F_{x→t}`, `Π_{T(F)}`, `m_c(F)`; `F(X)` in
   `fitting.py`, item N11). Add them? Not in the plan, so not added.
3. K9-docs registrations for the 14 glossary lines above: the session's to add.


## Appendix D. The CHANGELOG section and the upgrade section, as shipped

### D.1 `CHANGELOG.md` `[2026.5.0]` (= `release_notes_2026.5.0.md` with the heading)

~~~~markdown
## [2026.5.0] — unreleased

The headline is **the store rename** (breaking): the frame stores `up` and `down` swap names, so that a store is
named for the side of its edge that its cores occupy, as `left` and `right` always were. Every number is
bit-identical; names, letters and messages change. Upgrade notes: `docs/release_notes.md`, "Upgrading from 2026.4.0".

### Changed — breaking

- **The frame stores `up` and `down` have swapped names: a name that kept its spelling now means the other store.**
  These are listed first because nothing catches them: they need not raise, and when one does the message does not
  name the width. Same spelling, other store:
  - `up_ranks` / `down_ranks` (`T3Frame`, `T3FrameWeights`, `UT3Frame`, `UT3Variations`, `UT3FrameWeights`);
  - `T3FrameWeights.up_weights` / `down_weights`, `UT3FrameWeights.up_weight_supercore` / `down_weight_supercore`,
    `UT3FrameMasks.up_mask` / `down_mask` and `UT3VariationsMasks.variations_up_mask` / `variations_down_mask`
    (dataclass fields, so also constructor and `dataclasses.replace` keywords);
  - the `up_ranks=` / `down_ranks=` keywords of `ranks.frame_has_minimal_ranks` and `ufv_masking.ufv_make_frame_masks`;
  - the width properties `nU` / `nD` (`UT3Frame`, `UT3Variations`, `UT3FrameWeights`) and the width keywords `nU=` /
    `nD=` (`UT3Frame.from_t3frame`, `UT3Variations.from_t3variations`, `UT3FrameWeights.from_t3frameweights`,
    `UT3Tangent.from_t3tangent`, `ufv_conversions.t3frame_to_ut3frame`, `t3variations_to_ut3variations`,
    `t3frameweights_to_ut3frameweights`, `ufv_masking.ufv_make_frame_masks`). `nD` is now the down store's width
    (the Tucker factors' rows) and `nU` the up store's (the TT-shaped cores' mode leg).

  At a minimal frame both stores have the same ranks, so stale code returns plausible numbers and is wrong only at
  slack. A stale width keyword pads the other store; it raises only when the swapped width is below that store's
  ranks, and then from inside numpy (`index can't contain negative values`). The new meaning: the **up** store is the
  TT-shaped cores above each Tucker edge (up-orthogonal, built by `up_orthogonalize_tt_cores`), the **down** store
  the orthonormal Tucker factors below it (down-orthogonal, `down_orthogonalize_tucker_cores`). In 2026.4.0 the two
  words named the direction the cores' edges point, which is the other way round and disagreed with those method
  names. Not aliased, as for `GaussNewtonModel` in 2026.3.0: a meaning swap cannot be aliased.
- **Renamed, so stale code raises** (`AttributeError` or `TypeError`): `T3Frame.up_tucker_cores` →
  `down_tucker_cores` and `T3Frame.down_tt_cores` → `up_tt_cores` (dataclass fields, so also constructor and
  `dataclasses.replace` keywords); `UT3Frame.up_tucker_supercore` → `down_tucker_supercore` and
  `UT3Frame.down_tt_supercore` → `up_tt_supercore`; the backend keywords `up_tucker_cores[_dot]=` →
  `down_tucker_cores[_dot]=` (`probing.compute_xi`, `compute_deta_tilde`, `compute_deta_tilde_tangent`;
  `sampling_derivatives.compute_deta_tilde_jets`, `compute_deta_tilde_tangent_jets`) and `down_tt_cores[_dot]=` →
  `up_tt_cores[_dot]=` (`probing.compute_eta`, `compute_sigma`, `compute_tau`, `compute_dxi_tilde`,
  `compute_dxi_tilde_tangent`; `sampling_derivatives.compute_sigma_jets`, `compute_tau_jets`, `compute_dxi_tilde_jets`,
  `compute_dxi_tilde_tangent_jets`).
- **`update` takes a tuple of single store letters in the new meaning** (`T3Tangent.align_padded`,
  `UT3Tangent.align_padded`, `padded_frames.tv_align_padded_frame`, `uniform_padded_frames.utv_align_padded_frame`):
  `'D'` down, `'U'` up, `'L'` left, `'R'` right, any order, default `('D', 'U', 'L', 'R')`; the two-colour passes are
  `('D', 'L')`, then `('U', 'R')`. Convert a 2026.4.0 string letter by letter (`'U'` → `'D'`, `'O'` → `'U'`, `'L'`
  and `'R'` kept) and split it: `'ULRO'` → `('D', 'U', 'L', 'R')`, `'UL'` → `('D', 'L')`, `'RO'` → `('U', 'R')`.
  Every `str` raises:

  ```
  ValueError: update must be a non-empty tuple of the store letters 'D', 'U', 'L', 'R' without repeats (D down, U up, L left, R right), e.g. ('D', 'L'); got 'ULRO'; that is the 2026.4.0 form, whose 'U' was the Tucker store (now 'D') and 'O' the TT-shaped store (now 'U'): swap each letter and pass a tuple, e.g. 'UL' -> ('D', 'L'), not ('U', 'L'), which now means up and left
  ```

  A tuple or list of the old letters, which 2026.4.0 also accepted, raises only if it holds `'O'`; otherwise it runs
  with the new meaning: `('U', 'L')` now updates the up and left stores.
- **Migrate in one pass.** Two `sed` passes undo each other. Swap every name at once, longest first, and only these
  (never a `*_orthogonalize_*` or `*_svd*` name, whose words are unchanged):

  ```python
  import re
  SWAP = {'up_tucker_cores': 'down_tucker_cores', 'down_tt_cores': 'up_tt_cores',
          'up_tucker_cores_dot': 'down_tucker_cores_dot', 'down_tt_cores_dot': 'up_tt_cores_dot',
          'up_tucker_supercore': 'down_tucker_supercore', 'down_tt_supercore': 'up_tt_supercore',
          'up_ranks': 'down_ranks', 'down_ranks': 'up_ranks',
          'up_weights': 'down_weights', 'down_weights': 'up_weights',
          'up_weight_supercore': 'down_weight_supercore', 'down_weight_supercore': 'up_weight_supercore',
          'up_mask': 'down_mask', 'down_mask': 'up_mask',
          'variations_up_mask': 'variations_down_mask', 'variations_down_mask': 'variations_up_mask',
          'nU': 'nD', 'nD': 'nU'}
  pattern = re.compile(r'\b(' + '|'.join(sorted(SWAP, key=len, reverse=True)) + r')\b')
  source = pattern.sub(lambda m: SWAP[m.group(1)], source)    # then review the nU/nD hits: they may be your own names
  ```

  `\b` treats `_` as a word character, so `up_mask` does not match inside `variations_up_mask`, nor `up_tucker_cores`
  inside `up_tucker_cores_dot` (hence its own entry). `rL` and `rR` are unchanged. The `update` strings need the hand
  edit above.
- **What does not change.** Positions: `T3Frame.data` and every rank, mask, weight and uniform tuple keep their order,
  which now reads `(down, up, left, right)`, so `up, down, left, right = frame.data` keeps working (rename the locals
  to read right), and so do positional width calls (`UT3Frame.from_t3frame(F, N, 7, 5)`). Saved files load
  unchanged (`save`/`load` key arrays by position). Every number is bit-identical, seeded `padded_seed` /
  `cone_seed` streams included. The orthogonalization and SVD method names (`down_orthogonalize_tucker_cores`,
  `up_orthogonalize_tt_cores`, `up_svd`, …), every optimizer keyword, and every other public name are unchanged.
- **Text only:** the reprs (`T3Frame(… down_ranks=…, up_ranks=…)`, `UT3Frame(… nD=…, nU=…)`) and every error message
  that names a store use the new words.

### Changed

- **Letters in the docs, docstrings and math notes.** Diagrams, equations and shape comments write the stores `D`,
  `U`, `L`, `R` with widths `nD`, `nU`, `rL`, `rR` (`docs/frame_variations.md`, "Four rank stores"); a T3's own cores
  `B`, `G` (the corewise frame is `(B, G, G, G)`: `B` substituted for `D`, `G` for `U`, `L`, `R`); a
  variation-shaped array `V` or `H` by its shape, whatever computed it (e.g. the transposes' outputs `V_tilde`,
  `H_tilde`); the jet shift `sh` / `shᵀ` (was `D` / `U`; the functions `_shift_down` / `_shift_up` keep their names);
  the derivative an upright `d` (`dΦ`); edge variables' travel directions with `-ward` (`upward`, `adjoint-upward`),
  so a bare up or down always names a store. All six math notes are converted, the T4S-notation ones included; each
  states the correspondence once: T4S writes U, O, P, Q for our D, U, L, R; its δU is our V (`tucker_variations`),
  its δG our H (`tt_variations`). Docs and changelog entries before 2026.5.0 wrote `U` for the Tucker factor and `O`
  for the TT-shaped core.
~~~~

### D.2 `docs/release_notes.md` "Upgrading from 2026.4.0"

~~~~markdown
## Upgrading from 2026.4.0

**The frame stores `up` and `down` have swapped names: a name that kept its spelling now means the other store,
and it need not raise.** `frame.up_ranks` was the Tucker factors' ranks and is now the up cores' mode legs;
`UT3Frame.nU` was the padded Tucker width and is now the up store's. At a minimal frame the two stores agree, so
such code even returns the right numbers there: search for the names below rather than waiting for an error. When
a stale width keyword does raise (a swapped width below the other store's ranks), the error comes from inside numpy
and does not name the width.

**Why.** A store is now named for the side of its edge that its cores occupy, as `left` and `right` always were.
Draw the TT chain on top and the Tucker factors below it: the **up** store is the TT-shaped cores above each Tucker
edge (up-orthogonal), the **down** store the Tucker factors below it (down-orthogonal). The 2026.4.0 words named
where the cores' edges point, the opposite way, and disagreed with the method names `up_orthogonalize_tt_cores` and
`down_orthogonalize_tucker_cores`, which are unchanged. See
[Four rank stores](frame_variations.md#four-rank-stores--and-why-excess-rank-is-inert).

| 2026.4.0 | 2026.5.0 | raises if left? |
|---|---|---|
| `up_ranks` / `down_ranks` (`T3Frame`, `T3FrameWeights`, `UT3Frame`, `UT3Variations`, `UT3FrameWeights`) | `down_ranks` / `up_ranks` | **no** |
| `T3FrameWeights.up_weights` / `down_weights` (fields and keywords) | `down_weights` / `up_weights` | **no** |
| `UT3FrameWeights.up_weight_supercore` / `down_weight_supercore` (fields and keywords) | swapped | **no** |
| `UT3FrameMasks.up_mask` / `down_mask` (fields and keywords) | swapped | **no** |
| `UT3VariationsMasks.variations_up_mask` / `variations_down_mask` (fields and keywords) | swapped | **no** |
| keywords `up_ranks=` / `down_ranks=` (`ranks.frame_has_minimal_ranks`, `ufv_masking.ufv_make_frame_masks`) | swapped | **no** |
| properties `nU` / `nD` (`UT3Frame`, `UT3Variations`, `UT3FrameWeights`) | `nD` / `nU` | **no** |
| keywords `nU=` / `nD=` (the four `from_*` converters, `ufv_conversions.*_to_ut3*`, `ufv_make_frame_masks`) | `nD=` / `nU=` | **no** (it need not) |
| `T3Frame.up_tucker_cores` / `down_tt_cores` (fields and keywords) | `down_tucker_cores` / `up_tt_cores` | yes |
| `UT3Frame.up_tucker_supercore` / `down_tt_supercore` | `down_tucker_supercore` / `up_tt_supercore` | yes |
| keywords `up_tucker_cores[_dot]=` / `down_tt_cores[_dot]=` (`probing.compute_*`, `sampling_derivatives.compute_*_jets`) | `down_tucker_cores[_dot]=` / `up_tt_cores[_dot]=` | yes |
| `update='ULRO'` / `'UL'` / `'RO'` (`T3Tangent.align_padded`, `UT3Tangent.align_padded`, `tv_align_padded_frame`, `utv_align_padded_frame`) | `('D', 'U', 'L', 'R')` / `('D', 'L')` / `('U', 'R')` | yes for a `str`; see below |

**Migrate in one pass.** Two `sed` passes undo each other. Swap every name at once, longest first, and only these
(never a `*_orthogonalize_*` or `*_svd*` name, whose words are unchanged):

```python
import re
SWAP = {'up_tucker_cores': 'down_tucker_cores', 'down_tt_cores': 'up_tt_cores',
        'up_tucker_cores_dot': 'down_tucker_cores_dot', 'down_tt_cores_dot': 'up_tt_cores_dot',
        'up_tucker_supercore': 'down_tucker_supercore', 'down_tt_supercore': 'up_tt_supercore',
        'up_ranks': 'down_ranks', 'down_ranks': 'up_ranks',
        'up_weights': 'down_weights', 'down_weights': 'up_weights',
        'up_weight_supercore': 'down_weight_supercore', 'down_weight_supercore': 'up_weight_supercore',
        'up_mask': 'down_mask', 'down_mask': 'up_mask',
        'variations_up_mask': 'variations_down_mask', 'variations_down_mask': 'variations_up_mask',
        'nU': 'nD', 'nD': 'nU'}
pattern = re.compile(r'\b(' + '|'.join(sorted(SWAP, key=len, reverse=True)) + r')\b')
source = pattern.sub(lambda m: SWAP[m.group(1)], source)    # then review the nU/nD hits: they may be your own names
```

`\b` treats `_` as a word character, so `up_mask` does not match inside `variations_up_mask`, nor `up_tucker_cores`
inside `up_tucker_cores_dot` (hence its own entry). `rL` and `rR` are unchanged. Your own `nU`/`nD` locals are
swapped along with the keywords, which keeps them consistent.

**`update` needs a hand edit.** It takes a tuple of the new store letters, any order: `'D'` down, `'U'` up, `'L'`
left, `'R'` right. Swap each old letter (`'U'` → `'D'`, `'O'` → `'U'`; `'L'` and `'R'` stay) and split: `'ULRO'` →
`('D', 'U', 'L', 'R')` (the default), `'UL'` → `('D', 'L')`, not `('U', 'L')`, which now updates the up store; `'RO'`
→ `('U', 'R')`. Every `str` raises:

```
ValueError: update must be a non-empty tuple of the store letters 'D', 'U', 'L', 'R' without repeats (D down, U up, L left, R right), e.g. ('D', 'L'); got 'ULRO'; that is the 2026.4.0 form, whose 'U' was the Tucker store (now 'D') and 'O' the TT-shaped store (now 'U'): swap each letter and pass a tuple, e.g. 'UL' -> ('D', 'L'), not ('U', 'L'), which now means up and left
```

A tuple or list of the old letters, which 2026.4.0 also accepted, raises only if it holds `'O'`: convert it the same
way, or it runs with the new meaning.

**What does not change.** Positions: `T3Frame.data` and every rank, mask, weight and uniform tuple keep their order,
which now reads `(down, up, left, right)`, so `up, down, left, right = frame.data` keeps working (rename the locals
to read right), and so do positional width calls (`UT3Frame.from_t3frame(F, N, 7, 5)`). Saved files load unchanged.
Every number is bit-identical, seeded starts included. The orthogonalization and SVD method names, every optimizer
keyword and every other public name are unchanged.

**Letters.** Diagrams, equations and shape comments write `D`, `U`, `L`, `R` and the widths `nD`, `nU`, `rL`, `rR`;
a T3's own cores are `B`, `G`; a variation-shaped array is `V` or `H` by its shape; the jet shift is `sh` / `shᵀ`
and the derivative an upright `d`. Docs before 2026.5.0 wrote `U` for the Tucker factor and `O` for the TT-shaped
core. T4S writes U, O, P, Q for our D, U, L, R; its δU is our V (`tucker_variations`), its δG our H
(`tt_variations`).
~~~~

## Appendix A. Every docs and notes block (§4A): before (v2026.4.0; the glossary 88fbbedb) and after (the tree)

Generated from the inventory entries (`diagrams_inventory.md`, checked verbatim by K12). Old span is the base's, new
span the tree's.

### DOC-BAS-0040 — `docs/batching_and_stacking.md` 40–46 → 40–46, markdown table

before:

~~~
| Array | Blocks | Shape |
|---|---|---|
| frame Tucker core `Uᵢ` (`T3Frame`) | `C` | `(2,) + (nᵢ, Nᵢ)` |
| frame TT core `Pᵢ` (`T3Frame`) | `C` | `(2,) + (rᵢ, nᵢ, rᵢ₊₁)` |
| variation core `δUᵢ` (`T3Variations`) | `K + C` | `(3, 2) + (nᵢ, Nᵢ)` |
| probe vector `wᵢ` (`ww`) | `W` | `(4,) + (Nᵢ,)` |
| forward probe `zᵢ` (`tangent.probe(ww)`) | `W + K + C` | `(4, 3, 2) + (Nᵢ,)` |
~~~

after:

~~~
| Array | Blocks | Shape |
|---|---|---|
| frame Tucker core `Dᵢ` (`T3Frame`) | `C` | `(2,) + (nᵢ, Nᵢ)` |
| frame TT core `Lᵢ` (`T3Frame`) | `C` | `(2,) + (rᵢ, nᵢ, rᵢ₊₁)` |
| variation core `Vᵢ` (`T3Variations`) | `K + C` | `(3, 2) + (nᵢ, Nᵢ)` |
| probe vector `wᵢ` (`ww`) | `W` | `(4,) + (Nᵢ,)` |
| forward probe `zᵢ` (`tangent.probe(ww)`) | `W + K + C` | `(4, 3, 2) + (Nᵢ,)` |
~~~

### TEX-EHTH-0103 — `docs/exact_hessian_two_hole.tex` 103 → 102, LaTeX equation line

before:

~~~
\paragraph{The format.} A T3 with raw cores $\th = \big((U_i)_{i=1}^d, (G_i)_{i=1}^d\big)$, $U_i\in\R^{n_i\times N_i}$,
~~~

after:

~~~
\paragraph{The format.} A T3 with raw cores $\th = \big((B_i)_{i=1}^d, (G_i)_{i=1}^d\big)$, $B_i\in\R^{n_i\times N_i}$,
~~~

### TEX-EHTH-0105 — `docs/exact_hessian_two_hole.tex` 105 → 104, LaTeX equation line

before:

~~~
$X = \Phi(\th) = \mathrm{TT}(G_1,\dots,G_d)\times_1 U_1\cdots\times_d U_d$ (code: \code{TuckerTensorTrain.data}
~~~

after:

~~~
$X = \Phi(\th) = \mathrm{TT}(G_1,\dots,G_d)\times_1 B_1\cdots\times_d B_d$ (code: \code{TuckerTensorTrain.data}
~~~

### TEX-EHTH-0108 — `docs/exact_hessian_two_hole.tex` 108–112 → 107–111, LaTeX equation line

before:

~~~
\begin{equation}
  \label{eq:Dphi}
  D\Phi(\th)[\thh] = \sum_{a} \Phi(\th;\, a\leftarrow \thh_a), \qquad
  D^2\Phi(\th)[\thh,\thh'] = \sum_{a\neq b} \Phi(\th;\, a\leftarrow\thh_a,\, b\leftarrow\thh'_b),
\end{equation}
~~~

after:

~~~
\begin{equation}
  \label{eq:Dphi}
  \mathrm d\Phi(\th)[\thh] = \sum_{a} \Phi(\th;\, a\leftarrow \thh_a), \qquad
  \mathrm d^2\Phi(\th)[\thh,\thh'] = \sum_{a\neq b} \Phi(\th;\, a\leftarrow\thh_a,\, b\leftarrow\thh'_b),
\end{equation}
~~~

### TEX-EHTH-0117 — `docs/exact_hessian_two_hole.tex` 117–123 → 116–126, LaTeX equation line

before:

~~~
the representations are equivalent, and everything below is recomputed from the stored cores) is $(U_i, O_i, P_i, Q_i)$ = \code{T3Frame.data} = (up, down, left, right):
$U_i\in\R^{n_i\times N_i}$ with orthonormal rows, left-orthogonal $P_i\in\R^{r^L_{i-1}\times n_i\times r^L_i}$,
right-orthogonal $Q_i\in\R^{r^R_{i-1}\times n_i\times r^R_i}$, and $O_i\in\R^{r^L_{i-1}\times n^D_i\times r^R_i}$
orthonormal over its middle index ($n^D_i\le n_i$; $n^D_i<n_i$ is \emph{Tucker slack}, $r^R_i<r^L_i$ \emph{TT slack}).
The base point in its left-canonical gauge is $\th=(U,P)$ (\code{ManifoldGeometryOps.base_point}).
A tangent vector has gauged coordinates $v=(\delta U_i,\delta G_i)$, $\delta U_i\in\R^{n^D_i\times N_i}$,
$\delta G_i\in\R^{r^L_{i-1}\times n_i\times r^R_i}$ (code: \code{tucker_variations} $V$, \code{tt_variations} $H$),
~~~

after:

~~~
the representations are equivalent, and everything below is recomputed from the stored cores) is $(D_i, U_i, L_i, R_i)$ = \code{T3Frame.data} = (down, up, left, right):
$D_i\in\R^{n_i\times N_i}$ with orthonormal rows, left-orthogonal $L_i\in\R^{r^L_{i-1}\times n_i\times r^L_i}$,
right-orthogonal $R_i\in\R^{r^R_{i-1}\times n_i\times r^R_i}$, and $U_i\in\R^{r^L_{i-1}\times n^U_i\times r^R_i}$
orthonormal over its middle index ($n^U_i\le n_i$; $n^U_i<n_i$ is \emph{Tucker slack}, $r^R_i<r^L_i$ \emph{TT slack}).

T4S writes $U$, $O$, $P$, $Q$ for our $D$, $U$, $L$, $R$; its $\delta U$ is our $V$ (\code{tucker_variations}), its $\delta G$ our $H$ (\code{tt_variations}).
T4S, and this note before 2026.5.0, write a T3's own Tucker cores $U_i$, our $B_i$.

The base point in its left-canonical gauge is $\th=(D,L)$ (\code{ManifoldGeometryOps.base_point}).
A tangent vector has gauged coordinates $v=(V_i,H_i)$, $V_i\in\R^{n^U_i\times N_i}$,
$H_i\in\R^{r^L_{i-1}\times n_i\times r^R_i}$ (code: \code{tucker_variations}, \code{tt_variations}),
~~~

### TEX-EHTH-0125 — `docs/exact_hessian_two_hole.tex` 125–129 → 128–132, LaTeX equation line

before:

~~~
\begin{equation}
  \label{eq:E}
  \Emap v = \sum_i \big[P_{<i}\,\delta G_i\,Q_{>i}\big]\times_j U_j
          + \sum_i \big[P_{<i}\,O_i\,Q_{>i}\big]\times_i \delta U_i \times_{j\neq i} U_j ,
\end{equation}
~~~

after:

~~~
\begin{equation}
  \label{eq:E}
  \Emap v = \sum_i \big[L_{<i}\,H_i\,R_{>i}\big]\times_j D_j
          + \sum_i \big[L_{<i}\,U_i\,R_{>i}\big]\times_i V_i \times_{j\neq i} D_j ,
\end{equation}
~~~

### TEX-EHTH-0130 — `docs/exact_hessian_two_hole.tex` 130–132 → 133–135, LaTeX equation line

before:

~~~
where $P_{<i}=P_1\cdots P_{i-1}$ and $Q_{>i}=Q_{i+1}\cdots Q_d$ are TT chains, and $C\times_kM$ contracts
index $k$ of a core $C$ with the \emph{rows} of a matrix $M$ (so $\Phi(\th)=\mathrm{TT}(G)\times_1U_1\cdots$). The gauge conditions
$\delta U_i U_i^\T=0$ and $(P_i^{\langle 2\rangle})^\T\delta G_i^{\langle 2\rangle}=0$ ($i<d$; T4S (47)--(48)) are
~~~

after:

~~~
where $L_{<i}=L_1\cdots L_{i-1}$ and $R_{>i}=R_{i+1}\cdots R_d$ are TT chains, and $C\times_kM$ contracts
index $k$ of a core $C$ with the \emph{rows} of a matrix $M$ (so $\Phi(\th)=\mathrm{TT}(G)\times_1B_1\cdots$). The gauge conditions
$V_i D_i^\T=0$ and $(L_i^{\langle 2\rangle})^\T H_i^{\langle 2\rangle}=0$ ($i<d$; T4S (47)--(48)) are
~~~

### TEX-EHTH-0145 — `docs/exact_hessian_two_hole.tex` 145–146 → 148–149, LaTeX equation line

unchanged: R5: P_X is the tangent projector (§1.5 item 3); no store letter

### TEX-EHTH-0151 — `docs/exact_hessian_two_hole.tex` 151–200 → 154–203, LaTeX table or figure

before:

~~~
\begin{figure}[t]
\centering
\begin{tikzpicture}[x=9.5mm,y=9.5mm]
  % (a) the sampled network
  \begin{scope}
    \foreach \i/\n in {1/G_1,2/G_2,3/G_3,4/G_4} \node[ttc] (g\i) at (\i,0) {$\n$};
    \foreach \i/\n in {1/U_1,2/U_2,3/U_3,4/U_4} \node[tkc] (u\i) at (\i,-1.15) {$\n$};
    \node[vecn] (w1) at (1,-2.1) {$w_1$}; \node[vecn] (w2) at (2,-2.1) {$w_2$};
    \node[resn] (z3) at (3,-2.1) {$c_3$}; \node[vecn] (w4) at (4,-2.1) {$w_4$};
    \foreach \i in {1,2,3} { \pgfmathtruncatemacro\j{\i+1} \draw[bnd] (g\i)--(g\j); }
    \foreach \i in {1,2,3,4} { \draw[bnd] (g\i)--(u\i); }
    \draw[bnd] (u1)--(w1); \draw[bnd] (u2)--(w2); \draw[bnd] (u3)--(z3); \draw[bnd] (u4)--(w4);
    \node[font=\small] at (2.5,0.85) {(a) $\langle c, A\,\Phi(\theta)\rangle$ (one probe term)};
  \end{scope}
  % (b) one hole
  \begin{scope}[xshift=52mm]
    \node[ttc] (g1) at (1,0) {$G_1$}; \node[holec] (g2) at (2,0) {}; \node[ttc] (g3) at (3,0) {$G_3$};
    \node[ttc] (g4) at (4,0) {$G_4$};
    \foreach \i/\n in {1/U_1,2/U_2,3/U_3,4/U_4} \node[tkc] (u\i) at (\i,-1.15) {$\n$};
    \node[vecn] (w1) at (1,-2.1) {$w_1$}; \node[vecn] (w2) at (2,-2.1) {$w_2$};
    \node[resn] (z3) at (3,-2.1) {$c_3$}; \node[vecn] (w4) at (4,-2.1) {$w_4$};
    \draw[bnd] (g1)--(g2); \draw[bnd] (g2)--(g3); \draw[bnd] (g3)--(g4);
    \foreach \i in {1,2,3,4} { \draw[bnd] (g\i)--(u\i); }
    \draw[bnd] (u1)--(w1); \draw[bnd] (u2)--(w2); \draw[bnd] (u3)--(z3); \draw[bnd] (u4)--(w4);
    \draw[dashed,gray,rounded corners=3pt] (0.55,0.45)--(1.45,0.45)--(1.45,-2.45)--(0.55,-2.45)--cycle;
    \node[font=\scriptsize,anchor=east] at (0.55,-1.15) {$\mu_1$};
    \node[font=\small] at (2.5,0.85) {(b) one hole: $\nabla_{G_2}$};
  \end{scope}
  % (c) two holes
  \begin{scope}[xshift=104mm]
    \node[ttc] (g1) at (1,0) {$G_1$}; \node[holec] (g2) at (2,0) {}; \node[ttc] (g3) at (3,0) {$G_3$};
    \node[dotc] (g4) at (4,0) {$\dot G_4$};
    \foreach \i/\n in {1/U_1,2/U_2,3/U_3} \node[tkc] (u\i) at (\i,-1.15) {$\n$};
    \node[tkc] (u4) at (4,-1.15) {$U_4$};
    \node[vecn] (w1) at (1,-2.1) {$w_1$}; \node[vecn] (w2) at (2,-2.1) {$w_2$};
    \node[resn] (z3) at (3,-2.1) {$c_3$}; \node[vecn] (w4) at (4,-2.1) {$w_4$};
    \draw[bnd] (g1)--(g2); \draw[bnd] (g2)--(g3); \draw[bnd] (g3)--(g4);
    \foreach \i in {1,2,3,4} { \draw[bnd] (g\i)--(u\i); }
    \draw[bnd] (u1)--(w1); \draw[bnd] (u2)--(w2); \draw[bnd] (u3)--(z3); \draw[bnd] (u4)--(w4);
    \node[font=\small] at (2.5,0.85) {(c) two holes: one term of $(\K_c\hat\theta)_{G_2}$};
  \end{scope}
\end{tikzpicture}
\caption{Cores as nodes, bonds as edges (TT cores on top, Tucker factors below, sample vectors as squares).
(a) One term of $\ip{c}{A\Phi(\th)}$ for a probe: the weighted residual $c_3$ sits on the free mode.
(b) The one-hole sweep (the corewise transpose) leaves one core out: the open legs are its gradient block, here
the left environment $\mu_1$ (dashed box), the projected probe $\xi_2=U_2w_2$ and a right environment.
(c) The two-hole sweep leaves one core out \emph{and} replaces another by its direction ($\dot G_4$, red); summing
over the replaced core $b\neq a$ gives block $a$ of $\K_c\thh$ (eq.~\eqref{eq:Kdef}).}
\label{fig:holes}
\end{figure}
~~~

after:

~~~
\begin{figure}[t]
\centering
\begin{tikzpicture}[x=9.5mm,y=9.5mm]
  % (a) the sampled network
  \begin{scope}
    \foreach \i/\n in {1/G_1,2/G_2,3/G_3,4/G_4} \node[ttc] (g\i) at (\i,0) {$\n$};
    \foreach \i/\n in {1/B_1,2/B_2,3/B_3,4/B_4} \node[tkc] (u\i) at (\i,-1.15) {$\n$};
    \node[vecn] (w1) at (1,-2.1) {$w_1$}; \node[vecn] (w2) at (2,-2.1) {$w_2$};
    \node[resn] (z3) at (3,-2.1) {$c_3$}; \node[vecn] (w4) at (4,-2.1) {$w_4$};
    \foreach \i in {1,2,3} { \pgfmathtruncatemacro\j{\i+1} \draw[bnd] (g\i)--(g\j); }
    \foreach \i in {1,2,3,4} { \draw[bnd] (g\i)--(u\i); }
    \draw[bnd] (u1)--(w1); \draw[bnd] (u2)--(w2); \draw[bnd] (u3)--(z3); \draw[bnd] (u4)--(w4);
    \node[font=\small] at (2.5,0.85) {(a) $\langle c, A\,\Phi(\theta)\rangle$ (one probe term)};
  \end{scope}
  % (b) one hole
  \begin{scope}[xshift=52mm]
    \node[ttc] (g1) at (1,0) {$G_1$}; \node[holec] (g2) at (2,0) {}; \node[ttc] (g3) at (3,0) {$G_3$};
    \node[ttc] (g4) at (4,0) {$G_4$};
    \foreach \i/\n in {1/B_1,2/B_2,3/B_3,4/B_4} \node[tkc] (u\i) at (\i,-1.15) {$\n$};
    \node[vecn] (w1) at (1,-2.1) {$w_1$}; \node[vecn] (w2) at (2,-2.1) {$w_2$};
    \node[resn] (z3) at (3,-2.1) {$c_3$}; \node[vecn] (w4) at (4,-2.1) {$w_4$};
    \draw[bnd] (g1)--(g2); \draw[bnd] (g2)--(g3); \draw[bnd] (g3)--(g4);
    \foreach \i in {1,2,3,4} { \draw[bnd] (g\i)--(u\i); }
    \draw[bnd] (u1)--(w1); \draw[bnd] (u2)--(w2); \draw[bnd] (u3)--(z3); \draw[bnd] (u4)--(w4);
    \draw[dashed,gray,rounded corners=3pt] (0.55,0.45)--(1.45,0.45)--(1.45,-2.45)--(0.55,-2.45)--cycle;
    \node[font=\scriptsize,anchor=east] at (0.55,-1.15) {$\mu_1$};
    \node[font=\small] at (2.5,0.85) {(b) one hole: $\nabla_{G_2}$};
  \end{scope}
  % (c) two holes
  \begin{scope}[xshift=104mm]
    \node[ttc] (g1) at (1,0) {$G_1$}; \node[holec] (g2) at (2,0) {}; \node[ttc] (g3) at (3,0) {$G_3$};
    \node[dotc] (g4) at (4,0) {$\dot G_4$};
    \foreach \i/\n in {1/B_1,2/B_2,3/B_3} \node[tkc] (u\i) at (\i,-1.15) {$\n$};
    \node[tkc] (u4) at (4,-1.15) {$B_4$};
    \node[vecn] (w1) at (1,-2.1) {$w_1$}; \node[vecn] (w2) at (2,-2.1) {$w_2$};
    \node[resn] (z3) at (3,-2.1) {$c_3$}; \node[vecn] (w4) at (4,-2.1) {$w_4$};
    \draw[bnd] (g1)--(g2); \draw[bnd] (g2)--(g3); \draw[bnd] (g3)--(g4);
    \foreach \i in {1,2,3,4} { \draw[bnd] (g\i)--(u\i); }
    \draw[bnd] (u1)--(w1); \draw[bnd] (u2)--(w2); \draw[bnd] (u3)--(z3); \draw[bnd] (u4)--(w4);
    \node[font=\small] at (2.5,0.85) {(c) two holes: one term of $(\K_c\hat\theta)_{G_2}$};
  \end{scope}
\end{tikzpicture}
\caption{Cores as nodes, bonds as edges (TT cores on top, Tucker factors below, sample vectors as squares).
(a) One term of $\ip{c}{A\Phi(\th)}$ for a probe: the weighted residual $c_3$ sits on the free mode.
(b) The one-hole sweep (the corewise transpose) leaves one core out: the open legs are its gradient block, here
the left environment $\mu_1$ (dashed box), the projected probe $\xi_2=B_2w_2$ and a right environment.
(c) The two-hole sweep leaves one core out \emph{and} replaces another by its direction ($\dot G_4$, red); summing
over the replaced core $b\neq a$ gives block $a$ of $\K_c\thh$ (eq.~\eqref{eq:Kdef}).}
\label{fig:holes}
\end{figure}
~~~

### TEX-EHTH-0210 — `docs/exact_hessian_two_hole.tex` 210–211 → 213–214, LaTeX equation line

before:

~~~
Let $\zeta,\zeta'\in T_X\M$ and let $\thh,\thh'$ be any raw directions with $D\Phi(\th)[\thh]=\zeta$,
$D\Phi(\th)[\thh']=\zeta'$. Then
~~~

after:

~~~
Let $\zeta,\zeta'\in T_X\M$ and let $\thh,\thh'$ be any raw directions with $\mathrm d\Phi(\th)[\thh]=\zeta$,
$\mathrm d\Phi(\th)[\thh']=\zeta'$. Then
~~~

### TEX-EHTH-0212 — `docs/exact_hessian_two_hole.tex` 212–216 → 215–219, LaTeX equation line

before:

~~~
\begin{equation}
  \label{eq:hess}
  \ip{\Hess f(X)[\zeta]}{\zeta'} = \ip{\omega\odot A\zeta}{\omega\odot A\zeta'}
     + \ip{N}{D^2\Phi(\th)[\thh,\thh']},
\end{equation}
~~~

after:

~~~
\begin{equation}
  \label{eq:hess}
  \ip{\Hess f(X)[\zeta]}{\zeta'} = \ip{\omega\odot A\zeta}{\omega\odot A\zeta'}
     + \ip{N}{\mathrm d^2\Phi(\th)[\thh,\thh']},
\end{equation}
~~~

### TEX-EHTH-0221 — `docs/exact_hessian_two_hole.tex` 221–222 → 224–225, LaTeX equation line

before:

~~~
For an embedded submanifold, $\Hess f(X)[\zeta]=P_X\nabla^2 f\,\zeta+\W_X(\zeta,N)$ with the Weingarten map
$\W_X(\zeta,N)=P_X\,(D_\zeta P)\,N$, and $\ip{\W_X(\zeta,N)}{\zeta'}=\ip{N}{\mathrm{II}_X(\zeta,\zeta')}$
~~~

after:

~~~
For an embedded submanifold, $\Hess f(X)[\zeta]=P_X\nabla^2 f\,\zeta+\W_X(\zeta,N)$ with the Weingarten map
$\W_X(\zeta,N)=P_X\,(\mathrm d_\zeta P)\,N$, and $\ip{\W_X(\zeta,N)}{\zeta'}=\ip{N}{\mathrm{II}_X(\zeta,\zeta')}$
~~~

### TEX-EHTH-0225 — `docs/exact_hessian_two_hole.tex` 225–226 → 228–229, LaTeX equation line

before:

~~~
$\ip{N}{D^2\Phi[\thh,\thh']}=\ip{N}{\mathrm{II}_X(\zeta,\zeta')}$, which depends on $\zeta,\zeta'$ only. The first
term is $\ip{\zeta'}{P_XA^\T\omega^2A\zeta}=\ip{\omega A\zeta}{\omega A\zeta'}$ since $\zeta'$ is tangent.
~~~

after:

~~~
$\ip{N}{\mathrm d^2\Phi[\thh,\thh']}=\ip{N}{\mathrm{II}_X(\zeta,\zeta')}$, which depends on $\zeta,\zeta'$ only. The first
term is $\ip{\zeta'}{P_XA^\T\omega^2A\zeta}=\ip{\omega A\zeta}{\omega A\zeta'}$ since $\zeta'$ is tangent.
~~~

### TEX-EHTH-0230 — `docs/exact_hessian_two_hole.tex` 230–232 → 233–235, LaTeX equation line

before:

~~~
the gauge action ($D\Phi(\th)[V\th]=0$ for \emph{every} $\th$), satisfies, after differentiating that identity along
$\thh'$, $D^2\Phi[\thh_v,\thh']=-D\Phi[V\thh']\in T_X\M$, which $N$ annihilates. (At a slack frame
$\ker D\Phi(\th)$ is larger than the gauge orbit's tangent; Proposition~\ref{prop:weingarten} covers every lift.) No derivative of the frame (no SVD perturbation) enters --- this is the reason for the route
~~~

after:

~~~
the gauge action ($\mathrm d\Phi(\th)[V\th]=0$ for \emph{every} $\th$), satisfies, after differentiating that identity along
$\thh'$, $\mathrm d^2\Phi[\thh_v,\thh']=-\mathrm d\Phi[V\thh']\in T_X\M$, which $N$ annihilates. (At a slack frame
$\ker \mathrm d\Phi(\th)$ is larger than the gauge orbit's tangent; Proposition~\ref{prop:weingarten} covers every lift.) No derivative of the frame (no SVD perturbation) enters --- this is the reason for the route
~~~

### TEX-EHTH-0237 — `docs/exact_hessian_two_hole.tex` 237–241 → 240–244, LaTeX equation line

before:

~~~
\begin{equation}
  \label{eq:Kdef}
  \K_Z\,\thh \;=\; \frac{d}{ds}\nabla_\th\ip{Z}{\Phi(\th+s\thh)}\Big|_{s=0},
  \qquad \ip{\thh'}{\K_Z\thh}=\ip{Z}{D^2\Phi(\th)[\thh,\thh']},
\end{equation}
~~~

after:

~~~
\begin{equation}
  \label{eq:Kdef}
  \K_Z\,\thh \;=\; \frac{d}{ds}\nabla_\th\ip{Z}{\Phi(\th+s\thh)}\Big|_{s=0},
  \qquad \ip{\thh'}{\K_Z\thh}=\ip{Z}{\mathrm d^2\Phi(\th)[\thh,\thh']},
\end{equation}
~~~

### TEX-EHTH-0244 — `docs/exact_hessian_two_hole.tex` 244 → 247, LaTeX equation line

before:

~~~
(Figure~\ref{fig:holes}c). Two instances are needed, and $\ip{N}{D^2\Phi}$ is their difference:
~~~

after:

~~~
(Figure~\ref{fig:holes}c). Two instances are needed, and $\ip{N}{\mathrm d^2\Phi}$ is their difference:
~~~

### TEX-EHTH-0258 — `docs/exact_hessian_two_hole.tex` 258–263 → 261–266, LaTeX equation line

before:

~~~
\begin{equation}
  \label{eq:lagrangian}
  \Lambda(\th) = f(\Phi(\th)) - \ip{\grad f(X)}{\Phi(\th)}, \qquad
  \nabla_\th\Lambda = D\Phi^\T N = 0,\qquad
  \nabla^2_\th\Lambda = D\Phi^\T\,\Hess f(X)\,D\Phi,
\end{equation}
~~~

after:

~~~
\begin{equation}
  \label{eq:lagrangian}
  \Lambda(\th) = f(\Phi(\th)) - \ip{\grad f(X)}{\Phi(\th)}, \qquad
  \nabla_\th\Lambda = \mathrm d\Phi^\T N = 0,\qquad
  \nabla^2_\th\Lambda = \mathrm d\Phi^\T\,\Hess f(X)\,\mathrm d\Phi,
\end{equation}
~~~

### TEX-EHTH-0283 — `docs/exact_hessian_two_hole.tex` 283 → 286, LaTeX equation line

before:

~~~
$\ip{\grad f}{P_XD^2\Phi[\cdot,\cdot]}=O(\|\grad f\|)$ and depends on the lift. Level 2 is what the library ships;
~~~

after:

~~~
$\ip{\grad f}{P_X\mathrm d^2\Phi[\cdot,\cdot]}=O(\|\grad f\|)$ and depends on the lift. Level 2 is what the library ships;
~~~

### TEX-EHTH-0288 — `docs/exact_hessian_two_hole.tex` 288 → 291, LaTeX equation line

before:

~~~
\emph{second order}: $\frac{d^2}{dt^2}f(R_X(t\zeta))|_0=\ip{\Hess f[\zeta]}{\zeta}$. The archived design note
~~~

after:

~~~
\emph{second order}: $\frac{d^2}{dt^2}f(\mathcal{R}_X(t\zeta))|_0=\ip{\Hess f[\zeta]}{\zeta}$. The archived design note
~~~

### TEX-EHTH-0302 — `docs/exact_hessian_two_hole.tex` 302–307 → 305–310, LaTeX equation line

before:

~~~
\begin{equation}
  \label{eq:factors}
  T_i = \ip{P_{>i}}{Q_{>i}}\in\R^{r^L_i\times r^R_i}\ (i<d),\qquad
  H_i = P_i\times_3T_i\ (H_d=P_d),\qquad
  S_i^\T = \ip{O_i}{H_i}_{\{1,3\}}\in\R^{n^D_i\times n_i},
\end{equation}
~~~

after:

~~~
\begin{equation}
  \label{eq:factors}
  T_i = \ip{L_{>i}}{R_{>i}}\in\R^{r^L_i\times r^R_i}\ (i<d),\qquad
  \bar H_i = L_i\times_3T_i\ (\bar H_d=L_d),\qquad
  S_i^\T = \ip{U_i}{\bar H_i}_{\{1,3\}}\in\R^{n^U_i\times n_i},
\end{equation}
~~~

### TEX-EHTH-0309 — `docs/exact_hessian_two_hole.tex` 309–310 → 312–313, LaTeX equation line

before:

~~~
\code{tt_zipper_right_to_left}). Then $H_i=O_i\times_2S_i^\T$, the centers are the point's own core with the
orthogonality center at $i$ ($X=P_{<i}H_iQ_{>i}\times U$), $P_{>i}=T_i\,Q_{>i}$ (the left-canonical tail is
~~~

after:

~~~
\code{tt_zipper_right_to_left}). Then $\bar H_i=U_i\times_2S_i^\T$, the centers are the point's own core with the
orthogonality center at $i$ ($X=L_{<i}\bar H_iR_{>i}\times D$), $L_{>i}=T_i\,R_{>i}$ (the left-canonical tail is
~~~

### TEX-EHTH-0316 — `docs/exact_hessian_two_hole.tex` 316 → 319, LaTeX equation line

before:

~~~
Define $\Lift v=\thh=(\dot U_i,\dot G_i)$, a variation of the base point $\th=(U,P)$, by
~~~

after:

~~~
Define $\Lift v=\thh=(\dot B_i,\dot G_i)$, a variation of the base point $\th=(D,L)$, by
~~~

### TEX-EHTH-0317 — `docs/exact_hessian_two_hole.tex` 317–321 → 320–324, LaTeX equation line

before:

~~~
\begin{equation}
  \label{eq:lift}
  \dot U_i = (S_i^\T)^{+}\,\delta U_i,\qquad
  \dot G_i = \delta G_i\times_3 T_i^{+}\ (i<d),\qquad \dot G_d=\delta G_d .
\end{equation}
~~~

after:

~~~
\begin{equation}
  \label{eq:lift}
  \dot B_i = (S_i^\T)^{+}\,V_i,\qquad
  \dot G_i = H_i\times_3 T_i^{+}\ (i<d),\qquad \dot G_d=H_d .
\end{equation}
~~~

### TEX-EHTH-0322 — `docs/exact_hessian_two_hole.tex` 322 → 325, LaTeX equation line

before:

~~~
If every $S_i^\T$ has full row rank and every $T_i$ full column rank, then $D\Phi(\th)[\Lift v]=\Emap v$ for
~~~

after:

~~~
If every $S_i^\T$ has full row rank and every $T_i$ full column rank, then $\mathrm d\Phi(\th)[\Lift v]=\Emap v$ for
~~~

### TEX-EHTH-0327 — `docs/exact_hessian_two_hole.tex` 327–329 → 330–332, LaTeX equation line

before:

~~~
TT slot: $P_{<i}\dot G_iP_{>i}=P_{<i}\,(\dot G_iT_i)\,Q_{>i}=P_{<i}\,\delta G_i T_i^+T_i\,Q_{>i}$, and $T_i^+T_i=I$
for full column rank. Tucker slot: replacing $U_i$ by $\dot U_i$ in $X=P_{<i}H_iQ_{>i}\times U$ gives
$P_{<i}(H_i\times_2\dot U_i)Q_{>i}=P_{<i}\big(O_i\times_2(S_i^\T\dot U_i)\big)Q_{>i}$, and $S_i^\T(S_i^\T)^+=I$
~~~

after:

~~~
TT slot: $L_{<i}\dot G_iL_{>i}=L_{<i}\,(\dot G_iT_i)\,R_{>i}=L_{<i}\,H_i T_i^+T_i\,R_{>i}$, and $T_i^+T_i=I$
for full column rank. Tucker slot: replacing $D_i$ by $\dot B_i$ in $X=L_{<i}\bar H_iR_{>i}\times D$ gives
$L_{<i}(\bar H_i\times_2\dot B_i)R_{>i}=L_{<i}\big(U_i\times_2(S_i^\T\dot B_i)\big)R_{>i}$, and $S_i^\T(S_i^\T)^+=I$
~~~

### TEX-EHTH-0338 — `docs/exact_hessian_two_hole.tex` 338–379 → 341–382, LaTeX table or figure

before:

~~~
\begin{figure}[t]
\centering
\begin{tikzpicture}[x=10mm,y=10mm]
  % (a) TT slot
  \begin{scope}
    \node[ttc] (a1) at (0,0) {$P_{i-1}$}; \node[dotc] (a2) at (1.2,0) {$\delta G_i$};
    \node[ttc] (a3) at (2.4,0) {$Q_{i+1}$}; \node[ttc] (a4) at (3.6,0) {$Q_{i+2}$};
    \draw[bnd] (-0.7,0)--(a1)--(a2)--(a3)--(a4)--(4.25,0);
    \foreach \x in {0,1.2,2.4,3.6} \draw[bnd] (\x,-0.38)--(\x,-0.8);
    \node at (4.75,0) {$=$};
    \node[ttc] (b1) at (5.8,0) {$P_{i-1}$}; \node[dotc] (b2) at (6.9,0) {$\dot G_i$};
    \node[matn] (bt) at (7.85,0) {$T_i$};
    \node[ttc] (b3) at (8.8,0) {$Q_{i+1}$}; \node[ttc] (b4) at (10.0,0) {$Q_{i+2}$};
    \draw[bnd] (5.2,0)--(b1)--(b2)--(bt)--(b3)--(b4)--(10.7,0);
    \foreach \x in {5.8,6.9,8.8,10.0} \draw[bnd] (\x,-0.38)--(\x,-0.8);
    \draw[decorate,decoration={brace,amplitude=4pt}] (7.6,0.45)--(10.4,0.45)
      node[midway,above=3pt,font=\scriptsize]{$=P_{i+1}P_{i+2}=P_{>i}$};
    \node[font=\small,anchor=west] at (-0.7,-1.25) {(a) TT slot: $\dot G_i=\delta G_i\times_3T_i^{+\eps}$, exact when $T_i^+T_i=I$ (TT slack: $T_i$ tall)};
  \end{scope}
  % (b) Tucker slot
  \begin{scope}[yshift=-24mm]
    \node[ttc] (c1) at (0,0) {$P_{i-1}$}; \node[ttc] (c2) at (1.2,0) {$O_i$}; \node[ttc] (c3) at (2.4,0) {$Q_{i+1}$};
    \node[dotc] (cu) at (1.2,-1.1) {$\delta U_i$};
    \draw[bnd] (-0.7,0)--(c1)--(c2)--(c3)--(3.05,0); \draw[bnd] (c2)--(cu)--(1.2,-1.75);
    \draw[bnd] (0,-0.38)--(0,-0.8); \draw[bnd] (2.4,-0.38)--(2.4,-0.8);
    \node at (3.55,0) {$=$};
    \node[ttc] (d1) at (4.5,0) {$P_{i-1}$}; \node[ttc] (d2) at (5.7,0) {$O_i$}; \node[ttc] (d3) at (6.9,0) {$Q_{i+1}$};
    \node[matn] (ds) at (5.7,-0.95) {$S_i^\T$};
    \node[dotc] (du) at (5.7,-1.9) {$\dot U_i$};
    \draw[bnd] (3.95,0)--(d1)--(d2)--(d3)--(7.6,0); \draw[bnd] (d2)--(ds)--(du)--(5.7,-2.55);
    \draw[bnd] (4.5,-0.38)--(4.5,-0.8); \draw[bnd] (6.9,-0.38)--(6.9,-0.8);
    \draw[dashed,rounded corners=3pt,gray] (5.2,0.45)--(6.25,0.45)--(6.25,-1.3)--(5.2,-1.3)--cycle;
    \node[font=\scriptsize,anchor=west] at (6.3,-1.15) {$=H_i$ (the center)};
    \node[font=\small,anchor=west] at (-0.7,-2.9) {(b) Tucker slot: $\dot U_i=(S_i^\T)^{+\eps}\delta U_i$, exact when $S_i^\T(S_i^\T)^+=I$ (Tucker slack: $S_i^\T$ wide)};
  \end{scope}
\end{tikzpicture}
\caption{The lift, slot by slot (red: the variation). (a) The frame's TT slot $P_{<i}\,\delta G_i\,Q_{>i}$ equals the
raw slot $P_{<i}\,\dot G_i\,P_{>i}$ because the left-canonical tail is $P_{>i}=T_iQ_{>i}$. (b) The frame's Tucker
slot pairs $\delta U_i$ with the orthonormal $O_i$; the raw slot pairs $\dot U_i$ with the center
$H_i=O_i\times_2S_i^\T$. The pseudo-inverses are the only place an inverse singular value enters the Hessian.}
\label{fig:lift}
\end{figure}
~~~

after:

~~~
\begin{figure}[t]
\centering
\begin{tikzpicture}[x=10mm,y=10mm]
  % (a) TT slot
  \begin{scope}
    \node[ttc] (a1) at (0,0) {$L_{i-1}$}; \node[dotc] (a2) at (1.2,0) {$H_i$};
    \node[ttc] (a3) at (2.4,0) {$R_{i+1}$}; \node[ttc] (a4) at (3.6,0) {$R_{i+2}$};
    \draw[bnd] (-0.7,0)--(a1)--(a2)--(a3)--(a4)--(4.25,0);
    \foreach \x in {0,1.2,2.4,3.6} \draw[bnd] (\x,-0.38)--(\x,-0.8);
    \node at (4.75,0) {$=$};
    \node[ttc] (b1) at (5.8,0) {$L_{i-1}$}; \node[dotc] (b2) at (6.9,0) {$\dot G_i$};
    \node[matn] (bt) at (7.85,0) {$T_i$};
    \node[ttc] (b3) at (8.8,0) {$R_{i+1}$}; \node[ttc] (b4) at (10.0,0) {$R_{i+2}$};
    \draw[bnd] (5.2,0)--(b1)--(b2)--(bt)--(b3)--(b4)--(10.7,0);
    \foreach \x in {5.8,6.9,8.8,10.0} \draw[bnd] (\x,-0.38)--(\x,-0.8);
    \draw[decorate,decoration={brace,amplitude=4pt}] (7.6,0.45)--(10.4,0.45)
      node[midway,above=3pt,font=\scriptsize]{$=L_{i+1}L_{i+2}=L_{>i}$};
    \node[font=\small,anchor=west] at (-0.7,-1.25) {(a) TT slot: $\dot G_i=H_i\times_3T_i^{+\eps}$, exact when $T_i^+T_i=I$ (TT slack: $T_i$ tall)};
  \end{scope}
  % (b) Tucker slot
  \begin{scope}[yshift=-24mm]
    \node[ttc] (c1) at (0,0) {$L_{i-1}$}; \node[ttc] (c2) at (1.2,0) {$U_i$}; \node[ttc] (c3) at (2.4,0) {$R_{i+1}$};
    \node[dotc] (cu) at (1.2,-1.1) {$V_i$};
    \draw[bnd] (-0.7,0)--(c1)--(c2)--(c3)--(3.05,0); \draw[bnd] (c2)--(cu)--(1.2,-1.75);
    \draw[bnd] (0,-0.38)--(0,-0.8); \draw[bnd] (2.4,-0.38)--(2.4,-0.8);
    \node at (3.55,0) {$=$};
    \node[ttc] (d1) at (4.5,0) {$L_{i-1}$}; \node[ttc] (d2) at (5.7,0) {$U_i$}; \node[ttc] (d3) at (6.9,0) {$R_{i+1}$};
    \node[matn] (ds) at (5.7,-0.95) {$S_i^\T$};
    \node[dotc] (du) at (5.7,-1.9) {$\dot B_i$};
    \draw[bnd] (3.95,0)--(d1)--(d2)--(d3)--(7.6,0); \draw[bnd] (d2)--(ds)--(du)--(5.7,-2.55);
    \draw[bnd] (4.5,-0.38)--(4.5,-0.8); \draw[bnd] (6.9,-0.38)--(6.9,-0.8);
    \draw[dashed,rounded corners=3pt,gray] (5.2,0.45)--(6.25,0.45)--(6.25,-1.3)--(5.2,-1.3)--cycle;
    \node[font=\scriptsize,anchor=west] at (6.3,-1.15) {$=\bar H_i$ (the center)};
    \node[font=\small,anchor=west] at (-0.7,-2.9) {(b) Tucker slot: $\dot B_i=(S_i^\T)^{+\eps}V_i$, exact when $S_i^\T(S_i^\T)^+=I$ (Tucker slack: $S_i^\T$ wide)};
  \end{scope}
\end{tikzpicture}
\caption{The lift, slot by slot (red: the variation). (a) The frame's TT slot $L_{<i}\,H_i\,R_{>i}$ equals the
raw slot $L_{<i}\,\dot G_i\,L_{>i}$ because the left-canonical tail is $L_{>i}=T_iR_{>i}$. (b) The frame's Tucker
slot pairs $V_i$ with the orthonormal $U_i$; the raw slot pairs $\dot B_i$ with the center
$\bar H_i=U_i\times_2S_i^\T$. The pseudo-inverses are the only place an inverse singular value enters the Hessian.}
\label{fig:lift}
\end{figure}
~~~

### TEX-EHTH-0382 — `docs/exact_hessian_two_hole.tex` 382 → 385, LaTeX equation line

before:

~~~
$R_k^{-1}$ \cite{PB20}), so it is damped by the Tikhonov filter
~~~

after:

~~~
$R_k^{-1}$ \cite{PB20}, PB20's notation), so it is damped by the Tikhonov filter
~~~

### TEX-EHTH-0383 — `docs/exact_hessian_two_hole.tex` 383–387 → 386–390, LaTeX equation line

unchanged: R5: an SVD's V f(Sigma) U^T (§1.5 item 2)

### TEX-EHTH-0394 — `docs/exact_hessian_two_hole.tex` 394 → 397, LaTeX equation line

before:

~~~
$\sim10^{-17}$) needs no special case: there the lift satisfies $D\Phi[\Lift v]=\Emap(Q_{\mathrm c}v)$ with $Q_{\mathrm c}$ the
~~~

after:

~~~
$\sim10^{-17}$) needs no special case: there the lift satisfies $\mathrm d\Phi[\Lift v]=\Emap(Q_{\mathrm c}v)$ with $Q_{\mathrm c}$ the
~~~

### TEX-EHTH-0399 — `docs/exact_hessian_two_hole.tex` 399–400 → 402–403, LaTeX equation line

before:

~~~
\paragraph{The transpose.} $\Lift^\T y$ maps raw core variations $y=(Y^U_i,Y^G_i)$ to \emph{ungauged}
coordinates, $\delta U_i=((S_i^\T)^{+\eps})^\T Y^U_i$, $\delta G_i=Y^G_i\times_3(T_i^{+\eps})^\T$,
~~~

after:

~~~
\paragraph{The transpose.} $\Lift^\T y$ maps raw core variations $y=(Y^B_i,Y^G_i)$ to \emph{ungauged}
coordinates, $V_i=((S_i^\T)^{+\eps})^\T Y^B_i$, $H_i=Y^G_i\times_3(T_i^{+\eps})^\T$,
~~~

### TEX-EHTH-0404 — `docs/exact_hessian_two_hole.tex` 404–425 → 407–428, LaTeX equation line

before:

~~~
\begin{algorithm}[t]
\caption{The lift $\thh=\Lift_\eps v$ and its adjoint $\Lift_\eps^\T$ (per-frame factors once, per matvec two
contractions per mode)}
\label{alg:lift}
\begin{algorithmic}[1]
\Require orthogonal frame $(U,O,P,Q)$; relative damping $c\ge0$; coordinates $v=(\delta U_i,\delta G_i)$ (stack $\mathtt{K+C}$)
\Ensure raw directions $\thh=(\dot U_i,\dot G_i)$ of $\th=(U,P)$
\State $T_i\gets\ip{P_{>i}}{Q_{>i}}$ (zipper);\ $H_i\gets P_i\times_3T_i$;\ $S_i^\T\gets\ip{O_i}{H_i}_{\{1,3\}}$
   \Comment{\code{fv_center_factors}}
\State $\eps\gets c\,\|S_1^\T\|_F$;\ $(S_i^\T)^{+\eps},\ T_i^{+\eps}\gets$ one thin SVD each, filter \eqref{eq:filter}
   \Comment{\code{fv_lift_factors}}
\For{$i=1,\dots,d$} \Comment{\code{tv_corewise_lift}}
  \State $\dot U_i\gets(S_i^\T)^{+\eps}\,\delta U_i$ \Comment{\texttt{'Cux,KCxn->KCun'}}
  \State $\dot G_i\gets\delta G_i\times_3T_i^{+\eps}$ if $i<d$, else $\dot G_d\gets\delta G_d$ \Comment{\texttt{'KCaib,Cbc->KCaic'}}
\EndFor
\State adjoint: $\delta U_i\gets((S_i^\T)^{+\eps})^\T Y^U_i$,\ $\delta G_i\gets Y^G_i\times_3(T_i^{+\eps})^\T$
   \Comment{\code{tv_corewise_lift_transpose}}
\Impl{\code{fv_operations.fv_center_factors}, \code{fv_operations.fv_lift_factors} (\code{LiftFactors}),
\code{linalg.damped_pseudo_inverse}, \code{linalg.filtered_reciprocal}, \code{tv_operations.tv_corewise_lift[_transpose]};
uniform: \code{ufv_operations.ufv_lift_factors}, \code{utv_operations.utv_corewise_lift[_transpose]}.}
\end{algorithmic}
\end{algorithm}
~~~

after:

~~~
\begin{algorithm}[t]
\caption{The lift $\thh=\Lift_\eps v$ and its adjoint $\Lift_\eps^\T$ (per-frame factors once, per matvec two
contractions per mode)}
\label{alg:lift}
\begin{algorithmic}[1]
\Require orthogonal frame $(D,U,L,R)$; relative damping $c\ge0$; coordinates $v=(V_i,H_i)$ (stack $\mathtt{K+C}$)
\Ensure raw directions $\thh=(\dot B_i,\dot G_i)$ of $\th=(D,L)$
\State $T_i\gets\ip{L_{>i}}{R_{>i}}$ (zipper);\ $\bar H_i\gets L_i\times_3T_i$;\ $S_i^\T\gets\ip{U_i}{\bar H_i}_{\{1,3\}}$
   \Comment{\code{fv_center_factors}}
\State $\eps\gets c\,\|S_1^\T\|_F$;\ $(S_i^\T)^{+\eps},\ T_i^{+\eps}\gets$ one thin SVD each, filter \eqref{eq:filter}
   \Comment{\code{fv_lift_factors}}
\For{$i=1,\dots,d$} \Comment{\code{tv_corewise_lift}}
  \State $\dot B_i\gets(S_i^\T)^{+\eps}\,V_i$ \Comment{\texttt{'Cux,KCxn->KCun'}}
  \State $\dot G_i\gets H_i\times_3T_i^{+\eps}$ if $i<d$, else $\dot G_d\gets H_d$ \Comment{\texttt{'KCaib,Cbc->KCaic'}}
\EndFor
\State adjoint: $V_i\gets((S_i^\T)^{+\eps})^\T Y^B_i$,\ $H_i\gets Y^G_i\times_3(T_i^{+\eps})^\T$
   \Comment{\code{tv_corewise_lift_transpose}}
\Impl{\code{fv_operations.fv_center_factors}, \code{fv_operations.fv_lift_factors} (\code{LiftFactors}),
\code{linalg.damped_pseudo_inverse}, \code{linalg.filtered_reciprocal}, \code{tv_operations.tv_corewise_lift[_transpose]};
uniform: \code{ufv_operations.ufv_lift_factors}, \code{utv_operations.utv_corewise_lift[_transpose]}.}
\end{algorithmic}
\end{algorithm}
~~~

### TEX-EHTH-0478 — `docs/exact_hessian_two_hole.tex` 478 → 481, LaTeX equation line

before:

~~~
At the \emph{corewise frame} $(U,G,G,G)$ of raw cores $\th=(U,G)$ (\code{t3_corewise_frame}) the tangent
~~~

after:

~~~
At the \emph{corewise frame} $(B,G,G,G)$ of raw cores $\th=(B,G)$ (\code{t3_corewise_frame}) the tangent
~~~

### TEX-EHTH-0480 — `docs/exact_hessian_two_hole.tex` 480 → 483, LaTeX equation line

before:

~~~
substitution $P,Q,O\to G$; \code{t3_probe_corewise_transpose}). With edge variables on bonds $0,\dots,d$ (core $i$
~~~

after:

~~~
substitution $U,L,R\to G$; \code{t3_probe_corewise_transpose}). With edge variables on bonds $0,\dots,d$ (core $i$
~~~

### TEX-EHTH-0482 — `docs/exact_hessian_two_hole.tex` 482–495 → 485–498, LaTeX equation line

before:

~~~
\begin{align}
  \label{eq:fwd}
  &\xi_i=U_iw_i,\quad \mu_i=\mu_{i-1}G_i(\xi_i),\ \mu_0=1,\quad \nu_{i-1}=G_i(\xi_i)\nu_i,\ \nu_d=1,\quad
   \eta_i=\mu_{i-1}G_i\nu_i, \\
  \label{eq:adj}
  &\ddet_i=U_ic_i,\quad \tilde\tau_i=\tilde\tau_{i-1}G_i(\xi_i)+\mu_{i-1}G_i(\ddet_i),\ \tilde\tau_0=0,\quad
   \tilde\sigma_{i-1}=G_i(\xi_i)\tilde\sigma_i+G_i(\ddet_i)\nu_i,\ \tilde\sigma_d=0, \\
  \label{eq:dxt}
  &\dxt_i=\tilde\tau_{i-1}G_i\nu_i+\mu_{i-1}G_i\tilde\sigma_i, \\
  \label{eq:asm}
  &\delta\tilde U_i=\textstyle\sum_{\mathtt W}\big(\eta_i\otimes c_i+\dxt_i\otimes w_i\big),\qquad
   \delta\tilde G_i=\sum_{\mathtt W}\big(\mu_{i-1}\otimes\xi_i\otimes\tilde\sigma_i
   +\tilde\tau_{i-1}\otimes\xi_i\otimes\nu_i+\mu_{i-1}\otimes\ddet_i\otimes\nu_i\big).
\end{align}
~~~

after:

~~~
\begin{align}
  \label{eq:fwd}
  &\xi_i=B_iw_i,\quad \mu_i=\mu_{i-1}G_i(\xi_i),\ \mu_0=1,\quad \nu_{i-1}=G_i(\xi_i)\nu_i,\ \nu_d=1,\quad
   \eta_i=\mu_{i-1}G_i\nu_i, \\
  \label{eq:adj}
  &\ddet_i=B_ic_i,\quad \tilde\tau_i=\tilde\tau_{i-1}G_i(\xi_i)+\mu_{i-1}G_i(\ddet_i),\ \tilde\tau_0=0,\quad
   \tilde\sigma_{i-1}=G_i(\xi_i)\tilde\sigma_i+G_i(\ddet_i)\nu_i,\ \tilde\sigma_d=0, \\
  \label{eq:dxt}
  &\dxt_i=\tilde\tau_{i-1}G_i\nu_i+\mu_{i-1}G_i\tilde\sigma_i, \\
  \label{eq:asm}
  &\tilde V_i=\textstyle\sum_{\mathtt W}\big(\eta_i\otimes c_i+\dxt_i\otimes w_i\big),\qquad
   \tilde H_i=\sum_{\mathtt W}\big(\mu_{i-1}\otimes\xi_i\otimes\tilde\sigma_i
   +\tilde\tau_{i-1}\otimes\xi_i\otimes\nu_i+\mu_{i-1}\otimes\ddet_i\otimes\nu_i\big).
\end{align}
~~~

### TEX-EHTH-0500 — `docs/exact_hessian_two_hole.tex` 500 → 503, LaTeX equation line

before:

~~~
\xi_i\otimes\hat\sigma_i$, $\delta\tilde U_i=\sum_{\mathtt W}\delta\hat\xi_i\otimes w_i$ with
~~~

after:

~~~
\xi_i\otimes\hat\sigma_i$, $\tilde V_i=\sum_{\mathtt W}\delta\hat\xi_i\otimes w_i$ with
~~~

### TEX-EHTH-0503 — `docs/exact_hessian_two_hole.tex` 503–521 → 506–524, LaTeX equation line

before:

~~~
\begin{algorithm}[t]
\caption{The one-hole sweep (reference): the corewise probe transpose $\nabla_\th\ip{c}{A\Phi(\th)}$}
\label{alg:onehole}
\begin{algorithmic}[1]
\Require $\th=(U_i,G_i)$; probe vectors $w_i$ (stack $\mathtt W$); weighted residuals $c_i=\omega_i^2\odot r_i$
\Ensure $(\delta\tilde U_i,\delta\tilde G_i)$, $\mathtt W$ summed
\State $\xi_i,\mu_i,\nu_i,\eta_i$ by \eqref{eq:fwd} \Comment{frame sweep (T4S Alg.~5); \code{tv_precompute_probe_frame_sweep}}
\State $\ddet_i\gets U_ic_i$ \Comment{\code{compute_deta_tilde}}
\For{$i=1,\dots,d$} $\tilde\tau_i\gets\tilde\tau_{i-1}G_i(\xi_i)+\mu_{i-1}G_i(\ddet_i)$ \Comment{\code{compute_tau_tilde}}
\EndFor
\For{$i=d,\dots,1$} $\tilde\sigma_{i-1}\gets G_i(\xi_i)\tilde\sigma_i+G_i(\ddet_i)\nu_i$ \Comment{\code{compute_sigma_tilde}}
\EndFor
\State $\dxt_i\gets\tilde\tau_{i-1}G_i\nu_i+\mu_{i-1}G_i\tilde\sigma_i$ \Comment{\code{compute_dxi_tilde}}
\State $\delta\tilde U_i,\ \delta\tilde G_i$ by \eqref{eq:asm} \Comment{\code{assemble_tucker_variations}, \code{assemble_tt_variations}}
\Impl{\code{probing.t3_probe_corewise_transpose} = \code{probing.tv_probe_transpose} at \code{fv_conversions.t3_corewise_frame}
(T4S \S6.2.3, Algorithm~8; \S6.3). Jets: \code{sampling_derivatives.t3_probe_corewise_derivatives_transpose},
\code{tv_probe_transpose_derivatives_from_sweep}.}
\end{algorithmic}
\end{algorithm}
~~~

after:

~~~
\begin{algorithm}[t]
\caption{The one-hole sweep (reference): the corewise probe transpose $\nabla_\th\ip{c}{A\Phi(\th)}$}
\label{alg:onehole}
\begin{algorithmic}[1]
\Require $\th=(B_i,G_i)$; probe vectors $w_i$ (stack $\mathtt W$); weighted residuals $c_i=\omega_i^2\odot r_i$
\Ensure $(\tilde V_i,\tilde H_i)$, $\mathtt W$ summed
\State $\xi_i,\mu_i,\nu_i,\eta_i$ by \eqref{eq:fwd} \Comment{frame sweep (T4S Alg.~5); \code{tv_precompute_probe_frame_sweep}}
\State $\ddet_i\gets B_ic_i$ \Comment{\code{compute_deta_tilde}}
\For{$i=1,\dots,d$} $\tilde\tau_i\gets\tilde\tau_{i-1}G_i(\xi_i)+\mu_{i-1}G_i(\ddet_i)$ \Comment{\code{compute_tau_tilde}}
\EndFor
\For{$i=d,\dots,1$} $\tilde\sigma_{i-1}\gets G_i(\xi_i)\tilde\sigma_i+G_i(\ddet_i)\nu_i$ \Comment{\code{compute_sigma_tilde}}
\EndFor
\State $\dxt_i\gets\tilde\tau_{i-1}G_i\nu_i+\mu_{i-1}G_i\tilde\sigma_i$ \Comment{\code{compute_dxi_tilde}}
\State $\tilde V_i,\ \tilde H_i$ by \eqref{eq:asm} \Comment{\code{assemble_tucker_variations}, \code{assemble_tt_variations}}
\Impl{\code{probing.t3_probe_corewise_transpose} = \code{probing.tv_probe_transpose} at \code{fv_conversions.t3_corewise_frame}
(T4S \S6.2.3, Algorithm~8; \S6.3). Jets: \code{sampling_derivatives.t3_probe_corewise_derivatives_transpose},
\code{tv_probe_transpose_derivatives_from_sweep}.}
\end{algorithmic}
\end{algorithm}
~~~

### TEX-EHTH-0527 — `docs/exact_hessian_two_hole.tex` 527 → 530, LaTeX equation line

before:

~~~
derivative along $\thh=(\dot U_i,\dot G_i)$, i.e.\ the \emph{product rule} applied to every line of
~~~

after:

~~~
derivative along $\thh=(\dot B_i,\dot G_i)$, i.e.\ the \emph{product rule} applied to every line of
~~~

### TEX-EHTH-0535 — `docs/exact_hessian_two_hole.tex` 535 → 538, LaTeX equation line

before:

~~~
  $\dot\xi_i=\dot U_iw_i$,\
~~~

after:

~~~
  $\dot\xi_i=\dot B_iw_i$,\
~~~

### TEX-EHTH-0539 — `docs/exact_hessian_two_hole.tex` 539 → 542, LaTeX equation line

before:

~~~
  --- exactly T4S Algorithm~7 ($\sigma,\tau,\delta\eta$) with $Q=O=P=G$ and variation $(\dot U,\dot G)$. The
~~~

after:

~~~
  --- exactly T4S Algorithm~7 ($\sigma,\tau,\delta\eta$) with $U=L=R=G$ and variation $(\dot B,\dot G)$. The
~~~

### TEX-EHTH-0543 — `docs/exact_hessian_two_hole.tex` 543 → 546, LaTeX equation line

before:

~~~
  $\frac{d}{ds}\delta\tilde U_i=\sum_{\mathtt W}(\dot\eta_i\otimes c_i+\dot{\dxt}_i\otimes w_i)$, and $\delta\tilde G_i$
~~~

after:

~~~
  $\frac{d}{ds}\tilde V_i=\sum_{\mathtt W}(\dot\eta_i\otimes c_i+\dot{\dxt}_i\otimes w_i)$, and $\tilde H_i$
~~~

### TEX-EHTH-0545 — `docs/exact_hessian_two_hole.tex` 545 → 548, LaTeX equation line

before:

~~~
  the manifold frame $(U,O,P,Q)$ and the corewise frame $(U,P,P,P)$ of the base point), $\nu$ is re-swept through
~~~

after:

~~~
  the manifold frame $(D,U,L,R)$ and the corewise frame $(D,L,L,L)$ of the base point), $\nu$ is re-swept through
~~~

### TEX-EHTH-0551 — `docs/exact_hessian_two_hole.tex` 551 → 554, LaTeX equation line

before:

~~~
\paragraph{The block-sum oracle.} Since $D\Phi(\th)[\thh]=\Phi(Y)$ for the doubled-rank T3 $Y$ that
~~~

after:

~~~
\paragraph{The block-sum oracle.} Since $\mathrm d\Phi(\th)[\thh]=\Phi(Y)$ for the doubled-rank T3 $Y$ that
~~~

### TEX-EHTH-0554 — `docs/exact_hessian_two_hole.tex` 554–557 → 557–560, LaTeX equation line

unchanged: R5: B^T is the block-sum operator, not a T3's core (left; R5 handoff Q-EHTH-2)

### TEX-EHTH-0558 — `docs/exact_hessian_two_hole.tex` 558 → 561, LaTeX equation line

unchanged: R5: B^T, the block-sum operator (as TEX-EHTH-0554)

### TEX-EHTH-0573 — `docs/exact_hessian_two_hole.tex` 573–576 → 576–579, LaTeX equation line

before:

~~~
Write the down-shift $(\Dsh x)^{(t)}=t\,x^{(t-1)}$ and its adjoint, the up-shift $(\Ush x)^{(s)}=(s+1)\,x^{(s+1)}$
($x^{(m+1)}=0$), so that $\Dsh^r x=r!\binom{t}{r}x^{(t-r)}$ and $(\Ush^rx)^{(s)}=r!\binom{r+s}{r}x^{(r+s)}$
(\code{_shift_down}, \code{_shift_up}). The input jets are affine, $\xi_i=(U_iw_i,U_ip_i)$,
$\dot\xi_i=(\dot U_iw_i,\dot U_ip_i)$ (\code{build_input_jets}). Every jet product is one of four forms:
~~~

after:

~~~
Write the shift $\Sh$, $(\Sh x)^{(t)}=t\,x^{(t-1)}$, and its adjoint $\Sh^\T$, $(\Sh^\T x)^{(s)}=(s+1)\,x^{(s+1)}$
($x^{(m+1)}=0$), so that $\Sh^r x=r!\binom{t}{r}x^{(t-r)}$ and $((\Sh^\T)^rx)^{(s)}=r!\binom{r+s}{r}x^{(r+s)}$
(\code{_shift_down}, \code{_shift_up}). The input jets are affine, $\xi_i=(B_iw_i,B_ip_i)$,
$\dot\xi_i=(\dot B_iw_i,\dot B_ip_i)$ (\code{build_input_jets}). Every jet product is one of four forms:
~~~

### TEX-EHTH-0577 — `docs/exact_hessian_two_hole.tex` 577–584 → 580–587, LaTeX equation line

before:

~~~
\begin{align}
  \label{eq:band}
  &\text{banded (one affine factor } b\text{):} && (a\star b)^{(t)}=a^{(t)}b^{(0)}+(\Dsh a)^{(t)}b^{(1)},
     && (c\star^\T b)^{(s)}=c^{(s)}b^{(0)}+(\Ush c)^{(s)}b^{(1)},\\
  \label{eq:conv}
  &\text{full (both jets):} && (a\ast b)^{(t)}=\textstyle\sum_r\frac{a^{(r)}}{r!}(\Dsh^r b)^{(t)},
     && (a\ast^\T z)^{(s)}=\textstyle\sum_r\frac{a^{(r)}}{r!}(\Ush^r z)^{(s)},
\end{align}
~~~

after:

~~~
\begin{align}
  \label{eq:band}
  &\text{banded (one affine factor } b\text{):} && (a\star b)^{(t)}=a^{(t)}b^{(0)}+(\Sh a)^{(t)}b^{(1)},
     && (c\star^\T b)^{(s)}=c^{(s)}b^{(0)}+(\Sh^\T c)^{(s)}b^{(1)},\\
  \label{eq:conv}
  &\text{full (both jets):} && (a\ast b)^{(t)}=\textstyle\sum_r\frac{a^{(r)}}{r!}(\Sh^r b)^{(t)},
     && (a\ast^\T z)^{(s)}=\textstyle\sum_r\frac{a^{(r)}}{r!}((\Sh^\T)^r z)^{(s)},
\end{align}
~~~

### TEX-EHTH-0600 — `docs/exact_hessian_two_hole.tex` 600 → 603, LaTeX equation line

before:

~~~
and its product rule along $(\dot G,\dot\xi,\dot\mu,\dot{\ddet})$, with $\dot{\ddet}_i=\dot U_ic_i$, is the dotted step
~~~

after:

~~~
and its product rule along $(\dot G,\dot\xi,\dot\mu,\dot{\ddet})$, with $\dot{\ddet}_i=\dot B_ic_i$, is the dotted step
~~~

### TEX-EHTH-0612 — `docs/exact_hessian_two_hole.tex` 612 → 615, LaTeX equation line

before:

~~~
pairs them with $(w,p)$), and only those are formed: with $[\Ush^0x]=x$, $[\Ush^1x]=\Ush x$,
~~~

after:

~~~
pairs them with $(w,p)$), and only those are formed: with $[(\Sh^\T)^0x]=x$, $[(\Sh^\T)^1x]=\Sh^\T x$,
~~~

### TEX-EHTH-0613 — `docs/exact_hessian_two_hole.tex` 613–617 → 616–620, LaTeX equation line

before:

~~~
\begin{equation}
  \label{eq:dxitilde}
  \dxt_i^{(u)}=\sum_j\mu_{i-1}^{(j)}G_i\,[\Ush^u\tilde\sigma_i]^{(j)}+[\Ush^u\tilde\tau_{i-1}]^{(j)}G_i\,\nu_i^{(j)},
  \qquad u\in\{0,1\},
\end{equation}
~~~

after:

~~~
\begin{equation}
  \label{eq:dxitilde}
  \dxt_i^{(u)}=\sum_j\mu_{i-1}^{(j)}G_i\,[(\Sh^\T)^u\tilde\sigma_i]^{(j)}+[(\Sh^\T)^u\tilde\tau_{i-1}]^{(j)}G_i\,\nu_i^{(j)},
  \qquad u\in\{0,1\},
\end{equation}
~~~

### TEX-EHTH-0619 — `docs/exact_hessian_two_hole.tex` 619 → 622, LaTeX equation line

before:

~~~
$d(\mu G)[\Ush^u\tilde\sigma]+\mu G[\Ush^u\dot{\tilde\sigma}]+[\Ush^u\dot{\tilde\tau}]G\nu+[\Ush^u\tilde\tau]d(G\nu)$,
~~~

after:

~~~
$d(\mu G)[(\Sh^\T)^u\tilde\sigma]+\mu G[(\Sh^\T)^u\dot{\tilde\sigma}]+[(\Sh^\T)^u\dot{\tilde\tau}]G\nu+[(\Sh^\T)^u\tilde\tau]d(G\nu)$,
~~~

### TEX-EHTH-0622 — `docs/exact_hessian_two_hole.tex` 622–643 → 625–646, LaTeX equation line

before:

~~~
\begin{algorithm}[t]
\caption{Two-hole, forward half: the dotted forward jets along $\thh=(\dot U,\dot G)$}
\label{alg:fwd}
\begin{algorithmic}[1]
\Require $\th=(U,G)$; $\thh$ (stack $\mathtt{K+C}$); $w_i,p_i$ (stack $\mathtt W$); order $m$; model sweep $(\xi,\mu)$
\Ensure $\dot\xi_i,\dot\mu_i,\dot\nu_i,\dot\eta_i$ (orders $0..m$) and the undotted $\nu_i$
\State $\dot\xi_i\gets(\dot U_iw_i,\ \dot U_ip_i)$ \Comment{\code{build_input_jets}, \code{compute_xi}}
\State $\nu_{i-1}\gets G_i(\xi_i)\star\nu_i$ \Comment{re-swept through $G$: \code{compute_nu_jets}}
\For{$i=1,\dots,d$}
  \State $\dot\mu_i\gets\dot\mu_{i-1}\star G_i(\xi_i)+\mu_{i-1}\star\dot G_i(\xi_i)+\mu_{i-1}\star G_i(\dot\xi_i)$
  \Comment{\code{compute_sigma_jets}$(\dot G,G,G,\dots)$}
\EndFor
\For{$i=d,\dots,1$}
  \State $\dot\nu_{i-1}\gets G_i(\xi_i)\star\dot\nu_i+\dot G_i(\xi_i)\star\nu_i+G_i(\dot\xi_i)\star\nu_i$
  \Comment{\code{compute_tau_jets}}
\EndFor
\State $\dot\eta_i\gets\dot\mu_{i-1}G_i\ast\nu_i+\mu_{i-1}\dot G_i\ast\nu_i+\mu_{i-1}G_i\ast\dot\nu_i$
  \Comment{\code{compute_deta_jets} (order scan)}
\Impl{\code{sampling_derivatives}: the GN tangent forward (T4S Alg.~7 jet-ified) at $Q=O=P=G$; plain kinds
\code{probing.compute_dxi}, \code{compute_sigma}, \code{compute_tau}, \code{compute_deta}.}
\end{algorithmic}
\end{algorithm}
~~~

after:

~~~
\begin{algorithm}[t]
\caption{Two-hole, forward half: the dotted forward jets along $\thh=(\dot B,\dot G)$}
\label{alg:fwd}
\begin{algorithmic}[1]
\Require $\th=(B,G)$; $\thh$ (stack $\mathtt{K+C}$); $w_i,p_i$ (stack $\mathtt W$); order $m$; model sweep $(\xi,\mu)$
\Ensure $\dot\xi_i,\dot\mu_i,\dot\nu_i,\dot\eta_i$ (orders $0..m$) and the undotted $\nu_i$
\State $\dot\xi_i\gets(\dot B_iw_i,\ \dot B_ip_i)$ \Comment{\code{build_input_jets}, \code{compute_xi}}
\State $\nu_{i-1}\gets G_i(\xi_i)\star\nu_i$ \Comment{re-swept through $G$: \code{compute_nu_jets}}
\For{$i=1,\dots,d$}
  \State $\dot\mu_i\gets\dot\mu_{i-1}\star G_i(\xi_i)+\mu_{i-1}\star\dot G_i(\xi_i)+\mu_{i-1}\star G_i(\dot\xi_i)$
  \Comment{\code{compute_sigma_jets}$(\dot G,G,G,\dots)$}
\EndFor
\For{$i=d,\dots,1$}
  \State $\dot\nu_{i-1}\gets G_i(\xi_i)\star\dot\nu_i+\dot G_i(\xi_i)\star\nu_i+G_i(\dot\xi_i)\star\nu_i$
  \Comment{\code{compute_tau_jets}}
\EndFor
\State $\dot\eta_i\gets\dot\mu_{i-1}G_i\ast\nu_i+\mu_{i-1}\dot G_i\ast\nu_i+\mu_{i-1}G_i\ast\dot\nu_i$
  \Comment{\code{compute_deta_jets} (order scan)}
\Impl{\code{sampling_derivatives}: the GN tangent forward (T4S Alg.~7 jet-ified) at $U=L=R=G$; plain kinds
\code{probing.compute_dxi}, \code{compute_sigma}, \code{compute_tau}, \code{compute_deta}.}
\end{algorithmic}
\end{algorithm}
~~~

### TEX-EHTH-0645 — `docs/exact_hessian_two_hole.tex` 645–662 → 648–665, LaTeX equation line

before:

~~~
\begin{algorithm}[t]
\caption{Two-hole, adjoint half: the dotted adjoint sweeps}
\label{alg:adj}
\begin{algorithmic}[1]
\Require undotted $\xi,\mu,\nu$ and adjoint state $\ddet,\tilde\tau,\tilde\sigma$ at $\th$; dotted $\dot\xi,\dot\mu,\dot\nu$; residual jets $c_i$
\Ensure $\dot{\ddet}_i,\dot{\tilde\tau}_i,\dot{\tilde\sigma}_i$ and the two read rows $\dot{\dxt}_i^{(0,1)}$
\State $\dot{\ddet}_i\gets\dot U_ic_i$ \Comment{order-diagonal; \code{compute_deta_tilde_tangent_jets}}
\For{$i=1,\dots,d$} \Comment{carry $\dot{\tilde\tau}$; the undotted $\tilde\tau_{i-1}$ rides as scan data}
  \State $\dot{\tilde\tau}_i\gets$ \eqref{eq:tautildedot} \Comment{\code{compute_tau_tilde_tangent_jets}, step \code{_adj_tilde_tangent_step}}
\EndFor
\State $\dot{\tilde\sigma}\gets$ the same sweep on the reversed train \Comment{\code{compute_sigma_tilde_tangent_jets}}
\State $\dot{\dxt}_i^{(u)}\gets$ the product rule of \eqref{eq:dxitilde}, $u\in\{0,1\}$ \Comment{\code{compute_dxi_tilde_tangent_jets}}
\Impl{\code{sampling_derivatives}: \code{_adj_prop_tangent}, \code{_adj_tangent_src_order_step},
\code{_adj_tangent_sweep_scanned}, \code{_dxi_tilde_tangent_order_step}; plain probe: \code{probing.compute_deta_tilde_tangent},
\code{compute_tau_tilde_tangent} (two \code{_sigma_step} calls per core), \code{compute_sigma_tilde_tangent},
\code{compute_dxi_tilde_tangent} (two \code{compute_deta} calls).}
\end{algorithmic}
\end{algorithm}
~~~

after:

~~~
\begin{algorithm}[t]
\caption{Two-hole, adjoint half: the dotted adjoint sweeps}
\label{alg:adj}
\begin{algorithmic}[1]
\Require undotted $\xi,\mu,\nu$ and adjoint state $\ddet,\tilde\tau,\tilde\sigma$ at $\th$; dotted $\dot\xi,\dot\mu,\dot\nu$; residual jets $c_i$
\Ensure $\dot{\ddet}_i,\dot{\tilde\tau}_i,\dot{\tilde\sigma}_i$ and the two read rows $\dot{\dxt}_i^{(0,1)}$
\State $\dot{\ddet}_i\gets\dot B_ic_i$ \Comment{order-diagonal; \code{compute_deta_tilde_tangent_jets}}
\For{$i=1,\dots,d$} \Comment{carry $\dot{\tilde\tau}$; the undotted $\tilde\tau_{i-1}$ rides as scan data}
  \State $\dot{\tilde\tau}_i\gets$ \eqref{eq:tautildedot} \Comment{\code{compute_tau_tilde_tangent_jets}, step \code{_adj_tilde_tangent_step}}
\EndFor
\State $\dot{\tilde\sigma}\gets$ the same sweep on the reversed train \Comment{\code{compute_sigma_tilde_tangent_jets}}
\State $\dot{\dxt}_i^{(u)}\gets$ the product rule of \eqref{eq:dxitilde}, $u\in\{0,1\}$ \Comment{\code{compute_dxi_tilde_tangent_jets}}
\Impl{\code{sampling_derivatives}: \code{_adj_prop_tangent}, \code{_adj_tangent_src_order_step},
\code{_adj_tangent_sweep_scanned}, \code{_dxi_tilde_tangent_order_step}; plain probe: \code{probing.compute_deta_tilde_tangent},
\code{compute_tau_tilde_tangent} (two \code{_sigma_step} calls per core), \code{compute_sigma_tilde_tangent},
\code{compute_dxi_tilde_tangent} (two \code{compute_deta} calls).}
\end{algorithmic}
\end{algorithm}
~~~

### TEX-EHTH-0668 — `docs/exact_hessian_two_hole.tex` 668–672 → 671–675, LaTeX equation line

before:

~~~
\begin{equation}
  \label{eq:tuckerdot}
  (\K_c\thh)_{U_i}=\sum_{\mathtt W}\Big(\sum_t\dot\eta_i^{(t)}\otimes c_i^{(t)}+\sum_{u\in\{0,1\}}\dot{\dxt}_i^{(u)}\otimes w_i^{(u)}\Big),
  \qquad w^{(0)}=w,\ w^{(1)}=p .
\end{equation}
~~~

after:

~~~
\begin{equation}
  \label{eq:tuckerdot}
  (\K_c\thh)_{B_i}=\sum_{\mathtt W}\Big(\sum_t\dot\eta_i^{(t)}\otimes c_i^{(t)}+\sum_{u\in\{0,1\}}\dot{\dxt}_i^{(u)}\otimes w_i^{(u)}\Big),
  \qquad w^{(0)}=w,\ w^{(1)}=p .
\end{equation}
~~~

### TEX-EHTH-0696 — `docs/exact_hessian_two_hole.tex` 696–701 → 699–706, LaTeX equation line

before:

~~~
\begin{equation}
  \label{eq:asmjet}
  \delta\tilde G_i=\sum_{\mathtt W}\Big(\sum_{k\in\{0,1\}}\Pi_{i,k}\otimes_2\xi_i^{(k)}
     +\sum_r\mu_{i-1}^{(r)}\otimes F_i^{(r)}\Big),\quad
  \Pi_{i,k}=\sum_r\mu_{i-1}^{(r)}\otimes[\Ush^k\tilde\sigma_i]^{(r)}+\sum_s[\Ush^k\tilde\tau_{i-1}]^{(s)}\otimes\nu_i^{(s)},
\end{equation}
~~~

after:

~~~
\begin{equation}
  \label{eq:asmjet}
  \begin{gathered}
  \tilde H_i=\sum_{\mathtt W}\Big(\sum_{k\in\{0,1\}}\Pi_{i,k}\otimes_2\xi_i^{(k)}
     +\sum_r\mu_{i-1}^{(r)}\otimes F_i^{(r)}\Big),\\
  \Pi_{i,k}=\sum_r\mu_{i-1}^{(r)}\otimes[(\Sh^\T)^k\tilde\sigma_i]^{(r)}+\sum_s[(\Sh^\T)^k\tilde\tau_{i-1}]^{(s)}\otimes\nu_i^{(s)},
  \end{gathered}
\end{equation}
~~~

### TEX-EHTH-0702 — `docs/exact_hessian_two_hole.tex` 702–703 → 707–708, LaTeX equation line

before:

~~~
with the fold $F_i^{(r)}=\sum_s\binom{r+s}{r}\ddet_i^{(r+s)}\otimes\nu_i^{(s)}=\sum_s\frac{1}{r!}(\Ush^r\ddet_i)^{(s)}\otimes\nu_i^{(s)}$
(\code{_tt_det_lag_step}, which carries $\Ush^r\ddet$). Its derivative (\code{_tt_tangent_variation_jets_step},
~~~

after:

~~~
with the fold $F_i^{(r)}=\sum_s\binom{r+s}{r}\ddet_i^{(r+s)}\otimes\nu_i^{(s)}=\sum_s\frac{1}{r!}((\Sh^\T)^r\ddet_i)^{(s)}\otimes\nu_i^{(s)}$
(\code{_tt_det_lag_step}, which carries $(\Sh^\T)^r\ddet$). Its derivative (\code{_tt_tangent_variation_jets_step},
~~~

### TEX-EHTH-0744 — `docs/exact_hessian_two_hole.tex` 744–764 → 749–769, LaTeX equation line

before:

~~~
\begin{algorithm}[t]
\caption{The probe-derivative two-hole $\K_c\thh$ (driver)}
\label{alg:driver}
\begin{algorithmic}[1]
\Require $c_i=\omega^2\odot r_i$ (jets, $\mathtt{W+C}$); $w,p$; $\th=(U,G)$; $\thh$ ($\mathtt{K+C}$); order $m$; model sweep $(\xi,\mu)$
\Ensure $\K_c\thh$, raw core variations ($\mathtt{K+C}$, $\mathtt W$ summed)
\State undotted: $\nu$ re-swept through $G$; $\ddet=Uc$; $\tilde\tau$, $\tilde\sigma$ \Comment{GN kernels, recompute (I-11)}
\State dotted forward $\dot\xi,\dot\mu,\dot\nu,\dot\eta$ \Comment{Algorithm~\ref{alg:fwd}}
\State dotted adjoint $\dot{\ddet},\dot{\tilde\tau},\dot{\tilde\sigma},\dot{\dxt}^{(0,1)}$ \Comment{Algorithm~\ref{alg:adj}}
\State Tucker block \eqref{eq:tuckerdot}; TT block \eqref{eq:asmjetdot}, $\mathtt W$-chunked \Comment{Section~\ref{sec:assembly}}
\Impl{\code{sampling_derivatives.t3_probe_corewise_derivatives_hessian[_from_sweep]}; step 1 \code{compute_nu_jets},
\code{compute_deta_tilde_jets}, \code{compute_tau_tilde_jets}, \code{compute_sigma_tilde_jets}; step 4
\code{assemble_tucker_tangent_variation_jets}, \code{assemble_tt_tangent_variation_jets}; the other kinds:
\code{probing.t3_probe_corewise_hessian[_from_sweep]}, \code{apply.t3_apply_corewise_hessian[_from_sweep]}
(\code{_apply_corewise_hessian_adjoint}), \code{entries.t3_entries_corewise_hessian[_from_sweep]},
\code{sampling_derivatives.t3_apply_corewise_derivatives_hessian[_from_sweep]}, \code{t3_entries_corewise_derivatives_hessian[_from_sweep]}
(\code{_apply_derivatives_hessian_from_jets}, \code{compute_sigma_hat_tangent_jets} with step
\code{_sigma_hat_tangent_jets_step}); the hook \code{SamplingKind.corewise_hessian} (the required sixth operation)
in \code{fitting.py}, \code{uniform_fitting.py}.}
\end{algorithmic}
\end{algorithm}
~~~

after:

~~~
\begin{algorithm}[t]
\caption{The probe-derivative two-hole $\K_c\thh$ (driver)}
\label{alg:driver}
\begin{algorithmic}[1]
\Require $c_i=\omega^2\odot r_i$ (jets, $\mathtt{W+C}$); $w,p$; $\th=(B,G)$; $\thh$ ($\mathtt{K+C}$); order $m$; model sweep $(\xi,\mu)$
\Ensure $\K_c\thh$, raw core variations ($\mathtt{K+C}$, $\mathtt W$ summed)
\State undotted: $\nu$ re-swept through $G$; $\ddet=Bc$; $\tilde\tau$, $\tilde\sigma$ \Comment{GN kernels, recompute (I-11)}
\State dotted forward $\dot\xi,\dot\mu,\dot\nu,\dot\eta$ \Comment{Algorithm~\ref{alg:fwd}}
\State dotted adjoint $\dot{\ddet},\dot{\tilde\tau},\dot{\tilde\sigma},\dot{\dxt}^{(0,1)}$ \Comment{Algorithm~\ref{alg:adj}}
\State Tucker block \eqref{eq:tuckerdot}; TT block \eqref{eq:asmjetdot}, $\mathtt W$-chunked \Comment{Section~\ref{sec:assembly}}
\Impl{\code{sampling_derivatives.t3_probe_corewise_derivatives_hessian[_from_sweep]}; step 1 \code{compute_nu_jets},
\code{compute_deta_tilde_jets}, \code{compute_tau_tilde_jets}, \code{compute_sigma_tilde_jets}; step 4
\code{assemble_tucker_tangent_variation_jets}, \code{assemble_tt_tangent_variation_jets}; the other kinds:
\code{probing.t3_probe_corewise_hessian[_from_sweep]}, \code{apply.t3_apply_corewise_hessian[_from_sweep]}
(\code{_apply_corewise_hessian_adjoint}), \code{entries.t3_entries_corewise_hessian[_from_sweep]},
\code{sampling_derivatives.t3_apply_corewise_derivatives_hessian[_from_sweep]}, \code{t3_entries_corewise_derivatives_hessian[_from_sweep]}
(\code{_apply_derivatives_hessian_from_jets}, \code{compute_sigma_hat_tangent_jets} with step
\code{_sigma_hat_tangent_jets_step}); the hook \code{SamplingKind.corewise_hessian} (the required sixth operation)
in \code{fitting.py}, \code{uniform_fitting.py}.}
\end{algorithmic}
\end{algorithm}
~~~

### TEX-EHTH-0776 — `docs/exact_hessian_two_hole.tex` 776 → 781, LaTeX equation line

unchanged: R5: (B_z, G_z) are Z's own cores (D-6: B a T3's own Tucker cores, already written)

### TEX-EHTH-0779 — `docs/exact_hessian_two_hole.tex` 779–784 → 784–789, LaTeX equation line

before:

~~~
\begin{equation}
  \label{eq:t3t3}
  BU_i=B_{z,i}U_i^\T,\quad \hat G_i=G_{z,i}\times_2BU_i,\quad
  Z^L=\mathrm{zip}_\to(\hat G,G),\quad Z^R=\mathrm{zip}_\leftarrow(\hat G,G),\quad
  \mathrm{env}_i=Z^L_i\,G_{z,i}\,Z^R_i,
\end{equation}
~~~

after:

~~~
\begin{equation}
  \label{eq:t3t3}
  \hat B_i=B_{z,i}B_i^\T,\quad \hat G_i=G_{z,i}\times_2\hat B_i,\quad
  Z^L=\mathrm{zip}_\to(\hat G,G),\quad Z^R=\mathrm{zip}_\leftarrow(\hat G,G),\quad
  \mathrm{env}_i=Z^L_i\,G_{z,i}\,Z^R_i,
\end{equation}
~~~

### TEX-EHTH-0785 — `docs/exact_hessian_two_hole.tex` 785–788 → 790–793, LaTeX equation line

before:

~~~
with gradient blocks $\mathrm{env}_i\times_2BU_i$ (TT) and $\ip{\mathrm{env}_i}{G_i}^\T B_{z,i}$ (Tucker). The
two-hole adds the ``one core replaced'' companions: $\dot{BU}_i=B_{z,i}\dot U_i^\T$, $\dot{\hat G}_i$, the
\emph{tangent zippers} $\dot Z^{L}_{i+1}=\dot Z^L_iA_iB_i+Z^L_i\dot A_iB_i+Z^L_iA_i\dot B_i$ (for the chains
$A=\hat G$, $B=G$; \code{tt_zipper_tangent_left_to_right}, step \code{_tt_zipper_tangent_step}), and the product rule
~~~

after:

~~~
with gradient blocks $\mathrm{env}_i\times_2\hat B_i$ (TT) and $\ip{\mathrm{env}_i}{G_i}^\T B_{z,i}$ (Tucker). The
two-hole adds the ``one core replaced'' companions: $\dot{\hat B}_i=B_{z,i}\dot B_i^\T$, $\dot{\hat G}_i$, the
\emph{tangent zippers} $\dot Z^{L}_{i+1}=\dot Z^L_i\hat G_iG_i+Z^L_i\dot{\hat G}_iG_i+Z^L_i\hat G_i\dot G_i$ (with the chains
$\hat G$, $G$; \code{tt_zipper_tangent_left_to_right}, step \code{_tt_zipper_tangent_step}), and the product rule
~~~

### TEX-EHTH-0789 — `docs/exact_hessian_two_hole.tex` 789–796 → 794–801, LaTeX equation line

before:

~~~
\begin{equation}
  \label{eq:t3t3dot}
  \begin{gathered}
  \dot{\mathrm{env}}_i=\dot Z^L_iG_{z,i}Z^R_i+Z^L_iG_{z,i}\dot Z^R_i,\\
  (\K_Z\thh)_{G_i}=\dot{\mathrm{env}}_i\times_2BU_i+\mathrm{env}_i\times_2\dot{BU}_i,\qquad
  (\K_Z\thh)_{U_i}=\big(\ip{\dot{\mathrm{env}}_i}{G_i}+\ip{\mathrm{env}_i}{\dot G_i}\big)^\T B_{z,i}.
  \end{gathered}
\end{equation}
~~~

after:

~~~
\begin{equation}
  \label{eq:t3t3dot}
  \begin{gathered}
  \dot{\mathrm{env}}_i=\dot Z^L_iG_{z,i}Z^R_i+Z^L_iG_{z,i}\dot Z^R_i,\\
  (\K_Z\thh)_{G_i}=\dot{\mathrm{env}}_i\times_2\hat B_i+\mathrm{env}_i\times_2\dot{\hat B}_i,\qquad
  (\K_Z\thh)_{B_i}=\big(\ip{\dot{\mathrm{env}}_i}{G_i}+\ip{\mathrm{env}_i}{\dot G_i}\big)^\T B_{z,i}.
  \end{gathered}
\end{equation}
~~~

### TEX-EHTH-0801 — `docs/exact_hessian_two_hole.tex` 801–823 → 806–828, LaTeX table or figure

before:

~~~
\begin{figure}[t]
\centering
\begin{tikzpicture}[x=10mm,y=10mm]
  \foreach \i in {1,...,5} {
    \ifnum\i=4 \node[holec] (b\i) at (\i,0) {}; \else
      \ifnum\i=2 \node[dotc] (b\i) at (\i,0) {$\dot G_{\i}$}; \else \node[ttc] (b\i) at (\i,0) {$G_{\i}$}; \fi \fi
    \ifnum\i=4 \node[resn] (a\i) at (\i,1.3) {$G_{z,\i}$}; \else \node[ttc] (a\i) at (\i,1.3) {$\hat G_{\i}$}; \fi
    \draw[bnd] (a\i)--(b\i);
  }
  \foreach \i in {1,...,4} { \pgfmathtruncatemacro\j{\i+1} \draw[bnd] (a\i)--(a\j); \draw[bnd] (b\i)--(b\j); }
  \draw[decorate,decoration={brace,amplitude=4pt,mirror}] (0.6,-0.6)--(3.4,-0.6)
     node[midway,below=4pt,font=\scriptsize]{$\dot Z^L_4$ (one core replaced)};
  \draw[decorate,decoration={brace,amplitude=4pt,mirror}] (4.6,-0.6)--(5.4,-0.6)
     node[midway,below=4pt,font=\scriptsize]{$Z^R_4$};
  \node[font=\scriptsize,anchor=west] at (5.6,1.3) {$Z$'s chain in $\theta$'s basis ($\hat G=G_z\times_2BU$)};
  \node[font=\scriptsize,anchor=west] at (5.6,0) {$\theta$'s chain};
  \node[font=\scriptsize,anchor=west] at (5.6,0.65) {mode legs contracted};
\end{tikzpicture}
\caption{The T3-to-T3 two-hole as a ladder: $Z$'s re-based TT cores on top, $\theta$'s below. The hole at core 4 is
the gradient block; the bottom-left $\dot G_2$ makes $\dot Z^L_4$ one term of the tangent zipper. Core 4's own
top rung is $G_{z,4}$ itself, contracted with $BU_4$ (or $\dot{BU}_4$) on the mode leg.}
\label{fig:t3t3}
\end{figure}
~~~

after:

~~~
\begin{figure}[t]
\centering
\begin{tikzpicture}[x=10mm,y=10mm]
  \foreach \i in {1,...,5} {
    \ifnum\i=4 \node[holec] (b\i) at (\i,0) {}; \else
      \ifnum\i=2 \node[dotc] (b\i) at (\i,0) {$\dot G_{\i}$}; \else \node[ttc] (b\i) at (\i,0) {$G_{\i}$}; \fi \fi
    \ifnum\i=4 \node[resn] (a\i) at (\i,1.3) {$G_{z,\i}$}; \else \node[ttc] (a\i) at (\i,1.3) {$\hat G_{\i}$}; \fi
    \draw[bnd] (a\i)--(b\i);
  }
  \foreach \i in {1,...,4} { \pgfmathtruncatemacro\j{\i+1} \draw[bnd] (a\i)--(a\j); \draw[bnd] (b\i)--(b\j); }
  \draw[decorate,decoration={brace,amplitude=4pt,mirror}] (0.6,-0.6)--(3.4,-0.6)
     node[midway,below=4pt,font=\scriptsize]{$\dot Z^L_4$ (one core replaced)};
  \draw[decorate,decoration={brace,amplitude=4pt,mirror}] (4.6,-0.6)--(5.4,-0.6)
     node[midway,below=4pt,font=\scriptsize]{$Z^R_4$};
  \node[font=\scriptsize,anchor=west] at (5.6,1.3) {$Z$'s chain in $\theta$'s basis ($\hat G=G_z\times_2\hat B$)};
  \node[font=\scriptsize,anchor=west] at (5.6,0) {$\theta$'s chain};
  \node[font=\scriptsize,anchor=west] at (5.6,0.65) {mode legs contracted};
\end{tikzpicture}
\caption{The T3-to-T3 two-hole as a ladder: $Z$'s re-based TT cores on top, $\theta$'s below. The hole at core 4 is
the gradient block; the bottom-left $\dot G_2$ makes $\dot Z^L_4$ one term of the tangent zipper. Core 4's own
top rung is $G_{z,4}$ itself, contracted with $\hat B_4$ (or $\dot{\hat B}_4$) on the mode leg.}
\label{fig:t3t3}
\end{figure}
~~~

### TEX-EHTH-0825 — `docs/exact_hessian_two_hole.tex` 825–840 → 830–845, LaTeX equation line

before:

~~~
\begin{algorithm}[t]
\caption{The T3-to-T3 two-hole $\K_Z\thh$}
\label{alg:t3t3}
\begin{algorithmic}[1]
\Require $Z=(B_z,G_z)$ (tails squashed); $\th=(U,G)$ ($\mathtt C$); $\thh$ ($\mathtt{K+C}$)
\Ensure $\K_Z\thh$
\State $BU,\hat G,Z^L,Z^R,\mathrm{env}$ by \eqref{eq:t3t3} \Comment{per frame: \code{t3_precompute_inner_product_sweep}}
\State $\dot{BU}_i\gets B_{z,i}\dot U_i^\T$;\ $\dot{\hat G}_i\gets G_{z,i}\times_2\dot{BU}_i$
\State $\dot Z^L\gets$ \code{tt_zipper_tangent_left_to_right}$(\hat G,G,\dot{\hat G},\dot G;Z^L)$;\ $\dot Z^R$ likewise from the right
\State blocks by \eqref{eq:t3t3dot}
\Impl{\code{t3_linalg.t3_inner_product_corewise_hessian[_from_sweep]}, \code{t3_linalg.t3_inner_product_corewise_transpose},
\code{tt_operations.tt_zipper_tangent_left_to_right}, \code{tt_zipper_tangent_right_to_left}; called by
\code{ManifoldGeometryOps.tangential_curvature}; uniform: \code{ut3_linalg.ut3_precompute_inner_product_sweep},
\code{ut3_inner_product_corewise_hessian[_from_sweep]}.}
\end{algorithmic}
\end{algorithm}
~~~

after:

~~~
\begin{algorithm}[t]
\caption{The T3-to-T3 two-hole $\K_Z\thh$}
\label{alg:t3t3}
\begin{algorithmic}[1]
\Require $Z=(B_z,G_z)$ (tails squashed); $\th=(B,G)$ ($\mathtt C$); $\thh$ ($\mathtt{K+C}$)
\Ensure $\K_Z\thh$
\State $\hat B,\hat G,Z^L,Z^R,\mathrm{env}$ by \eqref{eq:t3t3} \Comment{per frame: \code{t3_precompute_inner_product_sweep}}
\State $\dot{\hat B}_i\gets B_{z,i}\dot B_i^\T$;\ $\dot{\hat G}_i\gets G_{z,i}\times_2\dot{\hat B}_i$
\State $\dot Z^L\gets$ \code{tt_zipper_tangent_left_to_right}$(\hat G,G,\dot{\hat G},\dot G;Z^L)$;\ $\dot Z^R$ likewise from the right
\State blocks by \eqref{eq:t3t3dot}
\Impl{\code{t3_linalg.t3_inner_product_corewise_hessian[_from_sweep]}, \code{t3_linalg.t3_inner_product_corewise_transpose},
\code{tt_operations.tt_zipper_tangent_left_to_right}, \code{tt_zipper_tangent_right_to_left}; called by
\code{ManifoldGeometryOps.tangential_curvature}; uniform: \code{ut3_linalg.ut3_precompute_inner_product_sweep},
\code{ut3_inner_product_corewise_hessian[_from_sweep]}.}
\end{algorithmic}
\end{algorithm}
~~~

### TEX-EHTH-0860 — `docs/exact_hessian_two_hole.tex` 860–861 → 865–866, LaTeX equation line

before:

~~~
$\delta U_i=S_i^\T\dot U_g$ (\texttt{docs/shared\_t3\_math.tex}, Prop.\ ``tied tangent space is a tilted subspace'';
that note writes the factors $N\times n$, so its $C_i=\dot US_i$ is this relation transposed) has one raw
~~~

after:

~~~
$V_i=S_i^\T\dot B_g$ (\texttt{docs/shared\_t3\_math.tex}, Prop.\ ``tied tangent space is a tilted subspace'';
that note writes the factors $N\times n$, so its $C_i=\dot DS_i$ is this relation transposed) has one raw
~~~

### TEX-EHTH-0863 — `docs/exact_hessian_two_hole.tex` 863 → 868, LaTeX equation line

unchanged: R5: U_M, V_M are SVD factors (§1.5 item 2)

### TEX-EHTH-0865 — `docs/exact_hessian_two_hole.tex` 865–870 → 870–875, LaTeX equation line

before:

~~~
\begin{equation}
  \label{eq:tied}
  \dot U_g=M_g^{+\eps}\begin{bmatrix}\delta U_{i_1}\\ \vdots\\ \delta U_{i_k}\end{bmatrix}
   =V_Mf_\eps(s_g)U_M^\T[\delta U_{i}]_{i\in g},\qquad
  \Lift^\T:\ [\delta U_i]_{i\in g}=(M_g^{+\eps})^\T\textstyle\sum_{i\in g}Y^U_i,
\end{equation}
~~~

after:

~~~
\begin{equation}
  \label{eq:tied}
  \dot B_g=M_g^{+\eps}\begin{bmatrix}V_{i_1}\\ \vdots\\ V_{i_k}\end{bmatrix}
   =V_Mf_\eps(s_g)U_M^\T[V_{i}]_{i\in g},\qquad
  \Lift^\T:\ [V_i]_{i\in g}=(M_g^{+\eps})^\T\textstyle\sum_{i\in g}Y^B_i,
\end{equation}
~~~

### TEX-EHTH-0871 — `docs/exact_hessian_two_hole.tex` 871 → 876, LaTeX equation line

before:

~~~
the same $\dot U_g$ at every group mode; the adjoint sums the group's copies ($\mathsf T^\T$) before the solve's
~~~

after:

~~~
the same $\dot B_g$ at every group mode; the adjoint sums the group's copies ($\mathsf T^\T$) before the solve's
~~~

### TEX-EHTH-0888 — `docs/exact_hessian_two_hole.tex` 888–901 → 893–906, LaTeX equation line

before:

~~~
\begin{algorithm}[t]
\caption{The tied lift and its adjoint (one nontrivial group $g$)}
\label{alg:tied}
\begin{algorithmic}[1]
\Require companion $(U_M,s_g,V_M,\text{row splits})$ of $M_g$; $\eps$; coordinates $[\delta U_i]_{i\in g}$, or raw $[Y^U_i]_{i\in g}$
\State $f\gets f_\eps(s_g)$, clip $\mathrm{rcond}=\mathrm{dtype}\,\eps\cdot\max(\sum_{i\in g}n^D_i,n_g)$ \Comment{\code{_tied_filter}}
\State lift: $\dot U_g\gets V_M\,\mathrm{diag}(f)\,U_M^\T\,[\delta U_i]_{i\in g}$; $\dot U_i\gets\dot U_g$ for $i\in g$ \Comment{\code{_tied_filtered_solve}}
\State adjoint: $Y_g\gets\sum_{i\in g}Y^U_i$;\ $[\delta U_i]\gets U_M\,\mathrm{diag}(f)\,V_M^\T Y_g$, split by rows \Comment{its transpose}
\Impl{\code{sharing.fv_tied_lift_factors} (fills \code{LiftFactors.shared_data}, \code{tied_filters}),
\code{sharing._tied_filter}, \code{_tied_filtered_solve[_transpose]} (with \code{eps=None} bit-identical to the
tied projection's solve), dispatched inside \code{tv_corewise_lift[_transpose]}; uniform:
\code{ufv_operations.ufv_tied_lift_factors}.}
\end{algorithmic}
\end{algorithm}
~~~

after:

~~~
\begin{algorithm}[t]
\caption{The tied lift and its adjoint (one nontrivial group $g$)}
\label{alg:tied}
\begin{algorithmic}[1]
\Require companion $(U_M,s_g,V_M,\text{row splits})$ of $M_g$; $\eps$; coordinates $[V_i]_{i\in g}$, or raw $[Y^B_i]_{i\in g}$
\State $f\gets f_\eps(s_g)$, clip $\mathrm{rcond}=\mathrm{dtype}\,\eps\cdot\max(\sum_{i\in g}n^U_i,n_g)$ \Comment{\code{_tied_filter}}
\State lift: $\dot B_g\gets V_M\,\mathrm{diag}(f)\,U_M^\T\,[V_i]_{i\in g}$; $\dot B_i\gets\dot B_g$ for $i\in g$ \Comment{\code{_tied_filtered_solve}}
\State adjoint: $Y_g\gets\sum_{i\in g}Y^B_i$;\ $[V_i]\gets U_M\,\mathrm{diag}(f)\,V_M^\T Y_g$, split by rows \Comment{its transpose}
\Impl{\code{sharing.fv_tied_lift_factors} (fills \code{LiftFactors.shared_data}, \code{tied_filters}),
\code{sharing._tied_filter}, \code{_tied_filtered_solve[_transpose]} (with \code{eps=None} bit-identical to the
tied projection's solve), dispatched inside \code{tv_corewise_lift[_transpose]}; uniform:
\code{ufv_operations.ufv_tied_lift_factors}.}
\end{algorithmic}
\end{algorithm}
~~~

### TEX-EHTH-0913 — `docs/exact_hessian_two_hole.tex` 913–927 → 918–932, LaTeX equation line

before:

~~~
\begin{algorithm}[h]
\caption{The canonical zero-point frame}
\label{alg:zero}
\begin{algorithmic}[1]
\Require a frame's \emph{structure} (core shapes; unstacked); sharing groups (singletons their own); seed $=0$
\For{each group $g$ with key $k_g=\min g$}
  \State $G\gets$ \code{RandomState([seed, k_g]).standard_normal((n_g, N_g))} \Comment{keyed legacy stream}
  \State $Q,R\gets\mathrm{qr}(G^\T)$ (reduced);\ $U_g\gets(Q\,\mathrm{sign}(\mathrm{diag}R))^\T$, replicated over $g$ \Comment{sign-fixed Haar rows}
\EndFor
\State $O_i,P_i,Q_i\gets$ identity prefixes (orthonormal), $P_d=0$, $Q_1=0$ \Comment{base point $=0$}
\Impl{\code{fv_operations.fv_zero_point_frame} (\code{_haar_rows}, \code{_identity_prefix}, \code{ZERO_FRAME_SEED});
uniform \code{ufv_operations.ufv_zero_point_frame} (unpack, build, pack); geometry member \code{zero_point_frame};
the reference: \code{optimizers._resolve_g0_reference}, \code{_zero_reference_gnorm}.}
\end{algorithmic}
\end{algorithm}
~~~

after:

~~~
\begin{algorithm}[h]
\caption{The canonical zero-point frame}
\label{alg:zero}
\begin{algorithmic}[1]
\Require a frame's \emph{structure} (core shapes; unstacked); sharing groups (singletons their own); seed $=0$
\For{each group $g$ with key $k_g=\min g$}
  \State $G\gets$ \code{RandomState([seed, k_g]).standard_normal((n_g, N_g))} \Comment{keyed legacy stream}
  \State $Q,R\gets\mathrm{qr}(G^\T)$ (reduced);\ $D_g\gets(Q\,\mathrm{sign}(\mathrm{diag}R))^\T$, replicated over $g$ \Comment{sign-fixed Haar rows}
\EndFor
\State $U_i,L_i,R_i\gets$ identity prefixes (orthonormal), $L_d=0$, $R_1=0$ \Comment{base point $=0$}
\Impl{\code{fv_operations.fv_zero_point_frame} (\code{_haar_rows}, \code{_identity_prefix}, \code{ZERO_FRAME_SEED});
uniform \code{ufv_operations.ufv_zero_point_frame} (unpack, build, pack); geometry member \code{zero_point_frame};
the reference: \code{optimizers._resolve_g0_reference}, \code{_zero_reference_gnorm}.}
\end{algorithmic}
\end{algorithm}
~~~

### TEX-EHTH-0934 — `docs/exact_hessian_two_hole.tex` 934 → 939, LaTeX equation line

before:

~~~
Haar only for $U$: the Tucker factor is the only part of the frame that meets the data, and any
~~~

after:

~~~
Haar only for $D$: the Tucker factor is the only part of the frame that meets the data, and any
~~~

### TEX-EHTH-0961 — `docs/exact_hessian_two_hole.tex` 961–988 → 966–993, LaTeX equation line

before:

~~~
\begin{algorithm}[t]
\caption{Newton--CG with the trust region (one row per trial)}
\label{alg:tr}
\begin{algorithmic}[1]
\Require $x_0$; references $\|g_0\|$ (Section~\ref{sec:zero}); forcing power $\kappa$ (default $\frac12$; $1$ for a q-quadratic tail)
\State build the local model at $x$; $g$, $f$ \Comment{\code{Problem.local_model(hessian='exact')}}
\Loop
  \If{$\|g\|\le\texttt{gtol\_rel}\,\|g_0\|$} \Return $x$ (\code{'gradient'}) \EndIf
  \State $\eta\gets\min(\frac12,(\|g\|/\|g_0\|_{cg})^\kappa)$ \Comment{forcing term}
  \State first step: $\Delta_0\gets\|g\|^3/g^\T\Hop_{GN}g$ (or $\|g\|$), $\Delta_{\max}\gets10^4\Delta_0$ \Comment{\code{_initial_radius}}
  \State $p\gets$ Steihaug CG on $\Hop p=-g$ to $\eta\|g\|$ inside $\Delta$ \Comment{\code{_cg_solve}}
  \State if $g^\T p\ge0$: $p\gets-g\min(1,\Delta/\|g\|)$ (a boundary step if scaled)
  \State $\mathrm{pred}\gets g^\T p+\frac12(p^\T\Hop_{GN}p+p^\T\Lift^\T\K_N\Lift p)$ \Comment{one two-hole}
  \State $\rho\gets(f(R_x(p))-f)/\mathrm{pred}$ if $\mathrm{pred}<0$, else NaN
  \State \textbf{if} $\rho<\frac14$ or NaN: $\Delta\gets\frac14\|p\|$;\ \textbf{elif} $\rho>\frac34$ and boundary: $\Delta\gets\min(2\Delta,\Delta_{\max})$
  \If{$\rho>0.1$} accept: $x\gets R_x(p)$, rebuild the model
  \Else \ keep $x$ and its model
    \State \textbf{stop} \code{'objective_floor'} if $|\mathrm{pred}|\le10\,\epsilon_{\mathrm{res}}(\|\omega\odot y\|\,\|\omega\odot r\|+|f|)$
    \State \textbf{stop} \code{'trust_region_floor'} if $\Delta<10^{-12}\Delta_0$ (checked second)
  \EndIf
\EndLoop
\Impl{\code{optimizers.newton_cg}; CG \code{_cg_solve}, \code{_cg_core}, \code{_cg_body}; \code{LocalModel.gn_quadratic},
\code{curvature_quadratic} (constants \code{_TR_ACCEPT}, \code{_TR_SHRINK_BELOW}, \code{_TR_SHRINK},
\code{_TR_EXPAND_ABOVE}, \code{_TR_EXPAND}, \code{_TR_RADIUS_MAX_REL}, \code{_TR_RESOLUTION_FACTOR},
\code{_TR_RADIUS_FLOOR_REL}); $\epsilon_{\mathrm{res}}$ = \code{_resolution_eps} (host-side, dtype-only); per-row
\code{NewtonInfo}.}
\end{algorithmic}
\end{algorithm}
~~~

after:

~~~
\begin{algorithm}[t]
\caption{Newton--CG with the trust region (one row per trial)}
\label{alg:tr}
\begin{algorithmic}[1]
\Require $x_0$; references $\|g_0\|$ (Section~\ref{sec:zero}); forcing power $\kappa$ (default $\frac12$; $1$ for a q-quadratic tail)
\State build the local model at $x$; $g$, $f$ \Comment{\code{Problem.local_model(hessian='exact')}}
\Loop
  \If{$\|g\|\le\texttt{gtol\_rel}\,\|g_0\|$} \Return $x$ (\code{'gradient'}) \EndIf
  \State $\eta\gets\min(\frac12,(\|g\|/\|g_0\|_{cg})^\kappa)$ \Comment{forcing term}
  \State first step: $\Delta_0\gets\|g\|^3/g^\T\Hop_{GN}g$ (or $\|g\|$), $\Delta_{\max}\gets10^4\Delta_0$ \Comment{\code{_initial_radius}}
  \State $p\gets$ Steihaug CG on $\Hop p=-g$ to $\eta\|g\|$ inside $\Delta$ \Comment{\code{_cg_solve}}
  \State if $g^\T p\ge0$: $p\gets-g\min(1,\Delta/\|g\|)$ (a boundary step if scaled)
  \State $\mathrm{pred}\gets g^\T p+\frac12(p^\T\Hop_{GN}p+p^\T\Lift^\T\K_N\Lift p)$ \Comment{one two-hole}
  \State $\rho\gets(f(\mathcal{R}_x(p))-f)/\mathrm{pred}$ if $\mathrm{pred}<0$, else NaN
  \State \textbf{if} $\rho<\frac14$ or NaN: $\Delta\gets\frac14\|p\|$;\ \textbf{elif} $\rho>\frac34$ and boundary: $\Delta\gets\min(2\Delta,\Delta_{\max})$
  \If{$\rho>0.1$} accept: $x\gets \mathcal{R}_x(p)$, rebuild the model
  \Else \ keep $x$ and its model
    \State \textbf{stop} \code{'objective_floor'} if $|\mathrm{pred}|\le10\,\epsilon_{\mathrm{res}}(\|\omega\odot y\|\,\|\omega\odot r\|+|f|)$
    \State \textbf{stop} \code{'trust_region_floor'} if $\Delta<10^{-12}\Delta_0$ (checked second)
  \EndIf
\EndLoop
\Impl{\code{optimizers.newton_cg}; CG \code{_cg_solve}, \code{_cg_core}, \code{_cg_body}; \code{LocalModel.gn_quadratic},
\code{curvature_quadratic} (constants \code{_TR_ACCEPT}, \code{_TR_SHRINK_BELOW}, \code{_TR_SHRINK},
\code{_TR_EXPAND_ABOVE}, \code{_TR_EXPAND}, \code{_TR_RADIUS_MAX_REL}, \code{_TR_RESOLUTION_FACTOR},
\code{_TR_RADIUS_FLOOR_REL}); $\epsilon_{\mathrm{res}}$ = \code{_resolution_eps} (host-side, dtype-only); per-row
\code{NewtonInfo}.}
\end{algorithmic}
\end{algorithm}
~~~

### TEX-EHTH-1006 — `docs/exact_hessian_two_hole.tex` 1006–1033 → 1011–1038, LaTeX table or figure

before:

~~~
\begin{table}[t]
\centering
\small
\begin{tabular}{@{}l p{0.68\textwidth} r@{}}
\toprule
& check & residual \\
\midrule
C1 & \eqref{eq:factors}: $H_i=O_i\times_2S_i^\T$; $P_{>i}=T_iQ_{>i}$; spectra; $\|X\|^2=\|S_1^\T\|_F^2$ & $\le8.4{\times}10^{-16}$\\
C2 & Prop.~\ref{prop:lift}: $D\Phi[\Lift v]=\Emap v$ on minimal / Tucker-slack / TT-slack frames & $\le2.4{\times}10^{-15}$\\
C3--4 & $\Lift_\eps^\T$ is the adjoint at $\eps=10^{-2}\|X\|$; $\max f_\eps=1/(2\eps)$; $M^{+\eps}=(M^\T M+\eps^2I)^{-1}M^\T$ & $\le1.9{\times}10^{-15}$\\
C5 & $\grad f=P_X\nabla f$; $N$ is normal & $\le9.9{\times}10^{-16}$\\
C6 & $\ip{N}{D^2\Phi[\thh,\thh']}=\ip{\thh'}{\K_N\thh}$ (dense double replacement); a vertical $\thh_v$ has
  $\K_N\thh_v=0$ and does not change the form, while $\|\K_{A^\T r}\thh_v\|\approx10^4$ & $\le1.2{\times}10^{-13}$\\
C7 & \eqref{eq:newton} $=\Pi\Lift^\T\nabla^2_\th\Lambda\Lift\Pi$ (\code{jax.jvp} of \code{jax.grad}); $\nabla_\th\Lambda=0$;
  level 1 $=\Pi\Lift^\T\nabla^2_\th(f\circ\Phi)\Lift\Pi$ & $\le4.5{\times}10^{-14}$\\
C8 & $\ip{p}{\Hop p}$ vs $\frac{d^2}{dt^2}f(R(tp))$ (Richardson; level 2) / vs $\frac{d^2}{dt^2}f(\Phi(\th+t\Lift p))$ (level 1) & $3.7{\times}10^{-6}$ / $2.5{\times}10^{-10}$\\
C9 & the two-holes (probe derivatives, apply derivatives, T3--T3) $=$ complex-step $d/ds$ of their one-holes; the
  block sum \eqref{eq:blocksum}; symmetry; order 0 $=$ plain probe & $\le4.6{\times}10^{-16}$\\
C10 & the written recurrences $\dot\mu$, \eqref{eq:tautilde}, \eqref{eq:tautildedot}, $\dot{\dxt}$ vs the kernels & $\le4.3{\times}10^{-16}$\\
C11 & the factored assemblies \eqref{eq:pairs}, \eqref{eq:asmjetdot} $=$ the nine-term binomial product rule & $\le6.3{\times}10^{-16}$\\
C12 & tied lift \eqref{eq:tied}: $D\Phi[\Lift v]=\Emap v$ at a group-ceiling point; $=M_g^+[\delta U_i]$; adjoint & $\le5.1{\times}10^{-16}$\\
C13 & zero frame: nested, rank-independent, base point 0, the recipe by hand (orthonormal to $6.7{\times}10^{-16}$) & $0$\\
C14 & $\Hop=$ GN $+$ \code{curvature_hvp}; quadratic split; exact gradient $\equiv$ GN gradient; default $\eps$ & $\le4.4{\times}10^{-16}$\\
\bottomrule
\end{tabular}
\caption{The equation checks (scratch script against \texttt{76aa37f}; relative residuals).}
\label{tab:checks}
\end{table}
~~~

after:

~~~
\begin{table}[t]
\centering
\small
\begin{tabular}{@{}l p{0.68\textwidth} r@{}}
\toprule
& check & residual \\
\midrule
C1 & \eqref{eq:factors}: $\bar H_i=U_i\times_2S_i^\T$; $L_{>i}=T_iR_{>i}$; spectra; $\|X\|^2=\|S_1^\T\|_F^2$ & $\le8.4{\times}10^{-16}$\\
C2 & Prop.~\ref{prop:lift}: $\mathrm d\Phi[\Lift v]=\Emap v$ on minimal / Tucker-slack / TT-slack frames & $\le2.4{\times}10^{-15}$\\
C3--4 & $\Lift_\eps^\T$ is the adjoint at $\eps=10^{-2}\|X\|$; $\max f_\eps=1/(2\eps)$; $M^{+\eps}=(M^\T M+\eps^2I)^{-1}M^\T$ & $\le1.9{\times}10^{-15}$\\
C5 & $\grad f=P_X\nabla f$; $N$ is normal & $\le9.9{\times}10^{-16}$\\
C6 & $\ip{N}{\mathrm d^2\Phi[\thh,\thh']}=\ip{\thh'}{\K_N\thh}$ (dense double replacement); a vertical $\thh_v$ has
  $\K_N\thh_v=0$ and does not change the form, while $\|\K_{A^\T r}\thh_v\|\approx10^4$ & $\le1.2{\times}10^{-13}$\\
C7 & \eqref{eq:newton} $=\Pi\Lift^\T\nabla^2_\th\Lambda\Lift\Pi$ (\code{jax.jvp} of \code{jax.grad}); $\nabla_\th\Lambda=0$;
  level 1 $=\Pi\Lift^\T\nabla^2_\th(f\circ\Phi)\Lift\Pi$ & $\le4.5{\times}10^{-14}$\\
C8 & $\ip{p}{\Hop p}$ vs $\frac{d^2}{dt^2}f(\mathcal{R}(tp))$ (Richardson; level 2) / vs $\frac{d^2}{dt^2}f(\Phi(\th+t\Lift p))$ (level 1) & $3.7{\times}10^{-6}$ / $2.5{\times}10^{-10}$\\
C9 & the two-holes (probe derivatives, apply derivatives, T3--T3) $=$ complex-step $d/ds$ of their one-holes; the
  block sum \eqref{eq:blocksum}; symmetry; order 0 $=$ plain probe & $\le4.6{\times}10^{-16}$\\
C10 & the written recurrences $\dot\mu$, \eqref{eq:tautilde}, \eqref{eq:tautildedot}, $\dot{\dxt}$ vs the kernels & $\le4.3{\times}10^{-16}$\\
C11 & the factored assemblies \eqref{eq:pairs}, \eqref{eq:asmjetdot} $=$ the nine-term binomial product rule & $\le6.3{\times}10^{-16}$\\
C12 & tied lift \eqref{eq:tied}: $\mathrm d\Phi[\Lift v]=\Emap v$ at a group-ceiling point; $=M_g^+[V_i]$; adjoint & $\le5.1{\times}10^{-16}$\\
C13 & zero frame: nested, rank-independent, base point 0, the recipe by hand (orthonormal to $6.7{\times}10^{-16}$) & $0$\\
C14 & $\Hop=$ GN $+$ \code{curvature_hvp}; quadratic split; exact gradient $\equiv$ GN gradient; default $\eps$ & $\le4.4{\times}10^{-16}$\\
\bottomrule
\end{tabular}
\caption{The equation checks (scratch script against \texttt{76aa37f}; relative residuals).}
\label{tab:checks}
\end{table}
~~~

### TEX-EHTH-1053 — `docs/exact_hessian_two_hole.tex` 1053 → 1058, LaTeX equation line

unchanged: R5: the PB20 bibliography entry (the citation itself, plan §4 item 8)

### DOC-FV-0014 — `docs/frame_variations.md` 14–16 → 14–16, in place

before:

~~~
1--(H0)--R1---R2--1     1---L0--(H1)--R2--1     1---L0---L1--(H2)--1
    |    |    |             |    |    |             |    |    |
    U0   U1   U2            U0   U1   U2            U0   U1   U2
~~~

after:

~~~
1--(H0)--R1---R2--1     1---L0--(H1)--R2--1     1---L0---L1--(H2)--1
    |    |    |             |    |    |             |    |    |
    D0   D1   D2            D0   D1   D2            D0   D1   D2
~~~

### DOC-FV-0023 — `docs/frame_variations.md` 23–30 → 24–31, markdown table

before:

~~~
| frame core | symbol | role | rank on its legs |
|---|---|---|---|
| up-Tucker `Uᵢ` | `U` | the Tucker factors (retained directions) | `nU` |
| down-TT `Oᵢ` | `O` | the Tucker *complement* (where `V` varies) | `nD` |
| left-TT `Lᵢ` | `L`/`P` | left environment (left of the variation) | `rL` |
| right-TT `Rᵢ` | `R`/`Q` | right environment (right of the variation) | `rR` |
| tucker-variation `Vᵢ` | `V` | Tucker-factor direction (in the `nD` complement) | `nD` |
| tt-variation `Hᵢ` | `H` | TT-core direction | `nU`, `rL`, `rR` |
~~~

after:

~~~
| frame core | symbol | role | rank on its legs |
|---|---|---|---|
| down-Tucker `Dᵢ` | `D` | the Tucker factors (retained directions) | `nD` |
| up-TT `Uᵢ` | `U` | the Tucker *complement* (where `V` varies) | `nU` |
| left-TT `Lᵢ` | `L` | left environment (left of the variation) | `rL` |
| right-TT `Rᵢ` | `R` | right environment (right of the variation) | `rR` |
| tucker-variation `Vᵢ` | `V` | Tucker-factor direction (in the `nU` complement) | `nU` |
| tt-variation `Hᵢ` | `H` | TT-core direction | `nD`, `rL`, `rR` |
~~~

### DOC-FV-0057 — `docs/frame_variations.md` 57 → 58, K10b (re-laid out)

before:

~~~
When the frame is **orthonormal** and the variations are **gauged** (`Uᵢᵀ Vᵢ = 0`; `Lᵢ` left-orthogonal
~~~

after:

~~~
When the frame is **orthonormal** and the variations are **gauged** (`Dᵢ Vᵢᵀ = 0`; `Lᵢ` left-orthogonal
~~~

### DOC-FV-0080 — `docs/frame_variations.md` 80–85 → 116–121, markdown table

before:

~~~
| core | orthonormality | the rank it caps |
|---|---|---|
| `Uᵢ` (up / Tucker) | rows orthonormal | `nᵢ ≤ Nᵢ` |
| `Oᵢ` (down / outer) | `nᵢ × (rᵢ·rᵢ₊₁)` unfolding, rows orthonormal | `nᵢ ≤ rᵢ·rᵢ₊₁` |
| `Lᵢ` (left) | `(rᵢ·nᵢ) × rᵢ₊₁`, columns orthonormal | `rᵢ₊₁ ≤ rᵢ·nᵢ` |
| `Rᵢ` (right) | `rᵢ × (nᵢ·rᵢ₊₁)`, rows orthonormal | `rᵢ ≤ nᵢ·rᵢ₊₁` |
~~~

after:

~~~
| core | orthonormality | the rank it caps |
|---|---|---|
| `Dᵢ` (down / Tucker) | rows orthonormal | `nᵢ ≤ Nᵢ` |
| `Uᵢ` (up) | `nᵢ × (rᵢ·rᵢ₊₁)` unfolding, rows orthonormal | `nᵢ ≤ rᵢ·rᵢ₊₁` |
| `Lᵢ` (left) | `(rᵢ·nᵢ) × rᵢ₊₁`, columns orthonormal | `rᵢ₊₁ ≤ rᵢ·nᵢ` |
| `Rᵢ` (right) | `rᵢ × (nᵢ·rᵢ₊₁)`, rows orthonormal | `rᵢ ≤ nᵢ·rᵢ₊₁` |
~~~

### DOC-G-0106 — `docs/glossary.md` 106 → 106, in place

unchanged: R5: B Bᵀ = I is a T3's own Tucker core (D-6, already B); no store letter or width

### DOC-G-0273 — `docs/glossary.md` 273 → 279, in place

before:

~~~
- **gauge conditions** (`is_gauged`, `gauge_residual`; also: gauged). `Ui Viᵀ = 0` for every `i` (code layout), and
~~~

after:

~~~
- **gauge conditions** (`is_gauged`, `gauge_residual`; also: gauged). `Di Viᵀ = 0` for every `i` (code layout), and
~~~

### TEX-PSS-0049 — `docs/pad_safe_svd.tex` 49 → 49, LaTeX equation line

unchanged: R5: the SVD factor U (§1.5 item 2); no frame store letter

### TEX-PSS-0065 — `docs/pad_safe_svd.tex` 65–67 → 65–67, LaTeX equation line

unchanged: R5: the pad counts P_r, P_c or the projector P_real; no frame store letter

### TEX-PSS-0068 — `docs/pad_safe_svd.tex` 68 → 68, LaTeX equation line

unchanged: R5: the pad counts P_r, P_c or the projector P_real; no frame store letter

### TEX-PSS-0086 — `docs/pad_safe_svd.tex` 86–87 → 86–87, LaTeX equation line

unchanged: R5: the SVD factor U (§1.5 item 2); no frame store letter

### TEX-PSS-0120 — `docs/pad_safe_svd.tex` 120 → 120, LaTeX equation line

unchanged: R5: U_0 is the cited BUG paper's basis, not a frame store

### TEX-PSS-0144 — `docs/pad_safe_svd.tex` 144 → 144, LaTeX equation line

unchanged: R5: the SVD factor U (§1.5 item 2); no frame store letter

### TEX-PSS-0147 — `docs/pad_safe_svd.tex` 147 → 147, LaTeX equation line

unchanged: R5: the SVD factor U (§1.5 item 2); no frame store letter

### TEX-PSS-0149 — `docs/pad_safe_svd.tex` 149 → 149, LaTeX equation line

unchanged: R5: the SVD factor U (§1.5 item 2); no frame store letter

### TEX-PSS-0163 — `docs/pad_safe_svd.tex` 163 → 163, LaTeX equation line

unchanged: R5: the SVD factor U (§1.5 item 2); no frame store letter

### TEX-PSS-0183 — `docs/pad_safe_svd.tex` 183 → 183, LaTeX equation line

unchanged: R5: QR's Q (§1.5 item 2); no frame store letter

### TEX-PSS-0185 — `docs/pad_safe_svd.tex` 185 → 185, LaTeX equation line

unchanged: R5: QR's Q (§1.5 item 2); no frame store letter

### TEX-PSS-0195 — `docs/pad_safe_svd.tex` 195 → 195, LaTeX equation line

unchanged: R5: QR's Q (§1.5 item 2); no frame store letter

### TEX-PSS-0197 — `docs/pad_safe_svd.tex` 197 → 197, LaTeX equation line

unchanged: R5: QR's Q (§1.5 item 2); no frame store letter

### TEX-PSS-0205 — `docs/pad_safe_svd.tex` 205–206 → 205–206, LaTeX equation line

unchanged: R5: B = Q^T A_pad (the projected core) or the GSVD's B, not a T3 core; no frame store letter

### TEX-PSS-0221 — `docs/pad_safe_svd.tex` 221 → 221, LaTeX equation line

unchanged: R5: QR's Q (§1.5 item 2); no frame store letter

### TEX-PSS-0223 — `docs/pad_safe_svd.tex` 223 → 223, LaTeX equation line

unchanged: R5: B = Q^T A_pad (the projected core) or the GSVD's B, not a T3 core; no frame store letter

### TEX-PSS-0226 — `docs/pad_safe_svd.tex` 226 → 226, LaTeX equation line

unchanged: R5: B = Q^T A_pad (the projected core) or the GSVD's B, not a T3 core; no frame store letter

### TEX-PSS-0235 — `docs/pad_safe_svd.tex` 235–236 → 235–236, LaTeX equation line

unchanged: R5: the SVD factor U (§1.5 item 2); no frame store letter

### TEX-PSS-0240 — `docs/pad_safe_svd.tex` 240 → 240, LaTeX equation line

unchanged: R5: the SVD factor U (§1.5 item 2); no frame store letter

### TEX-PSS-0243 — `docs/pad_safe_svd.tex` 243 → 243, LaTeX equation line

unchanged: R5: QR's R (§1.5 item 2); no frame store letter

### TEX-PSS-0274 — `docs/pad_safe_svd.tex` 274 → 274, LaTeX equation line

unchanged: R5: B = Q^T A_pad (the projected core) or the GSVD's B, not a T3 core; no frame store letter

### TEX-PSS-0276 — `docs/pad_safe_svd.tex` 276 → 276, LaTeX equation line

unchanged: R5: B = Q^T A_pad (the projected core) or the GSVD's B, not a T3 core; no frame store letter

### TEX-PSS-0322 — `docs/pad_safe_svd.tex` 322 → 322, LaTeX equation line

unchanged: R5: the SVD factor U (§1.5 item 2); no frame store letter

### TEX-PSS-0350 — `docs/pad_safe_svd.tex` 350 → 350, LaTeX equation line

unchanged: R5: P_r (a pad count); "Method D" a design name; O(N^3) big-O

### TEX-PSS-0354 — `docs/pad_safe_svd.tex` 354–355 → 354–355, LaTeX equation line

unchanged: R5: the pad counts P_r, P_c or the projector P_real; no frame store letter

### TEX-PSS-0358 — `docs/pad_safe_svd.tex` 358–359 → 358–359, LaTeX equation line

unchanged: R5: the pad counts P_r, P_c or the projector P_real; no frame store letter

### TEX-PSS-0362 — `docs/pad_safe_svd.tex` 362 → 362, LaTeX equation line

unchanged: R5: B = Q^T A_pad (the projected core) or the GSVD's B, not a T3 core; no frame store letter

### TEX-PSS-0384 — `docs/pad_safe_svd.tex` 384 → 384, LaTeX equation line

unchanged: R5: B = Q^T A_pad (the projected core) or the GSVD's B, not a T3 core; no frame store letter

### TEX-PSS-0424 — `docs/pad_safe_svd.tex` 424 → 424, LaTeX equation line

unchanged: R5: QR's Q (§1.5 item 2); no frame store letter

### TEX-PSS-0426 — `docs/pad_safe_svd.tex` 426 → 426, LaTeX equation line

unchanged: R5: B = Q^T A_pad (the projected core) or the GSVD's B, not a T3 core; no frame store letter

### TEX-PSS-0427 — `docs/pad_safe_svd.tex` 427–429 → 427–429, LaTeX equation line

unchanged: R5: B = Q^T A_pad (the projected core) or the GSVD's B, not a T3 core; no frame store letter

### TEX-PSS-0430 — `docs/pad_safe_svd.tex` 430–431 → 430–431, LaTeX equation line

unchanged: R5: B = Q^T A_pad (the projected core) or the GSVD's B, not a T3 core; no frame store letter

### TEX-PFC-0096 — `docs/padded_frame_cone.tex` 96–97 → 96–97, LaTeX equation line

before:

~~~
$x\to t$, the frame $U_{x\to t}$ is an orthonormal basis of the column space of the tensor unfolded with $x$'s side
as rows. Frames are nested: $U_{x\to t}$ lies in $H_x\otimes\bigotimes_{y\in\mathrm{nb}(x)\setminus t}U_{y\to x}$,
~~~

after:

~~~
$x\to t$, the frame $F_{x\to t}$ is an orthonormal basis of the column space of the tensor unfolded with $x$'s side
as rows. Frames are nested: $F_{x\to t}$ lies in $H_x\otimes\bigotimes_{y\in\mathrm{nb}(x)\setminus t}F_{y\to x}$,
~~~

### TEX-PFC-0102 — `docs/padded_frame_cone.tex` 102–116 → 107–121, LaTeX table or figure

before:

~~~
\begin{table}[h]
\centering
\begin{tabular}{@{}llll@{}}
\toprule
directed edge & store & core & output slot \\
\midrule
$\ell_i\to c_i$ & \code{up[i]} & $U_i\in\R^{n_{U,i}\times N_i}$ & the Tucker index \\
$c_i\to \ell_i$ & \code{down[i]} & $O_i\in\R^{r_{L,i}\times n_{D,i}\times r_{R,i+1}}$ & the down index \\
$c_i\to c_{i+1}$ & \code{left[i]}, $i\le d-2$ & $L_i\in\R^{r_{L,i}\times n_{U,i}\times r_{L,i+1}}$ & bond $i+1$ \\
$c_i\to c_{i-1}$ & \code{right[i]}, $i\ge 1$ & $R_i\in\R^{r_{R,i}\times n_{U,i}\times r_{R,i+1}}$ & bond $i$ \\
\bottomrule
\end{tabular}
\caption{Directed edges of a T3 and the frame stores. $L_{d-1}$ and $R_0$ are the \emph{carriers} of the base
point, not frame cores of an edge.}\label{tab:stores}
\end{table}
~~~

after:

~~~
\begin{table}[h]
\centering
\begin{tabular}{@{}llll@{}}
\toprule
directed edge & store & core & output slot \\
\midrule
$\ell_i\to c_i$ & \code{down[i]} & $D_i\in\R^{n_{D,i}\times N_i}$ & the Tucker index \\
$c_i\to \ell_i$ & \code{up[i]} & $U_i\in\R^{r_{L,i}\times n_{U,i}\times r_{R,i+1}}$ & the middle index \\
$c_i\to c_{i+1}$ & \code{left[i]}, $i\le d-2$ & $L_i\in\R^{r_{L,i}\times n_{D,i}\times r_{L,i+1}}$ & bond $i+1$ \\
$c_i\to c_{i-1}$ & \code{right[i]}, $i\ge 1$ & $R_i\in\R^{r_{R,i}\times n_{D,i}\times r_{R,i+1}}$ & bond $i$ \\
\bottomrule
\end{tabular}
\caption{Directed edges of a T3 and the frame stores. $L_{d-1}$ and $R_0$ are the \emph{carriers} of the base
point, not frame cores of an edge.}\label{tab:stores}
\end{table}
~~~

### TEX-PFC-0120 — `docs/padded_frame_cone.tex` 120 → 125, LaTeX equation line

unchanged: R5: the completion P_{x->t} and the P-block (§1.5 item 3)

### TEX-PFC-0140 — `docs/padded_frame_cone.tex` 140–141 → 145–146, LaTeX equation line

unchanged: R5: the matrix case's SVD factor U (§1.5 item 2); O(eps^2) big-O

### TEX-PFC-0152 — `docs/padded_frame_cone.tex` 152–153 → 157–158, LaTeX equation line

before:

~~~
space: at edge $x\to t$, the rows in $C'_{x\to t}\ominus U_{x\to t}$ with $C'$ built from the completed child frames,
and the columns in the complement of $A$'s frame $U_{t\to x}$. Its completions depend on the root: each edge sees the
~~~

after:

~~~
space: at edge $x\to t$, the rows in $C'_{x\to t}\ominus F_{x\to t}$ with $C'$ built from the completed child frames,
and the columns in the complement of $A$'s frame $F_{t\to x}$. Its completions depend on the root: each edge sees the
~~~

### TEX-PFC-0171 — `docs/padded_frame_cone.tex` 171–172 → 176–177, LaTeX equation line

before:

~~~
edge, tensors whose far side lies in the \emph{far completion} $Q_e$ (the completion of $U_{t\to x}$). Through them
the normal block of $S$ applied to $Q_e$ enters the step: information from outside the current completions.
~~~

after:

~~~
edge, tensors whose far side lies in the \emph{far completion} $Q_e$ (the completion of $F_{t\to x}$). Through them
the normal block of $S$ applied to $Q_e$ enters the step: information from outside the current completions.
~~~

### TEX-PFC-0174 — `docs/padded_frame_cone.tex` 174–175 → 179–180, LaTeX equation line

unchanged: R5: the matrix case: completions P, Q of the SVD's U, V, U'=[U P] (§1.5 items 2, 3)

### TEX-PFC-0188 — `docs/padded_frame_cone.tex` 188 → 193, LaTeX equation line

unchanged: R5: the completions P_{a->b}, P_{b->a} (§1.5 item 3)

### TEX-PFC-0195 — `docs/padded_frame_cone.tex` 195 → 200, LaTeX equation line

unchanged: R5: the SVD's U, V with completions P, Q (§1.5 items 2, 3)

### TEX-PFC-0196 — `docs/padded_frame_cone.tex` 196–198 → 201–203, LaTeX equation line

unchanged: R5: E(P,Q), the completions (§1.5 item 3)

### TEX-PFC-0199 — `docs/padded_frame_cone.tex` 199–200 → 204–205, LaTeX equation line

unchanged: R5: (P,Q), the completions (§1.5 item 3)

### TEX-PFC-0206 — `docs/padded_frame_cone.tex` 206–207 → 211–212, LaTeX equation line

unchanged: R5: U'=[U P], V'=[V Q]: SVD factors and completions (§1.5 items 2, 3)

### TEX-PFC-0209 — `docs/padded_frame_cone.tex` 209–212 → 214–217, LaTeX equation line

unchanged: R5: the completion P, the SVD's [U P], O(h^3) big-O

### TEX-PFC-0216 — `docs/padded_frame_cone.tex` 216–220 → 221–225, LaTeX equation line

unchanged: R5: R(theta) is a residual function (plan §4 item 8); O(theta^4) big-O

### TEX-PFC-0221 — `docs/padded_frame_cone.tex` 221 → 226, LaTeX equation line

unchanged: R5: R, the same residual function (plan §4 item 8)

### TEX-PFC-0225 — `docs/padded_frame_cone.tex` 225 → 230, LaTeX equation line

unchanged: R5: the completions P, Q (§1.5 item 3)

### TEX-PFC-0242 — `docs/padded_frame_cone.tex` 242 → 247, LaTeX equation line

unchanged: R5: the completion P (§1.5 item 3)

### TEX-PFC-0258 — `docs/padded_frame_cone.tex` 258–259 → 263–264, LaTeX equation line

unchanged: R5: L is an edge count, not the left store

### TEX-PFC-0261 — `docs/padded_frame_cone.tex` 261 → 266, LaTeX equation line

unchanged: R5: order 2L, the same edge count

### TEX-PFC-0278 — `docs/padded_frame_cone.tex` 278 → 283, LaTeX equation line

unchanged: R5: the completions P, P^+ (§1.5 item 3)

### TEX-PFC-0280 — `docs/padded_frame_cone.tex` 280–289 → 285–294, LaTeX table or figure

before:

~~~
\begin{tabular}{@{}llll@{}}
\toprule
message & size & definition & built \\
\midrule
$\Phi_{a\to b}$ & $k\times R$ & $P_{a\to b}^\T Z_{\mathrm{far}}U'_{b\to a}$ & stage 1, far leaves inward\\
$\Xi_{a\to b}$ & $k\times\cdot$ & $\Xi\Xi^\T=P_{a\to b}^\T Z_{\mathrm{far}}(I-U'_{b\to a}U'^\T_{b\to a})Z_{\mathrm{far}}^\T P_{a\to b}$ & stage 1, with $\Phi$\\
$D_{a\to b}$ & $R\times k$ & $U'^{+\T}_{a\to b}Z_{\mathrm{near}}P_{b\to a}$ & stage 2, message order\\
$o_{a\to b}$ & $k^+\times k$ & $P^{+\T}_{a\to b}P_{a\to b}$ & stage 2, with $D$\\
\bottomrule
\end{tabular}
~~~

after:

~~~
\begin{tabular}{@{}llll@{}}
\toprule
message & size & definition & built \\
\midrule
$\Phi_{a\to b}$ & $k\times R$ & $P_{a\to b}^\T Z_{\mathrm{far}}F'_{b\to a}$ & stage 1, far leaves inward\\
$\Xi_{a\to b}$ & $k\times\cdot$ & $\Xi\Xi^\T=P_{a\to b}^\T Z_{\mathrm{far}}(I-F'_{b\to a}F'^\T_{b\to a})Z_{\mathrm{far}}^\T P_{a\to b}$ & stage 1, with $\Phi$\\
$\Psi_{a\to b}$ & $R\times k$ & $F'^{+\T}_{a\to b}Z_{\mathrm{near}}P_{b\to a}$ & stage 2, message order\\
$o_{a\to b}$ & $k^+\times k$ & $P^{+\T}_{a\to b}P_{a\to b}$ & stage 2, with $\Psi$\\
\bottomrule
\end{tabular}
~~~

### TEX-PFC-0292 — `docs/padded_frame_cone.tex` 292 → 297, LaTeX equation line

before:

~~~
square root of the rest (any factor; only $\Xi\Xi^\T$ enters). $D$ and $o$ carry the near side, already rotated into
~~~

after:

~~~
square root of the rest (any factor; only $\Xi\Xi^\T$ enters). $\Psi$ and $o$ carry the near side, already rotated into
~~~

### TEX-PFC-0298 — `docs/padded_frame_cone.tex` 298–300 → 303–305, LaTeX equation line

unchanged: R5: Phi_P (the completion part of Phi) and Theta; no store letter

### TEX-PFC-0303 — `docs/padded_frame_cone.tex` 303 → 308, LaTeX equation line

unchanged: R5: the completion P^+ (§1.5 item 3)

### TEX-PFC-0306 — `docs/padded_frame_cone.tex` 306–323 → 311–328, LaTeX equation line

before:

~~~
\begin{algorithm}[t]
\caption{One alignment step (ragged)}\label{alg:step}
\begin{algorithmic}[1]
\Require padded frame $F$; variations $(V,H)$ of $Z$ at $F$; old ranks; output ranks; \code{update}
\State Stage 1 (old frames): down messages; left messages from the far end; right messages from the near end; up
messages; each $(\Phi,\Xi)$.
\Impl{\code{_down_message}, \code{_left_message_step}, \code{_right_message_step} (scan bodies), \code{_up_message};
$\Xi$ by \code{_root}, a factor from the thin SVD, never a Gram matrix.}
\State Stage 2 (message order): up edges (one per sharing group, the members' contents side by side); the left sweep
$i=0..d-2$; the right sweep $i=d-1..1$; down edges. At each: new coordinates, $Y$ by \eqref{eq:edge}, $P^+$, $o$, $D$,
movement.
\Impl{\code{_new_coordinates}, \code{_edge_content}, \code{_edge} $\to$ \code{linalg.pinned_leading_columns};
\code{_up_edge}, \code{_left_edge_step}, \code{_right_edge_step}, \code{_down_edge}.}
\State A store not in \code{update} is transported: $Y=g_{\mathrm{old}}$.
\State Carriers: the old content, embedded. Return the frame and the largest movement.
\Impl{\code{padded_frames.tv_align_padded_frame}; frontend one pass: \code{T3Tangent.align_padded}.}
\end{algorithmic}
\end{algorithm}
~~~

after:

~~~
\begin{algorithm}[t]
\caption{One alignment step (ragged)}\label{alg:step}
\begin{algorithmic}[1]
\Require padded frame $F$; variations $(V,H)$ of $Z$ at $F$; old ranks; output ranks; \code{update}
\State Stage 1 (old frames): the up store's messages ($c_i\to\ell_i$); left messages from the far end; right messages
from the near end; the down store's messages ($\ell_i\to c_i$); each $(\Phi,\Xi)$.
\Impl{\code{_up_message}, \code{_left_message_step}, \code{_right_message_step} (scan bodies), \code{_down_message};
$\Xi$ by \code{_root}, a factor from the thin SVD, never a Gram matrix.}
\State Stage 2 (message order): the down store's edges ($\ell_i\to c_i$; one per sharing group, the members' contents
side by side); the left sweep $i=0..d-2$; the right sweep $i=d-1..1$; the up store's edges ($c_i\to\ell_i$). At each:
new coordinates, $Y$ by \eqref{eq:edge}, $P^+$, $o$, $\Psi$, movement.
\Impl{\code{_new_coordinates}, \code{_edge_content}, \code{_edge} $\to$ \code{linalg.pinned_leading_columns};
\code{_down_edge}, \code{_left_edge_step}, \code{_right_edge_step}, \code{_up_edge}.}
\State A store not in \code{update} is transported: $Y=g_{\mathrm{old}}$.
\State Carriers: the old content, embedded. Return the frame and the largest movement.
\Impl{\code{padded_frames.tv_align_padded_frame}; frontend one pass: \code{T3Tangent.align_padded}.}
\end{algorithmic}
\end{algorithm}
~~~

### TEX-PFC-0330 — `docs/padded_frame_cone.tex` 330 → 335, LaTeX equation line

unchanged: R5: the P-block (§1.5 item 3); (N_a, R_b) are sizes

### TEX-PFC-0405 — `docs/padded_frame_cone.tex` 405 → 411, LaTeX equation line

unchanged: R5: the completion P^+ (§1.5 item 3)

### TEX-PFC-0451 — `docs/padded_frame_cone.tex` 451 → 457, LaTeX equation line

unchanged: R5: U_k is the thin SVD's left factor in pinned_leading_columns (§1.5 item 2)

### TEX-STM-0066 — `docs/shared_t3_math.tex` 66–70 → 66–70, LaTeX equation line

before:

~~~
\[
\T = \bigl[\bigl[\, G ;\; U_1, \ldots, U_d \,\bigr]\bigr],
\qquad
G = \mathrm{TT}(G_1, \ldots, G_d),
\]
~~~

after:

~~~
\[
\T = \bigl[\bigl[\, G ;\; B_1, \ldots, B_d \,\bigr]\bigr],
\qquad
G = \mathrm{TT}(G_1, \ldots, G_d),
\]
~~~

### TEX-STM-0071 — `docs/shared_t3_math.tex` 71 → 71, LaTeX equation line

before:

~~~
where $U_i \in \R^{N_i \times n_i}$ are the Tucker factors,
~~~

after:

~~~
where $B_i \in \R^{N_i \times n_i}$ are the Tucker factors,
~~~

### TEX-STM-0082 — `docs/shared_t3_math.tex` 82 → 82, LaTeX equation line

before:

~~~
$U_i$ have orthonormal columns, and, when a statement refers to mode $i$, the
~~~

after:

~~~
$B_i$ have orthonormal columns, and, when a statement refers to mode $i$, the
~~~

### TEX-STM-0104 — `docs/shared_t3_math.tex` 104 → 114, LaTeX equation line

before:

~~~
denoted $N_g$. A T3 is \emph{shared} (an SF-T3) if $U_{i} = U_{g}$ and
~~~

after:

~~~
denoted $N_g$. A T3 is \emph{shared} (an SF-T3) if $B_{i} = B_{g}$ and
~~~

### TEX-STM-0118 — `docs/shared_t3_math.tex` 118–120 → 128–130, LaTeX equation line

before:

~~~
\begin{equation}\label{eq:skeleton}
\T_{(i)} = U_g\, W_i\, M_i,
\end{equation}
~~~

after:

~~~
\begin{equation}\label{eq:skeleton}
\T_{(i)} = B_g\, W_i\, M_i,
\end{equation}
~~~

### TEX-STM-0122 — `docs/shared_t3_math.tex` 122–126 → 132–136, LaTeX equation line

before:

~~~
\begin{equation}\label{eq:stacked}
\bigl[\,\T_{(i_1)} \,|\, \cdots \,|\, \T_{(i_k)}\,\bigr]
= U_g \,\bigl[\, W_{i_1} \,|\, \cdots \,|\, W_{i_k} \,\bigr]\,
\operatorname{blockdiag}\!\bigl(M_{i_1}, \ldots, M_{i_k}\bigr).
\end{equation}
~~~

after:

~~~
\begin{equation}\label{eq:stacked}
\bigl[\,\T_{(i_1)} \,|\, \cdots \,|\, \T_{(i_k)}\,\bigr]
= B_g \,\bigl[\, W_{i_1} \,|\, \cdots \,|\, W_{i_k} \,\bigr]\,
\operatorname{blockdiag}\!\bigl(M_{i_1}, \ldots, M_{i_k}\bigr).
\end{equation}
~~~

### TEX-STM-0147 — `docs/shared_t3_math.tex` 147 → 157, LaTeX equation line

before:

~~~
factor $U_g$ is common by sharedness, \eqref{eq:stacked} follows by placing
~~~

after:

~~~
factor $B_g$ is common by sharedness, \eqref{eq:stacked} follows by placing
~~~

### TEX-STM-0168 — `docs/shared_t3_math.tex` 168 → 178, LaTeX equation line

before:

~~~
column space contained in $\operatorname{range}(U_g)$, hence so does the
~~~

after:

~~~
column space contained in $\operatorname{range}(B_g)$, hence so does the
~~~

### TEX-STM-0170 — `docs/shared_t3_math.tex` 170–171 → 180–181, LaTeX equation line

before:

~~~
$(\leq)$: take $U_g$ to be an orthonormal basis of the concatenation's column
space; projecting each mode ($U_g U_g^\top \T_{(i)} = \T_{(i)}$) yields an
~~~

after:

~~~
$(\leq)$: take $B_g$ to be an orthonormal basis of the concatenation's column
space; projecting each mode ($B_g B_g^\top \T_{(i)} = \T_{(i)}$) yields an
~~~

### TEX-STM-0179 — `docs/shared_t3_math.tex` 179–202 → 189–212, LaTeX equation line

before:

~~~
\begin{algorithm}[Grouped T3-SVD / SF-rounding; cf.~{\cite[Alg.~1]{MR2026}}]
\label{alg:rounding}
Input: shared T3 with orthonormal factors; target ranks. Output: shared T3
with target ranks.
\begin{enumerate}[label=\emph{(\arabic*)},itemsep=1pt,topsep=2pt]
\item TT-round the cores to the target TT ranks (standard sweep; factors
  untouched). In an interlaced implementation, steps (2)--(4) are woven into
  the same sweep; the interlacing order changes truncated results but not the
  bounds below.
\item For each mode $i$, obtain $W_i$ in the $i$-orthogonal representation
  (one orthogonalization sweep).
\item Singleton modes: truncated SVD of $W_i$; rotate factor and core as
  usual.
\item Each group $g$: one truncated SVD of the concatenation
  $[\,W_{i_1} | \cdots | W_{i_k}\,] \in \R^{n_g \times \sum_{i\in g}
  r_{i-1} r_i}$, yielding leading left singular vectors $Y \in \R^{n_g \times
  n_g'}$. Update $U_g \leftarrow U_g Y$ \emph{once} (assigned to every mode of
  $g$) and apply $Y^\top$ to the up-leg of every core in $g$.
\end{enumerate}
By Lemma~\ref{lem:skeleton}, step (4) computes the optimal rank-$n_g'$
truncation of the concatenation while touching the large dimension $N_g$ only
in the single product $U_g Y$, at cost $O(N_g n_g^2)$ per group; the SVD
itself acts on an $n_g \times \sum_i r_{i-1} r_i$ matrix.
\end{algorithm}
~~~

after:

~~~
\begin{algorithm}[Grouped T3-SVD / SF-rounding; cf.~{\cite[Alg.~1]{MR2026}}]
\label{alg:rounding}
Input: shared T3 with orthonormal factors; target ranks. Output: shared T3
with target ranks.
\begin{enumerate}[label=\emph{(\arabic*)},itemsep=1pt,topsep=2pt]
\item TT-round the cores to the target TT ranks (standard sweep; factors
  untouched). In an interlaced implementation, steps (2)--(4) are woven into
  the same sweep; the interlacing order changes truncated results but not the
  bounds below.
\item For each mode $i$, obtain $W_i$ in the $i$-orthogonal representation
  (one orthogonalization sweep).
\item Singleton modes: truncated SVD of $W_i$; rotate factor and core as
  usual.
\item Each group $g$: one truncated SVD of the concatenation
  $[\,W_{i_1} | \cdots | W_{i_k}\,] \in \R^{n_g \times \sum_{i\in g}
  r_{i-1} r_i}$, yielding leading left singular vectors $Y \in \R^{n_g \times
  n_g'}$. Update $B_g \leftarrow B_g Y$ \emph{once} (assigned to every mode of
  $g$) and apply $Y^\top$ to the Tucker leg of every core in $g$.
\end{enumerate}
By Lemma~\ref{lem:skeleton}, step (4) computes the optimal rank-$n_g'$
truncation of the concatenation while touching the large dimension $N_g$ only
in the single product $B_g Y$, at cost $O(N_g n_g^2)$ per group; the SVD
itself acts on an $n_g \times \sum_i r_{i-1} r_i$ matrix.
\end{algorithm}
~~~

### TEX-STM-0205 — `docs/shared_t3_math.tex` 205 → 215, LaTeX equation line

unchanged: R5: P_sf is the shared rounding operator, not a frame core

### TEX-STM-0208 — `docs/shared_t3_math.tex` 208–217 → 218–227, LaTeX equation line

unchanged: R5: P_sf, the shared rounding operator

### TEX-STM-0221 — `docs/shared_t3_math.tex` 221–222 → 231–232, LaTeX equation line

unchanged: R5: P_sf, P_{SF-Tucker}, P_TT, the rounding stages

### TEX-STM-0272 — `docs/shared_t3_math.tex` 272–289 → 282–299, LaTeX equation line

before:

~~~
\begin{algorithm}[Shared projection of an unshared T3]\label{alg:init}
Input: T3 with orthonormal but distinct factors $U_i$, $i \in g$.
For each group $g$:
\begin{enumerate}[label=\emph{(\arabic*)},itemsep=1pt,topsep=2pt]
\item Stack factors and compute one tall QR:
  $[\,U_{i_1} | \cdots | U_{i_k}\,] = Q R$, with
  $Q \in \R^{N_g \times m}$, $m \leq \sum_{i\in g} n_i$.
\item Express each block $B_i \coloneqq U_i W_i$ (never formed at size
  $N_g$) in the $Q$ basis: $Q^\top B_i = R_{[i]} W_i$, where $R_{[i]}$ is the
  corresponding column block of $R$. SVD the small stacked matrix
  $[\,R_{[i_1]} W_{i_1} | \cdots | R_{[i_k]} W_{i_k}\,] \in
  \R^{m \times \sum_i r_{i-1} r_i}$; retain leading left singular vectors
  $Y$.
\item Set $U_g \coloneqq Q Y$; project each mode by applying
  $(U_g^\top U_i)$ to the up-leg of core $i$ and replacing the factor by
  $U_g$. Follow with Algorithm~\ref{alg:rounding}.
\end{enumerate}
\end{algorithm}
~~~

after:

~~~
\begin{algorithm}[Shared projection of an unshared T3]\label{alg:init}
Input: T3 with orthonormal but distinct factors $B_i$, $i \in g$.
For each group $g$:
\begin{enumerate}[label=\emph{(\arabic*)},itemsep=1pt,topsep=2pt]
\item Stack factors and compute one tall QR:
  $[\,B_{i_1} | \cdots | B_{i_k}\,] = Q R$, with
  $Q \in \R^{N_g \times m}$, $m \leq \sum_{i\in g} n_i$.
\item Express each block $K_i \coloneqq B_i W_i$ (never formed at size
  $N_g$) in the $Q$ basis: $Q^\top K_i = R_{[i]} W_i$, where $R_{[i]}$ is the
  corresponding column block of $R$. SVD the small stacked matrix
  $[\,R_{[i_1]} W_{i_1} | \cdots | R_{[i_k]} W_{i_k}\,] \in
  \R^{m \times \sum_i r_{i-1} r_i}$; retain leading left singular vectors
  $Y$.
\item Set $B_g \coloneqq Q Y$; project each mode by applying
  $(B_g^\top B_i)$ to the Tucker leg of core $i$ and replacing the factor by
  $B_g$. Follow with Algorithm~\ref{alg:rounding}.
\end{enumerate}
\end{algorithm}
~~~

### TEX-STM-0292 — `docs/shared_t3_math.tex` 292–293 → 302–303, LaTeX equation line

before:

~~~
The column space of $[\,B_{i_1} | \cdots | B_{i_k}\,]$ lies in
$\operatorname{range}(Q)$, so step (2) computes the exact dominant left
~~~

after:

~~~
The column space of $[\,K_{i_1} | \cdots | K_{i_k}\,]$ lies in
$\operatorname{range}(Q)$, so step (2) computes the exact dominant left
~~~

### TEX-STM-0306 — `docs/shared_t3_math.tex` 306 → 316, LaTeX equation line

unchanged: R5: B_i = W_[i] diag(s) V^T is the factor itself (plan §4 item 8: stays)

### TEX-STM-0325 — `docs/shared_t3_math.tex` 325 → 335, LaTeX equation line

before:

~~~
$i$ satisfies $U_i^\top \dot V_i = 0$, and gauge conditions make all cross
~~~

after:

~~~
$i$ satisfies $D_i^\top \dot V_i = 0$, and gauge conditions make all cross
~~~

### TEX-STM-0347 — `docs/shared_t3_math.tex` 347–348 → 357–358, LaTeX equation line

before:

~~~
a single \emph{gauged ambient direction} $\dot U \in \R^{N_g \times n_g}$
with $U_g^\top \dot U = 0$: in the ($S$-absorbed) frame coordinates,
~~~

after:

~~~
a single \emph{gauged ambient direction} $\dot D \in \R^{N_g \times n_g}$
with $D_g^\top \dot D = 0$: in the ($S$-absorbed) frame coordinates,
~~~

### TEX-STM-0349 — `docs/shared_t3_math.tex` 349–351 → 359–361, LaTeX equation line

before:

~~~
\begin{equation}\label{eq:tied}
C_i = \dot U\, S_i \quad \text{for a common gauged } \dot U, \qquad i \in g .
\end{equation}
~~~

after:

~~~
\begin{equation}\label{eq:tied}
C_i = \dot D\, S_i \quad \text{for a common gauged } \dot D, \qquad i \in g .
\end{equation}
~~~

### TEX-STM-0358 — `docs/shared_t3_math.tex` 358 → 368, LaTeX equation line

before:

~~~
$O_g$ of $U_g$ ($\dot U = O_g B$, coordinates $C_i = B S_i$), suggesting a
~~~

after:

~~~
$O_g$ of $D_g$ ($\dot D = O_g Z$, coordinates $C_i = Z S_i$), suggesting a
~~~

### TEX-STM-0360 — `docs/shared_t3_math.tex` 360 → 370, LaTeX equation line

before:

~~~
subspace depends only on $U_g$ (tied by hypothesis) and the $S_i$; no
~~~

after:

~~~
subspace depends only on $D_g$ (tied by hypothesis) and the $S_i$; no
~~~

### TEX-STM-0363 — `docs/shared_t3_math.tex` 363 → 373, LaTeX equation line

unchanged: R5: Q_i is a rotation (§1.5 item 2; sample batch 3 §13 row 10)

### TEX-STM-0366 — `docs/shared_t3_math.tex` 366–367 → 376–377, LaTeX equation line

before:

~~~
implementation $S_i \in \R^{n_g \times n_{D,i}}$ with
$n_{D,i} = \min(n_g,\ r_{i-1} r_i)$ --- rectangular in general, so an
~~~

after:

~~~
implementation $S_i \in \R^{n_g \times n_{U,i}}$ with
$n_{U,i} = \min(n_g,\ r_{i-1} r_i)$ --- rectangular in general, so an
~~~

### TEX-STM-0377 — `docs/shared_t3_math.tex` 377–384 → 387–394, LaTeX equation line

before:

~~~
\begin{equation}\label{eq:gramsolve}
\dot U \;=\; \Bigl(\sum_{i \in g} C_i S_i^\top\Bigr)
        \Bigl(\sum_{i \in g} S_i S_i^\top\Bigr)^{-1}
   \;=\; \Bigl(\sum_{i \in g} C_i S_i^\top\Bigr)
        \Bigl(\sum_{i \in g} \Gamma_i\Bigr)^{-1},
\qquad
C_i \leftarrow \dot U S_i .
\end{equation}
~~~

after:

~~~
\begin{equation}\label{eq:gramsolve}
\dot D \;=\; \Bigl(\sum_{i \in g} C_i S_i^\top\Bigr)
        \Bigl(\sum_{i \in g} S_i S_i^\top\Bigr)^{-1}
   \;=\; \Bigl(\sum_{i \in g} C_i S_i^\top\Bigr)
        \Bigl(\sum_{i \in g} \Gamma_i\Bigr)^{-1},
\qquad
C_i \leftarrow \dot D S_i .
\end{equation}
~~~

### TEX-STM-0385 — `docs/shared_t3_math.tex` 385 → 395, LaTeX equation line

before:

~~~
TT variations and singleton modes are unaffected, and $\dot U$ is
~~~

after:

~~~
TT variations and singleton modes are unaffected, and $\dot D$ is
~~~

### TEX-STM-0393 — `docs/shared_t3_math.tex` 393 → 403, LaTeX equation line

before:

~~~
Minimize $\sum_{i\in g} \| C_i - \dot U S_i \|_F^2$ over $\dot U$; setting the
~~~

after:

~~~
Minimize $\sum_{i\in g} \| C_i - \dot D S_i \|_F^2$ over $\dot D$; setting the
~~~

### TEX-STM-0395 — `docs/shared_t3_math.tex` 395 → 405, LaTeX equation line

before:

~~~
$\sum_i C_i S_i^\top = \dot U \sum_i S_i S_i^\top$, invertible whenever the
~~~

after:

~~~
$\sum_i C_i S_i^\top = \dot D \sum_i S_i S_i^\top$, invertible whenever the
~~~

### TEX-STM-0397 — `docs/shared_t3_math.tex` 397–398 → 407–408, LaTeX equation line

before:

~~~
$U_g^\top \sum_i C_i S_i^\top = \sum_i (U_g^\top C_i) S_i^\top = 0$, and the
Gram sum acts on the other side, so $U_g^\top \dot U = 0$.
~~~

after:

~~~
$D_g^\top \sum_i C_i S_i^\top = \sum_i (D_g^\top C_i) S_i^\top = 0$, and the
Gram sum acts on the other side, so $D_g^\top \dot D = 0$.
~~~

### TEX-STM-0404 — `docs/shared_t3_math.tex` 404–410 → 414–420, LaTeX equation line

before:

~~~
\[
M_g = \begin{bmatrix} S_{i_1}^\top \\ \vdots \\ S_{i_k}^\top \end{bmatrix}
\in \R^{(\sum_{i\in g} n_{D,i}) \times n_g},
\qquad
C^{\mathrm{stack}} = \begin{bmatrix} C_{i_1}^\top \\ \vdots \\
C_{i_k}^\top \end{bmatrix},
\]
~~~

after:

~~~
\[
M_g = \begin{bmatrix} S_{i_1}^\top \\ \vdots \\ S_{i_k}^\top \end{bmatrix}
\in \R^{(\sum_{i\in g} n_{U,i}) \times n_g},
\qquad
C^{\mathrm{stack}} = \begin{bmatrix} C_{i_1}^\top \\ \vdots \\
C_{i_k}^\top \end{bmatrix},
\]
~~~

### TEX-STM-0412 — `docs/shared_t3_math.tex` 412–414 → 422–424, LaTeX equation line

before:

~~~
$\dot U^\top = M_g^{+}\, C^{\mathrm{stack}}$ (then
$C_i^\top \leftarrow S_i^\top \dot U^\top$, the corresponding row block of
$M_g \dot U^\top$), computed from one thin SVD of $M_g$ with a relative
~~~

after:

~~~
$\dot D^\top = M_g^{+}\, C^{\mathrm{stack}}$ (then
$C_i^\top \leftarrow S_i^\top \dot D^\top$, the corresponding row block of
$M_g \dot D^\top$), computed from one thin SVD of $M_g$ with a relative
~~~

### TEX-STM-0442 — `docs/shared_t3_math.tex` 442 → 452, LaTeX equation line

before:

~~~
the raw cores $((U_i)_i, (G_i)_i)$, the metric is Euclidean on the core
~~~

after:

~~~
the raw cores $((B_i)_i, (G_i)_i)$, the metric is Euclidean on the core
~~~

### TEX-STM-0444 — `docs/shared_t3_math.tex` 444 → 454, LaTeX equation line

before:

~~~
parameter set $\{\,U_i = U_j,\ i, j \in g,\ \forall g\,\}$ is a linear
~~~

after:

~~~
parameter set $\{\,B_i = B_j,\ i, j \in g,\ \forall g\,\}$ is a linear
~~~

### TEX-STM-0446 — `docs/shared_t3_math.tex` 446 → 456, LaTeX equation line

before:

~~~
$\{\dot U_i \text{ equal within each group}\}$ (TT-core perturbations
~~~

after:

~~~
$\{\dot B_i \text{ equal within each group}\}$ (TT-core perturbations
~~~

### TEX-STM-0449 — `docs/shared_t3_math.tex` 449 → 459, LaTeX equation line

before:

~~~
$\dot U_i \mapsto \tfrac{1}{k_g} \sum_{j \in g} \dot U_j$. The additive
~~~

after:

~~~
$\dot B_i \mapsto \tfrac{1}{k_g} \sum_{j \in g} \dot B_j$. The additive
~~~

### TEX-STM-0472 — `docs/shared_t3_math.tex` 472 → 482, LaTeX equation line

before:

~~~
factor is $[\,U_g \,|\, \dot U_g\,]$, identical across each group), and
~~~

after:

~~~
factor is $[\,D_g \,|\, \dot D_g\,]$, identical across each group), and
~~~

### TEX-STM-0473 — `docs/shared_t3_math.tex` 473–475 → 483–485, LaTeX equation line

unchanged: R5: R(T, xi) (plan §4 item 8: stays) and P_sf

### TEX-STM-0484 — `docs/shared_t3_math.tex` 484 → 494, LaTeX equation line

before:

~~~
after the fact: per group, the doubled factor is $[\,U_g \,|\, \dot U\,]$
~~~

after:

~~~
after the fact: per group, the doubled factor is $[\,D_g \,|\, \dot D\,]$
~~~

### TEX-STM-0489 — `docs/shared_t3_math.tex` 489 → 499, LaTeX equation line

before:

~~~
per-mode factor blocks $[\,U_g \,|\, \dot U S_i\,]$ (the $S$-absorbed
~~~

after:

~~~
per-mode factor blocks $[\,D_g \,|\, \dot D S_i\,]$ (the $S$-absorbed
~~~

### TEX-STM-0491 — `docs/shared_t3_math.tex` 491–492 → 501–502, LaTeX equation line

before:

~~~
and, when the $n_{D,i}$ differ, even in shape, and no per-mode averaging
repairs this. $\dot U$ is recovered from tied coordinates by the
~~~

after:

~~~
and, when the $n_{U,i}$ differ, even in shape, and no per-mode averaging
repairs this. $\dot D$ is recovered from tied coordinates by the
~~~

### TEX-STM-0551 — `docs/shared_t3_math.tex` 551 → 561, LaTeX equation line

before:

~~~
\item The Gram of the tied parameterization $B \mapsto (BS_i)_{i \in g}$ is
~~~

after:

~~~
\item The Gram of the tied parameterization $Z \mapsto (ZS_i)_{i \in g}$ is
~~~

### TEX-STM-0553 — `docs/shared_t3_math.tex` 553 → 563, LaTeX equation line

before:

~~~
  $\|\xi\|_{\HS}^2 = \tr\bigl(B (\sum_i \Gamma_i) B^\top\bigr)$. Its
~~~

after:

~~~
  $\|\xi\|_{\HS}^2 = \tr\bigl(Z (\sum_i \Gamma_i) Z^\top\bigr)$. Its
~~~

### TEX-STM-0566 — `docs/shared_t3_math.tex` 566 → 576, LaTeX equation line

before:

~~~
frame coordinates: $\sum_i \|BS_i\|_F^2 = \tr(B(\sum_i S_iS_i^\top)B^\top)$.
~~~

after:

~~~
frame coordinates: $\sum_i \|ZS_i\|_F^2 = \tr(Z(\sum_i S_iS_i^\top)Z^\top)$.
~~~

### TEX-SPD-0048 — `docs/symmetric_probe_derivatives.tex` 48 → 48, LaTeX equation line

unchanged: R5: P is the derivative direction (§1.5 item 3)

### TEX-SPD-0066 — `docs/symmetric_probe_derivatives.tex` 66–67 → 66–67, LaTeX equation line

before:

~~~
A \emph{Tucker tensor train} is a pair of core families $\mathbb T = \big((\mathbf U_i)_{i=1}^d,\,
(\mathbf G_i)_{i=1}^d\big)$, where each \emph{Tucker factor} $\mathbf U_i \in \R^{n_i \times N_i}$
~~~

after:

~~~
A \emph{Tucker tensor train} is a pair of core families $\mathbb T = \big((\mathbf B_i)_{i=1}^d,\,
(\mathbf G_i)_{i=1}^d\big)$, where each \emph{Tucker factor} $\mathbf B_i \in \R^{n_i \times N_i}$
~~~

### TEX-SPD-0085 — `docs/symmetric_probe_derivatives.tex` 85–98 → 85–98, LaTeX equation line

before:

~~~
\begin{align}
  \boldsymbol\xi_i      &= \mathbf U_i \mathbf w_i \in \R^{n_i},
  & &\text{(projected probe)} \label{eq:xi}\\
  \boldsymbol\mu_0^\top  &= 1, &
  \boldsymbol\mu_i^\top &= \boldsymbol\mu_{i-1}^\top\, \mathbf G_i(\boldsymbol\xi_i) \in \R^{r_i},
  & &\text{(left pushthrough)} \label{eq:mu}\\
  \boldsymbol\nu_d      &= 1, &
  \boldsymbol\nu_i      &= \mathbf G_{i+1}(\boldsymbol\xi_{i+1})\, \boldsymbol\nu_{i+1} \in \R^{r_i},
  & &\text{(right pushthrough)} \label{eq:nu}\\
  \boldsymbol\eta_i     &= \boldsymbol\mu_{i-1}^\top\, \mathbf G_i\, \boldsymbol\nu_i \in \R^{n_i},
  & &\text{(combine, mode $i$ free)} \label{eq:eta}\\
  \mathbf z_i           &= \mathbf U_i^\top \boldsymbol\eta_i \in \R^{N_i}.
  & &\text{(lift)} \label{eq:z}
\end{align}
~~~

after:

~~~
\begin{align}
  \boldsymbol\xi_i      &= \mathbf B_i \mathbf w_i \in \R^{n_i},
  & &\text{(projected probe)} \label{eq:xi}\\
  \boldsymbol\mu_0^\top  &= 1, &
  \boldsymbol\mu_i^\top &= \boldsymbol\mu_{i-1}^\top\, \mathbf G_i(\boldsymbol\xi_i) \in \R^{r_i},
  & &\text{(left pushthrough)} \label{eq:mu}\\
  \boldsymbol\nu_d      &= 1, &
  \boldsymbol\nu_i      &= \mathbf G_{i+1}(\boldsymbol\xi_{i+1})\, \boldsymbol\nu_{i+1} \in \R^{r_i},
  & &\text{(right pushthrough)} \label{eq:nu}\\
  \boldsymbol\eta_i     &= \boldsymbol\mu_{i-1}^\top\, \mathbf G_i\, \boldsymbol\nu_i \in \R^{n_i},
  & &\text{(combine, mode $i$ free)} \label{eq:eta}\\
  \mathbf z_i           &= \mathbf B_i^\top \boldsymbol\eta_i \in \R^{N_i}.
  & &\text{(lift)} \label{eq:z}
\end{align}
~~~

### TEX-SPD-0101 — `docs/symmetric_probe_derivatives.tex` 101 → 101, LaTeX equation line

before:

~~~
$\mathbf G_i$, leaving the middle index free. The single Tucker factor $\mathbf U_i$ is used twice:
~~~

after:

~~~
$\mathbf G_i$, leaving the middle index free. The single Tucker factor $\mathbf B_i$ is used twice:
~~~

### TEX-SPD-0105 — `docs/symmetric_probe_derivatives.tex` 105 → 105, LaTeX equation line

unchanged: R5: P is the derivative direction (§1.5 item 3)

### TEX-SPD-0107 — `docs/symmetric_probe_derivatives.tex` 107 → 107, LaTeX equation line

unchanged: R5: P is the derivative direction (§1.5 item 3)

### TEX-SPD-0108 — `docs/symmetric_probe_derivatives.tex` 108–113 → 108–113, LaTeX equation line

before:

~~~
\begin{equation}
  \label{eq:goal}
  \mathbf z_i^{(t)} \;:=\; \frac{\dd{^t}}{\dd s^t}\, \mathbf z_i(\mathbf W + s\mathbf P)\Big|_{s=0}
  \;=\; D^t \mathbf z_i(\mathbf W)\,\mathbf P^{t},
  \qquad t = 0, 1, \dots, m.
\end{equation}
~~~

after:

~~~
\begin{equation}
  \label{eq:goal}
  \mathbf z_i^{(t)} \;:=\; \frac{\dd{^t}}{\dd s^t}\, \mathbf z_i(\mathbf W + s\mathbf P)\Big|_{s=0}
  \;=\; \mathrm d^t \mathbf z_i(\mathbf W)\,\mathbf P^{t},
  \qquad t = 0, 1, \dots, m.
\end{equation}
~~~

### TEX-SPD-0121 — `docs/symmetric_probe_derivatives.tex` 121 → 121, LaTeX equation line

unchanged: R5: P is the derivative direction (§1.5 item 3)

### TEX-SPD-0126 — `docs/symmetric_probe_derivatives.tex` 126 → 126, LaTeX equation line

unchanged: R5: P is the derivative direction (§1.5 item 3)

### TEX-SPD-0127 — `docs/symmetric_probe_derivatives.tex` 127–131 → 127–131, LaTeX equation line

unchanged: R5: P is the derivative direction (§1.5 item 3)

### TEX-SPD-0144 — `docs/symmetric_probe_derivatives.tex` 144–147 → 144–147, LaTeX equation line

before:

~~~
\[
  \jet{\boldsymbol\xi_i} = (\boldsymbol\xi_i,\ \delta\boldsymbol\xi_i,\ 0, \dots, 0),
  \qquad \delta\boldsymbol\xi_i := \mathbf U_i \mathbf p_i.
\]
~~~

after:

~~~
\[
  \jet{\boldsymbol\xi_i} = (\boldsymbol\xi_i,\ \delta\boldsymbol\xi_i,\ 0, \dots, 0),
  \qquad \delta\boldsymbol\xi_i := \mathbf B_i \mathbf p_i.
\]
~~~

### TEX-SPD-0152 — `docs/symmetric_probe_derivatives.tex` 152 → 152, LaTeX equation line

before:

~~~
$\boldsymbol\xi_i(s) = \mathbf U_i \mathbf w_i(s) = \mathbf U_i \mathbf w_i + s\,\mathbf U_i \mathbf
~~~

after:

~~~
$\boldsymbol\xi_i(s) = \mathbf B_i \mathbf w_i(s) = \mathbf B_i \mathbf w_i + s\,\mathbf B_i \mathbf
~~~

### TEX-SPD-0158 — `docs/symmetric_probe_derivatives.tex` 158–159 → 158–159, LaTeX equation line

unchanged: R5: the italic B is a bilinear map (plan §4 item 8: differs by font, stays)

### TEX-SPD-0161 — `docs/symmetric_probe_derivatives.tex` 161–165 → 161–165, LaTeX equation line

unchanged: R5: the italic B is a bilinear map (plan §4 item 8: differs by font, stays)

### TEX-SPD-0169 — `docs/symmetric_probe_derivatives.tex` 169–170 → 169–170, LaTeX equation line

unchanged: R5: the italic B is a bilinear map (plan §4 item 8: differs by font, stays)

### TEX-SPD-0195 — `docs/symmetric_probe_derivatives.tex` 195 → 195, LaTeX equation line

before:

~~~
U_i, \mathbf G_i$ are constants (independent of $s$), so the only $s$-dependence is carried by the
~~~

after:

~~~
B_i, \mathbf G_i$ are constants (independent of $s$), so the only $s$-dependence is carried by the
~~~

### TEX-SPD-0217 — `docs/symmetric_probe_derivatives.tex` 217 → 217, LaTeX equation line

unchanged: R5: the italic B is a bilinear map (plan §4 item 8: differs by font, stays)

### TEX-SPD-0241 — `docs/symmetric_probe_derivatives.tex` 241–248 → 241–248, LaTeX equation line

before:

~~~
\begin{align}
  \boldsymbol\eta_i^{(t)}
    &= \sum_{r=0}^{t} \binom{t}{r}\,
       \big(\boldsymbol\mu_{i-1}^{(r)}\big)^\top \mathbf G_i\, \boldsymbol\nu_i^{(t-r)},
    \label{eq:deta}\\[2pt]
  \mathbf z_i^{(t)} &= \mathbf U_i^\top \boldsymbol\eta_i^{(t)}.
    \label{eq:dz}
\end{align}
~~~

after:

~~~
\begin{align}
  \boldsymbol\eta_i^{(t)}
    &= \sum_{r=0}^{t} \binom{t}{r}\,
       \big(\boldsymbol\mu_{i-1}^{(r)}\big)^\top \mathbf G_i\, \boldsymbol\nu_i^{(t-r)},
    \label{eq:deta}\\[2pt]
  \mathbf z_i^{(t)} &= \mathbf B_i^\top \boldsymbol\eta_i^{(t)}.
    \label{eq:dz}
\end{align}
~~~

### TEX-SPD-0252 — `docs/symmetric_probe_derivatives.tex` 252–253 → 252–253, LaTeX equation line

unchanged: R5: the italic B is a bilinear map (plan §4 item 8: differs by font, stays)

### TEX-SPD-0256 — `docs/symmetric_probe_derivatives.tex` 256–257 → 256–257, LaTeX equation line

before:

~~~
linear map $\boldsymbol\eta_i \mapsto \mathbf U_i^\top \boldsymbol\eta_i$, whose jet acts
componentwise: $\mathbf z_i^{(t)} = \mathbf U_i^\top \boldsymbol\eta_i^{(t)}$.
~~~

after:

~~~
linear map $\boldsymbol\eta_i \mapsto \mathbf B_i^\top \boldsymbol\eta_i$, whose jet acts
componentwise: $\mathbf z_i^{(t)} = \mathbf B_i^\top \boldsymbol\eta_i^{(t)}$.
~~~

### TEX-SPD-0276 — `docs/symmetric_probe_derivatives.tex` 276–302 → 276–302, LaTeX equation line

before:

~~~
\begin{algorithm}[h]
\caption{Symmetric probe derivatives of a Tucker tensor train}
\label{alg:sweep}
\begin{algorithmic}[1]
\Require T3 $\big((\mathbf U_i),(\mathbf G_i)\big)$; inputs $\mathbf W$; direction $\mathbf P$; order $m$
\Ensure derivative jets $\jet{\mathbf z_i}$ for $i=1,\dots,d$
\For{$i = 1,\dots,d$} \Comment{project, forming the order-one input jets}
  \State $\boldsymbol\xi_i \gets \mathbf U_i \mathbf w_i$,\quad
         $\delta\boldsymbol\xi_i \gets \mathbf U_i \mathbf p_i$
\EndFor
\State $\jet{\boldsymbol\mu_0} \gets (1,0,\dots,0)$
\For{$i = 1,\dots,d$} \Comment{left sweep, Eq.~\eqref{eq:dmu}}
  \State $\boldsymbol\mu_i^{(t)} \gets \boldsymbol\mu_{i-1}^{(t)}\mathbf G_i(\boldsymbol\xi_i)
         + t\,\boldsymbol\mu_{i-1}^{(t-1)}\mathbf G_i(\delta\boldsymbol\xi_i)$ \textbf{ for } $t=0,\dots,m$
\EndFor
\State $\jet{\boldsymbol\nu_d} \gets (1,0,\dots,0)$
\For{$i = d,\dots,1$} \Comment{right sweep, Eq.~\eqref{eq:dnu}}
  \State $\boldsymbol\nu_{i-1}^{(t)} \gets \mathbf G_i(\boldsymbol\xi_i)\boldsymbol\nu_i^{(t)}
         + t\,\mathbf G_i(\delta\boldsymbol\xi_i)\boldsymbol\nu_i^{(t-1)}$ \textbf{ for } $t=0,\dots,m$
\EndFor
\For{$i = 1,\dots,d$} \Comment{combine and lift, Eqs.~\eqref{eq:deta}--\eqref{eq:dz}}
  \State $\boldsymbol\eta_i^{(t)} \gets \sum_{r=0}^{t}\binom{t}{r}
         \big(\boldsymbol\mu_{i-1}^{(r)}\big)^\top \mathbf G_i\,\boldsymbol\nu_i^{(t-r)}$ \textbf{ for } $t=0,\dots,m$
  \State $\mathbf z_i^{(t)} \gets \mathbf U_i^\top \boldsymbol\eta_i^{(t)}$ \textbf{ for } $t=0,\dots,m$
\EndFor
\end{algorithmic}
\end{algorithm}
~~~

after:

~~~
\begin{algorithm}[h]
\caption{Symmetric probe derivatives of a Tucker tensor train}
\label{alg:sweep}
\begin{algorithmic}[1]
\Require T3 $\big((\mathbf B_i),(\mathbf G_i)\big)$; inputs $\mathbf W$; direction $\mathbf P$; order $m$
\Ensure derivative jets $\jet{\mathbf z_i}$ for $i=1,\dots,d$
\For{$i = 1,\dots,d$} \Comment{project, forming the order-one input jets}
  \State $\boldsymbol\xi_i \gets \mathbf B_i \mathbf w_i$,\quad
         $\delta\boldsymbol\xi_i \gets \mathbf B_i \mathbf p_i$
\EndFor
\State $\jet{\boldsymbol\mu_0} \gets (1,0,\dots,0)$
\For{$i = 1,\dots,d$} \Comment{left sweep, Eq.~\eqref{eq:dmu}}
  \State $\boldsymbol\mu_i^{(t)} \gets \boldsymbol\mu_{i-1}^{(t)}\mathbf G_i(\boldsymbol\xi_i)
         + t\,\boldsymbol\mu_{i-1}^{(t-1)}\mathbf G_i(\delta\boldsymbol\xi_i)$ \textbf{ for } $t=0,\dots,m$
\EndFor
\State $\jet{\boldsymbol\nu_d} \gets (1,0,\dots,0)$
\For{$i = d,\dots,1$} \Comment{right sweep, Eq.~\eqref{eq:dnu}}
  \State $\boldsymbol\nu_{i-1}^{(t)} \gets \mathbf G_i(\boldsymbol\xi_i)\boldsymbol\nu_i^{(t)}
         + t\,\mathbf G_i(\delta\boldsymbol\xi_i)\boldsymbol\nu_i^{(t-1)}$ \textbf{ for } $t=0,\dots,m$
\EndFor
\For{$i = 1,\dots,d$} \Comment{combine and lift, Eqs.~\eqref{eq:deta}--\eqref{eq:dz}}
  \State $\boldsymbol\eta_i^{(t)} \gets \sum_{r=0}^{t}\binom{t}{r}
         \big(\boldsymbol\mu_{i-1}^{(r)}\big)^\top \mathbf G_i\,\boldsymbol\nu_i^{(t-r)}$ \textbf{ for } $t=0,\dots,m$
  \State $\mathbf z_i^{(t)} \gets \mathbf B_i^\top \boldsymbol\eta_i^{(t)}$ \textbf{ for } $t=0,\dots,m$
\EndFor
\end{algorithmic}
\end{algorithm}
~~~

### TEX-SPD-0349 — `docs/symmetric_probe_derivatives.tex` 349 → 349, LaTeX equation line

unchanged: R5: P is the derivative direction (§1.5 item 3)

### TEX-SPD-0353 — `docs/symmetric_probe_derivatives.tex` 353 → 353, LaTeX equation line

unchanged: R5: P is the derivative direction (§1.5 item 3)

### TEX-SPD-0373 — `docs/symmetric_probe_derivatives.tex` 373–374 → 373–374, LaTeX equation line

before:

~~~
$\big((\mathbf U_i),(\mathbf O_i),(\mathbf P_i),(\mathbf Q_i)\big)$ (up-Tucker, down-, left-, and
right-TT cores) and a gauged variation $\delta\mathbb V = \big((\delta\mathbf U_i),
~~~

after:

~~~
$\big((\mathbf D_i),(\mathbf U_i),(\mathbf L_i),(\mathbf R_i)\big)$ (down-Tucker, up-, left-, and
right-TT cores) and a gauged variation $\delta\mathbb V = \big((\mathbf V_i),
~~~

### TEX-SPD-0377 — `docs/symmetric_probe_derivatives.tex` 377–383 → 377–383, LaTeX equation line

before:

~~~
\begin{equation}
  \label{eq:rbase}
  \boldsymbol\xi_i = \mathbf U_i \mathbf w_i, \quad
  \boldsymbol\mu_i^\top = \boldsymbol\mu_{i-1}^\top \mathbf P_i(\boldsymbol\xi_i), \quad
  \boldsymbol\nu_i = \mathbf Q_{i+1}(\boldsymbol\xi_{i+1})\,\boldsymbol\nu_{i+1}, \quad
  \boldsymbol\eta_i = \boldsymbol\mu_{i-1}^\top \mathbf O_i\, \boldsymbol\nu_i,
\end{equation}
~~~

after:

~~~
\begin{equation}
  \label{eq:rbase}
  \boldsymbol\xi_i = \mathbf D_i \mathbf w_i, \quad
  \boldsymbol\mu_i^\top = \boldsymbol\mu_{i-1}^\top \mathbf L_i(\boldsymbol\xi_i), \quad
  \boldsymbol\nu_i = \mathbf R_{i+1}(\boldsymbol\xi_{i+1})\,\boldsymbol\nu_{i+1}, \quad
  \boldsymbol\eta_i = \boldsymbol\mu_{i-1}^\top \mathbf U_i\, \boldsymbol\nu_i,
\end{equation}
~~~

### TEX-SPD-0384 — `docs/symmetric_probe_derivatives.tex` 384 → 384, LaTeX equation line

before:

~~~
(the plain pipeline \eqref{eq:xi}--\eqref{eq:z} with the frame cores $\mathbf P,\mathbf Q,\mathbf O$
~~~

after:

~~~
(the plain pipeline \eqref{eq:xi}--\eqref{eq:z} with the frame cores $\mathbf L,\mathbf R,\mathbf U$
~~~

### TEX-SPD-0386 — `docs/symmetric_probe_derivatives.tex` 386–399 → 386–399, LaTeX equation line

before:

~~~
\begin{align}
  \delta\boldsymbol\xi_i &= \delta\mathbf U_i \mathbf w_i, \label{eq:rdxi}\\
  \boldsymbol\sigma_i^\top &= \boldsymbol\sigma_{i-1}^\top \mathbf Q_i(\boldsymbol\xi_i)
    + \boldsymbol\mu_{i-1}^\top \delta\mathbf G_i(\boldsymbol\xi_i)
    + \boldsymbol\mu_{i-1}^\top \mathbf O_i(\delta\boldsymbol\xi_i), \label{eq:rsigma}\\
  \boldsymbol\tau_i &= \delta\mathbf G_{i+1}(\boldsymbol\xi_{i+1})\boldsymbol\nu_{i+1}
    + \mathbf O_{i+1}(\delta\boldsymbol\xi_{i+1})\boldsymbol\nu_{i+1}
    + \mathbf P_{i+1}(\boldsymbol\xi_{i+1})\boldsymbol\tau_{i+1}, \label{eq:rtau}\\
  \delta\boldsymbol\eta_i &= \boldsymbol\sigma_{i-1}^\top \mathbf Q_i \boldsymbol\nu_i
    + \boldsymbol\mu_{i-1}^\top \delta\mathbf G_i \boldsymbol\nu_i
    + \boldsymbol\mu_{i-1}^\top \mathbf P_i \boldsymbol\tau_i, \label{eq:rdeta}\\
  \mathbf z_i &= \mathbf U_i^\top \delta\boldsymbol\eta_i + \delta\mathbf U_i^\top \boldsymbol\eta_i,
    \label{eq:rz}
\end{align}
~~~

after:

~~~
\begin{align}
  \delta\boldsymbol\xi_i &= \mathbf V_i \mathbf w_i, \label{eq:rdxi}\\
  \boldsymbol\sigma_i^\top &= \boldsymbol\sigma_{i-1}^\top \mathbf R_i(\boldsymbol\xi_i)
    + \boldsymbol\mu_{i-1}^\top \mathbf H_i(\boldsymbol\xi_i)
    + \boldsymbol\mu_{i-1}^\top \mathbf U_i(\delta\boldsymbol\xi_i), \label{eq:rsigma}\\
  \boldsymbol\tau_i &= \mathbf H_{i+1}(\boldsymbol\xi_{i+1})\boldsymbol\nu_{i+1}
    + \mathbf U_{i+1}(\delta\boldsymbol\xi_{i+1})\boldsymbol\nu_{i+1}
    + \mathbf L_{i+1}(\boldsymbol\xi_{i+1})\boldsymbol\tau_{i+1}, \label{eq:rtau}\\
  \delta\boldsymbol\eta_i &= \boldsymbol\sigma_{i-1}^\top \mathbf R_i \boldsymbol\nu_i
    + \boldsymbol\mu_{i-1}^\top \mathbf H_i \boldsymbol\nu_i
    + \boldsymbol\mu_{i-1}^\top \mathbf L_i \boldsymbol\tau_i, \label{eq:rdeta}\\
  \mathbf z_i &= \mathbf D_i^\top \delta\boldsymbol\eta_i + \mathbf V_i^\top \boldsymbol\eta_i,
    \label{eq:rz}
\end{align}
~~~

### TEX-SPD-0403 — `docs/symmetric_probe_derivatives.tex` 403 → 408, LaTeX equation line

unchanged: R5: P is the derivative direction (§1.5 item 3)

### TEX-SPD-0407 — `docs/symmetric_probe_derivatives.tex` 407–408 → 412–413, LaTeX equation line

before:

~~~
order-one input jets $\jet{\boldsymbol\xi_i} = (\boldsymbol\xi_i, \mathbf U_i\mathbf p_i, 0, \dots)$
and $\jet{\delta\boldsymbol\xi_i} = (\delta\boldsymbol\xi_i, \delta\mathbf U_i\mathbf p_i, 0, \dots)$,
~~~

after:

~~~
order-one input jets $\jet{\boldsymbol\xi_i} = (\boldsymbol\xi_i, \mathbf D_i\mathbf p_i, 0, \dots)$
and $\jet{\delta\boldsymbol\xi_i} = (\delta\boldsymbol\xi_i, \mathbf V_i\mathbf p_i, 0, \dots)$,
~~~

### TEX-SPD-0410 — `docs/symmetric_probe_derivatives.tex` 410–419 → 415–424, LaTeX equation line

before:

~~~
\begin{align}
  \big(\boldsymbol\sigma_i^{(t)}\big)^\top
  &= \big(\boldsymbol\sigma_{i-1}^{(t)}\big)^\top \mathbf Q_i(\boldsymbol\xi_i)
   + t \big(\boldsymbol\sigma_{i-1}^{(t-1)}\big)^\top \mathbf Q_i(\mathbf U_i\mathbf p_i) \nonumber\\
  &\;+ \big(\boldsymbol\mu_{i-1}^{(t)}\big)^\top \delta\mathbf G_i(\boldsymbol\xi_i)
     + t \big(\boldsymbol\mu_{i-1}^{(t-1)}\big)^\top \delta\mathbf G_i(\mathbf U_i\mathbf p_i) \nonumber\\
  &\;+ \big(\boldsymbol\mu_{i-1}^{(t)}\big)^\top \mathbf O_i(\delta\boldsymbol\xi_i)
     + t \big(\boldsymbol\mu_{i-1}^{(t-1)}\big)^\top \mathbf O_i(\delta\mathbf U_i\mathbf p_i),
  \label{eq:rdsigma}
\end{align}
~~~

after:

~~~
\begin{align}
  \big(\boldsymbol\sigma_i^{(t)}\big)^\top
  &= \big(\boldsymbol\sigma_{i-1}^{(t)}\big)^\top \mathbf R_i(\boldsymbol\xi_i)
   + t \big(\boldsymbol\sigma_{i-1}^{(t-1)}\big)^\top \mathbf R_i(\mathbf D_i\mathbf p_i) \nonumber\\
  &\;+ \big(\boldsymbol\mu_{i-1}^{(t)}\big)^\top \mathbf H_i(\boldsymbol\xi_i)
     + t \big(\boldsymbol\mu_{i-1}^{(t-1)}\big)^\top \mathbf H_i(\mathbf D_i\mathbf p_i) \nonumber\\
  &\;+ \big(\boldsymbol\mu_{i-1}^{(t)}\big)^\top \mathbf U_i(\delta\boldsymbol\xi_i)
     + t \big(\boldsymbol\mu_{i-1}^{(t-1)}\big)^\top \mathbf U_i(\mathbf V_i\mathbf p_i),
  \label{eq:rdsigma}
\end{align}
~~~

### TEX-SPD-0428 — `docs/symmetric_probe_derivatives.tex` 428 → 433, LaTeX equation line

before:

~~~
Section~\ref{sec:impl}, fed a frame core ($\mathbf P,\mathbf Q,\mathbf O$) or the variation core
~~~

after:

~~~
Section~\ref{sec:impl}, fed a frame core ($\mathbf L,\mathbf R,\mathbf U$) or the variation core
~~~

### TEX-SPD-0438 — `docs/symmetric_probe_derivatives.tex` 438 → 443, LaTeX equation line

before:

~~~
the action-jet space, return the variation gradient $\big((\delta\widetilde{\mathbf U}_i),
~~~

after:

~~~
the action-jet space, return the variation gradient $\big((\widetilde{\mathbf V}_i),
~~~

### TEX-SPD-0445 — `docs/symmetric_probe_derivatives.tex` 445–449 → 450–454, LaTeX equation line

unchanged: R5: calligraphic L is the Lagrangian (not a store letter)

### TEX-SPD-0452 — `docs/symmetric_probe_derivatives.tex` 452 → 457, LaTeX equation line

unchanged: R5: calligraphic L is the Lagrangian (not a store letter)

### TEX-SPD-0454 — `docs/symmetric_probe_derivatives.tex` 454 → 459, LaTeX equation line

unchanged: R5: calligraphic L is the Lagrangian (not a store letter)

### TEX-SPD-0456 — `docs/symmetric_probe_derivatives.tex` 456 → 461, LaTeX equation line

unchanged: R5: calligraphic L is the Lagrangian (not a store letter)

### TEX-SPD-0457 — `docs/symmetric_probe_derivatives.tex` 457–461 → 462–466, LaTeX equation line

before:

~~~
\begin{equation}
  \label{eq:grad}
  \big((\delta\widetilde{\mathbf U}_i),(\delta\widetilde{\mathbf G}_i)\big)
  = \partial_{\delta\mathbb V}\, \mathcal L = \big(J^{(s)}\big)^\top(\mathbf r).
\end{equation}
~~~

after:

~~~
\begin{equation}
  \label{eq:grad}
  \big((\widetilde{\mathbf V}_i),(\widetilde{\mathbf H}_i)\big)
  = \partial_{\delta\mathbb V}\, \mathcal L = \big(J^{(s)}\big)^\top(\mathbf r).
\end{equation}
~~~

### TEX-SPD-0466 — `docs/symmetric_probe_derivatives.tex` 466 → 471, LaTeX equation line

unchanged: R5: calligraphic L is the Lagrangian (not a store letter)

### TEX-SPD-0482 — `docs/symmetric_probe_derivatives.tex` 482 → 487, LaTeX equation line

before:

~~~
\eta}_i = \mathbf U_i \mathbf r_i$ is the order-diagonal lift.
~~~

after:

~~~
\eta}_i = \mathbf D_i \mathbf r_i$ is the order-diagonal lift.
~~~

### TEX-SPD-0485 — `docs/symmetric_probe_derivatives.tex` 485 → 490, LaTeX equation line

before:

~~~
The variation gradients $\partial_{\delta\mathbf G_i}\mathcal L$ and $\partial_{\delta\mathbf U_i}
~~~

after:

~~~
The variation gradients $\partial_{\mathbf H_i}\mathcal L$ and $\partial_{\mathbf V_i}
~~~

### TEX-SPD-0500 — `docs/symmetric_probe_derivatives.tex` 500 → 505, LaTeX equation line

before:

~~~
  \item the Tucker core $\delta\mathbf U_i$ has one internal edge (the down-edge to its TT core), so
~~~

after:

~~~
  \item the Tucker core $\mathbf V_i$ has one internal edge (the edge up to its TT core), so
~~~

### TEX-SPD-0502 — `docs/symmetric_probe_derivatives.tex` 502–507 → 507–512, LaTeX equation line

before:

~~~
    \[
      \delta\widetilde{\mathbf U}_i
      = \sum_t \boldsymbol\eta_i^{(t)} \otimes \mathbf r_i^{(t)}
      + \sum_u \widehat{\delta\boldsymbol\xi}_i^{(u)} \otimes \mathbf w_i^{(u)},
      \qquad \jet{\mathbf w_i} = (\mathbf w_i, \mathbf p_i, 0, \dots).
    \]
~~~

after:

~~~
    \[
      \widetilde{\mathbf V}_i
      = \sum_t \boldsymbol\eta_i^{(t)} \otimes \mathbf r_i^{(t)}
      + \sum_u \widehat{\delta\boldsymbol\xi}_i^{(u)} \otimes \mathbf w_i^{(u)},
      \qquad \jet{\mathbf w_i} = (\mathbf w_i, \mathbf p_i, 0, \dots).
    \]
~~~

### TEX-SPD-0540 — `docs/symmetric_probe_derivatives.tex` 540 → 545, LaTeX equation line

before:

~~~
\sigma}_i$, $\delta\widetilde{\mathbf U}_i = \widehat{\delta\boldsymbol\xi}_i\otimes\mathbf w_i$ (about
~~~

after:

~~~
\sigma}_i$, $\widetilde{\mathbf V}_i = \widehat{\delta\boldsymbol\xi}_i\otimes\mathbf w_i$ (about
~~~

### TEX-SPD-0542 — `docs/symmetric_probe_derivatives.tex` 542–543 → 547–548, LaTeX equation line

unchanged: R5: P is the derivative direction (§1.5 item 3)

### TEX-SPD-0547 — `docs/symmetric_probe_derivatives.tex` 547 → 552, LaTeX equation line

unchanged: R5: P is the derivative direction (§1.5 item 3)

### TEX-SPD-0559 — `docs/symmetric_probe_derivatives.tex` 559 → 564–565, LaTeX equation line

before:

~~~
($\mathbf P,\mathbf Q,\mathbf O\to\mathbf G_i$, $\mathbf U$ no longer required orthogonal). It is a
~~~

after:

~~~
$(\mathbf B,\mathbf G,\mathbf G,\mathbf G)$ ($\mathbf B\to\mathbf D$, $\mathbf G\to\mathbf U,\mathbf L,\mathbf R$).
The manifold transpose requires an orthogonal frame; the corewise one works for any $\mathbf B$, $\mathbf G$.
~~~

### TEX-SPD-0631 — `docs/symmetric_probe_derivatives.tex` 631 → 637, LaTeX equation line

before:

~~~
\delta\widetilde{\mathbf U}_i$: order-summed outer products of the tilde edge variables with the base
~~~

after:

~~~
\widetilde{\mathbf V}_i$: order-summed outer products of the tilde edge variables with the base
~~~

### TEX-SPD-0633 — `docs/symmetric_probe_derivatives.tex` 633 → 639, LaTeX equation line

unchanged: R5: (r_L, n, r_R): r_L, r_R unchanged (D-3), generic n (§1.5 item 5)

### TEX-TTHN-0073 — `docs/ttm_t3m_ht_note.tex` 73–77 → 74–78, LaTeX equation line

before:

~~~
\begin{equation}
\T[x]=\sum_{n_0,\dots,n_{d-1}} \Big(\prod_{i=0}^{d-1} U_i[n_i,x_i]\Big)\,\C[n_0,\dots,n_{d-1}],
\qquad \C=\TT(G),
\label{eq:t3}
\end{equation}
~~~

after:

~~~
\begin{equation}
\T[x]=\sum_{n_0,\dots,n_{d-1}} \Big(\prod_{i=0}^{d-1} B_i[n_i,x_i]\Big)\,\C[n_0,\dots,n_{d-1}],
\qquad \C=\TT(G),
\label{eq:t3}
\end{equation}
~~~

### TEX-TTHN-0078 — `docs/ttm_t3m_ht_note.tex` 78 → 79, LaTeX equation line

before:

~~~
with \emph{Tucker factors} (leaf frames) $U_i\in\R^{n_i\times N_i}$ and central TT
~~~

after:

~~~
with \emph{Tucker factors} (leaf frames) $B_i\in\R^{n_i\times N_i}$ and central TT
~~~

### TEX-TTHN-0080 — `docs/ttm_t3m_ht_note.tex` 80 → 81–87, LaTeX equation line

before:

~~~
Plain TT is the special case $n_i=N_i$, $U_i=I$.
~~~

after:

~~~
Plain TT is the special case $n_i=N_i$, $B_i=I$.

T4S writes $U$, $O$, $P$, $Q$ for our $D$, $U$, $L$, $R$; its $\delta U$ is our $V$
(\texttt{tucker\_variations}), its $\delta G$ our $H$ (\texttt{tt\_variations}). It writes $U_i$ for the
Tucker factors $B_i$, and HT papers write $U_t$ for the leaf frames and $B_t$ for the transfer tensors
($T_t$ here, \S\ref{sec:ht}); this note wrote $\mathcal A$, $\mathcal B$, $\mathcal C$ (superscripts $A$, $B$, $C$)
for $\X$, $\Y$, $\Z$.
~~~

### TEX-TTHN-0084 — `docs/ttm_t3m_ht_note.tex` 84–87 → 91–94, LaTeX equation line

before:

~~~
\begin{align}
(U^A\krrow U^B)[(a,b),\,x] &= U^A[a,x]\,U^B[b,x] && \text{(row-wise Khatri--Rao, ``COPY'')},\\
(G^A\kron G^B)[(\alpha,\alpha'),(a,b),(\beta,\beta')] &= G^A[\alpha,a,\beta]\,G^B[\alpha',a',\beta'] && \text{(Kronecker)} .
\end{align}
~~~

after:

~~~
\begin{align}
(B^X\krrow B^Y)[(a,b),\,x] &= B^X[a,x]\,B^Y[b,x] && \text{(row-wise Khatri--Rao, ``COPY'')},\\
(G^X\kron G^Y)[(\alpha,\alpha'),(a,b),(\beta,\beta')] &= G^X[\alpha,a,\beta]\,G^Y[\alpha',a',\beta'] && \text{(Kronecker)} .
\end{align}
~~~

### TEX-TTHN-0092 — `docs/ttm_t3m_ht_note.tex` 92–97 → 99–104, LaTeX equation line

before:

~~~
\begin{equation}
W_i \;=\; U^A_i\krrow U^B_i \in\R^{(n^A_i n^B_i)\times N_i},
\qquad
G^C_i \;=\; G^A_i\kron G^B_i \in\R^{(r^A_i r^B_i)\times(n^A_i n^B_i)\times(r^A_{i+1}r^B_{i+1})}.
\label{eq:prod}
\end{equation}
~~~

after:

~~~
\begin{equation}
W_i \;=\; B^X_i\krrow B^Y_i \in\R^{(n^X_i n^Y_i)\times N_i},
\qquad
G^Z_i \;=\; G^X_i\kron G^Y_i \in\R^{(r^X_i r^Y_i)\times(n^X_i n^Y_i)\times(r^X_{i+1}r^Y_{i+1})}.
\label{eq:prod}
\end{equation}
~~~

### TEX-TTHN-0099 — `docs/ttm_t3m_ht_note.tex` 99 → 106, LaTeX equation line

before:

~~~
$n^A_i n^B_i$ and TT bond $r^A_i r^B_i$. The point of every algorithm below is to
~~~

after:

~~~
$n^X_i n^Y_i$ and TT bond $r^X_i r^Y_i$. The point of every algorithm below is to
~~~

### TEX-TTHN-0114 — `docs/ttm_t3m_ht_note.tex` 114–130 → 121–137, LaTeX equation line

before:

~~~
\begin{algorithm}[t]
\caption{\textsc{TTM} --- Hadamard product of two tensor trains (swap form)}
\label{alg:ttm}
\begin{algorithmic}[1]
\Require TT cores $\{G^A_i\}$, $\{G^B_i\}$; per-step tolerance / target bond $\rtil$
\State $H \gets [\,G^A_0,\dots,G^A_{d-1},\ \textsc{reverse}(G^B)\,]$ \Comment{length-$2d$ chain $=\A\kron\B$; reversal makes mode $d{-}1$ adjacent}
\State left-orthogonalize the $A$-half, right-orthogonalize the reversed-$B$ half \Comment{mixed-canonical center at the seam}
\For{$i=d-1$ \textbf{downto} $0$}
  \State move the orthogonality center to the right end of the merged block
  \For{each step bringing the mode-$i$ $B$-core toward its $A$-partner}
    \State \textsc{swap} the adjacent pair by a \emph{centered} truncated SVD \Comment{bond stays $O(\rtil)$}
  \EndFor
  \State \textsc{merge} the adjacent matching pair with the copy tensor $\delta$ (diagonal in $x_i$); single bonds in/out
\EndFor
\State \Return the length-$d$ chain $\{G^C_i\}$
\end{algorithmic}
\end{algorithm}
~~~

after:

~~~
\begin{algorithm}[t]
\caption{\textsc{TTM} --- Hadamard product of two tensor trains (swap form)}
\label{alg:ttm}
\begin{algorithmic}[1]
\Require TT cores $\{G^X_i\}$, $\{G^Y_i\}$; per-step tolerance / target bond $\rtil$
\State $H \gets [\,G^X_0,\dots,G^X_{d-1},\ \textsc{reverse}(G^Y)\,]$ \Comment{length-$2d$ chain $=\X\kron\Y$; reversal makes mode $d{-}1$ adjacent}
\State left-orthogonalize the $X$-half, right-orthogonalize the reversed-$Y$ half \Comment{mixed-canonical center at the seam}
\For{$i=d-1$ \textbf{downto} $0$}
  \State move the orthogonality center to the right end of the merged block
  \For{each step bringing the mode-$i$ $Y$-core toward its $X$-partner}
    \State \textsc{swap} the adjacent pair by a \emph{centered} truncated SVD \Comment{bond stays $O(\rtil)$}
  \EndFor
  \State \textsc{merge} the adjacent matching pair with the copy tensor $\delta$ (diagonal in $x_i$); single bonds in/out
\EndFor
\State \Return the length-$d$ chain $\{G^Z_i\}$
\end{algorithmic}
\end{algorithm}
~~~

### TEX-TTHN-0133 — `docs/ttm_t3m_ht_note.tex` 133 → 140, LaTeX equation line

before:

~~~
``bubbled past'' by later $B$-cores. Because every swap and merge is performed in
~~~

after:

~~~
``bubbled past'' by later $Y$-cores. Because every swap and merge is performed in
~~~

### TEX-TTHN-0149 — `docs/ttm_t3m_ht_note.tex` 149 → 156, LaTeX equation line

before:

~~~
is the pair $(n^A_i,n^B_i)$ carrying the Tucker factor $W_i=U^A_i\krrow U^B_i$ from
~~~

after:

~~~
is the pair $(n^X_i,n^Y_i)$ carrying the Tucker factor $W_i=B^X_i\krrow B^Y_i$ from
~~~

### TEX-TTHN-0151 — `docs/ttm_t3m_ht_note.tex` 151–153 → 158–160, LaTeX equation line

before:

~~~
\begin{equation}
\rank(W_i)\le \min(n^A_i n^B_i,\;N_i),\qquad\text{truncate to } \ntil_i,
\end{equation}
~~~

after:

~~~
\begin{equation}
\rank(W_i)\le \min(n^X_i n^Y_i,\;N_i),\qquad\text{truncate to } \ntil_i,
\end{equation}
~~~

### TEX-TTHN-0158 — `docs/ttm_t3m_ht_note.tex` 158–178 → 165–185, LaTeX equation line

before:

~~~
\begin{algorithm}[t]
\caption{\textsc{T3M-swap} --- Hadamard product of two Tucker tensor trains}
\label{alg:t3m}
\begin{algorithmic}[1]
\Require T3s $(U^A,G^A)$, $(U^B,G^B)$; targets $\ntil,\rtil$ (and/or $\mathrm{rtol},\mathrm{atol}$); oversample $k\ge1$
\State down-orthogonalize the Tucker factors of each input \Comment{HT: ``orthogonalize leaves toward the root''}
\State $H \gets [\,G^A_0,\dots,G^A_{d-1},\ \textsc{reverse}(G^B)\,]$ with each $G_i$ carrying $(U_i,\text{mode }i)$
\State left-orthogonalize the $A$-half, right-orthogonalize the reversed-$B$ half \Comment{center at the seam}
\For{$i=d-1$ \textbf{downto} $0$}
  \State move the center to the right end of the merged block
  \For{each step bringing the mode-$i$ $B$-core to its $A$-partner}
    \State centered truncated-SVD \textsc{swap}, intermediate bond capped at $k\,\rtil$ (or tol $/k$) \Comment{memory only}
  \EndFor
  \State \textsc{merge}: $W_i\gets U^A_i\krrow U^B_i$; centered weighted-SVD compress to $\le k\,\ntil$; Tucker factors/modes ride along
\EndFor
\If{$k>1$ \textbf{or} a non-uniform per-position bond cap was requested}
  \State $\{(U^C_i,G^C_i)\}\gets$ \textsc{T3-SVD-round}$(\{(U^C_i,G^C_i)\};\ \ntil,\rtil,\mathrm{rtol},\mathrm{atol})$ \Comment{the decisive truncation}
\EndIf
\State \Return $\{(U^C_i,G^C_i)\}$
\end{algorithmic}
\end{algorithm}
~~~

after:

~~~
\begin{algorithm}[t]
\caption{\textsc{T3M-swap} --- Hadamard product of two Tucker tensor trains}
\label{alg:t3m}
\begin{algorithmic}[1]
\Require T3s $(B^X,G^X)$, $(B^Y,G^Y)$; targets $\ntil,\rtil$ (and/or $\mathrm{rtol},\mathrm{atol}$); oversample $k\ge1$
\State down-orthogonalize the Tucker factors of each input \Comment{HT: ``orthogonalize leaves toward the root''}
\State $H \gets [\,G^X_0,\dots,G^X_{d-1},\ \textsc{reverse}(G^Y)\,]$ with each $G_i$ carrying $(B_i,\text{mode }i)$
\State left-orthogonalize the $X$-half, right-orthogonalize the reversed-$Y$ half \Comment{center at the seam}
\For{$i=d-1$ \textbf{downto} $0$}
  \State move the center to the right end of the merged block
  \For{each step bringing the mode-$i$ $Y$-core to its $X$-partner}
    \State centered truncated-SVD \textsc{swap}, intermediate bond capped at $k\,\rtil$ (or tol $/k$) \Comment{memory only}
  \EndFor
  \State \textsc{merge}: $W_i\gets B^X_i\krrow B^Y_i$; centered weighted-SVD compress to $\le k\,\ntil$; Tucker factors/modes ride along
\EndFor
\If{$k>1$ \textbf{or} a non-uniform per-position bond cap was requested}
  \State $\{(B^Z_i,G^Z_i)\}\gets$ \textsc{T3-SVD-round}$(\{(B^Z_i,G^Z_i)\};\ \ntil,\rtil,\mathrm{rtol},\mathrm{atol})$ \Comment{the decisive truncation}
\EndIf
\State \Return $\{(B^Z_i,G^Z_i)\}$
\end{algorithmic}
\end{algorithm}
~~~

### TEX-TTHN-0186 — `docs/ttm_t3m_ht_note.tex` 186 → 193, LaTeX equation line

before:

~~~
the Khatri--Rao compression $W_i\!=\!U^A_i\krrow U^B_i$ \textbf{couples} the two
~~~

after:

~~~
the Khatri--Rao compression $W_i\!=\!B^X_i\krrow B^Y_i$ \textbf{couples} the two
~~~

### TEX-TTHN-0217 — `docs/ttm_t3m_ht_note.tex` 217–218 → 224–225, LaTeX equation line

before:

~~~
An HT tensor is a \emph{dimension tree} with leaf frames $U_i\in\R^{n_i\times N_i}$
and transfer tensors $B_t\in\R^{n_{t_1}\times n_{t_2}\times n_t}$ at internal nodes
~~~

after:

~~~
An HT tensor is a \emph{dimension tree} with leaf frames $B_i\in\R^{n_i\times N_i}$
and transfer tensors $T_t\in\R^{n_{t_1}\times n_{t_2}\times n_t}$ at internal nodes
~~~

### TEX-TTHN-0222 — `docs/ttm_t3m_ht_note.tex` 222–230 → 229–237, LaTeX table or figure

before:

~~~
\begin{tabular}{ll}
\toprule
format & HT realization\\
\midrule
Tensor train & linear (caterpillar) tree, \textbf{trivial} leaf frames ($U_i=I$)\\
Tucker & star (one root transfer tensor, $d$ leaf frames)\\
Tucker tensor train (T3) & linear tree, \textbf{nontrivial} leaf frames\\
\bottomrule
\end{tabular}
~~~

after:

~~~
\begin{tabular}{ll}
\toprule
format & HT realization\\
\midrule
Tensor train & linear (caterpillar) tree, \textbf{trivial} leaf frames ($B_i=I$)\\
Tucker & star (one root transfer tensor, $d$ leaf frames)\\
Tucker tensor train (T3) & linear tree, \textbf{nontrivial} leaf frames\\
\bottomrule
\end{tabular}
~~~

### TEX-TTHN-0237 — `docs/ttm_t3m_ht_note.tex` 237–241 → 244–248, LaTeX equation line

before:

~~~
\[
\text{leaf frames } \;W_i = U^A_i\krrow U^B_i,
\qquad
\text{transfer tensors } \; B^C_t = B^A_t\kron B^B_t,
\]
~~~

after:

~~~
\[
\text{leaf frames } \;W_i = B^X_i\krrow B^Y_i,
\qquad
\text{transfer tensors } \; T^Z_t = T^X_t\kron T^Y_t,
\]
~~~

### DOC-UG-0125 — `docs/user_guide.rst` 125–133 → 125–133, in place

before:

~~~
	      1--H0--R1--R2--1   1--L0--H1--R2--1   1--L0--L1--H2--1
	         |   |   |          |   |   |          |   |   |
	v  =     U0  U1  U2    +    U0  U1  U2    +    U0  U1  U2
	         |   |   |          |   |   |          |   |   |

	      1--O0--R1--R2--1   1--L0--O1--R2--1   1--L0--L1--O2--1
	         |   |   |          |   |   |          |   |   |
	   +     V0  U1  U2    +    U0  V1  U2    +    U0  U1  V2
	         |   |   |          |   |   |          |   |   |
~~~

after:

~~~
	      1--H0--R1--R2--1   1--L0--H1--R2--1   1--L0--L1--H2--1
	         |   |   |          |   |   |          |   |   |
	v  =     D0  D1  D2    +    D0  D1  D2    +    D0  D1  D2
	         |   |   |          |   |   |          |   |   |

	      1--U0--R1--R2--1   1--L0--U1--R2--1   1--L0--L1--U2--1
	         |   |   |          |   |   |          |   |   |
	   +     V0  D1  D2    +    D0  V1  D2    +    D0  D1  V2
	         |   |   |          |   |   |          |   |   |
~~~

### DOC-UG-0139 — `docs/user_guide.rst` 139–142 → 139–142, K10b (re-laid out)

before:

~~~
  - up_tucker_cores   = (U0,...,U(d-1)), orthogonal
  - left_tt_cores     = (L0,...,L(d-1)), left-orthogonal
  - right_tt_cores    = (R0,...,R(d-1)), right-orthogonal
  - down_tt_cores     = (O0,...,O(d-1)), down-orthogonal
~~~

after:

~~~
  - down_tucker_cores = (D0,...,D(d-1)), down-orthogonal
  - up_tt_cores       = (U0,...,U(d-1)), up-orthogonal
  - left_tt_cores     = (L0,...,L(d-1)), left-orthogonal
  - right_tt_cores    = (R0,...,R(d-1)), right-orthogonal
~~~

### DOC-UG-0215 — `docs/user_guide.rst` 215–218 → 215–218, K10b (re-laid out)

before:

~~~
    frame Tucker core  U_i   (T3Frame)        C            (2,)      + (n_i, N_i)
    variation core    dU_i  (T3Variations)   K + C        (3, 2)    + (n_i, N_i)
    probe vector      w_i   (ww)             W            (4,)      + (N_i,)
    forward probe     z_i   (tangent.probe)  W + K + C     (4, 3, 2) + (N_i,)
~~~

after:

~~~
    frame Tucker core  D_i   (T3Frame)        C            (2,)      + (n_i, N_i)
    variation core    V_i   (T3Variations)   K + C        (3, 2)    + (n_i, N_i)
    probe vector      w_i   (ww)             W            (4,)      + (N_i,)
    forward probe     z_i   (tangent.probe)  W + K + C     (4, 3, 2) + (N_i,)
~~~

### DOC-W-0069 — `docs/weighting.md` 69–74 → 69–74, markdown table

before:

~~~
| family | weights | length |
|---|---|---|
| `up` | `H`'s Tucker (`nU`) leg | `d` |
| `down` | `V`'s complement (`nD`) leg | `d` |
| `left` | `H`'s left bond (`rL`) | `d` |
| `right` | `H`'s right bond (`rR`) | `d` |
~~~

after:

~~~
| family | weights | length |
|---|---|---|
| `down` | `H`'s Tucker (`nD`) leg | `d` |
| `up` | `V`'s complement (`nU`) leg | `d` |
| `left` | `H`'s left bond (`rL`) | `d` |
| `right` | `H`'s right bond (`rR`) | `d` |
~~~
