# R5 review: the docs, the six notes, the glossary and work list, CHANGELOG and release notes

Reviewer's record. Base HEAD `f4bb6357`; the working tree as of the implementer's Q1 edit (`\bar H`). Read: `CLAUDE.md`
(the tree's and HEAD's), the HANDOFF top block, plan draft 27 (§6, §4.8, §4.9, §4A, §3 with the T4S line and the shape
rule, §1.3, §5), `R5_handoff.md` (§1–§9, Appendices B–D), sample edits 3 (1, 2, 3, 7, 8, 10 and §13), batch 1 edit 8,
batch 2 edit 7. Nothing below is committed or edited by me except this file.

## Verdict in one paragraph

The notes are right. I re-derived every changed equation of the six notes by contraction sizes (below) and found no
wrong letter, no wrong width and no broken contraction; the 20 K16 forms all sign off. The pages, the glossary and the
work list apply the rows as planned. The CHANGELOG and the upgrade notes are complete against §2 (44 silent sites,
checked against a signature scan of both trees) and quote §1.3's message verbatim. The gates, the battery, both
doctest environments and `sphinx -W` pass (§6). One MINOR fix is the implementer's (M1). Two letter changes need the
session's ruling (M2 = Q3, M3 = Q5). The rest are NITs.

## MAJOR

None.

## MINOR

- **M1 — `CLAUDE.md:68–70`, the notation map reads like our letters.** After Nick's line, the clause "— `U` =
  `down_tucker_cores`, `P` = `left_tt_cores`, `Q` = `right_tt_cores`, `O` ("outer", our up-orthogonal) = `up_tt_cores`"
  writes a bare backticked `` `U` = `down_tucker_cores` ``. That is exactly the 2026.4.0 mapping an agent carries from
  memory, and `CLAUDE.md` is the file agents read first; four lines later the gotcha says the `U`/up core is
  `up_tt_cores`. **Fix:** make the owner explicit, e.g. "— T4S's `U` is `down_tucker_cores`, its `P` `left_tt_cores`,
  its `Q` `right_tt_cores`, its `O` ("outer", our up-orthogonal) `up_tt_cores`", or drop the clause (Nick's line already
  says it). This is wording only. `batching_and_stacking.md` §8 is fine as it stands: it is labelled "Paper ↔ code", and
  every entry carries "(our `D`)" and the like.
- **M2 — Q3, the session's ruling (it changes a letter).** `exact_hessian_two_hole.tex:790–791`:
  `\dot{\hat B}_i=B_{z,i}\dot B_i^\T`, with `\dot B_i` θ's Tucker direction, and then, in the same sentence, the
  zipper's `Z^L_iA_i\dot B_i` "for the chains `A=\hat G`, `B=G`", where `\dot B_i` means `\dot G_i`. One symbol has two
  meanings within one sentence, inside a formula. The rename created this: at the tag the direction was `\dot U_i`.
  The generic `A`/`B` mirror the code's `coresA`/`coresB` (`tt_operations.tt_zipper_tangent_left_to_right`).
  **Recommend** the implementer's proposal: write the chains out,
  `\dot Z^L_{i+1}=\dot Z^L_i\hat G_iG_i+Z^L_i\dot{\hat G}_iG_i+Z^L_i\hat G_i\dot G_i`, and keep the code pointer
  (e.g. "\code{tt_zipper_tangent_left_to_right} with the chains $\hat G$, $G$"). The size check holds:
  `Z^L_{i+1}=Z^L_i\hat G_iG_i` (`\hat G_i=G_{z,i}\times_2\hat B_i` is `r_z×n×r_z`, `G_i` is `r×n×r`; the mode legs
  contract `n` with `n`).
- **M3 — Q5, the session's ruling (it changes a letter).** `pad_safe_svd.tex:307`, `\ell_{j+1}=\min(\ell_j u_j,
  r_{j+1})`. Here `u_j` is the 2026.4.0 `up_i` (the Tucker store's rank) of `ranks.compute_orthogonal_representation_ranks`,
  which R4a rewrote as `left_{i+1} = min(left_i * down_i, r_{i+1})` (`ranks.py:645`). Read in the new words, `u` names
  the up store, and the recurrence would then be wrong at Tucker slack: the left unfolding's rows are the left bond
  times the Tucker width. The line is §4A's own candidate class ("a store-word index variable in a recurrence, as in
  `ranks.py:644–647`"), abbreviated to an initial. **Recommend** `n_{D,j}` (the padded-frame note's notation, the
  code's `nD`). The note has no other width symbol, so nothing collides.

## The questions Q2–Q9: my reading, with the deciding line

| Q | reading | deciding line |
|---|---|---|
| Q2 | **leave** `B^\T` | `:561` "$B^\T$ summing the gradient blocks where θ's cores sit" defines it on the spot; it has no subscript and never multiplies a `B_i`. `B_{z,i}` already sat beside it at the tag |
| Q3 | **change** (M2) | `:790–791`, as above |
| Q4 | **leave** the gauge generator `V` | `:232` "θ̂_v = Vθ, with V a linear generator of the gauge action": an operator applied to θ, without a subscript; the paragraph has no `V_i` |
| Q5 | **change** to `n_{D,j}` (M3) | `pad_safe_svd.tex:307`, as above |
| Q6 | **leave**, and record an audit item for Nick | `shared_t3_math.tex:335, 339, 575`: `\dot V_i`, `\dot W_i` are the note's own pair of unabsorbed Tucker variations of ξ, η (`N_i × n_i`, `n_i` the down width). By the shape rule they are `D`-shaped, but they are not store letters, no row lists them, and renaming them needs a second free letter for `\dot W_i`. `:316`'s `B_i` (the factor in the code's row layout) stays, as §4.8 rules |
| Q7 | **keep** `a_0`/`b_0`, the T4S line, `\delta\mathbb V` | `ttm:196`: `x_0` is `\T`'s index (`:63`). Plan §3 puts the line in "each tex note's notation paragraph". `\delta\mathbb V` names the whole variation, not a gauged coordinate (K17 targets `\delta U/D/G`). The `:93` typo (`G^Y[\alpha',a',\beta']` for `b`) is pre-existing math, for the docs pass |
| Q8 | **leave** | `padded_frame_cone.tex:51` `\author{T3Toolbox 2026.4.0}`: the release that shipped the note, not a letter |
| Q9 | **leave** | §4 item 9's symbol-table list (gains `sh`, `Ψ`, upright `d`; loses `O`; rewrites `D`, `U`, `P`) has no `R`/`F` rows; the hyphenation is a docs-pass row |

**Q1 (applied, verified).** `\bar H_i` replaces the center at its 12 tokens (`:308` ×2, `:309`, `:312`, `:313`,
`:331`, `:332`, `:373`, `:380`, `:414` ×2, `:1018`). `\bar` was free in the note. The coordinate keeps `H_i` at `:125`,
`:130`, `:135`, eq:lift `:323`, `:330`, fig:lift's red node and (a), caption (a), `:403–404`, alg:lift's Require,
`:420` and `:422`.

I read alg:lift line by line on the rendered page 6:
- line 1 builds `\bar H_i = L_i×_3T_i`: `r^L_{i−1}×n_i×r^L_i` times `r^L_i×r^R_i` gives `r^L_{i−1}×n_i×r^R_i`;
- `S_i^T = ⟨U_i, \bar H_i⟩_{1,3}` is `n^U_i×n_i`;
- line 4 gives `(n_i×n^U_i)(n^U_i×N_i)` → `B_i`'s shape;
- line 5 lifts the Require's coordinate `H_i` with `T_i^{+}` (`r^R_i×r^L_i`) → `L_i`'s shape;
- line 7's adjoint returns `V_i`, `H_i`.

Figure 2 (b) reads "= \bar H_i (the center)". The proof's Tucker slot `L_{<i}(\bar H_i×_2\dot B_i)R_{>i} =
L_{<i}(U_i×_2(S_i^T\dot B_i))R_{>i}` holds by the rows convention.

## NIT

- **N1** `aligned_padded_frame.md:138–139`, "(the oversampled and the truncating `('D', 'U', 'L', 'R')`, `('D', 'L')`,
  `('U', 'R')`)". This lost the tag's precision: at the tag the oversampled program was `'ULRO'` only. Write it as
  `padded_frame_internals.md:120–121` does: "the oversampled `('D', 'U', 'L', 'R')`, the truncating `('D', 'U', 'L',
  'R')`, `('D', 'L')` and `('U', 'R')`".
- **N2** `exact_hessian_two_hole.pdf`: the 2.72pt overfull at eq:conv (`:587`, from `((\Sh^\T)^r z)`) and the `h`→`ht`
  float on alg:zero are new against HEAD's build (0 warnings). Both are harmless; fix only if you are in the file.
- **N3** `numerical_contracts.md:105–107`: the two doctest comments sit at columns 89/88. The off-by-one predates the
  arc; `D.`→`diff.` shifted both by 3. For R6's alignment sweep; outputs are unchanged.
- **N4** The handoff §7 lists "Not run: … the examples". R5 changes no file under `t3toolbox/`, `tests/` or
  `examples/` (diff empty; K15 0 hunks), and CI's `tests` workflow, examples job included, is green on HEAD
  `f4bb6357`. Say so in §9 rather than "not run", so the plan's R5 gate row reads as covered.

## What I checked, and how

**The notes, re-derived (sizes on both sides of every contraction; code shapes `D (nD,N)`, `U (rL,nU,rR)`,
`L (rL,nD,rL')`, `R (rR,nD,rR')`, `V (nU,N)`, `H (rL,nD,rR)`).**

- `exact_hessian_two_hole.tex`, all changed lines:
  - The format: `B_i`, `X = TT(G)×_1B_1⋯`.
  - The frame: `D_i n_i×N_i` orthonormal rows; `L_i`, `R_i` with middle `n_i`; `U_i` `r^L×n^U_i×r^R`; `n^U_i ≤ n_i`,
    whose strict form is Tucker slack (the up-matricization's rank). The base point `θ=(D,L)`.
  - `v=(V_i,H_i)`, `V_i n^U_i×N_i`. eq:E: `×_jD_j` pairs `n_j` with `n_j`; `×_iV_i` pairs `n^U_i` with `n^U_i`.
  - The gauges: `V_iD_i^T` over `N_i`; `(L^{⟨2⟩})^TH^{⟨2⟩}` over `r^L_{i−1}n_i`.
  - eq:factors (#9), eq:lift, the proof, fig:lift, the transpose `((S^T)^{+ε})^TY^B` → `n^U×N` and
    `Y^G×_3(T^{+ε})^T` → `H`'s shape.
  - The corewise frame `(B,G,G,G)`, `U,L,R → G`, `U=L=R=G`, `(D,L,L,L)`.
  - eq:asm: `ξ=B w`, `δη̃=B c`. The shift: `sh^r` gives `r!C(t,r)`, `(shᵀ)^r` gives `r!C(r+s,r)`, adjoint in the plain
    inner product.
  - eq:t3t3: `\hat B=B_zB^T` (`n_z×N·N×n`); `G_z×_2\hat B`; the Tucker block `⟨env,G⟩^TB_z` → `n×N`.
  - The tied lift (#10: `M_g` `(Σn^U)×n_g`, `M_g^{+ε}[V]` → `n_g×N_g`), alg:zero, `\mathcal R` at the three sites
    plus C8, `\mathrm d` everywhere, PB20's `R_k^{-1}` marked.
  - `frame_data[5]`'s "down and left masks" checked against `uniform_fitting._frame_theta` (`[5][0]`, `[5][2]`).
- `symmetric_probe_derivatives.tex`:
  - §1–4 `\mathbf B_i` (`n_i×N_i`; `ξ = Bw` → `n`, `z = B^Tη` → `N`).
  - The frame `((D),(U),(L),(R))`. eq:rbase: `ξ=Dw` → `n_D`; `L(ξ)` and `R(ξ)` contract `n_D`; `η=μ^TUν` → `n_U`.
  - eq:rdxi `Vw` → `n_U`. eq:rsigma: `R(ξ)` and `H(ξ)` over `n_D`, `U(δξ)` over `n_U`, all → `r^R_i`; it is the
    left-to-right "variation already seen" recursion. eq:rtau mirrors it, right to left. eq:rdeta: the three
    placements, each leaving `n_D`.
  - eq:rz: `D^Tδη` and `V^Tη` → `N`. The jets `R(Dp)`, `H(Dp)`, `U(Vp)`. `\widehat{δη}=D r`.
  - The transposes `\widetilde V`, `\widetilde H`. "the edge down to its Tucker core" / "the edge up to its TT core"
    agree with the picture (chain on top).
  - The corewise sentence is the plan's template.
- `shared_t3_math.tex`:
  - The `B_i`/`B_g` point sections. `K_i=B_iW_i` (`N×rr`) with `Q^TK_i=R_{[i]}W_i`; `:316`'s `B_i` stays.
  - The notation sentence. The tangent section `D_i^T\dot V_i=0`.
  - eq:tied, eq:gramsolve, `M_g`, `O_gZ`, `ZS_i`, the Gram `tr(Z(ΣΓ)Z^T)`, `[D_g|\dot D_g]`/`[D_g|\dot D]` (as at the
    tag).
  - K16 #11–#14, cross-checked against `sharing.py:510, 561` (`(sum nU, n_g)`) and `fv_center_factors`' `S_i^T
    C+(nUi, nDi)`.
- `ttm_t3m_ht_note.tex`:
  - `B_i[n_i,x_i]` against `\C`'s mode `i`. Khatri–Rao `B^X⊙B^Y` → `(n^Xn^Y)×N`. The Kronecker
    `r^Xr^Y×n^Xn^Y×r^X'r^Y'`.
  - The HT `T^Z_t=T^X_t⊗T^Y_t`; the macros `\X\Y\Z`, with `\C` kept for eq:t3.
- `padded_frame_cone.tex`:
  - Table 1 (#1–#4): `ℓ_i→c_i` is down, `D_i n_D×N` (the leaf side's frame); `c_i→ℓ_i` is up, `U_i r_L×n_U×r_R` (the
    middle index); `L`, `R` have middle `n_D`.
  - Algorithm 1 against the code: `_up_message` is "core -> leaf", `_down_message` "leaf -> core", `_down_edge` is
    leaf→core (one per sharing group), `_up_edge` core→leaf.
  - The colours equal `_COLOURS`. The seed tags equal `_SEED_TAGS = {'down': 0, 'up': 1, …}`. The root sentence is §4
    item 2's text. `Ψ`, `F`, `F′`.
- `pad_safe_svd.tex`: "the TT up-orthogonalization" ×2 (Q5 apart). It cites no T4S, so it needs no T4S line.
- **The T4S line**: Nick's words verbatim in all five notes that use frame letters, `frame_variations.md`, the glossary,
  `CLAUDE.md` and `entries_apply_probe.md` §8.
- **PDFs**: I rebuilt all six myself with `pdflatex` ×3, sequentially, in scratch. They compile with no error and no
  undefined reference. The text of the four committed PDFs equals my builds (`pdftotext` diff empty), so the committed
  PDFs are the current sources, `\bar H` included. Rendered pages read:
  - exact-Hessian pp. 1, 2, 4, 5, 6, 8;
  - shared pp. 1, 5;
  - padded-frame pp. 2, 6;
  - symmetric p. 6;
  - ttm p. 1.

**The pages.**
- `frame_variations.md` § Four rank stores is sample edit 1's approved text. It takes the plan's final T4S line in place
  of the draft's `δD`, and J1 adds one sentence ("Before 2026.5.0 the words up and down … named the other store"),
  which `CLAUDE.md`'s pointer needs.
- The tangent diagram passes K10 in place.
- `user_guide.rst:125–133` (raw text, TABs kept) and `:139–142` (reordered; `down-orthogonal`/`up-orthogonal`, which
  fixes the backwards adjective at `:142`) are byte for byte sample edit 8's AFTER.
- `CLAUDE.md`: the gotcha `(D, U, L, R)` with "the `U`/up core is called `up_tt_cores`"; the letter list
  `(D U L R B G V H …)`; the history list; the width clause; `(U, L, R) → G`. M1 apart, all as planned.
- `naming_conventions.md` has §6's bullet verbatim.
- Every other page's word diff was read. Its letters follow the shape rule (`V, H` in `chunking.md`; `Ḋ` for the gauged
  tied direction, `Ḃ` for the lift's raw direction and the corewise copies; `diff` in `numerical_contracts.md`).

**The glossary and work list.**
- The frame cores entry is sample edit 2's AFTER verbatim. Orthogonal core: the Clash sentence is resolved, and
  "outer-orthogonal" is carved by entry.
- Store, padded widths, two-colour pass, corewise frame and the Symbols table (rows `D`, `U`, `Ψ`, `d`, `sh`; `P`
  rewritten; `O` gone) are as planned. Each "(2026.4.0 …)" group sits on one physical line.
- The T4S entry has Nick's line.
- Work list: N4 and N17 are resolved, and no HELD row remains.
- I spot-checked the current column against the code. `tv_operations.py:164, 181` "up/outer", `ufv_operations.py:277,
  294`, `frame_variations_format.py:356`, `padded_frames.py:167` "(down)" and `tucker_tensor_train.py:615` "Up TT
  core" all match their rows.
- K1/K2/K3 run over the whole file: K2 and K3 have 0 hits. K1 hits only the N17 evidence cell, "At 2026.4.0: …", a
  dated record.

**CHANGELOG and release notes.**
- The silent list has 18 attributes + 4 rank keywords + 16 width keywords + 6 properties = 44.
- An `inspect.signature` scan of every public function and method on the tag worktree and on the tree finds exactly the
  listed loud renames: the four fields and the 14 `compute_*` functions' 18 keywords. Every other change is a
  same-name position swap in the silent list.
- "need not raise" appears, with the numpy note and no shape-check claim.
- §1.3's message: I generated it from `_update_stores('ULRO')`. It is identical to the plan's §1.3 text and appears
  verbatim once in each of `CHANGELOG.md`, `docs/release_notes.md` and `release_notes_2026.5.0.md`.
- The table's `update` row reads "yes for a `str`", with the `'O'` sentence under the error.
- The SWAP recipe sorts longest first and includes the `_dot` forms and `nU`↔`nD`, without `rL`/`rR`. I ran it on
  `variations_up_mask`, `up_tucker_cores_dot`, `nU=7, nD=5`: all correct.
- What does not change; the letters; T4S; `sh`; the shape-rule renames. There is no "No store letters" bullet.
- The root file equals the CHANGELOG section, apart from blank lines.

**Tool changes (§8).**
1. `_t4s_normal`: a justified narrowing. The carve-out still works by block. The only blocks that hold the line are
   the T4S paragraphs and `CLAUDE.md`'s notation-map paragraph, which has no other old letter.
2. List items as blocks: a tightening.
3. The `outer-orthogonal` carve by entry and pattern: narrower than the line.
4. K9-docs: a new check (a tightening). Its PATTERN rows are the three R1 rows; the 32 new keeps are line-scoped
   (English idioms and glossary adjectives, each read).
5. The K8 carves: each one is a do-not-touch letter (completions, `P_sf`, the rotation `Q_i`, QR's `Q_{ij}`,
   `P_real`, the projections `Q_c` and `P_k`). `O_g` by text: the shared note's only `O_g` is the complement basis.
6. `K8_TEX_DECOR_R5`: justified, but it opens a hole: a missed T3-own `\mathbf U_i` (which should be `\mathbf B_i`)
   would now pass. I closed it by reading. Every `\mathbf U` left in the symmetric note is in the tangent section
   (`:373–433`) or in the T4S line, and every `\mathbf P` is the direction.

## Runs (sequential, one heavy job at a time)

| run | result |
|---|---|
| `store_rename_check.py --stage R5 --hits dev/store_rename_slices --prev HEAD` | every K gate PASS (K0–K9, K9-docs ×3, K10/K10b 37+23, K11, **K12 301/301**, K13, K14, K15 0 hunks, K17); K16 lists 20 forms, all signed off above |
| docs doctests (`docs/*.md`, `docs/contributor/*.md` minus `doctest_style.md`, `getting_started.rst`) | pass in `t3toolbox` (py3.11) and `tttt` (py3.9, numpy 1.22.4, jax 0.4.30) |
| battery `run` + `compare` against `battery_ref` | **PASS (bitwise)**, 2,190 arrays, 856 texts |
| six notes, `pdflatex` ×3 each | compile; text equal to the committed PDFs |
| `sphinx -W --keep-going -b html` | **build succeeded**, no warning (full build, run alone) |

Pass 1 verdict: REVISE (M1 for the implementer; M2/M3 await the session's ruling; NITs optional).

## Pass 2 (after the revision)

**The session's rulings.** Q3 YES, Q5 YES, M1 YES. Q2, Q4, Q7, Q8 and Q9 stay as they are. Q6 is recorded as an audit
item: the shared note's `\dot V_i`, `\dot W_i` are `D`-shaped, but they are not store letters.

**Who applied what.**
- The implementer applied Q3, Q5, N1 and the handoff records (N3, N4, §7).
- The session applied M1 itself. Neither agent may edit `CLAUDE.md` on another agent's request.

Verified:

- **M2 / Q3**: `exact_hessian_two_hole.tex:792–793` reads `\dot Z^{L}_{i+1}=\dot Z^L_i\hat G_iG_i+Z^L_i\dot{\hat
  G}_iG_i+Z^L_i\hat G_i\dot G_i` "(with the chains $\hat G$, $G$; \code{tt_zipper_tangent_left_to_right}, step
  \code{_tt_zipper_tangent_step})". Each term is the product rule of `Z^L_{i+1}=Z^L_i\hat G_iG_i`, and the mode legs
  contract `n` with `n`. Every remaining `\dot B` in the note is θ's Tucker direction or the tied lift's `\dot B_g`
  (23 sites, each read).
- **M3 / Q5**: `pad_safe_svd.tex:307` reads `\min(\ell_j n_{D,j}, r_{j+1})`, which is `ranks.py:645`'s
  `left_{i+1} = min(left_i * down_i, r_{i+1})`. K16 now lists 21 forms; the new one is #15, and I sign it off.
- **M1**: `CLAUDE.md:68–72` reads "— T4S's `U` is `down_tucker_cores`, its `P` `left_tt_cores`, its `Q`
  `right_tt_cores`, its `O` ("outer", our up-orthogonal) `up_tt_cores`". The Widths sentence is rewrapped, and no line
  exceeds 120 characters.
- **N1**: `aligned_padded_frame.md:138–140` names each program again.
- **N2**: left, as allowed. **N3**: recorded for R6. **N4**: §9 states that R5 changes no code and that CI run
  43493578933 on `f4bb6357` is green.
- **PDFs**: the committed exact-Hessian and `pad_safe_svd` PDFs show the new text (`pdftotext`), and no build litter
  remains.
- **`sphinx -W`**: a second pass after the revision re-read all 1,391 sources and succeeded with no warning.
- **K gates at R5** (after the M1 rewrap): all PASS except K9-docs. K9-docs failed on `CLAUDE.md:70–71` because the
  session's rewrap made the clause-(c) registration (`k9_registered.tsv:78`) stale. The text was correct; the
  implementer replaced row 78 with rows 78–79, one per new line. My re-run passes: every K gate, K9-docs and
  K9-docs-registered included; K12 301/301; K15 0 hunks; K16 21 forms, all signed off.

Agreed. The notes, pages, glossary, work list, CHANGELOG and release notes are ready for the R5 review stop with Nick.
For that stop: the Q6 audit item, and the code's `H` for companion centers (the session's pre-existing audit item).

VERDICT: APPROVE
