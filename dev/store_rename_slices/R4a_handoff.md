# R4a handoff: code prose outside the sampling kernels (store rename, 2026.5.0)

Implementer's record for the reviewer and the session. Base: HEAD `84f3a755` (plan draft 26; R3 = `74683a39`). Nothing
is committed. Scope (plan §6 R4a, the brief): every tier-D/E row and every §4A block in `t3toolbox/`, `tests/`,
`examples/`, excluding R4b's files (`probing.py`, `sampling_derivatives.py`, `apply.py`, `entries.py`,
`utv_sampling.py`, `ut3_sampling.py`, `tests/_jet_trs_twins.py`, `tests/test_jet_recurrence.py`,
`tests/backend/_curvature_oracles.py`, `test_curvature.py`, `test_uniform_curvature.py`), plus the realignment of
everything the rename touched (plan §4 item 10, §7 "within reason").

**Result:** 54 files (30 library, 23 tests and helpers, 1 example), 860+/851−, all text except the §4.11 loop variables
(40 NAME hunks after revision 1, listed). In R4a's files K1, K2, K4, K5, K6, K7, K8, K9, K13 report **0** (K9 with the clause (c)/(d)
registrations, §6); K10/K10b/K11/K12 **PASS** for R4a's 47 blocks; K14, K15, K15-cross, K17 **PASS**; battery
**bitwise**; suite, doctests, `sphinx -W` and the floor env: §7.

New files: this handoff; `dev/store_rename_slices/k9_registered.tsv` (K9's clauses (c)/(d), 48 lines);
`R4a_name_hunks.tsv` (K15's listed hunks, 40); `R4a_realignment.diff` (the realignment pass, 87 hunks, whitespace
only). Changed tools: `dev/store_rename_check.py`, `dev/store_rename_diagram_check.py` (§6, for the reviewer's eye);
`diagrams_inventory.md` (R4a's 47 entries filled).

## 0. Revision 1 (the session's rulings on `R4a_review.md`, 2026-11-13)

The review (REVISE: MINOR 1, MINOR 2; S1–S5) was ruled by the session; each ruling is now a row, and R4a applied it:
`dev/store_rename_letter_sites.md` gains the section "The R4a review rows" (RV1–RV15) and the SITES rows
`LS0315`–`LS0338` (the 24 NAME tokens, slice `R4a`); `R1_hit_table.tsv` gains the same 24 as tier `R4a-row` (tag
positions), so K15-cross reads them as map rows.

| ruling | applied |
|---|---|
| MINOR 1 | `utv_operations.py:449` "left-core ``L``", `:464` "the L-parallel component", `:499` "the right core ``R``"; `:465, 479, 487, 517, 557` `(L^L)^T H^L` (with S4's `H`, one spelling per object in the module) |
| MINOR 2 | `t3_linalg.py:418, 419, 493` `[G, B, mode]`, `B=stack+(nL, NL)`, `B=stack+(nR, NR)` |
| S1 | the nine `(B, G)` (§5 item 2) stand, now rows RV8, RV9 |
| S2 | `frame_variations_format.py:856–857, 941–942`: the comprehension `B` → `V` over `tucker_variations`, `G` → `H` over `tt_variations` (8 tokens) |
| S3 | `tv_operations.py:774, 779` ``D V^T``, ``max|D V^T|`` (``L^T H`` kept) |
| S4 | `udots` → `ddots` (`tv_operations.py:245, 250`, `utv_operations.py:224, 241`). utv `dG` by shape: `utv_to_ut3.last_core`'s `dG` → `H` (`:294, 298, 299`; the comments there say `(rL, nD, rR)` and `[R, H] -> (rR+rL, nD, rR)`); `_utv_oblique_gauge_step`'s `dG_in`/`dG` → `H_in`/`H` and `utv_oblique_gauge_projection`'s `dG_stack` → `H_stack` (`:485–489, 523–524`; the shape comment at `:480` gives the variation as `K+C+(rL,n,rR)` beside the frame's left core `C+(rL,n,rL)`, and the step runs only at a manifold frame, where `einsum('...iaj,...iak', L, H)` sums `L`'s middle leg, `nD`); the comments at `:480, 485, 519` follow. Kept: `utv_corewise_retract`'s `dB, dG` (`:396`, `(d,) + K + C + (n, N), (d,) + K + C + (rL, n, rR)`): the corewise derivative at the point's own ranks |
| S5 | `t3_orthogonalization.py:134` "Up-orthogonalize the TT cores, pushing the remainders down onto the Tucker cores below." |

The NAME hunks go from 18 to 40 (Appendix B; 22 new: S2's 8 tokens on 4 lines, `udots` 4, `dG` family 14 tokens on
12 lines). K14 needed one more narrowing (§6): `B` (file-scoped) and `G` (global) are DO-NOT-TOUCH item-5 tokens ("a
T3's own Tucker cores", built by R1 from every `B`/`G`), and S2's eight occurrences are variations; a session row in the
hit table now licenses exactly its own occurrences. Every edit is comment, docstring or a local NAME; no expected
doctest output changes (2,585 examples), and the edited modules compile.

## 1. What changed, by file

Library (`t3toolbox/`):

- `frame_variations_format.py`: the module convention sentence (line 8: `T3Frame.data = (down, up, left, right) =
  (D, U, L, R)`, "each store named for the side of its edges that its cores occupy"); 25 §4A blocks (Appendix A);
  `validate` messages (`len(DD)=`, `len(UU)=`, `up_tt`, `: D.shape[-2]=`, "Up TT core left/right rank mismatch",
  `: U.shape[-3]=`, `: U.shape[-1]=`), "down/up slack", `check_fv_pair` (D, U, L, R), the `T3FrameWeights` field words,
  `# (down, up, left, right)`, `down = up = tucker_weights`, `fv_absorb_weights` (`up`→V, `down`/left/right→H), the
  FVF-1302 companion's prose (Tucker (down) cores; `# down cores untouched`; "swaps ``V1`` into the down cores and the
  up core ``U1`` into the chain").
- `uniform_frame_variations_format.py`: UFVF-0110 (field comments), 1106, 1113, 1122; ~15 prose rows (padded
  down/up rank comments, store listings, `check_ufv_pair`, `down == up`, `||down·left…||`, weights).
- `manifold.py`: `<V_i,V_i'>+<H_i,H_i'>`; M-0685; store listings; "upward ``ξ̂``" and "the frame's upward edge
  variables" (§3.3 rows, see §5 item 3); `(D, U, L, R)`; "left/right/up-orthogonal"; the CorewiseGeometry text:
  `(B, G, G, G)`, "up/left/right cores", ``(U, L, R) -> G``, `(B, G)`, its two messages.
- `uniform_manifold.py`: UNMA-0478 (re-derived, §3.2), the corewise section ("the three TT-shaped slots",
  "down = up = tucker_edge_mask (the B-G Tucker edges)").
- `fitting.py`, `safety.py`, `shared_geometry.py` (`(D, L)`, `mean_i(B_i + V_i)`), `uniform_tucker_tensor_train.py`
  (``(U,L,R)->G`` ×2), `tucker_tensor_train.py` (up TT core unfoldings, up/down orthogonal, ``U,L,R -> G``, the
  §4.11 typo `rL_0*r_1`→`r_0*r_1`).
- Backend: `fv_conversions.py`, `fv_operations.py` (FO-0201; `fv_absorb_weights`' loop variables → `w`, §4.11),
  `geometry.py` (L_last, `(D, L)`, "theta_hat, the lift of v", the corewise `(B, G)` / `(B, G, G, G, shape, masks)`),
  `ranks.py` (RA-0644), `regularization.py`, `optimizers.py` (`# initial cores (B, G)`), `backend/fitting.py`,
  `padded_frames.py` (PF-0041/0043, the module docstring, the message signature labels PhiU/XiU/bU/U/PhiD/XiD/XiDs/
  D+/o_D/PsiD/Psi_L/Psi_R/U_P/U+_i, store comments, step comments), `uniform_padded_frames.py` (UPF-0047/0050, the
  `('down mask', mD)`/`('up mask', mU)` labels, the §4.2 step comments "the down store's edges (l_i -> c_i)" and "the
  up store's edges (c_i -> l_i)"), `tv_operations.py` (TVOP-0108; `tv_to_t3`'s `dU`→`H` (8 NAME sites, the V5 row);
  `H_i = <D_i, core_env_i>` / `V_i = <U_i, core_env_i>`; the `tv_corewise_lift` docstring), `utv_operations.py`
  (UTVOPE-0314/0538; `Ddot`; the lift comments), `ufv_conversions.py`, `ufv_masking.py`, `ufv_operations.py`
  (UFOP-0083), `sharing.py` (`Ddot`, `Bdot_g`, `U_i`, `U2_i`, `dB_i`, `dPhi`), `t3_linalg.py`, `ut3_linalg.py`,
  `t3_orthogonalization.py` (T3OR-0044 unchanged; "Down-orthogonalize the Tucker cores, pushing the remainders up onto
  the TT cores above"; the `(Bio, Haib)` / `Uaxb` / `new_Bxo` labels), `t3_svd.py` (Tucker down-SVD),
  `ut3_orthogonalization.py`, `uniform_fitting.py` (`(down, left)` ×9).

Tests and example: `test_padded_frame.py` (`stores_of`'s `u`/`o` → `D`/`U`, §4.11; "in-block up", "an up width";
the d = 1 note rewritten in the new convention, 4 lines), `_uniform_padded_frame_testing.py` (`Nsh`'s `u` →
`Di`, §4.11), `test_uniform_padded_frame.py`, `test_padded_start.py` (the shared Tucker variation `V`, plan §1.2a;
`D_i^T`; "up-frame"), `_padded_frame_oracles.py` (the `(D, phi, psi, U)` tuples, docstring mentions of the R2-renamed
`dense`/`old_dense`/`cur_dense`), `test_zero_reference.py` (`(D, L)`, `(D, R)`), `test_geometry.py`,
`test_frame_variations_format.py` (T-TFVF-0025; the corruption list; the rank-table header),
`test_uniform_frame_variations_format.py`, `test_uniform_manifold.py` (`[down | up]`, `[R-block | L-block]`, `(R0=0, Ri=…; Ld=0, Li=…)`, `[R | L]`, `(B, G, G, G)`
×3), `test_uniform_weighted.py`, `test_weighted.py`, `test_uniform_probing.py`, `test_fitting.py`,
`test_dispatch.py`, `test_contractions_interpreter.py` (sample batch 2 edit 9, "the up store's (c30, c31) edges",
"adjoint-upward", `Bdot`), `test_contractions_sharding.py`, `test_manifold.py`, `test_sharing.py`,
`test_tucker_tensor_train.py`, `backend/test_optimizers.py`, `backend/test_uniform_fitting.py`,
`backend/test_from_sweep_reuse.py`; `examples/fit_hilbert_from_entries_lbfgs.py` (`(B, G, G, G)`).

The locator skipped every LETTER-API row (S1). No sampling-kernel file, no doc, no `CLAUDE.md` is touched.

## 2. How the rows were worked

Per file, a worklist joined the tag's text with every inventory row on it (inv 1, inv 2 §3/§4/§9, the pqu text rows,
letter_sites, keeps, the K-check hits), and each row was edited by hand with an exact-replace helper (all-or-nothing per
spec). Every sentence was written from the new convention (down = the Tucker factors, below each Tucker edge; up = the
TT-shaped cores above it; the shape rule for V/H/D/B), not swapped word by word; convention sentences, backwards
adjectives, the corewise sentences (Nick's D-6 template: "the manifold operation needs an orthogonal frame, the
corewise one works for any `B`, `G`") and the §4.2 store-and-arrow sentences were rewritten whole. After the hand pass,
a residual-letter scan over R4a's files (every store letter, decorated letter, store word and `-up`/`-down` token in
comments and strings) was read line by line; what it found is in §5.

## 3. The §4A blocks

### 3.1 Status

47 inventoried blocks are R4a's: 41 fenced blocks rewritten (28 in place, K10; 13 re-laid out, K10b), the unfenced
FVF-1302 companion doctest (R2 renamed its NAMEs; R4a its prose), and 5 unchanged with a same-reason (T3OR-0044 and
TTT-2765/2837/2908/2979: a T3's own `B`, `G`, D-6). Every entry in `diagrams_inventory.md` carries its new span, its
verbatim new text, `re-laid out: y|n (R4a: …)` and, where unchanged, the same-reason. Before/after of every block: Appendix A.

Re-laid out (K10b): FO-0201, PF-0041, RA-0644, UFOP-0083, UPF-0047, UTVOPE-0314, UTVOPE-0538, FVF-0071, FVF-0117,
FVF-0354, FVF-1371, UFVF-1122, UNMA-0478. In place (K10): the rest, including sample batch 1 edits 5 (FVF-0088,
FVF-0100) and 1 (FVF-0156, the field comments, verbatim) and the six side-by-side variation diagrams (FVF-0058a/b/c,
0063a/b/c).
Sample batch 1 edit 6 is FVF-0117/0123 verbatim, with the colon after `right-orthogonal` added (plan §4A d, Nick).

Two old typos were fixed inside re-laid-out blocks, where the fix fits the width (§4A: a typo is left unless it fits):
`(U0,U1, U2, U3)` → `(D0, D1, D2, D3)` (FVF-1371, UFVF-1122) and UFVF-1122's stray space before `=`.

### 3.2 Derivations, index by index (code layout: `D_i` is `(nD_i, N_i)`, `V_i` is `(nU_i, N_i)`)

- **FVF-0117** — `D_ia D_ja = delta_ij`: the down core `(nD_i, N_i)` summed over `a`, the leaf index `N_i` below the
  edge → down-orthogonal. `U_aib U_ajb = delta_ij`: the up core `(rL_i, nU_i, rR_(i+1))` summed over both bonds `a`, `b`
  above the edge → up-orthogonal (orthonormal over the middle leg). `L_abi L_abj`: the left core
  `(rL_i, nD_i, rL_(i+1))` summed over the left bond and the middle leg → left-orthogonal (unchanged). `R_ibc R_jbc`:
  the right core `(rR_i, nD_i, rR_(i+1))` summed over the middle leg and the right bond → right-orthogonal (unchanged,
  colon added).
- **FVF-0123** — `D_ia V_ja = 0`: `D_i` and `V_i` share the leaf index `a = N_i`; the result is `(nD_i, nU_i)`: Tucker
  variations ⟂ the down core. `L_abi H_abj = 0`: unchanged, `(rL_(i+1), rR_(i+1))`.
- **FVF-0156** (field comments, sample edit 1) — `D_xo D_yo = I_xy`, `Di (nDi, Ni)`: sum over `o = N_i` → rows
  orthonormal. `U_ixj U_iyj = I_xy`, `Ui (rLi, nUi, rR(i+1))`: sum over both bonds → up-orthogonal. `L_iax L_iay`,
  `Li (rLi, nDi, rL(i+1))`: sum over left bond and middle leg, free right bond → left-orthogonal. `R_xaj R_yaj`,
  `Ri (rRi, nDi, rR(i+1))`: sum over middle leg and right bond, free left bond → right-orthogonal. (Old letters
  `B, R, P, Q` were the tag's per-field labels: up_tucker, down_tt, left, right.)
- **UFVF-0110** — the same four with the leading supercore index `d`: `D_dxo D_dyo`, `U_dixj U_diyj`, `L_diax L_diay`,
  `R_dxaj R_dyaj`, shapes `(d,)+stack_shape+(nD, N)`, `(rL, nU, rR)`, `(rL, nD, rL)`, `(rR, nD, rR)`.
- **FVF-0354** — `einsum('...io,...jo->...ij', D, D) = I`: `D (nD, N)`, sum over `o = N` → `(nD, nD)`.
  `einsum('...iaj,...ibj->...ab', U, U) = I`: `U (rL, nU, rR)`, sum over `i, j` (bonds) → `(nU, nU)`. Left/right
  unchanged. The ``einsum`` column stays at 40.
- **FO-0201** — "down_tucker `D_i` (all i), outer/up `U_i` (all i)": the store adjectives of the two all-`i`
  relations above (T4S's "outer" kept beside "up", as FVF-0354's "up/outer").
- **TVOP-0108** — `D_i V_i^T = 0`: `(nD_i, N_i)·(N_i, nU_i)` = `(nD_i, nU_i)`; the einsum half unchanged.
- **M-0685** — `einsum('...ia,...ja->...ij', D_i, V_i) = 0`: `i` over `nD`, `j` over `nU`, `a = N` summed;
  "(Tucker variations ⟂ D)".
- **UNMA-0478** — `D_i V_i^T = 0` and `(L_i^L)^T H_i^L = 0`: `L_i^L` is the left unfolding `(rL_i nD_i, rL_(i+1))`,
  `H_i^L` is `(rL_i nD_i, rR_(i+1))`, the product `(rL_(i+1), rR_(i+1))`. The tag wrote `U_i^T dU_i` (math layout,
  `U` as `(N, n)`); rewritten in the code layout, as TVOP-0108 and the code (§5 item 1).
- **UTVOPE-0538** — `D V^T = 0`: the code's gram `g_tk = einsum('d...ia,d...ja->d...ij', down_sc, tkv)` is
  `(nD, nU)`; `(L^L)^T H^L = 0`: `g_tt = einsum('d...abi,d...abj->d...ij', left_sc, ttv)`, `(rL', rR')`.
- **UTVOPE-0314** — the doubled Tucker axis is `[D | V]`: the frame's down core (width `nD`), then the Tucker
  variation, whose rows are the up store's (`nU`) → masks `[down | up]`. The TT bond is `[R | L]` in the code's order,
  with T4S's `[Q | P]` kept as the citation (§1.5 item 3).
- **RA-0644** — the minimal-rank recursion, step by step against `fv_conversions.py:207–215`/`ranks.py`: 1.
  `down_i = min(n_i, N_i)` (the Tucker down-orthogonalization); 2. `left_{i+1} = min(left_i * down_i, r_{i+1})`; 3.
  `right_i = min(left_i, down_i * right_{i+1})`; 4. "the up cores": `up_i = min(down_i, left_i * right_{i+1})`.
- **UFOP-0083** — `nD = down.shape[-2]` (`(…, nD, N)`), `nU = up.shape[-2]` (`(…, rL, nU, rR)`), `rL =
  left.shape[-1]`, `rR = right.shape[-1]`, `N = down.shape[-1]`; the code below it reads exactly these.
- Listings by the shape rule (PF-0041/0043, UPF-0047/0050, FVF-0071/0077/0784/1371, UFVF-1122, T-TFVF-0025): down
  `(nDi, Ni)`, up `(rLi, nUi, rR(i+1))`, left `(rLi, nDi, rL(i+1))`, right `(rRi, nDi, rR(i+1))`; `V (nUi, Ni)`,
  `H (rLi, nDi, rR(i+1))`.
- Diagrams in place (FVF-0058a–0063c, 0088, 0100, 0217, 0222, 0845, 0850, 1264, 1271, 1355, 1362, UFVF-1106, 1113):
  labels only (`U`→`D` on Tucker legs, `D`/`O`→`U` on TT-shaped cores, `nU`↔`nD`); K10 confirms every other column is
  untouched.

## 4. Realignment (plan §4 item 10; "within reason", Nick 2026-11-12)

**Rule** (Option A, per run of aligned lines): an alignment column of the tag (a field start that two or more
consecutive lines share, at least one after two or more spaces, or a shared `#` comment start) is restored at
`T = max(the tag's column, the longest member's field end + min(2, the tag's smallest gap))`. So a block whose names
did not grow keeps its exact tag column; one whose longest name grew moves right only as far as needed; a block whose
columns stayed consistent but past a tag column that no longer fits is left alone. Only code lines, comments and
doctest source lines; never docstring prose, doctest output, or a fenced block. Signature blocks, field comments,
`=` columns, aligned trailing comments, `.reshape(` columns, store-pair code lines and doctest examples are all in
scope; style reference `tucker_tensor_train.py` (one column per aligned group).

**Done by a tool, read hunk by hand:** 87 hunks, 179 lines, 20 files (`R4a_realignment.diff`). Every changed line
has the same tokens (`split()` equal), the same indentation, and no new line over 120. The tool was refined until no
hunk needed a correction (rejected behaviour on the way: a single space before a code token counted as an alignment
gap; a run bridging over a closing-bracket line into the return block; a swapped `(down, up)` pair mapped through its
token instead of its column; prose words in a comment counted as fields).

**Hand decisions:**

1. `tests/test_frame_variations_format.py:126–134` (the corruption list): all nine lambdas at one column (41), the
   longest labels with one space. The tag's list was itself ragged after its fifth line; with the renamed labels
   changing length, one column is the readable choice.
2. `tests/test_frame_variations_format.py:69` (the `frame_structures` header): `down_ranks,` and `up_ranks,` placed over
   their data columns (the data order did not change; the header's words did).
3. `manifold.py:174–175` (`sharing` / `update` field comments): one column at the `update` line's code end + 2, as the
   tool did for the same pair in `padded_frames.py:573–574`. R3's longer `update` default no longer fits the tag's
   column; `uniform_manifold.py:151–152` and `uniform_padded_frames.py:621–622` still fit and are unchanged.
4. Error-message tables: `UT3Frame`'s and `UT3Variations`' `shapes_string` labels
   (`uniform_frame_variations_format.py:498–506, 799–800`) are padded inside the string to one `=` column again, as at
   the tag. This changes the message's whitespace only; no test or doctest compares it (checked).
5. Left as is, consistent past an unreachable tag column: `uniform_padded_frames.py:595–596` (R3's `PsiL`/`PsiR`
   lines keep their 3-space comment gap).
6. Two signature comments in `uniform_padded_frames.py` (`_u_left_edge`/`_u_right_edge`'s `x`, lines 407, 452) are
   121 characters once their labels follow the code's names (`DU`→`PsiD`, D-5, +2). The shape table below is aligned
   to the first line's entries, so they are left one over (the file already has 30 lines over 120). The `carry` lines
   above them were split onto a continuation line.

## 5. Judgement calls

1. **Gauge relations in the code layout.** UNMA-0478 and UTVOPE-0538 are written `D_i V_i^T = 0` (as the code's gram
   and TVOP-0108), not the tag's math-layout `U_i^T dU_i`. `tv_operations.py:774, 779` followed in revision 1 (S3,
   ``D V^T``), so the code tree now states the gauge in one layout.
2. **Corewise `(U, G)` → `(B, G)`** at `geometry.py:522, 536, 554, 1007, 1035` and `optimizers.py:879, 958, 1003, 1315`
   (`# initial cores (B, G)`). Found by the residual scan, not on a row at the time; the session ruled them (S1, rows
   RV8, RV9): each is a point's own cores (D-6). `geometry.py:995` `(U, G, G, G, shape, masks)` was a K4 hit (tier D).
3. **`-ward` rows in R4a's files.** Plan §3.3 files every travel-direction row under R4b; four sit in R4a's files
   (`manifold.py:949, 1172`, now 950, 1173; `tests/test_contractions_interpreter.py:227, 335`). The brief lists `-ward` travel words
   among R4a's row kinds and R4a owns those files, so they are done here ("upward ``ξ̂``", "the frame's upward edge
   variables", "adjoint-upward"). R4b's list should drop them.
4. **§4.11 loop-variable images** (the plan names `w` only): `stores_of` in `test_padded_frame.py:491` uses `D`/`U`
   (`tuple(D.shape[-2] for D in down), tuple(U.shape[-2] for U in up)`); `_uniform_padded_frame_testing.py:41` uses
   `Di` (`Nsh = tuple(Di.shape[-1] for Di in D)`). Listed in `R4a_name_hunks.tsv`; K15-cross takes `--names w,o`.
5. **`test_padded_start.py:568–570`**: the shared Tucker variation is `V` (plan §1.2a's text rows), not `D` as R1's
   handoff said. The sentence also drops the parameter letter ("at the dense frames of the point X"; the parameter is
   now `dense`) and writes "the untied up-frame Tucker columns": in the untied basis the Tucker columns are built from
   the up (core → leaf) frame, `eye(N_i) ⊗ U[i]`.
6. **`_padded_frame_oracles.py:66, 157, 170`**: the docstrings now name the R2-renamed parameters (``old_dense`` /
   ``cur_dense``, ``dense``) and the frame tuple `(D, phi, psi, U)`, the images of the map's dense-oracle rows; line
   41's tuple was a letter_sites row and these repeat it. Line 66 was rewrapped (over 120 otherwise).
7. **`utv_operations.py`**: `utv_to_ut3`'s comments say `[H, L]`, `[R, H]`, "to H"; the local, `dG` in the first
   pass, is `H` since revision 1 (S4); `utv_corewise_retract`'s comments use the generic `n` (item 12's rule);
   `Q_ii`→`R_ii` (K8) with "right core R" completing the sentence.
8. **Kept words:** "up/outer" (FVF-0354, FO-0201: T4S's "outer" beside the store word); "left/right/up-orthogonal"
   (`manifold.py:1344`); "an up width" (a test label); `fv_conversions.py:207` "Tucker down-orth" (correct in the new
   convention: the Tucker cores are down-orthogonalized); `utv_operations.py:585` "up front" (English).
   (`t3_orthogonalization.py:134`, kept in the first pass, was rewritten in revision 1, S5.)
9. **`tv_corewise_lift`** gained one sentence saying why `theta = (D, L)` is written by the frame's stores while its
   direction `theta_hat = (Bdot, Gdot)` takes `B` (the two letters sit side by side in that docstring).

## 6. Not done, reported, and tool changes

**Reported in the first pass, ruled and applied in revision 1 (§0):** `utv_operations.py`'s frame `P`/`Q` (MINOR 1);
the comprehension `B` at `frame_variations_format.py:856` and its siblings (S2); `tv_operations.py:774, 779` (S3).

**Reported, unchanged:**

- pqu misclassifications that K8 flags (carved out, below): `t3_svd.py:424, 426` `P_i` (a coefficient block),
  `tests/backend/test_uniform_fitting.py:430` `O` (the order count), `tests/test_dispatch.py:2006, 2029` `P_{T_F}` (the
  tangent projector). Also `examples/fit_hilbert_uniform_probe_derivatives_newton_cg.py:21` (`P` is a direction).

**Tool changes (dev/, for the reviewer's eye):**

- `store_rename_check.py`:
  - **K9 clauses (c)/(d)** as registrations (`k9_registered.tsv`, matched by whitespace-normalised line text, so
    they survive line shifts and realignment and lapse when the words change): 21 R3 Appendix C lines (clause d, S4;
    their texts checked equal to Appendix C), 14 R4a tier-E rewrites (c), 1 keep, 12 `swap` lines (pairs R1–R3
    reordered into `(down, up)` order, each line's words matching its own NAMEs, which line alignment pairs with the
    neighbour). New `K9-registered` reports registrations that no longer meet a residual line (0).
  - **K9 skips doctest source** (`>>>`/`...` lines; their trailing comments are still read): doctest code is NAMEs,
    K14/K15's. This removed 4 R4a lines (`uniform_frame_variations_format.py:1029–1046`) and none elsewhere.
  - **K8 carve-outs** `K8_R4A_CARVES`, by (file, hit, line text): the five lines above.
  - **K14 after R3**: the "previous slice vs base" comparison is the previous slice's own gate (R3's §5.2 tests are in
    HEAD); from R4a on it is listed as info, not failed (`recheck_prev = stage == 'R3'`). The current slice's own
    hunks are still checked against HEAD (0 lost, 0 count differences).
  - **K14 and session rows (revision 1)**: a NAME row of tier `R4a-row` in `R1_hit_table.tsv` licenses exactly its
    own occurrences of a DO-NOT-TOUCH token (one use per row, per (file, token)); each use is listed as `ruled:` in
    the details. Without it, S2's eight `B`/`G` tokens fail K14 as lost (item 5 was built from every `B`/`G`).
- `store_rename_diagram_check.py`:
  - **K10b**: a `:` that closes a label (non-space before, space after) is not an alignment column. FVF-0354's old
    labels end at columns 35/36 by chance and flagged a false re-layout error; the real column (``einsum`` at 40)
    maps consistently. Self-test still PASS.
  - **K11 exceptions**: a fenced block whose inventory new text is recorded and differs is excepted (K12 checks it
    verbatim instead); `--list-exceptions` prints them (41). Synthetic fence lists (the self-test) get no exceptions.

**Plan text the writer may want** (not edited): §3.3's R4b list loses the four `-ward` rows above; R1 handoff's
`test_padded_start.py:568–569` "stays `D`" is superseded by §1.2a's `V`.

## 7. Gates, battery, suite, doctests, docs, floor env

All in the `t3toolbox` env unless stated; tree = the working tree as handed over, except that the battery and the
suite ran before the last edit (§5 item 2's nine comment/docstring lines in `geometry.py`/`optimizers.py`); the gates,
docs, doctests and the floor env ran after it. **Revision 1** (§0): the gates, the diagram check and the static
doctest comparison were rerun on the revised tree (results in the table); the battery, the suite and the floor env on
revision 1 are the reviewer's runs (one heavy job at a time on the machine), quoted in `R4a_review.md`: battery
**PASS (bitwise)**, 2,190 arrays, 856 texts, 0 errors, `NEW`/`LETTERS_NEW` True; suite `-n 3` **1,347 passed, 4
skipped, 48,724 subtests, 0 failed** (17 min 44 s); `--doctest-modules t3toolbox` **241 passed**; `sphinx -W` build
succeeded, exit 0; floor env, the 21 touched modules, **893 passed, 1 skipped** (started after the revision-1 edits);
gates K14/K15 (40 hunks)/K15-cross/K17 **PASS**; static doctest outputs 0 changed. Verdict: **APPROVE**.

**Gates** (`python dev/store_rename_check.py --stage R4a --hits dev/store_rename_slices --hunks
dev/store_rename_slices/R4a_name_hunks.tsv --names w,o`; `python dev/store_rename_diagram_check.py all`):

| gate | result |
|---|---|
| K1, K2, K4, K5, K6, K7, K8, K9 in R4a's files | **0** (whole tree: REPORT 10, 6, 119, 23, 0, 139, 630, 37 — all in R4b's files, the docs and the notes) |
| K8 | R4a's five carve-outs (§6) only; before them R4a's files had exactly those 6 hits |
| K9 | 0 in R4a's files with the registrations; `K9-converse` **PASS**; `K9-registered` **PASS** (all 48 met) |
| K0, K3, K6, K8b | **PASS** |
| K10/K10b | **PASS**: 33 in place (28 rewritten + 5 same), 13 re-laid out; 14 pending, all R4b's (A-0465, P-*) and the docs' |
| K11 | **PASS**: 259 untouched fences byte-identical to `v2026.4.0`; 41 rewritten blocks excepted, each verified by K12 |
| K12 | R4a's 47 of 47 filled and verbatim (PENDING overall: 254 blocks are R4b's and R5's) |
| K13 | R4a's files: **0**. The tool at `--stage R4a` stops skipping fenced spans (by design) and FAILs on 8 lines, all in R4b's fenced blocks P-0640/0646/1215/1221 of `probing.py` |
| K14 | **PASS**: 0 DNT tokens lost in a hunk against HEAD (8 `B`/`G` occurrences licensed by S2's session rows), 0 count differences after R4a's hunks, contract/einsum strings identical; 24 added (`H` in `tv_to_t3` and `utv_operations`, `V`/`H` in S2's comprehensions: map images) |
| K15 | **PASS**: 40 NAME hunks against HEAD, exactly `R4a_name_hunks.tsv` (Appendix B); the realignment changes no NAME |
| K15-cross | **PASS** with `--names w,o` (the §4.11 images `w`; the removed loop variable `o`) |
| K17 (+ shape) | **PASS**: 891 checked; shape-frame 0, shape-converse 0; shape-point 4, all R4b's |
| K16 | REPORT 20, R5's |

**Battery** (`PYTHONPATH=$PWD python dev/store_rename_battery.py run --out <scratch> --b9
dev/store_rename_slices/battery_ref/b9`, then `compare dev/store_rename_slices/battery_ref <scratch>`): **PASS
(bitwise)**, 2,190 arrays and 856 texts, `NEW=True`, `LETTERS_NEW=True`.

**Doctest outputs**: static comparison of every docstring example in the 54 changed files against HEAD: 2,585 examples,
**0 expected outputs changed**; source changes are comments and the realigned doctest lines only.

**Suite** (`PYTHONPATH=$PWD python -m pytest tests/ -q -n 3`): **1,347 passed, 4 skipped, 48,724 subtests passed, 0
failed** (19 min 01 s; R3's counts exactly). The one warning is the intended adam-on-manifold one.

**Doctests** (CI's three commands, run one after another): `pytest --doctest-modules t3toolbox` **241 passed**;
`python -m doctest docs/getting_started.rst` **pass**; `python -m doctest` over every `docs/*.md` and
`docs/contributor/*.md` except `doctest_style.md` **pass**.

**Docs** (`python -m sphinx -W --keep-going -b html docs <scratch>`): **build succeeded**, exit 0, no warning. (A
first build ran beside the suite and the session stopped it for memory; this is the rerun, alone.)

**Floor env** (`tttt`: Python 3.9, numpy 1.22.4, jax 0.4.30; one plain serial pytest, no `xdist`/`subtests`, run
alone): the 21 touched test modules (`test_dispatch`, `backend/test_padded_frame`, `test_frame_variations_format`,
`backend/test_zero_reference`, `test_contractions_interpreter`, `test_contractions_sharding`, `test_sharing`,
`test_manifold`, `test_uniform_manifold`, `backend/test_padded_start`, `backend/test_uniform_padded_frame`,
`test_tucker_tensor_train`, `test_uniform_probing`, `test_fitting`, `backend/test_geometry`, `backend/test_optimizers`,
`backend/test_uniform_fitting`, `backend/test_from_sweep_reuse`, `test_uniform_frame_variations_format`,
`test_uniform_weighted`, `test_weighted`; the two helpers are imported by them): **893 passed, 1 skipped, 0 failed**
(26 min 49 s).
`pytest --doctest-modules t3toolbox` in the floor env: **241 passed**.

## Appendix A. Every §4A block: before (v2026.4.0) and after (the tree)

Generated from the inventory entries (`diagrams_inventory.md`, filled by R4a and checked verbatim by K12). Old span
is the tag's; new span the tree's. `K10` = rewritten in place, `K10b` = re-laid out, `same` = unchanged with a
recorded same-reason. Derivations of the relations are in §3.2.

### FO-0201 — `t3toolbox/backend/fv_operations.py` 201–202 → 201–202, K10b (re-laid out)

before:

~~~
    - up_tucker U_i (all i), outer/down D_i (all i),
    - left L_i (i=0..d-2), right R_i (i=1..d-1).
~~~

after:

~~~
    - down_tucker D_i (all i), outer/up U_i (all i),
    - left L_i (i=0..d-2), right R_i (i=1..d-1).
~~~

### PF-0041 — `t3toolbox/backend/padded_frames.py` 41 → 42, K10b (re-laid out)

before:

~~~
#   (up, down, left, right), each len=d, elm_shapes C+(nUi, Ni), C+(rLi, nDi, rR(i+1)), C+(rLi, nUi, rL(i+1)), C+(rRi, nUi, rR(i+1))
~~~

after:

~~~
#   (down, up, left, right), each len=d, elm_shapes C+(nDi, Ni), C+(rLi, nUi, rR(i+1)), C+(rLi, nDi, rL(i+1)), C+(rRi, nDi, rR(i+1))
~~~

### PF-0043 — `t3toolbox/backend/padded_frames.py` 43 → 44, K10 (in place)

before:

~~~
#   (tucker_variations, tt_variations), each len=d, elm_shapes C+(nDi, Ni), C+(rLi, nUi, rR(i+1))
~~~

after:

~~~
#   (tucker_variations, tt_variations), each len=d, elm_shapes C+(nUi, Ni), C+(rLi, nDi, rR(i+1))
~~~

### RA-0644 — `t3toolbox/backend/ranks.py` 644–647 → 644–647, K10b (re-laid out)

before:

~~~
    1. Tucker down-orthogonalization: ``up_i = min(n_i, N_i)``;
    2. a LEFT sweep on the original TT bonds: ``left_{i+1} = min(left_i * up_i, r_{i+1})``;
    3. a RIGHT sweep over the left-orthogonal chain: ``right_i = min(left_i, up_i * right_{i+1})``;
    4. the down cores: ``down_i = min(up_i, left_i * right_{i+1})``.
~~~

after:

~~~
    1. Tucker down-orthogonalization: ``down_i = min(n_i, N_i)``;
    2. a LEFT sweep on the original TT bonds: ``left_{i+1} = min(left_i * down_i, r_{i+1})``;
    3. a RIGHT sweep over the left-orthogonal chain: ``right_i = min(left_i, down_i * right_{i+1})``;
    4. the up cores: ``up_i = min(down_i, left_i * right_{i+1})``.
~~~

### T3OR-0044 — `t3toolbox/backend/t3_orthogonalization.py` 44–46 → 44–46, same

~~~
    - Tucker B_i, all i:                 ``einsum('...io,...jo->...ij', B, B) = I``
    - TT G_i, left  (i = 0..d-2):        ``einsum('...aib,...aic->...bc', G, G) = I``
    - TT G_i, right (i = 1..d-1):        ``einsum('...aib,...cib->...ac', G, G) = I``
~~~

### TVOP-0108 — `t3toolbox/backend/tv_operations.py` 108 → 108, K10 (in place)

before:

~~~
    ``U_i V_i^T = 0`` (all i) and ``einsum('...abi,...abj->...ij', L_i, H_i) = 0`` (i = 0..d-2).
~~~

after:

~~~
    ``D_i V_i^T = 0`` (all i) and ``einsum('...abi,...abj->...ij', L_i, H_i) = 0`` (i = 0..d-2).
~~~

### UFOP-0083 — `t3toolbox/backend/ufv_operations.py` 83 → 83, K10b (re-laid out)

before:

~~~
        nU = up.shape[-2]    nD = down.shape[-2]    rL = left.shape[-1]    rR = right.shape[-1]    N = up.shape[-1]
~~~

after:

~~~
        nD = down.shape[-2]    nU = up.shape[-2]    rL = left.shape[-1]    rR = right.shape[-1]    N = down.shape[-1]
~~~

### UPF-0047 — `t3toolbox/backend/uniform_padded_frames.py` 47–48 → 47–48, K10b (re-laid out)

before:

~~~
#   up (d,)+C+(nU, N), down (d,)+C+(rL, nD, rR), left (d,)+C+(rL, nU, rL), right (d,)+C+(rR, nU, rR), shape (len=d),
#   masks HOST bool (up (d,)+C+(nU,), down (d,)+C+(nD,), left (d+1,)+C+(rL,), right (d+1,)+C+(rR,)); block layout: nU = nD, rL = rR
~~~

after:

~~~
#   down (d,)+C+(nD, N), up (d,)+C+(rL, nU, rR), left (d,)+C+(rL, nD, rL), right (d,)+C+(rR, nD, rR), shape (len=d),
#   masks HOST bool (down (d,)+C+(nD,), up (d,)+C+(nU,), left (d+1,)+C+(rL,), right (d+1,)+C+(rR,)); block layout: nU = nD, rL = rR
~~~

### UPF-0050 — `t3toolbox/backend/uniform_padded_frames.py` 50 → 50, K10 (in place)

before:

~~~
#   V (d,)+C+(nD, N), H (d,)+C+(rL, nU, rR), shape, variation masks; only V, H are read (masked with the frame's own masks)
~~~

after:

~~~
#   V (d,)+C+(nU, N), H (d,)+C+(rL, nD, rR), shape, variation masks; only V, H are read (masked with the frame's own masks)
~~~

### UTVOPE-0314 — `t3toolbox/backend/utv_operations.py` 314–315 → 314–315, K10b (re-laid out)

before:

~~~
    # ---- Doubled rank masks (HOST numpy; carry K+C from the variation masks). The order respects the
    # block structure (T4S (49)-(52)): Tucker axis [U | dU] = [up | down]; the TT bond is [Q | P] = [R | L], so
~~~

after:

~~~
    # ---- Doubled rank masks (HOST numpy; carry K+C from the variation masks). The order respects the
    # block structure (T4S (49)-(52)): Tucker axis [D | V] = [down | up]; the TT bond is [R | L] (T4S's [Q | P]), so
~~~

### UTVOPE-0538 — `t3toolbox/backend/utv_operations.py` 538–539 → 538–539, K10b (re-laid out)

before:

~~~
    Each Tucker variation must be orthogonal to its up-core (``U^T dU = 0``) and each left-interior TT
    variation orthogonal to its left-core (``(P^L)^T dG^L = 0``). Mask-once, then the gauge grams vectorized
~~~

after:

~~~
    Each Tucker variation must be orthogonal to its down-core (``D V^T = 0``) and each left-interior TT
    variation orthogonal to its left-core (``(L^L)^T H^L = 0``). Mask-once, then the gauge grams vectorized
~~~

### FVF-0058a — `t3toolbox/frame_variations_format.py` 58–61, columns 8–32 → 59–62, columns 8–32, K10 (in place)

before:

~~~
1--(H0)--R1---R2---1    
    |    |    |         
    U0   U1   U2        
    |    |    |         
~~~

after:

~~~
1--(H0)--R1---R2---1    
    |    |    |         
    D0   D1   D2        
    |    |    |         
~~~

### FVF-0058b — `t3toolbox/frame_variations_format.py` 58–61, columns 32–56 → 59–62, columns 32–56, K10 (in place)

before:

~~~
1---L0--(H1)--R2---1    
    |    |    |         
    U0   U1   U2        
    |    |    |         
~~~

after:

~~~
1---L0--(H1)--R2---1    
    |    |    |         
    D0   D1   D2        
    |    |    |         
~~~

### FVF-0058c — `t3toolbox/frame_variations_format.py` 58–61, columns 56–end → 59–62, columns 56–end, K10 (in place)

before:

~~~
1---L0---L1--(H2)--1
    |    |    |
    U0   U1   U2
    |    |    |
~~~

after:

~~~
1---L0---L1--(H2)--1
    |    |    |
    D0   D1   D2
    |    |    |
~~~

### FVF-0063a — `t3toolbox/frame_variations_format.py` 63–66, columns 8–32 → 64–67, columns 8–32, K10 (in place)

before:

~~~
1---D0---R1---R2---1    
    |    |    |         
   (V0)  U1   U2        
    |    |    |         
~~~

after:

~~~
1---U0---R1---R2---1    
    |    |    |         
   (V0)  D1   D2        
    |    |    |         
~~~

### FVF-0063b — `t3toolbox/frame_variations_format.py` 63–66, columns 32–56 → 64–67, columns 32–56, K10 (in place)

before:

~~~
1---L0---D1---R2---1    
    |    |    |         
    U0  (V1)  U2        
    |    |    |         
~~~

after:

~~~
1---L0---U1---R2---1    
    |    |    |         
    D0  (V1)  D2        
    |    |    |         
~~~

### FVF-0063c — `t3toolbox/frame_variations_format.py` 63–66, columns 56–end → 64–67, columns 56–end, K10 (in place)

before:

~~~
1---L0---L1---D2---1
    |    |    |
    U0   U1  (V2)
    |    |    |
~~~

after:

~~~
1---L0---L1---U2---1
    |    |    |
    D0   D1  (V2)
    |    |    |
~~~

### FVF-0071 — `t3toolbox/frame_variations_format.py` 71–74 → 72–75, K10b (re-laid out)

before:

~~~
        - up_tucker_cores   = (U0, ..., U(d-1)), elm_shape=(nUi, Ni)
        - down_tt_cores     = (D0, ..., D(d-1)), elm_shape=(rLi, nDi, rR(i+1))
        - left_tt_cores     = (L0, ..., L(d-1)), elm_shape=(rLi, ni, rL(i+1))
        - right_tt_cores    = (R0, ..., R(d-1)), elm_shape=(rRi, ni, rR(i+1))
~~~

after:

~~~
        - down_tucker_cores = (D0, ..., D(d-1)), elm_shape=(nDi, Ni)
        - up_tt_cores       = (U0, ..., U(d-1)), elm_shape=(rLi, nUi, rR(i+1))
        - left_tt_cores     = (L0, ..., L(d-1)), elm_shape=(rLi, ni, rL(i+1))
        - right_tt_cores    = (R0, ..., R(d-1)), elm_shape=(rRi, ni, rR(i+1))
~~~

### FVF-0077 — `t3toolbox/frame_variations_format.py` 77–78 → 78–79, K10 (in place)

before:

~~~
        - tucker_variations = (V0, ..., V(d-1)), elm_shape=(nDi, Ni)
        - tt_variations     = (H0, ..., H(d-1)), elm_shape=(rLi, nUi, rRi)
~~~

after:

~~~
        - tucker_variations = (V0, ..., V(d-1)), elm_shape=(nUi, Ni)
        - tt_variations     = (H0, ..., H(d-1)), elm_shape=(rLi, nDi, rRi)
~~~

### FVF-0088 — `t3toolbox/frame_variations_format.py` 88–96 → 89–97, K10 (in place)

before:

~~~
           rL0       rL1       rR2      rR(d-1)         rRd
        1 ------ L0 ----- (H1) ----- ... ------ R(d-1) ------ 1
                 |         |                    |
                 | nU0     | nU1                | nU(d-1)
                 |         |                    |
                 U0        U1                   Ud
                 |         |                    |
                 | N0      | N1                 | N(d-1)
                 |         |                    |
~~~

after:

~~~
           rL0       rL1       rR2      rR(d-1)         rRd
        1 ------ L0 ----- (H1) ----- ... ------ R(d-1) ------ 1
                 |         |                    |
                 | nD0     | nD1                | nD(d-1)
                 |         |                    |
                 D0        D1                   Dd
                 |         |                    |
                 | N0      | N1                 | N(d-1)
                 |         |                    |
~~~

### FVF-0100 — `t3toolbox/frame_variations_format.py` 100–108 → 101–109, K10 (in place)

before:

~~~
           rL0       rL1       rR2      rR(d-1)         rRd
        1 ------ L0 ------ D1 ------ ... ------ R(d-1) ------ 1
                 |         |                    |
                 | nU0     | nO1                | nU(d-1)
                 |         |                    |
                 U0       (V1)                   Ud
                 |         |                    |
                 | N0      | N1                 | N(d-1)
                 |         |                    |
~~~

after:

~~~
           rL0       rL1       rR2      rR(d-1)         rRd
        1 ------ L0 ------ U1 ------ ... ------ R(d-1) ------ 1
                 |         |                    |
                 | nD0     | nU1                | nD(d-1)
                 |         |                    |
                 D0       (V1)                   Dd
                 |         |                    |
                 | N0      | N1                 | N(d-1)
                 |         |                    |
~~~

### FVF-0117 — `t3toolbox/frame_variations_format.py` 117–120 → 118–121, K10b (re-laid out)

before:

~~~
        - up_tucker_cores   = (U0,...,U(d-1)), orthogonal:       U_ia U_ja = delta_ij
        - down_tt_cores     = (O0,...,O(d-1)), outer-orthogonal  O_aib O_ajb = delta_ij
        - left_tt_cores     = (L0,...,L(d-1)), left-orthogonal:  L_abi L_abj = delta_ij
        - right_tt_cores    = (R0,...,R(d-1)), right-orthogonal  R_ibc R_jbc = delta_ij
~~~

after:

~~~
        - down_tucker_cores = (D0,...,D(d-1)), down-orthogonal:  D_ia D_ja = delta_ij
        - up_tt_cores       = (U0,...,U(d-1)), up-orthogonal:    U_aib U_ajb = delta_ij
        - left_tt_cores     = (L0,...,L(d-1)), left-orthogonal:  L_abi L_abj = delta_ij
        - right_tt_cores    = (R0,...,R(d-1)), right-orthogonal: R_ibc R_jbc = delta_ij
~~~

### FVF-0123 — `t3toolbox/frame_variations_format.py` 123–124 → 124–125, K10 (in place)

before:

~~~
        - U_ia V_ja = 0    (all V)
        - L_abi H_abj = 0  (all but the last H)
~~~

after:

~~~
        - D_ia V_ja = 0    (all V)
        - L_abi H_abj = 0  (all but the last H)
~~~

### FVF-0156 — `t3toolbox/frame_variations_format.py` 156–159, columns 48–end → 157–160, columns 48–end, K10 (in place)

before:

~~~
# len=d. B_xo B_yo   = I_xy, Bi.shape = stack_shape+(nUi, Ni)
# len=d. R_ixj R_iyj = I_xy  Ri.shape = stack_shape+(rLi, nDi, rR(i+1))
# len=d. P_iax P_iay = I_xy, Pi.shape = stack_shape+(rLi, nUi, rL(i+1))
# len=d. Q_xaj Q_yaj = I_xy  Qi.shape = stack_shape+(rRi, nUi, rR(i+1))
~~~

after:

~~~
# len=d. D_xo D_yo   = I_xy, Di.shape = stack_shape+(nDi, Ni)
# len=d. U_ixj U_iyj = I_xy  Ui.shape = stack_shape+(rLi, nUi, rR(i+1))
# len=d. L_iax L_iay = I_xy, Li.shape = stack_shape+(rLi, nDi, rL(i+1))
# len=d. R_xaj R_yaj = I_xy  Ri.shape = stack_shape+(rRi, nDi, rR(i+1))
~~~

### FVF-0217 — `t3toolbox/frame_variations_format.py` 217–220 → 218–221, K10 (in place)

before:

~~~
            1 -- L0 -- ( ) -- R2 -- R3 -- 1
                 |      |      |      |
                 U0     U1     U2     U3
                 |      |      |      |
~~~

after:

~~~
            1 -- L0 -- ( ) -- R2 -- R3 -- 1
                 |      |      |      |
                 D0     D1     D2     D3
                 |      |      |      |
~~~

### FVF-0222 — `t3toolbox/frame_variations_format.py` 222–225 → 223–226, K10 (in place)

before:

~~~
            1 -- L0 -- L1 -- O2 -- R3 -- 1
                 |     |     |     |
                 U0    U1    ( )   U3
                 |     |     |     |
~~~

after:

~~~
            1 -- L0 -- L1 -- U2 -- R3 -- 1
                 |     |     |     |
                 D0    D1    ( )   D3
                 |     |     |     |
~~~

### FVF-0354 — `t3toolbox/frame_variations_format.py` 354–357 → 355–358, K10b (re-laid out)

before:

~~~
            - up_tucker U_i (all i):    ``einsum('...io,...jo->...ij', U, U) = I``
            - down/outer D_i (all i):   ``einsum('...iaj,...ibj->...ab', D, D) = I``
            - left L_i (i = 0..d-2):    ``einsum('...iaj,...iak->...jk', L, L) = I``
            - right R_i (i = 1..d-1):   ``einsum('...iaj,...kaj->...ik', R, R) = I``
~~~

after:

~~~
            - down_tucker D_i (all i):  ``einsum('...io,...jo->...ij', D, D) = I``
            - up/outer U_i (all i):     ``einsum('...iaj,...ibj->...ab', U, U) = I``
            - left L_i (i = 0..d-2):    ``einsum('...iaj,...iak->...jk', L, L) = I``
            - right R_i (i = 1..d-1):   ``einsum('...iaj,...kaj->...ik', R, R) = I``
~~~

### FVF-0784 — `t3toolbox/frame_variations_format.py` 784–785 → 785–786, K10 (in place)

before:

~~~
        - tucker_variations    = (V0, ..., V(d-1)), elm_shape=stack_shape+(nDi, Ni)
        - tt_variations        = (H0, ..., H(d-1)), elm_shape=stack_shape+(rLi, nUi, rRi)
~~~

after:

~~~
        - tucker_variations    = (V0, ..., V(d-1)), elm_shape=stack_shape+(nUi, Ni)
        - tt_variations        = (H0, ..., H(d-1)), elm_shape=stack_shape+(rLi, nDi, rRi)
~~~

### FVF-0845 — `t3toolbox/frame_variations_format.py` 845–848 → 846–849, K10 (in place)

before:

~~~
            1 -- L0 -- ( ) -- R2 -- R3 -- 1
                 |      |      |      |
                 U0     U1     U2     U3
                 |      |      |      |
~~~

after:

~~~
            1 -- L0 -- ( ) -- R2 -- R3 -- 1
                 |      |      |      |
                 D0     D1     D2     D3
                 |      |      |      |
~~~

### FVF-0850 — `t3toolbox/frame_variations_format.py` 850–853 → 851–854, K10 (in place)

before:

~~~
            1 -- L0 -- L1 -- O2 -- R3 -- 1
                 |     |     |     |
                 U0    U1    ( )   U3
                 |     |     |     |
~~~

after:

~~~
            1 -- L0 -- L1 -- U2 -- R3 -- 1
                 |     |     |     |
                 D0    D1    ( )   D3
                 |     |     |     |
~~~

### FVF-1264 — `t3toolbox/frame_variations_format.py` 1264–1267 → 1265–1268, K10 (in place)

before:

~~~
        1 -- L0 --(H1)-- R2 -- R3 -- 1
             |     |     |     |
             U0    U1    U2    U3
             |     |     |     |
~~~

after:

~~~
        1 -- L0 --(H1)-- R2 -- R3 -- 1
             |     |     |     |
             D0    D1    D2    D3
             |     |     |     |
~~~

### FVF-1271 — `t3toolbox/frame_variations_format.py` 1271–1274 → 1272–1275, K10 (in place)

before:

~~~
        1 -- L0 -- L1 -- D2 -- R3 -- 1
             |     |     |     |
             U0    U1   (V2)   U3
             |     |     |     |
~~~

after:

~~~
        1 -- L0 -- L1 -- U2 -- R3 -- 1
             |     |     |     |
             D0    D1   (V2)   D3
             |     |     |     |
~~~

### FVF-1302 — `t3toolbox/frame_variations_format.py` 1302–1322 → 1303–1323, K10b (re-laid out)

before:

~~~
    >>> (U0, U1, U2) = (randn(10, 14), randn(11, 15), randn(12, 16))
    >>> (L0, L1, L2) = (randn(1, 10, 2), randn(2, 11, 3), randn(3, 12, 4))
    >>> (R0, R1, R2) = (randn(2, 10, 4), randn(4, 11, 5), randn(5, 12, 1))
    >>> (D0, D1, D2) = (randn(1, 9, 4), randn(2, 8, 5), randn(3, 7, 1))
    >>> frame = bvf.T3Frame((U0, U1, U2), (D0, D1, D2), (L0, L1, L2), (R0, R1, R2))
    >>> (V0, V1, V2) = (randn(9, 14), randn(8, 15), randn(7, 16))
    >>> (H0, H1, H2) = (randn(1, 10, 4), randn(2, 11, 5), randn(3, 12, 1))
    >>> variations = bvf.T3Variations((V0, V1, V2), (H0, H1, H2))

    Replacing the index-1 TT-core swaps ``H1`` into the right-orthogonal chain ``L0, ?, R2``; the
    Tucker (up) cores are unchanged:

    >>> tt_term = bvf.fv_to_t3((True, 1), frame, variations)
    >>> expected = ((U0, U1, U2), (L0, H1, R2))      # up cores untouched; TT chain = L0, H1, R2
    >>> print(np.allclose(cw.corewise_norm(cw.corewise_sub(tt_term.data, expected)), 0.0))
    True

    Replacing the index-1 Tucker core swaps ``V1`` into the up cores and the down core ``D1`` into the chain:

    >>> tucker_term = bvf.fv_to_t3((False, 1), frame, variations)
    >>> expected = ((U0, V1, U2), (L0, D1, R2))
~~~

after:

~~~
    >>> (D0, D1, D2) = (randn(10, 14), randn(11, 15), randn(12, 16))
    >>> (L0, L1, L2) = (randn(1, 10, 2), randn(2, 11, 3), randn(3, 12, 4))
    >>> (R0, R1, R2) = (randn(2, 10, 4), randn(4, 11, 5), randn(5, 12, 1))
    >>> (U0, U1, U2) = (randn(1, 9, 4), randn(2, 8, 5), randn(3, 7, 1))
    >>> frame = bvf.T3Frame((D0, D1, D2), (U0, U1, U2), (L0, L1, L2), (R0, R1, R2))
    >>> (V0, V1, V2) = (randn(9, 14), randn(8, 15), randn(7, 16))
    >>> (H0, H1, H2) = (randn(1, 10, 4), randn(2, 11, 5), randn(3, 12, 1))
    >>> variations = bvf.T3Variations((V0, V1, V2), (H0, H1, H2))

    Replacing the index-1 TT-core swaps ``H1`` into the right-orthogonal chain ``L0, ?, R2``; the
    Tucker (down) cores are unchanged:

    >>> tt_term = bvf.fv_to_t3((True, 1), frame, variations)
    >>> expected = ((D0, D1, D2), (L0, H1, R2))      # down cores untouched; TT chain = L0, H1, R2
    >>> print(np.allclose(cw.corewise_norm(cw.corewise_sub(tt_term.data, expected)), 0.0))
    True

    Replacing the index-1 Tucker core swaps ``V1`` into the down cores and the up core ``U1`` into the chain:

    >>> tucker_term = bvf.fv_to_t3((False, 1), frame, variations)
    >>> expected = ((D0, V1, D2), (L0, U1, R2))
~~~

### FVF-1355 — `t3toolbox/frame_variations_format.py` 1355–1358 → 1356–1359, K10 (in place)

before:

~~~
                  1 -- L0 -- H1 -- R2 -- R3 -- 1
        X    =         |     |     |     |
                       U0    U1    U2    U3
                       |     |     |     |
~~~

after:

~~~
                  1 -- L0 -- H1 -- R2 -- R3 -- 1
        X    =         |     |     |     |
                       D0    D1    D2    D3
                       |     |     |     |
~~~

### FVF-1362 — `t3toolbox/frame_variations_format.py` 1362–1365 → 1363–1366, K10 (in place)

before:

~~~
                  1 -- L0 -- L1 -- D2 -- R3 -- 1
        X    =         |     |     |     |
                       U0    U1    V2    U3
                       |     |     |     |
~~~

after:

~~~
                  1 -- L0 -- L1 -- U2 -- R3 -- 1
        X    =         |     |     |     |
                       D0    D1    V2    D3
                       |     |     |     |
~~~

### FVF-1371 — `t3toolbox/frame_variations_format.py` 1371–1374 → 1372–1375, K10b (re-laid out)

before:

~~~
        - tucker_cores      = (U0,U1, U2, U3), up orthogonal
        - down_tt_cores     = (O0, O1, O2, O3), down orthogonal
        - left_tt_cores     = (L0, L1, L2),     left orthogonal
        - right_tt_cores    = (R1, R2, R3),     right orthogonal
~~~

after:

~~~
        - down_tucker_cores = (D0, D1, D2, D3), down orthogonal
        - up_tt_cores       = (U0, U1, U2, U3), up orthogonal
        - left_tt_cores     = (L0, L1, L2),     left orthogonal
        - right_tt_cores    = (R1, R2, R3),     right orthogonal
~~~

### M-0685 — `t3toolbox/manifold.py` 685–686 → 686–687, K10 (in place)

before:

~~~
        - ``einsum('...ia,...ja->...ij', U_i, V_i) = 0`` for all i (Tucker variations ⟂ U).
        - ``einsum('...abi,...abj->...ij', L_i, H_i) = 0`` for i = 0..d-2 (TT variations ⟂ L).
~~~

after:

~~~
        - ``einsum('...ia,...ja->...ij', D_i, V_i) = 0`` for all i (Tucker variations ⟂ D).
        - ``einsum('...abi,...abj->...ij', L_i, H_i) = 0`` for i = 0..d-2 (TT variations ⟂ L).
~~~

### TTT-2765 — `t3toolbox/tucker_tensor_train.py` 2765 → 2765, same

~~~
            I.e., ``np.einsum('...io,jo->...ij', B, B) = (stacked) identity matrix``, where ``B=new_x.tucker_cores[ii]``.
~~~

### TTT-2837 — `t3toolbox/tucker_tensor_train.py` 2837 → 2837, same

~~~
            I.e., ``einsum('...iaj,...iak->...jk', G, G) = (stacked) identity matrix``, where ``G=new_x.tt_cores[ii]``.
~~~

### TTT-2908 — `t3toolbox/tucker_tensor_train.py` 2908 → 2908, same

~~~
            I.e., ``einsum('...iaj,...kaj->...ik', G, G) = (stacked) identity matrix``, where ``G=new_tt_cores[ii]``.
~~~

### TTT-2979 — `t3toolbox/tucker_tensor_train.py` 2979 → 2979, same

~~~
            I.e., ``einsum('...iaj,...ibj->...ab', G, G) = (stacked) identity matrix``, where ``G=new_tt_cores[ii]``.
~~~

### UFVF-0110 — `t3toolbox/uniform_frame_variations_format.py` 110–113, columns 49–end → 110–113, columns 49–end, K10 (in place)

before:

~~~
# B_dxo B_dyo   = I_dxy, shape = (d,)+stack_shape+(nU, N)
# R_dixj R_diyj = I_dxy  shape = (d,)+stack_shape+(rL, nD, rR)
# P_diax P_diay = I_dxy, shape = (d,)+stack_shape+(rL, nU, rL)
# Q_dxaj Q_dyaj = I_dxy  shape = (d,)+stack_shape+(rR, nU, rR)
~~~

after:

~~~
# D_dxo D_dyo   = I_dxy, shape = (d,)+stack_shape+(nD, N)
# U_dixj U_diyj = I_dxy  shape = (d,)+stack_shape+(rL, nU, rR)
# L_diax L_diay = I_dxy, shape = (d,)+stack_shape+(rL, nD, rL)
# R_dxaj R_dyaj = I_dxy  shape = (d,)+stack_shape+(rR, nD, rR)
~~~

### UFVF-1106 — `t3toolbox/uniform_frame_variations_format.py` 1106–1109 → 1106–1109, K10 (in place)

before:

~~~
                  1 -- L0 -- H1 -- R2 -- R3 -- 1
        X    =         |     |     |     |
                       U0    U1    U2    U3
                       |     |     |     |
~~~

after:

~~~
                  1 -- L0 -- H1 -- R2 -- R3 -- 1
        X    =         |     |     |     |
                       D0    D1    D2    D3
                       |     |     |     |
~~~

### UFVF-1113 — `t3toolbox/uniform_frame_variations_format.py` 1113–1116 → 1113–1116, K10 (in place)

before:

~~~
                  1 -- L0 -- L1 -- O2 -- R3 -- 1
        X    =         |     |     |     |
                       U0    U1    V2    U3
                       |     |     |     |
~~~

after:

~~~
                  1 -- L0 -- L1 -- U2 -- R3 -- 1
        X    =         |     |     |     |
                       D0    D1    V2    D3
                       |     |     |     |
~~~

### UFVF-1122 — `t3toolbox/uniform_frame_variations_format.py` 1122–1125 → 1122–1125, K10b (re-laid out)

before:

~~~
        - tucker_cores       = (U0,U1, U2, U3), up orthogonal
        - left_tt_cores     = (L0, L1, L2),     left orthogonal
        - right_tt_cores    = (R1, R2, R3),     right orthogonal
        - outer_tt_cores    = (O0, O1, O2, O3), down orthogonal
~~~

after:

~~~
        - down_tucker_cores = (D0, D1, D2, D3), down orthogonal
        - left_tt_cores     = (L0, L1, L2),     left orthogonal
        - right_tt_cores    = (R1, R2, R3),     right orthogonal
        - up_tt_cores       = (U0, U1, U2, U3), up orthogonal
~~~

### UNMA-0478 — `t3toolbox/uniform_manifold.py` 478–479 → 478–479, K10b (re-laid out)

before:

~~~
        coordinate ones; not enforced at construction): ``U_i^T dU_i = 0`` (all i) and
        ``(P_i^L)^T dG_i^L = 0`` (i = 0..d-2) -- conditions (47)-(48), Appendix A.3 of Alger et al. (2026).
~~~

after:

~~~
        coordinate ones; not enforced at construction): ``D_i V_i^T = 0`` (all i) and
        ``(L_i^L)^T H_i^L = 0`` (i = 0..d-2) -- conditions (47)-(48), Appendix A.3 of Alger et al. (2026).
~~~

### T-TFVF-0025 — `tests/test_frame_variations_format.py` 25–27 → 25–27, K10 (in place)

before:

~~~
        U_i = (nU_i, N_i)           L_i = (rL_i, nU_i, rL_(i+1))
        R_i = (rR_i, nU_i, rR_(i+1))  D_i = (rL_i, nD_i, rR_(i+1))
        V_i = (nD_i, N_i)           H_i = (rL_i, nU_i, rR_(i+1))
~~~

after:

~~~
        D_i = (nD_i, N_i)           L_i = (rL_i, nD_i, rL_(i+1))
        R_i = (rR_i, nD_i, rR_(i+1))  U_i = (rL_i, nU_i, rR_(i+1))
        V_i = (nU_i, N_i)           H_i = (rL_i, nD_i, rR_(i+1))
~~~

## Appendix B. The NAME hunks (K15; `R4a_name_hunks.tsv`)

Line numbers HEAD → tree; one row per hunk occurrence, as the tool writes them. First pass (18): `tv_to_t3`'s `dU` → `H`
(the V5 row, §3.3) and §4.11's loop variables. Revision 1 (22): S2's comprehensions in `frame_variations_format.py`,
S4's `udots` → `ddots` and the `dG` family in `utv_operations.py` (rows `LS0315`–`LS0338`, tier `R4a-row`).

| file | lines | old → new |
|---|---|---|
| `t3toolbox/backend/fv_operations.py` | 653-653 → 653-653 | `dn` → `w` |
| `t3toolbox/backend/fv_operations.py` | 653-653 → 653-653 | `dn` → `w` |
| `t3toolbox/backend/fv_operations.py` | 654-654 → 654-654 | `u` → `w` |
| `t3toolbox/backend/fv_operations.py` | 655-655 → 655-655 | `u` → `w` |
| `t3toolbox/backend/tv_operations.py` | 245-245 → 245-245 | `udots` → `ddots` |
| `t3toolbox/backend/tv_operations.py` | 250-250 → 250-250 | `udots` → `ddots` |
| `t3toolbox/backend/tv_operations.py` | 292-292 → 292-292 | `dU` → `H` |
| `t3toolbox/backend/tv_operations.py` | 296-296 → 296-296 | `dU` → `H` |
| `t3toolbox/backend/tv_operations.py` | 306-306 → 306-306 | `dU` → `H` |
| `t3toolbox/backend/tv_operations.py` | 307-307 → 307-307 | `dU` → `H` |
| `t3toolbox/backend/tv_operations.py` | 313-313 → 313-313 | `dU` → `H` |
| `t3toolbox/backend/tv_operations.py` | 323-323 → 323-323 | `dU` → `H` |
| `t3toolbox/backend/tv_operations.py` | 329-329 → 329-329 | `dU` → `H` |
| `t3toolbox/backend/tv_operations.py` | 331-331 → 331-331 | `dU` → `H` |
| `t3toolbox/backend/utv_operations.py` | 224-224 → 224-224 | `udots` → `ddots` |
| `t3toolbox/backend/utv_operations.py` | 241-241 → 241-241 | `udots` → `ddots` |
| `t3toolbox/backend/utv_operations.py` | 294-294 → 294-294 | `dG` → `H` |
| `t3toolbox/backend/utv_operations.py` | 298-298 → 298-298 | `dG dG` → `H H` |
| `t3toolbox/backend/utv_operations.py` | 299-299 → 299-299 | `dG` → `H` |
| `t3toolbox/backend/utv_operations.py` | 485-485 → 485-485 | `dG_in` → `H_in` |
| `t3toolbox/backend/utv_operations.py` | 486-486 → 486-486 | `dG dG_in` → `H H_in` |
| `t3toolbox/backend/utv_operations.py` | 487-487 → 487-487 | `dG` → `H` |
| `t3toolbox/backend/utv_operations.py` | 488-488 → 488-488 | `dG` → `H` |
| `t3toolbox/backend/utv_operations.py` | 489-489 → 489-489 | `dG` → `H` |
| `t3toolbox/backend/utv_operations.py` | 523-523 → 523-523 | `dG_stack` → `H_stack` |
| `t3toolbox/backend/utv_operations.py` | 524-524 → 524-524 | `dG_stack` → `H_stack` |
| `t3toolbox/frame_variations_format.py` | 855-855 → 856-856 | `B` → `V` |
| `t3toolbox/frame_variations_format.py` | 855-855 → 856-856 | `B` → `V` |
| `t3toolbox/frame_variations_format.py` | 856-856 → 857-857 | `G` → `H` |
| `t3toolbox/frame_variations_format.py` | 856-856 → 857-857 | `G` → `H` |
| `t3toolbox/frame_variations_format.py` | 940-940 → 941-941 | `B` → `V` |
| `t3toolbox/frame_variations_format.py` | 940-940 → 941-941 | `B` → `V` |
| `t3toolbox/frame_variations_format.py` | 941-941 → 942-942 | `G` → `H` |
| `t3toolbox/frame_variations_format.py` | 941-941 → 942-942 | `G` → `H` |
| `tests/backend/_uniform_padded_frame_testing.py` | 41-41 → 41-41 | `u` → `Di` |
| `tests/backend/_uniform_padded_frame_testing.py` | 41-41 → 41-41 | `u` → `Di` |
| `tests/backend/test_padded_frame.py` | 491-491 → 491-491 | `u` → `D` |
| `tests/backend/test_padded_frame.py` | 491-491 → 491-491 | `u` → `D` |
| `tests/backend/test_padded_frame.py` | 491-491 → 491-491 | `o` → `U` |
| `tests/backend/test_padded_frame.py` | 491-491 → 491-491 | `o` → `U` |

The lines as they read now:

~~~
t3toolbox/backend/fv_operations.py:653: wV = tuple(xnp.einsum('...i,...io->...io', w, V) for w, V in zip(up, V_cores))
t3toolbox/backend/fv_operations.py:654: wH = tuple(xnp.einsum('...aib,...a,...i,...b->...aib', H, lf, w, rt)
t3toolbox/backend/fv_operations.py:655: for H, lf, w, rt in zip(H_cores, left, down, right))
tests/backend/_uniform_padded_frame_testing.py:41: Nsh = tuple(Di.shape[-1] for Di in D)
tests/backend/test_padded_frame.py:491: return (tuple(D.shape[-2] for D in down), tuple(U.shape[-2] for U in up),
t3toolbox/backend/tv_operations.py:245: ddots = sharing_module.fv_tied_ambient_directions(variations, shared_data)
t3toolbox/backend/tv_operations.py:250: tucker_variations[ii] = ddots[gi]                    # ONE array per group
t3toolbox/backend/tv_operations.py:292: H = tt_variations[0]
t3toolbox/frame_variations_format.py:856: tucker_variation_shapes = tuple([V.shape[-2:] for V in self.tucker_variations])
t3toolbox/frame_variations_format.py:857: tt_variation_shapes = tuple([H.shape[-3:] for H in self.tt_variations])
t3toolbox/frame_variations_format.py:941: tucker_stack_shapes = tuple([V.shape[:-2] for V in self.tucker_variations])
t3toolbox/frame_variations_format.py:942: tt_stack_shapes = tuple([H.shape[:-3] for H in self.tt_variations])
t3toolbox/backend/utv_operations.py:224: ddots = sharing_module.fv_tied_ambient_directions((tkv, ttv), shared_data)
t3toolbox/backend/utv_operations.py:241: tkv_slices.append(ddots[gi])                             # ONE array per group, (K+C)+(nD, N)
t3toolbox/backend/utv_operations.py:294: H = ttv[-1:]
t3toolbox/backend/utv_operations.py:298: H = H + Lf_boundary
t3toolbox/backend/utv_operations.py:299: Gtop = xnp.concatenate([R[-1:], H], axis=-3)                           # [R, H] -> (rR+rL, nD, rR)
t3toolbox/backend/utv_operations.py:485: L, R, H_in = core                                                     # slices: L,R (stack C), H (stack K+C)
t3toolbox/backend/utv_operations.py:486: H = H_in + xnp.einsum('...jk,...kbl->...jbl', X_prev, R)              # incoming correction X_prev @ R_ii
t3toolbox/backend/utv_operations.py:487: Xi = xnp.einsum('...iaj,...iak->...jk', L, H)                          # (L^L)^T H^L, stack+(rL, rR)
t3toolbox/backend/utv_operations.py:488: proj = H - xnp.einsum('...iaj,...jk->...iak', L, Xi)
t3toolbox/backend/utv_operations.py:489: return Xi, (proj, H)
t3toolbox/backend/utv_operations.py:523: _, (proj_stack, H_stack) = xscan(_utv_oblique_gauge_step, xnp.zeros(ss + (rL, rR)), (left_sc, right_sc, ttv))
t3toolbox/backend/utv_operations.py:524: new_ttv = xnp.concatenate([proj_stack[:-1], H_stack[-1:]], axis=0)         # last core: un-projected
~~~
