# R4b handoff: the sampling kernels' prose and the shift symbols (store rename, 2026.5.0)

Implementer's record for the reviewer and the session. Base: HEAD `ebf563a9` (plan draft 26; R4a = `e3085085`).
Nothing is committed. Scope (plan §7 row R4b, the brief): every tier-D/E row, the shift and derivative notation
(§3.1–3.2), the `-ward` travel words (§3.3) and every §4A block in R4b's eleven files — `probing.py`,
`sampling_derivatives.py`, `apply.py`, `entries.py`, `utv_sampling.py`, `ut3_sampling.py` (`t3toolbox/backend/`),
`tests/_jet_trs_twins.py`, `tests/test_jet_recurrence.py`, `tests/backend/_curvature_oracles.py`,
`tests/backend/test_curvature.py`, `tests/backend/test_uniform_curvature.py` — plus the realignment of every block the
rename touched in them (plan §4 item 10, "within reason"). No other library or test file is touched.

**Revision 1** (the reviewer's first pass, REVISE, with the session's rulings) is §0. §1–§7 are the first pass's record; §5, §6, §7
and Appendix B are amended where Revision 1 changed them.

**Result:** 11 files. The first pass was prose only; Revision 1 adds the session's NAME rows (K15: **36** NAME hunks
against HEAD, all listed, Appendix B), and every contraction string is byte-identical (K14). In the code tree **K1, K2, K4, K5, K6, K7, K8, K9 report 0** (every remaining hit of those
patterns is in the docs and notes, R5's); K13 **PASS** (the eight fenced `probing.py` lines R4a reported now pass);
K14, K15, K15-cross, K17 (+ the three shape checks) **PASS**; K10/K10b/K11 **PASS** and **K12 complete for the code
tree (54 of 54 blocks filled)**; battery **bitwise**; 297 docstring examples in the changed files, **0 expected
outputs changed** (one example's trailing comment changed); suite, doctests, `sphinx -W` and the floor env: §7.

New files: this handoff; `R4b_name_hunks.tsv` (K15's listed hunks, 36 after Revision 1); `R4b_realignment.diff` (the
first pass's realignment, 25 whitespace-only hunks). Changed: `diagrams_inventory.md` (R4b's seven entries filled),
`dev/store_rename_check.py` (one K8 narrowing, §6), and in Revision 1 `dev/store_rename_keeps.txt` (one row KEEP →
EXEMPT) and `R1_s33_ward_sites.tsv` (one row added), §0.

## 0. Revision 1 (2026-11-13): the reviewer's first pass (REVISE) and the session's rulings

The reviewer's findings and the session's rulings are in `R4b_review.md`. Each item below is a listed row; no other
change was made.

**The reviewer's verified rulings:**

1. **R1** (the 104 `P`/`Q` → `L`/`R` prose lines, Appendix C) stand. No change. The session may register them as §3.4
   text rows.
2. **R2, item 12's generic `n`:** `sampling_derivatives.py:473, 476` (`compute_eta_jets`, "either"), the carry of
   `_eta_jets_order_step` (`:434`) and the return of `_eta_jets_core` (`:450`); `apply._t3_apply_step` (`:39`) and
   `entries._t3_entries_step` (`:44`) now read `C+(ni,Ni) ; C+(rLi,ni,rL(i+1))`. `probing.compute_eta` keeps `nU`.
3. **R3, NAME rows `D → B`** (a T3's own Tucker core): `probing._assemble_z_step` and `sampling_derivatives._z_jets_step`,
   2 tokens each, with their tuple comments `(eta, B)` / `(eta_jet, B)`.
4. **R7, NAME rows `dG → H`** (the forward kernels' TT variation, `(rL, nD, rR)` by `var_tt_cores`), plus `dGxi → Hxi`:
   `probing._sigma_step`, `_sigma_sweep_step`, `_deta_step`; `apply._apply_from_xis_step`;
   `sampling_derivatives._sigma_banded_step`, `_sigma_jets_step`, `_deta_jets_order_step`, `_deta_jets_core`; the twins
   `_sigma_jet_step`, `_sigma_sweep_trs`, `compute_deta_jets_trs`. `H` and `Hxi` were checked free in each function
   (ast) before the edit. The prose follows the code:
   - the tuple comments `(R, U, H, ...)` / `(L, R, H, ...)` (`probing.py:392, 474`; `sampling_derivatives.py:723, 782,
     803, 1038`; `apply.py:436`);
   - the labels `# H.  len=d` and `var_tt_cores     H.  len=d` (`sampling_derivatives.py:733, 763, 829, 900, 933, 1052,
     1077, 1107, 1161, 1191`); `H.` takes two spaces so `len=d` stays in its column, as for `R.` and `V.`;
   - the formulas and words: `mu H xi`, `sigma R nu + mu H nu + mu L tau`, `H_i(xi_i)`, "the variation core H", "the
     variation cores V, H", "(sigma / H / tau)", "U and H reversed", `# mu H` (`probing.py:380, 480, 1504, 1578`;
     `sampling_derivatives.py:672, 697, 699, 743, 771, 785, 794, 839, 843`; `_jet_trs_twins.py:79, 99, 112`).

   `dG` stays on the corewise Hessian's TT block and the raw directions of `G`, as the ruling lists. The only `dG` left
   in the prose of these files is `sampling_derivatives.py:3212` (`_apply_tangent_assemble_step`, corewise).
5. **MINOR 1:** `probing._assemble_tangent_z_step` now reads `# (D, V, eta, deta)`, "the C-only frame core D", "V is
   the variation core K+C" and "the C-only core D", as its twin `_tangent_z_jets_step` does.
6. **N1:** in `diagrams_inventory.md`, A-0465's roles line now reads "→ V-tilde_k over (nU, N), dG-tilde_k → H-tilde_k
   over (rL, nD, rR)".

**The session's rulings (relayed by the reviewer):**

7. **`rsh → rsig_hat`** in `_curvature_oracles.sigma_hat_tangent_jets_trs` (`:483, 489, 490`): a NAME row of 3 tokens,
   with `rsig_hat` checked free. The three `contract` lines keep their argument columns: `cd,` is padded to the width
   of `rsig_hat[k],`.
8. **`entries.py:344`** "Up-index edge variables …" now reads "Upward edge variables …". Registration:
   - `dev/store_rename_keeps.txt`: row `entries.py 344` changes from KEEP to EXEMPT, source "inv2 §9 doc UNRELATED;
     §3.3 -ward rewording (session, R4b S1)". This is the kind its siblings 124, 151 and 314 carry.
   - `R1_s33_ward_sites.tsv`: one row added (`Up-index`, source "session (R4b S1; the case-sensitive grep missed it)").

   K9, K9-converse and K9-registered pass.
9. **`test_curvature.py:10`** writes the lift in words (§3.1's template) beside the left store `L`:
   "``tv_to_dense(t3_corewise_frame(D, L), theta_hat) == tv_to_dense(frame, v)`` for ``theta_hat``, the lift of
   ``v``, at ``eps = 0``, …". The bullet re-wraps from two lines to three. Line 12 (`<L v, y> == <v, L^T y>`) stays
   as the session allowed: it has no store letter, and its bullet title "T3 lift adjoint" names the lift.
10. **`apply.py:168–169`** now read `xi-hat_i  = D_i w_i` and `delta-xi_i = V_i w_i`, the code's layout
    (`contract('Cio,Wo->WCi', D, w)`). The comment column is unchanged.
11. **A test-only change outside the token map:** `tests/test_jet_recurrence.py:32` changes `TOL = 1e-14` to `1e-13`,
    and the trailing comment is kept. In the reviewer's floor-env run (py3.9, numpy 1.22, jax 0.4.30, threaded
    OpenBLAS), `'apply (Euclidean)'` core 0 had a relative error of 1.22e-14, above 1e-14, with the three touched test
    modules run together. It passes single-threaded, and R4b does not change the numbers (the battery is bitwise). The
    arc's standing rule says a 1e-14 cross-path tolerance is suspect; N5/N6 saw the same on CI's BLAS.

**Realignment** (whitespace only, on the renamed lines):
- `sampling_derivatives.py:706`: `Hxi`'s trailing comment is back in its column (+2).
- `:712`: t2's `# K on core` (+1).
- `:794`: `# mu H     (K on core)`. The `#` column is +1, and the `(` column of the three labels is kept.
- `_jet_trs_twins.compute_deta_jets_trs`: the continuation `sigma_jets, tau_jets):` sits back under `zip(`'s first
  argument (−1).
- `_curvature_oracles`: the `cd,` padding (item 7).

Three changed lines are over 120 characters (`sampling_derivatives.py:434, 723, 1038`). Each is shorter than it was.

**Checks run** (light, one at a time):
- Gates (§7's table): K0, K2, K3, K6, K8b, K9 (+ converse, registered), K10/K10b, K11, K13, K14, K15 (36 of 36 listed), K15-cross
  and K17 (+ shape) PASS. Every K1/K4/K5/K7/K8 hit is outside the code tree (the docs and `CLAUDE.md`, R5's), and the
  counts are the first pass's (2/16/7/49/395).
- `store_rename_diagram_check.py all` PASS.
- `py_compile` on the eleven files.
- Static doctest comparison: 297 examples, 0 expected outputs changed, 1 source changed (the first pass's trailing
  comment).
- Sanity runs:
  - `tests/test_jet_recurrence.py`: 5 passed, 372 subtests;
  - `TestDerivativeKindsTwoHole::test_production_equals_the_trs_twins`, which runs `rsig_hat`: 1 passed, 180
    subtests;
  - `--doctest-modules` on probing, sampling_derivatives, apply and entries: 10 passed.

Not re-run, since the reviewer re-runs them on the final tree: the battery, the full suite, CI's doctests, `sphinx -W`
and the floor env.

**Added after the first report of this revision (the reviewer, under R7):** `tests/test_jet_recurrence.py:173, 179`
(`test_uniform_equals_ragged`) is a NAME row `dG → H`, 2 tokens. This `dG` is `core(*K)`, the K-stacked TT variation
passed in `compute_deta_jets`' `var_tt_cores` slot, and `_problem` in the same file already calls that object `H`. `H`
is free in the function; `R` there is the RNG lambda, and `Uc`/`Lc`/`Rc` stay. No column moves.

## 1. What changed, by file

- `probing.py`: the frame-letter tuple comments of every kernel (`(D, w)`, `(L, xi)`, `(R, U, dG, ...)`, `(L, R, dG,
  ...)`, `(R, xi)`, `(D, zt)`, `(L, xi, deta_tilde, mu)`, `(U, mu, nu, st, tt)`, `(D, Ddot, z~)`, `(L, L_dot, ...)`),
  the signature labels (`D. U. L. R.`, `L_dot.`, `R_dot.`, `U_dot.`, `Ddot.`), the four §4A blocks ×2 (P-0640/0646/0651,
  P-1215/1221/1226), `(D, U, L, R)` ×4, the module comment on apply/entries (`via L`, `via R`, "upward xis"), "frame upward
  edge vars", the transpose labels `V_tilde`/`V_tildes`/`H_tilde`/`H_tildes` (code names since R2), `delta_U_tilde` /
  `delta_G_tilde` → `V_tilde` / `H_tilde` (§5 item 6), the corewise substitution sentence of
  `t3_probe_corewise_transpose`, the corewise two-hole header (`(B, G, G, G)` of `theta = (B, G)`, `U = L = R = G` along
  `(Bdot, Gdot)`, "adjoint-upward lift `deta~dot = Bdot z~`"), the dotted-sweep docstrings (`tau~ L xi + mu L deta~`,
  `_sigma_step(..., L, L, L_dot, ...)`, `tau~ U nu + mu U sigma~`, `compute_deta(U_dot, U, U, ...)`),
  `t3_probe_corewise_hessian_from_sweep`'s `(D, U, L, R)` / `core_pair = (D, L)` and its numbered comments, and item 12's
  generic `n` on `assemble_tucker_tangent_variations` / `assemble_tt_tangent_variations` and their map bodies (§5 item 2).
- `sampling_derivatives.py`: the shift notation everywhere (§3.2): `_shift_down`/`_shift_up` signatures and docstrings,
  the jet-kernel note, `_inverse_factorials`, the eta/deta/adjoint-source/TT-assembly scans (`sh^r nu`, `(shᵀ)^r deta~`,
  `[(shᵀ)^u sigma~]`, `[(shᵀ)^k sigma~]`), the two dxi~ / dxi^ correlations (sample edit 7, with `shᵀ`); every frame
  letter in prose (`O→U`, frame `U→D`, `P/Q→L/R`, `O_dot/P_dot/Q_dot → U_dot/L_dot/R_dot`); the `-ward` words of R1's list
  (`upward`, `adjoint-upward`, `adjoint-var-downward`, `adjoint-downward`, `frame downward jets`); the transpose labels
  (`V_tilde(s)`, `H_tilde(s)`, `(V_tildes, H_tildes)`, `for V`); the corewise sentences (§1.2a: `(B, G, G, G)`,
  `theta = (B, G)`, `U = L = R = G`, `R = U = G`, `(Bdot, Gdot)`, `Bdot (w, p)`); the deta~-tangent kernel's `D`/`Ddot`
  (sample edit 3(a)); `(D, U, L, R)` ×14 and the tuple listings ×3; item 12's generic `n` (§5 item 2); the chunk
  estimators' "up-core mode" (§5 item 9).
- `apply.py`: tuple labels and `(down, up, left, right)` ×5; "via L / via R"; `D_i^T w_i`, `V_i^T w_i`; `downward
  ``eta```; `(V_tildes, H_tildes)` ×2; "the frame cores ``U, R``"; the corewise substitution sentence; `(D, U, L, R)` /
  `(D, L, L, L)` / `theta = (D, L)`; `Bdot w`; `(B, G, G, G)` ×2 and `(Bdot, Gdot)`; `(R, U, dG, ...)`; A-0465; the
  Hessian block's `dB = sum_W w (x) dxi_hat_dot`.
- `entries.py`: `(down, up, left, right)` ×3, "upward" ×3 (R1's list), `D_i` / `V_i` fibers, `V-tilde`, `(V_tildes,
  H_tildes)` ×2, tuple labels, `(D, U, L, R); uses R (right) and U (up)`, `uses D (one-hot), U, R`, `theta = (D, L)`,
  `Bdot[..., index]`.
- `utv_sampling.py`: `(down, up, left, right, shape, masks)`, `(V_tilde, H_tilde)` ×7 and `(V_tilde, H_tilde, shape,
  masks)` ×7, `(down, up, frame_left[:-1], frame_right[1:])`.
- `ut3_sampling.py`: the two corewise comment blocks (`(U,L,R)->G`, frame `(B, G, G, G)`, raw `(dB, dG)`), "the upward
  jet", `(D, U, L, R)` of `theta = (D, L)` ×2, "frame's ``D`` and ``L``", `(down, left)` masks ×6.
- `tests/_jet_trs_twins.py`: `R_i`/`U_i`/`L` in the sigma/deta/adjoint docstrings, "the up frame U", "Adjoint-var-downward",
  `tau~ U nu + mu U sigma~`, "via R", the apply-transpose twin's `mu U sigma^`, `H = ...`, `V = ...`, "Returns ``(V, H)``".
- `tests/test_jet_recurrence.py`: B7's label `'apply transpose V (...)'` and its `'apply transpose H (...)'` (§5 item 7).
- `tests/backend/_curvature_oracles.py`: `dF`/`dPhi` (§5 item 8), `B_i` in `blocksum`, the block-position comments
  (`[[R ; Gdot], [0 ; U]]`, `U (bottom)`), `theta = (B, G)` ×3, `(B, G, G, G)` ×2, `Bdot_g`, the gauge docstring
  (`B_i -> A B_i`, `G_i -> G_i M`), `prop(tilde, L, xi)`, `Bdot_i z~_i`, `U = G` ×2, `U = L = R = G` along `(Bd, Gd)`,
  `R = U = G`, `mu U sigma^`, `dB = dxi^_dot (x) w`.
- `tests/backend/test_curvature.py`: `V_i D_i^T` ×2, `t3_corewise_frame(D, L)`, `theta = (D, L)` / base point `(D, L)`
  ×7, `(D, U, L, R)` ×3, `zB_i`, `dPhi` ×2, and the store words of `_TUCKER_SLACK` and the shared-structure comments
  (old `down` = the TT store → `up`, §5 item 9).
- `tests/backend/test_uniform_curvature.py`: base point `(D, L)` with `Bdot (nD, N)` ×2, the frame's `(down, left)` masks
  ×4, `(B, G, G, G)` ×5, `dPhi` ×2.

## 2. How the rows were worked

File by file, the tag's text of every comment and docstring line holding a store letter, store word, width,
shift/derivative form or `U`-family name was scanned (a tokenize-based scanner over COMMENT and STRING tokens, 999
lines), read against inv 2 §9's rows, inv 1's rows, the pqu text rows, R1's `-ward` list, §1.2a's lists, tier D's
patterns and the K-check hits, and each line was edited by hand with an all-or-nothing exact-replace helper. Every
sentence was written from the new convention, not swapped word by word; the rule applied per object:

- a frame store (by its A1 keyword, or the kernel's code local that R2 renamed by tier C) → `D`, `U`, `L`, `R`;
- the shift (§3.2) → `sh`, `shᵀ`, `sh^r`, `(shᵀ)^r`; the functions `_shift_*` keep their names;
- a variation-shaped array (a transpose output, a tangent's variation) → `V`/`H` forms;
- a point's own Tucker core or its direction at the corewise frame → `B` forms; the deta~-tangent kernels keep `D`/`Ddot`
  by their A1 keywords (§1.2a "kernels lettered by their keywords"), their oracle twin `Bdot` (by design);
- the corewise Hessian blocks keep `dB`/`dG` (core-shaped, §3.1).

After the pass, a residual scan (the same scanner) was read line by line; what it found is in §5 and §6.

## 3. The §4A blocks

### 3.1 Status

Seven inventoried blocks are R4b's, all fenced, all rewritten whole by hand and re-laid out (K10b): A-0465 (`apply.py`),
P-0640, P-0646, P-0651 (`probing.py`, `tv_probe`'s signature) and P-1215, P-1221, P-1226 (`tv_probe_transpose`'s, the same
text). Each inventory entry carries its new span, its verbatim new text and the re-layout reason; K12 checks them
verbatim. With them, every code-tree block of the inventory is filled (54/54); the seven pending are the docs' (R5).
Before/after of every block: Appendix A.

### 3.2 Derivations, index by index (code layout: `D_i` is `(nD_i, N_i)`, `U_i` is `(rL_i, nU_i, rR_(i+1))`, `L_i` is
`(rL_i, nD_i, rL_(i+1))`, `R_i` is `(rR_i, nD_i, rR_(i+1))`, `V_i` is `(nU_i, N_i)`, `H_i` is `(rL_i, nD_i, rR_(i+1))`)

- **A-0465** (`_apply_transpose_adjoint`'s docstring, the adjoint-state assembly; the code is the ragged loop below it):
  - `dxi-hat_k = mu-hat_{k-1} . U_k . sigma-hat_k  # over the up mode nU`: `mu-hat` is `W+C+(rL_k,)`, `U_k` is
    `(rL_k, nU_k, rR_(k+1))`, `sigma-hat_k` is `W+K+C+(rR_(k+1),)`; the two bonds are summed, so the result runs over the up
    core's middle leg `nU_k` — the code's `contract('WCa,Caib,WKCb->WKCi', mu, U, sig_hat)` (index `i` = `U`'s middle).
    Old `O_k`/`nO` was the same core.
  - `H-tilde_k = mu-hat_{k-1} (x) xi-hat_k (x) sigma-hat_k  # over (rL, nD, rR)`: `xi-hat_k = D_k w_k` is `(nD_k,)`, so the
    outer product is `(rL_k, nD_k, rR_(k+1))`, a TT variation's shape — the code's `'WCa,WCi,WKCb->KCaib'` into `H_tildes`.
    Old `(rL, nU, rR)` was the old Tucker width.
  - `V-tilde_k = dxi-hat_k (x) w_k  # over (nU, N)`: `(nU_k,) (x) (N_k,)` = a Tucker variation's shape, the code's
    `'Wo,WKCa->KCao'` into `V_tildes`. Sizes agree across the three lines: `nU` enters through `U_k` and leaves through
    `V-tilde_k`; `nD` enters through `xi-hat_k` and leaves through `H-tilde_k`.
  - Letters: `H-tilde_k` / `V-tilde_k` (the block's own `-hat_k` / `-tilde_k` style) in place of §1.2a's sketch `H~_k` /
    `V~_k`; they have the length of `dxi-hat_k`, so the `=` column (19) is the tag's (§5 item 5).
- **P-0640 / P-1215** (the ragged frame's four tuple-element comments):
  - `D_xo D_yo = I_xy`, `D.shape = (nD, N)`: sum over `o = N`, free `x, y` over `nD` — rows orthonormal, down-orthogonal
    (the code's `einsum('...io,...jo->...ij', D, D) = I`).
  - `U_ixj U_iyj = I_xy`, `U.shape = (rL, nU, rR)`: sum over the left bond `i` and the right bond `j`, free `x, y` over the
    middle leg `nU` — up-orthogonal (`einsum('...iaj,...ibj->...ab', U, U) = I`).
  - `L_iax L_iay = I_xy`, `L.shape = (rL, nD, rL)`: sum over the left bond `i` and the middle leg `a` (`nD`), free `x, y`
    over the right bond `rL_(i+1)` — left-orthogonal.
  - `R_xaj R_yaj = I_xy`, `R.shape = (rR, nD, rR)`: sum over the middle leg `a` and the right bond `j`, free `x, y` over the
    left bond `rR_i` — right-orthogonal.
  - The tag's `P.shape = (rL, nU, rR)` / `Q.shape = (rL, nU, rR)` wrote generic bonds; with `rL`/`rR` the left/right
    store widths (D-3), the left core's bonds are both `rL` and the right core's both `rR`, as FVF-0156 and the kernels'
    own shape comments (`compute_mu`'s `C+(rLi,nDi,rL(i+1))`, `compute_nu`'s `C+(rRi,nDi,rR(i+1))`) write them (§5 item 5).
- **P-0646 / P-1221** (the uniform supercores): `down_tucker_supercore (d, nD, N)` down-orthogonal elements; `up_tt_supercore
  (d, rL, nU, rR)` up-orthogonal; `left_tt_supercore (d, rL, nD, rL)` left; `right_tt_supercore (d, rR, nD, rR)` right —
  UFVF-0110's shapes. The tag's backwards adjectives (§4.3, `probing.py:646–647, 1221–1222`: "up orthogonal" on the
  Tucker supercore) now agree with their store words.
- **P-0651 / P-1226**: `T3Frame.data = (down, up, left, right) = (D, U, L, R)` — the physical order read in the new
  words (tier D's pattern, applied by hand inside the fence).

## 4. Realignment (plan §4 item 10; "within reason", Nick 2026-11-12)

R4a's tool (the tag's alignment column of every run of consecutive lines restored at `max(tag column, longest field end
+ min(2, smallest gap))`, code lines, comments and doctest source only, never fenced spans) was run over R4b's eleven
files and every hunk read by hand: 22 hunks, then three hand fixes and one hand revert. `R4b_realignment.diff` holds the
25 whitespace-only hunks (32 lines; `split()`-equal, same indentation); the token stream is unchanged (K15).

- **Signature type columns** restored where R2's `up_tucker_cores → down_tucker_cores` (+2) and `down_tt_cores →
  up_tt_cores` (−2) broke them: `probing.py` `compute_xi`, `compute_sigma`, `compute_tau`, `compute_deta_tilde`,
  `compute_deta_tilde_tangent`, `compute_dxi_tilde_tangent`; `sampling_derivatives.py` `compute_sigma_jets`,
  `compute_tau_jets`, `_apply_derivatives_from_jets`, `compute_deta_tilde_jets`, `compute_dxi_tilde_jets`,
  `_apply_derivatives_transpose_from_jets`, `_entries_ambient_jets`, `compute_deta_tilde_tangent_jets`,
  `compute_dxi_tilde_tangent_jets`; `utv_sampling.py` `_remasked_transpose_output` (R2's `dU_tilde → V_tilde`, −1).
- **Trailing-comment columns**: `apply.py:168–170` and `entries.py:135–136` (the `#` column and the inner `(fiber slice)`
  column), the chunk estimators' comment columns (`_jet_floor_bytes` 53/53/60×4, the tag's).
- **Hand fixes** (the tool cannot see a column only one line had at the tag):
  1. `probing.py` `compute_eta` (`:284`) and `compute_dxi_tilde` (`:976`): the up-core parameter line's type column at 32
     with the others (it was already one column off at the tag and three after R2), and `compute_dxi_tilde`'s comment
     gap two spaces like its neighbours.
  2. `sampling_derivatives.py` `_two_hole_jet_floor_bytes` (`:1891`): the first `per_core_row` line's comment back to the
     tag's column 70 (R2's `nU → n` ×2 had pulled it two left; the block is a deliberate staircase at the tag, 70/72/78/72,
     which is kept).
  3. **Reverted** the tool's move of `ww`/`pp`'s `len=d` in `_assemble_tucker_variation_jets_dense` and
     `assemble_tucker_variation_jets` (it aligned them toward the renamed `adjoint-var-downward` line): those two lines
     are unchanged since the tag, whose layout aligns only them with each other.
- **Inside content edits**: the four-line tuple-label blocks (`down_tucker_cores  D.`, `up_tt_cores        U.`, …) re-padded
  to one letter column (`apply.py` ×4, `entries.py`, `probing.py`, `sampling_derivatives.py` ×3); `_adj_tilde_tangent_step`'s
  argument comments (`sampling_derivatives.py:2565–2572`) moved +3 so "adjoint-upward jet," fits the shape column, the
  `C+(` / `K+C+(` right-alignment kept; A-0465's third comment column 69 → 70 (a one-column slip at the tag, fixed inside
  the re-laid-out block).
- **Checked**: an adjacent-pair scan (every pair of lines that shared a trailing-`#` or a `name:  type` column at the tag
  must share it in the tree) finds **0** ragged pairs in R4b's files (52 on HEAD, so it sees the drift). Five pre-existing
  runs stay ragged by design (`sum_over_probes:` / `n_probe:` / `chunk_size:` lines longer than their blocks'
  `name:` column at the tag, untouched by the rename).
- **Line length**: no line newly over 120 except `probing.py:1566` (121, the tag's own length restored by the type-column
  fix). One comment the shift notation pushed to 124 (`sampling_derivatives.py:2512`) was shortened to 120 ("the cores
  ride unchanged" → "cores ride unchanged"). Pre-existing long lines that the letters lengthened stay (e.g. the
  `_dxi_tilde_order_step` `xs` comment, 197, as sample edit 7's approved AFTER writes it).

## 5. Judgement calls

1. **Frame `P`/`Q` → `L`/`R` in prose without a pqu or inv row** (104 lines; Appendix C). The §3.4 text scan caught only
   `P_i`/`Q_i`, tuples and backticked forms, so the bare `P`/`Q` of the kernels' tuple comments (`(P, xi)`, `# Q.  len=d`,
   `(L, L_dot, ...)`), docstring formulas (`tau~ P xi`, `sigma Q nu`, `d(P xi)`) and "via P / via Q" carry no row.
   Each names the left/right frame core the same function's code already calls `L`/`R` (tier C's `P→L`, `Q→R`,
   `P_dot→L_dot`, `Q_dot→R_dot`, `Pxi→Lxi` in these files), and the brief lists `P`/`Q`→`L`/`R` among R4b's prose rows.
   Applied, and listed for the session to register as rows (as R4a's MINOR 1 was). Kept: the perturbation `P` /
   ``pp`` (every "perturbation vectors P", "contracting ``P`` (order 1)"), the projector `P_X`.
2. **Item 12's generic `n`** (§4 item 12; K17-shape-point forces two of them):
   - applied where listed: `sampling_derivatives.py` `t3_probe_derivatives`, `t3_apply_derivatives`,
     `t3_entries_derivatives` (`x`), `compute_mu_jets` / `compute_nu_jets` (`tt_cores` and `xi_jets`; frame left/right
     stores in the sweeps, a T3's own `tt_cores` in `t3_probe_derivatives`: "either" → `n`), `assemble_z_jets` (only a
     T3's own cores), `_apply_derivatives_t3_from_xi_jets` (`:995`, its two library callers pass a T3's own cores),
     `_apply_derivatives_hessian_from_jets` (`:3230–3240`, as the plan writes it) and its map body
     `_apply_tangent_assemble_step` (`(dB, dG): K+C+(ni,Ni) ; K+C+(rLi,ni,rR(i+1))`);
   - applied to the probe two-hole's assemblies, not on the list: `probing.py` `assemble_tucker_tangent_variations` /
     `assemble_tt_tangent_variations` and their steps (K17-shape-point FAILed on them: `dB`/`dG` returned under `(nUi,Ni)`
     / `(rLi,nDi,rR(i+1))`), and for one rule across the twins `sampling_derivatives.py`
     `assemble_tucker_tangent_variation_jets` / `assemble_tt_tangent_variation_jets` and `_tucker_tangent_variation_jets_step`
     (the corewise Hessian's blocks, called only at the corewise frame; inputs of the same size take `n` too);
   - kept a store letter, by their call sites: `probing.py:550` (`assemble_tangent_z`'s `tucker_cores`, `nD`: both callers
     pass `down_tucker_cores`), `sampling_derivatives.py:872` (`assemble_tangent_z_jets`, likewise);
   - Revision 1 (the session's ruling R2): `:473` (`compute_eta_jets`, "either": `t3_probe_derivatives:200` passes a T3's
     own `tt_cores`, `_probe_frame_sweep_jets_from_xi:597` the frame's `up_tt_cores`) takes `n` at `:473, 476` and in its
     map bodies (`_eta_jets_order_step`'s carry, `_eta_jets_core`'s return); the T3-only step comments
     `apply._t3_apply_step` (`:39`) and `entries._t3_entries_step` (`:44`) read `C+(ni,Ni) ; C+(rLi,ni,rL(i+1))`.
     `probing.compute_eta` keeps `nU` (its keyword is `up_tt_cores`).
3. **`B` for a T3's own Tucker cores in prose where the code local is `D`.** `t3_probe_derivatives`' `B_i x_i`, `B_i p_i`
   and `assemble_z_jets`' `z_i^(t) = B_i eta_i^(t)` (inv 2's rows say `D_i`; L2 and D-6 — "definitely should not be D" —
   say `B` for a T3's own core). Revision 1 (ruling R3): the code locals follow — `_z_jets_step` and probing's
   `_assemble_z_step` are NAME rows `D → B` (2 tokens each), their tuple comments `(eta_jet, B)` / `(eta, B)`.
4. **Corewise sentences**: `apply.py:320–324`, `probing.py:1426–1430`, `sampling_derivatives.py:2385–2389, 2409–2411` use
   §1.2a's template in the module's ASCII arrows (`B -> D`, `G -> U, L, R`; "the manifold operation requires an orthogonal
   frame, the corewise one works for any `B`, `G`"). Sentences naming only the TT slots (`ut3_sampling.py:111, 229`)
   write `(U,L,R)->G`, as R4a's `uniform_tucker_tensor_train.py`. The "blow-up" line (`probing.py:1430`, the tag's `:1429`, a K9
   keep) is kept byte-identical by ending the line before it short.
5. **Block letters and shapes**: A-0465 writes `H-tilde_k` / `V-tilde_k` (the block's style, the length of `dxi-hat_k`)
   for §1.2a's `H~_k` / `V~_k` sketch; P-0640/P-1215 correct the left/right cores' bond letters to `(rL, nD, rL)` /
   `(rR, nD, rR)` (§3.2); A-0465's third `#` moves from column 69 to 70.
6. **`delta_U_tilde` / `delta_G_tilde` → `V_tilde` / `H_tilde`** (`probing.py` `assemble_tucker_variations` /
   `assemble_tt_variations` docstrings): T4S's `δŨ`, `δG̃` spelled out, the transpose outputs, which the shape rule makes
   `V`/`H` (as §4.8 converts the notes' `\delta\widetilde{\mathbf U}_i`); no gate sees the spelled-out form.
7. **`test_jet_recurrence.py:154`**: B7 renames the label `'apply transpose dU (...)'` to `V`; its twin label
   `'apply transpose dG (...)'` (the dict key `'dG'` that B7 made `'H'`) becomes `'apply transpose H (...)'` with it. A
   string label only, no NAME.
8. **`DF(theta)[theta_hat]` → `dF(...)`** (`_curvature_oracles.py:10`), beside tier D's listed `dPhi` on the next line: the
   same derivative `D` (D-11), not in tier D's enumerated 15 lines.
9. **Store words read in the new meaning** (K9 clause (b) — each is the opposite of its aligned old word):
   `test_curvature.py:114` "frame up ranks (2, 4, 2) < down (4, 4, 4)" (at the Tucker-slack structure the TT-shaped store
   is the one at `(2, 4, 2)`); `:2800, 2804–2806` "(its up rank)", "up rank 2 < ..." (a member's `S_i^T` is `(nU_i, nD_i)`,
   wide where `nU_i` is below the group rank); `_jet_trs_twins.py:79` "the up frame U"; `sampling_derivatives.py:1820,
   1889` "# up-core mode ~ the Tucker rank" (`nU = n`, R2's A3 exception); the `(down, left)` masks of the frame's base
   point (`ut3_sampling.py` ×6, `test_uniform_curvature.py` ×4).
10. **`_curvature_oracles.py` App. B lines** (inv 1 keeps whose letters the plan's overrides reach, §1.5 item 9):
    `blocksum`'s `# B_i sits in the top Tucker rows` (its code is `B` since R2) and the doubled-core slot comments
    `[[R ; Gdot], [0 ; U]]`, `U (bottom)` (old `O` = the up slot of `tv_to_t3`'s last/middle cores, `G_bot = [Z, U]`);
    `vertical_direction`'s docstring `B_i -> A B_i` beside tier C's `G_i -> G_i M`.
11. **Shift wording kept**: "up-shift", "up-shifted" (`sampling_derivatives.py:1230, 1288, 1550, 2517`) are K9
    keeps and name the reverse shift `shᵀ` in words. "Up-index edge variables" (`entries.py:344`, `_entry_xis`), a K9
    keep the case-sensitive `-ward` grep missed, is "Upward edge variables" since Revision 1 (session ruling S1, §0).
12. **No operator variable**: no code variable holds `sh`/`shᵀ` (the tree calls `_shift_down`/`_shift_up` directly), so no
    `shift`/`shift_t` NAME is introduced; the σ̂ locals were already `sig_hat`/`sig_hat_d` since R2; `_flatten_w`'s `sh`,
    `psh`/`pshd`, `sh_pair_j` and the doctests' `shift = lambda` stay (§1.5 item 14).

## 6. Not done, reported, and tool changes

**Reported, unchanged (NAMEs not on any row):**

- `Ud` in `sampling_derivatives.py` (`_dxi_tilde_tangent_order_step`, `_dxi_tilde_tangent_jets_core`,
  `_dxi_hat_tangent_order_step`, `_dxi_hat_tangent_jets_core`): tier C's image of `Od` (the up slot's direction), which
  K8's `Ud` pattern flags (11 lines). Kept by the map; K8 narrowed (below).
- The first pass also reported the forward kernels' `dG`, the T3-own `D`, `rsh`, `entries.py:344`, item 12 at `:473`
  and the two T3-only step comments; the session ruled on all six and Revision 1 applied them (§0).
- `tests/test_jet_recurrence.py:173, 179` (a test local `dG`, the TT variation passed to `compute_deta_jets`) was
  reported in Revision 1. The reviewer ruled it under R7, and it is now `H` (§0).

**Tool change (`dev/store_rename_check.py`, for the reviewer's eye):** K8's carve gains one clause — in
`t3toolbox/backend/sampling_derivatives.py`, a `Ud` hit on a code line (outside docstrings and comments, by the module's
`_code_prose`) is the NAME tier C maps from `Od` and is not a U-family leftover; K15 checks those NAMEs. Prose `Ud` in that
file still hits, and no other file is affected. Without it K8 FAILs on the 11 code lines above and nothing else in the
code tree.

**Plan text the writer may want** (not edited): §5 item 1's rows (Appendix C) and the App. B overrides of §5 item 10;
item 12's dispositions (§5 item 2), with `:473`'s tension; §1.2a's `V~_k`/`H~_k` sketch vs A-0465's `-tilde_k`; the K8
narrowing; §3.3's `-ward` list drops `manifold.py:949, 1172` and `test_contractions_interpreter.py:227, 335` (R4a's).

## 7. Gates, battery, suite, doctests, docs, floor env

All in the `t3toolbox` env unless stated; one heavy job at a time.

**Gates** (`python dev/store_rename_check.py --stage R4b --hits dev/store_rename_slices --hunks
dev/store_rename_slices/R4b_name_hunks.tsv --names dGxi,Hxi,rsh,rsig_hat`; `python dev/store_rename_diagram_check.py
all`). The table is Revision 1's run where it says so; the battery, suite, doctests, docs and floor env below are the
first pass's, and the reviewer re-runs them on the final tree:

| gate | result |
|---|---|
| K1, K2, K4, K5, K6, K7, K8, K9 in the code tree | **0** (whole tree: K1 2, K4 16, K5 7, K7 49, K8 395 — all in the docs and notes, R5's; K2, K6, K9 0) |
| K9 | **PASS** (0 residual lines, with `k9_registered.tsv`; no new registration needed); `K9-converse` **PASS**; `K9-registered` **PASS**. Revision 1: the same, after `entries.py:344`'s keep row changed from KEEP to EXEMPT (§0 item 8) |
| K0, K3, K8b | **PASS** |
| K10/K10b | **PASS**: 33 in place, 20 re-laid out (R4b's 7 included); 7 pending, all the docs' |
| K11 | **PASS**: 252 untouched fences byte-identical to `v2026.4.0`; 48 rewritten blocks excepted and checked by K12 |
| K12 | code tree **complete**: 54 of 54 filled and verbatim (PENDING overall: 247 docs/notes blocks, R5's) |
| K13 | **PASS** (0; R4a's eight fenced `probing.py` lines fixed by P-0640/0646/1215/1221); K13-generic INFO 2 after Revision 1 (`probing.py:550`, `sampling_derivatives.py:872`, dispositions §5 item 2; `:473` is `n` now) |
| K14 | **PASS**: 0 DNT tokens lost, 0 count differences, contract/einsum strings identical; Revision 1: 26 DNT `H` tokens added by the `dG → H` rows (counted, licensed by the tag comparison) |
| K15 | **PASS**: first pass 0 NAME hunks; Revision 1 36, all listed (Appendix B) |
| K15-cross | **PASS** (Revision 1: `--names dGxi,Hxi,rsh,rsig_hat`) |
| K17 (+ shape-frame, shape-point, shape-converse) | **PASS** (shape-point was FAIL 4 before item 12's `n`) |
| K16 | REPORT 20, R5's |

**Battery** (`run --out <scratch> --b9 dev/store_rename_slices/battery_ref/b9`, then `compare
dev/store_rename_slices/battery_ref <scratch>`): **PASS (bitwise)**, 2,190 arrays and 856 texts, `NEW=True`,
`LETTERS_NEW=True`, 0 errors.

**Doctest outputs** (static `DocTestParser` comparison, HEAD → tree, every changed `.py`): 297 examples, **0 expected
outputs changed**; 1 source changed (a trailing comment, `probing.py:1294`).

**Suite** (`PYTHONPATH=$PWD python -m pytest tests/ -q -n 3`, alone): **1,347 passed, 4 skipped, 48,724 subtests passed,
0 failed** (16 min 05 s; R4a's counts exactly). The one warning is the intended adam-on-manifold one. (A whitespace-only
docstring edit, A-0465's third `#` column, landed while it ran; the doctests, gates and the floor env below ran after it.)

**Doctests** (CI's three commands, one after another): `pytest --doctest-modules t3toolbox` **241 passed**;
`python -m doctest docs/getting_started.rst` **pass**; `python -m doctest` over every `docs/*.md` and
`docs/contributor/*.md` except `doctest_style.md` **pass**.

**Docs** (`python -m sphinx -W --keep-going -b html docs <scratch>`, alone): **build succeeded**, exit 0, no warning
(15 min 05 s).

**Floor env** (`tttt`: Python 3.9.19, numpy 1.22.4, jax 0.4.30; plain serial pytest, run alone): the touched test modules
`tests/test_jet_recurrence.py`, `tests/backend/test_curvature.py`, `tests/backend/test_uniform_curvature.py` (the two
helpers are imported by them): **159 passed, 2 skipped, 0 failed** (1 min 40 s); `pytest --doctest-modules t3toolbox` in
the floor env: **241 passed**.

## Appendix A. Every §4A block: before (v2026.4.0) and after (the tree)

Generated from the inventory entries (checked verbatim by K12). Derivations in §3.2.

### A-0465 — `t3toolbox/backend/apply.py` 465–467 → 465–467, K10b (re-laid out)

before:

~~~
        dxi-hat_k  = mu-hat_{k-1} . O_k . sigma-hat_k                 # over the down mode nO
        dG-tilde_k = mu-hat_{k-1} (x) xi-hat_k (x) sigma-hat_k        # over (rL, nU, rR)
        dU-tilde_k = dxi-hat_k (x) w_k                               # over (nO, N)
~~~

after:

~~~
        dxi-hat_k  = mu-hat_{k-1} . U_k . sigma-hat_k                 # over the up mode nU
        H-tilde_k  = mu-hat_{k-1} (x) xi-hat_k (x) sigma-hat_k        # over (rL, nD, rR)
        V-tilde_k  = dxi-hat_k (x) w_k                                # over (nU, N)
~~~

### P-0640 — `t3toolbox/backend/probing.py` 640–643, columns 40–end → 640–643, columns 40–end, K10b (re-laid out)

before:

~~~
# up_tucker_cores. len=d. U_xo U_yo   = I_xy, U.shape = (nU, N)
# down_tt_cores.   len=d. O_ixj O_iyj = I_xy  O.shape = (rL, nO, rR)
# left_tt_cores.   len=d. P_iax P_iay = I_xy, P.shape = (rL, nU, rR)
# right_tt_cores.  len=d. Q_xaj Q_yaj = I_xy  Q.shape = (rL, nU, rR)
~~~

after:

~~~
# down_tucker_cores. len=d. D_xo D_yo   = I_xy, D.shape = (nD, N)
# up_tt_cores.       len=d. U_ixj U_iyj = I_xy  U.shape = (rL, nU, rR)
# left_tt_cores.     len=d. L_iax L_iay = I_xy, L.shape = (rL, nD, rL)
# right_tt_cores.    len=d. R_xaj R_yaj = I_xy  R.shape = (rR, nD, rR)
~~~

### P-0646 — `t3toolbox/backend/probing.py` 646–649, columns 26–end → 646–649, columns 26–end, K10b (re-laid out)

before:

~~~
# up_tucker_supercore. shape=(d, nU, N),      up orthogonal elements
# down_tt_supercore.   shape=(d, rL, nO, rR), down orthogonal elements
# left_tt_supercore.   shape=(d, rL, nU, rR), left orthogonal elements
# right_tt_supercore.  shape=(d, rL, nU, rR), right orthogonal elements
~~~

after:

~~~
# down_tucker_supercore. shape=(d, nD, N),      down orthogonal elements
# up_tt_supercore.       shape=(d, rL, nU, rR), up orthogonal elements
# left_tt_supercore.     shape=(d, rL, nD, rL), left orthogonal elements
# right_tt_supercore.    shape=(d, rR, nD, rR), right orthogonal elements
~~~

### P-0651 — `t3toolbox/backend/probing.py` 651, columns 11–end → 651, columns 11–end, K10b (re-laid out)

before:

~~~
# frame order = T3Frame.data = (up, down, left, right) = (U, O, P, Q)
~~~

after:

~~~
# frame order = T3Frame.data = (down, up, left, right) = (D, U, L, R)
~~~

### P-1215 — `t3toolbox/backend/probing.py` 1215–1218, columns 40–end → 1215–1218, columns 40–end, K10b (re-laid out)

before:

~~~
# up_tucker_cores. len=d. U_xo U_yo   = I_xy, U.shape = (nU, N)
# down_tt_cores.   len=d. O_ixj O_iyj = I_xy  O.shape = (rL, nO, rR)
# left_tt_cores.   len=d. P_iax P_iay = I_xy, P.shape = (rL, nU, rR)
# right_tt_cores.  len=d. Q_xaj Q_yaj = I_xy  Q.shape = (rL, nU, rR)
~~~

after:

~~~
# down_tucker_cores. len=d. D_xo D_yo   = I_xy, D.shape = (nD, N)
# up_tt_cores.       len=d. U_ixj U_iyj = I_xy  U.shape = (rL, nU, rR)
# left_tt_cores.     len=d. L_iax L_iay = I_xy, L.shape = (rL, nD, rL)
# right_tt_cores.    len=d. R_xaj R_yaj = I_xy  R.shape = (rR, nD, rR)
~~~

### P-1221 — `t3toolbox/backend/probing.py` 1221–1224, columns 26–end → 1221–1224, columns 26–end, K10b (re-laid out)

before:

~~~
# up_tucker_supercore. shape=(d, nU, N),      up orthogonal elements
# down_tt_supercore.   shape=(d, rL, nO, rR), down orthogonal elements
# left_tt_supercore.   shape=(d, rL, nU, rR), left orthogonal elements
# right_tt_supercore.  shape=(d, rL, nU, rR), right orthogonal elements
~~~

after:

~~~
# down_tucker_supercore. shape=(d, nD, N),      down orthogonal elements
# up_tt_supercore.       shape=(d, rL, nU, rR), up orthogonal elements
# left_tt_supercore.     shape=(d, rL, nD, rL), left orthogonal elements
# right_tt_supercore.    shape=(d, rR, nD, rR), right orthogonal elements
~~~

### P-1226 — `t3toolbox/backend/probing.py` 1226, columns 11–end → 1226, columns 11–end, K10b (re-laid out)

before:

~~~
# frame order = T3Frame.data = (up, down, left, right) = (U, O, P, Q)
~~~

after:

~~~
# frame order = T3Frame.data = (down, up, left, right) = (D, U, L, R)
~~~

## Appendix B. The NAME hunks (K15; `R4b_name_hunks.tsv`)

The first pass had none (prose only). Revision 1 adds the session's NAME rows (§0); `R4b_name_hunks.tsv` holds
exactly these 36 hunks (written by `--write-hunks`, checked by `--hunks`: 0 unlisted, 0 missing). Lines are the
tree's; every hunk is a one-line `replace`.

| file | lines | old → new |
|---|---|---|
| `t3toolbox/backend/apply.py` | 440, 441 | `dG` → `H` |
| `t3toolbox/backend/probing.py` | 314, 315 | `D` → `B` |
| `t3toolbox/backend/probing.py` | 373, 385, 396, 397, 478, 483 | `dG` → `H` |
| `t3toolbox/backend/sampling_derivatives.py` | 513, 514 | `D` → `B` |
| `t3toolbox/backend/sampling_derivatives.py` | 695, 706, 728, 729, 788, 794, 798, 807, 811, 812 | `dG` → `H` |
| `t3toolbox/backend/sampling_derivatives.py` | 706, 712, 716 | `dGxi` → `Hxi` |
| `tests/_jet_trs_twins.py` | 77, 84, 92, 94, 114, 118 | `dG` → `H` |
| `tests/backend/_curvature_oracles.py` | 483, 489, 490 | `rsh` → `rsig_hat` |
| `tests/test_jet_recurrence.py` | 173, 179 | `dG` → `H` |

K15-cross lists the four names the hit table has no row for: `--names dGxi,Hxi,rsh,rsig_hat`. `D`/`B` and `dG`/`H`
are map sources and images already (tier C, R2).

## Appendix C. `P`/`Q` → `L`/`R` prose sites without a pqu or inventory row (§5 item 1)

Line numbers are HEAD's (= `v2026.4.0`'s for these files); each line's frame `P` → `L`, `Q` → `R` (and `P_dot` → `L_dot`,
`Q_dot` → `R_dot`, `Pxi` → `Lxi`):

- `t3toolbox/backend/probing.py`: 220, 380, 480, 771, 781, 782, 814, 823, 897, 903, 1499, 1502, 1514, 1515, 1522, 1524, 1544, 1545, 1552, 1577 (20)
- `t3toolbox/backend/sampling_derivatives.py`: 734, 764, 785, 791, 793, 795, 830, 831, 839, 840, 938, 939, 976, 1050, 1112, 1113, 1122, 1196, 1197, 1270, 1272, 1283, 1287, 1305, 1315, 1326, 1354, 1365, 2101, 2109, 2120, 2124, 2224, 2510, 2514, 2515, 2520, 2522, 2523, 2540, 2541, 2543, 2549, 2576, 2577, 2578, 2579, 2583, 2610, 2622, 2654, 2655, 2662, 2664, 2673, 2674, 2681, 3123, 3134, 3135, 3138, 3140, 3145 (63)
- `t3toolbox/backend/apply.py`: 150, 151, 156, 157, 170, 202, 203, 242, 243, 273, 274, 285, 445 (13)
- `t3toolbox/backend/entries.py`: 168, 169 (2)
- `tests/_jet_trs_twins.py`: 78, 112, 158, 192 (4)
- `tests/backend/_curvature_oracles.py`: 272, 273 (2)

Total: 104 lines. (`probing.py:642, 1217` are inside P-0640 / P-1215, fenced blocks, so not counted.)
