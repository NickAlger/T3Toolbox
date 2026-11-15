# R4a review — code prose outside the sampling kernels (plan draft 26, §7 row R4a; §4.1–4.4, 4.7, 4.10, 4.11, §4A, §1.2a, §3, §5)

Reviewer, 2026-11-13. Reviewed: the working tree on `d534714d` (HEAD; its code tree is R3's `74683a39`), which holds
R4a uncommitted (54 files: 30 library, 23 tests and helpers, 1 example; a snapshot of the diff was taken before any
run), with `R4a_handoff.md`, `R4a_name_hunks.tsv`, `R4a_realignment.diff`, `k9_registered.tsv`, the inventory's 47
filled entries and the two tool diffs. Every check below is my own, run independently of the implementer's runs;
scripts and outputs are in the session scratchpad under `rev4a/`, not in the repo. Heavy jobs ran one after another.

## Findings

### MINOR 1 — `utv_operations.py`: the frame's left/right cores still lettered `P`/`Q`, beside R4a's own `L`/`R`

Lines 449 ("left-core ``P``"), 464 ("the P-parallel component"), 465, 479, 487, 517, 557 (`(P^L)^T dG^L`) and 499
("the right core ``Q``"). The implementer reported them and left them (no row). **My reading: every one is a frame
store, so `L`/`R` by D-7.** The code decides it:

- 465 and 557: `einsum('d...iaj,d...iak->d...jk', left_sc, ttv)` / `einsum('d...abi,d...abj->d...ij', left_sc, ttv)`,
  the frame's left supercore against the TT variations;
- 487: `Xi = einsum('...iaj,...iak->...jk', L, dG)` after `L, R, dG_in = core` (the code's own local is `L`);
- 479, 517: the carry of that same scan;
- 499: the compensation runs through `right_sc` (`X_prev @ R`, line 486), which R4a's own comment at 512 now calls
  "the right core R";
- 449, 464: `left_sc` in `utv_orthogonal_gauge_projection`.

Why MINOR and not "leave": R4a's edits left the module contradicting itself. 498 says "the up core ``U``" and the next
line "the right core ``Q``"; 512 says "R" for the same core 499 calls `Q`; `utv_gauge_residual`'s docstring (539, the
§4A block UTVOPE-0538) says `(L^L)^T H^L = 0` and its code comment eighteen lines below (557) says `(P^L)^T dG^L` for
the same gram, two lines under 555's new `D V^T`. The §3.4 scan missed them (`P^L`, a backticked `P`, `P-parallel`;
K8's `\b[PQ]_(?:i|last|\{)` cannot see any of them). **I agree to them as §3.4 text rows (the R1 gate's "with the
reviewer's agreement"); the session adds them and R4a applies them:** 449 "left-core ``L``", 464 "the L-parallel
component", 499 "the right core ``R``", 465 and 557 `(L^L)^T H^L` (the operand is `ttv`, matching UTVOPE-0538). For
479, 487 and 517 the local is the NAME `dG` (see S4): `(L^L)^T H^L` keeps one spelling per object in the module, which I
recommend; `(L^L)^T dG^L` would match the local. The session picks.

### MINOR 2 — `t3_linalg.py:418, 419, 493`: a T3's own Tucker factor still written `U`, in a function whose `U` is the SVD factor

`_t3m_swap_pair(eL, eR)`'s signature comments read `# [G, U, mode]: G=stack+(a, nL, b), U=stack+(nL, NL)` (and the
`eR` twin), and `_t3m_move_center(chain)` reads `# list of [G, U, mode]`. The element's middle entry is a T3's own
Tucker core: the body unpacks `GL, BL, mL = eL` (R2's tier-C B row `UL→BL`) and the chain is built as
`[[Gx_o[i], Bx[i], i] ...]` (:581). Inside `_t3m_swap_pair` the body's `U` is the SVD factor
(`U, s, Vt = linalg.truncated_svd(M)`, §1.5 item 11), so one function now uses `U` for two objects: exactly what §0
item 11 rules out ("after the rename a `U` in code names the up store or an SVD factor, and nothing else"). The plan's
tier-C B row lists the twin comment at `:455–456` (`_t3m_contract_pair`), which R4a applied; these three lines are the
same object in the same form and were not listed. Not an implementer error, but R4a's file. **Ask the session to add
`:418, 419, 493` as rows (`[G, B, mode]`, `B=stack+(nL, NL)`, `B=stack+(nR, NR)`) and apply them in R4a.**

## For the session (rulings; R4a's diff correctly reported rather than improvised)

- **S1 — the nine `(U, G)` → `(B, G)` (item 3). Right under D-6; record nine rows.** Each tuple is a point's own cores,
  none a frame tuple: `CorewiseGeometryOps.base_point` (`geometry.py:522`) returns `(frame[0], frame[2])` of the
  corewise frame `(B, G, G, G)`; `retract` (`:536`) adds the variations to `self.base_point(frame)`; `point_tangent`
  (`:554`) returns `self.base_point(frame)`; the uniform twins (`:1007`, `:1035`) the same on bare supercores; and
  `x0` in `gradient_descent`, `mc_sgd`, `adam`, `newton_cg` (`optimizers.py:879, 958, 1003, 1315`) is the initial
  point's `(tucker_cores, tt_cores)`. I agree to them as §3.4 text rows; the writer records them.
- **S2 — the comprehension `B` at `frame_variations_format.py:856` (item 6b), and three siblings the handoff does not
  name.** `tuple([B.shape[-2:] for B in self.tucker_variations])` iterates the Tucker variations, `(nU, N)`: by Nick's
  shape rule `V`. Beside it, `:857` `G.shape[-3:] for G in self.tt_variations` is a TT variation, `(rL, nD, rR)`: `H`.
  `T3Variations.validate` repeats the pair at `:941–942`. These are point letters on variations, like the four locals
  ruled in draft 24 and `:821`'s model row `U→V`. They are NAMEs outside the token map, so not R4a's (prose); each name
  is free in its comprehension. Recommend four rows (`B→V`, `G→H`, twice), listed for K15 in whichever slice the
  session chooses.
- **S3 — `tv_operations.py:774, 779`, ``D^T V`` and ``max|D^T V|`` (item 6c). Recommend `D V^T`.** The line that
  decides is the gram itself, `g = xnp.einsum('...ia,...ja->...ij', D, V)` (`tv_gauge_residual`'s body): `D V^T`,
  `(nD, nU)`. With the library's `V` (`(nU, N)`), `D^T V` is `(N×nD)·(nU×N)` and the sizes do not conform: the prose
  fails §4A(f)'s contraction-size test; it reads right only in T4S's layout (factors as `N×n`). The same module states
  the gauge as ``D_i V_i^T = 0`` (TVOP-0108, `:108`), and R4a re-laid UNMA-0478 and UTVOPE-0538 into the code layout,
  so after R4a this is the one place in the code tree that still uses the other layout. Two lines; add them as rows and
  apply in R4a (`L^T H` may stay, or read `(L^L)^T H^L` like its uniform twin).
- **S4 — NAMEs that the shape rule would rename, not on any row (report only).** `udots` (`tv_operations.py:245, 250`,
  `utv_operations.py:224, 241`) is the gauged tied ambient direction, written `Ddot` in the prose two lines above;
  `_utv_oblique_gauge_step`'s `dG`/`dG_in` and `utv_to_ut3`'s `dG` are TT variations, `(rL, nD, rR)`, whose ragged twin
  `tv_oblique_gauge_projection` took `dG→H` in R2. Rows for a later slice if the session wants them.
- **S5 — `t3_orthogonalization.py:134`** "Outer orthogonalize TT cores, pushing remainders downward onto tucker cores
  below.", unchanged from the tag. T4S's "outer" as a verb; K8's `outer[- ]orthogonal` is case-sensitive, so the
  capital `O` escapes it. The twin docstring right below now reads "Down-orthogonalize the Tucker cores, pushing the
  remainders up onto the TT cores above." If the session adds a row: "Up-orthogonalize the TT cores, pushing the
  remainders down onto the Tucker cores below." Otherwise it can stay (no store word, nothing wrong).
- **Plan text for the writer**, beyond the handoff's two: S1's nine rows; MINOR 1's and 2's rows; §4.11's images `D`,
  `U` (`test_padded_frame.py:491`) and `Di` (`_uniform_padded_frame_testing.py:41`) besides `w`;
  `utv_corewise_retract`'s three shape comments with the generic `n` (§4 item 12's principle applied to a
  non-enumerated site; I agree: at the corewise frame every slot carries the point's own rank, `nD = nU = n`); §5.3's
  record of the tool changes (K9 registrations with their `keep` and `swap` categories, the doctest-source skip,
  `K8_R4A_CARVES`, K14's previous-slice comparison informational from R4a on, K10b's colon rule, K11's exceptions).

## NITs

- **N1 (tool).** K9's doctest-source skip makes a store-word string literal inside a `>>>`/`...` line invisible to
  K9; NAMEs there are K15's, but literals are nobody's. None exists in the tree today (my grep of every doctest source
  line for a quoted `up`/`down`: 0). Reporting such literals instead of skipping them would close it for R4b/R5.
- **N2 (tool).** `k9_registered.tsv` matches by whitespace-normalised text per file, so a later identical line
  anywhere in the same file is exempt too (the two `nD:` lines of `ufv_conversions.py` are already one text). Fine for a
  one-off tool with `K9-registered` reporting stale rows; worth a sentence in §5.3.
- **N3 (layout).** `uniform_padded_frames.py:404–405, 450–451`: the `carry` comment was split onto a 14-character
  continuation to stay under 120 while the `x` line two lines below is left at 121 (handoff §4 item 6). Either is
  within reason; one treatment for both would read more evenly. No change required.

## (1) The diagrams and equations

**All 47 blocks read against the tree** (inventory new text = the file verbatim: K12 below), before/after as in
Appendix A, which I checked line by line against `git show v2026.4.0` and the tree.

- **Diagrams by adjacency** (FVF-0058a–c, 0063a–c, 0088, 0100, 0217, 0222, 0845, 0850, 1264, 1271, 1355, 1362,
  UFVF-1106, 1113): every leaf is `D`, every TT-shaped core above a Tucker slot or a `( )`/`(Vi)` hole is `U`, `L`/`R`/
  `H` unchanged. The vertical widths: `L0–D0`, `(H1)–D1`, `R(d-1)–Dd` are `nD0`, `nD1`, `nD(d-1)` (an edge touching a
  Tucker factor); `U1–(V1)` is `nU1` (the up core's middle leg and `V1`'s rows). `Dd` keeps the known `Ud` typo at its
  width, as sample edit 5 (approved) writes it. K10: same line counts and lengths, every changed column a label
  character (the tool, and Appendix A by eye).
- **The `T3Frame` relations (FVF-0117/0123), re-derived.** `D_ia D_ja = δ_ij`: `D_i` is `(nD_i, N_i)`, `a` runs over
  `N_i` (below the edge) → `(nD, nD)`, down-orthogonal; the code's check is `einsum('...io,...jo->...ij', D, D)`
  (`ufv_operations.py:293`, `fv_frame_orthogonality_residual`). `U_aib U_ajb = δ_ij`: `U_i` is `(rL_i, nU_i, rR_(i+1))`,
  both bonds summed → `(nU, nU)`, up-orthogonal; the code's `einsum('...iaj,...ibj->...ab', U, U)`. `L_abi L_abj`:
  `(rL, nD, rL')`, left bond and leg summed → `(rL', rL')`. `R_ibc R_jbc`: `(rR, nD, rR')`, leg and right bond summed →
  `(rR, rR)`. The equations start in one column (column 65, 0-based) after `down-orthogonal:`, `up-orthogonal:`, `left-orthogonal:`,
  `right-orthogonal:` — the colon after `right-orthogonal` added (Nick), sample edit 6 otherwise verbatim. Gauge
  `D_ia V_ja = 0`: `(nD, N)·(nU, N)ᵀ` → `(nD, nU)`, the gram of `tv_gauge_residual`'s `einsum('...ia,...ja->...ij', D,
  V)`; `L_abi H_abj = 0`: `(rL nD, rL')ᵀ(rL nD, rR')` → `(rL', rR')`.
- **Field comments (FVF-0156, UFVF-0110)**: sample edit 1 verbatim (comment at column 48, `typ` at 24). `U_ixj U_iyj`:
  `i`, `j` the bonds summed, `x`, `y` the middle leg; `L_iax L_iay`: left bond and leg summed, free right bond;
  `R_xaj R_yaj`: leg and right bond summed, free left bond. Shapes `nDi` on down/left/right, `nUi` on up (K13 clean on
  them). The uniform block the same with the leading `d`.
- **Every other relation and listing**: FVF-0354 (the einsum strings unchanged, `D` → `(nD, nD)`, `U` → `(nU, nU)`);
  FO-0201; TVOP-0108 `D_i V_i^T`: `(nD,N)(N,nU)`; M-0685 (`i` over `nD`, `j` over `nU`, `a = N` summed); UNMA-0478 and
  UTVOPE-0538 `D V^T`, `(L^L)^T H^L` = the code grams at `utv_operations.py:555, 557`; UFOP-0083 = the code at
  `ufv_operations.py:94–96` (`N = down.shape[-1]` now right); RA-0644 step by step against `ranks.py:672–681` and
  `fv_conversions.py:207–215` (down, left sweep, right sweep, up); UTVOPE-0314 against the mask concatenations at
  `:335–338` (`[down | up]`; `[right_ext | left_ext]`; the content `1` in L at bond 0, R at bond d from the zero rows);
  the listings PF-0041/0043, UPF-0047/0050, FVF-0071/0077/0784/1371, UFVF-1122, T-TFVF-0025 against `_frame_core_shapes`,
  `ufv_variation_shapes`, `_random_frame_variations` and the doctests' printed `variation_shapes` (`(9, 14)`… with up
  cores `(1, 9, 4)`…). FVF-1302's companion: `U1 (2, 8, 5)` meets `V1 (8, 15)` on 8, `H1 (2, 11, 5)` meets `D1 (11, 15)`
  on 11; both `expected` tuples read right in the new letters.
- **Equations outside the blocks (tier-E prose), re-derived with sizes:** `S_i^T = <U_i, H_i>` `(nU, nD)` and
  `V_i = S_i^T D_i` `(nU, N)` (`fv_operations.py:459–460`); `Bdot_i = (S_i^T)^+ V_i` `(nD, N)`, `Gdot_i = H_i ×_3 T_i^+`
  `(rL, nD, rL')` and the transposes `V_i = ((S_i^T)^+)^T Bdot_i`, `H_i = Gdot_i ×_3 (T_i^+)^T` (`fv_operations`,
  `tv_operations`, `utv_operations`); `V_i = S_i^T Ddot`, `Ddot = M_g^+[V_i1; …]` `(n_g, N)`, the transpose's row blocks
  `(nUi, N)` (`sharing.py`); `H_i = <D_i, core_env_i>` `(r, nD, r')` and `V_i = <U_i, core_env_i>` `(nU, N)`
  (`tv_operations.py:400–401`, V5); `zB_i = B_z,i B_iᵀ` `(nZ, n)`, `G2_i`, `dG_i`, `dB_i = <env_i, G_i>ᵀ B_z,i`
  `(n, N)`, `zBdot_i` and the two Hessian blocks (`t3_linalg.py:814–820, 945–949`); `_inner_product_*_step`'s comment
  shapes against their einsums; `[D_i ; V_i]` `(nD + nU, N)`; `utv_to_ut3`'s block comments `[H, L]`, `[U, Z111]`,
  `[R, H]` against their concatenations; `V D^T (nU, nD)`, `B_x D^T (n_x, nD)`. All sizes agree.
- **The five "unchanged with a reason" blocks are right to leave.** T3OR-0044 is `t3_orthogonality_residual` of a T3
  (`B_i`, `G_i`: a point's own cores); TTT-2765/2837/2908/2979 state the orthogonality of `new_x.tucker_cores[ii]` /
  `new_tt_cores[ii]` after a T3 orthogonalization. No store letter, word or width; §3.3 keeps a `B` already written.
- **K12 completion.** The inventory has exactly 47 code-tree blocks outside R4b's eleven files, all filled, verbatim,
  with a same-reason where new equals old; the 254 pending are R4b's (A-0465, P-*) and R5's. The diagram tool: K10/K10b
  PASS (33 in place incl. the 5 same, 13 re-laid out), K11 PASS (259 fences byte-identical; the 41 rewritten excepted
  and checked by K12), K12 PENDING with no error.

## (2) The prose

- **Gates in R4a's files** (my run): every REPORT hit of K1, K2, K4, K5, K7, K8, K9, K13 and K17-shape-point is in R4b's
  files (8 + 6 + 103 + 16 + 90 + 235 + 37 + 8 + 4) or the docs (2 + 16 + 7 + 49 + 395). None in R4a's.
- **Tier D/E rows, sampled against the tree:** inv 2 §3's and §4's rows in R4a's files, inv 1 §3's sentences
  (module line 8, `random_orthogonal`'s slack sentence now `down_ranks=(4,4,4)` / `up_ranks=(2,4,2)`, `UT3Frame`'s
  supercore list, `UT3Tangent.structure`, `safety.py:18`, both `has_minimal_ranks`, `check_ufv_pair`, the
  CorewiseGeometry text, `T3FrameWeights`/`from_t3weights`/`check_fw_pair`/`fv_absorb_weights` and their uniform
  twins, the test docstrings), and the pqu text rows of `geometry`, `sharing`, `t3_linalg`, `tv_operations`. Each reads
  in the new convention; the validate messages (`len(DD)=`, `up_tt_cores[i]`, `Up TT core … rank mismatch`,
  `D.shape[-2]=`) match their code; the corruption labels match their lambdas.
- **The 28 backwards adjectives:** all of R4a's are right now (`frame_variations_format.py:1372–1373`,
  `uniform_frame_variations_format.py:1122, 1125`, `tucker_tensor_train.py:615, 2956, 2978, 3002, 3108` — each checked
  against its einsum —, `t3_orthogonalization.py:147, 171`, `ut3_orthogonalization.py:189`, `geometry.py:348`,
  `t3_svd.py:312, 345`, `fv_conversions.py:219, 251`, `test_frame_variations_format.py:447`,
  `test_tucker_tensor_train.py:3202`); `probing.py`'s four are R4b's, `user_guide.rst` R5's.
- **The corewise sentences:** every frame tuple in R4a's files is `(B, G, G, G)`; every substitution sentence names
  only the TT-shaped slots and writes `(U, L, R) -> G` (plan §1.2a's rule for that case); no "no longer required
  orthogonal"; no sentence says what the down slot holds. `fitting.py`'s three `ApplyKind`/`ProbeDerivativesKind`
  comments read "through D … at the manifold frame (D, U, L, R); through B and G at the corewise frame".
- **`-ward`:** the four rows in R4a's files are done ("upward ``ξ̂``", "the frame's upward edge variables",
  "adjoint-upward" ×2); §3.3's grep finds nothing outside R4b's files.
- **Store-and-arrow (§4.2):** `uniform_padded_frames.py:563, 592` are sample batch 3 edit 9's recommended text
  verbatim; `test_padded_frame.py:562–565` and `test_contractions_interpreter.py:363` as the plan writes them.
- **Shape-rule letters:** no `Udot…`, `dU…`, `Ud`, `gU`, `Ug`, `U_g` left outside R4b's files and the `o1_*` "u dense"
  carve-out; the only `U`-family survivor is the NAME `udots` (S4). Prose `U`/`D` letters scanned token by token in every
  comment and docstring of R4a's files: each `U` is the up store or an SVD factor except MINOR 2's three; each `D` the
  down store. `P`/`Q` in prose: only MINOR 1's eight are frame cores; the rest are the direction `P`, the completion `P⁺`,
  the P-block, QR's `Q`, the projector `Q` and `P_{T_F}`.

## (3) The nine `(U, G)` → `(B, G)` — S1: all nine are a point's own cores; right under D-6.

## (4) The realignment

- **Whitespace only.** `R4a_realignment.diff`: 87 hunks, 179 lines, every `-`/`+` pair token-equal (my parse: 0
  exceptions). HEAD → tree over all 54 files: 171 changed lines are whitespace-only; every other changed line carries a
  prose change; K15 shows the NAME stream changes only in the 18 listed hunks.
- **Nothing left ragged.** My own scan, independent of the implementer's tool: over every non-R4b file changed since
  `v2026.4.0` (66 files, including the twelve R4a did not touch), every adjacent pair of lines that shared a deliberate
  column at the tag (a trailing `#`, an assignment `=`, a signature/field type column, with a ≥ 2-space gap on one of
  them) shares it in the tree: **0 ragged pairs**. The same scan on HEAD finds 106, so it sees the drift it is meant to.
- **Within reason.** Hand layouts 1–3 read well (the corruption list at one column; the `frame_structures` header over
  its data — `down_ranks` > `up_ranks` matches the rows; `sharing`/`update` at one column per blank-line group, as in
  `tucker_tensor_train.py`). Hand layout 4 re-pads the two shape-message tables to one `=` column as at the tag (no test
  reads them). Lines R4a lengthens past 120: `uniform_padded_frames.py:407, 452` (121), `tucker_tensor_train.py:3108`
  (a doctest line, 134 → 136) and the message tables (163 → 166, already long). Nothing absurd.

## (5) The gate-tool changes

| change | verdict |
|---|---|
| K9 registrations (`k9_registered.tsv`): 21 clause (d) = R3 Appendix C (I compared the texts), 14 clause (c) | **justified narrowing**: (c)/(d) are the plan's clauses, made checkable; keyed to text, they lapse when the words change, and `K9-registered` reports stale rows (0) |
| its `keep` row (`fv_conversions.py:207`, "Tucker down-orth") | **justified**: an ORTH adjective already right at the tag (`down_orthogonalize_tucker_cores`), a missed clause-(a) keep |
| its 12 `swap` rows | **justified**: a `difflib` artifact. Each new line equals the tag's neighbouring line (the A3/A2 swap turned `nU … up` into `nD … down` in place), so line alignment pairs it with that neighbour; each line's words match its own NAMEs (checked) |
| K9 skips doctest source lines (their trailing comments still read) | **justified narrowing** with a small hole (N1). It removed 4 R4a lines (`uniform_frame_variations_format.py:1029–1046`), all NAMEs `down`/`up` of the B1/B3 rows, which K15 covers |
| `K8_R4A_CARVES` (4 tuples, 5 lines) | **justified**, verified against the code: `t3_svd.py:424, 426` `P_i` is the coefficient block `W_block · s_q` (`B_i = P_i @ Vt_q`); `test_uniform_fitting.py:430` `O` is the order count (`d, O, C = …, self._ORDER + 1, …`; §1.5 item 13's meaning); `test_dispatch.py:2006, 2029` `P_{T_F} Y` is the tangent projector. Matched by (file, hit, line text), so they cannot hide another hit |
| K14 from R4a: the previous-slice-vs-base recount listed, not failed | **justified narrowing**: the 101 listed lines are all increases in R3's §5.2 test files (my check: 0 decreases), i.e. R3's own additions, which R3's gate passed; the slice's own hunks are still checked (0 lost) |
| K10b: a `:` closing a label is not an alignment column | **justified narrowing**: such a colon moves with its label; a real column after it is still a field start (≥ 2 spaces), so only coincidental colon columns drop out |
| K11: blocks whose recorded new text differs are excepted and checked by K12 | **justified**: K12 holds them to the recorded text verbatim (which Appendix A shows the reviewer), so a script edit inside one would still fail; untouched fences stay byte-checked |

## (6) The three reported-and-left items

- **(a) `P`/`Q` in `utv_operations.py`'s derivation comments:** frame cores → `L`/`R` (MINOR 1, with the deciding lines).
- **(b) the comprehension `B` at `frame_variations_format.py:856`:** a Tucker variation → `V` by the shape rule, with
  its `G` → `H` and the pair at `:941–942`; NAMEs, the session's ruling (S2).
- **(c) `tv_operations.py:774, 779` `D^T V`:** the code's gram is `D V^T`; recommend that layout (S3).

## (7) Runs (one after another, `t3toolbox` env unless stated)

| run | result |
|---|---|
| battery: `run --b9 battery_ref/b9`, then `compare battery_ref <mine>` | **PASS (bitwise)**, 2,190 arrays, 856 texts, 0 errors, `NEW=True`, `LETTERS_NEW=True` |
| gates: `store_rename_check.py --stage R4a … --names w,o` | K0, K3, K6, K8b, K9-converse, K9-registered, K14, K15, K15-cross, K17 (+ shape-frame, shape-converse) **PASS**; K13 FAIL 8 = R4b's fenced `probing.py` blocks; every REPORT in R4b's files or the docs (§2) |
| `store_rename_diagram_check.py all` | self-test PASS; K10/K10b PASS (33 + 13); K11 PASS (259; 41 excepted); K12 PENDING 47/301, no error |
| suite `pytest tests/ -q -n 3` | **1,347 passed, 4 skipped, 48,724 subtests passed, 0 failed** (17 min 26 s); the one warning is the intended adam-on-manifold one |
| doctests: `--doctest-modules t3toolbox`; `getting_started.rst`; every `docs/*.md`, `docs/contributor/*.md` but `doctest_style.md` | **241 passed**; pass; 0 failures |
| doctest outputs, static (`DocTestParser` over every changed file) | HEAD → tree: 2,573 examples, **0 expected outputs changed** (16 sources changed, whitespace or trailing comment only); `v2026.4.0` → tree: 2,953 examples, 0 changed |
| `sphinx -W --keep-going -b html` | **build succeeded**, exit 0, no warning (16 min 46 s, alone) |
| floor env (`tttt`: py 3.9, numpy 1.22.4, jax 0.4.30), the 21 touched test modules, plain serial pytest | **893 passed, 1 skipped** (27 min 37 s, alone). It started at 03:01, after revision 1's last code edit (02:57; no `.py` file under `t3toolbox/`, `tests/`, `examples/` is newer than 03:01), so it covers revision 1 |

First-pass verdict: REVISE (MINOR 1 and MINOR 2, each needing the session's rows first; S1–S5 rulings).

## Revision 1 (the session's rulings, applied by the implementer; reviewed 2026-11-13)

The session ruled MINOR 1, MINOR 2 and S1–S5; the rows are `store_rename_letter_sites.md` § "The R4a review rows"
(RV1–RV15), the SITES rows LS0315–LS0338 and 24 hit-table rows of tier `R4a-row`. I compared the tree with my
first-pass snapshot of the diff, line by line:

- **MINOR 1, closed.** `utv_operations.py:449` "left-core ``L``", `:464` "the L-parallel component", `:499` "the right
  core ``R``", and `(L^L)^T H^L` at `:465, 479, 487, 517, 557` (the session chose `H`, with S4's local rename), so
  `utv_gauge_residual`'s docstring and its two gram comments now say `D V^T` / `(L^L)^T H^L` alike. Every frame
  letter in the module's prose is `D`, `U`, `L`, `R` (my token scan).
- **MINOR 2, closed.** `t3_linalg.py:418–419, 493` read `[G, B, mode]`, `B=stack+(nL, NL)`, `B=stack+(nR, NR)`; the
  module's only `U` left in that function is the SVD factor.
- **S2–S5 as ruled.** `frame_variations_format.py:856–857, 941–942`: `V` over `tucker_variations`, `H` over
  `tt_variations`. `tv_operations.py:774, 779`: ``D V^T``, ``max|D V^T|`` (sizes `(nD, N)(N, nU)` agree; the code tree
  now states the gauge in one layout). `udots` → `ddots` (4). `utv_to_ut3.last_core`'s `dG` → `H` (`ttv[-1:]`,
  `(rL, nD, rR)`; its comments already said `H`); `_utv_oblique_gauge_step`'s `dG_in`/`dG` → `H_in`/`H` and
  `dG_stack` → `H_stack`, with their comments. `t3_orthogonalization.py:134` "Up-orthogonalize the TT cores, pushing
  the remainders down onto the Tucker cores below." Each new name is free in its scope (`H` in `last_core` is
  assigned before any read and no outer `H` exists in `utv_to_ut3`; `ddots` is new to both functions).
  `utv_corewise_retract`'s `dB, dG` stay, rightly: they have the point's own core shapes at the corewise frame.
- **Alignment of the touched lines.** The renamed lines keep their blocks' comment columns; the 1-column steps at
  `utv_operations.py:486–487` (74/75) and `:300–302` (79/78/79) are the tag's own (checked against `v2026.4.0`).
- **The K14 narrowing** (a tier-`R4a-row` hit row licenses its own DNT occurrences, counted per file and token):
  **justified**. R1's `dnt.txt` filed every `B`/`G` under §1.5 item 5 ("a T3's own Tucker cores", "the TT cores"), but
  S2's eight are variations, so they were never item 5 in meaning; K15's listed hunks pin their positions, so the
  count-based licence cannot let a different occurrence go. §1.5 says a change to a listed name goes to Nick: the
  session should name S2 in its report to him at the R4 stop, as a dnt.txt over-inclusion, not an exception.
- **Runs on revision 1** (mine, one after another): battery **PASS (bitwise)**, 2,190 arrays, 856 texts, 0 errors,
  `NEW=True`, `LETTERS_NEW=True`; gates as in the first pass (K14 PASS, **K15 PASS: 40 hunks, all listed**, K15-cross
  PASS, K17 PASS; every REPORT hit still in R4b's files or the docs, 0 elsewhere); diagram check unchanged (K10/K10b,
  K11 PASS; K12 47/47); doctest outputs, static, HEAD → tree: 2,573 examples, 0 changed; my raggedness scan: 0;
  floor env above; suite `-n 3` **1,347 passed, 4 skipped, 48,724 subtests passed, 0 failed** (17 min 44 s);
  `pytest --doctest-modules t3toolbox` **241 passed**; `sphinx -W` **build succeeded**, exit 0 (15 min 42 s). No
  `.py` file under `t3toolbox/`, `tests/`, `examples/` changed after 02:57, so every run covers revision 1.

NITs on revision 1 (none required): the handoff's §5 item 7 still says `utv_to_ut3`'s local "stays `dG`" and item 8
still quotes `t3_orthogonalization.py:134`'s old "pushing remainders downward" as kept; both changed in revision 1.
Writing the 24 rows into `R1_hit_table.tsv` (an R1 artifact) works for the tool; a separate R4a rows file would keep
the slices' records apart, if the writer prefers. And for R6's alignment sweep: my scan finds 21 one- or two-column
comment near-misses in lines this arc touched, all pre-existing at the tag (e.g. `optimizers.py:228` `frame:` one
column right of its block, `utv_operations.py:276–279`, `ufv_conversions.py:407–408`, `test_uniform_manifold.py:111–112`)
plus R3's new `test_frame_variations_format.py:544–545`; whitespace-only fixes whenever the session wants them.

No MAJOR or MINOR remains: MINOR 1 and MINOR 2 are closed, S1–S5 are applied as ruled, and every gate and run passes.

VERDICT: APPROVE
