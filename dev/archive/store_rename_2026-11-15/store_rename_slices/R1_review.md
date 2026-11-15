# R1 review — the store rename's tools and inventories (plan draft 20, §7 R1)

Reviewer, 2026-11-11, first pass. Reviewed: `R1_handoff.md` and every deliverable it lists. My own scripts and
outputs are in the session scratchpad under `rv1/` (`rv_hits.py`, `rv_diff.py`, `rv_dnt.py`, `rv_diagrams.py`,
`rv_ls_complete.py`), not in the repo.

## What I checked independently, and the result

1. **Hit table re-derived from the plan's tiers** (my own tokenizer script, written from §1.2/§1.2a only). After I
   fixed two slips of my own, it agrees with `R1_hit_table.tsv` on 4,390 hits at the same (file, line, column,
   old → new). Every difference traces to a plan row or a ruling:
   - mine only, 81: text hits inside §4A fences (A3 64, A1 15, A1-label 2), which the script skips by design;
   - theirs only, 267: the 221 `letter_sites` R2 rows (I audited these separately, item 5), the 4 §4.11 typo rows,
     27 `padded_frames.py` `kU`/`kD` comment forms (same positions and images; only the recorded token differs), and
     15 `_curvature_oracles.py` C-B hits where the plan's anchors (`:73`, `:241`, `:398`, `:524`) are read as
     function-scoped, so the bound name's later uses are renamed too (the only reading that does not leave a
     `NameError`);
   - `tv_operations.py:547`: the file map gives `U→D`, §1.2a's specific row gives `U→b`. The table takes `b`, which is
     right.

   No difference is left unexplained. My own check also finds every R2 letter row's function fully covered: no
   function is left with a mix of renamed and unrenamed occurrences of a mapped letter (32 functions).
2. **Battery.** Fresh reference run on the `v2026.4.0` worktree (B9 files saved by me). It is bitwise equal to the
   stored `battery_ref`. Then a fresh dry run of my own, overlaid on `git archive v2026.4.0`, against my reference:
   **PASS (bitwise)**, 2,190 arrays and 856 texts, with `NEW=True` and `LETTERS_NEW=False`. Negative control: I
   undid one swap (`ufv_conversions.py:237–238`, the default widths) and the battery fails, as it should. All 14
   keyword-renamed kernels are called directly, positionally (`B6/k/*`). The doctest expected outputs in
   `t3toolbox/` are unchanged on the preview (3,441 examples; 105 have changed input lines).
   **K-gates:** at R1 on the working tree, K0, K9-converse, K11, K14 and K15 PASS. At R2 on my preview, K0, K8b, K11
   (no sentinel inside a fence in pass 1), K13, K14, K15 and K15-realized PASS. The handoff's report counts
   reproduce.
3. **§4A inventory.** I ran my own scans of `t3toolbox/`, `tests/`, `examples/`, `docs/` (`.md`, `.rst`, `.tex`),
   `README.md` and `CLAUDE.md` for box, pipe and dash art; `::` and fenced literal blocks; `= delta`, `= I`, `= 0`,
   `δ_ij`; aligned listings; LaTeX width forms; and every store letter in the notes.
   - Every ascii diagram and orthogonality relation is a fenced block of its own. The six side-by-side panels at
     `frame_variations_format.py:58–66` are correctly split by column.
   - Every LaTeX width form is fenced and on K16 (16 lines).
   - Inline formulas inside prose sentences go to the R4/R5 text rows (`PROSE_LETTER`), the same treatment the plan
     gives `test_contractions_interpreter.py:207`'s `S^T = <O, H>`. I accept that line.
   - Exceptions: MAJOR-1 and MAJOR-2.
4. **`dnt.txt`.** My own check finds it disjoint from the map, the 80 proposed rows included. It holds the SVD `U`
   factors, the Greek names, `B` per file, `UB`/`UV`/`UW`/`UF`/`UFb`/`Usc`, `_shift_*` and `c_up`, the RNG `R`
   (5 scopes), the oracle `O`, the order counts `O` and `P` as the completion and pseudo-inverse. The contraction
   strings go through K14 directly. Gaps: MINOR-1. **`keeps.txt`** covers every §4.3 line (28), every §4.2 site
   (13) and every §3.3 site (39), with the EXEMPT rows matching the authoritative grep. Defect: MINOR-3.

## MAJOR

**MAJOR-1. A hand-aligned shape listing is dismissed as prose, so R2's script would edit inside it.**
`tests/test_frame_variations_format.py:25–27` (the `_random_frame_variations` docstring) is a two-column listing
with store letters and widths:

```
        U_i = (nU_i, N_i)           L_i = (rL_i, nU_i, rL_(i+1))
        R_i = (rR_i, nU_i, rR_(i+1))  D_i = (rL_i, nD_i, rR_(i+1))
        V_i = (nD_i, N_i)           H_i = (rL_i, nU_i, rR_(i+1))
```

`diagram_candidates.tsv` dismisses it as `PROSE_LETTER`/`PROSE_WIDTH`. Inv 1 itself names `:21–27` "the structure /
shape equations". It is unfenced, so `R1_hit_table.tsv` holds six A3 hits inside it (`:25` col 15, 49; `:26` col
21, 51; `:27` col 15, 49): R2 would rewrite a hand-crafted block token by token, and K12 and Nick's R4b
before/after would never see it. It is the same kind as `FVF-0071`/`FVF-0077`, which are fenced.
**Fix:** inventory it as one shape-listing block (fenced, lines 25–27), so its A3 hits leave the table, and re-run
the dry run, reconciliation and gates. Make the curation rule key on alignment, not on "prose": two or more
consecutive lines of `X = (shape)` entries. My scans found no other instance in the code tree.

**MAJOR-2. Three letter-bearing LaTeX lines are not inventoried (R1 gate: "every … letter-bearing equation line").**
- `docs/exact_hessian_two_hole.tex:861`: `$C_i=\dot US_i$`, the continuation of `TEX-EHTH-0860`'s sentence (the
  tied direction, `\dot D` under the session's ruling).
- `docs/symmetric_probe_derivatives.tex:540` and `:631`: `\delta\widetilde{\mathbf U}_i`, the transpose's Tucker
  block (`D` form).

The scanner misses `\dot US_i` (no word boundary after `U`) and `\widetilde{\mathbf U}`. `pqu_inventory.md` has the
three lines as R5 hand rows, but no gate sees them: K8's tex patterns (`\\delta\s*U\b`, `\\dot\s*U\b`, `\\dot\{U\}`,
`\\mathbf\s*\{?O`) match neither form.
**Fix:**
- Fold `:861` into `TEX-EHTH-0860` as one paragraph run, by your own convention for text-mode runs.
- Add `SPD-0540` and `SPD-0631` (or extend the neighbouring blocks if they form one paragraph run).
- Widen the scanner's tex letter pattern to any `\mathbf`/`\dot`/`\delta`/`\widetilde`/`\widehat` + `[UOPQ]`, with
  no trailing-boundary requirement, and re-run it. My broad scan found only these three.
- Separately, for the session: K8's tex patterns need the same widening (a plan item).

## MINOR

**MINOR-1. `dnt.txt` misses names that §1.5 and §3.2 keep.**
- (a) The QR factor `R`: `linalg.py:562`, `fv_operations.py:284–285` (§1.5 item 2 cites them) and
  `test_padded_frame.py:99` (also cited). `Q` is listed at these sites and `R` is not. Scope it per function,
  since `R` is also the right store.
- (b) The kept `sh` shape locals and `psh`/`pshd` (§3.2: "the shape locals `sh` stay, `_flatten_w`'s included"; §9
  names "a shape local `sh`" as the over-reach K14 must stop):
  - `sampling_derivatives.py` `_flatten_w` (`:1693–1694`) and `:3220`;
  - `backend/fitting.py:162–163`;
  - `tests/backend/test_uniform_padded_frame.py`;
  - `tests/oracle_sweeps/o1_common.py`, `o1_ragged.py`, `o1_uniform.py`;
  - `tests/test_oracle_sweep.py`.

  Scope them so the σ̂ rows (`sh→sig_hat`) stay disjoint.
- (c) After the session rules on MINOR-2 and the ASK items: rows for whatever is kept (the factored `P`/`Pd`/`Q`/`Qd`
  of `tt_tangent_assembly_factored_trs`; `test_curvature`'s σ̂ `sh` if kept).

**MINOR-2. σ̂ locals outside the plan's σ̂ row, not flagged.** `tests/backend/test_curvature.py:1063–1066`
(`test_apply_tt_assembly_is_the_three_term_product_rule`): `sh = probing.compute_sigma_hat(G, xi, c)`, `shd = …`
(3 + 2 tokens). §3.2's row lists `sampling_derivatives.py`, `apply.py` and `_jet_trs_twins.py` only. Report it as an
ASK, for the session to choose between `sig_hat`/`sig_hat_d` like the row, or keep plus a DNT row. Add test_curvature's
five finite-difference `shift = lambda` sites (`:583, 706, 865, 1168, 1496`) to §7.7's list, which has only the
seven doctests.

**MINOR-3. `keeps.txt` has a bad path.** The row `contributor/curvature_internals.md	23` is inv 3's relative path,
missing `docs/`. It can never match, so K9 at R5 misreports that line, and the converse skips it. Fix it, and assert
in `r1_build_lists.py` that every keep path exists.

**MINOR-4. `R1_s33_ward_sites.tsv` lacks the explicit row.** It holds 38 grep rows. Handoff §1 and §7.6 say it is "38
grep lines + `apply.py:215`", the authoritative list for R4b. `keeps.txt` has `apply.py:215` EXEMPT; the tsv does
not. Add it.

**MINOR-5. A stale "landed" row in `pqu_inventory.md`.** It says `test_from_sweep_reuse.py`'s `dU`: "no `dU` NAME
exists there (tier C's §1.2a row hits 0)". That came from the short path. The hit table has 8 hits in
`tests/backend/test_from_sweep_reuse.py` (`_bare_to_dense`, `test_transposes_per_element`). Correct the row.

**MINOR-6. K13's generic list is not reconciled with §4 item 12's ten.**
- At the R2 preview, K13-generic lists 14 lines, including `sampling_derivatives.py:145–146, 1017–1018, 1140–1141`:
  the `x` of `t3_*_derivatives`, a T3 point's own cores, which R2 turns into `nDi`. By item 12's rule they take the
  generic `n`.
- The plan's `ufv_conversions.py:334–335` are not generic. They are `up_ranks`/`down_ranks` locals that K13 checks
  as store-named entries.

List both facts in the handoff, so the session gives R4b the six extra dispositions and the writer corrects item 12.

## NIT

1. **`pqu_inventory.md:949–950`.** The ASK reason for `test_uniform_fitting.py:360–361` says the freeness check
   fails. Under your judgement call 8 there is no clash: `fam`'s `R` is a local of a nested def that contains no
   renamed image. Reword it when the rows move.
2. **§7.10.** Add the `_curvature_oracles.py` function-wide B-row completions (`:74–77`, `:242–248`, `:400–409`,
   `:526–535`) to its list for transparency. They are correct.
3. **Plan text for the writer.** The two short paths; `dc` is in 16 modules, not 7; `U_3→D_3` is an R4a text row,
   not tier C; and §5.1's adapter should name `LETTERS_NEW`.

## The implementer's 13 judgement calls

All agreed:

- Calls 1–7, 9 and 13: my independent derivation reproduces them.
- Call 8: generalizing the nested-def exception is sound under Python scoping. My own freeness check finds only the
  `core_dot` exceptions.
- Call 10: agreed, with MINOR-1's gaps.
- Call 11: agreed, with MINOR-3's path.
- Call 12: agreed.

## The 80 R2-proposed rows

**Agreed, all 80.** Each is a §3.4-classified hit whose image passes the freeness check:

- **`KU→KB` (27) and `U→B` (`ut3_linalg.py:417`, 4).** The corewise Hessian's Tucker block of θ's own cores (D-12).
- **`kU→kB` (`uniform_tucker_tensor_train.py:569–570`).** Beside the plan's `gU→gB`.
- **`oU→oB` (9) and `Uc→Bc` (`test_curvature.py:1003–1006`).** θ's own raw cores.
- **`U→D`, the frame's down store.** `frame_variations_format.py:167, 171, 383, 1609`; `test_tv_operations.py:151–152`;
  `test_uniform_curvature.py:249–250` (θ = (D, L) by its stores); `test_frame_variations_format.py:224`;
  `test_dispatch.py:337–338` (it mirrors `_z_jets_step`'s `U`, which tier C maps to `D`).
- **`Ux→Bx` (`:1609`).**
- **`U→B` (`test_padded_start.py:502`).** Over `x[0]`; its `Q` is a QR factor and stays.
- **`O→U`/`U→D` (`sharing.py:619–620`).**
- **`VUt→VDt` (`test_curvature.py:181–182`).** `V_i D_iᵀ`.
- **`OO→UU`, `SO→SU` (`test_sharing.py`).** `frame_d[1]`, the up store.

Flip them to `R2`. None needs the session: each lies inside §3.4's reviewer-agreement mechanism.

## The 10 "could not follow as written" items

| # | who is right | note |
|---|---|---|
| 1 `U_3→D_3` | implementer | prose at `test_dispatch.py:1986`, an R4a text row |
| 2 short paths | implementer | both files exist only under `tests/backend/`; fix MINOR-5 |
| 3 ASK store locals | implementer to ask; both are stores (below) | token-map additions: the session's call |
| 4 `LETTERS_NEW` | implementer | after R2 and before R3 the tree is `NEW` with the old letter API; reading the type of `update`'s default is sound; the plan's §5.1 should say so |
| 5 nearest faithful forms | implementer | all acceptable. B6 calls all 14 kernels directly; B2's per-store weights are what §5.2 asks for |
| 6 §3.3 grep misses `apply.py:215` | implementer | but the tsv must carry the row (MINOR-4) |
| 7 no shift variable | implementer | add the 5 test lambdas (MINOR-2) |
| 8 `.tex` copied into the pass trees | implementer | |
| 9 the 17 §3.4 ASKs | my readings below | |
| 10 tier-C completions | implementer | completions of the plan's bindings, not new renames |

## The ASK renames: my readings (the deciding line for each)

- **`up_more`, `up_shared` are stores.**
  - `fv_operations.py:368`: `up_more = fvo.fv_zero_point_frame(…)[0]` is slot 0 of a frame, the Tucker store (new:
    down). `:378` is the same.
  - The plan missed them (inv 2 §9 proposed them). Add both to A2 (`→down_more`, `→down_shared`; the images are
    free). Otherwise R2 leaves `np.allclose(up_more[0][:2], down[0])`. Session.
- **`um`, `dm` are stores.**
  - `test_uniform_frame_variations_format.py:892`: `um, dm, lm, rm = uframe.data[5]` (the frame masks in physical
    order), beside A2's `mup, mdown` at `:891`. They pair at `:896`, `int(um[i].sum()) - rank(mup[i])`.
  - Add a tier-C `um↔dm` row scoped to `_lost_directions`. Session.
- **The 17 §3.4 occurrences:**
  - **`frame_variations_format.py:821` `U` (×2): not a store.** `for U in self.tucker_variations` is a Tucker
    variation. It cannot stay `U` (§0 item 11; §3.1: a `U` that is neither the up store nor an SVD factor is never
    left as is). Recommend `V`, by the `dU = tt_variations[i] → H` precedent. A new image: session.
  - **`_curvature_oracles.py` `tt_tangent_assembly_factored_trs` `P`/`Pd`/`Q`/`Qd` (×8): not frame cores.**
    `:377` `P = contract('tsWCa,sWCi->tWCai', M[:, :s2], xs)` is a product laid out (t, W, C, a, i). D-7 applies
    only where they name frame cores, so they stay. §3.4's "sites known now" over-generalizes this file (writer); add
    DNT rows (MINOR-1c).
  - **`test_t3_orthogonalization.py:123–125` `Oc` (×3): the up store.** `:123` `V, Oc =
    O.t3_up_orthogonalize_tt_cores(x)` is the output the session ruled `U` in the kernel itself (`Oaxb→Uaxb`), and
    `:125` checks its up-orthogonality. Rename it `Uc`; the module alias `O` stays (DNT item 13). This is within
    §3.4's O-family rule; I agree, and the session may confirm.
  - **`test_uniform_fitting.py:360–361` `P`/`Q` (×4): frame cores.** `:361` `lambda f: [Q.shape[-1] for Q in
    f.right_tt_cores]`. `→L`/`R`. `fam`'s `R` (`:354`) is a nested-def local with no renamed image inside it, so
    there is no clash under call 8 and the plain letters stand. I agree, as the clash disposition §3.4 gives the
    reviewer.

## Re-check against the maintainer's V/H confirmation (2026-11-11)

The ruling: "Tucker variations are V and TT variations are H. The D U L R are for the frames." Variation cores
are `V`/`H` everywhere. `D`/`U`/`L`/`R` name frame cores only, and a derivative with respect to a frame core takes
the core's letter (`dD_tildes`, `Ddot`). `B`/`G` name a point's own cores.

**Method.** I re-checked every frame-letter NAME rename in the hit table (tier C file maps, B4, C-B, C-D, §1.2a, the
225 R2 letter rows) and the 80 proposed rows. For each, I resolved the binding of the old name to its source:
- the zip position or tuple element for unpacks;
- whole-RHS for calls, then read by hand;
- the shape comment for parameters and `data` unpacks.

### Flagged: a frame letter on a variation-holding name (new dispositions, for the session)

| # | site | today | the name holds | plan's image | by the ruling |
|---|---|---|---|---|---|
| V1 | `sampling_derivatives.py` `_tangent_z_jets_step` (`:863, 868`) | `U, dU, eta_jet, deta_jet = data`; "dU is the variation core (K+C)" | the tangent's Tucker variation (`var_tucker_cores`) | `dD` (§1.2a's own reason: "a tangent's Tucker variation V, so it is dD") | `V` (free); `:859`'s data comment and `:865`'s prose follow in R4b |
| V2 | `tests/test_jet_recurrence.py` `_problem` (`:57, 62, 67`, the `dict(dU=…)` keyword included) | `dU, dG = v.variations.data` | a tangent's variations | `dD` | `V` (free); partner `dG` → `H`, see below |
| V3 | `tests/test_probe_derivatives.py` `_var` (`:499, 501`) | `dU = tuple(R(*K, *C, self._ND[i], self._N[i]) …)` | a random tangent Tucker variation | `dD` | `V` (free); partner `dG` → `H` |
| V4 | `tests/test_probe_derivatives.py` `dU_v` in `test_tangent_transpose_adjoint_identity` (`:159, 169`), `test_tangent_transpose_K_stacked` (`:273, 283`), `test_apply_entries_transpose_adjoint_identity` (`:370, 386`) and `test_adjoint_identities` (`:533, 556`) | `dU_v, dG_v = v.variations.data` (or `= var`) | the input tangent's variations | `dD_v` | `V` (free in all four); partner `dG_v` → `H`. The same functions' `dU`/`dU0` are transpose outputs, so `dD`/`dD0` stay right |
| V5 | plan §3.3 text row, `tv_operations.py:400–401` | "`dG_i = <U_i, core_env_i>` (the TT variation) and `dU_i = <O_i, core_env_i>` (the Tucker variation)" | the two variations | `dD_i = <U_i, core_env_i>` | `V_i = <U_i, core_env_i>` and `H_i = <D_i, core_env_i>` |

**The partners.** In V2–V4 the partner is `dG`/`dG_v`, which holds the tangent's TT variation. The plan keeps the G
family ("the library's TT letter"), so `(V, dG)` would be a half-renamed pair. Renaming the partner to `H` is
outside the token map. `test_jet_recurrence` would also need its string key `p['dG']` (`:104–113`) as a scoped
string row if its `dict` key follows. These are session rulings, and so is V1–V5's new image `V`/`H`.

### Confirmed correct under the ruling (no change)

- **Transpose outputs: derivatives with respect to the frame core.**
  - `dU_tilde(s)→dD_tilde(s)` in `probing`, `sampling_derivatives`, `utv_sampling` (incl. `_remasked_transpose_output`),
    `apply._apply_transpose_adjoint` and `manifold.probe_transpose`;
  - `manifold`'s five transposes (`dU→dD`) and the `dU_sum` doctest;
  - `_apply_transpose_assemble_core`'s `dU_t→dD_t` (its `dU`/`dG` are subscript strings);
  - `_jet_trs_twins.apply_derivatives_transpose_trs` (`gU→gD`, `dU→dD`);
  - `test_from_sweep_reuse`, `test_manifold.test_tangent_apply_transpose_kstack`, and `test_probe_derivatives`'s
    `dU`/`dU0`;
  - the text rows `entries.py:152`, `probing.py:1294`, `backend/fitting.py:351`, and `test_padded_start.py:568–569`
    (the tied group direction, like `Ddot`).
- **Dotted frame-core derivatives.** `Ud→Dd`, `Od→Ud`, `Pd→Ld`, `P_dot→L_dot`, `Q_dot→R_dot`, `dmO→dmU`,
  `dOn→dUn`, `Odxi→Udxi`, and `Udot→Ddot` (the gauged tied direction, the ruling's own example).
- **No `V`/`H` on a frame core.** The only `V`/`H` images are `tv_operations.py:292–331`'s `dU→H` (it holds
  `tt_variations[i]`) and my recommended `frame_variations_format.py:821` `U→V` (it iterates
  `self.tucker_variations`). Both hold variations.
- **The 80 proposed rows.** Frame letters land on frame cores only. `VUt→VDt` keeps `V` for the variation and `D`
  for the frame core. The B forms sit on directions and Hessian blocks of a point's own cores.
- **Every tier-C file map's bare letters** (`U`/`O`/`P`/`Q` and their forms): no binding comes from a variation.
  The flags my heuristic raised were `zip(frame_cores, variations)` loops whose renamed member is the frame core.

**A question for the session, outside `letter_sites`.** The plan's prose and notes convention writes a tangent's
Tucker variation as `δD`: §3.3 "in prose `δD`", §4.8 "`\delta D`", and the T4S line "its δU is our δD (… V)". The
session should say whether that stays as math notation for `V` or becomes `V` under "V everywhere".

## Verdict (first pass)

The tools are sound and the numbers reproduce independently: hit table, sentinels, bijection, battery (bitwise, with
a working negative control) and every K-gate. Two inventory misses (MAJOR-1, MAJOR-2) and six small defects remain.
The session also has rulings to make on `up_more`/`up_shared`, `um`/`dm`, `:821`'s `V`, the factored `P`/`Q`
(keep), `test_curvature`'s σ̂ `sh`, the six extra generic shape lines, and, under the maintainer's V/H
confirmation, rows V1–V5 and their `dG` partners.

First-pass verdict: REVISE

---

## Second pass: revisions 1 and 2 (plan drafts 21–23; the maintainer's shape ruling)

**Revision 1.**
- The hit-table delta from my first dry run is +142/−23. Every row traces to drafts 21–23:
  - the 80 agreed rows and `:821 U→V`;
  - A2 `up_more`/`up_shared`;
  - `um↔dm`;
  - the σ̂ row at `test_curvature.py:1063–1066`;
  - `Oc→Uc`, `P/Q→L/R`;
  - V1–V4 (`C-VH`) and the `p['dG']` keys;
  - MAJOR-1's six A3 hits removed.
- MAJOR-1 is fixed (`T-TFVF-0025`), MAJOR-2 too (`TEX-EHTH-0860` at 860–861, `TEX-SPD-0540`, `-0542`, `-0631`), and
  MINOR-1, 3, 4 and 5 are in place. MINOR-2 and MINOR-6 were resolved by rulings.
- The `psh`/`pshd` rows missed their binding line 3220. Revision 2 fixed it, scoping them to
  `_apply_tangent_assemble_step`.

**Revision 2 (the shape ruling: `(nU, N)` → the V family, `(rL, nD, rR)` → the H family, `(nD, N)` → `D` or
`B`).**
- The hit-table delta is +204/−93: the transpose outputs and the held locals, all tier `C-VH`.
- I checked every new image against its shape comment (old widths, then the new reading after the A3 swap):
  - `tv_orthogonal_gauge_projection`: `dB`/`dV` iterate the Tucker/TT variations → `V`/`H`.
  - `tv_oblique_gauge_projection`: `dB`, `dB_parallel`, `dG`, `dG1` come from `tucker_vars`/`tt_vars` → `V`,
    `V_parallel`, `H`, `H1`.
  - `t3_linalg._inner_product_variations_step` returns `(rLi,nUi,rR)`, `(nOi,Ni)` → H, V.
  - `utv_project_ut3_onto_tangent_space`: `(rL,nU,rR)`, `(nD,N)` → H, V.
  - `probing._assemble_tangent_z_step`: `B` is "the C-only frame core" → `D`, and `dB` "the variation core" → `V`.
  - Every transpose output is a T3Variations or `(nOi,Ni)`/`(rLi,nUi,rR)` block → V/H.
- **No image starts with `dD`.** `Dd` survives only in the two `_deta_tilde_tangent*_step` kernels, where
  `(U, Udot)` is `(nUi,Ni)` → `(nDi,Ni)`, the core's own shape. `Ddot` survives only where it is `(K+C)+(n_g, N)`,
  the group factor's shape. `Bdot` in the corewise lift and its transpose is `(nUi,Ni)` → `(nDi,Ni)`, also
  core-shaped.
- **Pending items (a)–(d): agreed.**
  - (a) `_apply_transpose_assemble_core`'s spec strings `dU`, `dG` → `gV`, `gH`, matching the twin, and its arrays
    → `V_t`, `H_t`.
  - (b) The twin's `gG→gH`.
  - (c) `_bare_to_dense`'s `dU`, `dG` → `tkv`, `ttv`. `V` is bound there, and `tkv`/`ttv` are the uniform
    doctests' names for these supercores.
  - (d) `dV→H`, `dG1→H1`.

  All the images are free: my freeness check finds only the sanctioned `core_dot` and `fam` cases.
- **Width first, then the letter: agreed** for the five corewise-Hessian kernels (item 12's generic `n` makes them
  core-shaped, so `dB`/`dG` stand).
- **Reproduced on my own runs:**
  - a fresh dry run (4,887 hits, 0 mismatches or bijection errors);
  - the battery, **bitwise PASS** against my own reference;
  - the R2 gates: K0, K8b, K9-converse, K11, K13, K14, K15, K15-realized, K17 (891) and K17-shape (83 pairs) all
    PASS;
  - DNT disjoint (1,393 rows);
  - K17-shape catches both of my injected `dD`-on-V cases (a `NAME = … # (nUi, Ni)` line and a returned name) at a
    site other than the implementer's.

### Remaining (MINOR)

**MINOR-7. K17-shape and K17-labels see only `D`/`dD`/`U`/`dU`/`gD`/`gU` names.** The shape ruling also puts the
`B`/`G` letters out of bounds on V/H-shaped arrays (the held locals were exactly `dB`/`dG`). My scan of the R2
preview finds:
- `B`/`G`-lettered labels on V/H shapes whose arrays are now `V`/`H`: `t3_linalg.py:682–683` (`# dG.`, `# dB.`) and
  `sampling_derivatives.py:1610, 2228` (`# dG_tildes.`). These are R4 comment rewrites.
- Corewise ones that item 12's `n` will make core-shaped: `sampling_derivatives.py:2925, 3240`, `probing.py:1663`,
  `utv_operations.py:392, 396`.

Extend `FRAME_LETTERED` (or add a sibling pattern) to `B`/`G` forms (`(d|g|o|k|K)?[BG]…`, `dG_tildes`). Report them
at R2, gated from R4b once item 12's `n` lands, and put them in K17-labels. Add the converse too, a V/H-family name
on a core-shaped array; it is 0 today.

**MINOR-8. The width-first list misses `utv_corewise_retract`** (`utv_operations.py`). After A3, `:392` (`G_sc …
(rL, nD, rR)`) and `:396` (`dB, dG … (nU, N), (rL, nD, rR)`) read V/H shapes on the corewise frame's raw
perturbations. Its `dB`/`dG` rightly stay, so its comments take the generic `n`, in R4a: it is not a sampling
kernel. Add it to §0b's list.

Second-pass verdict: REVISE (MINOR-7 and MINOR-8 only)

---

## Third pass: revision 3 (MINOR-7, MINOR-8)

- **MINOR-7 fixed.** K17's shape half now gives three verdicts:
  - `K17-shape-frame`: `D`/`dD`/`U`/`dU` on a V/H array, gated from R2;
  - `K17-shape-point`: `(d|g|o|k|K)?[BG]` forms on a V/H array, reported until R4b and gated after it;
  - `K17-shape-converse`: an introduced V/H name on a core-shaped array, gated from R2.

  Composite widths are no longer read as a family, and K17-labels lists all 13 comment labels. On my own copy of the
  pass-2 tree:
  - an injected `dB = V  # C+(nUi, Ni)` is reported as **point**;
  - an injected `V_x = Dd  # K+C+(nDi,Ni)` FAILS the **converse**;
  - the earlier `dD` injections FAIL **frame**.
- **The scan's three new NAME sites, agreed** (9 hits): `dG_tilde(s)→H_tilde(s)` in
  `probing._assemble_tt_variations_core` and `assemble_tt_variations`, and in
  `sampling_derivatives._assemble_tt_variation_jets_dense`. Each declares `(rLi,nUi,rR)`, which reads `(rLi,nDi,rR)`
  after A3: the H shape, the TT twins of the `V_tilde(s)` already renamed. All are free.
- **MINOR-8 fixed.** `utv_corewise_retract` (`:392`, `:396`) is on §0b's R4a width-first list; the TT twins
  `assemble_tt_tangent_variations` and `assemble_tt_tangent_variation_jets` are on the R4b list. The five point
  reports at R2 are exactly the corewise kernels that list covers.
- **Reproduced on my own runs:**
  - a fresh dry run: 4,896 hits, 283 keys, 0 mismatches, bijection errors or conflicts;
  - the battery, **bitwise PASS** against my own `v2026.4.0` reference (2,190 arrays, 856 texts);
  - the R2 gates: K0, K8b, K9-converse, K11, K13, K14 (1,393 DNT rows), K15, K15-realized, K17,
    K17-shape-frame and K17-shape-converse all PASS;
  - DNT disjoint, and freeness showing only the sanctioned nested-def cases;
  - doctest expected outputs unchanged (3,441 examples; 138 changed only in input lines);
  - the test files the letter and V/H rows touch, at `-n 3` on my preview: 227 passed, 1,340 subtests. The files were
    `test_probe_derivatives`, `test_jet_recurrence`, `test_manifold`, `backend/test_from_sweep_reuse`,
    `test_frame_variations_format`, `backend/test_tv_operations`, `test_uniform_frame_variations_format`,
    `backend/test_t3_orthogonalization` and `backend/test_uniform_fitting`. The full suite is the implementer's run.
- **NIT.** `VH_LETTERED` (`^g?[VH](?:$|[0-9_])`) does not match a hand-written `Vd`/`Vdot`/`Hd`. Every V/H image the map
  introduces matches it today. For parity with the other patterns, add `dot|d$|i$|s$` so a later slice cannot slip one
  in. This is optional.

## Verdict

No MAJOR or MINOR remains. The tools, inventories, battery reference and gates are fit for R2. The only open item is
a NIT (`VH_LETTERED`).

VERDICT: APPROVE
