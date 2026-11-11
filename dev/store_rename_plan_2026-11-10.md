# The store rename — plan (draft 4, writer, 2026-11-11)

> **Status:** draft 4: Nick's rulings of 2026-11-11 and critic rounds 1–3 (`dev/store_rename_plan_review_r{1,2,3}.md`)
> applied. **The critic agreed in round 4** (`dev/store_rename_plan_review_r4.md`). Next: Nick, then the
> implementer/reviewer slices R1–R6 (§7). **Breaking**; the first item of 2026.5.0. The glossary update and the 601-row docs-consistency pass (`dev/glossary_naming_decisions.md`) run after
> it, on the renamed code.
>
> **Sources:** `dev/store_rename_inventory_1_frontend_tests.md`, `_2_backend.md`, `_3_docs.md` (inv 1/2/3 below) and
> `dev/up_down_naming_blast_radius.md` (BR). Their line numbers are of `b272e5de`; nothing outside `dev/` changed through
> `79b2bd6e`. `t3toolbox/`, `tests/` and `examples/` equal the `v2026.4.0` tag; `docs/` differs from it by `glossary.md`
> and `index.rst`.
> Every hand edit asserts its row's current text before replacing it, so a stale row fails loudly.

## CHANGES

**Draft 4 (critic r3).**

- K9 gains the converse: a keep word must survive at its aligned position.
- Tier D lists the hybrid `(D, U, P, Q)`.
- The dense-frame names `startD`, `Da`, `Db` become words.
- `gYU`→`gYD` is added.

**Draft 3 (critic r2).**

- **3.1 N1: `Λ` was not free.** It is the exact-Hessian note's Lagrangian (7 uses). That Lagrangian becomes `ℓ`, which
  is free in the note, and the shift keeps `Λ` (§3.2, D-1).
- **3.2 N2: the rulings had no inventory.** R1 now builds the `P`/`Q`/`U`-family inventory with a classification
  (§3.4). Tier-E texts are applied with the rulings layered on, tier E before tier D on a shared line (§1.4), and
  K4/K8 cover the hybrids.
- **3.3 Minors and nits.**
  - Checks: K9 is token-level; K7, K8 and K8b are fixed.
  - Sentinels no longer embed old text (`ZQX%04d_XQZ`).
  - Scoping: the catch-all is function-scoped and the `D→th` rename is dropped; the dense oracle gets
    `Uo→Do, Uc→Dc, E→U` and its letter API moves to R3; B5 is the T3 test's `'U'→'B'` string map.
  - B7's call wording.

**Draft 2 (Nick's rulings, 2026-11-11).**

- **2.1 No exemption (D-1).** `U/D/L/R` everywhere: `probing`, `sampling_derivatives`, `apply`, `entries`, their test
  twins and all four tex notes. Each note gets one historical T4S line. The library's own jet shifts take a new symbol,
  `Λ`/`Λᵀ`, and the derivative `D` (`DΦ`) becomes the upright differential `d`, so `D` keeps one meaning (§3).
- **2.2 D-6:** the corewise frame is `(D, G, G, G)`. The residual case, the T3 pages, is §3.3.
- **2.3 D-7:** `P`/`Q` → `L`/`R` wherever they name frame cores, the cited §6.3 substitution included.
- **2.4 Sentinels (D-10)** implement the simultaneous map in every pass (§1.3).
- **2.5 Critic r1 M1:** tier E now takes every free-prose store word, with per-file counts, and the residual gate K9
  is added (§1.1, §5.3).
- **2.6 Critic r1 M2:** tier C is derived from every CL row (§1.2), and the R1 gate proves none is unassigned. The
  `w` widths and the test modules' own letters are in.
- **2.7 Critic r1 M3:** the battery gets slack padded paths, sharing and tied paths, a `K` stack and the stacked uniform
  twin (§5.1).
- **2.8 Critic r1 minors and nits.**
  - Battery: the adapter's tree test; host-clock fields excluded; all 15 `update` subsets.
  - Token map: KEEP_BARE re-derived over doctests; `down_tt_ranks`; explicit scoped lists; `nU`→`n` for a UT3 point's
    Tucker rank.
  - Script: the docstring partition; Python pinned to 3.11 (f-strings).
  - Elsewhere: the migration snippet's `_dot` forms; the `dU` family (§3.3); `typ.Collection[str]`; loop variables.
  - Session: `examples/` is in the checks, and R5 runs every example.
- **2.9 R4 split** into R4a (frame, padded and uniform prose) and R4b (sampling, jets, shifts), with a review stop.

**Draft 1** corrected draft 0's counts and claims: the silent swaps do not fail a shape check; Ψ instead of a
pre-rename; D-4 impossible for `padded_start`. It also added the token map, the scripts, the battery and the checks.

## 0. The decision and the principle

Nick, 2026-11-10 (HANDOFF top block). A T3 is a TT chain with a Tucker factor hanging below each core. Its frame holds
one store of orthonormal cores per side of every internal edge. **A store is named for the side of its edge that its
cores occupy**, with the chain drawn on top:

- the TT bonds have a **left** and a **right** store;
- each Tucker edge has an **up** store, the TT-shaped cores above it (up-orthogonal, `up_orthogonalize_tt_cores`);
- and a **down** store, the Tucker factors below it (down-orthogonal, `down_orthogonalize_tucker_cores`).

Every orthogonalization and SVD name already has the right adjective (inv 2 §7). The 2026.4.0 words instead named the
direction a Tucker-edge core's edge points: leaf → core points up (BR §0). That is the opposite of the left/right
convention, so **the words flip**.

Consequences:

- **No letters in the API.** `update` takes store words. The widths are `n_down, n_up, r_left, r_right`: old `nU` (the
  Tucker store) is `n_down`, and old `nD`/`nO` is `n_up`.
- **Letters stay in diagrams and equations:** `D` (down, the Tucker factor), `U` (up, TT-shaped), `L`, `R`,
  everywhere (Nick, 2026-11-11). T4S is historical: internal consistency beats consistency with the paper. Whatever
  the new letters collide with is renamed instead (§3).
- **Physical order is unchanged.** `T3Frame.data` and every rank, mask, weight and uniform tuple keep their positions,
  which now read `(down, up, left, right)`. `save`/`load` key arrays by position (`'f%d_%d'`, `ufv_save`), so
  2026.4.0 files load unchanged (inv 1 §5).
- **Same-spelling names swap meaning (§2).** This is accepted, and the release notes list them first. There are no
  aliases. The house rule (2026.2.0, 2026.3.0) is that a breaking release breaks loudly once, and a meaning swap
  cannot be aliased anyway.

## 1. The mapping

### 1.1 One simultaneous mapping, five tiers

The two vocabularies overlap: old up is new down, and `D` already marks the TT store in `T3Frame`'s docstring. So every
substitution is **simultaneous**: one lookup per token and one pass per file, never a sequence of replacements.

- **A — identifiers.** Exact compound tokens, safe everywhere. They apply as NAME tokens in Python and `\b`-bounded in
  comments, docstrings, string literals, `docs/*.md|rst`, `docs/contributor/*`, `README.md` and `CLAUDE.md`. They never
  apply in LaTeX or in CHANGELOG history.
- **B — bare words.** Only NAME tokens `up`/`down` (in code and in doctest examples) and the exact literals
  `'up'`/`'down'`. Prose words are never scripted: of 1,038 bare tokens in the tree, 208 are not store names (BR §1.1).
- **C — letters.** Per-file or per-line-range NAME-token maps, never global. `U` is also the SVD factor in 9 modules
  (inv 2 C3); `D` is the dense frame in the test oracles; `Ud` is "u dense" in `oracle_sweeps/o1_ragged.py`.
- **D — text patterns.** Exact frame tuples and listings in comments, docstrings and docs.
- **E — hand rows.** Every inventory row marked HAND, AMBIG, CONV, CONFLICT, BACKWARDS, MSG or EXPECTED MESSAGE, with
  the full text the inventory gives, plus all of inv 3 (docs are hand edits except tier A).
  - **Free-prose store words**: every STORE-WORD row no A/B/D tier rewrites (critic r1 M1). These are a bare `up`/`down`
    in a comment, docstring or non-exact string, e.g. `uniform_padded_frames.py:563` "up edges", the subTest label
    `'in-block down'` (`test_padded_frame.py:296`), and `# 'down' | 'left' | 'right'` (`padded_frames.py:143`).
  - inv 2 §9 has 175 such rows (some are tier-D listings): `fv_operations` 25, `padded_frames` 20, `ufv_operations` 18,
    `uniform_padded_frames` 17, `utv_operations` 16, `geometry` 14, `ufv_conversions` 10, `uniform_fitting` 9,
    `tv_operations` 7, `ufv_masking` 7, `apply` 6, `ut3_sampling` 5, `entries` 4, `ranks` 4, `probing` 3,
    `sampling_derivatives` 3, `t3_linalg` 3, `utv_sampling` 3, `sharing` 1.
  - inv 1 §7 has about 118 such rows: 41 frontend (`uniform_frame_variations_format` 19, `manifold` 10,
    `uniform_manifold` 7, `frame_variations_format` 5) and 77 tests (`test_uniform_frame_variations_format` 17,
    `test_padded_frame` 13, …).
  - R1's hit table gives the exact list, and K9 (§5.3) catches what any list misses.

### 1.2 The token map

| tier | old → new (one simultaneous pass) | scope |
|---|---|---|
| A1 renamed | `up_tucker_cores`→`down_tucker_cores`, `down_tt_cores`→`up_tt_cores`, `up_tucker_cores_dot`→`down_tucker_cores_dot`, `down_tt_cores_dot`→`up_tt_cores_dot`, `up_tucker_supercore`→`down_tucker_supercore`, `down_tt_supercore`→`up_tt_supercore`, `masked_up_tucker_supercore`→`masked_down_tucker_supercore`, `masked_down_tt_supercore`→`masked_up_tt_supercore` | everywhere |
| A1 shifts (§3.2) | `_shift_down`→`_jet_shift`, `_shift_up`→`_jet_shift_transpose`, `_shift_down_pair`→`_jet_shift_pair`, `_shift_up_pair`→`_jet_shift_transpose_pair` | `backend/sampling_derivatives.py` (33 tokens) and text mentions |
| A1 aliases, **one-way only** | `up_orthogonalize_tucker_cores`→`down_orthogonalize_tucker_cores`, `down_orthogonalize_tt_cores`→`up_orthogonalize_tt_cores` | `backend/fv_conversions.py` only (181–185, 220, 253). The targets are public method names, so no reverse entry exists anywhere |
| A1 labels | `up_tucker`→`down_tucker`, `down_tt`→`up_tt`, `up_tucker_ranks`→`down_ranks`, `down_tucker_ranks`→`up_ranks`, `down_tt_ranks`→`up_ranks` (`uniform_frame_variations_format.py:203, 665`) | text only (comments, docstrings, docs) |
| A2 swaps (a ↔ b) | each of `up_ranks`, `up_mask`, `variations_up_mask`, `var_up_mask`, `up_weights`, `up_weight_supercore`, `up_sc`, `up_cores`, `up_supercore`, `up_inds`, `up_r`, `v_up_r`, `up_out`, `up0`, `up3`, `up4`, `vup`, `mup`, `up_shapes`, `up_stack_shapes`, `up_slices`, `up_masks`, `up_widths`, `up_mask_frame`, `up_mask_v`, `_up`, `_up_mask`, `_up_edge`, `_up_message`, `_u_up_messages`, `_UP_R`, `bad_up`, `masked_up`, `point_up`, `variations_up` ↔ its twin with `up`/`UP` → `down`/`DOWN`; also `test_unequal_old_up_masks_in_a_group`→`test_unequal_old_down_masks_in_a_group` | everywhere |
| A3 widths | `nU`→`n_down`, `nD`→`n_up`, `nO`→`n_up`, `rL`→`r_left`, `rR`→`r_right`. Applied as the family `\b(nU\|nD\|nO\|rL\|rR)(\d*)([ijkdnlrxy]?)(?![A-Za-z0-9])` → `NEW[g1] + g2 + ('_' + g3 if g3 else '')`, so `nU0`→`n_down0`, `nUi`→`n_down_i`, `rL1i`→`r_left1_i`, `rR(i+1)`→`r_right(i+1)`, `nU_pad`→`n_down_pad`, `rLl`→`r_left_l`. Explicit: `nDs`→`n_ups`, `nnU`/`nnD`→`nn_down`/`nn_up`, `rrL`/`rrR`→`rr_left`/`rr_right`, `_NU`/`_ND`/`_RL`/`_RR`→`_N_DOWN`/`_N_UP`/`_R_LEFT`/`_R_RIGHT`, `sum_nD`→`sum_n_up` | everywhere but LaTeX; under D-3 (recommended) also the plain-TT bonds. **Exception:** where `nU` is a UT3 point's Tucker rank, it becomes the house UT3 symbol `n` (inv 1 C7; `n` is free in each scope): `optimizers.py:176–180`, `tests/test_chunk_size_estimator.py:274–276`, and `sampling_derivatives.py`'s chunk estimators (1819–1824, 1870–1912), where `nO = nU` becomes `n_up = n` |
| A3 scoped | for `X` in `k, m, p, i, w, W`: `XU`→`X_down`, `XD`→`X_up`, `XL`→`X_left`, `XR`→`X_right`. That is `kU kD kL kR`, `mU mD mL mR`, `mU0 mD0 mL0 mR0`, `mU1 mD1 mL1 mR1` (digit kept: `m_down0`), `pU pD pL pR`, `iU iD iL iR`, `wU wD wL wR` (the per-store widths, `_uniform_padded_frame_testing.py:43–85`), `WU WD WL WR` | `backend/uniform_padded_frames.py`, `tests/backend/_uniform_padded_frame_testing.py`, `tests/backend/test_uniform_padded_frame.py` only (`mL`/`mR` are modes in `t3_linalg`) |
| B1 bare | NAME `up` ↔ `down`, in code and in doctest examples | everywhere except KEEP_BARE: `tests/test_fitting.py:1148–1152` (a tangent), `tests/backend/_curvature_oracles.py:529–536` (the ξ lambda), `t3toolbox/fitting.py:290–291` (a doctest tangent). R1 re-derives the list over doctest lines (BR's eight were code tokens only) |
| B2 literals | STRING tokens exactly `'up'` ↔ `'down'` (either quote) | everywhere: `_STORE_NAMES`, the `_SEED_TAGS` keys, `_seed_rows` arguments, the test copy `tests/backend/test_padded_frame.py:91–92`. The `_SEED_TAGS` integers stay with the physical store (`{'down': 0, 'up': 1, …}`), so the seeds are bitwise unchanged. Messages built with `%s` from `_STORE_NAMES` flip with it |
| B3 scoped | NAME `dn`→`up` (`dn` is a difference norm everywhere else) | inside the enclosing function or doctest of `uniform_frame_variations_format.py:1029`, `tests/test_uniform_frame_variations_format.py:250, 292, 323`, `tests/test_dispatch.py:1680, 2840`, `tests/test_weighted.py:211` |
| B4 scoped letter strings | STRING `'U'`→`'D'`, `'O'`→`'U'`, `'P'`→`'L'`, `'Q'`→`'R'` (simultaneous) | `tests/test_jet_recurrence.py` only: the 34 dict keys `p['O']`, `p['P']`, `p['Q']`, … (its `dict(U=…, O=…)` keywords are NAMEs, tier C). Elsewhere `kind == 'U'` is §3.3's `B` |
| B5 scoped T3 strings | STRING `'U'`→`'B'` (the T3 Tucker kind) | `tests/test_tucker_tensor_train.py:2236, 2255, 2283, 2301, 2370, 2391` (`kind == 'U'`, `replace(x, 'U', …)`, `('U', dU)`), together with its `dU`→`dB` (§3.3), so call site and comparison flip at once |

Tier C, letters (NAME tokens, simultaneous within each row; inv 2 §8's per-file maps, amended):

| file (range) | map |
|---|---|
| `backend/padded_frames.py` | `U→D, O→U, Ui→Di, U0→D0, U1→D1, U_new→D_new, oU→oD, o_U→o_D, PhiU→PhiD, XiU→XiD, XiUs→XiDs, PhiO→PhiU, XiO→XiU, bO→bU, O_P→U_P, moves_U→moves_D, moves_O→moves_U`; D-5: `D→Psi, D_L→Psi_L, D_R→Psi_R, DL→PsiL, DR→PsiR, DU→PsiD, DUs→PsiDs` |
| `backend/uniform_padded_frames.py` | the row above, plus `O0→U0, O1→U1, old_U→old_D, new_U→new_D, new_O→new_U, seeds_O→seeds_U, real_O→real_U, P_O→P_U, rowU→rowD, rowO→rowU, pinU→pinD, pinO→pinU, selU→selD, selO→selU, mvO→mvU` (`Us` is an SVD factor, unmapped) |
| `backend/fv_operations.py` | `UU↔DD, U→D, D→U` (here `D` is the old down core, inv 2 C2) |
| `backend/fv_conversions.py` | `D→U, left_UU→left_DD, right_UU→right_DD` |
| `backend/ufv_conversions.py` | `U→D, um↔dm, uc→c_down, dc→c_up` (`dc` is the house `dataclasses` alias in 7 modules, so no swap) |
| `backend/ufv_operations.py` | `um↔dm` (`dc` here *is* `dataclasses`) |
| `backend/tv_operations.py` | `U→D, O→U` |
| `backend/utv_operations.py` | `U→D, O→U, BU→BD, BU1→BD1` |
| `backend/t3_linalg.py` (660–760) | `U→D, O→U, BU→BD`; outside the range `U` is a T3's Tucker core and stays |
| `backend/geometry.py` (860–892) | `U→D, O→U` |
| `frame_variations_format.py` (483–573) | `UU↔DD, U↔D` (the `__post_init__` locals; the message at :556 changes with them, see §4.6) |
| `uniform_frame_variations_format.py` | `uc→c_down, dc→c_up, UU_good↔DD_good, UM_good↔DM_good` |
| `backend/probing.py` | `U→D, O→U, P→L, Q→R, Ud→Dd, Pd→Ld, rev_Q→rev_R` (no `L`, `R` or `D` local exists there) |
| `backend/sampling_derivatives.py` | `U→D, O→U, P→L, Q→R, P_dot→L_dot, Q_dot→R_dot, P_cores→L_cores, P_cores_dot→L_cores_dot, Pxi→Lxi, dPxi→dLxi, Qxi→Rxi, rev_Q→rev_R, Ud→Dd, Od→Ud, mO→mU, On→Un, dmO→dmU, dOn→dUn, Odxi→Udxi` (no `L`, `R` or `D` local exists there) |
| `backend/apply.py`; `backend/entries.py` (361–363) | `O→U, Q→R`; `D→n_cores` (the core count, beside the Tucker cores) |
| T4S-derived Tucker names (§3.3) | `Udot→Ddot, Udot_g→Ddot_g, dU→dD, dU0→dD0, dU_v→dD_v, dU_t→dD_t, dU_tilde→dD_tilde, dU_tildes→dD_tildes, gU→gD, gYU→gYD, Ug→Dg`, and `Ud→Dd` where it is the Tucker direction (`_curvature_oracles.py`, `test_curvature.py`, `o1_uniform.py`, `utv_operations.py`; not `o1_ragged.py`, where `Ud` is "u dense"). Files: `manifold.py`, `tv_operations.py`, `utv_operations.py`, `utv_sampling.py`, `sharing.py`, `apply.py`, `probing.py`, `sampling_derivatives.py` and the tests in the R1 scan. The `G` family (`dG`, `Gdot`, `Gd`) is the library's TT letter and stays. Exceptions: `tv_operations.py:292–331`'s `dU = tt_variations[i]` is a TT variation, so it becomes `H` (hand row); `test_tucker_tensor_train.py` uses `dB` (§3.3) |
| T4S test twins | `tests/_jet_trs_twins.py`: `O→U, P→L, Q→R, P_cores→L_cores, rev_Q→rev_R`. `tests/test_jet_recurrence.py` (166–176 is an RNG helper `R`): `U→D, O→U, P→L, Q→R, R→rand`, plus B4. `Q→R` is mapped only in the files listed; `test_probe_derivatives.py:488, 497` keeps its RNG `R` |
| tests that mirror a mapped module | `tests/backend/_uniform_padded_frame_testing.py`, `tests/backend/test_uniform_padded_frame.py` and `tests/backend/test_padded_start.py` (290–307): `U→D, O→U, Uu→Du, Ou→Uu`. `tests/backend/test_padded_frame.py` (111, 135–139, 205–216, 619–629): `U0→D0, O0→U0, U→D, O→U` (inv 1 §7 gives `D0, U0, L0, R0 = old`). `tests/test_uniform_frame_variations_format.py`: `U_r→D_r, U_u→D_u` (958–959 by row). `tests/test_dispatch.py`: `U_3→D_3` |
| every other CL row (critic r1 M2) | The rule map `U→D, O→U, P→L, Q→R` (plus letter-suffixed locals), **scoped by the enclosing function** of each row (as B3; a line scope breaks def/use pairs), after a per-function collision check, and for `P`/`Q` only where §3.4 classifies them as frame cores. Rows: inv 1 §7 (`test_frame_variations_format.py`, `test_manifold.py:217–218`, `test_uniform_manifold.py:1232`) and App. A (`test_curvature.py`, `test_uniform_curvature.py`, `_curvature_oracles.py`, `test_tv_operations.py`, `test_sharing.py`, `test_manifold.py:225–226, 470`, `test_optimizers.py:533`, `test_zero_reference.py:94`, `test_contractions_sharding.py:199`, `test_fitting.py:8`). In `test_uniform_curvature.py` the `U` (:99–104, 249–250) never shares a function with the `D` family (`D`, `D_r`, `Dr`, `Du`, `Dth`, `Dv`: raw directions and dense tensors), so nothing is renamed first. Gate: R1's hit table assigns every CL row of inv 1 §7, App. A and inv 2 §8 to a row of this table |
| dense oracles | `tests/backend/_padded_frame_oracles.py` and `test_padded_start.py` (566–580), function-scoped: `D→dense, oldD→old_dense, curD→cur_dense, U→D, Uo→Do, Uc→Dc, E→U, Eo→Uo, Ec→Uc`. `D` is the dense frame there, and `E` the core → leaf basis, now the up edge. `Uc` in the SVD helper (:54) and `Uy` are SVD factors and stay. The oracle's letter API (`dense_pass(update='ULRO')`, `'U' in update`, the `zip('ULRO', …)` keys) goes to R3 with the library's. For one convention, the remaining dense-frame names become words: `startD`, `Da`, `Db` (`_padded_frame_oracles.py:123–178`) and `startD` and `D` in `test_padded_start.py:478–481` → `start_dense`, `dense_a`, `dense_b`, `dense` |
| letters kept | the SVD and QR factors (`U` of `UΣVᵀ`, `Q, R = qr(…)`, `Us`); the completion `P`/`P⁺` and the P-block; `o1_ragged.py`'s `Ud`/`Dd` (dense arrays); every letter outside the rows above |

Tier D, text patterns (R4a/R4b; comments, docstrings and docs; the tex notes by hand in R5):

- `(U, O, P, Q)`, `(U,O,P,Q)` and `(U, O, L, R)` → `(D, U, L, R)` (D-7; 65 lines in code and docs, 2 in the notes).
  The hybrid `(D, U, P, Q` (inv 2 §4's proposals, e.g. `geometry.py:602, 707, 893`) → `(D, U, L, R`.
- `(U, G, G, G)` and `(U,G,G,G)` → `(D, G, G, G)` (D-6; 62 lines).
- `(P, Q, O) → G` and `P,Q,O → G` → `(L, R, U) → G`, written "T4S §6.3's substitution" where cited (13 lines: the
  11 inv 2 C4 rows, inv 1 C3, `CLAUDE.md:115`, `transposes.md:37`).
- `(up, down, left, right)` → `(down, up, left, right)`, and the same for `(up, down, left[:-1], right[1:])` and
  `up/down/left/right`.
- The jet-shift and derivative notation of §3.2: `D x`, `U x`, `D^r`, `U^r`, `U^u`, `U^k`, `[U^` → `Λ x`, `Λᵀ x`,
  `Λ^r`, `(Λᵀ)^r`, …; `DPhi`, `D^2 Phi`, `DΦ` → `dPhi`, `d^2 Phi`, `dΦ`. These are in `sampling_derivatives.py`
  (41 lines), about 13 docstrings and comments, `deferred_and_rejected.md:134` and `curvature_internals.md:310`.

Tier E: inv 2 §3 (25 rows) and §4 (73 rows); inv 1 §3 (35 sentences) and its flagged main-table rows; inv 3 §5–§7 and
App. A.

### 1.3 The scripts (in `dev/`, never imported by the library; R1 writes them)

```python
# dev/store_rename_apply.py
ONE_WAY:    typ.Dict[str, str]                                    # A1 + A1 aliases; A1 labels flagged text-only
SWAPS:      typ.Tuple[typ.Tuple[str, str], ...]                   # A2, each pair applied both ways
WIDTH_RE:   typ.Pattern                                           # A3 family, plus WIDTH_EXPLICIT: typ.Dict[str, str]
SCOPED:     typ.Dict[str, typ.Tuple[typ.Tuple[typ.Optional[typ.Tuple[int, int]], typ.Dict[str, str]], ...]]
                                                                  # file -> ((line range | None, map), ...): A3 scoped, B3, B4, C
KEEP_BARE:  typ.Tuple[typ.Tuple[str, int, int], ...]              # B1 exclusions: (file, first line, last line)

ENTRIES:    typ.List[typ.Tuple[str, str]]                     # every (old, new) pair of every tier, indexed k = 0, 1, ...
                                                              # (width-family and text-pattern matches registered as found)

def sentinelize_python(path: str, source: str) -> str  # PASS 1. One tokenize pass; each token edited in place by
                                                     # position (never untokenize). An old token in scope becomes the
                                                     # sentinel 'ZQX%04d_XQZ' % k (no old text inside, so no later
                                                     # pass-1 regex matches in it). NAME: ONE_WAY|SWAPS|widths|B1|
                                                     # SCOPED[path]. COMMENT/STRING: the \b-bounded A tiers; STRING
                                                     # 'up'/'down': B2. Doctest examples inside docstrings (>>> / ...
                                                     # lines) are tokenized as code, so B1, B3 and C reach them too
                                                     # (e.g. uniform_frame_variations_format.py:1023-1032).
                                                     # PARTITION: example lines go by the code path only; prose and
                                                     # expected output by the text path only (never both, or an A2
                                                     # swap applies twice and cancels)
def sentinelize_doc(path: str, text: str) -> str      # PASS 1 for .md/.rst, README.md, CLAUDE.md: the A tiers only
def resolve(text: str) -> str                         # PASS 2, global: ZQX(\d+)_XQZ -> ENTRIES[k][1], asserting each
                                                     # key's sentinel count equals its hit-table count
def hit_table(paths: typ.Sequence[str]) -> str        # --dry-run: sentinel counts per key x file after PASS 1,
                                                     # reconciled with inv 2 §8 / inv 1 §7
def bijection_errors(before: str, after: str) -> typ.List[str]   # per file and swap pair (a, b):
                                                     # count_after(a) == count_before(b) and count_after(b) == count_before(a)
```

The script asserts `sys.version_info[:2] == (3, 11)` (the `t3toolbox` env). On Python 3.12 and later, tokenize splits
f-strings into FSTRING_MIDDLE + NAME, so `up_ranks={self.up_ranks}` (`frame_variations_format.py:288`,
`manifold.py:211`, `uniform_manifold.py:184`) would get its NAME renamed and keep its label. B2's repr comparison is
the net.

**Sentinels implement the simultaneous map (D-10).**

- Pass 1 rewrites only old tokens, each into a sentinel that matches no old or new token. Pass 2 rewrites only
  sentinels. So no token is touched twice, whatever tool or order runs either pass, and a surviving `ZQX` fails K0
  (§5.3).
- **Where they are required:** every text rewrite (tier A in prose, tier D, the docs, the hand passes over the notes),
  because those are regexes run in several passes.
- **In the Python NAME pass**, one lookup per token is already simultaneous (critic r1 R-4). Sentinels add no safety
  there. They are kept anyway so that R2 reviews *one* pass-1 diff: every token the rename will touch, and nothing
  else.
- They settle ordering only. *Which* tokens are stores is the tiers' and scopes' job, checked by the hit table,
  the bijection and K9.
- The pass-1 tree is never run or committed.

The validation that replaces the letter check, in `backend/padded_frames.py` (also imported by
`uniform_padded_frames.py`, the D-46 sibling convention):

```python
_STORE_NAMES = ('down', 'up', 'left', 'right')               # a frame's stores in physical order (T3Frame.data)
_COLOURS = (('down', 'left'), ('right', 'up'))               # the two-colour passes (alternate=True), in pass order

def _update_stores(
        update:  typ.Collection[str],   # store words, any order, e.g. ('down', 'left'); never a str
) -> typ.Tuple[str, ...]:               # the same stores in _STORE_NAMES order: one jit static per set
    """``update`` validated and canonicalized; ``ValueError`` on a str, an empty or repeating collection, or a non-store."""
    if (isinstance(update, str) or not update or len(set(update)) != len(update)
            or not set(update) <= set(_STORE_NAMES)):
        raise ValueError("update must be a non-empty tuple of the store words 'down', 'up', 'left', 'right' "
                         "without repeats, e.g. ('down', 'left'); got %r" % (update,))
    return tuple(store for store in _STORE_NAMES if store in update)
```

`tv_align_padded_frame`, `utv_align_padded_frame`, `T3Tangent.align_padded` and `UT3Tangent.align_padded` take
`update: typ.Collection[str] = ('down', 'up', 'left', 'right')` (store words, any order; the rest transported). A
generator raises `TypeError` at `len`, as any non-collection does. The two
steps call `_update_stores` first, and the uniform kernel receives the canonical tuple as its static. The loops pass
`_STORE_NAMES` and `_COLOURS[0]`/`_COLOURS[1]`. The truncation check drops its `'UOLR'` column:
`for store, old, new, own in zip(_STORE_NAMES, …): … if b < c and store not in update`. The message keeps the
substrings `update must be` and `without repeats`, which the tests' regexes match.

### 1.4 Applying tier E (critic r2 N2)

The inventories predate Nick's rulings: their proposals keep `P`/`Q`, the `U` family and `(U, G, G, G)`. For example,
inv 2 §4 gives `(D, U, P, Q, shape, …)` at `geometry.py:602, 707, 893` and "frame's ``D`` and ``P``" at
`ut3_sampling.py:374`. A tier-E row is therefore applied in three steps:

1. **Locate** the line by content, not number: the unique line equal to the R2 image of the row's "current" text,
   computed by the script's line function. Zero or several matches fail.
2. **Replace** it with the row's proposed text (already in the new spelling; never re-mapped).
3. **Layer the rulings** on top:
   - L1: frame `P`/`Q` → `L`/`R`, by the §3.4 classification.
   - L2: the `U` family → `D`, or `B` on the T3 pages (§3.3).
   - L3: `(U, G, G, G)` → `(D, G, G, G)`.
   - L4: the `Λ` and `d` notation (§3.1–3.2).
   - L5: D-5's `Ψ`/`F`, and D-8's widths as words. This replaces inv 1's letter-swapped diagram labels: `nD0` →
     `n_down0`.
   - L6: the reworded message directions (§3.3).

On a line that carries both a tier-E row and a tier-D pattern, **tier E runs first**, because its locator needs the
line's R2 image. Tier D then runs over every line; its patterns include the hybrids, so it is idempotent on correct
text. The reviewer checks the layered text, and K4, K7 and K8 check the outcome.

## 2. The compatibility surface

Full tables: inv 1 §2 (frontend) and inv 2 §2 (backend). Counts from BR §1.3, checked against the signatures.

- **Loud** (`AttributeError` or `TypeError` on stale code):
  - 4 dataclass fields, also constructor and `dataclasses.replace` keywords: `T3Frame.up_tucker_cores →
    down_tucker_cores`, `T3Frame.down_tt_cores → up_tt_cores`, and `UT3Frame`'s two supercores.
  - 18 backend core keywords in 14 functions (8 `probing.compute_*` with 10 keywords, 6
    `sampling_derivatives.compute_*_jets` with 8): `up_tucker_cores[_dot] → down_tucker_cores[_dot]`,
    `down_tt_cores[_dot] → up_tt_cores[_dot]`.
  - 32 width keywords in 8 functions. Frontend: `UT3Frame.from_t3frame`, `UT3Variations.from_t3variations`,
    `UT3FrameWeights.from_t3frameweights`, `UT3Tangent.from_t3tangent`. Backend: `ufv_conversions.t3frame_to_ut3frame`,
    `t3variations_to_ut3variations`, `t3frameweights_to_ut3frameweights`, `ufv_masking.ufv_make_frame_masks`.
  - 12 width properties: `nU/nD/rL/rR` on `UT3Frame`, `UT3Variations` and `UT3FrameWeights`.
  - `update` on 4 functions: a letter string raises `ValueError`.
- **Silent** (same spelling, swapped meaning), 22 sites:
  - 18 attributes on 7 classes: `up_ranks`/`down_ranks` (`T3Frame`, `T3FrameWeights`, `UT3Frame`, `UT3Variations`,
    `UT3FrameWeights`), `up_weights`/`down_weights`, `up_weight_supercore`/`down_weight_supercore`,
    `UT3FrameMasks.up_mask`/`down_mask` and `UT3VariationsMasks.variations_up_mask`/`variations_down_mask`. The last
    three pairs are also constructor keywords.
  - 4 backend keywords: `up_ranks=`/`down_ranks=` of `ranks.frame_has_minimal_ranks` and `ufv_make_frame_masks`.
  - **Nothing catches these.** Both stores give int tuples or masks of length `d`, and at a minimal frame the ranks
    agree, so stale code returns plausible numbers and is wrong only at slack (or at unequal uniform padded widths).
    The release notes therefore lead with this list.
- **Unchanged:** the positional order of every tuple, `uniform_structure` and every `from_*` signature (positional
  width calls still work); `save`/`load`; every orthogonalization and SVD name; every optimizer keyword (`padded_*`,
  `cone_*`, `alternate=` carry no store word); every HISTORICAL contraction string (inv 2 §6); every number (§5.1).
- **Text only:** the reprs (`T3Frame(… down_ranks=…, up_ranks=…)`, `UT3Frame(… n_down=…)`) and store-naming messages.
- **Downstream:** T3Polynomial has one positional unpack (`scripts/g01_g0_common.py:121`), which keeps working; its
  locals are renamed after the release. `implicit_probing` does not import T3Toolbox, and the research repo makes no
  library call with these names (its prototypes keep the old words in their own locals).

## 3. Letters: one alphabet (Nick, 2026-11-11)

T4S is historical, and internal consistency within T3Toolbox beats consistency with the paper. **The frame letters are
`D`, `U`, `L`, `R` everywhere**, with no exempt file: code, docstrings, tests, docs and the four tex notes. That
includes the sampling kernels (`probing`, `sampling_derivatives`, `apply`, `entries`) and their test twins
(`tests/_jet_trs_twins.py`, `tests/test_jet_recurrence.py`). T4S's `U`, `O`, `P`, `Q`, `δU` go wherever they appear.

One historical line records them: *"T4S writes `U`, `O`, `P`, `Q` (and `δU`, `δG`) for our `D`, `U`, `L`, `R` (and
`V`, `H`)."* It appears only in these places: each tex note's notation paragraph; `docs/frame_variations.md` § "Four
rank stores" (inv 3 §1(a)'s T4S note); the glossary's T4S entry; `CLAUDE.md`'s notation map; and the paper ↔ code maps
(`entries_apply_probe.md` §8, `batching_and_stacking.md` §8), which keep the paper's letters as the correspondence
itself.

### 3.1 What the new letters collide with, and what is renamed instead

- **No `D` is a diagonal.** Every diagonal is spelled `diag(…)`; the one `D⁻¹` (`tests/backend/test_optimizers.py:344`)
  is in an unrelated equation (inv 3 §2, inv 1 C5).
- **The jet shifts.** These are the library's own notation, not T4S's: forward `D x = [t x^(t-1)]_t` and reverse
  `U x = [(s+1) x^(s+1)]_s`. They appear in `sampling_derivatives.py` (41 lines), `deferred_and_rejected.md:134` and
  the exact-Hessian note (`\Dsh`/`\Ush`, 14 lines). Both collide with the store letters inside single equations: after
  the rename, `(mu^(j) O)·[U^u sigma~]^(j)` would read `(mu^(j) U)·[U^u sigma~]^(j)` (inv 2 C2, C3). They are renamed
  to `Λ`, `Λᵀ` (§3.2, D-1).
- **The derivative `D`.** It appears as `DΦ`, `D²Φ`, `D_ζP` (21 lines of the exact-Hessian note, beside its frame
  cores), `D^t` (`symmetric_probe_derivatives.tex:111`), `DPhi`/`D^2 Phi` in 13 docstrings and comments, and the
  curvature page's `DΦ[Lv] = Ev`. It becomes the upright differential: `\mathrm{d}\Phi`, `\mathrm{d}^t` in LaTeX, and
  `dPhi`, `d^2 Phi`, `dΦ` in text (D-11). The italic `d` for the mode count is distinguished by font in LaTeX and is
  never followed by `Φ`/`Phi` in text.
- **The padded-frame material** (D-5): the products and message `D` become `Ψ`, and the edge frame `U_{x→t}`, `U′`
  becomes `F_{x→t}`, `F′`.
- **A T3's own Tucker cores**, which T4S writes `U`: the corewise frame `(U, G, G, G)` at 62 sites, `θ = (U, G)`,
  `U̇`, `U_g`, `δU`, and the code names `Udot`, `dU*`, `Ud` (inv 2 C1, inv 3 C13–C15). These become `D`; §3.3 has the
  rule.
- **Kept, because they are other meanings, not frame letters:** the SVD factor `U` in `UΣVᵀ` (inv 2 C3: 9 modules,
  100+ tokens; the padded-frame note's matrix-case proof), QR's `Q, R`, the completion `P⁺` and the P-block, the
  derivative direction `P^t`/`pp`, and `utv_sampling`'s "U3". Tier C never maps a letter outside its listed files and
  ranges.

### 3.2 The jet-shift symbol (D-1, replaced)

Candidates were checked against the code, the docs, the notes and the glossary's symbol table:

- **`S`/`S̃`:** taken. The glossary lists four meanings: the sampling operator `S(x)`, the hint `S`, the frame factors
  `S_i` and the T/S transforms. `S_i` is also in the exact-Hessian note that defines the shifts (21 uses).
- **`J⁺`/`J⁻`:** read as the sampling Jacobian `𝒥`/`𝒥ᵀ`. That symbol appears in the jet kernels' own docstrings
  (`probing`, `apply`, `entries`) and in the note's `\Jac`.
- **`Γ`, `Υ`, `Δ`, `Ψ`, `E`, `M`, `T`:** `Γ` is the sharing Gram `Γ_i`, `Υ` looks like the content `Y`, `Δ`/`δ` are
  variation notation, `Ψ` is taken by D-5, and `E`, `M`, `T` are the captured energy, the proof matrix and the tangent
  space. (`entries.py:361`'s local `D`, the core count, becomes `n_cores` under tier C.)
- **`Λ` and `Λᵀ`, with the note's Lagrangian renamed: recommended.**
  - `Λ` has one other use: the exact-Hessian note's Lagrangian `Λ(θ) = f(Φ(θ)) − ⟨grad f(X), Φ(θ)⟩`
    (`exact_hessian_two_hole.tex:255–275, 1019`, 7 uses). That is the very note that defines the shifts (critic r2 N1).
    It has no other use in the code, the docs or the glossary; `curvature_internals.md:23` names the Lagrangian in
    words.
  - The Lagrangian becomes `ℓ` (`\ell`, unused in that note; `\mathcal L` is `\Lift`), and the shift keeps `Λ`.
  - One letter plus its transpose states what the code relies on: the reverse shift *is* the forward shift's adjoint in
    the plain inner product, truncation included,
    `⟨Λx, y⟩ = Σ_{t≤m} t x^(t-1) y^(t) = Σ_{s<m} (s+1) x^(s) y^(s+1) = ⟨x, Λᵀy⟩`.
  - Caveats: `Λ` often names a diagonal (no such use in scope), and lowercase `λ` is the ridge weight (3 uses in the
    note), distinguished by case.
- **Alternative:** a spelled operator `sh`/`shᵀ` (`\mathsf{sh}`). It needs no Lagrangian rename, but it is heavier in
  the correlation formulas, and `sh` is a shape local in the uniform tests.

With `Λ`, the examples read `dxi~^(u) += (mu^(j) U)·[(Λᵀ)^u sigma~]^(j) + [(Λᵀ)^u tau~]^(j)·(U nu^(j))` and
`eta = Σ_r (mu^(r)/r!·G)·Λ^r nu`.

The code names follow (tier A1 shifts). Their up/down words are jet order, and after the rename they would read as
store words. `test_curvature.py`'s `c_up` becomes `c_next`. The note's macros become `\Sh` = `\Lambda` and `\Sh^{\T}`.

### 3.3 Fixed rules

- **Frames:** a frame tuple is `(D, U, L, R)`, and the corewise frame is `(D, G, G, G)` (D-6, Nick).
  - Whatever occupies or moves the down slot is `D`. That covers the raw Tucker directions and the T4S-derived Tucker
    variation names (`Ḋ`, `D_g`, `δD`; in code `Ddot`, `dD…`, `Dd`, `gD`), so pairs stay symmetric with the
    unchanged `G` family: `(dD, dG)`, `(Ddot, Gdot)`, `(Dd, Gd)`.
  - A misnamed variation takes the house letter instead (`tv_operations.py:292–331`'s `dU = tt_variations[i]` → `H`),
    and `:401`'s `dU_i = <O_i, core_env_i>` becomes `dD_i = <U_i, core_env_i>`.
  - A point given by a frame's stores is written in store letters, e.g. `θ = (D, L)`.
- **The T3 pages** (D-6 residual): a T3 as a tensor format keeps `B`, its letter in the README, the user guide and the
  `TuckerTensorTrain` doctests. This covers `t3m_methods.md`, `entries_apply_probe.md` §1 and §3, and
  `test_tucker_tensor_train.py` (`kind == 'U'`→`'B'`, `dU`→`dB`, its `dG` unchanged). So `D` is the down slot's letter
  and `B` a T3's own Tucker cores as stored; they meet only at the corewise frame, `(D, G, G, G)` built from `(B, G)`.
- **T4S's message directions that now read as store words** are reworded: "frame down jets"
  (`sampling_derivatives.py:1479, 1513`) → "the η jets"; "up-index" (`probing.py:773, 824`) → "the ξ index"; "down
  `eta`" (`apply.py:215`) → "the η sweep". Directions spelled as directions stay ("upward edge variables", "contract
  up", "pushing the remainder down into the Tucker core").
- **Cited T4S formulas** convert, with the citation kept, e.g. "T4S §6.3's substitution `(L, R, U) → G`" (tier D).

### 3.4 The `P`/`Q`/`U`-family inventory (R1; critic r2 N2)

The inventories classified `U`/`O`/`D` only. They kept `P`/`Q` and the `U` family by design (inv 2 M4, C1), so the
rulings need their own inventory before any slice applies them.

- **Built in R1** as `dev/store_rename_pqu_inventory.md`: every NAME token and every text occurrence of `P`, `Q`, their
  derived names (`P_cores`, `P_cores_dot`, `P_dot`, `Pd`, `Pxi`, `dPxi`, `Q_dot`, `Qd`, `Qxi`, `rev_Q`, `P_last`, …),
  `P_i`/`Q_i` in text, and the `U` family (`Udot*`, `dU*`, `gU`, `gYU` (`_curvature_oracles.py:72–75`), `Ug`, `Ud`, `U_g`,
  `U̇`, `δU`). It covers `t3toolbox/`,
  `tests/`, `examples/`, the docs and the notes.
- **Each hit is classified, with its disposition:**
  - frame core → `L`/`R`;
  - a direction or variation of the down slot → `D`;
  - a T3's own cores on the T3 pages → `B`;
  - kept: the derivative direction (`P`, `pp`, `P^t`), the completion and P-block, QR's `Q`, a pseudo-inverse
    (`test_curvature.py:215–220, 353`: `P = pinv`), an RNG helper.
- **Sites known now:**
  - frame `P`/`Q` on lines with no `U`/`O`: `geometry.py:122, 139` (`P_last`), `uniform_manifold.py:479`
    (`(P_i^L)^T dG_i^L = 0`), `_curvature_oracles.py`'s `P`, `Pd`, `Q`, `Qd` and its `Q = O = P = G`, and the
    exact-Hessian note's `:117–132, 305, 343–359` (beside `H_i = P_i ×_3 T_i`);
  - the `U` family: `backend/fitting.py:351, 374, 380, 389` (`(dU, dG)`, `theta_hat = (Udot, Gdot)`),
    `ut3_sampling.py:114, 231`, `entries.py:152, 290`, and the T3-class doctests `tucker_tensor_train.py:3870–3871,
    4072–4073` and `uniform_tucker_tensor_train.py:566–567, 631–632` (`gU, gG = x.apply_corewise_transpose(…)`: T3
    pages, so `gB`).
- **R1 gate:** every hit is classified. R2's tier C and R4's L1/L2 apply only what the classification marks.

## 4. Hand edits (R3–R5)

1. **Convention statements** (about 69: inv 2 §3 25, inv 1 §3 35, inv 3 9). Rewrite them from the rule, not by word
   swap. Their home is `docs/frame_variations.md` § "Four rank stores", which gets inv 3 §1(a)'s paragraph, picture and
   T4S note. Docstrings state the rule in one clause and link to it.
2. **Sentences that use both senses** (edge direction and store side, inv 3 C19). Rewrite them, e.g. "the stores
   pointing toward the root (up, left)" becomes "the stores whose edges point toward the root (down, left)".
3. **Backwards adjectives, 28 lines.** Frontend: `frame_variations_format.py:1371–1372`,
   `uniform_frame_variations_format.py:1122, 1125`, `tucker_tensor_train.py:615, 2956, 2978, 3002, 3108`. Backend:
   `probing.py:646–647, 1221–1222`, `t3_orthogonalization.py:147, 171`, `ut3_orthogonalization.py:189`,
   `geometry.py:348`, `t3_svd.py:312, 345`, `fv_conversions.py:181–182, 184–185` (the aliases, by A1), `:219, 251`.
   Tests and docs: `test_frame_variations_format.py:447`, `test_tucker_tensor_train.py:3202`, `user_guide.rst:142`.
   (`CHANGELOG.md:714` is a record and stays.)
4. **Letters in prose:** the tier D patterns; the CL rows of inv 1 §7 and inv 2 §9 that fall in comments and
   docstrings; the field comments `frame_variations_format.py:156–159` and `uniform_frame_variations_format.py:110–113`
   (`B_xo`→`D_xo`, `R_ixj`, which clashes with *right*, →`U_ixj`, `P`/`Q`→`L`/`R` by D-7); the docstring diagrams,
   redrawn with `D` below, `U` above and widths as words (D-8): `frame_variations_format.py:56–108, 219–224, 1271,
   1305–1322, 1362`, `uniform_frame_variations_format.py:1113, 1125`.
5. **The letter API (R3)**, §1.3: the 37 backend and 93 frontend/test LETTER-API rows. The tests' letter arithmetic is
   rewritten, not swapped: `test_padded_frame.py:723–737` builds subsets by deleting letters and `:730` pairs
   `('U', 'up'), ('O', 'down')`; the invalid cases become `(), ('down', 'down'), ('X',), 'down', 'ULRO', ('Down',)`;
   the recorded schedules `('ULRO', t)` become word tuples (inv 2 §10).
6. **Messages that name a store** (the `str` rows of inv 2 §9) and the 12 tests that assert on them (inv 1 §1.2:
   `test_padded_frame.py:405, 409, 732, 737, 760`, `test_uniform_padded_frame.py:303, 320, 518, 551`,
   `test_padded_start.py:277/283, 291/296`, `test_frame_variations_format.py:562`). These go in R2, so the suite stays
   green.
7. **D-5 in the note and the internals page.**
8. **The four tex notes, converted (R5)**: frame letters and widths in the new meaning, the T4S line in each, `Λ` and
   `d` (§3.1). The known in-note collisions: in `exact_hessian_two_hole.tex`, the Lagrangian `\Lambda`→`\ell` (7 uses,
   first), `DΦ`/`D²Φ`/`D_ζP`→`\mathrm d`,
   `\Dsh`/`\Ush`→`\Sh`/`\Sh^\T`, `n^D_i`→`n^U_i`, the two tikz figures relabelled (inv 3 §3); in
   `symmetric_probe_derivatives.tex`, `D^t`→`\mathrm d^t` at :111, while its `P^t` is the direction and stays; in
   `shared_t3_math.tex`, `U_i`, `U_g`, `U̇`→`D…`, `n_{D,i}`→`n_{U,i}`, "up-leg"→"Tucker leg", with `O_g` (a complement
   basis) kept; in `ttm_t3m_ht_note.tex`, the leaf frames `U_i`→`D_i`.
9. **Glossary** (inv 3 §5.2) and the 51 work-list rows (inv 3 §7; N4 and N17 resolve). The retired spellings are listed
   under "also: (2026.4.0 …)" for one release (BR §1.5). The symbol table gains `Λ`, `Ψ` and the upright `d`, and loses
   `O`; `D`, `U`, `P` are rewritten.
10. **Signature blocks realigned** wherever a name grew (`nU:` → `n_down:`), per
    `docs/contributor/signature_style.md`.
11. **Typos.** In R2, with the script, because after A1 they would read as right: `fv_conversions.py:58, 157`
    `down_tucker_supercore`→`up_tt_supercore`, and `:60, 159` `right_tucker_supercore`→`right_tt_supercore`. In R4a:
    `tucker_tensor_train.py:4264` `rL_0*r_1`→`r_0*r_1`. Loop variables named for the old store are renamed by hand:
    `fv_operations.py:653–654`'s `dn`, `u` → `w`; `u`/`o` at `test_padded_frame.py:491` and
    `_uniform_padded_frame_testing.py:41`.

## 5. Verification (the rename changes no number)

### 5.1 The bitwise battery: `dev/store_rename_battery.py`

The same seeded calls run on two trees in one environment: the reference, a `git worktree` of `v2026.4.0` (R1), and
the renamed tree (after R2, R3, R4b and R6). Results go to the scratchpad as `{key: array | text}`, never as `*.npz` in the
repo.

**The adapter** is the only tree-dependent code:

- `NEW = 'up_tt_cores' in {f.name for f in dataclasses.fields(bvf.T3Frame)}`. `hasattr` is False on both trees, because
  a field without a default is no class attribute.
- `upd(words)` gives the words, or on the old tree the joined letters (`down→U`, `up→O`, `left→L`, `right→R`).
- `attr(obj, new_name)` reads the new name, or on the old tree the old spelling: A1, A2 and A3 reversed (B2 reads the
  width properties).
- Widths are passed positionally.

**Fixtures** (each after `np.random.seed(0)`):

- `xS = TuckerTensorTrain.randn((5, 6, 7), (3, 6, 2), (1, 2, 5, 1))`: frame stores down `(3, 6, 2)`, up `(2, 4, 2)`,
  left `(1, 2, 5, 1)`, right `(1, 2, 2, 1)`, all four different. `xC`: the same with `stack_shape=(2,)`.
- `x_old = randn((5, 6, 7), (2, 3, 2), (1, 2, 2, 1))`, target `tk1, tt1 = (3, 5, 3), (1, 3, 3, 1)`,
  `x0 = x_old.resize(shape, tk1, tt1)`; `xG = randn((6, 6, 7), (3, 3, 2), (1, 3, 2, 1)).share((0, 0, 1))`;
  `y = randn((5, 6, 7), (5, 6, 7), (1, 5, 7, 1))`; seeded `ww`, `pp`, `idx`.
- **Slack padded paths** (critic r1 M3: at minimal frames a swapped pair of equal width ints is bitwise-invisible).
  - Slack target: `T3Frame.padded(x_old, (4, 5, 4), (1, 2, 3, 1))`, whose stores are `(4, 5, 4)`/`(2, 5, 3)`.
  - Slack old frame: the orthogonal representation of `xO = randn((5, 6, 6, 5), (4, 3, 3, 2), (1, 2, 3, 2, 1))`, with
    old stores `(4, 3, 3, 2)`/`(2, 3, 3, 2)` (the `test_padded_frame.py:743` structure).
  - Its target is `(5, 5, 4, 3), (1, 3, 5, 3, 1)`, with stores `(5, 5, 4, 3)`/`(3, 5, 4, 3)`.
  - These go through the backend step and loop (`tv_align_padded_frame`, `padded_frames.fv_aligned_padded_frame` with a
    module-level `project`) and their uniform twins.
- **Uniform twins:** `UniformTuckerTensorTrain.from_t3(·)`, the stacked `uxC = UniformTuckerTensorTrain.from_t3(xC)`, and
  `UT3Frame.from_t3frame(F, 8, 7, 5, 6, 3)` (positional; padded widths `n_down = 7`, `n_up = 5`, `r_left = 6`,
  `r_right = 3`, all different).

**Calls** (every array leaf is recorded; ints and tuples are recorded as arrays):

| id | calls |
|---|---|
| B1 representations | `t3_orthogonal_representations(x)` for `xS`, `xC`, `xG` → `(F.data, V.data)`; `T3Frame.from_t3(xS)`; `MANIFOLD.frame(xS)`; `T3Frame.random_orthogonal((5, 6, 7), (3, 6, 2), (1, 2, 5, 1), (2,))`; `t3svd_orthogonal_representations(xS)`; `fv_operations.fv_zero_point_frame(F.data)` and with `groups`; `ut3_orthogonal_representations`; `UT3Frame.from_t3frame(F, 8, 7, 5, 6, 3)` and back with `to_t3frame()`; `UT3Variations.from_t3variations(V, 8, 7, 5, 6, 3)` |
| B2 renamed attributes, keyed by physical store | the old names on the old tree and the new names on the new: `T3Frame` cores and the two ranks; `T3FrameWeights` (from `T3Weights.from_t3svd`) fields and ranks; `UT3Frame` supercores, ranks, the four widths and `masks` fields; `UT3Variations` ranks, widths and `masks`; `UT3FrameWeights` supercores, ranks and widths; the reprs of all five classes and of `T3Tangent` (text compared after the A-tier map) |
| B3 rank and mask functions, positional | `ranks.compute_orthogonal_representation_ranks` on `xS`'s structure and a stacked host array; `ranks.frame_has_minimal_ranks(shape, *stores, None)` on `xS` (False) and `x0` (True); `ufv_masking.ufv_make_frame_masks(*stores, 7, 5, 6, 3)`; `ufv_masking.ufv_frame_widths` |
| B4 orthogonalizations and SVD | `xS.down_orthogonalize_tucker_cores()`, `.up_orthogonalize_tt_cores()`, `.left_orthogonalize_tt_cores()`, `.right_orthogonalize_tt_cores()`, `.t3svd()`, `.t3svd(max_tt_ranks=2)`; the uniform twins; `xG.t3svd()` |
| B5 tangents | `v = MANIFOLD.project_ambient(F, y)`; `retract`, `transport` to `MANIFOLD.frame(x0)`, `inner`, `norm`; `v.to_dense()`, `v.to_t3()`, `v.gauge_residual()`; a `K`-stacked tangent `T3Tangent.stack_tangents((v, 2·v))` through `to_dense`, `inner`, `retract`; `fv_absorb_weights` with the B2 weights; the uniform twins via `UT3Tangent.from_t3tangent(v, 8, 7, 5, 6, 3)`, and `uxC`'s; `COREWISE.frame(xS)`; the shared and tied paths: `sharing.ufv_shared_frame_data`, `utv_operations.utv_orthogonal_gauge_projection(…, shared_data=…)` (the `up_mask_frame`, `up_mask_v`, `down_slices` code, `utv_operations.py:230–254`), `shared(MANIFOLD, sharing).project/retract` and `shared(UNIFORM_MANIFOLD, sharing)` on `xG` |
| B6 sampling | `v.probe(ww)`, `.apply(ww)`, `.entries(idx)`, `.probe_derivatives(ww, pp, 2)`, `.probe_transpose`, `.probe_derivatives_transpose`; `xS.probe_derivatives(ww, pp, 2)`; the uniform twins; the 14 keyword-renamed `probing.compute_*` / `sampling_derivatives.compute_*_jets`, positionally |
| B7 padded frames | `T3Frame.padded(x_old, tk1, tt1)`, also at the slack target and with `sharing`; `UT3Frame.padded`; `T3Tangent.align_padded` and `UT3Tangent.align_padded` with `upd(u)` for all 15 non-empty subsets of the four stores, plus one permuted tuple (`('left', 'down')` against `('down', 'left')`: the canonicalization), at the minimal and the slack target; the frontend `bvf.fv_aligned_padded_frame(x_old, tk1, tt1, y, max_passes=m, alternate=a)` for `(m, a)` in `(0, True), (3, True), (3, False), (10, True)`; the slack-old-frame backend step and the backend loop `padded_frames.fv_aligned_padded_frame(old_frame, tucker_ranks, tt_ranks, hint, project, …)`; `ufv_aligned_padded_frame` the same, plus `use_jit=True`; `backend.optimizers.padded_start(setup.problem, setup.x0_cores, x_old.data)` |
| B8 optimizers | `newton_cg(MANIFOLD, 'probe', ww, data, x0, max_newton=3)`, again with `padded_from=x_old`, with `g0_reference='cone'`, and with `hessian='exact'`; `newton_cg(UNIFORM_MANIFOLD, …, padded_from=…)`; `newton_cg(shared(MANIFOLD, (0, 0, 1)), …)` on `xG`; `gradient_descent(…, g0_reference='cone')`. Record `x.data`, `start_movements`, `cone_movements`, the integer/boolean `stats` and the numeric `history` fields stacked into arrays; host-clock fields are excluded |
| B9 files across versions | the old tree saves `T3Frame`, `T3Variations`, `T3Tangent`, `UT3Frame` and `UT3Variations` of `xS` to the scratchpad; the new tree loads them and records `.data`, which must equal its own B1/B5 arrays |

**Pass criterion:**

- **Compared:** arrays (every result, and the numeric history fields stacked into arrays) and the integer and boolean
  stats. **Excluded by name:** host-clock fields (`wall_time` in `stats['history']`, `backend/optimizers.py:1207, 1657`,
  and any key with `time`).
- The key sets are identical, and every array has the same dtype and shape with `np.array_equal(a, b, equal_nan=True)`
  (the uniform movements are NaN-padded).
- Texts are equal after the A-tier map.

This is what proves the swap is a rename.

### 5.2 Unequal stores

The ragged fixtures and five doctests have equal stores, so a missed swap passes them (inv 1 §1.2). R3 adds one
assertion per silent site (§2: 18 attributes and 4 keywords) at `xS`'s structure. Each checks the attribute against the
physical core shapes, e.g. `F.down_ranks == tuple(c.shape[-2] for c in F.down_tucker_cores)` (the weights are built
per store from `F`'s ranks, since `from_t3weights` copies one Tucker vector into both). They go in each class's
test module (`test_frame_variations_format`, `test_uniform_frame_variations_format`, `test_weighted`,
`test_uniform_weighted`, `backend/test_ranks`). The battery uses unequal stores and widths throughout.

### 5.3 The completeness check: `dev/store_rename_check.py` (code-tree part permanent per D-9)

**Patterns:**

| id | what | pattern |
|---|---|---|
| K1 | old spellings | `\b(?:masked_)?(?:up_tucker_cores\|down_tt_cores\|up_tucker_supercore\|down_tt_supercore\|up_tucker\|down_tt\|up_tucker_ranks\|down_tucker_ranks\|down_tt_ranks\|up_orthogonalize_tucker_cores\|down_orthogonalize_tt_cores)(?:_dot)?\b` |
| K2 | widths | `\b(?:nU\|nD\|nO\|rL\|rR)\d*[ijkdnlrxy]?(?![A-Za-z0-9])` and `\b(?:nn[UD]\|rr[LR]\|_N[UD]\|_R[LR]\|nDs)\b`; plus `\b[kmpiwW][UDLR]\d?\b` in the three A3-scoped files |
| K3 | the letter API | `update\s*:\s*str\b`; `update\s*=\s*['"]`; `['"](?:[ULRO]{2,4})['"]`; `['"][ULRO]['"]\s+(?:not\s+)?in\s+update`; in docs `` `'(?:ULRO\|UL\|RO)'` `` |
| K4 | frame-letter tuples, hybrids included | `\(U, ?O, ?[PL], ?[QR]\)`, `\((?:D\|U), ?(?:U\|O), ?P, ?Q\b` and `\(U, ?G, ?G, ?G\)` |
| K5 | old-order listings | `\(up, down, left` and `up/down/left/right` |
| K6 | backwards phrases | `Tucker up-(?:SVD\|orth)`, `Tucker cores upward`, `TT cores downward`, `down-orthogonal O\b`, `up-orthogonal tucker`, `Down TT core` |
| K7 | old shift and derivative notation | `_shift_(?:up\|down)`, `\b[DU]\^[rku]\b`, `\[U\^`, `\b[DU] (?:x\|z\|jet)\b` (catches `D x =`, `# U jet,`, `[jet, D jet]`), `\bDPhi\b`, `\bD\^2 ?Phi\b`, `DΦ`, `D²Φ`; in tex `\\Dsh\|\\Ush\|D\\Phi\|D\^2\\Phi\|D_\\zeta` and the Lagrangian's old `\\Lambda\(\\th\|_\\th\\Lambda` |
| K8 | T4S letters left behind | `\b(?:O_i\|O_aib\|Oᵢ\|δU\|U_g)\b`, `U̇` (no trailing `\b`: it never matches after a combining mark), `outer[- ]orthogonal`; frame `P`/`Q` forms `\b[PQ]_(?:i\|last\|\{)`; the code names `\b(?:Udot\w*\|dU(?![a-z])\w*\|gU\|Ug)\b` (so the new `dUn`, from `dOn`, is not hit). Carve-outs: the T4S correspondence lines and the §3.4 inventory's keep list |
| K0 | sentinels | `ZQX\d+_XQZ` anywhere (after pass 2) |
| K8b | T4S string keys | `\[['"][OPQ]['"]\]` in `tests/test_jet_recurrence.py` (after B4, `p['U']` is the correct up-core key) |
| K9 | residual store words, token-level (critic r1 M1, r2 MINOR 1) | every bare `up`, `down`, `Up`, `Down` in the COMMENT and STRING tokens of the final code tree (and every bare word in the docs after R5) must be one of three things: (a) an inventory keep row (ORTH, MSG, UNREL; inv 2 §9 keeps, inv 1 App. B–D, inv 3 §9, collected in R1 as `dev/store_rename_keeps.txt`); (b) a word whose aligned old-tree token was the opposite store word (`git diff` line alignment, then token position); or (c) part of a tier-E proposed text after §1.4's layering. Anything else fails. **Converse** (critic r3): every old-tree token on a keep row (ORTH, MSG, UNREL) survives unchanged at its aligned position, unless its line is a tier-E proposed text (e.g. §3.3's rewordings). So clause (b) cannot pass an adjective or a message direction flipped by mistake ("up-orthogonal" → "down-orthogonal" on a TT core, "contract up"), which K6's six phrasings would miss |

**Scope:** `t3toolbox/`, `tests/` and `examples/` (fourteen scripts, nothing to change, checked all the same) for
K0–K9; `docs/*.md|rst`, `docs/contributor/*`, `README.md` and
`CLAUDE.md` for all but K3's code patterns; the tex notes for K4, K7 and K8 (their equation letters `D`, `U`, `n_{D,i}`
are legal there, K2 is not run on them).

**Carve-outs, all of them:** `CHANGELOG.md` and `docs/release_notes.md` (records; the new sections name the old
spellings on purpose); lines of `docs/glossary.md` containing "(2026.4.0"; the T4S correspondence lines (§3, for K4
and K8); the paper ↔ code maps (`entries_apply_probe.md` §8, `batching_and_stacking.md` §8); the check file itself.
K3's `{2,4}` skips single-letter strings.

Before R2 the check runs in report mode, and its hits are the to-do list. After R4b it is clean on the code tree, and
after R5 on the docs. The script's per-file swap bijection (§1.3) is the R2-only check for A2, which no grep can do.
K9 is the only check that sees a bare word left in the old sense.

### 5.4 Suite, doctests, docs, PDFs

- The full suite at `-n 3` (the 13 GB laptop rule), in the current-leg environment and in the floor environment
  (numpy 1.26 / jax 0.4.30). The slow tier runs serially at R6, as the release gate.
- Module doctests: the frontend's 2,047 examples (inv 1 §4 found no expected-output change) and the backend's; every
  `docs/*.md` and `docs/contributor/*.md` page plus `getting_started.rst`; both environments.
- The dispatch rows. A word tuple stays a jit static: `_STEP_STATICS` position 7, asserted at
  `test_dispatch.py:2416–2418`. Then the import-graph test, `sphinx -W`, and `gh run list` after each push.
- PDFs rebuilt with `latexmk -pdf`: `padded_frame_cone.pdf`, `exact_hessian_two_hole.pdf`, `shared_t3_math.pdf`,
  `pad_safe_svd.pdf`. `symmetric_probe_derivatives.tex` and `ttm_t3m_ht_note.tex` have no PDF; their sources are
  edited only.

## 6. Documentation and release notes (R5)

- **`docs/frame_variations.md` § "Four rank stores"** gets the convention paragraph, the picture and the T4S note
  (inv 3 §1(a)), and every other page links to it. `naming_conventions.md` gets the bullet "frame stores are words".
- **`CHANGELOG.md` `[2026.5.0]` "Changed — breaking"** and **`docs/release_notes.md` "Upgrading from 2026.4.0"** come
  from inv 3 §1(b), in house format: the verdict first, then the same-spelling list, stated as silent, full stop ("these
  do not raise"; no claim that a shape check catches misuse, since at a minimal frame both stores agree); the
  old → new table with its "raises if left?" column; the one-pass migration snippet, whose `SWAP` map gains the `_dot`
  forms (`\b` does not match inside them); what does not change (positions,
  files, numbers, method names); the letters and the T4S correspondence; the `update` canonicalization. The root
  `release_notes_2026.5.0.md` is the CHANGELOG section.
- **The glossary and its work list** (§4.9); **`CLAUDE.md`**, inv 3 App. A (the notation map, the canonical-orderings
  gotcha).
- **The math notes, all converted.** `padded_frame_cone.tex`: Table 1, D-5, store words, seed tags, colours, the §5
  root sentence. The four T4S-notation notes are converted per §4.8, each with the T4S line. `pad_safe_svd.tex`: "the
  down step" → "the TT up-orthogonalization" (×2). Every PDF is rebuilt.
- **After R6:** the 601-row docs-consistency pass, on the renamed text.

## 7. Slices

| slice | content | gate |
|---|---|---|
| **R1** | Tools and inventories only, no library change: the battery (§5.1) and its reference run on the `v2026.4.0` worktree (plus the B9 files); `store_rename_apply.py` (§1.3) in dry-run mode (pass 1 into a scratch copy); `store_rename_check.py` (§5.3) in report mode; the `P`/`Q`/`U`-family inventory (§3.4) and `dev/store_rename_keeps.txt` (K9) | the dry-run hit table reconciled key by key with inv 2 §8, inv 1 §7 and a fresh tokenizer scan (every difference explained); every CL row and every §3.4 hit assigned; nothing outside `dev/` changed |
| **R2** | The script (pass 1, review of the sentinel diff, pass 2), tiers A–C, over `t3toolbox/`, `tests/` and `examples/`, the four sampling modules and their twins included; tier A over the docs and `CLAUDE.md`, so the markdown doctests stay green; the 12 message tests and the 4 typo lines (§4.6, §4.11) | K0 clean; battery bitwise against R1; no bijection errors; the full suite, the module and markdown doctests, and the dispatch rows green. **Review stop** |
| **R3** | The letter API (§1.3, §4.5) in the backend steps, loops and kernels and the frontend twins, with its tests; the unequal-store assertions (§5.2) | battery bitwise (through the adapter); full suite; dispatch; K3 clean |
| **R4a** | Code prose outside the sampling kernels: every tier D and tier E row in `t3toolbox/`, `tests/` and `examples/` (§4.1–4.4, 4.7, 4.10, 4.11), the free-prose store words included | suite; doctests in both environments; `sphinx -W`; K0–K9 clean on those files |
| **R4b** | The sampling kernels' prose (`probing`, `sampling_derivatives`, `apply`, `entries`, `utv_sampling`, `ut3_sampling`, the jet twins and curvature tests): their tier D and E rows, the shift and derivative notation (§3.1–3.2), the reworded message directions (§3.3) | battery bitwise; suite; doctests; K0–K9 clean on the code tree. **Review stop** |
| **R5** | Docs (§6): pages, diagrams, the six notes (§4.8) and their PDFs, the glossary and its work list, `CLAUDE.md`, CHANGELOG and release notes | `sphinx -W`; markdown doctests in both environments; K-checks clean on the docs and notes; PDFs rebuilt; every example runs (`examples/*.py`, as CI's examples job does). **Review stop** |
| **R6** | Acceptance: §5 end to end in both environments, the slow tier, the battery also in the floor environment; the T3Polynomial follow-up prepared (`g01_g0_common.py:121`, its suite), committed only after the release; HANDOFF | — |

Each slice follows the house process. The implementer builds and writes `dev/store_rename_slices/R<k>_handoff.md`; the
reviewer writes `R<k>_review.md` with a verdict (for R2, after its own re-derivation of the hit table and its own
battery comparison); the session inspects the diff, runs the suite, commits by name (never `-a`), pushes and checks
`gh run list`.

## 8. Decisions for Nick

Rulings of 2026-11-11 are recorded as RESOLVED; the rest carry a recommendation.

- **D-1. RESOLVED (Nick): no exemption.** What remains for Nick is the jet-shift symbol (§3.2). *Recommended: `Λ`,
  `Λᵀ`, with the exact-Hessian note's Lagrangian `Λ` (7 uses) renamed `ℓ`.* The alternative is a spelled
  `sh`/`shᵀ`, which needs no rename. `S`, `J⁺`/`J⁻`, `Γ`, `Υ`, `Δ`, `Ψ`, `E`, `M` and `T` are rejected for
  collisions.
- **D-2. `update` as a collection of store words, any order** (§1.3). Any non-`str` collection is accepted and
  canonicalized to physical order before it becomes a jit static. The message names the words; it gives no 2026.4.0
  letter hint, which the release notes carry. *Recommended.* Alternatives: a frozenset (an awkward literal, printed
  unordered); a space-separated string (letters by another name); `None` meaning all.
- **D-3. Plain-TT bond widths** (inv 2 WIDTH-GENERIC: 86 rows, 163 tokens; plus the T3 docs' `rL_i`). *Recommended:
  fold them in as `r_left`/`r_right`.* One spelling, and no carve-outs inside the mixed files (`ranks.py`,
  `t3_linalg.py`), which is where a missed frame width would hide. A UT3 point's Tucker rank `nU` becomes `n` (A3
  exception), not `n_down`. The alternative is to leave them; they are not store widths.
- **D-4. The five doctests that print `F.up_ranks`** (`frame_variations_format.py:338, 1840`, `manifold.py:202`,
  `uniform_frame_variations_format.py:267`, `optimizers.py:807`). All are padded frames, minimal by construction.
  *Recommended: rename only*, and detect stale names with §5.2 and the battery's slack paths. An unequal structure is
  impossible for `padded_start`, whose `x0` must be minimal.
- **D-5. The padded-frame letters.** The products and messages `D` become `Ψ` (code `Psi…`, the note's
  `\Psi_{a\to b}`, the internals page), and the edge frame `U_{x→t}`, `U′` becomes `F_{x→t}`, `F′`. *Recommended.*
  `Ψ` is free in the code, the note and the page (the dense oracles' lowercase `psi`, the right interfaces, is unrelated;
  their letter note says so). `M` is the note's proof matrix (:209); `E` is the captured energy (internals :53, note).
- **D-6. RESOLVED (Nick): the corewise frame is `(D, G, G, G)`.** *Recommended for the residual case* (§3.3): `D`
  for everything that occupies or moves the down slot, the T4S-derived `U`-family code names included (`dU`→`dD`,
  `Udot`→`Ddot`, …, with the `G` family unchanged so pairs stay symmetric). `B` stays for a T3's own cores on the T3
  pages, where the README, the user guide and the `TuckerTensorTrain` doctests already use it. The alternative is `D`
  there too, which would rename the user guide's `B`.
- **D-7. RESOLVED (Nick): `P`/`Q` → `L`/`R` wherever they name frame cores.** This uses per-file NAME maps and exact
  text patterns. `P` keeps its other meanings (the completion `P⁺`, the P-block, `P_U`, the direction `P^t`), `Q` its
  QR meaning, and `R` the RNG helpers.
- **D-8. Width spelling** (inv 3 C18). *Recommended:* words everywhere outside LaTeX, shape comments and ASCII diagrams
  included (`n_down0`), overruling inv 1's letter-swapped diagram labels. In LaTeX every note uses subscript letters
  in the new meaning: `n_{D,i}`, `n_{U,i}`, `r_{L,i}`, `r_{R,i}`, swapped simultaneously with today's; the exact-Hessian
  note's `n^D_i` → `n^U_i`. Why: an old reader's `nD` is the TT store, and a diagram `nD` in the new meaning is a silent
  swap that no check can tell from a stale one.
- **D-9. Keep §5.3's K0–K8 code-tree checks as a permanent test** `tests/test_store_vocabulary.py`, like the HISTORICAL
  vocabulary scan: cheap, and it stops old spellings coming back. K9 needs the tag diff, so it stays a one-off.
  *Recommended.* The alternative is to keep all of it one-off in `dev/`.
- **D-10. Sentinels** (Nick's suggestion; §1.3). *Recommended: sentinels implement every pass*: required for the text
  rewrites, and in the Python pass for the single reviewable pass-1 diff. The critic's variant, sentinels for text only
  with the tokenizer's lookup for Python, is equally safe; the difference is one review artifact or two.
- **D-11. The derivative `D`** (`DΦ`, `D²Φ`, `D_ζP`, `D^t`, `DPhi`) becomes the upright differential `d`, so `D` has
  one meaning in the notes and on the curvature page. *Recommended.* The alternative is a prime, `Φ′(θ)[·]`, which has
  no form for `D_ζP`.

## 9. Risks

- **A sequential replace corrupting double-touched tokens.** Prevented by the sentinels (K0), the bijection check and
  the battery.
- **A silent swap passing on equal-store fixtures.** Prevented by §5.2 and the battery's slack fixtures.
- **Collisions the rename itself creates.** Found so far: `E` (D-5); `dc`, the `dataclasses` alias (hence
  `c_down`/`c_up`); `D`, the oracles' dense frame (renamed `dense`); `R`, `test_jet_recurrence`'s random lambda
  (renamed `rand`); `test_uniform_curvature.py`'s `D` family (renamed `th…`); `Ud`, "u dense" in `o1_ragged.py` (not
  mapped); the alias pair, which must stay one-way (A1).
  The R1 hit table lists every key's hits per file before anything is written.
- **Text that reads right by accident but says the wrong thing.** The inventories are exhaustive by construction, the
  reviewer reads every convention sentence, K5/K6 catch the common forms, and K9 catches every unchanged bare word.
- **T4S readers** meet `U` as the up core and the paper's `U` as our `D`. The T4S line sits where each reader sets
  notation (§3), and in the release notes.
- **Converting the sampling kernels** (about 905 inventory rows plus the `P`/`Q` maps) in the library's most intricate
  module. Mitigated by the per-file maps' collision check (no `L`, `R`, `D` local exists in `probing` or
  `sampling_derivatives`), the battery's B6 jets, the jet twins converted in the same pass, and R4b's own review stop.
- **Two `U`s in one note.** The padded-frame note's matrix-case proof keeps the SVD's `U` (inv 3 C11) one section
  before the store table. It is a self-contained proof; renaming its SVD factor is optional.
- **Formatting churn.** Grown names break the aligned signature columns (§4.10); `untokenize` is never used.
- **Bitwise holds within one environment.** The battery compares the two trees in the same environment, never across
  environments (R6 repeats it in the floor environment).
