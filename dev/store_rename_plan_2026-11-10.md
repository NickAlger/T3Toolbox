# The store rename — plan (draft 11, writer, 2026-11-11)

> **Status:** draft 11: Nick's rulings of 2026-11-11, critic rounds 1–11 (`dev/store_rename_plan_review_r{1,…,11}.md`),
> Nick's review of draft 4 (§4A), his width ruling (D-8 final), and his withdrawal of "no letters in the API"
> (draft 9). **The critic agreed in round 11.** Next: Nick, then the implementer/reviewer slices R1–R6 (§7).
> **Breaking**; the first item of 2026.5.0. The glossary update and the 601-row docs-consistency pass
> (`dev/glossary_naming_decisions.md`) run after it, on the renamed code.
>
> **Sources:** `dev/store_rename_inventory_1_frontend_tests.md`, `_2_backend.md`, `_3_docs.md` (inv 1/2/3 below) and
> `dev/up_down_naming_blast_radius.md` (BR). Their line numbers are of `b272e5de`; nothing outside `dev/` changed through
> `79b2bd6e`. `t3toolbox/`, `tests/` and `examples/` equal the `v2026.4.0` tag; `docs/` differs from it by `glossary.md`
> and `index.rst`.
> Every hand edit asserts its row's current text before replacing it, so a stale row fails loudly.

## CHANGES

*History only: the record of how the plan got here. The instructions are the numbered sections below; where a CHANGES
line and a section differ, the section governs.*

**Draft 11 (critic r10).**

- Doctest-example letters are scheduled once: R2, via `letter_sites.md`. §4.4 covers only comment and prose letters,
  and §4A's companion sentence is fixed.
- §1.5 item 12's `u` is bounded: `uc`/`um` are up-store locals, and `_u_` names change only by the token map.
- Item 11 adds the uniform `gU` doctests.

**Draft 10 (critic r9, over-reach).**

- **K15 closes the token map.** It is a positional NAME check after R2 and a hunk check after R3–R5.
- **§1.5 gains two classes:**
  - `U` as a T3's own Tucker core in non-frame code. §3.1 no longer licenses renaming them, and §3.4's `Uio→Bio` is
    withdrawn: the T3-level `O_ab_x` stays and is carved out of K8.
  - `U`/`u` meaning *uniform*.
- **Tier C's catch-all** is replaced by R1's enumerated `(file, function, token → image)` list.
- **B5 and the T3 pages' `B` renames are withdrawn:** T3-level code keeps its letters.
- **Bounded sentences:** §3.3, §4.1, §4.11, tier D's enumerated derivative sites, and §1.5 item 2's sites.
- **D-2** gains form (c).

**Draft 9 (Nick: the "no letters in the API" rule is withdrawn).**

- Every rule-like principle is replaced by enumerated facts: §0's "What changes" lists exactly the names that change.
  Each sentence an implementer could over-apply ("one alphabet", "letters stay", "everywhere", "wherever") is now a
  bounded instruction with its scope and exceptions (§0, §3, §3.3, D-7, D-8).
- New **§1.5 DO-NOT-TOUCH** list and gate **K14**: those identifiers must be byte-identical between the trees.
  - `_shift_up`/`_shift_down` (and `_pair`) and `c_up` keep their names. Only the shift *letters* in equations
    change.
  - `test_jet_recurrence`'s RNG `R` stays. The frame-core locals that share its function take a `c` suffix.
- **D-2 reopened on its merits:** (a) letter strings in the new meaning (recommended) or (b) word tuples.
- The CHANGELOG/release-notes "No store letters in the API" bullet (inv 3 §1(b)) is replaced by the enumerated facts.
- §7 gains Nick's implementer brief: no escape from case-by-case judgement; no rename outside the token map without
  asking.

**Draft 8 (critic r7).**

- The width rules gain the lookbehind `(?<![\\\w])`, so the `\nOn` escape in `examples/fit_hilbert_regularized.py:167` is
  not hit.
- K13 checks only store-named entries and `V`/`H`. Ten generic kernel shape lines get a disposition.
- The battery adds default-width conversions.

**Draft 7 (Nick's width ruling, 2026-11-11).**

- Widths keep the house symbols `nU`, `nD`, `rL`, `rR` everywhere: code, shape comments, keywords, properties, pictures
  and LaTeX. The postfix letter now names the store. There are no `n_up`/`n_down`/`r_left`/`r_right` anywhere.
- Consequences:
  - A3 is the same simultaneous swap as the rest (`nU`↔`nD`, `nO`→`nU`, `rL`/`rR` untouched).
  - "No letters in the API" narrows to `update`.
  - The width keywords and properties become silent swaps (§2, release notes).
  - D-3 and D-8 are resolved.
  - The width gates ban only stale spellings (K2).
  - New K13 checks each shape comment's width letter against its entry. Nick's sketch is corrected: the Tucker
    *variations* carry `nU`, and the left/right cores and TT variations carry `nD`.

**Draft 6 (critic r5, on §4A).**

- Fences are spans, not lines.
- K12 is the completion gate, and K10b checks the alignment of re-laid-out blocks.
- Fenced tier-E rows feed the hand rewrite only.
- §3.4 now also covers the `O` family, with `O_g` carved out.
- The seed list is extended, with companion doctests and store-word index variables.
- K10 compares raw text.

**Draft 5 (Nick, 2026-11-11).**

- **Amended D-8** (superseded by draft 7's final D-8): diagram and equation labels keep the compact letters `nU, nD, rL, rR` in the new meaning, and `nO`
  disappears.
- **New §4A:** every ASCII diagram and equation is a hand-only, first-class artifact. It gets an R1 inventory, a
  whole-block hand rewrite with before/after shown, an alignment check (K10), re-derived orthogonality relations,
  unchanged doctest outputs, a full read by the reviewer, Nick's before/after review, and a script fence (K11).
- §5.3, §5.4, §6, §7 and §8 carry the matching gates and rows.

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

**What changes: this list is exhaustive** (Nick, 2026-11-11). Any rename not on it, and not in §1.2's token map, is
forbidden without asking.

1. **The store words** `up`/`down` where they name a frame store: identifiers (tiers A1, A2), bare code words and the
   exact literals (tier B), and prose (tier E). Adjectives, message directions and English stay (§1.5).
2. **The frame-core letters.** Old `U` (Tucker factor) → `D`; old `O` (TT-shaped core) → `U`; and `P`/`Q` → `L`/`R`
   where they name the left/right frame cores. This applies only at the sites tier C, tier D, §3.4 and §4A enumerate.
   T4S is historical: within those sites, internal consistency beats consistency with the paper.
3. **The width letters**, same spelling, swapped: `nU` ↔ `nD` (`nD` is now the down store's width, the Tucker
   factor's; `nU` the up store's, the TT-shaped core's), and `nO` → `nU`. `rL`, `rR` are unchanged. These are house
   symbols, kept as letters in code, shape comments, keywords, properties, diagrams and LaTeX (D-8). No word forms
   (`n_up`, `n_down`, …) are introduced.
4. **The T4S-derived `U`-family code names** listed in tier C's row (`Udot`, `dU…`, `gU`, `gYU`, `Ug`, `Ud`), and
   their `O` counterparts per §3.4.
5. **The jet-shift letters** in equations and prose: `D x`, `U x`, `D^r`, `U^u` → `Λ x`, `Λᵀ x`, … (D-1). The shift
   *functions* keep their names.
6. **The derivative `D`** → upright `d` (`DΦ`, `DPhi`, …), if D-11 stands.
7. **The padded-frame letters** of D-5 (`D` products → `Ψ`, `U_{x→t}` → `F_{x→t}`).
8. **The `update` argument's values**, per D-2.
9. **The dense oracles' local names** and the collision renames listed in tier C.

Every other identifier and code token stays byte-identical; prose changes only by tier E. §1.5 names the parts most
at risk (gate K14), and K15 checks every NAME token (§5.3).
- **Physical order is unchanged.** `T3Frame.data` and every rank, mask, weight and uniform tuple keep their positions,
  which now read `(down, up, left, right)`. `save`/`load` key arrays by position (`'f%d_%d'`, `ufv_save`), so
  2026.4.0 files load unchanged (inv 1 §5).
- **Same-spelling names swap meaning (§2).** This is accepted, and the release notes list them first. There are no
  aliases. The house rule (2026.2.0, 2026.3.0) is that a breaking release breaks loudly once, and a meaning swap
  cannot be aliased anyway.

## 1. The mapping

### 1.1 One simultaneous mapping, five tiers, and fences

The two vocabularies overlap: old up is new down, and `D` already marks the TT store in `T3Frame`'s docstring. So every
substitution is **simultaneous**: one lookup per token and one pass per file, never a sequence of replacements.

- **A — identifiers.** The exact compound tokens of §1.2 and no others; each has one meaning across the tree. They apply as NAME tokens in Python and `\b`-bounded in
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
- **F — fenced blocks (§4A).** Diagrams and equations: hand-only, skipped by every script pass and checked by K10 and
  K11.

### 1.2 The token map

| tier | old → new (one simultaneous pass) | scope |
|---|---|---|
| A1 renamed | `up_tucker_cores`→`down_tucker_cores`, `down_tt_cores`→`up_tt_cores`, `up_tucker_cores_dot`→`down_tucker_cores_dot`, `down_tt_cores_dot`→`up_tt_cores_dot`, `up_tucker_supercore`→`down_tucker_supercore`, `down_tt_supercore`→`up_tt_supercore`, `masked_up_tucker_supercore`→`masked_down_tucker_supercore`, `masked_down_tt_supercore`→`masked_up_tt_supercore` | everywhere |
| A1 aliases, **one-way only** | `up_orthogonalize_tucker_cores`→`down_orthogonalize_tucker_cores`, `down_orthogonalize_tt_cores`→`up_orthogonalize_tt_cores` | `backend/fv_conversions.py` only (181–185, 220, 253). The targets are public method names, so no reverse entry exists anywhere |
| A1 labels | `up_tucker`→`down_tucker`, `down_tt`→`up_tt`, `up_tucker_ranks`→`down_ranks`, `down_tucker_ranks`→`up_ranks`, `down_tt_ranks`→`up_ranks` (`uniform_frame_variations_format.py:203, 665`) | text only (comments, docstrings, docs) |
| A2 swaps (a ↔ b) | each of `up_ranks`, `up_mask`, `variations_up_mask`, `var_up_mask`, `up_weights`, `up_weight_supercore`, `up_sc`, `up_cores`, `up_supercore`, `up_inds`, `up_r`, `v_up_r`, `up_out`, `up0`, `up3`, `up4`, `vup`, `mup`, `up_shapes`, `up_stack_shapes`, `up_slices`, `up_masks`, `up_widths`, `up_mask_frame`, `up_mask_v`, `_up`, `_up_mask`, `_up_edge`, `_up_message`, `_u_up_messages`, `_UP_R`, `bad_up`, `masked_up`, `point_up`, `variations_up` ↔ its twin with `up`/`UP` → `down`/`DOWN`; also `test_unequal_old_up_masks_in_a_group`→`test_unequal_old_down_masks_in_a_group` | everywhere |
| A3 widths, a swap (a ↔ b) | `nU` ↔ `nD` with its suffix kept verbatim, by the family `(?<![\\\w])(nU\|nD)(?=\d\|[ijkdn]\b\|_\|\(\|\b)` (the lookbehind, not `\b`, so an escape such as the `\nOn` of `examples/fit_hilbert_regularized.py:167` is not a width; so `nU0`↔`nD0`, `nUi`↔`nDi`, `nU_pad`↔`nD_pad`, `nU_i`↔`nD_i`, `nU(d-1)`↔`nD(d-1)`); `nO`→`nU` (one-way, same lookbehind and suffix rule); explicit: `_NU`↔`_ND`, `nnU`↔`nnD`, `nDs`→`nUs`, `sum_nD`→`sum_nU`. `rL`, `rR` and their forms are untouched, frame or generic (D-3). The widths join the bijection check, with `nO` counted into `nU`. **Exception:** where `nU` is a UT3 *point's* Tucker rank (no store), it becomes the house symbol `n` (inv 1 C7; `n` is free in each scope): `optimizers.py:176–180`, `tests/test_chunk_size_estimator.py:274–276`, and `sampling_derivatives.py`'s chunk estimators (1819–1824, 1870–1912), where `nO = nU` becomes `nU = n` | everywhere, the §4A fences included (by hand there), and LaTeX in its `n_{D,i}` form |
| A3 scoped swaps | for `X` in `k, m, p, i, w, W`: `XU`↔`XD`, the `L`/`R` members unchanged. That is `kU`↔`kD`, `mU`↔`mD`, `mU0`↔`mD0`, `mU1`↔`mD1`, `pU`↔`pD`, `iU`↔`iD`, `wU`↔`wD` (the per-store widths, `_uniform_padded_frame_testing.py:43–85`), `WU`↔`WD` | `backend/uniform_padded_frames.py`, `tests/backend/_uniform_padded_frame_testing.py`, `tests/backend/test_uniform_padded_frame.py` only |
| B1 bare | NAME `up` ↔ `down`, in code and in doctest examples | everywhere except KEEP_BARE: `tests/test_fitting.py:1148–1152` (a tangent), `tests/backend/_curvature_oracles.py:529–536` (the ξ lambda), `t3toolbox/fitting.py:290–291` (a doctest tangent). R1 re-derives the list over doctest lines (BR's eight were code tokens only) |
| B2 literals | STRING tokens exactly `'up'` ↔ `'down'` (either quote) | everywhere: `_STORE_NAMES`, the `_SEED_TAGS` keys, `_seed_rows` arguments, the test copy `tests/backend/test_padded_frame.py:91–92`. The `_SEED_TAGS` integers stay with the physical store (`{'down': 0, 'up': 1, …}`), so the seeds are bitwise unchanged. Messages built with `%s` from `_STORE_NAMES` flip with it |
| B3 scoped | NAME `dn`→`up` (`dn` is a difference norm everywhere else) | inside the enclosing function or doctest of `uniform_frame_variations_format.py:1029`, `tests/test_uniform_frame_variations_format.py:250, 292, 323`, `tests/test_dispatch.py:1680, 2840`, `tests/test_weighted.py:211` |
| B4 scoped letter strings | STRING `'U'`→`'D'`, `'O'`→`'U'`, `'P'`→`'L'`, `'Q'`→`'R'` (simultaneous) | `tests/test_jet_recurrence.py` only: the 34 dict keys `p['O']`, `p['P']`, `p['Q']`, … (its `dict(U=…, O=…)` keywords are NAMEs, tier C). Elsewhere a `'U'` string is untouched (e.g. `test_tucker_tensor_train.py`'s T3 kind `'U'`, §1.5) |

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
| T4S-derived Tucker names (§3.3) | `Udot→Ddot, Udot_g→Ddot_g, dU→dD, dU0→dD0, dU_v→dD_v, dU_t→dD_t, dU_tilde→dD_tilde, dU_tildes→dD_tildes, gU→gD, gYU→gYD, Ug→Dg`, and `Ud→Dd` where it is the Tucker direction (`_curvature_oracles.py`, `test_curvature.py`, `utv_operations.py`; not `o1_ragged.py`, where `Ud` is "u dense", nor `o1_uniform.py`, whose `U` is a uniform T3). Only in frame, tangent, sampling and curvature code: `manifold.py`, `tv_operations.py`, `utv_operations.py`, `utv_sampling.py`, `sharing.py`, `apply.py`, `probing.py`, `sampling_derivatives.py`, `backend/fitting.py`, `ut3_sampling.py`, `entries.py`, and the tests R1's enumerated list names, occurrence by occurrence. The `G` family (`dG`, `Gdot`, `Gd`) is the library's TT letter and stays. Exceptions: `tv_operations.py:292–331`'s `dU = tt_variations[i]` is a TT variation, so it becomes `H` (hand row). T3-level code keeps its `U` family (`test_tucker_tensor_train.py`'s `dU`, the `TuckerTensorTrain` doctests' `gU`; §1.5 item 11) |
| T4S test twins | `tests/_jet_trs_twins.py`: `O→U, P→L, Q→R, P_cores→L_cores, rev_Q→rev_R`. `tests/test_jet_recurrence.py`: `U→D, O→U, P→L, Q→R` in `_problem` (49–68), plus B4. In `test_uniform_equals_ragged` (163–200) the RNG helper `R` is on §1.5's list, so the frame-core locals there take a `c` suffix: `O→Uc, P→Lc, Q→Rc`. The RNG `R` at 166, 208, 228 is untouched. `Q→R` is mapped only in the files listed; `test_probe_derivatives.py:488, 497` keeps its RNG `R` |
| tests that mirror a mapped module | `tests/backend/_uniform_padded_frame_testing.py`, `tests/backend/test_uniform_padded_frame.py` and `tests/backend/test_padded_start.py` (290–307): `U→D, O→U, Uu→Du, Ou→Uu`. `tests/backend/test_padded_frame.py` (111, 135–139, 205–216, 619–629): `U0→D0, O0→U0, U→D, O→U` (inv 1 §7 gives `D0, U0, L0, R0 = old`). `tests/test_uniform_frame_variations_format.py`: `U_r→D_r, U_u→D_u` (958–959 by row). `tests/test_dispatch.py`: `U_3→D_3` |
| every other CL row (critic r1 M2, r9) | **R1's enumerated list** `dev/store_rename_letter_sites.md`: one row per occurrence, `(file, function, token → image)`, for every CL row of inv 1 §7 and App. A, every CORE-LETTER entry of inv 2 §8 not in the rows above, and every §3.4 hit classified as a frame core. R2 applies that list and nothing else; no rule map is applied. Files expected on it: `test_frame_variations_format.py`, `test_manifold.py`, `test_uniform_manifold.py`, `test_curvature.py`, `test_uniform_curvature.py`, `_curvature_oracles.py`, `test_tv_operations.py`, `test_sharing.py` (its frame `UU, OO` at 576–598, not its `U_rows`), `test_optimizers.py:533`, `test_zero_reference.py:94`, `test_contractions_sharding.py:199`, `test_fitting.py:8`. A row whose token has another meaning (§1.5) is excluded, with its reason. Gate: R1 shows every CL row and every classified hit on the list |
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
  (41 lines). Outside it, exactly these 15 code lines: `sharing.py:757`, `t3_linalg.py:887, 889`, `optimizers.py:220`,
  `tv_operations.py:510`, `fv_operations.py:560, 579`, `_curvature_oracles.py:11, 64, 178, 238`,
  `test_uniform_curvature.py:1617, 1704`, `test_curvature.py:1577, 2852`. In the docs: `deferred_and_rejected.md:134`
  and `curvature_internals.md:310`.

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
FENCES:     typ.Dict[str, typ.Tuple[typ.Tuple[int, int, str], ...]]   # §4A: file -> (first, last, anchor) per block;
                                                              # text inside is skipped by every pass

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

**The `update` argument (D-2, open for Nick).** It lives in `backend/padded_frames.py`, also imported by
`uniform_padded_frames.py` (the D-46 sibling convention). Form (a), recommended:

```python
_STORE_LETTERS = 'DULR'                                     # a frame's stores in physical order (T3Frame.data)
_COLOURS = ('DL', 'UR')                                     # the two-colour passes (alternate=True), in pass order

def _update_stores(
        update:  str,   # store letters, any order, e.g. 'DL': D down, U up, L left, R right
) -> str:               # the same letters in _STORE_LETTERS order: one jit static per set
    """``update`` validated and canonicalized; ``ValueError`` otherwise, naming the letters."""
    if not (isinstance(update, str) and update and len(set(update)) == len(update)
            and set(update) <= set(_STORE_LETTERS)):
        message = ("update must be a non-empty string of the store letters 'D', 'U', 'L', 'R' without repeats "
                   "(D down, U up, L left, R right), e.g. 'DL'; got %r" % (update,))
        if isinstance(update, str) and 'O' in update:
            message += ("; 'O' was the 2026.4.0 letter for the TT-shaped store, now 'U' "
                        "(and 2026.4.0's 'U', the Tucker store, is now 'D')")
        raise ValueError(message)
    return ''.join(c for c in _STORE_LETTERS if c in update)
```

- **The steps** take `update: str = 'DULR'` (any order; the rest transported), call `_update_stores` first, and pass
  the canonical string to the uniform kernel as its static.
- **The loops** pass `_STORE_LETTERS` and `_COLOURS[0]`/`_COLOURS[1]`.
- **The truncation check** keys on the letter: `zip(_STORE_LETTERS, …)`, then `if b < c and letter not in update`.
- **The conversion of the existing letter strings** is a simultaneous swap scoped to the inventories' LETTER-API rows:
  `'U'`→`'D'`, `'O'`→`'U'`, `'L'` and `'R'` kept, inside the update strings and the `'X' in update` tests. So `'ULRO'`
  becomes `'DULR'`, `'UL'` becomes `'DL'`, and `'RO'` becomes `'UR'`.
- **The silent case:** a 2026.4.0 string without `'O'` (e.g. `'UL'`) silently means other stores, the same accepted
  swap as `nU`/`up_ranks`. A string with `'O'` raises.

**Form (b)** is a tuple of words, `('down', 'up', 'left', 'right')`, with `_COLOURS = (('down', 'left'), ('right',
'up'))`. Any non-`str` collection is accepted and canonicalized to that order, and a `str` is refused, with the message
"update must be a non-empty tuple of the store words …".

**Form (c)** is a collection of single store letters, `update: typ.Collection[str] = ('D', 'U', 'L', 'R')`, with
`_COLOURS = (('D', 'L'), ('U', 'R'))` and canonicalization to that order. A `str` is refused; one containing `'O'`
gets the pointed message. The conversion of the existing strings splits each into its letters and applies (a)'s swap:
`'UL'` → `('D', 'L')`. In every form the messages keep `update must be` and `without repeats`, which the tests' regexes
match.

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
   - L5: D-5's `Ψ`/`F`, and D-8's width swap. The inventories' word proposals (`n_down…`, `r_left…`) are re-expressed as
     the swapped letters (`nD…`, `rL…`); inv 1's letter-swapped diagram labels (`nU0`→`nD0`, `nO1`→`nU1`) already are.
   - L6: the reworded message directions (§3.3).

A tier-E row whose line lies in a §4A fence (e.g. inv 1's rows inside diagrams: `frame_variations_format.py:86–108`'s
widths, `:1266, 1271, 1273`) is **not** applied by the locator. It only feeds the block's hand rewrite as a proposal.

On an unfenced line that carries both a tier-E row and a tier-D pattern, **tier E runs first**, because its locator
needs the line's R2 image. Tier D then runs over every unfenced line; its patterns include the hybrids, so it is idempotent on correct
text. The reviewer checks the layered text, and K4, K7 and K8 check the outcome.

### 1.5 DO NOT TOUCH (Nick, 2026-11-11)

These names stay **byte-identical**. Gate K14 compares them between the two trees, and R1 proves that the token map and
this list are disjoint. The library uses letter and Greek names in its API throughout. A name below may share a letter
with a store, but it is not a store.

1. **Greek-letter names and their derived names**: `xi`, `xis`, `eta`, `etas`, `mu`, `mus`, `nu`, `nus`, `sigma`,
   `tau`, `zeta`, `theta`, `theta_hat`, `omega`, `rho`, `kappa`, `Phi`, `Xi`, `Psi` (new, D-5), and the jet and tilde
   forms (`xi_jets`, `sigma_tilde(s)`, `tau_tilde(s)`, `deta`, `deta_tilde`, `dxi_tilde`, `sigma_hat`, …).
   - Only a store-letter suffix on a Greek stem changes, as tier C lists it (`PhiO→PhiU`, `PhiU→PhiD`, `XiO→XiU`,
     `XiU→XiD`, `XiUs→XiDs`). The stem never changes.
2. **Decomposition factors**: `U`, `S`, `V`, `Vt`, `Q`, `R` and their forms (`Us`, `U_M`, `svd_U`, `svd_Vt`, `U_a_x_b`,
   `U_i_a_x`, `U_o_x`, `U_nk`, `U_T`, the SVD helpers' `Uc`/`Uy` at `_padded_frame_oracles.py:54`) where they are
   factors of an SVD, QR or eigendecomposition.
   - Sites include `linalg.py`, `t3_svd.py`, `ut3_svd.py`, `tt_orthogonalization.py`, `sharing.py`,
     `uniform_padded_frames.py` (`Us`), the SVD locals of `ut3_orthogonalization.py` and `t3_orthogonalization.py`
     (incl. their `O_ab_x`/`Oaxb`, §3.4), `fv_operations.py:284–285`, `tests/backend/test_linalg.py`,
     `test_sharing.py:372`, `test_padded_frame.py:99`, `test_padded_start.py:582`, `_padded_frame_oracles.py:164`,
     `test_dispatch.py:163, 2095` (a QR's `P`), and the padded-frame note's matrix proof (`A = UΣVᵀ`).
   - R1's `dnt.txt` is the authority, not this list.
3. **`P` as a completion**, `P⁺`, the P-block and A-block, and the derivative direction `P`, `pp`, `P^t`. Also the
   pseudo-inverse `P` (`test_curvature.py:215–220, 353`).
   - Only a store suffix changes (`P_O→P_U`, `O_P→U_P`, tier C).
4. **The RNG helpers `R`**: `test_jet_recurrence.py:166, 208, 228` and `test_probe_derivatives.py:488, 497`.
5. **The other letters:**
   - `B`, a T3's own Tucker cores;
   - `G`, the TT cores;
   - `V`, `H`, the variations;
   - `K`, `C`, `W`, the batch blocks;
   - `N`, `n`, `r`, `d`;
   - `rL`, `rR` and every form of them (D-3);
   - the `dataclasses` alias `dc` where it is imported (7 modules).
6. **The index letters inside contraction strings:** every string argument of `contract(…)` and `einsum(…)`, and the
   HISTORICAL table.
7. **Every `*_orthogonalize_*` and `*_svd_*` name**, and the unfolding names (`_up_matricization`, `_up_res`,
   `up_svd`, `up_svd_pair`, `down_svd_pair`, `t3_down_svd_tt_core`, …). The one exception is the two private
   `fv_conversions` aliases of A1.
8. **The jet-shift functions and locals:** `_shift_up`, `_shift_down`, `_shift_up_pair`, `_shift_down_pair`, and
   `test_curvature.py`'s `c_up` (jet order, not stores). Only the shift letters in equations change (§3.2).
9. **Everything the inventories file as UNRELATED or as a keep:** inv 2 §9's UNRELATED and ADJECTIVE keep rows, inv 1
   App. B–D, inv 3 §9, and BR's KEEP_BARE locals.
10. **Every public name not in the token map.**
11. **`U` as a T3's own Tucker core in non-frame code** (critic r9):
    - `t3_linalg.py` outside 660–760: `U`, `Ux`, `Uy`, `Uxi`, `Uyi`, `Uw`, `UL`, `UR`, `Ua`, `Ub`, `Utilde`,
      `Uy_rev`;
    - `geometry.py:395–418`, `t3_operations.py:89`;
    - `t3_orthogonalization.py` (`Uio`, `Uox`, `U_i_o`), `oracle_sweeps/sweep_models.py:49–53`,
      `test_weighted.py:401` (`Ux`), `test_sharing.py:245, 598` (`U_rows`);
    - `test_tucker_tensor_train.py` (its `dU`, and the T3 kind `'U'`) and the T3-class doctests' `gU`
      (`tucker_tensor_train.py:3870–3871, 4072–4073`, `uniform_tucker_tensor_train.py:566–567, 631–632`).
12. **`U`/`u` meaning *uniform*:**
    - `UT3*` names; `UB`, `UV`, `UW`, `UF`, `UFb`, `Usc` (`test_dispatch.py:1705–1708`,
      `test_aligned_padded_frame_frontend.py:72`, `test_uniform_frame_variations_format.py:398`);
    - `oracle_sweeps/o1_uniform.py:193`'s `U`, a uniform T3; `utv_sampling`'s "U3";
    - the prefix `u` meaning uniform, and the `ut3_`, `ufv_`, `utv_` module and function prefixes. Names under a
      private `_u_` prefix change only where the token map lists them (`_u_up_messages` ↔ `_u_down_messages`, A2).
      The up-store locals are not uniform and change by tier C: `uc` in `ufv_conversions.py` and
      `uniform_frame_variations_format.py` (`uc→c_down`), and `um` in `ufv_conversions.py` and `ufv_operations.py`
      (`um↔dm`).

**K14 (DO-NOT-TOUCH identity).**

- For every file, the multiset of NAME tokens drawn from items 1–10 must be identical between `v2026.4.0` and the
  renamed tree. The identifiers are listed in R1's `dev/store_rename_dnt.txt`, file- or function-scoped where a letter
  also names a frame core (`U`, `Q`, `R`, `P`).
- So must the multiset of string arguments of `contract(`/`einsum(` calls.
- K14 has no exceptions. A legitimate need to change a listed name goes to Nick.

## 2. The compatibility surface

Full tables: inv 1 §2 (frontend) and inv 2 §2 (backend). Counts from BR §1.3, checked against the signatures.

- **Loud** (`AttributeError` or `TypeError` on stale code):
  - 4 dataclass fields, also constructor and `dataclasses.replace` keywords: `T3Frame.up_tucker_cores →
    down_tucker_cores`, `T3Frame.down_tt_cores → up_tt_cores`, and `UT3Frame`'s two supercores.
  - 18 backend core keywords in 14 functions (8 `probing.compute_*` with 10 keywords, 6
    `sampling_derivatives.compute_*_jets` with 8): `up_tucker_cores[_dot] → down_tucker_cores[_dot]`,
    `down_tt_cores[_dot] → up_tt_cores[_dot]`.
  - `update` on 4 functions. Under D-2(a), a 2026.4.0 string containing `'O'` raises `ValueError`; one without it
    (e.g. `'UL'`) is a silent swap and is listed below. Under (b), every letter string raises.
- **Silent** (same spelling, swapped meaning), 44 sites:
  - 18 attributes on 7 classes: `up_ranks`/`down_ranks` (`T3Frame`, `T3FrameWeights`, `UT3Frame`, `UT3Variations`,
    `UT3FrameWeights`), `up_weights`/`down_weights`, `up_weight_supercore`/`down_weight_supercore`,
    `UT3FrameMasks.up_mask`/`down_mask` and `UT3VariationsMasks.variations_up_mask`/`variations_down_mask`. The last
    three pairs are also constructor keywords.
  - 4 backend keywords: `up_ranks=`/`down_ranks=` of `ranks.frame_has_minimal_ranks` and `ufv_make_frame_masks`.
  - **The widths (D-8).** 16 keywords `nU=`/`nD=` in 8 functions: frontend `UT3Frame.from_t3frame`,
    `UT3Variations.from_t3variations`, `UT3FrameWeights.from_t3frameweights`, `UT3Tangent.from_t3tangent`; backend
    `ufv_conversions.t3frame_to_ut3frame`, `t3variations_to_ut3variations`, `t3frameweights_to_ut3frameweights`,
    `ufv_masking.ufv_make_frame_masks`. And 6 properties, `nU`/`nD` on `UT3Frame`, `UT3Variations`,
    `UT3FrameWeights`. Their `rL`/`rR` twins are unchanged.
  - **Nothing catches these.** Both stores give int tuples or masks of length `d`, and at a minimal frame the ranks
    agree, so stale code returns plausible numbers and is wrong only at slack. A keyword call
    `from_t3frame(F, nU=7, nD=5)` silently pads the other stores, and is wrong whenever the two widths differ.
    The release notes therefore lead with this list.
- **Unchanged:** the positional order of every tuple, `uniform_structure` and every `from_*` signature (positional
  width calls still work); `save`/`load`; every orthogonalization and SVD name; every optimizer keyword (`padded_*`,
  `cone_*`, `alternate=` carry no store word); every HISTORICAL contraction string (inv 2 §6); every number (§5.1).
- **Text only:** the reprs (`T3Frame(… down_ranks=…, up_ranks=…)`, `UT3Frame(… nD=…, nU=…)`) and store-naming messages.
- **Downstream:** T3Polynomial has one positional unpack (`scripts/g01_g0_common.py:121`), which keeps working; its
  locals are renamed after the release. `implicit_probing` does not import T3Toolbox, and the research repo makes no
  library call with these names (its prototypes keep the old words in their own locals).

## 3. Letters (Nick, 2026-11-11)

T4S is historical, and within T3Toolbox internal consistency beats consistency with the paper. **Bounded
instruction:** where a letter *names a frame core* in an equation, a diagram, a frame tuple, a code local or a shape
comment, the letter is `D`, `U`, `L` or `R`. The sites are exactly those enumerated in tier C (per file and function),
tier D (exact patterns), §3.4 (the `O`/`P`/`Q`/`U`-family hits classified as frame cores) and §4A (the inventoried
blocks).

- This covers the sampling kernels (`probing`, `sampling_derivatives`, `apply`, `entries`), their test twins and the
  four tex notes: there is no exempt file.
- It does not cover any name on §1.5's list, nor any letter whose meaning is something other than a frame core (an SVD
  factor, a completion, a direction, an RNG, a T3's own `B`). Those stay.
- A letter at a site not enumerated is not renamed. If it looks like a frame core, ask.

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
- **A T3's own Tucker cores**, which T4S writes `U`. Only these become `D`:
  - tier D's corewise-frame tuples (62 lines);
  - the frame and curvature prose sites §3.3 lists;
  - tier C's `U`-family row, occurrence by occurrence;
  - §3.4's hits classified as frame cores.

  Every other T3 `U` stays (§1.5 item 11).
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

The shift *functions* keep their names: `_shift_down`, `_shift_up` and their `_pair` forms, and `test_curvature.py`'s
`c_up`, are on §1.5's list. Their up/down is jet order, and the docstrings now define `Λ`/`Λᵀ` beside them. Only the
letters `D`/`U` used for the shifts in equations and comments change (the 41 lines). The note's macros become `\Sh` =
`\Lambda` and `\Sh^{\T}`.

### 3.3 Fixed rules

- **Frames:** the frame tuples tier D lists become `(D, U, L, R)`, and the corewise frame becomes `(D, G, G, G)` (D-6,
  Nick; 62 sites).
  - The T4S-derived Tucker names listed in tier C's row become `D` forms: in prose `Ḋ`, `D_g`, `δD`; in code `Ddot`,
    `dD…`, `Dd`, `gD`, `gYD`, `Dg`. Pairs stay symmetric with the unchanged `G` family: `(dD, dG)`, `(Ddot, Gdot)`,
    `(Dd, Gd)`.
  - A `U`-family name not on that list is not renamed without asking.
  - One site misnames a variation: `tv_operations.py:292–331`'s `dU = tt_variations[i]` → `H`. And `:401`'s
    `dU_i = <O_i, core_env_i>` becomes `dD_i = <U_i, core_env_i>`. No other site.
  - A point given by a frame's stores is written in store letters at exactly these sites: `fitting.py:380` (`(U, P)` →
    `(D, L)`), `ut3_sampling.py:374`, `shared_geometry.py:45`, and `curvature_internals.md:29, 47, 65–67`. More sites
    go to R1's list with the reviewer's agreement.
- **T3-level code and the T3 pages keep their letters** (D-6 residual, critic r9). `B` stays where the README, the
  user guide and the `TuckerTensorTrain` doctests write it, and `U` where T3-level code writes it (§1.5 item 11). Also
  unchanged: inv 3 C13's `Uᵢ` on `entries_apply_probe.md` §1 and §3 and on `t3m_methods.md`, `test_tucker_tensor_train.py`
  (its `dU` and kind `'U'`), and the `gU` doctests. Nothing here is renamed without Nick's ruling.
- **T4S's message directions that now read as store words** are reworded: "frame down jets"
  (`sampling_derivatives.py:1479, 1513`) → "the η jets"; "up-index" (`probing.py:773, 824`) → "the ξ index"; "down
  `eta`" (`apply.py:215`) → "the η sweep". Directions spelled as directions stay ("upward edge variables", "contract
  up", "pushing the remainder down into the Tucker core").
- **Cited T4S formulas:** exactly the 13 lines tier D lists convert, with the citation kept, e.g. "T4S §6.3's
  substitution `(L, R, U) → G`".

### 3.4 The `O`/`P`/`Q`/`U`-family inventory (R1; critic r2 N2, r5)

The inventories classified `U`/`O`/`D` only. They kept `P`/`Q` and the `U` family by design (inv 2 M4, C1), so the
rulings need their own inventory before any slice applies them.

- **Built in R1** as `dev/store_rename_pqu_inventory.md`: every NAME token and every text occurrence of `O` and its
  derived names outside tier C's files (critic r5), and of `P`, `Q`, their derived names (`P_cores`, `P_cores_dot`, `P_dot`, `Pd`, `Pxi`, `dPxi`, `Q_dot`, `Qd`, `Qxi`, `rev_Q`, `P_last`, …),
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
    `ut3_sampling.py:114, 231` and `entries.py:152, 290`. Kept (T3-level, §1.5 item 11): the T3-class doctests
    `tucker_tensor_train.py:3870–3871, 4072–4073` and `uniform_tucker_tensor_train.py:566–567, 631–632`.
  - the `O` family outside tier C's files: `t3_orthogonalization.py:118–120` (`O_ab_x`, `Oaxb`: the up-orthogonal TT
    core, beside the T3's input Tucker core `Uio`) and `ut3_orthogonalization.py:201–208`. Disposition: **kept**. They
    are T3-level SVD outputs beside a T3's `Uio` (§1.5 items 2 and 11), and K8 carves them out; only the return
    comment's adjective is corrected in prose. Also kept: `shared_t3_math.tex:358`'s `O_g`, a complement basis (K8
    carves it out).
- **R1 gate:** every hit is classified. R2's tier C and R4's L1/L2 apply only what the classification marks.

## 4. Hand edits (R3–R5)

1. **Convention statements** (about 69: inv 2 §3 25, inv 1 §3 35, inv 3 9). Rewrite them from the rule, not by word
   swap. Their home is `docs/frame_variations.md` § "Four rank stores", which gets inv 3 §1(a)'s paragraph, picture and
   T4S note. Only the listed convention-statement docstrings (inv 1 §3, inv 2 §3) are rewritten, each stating the
   convention in one clause and linking there; no other docstring gains a convention sentence.
2. **Sentences that use both senses** (edge direction and store side, inv 3 C19). Rewrite them, e.g. "the stores
   pointing toward the root (up, left)" becomes "the stores whose edges point toward the root (down, left)".
3. **Backwards adjectives, 28 lines.** Frontend: `frame_variations_format.py:1371–1372`,
   `uniform_frame_variations_format.py:1122, 1125`, `tucker_tensor_train.py:615, 2956, 2978, 3002, 3108`. Backend:
   `probing.py:646–647, 1221–1222`, `t3_orthogonalization.py:147, 171`, `ut3_orthogonalization.py:189`,
   `geometry.py:348`, `t3_svd.py:312, 345`, `fv_conversions.py:181–182, 184–185` (the aliases, by A1), `:219, 251`.
   Tests and docs: `test_frame_variations_format.py:447`, `test_tucker_tensor_train.py:3202`, `user_guide.rst:142`.
   (`CHANGELOG.md:714` is a record and stays.)
4. **Letters in prose:** the tier D patterns; the CL rows of inv 1 §7 and inv 2 §9 that fall in comments and
   docstring *prose*. Doctest example lines are code (NAME tokens), so R2 renames them via `letter_sites.md`, scheduled
   once and checked by K15. Also the field comments `frame_variations_format.py:156–159` and `uniform_frame_variations_format.py:110–113`
   (`B_xo`→`D_xo`, `R_ixj`, which clashes with *right*, →`U_ixj`, `P`/`Q`→`L`/`R` by D-7). Every diagram and equation
   is a §4A block, redrawn by hand as a whole with `D` below, `U` above and the compact width letters in the new
   meaning (D-8).
5. **The `update` values (R3)**, §1.3, per D-2: the 37 backend and 93 frontend/test LETTER-API rows.
   - **Under (a):** the scoped letter swap (`'U'`→`'D'`, `'O'`→`'U'`) inside those rows' strings and `in update` tests.
     The tests' letter arithmetic is re-checked by hand (`test_padded_frame.py:723–737` builds subsets by deleting
     letters, and `:730` pairs `('U', 'up'), ('O', 'down')` → `('D', 'down'), ('U', 'up')`). The invalid cases gain
     `'ULRO'` (the `'O'` message) and `'dulr'`, and keep `''`, `'DD'`, `'X'`, `'DULRX'`. The recorded schedules
     `('ULRO', t)` become `('DULR', t)`.
   - **Under (b):** the word-tuple rewrite, as before: invalid cases `(), ('down', 'down'), ('X',), 'down', 'ULRO',
     ('Down',)`.
   - **Under (c):** (a)'s swap, then each string split into a tuple of letters. Invalid cases: `()`, `('D', 'D')`,
     `('X',)`, `'DL'` (a `str`), `'ULRO'` (a `str` with the `'O'` message).
   - The dense oracle's letter API follows the same form.
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
10. **Signature blocks realigned** wherever a name changed length (`up_tucker_cores` → `down_tucker_cores`), per
    `docs/contributor/signature_style.md`.
11. **Typos.** In R2, with the script, because after A1 they would read as right: `fv_conversions.py:58, 157`
    `down_tucker_supercore`→`up_tt_supercore`, and `:60, 159` `right_tucker_supercore`→`right_tt_supercore`. In R4a:
    `tucker_tensor_train.py:4264` `rL_0*r_1`→`r_0*r_1`. These loop variables named for the old store, and no others, are renamed by hand:
    `fv_operations.py:653–654`'s `dn`, `u` → `w`; `u`/`o` at `test_padded_frame.py:491` and
    `_uniform_padded_frame_testing.py:41`.
12. **Generic kernel shape lines** (critic r7). R2's swap writes `nD`/`nU` into the shape comments of generic
    `tucker_cores`/`tt_cores` kernel parameters: `sampling_derivatives.py:388, 416, 473, 518, 872, 995, 3236`,
    `probing.py:550`, `ufv_conversions.py:334–335`. Each is a tier-E row in R4a/R4b, disposed by its call sites:
    - a parameter that receives a frame store keeps the store's letter (`sampling_derivatives.py:473`'s `tt_cores`
      holds the up store, so its old `(rLi,nOi,rR(i+1))` becomes `(rLi,nUi,rR(i+1))`);
    - one that receives a T3's own cores, or either, takes the generic `n` for its width (`rLi`/`rR(i+1)` stay
      unchanged, per D-3).

    The reviewer confirms each, and K13 lists them.

## 4A. Diagrams and equations: hand-only, first-class (Nick, 2026-11-11)

The hand-crafted ASCII diagrams and the equations in docstrings and docs are of great value to readers, and every one
must come through the rename **intact**. They are artifacts, not text: no script, sentinel or regex touches them.

**(a) The inventory.** R1 builds `dev/store_rename_slices/diagrams_inventory.md` and commits it. It covers every ASCII
diagram and every equation (docstring, comment, markdown or LaTeX) that mentions a store letter, a width symbol or an
orthogonality relation.

- Columns: id; file; the old **span** at `v2026.4.0` (lines, plus a column range when the block shares its lines with
  code, as the field comments do); kind (ascii diagram, index equation, shape listing, field-comment relation, LaTeX
  table or figure, or a companion doctest); the span text (the anchor); the letters and widths in it; the new span
  (filled in R4/R5); re-laid out (y/n and why); the reason when new equals old.
- A scanner proposes candidates and R1 curates them. Candidates are lines with box or edge characters (`--`, `|`, `(X)`
  nodes) next to labels; index relations (`_[a-z]{2,3}` with `= delta_ij` / `= I_xy` / `= 0`, `einsum('…')`, `δ_ij`);
  width symbols; a store letter in a core role (`U0`, `O_i`, `D(d-1)`, …); or a store-word index variable in a
  recurrence (`\b(?:up|down)_[a-z]\b`, as in `ranks.py:644–647`).
- **R1 gate:** every candidate is inventoried or dismissed with a reason.
- **Seed list**, from the inventories:
  - `frame_variations_format.py`: 56–66, 70–78, 86–108, 116–124, 156–159, 219–224, 353–357, 843–853, 1262–1274,
    1305–1322, 1346–1365, 1370–1374 (R1 takes the exact bounds from the scanner);
  - `uniform_frame_variations_format.py`: 110–113, 1097–1125;
  - `manifold.py:685`, `uniform_manifold.py:478`; `probing.py:640–651, 1215–1226`; `ufv_operations.py:81–84`;
    `utv_operations.py:314–315`; `ranks.py:644–647`;
  - `tucker_tensor_train.py:61–72` (a B/G diagram: to be dismissed, with its reason);
  - companion doctests, inventoried but not fenced: `frame_variations_format.py:1302–1322`. Their example lines'
    letters are rows of R1's `letter_sites.md`, renamed in R2 (K15 checks them) and listed in K12.
    The reviewer checks that their letters agree with the diagram beside them;
  - `frame_variations.md:13–17` and its symbol table; `user_guide.rst:124–142`;
  - `padded_frame_cone.tex` Table 1; the exact-Hessian note's two tikz figures and its equation ranges;
  - the four notes' frame equations.

**(b) Each one rewritten by hand, as a whole.** Never token by token. The slice handoff shows every block's before and
after side by side.

**(c) Alignment preserved exactly.** `dev/store_rename_diagram_check.py` compares each ascii block's old text (`git
show v2026.4.0:<file>`) with its new span, as **raw text** (tabs not expanded: `user_guide.rst`'s diagrams are
tab-indented). **K10**, for blocks rewritten in place, requires:

- the same line count;
- the same length per line;
- that every column where old and new differ holds a label character (`[A-Za-z0-9_']`) on both sides, so box
  characters, pipes, dashes, parentheses and spaces stay in their columns.

D-8 keeps the labels' length (`U0`→`D0`, `nU0`→`nD0`, `nO1`→`nU1`, `rR(d-1)` unchanged). A label that must grow
makes the block a deliberate re-layout: it is marked in the inventory, shown in the handoff, never truncated, and
checked by **K10b** instead.

- K10b requires the same line count.
- It requires every old alignment column to map to **one** consistent new column. An alignment column is a field start
  after two or more spaces in at least two lines, or a shared separator (`=`, `:`, `|`, `#`).
- Most equation and listing blocks are re-layouts, because their identifiers grow: `frame_variations_format.py:70–78,
  116–124, 353–357`, `probing.py:640–651`, `ufv_operations.py:81–84`.
- For field-comment blocks, the comment's start column is checked against the realigned signature block (§4.10).

A typo inside a diagram (`Ud` for `U(d-1)`, glossary E17) is left to the docs-consistency pass unless it fits the width.

**(d) Orthogonality relations re-derived index by index.** Each relation gets the new letters and a one-line derivation
in the handoff: the core's shape, the contracted indices, and the side they belong to, which is the store's adjective.
For example:

- `D_ia D_ja = delta_ij`: the down core `(n_i, N_i)` is summed over the leaf index, below the edge;
- `U_aib U_ajb = delta_ij`: the up core `(r_i, n_i, r_{i+1})` is summed over both bonds, above the edge;
- `L_abi L_abj` and `R_ibc R_jbc`;
- the field comments `D_xo D_yo = I_xy` and `U_ixj U_iyj = I_xy`;
- the gauge condition `D_ia V_ja = 0`.

The reviewer re-derives each one independently and checks it against the code's own contraction where there is one
(e.g. `is_orthogonal`'s `einsum('...io,...jo->...ij', D, D) = I`).

**(e) The doctests beside them run unchanged.** Every module doctest's expected output is identical before and after
(a `DocTestFinder` comparison over the frontend's 2,047 examples and the backend's). Only input lines change, through
the renames.

**(f) Review.** The reviewer's brief for R4a, R4b and R5 includes reading every inventoried block against the new
convention. Width labels are checked by adjacency: an edge touching a Tucker factor is `nD`, one touching a TT-shaped
frame core is `nU`. The review stop after R4b shows Nick the before and after of every code-tree block, and the stop
after R5 those of the docs and notes.

**(g) Fenced from every script, by span.**

- The script reads the inventory's spans as FENCES: a line range, plus a column range where the block shares its lines
  with code.
- Pass 1 skips all text inside a fence (COMMENT and STRING content, markdown and LaTeX lines). So do tier D and every
  regex pass of R4/R5.
- NAME tokens next to a fenced span are still renamed. A field comment shares its line with the field
  (`frame_variations_format.py:156`), so R2 renames the field and the comment shifts by the name's length difference
  (±2 columns), which §4.10's realignment absorbs.
- No `>>>` example line is fenced.
- **Gate K11**, after R2 and after R3: every fenced span's **text** is byte-identical to `v2026.4.0` (located by the
  span text, not the line), and the pass-1 tree has no sentinel inside a fence.
- **Gate K12** (completion), after R4b for the code tree and after R5 for the docs: every inventoried block's new span
  is filled and equals the file verbatim, and a block whose new text equals its old text carries a recorded reason.
  Without it, a forgotten block with no `O` or `nO` (`frame_variations_format.py:86–96`: `nU0`, `U0`, `Ud`) would pass
  every other check while reading backwards.

**Letters inside fences** are the code's (D-8), for the labels the inventory records as frame cores or store widths:
core letters `D`, `U`, `L`, `R`; widths `nD`, `nU`, `rL`, `rR`, with `nO` gone. Other letters in a block (`G`, `V`,
`H`, `N`, an SVD's `U`) stay. The hand rewrite swaps them, the reviewer checks by adjacency (an edge touching a Tucker factor is `nD`, one
touching a TT-shaped frame core is `nU`), and K2 (stale spellings) and K8 (`O` as a core letter) hold everywhere,
fences included.

## 5. Verification (the rename changes no number)

### 5.1 The bitwise battery: `dev/store_rename_battery.py`

The same seeded calls run on two trees in one environment: the reference, a `git worktree` of `v2026.4.0` (R1), and
the renamed tree (after R2, R3, R4b and R6). Results go to the scratchpad as `{key: array | text}`, never as `*.npz` in the
repo.

**The adapter** is the only tree-dependent code:

- `NEW = 'up_tt_cores' in {f.name for f in dataclasses.fields(bvf.T3Frame)}`. `hasattr` is False on both trees, because
  a field without a default is no class attribute.
- `upd(stores)` gives the new tree's form: a letter string under D-2(a) (`down→D`, `up→U`), words under (b), a letter
  tuple under (c). On the old tree it gives the old letters (`down→U`, `up→O`, `left→L`, `right→R`).
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
  `UT3Frame.from_t3frame(F, 8, 7, 5, 6, 3)` (positional; padded widths `nD = 7`, `nU = 5`, `rL = 6`, `rR = 3`, all
  different, so a swapped width keyword or property shows).

**Calls** (every array leaf is recorded; ints and tuples are recorded as arrays):

| id | calls |
|---|---|
| B1 representations | `t3_orthogonal_representations(x)` for `xS`, `xC`, `xG` → `(F.data, V.data)`; `T3Frame.from_t3(xS)`; `MANIFOLD.frame(xS)`; `T3Frame.random_orthogonal((5, 6, 7), (3, 6, 2), (1, 2, 5, 1), (2,))`; `t3svd_orthogonal_representations(xS)`; `fv_operations.fv_zero_point_frame(F.data)` and with `groups`; `ut3_orthogonal_representations`; `UT3Frame.from_t3frame(F, 8, 7, 5, 6, 3)` and back with `to_t3frame()`; `UT3Variations.from_t3variations(V, 8, 7, 5, 6, 3)`; the default-width forms `UT3Frame.from_t3frame(F)`, `UT3Variations.from_t3variations(V)` (their `nU = int(max(b.up_ranks)) if nU is None` lines) |
| B2 renamed attributes, keyed by physical store | the old names on the old tree and the new names on the new: `T3Frame` cores and the two ranks; `T3FrameWeights` (from `T3Weights.from_t3svd`) fields and ranks; `UT3Frame` supercores, ranks, the four widths and `masks` fields; `UT3Variations` ranks, widths and `masks`; `UT3FrameWeights` supercores, ranks and widths; the reprs of all five classes and of `T3Tangent` (text compared after the A-tier map) |
| B3 rank and mask functions, positional | `ranks.compute_orthogonal_representation_ranks` on `xS`'s structure and a stacked host array; `ranks.frame_has_minimal_ranks(shape, *stores, None)` on `xS` (False) and `x0` (True); `ufv_masking.ufv_make_frame_masks(*stores, 7, 5, 6, 3)`; `ufv_masking.ufv_frame_widths` |
| B4 orthogonalizations and SVD | `xS.down_orthogonalize_tucker_cores()`, `.up_orthogonalize_tt_cores()`, `.left_orthogonalize_tt_cores()`, `.right_orthogonalize_tt_cores()`, `.t3svd()`, `.t3svd(max_tt_ranks=2)`; the uniform twins; `xG.t3svd()` |
| B5 tangents | `v = MANIFOLD.project_ambient(F, y)`; `retract`, `transport` to `MANIFOLD.frame(x0)`, `inner`, `norm`; `v.to_dense()`, `v.to_t3()`, `v.gauge_residual()`; a `K`-stacked tangent `T3Tangent.stack_tangents((v, 2·v))` through `to_dense`, `inner`, `retract`; `fv_absorb_weights` with the B2 weights; the uniform twins via `UT3Tangent.from_t3tangent(v, 8, 7, 5, 6, 3)` and the default `UT3Tangent.from_t3tangent(v)` (its `dict(nU=nU, …)` forwarding), and `uxC`'s; `COREWISE.frame(xS)`; the shared and tied paths: `sharing.ufv_shared_frame_data`, `utv_operations.utv_orthogonal_gauge_projection(…, shared_data=…)` (the `up_mask_frame`, `up_mask_v`, `down_slices` code, `utv_operations.py:230–254`), `shared(MANIFOLD, sharing).project/retract` and `shared(UNIFORM_MANIFOLD, sharing)` on `xG` |
| B6 sampling | `v.probe(ww)`, `.apply(ww)`, `.entries(idx)`, `.probe_derivatives(ww, pp, 2)`, `.probe_transpose`, `.probe_derivatives_transpose`; `xS.probe_derivatives(ww, pp, 2)`; the uniform twins; the 14 keyword-renamed `probing.compute_*` / `sampling_derivatives.compute_*_jets`, positionally |
| B7 padded frames | `T3Frame.padded(x_old, tk1, tt1)`, also at the slack target and with `sharing`; `UT3Frame.padded`; `T3Tangent.align_padded` and `UT3Tangent.align_padded` with `upd(u)` for all 15 non-empty subsets of the four stores, plus one permuted form (`'LD'` against `'DL'` under D-2(a), `('left', 'down')` under (b): the canonicalization), at the minimal and the slack target; the frontend `bvf.fv_aligned_padded_frame(x_old, tk1, tt1, y, max_passes=m, alternate=a)` for `(m, a)` in `(0, True), (3, True), (3, False), (10, True)`; the slack-old-frame backend step and the backend loop `padded_frames.fv_aligned_padded_frame(old_frame, tucker_ranks, tt_ranks, hint, project, …)`; `ufv_aligned_padded_frame` the same, plus `use_jit=True`; `backend.optimizers.padded_start(setup.problem, setup.x0_cores, x_old.data)` |
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
assertion per silent site (§2: 18 attributes, 4 rank keywords, 16 width keywords, 6 width properties) at `xS`'s
structure, the widths at `UT3Frame.from_t3frame(F, 8, 7, 5, 6, 3)` (`F.nD == 7`, `F.nU == 5`). Each checks the attribute against the
physical core shapes, e.g. `F.down_ranks == tuple(c.shape[-2] for c in F.down_tucker_cores)` (the weights are built
per store from `F`'s ranks, since `from_t3weights` copies one Tucker vector into both). They go in each class's
test module (`test_frame_variations_format`, `test_uniform_frame_variations_format`, `test_weighted`,
`test_uniform_weighted`, `backend/test_ranks`). The battery uses unequal stores and widths throughout.

### 5.3 The completeness check: `dev/store_rename_check.py` (code-tree part permanent per D-9)

**Patterns:**

| id | what | pattern |
|---|---|---|
| K1 | old spellings | `\b(?:masked_)?(?:up_tucker_cores\|down_tt_cores\|up_tucker_supercore\|down_tt_supercore\|up_tucker\|down_tt\|up_tucker_ranks\|down_tucker_ranks\|down_tt_ranks\|up_orthogonalize_tucker_cores\|down_orthogonalize_tt_cores)(?:_dot)?\b` |
| K2 | stale width spellings (D-8: nothing can grep the meaning of `nU`/`nD`) | `(?<![\\\w])nO` (not an escape's `\nO`), LaTeX `n_\{O` and `n\^O`, and any word width `\b(?:n_up\|n_down\|r_left\|r_right\|k_up\|k_down\|m_up\|m_down)\w*` (none may appear anywhere: Nick). This holds everywhere, fences and notes included. The proof of the `nU`↔`nD` swap is the battery on slack frames (§5.1), the bijection check, K13, and the reviewer's adjacency reading |
| K3 | the `update` values (per D-2) | Under (a): in the LETTER-API files (`padded_frames`, `uniform_padded_frames`, `manifold`, `uniform_manifold`, `frame_variations_format`, `uniform_frame_variations_format`, the padded-frame tests and oracles, `test_dispatch`, `test_aligned_padded_frame_frontend`) no update string or `in update` test contains `'O'`: `['"][DULRO]*O[DULRO]*['"]` on update lines, and `['"]O['"]\s+(?:not\s+)?in\s+update`. In docs: `` `'(?:ULRO\|RO)'` ``. Under (c): (a)'s `'O'` checks plus `update\s*:\s*str\b` and `update\s*=\s*['"]`. Under (b): `update\s*:\s*str\b`, `update\s*=\s*['"]`, `['"](?:[ULRO]{2,4})['"]`, `['"][ULRO]['"]\s+(?:not\s+)?in\s+update`, and in docs `` `'(?:ULRO\|UL\|RO)'` `` |
| K4 | frame-letter tuples, hybrids included | `\(U, ?O, ?[PL], ?[QR]\)`, `\((?:D\|U), ?(?:U\|O), ?P, ?Q\b` and `\(U, ?G, ?G, ?G\)` |
| K5 | old-order listings | `\(up, down, left` and `up/down/left/right` |
| K6 | backwards phrases | `Tucker up-(?:SVD\|orth)`, `Tucker cores upward`, `TT cores downward`, `down-orthogonal O\b`, `up-orthogonal tucker`, `Down TT core` |
| K7 | old shift and derivative notation | `\b[DU]\^[rku]\b`, `\[U\^`, `\b[DU] (?:x\|z\|jet)\b` (catches `D x =`, `# U jet,`, `[jet, D jet]`), `\bDPhi\b`, `\bD\^2 ?Phi\b`, `DΦ`, `D²Φ`; in tex `\\Dsh\|\\Ush\|D\\Phi\|D\^2\\Phi\|D_\\zeta` and the Lagrangian's old `\\Lambda\(\\th\|_\\th\\Lambda` |
| K8 | T4S letters left behind | `O` as a core letter everywhere, fences included: `\bO(?:\d\|_[a-z]\|ᵢ\|i\b\|\(d-1\))`; `\b(?:O_i\|O_aib\|Oᵢ\|δU\|U_g)\b`, `U̇` (no trailing `\b`: it never matches after a combining mark), `outer[- ]orthogonal`; frame `P`/`Q` forms `\b[PQ]_(?:i\|last\|\{)`; the code names `\b(?:Udot\w*\|dU(?![a-z])\w*\|gU\|Ug)\b` (so the new `dUn`, from `dOn`, is not hit). Carve-outs: the T4S correspondence lines, the §3.4 inventory's keep list, and for the `O` pattern the oracle and review codes (`O1`, `O2`, `O-2`; inv 3 §9) and the kept `O_g` (`shared_t3_math.tex:358`), and the T3-level SVD outputs `O_ab_x`, `Oaxb` (`t3_orthogonalization.py:118–120`, `ut3_orthogonalization.py:201–208`) |
| K0 | sentinels | `ZQX\d+_XQZ` anywhere (after pass 2) |
| K8b | T4S string keys | `\[['"][OPQ]['"]\]` in `tests/test_jet_recurrence.py` (after B4, `p['U']` is the correct up-core key) |
| K9 | residual store words, token-level (critic r1 M1, r2 MINOR 1) | every bare `up`, `down`, `Up`, `Down`, and every store-word index variable `\b(?:up|down)_[a-z]\b` (`up_i`, `down_i`), in the COMMENT and STRING tokens of the final code tree (and every bare word in the docs after R5) must be one of three things: (a) an inventory keep row (ORTH, MSG, UNREL; inv 2 §9 keeps, inv 1 App. B–D, inv 3 §9, collected in R1 as `dev/store_rename_keeps.txt`); (b) a word whose aligned old-tree token was the opposite store word (`git diff` line alignment, then token position); or (c) part of a tier-E proposed text after §1.4's layering. Anything else fails. **Converse** (critic r3): every old-tree token on a keep row (ORTH, MSG, UNREL) survives unchanged at its aligned position, unless its line is a tier-E proposed text (e.g. §3.3's rewordings). So clause (b) cannot pass an adjective or a message direction flipped by mistake ("up-orthogonal" → "down-orthogonal" on a TT core, "contract up"), which K6's six phrasings would miss |
| K10 | diagram alignment, in place (§4A c) | `dev/store_rename_diagram_check.py` on raw text: per inventoried ascii block rewritten in place, the same line count and line lengths, and every differing column a label character on both sides |
| K10b | diagram alignment, re-laid out (§4A c) | the same line count, and every old alignment column (a field start after ≥ 2 spaces in ≥ 2 lines, or a shared `=`, `:`, `\|`, `#`) maps to one consistent new column. Field-comment blocks: the comment's start column matches the realigned signature block |
| K11 | fences untouched by scripts (§4A g) | after R2 and after R3, every fenced span's text is byte-identical to `v2026.4.0` (located by span text), and the pass-1 tree has no sentinel inside a fence |
| K12 | every block done (§4A g) | after R4b (code tree) and R5 (docs): each inventoried block's new span is filled and equals the file verbatim; new text equal to old needs a recorded reason |
| K13 | shape comments agree with their entry (D-8) | Over every single-entry shape line: a signature or field comment `NAME: type  # …` and a docstring listing `- NAME = (…), elm_shape=(…)`, plus their uniform supercore forms. Extract the entry name and every width token `\bn[UD]\w*` in its shape, and require the letter the table gives: **`nD`** for `down_*` entries (`down_tucker_cores[_dot]`, `down_tucker_supercore`, `down_weights`, `down_weight_supercore`, `down_mask`, `variations_down_mask`, `down_ranks`), for `left_tt_*`/`right_tt_*` (their middle leg), and for `tt_variations` / `H` (middle leg); **`nU`** for `up_*` entries (`up_tt_cores[_dot]`, `up_tt_supercore`, `up_weights`, `up_weight_supercore`, `up_mask`, `variations_up_mask`, `up_ranks`) and for `tucker_variations` / `V` (its rows: the complement directions live in the up width). K13 recognises **only** store-named entries and `V`/`H`. Generic `tucker_cores`/`tt_cores` lines are reported to the reviewer, not checked, because a generic kernel parameter may hold a frame store. The ten known ones get a disposition in §4 item 12. A shape line with width tokens but no recognised entry is reported, not failed. This corrects the sketch "tucker → `nD`, tt → `nU`", which the variations and the left/right cores contradict |
| K14 | DO-NOT-TOUCH identity (§1.5) | per file, the multiset of NAME tokens on R1's `dev/store_rename_dnt.txt` (file- or function-scoped for `U`, `Q`, `R`, `P`) and of `contract(`/`einsum(` string arguments is identical between `v2026.4.0` and the renamed tree. No exceptions |
| K15 | the token map is closed (critic r9 O1) | Per file, on NAME tokens (code and doctest examples). **After R2:** R2's edits are in-place token replacements, so the old and new NAME sequences have equal length, and at every position new equals old unless R1's hit table lists that (file, line, column) with exactly that image. **After R3, R4a, R4b and R5:** `difflib` the NAME sequences against the previous slice's commit; every unequal hunk must be one the slice's handoff lists (R3: the `update` helper, steps, loops, tests; R4: §4.11's loop variables, §4 item 12's lines, §3.4's R4 dispositions). A prose-only slice leaves the NAME stream identical. Cross-check: old − new ⊆ map sources and new − old ⊆ map images ∪ the handoffs' new names (`_update_stores`, `_STORE_LETTERS`, `_COLOURS`, `n_cores`, `start_dense`, the `c`-suffixed locals, …) |

**Scope:** `t3toolbox/`, `tests/` and `examples/` (fourteen scripts, nothing to change, checked all the same) for
K0–K15; `docs/*.md|rst`, `docs/contributor/*`, `README.md` and `CLAUDE.md` for all but K3's code patterns and K13;
the tex notes for K2, K4, K7 and K8.

**Carve-outs, all of them:** `CHANGELOG.md` and `docs/release_notes.md` (records; the new sections name the old
spellings on purpose); lines of `docs/glossary.md` containing "(2026.4.0"; the T4S correspondence lines (§3, for K4
and K8); the paper ↔ code maps (`entries_apply_probe.md` §8, `batching_and_stacking.md` §8); the check file itself.
K3's `{2,4}` skips single-letter strings.

Before R2 the check runs in report mode, and its hits are the to-do list. After R4b it is clean on the code tree, and
after R5 on the docs. The script's per-file swap bijection (§1.3) is the R2-only check for A2 and the width swaps, which no grep can do.
K9 is the only check that sees a bare word left in the old sense.

### 5.4 Suite, doctests, docs, PDFs

- The full suite at `-n 3` (the 13 GB laptop rule), in the current-leg environment and in the floor environment
  (numpy 1.26 / jax 0.4.30). The slow tier runs serially at R6, as the release gate.
- Module doctests: the frontend's 2,047 examples (inv 1 §4 found no expected-output change) and the backend's; every
  `docs/*.md` and `docs/contributor/*.md` page plus `getting_started.rst`; both environments. **Expected outputs are
  unchanged** (§4A e): a `DocTestFinder` comparison of every example's `want` before and after, at R2, R4a, R4b and R5.
- The dispatch rows. A word tuple stays a jit static: `_STEP_STATICS` position 7, asserted at
  `test_dispatch.py:2416–2418`. Then the import-graph test, `sphinx -W`, and `gh run list` after each push.
- PDFs rebuilt with `latexmk -pdf`: `padded_frame_cone.pdf`, `exact_hessian_two_hole.pdf`, `shared_t3_math.pdf`,
  `pad_safe_svd.pdf`. `symmetric_probe_derivatives.tex` and `ttm_t3m_ht_note.tex` have no PDF; their sources are
  edited only.

## 6. Documentation and release notes (R5)

- **`docs/frame_variations.md` § "Four rank stores"** gets the convention paragraph, the picture and the T4S note
  (inv 3 §1(a)), and every other page links to it. `naming_conventions.md` gets the bullet "frame stores are words".
  The letter ↔ word mapping is stated once, in the convention paragraph and the glossary's symbol table: `D`/`U`/`L`/`R`
  for the stores; `nD` the down store's width (the Tucker factor's), `nU` the up store's (the TT-shaped core's), `rL`,
  `rR` the left and right stores' (D-8).
- **`CHANGELOG.md` `[2026.5.0]` "Changed — breaking"** and **`docs/release_notes.md` "Upgrading from 2026.4.0"** come
  from inv 3 §1(b), in house format: the verdict first, then the same-spelling list, stated as silent, full stop. That list includes
  the width keywords `nU=`/`nD=` and the properties `.nU`/`.nD` ("these do not raise"; no claim that a shape check
  catches misuse, since at a minimal frame both stores agree). Then the old → new table with its "raises if left?"
  column; the one-pass migration snippet, whose `SWAP` map gains the `_dot` forms (`\b` does not match inside them)
  and swaps `nU`↔`nD`, with its `rL`/`rR` entries dropped because they are unchanged; what does not change (positions,
  files, numbers, method names); the letters and the T4S correspondence; and the `update` form chosen under D-2.
  **inv 3 §1(b)'s bullet "No store letters in the API" is not used.** In its place, the release notes enumerate exactly
  what changed (§0's list) and what did not (§1.5's summary), with no general rule about letters. The root
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
| **R1** | Tools and inventories only, no library change: the battery (§5.1) and its reference run on the `v2026.4.0` worktree (plus the B9 files); `store_rename_apply.py` (§1.3) in dry-run mode (pass 1 into a scratch copy); `store_rename_check.py` (§5.3) in report mode; the `P`/`Q`/`U`-family inventory (§3.4) and `dev/store_rename_keeps.txt` (K9); the diagrams-and-equations inventory `dev/store_rename_slices/diagrams_inventory.md` (§4A a, committed), the FENCES drawn from it, and `store_rename_diagram_check.py` (K10) | the dry-run hit table reconciled key by key with inv 2 §8, inv 1 §7 and a fresh tokenizer scan (every difference explained); every CL row and every §3.4 hit on `store_rename_letter_sites.md` or excluded with a reason; `dnt.txt` built and disjoint from the token map; every §4A scanner candidate inventoried or dismissed; nothing outside `dev/` changed |
| **R2** | The script (pass 1, review of the sentinel diff, pass 2), tiers A–C, over `t3toolbox/`, `tests/` and `examples/`, the four sampling modules and their twins included; tier A over the docs and `CLAUDE.md`, so the markdown doctests stay green; the 12 message tests and the 4 typo lines (§4.6, §4.11) | K0, K11, K13, K14 and K15 clean; battery bitwise against R1; no bijection errors; doctest outputs unchanged; the full suite, the module and markdown doctests, and the dispatch rows green. **Review stop** |
| **R3** | The `update` values (§1.3, §4.5, per D-2) in the backend steps, loops and kernels and the frontend twins, with its tests; the unequal-store assertions (§5.2) | battery bitwise (through the adapter); full suite; dispatch; K3, K11, K14 and K15 clean |
| **R4a** | Code prose outside the sampling kernels: every tier D and tier E row in `t3toolbox/`, `tests/` and `examples/` (§4.1–4.4, 4.7, 4.10, 4.11), the free-prose store words included; every §4A block in those files rewritten by hand, before/after and the relation derivations in the handoff | suite; doctests in both environments with outputs unchanged; `sphinx -W`; K0–K10b and K13–K15 clean on those files; the reviewer has read every block |
| **R4b** | The sampling kernels' prose (`probing`, `sampling_derivatives`, `apply`, `entries`, `utv_sampling`, `ut3_sampling`, the jet twins and curvature tests): their tier D and E rows, the shift and derivative notation (§3.1–3.2), the reworded message directions (§3.3), their §4A blocks by hand | battery bitwise; suite; doctests with outputs unchanged; K0–K15 clean on the code tree; the reviewer has read every block. **Review stop**: Nick sees the before/after of every code-tree diagram and equation (R4a's and R4b's) |
| **R5** | Docs (§6): pages, the six notes (§4.8) and their PDFs, the glossary and its work list, `CLAUDE.md`, CHANGELOG and release notes; every §4A block in the docs and notes by hand (diagrams, tables, tikz labels, equations), before/after in the handoff | `sphinx -W`; markdown doctests in both environments; K-checks clean on the docs and notes; PDFs rebuilt; K10/K10b on the docs' ascii blocks and K12 on the docs; every example runs (`examples/*.py`, as CI's examples job does). **Review stop**: Nick sees the before/after of every docs and notes block |
| **R6** | Acceptance: §5 end to end in both environments, the slow tier, the battery also in the floor environment; the T3Polynomial follow-up prepared (`g01_g0_common.py:121`, its suite), committed only after the release; HANDOFF | — |

**The implementer's brief (Nick's words, 2026-11-11), read before every slice.**

- There is no escape from case-by-case judgement: no rule in this plan may be applied blindly or beyond its enumerated
  scope.
- Any rename that is not in the token map (§1.2), §3.4's classified inventory or §4A's inventoried blocks is forbidden
  without asking. That includes "obvious" extensions of a pattern to a site the plan does not list.
- §1.5's names are never touched (K14).
- When a site looks like a store but is not listed, stop and ask the session. The reviewer checks for over-reach as
  hard as for misses: every changed token must trace to a token-map row, a classified hit or an inventoried block.

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
- **D-2. REOPENED on its merits (Nick, 2026-11-11): the form of `update`** (§1.3).
  - **(a) Letter strings in the new meaning** (`'DULR'`, any subset, any order, default all four). Validation names the
    four letters and rejects `'O'` explicitly ("'O' was the 2026.4.0 letter for the TT-shaped store, now 'U'"). Pro:
    consistent with the diagrams, the widths (`nU`/`nD`) and the LaTeX. The `'U'` inversion is the same accepted
    same-spelling swap as `nU`/`up_ranks`, and every 2026.4.0 string containing `'O'` (all four colours of the default
    schedule) fails loudly. Con: an old `'UL'` silently means down+left now.
  - **(b) A tuple of words.** Pro: it cannot be misread across the version boundary, and an old letter string raises.
    Con: a second vocabulary beside `nU`/`nD` and the pictures.
  - **(c) A collection of single-letter strings in the new meaning** (`('D', 'L')`, any order, default
    `('D', 'U', 'L', 'R')`), with a `str` refused (critic r9). Pro: it keeps Nick's letters, consistent with the
    pictures and widths, yet every 2026.4.0 call (always a `str`) fails loudly. A `str` containing `'O'` still gets
    the pointed message. Con: tuple syntax for single letters (`update=('U',)`).
  - **(a)'s silent case is the common one:** `'UL'` is a 2026.4.0 colour and `'U'` the obvious single-store call. Both
    are accepted with the other meaning, and only a string containing `'O'` raises.
  - *The session recommends (a). The writer and the critic note that (c) removes (a)'s silent case at the cost of
    tuple syntax. The choice is Nick's.*
- **D-3. RESOLVED (Nick, 2026-11-11): no folding.** `rL`/`rR` stay as they are, frame or generic, and the only width
  change is the `nU`↔`nD` swap. A UT3 point's Tucker rank `nU` becomes the house `n` (A3 exception).
- **D-4. The five doctests that print `F.up_ranks`** (`frame_variations_format.py:338, 1840`, `manifold.py:202`,
  `uniform_frame_variations_format.py:267`, `optimizers.py:807`). All are padded frames, minimal by construction.
  *Recommended: rename only*, and detect stale names with §5.2 and the battery's slack paths. An unequal structure is
  impossible for `padded_start`, whose `x0` must be minimal.
- **D-5. The padded-frame letters.** The products and messages `D` become `Ψ` (code `Psi…`, the note's
  `\Psi_{a\to b}`, the internals page), and the edge frame `U_{x→t}`, `U′` becomes `F_{x→t}`, `F′`. *Recommended.*
  `Ψ` is free in the code, the note and the page (the dense oracles' lowercase `psi`, the right interfaces, is unrelated;
  their letter note says so). `M` is the note's proof matrix (:209); `E` is the captured energy (internals :53, note).
- **D-6. RESOLVED (Nick): the corewise frame is `(D, G, G, G)`.** *Recommended for the residual case* (§3.3):
  - Only the enumerated frame, tangent, sampling and curvature sites take `D`. That covers the T4S-derived `U`-family
    code names in tier C's row (`dU`→`dD`, `Udot`→`Ddot`, …, with the `G` family unchanged).
  - T3-level code and the T3 pages keep their letters: `U` in T3-level code, `B` where the docs write it.
  - The alternative is to rename those too (to `B` or `D`), which Nick would rule on site by site.
- **D-7. RESOLVED (Nick): `P`/`Q` → `L`/`R` at the sites where they name frame cores**, i.e. those §3.4 classifies
  so, plus tier C's and tier D's listed rows and §4A's blocks. This uses per-file NAME maps and exact
  text patterns. `P` keeps its other meanings (the completion `P⁺`, the P-block, `P_U`, the direction `P^t`), `Q` its
  QR meaning, and `R` the RNG helpers.
- **D-8. RESOLVED (Nick, final, 2026-11-11): one width letter set**, applied exactly by A3's swap, its scoped swaps and
  §4A's blocks.
  - `nU`, `nD`, `rL`, `rR` in code identifiers, shape comments, keyword names, properties, diagrams and equations;
    `n_{D,i}`-style subscripts in LaTeX. The postfix letter names the store: `nD` is the down store's width (today's
    `nU`), `nU` the up store's (today's `nD`/`nO`). `nO` disappears.
  - No `n_up`/`n_down`/`r_left`/`r_right` anywhere (K2).
  - So the width keywords and properties are same-spelling silent swaps (§2), listed in the release notes' first
    paragraph.
  - Proof of the swap: the battery's slack frames and distinct padded widths, the bijection check, K13's shape-comment
    agreement, and the reviewer's adjacency reading. The Tucker factor's row count is `nDi`; a TT-shaped up core's
    middle axis is `nUi`; the left/right cores' and the TT variations' middle axis is `nDi`; the Tucker variations'
    rows are `nUi`.
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

- **Over-reach (Nick's chief worry).** An agent "completes the pattern" at a site the plan does not list: renaming an
  SVD's `U`, a Greek stem, an RNG `R`, a `_shift_*` function or an adjective. Mitigated by:
  - §0's exhaustive list and §1.5's DO-NOT-TOUCH list;
  - K14's byte-identity check;
  - the implementer brief (§7);
  - the reviewer's trace rule: every changed token maps to a token-map row, a classified hit or an inventoried block.
- **A sequential replace corrupting double-touched tokens.** Prevented by the sentinels (K0), the bijection check and
  the battery.
- **A silent swap passing on equal-store fixtures.** Prevented by §5.2 and the battery's slack fixtures.
- **Collisions the rename itself creates.** Found so far: `E` (D-5); `dc`, the `dataclasses` alias (hence
  `c_down`/`c_up`); `D`, the oracles' dense frame (renamed `dense`); `R`, `test_jet_recurrence`'s random lambda
  (kept; the frame locals in its function take a `c` suffix); `test_uniform_curvature.py`'s `D` family (no shared
  function with its `U`, so nothing renamed); `Ud`, "u dense" in `o1_ragged.py` (not
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
