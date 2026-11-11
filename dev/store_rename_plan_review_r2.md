# The store rename plan — critic review, round 2

Reviewed: draft 2 (`dev/store_rename_plan_2026-11-10.md`, 607 lines, 00:34), the changed parts, against Nick's rulings of
2026-11-11, round 1 and the code at `79b2bd6e`. Measurements in the `t3toolbox` env.

**Resolved from r1.** M1 (free-prose rows in tier E with per-file counts; K9 added — one gap, MINOR 1), M2 (tier C
derived from every CL row — one scoping gap, MINOR 3), M3 (verified: `xO`'s old stores `(4,3,3,2)/(2,3,3,2)`, its
target `(5,5,4,3)/(3,5,4,3)`, and `fv_padded_frame` accepts it; `T3Frame.padded(x_old, (4,5,4), (1,2,3,1))` gives
`(4,5,4)/(2,5,3)`), all eleven minors and the nits. **Checked and sound:** no tier-C image pre-exists in any mapped
file (`probing`, `sampling_derivatives`, `apply`, both twins, `test_jet_recurrence`, `tv_`/`utv_operations`, `sharing`,
`manifold`, `utv_sampling`, `padded_frames`; tokenizer scan), and no `L`, `R` or `D` local exists in `probing` or
`sampling_derivatives`; no public signature takes a letter-named parameter (only `SharedFrameData.svd_U`, an SVD factor);
`Λᵀ` is the exact adjoint of `Λ` with truncation (the §3.2 identity is right); §3.3's split — `D` for the down slot, `B`
for a T3 as stored — is a clean rule; D-10 is acceptable either way, and the single pass-1 diff for R2 is a fair reason.

## MAJOR

**N1. `Λ` is not free.** `\Lambda` is the Lagrangian of the exact-Hessian note — `\Lambda(\th) = f(\Phi(\th)) -
\ip{\grad f(X)}{\Phi(\th)}`, `\nabla_\th\Lambda`, `\nabla^2_\th\Lambda` — at `exact_hessian_two_hole.tex:255-275` and
`:1019` (7 uses), the note that defines the shifts (`:573-583`); `curvature_internals.md:23` names the same Lagrangian.
§3.2 and D-1 recommend `Λ` on "zero uses in the code, the docs, the notes and the glossary". Either rename the
Lagrangian (not `\mathcal L`: that is `\Lift`, `:38`) or pick another shift symbol, re-checked the same way, and correct
the claim before it goes to Nick.

**N2. `P`/`Q` → `L`/`R` and the `U` family have no inventory, and tier E applies pre-ruling texts.** The inventories
classified `U`/`O`/`D` only and kept `P`/`Q` by design (inv 2 M4: `(U, O, P, Q)` → `(D, U, P, Q)`), and their C1 rows
keep `Udot`, `dU`, `(U, G, G, G)`. Draft 2's tier E applies "the full text the inventory gives", so R4 writes
`(D, U, P, Q, shape, …)` at `geometry.py:602, 707, 893` (inv 2 §4) and "frame's ``D`` and ``P``" at
`ut3_sampling.py:374`, and K4 (`\(U, ?O, ?[PL], ?[QR]\)`) cannot see the hybrid. Frame `P`/`Q` on lines without a
`U`/`O` are in no row at all: `geometry.py:122, 139` (`P_last`, the last left core), `uniform_manifold.py:479`
(`(P_i^L)^T dG_i^L = 0`, eqs (47)-(48)), `_curvature_oracles.py`'s `P`, `Pd`, `Q`, `Qd` and its docstrings
`Q = O = P = G`, and in the notes `exact_hessian_two_hole.tex:117-132, 305, 343-359` (beside `H_i = P_i ×_3 T_i`)
and `symmetric_probe_derivatives.tex` (where `P^t` is the direction and stays). The same holds for the `U` family: the
row's file list misses `backend/fitting.py:351, 374, 380, 389` (`(dU, dG)`, `theta_hat = (Udot, Gdot)`),
`ut3_sampling.py:114, 231`, `entries.py:152, 290`, and the T3-class doctests `tucker_tensor_train.py:3870-3871,
4072-4073`, `uniform_tucker_tensor_train.py:566-567, 631-632` (`gU, gG = x.apply_corewise_transpose(…)`: by §3.3 T3
pages, so `gB`?). Fix: (a) R1 builds a `P`/`Q`/`U`-family inventory (NAME tokens and text), each hit classified —
frame core, derivative direction (`P`, `pp`, `P^t`), completion/P-block, QR, pseudo-inverse (`test_curvature.py:215-220,
353` is `P = pinv`), RNG, T3-as-stored `B` — with its disposition; (b) a rule that every tier-E text is applied with
the rulings layered on (frame `P`/`Q` → `L`/`R`, the `U` family → `D` or `B`, `(U,G,G,G)` → `(D,G,G,G)`, the shift and
`d` notation), plus the order of tier D and tier E on a shared line (a tier-E row asserts its current text, so tier D
first breaks it; tier E first leaves text tier D does not recognise); (c) K4 gains `\((?:D|U), ?(?:U|O), ?P, ?Q\b`,
and K8 the frame forms `\b[PQ]_(?:i|last|\{)` and the code names `\b(?:Udot\w*|dU\w*|gU|Ug)\b` (carve-out `o1_ragged.py`'s
`Ud`), each with the inventory's keep list.

## MINOR

1. **K9 skips the lines most likely to be wrong.** A bare word on a line "changed since `v2026.4.0`" is left to the
   reviewer, but most prose lines with a store word also carry an identifier or width that tier A rewrites, so they
   change anyway (e.g. `fv_operations.py:640`, `nD` beside bare `down`/`up`). Make K9 token-level: every bare
   `up`/`down` in the final tree is a keep row, or a word whose aligned old-tree token was the opposite store word
   (`git diff` line alignment, then token position), or part of a tier-E proposed text; otherwise it fails.
2. **The sentinel encoding fails for text.** `'ZQX%03d_%s_XQZ' % (k, old)` embeds the old text. For text patterns
   (`(up, down, left, right)`, `D^r`, `DΦ`, `U̇`) the sentinel still contains old words, so a later pass-1 regex matches
   inside it (`\(up, down` inside `ZQX040_(up, down, left, right)_XQZ`), and `resolve`'s `(\w+?)` cannot match non-word
   text, so such sentinels never resolve. That is exactly where §1.3 says sentinels are required. Use `ZQX%04d_XQZ`
   (ENTRIES keeps the old text; `resolve` checks the count per key), and `\d+` (`%03d` caps ENTRIES at 1,000).
3. **The catch-all row is line-scoped.** A NAME map on "each row's lines" breaks every def/use pair whose other line is
   not an inventory row, and P/Q lines never were (N2). Scope by enclosing function, as B3 does, with the collision
   check per function. That also removes the prescribed `D` → `th` rename in `test_uniform_curvature.py`, which
   collides: `th` and `th_r` already exist in the same functions as `D`/`D_r` (`:977-978`, `:1719-1726`), while the
   `U` there (`:99-104, 249-250`) never shares a function with the `D` family. And `Dr`, `Du`, `Dth`, `Dv`
   (`:1237-1240, 1726-1728`) are dense tensors, not raw directions.
4. **The dense-oracle row misses `Uo`/`Uc`.** `_padded_frame_oracles.py:87-98` would read
   `D[i] = np.concatenate([Uo[i0], W], axis=1)`. Add `Uo→Do, Uc→Dc` (free; `Uy` is an SVD factor). Its letter API
   (`dense_pass(update='ULRO')`, `'U' in update`, `distances`' `zip('ULRO', …)` keys) belongs in R3 with the library's,
   or the oracle tests word tuples with letters (`'U' in ('down', 'left')` is False).
5. **The checks misfire.** K8b flags correct output: after B4 `p['U']` is the new up-core key; use `[OPQ]`. K8's `U̇`
   never matches before a space or punctuation (`\b` after a combining mark; measured on `'the U̇ term'`); drop its
   trailing `\b`. K7 misses today's unparenthesized forms: `D x =` (`sampling_derivatives.py:300`,
   `deferred_and_rejected.md:134`), `U x =` (`:303`), `# D jet,` (`:318`), `# U jet,` (`:330`), `[jet, D jet]`
   (`:342, 352`); use `\b[DU] (?:x|z|jet)\b`.
6. **Letter strings outside B4.** `test_tucker_tensor_train.py:2236, 2255, 2283, 2301, 2370, 2391` (`kind == 'U'`,
   `replace(x, 'U', …)`, `('U', dU)`) must flip together: a call site at `'B'` with the comparison still `'U'` perturbs
   the TT core instead. Give them a scoped STRING map like B4.

## NIT

- B7's `fv_aligned_padded_frame(x_old, tk1, tt1, y, …)` does not match `(old_frame, tucker_ranks, tt_ranks, hint,
  project, …)`.
- The dense oracle's `E` (the core → leaf basis, now the up edge) keeps a letter §3.1 reserves for the captured energy;
  under one alphabet, `U`.

VERDICT: REVISE (2 MAJOR, 6 MINOR)
