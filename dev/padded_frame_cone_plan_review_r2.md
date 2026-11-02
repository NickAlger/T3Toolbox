# Padded frame / cone reference plan — critic review, round 2

**Plan:** `dev/padded_frame_cone_impl_plan_2026-11-02.md` (round 2, 1705 lines). **Round 1:**
`dev/padded_frame_cone_plan_review_r1.md`. **New probes** (read-only, `t3toolbox` env, session scratchpad):
`probe_layout.py` (ragged slice vs `ut3_to_t3` old frames) and `probe_gappy.py` (a gappy-mask `x0`); the round-1
probes (`probe_sup_vs_cone.py`, `probe_oracle.py`, `probe_gauge.py`) still apply.

## Overall assessment

Round 2 fixes all four round-1 MAJORs properly:

- The cone reference is now defined as the projection onto the tangent cone, and §2.2's matrix analysis is exact.
- Default runs keep their `stats` and history unchanged.
- The cross-layer claim now matches what the code does.
- T3(h) turns the per-step dense oracle into a test.

Every MINOR and NIT is addressed, mostly in the way suggested. On the coordinator's four questions:

- The `K` lift is correct and its cost is stated honestly.
- Building the uniform old frame through the ragged path gives bitwise-equal frames in practice. It runs eagerly with host-side masks only.
- The stats keys satisfy decision 2.
- The function-level import works, but the precedents it cites are wrong; the house's documented answer to a cycle is placement.

One new correctness gap comes from the M3 change. The uniform old-rank slice reads prefix masks, so it gathers the
wrong slots for a gappy `x0` (m32). One structural check fixes it. The import question (m33) I accept as a maintainer judgment call. The rest are
wording NITs, mostly about how precisely the cone value is described for T3.

## Round-1 findings — status

| r1 | status | where verified / residual |
|---|---|---|
| M1 | resolved | §2.2 :258-298: the three matrix statements (aligned completion gives `P_{TC}G` exactly; the supremum is `‖N_{2k}‖` at a staggered completion; the staggered completion goes to the aligned one in one pass) re-derived and consistent with the round-1 probe (5.006 vs 6.104 at `r = 1`). §2.7 :405-434. T8(a) asserts `‖G_r‖`, not `‖G_{2r}‖`; T8(h) tests the staggered start. Every remaining "sup" is a negation (:125, :282, :423, :1377, :1397, :1502). Residual wording: n37, n38. |
| M2 | resolved | No `NewtonInfo` field. `stats` keys only on runs that use them (:668-671, §5.5). The pins at `test_optimizers.py:477`, `:860` stay. G-bit (:1241-1243) compares whole `stats` dicts, every history field bitwise but `wall_time`. Decision 2 holds. |
| M3 | resolved (option b) | §3.3 :573-583, §3.4 :595. T4(e) bitwise; T5(f), T8(c), T9(e) gauge-invariant. `probe_layout.py`: the ragged slice and `ut3_to_t3` under the old-rank masks give identical blocks and bitwise-equal old frames on 12/12 seeds. New edge case: m32. |
| M4 | resolved | T3(h) (:1225) states the oracle exactly as computed in round 1: Tucker with `E_d`; `L`/`R` two-sided and pinned; `O` on the new slots. At `d = 3, 4`, `C = ()` and `(2,)`, shared and unshared. |
| m5 | resolved in substance | Function-level import plus G-acyclic. The cited precedents are wrong (n34); the placement alternative is m33. |
| m6 | resolved | Ratios come from the dense column; label nuance in n38. |
| m7 | resolved | §0 :125-129, §11, O-9. |
| m8 | resolved | R1, I-27; the loop sketch passes `padded_from` only when `stop_reason == 'gradient'` (:1309); release notes. |
| m9 | resolved | §3.9 :747-752, §4.2 step 4 (:819-821), T3(g), R7. See focus 2. |
| m10 | resolved | §3.6 :651-655, :696-698; `numerical_contracts.md` rows; T9(d). |
| m11 | resolved | §3.2 :543 (old block from `U°_{min(g)}`), §3.6 :699-700, T9(d). |
| m12 | resolved | §2.3 :301-308 (now a caveat), §3.2 :543-545, I-30. Carry-over phrasings: n35, n36. |
| m13 | resolved | §3.4 :604-608, I-29, T5(h). |
| m14 | resolved | T10. |
| m15 | resolved | `PaddedStart.frame` (:1132), T5(c), T9(c), I-19. |
| m16 | resolved | T5(b), P13. |
| n17–n31 | resolved | n17: no `dev/` reference in the CHANGELOG draft. n18 :1043. n19: R2. n20 :673-677. n21: T7(e). n22 :792-793. n23 :511, :635, :969. n24 :685-695. n25: T7(g), T3(i). n26 :1176. n27 :125, :421. n28: I-33. n29: I-31. n30: R5, §4.1. n31 :1146-1149. |

## The four focus items

**1. The function-level import, or moving the primitive.**

Moving `tv_to_t3` below `fv_operations` is not possible as things stand. It calls
`sharing.fv_tied_ambient_directions` (`tv_operations.py:244-245`), and `sharing` imports `fv_operations` at module
level (`sharing.py:34`). So the real choice is between the function-level import and moving the *constructor*.

- **The import is safe at runtime.** When `fv_padded_frame` runs, both modules are already loaded. The module-level graph stays acyclic, and G-acyclic will confirm that.
- **The stated reason is wrong.**
  - `_bfit_pd` is a lazy import ("only needed to resolve chunk_size='auto'", `t3toolbox/optimizers.py:180`).
  - `_jitted`'s `import jax` keeps jax optional (`backend/optimizers.py:553-555`).
  - Neither one breaks a cycle.
- **It would be the first of its kind.** The backend has no function-level import of another library module today.
- **The house's recorded answer to a would-be cycle is placement.** `utv_operations.py:769-776`: "Named for its operand, placed for its imports", and a module may host several families.
- **The gate cannot see this edge.** G-acyclic scans module-level imports only. "Acyclicity checked" is true of the module graph, not of the layering.

See m33 and n34.

**2. The `K` lift: correctness and cost.**

- **Correct.** Element `(k, c)` of the output is the atom on `(template_c, old_c, hint_frame_c, v_{k,c})`.
  - A ragged stack has one structure, so every count and index set is shared across `K + C`.
  - The host draws are keyed by `[seed, k_g]` and `[seed, i, family]`, never by a stack index, and they broadcast.
  - So the lifted output equals per-hint calls up to batched rounding, which is what T3(g) asserts.
  - `tv_to_t3` already broadcasts a `C` frame to `K + C` (`tv_operations.py:258-262`), so broadcasting the hint frame first is harmless.
- **Cost: |K| full constructions.** Each hint pays its own orthogonalization of `Z` at doubled ranks, the three zippers, both sweeps and the down SVDs. Only the cheap pin embedding is repeated without need.
  - Eager jax materializes |K| copies of the template and old frame. Each is one frame in size, small next to the per-hint work.
  - The uniform twin loops over `K + C` on the host.

R7 says all this, and no library consumer passes `K`. Fine as written.

**3. The uniform old frame through the ragged path.**

- **Bitwise in practice.** `probe_layout.py` (resized points packed by `from_t3`, 12 seeds) gives identical gathered blocks and bitwise-equal `t3_orthogonal_representations` frames.
- **Mask-clean.**
  - `ut3_make_masks` returns host numpy masks.
  - `ut3_to_t3` requires concrete masks (`ut3_conversions.py:145`) and gathers by boolean index, eagerly, on numpy or jax supercores.
  - The repack at the target's padded sizes and `ufv_stack` keep the masks on the host.
- **Never inside a trace.** `padded_structures` runs eagerly in the driver, by design (R4).
  - The jitted CG sees only the finished start frame.
  - That frame's masks are value-equal to the template's, so there is no recompile (T7(e)).
- **No cycle.** The new `geometry` edges are to `ranks`, `t3_constructors`, `ut3_constructors` and `ut3_conversions`; none of these imports `geometry`.
- **The gap** is the prefix slice itself: m32.

**4. The stats keys.**

- They satisfy decision 2: a default run has exactly today's keys (T7(a)), the history is untouched, and G-bit compares whole dicts.
- `padded_start_gnorms` holds floats when unstacked (§3.9), like the house's float-valued records.
- Only a naming remark remains (n39).

## New findings

### m32 (MINOR) — The uniform old-rank slice assumes prefix masks; a gappy `x0` makes it gather the wrong slots

**Where.**

- §3.4 :595: `ut3_to_t3` "under the old-rank masks (`ut3_make_masks(old ranks, n, r)`)".
- §3.4 :610-613: "the slice … is structural".
- §3.6 :693-695: the safe-mode zero-padding check.

**What.**

- **Gappy masks are legal.** Uniform rank masks can be gappy after `+` or `×` (`docs/uniform_masks_vs_ranks.md` :50-72). The default `newton_cg` path handles them: `geometry.frame` uses the geometry's held masks and pad-safe SVDs.
- **The padded path assumes prefix masks.** It gathers through *prefix* masks at the old ranks, so a gappy `x0` makes it read padding or the wrong block, and nothing checks for it.
- **Probe** (`probe_gappy.py`). Take `x0 = uz + ua`: `uz` is a rank-1 zero tensor, `ua` a rank-1 tensor, both built with `from_t3(…, n=2, r=2)`.
  - `x0` has masks `[1 0 1 0]`, and its ranks are minimal.
  - With old ranks `(1, …)`, the prefix gather `[1 0 0 0]` returns the zero block: ‖X_old‖ = 0 against ‖x0‖ = 3.23.
- **Safe mode catches only part of it.**
  - The zero-padding check rejects this example, but with a misleading message: `x0` is a zero-padding of a rank-1 tensor, just not in prefix layout.
  - The check reads only real slots *outside* the prefix. Padding *inside* the prefix (slot 1 when the old ranks are 2) is gathered as old content unchecked.
- **Under `unsafe()` the result is silently wrong.** The old frame here is the zero tensor's, so iteration 0's model and retraction use a frame of a different tensor.

**Why.** Mask layout is a structural property, read from host masks, and structural conditions are always checked
(house rule; §3.8). §3.4's "the slice is structural" holds only for prefix layouts.

**Change.**

- In the uniform `padded_structures`, per element, require the old-rank prefix masks to lie inside `x0`'s rank masks: `np.all(old_mask <= x_mask)` per store.
- Otherwise raise a structural `ValueError` that names the remedy: canonicalize with `ut3svd`, or build `x0` by `from_t3` of a `resize`d point.
- Add a T5(g) row with the probe's `uz + ua` case.
- Add one sentence to §3.4 and to the `numerical_contracts.md` row.

The realistic path never triggers the check: there is no uniform `resize` (`rank_continuation.md` :200), and
`from_t3` produces prefix masks.

### m33 (MINOR — accepted as a maintainer judgment call) — function-level import or placement

**Where.** §3.2 :551-557; §4.4 :855-856; I-32 (:1513); G-acyclic (:1243).

**What.** See focus 1. Both options are sound.

| option | for | against |
|---|---|---|
| Keep the function-level import | Smallest change; `fv_padded_frame` stays beside `fv_zero_point_frame` | Adds the backend's first upward function-level edge, `fv_operations → tv_operations` |
| Move the constructor, e.g. a `backend/padded_frames.py` hosting both twins | The house precedent (`utv_operations.py:769-776`); imports go downward; the twin pair sits in one file | Imports `_haar_rows` / `_identity_prefix` across modules; the docs' "in `fv_operations`" needs editing |

**Change.** Pick one; either is acceptable to me, and this finding does not decide the verdict. If the import stays, record it as the one deliberate exception:

- a comment at the import;
- a line on the internals page;
- G-acyclic worded as "the module-level graph is acyclic; one function-level edge, `fv_operations._hint_coordinates → tv_operations.tv_to_t3`, is recorded".

### NITs

- **n34** — §3.2 :554-555 calls `_bfit_pd` and `_jitted`'s `import jax` "the house precedent for breaking a cycle".
  They are a lazy import and an optional-dependency import. Describe them as such, or cite
  `utv_operations.py:769-776` for the cycle question.
- **n35** — Two places still say "a frame of `X_old`": §4.4 :871 ("Because the hint frame is a frame of `X_old`,
  `Z`'s chains contain…") and the `hint_frame` comment in §5.4 (:1085). Use §2.3's condition: "a frame containing the
  old frame's subspaces (§2.3)".
- **n36** — §2.3 :303-305: minimal old ranks are necessary, not sufficient, for "any frame of `X_old`" to contain
  the old subspaces. `X_old` must also have full rank at those ranks, which is generic but not structural. One clause is enough, since this is a caveat.
- **n37** — §2.7 :417-421: "contains … rank-`2r` directions, **so** the number sits at or slightly above the value
  at the dense quasi-optimal frame". The "so" does not follow: the containment compares `T_{F*}` with the cone, not with another
  frame's tangent space. The measured range also starts at 0.99. What holds:
  - `‖P_{T_{F*}}G‖` is at least the norm of `G`'s projection onto `F*`'s own rank-`r` tensor space (a HOOI-like quasi-optimal approximation);
  - it can exceed the true cone value through the rank-`2r` directions;
  - it measured 0.99–1.04× the value at the T3-SVD frame.

  The tex paragraph (:1395-1397, "at or slightly above the quasi-optimal value") needs the same fix.
- **n38** — User-facing wording for T3:
  - The fitting §5 row (:1271-1273), the CHANGELOG (:1433-1435) and the release notes say the reference is
    "computed as" the fixed-point frame gradient. For T3 that value is an estimate (§2.7, R6). Suggested wording: "exact for matrices; for T3, the frame-gradient norm at the iteration's fixed point, within a few percent of the quasi-optimal value".
  - The 15–76× figures (:1280, :1406) are `‖T3SVD_r G‖` / Haar. That is a lower bound on cone / Haar, so write
    "at least 15–76×", or name the quantity.
- **n39** — `stats['start_frame']` holds a label (`'given'` / `'padded'`), not a frame. Either rename it, e.g.
  `stats['start_frame_source']`, or add a doc line saying it is a label. A matter of taste.

## On the writer's round-2 "(weigh)" items

- **I-28** (the uniform old frame through the ragged path): agree. It is the simplest way to make "same frame up to rounding"
  true, and it is verified bitwise on prefix inputs (focus 3). It needs m32.
- **I-30** (structurally minimal old ranks): agree. It is the right structural stand-in for "no LAPACK slack in the old block"; the
  numerical-rank remainder is a caveat (n36).
- **I-13** (passes and oversampling): unchanged since round 1; provisional until S11.

## Checked and found correct (round 2, in addition to round 1's list)

- **§2.2's matrix algebra.**
  - The tangent cone at a rank-`s` point.
  - `P_{TC}G = P_{T_X M_s}G + N_k`.
  - The frame projection's normal part `N − P^⊥ N P^⊥` has rank ≤ `2k`.
  - The supremum is `‖N_{2k}‖`, attained at the staggered completion.
  - The staggered completion is not a fixed point: its frame gradient's normal part is `N_{2k}`, whose top-`k` directions are the aligned ones.
- **§2.7.** "What it is": at its apex, the cone is the rank-`r` set itself. "What it is not": neither the supremum nor a seed average, and the Haar value is no bound on it.
- **The `K` lift (focus 2).** The in-block key `[seed, i, c]` uses `c` for the core family, not a stack index.
- **The uniform old frame (focus 3).** Built through the ragged path; bitwise on identical prefix inputs (`probe_layout.py`); host masks; eager; the new geometry edges are acyclic.
- **The stats keys and G-bit (focus 4).** History rows are unchanged.
- **I-29, the oversampled maximum rule.** `old ≤ target ≤ oversampled` holds in all four stores, and `p = 0` gives the target exactly.
- **I-30, the minimal-old-ranks check.** Structural, and well defined with sharing (`compute_minimal_ranks(..., sharing)`).
- **`tv_to_t3` without `shared_data` (§4.4).** It represents the same tensor as the tied embedding ("an exact rewrite", `tv_operations.py:230-236`). The construction reads `Z` only as a tensor, so the shared path needs no companion.
- **The frontend checks.** Structural: non-minimal `x0`, wrong representation, corewise. Safe-mode numerical: ORTH, TIED, zero padding. Each is on the right side of the house's structural/numerical split.
- **T7(g)'s zero-count case.** The final construction has no new slots, so the frame is the old frame, which is the library frame, bitwise on the ragged layer; the oversampled passes do not reach it.
- **NIT follow-ups.** The `aux=` threading, `_prepare_jit_frame` via `partition_static` / `rebuild_static`, the float sign literals, the doctest margin, and version 2026.4.0.
- **§7.** Every "O(1)" claim has its replacement text, and no shipped text cites `dev/`; O-2's T3Polynomial path appears in the plan only.
- **§8 slicing.** Review stops at S3, S7 and S10; G-acyclic sits in S3 and G-bit in S7.

## For round 3

Only m32 needs a change: one structural check, one test row and two sentences. For m33, record the choice. The NITs are
optional. I expect to approve on those alone.

VERDICT: REVISE
