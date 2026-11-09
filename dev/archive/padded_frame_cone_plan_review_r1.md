# Critic review, round 1 — `dev/padded_frame_cone_impl_plan_2026-11-02.md`

Reviewer: critic agent (Opus 5.5), 2026-11-02. Read in full: CLAUDE.md, every `docs/*.md` / `*.rst`, `docs/pad_safe_svd.tex`,
every `docs/contributor/*.md`, the two archived plans, `dev/HANDOFF.md`, CHANGELOG `[2026.3.0]`, the downstream record
`T3Polynomial/dev/zero_reference_cone_2026_11_02.md` and `t3polynomial/cone.py`, and the code the plan touches.

## Overall assessment

The algorithm is right. I ran the Appendix B prototype against an independent dense oracle for every step (the
two-sided Tucker step, the pinned left sweep, the mirrored right sweep, the down step) and it matches to ≤ 4.4e-11 on
three structures (d = 3, 4); the frame invariants hold (orthonormality ≤ 1.8e-15, both base-point reconstructions equal
`X_old` to ≤ 1.8e-15, the old Tucker block bitwise). The structural design (counts from both rank sets, Householder
completion, no threshold, no content in the point, retraction untouched) is sound, and the seam through
`Problem.local_model(frame=)` reaches every frame-dependent consumer.

What needs revision is around the algorithm: the cone reference is defined as a supremum over frames, which is a
different (larger) number from the one the iteration computes and the tests measure; the default runs gain a history
key, against binding decision 2; the cross-layer "identical real blocks" claims are false at a padded restart and one
test as written cannot pass; and nothing in the test plan can see a wrong subspace in the TT sweeps or the down step.
Four MAJOR findings, twelve MINOR, fifteen NIT. No BLOCKER.

Probes run (read-only, `t3toolbox` env, scripts in the critic's scratchpad — each is described in the finding that uses
it so it can be re-run): (P-a) the matrix supremum vs the cone value; (P-b) Appendix B vs a dense oracle per step;
(P-c) ragged vs uniform frame real blocks at generic and padded points.

---

## Findings

### M1 (MAJOR) — The cone reference is defined as `sup_F ‖P_{T_F} G‖`, which is not the tangent-cone projection

**Where.** §0 :102-103; §2.2 :247-253 ("It is maximized by `U_new, V_new` = the top-`(r − s)` singular subspaces of
`N`"); §2.7 :366-374; §7 :1133-1134 (the §5 table row: "the frame-free value the 'zero' convention samples"),
:1220-1222 ("takes the supremum the seed average would only estimate"), :1238-1240 (the math-note paragraph:
`\sup_Q \|P_{T_Q} G\|`, "the matrix case equal to `\|G_r\|`"); CHANGELOG draft :1273-1275.

**What is wrong.** In §2.2's own matrix setting, with `N = P_{U_X}^⊥ G P_{V_X}^⊥` and `k = r − s`,
`‖P_{T_Q}G‖² = ‖G‖² − ‖P_{U_new}^⊥ N P_{V_new}^⊥‖²`. The minimum of the subtracted term over completions is
`Σ_{j>2k} σ_j(N)²` (Eckart–Young: `P_{U_new}^⊥ N P_{V_new}^⊥ = N − (a matrix of rank ≤ 2k)`), attained by a
*staggered* completion `U_new = u_{1..k}`, `V_new = v_{k+1..2k}`. So

- `sup_Q ‖P_{T_Q}G‖² = ‖P_{T_X}G‖² + ‖N_{2k}‖²`, while
- the cone value is `‖P_{T_X}G‖² + ‖N_k‖²`.

§2.2 :243-244 already says why: `T_Q`'s normal part reaches rank `2(r − s)`, the cone's only `r − s`. Probe P-a
(9×8 Gaussian `G`, `s = 0`): `r = 1`, the top-`r` frame captures 5.006 = `‖G_1‖`, the staggered one 6.104 = `‖G_2‖`;
`r = 2`, 6.104 vs 7.484 = `‖G_4‖`. The downstream record (§2) and `cone.py`'s docstring carry the same identity;
"the tangent cone is the union of the frame tangent spaces" is also not right (the union is larger).

What *is* true, and what P14, T5(a) and T8(a) actually measure: the top-`(r − s)` completion is the iteration's
attracting fixed point (the staggered maximizer is not a fixed point — one pass maps it to the top-`k` completion,
since the hint's normal part there is `N_{2k}`), and at it `P_{T_Q} G = P_{TC} G` exactly (§2.2's statement of that is
correct). The iteration computes the right thing; only its definition is wrong.

**Change.** Define the cone reference as the norm of the projection onto the tangent cone, computed as the frame
gradient at the iteration's fixed point: exact for matrices; in T3 the frame tangent space at that frame is slightly
larger than the cone, so the value sits at or slightly above it (P12: 1.00–1.04×; downstream e01 `r = 3`: 75.8 vs
69.8, +8.6%). Remove every "sup" / "maximized by" / "takes the supremum" (a seed average estimates
`E_F ‖P_{T_F}G‖`, a third number). Drop "the frame-free value the 'zero' convention samples": a Haar frame's value is
neither a sample of nor a bound on the cone value (P-a: random frames reached 5.59 > 5.006). Fix the math-note
paragraph to state the fixed-point characterization. (The downstream record's §2 wording deserves the same correction,
outside this plan.)

### M2 (MAJOR) — `NewtonInfo.start_frame` changes the history of every default run (binding decision 2)

**Where.** §3.6 :577-578; §5.5 :1038-1039; I-18 :1338; §7 :1120-1121, the §5 doctest :1153-1154, release notes
:1242-1243 ("the defaults run bit-identically"); CHANGELOG :1268-1269; G-bit :1106-1109; T7(a) :1094.

**What is wrong.** History rows are `_newton_scalar_record(info)` over every dataclass field
(`backend/optimizers.py:1031-1036`), so a new field recorded as `'library'` adds a key to every row of every default
run. Decision 2 says default runs stay bit-identical "including stats/history content". The plan also contradicts
itself: G-bit compares default histories bitwise between `main` and the branch, which this key makes impossible, and the
release notes' "bit-identically" becomes false. The field is also redundant: the caller knows what it passed, and the
start frame affects iteration 0 only.

**Change.** Drop the `NewtonInfo` field. If a record is wanted, put it in `stats` **only when a start frame was used**
— e.g. `stats['start_frame'] = 'given' | 'padded'`, and for `padded_from` the `padded_start` diagnostics
(`gnorms`, `gnorm`). Default `set(stats)` then stays `{'losses', 'newton_steps', 'newton_accepted', 'stop_reason',
'history'}` (pinned at `tests/backend/test_optimizers.py:477`, `:860`). Update T7(a)/(b), the §7 doctest (print the
stats key), the CHANGELOG line, and state G-bit as full-dict equality of `stats`.

### M3 (MAJOR) — "Identical on the two layers" is false at a padded restart; T5(f) as written fails

**Where.** §3.3 :490-493; §4.8 :832; §7 :1126-1129 (§4.11 text), :1166-1168 (rank_continuation), :1192-1197
(uniform_equivalence_contract); CHANGELOG :1267; T5(f) :1092.

**What is wrong.** `ufv_padded_frame` equals `fv_padded_frame` bitwise *on identical inputs*, but `padded_start` gives
the two layers different inputs. The old frame is `t3_orthogonal_representations` (LAPACK) on the ragged layer and
`ut3_orthogonal_representations` (pad-safe SVD) on the uniform one, and those differ by a sign gauge even at a generic
point. Probe P-c (d = 4, four seeds): `max |ragged − uniform|` over real blocks 2.0 to 3.5e3, while the absolute values
agree to ≤ 2.7e-12 — the gauge equivalence `uniform_equivalence_contract.md` §"Gauge-carrying operations" states. The
pinned old block therefore differs in sign between layers, the new blocks are expressed in those bases, and the hints
differ at rounding. The padded frames are gauge-*equivalent* (same subspaces, same tangent space, same first step to
rounding), not equal; T5(f)'s "the frames' real blocks allclose 1e-12" will fail. (At `s = 0` there is no old block and
the claim holds to rounding.)

**Change.** Either

- (a) restate every such claim as gauge equivalence ("the same subspaces on both layers, hence the same first step"),
  and make T5(f) gauge-invariant: per-family projectors in ambient terms (`U_iᵀU_i`, the left/right basis projectors),
  or the dense `P_{T_F} Y` of a fixed random `Y`, plus `gnorms` and the base points; or
- (b) make the strong claim true by construction: the uniform `padded_structures` builds the old frame from the
  unpacked real blocks with the ragged `t3_orthogonal_representations` (host, once per call — `ufv_padded_frame` unpacks
  it anyway), so the old block and the hint-free start are bitwise the ragged ones and only rounding in the hint
  separates the layers.

(a) is the minimum; (b) is what the docs currently promise. Either way, add that a near-tie at a cut can still split the
layers, as for any SVD truncation.

### M4 (MAJOR) — Nothing tests the TT sweeps or the down step against ground truth

**Where.** §6 T3 :1090, T5 :1092 (and T8).

**What is wrong.** The directional tests are the matrix fixed point (T3(c), T5(a), T8(a)) and the group Gram oracle
(T3(e)); both exercise the Tucker step only. At `d = 2` with Tucker rank = TT rank every TT and down core is forced
(`L_0` square orthogonal, `O` full), so a wrong index in `_pinned_left_sweep` (`Ω[k]` for `Ω[k+1]`, `Mp` in the carry,
the reversed-store bookkeeping) or in `_down_completion` still gives a valid frame of `X_old` and passes every listed
test. It would show only as a weaker G-P4 measurement. `testing_strategy.md`: correctness is checked against dense
ground truth.

**Change.** Add T3(h), a dense oracle per step, against the library function (not the prototype). Probe P-b did exactly
this on Appendix B. With `Z` the dense hint, `Φ_k` / `Ψ_k` the dense left/right bases of the constructed frame and
`Φ°`, `Ψ°` the old ones:

- Tucker `i`: the leading left singular vectors of `P_{U°_i}^⊥ Z_(i) (I − E_dᵀE_d)`, where
  `E_d[m] = Σ_ab O°_i[a,m,b] Φ°_i[a] ⊗ Ψ°_{i+1}[b]`;
- `L_k`: the leading left singular vectors of `(I − PPᵀ) M (I − Ψ°_{k+1}ᵀΨ°_{k+1})`, with
  `M = (Φ_k ⊗ U_k)ᵀ Z_{(0..k | k+1..)}` and `P` the embedded `L°_k`;
- `R_k`: the mirror image (columns projected off `Φ°_k`, pinned off the embedded `R°_k` rows);
- `O_i`: the leading bond-space directions of Z's coordinates in `Φ_i ⊗ U_i ⊗ Ψ_{i+1}` on the new Tucker slots,
  pinned off the embedded `O°_i`.

Compare projectors. The prototype matched every step to ≤ 4.4e-11 on (5,6,7), (4,5,6,5) and (6,6,6,6) with
continuation-style rank increases. Run it at `C = ()` and `(2,)`, unshared and shared, d = 3 and 4. It is tiny and it is
the only test that can see a wrong-subspace bug.

### m5 (MINOR) — `fv_padded_frame` in `fv_operations` closes an import cycle

**Where.** §1 :147; §3.1 :417-421 ("reused: `tv_to_t3`, …").

`tv_operations` imports `fv_operations` at module level (`tv_operations.py:23`), as do `sharing` (`:34`) and, through it,
`t3_svd`. A module-level `import tv_operations` in `fv_operations` makes `fv_operations ↔ tv_operations` circular. The
precedent plan kept the backend DAG acyclic and verified it (its §M, F35). **Change:** state the placement and every new
edge (a function-level import of `tv_to_t3` inside `fv_padded_frame` has house precedent — `t3toolbox/optimizers.py`'s
`_bfit_pd`, the in-function `import jax` of `_jitted`; or host the hinted helpers in a module allowed to import
`tv_operations`), list the other new edges (`fv_operations → t3_orthogonalization, tt_orthogonalization`;
`geometry → ranks, t3_constructors, ut3_constructors`), and add the acyclicity check to S3's gate.

### m6 (MINOR) — The docs misattribute the downstream ratios

**Where.** §7 :1140-1142 ("10–25× smaller than the projection onto the tangent cone … 13×, 24×, 14×, 10×, 7×, 7×"),
:1171 ("up to ~25× below the cone value"); CHANGELOG :1285-1286 ("7–25× below the tangent-cone value").

Those numbers are the §0 table's "old / Haar" column: the coordinate-aligned (LAPACK) reference over the Haar one. The
same table gives dense cone / Haar = 14.7/0.207, 47.3/0.622, 55.9/1.368, 60.7/2.329, 64.4/3.716, 66.1/4.327 ≈ 71, 76,
41, 26, 17, 15. **Change:** quote cone/Haar as 15–76× (Darcy o01), or say "7–24× below a coordinate-aligned completion's
value and 15–76× below the dense cone value", in all three places.

### m7 (MINOR) — "`gtol_rel = 3e-4` ≈ the pre-2026.3.0 `1e-3`" compares against the wrong reference

**Where.** §0 :104-105; §3.7 :609-610; §7 :1142-1144.

The downstream ladder (record §4) compared the cone at 3e-4 with **T3Polynomial's own** coordinate-aligned zero
reference (`_zero_gradient_norm`) at 1e-3. The library's pre-2026.3.0 default was the **warm start** `‖g(x0)‖`
(CHANGELOG `[2026.3.0]`), which is a different stop. **Change:** state the equivalence as measured ("where a
coordinate-aligned zero reference was used at 1e-3"), or leave the number out of the user docs until S11 measures it
against the library's own references; keep 3e-4 as the downstream ladder setting in §11.

### m8 (MINOR) — The unconverged-level result is softened, and the release notes recommend `padded_from` everywhere

**Where.** §7 :1152 ("it is neutral"); release notes :1243-1245; against P11 :72 and R1 :1357.

P11: `f1/f0` is worse on all six unconverged cases and the Newton totals move +3, 0, +4, −1, −1, −1. Downstream ladders
often end levels at the cap (record §4: 1.7–4.3 of 6 levels converged), so "pass `padded_from` at each warm-started
level" is not yet supported. **Change:** say "neutral to worse (up to 4 more Newton steps measured)", make the advice
conditional on a converged previous level, and settle the general recommendation after S11's ladders.

### m9 (MINOR) — The `K`-stack rejection rests on a false premise

**Where.** §3.9 :636-638; I-5 :1325; R7 :1363.

"A stack of tangents at one frame has no canonical reduction to one hint" is true, but the library's canonical lift of
an extra stack is a map, not a reduction (`batching_and_stacking.md` §11: "everything batched is that atom lifted over
the stacks"). Here that is a `K + C`-stacked output frame, one padded frame per hint: broadcast the template and the old
frame to `K + C` and run the `C`-vectorized code with `C := K + C` (`tv_to_t3` already materializes the `K + C`
broadcast). Decision 5 asks for stacks "as far as reasonably possible". **Change:** support `K` that way (a few lines and
a T3(f) row), or keep the rejection with its real reason (scope, no consumer) and name the lift in the error message
instead of `sum_tangents`.

### m10 (MINOR) — A caller-supplied frame needs ORTH, not only the same tensor

**Where.** §3.6 :587-591; §3.8 :623; §7 :1172-1176.

`numerical_contracts.md` §`fitting.LocalModel`: "the frame's orthogonality is a construction invariant
(`geometry.frame(x)` produces it), not a per-op check". With `local_model(frame=)` / `start_frame=` the caller supplies
the frame, so ORTH becomes a precondition (project, retract and inner are wrong on a non-orthonormal frame;
`T3Frame.validate` checks shapes only). **Change:** add ORTH (the cached `orthogonality_residual`) to the frontend
safe-mode checks for `start_frame`, to the `numerical_contracts.md` row, and to `Problem.local_model(frame=)`'s
documented preconditions beside "same tensor".

### m11 (MINOR) — Shared groups: the old block must be tied

**Where.** §4.5 :763-768; §2.9 :399-408.

The group's factor is `concatenate([U°_g, new])`, "replicated over members". If the members' old factors differ (an
untied `x0`), every member but the one read gets a different old block and the frame no longer represents `X_old`.
TIED is the numerical precondition of every grouped entry point (`numerical_contracts.md` §"The shared surface").
**Change:** say which member's `U°` is read, require TIED (frontend safe mode, shared geometry with `padded_from`), and
add the row to §3.8.

### m12 (MINOR) — The hint-frame condition of `fv_padded_frame` is mis-stated and mis-classified

**Where.** §2.3 :265-268; §3.2 :471-473; §3.8 :623.

- *Mis-stated.* "A frame of `X_old`" is not enough for `A_i`, `Ω^L`, `Ω^R` to have orthonormal rows. That needs
  `span(U°_i) ⊆ span(U'_i)` and the old interfaces inside the hint's chains. It holds for every frame of `X_old` only
  when the old frame has no slack: a Tucker-slack old frame carries LAPACK completion rows that another frame of `X_old`
  need not contain. (The driver is safe: its hint frames are padded frames of the same old frame.)
- *Mis-classified.* The output is an orthonormal frame of `X_old` for **any** hint — the two-sided removals are
  contractions (overlaps of orthonormal sets) and the pins are untouched. By the house rule the condition is therefore a
  caveat (it decides which directions), not a precondition.
- §3.8 lists a frontend safe-mode check for it, but no frontend entry point takes a hint.

**Change:** state the condition as "a frame of `X_old` containing the old frame's subspaces (any frame of `X_old` when
the old frame is minimal; any padded frame of the same old frame)", classify it as a caveat, and drop the §3.8 row.
Consider requiring minimal old ranks structurally in `padded_structures`, since an old frame's slack directions are
LAPACK completions that would be pinned as "old".

### m13 (MINOR) — The oversampled structure is not always ≥ the target

**Where.** §3.4 :510-513; §3.5 :538-539; §4.9 :840.

`compute_minimal_ranks(shape, r + p, ρ + p)` at `p = 0` is `minimal(r, ρ)`, which equals `x0`'s ranks only for a
minimal `x0`. For a non-minimal backend `x0` (Tucker slack, e.g. `(6,6,6)`, `(4,4,4)`, `(1,2,2,1)`: up stores
`(4,4,4)` vs minimal `(2,4,2)`) the oversampled stores fall below the target's, and can fall below the old frame's
(a structural error from the constructor). "`oversample = 0` makes the two structures equal" is then false.
**Change:** take the oversampled ranks as the elementwise max of `x0`'s ranks and `minimal(r + p, ρ + p)`, or require a
minimal `x0` in `padded_structures` (structural), so `old ≤ target ≤ oversampled` always and `p = 0` is the target.

### m14 (MINOR) — `ufv_padded_frame` has no dispatch row

**Where.** T10 :1097.

The precedent `ufv_zero_point_frame` has an eager jax-in → jax-out check with host masks, directly and through
`ugeom.zero_point_frame` (`tests/test_dispatch.py:1898-1912`); `testing_strategy.md` checklist item 4. **Change:** add
`ufv_padded_frame` and the uniform geometry members (`padded_frame`, `padded_structures`) — jax supercores out, masks
numpy — and `padded_start` on both layers.

### m15 (MINOR) — With `old_ranks=None`, `padded_start` returns a frame of the zero tensor, not of `x0`

**Where.** §5.7 :1058, :1067-1068; T9(c) :1096.

With `old_ranks=None` the residual is taken at zero and the carriers are zero, so the returned frame's base point is
0 whatever `x0` holds. "The padded frame of `x0`'s tensor at `x0`'s structure", and T9(c)'s `frame.to_t3()` equals
`x0`, hold only for a zero `x0` or with `old_ranks` given. **Change:** "`old_ranks=None`: the cone frame at the zero
tensor of `x0`'s ranks (usable as `start_frame` for a zero `x0` only)", and restrict T9(c). Also state the frontend's
stack contract (a `C`-stacked `x0` gives a `C`-stacked frame and per-element norms — the `'gnorm': float` annotation is
then an array; a varying-rank uniform stack needs `(d,)+C` rank arrays, which the `Sequence[int]` annotation excludes).

### m16 (MINOR) — T5(b)'s monotonicity is an observation, not a property

**Where.** P13 :74; T5(b) :1092.

The captured norm is not monotone in general: from the staggered completion of M1 one pass lowers it from
`‖G_{2k}‖` to `‖G_k‖`, and a Haar frame can start above the fixed-point value (P-a: 5.59 > 5.006). **Change:** keep it as
a seeded regression with an explicit margin and say so, or replace it by T5(a)'s convergence and a final-vs-first
comparison.

### NITs

- **n17** — The CHANGELOG draft cites "plan §11" (:1267). No `dev/` references in shipped text (CLAUDE.md, the
  2026-08-24 sweep).
- **n18** — §5.2 :913: `seed` is "a convention, not a tunable", against decision 3 and `PaddedFrameOptions.seed`. Say
  "a convention; tunable through `PaddedFrameOptions(seed=)`".
- **n19** — R2 :1358: only `padded_start(..., old_ranks=None).gnorm` is the cone value; a `padded_from` start's `gnorm`
  is the deflated warm-start gradient and must not be fed to `g0norm_newton`.
- **n20** — `_prepare_jit_frame` (:569-571, :1042): reuse `common.partition_static` / `rebuild_static` with
  `tree_to_jax` on the dynamic leaves — the house split of a raw frame tuple. And :572-573 "never a cache key": a uniform
  frame's masks are value-hashed aux; the start frame adds no compile because its masks are value-equal to the
  template's.
- **n21** — T7(e) :1094: "the `_jitted` cache gains no entry" — the counter is the compiled program's `_cache_size()`
  (as `tests/test_dispatch.py:1194-1207`); `_jitted`'s `lru_cache` never grows per model.
- **n22** — `pinned_completion` step 4 :674: `xnp.where(…, 1, −1)` is an int64 array on numpy, which promotes a
  float32 `Q` to float64. Use float literals cast to `Q.dtype` (the float32 backlog).
- **n23** — §3.1 :446: pass `aux=geom_os.precompute(F)` to `project`, as `_zero_reference_gnorm` does
  (`backend/optimizers.py:646`); without it a shared geometry rebuilds its companion each pass (correct, slower).
- **n24** — Frontend: a `start_frame` with a non-minimal `x0` fails late in `require_frame_structure` after the
  minimal reduction changed the structure; give it `padded_from`'s clear structural error (:583-586). The uniform
  zero-padding check (:588-589) must read only `x0`'s real slots (padding beyond `x0`'s own masks is legal garbage).
- **n25** — Two cheap tests: `padded_from` equal to `x0`'s own ranks gives zero counts everywhere, so the start frame
  is the library frame bitwise and the run equals the default one — a direct check of the padded path's seam. And a
  `d = 1` hinted call (empty sweeps, zero-size `Ω`).
- **n26** — §5.6 :1047-1048: the display prints no history keys, so a padded start is not visible in the display at
  all; say that.
- **n27** — §0 :104 / §2.7 :370 "within 2% at ranks ≥ 7": record §2 has e01 `r = 7` at 96.9 vs 94.6 (2.4%) and e01
  `r = 3` at 75.8 vs 69.8 (+8.6%). Quote the measured range.
- **n28** — O-9 is settled by the house numbering ("a minor number that increments with each release of that year",
  `release_notes.md`): 2026.4.0.
- **n29** — `require_frame_structure` on the corewise geometries (:508): a corewise frame is the point itself, so a
  `start_frame` there makes no choice. Rejecting `start_frame` / `padded_from` structurally on corewise would drop the
  member (judgment call).
- **n30** — R5 :1361: a tie at a cut (`σ_k = σ_{k+1}`) is the other LAPACK-dependent case besides a rank-deficient hint.
- **n31** — The backend `padded_start` doctest (:1019-1021): assert "`st.gnorm` exceeds the library frame's norm" with
  a wide margin (the HANDOFF rule for doctests that run on both numpy generations).

---

## On the writer's "(weigh)" items

- **I-3** (one `PaddedFrameOptions`, two keywords): agree. A frozen dataclass passed per call, never a field, is the
  "options are not parameters" pattern; two call sites share three knobs.
- **I-5** (`K` rejection): see m9.
- **I-6** (HOSVD-style two-sided Tucker + mirrored sweeps + the down step): agree; verified step by step (P-b).
- **I-7** (not the δ-limit): agree; P9's `O(t)` rotations of the old block are the right reason.
- **I-8** (the down step from the new Tucker slots): agree. It is also the choice consistent with the cone reference:
  at `s = 0` it gives the down rows of the truncated hint's center, i.e. the frame P12's dense reference uses.
- **I-11** (in-block Gaussian keyed `[seed, i, c]`): agree.
- **I-13** (4 / 10 passes, `p = 5`): provisional until S11; fine as module constants.
- **I-19** (zero start = `padded_start(old_ranks=None)` + `start_frame`): agree.

## Checked and found correct (do not re-litigate)

- **The inventory.** Spot-checked `file:line`: `linalg.py:562/722/778`, `fv_operations.py:29-49/262-430` (incl. `_identity_prefix` / `_haar_rows`),
  `ufv_operations.py:302-354`, `geometry.py:326/462/706/863`, `optimizers.py:65-131/405/516-532/604/650-699/958-1028/
  1299/1320`, `optimizer_display.py:121`, `t3toolbox/optimizers.py:186/197/420-611`, HISTORICAL `:74`, AXIS_SIZE `:49-54`.
  All match.
- **§2.1 frame validity.** Per-core orthonormality plus the gauge make the 2d terms orthogonal isometries, with no
  relation required between the L and R chains beyond the carriers. Both reconstructions equal `X_old` because every
  path entering a new index ends at a carrier zero on new indices. P-b: orthogonality ≤ 1.8e-15, `(U,L)` and `(U,R)`
  vs `X_old` ≤ 1.8e-15, old `U` rows bitwise.
- **Monotone stores.** `compute_orthogonal_representation_ranks` (`ranks.py:501-568`) is min/product of its inputs at
  every step, so `(r, ρ) ≥ (s, σ)` gives old ≤ new in all four stores; `_empty_old_frame`'s `(1, 0, …, 0, 1)` is the
  recurrence at zero ranks.
- **The hinted construction** as specified (Appendix B; I mapped §4.4–§4.7's twelve strings to it one by one) equals an
  independent dense oracle at every step (P-b, ≤ 4.4e-11). The §4.5 ambient Tucker step equals the prototype's
  `w`-coordinate step in exact arithmetic whenever the hint has `≥ kU` directions
  (`Wᵀ(I − AᵀA) = (I − P_{U°})Wᵀ` since `U° ⊂ span W`); the two differ only where Householder completes, and both are
  valid.
- **P3's mechanism.** `tv_to_t3`'s boundary bonds are 1 (`tv_operations.py:290-330`), so the zippers' ones-boundary is
  exact, and the doubled-rank chains contain the hint frame's L and R blocks.
- **The `s = 0` reduction.** The Tucker draw is `_haar_rows` itself with the same `≥ 0` sign rule; outside-first
  reproduces `_identity_prefix`'s L `(a,n)`-major columns, R `(n,b)`-major rows and O `(a,b)`-major rows
  (`fv_operations.py:420-426`); the carriers are zero; QR column nesting gives nesting in `kU`.
- **Feasibility of the counts.** The in-block completion exists (`count − t ≤` in-block complement ⟺
  `rL_{k+1} ≤ rL_k·nU_k`, the recurrence), as does the down completion (`nD ≤ rL·rR`).
- **The primitives** take no threshold and are valid for any seeds; neither defect of the pad-safe record's Methods B/C
  (a rank tolerance with per-element replacement counts; masked noise) applies.
- **The seam.** Every frame-dependent consumer reads `lm.frame`: sweep, geometry aux, gradient, curvature (including
  `base_point(frame)`), retract, and the regularizer's value / gradient (`backend/optimizers.py:186-320, 395-416`); the
  residual reads `x_cores` only. At a padded frame the exact-mode centers / `S_i` / `T_i` (by zipper) and the shared
  companion are the old ones embedded with zeros, so the new slots get exactly zero singular values and the existing
  clip handles them — no content, no ε failure.
- **The driver.** Residual once (at `x_cores`, or at zero for the cone), gradient per frame through `kind.transpose`
  (`ω²` inside the kind, as `LocalModel.gradient`) then `geom.project`. The uniform kinds take their masks from the
  frame they are handed and re-mask (`uniform_fitting.py:88-140`), so the problem's own kind works at the oversampled
  structure; only the geometry needs an oversampled twin, because its `project` / `inner` use held `var_masks` (I-16
  is right). Oversample-then-truncate keeps the old block: it is pinned from `old_frame`, never from the hint frame.
- **References.** With a start frame the default `'zero'` reference reads only `lm.frame`'s structure, so its value is
  today's bit for bit. The no-effect errors (`start_options` / `cone_options`) follow the n16 rule, and a vanishing
  explicit `'cone'` raising follows the explicit-`'zero'` rule.
- **Options.** Solver keywords, constants module-level in the T3-free module, `seed=None` resolved by the geometry (the
  `curvature_damping` precedent).
- **Contractions.** All twelve new strings use letters sized in both AXIS_SIZE and the sharding test's `SINGLE`
  (`tests/test_contractions_sharding.py:62-64`); none is already in HISTORICAL; each is fully determined (`''`). The two
  reused strings exist (`t3_linalg.py:840-841`; HISTORICAL `:215-216`). Reshapes merge only trailing non-stack axes with
  `C` leading.
- **The uniform twin's garbage prong.** `ut3frame_to_t3frame` / `ut3variations_to_t3variations` gather through
  `argwhere` and never multiply by masks, so T4(c)'s NaN padding is a valid (stronger) garbage test. T4(a)'s bitwise
  comparison checks shapes too, so it also catches too-permissive or too-restrictive masks.
- **Frontend minimality.** `_ragged_minimal` and `uniform_minimal` are no-ops on minimal ranks
  (`optimizers.py:125-136`, `uniform_fitting.py:435-462`), so requiring a minimal `x0` preserves the padding;
  `test_newton_cg_keywords_mirror_backend` picks up the new keywords automatically.
- **Decision 4.** No threshold, no ε content, retraction and orthogonal representation untouched, counts from both rank
  sets; `fv_zero_point_frame` untouched (I-26).

VERDICT: REVISE
