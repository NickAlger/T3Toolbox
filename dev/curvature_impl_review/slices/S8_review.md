# S8 review: uniform exact mode (unshared). Base main @ fdcff71, tree uncommitted

**Verdict: APPROVE-WITH-FIXES.** Three reviewer fixes are in the tree. They touch tests, a docstring and CHANGELOG only; no `t3toolbox/` code. Nothing blocks. Scripts are in `review/s8/` (`check1_equiv`..`check6_jitkey`, `mut_review`, `bitident`, `time_*`, `gates.sh`).

## Reviewer fixes (uncommitted)
- **F3 (MAJOR test gap, closed)** `tests/backend/test_uniform_curvature.py:622` `test_the_frame_masks_at_tt_slack_through_the_model`.
  - m29 also covers the geometry members. A mutant `tangential_curvature` that masks θ with `self.masks` instead of the frame's `(up, left)` passed the whole suite and my clean-frame checks.
  - It only fails at TT slack (`self.masks` ⊋ frame masks) with garbage between the frame's and the point's extents, which is where a uniform frame keeps its completions.
  - The new test (0.16 s): kind and geometry hold the POINT masks, the frame is dirty; hvp, curvature_hvp and curvature_quadratic are bitwise equal to the clean frame's and equal ragged per element. It kills that mutant and M4.
- **F1 (budget, plan §F "≤ 3 s or slow tier")** `:901-1015`:
  - Always-on T21 truncates its template's level-1 fit (`_restart_template(W, level1_max_newton=12)`). It is still a zero-padded, rank-deficient restart: ragged 8/0, uniform 9/0, ~2 s.
  - Always-on TT-slack T25 compares the first 10 rows (2 rejected rows, kept by an assert): 1.2 s.
  - The full-length versions `..._full_template` (3.5 s) and `..._to_convergence` (4.4 s) are slow-tier (`_SLOW`, `:922`), both green. Tucker slack stays always-on (2.2-2.5 s).
- **F2 (text)** The module docstring said T21 is "row by row == ragged"; it is not, by contract (fixed `:28-31`). `CHANGELOG.md:147` said `inner` "kept |C| axes", but at fdcff71 it raised before that point (check4); reworded.

## Independent checks (packing and masks built by hand)
1. **Equivalence** (`check1`/`check2`):
   - Stacks: d=1 C=(2,); d=4 C=(3,) mixing minimal, Tucker-slack and TT-slack elements; **C=(2,2)**; C=(); K up to (2,3).
   - Per element vs ragged ≤ 9.5e-16 for every op. Model hvp 5.7e-15, curvature_hvp 9.5e-15, quadratic 4.3e-15.
   - Output masks derived from the RAGGED shapes: every padded slot is exactly 0.
   - Garbage of 1e6 in every padded slot (frame, θ, θ̂, v, y, z, ww/pp/z̃, and the corewise boundary-bond slots) leaves outputs **bitwise** unchanged.
   - NaN garbage poisons everything, because masking multiplies (`ut3_apply_masks`). This is library-wide and pre-existing, so "garbage" means finite garbage.
2. **m29** (`check3`, TT slack, C=(2,), kind and geometry on point masks, clean and dirty frame): equal to ragged and bitwise equal clean vs dirty. M4 is RED.
3. **Lift factors**:
   - A padded σ of 0 gives f = 0. Raw pinv padding and padded svals are exactly 0 on numpy and jax-CPU (`check5`). The real-dimension rcond uses host `np` masks.
   - The clip test is discriminating: the second σ sits between 2·eps and 8·eps. It kills M7.
   - **Jit key** (`check6`): one trace across 3 rebuilt models and 3 dampings, on both geometries. A changed mask *value* recompiles. `_cache_size` exists on jax 0.12 and 0.4.30, so the "CG compiles once" assert really runs.
4. **K-stack** (`check4`): at fdcff71 the uniform `inner` / `project` (manifold) raised on every K-stacked pair, so `inner` keeping K+C is a fix with no behaviour change for working code. It equals ragged `Geometry.inner` (all leading stacks); no divergence.
5. **I-33**: no `trs`, binomial or `_trs` in `t3toolbox/`. The wrappers delegate to the polymorphic recurrence kernels.
6. **Sharding**: all 18 new d-prefixed strings are in `HISTORICAL`, and `SINGLE` has every letter. No moveaxis, transpose or reshape on W/K/C. The only polymorphic branches are host-side (`is_ndarray`, static `d == 1`).
7. **Bit identity vs fdcff71** (`bitident`, separate processes): 326 arrays, 0 differ. Covered: uniform GN (4 kinds × 2 geometries: gradient / hvp / forward / inner / gn_quadratic / newton_cg / GD), uniform frontend GN, ragged GN and EXACT (manifold, corewise, shared manifold).
8. **Protocol**:
   - The uniform exact members mirror the ragged ones; on corewise they are identity / `()` / `None`, and a damping raises.
   - `ManifoldCurvatureData` is reused. The stub table holds only the S10–S12 rows.
   - Shared uniform raises NIE naming S9 (tested). The import DAG is acyclic.

## Mutants (own runner)
- RED: M4, Mgeo_tc (after F3), M6+M16, M5+Mlift_out, Mframe_entry, Mt3t3_dir, Mt3t3_out.
- GREEN: M5, M6, M8, M8b, M16, Mlift_out, Mz_once, Msvals_real, Mgeo_lt, Mgeo_lift. Each also survives my garbage and dirty-TT-slack checks, so they are redundant *under garbage*, not just on clean padding.
- Each is covered by an adjacent mask (entry × masked factors × output re-mask); removing an adjacent pair goes RED. M5+M8+M8b stays green only because a zero-bordered block's pinv padding is exactly 0. **Ruling:** accept as defence in depth.

## Rulings on the judgment calls
1. **T21: the implementer's reading is correct.**
   - The contract §"The full-rank qualifier" says never compare per-iteration trajectories from a padded restart.
   - Plan T21 asks only for stop=gradient and ≤ 2 rejections, "ragged + uniform"; T25 asks for "a uniform twin on the frontend path". The uniform test asserts more than that (same minimizer to 1e-9, frontend == backend).
   - The 1e-9 minimizer check assumes both layers land in the same basin (S0b); loosen it if it flakes.
2-5. Blocking shared corewise too (S9 lifts both), the K-stack fixes, the `LiftFactors` layout with real-dimension rcond (M7 is load-bearing), and `from_sweep` taking the uniform sweep: all OK.
6. No uniform `_trs` twins: OK. I-33's "written once in S8" means the uniform kernel is written once on the converted kernels; pinning is transitive.
7. Budget: resolved by F1.

## Open nits (non-blocking)
- `ut3_inner_product_corewise_transpose` (`ut3_linalg.py:343`) runs the ragged one-hole on supercores (`xmap`, unrolled over d under jit). It is correct and not on the hvp path; document it or give it a scan form.
- `ufv_masking._masks_over_stack` is private but used across modules (`geometry.py:499`, `utv_operations.py:683`); make it public. The shared-uniform NIE is raised before the d=1 shortcut (`geometry.py:722`); ragged returns None first. Harmonize in S9.
- Compile time (~15 s exact vs 9.5 s GN) is recorded, not asserted (the `< 300 s` is a 20× guard). That is fine.

## Plan text for S13
- **§M** actual edges: `ut3_linalg→t3_linalg` (not `ufv_conversions`), `ufv_operations→contractions`, `utv_operations→contractions`, `geometry→ut3_masking`.
- **§F T21/T25 uniform rows** per ruling 1:
  - The uniform Problem requires a minimal x0, so T25's "no reduction" frame claim is vacuous on uniform; m29 is pinned by unit tests.
  - Record the F1 split. **The release-gate slow-tier command must include `tests/backend/test_uniform_curvature.py`** (HANDOFF/CLAUDE.md name only `test_oracle_sweep.py`).
- **§C.10**: uniform `LiftFactors` has d bond slots (the last is the pass-through) and d-1 svals; rcond per REAL block; `ut3_*_from_sweep` take the uniform sweep; m29 extends to `lift`, `lift_transpose` and `tangential_curvature`.
- **testing_strategy §2**: garbage means finite garbage. DEFERRED rows closed: S0c m1, S6 review (S8), S6 n4, S3 implementer.

## For S9 (shared uniform)
- Lift the NIE in both uniform geometries and `_check_uniform_lift_factors`; flip the NIE tests. DEFERRED S7 (S8/S9) row: zero padded `s_g` by the group rank mask; keep a real zero level at a padded restart; real-slot-only `sigma_min`; one slice per group mode, re-masked.
- Shared corewise is `project_mean(K θ̂)`. Add a T12 uniform twin under the T17 rules.

## For S10–S12
- Copy the `UniformProbeDerivativesKind.corewise_hessian` pattern (frame masks, θ̂ masked with θ's masks, output re-masked), with a TT-slack test per kind in the F3 shape.
- Update the `in_window` predicate in `test_the_uniform_window_stub`. S12 deletes the stub and the table.

## Gates (final tree)
- Targeted (4 files, -n 2): 165 passed, 2 skipped / 987 subtests.
- Contractions (interpreter + sharding, no skips): 18 passed / 2867 subtests.
- Module doctests: 217 passed. Doc pages, getting_started: OK. sphinx -W: exit 0. `git diff --check`: clean.
- Floor leg (py3.11 / numpy 1.26 / jax 0.4.30): 117 passed, 2 skipped / 360 subtests; after F3, uniform_curvature 23 passed, 2 skipped. The brief's path `tests/test_uniform_fitting.py` should be `tests/backend/`. `docs/chunking.md` doctest: OK.
- Full suite: half 1 344 passed, 2 skipped / 2212 subtests (after F3); half 2 746 passed, 1 skipped / 42839 subtests (1 pre-existing warning).
- Slow tier, serial (oracle + uniform_curvature): 29 passed / 12902 subtests.
Not committed; `dev/` and CLAUDE.md untouched.
