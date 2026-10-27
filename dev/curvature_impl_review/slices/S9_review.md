# S9 review: exact mode on the SHARED UNIFORM geometries. Base 2c42d4c, tree uncommitted. Scripts: review/s9/
## Verdict: APPROVE-WITH-FIXES (reviewer fixes are text only, no code logic; nothing blocks)
- **F1 (MAJOR, user-facing API docs).** Stale docstrings outside the diff: `t3toolbox/fitting.py:216-219` and
  `:909-911` said a shared uniform geometry raises `NotImplementedError`; `t3toolbox/optimizers.py:466-468` said
  "ragged ... so far" (stale since S8). Now: every geometry, uniform twins and shared wrappers; other kinds raise.
- **F2 (text).** `tests/backend/test_uniform_curvature.py:24` stale ("shared -> next slice's error"); + S9 paragraph.
- **F3 (accuracy).** "Padded s_g levels sit at roundoff, not at zero" is overstated. `survey_padded_levels.py` (300
  random shared structures, packed and uniform-built frames): 1357/1362 padded levels are EXACTLY 0, the other 5
  <= 3.3e-17 s1 (<= 5% of the real-dims clip). U_M's padded rows reach 1.7e-15, which the projection re-mask removes.
  Reworded `CHANGELOG.md:313`, `geometry.py:728-730`, `ufv_operations.py:530-534`, `sharing_internals.md:154-166`.

## Independent checks (all pass)
1. **Equivalence (`check1_equiv`).** Own packing (hand zero-pad, prefix masks from ragged shapes); C=(3,)/(2,)/(),
   K=()/(2,), damping 0/1e-3/1; ceiling (n_g 4 > member nD 2, n_g < N), two-group, all-mode; 3 hash seeds.
   - Per element: filters 6.5e-16, lift 1.4e-15, transpose 9e-16, grad/hvp/curvature_hvp <= 4.9e-15 (both
     geometries); sigma_min == ragged (4e-16) and > 0 on every element; masks exact; outputs tied.
   - Finite garbage leaves all outputs BITWISE equal: frame, sample, residual, tangents, and the companion's
     U_M/Vt_M/s_g padding (padded levels above s_g[0]). Scalar quadratics reach 1.5e-13 rel on random draws.
2. **S7 warnings (`check2_levels`).** (a) A real level at 1.77e-15 s1 is kept on both layers (f 4.0e14); a padded
   level injected at 1.5e-15 gets exactly 0 (clip-only: 4.7e14). (b) Stack [padded restart (2,2,2)->(3,3,2),
   full rank]: sigma [0, 1.152] == ragged; uniform-built, sigma reads the real zero level (1.5e-29), not a padded 0.
   (c) `_tie_group_slices` / `_group_sum` touch only the d axis and sum in ragged order (no W/K/C reorder).
3. **Mutants (`mutants_rev`).** 16/16 RED, incl. m29 `self.masks` via the TT-slack shared test. The
   padded-s_g-not-zeroed mutant is killed ONLY by the garbage-companion test (mask = defence in depth; keep it).
4. **Bit identity vs 2c42d4c (`bitid_rev`, separate processes).** 3 structures x tight/forced padding x 2 seeds x 2
   geometries, plus zero-padded restart, ragged shared exact, unshared uniform exact. 730 arrays; 14 differ, all in
   mask-padded slots (old <= 8e-17 rel, now 0), 0 in real slots. Every newton_cg / GD loss, history, final
   supercore: bitwise equal.
5. **End to end (`check6_e2e`; outer/rej/CG).** T22: ragged 10/0/41, uniform 13/0/40 (rel 0), per-mode 13/0/40
   (does not discriminate). Ceiling restart: 7/0/28, 7/0/24, per-mode 12/0/59 (fails <= 9: adequate). Serial times:
   T22 2.90 s, ceiling 2.58 s, full-rank 1.58 s, uniform frontend T22 3.26 s. Dispatch 30.9 s (precedent 36.4 s).
6. **Structure.** Import DAG (AST) acyclic; only new edge `ufv_operations -> sharing`. No `trs` (I-33). `git diff
   --check` clean; no `dev/` or CLAUDE.md edits. "Fixed" fits the re-mask (masked-clean contract; no real value
   changes).

## Non-blocking findings
- **n1 (budget, S13).** Uniform frontend T22 3.26 s and S8's frontend T21 3.61 s (pre-existing) exceed the 3 s
  budget. Move to slow tier, or drop the bitwise backend re-run.
- **n2.** `ufv_tied_lift_factors`: `shared_data` comment 3 columns right of its neighbours (cosmetic).
- **n3.** More private cross-module calls (`sharing._tied_*`, `ufv_operations._real_group_levels`; S7 precedent).
  S8's "make `_masks_over_stack` public" is still open.
- **n4 (pre-existing).** The slow tier's jax UserWarning (float64 ShapeDtypeStruct truncated) comes from the
  `chunk_size='auto'` estimator (`sampling_derivatives.py:1868/1904`) on uniform `probe_derivatives`. Base
  2c42d4c emits it too. S13: request a supported dtype, or suppress it.

## Rulings on the judgment calls
1. **AGREE.** The reverse edge would cycle (`ufv_conversions -> ranks -> sharing`). §M: add the new edge.
2. **AGREE: hygiene.** Padded slots only; every real value and every run is bitwise equal (check 4).
3. **DEFER (new ledger row).** `check_call3`: at a window point (real level 1.8e-15 s1) the uniform projection is
   17% from ragged; garbage padded levels move it 564%. Why defer:
   - The window point is numerically rank-deficient (inside the contract's full-rank qualifier).
   - No library path builds a garbage companion (frame masked first; natural padded levels <= 5% of the clip).
   - At the default damping the lift's filter there is ~s/eps^2, about 0, so the Hessian cannot see it.
   - The fix spans GN-path code (projection, `fv_tied_ambient_directions` for the retraction,
     `ufv_tied_variations_residual`) and needs its own bit-identity run.
   Proposed row: one helper (real-dims clip x real-level mask) for all uniform tied solves, plus a
   garbage-companion test of the projection; expect bitwise at every full-rank point.
4. **AGREE** (S8 T21 ruling; row by row in the full-rank ceiling start, observed 1.7e-14). 5. **AGREE** (own test).

## Plan text for S13
- **§M**: `ufv_operations -> sharing`.
- **§C.10 / C.11**: `ufv_tied_lift_factors` lives in `ufv_operations`. The filter clips at the real dims
  (`eps*max(sum nD_i, nU)`) times the host real-level mask (`min(sum nD_i, nU)` levels). The lift checks the
  companion's padded-nD row splits. sigma_min uses each group's real s_g levels in place of its modes' own.
  Shared uniform corewise needed no code.
- **§F**: T12's uniform twin adds hand-built companion garbage to T17. T22 uniform asserts its own properties; the
  ceiling restart is the discriminating check.
- **testing_strategy**: finite garbage includes derived containers (the companion).
- **DEFERRED**: close rows 42 and 51; add the call-3 row and n4.

**For S10-S12:** shared uniform exact comes via the geometry; add one shared-uniform subcase per kind (F3 shape).

## Gates (final tree, after F1-F3)
- Targeted 9 files (-n 2): 334 passed, 2 skipped / 4306 sub. `test_dispatch`: 53 passed / 40 sub.
- Module doctests 218 passed. Doc pages + `getting_started`: OK. `sphinx -W`: exit 0.
- Floor leg (numpy 1.26 / jax 0.4.30): 215 passed, 2 skipped / 599 sub; `chunking.md` OK.
- Full suite: dispatch+backend 363 passed, 2 skipped / 2298 sub; rest 748 passed, 1 skipped / 42845 sub,
  1 warning (pre-existing adam).
- **Slow tier SERIAL** (oracle + uniform_curvature): 48 passed / 12988 sub, 3 warnings, none new: adam x2, plus the
  jax float64 warning (n4), which base 2c42d4c reproduces in `test_full_matrix_slow`.
