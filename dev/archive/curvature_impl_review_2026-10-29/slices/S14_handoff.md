# S14 handoff — T3Polynomial updated for T3Toolbox 2026.3.0-rc (1f8ab24); T3Polynomial tree UNCOMMITTED on 388090d
T3Toolbox not touched. 12 files, +473/−143. Side effect: `pip install pyflakes` into the system python (for checks).
## Files (lines in the edited tree)
- `t3polynomial/fitting.py`: `_zero_gradient_norm` deleted (and its private `topt._geometry_ops` use); `_level_newton_kwargs` :105 (new);
  `_newton_fit` :176 — `g0norm_newton=None` RAISES ValueError :216 (judgment: P3 made the old opt-out spelling silently mean the zero ref);
  `g0_reference='warm_start'` (case-insensitive) warns, else `gtol_rel` defaults to 1e-3 :223; ladders build `kw` (max_newton/verbose/val,
  then `g0_reference='warm_start'` | `gtol_rel=gtol_ref`, then the level's options) :379 / :566; records `newton=stats['newton_steps']`
  + new `newton_accepted`, `newton_stop_reason` :389 / :577 (adaptive `stop_reason` stays the LADDER's); prints `newton_steps`; docstrings
  (module :28-42 + :65, `_newton_fit`, both ladders; `ls_steps` None on TR rows; `cg_truncated` excludes boundary stops). WARNING text respelled.
- `tests/test_fitting_selection.py`: K.2 as planned + stronger: X3 is now a resize of a FITTED rank-2 tensor (the unfitted seed's resize is not
  deflated: warm/zero 0.98 → 1≥1 vacuous); asserts labels 'zero'/'warm_start' (ladder + direct), warm g0 < 0.5·zero g0, strict step order,
  `g0norm_newton=None` raises. New `test_cold_zero_start` (I-37), `test_newton_schedule_hook` (I-36).
- Scripts: d07 :61 `topt.least_squares_problem(...).problem` (bopt/bfit imports dropped); d08 :114 (named setup fields); d05 :236 explicit
  LS filter + `cg_boundary` in the work dict/print; b01 :279 LS rows only (+`n_rows`); x06 `EXACT=1` column (`corewise_norm` of the diff,
  per S0b(d)); x07 `HESS=fd|exact`, `CONTROL_GTOL` (1e-12) on the controls, `EXACT_CONTROL=1` library exact-TR control, JSON extended.
- Docs: `dev/HANDOFF.md` new top block; `dev/t3toolbox-upstream-notes.md` 3× GaussNewtonModel→`fitting.LocalModel`, two ✅ FIXED UPSTREAM
  banners (2026-07-13 / 2026-09-03 reference notes), dated section :815; README + pyproject comment: requires T3Toolbox ≥ 2026.3.0.
## Schedule hook API (I-36)
`newton_kwargs` on both ladders: None (default, unchanged: GN + line search) | dict (every level) | callable `schedule(info) -> dict|None`,
called once per level before its solve; `info = dict(level, tucker_ranks, tt_ranks, dof, records)` (records = finished levels, a tuple).
Merged LAST: level options override the ladder's max_newton, gtol_rel (from gtol_ref), warm-start ref, val pair (was: duplicates → TypeError).
Any newton_cg keyword passes (hessian, step_control, tr_radius0, curvature_damping, cg_forcing_power, g0_reference, …); misspelt → TypeError;
non-dict return → TypeError. No script uses a schedule yet (e01/o01 knob = possible follow-up).
## Cold-start options after the change (I-37) — none removed (no per-level cold start existed before or after)
(1) all-zero (or randn) seed `x0` to either ladder → level 0 starts there, later levels warm-start via resize; (2) `_newton_fit` on
`TuckerTensorTrain.zeros(...)`. Verified GN + exact, ragged + uniform, ref 'zero' (`test_cold_zero_start` + toy probe: 3.5→0.105).
## e01 ratios (§K discrepancy (a)) — re-measured on the RC with s0b_g0_reference.py: library/downstream 2.175, 0.727, 0.943, 0.967
(L0..L3; identical to S0b); 0.95 on the unit-test toy at rank 3. Recorded in fitting.py docstring, HANDOFF, upstream notes.
## Tests (system py3.13 / numpy 2.5.3 / jax 0.12; no implicit_probing, no dolfinx; `--continue-on-collection-errors -n 2`)
- Baseline 388090d: 10 failed / 22 passed / 34 skipped / 2 collection errors (test_eigen, test_ot import implicit_probing). All 10 =
  KeyError 'newton' (fitting.py:353/513) + one consequent KeyError 'ragged' (test_fitting_recovery subtest dict).
- After: **31 passed / 0 failed** / 34 skipped / same 2 collection errors (exit 1 only from those). +2 new tests. With the S0b shim,
  test_eigen+test_ot: 10 failed / 24 passed on BOTH trees (shim limits; they never touch fitting/T3Toolbox).
## Scripts run (scratchpad copies of S0b data; S0b shim; 1 thread) — all EXIT 0
e01 small (4 lvls): 30/30/30/30, q 0.158, 42 s. o01 small: 30/30/23, dof 380, q 0.0478 (= S0b). b01 ragged+uniform arms 16x6×3; `--ls-hist`
GN 47/50 LS rows, exact 0/68. x06 L3: exact col O(t²) to 2e-8..5e-8 at t=1e-5 (= S0b). x07 L3: HF end |g| 2.1e-6 (fd) / 9.5e-7 (exact);
GN ctrl 10 steps |g| 3.3e-2; exact ctrl 7 steps → 2.2e-12. d05/s05/d07/d08 need Darcy data: changed lines smoke-tested on toy (d07 new route
bit-identical to old; d08 setup on ragged/uniform/uniform_jax; d05 filter on GN+exact records). pyflakes clean except 3 PRE-EXISTING
(x06 `rng`, x06 f-string, d08 `shape`). x01 has a PRE-EXISTING NameError (`K`), not touched.
## §K items that were wrong / incomplete
- K.3 d07 `least_squares_problem(...)[0]`: `LeastSquaresSetup` is a frozen dataclass, not subscriptable → `.problem`.
- K.3 d05 filter `step_control == 'line_search'` alone crashes: the final converged LS row has `ls_steps=None` → keep the None filter too.
- K.2 `:91` vacuous beyond the label (the X3 warm start is not deflated) → fitted warm start + strict asserts.
- K.1 `else: setdefault('gtol_rel')` now also covers an explicit float `g0norm_newton` (was library 1e-8); no caller passes one.
- s05/n01/x02 need no edit (break only via fitting.py, now fixed); x01's `_newton_fit(..., chunk_size=None)` is fine.
## Draft commit message (T3Polynomial)
fitting: update for T3Toolbox 2026.3.0 -- library zero reference, newton_steps, Newton schedule hook

T3Toolbox 2026.3.0 (the exact-Hessian release) removed stats['newton'] (KeyError at the end of every ladder level), made the zero-tensor
Newton reference its default (g0_reference) and gave g0norm_newton=None the meaning "no override". fitting.py reads newton_steps (records
keep `newton`, plus newton_accepted / newton_stop_reason), drops the hand-rolled _zero_gradient_norm (library/old reference 2.18 / 0.73 /
0.94 / 0.97 on e01's first levels: stops move by O(1)), spells the warned warm-start opt-out g0_reference='warm_start' and refuses the old
g0norm_newton=None; newton_kwargs also takes a per-level schedule callable(info) -> dict (GN vs exact is an open A/B question upstream).
Cold zero starts verified in both Hessian modes. Scripts: d07/d08 use optimizers.least_squares_problem; d05/b01/d08 count line-search rows;
x06 gains an exact-Hessian column, x07 a HESS=exact switch, an exact-Newton control and CONTROL_GTOL. Tests: +cold start, +schedule hook;
the reference test now warm-starts from a fitted tensor and checks the g0_reference labels. Requires T3Toolbox >= 2026.3.0.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017qDbbe5KVzCbMP39xChGE3
