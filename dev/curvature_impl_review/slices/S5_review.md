# S5 independent review — VERDICT: APPROVE-WITH-FIXES
Base main @ fffdfde, tree uncommitted (11 files; no new files). No BLOCKER, no MAJOR. Scripts: `scratchpad/review/s5/`.

## Independent checks (all run from the repo root, `PYTHONPATH=$PWD`)
- **Bit-identity vs fffdfde** (`bitident.py` + `bitident_compare.py`; fffdfde unpacked by `git archive` into `s5/old`, run in
  a separate process). 271 cases: 95 CG solves (SPD / indefinite / singular, three tols, plus 5 float32 runs), 168 ragged GN
  `newton_cg` runs (apply/probe/pd1/pd2 weighted × MANIFOLD/COREWISE × random/zero start × default/power 1/'accept'/'stop'
  with Armijo exhaustion, + regularized), and 8 uniform frontend runs. 2500 history rows. Every pre-S5 field, every loss and
  the final x bytes are **identical (0 diffs)**. The new fields are None/False/'gauss_newton'.
- **Steihaug body** (`optimizers.py` `_cg_cond`/`_cg_body`/`_cg_core`). State lives in the tuple; the body is module-level
  and closure-free; every branch is a `where`. inf only meets `>` and `isfinite`; `rad` is replaced by 0 where unused. τ is
  the positive root (`pp ≤ Δ²` ⇒ τ ≥ 0). Negative curvature with a finite Δ goes to the boundary; with Δ = inf it truncates.
  The outcomes are exclusive. Radius is cast to rs0's dtype, so float32 stays float32. The 3 call sites are updated.
  **`radius` is traced** (static_argnums=(4,) = use_jit). `test_cg_radius_is_not_stale` catches both a static radius
  (mutant: 5 compiles ≠ 1) and an ignored radius (mutant: no boundary stop) (`mutate_jit.py`).
- **The loop vs §C.9 steps 1–9**: all match. Exact mode reaches `Problem.local_model(hessian, curvature_damping)`. The only
  new imports are numpy and sys (no T3 module). The lift diagnostics are duck-typed.
- **Mutants** (`mutate.py`, 18): 16 RED from the start, including the four required ones (floor on accepted steps, no
  Δ_max, expand without the boundary, model rebuilt on reject). **2 survived**: (a) Δ←¼Δ instead of ¼‖p‖; (b) the guard step
  not counted as boundary (judgment call 6). Both are RED after F1/F2.
- **Display**: the 4 frozen LS strings == fffdfde's `_format_header` output (`display_frozen.py`), so the freeze is
  genuine. κ/σ/ε/Δ/‖Δx‖ go through `_num` (None and non-finite print as —).
- **T16 numbers** (`t16_numbers.py`). At power 1.0 the window points are e_k = 2e-2, 9.3e-5 (seed 1) and 6e-2, 1.9e-3,
  3.8e-6 (seed 2), with e_{k+1}/e_k² ≤ 1.02. At e_k = 9.3e-5 an order-1.5 tail would give a ratio of about 0.7, but the
  measured e_{k+1}/e_k^1.5 is 0.0074. Power 0.5 fails C = 2 quadratic (ratio up to 96.8). This demonstrates q-quadratic
  convergence; it does not merely pass.
- **T25**: the slack and minimal runs have the same zero reference g0 (the frame ranks are the minimal ranks), 14/13 and
  22/16 steps, and the same final e. `uniform_tr_smoke.py`: uniform GN+TR runs through the backend, and uniform exact gives
  the window stub's structural error. `gn_tr_vs_ls.py`: GN+TR ≈ GN+LS, so the radius shows no pathology.

## Fixes applied (in the tree)
- **F1 (MINOR, test gap)** `tests/backend/test_optimizers.py:976` new `test_shrink_is_a_quarter_of_the_step_not_of_the_radius`.
  A rejected interior GN+TR step (‖p‖=127 vs Δ=1e3) has next Δ = ¼‖p‖.
- **F2 (MINOR, test gap)** `tests/backend/test_optimizers.py:962-974` adds to `test_descent_guard` a 2-row case.
  A guard step with scale < 1 and ρ > .75 doubles Δ while `cg_boundary` is False. This pins judgment call 6.
- **F3 (MINOR, real defect)** `backend/optimizer_display.py:304,314` (+ docstring :281-285). Table reuse was keyed on
  "the previous row was rejected" only. With a `make_newton_display` callback shared across runs (continuation levels), a
  run ending on a rejected row (every floor stop does) handed its stale tables to the next run's first row, which is a
  different point. Repro: `test_optimizer_display.py:256` (new): the implementer's rule fails, the fix passes. Reuse now
  requires `info.lm is` the rejected row's model. Mutants (original rule, plan-literal rule) are both RED.
- **F4 (MINOR)** `CHANGELOG.md:156-157`: a third stale "not yet wired into an optimizer" note (the S1 lift entry) is removed.
- **F5 (NIT)** `docs/contributor/scan_body_principles.md:197`: "bit for bit" is now qualified (eager; to rounding under jit).

## Remaining findings (not fixed; NIT)
- n1: `optimizers.py:1070,1081`: `_floating_dtypes(tree)` / `_resolution_eps(residual,` have no type annotation. At :1127
  `curvature_damping:` overflows the name column by one, as the pre-existing `on_line_search_failure` does.
- n2: `import sys` (:34) only for `sys.float_info.min` (:1100); np is imported anyway. n3: non-breaking S5 CHANGELOG items sit under "Changed — breaking" (re-file in S13).

## Rulings on the 14 judgment calls
1 agree (plan + I-13 list ε). 2 **agree**: rows describe x *before* the step, so only the row *after* a rejected row
repeats a point. The plan's literal rule is off by one, and so is T19's "rejected rows do not call point_forward" →
S13 amendment (see F3 for the extra identity guard). 3 agree: C.14 also says "stop_reason appears on the final row", and the
change is display-only and in the CHANGELOG. "The only GN-visible change" is the plan's inaccuracy → S13; T3Polynomial does
not parse the display text (checked). 4 agree (exclusive outcomes). 5 agree (n16). 6 agree (the step is Δ-limited);
now tested (F2). 7 agree: `gn + cq` is exactly `lm.quadratic(p)`'s expression with one two-hole. 8 agree. 9 agree: 5 window
points total (2 and 3 per seed) ≥ the plan's 3, and they are discriminating (above). 10 agree: the try/finally restores the
saved value; xdist workers are separate processes and tests run sequentially within a worker; jax's jit cache keys
include x64. 11 agree. 12 agree. 13 agree. 14 agree: **S6 must delete `not_yet_frontend`
(`tests/test_optimizers_frontend.py:454-456`)**; add this to S6's gate.

## Prototype drift (handoff 25–27)
The explanation is correct, but the mechanism is the CG forcing term, not the stop (`drift.py`). The c20 warm-up uses
gtol_rel=0, so the reference enters only through η. x15 under the zero vs warm-start reference differs by 1e-9 / 2.5e-2 /
0.83 across c20's three problems. This is not an S5 defect. S13: pin `g0_reference='warm_start'` in the
c20/c21/c31/c31c warm-ups, or note it in the archive banner.

## Plan-text amendments to queue for S13
(a) C.14 n14 + T19: reuse after a rejected row, keyed on the same model (F3). (b) C.14: GN-visible = the converged suffix
**and** `| stop: <reason>` on a non-gradient final row. (c) C.9 step 6: pHp = gn_quadratic + curvature_quadratic, one
two-hole, == lm.quadratic. (d) NewtonInfo: define `cg_converged = rs ≤ tol² ∧ ¬boundary`. (e) C.9: 'accept'+TR raises;
tr_radius0 must be finite > 0; a guard step with scale < 1 counts as boundary. (f) T16/T21/T25 re-based on
probe_derivatives (o1 W200 6 GN steps; W60; 14/1 and 22/6, row-by-row losses ≤ 1e-8 instead of the final ‖g‖ ratio).
F22/F28/F29's apply numbers are historical. (g) S6 gate: drop `not_yet_frontend`. (h) §K/S14: T3Polynomial
`scripts/b01_backend_bench.py:275` appends `info.ls_steps`, which is None on TR rows, if it adopts exact.

## Gates (re-run by me, after my fixes)
Targeted 6 files `-n 2`: 191 passed / 849 subtests. test_dispatch: 46 passed / 36 subtests (235 s). Module doctests: 209
passed. All docs/*.md + contributor pages and getting_started.rst: OK. sphinx -W: exit 0. Slow tier (serial): 5 passed /
12,790 subtests. FULL SUITE, re-run last after all fixes: dispatch + backend 300 passed / 1,963 subtests; the rest 712
passed, 1 skipped / 42,410 subtests. Tree left uncommitted (the 11 files; my edits are F1–F5).
