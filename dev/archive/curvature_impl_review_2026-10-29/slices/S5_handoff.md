# S5 handoff (implementer) -- DONE, all gates green; tree UNCOMMITTED (base main @ fffdfde). Scratch (session-local, not archived): impl/s5/
## Files (11)
- backend/optimizers.py: Steihaug `_cg_cond`/`_cg_body` (state +radius, boundary; `where`-only; radius cast to rs0's dtype),
  `_cg_core(..., use_jit, radius)` -> (p,i,rs,ok,boundary), `_cg_solve(..., radius=inf)`; TR constants `_TR_*`;
  `_check_step_control`, `_resolution_eps`, `_initial_radius`, `_lift_diagnostics`; `newton_cg(hessian, curvature_damping,
  step_control, tr_radius0)` (+ doctest: exact+TR vs GN); NewtonInfo +hessian/tr_radius/tr_accepted/cg_boundary/pKp/
  lift_sigma_min/lift_eps; Geometry protocol: optional curvature-data diagnostics `sigma_min`/`eps`. Imports +numpy, +sys only.
- backend/geometry.py: `ManifoldCurvatureData.eps` (property -> lift_factors.eps). backend/optimizer_display.py: `_cg_symbol`
  (∂), `_step_segment` (ls byte-identical / `TR Δ … ✓/✗`), `_curvature_segment` (κ σ ε), `stop: <reason>` on a final
  non-gradient row, table reuse after a rejected row; + doctest (exact TR row).
- tests: test_optimizers.py (+TestSteihaugCG T14, +TestTrustRegionNewton T15 a-e numpy, 5-tuple call sites);
  test_optimizer_display.py (+LS byte-identity vs frozen fffdfde strings, test_exact_mode_header, table reuse/no
  val forward); test_curvature.py (+T16 x3, T21, T25 x2); test_dispatch.py (5-tuple at :791, +test_cg_radius_is_not_stale,
  +exact jit: CG compiles <= 1x across two runs + T15(e) float32 floor). docs/contributor/{scan_body_principles,
  parameters_not_closures}.md (CG state/radius prose). CHANGELOG (Changed x4, Added x1, two "not yet wired" notes fixed).
  test_optimizers_frontend.py: the S0e keyword-mirror test skips the 4 backend-only keywords (S6 adds them, drops the skip).
## Verified (scratch)
- GN/line-search newton_cg BIT-IDENTICAL to fffdfde on 64 cases (ragged apply/probe/pd1/apply_d2 x man/cw/zero start/
  Armijo floor/'accept'/power 1/regularized x 2 seeds + uniform apply/pd1 x man/cw): every old history field + final x bytes.
- Real newton_cg(exact) == the prototype TR loop (critic steihaug + c21.run_v3) on the same exact hvp, step for step
  (e, Δ, ρ, accept, stop) on T16 s1/s2 x p1/p0.5 and c21 (A)/(B) analogs (proto_vs_real.log).
- 14 mutants RED (floor-on-every-step, no Δ_max, expand w/o boundary, model rebuilt on reject, old cg_truncated, no
  boundary on neg. curvature, CG bits, unscaled guard, eps min, accept>0, pHp w/o curvature, literal display rule, LS
  text drift, κ unguarded) (mutate.py).
## Prototype scripts as-is: c20/c21/c31c RUN (no shim) but DRIFT: their warm-up uses the frontend newton_cg (zero ref
  since S0d). c20 now: exact quadratic constants 0.79/0.74 (s1), 0.94/0.86/0.73 (s2); GN 0.50-0.81. c21: (A) floor at
  it 3 (|pred|/res 1.5), (B) floor at it 1. c31c rates 0.48-0.92 / 0.37-0.92 (n19: 0.98,0.78 / 0.95,0.43,0.50).
## Numbers through the REAL solver (probe_derivatives; apply has no two-hole until S11)
- T16 (o1, W200, rank-2 fit of rank 4, 1%, start = 6 GN steps): p1.0 e_{k+1}/e_k^2 = .234,.768 | .538,1.02,.813 (C=2);
  p0.5 e_{k+1}/e_k^1.5 = .605,.754,.706 | .320,.407,.692 (C=2; e/e^2 up to 18.2/5.95: not quadratic); |1-ρ| <= 1.6e-3;
  both 'objective_floor' in 5-7 rows; GN ratios 0.45-0.68 (F22 apply: GN .56-.68, exact <= .92; F29 p.5 .43-.98).
- T21/F19/F26 (c24 template W120 o2): exact zero/p.5 9 outer 22 CG 0 rej (F26 9/27); warm/p.5 10/28 (F26 10/28);
  zero/p1 7/24 (F26 7/24); warm/p1 8/26 (F19 8/23); GN+Armijo 37/128 (F26 36, F19 39). Test uses W60: 8 outer, 0 rej.
- c21 analogs (o1 W100): (A) 4 acc -> objective_floor at |pred|/res 2.4; (B) floor at the first trial (1.1e-4).
  float32 jit (A): floor at the first trial, 0 accepted (pinned: O-5). F27 (apply E2) not reproducible until S11.
- T25 (c42 setup): Tucker slack (W100 o1) 14 outer/1 rej, TT slack (W200 o0) 22/6, slack == minimal outer & accepted,
  losses row by row <= 3.6e-10; damping 0 identical (F28/F38 apply: 20/34).
## Gates (all green): full suite two halves -n 2 -- dispatch+backend 299 passed / 1,963 subtests; rest 712 passed, 1 skip
  / 42,410 subtests. Slow tier SERIAL 5 / 12,790. Module doctests 209; quickstart + all doc pages; sphinx -W exit 0.
## Conflicts / judgment calls
1. Exact cells: plan C.14 `κ σ ε` vs brief `κ σ` -> followed the plan (I-13 lists ε).
2. Rejected-row tables: plan's literal "reuse when info.tr_accepted is False" would print the PREVIOUS point's errors on a
   rejected row (rows describe x before the step). Implemented: reuse when the PREVIOUS row was rejected (the only rows
   whose x is unchanged). Literal-rule mutant RED.
3. `stop: <reason>` on a last non-gradient row is GN-visible too (max_newton/line_search) -- C.14 also says "converged
   suffix is the only GN-visible change". Non-final GN rows byte-identical (frozen strings); CHANGELOG lists it.
4. `cg_converged` excludes boundary stops (outcomes mutually exclusive); plan defines only cg_truncated.
5. Raise on on_line_search_failure='accept' with TR, and on tr_radius0 not finite > 0 (n16 spirit; plan silent).
6. Descent guard in TR: the scaled step counts as "on the boundary" (scale < 1) for the 2Δ rule; cg_boundary = CG's flag.
7. Exact pHp = float(gn_quadratic(p) + curvature_quadratic(p)) == lm.quadratic(p) with ONE two-hole (plan's literal
   pHp = quadratic + pKp = curvature_quadratic evaluates it twice).
8. lift_* read duck-typed (`sigma_min`, `eps`) off geometry_data; set on every exact row (they describe the model).
9. T16 recalibrated: o1; start = 6 GN steps (at 15 the pd fit is 1-2 steps from its floor); >= 2 window points per seed
   for p1.0 (plan: 3); window = e_k < 1e-1 AND |pred| > 1e3 res for both powers.
10. T15(e) jax run in test_dispatch (compile ~19 s; x64 toggled off in-test, try/finally: test_dispatch turns x64 on at
   import, so x64-off can't be assumed in a suite run).
11. Δ0 falls back to ‖g‖ also when ‖g‖³/q is 0. ‖ω⊙y‖ computed lazily at the first rejection (LS mode does no extra work).
12. Chunk size: confirmed -- backend newton_cg never resolves 'auto' (the kind arrives resolved); S6 owns hessian= there.
13. User docs (fitting_and_optimization §4.12, stats table) untouched (S13; the frontend can't reach TR before S6).
14. Scope guard vs S0e's `test_newton_cg_keywords_mirror_backend` (every backend option must be a frontend keyword):
   the frontend keywords are S6, so the test now excludes hessian/curvature_damping/step_control/tr_radius0 with a
   comment -- **S6 must delete that exclusion**.
## Draft commit message
backend: Steihaug-Toint CG, the trust region and hessian='exact' on newton_cg (S5) -- _cg_solve(radius) -> (p,i,rs,ok,
boundary), radius=inf bit-identical; newton_cg(hessian, curvature_damping, step_control, tr_radius0): TR acceptance,
radius rules, the objective/trust-region floors; NewtonInfo TR/exact fields; cg_truncated excludes boundary stops; the
display's step segment (TR Δ ✓/✗, ∂, κ σ ε, stop)
  Body: Slice S5 (C.9, C.14). T14, T15 (a-e), T16 (recalibrated on probe_derivatives), T19, T21, T25; dispatch: radius
  not stale, exact CG compiles once. GN line search bit-identical on 64 cases; real TR == prototype loop step for step.
