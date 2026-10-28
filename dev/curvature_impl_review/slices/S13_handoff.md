# S13 handoff (implementer) -- DONE; base a2fc053; UNCOMMITTED (orchestrator commits). Scratch: impl/s13/
## Files
- tests: test_jax_absent.py (T28: LiftFactors / ManifoldCurvatureData / CurvatureState / exact fitting.LocalModel built
  + hessian(p) + newton_cg(hessian='exact', use_jit=True) in the jax-hidden subprocess, warning re-armed); NEW
  oracle_sweeps/sweep_exact.py (dense EXTRINSIC oracle: Weingarten form via the parametrization's exact Jacobians;
  corewise = plain Hessian of f∘Φ; tangent-dim check; group-ceiling shared d=3) + test_oracle_sweep.py (both tiers); test_dispatch.py
  TestCurvatureJvpOracle (jvp of 7 ragged corewise transposes + the uniform pd one vs the two-holes, f64, 1e-12);
  O2 sweeps seeded per case (sweep_models/sweep_optimizers run_case: one line each; README records it);
  backend/test_curvature.py (_plain_fit_away_from_singular: T7 predicate redraw; aux=None fallback test; T16's I-34 docstrings).
- code (no numerics change): sampling_derivatives.py estimator requests jax.dtypes.canonicalize_dtype (no float64
  UserWarning; same program); backend/optimizers.py (_floating_dtypes/_resolution_eps annotated, `sys`->_TINY,
  newton_cg signature regrouped); optimizers.py (signature name column realigned); fitting.py / backend optimizers
  docstring qualifier; geometry.py (from_frame named); sharing.py (tied-lift exactness: sufficient not necessary).
- docs: fitting_and_optimization (stats table, §2.1, §2.2 LocalModel method table + doctest, §3 bitwise qualifier +
  six-op / exact geometry members, §4.2 qualifier, NEW §4.12 + doctest, §5 schedule bullet); rank_continuation
  (per-level schedule, exact at restarts, cold start); sharing; chunking (reworded; shard_map fixed per the
  sharding test; exact estimator doctest); numerical_contracts (exact caveats); transposes; naming_conventions;
  api_reference; user_guide; release_notes (Upgrading from 2026.2.0 rewritten); CHANGELOG [2026.3.0] consolidated
  (Changed-breaking / Changed / Fixed / Added; every behaviour kept); NEW contributor/curvature_internals.md
  (+toctree); deferred_and_rejected; verification_records; numerical_contract_catalog; parameters_not_closures
  ("Options are not parameters"); testing_strategy. examples/fit_hilbert_uniform_newton_cg.py (claim fixed).
- dev/: DP -> archive/riemannian_newton_curvature_plan_2026-10-27_implemented.md, G0P ->
  archive/zero_reference_g0_plan_2026-10-27_implemented.md, curvature_oracles/ -> archive/curvature_oracles_2026-10-27/
  (banners; plain mv: shows as D + ??); 22 check scripts bannered (pre-S3b x10, stats['newton'] x8, drift x4);
  HANDOFF refreshed; DEFERRED.md every S13 row marked; OPEN_QUESTION (+sixth op, ten stubs); paper_scope.
## DEFERRED disposition (one line per S13 row; full text in DEFERRED.md)
done: S0c m2 | S0d example | S0d handoff | S0e paper_scope | S1S2 O2 seed | S3b chunking reword | S3b shard_map | S4
OPEN_QUESTION | S5 drift banners | S5 n1-n3 | S6 (a)(c)(d) doc | S6 (b)(e) docstrings | S6 n2/n4 (HANDOFF) | S0b §6
(policy data; S14 part left) | S7 n2,n3,n4 | CI gate (floor leg) | both I-36 experiments | S9 n4 (warning fixed) |
S10S11 (E2, naming, T7 guard) | S12 checklist.  ledger: S3 _wchunked_sum | S3b M7 | S7 n1 | CI leftover-chunk | S9
tied projection clip | S7b follow-ups.  plan text (orchestrator): S0e C.16 | S1S2 M5 | S5 x4 | S6 x2 | S7 | S7b §H |
S8 | S9 | S10S11 x2 | S12.  left for Nick: S3 memory gate | S7b F1.  left: S0e xref transform (optional, pre-existing).
## Plan-text amendments (the orchestrator applies; all collected from DEFERRED + S13)
1 C.16: `_setup`/least_squares_problem reduces a UNIFORM x0 on both geometries, a ragged one only on manifolds.
2 M5: tests/test_contractions_sharding.py has its own size table (l/n/x/z added in S1).
3 C.14 n14/T19: reuse the error tables when the previous row was rejected AND `info.lm is` that row's model.
4 C.14 (I-35): `| stop: <reason>` also prints on a non-gradient final GN row (display only).
5 C.9 step 6: pHp = gn_quadratic + curvature_quadratic with ONE two-hole (== lm.quadratic); cg_converged = rs<=tol²
  ∧ ¬boundary; 'accept'+TR raises; tr_radius0 finite > 0; a descent-guard step with scale < 1 is a boundary step.
6 T16/T21/T25 re-based (S5): probe_derivatives o1, W=200, 6 GN warm steps, ≥2 window pts/seed; T21 W=60; T25 14/1, 22/6.
7 C.13/T18: frontend exact hessian(p) == backend hvp to rounding (≤1e-12), curvature actions/gradient/quadratic bitwise;
  `_with_curvature(model, hessian, curvature_damping, who)` via dc.replace.
8 C.11/C.7/I-13 (S7): lift_sigma_min uses s_g IN PLACE OF the group modes' own spectra; §M sharing -> contractions;
  C.2 LiftFactors.shared_data / tied_filters, builder sharing.fv_tied_lift_factors; T12 group-ceiling + the
  retraction-free oracle; T22 non-discriminating for the lift type.
9 §H S7b: floor max(own, scale); recursion max(‖a‖,‖b‖), |c|·‖a‖ through * and /, stack_tangents; scope; F1; F2.
10 §M (S8): ut3_linalg -> t3_linalg, ufv_operations -> contractions, utv_operations -> contractions, geometry ->
  ut3_masking; T21/T25 uniform assert own properties; C.10 uniform LiftFactors d bond slots, rcond per REAL block;
  m29 extends to lift / lift_transpose / tangential_curvature; "garbage" = finite garbage.
11 §M (S9): ufv_operations -> sharing; ufv_tied_lift_factors lives in ufv_operations; REAL-dim clip × real-level
  mask; T12 uniform twin with companion garbage; T22 ceiling restart (7,7,4)(3,3,2)->(4,4,2); padded s_g mostly
  EXACTLY 0 (1357/1362) -- "sit at roundoff" overstated.
12 F27 (S10S11): library == critic harness row for row; E2 = ONE unseeded draw (cite tr2/c29b); chunk 'auto' -> None
  for the five other kinds both modes; T11 vacuous for apply/entries; assemblies factored (9->4, 3->2).
13 S12: F28 reproduced EXACTLY (+ apply_derivatives o1 17/4/56, 31/10/240); F22/F29 state the GN warm-up's
  reference (warm_start -> 0.917; zero -> 0.941); F11 superseded; twins' homes; window/stub history; §E ~2× GN peak,
  no W-chunk path; an apply_derivatives T16 draws by predicate; bitwise garbage for equal (C-contiguous) layouts.
14 S13: §F oracle sweeps -- sweep_exact.py is a dense extrinsic oracle (always-on 720 cases, slow 8000), group-ceiling
  shared point; the jvp oracle lives in test_dispatch (TestCurvatureJvpOracle), not a separate file; §G "dotted
  kernels hold the GN standard" -> "are recurrences" (I-33); §G dev routing paths (archive names above) -- the plan's
  own references at :10, :21-22, :1499 point to the old paths; CHANGELOG: the sixth op is under "Changed" (S5 n3).
## Suggested plan status banner (replace the :3-13 status block's first line)
"STATUS 2026-10-27, after S13: the curvature arc is COMPLETE (S0c–S13 implemented + reviewed; records in
curvature_impl_review/slices/). Next S15a, then the PAUSE. The plan stays LIVE: it sequences R (dev/reorg_scope.md),
S14 (§K) and S15b before the tag. DP and G0P are archived (dev/archive/*_2026-10-27_implemented.md); the durable record
is docs/fitting_and_optimization.md §4.12 + docs/contributor/curvature_internals.md."
## CLAUDE.md patch (impl/S13_claude_md.patch)
Verification: names sweep_exact + the SERIAL slow-tier release command incl. test_uniform_curvature.py + the jax-0.4.30
floor leg. Current state: 2026.3.0 in progress (arc complete through S13; S15a -> PAUSE -> R -> S14 -> S15b -> tag);
suite counts; a new bullet for the exact Hessian (surface, rules, docs, open-for-Nick items).
## Gates (both pythons)
py3.13/np2.5.3/jax0.12: module doctests 225 passed; every docs/*.md + contributor/*.md page + getting_started OK; sphinx -W
rc 0; touched example (fit_hilbert_uniform_newton_cg) runs (16 s); jax_absent + oracle_sweep (always-on) + dispatch: 63
passed, 2 skipped / 12851 sub; full suite: dispatch+backend 414 passed, 2 skipped / 3270 sub; rest 756 passed, 2 skipped
/ 43094 sub (1 warning: adam-on-manifold, by design); SLOW SERIAL 66 passed / 13173 sub (2 warnings, both adam; the S9-n4
float64 estimator warning is gone). py3.11/np1.26.4/jax0.4.30: pages + getting_started OK (chunking.md incl.); floor leg
226 passed, 2 skipped / 1890 sub; full suite 414 / 3270 sub + 756 / 43094 sub (2+2 skipped; adam warnings). diff --check OK.
## Judgment calls
1 sweep_exact's oracle is EXTRINSIC (P = JJ⁺, DP = P⊥DJ J⁺ + (.)ᵀ with exact multilinear DJ) -- independent of every
  library lift/two-hole; so fast (720 cases/25 s) that the whole matrix is always-on; slow = seeds x layouts x order 3.
2 jvp oracle in test_dispatch (x64 already on there) rather than a new file. 3 Estimator warning fixed (2 lines) rather
  than filtered: same compiled program, same answer. 4 CHANGELOG: "Changed" (non-breaking) section added for the sixth
  op, cg_truncated, display, _cg_solve, bit-identical refactors. 5 Not done: S3b M7 plain-op dtypes, S9 tied-projection
  clip, S7 n1 -- all GN-path code changes outside S13's remit -> ledger. 6 dev moves are plain mv (orchestrator stages).
## Draft commit message
docs+tests: the exact-Hessian arc's docs, CHANGELOG and remaining tests (S13) -- fitting §4.12 + the LocalModel table,
NEW contributor/curvature_internals.md, release notes, consolidated CHANGELOG [2026.3.0]; T28 jax-absent construction,
sweep_exact.py (dense extrinsic oracle, both tiers), the jvp oracle; O2 sweeps seeded; dev/ archive (DP, G0P, oracles)
  Body: Slice S13 (§F, §G). No tag. Estimator no longer warns on float64 with x64 off (same program). Every
  S13-tagged DEFERRED row disposed; plan-text amendments collected for the orchestrator.
