# S13 review (independent) — docs, CHANGELOG, release notes, T28, sweep_exact, jvp oracle, dev/ tidy, gate
Base a2fc053 + the implementer's uncommitted tree; reviewer fixes are in the tree (uncommitted), each listed below.
**Verdict: APPROVE-WITH-FIXES** — all fixes applied by the reviewer (1 MAJOR user-doc statement, 8 MINOR, 2 NIT). No
code defect; the four code changes are behaviour-neutral (verified). Every gate green on both pythons (end).
## Findings (all FIXED in the tree unless marked)
- **MAJOR (user doc, misleading) — docs/fitting_and_optimization.md §4.12 "When it helps" (:535-544) + "Restarts need
  nothing special" (:596-602); rank_continuation.md:129-130.** The worse-basin / radius-limited risk was pinned on "a cold
  zero start" only, but the one measured case (S0b o01 L1: exact 1.108 vs GN 0.987, a verified PSD minimum) was from a
  **zero-padded restart**; beside "exact mode runs from the restart as from any other point" a user reads restarts as
  risk-free. Now: "far from a minimizer — a cold zero start, or the first steps after a zero-padded restart — ... a
  different local minimum (measured from a zero-padded restart: ~12 % above GN's)", cross-referenced from both places.
- MINOR — curvature_internals.md:284-287 stated the display's table-reuse rule backwards; it is the row AFTER a rejected
  trial that reuses, and only when `info.lm is` that row's model (optimizer_display.py:301-315).
- MINOR — §4.12 Cost (:590-593): "For `probe_derivatives`, 'auto' sizes the chunk" — only a UNIFORM fit chunks (ragged ->
  None, optimizers.py:164); "a little slower to compile" -> "about 1.5×" (records: ~15 s vs ~9.5 s).
- MINOR — §4.12 (:599-602): "runs from non-minimal frames without reducing them" — the FRONTEND reduces a ragged manifold
  x0 (optimizers.py:125-135, :313-314); scoped to the model / backend `newton_cg`.
- MINOR — §3 (:288-290) listed `.quadratic` as exact-only; the backend `quadratic` exists in both modes (optimizers.py:303).
- MINOR — two-hole per-row ratio read as contradictory: verification_records.md:86 / curvature_internals.md:329 "2.2–2.8×"
  (the unchunked TT-assembly KERNEL) vs CHANGELOG:284 + docstring "~1.3–2.2×" (the running chunked program; I measure
  2.0× at (8⁴, r6, o3)). Both now say which.
- MINOR — §4.12 "a battery of shared and unshared restarts" overstated (one template, 9+3 seeds) -> "a seeded set ... of
  one such problem".
- MINOR — pointers (item 4) are NOT robust: deferred_and_rejected.md:41 (live plan) and :211, verification_records.md:29
  /:116 point into `dev/curvature_impl_review/`, which moves at the release -> HANDOFF checklist line to retarget them.
  dev/curvature_impl_review/README.md said "archive when S13 archives the plan" and "scripts kept runnable" — updated.
- MINOR — dev/HANDOFF.md: the floor leg had no command (added: full suite, targeted form, page doctests); "S14 blockers
  in the S0b report's §1" -> §1 call sites, §6 what S14 must change.
- NIT — testing_strategy.md "Two precisions" over three bullets; the new sweep paragraph split the checklist from
  "Numbers 2 and 3 ..." (moved). §4.12 κ called "share of the model's curvature" (it is the ratio to the GN part) —
  reworded. NOT fixed (pre-existing, S5 n2): `on_line_search_failure: str = 'stop',` breaks both signatures' type column.

## Truth of the docs (run; scripts review/s13/{spot,spot2,spot3,cold,ru,bitid,mutants,names}.py)
OK: defaults both layers (`hessian='gauss_newton'`, power 0.5, `step_control`/`tr_radius0`/`curvature_damping` None, no
`**kwargs`); step_control resolution ('EXACT' accepted; either with either); TR rows `ls_steps None`, `alpha 1.0`; Δ0 ==
‖g‖³/⟨g,H_GN g⟩ exactly; `lift_eps == 1e-6·‖X‖`, tracks `curvature_damping`; every no-effect option raises (damping with
GN / on all four corewise geometries incl. both shared; Δ0 with LS; accept+TR; Δ0 ≤ 0; g0norm ≤ 0; misspelt kw TypeError);
`g0_reference` 'zero' on the four manifold geometries, 'warm_start' + explicit-'zero'-raises on the four corewise; the
overrides; 'auto' -> None for the five kinds (both modes), int for uniform probe_derivatives; frontend `hessian(p)` vs
backend `hvp` 1.8e-16, curvature actions / quadratic / gradient bitwise; frontend exact run == backend run bitwise; d = 1
manifold model is GN (`geometry_data None`), corewise d = 1 curvature ≠ 0; a five-op kind runs GN and exact raises naming
`corewise_hessian`; `| stop: max_newton` on a GN last row (I-35); NewtonInfo fields default None; ragged == uniform exact
from a full-rank start (10/10 steps, CG equal, losses 1.7e-13); cold zero vs warm restart: same ‖g₀‖, exact converges
from both, ragged + uniform (I-37). Support matrix: entries/MANIFOLD/ragged (objective_floor, 7 vs GN 38 steps),
apply_derivatives/UNIFORM_COREWISE (gradient, 19), probe/shared-UNIFORM_MANIFOLD (objective_floor, 8; display as
documented), exact apply on all eight geometries. Every ≥2-underscore name in the touched pages / CHANGELOG resolves.
**CHANGELOG [2026.3.0]:** walked a62e621..a2fc053 + S0c–S0e: every behaviour change is present and accurate (I-35 stop
cell, `**kwargs` removal, `least_squares_problem(hessian=)`, sixth op (Changed, correctly non-breaking), padding hygiene,
jet dtype, compiler-dependent estimator, gauge check, the S13 warning fix); nothing non-breaking under breaking.
**curvature_internals.md:** every §G bullet present (I-33 wording, frame masks, Haar + 3 properties, tied lift / `s_g`,
padded-`s_g` mask, gauge scale, E1–E10 / F13–F35 — F20 omitted, superseded; E2 = one unseeded draw; F22's reference;
F28 exact); in the toctree. **Ledger:** §G Rejected + Deferred lists complete, the sector-norms paragraph is VERBATIM
(normalized diff: identical), I-36 item, no-chunk path, S9 clip, leftover-chunk, S7b follow-ups, S3b M7, S7 n1 all present.

## Tests
T28: constructs the four classes jax-hidden, applies `hessian(p)`, runs exact `use_jit=True` with the re-armed warning —
correct. sweep_exact.py: oracle INDEPENDENT (dense Φ by tensordot, exact multilinear DJ, DP = P⊥DJJ⁺ + (.)ᵀ — re-derived;
only `to_dense`/`randn`/`project`/`manifold_dim` from the library); per-case seeded; 720 = 72/geometry × 10, slow 8000;
~25 s always-on is fine. Mutants: GN-for-exact kills 144/144 ragged d = 3; per-mode-for-tied kills 36/72 ragged shared
(all d = 3, no d = 2 — square S_i by construction, as documented). jvp oracle: a real `jax.jvp` through the shipped
transposes, 7 ragged + uniform probe_derivatives (garbage-padded directions). Code changes neutral: GN `newton_cg`
bit-identical vs a2fc053 (ragged/uniform × manifold/corewise, shared ×2, uniform jit f32: cores sha256 + losses + CG);
estimator answers + both per-row byte measurements identical, float64 UserWarning gone. DEFERRED: agree with every S13 row.

## The 14 plan amendments — all APPROVE
1 (optimizers.py:302-314) ✓; 2 (test_contractions_sharding.py:64) ✓; 3 (display :301-315) ✓; 4 ✓ (run); 5 (backend
optimizers :1271-1355) ✓; 6 ✓; 7 ✓ (1.8e-16; fitting.py:568-598 dc.replace); 8 ✓; 9 ✓; 10 ✓ (imports); 11 ✓; 12 ✓ — add
"and every ragged fit" to 'auto' -> None; 13 ✓; 14 ✓ — also §F's jvp row (`tests/test_curvature_jvp_oracle.py` ->
`test_dispatch.TestCurvatureJvpOracle`), and the plan's "Nothing implemented yet" (:3) / "Nothing is implemented." (:17).
**Banner:** right (complete through S13; S15a, PAUSE; R / S14 / S15b, tag). Add: "`dev/curvature_impl_review/` is
archived with the plan at the release."

## CLAUDE.md patch — apply WITH EDITS (`git apply --check` OK)
Right: 2026.3.0 unreleased + order; the SERIAL slow-tier command incl. test_uniform_curvature.py; the floor leg; counts
(414+756 = 1,170; 3,270+43,094 = 46,364); the curvature_internals pointer; nothing true removed. Edits: (a) drop the new
bullet's "Open for Nick: ..." (ephemeral; HANDOFF has it) and cut its "Same release: ..." list to a CHANGELOG pointer
(changelog material); keep the durable rules (sixth op structural / exact-only; damping a solver option; θ̂ masked with
the FRAME's masks; I-33 recurrence-only kernels) + doc pointers; (b) "~10 min with `pytest -n 2` in two halves" is this
box's setup — say "on 2 cores" or keep the `-n auto` phrasing.

## Gates (reviewer, final tree) — ALL GREEN
- py3.13 / np 2.5.3 / jax 0.12: module doctests 225 passed; all docs/*.md + contributor/*.md pages and getting_started OK
  (pages twice); `sphinx -W -E` rc 0 (twice); fit_hilbert_uniform_newton_cg.py runs (17 s); jax_absent + oracle_sweep +
  dispatch 63 passed, 2 skipped / 12,851 sub; full suite 414 passed, 2 skipped / 3,270 sub + 756 passed, 2 skipped /
  43,094 sub (adam-on-manifold warning, by design); **slow tier SERIAL twice: 66 passed / 13,173 sub both** (no O2 flake).
- py3.11 / np 1.26.4 / jax 0.4.30: module doctests 225; pages + getting_started OK (twice); floor leg 231 passed, 4 skipped
  / 14,680 sub; full suite 414 / 3,270 + 756 / 43,094 sub (2+2 skipped; adam warnings).
- `git diff --check` clean; patch applies; plan + CLAUDE.md untouched; no reflow. Reviewer-touched: docs/{fitting_and_
  optimization,rank_continuation}.md, docs/contributor/{curvature_internals,testing_strategy,verification_records}.md, dev/
  HANDOFF.md, dev/curvature_impl_review/README.md.
