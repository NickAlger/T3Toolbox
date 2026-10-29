# S0d handoff — zero reference (g0_reference), zero-point frame, stats keys. Tree UNCOMMITTED.

**Files.** `backend/fv_operations.py` (ZERO_FRAME_SEED, _identity_prefix, _haar_rows, fv_zero_point_frame);
`backend/ufv_operations.py` (ufv_zero_point_frame; new edges → fv_operations, ufv_conversions, acyclic);
`backend/geometry.py` (zero_point_frame ×4, corewise → None); `backend/optimizers.py` (Protocol doc: optional
member; _check_g0_reference / _check_positive_reference / _zero_reference_gnorm / _resolve_g0_reference;
`g0_reference` on gradient_descent + newton_cg; NewtonInfo += g0_reference, cg_reference, step_control,
stop_reason; stats keys); `backend/optimizer_display.py` (converged line names the reference + doctest);
`optimizers.py` (explicit `g0_reference` on gradient_descent/newton_cg; **kwargs untouched). Tests: NEW
`tests/backend/test_zero_reference.py` (T26, 17 tests); `test_optimizers.py` (:238 flipped; :262 re-pinned to
'warm_start' + a 'zero' row + non-positive raises; T29 `test_newton_cg_stats_contract`); `test_optimizer_display.py`;
`test_dispatch.py::test_zero_point_frame_dispatch`; `test_sharing.py` 14h. Examples: 4× `stats['newton']` →
`newton_steps` (hand-rolled dict keys too); shared example drops manual g0 tracking. Docs: fitting_and_optimization
(§2.1 stats table, §3 optional member, §4.11 note, §5 rewrite + doctest), rank_continuation, sharing,
numerical_contracts (caveat table), contributor/sharing_internals; CHANGELOG [2026.3.0] (4 breaking, 4 added).

**Reproduced through the new code** (`impl/checks/repro_c41_c38_c29.py` → `impl/logs/repro.txt`). c41: (1) True/True,
(2) nesting 0.0, library frame == the plan recipe written out independently (0.0), (4) 4.740e+01, (5) 44 O2 pairs
ref 3.4e-16 / losses 3.5e-12 — exact; (3) non-decreasing on all 4 ladders (digits vary run to run for c41
itself: its targets use the unseeded global rng). c38: trailing data LAPACK 0 / prefix 0 / library 1.867e1
(c38's v5-draft recipe: 1.596e1); generic 7.662e1 vs LAPACK 6.985e1; ragged vs uniform 1.2e-16. c29 (GN):
warm 39/15 steps, c29's LAPACK zero ref as override 36/13 (= F26), canonical default 37/13 (gtol 1e-8/1e-3);
corewise zero ref 0.0 → 'warm_start'; g0norm_newton=0.0 and explicit 'zero' on corewise → ValueError.

**Gates.** Full suite `-n 2` (two foreground halves, 10-min tool cap): 921 passed, 1 skipped (slow gate),
43,247 subtests. Slow tier `T3TOOLBOX_SLOW_TESTS=1 test_oracle_sweep.py`: 5 passed / 12,790 subtests;
tests/oracle_sweeps/ untouched. Module doctests 196 passed; all docs/*.md + contributor/*.md + getting_started:
pass; sphinx -W (scratch venv) clean; examples 13/14 (optax not installed locally). Post-suite: a vacuous
assert removed from test_zero_reference.py, a doc figure fixed (~1.6× → ~1.7× measured); both re-run green.

**Judgment calls / conflicts.**
1. ufv_zero_point_frame packs via t3frame_to_ut3frame (plan), returns the TEMPLATE's mask objects after checking
   they equal the packed prefix masks (else ValueError); takes `seed=` to mirror the ragged signature.
2. fv_zero_point_frame raises ValueError for a group with unequal (nU, N) or shapes admitting no orthonormal frame.
3. NewtonInfo: only this slice's 4 fields. stop_reason on the LAST row only (None before).
4. Explicit 'zero' + g0norm_newton: zpf called to validate, nothing computed, recorded 'override'; None +
   g0norm_newton: zpf not called (the T26(f) counter is 0).
5. gradient_descent gets only g0_reference (no g0norm_*); its stats unchanged (resolution tested via counter /
   warning / raise and the U7a pairing).
6. Beyond the H row: §G's "drop manual level-0 g0 tracking" in fit_shared_factors_jetted_probes.py and
   test_sharing 14h (the new docs would contradict them). Test reaches 3.1e-10 at true ranks; example's best
   level unchanged, 35% (was 30%) lower true error, inside its documented ~25-35%.
7. Left for S13 (§G): contributor/curvature_internals.md (recipe + 3 properties are in the fv_zero_point_frame
   docstring), deferred_and_rejected alternatives, release_notes "Upgrading", numerical_contract_catalog, CLAUDE.md.

**Draft commit message.**
optimizers: the zero reference (S0d) -- g0_reference on newton_cg + gradient_descent (both layers): None = the
geometry's own ('zero' on the manifold geometries; 'warm_start' on corewise and on a custom geometry without the
optional zero_point_frame), evaluated at the canonical zero-point frame fv_/ufv_zero_point_frame (seeded keyed
Haar Tucker factors, identity-prefix TT cores: rank-independent, nested, layer-independent); overrides win;
non-positive overrides and a vanishing explicit 'zero' raise, a vanishing default warns and falls back;
stats['newton'] -> newton_steps / newton_accepted / stop_reason; NewtonInfo g0_reference / cg_reference /
step_control / stop_reason; converged line names its reference; T26, T29; docs §5 / rank_continuation /
sharing / numerical_contracts; CHANGELOG [2026.3.0]
