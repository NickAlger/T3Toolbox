# S10+S11 handoff (implementer) -- DONE; base ed2d9bf (main now 7acabc5: dev/-only commit landed meanwhile, no overlap); tree UNCOMMITTED (16 files). Scratch: impl/s10/
## Files
- backend/probing.py: t3_probe_corewise_hessian[_from_sweep] + public pieces compute_deta_tilde_tangent,
  compute_{tau,sigma}_tilde_tangent (scan body _tau_tilde_tangent_step = two _sigma_step calls), compute_dxi_tilde_tangent
  (compute_deta x2), assemble_tucker_tangent_variations, assemble_tt_tangent_variations (factored, 4 finals); nu
  re-swept through G; no corewise eta / dxi~. Polymorphic ragged/uniform; doctest (FD + symmetry).
- backend/apply.py: t3_apply_corewise_hessian[_from_sweep] + private _apply_corewise_hessian_adjoint (sigma_hat through
  G, compute_tau for its dot, compute_deta for dxi^dot, factored TT 2 finals); entries.py: t3_entries_corewise_hessian
  [_from_sweep] (_entry_xis / _onehot_vectors into the shared adjoint). Doctests.
- backend/fitting.py: ProbeKind (omega^2 r) / ApplyKind / EntriesKind.corewise_hessian on the model sweep.
- backend/ut3_sampling.py: ut3_{probe,apply,entries}_corewise_hessian[_from_sweep] (theta + theta_hat masked with
  theta's masks, output re-masked; _masked_point_and_directions). uniform_fitting.py: the 3 uniform overrides (masks
  from the FRAME via new _frame_theta; UniformProbeDerivativesKind refactored onto it, bit-identical); S10/S11 rows
  of _UNIFORM_HESSIAN_SLICE removed (stub + S12 rows stay).
- fitting.py / optimizers.py docstrings flipped; CHANGELOG [2026.3.0] Added bullet + 3 "only probe_derivatives" flips.
- tests: test_curvature +14 (TestPlainKindsTwoHole 9, TestPlainKindsExactLocalModel 5) + T23 flips (in_window;
  implemented/pending kinds; the plain uniform kinds build); _curvature_oracles +2 unfactored references;
  test_uniform_curvature +8 (TestUniformPlainKindsTwoHole 2, TestUniformPlainKindCorewiseHessian 3,
  TestUniformPlainKindsExactLocalModel 3; _kind_setup generalizes _exact_models/_shared_exact_models, S8/S9 draws
  unchanged); test_fitting flip + plain frontend==backend; test_dispatch +1 (27 s); test_optimizers_frontend +3;
  interpreter HISTORICAL +12 (5 ragged + 7 d-prefixed; hand supplements '' / 'C' / 'W' == solver).
## Numbers (ragged d=1,3,4 + Tucker/TT slack + raw unsquashed, K up to (2,3), C=(2,); impl/s10/measure.py)
block-sum probe 6.8e-16 / apply 6.1e-16 / entries 3.5e-16 (bar 1e-13); FD of the one-hole <= 3.9e-15 (bar 1e-12:
polynomial-exact Richardson, d step sizes); T6 Gram asym < 1e-12 rel; T7 retraction FD margins 6.9e-7 / 1.3e-6 (bars
1e-5 / 1e-3); T8 K_N L p_vert < 1e-11; T13 corewise == GN + block-sum 1e-12, FD 1e-8; factored TT == unfactored 1e-14.
Uniform per element vs ragged: two-hole 5.9e-16 / 3.6e-16 / 2.2e-16; exact hvp 2.1e-14 / 8.1e-15 / 8.9e-15 (bar
1e-13; quadratic forms 1e-12, observed <= 1.3e-13); exact masks; finite garbage (theta/frame incl. the corewise
boundary-bond slots, theta_hat, residual, vectors) BITWISE; shared-uniform subcase per kind (tied, per-mode lift off).
Cost (numpy CPU): exact/GN hvp probe 2.8-2.9x, apply 3.1-3.2x, entries 3.1-3.3x; two-hole/GN transpose 3.4-3.7x.
## F27 / E2 apply restarts through the LIBRARY (impl/s10/e2_apply.py, e2_gn_tr.py; outer/rejected/CG)
Library exact+TR == the critic harness (block-sum oracle + its own TR) in EVERY row, counts identical.
A above target, gtol 1e-8: GN+Armijo 80 max_newton (warm and zero); exact p=1.0 18/1/117 (warm) 20/4/130 (zero),
p=0.5 23/3/106 / 20/4/105; gtol 1e-3: GN zero 31 vs warm 80; exact 17/1/92 (warm), 17/4/67 (zero) -> F27's 17-23.
B noise-free: zero ref -> 0 steps; warm -> 4/1/12 (p=1) 3/1/5 (p=0.5), stop objective_floor (harness "unconverged").
C below target, 1e-8: GN 32/30 vs exact 11/1/53 (warm) 10/1/47 (zero). warm/zero 1.9e-2, 1.6e-14, 3.1e-1 == F27.
E2 base (60-step level 1): exact eps 0 and 1e-6 12/1/57 (p=1), 14/1/57 (p=0.5) vs GN+Armijo 34/273, harness GN+TR
45/10 unconverged @60; E2's recorded 13/69/0 vs GN 54 were on the 920d776 restart (it differs today).
check_restart_tr*.py, c29b, c29c BREAK as-is on st['newton'] (S0d rename); c5 / c6 run unchanged (2.3e-16..7.5e-16).
## Bit-identity vs ed2d9bf (git archive, separate processes; impl/s10/bitident.py): 524 arrays, 0 differ -- the 3
kinds' corewise transposes (ragged + ut3), GN grad/hvp/newton_cg on all 4 geometries + frontend, probe_derivatives
EXACT hvp/newton_cg ragged + uniform. Mutants (impl/s10/mutate.py, tree copies): 14/14 RED.
## Gates
targeted 13 files -n 2: 380 passed, 2 skipped / 6040 sub; module doctests 222 (S9: 218); doc pages + getting_started
OK; sphinx -W exit 0 (a split-literal autosummary warning fixed); floor leg (py3.11/np1.26/jax0.4.30) 197 passed,
2 skipped / 1225 sub + chunking.md OK (run 1 caught a T7 FD draw at sigma_min 2e-4, fixed: per-case seeds + a
sigma_min > 1e-3 guard); full -n 2: dispatch+backend 386 passed, 2 skipped / 2615 sub; rest 752 passed, 1 skipped /
43048 sub (1 warning = pre-existing adam); slow tier SERIAL (oracle sweep + uniform curvature): 56 passed / 13087 sub (3 warnings: pre-existing jax f64, S9 n4). `git diff --check` clean.
## Conflicts / judgment calls
1. TT assemblies FACTORED (plan_v4 C.5: "three-term"/existing strings; orchestrator: mirror S3's factored form):
   probe mu.(xi(x)s~ + deta~(x)nu) + tau~.(xi(x)nu) = 4 finals vs 9; apply/entries mu_dot.E + mu.E_dot = 2 vs 3. Finals
   reuse S3b's strings; new: 5 ragged + 7 d-prefixed (plan listed 2). Pinned to the unfactored forms at 1e-14.
2. No `_trs` twins: no order axis, no binomial structure (T5's twin clause vacuous).
3. T11 VACUOUS for apply/entries: no weight field anywhere (apply_model has no weight=); the FD-minus-GN curvature
   check runs unweighted; probe uses its per-mode row (the orchestrator's "scalar/vector omega" does not exist).
4. chunk_size='auto' -> None for these kinds in both modes (estimator probe_derivatives-only; their transposes do not
   chunk); spy-tested. Sharding test: no new letters, its AST sweep covers the new strings (nothing to add).
5. T7 per kind: seeded per case + sigma_min > 1e-3 asserted (FD rounding-limited near a singular lift, E1).
## Draft commit message
backend+frontend: the exact Hessian for probe, apply and entries (S10 + S11) -- probing.t3_probe_corewise_hessian
[_from_sweep] (+ its pieces), apply.t3_apply_corewise_hessian[_from_sweep], entries.t3_entries_corewise_hessian
[_from_sweep], the ut3_* masked twins; ProbeKind / ApplyKind / EntriesKind and their Uniform* twins implement corewise_hessian
  Body: Slices S10/S11 (C.5; T5, T6, T11, T17, T23). Two-holes in the GN kernels' forms (two _sigma_step per dotted
  tilde step, compute_deta x2, nu re-swept through G; apply/entries differentiate the adjoint-state transpose), TT
  assemblies factored. Block-sum <= 6.8e-16, exact FD <= 3.9e-15; uniform per element == ragged, garbage-invariant,
  masks from the frame; F27's apply restarts reproduce (library == critic harness). GN bit-identical to ed2d9bf.
