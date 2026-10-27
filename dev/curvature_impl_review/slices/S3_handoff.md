# S3 handoff (implementer) -- DONE + review A1/NITs folded in, tree UNCOMMITTED (base main @ 930686a)

## Files (7)
- t3toolbox/backend/sampling_derivatives.py: section "Corewise two-hole" -- t3_probe_corewise_derivatives_hessian
  [_from_sweep]; compute_deta_tilde_tangent_jets, compute_tau/sigma_tilde_tangent_jets (module-level
  _adj_tilde_tangent_step + _adj_tangent_sweep_scanned_step_{uniform,ragged}), compute_dxi_tilde_tangent_jets,
  assemble_{tucker,tt}_tangent_variation_jets (+ _tucker/_tt_tangent_chunk_summed via _wchunked_sum); 8 names in
  __all__. Estimators: _two_hole_jet_floor_bytes, _two_hole_assembly_per_row_bytes, and (review A1) the solver's
  mode vocabulary `hessian='gauss_newton'|'exact'` on estimate_chunk_size / max_chunk_size_within (lower-cased;
  `_check_hessian_mode` raises ValueError on anything else, a bool included; default answers unchanged).
  One seeded doctest (FD + symmetry).
- tests/backend/_curvature_oracles.py: dense-trs twins (adjoint_sweep_tangent_trs, tau/sigma/deta/dxi_tilde_
  tangent_jets_trs, tucker/tt_tangent_assembly_trs [9-term], probe_derivatives_two_hole[_states]_trs).
- tests/backend/test_curvature.py: TestProbeDerivativesTwoHole (8 tests: block-sum; FD; == twins whole + 6
  pieces at 1e-14 [observed <= 4.5e-16; 1e-15 had ~2x margin across numpy generations]; sweep reuse bitwise;
  stacked == per element; uniform == ragged dense/chunked; T6; T11). tests/test_contractions_interpreter.py +51
  rows (143 -> 194); test_dispatch.py test_jit_probe_derivatives_two_hole; test_chunk_size_estimator.py
  test_hessian_also_bounds_the_two_hole + test_hessian_mode_is_the_solver_vocabulary (rejection cases).
- CHANGELOG.md: one Added bullet (kernel, pieces, the `hessian=` mode on the estimators).

## Measurements (numpy CPU; scratch impl/s3/{repro_s3,cost,memory,est}.py)
- Through the NEW kernel: c22 vs block-sum <= 4.4e-16, vs c22 prototype <= 3.1e-16, one-shot == from_sweep(model
  sweep) bitwise; c14 / check_c14_extra / check_c14_d1 <= 4.1e-16; c35d sizes == prototype 1.3e-15 / 2.0e-15.
- exact/GN: d5 r8 W300 o2 2.34-2.67x (reviewer 2.81/3.09x); d4 r16 W2000 o3 2.26x (reviewer 2.15/2.21x) -> gate
  <= 3.5x PASS. nu re-sweep 2.0-2.9% of the two-hole.
- First cut failed (3.8-4.0x): grouped 3-operand pair strings get a degenerate numpy pairwise path (W, C shared ->
  (o+1)*2*W*r*n outer product first). Fix: binomial weights first (M = trs.mu, N = trs.tau~): TT assembly
  2.59 -> 0.4 s. Trade on jax CPU (reviewer): the two-step pairs cost +25% time on that term (0.214 vs 0.172 s,
  d4 r16 o3 W2000) for -9% per-row scratch -- read S8's jax timings with that in mind.
- Memory (XLA temp bytes, uniform+jax f32): per-W-row assembly two-hole/GN 2.50x (d5 r8 o2), 3.31x (d4 r16 o3),
  2.0-3.4x (d4 r32), 3.48x (K=3) -> over the 1.5x gate -> the plan's remedy, `hessian='exact'` (balanced: min of
  the two balanced chunks; budget: the larger per-row). After it (reviewer, for Nick / verification_records): the
  two-hole's chunked assembly peaks at 1.48 / 1.56 / 1.61 / 1.25x the GN assembly at its default chunk -- each
  held at its OWN jet floor (the two-hole's floor is 1.1-1.6x the GN's), so near, not strictly under, 1.5x;
  a strict 1.5x would balance against the GN floor instead (one line). Recorded, not changed.

## Tests / gates
Full suite (pre-fix) halves 354 + 609 passed, 1 skipped, 44,703 subtests. After the fix: test_curvature +
estimator 42 passed / 447 subtests; interpreter 15 / 1772; module doctest; docs/chunking.md page. Mutation: 64
single-term drops/scales -> 64/64 RED. Slow tier: run 1 failed the known-flaky O2 grad_fd (1.86e-6, GN path),
run 2 green. NOT run: sphinx -W (not installed).

## Conflicts / judgment calls
1. Kernel POLYMORPHIC (d-prefixed uniform branches) in S3, not S8 (C.10: S8 "delegates to the polymorphic
   kernel"; the chunked dispatch row needs it). S8's wrapper = mask theta + theta_hat, pack, call, re-mask.
2. Assembly strings differ from plan D (two-step pairs); call sites pass len_W, so rows record 'W' where D says 'C'.
3. TT finals 4, not 6: the tau~ and deta~ groups share nu. 4. Adjoint step fused by shared factor (prop: 2 GEMMs;
   src: one inner scan); n_frame explicit. 5. from_sweep reads only frame_sweep[0], [1].
6. Later slices: S4/S6 pass the solver's `hessian` straight to estimate_chunk_size in the chunk_size='auto'
   resolver; docs/chunking.md paragraph (S13). Backlog (reviewer O1, pre-existing, GN too): _wchunked_sum's
   chunk 0 + remainder sit outside the scan, so a near-full remainder peaks ~1.8-2x the balanced estimate.

## Draft commit message
backend: the probe-derivative two-hole (S3) -- sampling_derivatives.t3_probe_corewise_derivatives_hessian[_from_sweep]
(K_z theta_hat = d/ds grad <z~, S(Phi(theta + s theta_hat))>, the derivative of the corewise transpose) in the GN
kernels' forms: compute_deta_tilde_tangent_jets, compute_{tau,sigma}_tilde_tangent_jets (_adj_tilde_tangent_step:
fused reverse recurrence + inner order scan), compute_dxi_tilde_tangent_jets, assemble_{tucker,tt}_tangent_
variation_jets (product-rule factored, W-chunked); estimate_chunk_size / max_chunk_size_within hessian='exact'
  Body: Slice S3 (C.4; F13, F33). Polymorphic ragged/uniform. T5 (block-sum, FD, == dense-trs twins), T6, T11;
  51 HISTORICAL rows; dispatch. exact/GN 2.2-3.1x; scratch 2-3.5x GN's -> hessian='exact'. Suite 963 passed.
