# S3b handoff (implementer) -- DONE + reviews M1/M2/N3/N4/M5 + M6 folded in; tree UNCOMMITTED (base a62e621; S4 stash untouched)
Files: t3toolbox/backend/sampling_derivatives.py; tests/_jet_trs_twins.py (NEW, GN dense twins + binomial_combine_tensor);
tests/test_jet_recurrence.py (rewritten); tests/backend/{_curvature_oracles,test_curvature}.py; tests/test_{chunk_size_estimator,
contractions_interpreter,contractions_sharding,dispatch}.py; docs/{chunking,naming_conventions}.md; docs/contributor/deferred_and_rejected.md;
CHANGELOG.md. Scratch evidence: scratchpad/s3b/ (measure.py + before/after{1,2}.json, profile_twohole.py, jax_uniform_time.py, mutate.py, gradfd.py).
**Method:** affine factor -> banded pair [x, t x^(t-1)] / [x, (s+1) x^(s+1)] (`_shift_down_pair`/`_shift_up_pair`), one GEMM. Full conv/corr ->
order scan over r with the binomial weights from a CARRIED shifted jet (D^r b = r!C(t,r) b^(t-r)); xs = (jet, 1/r!). No weight matrix, no
dynamic slice, no moveaxis/reshape of W/K/C (uniform: xmap over cores, so the order axis leads). Inner bodies module-level / lru_cache factories.
## Sites (survey lines @232b1b0 -> now). Exceptions: NONE -- no trs or trs slice anywhere in t3toolbox/ (0 'trs,' strings).
| 1197 apply t3 fwd -> `_mu_jets_step` | 761-3 (via 1251) apply tv fwd -> `_sigma_jets_step`; `_sigma_jet_step` moved to twins |
| 2334 sigma^ -> `_adj_prop` (shared w/ `_adj_tilde_step`) | 2407/2425 dxi^ -> `_dxi_hat_order_step` scan, 2 rows |
| 2386/2428 apply asm -> banded pair + 1 final | 1637-8/1659-60 dxi~ -> `_dxi_tilde_order_step` scan, 2 rows |
| 1779-81/1829-31 TT asm (all branches; per-chunk kernel) -> `_tt_variation_jets_step` (sig~+tau~ pairs, 1 final) + `_tt_det_lag_step` |
| F7 505/910/1565 -> CONVERTED (carried D^r nu, D^r tau, U^r deta~; fewer ops than the weight slice) | S3 2759 -> `_adj_tangent_src_order_step` |
| S3 2865-70/2899-904 dotted dxi~ -> `_dxi_tilde_tangent_order_step`, 2 rows | S3 2998-3001/3050-3 M/N -> banded pairs + `_tt_tangent_det_lag_step` |
## Signature / shape-contract changes (all consumers in T3Toolbox updated; T3Polynomial uses only dense_probe_derivatives)
- `trs` -> `order`: compute_mu_jets, compute_nu_jets, private _apply_derivatives_t3_from_xi_jets.
- `trs` dropped: compute_{eta,sigma,tau,deta}_jets, compute_{tau,sigma}_tilde_jets, compute_dxi_tilde_jets, compute_sigma_hat_jets,
  assemble_tt_variation_jets, compute_{tau,sigma}_tilde_tangent_jets, compute_dxi_tilde_tangent_jets, assemble_tt_tangent_variation_jets;
  private: _apply_derivatives_from_jets, _apply_derivatives_transpose_from_jets (also n_probe: read off w_jets), _adj_tilde_step /
  _adj_tilde_tangent_step (trs_r), _adj_sweep_scanned, _adj_tangent_sweep_scanned, _sigma_banded_step (s_size,tvec,order),
  _assemble_tt_tangent_variation_jets_dense, map/scan data tuples (trs_r dropped).
- SHAPE: compute_dxi_tilde_jets, compute_dxi_tilde_tangent_jets, private dxi^ -> (2,)+W+K+C+(nOi,) (row 1 = 0 at order 0); consumers
  assemble_tucker_variation_jets / assemble_tucker_tangent_variation_jets (+ dense kernels) take 2 rows (code unchanged: [:s_size]).
- REMOVED public: binomial_combine_tensor, compute_{mu,nu,eta,sigma,tau,deta}_jets_trs, compute_{tau,sigma}_tilde_jets_trs,
  assemble_tt_variation_jets_trs (-> tests/_jet_trs_twins.py); assemble_tucker_variation_jets_trs -> private _assemble_tucker_variation_jets_dense.
- Estimator (rev. M2): per-row = d(temp)/d(chunk) of the COMPILED chunked transpose / two-hole (`_chunk_slope`, chunks 128/256 at W=1024),
  not an isolated kernel (that missed the chunk's jet slices: under 1.2-1.5x at small ranks); floors count dxi~ as 2 rows. CHANGELOG has it.
## Measurements (two runs; numpy ragged CPU unless noted) -- band before -> band after
- c35d d5 r8 o2 W300 (CORRECTED, min-of-25 x2; v1's min-of-3 was noise): GN T 23-27 -> 17-20 ms (FASTER); two-hole 63-77 -> 65-80 ms;
  exact/GN 2.3-2.8 -> 2.8-3.2 (rose here too). (Reviewer: 23-25 -> 15-20, exact/GN 2.4-2.7 -> 3.3.)
- c35d d4 r16 o3 W2000: GN T 0.99-1.01 -> 0.41-0.44 (2.3x); TT asm 0.753 -> 0.123-0.130 (6x); apply T 0.25-0.29 -> 0.12-0.15; two-hole
  1.65-1.72 -> 1.48-1.71; **exact/GN 2.22-2.29 -> 2.93-3.48 (ROSE: GN got 2.3x faster; the two-hole's dominant pieces -- dotted adjoint
  sweeps 0.45 s, banded forward 0.24 s -- were recurrences already).** dxi~ GN 0.067 -> 0.092 s (per-order Python ops; memory 14x less).
- jax uniform jit CPU f32, d4 r16 o3 W2000 unchunked: GN T 0.127 -> 0.118 s, temp 153 -> 35 MB; two-hole 0.50 -> 0.31 s, 443 -> 69 MB
  (two-hole/GN-T 3.9 -> 2.6); cs=100: GN T 0.113 -> 0.144 s. d5 r8 W300: GN T 6.2 -> 9.7 ms, two-hole 18 -> 30 ms (sequential-scan latency).
- memory_analysis per W-row (uniform f32): TT asm 50688 -> 6592 B (7.7x, r16 o3), 99840 -> 24704 (4.0x, r32), 153088 -> 20544 (7.5x, o5 K3);
  dxi~ 12.9-14.7x; two-hole TT asm 5.6-13.6x (two-hole/GN per-row now 2.2-2.8x, was 2.5-3.5x).
- Estimator (in situ, rev.): (128^6, o5) per-row 6.32 MB -> 86 KB (two-hole 184 KB); 16 MiB budget 2 -> 195 (exact 1 -> 91); balanced
  now = W in every measured case (in-situ per-row < the jet floor per row): W32000 809 -> 32000; doc case 219 -> 512; (24^4,r16,o3) 393 -> 2000.
- c22/c35d BREAK (sd.binomial_combine_tensor, trs args, `_adj_tilde_step(..., trs_r, ...)`); via a shim (s3b/run_dev_check_shimmed.py) both pass.
## Tests
- vs twins (1e-14 RELATIVE, not bitwise): observed GN fwd 1.1e-15, GN transpose 7.9e-16, curvature twins 4.4e-16; orders 0-4, d in {1,3,4}
  (+ slack), W/K/C empty/single/multi-axis; uniform == ragged; chunked == twin. Mutation: 15/15 single-term drops/shifts RED.
- Interpreter HISTORICAL +22 rows, hand-derived, solver agrees ('' = Caib,WCb->WCai; Caib,WKCb->WKCai; krWCb,rWCa->kWCab; ksWCa,sWCb->kWCab;
  sWCi,sWCb->WCib | C = KCaib,WCb->WKCai and the 5 u-row dxi strings | W = the 12 K-carrying k-pair / lag-fold strings); none removed.
  Sharding: its sweep is AST-driven (no table) -- all new strings swept, 708 subtests, 0 all-gathers; + letter-coverage test.
- NEW shard_map smoke (4 virtual CPUs, x64): probe transpose + two-hole, W sharded, psum, chunk_size None/2: == unsharded <= 1e-14, HLO
  0 all-gather, all-reduce present. Rev. M1: `jax.shard_map(check_vma=False)` if present, else `jax.experimental.shard_map(check_rep=False)`
  (the 0.4.30 floor; legacy branch exercised here by hiding jax.shard_map: passes; a lowering error there -> SkipTest naming the version).
  check off is PRE-EXISTING need (HEAD fails the vma check identically); docs/chunking.md recipe flagged for S13, not changed.
- NEW in-situ budget test (rev. M2): max_chunk_size_within(target=2 MiB) at (3,4,8,64,3); the compiled one-shot transpose, transpose from
  sweep, and (exact) two-hole: cs*slope <= target. Fails against the v1 estimator (2.42 / 3.20 MB > 2.10 MB); passes now.
- M6 (dtype leak, pre-existing since 2026.2.0): every jet scan init (`_init_jet`/`_zero_jet` now take a dtype; adjoint-sweep, inner-scan,
  sigma^ seed, entries one-hots) uses the operands' result_type -> float32 stays float32 under x64 and on numpy. New tests: estimator
  per-row identical x64 off/on (bypassing the cache); test_dispatch TestJetDtypeUnderX64 (7 ops, uniform eager/jit + numpy + ragged).
- Full suite after M6 (two halves, -n 2, test_dispatch first): 406 + 559 passed, 1 skipped, 5,268 + 38,966 subtests. Module doctests 207; doc pages; quickstart;
  sphinx -W green. Slow tier (pre-M6): O2 grad_fd flake under -n 2; serial 5/5 green; grad_fd identical base vs new over 40 seeds.
## Judgment calls / conflicts
1. "Beside the existing *_trs twins in the test helpers": none existed -- the old twins were PUBLIC package functions. Read I-33 + "never
   imported by the package" + Nick's "only build binomial_combine_tensor in the test twins": ALL dense twins (old + new) moved to tests;
   11 public names removed (breaking, recorded). Twins are ragged-only (uniform d-prefixed dense branches dropped).
2. F7 sites converted, not excepted. 3. dxi~ always (2,) rows. 4. Uniform: xmap over cores + inner scan replaces the d-vectorized einsums
   (memory, no axis moves; small-jit latency, ledgered). 5. HISTORICAL kept as frozen history (no removals).
6. Existing strings reused where operands matched. 7. Curvature: the dotted-dxi dense twin already existed (same 6 strings); only the factored
   M/N assembly moved (`tt_tangent_assembly_factored_trs`, also pinned). S3's unreleased "~2-3.5x" bullet -> "~2-3x".
8. CHANGELOG 2026.1.0: one italic "(see 2026.3.0)" line (no house precedent either way). 9. chunking.md: (order+1)^2 r^2 memory text
   corrected; frame-stack doctest budget 2**24 -> 2**22 (free chunk otherwise capped at |W|). 10. Estimator tests re-pinned: budget near-
   maximal unless W fits; frame stack ~prod(C) (0.6-1.05x; in situ 3.4x for C=4 -- ww/pp carry no C; chunking.md doctest likewise
   `2.5 < free/stacked <= 4`); 'exact' balanced <= plain (both clamp at W); ratio comment -> 2.0x. Ledger: small-jit latency line (M5).
11. Follow-ups: chunking.md shard_map recipe (check_vma, per-layout in_specs; S13); tex sec:recurrence; c22/c35d API; S4 un-park re-pins test_resolver (M3).
## Draft commit message
backend: S3b -- every production jet kernel a recurrence / order scan, no binomial (trs) contraction anywhere (I-33)
Body: the 19 pre-existing GN dense-trs sites (apply/entries-derivative fwd + transpose, the probe transpose's dxi~ combine, the TT
assembly's per-chunk kernel on every branch), the three order-scan weight sites, and S3's dotted twins become banded two-term
recurrences (_shift_down_pair/_shift_up_pair) and order scans whose binomial weights come from a carried shifted jet (D^r = r! C(t,r)
shift). dxi~ returns its two read rows. trs params dropped (mu/nu take order); binomial_combine_tensor + the *_trs twins move to
tests/_jet_trs_twins.py (never imported by the package). Estimator measures the compiled chunked programs. Equal to the twins <= 1.1e-15 rel; per-row
assembly scratch 4-8x (dxi~ 13-15x, two-hole 5-14x) smaller; numpy GN probe transpose 2.3x faster at d4 r16 o3 W2000. HISTORICAL +22;
shard_map smoke test (0 all-gather). Ledger, naming conventions, chunking note, CHANGELOG corrected.
