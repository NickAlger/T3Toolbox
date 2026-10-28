# S12 review (independent) -- apply / entries derivative two-holes, ragged + uniform; the window stub deleted
Base f5c0e1b (main 4c908d8, dev/-only); tree UNCOMMITTED (the 15 files + my 2 test-only fixes). Scripts: review/s12/.
**Verdict: APPROVE-WITH-FIXES** (two cosmetic test-file fixes, applied; no blocking or major finding).

## Independent checks (mine; none reuse the implementer's oracles, twins or harnesses)
- **Math** (`oracle.py`): own dense A = sum_w sum_t c^(t) t! sum_{|S|=t} (x)_j(p_j | w_j) (entries: one-hot w), two-hole
  block k = sum_{j!=k} d/dX_k <A, Phi(X_j <- th^_j)>, boundary bonds summed; einsum only, no FD, no library kernel. d = 1,
  3, 4, Tucker + TT slack frames, K = (2,3), C = (2,), orders 0-3, both kinds, 3 seeds: **max 9.7e-16**; raw unsquashed
  tails (r0, rd != 1: the seed broadcast) 9.0e-16 (`check2.py`).
- `check2.py`: T6 Gram asym (K = 5) 5.5e-16. **T11**: the KIND op at a MANIFOLD frame on the model's sweep, per-ORDER
  omega, == oracle(omega^2 r) 6.2e-16. **entries == apply at one-hots BITWISE.** **apply == probe_derivatives** at MY
  derivation z~_i^(t) = a_i(c^(t) w_i + (t+1) c^(t+1) p_i), sum a = 1 (from y(s) = <w_i + s p_i, Y_i(s)> for each i):
  3.4e-16. **Order 0 vs S11's plain two-hole** 2.6e-16 / 3.4e-16, bitwise at d = 1: acceptable -- the jet kernels' order-0
  branches associate differently (P.xi and the banded pair formed first); at d = 1 there is nothing to reassociate.
- **I-33**: no trs / binomial in the diff; new bodies module-level, closure-free; all 18 contract strings are HISTORICAL
  rows already used at base. dtype (`dtype32.py`): f32 in -> f32 out, numpy + jax, x64 off AND on, ragged + uniform.
- **Mutants** (`mutate.py`): theta_hat mask removed x2, self.masks for the frame's x2, no re-mask, 5 dropped product-rule
  terms, a wrong entries xi_dot, omega dropped / not squared, the uniform override deleted: **14/14 RED**.
- **Bit-identity vs 4c908d8** (`bitident.py`, git archive, separate processes): **1022 arrays, 0 differ** -- both kinds' GN
  derivative transposes (orders 0-3), ragged + uniform + jit f32; the probe_d two-hole (`_adj_prop_tangent`) orders 0-3,
  dense + chunked + jit; grad/hvp/forms/3-step newton_cg of 6 kinds x 5 geometries x 2 modes (196 new = the S12 exact rows).
- **T17** (`uniform_check.py`): varying-rank C = (2,) stack with a TT-slack element, forced padding: real blocks vs MY oracle
  8.0e-16, padding exactly 0, 1e3 garbage in theta (both boundary-bond slots incl.), theta_hat, vectors beyond N_i: bitwise.
- **F28** (`rerate_review.py`, c42's problem, library newton_cg, plain apply): 20/4/78 -> 2.6e-9; 34/9/272 -> 2.2e-12; slack ==
  minimal; eps 0 == 1e-6; cg cap 80 == 200: **exact**. apply_derivatives o1 (`t25_ad.py`): 17/4/56, 31/10/240 (handoff's).
- **F22/F29**: with the PRE-S0d warm-up (`g0_reference='warm_start'`) power-1.0 max e1/e0^2 = **0.917** (F22 <= 0.92); the
  default zero-reference warm-up moves x15 -> 0.941. Warm-up drift (DEFERRED "S5 review (drift)"), not the kernels (plain
  apply's exact path is bit-identical to base). F29 0.43-0.98 in [1e-6, 1e-1] reproduced; its 1.53 is at |pred|/res 3e-7.
- **apply_derivatives seed 2** (`seed2.py`): rho(H_GN^-1 K) = 1.03 / 1.10 (> 1: no GN linear rate) and the exact Hessian's
  min eigenvalue on range(Pi) is NEGATIVE (-12 / -21) at the warm start: indefinite, pre-asymptotic. Exact <p,Hp> ==
  Richardson FD of f(R(tp)) to 5e-7..3e-4 of the curvature term (random + Newton directions); GN misses by 100 %. Not a bug.
- **Memory** (`memory.py`, jax f32 uniform unchunked, 3 sizes): two-hole/GN 1.85-2.08 (probe_d pair, dense: 1.94-1.95).
  **Quadratic bar** (`formtol_review.py`, numpy 2.5 + 1.26): |q| cancels to 1.2e-3 |p||Hp|; err/(|p||Hp|) <= 1.1e-14.

## Rulings on the judgment calls
1. **'auto' -> None for these kinds, both modes: ACCEPT (no MAJOR).** `_resolve_chunk_size` (optimizers.py:164) returns
   None for every kind but probe_derivatives at base too; `ApplyDerivativesKind` has no chunk field, the GN transposes no
   chunk parameter, so GN never chunks them: exact unchunked is no regression. Its ~2x-GN peak is the probe pair's dense ratio.
2. **Twins: ACCEPT, coherent.** GN twins in tests/_jet_trs_twins.py (S3b: top level, the common parent of its importers);
   every two-hole (dotted) twin in tests/backend/_curvature_oracles.py, built on them (S3's and S12's). No consolidation.
3. **`_adj_prop_tangent` / `_entries_ambient_jets`: ACCEPT** -- verbatim extractions, bit-identical (above).
4. **`_UniformLayerMixin`: ACCEPT** -- f5c0e1b's `_UniformKind` minus the stub (diffed); all 6 uniform kinds override on
   their own class through `_frame_theta`; `test_no_uniform_stub_remains` replaces the stub as the guard (kills the
   deleted-override mutant). The unused `order` of `_from_sweep`: accept (parity). 5. **One public piece: ACCEPT** (GN parity).
6. **Bar 1e-12|q| -> 1e-13|p||Hp|: ACCEPT, principled, not a loosening.** |<p,H_u p> - <p,H_r p>| <= |p| |H_u p - H_r p|
   (Cauchy-Schwarz), so it is exactly what the 1e-13 vector bar implies; relative-to-|q| is wrong under cancellation; 9x margin
   on 1.26; tighter than the old bar wherever |q| > 0.1 |p||Hp|.
- **T16 on apply_derivatives**: no such test exists (the numbers are measurements). If S13 adds one, DRAW the start by a
  predicate (exact Hessian PD on range(Pi) and rho(H_GN^-1 K) < 0.9; seed 2 fails it) -- never a loosened bound.

## Findings
- P3 FIXED tests/test_optimizers_frontend.py:817-819 (three blank lines before `_derivative_fit_problem` -> two) and :884
  (one before `if __name__` -> two) -- the same slip S10/S11's review fixed in this file.
- P3 FIXED tests/backend/test_curvature.py:2220: `_deriv_fit_away_from_singular` ignored its `W` (hard-coded `(40,)`); now
  passes `W` (the only caller passes (40,): behaviour identical).
- P3 info (`align4.py`): "garbage bitwise-invariant" holds for C-contiguous garbage; with non-C strides the output moves <= 4e-16
  though every masked kernel input is bitwise equal (numpy orders sums by stride) -- pre-existing, the GN transpose too. The
  suite's bitwise asserts pass (both numpy generations); testing_strategy.md could say "equal layouts".

## Plan-text amendments for S13
- F28: "reproduced EXACTLY by the library's exact newton_cg on plain apply (S12 review)"; add the apply_derivatives o1 rows.
- F22/F29: state the GN warm-up's reference (`g0_reference='warm_start'`; the zero default gives 0.941); re-pin, not loosen.
- F11: superseded -- c13 breaks on the S3b API; the S12 two-hole == an independent dense oracle at orders 0-3 (9.7e-16).
- §F: GN twins in tests/_jet_trs_twins.py, two-hole twins in tests/backend/_curvature_oracles.py. §C.5/§C.6: the window
  and its stub are history (deleted S12); `test_no_uniform_stub_remains` is T23's post-window form. §E: apply/entries(-
  derivative) exact ~ 2x the GN transpose peak, no W-chunked path (DEFERRED S10S11 handoff row extends to these kinds).
## S13 checklist (noticed)
- fitting_and_optimization.md §4.12: "every kind" + where 'auto' chunks (probe_derivatives only); no kind table exists yet.
- naming_conventions.md: `*_tangent[_jets]` dotted twins; `t3_<kind>_corewise[_derivatives]_hessian[_from_sweep]` / `ut3_*`.
- deferred_and_rejected.md: the no-chunk path for the apply/entries(-derivative) transposes + two-holes (cost above).
- dev/ scripts on the pre-S3b API (trs / `sd.binomial_combine_tensor`): c13 c14 c18 c22 c25 c26 c35{,b,c,d}; on
  `stats['newton']`: c24 c29{,b,c} c30 + critic check_restart_tr{,2,3} etc. -- banners or shims. OPEN_QUESTION: "sixth op".

## Gates (mine; post-fix unless noted)
- Targeted 11 files -n 2: 367 passed, 2 skipped / 5995 sub; dispatch 55 / 45 sub; 225 module doctests; quickstart + all doc
  pages OK; sphinx -W rc 0.
- **Floor leg** (py3.11 / numpy 1.26.4 / jax 0.4.30): 222 passed, 2 skipped / 1872 sub; docs/chunking.md doctest OK.
- **Full suite**: backend + dispatch 411 passed, 2 skipped / 3252 sub; the rest (two parts) 755 passed, 1 skipped / 43094 sub
  (warnings pre-existing: the resolver's float64 one is on base too). `git diff --check` clean; no dev/, CLAUDE.md, t3polynomial edits.
- **Slow tier, SERIAL**: 64 passed / 13173 sub; 3 pre-existing warnings (adam, jax float64 estimator, S9 n4).
