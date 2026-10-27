# Dense-`trs` contractions in `t3toolbox/` — read-only survey

Repo `/home/claude/t3toolbox`, `main` @ 232b1b0. All line numbers are at HEAD of
`t3toolbox/backend/sampling_derivatives.py` (the S4 working tree does not modify this file).
Nothing was modified and no tests were run. One scratch script was used to cost and check the forms:
`/tmp/claude-0/-home-claude/29728d3c-c433-5028-9de9-1af7ead3c123/scratchpad/impl/trs_cost.py`.

`trs[t,r,s] = C(t,r)·[r+s=t]` (`binomial_combine_tensor`, L250-265). Throughout, `o` = `order`.

---

## 0. Bottom line

1. **The ledger's claim is false at HEAD, and it was already false before the curvature work.**
   `deferred_and_rejected.md:85-96` says the dense forms survive only as `*_trs` reference twins.
   In fact **19 production 4-operand dense-`trs` contractions predate 920d776**, and several of them are
   the very "bidiagonal" cases the ledger says were converted:
   - the **apply/entries derivative forward** (L1197; L761-763 via L1251);
   - the **whole apply/entries derivative transpose** (L2334, L2386/2428, L2407/2425);
   - the **probe transpose's dx̃i combine** (L1637-1638 / L1659-1660);
   - the **TT gradient assembly** (L1779-1781 / L1829-1831).

   For the assembly, the "standard" `assemble_tt_variation_jets` is just W-chunking wrapped around the
   dense `assemble_tt_variation_jets_trs`, which it calls on every branch (L1983, L1868, L1873).
2. **S3 (8c87123) added 20 more production dense-`trs` contractions**, plus 1 order-scan weight site:
   - the dotted dx̃i: L2865-2870 ragged, L2899-2904 uniform;
   - the factored TT assembly's `M`/`N` builds: L2998-3001 ragged, L3050-3053 uniform;
   - the weight site: L2759.

   S3 did this on purpose. It follows plan §C.4 and §L.5, which turned Nick's "banded recurrences only"
   into "each dotted kernel takes the form of the GN kernel it differentiates, including dense `trs`
   where the GN kernel is dense". The plan records that turn as critic finding P5. S3 conforms to the
   plan. It does not conform to Nick's rule. It does not conform to the ledger's text either, though
   the ledger misdescribes the GN code.
3. **The pending S4 tree adds no contractions.** It adds 0 lines containing `contract(`, `einsum`,
   `trs` or `binomial`. It only wires `ProbeDerivativesKind.corewise_hessian` to the S3 kernel
   (`backend/fitting.py` working tree :691-696), so the S3 dense sites become reachable from
   `newton_cg(hessian='exact')`.
4. **A slice with one order axis capped at 2 is not order-independent.**
   - `trs[:, :, :2]` has shape `(o+1, o+1, 2)` and `trs[:, :2, :]` has shape `(o+1, 2, o+1)`. Both are
     O((o+1)²), with only 2o+1 nonzeros.
   - Only at `order ≤ 1` is every slice `≤ (2,2,2)`.
   - These banded slices are where numpy's forced greedy path does worst: 1.4–7.6× the flops and peaks
     2–80× larger than the recurrence.
   - On the jax path (opt_einsum, then XLA) the same slices cost only 1.0–1.35× in flops.
5. **For most sites the recurrence already exists in the module, for the same operation.**
   - Affine forward push: `_mu_jets_step` / `_sigma_banded_step`.
   - Affine reverse prop: the prop half of `_adj_tilde_step`.
   - The S3 `N` is exactly `pair()` (L2739-2742), and `M[:, :2]` is exactly `stacked()` (L980-983).

   Two sites have no non-dense form anywhere: the dx̃i-type correlation and the TT-assembly `t_det`
   term. Both are verified replaceable (≤1.4e-14).

---

## 1. Provenance method

- The clone is **shallow**: `.git/shallow` = 45c6742 (2026-09-04), and there are 12 commits in all.
  `git blame` attributes every line present at that boundary to `^45c6742`. So `^45c6742` means
  "present before the curvature plan (920d776, 2026-10-25)". The exact earlier commit is not
  recoverable here.
- The CHANGELOG `[2026.1.0]` (2026-08-20, L695-702) is where "Recurrence/scan jets are now the
  standard derivative path" shipped. The pre-existing dense production sites survived that change.
- Blame results:
  - **^45c6742 (pre-existing):** L311 331 333 461 483 505 559 761-763 798 843-845 873-875 910 959 1197
    1210 1267 1477-1478 1481-1482 1565 1605 1637-1638 1659-1660 1779-1781 1829-1831 2334 2354 2386 2407
    2425 2428.
  - **8c87123 (S3):** L2759 2803 2865-2870 2899-2904 2994 2998-3001 3047 3050-3053.
  - S3's diff of this file removed only 3 estimator lines. It changed no pre-existing contraction.

---

## 2. Inventory: every contraction whose operand is the binomial tensor or a slice of it

Search: `trs`, `binomial_combine_tensor`, `trs_*`, `t_sig`/`t_tau`, `binom`, `comb`, and every
3-letter order-block string. Outside `sampling_derivatives.py` the package has only comments
(`contractions.py:63,71,78`). The two `probing.py` hits were false positives (variable names). No other
spelling (`rts`, `tsr`, `srt`, ...) occurs.

### 2a. PRODUCTION, pre-existing GN code (category a)

| line(s) | enclosing fn | string | trs operand | production entry points | S3 dotted mirror? |
|---|---|---|---|---|---|
| **1197** | `_apply_derivatives_t3_step` (scan body of `_apply_derivatives_t3_from_xi_jets` L1200-1218; slice L1210) | `'trs,rWCa,Caib,sWCi->tWCb'` | `trs[:,:,:2]` (o+1,o+1,2) | `t3_apply_derivatives`, `t3_entries_derivatives`; `ApplyDerivativesKind`/`EntriesDerivativesKind.point_forward` (fitting.py:634-635, 657-658); `ut3_sampling.py:198,212`; `tucker_tensor_train.py:3959,3994` | not yet (S12 per plan C.5) |
| **761, 762, 763** | `_sigma_jet_step`; used by `_apply_derivatives_tv_step` L1251 → `_apply_derivatives_from_jets` L1254-1276 (slice L1267) | `'trs,rWKCa,Caib,sWCi->tWKCb'`, `'trs,rWCa,KCaib,sWCi->tWKCb'`, `'trs,rWCa,Caib,sWKCi->tWKCb'` | `trs[:,:,:2]` | `tv_apply/entries_jacobian_derivatives_from_sweep` → `Apply/EntriesDerivativesKind.forward` (fitting.py:627-628, 650-651); `utv_sampling.py:298,323,406,420`; `manifold.py:946,985`. Also called by the twin `compute_sigma_jets_trs` (L774) | not yet (S12) |
| **2334** | `_sigma_hat_jets_step` (scan body of `compute_sigma_hat_jets` L2337-2368; slice L2354) | `'trs,tWKCa,Caib,rWCi->sWKCb'` | `trs[:,:2,:]` (o+1,2,o+1) | `tv_apply/entries_transpose_derivatives_from_sweep` → `Apply/EntriesDerivativesKind.transpose` (fitting.py:630-632, 653-655); `utv_sampling.py:335,364,548,565`; `manifold.py:1058,1083`; corewise `tucker_tensor_train.py:4058,4078`, `ut3_sampling.py:235,252` | not yet (S12) |
| **2407** (ragged), **2425** (uniform) | `_apply_transpose_dxi_hat_step`; `_apply_derivatives_transpose_from_jets` | `'trs,rWCa,Caib,tWKCb->sWKCi'` / `'trs,drWCa,dCaib,dtWKCb->dsWKCi'` | full (o+1)³ | same as 2334 | not yet (S12) |
| **2386** (ragged), **2428** (uniform) | `_apply_transpose_assemble_core`; `_apply_derivatives_transpose_from_jets` | `'trs,rWCa,sWCi,tWKCb->KCaib'` (`->WKCaib` unsummed); d-prefixed twin | `trs[:,:,:2]` | same as 2334 | not yet (S12) |
| **1637, 1638** (ragged), **1659, 1660** (uniform) | `_dxi_tilde_jets_step`; `compute_dxi_tilde_jets` | `'trs,tWKCa,Caib,sWCb->rWKCi'` (from_tau), `'trs,rWCa,Caib,tWKCb->sWKCi'` (from_sig); d-prefixed twins | full (o+1)³ | `tv_probe_transpose_derivatives_from_sweep` L2263 → `ProbeDerivativesKind.transpose` (fitting.py:680-683); `manifold.py:1034`; corewise `tucker_tensor_train.py:4023`, `ut3_sampling.py:273`; `utv_sampling.py:350,529` | **yes**: L2865-2870 / L2899-2904 (6 product-rule terms each, same strings with K moved) |
| **1779, 1780, 1781** (ragged), **1829, 1830, 1831** (uniform) | `_tt_variation_jets_trs_core`; `assemble_tt_variation_jets_trs`, called on **every** branch of the "standard" `assemble_tt_variation_jets` (L1983 dense; L1868/1873 per W-chunk) | t_sig `'trs,rWCa,sWCi,tWKCb->KCaib'`; t_tau `'trs,tWKCa,rWCi,sWCb->KCaib'`; t_det `'trs,rWCa,tWKCi,sWCb->KCaib'` (`->WKCaib` unsummed); d-prefixed twins | t_sig `trs[:,:,:2]`; t_tau `trs[:,:2,:]`; t_det full (o+1)³ | same as 1637 (the dG half of the probe-derivative transpose) | **yes, refactored**: L2998-3001 / L3050-3053 (`M` = full trs·μ, `N` = `trs[:,:2,:]`·τ̃) + 4 finals |
| **505** | `_eta_jets_core._accumulate` (order scan; operand prep L559-560) | `'ts,sWCi->tWCi'` | per scan step `trs[:, r, :]` (o+1,o+1); scanned operand is (o+1)³, broadcast to `(d,)+…` on uniform | `compute_eta_jets` → `t3_probe_derivatives` (`ProbeDerivativesKind.point_forward` fitting.py:686) and `tv_precompute_probe_frame_sweep_jets` (`.precompute` :675) | not needed (no corewise η) |
| **910** | `_deta_jets_core._step` (prep L959-960) | `'ts,sWKCi->tWKCi'` | `trs[:, r, :]` | `compute_deta_jets` → `tv_probe_jacobian_derivatives_from_sweep` (`.forward` :678); also S3 L3161 | reused unchanged (S3 L3161 calls it for η̇) |
| **1565** | `_adj_tilde_step._src_step` (prep L1605-1606) | `'ts,tWKCb->sWKCb'` | `trs[:, r, :]` | `compute_tau/sigma_tilde_jets` → `tv_probe_transpose_derivatives_from_sweep`; also S3 L3153-3154 | **yes**: L2759 |

That is 19 four-operand dense-`trs` calls plus 3 order-scan weight sites.

### 2b. PRODUCTION, NEW in S3 (8c87123). S4 adds none.

They are reachable through the public backend `t3_probe_corewise_derivatives_hessian[_from_sweep]`
(both in `__all__`), and through the solver once S4 lands.

| line(s) | enclosing fn | string | trs operand | mirrors |
|---|---|---|---|---|
| **2759** | `_adj_tilde_tangent_step._src_step` (prep L2803-2804) | `'ts,tWKCb->sWKCb'` | `trs[:, r, :]` (o+1,o+1) | L1565 |
| **2865-2870** (ragged), **2899-2904** (uniform) | `_dxi_tilde_tangent_jets_step`; `compute_dxi_tilde_tangent_jets` | 6 terms: `'trs,tWKCa,Caib,sWCb->rWKCi'`, `'trs,tWCa,KCaib,sWCb->rWKCi'`, `'trs,tWCa,Caib,sWKCb->rWKCi'`, `'trs,rWKCa,Caib,tWCb->sWKCi'`, `'trs,rWCa,KCaib,tWCb->sWKCi'`, `'trs,rWCa,Caib,tWKCb->sWKCi'`; d-prefixed twins | full (o+1)³ | L1637-1638 / L1659-1660 |
| **2998, 2999** (ragged), **3050, 3051** (uniform) | `_tt_tangent_variation_jets_core` / `_tt_tangent_variation_jets_uniform` | `M = 'trs,rWCa->tsWCa'`, `Md = 'trs,rWKCa->tsWKCa'` (d-prefixed) | full (o+1)³ | L1779-1781 (the "binomial weights first" refactor; handoff judgment call 2) |
| **3000, 3001** (ragged), **3052, 3053** (uniform) | same | `N = 'trs,tWCa->rsWCa'`, `Nd = 'trs,tWKCa->rsWKCa'` | `t_tau = trs[:,:2,:]` (o+1,2,o+1) (slices L2994 / L3047) | L1780 |

The downstream contractions L3003-3016 / L3054-3066 contract M/N (`'tsWCa,…'`, `'rsWCa,…'`), not
`trs` itself. They inherit M's zero blocks, though: M[t,s] = 0 for s > t, so the deta-group terms
`'tsWCa,tWCi->sWCai'` (L3008, L3011-3012, L3058, L3061-3062) spend about half their
(o+1)²·W·r·n MACs on zeros.

### 2c. Reference twins and test-only forms (category c)

These are never reached from production. Callers were checked: only `compute_nu_jets_trs` →
`compute_mu_jets_trs`, `compute_tau_jets_trs` → `compute_sigma_jets_trs`, the `_adj_sweep`
wrappers, and tests.

| line(s) | fn | string | trs |
|---|---|---|---|
| 311 | `_mu_jets_trs_step` (`compute_mu_jets_trs`, slices L331/333; `compute_nu_jets_trs` L434) | `'trs,rWCa,Caib,sWCi->tWCb'` | `[:,:,:2]` |
| 461, 483 | `_eta_jets_trs_step` / `compute_eta_jets_trs` | `'trs,rWCa,Caib,sWCb->tWCi'` (+d) | full |
| 761-763 (as called from L774) | `_sigma_jets_trs_step` → `compute_sigma_jets_trs` / `compute_tau_jets_trs` | (see 2a) | `[:,:,:2]` |
| 843-845, 873-875 | `_deta_jets_trs_step` / `compute_deta_jets_trs` | `'trs,rWKCa,Caib,sWCb->tWKCi'` etc. | full |
| 1481, 1482 | `_adj_sweep_step` (`_adj_sweep` → `compute_tau/sigma_tilde_jets_trs`; slices L1477-1478) | prop `'trs,tWKCa,Caib,rWCi->sWKCb'`, src `'trs,rWCa,Caib,tWKCi->sWKCb'` | `[:,:2,:]`, full |
| tests | `tests/backend/_curvature_oracles.py:206-290` (two-hole 6- and 9-term twins); `tests/test_jet_recurrence.py` (uses the package's `*_trs`) | — | — |

Two misnomers:
- `assemble_tt_variation_jets_trs` is labelled the "dense reference", but it is the production kernel
  (row 2a).
- `assemble_tucker_variation_jets_trs` (L1698-1727) contains **no** `trs` operand at all. Its order sum
  `'tWCa,tWKCo->KCao'` is diagonal.

### 2d. Structural or shape-only, never contracted (category d)

- **Construction and slicing.** The tensor is built at L250-265 and by every
  `binomial_combine_tensor(order)` call: L201, 633, 654, 1124, 1241, 1302, 1361, 1387, 2257, 2465,
  2527, 3146. Slices are taken at L331/333, 798/800, 1210/1212, 1267/1269, 1477-1478/1494,
  2354/2356, 2994, 3047. `moveaxis`/`broadcast_to` prep is at L559-560, 959-960, 1605-1606,
  2803-2804.
- **Shape-only consumers.** `compute_mu_jets`, `compute_nu_jets`, `compute_sigma_jets` and
  `compute_tau_jets` read only `trs.shape[0]` (L411, 1034).
- **Estimator structs.** `S(op,op,op)` at L2073 and L2125 is abstract lowering for measurement only.
  Note the coupling, though: `_assembly_per_row_bytes` hard-codes the dense
  `assemble_tt_variation_jets_trs` (L2081).
- **Order-led strings that do not involve trs.** Examples: the recurrences L384/386, 986-992, 1554/1556
  and 2743-2747; the lifts L573/592/1071/1093/1447/1464/2671/2686; the Tucker assemblies
  L1684-1685/1721-1722/2920-2921/2934-2935.

---

## 3. What each production family computes, and whether a recurrence already exists

| family | sites | math (s ∈ {0,1} is the affine input-jet leg) | non-dense form already in module? |
|---|---|---|---|
| **F1 affine forward push** (binomial convolution, banded) | 1197; 761-763 | `out^(t) = J^(t)·A^(0) + t·J^(t-1)·A^(1)` | **Yes, same op.** `_mu_jets_step` (L368-387) for 1197; `_sigma_banded_step` (L968-993, the identical 3 terms) for 761-763. The dense form is pure duplication. |
| **F2 affine reverse prop** (correlation = adjoint, banded) | 2334 | `out^(s) = c^(s)·A^(0) + (s+1)·c^(s+1)·A^(1)` | **Yes, same op.** The prop half of `_adj_tilde_step` (L1548-1556), with Q in place of P. |
| **F3 dx̃i-type: adjoint of the full combine onto its affine leg** (correlation) | 1637-38/1659-60; 2407/2425; S3 2865-70/2899-904 | from_tau `d^(u) = Σ_s C(u+s,u) τ̃^(u+s)·O·ν^(s)`; from_sig `d^(u) = Σ_r C(r+u,r) μ^(r)·O·σ̃^(r+u)`. **Only u ∈ {0,1} are consumed** (L1685, L1722 `dxt[:s_size]`; L2387, L2429; S3 L2921, L2935), yet the dense form computes all u = 0..o. Rows u ≥ 2 are nonzero (max \|·\| 8.7e1 in the check) and dead. | **No.** The natural form is the two-row `pair()` correlation `pair_rev(τ̃)[u,s]·(O·ν^(s))` (verified 1.4e-14). |
| **F4 TT assembly, banded terms** (correlation, W summed) | 1779/1829, 2386/2428 (t_sig); 1780/1830 (t_tau) | t_sig `Σ_r μ^(r)⊗ξ^(0)⊗σ̃^(r) + (r+1)μ^(r)⊗ξ^(1)⊗σ̃^(r+1)`; t_tau the same with τ̃, ν | **No assembly form exists.** The fix is `pair_rev` + two GEMMs (verified 8.9e-15 / 7.1e-15). The ledger itself names "the assembly contractions" as the bidiagonal case. |
| **F5 TT assembly t_det** (genuinely full 2-D correlation) | 1781/1831 | `Σ_{r+s≤o} C(r+s,r) μ^(r)⊗δη̃^(r+s)⊗ν^(s)` ((o+1)(o+2)/2 order pairs) | **No.** The analog of `_src_step` is an order scan over the lag r (verified 1.4e-14). |
| **F6 S3 M/N builds** | 2998-3001/3050-3053 | `N[r,s] = C(r+s,r) τ̃^(r+s)`, r ∈ {0,1}, which **is** `pair_rev(τ̃)`. `M[t,s] = C(t,t-s) μ^(t-s)` (lower-triangular binomial Toeplitz). `M[:, :2]` (the σ̃ group, L3003-3005) **is** `stacked(μ)` transposed. Full M is used only in the deta group. | **Yes for N and M[:, :2]:** `pair()` L2739-2742 and `stacked()` L980-983. Full M can be built by shift-scale. |
| **F7 order-scan binomial weights** | 505, 910, 1565; S3 2759 | forward `out^(t) += C(t,r)·X^(t-r)`; adjoint `out^(s) += C(s+r,r)·X^(s+r)` | The dense 2-D slice is how the "order scan" applies its weights. Shift+scale is exactly equal (0.0). |

---

## 4. Cost: dense versus the recurrence or scan form

### 4.1 Size of the trs operand itself

The operand is negligible: full (o+1)³ is 216 floats at o=5, a host constant that folds into jit.
The cost comes from what its density does to the contraction path:
- **numpy** forces the greedy `_pairwise_path` (`contractions.py:77-94`). That path pairs by shared
  letters, so `trs` (one letter shared per operand) is contracted **last**. The two order-carrying
  jets are first combined at all orders.
- **jax** uses opt_einsum `'auto'`, which costs `trs` as a small dense operand and contracts it early
  against the smallest jet.

### 4.2 Measured flops

Units: flops (opt_einsum's count) / largest single intermediate (elements), per core. The full sweep is
in the script.

**W = 2000, r = n = 16, K = ()**

| site | o=3 numpy/alt | o=3 jax/alt | o=5 numpy/alt | o=5 jax/alt | o=8 numpy/alt | o=8 jax/alt |
|---|---|---|---|---|---|---|
| F1 push (each of 1197, 761-3) | 1.85 | 1.05 | 2.56 | 1.10 | 3.44 | 1.20 |
| F2 prop 2334 | 1.85 | 1.05 | 2.56 | 1.10 | 3.44 | 1.20 |
| F3 dx̃i (each term) | 1.17 | 1.17 | 1.35 | 1.35 | 1.67 | 1.67 |
| F4 t_sig / t_tau | 2.50 | 1.05 | 3.95 | 1.10 | 6.30 | 1.20 |
| F5 t_det | 1.70 | 1.05 | 2.50 | 1.10 | 4.06 | 1.20 |

**W = 2000, r = n = 16, K = 3**

| site | o=3 numpy/alt | o=3 jax/alt | o=5 numpy/alt | o=5 jax/alt | o=8 numpy/alt | o=8 jax/alt |
|---|---|---|---|---|---|---|
| F2 prop 2334 | 3.96 | 1.11 | 4.96 | 1.20 | 6.00 | 1.35 |
| F3 dx̃i from_tau | 2.86 | 1.41 | 3.31 | 1.85 | 4.10 | 2.64 |
| F5 t_det | 1.70 | 0.88 | 2.50 | 0.85 | 4.06 | 0.83 |

What the table shows:
- At r = 64, n = 32 the jax ratios fall to 1.00–1.35×. The numpy ratios stay at 1.4–7.6×.
- **Peak intermediates on numpy.** F5 t_det builds an (o+1)²·W·K·r·n outer product. At o=5, W=2000,
  r=n=16, K=3 that is 5.5e7 elements versus 1.5e6 for the scan form; at o=8 it is 1.2e8. This happens
  on the full W on ragged, since only the uniform path W-chunks (L1984).
- **On jax, F5 t_det's scan form costs slightly MORE flops (0.83–0.9×) when K > 1.** It re-forms the
  K-carrying pair per lag. On jax, F1/F2/F4/F5 are not a flop problem.
- **F3 is a waste on both backends.** It computes (o+1) output rows where 2 are used.

### 4.3 Closed forms (MACs, K on the variation operand)

| family | dense | recurrence or scan |
|---|---|---|
| F1 | numpy ≈ (o+1)Wr²n + 2(o+1)Wrn + 2(o+1)²Wr | 2Wr²n + 2(o+1)Wr² |
| F3 | (o+1)Wr²n + (o+1)²WKrn + (o+1)³WKn | (o+1)Wr²n + 2(o+1)WKrn; output 2 rows, not o+1 |
| F5 | numpy ≈ (o+1)²WKrn + (o+1)³WKrn + (o+1)WKr²n | ≈ (o+1)²WKrn + (o+1)WKr²n; peak W·K·r·n |
| F6 | M build (o+1)³Wr | (o+1)(o+2)/2·Wr nonzeros |
| F7 | (o+1)³Wn per core | ~(o+1)(o+2)/2·Wn |

Shares of the surrounding work:
- **F6** is 9–81× its alternative, but small in absolute terms: about 7% of the S3 assembly at o=5 and
  16% at o=8 (r=n=16), and under 1% at r=64.
- **F7** is a share (o+1)²/(r²+(o+1)r) of its scan step: about 10% at o=5, r=16; 20% at o=8; ≤2% at
  r=64.

---

## 5. What the governing texts say, and who conforms to what

- **`docs/contributor/deferred_and_rejected.md:85-96`** (verbatim core):
  > "The pushthrough/assembly contractions convolve a full-order jet with the input/perturbation jet,
  > which is capped at 2 orders … bidiagonal, `O(K)` where the dense `(K+1)×(K+1)` einsum is `O(K²)`.
  > … It was built afterwards as the recurrence/scan forms … `compute_mu_jets` is a fused two-term
  > recurrence with no `trs` operand at all, and the genuinely-full convolutions (`eta`/`deta`/tilde)
  > scan the summed order axis instead. **Those are the defaults at every call site; the dense binomial
  > forms survive as the `*_trs` reference twins**."

  It contains **no** "otherwise dense `trs`" leg. It is factually wrong about families F1–F5. Ironically,
  its motivating measurement (derivative apply forward ~32×, transpose ~78×) is the apply path that was
  never converted.
- **Plan §C.4, "The production standard (P5, settled as the house applies it)":**
  > "Each dotted kernel takes the form of the GN kernel it differentiates (`deferred_and_rejected.md:85-96`):
  > a fused bidiagonal recurrence where the input jet is affine …; an order scan where the convolution is
  > genuinely full …; **otherwise the GN counterpart's form: dense `trs` inside `compute_dxi_tilde_jets`
  > …, and dense `trs` per W-chunk in `assemble_tt_variation_jets`**."

  It adds: "The `_trs` dense forms exist only as test references … never called in production". That
  sentence refers to the prototype two-hole twins. It does not mean "no dense trs in production".
- **Plan §L.5:**
  > "The intent of Nick's rule is that small-scale dense timings must not choose production forms. The
  > house already applies that rule as 'the right form per convolution type' (`deferred_and_rejected.md:85-96`).
  > Holding each dotted kernel to its GN counterpart's form satisfies the intent exactly. Demanding more
  > would make the derivative faster than the operator it differentiates, with no evidence it is needed."
- **Plan P5 row (L126):** "'banded recurrences only' contradicts today's production code → Restated as
  the house rule (`deferred_and_rejected.md:85-96`)". This is where Nick's rule was explicitly
  downgraded. Plan §C.5 extends the same rule to S10–S12, so **S12 would mirror F1/F2 dense** even
  though recurrences exist for them.
- **The S3 module comment (L2656-2659) states the rule in code:** "dense trs where the GN kernel is dense
  trs". The S3 review §2 verified it line by line.
- **Verdict:**
  - S3 **conforms to C.4/L.5**.
  - S3 does **not** conform to Nick's "recurrences, not dense trs": it adds 20 new production dense
    contractions (21 sites counting L2759).
  - S3 does **not** conform to the ledger as written. But the C.4, L.5 and P5 citations of :85-96 are
    **mis-citations**: the "form per convolution type, dense otherwise" rule was read off the code, not
    the ledger.
- **The same inaccuracy appears in two shipped texts.**
  - **CHANGELOG `[2026.1.0]` L695-702** says "the pushthrough, combine and variation-assembly jets are
    computed by … recurrences and order-scans rather than by contracting the dense binomial `trs`
    tensor". It names `assemble_{tt,tucker}_variation_jets` as lean forms. In fact the TT one is dense
    `trs` chunked over W (its 168 GB → 2.63 GB win is from chunking), and the Tucker one has no `trs`.
  - **`docs/naming_conventions.md:144-149`** says "bare name = recurrence/scan". That is false for
    `assemble_tt_variation_jets`, `compute_dxi_tilde_jets`, `compute_sigma_hat_jets` and
    `t3_/tv_{apply,entries}_derivatives[_transpose]`.

---

## 6. Replacing each production dense site

Each recipe was checked numerically in the script.

| # | sites | recipe | carry / W-chunk interaction | effort |
|---|---|---|---|---|
| R1 | 1197 | Body → `_mu_jets_step` math (`next, _ = _mu_jets_step(mu, (G, xi))`; return `(next, (0,))`). Drop `trs_push(es)` L1208-1212 from scan data. | carry = μ jet, unchanged; no chunking | XS (<1 h + a twin test) |
| R2 | 761-763 via 1251 | `_apply_derivatives_tv_step` → `_sigma_jets_step(...)[0]`. Drop L1265-1269. `_sigma_jet_step` then serves only the twin. | carry = σ jet; none | XS |
| R3 | 2334 | Use the reverse two-term prop of `_adj_tilde_step` L1548-1556. Ideally factor a shared `_reverse_banded_prop` used by `_adj_tilde_step`, `_adj_tilde_tangent_step` (`pair()`) and here. Strings already exist in HISTORICAL. | carry = σ̂ jet (K from c); none | S (1–2 h) |
| R4 | 1637-38/1659-60; 2407/2425; S3 2865-70/2899-904 | Two-row `pair_rev` correlation: from_tau = `'usWKCa,sWCai->uWKCi'(pair_rev(τ̃), 'Caib,sWCb->sWCai'(O,ν))`; from_sig = `'rWCib,urWKCb->uWKCi'('rWCa,Caib->rWCib'(μ,O), pair_rev(σ̃))`; S3 = the 6-term product rule of the same. **Decision needed:** return 2 rows (shape-contract change to `compute_dxi_tilde[_tangent]_jets`; consumers already slice `[:s_size]`; `_jet_floor_bytes`/`_two_hole_jet_floor_bytes` count `op·nO` for dx̃i and would shrink), or zero-pad back to o+1. Keep the current code as a `*_trs` twin. Add the new strings as HISTORICAL rows (`TestVocabularyCompleteness`). | map over cores, no carry; runs before the assembly, so no chunk interaction | S–M (GN ½ day; S3 dotted twin ½ day) |
| R5 | 1779-81/1829-31; 2386/2428 | t_sig / t_tau → `pair_rev` + 2 GEMMs. t_det → order scan over the lag r (carry = the (K,C,a,i,b) accumulator; xs = μ^(r) plus the r-shifted, C(r+s,r)-weighted δη̃; on jax the traced shift needs a padded `dynamic_slice`), or M-first with M built by shift-scale. The new kernel becomes the per-chunk `chunk_fn` (`_wchunked_sum`/`_concat` untouched); the order scan nests inside the W-chunk `lax.scan`. **Must re-point `_assembly_per_row_bytes` L2081** off the dense kernel and re-baseline `tests/test_chunk_size_estimator.py`, then re-measure jax memory. On jax the flop gain is ~0–20% (t_det may lose with K>1); the gain is numpy speed/peak plus the principle. | yes, inside chunks | M (~1 day incl. jax measurement) |
| R6 | S3 2998-3001/3050-3053 | `N, Nd` = `pair_rev(τ̃), pair_rev(τ̃̇)` (exact identity). `M[:, :2]` = `stacked(μ)` axes swapped (exact). Full M (deta group only) by (o+1) shift-scales, or an order scan over the lag. `_two_hole_assembly_per_row_bytes` follows automatically (it measures `_assemble_tt_tangent_variation_jets_dense`). Re-run S3's cost/memory gates. | inside chunks; unchanged | S (½–1 day incl. gates) |
| R7 | 505, 910, 1565, S3 2759 | `'ts,…'(trs[:,r,:], X)` → shift X by r and scale by the binomial column (exact). In `lax.scan`, r is traced, so this needs a dynamic slice of zero-padded X. | inside order scans | S (~2 h), low value |

---

## 7. Where the dense form is defensible

- **order ≤ 1.** Every slice is ≤ (2,2,2) (`s_size = min(2, o+1)`). The dense form is harmless there,
  but it is not a reason to keep it for general order.
- **F7 (the order-scan weights).** This is the ledger's own named "scan" standard. The operand is a
  constant (o+1)² matrix applied to a W·r slice, which XLA handles well, and its share is about
  (o+1)²/r² of the step. It is defensible as-is; low priority.
- **On the jax path only, F1/F2/F4/F5 cost 1.0–1.35× in flops**, because opt_einsum routes `trs`
  well. They are defensible on cost there, but not on principle, and not on numpy or ragged
  (1.4–7.6× flops, 2–80× peak).
- **Not defensible:**
  - **F1/F2.** An identical recurrence already exists in the module.
  - **F3.** It wastes (o+1)/2 of its output rows on both backends (up to 2.6× on jax).
  - **F6's N and M[:, :2].** These are literally the existing `pair()`/`stacked()`.

Two follow-ons need attention:
- The **chunk-size estimator** is calibrated on the dense TT kernel (L2081), so any R5 change must
  re-point it.
- **S12 will copy F1/F2 dense into the apply/entries-derivative two-holes** unless plan C.5/P5 is
  revised first.
