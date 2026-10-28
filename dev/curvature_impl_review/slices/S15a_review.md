# S15a review: docs/exact_hessian_two_hole.tex + .pdf (independent reviewer)

**Verdict: APPROVE-WITH-FIXES, fixes applied in the .tex, PDF rebuilt** (16 pp, 0 overfull, 0 `??`, aux cleaned; only the
2 untracked files; nothing committed or edited elsewhere). No MAJOR finding: no wrong equation, no wrong code reference.

## Independent checks (all in review/s15a/)
- **R1 `r1_dense_hessian.py`.** Boxed eq. (10) against a library-free extrinsic oracle: tangent space = range(J_Phi) from my
  own einsum, P = JJ^+, AMT Hess f[z] = P A^T W^2 A z + P(D_z P) grad_E, D_z P by Richardson FD; my own dense A, validated
  vs `t3_probe/apply_derivatives` to 5e-16. `lm.hvp` (exact, eps = 0) agrees to 1e-9…5e-12 (FD floor) on: probe-derivs
  o2 + omega[mode,order]; plain probe + per-mode omega at a GENUINE Tucker-slack frame and at a TT-slack frame; apply-derivs
  o2; shared (0,0,1) group-ceiling. **K_{grad f} sign confirmed:** flipping it gives O(1) error (0.7–33); GN is off by 0.6–1.0.
- **R2 `r2_prop1_slack.py`.** Eq. (1), the double replacement, equals the exact roots-of-unity st-coefficient (<= 2e-15).
  Prop. 1 lift independence holds over ALL of ker DPhi (1e-15), including the Tucker-slack kernel (dim 44), which is larger
  than the gauge orbit. D^2Phi[v_ker, b] is tangent (2e-15).
- **R3 `r3_kdef_dense.py`.** Eq. (5) vs my own dense Z and D^2Phi for every boxed two-hole family — probe, apply, entries,
  probe-derivs o3, apply-derivs o2, T3-T3 — with K = (2,) plus one K = (2,), C = (2,) case: <= 1.9e-15. The writer's
  `checks.py` C1–C14 reproduce exactly (Table 1).
- **Code read against the boxes**, 8 of 10 (Alg. 1, 2, 3, 5, 7, 8, 9, 10): the lift/filter/factors, tied filter clip
  max(sum nD, n_g), `LocalModel.hvp/_curvature_block`, `_adj_prop_tangent` (2 GEMMs), `_adj_tangent_src_order_step` (U^r
  deta~, live slice WKr^2), the jet TT assembly (2 banded finals + 2 per lag step; eqs. 23/24 reduce to the binomial forms
  by hand), plain probe (mu.E1 + tau~.E2), T3-T3 (25)/(26), the zero frame and the TR loop. The TR constants, Delta0,
  descent guard, both floors and Steihaug tau (28) match line by line. Tuple order is canonical (U, O, P, Q) / (V, H).
- **Measured facts** (F6, F14, F19, F22, F37, 720 / 8000 cases, cost 2.8–3.9x) all trace to verification_records / slices.
- **References.** My stricter `refs.py` (log `refs.log`: AST def sites + qualifier check): all 228 `\code{}` names resolve.
  The writer's 7 FAILs are false positives (`\urlstyle`, `None`, `_trs`, 3 module names, `TestCurvatureJvpOracle` at
  tests/test_dispatch.py:2170). My only extra flags are literals, paths and role names (`geom.lift`, `kind.*`, `self.masks`).

## Fixes applied (section/eq — what, why)
1. **§3.4 intro.** "Nine → four (F33 …)" now reads: four for the plain probe, two plus two per lag step for the jets, two for
   apply/entries; F33 is attributed to the planned six-final jet form. Why: F33 timed the plan's 6-final form (S3 shipped 4;
   S3b moved to banded + lag), and "four" contradicted the note's own jets paragraph.
2. **Bib [5].** "I." → **A.** Molozhavenko (Alexander, arXiv:2508.20928); added Comput. Appl. Math. 45:221 (2026), as
   shared_t3_math.tex cites it.
3. **Table C3–4.** (M^T M + eps^2)^{-1} → (M^T M + eps^2 I)^{-1}.
4. **§1 "The frame".** Notes the Algorithm-11 sweep-order divergence (the code sweeps left-then-right, the paper
   right-then-left; equivalent). The note cites Alg. 11, so CLAUDE.md asks for this.
5. **§2.1.** Lift independence is "visible directly *for the gauge directions*"; at slack, ker DPhi exceeds the orbit and
   Prop. 1 covers it (R2).
6. **§2.2, after the table.** The lift is the undamped one; with eps > 0 the GN term stays exact and only the curvature term
   is filtered. Why: the "= Pi L^T Hess L Pi" identities hold only at eps = 0.
7. **Wording.** §2.3: clip projection Q → Q_c (Q is the right core). §3.2: Y "linear" → "affine in theta". §3.5: the
   bit-identity is the frame kernel at a manifold frame, gauged. §4: shared_t3_math.tex writes factors N x n (transposed).
8. **Table C7.** `jax.hessian` → `jax.jvp` of `jax.grad` (what the check ran). §7 tightened by one line, so that
   "internals.md." no longer dangles after the p.16 table float.

## Remaining (minor/nit — not changed)
- **m1.** The `d` small SVDs count in two docs (ruling below). **m2.** curvature_internals.md, the F33 paragraph: "6 instead of 9 for the probe-derivative assembly" describes the
  planned form, not the shipped one. Doc fix for S15b; the note is right.
- **n1.** "A TT rank equal to the product of its neighbours" (§2.4, also §4.12 and the design record) is loose; the ceiling
  is min(r_{i-1} n_i, n_{i+1} r_{i+1}). Harmonize all three at S15b.
- **n2.** Underfull hbox in bib [3] (cosmetic). **n3.** The plan IDs (I-xx, F-xx, "check F6") resolve only in dev/, as in
  curvature_internals.md; S15b decides whether shipped docs keep them.

## curvature_internals.md discrepancy — RULING: the note is right, the docs are wrong
`fv_lift_factors` runs one `damped_pseudo_inverse` per Tucker factor (d) and per bond (d-1): **2d-1** thin SVDs. The
uniform twin does the same count in 2 batched calls. The shared lift adds none (it reuses the precompute companion's s_g
SVD). "`d` small SVDs" is wrong in **curvature_internals.md** ("Misfit gradient once") **and fitting_and_optimization.md
§4.12** ("Cost"). These are S15b/R-era doc fixes; the note was not changed.

## S15b must re-check
- **Every T4S number** (unverified against t4s.pdf, which is absent here; all docstring-sourced and flagged by the box):
  App. A.1/A.2/A.3/A.3.1; eqs. (47), (48)–(49), (50)–(53); Algs. 5, 6–7, 7, 8, 10, 11; §6, §6.2.3, §6.3; the bib for [1].
- **Re-run `refcheck.py` + `refs.py` after R.** Most at risk: private kernels (`_adj_*`, `_tt_*`, `_shift_*`, `_cg_*`,
  `_TR_*`, `_haar_rows`, `_lift_sigma_min`, `_frame_theta`, ...), `LocalModel` (backend vs frontend) and the other classes,
  the module qualifiers, and the test paths.
- **Drop the PRELIMINARY box, update the hash, link the note** from curvature_internals.md / §4.12; apply m1/m2/n1.

## Figures (80 dpi pages, 150 dpi crops; T4S convention: TT top, Tucker below, hole = dashed node; no box overflows)
F1 (p.2) clear, all panels readable · F2 (p.5) clear, the brace label is small but legible · F3 (p.11) legible, the
smallest · F4 (p.11) ladder clear.
