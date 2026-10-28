# S15a handoff -- preliminary math note (DONE, uncommitted)

Outputs (untracked, nothing else touched): docs/exact_hessian_two_hole.tex, docs/exact_hessian_two_hole.pdf
(16 pp, 11pt article; builds clean with `latexmk -pdf`, 0 overfull, aux cleaned with `latexmk -c`). No toctree edit:
no .rst/.md toctree lists the .tex notes (they are only linked from md pages).

Sections: front-page PRELIMINARY (S15a) box (refs correct at f739085, re-checked in S15b; T4S numbers as the
docstrings cite them) / 1 Setup+notation (frame U,O,P,Q; E (47); gauge (48)-(49); f, grad f, N; W/K/C) /
2 Exact Hessian: 2.1 Weingarten = <N, D2Phi> (Prop 1, AMT13; lift independence via gauge generator; K_Z def) /
2.2 Lagrangian form + level 0/1/2 table + retraction second order (MEASURED: F6, F14, T7, C8a) / 2.3 lift
(frame factors S_i,T_i,H_i; Prop 2 with proof; slack frames; damping sigma/(sigma^2+eps^2), eps = c*||X||, clip,
restarts; transpose) / 2.4 Newton system (boxed eq (10)), corewise exact, misfit-gradient-once, flat slots /
3 Two-hole: 3.1 one-hole recap (11)-(14) / 3.2 product rule, dotted forward = GN tangent forward at Q=O=P=G,
recompute rule, block-sum oracle / 3.3 two-index jets, D/U shifts, 4 product forms, recurrence-only rule (I-33),
tau~ (18), tau~dot (19), dxi~ two rows (20) / 3.4 factored assembly: Tucker (21), plain probe 9->4 (22),
jets banded pair + lag fold (23)-(24), apply 3->2, W-chunking, apply/entries seeded sweep / 3.5 T3-T3 (25)-(26) /
3.6 uniform masking (FRAME's masks) / 4 shared tied lift (27), s_g in lift_sigma_min, padded restarts /
5 zero-point frame / 6 trust region (Steihaug tau (28), floors) / 7 verification table + costs / refs (8).

Figures (TikZ, T4S convention: TT top, Tucker below, squares = vectors, dashed circle = hole, red = direction):
F1 one-hole vs two-hole (3 panels); F2 the lift (TT slot via P_{>i}=T_i Q_{>i}; Tucker slot via H_i=O_i x2 S_i^T);
F3 pair intermediates E1/E2 and the factored derivative; F4 T3-T3 ladder with tangent zipper.
Algorithm boxes (10, each with a Code: line): lift+adjoint; exact hvp; one-hole (reference); two-hole forward;
two-hole adjoint; probe-derivative driver; T3-T3 two-hole; tied lift+adjoint; zero-point frame; TR Newton loop.

Equation checks: impl/s15a/checks.py
(log checks.log; run `PYTHONPATH=$PWD python3 <it>` from the repo root). ALL PASS, rel. residuals:
C1 frame factors (H=O x2 S^T, P_{>i}=T_i Q_{>i}, spectra, ||X||^2) <=8.4e-16 | C2 lift DPhi[Lv]=Ev minimal /
Tucker-slack / TT-slack 1.1e-15/6.7e-16/2.4e-15 | C3 damped adjoint 1.1e-16 | C4 filter max 1/(2eps), pinv form
<=1.9e-15 | C5 grad f = P_X A^T r, N normal <=9.9e-16 | C6 <N,D2Phi[th,th']> (dense double replacement) =
<th',K_N th> 1.2e-13; K_N th_vert = 0 6.0e-14; lift independence 5.2e-14 (K_{A^T r} th_vert ~1e4) | C7 hvp =
Pi L^T Hess(Lagrangian) L Pi via jax.hessian f64 4.5e-14; grad Lagrangian = 0 1.2e-15; level 1 = Hess(f o Phi)
1.3e-14 | C8 <p,Hp> vs FD along tv_retract (Richardson) 3.7e-6; level 1 vs FD along Phi(th+tLp) 2.5e-10 |
C9 two-holes = complex-step d/ds of the one-holes (probe-deriv o2 d4, apply-deriv, T3-T3) <=4.6e-16; block sum
3.4e-16; symmetry 4.1e-16; order 0 = plain probe 3.1e-16 | C10 written recurrences mu-dot, tau~ (18), tau~dot (19),
dxi~dot vs the kernels <=4.3e-16 | C11 factored = nine-term binomial (jets 6.3e-16, GN 3.7e-16, plain 2.1e-16) |
C12 tied lift identity 5.1e-16, = pinv(M_g)[V] 1.1e-16, adjoint 0 | C13 zero frame nested / rank-indep / base 0 /
recipe by hand: exactly 0 | C14 hvp = GN + curvature_hvp, quadratic split, gradient bitwise, default eps <=4.4e-16.

Code references: refcheck.py (same dir; log refcheck.log) greps all 228 \code{} names at the current tree (=f739085
plus the 2 new files): every function/class/constant/field resolves; its 7 "FAIL" lines are false positives (the
\urlstyle macro def, `None`, the `_trs` suffix, the module names sampling_derivatives / tv_operations / t3_linalg,
and TestCurvatureJvpOracle, which is tests/test_dispatch.py:2170). Modules cited: sampling_derivatives, probing,
apply, entries, fv_operations, fv_conversions, tv_operations, t3_linalg, tt_operations, sharing, linalg, geometry,
optimizers, fitting, uniform_fitting, ufv_operations, utv_operations, ut3_sampling, ut3_linalg, ut3_masking.

TODO for S15b / notes for the reviewer:
- t4s.pdf is NOT in this container (untracked); every T4S number (App A.1/A.2/A.3/A.3.1, (47), (48)-(49), (50)-(53),
  Alg 5-8, 10, 11, §6.2.3, §6.3) was taken from the code docstrings -- re-verify against the local t4s.pdf.
- Re-run refcheck.py after R (sampling_derivatives / fitting / optimizers are R's targets).
- Small inconsistency found: curvature_internals.md says the exact model adds "d small SVDs" per step; the code
  (fv_lift_factors) does 2d-1 (d Tucker + d-1 bond). The note says 2d-1.
- TT-assembly grouping is described AS SHIPPED (plain probe: by the left factor mu / tau~; jets: sigma~/tau~ banded
  through the shared affine xi + the deta~ lag scan); S3's dense grouping by the common nu is a one-line remark.
  F33's 1.4-2.0x was measured on S3's factored form -- worded as "the factored form".
- The retraction's second order is stated as MEASURED (the design-note argument cited as motivation only).

Draft commit message:
  docs: the exact-Hessian math note, preliminary (S15a)
  docs/exact_hessian_two_hole.tex + .pdf: the Weingarten term as <N, D^2 Phi>, the lift and its damping, the
  Lagrangian form and the levels, the two-hole sweep as implemented (dotted forward/adjoint jets, the
  recurrence-only rule, the factored assembly, T3-T3, uniform masking), the tied lift, the canonical zero-point
  frame and the trust region; 10 algorithm boxes citing their functions, 4 TikZ figures. Every equation checked
  numerically; code references verified at f739085, re-checked after the reorganization (S15b).
