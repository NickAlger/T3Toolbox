# S13b + S15b handoff — main @ 76aa37f, tree UNCOMMITTED (20 files, incl. the rebuilt PDF)
## Review items -> done (file:line)
- C1 rcond: linalg.py:722 filtered_reciprocal, :778 damped_pseudo_inverse, sharing.py:647 _tied_filter ->
  `Optional[Union[float, NDArray]]` in its own signature group; "a scalar, or broadcastable to C+(1,)" documented.
- C2: fitting.py:802-803 apply_derivatives_model = entries_derivatives_model's two lines (uniform x).
- C3: geometry.py:44-46: raises on a corewise geometry; on the manifold at d = 1 it is validated (>= 0) but unused (data None).
- C4 / B float32: LEDGER (below). D1: "d small SVDs" -> "`2d − 1` small SVDs (the lift's d Tucker and d − 1 bond
  pseudo-inverses)": curvature_internals.md:143, fitting_and_optimization.md:592 (the Cost paragraph is reflowed).
- D2 §4.12 (fitting_and_optimization.md:557-560): `ex <= 10 and ex < gn` + progress `rel(ex,5) < 1e-8 < rel(gn,5)`.
  Measured: ex row 5 = 1.1e-10 and gn row 5 = 1.4e-3 on all 3 envs. GN ends 15 line_search / 20 gradient / 16 line_search.
- D3: the module doctests optimizers.py:528-548 and backend/optimizers.py:1257-1263 now use the default gtol_rel and
  assert stop_reason in ('gradient', 'objective_floor'); their prose states only facts that hold on every BLAS.
  Measured: ex 5 steps, 'gradient' (2.3e-7 -> <1e-13) everywhere; GN 28 line_search (sys) / 29 / 30 gradient.
- D4 CLAUDE.md -> patch E. D5 -> D. D6: README.md:108-113 adds the hessian='exact' + trust-region sentence.
- D7: F28 is "78 CG iterations" in both places: curvature_internals.md:321 (one Hessian-apply each; the per-step pᵀHp
  is extra; + 2.6e-9) and verification_records.md:62. Source: S12 rerate.py sums `cg_iters`.
- D8: CHANGELOG:101-103: "removed too -- lives on privately as `_assemble_tucker_variation_jets_dense`".
## B verdict: NOT simple -> ledger + CHANGELOG reword, no code change
- Tracer (float32 numpy fits, 6 kinds x ragged/uniform x manifold/corewise/shared x GN/exact): 38 default-dtype
  constructors in 8 modules (apply, entries, probing, t3_conversions, tt_operations :78/:83 squash tails + :190 zipper,
  tv_operations, ut3_svd uniform SVD paddings, utv_operations). Patching all 38 -> every fit float32 end to end, float64
  bit-identical (80 arrays). Own slice because each jax scan init must take ALL its scan operands' result_type (x64 +
  mixed dtypes). Patch: scratchpad/impl/s13b/float32_followup.patch (some inits use partial operand sets; fix first).
- Ledger: deferred_and_rejected.md:252-264. CHANGELOG [2026.3.0] Fixed now reads "each jet kernel preserves
  float32 ... a float32 fit can still come back float64 (every manifold fit does) ... ledger".
## D math note (exact_hessian_two_hole.tex/.pdf, 16 pp, final LaTeX pass 0 warnings; p1, p15, p16 viewed as PNG)
- PRELIMINARY box + header comment removed; author `... collaborators\\ co-authored by Claude Fable 5.1`; bib T4S = plain
  arXiv + URL; re-pinned f739085 -> 76aa37f (:1002, :1030); checks.py re-run on a 76aa37f archive: all 53 residuals identical.
- Code-reference check (refcheck.py / refs.py): 228 names. The only 7 misses are the same benign ones as in S15a
  (module names, None, _trs, \urlstyle, and TestCurvatureJvpOracle, which lives in tests/test_dispatch.py).
## T4S citations vs arXiv
WebFetch truncates the arXiv PDF/HTML at 85,534 chars (pp. 1-~24, through §5.3). curl to arxiv.org gets a proxy 403,
e-print is blocked by robots.txt, and ar5iv failed. Everything below was checked against §3.2 and the TOC.
- OK: Alg 9 = dense T3-SVD (tucker_tensor_train.py:4666). Alg 10 = implicit T3-SVD, App. A.2 (t3_svd.py:56,
  tv_operations.py:881, manifold.py:1433, note). Section titles: A.1 orthogonalization, A.2 unfoldings,
  A.3.1 doubled-rank tangents, §5.3.2 MC-SGD, §5.4.1 new ranks, §6.2 / 6.2.1-6.2.3 / 6.3.
- FIXED: tucker_tensor_train.py:212 "Section 2.3" -> "Section 3.2 and Appendix A.2" (arXiv §2.3 = T4S model); stale
  "local t4s.pdf is newer" dropped (entries_apply_probe.md:12, transposes.md:13); t4s.pdf -> arXiv (curvature_internals.md:21, 3 examples).
- UNVERIFIABLE (past the cut): Algs 5, 6-7, 8, 11; eqs (45)-(46), (47), (48)-(49), (50)-(53); Figs 7, 20. Consistent
  with what is visible (Algs 1-4 in §4-5.1, 9-10 in App. A -> exactly 4 boxes for §6.2.1-6.2.3, as the code numbers them).
- AMBIGUOUS (left as is): sampling_derivatives.py:1222/:2069 "t4s.pdf Thm 7" (one shared counter: Defs 1-4 by §3.2,
  Thm 14 in §7); t3_svd.py:57 + t3svd_minimal_ranks.md:4 call Alg 10 "analogous to Oseledets' TT-SVD" (arXiv: Alg 9 is
  the TT-SVD generalization; wording, not a number); the §5.4 title is unreadable in the arXiv TOC.
## E: scratchpad/impl/S13b_claude_md.patch (git apply --check OK)
The paper section now names arXiv as THE reference and notes the WebFetch truncation. The current state reads: R
cancelled, S13b/S15b done, then S14, release prep, the tag. The py3.11 / numpy 2.4 / jax 0.10.2 twin is the CI current-leg check.
## Gates (all PASS)
module doctests 225 passed on sys 3.13/np2.5/jax0.12, py311 3.11/np2.4/jax0.10.2 and j0430 3.11/np1.26/jax0.4.30.
getting_started.rst + every docs/*.md and contributor/*.md page (minus doctest_style.md) OK on all 3.
sphinx -W -q: rc 0, zero warnings. No library code changed (annotations/docstrings only): full suite + slow tier not run.
## Draft commit message
docs+backend: S13b final-review fixes and S15b final pass of the exact-Hessian math note
- rcond annotated Optional[Union[float, NDArray]] (per-block clips); apply_derivatives_model takes a uniform x;
  geometry docstring: curvature_damping raises on corewise, unused at d = 1
- 2d − 1 lift SVDs; BLAS-robust exact-vs-GN doctests (§4.12, newton_cg front/back); README exact Hessian;
  F28 in CG iterations; CHANGELOG names _assemble_tucker_variation_jets_dense
- float32: a float32 fit still upcasts upstream of the jet kernels; ledger + CHANGELOG reworded (fix deferred)
- math note: preliminary box dropped (R cancelled), refs and checks re-pinned at 76aa37f, author line
  "co-authored by Claude Fable 5.1" (Nick, 2026-10-29), T4S cited as arXiv:2603.21141; stale local-t4s.pdf
  pointers retargeted; minimal ranks cited as T4S §3.2 / App. A.2
