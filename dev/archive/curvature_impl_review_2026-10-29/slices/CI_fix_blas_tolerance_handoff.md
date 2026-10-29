# CI fix 2 (tests workflow, both legs) -- base 9f72a97, UNCOMMITTED: tests/backend/test_uniform_curvature.py only (+33/-3, diff --check OK)
**Failure:** TestUniformPlainKindsExactLocalModel::test_every_exact_action_per_element (entries, manifold, 'd3 varying', K=(2,)),
'hvp 1' 1.207e-13 > 1e-13. DETERMINISTIC, BLAS-order dependent; reproduced in py311latest (np 2.4.6) at 1.21e-13; floor twin
8.5e-14, sys np 2.5.3 3.9e-14. Not jax, not a library bug. The floor "flake" = whether that leg reached it.
**Root cause:** `_compare` was relative to each PART's own norm (Tucker / TT separately), but an exact hvp is a cancelled
sum. Worst element: Hp TT 26.1, Tucker 261; TT terms L^T K_a th 208, L^T K_g th 198 (raw K_a th / K_g th 144 each), GN 0.15;
the lift amplifies x125 (|th|/|Pi p|). Abs error ~3e-12 in BOTH parts: the terms set it, not the TT block. Whole-tangent
normalisation alone was rejected (floor worst 5.3e-14, 1.9x margin).
**Fix:** `_compare(..., scale_of=None)` (:172, :190); new `_hvp_term_scale` (:205-226) = per K element |GN| + |L^T K_a th| +
|L^T K_g th| over the whole tangent (curvature_hvp: without GN), from the ragged model's public geometry/kind members;
`_per_element` (:1320-1324) uses it for hvp / curvature_hvp (plain + derivative-kind classes, both tests). Bar stays 1e-13
(S12's quadratic-rebase principle: under cancellation, round-off is relative to the summands). New worst 1.32e-14 (np 1.26) /
8.5e-15 (2.4) / 4.5e-15 (2.5), so >= 7.6x margin; the CI subtest is now 7e-15. Still discriminating: terms/|Hp| <= 45, so the
bar is <= 4.5e-12 of |Hp|. Unchanged: the quadratic, gradient and mask checks; the same pattern in TestUniformExactLocalModel
:1156 / TestUniformSharedExactModel :1872 (own loops, >= 6x margin). test_curvature.py has no `_compare`.
Floor-only note: every 'tt slack' manifold subtest errs ~50x more than on 2.x (probe 7.1e-14 on the old bar; 1.2e-14 now).
**Gates:** test_uniform_curvature.py 57 passed, 2 skipped / 383 sub, and test_curvature.py 97 / 1416 sub, in ALL THREE envs
(py311latest, jax0430, sys py3.13). Full-suite halves in py311latest (-n 2): 414 passed, 2 skipped / 3270 sub + 756 passed,
2 skipped / 43094 sub, 1 adam warning (by design) = the S13 counts. Scripts + logs: scratchpad/ci2/.
**Draft commit:** tests: the uniform exact-hvp per-element bar is relative to the hvp's terms, not to each block (CI)
| The tests workflow has failed since a2fc053 (S12) on one subtest, TestUniformPlainKindsExactLocalModel (entries, manifold,
| 'd3 varying', K=(2,)): 'hvp 1' 1.207e-13 vs 1e-13 on both legs. Uniform and ragged agree to round-off; `_compare` measured
| the error against the TT block alone, and an exact hvp is a cancelled sum: a TT block of 26 against TT terms of 208 and 198
| (L^T K_a th, L^T K_g th) and a Tucker block of 261, so the error moves with BLAS order (3.9e-14 numpy 2.5, 1.2e-13 2.4).
| The hvp / curvature_hvp bar is now 1e-13 of |GN| + |L^T K_a th| + |L^T K_g th|: worst 1.3e-14 (1.26) / 8.5e-15 (2.x). Test-only.
| (+ the two attribution lines from the session reminder)
