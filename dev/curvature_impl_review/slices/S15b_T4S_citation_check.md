# T4S citation check vs arXiv v2 (2026-10-29)
Reference: t4s.pdf = "arXiv:2603.21141v2 [math.NA] 9 Jul 2026", 81 pp (identical to scratchpad/t4s/t4s_v2.pdf).

## Table (cited -> arXiv v2 object -> verdict)
| cited in tree | v2 object | verdict |
|---|---|---|
| Alg 5, Fig 7, §6.2/6.2.1 (probing.py x6, entries_apply_probe.md, tex:509) | Alg 5 probing a T3; Fig 7 edge variables (ξ,μ,ν,η) | MATCH |
| Algs 6-7, §6.2.2 (probing.py, apply.py:158/218, manifold.py:668/796, tex:135/497/539/640) | Alg 6 base-point probe, Alg 7 tangent probe (σ,τ,δη) | MATCH |
| Alg 8, §6.2.3 (probing.py x7, manifold.py:728, transposes.md, tex:136/479/518) | Alg 8 transpose of tangent->probes map | MATCH |
| §6.3 corewise substitution P,Q,O->G (~25 sites) | §6.3 Corewise (non-manifold) simplification | MATCH |
| Alg 9 dense T3-SVD (tucker_tensor_train.py:4667) | Alg 9 | MATCH |
| Alg 10 implicit T3-SVD, App A.2 (t3_svd.py:56, tv_operations.py:881, manifold.py:1433, uniform_manifold.py:1191, utv_operations.py:340, tex:287) | Alg 10, introduced in A.2 | MATCH |
| Alg 11 sweeping orthogonalization (fv_conversions.py:169, frame_variations_format.py:1324, geometry.py:238/257, tex:115) | Alg 11 (right-to-left Q, then left-to-right P) | MATCH; divergence noted at fv_conversions, f_v_format, tex:116; ADDED at geometry.py:238 |
| "Thm 7" (sampling_derivatives.py:1222, :2069) | Thm 7 = correctness of Alg 8 (J^T as gradient of <z,J δV>) | MATCH number; "t4s.pdf" -> "T4S Thm 7, Alg. 8" |
| orthogonal reps (45)-(46) (fv_conversions.py:169, frame_variations_format.py:124/1325) | (44)-(45) | MISMATCH -> fixed |
| tangent sum eq (47) (frame_variations_format.py:1221, manifold.py:125, tex:124) | (46) | MISMATCH -> fixed |
| gauge (48)-(49) (tv_operations.py:118/168, utv_operations.py:449/505, frame_variations_format.py:124, manifold.py:643/1353, uniform_manifold.py:439/1116, tex:132) | (47)-(48) | MISMATCH -> fixed |
| doubled-rank (50)-(53) + Figure 20 (tv_operations.py:238, utv_operations.py:195/315, manifold.py:259, tex:249) | B_i (49), W_i (50)-(52); Figure 22 (Fig 20 is an Example-2 experiment plot) | MISMATCH -> fixed to (49)-(52) / Figure 22 |
| App A.1 / A.2 / A.3 / A.3.1 (many) | A.1 Orthogonalization, A.2 unfoldings, A.3 manifold+tangent (gauge, HS identity unnumbered), A.3.1 doubled rank | MATCH |
| Section 3.2 + App A.2 minimal ranks (tucker_tensor_train.py:212) | §3.2 Tucker tensor trains (minimal ranks para) | MATCH |
| §5.3.2 MC-SGD (optimizers.py:778); §5.4 / 5.4.1 / 5.4.2 (ranks.py, rank_continuation.md) | §5.3.2 Manifold Cauchy SGD; §5.4 Rank continuation, 5.4.1 new ranks, 5.4.2 initial guess | MATCH |
| user_guide.rst:394/396 (App A; §4, §6) | as stated | MATCH |
| Molozhavenko & Rakhuba Alg 1 / Thm 5; Nocedal & Wright Alg 4.1/7.2; Oseledets Thm 2.2; t3svd_verification "Algorithm 1/2"; shared_t3_math.tex Lemma 1 / Alg 3; "Batching & stacking §11" | not T4S | NOT-A-T4S-CITATION |

## Fixed (docstrings/comments only; no `>>>` lines touched)
- t3toolbox/backend/fv_conversions.py:169; t3toolbox/frame_variations_format.py:124, 1221, 1325
- t3toolbox/backend/tv_operations.py:118, 168, 238; t3toolbox/backend/utv_operations.py:195, 315 (code comment), 449, 505
- t3toolbox/manifold.py:125, 259, 643, 1353; t3toolbox/uniform_manifold.py:439, 1116
- t3toolbox/backend/sampling_derivatives.py:1222 (comment), 2069 ("t4s.pdf Thm 7" -> "T4S Thm 7, Alg. 8")
- t3toolbox/backend/geometry.py:238-239 (Alg 11 sweep-order divergence note added)
- docs/exact_hessian_two_hole.tex:124 (47)->(46); :132 (48)-(49)->(47)-(48); :249 (50)-(53)->(49)-(52);
  :12-14 header now "T4S citations verified against arXiv v2 (2603.21141v2) on 2026-10-29; 3 fixed";
  :1050 bibitem now arXiv:2603.21141v2 + v2 URL. PDF rebuilt from the final .tex (0 warnings, aux cleaned).

## Left for the orchestrator (docs/*.md, CLAUDE.md -- not edited)
- docs/entries_apply_probe.md:233 gauge "(48)–(49)" -> "(47)–(48)"; :230-231 "numbers per the local t4s.pdf" -> "per arXiv v2" (numbers there match v2).
- docs/numerical_contracts.md:52 and docs/contributor/numerical_contract_catalog.md:57 "(48)–(49)" -> "(47)–(48)".
- docs/contributor/testing_strategy.md:41 "eqs 50–53" -> "eqs (49)–(52)".
- docs/rank_continuation.md:11 "local t4s.pdf numbering is newer than the public arXiv" -- stale; drop (v2 is current; §5.4.x match).
- CLAUDE.md:71-74 "local t4s.pdf is newer ... cite numbers as in the local t4s.pdf" -- stale; arXiv v2 is authoritative and equal to t4s.pdf.
- Wording only (no number wrong): t3_svd.py:57, docs/t3svd_minimal_ranks.md:4, docs/contributor/t3svd_design_rationale.md:42 call Alg 10
  "Oseledets' TT-SVD"; v2 says Alg 9 (dense) generalizes TT-SVD and Alg 10 "mirrors the dense procedure" on cores. Left as is.

## Gates
- `pytest --doctest-modules t3toolbox`: 225 passed. `sphinx-build -W -q`: rc 0 (first attempt hit a FileNotFoundError race
  in docs/autoapi from a concurrent build by the other reviewer; clean rerun passed). latexmk: rc 0, 0 warnings, only .tex/.pdf left.
- Nothing committed.
