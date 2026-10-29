# S13b + S15b review (main @ 76aa37f + 20 uncommitted files) -- verdict: APPROVE-WITH-FIXES (2 small, applied)

## Fixes applied by the reviewer (tree still uncommitted)
1. `docs/exact_hessian_two_hole.tex:12-14` (a TeX comment, so it does not render and the PDF needs no rebuild): the new comment said
   "T4S numbers are those of the arXiv version", which the handoff itself could not verify for the appendix. It now reads
   "cited as arXiv:2603.21141 (its appendix numbers -- Algs 5-8, 11; eqs (47)-(53) -- not yet re-verified against the arXiv text)".
2. `t3toolbox/tucker_tensor_train.py:212-213`: rewrapped the reworded "Section 3.2 and Appendix A.2" docstring line (137 chars) to two lines.

## 1. No library behaviour changed
Every hunk under `t3toolbox/` (7 files) touches only docstrings, doctest text, signature annotations or comments (`rcond` Union,
the `apply_derivatives_model` x/geometry annotation). No executable statement changed, so I skipped the bit-identity run.
The module doctests still run both a GN and an exact `newton_cg` and pass on all 3 environments.

## 2. Review items (code report A, docs report B)
- A1 rcond: FIXED. `linalg.py:722/778` and `sharing.py:647` read `Optional[Union[float, NDArray]]` and document the
  "scalar or broadcastable to C+(1,)" case. They sit in their own blank-line signature group, per signature_style.
- A2 apply_derivatives_model: FIXED. `fitting.py:802-803` is now identical to `entries_derivatives_model` (:860-861), and the body (:847) dispatches uniform.
- A3 curvature_damping: FIXED. `geometry.py:44-46` now says it raises on corewise, and on the manifold at d = 1 it is validated (>= 0) and unused (data `None`).
- A4 float32 (decision B): FIXED as a ledger entry plus a reworded CHANGELOG. `deferred_and_rejected.md:252-264` and CHANGELOG:156-162 are accurate.
  I re-checked: a numpy float32 `apply` fit returns float64 on MANIFOLD and COREWISE, under both GN and exact
  ("every manifold fit and most others" holds). `float32_followup.patch` is NOT applied: there is no diff in tt_operations or
  apply/entries/probing.
- B1 2d−1: FIXED in both places (`curvature_internals.md:143`, `fitting_and_optimization.md:592`), and it matches `fv_lift_factors`.
- B2 §4.12 doctest: FIXED, robust. Measured on sys / py311 / j0430:
  - ex steps are 6 / 6 / 6, with stop reasons gradient / objective_floor / gradient.
  - GN steps are 15 / 20 / 16.
  - rel(ex, 5) = 1.05e-10 on all three, against the 1e-8 bar (100× margin).
  - rel(gn, 5) = 1.41e-3 on all three.
  - The bound `ex <= 10` leaves 4 steps of slack.
  - The `history[5]` index always exists, because ex needs at least 6 steps to get below 1e-5.
- B3 module doctests: FIXED, robust. They now use the default gtol and the `in (...)` stop test.
  - ex: 5 steps, 'gradient' (2.3e-7 → ≤7.6e-14).
  - GN: 28 line_search / 30 gradient / 29 gradient. A one-step BLAS change cannot flip `gn > 2*ex` (needs > 10).
  - The new prose ("about halving ‖g‖ per step") is true: 3.25e-2 at step 5 is a factor of 0.50 per step.
- B3 scan of the other arc doctests: `optimizers.py:262/274` and `fitting_and_optimization.md:689` all assert 'gradient' on noise-free fits.
  They end with a jump far past gtol (to 1.7e-9, 3.6e-11 and 2.1e-11 against 1e-8), identically on all 3 environments. Not fragile.
- B4 CLAUDE.md: FIXED via the patch (see below).
- B5 math-note banner: FIXED (see §4).
- B6 README: FIXED (`README.md:108-113`); the sentence is accurate.
- B7 F28: FIXED. It reads "78 CG iterations" in both pages, plus 2.6e-9 (matches verification_records).
- B8 CHANGELOG: FIXED (`CHANGELOG.md:101-103`); `_assemble_tucker_variation_jets_dense` exists.
- Extra stale references, all FIXED: the `t4s.pdf` mentions in `entries_apply_probe.md`, `transposes.md`, `curvature_internals.md:21` and the 3 examples.

## 4. Math note
- The PRELIMINARY box and the header comment are gone.
- Author line: `...collaborators\\ co-authored by Claude Fable 5.1`.
- The T4S bib entry is the plain arXiv citation plus its URL.
- The commit is re-pinned to 76aa37f at :1002 and :1030.
- `grep -in preliminary|S15a|S15b|reorganiz|t4s.pdf` finds nothing.
- PDF: 16 pp. A scratch rebuild from this .tex gives text identical to the committed PDF (pdftotext diff empty).
  The title page renders correctly.
- No aux files in `docs/` and no untracked files.

## Remaining T4S-unverified (handoff lists them exactly; nothing changed on a guess)
- Not checkable: Algs 5, 6-7, 8, 11; eqs (45)-(46), (47), (48)-(49), (50)-(53); Figs 7, 20. All are past WebFetch's ~p.24 cut.
- Ambiguous, left as is: `sampling_derivatives.py:1222/:2069` "Thm 7"; the "analogous to Oseledets' TT-SVD" wording at
  `t3_svd.py:57` and `t3svd_minimal_ranks.md:4`; the §5.4 title.
- Verified by the implementer: Algs 9-10, A.1/A.2/A.3.1, §5.3.2, §5.4.1, §6.2-6.3, and §3.2 (the tucker_tensor_train fix).

## CLAUDE.md patch: APPROVE
- `git apply --check` passes.
- Current state: R cancelled; S13b and S15b done → S14 → release prep → tag.
- The T4S section: arXiv is THE reference, the local-newer claim is dropped, and a WebFetch truncation caveat is added.
- The py3.11 / np2.4 / jax0.10.2 twin is noted as the local check of the CI current leg.
- Nothing true is removed and nothing is bloated.
- Nit, optional: the "What this is" line 13 still says "with only minor revisions still in flight".

## Gates (re-run by me, after my fixes)
- Module doctests: 225 passed on each environment.
  - sys 3.13 / np 2.5.3 / jax 0.12
  - py311 3.11 / np 2.4.6 / jax 0.10.2
  - j0430 3.11 / np 1.26.4 / jax 0.4.30
- `getting_started.rst` and every docs/*.md and contributor/*.md page (minus doctest_style.md): OK on all 3.
- `sphinx-build -W -q`: rc 0. `git diff --check`: clean.
