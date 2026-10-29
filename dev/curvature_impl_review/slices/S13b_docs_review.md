# S13b reviewer B: the documents checked against the code (main @ ede08fd)

I report clear errors only. I re-ran each finding against the code or with a short script from the repo root.

## Findings (most severe first)

1. **FALSE-STATEMENT (known, still wrong): "d small SVDs".** `docs/contributor/curvature_internals.md:143` and
   `docs/fitting_and_optimization.md:589`. Evidence: `fv_lift_factors` (`backend/fv_operations.py:624-625`) builds
   `d` Tucker pseudo-inverses and `d-1` bond pseudo-inverses, which is 2d-1 SVDs. The math note
   (`exact_hessian_two_hole.tex:451`) already says `$2d-1$`. Fix: change "`d` small SVDs" to "`2d − 1` small SVDs" in both places.

2. **FRAGILE-DOCTEST: the §4.12 example passes by one step.** `docs/fitting_and_optimization.md:557`
   asserts `ex['newton_steps'] < gn['newton_steps'] / 2`. On this machine Gauss-Newton stops on `'line_search'`
   at 15 steps (‖g‖/‖g₀‖ = 5e-9 against gtol 1e-10) and exact stops on `'gradient'` at 6. The bound is therefore 7.5.
   - Per a2d39c8, the CI BLAS gives exact `'objective_floor'`. That adds one rejected row, so exact takes about 7 steps, and the assertion then needs Gauss-Newton ≥ 15.
   - Gauss-Newton's step count is wherever its Armijo search fails at the resolution floor, and that point depends on the BLAS.
   - So a Gauss-Newton run that stops one step earlier fails the doc build.
   Fix: compare with slack (for example `ex['newton_steps'] <= 0.6 * gn['newton_steps']`), or compare progress
   (`ex` reaches gnorm ≤ X in k steps while `gn` does not).

3. **FRAGILE-DOCTEST (low): the same stop-reason pattern in the module doctests, plus prose that depends on the BLAS.**
   - `t3toolbox/optimizers.py:546` and `t3toolbox/backend/optimizers.py:1260` assert `ex['stop_reason'] == 'gradient'` at
     `gtol_rel=1e-10`. This is the same pattern a2d39c8 fixed in the page. Locally it is safe: exact jumps from 2.3e-7 to 6.9e-14, and `gn` at 28 steps is well over 2·5.
   - The prose at `optimizers.py:529-530`, "Gauss-Newton ... ends here on its line search", is the outcome that flipped to `'gradient'` on the CI BLAS in the page example.
   Fix: assert `in ('gradient', 'objective_floor')` as the page now does, and soften the prose to "stalls short of this tight `gtol_rel`".

4. **STALE (the reorganization R is cancelled): CLAUDE.md:403-406.** It says "next a preliminary math note (S15a), then a PAUSE,
   then the reorganization R (`dev/reorg_scope.md`) ... the math note's / CHANGELOG's final pass for R (S15b)".
   S15a is done (78580d8) and R is cancelled (ede08fd). Fix: "next: S13b review → S14 → S15b + release prep → tag".

5. **STALE (the reorganization R is cancelled): the math note's banner.** `docs/exact_hessian_two_hole.tex:84-88` (and the comment at :12-13) say
   "The module layout is about to be reorganized; the references are re-checked, and renamed where needed, in the
   final pass (S15b)". The committed `exact_hessian_two_hole.pdf` carries the same text. Fix: in S15b, drop the
   reorganization sentence and the "preliminary" banner, re-pin the commit, and rebuild the PDF.

6. **STALE (low): the README feature list.** `README.md:108-109` says "Least-squares fitting ... (Gauss-Newton models) with four
   optimizers". The headline of 2026.3.0, `hessian='exact'` with a trust region, is absent, and the arc did not touch the README.
   Fix: "(Gauss-Newton or exact-Hessian local models; `newton_cg(..., hessian='exact')` with a trust region)".

7. **INCONSISTENT-NUMBERS (low): F28 units.** `curvature_internals.md` (F28/F38 row) gives "Tucker slack 20 outer / 4 rejected /
   78 **hvps**". `verification_records.md` gives the same triple as "outer / rejected / **CG**" ("78 CG"). In exact mode the two
   counts differ, because each row also evaluates `pHp`. Fix: use one unit in both, matching the source measurement.

8. **FALSE-STATEMENT (minor): the CHANGELOG on `assemble_tucker_variation_jets_trs`.** CHANGELOG `[2026.3.0]` (the jet-kernel
   bullet) says "`assemble_tucker_variation_jets_trs` (which contracted no `trs`) is private". There is no
   `_assemble_tucker_variation_jets_trs`. It is now `_assemble_tucker_variation_jets_dense`
   (`sampling_derivatives.py:1474`), and `naming_conventions.md` correctly says that no `_trs` function exists in the package.
   Fix: "is removed (private as `_assemble_tucker_variation_jets_dense`); pass `chunk_size=None` for the unchunked assembly".

## Checked and clean

- **Defaults and signatures.** `newton_cg` frontend and backend: `hessian='gauss_newton'`, `step_control=None` (line search for GN,
  trust region for exact), `tr_radius0=None` (the GN Cauchy length `‖g‖³/⟨g,H_GN g⟩`), `curvature_damping=None` (resolves to
  `DEFAULT_CURVATURE_DAMPING=1e-6`), `cg_forcing_power=0.5`, `max_newton=30`, `gtol_rel=1e-8`. The frontend takes explicit keywords
  only; `gradient_descent`/`mc_sgd`/`adam` still take `**kwargs`; `g0_reference` is on `gradient_descent`.
- **Trust region.** The constants (0.1/0.25/0.75, ¼‖p‖, 2Δ, 10⁴Δ0), the floors checked only after a rejection, the
  `objective_floor` formula, and `objective_floor` not counting as a failure all agree across the CHANGELOG, §4.12,
  curvature_internals, the math note and `NewtonInfo`.
- **Stats and history.** The stats keys (`losses`/`newton_steps`/`newton_accepted`/`stop_reason`/`history`, plus `diagnostics` when verbose)
  and the `NewtonInfo` fields match the code and the §2.1 table.
- **What raises.** Each structural `ValueError` the docs list raises as documented, as checked by running each case:
  - `curvature_damping` with GN, and on a corewise geometry;
  - `tr_radius0` with the line search;
  - `'accept'` with the trust region;
  - `g0norm_newton=0`;
  - `g0_reference='zero'` on a corewise geometry;
  - `curvature_hessian` on a GN model.
- **g0 reference resolution.** `None` resolves to `'zero'` on the manifold and to `'warm_start'` on corewise or without `zero_point_frame`.
  An override skips computing the zero reference. Cold and warm starts at the same ranks give an identical `g0norm`
  (46715.72 for both), which confirms the claims in `rank_continuation.md` and the release notes.
- **Chunking.** `chunk_size='auto'` chunks only a uniform `probe_derivatives` fit, in both modes. `estimate_chunk_size(hessian='exact')` is never
  larger than the GN answer (it takes the `min`). `newton_cg` passes its mode to `least_squares_problem`.
- **Release-note upgrade steps, against the code.**
  - `GaussNewtonModel` is gone and `t3toolbox.LocalModel` exists.
  - `_setup` is gone, and `least_squares_problem` takes the stated arguments and is not exported at the root.
  - The sixth operation is checked only in exact mode, through the abstractmethod marker.
  - The four geometry members are checked as listed, and `zero_point_frame` is optional.
- **Code names.** All 219 `\code{}` names in the math note resolve to real defs, files or tests. The math note's figures agree with
  verification_records: 720 / 2.5e-14 and 8000 / 2.6e-11; 2.8–3.9×; about 2× memory.
- **Numbers that agree across documents.**
  - F19/F26/F27/F33.
  - 3.5–5× / 5–6.6× / 1.6–2×.
  - 12 % basin (1.108/0.987).
  - Compile time 1.5× (~14.7 s / 9.5 s).
  - Memory 1.85–2.1× and 1.3–2.2× per row.
- **Docs build.** `sphinx-build` reports zero warnings. The new toctree entry `contributor/curvature_internals` resolves; the
  anchor `#the-gauge-check-scale-own-norm-or-operand-scale` resolves; every relative `.md` link in the touched pages resolves.
- **Doctest margins.** The numerical_contracts FD-gauge doctest has ample margin (1.4e-16 against a 1e-13 bar; 2.1e-2 against `> 1e-9`). The §5
  `('zero','gradient')` doctest also has margin (4e-11 at step 12).
- **Out of scope here.** `pyproject.toml`/`__init__` still read 2026.2.0, which is a release-prep item.
- **Not re-reported.** Two items are already in DEFERRED.md for Nick: the 1.25–1.61× memory gate and the F1 FD-gauge limit.
