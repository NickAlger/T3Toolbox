# Response to review r2 of `PART1_PLAN.md`

Writer's response, 2026-11-16. The plan is now draft r3. Every finding was checked against the code first. All
seven are accepted; for finding 21 one cited caller was found not to be one.

| # | Verdict | Where it is resolved in r3 |
| --- | --- | --- |
| 21 | Accepted, with one correction. The callers are verified. `backend/sharing.py:876` is not a caller: its doctest calls `sg.shared_manifold(...).frame(x)`, a frontend geometry, which does not change. | P15 lists every caller of a backend geometry's `frame`. It drops the frontend `fitting.py`, and says the frontend geometries are untouched. Also found: `tests/backend/test_padded_start.py:725`, a test subclass's `aligned_padded_frame`. |
| 22 | Accepted. Verified `tests/test_manifold.py:994`, `test_resolver` (`tests/test_chunk_size_estimator.py:246`), and `uniform_fitting.py:476` (no `from __future__ import annotations`). | P16 lists all three; `uniform_least_squares_problem` goes in P16 with its doctest. Section 6 table updated. |
| 23 | Accepted. Re-read "The two parts" as it stands now. | O-1: the stale parenthetical removed. O-19: restated to match the decisions' correction (open, both costs named). Section 1, section 2.3 step 6, section 4, P11 and section 7.2 no longer follow one form. Every slice now names the open points it needs: a "Needs" column in 5.2, and a line at the head of each slice. |
| 24 | Accepted. Verified `scripts/b01_backend_bench.py:136-142`, `:270-273`. | Section 10 item 3. |
| 25 | Accepted. | O-22 narrowed to the corewise Hessian. Section 2.4 carries out the decided part for the transposes, by 6.6 with 2.4 and 3.6. |
| 26 | Accepted. Verified `backend/fitting.py:538`, `565`, `604`. | Section 5.1, "The bridge". |
| 27 | Accepted. | Section 5.1 names P12, P13, P15, P16. `test_newton_cg_exact_across_chunk_size` is kept until P16, fed through `taylor_to_derivatives(x.probe(...))` (P13, P16, section 6). |

## The searches, per slice

Each was run on `t3toolbox/`, `tests/`, `docs/`, `examples/` and `CLAUDE.md`. Python files were read with an `ast`
scan of calls, which records the keywords each call passes. The same scan was run over every module doctest and every
page doctest, through `doctest.DocTestParser`. A text grep caught the rest: calls inside strings and subprocess
scripts, and calls by reference.

- **P12** (the 30 changed names). What I looked for: callers passing `sum_over_probes=`; callers relying on a transpose's default; positional calls.
  - Beyond the reviewer's items: `tests/backend/test_from_sweep_reuse.py` passes `sum_over_probes=` to the changed `tv_*_transpose` six times. It stays until P19, so it must be updated in P12.
  - Also found: callers in `tests/test_curvature.py`, `test_dispatch.py`, `test_manifold.py` and `test_uniform_probing.py`; the four frontend modules' call sites.
  - No page doctest calls a changed name; prose mentions go to P18.
  - The positional signatures differ between the functions: `t3_apply(x, vecs)` but `t3_probe(ww, x)`, and `tv_probe(ww, variation, frame)` but `utv_probe(ww, frame_data, variations_data)`. The r2 sketch of `tv_probe` had the order wrong. Section 2.4 and N16 now keep each function's positional order and names.
- **P13** (the 24 removed methods; the frontend transposes' keyword and default; links to `docs/chunking.md`).
  - Beyond the reviewer's items: `tests/test_jax_absent.py` calls `probe_derivatives` inside its subprocess script (lines 66 and 79), which the call scan cannot see.
  - The `:doc:` links to `/chunking` sit in docstrings as well as pages: `t3toolbox/optimizers.py` (three), `manifold.py` (one), and `backend/sampling_derivatives.py` (four). The last stays until P19, so its links must be cut in P13, or `sphinx -W` fails.
  - The frontend transposes' callers that pass the keyword or rely on the default.
  - Corrections of over-inclusion: r2's list named files with no call of a removed method (`test_uniform_tucker_tensor_train.py`, `test_uniform_manifold.py`, `test_oracle_sweep.py`, `test_optimizer_display.py`, `sweep_exact.py`, `sweep_optimizers.py`, `oracle.py`; they hold kind strings, which change in P16). `docs/entries_apply_probe.md` names the methods in prose only (P18).
- **P15** (every call of a backend geometry's `frame`). I separated it from calls of the frontend geometries.
  - Beyond the reviewer's items: `tests/backend/test_padded_start.py:333` and `:428`, `:524`, `:725`; `tests/backend/test_zero_reference.py:222` (a wrapper geometry); `tests/backend/test_uniform_fitting.py:57` (`_compare_ops`); `tests/backend/test_geometry.py` (five calls).
  - No module or page doctest calls a backend `frame`.
- **P16** (the names in this list).
  - What I searched: `Problem`; `least_squares_problem` (both layers); `LeastSquaresSetup`; the six factories; the four optimizers' calls; `flat_draw`; `make_newton_display`; `local_model`; `_resolve_chunk_size`; `_canonical_weight`; `_require_exact_capable`; `_minibatch_step_problem`; the `_derivatives` kind strings; `CorewiseGeometryOps` and its uniform twin.
  - Beyond the reviewer's items: `backend/optimizer_display.py`'s doctest builds `PhaseSeconds` with a `sweep` field, which O-17 would change; `tests/oracle_sweeps/sweep_exact.py`, `sweep_optimizers.py` and `oracle.py` use the kind strings.
  - The page doctests that call an optimizer or a factory are those of `fitting_and_optimization.md`, `getting_started.rst`, `sharing.md` and `numerical_contracts.md`. r2 also named `rank_continuation.md` and `aligned_padded_frame.md`, which have none.
  - Thirteen of the fourteen examples use the fitting API; `fit_hilbert_from_apply_derivatives_flat.py` does not.
  - `docs/conf.py` is not nitpicky, so a stale `:py:` reference does not fail `sphinx -W`, while a stale `:doc:` link does.
- **P19** (the 129 removed names, and `sampling_derivatives` as a module). Every file that still references one, with the slice that removes the reference, is now listed in section 10. Those left at P19:
  - the comparisons in the curvature tests;
  - the old rows of `test_dispatch.py`;
  - the four old-kernel test files.

## Open points changed

- O-1 and O-19 rewritten (finding 23).
- O-22 narrowed (finding 25).
- Section 11's introduction points to the "Needs" column.

No open point was added or removed.
