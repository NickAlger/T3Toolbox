# Reviewer carryover (K arc), written after K0's design review r2

Claude (reviewer), 2026-11-25. This is what a successor needs that the slice records do not already hold. Read it
with `K_BRIEF.md`, `BRIEF.md` ("What the reviewer does"), `K0_DESIGN_NOTE.md` (revision 2 or later),
`K0_DESIGN_REVIEW_r1.md`, `K0_DESIGN_REVIEW_r2.md` and `k0/REVIEWER_PRENOTES.md`.

## Where K0 stands

The plan phase ended at round 3, SIGN OFF (`K0_DESIGN_REVIEW_r3.md`), with one wording edit to apply in the build:
`fitting`'s module docstring calls `Π` "the gauge projection", but on the corewise geometry `project` is the identity,
or the per-group mean when the Tucker factors are tied. The exact replacement text is in r3, M1. Check that it was
applied when reviewing the build.

## Tools (scratch: `/tmp/claude-1000/-home-nick-repos-T3Toolbox/7a03b046-9177-4c43-923a-b31595edb27e/scratchpad/reviewer/k0/`)

The scratch folder can be cleared; rewrite a tool from this description if it is gone.

- `scan.py`: my own `ast` inventory. It reads every module-level statement, closes over private orphans
  transitively, and writes `deleted_names.json` (313 names; the note's `k0_names.json` has 315, the extra two being
  `_t3_apply_step` and `_t3_entries_step`, deleted by decision).
- `mysim.py OUT [x]`: copies `t3toolbox/` and `tests/` to OUT, removes `sampling_derivatives.py` and every deleted unit
  (by `ast` line range), the dead imports and the `__all__` entries. Pass a second argument to also delete the two dead
  scan bodies. It does not delete comment blocks. On that tree, today's tests gave 11 files failing at import (exactly
  the note's) and 6 failures elsewhere (exactly the note's), in 14 minutes at `-n 2`.
- `import_all.py [nojax]`: imports every module with `pkgutil`; `nojax` hides jax. 59 modules today, 58 after K0.
- `tables_scan.py`, `tests_scan.py`, `doctest_scan.py`: module- and class-level test tables; `%s` templates in tests;
  doctest lines naming deleted names.
- `jvp_rebase.py`: the re-based jvp oracle at `order=(2,)` (probe, apply, uniform probe). Mutate a step by appending
  a wrapper to `sampling_steps` in a scratch copy, as `jvp_mut2.py` does, or monkeypatching before `exec`.
- `f32_x64.py`, `f32_mutant.py`: the engine's float32 under x64, and the `start_zeros` mutant without `dtype`.
- `base_nodes.txt`: the 1,985 nodes of the baseline log with their outcomes.

## What I verified, so a successor need not redo it

- The 98 deleted tests all PASS at baseline. The expected count is 1,882 (1,979 - 98 + 1).
- The `start_zeros` dtype mutant is caught by the surviving `test_sampling_stacks.TestTheDtypeMatrix`. My pre-note
  claimed it would be unguarded; that was wrong.
- A `Kp` mutant (`H_tilde_dot_step`) is caught by `TestByDefinition` and by the kept block-sum test.
  `H_tilde_step` reaches apply's `Kp`; `V_top_step` is invisible to the identity tests (pinned by
  `test_sampling_apply.TestByDefinition`).
- Mutation (i) of note r1, the square dropped in the test's own weight, survives `test_residual_weights`: the
  identity holds for any diagonal weight.
- E1, E1c and E2 reproduce. The direction mask is not load-bearing on the real block in general position.
  Parametrizing `TestKpAtTTSlack` over the three kinds needs real harness work (`ttslack_kinds.py` errors): K5's job,
  by ruling.
- `docs/conf.py` is not nitpicky, so Sphinx `-W` never catches a dangling `:py:` role. Check pages by script (import
  plus `getattr` per role).
- No kept doctest of the seven modules reads a module global that the note's import cleanup removes.

## What to check in K0's build review

- The prose against `prose_edits.md`, word for word. The deleted comment blocks against the note's ranges, especially
  that no section banner of kept code is lost. Signature columns and diagrams of kept functions untouched (diff them).
- `k0_inventory.py --check` gives 0. Rerun my `scan.py` on the built tree: it should find nothing left of the 315.
- `import_all.py` both ways on the built tree.
- The tally against `k0/baseline_v.log.gz` (renames mapped). Each edited test's subtest count as stated in the
  handover.
- The mutations of note §6, rerun on the **built** tests, not the prototypes. Add mine: `H_tilde_step` on the jvp
  apply row; `start_zeros` dtype on the re-based dtype test; the hygiene control with masks in place (must pass) and
  removed (must raise).
- The re-based `test_tucker_tensor_train` forward uses `dense_*` and inline factorials, no `taylor_to_derivatives`.
- The negative control patches `_UNIFORM_MASKS`, `_mask_variable` and `ufv_masking.ufv_apply_variations_masks`, and
  asserts `'padding not zero'`.
- The slow tier of `test_uniform_curvature.py` was run before the deletion (the recorded step), and again in the gate.
- `test_import_graph.py`'s module docstring mentions the new test.

## Carried to later slices

- K5: point the hygiene negative control's patches at K5's masking points, and add the apply and entries cases of
  `TestKpAtTTSlack` first, against the engine as it then is, before changing the masking (ruling, `K_ARC_DECISIONS.md`).
- P17: `sum_over_probes` in `tests/_derivative_data.py` and its callers, by name.
- P19: the pages' prose listed in note §2.2, including `signature_style.md`/`reading_signatures.md` quoting
  `compute_mu`; adding new names to `api_reference.rst`.
