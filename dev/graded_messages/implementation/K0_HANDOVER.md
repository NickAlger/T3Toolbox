# K0 handover: the deletion of the old sampling kernels

Implementer, 2026-11-25. Built on `graded-messages` at the approved design (`K0_DESIGN_NOTE.md`, revision 2, signed off
in `K0_DESIGN_REVIEW_r3.md`, approved by the orchestrator), from the tree of `b5125f76` (library and tests as at
`3375d213`). Nothing committed. Working files: `implementation/k0/`.

## 1. What changed

**Library** (32 files changed in all, 378 insertions, 11,434 deletions; `git diff --stat`):

- `t3toolbox/backend/sampling_derivatives.py`: **deleted** (3,534 lines).
- The seven modules (`probing`, `apply`, `entries`, `ut3_sampling`, `utv_sampling`, `fitting`, `uniform_fitting`):
  the 161 names of the note §1.2 cut, with their `__all__` entries, the imports the note lists, and the module-level
  comment blocks the note deletes whole. Done by `k0/delete_library.py` (it refuses a file that differs from
  `b5125f76`, and checks the first line of every block it cuts). Lines (`wc -l`, after the rewrites): `probing` 1,827 → 605,
  `apply` 702 → 436, `entries` 573 → 386, `ut3_sampling` 744 → 312, `utv_sampling` 591 → 168, `fitting` 1,240 → 550,
  `uniform_fitting` 518 → 149.
- The rewrites of `k0/prose_edits.md` (the seven module docstrings that change, the section headers kept or rewritten,
  the comments cut to what stays, `entries.py`'s comment, `common.readonly_mask_copies`, `utv_sampling`'s
  `sum_over_samples`, `probing.__all__`'s section comments, `docs/api_reference.rst`), applied by
  `k0/apply_prose.py`, each an exact replacement of its before-text. `fitting`'s module docstring is round 3's final
  text (`Jp = 𝒥p`, `g = 𝒥ᵀ(M r)`, `JtJp = 𝒥ᵀ M 𝒥p`, checked against `sampling_definitions.g` / `JtJp` and the oracle's
  `_SCALED`: the minibatch scale multiplies `g` and `JtJp`, not `Jp`; the projection on both geometries).
- Two more imports than the note's list: `contractions` in `apply.py` and `entries.py` (only the dead scan bodies
  `_t3_apply_step`, `_t3_entries_step` read it; section 4).
- Three double blank lines after `probing`'s banners collapsed to one (the house form, banner then the function).
- `docs/api_reference.rst`: the six model factories' bullet and the `least_squares_problem` bullet removed, `:130`
  loses `sampling_derivatives`.
- `dev/graded_messages/README.md`: the O-26 line (the scripts of `scripts/` run against `v2026.5.0`).

**Tests.** Deleted whole: `tests/bridge/test_old_kernels_bridge.py` (and the folder),
`tests/backend/test_from_sweep_reuse.py`, `tests/test_chunk_size_estimator.py`, `tests/test_jet_recurrence.py`,
`tests/test_probe_derivatives.py`, the helper `tests/_jet_trs_twins.py`. Edited, as the note §3 says:

- `tests/backend/test_curvature.py`: 21 tests deleted (`TestProbeDerivativesTwoHole`, `TestDerivativeKindsTwoHole`,
  three of `TestPlainKindsTwoHole`), six orphaned helpers; `_plain_ops` returns `(two-hole, one-hole)`;
  `test_residual_weights` re-based (the engine's forwards, the weight written out); the module docstring's T5 / T6 /
  T11 bullets cut to what is tested; the S12 section header renamed for what it heads; `sd` and `bfit` imports gone.
- `tests/backend/_curvature_oracles.py`: the twin family and its paragraph; the imports it alone used.
- `tests/backend/test_uniform_curvature.py`: 14 tests deleted, five helpers; `test_per_element_vs_ragged` loses its
  sweep lines; three section headers renamed for the helpers they now head; the docstring's pieces list; `sd`, `bfit`,
  `utvs` imports.
- `tests/backend/test_uniform_fitting.py`: 11 tests deleted, three helpers, two imports.
- `tests/test_dispatch.py`: `test_jit_probe_derivatives` → `test_jit_the_interpreter_on_multi_group_strings`
  (the seven interpreter checks, the binomial tensor inlined); `test_jit_probe_derivatives_two_hole` deleted;
  `test_jit_plain_kind_two_holes`, `test_jit_derivative_kind_two_holes`, `test_jit_uniform_exact_ops` edited;
  `test_jit_shared_uniform_exact` re-based on `dd.forward`; the parameterized-kind message; `TestJetDtypeUnderX64`
  re-based on the engine's faces (ten ops, ragged and uniform, numpy, jax, jit); `TestCurvatureJvpOracle`'s derivative
  rows re-based at a line, the uniform one renamed `test_uniform_probe_two_hole_at_jets_is_the_jvp_of_its_transpose`;
  `import math` added, `pd` and `twins` gone.
- `tests/test_uniform_probing.py`: `TestTransposeOutputHygiene` rewritten as the note §3.6 says.
- `tests/test_contractions_sharding.py`: `TestJetKernelsShardOverW` deleted, its `setUpClass` and `_check` moved
  verbatim into `TestTheEngineShardsOverW` (which borrowed them; section 4); the floors (section 4).
- `tests/test_contractions_interpreter.py`: the floor 100 → 60 (the comment gives 84 at K0, measured on the built
  tree; review r1 m1).
- After review r1: `tests/test_dispatch.py`'s `test_jit_uniform_derivatives` comments (m3); two docstrings that
  described the deleted code as present, `tests/test_transposes_typed.py:6-7` and `tests/backend/test_curvature.py`
  `TestRequiredSixthOperation` (the sweep of m3); a docstring line of `test_uniform_curvature.py` rewrapped (m2)
  (`K0_RESPONSE_r1.md`).
- `tests/test_least_squares_oracle.py`, `tests/test_sampling_definitions.py`: one class each deleted, with their lines
  of the module docstrings; `test_the_jets_of_entries_off_the_grid` re-based on the subset-sum definition.
- `tests/test_tucker_tensor_train.py`: the two finite-difference forwards dense, factorials inline.
- `tests/test_jets.py`: one docstring. `tests/_derivative_data.py`: its unread `chunk_size` parameter and the
  docstring's "until P16 / P20".
- `tests/test_import_graph.py`: `test_the_old_sampling_kernels_are_gone`, and its line in the module docstring.

## 2. The tests, and what each re-base establishes

| Test | Establishes | Oracle |
|---|---|---|
| `TestPlainKindsTwoHole.test_residual_weights` | the plain two-hole at `omega^2 r` is the curvature part of the weighted objective's Hessian | a central difference of the engine's gradient minus the GN term (an identity between different engine variables) |
| `TestCurvatureJvpOracle` (two tests) | the engine's two-holes at a line (probe, apply, apply at the one-hot line; uniform probe with junk in the directions' padding) are the jvp of the corewise transposes | `jax.jvp` |
| `TestJetDtypeUnderX64.test_float32_in_float32_out` | float32 in, float32 out with x64 on: ten faces, ragged and uniform, numpy eager, jax eager and jitted (30 subtests) | the dtype |
| `TestTransposeOutputHygiene` (two tests) | (a) exact masks, (b) the real block is the ragged face's per element on `to_t3frame()` / `to_t3()`, (c) zero padding, (d) junk-robust; and the negative control: without the masks (c) fails | ragged per element; the masks |
| `test_the_jets_of_entries_off_the_grid` | `dense_apply` at the one-hot line is the entries derivative jet | the subset-sum definition (`np.take`, `np.tensordot`) |
| `test_derivative_methods`, `test_corewise_derivative_transpose_nontrivial_boundary` | the corewise derivative transposes are the gradients of the forwards | a central difference of the dense forward (`dense_*`, factorials inline) |
| `test_jit_shared_uniform_exact` | (unchanged) the shared uniform exact model jits | its data now from `dd.forward` (same numbers through the engine) |
| `test_the_old_sampling_kernels_are_gone` | the module is gone | `MODULES`, `find_spec` |

## 3. The gate

**Recorded step before the deletion** (the edited tests on the untouched library, note §6):
`k0/logs/pre_deletion_edited_tests_v.log.gz`: **1 failed, 519 passed, 2 skipped, 37,798 subtests** in 21:56; the one
failure is `test_the_old_sampling_kernels_are_gone` (the module still existed), as expected. The slow tier of the edited
`test_uniform_curvature.py`: `k0/logs/pre_deletion_slow_uniform_curvature.log`: **45 passed, 286 subtests** (1:16).

**The suite** (my choice of the two ways: one `-v` run serves as the gate's suite step and the tally's input;
`k0/k0_suite_v.log.gz`): **1 failed, 1,881 passed, 5 skipped, 1 xfailed, 1 warning, 90,134 subtests passed** (34:23,
`-n 2`). The failure was a third scan floor the note did not list (section 4):
`TestInterpreterAnyAxisSharding.test_every_sub_axis_of_every_group_shards_free` asserts `checked > 400`, and the sweep
now checks 228 cases (930 before). Fixed as the two floors of ruling 1; the file rerun alone: **4 passed, 476
subtests** (1:27). So **1,882 passed**, the note's count. The 5 skips and the xfail are the baseline's.

**The other steps** (`run_gate.sh -n 2`, by name, after the suite):

| step | result | counts | seconds | peak MB | lowest MemAvailable MB |
|---|---|---|---|---|---|
| doctest-modules | PASS | 320 passed | 54 | 1,673 | 9,988 |
| doctest-quickstart | PASS | 1 file, 0 failures | 3 | 267 | 11,375 |
| doctest-pages | PASS | 47 files, 0 failures | 20 | 333 | 11,359 |
| examples | PASS | 14 examples, 0 failed | 583 | 652 | 11,041 |
| sphinx (`-W`) | PASS | 0 warning/error lines | 755 | 565 | 10,966 |
| slow (serial) | PASS | 52 passed, 2 warnings, 13,076 subtests | 1,432 | 262 | 11,325 |

Sphinx output deleted once read. Module doctests 320, as the note expected (5 fewer docstrings with examples).

**The tally** (`k0/tally.py`, `k0/tally_baseline_vs_k0.txt`): 1,985 baseline nodes, 1,888 new; **106 nodes differ,
each accounted for**:
- 98 in the baseline only: exactly the 98 deleted tests of the note §3 (checked by name against its lists);
- 1 new: `test_the_old_sampling_kernels_are_gone`;
- the two renamed tests read under their new names: equal counts (1 passed; 14 and 2 subtests);
- 7 whose subtest count changed, each expected:
  - `TestTransposeOutputHygiene.test_tangent_transposes` 336 → 168 and `test_corewise_transposes` 192 → 96 (the
    derivative rows went, the plain rows kept theirs);
  - `TestJetDtypeUnderX64.test_float32_in_float32_out` 22 → 30 (the re-based op list);
  - `TestCallSiteConsistency.test_every_literal_call_site_supplies_enough` 286 → 113 (one subtest per library call
    site);
  - `TestInterpreterAnyAxisSharding.test_every_sub_axis_of_every_group_shards_free` 930 → 228 (one per scanned case;
    the floor above);
  - `test_import_graph.test_no_backend_module_imports_safety` 48 → 47 and `test_jets.test_no_backend_module_calls_a_checker`
    48 → 47 (one per backend module: one module fewer).

Every other node has the same outcome and the same subtest count, the edited ones included
(`TestUniformPlainKindsTwoHole.test_per_element_vs_ragged` 30, `test_residual_weights` 4, the entries jets 12, the
jit tests 3 and 2, `TestCurvatureJvpOracle` 14 and 2). Subtests: 93,034 − 1,767 (the deleted tests) − 168 − 96 + 8 −
173 − 702 − 1 − 1 = 90,134, the run's count (with the sharding test then failing; its 228 are in that count).

**No reference left:** `k0_inventory.py --check`: **0 references to deleted names; 15 allowed strings** (the O1 sweep's
labels, the two absence tests of `test_jets`, the word "PROBE" in an example's heading).

**Imports** (`k0/logs/imports_both_ways.txt`): all 58 modules import with jax (`jax_available = True`) and with jax
unimportable (`False`), no failure.

**The kept-comments check** (`k0/kept_text_check.py`, output `k0/logs/kept_text_check.txt`): every top-level statement
the seven modules and `common.py` keep, its text whole (signature, the shape comments and their columns, docstring,
body, comments), compared with `b5125f76` line by line. 13 changed, each one of `prose_edits.md`: the seven module
docstrings, the six `__all__` lists (entries removed, `probing`'s section comments), and three function texts whose
diff is exactly the edited prose (`entries.t3_entries_ambient_transpose`'s comment, `utv_sampling._gauge_masks_over_Knew`'s
one word, `common.readonly_mask_copies`'s last line). No signature, shape comment, aligned column or diagram of a kept
function changed. The comments between statements (the section headers) are not in that check; they are the ones
`prose_edits.md` §2 lists, applied by `apply_prose.py` as exact replacements, and read by hand after (the headers and
what follows them, listed with `grep -A3`).

**The mutations of the note §6, rerun on the built tests** (a scratch copy of `t3toolbox/` and `tests/`, `python -B`;
the step mutants by `k0/experiments/mutate_step.py`, which scales one step's output; each restored after, the copy
compared with the tree and deleted). Every one is caught, as the note expected:

| Mutation | Test | Outcome |
|---|---|---|
| (i) `H_tilde_dot_step` ×(1+1e-3) (probe's `Kp` TT block) | `TestPlainKindsTwoHole.test_residual_weights` | both probe subtests fail |
| (i) `H_tilde_step` ×(1+1e-3) (apply's and entries' `Kp` TT block; probe's `g`) | the same | all four subtests fail |
| (ii) one `t`-subset term dropped from the oracle | `test_the_jets_of_entries_off_the_grid` | 12 of 12 subtests fail (0.21 to 1.8 against 1e-12) |
| (iii) the dense forward without `t!` | the two `test_tucker_tensor_train` tests | probe 0.23, apply 0.92, entries 1.0 against 1e-5 |
| (iv) `H_tilde_dot_step` ×(1+1e-6) | `TestCurvatureJvpOracle` | the probe, probe-at-a-line and uniform-probe-at-a-line rows fail; apply untouched |
| (iv) `H_tilde_step` ×(1+1e-6) | the same | apply, apply at a line, apply at the one-hot line, entries and the probe rows fail |
| (b) `ut3_probe_corewise_transpose`'s output and every `utv_*_transpose`'s `V` block ×(1+1e-6) | `TestTransposeOutputHygiene` | the real-block clause fails on those rows (1e-7 to 1.5e-4 against 1e-12 of the scale) |
| (v) the negative control with the masks left in place | the same | 264 of 264 controls fail ("AssertionError not raised"): it cannot pass vacuously |
| (vi) `start_zeros` built without `dtype` | `TestJetDtypeUnderX64.test_float32_in_float32_out` | 21 of 30 subtests fail (the ops whose rows start from zeros, numpy and jax, eager and jit) |
| the floors: the scan pointed at an empty directory | `TestVocabularyCompleteness` | fails |

## 4. Departures from the note, and choices

- **A third scan floor** (the note §2.4 listed two): `test_contractions_sharding.py:136`, `checked > 400`, now 228. The
  note's simulated count did not reach it: the vocabulary floor fails first in the same test. Lowered to 150 with a
  comment giving the counts (228 at K0, 930 before), as ruling 1 did for the other two (150 is 2.5 times the
  vocabulary floor of 60, the cases per subscript being about 2.65). **Confirmed by the orchestrator** (2026-11-25).
- The floors' comments give **84** subscripts at K0, measured on the built tree (the note's 86 was the simulated
  tree, which still held the two dead scan bodies; review r1 m1).
- **`TestTheEngineShardsOverW` borrowed `setUpClass` and `_check` from the deleted `TestJetKernelsShardOverW`.** The
  note did not see the borrowing; the two methods are moved into it verbatim, and its docstring's reference to the
  deleted class goes. No change to what it tests.
- **Two imports more** (`contractions` in `apply` and `entries`): read only by the two dead scan bodies the note deletes
  (Decision 2), so they fall under Decision 3.
- **`_derivative_data.py`**: its `chunk_size` parameter removed (no caller passed it; the note §3.7 said so), and its
  docstring's "(until P16) ... (until P20)" corrected; `sum_over_probes` stays (ruling 3).
- **`test_jit_derivative_kind_two_holes`** keeps its name: its kept half is the exact local model of the derivative fits,
  whose hvp is the two-hole through the oracle.
- **`test_dispatch._uniform_pd_problem_jax`'s `chunk_size` parameter** is unread and stays, with its two callers (the
  note §5: "else stay").
- **`test_import_graph.py`'s docstring** said "Three facts are pinned" over a list of five; with the new line it says
  "Six".
- **`test_contractions_interpreter.py`'s three history comments** left (ruling on M3).
- `git rm` was used for the five deleted files and then reset: the deletions are in the working tree only, nothing is
  staged.

## 5. For the orchestrator, and outside the slice

- **Left for P17** (`sum_over_probes`, ruling 3; `chunk_size`): `sum_over_probes` in `tests/_derivative_data.py` (6),
  `tests/oracle_sweeps/o1_ragged.py` (6), `o1_uniform.py` (6), `tests/test_dispatch.py` (6), `tests/test_manifold.py`
  (3), `tests/test_tucker_tensor_train.py` (3), `tests/test_uniform_probing.py` (31) -- a keyword of the test helper
  and its callers, and labels; none in the library. `chunk_size`: `t3toolbox/optimizers.py` (4: the translation's old
  signatures and its docstring), `tests/test_optimizers_frontend.py` (5: the old call form), `tests/test_dispatch.py`
  (6: `_uniform_pd_problem_jax`'s parameter and two callers, two old-form `newton_cg` calls, a docstring).
  `ztildes`: none. `pp=`: one keyword, a test-local dict (`tests/test_fitting.py:436`).
- **Left for P19, the pages** (the note §2.2; lines at `b5125f76`, unchanged by K0; `k0/inventory_before_k0.md` §5):
  `docs/entries_apply_probe.md` 243-269 (the `compute_*` helper chain, 13 lines); `docs/fitting_and_optimization.md`
  19, 247-264, 349 (`SamplingKind` and its subclasses; two fenced lines); `docs/naming_conventions.md` 82, 88, 150,
  160-164, 193; `docs/contributor/curvature_internals.md` 31, 102, 112, 114, 122, 349; `docs/glossary.md` 404, 421,
  461, 789; `docs/contributor/parameters_not_closures.md` 24, 28, 156, 187; `docs/contributor/scan_body_principles.md`
  223, 228; `docs/contributor/deferred_and_rejected.md` 122, 257-260; `docs/contributor/batching_internals.md` 66;
  `docs/contributor/ambient_derivative_transpose_note.md` 13; `docs/contributor/signature_style.md` 55 and
  `docs/reading_signatures.md` 11 (they quote `compute_mu`'s signature as the house exemplar);
  `docs/contributor/verification_records.md` 87; `docs/uniform_equivalence_contract.md` 118 (`sumsq_over_probes`);
  `docs/release_notes.md` (on purpose).
- **Noticed, not K0's:** `test_uniform_curvature.py`'s module docstring still names `uniform_least_squares_problem`
  (gone in P16); `test_curvature.py`'s `_flat_inner` is unused; `TestTheEngineShardsOverW` has no `skipUnless(jax)`
  (its `setUpClass` imports jax; it had none before either); `_curvature_oracles.py`'s docstring says "Two oracles"
  over three bullets.
