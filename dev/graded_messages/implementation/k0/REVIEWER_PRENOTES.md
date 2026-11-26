# K0: the reviewer's first look, before the design note

Claude (the reviewer), 2026-11-25. A working note, not a review. Tree: `b5125f76` (library and tests as at
`3375d213`). Nothing in the repository was changed or run except reading; the scripts below are in
`/tmp/claude-1000/-home-nick-repos-T3Toolbox/7a03b046-9177-4c43-923a-b31595edb27e/scratchpad/reviewer/k0/`
(`scan.py`, `tests_scan.py`, `tables_scan.py`, `doctest_scan.py`, `import_all.py`, `jvp_rebase.py`, `f32_x64.py`,
`f32_mutant.py`). No test was run.

## 1. Where a careless deletion goes wrong

**1a. "Touched" is not "tests old code".** Several test units reach old code only through a shared table,
helper or import, while what they check is the engine. Deleting them, rather than cutting the old member out,
drops engine coverage:

- `tests/backend/test_curvature.py:889-906`: `_plain_ops(kind)` builds a 4-tuple: the engine's two-hole and one-hole
  ([0], [2]) and the old `*_from_sweep` and `tv_precompute_*` ([1], [3]). Building the tuple raises once [1] is gone,
  so `TestPlainKindsTwoHole` (block-sum oracle, finite differences, stacks, symmetry, weights, uniform) and
  `TestPlainKindsExactLocalModel.test_corewise_is_the_plain_hessian` (`:2178`, which uses [2] only) all go red. These
  are the engine's main plain-kind `Kp` tests. They should be re-based by trimming the tuple, not deleted.
- `tests/backend/test_uniform_curvature.py:637-642`: the module-level `_PLAIN_UNIFORM` names old twins, so the whole
  file (56 tests, part of the slow tier) fails at import. `TestUniformPlainKindsTwoHole.test_per_element_vs_ragged`
  (`:671`) checks the engine twin per element against ragged with junk in the padding. Only its sweep-reuse lines
  (`:701-703`) use old code.
- Module-level `import ... sampling_derivatives` fails a whole file: `test_tucker_tensor_train.py:13` (81 tests),
  `test_dispatch.py:37` (108), `test_uniform_probing.py:24` (35), `test_curvature.py:100`,
  `test_uniform_curvature.py:53`, `_curvature_oracles.py:53`.
- `tests/test_dispatch.py:4214-4250` (`TestCurvatureJvpOracle`): jax `jvp` of the one-hole checked against the
  two-hole. This is an AD oracle that shares no code with the two-hole sweeps. Its plain rows are engine calls; its
  three derivative rows and the uniform test (`:4253`) are old kernels. **Re-basing works:** `jvp_rebase.py` runs the
  engine twins with `order=(2,)` (`probing.t3_probe_corewise_{transpose,hessian}` on `affine_jet(ww, pp)`, the apply
  pair, and the uniform `ut3_probe_corewise_*` with junk in the direction padding). The relative errors are ≤ 3.4e-16.
  The design note should re-base these rows, not drop them.
- `tests/test_dispatch.py:3983` (`TestJetDtypeUnderX64`) pins a real class of bug: a float32 jet pipeline promoted to
  float64 under x64 by scan starts built in the default dtype (2026.2.0 had this bug). The engine passes the same check
  today (`f32_x64.py`: uniform jax eager and jit, uniform numpy, ragged numpy and jax, wanting
  `z, f, g, Jp, JtJp, Kp`: all float32). The mutant `start_zeros` without a `dtype` argument (`f32_mutant.py`) promotes
  `g`, `Kp` and `JtJp` to float64 on numpy too. `test_sampling_variables.py:401-419` (`TestResultDtype`) checks only
  `z` and `f`, which never read `start_zeros`, and by grep nothing else asserts the dtype of `g`, `Kp` or `JtJp` in
  float32. So deleting the class without a re-base on the engine leaves this bug class unguarded. To be confirmed by
  running the suite against the mutant.

**1b. Re-bases that turn circular or vacuous.**

- `tests/test_tucker_tensor_train.py:3040` and `:3066-3068`: the engine's corewise derivative transpose
  (`dd.corewise_transpose`) is checked against a central difference of `<r, forward>`, where `forward` is the OLD
  `sampling_derivatives.t3_*_derivatives` in derivative coordinates (`t!` times Taylor). Swapping in `dd.forward`
  keeps the coordinates right, but the test becomes the engine against the engine: an adjoint identity, still a check,
  though not an independent oracle. The independent choice is the dense forward,
  `probing.dense_probe(x.to_dense(), affine_jet(ww, pp), (n,))` (resp. `dense_apply`) times `t!`. Mixing Taylor and
  derivative coordinates here gives a factor `t!` per slot, the kind of mismatch that tempts a loosened tolerance.
- `tests/test_probe_derivatives.py:38`, the one test of 17 with no old reference, is vacuous: `chunk_size` is ignored
  by `tests/_derivative_data.py` ("accepted and ignored: the engine does not chunk", its docstring), so it compares the
  engine with itself. The whole file should go.
- `test_curvature.py:1060` `test_apply_tt_assembly_is_the_three_term_product_rule` checks the engine
  (`bapply.t3_apply_corewise_hessian(...)[1]`) against a product rule built from OLD pieces (`compute_xi`,
  `compute_sigma_hat`, `compute_tau`, ...). Rebuilding those pieces from the engine would be "a helper that calls the
  engine". Either re-base them on the dense reference (`sampling_definitions`) or delete the test, with the block-sum
  and finite-difference tests of the same class as the stated cover.

**1c. The plan's K0 text, measured against today's tree.**

- `tests/oracle_sweeps/o1_ragged.py:90-94`, `:184-188` and `o1_uniform.py:111-113` do not use an old kernel as their
  oracle. The old names there are `check(...)` labels; the code is `dd.forward(...)` against `dense_*_jets` (P13b).
  Nothing needs re-basing. The labels should stay too, since `tests/oracle_sweeps/README.md` asks for those files to
  be kept byte-close to the archived originals.
- `tests/backend/_curvature_oracles.py`: no oracle that a staying test uses depends on old code. `blocksum`, `doubled`,
  the dense Hessians, `tied_lift` and `vertical_direction` are clean. Only the twin state builders
  `probe_derivatives_two_hole_states_trs` (`:392`) and `apply_derivatives_two_hole_states_trs` (`:520`) use
  `sd.build_input_jets`, `probing.compute_xi`, `bentries._entry_xis` and `_onehot_vectors`, and their only users are
  test classes that go. So K0 deletes from this file; it does not re-base it.
- `tests/_jet_trs_twins.py` must **stay**, although P20's table (`PART1_PLAN.md:2295`, and P20's table at `:2650`) lists it under "old kernels
  only". `test_dispatch.py:47` imports it at module level (`twins.binomial_combine_tensor`, `:343`), and the oracles
  use it too.
- P20 also edited `docs/api_reference.rst` and added a line to `dev/graded_messages/README.md` saying that the
  prototypes run against `v2026.5.0` (O-26). The packing plan's K0 does not carry either. The prototypes break at K0's
  commit, not P20's, so the README line belongs in K0.

## 2. What an inventory misses (measured)

- **Second-order orphans.** M1(a2) reads one level only. A transitive closure over every module-level statement
  (`scan.py`) adds `apply._apply_from_xis_step`, `probing._sigma_step` (imported by `apply.py:23`),
  `_assemble_tucker_variations_core`, `_assemble_tt_variations_core` and `_tt_tangent_variations_core`. It finds
  `_BATCH_FIELDS` referenced, correctly, through the `if jax_available:` registration at `backend/fitting.py:811-815`.
  `apply._t3_apply_step` (`:41`) and `entries._t3_entries_step` (`:46`) have no reference at all today. They are dead
  already, and deleting them is a decision for the note.
- **Library orphans used by the tests:** `entries._entry_xis` and `_onehot_vectors`, at `_curvature_oracles.py:529-530`.
- **Names built at run time:** `getattr(module, 'tv_precompute_%s_frame_sweep' % kind)` at
  `test_uniform_curvature.py:727-728` (the negative control `test_the_direction_mask_is_load_bearing`, in a class M1(d)
  counts as untouched, "0/0") and `test_uniform_probing.py:913-914`. Module- and class-level tables are missed too
  (`test_uniform_curvature.py:637-642`, `:740-747`, `:907-909`; `test_uniform_fitting.py:105`, `:256-258`,
  `:395-403`). **M1(d)'s per-test counts are lower bounds.** The set of affected tests must come from a run.
- **False positives in a name-based "no reference" gate:** the labels above; `_HESSIAN_MODES`, which is also defined in
  `backend/optimizers.py:234`; the test-local `_init_jet` at `tests/_jet_trs_twins.py:48`; the absence assertions
  `test_jets.py:895-900` and `:933-937`, which name the deleted validators in strings; prose `APPLY`/`PROBE` in
  examples' docstrings. The gate needs classes (code, string, prose), not a bare grep.
- **Sphinx `-W` does not catch dangling Python references.** `docs/conf.py` is not nitpicky. `docs/api_reference.rst:75-80`
  names six model factories that P16 deleted, and the last gates recorded "Sphinx 0 warnings". After K0 these go stale
  with no warning: `api_reference.rst:130` (`:py:mod:` to `sampling_derivatives`), `backend/fitting.py:1-23` (a module
  docstring that is all about `SamplingKind`, with `:py:class:` and `:py:data:` to deleted names),
  `uniform_fitting.py:5`, `ut3_sampling.py:11`, `:18`, the comment block `probing.py:1245-1260`, and
  `common.py:526` (`_weight_matrix`). The docstrings and comments of library modules are not "pages' prose", so the note
  should decide them in K0 or list them for P19.
- Library doctests: every one that names a deleted name sits in a deleted unit (`doctest_scan.py`; `ut3_sampling.py:380`,
  `:640`). Page doctests and examples: none.
- `tests/test_jax_absent.py` imports a subset of modules only. Importing every module, with jax and without, is a
  check K0 must add itself. Baseline today (`import_all.py`): 59 modules, 0 failures both ways.

## 3. What goes with the deleted tests

- **The bridge** (`tests/bridge/test_old_kernels_bridge.py`): no loss that I can find. Every engine name, the tilde
  variables and `Kp` included, is checked against the dense reference by `test_sampling_variables.py:136`
  (`TestByDefinition`) and its siblings for apply and entries. The Taylor coefficients are pinned independently by
  `test_sampling_definitions.py:1107` (`TestJetsTwoWays`, interpolation) and `:1818` (`TestDenseFunctions`, DFT). The
  dense `Kp` has two routes, `:1202`.
- **The jetted two-hole after `TestProbeDerivativesTwoHole` and `TestDerivativeKindsTwoHole` go:** what still covers it
  is the dense reference's `Kp`, `sweep_exact.py` (every kind × geometry × representation against a dense extrinsic
  oracle), the T7 retraction tests, and `TestCurvatureJvpOracle` if it is re-based (1a). Without the re-base, the jax
  path and raw unsquashed cores at `order > 0` lose their AD check.
- **The direction-mask negative control** (`test_uniform_curvature.py:705-733`) is replaced only in part:
  `test_sampling_uniform.py:441-460` shows that the *padding* is not zero without the masks. The old test showed that
  the *real block* is corrupted.
- **`TestTransposeOutputHygiene`** (`test_uniform_probing.py:764`): its (b), the real block against the bare old seam,
  and its proof of non-vacuity (`_assert_nonvacuous`, `:802`, which needs a seam that leaves junk) both rest on old
  kernels. Its (a), (c) and (d) (exact masks from the ragged shapes, zero padding, junk robustness) should be kept,
  with (b) re-based on the ragged twin per element and non-vacuity shown on the inputs.

## 4. Checks I intend to run

On the note: re-run `scan.py` (transitive orphans, all module-level statements, templates) on the note's list. Check
each "re-based" test's new oracle against the rule "no code shared with the engine, or an identity between different
variables, said as such". Check that every deleted test has a named survivor for what it pinned (the list in 3).

On the built slice, in a scratch copy, with `python -B`:
- my inventory over `t3toolbox/`, `tests/`, `examples/`, classified;
- `import_all.py` with and without jax;
- `pytest --collect-only` counts before and after, and a `-v` per-test tally against the starting tree (every
  disappeared test either deleted on purpose or renamed);
- mutations: the `start_zeros` dtype mutant against the re-based `TestJetDtypeUnderX64`; a scaled `Kp` (×(1+1e-6)) and
  a dropped jet slot against the re-based jvp rows; a dropped exit mask against the kept hygiene test; a wrong `t!`
  in the re-based `test_tucker_tensor_train` difference;
- a check that no test compares the engine with itself (each re-based oracle traced to its imports);
- the slow tier serially, since `test_uniform_curvature.py` is in it and loses a module-level table.
