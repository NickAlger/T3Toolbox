# PART1_PLAN.md: revision for the maintainer's answers to the open points

This revision applies the answers in `TARGET_DECISIONS.md`: the note "The names of Part 1, and the form of the merged
sampling functions" (with the ambient transposes, 18:35), "The plan's open points, as the maintainer answers them",
and every note after it up to "What remains, in groups". It was done in one pass, and the reviewer is to check it.
All counts are from an AST scan (alias-aware, doctests included) of `t3toolbox/`, `tests/`, `docs/` and `examples/`
at `a9c31b2`.

## Slice numbering, old to new

| Old | New | Content |
| --- | --- | --- |
| P0 to P15 | P0 to P15 | unchanged in place. P2 now also does the dense functions; P3 the checkers and the frontend reach; P4 the frames' cached residual; P5 `HeldVariables` (moved from P14) |
| P16 | P16 | the switch, first part: the library and the tests, with a translation of the old call form in the frontend optimizers |
| (new) | P17 | the switch, second part: the examples and the pages; the translation deleted |
| P17 | P18 | the dry port |
| P18 | P19 | documentation |
| P19 | P20 | the deletion (no benchmark gate) |

Every cross-reference in the plan was renumbered. The bridge and the old kernels now go in P20.

## What changed, section by section

- **Status and "How to read it".** The status line is the one asked for. The answers are named as an authority
  beside the rows.
- **1 Scope.**
  - The 5.3 row is now "one site at a time" (`xmap` on uniform, at every order).
  - 2.9 now reads: the engine sets `'scan'` on uniform and `'loop'` on ragged; both forms are built.
  - A section "What Part 1 gives up until Part 2" covers memory, time at order 0 on uniform, and the absence of a
    benchmark.
  - The seams gain "The sites (5.3)" and "The benchmark". The form-over-the-orders seam now reads "the engine sets
    it, Part 2's plan later".
- **2.1 Modules.**
  - `HeldVariables` is in `sampling_variables`.
  - `jets` gains the checkers.
  - The dense functions are listed beside their kinds.
  - `fitting` gains `check_batch`; `uniform_fitting` gains `pack_batch`.
  - The frame checkers are listed, with their uniform twins in `ufv_operations`.
  - The frame classes gain the cached `consistency_residual`.
  - The frontend `fitting` gains its reach (re-exports and its own `dual_jet_to_jet`), and `TuckerTensorTrain` gains
    `unscale_coordinates`.
  - Import rule: the backend never imports `safety`. The old line "O-6 asks to add `safety`" is gone.
- **2.2 Data.**
  - The uniform frame table is decided (O-3).
  - O-13 is stated.
  - The weight: the engine promises nothing about it; the backend has `check_weight`, and the frontend calls it.
  - The batch and the oracle are one class each, reachable from `t3toolbox.fitting`.
  - `held` is rewritten as `HeldVariables` per O-8 (dictionary-like, `with_values`, no shape check, a pytree only
    where jax is importable); the protocol types it as any object.
  - The `held` row of the model table is updated.
- **2.3 Engine.**
  - The signature takes `held=None` in place of `**held`.
  - New paragraph: the engine gives no guarantees beyond what it needs to run, has no safe mode and does not import
    `safety`.
  - The steps are renumbered:
    1. what it needs to run (unknown names, missing arguments, held shapes; ranks in `jet_contract`); the sample
       form, sizes, `y` and weight checks are removed;
    2. layout;
    3. the closure, moved before the dtype step;
    4. dtype: the inputs counted per O-11, only those the closure reads, no integer index;
    5. sweeps and maps: one site at a time through `xmap` on uniform, and the `over_orders` choice by layout.
  - The old step 4 (the safe-mode center check) is deleted.
  - O-12 is stated as decided.
  - The graph test covers the `HeldVariables` slots.
  - The engine has no frontend entry.
- **2.4 Backend surface.**
  - The New table gains the checkers, the dense functions, the frame residuals and `pack_batch`.
  - `GroupLengths.S` is always 0 in Part 1.
  - `jet_contract` takes `over_orders` (no default), and both forms are described.
  - "The strings in Part 1": 41 literal strings, the 25 map-type ones written without `S`.
  - `dual_jet_to_jet` divides with no check.
  - New: the checkers' signatures.
  - `rescale_coordinates` and `t3_unscale_coordinates` take per-coordinate `scales`, as the prototype does (O-35).
  - The two estimates are not shipped (O-14).
  - The dense functions get sketches.
  - The reference module gets `z` from the dense functions and is marked backend only.
  - New: the frame checkers.
  - The protocol and the oracle gain `check_batch`.
  - The minibatch scale per O-21, with the identities.
  - `local_model` checks nothing.
  - The corewise Hessian is defined through `phi(x) = <c, z(x)>` (O-22).
  - `dense_probe` follows N16.
  - The geometry's `frame` form is decided (O-4).
  - The removed list gains the frontend `_canonical_weight` and `_check_ragged_sample`.
- **2.5 Frontend surface.**
  - The merged methods call the `pp` validators.
  - New: `unscale_coordinates`, and the frames' cached residual.
  - New: what the frontend optimizers check (the batch, and in safe mode a supplied `start_frame`).
  - The frontend `local_model` signature (O-9, with `frame=` from O-5).
  - New: what `t3toolbox.fitting` holds, and what the frontend does not have (the stated exceptions).
- **3 Names.** N26 is amended (`HeldVariables` in `sampling_variables`). The N29 reference to O-23 is dropped. A
  pointer to the new names table in 11.3.
- **4 Code.** No string carries `S`, and `over_orders` is named at every call; the razor bullet now covers the two
  users; new bullet "Checkers, not guarantees".
- **5.1.**
  - The additive phase now includes P2's `dense_probe` change.
  - The two-part switch (O-24) is described.
  - The old kernels are deleted on the tests alone.
  - The bridge is decided (O-25).
- **5.2.** The overview is rewritten for P0 to P20. The "Needs" column now holds only the new open points O-29 to
  O-35. A sentence says that each proposed name is agreed before the slice that creates it.
- **5.3, P1.** Both forms over the orders, with tests that they agree to tolerance at orders 1 to 16; the strings
  are tested as Part 1 writes them.
- **5.3, P2.** The dense functions and the inventory of the callers of `dense_probe` (below).
- **5.3, P3.** The checkers; `dual_jet_to_jet` in its backend form (divides) and its frontend form (raises in safe
  mode); the frontend reach; `unscale_coordinates`; per-coordinate scales. New test: no backend module imports
  `safety`.
- **5.3, P4.** The checkers of a frame: residuals only, with their uniform twins, and the cached frontend property.
- **5.3, P5.** `HeldVariables`, with its own test file. The safe-mode center check is removed; a test now checks
  that no engine check depends on the safety mode.
- **5.3, P8, P10, P11.** P8 states O-12 and O-13. P10 gets the dtype tests for O-11. P11 maps one site at a time
  through `xmap` with `'scan'` over the orders.
- **5.3, P12.** The corewise Hessian convention, and its test against automatic differentiation. The ambient
  transposes' default is tested. O-26 is stated.
- **5.3, P13.** The eleven `dense_*_derivatives` calls in five examples move to `dense_apply` and `dense_probe`.
  `test_probe_derivatives.py` is folded in on the dense functions.
- **5.3, P14.** `HeldVariables` is no longer here. Added: `check_batch`, `pack_batch`, the reach from
  `t3toolbox.fitting`, and the tests of the scale and its identities (O-21).
- **5.3, P15.** The consistency member is a checker (O-31).
- **5.3, P16 (rewritten).**
  - The backend model checks nothing.
  - The frontend calls `check_batch`, packs with `pack_batch` and checks a supplied `start_frame` in safe mode.
  - The timing laps are now `frame`, `oracle`, `companion`, `gradient`, `curvature`.
  - The translation is specified.
  - The tests and module doctests move here, and so do the page doctests and examples that call a factory, which
    the translation cannot cover.
  - New tests: the frontend checks, the translation, and the frontend-only fit.
- **5.3, P17 (new).** The seven examples and the page doctests that call the optimizers in the old form (lists
  below). The translation and `_canonical_weight` are deleted. A search for the old form must find none.
- **5.3, P18 to P20.** The dry port, documentation (now including the paper-to-code page), and the deletion.
- **5.4.** Renumbered, with a new row for the old call form and `_canonical_weight`.
- **6.** New rows: Checkers, Reference (the automatic-differentiation test), and Translation. The structural row
  covers the `HeldVariables` slots. The fitting row covers the scale identities and the frontend-only fit. The file
  table is renumbered and gains the P2 `dense_probe` calls.
- **7.1 (rewritten).** The rule; the backend's checkers; what the backend still checks structurally; what the
  frontend enforces, and when; the compiled phases (O-32); tests for both tiers.
- **7.2, 7.3, 7.6, 7.7.** `xmap` and the forms over the orders; `HeldVariables` without jax; `over_orders` as a
  static; both forms slice only the order axis.
- **7.5 (rewritten).** Built around the maintainer's rule of the two users, quoted in full: what each user has,
  where the logic lives, the stated exceptions, and the two fit tests.
- **8.**
  - The diagrams are decided (O-23).
  - The bad-scaling example computes the estimates inline (O-14).
  - New row: the paper-to-code page.
  - `numerical_contracts` is rewritten for checkers, not guarantees.
  - `symmetric_probe_derivatives.tex` is archived to `dev/archive/`.
  - The CLAUDE.md record gains the map pointer and the safe-mode paragraph.
- **9.**
  - `_newton_fit` now breaks in P17, because the translation keeps it working through P16.
  - The keys line is reworded.
  - "What it is for" now points at Part 2's benchmark.
- **10.** "The last step: the deletion", on the tests alone; the benchmark item is removed. O-26: the folder's
  README gets its line here.
- **11 (replaced).** 11.1 records O-1 to O-28, one line each with where the plan uses the answer. 11.2 lists the
  new open points O-29 to O-35. 11.3 is the table of names added after the table was settled, all proposed.

## What the new inventories found

- **`dense_probe` (library) callers: 12, all positional** (P2).
  - Doctests: `backend/probing.py`, 8 (three in `t3_probe`, two in `tv_probe`, three in `dense_probe`);
    `manifold.py` (`T3Tangent.probe`) 1; `uniform_manifold.py` (`UT3Tangent.probe`) 1.
  - Tests: `tests/test_manifold.py:524`, `tests/test_tucker_tensor_train.py:3166`.
  - No page calls it.
  - **Surprise:** seven files define a local helper of the same name, already in N16 order, and are not affected:
    `tests/test_fitting.py`, `tests/backend/test_optimizers.py`, `tests/test_optimizers_frontend.py`,
    `tests/oracle_sweeps/o1_common.py` (used by `o1_ragged.py` and `o1_uniform.py`),
    `examples/fit_hilbert_from_probes_adam.py`, `fit_hilbert_from_probes_optax.py` and
    `fit_per_mode_weight_probes.py`. A name search counts 43 calls; only 12 are the library's.
- **`dense_*_derivatives` callers** (P13).
  - Examples, eleven calls:
    - `dense_apply_derivatives`: three each in `fit_hilbert_from_apply_derivatives.py`, `_flat.py` and `_topt.py`;
    - `dense_probe_derivatives`: one each in `fit_hilbert_uniform_probe_derivatives_newton_cg.py` and
      `fit_shared_factors_jetted_probes.py`.
  - `tests/test_probe_derivatives.py`: nine calls.
  - One call inside `sampling_derivatives` itself.
- **The checkers' call sites in the frontend today.**
  - `_canonical_weight`: `t3toolbox/optimizers.py:309` (in `least_squares_problem`); `t3toolbox/fitting.py` 783,
    846 and 876 (the factories); `tests/test_fitting.py` 831, 1230 and 1295.
  - `_check_ragged_sample`: six calls in the `fitting.py` factories.
  - The `pp` validators: 24 calls in the frontend `*_derivatives` methods (six per class), which go in P13.
  - `optimizers._check_kind`: one call.
  - Supplied `start_frame` today: `_padded_start_keywords` checks `is_orthogonal` and "represents `x0`" in safe mode.
    The consistency check joins it.
- **No backend module imports `safety` today** (checked); P3 pins it.
- **The benchmark gate** was referenced in sections 1, 5.1, 9, 10 and old O-1 and O-2. All are removed or redirected
  to Part 2.
- **The `CLAUDE.md` map** is the notation paragraph of "The paper (T4S)". The probing map with the old kernel names
  is `docs/entries_apply_probe.md` section 8 (O-33).
- **`docs/symmetric_probe_derivatives.tex`**: nothing outside `dev/` links to it; `dev/paper_scope.md` names it.
- **The two estimates:** only the prototype `scripts/review/bad_scaling_example.py` computes them. No existing
  example does; three examples only rescale their start.
- **Examples and the switch.**
  - Five call a removed factory and must move in P16: `fit_hilbert_from_entries_lbfgs.py`,
    `fit_hilbert_from_probes_adam.py`, `fit_hilbert_from_probes_optax.py`, `fit_hilbert_tensor_newton_cg.py` and
    `fit_varied_rank_tensor_newton_cg.py`.
  - Seven call only the frontend optimizers and move in P17: `_topt.py` (`mc_sgd` with a custom draw),
    `fit_hilbert_regularized.py`, `fit_hilbert_uniform_newton_cg.py`,
    `fit_hilbert_uniform_probe_derivatives_newton_cg.py`, `fit_per_mode_weight_probes.py`, `fit_probe_display.py`
    (with `val_sample` and `val_data`) and `fit_shared_factors_jetted_probes.py`.
  - `fit_hilbert_from_apply_derivatives.py` names a factory only in comments.
  - No example or page passes `chunk_size`.
- **Pages and the switch.**
  - Factory doctests move in P16: `docs/fitting_and_optimization.md` lines 147, 182, 424 and 430, and
    `docs/numerical_contracts.md` line 94.
  - Optimizer doctests move in P17: `docs/fitting_and_optimization.md` (one with a custom `draw`),
    `docs/sharing.md` (2) and `docs/getting_started.rst` (2).
  - `docs/release_notes.md` mentions the factories in prose only.
- **Module doctests:** `t3toolbox/optimizers.py` has 16 optimizer calls, 4 calls of `least_squares_problem` and 1
  of `Problem.local_model`; `t3toolbox/fitting.py` has 6 factory calls.
- **`UT3Frame.is_consistent`** goes through `UniformTuckerTensorTrain` arithmetic. There is no backend uniform
  consistency residual today, so `ufv_frame_consistency_residual` is new.
- **The strings:** of the 41, 25 carry `S` in the prototype (`xi` 5, `z` 3, `eta` 4, `V` 5, `H` 8). In Part 1 they
  are written without it.

## Found against the answers or the rows (listed in the plan as open points, not resolved)

- **O-35, the shape of `scales`.** The plan's earlier sketch gave `(d,)`. The prototype uses one scale per
  coordinate, and the plan now follows it.
- **O-32, rows 4.2 and 8.3.** They state checks (at every call that requires consistency; before each compiled
  program) that O-5 and O-6 no longer make.
- **O-30, `held` as "any object".** This does not fit the pytree model under `use_jit`.
- **O-29, `check_batch` and user-written oracles.** It is a method of the least-squares oracle only.
- **O-31, the geometry's consistency member (N21).** Nothing in the library calls it any more.
- **O-34, the uniform `unscale_coordinates`.**
- **O-33, which maps move.**
- **The `local_model` signature.** O-9's signature has no `frame=`, while O-5 speaks of "a frame given to
  `fitting.local_model`". The keyword is listed among the proposed names.
- **`TARGET_DECISIONS.md` itself.**
  - "The two parts" still has the 5.3 row "at all sites at once on the uniform one", which the answer to O-19
    supersedes.
  - Small item 2 still says the engine checks centers; the O-6 note says it is reworded, but the item text is
    unchanged.
  - Small item 3 still names the benchmark gate.
  - These are not the plan's to edit.
