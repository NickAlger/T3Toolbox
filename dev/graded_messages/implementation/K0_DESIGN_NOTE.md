# K0 design note: the deletion of the old sampling kernels

Implementer, 2026-11-25, at `b5125f76` on `graded-messages` (library, tests, docs and examples as at `3375d213`).
Plan phase: nothing under `t3toolbox/`, `tests/`, `docs/` or `examples/` is changed. Sources: `PACKING_PLAN.md` §7
(K0), §8, §9; `PART1_PLAN.md` §5.4 (`:2234-2252`) and §10 (`:2609-2660`); `K_MEASUREMENTS.md` M1; `K_BRIEF.md`.
Working files: `implementation/k0/` (below). Scratch: `scratchpad/implementer/`.

**Revision 2** (after `K0_DESIGN_REVIEW_r1.md` and the orchestrator's rulings; answers in `K0_DESIGN_RESPONSE_r1.md`):
the four open points are ruled (§8); the rewritten library prose is written out in `k0/prose_edits.md` (S2); the
added test is one assertion in `test_import_graph.py` (S3); the mutations name a library step per kind with their
expected outcome, measured (S1, §6); `docs/api_reference.rst` loses the nine other names of things that no longer
exist (M2, §2.2); the factorials of the `test_tucker_tensor_train` re-base are inline (M1); the three history comments
of `test_contractions_interpreter.py` are all left (M3); the negative control asserts that the zero-padding check fails
(M4, §3.6).

**Library methods traced first** (the maintainer's instruction): `TuckerTensorTrain.probe` (`tucker_tensor_train.py:3660`)
→ `jets.check_sample` and `probing.t3_probe` (`backend/probing.py:87`) → `sampling_variables.sampling_variables`
(`sampling_variables.py:1678-1936`: `_closure_facts`, the uniform packing and masking, `_read_setting`, `_check_stacks`,
`_run` over the rows) → the step functions (`sampling_steps.py`, e.g. `xi_step`, `mu_step`: one literal
`jet_contract` string each); `probe_corewise_transpose`/`_hessian` (the engine with `held`); `t3svd`
(`tucker_tensor_train.py:4218` → safety check → `backend/t3_svd.t3svd`); `__add__` (→ `t3_linalg.t3_add`, the
block cores). Then the seven modules K0 touches, whole, and `sampling_derivatives.py` by its structure.

## 0. Summary

K0 deletes 315 top-level names (`sampling_derivatives.py` whole, 154; 161 in the other seven modules), about 6,500
lines, 98 tests (48 in five whole files, 50 in seven others), and edits 17 tests, 10 of them re-based (their oracle,
or a subject or data maker beside kept ones, moved off deleted code onto the dense reference, jax AD, a dense einsum
or the engine's own faces); 7 more are kept unchanged through an edited helper. Where the plan's K0 does not fit
today's tree:

1. **Two scan floors turn red** (§2.4). `tests/test_contractions_interpreter.py:551` and
   `tests/test_contractions_sharding.py:102` assert that the library holds more than 100 grouped subscripts; on the
   simulated K0 tree it holds 86 (209 today). Ruled: both floors go to 60 (§8.1).
2. **The plan's oracle list is partly out of date** (§3). `tests/oracle_sweeps/o1_*.py` no longer use an old kernel:
   since P13b their subject is `_derivative_data` (the engine) and their oracle the dense jets of `o1_common`; their
   "references" are check labels. The tests that do lose their oracle are others: `test_tucker_tensor_train`'s two
   derivative tests, `test_sampling_definitions.test_the_jets_of_entries_off_the_grid`,
   `TestTransposeOutputHygiene` (two tests), `TestPlainKindsTwoHole.test_residual_weights`, and three tests of
   `test_dispatch` whose old kernels were a data maker or a subject beside kept ones.
3. **The static inventory misses what the plan's method misses**, and more: names built at run time
   (`getattr(module, 'tv_precompute_%s_frame_sweep' % kind)`, two tests), and tables of old functions read by tests
   (`_PLAIN_UNIFORM`, class attributes like `_RKIND`). The script handles both now (§1.1).
4. **The examples have no reference** (confirmed). **No page doctest** names a deleted name, and **nothing a
   deletion removes can fail Sphinx `-W`** here (`nitpicky` is off, no `:doc:` or toctree entry points into a
   deleted module's page) (§2.2).
5. **`sum_over_probes` is not in deleted code only** (plan §7 K0, "goes here"): it survives as a keyword of the test
   helper `tests/_derivative_data.py` (passed in 60-odd test lines) and in one kept docstring. K0 fixes the
   docstring and leaves the helper's keyword; ruled for P17 by name (§8.3).
6. **Coverage** (§4): every property the deleted tests pinned on code that stays is still pinned against an
   independent oracle (the dense reference, jax AD, a dense einsum, a Richardson difference), or is re-based here.
   What is lost: the agreement with the released 2026.5.0 kernels (by ruling, O-1/O-2), the internal structure of
   the deleted kernels (no subject left), the containment identity of the derivative two-holes as an explicit test
   (each side pinned on its own), and a per-kind negative control that has no analogue in the engine's layered
   masking (measured, E2; ruled: deleted, its apply and entries cases go to K5's design note, §8.4).

## 1. The inventory, on today's tree

### 1.1 Method and files (`implementation/k0/`)

- **`k0_inventory.py`** (standard library only; never imports the library). It reads **every module-level
  statement** of every library module (a module-level `if`, a decorator, a registration call, a default value, an
  assignment), not top-level definitions only: a statement's references are its `Name` nodes resolved to their
  module, attributes of any alias of a module (and dotted chains), `from M import n [as a]`, and string constants
  (identifiers inside them). Doctests are parsed with `doctest` and resolved the same way, each example seeing the
  imports before it. Prose (docstrings, comments, pages) is reported apart, never as a reference.
  The deletion set: the plan's rule applied to today's `__all__` lists, plus `sampling_derivatives.py` whole, plus a
  closure: any other top-level statement of the seven modules all of whose library referrers are deleted statements
  is deleted, iterated to a fixed point. The tests: every `def test_*`, its direct references, those reached
  through a helper (a function, a method, a module-level or class-level assignment of the same file, or of a
  helper module imported as `import _fits as F`; transitively), and whether its file fails at import. Names built
  at run time: every `getattr(x, <computed>)` and every `'..%s..'` string, completed with the three kinds.
  `--freeze` writes `k0_names.json`; `--check` is the gate on K0's tree (§6), with a short `STRING_ALLOWLIST`.
- **`inventory_before_k0.md`**: its report on this tree (432 lines; the source of every list below).
- **`k0_names.json`**: the 315 deleted names per module, frozen, for `--check`.
- **`simulate.py`**: a rough K0 library in scratch (deleted statements cut, `sampling_derivatives.py` removed, its
  imports and the deleted `__all__` entries dropped), for the scans whose result depends on what code exists.
- **`tally.py`**: M0's per-test comparison, with K0's two renames.
- **`baseline_v.log.gz`**: the `-v` run of the untouched tree (§6).
- **`experiments/`**: E1, E1b, E2 (the uniform masks), E3 (the jvp oracle at jets), E4, E5 (the dense oracles of
  §3.7): quick scripts run on today's library, each printing what §3 cites.

`_BATCH_FIELDS` (the trap of M1(a2)) is read by the `if jax_available:` statement that registers the batch
(`backend/fitting.py:811-815`), so this scan keeps it.

### 1.2 What is deleted, per module (counts: names; the lists are in `inventory_before_k0.md` §1)

| Module | By the plan's rule | By the closure | Total | `__all__` after |
|---|---|---|---|---|
| `sampling_derivatives` | whole: 169 statements, 3,534 lines, `__all__` of 57 | | 154 names | (module gone) |
| `probing` | 27 (`compute_*`, `assemble_*`, `tv_precompute_probe_frame_sweep`, `*_from_sweep`) | 23 private steps and cores (`_xi_step` … `_tt_tangent_variations_core`, `_sigma_step` included: its only other importer is `apply`'s deleted code) | 50 | `t3_probe`, `tv_probe`, `tv_probe_transpose`, `t3_probe_ambient_transpose`, `t3_probe_corewise_tangent`, `_transpose`, `_hessian`, `dense_probe` (8) |
| `apply` | 4 + `_t3_apply_step` (dead, Decision 2) | `_apply_from_xis_step`, `_apply_from_xis`, `_apply_transpose_adjoint`, `_apply_corewise_hessian_adjoint` | 9 | the 8 of `t3_apply`, `t3_apply_ambient_transpose`, `tv_apply`, `tv_apply_transpose`, `t3_apply_corewise_{tangent,transpose,hessian}`, `dense_apply` |
| `entries` | 4 + `_t3_entries_step` (dead) | `_entry_xis`, `_onehot_vectors` | 7 | likewise, 8 |
| `ut3_sampling` | 15 (`*derivatives*`, `*_from_sweep`) | `_remasked_core_gradient`, `_masked_point_and_directions` | 17 | 13: `ut3_{entries,apply,probe}`, `ut3_full_sum`, the three kinds' `_corewise_{tangent,transpose,hessian}` |
| `utv_sampling` | 24 (all but the six) | `_mask_once` | 25 | `utv_{probe,apply,entries}[_transpose]` |
| `fitting` | 20 (the nine kind classes, `APPLY`/`ENTRIES`/`PROBE`, the four aliases, the four `*sumsq_over_*`) | `_prod_w`, the six `_take_*`, `_weight_matrix`, `_make_weight`, the four `_*Sample` bases, `_DerivativesKind` | 34 | `LeastSquaresOracle`, `LeastSquaresBatch` |
| `uniform_fitting` | 8 (the six uniform kinds, two factories) | `_frame_theta`, `_UniformLayerMixin`, `_SAMPLING_KIND`, `_DERIV_SAMPLING_KIND`, `_fg`, the six `_ptake_*` | 19 | `uniform_minimal`, `pack_sample`, `pack_data`, `pack_batch` |

No public name outside the rule is swept by the closure. **M1(a)'s ambiguous names, decided:** `_flat_gather` and
`_take_W`, `_take_data_space` stay (the oracle's `take` and its helpers, `backend/fitting.py:939-976`, `:1190-1213`);
`_prod_w` goes (only the deleted sample bases read it); `_fg` and `_frame_theta` go (only `_ptake_*` and the uniform
kinds read them). Kept besides: `_KINDS`, `_HOLD_LIST`, `_SCALED`, `_BATCH_FIELDS` and its registration,
`_same_leaves`, `_scale_tree`, `_arrays_of`, `_stack_of_frame`, the three `_y_*` sums; in `utv_sampling`
`_gauge_masks_over_Knew` and `_remasked_transpose_output` (the kept transposes use them); in `ut3_sampling` `UT3Data`
and `Vectors`; in `probing` `_dense_contract_one_mode` and `_dense_jet` (`dense_apply`, `dense_entries` use them).

Line counts on the simulated tree (before docstring edits): `probing` 1,827 → 759, `apply` 702 → 467, `entries`
573 → 424, `ut3_sampling` 744 → 389, `utv_sampling` 591 → 273, `fitting` 1,240 → 652, `uniform_fitting` 518 → 224;
with `sampling_derivatives` about 6,540 lines go.

### 1.3 What stays in each module, and what is edited there

- **Imports.** Must go (they would fail): `fitting.py:34`, `ut3_sampling.py:26`, `utv_sampling.py:35` (the module);
  `apply.py:23` and `entries.py:26-27` (from-imports of deleted names). Unused after K0 (used before by deleted code
  only; `inventory_before_k0.md` §2): `probing` `tt_operations`; `entries` `probing`; `ut3_sampling` `apply`,
  `entries`, `probing`, `ut3_masking`; `utv_sampling` `probing`, `apply`, `entries`; `fitting` `abc`, `ft`,
  `probing`, `bapply`, `bentries`; `uniform_fitting` `readonly_mask_copies`, `ufv_masking`, `utv_sampling`,
  `ut3_sampling`. Unused already before K0 in the same seven modules: `probing` `np`, `t3_conversions`,
  `fv_conversions`, `ragged_ops`, `uniform_ops`; `apply` `t3_conversions`, `fv_conversions`, `math`; `entries` `np`,
  `fv_conversions`, `math`; `ut3_sampling` `np`; `utv_sampling` `np`; `uniform_fitting` `geometry_module`,
  `ufv_conversions`, `utv_ops`, `sharing_module`. Decision 3: both lists go, `uniform_fitting`'s `bfit` stays
  (`pack_batch`'s annotation and docstrings name it). The gate's doctest step is the check that no doctest leaned on
  a module global (`--doctest-modules` runs a docstring in its module's namespace).
- **`__all__`**: the table above; the section comments inside the lists that describe deleted entries go
  (`probing.py:43-52`).
- **The new texts of everything rewritten** (module docstrings, the comments cut to what stays, the corrected
  sentences) are written out, before and after, in **`k0/prose_edits.md`** (review S2). The build applies them as
  written; a change found necessary while building is reported in the handover. What is deleted whole stays here as
  line ranges.
- **Module docstrings** that describe deleted code are cut to what stays, nothing anticipating K1 to K7:
  `probing` (`:5-13`: "the xi/mu/nu/eta helper chain, assemble_z, the tv probe + frame sweeps"), `apply` (`:7-8`,
  "the frame sweeps for applies"), `entries` (likewise), `ut3_sampling` (`:5-19`: the jet wrappers, "still share
  sampling_derivatives"), `utv_sampling` (`:12-27`: the jet functions and the `*_from_sweep` seam), `fitting`
  (`:1-23`, about the `SamplingKind` descriptor; becomes a short docstring of the oracle and its batch),
  `uniform_fitting` (`:1-25`: "and the old packed SamplingKind classes", the protocol note).
- **Comments and docstrings of kept code that name deleted code** (`inventory_before_k0.md` §2, "Prose of kept
  code"). The module-level comment blocks between statements, measured with their neighbours (a run of comment
  lines no statement covers). Each was judged by one question (review r2, S2): **after K0, is the next statement below
  the block a kept one that the block heads or explains?** Deleted whole, since the answer is no: `probing.py:586-595`,
  `ut3_sampling.py:256-261`, `:310-315`, `:373-378`, `:634-638`; `utv_sampling.py:135-148`, `:252-262`, `:389-394`,
  `:528-534`; `fitting.py:107-113`, `:177-179`, `:297-314`, `:488-490`, `:527-529`, `:737-738`;
  `uniform_fitting.py:61-78`, `:457-462`. Kept or cut to what stays, since the answer is yes (`prose_edits.md` §2):
  the section headers `probing.py:274-276` (kept), `:672-674` (new banner), `:1236-1266` (banner kept, comment
  deleted), `ut3_sampling.py:124-133` and `:469-473` (rewritten), `utv_sampling.py:444` (one word); and
  `ut3_sampling.py:63-69` (the pre-engine pack / unpack / re-mask, deleted, the `.data` lines kept),
  `fitting.py:154-160` (the default-draw helpers, of which `_flat_gather` stays), `fitting.py:745-749` ("Beside the
  old kinds above, which the optimizers use until P16"). In kept code: `entries.py:115` (`_onehot_vectors`),
  `common.py:526` (`readonly_mask_copies`: "the `_weight_matrix` precedent"), `utv_sampling.py:451`
  (`_gauge_masks_over_Knew`: `sum_over_probes` → `sum_over_samples`), and any docstring of a kept function that cites
  a deleted one (the list is the inventory's; none is a cross-reference that Sphinx resolves, §2.2). Each edit
  removes or corrects a statement; none adds a description of the coming slices.
  Comments are content (`K_BRIEF.md` §5): the signature columns and diagrams of the kept functions are untouched.
- **Module doctests:** 5 docstrings with examples are deleted (all in deleted functions); 320 stay. No kept doctest
  names a deleted name (inventory §2).

### 1.4 Library code outside the seven modules

None refers to a deleted name in code or doctests (`t3toolbox/__init__.py`, `fitting.py`, `optimizers.py`,
`tucker_tensor_train.py`, `manifold.py`, the uniform frontends, `backend/optimizers.py`, `geometry.py`, `jets.py`,
`data_preparation.py`, `sampling_definitions.py`). Importing every module of the simulated K0 tree (58 modules, with
`pkgutil.walk_packages`) succeeds with jax and with jax made unimportable.

## 2. Every other place a deleted name is referenced

### 2.1 Examples

None. (`inventory_before_k0.md` §4 lists `fitting.PROBE` twice in `examples/fit_probe_display.py`: the word in a
printed heading, "PLAIN PROBE"; allowlisted in the gate.)

### 2.2 Pages, `api_reference.rst`, Sphinx

- **Page doctests:** none name a deleted name (§5 of the inventory: "None").
- **Sphinx `-W`:** `docs/conf.py` does not set `nitpicky`, so an unresolved `:py:func:` or `:py:mod:` role does
  not warn; no `:doc:` link or toctree names an autoapi page of a deleted module (`grep` over `docs/`); autoapi
  documents the modules that exist. So the least K0 must change for the docs build is nothing; the gate's Sphinx
  step is the check.
- **`docs/api_reference.rst`** (ruled, review M2: K0 removes the names of things that no longer exist, where the
  removal leaves the text correct; adding the new names is P19's). A scan of every `:py:` role of the page against
  the library finds nine missing targets today, all deleted in P16: the six model factories (`:75-80`) and
  `least_squares_problem`, `Problem`, `LeastSquaresSetup` (`:87-90`); and after K0 `sampling_derivatives` (`:130`).
  The two bullets go whole and `:130` loses the one module; the text around them stays correct (before and after:
  `k0/prose_edits.md` §3). Decision 4.
- **Left for P19** (prose and fenced lines, 16 pages; the full list is inventory §5): `entries_apply_probe.md`
  (13 lines, the `compute_*` helper chain), `fitting_and_optimization.md` (11, `SamplingKind` and its subclasses;
  two fenced lines), `naming_conventions.md` (10), `curvature_internals.md` (8), `glossary.md` (7),
  `parameters_not_closures.md` (4), `scan_body_principles.md` (4, one fenced), `deferred_and_rejected.md` (4),
  `batching_internals.md` (2), `ambient_derivative_transpose_note.md`, `signature_style.md` (fenced `compute_mu`,
  the signature exemplar), `reading_signatures.md` (fenced `compute_mu`), `verification_records.md`,
  `uniform_equivalence_contract.md` (`sumsq_over_probes`), `docs/release_notes.md` (on purpose). Two of these are
  more than prose and P19 should know: `signature_style.md` and `reading_signatures.md` quote `compute_mu`'s
  signature as the house exemplar of the shape comments; the quoted function no longer exists after K0.

### 2.3 Import-graph pins, module enumerations, conftest, CI

- `tests/test_import_graph.py` pins nothing K0 changes (acyclicity, `backend/optimizers`, the padded-frame modules,
  `safety`, `jets`, `data_preparation`); K0 removes edges only.
- `tests/conftest.py`, `pyproject.toml` and the CI workflows name no deleted file (`tests/bridge/` is collected by
  `pytest tests/`; the folder goes with its one file).
- `tests/test_jax_absent.py`: its subprocess scripts call no deleted name (P16 removed `probe_derivatives_kind`).

### 2.4 The two scan floors

`scan_vocabulary()` (`tests/test_contractions_interpreter.py:437`) collects the grouped-subscript literals of the
library; `TestVocabularyCompleteness` asserts more than 100, as does the copy in
`tests/test_contractions_sharding.py:102` (`TestInterpreterAnyAxisSharding`, which runs where jax has two devices).
Measured with the test module's own functions on the simulated tree (`simulate.py`):

| scan | today | after K0 | floor |
|---|---|---|---|
| `scan_vocabulary` (non-jet subscripts) | 209 | **86** | > 100 (two tests) |
| literal `contract(...)` call sites | 286 | 115 | > 90 |
| subscript literals scanned | 407 | 210 | > 100 |
| `jet_contract` literal call sites | 32 | 32 | >= 29 |

The other three hold. The floors are sanity checks against a scan that silently finds nothing, not tolerances. Open
point 1 recommends 60 for both (70% of what remains), with the reason in a comment.

## 3. The tests, file by file

Baseline: the `-v` run of §6. "Kept" = unchanged; "edited" = changed with the property kept; "re-based" = its
oracle (or its subject) moved off deleted code. Every deleted test is listed with its reason; the coverage of what
it pinned is §4.

### 3.1 Files deleted whole (48 tests)

| File | Tests | Reason |
|---|---|---|
| `tests/bridge/test_old_kernels_bridge.py` (and `tests/bridge/`) | 9 | the engine against the old kernels: the bridge, temporary by its docstring |
| `tests/backend/test_from_sweep_reuse.py` | 5 | old `*_from_sweep` against old one-shots |
| `tests/test_chunk_size_estimator.py` | 12 | the deleted chunk estimator |
| `tests/test_jet_recurrence.py` | 5 | old jet kernels against their `trs` twins |
| `tests/test_probe_derivatives.py` | 17 | 16: old derivative kernels against old dense oracles; 1 (`test_frontend_corewise_transpose_chunk_size`) compares one engine call with itself: `chunk_size` is accepted and ignored by `_derivative_data` (its docstring), so "chunked == dense" is the same call twice |
| `tests/_jet_trs_twins.py` (helper) | | twins of deleted kernels; its one other use, `binomial_combine_tensor` in `test_dispatch`, moves there (8 lines) |

### 3.2 `tests/backend/test_curvature.py` (96 tests: 21 deleted, 1 re-based, 7 kept unchanged through an edited helper)

- `import sampling_derivatives as sd` goes; the module docstring's T5 bullets for the deleted kernels go.
- **`TestProbeDerivativesTwoHole`** (8): deleted, subject `sd.t3_probe_corewise_derivatives_hessian[_from_sweep]`.
- **`TestDerivativeKindsTwoHole`** (10): deleted, subject `sd.t3_{apply,entries}_corewise_derivatives_hessian`;
  `test_order_zero_is_the_plain_two_hole` is a bridge (sd against the engine).
- **`TestPlainKindsTwoHole`** (9): the subject is the engine's plain two-holes, kept.
  - `_plain_ops(kind)` returns `(two-hole, one-hole)`; the `_from_sweep` and precompute entries go, callers updated.
  - kept: `test_against_the_block_sum_oracle`, `test_against_a_finite_difference_of_the_one_hole` (Richardson,
    exact to rounding), `test_stacked_equals_per_element`, `test_uniform_supercores`, `test_symmetry`.
  - deleted: `test_from_sweep_reuses_the_model_sweep` (subject deleted);
    `test_probe_pieces_against_finite_differences_of_the_gn_kernels` (subject: the deleted `probing.compute_*`);
    `test_apply_tt_assembly_is_the_three_term_product_rule` (its oracle is built from `compute_xi`, `compute_mu`,
    `compute_sigma_hat`, ...; it pinned the old factored assembly; the engine's apply two-hole is pinned whole by
    the block-sum and Richardson tests beside it).
  - **re-based: `test_residual_weights`.** Today the forward is `bfit.ProbeKind(weight=w).point_forward` /
    `bfit.APPLY` / `bfit.ENTRIES` and the weight `kind._apply_weight(r, 2)`. After: the forward is
    `probing.t3_probe(cores, sample)` / `bapply.t3_apply` / `bentries.t3_entries` and `w2(r)` is written out
    (probe: `[r_i * w[i]**2 for i]`, the per-mode row; apply, entries: none). The statement and the tolerances are
    unchanged (a central difference of `J^T(omega^2 r)` minus the Gauss-Newton term).
- **`TestPlainKindsExactLocalModel`** `test_corewise_is_the_plain_hessian`,
  `test_the_curvature_block_is_the_oracles_Kp_at_the_weighted_residual`: kept, through the edited `_plain_ops`.
- Helpers orphaned (measured): `_pd_problem`, `_pd_blocksum`, `_tree_relerr`, `_deriv_ops`, `_deriv_problem`,
  `_deriv_residual_at` go. (`_flat_inner` is unused already today; not K0's.)

**`tests/backend/_curvature_oracles.py`:** `import sd` goes; the "reference twin" family goes
(`_w_c_counts` through `apply_derivatives_two_hole_states_trs`, lines 265-546: twins of deleted kernels, and the
nine-term / three-term references of deleted assemblies), with its docstring paragraph; the imports `bentries`,
`probing`, `twins` then go. Kept: `doubled`, `blocksum`, the dense inner-product oracles, `dense_core_jacobian`,
`flatten`/`unflatten`, `dense_lagrangian_hessian`, the tied lift, `vertical_direction` (all used by kept tests).

### 3.3 `tests/backend/test_uniform_curvature.py` (56 tests: 14 deleted, 1 edited)

- `import sd` goes; module docstring follows.
- Deleted: `TestUniformProbeDerivativesTwoHole` (2), `TestUniformKindCorewiseHessian` (3),
  `TestUniformPlainKindCorewiseHessian` (3), `TestUniformDerivativeKindsTwoHole` (2),
  `TestUniformDerivativeKindCorewiseHessian` (3): their subjects are deleted (the uniform derivative two-holes, the
  uniform kinds).
- `TestUniformPlainKindsTwoHole.test_per_element_vs_ragged`: **edited** — the subject (`ut3_*_corewise_hessian`, the
  engine) against the ragged engine per element, exact masks, garbage in every padding: kept; the three lines
  comparing with `ut3_*_corewise_hessian_from_sweep` go. `_PLAIN_UNIFORM` keeps the two-hole only.
- `TestUniformPlainKindsTwoHole.test_the_direction_mask_is_load_bearing`: **deleted.** It is a negative control on
  the old kernel (`getattr(module, 't3_%s_corewise_hessian_from_sweep' % kind)` with the direction unmasked). Its
  engine analogue does not hold: measured (E2, `scratchpad/implementer/exp/e1_e2.py`), on a padded point with
  garbage in every padded slot of the direction, removing the engine's direction masks, and also its exit mask,
  leaves the real block of all three kinds' `Kp` exactly unchanged (relative change 0.0), because every other
  operand is masked on entry. The direction masks are load-bearing at TT slack, where
  `test_sampling_uniform.TestKpAtTTSlack` pins them permanently (probe). Ruled: deleted; apply and entries go to K5 (§8.4).
- Orphaned helpers: `_old_pack_sample`, `_old_pack_data`, `_deriv_uniform`, `_deriv_uniform_sample`,
  `_dirty_deriv_sample` go.

### 3.4 `tests/backend/test_uniform_fitting.py` (30 tests: 11 deleted)

Deleted, subjects deleted (the uniform kinds and the old seams): `TestUniformSamplingKind` (4),
`TestUniformBlockSumsq` (1), `TestUniformDerivativeSamplingKind` (4), `TestKindTransposeOutputHygiene` (1),
`TestSplitSeamDefaultParity` (1). Their class-level tables (`_KINDS` at `:105`, `:256-258`, `_SEAMS` at `:394`) are
evaluated at import, so they go with the classes. Orphaned helpers `_prefix`, `_corrupt`, `_expected_gauge_indicator`
go; imports `bfit`, `bapply`, `utv_sampling` if unused then. Kept (19): `TestUniformGeometryOps`,
`TestUniformSamplingMirror`, `TestUniformProblem`, `TestUniformOptimizers`, `TestUniformMinimalRank`,
`TestPackSampleAndData`, `TestSlackPaddedStart`.

### 3.5 `tests/test_dispatch.py` (108 tests: 1 deleted, 9 edited or re-based)

`import sampling_derivatives as pd` (`:37`) and `import _jet_trs_twins as twins` go.

| Test | Action |
|---|---|
| `test_jit_probe_derivatives` | **edited, renamed `test_jit_the_interpreter_on_multi_group_strings`**: the old kernels' jit checks go; the seven `contractions.contract` jit checks on multi-group strings stay (they test the interpreter, which stays), with `binomial_combine_tensor` inlined |
| `test_jit_probe_derivatives_two_hole` | **deleted**: subject `sd` |
| `test_jit_plain_kind_two_holes` | **edited**: the engine two-hole jit, the uniform face jit == eager and the exact `LocalModel` one-cache-key check stay; the `_from_sweep` recompile check and the old kinds in the table go |
| `test_jit_derivative_kind_two_holes` | **edited**: the kernel half goes; the exact `LocalModel` of the derivative fits (`F.derivative_fit`, the engine through the oracle), ragged and uniform, one cache key across points, stays |
| `test_jit_uniform_exact_ops` | **edited**: the old uniform kind (`okind`), its sweep and `ut3_probe_corewise_derivatives_hessian[_from_sweep]` go; the lift factors, lift, transpose, the T3-T3 ops and the geometry's four exact members stay |
| `test_jit_shared_uniform_exact` | **re-based data**: `pd.t3_probe_derivatives(ww, pp, x, order)` (synthetic data of a fit) becomes `dd.forward(x, 'probe', ww, pp, order)`, the same numbers to rounding through the engine |
| `test_jit_ragged_local_model_parameterized_kind` | **edited**: the assertion message names `SamplingKind`; it says "the oracle" |
| `TestJetDtypeUnderX64.test_float32_in_float32_out` | **re-based subject** (Decision 6): float32 in, float32 out with x64 on, through the engine's faces at a line `(2,)`: `t3_probe`, `tv_probe`, `tv_probe_transpose`, `t3_probe_corewise_hessian`, `tv_apply`, `tv_apply_transpose`, `tv_entries_transpose`, eager and under jit, uniform supercores and ragged numpy; the property (start values built in the operands' dtype) is the engine's too |
| `TestCurvatureJvpOracle.test_every_two_hole_is_the_jvp_of_its_transpose` | **re-based**: the three derivative rows become the engine's two-holes at jets, `t3_probe_corewise_hessian(x, line, zt, d, order=(2,))` against `jax.jvp` of `t3_probe_corewise_transpose(x, line, zt, order=(2,))`; apply likewise; the entries row is apply at the one-hot line (the entries kind has no jets). Measured (E3, `experiments/e3.py`): relative error <= 2.7e-16 on every row |
| `TestCurvatureJvpOracle.test_uniform_probe_derivatives_two_hole_is_the_jvp_of_its_transpose` | **re-based, renamed `test_uniform_probe_two_hole_at_jets_is_the_jvp_of_its_transpose`**: `ut3_probe_corewise_hessian` / `_transpose` at a packed line, garbage in the directions' padding; E3: 2.7e-16 |

### 3.6 `tests/test_uniform_probing.py` (35 tests: 2 re-based)

`import sampling_derivatives as bsd` goes. **`TestTransposeOutputHygiene`** (`test_tangent_transposes`,
`test_corewise_transposes`): the six derivative rows go (subjects deleted). The three plain rows stay with their
clauses (a) exact masks from the ragged shapes, (c) padding exactly zero, (d) garbage in the frame's or point's
padding changes nothing. Re-based:
- **(b) the real block**, today against the old `*_from_sweep` seam (by the dynamic `getattr` of `:913-914`) to
  1e-12: against the **ragged** face per stack element (`tv_*_transpose` on each ragged frame of `_frames()`,
  `t3_*_corewise_transpose` on each ragged point of `_points()`), to 1e-12 relative, as P12 set it.
- **The non-vacuity clause** ("the bare seam's padding is NOT zero", `_assert_nonvacuous`) cannot survive: E1
  (`experiments/e1_e2.py`) measured that the engine with its exit mask removed already returns zero padding on these
  inputs, the real block bitwise the same (the entry masks suffice), so there is no bare seam with garbage. Ruled
  (§8.2, with the reviewer's M4): it becomes a permanent negative control, in the form of `TestKpAtTTSlack`, that
  **the zero-padding check (c) itself fails** without the masks: on the garbage-padded input, with the engine's entry
  masks removed (`_UNIFORM_MASKS` emptied) and its exit mask removed (`_mask_variable` the identity), and for the
  tangent faces also their own re-mask (`ufv_masking.ufv_apply_variations_masks` returning the supercores as they
  are, inside `_remasked_transpose_output`), the assertion of (c) raises `'padding not zero'`. Measured (E1c,
  `experiments/e1c.py`): the padding then holds 8e31 to 5e32 on all six rows; without the third patch the tangent
  faces' padding stays 0.0, which is why it is needed. The control is one helper called per row, asserting with
  `assertRaisesRegex(AssertionError, 'padding not zero')`.

### 3.7 The others

| File: test | Action |
|---|---|
| `test_contractions_sharding.py`: `TestJetKernelsShardOverW` (1) | deleted (plan); `TestTheEngineShardsOverW` is its replacement. Floor at `:102` to 60 (§8.1) |
| `test_contractions_interpreter.py` | floor at `:551` to 60 (§8.1); the comments at `:237`, `:241`, `:299` annotate frozen rows of `HISTORICAL` and are all left as history (review M3) |
| `test_least_squares_oracle.py`: `TestTheBridgeToTheOldKinds` (1) | deleted: a bridge (the oracle against the old derivative kinds) |
| `test_sampling_definitions.py`: `TestAgainstTheOldKernels` (1) | deleted: the reference against the old kernels; the reference is pinned by brute force in the same file |
| `test_sampling_definitions.py`: `TestDenseFunctions.test_the_jets_of_entries_off_the_grid` | **re-based**: its subject is `dense_apply` at the one-hot line; the oracle `dense_entries_derivatives / t!` becomes its definition written out: slot `t` is the sum over the `t`-subsets `S` of the modes of `T` contracted with `pp[i]` for `i` in `S` and indexed at `index[i]` elsewhere (`itertools.combinations`, `np.take` and `np.tensordot`), zero above `d`. Independent of every library function but `dense_apply`, the subject. E4: agrees to 3.6e-15 at `n` = 1, 2, 4 |
| `test_tucker_tensor_train.py`: `test_derivative_methods`, `test_corewise_derivative_transpose_nontrivial_boundary` | **re-based**: the finite-difference forward `sampling_derivatives.t3_{probe,apply,entries}_derivatives(.., data, ORDER)` becomes the dense one, slot `t` of `dense_<kind>(t3_to_dense(data), affine_jet(ww, pp), (ORDER,))` times `t!`, the factorials written inline (`np.array([math.factorial(t) for t in range(ORDER + 1)])`, review M1: so the oracle names no library function but `dense_*`, and K6's renames do not touch it) (entries: `dense_apply` at the one-hot line), which shares no code with the engine (`probing._dense_jet`). E5: the finite differences agree to 3e-10 (apply), 3e-11 (entries), 5e-12 (probe) against the bar of 1e-5. The module import goes |
| `test_jets.py`: `test_the_validators_of_pp_are_gone`, `test_the_names_of_t3toolbox_fitting` | kept; they assert the names are absent (strings, allowlisted); the first's docstring ("the originals stay in `sampling_derivatives` ... until P20") is corrected |
| `tests/oracle_sweeps/o1_ragged.py`, `o1_uniform.py` | kept: labels only (§0.2); allowlisted |
| `tests/_derivative_data.py` | kept; its `chunk_size` parameter goes if no caller passes it after K0 (today only the deleted `test_probe_derivatives` and `test_curvature` do); `sum_over_probes` stays (ruled for P17, §8.3) |
| `tests/test_import_graph.py` | one test added, §3.8 |

### 3.8 One test added (Decision 5, revised by review S3)

`tests/test_import_graph.py`, `TestImportGraph.test_the_old_sampling_kernels_are_gone`:
`'t3toolbox.backend.sampling_derivatives'` is not among the package's module files (`MODULES`, the file's own map)
and `importlib.util.find_spec` finds nothing by that name. One assertion of absence, beside the file's other
statements about which modules exist and import what; no export list, no name pattern (K1, K2 and K5 change the
exports). The deletion of the names is pinned by `k0_inventory.py --check` at K0's gate.

## 4. What coverage the deletion removes

The deleted tests pinned three kinds of property. For each, where it is pinned after K0, against what.

**(a) The engine's numbers** (the bridge; `TestAgainstTheOldKernels`; `TestTheBridgeToTheOldKinds`; the "P12:
the engine against the old kernel" checks). Every variable the bridge compared (`z`, `mu`, `nu`, `eta`, the
tildes, `g`, `Jp`, `JtJp`, `Kp`; probe, apply, entries; jets `n = 0..3`; `K = (2,)`) is compared, every name, with
the dense reference `sampling_definitions` (one dense contraction per variable, evaluated on roots of unity, itself
pinned by brute force and by jax AD in `test_sampling_definitions.py`, `TestKp` among them):
`test_sampling_variables.TestByDefinition` (probe, every name incl. `Kp`, five problems: tails, `d = 1`, `d = 4`,
`W = ()`; seven jet cases), `TestTheDirectionStack` (`K = (2,)`), `test_sampling_apply`/`_entries`/`_corewise`
(`TestByDefinition`, both call types), `test_engine_data_space.TestAgainstTheReference` (weights, `Kp` included),
`test_sampling_stacks.TestTheReferenceAnchor`; uniform against ragged with garbage and exact zero padding:
`test_sampling_uniform.TestEquivalence`. **Lost:** the agreement with the released 2026.5.0 kernels, a second
implementation. By ruling (O-1, O-2) that comparison moves to Part 2's benchmark against the release. It matters
only as a cross-check of the dense reference's conventions (derivative vs Taylor coordinates, the turned seed);
those conversions are pinned in `test_jets` (`TestDerivativeData`, `TestDualJet`).

**(b) Properties of the deleted code** (sweep reuse bitwise, `chunk_size` and its estimator, the `trs` twins, the
factored assemblies, the uniform kinds' masking at TT slack, the old kinds' `take`): no subject is left. Where the
engine has the same concern it has its own test: one cache key across points through the oracle
(`test_dispatch`'s exact `LocalModel` checks, kept), sharding (`TestTheEngineShardsOverW`), the frame's masks at TT
slack (`TestKpAtTTSlack`; `TestUniformGeometryExactMembers.test_the_frame_masks_at_tt_slack_through_the_model`),
the batch's `take` (`test_least_squares_oracle`), the block sums (`test_block_sums`).

**(c) Mathematical identities of the two-holes at jets**, which the old tests pinned on the old kernels: the
block-sum identity, a Richardson difference of the one-hole, symmetry, the weighted curvature (T11), stacked ==
per element, uniform == ragged with garbage. For the engine at jets: `Kp` against the dense reference (whose `Kp` is
pinned by jax AD and finite differences, `test_sampling_definitions.TestKp`) for all three kinds and both call
types, with weights (`test_engine_data_space`) and stacks (`test_sampling_stacks`), uniform with garbage
(`test_sampling_uniform`); and `Kp` as the jvp of the corewise transpose at jets, by jax AD, re-based here
(§3.5). The plain kinds keep the block-sum, Richardson, symmetry and weights tests (§3.2). **Lost:** (i) the
containment identities as explicit tests (entries two-hole == apply at the one-hot vectors, bitwise; apply ==
probe at the paired residual jets): each side is pinned against the reference, and the first containment is pinned
for every engine name by `test_sampling_entries.TestAgainstApply`; it does not matter. (ii) A per-kind negative
control for the uniform direction mask (§3.3): the engine has no such property in general position (E2); at TT
slack it is pinned for the probe kind only; K5 adds apply and entries first (§8.4).

**(d) Re-based here, nothing lost:** the dtype under x64 (§3.5), the jvp oracle (§3.5), the hygiene of the plain
transposes (§3.6), the finite differences of `test_tucker_tensor_train` (§3.7), the entries jets off the grid
(§3.7), the plain kinds' weights (§3.2).

## 5. Not in K0, and the remnants

- Not in K0 (plan §7): the check-in on `t3_corewise_frame` / `ut3_corewise_frame` (after P19); the translation of
  the 2026.5.0 call form (until P17); the pages' prose (P19, §2.2).
- `chunk_size` after K0: the translation's old signatures (`t3toolbox/optimizers.py:133`, `:138`, `:143`, `:271`)
  and the old-form calls that pass it (`test_optimizers_frontend.py`, five lines; `test_dispatch.py:2137`, `:2155`)
  — P17's; test-local parameters that nothing reads any more (`tests/_derivative_data.py`, §3.7;
  `test_dispatch._uniform_pd_problem_jax`) go where K0 edits their callers anyway, else stay.
- `sum_over_probes` after K0: no library code; one docstring fixed in K0 (`utv_sampling.py:451`); tests: the helper
  keyword of `_derivative_data` and its callers, test-local labels (`test_uniform_probing`, `test_manifold`,
  `test_tucker_tensor_train`, `test_sampling_definitions:1369`, `oracle_sweeps/o1_*`). Ruled for P17 (§8.3).
- `ztildes`: none after K0 (only in deleted code and the deleted twin family). `pp=`: two keyword uses in tests,
  test-local helpers' (`test_fitting.py`, `test_jet_recurrence.py`, the latter deleted).
- `dev/graded_messages/README.md`: one line saying the prototype scripts of `scripts/` run against `v2026.5.0`
  (O-26, P20's list). Decision 7.

## 6. The order of the work, the gate and the evidence

**Order** (one slice, not in parts: no intermediate tree runs, since a module import fails until both its kernels
and its importers are gone):

1. Tests first, so that the suite can be run at once on the edited library: delete the five files and
   `tests/bridge/`; edit the files of §3.2 to §3.7; add §3.8 last.
   **Recorded step (review §6):** the edited test files run on the untouched library, where every re-base must pass
   (they use no deleted name), log kept in `k0/`; and the slow tier of the edited `test_uniform_curvature.py`
   serially, since its `_SLOW` tests do not run in the always-on suite. Only §3.8's assertion and the two lowered
   floors cannot be checked there (the module still exists; the floors pass a fortiori). This is the one point where a
   re-base can be told apart from a side effect of the deletion.
2. The library, module by module bottom up: `sampling_derivatives.py`; `probing`, `apply`, `entries`; `ut3_sampling`,
   `utv_sampling`; `fitting`, `uniform_fitting`; imports, `__all__`, docstrings and comments of §1.3.
3. `docs/api_reference.rst`, `dev/graded_messages/README.md`.

**Evidence:**

- **Baseline:** `k0/baseline_v.log.gz`, the untouched tree (a `git archive` of HEAD in scratch, `python -B`) at
  `-n 2 -v`: **1,979 passed, 5 skipped, 1 xfailed, 93,034 subtests passed, 1 warning, in 38:50**; 1,985 nodes, the
  same totals as M5's gate. Every node of the 98 deleted tests passes there (no skip among them).
- **The gate:** `run_gate.sh --detach -n 2` (suite, module doctests, quickstart, pages, examples, Sphinx `-W`), then
  the slow tier serially (`slow`: `test_oracle_sweep.py`, `test_uniform_curvature.py`, which K0 edits). Both CI
  legs after the push.
- **Per test:** a `-v` run of K0's tree and `k0/tally.py baseline new`: every node that differs is a deleted test
  of §3 (in the baseline only), a renamed one (read under its new name), §3.8's, or an edited test whose subtest
  count changes (listed in the handover with its expected count).
- **No reference:** `k0_inventory.py --check` on K0's tree: 0 references (code, doctests, strings outside the
  allowlist, computed names, page doctests; the deleted file absent). Today it reports 1,277.
- **Imports:** every module of the package imported with jax and with jax unimportable (the `pkgutil` loop of
  §1.4), on K0's tree.
- **Mutations** (scratch copies, `python -B`), each with its expected outcome. The step mutants scale one step's
  output by `(1 + eps)` (`k0/experiments/mutate_step.py`, appended to a scratch copy of `sampling_steps.py`). Measured on
  prototypes of the re-bases in a scratch copy (review r1 asked for a library mutation and a step per kind):
  - (i) **`TestPlainKindsTwoHole.test_residual_weights`** (re-based): `H_tilde_dot_step` ×(1+1e-3), the probe kind's
    `Kp` TT block: both probe subtests fail; `H_tilde_step` ×(1+1e-3), the apply and entries kinds' `Kp` TT block
    (and probe's `g`): all four subtests fail. Not catchable here, and not expected: a step shared by both sides of
    the identity (`V_top_step`, apply's `V` block of both `g` and `Kp`): measured, it passes; that block is pinned by
    `test_sampling_apply.TestByDefinition` against the dense reference. (The r1 mutation, the square dropped in the
    test's own weight, cannot fail: the identity holds for any weight. Withdrawn.)
  - (ii) the entries jets off the grid: one subset term dropped from the oracle fails the slots `t >= 1`.
  - (iii) `test_tucker_tensor_train`: the dense forward without its `t!` fails: measured on E5's apply and entries
    rows, relative error 0.92 and 1.0 against the bar of 1e-5 (the transpose takes its residual in derivative
    coordinates, so the pairing no longer matches).
  - (iv) **the jvp rows** (re-based): `H_tilde_dot_step` ×(1+1e-6) fails the probe and the uniform probe rows (4e-7 to
    8e-7 against 1e-12) and leaves apply alone; `H_tilde_step` ×(1+1e-6) fails the apply and the one-hot apply rows
    (4e-7 to 6.5e-7) and the probe rows. So every row has a step on its path that it catches (measured with E3).
  - (v) the hygiene negative control: with the masks in place the padding assertion passes; with them removed it
    raises (E1c), so the control cannot pass vacuously.
  - (vi) the dtype test: `start_zeros` built without `dtype` promotes `g`, `Kp`, `JtJp` to float64 (reviewer's run);
    the re-based op list reads them, so it fails. (A second guard: `test_sampling_stacks.TestTheDtypeMatrix` catches
    the same mutant, review §1.3.)
  - The floors of §8.1: a scan pointed at an empty directory fails both.
  - **For the build** (review r2): these mutations are rerun on the built tests, not on the prototypes, and their
    outcomes recorded in the handover.
- **`tests/test_import_graph.py`'s module docstring** gains a line for the new test (its list of pinned facts).

**Counts, measured on the baseline log** (to be confirmed by the tally): the 98 deleted tests hold 1,767 subtests.
After K0, expected: **1,882 passed** (1,979 - 98 + 1, the one being §3.8's), 5 skipped, 1 xfailed; subtests **93,034 - 1,767 = 91,267**,
less what the edited tests lose and plus what they gain, each stated in the handover against its baseline count:
`TestTransposeOutputHygiene` 336 and 192 (half of each are the derivative rows, which go; the plain rows keep theirs),
`TestJetDtypeUnderX64` 22, `TestCurvatureJvpOracle` 14 and 2, `test_jit_plain_kind_two_holes` 3,
`test_jit_derivative_kind_two_holes` 2, `TestUniformPlainKindsTwoHole.test_per_element_vs_ragged` 30,
`TestPlainKindsTwoHole.test_residual_weights` 4, `test_the_jets_of_entries_off_the_grid` 12. Module doctests: 5
fewer docstrings with examples (320 stay).

## 7. Decisions (mine; each recorded with its reason)

1. **The closure decides the private helpers** (§1.2), reading every module-level statement; the ambiguous names
   of M1(a) are settled by it (§1.2).
2. **`_t3_apply_step` and `_t3_entries_step` go.** Dead before K0 (no reference but `dev/` records): the scan
   bodies of the pre-engine `t3_apply`/`t3_entries`, pieces of the old kernels. Listed in `k0_inventory.py`'s `DEAD`.
3. **Imports unused after K0 go in the seven modules**, including those already unused before (remnants of the
   kernels the P slices moved); `uniform_fitting`'s `bfit` stays.
4. **`docs/api_reference.rst`** loses the names of what no longer exists (`:75-80`, `:87-90`, `:130`); ruled into K0
   (review M2), the least for `-W` being nothing.
5. **One permanent assertion of the deletion** (§3.8): the module's absence, in `test_import_graph.py` (review S3).
6. **Re-base rather than delete where the property is the engine's too**: the dtype under x64, the jvp oracle at
   jets, the hygiene of the plain transposes, the plain kinds' weights; **delete** where the subject is deleted code
   or the property has no engine analogue (§3.3).
7. **The prototype README line** (O-26) is in K0, as it was in P20.
8. **Two renames** (§3.5) where a kept test's name names a deleted kernel; the tally reads them.
9. **`tests/_jet_trs_twins.py` goes** with its subjects; `binomial_combine_tensor` moves to `test_dispatch`.
10. **Library prose is K0's, pages' prose P19's**: docstrings and comments of the kept library code that name
    deleted code are corrected here (they are part of the modules K0 edits and are rendered as the API reference),
    with nothing about the coming slices.

## 8. The open points, as ruled (2026-11-25)

1. **The scan floors:** both go from 100 to 60, each with a comment giving the count at K0 (86) and what the floor
   is for (a scan that silently finds nothing; every string found is still checked).
2. **The hygiene test's non-vacuity clause:** replaced by the masks-removed negative control, which asserts that the
   zero-padding check (c) fails without the masks (§3.6, E1c).
3. **`sum_over_probes` in the test helper:** left, not renamed in K0; the orchestrator records it for P17 by name.
   K0's gate does not check the word; the handover lists where it remains (§5).
4. **The per-kind direction-mask negative control:** deleted (§3.3). Its apply and entries cases go to K5, whose
   design note adds them first, against the engine as it then is, before it changes the masking (recorded by the
   orchestrator for K5).

No point is open.
