# K0 inventory

Tree: `git rev-parse HEAD` = 5848e687 (library, tests, docs and examples as at that commit).

## 1. What is deleted, per module

### `t3toolbox.backend.probing`

- by the rule (and the DEAD list) (27 names): `compute_xi`, `compute_mu`, `compute_nu`, `compute_eta`, `assemble_z`, `compute_dxi`, `compute_sigma`, `compute_tau`, `compute_deta`, `assemble_tangent_z`, `tv_probe_jacobian_from_sweep`, `tv_precompute_probe_frame_sweep`, `compute_sigma_hat`, `compute_deta_tilde`, `compute_tau_tilde`, `compute_sigma_tilde`, `compute_dxi_tilde`, `assemble_tucker_variations`, `assemble_tt_variations`, `tv_probe_transpose_from_sweep`, `compute_deta_tilde_tangent`, `compute_tau_tilde_tangent`, `compute_sigma_tilde_tangent`, `compute_dxi_tilde_tangent`, `assemble_tucker_tangent_variations`, `assemble_tt_tangent_variations`, `t3_probe_corewise_hessian_from_sweep`
- by the closure (23 statements):
  - `_xi_step` (line 116): orphaned: referred to only by probing.compute_xi
  - `_mu_step` (line 150): orphaned: referred to only by probing.compute_mu
  - `_eta_step` (line 206): orphaned: referred to only by probing.compute_eta
  - `_assemble_z_step` (line 241): orphaned: referred to only by probing.assemble_z
  - `_sigma_step` (line 305): orphaned: referred to only by apply._apply_from_xis_step, probing._sigma_sweep_step, probing._tau_tilde_tangent_step
  - `_sigma_sweep_step` (line 322): orphaned: referred to only by probing.compute_sigma
  - `_deta_step` (line 404): orphaned: referred to only by probing.compute_deta
  - `_assemble_tangent_z_step` (line 466): orphaned: referred to only by probing.assemble_tangent_z
  - `_sigma_hat_step` (line 633): orphaned: referred to only by probing.compute_sigma_hat
  - `_deta_tilde_step` (line 676): orphaned: referred to only by probing.compute_deta_tilde
  - `_tau_tilde_step` (line 716): orphaned: referred to only by probing.compute_tau_tilde
  - `_dxi_tilde_step` (line 781): orphaned: referred to only by probing.compute_dxi_tilde
  - `_assemble_tucker_variations_core` (line 827): orphaned: referred to only by probing._assemble_tucker_variations_step_summed, probing._assemble_tucker_variations_step_unsummed
  - `_assemble_tucker_variations_step_summed` (line 855): orphaned: referred to only by probing.assemble_tucker_variations
  - `_assemble_tucker_variations_step_unsummed` (line 860): orphaned: referred to only by probing.assemble_tucker_variations
  - `_assemble_tt_variations_core` (line 903): orphaned: referred to only by probing._assemble_tt_variations_step_summed, probing._assemble_tt_variations_step_unsummed
  - `_assemble_tt_variations_step_summed` (line 936): orphaned: referred to only by probing.assemble_tt_variations
  - `_assemble_tt_variations_step_unsummed` (line 942): orphaned: referred to only by probing.assemble_tt_variations
  - `_deta_tilde_tangent_step` (line 1268): orphaned: referred to only by probing.compute_deta_tilde_tangent
  - `_tau_tilde_tangent_step` (line 1295): orphaned: referred to only by probing.compute_tau_tilde_tangent
  - `_tucker_tangent_variations_step` (line 1387): orphaned: referred to only by probing.assemble_tucker_tangent_variations
  - `_tt_tangent_variations_step` (line 1417): orphaned: referred to only by probing.assemble_tt_tangent_variations
  - `_tt_tangent_variations_core` (line 1430): orphaned: referred to only by probing._tt_tangent_variations_step, probing.assemble_tt_tangent_variations
- kept: `Expr at line 5`, `__all__`, `t3_probe`, `tv_probe`, `tv_probe_transpose`, `t3_probe_ambient_transpose`, `t3_probe_corewise_tangent`, `t3_probe_corewise_transpose`, `t3_probe_corewise_hessian`, `_dense_contract_one_mode`, `_dense_jet`, `dense_probe`
- `__all__` after: `t3_probe`, `tv_probe`, `tv_probe_transpose`, `t3_probe_ambient_transpose`, `t3_probe_corewise_tangent`, `t3_probe_corewise_transpose`, `t3_probe_corewise_hessian`, `dense_probe`

### `t3toolbox.backend.apply`

- by the rule (and the DEAD list) (5 names): `_t3_apply_step`, `tv_precompute_apply_frame_sweep`, `tv_apply_jacobian_from_sweep`, `tv_apply_transpose_from_sweep`, `t3_apply_corewise_hessian_from_sweep`
- by the closure (4 statements):
  - `_apply_from_xis_step` (line 591): orphaned: referred to only by apply._apply_from_xis
  - `_apply_from_xis` (line 601): orphaned: referred to only by apply.tv_apply_jacobian_from_sweep, entries.tv_entries_jacobian_from_sweep
  - `_apply_transpose_adjoint` (line 617): orphaned: referred to only by apply.tv_apply_transpose_from_sweep, entries.tv_entries_transpose_from_sweep
  - `_apply_corewise_hessian_adjoint` (line 663): orphaned: referred to only by apply.t3_apply_corewise_hessian_from_sweep, entries.t3_entries_corewise_hessian_from_sweep
- kept: `Expr at line 5`, `__all__`, `t3_apply`, `t3_apply_ambient_transpose`, `tv_apply`, `tv_apply_transpose`, `t3_apply_corewise_tangent`, `t3_apply_corewise_transpose`, `t3_apply_corewise_hessian`, `dense_apply`
- `__all__` after: `t3_apply`, `t3_apply_ambient_transpose`, `tv_apply`, `tv_apply_transpose`, `t3_apply_corewise_tangent`, `t3_apply_corewise_transpose`, `t3_apply_corewise_hessian`, `dense_apply`

### `t3toolbox.backend.entries`

- by the rule (and the DEAD list) (5 names): `_t3_entries_step`, `tv_precompute_entries_frame_sweep`, `tv_entries_jacobian_from_sweep`, `tv_entries_transpose_from_sweep`, `t3_entries_corewise_hessian_from_sweep`
- by the closure (2 statements):
  - `_entry_xis` (line 528): orphaned: referred to only by entries.t3_entries_corewise_hessian_from_sweep, entries.tv_entries_jacobian_from_sweep, entries.tv_precompute_entries_frame_sweep, sampling_derivatives.t3_entries_corewise_derivatives_hessian_from_sweep, sampling_derivatives.t3_entries_derivatives, sampling_derivatives.tv_entries_jacobian_derivatives_from_sweep, sampling_derivatives.tv_precompute_entries_frame_sweep_jets
  - `_onehot_vectors` (line 560): orphaned: referred to only by entries.t3_entries_corewise_hessian_from_sweep, entries.tv_entries_transpose_from_sweep, sampling_derivatives._entries_ambient_jets
- kept: `Expr at line 5`, `__all__`, `t3_entries`, `t3_entries_ambient_transpose`, `tv_entries`, `tv_entries_transpose`, `t3_entries_corewise_tangent`, `t3_entries_corewise_transpose`, `t3_entries_corewise_hessian`, `dense_entries`
- `__all__` after: `t3_entries`, `t3_entries_ambient_transpose`, `tv_entries`, `tv_entries_transpose`, `t3_entries_corewise_tangent`, `t3_entries_corewise_transpose`, `t3_entries_corewise_hessian`, `dense_entries`

### `t3toolbox.backend.ut3_sampling`

- by the rule (and the DEAD list) (15 names): `ut3_probe_derivatives`, `ut3_apply_derivatives`, `ut3_entries_derivatives`, `ut3_apply_corewise_derivatives_transpose`, `ut3_entries_corewise_derivatives_transpose`, `ut3_probe_corewise_derivatives_transpose`, `ut3_probe_corewise_derivatives_hessian`, `ut3_probe_corewise_derivatives_hessian_from_sweep`, `ut3_probe_corewise_hessian_from_sweep`, `ut3_apply_corewise_hessian_from_sweep`, `ut3_entries_corewise_hessian_from_sweep`, `ut3_apply_corewise_derivatives_hessian`, `ut3_apply_corewise_derivatives_hessian_from_sweep`, `ut3_entries_corewise_derivatives_hessian`, `ut3_entries_corewise_derivatives_hessian_from_sweep`
- by the closure (2 statements):
  - `_remasked_core_gradient` (line 135): orphaned: referred to only by ut3_sampling.ut3_apply_corewise_derivatives_hessian, ut3_sampling.ut3_apply_corewise_derivatives_hessian_from_sweep, ut3_sampling.ut3_apply_corewise_derivatives_transpose, ut3_sampling.ut3_apply_corewise_hessian_from_sweep, ut3_sampling.ut3_entries_corewise_derivatives_hessian, ut3_sampling.ut3_entries_corewise_derivatives_hessian_from_sweep, ut3_sampling.ut3_entries_corewise_derivatives_transpose, ut3_sampling.ut3_entries_corewise_hessian_from_sweep, ut3_sampling.ut3_probe_corewise_derivatives_hessian, ut3_sampling.ut3_probe_corewise_derivatives_hessian_from_sweep, ut3_sampling.ut3_probe_corewise_derivatives_transpose, ut3_sampling.ut3_probe_corewise_hessian_from_sweep
  - `_masked_point_and_directions` (line 475): orphaned: referred to only by ut3_sampling.ut3_apply_corewise_derivatives_hessian, ut3_sampling.ut3_apply_corewise_derivatives_hessian_from_sweep, ut3_sampling.ut3_apply_corewise_hessian_from_sweep, ut3_sampling.ut3_entries_corewise_derivatives_hessian, ut3_sampling.ut3_entries_corewise_derivatives_hessian_from_sweep, ut3_sampling.ut3_entries_corewise_hessian_from_sweep, ut3_sampling.ut3_probe_corewise_hessian_from_sweep
- kept: `Expr at line 5`, `__all__`, `UT3Data`, `Vectors`, `ut3_entries`, `ut3_apply`, `ut3_probe`, `ut3_full_sum`, `ut3_apply_corewise_tangent`, `ut3_apply_corewise_transpose`, `ut3_entries_corewise_tangent`, `ut3_entries_corewise_transpose`, `ut3_probe_corewise_tangent`, `ut3_probe_corewise_transpose`, `ut3_probe_corewise_hessian`, `ut3_apply_corewise_hessian`, `ut3_entries_corewise_hessian`
- `__all__` after: `ut3_entries`, `ut3_apply`, `ut3_probe`, `ut3_full_sum`, `ut3_apply_corewise_tangent`, `ut3_apply_corewise_transpose`, `ut3_entries_corewise_tangent`, `ut3_entries_corewise_transpose`, `ut3_probe_corewise_tangent`, `ut3_probe_corewise_transpose`, `ut3_probe_corewise_hessian`, `ut3_apply_corewise_hessian`, `ut3_entries_corewise_hessian`

### `t3toolbox.backend.utv_sampling`

- by the rule (and the DEAD list) (24 names): `utv_precompute_apply_frame_sweep`, `utv_precompute_probe_frame_sweep`, `utv_precompute_entries_frame_sweep`, `utv_apply_jacobian_from_sweep`, `utv_probe_jacobian_from_sweep`, `utv_entries_jacobian_from_sweep`, `utv_apply_transpose_from_sweep`, `utv_probe_transpose_from_sweep`, `utv_entries_transpose_from_sweep`, `utv_precompute_apply_frame_sweep_jets`, `utv_precompute_probe_frame_sweep_jets`, `utv_precompute_entries_frame_sweep_jets`, `utv_apply_jacobian_derivatives_from_sweep`, `utv_probe_jacobian_derivatives_from_sweep`, `utv_entries_jacobian_derivatives_from_sweep`, `utv_apply_transpose_derivatives_from_sweep`, `utv_probe_transpose_derivatives_from_sweep`, `utv_entries_transpose_derivatives_from_sweep`, `utv_probe_derivatives`, `utv_apply_derivatives`, `utv_entries_derivatives`, `utv_probe_derivatives_transpose`, `utv_apply_derivatives_transpose`, `utv_entries_derivatives_transpose`
- by the closure (1 statements):
  - `_mask_once` (line 77): orphaned: referred to only by utv_sampling.utv_apply_derivatives, utv_sampling.utv_entries_derivatives, utv_sampling.utv_probe_derivatives
- kept: `Expr at line 5`, `Vectors`, `__all__`, `utv_probe`, `utv_apply`, `utv_entries`, `_gauge_masks_over_Knew`, `_remasked_transpose_output`, `utv_probe_transpose`, `utv_apply_transpose`, `utv_entries_transpose`
- `__all__` after: `utv_probe`, `utv_apply`, `utv_entries`, `utv_probe_transpose`, `utv_apply_transpose`, `utv_entries_transpose`

### `t3toolbox.backend.fitting`

- by the rule (and the DEAD list) (20 names): `sumsq_over_samples`, `sumsq_over_probes`, `block_sumsq_over_samples`, `block_sumsq_over_probes`, `SamplingKind`, `ScalarOutputKind`, `ProbeOutputKind`, `ApplyKind`, `EntriesKind`, `ProbeKind`, `ApplyDerivativesKind`, `EntriesDerivativesKind`, `ProbeDerivativesKind`, `APPLY`, `ENTRIES`, `PROBE`, `probe_kind`, `apply_derivatives_kind`, `entries_derivatives_kind`, `probe_derivatives_kind`
- by the closure (14 statements):
  - `_prod_w` (line 174): orphaned: referred to only by fitting._IndexPpSample, fitting._IndexSample, fitting._WwPpSample, fitting._WwSample
  - `_take_apply` (line 180): orphaned: referred to only by fitting.ApplyKind
  - `_take_probe` (line 184): orphaned: referred to only by fitting.ProbeKind
  - `_take_entries` (line 188): orphaned: referred to only by fitting.EntriesKind
  - `_take_deriv_apply` (line 192): orphaned: referred to only by fitting.ApplyDerivativesKind
  - `_take_deriv_entries` (line 197): orphaned: referred to only by fitting.EntriesDerivativesKind
  - `_take_deriv_probe` (line 202): orphaned: referred to only by fitting.ProbeDerivativesKind
  - `_weight_matrix` (line 208): orphaned: referred to only by fitting.ProbeKind, fitting._DerivativesKind
  - `_make_weight` (line 246): orphaned: referred to only by fitting.SamplingKind
  - `_WwSample` (line 491): orphaned: referred to only by fitting.ApplyKind, fitting.ProbeKind
  - `_IndexSample` (line 500): orphaned: referred to only by fitting.EntriesKind
  - `_WwPpSample` (line 509): orphaned: referred to only by fitting.ApplyDerivativesKind, fitting.ProbeDerivativesKind
  - `_IndexPpSample` (line 518): orphaned: referred to only by fitting.EntriesDerivativesKind
  - `_DerivativesKind` (line 624): orphaned: referred to only by fitting.ApplyDerivativesKind, fitting.EntriesDerivativesKind, fitting.ProbeDerivativesKind
- kept: `Expr at line 1`, `If at line 42`, `__all__`, `_flat_gather`, `_KINDS`, `_HOLD_LIST`, `_SCALED`, `LeastSquaresBatch`, `_BATCH_FIELDS`, `If at line 811`, `_same_leaves`, `_scale_tree`, `_arrays_of`, `_stack_of_frame`, `_y_times_metric_y`, `_y_metric_y`, `_y_block_metric_y`, `_take_W`, `_take_data_space`, `LeastSquaresOracle`
- `__all__` after: `LeastSquaresOracle`, `LeastSquaresBatch`

### `t3toolbox.backend.uniform_fitting`

- by the rule (and the DEAD list) (8 names): `UniformApplyKind`, `UniformEntriesKind`, `UniformProbeKind`, `UniformApplyDerivativesKind`, `UniformEntriesDerivativesKind`, `UniformProbeDerivativesKind`, `uniform_sampling_kind`, `uniform_derivatives_kind`
- by the closure (11 statements):
  - `_frame_theta` (line 79): orphaned: referred to only by uniform_fitting.UniformApplyDerivativesKind, uniform_fitting.UniformApplyKind, uniform_fitting.UniformEntriesDerivativesKind, uniform_fitting.UniformEntriesKind, uniform_fitting.UniformProbeDerivativesKind, uniform_fitting.UniformProbeKind
  - `_UniformLayerMixin` (line 90): orphaned: referred to only by uniform_fitting.UniformApplyDerivativesKind, uniform_fitting.UniformApplyKind, uniform_fitting.UniformEntriesDerivativesKind, uniform_fitting.UniformEntriesKind, uniform_fitting.UniformProbeDerivativesKind, uniform_fitting.UniformProbeKind
  - `_SAMPLING_KIND` (line 327): orphaned: referred to only by uniform_fitting.uniform_sampling_kind
  - `_DERIV_SAMPLING_KIND` (line 328): orphaned: referred to only by uniform_fitting.uniform_derivatives_kind
  - `_fg` (line 463): orphaned: referred to only by uniform_fitting._ptake_apply, uniform_fitting._ptake_deriv_apply, uniform_fitting._ptake_deriv_entries, uniform_fitting._ptake_deriv_probe, uniform_fitting._ptake_entries, uniform_fitting._ptake_probe
  - `_ptake_apply` (line 465): orphaned: referred to only by uniform_fitting.UniformApplyKind
  - `_ptake_probe` (line 469): orphaned: referred to only by uniform_fitting.UniformProbeKind
  - `_ptake_entries` (line 473): orphaned: referred to only by uniform_fitting.UniformEntriesKind
  - `_ptake_deriv_apply` (line 477): orphaned: referred to only by uniform_fitting.UniformApplyDerivativesKind
  - `_ptake_deriv_entries` (line 481): orphaned: referred to only by uniform_fitting.UniformEntriesDerivativesKind
  - `_ptake_deriv_probe` (line 485): orphaned: referred to only by uniform_fitting.UniformProbeDerivativesKind
- kept: `Expr at line 4`, `__all__`, `pack_sample`, `pack_data`, `pack_batch`, `uniform_minimal`
- `__all__` after: `uniform_minimal`, `pack_sample`, `pack_data`, `pack_batch`

### `t3toolbox.backend.sampling_derivatives`: whole, 169 top-level statements, 3535 lines, `__all__` of 57

### Public names swept by the closure (not named by the rule)

None.

### Top-level statements of the seven modules that no library code refers to (dead before K0)

None.

## 2. Kept library code that refers to a deleted name

### Code (imports, calls, attributes, strings)

- `t3toolbox/backend/apply.py:23` (`import line 23`): `probing.compute_xi` (import)
- `t3toolbox/backend/apply.py:23` (`import line 23`): `probing.compute_mu` (import)
- `t3toolbox/backend/apply.py:23` (`import line 23`): `probing.compute_dxi` (import)
- `t3toolbox/backend/apply.py:23` (`import line 23`): `probing.compute_sigma_hat` (import)
- `t3toolbox/backend/apply.py:23` (`import line 23`): `probing._sigma_step` (import)
- `t3toolbox/backend/entries.py:26` (`import line 26`): `probing.compute_mu` (import)
- `t3toolbox/backend/entries.py:27` (`import line 27`): `apply._apply_from_xis` (import)
- `t3toolbox/backend/entries.py:27` (`import line 27`): `apply._apply_transpose_adjoint` (import)
- `t3toolbox/backend/entries.py:27` (`import line 27`): `apply._apply_corewise_hessian_adjoint` (import)
- `t3toolbox/backend/fitting.py:34` (`import line 34`): imports `t3toolbox.backend.sampling_derivatives`
- `t3toolbox/backend/ut3_sampling.py:26` (`import line 26`): imports `t3toolbox.backend.sampling_derivatives`
- `t3toolbox/backend/utv_sampling.py:35` (`import line 35`): imports `t3toolbox.backend.sampling_derivatives`

### Doctests of kept code

None.

### Prose of kept code (docstrings and comments; text, not references)

- `t3toolbox/backend/common.py`: 1: 526 _weight_matrix (docstring)
- `t3toolbox/backend/entries.py`: 1: 115 _onehot_vectors (comment)
- `t3toolbox/backend/fitting.py`: 4: 12 SamplingKind (docstring); 309 ScalarOutputKind (comment); 309 ProbeOutputKind (comment); 738 probe_derivatives_kind (comment)
- `t3toolbox/backend/probing.py`: 8: 8 assemble_z (docstring); 1245 sampling_derivatives (comment); 1245 t3_probe_corewise_derivatives_hessian (comment); 1256 compute_sigma (comment); 1256 compute_tau (comment); 1256 compute_deta (comment); 1258 _sigma_step (comment); 1260 compute_deta (comment)
- `t3toolbox/backend/sampling_derivatives.py`: 23: 44 compute_mu_jets (comment); 45 compute_nu_jets (comment); 46 compute_eta_jets (comment); 47 assemble_z_jets (comment); 291 build_input_jets (comment); 293 _shift_down_pair (comment); 294 _shift_up_pair (comment); 300 _shift_down (comment); 303 _shift_up (comment); 312 compute_eta_jets (comment); 547 SamplingKind (comment); 555 tv_precompute_apply_frame_sweep (comment); 1221 tv_probe_derivatives (comment); 1785 estimate_chunk_size (comment); 1786 max_chunk_size_within (comment); 2100 compute_sigma_hat_jets (comment); 2469 _adj_tilde_step (comment); 2470 _dxi_tilde_order_step (comment); 3099 t3_apply_corewise_derivatives_transpose (comment); 3099 t3_entries_corewise_derivatives_transpose (comment); 3108 _sigma_hat_jets_step (comment); 3108 _sigma_hat_tangent_jets_step (comment); 3110 _dxi_hat_order_step (comment)
- `t3toolbox/backend/uniform_fitting.py`: 3: 5 SamplingKind (docstring); 74 _UniformLayerMixin (comment); 457 SamplingKind (comment)
- `t3toolbox/backend/ut3_sampling.py`: 5: 11 sampling_derivatives (docstring); 18 sampling_derivatives (docstring); 259 sampling_derivatives (comment); 313 sampling_derivatives (comment); 374 ut3_probe_corewise_derivatives_transpose (comment)
- `t3toolbox/backend/utv_sampling.py`: 5: 136 SamplingKind (comment); 254 sampling_derivatives (comment); 262 SamplingKind (comment); 392 sampling_derivatives (comment); 532 sampling_derivatives (comment)

### Imports of the seven modules that no kept code uses after K0 (used before, by deleted code only)

A name counts as used when a `Name` node of kept code (its doctests excluded: they import for themselves) reads it; a string annotation is not a use.

- `t3toolbox/backend/probing.py:21`: `tt_operations`
- `t3toolbox/backend/apply.py:18`: `contractions`
- `t3toolbox/backend/apply.py:23`: `compute_xi`
- `t3toolbox/backend/apply.py:23`: `compute_mu`
- `t3toolbox/backend/apply.py:23`: `compute_dxi`
- `t3toolbox/backend/apply.py:23`: `compute_sigma_hat`
- `t3toolbox/backend/apply.py:23`: `_sigma_step`
- `t3toolbox/backend/entries.py:19`: `contractions`
- `t3toolbox/backend/entries.py:25`: `probing`
- `t3toolbox/backend/entries.py:26`: `compute_mu`
- `t3toolbox/backend/entries.py:27`: `_apply_from_xis`
- `t3toolbox/backend/entries.py:27`: `_apply_transpose_adjoint`
- `t3toolbox/backend/entries.py:27`: `_apply_corewise_hessian_adjoint`
- `t3toolbox/backend/ut3_sampling.py:23`: `apply`
- `t3toolbox/backend/ut3_sampling.py:24`: `entries`
- `t3toolbox/backend/ut3_sampling.py:25`: `probing`
- `t3toolbox/backend/ut3_sampling.py:26`: `sampling_derivatives`
- `t3toolbox/backend/ut3_sampling.py:27`: `ut3_masking`
- `t3toolbox/backend/utv_sampling.py:32`: `probing`
- `t3toolbox/backend/utv_sampling.py:33`: `apply`
- `t3toolbox/backend/utv_sampling.py:34`: `entries`
- `t3toolbox/backend/utv_sampling.py:35`: `sampling_derivatives`
- `t3toolbox/backend/fitting.py:25`: `abc`
- `t3toolbox/backend/fitting.py:26`: `ft`
- `t3toolbox/backend/fitting.py:33`: `probing`
- `t3toolbox/backend/fitting.py:34`: `pd`
- `t3toolbox/backend/fitting.py:35`: `bapply`
- `t3toolbox/backend/fitting.py:36`: `bentries`
- `t3toolbox/backend/uniform_fitting.py:32`: `readonly_mask_copies`
- `t3toolbox/backend/uniform_fitting.py:33`: `bfit`
- `t3toolbox/backend/uniform_fitting.py:36`: `ufv_masking`
- `t3toolbox/backend/uniform_fitting.py:38`: `utv_sampling`
- `t3toolbox/backend/uniform_fitting.py:39`: `ut3_sampling`

Unused already before K0 (not K0's; listed for the record):

- `t3toolbox/backend/probing.py:16`: `np`
- `t3toolbox/backend/probing.py:19`: `t3_conversions`
- `t3toolbox/backend/probing.py:20`: `fv_conversions`
- `t3toolbox/backend/probing.py:23`: `ragged_ops`
- `t3toolbox/backend/probing.py:24`: `uniform_ops`
- `t3toolbox/backend/apply.py:16`: `t3_conversions`
- `t3toolbox/backend/apply.py:17`: `fv_conversions`
- `t3toolbox/backend/apply.py:21`: `math`
- `t3toolbox/backend/entries.py:16`: `np`
- `t3toolbox/backend/entries.py:20`: `fv_conversions`
- `t3toolbox/backend/entries.py:24`: `math`
- `t3toolbox/backend/ut3_sampling.py:20`: `np`
- `t3toolbox/backend/utv_sampling.py:29`: `np`
- `t3toolbox/backend/uniform_fitting.py:34`: `geometry_module`
- `t3toolbox/backend/uniform_fitting.py:35`: `ufv_conversions`
- `t3toolbox/backend/uniform_fitting.py:37`: `utv_ops`
- `t3toolbox/backend/uniform_fitting.py:42`: `sharing_module`

## 3. Tests

### `tests/test_chunk_size_estimator.py`: 12 tests; at import: YES; direct 12; through a helper 0

- module level: 26 sampling_derivatives.<module> (import-module)
- `TestChunkSizeEstimator.test_balanced_tracks_the_jet_floor` (line 43): `sampling_derivatives._assembly_per_row_bytes`, `sampling_derivatives._jet_floor_bytes`, `sampling_derivatives.estimate_chunk_size`; through tests/test_chunk_size_estimator.py:_per_row
- `TestChunkSizeEstimator.test_budget_is_respected` (line 63): `sampling_derivatives.max_chunk_size_within`; through tests/test_chunk_size_estimator.py:_per_row
- `TestChunkSizeEstimator.test_sharding_sizes_the_local_shard` (line 75): `sampling_derivatives.estimate_chunk_size`
- `TestChunkSizeEstimator.test_small_problem_is_dense` (line 81): `sampling_derivatives.estimate_chunk_size`
- `TestChunkSizeEstimator.test_per_row_cost_is_independent_of_x64` (line 86): `sampling_derivatives._assembly_per_row_bytes`, `sampling_derivatives._two_hole_assembly_per_row_bytes`
- `TestChunkSizeEstimator.test_returns_plain_int` (line 102): `sampling_derivatives.estimate_chunk_size`
- `TestChunkSizeEstimator.test_frame_stack_shrinks_the_budget_chunk` (line 106): `sampling_derivatives._assembly_per_row_bytes`, `sampling_derivatives.max_chunk_size_within`
- `TestChunkSizeEstimator.test_frame_stack_leaves_the_default_untouched` (line 126): `sampling_derivatives.estimate_chunk_size`
- `TestChunkSizeEstimator.test_balanced_is_only_mildly_sensitive_to_the_frame_stack` (line 134): `sampling_derivatives.estimate_chunk_size`
- `TestChunkSizeEstimator.test_hessian_also_bounds_the_two_hole` (line 143): `sampling_derivatives._assembly_per_row_bytes`, `sampling_derivatives._jet_floor_bytes`, `sampling_derivatives._two_hole_assembly_per_row_bytes`, `sampling_derivatives._two_hole_jet_floor_bytes`, `sampling_derivatives.estimate_chunk_size`, `sampling_derivatives.max_chunk_size_within`
- `TestChunkSizeEstimator.test_hessian_mode_is_the_solver_vocabulary` (line 175): `sampling_derivatives.estimate_chunk_size`, `sampling_derivatives.max_chunk_size_within`
- `TestBudgetHoldsInTheCompiledProgram.test_chunk_dependent_peak_within_the_budget` (line 205): `sampling_derivatives.max_chunk_size_within`, `sampling_derivatives.t3_probe_corewise_derivatives_hessian`, `sampling_derivatives.tv_probe_derivatives_transpose`, `sampling_derivatives.tv_probe_transpose_derivatives_from_sweep`

### `tests/test_contractions_sharding.py`: 5 tests; at import: no; direct 1; through a helper 0

- `TestJetKernelsShardOverW.test_probe_transpose_and_two_hole_shard_over_W` (line 260): `sampling_derivatives.<module>`, `sampling_derivatives.t3_probe_corewise_derivatives_hessian`, `sampling_derivatives.tv_probe_derivatives_transpose`

### `tests/test_dispatch.py`: 108 tests; at import: YES; direct 10; through a helper 0

- module level: 37 sampling_derivatives.<module> (import-module)
- `TestDispatch.test_jit_probe_derivatives` (line 330): `sampling_derivatives.t3_apply_corewise_derivatives_transpose`, `sampling_derivatives.t3_apply_derivatives`, `sampling_derivatives.t3_entries_corewise_derivatives_transpose`, `sampling_derivatives.t3_entries_derivatives`, `sampling_derivatives.t3_probe_corewise_derivatives_transpose`, `sampling_derivatives.t3_probe_derivatives`, `sampling_derivatives.tv_apply_derivatives`, `sampling_derivatives.tv_apply_derivatives_transpose`, `sampling_derivatives.tv_entries_derivatives`, `sampling_derivatives.tv_entries_derivatives_transpose`, `sampling_derivatives.tv_probe_derivatives`, `sampling_derivatives.tv_probe_derivatives_transpose`
- `TestDispatch.test_jit_probe_derivatives_two_hole` (line 1455): `sampling_derivatives.t3_probe_corewise_derivatives_hessian`, `sampling_derivatives.t3_probe_corewise_derivatives_hessian_from_sweep`, `sampling_derivatives.tv_precompute_probe_frame_sweep_jets`
- `TestDispatch.test_jit_plain_kind_two_holes` (line 1500): `apply.t3_apply_corewise_hessian_from_sweep`, `apply.tv_precompute_apply_frame_sweep`, `entries.t3_entries_corewise_hessian_from_sweep`, `entries.tv_precompute_entries_frame_sweep`, `fitting.APPLY`, `fitting.ENTRIES`, `fitting.ProbeKind`, `probing.t3_probe_corewise_hessian_from_sweep`, `probing.tv_precompute_probe_frame_sweep`
- `TestDispatch.test_jit_derivative_kind_two_holes` (line 1575): `fitting.ApplyDerivativesKind`, `fitting.EntriesDerivativesKind`, `sampling_derivatives.t3_apply_corewise_derivatives_hessian`, `sampling_derivatives.t3_apply_corewise_derivatives_hessian_from_sweep`, `sampling_derivatives.t3_entries_corewise_derivatives_hessian`, `sampling_derivatives.t3_entries_corewise_derivatives_hessian_from_sweep`, `sampling_derivatives.tv_precompute_apply_frame_sweep_jets`, `sampling_derivatives.tv_precompute_entries_frame_sweep_jets`, `ut3_sampling.ut3_apply_corewise_derivatives_hessian`, `ut3_sampling.ut3_entries_corewise_derivatives_hessian`
- `TestDispatch.test_jit_uniform_exact_ops` (line 2023): `uniform_fitting.UniformProbeDerivativesKind`, `ut3_sampling.ut3_probe_corewise_derivatives_hessian`, `ut3_sampling.ut3_probe_corewise_derivatives_hessian_from_sweep`
- `TestDispatch.test_jit_shared_uniform_exact` (line 2160): `sampling_derivatives.t3_probe_derivatives`
- `TestDispatch.test_jit_ragged_local_model_parameterized_kind` (line 2403): `fitting.SamplingKind`
- `TestJetDtypeUnderX64.test_float32_in_float32_out` (line 3990): `sampling_derivatives.<module>`, `sampling_derivatives.t3_probe_corewise_derivatives_hessian`, `sampling_derivatives.t3_probe_derivatives`, `sampling_derivatives.tv_apply_derivatives`, `sampling_derivatives.tv_apply_derivatives_transpose`, `sampling_derivatives.tv_entries_derivatives_transpose`, `sampling_derivatives.tv_probe_derivatives`, `sampling_derivatives.tv_probe_derivatives_transpose`
- `TestCurvatureJvpOracle.test_every_two_hole_is_the_jvp_of_its_transpose` (line 4214): `sampling_derivatives.t3_apply_corewise_derivatives_hessian`, `sampling_derivatives.t3_apply_corewise_derivatives_transpose`, `sampling_derivatives.t3_entries_corewise_derivatives_hessian`, `sampling_derivatives.t3_entries_corewise_derivatives_transpose`, `sampling_derivatives.t3_probe_corewise_derivatives_hessian`, `sampling_derivatives.t3_probe_corewise_derivatives_transpose`
- `TestCurvatureJvpOracle.test_uniform_probe_derivatives_two_hole_is_the_jvp_of_its_transpose` (line 4253): `ut3_sampling.ut3_probe_corewise_derivatives_hessian`, `ut3_sampling.ut3_probe_corewise_derivatives_transpose`

### `tests/test_jet_recurrence.py`: 5 tests; at import: YES; direct 5; through a helper 0

- module level: 27 sampling_derivatives.<module> (import-module)
- `TestForwardJets.test_forward_jets_match_trs` (line 97): `sampling_derivatives._apply_derivatives_from_jets`, `sampling_derivatives._apply_derivatives_t3_from_xi_jets`, `sampling_derivatives.compute_deta_jets`, `sampling_derivatives.compute_eta_jets`, `sampling_derivatives.compute_mu_jets`, `sampling_derivatives.compute_nu_jets`, `sampling_derivatives.compute_sigma_jets`, `sampling_derivatives.compute_tau_jets`; through tests/test_jet_recurrence.py:_problem
- `TestTransposeJets.test_transpose_jets_match_trs` (line 124): `sampling_derivatives._apply_derivatives_transpose_from_jets`, `sampling_derivatives.assemble_tt_variation_jets`, `sampling_derivatives.compute_dxi_tilde_jets`, `sampling_derivatives.compute_sigma_hat_jets`, `sampling_derivatives.compute_sigma_tilde_jets`, `sampling_derivatives.compute_tau_tilde_jets`; through tests/test_jet_recurrence.py:_problem
- `TestUniformKernelsMatchRagged.test_uniform_equals_ragged` (line 163): `sampling_derivatives.assemble_tt_variation_jets`, `sampling_derivatives.compute_deta_jets`, `sampling_derivatives.compute_dxi_tilde_jets`, `sampling_derivatives.compute_eta_jets`, `sampling_derivatives.compute_tau_tilde_jets`
- `TestAssemblyChunked.test_chunked_matches_dense` (line 201): `sampling_derivatives.assemble_tt_variation_jets`
- `TestAssemblyChunked.test_tucker_chunked_matches_dense` (line 221): `sampling_derivatives.assemble_tucker_variation_jets`

### `tests/test_jets.py`: 39 tests; at import: no; direct 2; through a helper 0

- `TestCheckers.test_the_validators_of_pp_are_gone` (line 895): `sampling_derivatives.check_perturbation_index`, `sampling_derivatives.check_perturbation_vectors`
- `TestFrontendReach.test_the_names_of_t3toolbox_fitting` (line 911): `sampling_derivatives.check_perturbation_index`, `sampling_derivatives.check_perturbation_vectors`

### `tests/test_least_squares_oracle.py`: 18 tests; at import: no; direct 1; through a helper 0

- `TestTheBridgeToTheOldKinds.test_against_the_old_kinds` (line 542): `fitting.ApplyDerivativesKind`, `fitting.EntriesDerivativesKind`, `fitting.ProbeDerivativesKind`

### `tests/test_probe_derivatives.py`: 17 tests; at import: YES; direct 16; through a helper 0

- module level: 14 sampling_derivatives.<module> (import-module)
- `TestProbeDerivatives.test_probe_derivatives_match_dense` (line 57): `sampling_derivatives.dense_probe_derivatives`, `sampling_derivatives.t3_probe_derivatives`
- `TestProbeDerivatives.test_frame_core_stack` (line 92): `sampling_derivatives.dense_probe_derivatives`, `sampling_derivatives.t3_probe_derivatives`
- `TestProbeDerivatives.test_tangent_derivatives_match_dense` (line 120): `sampling_derivatives.dense_probe_derivatives`, `sampling_derivatives.tv_probe_derivatives`
- `TestProbeDerivatives.test_tangent_transpose_adjoint_identity` (line 150): `sampling_derivatives.tv_probe_derivatives`, `sampling_derivatives.tv_probe_derivatives_transpose`
- `TestProbeDerivatives.test_tangent_order_zero_is_plain_tangent_probe` (line 186): `sampling_derivatives.tv_probe_derivatives`
- `TestProbeDerivatives.test_order_zero_is_plain_probe` (line 200): `sampling_derivatives.t3_probe_derivatives`
- `TestProbeDerivatives.test_first_derivative_matches_finite_difference` (line 212): `sampling_derivatives.t3_probe_derivatives`
- `TestProbeDerivatives.test_tangent_derivatives_K_stacked` (line 235): `sampling_derivatives.dense_probe_derivatives`, `sampling_derivatives.tv_probe_derivatives`
- `TestProbeDerivatives.test_tangent_transpose_K_stacked` (line 264): `sampling_derivatives.tv_probe_derivatives`, `sampling_derivatives.tv_probe_derivatives_transpose`
- `TestProbeDerivatives.test_apply_derivatives_match_dense` (line 299): `sampling_derivatives.dense_apply_derivatives`, `sampling_derivatives.t3_apply_derivatives`, `sampling_derivatives.tv_apply_derivatives`
- `TestProbeDerivatives.test_entries_derivatives_match_dense` (line 329): `sampling_derivatives.dense_entries_derivatives`, `sampling_derivatives.t3_entries_derivatives`, `sampling_derivatives.tv_entries_derivatives`
- `TestProbeDerivatives.test_apply_entries_transpose_adjoint_identity` (line 360): `sampling_derivatives.tv_apply_derivatives`, `sampling_derivatives.tv_apply_derivatives_transpose`, `sampling_derivatives.tv_entries_derivatives`, `sampling_derivatives.tv_entries_derivatives_transpose`
- `TestProbeDerivatives.test_corewise_derivatives_finite_difference` (line 401): `sampling_derivatives.t3_apply_corewise_derivatives_transpose`, `sampling_derivatives.t3_apply_derivatives`, `sampling_derivatives.t3_entries_corewise_derivatives_transpose`, `sampling_derivatives.t3_entries_derivatives`, `sampling_derivatives.t3_probe_corewise_derivatives_transpose`, `sampling_derivatives.t3_probe_derivatives`
- `TestProbeDerivatives.test_high_order_vanishes` (line 448): `sampling_derivatives.t3_probe_derivatives`
- `TestAsymmetricGenericFrame.test_forward_jets_match_dense_oracle` (line 506): `sampling_derivatives.dense_probe_derivatives`, `sampling_derivatives.tv_probe_derivatives`
- `TestAsymmetricGenericFrame.test_adjoint_identities` (line 527): `sampling_derivatives.tv_apply_derivatives`, `sampling_derivatives.tv_apply_derivatives_transpose`, `sampling_derivatives.tv_entries_derivatives`, `sampling_derivatives.tv_entries_derivatives_transpose`, `sampling_derivatives.tv_probe_derivatives`, `sampling_derivatives.tv_probe_derivatives_transpose`

### `tests/test_sampling_definitions.py`: 72 tests; at import: no; direct 2; through a helper 0

- `TestAgainstTheOldKernels.test_z_and_g_of_the_probe_kind` (line 1345): `sampling_derivatives.<module>`, `sampling_derivatives.t3_probe_derivatives`, `sampling_derivatives.tv_probe_derivatives_transpose`
- `TestDenseFunctions.test_the_jets_of_entries_off_the_grid` (line 1988): `sampling_derivatives.<module>`, `sampling_derivatives.dense_entries_derivatives`

### `tests/test_tucker_tensor_train.py`: 81 tests; at import: YES; direct 2; through a helper 0

- module level: 13 sampling_derivatives.<module> (import-module)
- `TestTuckerTensorTrain.test_derivative_methods` (line 3013): `sampling_derivatives.t3_probe_derivatives`
- `TestTuckerTensorTrain.test_corewise_derivative_transpose_nontrivial_boundary` (line 3054): `sampling_derivatives.t3_apply_derivatives`, `sampling_derivatives.t3_entries_derivatives`

### `tests/test_uniform_probing.py`: 35 tests; at import: YES; direct 0; through a helper 2

- module level: 24 sampling_derivatives.<module> (import-module)
- `TestTransposeOutputHygiene.test_tangent_transposes` (line 854): -; through tests/test_uniform_probing.py:TestTransposeOutputHygiene._tangent_wrappers
- `TestTransposeOutputHygiene.test_corewise_transposes` (line 935): -; through tests/test_uniform_probing.py:TestTransposeOutputHygiene._corewise_wrappers

### `tests/backend/test_curvature.py`: 96 tests; at import: YES; direct 14; through a helper 15

- module level: 100 sampling_derivatives.<module> (import-module)
- `TestProbeDerivativesTwoHole.test_against_the_block_sum_oracle` (line 680): `sampling_derivatives.t3_probe_corewise_derivatives_hessian`; through tests/backend/test_curvature.py:_pd_blocksum
- `TestProbeDerivativesTwoHole.test_against_a_finite_difference_of_the_one_hole` (line 702): `sampling_derivatives.t3_probe_corewise_derivatives_hessian`, `sampling_derivatives.t3_probe_corewise_derivatives_transpose`
- `TestProbeDerivativesTwoHole.test_production_equals_the_trs_twins` (line 716): `sampling_derivatives.assemble_tt_tangent_variation_jets`, `sampling_derivatives.assemble_tucker_tangent_variation_jets`, `sampling_derivatives.compute_deta_tilde_tangent_jets`, `sampling_derivatives.compute_dxi_tilde_tangent_jets`, `sampling_derivatives.compute_sigma_tilde_tangent_jets`, `sampling_derivatives.compute_tau_tilde_tangent_jets`, `sampling_derivatives.t3_probe_corewise_derivatives_hessian`; through tests/backend/_curvature_oracles.py:probe_derivatives_two_hole_states_trs
- `TestProbeDerivativesTwoHole.test_from_sweep_reuses_the_model_sweep` (line 772): `sampling_derivatives.t3_probe_corewise_derivatives_hessian`, `sampling_derivatives.t3_probe_corewise_derivatives_hessian_from_sweep`, `sampling_derivatives.tv_precompute_apply_frame_sweep_jets`, `sampling_derivatives.tv_precompute_probe_frame_sweep_jets`
- `TestProbeDerivativesTwoHole.test_stacked_equals_per_element` (line 791): `sampling_derivatives.t3_probe_corewise_derivatives_hessian`
- `TestProbeDerivativesTwoHole.test_uniform_supercores_and_chunking` (line 806): `sampling_derivatives.t3_probe_corewise_derivatives_hessian`
- `TestProbeDerivativesTwoHole.test_symmetry` (line 833): `sampling_derivatives.t3_probe_corewise_derivatives_hessian`
- `TestProbeDerivativesTwoHole.test_residual_weights` (line 851): `fitting.ProbeDerivativesKind`, `sampling_derivatives.t3_probe_corewise_derivatives_hessian`, `sampling_derivatives.t3_probe_corewise_derivatives_transpose`, `sampling_derivatives.t3_probe_derivatives`
- `TestPlainKindsTwoHole.test_against_the_block_sum_oracle` (line 954): -; through tests/backend/test_curvature.py:_plain_ops
- `TestPlainKindsTwoHole.test_against_a_finite_difference_of_the_one_hole` (line 977): -; through tests/backend/test_curvature.py:_plain_ops
- `TestPlainKindsTwoHole.test_probe_pieces_against_finite_differences_of_the_gn_kernels` (line 999): `probing.assemble_tt_tangent_variations`, `probing.compute_deta`, `probing.compute_deta_tilde`, `probing.compute_deta_tilde_tangent`, `probing.compute_dxi`, `probing.compute_dxi_tilde`, `probing.compute_dxi_tilde_tangent`, `probing.compute_eta`, `probing.compute_mu`, `probing.compute_nu`, `probing.compute_sigma`, `probing.compute_sigma_tilde`, `probing.compute_sigma_tilde_tangent`, `probing.compute_tau`, `probing.compute_tau_tilde`, `probing.compute_tau_tilde_tangent`, `probing.compute_xi`
- `TestPlainKindsTwoHole.test_apply_tt_assembly_is_the_three_term_product_rule` (line 1060): `probing.compute_dxi`, `probing.compute_mu`, `probing.compute_sigma`, `probing.compute_sigma_hat`, `probing.compute_tau`, `probing.compute_xi`
- `TestPlainKindsTwoHole.test_from_sweep_reuses_the_model_sweep` (line 1078): -; through tests/backend/test_curvature.py:_plain_ops
- `TestPlainKindsTwoHole.test_stacked_equals_per_element` (line 1097): -; through tests/backend/test_curvature.py:_plain_ops
- `TestPlainKindsTwoHole.test_uniform_supercores` (line 1112): -; through tests/backend/test_curvature.py:_plain_ops
- `TestPlainKindsTwoHole.test_symmetry` (line 1138): -; through tests/backend/test_curvature.py:_plain_ops
- `TestPlainKindsTwoHole.test_residual_weights` (line 1157): `fitting.APPLY`, `fitting.ENTRIES`, `fitting.ProbeKind`; through tests/backend/test_curvature.py:_plain_ops
- `TestDerivativeKindsTwoHole.test_against_the_block_sum_oracle` (line 1256): -; through tests/backend/test_curvature.py:_deriv_ops
- `TestDerivativeKindsTwoHole.test_against_a_finite_difference_of_the_one_hole` (line 1282): -; through tests/backend/test_curvature.py:_deriv_ops
- `TestDerivativeKindsTwoHole.test_production_equals_the_trs_twins` (line 1305): `sampling_derivatives._apply_tangent_assemble_step`, `sampling_derivatives._dxi_hat_tangent_jets_step_ragged`, `sampling_derivatives.compute_sigma_hat_jets`, `sampling_derivatives.compute_sigma_hat_tangent_jets`; through tests/backend/_curvature_oracles.py:apply_derivatives_two_hole_states_trs, tests/backend/test_curvature.py:_deriv_ops
- `TestDerivativeKindsTwoHole.test_order_zero_is_the_plain_two_hole` (line 1358): -; through tests/backend/test_curvature.py:_deriv_ops
- `TestDerivativeKindsTwoHole.test_containment` (line 1373): `sampling_derivatives.t3_apply_corewise_derivatives_hessian`, `sampling_derivatives.t3_entries_corewise_derivatives_hessian`, `sampling_derivatives.t3_probe_corewise_derivatives_hessian`
- `TestDerivativeKindsTwoHole.test_from_sweep_reuses_the_model_sweep` (line 1405): -; through tests/backend/test_curvature.py:_deriv_ops
- `TestDerivativeKindsTwoHole.test_stacked_equals_per_element` (line 1425): -; through tests/backend/test_curvature.py:_deriv_ops
- `TestDerivativeKindsTwoHole.test_uniform_supercores` (line 1441): -; through tests/backend/test_curvature.py:_deriv_ops
- `TestDerivativeKindsTwoHole.test_symmetry` (line 1466): -; through tests/backend/test_curvature.py:_deriv_ops
- `TestDerivativeKindsTwoHole.test_residual_weights` (line 1486): `fitting.ApplyDerivativesKind`, `fitting.EntriesDerivativesKind`; through tests/backend/test_curvature.py:_deriv_ops
- `TestPlainKindsExactLocalModel.test_corewise_is_the_plain_hessian` (line 2164): -; through tests/backend/test_curvature.py:_plain_ops
- `TestPlainKindsExactLocalModel.test_the_curvature_block_is_the_oracles_Kp_at_the_weighted_residual` (line 2216): -; through tests/backend/test_curvature.py:_plain_ops

### `tests/backend/test_from_sweep_reuse.py`: 5 tests; at import: YES; direct 3; through a helper 0

- module level: 19 sampling_derivatives.<module> (import-module)
- `TestRaggedFromSweepReuse.test_probe` (line 47): `probing.tv_precompute_probe_frame_sweep`, `probing.tv_probe_jacobian_from_sweep`, `probing.tv_probe_transpose_from_sweep`, `sampling_derivatives.tv_precompute_probe_frame_sweep_jets`, `sampling_derivatives.tv_probe_derivatives`, `sampling_derivatives.tv_probe_derivatives_transpose`, `sampling_derivatives.tv_probe_jacobian_derivatives_from_sweep`, `sampling_derivatives.tv_probe_transpose_derivatives_from_sweep`
- `TestRaggedFromSweepReuse.test_apply` (line 76): `apply.tv_apply_jacobian_from_sweep`, `apply.tv_apply_transpose_from_sweep`, `apply.tv_precompute_apply_frame_sweep`, `sampling_derivatives.tv_apply_derivatives`, `sampling_derivatives.tv_apply_derivatives_transpose`, `sampling_derivatives.tv_apply_jacobian_derivatives_from_sweep`, `sampling_derivatives.tv_apply_transpose_derivatives_from_sweep`, `sampling_derivatives.tv_precompute_apply_frame_sweep_jets`
- `TestRaggedFromSweepReuse.test_entries` (line 102): `entries.tv_entries_jacobian_from_sweep`, `entries.tv_entries_transpose_from_sweep`, `entries.tv_precompute_entries_frame_sweep`, `sampling_derivatives.tv_entries_derivatives`, `sampling_derivatives.tv_entries_derivatives_transpose`, `sampling_derivatives.tv_entries_jacobian_derivatives_from_sweep`, `sampling_derivatives.tv_entries_transpose_derivatives_from_sweep`, `sampling_derivatives.tv_precompute_entries_frame_sweep_jets`

### `tests/backend/test_uniform_curvature.py`: 56 tests; at import: YES; direct 7; through a helper 8

- module level: 53 sampling_derivatives.<module> (import-module)
- `TestUniformProbeDerivativesTwoHole.test_per_element_vs_ragged` (line 491): `sampling_derivatives.t3_probe_corewise_derivatives_hessian`, `ut3_sampling.ut3_probe_corewise_derivatives_hessian`, `ut3_sampling.ut3_probe_corewise_derivatives_hessian_from_sweep`, `utv_sampling.utv_precompute_apply_frame_sweep_jets`
- `TestUniformProbeDerivativesTwoHole.test_the_direction_mask_is_load_bearing` (line 526): `sampling_derivatives.t3_probe_corewise_derivatives_hessian`, `ut3_sampling.ut3_probe_corewise_derivatives_hessian`
- `TestUniformKindCorewiseHessian.test_the_override_is_its_own` (line 566): `fitting.ProbeDerivativesKind`, `uniform_fitting.UniformProbeDerivativesKind`, `uniform_fitting._UniformLayerMixin`
- `TestUniformKindCorewiseHessian.test_masks_from_the_frame_at_tt_slack` (line 573): `fitting.ProbeDerivativesKind`, `uniform_fitting.UniformProbeDerivativesKind`, `ut3_sampling.ut3_probe_corewise_derivatives_hessian_from_sweep`
- `TestUniformKindCorewiseHessian.test_theta_hat_masking_unit` (line 603): `uniform_fitting.UniformProbeDerivativesKind`
- `TestUniformPlainKindsTwoHole.test_per_element_vs_ragged` (line 673): -; through tests/backend/test_uniform_curvature.py:_PLAIN_UNIFORM
- `TestUniformPlainKindsTwoHole.test_the_direction_mask_is_load_bearing` (line 705): -; through tests/backend/test_uniform_curvature.py:_PLAIN_UNIFORM
- `TestUniformPlainKindCorewiseHessian.test_the_override_is_its_own` (line 749): `uniform_fitting._UniformLayerMixin`; through tests/backend/test_uniform_curvature.py:TestUniformPlainKindCorewiseHessian._CLASSES
- `TestUniformPlainKindCorewiseHessian.test_masks_from_the_frame_at_tt_slack` (line 758): -; through tests/backend/test_uniform_curvature.py:TestUniformPlainKindCorewiseHessian._RKIND, tests/backend/test_uniform_curvature.py:TestUniformPlainKindCorewiseHessian._UKIND, tests/backend/test_uniform_curvature.py:_PLAIN_UNIFORM
- `TestUniformPlainKindCorewiseHessian.test_theta_hat_masking_unit` (line 784): -; through tests/backend/test_uniform_curvature.py:TestUniformPlainKindCorewiseHessian._UKIND
- `TestUniformDerivativeKindsTwoHole.test_per_element_vs_ragged` (line 850): -; through tests/backend/test_uniform_curvature.py:_deriv_uniform
- `TestUniformDerivativeKindsTwoHole.test_the_direction_mask_is_load_bearing` (line 878): -; through tests/backend/test_uniform_curvature.py:_deriv_uniform
- `TestUniformDerivativeKindCorewiseHessian.test_the_override_is_its_own` (line 911): `uniform_fitting._UniformLayerMixin`; through tests/backend/test_uniform_curvature.py:TestUniformDerivativeKindCorewiseHessian._RCLS, tests/backend/test_uniform_curvature.py:TestUniformDerivativeKindCorewiseHessian._UCLS
- `TestUniformDerivativeKindCorewiseHessian.test_masks_from_the_frame_at_tt_slack` (line 921): -; through tests/backend/test_uniform_curvature.py:TestUniformDerivativeKindCorewiseHessian._RCLS, tests/backend/test_uniform_curvature.py:TestUniformDerivativeKindCorewiseHessian._UCLS, tests/backend/test_uniform_curvature.py:_deriv_uniform
- `TestUniformDerivativeKindCorewiseHessian.test_theta_hat_masking_unit` (line 950): -; through tests/backend/test_uniform_curvature.py:TestUniformDerivativeKindCorewiseHessian._UCLS

### `tests/backend/test_uniform_fitting.py`: 30 tests; at import: no; direct 6; through a helper 5

- `TestUniformSamplingKind.test_forward_matches_ragged` (line 122): `uniform_fitting.uniform_sampling_kind`; through tests/backend/test_uniform_fitting.py:TestUniformDerivativeSamplingKind._KINDS, tests/backend/test_uniform_fitting.py:TestUniformSamplingKind._KINDS
- `TestUniformSamplingKind.test_point_forward_matches_ragged` (line 137): `uniform_fitting.uniform_sampling_kind`; through tests/backend/test_uniform_fitting.py:TestUniformDerivativeSamplingKind._KINDS, tests/backend/test_uniform_fitting.py:TestUniformSamplingKind._KINDS
- `TestUniformSamplingKind.test_adjoint_identity` (line 150): `uniform_fitting.uniform_sampling_kind`; through tests/backend/test_uniform_fitting.py:TestUniformDerivativeSamplingKind._KINDS, tests/backend/test_uniform_fitting.py:TestUniformSamplingKind._KINDS
- `TestUniformSamplingKind.test_forward_garbage_robust` (line 165): `uniform_fitting.uniform_sampling_kind`; through tests/backend/test_uniform_fitting.py:TestUniformDerivativeSamplingKind._KINDS, tests/backend/test_uniform_fitting.py:TestUniformSamplingKind._KINDS
- `TestUniformBlockSumsq.test_probe_block_sumsq_matches_ragged` (line 188): `fitting.PROBE`, `fitting.probe_derivatives_kind`, `uniform_fitting.UniformApplyKind`, `uniform_fitting.UniformProbeDerivativesKind`, `uniform_fitting.UniformProbeKind`
- `TestUniformDerivativeSamplingKind.test_forward_matches_ragged` (line 283): -; through tests/backend/test_uniform_fitting.py:TestUniformDerivativeSamplingKind._KINDS, tests/backend/test_uniform_fitting.py:TestUniformDerivativeSamplingKind._kinds, tests/backend/test_uniform_fitting.py:TestUniformSamplingKind._KINDS
- `TestUniformDerivativeSamplingKind.test_point_forward_matches_ragged` (line 298): -; through tests/backend/test_uniform_fitting.py:TestUniformDerivativeSamplingKind._KINDS, tests/backend/test_uniform_fitting.py:TestUniformDerivativeSamplingKind._kinds, tests/backend/test_uniform_fitting.py:TestUniformSamplingKind._KINDS
- `TestUniformDerivativeSamplingKind.test_adjoint_identity_weighted` (line 311): -; through tests/backend/test_uniform_fitting.py:TestUniformDerivativeSamplingKind._KINDS, tests/backend/test_uniform_fitting.py:TestUniformDerivativeSamplingKind._kinds, tests/backend/test_uniform_fitting.py:TestUniformSamplingKind._KINDS
- `TestUniformDerivativeSamplingKind.test_forward_garbage_robust` (line 331): -; through tests/backend/test_uniform_fitting.py:TestUniformDerivativeSamplingKind._KINDS, tests/backend/test_uniform_fitting.py:TestUniformDerivativeSamplingKind._kinds, tests/backend/test_uniform_fitting.py:TestUniformSamplingKind._KINDS
- `TestKindTransposeOutputHygiene.test_kind_transposes` (line 449): -; through tests/backend/test_uniform_fitting.py:TestKindTransposeOutputHygiene._SEAMS, tests/backend/test_uniform_fitting.py:TestKindTransposeOutputHygiene._kind_and_sample
- `TestSplitSeamDefaultParity.test_sum_over_probes_defaults_match_ragged` (line 484): `apply.tv_apply_transpose_from_sweep`, `entries.tv_entries_transpose_from_sweep`, `probing.tv_probe_transpose_from_sweep`, `sampling_derivatives.<module>`, `sampling_derivatives.tv_apply_transpose_derivatives_from_sweep`, `sampling_derivatives.tv_entries_transpose_derivatives_from_sweep`, `sampling_derivatives.tv_probe_transpose_derivatives_from_sweep`, `utv_sampling.utv_apply_transpose_derivatives_from_sweep`, `utv_sampling.utv_apply_transpose_from_sweep`, `utv_sampling.utv_entries_transpose_derivatives_from_sweep`, `utv_sampling.utv_entries_transpose_from_sweep`, `utv_sampling.utv_probe_transpose_derivatives_from_sweep`, `utv_sampling.utv_probe_transpose_from_sweep`

### `tests/bridge/test_old_kernels_bridge.py`: 9 tests; at import: YES; direct 9; through a helper 0

- module level: 71 sampling_derivatives.<module> (import-module)
- `TestOldKernels.test_z_and_the_plain_variables` (line 143): `sampling_derivatives.compute_eta_jets`, `sampling_derivatives.compute_mu_jets`, `sampling_derivatives.compute_nu_jets`, `sampling_derivatives.t3_probe_derivatives`; through tests/bridge/test_old_kernels_bridge.py:old_point_value
- `TestOldKernels.test_g_through_the_forward_seed` (line 164): `fitting.ProbeDerivativesKind`, `fitting.ProbeKind`, `sampling_derivatives.tv_probe_derivatives_transpose`
- `TestOldKernels.test_the_adjoint_variables` (line 187): `sampling_derivatives.compute_deta_tilde_jets`, `sampling_derivatives.compute_dxi_tilde_jets`, `sampling_derivatives.compute_mu_jets`, `sampling_derivatives.compute_nu_jets`, `sampling_derivatives.compute_sigma_tilde_jets`, `sampling_derivatives.compute_tau_tilde_jets`
- `TestTheGuard.test_the_old_side_refuses_the_engine` (line 211): `fitting.ApplyKind`; through tests/bridge/test_old_kernels_bridge.py:old_point_value
- `TestTheDirection.test_forward_and_transpose` (line 230): `fitting.ProbeDerivativesKind`
- `TestTheCurvature.test_Kp_against_the_old_corewise_hessian` (line 262): `fitting.ProbeDerivativesKind`
- `TestTheApplyKind.test_against_the_old_kernels` (line 301): `fitting.ApplyDerivativesKind`, `fitting.ApplyKind`, `sampling_derivatives.t3_apply_derivatives`; through tests/bridge/test_old_kernels_bridge.py:old_point_value
- `TestTheEntriesKind.test_against_the_old_entries_kernels` (line 366): `fitting.EntriesKind`; through tests/bridge/test_old_kernels_bridge.py:old_point_value
- `TestTheEntriesKind.test_the_old_entries_derivatives_against_apply_at_the_one_hot_jet` (line 397): `fitting.EntriesDerivativesKind`, `sampling_derivatives.t3_entries_derivatives`

### Names built at run time in the tests (`getattr` with a computed name; reviewed by hand)

- `tests/backend/test_uniform_curvature.py:727`: `getattr(module, 'tv_precompute_%s_frame_sweep' % kind)` -> `tv_precompute_apply_frame_sweep`, `tv_precompute_entries_frame_sweep`, `tv_precompute_probe_frame_sweep`
- `tests/backend/test_uniform_curvature.py:728`: `getattr(module, 't3_%s_corewise_hessian_from_sweep' % kind)` -> `t3_apply_corewise_hessian_from_sweep`, `t3_entries_corewise_hessian_from_sweep`, `t3_probe_corewise_hessian_from_sweep`
- `tests/test_uniform_probing.py:913`: `getattr(module, 'tv_precompute_%s_frame_sweep' % kind)` -> `tv_precompute_apply_frame_sweep`, `tv_precompute_entries_frame_sweep`, `tv_precompute_probe_frame_sweep`
- `tests/test_uniform_probing.py:914`: `getattr(module, 'tv_%s_transpose_from_sweep' % kind)` -> `tv_apply_transpose_from_sweep`, `tv_entries_transpose_from_sweep`, `tv_probe_transpose_from_sweep`

139 other computed names (a variable, not a kind pattern): read by hand; none reaches a deleted name (the list: `--dynamic`).

### Files of the tests with no tests of their own (helpers, the oracle sweeps)

- `tests/backend/_curvature_oracles.py`: 403 probing.compute_xi (attr); 403 sampling_derivatives.build_input_jets (attr); 409 sampling_derivatives.compute_deta_tilde_jets (attr); 412 probing.compute_xi (attr); 412 sampling_derivatives.build_input_jets (attr); 529 entries._entry_xis (attr); 53 sampling_derivatives.<module> (import-module); 530 entries._onehot_vectors (attr); 532 probing.compute_xi (attr); 535 probing.compute_xi (attr); 535 sampling_derivatives.build_input_jets (attr); 536 probing.compute_xi (attr); 536 sampling_derivatives.build_input_jets (attr)
- `tests/oracle_sweeps/o1_ragged.py`: 184 sampling_derivatives.tv_apply_derivatives (string); 186 sampling_derivatives.tv_entries_derivatives (string); 188 sampling_derivatives.tv_probe_derivatives (string); 90 sampling_derivatives.t3_apply_derivatives (string); 92 sampling_derivatives.t3_entries_derivatives (string); 94 sampling_derivatives.t3_probe_derivatives (string)
- `tests/oracle_sweeps/o1_uniform.py`: 111 utv_sampling.utv_apply_derivatives (string); 112 utv_sampling.utv_entries_derivatives (string); 113 utv_sampling.utv_probe_derivatives (string)

## 4. Examples

- `examples/fit_probe_display.py:68`: `fitting.PROBE`
- `examples/fit_probe_display.py:79`: `fitting.PROBE`

## 5. Pages (`docs/*.md`, `docs/contributor/*.md`, `docs/*.rst`; not `docs/release_notes.md`)

### In doctest examples (run by the gate)

None.

### In prose, fenced code without prompts, and Sphinx roles (P19's, unless a role or a toctree fails `-W`)

- `docs/api_reference.rst`: 1: 130 sampling_derivatives (role)
- `docs/contributor/ambient_derivative_transpose_note.md`: 1: 13 sampling_derivatives (prose)
- `docs/contributor/batching_internals.md`: 2: 66 _apply_transpose_adjoint (prose); 66 tv_entries_transpose_from_sweep (prose)
- `docs/contributor/curvature_internals.md`: 8: 31 SamplingKind (prose); 102 build_input_jets (prose); 102 compute_xi (prose); 102 compute_xi (prose); 112 _adj_tilde_step (prose); 114 _adj_prop_tangent (prose); 122 _wchunked_sum (prose); 349 estimate_chunk_size (prose)
- `docs/contributor/deferred_and_rejected.md`: 4: 122 sampling_derivatives (prose); 257 _wchunked_sum (prose); 258 _wchunked_sum (prose); 260 max_chunk_size_within (prose)
- `docs/contributor/parameters_not_closures.md`: 4: 24 probe_derivatives_kind (prose); 28 SamplingKind (prose); 156 SamplingKind (prose); 187 ApplyKind (prose)
- `docs/contributor/scan_body_principles.md`: 4: 223 _mu_jets_step (prose); 223 compute_mu_jets (prose); 223 sampling_derivatives (prose); 228 _mu_jets_step (fence)
- `docs/contributor/signature_style.md`: 1: 55 compute_mu (fence)
- `docs/contributor/verification_records.md`: 1: 87 estimate_chunk_size (prose)
- `docs/entries_apply_probe.md`: 13: 243 compute_xi (prose); 244 compute_mu (prose); 245 compute_nu (prose); 246 compute_eta (prose); 247 assemble_z (prose); 259 compute_dxi (prose); 260 compute_sigma (prose); 261 compute_tau (prose); 262 compute_deta (prose); 263 assemble_tangent_z (prose); 268 compute_deta_tilde (prose); 269 assemble_tucker_variations (prose); 269 assemble_tt_variations (prose)
- `docs/fitting_and_optimization.md`: 11: 19 SamplingKind (prose); 247 SamplingKind (fence); 249 sampling_derivatives (fence); 253 SamplingKind (prose); 258 ProbeKind (prose); 261 ApplyKind (prose); 261 ProbeKind (prose); 262 ScalarOutputKind (prose); 262 ProbeOutputKind (prose); 264 SamplingKind (prose); 349 SamplingKind (prose)
- `docs/glossary.md`: 7: 404 estimate_chunk_size (prose); 404 max_chunk_size_within (prose); 421 SamplingKind (prose); 461 SamplingKind (prose); 789 sampling_derivatives (prose); 789 _shift_down (prose); 789 _shift_up (prose)
- `docs/naming_conventions.md`: 10: 82 sampling_derivatives (prose); 88 dense_apply_derivatives (prose); 150 compute_mu_jets (prose); 160 SamplingKind (prose); 162 compute_tau_tilde_tangent_jets (prose); 162 compute_tau_tilde_jets (prose); 162 assemble_tt_tangent_variation_jets (prose); 163 assemble_tt_variation_jets (prose); 164 compute_dxi_tilde_tangent (prose); 193 sampling_derivatives (prose)
- `docs/reading_signatures.md`: 1: 11 compute_mu (fence)
- `docs/uniform_equivalence_contract.md`: 1: 118 sumsq_over_probes (prose)
