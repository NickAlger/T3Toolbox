# K measurements

Static measurements for the packing plan. Tree: branch `graded-messages`, HEAD `3375d213` (working tree has only the untracked/modified dev files from the session start; nothing under `t3toolbox/ tests/ docs/ examples/` was edited).
Scripts (re-runnable, `ast` + regex, no library import, no tests run): `/tmp/claude-1000/-home-nick-repos-T3Toolbox/7a03b046-9177-4c43-923a-b31595edb27e/scratchpad/kmeasure/`.
Areas: `library` = `t3toolbox/**/*.py` non-doctest code; `library doctests` = lines starting `>>>`/`...` inside library docstrings; `tests` = `tests/**/*.py`; `docs` = `docs/*.md`, `docs/contributor/*.md`, `docs/*.rst`; `examples` = `examples/**/*.py`.
Line numbers are those of HEAD (working tree identical for these paths).

## M1. What a deletion of the old sampling kernels would touch

Method: `ast` for code references (Name, Attribute attr, ImportFrom alias, keyword, parameter), regex word-boundary on lines for
doctest lines (`>>>`/`...` inside docstrings), docstring prose, comments, and non-docstring string constants; regex on `docs/*.md`,
`docs/contributor/*.md`, `docs/*.rst`. Scripts: `/tmp/claude-1000/-home-nick-repos-T3Toolbox/7a03b046-9177-4c43-923a-b31595edb27e/scratchpad/kmeasure/{scan,m1}.py`.
"Enclosing function" is the innermost def/class (qualified); "top" is the outermost def/class (the unit that a deletion removes whole).
State of the tree: HEAD 3375d213. NOTE the plan text (sections 2.4, 10) was written before P16; the tree is now after P16: `backend/optimizers.py`
no longer has `Problem`/`least_squares_problem`; `uniform_least_squares_problem`, `_ScaledRegularizer`, `_check_ragged_sample` do not exist (see table in (a)).

### (a) Names to be deleted, per module (how determined)

Determination: the six sampling modules by the section 2.4 rule applied to today's `__all__` (ast): probing = every `compute_*`, `assemble_*`,
`tv_precompute_probe_frame_sweep`, every `*_from_sweep`; apply/entries = `tv_precompute_*_frame_sweep` and `*_from_sweep`; ut3_sampling = every name containing
`derivatives` or ending `_from_sweep`; utv_sampling = everything except the six `utv_{probe,apply,entries}[_transpose]`; sampling_derivatives = whole module
(all top-level names, public `__all__` plus private). `backend/fitting.py` / `uniform_fitting.py` lists are from the task text + class hierarchy by `ast` (private
hierarchy helpers `_WwSample` etc. included because they are the bases of the old kinds). Today's `__all__` sizes: probing 35, apply 12, entries 12, ut3_sampling 28, utv_sampling 30
(plan table said 34/10/10/25/30: new names added by P-slices since).

- `t3toolbox/backend/probing.py`: 27 names
- `t3toolbox/backend/apply.py`: 4 names
- `t3toolbox/backend/entries.py`: 4 names
- `t3toolbox/backend/ut3_sampling.py`: 15 names
- `t3toolbox/backend/utv_sampling.py`: 24 names
- `t3toolbox/backend/fitting.py`: 33 names
- `t3toolbox/backend/uniform_fitting.py`: 17 names
- `t3toolbox/backend/sampling_derivatives.py`: whole module; 154 top-level names (57 public in `__all__`), 3534 lines

**`t3toolbox/backend/probing.py`** (27): `compute_xi`, `compute_mu`, `compute_nu`, `compute_eta`, `assemble_z`, `compute_dxi`, `compute_sigma`, `compute_tau`, `compute_deta`, `assemble_tangent_z`, `compute_sigma_hat`, `tv_precompute_probe_frame_sweep`, `tv_probe_jacobian_from_sweep`, `tv_probe_transpose_from_sweep`, `compute_deta_tilde`, `compute_tau_tilde`, `compute_sigma_tilde`, `compute_dxi_tilde`, `assemble_tucker_variations`, `assemble_tt_variations`, `compute_deta_tilde_tangent`, `compute_tau_tilde_tangent`, `compute_sigma_tilde_tangent`, `compute_dxi_tilde_tangent`, `assemble_tucker_tangent_variations`, `assemble_tt_tangent_variations`, `t3_probe_corewise_hessian_from_sweep`

**`t3toolbox/backend/apply.py`** (4): `tv_precompute_apply_frame_sweep`, `tv_apply_jacobian_from_sweep`, `tv_apply_transpose_from_sweep`, `t3_apply_corewise_hessian_from_sweep`

**`t3toolbox/backend/entries.py`** (4): `tv_precompute_entries_frame_sweep`, `tv_entries_jacobian_from_sweep`, `tv_entries_transpose_from_sweep`, `t3_entries_corewise_hessian_from_sweep`

**`t3toolbox/backend/ut3_sampling.py`** (15): `ut3_probe_derivatives`, `ut3_apply_derivatives`, `ut3_entries_derivatives`, `ut3_apply_corewise_derivatives_transpose`, `ut3_entries_corewise_derivatives_transpose`, `ut3_probe_corewise_derivatives_transpose`, `ut3_probe_corewise_derivatives_hessian`, `ut3_probe_corewise_derivatives_hessian_from_sweep`, `ut3_probe_corewise_hessian_from_sweep`, `ut3_apply_corewise_hessian_from_sweep`, `ut3_entries_corewise_hessian_from_sweep`, `ut3_apply_corewise_derivatives_hessian`, `ut3_apply_corewise_derivatives_hessian_from_sweep`, `ut3_entries_corewise_derivatives_hessian`, `ut3_entries_corewise_derivatives_hessian_from_sweep`

**`t3toolbox/backend/utv_sampling.py`** (24): `utv_precompute_apply_frame_sweep`, `utv_precompute_probe_frame_sweep`, `utv_precompute_entries_frame_sweep`, `utv_apply_jacobian_from_sweep`, `utv_probe_jacobian_from_sweep`, `utv_entries_jacobian_from_sweep`, `utv_apply_transpose_from_sweep`, `utv_probe_transpose_from_sweep`, `utv_entries_transpose_from_sweep`, `utv_precompute_apply_frame_sweep_jets`, `utv_precompute_probe_frame_sweep_jets`, `utv_precompute_entries_frame_sweep_jets`, `utv_apply_jacobian_derivatives_from_sweep`, `utv_probe_jacobian_derivatives_from_sweep`, `utv_entries_jacobian_derivatives_from_sweep`, `utv_apply_transpose_derivatives_from_sweep`, `utv_probe_transpose_derivatives_from_sweep`, `utv_entries_transpose_derivatives_from_sweep`, `utv_probe_derivatives`, `utv_apply_derivatives`, `utv_entries_derivatives`, `utv_probe_derivatives_transpose`, `utv_apply_derivatives_transpose`, `utv_entries_derivatives_transpose`

**`t3toolbox/backend/fitting.py`** (33): `SamplingKind`, `ScalarOutputKind`, `ProbeOutputKind`, `ApplyKind`, `EntriesKind`, `ProbeKind`, `ApplyDerivativesKind`, `EntriesDerivativesKind`, `ProbeDerivativesKind`, `APPLY`, `ENTRIES`, `PROBE`, `probe_kind`, `apply_derivatives_kind`, `entries_derivatives_kind`, `probe_derivatives_kind`, `sumsq_over_samples`, `sumsq_over_probes`, `block_sumsq_over_samples`, `block_sumsq_over_probes`, `_WwSample`, `_IndexSample`, `_WwPpSample`, `_IndexPpSample`, `_DerivativesKind`, `_take_apply`, `_take_probe`, `_take_entries`, `_take_deriv_apply`, `_take_deriv_entries`, `_take_deriv_probe`, `_weight_matrix`, `_make_weight`

**`t3toolbox/backend/uniform_fitting.py`** (17): `UniformApplyKind`, `UniformEntriesKind`, `UniformProbeKind`, `UniformApplyDerivativesKind`, `UniformEntriesDerivativesKind`, `UniformProbeDerivativesKind`, `uniform_sampling_kind`, `uniform_derivatives_kind`, `_UniformLayerMixin`, `_SAMPLING_KIND`, `_DERIV_SAMPLING_KIND`, `_ptake_apply`, `_ptake_probe`, `_ptake_entries`, `_ptake_deriv_apply`, `_ptake_deriv_entries`, `_ptake_deriv_probe`

Sampling-module removed names (excluding sampling_derivatives): 74 (probing+apply+entries+ut3+utv)

**`sampling_derivatives.__all__` (57):** `check_perturbation_vectors`, `check_perturbation_index`, `t3_probe_derivatives`, `build_input_jets`, `compute_mu_jets`, `compute_nu_jets`, `compute_eta_jets`, `assemble_z_jets`, `tv_precompute_apply_frame_sweep_jets`, `tv_precompute_entries_frame_sweep_jets`, `tv_precompute_probe_frame_sweep_jets`, `tv_probe_derivatives`, `tv_probe_jacobian_derivatives_from_sweep`, `compute_sigma_jets`, `compute_tau_jets`, `compute_deta_jets`, `assemble_tangent_z_jets`, `t3_apply_derivatives`, `tv_apply_derivatives`, `tv_apply_jacobian_derivatives_from_sweep`, `t3_entries_derivatives`, `tv_entries_derivatives`, `tv_entries_jacobian_derivatives_from_sweep`, `tv_probe_derivatives_transpose`, `tv_probe_transpose_derivatives_from_sweep`, `estimate_chunk_size`, `max_chunk_size_within`, `compute_deta_tilde_jets`, `compute_tau_tilde_jets`, `compute_sigma_tilde_jets`, `compute_dxi_tilde_jets`, `assemble_tucker_variation_jets`, `assemble_tt_variation_jets`, `tv_apply_derivatives_transpose`, `tv_apply_transpose_derivatives_from_sweep`, `tv_entries_derivatives_transpose`, `tv_entries_transpose_derivatives_from_sweep`, `compute_sigma_hat_jets`, `t3_probe_corewise_derivatives_transpose`, `t3_apply_corewise_derivatives_transpose`, `t3_entries_corewise_derivatives_transpose`, `t3_probe_corewise_derivatives_hessian`, `t3_probe_corewise_derivatives_hessian_from_sweep`, `compute_deta_tilde_tangent_jets`, `compute_tau_tilde_tangent_jets`, `compute_sigma_tilde_tangent_jets`, `compute_dxi_tilde_tangent_jets`, `assemble_tucker_tangent_variation_jets`, `assemble_tt_tangent_variation_jets`, `t3_apply_corewise_derivatives_hessian`, `t3_apply_corewise_derivatives_hessian_from_sweep`, `t3_entries_corewise_derivatives_hessian`, `t3_entries_corewise_derivatives_hessian_from_sweep`, `compute_sigma_hat_tangent_jets`, `dense_probe_derivatives`, `dense_apply_derivatives`, `dense_entries_derivatives`

**Private top-level names of `sampling_derivatives` (97):** `DEFAULT_CHUNK_SIZE`, `_init_jet`, `_shift_down`, `_shift_up`, `_shift_down_pair`, `_shift_up_pair`, `_inverse_factorials`, `_mu_jets_step`, `_eta_jets_order_step`, `_eta_jets_core`, `_eta_jets_step_uniform`, `_eta_jets_step_ragged`, `_z_jets_step`, `_apply_frame_sweep_jets_from_xi`, `_probe_frame_sweep_jets_from_xi`, `_zero_jet`, `_sigma_banded_step`, `_sigma_jets_step`, `_deta_jets_order_step`, `_deta_jets_core`, `_deta_jets_step_uniform`, `_deta_jets_step_ragged`, `_tangent_z_jets_step`, `_apply_derivatives_t3_step`, `_apply_derivatives_t3_from_xi_jets`, `_apply_derivatives_tv_step`, `_apply_derivatives_from_jets`, `_deta_tilde_jets_step`, `_adj_prop`, `_adj_src_order_step`, `_adj_tilde_step`, `_adj_sweep_scanned_step_uniform`, `_adj_sweep_scanned_step_ragged`, `_adj_sweep_scanned`, `_dxi_tilde_order_step`, `_dxi_tilde_jets_core`, `_dxi_tilde_jets_step_uniform`, `_dxi_tilde_jets_step_ragged`, `_tucker_variation_jets_core`, `_tucker_variation_jets_step_summed`, `_tucker_variation_jets_step_unsummed`, `_assemble_tucker_variation_jets_dense`, `_tt_det_lag_step`, `_tt_variation_jets_step`, `_assemble_tt_variation_jets_dense`, `_slice_w`, `_tucker_chunk_summed`, `_tucker_chunk_kept`, `_tt_chunk_summed`, `_tt_chunk_kept`, `_wchunked_sum_step`, `_flatten_w`, `_wchunked_sum`, `_wchunked_concat`, `_HESSIAN_MODES`, `_check_hessian_mode`, `_jet_floor_bytes`, `_SLOPE_CHUNKS`, `_chunk_slope`, `_assembly_per_row_bytes`, `_two_hole_jet_floor_bytes`, `_two_hole_assembly_per_row_bytes`, `_sigma_hat_jets_step`, `_apply_transpose_assemble_core`, `_apply_transpose_assemble_step_summed`, `_apply_transpose_assemble_step_unsummed`, `_dxi_hat_order_step`, `_dxi_hat_jets_core`, `_dxi_hat_jets_step_uniform`, `_dxi_hat_jets_step_ragged`, `_apply_derivatives_transpose_from_jets`, `_entries_ambient_jets`, `_deta_tilde_tangent_jets_step`, `_adj_tangent_src_order_step`, `_adj_prop_tangent`, `_adj_tilde_tangent_step`, `_adj_tangent_sweep_scanned_step_uniform`, `_adj_tangent_sweep_scanned_step_ragged`, `_adj_tangent_sweep_scanned`, `_dxi_tilde_tangent_order_step`, `_dxi_tilde_tangent_jets_core`, `_dxi_tilde_tangent_jets_step_uniform`, `_dxi_tilde_tangent_jets_step_ragged`, `_tucker_tangent_variation_jets_step`, `_assemble_tucker_tangent_variation_jets_dense`, `_tucker_tangent_chunk_summed`, `_tt_tangent_det_lag_step`, `_tt_tangent_variation_jets_step`, `_assemble_tt_tangent_variation_jets_dense`, `_tt_tangent_chunk_summed`, `_sigma_hat_tangent_jets_step`, `_dxi_hat_tangent_order_step`, `_dxi_hat_tangent_jets_core`, `_dxi_hat_tangent_jets_step_uniform`, `_dxi_hat_tangent_jets_step_ragged`, `_apply_tangent_assemble_step`, `_apply_derivatives_hessian_from_jets`

Existence of the other plan-named items in the tree (ast top-level defs, all `t3toolbox/**`):
- `Problem`: NOT DEFINED anywhere in t3toolbox/
- `least_squares_problem`: NOT DEFINED anywhere in t3toolbox/
- `_minibatch_step_problem`: NOT DEFINED anywhere in t3toolbox/
- `uniform_least_squares_problem`: NOT DEFINED anywhere in t3toolbox/
- `_ScaledRegularizer`: NOT DEFINED anywhere in t3toolbox/
- `_canonical_weight`: t3toolbox/fitting.py
- `_check_ragged_sample`: NOT DEFINED anywhere in t3toolbox/
- `LeastSquaresSetup`: NOT DEFINED anywhere in t3toolbox/

**Ambiguous (not in the deletion list as I determined it; callers are in (b)/(d)):** `t3toolbox/backend/fitting.py`: `_flat_gather`, `_prod_w`, `_take_W`, `_take_data_space`; `t3toolbox/backend/uniform_fitting.py`: `_fg`, `_frame_theta`. Also NOT deleted by my determination, because they belong to the new oracle: `_KINDS`, `_HOLD_LIST`, `_SCALED` (fitting.py 751-753, comment says new oracle), `_arrays_of`, `_stack_of_frame`, `_y_*_metric_y`, `LeastSquaresBatch`, `LeastSquaresOracle`.


### (b) References from LIBRARY code that is NOT itself deleted (the key list)

Counts: in `sampling_derivatives.py` itself 248; inside other to-be-deleted units (a removed name's own def/class/assign) 420; in kept library code 166 (by kind: {'importfrom': 5, 'all-entry': 99, 'docstring': 29, 'name': 2, 'attr': 2, 'comment': 17, 'sd-import-module': 3, 'sd-text-comment': 7, 'sd-text-docstring': 2}).
A "deleted unit" = top-level def/class/assign whose name is in the (a) lists of the same file. References inside them are not listed.

**Code references (ast) in kept library code** `file:line | enclosing function | name | kind (receiver)`:

- `t3toolbox/backend/apply.py:23` | `<module>` | `compute_dxi` | importfrom (`t3toolbox.backend.probing`)
- `t3toolbox/backend/apply.py:23` | `<module>` | `compute_mu` | importfrom (`t3toolbox.backend.probing`)
- `t3toolbox/backend/apply.py:23` | `<module>` | `compute_sigma_hat` | importfrom (`t3toolbox.backend.probing`)
- `t3toolbox/backend/apply.py:23` | `<module>` | `compute_xi` | importfrom (`t3toolbox.backend.probing`)
- `t3toolbox/backend/apply.py:633` | `_apply_transpose_adjoint` | `compute_sigma_hat` | name
- `t3toolbox/backend/apply.py:679` | `_apply_corewise_hessian_adjoint` | `compute_sigma_hat` | name
- `t3toolbox/backend/apply.py:680` | `_apply_corewise_hessian_adjoint` | `compute_tau` | attr (`probing`)
- `t3toolbox/backend/apply.py:681` | `_apply_corewise_hessian_adjoint` | `compute_deta` | attr (`probing`)
- `t3toolbox/backend/entries.py:26` | `<module>` | `compute_mu` | importfrom (`t3toolbox.backend.probing`)
- `t3toolbox/backend/fitting.py:34` | `<module>` | `<module>` | sd-import-module (`t3toolbox.backend.sampling_derivatives`)
- `t3toolbox/backend/ut3_sampling.py:26` | `<module>` | `<module>` | sd-import-module (`t3toolbox.backend.sampling_derivatives`)
- `t3toolbox/backend/utv_sampling.py:35` | `<module>` | `<module>` | sd-import-module (`t3toolbox.backend.sampling_derivatives`)

Of these, the `import` lines of `sampling_derivatives` are listed again in (e).

**`__all__` entries naming deleted names (kept modules; go with the names):** 
`t3toolbox/backend/apply.py` 4, `t3toolbox/backend/entries.py` 4, `t3toolbox/backend/fitting.py` 17, `t3toolbox/backend/probing.py` 27, `t3toolbox/backend/uniform_fitting.py` 8, `t3toolbox/backend/ut3_sampling.py` 15, `t3toolbox/backend/utv_sampling.py` 24 (count of removed-name entries in each `__all__`)

Doctest lines in library naming deleted names (kept or deleted code): 16.
Docstring-prose / comment mentions in KEPT library code (not code references; informational) `file:line | enclosing | name`:

- `t3toolbox/backend/apply.py:618` | `_apply_transpose_adjoint` | `tv_entries_transpose_from_sweep` (docstring) | `'''Adjoint-state assembly shared by apply/tv_entries_transpose_from_sweep -- the **low-memory,`
- `t3toolbox/backend/apply.py:665` | `_apply_corewise_hessian_adjoint` | `t3_apply_corewise_hessian_from_sweep` (docstring) | ```(B, G, G, G)`` along ``(Bdot, Gdot)``, shared by :py:func:`t3_apply_corewise_hessian_from_sweep` and`
- `t3toolbox/backend/apply.py:666` | `_apply_corewise_hessian_adjoint` | `t3_entries_corewise_hessian_from_sweep` (docstring) | `:py:func:`~t3toolbox.backend.entries.t3_entries_corewise_hessian_from_sweep` (which differ only in how`
- `t3toolbox/backend/apply.py:670` | `_apply_corewise_hessian_adjoint` | `compute_sigma_hat` (docstring) | `sigma_hat       = the c-seeded reverse sweep through G                (compute_sigma_hat)`
- `t3toolbox/backend/apply.py:671` | `_apply_corewise_hessian_adjoint` | `compute_tau` (docstring) | `sigma_hat_dot   = its derivative: the reverse GN tangent sweep        (compute_tau(Gdot, G, G, xi, xi_dot, sig`
- `t3toolbox/backend/apply.py:672` | `_apply_corewise_hessian_adjoint` | `compute_deta` (docstring) | `dxi_hat_dot     = d(mu G sigma_hat)                                    (compute_deta(Gdot, G, G, mu, sigma_hat`
- `t3toolbox/backend/common.py:526` | `readonly_mask_copies` | `_weight_matrix` (docstring) | `(the ``_weight_matrix`` precedent)."""`
- `t3toolbox/backend/entries.py:532` | `_entry_xis` | `compute_xi` (docstring) | `Returns xis with elm_shape = W + C + (p_i,) -- the same layout compute_xi produces, so the`
- `t3toolbox/backend/fitting.py:12` | `<module>` | `SamplingKind` (docstring) | `This module supplies the **kind** half: a :py:class:`SamplingKind` descriptor bundling the bare probing`
- `t3toolbox/backend/fitting.py:309` | `<module>` | `ProbeOutputKind` (comment) | `#                      reductions, supplied by ScalarOutputKind / ProbeOutputKind.`
- `t3toolbox/backend/fitting.py:309` | `<module>` | `ScalarOutputKind` (comment) | `#                      reductions, supplied by ScalarOutputKind / ProbeOutputKind.`
- `t3toolbox/backend/fitting.py:738` | `<module>` | `probe_derivatives_kind` (comment) | `# (`probe_derivatives_kind(order, weight)`) and because it is the documented public surface.`
- `t3toolbox/backend/probing.py:8` | `<module>` | `assemble_z` (docstring) | `module holds the general machinery -- the ``xi/mu/nu/eta`` helper chain, ``assemble_z``, the tv`
- `t3toolbox/backend/probing.py:119` | `_xi_step` | `compute_xi` (docstring) | `'''One core of the ragged map of :py:func:`compute_xi`. Closure-free map body --`
- `t3toolbox/backend/probing.py:154` | `_mu_step` | `compute_mu` (docstring) | `'''One edge of the left-to-right sweep of :py:func:`compute_mu`. Closure-free scan body --`
- `t3toolbox/backend/probing.py:209` | `_eta_step` | `compute_eta` (docstring) | `'''One core of the ragged map of :py:func:`compute_eta`. Closure-free map body --`
- `t3toolbox/backend/probing.py:244` | `_assemble_z_step` | `assemble_z` (docstring) | `'''One core of the ragged map of :py:func:`assemble_z`. Closure-free map body --`
- `t3toolbox/backend/probing.py:307` | `_sigma_step` | `compute_sigma` (docstring) | `compute_sigma (which keeps the per-core sequence, for probing) and tv_apply/tv_entries`
- `t3toolbox/backend/probing.py:326` | `_sigma_sweep_step` | `compute_sigma` (docstring) | `'''One edge of the sweep of :py:func:`compute_sigma`, keeping the per-core sequence. Closure-free`
- `t3toolbox/backend/probing.py:408` | `_deta_step` | `compute_deta` (docstring) | `'''One core of the ragged map of :py:func:`compute_deta`. Closure-free map body --`
- `t3toolbox/backend/probing.py:411` | `_deta_step` | `compute_sigma` (comment) | `# Three-group contractions (see compute_sigma): sigma/tau carry K, mu/nu and frame cores`
- `t3toolbox/backend/probing.py:469` | `_assemble_tangent_z_step` | `assemble_tangent_z` (docstring) | `'''One core of the ragged map of :py:func:`assemble_tangent_z`. Closure-free map body --`
- `t3toolbox/backend/probing.py:472` | `_assemble_tangent_z_step` | `compute_sigma` (comment) | `# Three-group contractions (see compute_sigma): deta carries K (in term1 both W and K ride`
- `t3toolbox/backend/probing.py:637` | `_sigma_hat_step` | `compute_sigma_hat` (docstring) | `'''One edge of the adjoint reverse sweep of :py:func:`compute_sigma_hat`. Closure-free scan body`
- `t3toolbox/backend/probing.py:679` | `_deta_tilde_step` | `compute_deta_tilde` (docstring) | `'''One core of the ragged map of :py:func:`compute_deta_tilde`. Closure-free map body --`
- `t3toolbox/backend/probing.py:683` | `_deta_tilde_step` | `compute_xi` (comment) | `# zt. This is NOT compute_xi (which forms an outer product over the two stacks).`
- `t3toolbox/backend/probing.py:720` | `_tau_tilde_step` | `compute_tau_tilde` (docstring) | `'''One edge of the adjoint left-to-right sweep of :py:func:`compute_tau_tilde`. Closure-free scan`
- `t3toolbox/backend/probing.py:784` | `_dxi_tilde_step` | `compute_dxi_tilde` (docstring) | `'''One core of the ragged map of :py:func:`compute_dxi_tilde`. Closure-free map body --`
- `t3toolbox/backend/probing.py:787` | `_dxi_tilde_step` | `compute_tau_tilde` (comment) | `# Three-group (see compute_tau_tilde): tt/st carry K, mu/nu and the frame core U do not.`
- `t3toolbox/backend/probing.py:831` | `_assemble_tucker_variations_core` | `assemble_tucker_variations` (docstring) | `'''One core of the ragged map of :py:func:`assemble_tucker_variations`, shared by the two`
- `t3toolbox/backend/probing.py:906` | `_assemble_tt_variations_core` | `assemble_tt_variations` (comment) | `n_probe:         int,      # len(W); rides as an extra map operand -- see assemble_tt_variations`
- `t3toolbox/backend/probing.py:909` | `_assemble_tt_variations_core` | `assemble_tt_variations` (docstring) | `'''One core of the ragged map of :py:func:`assemble_tt_variations`, shared by the two`
- `t3toolbox/backend/probing.py:1245` | `<module>` | `sampling_derivatives` (sd-text-comment) | `# probe-derivative two-hole (sampling_derivatives.t3_probe_corewise_derivatives_hessian), written in plain`
- `t3toolbox/backend/probing.py:1245` | `<module>` | `t3_probe_corewise_derivatives_hessian` (sd-text-comment) | `# probe-derivative two-hole (sampling_derivatives.t3_probe_corewise_derivatives_hessian), written in plain`
- `t3toolbox/backend/probing.py:1256` | `<module>` | `compute_deta` (comment) | `#      mudot / nudot = compute_sigma / compute_tau, etadot = compute_deta;`
- `t3toolbox/backend/probing.py:1256` | `<module>` | `compute_sigma` (comment) | `#      mudot / nudot = compute_sigma / compute_tau, etadot = compute_deta;`
- `t3toolbox/backend/probing.py:1256` | `<module>` | `compute_tau` (comment) | `#      mudot / nudot = compute_sigma / compute_tau, etadot = compute_deta;`
- `t3toolbox/backend/probing.py:1260` | `<module>` | `compute_deta` (comment) | `#   5. dotted dxi~: two compute_deta calls (the product rule of its two trilinear combines);`
- `t3toolbox/backend/probing.py:1271` | `_deta_tilde_tangent_step` | `compute_deta_tilde_tangent` (docstring) | `'''One core of the ragged map of :py:func:`compute_deta_tilde_tangent`. The C-only ``D`` is passed only to`
- `t3toolbox/backend/probing.py:1299` | `_tau_tilde_tangent_step` | `compute_tau_tilde_tangent` (docstring) | `'''One edge of the dotted adjoint sweep of :py:func:`compute_tau_tilde_tangent`: the product rule of`
- `t3toolbox/backend/probing.py:1390` | `_tucker_tangent_variations_step` | `assemble_tucker_tangent_variations` (docstring) | `'''One core of the ragged map of :py:func:`assemble_tucker_tangent_variations`; ``len(W)`` is read off the`
- `t3toolbox/backend/probing.py:1421` | `_tt_tangent_variations_step` | `assemble_tt_tangent_variations` (docstring) | `'''One core of the factored TT tangent assembly (:py:func:`assemble_tt_tangent_variations`). ``n_probe```
- `t3toolbox/backend/uniform_fitting.py:5` | `<module>` | `SamplingKind` (docstring) | `uniform start, and the old packed ``SamplingKind`` classes.`
- `t3toolbox/backend/uniform_fitting.py:74` | `<module>` | `_UniformLayerMixin` (comment) | `# `_UniformLayerMixin` precedes the ragged kind in the MRO but defines no corewise_hessian, so a uniform kind `
- `t3toolbox/backend/uniform_fitting.py:457` | `<module>` | `SamplingKind` (comment) | `# Packed-aware minibatch `take` (the SamplingKind default-draw hook), so mc_sgd/adam keep minibatches`
- `t3toolbox/backend/ut3_sampling.py:11` | `<module>` | `sampling_derivatives` (sd-text-docstring) | ```probing``/``apply``/``entries``/``sampling_derivatives`` (``docs/naming_conventions.md``).`
- `t3toolbox/backend/ut3_sampling.py:18` | `<module>` | `sampling_derivatives` (sd-text-docstring) | `in, packed out. The jet wrappers (``*_derivatives*``) still share ``sampling_derivatives``.`
- `t3toolbox/backend/ut3_sampling.py:259` | `<module>` | `sampling_derivatives` (sd-text-comment) | `# the polymorphic sampling_derivatives.t3_*_derivatives, and unpack the probe output (which now carries a`
- `t3toolbox/backend/ut3_sampling.py:313` | `<module>` | `sampling_derivatives` (sd-text-comment) | `# sampling_derivatives.*_corewise_derivatives_transpose). Mask-once + pack ww/pp (entries: pp only) at the`
- `t3toolbox/backend/ut3_sampling.py:374` | `<module>` | `ut3_probe_corewise_derivatives_transpose` (comment) | `# The derivative of ut3_probe_corewise_derivatives_transpose along raw core directions (the exact Riemannian`
- `t3toolbox/backend/utv_sampling.py:136` | `<module>` | `SamplingKind` (comment) | `# The SamplingKind hooks for the uniform Gauss-Newton fitting model (optimizers-on-uniform U3), the uniform`
- `t3toolbox/backend/utv_sampling.py:254` | `<module>` | `sampling_derivatives` (sd-text-comment) | `# uniform U3'), the uniform twins of the ragged jet sweep machinery (sampling_derivatives: tv_precompute_*_fra`
- `t3toolbox/backend/utv_sampling.py:262` | `<module>` | `SamplingKind` (comment) | `# geometry-agnostic SamplingKind (sumsq / transpose), not here. The bare 𝒥 / 𝒥ᵀ (no gauge projector Π).`
- `t3toolbox/backend/utv_sampling.py:392` | `<module>` | `sampling_derivatives` (sd-text-comment) | `# fibers -> only pp packed), share sampling_derivatives.*_tangent_derivatives, unpack the probe output (which`
- `t3toolbox/backend/utv_sampling.py:532` | `<module>` | `sampling_derivatives` (sd-text-comment) | `# (entries: pp only), share sampling_derivatives.*_tangent_derivatives_transpose, attach the gauge masks over`

Per-file ref counts in library, all kinds, split: (in sd file / in deleted unit / kept):

- `t3toolbox/backend/apply.py`: 0 / 14 / 18
- `t3toolbox/backend/common.py`: 0 / 0 / 1
- `t3toolbox/backend/entries.py`: 0 / 10 / 6
- `t3toolbox/backend/fitting.py`: 0 / 111 / 22
- `t3toolbox/backend/probing.py`: 0 / 87 / 57
- `t3toolbox/backend/sampling_derivatives.py`: 248 / 0 / 0
- `t3toolbox/backend/uniform_fitting.py`: 0 / 74 / 11
- `t3toolbox/backend/ut3_sampling.py`: 0 / 57 / 21
- `t3toolbox/backend/utv_sampling.py`: 0 / 67 / 30

### (c) References from `examples/` and `docs/` pages

**examples/** (ast + text; `file:line | enclosing | name | kind`):

- none

**docs/ pages** (`.md`/`.rst`, regex on every line; kind = doctest line / other fenced-code line / prose). Each line: `file:line | name | kind`. Names matched: every name of (a) plus the 57 `sampling_derivatives` public names, `DEFAULT_CHUNK_SIZE` and the word `sampling_derivatives`. NOTE: generic upper-case constant names `APPLY`/`ENTRIES`/`PROBE` match prose words; those are listed separately.

- `docs/api_reference.rst`: 1 lines-matches (prose 1)
- `docs/contributor/ambient_derivative_transpose_note.md`: 1 lines-matches (prose 1)
- `docs/contributor/batching_internals.md`: 1 lines-matches (prose 1)
- `docs/contributor/curvature_internals.md`: 5 lines-matches (prose 5)
- `docs/contributor/deferred_and_rejected.md`: 2 lines-matches (prose 2)
- `docs/contributor/parameters_not_closures.md`: 4 lines-matches (prose 4); plus 5 generic APPLY/ENTRIES/PROBE word matches
- `docs/contributor/scan_body_principles.md`: 2 lines-matches (prose 2)
- `docs/contributor/signature_style.md`: 1 lines-matches (fence 1)
- `docs/contributor/verification_records.md`: 1 lines-matches (prose 1)
- `docs/entries_apply_probe.md`: 13 lines-matches (prose 13)
- `docs/fitting_and_optimization.md`: 11 lines-matches (fence 2, prose 9); plus 6 generic APPLY/ENTRIES/PROBE word matches
- `docs/glossary.md`: 5 lines-matches (prose 5)
- `docs/naming_conventions.md`: 10 lines-matches (prose 10)
- `docs/reading_signatures.md`: 1 lines-matches (fence 1)
- `docs/release_notes.md`: 11 lines-matches (prose 11); plus 3 generic APPLY/ENTRIES/PROBE word matches
- `docs/uniform_equivalence_contract.md`: 1 lines-matches (prose 1)

Docs totals (excluding generic words): 70 matches in 16 pages; by kind {'prose': 66, 'fence': 4}

Detail, `file:line | name | kind`:

- `docs/api_reference.rst:130` | `sampling_derivatives` | prose
- `docs/contributor/ambient_derivative_transpose_note.md:13` | `sampling_derivatives` | prose
- `docs/contributor/batching_internals.md:66` | `tv_entries_transpose_from_sweep` | prose
- `docs/contributor/curvature_internals.md:31` | `SamplingKind` | prose
- `docs/contributor/curvature_internals.md:102` | `build_input_jets` | prose
- `docs/contributor/curvature_internals.md:102` | `compute_xi` | prose
- `docs/contributor/curvature_internals.md:102` | `compute_xi` | prose
- `docs/contributor/curvature_internals.md:349` | `estimate_chunk_size` | prose
- `docs/contributor/deferred_and_rejected.md:122` | `sampling_derivatives` | prose
- `docs/contributor/deferred_and_rejected.md:260` | `max_chunk_size_within` | prose
- `docs/contributor/parameters_not_closures.md:24` | `probe_derivatives_kind` | prose
- `docs/contributor/parameters_not_closures.md:28` | `SamplingKind` | prose
- `docs/contributor/parameters_not_closures.md:156` | `SamplingKind` | prose
- `docs/contributor/parameters_not_closures.md:187` | `ApplyKind` | prose
- `docs/contributor/scan_body_principles.md:223` | `compute_mu_jets` | prose
- `docs/contributor/scan_body_principles.md:223` | `sampling_derivatives` | prose
- `docs/contributor/signature_style.md:55` | `compute_mu` | fence
- `docs/contributor/verification_records.md:87` | `estimate_chunk_size` | prose
- `docs/entries_apply_probe.md:243` | `compute_xi` | prose
- `docs/entries_apply_probe.md:244` | `compute_mu` | prose
- `docs/entries_apply_probe.md:245` | `compute_nu` | prose
- `docs/entries_apply_probe.md:246` | `compute_eta` | prose
- `docs/entries_apply_probe.md:247` | `assemble_z` | prose
- `docs/entries_apply_probe.md:259` | `compute_dxi` | prose
- `docs/entries_apply_probe.md:260` | `compute_sigma` | prose
- `docs/entries_apply_probe.md:261` | `compute_tau` | prose
- `docs/entries_apply_probe.md:262` | `compute_deta` | prose
- `docs/entries_apply_probe.md:263` | `assemble_tangent_z` | prose
- `docs/entries_apply_probe.md:268` | `compute_deta_tilde` | prose
- `docs/entries_apply_probe.md:269` | `assemble_tucker_variations` | prose
- `docs/entries_apply_probe.md:269` | `assemble_tt_variations` | prose
- `docs/fitting_and_optimization.md:19` | `SamplingKind` | prose
- `docs/fitting_and_optimization.md:247` | `SamplingKind` | fence
- `docs/fitting_and_optimization.md:249` | `sampling_derivatives` | fence
- `docs/fitting_and_optimization.md:253` | `SamplingKind` | prose
- `docs/fitting_and_optimization.md:258` | `ProbeKind` | prose
- `docs/fitting_and_optimization.md:261` | `ApplyKind` | prose
- `docs/fitting_and_optimization.md:261` | `ProbeKind` | prose
- `docs/fitting_and_optimization.md:262` | `ScalarOutputKind` | prose
- `docs/fitting_and_optimization.md:262` | `ProbeOutputKind` | prose
- `docs/fitting_and_optimization.md:264` | `SamplingKind` | prose
- `docs/fitting_and_optimization.md:349` | `SamplingKind` | prose
- `docs/glossary.md:404` | `estimate_chunk_size` | prose
- `docs/glossary.md:404` | `max_chunk_size_within` | prose
- `docs/glossary.md:421` | `SamplingKind` | prose
- `docs/glossary.md:461` | `SamplingKind` | prose
- `docs/glossary.md:789` | `sampling_derivatives` | prose
- `docs/naming_conventions.md:82` | `sampling_derivatives` | prose
- `docs/naming_conventions.md:88` | `dense_apply_derivatives` | prose
- `docs/naming_conventions.md:150` | `compute_mu_jets` | prose
- `docs/naming_conventions.md:160` | `SamplingKind` | prose
- `docs/naming_conventions.md:162` | `compute_tau_tilde_tangent_jets` | prose
- `docs/naming_conventions.md:162` | `compute_tau_tilde_jets` | prose
- `docs/naming_conventions.md:162` | `assemble_tt_tangent_variation_jets` | prose
- `docs/naming_conventions.md:163` | `assemble_tt_variation_jets` | prose
- `docs/naming_conventions.md:164` | `compute_dxi_tilde_tangent` | prose
- `docs/naming_conventions.md:193` | `sampling_derivatives` | prose
- `docs/reading_signatures.md:11` | `compute_mu` | fence
- `docs/release_notes.md:36` | `sampling_derivatives` | prose
- `docs/release_notes.md:183` | `SamplingKind` | prose
- `docs/release_notes.md:190` | `sampling_derivatives` | prose
- `docs/release_notes.md:191` | `compute_dxi_tilde_jets` | prose
- `docs/release_notes.md:193` | `estimate_chunk_size` | prose
- `docs/release_notes.md:193` | `max_chunk_size_within` | prose
- `docs/release_notes.md:226` | `ApplyKind` | prose
- `docs/release_notes.md:238` | `UniformApplyKind` | prose
- `docs/release_notes.md:238` | `uniform_sampling_kind` | prose
- `docs/release_notes.md:239` | `uniform_derivatives_kind` | prose
- `docs/release_notes.md:240` | `probe_kind` | prose
- `docs/uniform_equivalence_contract.md:118` | `sumsq_over_probes` | prose

### (d) Tests: references per file, and whole-file-or-partial

References counted in test files: code refs (ast Name/Attribute/import/kwarg), `sampling_derivatives` alias refs, and string-constant refs (e.g. `mock.patch` strings, subprocess script strings); docstring/comment/doctest mentions counted separately. A test = every `def test_*` (functions and methods). "direct" = the test body (incl. nested defs) has a ref; "indirect" = the test body calls (by Name or `self.<attr>`) a same-file function/method whose own body has a ref. "setUp/module" = refs outside any test (imports, helpers, class setup).

| test file | refs (code+string) | refs (docstring/comment) | n tests | direct | direct+indirect | refs outside tests | verdict |
|---|---|---|---|---|---|---|---|
| `tests/_jet_trs_twins.py` | 0 | 1 | 0 | 0 | 0 | 0 | no tests (helper module) |
| `tests/backend/_curvature_oracles.py` | 13 | 11 | 0 | 0 | 0 | 13 | no tests (helper module) |
| `tests/backend/test_curvature.py` | 83 | 1 | 96 | 14 | 29 | 16 | PARTIAL: 14 of 96 |
| `tests/backend/test_from_sweep_reuse.py` | 37 | 0 | 5 | 5 | 5 | 1 | ALL tests reference (whole-file delete candidate) |
| `tests/backend/test_uniform_curvature.py` | 51 | 9 | 56 | 7 | 10 | 31 | PARTIAL: 7 of 56 |
| `tests/backend/test_uniform_fitting.py` | 41 | 4 | 30 | 6 | 11 | 19 | PARTIAL: 6 of 30 |
| `tests/bridge/test_old_kernels_bridge.py` | 26 | 23 | 9 | 9 | 9 | 4 | ALL tests reference (whole-file delete candidate) |
| `tests/oracle_sweeps/o1_ragged.py` | 6 | 0 | 0 | 0 | 0 | 6 | no tests (helper module) |
| `tests/oracle_sweeps/o1_uniform.py` | 3 | 0 | 0 | 0 | 0 | 3 | no tests (helper module) |
| `tests/test_chunk_size_estimator.py` | 38 | 4 | 12 | 12 | 12 | 2 | ALL tests reference (whole-file delete candidate) |
| `tests/test_contractions_interpreter.py` | 0 | 1 | 20 | 0 | 0 | 0 | no test references directly (refs only in helpers/imports) |
| `tests/test_contractions_sharding.py` | 3 | 0 | 5 | 1 | 1 | 0 | PARTIAL: 1 of 5 |
| `tests/test_dispatch.py` | 71 | 1 | 108 | 10 | 11 | 1 | PARTIAL: 10 of 108 |
| `tests/test_jet_recurrence.py` | 32 | 1 | 5 | 5 | 5 | 9 | ALL tests reference (whole-file delete candidate) |
| `tests/test_jets.py` | 4 | 2 | 39 | 2 | 2 | 0 | PARTIAL: 2 of 39 |
| `tests/test_least_squares_oracle.py` | 3 | 0 | 18 | 1 | 1 | 0 | PARTIAL: 1 of 18 |
| `tests/test_manifold.py` | 2 | 0 | 57 | 2 | 2 | 0 | PARTIAL: 2 of 57 |
| `tests/test_probe_derivatives.py` | 45 | 1 | 17 | 16 | 16 | 1 | PARTIAL: 16 of 17 |
| `tests/test_sampling_definitions.py` | 5 | 3 | 72 | 2 | 2 | 0 | PARTIAL: 2 of 72 |
| `tests/test_tucker_tensor_train.py` | 4 | 0 | 81 | 2 | 2 | 1 | PARTIAL: 2 of 81 |
| `tests/test_uniform_probing.py` | 22 | 0 | 35 | 0 | 2 | 22 | PARTIAL: 0 of 35 |

Test files with no reference at all are not listed. Totals: 21 test files with refs; 489 code/string refs.

**Per-class breakdown for partial files** (`class: tests / direct / direct+indirect`; tests outside any class are `<module>`):

- `tests/backend/test_curvature.py`: TestProbeDerivativesTwoHole 8/8/8; TestPlainKindsTwoHole 9/3/9; TestDerivativeKindsTwoHole 10/3/10; TestPlainKindsExactLocalModel 5/0/2; (classes with 0/0: TestFrameFactors 3, TestLiftIdentity 3, TestLiftAdjoint 1, TestDamping 6, TestTangentZipper 1, TestInnerProductTwoHole 5, TestFlatSlotsAndVerticalDirections 3, TestExactLocalModel 10, TestDerivativeKindsExactLocalModel 6, TestCorewiseExact 2, TestFlatSlotsModel 3, TestRequiredSixthOperation 3, TestExactNewtonRate 3, TestExactNewtonRestart 1, TestExactNewtonFromNonMinimalPoints 2, TestSharedExactLift 3, TestSharedExactLocalModel 8, TestSharedExactNewtonRestart 1)
- `tests/backend/test_uniform_curvature.py`: TestUniformProbeDerivativesTwoHole 2/2/2; TestUniformKindCorewiseHessian 3/3/3; TestUniformPlainKindCorewiseHessian 3/1/1; TestUniformDerivativeKindsTwoHole 2/0/2; TestUniformDerivativeKindCorewiseHessian 3/1/2; (classes with 0/0: TestUniformLiftFactors 3, TestUniformLift 2, TestUniformInnerProductTwoHole 1, TestUniformPlainKindsTwoHole 2, TestUniformGeometryExactMembers 3, TestUniformExactLocalModel 5, TestUniformKStackedGaussNewton 1, TestUniformPlainKindsExactLocalModel 3, TestUniformExactNewton 5, TestUniformSharedTiedLift 7, TestUniformSharedExactModel 8, TestUniformSharedExactNewton 3)
- `tests/backend/test_uniform_fitting.py`: TestUniformSamplingKind 4/4/4; TestUniformBlockSumsq 1/1/1; TestUniformDerivativeSamplingKind 4/0/4; TestKindTransposeOutputHygiene 1/0/1; TestSplitSeamDefaultParity 1/1/1; (classes with 0/0: TestUniformGeometryOps 2, TestUniformSamplingMirror 3, TestUniformProblem 2, TestUniformOptimizers 4, TestUniformMinimalRank 6, TestPackSampleAndData 1, TestSlackPaddedStart 1)
- `tests/test_contractions_interpreter.py`: ; (classes with 0/0: TestVocabularyCompleteness 1, TestVocabularyOracle 1, TestSupplementContract 1, TestCallSiteConsistency 1, TestSplitInvariance 1, TestGenericStrings 1, TestErrorsAndValidation 9, TestJetLiteralsStayInJetContract 3, TestJetContractDerivedStrings 2)
- `tests/test_contractions_sharding.py`: TestJetKernelsShardOverW 1/1/1; (classes with 0/0: TestInterpreterAnyAxisSharding 3, TestTheEngineShardsOverW 1)
- `tests/test_dispatch.py`: TestDispatch 79/7/8; TestJetDtypeUnderX64 1/1/1; TestCurvatureJvpOracle 2/2/2; (classes with 0/0: TestAlignedLoopJit 3, TestUniformAlignedLoopJit 3, TestPaddedStartSeam 2, TestAlignmentTiming 1, TestNewtonTiming 3, TestSwitchNewton 4, TestAlignedPaddedFrameFrontend 2, TestJitStack 2, TestTracedMaskGuard 2, TestJitUniformWeighted 4)
- `tests/test_jets.py`: TestCheckers 15/1/1; TestFrontendReach 2/1/1; (classes with 0/0: TestDualJet 13, TestDerivativeData 9)
- `tests/test_least_squares_oracle.py`: TestTheBridgeToTheOldKinds 1/1/1; (classes with 0/0: TestTheEngineAdditions 5, TestQuery 2, TestTheResolutionDtype 1, TestTheCorewiseDirection 1, TestTheMinibatchScale 2, TestTheBatch 3, TestTheOracle 3)
- `tests/test_manifold.py`: TestManifold 42/2/2; (classes with 0/0: TestStructuralTangentMismatch 2, TestD1Degenerate 1, TestGaugeResidualIsRelative 1, TestCorewiseRetractFrameGuard 4, TestGaugeOperandScale 7)
- `tests/test_probe_derivatives.py`: TestProbeDerivatives 15/14/14; TestAsymmetricGenericFrame 2/2/2; (classes with 0/0: )
- `tests/test_sampling_definitions.py`: TestAgainstTheOldKernels 1/1/1; TestDenseFunctions 17/1/1; (classes with 0/0: TestOutputsAgainstDense 5, TestTheWeightIsOfTheTypeOfZ 4, TestRecurrences 5, TestReadouts 3, TestPairings 3, TestApplyAndEntries 4, TestJetsTwoWays 3, TestKp 5, TestCorewiseFrame 2, TestStructure 15, TestJax 1, TestInputJets 4)
- `tests/test_tucker_tensor_train.py`: TestTuckerTensorTrain 72/2/2; (classes with 0/0: TestEntriesIndexSemantics 3, TestDenseProbeD1 1, TestReviewC13Tucker 3, TestRankAdjustmentSweepTuckerAboveModeSize 1, TestScalarOperatorSymmetry 1)
- `tests/test_uniform_probing.py`: TestTransposeOutputHygiene 2/0/2; (classes with 0/0: TestUT3TangentForward 4, TestUT3TangentForwardDerivatives 4, TestUniformT3PlainDerivatives 2, TestUT3TangentTranspose 5, TestUT3CorewiseTranspose 3, TestUT3TangentTransposeDerivatives 4, TestUT3CorewiseTransposeDerivatives 3, TestUT3ProbingHardening 4, TestUT3DerivativeHardening 4)

### (e) Every `import` of `sampling_derivatives`

Method: `rg -n "(import|from)[^#\n]*sampling_derivatives|sampling_derivatives import"` over the whole repo excluding `.git`, plus an `ast` check of all `.py` files under `t3toolbox/ tests/ examples/` for Import/ImportFrom nodes (incl. function-local) that end in `sampling_derivatives` or import the name `sampling_derivatives`. Library, tests, examples by ast:

- [library] `t3toolbox/backend/fitting.py:34` | `<module>` | `import t3toolbox.backend.sampling_derivatives as pd`
- [library] `t3toolbox/backend/ut3_sampling.py:26` | `<module>` | `import t3toolbox.backend.sampling_derivatives as sampling_derivatives`
- [library] `t3toolbox/backend/utv_sampling.py:35` | `<module>` | `import t3toolbox.backend.sampling_derivatives as sampling_derivatives`
- [tests] `tests/backend/_curvature_oracles.py:53` | `<module>` | `import t3toolbox.backend.sampling_derivatives as sd`
- [tests] `tests/backend/test_curvature.py:100` | `<module>` | `import t3toolbox.backend.sampling_derivatives as sd`
- [tests] `tests/backend/test_from_sweep_reuse.py:19` | `<module>` | `import t3toolbox.backend.sampling_derivatives as pd`
- [tests] `tests/backend/test_uniform_curvature.py:53` | `<module>` | `import t3toolbox.backend.sampling_derivatives as sd`
- [tests] `tests/backend/test_uniform_fitting.py:490` | `TestSplitSeamDefaultParity.test_sum_over_probes_defaults_match_ragged` | `import t3toolbox.backend.sampling_derivatives as bsd`
- [tests] `tests/bridge/test_old_kernels_bridge.py:71` | `<module>` | `import t3toolbox.backend.sampling_derivatives as sampling_derivatives`
- [tests] `tests/test_chunk_size_estimator.py:26` | `<module>` | `import t3toolbox.backend.sampling_derivatives as pd`
- [tests] `tests/test_contractions_sharding.py:262` | `TestJetKernelsShardOverW.test_probe_transpose_and_two_hole_shard_over_W` | `import t3toolbox.backend.sampling_derivatives as sd`
- [tests] `tests/test_dispatch.py:37` | `<module>` | `import t3toolbox.backend.sampling_derivatives as pd`
- [tests] `tests/test_dispatch.py:3991` | `TestJetDtypeUnderX64.test_float32_in_float32_out` | `import t3toolbox.backend.sampling_derivatives as sd`
- [tests] `tests/test_jet_recurrence.py:27` | `<module>` | `import t3toolbox.backend.sampling_derivatives as pd`
- [tests] `tests/test_probe_derivatives.py:14` | `<module>` | `import t3toolbox.backend.sampling_derivatives as pd`
- [tests] `tests/test_sampling_definitions.py:1346` | `TestAgainstTheOldKernels.test_z_and_g_of_the_probe_kind` | `import t3toolbox.backend.sampling_derivatives as sampling_derivatives`
- [tests] `tests/test_sampling_definitions.py:1992` | `TestDenseFunctions.test_the_jets_of_entries_off_the_grid` | `import t3toolbox.backend.sampling_derivatives as sampling_derivatives`
- [tests] `tests/test_tucker_tensor_train.py:13` | `<module>` | `import t3toolbox.backend.sampling_derivatives as sampling_derivatives`
- [tests] `tests/test_uniform_probing.py:24` | `<module>` | `import t3toolbox.backend.sampling_derivatives as bsd`

Counts by area: {'library': 3, 'tests': 16}; function-local (lazy) imports: [('tests/backend/test_uniform_fitting.py', 490), ('tests/test_contractions_sharding.py', 262), ('tests/test_dispatch.py', 3991), ('tests/test_sampling_definitions.py', 1346), ('tests/test_sampling_definitions.py', 1992)].

Non-.py files and other places (grep, outside `dev/`; the same regex; a line in a doc page that mentions import in prose is included):

- none

`dev/` Python files importing it (informational; the plan keeps these as records): 33 files: `dev/archive/review_2026-08-22/repros/R6/followup.py`, `dev/archive/review_2026-08-22/repros/R6/two_chunk_mem.py`, `dev/archive/review_2026-08-22/repros/R6/common_r6.py`, `dev/archive/review_2026-08-22/repros/R6/chunk_jax.py`, `dev/archive/review_2026-08-22/repros/R6/asym_numpy.py`, `dev/archive/review_2026-08-22/repros/H7/changelog_symbols.py`, `dev/archive/curvature_impl_review_2026-10-29/critic_checks/check_onehole_boundary_leak.py`, `dev/archive/curvature_impl_review_2026-10-29/critic_checks/check_c14_d1.py`, `dev/archive/curvature_impl_review_2026-10-29/critic_checks/check_c14_extra.py`, `dev/archive/curvature_impl_review_2026-10-29/checks/c24_restart_probe_derivs.py`, `dev/archive/curvature_impl_review_2026-10-29/checks/c35c_tt_assembly_factored.py`, `dev/archive/curvature_impl_review_2026-10-29/checks/c14_probe_derivs_twohole.py`, `dev/archive/curvature_impl_review_2026-10-29/checks/c25_cost_probe_derivs.py`, `dev/archive/curvature_impl_review_2026-10-29/checks/c35d_standard_form_cost.py`, `dev/archive/curvature_impl_review_2026-10-29/checks/c13_apply_jets_twohole.py`, `dev/archive/curvature_impl_review_2026-10-29/checks/c18_corewise_sweep_from_manifold.py`, `dev/archive/curvature_impl_review_2026-10-29/checks/c26_nu_transform_frame_specific.py`, `dev/archive/curvature_impl_review_2026-10-29/checks/c22_probe_derivs_v3_recipe.py`, `dev/archive/curvature_impl_review_2026-10-29/checks/c15_shared_exact.py`, `dev/archive/curvature_impl_review_2026-10-29/checks/c35_cost_breakdown.py`, `dev/archive/curvature_impl_review_2026-10-29/checks/c9_blocksum_jets.py`, `dev/archive/curvature_impl_review_2026-10-29/checks/c35b_twohole_phase_breakdown.py`, `dev/archive/curvature_impl_review_2026-10-29/slices/trs_cost.py`, `dev/archive/store_rename_2026-11-15/store_rename_battery.py`, `dev/graded_messages/reviews/review_1_scratch/proto_tails.py`, `dev/graded_messages/implementation/bench/bench_probe_jets.py`, `dev/graded_messages/implementation/bench/bench_apply_jets.py`, `dev/graded_messages/scripts/review/slot_tuple_examples.py`, `dev/graded_messages/scripts/extra_core_experiment.py`, `dev/graded_messages/scripts/correspondence_check.py`, `dev/graded_messages/scripts/f32_experiment.py`, `dev/graded_messages/scripts/graded_messages_prototype.py`, `dev/graded_messages/scripts/eta_axes.py`

### (f) `estimate_chunk_size`, `max_chunk_size_within`, `DEFAULT_CHUNK_SIZE`

Definitions (ast top-level in `t3toolbox/`):
- `DEFAULT_CHUNK_SIZE`: `t3toolbox/backend/sampling_derivatives.py:27`
- `estimate_chunk_size`: `t3toolbox/backend/sampling_derivatives.py:1919`
- `max_chunk_size_within`: `t3toolbox/backend/sampling_derivatives.py:1977`

**library** (every reference, ast + text; `file:line | enclosing | name | kind (receiver)`; definition sites excluded):

- `t3toolbox/backend/sampling_derivatives.py:101` | `__all__` | `estimate_chunk_size` | string
- `t3toolbox/backend/sampling_derivatives.py:102` | `__all__` | `max_chunk_size_within` | string
- `t3toolbox/backend/sampling_derivatives.py:1516` | `assemble_tucker_variation_jets` | `DEFAULT_CHUNK_SIZE` | name
- `t3toolbox/backend/sampling_derivatives.py:1741` | `assemble_tt_variation_jets` | `DEFAULT_CHUNK_SIZE` | name
- `t3toolbox/backend/sampling_derivatives.py:1785` | `<module>` | `estimate_chunk_size` | comment
- `t3toolbox/backend/sampling_derivatives.py:1786` | `<module>` | `max_chunk_size_within` | comment
- `t3toolbox/backend/sampling_derivatives.py:1940` | `estimate_chunk_size` | `max_chunk_size_within` | docstring
- `t3toolbox/backend/sampling_derivatives.py:1951` | `estimate_chunk_size` | `max_chunk_size_within` | docstring
- `t3toolbox/backend/sampling_derivatives.py:1962` | `estimate_chunk_size` | `estimate_chunk_size` | string
- `t3toolbox/backend/sampling_derivatives.py:1993` | `max_chunk_size_within` | `estimate_chunk_size` | docstring
- `t3toolbox/backend/sampling_derivatives.py:2002` | `max_chunk_size_within` | `estimate_chunk_size` | docstring
- `t3toolbox/backend/sampling_derivatives.py:2003` | `max_chunk_size_within` | `max_chunk_size_within` | string
- `t3toolbox/backend/sampling_derivatives.py:2028` | `tv_probe_transpose_derivatives_from_sweep` | `DEFAULT_CHUNK_SIZE` | name
- `t3toolbox/backend/sampling_derivatives.py:2063` | `tv_probe_derivatives_transpose` | `DEFAULT_CHUNK_SIZE` | name
- `t3toolbox/backend/sampling_derivatives.py:2085` | `tv_probe_derivatives_transpose` | `estimate_chunk_size` | docstring
- `t3toolbox/backend/sampling_derivatives.py:2402` | `t3_probe_corewise_derivatives_transpose` | `DEFAULT_CHUNK_SIZE` | name
- `t3toolbox/backend/sampling_derivatives.py:2808` | `assemble_tucker_tangent_variation_jets` | `DEFAULT_CHUNK_SIZE` | name
- `t3toolbox/backend/sampling_derivatives.py:2926` | `assemble_tt_tangent_variation_jets` | `DEFAULT_CHUNK_SIZE` | name
- `t3toolbox/backend/sampling_derivatives.py:2964` | `t3_probe_corewise_derivatives_hessian_from_sweep` | `DEFAULT_CHUNK_SIZE` | name
- `t3toolbox/backend/sampling_derivatives.py:3028` | `t3_probe_corewise_derivatives_hessian` | `DEFAULT_CHUNK_SIZE` | name
- `t3toolbox/backend/ut3_sampling.py:359` | `ut3_probe_corewise_derivatives_transpose` | `DEFAULT_CHUNK_SIZE` | attr (`sampling_derivatives`)
- `t3toolbox/backend/ut3_sampling.py:387` | `ut3_probe_corewise_derivatives_hessian` | `DEFAULT_CHUNK_SIZE` | attr (`sampling_derivatives`)
- `t3toolbox/backend/ut3_sampling.py:449` | `ut3_probe_corewise_derivatives_hessian_from_sweep` | `DEFAULT_CHUNK_SIZE` | attr (`sampling_derivatives`)
Per-file counts: `t3toolbox/backend/sampling_derivatives.py` 20, `t3toolbox/backend/ut3_sampling.py` 3

**tests** (every reference, ast + text; `file:line | enclosing | name | kind (receiver)`; definition sites excluded):

- `tests/test_chunk_size_estimator.py:4` | `<module>` | `estimate_chunk_size` | docstring
- `tests/test_chunk_size_estimator.py:4` | `<module>` | `max_chunk_size_within` | docstring
- `tests/test_chunk_size_estimator.py:50` | `TestChunkSizeEstimator.test_balanced_tracks_the_jet_floor` | `estimate_chunk_size` | attr (`pd`)
- `tests/test_chunk_size_estimator.py:60` | `TestChunkSizeEstimator.test_balanced_tracks_the_jet_floor` | `estimate_chunk_size` | attr (`pd`)
- `tests/test_chunk_size_estimator.py:68` | `TestChunkSizeEstimator.test_budget_is_respected` | `max_chunk_size_within` | attr (`pd`)
- `tests/test_chunk_size_estimator.py:77` | `TestChunkSizeEstimator.test_sharding_sizes_the_local_shard` | `estimate_chunk_size` | attr (`pd`)
- `tests/test_chunk_size_estimator.py:78` | `TestChunkSizeEstimator.test_sharding_sizes_the_local_shard` | `estimate_chunk_size` | attr (`pd`)
- `tests/test_chunk_size_estimator.py:83` | `TestChunkSizeEstimator.test_small_problem_is_dense` | `estimate_chunk_size` | attr (`pd`)
- `tests/test_chunk_size_estimator.py:103` | `TestChunkSizeEstimator.test_returns_plain_int` | `estimate_chunk_size` | attr (`pd`)
- `tests/test_chunk_size_estimator.py:114` | `TestChunkSizeEstimator.test_frame_stack_shrinks_the_budget_chunk` | `max_chunk_size_within` | attr (`pd`)
- `tests/test_chunk_size_estimator.py:117` | `TestChunkSizeEstimator.test_frame_stack_shrinks_the_budget_chunk` | `max_chunk_size_within` | attr (`pd`)
- `tests/test_chunk_size_estimator.py:130` | `TestChunkSizeEstimator.test_frame_stack_leaves_the_default_untouched` | `estimate_chunk_size` | attr (`pd`)
- `tests/test_chunk_size_estimator.py:131` | `TestChunkSizeEstimator.test_frame_stack_leaves_the_default_untouched` | `estimate_chunk_size` | attr (`pd`)
- `tests/test_chunk_size_estimator.py:138` | `TestChunkSizeEstimator.test_balanced_is_only_mildly_sensitive_to_the_frame_stack` | `estimate_chunk_size` | attr (`pd`)
- `tests/test_chunk_size_estimator.py:139` | `TestChunkSizeEstimator.test_balanced_is_only_mildly_sensitive_to_the_frame_stack` | `estimate_chunk_size` | attr (`pd`)
- `tests/test_chunk_size_estimator.py:156` | `TestChunkSizeEstimator.test_hessian_also_bounds_the_two_hole` | `estimate_chunk_size` | attr (`pd`)
- `tests/test_chunk_size_estimator.py:157` | `TestChunkSizeEstimator.test_hessian_also_bounds_the_two_hole` | `estimate_chunk_size` | attr (`pd`)
- `tests/test_chunk_size_estimator.py:158` | `TestChunkSizeEstimator.test_hessian_also_bounds_the_two_hole` | `estimate_chunk_size` | attr (`pd`)
- `tests/test_chunk_size_estimator.py:159` | `TestChunkSizeEstimator.test_hessian_also_bounds_the_two_hole` | `estimate_chunk_size` | attr (`pd`)
- `tests/test_chunk_size_estimator.py:160` | `TestChunkSizeEstimator.test_hessian_also_bounds_the_two_hole` | `estimate_chunk_size` | attr (`pd`)
- `tests/test_chunk_size_estimator.py:166` | `TestChunkSizeEstimator.test_hessian_also_bounds_the_two_hole` | `max_chunk_size_within` | attr (`pd`)
- `tests/test_chunk_size_estimator.py:167` | `TestChunkSizeEstimator.test_hessian_also_bounds_the_two_hole` | `max_chunk_size_within` | attr (`pd`)
- `tests/test_chunk_size_estimator.py:168` | `TestChunkSizeEstimator.test_hessian_also_bounds_the_two_hole` | `max_chunk_size_within` | attr (`pd`)
- `tests/test_chunk_size_estimator.py:183` | `TestChunkSizeEstimator.test_hessian_mode_is_the_solver_vocabulary` | `estimate_chunk_size` | attr (`pd`)
- `tests/test_chunk_size_estimator.py:185` | `TestChunkSizeEstimator.test_hessian_mode_is_the_solver_vocabulary` | `max_chunk_size_within` | attr (`pd`)
- `tests/test_chunk_size_estimator.py:192` | `TestBudgetHoldsInTheCompiledProgram` | `max_chunk_size_within` | docstring
- `tests/test_chunk_size_estimator.py:213` | `TestBudgetHoldsInTheCompiledProgram.test_chunk_dependent_peak_within_the_budget` | `max_chunk_size_within` | attr (`pd`)
Per-file counts: `tests/test_chunk_size_estimator.py` 27

**examples** (every reference, ast + text; `file:line | enclosing | name | kind (receiver)`; definition sites excluded):

Per-file counts: none

**docs pages**:

- `docs/contributor/curvature_internals.md:349` | `estimate_chunk_size` | prose
- `docs/contributor/deferred_and_rejected.md:260` | `max_chunk_size_within` | prose
- `docs/contributor/verification_records.md:87` | `estimate_chunk_size` | prose
- `docs/glossary.md:404` | `estimate_chunk_size` | prose
- `docs/glossary.md:404` | `max_chunk_size_within` | prose
- `docs/release_notes.md:193` | `estimate_chunk_size` | prose
- `docs/release_notes.md:193` | `max_chunk_size_within` | prose

Also the plan text says `chunk_size` (frontend keyword) is deleted at P16: library occurrences of the identifier `chunk_size` (ast Name/Attribute/kwarg/param, any receiver): `t3toolbox/backend/fitting.py` 5, `t3toolbox/backend/sampling_derivatives.py` 39, `t3toolbox/backend/uniform_fitting.py` 7, `t3toolbox/backend/ut3_sampling.py` 9, `t3toolbox/backend/utv_sampling.py` 6, `t3toolbox/optimizers.py` 3

### (a2) Private top-level helpers of the six sampling/fitting modules that only deleted units reference (candidate orphans)

Method: for each module, every top-level def/class/assign not in the (a) deleted list; ast references (Name/Attribute attr) to its name from other top-level units of ANY library file (excluding itself); if every referencing unit is a deleted unit (or the sampling_derivatives module) it is a candidate orphan; if there is no reference at all it is "unreferenced". Public names exported in `__all__` that are kept are skipped. This is mechanical: names referenced via strings/getattr are not seen.

- `t3toolbox/backend/probing.py`: candidate orphans (19): `_xi_step`, `_mu_step`, `_eta_step`, `_assemble_z_step`, `_sigma_sweep_step`, `_deta_step`, `_assemble_tangent_z_step`, `_sigma_hat_step`, `_deta_tilde_step`, `_tau_tilde_step`, `_dxi_tilde_step`, `_assemble_tucker_variations_step_summed`, `_assemble_tucker_variations_step_unsummed`, `_assemble_tt_variations_step_summed`, `_assemble_tt_variations_step_unsummed`, `_deta_tilde_tangent_step`, `_tau_tilde_tangent_step`, `_tucker_tangent_variations_step`, `_tt_tangent_variations_step`; unreferenced anywhere in library (0): none
- `t3toolbox/backend/apply.py`: candidate orphans (3): `_apply_from_xis`, `_apply_transpose_adjoint`, `_apply_corewise_hessian_adjoint`; unreferenced anywhere in library (1): `_t3_apply_step`
- `t3toolbox/backend/entries.py`: candidate orphans (2): `_entry_xis`, `_onehot_vectors`; unreferenced anywhere in library (1): `_t3_entries_step`
- `t3toolbox/backend/ut3_sampling.py`: candidate orphans (2): `_remasked_core_gradient`, `_masked_point_and_directions`; unreferenced anywhere in library (0): none
- `t3toolbox/backend/utv_sampling.py`: candidate orphans (1): `_mask_once`; unreferenced anywhere in library (0): none
- `t3toolbox/backend/fitting.py`: candidate orphans (1): `_prod_w`; unreferenced anywhere in library (1): `_BATCH_FIELDS`
- `t3toolbox/backend/uniform_fitting.py`: candidate orphans (2): `_frame_theta`, `_fg`; unreferenced anywhere in library (0): none


## M2. The weight

### (a) Library sites that read or pass the engine `weight` / a batch `.weight`

Method: `ast` walk of each file for any Name / Attribute / arg / keyword / def-name whose identifier CONTAINS `weight` (so `weight`, `.weight`, `weight=`, `blocks_to_weight`, `_apply_weight`, `weight_taylor` ...), reduced to one row per source line: `file:line | enclosing function | identifiers on the line (a=param, k=keyword, .=attribute, d=def name) | the line`. Docstrings, comments and doctest lines are NOT included here (counts below). `ut3_operations.py` rows are the unrelated `T3Weights` machinery (`weights`, `weighted_tucker` ...), listed because the file was named; they do not touch the engine weight.

**`t3toolbox/backend/sampling_variables.py`**: 7 code lines; (not listed: 18 docstring-prose lines, 0 doctest lines, 2 comment lines mention "weight")

- `t3toolbox/backend/sampling_variables.py:1587` | `computed_names` | a:weight | `weight:            typ.Optional[NDArray] = None,     # not read`
- `t3toolbox/backend/sampling_variables.py:1690` | `sampling_variables` | a:weight | `weight:            typ.Optional[typ.Any] = None,     # omega, of the type of z likewise (no K); None = ones`
- `t3toolbox/backend/sampling_variables.py:1861` | `sampling_variables` | weight weight_given | `y_given, weight_given = y, weight                   # as given, for their check (before any packing)`
- `t3toolbox/backend/sampling_variables.py:1873` | `sampling_variables` | weight | `weight = None if weight is None else ut3_operations._pack_data_space(weight, N)`
- `t3toolbox/backend/sampling_variables.py:1879` | `sampling_variables` | weight_given | `for name, given in (('y', y_given), ('the weight', weight_given)):`
- `t3toolbox/backend/sampling_variables.py:1893` | `sampling_variables` | weight | `use_jax = tree_contains_jax((frame, sample, y, weight, variations, core_directions, Vc,`
- `t3toolbox/backend/sampling_variables.py:1898` | `sampling_variables` | k:weight weight | `inputs = dict(stores, y=y, weight=weight, Vc=Vc, V=V, H=H, D_dot=D_dot, L_dot=L_dot,`

**`t3toolbox/backend/sampling_steps.py`**: 26 code lines; (not listed: 7 docstring-prose lines, 0 doctest lines, 6 comment lines mention "weight")

- `t3toolbox/backend/sampling_steps.py:918` | `_modes` | a:weight | `weight:  typ.Any,                   # omega, of the type of z with [K] placed (_with_K): probe len=d or (d,)+...; else one array; None = one`
- `t3toolbox/backend/sampling_steps.py:922` | `_modes` | weight | `return [(xnp.asarray(value[i]), None if weight is None else weight[i], 1) for i in range(len(value))]`
- `t3toolbox/backend/sampling_steps.py:923` | `_modes` | weight | `return [(xnp.asarray(value), weight, 0)]`
- `t3toolbox/backend/sampling_steps.py:927` | `_with_K` | a:weight | `weight:  typ.Any,                   # omega, of the type of z with no K: probe len=d or (d,)+F+W+C+(e,); else F+W+C or W+C`
- `t3toolbox/backend/sampling_steps.py:935` | `_with_K` | weight | `if weight is None or lens.K == 0:`
- `t3toolbox/backend/sampling_steps.py:936` | `_with_K` | weight | `return weight`
- `t3toolbox/backend/sampling_steps.py:943` | `_with_K` | weight | `if kind == 'probe' and isinstance(weight, (tuple, list)):`
- `t3toolbox/backend/sampling_steps.py:944` | `_with_K` | weight | `return tuple(placed(omega, 0) for omega in weight)`
- `t3toolbox/backend/sampling_steps.py:945` | `_with_K` | weight | `return placed(weight, 1 if kind == 'probe' else 0)`
- `t3toolbox/backend/sampling_steps.py:948` | `_weighted_sum` | d:_weighted_sum | `def _weighted_sum(`
- `t3toolbox/backend/sampling_steps.py:951` | `_weighted_sum` | a:weight | `weight:  typ.Any,                   # placed against value (with K where value has it); None = ones`
- `t3toolbox/backend/sampling_steps.py:958` | `_weighted_sum` | weight | `for array, omega, edge in _modes(kind, value, weight, xnp):`
- `t3toolbox/backend/sampling_steps.py:959` | `_weighted_sum` | weighted | `weighted = jets.data_metric(array, omega, None)`
- `t3toolbox/backend/sampling_steps.py:960` | `_weighted_sum` | weighted | `summed = xnp.sum(array * weighted, axis=tuple(range(n_sum)) + tuple(range(array.ndim - edge, array.ndim)))`
- `t3toolbox/backend/sampling_steps.py:995` | `Jp_norm_sq_step` | a:weight | `weight:  typ.Any,                   # omega, of the type of z (no K: placed here); None = ones`
- `t3toolbox/backend/sampling_steps.py:1005` | `Jp_norm_sq_step` | weight | `xnp, _, _ = get_backend(False, tree_contains_jax((Jp, weight)))`
- `t3toolbox/backend/sampling_steps.py:1006` | `Jp_norm_sq_step` | _weighted_sum weight | `return xnp.asarray(_weighted_sum(kind, Jp, _with_K(weight, kind, lens), lens.P + lens.W, xnp))`
- `t3toolbox/backend/sampling_steps.py:1011` | `w_tilde_step` | a:weight | `weight:  typ.Any,                   # omega, of the type of z (no K: placed here); None = ones`
- `t3toolbox/backend/sampling_steps.py:1022` | `w_tilde_step` | weight | `xnp, _, _ = get_backend(False, tree_contains_jax((r, weight)))`
- `t3toolbox/backend/sampling_steps.py:1024` | `w_tilde_step` | weight | `for array, omega, _ in _modes(kind, r, _with_K(weight, kind, lens), xnp)]`
- `t3toolbox/backend/sampling_steps.py:1068` | `f_step` | a:weight | `weight:  typ.Any,                   # omega, of the type of z; None = ones`
- `t3toolbox/backend/sampling_steps.py:1077` | `f_step` | weight | `xnp, _, _ = get_backend(False, tree_contains_jax((r, weight)))`
- `t3toolbox/backend/sampling_steps.py:1078` | `f_step` | _weighted_sum weight | `total = _weighted_sum(kind, r, weight, lens.P + lens.W, xnp)`
- `t3toolbox/backend/sampling_steps.py:1085` | `r_block_sumsq_step` | a:weight | `weight:  typ.Any,                   # omega, of the type of z; None = ones`
- `t3toolbox/backend/sampling_steps.py:1095` | `r_block_sumsq_step` | weight | `xnp, _, _ = get_backend(False, tree_contains_jax((r, weight)))`
- `t3toolbox/backend/sampling_steps.py:1098` | `r_block_sumsq_step` | weight | `for r_FWCo, omega, _ in _modes(kind, r, weight, xnp)]`

**`t3toolbox/backend/fitting.py`**: 39 code lines; (not listed: 41 docstring-prose lines, 6 doctest lines, 18 comment lines mention "weight")

- `t3toolbox/backend/fitting.py:208` | `_weight_matrix` | d:_weight_matrix | `def _weight_matrix(`
- `t3toolbox/backend/fitting.py:209` | `_weight_matrix` | a:weight | `weight:  typ.Optional[typ.Any],    # None, or array / (nested) sequence -- the raw ω input`
- `t3toolbox/backend/fitting.py:226` | `_weight_matrix` | weight | `if weight is None:`
- `t3toolbox/backend/fitting.py:228` | `_weight_matrix` | weight | `w = np.array(weight, dtype=float)     # COPY, never a view: the canonical matrix is part of the`
- `t3toolbox/backend/fitting.py:246` | `_make_weight` | d:_make_weight | `def _make_weight(`
- `t3toolbox/backend/fitting.py:450` | `SamplingKind._apply_weight` | d:_apply_weight | `def _apply_weight(self):`
- `t3toolbox/backend/fitting.py:457` | `SamplingKind._apply_weight` | _make_weight | `return _make_weight(getattr(self, 'weight', None), self._order_axis, self._mode_axis)`
- `t3toolbox/backend/fitting.py:468` | `ScalarOutputKind.sumsq` | ._apply_weight | `return sumsq_over_samples(self._apply_weight(out, 1), n_w + (1 if self.has_order else 0))`
- `t3toolbox/backend/fitting.py:482` | `ProbeOutputKind.sumsq` | ._apply_weight | `return sumsq_over_probes(self._apply_weight(out, 1), n_w + (1 if self.has_order else 0))`
- `t3toolbox/backend/fitting.py:595` | `ProbeKind` | weight | `weight:  typ.Optional[typ.Any] = None   # per-mode omega, (d,) / (d,1); None = unweighted`
- `t3toolbox/backend/fitting.py:599` | `ProbeKind.__post_init__` | .weight _weight_matrix | `object.__setattr__(self, 'weight', _weight_matrix(self.weight, 0, self._bare_binds_to))`
- `t3toolbox/backend/fitting.py:608` | `ProbeKind.transpose` | ._apply_weight | `return probing.tv_probe_transpose_from_sweep(self._apply_weight(r, 2), ww, frame, sweep,`
- `t3toolbox/backend/fitting.py:620` | `ProbeKind.corewise_hessian` | ._apply_weight | `return probing.t3_probe_corewise_hessian_from_sweep(self._apply_weight(r, 2), ww, base_point,`
- `t3toolbox/backend/fitting.py:634` | `_DerivativesKind` | weight | `weight:  typ.Optional[typ.Any] = None   # omega; see the concrete kinds for its shape`
- `t3toolbox/backend/fitting.py:637` | `_DerivativesKind.__post_init__` | .weight _weight_matrix | `object.__setattr__(self, 'weight', _weight_matrix(self.weight, self.order, self._bare_binds_to))`
- `t3toolbox/backend/fitting.py:655` | `ApplyDerivativesKind.transpose` | ._apply_weight | `self._apply_weight(r, 2), s[0], s[1], frame, sweep, self.order, sum_over_probes=True)`
- `t3toolbox/backend/fitting.py:667` | `ApplyDerivativesKind.corewise_hessian` | ._apply_weight | `self._apply_weight(r, 2), s[0], s[1], base_point, core_directions, self.order, sweep)`
- `t3toolbox/backend/fitting.py:684` | `EntriesDerivativesKind.transpose` | ._apply_weight | `self._apply_weight(r, 2), s[0], s[1], frame, sweep, self.order, sum_over_probes=True)`
- `t3toolbox/backend/fitting.py:695` | `EntriesDerivativesKind.corewise_hessian` | ._apply_weight | `self._apply_weight(r, 2), s[0], s[1], base_point, core_directions, self.order, sweep)`
- `t3toolbox/backend/fitting.py:716` | `ProbeDerivativesKind.transpose` | ._apply_weight | `self._apply_weight(r, 2), s[0], s[1], frame, sweep, self.order,`
- `t3toolbox/backend/fitting.py:729` | `ProbeDerivativesKind.corewise_hessian` | ._apply_weight | `self._apply_weight(r, 2), s[0], s[1], base_point, core_directions, self.order, sweep,`
- `t3toolbox/backend/fitting.py:776` | `LeastSquaresBatch` | weight | `weight:           typ.Any = None                             # omega, of the type of z likewise; None = ones`
- `t3toolbox/backend/fitting.py:806` | `LeastSquaresBatch.data_metric` | .weight | `return jets.data_metric(r, self.weight, self.minibatch_scale)`
- `t3toolbox/backend/fitting.py:875` | `_y_times_metric_y` | a:weight | `weight:  typ.Any,                   # omega, of the type of z likewise; None = ones`
- `t3toolbox/backend/fitting.py:884` | `_y_times_metric_y` | weight | `xnp, _, _ = get_backend(False, tree_contains_jax((y, weight)))`
- `t3toolbox/backend/fitting.py:890` | `_y_times_metric_y` | weight | `for y_i, omega_i, z_shape in zip(_arrays_of(kind, y, shape), _arrays_of(kind, weight, shape), targets):`
- `t3toolbox/backend/fitting.py:900` | `_y_metric_y` | a:weight | `weight:  typ.Any,                   # omega, of the type of z likewise; None = ones`
- `t3toolbox/backend/fitting.py:911` | `_y_metric_y` | weight | `xnp, _, _ = get_backend(False, tree_contains_jax((y, weight)))`
- `t3toolbox/backend/fitting.py:913` | `_y_metric_y` | weight | `for product in _y_times_metric_y(kind, order, y, weight, W, C, shape):`
- `t3toolbox/backend/fitting.py:924` | `_y_block_metric_y` | a:weight | `weight:  typ.Any,                   # omega, of the type of z likewise; None = ones`
- `t3toolbox/backend/fitting.py:933` | `_y_block_metric_y` | weight | `xnp, _, _ = get_backend(False, tree_contains_jax((y, weight)))`
- `t3toolbox/backend/fitting.py:935` | `_y_block_metric_y` | weight | `for product in _y_times_metric_y(kind, order, y, weight, W, C, shape)]`
- `t3toolbox/backend/fitting.py:1104` | `LeastSquaresOracle.query` | .weight k:weight | `args = dict(order=self.order, y=batch.y, weight=batch.weight, variations=variations,`
- `t3toolbox/backend/fitting.py:1116` | `LeastSquaresOracle.query` | .weight | `values['y_block_sumsq'] = _y_block_metric_y(self.kind, self.order, batch.y, batch.weight, self._W(batch),`
- `t3toolbox/backend/fitting.py:1124` | `LeastSquaresOracle.query` | .weight | `xnp, _, _ = get_backend(False, tree_contains_jax((f, batch.y, batch.weight)))`
- `t3toolbox/backend/fitting.py:1125` | `LeastSquaresOracle.query` | .weight | `y_norm_sq = _y_metric_y(self.kind, self.order, batch.y, batch.weight, self._W(batch),`
- `t3toolbox/backend/fitting.py:1164` | `LeastSquaresOracle.check_batch` | .check_weight .weight | `jets.check_weight(self.kind, self.order, batch.sample, batch.weight, stack_shape)`
- `t3toolbox/backend/fitting.py:1236` | `LeastSquaresOracle.take` | .weight weight | `weight = _take_data_space(self.kind, m, W, batch.weight, indices, 'the weight')`
- `t3toolbox/backend/fitting.py:1240` | `LeastSquaresOracle.take` | weight | `return LeastSquaresBatch(sample, y, weight, scale, indices)`

**`t3toolbox/backend/uniform_fitting.py`**: 16 code lines; (not listed: 8 docstring-prose lines, 1 doctest lines, 3 comment lines mention "weight")

- `t3toolbox/backend/uniform_fitting.py:205` | `UniformProbeKind.transpose` | ._apply_weight | `v_sc = utv_sampling.utv_probe_transpose_from_sweep(self._apply_weight(r, 2), sweep,`
- `t3toolbox/backend/uniform_fitting.py:221` | `UniformProbeKind.corewise_hessian` | ._apply_weight | `self._apply_weight(r, 2), _frame_theta(frame_data, base_point), core_directions, sweep)`
- `t3toolbox/backend/uniform_fitting.py:237` | `UniformApplyDerivativesKind.transpose` | ._apply_weight | `self._apply_weight(r, 2), sweep, self.order, sum_over_probes=True)`
- `t3toolbox/backend/uniform_fitting.py:252` | `UniformApplyDerivativesKind.corewise_hessian` | ._apply_weight | `self._apply_weight(r, 2), _frame_theta(frame_data, base_point), core_directions, self.order, sweep)`
- `t3toolbox/backend/uniform_fitting.py:268` | `UniformEntriesDerivativesKind.transpose` | ._apply_weight | `self._apply_weight(r, 2), sweep, self.order, sum_over_probes=True)`
- `t3toolbox/backend/uniform_fitting.py:282` | `UniformEntriesDerivativesKind.corewise_hessian` | ._apply_weight | `self._apply_weight(r, 2), _frame_theta(frame_data, base_point), core_directions, self.order, sweep)`
- `t3toolbox/backend/uniform_fitting.py:305` | `UniformProbeDerivativesKind.transpose` | ._apply_weight | `self._apply_weight(r, 2), sweep, self.order, sum_over_probes=True,`
- `t3toolbox/backend/uniform_fitting.py:323` | `UniformProbeDerivativesKind.corewise_hessian` | ._apply_weight | `self._apply_weight(r, 2), _frame_theta(frame_data, base_point), core_directions, self.order, sweep,`
- `t3toolbox/backend/uniform_fitting.py:336` | `uniform_sampling_kind` | a:weight | `weight:   typ.Optional[typ.Any] = None,  # per-mode weight omega (probe only); apply/entries take none`
- `t3toolbox/backend/uniform_fitting.py:342` | `uniform_sampling_kind` | k:weight weight | `return UniformProbeKind.from_point(x0_data, weight=weight)`
- `t3toolbox/backend/uniform_fitting.py:344` | `uniform_sampling_kind` | weight | `if weight is not None:`
- `t3toolbox/backend/uniform_fitting.py:355` | `uniform_derivatives_kind` | a:weight | `weight:     typ.Optional[typ.Sequence[float]] = None,`
- `t3toolbox/backend/uniform_fitting.py:364` | `uniform_derivatives_kind` | k:weight weight | `return cls.from_point(x0_data, order=order, weight=weight, chunk_size=chunk_size)`
- `t3toolbox/backend/uniform_fitting.py:365` | `uniform_derivatives_kind` | k:weight weight | `return cls.from_point(x0_data, order=order, weight=weight)`
- `t3toolbox/backend/uniform_fitting.py:454` | `pack_batch` | .weight weight | `weight = batch.weight if batch.weight is None else pack_data(kind, order, batch.weight, N)`
- `t3toolbox/backend/uniform_fitting.py:455` | `pack_batch` | k:weight weight | `return dc.replace(batch, sample=pack_sample(kind, order, batch.sample, N), y=y, weight=weight)`

**`t3toolbox/backend/jets.py`**: 62 code lines; (not listed: 142 docstring-prose lines, 60 doctest lines, 12 comment lines mention "weight")

- `t3toolbox/backend/jets.py:268` | `_taylor_weight` | d:_taylor_weight | `def _taylor_weight(`
- `t3toolbox/backend/jets.py:269` | `_taylor_weight` | a:weight | `weight:  typ.Any,           # one array of a weight of the type of z: jet axes after 'lead' axes, each 1 or F's`
- `t3toolbox/backend/jets.py:274` | `_taylor_weight` | weight weight_X | `weight_X = xnp.asarray(weight)`
- `t3toolbox/backend/jets.py:276` | `_taylor_weight` | weight_X | `return weight_X`
- `t3toolbox/backend/jets.py:278` | `_taylor_weight` | weight_X | `if weight_X.ndim < lead + m:`
- `t3toolbox/backend/jets.py:281` | `_taylor_weight` | weight_X | `'mode-leading probe array)' % (weight_X.ndim, m, lead))`
- `t3toolbox/backend/jets.py:282` | `_taylor_weight` | weight_X | `factors_F = xnp.asarray(_factorials(order), dtype=_lowest_precision_dtype(xnp, weight_X))`
- `t3toolbox/backend/jets.py:283` | `_taylor_weight` | weight_X | `return weight_X * factors_F.reshape((1,) * lead + factors_F.shape + (1,) * (weight_X.ndim - lead - m))`
- `t3toolbox/backend/jets.py:293` | `_mode_weights` | d:_mode_weights | `def _mode_weights(`
- `t3toolbox/backend/jets.py:295` | `_mode_weights` | a:weight | `weight:  typ.Optional[typ.Any],     # omega, of the type of z: probe len=d or a mode-leading array; else one array`
- `t3toolbox/backend/jets.py:302` | `_mode_weights` | weight | `if weight is None:`
- `t3toolbox/backend/jets.py:305` | `_mode_weights` | weight | `if isinstance(weight, (tuple, list)):`
- `t3toolbox/backend/jets.py:307` | `_mode_weights` | weight | `% (name, kind, type(weight).__name__))`
- `t3toolbox/backend/jets.py:308` | `_mode_weights` | weight | `return [weight]`
- `t3toolbox/backend/jets.py:309` | `_mode_weights` | weight | `if isinstance(weight, (tuple, list)):`
- `t3toolbox/backend/jets.py:310` | `_mode_weights` | weight | `if len(weight) != d:`
- `t3toolbox/backend/jets.py:311` | `_mode_weights` | weight | `raise ValueError('%s: the value has %d arrays (modes) but the weight has %d' % (name, d, len(weight)))`
- `t3toolbox/backend/jets.py:312` | `_mode_weights` | weight | `return list(weight)`
- `t3toolbox/backend/jets.py:313` | `_mode_weights` | weight | `if np.ndim(weight) < 1 or np.shape(weight)[0] != d:`
- `t3toolbox/backend/jets.py:316` | `_mode_weights` | weight | `% (name, d, tuple(np.shape(weight))))`
- `t3toolbox/backend/jets.py:317` | `_mode_weights` | weight | `return [weight[i] for i in range(d)]`
- `t3toolbox/backend/jets.py:328` | `jet_to_dual_jet` | a:weight | `weight:  typ.Optional[typ.Any],    # omega, of the type of z (each axis 1 or the jet's size, no K); None = ones`
- `t3toolbox/backend/jets.py:445` | `jet_to_dual_jet` | weight | `xnp, _, _ = get_backend(False, tree_contains_jax((jet, weight)))        # once, over all the inputs`
- `t3toolbox/backend/jets.py:448` | `jet_to_dual_jet` | _mode_weights weight | `for i, (jet_FX, omega_FX) in enumerate(zip(arrays, _mode_weights(kind, weight, len(arrays), 'jet_to_dual_jet'))):`
- `t3toolbox/backend/jets.py:460` | `dual_jet_to_jet` | a:weight | `weight:    typ.Optional[typ.Any],  # omega, of the type of z (each axis 1 or the dual jet's size); None = ones`
- `t3toolbox/backend/jets.py:525` | `dual_jet_to_jet` | weight | `use_jax = tree_contains_jax((dual_jet, weight))                         # once, over all the inputs`
- `t3toolbox/backend/jets.py:529` | `dual_jet_to_jet` | _mode_weights weight | `for i, (dual_FX, omega_FX) in enumerate(zip(arrays, _mode_weights(kind, weight, len(arrays), 'dual_jet_to_jet'))):`
- `t3toolbox/backend/jets.py:581` | `data_metric` | a:weight | `weight:           typ.Optional[typ.Any],    # omega, of the type of z: each axis 1 or r's size; None = ones`
- `t3toolbox/backend/jets.py:725` | `data_metric` | weight | `xnp, _, _ = get_backend(False, tree_contains_jax((r, weight, minibatch_scale)))   # once, over all the inputs`
- `t3toolbox/backend/jets.py:733` | `data_metric` | weight | `if weight is None:`
- `t3toolbox/backend/jets.py:734` | `data_metric` | weights | `weights = [None] * len(arrays)`
- `t3toolbox/backend/jets.py:735` | `data_metric` | weight | `elif isinstance(weight, (tuple, list)):`
- `t3toolbox/backend/jets.py:736` | `data_metric` | weight | `if len(weight) != len(arrays):`
- `t3toolbox/backend/jets.py:738` | `data_metric` | weight | `% (len(arrays), len(weight)))`
- `t3toolbox/backend/jets.py:739` | `data_metric` | weight weights | `weights = list(weight)`
- `t3toolbox/backend/jets.py:741` | `data_metric` | weight | `if np.ndim(weight) < 1 or np.shape(weight)[0] != len(arrays):`
- `t3toolbox/backend/jets.py:744` | `data_metric` | weight | `% (len(arrays), len(arrays), np.shape(weight)))`
- `t3toolbox/backend/jets.py:745` | `data_metric` | weight weights | `weights = [weight[i] for i in range(len(arrays))]`
- `t3toolbox/backend/jets.py:747` | `data_metric` | weight | `if isinstance(weight, (tuple, list)):`
- `t3toolbox/backend/jets.py:751` | `data_metric` | weight weights | `arrays, weights = [r], [weight]`
- `t3toolbox/backend/jets.py:753` | `data_metric` | weights | `for i, (r_X, omega_X) in enumerate(zip(arrays, weights)):`
- `t3toolbox/backend/jets.py:767` | `blocks_to_weight` | d:blocks_to_weight | `def blocks_to_weight(`
- `t3toolbox/backend/jets.py:997` | `derivative_weight_to_taylor_weight` | d:derivative_weight_to_taylor_weight | `def derivative_weight_to_taylor_weight(`
- `t3toolbox/backend/jets.py:1000` | `derivative_weight_to_taylor_weight` | a:weight | `weight:  typ.Any,                  # omega on derivative residuals, of the type of z: probe len=d or (d,)+...; else one array`
- `t3toolbox/backend/jets.py:1065` | `derivative_weight_to_taylor_weight` | weight | `xnp, _, _ = get_backend(False, tree_contains_jax(weight))`
- `t3toolbox/backend/jets.py:1066` | `derivative_weight_to_taylor_weight` | _taylor_weights weight | `return _taylor_weights(kind, order, weight, xnp)`
- `t3toolbox/backend/jets.py:1069` | `_taylor_weights` | d:_taylor_weights | `def _taylor_weights(`
- `t3toolbox/backend/jets.py:1072` | `_taylor_weights` | a:weight | `weight:  typ.Any,                   # omega on derivative residuals, of the type of z`
- `t3toolbox/backend/jets.py:1075` | `_taylor_weights` | weight | `if kind == 'probe' and isinstance(weight, (tuple, list)):`
- `t3toolbox/backend/jets.py:1076` | `_taylor_weights` | _taylor_weight weight | `return tuple(_taylor_weight(omega, order, 0, xnp) for omega in weight)`
- `t3toolbox/backend/jets.py:1077` | `_taylor_weights` | _taylor_weight weight | `return _taylor_weight(weight, order, 1 if kind == 'probe' else 0, xnp)`
- `t3toolbox/backend/jets.py:1085` | `prepare_derivative_data` | a:weight | `weight:      typ.Optional[typ.Any] = None,     # omega on derivative residuals, of the type of z; None = ones on the derivatives`
- `t3toolbox/backend/jets.py:1171` | `prepare_derivative_data` | weight | `xnp, _, _ = get_backend(False, tree_contains_jax((y, weight, validation)))     # once, over all the inputs`
- `t3toolbox/backend/jets.py:1174` | `prepare_derivative_data` | weight | `if weight is None:`
- `t3toolbox/backend/jets.py:1178` | `prepare_derivative_data` | blocks_to_weight weight_taylor | `weight_taylor = blocks_to_weight(kind, order, blocks, y_taylor)`
- `t3toolbox/backend/jets.py:1180` | `prepare_derivative_data` | _mode_weights weight | `for i, (y_X, omega_X) in enumerate(zip(arrays, _mode_weights(kind, weight, len(arrays),`
- `t3toolbox/backend/jets.py:1183` | `prepare_derivative_data` | _taylor_weights weight weight_taylor | `weight_taylor = _taylor_weights(kind, order, weight, xnp)`
- `t3toolbox/backend/jets.py:1185` | `prepare_derivative_data` | weight_taylor | `return y_taylor, weight_taylor, validation_taylor`
- `t3toolbox/backend/jets.py:1657` | `check_weight` | d:check_weight | `def check_weight(`
- `t3toolbox/backend/jets.py:1661` | `check_weight` | a:weight | `weight:       typ.Optional[typ.Any],    # omega, of the type of z (each axis 1 or z's size), or None`
- `t3toolbox/backend/jets.py:1733` | `check_weight` | weight | `if weight is None:`
- `t3toolbox/backend/jets.py:1735` | `check_weight` | weight | `_check_of_type_of_z(kind, order, sample, weight, stack_shape, 'the weight')`

**`t3toolbox/backend/data_preparation.py`**: 1 code lines; (not listed: 13 docstring-prose lines, 7 doctest lines, 0 comment lines mention "weight")

- `t3toolbox/backend/data_preparation.py:199` | `equalizing_weight` | d:equalizing_weight | `def equalizing_weight(`

**`t3toolbox/backend/optimizers.py`**: 0 code lines; (not listed: 2 docstring-prose lines, 0 doctest lines, 1 comment lines mention "weight")


**`t3toolbox/backend/optimizer_display.py`**: 0 code lines; (not listed: 7 docstring-prose lines, 0 doctest lines, 0 comment lines mention "weight")


**`t3toolbox/backend/sampling_definitions.py`**: 67 code lines; (not listed: 27 docstring-prose lines, 22 doctest lines, 2 comment lines mention "weight")

- `t3toolbox/backend/sampling_definitions.py:700` | `_weight_squared` | d:_weight_squared | `def _weight_squared(`
- `t3toolbox/backend/sampling_definitions.py:702` | `_weight_squared` | a:weight | `weight:  typ.Any,                   # of the type of z: each axis 1 or z's size; None = ones`
- `t3toolbox/backend/sampling_definitions.py:708` | `_weight_squared` | weight | `if weight is None:`
- `t3toolbox/backend/sampling_definitions.py:710` | `_weight_squared` | weight | `xnp, _, _ = get_backend(False, tree_contains_jax((weight,)))`
- `t3toolbox/backend/sampling_definitions.py:711` | `_weight_squared` | weight | `omega_X = _modes_of_data(S, weight, 'the weight')`
- `t3toolbox/backend/sampling_definitions.py:726` | `_dual_jet` | a:weight | `weight:  typ.Any,                       # of the type of z: each axis 1 or z's size; None = ones`
- `t3toolbox/backend/sampling_definitions.py:729` | `_dual_jet` | _weight_squared weight | `return _map_type_of_z(S, lambda i, jet_TX: _turn(S, _weight_squared(S, weight, i, jet_TX.shape) * jet_TX), jet)`
- `t3toolbox/backend/sampling_definitions.py:736` | `_the_seed` | a:weight | `weight:   typ.Optional[NDArray],`
- `t3toolbox/backend/sampling_definitions.py:745` | `_the_seed` | weight | `return _dual_jet(S, _r(S, sample, y), weight)`
- `t3toolbox/backend/sampling_definitions.py:904` | `w` | a:weight | `weight:           typ.Optional[typ.Any] = None,      # of the type of z, each axis 1 or z's size; None = ones`
- `t3toolbox/backend/sampling_definitions.py:973` | `w_tilde` | a:weight | `weight:           typ.Optional[typ.Any] = None,      # of the type of z, each axis 1 or z's size; None = ones`
- `t3toolbox/backend/sampling_definitions.py:1040` | `w_tilde` | weight | `return _per_mode(S, i, _the_seed(S, sample, y, weight, w_tilde, 'w_tilde'), 'w_tilde')`
- `t3toolbox/backend/sampling_definitions.py:1058` | `xi` | a:weight | `weight:           typ.Optional[typ.Any] = None,      # of the type of z, each axis 1 or z's size; None = ones`
- `t3toolbox/backend/sampling_definitions.py:1104` | `mu` | a:weight | `weight:           typ.Optional[typ.Any] = None,      # of the type of z, each axis 1 or z's size; None = ones`
- `t3toolbox/backend/sampling_definitions.py:1189` | `nu` | a:weight | `weight:           typ.Optional[typ.Any] = None,      # of the type of z, each axis 1 or z's size; None = ones`
- `t3toolbox/backend/sampling_definitions.py:1237` | `eta` | a:weight | `weight:           typ.Optional[typ.Any] = None,      # of the type of z, each axis 1 or z's size; None = ones`
- `t3toolbox/backend/sampling_definitions.py:1287` | `nu_L` | a:weight | `weight:           typ.Optional[typ.Any] = None,      # of the type of z, each axis 1 or z's size; None = ones`
- `t3toolbox/backend/sampling_definitions.py:1334` | `eta_D` | a:weight | `weight:           typ.Optional[typ.Any] = None,      # of the type of z, each axis 1 or z's size; None = ones`
- `t3toolbox/backend/sampling_definitions.py:1389` | `xi_U_dot` | a:weight | `weight:           typ.Optional[typ.Any] = None,      # of the type of z, each axis 1 or z's size; None = ones`
- `t3toolbox/backend/sampling_definitions.py:1438` | `mu_R_dot` | a:weight | `weight:           typ.Optional[typ.Any] = None,      # of the type of z, each axis 1 or z's size; None = ones`
- `t3toolbox/backend/sampling_definitions.py:1500` | `nu_L_dot` | a:weight | `weight:           typ.Optional[typ.Any] = None,      # of the type of z, each axis 1 or z's size; None = ones`
- `t3toolbox/backend/sampling_definitions.py:1607` | `eta_D_dot` | a:weight | `weight:           typ.Optional[typ.Any] = None,      # of the type of z, each axis 1 or z's size; None = ones`
- `t3toolbox/backend/sampling_definitions.py:1683` | `xi_dot` | a:weight | `weight:           typ.Optional[typ.Any] = None,      # of the type of z, each axis 1 or z's size; None = ones`
- `t3toolbox/backend/sampling_definitions.py:1732` | `mu_dot` | a:weight | `weight:           typ.Optional[typ.Any] = None,      # of the type of z, each axis 1 or z's size; None = ones`
- `t3toolbox/backend/sampling_definitions.py:1806` | `xi_tilde` | a:weight | `weight:           typ.Optional[typ.Any] = None,      # of the type of z, each axis 1 or z's size; None = ones`
- `t3toolbox/backend/sampling_definitions.py:1849` | `xi_tilde` | weight | `seed_M = _at_nodes(S, _the_seed(S, sample, y, weight, w_tilde, 'xi_tilde'))`
- `t3toolbox/backend/sampling_definitions.py:1868` | `mu_tilde` | a:weight | `weight:           typ.Optional[typ.Any] = None,      # of the type of z, each axis 1 or z's size; None = ones`
- `t3toolbox/backend/sampling_definitions.py:1917` | `mu_tilde` | weight | `seed_M = _at_nodes(S, _the_seed(S, sample, y, weight, w_tilde, 'mu_tilde'))`
- `t3toolbox/backend/sampling_definitions.py:1936` | `nu_tilde` | a:weight | `weight:           typ.Optional[typ.Any] = None,      # of the type of z, each axis 1 or z's size; None = ones`
- `t3toolbox/backend/sampling_definitions.py:2002` | `nu_tilde` | weight | `seed_M = _at_nodes(S, _the_seed(S, sample, y, weight, w_tilde, 'nu_tilde'))`
- `t3toolbox/backend/sampling_definitions.py:2021` | `eta_tilde` | a:weight | `weight:           typ.Optional[typ.Any] = None,      # of the type of z, each axis 1 or z's size; None = ones`
- `t3toolbox/backend/sampling_definitions.py:2066` | `eta_tilde` | weight | `seed_M = _at_nodes(S, _the_seed(S, sample, y, weight, w_tilde, 'eta_tilde'))`
- `t3toolbox/backend/sampling_definitions.py:2086` | `nu_L_tilde` | a:weight | `weight:           typ.Optional[typ.Any] = None,      # of the type of z, each axis 1 or z's size; None = ones`
- `t3toolbox/backend/sampling_definitions.py:2131` | `nu_L_tilde` | weight | `seed_M = _at_nodes(S, _the_seed(S, sample, y, weight, w_tilde, 'nu_L_tilde'))`
- `t3toolbox/backend/sampling_definitions.py:2150` | `eta_D_tilde` | a:weight | `weight:           typ.Optional[typ.Any] = None,      # of the type of z, each axis 1 or z's size; None = ones`
- `t3toolbox/backend/sampling_definitions.py:2199` | `eta_D_tilde` | weight | `seed_M = _at_nodes(S, _the_seed(S, sample, y, weight, w_tilde, 'eta_D_tilde'))`
- `t3toolbox/backend/sampling_definitions.py:2223` | `xi_tilde_dot` | a:weight | `weight:           typ.Optional[typ.Any] = None,      # of the type of z, each axis 1 or z's size; None = ones`
- `t3toolbox/backend/sampling_definitions.py:2264` | `xi_tilde_dot` | weight | `seed_M = _at_nodes(S, _the_seed(S, sample, y, weight, w_tilde, 'xi_tilde_dot'))`
- `t3toolbox/backend/sampling_definitions.py:2283` | `mu_tilde_dot` | a:weight | `weight:           typ.Optional[typ.Any] = None,      # of the type of z, each axis 1 or z's size; None = ones`
- `t3toolbox/backend/sampling_definitions.py:2338` | `mu_tilde_dot` | weight | `seed_M = _at_nodes(S, _the_seed(S, sample, y, weight, w_tilde, 'mu_tilde_dot'))`
- `t3toolbox/backend/sampling_definitions.py:2358` | `nu_L_tilde_dot` | a:weight | `weight:           typ.Optional[typ.Any] = None,      # of the type of z, each axis 1 or z's size; None = ones`
- `t3toolbox/backend/sampling_definitions.py:2412` | `nu_L_tilde_dot` | weight | `seed_M = _at_nodes(S, _the_seed(S, sample, y, weight, w_tilde, 'nu_L_tilde_dot'))`
- `t3toolbox/backend/sampling_definitions.py:2432` | `eta_D_tilde_dot` | a:weight | `weight:           typ.Optional[typ.Any] = None,      # of the type of z, each axis 1 or z's size; None = ones`
- `t3toolbox/backend/sampling_definitions.py:2489` | `eta_D_tilde_dot` | weight | `seed_M = _at_nodes(S, _the_seed(S, sample, y, weight, w_tilde, 'eta_D_tilde_dot'))`
- `t3toolbox/backend/sampling_definitions.py:2514` | `z` | a:weight | `weight:           typ.Optional[typ.Any] = None,      # of the type of z, each axis 1 or z's size; None = ones`
- `t3toolbox/backend/sampling_definitions.py:2610` | `r` | a:weight | `weight:           typ.Optional[typ.Any] = None,      # of the type of z, each axis 1 or z's size; None = ones`
- `t3toolbox/backend/sampling_definitions.py:2655` | `Jp` | a:weight | `weight:           typ.Optional[typ.Any] = None,      # of the type of z, each axis 1 or z's size; None = ones`
- `t3toolbox/backend/sampling_definitions.py:2716` | `g` | a:weight | `weight:           typ.Optional[typ.Any] = None,      # of the type of z, each axis 1 or z's size; None = ones`
- `t3toolbox/backend/sampling_definitions.py:2766` | `g` | weight | `return _jacobian_transpose(S, _the_seed(S, sample, y, weight, w_tilde, 'g'))`
- `t3toolbox/backend/sampling_definitions.py:2779` | `JtJp` | a:weight | `weight:           typ.Optional[typ.Any] = None,      # of the type of z, each axis 1 or z's size; None = ones`
- `t3toolbox/backend/sampling_definitions.py:2819` | `JtJp` | weight | `return _jacobian_transpose(S, _dual_jet(S, _Jp(S, _need(variations, 'JtJp', 'variations (V, H)')), weight))`
- `t3toolbox/backend/sampling_definitions.py:2832` | `Kp` | a:weight | `weight:           typ.Optional[typ.Any] = None,      # of the type of z, each axis 1 or z's size; None = ones`
- `t3toolbox/backend/sampling_definitions.py:2887` | `Kp` | weight | `return _Kp(S, _the_seed(S, sample, y, weight, w_tilde, 'Kp'),`
- `t3toolbox/backend/sampling_definitions.py:2901` | `V_tilde` | a:weight | `weight:           typ.Optional[typ.Any] = None,      # of the type of z, each axis 1 or z's size; None = ones`
- `t3toolbox/backend/sampling_definitions.py:2941` | `V_tilde` | weight | `V_blocks, H_blocks = _jacobian_transpose(S, _the_seed(S, sample, y, weight, w_tilde, 'V_tilde'))`
- `t3toolbox/backend/sampling_definitions.py:2955` | `H_tilde` | a:weight | `weight:           typ.Optional[typ.Any] = None,      # of the type of z, each axis 1 or z's size; None = ones`
- `t3toolbox/backend/sampling_definitions.py:2999` | `H_tilde` | weight | `V_blocks, H_blocks = _jacobian_transpose(S, _the_seed(S, sample, y, weight, w_tilde, 'H_tilde'))`
- `t3toolbox/backend/sampling_definitions.py:3013` | `V_tilde_dot` | a:weight | `weight:           typ.Optional[typ.Any] = None,      # of the type of z, each axis 1 or z's size; None = ones`
- `t3toolbox/backend/sampling_definitions.py:3057` | `V_tilde_dot` | weight | `V_blocks, H_blocks = _Kp(S, _the_seed(S, sample, y, weight, w_tilde, 'V_tilde_dot'),`
- `t3toolbox/backend/sampling_definitions.py:3072` | `H_tilde_dot` | a:weight | `weight:           typ.Optional[typ.Any] = None,      # of the type of z, each axis 1 or z's size; None = ones`
- `t3toolbox/backend/sampling_definitions.py:3127` | `H_tilde_dot` | weight | `V_blocks, H_blocks = _Kp(S, _the_seed(S, sample, y, weight, w_tilde, 'H_tilde_dot'),`
- `t3toolbox/backend/sampling_definitions.py:3147` | `f` | a:weight | `weight:           typ.Optional[typ.Any] = None,      # of the type of z, each axis 1 or z's size; None = ones`
- `t3toolbox/backend/sampling_definitions.py:3193` | `f` | _weight_squared weight | `halves = _map_type_of_z(S, lambda i, r_TX: 0.5 * xnp.sum(_weight_squared(S, weight, i, r_TX.shape) * r_TX ** 2), r_)`
- `t3toolbox/backend/sampling_definitions.py:3207` | `r_block_sumsq` | a:weight | `weight:           typ.Optional[typ.Any] = None,      # of the type of z, each axis 1 or z's size; None = ones`
- `t3toolbox/backend/sampling_definitions.py:3257` | `r_block_sumsq.sumsq` | _weight_squared weight | `return xnp.sum(_weight_squared(S, weight, i, r_FX.shape) * r_FX ** 2, axis=tuple(range(m, r_FX.ndim)))`
- `t3toolbox/backend/sampling_definitions.py:3272` | `Jp_norm_sq` | a:weight | `weight:           typ.Optional[typ.Any] = None,      # of the type of z, each axis 1 or z's size; None = ones`
- `t3toolbox/backend/sampling_definitions.py:3307` | `Jp_norm_sq` | _weight_squared weight | `squares = _map_type_of_z(S, lambda i, Jp_TX: xnp.sum(_weight_squared(S, weight, i, Jp_TX.shape) * Jp_TX ** 2), Jp_)`

**`t3toolbox/backend/ut3_operations.py`**: 40 code lines; (not listed: 39 docstring-prose lines, 0 doctest lines, 16 comment lines mention "weight")

- `t3toolbox/backend/ut3_operations.py:313` | `ut3_absorb_weights` | d:ut3_absorb_weights | `def ut3_absorb_weights(`
- `t3toolbox/backend/ut3_operations.py:315` | `ut3_absorb_weights` | a:weights | `weights: UT3WeightsData,  # (tucker_weight_supercore, tt_weight_supercore, masks)`
- `t3toolbox/backend/ut3_operations.py:341` | `ut3_absorb_weights` | weights | `xnp, _, _ = get_backend(True, tree_contains_jax((x[:2], weights[:2])))`
- `t3toolbox/backend/ut3_operations.py:343` | `ut3_absorb_weights` | tt_weight_supercore tucker_weight_supercore weights | `tucker_weight_supercore, tt_weight_supercore, _ = weights`
- `t3toolbox/backend/ut3_operations.py:345` | `ut3_absorb_weights` | tucker_weight_supercore weighted_tucker | `weighted_tucker = xnp.einsum('d...n,d...nN->d...nN', tucker_weight_supercore, tucker_supercore)`
- `t3toolbox/backend/ut3_operations.py:348` | `ut3_absorb_weights` | tt_weight_supercore weighted_tt | `weighted_tt = xnp.einsum('d...lnr,d...r->d...lnr', tt_supercore, tt_weight_supercore[1:])`
- `t3toolbox/backend/ut3_operations.py:351` | `ut3_absorb_weights` | tt_weight_supercore weighted_tt | `G0 = xnp.einsum('d...lnr,d...l->d...lnr', weighted_tt[:1], tt_weight_supercore[:1])`
- `t3toolbox/backend/ut3_operations.py:352` | `ut3_absorb_weights` | weighted_tt | `weighted_tt = xnp.concatenate([G0, weighted_tt[1:]], axis=0)`
- `t3toolbox/backend/ut3_operations.py:354` | `ut3_absorb_weights` | weighted_tt weighted_tucker | `return weighted_tucker, weighted_tt, shape, masks`
- `t3toolbox/backend/ut3_operations.py:357` | `ut3_weights_consistent` | d:ut3_weights_consistent | `def ut3_weights_consistent(`
- `t3toolbox/backend/ut3_operations.py:359` | `ut3_weights_consistent` | a:weights | `weights: UT3WeightsData,  # (tucker_weight_supercore, tt_weight_supercore, masks)`
- `t3toolbox/backend/ut3_operations.py:373` | `ut3_weights_consistent` | tt_weight_supercore tucker_weight_supercore weight_tt_mask weight_tucker_mask weights | `tucker_weight_supercore, tt_weight_supercore, (weight_tucker_mask, weight_tt_mask) = weights`
- `t3toolbox/backend/ut3_operations.py:380` | `ut3_weights_consistent` | tucker_weight_supercore | `if tuple(tucker_weight_supercore.shape) != (d,) + stack + (n,):`
- `t3toolbox/backend/ut3_operations.py:382` | `ut3_weights_consistent` | tt_weight_supercore | `if tuple(tt_weight_supercore.shape) != (d + 1,) + stack + (r,):`
- `t3toolbox/backend/ut3_operations.py:386` | `ut3_weights_consistent` | weight_tucker_mask | `return (np.array_equal(weight_tucker_mask, tucker_edge_mask)`
- `t3toolbox/backend/ut3_operations.py:387` | `ut3_weights_consistent` | weight_tt_mask | `and np.array_equal(weight_tt_mask, tt_edge_mask))`
- `t3toolbox/backend/ut3_operations.py:390` | `_ut3_map_real_weights` | d:_ut3_map_real_weights | `def _ut3_map_real_weights(`
- `t3toolbox/backend/ut3_operations.py:391` | `_ut3_map_real_weights` | a:weights | `weights: UT3WeightsData,  # (tucker_weight_supercore, tt_weight_supercore, masks)`
- `t3toolbox/backend/ut3_operations.py:405` | `_ut3_map_real_weights` | weights | `xnp, _, _ = get_backend(True, tree_contains_jax(weights[:2]))`
- `t3toolbox/backend/ut3_operations.py:406` | `_ut3_map_real_weights` | tt_weight_supercore tucker_weight_supercore weights | `tucker_weight_supercore, tt_weight_supercore, (tucker_edge_mask, tt_edge_mask) = weights`
- `t3toolbox/backend/ut3_operations.py:413` | `_ut3_map_real_weights` | tucker_weight_supercore | `return (go(tucker_weight_supercore, tucker_edge_mask),`
- `t3toolbox/backend/ut3_operations.py:414` | `_ut3_map_real_weights` | tt_weight_supercore | `go(tt_weight_supercore, tt_edge_mask),`
- `t3toolbox/backend/ut3_operations.py:418` | `ut3_reciprocal_weights` | d:ut3_reciprocal_weights | `def ut3_reciprocal_weights(`
- `t3toolbox/backend/ut3_operations.py:419` | `ut3_reciprocal_weights` | a:weights | `weights: UT3WeightsData,  # (tucker_weight_supercore, tt_weight_supercore, masks)`
- `t3toolbox/backend/ut3_operations.py:433` | `ut3_reciprocal_weights` | _ut3_map_real_weights weights | `return _ut3_map_real_weights(weights, lambda xnp, w: 1.0 / w)`
- `t3toolbox/backend/ut3_operations.py:436` | `ut3_sqrt_weights` | d:ut3_sqrt_weights | `def ut3_sqrt_weights(`
- `t3toolbox/backend/ut3_operations.py:437` | `ut3_sqrt_weights` | a:weights | `weights: UT3WeightsData,  # (tucker_weight_supercore, tt_weight_supercore, masks)`
- `t3toolbox/backend/ut3_operations.py:442` | `ut3_sqrt_weights` | _ut3_map_real_weights weights | `return _ut3_map_real_weights(weights, lambda xnp, w: xnp.sqrt(w))`
- `t3toolbox/backend/ut3_operations.py:445` | `_weight_pair_backend` | a:weights_A a:weights_B d:_weight_pair_backend | `def _weight_pair_backend(weights_A: UT3WeightsData, weights_B: UT3WeightsData):`
- `t3toolbox/backend/ut3_operations.py:447` | `_weight_pair_backend` | weights_A weights_B | `xnp, _, _ = get_backend(True, tree_contains_jax((weights_A[:2], weights_B[:2])))`
- `t3toolbox/backend/ut3_operations.py:448` | `_weight_pair_backend` | weights_A weights_B | `require_concrete_masks(*weights_A[2], *weights_B[2])  # masks are host, not traced`
- `t3toolbox/backend/ut3_operations.py:449` | `_weight_pair_backend` | weights_A weights_B | `return xnp, weights_A[:2], weights_B[:2], weights_A[2], weights_B[2]`
- `t3toolbox/backend/ut3_operations.py:452` | `ut3_concatenate_weights` | d:ut3_concatenate_weights | `def ut3_concatenate_weights(`
- `t3toolbox/backend/ut3_operations.py:453` | `ut3_concatenate_weights` | a:weights_A | `weights_A: UT3WeightsData,  # (tucker_weight_supercore, tt_weight_supercore, masks)`
- `t3toolbox/backend/ut3_operations.py:454` | `ut3_concatenate_weights` | a:weights_B | `weights_B: UT3WeightsData,  # (tucker_weight_supercore, tt_weight_supercore, masks)`
- `t3toolbox/backend/ut3_operations.py:471` | `ut3_concatenate_weights` | _weight_pair_backend weights_A weights_B | `xnp, (tkA, ttA), (tkB, ttB), (tkmA, ttmA), (tkmB, ttmB) = _weight_pair_backend(weights_A, weights_B)`
- `t3toolbox/backend/ut3_operations.py:479` | `ut3_kronecker_weights` | d:ut3_kronecker_weights | `def ut3_kronecker_weights(`
- `t3toolbox/backend/ut3_operations.py:480` | `ut3_kronecker_weights` | a:weights_A | `weights_A: UT3WeightsData,  # (tucker_weight_supercore, tt_weight_supercore, masks)`
- `t3toolbox/backend/ut3_operations.py:481` | `ut3_kronecker_weights` | a:weights_B | `weights_B: UT3WeightsData,  # (tucker_weight_supercore, tt_weight_supercore, masks)`
- `t3toolbox/backend/ut3_operations.py:511` | `ut3_kronecker_weights` | _weight_pair_backend weights_A weights_B | `xnp, (tkA, ttA), (tkB, ttB), (tkmA, ttmA), (tkmB, ttmB) = _weight_pair_backend(weights_A, weights_B)`

**`t3toolbox/fitting.py`**: 4 code lines; (not listed: 10 docstring-prose lines, 0 doctest lines, 4 comment lines mention "weight")

- `t3toolbox/fitting.py:244` | `_canonical_weight` | d:_canonical_weight | `def _canonical_weight(`
- `t3toolbox/fitting.py:245` | `_canonical_weight` | a:weight | `weight:    typ.Optional[typ.Any],  # None, or array / (nested) sequence -- the raw ω input`
- `t3toolbox/fitting.py:257` | `_canonical_weight` | weight | `if weight is None:`
- `t3toolbox/fitting.py:259` | `_canonical_weight` | weight | `w = np.asarray(weight, dtype=float)`

**`t3toolbox/optimizers.py`**: 11 code lines; (not listed: 6 docstring-prose lines, 0 doctest lines, 2 comment lines mention "weight")

- `t3toolbox/optimizers.py:132` | `_old_gradient_descent` | a:weight | `def _old_gradient_descent(geometry, kind, sample, data, x0, order=None, weight=None, regularizer=None,`
- `t3toolbox/optimizers.py:137` | `_old_mc_sgd` | a:weight | `def _old_mc_sgd(geometry, kind, sample, data, x0, rng, batch, order=None, weight=None, draw=None, use_jit=False,`
- `t3toolbox/optimizers.py:142` | `_old_newton_cg` | a:weight | `def _old_newton_cg(geometry, kind, sample, data, x0, order=None, weight=None, verbose=False, val_sample=None,`
- `t3toolbox/optimizers.py:177` | `_old_fit` | a:weight | `weight:       typ.Any,             # the old residual weight, or None`
- `t3toolbox/optimizers.py:191` | `_old_fit` | weight | `if weight is not None and kind in ('apply', 'entries'):`
- `t3toolbox/optimizers.py:194` | `_old_fit` | ._canonical_weight weight | `wm = _fitting._canonical_weight(weight, kind, d, 0)       # (1, 1) or (m >= d, 1): one row broadcasts`
- `t3toolbox/optimizers.py:196` | `_old_fit` | .blocks_to_weight | `w = None if wm is None else jets.blocks_to_weight(kind, (), np.ascontiguousarray(`
- `t3toolbox/optimizers.py:206` | `_old_fit` | ._canonical_weight weight | `wm = _fitting._canonical_weight(weight, kind, d, n)`
- `t3toolbox/optimizers.py:216` | `_old_fit` | .blocks_to_weight | `w = None if w is None else jets.blocks_to_weight(base, (n,), w, data)          # on the derivatives, then converted`
- `t3toolbox/optimizers.py:221` | `_old_draw` | a:weight | `def _old_draw(draw, kind, shape, order, weight, scale, who, stack_shape=()):`
- `t3toolbox/optimizers.py:227` | `_old_draw.new_draw` | weight | `_, b = _old_fit(kind, sample_b, data_b, shape, order, weight, who, stack_shape)`

Rows per file: `backend/sampling_variables.py` 7, `backend/sampling_steps.py` 26, `backend/fitting.py` 39, `backend/uniform_fitting.py` 16, `backend/jets.py` 62, `backend/data_preparation.py` 1, `backend/optimizers.py` 0, `backend/optimizer_display.py` 0, `backend/sampling_definitions.py` 67, `backend/ut3_operations.py` 40, `fitting.py` 4, `optimizers.py` 11

### (b) Call counts by area and file

Method: library/tests/examples = `ast.Call` nodes whose callee is a bare Name or an Attribute with this name (any receiver); "library doctests" = regex `name(` on `>>>`/`...` lines in library docstrings; docs = regex `name(` on lines of `docs/*.md|rst`, `docs/contributor/*.md` split by doctest / fenced / prose. Definitions are not calls. A reference that is not a call (passed as a value, imported) is not counted here.

**`blocks_to_weight(`**
- library: 3 calls -- `t3toolbox/backend/jets.py` 1, `t3toolbox/optimizers.py` 2
- tests: 48 calls -- `tests/_fits.py` 2, `tests/backend/test_optimizer_display.py` 2, `tests/bridge/test_old_kernels_bridge.py` 1, `tests/test_block_sums.py` 2, `tests/test_data_metric.py` 10, `tests/test_data_preparation.py` 1, `tests/test_dispatch.py` 12, `tests/test_engine_data_space.py` 1, `tests/test_fitting.py` 2, `tests/test_jets.py` 1, `tests/test_least_squares_oracle.py` 2, `tests/test_optimizers_frontend.py` 5, `tests/test_sampling_definitions.py` 1, `tests/test_sampling_typed.py` 1, `tests/test_type_laws.py` 5
- examples: 0 calls
- library doctests: 7 -- `t3toolbox/backend/fitting.py` 1, `t3toolbox/backend/jets.py` 6
- docs pages: 2 lines -- `docs/fitting_and_optimization.md` {'doctest': 2}
- receivers (area, receiver): library `<bare>` x1, library `jets` x2, tests `<bare>` x12, tests `fitting` x1, tests `jets` x35
- library call sites:
  - `t3toolbox/backend/jets.py:1178` | `prepare_derivative_data` | recv `<bare>`
  - `t3toolbox/optimizers.py:196` | `_old_fit` | recv `jets`
  - `t3toolbox/optimizers.py:216` | `_old_fit` | recv `jets`

**`data_metric(`**
- library: 6 calls -- `t3toolbox/backend/fitting.py` 2, `t3toolbox/backend/jets.py` 1, `t3toolbox/backend/sampling_steps.py` 3
- tests: 41 calls -- `tests/test_data_metric.py` 24, `tests/test_dispatch.py` 6, `tests/test_sampling_typed.py` 4, `tests/test_transposes_typed.py` 1, `tests/test_type_laws.py` 6
- examples: 0 calls
- library doctests: 8 -- `t3toolbox/backend/fitting.py` 1, `t3toolbox/backend/jets.py` 7
- docs pages: 1 lines -- `docs/contributor/signature_style.md` {'fence': 1}
- receivers (area, receiver): library `<bare>` x1, library `jets` x5, tests `Call` x1, tests `batch` x2, tests `drawn` x1, tests `jets` x37
- library call sites:
  - `t3toolbox/backend/fitting.py:806` | `LeastSquaresBatch.data_metric` | recv `jets`
  - `t3toolbox/backend/fitting.py:892` | `_y_times_metric_y` | recv `jets`
  - `t3toolbox/backend/jets.py:452` | `jet_to_dual_jet` | recv `<bare>`
  - `t3toolbox/backend/sampling_steps.py:959` | `_weighted_sum` | recv `jets`
  - `t3toolbox/backend/sampling_steps.py:1023` | `w_tilde_step` | recv `jets`
  - `t3toolbox/backend/sampling_steps.py:1097` | `r_block_sumsq_step` | recv `jets`

**`check_weight(`**
- library: 1 calls -- `t3toolbox/backend/fitting.py` 1
- tests: 9 calls -- `tests/test_data_preparation.py` 1, `tests/test_dispatch.py` 1, `tests/test_jets.py` 7
- examples: 0 calls
- library doctests: 6 -- `t3toolbox/backend/jets.py` 6
- docs pages: 0 lines
- receivers (area, receiver): library `jets` x1, tests `jets` x9
- library call sites:
  - `t3toolbox/backend/fitting.py:1164` | `LeastSquaresOracle.check_batch` | recv `jets`

**`check_data(`**
- library: 1 calls -- `t3toolbox/backend/fitting.py` 1
- tests: 9 calls -- `tests/test_dispatch.py` 1, `tests/test_jets.py` 8
- examples: 0 calls
- library doctests: 7 -- `t3toolbox/backend/jets.py` 7
- docs pages: 0 lines
- receivers (area, receiver): library `jets` x1, tests `jets` x9
- library call sites:
  - `t3toolbox/backend/fitting.py:1163` | `LeastSquaresOracle.check_batch` | recv `jets`

**`derivative_weight_to_taylor_weight(`**
- library: 0 calls
- tests: 8 calls -- `tests/test_dispatch.py` 1, `tests/test_jets.py` 5, `tests/test_least_squares_oracle.py` 1, `tests/test_sampling_typed.py` 1
- examples: 0 calls
- library doctests: 4 -- `t3toolbox/backend/jets.py` 4
- docs pages: 0 lines
- receivers (area, receiver): tests `jets` x8
- library call sites:

**`prepare_derivative_data(`**
- library: 1 calls -- `t3toolbox/optimizers.py` 1
- tests: 20 calls -- `tests/_fits.py` 1, `tests/test_dispatch.py` 5, `tests/test_jets.py` 12, `tests/test_optimizers_frontend.py` 1, `tests/test_sampling_typed.py` 1
- examples: 0 calls
- library doctests: 8 -- `t3toolbox/backend/jets.py` 8
- docs pages: 2 lines -- `docs/fitting_and_optimization.md` {'doctest': 1}, `docs/numerical_contracts.md` {'doctest': 1}
- receivers (area, receiver): library `jets` x1, tests `fitting` x1, tests `jets` x19
- library call sites:
  - `t3toolbox/optimizers.py:217` | `_old_fit` | recv `jets`

**`equalizing_weight(`**
- library: 0 calls
- tests: 13 calls -- `tests/test_data_preparation.py` 11, `tests/test_dispatch.py` 1, `tests/test_sampling_typed.py` 1
- examples: 0 calls
- library doctests: 3 -- `t3toolbox/backend/data_preparation.py` 3
- docs pages: 0 lines
- receivers (area, receiver): tests `prep` x13
- library call sites:

**`data_block_norms(`**
- library: 0 calls
- tests: 16 calls -- `tests/test_block_sums.py` 1, `tests/test_data_preparation.py` 10, `tests/test_dispatch.py` 4, `tests/test_sampling_typed.py` 1
- examples: 0 calls
- library doctests: 7 -- `t3toolbox/backend/data_preparation.py` 7
- docs pages: 0 lines
- receivers (area, receiver): tests `data_preparation` x1, tests `prep` x15
- library call sites:

**`_with_K(`**
- library: 2 calls -- `t3toolbox/backend/sampling_steps.py` 2
- tests: 0 calls
- examples: 0 calls
- library doctests: 0
- docs pages: 0 lines
- receivers (area, receiver): library `<bare>` x2
- library call sites:
  - `t3toolbox/backend/sampling_steps.py:1006` | `Jp_norm_sq_step` | recv `<bare>`
  - `t3toolbox/backend/sampling_steps.py:1024` | `w_tilde_step` | recv `<bare>`

**`_pack_data_space(`**
- library: 3 calls -- `t3toolbox/backend/sampling_variables.py` 2, `t3toolbox/backend/uniform_fitting.py` 1
- tests: 0 calls
- examples: 0 calls
- library doctests: 0
- docs pages: 0 lines
- receivers (area, receiver): library `ut3_operations` x3
- library call sites:
  - `t3toolbox/backend/sampling_variables.py:1872` | `sampling_variables` | recv `ut3_operations`
  - `t3toolbox/backend/sampling_variables.py:1873` | `sampling_variables` | recv `ut3_operations`
  - `t3toolbox/backend/uniform_fitting.py:427` | `pack_data` | recv `ut3_operations`

**`_take_data_space(`**
- library: 2 calls -- `t3toolbox/backend/fitting.py` 2
- tests: 0 calls
- examples: 0 calls
- library doctests: 0
- docs pages: 0 lines
- receivers (area, receiver): library `<bare>` x2
- library call sites:
  - `t3toolbox/backend/fitting.py:1235` | `LeastSquaresOracle.take` | recv `<bare>`
  - `t3toolbox/backend/fitting.py:1236` | `LeastSquaresOracle.take` | recv `<bare>`

**`_arrays_of(`**
- library: 2 calls -- `t3toolbox/backend/fitting.py` 2
- tests: 0 calls
- examples: 0 calls
- library doctests: 0
- docs pages: 0 lines
- receivers (area, receiver): library `<bare>` x2
- library call sites:
  - `t3toolbox/backend/fitting.py:890` | `_y_times_metric_y` | recv `<bare>`
  - `t3toolbox/backend/fitting.py:890` | `_y_times_metric_y` | recv `<bare>`

`data_metric(` split by receiver: library function (receiver is the jets module or bare) x6; tests METHOD (receiver Call) x1; tests METHOD (receiver batch) x2; tests METHOD (receiver drawn) x1; tests function (receiver is the jets module or bare) x37
`data_metric` function-vs-method: the DEFINITIONS are `jets.data_metric` (function, `t3toolbox/backend/jets.py:579`) and `LeastSquaresBatch.data_metric` (method, `t3toolbox/backend/fitting.py:785`). The receiver lists above separate them: a receiver that is a module alias (`jets`, `bjets`, ...) is the function; a receiver that is an object (`oracle`, `self`, `o`) is the method. Ambiguous receivers (names I cannot resolve statically) are those not obviously one or the other: see the receiver line.

### (c) `tests/`: occurrences of keyword `weight=` and of `LeastSquaresBatch(`

Method: `weight=` = `ast.keyword` with arg `weight` anywhere (calls, incl. nested) plus `weight=` inside doctest lines/strings is NOT counted; `LeastSquaresBatch(` = `ast.Call` of Name/Attribute `LeastSquaresBatch`.

`weight=` keyword: total 206 in 29 files
- `tests/_fits.py`: 4
- `tests/backend/test_curvature.py`: 38
- `tests/backend/test_optimizer_display.py`: 2
- `tests/backend/test_optimizers.py`: 5
- `tests/backend/test_uniform_curvature.py`: 11
- `tests/backend/test_uniform_fitting.py`: 2
- `tests/backend/test_zero_reference.py`: 1
- `tests/bridge/test_old_kernels_bridge.py`: 11
- `tests/oracle_sweeps/sweep_optimizers.py`: 1
- `tests/test_block_sums.py`: 4
- `tests/test_contractions_sharding.py`: 2
- `tests/test_data_metric.py`: 8
- `tests/test_dense_oracle.py`: 2
- `tests/test_dispatch.py`: 19
- `tests/test_engine_data_space.py`: 22
- `tests/test_jets.py`: 2
- `tests/test_least_squares_oracle.py`: 8
- `tests/test_optimizers_frontend.py`: 16
- `tests/test_oracle_data_space.py`: 1
- `tests/test_sampling_apply.py`: 4
- `tests/test_sampling_corewise.py`: 4
- `tests/test_sampling_definitions.py`: 13
- `tests/test_sampling_entries.py`: 8
- `tests/test_sampling_graph.py`: 2
- `tests/test_sampling_identities.py`: 5
- `tests/test_sampling_stacks.py`: 3
- `tests/test_sampling_typed.py`: 1
- `tests/test_sampling_uniform.py`: 2
- `tests/test_sampling_variables.py`: 5

`LeastSquaresBatch(` calls: total 76 in 18 files
- `tests/_fits.py`: 2
- `tests/backend/test_curvature.py`: 2
- `tests/backend/test_optimizer_display.py`: 3
- `tests/backend/test_optimizers.py`: 1
- `tests/backend/test_uniform_fitting.py`: 1
- `tests/test_aligned_padded_frame_frontend.py`: 1
- `tests/test_block_sums.py`: 4
- `tests/test_data_metric.py`: 4
- `tests/test_dispatch.py`: 1
- `tests/test_fitting.py`: 11
- `tests/test_least_squares_oracle.py`: 5
- `tests/test_manifold.py`: 1
- `tests/test_optimizers_frontend.py`: 11
- `tests/test_oracle_data_space.py`: 25
- `tests/test_sampling_typed.py`: 1
- `tests/test_sharing.py`: 1
- `tests/test_transposes_typed.py`: 1
- `tests/test_type_laws.py`: 1

For comparison, the same two patterns in the other areas: library `weight=` keywords 6 (files: `t3toolbox/backend/fitting.py`, `t3toolbox/backend/sampling_variables.py`, `t3toolbox/backend/uniform_fitting.py`); library `LeastSquaresBatch(` calls 4; examples `weight=` 5, `LeastSquaresBatch(` 5; docs pages (regex, lines) `weight=` 9, `LeastSquaresBatch(` 4.

### (d) Current signatures, verbatim (def header through the closing colon, trailing comments kept)

- `t3toolbox/backend/jets.py:579-583` `data_metric`
```python
def data_metric(
        r:                typ.Any,                  # Y*, of the type of z, no K: probe len=d (or packed (d,)+...), elm_shape=F+W+C+(Ni,); apply F+W+C; entries W+C
        weight:           typ.Optional[typ.Any],    # omega, of the type of z: each axis 1 or r's size; None = ones
        minibatch_scale:  typ.Optional[typ.Any],    # gamma: a number or a 0-d array (traced under jit); None = 1
) -> typ.Any:                                       # M r: Y, gamma * omega**2 * r elementwise, r's structure (a tuple for a tuple or a list)
```
- `t3toolbox/backend/jets.py:767-772` `blocks_to_weight`
```python
def blocks_to_weight(
        kind:    str,                               # 'probe' | 'apply' | 'entries'
        order:   typ.Tuple[int, ...],               # the jet shape (n_1, ..., n_m); F = (n_1 + 1, ...). entries: ()
        blocks:  typ.Any,                           # one value per block: probe (d,)+F; apply F; entries ()
        y:       typ.Any,                           # Y*, of the type of z: only its rank is read (C included)
) -> NDArray:                                       # omega: probe (d,)+F+(1,)*(len W+len C)+(1,); apply F+(1,)*(len W+len C); entries (1,)*(len W+len C)
```
- `t3toolbox/backend/jets.py:997-1001` `derivative_weight_to_taylor_weight`
```python
def derivative_weight_to_taylor_weight(
        kind:    str,                      # 'probe' | 'apply' | 'entries': which type the weight is of
        order:   typ.Tuple[int, ...],      # the jet shape (n_1, ..., n_m); F = (n_1 + 1, ...). entries: ()
        weight:  typ.Any,                  # omega on derivative residuals, of the type of z: probe len=d or (d,)+...; else one array
) -> typ.Any:                              # omega on Taylor residuals, in the form given: slot k times k_1! ... k_m!
```
- `t3toolbox/backend/jets.py:1080-1092` `prepare_derivative_data`
```python
def prepare_derivative_data(
        kind:        str,                              # 'probe' | 'apply' | 'entries'
        order:       typ.Tuple[int, ...],              # the jet shape (n_1, ..., n_m); F = (n_1 + 1, ...). entries: ()
        y:           typ.Any,                          # Y*, in derivative coordinates: probe len=d,
                                                       #   elm_shape=F+W+C+(Ni,); apply F+W+C; entries W+C
        weight:      typ.Optional[typ.Any] = None,     # omega on derivative residuals, of the type of z; None = ones on the derivatives
        validation:  typ.Optional[typ.Any] = None,     # Y*, validation data in derivative coordinates, or None
) -> typ.Tuple[
        typ.Any,                                       # y: Y*, in Taylor coordinates, same shapes
        typ.Any,                                       # omega, the Taylor weight, of the type of z: k_1! ... k_m! per
                                                       #   slot when weight is None
        typ.Optional[typ.Any],                         # validation: Y*, in Taylor coordinates, or None
]:
```
- `t3toolbox/backend/data_preparation.py:130-134` `data_block_norms`
```python
def data_block_norms(
        kind:   str,                       # 'probe' | 'apply' | 'entries'
        order:  typ.Tuple[int, ...],       # the jet shape (n_1, ..., n_m); F = (n_1 + 1, ...). entries: ()
        y:      typ.Any,                   # the data: Y*, probe len=d, elm_shape=F+W+(Ni,); apply F+W; entries W
) -> NDArray:                              # ℝ per block, the block shape: probe (d,)+F; apply F; entries ()
```
- `t3toolbox/backend/data_preparation.py:199-202` `equalizing_weight`
```python
def equalizing_weight(
        block_norms:  NDArray,             # ℝ per block, from data_block_norms: probe (d,)+F; apply F; entries ()
        floor:        float = 1e-6,        # relative to the largest block
) -> NDArray:                              # same shape: 1 / max(x, floor * max(x))
```
- `t3toolbox/backend/data_preparation.py:279-289` `rescale_direction`
```python
def rescale_direction(
        kind:    str,                      # 'probe' | 'apply'
        order:   typ.Tuple[int, ...],      # the jet shape (n_1, ..., n_m), at least one jet variable
        sample:  typ.Any,                  # the input jets ww: P_in_i⊗V_i, len=d, elm_shape=F_in_i+W+(Ni,)
        y:       typ.Any,                  # Y*, the data: probe len=d, elm_shape=F+W+(Ni,); apply F+W
        length:  typ.Any,                  # a number or a 0-d array, or one per jet variable (len=m): slot k of the
                                           #   input jets and of y times length_1**k_1 ... length_m**k_m   (loss-only)
) -> typ.Tuple[
        typ.Any,                           # the sample: P_in_i⊗V_i, the input jets rescaled, a tuple
        typ.Any,                           # y: Y*, same shapes, rescaled the same way
]:
```
- `t3toolbox/backend/data_preparation.py:391-401` `rescale_coordinates`
```python
def rescale_coordinates(
        kind:    str,                      # 'probe' | 'apply'
        order:   typ.Tuple[int, ...],      # the jet shape (n_1, ..., n_m); F = (n_1 + 1, ...)
        sample:  typ.Any,                  # the input jets ww: P_in_i⊗V_i, len=d, elm_shape=F_in_i+W+(Ni,)
        y:       typ.Any,                  # Y*, the data: probe len=d, elm_shape=F+W+(Ni,); apply F+W
        scales:  typ.Sequence[NDArray],    # len=d, elm_shape=(Ni,): coordinate j of mode i times scales[i][j]
) -> typ.Tuple[
        typ.Any,                           # the sample: P_in_i⊗V_i, every slot of every input jet times its mode's
                                           #   scales
        typ.Any,                           # y: Y*; for probe, mode i's data divided by scales[i]; for apply, as given
]:
```
- `t3toolbox/backend/data_preparation.py:566-579` `ut3_unscale_coordinates`
```python
def ut3_unscale_coordinates(
        x:       typ.Tuple[
                     NDArray,                           # tucker_supercore, shape=(d,)+C+(n,N)
                     NDArray,                           # tt_supercore,     shape=(d,)+C+(r,n,r)
                     typ.Tuple[int, ...],               # shape, len=d: the mode sizes Ni
                     typ.Tuple[NDArray, NDArray],       # (tucker_edge_mask, tt_edge_mask), host bool
                 ],                                     # fitted in rescaled coordinates
        scales:  typ.Union[typ.Sequence[NDArray], NDArray],     # len=d, elm_shape=(Ni,); or packed, shape=(d,N)
) -> typ.Tuple[
        NDArray,                                        # tucker_supercore, shape=(d,)+C+(n,N): the last axis scaled
        NDArray,                                        # tt_supercore, as given
        typ.Tuple[int, ...],                            # shape, as given
        typ.Tuple[NDArray, NDArray],                    # masks, as given
]:
```
- `t3toolbox/backend/data_preparation.py:649-652` `check_coordinate_scales`
```python
def check_coordinate_scales(
        scales:  typ.Sequence[NDArray],    # expected: len=d, elm_shape=(Ni,)
        shape:   typ.Sequence[int],        # the mode sizes N_i, len=d
) -> None:
```
- `t3toolbox/backend/jets.py:324-329` `jet_to_dual_jet`
```python
def jet_to_dual_jet(
        kind:    str,                      # 'probe' | 'apply' | 'entries': which type the jet is of
        order:   typ.Tuple[int, ...],      # the jet shape (n_1, ..., n_m); F = (n_1 + 1, ...). entries: ()
        jet:     typ.Any,                  # Y*, of the type of z: probe len=d (or packed), elm_shape=F+W+[K]+C+(Ni,); apply F+W+[K]+C; entries W+[K]+C
        weight:  typ.Optional[typ.Any],    # omega, of the type of z (each axis 1 or the jet's size, no K); None = ones
) -> typ.Any:                              # Y, turned: data_metric(jet, weight, None), every jet axis flipped. probe: a tuple
```
- `t3toolbox/backend/jets.py:456-461` `dual_jet_to_jet`
```python
def dual_jet_to_jet(
        kind:      str,                    # 'probe' | 'apply' | 'entries': which type the dual jet is of
        order:     typ.Tuple[int, ...],    # the jet shape (n_1, ..., n_m); F = (n_1 + 1, ...). entries: ()
        dual_jet:  typ.Any,                # Y, turned: same type and shapes as jet_to_dual_jet's result
        weight:    typ.Optional[typ.Any],  # omega, of the type of z (each axis 1 or the dual jet's size); None = ones
) -> typ.Any:                              # Y*: every jet axis flipped back, then divided by weight**2. probe: a tuple
```
- `t3toolbox/backend/jets.py:852-857` `derivatives_to_taylor`
```python
def derivatives_to_taylor(
        kind:   str,                       # 'probe' | 'apply' | 'entries': which type the jets are of
        order:  typ.Tuple[int, ...],       # the jet shape (n_1, ..., n_m); F = (n_1 + 1, ...). entries: ()
        jets:   typ.Any,                   # Y* in derivative coordinates (or Y in Taylor ones: the contragredient):
                                           #   probe len=d, elm_shape=F+W+(Ni,); apply F+W; entries W
) -> typ.Any:                              # Y* in Taylor coordinates (or Y in derivative ones), same shapes: slot k
```
- `t3toolbox/backend/jets.py:940-945` `taylor_to_derivatives`
```python
def taylor_to_derivatives(
        kind:   str,                       # 'probe' | 'apply' | 'entries': which type the jets are of
        order:  typ.Tuple[int, ...],       # the jet shape (n_1, ..., n_m); F = (n_1 + 1, ...). entries: ()
        jets:   typ.Any,                   # Y* in Taylor coordinates (or Y in derivative ones: the contragredient):
                                           #   probe len=d, elm_shape=F+W+(Ni,); apply F+W; entries W
) -> typ.Any:                              # Y* in derivative coordinates (or Y in Taylor ones), same shapes: slot k
```
- `t3toolbox/backend/jets.py:1208-1212` `affine_jet`
```python
def affine_jet(
        ww:             typ.Union[typ.Sequence[NDArray], NDArray],  # V_i, len=d, elm_shape=W+(Ni,); or packed
                                                                    #   (d,)+W+(N,)
        *directions:    typ.Union[typ.Sequence[NDArray], NDArray],  # V_i, m of them, each of the form and shapes of ww
) -> typ.Union[typ.Tuple[NDArray, ...], NDArray]:   # P_in_i⊗V_i, len=d, elm_shape=(2,)*m+W+(Ni,); or packed
```
- `t3toolbox/backend/jets.py:1303-1306` `index_to_one_hot`
```python
def index_to_one_hot(
        index:  NDArray,                # index^d, shape=(d,)+W, int
        shape:  typ.Sequence[int],      # the mode sizes N_i, len=d
) -> typ.Tuple[NDArray, ...]:           # e_index: V_i, len=d, elm_shape=W+(Ni,), in the default float dtype
```
- `t3toolbox/backend/jets.py:1424-1429` `check_sample`
```python
def check_sample(
        kind:    str,                      # 'probe' | 'apply' | 'entries'
        order:   typ.Tuple[int, ...],      # the jet shape (n_1, ..., n_m); entries: ()
        sample:  typ.Any,                  # P_in_i⊗V_i (probe, apply: the input jets ww) | index^d (entries)
        shape:   typ.Sequence[int],        # the mode sizes N_i, len=d
) -> None:                                 # P_in_i⊗V_i: d arrays F_in_i+W+(Ni,), each with len(order) jet axes
```
- `t3toolbox/backend/jets.py:1568-1574` `check_data`
```python
def check_data(
        kind:         str,                      # 'probe' | 'apply' | 'entries'
        order:        typ.Tuple[int, ...],      # the jet shape (n_1, ..., n_m); F = (n_1 + 1, ...). entries: ()
        sample:       typ.Any,                  # P_in_i⊗V_i (the input jets ww) | index^d, a good sample
        y:            typ.Any,                  # Y*, of the type of z: probe len=d (or (d,)+...), elm_shape=F+W+C+(Ni,); apply F+W+C; entries W+C
        stack_shape:  typ.Tuple[int, ...] = (), # C, the model's stack of base points; () for an unstacked model
) -> None:
```
- `t3toolbox/backend/jets.py:1657-1663` `check_weight`
```python
def check_weight(
        kind:         str,                      # 'probe' | 'apply' | 'entries'
        order:        typ.Tuple[int, ...],      # the jet shape (n_1, ..., n_m); F = (n_1 + 1, ...). entries: ()
        sample:       typ.Any,                  # P_in_i⊗V_i (the input jets ww) | index^d, a good sample
        weight:       typ.Optional[typ.Any],    # omega, of the type of z (each axis 1 or z's size), or None
        stack_shape:  typ.Tuple[int, ...] = (), # C, the model's stack of base points; () for an unstacked model
) -> None:
```

`fitting.LeastSquaresBatch` (the dataclass: class header and the field lines; `t3toolbox/backend/fitting.py`):
```python
@dataclass(frozen=True, eq=False)
class LeastSquaresBatch:
    sample:           typ.Any                                    # P_in_i⊗V_i (the input jets ww) | index^d (entries)
    y:                typ.Any = None                             # Y*: probe len=d, F+W+C+(Ni,); apply F+W+C; entries W+C
    weight:           typ.Any = None                             # omega, of the type of z likewise; None = ones
    minibatch_scale:  typ.Optional[typ.Any] = None               # gamma: n / batch_size on a drawn batch; None = 1
    drawn_indices:    typ.Optional[NDArray] = None               # int, (batch_size,): the draw, flat over W
```
(other statements in the class body: FunctionDef:__eq__@780, Assign:@783, FunctionDef:data_metric@785)

- `t3toolbox/backend/fitting.py:1069-1083` `LeastSquaresOracle.query`
```python
    def query(
            self,
            frame:            typ.Tuple,                      # a manifold frame or a corewise point (frame_form)
            batch:            LeastSquaresBatch,              # its data
            want:             typ.Sequence[str],              # f, g, JtJp, Jp_norm_sq; optional: optional_outputs
            Vc:               typ.Optional[typ.Sequence[NDArray]] = None,
            variations:       typ.Optional[typ.Tuple] = None, # X, (V, H): JtJp, Jp_norm_sq, Jp; raw on a point
            core_directions:  typ.Optional[typ.Tuple] = None, # X, (D_dot, L_dot): Kp
            held:             typ.Any = None,                 # a HeldVariables from an earlier query here, or None
    ) -> typ.Tuple[
            typ.Tuple[typ.Any, ...],                          # values, aligned with want: r, Jp Y*; g, JtJp, Kp X*
                                                              #   (before the projection); f, Jp_norm_sq, f_resolution
                                                              #   ℝ; r_block_sumsq, y_block_sumsq ℝ per block
            typ.Any,                                          # held: a HeldVariables
    ]:
```
- `t3toolbox/backend/fitting.py:1137-1142` `LeastSquaresOracle.check_batch`
```python
    def check_batch(
            self,
            batch:        LeastSquaresBatch,                  # ragged (check before packing)
            shape:        typ.Sequence[int],                  # the mode sizes (N0, ..., N(d-1))
            stack_shape:  typ.Tuple[int, ...] = (),           # C, the model's stack of base points; () unstacked
    ) -> None:
```
- `t3toolbox/backend/fitting.py:1190-1194` `LeastSquaresOracle.take`
```python
    def take(
            self,
            batch:    LeastSquaresBatch,
            indices:  NDArray,                                # int, (b,): flat positions over W; len(indices) static
    ) -> LeastSquaresBatch:                                   # the drawn batch, W flattened to (b,)
```
- `t3toolbox/backend/uniform_fitting.py:375-381` `pack_sample`
```python
def pack_sample(
        kind:    str,                       # 'probe' | 'apply' | 'entries'
        order:   typ.Tuple[int, ...],       # the jet shape: the number of leading jet axes of each input jet
        sample:  typ.Any,                   # P_in_i⊗V_i (the input jets ww, len=d, elm_shape=F_in_i+W+(Ni,)) | index^d
                                            #   ((d,)+W, int)
        N:       int,                       # the packed mode width
) -> typ.Any:                               # P_in_i⊗V_i packed (d,)+F_in+W+(N,) | index^d unchanged
```
- `t3toolbox/backend/uniform_fitting.py:398-405` `pack_data`
```python
def pack_data(
        kind:   str,                        # 'probe' | 'apply' | 'entries'
        order:  typ.Tuple[int, ...],        # the jet shape (the probe data's leading F axes ride along)
        y:      typ.Any,                    # Y* or omega, of the type of z (the data, a weight, a residual):
                                            #   probe len=d, elm_shape=F+W+C+(Ni,); apply F+W+C; entries W+C; each axis
                                            #   1 or z's size
        N:      int,                        # the packed mode width
) -> typ.Any:                               # probe: (d,)+F+W+C+(e,), e = 1 or N; apply, entries: y unchanged
```
- `t3toolbox/backend/uniform_fitting.py:431-436` `pack_batch`
```python
def pack_batch(
        kind:   str,                                # 'probe' | 'apply' | 'entries'
        order:  typ.Tuple[int, ...],                # the jet shape
        batch:  'bfit.LeastSquaresBatch',           # ragged (or already packed: kept)
        N:      int,                                # the packed mode width
) -> 'bfit.LeastSquaresBatch':                      # the sample's vectors and the probe data packed to N
```
- `t3toolbox/backend/ut3_operations.py:128-132` `pack_vectors`
```python
def pack_vectors(
        unpacked_vectors: typ.Sequence[NDArray],  # len=d, ith elm_shape=F_in_i+stack_shape+(Ni,)
        N: int = None,                            # padded length (default max(Ni))
        len_P: int = 0,                           # the number of leading jet axes (sizes may differ by mode)
) -> NDArray:                                     # packed, shape=(d,)+F_in+stack_shape+(N,), F_in the largest F_in_i
```
- `t3toolbox/backend/ut3_operations.py:179-182` `unpack_vectors`
```python
def unpack_vectors(
        packed_vectors:  NDArray,            # shape=(d,)+stack_shape+(N,)
        unpacking_shape: typ.Sequence[int],  # (N0,...,N(d-1))
) -> typ.Tuple[NDArray, ...]:                # len=d, ith elm_shape=stack_shape+(Ni,)
```
- `t3toolbox/backend/ut3_operations.py:191-193` `is_packed`
```python
def is_packed(
        vectors,  # a packed supercore array (single ndarray) OR a ragged len=d sequence of per-mode vectors
) -> bool:        # True if already packed (one array), False if a ragged list/tuple of d per-mode arrays
```
- `t3toolbox/backend/ut3_operations.py:200-204` `pack_if_ragged`
```python
def pack_if_ragged(
        vectors,        # packed array (returned as-is) OR ragged len=d sequence, ith elm_shape=F_in_i+stack+(Ni,)
        N:  int = None,  # padded length (default max Ni); ignored when ``vectors`` is already packed
        len_P: int = 0,  # the number of leading jet axes, brought to their largest size with zero coefficients
) -> NDArray:            # packed, shape=(d,)+F_in+stack_shape+(N,)
```
- `t3toolbox/backend/ut3_operations.py:211-214` `_pack_data_space`
```python
def _pack_data_space(
        value,          # a probe data-space object: packed (returned as is), or len=d arrays, elm_shape=F+W+C+(e_i,)
        N:  int,        # the packed mode width
) -> NDArray:           # (d,)+F+W+C+(e,): each axis of F+W+C the join across the modes; e = 1 if every e_i is 1, else N
```
- `t3toolbox/backend/sampling_variables.py:1678-1697` `sampling_variables`
```python
def sampling_variables(
        kind:              str,                              # 'probe' | 'apply' | 'entries'
        frame:             typ.Tuple,                        # the base point (D, U, L, R), four tuples of d arrays; or
                                                             #   the point (B, G): X
        sample:            typ.Any,                          # ww: P_in_i⊗V_i, the input jets, len=d,
                                                             #   elm_shape=F_in_i+W+(Ni,);
                                                             #   entries: index^d, shape=(d,)+W, int
        want:              typ.Sequence[str],                # variable names; the return is aligned with them

        order:             typ.Tuple[int, ...] = (),         # the jet shape (n_1, ..., n_m); F = (n_1 + 1, ...)
        y:                 typ.Optional[typ.Any] = None,     # the data: Y*, of the type of z, each axis 1 or z's size:
                                                             #   probe len=d, elm_shape=F+W+C+(Ni,); apply F+W+C; entries W+C
        weight:            typ.Optional[typ.Any] = None,     # omega, of the type of z likewise (no K); None = ones
        variations:        typ.Optional[typ.Tuple] = None,   # X, (V, H), elm_shapes K+C+(nUi,Ni), K+C+(rLi,nDi,rR(i+1))
        core_directions:   typ.Optional[typ.Tuple] = None,   # X, (D_dot, L_dot), the shapes of D, L with K+C
        Vc:                typ.Optional[typ.Sequence[NDArray]] = None,   # the frame's Tucker centers: len=d,
                                                                         #   elm_shape=C+(nUi,Ni)
        sum_over_samples:  bool = True,                      # on the blocks; False: W joins their stack, W+K+C
        held:              typ.Optional[HeldVariables] = None,   # held variables; shapes checked, values trusted
) -> typ.Tuple[typ.Any, ...]:                                # aligned with want; types and shapes in the module's table
```
- `t3toolbox/backend/sampling_variables.py:1556-1558` `frame_shape`
```python
def frame_shape(
        frame:  typ.Tuple,                  # any of the four forms (frame_form)
) -> typ.Tuple[int, ...]:                   # the mode sizes (N_0, ..., N_(d-1))
```
- `t3toolbox/backend/sampling_variables.py:1643-1648` `frame_form`
```python
def frame_form(
        frame:  typ.Tuple,                  # (D, U, L, R) or (B, G); uniform: + (shape, masks) after the cores
) -> typ.Tuple[
        str,                                # the layout: 'ragged' | 'uniform'
        str,                                # the call: 'manifold' (a frame) | 'corewise' (a point)
]:
```
- `t3toolbox/backend/fitting.py:858-860` `_stack_of_frame`
```python
def _stack_of_frame(
        frame:  typ.Tuple,                  # any of the four forms (sampling_variables.frame_form)
) -> typ.Tuple[int, ...]:                   # C, the stack of base points
```
- `t3toolbox/fitting.py:589-594` `prepare_batch`
```python
def prepare_batch(
        oracle,
        batch,
        x,                           # the point: its shape, and on the uniform layer its width N
        check:  bool = True,         # False: pack only (a draw's batch, made by the caller's draw every step)
):                                   # the batch the backend takes (packed on the uniform layer)
```
- `t3toolbox/fitting.py:582-582` `_is_packed`
```python
def _is_packed(oracle, batch) -> bool:
```
- `t3toolbox/fitting.py:648-657` `local_model`
```python
def local_model(
        geometry,                                  # MANIFOLD / COREWISE / UNIFORM_* / a SharedGeometry over one
        oracle,                                    # e.g. LeastSquaresOracle('probe', order=(3,))
        batch,                                     # e.g. LeastSquaresBatch(affine_jet(ww, pp), y, weight)
        x,                                         # the point: a TuckerTensorTrain or a UniformTuckerTensorTrain
        regularizer:        typ.Any = None,        # e.g. IdentityRegularizer(lam)
        hessian:            str = 'gauss_newton',  # or 'exact'
        curvature_damping:  typ.Optional[float] = None,   # exact only; None = the geometry's default
        frame:              typ.Any = None,        # manifold only: a T3Frame / UT3Frame of x the caller supplies (O-5)
) -> LocalModel:                                   # tangents in and out, the same-frame guard
```
- `t3toolbox/optimizers.py:117-117` `_frontend_draw`
```python
def _frontend_draw(oracle, draw, x0):
```
- `t3toolbox/optimizers.py:103-103` `_setup`
```python
def _setup(geometry, oracle, batch, x0) -> _Setup:
```


## M3. Packing and checking call sites

Method: library/tests/examples = `ast.Call` nodes whose callee is a bare Name or an Attribute with exactly this name (any receiver; `_is_packed` and `check_sample`-prefixed names are different names and are not counted); "library doctests" = regex `name(` on `>>>`/`...` lines inside library docstrings; docs = regex `name(` per line of `docs/*.md|rst`, `docs/contributor/*.md`, split doctest/fenced/prose. Definitions are not calls. Non-call references (name passed as a value, imported) are listed separately per name as "other references" for the library (ast Name/Attribute/ImportFrom, excluding calls and def sites).

**`pack_vectors`**
- library: 1 calls -- `t3toolbox/backend/ut3_operations.py` 1
- tests: 16 calls -- `tests/backend/test_curvature.py` 5, `tests/backend/test_uniform_fitting.py` 4, `tests/test_dispatch.py` 1, `tests/test_frontend_sampling.py` 1, `tests/test_sampling_functions.py` 1, `tests/test_sampling_uniform.py` 4
- examples: 0 calls
- library doctests: 3 -- `t3toolbox/backend/jets.py` 2, `t3toolbox/backend/ut3_operations.py` 1
- docs pages: 0 lines
- library call sites:
  - `t3toolbox/backend/ut3_operations.py:208` | `pack_if_ragged`
- library other references (not calls): none

**`unpack_vectors`**
- library: 6 calls -- `t3toolbox/backend/fitting.py` 1, `t3toolbox/backend/ut3_sampling.py` 3, `t3toolbox/backend/utv_sampling.py` 2
- tests: 5 calls -- `tests/backend/test_uniform_fitting.py` 4, `tests/test_frontend_sampling.py` 1
- examples: 0 calls
- library doctests: 0
- docs pages: 0 lines
- library call sites:
  - `t3toolbox/backend/fitting.py:855` | `_arrays_of` | recv `ut3_operations`
  - `t3toolbox/backend/ut3_sampling.py:109` | `ut3_probe` | recv `ut3_operations`
  - `t3toolbox/backend/ut3_sampling.py:232` | `ut3_probe_corewise_tangent` | recv `ut3_operations`
  - `t3toolbox/backend/ut3_sampling.py:278` | `ut3_probe_derivatives` | recv `ut3_operations`
  - `t3toolbox/backend/utv_sampling.py:102` | `utv_probe` | recv `ut3_operations`
  - `t3toolbox/backend/utv_sampling.py:411` | `utv_probe_derivatives` | recv `ut3_operations`
- library other references (not calls): none

**`is_packed`**
- library: 8 calls -- `t3toolbox/backend/ut3_operations.py` 2, `t3toolbox/backend/ut3_sampling.py` 3, `t3toolbox/backend/utv_sampling.py` 2, `t3toolbox/uniform_tucker_tensor_train.py` 1
- tests: 3 calls -- `tests/_derivative_data.py` 1, `tests/test_frontend_sampling.py` 1, `tests/test_least_squares_oracle.py` 1
- examples: 0 calls
- library doctests: 0
- docs pages: 0 lines
- library call sites:
  - `t3toolbox/backend/ut3_operations.py:208` | `pack_if_ragged`
  - `t3toolbox/backend/ut3_operations.py:221` | `_pack_data_space`
  - `t3toolbox/backend/ut3_sampling.py:107` | `ut3_probe` | recv `ut3_operations`
  - `t3toolbox/backend/ut3_sampling.py:229` | `ut3_probe_corewise_tangent` | recv `ut3_operations`
  - `t3toolbox/backend/ut3_sampling.py:274` | `ut3_probe_derivatives` | recv `ut3_operations`
  - `t3toolbox/backend/utv_sampling.py:99` | `utv_probe` | recv `ut3_operations`
  - `t3toolbox/backend/utv_sampling.py:407` | `utv_probe_derivatives` | recv `ut3_operations`
  - `t3toolbox/uniform_tucker_tensor_train.py:60` | `_check_sample` | recv `ut3_operations`
- library other references (not calls): none

**`pack_if_ragged`**
- library: 45 calls -- `t3toolbox/backend/data_preparation.py` 1, `t3toolbox/backend/sampling_variables.py` 1, `t3toolbox/backend/uniform_fitting.py` 1, `t3toolbox/backend/ut3_sampling.py` 21, `t3toolbox/backend/utv_sampling.py` 21
- tests: 21 calls -- `tests/backend/test_uniform_curvature.py` 9, `tests/backend/test_uniform_fitting.py` 2, `tests/test_dispatch.py` 5, `tests/test_sampling_uniform.py` 4, `tests/test_uniform_probing.py` 1
- examples: 0 calls
- library doctests: 1 -- `t3toolbox/backend/ut3_sampling.py` 1
- docs pages: 0 lines
- library call sites:
  - `t3toolbox/backend/data_preparation.py:641` | `ut3_unscale_coordinates` | recv `ut3_operations`
  - `t3toolbox/backend/sampling_variables.py:1870` | `sampling_variables` | recv `ut3_operations`
  - `t3toolbox/backend/uniform_fitting.py:395` | `pack_sample` | recv `ut3_operations`
  - `t3toolbox/backend/ut3_sampling.py:250` | `ut3_probe_corewise_transpose` | recv `ut3_operations`
  - `t3toolbox/backend/ut3_sampling.py:275` | `ut3_probe_derivatives` | recv `ut3_operations`
  - `t3toolbox/backend/ut3_sampling.py:276` | `ut3_probe_derivatives` | recv `ut3_operations`
  - `t3toolbox/backend/ut3_sampling.py:291` | `ut3_apply_derivatives` | recv `ut3_operations`
  - `t3toolbox/backend/ut3_sampling.py:292` | `ut3_apply_derivatives` | recv `ut3_operations`
  - `t3toolbox/backend/ut3_sampling.py:306` | `ut3_entries_derivatives` | recv `ut3_operations`
  - `t3toolbox/backend/ut3_sampling.py:328` | `ut3_apply_corewise_derivatives_transpose` | recv `ut3_operations`
  - `t3toolbox/backend/ut3_sampling.py:329` | `ut3_apply_corewise_derivatives_transpose` | recv `ut3_operations`
  - `t3toolbox/backend/ut3_sampling.py:346` | `ut3_entries_corewise_derivatives_transpose` | recv `ut3_operations`
  - `t3toolbox/backend/ut3_sampling.py:365` | `ut3_probe_corewise_derivatives_transpose` | recv `ut3_operations`
  - `t3toolbox/backend/ut3_sampling.py:366` | `ut3_probe_corewise_derivatives_transpose` | recv `ut3_operations`
  - `t3toolbox/backend/ut3_sampling.py:367` | `ut3_probe_corewise_derivatives_transpose` | recv `ut3_operations`
  - `t3toolbox/backend/ut3_sampling.py:434` | `ut3_probe_corewise_derivatives_hessian` | recv `ut3_operations`
  - `t3toolbox/backend/ut3_sampling.py:435` | `ut3_probe_corewise_derivatives_hessian` | recv `ut3_operations`
  - `t3toolbox/backend/ut3_sampling.py:436` | `ut3_probe_corewise_derivatives_hessian` | recv `ut3_operations`
  - `t3toolbox/backend/ut3_sampling.py:463` | `ut3_probe_corewise_derivatives_hessian_from_sweep` | recv `ut3_operations`
  - `t3toolbox/backend/ut3_sampling.py:536` | `ut3_probe_corewise_hessian` | recv `ut3_operations`
  - `t3toolbox/backend/ut3_sampling.py:559` | `ut3_probe_corewise_hessian_from_sweep` | recv `ut3_operations`
  - `t3toolbox/backend/ut3_sampling.py:688` | `ut3_apply_corewise_derivatives_hessian` | recv `ut3_operations`
  - `t3toolbox/backend/ut3_sampling.py:688` | `ut3_apply_corewise_derivatives_hessian` | recv `ut3_operations`
  - `t3toolbox/backend/ut3_sampling.py:727` | `ut3_entries_corewise_derivatives_hessian` | recv `ut3_operations`
  - `t3toolbox/backend/utv_sampling.py:158` | `utv_precompute_apply_frame_sweep` | recv `ut3_operations`
  - `t3toolbox/backend/utv_sampling.py:169` | `utv_precompute_probe_frame_sweep` | recv `ut3_operations`
  - `t3toolbox/backend/utv_sampling.py:237` | `utv_probe_transpose_from_sweep` | recv `ut3_operations`
  - `t3toolbox/backend/utv_sampling.py:274` | `utv_precompute_apply_frame_sweep_jets` | recv `ut3_operations`
  - `t3toolbox/backend/utv_sampling.py:275` | `utv_precompute_apply_frame_sweep_jets` | recv `ut3_operations`
  - `t3toolbox/backend/utv_sampling.py:290` | `utv_precompute_probe_frame_sweep_jets` | recv `ut3_operations`
  - `t3toolbox/backend/utv_sampling.py:291` | `utv_precompute_probe_frame_sweep_jets` | recv `ut3_operations`
  - `t3toolbox/backend/utv_sampling.py:305` | `utv_precompute_entries_frame_sweep_jets` | recv `ut3_operations`
  - `t3toolbox/backend/utv_sampling.py:370` | `utv_probe_transpose_derivatives_from_sweep` | recv `ut3_operations`
  - `t3toolbox/backend/utv_sampling.py:408` | `utv_probe_derivatives` | recv `ut3_operations`
  - `t3toolbox/backend/utv_sampling.py:409` | `utv_probe_derivatives` | recv `ut3_operations`
  - `t3toolbox/backend/utv_sampling.py:425` | `utv_apply_derivatives` | recv `ut3_operations`
  - `t3toolbox/backend/utv_sampling.py:426` | `utv_apply_derivatives` | recv `ut3_operations`
  - `t3toolbox/backend/utv_sampling.py:440` | `utv_entries_derivatives` | recv `ut3_operations`
  - `t3toolbox/backend/utv_sampling.py:486` | `utv_probe_transpose` | recv `ut3_operations`
  - `t3toolbox/backend/utv_sampling.py:550` | `utv_probe_derivatives_transpose` | recv `ut3_operations`
  - `t3toolbox/backend/utv_sampling.py:551` | `utv_probe_derivatives_transpose` | recv `ut3_operations`
  - `t3toolbox/backend/utv_sampling.py:552` | `utv_probe_derivatives_transpose` | recv `ut3_operations`
  - `t3toolbox/backend/utv_sampling.py:570` | `utv_apply_derivatives_transpose` | recv `ut3_operations`
  - `t3toolbox/backend/utv_sampling.py:571` | `utv_apply_derivatives_transpose` | recv `ut3_operations`
  - `t3toolbox/backend/utv_sampling.py:588` | `utv_entries_derivatives_transpose` | recv `ut3_operations`
- library other references (not calls): none

**`pack_sample`**
- library: 1 calls -- `t3toolbox/backend/uniform_fitting.py` 1
- tests: 3 calls -- `tests/backend/test_uniform_fitting.py` 3
- examples: 0 calls
- library doctests: 1 -- `t3toolbox/backend/uniform_fitting.py` 1
- docs pages: 0 lines
- library call sites:
  - `t3toolbox/backend/uniform_fitting.py:455` | `pack_batch`
- library other references (not calls): none

**`pack_data`**
- library: 2 calls -- `t3toolbox/backend/uniform_fitting.py` 2
- tests: 4 calls -- `tests/backend/test_uniform_fitting.py` 2, `tests/test_oracle_data_space.py` 2
- examples: 0 calls
- library doctests: 3 -- `t3toolbox/backend/uniform_fitting.py` 3
- docs pages: 0 lines
- library call sites:
  - `t3toolbox/backend/uniform_fitting.py:453` | `pack_batch`
  - `t3toolbox/backend/uniform_fitting.py:454` | `pack_batch`
- library other references (not calls): none

**`pack_batch`**
- library: 1 calls -- `t3toolbox/fitting.py` 1
- tests: 24 calls -- `tests/_fits.py` 1, `tests/backend/test_optimizer_display.py` 1, `tests/backend/test_uniform_curvature.py` 2, `tests/backend/test_uniform_fitting.py` 1, `tests/test_aligned_padded_frame_frontend.py` 1, `tests/test_block_sums.py` 1, `tests/test_dispatch.py` 1, `tests/test_fitting.py` 1, `tests/test_least_squares_oracle.py` 4, `tests/test_optimizers_frontend.py` 2, `tests/test_oracle_data_space.py` 9
- examples: 0 calls
- library doctests: 1 -- `t3toolbox/backend/uniform_fitting.py` 1
- docs pages: 0 lines
- library call sites:
  - `t3toolbox/fitting.py:608` | `prepare_batch` | recv `ufit`
- library other references (not calls): none

**`check_sample`**
- library: 23 calls -- `t3toolbox/backend/fitting.py` 1, `t3toolbox/manifold.py` 6, `t3toolbox/tucker_tensor_train.py` 15, `t3toolbox/uniform_tucker_tensor_train.py` 1
- tests: 9 calls -- `tests/test_dispatch.py` 1, `tests/test_jets.py` 8
- examples: 0 calls
- library doctests: 7 -- `t3toolbox/backend/jets.py` 7
- docs pages: 0 lines
- library call sites:
  - `t3toolbox/backend/fitting.py:1161` | `LeastSquaresOracle.check_batch` | recv `jets`
  - `t3toolbox/manifold.py:747` | `T3Tangent.probe` | recv `jets`
  - `t3toolbox/manifold.py:832` | `T3Tangent.probe_transpose` | recv `jets`
  - `t3toolbox/manifold.py:873` | `T3Tangent.apply` | recv `jets`
  - `t3toolbox/manifold.py:907` | `T3Tangent.entries` | recv `jets`
  - `t3toolbox/manifold.py:952` | `T3Tangent.apply_transpose` | recv `jets`
  - `t3toolbox/manifold.py:977` | `T3Tangent.entries_transpose` | recv `jets`
  - `t3toolbox/tucker_tensor_train.py:3565` | `TuckerTensorTrain.entries` | recv `jets`
  - `t3toolbox/tucker_tensor_train.py:3657` | `TuckerTensorTrain.apply` | recv `jets`
  - `t3toolbox/tucker_tensor_train.py:3755` | `TuckerTensorTrain.probe` | recv `jets`
  - `t3toolbox/tucker_tensor_train.py:3814` | `TuckerTensorTrain.probe_ambient_transpose` | recv `jets`
  - `t3toolbox/tucker_tensor_train.py:3867` | `TuckerTensorTrain.probe_corewise_transpose` | recv `jets`
  - `t3toolbox/tucker_tensor_train.py:3929` | `TuckerTensorTrain.apply_ambient_transpose` | recv `jets`
  - `t3toolbox/tucker_tensor_train.py:3977` | `TuckerTensorTrain.entries_ambient_transpose` | recv `jets`
  - `t3toolbox/tucker_tensor_train.py:4037` | `TuckerTensorTrain.apply_corewise_transpose` | recv `jets`
  - `t3toolbox/tucker_tensor_train.py:4066` | `TuckerTensorTrain.entries_corewise_transpose` | recv `jets`
  - `t3toolbox/tucker_tensor_train.py:4107` | `TuckerTensorTrain.probe_corewise_tangent` | recv `jets`
  - `t3toolbox/tucker_tensor_train.py:4118` | `TuckerTensorTrain.apply_corewise_tangent` | recv `jets`
  - `t3toolbox/tucker_tensor_train.py:4129` | `TuckerTensorTrain.entries_corewise_tangent` | recv `jets`
  - `t3toolbox/tucker_tensor_train.py:4175` | `TuckerTensorTrain.probe_corewise_hessian` | recv `jets`
  - `t3toolbox/tucker_tensor_train.py:4192` | `TuckerTensorTrain.apply_corewise_hessian` | recv `jets`
  - `t3toolbox/tucker_tensor_train.py:4209` | `TuckerTensorTrain.entries_corewise_hessian` | recv `jets`
  - `t3toolbox/uniform_tucker_tensor_train.py:61` | `_check_sample` | recv `jets`
- library other references (not calls): none

**`_check_sample`**
- library: 18 calls -- `t3toolbox/uniform_manifold.py` 6, `t3toolbox/uniform_tucker_tensor_train.py` 12
- tests: 0 calls
- examples: 0 calls
- library doctests: 0
- docs pages: 0 lines
- library call sites:
  - `t3toolbox/uniform_manifold.py:581` | `UT3Tangent.probe` | recv `ut3`
  - `t3toolbox/uniform_manifold.py:607` | `UT3Tangent.apply` | recv `ut3`
  - `t3toolbox/uniform_manifold.py:630` | `UT3Tangent.entries` | recv `ut3`
  - `t3toolbox/uniform_manifold.py:677` | `UT3Tangent.probe_transpose` | recv `ut3`
  - `t3toolbox/uniform_manifold.py:695` | `UT3Tangent.apply_transpose` | recv `ut3`
  - `t3toolbox/uniform_manifold.py:712` | `UT3Tangent.entries_transpose` | recv `ut3`
  - `t3toolbox/uniform_tucker_tensor_train.py:421` | `UniformTuckerTensorTrain.entries`
  - `t3toolbox/uniform_tucker_tensor_train.py:447` | `UniformTuckerTensorTrain.apply`
  - `t3toolbox/uniform_tucker_tensor_train.py:484` | `UniformTuckerTensorTrain.probe`
  - `t3toolbox/uniform_tucker_tensor_train.py:528` | `UniformTuckerTensorTrain.apply_corewise_transpose`
  - `t3toolbox/uniform_tucker_tensor_train.py:542` | `UniformTuckerTensorTrain.entries_corewise_transpose`
  - `t3toolbox/uniform_tucker_tensor_train.py:557` | `UniformTuckerTensorTrain.probe_corewise_transpose`
  - `t3toolbox/uniform_tucker_tensor_train.py:575` | `UniformTuckerTensorTrain.probe_corewise_tangent`
  - `t3toolbox/uniform_tucker_tensor_train.py:593` | `UniformTuckerTensorTrain.probe_corewise_hessian`
  - `t3toolbox/uniform_tucker_tensor_train.py:610` | `UniformTuckerTensorTrain.apply_corewise_tangent`
  - `t3toolbox/uniform_tucker_tensor_train.py:628` | `UniformTuckerTensorTrain.apply_corewise_hessian`
  - `t3toolbox/uniform_tucker_tensor_train.py:644` | `UniformTuckerTensorTrain.entries_corewise_tangent`
  - `t3toolbox/uniform_tucker_tensor_train.py:661` | `UniformTuckerTensorTrain.entries_corewise_hessian`
- library other references (not calls): none

**`check_batch`**
- library: 1 calls -- `t3toolbox/fitting.py` 1
- tests: 6 calls -- `tests/test_least_squares_oracle.py` 2, `tests/test_oracle_data_space.py` 4
- examples: 0 calls
- library doctests: 2 -- `t3toolbox/backend/fitting.py` 2
- docs pages: 0 lines
- library call sites:
  - `t3toolbox/fitting.py:606` | `prepare_batch` | recv `oracle`
- library other references (not calls): none

**`prepare_batch`**
- library: 4 calls -- `t3toolbox/fitting.py` 1, `t3toolbox/optimizers.py` 3
- tests: 2 calls -- `tests/test_optimizers_frontend.py` 2
- examples: 0 calls
- library doctests: 0
- docs pages: 0 lines
- library call sites:
  - `t3toolbox/fitting.py:699` | `local_model`
  - `t3toolbox/optimizers.py:108` | `_setup` | recv `_fitting`
  - `t3toolbox/optimizers.py:122` | `_frontend_draw` | recv `_fitting`
  - `t3toolbox/optimizers.py:841` | `newton_cg` | recv `_fitting`
- library other references (not calls): none

**`_frontend_draw`**
- library: 2 calls -- `t3toolbox/optimizers.py` 2
- tests: 0 calls
- examples: 0 calls
- library doctests: 0
- docs pages: 0 lines
- library call sites:
  - `t3toolbox/optimizers.py:347` | `mc_sgd`
  - `t3toolbox/optimizers.py:375` | `adam`
- library other references (not calls): none

**`_as_index`**
- library: 13 calls -- `t3toolbox/manifold.py` 2, `t3toolbox/tucker_tensor_train.py` 5, `t3toolbox/uniform_manifold.py` 2, `t3toolbox/uniform_tucker_tensor_train.py` 4
- tests: 0 calls
- examples: 0 calls
- library doctests: 0
- docs pages: 0 lines
- library call sites:
  - `t3toolbox/manifold.py:906` | `T3Tangent.entries` | recv `t3`
  - `t3toolbox/manifold.py:976` | `T3Tangent.entries_transpose` | recv `t3`
  - `t3toolbox/tucker_tensor_train.py:3564` | `TuckerTensorTrain.entries`
  - `t3toolbox/tucker_tensor_train.py:3976` | `TuckerTensorTrain.entries_ambient_transpose`
  - `t3toolbox/tucker_tensor_train.py:4065` | `TuckerTensorTrain.entries_corewise_transpose`
  - `t3toolbox/tucker_tensor_train.py:4128` | `TuckerTensorTrain.entries_corewise_tangent`
  - `t3toolbox/tucker_tensor_train.py:4208` | `TuckerTensorTrain.entries_corewise_hessian`
  - `t3toolbox/uniform_manifold.py:629` | `UT3Tangent.entries` | recv `t3`
  - `t3toolbox/uniform_manifold.py:711` | `UT3Tangent.entries_transpose` | recv `t3`
  - `t3toolbox/uniform_tucker_tensor_train.py:420` | `UniformTuckerTensorTrain.entries` | recv `t3`
  - `t3toolbox/uniform_tucker_tensor_train.py:541` | `UniformTuckerTensorTrain.entries_corewise_transpose` | recv `t3`
  - `t3toolbox/uniform_tucker_tensor_train.py:643` | `UniformTuckerTensorTrain.entries_corewise_tangent` | recv `t3`
  - `t3toolbox/uniform_tucker_tensor_train.py:660` | `UniformTuckerTensorTrain.entries_corewise_hessian` | recv `t3`
- library other references (not calls): none

**`_sample_shape`**
- library: 2 calls -- `t3toolbox/tucker_tensor_train.py` 2
- tests: 0 calls
- examples: 0 calls
- library doctests: 0
- docs pages: 0 lines
- library call sites:
  - `t3toolbox/tucker_tensor_train.py:3814` | `TuckerTensorTrain.probe_ambient_transpose`
  - `t3toolbox/tucker_tensor_train.py:3929` | `TuckerTensorTrain.apply_ambient_transpose`
- library other references (not calls): none

**`index_to_one_hot`**
- library: 2 calls -- `t3toolbox/backend/sampling_steps.py` 1, `t3toolbox/optimizers.py` 1
- tests: 20 calls -- `tests/_derivative_data.py` 1, `tests/_fits.py` 1, `tests/bridge/test_old_kernels_bridge.py` 1, `tests/test_dispatch.py` 2, `tests/test_least_squares_oracle.py` 1, `tests/test_optimizers_frontend.py` 1, `tests/test_sampling_definitions.py` 9, `tests/test_sampling_entries.py` 2, `tests/test_sampling_stacks.py` 1, `tests/test_sampling_uniform.py` 1
- examples: 0 calls
- library doctests: 4 -- `t3toolbox/backend/entries.py` 1, `t3toolbox/backend/jets.py` 2, `t3toolbox/backend/sampling_variables.py` 1
- docs pages: 0 lines
- library call sites:
  - `t3toolbox/backend/sampling_steps.py:861` | `w_one_hot_step` | recv `jets`
  - `t3toolbox/optimizers.py:214` | `_old_fit` | recv `jets`
- library other references (not calls): `t3toolbox/__init__.py:39` `<module>` importfrom (`t3toolbox.fitting`); `t3toolbox/fitting.py:79` `<module>` importfrom (`t3toolbox.backend.jets`)

**`_uniform_index`**
- library: 1 calls -- `t3toolbox/backend/sampling_variables.py` 1
- tests: 0 calls
- examples: 0 calls
- library doctests: 0
- docs pages: 0 lines
- library call sites:
  - `t3toolbox/backend/sampling_variables.py:1868` | `sampling_variables`
- library other references (not calls): none


## M4. Where the library reads a container's type to tell forms apart

Method: `ast.walk` over `Call` nodes (so docstrings, comments and doctests are excluded by construction) whose callee is Name `isinstance` or Name/Attribute named `is_ndarray` or `is_packed`. For `isinstance`, the second argument is classified: `(tuple, list)` (a tuple of exactly Name tuple and Name list, either order), `tuple`, `list`; any other `isinstance` (other classes, e.g. `dict`, `int`, `np.ndarray`, `str`, T3 classes) is NOT listed here but counted per file under "other isinstance". Row format: `file:line | enclosing function | the source line`. Lines with several matching calls appear once per call.

**Totals per file** (listed kinds | other isinstance not listed):

| file | total listed | isinstance(tuple,list) | isinstance(tuple) | isinstance(list) | is_ndarray( | is_packed( | other isinstance |
|---|---|---|---|---|---|---|---|
| `t3toolbox/backend/sampling_variables.py` | 11 | 10 | 0 | 0 | 1 | 0 | 3 |
| `t3toolbox/backend/sampling_steps.py` | 5 | 1 | 0 | 0 | 4 | 0 | 0 |
| `t3toolbox/backend/jets.py` | 18 | 13 | 0 | 0 | 5 | 0 | 0 |
| `t3toolbox/backend/data_preparation.py` | 3 | 3 | 0 | 0 | 0 | 0 | 0 |
| `t3toolbox/backend/fitting.py` | 12 | 11 | 1 | 0 | 0 | 0 | 2 |
| `t3toolbox/backend/uniform_fitting.py` | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `t3toolbox/backend/ut3_sampling.py` | 3 | 0 | 0 | 0 | 0 | 3 | 0 |
| `t3toolbox/backend/utv_sampling.py` | 2 | 0 | 0 | 0 | 0 | 2 | 0 |
| `t3toolbox/backend/ut3_operations.py` | 4 | 1 | 0 | 0 | 1 | 2 | 0 |
| `t3toolbox/backend/probing.py` | 10 | 0 | 0 | 0 | 10 | 0 | 11 |
| `t3toolbox/backend/apply.py` | 2 | 0 | 0 | 0 | 2 | 0 | 1 |
| `t3toolbox/backend/entries.py` | 2 | 0 | 0 | 0 | 2 | 0 | 0 |
| `t3toolbox/backend/optimizers.py` | 5 | 1 | 1 | 0 | 3 | 0 | 9 |
| `t3toolbox/backend/optimizer_display.py` | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `t3toolbox/fitting.py` | 4 | 4 | 0 | 0 | 0 | 0 | 18 |
| `t3toolbox/optimizers.py` | 0 | 0 | 0 | 0 | 0 | 0 | 8 |
| `t3toolbox/tucker_tensor_train.py` | 9 | 3 | 0 | 0 | 6 | 0 | 6 |
| `t3toolbox/uniform_tucker_tensor_train.py` | 3 | 0 | 0 | 0 | 2 | 1 | 1 |
| `t3toolbox/manifold.py` | 0 | 0 | 0 | 0 | 0 | 0 | 2 |
| `t3toolbox/uniform_manifold.py` | 0 | 0 | 0 | 0 | 0 | 0 | 2 |

Grand total listed: 93 ({'isinstance(tuple,list)': 47, 'is_ndarray(': 36, 'isinstance(tuple)': 2, 'is_packed(': 8})

**`t3toolbox/backend/sampling_variables.py`** (11)

- `t3toolbox/backend/sampling_variables.py:1187` | `_check_data_space` | isinstance(tuple,list) | `if isinstance(value, (tuple, list)):`
- `t3toolbox/backend/sampling_variables.py:1193` | `_check_data_space` | isinstance(tuple,list) | `count = len(value) if isinstance(value, (tuple, list)) else (np.shape(value)[0] if np.ndim(value) else 0)`
- `t3toolbox/backend/sampling_variables.py:1274` | `_check_form` | isinstance(tuple,list) | `if isinstance(value, (tuple, list)) or tuple(np.shape(value)) != expected:`
- `t3toolbox/backend/sampling_variables.py:1276` | `_check_form` | isinstance(tuple,list) | `% (where, type(value).__name__ if isinstance(value, (tuple, list)) else`
- `t3toolbox/backend/sampling_variables.py:1279` | `_check_form` | isinstance(tuple,list) | `if not isinstance(value, (tuple, list)) or len(value) != len(expected):`
- `t3toolbox/backend/sampling_variables.py:1281` | `_check_form` | isinstance(tuple,list) | `% (where, len(expected), type(value).__name__ if not isinstance(value, (tuple, list))`
- `t3toolbox/backend/sampling_variables.py:1383` | `_leaves` | isinstance(tuple,list) | `if isinstance(value, (tuple, list)):`
- `t3toolbox/backend/sampling_variables.py:1392` | `_cast` | isinstance(tuple,list) | `if isinstance(value, (tuple, list)):`
- `t3toolbox/backend/sampling_variables.py:1666` | `frame_form` | isinstance(tuple,list) | `if not isinstance(frame, (tuple, list)) or not frame:`
- `t3toolbox/backend/sampling_variables.py:1668` | `frame_form` | is_ndarray( | `uniform = is_ndarray(frame[0])                      # the layout first, then the count`
- `t3toolbox/backend/sampling_variables.py:1883` | `sampling_variables` | isinstance(tuple,list) | `elif isinstance(given, (tuple, list)):`

**`t3toolbox/backend/sampling_steps.py`** (5)

- `t3toolbox/backend/sampling_steps.py:862` | `w_one_hot_step` | is_ndarray( | `if is_ndarray(D):                                   # the uniform layout: the padded N, packed (the index is real)`
- `t3toolbox/backend/sampling_steps.py:943` | `_with_K` | isinstance(tuple,list) | `if kind == 'probe' and isinstance(weight, (tuple, list)):`
- `t3toolbox/backend/sampling_steps.py:1027` | `w_tilde_step` | is_ndarray( | `if is_ndarray(r):                                   # the uniform layout: packed in, packed out`
- `t3toolbox/backend/sampling_steps.py:1043` | `g_step` | is_ndarray( | `if is_ndarray(V_tilde):`
- `t3toolbox/backend/sampling_steps.py:1061` | `r_step` | is_ndarray( | `if kind != 'probe' or is_ndarray(z):`

**`t3toolbox/backend/jets.py`** (18)

- `t3toolbox/backend/jets.py:305` | `_mode_weights` | isinstance(tuple,list) | `if isinstance(weight, (tuple, list)):`
- `t3toolbox/backend/jets.py:309` | `_mode_weights` | isinstance(tuple,list) | `if isinstance(weight, (tuple, list)):`
- `t3toolbox/backend/jets.py:730` | `data_metric` | isinstance(tuple,list) | `ragged = isinstance(r, (tuple, list))`
- `t3toolbox/backend/jets.py:735` | `data_metric` | isinstance(tuple,list) | `elif isinstance(weight, (tuple, list)):`
- `t3toolbox/backend/jets.py:747` | `data_metric` | isinstance(tuple,list) | `if isinstance(weight, (tuple, list)):`
- `t3toolbox/backend/jets.py:825` | `blocks_to_weight` | isinstance(tuple,list) | `if isinstance(y, (tuple, list)):`
- `t3toolbox/backend/jets.py:1075` | `_taylor_weights` | isinstance(tuple,list) | `if kind == 'probe' and isinstance(weight, (tuple, list)):`
- `t3toolbox/backend/jets.py:1286` | `affine_jet` | is_ndarray( | `if is_ndarray(ww):`
- `t3toolbox/backend/jets.py:1288` | `affine_jet` | is_ndarray( | `if not is_ndarray(p) or tuple(p.shape) != tuple(ww.shape):`
- `t3toolbox/backend/jets.py:1290` | `affine_jet` | is_ndarray( | `% (j, tuple(ww.shape), tuple(p.shape) if is_ndarray(p) else type(p).__name__))`
- `t3toolbox/backend/jets.py:1294` | `affine_jet` | is_ndarray( | `if is_ndarray(p) or len(p) != d:`
- `t3toolbox/backend/jets.py:1365` | `index_to_one_hot` | is_ndarray( | `if not is_ndarray(index):`
- `t3toolbox/backend/jets.py:1399` | `_require_input_jets` | isinstance(tuple,list) | `if not isinstance(ww, (tuple, list)):`
- `t3toolbox/backend/jets.py:1405` | `_require_input_jets` | isinstance(tuple,list) | `if isinstance(w, (tuple, list)):`
- `t3toolbox/backend/jets.py:1514` | `check_sample` | isinstance(tuple,list) | `if isinstance(sample, (tuple, list)):`
- `t3toolbox/backend/jets.py:1544` | `_check_of_type_of_z` | isinstance(tuple,list) | `if isinstance(value, (tuple, list)):`
- `t3toolbox/backend/jets.py:1552` | `_check_of_type_of_z` | isinstance(tuple,list) | `isinstance(value, (tuple, list)) else '%d arrays' % count))`
- `t3toolbox/backend/jets.py:1556` | `_check_of_type_of_z` | isinstance(tuple,list) | `if isinstance(value, (tuple, list)):`

**`t3toolbox/backend/data_preparation.py`** (3)

- `t3toolbox/backend/data_preparation.py:712` | `check_coordinate_scales` | isinstance(tuple,list) | `if not isinstance(scales, (tuple, list)):`
- `t3toolbox/backend/data_preparation.py:719` | `check_coordinate_scales` | isinstance(tuple,list) | `if isinstance(scale, (tuple, list)) or tuple(np.shape(scale)) != (N,):`
- `t3toolbox/backend/data_preparation.py:720` | `check_coordinate_scales` | isinstance(tuple,list) | `got = 'a %s' % type(scale).__name__ if isinstance(scale, (tuple, list)) else 'shape %s' % (tuple(np.shape(scale)),)`

**`t3toolbox/backend/fitting.py`** (12)

- `t3toolbox/backend/fitting.py:92` | `sumsq_over_probes` | isinstance(tuple,list) | `if not isinstance(zz, (list, tuple)):               # packed (d,)+W+C+(N,): d (axis 0) + W + N, keep C`
- `t3toolbox/backend/fitting.py:141` | `block_sumsq_over_probes` | isinstance(tuple,list) | `if not isinstance(zz, (list, tuple)):               # packed (d,)+[order]+W+C+(N,)`
- `t3toolbox/backend/fitting.py:277` | `_make_weight.apply_w` | isinstance(tuple,list) | `if isinstance(x, (list, tuple)):                 # ragged probe: a list of d order-leading arrays`
- `t3toolbox/backend/fitting.py:826` | `_same_leaves` | isinstance(tuple,list) | `if isinstance(a, (tuple, list)) and isinstance(b, (tuple, list)):`
- `t3toolbox/backend/fitting.py:826` | `_same_leaves` | isinstance(tuple,list) | `if isinstance(a, (tuple, list)) and isinstance(b, (tuple, list)):`
- `t3toolbox/backend/fitting.py:835` | `_scale_tree` | isinstance(tuple,list) | `if isinstance(value, (tuple, list)):`
- `t3toolbox/backend/fitting.py:853` | `_arrays_of` | isinstance(tuple,list) | `if isinstance(value, (tuple, list)):`
- `t3toolbox/backend/fitting.py:974` | `_take_data_space` | isinstance(tuple,list) | `if kind == 'probe' and isinstance(value, (tuple, list)):`
- `t3toolbox/backend/fitting.py:1060` | `LeastSquaresOracle.__post_init__` | isinstance(tuple) | `if not isinstance(self.order, tuple) or not all(isinstance(n, (int, np.integer)) and not isinstance(n, bool)`
- `t3toolbox/backend/fitting.py:1172` | `LeastSquaresOracle._w_axes` | isinstance(tuple,list) | `if isinstance(batch.sample, (tuple, list)):`
- `t3toolbox/backend/fitting.py:1179` | `LeastSquaresOracle._W` | isinstance(tuple,list) | `lead = batch.sample if self.kind == 'entries' or not isinstance(batch.sample, (tuple, list)) \`
- `t3toolbox/backend/fitting.py:1231` | `LeastSquaresOracle.take` | isinstance(tuple,list) | `elif isinstance(batch.sample, (tuple, list)):`

**`t3toolbox/backend/uniform_fitting.py`** (0)

- none

**`t3toolbox/backend/ut3_sampling.py`** (3)

- `t3toolbox/backend/ut3_sampling.py:107` | `ut3_probe` | is_packed( | `ragged_in = not ut3_operations.is_packed(ww)`
- `t3toolbox/backend/ut3_sampling.py:229` | `ut3_probe_corewise_tangent` | is_packed( | `ragged_in = not ut3_operations.is_packed(ww)`
- `t3toolbox/backend/ut3_sampling.py:274` | `ut3_probe_derivatives` | is_packed( | `ragged_in = not ut3_operations.is_packed(ww)`

**`t3toolbox/backend/utv_sampling.py`** (2)

- `t3toolbox/backend/utv_sampling.py:99` | `utv_probe` | is_packed( | `ragged_in = not ut3_operations.is_packed(ww)`
- `t3toolbox/backend/utv_sampling.py:407` | `utv_probe_derivatives` | is_packed( | `ragged_in = not ut3_operations.is_packed(ww)`

**`t3toolbox/backend/ut3_operations.py`** (4)

- `t3toolbox/backend/ut3_operations.py:72` | `_first_data_leaf` | is_ndarray( | `while not is_ndarray(xx[0]):`
- `t3toolbox/backend/ut3_operations.py:197` | `is_packed` | isinstance(tuple,list) | `return not isinstance(vectors, (list, tuple))`
- `t3toolbox/backend/ut3_operations.py:208` | `pack_if_ragged` | is_packed( | `return vectors if is_packed(vectors) else pack_vectors(vectors, N, len_P=len_P)`
- `t3toolbox/backend/ut3_operations.py:221` | `_pack_data_space` | is_packed( | `if is_packed(value):`

**`t3toolbox/backend/probing.py`** (10)

- `t3toolbox/backend/probing.py:139` | `compute_xi` | is_ndarray( | `is_uniform = is_ndarray(down_tucker_cores)`
- `t3toolbox/backend/probing.py:199` | `compute_nu` | is_ndarray( | `is_uniform = is_ndarray(right_tt_cores)`
- `t3toolbox/backend/probing.py:230` | `compute_eta` | is_ndarray( | `is_uniform = is_ndarray(up_tt_cores)`
- `t3toolbox/backend/probing.py:263` | `assemble_z` | is_ndarray( | `is_uniform = is_ndarray(tucker_cores)`
- `t3toolbox/backend/probing.py:394` | `compute_tau` | is_ndarray( | `is_uniform = is_ndarray(var_tt_cores)`
- `t3toolbox/backend/probing.py:656` | `compute_sigma_hat` | is_ndarray( | `is_uniform = is_ndarray(right_tt_cores)`
- `t3toolbox/backend/probing.py:701` | `compute_deta_tilde` | is_ndarray( | `is_uniform = is_ndarray(down_tucker_cores)`
- `t3toolbox/backend/probing.py:773` | `compute_sigma_tilde` | is_ndarray( | `is_uniform = is_ndarray(deta_tildes)`
- `t3toolbox/backend/probing.py:1286` | `compute_deta_tilde_tangent` | is_ndarray( | `is_uniform = is_ndarray(down_tucker_cores)`
- `t3toolbox/backend/probing.py:1382` | `compute_dxi_tilde_tangent` | is_ndarray( | `if is_ndarray(from_tau):`

**`t3toolbox/backend/apply.py`** (2)

- `t3toolbox/backend/apply.py:632` | `_apply_transpose_adjoint` | is_ndarray( | `is_uniform = is_ndarray(mus)`
- `t3toolbox/backend/apply.py:678` | `_apply_corewise_hessian_adjoint` | is_ndarray( | `is_uniform = is_ndarray(mus)`

**`t3toolbox/backend/entries.py`** (2)

- `t3toolbox/backend/entries.py:536` | `_entry_xis` | is_ndarray( | `is_uniform = is_ndarray(tucker_cores)`
- `t3toolbox/backend/entries.py:564` | `_onehot_vectors` | is_ndarray( | `is_uniform = is_ndarray(down_tucker_cores)`

**`t3toolbox/backend/optimizers.py`** (5)

- `t3toolbox/backend/optimizers.py:705` | `_array_leaves` | isinstance(tuple,list) | `if isinstance(node, (tuple, list)):`
- `t3toolbox/backend/optimizers.py:709` | `_array_leaves` | is_ndarray( | `return [leaf for leaf in leaves if is_ndarray(leaf)]`
- `t3toolbox/backend/optimizers.py:732` | `_prepare_jit_inputs.converted` | is_ndarray( | `if all(is_jax_ndarray(a) or not is_ndarray(a) for a in leaves):`
- `t3toolbox/backend/optimizers.py:734` | `_prepare_jit_inputs.converted` | is_ndarray( | `return jax.tree_util.tree_map(lambda a: jnp.asarray(a) if is_ndarray(a) else a, tree)`
- `t3toolbox/backend/optimizers.py:1133` | `_movement_record` | isinstance(tuple) | `if not isinstance(movements, tuple):`

**`t3toolbox/backend/optimizer_display.py`** (0)

- none

**`t3toolbox/fitting.py`** (4)

- `t3toolbox/fitting.py:213` | `_require_at_frame` | isinstance(tuple,list) | `sa = tuple(tuple(tuple(c.shape) for c in fam) if isinstance(fam, (list, tuple)) else tuple(fam.shape) for fam in a)`
- `t3toolbox/fitting.py:214` | `_require_at_frame` | isinstance(tuple,list) | `sb = tuple(tuple(tuple(c.shape) for c in fam) if isinstance(fam, (list, tuple)) else tuple(fam.shape) for fam in b)`
- `t3toolbox/fitting.py:507` | `_frontend_frame` | isinstance(tuple,list) | `uniform = not isinstance(frame_data[0], (tuple, list))`
- `t3toolbox/fitting.py:586` | `_is_packed` | isinstance(tuple,list) | `and not isinstance(batch.sample, (tuple, list)))`

**`t3toolbox/optimizers.py`** (0)

- none

**`t3toolbox/tucker_tensor_train.py`** (9)

- `t3toolbox/tucker_tensor_train.py:68` | `_as_index` | is_ndarray( | `return index if common.is_ndarray(index) else np.asarray(index)`
- `t3toolbox/tucker_tensor_train.py:75` | `_sample_shape` | isinstance(tuple,list) | `return tuple(np.shape(w)[-1] for w in ww) if isinstance(ww, (tuple, list)) else ()`
- `t3toolbox/tucker_tensor_train.py:85` | `_check_core_directions` | isinstance(tuple,list) | `is_pair = isinstance(core_directions, (tuple, list)) and len(core_directions) == 2`
- `t3toolbox/tucker_tensor_train.py:87` | `_check_core_directions` | is_ndarray( | `good = is_pair and all(common.is_ndarray(part) for part in core_directions)`
- `t3toolbox/tucker_tensor_train.py:91` | `_check_core_directions` | isinstance(tuple,list) | `good = is_pair and all(isinstance(part, (tuple, list)) for part in core_directions)`
- `t3toolbox/tucker_tensor_train.py:1408` | `TuckerTensorTrain.unstack._dfs` | is_ndarray( | `if common.is_ndarray(xx[0][0]):`
- `t3toolbox/tucker_tensor_train.py:2051` | `TuckerTensorTrain.__add__` | is_ndarray( | `elif common.is_ndarray(other):`
- `t3toolbox/tucker_tensor_train.py:2150` | `TuckerTensorTrain.__mul__` | is_ndarray( | `if common.is_ndarray(other):`
- `t3toolbox/tucker_tensor_train.py:2530` | `TuckerTensorTrain.inner` | is_ndarray( | `elif common.is_ndarray(other):`

**`t3toolbox/uniform_tucker_tensor_train.py`** (3)

- `t3toolbox/uniform_tucker_tensor_train.py:60` | `_check_sample` | is_packed( | `if kind == 'entries' or not ut3_operations.is_packed(sample):`
- `t3toolbox/uniform_tucker_tensor_train.py:250` | `UniformTuckerTensorTrain.to_t3._wrap` | is_ndarray( | `if common.is_ndarray(res[0][0]):   # res = (tucker_cores, tt_cores) leaf`
- `t3toolbox/uniform_tucker_tensor_train.py:1357` | `UT3Weights.to_t3weights._wrap` | is_ndarray( | `if common.is_ndarray(res[0][0]):   # res = (tucker_weights, tt_weights) leaf`

**`t3toolbox/manifold.py`** (0)

- none

**`t3toolbox/uniform_manifold.py`** (0)

- none


## M5. Python loops over modes in the data-space and helper code

Method: `ast.walk`; every `ast.For` statement and every comprehension generator (list/set/dict/generator expression) whose iterable, after unwrapping `enumerate(...)`, `reversed(...)`, `list(...)`, `tuple(...)`, is: `range(d)` (argument is the Name `d`), `range(len(...))`, `range(<anything else>)` (reported in its own group), `zip(...)` (every zip; whether it is over per-mode sequences is not decided here), or a call of one of `_modes`, `_arrays`, `_arrays_of`, `_mode_weights` (Name or attribute). Row: `file:line | enclosing function | tag | the line`. Loops over other iterables (e.g. `for k in order`, `for name in names`) are not listed.

| file | range(d) | range(len) | range(other) | zip | call _modes | call _arrays | call _arrays_of | call _mode_weights | total |
|---|---|---|---|---|---|---|---|---|---|
| `t3toolbox/backend/sampling_steps.py` | 0 | 2 | 0 | 7 | 3 | 0 | 0 | 0 | 12 |
| `t3toolbox/backend/jets.py` | 3 | 2 | 1 | 7 | 0 | 1 | 0 | 0 | 14 |
| `t3toolbox/backend/data_preparation.py` | 0 | 0 | 1 | 4 | 0 | 0 | 0 | 0 | 5 |
| `t3toolbox/backend/fitting.py` | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 | 3 |
| `t3toolbox/backend/uniform_fitting.py` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

### Main list (range(d), range(len), zip, helper calls)

**`t3toolbox/backend/sampling_steps.py`** (12)

- `t3toolbox/backend/sampling_steps.py:69` | `_leaf_order` | zip | `return tuple(min(a, n + 1) - 1 for a, n in zip(np.shape(w_PWo)[:len(order)], order))`
- `t3toolbox/backend/sampling_steps.py:238` | `eta_tilde_top_step` | zip | `corner = tuple(n - a_k + 1 + j_k for n, a_k, j_k in zip(order, a, j))`
- `t3toolbox/backend/sampling_steps.py:570` | `eta_tilde_dot_top_step` | zip | `corner = tuple(n - a_k + 1 + j_k for n, a_k, j_k in zip(order, a, j))`
- `t3toolbox/backend/sampling_steps.py:677` | `H_tilde_dot_step.eta_top_step` | zip | `corner = tuple(n - a_k + 1 + j_k for n, a_k, j_k in zip(order, a, j))`
- `t3toolbox/backend/sampling_steps.py:726` | `eta_top_step` | zip | `corner = tuple(n - a_k + 1 + j_k for n, a_k, j_k in zip(order, a, j))`
- `t3toolbox/backend/sampling_steps.py:754` | `eta_dot_top_step` | zip | `corner = tuple(n - a_k + 1 + j_k for n, a_k, j_k in zip(order, a, j))`
- `t3toolbox/backend/sampling_steps.py:922` | `_modes` | range(len) | `return [(xnp.asarray(value[i]), None if weight is None else weight[i], 1) for i in range(len(value))]`
- `t3toolbox/backend/sampling_steps.py:958` | `_weighted_sum` | call _modes | `for array, omega, edge in _modes(kind, value, weight, xnp):`
- `t3toolbox/backend/sampling_steps.py:1024` | `w_tilde_step` | call _modes | `for array, omega, _ in _modes(kind, r, _with_K(weight, kind, lens), xnp)]`
- `t3toolbox/backend/sampling_steps.py:1063` | `r_step` | range(len) | `return tuple(z_FWCo - xnp.asarray(y_FWCo) for z_FWCo, y_FWCo in zip(z, (y[i] for i in range(len(z)))))`
- `t3toolbox/backend/sampling_steps.py:1063` | `r_step` | zip | `return tuple(z_FWCo - xnp.asarray(y_FWCo) for z_FWCo, y_FWCo in zip(z, (y[i] for i in range(len(z)))))`
- `t3toolbox/backend/sampling_steps.py:1098` | `r_block_sumsq_step` | call _modes | `for r_FWCo, omega, _ in _modes(kind, r, weight, xnp)]`

**`t3toolbox/backend/jets.py`** (13)

- `t3toolbox/backend/jets.py:229` | `_arrays` | range(len) | `return [value[i] for i in range(len(value))] if kind == 'probe' else [value]`
- `t3toolbox/backend/jets.py:258` | `_scale_slots` | call _arrays | `for jet_FX in _arrays(kind, jets):`
- `t3toolbox/backend/jets.py:317` | `_mode_weights` | range(d) | `return [weight[i] for i in range(d)]`
- `t3toolbox/backend/jets.py:448` | `jet_to_dual_jet` | zip | `for i, (jet_FX, omega_FX) in enumerate(zip(arrays, _mode_weights(kind, weight, len(arrays), 'jet_to_dual_jet'))):`
- `t3toolbox/backend/jets.py:529` | `dual_jet_to_jet` | zip | `for i, (dual_FX, omega_FX) in enumerate(zip(arrays, _mode_weights(kind, weight, len(arrays), 'dual_jet_to_jet'))):`
- `t3toolbox/backend/jets.py:572` | `_check_pair` | zip | `for axis, (a, b) in enumerate(zip(omega_shape, r_shape)):`
- `t3toolbox/backend/jets.py:745` | `data_metric` | range(len) | `weights = [weight[i] for i in range(len(arrays))]`
- `t3toolbox/backend/jets.py:753` | `data_metric` | zip | `for i, (r_X, omega_X) in enumerate(zip(arrays, weights)):`
- `t3toolbox/backend/jets.py:1180` | `prepare_derivative_data` | zip | `for i, (y_X, omega_X) in enumerate(zip(arrays, _mode_weights(kind, weight, len(arrays),`
- `t3toolbox/backend/jets.py:1296` | `affine_jet` | zip | `for i, (w_i, p_i) in enumerate(zip(ww, p)):`
- `t3toolbox/backend/jets.py:1300` | `affine_jet` | range(d) | `return tuple(_affine_jet_of_one_array(ww[i], [p[i] for p in directions], 0, xnp) for i in range(d))`
- `t3toolbox/backend/jets.py:1554` | `_check_of_type_of_z` | range(d) | `for i in range(d)]`
- `t3toolbox/backend/jets.py:1562` | `_check_of_type_of_z` | zip | `if len(got) != len(expected) or any(a not in (1, b) for a, b in zip(got, expected)):`

**`t3toolbox/backend/data_preparation.py`** (4)

- `t3toolbox/backend/data_preparation.py:490` | `rescale_coordinates` | zip | `scaled_sample = tuple(_scaled(w_PWo, s_o, xnp) for w_PWo, s_o in zip(sample, scales))`
- `t3toolbox/backend/data_preparation.py:492` | `rescale_coordinates` | zip | `return scaled_sample, tuple(_scaled(y_FWo, s_o, xnp, divide=True) for y_FWo, s_o in zip(y, scales))`
- `t3toolbox/backend/data_preparation.py:563` | `t3_unscale_coordinates` | zip | `return tuple(_scaled(B_Cio, s_o, xnp) for B_Cio, s_o in zip(tucker_cores, scales)), tuple(tt_cores)`
- `t3toolbox/backend/data_preparation.py:718` | `check_coordinate_scales` | zip | `for i, (scale, N) in enumerate(zip(scales, shape)):`

**`t3toolbox/backend/fitting.py`** (3)

- `t3toolbox/backend/fitting.py:827` | `_same_leaves` | zip | `return len(a) == len(b) and all(_same_leaves(p, q) for p, q in zip(a, b))`
- `t3toolbox/backend/fitting.py:890` | `_y_times_metric_y` | zip | `for y_i, omega_i, z_shape in zip(_arrays_of(kind, y, shape), _arrays_of(kind, weight, shape), targets):`
- `t3toolbox/backend/fitting.py:950` | `_take_W` | zip | `if len(shape) < w_start + n_w or any(a not in (1, b) for a, b in zip(shape[w_start:w_start + n_w], W)):`

**`t3toolbox/backend/uniform_fitting.py`** (0)

- none

### range(other) (not in the main list; listed for completeness)

**`t3toolbox/backend/sampling_steps.py`** (0)

- none

**`t3toolbox/backend/jets.py`** (1)

- `t3toolbox/backend/jets.py:244` | `_factorials` | range(other) | `along_j = np.array([float(math.factorial(t)) for t in range(n + 1)])`

**`t3toolbox/backend/data_preparation.py`** (1)

- `t3toolbox/backend/data_preparation.py:380` | `rescale_direction` | range(other) | `lengths = [length[j] for j in range(m)]`

**`t3toolbox/backend/fitting.py`** (0)

- none

**`t3toolbox/backend/uniform_fitting.py`** (0)

- none

### `stack(` calls

Method: `ast.Call` whose callee is an Attribute named `stack` (receiver recorded) or a bare Name `stack`. Row: `file:line | enclosing function | receiver | the line`.

**`t3toolbox/backend/sampling_steps.py`** (8; receivers {'xnp': 8})

- `t3toolbox/backend/sampling_steps.py:243` | `eta_tilde_top_step` | `xnp` | `return xnp.stack(slots).reshape(a + slots[0].shape)`
- `t3toolbox/backend/sampling_steps.py:583` | `eta_tilde_dot_top_step` | `xnp` | `return xnp.stack(slots).reshape(a + slots[0].shape)`
- `t3toolbox/backend/sampling_steps.py:680` | `H_tilde_dot_step.eta_top_step` | `xnp` | `return xnp.stack(slots).reshape(a + slots[0].shape)`
- `t3toolbox/backend/sampling_steps.py:729` | `eta_top_step` | `xnp` | `return xnp.stack(slots).reshape(a + slots[0].shape)`
- `t3toolbox/backend/sampling_steps.py:761` | `eta_dot_top_step` | `xnp` | `return xnp.stack(slots).reshape(a + slots[0].shape)`
- `t3toolbox/backend/sampling_steps.py:863` | `w_one_hot_step` | `xnp` | `return xnp.stack([xnp.asarray(e, dtype=D.dtype) for e in vectors])`
- `t3toolbox/backend/sampling_steps.py:1028` | `w_tilde_step` | `xnp` | `return xnp.stack(seed)`
- `t3toolbox/backend/sampling_steps.py:1099` | `r_block_sumsq_step` | `xnp` | `return xnp.stack(sums) if kind == 'probe' else sums[0]`

**`t3toolbox/backend/jets.py`** (1; receivers {'xnp': 1})

- `t3toolbox/backend/jets.py:1204` | `_affine_jet_of_one_array` | `xnp` | `stacked = xnp.stack([xnp.asarray(slot, dtype=dtype) for slot in slots], axis=axis)`

**`t3toolbox/backend/data_preparation.py`** (1; receivers {'xnp': 1})

- `t3toolbox/backend/data_preparation.py:195` | `data_block_norms` | `xnp` | `return xnp.stack([_block_norms_F(y_FWo, len(order), xnp) for y_FWo in y])`

**`t3toolbox/backend/fitting.py`** (2; receivers {'xnp': 2})

- `t3toolbox/backend/fitting.py:151` | `block_sumsq_over_probes` | `xnp` | `return xnp.stack(rows, axis=0)`
- `t3toolbox/backend/fitting.py:936` | `_y_block_metric_y` | `xnp` | `return xnp.stack(sums) if kind == 'probe' else sums[0]`

**`t3toolbox/backend/uniform_fitting.py`** (0; receivers {})

- none

### All calls of `_modes(`, `_arrays(`, `_arrays_of(`, `_mode_weights(` in the five files (loop-header or not)

- `t3toolbox/backend/sampling_steps.py:958` | `_weighted_sum` | `_modes` | `for array, omega, edge in _modes(kind, value, weight, xnp):`
- `t3toolbox/backend/sampling_steps.py:1024` | `w_tilde_step` | `_modes` | `for array, omega, _ in _modes(kind, r, _with_K(weight, kind, lens), xnp)]`
- `t3toolbox/backend/sampling_steps.py:1098` | `r_block_sumsq_step` | `_modes` | `for r_FWCo, omega, _ in _modes(kind, r, weight, xnp)]`
- `t3toolbox/backend/jets.py:258` | `_scale_slots` | `_arrays` | `for jet_FX in _arrays(kind, jets):`
- `t3toolbox/backend/jets.py:446` | `jet_to_dual_jet` | `_arrays` | `arrays = _arrays(kind, jet)`
- `t3toolbox/backend/jets.py:448` | `jet_to_dual_jet` | `_mode_weights` | `for i, (jet_FX, omega_FX) in enumerate(zip(arrays, _mode_weights(kind, weight, len(arrays), 'jet_to_dual_jet'))):`
- `t3toolbox/backend/jets.py:527` | `dual_jet_to_jet` | `_arrays` | `arrays = _arrays(kind, dual_jet)`
- `t3toolbox/backend/jets.py:529` | `dual_jet_to_jet` | `_mode_weights` | `for i, (dual_FX, omega_FX) in enumerate(zip(arrays, _mode_weights(kind, weight, len(arrays), 'dual_jet_to_jet'))):`
- `t3toolbox/backend/jets.py:1173` | `prepare_derivative_data` | `_arrays` | `arrays = _arrays(kind, y_taylor)`
- `t3toolbox/backend/jets.py:1180` | `prepare_derivative_data` | `_mode_weights` | `for i, (y_X, omega_X) in enumerate(zip(arrays, _mode_weights(kind, weight, len(arrays),`
- `t3toolbox/backend/fitting.py:890` | `_y_times_metric_y` | `_arrays_of` | `for y_i, omega_i, z_shape in zip(_arrays_of(kind, y, shape), _arrays_of(kind, weight, shape), targets):`
- `t3toolbox/backend/fitting.py:890` | `_y_times_metric_y` | `_arrays_of` | `for y_i, omega_i, z_shape in zip(_arrays_of(kind, y, shape), _arrays_of(kind, weight, shape), targets):`


## M6. The uniform frontend's per-mode surface

Method: `ast` over `t3toolbox/uniform_tucker_tensor_train.py` and `t3toolbox/uniform_manifold.py`: every def (module function or method of any class; nested defs excluded) whose name does not start with `_`. Selected if (P) a parameter is named `ww`, `index`, `c`, `scales` or `sample`, or (R) the return annotation or the header's text from the `->` onward mentions `Vectors`, `packed` or `MIRRORS` (case-insensitive) or `len=d`, or (C) any trailing comment anywhere in the header does. Row: `file:line-line | class.method | selected by | ` then the verbatim header. `_`-prefixed (private) defs are not listed.

**`t3toolbox/uniform_tucker_tensor_train.py`**: 84 public defs

- `t3toolbox/uniform_tucker_tensor_train.py:264-267` | `UniformTuckerTensorTrain.unscale_coordinates` | P:scales C
```python
    def unscale_coordinates(
            self:    'UniformTuckerTensorTrain',
            scales:  Sequence[NDArray],     # len=d, elm_shape=(Ni,): as given to rescale_coordinates
    ) -> 'UniformTuckerTensorTrain':        # same structure and masks: the Tucker supercore's last axis scaled
```
- `t3toolbox/uniform_tucker_tensor_train.py:400-403` | `UniformTuckerTensorTrain.entries` | P:index
```python
    def entries(
            self,
            index:             NDArray,                    # index^d, shape=(d,)+W, int (a nested list is converted)
    ) -> NDArray:                                          # z: ℝ, shape=W+C
```
- `t3toolbox/uniform_tucker_tensor_train.py:424-429` | `UniformTuckerTensorTrain.apply` | P:ww C
```python
    def apply(
            self,
            ww:                Sequence[NDArray],          # input jets: P_in_i⊗V_i, len=d, F_in_i+W+(Ni,); or packed
                                                           #   (d,)+F_in+W+(N,)
            order:             Tuple[int, ...] = (),       # the jet shape; F=tuple(n+1 for n in order)
    ) -> NDArray:                                          # z: P, shape=F+W+C
```
- `t3toolbox/uniform_tucker_tensor_train.py:450-455` | `UniformTuckerTensorTrain.probe` | P:ww R
```python
    def probe(
            self,
            ww:                Sequence[NDArray],          # input jets: P_in_i⊗V_i, len=d, F_in_i+W+(Ni,); or packed
                                                           #   (d,)+F_in+W+(N,)
            order:             Tuple[int, ...] = (),       # the jet shape; F=tuple(n+1 for n in order)
    ) -> Sequence[NDArray]:                                # z: P⊗V_i*, MIRRORS ww: len=d, F+W+C+(Ni,); or packed
```
- `t3toolbox/uniform_tucker_tensor_train.py:488-495` | `UniformTuckerTensorTrain.apply_corewise_transpose` | P:c,ww C
```python
    def apply_corewise_transpose(
            self,
            ww:                Sequence[NDArray],          # input jets: P_in_i⊗V_i, len=d, F_in_i+W+(Ni,); or packed
                                                           #   (d,)+F_in+W+(N,)
            c:                 NDArray,                    # P*, shape=F+W+[K]+C
            order:             Tuple[int, ...] = (),       # the jet shape; F=tuple(n+1 for n in order)
            sum_over_samples:  bool = True,                # False: W joins the stack
    ) -> Tuple[NDArray, NDArray]:                          # (tucker, tt) supercores: X*, (d,)+[W]+[K]+C+(...)
```
- `t3toolbox/uniform_tucker_tensor_train.py:532-537` | `UniformTuckerTensorTrain.entries_corewise_transpose` | P:c,index
```python
    def entries_corewise_transpose(
            self,
            index:             NDArray,                    # index^d, shape=(d,)+W, int (a nested list is converted)
            c:                 NDArray,                    # ℝ, shape=W+[K]+C
            sum_over_samples:  bool = True,                # False: W joins the stack
    ) -> Tuple[NDArray, NDArray]:                          # (tucker, tt) supercores: X*, (d,)+[W]+[K]+C+(...)
```
- `t3toolbox/uniform_tucker_tensor_train.py:546-553` | `UniformTuckerTensorTrain.probe_corewise_transpose` | P:c,ww C
```python
    def probe_corewise_transpose(
            self,
            ww:                Sequence[NDArray],          # input jets: P_in_i⊗V_i, len=d, F_in_i+W+(Ni,); or packed
                                                           #   (d,)+F_in+W+(N,)
            c:                 Sequence[NDArray],          # P*⊗V_i, len=d, F+W+[K]+C+(Ni,); or packed
            order:             Tuple[int, ...] = (),       # the jet shape; F=tuple(n+1 for n in order)
            sum_over_samples:  bool = True,                # False: W joins the stack
    ) -> Tuple[NDArray, NDArray]:                          # (tucker, tt) supercores: X*, (d,)+[W]+[K]+C+(...)
```
- `t3toolbox/uniform_tucker_tensor_train.py:562-569` | `UniformTuckerTensorTrain.probe_corewise_tangent` | P:ww R
```python
    def probe_corewise_tangent(
            self,
            ww:                Sequence[NDArray],          # input jets: P_in_i⊗V_i, len=d, F_in_i+W+(Ni,); or packed
                                                           #   (d,)+F_in+W+(N,)
            core_directions:   Tuple[NDArray, NDArray],    # X, (tucker, tt) supercores (d,)+K+C+(...): the point's
                                                           #   sizes
            order:             Tuple[int, ...] = (),       # the jet shape; F=tuple(n+1 for n in order)
    ) -> Sequence[NDArray]:                                # J x_dot: P⊗V_i*, MIRRORS ww: len=d, F+W+K+C+(Ni,); or
```
- `t3toolbox/uniform_tucker_tensor_train.py:579-588` | `UniformTuckerTensorTrain.probe_corewise_hessian` | P:c,ww C
```python
    def probe_corewise_hessian(
            self,
            ww:                Sequence[NDArray],          # input jets: P_in_i⊗V_i, len=d, F_in_i+W+(Ni,); or packed
                                                           #   (d,)+F_in+W+(N,)
            c:                 Sequence[NDArray],          # P*⊗V_i, len=d, F+W+C+(Ni,); or packed
            core_directions:   Tuple[NDArray, NDArray],    # X, (tucker, tt) supercores (d,)+K+C+(...): the point's
                                                           #   sizes
            order:             Tuple[int, ...] = (),       # the jet shape; F=tuple(n+1 for n in order)
            sum_over_samples:  bool = True,                # False: W joins the stack
    ) -> Tuple[NDArray, NDArray]:                          # (tucker, tt) blocks: X*, (d,)+[W]+K+C+(...)
```
- `t3toolbox/uniform_tucker_tensor_train.py:598-605` | `UniformTuckerTensorTrain.apply_corewise_tangent` | P:ww C
```python
    def apply_corewise_tangent(
            self,
            ww:                Sequence[NDArray],          # input jets: P_in_i⊗V_i, len=d, F_in_i+W+(Ni,); or packed
                                                           #   (d,)+F_in+W+(N,)
            core_directions:   Tuple[NDArray, NDArray],    # X, (tucker, tt) supercores (d,)+K+C+(...): the point's
                                                           #   sizes
            order:             Tuple[int, ...] = (),       # the jet shape; F=tuple(n+1 for n in order)
    ) -> NDArray:                                          # J x_dot: P, shape=F+W+K+C
```
- `t3toolbox/uniform_tucker_tensor_train.py:614-623` | `UniformTuckerTensorTrain.apply_corewise_hessian` | P:c,ww C
```python
    def apply_corewise_hessian(
            self,
            ww:                Sequence[NDArray],          # input jets: P_in_i⊗V_i, len=d, F_in_i+W+(Ni,); or packed
                                                           #   (d,)+F_in+W+(N,)
            c:                 NDArray,                    # P*, shape=F+W+C
            core_directions:   Tuple[NDArray, NDArray],    # X, (tucker, tt) supercores (d,)+K+C+(...): the point's
                                                           #   sizes
            order:             Tuple[int, ...] = (),       # the jet shape; F=tuple(n+1 for n in order)
            sum_over_samples:  bool = True,                # False: W joins the stack
    ) -> Tuple[NDArray, NDArray]:                          # (tucker, tt) blocks: X*, (d,)+[W]+K+C+(...)
```
- `t3toolbox/uniform_tucker_tensor_train.py:633-638` | `UniformTuckerTensorTrain.entries_corewise_tangent` | P:index
```python
    def entries_corewise_tangent(
            self,
            index:             NDArray,                    # index^d, shape=(d,)+W, int (a nested list is converted)
            core_directions:   Tuple[NDArray, NDArray],    # X, (tucker, tt) supercores (d,)+K+C+(...): the point's
                                                           #   sizes
    ) -> NDArray:                                          # J x_dot: ℝ, shape=W+K+C
```
- `t3toolbox/uniform_tucker_tensor_train.py:648-655` | `UniformTuckerTensorTrain.entries_corewise_hessian` | P:c,index
```python
    def entries_corewise_hessian(
            self,
            index:             NDArray,                    # index^d, shape=(d,)+W, int (a nested list is converted)
            c:                 NDArray,                    # ℝ, shape=W+C
            core_directions:   Tuple[NDArray, NDArray],    # X, (tucker, tt) supercores (d,)+K+C+(...): the point's
                                                           #   sizes
            sum_over_samples:  bool = True,                # False: W joins the stack
    ) -> Tuple[NDArray, NDArray]:                          # (tucker, tt) blocks: X*, (d,)+[W]+K+C+(...)
```
- `t3toolbox/uniform_tucker_tensor_train.py:716-723` | `UniformTuckerTensorTrain.oversampled_ranks` | C
```python
    def oversampled_ranks(
            self,
            p:        int,                                 # the rank increment, p >= 0
            sharing:  typ.Optional[typ.Sequence] = None,   # len=d group labels; None = unshared
    ) -> typ.Tuple[
        NDArray,   # tucker ranks, HOST int (d,)+stack_shape
        NDArray,   # tt ranks, HOST int (d+1,)+stack_shape
    ]:
```
- `t3toolbox/uniform_tucker_tensor_train.py:773-777` | `UniformTuckerTensorTrain.has_shared_tucker_factors` | C
```python
    def has_shared_tucker_factors(
            self,
            sharing:    typ.Sequence,   # len=d; one hashable group label per mode
            rtol:       float = 1e-9,   # relative tolerance on the factor deviation
    ) -> NDArray:  # bool array, shape = stack_shape (scalar/0-d when unstacked)
```
- `t3toolbox/uniform_tucker_tensor_train.py:1199-1203` | `UT3Weights.has_shared_tucker_weights` | C
```python
    def has_shared_tucker_weights(
            self,
            sharing:    typ.Sequence,   # len=d; one hashable group label per mode
            rtol:       float = 1e-9,   # relative tolerance on the Tucker-weight deviation
    ) -> NDArray:  # bool array, shape = stack_shape (scalar/0-d when unstacked)
```
(selected 16 of 84 public defs)

**`t3toolbox/uniform_manifold.py`**: 65 public defs

- `t3toolbox/uniform_manifold.py:144-156` | `UT3Tangent.align_padded` | C
```python
    def align_padded(
            self,
            split:         typ.Tuple[int, int],                                       # (Na, Rb), from UT3Frame.padded

            tucker_ranks:  typ.Optional[typ.Union[typ.Sequence[int], NDArray]] = None,   # len=d or HOST int (d,)+C, output; None = the frame's own
            tt_ranks:      typ.Optional[typ.Union[typ.Sequence[int], NDArray]] = None,   # len=d+1 or HOST int (d+1,)+C; None = the frame's own

            sharing:       typ.Optional[typ.Sequence] = None,                         # len=d group labels; None = unshared
            update:        typ.Collection[str] = ('D', 'U', 'L', 'R'),                # as T3Tangent.align_padded
    ) -> typ.Tuple[
        'ubv.UT3Frame',   # the aligned frame, block layout, new blocks real-first
        NDArray,          # movement, shape = frame stack C
    ]:
```
- `t3toolbox/uniform_manifold.py:551-556` | `UT3Tangent.probe` | P:ww R
```python
    def probe(
            self,
            ww:                typ.Sequence[NDArray],      # input jets: P_in_i⊗V_i, len=d, F_in_i+W+(Ni,); or packed
                                                           #   (d,)+F_in+W+(N,)
            order:             typ.Tuple[int, ...] = (),   # the jet shape; F=tuple(n+1 for n in order)
    ) -> typ.Sequence[NDArray]:                            # Jp: P⊗V_i*, MIRRORS ww: len=d, F+W+K+C+(Ni,); or packed
```
- `t3toolbox/uniform_manifold.py:584-589` | `UT3Tangent.apply` | P:ww C
```python
    def apply(
            self,
            ww:                typ.Sequence[NDArray],      # input jets: P_in_i⊗V_i, len=d, F_in_i+W+(Ni,); or packed
                                                           #   (d,)+F_in+W+(N,)
            order:             typ.Tuple[int, ...] = (),   # the jet shape; F=tuple(n+1 for n in order)
    ) -> NDArray:                                          # Jp: P, shape=F+W+K+C
```
- `t3toolbox/uniform_manifold.py:610-613` | `UT3Tangent.entries` | P:index
```python
    def entries(
            self,
            index:             NDArray,                    # index^d, shape=(d,)+W, int (a nested list is converted)
    ) -> NDArray:                                          # Jp: ℝ, shape=W+K+C
```
- `t3toolbox/uniform_manifold.py:634-641` | `UT3Tangent.probe_transpose` | P:c,ww C
```python
    def probe_transpose(
            frame:             ubv.UT3Frame,               # the orthogonal frame the tangent attaches at
            ww:                typ.Sequence[NDArray],      # input jets: P_in_i⊗V_i, len=d, F_in_i+W+(Ni,); or packed
                                                           #   (d,)+F_in+W+(N,)
            c:                 typ.Sequence[NDArray],      # P*⊗V_i, len=d, F+W+[K]+C+(Ni,); or packed
            order:             typ.Tuple[int, ...] = (),   # the jet shape; F=tuple(n+1 for n in order)
            sum_over_samples:  bool = True,                # False: keep the sample stack W (W+K tangents)
    ) -> 'UT3Tangent':                                     # X*, in a UT3Tangent at frame (before the projection);
```
- `t3toolbox/uniform_manifold.py:682-689` | `UT3Tangent.apply_transpose` | P:c,ww C
```python
    def apply_transpose(
            frame:             ubv.UT3Frame,               # the orthogonal frame the tangent attaches at
            ww:                typ.Sequence[NDArray],      # input jets: P_in_i⊗V_i, len=d, F_in_i+W+(Ni,); or packed
                                                           #   (d,)+F_in+W+(N,)
            c:                 NDArray,                    # P*, shape=F+W+[K]+C
            order:             typ.Tuple[int, ...] = (),   # the jet shape; F=tuple(n+1 for n in order)
            sum_over_samples:  bool = True,                # False: keep the sample stack W (W+K tangents)
    ) -> 'UT3Tangent':                                     # X*, in a UT3Tangent at frame (before the projection);
```
- `t3toolbox/uniform_manifold.py:700-705` | `UT3Tangent.entries_transpose` | P:c,index
```python
    def entries_transpose(
            frame:             ubv.UT3Frame,               # the orthogonal frame the tangent attaches at
            index:             NDArray,                    # index^d, shape=(d,)+W, int (a nested list is converted)
            c:                 NDArray,                    # ℝ, shape=W+[K]+C
            sum_over_samples:  bool = True,                # False: keep the sample stack W (W+K tangents)
    ) -> 'UT3Tangent':                                     # X*, in a UT3Tangent at frame (before the projection);
```
- `t3toolbox/uniform_manifold.py:746-749` | `UT3Tangent.unit` | P:index
```python
    def unit(
            frame:  ubv.UT3Frame,
            index:  typ.Tuple[bool, int, typ.Sequence[int]],  # (use_tt_coordinate, i, within_index)
    ) -> 'UT3Tangent':
```
(selected 8 of 65 public defs)

### `LocalModel.residual`, `.jacobian`, `.gradient` (`t3toolbox/fitting.py`)

- `t3toolbox/fitting.py:451-451` `LocalModel.residual`
```python
    def residual(self):                          # r = z(X) - y: Y*, of the type of z (the oracle's optional output)
```
- `t3toolbox/fitting.py:458-461` `LocalModel.jacobian`
```python
    def jacobian(
            self,
            p,      # a trial tangent at this model's frame
    ) -> typ.Any:   # J p = J(Π p): Y*, of the type of z with p's stack K
```
- `t3toolbox/fitting.py:455-455` `LocalModel.gradient`
```python
    def gradient(self):                          # g = Π Jᵀ(M r) (+ g_R), M = gamma * omega**2: a tangent at frame
```

### Public backend functions of `backend/ut3_sampling.py` and `backend/utv_sampling.py`

Method: names in `__all__` (ast); for each, the function body (ast, nested defs included) is searched for `ast.Call`s whose callee name is `is_packed`, `pack_if_ragged` or `unpack_vectors` (also `pack_vectors` shown, since the task is about packing). Lines are listed. A "no" means no such call in the body (a helper it calls may contain one).

**`t3toolbox/backend/ut3_sampling.py`** (`__all__` has 28 names)

| name | def line | is_packed | pack_if_ragged | unpack_vectors | pack_vectors |
|---|---|---|---|---|---|
| `ut3_entries` | 74 | no | no | no | no |
| `ut3_apply` | 84 | no | no | no | no |
| `ut3_probe` | 97 | yes: 107 | no | yes: 109 | no |
| `ut3_full_sum` | 112 | no | no | no | no |
| `ut3_apply_corewise_tangent` | 145 | no | no | no | no |
| `ut3_apply_corewise_transpose` | 162 | no | no | no | no |
| `ut3_entries_corewise_tangent` | 183 | no | no | no | no |
| `ut3_entries_corewise_transpose` | 198 | no | no | no | no |
| `ut3_probe_corewise_tangent` | 217 | yes: 229 | no | yes: 232 | no |
| `ut3_probe_corewise_transpose` | 235 | no | yes: 250 | no | no |
| `ut3_probe_derivatives` | 263 | yes: 274 | yes: 275,276 | yes: 278 | no |
| `ut3_apply_derivatives` | 281 | no | yes: 291,292 | no | no |
| `ut3_entries_derivatives` | 296 | no | yes: 306 | no | no |
| `ut3_apply_corewise_derivatives_transpose` | 317 | no | yes: 328,329 | no | no |
| `ut3_entries_corewise_derivatives_transpose` | 335 | no | yes: 346 | no | no |
| `ut3_probe_corewise_derivatives_transpose` | 352 | no | yes: 365,366,367 | no | no |
| `ut3_probe_corewise_derivatives_hessian` | 380 | no | yes: 434,435,436 | no | no |
| `ut3_probe_corewise_derivatives_hessian_from_sweep` | 442 | no | yes: 463 | no | no |
| `ut3_probe_corewise_hessian` | 490 | no | yes: 536 | no | no |
| `ut3_probe_corewise_hessian_from_sweep` | 543 | no | yes: 559 | no | no |
| `ut3_apply_corewise_hessian` | 564 | no | no | no | no |
| `ut3_apply_corewise_hessian_from_sweep` | 586 | no | no | no | no |
| `ut3_entries_corewise_hessian` | 600 | no | no | no | no |
| `ut3_entries_corewise_hessian_from_sweep` | 620 | no | no | no | no |
| `ut3_apply_corewise_derivatives_hessian` | 640 | no | yes: 688 | no | no |
| `ut3_apply_corewise_derivatives_hessian_from_sweep` | 692 | no | no | no | no |
| `ut3_entries_corewise_derivatives_hessian` | 711 | no | yes: 727 | no | no |
| `ut3_entries_corewise_derivatives_hessian_from_sweep` | 731 | no | no | no | no |

Functions in `__all__` with at least one: is_packed 3, pack_if_ragged 13, unpack_vectors 3 of 28

**`t3toolbox/backend/utv_sampling.py`** (`__all__` has 30 names)

| name | def line | is_packed | pack_if_ragged | unpack_vectors | pack_vectors |
|---|---|---|---|---|---|
| `utv_probe` | 87 | yes: 99 | no | yes: 102 | no |
| `utv_apply` | 105 | no | no | no | no |
| `utv_entries` | 121 | no | no | no | no |
| `utv_probe_transpose` | 472 | no | yes: 486 | no | no |
| `utv_apply_transpose` | 492 | no | no | no | no |
| `utv_entries_transpose` | 511 | no | no | no | no |
| `utv_precompute_apply_frame_sweep` | 150 | no | yes: 158 | no | no |
| `utv_precompute_probe_frame_sweep` | 162 | no | yes: 169 | no | no |
| `utv_precompute_entries_frame_sweep` | 173 | no | no | no | no |
| `utv_apply_jacobian_from_sweep` | 183 | no | no | no | no |
| `utv_probe_jacobian_from_sweep` | 194 | no | no | no | no |
| `utv_entries_jacobian_from_sweep` | 206 | no | no | no | no |
| `utv_apply_transpose_from_sweep` | 217 | no | no | no | no |
| `utv_probe_transpose_from_sweep` | 229 | no | yes: 237 | no | no |
| `utv_entries_transpose_from_sweep` | 241 | no | no | no | no |
| `utv_precompute_apply_frame_sweep_jets` | 264 | no | yes: 274,275 | no | no |
| `utv_precompute_probe_frame_sweep_jets` | 280 | no | yes: 290,291 | no | no |
| `utv_precompute_entries_frame_sweep_jets` | 296 | no | yes: 305 | no | no |
| `utv_apply_jacobian_derivatives_from_sweep` | 310 | no | no | no | no |
| `utv_probe_jacobian_derivatives_from_sweep` | 322 | no | no | no | no |
| `utv_entries_jacobian_derivatives_from_sweep` | 335 | no | no | no | no |
| `utv_apply_transpose_derivatives_from_sweep` | 347 | no | no | no | no |
| `utv_probe_transpose_derivatives_from_sweep` | 360 | no | yes: 370 | no | no |
| `utv_entries_transpose_derivatives_from_sweep` | 376 | no | no | no | no |
| `utv_probe_derivatives` | 396 | yes: 407 | yes: 408,409 | yes: 411 | no |
| `utv_apply_derivatives` | 414 | no | yes: 425,426 | no | no |
| `utv_entries_derivatives` | 430 | no | yes: 440 | no | no |
| `utv_probe_derivatives_transpose` | 536 | no | yes: 550,551,552 | no | no |
| `utv_apply_derivatives_transpose` | 558 | no | yes: 570,571 | no | no |
| `utv_entries_derivatives_transpose` | 577 | no | yes: 588 | no | no |

Functions in `__all__` with at least one: is_packed 2, pack_if_ragged 14, unpack_vectors 2 of 30


## M7. The entries index as one array

Method: for each named function (matched on the LAST component of its qualified name, so every method of that name in the file is included, tagged by its class), every `ast.Name` load/store of an index-like identifier (`index`, `ind`, `idx`, `indices`, `sample`) inside its body is collected, one row per source line. A row is tagged **ONE-ARRAY** when the identifier is (a) subscripted `x[...]`, (b) an attribute read `.shape`/`.ndim`/`.size`/`.dtype`/`.astype`, or (c) an argument of `np.shape/ndim/size/asarray/array/ravel/broadcast_to`, `xnp.<same>`, `len`, `tuple`, `int` -- the forms that assume the index is one array; otherwise **mention** (passed through, assigned, compared). `sample` is included only because for the probe/apply kinds it is the jets, so a `sample` row is relevant only where the code is entries-specific (the line shows). Text of the line is quoted. Functions named in the task but not found are listed as NOT FOUND.

**`t3toolbox/backend/jets.py`**

- `t3toolbox/backend/jets.py:1363` | `index_to_one_hot` | index: mention | `use_jax = tree_contains_jax(index)`
- `t3toolbox/backend/jets.py:1365` | `index_to_one_hot` | index: mention | `if not is_ndarray(index):`
- `t3toolbox/backend/jets.py:1366` | `index_to_one_hot` | index: ONE-ARRAY asarray(; index: mention | `index = np.asarray(index)`
- `t3toolbox/backend/jets.py:1367` | `index_to_one_hot` | index: ONE-ARRAY .dtype | `if not np.issubdtype(index.dtype, np.integer):`
- `t3toolbox/backend/jets.py:1368` | `index_to_one_hot` | index: ONE-ARRAY .dtype | `raise ValueError('index_to_one_hot: the index is of an integer dtype; got %s' % (index.dtype,))`
- `t3toolbox/backend/jets.py:1369` | `index_to_one_hot` | index: ONE-ARRAY .ndim; index: ONE-ARRAY .shape | `if index.ndim < 1 or index.shape[0] != len(shape):`
- `t3toolbox/backend/jets.py:1371` | `index_to_one_hot` | index: ONE-ARRAY .shape | `% (len(shape), tuple(index.shape)))`
- `t3toolbox/backend/jets.py:1374` | `index_to_one_hot` | index: ONE-ARRAY subscript | `outside = (index[i] < -N) | (index[i] >= N)`
- `t3toolbox/backend/jets.py:1377` | `index_to_one_hot` | index: ONE-ARRAY subscript | `% (i, sorted(set(index[i][outside].tolist()))[:5], -N, N))`
- `t3toolbox/backend/jets.py:1381` | `index_to_one_hot` | index: ONE-ARRAY subscript | `index_i = xnp.where(index[i] < 0, index[i] + N, index[i])`
- `t3toolbox/backend/jets.py:1514` | `check_sample` | sample: mention | `if isinstance(sample, (tuple, list)):`
- `t3toolbox/backend/jets.py:1516` | `check_sample` | sample: mention | `% (d, type(sample).__name__))`
- `t3toolbox/backend/jets.py:1517` | `check_sample` | sample: ONE-ARRAY shape( | `index_shape = tuple(np.shape(sample))`
- `t3toolbox/backend/jets.py:1520` | `check_sample` | sample: ONE-ARRAY .dtype | `if not np.issubdtype(np.dtype(sample.dtype), np.integer):`
- `t3toolbox/backend/jets.py:1521` | `check_sample` | sample: ONE-ARRAY .dtype | `raise ValueError('index must be an integer array; got dtype %s' % (np.dtype(sample.dtype),))`
- `t3toolbox/backend/jets.py:1523` | `check_sample` | sample: mention | `_require_input_jets(sample, shape, len(order))`

(rows: ONE-ARRAY 12, mention 6)

**`t3toolbox/backend/sampling_variables.py`**

- `t3toolbox/backend/sampling_variables.py:1072` | `_read_setting` | sample: ONE-ARRAY shape( | `F_in, W = ((),) * len(D), tuple(np.shape(sample)[1:])`
- `t3toolbox/backend/sampling_variables.py:1074` | `_read_setting` | sample: ONE-ARRAY subscript; sample: mention | `F_in, W = tuple(tuple(np.shape(w)[:m]) for w in sample), tuple(np.shape(sample[0])[m:-1])`
- `t3toolbox/backend/sampling_variables.py:1151` | `_check_stacks` | sample: mention | `for i, w in enumerate(sample):`
- `t3toolbox/backend/sampling_variables.py:1167` | `_check_stacks` | sample: ONE-ARRAY shape( | `if np.shape(sample)[0] != d or np.shape(sample)[-1] != N:`
- `t3toolbox/backend/sampling_variables.py:1169` | `_check_stacks` | sample: ONE-ARRAY shape( | `'%s, where d = %d and N = %d' % (np.shape(sample), d, N))`
- `t3toolbox/backend/sampling_variables.py:1491` | `_uniform_index` | index: ONE-ARRAY ndim( | `sizes = np.asarray(shape).reshape((len(shape),) + (1,) * (np.ndim(index) - 1))`
- `t3toolbox/backend/sampling_variables.py:1492` | `_uniform_index` | index: mention | `if not is_jax_ndarray(index):`
- `t3toolbox/backend/sampling_variables.py:1493` | `_uniform_index` | index: ONE-ARRAY asarray( | `array = np.asarray(index)`
- `t3toolbox/backend/sampling_variables.py:1501` | `_uniform_index` | index: mention | `return jnp.where(index < 0, index + sizes, index)`
- `t3toolbox/backend/sampling_variables.py:1858` | `sampling_variables` | sample: ONE-ARRAY ndim(; sample: ONE-ARRAY shape( | `if np.ndim(sample) < 1 or np.shape(sample)[0] != d:`
- `t3toolbox/backend/sampling_variables.py:1860` | `sampling_variables` | sample: ONE-ARRAY shape( | `'d = %d; got shape %s' % (d, np.shape(sample)))`
- `t3toolbox/backend/sampling_variables.py:1868` | `sampling_variables` | sample: mention | `sample = _uniform_index(sample, shape)`
- `t3toolbox/backend/sampling_variables.py:1870` | `sampling_variables` | sample: mention | `sample = ut3_operations.pack_if_ragged(sample, N, len_P=len(order))`
- `t3toolbox/backend/sampling_variables.py:1876` | `sampling_variables` | sample: mention | `setting = _read_setting(kind, stores, sample, order, variations, graph, core_directions,`
- `t3toolbox/backend/sampling_variables.py:1878` | `sampling_variables` | sample: mention | `_check_stacks(setting, stores, sample, Vc, variations, core_directions)`
- `t3toolbox/backend/sampling_variables.py:1893` | `sampling_variables` | sample: mention | `use_jax = tree_contains_jax((frame, sample, y, weight, variations, core_directions, Vc,`
- `t3toolbox/backend/sampling_variables.py:1899` | `sampling_variables` | sample: mention | `**{('index' if kind == 'entries' else 'w'): sample})`
- `t3toolbox/backend/sampling_variables.py:1932` | `sampling_variables` | sample: mention | `out = tuple(sample if (name == 'w' and w_given) else values[name] for name in want)`

(rows: ONE-ARRAY 9, mention 11)

**`t3toolbox/backend/sampling_steps.py`**

- `t3toolbox/backend/sampling_steps.py:860` | `w_one_hot_step` | index: mention | `xnp, _, _ = get_backend(False, tree_contains_jax((index, D)))`
- `t3toolbox/backend/sampling_steps.py:861` | `w_one_hot_step` | index: mention | `vectors = jets.index_to_one_hot(index, [np.shape(B)[-1] for B in D])`

(rows: ONE-ARRAY 0, mention 2)

**`t3toolbox/backend/entries.py`**

- `t3toolbox/backend/entries.py:53` | `_t3_entries_step` | ind: mention | `ind, B_Cpo, G_Capb = ind_B_G`
- `t3toolbox/backend/entries.py:54` | `_t3_entries_step` | ind: ONE-ARRAY .ndim | `n_idx = ind.ndim                                           # this mode's index carries exactly the W axes`
- `t3toolbox/backend/entries.py:55` | `_t3_entries_step` | ind: mention | `xi_CpW = B_Cpo[..., ind]                                   # C + (p,) + W (index batch trails)`
- `t3toolbox/backend/entries.py:91` | `t3_entries` | index: mention | `(z,) = sampling_variables.sampling_variables('entries', x, index, ('z',))`
- `t3toolbox/backend/entries.py:110` | `t3_entries_ambient_transpose` | index: mention | `use_jax = tree_contains_jax((c, index))`
- `t3toolbox/backend/entries.py:112` | `t3_entries_ambient_transpose` | index: ONE-ARRAY array(; index: mention | `index = xnp.array(index)`
- `t3toolbox/backend/entries.py:117` | `t3_entries_ambient_transpose` | index: ONE-ARRAY subscript | `ww = tuple(xnp.eye(N)[index[i]] for i, N in enumerate(shape))`
- `t3toolbox/backend/entries.py:144` | `tv_entries` | index: mention | `(Jp,) = sampling_variables.sampling_variables('entries', frame, index, ('Jp',), variations=variations)`
- `t3toolbox/backend/entries.py:185` | `tv_entries_transpose` | index: mention | `(blocks,) = sampling_variables.sampling_variables('entries', frame, index, ('JtJp',), held=held,`
- `t3toolbox/backend/entries.py:215` | `tv_precompute_entries_frame_sweep` | index: mention | `xis = _entry_xis(down_tucker_cores, index)`
- `t3toolbox/backend/entries.py:241` | `tv_entries_jacobian_from_sweep` | index: mention | `dxis = _entry_xis(var_tucker_cores, index)    # fiber slice; variation-dependent, not in the frame sweep`
- `t3toolbox/backend/entries.py:263` | `tv_entries_transpose_from_sweep` | index: mention | `ww = _onehot_vectors(index, down_tucker_cores)`
- `t3toolbox/backend/entries.py:302` | `t3_entries_corewise_tangent` | index: mention | `(Jp,) = sampling_variables.sampling_variables('entries', x, index, ('Jp',),`
- `t3toolbox/backend/entries.py:344` | `t3_entries_corewise_transpose` | index: mention | `(blocks,) = sampling_variables.sampling_variables('entries', x, index, ('JtJp',), held=held,`
- `t3toolbox/backend/entries.py:372` | `t3_entries_corewise_hessian_from_sweep` | index: mention | `xis_dot = _entry_xis(tucker_dirs, index)                                     # xi_dot = Bdot[..., index] (fiber slice)`
- `t3toolbox/backend/entries.py:374` | `t3_entries_corewise_hessian_from_sweep` | index: mention | `ww = _onehot_vectors(index, tucker_cores)                                    # the one-hot "apply vectors"`
- `t3toolbox/backend/entries.py:417` | `t3_entries_corewise_hessian` | index: mention | `(blocks,) = sampling_variables.sampling_variables('entries', x, index, ('Kp',), held=held,`
- `t3toolbox/backend/entries.py:517` | `dense_entries` | index: mention | `xnp, _, _ = get_backend(True, tree_contains_jax((T, index)))`
- `t3toolbox/backend/entries.py:519` | `dense_entries` | index: ONE-ARRAY asarray(; index: mention | `index = xnp.asarray(index)`
- `t3toolbox/backend/entries.py:520` | `dense_entries` | index: ONE-ARRAY .shape | `d = index.shape[0]`
- `t3toolbox/backend/entries.py:523` | `dense_entries` | index: ONE-ARRAY .ndim | `n_C, n_W = T.ndim - d, index.ndim - 1`
- `t3toolbox/backend/entries.py:524` | `dense_entries` | index: ONE-ARRAY subscript | `z_CW = T[(Ellipsis,) + tuple(index[ii] for ii in range(d))]                 # indexing: no arithmetic`
- `t3toolbox/backend/entries.py:538` | `_entry_xis` | index: ONE-ARRAY array(; index: mention | `index = xnp.array(index)`
- `t3toolbox/backend/entries.py:547` | `_entry_xis` | index: ONE-ARRAY .ndim | `n_W = index.ndim - 1`
- `t3toolbox/backend/entries.py:549` | `_entry_xis` | index: mention | `return tucker_cores[d_idx, ..., index]                   # (d,) + W + C + (p,)`
- `t3toolbox/backend/entries.py:551` | `_entry_xis` | index: ONE-ARRAY .shape | `n_idx = len(index.shape[1:])                                  # number of index-stack (W) axes`
- `t3toolbox/backend/entries.py:554` | `_entry_xis` | index: ONE-ARRAY subscript | `xi_CpW = B[..., index[i]]                                 # C + (p,) + W (index batch trails)`
- `t3toolbox/backend/entries.py:563` | `_onehot_vectors` | index: mention | `use_jax = tree_contains_jax((index, down_tucker_cores))`
- `t3toolbox/backend/entries.py:566` | `_onehot_vectors` | index: ONE-ARRAY array(; index: mention | `index = xnp.array(index)`
- `t3toolbox/backend/entries.py:572` | `_onehot_vectors` | index: mention | `return xnp.eye(N)[index]`
- `t3toolbox/backend/entries.py:573` | `_onehot_vectors` | index: ONE-ARRAY subscript | `return tuple(xnp.eye(B.shape[-1])[index[i]] for i, B in enumerate(down_tucker_cores))`

(rows: ONE-ARRAY 13, mention 22)

**`t3toolbox/backend/fitting.py`**

- `t3toolbox/backend/fitting.py:189` | `_take_entries` | index: ONE-ARRAY .ndim | `n_w = index.ndim - 1`
- `t3toolbox/backend/fitting.py:190` | `_take_entries` | idx: mention; index: mention | `return _flat_gather(index, 1, n_w, idx), _flat_gather(data, 0, n_w, idx)`
- `t3toolbox/backend/fitting.py:198` | `_take_deriv_entries` | index: ONE-ARRAY .ndim; index: mention; sample: mention | `index, pp = sample; n_w = index.ndim - 1`
- `t3toolbox/backend/fitting.py:199` | `_take_deriv_entries` | idx: mention; index: mention | `return (_flat_gather(index, 1, n_w, idx), [_flat_gather(p, 0, n_w, idx) for p in pp]), \`
- `t3toolbox/backend/fitting.py:200` | `_take_deriv_entries` | idx: mention | `_flat_gather(data, 1, n_w, idx)`
- `t3toolbox/backend/fitting.py:549` | `ApplyKind.take` | idx: mention; sample: mention | `return _take_apply(sample, data, idx)`
- `t3toolbox/backend/fitting.py:576` | `EntriesKind.take` | idx: mention; sample: mention | `return _take_entries(sample, data, idx)`
- `t3toolbox/backend/fitting.py:615` | `ProbeKind.take` | idx: mention; sample: mention | `return _take_probe(sample, data, idx)`
- `t3toolbox/backend/fitting.py:661` | `ApplyDerivativesKind.take` | idx: mention; sample: mention | `return _take_deriv_apply(sample, data, idx)`
- `t3toolbox/backend/fitting.py:690` | `EntriesDerivativesKind.take` | idx: mention; sample: mention | `return _take_deriv_entries(sample, data, idx)`
- `t3toolbox/backend/fitting.py:723` | `ProbeDerivativesKind.take` | idx: mention; sample: mention | `return _take_deriv_probe(sample, data, idx)`
- `t3toolbox/backend/fitting.py:1230` | `LeastSquaresOracle.take` | indices: mention; sample: mention | `sample = _flat_gather(batch.sample, 1, n_w, indices)`
- `t3toolbox/backend/fitting.py:1232` | `LeastSquaresOracle.take` | indices: mention; sample: mention | `sample = [_flat_gather(w, m, n_w, indices) for w in batch.sample]`
- `t3toolbox/backend/fitting.py:1234` | `LeastSquaresOracle.take` | indices: mention; sample: mention | `sample = _flat_gather(batch.sample, 1 + m, n_w, indices)`
- `t3toolbox/backend/fitting.py:1235` | `LeastSquaresOracle.take` | indices: mention | `y = _take_data_space(self.kind, m, W, batch.y, indices, 'the data y')`
- `t3toolbox/backend/fitting.py:1236` | `LeastSquaresOracle.take` | indices: mention | `weight = _take_data_space(self.kind, m, W, batch.weight, indices, 'the weight')`
- `t3toolbox/backend/fitting.py:1237` | `LeastSquaresOracle.take` | indices: ONE-ARRAY len( | `scale = n / len(indices)`
- `t3toolbox/backend/fitting.py:1240` | `LeastSquaresOracle.take` | indices: mention; sample: mention | `return LeastSquaresBatch(sample, y, weight, scale, indices)`

(rows: ONE-ARRAY 3, mention 29)

**`t3toolbox/backend/probing.py`**

- NOT FOUND: `dense_entries`

(rows: ONE-ARRAY 0, mention 0)

**`t3toolbox/tucker_tensor_train.py`**

- `t3toolbox/tucker_tensor_train.py:68` | `_as_index` | index: ONE-ARRAY asarray(; index: mention | `return index if common.is_ndarray(index) else np.asarray(index)`

(rows: ONE-ARRAY 1, mention 1)

**`t3toolbox/manifold.py`**

- NOT FOUND: `_as_index`

(rows: ONE-ARRAY 0, mention 0)

**`t3toolbox/uniform_tucker_tensor_train.py`**

- NOT FOUND: `_as_index`

(rows: ONE-ARRAY 0, mention 0)

**`t3toolbox/uniform_manifold.py`**

- NOT FOUND: `_as_index`

(rows: ONE-ARRAY 0, mention 0)

Notes. `dense_entries` is defined in `t3toolbox/backend/entries.py:427` (not `probing.py`; no such name in `probing.py`), so its rows are under `entries.py` above. `_as_index` is DEFINED only in `t3toolbox/tucker_tensor_train.py:63` and imported/called from `manifold.py`, `uniform_manifold.py`, `uniform_tucker_tensor_train.py`; its 13 library call sites (file:line, enclosing function) are listed under M3 (`_as_index`). Functions `xi_gather_step` and `V_scatter_step` were found and contain no index-like identifier (rows: none). In `fitting.py` the old-kind `take` methods (`ApplyKind.take` ... , deleted at P20) are in the same list as `LeastSquaresOracle.take`; the class name is in the enclosing-function column.

## M8. Sizes

Method: `wc -l` semantics (number of newline characters) of every `.py` file in `t3toolbox/` and `t3toolbox/backend/` (not `__pycache__`); tests by `ast` (a test = any `def test_*` / `async def test_*` anywhere in a `tests/**/*.py` file, methods included); `>>>` lines = lines whose stripped text starts with `>>>` in library `.py` files.

**`t3toolbox/`** (12 files, 15776 lines total)

| file | lines |
|---|---|
| `__init__.py` | 117 |
| `corewise.py` | 416 |
| `fitting.py` | 722 |
| `frame_variations_format.py` | 1933 |
| `manifold.py` | 1552 |
| `optimizers.py` | 868 |
| `safety.py` | 275 |
| `shared_geometry.py` | 433 |
| `tucker_tensor_train.py` | 5106 |
| `uniform_frame_variations_format.py` | 1701 |
| `uniform_manifold.py` | 1204 |
| `uniform_tucker_tensor_train.py` | 1449 |

**`t3toolbox/backend/`** (48 files, 40866 lines total)

| file | lines |
|---|---|
| `__init__.py` | 7 |
| `apply.py` | 702 |
| `common.py` | 812 |
| `contractions.py` | 977 |
| `data_preparation.py` | 722 |
| `entries.py` | 573 |
| `fitting.py` | 1240 |
| `fv_conversions.py` | 267 |
| `fv_operations.py` | 1118 |
| `geometry.py` | 1270 |
| `jets.py` | 1735 |
| `linalg.py` | 1091 |
| `optimizer_display.py` | 676 |
| `optimizers.py` | 2546 |
| `padded_frames.py` | 857 |
| `probing.py` | 1827 |
| `ranks.py` | 758 |
| `regularization.py` | 98 |
| `sampling_definitions.py` | 3308 |
| `sampling_derivatives.py` | 3534 |
| `sampling_steps.py` | 1142 |
| `sampling_variables.py` | 1936 |
| `sharing.py` | 1220 |
| `stacking.py` | 530 |
| `t3_constructors.py` | 95 |
| `t3_conversions.py` | 198 |
| `t3_linalg.py` | 1000 |
| `t3_operations.py` | 424 |
| `t3_orthogonalization.py` | 426 |
| `t3_svd.py` | 804 |
| `tt_operations.py` | 296 |
| `tt_orthogonalization.py` | 139 |
| `tv_operations.py` | 894 |
| `ufv_conversions.py` | 430 |
| `ufv_masking.py` | 225 |
| `ufv_operations.py` | 882 |
| `uniform_fitting.py` | 518 |
| `uniform_padded_frames.py` | 1030 |
| `ut3_constructors.py` | 185 |
| `ut3_conversions.py` | 273 |
| `ut3_linalg.py` | 434 |
| `ut3_masking.py` | 89 |
| `ut3_operations.py` | 522 |
| `ut3_orthogonalization.py` | 260 |
| `ut3_sampling.py` | 744 |
| `ut3_svd.py` | 667 |
| `utv_operations.py` | 794 |
| `utv_sampling.py` | 591 |

**tests/**: 88 `.py` files under `tests/` (all), of which 70 are named `test_*.py`; files with at least one `def test_`: 70; total `def test_` functions: 1982.

Largest 10 test files by `def test_` count: `tests/test_dispatch.py` 108, `tests/test_uniform_manifold.py` 103, `tests/backend/test_curvature.py` 96, `tests/test_tucker_tensor_train.py` 81, `tests/test_uniform_frame_variations_format.py` 80, `tests/test_sharing.py` 78, `tests/backend/test_optimizers.py` 77, `tests/test_sampling_definitions.py` 72, `tests/test_optimizers_frontend.py` 64, `tests/test_uniform_tucker_tensor_train.py` 61

**`>>>` lines per library module, top 15** (total in library: 4813 over 43 files with any):

| module | `>>>` lines |
|---|---|
| `t3toolbox/tucker_tensor_train.py` | 1148 |
| `t3toolbox/backend/sampling_definitions.py` | 772 |
| `t3toolbox/frame_variations_format.py` | 225 |
| `t3toolbox/backend/jets.py` | 195 |
| `t3toolbox/backend/linalg.py` | 184 |
| `t3toolbox/backend/stacking.py` | 176 |
| `t3toolbox/manifold.py` | 164 |
| `t3toolbox/backend/sharing.py` | 148 |
| `t3toolbox/backend/probing.py` | 133 |
| `t3toolbox/uniform_tucker_tensor_train.py` | 125 |
| `t3toolbox/backend/data_preparation.py` | 113 |
| `t3toolbox/uniform_frame_variations_format.py` | 105 |
| `t3toolbox/backend/apply.py` | 98 |
| `t3toolbox/optimizers.py` | 94 |
| `t3toolbox/backend/entries.py` | 93 |

