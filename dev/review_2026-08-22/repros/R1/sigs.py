import ast, sys
REPO='/home/nick/repos/T3Toolbox/t3toolbox/'
want = {
 'backend/t3_operations.py': ['t3_core_shapes','t3_segment','t3_concatenate','t3_squash_tails','tucker_change_core_shapes','t3_sum','t3_unstack','t3_stack','t3_weights_consistent','t3_concatenate_weights','t3_kronecker_weights','t3_absorb_weights'],
 'backend/tt_operations.py': ['tt_reverse','tt_change_core_shapes'],
 'backend/t3_conversions.py': ['t3_to_dense','t3_from_canonical','t3_from_tensor_train','t3_to_tensor_train','t3_to_vector','t3_from_vector'],
 'backend/t3_constructors.py': ['t3_zeros','t3_ones','t3_corewise_randn'],
 'backend/t3_linalg.py': ['t3_add','t3_plus_scalar','t3_scale','t3_inner_product','t3_norm','t3_sum_stack','t3_weighted_norm','t3_weighted_inner','t3m_form_then_round','t3m_inplace_fused','t3m_swap'],
 'backend/t3_orthogonalization.py': ['t3_orthogonality_residual','t3_down_svd_tucker_core','t3_left_svd_tt_core','t3_right_svd_tt_core','t3_up_svd_tt_core','t3_orthogonalize_relative_to_tucker_core','t3_orthogonalize_relative_to_tt_core','t3_down_orthogonalize_tucker_cores','t3_up_orthogonalize_tt_cores'],
 'backend/tt_orthogonalization.py': ['tt_left_orthogonalize','tt_right_orthogonalize'],
 'backend/probing.py': ['t3_probe','t3_probe_ambient_transpose','t3_probe_corewise_transpose'],
 'backend/apply.py': ['t3_apply','t3_apply_ambient_transpose','t3_apply_corewise_transpose'],
 'backend/entries.py': ['t3_entries','t3_entries_ambient_transpose','t3_entries_corewise_transpose'],
 'backend/sampling_derivatives.py': ['check_perturbation_vectors','check_perturbation_index','t3_probe_derivatives','t3_apply_derivatives','t3_entries_derivatives','t3_probe_corewise_derivatives_transpose','t3_apply_corewise_derivatives_transpose','t3_entries_corewise_derivatives_transpose'],
 'backend/ranks.py': ['compute_minimal_ranks','compute_continuation_ranks'],
 'backend/t3_svd.py': ['t3svd','t3_rank_adjustment_sweep','t3_share_tucker_factors','dense_t3svd'],
 'backend/sharing.py': ['t3_sharing_residual','t3_tie_tucker_factors','t3_tucker_factors_shared','t3_tucker_weights_shared'],
 'backend/stacking.py': ['basic_ragged_unstack','basic_ragged_stack','apply_func_to_leaf_subtrees'],
 'corewise.py': ['corewise_stack_sum'],
 'backend/common.py': ['save_core_families','load_core_families'],
}
for f, names in want.items():
    src = open(REPO+f).read().splitlines()
    tree = ast.parse('\n'.join(src))
    found=set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in names:
            found.add(node.name)
            end = node.body[0].lineno - 1
            print(f'--- {f}:{node.lineno}')
            print('\n'.join(src[node.lineno-1:end]))
    for n in names:
        if n not in found: print(f'!!! NOT FOUND {f}: {n}')
