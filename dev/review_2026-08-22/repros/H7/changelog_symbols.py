import sys, inspect, importlib
sys.path.insert(0,'/home/nick/repos/T3Toolbox')
import t3toolbox as t3t
from t3toolbox.backend import geometry, fitting as bf, optimizers as bo, sharing, t3_svd, ranks, common, sampling_derivatives as sd, fv_conversions, ufv_conversions, ufv_operations, utv_operations, ut3_operations, ut3_svd, uniform_fitting, optimizer_display, fv_operations, ut3_masking
from t3toolbox import manifold, fitting, optimizers, tucker_tensor_train as ttt, uniform_tucker_tensor_train as uttt, frame_variations_format as fvf, uniform_frame_variations_format as ufvf
def has(mod, name):
    ok = hasattr(mod, name)
    print(('OK  ' if ok else 'MISSING ') + f'{mod.__name__}.{name}')
    return ok
def sig(obj):
    try: return str(inspect.signature(obj))
    except Exception as e: return f'<{e}>'
# 2026.2.0
for n in ['ManifoldGeometryOps','CorewiseGeometryOps','UniformManifoldGeometryOps','UniformCorewiseGeometryOps']: has(geometry, n)
print(' from_point:', hasattr(geometry.UniformManifoldGeometryOps,'from_point'), sig(geometry.UniformManifoldGeometryOps.from_point))
print(' with_sharing ragged:', sig(geometry.ManifoldGeometryOps.with_sharing), ' uniform:', sig(geometry.UniformManifoldGeometryOps.with_sharing))
for n in ['ApplyKind','EntriesKind','ProbeKind','ApplyDerivativesKind','EntriesDerivativesKind','ProbeDerivativesKind','UniformApplyKind','UniformEntriesKind','UniformProbeKind','UniformApplyDerivativesKind','UniformEntriesDerivativesKind','UniformProbeDerivativesKind','ScalarOutputKind','ProbeOutputKind','SamplingKind','apply_kind','entries_kind','probe_kind','apply_derivatives_kind','entries_derivatives_kind','probe_derivatives_kind']: has(bf, n)
for n in ['uniform_sampling_kind','uniform_derivatives_kind']: has(uniform_fitting, n); has(bf, n)
print(' UniformApplyKind.from_point', sig(bf.UniformApplyKind.from_point) if hasattr(bf,'UniformApplyKind') else None)
print(' has_block_sumsq on ApplyKind:', hasattr(bf.ApplyKind, 'has_block_sumsq'))
print('bf.__all__=', bf.__all__)
print('Geometry protocol members:', [n for n in dir(bo.Geometry) if not n.startswith('_')] if hasattr(bo,'Geometry') else 'NO Geometry')
for n in ['GeometryOps','MANIFOLD_OPS','COREWISE_OPS','shared_geometry_ops']: print('removed?', n, not hasattr(bo,n))
for n in ['uniform_manifold_ops','uniform_corewise_ops','uniform_geometry_ops','uniform_apply_kind','uniform_apply_derivatives_kind']: print('removed?', n, not hasattr(uniform_fitting,n))
print('UniformGaussNewtonModel removed?', not hasattr(fitting,'UniformGaussNewtonModel'))
print('_cg_solve sig:', sig(bo._cg_solve) if hasattr(bo,'_cg_solve') else 'MISSING')
# 2026.1.0
print('has_shared_tucker_factors', sig(ttt.TuckerTensorTrain.has_shared_tucker_factors))
print('T3Weights.has_shared_tucker_weights', sig(ttt.T3Weights.has_shared_tucker_weights))
print('UT3Weights.has_shared_tucker_weights', sig(uttt.UT3Weights.has_shared_tucker_weights))
print('manifold_dim', sig(manifold.manifold_dim))
has(fvf, 'frame_has_minimal_ranks') or has(fv_conversions,'frame_has_minimal_ranks') or print([m for m in dir(t3t) if 'minimal' in m])
import t3toolbox.backend as B
for modname in B.__all__ if hasattr(B,'__all__') else []: pass
# grep whole package for frame_has_minimal_ranks
import subprocess
print(subprocess.run(['grep','-rn','def frame_has_minimal_ranks','/home/nick/repos/T3Toolbox/t3toolbox'],capture_output=True,text=True).stdout)
print('continuation_ranks', sig(ttt.TuckerTensorTrain.continuation_ranks))
print('compute_continuation_ranks', sig(ranks.compute_continuation_ranks))
print('resize', sig(ttt.TuckerTensorTrain.resize))
print('get_minimal_ranks', sig(ttt.TuckerTensorTrain.get_minimal_ranks))
print('compute_minimal_ranks', sig(ranks.compute_minimal_ranks))
for n in ['validate_sharing','t3_sharing_residual','t3_tucker_factors_shared','t3_tie_tucker_factors','SharedFrameData','fv_shared_frame_data','t3_tucker_weights_sharing_residual','t3_tucker_weights_shared','ut3_tucker_weights_sharing_residual','ut3_tucker_weights_shared','ut3_sharing_residual','ut3_tucker_factors_shared','ut3_tie_tucker_factors','ufv_shared_frame_data','ufv_share_tucker_variations','ufv_share_tucker_variations_corewise','fv_tied_variations_residual','ufv_tied_variations_residual']: has(sharing, n)
has(t3_svd,'t3_share_tucker_factors')
print('compute_raw_sweep_ranks', sig(ranks.compute_raw_sweep_ranks) if hasattr(ranks,'compute_raw_sweep_ranks') else 'MISSING')
print('uniform_minimal', [m for m in dir(ranks) if 'uniform_minimal' in m], [m for m in dir(uniform_fitting) if 'minimal' in m])
print('utv_orthogonal_gauge_projection', sig(utv_operations.utv_orthogonal_gauge_projection))
print('utv_to_ut3', sig(utv_operations.utv_to_ut3)); print('utv_retract', sig(utv_operations.utv_retract))
has(uniform_fitting,'uniform_least_squares_problem'); 
print('ut3svd', sig(ut3_svd.ut3svd)); 
for n in ['compute_mu_jets','compute_eta_jets','compute_sigma_jets','compute_tau_jets','compute_deta_jets','assemble_tt_variation_jets','assemble_tucker_variation_jets','compute_mu_jets_trs','compute_eta_jets_trs','estimate_chunk_size','max_chunk_size_within']: has(sd, n)
print([n for n in sd.__all__ if 'tilde' in n or '_trs' in n])
print('estimate_chunk_size', sig(sd.estimate_chunk_size)); print('max_chunk_size_within', sig(sd.max_chunk_size_within))
for n in ['prefix_mask','require_concrete_masks']: has(common, n)
print('require_concrete_masks in ut3_masking?', hasattr(ut3_masking,'require_concrete_masks'))
print('newton_cg', sig(optimizers.newton_cg)); print('mc_sgd', sig(optimizers.mc_sgd)); print('adam', sig(optimizers.adam)); print('gradient_descent', sig(optimizers.gradient_descent))
print('backend newton_cg', sig(bo.newton_cg))
print('NewtonInfo fields', [f for f in getattr(bo.NewtonInfo,'__dataclass_fields__',{})] if hasattr(bo,'NewtonInfo') else 'MISSING')
has(optimizer_display,'make_newton_display')
print('probe_model', sig(fitting.probe_model)); print('probe_derivatives_model', sig(fitting.probe_derivatives_model)); print('apply_model', sig(fitting.apply_model))
print('T3Tangent.probe_derivatives_transpose', sig(manifold.T3Tangent.probe_derivatives_transpose))
import t3toolbox.backend.regularization as R
print('IdentityRegularizer methods', [m for m in dir(R.IdentityRegularizer) if not m.startswith('_')], sig(R.IdentityRegularizer.gradient))
print('Regularizer protocol members', [m for m in dir(R.Regularizer) if not m.startswith('_')])
print('safety __all__', t3t.safety.__all__)
print('root __all__', t3t.__all__)
