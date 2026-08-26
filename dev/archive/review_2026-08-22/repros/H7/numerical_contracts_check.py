import numpy as np, warnings, inspect; warnings.filterwarnings('ignore')
import t3toolbox as t3t
from t3toolbox import manifold, safety
np.random.seed(0)
shape, tr, ttr = (4,5,3), (2,3,2), (1,2,2,1)
x = t3t.TuckerTensorTrain.randn(shape, tr, ttr)
frame, var = t3t.t3_orthogonal_representations(x)
print('frame orthogonal?', frame.is_orthogonal())
# non-orthogonal frame with same structure: perturb cores
def perturb(cores): return tuple(c + 0.3*np.random.randn(*c.shape) for c in cores)
bad_frame = t3t.T3Frame(perturb(frame.up_tucker_cores), perturb(frame.down_tt_cores), perturb(frame.left_tt_cores), perturb(frame.right_tt_cores))
print('bad_frame orthogonal?', bad_frame.is_orthogonal())
v = t3t.MANIFOLD.randn(frame)
v_bad = t3t.T3Tangent(bad_frame, v.variations)           # non-orthogonal frame
ungauged = t3t.T3Tangent(frame, t3t.T3Variations(perturb(v.variations.tucker_variations), perturb(v.variations.tt_variations)))
print('ungauged is_gauged?', ungauged.is_gauged())
y = t3t.TuckerTensorTrain.randn(shape, tr, ttr); frame2,_ = t3t.t3_orthogonal_representations(y); v2 = t3t.MANIFOLD.randn(frame2)
def expect(label, fn, should_raise):
    try:
        fn(); res = 'no raise'
    except Exception as e: res = f'RAISED {type(e).__name__}'
    flag = 'OK ' if (should_raise == res.startswith('RAISED')) else 'MISMATCH '
    print(f'{flag}{label}: doc says {"raises" if should_raise else "no check"} -> {res}')
M, CW = t3t.MANIFOLD, t3t.COREWISE
expect('T3Tangent + (SF)', lambda: v + v2, True)
expect('T3Tangent.allclose (SF)', lambda: v.allclose(v2), True)
expect('corewise_inner (SF)', lambda: v.corewise_inner(v2), True)
expect('corewise_inner ungauged same frame (no check)', lambda: v.corewise_inner(ungauged), False)
expect('corewise_norm bad frame (no check)', lambda: v_bad.corewise_norm(), False)
expect('MANIFOLD.inner SF', lambda: M.inner(v, v2), True)
expect('MANIFOLD.inner ungauged (GAUGE)', lambda: M.inner(v, ungauged), True)
expect('MANIFOLD.inner bad frame (ORTH)', lambda: M.inner(v_bad, v_bad), True)
expect('MANIFOLD.norm ungauged (GAUGE)', lambda: M.norm(ungauged), True)
expect('MANIFOLD.norm bad frame (ORTH)', lambda: M.norm(v_bad), True)
expect('COREWISE.inner SF', lambda: CW.inner(v, v2), True)
expect('COREWISE.inner ungauged (no check)', lambda: CW.inner(v, ungauged), False)
expect('COREWISE.norm bad frame (no check)', lambda: CW.norm(v_bad), False)
expect('MANIFOLD.project bad frame (ORTH)', lambda: M.project(v_bad), True)
expect('MANIFOLD.project ungauged (ORTH only)', lambda: M.project(ungauged), False)
expect('MANIFOLD.project_oblique bad frame (ORTH)', lambda: M.project_oblique(v_bad), True)
expect('MANIFOLD.retract bad frame (ORTH)', lambda: M.retract(v_bad), True)
expect('MANIFOLD.retract ungauged (ORTH only)', lambda: M.retract(ungauged), False)
expect('MANIFOLD.project_ambient bad frame (ORTH)', lambda: M.project_ambient(bad_frame, np.random.randn(*shape)), True)
expect('MANIFOLD.transport to bad frame (ORTH)', lambda: M.transport(v, bad_frame), True)
expect('MANIFOLD.randn bad frame (ORTH)', lambda: M.randn(bad_frame), True)
expect('MANIFOLD.random_orthogonal exists', lambda: M.random_orthogonal(frame), False)
expect('MANIFOLD.randn_like', lambda: M.randn_like(v), False)
expect('COREWISE.project bad frame (no check)', lambda: CW.project(v_bad), False)
expect('COREWISE.retract bad frame (no check)', lambda: CW.retract(v_bad), False)
expect('stack_tangents SF', lambda: t3t.T3Tangent.stack_tangents([v, v2]), True)
for n in ['sum_tangents','unit','zeros_like','normalized','reverse','to_vector','from_vector','stack_frame','unstack_tangents','unstack_frame','gauge_residual','has_numerically_minimal_ranks','to_t3','to_dense']:
    print('  T3Tangent.'+n, 'exists' if hasattr(t3t.T3Tangent, n) else 'MISSING')
for n in ['orthogonality_residual','random_orthogonal','up_ranks','down_ranks','is_consistent']:
    print('  T3Frame.'+n, 'exists' if hasattr(t3t.T3Frame, n) else 'MISSING')
print('  manifold.manifold_dim', hasattr(manifold,'manifold_dim'), ' tangent_space_dimension:', [m for m in dir(t3t.T3Tangent)+dir(t3t.T3Frame)+dir(manifold) if 'dimension' in m])
print('  TuckerTensorTrain.inner sig', inspect.signature(t3t.TuckerTensorTrain.inner))
# GaussNewtonModel
ww = [np.random.randn(20, N) for N in shape]; b = x.apply(ww)
m = t3t.apply_model(M, x, ww, b - x.apply(ww))
for n in ['gradient','gn_hessian','jacobian','gn_quadratic','evaluate']:
    print('  GaussNewtonModel.'+n, 'exists' if hasattr(m, n) else 'MISSING')
p_other = t3t.MANIFOLD.randn(frame2)
expect('GaussNewtonModel.gn_hessian foreign p (SF)', lambda: m.gn_hessian(p_other), True)
expect('GaussNewtonModel.jacobian foreign p (SF)', lambda: m.jacobian(p_other), True)
expect('GaussNewtonModel.gn_quadratic foreign p (SF)', lambda: m.gn_quadratic(p_other), True)
expect('GaussNewtonModel.evaluate foreign p (SF)', lambda: m.evaluate(p_other), True)
# TIED
xu = t3t.TuckerTensorTrain.randn((4,4,3),(2,2,2),(1,2,2,1))
expect('t3svd(sharing) untied (TIED)', lambda: xu.t3svd(sharing=(0,0,1)), True)
expect('rank_adjustment_sweep(sharing) untied (TIED)', lambda: xu.rank_adjustment_sweep(sharing=(0,0,1)), True)
expect('resize(sharing) untied (TIED)', lambda: xu.resize((4,4,3),(3,3,2),(1,3,3,1), sharing=(0,0,1)), True)
expect('continuation_ranks(sharing) untied (TIED)', lambda: xu.continuation_ranks(sharing=(0,0,1)), True)
expect('share untied (no check)', lambda: xu.share((0,0,1)), False)
sm = t3t.shared_manifold((0,0,1))
expect('shared.frame untied (TIED) -- CHANGELOG says ties silently', lambda: sm.frame(xu), False)
xs = xu.share((0,0,1)); fs = sm.frame(xs); vs = sm.randn(fs)
expect('shared.transport to untied frame (TIED)', lambda: sm.transport(vs, t3t.t3_orthogonal_representations(xu)[0]), True)
