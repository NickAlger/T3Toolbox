import numpy as np, inspect, warnings
warnings.filterwarnings('ignore')
import t3toolbox as t3t
from t3toolbox import manifold, uniform_manifold, tucker_tensor_train as ttt
from t3toolbox.backend import contractions
np.random.seed(0)
x = t3t.TuckerTensorTrain.randn((5,6,7),(2,3,2),(1,2,2,1))
print('user_guide:31/215 tucker core shapes (claimed (n_i, N_i)):', [c.shape for c in x.tucker_cores], ' tt:', [c.shape for c in x.tt_cores])
fr, var = t3t.t3_orthogonal_representations(x)
print('frame up_tucker_cores shapes:', [c.shape for c in fr.up_tucker_cores])
print('user_guide:156 MANIFOLD.transport exists:', hasattr(t3t.MANIFOLD,'transport'), ' methods:', [m for m in dir(t3t.MANIFOLD) if not m.startswith('_')])
print('user_guide:87 sum_stack_corewise exists:', hasattr(t3t.TuckerTensorTrain,'sum_stack_corewise'), [m for m in dir(t3t.TuckerTensorTrain) if 'corewise' in m])
print('user_guide:231 contractions.__all__ =', contractions.__all__)
print('user_guide:223 apply_transpose sig:', inspect.signature(manifold.T3Tangent.apply_transpose))
print('user_guide:223 entries_transpose sig:', inspect.signature(manifold.T3Tangent.entries_transpose))
ux = x.to_uniform() if hasattr(x,'to_uniform') else t3t.UniformTuckerTensorTrain.from_t3(x)
print('user_guide:248-251 supercore shapes: tucker', ux.tucker_supercore.shape, 'tt', ux.tt_supercore.shape, ' masks:', ux.masks.tucker_rank_masks.shape if hasattr(ux.masks,'tucker_rank_masks') else [ (k, getattr(ux.masks,k).shape) for k in dir(ux.masks) if 'mask' in k and not k.startswith('_')])
# stacked x0 -> NotImplementedError at entry (release_notes:47, CHANGELOG:37)
A = t3t.TuckerTensorTrain.randn((4,5,3),(2,2,2),(1,2,2,1))
ww = [np.random.randn(30, N) for N in A.shape]; ww=[w/np.linalg.norm(w,axis=1,keepdims=True) for w in ww]
b = A.apply(ww)
xs = t3t.TuckerTensorTrain.zeros((4,5,3),(2,2,2),(1,2,2,1), stack_shape=(2,))
for name, fn, extra in [('newton_cg', t3t.newton_cg, dict(max_newton=2)), ('gradient_descent', t3t.gradient_descent, dict(max_iter=2)), ('mc_sgd', t3t.mc_sgd, dict(rng=np.random.default_rng(0), batch=5, max_iter=2)), ('adam', t3t.adam, dict(rng=np.random.default_rng(0), batch=5, max_iter=2))]:
    try:
        fn(t3t.MANIFOLD, 'apply', ww, b, xs, **extra); print(name, 'stacked x0: NO ERROR')
    except NotImplementedError as e: print(name, 'stacked x0: NotImplementedError:', str(e)[:90])
    except Exception as e: print(name, 'stacked x0: OTHER', type(e).__name__, str(e)[:120])
# user_guide:272-277 uniform fitting requires minimal-rank x0 "says so with an error"; "frontend handles it transparently"
x0_nonmin = t3t.UniformTuckerTensorTrain.from_t3(t3t.TuckerTensorTrain.zeros((4,5,3),(4,4,3),(1,4,4,1)))
print('nonminimal uniform x0 ranks', x0_nonmin.tucker_ranks, x0_nonmin.tt_ranks, 'minimal?', x0_nonmin.has_minimal_ranks() if hasattr(x0_nonmin,'has_minimal_ranks') else '?')
try:
    xf, st = t3t.newton_cg(t3t.UNIFORM_MANIFOLD, 'apply', ww, b, x0_nonmin, max_newton=2)
    print('frontend newton_cg non-minimal uniform x0: ran; result ranks', xf.tucker_ranks, xf.tt_ranks)
except Exception as e: print('frontend newton_cg non-minimal uniform x0 RAISED', type(e).__name__, str(e)[:150])
