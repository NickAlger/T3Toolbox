import numpy as np, inspect, warnings
warnings.filterwarnings('ignore')
import t3toolbox as t3t
from t3toolbox.backend import optimizers as bo
print('backend gradient_descent sig', inspect.signature(bo.gradient_descent))
np.random.seed(0)
A = t3t.TuckerTensorTrain.randn((4,5,3),(2,2,2),(1,2,2,1))
ww = [np.random.randn(30, N) for N in A.shape]; ww=[w/np.linalg.norm(w,axis=1,keepdims=True) for w in ww]
b = A.apply(ww)
xs = t3t.TuckerTensorTrain.zeros((4,5,3),(2,2,2),(1,2,2,1), stack_shape=(2,))
try:
    t3t.gradient_descent(t3t.MANIFOLD, 'apply', ww, b, xs, max_iters=2); print('gradient_descent stacked x0: NO ERROR')
except NotImplementedError as e: print('gradient_descent stacked x0: NotImplementedError:', str(e)[:80])
except Exception as e: print('gradient_descent stacked x0: OTHER', type(e).__name__, str(e)[:150])
print('T3.has_minimal_ranks is', type(t3t.TuckerTensorTrain.__dict__.get('has_minimal_ranks')).__name__, '; UT3.has_minimal_ranks is', type(t3t.UniformTuckerTensorTrain.__dict__.get('has_minimal_ranks')).__name__)
for cls in (t3t.TuckerTensorTrain, t3t.UniformTuckerTensorTrain, t3t.T3Frame, t3t.UT3Frame, t3t.T3Variations, t3t.UT3Variations, t3t.T3Tangent, t3t.UT3Tangent, t3t.T3Weights, t3t.UT3Weights):
    kinds = {n: type(v).__name__ for n,v in cls.__dict__.items() if n.startswith(('has_','is_'))}
    print(cls.__name__, kinds)
x0_nonmin = t3t.UniformTuckerTensorTrain.from_t3(t3t.TuckerTensorTrain.zeros((4,5,3),(4,4,3),(1,4,4,1)))
print('nonminimal uniform x0 ranks', x0_nonmin.tucker_ranks, x0_nonmin.tt_ranks, 'minimal?', x0_nonmin.has_minimal_ranks)
try:
    xf, st = t3t.newton_cg(t3t.UNIFORM_MANIFOLD, 'apply', ww, b, x0_nonmin, max_newton=2)
    print('frontend newton_cg non-minimal uniform x0: ran; result ranks', xf.tucker_ranks, xf.tt_ranks)
except Exception as e: print('frontend newton_cg non-minimal uniform x0 RAISED', type(e).__name__, str(e)[:200])
