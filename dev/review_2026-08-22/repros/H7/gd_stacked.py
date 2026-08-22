import numpy as np, warnings; warnings.filterwarnings('ignore')
import t3toolbox as t3t
np.random.seed(0)
A = t3t.TuckerTensorTrain.randn((4,5,3),(2,2,2),(1,2,2,1))
ww = [np.random.randn(30, N) for N in A.shape]; ww=[w/np.linalg.norm(w,axis=1,keepdims=True) for w in ww]
b = A.apply(ww)
xs = t3t.TuckerTensorTrain.zeros((4,5,3),(2,2,2),(1,2,2,1), stack_shape=(2,))
try:
    t3t.gradient_descent(t3t.MANIFOLD, 'apply', ww, b, xs, n_iter=2); print('gradient_descent stacked x0: NO ERROR')
except NotImplementedError as e: print('gradient_descent stacked x0: NotImplementedError:', str(e)[:80])
except Exception as e: print('gradient_descent stacked x0: OTHER', type(e).__name__, str(e)[:150])
