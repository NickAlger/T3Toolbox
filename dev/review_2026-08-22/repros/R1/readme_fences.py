import numpy as np
import t3toolbox as t3t

np.random.seed(0)
x = t3t.TuckerTensorTrain.randn((10, 11, 12), (3, 4, 3), (1, 2, 2, 1))
y = t3t.TuckerTensorTrain.randn((10, 11, 12), (2, 2, 2), (1, 2, 2, 1))

z = x + y
print(z.tucker_ranks, z.tt_ranks)      # (5, 6, 5) (2, 4, 4, 2)   <- ranks add ...
print(np.linalg.norm(z.to_dense() - (x.to_dense() + y.to_dense())))  # ... tensors add: 0.0

z2, ss_tucker, ss_tt = z.t3svd()       # reduce to minimal ranks (lossless)
print(z2.tucker_ranks, z2.tt_ranks)    # (4, 6, 4) (1, 4, 4, 1)
ww = [np.random.randn(N) for N in x.shape]
zz = x.probe(ww)                       # d vectors, one per mode (all but one mode contracted)
a  = x.apply(ww)                       # a scalar (all modes contracted)
e  = x.entries(np.array([3, 1, 2]))    # one entry
print([z.shape for z in zz], np.shape(a), np.shape(e))
A  = t3t.TuckerTensorTrain.randn((6, 7, 8), (2, 2, 2), (1, 2, 2, 1))     # the unknown target
ww = [np.random.randn(120, N) for N in A.shape]
ww = [w / np.linalg.norm(w, axis=1, keepdims=True) for w in ww]          # unit-norm rows
b  = A.apply(ww)                                                         # 120 measurements

x0 = t3t.TuckerTensorTrain.zeros((6, 7, 8), (2, 2, 2), (1, 2, 2, 1))
x_fit, stats = t3t.newton_cg(t3t.MANIFOLD, 'apply', ww, b, x0, max_newton=30)
print('rel err', np.linalg.norm(x_fit.to_dense()-A.to_dense())/np.linalg.norm(A.to_dense()))
