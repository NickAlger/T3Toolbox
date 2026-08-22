import numpy as np
import t3toolbox as t3t

np.random.seed(0)
x = t3t.TuckerTensorTrain.randn((10, 11, 12), (3, 4, 3), (1, 2, 2, 1))
y = t3t.TuckerTensorTrain.randn((10, 11, 12), (2, 2, 2), (1, 2, 2, 1))

z = x + y
print('README:60 expects (5, 6, 5) (2, 4, 4, 2) ->', z.tucker_ranks, z.tt_ranks)
print('README:61 expects 0.0 ->', np.linalg.norm(z.to_dense() - (x.to_dense() + y.to_dense())))
z2, ss_tucker, ss_tt = z.t3svd()
print('README:64 expects (4, 6, 4) (1, 4, 4, 1) ->', z2.tucker_ranks, z2.tt_ranks)

ww = [np.random.randn(N) for N in x.shape]
zz = x.probe(ww)
print('probe ->', [v.shape for v in zz])
a  = x.apply(ww)
print('apply ->', np.shape(a))
try:
    e  = x.entries(np.array([3, 1, 2]))
    print('entries ->', np.shape(e), e, 'dense:', x.to_dense()[3,1,2])
except Exception as ex:
    print('README:74 entries RAISES:', type(ex).__name__, ex)

A  = t3t.TuckerTensorTrain.randn((6, 7, 8), (2, 2, 2), (1, 2, 2, 1))
ww = [np.random.randn(120, N) for N in A.shape]
ww = [w / np.linalg.norm(w, axis=1, keepdims=True) for w in ww]
b  = A.apply(ww)
x0 = t3t.TuckerTensorTrain.zeros((6, 7, 8), (2, 2, 2), (1, 2, 2, 1))
x_fit, stats = t3t.newton_cg(t3t.MANIFOLD, 'apply', ww, b, x0, max_newton=30)
rel = np.linalg.norm(x_fit.to_dense() - A.to_dense()) / np.linalg.norm(A.to_dense())
print('README:88 expects relerr < 1e-6 ->', rel)
