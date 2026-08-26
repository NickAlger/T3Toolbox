"""UniformTuckerTensorTrain.norm() after inner()/arithmetic on the same (padded) object -- sweep-order replication."""
import numpy as np
import t3toolbox.tucker_tensor_train as t3, t3toolbox.uniform_tucker_tensor_train as ut3
np.random.seed(1)
shape, tr, ttr = (3, 5), (2, 3), (1, 2, 1)
for C in [(), (2,), (2, 3)]:
    np.random.seed(1)
    x = t3.TuckerTensorTrain.randn(shape, tr, ttr, stack_shape=C); X = np.asarray(x.to_dense())
    ux = ut3.UniformTuckerTensorTrain.from_t3(x, N=8, n=5, r=5)
    y = t3.TuckerTensorTrain.randn(shape, tr, ttr, stack_shape=C)
    uy = ut3.UniformTuckerTensorTrain.from_t3(y, N=8, n=5, r=5)
    ref = np.asarray(x.norm())
    def show(tag):
        try: got = np.asarray(ux.norm()); print('  C=%-6s after %-22s ux.norm() relerr %.1e' % (C, tag, np.linalg.norm(got - ref) / np.linalg.norm(ref)))
        except Exception as e: print('  C=%-6s after %-22s ux.norm() RAISES %s: %s' % (C, tag, type(e).__name__, e))
    show('nothing'); ux.to_dense(); show('to_dense'); ux + uy; show('add'); ux - uy; show('sub'); ux * 2.5; show('scalar mul'); ux.inner(uy); show('inner(uy)')
    print('  C=%-6s ux.inner(uy) relerr %.1e' % (C, np.linalg.norm(np.asarray(ux.inner(uy)) - np.sum((X * np.asarray(y.to_dense())).reshape(C + (-1,)), -1)) / np.linalg.norm(ref)))
