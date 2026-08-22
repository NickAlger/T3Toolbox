"""d=1: UniformTuckerTensorTrain.from_t3 on a single-mode T3."""
import numpy as np
import t3toolbox.tucker_tensor_train as t3
import t3toolbox.uniform_tucker_tensor_train as ut3
import t3toolbox.backend.tt_operations as tt_ops
np.random.seed(0)
x = t3.TuckerTensorTrain.randn((5,), (3,), (1, 1))
print('ragged d=1 ok: ncores', len(x.tt_cores), [G.shape for G in x.tt_cores], x.to_dense().shape)
sq = tt_ops.tt_squash_tails(x.tt_cores)
print('tt_squash_tails(ragged d=1) -> len', len(sq), [G.shape for G in sq])
for squash in (True, False):
    try:
        ux = ut3.UniformTuckerTensorTrain.from_t3(x, squash_tails=squash)
        print('from_t3 squash=%s OK' % squash, ux, np.allclose(ux.to_dense(), x.to_dense()))
    except Exception as e:
        print('from_t3 squash=%s FAILS: %s: %s' % (squash, type(e).__name__, e))
# the other d=1 constructors
for name, fn in (('zeros', lambda: ut3.UniformTuckerTensorTrain.zeros((5,), (3,), (1, 1))),
                 ('randn', lambda: ut3.UniformTuckerTensorTrain.randn((5,), (3,), (1, 1))),
                 ('ones', lambda: ut3.UniformTuckerTensorTrain.ones((5,)))):
    try:
        u = fn(); print(name, 'd=1 OK', u, 'to_t3 ok:', np.allclose(u.to_t3().to_dense(), u.to_dense()))
        print('   t3svd d=1:', u.t3svd()[0], '  norm', float(u.norm()), float(np.linalg.norm(u.to_dense())))
    except Exception as e:
        print(name, 'd=1 FAILS: %s: %s' % (type(e).__name__, e))
