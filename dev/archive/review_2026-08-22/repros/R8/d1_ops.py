"""R8: d=1 (a single mode) -- every uniform op vs its ragged twin."""
import numpy as np, traceback
import t3toolbox.tucker_tensor_train as t3
import t3toolbox.uniform_tucker_tensor_train as ut3
import t3toolbox.backend.tt_operations as tt_ops
np.random.seed(0)
def relerr(a, b): return float(np.linalg.norm(np.asarray(a, float) - np.asarray(b, float)) / max(np.linalg.norm(b), 1e-300))

x = t3.TuckerTensorTrain.randn((5,), (3,), (1, 1))
y = t3.TuckerTensorTrain.randn((5,), (2,), (1, 1))
print('ragged d=1 sanity: x+y', relerr((x + y).to_dense(), x.to_dense() + y.to_dense()), ' norm', abs(x.norm() - np.linalg.norm(x.to_dense())),
      ' t3svd', relerr(x.t3svd()[0].to_dense(), x.to_dense()), ' squash', relerr(t3.TuckerTensorTrain(x.data[0], tt_ops.tt_squash_tails(x.data[1])).to_dense(), x.to_dense()))
ux, uy = ut3.UniformTuckerTensorTrain.from_t3(x), ut3.UniformTuckerTensorTrain.from_t3(y)
print('uniform tt_supercore shape:', ux.tt_supercore.shape)
print('backend _tt_squash_tails_uniform on a d=1 supercore -> shape', tt_ops.tt_squash_tails(ux.tt_supercore).shape, '(expected (1,...))')
ops = {
    'to_dense':      lambda: relerr(ux.to_dense(), x.to_dense()),
    'squash_tails':  lambda: relerr(ux.squash_tails().to_dense(), x.to_dense()),
    'x + y':         lambda: relerr((ux + uy).to_dense(), (x + y).to_dense()),
    'norm()':        lambda: abs(float(ux.norm()) - x.norm()),
    'norm(False)':   lambda: abs(float(ux.norm(False)) - x.norm()),
    'inner()':       lambda: abs(float(ux.inner(uy)) - x.inner(y)),
    'inner(False)':  lambda: abs(float(ux.inner(uy, False)) - x.inner(y)),
    't3svd':         lambda: relerr(ux.t3svd()[0].to_dense(), x.to_dense()),
    'is_left_orthogonal':  lambda: bool(ux.t3svd()[0].is_left_orthogonal()),
    'is_right_orthogonal': lambda: bool(ux.is_right_orthogonal()),
    'rank_adjustment_sweep': lambda: relerr(ux.t3svd()[0].rank_adjustment_sweep().to_dense(), x.to_dense()),
    'sum':           lambda: abs(float(ux.sum()) - x.to_dense().sum()),
    'entries':       lambda: abs(float(ux.entries(np.array([2]))) - x.to_dense()[2]),
    'probe':         lambda: relerr(ux.probe([np.random.randn(5)])[0], x.to_dense()),
    'left_orth':     lambda: relerr(ux.left_orthogonalize_tt_cores().to_dense(), x.to_dense()),
    'up_orth':       lambda: relerr(ux.up_orthogonalize_tt_cores().to_dense(), x.to_dense()),
}
xs = t3.TuckerTensorTrain.randn((5,), (3,), (1, 1), stack_shape=(2,))
uxs = ut3.UniformTuckerTensorTrain.from_t3(xs)
ops['sum_stack'] = lambda: relerr(uxs.sum_stack().to_dense(), xs.to_dense().sum(axis=0))
for k, f in ops.items():
    try:
        print('%-24s -> %s' % (k, f()))
    except Exception as e:
        print('%-24s -> EXC %s: %s' % (k, type(e).__name__, str(e)[:160]))
