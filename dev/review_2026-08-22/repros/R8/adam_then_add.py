"""R8: a realistic path to the squash/+ garbage bug -- the uniform corewise optimizer (adam on
UNIFORM_COREWISE) writes the raw gradient into the supercores, and the raw gradient is NONZERO in the
padded boundary-bond slots (mode 0 left leg / mode d-1 right leg).  After the fit, `x_fit + y`,
`x_fit - y`, `x_fit.squash_tails()` and (stacked) `sum_stack()` sum that padding into the real slot."""
import numpy as np
import t3toolbox.tucker_tensor_train as t3
import t3toolbox.uniform_tucker_tensor_train as ut3
import t3toolbox.uniform_manifold as ut3m
import t3toolbox.optimizers as topt
np.random.seed(0)
def relerr(a, b): return float(np.linalg.norm(np.asarray(a, float) - np.asarray(b, float)) / max(np.linalg.norm(np.asarray(b, float)), 1e-300))

shape, tk, tt = (4, 5, 3), (2, 3, 2), (1, 2, 3, 1)
x_true = t3.TuckerTensorTrain.randn(shape, tk, tt)
ww = [np.random.randn(60, Ni) for Ni in shape]
ww = [w / np.linalg.norm(w, axis=-1, keepdims=True) for w in ww]
data = x_true.probe(ww)
x0 = ut3.UniformTuckerTensorTrain.from_t3(t3.TuckerTensorTrain.randn(shape, tk, tt) * 0.1)
x_fit, stats = topt.adam(ut3m.UNIFORM_COREWISE, 'probe', ww, data, x0, np.random.default_rng(1), batch=20, lr=1e-2, max_iter=50)
print('fit object:', x_fit)
tkm, ttm = x_fit.masks.data
G0, Gf = x_fit.tt_supercore[0], x_fit.tt_supercore[-1]
print('x0 structure:', x0, '| x_fit tt_ranks', x_fit.tt_ranks.tolist(), '(the optimizer re-padded x0)')
print('x_fit padded boundary-bond slots max |.| : %.3g   (nonzero -> the corewise gradient is not clean-padded there)' % max(np.abs(G0[~ttm[0]]).max(), np.abs(Gf[..., ~ttm[-1]]).max()))
print('x_fit padded TUCKER slots max |.|        : %.3g' % np.abs(x_fit.tucker_supercore[~tkm]).max())

clean = x_fit.apply_masks()                     # the same tensor, padding zeroed
y = ut3.UniformTuckerTensorTrain.from_t3(t3.TuckerTensorTrain.randn(shape, (2, 2, 2), (1, 2, 2, 1)))
print('to_dense(x_fit) == to_dense(clean)       :', relerr(x_fit.to_dense(), clean.to_dense()))
print('norm(x_fit)      vs norm(clean)          :', relerr(x_fit.norm(), clean.norm()))
print('x_fit.squash_tails() vs clean  rel err   : %.3g' % relerr(x_fit.squash_tails().to_dense(), clean.to_dense()))
print('x_fit + y  vs clean + y        rel err   : %.3g' % relerr((x_fit + y).to_dense(), (clean + y).to_dense()))
print('x_fit - y  vs clean - y        rel err   : %.3g' % relerr((x_fit - y).to_dense(), (clean - y).to_dense()))
print('x_fit.to_t3() + y (ragged)     rel err   : %.3g' % relerr((x_fit.to_t3() + y.to_t3()).to_dense(), (clean + y).to_dense()))
st = ut3.UniformTuckerTensorTrain.stack((x_fit, x_fit))
print('stack(x_fit,x_fit).sum_stack() rel err   : %.3g' % relerr(st.sum_stack().to_dense(), 2 * clean.to_dense()))
