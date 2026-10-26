"""Critic pre-check (read-only): is the zero-tensor reference an intrinsic property of the problem? Rotate the
problem by an orthogonal Q on mode 0 (samples w0 -> w0 Q^T, target T -> T x_0 Q): the problem is an exact isometry
(same objective landscape, same data), so any intrinsic scale is unchanged. The zero-tensor frame is LAPACK's
fixed completion (canonical basis), which does not rotate with the problem."""
import sys; sys.path.insert(0, __import__("os").getcwd())
import numpy as np
import t3toolbox.tucker_tensor_train as t3
import t3toolbox.backend.optimizers as bopt
import t3toolbox.backend.geometry as bgeo
import t3toolbox.backend.fitting as bfit
rng = np.random.default_rng(1); np.random.seed(1)
shape, tk, tt = (5, 6, 7), (2, 3, 2), (1, 2, 2, 1)
W = 60
ww = [rng.standard_normal((W, N)) for N in shape]
target = t3.TuckerTensorTrain.randn(shape, tk, tt)
data = np.asarray(target.apply(ww))
geom = bgeo.ManifoldGeometryOps()
zero = t3.TuckerTensorTrain.zeros(shape, tk, tt)
def g0(ww_):
    prob = bopt.least_squares_problem(geom, bfit.APPLY, ww_, data)
    g = prob.local_model(zero.data).gradient
    return float(geom.inner(g, g)) ** 0.5
base = g0(ww)
vals = []
for k in range(6):
    Q = np.linalg.qr(rng.standard_normal((shape[0], shape[0])))[0]
    vals.append(g0([ww[0] @ Q.T] + ww[1:]))          # data unchanged: <T x0 Q, w0 Q^T ...> = <T, w0 ...>
print(f'zero-tensor reference, original problem: {base:.4e}')
print('under exact isometries of the problem (mode-0 rotations):', ' '.join(f'{v:.4e}' for v in vals))
print(f'spread max/min = {max(vals + [base]) / min(vals + [base]):.2f}')
# the warm-start reference at a nonzero point IS invariant (frame rotates with the point):
x = t3.TuckerTensorTrain.randn(shape, tk, tt)
def gx(ww_, xx):
    prob = bopt.least_squares_problem(geom, bfit.APPLY, ww_, data)
    g = prob.local_model(xx.data).gradient
    return float(geom.inner(g, g)) ** 0.5
Q = np.linalg.qr(rng.standard_normal((shape[0], shape[0])))[0]
xr = t3.TuckerTensorTrain((x.data[0][0] @ Q.T,) + tuple(x.data[0][1:]), x.data[1])
print(f'nonzero start, original {gx(ww, x):.6e} vs rotated problem + rotated point {gx([ww[0] @ Q.T] + ww[1:], xr):.6e}')
