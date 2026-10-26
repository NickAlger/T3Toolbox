"""Critic pre-check (read-only) for v5's g0_reference='zero' default: the gradient norm at the zero tensor of the
level's ranks, per geometry. On COREWISE the map is multilinear in the cores, so for d >= 2 its Jacobian at
all-zero cores vanishes and the zero reference is 0 -- check, and see what newton_cg's fallback does with it."""
import sys; sys.path.insert(0, __import__("os").getcwd())
import numpy as np
import t3toolbox.tucker_tensor_train as t3
import t3toolbox.uniform_tucker_tensor_train as ut3
import t3toolbox.corewise as cw
import t3toolbox.backend.optimizers as bopt
import t3toolbox.backend.geometry as bgeo
import t3toolbox.backend.fitting as bfit
import t3toolbox.backend.uniform_fitting as uf
rng = np.random.default_rng(0); np.random.seed(0)
shape, tk, tt = (5, 6, 7), (2, 3, 2), (1, 2, 2, 1)
W = 50
ww = [rng.standard_normal((W, N)) for N in shape]
target = t3.TuckerTensorTrain.randn(shape, tk, tt)
data = np.asarray(target.apply(ww))
zero = t3.TuckerTensorTrain.zeros(shape, tk, tt)
for name, geom in (('ragged MANIFOLD', bgeo.ManifoldGeometryOps()), ('ragged COREWISE', bgeo.CorewiseGeometryOps())):
    prob = bopt.least_squares_problem(geom, bfit.APPLY, ww, data)
    g = prob.local_model(zero.data).gradient
    print(f'{name:18s} ||g(zero tensor)|| = {float(geom.inner(g, g)) ** 0.5:.3e}   (||A^T y|| scale {np.linalg.norm(np.einsum("w,wa,wb,wc->abc", data, *ww)):.3e})')
uz = ut3.UniformTuckerTensorTrain.from_t3(zero)
for name, cls in (('uniform MANIFOLD', bgeo.UniformManifoldGeometryOps), ('uniform COREWISE', bgeo.UniformCorewiseGeometryOps)):
    geom = cls.from_point(uz.data)
    kind = uf.uniform_sampling_kind('apply', uz.data, None)
    prob = bopt.least_squares_problem(geom, kind, uf.pack_sample('apply', ww, uz.N), data)
    g = prob.local_model((uz.data[0], uz.data[1])).gradient
    print(f'{name:18s} ||g(zero tensor)|| = {float(geom.inner(g, g)) ** 0.5:.3e}')
