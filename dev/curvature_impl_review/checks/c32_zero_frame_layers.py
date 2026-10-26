"""Read-only plan check (v5, I-18a): the zero-tensor reference ||g(0)|| is a property of the FRAME chosen at the
fully degenerate zero point (any orthonormal completion is valid there). Compare ragged vs uniform completions,
and the reference under a few valid completions (random orthogonal rotations of the ragged zero frame)."""
import sys; sys.path.insert(0, __import__("os").getcwd())
import numpy as np, math
import t3toolbox.tucker_tensor_train as t3
import t3toolbox.uniform_tucker_tensor_train as ut3
import t3toolbox.corewise as cw
import t3toolbox.backend.optimizers as bopt, t3toolbox.backend.geometry as bgeo, t3toolbox.backend.fitting as bfit
import t3toolbox.backend.uniform_fitting as uf
import t3toolbox.backend.tv_operations as tv
rng = np.random.default_rng(0); np.random.seed(0)
shape, tk, tt, W = (5, 6, 7), (2, 3, 2), (1, 2, 2, 1), 50
ww = [rng.standard_normal((W, N)) for N in shape]
data = np.asarray(t3.TuckerTensorTrain.randn(shape, tk, tt).apply(ww))
zero = t3.TuckerTensorTrain.zeros(shape, tk, tt)
geom = bgeo.ManifoldGeometryOps()
prob = bopt.least_squares_problem(geom, bfit.APPLY, ww, data)
lm = prob.local_model(zero.data)
U, O, L, R = lm.frame
print('ragged zero frame U[0] =\n', np.round(U[0], 3))
g = lm.gradient; print(f'ragged ||g(0)|| = {float(geom.inner(g, g)) ** 0.5:.4e}')
uz = ut3.UniformTuckerTensorTrain.from_t3(zero)
ug = bgeo.UniformManifoldGeometryOps.from_point(uz.data)
uprob = bopt.least_squares_problem(ug, uf.uniform_sampling_kind('apply', uz.data, None), uf.pack_sample('apply', ww, uz.N), data)
ulm = uprob.local_model((uz.data[0], uz.data[1]))
print('uniform zero frame U[0] (real block) =\n', np.round(np.asarray(ulm.frame[0])[0][:tk[0], :shape[0]], 3))
gu = ulm.gradient; print(f'uniform ||g(0)|| = {float(ug.inner(gu, gu)) ** 0.5:.4e}')
# other valid completions: rotate each orthonormal factor of the ragged zero frame by a random orthogonal map
# of the ambient mode space (the tangent space at 0 moves with the completion; the objective does not)
vals = []
for trial in range(5):
    Qs = [np.linalg.qr(rng.standard_normal((N, N)))[0] for N in shape]
    Ur = tuple(u @ q for u, q in zip(U, Qs))
    frame = (Ur, O, L, R)
    z = prob.kind.transpose(lm.residual, ww, frame, prob.kind.precompute(frame, ww))
    gp = geom.project(frame, z)
    vals.append(float(geom.inner(gp, gp)) ** 0.5)
print('||g(0)|| under 5 random Tucker-basis completions:', ' '.join('%.3e' % v for v in vals))
