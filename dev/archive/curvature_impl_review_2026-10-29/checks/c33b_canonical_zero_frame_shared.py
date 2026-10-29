"""Read-only plan check (v5, I-18a): the canonical zero frame on a SHARED manifold geometry, ragged vs uniform:
pack the ragged shared zero frame (t3frame_to_ut3frame, no SVD) and evaluate the uniform shared geometry there."""
import sys; sys.path.insert(0, __import__("os").getcwd())
import numpy as np
import t3toolbox.tucker_tensor_train as t3, t3toolbox.uniform_tucker_tensor_train as ut3
import t3toolbox.backend.optimizers as bopt, t3toolbox.backend.geometry as bgeo, t3toolbox.backend.fitting as bfit
import t3toolbox.backend.uniform_fitting as uf, t3toolbox.backend.ufv_conversions as ufc
rng = np.random.default_rng(3); np.random.seed(3)
shape, tk, tt, sharing, order = (6, 6, 5), (3, 3, 2), (1, 3, 2, 1), (0, 0, 1), 2
W = 30
ww = [rng.standard_normal((W, N)) for N in shape]; pp = [rng.standard_normal((W, N)) for N in shape]
kind = bfit.probe_derivatives_kind(order, None)
tgt = t3.TuckerTensorTrain.randn(shape, tk, tt)
data = [np.asarray(z) for z in kind.point_forward(tgt.data, (ww, pp))]
zero = t3.TuckerTensorTrain.zeros(shape, tk, tt)
geom = bgeo.ManifoldGeometryOps().with_sharing(sharing, shape)
prob = bopt.least_squares_problem(geom, kind, (ww, pp), data)
lm = prob.local_model(zero.data)
g = lm.gradient; ref_r = float(geom.inner(g, g)) ** 0.5
U = lm.frame[0]
print(f"shared ragged zero frame: tied U0 == U1? {np.array_equal(U[0], U[1])}; ref {ref_r:.6e}")
uz = ut3.UniformTuckerTensorTrain.from_t3(zero)
ug = bgeo.UniformManifoldGeometryOps.from_point(uz.data, sharing)
ukind = uf.uniform_derivatives_kind('probe_derivatives', uz.data, order, None)
ps = uf.pack_sample('probe_derivatives', (ww, pp), uz.N); pd = uf.pack_data('probe_derivatives', data, uz.N)
ufr = ufc.t3frame_to_ut3frame(lm.frame, N=uz.data[0].shape[-1])
sweep = ukind.precompute(ufr, ps)
resid = [-z for z in pd] if isinstance(pd, (list, tuple)) else -pd
aux = ug.precompute(ufr)
gu = ug.project(ufr, ukind.transpose(resid, ps, ufr, sweep), aux=aux)
ref_u = float(ug.inner(gu, gu)) ** 0.5
own = bopt.least_squares_problem(ug, ukind, ps, pd).local_model((uz.data[0], uz.data[1])).gradient
print(f"uniform shared at the packed canonical frame: ref {ref_u:.6e} (rel diff {abs(ref_u - ref_r) / ref_r:.1e}); "
      f"at its own frame: {float(ug.inner(own, own)) ** 0.5:.6e}")
