"""Backend ProbeKind accepts a per-mode weight whose length != d (frontend rejects it)."""
import numpy as np, t3toolbox as t3t
import t3toolbox.backend.fitting as bfit, t3toolbox.backend.optimizers as bopt, t3toolbox.backend.geometry as bgeo
import t3toolbox.fitting as ffit, t3toolbox.manifold as t3m
np.random.seed(0)
shape = (5, 6, 7)
x = t3t.TuckerTensorTrain.randn(shape, (2, 2, 2), (1, 2, 2, 1))
ww = [np.random.randn(4, N) for N in shape]
zz = x.probe(ww)
x0 = t3t.TuckerTensorTrain.randn(shape, (2, 2, 2), (1, 2, 2, 1))
for w in (np.array([1., 2., 3., 4., 5.]), np.array([1., 2.])):
    print('--- weight of length', len(w), '(d = 3)')
    kind = bfit.ProbeKind(weight=w)
    prob = bopt.least_squares_problem(bgeo.ManifoldGeometryOps(), kind, ww, zz)
    try:
        f = float(prob.objective(x0.data))
        g = prob.local_model(x0.data).gradient
        print('backend: objective =', f, '; gradient ok (no error)')
        f3 = float(bopt.least_squares_problem(bgeo.ManifoldGeometryOps(), bfit.ProbeKind(weight=w[:3]), ww, zz).objective(x0.data))
        print('backend: same objective as weight[:3]?', np.isclose(f, f3))
    except Exception as e:
        print('backend:', type(e).__name__, ':', str(e)[:100])
    try:
        ffit.probe_model(t3m.MANIFOLD, x0, ww, [a - b for a, b in zip(x0.probe(ww), zz)], weight=w)
        print('frontend: accepted')
    except Exception as e:
        print('frontend:', type(e).__name__, ':', str(e)[:90])
