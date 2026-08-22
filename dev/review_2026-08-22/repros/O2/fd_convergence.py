import numpy as np, sys
sys.path.insert(0, '.')
import sweep_models as SM, oracle as O
import t3toolbox.manifold as t3m, t3toolbox.fitting as fitting
d, rep, geom_name, shared, kind, order, wspec, regspec, W = (2, 'ragged', 'manifold', False, 'probe', None, 'none', False, (3, 4))
x, sharing = SM.build_point(d, shared, 7)
rng = np.random.default_rng(8)
sample = SM.make_sample(kind, x.shape, W, rng)
A = SM.t3.TuckerTensorTrain.randn(x.shape, x.tucker_ranks, x.tt_ranks)
y = O.S(kind, A.to_dense(), sample, order)
residual = [np.asarray(a) - np.asarray(b) for a, b in zip(x.probe(sample), y)]
model = fitting.probe_model(t3m.MANIFOLD, x, sample, residual)
g = model.gradient
p = t3m.MANIFOLD.project(t3m.MANIFOLD.randn(model.frame))
gv = float(g.corewise_inner(p))
def f(t): return O.misfit(kind, np.asarray(t3m.MANIFOLD.retract(p * t).to_dense()), sample, y)
for h in [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]:
    fd = (f(h) - f(-h)) / (2 * h)
    print(f'h={h:.0e}  central FD={fd:.12g}  <g,v>={gv:.12g}  rel={(fd-gv)/abs(gv):.2e}')
