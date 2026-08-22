"""R7: regularizer lambda scaling by batch/n with a custom draw of non-nominal size; 'batch ... ignored if draw given'."""
import numpy as np
import t3toolbox.tucker_tensor_train as t3
import t3toolbox.manifold as t3m
import t3toolbox.optimizers as topt
import t3toolbox.backend.optimizers as bopt
import t3toolbox.backend.geometry as bgeo
import t3toolbox.backend.fitting as bfit
import t3toolbox.corewise as cw

np.random.seed(0)
shape, tr, rr = (4, 5, 6), (2, 2, 2), (1, 2, 2, 1)
A = t3.TuckerTensorTrain.randn(shape, tr, rr)
n = 100
ww = [np.random.randn(n, N) for N in shape]
ww = [w / np.linalg.norm(w, axis=1, keepdims=True) for w in ww]
b = A.apply(ww)
x0 = t3.TuckerTensorTrain.randn(shape, tr, rr)
lam = 1.0
P = bopt.least_squares_problem(bgeo.ManifoldGeometryOps(), bfit.APPLY, ww, b, regularizer=bopt.IdentityRegularizer(lam))

draw_size = 5
def my_draw(rng):
    idx = rng.choice(n, size=draw_size, replace=False)
    return [w[idx] for w in ww], b[idx]

for batch in (5, 50, 100):
    sp = bopt._minibatch_step_problem(P, batch)
    print("batch=%3d custom draw size=%d: regularizer factor used = %.3f   (draw-consistent factor = %.3f)"
          % (batch, draw_size, sp.regularizer.factor, draw_size / n))

# user-visible: same custom draw, same seed, different `batch` -> different iterates (batch is NOT ignored)
outs = []
for batch in (5, 50):
    rng = np.random.default_rng(1)
    x, st = topt.mc_sgd(t3m.MANIFOLD, 'apply', ww, b, x0, rng, batch, draw=my_draw,
                        regularizer=topt.IdentityRegularizer(lam), max_iter=10, check_every=5)
    outs.append(x.to_dense())
print("mc_sgd with draw=my_draw: result differs between batch=5 and batch=50 (same rng seed): rel diff = %.3e"
      % (np.linalg.norm(outs[0] - outs[1]) / np.linalg.norm(outs[0])))
# the unregularized case really does ignore batch
outs = []
for batch in (5, 50):
    rng = np.random.default_rng(1)
    x, st = topt.mc_sgd(t3m.MANIFOLD, 'apply', ww, b, x0, rng, batch, draw=my_draw, max_iter=10, check_every=5)
    outs.append(x.to_dense())
print("unregularized: rel diff = %.3e (batch ignored, as documented)" % (np.linalg.norm(outs[0] - outs[1]) / np.linalg.norm(outs[0])))
