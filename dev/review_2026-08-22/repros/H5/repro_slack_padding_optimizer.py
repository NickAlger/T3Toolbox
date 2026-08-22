"""A MINIMAL-rank but SLACK-PADDED x0 (from_t3(x, n=, r=) -- a documented option) through the uniform
optimizers: uniform_minimal passes it (ranks are minimal) but the retraction shrinks the padded dims to
max(raw ranks), so the loop-invariant masks desync from the supercores."""
import numpy as np, traceback
import t3toolbox.tucker_tensor_train as t3
import t3toolbox.uniform_tucker_tensor_train as ut3
import t3toolbox.uniform_manifold as ut3m
import t3toolbox.optimizers as opt
import t3toolbox.backend.uniform_fitting as uf
import t3toolbox.backend.optimizers as bopt
np.random.seed(0)
x_true = t3.TuckerTensorTrain.randn((6, 7, 5), (2, 3, 2), (1, 2, 2, 1))
ww = [np.random.randn(30, n) for n in (6, 7, 5)]
b = x_true.apply(ww)
x0r = t3.TuckerTensorTrain.randn((6, 7, 5), (2, 3, 2), (1, 2, 2, 1))
for (n, r) in ((3, 2), (5, 4)):
    x0 = ut3.UniformTuckerTensorTrain.from_t3(x0r, n=n, r=r)
    print('\n=== x0 padded n=%d r=%d (max real ranks 3,2); minimal? %s; uniform_minimal returns same object? %s'
          % (n, r, bool(np.all(x0.has_minimal_ranks)), uf.uniform_minimal(x0) is x0))
    fr = ut3m.UNIFORM_MANIFOLD.frame(x0)
    y = ut3m.UNIFORM_MANIFOLD.retract(ut3m.UT3Tangent.zeros(fr))
    print('  frontend retract of zero tangent: output (n, r) = (%d, %d); frame (nU, rL) = (%d, %d); x0 (n, r) = (%d, %d)'
          % (y.n, y.r, fr.nU, fr.rL, x0.n, x0.r))
    for name, fn in (('newton_cg', lambda: opt.newton_cg(ut3m.UNIFORM_MANIFOLD, 'apply', ww, b, x0, max_newton=2)),
                     ('mc_sgd', lambda: opt.mc_sgd(ut3m.UNIFORM_MANIFOLD, 'apply', ww, b, x0, np.random.default_rng(0), 10, max_iter=3)),
                     ('adam', lambda: opt.adam(ut3m.UNIFORM_MANIFOLD, 'apply', ww, b, x0, np.random.default_rng(0), 10, max_iter=3)),
                     ('backend gradient_descent', lambda: bopt.gradient_descent(uf.uniform_least_squares_problem('manifold', 'apply', x0, ww, b), (x0.data[0], x0.data[1]), n_iter=3)),
                     ('newton_cg COREWISE', lambda: opt.newton_cg(ut3m.UNIFORM_COREWISE, 'apply', ww, b, x0, max_newton=2))):
        try:
            out = fn()
            print('  %-28s OK' % name)
        except Exception as e:
            msg = str(e).replace('\n', ' ')[:230]
            print('  %-28s CRASH %s: %s' % (name, type(e).__name__, msg))
