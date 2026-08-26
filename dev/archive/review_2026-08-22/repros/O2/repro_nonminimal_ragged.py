"""Ragged MANIFOLD optimizers on a NON-minimal-rank x0 (d=2, shape (4,5), tucker (2,3), tt (1,2,1):
n_1 = 3 > r_1*r_2 = 2). The uniform path reduces transparently; the ragged path crashes."""
import numpy as np, traceback
import t3toolbox.tucker_tensor_train as t3
import t3toolbox.manifold as t3m
import t3toolbox.optimizers as topt
np.random.seed(0)
shape, tk, tt = (4, 5), (2, 3), (1, 2, 1)
x0 = t3.TuckerTensorTrain.randn(shape, tk, tt)
print('x0 ranks', x0.tucker_ranks, x0.tt_ranks, 'minimal:', x0.minimal_ranks, 'has_minimal_ranks:', x0.has_minimal_ranks)
A = t3.TuckerTensorTrain.randn(shape, (2, 2), (1, 2, 1))
ww = [np.random.randn(12, N) for N in shape]
ww = [w / np.linalg.norm(w, axis=1, keepdims=True) for w in ww]
b = A.apply(ww)
for name, fn, kw in [('newton_cg', topt.newton_cg, dict(max_newton=3)),
                     ('gradient_descent', topt.gradient_descent, dict(n_iter=3)),
                     ('mc_sgd', topt.mc_sgd, dict(max_iter=3, check_every=3)),
                     ('adam', topt.adam, dict(max_iter=3))]:
    args = (t3m.MANIFOLD, 'apply', ww, b, x0)
    if name in ('mc_sgd', 'adam'):
        args = args + (np.random.default_rng(0), 6)
    try:
        x, st = fn(*args, **kw)
        print(name, 'OK ranks ->', x.tucker_ranks, x.tt_ranks)
    except Exception as e:
        print(name, 'RAISED', type(e).__name__, str(e)[:120])
        if name == 'newton_cg':
            traceback.print_exc()
# the same x0 reduced to minimal ranks first works:
x0m = x0.rank_adjustment_sweep('right_to_left') if hasattr(x0, 'rank_adjustment_sweep') else x0
print('reduced x0 ranks', x0m.tucker_ranks, x0m.tt_ranks, 'same tensor:', bool(np.allclose(x0m.to_dense(), x0.to_dense())))
x, st = topt.newton_cg(t3m.MANIFOLD, 'apply', ww, b, x0m, max_newton=3)
print('newton_cg from minimal x0 OK, losses', [f'{v:.3e}' for v in st['losses']])
# d=3 analogue: tucker (2,3,2) with tt (1,2,1,1)  -> n_1 = 3 > r_1 r_2 = 2
x3 = t3.TuckerTensorTrain.randn((4, 5, 6), (2, 3, 2), (1, 2, 1, 1))
print('d=3 x0 has_minimal_ranks:', x3.has_minimal_ranks)
ww3 = [np.random.randn(12, N) for N in (4, 5, 6)]
b3 = t3.TuckerTensorTrain.randn((4, 5, 6), (2, 2, 2), (1, 2, 1, 1)).apply(ww3)
try:
    x, st = topt.newton_cg(t3m.MANIFOLD, 'apply', ww3, b3, x3, max_newton=3)
    print('d=3 newton_cg OK')
except Exception as e:
    print('d=3 newton_cg RAISED', type(e).__name__, str(e)[:120])
