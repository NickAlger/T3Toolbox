"""Old derivative kernels vs the engine at the jet, apply kind: time and einsum counts across sizes.

usage: python bench_apply_jets.py [count|time] [config names...]
"""
import sys, time, math, functools
import numpy as np
import t3toolbox.tucker_tensor_train as t3
import t3toolbox.backend.sampling_derivatives as sd
import t3toolbox.backend.apply as ap
from t3toolbox.backend.fv_conversions import t3_orthogonal_representations
from t3toolbox.backend.jets import affine_jet
from t3toolbox.fitting import taylor_to_derivatives

CONFIGS = {   # name: (d, N, r, W, order)
    'example':  (4, 12, 5, 60, 4),
    'W600':     (4, 12, 5, 600, 4),
    'W6000':    (4, 12, 5, 6000, 4),
    'r20':      (4, 12, 20, 60, 4),
    'r20W2000': (4, 30, 20, 2000, 4),
    'r40W2000': (4, 50, 40, 2000, 4),
    'big':      (6, 100, 30, 4000, 4),
    'o1W2000':  (4, 30, 20, 2000, 1),
    'o8W2000':  (4, 30, 20, 2000, 8),
}


def setup(d, N, r, W, order, seed=0):
    np.random.seed(seed)
    ranks_tucker = (min(r, N),) * d
    ranks_tt = (1,) + (r,) * (d - 1) + (1,)
    x = t3.TuckerTensorTrain.randn((N,) * d, ranks_tucker, ranks_tt)
    frame, centers = t3_orthogonal_representations(x.data)
    var = tuple(tuple(np.random.randn(*c.shape) for c in part) for part in centers)
    ww = [np.random.randn(W, N) for _ in range(d)]
    pp = [np.random.randn(W, N) for _ in range(d)]
    c = np.random.randn(order + 1, W)
    return x, frame, var, ww, pp, c


def calls(d, N, r, W, order):
    x, frame, var, ww, pp, c = setup(d, N, r, W, order)
    jet = affine_jet(ww, pp)
    o = (order,)
    fact = np.array([math.factorial(t) for t in range(order + 1)]).reshape((-1,) + (1,))
    old = {
        'point_fwd':   lambda: sd.t3_apply_derivatives(ww, pp, x.data, order),
        'tangent_fwd': lambda: sd.tv_apply_derivatives(ww, pp, var, frame, order),
        'transpose':   lambda: sd.tv_apply_derivatives_transpose(c, ww, pp, frame, order, sum_over_probes=True),
    }
    new = {   # the example's path: the engine at the jet, then t! (forward) / t! on the residual (transpose)
        'point_fwd':   lambda: fact * ap.t3_apply(x.data, jet, o),
        'tangent_fwd': lambda: fact * ap.tv_apply(frame, var, jet, o),
        'transpose':   lambda: ap.tv_apply_transpose(frame, jet, fact * c, o),
    }
    return old, new


def check(old, new):
    for k in old:
        a, b = old[k](), new[k]()
        if k == 'transpose':
            err = max(float(np.max(abs(u - v))) / (float(np.max(abs(u))) + 1e-300)
                      for u, v in zip(a[0] + a[1], b[0] + b[1]))
        else:
            err = float(np.max(abs(np.asarray(a) - np.asarray(b)))) / float(np.max(abs(np.asarray(a))))
        print('   check %-12s rel err %.1e' % (k, err))


def count(fn):
    n = {'einsum': 0, 'einsum_path': 0, 'tensordot': 0, 'matmul': 0}
    orig = {k: getattr(np, k) for k in ('einsum', 'einsum_path', 'tensordot')}
    def wrap(name):
        f = orig[name]
        @functools.wraps(f)
        def g(*a, **k):
            n[name] += 1
            return f(*a, **k)
        return g
    for k in orig:
        setattr(np, k, wrap(k))
    try:
        fn()
    finally:
        for k, f in orig.items():
            setattr(np, k, f)
    return n


def best(fn, budget=2.0, min_rep=3):
    fn()
    ts, t0 = [], time.perf_counter()
    while len(ts) < min_rep or time.perf_counter() - t0 < budget:
        a = time.perf_counter(); fn(); ts.append(time.perf_counter() - a)
        if len(ts) > 200: break
    return min(ts)


if __name__ == '__main__':
    mode = sys.argv[1]
    names = sys.argv[2:] or list(CONFIGS)
    for name in names:
        cfg = CONFIGS[name]
        old, new = calls(*cfg)
        print('%s  (d, N, r, W, order) = %s' % (name, cfg))
        if mode == 'count':
            check(old, new)
            for k in old:
                print('   %-12s old %s   new %s' % (k, count(old[k]), count(new[k])))
        else:
            for k in old:
                to, tn = best(old[k]), best(new[k])
                print('   %-12s old %9.2f ms   new %9.2f ms   new/old %.2f' % (k, 1e3 * to, 1e3 * tn, tn / to))
        sys.stdout.flush()
