import numpy as np, sys
sys.path.insert(0, '.')
import oracle as O
import t3toolbox.tucker_tensor_train as t3
np.random.seed(1)
for shape, tk, tt in [((4, 5, 6), (2, 3, 2), (1, 2, 2, 1)), ((4, 5), (2, 3), (1, 2, 1)), ((5,), (3,), (1, 1))]:
    d = len(shape)
    x = t3.TuckerTensorTrain.randn(shape, tk, tt)
    X = x.to_dense()
    for W in [(7,), (3, 4)]:
        ww = [np.random.randn(*W, N) for N in shape]
        pp = [np.random.randn(*W, N) for N in shape]
        index = np.stack([np.random.randint(0, N, size=W) for N in shape])
        for order in (1, 2, 3):
            checks = {
                'apply': (x.apply(ww), O.S_apply(X, ww)),
                'entries': (x.entries(index), O.S_entries(X, index)),
                'probe': (x.probe(ww), O.S_probe(X, ww)),
                'apply_d': (x.apply_derivatives(ww, pp, order), O.S_apply_derivatives(X, ww, pp, order)),
                'entries_d': (x.entries_derivatives(index, pp, order), O.S_entries_derivatives(X, index, pp, order)),
                'probe_d': (x.probe_derivatives(ww, pp, order), O.S_probe_derivatives(X, ww, pp, order)),
            }
            for k, (a, b) in checks.items():
                if isinstance(a, (list, tuple)):
                    err = max(np.abs(np.asarray(ai) - bi).max() / (np.abs(bi).max() + 1e-300) for ai, bi in zip(a, b))
                else:
                    err = np.abs(np.asarray(a) - b).max() / (np.abs(b).max() + 1e-300)
                print(f'd={d} W={W} order={order} {k:10s} relerr={err:.1e}', 'OK' if err < 1e-9 else 'MISMATCH')
