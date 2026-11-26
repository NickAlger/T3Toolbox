"""E4: dense_apply at the one-hot line against the subset-sum definition. E5: a dense forward for the finite
difference of test_tucker_tensor_train's corewise derivative transposes."""
import itertools, math
import numpy as np
import t3toolbox.tucker_tensor_train as t3
import t3toolbox.backend.apply as apply
import t3toolbox.backend.probing as probing
import t3toolbox.backend.t3_conversions as t3c
from t3toolbox.backend import jets
import sys; sys.path.insert(0, '/home/nick/repos/T3Toolbox/tests'); import _derivative_data as dd
rng = np.random.default_rng(0)
shape = (4, 5, 3); d = 3
T = rng.standard_normal(shape)
index = np.stack([rng.integers(0, N, size=(2, 2)) for N in shape])
pp = [rng.standard_normal((2, 2, N)) for N in shape]
for n in (1, 2, 4):
    got = apply.dense_apply(T, jets.affine_jet(jets.index_to_one_hot(index, shape), pp), (n,))
    ref = np.zeros((n + 1, 2, 2))
    for t in range(min(n, d) + 1):
        for S in itertools.combinations(range(d), t):
            for a, b in itertools.product(range(2), range(2)):
                A = T
                for i in reversed(range(d)):              # contract the last mode first
                    A = np.tensordot(A, pp[i][a, b], axes=([i], [0])) if i in S else np.take(A, index[i, a, b], axis=i)
                ref[t, a, b] += A
    print('E4 n=%d max err %.1e' % (n, np.abs(got - ref).max()), 'slot>d zero:', bool(np.all(got[d + 1:] == 0)))
# E5
np.random.seed(0)
x = t3.TuckerTensorTrain.randn((6, 7, 8), (3, 4, 3), (1, 3, 2, 3))
W = (2,); ORDER = 2
ww = [np.random.randn(*(W + (N,))) for N in x.shape]; pp = [np.random.randn(*(W + (N,))) for N in x.shape]
index = np.stack([np.random.randint(0, N, size=W) for N in x.shape], axis=0)
oh = jets.index_to_one_hot(index, x.shape)
for kind, transpose, forward in (
    ('apply', lambda rr: dd.corewise_transpose(x, 'apply', rr, ww, pp, ORDER, sum_over_probes=True),
              lambda data: jets.taylor_to_derivatives('apply', (ORDER,), apply.dense_apply(t3c.t3_to_dense(data), jets.affine_jet(ww, pp), (ORDER,)))),
    ('entries', lambda rr: dd.corewise_transpose(x, 'entries', rr, index, pp, ORDER, sum_over_probes=True),
                lambda data: jets.taylor_to_derivatives('apply', (ORDER,), apply.dense_apply(t3c.t3_to_dense(data), jets.affine_jet(oh, pp), (ORDER,)))),
    ('probe', lambda rr: dd.corewise_transpose(x, 'probe', rr, ww, pp, ORDER, sum_over_probes=True),
              lambda data: jets.taylor_to_derivatives('probe', (ORDER,), probing.dense_probe(t3c.t3_to_dense(data), jets.affine_jet(ww, pp), (ORDER,))))):
    f0 = forward(x.data)
    r = [np.random.randn(*np.shape(a)) for a in f0] if kind == 'probe' else np.random.randn(*np.shape(f0))
    gB, gG = transpose(r)
    dB = [np.random.randn(*B.shape) for B in x.tucker_cores]; dG = [np.random.randn(*G.shape) for G in x.tt_cores]
    inner = sum(np.sum(np.asarray(a) * b) for a, b in zip(list(gB) + list(gG), dB + dG))
    eps = 1e-6
    dot = (lambda data: sum(float(np.sum(a * b)) for a, b in zip(r, forward(data)))) if kind == 'probe' else (lambda data: float(np.sum(r * forward(data))))
    plus = ([B + eps * db for B, db in zip(x.tucker_cores, dB)], [G + eps * dg for G, dg in zip(x.tt_cores, dG)])
    minus = ([B - eps * db for B, db in zip(x.tucker_cores, dB)], [G - eps * dg for G, dg in zip(x.tt_cores, dG)])
    fd = (dot(plus) - dot(minus)) / (2 * eps)
    print('E5', kind, 'rel %.1e' % (abs(inner - fd) / abs(fd)))
