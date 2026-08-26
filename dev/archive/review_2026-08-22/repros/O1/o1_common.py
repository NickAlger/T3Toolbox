"""Shared oracles + result table for the O1 sweep."""
import itertools, math, traceback, sys
import numpy as np
import t3toolbox.tucker_tensor_train as t3
import t3toolbox.frame_variations_format as bvf
import t3toolbox.manifold as t3m
import t3toolbox.uniform_tucker_tensor_train as ut3
import t3toolbox.uniform_frame_variations_format as ubv
import t3toolbox.uniform_manifold as ut3m
import t3toolbox.corewise as cw

RESULTS = []   # (op, struct_name, repr, C, W, K, sharing, status, err, note)

def record(op, sname, rep, C, W, K, sh, status, err=float('nan'), note=''):
    RESULTS.append((op, sname, rep, C, W, K, sh, status, err, note))

def relerr(a, b):
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    den = max(np.linalg.norm(b.reshape(-1)), 1e-300)
    return float(np.linalg.norm((a - b).reshape(-1)) / den)

def check(op, sname, rep, C, W, K, sh, fn, tol=1e-9):
    """fn() -> (err, note) or err. Records PASS/FAIL/EXC."""
    try:
        out = fn()
        if isinstance(out, tuple):
            err, note = out
        else:
            err, note = out, ''
        status = 'PASS' if err <= tol else 'FAIL'
        record(op, sname, rep, C, W, K, sh, status, err, note)
        return status == 'PASS'
    except Exception as e:
        record(op, sname, rep, C, W, K, sh, 'EXC', float('nan'), '%s: %s' % (type(e).__name__, str(e).splitlines()[0][:160]))
        return False

def dump(path):
    with open(path, 'w') as f:
        f.write('| op | struct | repr | C | W | K | sharing | status | max relerr | note |\n|---|---|---|---|---|---|---|---|---|---|\n')
        for r in RESULTS:
            f.write('| %s | %s | %s | %s | %s | %s | %s | %s | %.2e | %s |\n' % r)
    n = len(RESULTS); p = sum(r[7] == 'PASS' for r in RESULTS)
    fails = [r for r in RESULTS if r[7] != 'PASS']
    print('TOTAL %d  PASS %d  non-PASS %d' % (n, p, n - p))
    for r in fails:
        print('  %s | %s | %s | C=%s W=%s K=%s sh=%s | %s | %.2e | %s' % (r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8], r[9]))

# --------------------------------------------------------------------- structures
STRUCTS = {
    'd1':      ((5,),         (3,),         (1, 1)),
    'd2':      ((3, 5),       (2, 3),       (1, 2, 1)),
    'd3':      ((3, 5, 4),    (2, 3, 2),    (1, 2, 3, 1)),
    'd4':      ((3, 5, 4, 6), (2, 3, 2, 4), (1, 2, 3, 2, 1)),
    'rank1':   ((3, 5, 4),    (1, 1, 1),    (1, 1, 1, 1)),
    'nonmin':  ((3, 5, 4),    (2, 3, 2),    (1, 4, 3, 1)),   # r1=4 > n0*r0=2 -> non-minimal TT rank
    'sh2':     ((4, 4, 5),    (2, 2, 3),    (1, 2, 3, 1)),   # sharing (0,0,1)
    'shall':   ((4, 4, 4),    (2, 2, 2),    (1, 2, 3, 1)),   # sharing (0,0,0)
    'sh4':     ((5, 5, 4, 4), (2, 2, 3, 3), (1, 2, 3, 2, 1)),  # sharing ('a','a','b','b')
}
SHARING = {'sh2': (0, 0, 1), 'shall': (0, 0, 0), 'sh4': ('a', 'a', 'b', 'b')}
CS = [(), (2,), (2, 3)]
WS = [(), (3,), (2, 2)]
KS = [(), (2,)]

# --------------------------------------------------------------------- dense oracles
LET = 'ijklmnop'

def flatC(X, d):
    X = np.asarray(X); C = X.shape[:X.ndim - d]
    return X.reshape((-1,) + X.shape[X.ndim - d:]), C

def flatW(ww):
    W = np.asarray(ww[0]).shape[:-1]
    return [np.asarray(w).reshape(-1, np.asarray(w).shape[-1]) for w in ww], W

def dense_apply(X, ww, d):
    Xf, C = flatC(X, d); wf, W = flatW(ww)
    s = 'c' + LET[:d] + ',' + ','.join('w' + LET[i] for i in range(d)) + '->wc'
    return np.einsum(s, Xf, *wf).reshape(W + C)

def dense_probe(X, ww, d):
    Xf, C = flatC(X, d); wf, W = flatW(ww)
    out = []
    if d == 1:
        return [np.broadcast_to(Xf.reshape((1,) * len(W) + C + (Xf.shape[1],)), W + C + (Xf.shape[1],)).copy()]
    for m in range(d):
        ops = [wf[i] for i in range(d) if i != m]
        s = 'c' + LET[:d] + ''.join(',w' + LET[i] for i in range(d) if i != m) + '->wc' + LET[m]
        out.append(np.einsum(s, Xf, *ops).reshape(W + C + (Xf.shape[1 + m],)))
    return out

def onehots(index, shape):
    index = np.asarray(index); d = index.shape[0]; W = index.shape[1:]
    return [np.eye(shape[i])[index[i]] for i in range(d)]   # W + (Ni,)

def dense_entries(X, index, d, shape):
    return dense_apply(X, onehots(index, shape), d)

def dense_apply_jets(X, ww, pp, d, order):
    """t-th symmetric derivative of apply(ww + s pp) at s=0: t! * sum_{|S|=t} apply(pp on S, ww else)."""
    out = []
    for t in range(order + 1):
        acc = np.zeros_like(dense_apply(X, ww, d))
        for S in itertools.combinations(range(d), t):
            vv = [pp[i] if i in S else ww[i] for i in range(d)]
            acc = acc + dense_apply(X, vv, d)
        out.append(math.factorial(t) * acc)
    return np.stack(out, axis=0)

def dense_probe_jets(X, ww, pp, d, order):
    res = [[] for _ in range(d)]
    for t in range(order + 1):
        accs = [0] * d
        for m in range(d):
            others = [i for i in range(d) if i != m]
            for S in itertools.combinations(others, t):
                vv = [pp[i] if i in S else ww[i] for i in range(d)]
                accs[m] = accs[m] + dense_probe(X, vv, d)[m]
        for m in range(d):
            z0 = dense_probe(X, ww, d)[m]
            res[m].append(math.factorial(t) * (accs[m] if not isinstance(accs[m], int) else np.zeros_like(z0)))
    return [np.stack(r, axis=0) for r in res]

def rand_ww(shape, W, seed):
    rng = np.random.RandomState(seed)
    return [rng.randn(*(W + (N,))) for N in shape]

def rand_index(shape, W, seed):
    rng = np.random.RandomState(seed)
    return np.stack([rng.randint(0, N, size=W) for N in shape], axis=0)

def tdot(a, b):
    """sum of elementwise product with broadcasting over leading axes (a may have extra leading axes)."""
    a = np.asarray(a); b = np.asarray(b)
    return float(np.sum(a * b.reshape((1,) * (a.ndim - b.ndim) + b.shape)))

def var_dot(v_data, u_data):
    """<variations, variations> summing everything, broadcasting extra leading stacks of v over u."""
    tot = 0.0
    for fam_v, fam_u in zip(v_data, u_data):
        for a, b in zip(fam_v, fam_u):
            tot += tdot(a, b)
    return tot
