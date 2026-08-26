"""Independent dense oracles for the six sampling kinds and the objective. No library jet code is used:
derivatives are recovered by exact polynomial interpolation of S(X; ww + s*pp) in s."""
import math
import numpy as np

LET = 'ijklmn'


def _flatW(v):                      # (W..., N) -> (prodW, N), and the W shape
    return v.reshape(-1, v.shape[-1]), v.shape[:-1]


def S_apply(X, ww):
    d = X.ndim
    ws = [_flatW(np.asarray(w))[0] for w in ww]
    Wsh = np.asarray(ww[0]).shape[:-1]
    sub = LET[:d] + ',' + ','.join('w' + LET[i] for i in range(d)) + '->w'
    return np.einsum(sub, X, *ws).reshape(Wsh)


def S_probe(X, ww):
    d = X.ndim
    ws = [_flatW(np.asarray(w))[0] for w in ww]
    Wsh = np.asarray(ww[0]).shape[:-1]
    out = []
    for i in range(d):
        if d == 1:
            out.append(np.broadcast_to(X, Wsh + (X.shape[0],)).copy())
            continue
        sub = LET[:d] + ',' + ','.join('w' + LET[j] for j in range(d) if j != i) + '->w' + LET[i]
        out.append(np.einsum(sub, X, *[ws[j] for j in range(d) if j != i]).reshape(Wsh + (X.shape[i],)))
    return out


def onehots(index, shape):           # index (d,)+W -> list of d one-hot (W..., Ni)
    d = index.shape[0]
    Wsh = index.shape[1:]
    out = []
    for i in range(d):
        e = np.zeros(Wsh + (shape[i],))
        flat = index[i].reshape(-1)
        ef = e.reshape(-1, shape[i])
        ef[np.arange(flat.size), flat] = 1.0
        out.append(ef.reshape(Wsh + (shape[i],)))
    return out


def S_entries(X, index):
    return S_apply(X, onehots(np.asarray(index), X.shape))


def _poly_derivs(fun, deg, order):
    """fun(s) -> array; fun is a polynomial of degree <= deg in s. Returns stack over t=0..order of
    d^t/ds^t fun(0), via exact interpolation at Chebyshev nodes."""
    n = deg + 1
    nodes = np.cos(np.pi * (2 * np.arange(n) + 1) / (2 * n)) if n > 1 else np.array([0.0])
    vals = np.stack([np.asarray(fun(s)) for s in nodes])            # (n, ...)
    V = np.vander(nodes, n, increasing=True)                        # (n, n): V[k, t] = s_k^t
    coef = np.linalg.solve(V, vals.reshape(n, -1)).reshape((n,) + vals.shape[1:])
    out = []
    for t in range(order + 1):
        out.append(math.factorial(t) * coef[t] if t < n else np.zeros(vals.shape[1:]))
    return np.stack(out)


def S_apply_derivatives(X, ww, pp, order):
    d = X.ndim
    return _poly_derivs(lambda s: S_apply(X, [w + s * p for w, p in zip(ww, pp)]), d, order)


def S_entries_derivatives(X, index, pp, order):
    return S_apply_derivatives(X, onehots(np.asarray(index), X.shape), pp, order)


def S_probe_derivatives(X, ww, pp, order):
    d = X.ndim
    jets = _poly_derivs(lambda s: np.concatenate([z.reshape(z.shape[:-1] + (-1,)) for z in
                                                  S_probe(X, [w + s * p for w, p in zip(ww, pp)])], axis=-1),
                        d - 1, order)                                 # (order+1, W..., sum Ni)
    out, off = [], 0
    for i in range(d):
        out.append(jets[..., off:off + X.shape[i]])
        off += X.shape[i]
    return out


def S(kind, X, sample, order=None):
    if kind == 'apply':
        return S_apply(X, sample)
    if kind == 'entries':
        return S_entries(X, sample)
    if kind == 'probe':
        return S_probe(X, sample)
    if kind == 'apply_derivatives':
        return S_apply_derivatives(X, sample[0], sample[1], order)
    if kind == 'entries_derivatives':
        return S_entries_derivatives(X, sample[0], sample[1], order)
    if kind == 'probe_derivatives':
        return S_probe_derivatives(X, sample[0], sample[1], order)
    raise ValueError(kind)


def weighted_residual(kind, Sx, y, weight, order):
    """omega ⊙ (S(x) - y) with the documented omega[mode, order] broadcasting. weight: None, 1-D, or 2-D."""
    is_probe = kind.startswith('probe')
    has_order = kind.endswith('_derivatives')
    if is_probe:
        r = [np.asarray(a) - np.asarray(b) for a, b in zip(Sx, y)]
        if weight is None:
            return r
        w = np.asarray(weight, dtype=float)
        d = len(r)
        if kind == 'probe':                      # 1-D per-mode
            wm = w.reshape(d, 1)
        else:
            wm = w.reshape(1, -1) if w.ndim == 1 else w   # (1,order+1) or (d, order+1)
        out = []
        for i, ri in enumerate(r):
            wi = wm[i if wm.shape[0] > 1 else 0]          # (o,)
            if has_order:
                out.append(ri * wi.reshape((wi.size,) + (1,) * (ri.ndim - 1)))
            else:
                out.append(ri * wi[0])
        return out
    r = np.asarray(Sx) - np.asarray(y)
    if weight is None:
        return r
    w = np.asarray(weight, dtype=float).reshape(-1)        # order-only
    return r * w.reshape((w.size,) + (1,) * (r.ndim - 1))


def sumsq(r):
    if isinstance(r, (list, tuple)):
        return float(sum(np.sum(np.asarray(a) ** 2) for a in r))
    return float(np.sum(np.asarray(r) ** 2))


def misfit(kind, X, sample, y, weight=None, order=None):
    return 0.5 * sumsq(weighted_residual(kind, S(kind, X, sample, order), y, weight, order))
