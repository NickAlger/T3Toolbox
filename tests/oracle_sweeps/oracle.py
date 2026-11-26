"""Independent dense oracles for the six sampling kinds and the objective. No library jet code is used:
derivatives are recovered by exact polynomial interpolation of S(X; ww + s*pp) in s. At the end, the full form of
the data-metric arc (M2): the data and the weight of the type of z, in Taylor coordinates (``misfit_full``)."""
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


# ------------------------------------------------------------------------------------------------------------------
# The full form (the data-metric arc, M2): every data-space object (the data y, the weight omega, the residual) has
# the type of z, each axis 1 or z's size, read by position; data in TAYLOR coordinates. Additive: the old forms above
# stay for the sweeps until the old kinds go. Pinned in tests/test_dense_oracle.py.
# ------------------------------------------------------------------------------------------------------------------

def expand(a, shape):
    """A data-space array (each axis 1 or ``shape``'s size, the same rank) written out to ``shape`` by ``np.repeat``
    along its size-1 axes: no broadcasting. Any other shape raises."""
    a = np.asarray(a, dtype=float)
    if a.ndim != len(shape):
        raise ValueError('rank %d, expected the rank of %s' % (a.ndim, tuple(shape)))
    for axis, (s, n) in enumerate(zip(a.shape, shape)):
        if s != n:
            if s != 1:
                raise ValueError('axis %d has size %d, expected 1 or %d' % (axis, s, n))
            a = np.repeat(a, n, axis=axis)
    return a


def _modes(kind, value, d):
    """The arrays of a data-space object: probe exactly d (a list, a tuple, or along the first axis); else one."""
    if kind != 'probe':
        return [value]
    items = [value[i] for i in range(len(value))]
    if len(items) != d:
        raise ValueError('probe: exactly d = %d arrays, got %d' % (d, len(items)))
    return items


def S_taylor(kind, X, sample, order=()):
    """The output of ``kind`` ('probe', 'apply', 'entries') in Taylor coordinates, of the type of z (probe a list).
    ``order = ()``: ``S`` at ``sample``. ``order = (n,)``: ``sample = (main, pp)``, the line ``main + s pp``; slot ``t``
    is the coefficient of ``s^t``, the ``t``-th derivative of ``_poly_derivs`` divided by ``t!`` (entries: the apply
    kind at the one-hot vectors of ``main``)."""
    if order == ():
        return S(kind, X, sample)
    (n,) = order
    derivatives = S(kind + '_derivatives', X, sample, n)
    factorials = np.array([math.factorial(t) for t in range(n + 1)], dtype=float)
    per_slot = lambda D: D / factorials.reshape((n + 1,) + (1,) * (D.ndim - 1))
    return [per_slot(D) for D in derivatives] if kind == 'probe' else per_slot(derivatives)


def weighted_residual_full(kind, Sx, y, weight):
    """``omega * (S(x) - y)`` entry by entry: ``y`` and ``omega`` of the type of z, written out to z's shape by
    ``expand`` (no broadcasting). ``weight=None`` is ones. Probe: a list."""
    zs = _modes(kind, Sx, len(Sx) if kind == 'probe' else None)
    ys = _modes(kind, y, len(zs))
    ws = [None] * len(zs) if weight is None else _modes(kind, weight, len(zs))
    out = []
    for z, y_, w in zip(zs, ys, ws):
        z = np.asarray(z, dtype=float)
        r = z - expand(y_, z.shape)
        out.append(r if w is None else expand(w, z.shape) * r)
    return out if kind == 'probe' else out[0]


def misfit_full(kind, X, sample, y, weight=None, order=()):
    """``1/2 sum (omega * (S(x) - y))**2`` over every entry of z (modes, slots, samples, the edge), Taylor
    coordinates, the full form. One dense tensor ``X`` (no stack: loop over the elements of a stack)."""
    return 0.5 * sumsq(weighted_residual_full(kind, S_taylor(kind, X, sample, order), y, weight))


def weighted_full(kind, like, value, weight):
    """``omega * value`` entry by entry at ``like``'s shapes (``like`` of the type of z, e.g. ``S_taylor``'s output):
    ``value`` and ``omega`` of the type of z, both written out by ``expand`` (no broadcasting). ``weight=None`` is
    ones. Probe: a list. The data's half of the block sums (M3), as ``weighted_residual_full`` is the residual's."""
    zs = _modes(kind, like, len(like) if kind == 'probe' else None)
    vs = _modes(kind, value, len(zs))
    ws = [None] * len(zs) if weight is None else _modes(kind, weight, len(zs))
    out = []
    for z, v, w in zip(zs, vs, ws):
        shape = np.shape(z)
        v = expand(v, shape)
        out.append(v if w is None else expand(w, shape) * v)
    return out if kind == 'probe' else out[0]


# ------------------------------------------------------------------------------------------------------------------
# The two weights (K1): a block weight, one number per block (probe (d,)+F, apply F, entries ()), and a full weight of
# the type of z, multiplying. The full weight is written out by ``expand`` (no broadcasting); the block row is placed
# by an explicit loop over the blocks. Additive: the full form above stays for the callers that still pass one weight.
# ------------------------------------------------------------------------------------------------------------------

def _block_factor(kind, block_weight, i, shape, m):
    """The block weight of mode ``i`` (probe) written out to ``shape`` by a loop over the blocks: the slot ``k`` of
    the ``m`` leading jet axes holds ``block_weight[i][k]`` (probe), ``block_weight[k]`` (apply) or the one number
    (entries), at every entry after them. Ones for ``None``."""
    out = np.ones(shape)
    if block_weight is None:
        return out
    b = np.asarray(block_weight, dtype=float)
    row = b[i] if kind == 'probe' else b
    if row.shape != tuple(shape[:m]):
        raise ValueError('the block row has shape %s, expected the jet box %s' % (row.shape, tuple(shape[:m])))
    for slot in np.ndindex(*shape[:m]):
        out[slot] = row[slot]
    return out


def weighted_residual_two(kind, Sx, y, block_weight, full_weight):
    """``b * f * (S(x) - y)`` entry by entry: ``b`` the block weight placed on the leading jet axes by a loop over the
    blocks, ``f`` the full weight and ``y`` written out to z's shape by ``expand``. ``None`` is ones. Probe: a list.
    The jet count ``m`` is read off the block weight (probe ``ndim - 1``, apply ``ndim``); with no block weight it
    is not needed."""
    zs = _modes(kind, Sx, len(Sx) if kind == 'probe' else None)
    ys = _modes(kind, y, len(zs))
    fs = [None] * len(zs) if full_weight is None else _modes(kind, full_weight, len(zs))
    m = 0 if block_weight is None else np.ndim(block_weight) - (1 if kind == 'probe' else 0)
    out = []
    for i, (z, y_, f) in enumerate(zip(zs, ys, fs)):
        z = np.asarray(z, dtype=float)
        r = (z - expand(y_, z.shape)) * _block_factor(kind, block_weight, i, z.shape, m)
        out.append(r if f is None else expand(f, z.shape) * r)
    return out if kind == 'probe' else out[0]


def weighted_two(kind, like, value, block_weight, full_weight):
    """``b * f * value`` entry by entry at ``like``'s shapes, as ``weighted_residual_two`` weighs the residual: the
    data's half of the block sums with two weights."""
    zs = _modes(kind, like, len(like) if kind == 'probe' else None)
    vs = _modes(kind, value, len(zs))
    fs = [None] * len(zs) if full_weight is None else _modes(kind, full_weight, len(zs))
    m = 0 if block_weight is None else np.ndim(block_weight) - (1 if kind == 'probe' else 0)
    out = []
    for i, (z, v, f) in enumerate(zip(zs, vs, fs)):
        shape = np.shape(z)
        v = expand(v, shape) * _block_factor(kind, block_weight, i, shape, m)
        out.append(v if f is None else expand(f, shape) * v)
    return out if kind == 'probe' else out[0]


def block_sumsq_full(kind, values, m):
    """The sum of squares of each block of ``values`` (of the type of z, e.g. ``weighted_full`` or
    ``weighted_residual_full``), by loops over the blocks: the block shape ``(d,) + F`` (probe), ``F`` (apply), ``()``
    (entries), with ``m`` jet axes leading each array; everything after them (the samples, the edge) is summed."""
    arrays = [np.asarray(v, dtype=float) for v in (values if kind == 'probe' else [values])]
    F = arrays[0].shape[:m]
    out = np.zeros((len(arrays),) + F)
    for i, a in enumerate(arrays):
        for slot in np.ndindex(*F):
            out[(i,) + slot] = float(np.sum(a[slot] ** 2))
    return out if kind == 'probe' else out[0]
