# E7 (round 2): masking by selection instead of by product. (a) On clean inputs, are the engine's outputs the same BYTES
# (tobytes, so signed zeros count)? (b) With -0.0 in a clean-looking padding (y = -y0 packed: padding -0.0), where do the
# two differ? (c) With inf in y's / a held c's padding, selection gives finite outputs.
import itertools
import numpy as np
import t3toolbox.tucker_tensor_train as t3
import t3toolbox.uniform_tucker_tensor_train as ut3
import t3toolbox.uniform_manifold as ut3m
import t3toolbox.backend.sampling_variables as sv
import t3toolbox.backend.ut3_operations as uo
from t3toolbox.backend.sampling_variables import HeldVariables

_product = sv._masked

def _selected(x, spec, masks):
    if not spec:
        return x
    n_core = len(spec); n_stack = np.ndim(x) - 1 - n_core
    total = None
    for j, (family, part) in enumerate(spec):
        mask = masks[family]
        mask = mask[:-1] if part == 'lo' else (mask[1:] if part == 'hi' else mask)
        stack = mask.shape[1:-1]
        mask = mask.reshape(mask.shape[:1] + (1,) * (n_stack - len(stack)) + stack + (1,) * j + mask.shape[-1:]
                            + (1,) * (n_core - 1 - j)).astype(bool)
        total = mask if total is None else (total & mask)
    return np.where(total, x, np.zeros((), dtype=np.asarray(x).dtype))

rng = np.random.default_rng(0)
shape, N = (4, 5, 6), 7
x = t3.TuckerTensorTrain.randn(shape, (2, 3, 2), (1, 2, 2, 1))
ux = ut3.UniformTuckerTensorTrain.from_t3(x, N=N)
fr = ut3m.UNIFORM_MANIFOLD.frame(ux).data
tang = ut3m.UNIFORM_MANIFOLD.randn(ut3m.UNIFORM_MANIFOLD.frame(ux))
V, H = tang.variations.data[0], tang.variations.data[1]

def flat(v):
    return [q for u in v for q in flat(u)] if isinstance(v, (tuple, list)) else [np.asarray(v)]

def run(kind, order, y, name, held=None):
    F = tuple(n + 1 for n in order)
    ww = uo.pack_vectors([np.random.default_rng(1).standard_normal((min(2, F[0]) if F else 1,)[:len(order)] + (3, n)) if order else
                          np.random.default_rng(1).standard_normal((3, n)) for n in shape], N, len_P=len(order))
    out = {}
    for how, fn in (('product', _product), ('selection', _selected)):
        sv._masked = fn
        try:
            (v,) = sv.sampling_variables(kind, fr, ww, (name,), order=order, y=y, variations=(V, H), held=held)
        finally:
            sv._masked = _product
        out[how] = flat(v)
    return out

diff_bytes = diff_value = checked = 0
for kind, order in itertools.product(('probe', 'apply'), ((), (2,))):
    F = tuple(n + 1 for n in order)
    r2 = np.random.default_rng(2)
    y = uo.pack_vectors([r2.standard_normal(F + (3, n)) for n in shape], N) if kind == 'probe' else r2.standard_normal(F + (3,))
    for name in [n for n in sv._names(kind, '_manifold') if n not in ('w',)]:
        try:
            o = run(kind, order, y, name)
        except ValueError:
            continue
        checked += 1
        if any(a.tobytes() != b.tobytes() for a, b in zip(o['product'], o['selection'])):
            diff_bytes += 1
        if any(not np.array_equal(a, b, equal_nan=True) for a, b in zip(o['product'], o['selection'])):
            diff_value += 1
print('(a) clean inputs: %d names checked; differ in bytes: %d; differ in value: %d' % (checked, diff_bytes, diff_value))

y0 = uo.pack_vectors([rng.standard_normal((3, n)) for n in shape], N)
yneg = -y0                                                       # padding is -0.0
o = run('probe', (), yneg, 'r')
pad = [np.signbit(a[0, :, 4:]).any() for a in o['product'][:1]], [np.signbit(a[0, :, 4:]).any() for a in o['selection'][:1]]
print('(b) y = -y0: r padding has a -0.0 under product:', pad[0], ' under selection:', pad[1],
      '; real entries bitwise equal:', all(np.array_equal(a, b) for a, b in zip(o['product'], o['selection'])))
yinf = y0.copy(); yinf[0, :, 4:] = np.inf
o = run('probe', (), yinf, 'f')
print('(c) inf in y padding: f product', float(o['product'][0]), ' selection', float(o['selection'][0]))
c = y0.copy(); c[0, :, 4:] = np.inf
o = run('probe', (), None, 'JtJp', held=HeldVariables(Jp=c))
print('    inf in a held c: JtJp finite under product', all(np.isfinite(a).all() for a in o['product']),
      ' under selection', all(np.isfinite(a).all() for a in o['selection']))
for kind, order in itertools.product(('probe', 'apply'), ((), (2,))):
    F = tuple(n + 1 for n in order)
    r2 = np.random.default_rng(2)
    y = uo.pack_vectors([r2.standard_normal(F + (3, n)) for n in shape], N) if kind == 'probe' else r2.standard_normal(F + (3,))
    for name in [n for n in sv._names(kind, '_manifold') if n not in ('w',)]:
        try:
            o = run(kind, order, y, name)
        except ValueError:
            continue
        for a, b in zip(o['product'], o['selection']):
            if a.tobytes() != b.tobytes():
                where = np.argwhere(np.signbit(a) != np.signbit(b))
                print('   bytes differ:', kind, order, name, 'entries with a different sign bit:', len(where),
                      'all zero there:', bool(np.all(a[tuple(where.T)] == 0)))
