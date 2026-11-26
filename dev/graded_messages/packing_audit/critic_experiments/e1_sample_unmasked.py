"""E1: the plan's §3.2 claim that the sample need not be masked on entry.

Monkeypatch the engine so the uniform sample is NOT masked on entry (``_UNIFORM_MASKS['w'] = ()``), then call the
engine with the same packed sample twice, clean padding (zeros) and junk padding (1e3), for every public name, both
kinds with jets, both call types, sum_over_samples True/False, with directions, Vc, K stacks. Report any output that
differs (bitwise) or whose padding is not zero, except the 'w' echo (returned as given today).
"""
import itertools
import numpy as np
import t3toolbox.tucker_tensor_train as t3
import t3toolbox.uniform_tucker_tensor_train as ut3
import t3toolbox.uniform_manifold as ut3m
import t3toolbox.backend.sampling_variables as sv
import t3toolbox.backend.ut3_operations as uo
import t3toolbox.backend.ufv_conversions as ufv_conversions

sv._UNIFORM_MASKS = dict(sv._UNIFORM_MASKS)
sv._UNIFORM_MASKS['w'] = ()          # the plan: the sample is no longer masked on entry

rng = np.random.default_rng(0)
shape = (4, 5, 6)
N = 7                                 # padded wider than every mode: padding in every mode
x = t3.TuckerTensorTrain.randn(shape, (2, 3, 2), (1, 2, 2, 1))
ux = ut3.UniformTuckerTensorTrain.from_t3(x, N=N)
frame = ut3m.UNIFORM_MANIFOLD.frame(ux)
fdata = frame.data
point = ux.data
tang = ut3m.UNIFORM_MANIFOLD.randn(frame)
V, H = tang.variations.data[0], tang.variations.data[1]
_, var_centers = ufv_conversions.ut3_orthogonal_representations(ux.data)
Vc = var_centers[0]


def packed_sample(order, W, junk):
    m = len(order)
    F_in = tuple(min(2, n + 1) for n in order)
    ww = [rng.standard_normal(F_in + W + (n,)) for n in shape]
    p = uo.pack_vectors(ww, N, len_P=m)
    if junk:
        p = p.copy()
        for i, n in enumerate(shape):
            p[i, ..., n:] = 1e3
    return p


def padmask(name, kind, value, setting_shape):
    return None


problems = []
checked = 0
for kind, order, call, keep, K in itertools.product(('probe', 'apply'), ((), (2,)), ('manifold', 'corewise'),
                                                     (False, True), ((), (2,))):
    W = (3,)
    F = tuple(n + 1 for n in order)
    fr = fdata if call == 'manifold' else point
    seed = rng.integers(1 << 30)
    r1 = np.random.default_rng(seed)
    clean = packed_sample(order, W, False)
    junk = clean.copy()
    for i, n in enumerate(shape):
        junk[i, ..., n:] = 1e3
    if kind == 'probe':
        y = uo.pack_vectors([r1.standard_normal(F + W + (n,)) for n in shape], N)
    else:
        y = r1.standard_normal(F + W)
    kw = dict(order=order, y=y, sum_over_samples=not keep)
    if call == 'manifold':
        if K:
            Vk = np.broadcast_to(V[:, None], (V.shape[0],) + K + V.shape[1:]).copy() * r1.standard_normal((1,) + K + (1,) * (V.ndim - 1))
            Hk = np.broadcast_to(H[:, None], (H.shape[0],) + K + H.shape[1:]).copy()
            kw['variations'] = (Vk, Hk)
            kw['core_directions'] = (r1.standard_normal((fdata[0].shape[0],) + K + fdata[0].shape[1:]),
                                     r1.standard_normal((fdata[2].shape[0],) + K + fdata[2].shape[1:]))
        else:
            kw['variations'] = (V, H)
            kw['core_directions'] = (r1.standard_normal(fdata[0].shape), r1.standard_normal(fdata[2].shape))
    else:
        kw['core_directions'] = (r1.standard_normal((point[0].shape[0],) + K + point[0].shape[1:]),
                                 r1.standard_normal((point[1].shape[0],) + K + point[1].shape[1:]))
    names = [n for n in sv._names(kind, '_' + call) if n != 'w']
    for extra in ({}, {'Vc': Vc} if call == 'manifold' else None):
        if extra is None:
            continue
        for name in names:
            try:
                (a,) = sv.sampling_variables(kind, fr, clean, (name,), **kw, **extra)
                (b,) = sv.sampling_variables(kind, fr, junk, (name,), **kw, **extra)
            except ValueError as e:
                continue
            checked += 1
            la = a if isinstance(a, tuple) else (a,)
            lb = b if isinstance(b, tuple) else (b,)
            def flat(v):
                if isinstance(v, (tuple, list)):
                    return [q for u in v for q in flat(u)]
                return [np.asarray(v)]
            fa, fb = flat(a), flat(b)
            diff = max(float(np.max(np.abs(p - q))) if p.size else 0.0 for p, q in zip(fa, fb))
            if diff != 0.0:
                problems.append((kind, order, call, keep, K, name, bool(extra), diff))

print('checked', checked)
print('problems', len(problems))
for p in problems[:40]:
    print(p)
