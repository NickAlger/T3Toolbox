"""O2 oracle sweep, part 1: GaussNewtonModel vs dense ground truth, over representation x geometry x
kind x order x weight x regularizer x W-layout. Every check is against oracle.py (no library jet code)."""
import itertools, sys, traceback, time
import numpy as np
sys.path.insert(0, '.')
import oracle as O
import t3toolbox.tucker_tensor_train as t3
import t3toolbox.uniform_tucker_tensor_train as ut3
import t3toolbox.manifold as t3m
import t3toolbox.uniform_manifold as ut3m
import t3toolbox.shared_geometry as sg
import t3toolbox.fitting as fitting
import t3toolbox.corewise as cw
from t3toolbox.backend.regularization import IdentityRegularizer

LAM = 0.3
TOL = 1e-8
rows, failures = [], []


def rel(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.abs(a - b).max() / (np.abs(b).max() + 1e-300))


def rel_list(a, b):
    return max(rel(x, y) for x, y in zip(a, b))


def dense_of(tangent):
    return np.asarray(tangent.to_dense())


def build_point(d, shared, seed):
    rng = np.random.default_rng(seed)
    if d == 3:
        shape, tk, tt = ((4, 4, 6), (2, 2, 2), (1, 2, 2, 1)) if shared else ((4, 5, 6), (2, 3, 2), (1, 2, 2, 1))
        sharing = (0, 0, 1)
    elif d == 2:
        shape, tk, tt = ((5, 5), (2, 2), (1, 2, 1)) if shared else ((4, 5), (2, 2), (1, 2, 1))   # minimal: n_i <= r_i r_{i+1}
        sharing = (0, 0)
    else:
        shape, tk, tt = (5,), (3,), (1, 1)
        sharing = None
    x = t3.TuckerTensorTrain.randn(shape, tk, tt)
    x = t3.TuckerTensorTrain(tuple(rng.standard_normal(c.shape) for c in x.data[0]),
                             tuple(rng.standard_normal(c.shape) for c in x.data[1]))
    if shared:
        U = list(x.data[0])
        for i in range(1, d):
            if sharing[i] == sharing[0]:
                U[i] = U[0]
        x = t3.TuckerTensorTrain(tuple(U), x.data[1])
    return x, (sharing if shared else None)


def make_sample(kind, shape, W, rng):
    d = len(shape)
    ww = [rng.standard_normal(W + (N,)) for N in shape]
    ww = [w / np.linalg.norm(w, axis=-1, keepdims=True) for w in ww]
    pp = [rng.standard_normal(W + (N,)) for N in shape]
    index = np.stack([rng.integers(0, N, size=W) for N in shape])
    if kind == 'apply':
        return ww
    if kind == 'probe':
        return ww
    if kind == 'entries':
        return index
    if kind in ('apply_derivatives', 'probe_derivatives'):
        return (ww, pp)
    return (index, pp)


def make_weight(kind, d, order, wspec):
    if wspec == 'none':
        return None
    if wspec == 'order':
        return 1.0 + 0.5 * np.arange(order + 1)
    if wspec == 'mode':
        return 0.5 + np.arange(d) / d
    if wspec == 'matrix':
        return np.outer(0.5 + np.arange(d) / d, 1.0 + 0.5 * np.arange(order + 1))
    raise ValueError(wspec)


def build_model(geom, x, kind, sample, residual, order, weight, reg):
    if kind == 'apply':
        return fitting.apply_model(geom, x, sample, residual, regularizer=reg)
    if kind == 'entries':
        return fitting.entries_model(geom, x, sample, residual, regularizer=reg)
    if kind == 'probe':
        return fitting.probe_model(geom, x, sample, residual, weight=weight, regularizer=reg)
    if kind == 'apply_derivatives':
        return fitting.apply_derivatives_model(geom, x, sample[0], sample[1], order, residual, weight=weight, regularizer=reg)
    if kind == 'entries_derivatives':
        return fitting.entries_derivatives_model(geom, x, sample[0], sample[1], order, residual, weight=weight, regularizer=reg)
    return fitting.probe_derivatives_model(geom, x, sample[0], sample[1], order, residual, weight=weight, regularizer=reg)


def lib_S(x, kind, sample, order):
    if kind == 'apply': return x.apply(sample)
    if kind == 'entries': return x.entries(sample)
    if kind == 'probe': return x.probe(sample)
    if kind == 'apply_derivatives': return x.apply_derivatives(sample[0], sample[1], order)
    if kind == 'entries_derivatives': return x.entries_derivatives(sample[0], sample[1], order)
    return x.probe_derivatives(sample[0], sample[1], order)


def reg_value(geom_kind, x_dense, cores):
    if geom_kind == 'manifold':
        return 0.5 * LAM * float(np.sum(x_dense ** 2))
    return 0.5 * LAM * float(sum(np.sum(np.asarray(c) ** 2) for c in cores[0]) + sum(np.sum(np.asarray(c) ** 2) for c in cores[1]))


def point_cores(pt):
    """(tucker_cores, tt_cores) of a ragged or uniform point, as ragged arrays (uniform -> to_t3)."""
    if isinstance(pt, ut3.UniformTuckerTensorTrain):
        pt = pt.to_t3()
    return pt.data


def tangent_ragged(v):
    return v.to_t3tangent() if isinstance(v, ut3m.UT3Tangent) else v


def run_case(d, rep, geom_name, shared, kind, order, wspec, regspec, W, seed):
    x, sharing = build_point(d, shared, seed)
    rng = np.random.default_rng(seed + 1)
    shape = x.shape
    X = x.to_dense()
    sample = make_sample(kind, shape, W, rng)
    weight = make_weight(kind, d, order or 0, wspec)
    reg = IdentityRegularizer(LAM) if regspec else None
    # data: S(A) for a different tensor A, plus noise
    A0 = t3.TuckerTensorTrain.randn(shape, x.tucker_ranks, x.tt_ranks)
    A = t3.TuckerTensorTrain(tuple(rng.standard_normal(c.shape) for c in A0.data[0]),
                             tuple(rng.standard_normal(c.shape) for c in A0.data[1]))
    y = O.S(kind, A.to_dense(), sample, order)
    Sx = lib_S(x, kind, sample, order)
    residual = [np.asarray(a) - np.asarray(b) for a, b in zip(Sx, y)] if kind.startswith('probe') else np.asarray(Sx) - y

    # the geometry + point in the requested representation
    if rep == 'ragged':
        base = t3m.MANIFOLD if geom_name == 'manifold' else t3m.COREWISE
        pt = x
    else:
        base = ut3m.UNIFORM_MANIFOLD if geom_name == 'manifold' else ut3m.UNIFORM_COREWISE
        pt = ut3.UniformTuckerTensorTrain.from_t3(x)
    geom = sg.shared(base, sharing) if shared else base

    model = build_model(geom, pt, kind, sample, residual, order, weight, reg)
    res = {}
    # 1. objective
    obj_hand = O.misfit(kind, X, sample, y, weight, order) + (reg_value(geom_name, X, x.data) if reg else 0.0)
    res['objective'] = rel(model.objective_value, obj_hand)

    # tangents at the model frame
    p = geom.randn(model.frame)
    q = geom.randn(model.frame)
    Pp = geom.project(p) if geom_name == 'manifold' or shared else p
    Pq = geom.project(q) if geom_name == 'manifold' or shared else q
    # 2. jacobian vs S_lin(dense(Πp))
    Jp_lib = model.jacobian(p)
    if kind.startswith('probe') and not isinstance(Jp_lib, (list, tuple)):   # uniform: packed (d,)+[o]+W+(N,)
        Jp_lib = np.asarray(Jp_lib)
        res['packed_pad_zero'] = max(float(np.abs(Jp_lib[i][..., shape[i]:]).max()) if shape[i] < Jp_lib.shape[-1] else 0.0
                                     for i in range(d))
        Jp_lib = [Jp_lib[i][..., :shape[i]] for i in range(d)]
    Jp_hand = O.S(kind, dense_of(Pp), sample, order)
    Jq_hand = O.S(kind, dense_of(Pq), sample, order)
    res['jacobian'] = rel_list(Jp_lib, Jp_hand) if kind.startswith('probe') else rel(Jp_lib, Jp_hand)
    # 3. gn_quadratic
    zero = [np.zeros_like(np.asarray(a)) for a in Jp_hand] if kind.startswith('probe') else np.zeros_like(Jp_hand)
    wJp = O.weighted_residual(kind, Jp_hand, zero, weight, order)
    wJq = O.weighted_residual(kind, Jq_hand, zero, weight, order)
    def pair(a, b):
        if isinstance(a, (list, tuple)):
            return float(sum(np.sum(np.asarray(u) * np.asarray(v)) for u, v in zip(a, b)))
        return float(np.sum(np.asarray(a) * np.asarray(b)))
    reg_pp = LAM * float(Pp.corewise_inner(Pp)) if reg else 0.0
    reg_pq = LAM * float(Pp.corewise_inner(Pq)) if reg else 0.0
    res['gn_quadratic'] = rel(model.gn_quadratic(p), pair(wJp, wJp) + reg_pp)
    # 4. <q, H p>
    Hp = model.gn_hessian(p)
    res['gn_hessian'] = rel(float(q.corewise_inner(Hp)), pair(wJq, wJp) + reg_pq)
    # 5. gradient adjoint identity (unregularized part) : <g, v> = <ωr, ω J v> + reg pairing
    g = model.gradient
    wr = O.weighted_residual(kind, Sx, y, weight, order)
    if reg:
        if geom_name == 'manifold':
            reg_gv = LAM * float(np.sum(X * dense_of(Pp)))
        else:
            vr = tangent_ragged(Pp)
            reg_gv = LAM * float(cw.corewise_dot(x.data, vr.variations.data))
    else:
        reg_gv = 0.0
    res['grad_adjoint'] = rel(float(g.corewise_inner(Pp)), pair(wr, wJp) + reg_gv)
    # 6. gradient vs finite difference of the hand objective along the retraction
    def f_along(t):
        pt_t = geom.retract(Pp * t)
        Xt = np.asarray(pt_t.to_dense())
        c = point_cores(pt_t)
        return O.misfit(kind, Xt, sample, y, weight, order) + (reg_value(geom_name, Xt, c) if reg else 0.0)
    h = 1e-3
    d1 = (f_along(h) - f_along(-h)) / (2 * h)
    d2 = (f_along(h / 2) - f_along(-h / 2)) / h
    fd = (4 * d2 - d1) / 3
    res['grad_fd'] = abs(fd - float(g.corewise_inner(Pp))) / (abs(fd) + 1e-300)
    res['_gnorm'] = float(g.corewise_norm())
    res['_gdense'] = dense_of(g)
    res['_obj'] = float(model.objective_value)
    res['_gnq'] = float(model.gn_quadratic(g))
    return res


def main():
    t0 = time.time()
    kinds = ['apply', 'entries', 'probe', 'apply_derivatives', 'entries_derivatives', 'probe_derivatives']
    n = 0
    pair_cache = {}
    for d in (3, 2):
        for geom_name, shared in [('manifold', False), ('corewise', False), ('manifold', True), ('corewise', True)]:
            for kind in kinds:
                orders = [None] if not kind.endswith('_derivatives') else [1, 2, 3]
                for order in orders:
                    wspecs = ['none']
                    if kind == 'probe': wspecs += ['mode']
                    if kind.endswith('_derivatives'): wspecs += ['order']
                    if kind == 'probe_derivatives': wspecs += ['matrix']
                    for wspec in wspecs:
                        for regspec in (False, True):
                            Ws = [(12,)] + ([(3, 4)] if wspec == 'none' and not regspec else [])
                            for W in Ws:
                                for rep in ('ragged', 'uniform'):
                                    key = (d, rep, geom_name, shared, kind, order, wspec, regspec, W)
                                    n += 1
                                    try:
                                        res = run_case(d, rep, geom_name, shared, kind, order, wspec, regspec, W, seed=7)
                                    except Exception as e:
                                        failures.append((key, 'EXC', f'{type(e).__name__}: {str(e)[:160]}'))
                                        continue
                                    bad = {k: v for k, v in res.items() if not k.startswith('_') and not (v < (1e-6 if k == 'grad_fd' else TOL))}
                                    if bad:
                                        failures.append((key, 'NUM', bad))
                                    pair_cache.setdefault(key[:1] + key[2:], {})[rep] = res
    # ragged == uniform per value
    for key, both in pair_cache.items():
        if len(both) == 2:
            a, b = both['ragged'], both['uniform']
            e = {'obj': rel(a['_obj'], b['_obj']), 'gnorm': rel(a['_gnorm'], b['_gnorm']),
                 'gdense': rel(a['_gdense'], b['_gdense']), 'gnq(g)': rel(a['_gnq'], b['_gnq'])}
            bad = {k: v for k, v in e.items() if not (v < 1e-8)}
            if bad:
                failures.append((('ragged-vs-uniform',) + key, 'PAIR', bad))
    print(f'{n} model cases run in {time.time()-t0:.0f}s; {len(failures)} failures')
    for f in failures:
        print(' FAIL', f)


if __name__ == '__main__':
    main()
