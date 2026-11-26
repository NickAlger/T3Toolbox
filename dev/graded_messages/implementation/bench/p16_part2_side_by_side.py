"""Side by side: the pre-P16 library (git archive 0acf7868) against P16 part 2, on the same problems and seeds.

Run twice, in separate processes (both libraries are the package `t3toolbox`):
    PYTHONPATH=<old_lib> python side_by_side.py old <out.pkl>
    PYTHONPATH=<repo>    python side_by_side.py new <out.pkl>
then   python side_by_side.py compare <old.pkl> <new.pkl>

Every case builds the same data from the same seeds; the old run uses Problem / least_squares_problem / the old kinds,
the new one the oracle, the batch and the geometry. Compared: the history (objective, gnorm, step fields), stop_reason,
step counts, and the final point (through its dense tensor, so the representation's gauge does not matter).
"""
import pickle
import sys

import numpy as np

MODE = sys.argv[1]


def dense(x):
    import t3toolbox.tucker_tensor_train as t3
    if isinstance(x[0], (tuple, list)):
        return t3.TuckerTensorTrain(*x).to_dense()
    return None


def udense(x_sc, ux):
    import dataclasses as dc
    names = [f.name for f in dc.fields(ux)]
    return dc.replace(ux, **{names[0]: np.asarray(x_sc[0]), names[1]: np.asarray(x_sc[1])}).to_dense()


def record(x, stats, dense_x):
    keep = ('objective', 'gnorm', 'g0norm', 'cg_iters', 'alpha', 'slope', 'pHp', 'delta_f', 'rho', 'tr_radius',
            'tr_accepted', 'stop_reason', 'hessian', 'g0_reference', 'pKp', 'switch_count', 'switch_fired',
            'misfit', 'regularization', 'ls_steps')
    hist = [{k: row.get(k) for k in keep} for row in stats.get('history', [])]
    return dict(dense=dense_x, losses=[float(v) for v in stats['losses']], history=hist,
                stop_reason=stats.get('stop_reason'), newton_steps=stats.get('newton_steps'),
                newton_accepted=stats.get('newton_accepted'), n_iter=stats.get('n_iter'),
                switched_at=stats.get('switched_at'), cone=stats.get('cone_movements'),
                start=stats.get('start_movements'))


def data(kind, seed=0, W=60, shape=(5, 6, 5), target=((3, 3, 3), (1, 3, 3, 1)), order=None, noise=0.01):
    """The same arrays in both runs: ww (and pp), the target's derivative data (order n) or plain data, noisy."""
    import t3toolbox.tucker_tensor_train as t3
    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    T = t3.TuckerTensorTrain.randn(shape, *target)
    ww = [rng.standard_normal((W, n)) / np.sqrt(n) for n in shape]
    pp = [rng.standard_normal((W, n)) / np.sqrt(n) for n in shape]
    index = np.stack([rng.integers(0, n, W) for n in shape])
    Td = T.to_dense()
    if kind == 'apply':
        y = np.einsum('abc,wa,wb,wc->w', Td, *ww)
    elif kind == 'entries':
        y = Td[tuple(index)]
    elif kind == 'probe':
        y = [np.einsum('abc,wb,wc->wa', Td, ww[1], ww[2]), np.einsum('abc,wa,wc->wb', Td, ww[0], ww[2]),
             np.einsum('abc,wa,wb->wc', Td, ww[0], ww[1])]
    elif kind == 'probe_derivatives':      # derivatives along the line, order n: z_t = d^t/ds^t probe(w + s p)
        import math
        ys = []
        for i in range(3):
            o = [j for j in range(3) if j != i]
            zs = []
            for t in range(order + 1):
                # the coefficient of s^t of contract(T, (w+sp) on modes o): two modes, so t <= 2
                if t == 0:
                    c = np.einsum('abc,w%s,w%s->w%s' % ('abc'[o[0]], 'abc'[o[1]], 'abc'[i]), Td, ww[o[0]], ww[o[1]])
                elif t == 1:
                    c = (np.einsum('abc,w%s,w%s->w%s' % ('abc'[o[0]], 'abc'[o[1]], 'abc'[i]), Td, pp[o[0]], ww[o[1]])
                         + np.einsum('abc,w%s,w%s->w%s' % ('abc'[o[0]], 'abc'[o[1]], 'abc'[i]), Td, ww[o[0]], pp[o[1]]))
                elif t == 2:
                    c = np.einsum('abc,w%s,w%s->w%s' % ('abc'[o[0]], 'abc'[o[1]], 'abc'[i]), Td, pp[o[0]], pp[o[1]])
                else:
                    c = np.zeros((W, shape[i]))
                zs.append(math.factorial(t) * c)
            ys.append(np.stack(zs))
        y = ys
    else:
        raise ValueError(kind)
    nrng = np.random.default_rng(seed + 100)
    if isinstance(y, list):
        y = [a + noise * nrng.standard_normal(a.shape) for a in y]
    else:
        y = y + noise * nrng.standard_normal(y.shape)
    return dict(ww=ww, pp=pp, index=index, y=y, shape=shape)


def start(seed, shape=(5, 6, 5), ranks=((2, 3, 2), (1, 2, 2, 1))):
    import t3toolbox.tucker_tensor_train as t3
    np.random.seed(1000 + seed)
    return t3.TuckerTensorTrain.randn(shape, *ranks)


# ---- the fit, in each library -------------------------------------------------------------------------------------
def fit_old(kind, D, geometry, layer, reg, x0, order=None, weight=None):
    import t3toolbox.backend.optimizers as bopt
    import t3toolbox.backend.fitting as bfit
    import t3toolbox.backend.geometry as bgeo
    import t3toolbox.backend.uniform_fitting as uf
    import t3toolbox.uniform_tucker_tensor_train as ut3
    sample = {'apply': D['ww'], 'probe': D['ww'], 'entries': D['index'],
              'probe_derivatives': (D['ww'], D['pp'])}[kind]
    if layer == 'ragged':
        if kind == 'probe_derivatives':
            k = bfit.ProbeDerivativesKind(order=order, weight=weight)
        elif kind == 'probe' and weight is not None:
            k = bfit.ProbeKind(weight)
        else:
            k = {'apply': bfit.APPLY, 'entries': bfit.ENTRIES, 'probe': bfit.PROBE}[kind]
        g = bgeo.ManifoldGeometryOps() if geometry == 'manifold' else bgeo.CorewiseGeometryOps()
        return bopt.least_squares_problem(g, k, sample, D['y'], reg), x0.data, None
    ux = ut3.UniformTuckerTensorTrain.from_t3(x0)
    p = uf.uniform_least_squares_problem(geometry, kind, ux, sample, D['y'], order, weight, reg)
    return p, (ux.data[0], ux.data[1]), ux


def fit_new(kind, D, geometry, layer, reg, x0, order=None, weight=None):
    import t3toolbox.backend.geometry as bgeo
    import t3toolbox.backend.uniform_fitting as uf
    import t3toolbox.uniform_tucker_tensor_train as ut3
    from t3toolbox.backend.fitting import LeastSquaresOracle, LeastSquaresBatch
    from t3toolbox.backend.jets import affine_jet, prepare_derivative_data
    if kind == 'probe_derivatives':
        oracle = LeastSquaresOracle('probe', (order,))
        y, w, _ = prepare_derivative_data('probe', (order,), D['y'], weight)
        batch = LeastSquaresBatch(affine_jet(D['ww'], D['pp']), y, w)
        k = 'probe'
    else:
        oracle = LeastSquaresOracle(kind, ())
        batch = LeastSquaresBatch(D['index'] if kind == 'entries' else D['ww'], D['y'], weight)
        k = kind
    if layer == 'ragged':
        g = bgeo.ManifoldGeometryOps() if geometry == 'manifold' else bgeo.CorewiseGeometryOps()
        return (g, oracle, batch, reg), x0.data, None
    ux = ut3.UniformTuckerTensorTrain.from_t3(x0)
    cls = bgeo.UniformManifoldGeometryOps if geometry == 'manifold' else bgeo.UniformCorewiseGeometryOps
    g = cls.from_point(ux.data)
    return (g, oracle, uf.pack_batch(k, oracle.order, batch, ux.N), reg), (ux.data[0], ux.data[1]), ux


def run(name, kind, geometry, layer, method, reg_strength=None, order=None, weight=None, seed=0, **kw):
    import t3toolbox.backend.optimizers as bopt
    from t3toolbox.backend.regularization import IdentityRegularizer
    D = data(kind, seed=seed, order=order)
    x0 = start(seed)
    reg = IdentityRegularizer(reg_strength) if reg_strength is not None else None
    if 'padded_from' in kw:                      # a zero-padded start from a lower-rank old point
        old = start(seed + 7, ranks=((2, 2, 2), (1, 2, 2, 1)))
        x0 = old.resize(D['shape'], (3, 3, 3), (1, 3, 3, 1))
        kw = dict(kw, padded_from=old.data if layer == 'ragged' else None)
    fit = fit_old if MODE == 'old' else fit_new
    P, xc, ux = fit(kind, D, geometry, layer, reg, x0, order, weight)
    if layer == 'uniform' and 'padded_from' in kw:
        import t3toolbox.uniform_tucker_tensor_train as ut3
        old_u = ut3.UniformTuckerTensorTrain.from_t3(old, n=ux.n, r=ux.r)
        kw['padded_from'] = old_u.data
    if method in ('mc_sgd', 'adam'):
        rng = np.random.default_rng(seed + 5)
        batch_size = kw.pop('batch_size')
        custom = kw.pop('custom_draw', False)
        if custom:                          # step for step: the same rng calls as the flat draw, by hand
            if MODE == 'old':
                n = P.kind.n_measurements(P.sample)
                draw = lambda r: P.kind.take(P.sample, P.data, r.choice(n, size=min(batch_size, n), replace=False))
            else:
                g, oracle, batch, _ = P
                n = oracle.n_measurements(batch)
                draw = lambda r: oracle.take(batch, r.choice(n, size=min(batch_size, n), replace=False))
            kw['draw'] = draw
        if MODE == 'old':
            x, st = getattr(bopt, method)(P, xc, rng, batch_size, **kw)
        else:
            g, oracle, batch, reg_ = P
            x, st = getattr(bopt, method)(g, oracle, batch, xc, rng, batch_size, regularizer=reg_, **kw)
    else:
        if MODE == 'old':
            x, st = getattr(bopt, method)(P, xc, **kw)
        else:
            g, oracle, batch, reg_ = P
            x, st = getattr(bopt, method)(g, oracle, batch, xc, regularizer=reg_, **kw)
    dx = dense(x) if layer == 'ragged' else udense(x, ux)
    return name, record(x, st, dx)


CASES = [
    ('gn_ls_probe_manifold_zero', 'probe', 'manifold', 'ragged', 'newton_cg', dict(max_newton=12)),
    ('gn_ls_apply_manifold_warm', 'apply', 'manifold', 'ragged', 'newton_cg', dict(max_newton=12, g0_reference='warm_start')),
    ('gn_ls_entries_manifold', 'entries', 'manifold', 'ragged', 'newton_cg', dict(max_newton=12)),
    ('gn_ls_probe_corewise', 'probe', 'corewise', 'ragged', 'newton_cg', dict(max_newton=10)),
    ('gn_tr_apply_manifold', 'apply', 'manifold', 'ragged', 'newton_cg', dict(max_newton=12, step_control='trust_region')),
    ('exact_tr_probe_manifold', 'probe', 'manifold', 'ragged', 'newton_cg', dict(max_newton=12, hessian='exact')),
    ('exact_ls_apply_manifold', 'apply', 'manifold', 'ragged', 'newton_cg', dict(max_newton=10, hessian='exact', step_control='line_search')),
    ('exact_tr_entries_corewise', 'entries', 'corewise', 'ragged', 'newton_cg', dict(max_newton=10, hessian='exact')),
    ('switch_probe_manifold', 'probe', 'manifold', 'ragged', 'newton_cg', dict(max_newton=15, hessian='switch')),
    ('switch_all_apply_manifold', 'apply', 'manifold', 'ragged', 'newton_cg', dict(max_newton=12, hessian='switch', switch_track='all')),
    ('cone_probe_manifold', 'probe', 'manifold', 'ragged', 'newton_cg', dict(max_newton=8, g0_reference='cone')),
    ('padded_probe_manifold', 'probe', 'manifold', 'ragged', 'newton_cg', dict(max_newton=8, padded_from=True)),
    ('padded_exact_apply_manifold', 'apply', 'manifold', 'ragged', 'newton_cg', dict(max_newton=8, padded_from=True, hessian='exact')),
    ('reg_gn_probe_manifold', 'probe', 'manifold', 'ragged', 'newton_cg', dict(max_newton=10), 0.05),
    ('reg_exact_apply_corewise', 'apply', 'corewise', 'ragged', 'newton_cg', dict(max_newton=10, hessian='exact'), 0.05),
    ('reg_switch_probe_manifold_tr', 'probe', 'manifold', 'ragged', 'newton_cg', dict(max_newton=12, hessian='switch'), 0.05),
    ('weighted_probe_manifold', 'probe', 'manifold', 'ragged', 'newton_cg', dict(max_newton=10), None, None, np.array([1.0, 2.0, 0.5])),
    ('deriv2_probe_manifold', 'probe_derivatives', 'manifold', 'ragged', 'newton_cg', dict(max_newton=10), None, 2),
    ('deriv1_exact_probe_manifold', 'probe_derivatives', 'manifold', 'ragged', 'newton_cg', dict(max_newton=10, hessian='exact'), None, 1),
    ('gd_probe_manifold', 'probe', 'manifold', 'ragged', 'gradient_descent', dict(n_iter=15)),
    ('gd_apply_corewise_reg', 'apply', 'corewise', 'ragged', 'gradient_descent', dict(n_iter=15), 0.05),
    ('gd_cone_apply_manifold', 'apply', 'manifold', 'ragged', 'gradient_descent', dict(n_iter=6, g0_reference='cone')),
    ('mcsgd_flat_probe_manifold', 'probe', 'manifold', 'ragged', 'mc_sgd', dict(batch_size=20, max_iter=40, check_every=10)),
    ('mcsgd_flat_reg_apply_manifold', 'apply', 'manifold', 'ragged', 'mc_sgd', dict(batch_size=20, max_iter=40, check_every=10), 0.05),
    ('mcsgd_custom_entries_manifold', 'entries', 'manifold', 'ragged', 'mc_sgd', dict(batch_size=25, max_iter=30, check_every=10, custom_draw=True)),
    ('adam_flat_apply_corewise', 'apply', 'corewise', 'ragged', 'adam', dict(batch_size=20, max_iter=100)),
    ('adam_reg_probe_corewise', 'probe', 'corewise', 'ragged', 'adam', dict(batch_size=20, max_iter=100), 0.05),
    ('u_gn_probe_manifold', 'probe', 'manifold', 'uniform', 'newton_cg', dict(max_newton=10)),
    ('u_exact_apply_manifold', 'apply', 'manifold', 'uniform', 'newton_cg', dict(max_newton=8, hessian='exact')),
    ('u_switch_probe_manifold_reg', 'probe', 'manifold', 'uniform', 'newton_cg', dict(max_newton=10, hessian='switch'), 0.05),
    ('u_exact_entries_corewise', 'entries', 'corewise', 'uniform', 'newton_cg', dict(max_newton=8, hessian='exact')),
    ('u_cone_apply_manifold', 'apply', 'manifold', 'uniform', 'newton_cg', dict(max_newton=6, g0_reference='cone')),
    ('u_padded_probe_manifold', 'probe', 'manifold', 'uniform', 'newton_cg', dict(max_newton=6, padded_from=True)),
    ('u_mcsgd_probe_manifold', 'probe', 'manifold', 'uniform', 'mc_sgd', dict(batch_size=20, max_iter=30, check_every=10)),
    ('u_adam_apply_corewise', 'apply', 'corewise', 'uniform', 'adam', dict(batch_size=20, max_iter=100)),
    ('u_deriv1_probe_manifold', 'probe_derivatives', 'manifold', 'uniform', 'newton_cg', dict(max_newton=8), None, 1),
]


def main():
    if MODE == 'compare':
        old, new = (pickle.load(open(p, 'rb')) for p in sys.argv[2:4])
        worst = {}
        for name in old:
            a, b = old[name], new[name]
            if isinstance(a, str) or isinstance(b, str):
                print('%-34s ERROR old=%s new=%s' % (name, a if isinstance(a, str) else 'ok', b if isinstance(b, str) else 'ok'))
                continue
            same = (a['stop_reason'] == b['stop_reason'] and a['newton_steps'] == b['newton_steps']
                    and a['newton_accepted'] == b['newton_accepted'] and a['n_iter'] == b['n_iter']
                    and a['switched_at'] == b['switched_at'] and len(a['losses']) == len(b['losses']))
            la, lb = np.array(a['losses']), np.array(b['losses'])
            dl = float(np.max(np.abs(la - lb) / np.maximum(np.abs(la), 1e-300))) if len(la) == len(lb) and len(la) else float('nan')
            dx = float(np.linalg.norm(a['dense'] - b['dense']) / max(np.linalg.norm(a['dense']), 1e-300))
            fields = []
            for ra, rb in zip(a['history'], b['history']):
                for k in ra:
                    va, vb = ra[k], rb[k]
                    if isinstance(va, float) and isinstance(vb, float):
                        if not np.isclose(va, vb, rtol=1e-6, atol=1e-12) and not (np.isnan(va) and np.isnan(vb)):
                            fields.append(k)
                    elif va != vb:
                        fields.append(k)
            cone = ''
            if a['cone'] is not None:
                cone = ' cone %.1e' % float(np.max(np.abs(np.array(a['cone']) - np.array(b['cone']))))
            if a['start'] is not None:
                cone += ' start %.1e' % float(np.max(np.abs(np.array(a['start']) - np.array(b['start']))))
            print('%-34s %s steps %s/%s stop %s/%s losses %.1e x %.1e%s %s' % (
                name, 'SAME ' if same else 'DIFF ', a['newton_steps'] or a['n_iter'], b['newton_steps'] or b['n_iter'],
                a['stop_reason'], b['stop_reason'], dl, dx, cone, ('fields:' + ','.join(sorted(set(fields)))) if fields else ''))
        return
    out = {}
    only = sys.argv[3:] if len(sys.argv) > 3 else None
    for case in CASES:
        name, kind, geometry, layer, method, kw = case[:6]
        if only and name not in only:
            continue
        reg = case[6] if len(case) > 6 else None
        order = case[7] if len(case) > 7 else None
        weight = case[8] if len(case) > 8 else None
        try:
            out[name] = run(name, kind, geometry, layer, method, reg, order, weight, **dict(kw))[1]
        except Exception as e:
            import traceback
            out[name] = 'ERR ' + repr(e)[:300]
            traceback.print_exc()
        print(name, 'done', flush=True)
    pickle.dump(out, open(sys.argv[2], 'wb'))


main()
