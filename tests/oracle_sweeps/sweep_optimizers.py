"""O2 oracle sweep, part 2: the four frontend optimizers. Checks: (a) every reported objective equals a
hand recomputation at the corresponding iterate; (b) ragged and uniform trajectories agree for the
deterministic optimizers; (c) a regularized run's history splits obj = misfit + reg consistently."""
import sys, time, traceback
import numpy as np
sys.path.insert(0, '.')
import oracle as O
import sweep_models as SM
import t3toolbox.tucker_tensor_train as t3
import t3toolbox.uniform_tucker_tensor_train as ut3
import t3toolbox.manifold as t3m
import t3toolbox.uniform_manifold as ut3m
import t3toolbox.shared_geometry as sg
import t3toolbox.optimizers as topt
from t3toolbox.backend.regularization import IdentityRegularizer

LAM = 0.3
failures = []
rows = []


def hand_objective(pt, kind, sample, y, weight, order, geom_name, reg):
    X = np.asarray(pt.to_dense())
    m = O.misfit(kind, X, sample, y, weight, order)
    r = SM.reg_value(geom_name, X, SM.point_cores(pt)) if reg else 0.0
    return m, r


import t3toolbox.backend.uniform_fitting as uf
def rewrap(x_cores, x0):
    if isinstance(x0, ut3.UniformTuckerTensorTrain):
        x0m = uf.uniform_minimal(x0)           # the frontend optimizes the minimal-reduced point
        return ut3.UniformTuckerTensorTrain(x_cores[0], x_cores[1], x0m.shape, x0m.masks)
    return t3.TuckerTensorTrain(*x_cores)


def setup(d, shared, kind, order, wspec, seed, W=(12,)):
    x, sharing = SM.build_point(d, shared, seed)
    rng = np.random.default_rng(seed + 1)
    sample = SM.make_sample(kind, x.shape, W, rng)
    weight = SM.make_weight(kind, d, order or 0, wspec)
    A0 = t3.TuckerTensorTrain.randn(x.shape, x.tucker_ranks, x.tt_ranks)
    A = t3.TuckerTensorTrain(tuple(rng.standard_normal(c.shape) for c in A0.data[0]),
                             tuple(rng.standard_normal(c.shape) for c in A0.data[1]))
    y = O.S(kind, A.to_dense(), sample, order)
    y = [np.asarray(v) for v in y] if kind.startswith('probe') else np.asarray(y)
    return x, sharing, sample, weight, y


def geometry_for(rep, geom_name, shared, sharing):
    if rep == 'ragged':
        base = t3m.MANIFOLD if geom_name == 'manifold' else t3m.COREWISE
    else:
        base = ut3m.UNIFORM_MANIFOLD if geom_name == 'manifold' else ut3m.UNIFORM_COREWISE
    return sg.shared(base, sharing) if shared else base


def run_case(d, rep, geom_name, shared, kind, order, wspec, regspec, seed=11):
    x, sharing, sample, weight, y = setup(d, shared, kind, order, wspec, seed)
    reg = IdentityRegularizer(LAM) if regspec else None
    geom = geometry_for(rep, geom_name, shared, sharing)
    x0 = x if rep == 'ragged' else ut3.UniformTuckerTensorTrain.from_t3(x)
    kw = dict(order=order, weight=weight, regularizer=reg)
    res = {}
    # ---- newton_cg with a callback capturing x_cores
    recs = []
    def cb(info):
        recs.append(info)
    x_nt, st = topt.newton_cg(geom, kind, sample, y, x0, max_newton=3, callback=cb, **kw)
    errs = []
    for info, h in zip(recs, st['history']):
        pt = rewrap(info.x_cores, x0)
        m, r = hand_objective(pt, kind, sample, y, weight, order, geom_name, reg)
        errs.append(SM.rel(info.objective, m + r))
        errs.append(SM.rel(info.misfit, m))
        if reg:
            errs.append(SM.rel(info.regularization, r))
            errs.append(SM.rel(h['objective'], h['misfit'] + h['regularization']))
        else:
            errs.append(0.0 if h['regularization'] is None else 1.0)
        errs.append(SM.rel(h['objective'], info.objective))
    res['newton_obj'] = max(errs)
    res['_newton_losses'] = list(st['losses'])
    res['_newton_dense'] = np.asarray(x_nt.to_dense())
    res['newton_descends'] = 0.0 if st['losses'][-1] <= st['losses'][0] else 1.0
    # ---- gradient_descent: losses[k] == hand objective at the iterate the (k)-iteration run returns
    x_g3, st3 = topt.gradient_descent(geom, kind, sample, y, x0, n_iter=3, **kw)
    x_g4, st4 = topt.gradient_descent(geom, kind, sample, y, x0, n_iter=4, **kw)
    m0, r0 = hand_objective(x0, kind, sample, y, weight, order, geom_name, reg)
    m3, r3 = hand_objective(x_g3, kind, sample, y, weight, order, geom_name, reg)
    res['gd_obj'] = max(SM.rel(st3['losses'][0], m0 + r0), SM.rel(st4['losses'][3], m3 + r3))
    res['_gd_losses'] = list(st4['losses'])
    res['_gd_dense'] = np.asarray(x_g4.to_dense())
    # ---- mc_sgd: one full-batch check at the end -> losses[0] == hand objective at the returned x
    rng = np.random.default_rng(3)
    x_mc, stm = topt.mc_sgd(geom, kind, sample, y, x0, rng, 6, max_iter=6, check_every=6, **kw)
    m, r = hand_objective(x_mc, kind, sample, y, weight, order, geom_name, reg)
    res['mc_obj'] = SM.rel(stm['losses'][0], m + r)
    # ---- adam: losses every 50 iterations -> losses[0] == hand objective at the returned cores
    rng = np.random.default_rng(3)
    x_ad, sta = topt.adam(geom, kind, sample, y, x0, rng, 6, max_iter=50, lr=1e-2, **kw)
    m, r = hand_objective(x_ad, kind, sample, y, weight, order, geom_name, reg)
    res['adam_obj'] = SM.rel(sta['losses'][0], m + r)
    return res


def main(case_filter=None):    # case_filter(key) -> bool; rep-agnostic so ragged/uniform pairs stay whole
    t0 = time.time()
    kinds = ['apply', 'entries', 'probe', 'apply_derivatives', 'entries_derivatives', 'probe_derivatives']
    pair = {}
    n = 0
    for d in ((2,) if len(sys.argv) > 1 and sys.argv[1] == 'd2' else (3, 2)):
        for geom_name, shared in [('manifold', False), ('corewise', False), ('manifold', True), ('corewise', True)]:
            for kind in kinds:
                order = 2 if kind.endswith('_derivatives') else None
                wspecs = ['none']
                if kind == 'probe': wspecs.append('mode')
                if kind.endswith('_derivatives'): wspecs.append('order')
                if kind == 'probe_derivatives': wspecs.append('matrix')
                for wspec in wspecs:
                    for regspec in (False, True):
                        for rep in ('ragged', 'uniform'):
                            key = (d, rep, geom_name, shared, kind, order, wspec, regspec)
                            if case_filter is not None and not case_filter(key):
                                continue
                            n += 1
                            try:
                                res = run_case(d, rep, geom_name, shared, kind, order, wspec, regspec)
                            except Exception as e:
                                failures.append((key, 'EXC', f'{type(e).__name__}: {str(e)[:200]}'))
                                continue
                            bad = {k: v for k, v in res.items() if not k.startswith('_') and not (v < 1e-8)}
                            if bad:
                                failures.append((key, 'NUM', bad))
                            pair.setdefault(key[:1] + key[2:], {})[rep] = res
    for key, both in pair.items():
        if len(both) == 2:
            a, b = both['ragged'], both['uniform']
            e = {'newton_losses': SM.rel(a['_newton_losses'], b['_newton_losses']),
                 'newton_dense': SM.rel(a['_newton_dense'], b['_newton_dense']),
                 'gd_losses': SM.rel(a['_gd_losses'], b['_gd_losses']),
                 'gd_dense': SM.rel(a['_gd_dense'], b['_gd_dense'])}
            bad = {k: v for k, v in e.items() if not (v < 1e-7)}
            if bad:
                failures.append((('ragged-vs-uniform',) + key, 'PAIR', bad))
    print(f'{n} optimizer cases in {time.time()-t0:.0f}s; {len(failures)} failures')
    for f in failures:
        print(' FAIL', f)


if __name__ == '__main__':
    main()
