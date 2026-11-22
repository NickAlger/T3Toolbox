"""Frontend side by side (P16 part 4; the part 4a reviewer's script, kept with the slice): the OLD call form on the
pre-P16 library (`git archive 0acf7868`) against the same old-form call on the new tree (through the translation), and,
in the new tree, against a NEW-form call built independently here. `tests/test_optimizers_frontend.py`'s
`TestTheTranslation` pins numbers from its old run. The translation is deleted in P17, and this script with it.

    PYTHONPATH=<old_lib> python fe_sbs.py old  old.pkl
    PYTHONPATH=<repo>    python fe_sbs.py new  new.pkl
    python fe_sbs.py compare old.pkl new.pkl
"""
import itertools
import math
import pickle
import sys
import warnings

import numpy as np

MODE = sys.argv[1]
SHAPE = (5, 6, 5)
D = len(SHAPE)
W = 40


def target(seed):
    rng = np.random.default_rng(seed)
    nD, r = (2, 3, 2), (1, 2, 2, 1)
    B = [rng.standard_normal((nD[i], SHAPE[i])) for i in range(D)]
    G = [rng.standard_normal((r[i], nD[i], r[i + 1])) for i in range(D)]
    return B, G


def dense(B, G):
    T = np.einsum('aib,bjc,ckd->ijk', *G)[...] if False else None
    core = np.einsum('aib,bjc,ckd->aijkd', *G)[0, ..., 0]
    return np.einsum('ijk,ix,jy,kz->xyz', core, *B)


def coeffs_apply(Td, ww, pp, n):
    """Taylor coefficients in s of <T, (w1+s p1) x (w2+s p2) x (w3+s p3)>, t = 0..n, shape (n+1, W)."""
    out = np.zeros((n + 1, ww[0].shape[0]))
    for t in range(min(n, D) + 1):
        for S in itertools.combinations(range(D), t):
            vs = [pp[j] if j in S else ww[j] for j in range(D)]
            out[t] += np.einsum('abc,wa,wb,wc->w', Td, *vs)
    return out


def coeffs_probe(Td, ww, pp, n):
    outs = []
    subs = {0: 'abc,wb,wc->wa', 1: 'abc,wa,wc->wb', 2: 'abc,wa,wb->wc'}
    for i in range(D):
        others = [j for j in range(D) if j != i]
        out = np.zeros((n + 1, ww[0].shape[0], SHAPE[i]))
        for t in range(min(n, D - 1) + 1):
            for S in itertools.combinations(others, t):
                vs = [pp[j] if j in S else ww[j] for j in others]
                out[t] += np.einsum(subs[i], Td, *vs)
        outs.append(out)
    return outs


def one_hot(index):
    return [np.eye(N)[index[i]] for i, N in enumerate(SHAPE)]


def problem(kind, n, seed=0, Wn=W):
    rng = np.random.default_rng(100 + seed)
    Td = dense(*target(seed))
    ww = [rng.standard_normal((Wn, N)) / np.sqrt(N) for N in SHAPE]
    pp = [rng.standard_normal((Wn, N)) / np.sqrt(N) for N in SHAPE]
    index = np.stack([rng.integers(0, N, Wn) for N in SHAPE])
    noise = lambda a: a + 0.01 * rng.standard_normal(np.shape(a))
    if kind == 'apply':
        return ww, noise(coeffs_apply(Td, ww, pp, 0)[0]), ww, pp
    if kind == 'entries':
        return index, noise(Td[tuple(index)]), one_hot(index), None
    if kind == 'probe':
        return ww, [noise(z[0]) for z in coeffs_probe(Td, ww, pp, 0)], ww, pp
    fact = np.array([math.factorial(t) for t in range(n + 1)])
    if kind == 'apply_derivatives':
        c = noise(coeffs_apply(Td, ww, pp, n))
        return (ww, pp), c * fact[:, None], c, (ww, pp)
    if kind == 'entries_derivatives':
        c = noise(coeffs_apply(Td, one_hot(index), pp, n))
        return (index, pp), c * fact[:, None], c, (one_hot(index), pp)
    if kind == 'probe_derivatives':
        cs = [noise(z) for z in coeffs_probe(Td, ww, pp, n)]
        return (ww, pp), [z * fact[:, None, None] for z in cs], cs, (ww, pp)


def x0_of(seed=7, uniform=False):
    import t3toolbox.tucker_tensor_train as t3
    B, G = target(seed)
    x = t3.TuckerTensorTrain(tuple(0.3 * b for b in B), tuple(G))
    if uniform:
        import t3toolbox.uniform_tucker_tensor_train as ut3
        return ut3.UniformTuckerTensorTrain.from_t3(x)
    return x


def dense_of(x):
    return np.asarray(x.to_dense())


def rec(x, stats):
    keep = ('objective', 'gnorm', 'stop_reason', 'cg_iters', 'misfit', 'regularization')
    hist = [{k: row.get(k) for k in keep} for row in stats.get('history', [])]
    diag = stats.get('diagnostics')
    if diag is not None:
        diag = [{k: (np.asarray(v) if v is not None else None) for k, v in r.items()
                 if k in ('train_err', 'val_err')} for r in diag]
    return dict(dense=dense_of(x), losses=[float(v) for v in stats.get('losses', [])], history=hist,
                stop=stats.get('stop_reason'), diag=diag)


def draw_of(sample, data, kind, b):
    """An OLD custom draw: (sample_B, data_B), the same rng calls in both runs."""
    deriv = kind.endswith('_derivatives')

    def draw(rng):
        idx = rng.choice(W, size=b, replace=False)
        if kind.startswith('entries'):
            s = (sample[0][:, idx], [p[idx] for p in sample[1]]) if deriv else sample[:, idx]
        else:
            s = ([w[idx] for w in sample[0]], [p[idx] for p in sample[1]]) if deriv else [w[idx] for w in sample]
        if kind.startswith('probe'):
            dd = [y[:, idx] for y in data] if deriv else [y[idx] for y in data]
        else:
            dd = data[:, idx] if deriv else data[idx]
        return s, dd
    return draw


def cases():
    import t3toolbox.optimizers as topt
    import t3toolbox.manifold as t3m
    import t3toolbox.uniform_manifold as ut3m
    from t3toolbox.shared_geometry import shared
    reg = topt.IdentityRegularizer(0.05)
    out = {}

    def run(name, fn):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                out[name] = rec(*fn())
        except Exception as e:                                   # noqa
            out[name] = 'RAISED %s: %s' % (type(e).__name__, str(e)[:150])

    nc = dict(max_newton=6)
    s, y, *_ = problem('probe', 0)
    run('A_probe_permode', lambda: topt.newton_cg(t3m.MANIFOLD, 'probe', s, y, x0_of(), weight=[1., 2., .5], **nc))
    run('B_probe_len1', lambda: topt.newton_cg(t3m.MANIFOLD, 'probe', s, y, x0_of(), weight=[2.], **nc))
    run('B2_probe_extra_rows', lambda: topt.newton_cg(t3m.MANIFOLD, 'probe', s, y, x0_of(), weight=[1., 2., .5, 9.], **nc))
    s, y, *_ = problem('apply_derivatives', 2)
    run('C_applyd2_bare3', lambda: topt.newton_cg(t3m.MANIFOLD, 'apply_derivatives', s, y, x0_of(), order=2,
                                                  weight=[1., .5, .25], **nc))
    s, y, *_ = problem('probe_derivatives', 2)
    run('D_probed2_bare3', lambda: topt.newton_cg(t3m.MANIFOLD, 'probe_derivatives', s, y, x0_of(), order=2,
                                                  weight=[1., .5, .25], **nc))
    s, y, *_ = problem('probe_derivatives', 1)
    run('E_probed1_matrix', lambda: topt.newton_cg(t3m.MANIFOLD, 'probe_derivatives', s, y, x0_of(), order=1,
                                                   weight=[[1., .5], [2., .3], [.7, 1.1]], **nc))
    run('F_probed1_extra', lambda: topt.newton_cg(t3m.MANIFOLD, 'probe_derivatives', s, y, x0_of(), order=1,
                                                  weight=[[1., .5], [2., .3], [.7, 1.1], [5., 5.]], **nc))
    run('G_probed1_col', lambda: topt.newton_cg(t3m.MANIFOLD, 'probe_derivatives', s, y, x0_of(), order=1,
                                                weight=[[1.], [2.], [.7]], **nc))
    run('G2_probed1_row', lambda: topt.newton_cg(t3m.MANIFOLD, 'probe_derivatives', s, y, x0_of(), order=1,
                                                 weight=[[1., 3.]], **nc))
    s, y, *_ = problem('entries_derivatives', 1)
    run('H_entriesd1', lambda: topt.newton_cg(t3m.MANIFOLD, 'entries_derivatives', s, y, x0_of(), order=1,
                                              weight=[1., 2.], **nc))
    s, y, *_ = problem('probe_derivatives', 0)
    run('I_probed0', lambda: topt.newton_cg(t3m.MANIFOLD, 'probe_derivatives', s, y, x0_of(), order=0,
                                            weight=[[1.], [2.], [.5]], **nc))
    s, y, *_ = problem('apply_derivatives', 0)
    run('I2_applyd0', lambda: topt.newton_cg(t3m.MANIFOLD, 'apply_derivatives', s, y, x0_of(), order=0, **nc))
    # positional order and weight; exact Hessian
    s, y, *_ = problem('apply_derivatives', 1)
    run('P_applyd1_positional_exact', lambda: topt.newton_cg(t3m.MANIFOLD, 'apply_derivatives', s, y, x0_of(), 1,
                                                             [1., 2.], hessian='exact', max_newton=5))
    # validation, verbose records
    s, y, *_ = problem('probe_derivatives', 1, seed=0)
    vs, vy, *_ = problem('probe_derivatives', 1, seed=1)

    def J():
        import io, contextlib
        with contextlib.redirect_stdout(io.StringIO()):
            return topt.newton_cg(t3m.MANIFOLD, 'probe_derivatives', s, y, x0_of(), order=1, weight=[1., 2.],
                                  verbose=True, val_sample=vs, val_data=vy, max_newton=4)
    run('J_val_verbose', J)
    run('J2_val_only_sample', lambda: topt.newton_cg(t3m.MANIFOLD, 'probe_derivatives', s, y, x0_of(), order=1,
                                                     val_sample=vs, max_newton=2))
    # mc_sgd, custom draw, regularizer, weight, POSITIONAL batch
    run('K_mcsgd_probed1_draw_reg', lambda: topt.mc_sgd(
        t3m.MANIFOLD, 'probe_derivatives', s, y, x0_of(), np.random.default_rng(3), 15, order=1, weight=[1., 2.],
        draw=draw_of(s, y, 'probe_derivatives', 15), regularizer=reg, max_iter=8))
    run('K2_mcsgd_probed1_draw_noreg', lambda: topt.mc_sgd(
        t3m.MANIFOLD, 'probe_derivatives', s, y, x0_of(), np.random.default_rng(3), 15, order=1, weight=[1., 2.],
        draw=draw_of(s, y, 'probe_derivatives', 15), max_iter=8))
    sa, ya, *_ = problem('apply', 0)
    run('L_mcsgd_apply_flat_kwbatch_reg', lambda: topt.mc_sgd(
        t3m.MANIFOLD, 'apply', sa, ya, x0_of(), np.random.default_rng(4), batch=20, regularizer=reg, max_iter=8))
    se, ye, *_ = problem('entries_derivatives', 1)
    run('M_adam_cw_entriesd_draw_reg', lambda: topt.adam(
        t3m.COREWISE, 'entries_derivatives', se, ye, x0_of(), np.random.default_rng(5), 12, order=1,
        draw=draw_of(se, ye, 'entries_derivatives', 12), regularizer=reg, max_iter=8, lr=1e-2))
    run('N_gd_apply_chunk', lambda: topt.gradient_descent(t3m.MANIFOLD, 'apply', sa, ya, x0_of(), chunk_size=None,
                                                          n_iter=5))
    run('N2_gd_cw_probe_reg', lambda: topt.gradient_descent(t3m.COREWISE, 'probe', *problem('probe', 0)[:2], x0_of(),
                                                            regularizer=reg, n_iter=5))
    # uniform
    run('O_uniform_probed1_matrix', lambda: topt.newton_cg(
        ut3m.UNIFORM_MANIFOLD, 'probe_derivatives', s, y, x0_of(uniform=True), order=1,
        weight=[[1., .5], [2., .3], [.7, 1.1]], **nc))
    run('O2_uniform_mcsgd_draw_reg', lambda: topt.mc_sgd(
        ut3m.UNIFORM_MANIFOLD, 'probe_derivatives', s, y, x0_of(uniform=True), np.random.default_rng(3), 15,
        order=1, draw=draw_of(s, y, 'probe_derivatives', 15), regularizer=reg, max_iter=6))
    # shared
    sp, yp, *_ = problem('probe', 0)
    run('Q_shared_probe', lambda: topt.newton_cg(shared(t3m.MANIFOLD, (0, 1, 0)), 'probe', sp, yp,
                                                 x0_of().share((0, 1, 0)) if hasattr(x0_of(), 'share') else x0_of(),
                                                 **nc))
    run('R_cw_apply_reg_exact', lambda: topt.newton_cg(t3m.COREWISE, 'apply', sa, ya, x0_of(), regularizer=reg,
                                                       hessian='exact', max_newton=5))
    return out


def newform():
    """New-form calls built independently (Taylor data from the coefficients, weight times t!)."""
    import t3toolbox.optimizers as topt
    import t3toolbox.manifold as t3m
    from t3toolbox.fitting import LeastSquaresOracle as O, LeastSquaresBatch as Bt, affine_jet
    out = {}
    nc = dict(max_newton=6)
    s, y, c, (ww, pp) = problem('probe_derivatives', 2)
    w = np.tile(np.array([1., .5, .25]) * np.array([1, 1, 2]), (D, 1))
    out['D_probed2_bare3'] = rec(*topt.newton_cg(t3m.MANIFOLD, O('probe', (2,)), Bt(affine_jet(ww, pp), c, w),
                                                 x0_of(), **nc))
    s, y, c, (ww, pp) = problem('apply_derivatives', 2)
    out['C_applyd2_bare3'] = rec(*topt.newton_cg(t3m.MANIFOLD, O('apply', (2,)),
                                                 Bt(affine_jet(ww, pp), c, np.array([1., .5, .5])), x0_of(), **nc))
    s, y, c, (oh, pp) = problem('entries_derivatives', 1)
    out['H_entriesd1'] = rec(*topt.newton_cg(t3m.MANIFOLD, O('apply', (1,)),
                                             Bt(affine_jet(oh, pp), c, np.array([1., 2.])), x0_of(), **nc))
    s, y, c, (ww, pp) = problem('probe_derivatives', 0)
    out['I_probed0'] = rec(*topt.newton_cg(t3m.MANIFOLD, O('probe', (0,)),
                                           Bt(affine_jet(ww, pp), c, np.array([[1.], [2.], [.5]])), x0_of(), **nc))
    return out


if __name__ == '__main__':
    if MODE in ('old', 'new'):
        res = cases()
        if MODE == 'new':
            res['__newform__'] = newform()
        with open(sys.argv[2], 'wb') as f:
            pickle.dump(res, f)
        print('wrote', len(res))
    else:
        old, new = (pickle.load(open(p, 'rb')) for p in sys.argv[2:4])

        def cmp(a, b):
            if isinstance(a, str) or isinstance(b, str):
                return 'old=%s | new=%s' % (a if isinstance(a, str) else 'ran', b if isinstance(b, str) else 'ran')
            dx = np.linalg.norm(a['dense'] - b['dense']) / max(np.linalg.norm(a['dense']), 1e-300)
            la, lb = np.array(a['losses']), np.array(b['losses'])
            dl = (np.max(np.abs(la - lb) / np.maximum(np.abs(la), 1e-300)) if la.shape == lb.shape and la.size
                  else 'len %d vs %d' % (la.size, lb.size))
            ha = [r['objective'] for r in a['history']]
            hb = [r['objective'] for r in b['history']]
            dh = (max([abs(p - q) / max(abs(p), 1e-300) for p, q in zip(ha, hb)] or [0.])
                  if len(ha) == len(hb) else 'hist len %d vs %d' % (len(ha), len(hb)))
            ddg = ''
            if a.get('diag') is not None and b.get('diag') is not None:
                m = 0.
                for ra, rb in zip(a['diag'], b['diag']):
                    for k in ('train_err', 'val_err'):
                        if ra.get(k) is not None:
                            m = max(m, float(np.max(np.abs(np.ravel(ra[k]) - np.ravel(rb[k])))))
                ddg = ' diag %.1e (n %d/%d)' % (m, len(a['diag']), len(b['diag']))
            return 'dx %.1e dloss %s dhist %s stop %s/%s%s' % (dx, dl if isinstance(dl, str) else '%.1e' % dl,
                                                            dh if isinstance(dh, str) else '%.1e' % dh,
                                                            a['stop'], b['stop'], ddg)
        for k in old:
            print('%-32s %s' % (k, cmp(old[k], new.get(k, 'MISSING'))))
        for k, v in new.get('__newform__', {}).items():
            print('%-32s %s' % ('newform/' + k, cmp(old[k], v)))
