"""Nonlinearity vs the number of tied factors m: one (d, seed, m, geom) cell per call.

Group = modes 0..m-1 tied (one group), the other modes singletons; m = 1 is untied (plain MANIFOLD).
Ranks generalize the d = 4 setup of shared_vs_unshared_exact.md (d = 4, m = 4 reproduces its (0,0,0,0) cell):
  target   Tucker 5 everywhere, TT (1,3,4,..,4,3,1)       (misspecified; edge modes' own ceiling 3 < 5)
  level 1  Tucker 2, TT (1,2,..,2,1), 40 GN steps, then the zero-padded resize restart
  restart  TT (1,2,3,..,3,2,1); unshared Tucker (2,3,..,3,2) (clipped to per-mode ceilings);
           shared: group rank 3 at every group mode (> mode 0's own ceiling 1*2: the group-ceiling case),
           singletons as unshared.
geom 'shared' fits on MANIFOLD.with_sharing(labels); 'unshared' fits the SAME data on MANIFOLD.
Exact (TR, power .5, gtol 1e-8, <= 40 steps) from the restart (fallback: 10 GN steps first if it does not
stop on `gradient`); GN (line search) from the restart, <= 60 steps, gtol 1e-8; the pencil at the exact x*.

usage (snapshot root): OMP_NUM_THREADS=1 PYTHONPATH=$PWD python run_scaling.py D SEED M GEOM
       writes results_scaling/d{D}_m{M}_{GEOM}_{SEED}.json
"""
import json
import os
import sys
import time
import numpy as np

import t3toolbox.tucker_tensor_train as t3
import t3toolbox.manifold as t3m
import t3toolbox.backend.fitting as bfit
import t3toolbox.backend.optimizers as bopt
import t3toolbox.backend.sharing as sharing
import exp_common as ec
from run_case import _slim, _jd

N, W, NOISE = 6, 60, 0.01
LEVEL1_STEPS, WARM_STEPS, EXACT_MAX, GN_MAX, GTOL = 40, 10, 40, 60, 1e-8


def ranks(d):
    tt_t = (1, 3) + (4,) * (d - 3) + (3, 1)
    tt_r = (1, 2) + (3,) * (d - 3) + (2, 1)
    return dict(target=((5,) * d, tt_t), level1=((2,) * d, (1,) + (2,) * (d - 1) + (1,)),
                unshared=((2,) + (3,) * (d - 2) + (2,), tt_r))


def labels_for(d, m):
    return tuple([0] * m + list(range(1, d - m + 1)))


def shared_restart(d, m):
    tk = list(ranks(d)['unshared'][0])
    for i in range(m):
        tk[i] = 3
    return tuple(tk), ranks(d)['unshared'][1]


def make_problem(d, seed, m):
    R = ranks(d)
    shape = (N,) * d
    labels = labels_for(d, m) if m >= 2 else None
    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    target = t3.TuckerTensorTrain.randn(shape, *R['target'])
    if labels is not None:
        target = t3.TuckerTensorTrain(*sharing.t3_tie_tucker_factors(target.data, labels))
    ww = [rng.standard_normal((W, N)) for _ in shape]
    pp = [rng.standard_normal((W, N)) for _ in shape]
    kind = bfit.probe_derivatives_kind(2, np.array([1.0, 0.5, 0.25]))
    data = [z + NOISE * np.linalg.norm(z) / z.size ** 0.5 * rng.standard_normal(z.shape)
            for z in kind.point_forward(target.data, (ww, pp))]
    x0 = t3.TuckerTensorTrain.randn(shape, *R['level1']).data
    if labels is not None:
        x0 = sharing.t3_tie_tucker_factors(x0, labels)
    probs = {'unshared': bopt.least_squares_problem(ec.MANIFOLD, kind, (ww, pp), data)}
    if labels is not None:
        probs['shared'] = bopt.least_squares_problem(ec.MANIFOLD.with_sharing(labels, shape), kind, (ww, pp), data)
    return probs, x0, labels, shape


def main(d, seed, m, geom):
    probs, x0, labels, shape = make_problem(d, seed, m)
    prob = probs[geom]
    shared = geom == 'shared'
    yn = ec.data_norm(prob)
    t0 = time.perf_counter()
    x1, st1 = bopt.newton_cg(prob, x0, max_newton=LEVEL1_STEPS, gtol_rel=1e-10)
    if shared:
        rr = shared_restart(d, m)
        restart = t3.TuckerTensorTrain(*x1).resize(shape, *rr, sharing=labels).data
        mdim = t3m.manifold_dim((shape,) + rr, sharing=labels)
    else:
        rr = ranks(d)['unshared']
        restart = t3.TuckerTensorTrain(*x1).resize(shape, *rr).data
        mdim = t3m.manifold_dim((shape,) + rr)
    out = dict(d=d, seed=seed, m=m, geom=geom, labels=labels, restart_ranks=rr, mdim=mdim)
    x_ex, st_ex = bopt.newton_cg(prob, restart, max_newton=EXACT_MAX, hessian='exact', gtol_rel=GTOL)
    start = 'restart'
    if st_ex['history'][-1].get('stop_reason') != 'gradient':      # fallback: GN warm start
        xw, _ = bopt.newton_cg(prob, restart, max_newton=WARM_STEPS, gtol_rel=0.0)
        x_ex, st_ex = bopt.newton_cg(prob, xw, max_newton=EXACT_MAX, hessian='exact', gtol_rel=GTOL)
        start = 'warm'
    _, st_gn = bopt.newton_cg(prob, restart, max_newton=GN_MAX, gtol_rel=GTOL)
    sgn = ec.summarize(st_gn['history'], yn)
    sgn['gn_tail_rate'] = ec.gn_tail_rate(st_gn['history'])
    out['exact_start'] = start
    out['runs'] = dict(exact=dict(summary=ec.summarize(st_ex['history'], yn), history=_slim(st_ex['history'])),
                       gn=dict(summary=sgn, history=_slim(st_gn['history'])))
    out['final'] = ec.final_point_curvature(prob, x_ex, seed=seed)
    out['final']['rel_misfit'] = out['runs']['exact']['summary']['rel_misfit']
    out['final']['xnorm'] = float(np.linalg.norm(t3.TuckerTensorTrain(*x_ex).to_dense()))   # sigma_min is absolute
    out['elapsed'] = time.perf_counter() - t0
    os.makedirs('results_scaling', exist_ok=True)
    with open('results_scaling/d%d_m%d_%s_%d.json' % (d, m, geom, seed), 'w') as f:
        json.dump(out, f, default=_jd)
    se, sg = out['runs']['exact']['summary'], out['runs']['gn']['summary']
    print('d%d m%d %-8s s%d %.1fs | exact(%s) %d/%d %s | GN %d/%d %s | rho %.3f lamRMS %.3f dim %d/%d mis %.3f' % (
        d, m, geom, seed, out['elapsed'], start, se['outer'], se['cg'], se['stop'], sg['outer'], sg['cg'], sg['stop'],
        out['final']['rho_gn'], float(np.sqrt(np.mean(np.square(out['final']['lam_all'])))),
        out['final']['horiz_dim'], mdim, se['rel_misfit']), flush=True)


if __name__ == '__main__':
    main(int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4])
