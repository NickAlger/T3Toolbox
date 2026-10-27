"""One (seed, case, geometry) cell: level-1 fit, zero-padded restart, 10-step GN warm start; exact (TR, power 0.5)
and GN (line search, power 0.5) from both starts; the final-point curvature spectrum. Writes results/<tag>.json.

usage: PYTHONPATH=$PWD python run_case.py SEED CASE GEOM     CASE in s0011 | s0000 | untied, GEOM in shared | unshared
"""
import json
import os
import sys
import time
import numpy as np
import t3toolbox.backend.optimizers as bopt
import exp_common as ec

LABELS = {'s0011': (0, 0, 1, 1), 's0000': (0, 0, 0, 0), 'untied': None}


def main(seed, case, geom):
    labels = LABELS[case]
    probs, x0 = ec.make_problem(seed, labels)
    prob = probs[geom]
    shared = geom == 'shared'
    yn = ec.data_norm(prob)
    t0 = time.perf_counter()
    restart, st1 = ec.restart_point(prob, x0, labels if shared else None)
    out = dict(seed=seed, case=case, geom=geom, labels=labels, level1_e=st1['history'][-1]['gnorm'] / st1['history'][-1]['g0norm'],
               mdim=ec.manifold_dim(labels, shared))
    xw, _ = bopt.newton_cg(prob, restart, max_newton=ec.WARM_STEPS, gtol_rel=0.0)
    runs = {}
    finals = {}
    for start_name, xs in (('restart', restart), ('warm', xw)):
        x_ex, st_ex = bopt.newton_cg(prob, xs, max_newton=ec.EXACT_MAX, hessian='exact', gtol_rel=ec.GTOL)
        finals[start_name] = x_ex
        runs['exact_' + start_name] = dict(summary=ec.summarize(st_ex['history'], yn), history=_slim(st_ex['history']))
    _, st_gn = bopt.newton_cg(prob, restart, max_newton=ec.GN_MAX, gtol_rel=1e-15)
    h = st_gn['history']
    for start_name, hh in (('restart', h[:61]), ('warm', h[ec.WARM_STEPS:ec.WARM_STEPS + 61])):
        s = ec.summarize(hh, yn, cut_e=ec.GTOL)
        gt = runs['exact_' + start_name]['summary']['final_g']
        s['gn_steps_to_exact_g'] = ec.gn_steps_to(hh, gt, cap=60)
        s['gn_tail_rate'] = ec.gn_tail_rate(hh)
        s['gn_min_e60'] = min(ec._e(hh[:61]))
        s['gn_one_minus_rho'] = [1 - r['rho'] for r, e in zip(hh, ec._e(hh))
                                 if r['rho'] is not None and 1e-7 < e < 1e-2]
        runs['gn_' + start_name] = dict(summary=s, history=_slim(hh))
    out['runs'] = runs
    out['final'] = ec.final_point_curvature(prob, finals['restart'], seed=seed)
    out['final']['rel_misfit'] = runs['exact_restart']['summary']['rel_misfit']
    out['elapsed'] = time.perf_counter() - t0
    rd = 'results_W%d' % ec.W
    os.makedirs(rd, exist_ok=True)
    with open('%s/%s_%s_%d.json' % (rd, case, geom, seed), 'w') as f:
        json.dump(out, f, default=_jd)
    s = {k: (v['summary']['outer'], v['summary']['cg'], v['summary']['stop']) for k, v in runs.items()}
    print(case, geom, seed, 'elapsed %.1f' % out['elapsed'], s, 'rho_gn=%.3g' % out['final']['rho_gn'],
          'dim', out['final']['horiz_dim'], out['mdim'])


KEEP = ('iteration', 'objective', 'misfit', 'gnorm', 'g0norm', 'converged', 'cg_iters', 'pHp', 'pKp', 'rho', 'tr_radius',
        'tr_accepted', 'ls_steps', 'ls_failed', 'wall_time', 'stop_reason', 'lift_sigma_min', 'slope')


def _slim(h):
    return [{k: r.get(k) for k in KEEP} for r in h]


def _jd(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.bool_):
        return bool(o)
    raise TypeError(type(o))


if __name__ == '__main__':
    main(int(sys.argv[1]), sys.argv[2], sys.argv[3])
