"""Aggregate results/*.json into the two tables (medians [min-max] over seeds). usage: python aggregate.py [prefix]"""
import glob
import json
import sys
import numpy as np

pre = sys.argv[1] if len(sys.argv) > 1 else 'results_W60'
SEEDS = [int(x) for x in sys.argv[2].split(',')] if len(sys.argv) > 2 else [1, 2, 3]
CELLS = [('s0011', 'shared'), ('s0011', 'unshared'), ('s0000', 'shared'), ('s0000', 'unshared'), ('untied', 'unshared')]


def load(case, geom):
    ds = [json.load(open(f)) for f in sorted(glob.glob('%s/%s_%s_*.json' % (pre, case, geom)))]
    return sorted([d for d in ds if d['seed'] in SEEDS], key=lambda d: d['seed'])


def mr(xs, fmt='%.3g'):
    xs = [x for x in xs if x is not None]
    if not xs:
        return '-'
    return (fmt + ' [' + fmt + '-' + fmt + ']') % (np.median(xs), min(xs), max(xs))


def cnt(xs):
    return '%d' % len(xs)


print('## runs')
print('| cell | run | outer | acc | CG | stop | rel misfit | wall s | tail e_{k+1}/e_k (last 3, seed-median) | final e (exact) / min e in 60 (GN) | GN steps to exact final g |')
for case, geom in CELLS:
    ds = load(case, geom)
    if not ds:
        continue
    for run in ('exact_restart', 'gn_restart', 'exact_warm', 'gn_warm'):
        S = [d['runs'][run]['summary'] for d in ds]
        stops = {}
        for s in S:
            stops[s['stop']] = stops.get(s['stop'], 0) + 1
        tail = np.median(np.array([s['tail_ratios'][-3:] for s in S if len(s['tail_ratios']) >= 3]), axis=0)
        to = [s.get('gn_steps_to_exact_g') for s in S] if run.startswith('gn') else None
        tos = '-' if to is None else ','.join('>60' if t is None else str(t) for t in to)
        print('| %s/%s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |' % (
            case, geom, run, mr([s['outer'] for s in S], '%d'), mr([s['accepted'] for s in S], '%d'),
            mr([s['cg'] for s in S], '%d'), ' '.join('%s:%d' % kv for kv in stops.items()),
            mr([s['rel_misfit'] for s in S], '%.4f'), mr([s['wall'] for s in S], '%.2f'),
            ', '.join('%.2g' % t for t in tail),
            mr([s['final_e'] for s in S], '%.1e') if run.startswith('exact') else mr([s['gn_min_e60'] for s in S], '%.1e'), tos))

print('\n## curvature')
print('| cell | dim | kappa on accepted exact steps: median |k| ; max |k| (restart) | random p: pKp/pGNp median |.| ; ||Kp||/||H_GN p|| median | rho(H_GN^-1 K) at x* | GN tail rate (median, e in 1e-9..1e-2) | GN 1-rho_ls | H_GN cond | lift sigma_min |')
for case, geom in CELLS:
    ds = load(case, geom)
    if not ds:
        continue
    ks = [np.abs(d['runs']['exact_restart']['summary']['kappa_steps']) for d in ds]
    kw = [np.abs(d['runs']['exact_warm']['summary']['kappa_steps']) for d in ds]
    F = [d['final'] for d in ds]
    rates = [np.median(d['runs']['gn_restart']['summary']['gn_tail_rate']) if d['runs']['gn_restart']['summary']['gn_tail_rate'] else None for d in ds]
    omr = [np.median(d['runs']['gn_restart']['summary']['gn_one_minus_rho']) if d['runs']['gn_restart']['summary']['gn_one_minus_rho'] else None for d in ds]
    print('| %s/%s | %d | %s ; %s (warm: %s ; %s) | %s ; %s | %s | %s | %s | %s | %s |' % (
        case, geom, F[0]['horiz_dim'],
        mr([float(np.median(k)) for k in ks], '%.2f'), mr([float(np.max(k)) for k in ks], '%.2f'),
        mr([float(np.median(k)) for k in kw], '%.2f'), mr([float(np.max(k)) for k in kw], '%.2f'),
        mr([float(np.median(np.abs(f['rayleigh']))) for f in F], '%.3f'), mr([float(np.median(f['norm_ratio'])) for f in F], '%.3f'),
        mr([f['rho_gn'] for f in F], '%.3f'), mr(rates, '%.3f'), mr(omr, '%.2f'), mr([f['gn_cond'] for f in F], '%.1f'),
        mr([f['lift_sigma_min'] for f in F], '%.1f')))
    print('    lam RMS:', mr([float(np.sqrt(np.mean(np.square(f['lam_all'])))) for f in F], '%.3f'), ' frac |lam|>0.3:', mr([float(np.mean(np.abs(f['lam_all']) > 0.3)) for f in F], '%.2f'))
    print('    per seed rho(H_GN^-1K):', ['%.3f' % f['rho_gn'] for f in F], ' GN rate:', ['%.3f' % r if r else '-' for r in rates],
          ' lam[min,max]:', ['[%.2f,%.2f]' % (f['lam_min'], f['lam_max']) for f in F], 'asym', '%.1e' % max(f['asym'] for f in F))

print('\n## paired rho(H_GN^-1 K) at x*, shared minus unshared (same data), seeds', SEEDS)
for case in ('s0011', 's0000'):
    a = {d['seed']: d['final'] for d in load(case, 'shared')}
    b = {d['seed']: d['final'] for d in load(case, 'unshared')}
    diff = [a[k]['rho_gn'] - b[k]['rho_gn'] for k in sorted(a) if k in b]
    rr = [float(np.sqrt(np.mean(np.square(a[k]['lam_all'])))) / float(np.sqrt(np.mean(np.square(b[k]['lam_all'])))) for k in sorted(a) if k in b]
    print(case, 'rho diff median %.3f range [%.3f, %.3f]; shared larger in %d/%d' % (np.median(diff), min(diff), max(diff), sum(x > 0 for x in diff), len(diff)),
          '| lam-RMS ratio shared/unshared median %.2f [%.2f-%.2f]' % (np.median(rr), min(rr), max(rr)))
    ex = [ (d['seed'], d['runs']['exact_restart']['summary']['outer'], d['runs']['gn_restart']['summary']['outer']) for d in load(case, 'shared')]
    ey = [ (d['seed'], d['runs']['exact_restart']['summary']['outer'], d['runs']['gn_restart']['summary']['outer']) for d in load(case, 'unshared')]
    print('   GN/exact outer-step ratio (restart): shared', ['%.1f' % (g / e) for _, e, g in ex], ' unshared', ['%.1f' % (g / e) for _, e, g in ey])
u = [d['final']['rho_gn'] for d in load('untied', 'unshared')]
print('untied rho median %.3f [%.3f-%.3f]' % (np.median(u), min(u), max(u)))
