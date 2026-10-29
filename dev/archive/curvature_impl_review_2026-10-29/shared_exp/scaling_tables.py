"""Tables for nonlinearity_vs_tied_factors.md from results_scaling/. usage: python scaling_tables.py D [SEEDS]"""
import glob
import json
import sys
import numpy as np

D = int(sys.argv[1]) if len(sys.argv) > 1 else 6
SEEDS = [int(s) for s in sys.argv[2].split(',')] if len(sys.argv) > 2 else [1, 2, 3, 4, 5]


PREV = {1: 'untied', 4: 's0000'}   # d = 4: these m are the SAME problems as results_W60/ (shared_vs_unshared_exact.md)


def load(m, geom):
    out = {}
    if D == 4 and m in PREV:           # reuse, adapted to this script's layout (all were exact-from-restart)
        for f in glob.glob('results_W60/%s_%s_*.json' % (PREV[m], geom)):
            d = json.load(open(f))
            if d['seed'] in SEEDS:
                d.update(m=m, exact_start='restart',
                         runs=dict(exact=d['runs']['exact_restart'], gn=d['runs']['gn_restart']))
                out[d['seed']] = d
    for f in glob.glob('results_scaling/d%d_m%d_%s_*.json' % (D, m, geom)):
        d = json.load(open(f))
        if d['seed'] in SEEDS:
            out[d['seed']] = d
    return out


def lrms(F):
    return float(np.sqrt(np.mean(np.square(F['lam_all']))))


MEAS = {  # name -> per-run scalar
    'rho': lambda d: d['final']['rho_gn'],
    'lrms': lambda d: lrms(d['final']),
    'rt': lambda d: float(np.median(d['final']['norm_ratio'])),        # ||Kp||/||H_GN p||, median of 10
    'rq': lambda d: float(np.median(np.abs(d['final']['rayleigh']))),  # |pKp/pH_GN p|
    'kmed': lambda d: float(np.median(np.abs(d['runs']['exact']['summary']['kappa_steps']))) if d['runs']['exact']['summary']['kappa_steps'] else None,
    'kmax': lambda d: float(np.max(np.abs(d['runs']['exact']['summary']['kappa_steps']))) if d['runs']['exact']['summary']['kappa_steps'] else None,
    'eo': lambda d: d['runs']['exact']['summary']['outer'],
    'ecg': lambda d: d['runs']['exact']['summary']['cg'],
    'ew': lambda d: d['runs']['exact']['summary']['wall'],
    'go': lambda d: d['runs']['gn']['summary']['outer'],
    'gcg': lambda d: d['runs']['gn']['summary']['cg'],
    'gw': lambda d: d['runs']['gn']['summary']['wall'],
    'gcap': lambda d: int(d['runs']['gn']['summary']['stop'] != 'gradient'),
    'grate': lambda d: float(np.median(d['runs']['gn']['summary']['gn_tail_rate'])) if d['runs']['gn']['summary']['gn_tail_rate'] else None,
    'dim': lambda d: d['final']['horiz_dim'],
    'mis': lambda d: d['final']['rel_misfit'],
    'cond': lambda d: d['final']['gn_cond'],
    'warm': lambda d: int(d['exact_start'] != 'restart'),
    'sig': lambda d: d['final']['lift_sigma_min'] / d['final']['xnorm'] if d['final'].get('xnorm') else None,   # relative
    'msig': lambda d: d['final']['rel_misfit'] * d['final']['xnorm'] / d['final']['lift_sigma_min'] if d['final'].get('xnorm') else None,
    'mis2': lambda d: d['final']['rel_misfit'],
}


def col(ds, k):
    return [MEAS[k](ds[s]) for s in sorted(ds)]


def mr(xs, f):
    xs = [x for x in xs if x is not None]
    if not xs:
        return '-'
    return (f + ' [' + f + '–' + f + ']') % (np.median(xs), min(xs), max(xs))


def md(xs, f):
    xs = [x for x in xs if x is not None]
    return f % np.median(xs) if xs else '-'


rows = [(1, 'unshared', 'untied')] + [(m, g, None) for m in range(2, D + 1) for g in ('shared', 'unshared')]
print('| m | target / fit | n | dim | rel misfit | **ρ(H_GN⁻¹K)** | λ-RMS | ‖Kp‖/‖H_GN p‖ | ‖pKp/pH_GNp‖ | κ med (max) | exact outer / CG / wall s | GN outer / CG / wall s (capped) | GN tail rate | cond H_GN |')
print('|---|---|---|---|---|---|---|---|---|---|---|---|---|---|')
for m, g, tag in rows:
    ds = load(m, g)
    if not ds:
        continue
    name = 'untied / unshared' if tag else ('tied / **shared**' if g == 'shared' else 'tied / unshared')
    print('| %d | %s | %d | %s | %s | %s | %s | %s | %s | %s (%s) | %s / %s / %s | %s / %s / %s (%d) | %s | %s |' % (
        m, name, len(ds), md(col(ds, 'dim'), '%d'), md(col(ds, 'mis'), '%.2f'),
        mr(col(ds, 'rho'), '%.2f'), mr(col(ds, 'lrms'), '%.2f'), mr(col(ds, 'rt'), '%.2f'), md(col(ds, 'rq'), '%.3f'),
        md(col(ds, 'kmed'), '%.2f'), md(col(ds, 'kmax'), '%.2f'),
        md(col(ds, 'eo'), '%d'), md(col(ds, 'ecg'), '%d'), md(col(ds, 'ew'), '%.1f'),
        md(col(ds, 'go'), '%d'), md(col(ds, 'gcg'), '%d'), md(col(ds, 'gw'), '%.1f'), sum(col(ds, 'gcap')),
        mr(col(ds, 'grate'), '%.2f'), md(col(ds, 'cond'), '%.1f')))
    if sum(col(ds, 'warm')):
        print('    (warm-start fallback used in %d runs)' % sum(col(ds, 'warm')))

print('\nper-seed rho:')
for m, g, tag in rows:
    ds = load(m, g)
    print('  m%d %-8s' % (m, g), ' '.join('%.2f' % MEAS['rho'](ds[s]) for s in sorted(ds)),
          '| lrms', ' '.join('%.2f' % MEAS['lrms'](ds[s]) for s in sorted(ds)),
          '| rt', ' '.join('%.2f' % MEAS['rt'](ds[s]) for s in sorted(ds)),
          '| mis', ' '.join('%.2f' % MEAS['mis'](ds[s]) for s in sorted(ds)))


def seq(g):
    """the m-series for one fit geometry; m = 1 is the untied/unshared cell in both series"""
    return [(1, load(1, 'unshared'))] + [(m, load(m, g)) for m in range(2, D + 1)]


def spearman(x, y):
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


print('\npaired consecutive-m differences (same seed): median [min, max], #positive/n')
for g in ('shared', 'unshared'):
    S = seq(g)
    for k in ('rho', 'lrms', 'rt', 'mis', 'dim'):
        parts = []
        for (m0, a), (m1, b) in zip(S, S[1:]):
            dd = [MEAS[k](b[s]) - MEAS[k](a[s]) for s in sorted(a) if s in b]
            if dd:
                parts.append('%d→%d %+.2f [%+.2f,%+.2f] %d/%d' % (m0, m1, np.median(dd), min(dd), max(dd), sum(x > 0 for x in dd), len(dd)))
        # per-seed rank correlation of the measure with m
        rs = []
        for s in SEEDS:
            ms = [m for m, ds in S if s in ds]
            if len(ms) >= 3:
                rs.append(spearman(np.array(ms), np.array([MEAS[k](ds[s]) for m, ds in S if s in ds])))
        print('  %-8s %-4s per-seed Spearman(m, .) %s | %s' % (g, k, ' '.join('%+.2f' % r for r in rs), '; '.join(parts)))

print('\nshared minus unshared on the same (tied) data, per m: median [min,max] #pos/n')
for k in ('rho', 'lrms', 'rt', 'mis'):
    parts = []
    for m in range(2, D + 1):
        a, b = load(m, 'shared'), load(m, 'unshared')
        dd = [MEAS[k](a[s]) - MEAS[k](b[s]) for s in sorted(a) if s in b]
        if dd:
            parts.append('m%d %+.2f [%+.2f,%+.2f] %d/%d' % (m, np.median(dd), min(dd), max(dd), sum(x > 0 for x in dd), len(dd)))
    print('  %-4s %s' % (k, '; '.join(parts)))

print('\npooled correlations over all runs (Spearman):')
allr = [d for m, g, _ in rows for d in load(m, g).values()]
for g in ('shared', 'unshared', 'all'):
    rr = [d for d in allr if g == 'all' or d['geom'] == g or (d['m'] == 1)]
    for a, b in (('rho', 'mis'), ('lrms', 'mis'), ('rt', 'mis'), ('rho', 'lrms'), ('lrms', 'dim'),
                 ('rho', 'sig'), ('rho', 'msig'), ('lrms', 'msig'), ('rt', 'msig')):
        q = [d for d in rr if MEAS[a](d) is not None and MEAS[b](d) is not None]
        if len(q) >= 4:
            print('  %-8s %s~%s %+.2f (n=%d)' % (g, a, b, spearman(np.array([MEAS[a](d) for d in q]), np.array([MEAS[b](d) for d in q])), len(q)))

print('\nrelative lift sigma_min/||X*|| and misfit*||X*||/sigma_min per row (median [range]; runs with xnorm only):')
for m, g, tag in rows:
    ds = load(m, g)
    if ds:
        print('  m%d %-8s sig %s  mis/sig %s' % (m, g, mr(col(ds, 'sig'), '%.3f'), mr(col(ds, 'msig'), '%.1f')))
