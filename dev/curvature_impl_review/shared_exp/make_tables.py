"""Print the two markdown tables of shared_vs_unshared_exact.md from results_W60/ (+ results_W120/)."""
import glob
import json
import numpy as np

CELLS = [('s0011', 'shared'), ('s0011', 'unshared'), ('s0000', 'shared'), ('s0000', 'unshared'), ('untied', 'unshared')]
NAME = {('s0011', 'shared'): '(0,0,1,1) shared', ('s0011', 'unshared'): '(0,0,1,1) data, unshared',
        ('s0000', 'shared'): '(0,0,0,0) shared', ('s0000', 'unshared'): '(0,0,0,0) data, unshared',
        ('untied', 'unshared'): 'untied, unshared'}


def load(W, case, geom, seeds):
    ds = [json.load(open(f)) for f in glob.glob('results_W%d/%s_%s_*.json' % (W, case, geom))]
    return sorted([d for d in ds if d['seed'] in seeds], key=lambda d: d['seed'])


def mr(xs, f='%.2g'):
    xs = [x for x in xs if x is not None]
    return (f + ' [' + f + '–' + f + ']') % (np.median(xs), min(xs), max(xs))


def md(xs, f='%d'):
    return f % np.median(xs)


print('| cell | start | exact: outer / acc / CG | exact tail e (last 4) | GN: outer / CG to e≤1e-8 | GN capped | GN tail e_{k+1}/e_k | GN steps→exact final ‖g‖ | rel misfit | wall exact / GN s |')
print('|---|---|---|---|---|---|---|---|---|---|')
for c in CELLS:
    ds = load(60, *c, seeds=(1, 2, 3))
    for st in ('restart', 'warm'):
        E = [d['runs']['exact_' + st]['summary'] for d in ds]
        G = [d['runs']['gn_' + st]['summary'] for d in ds]
        te = np.median(np.array([s['tail_e'] for s in E]), axis=0)
        gr = [np.median(s['gn_tail_rate']) for s in G if s['gn_tail_rate']]
        to = ','.join('>60' if s['gn_steps_to_exact_g'] is None else str(s['gn_steps_to_exact_g']) for s in G)
        print('| %s | %s | %s / %s / %s | %s | %s / %s | %d/3 | %s | %s | %s | %s / %s |' % (
            NAME[c], st, md([s['outer'] for s in E]), md([s['accepted'] for s in E]), mr([s['cg'] for s in E], '%d'),
            '→'.join('%.0e' % x for x in te), md([s['outer'] for s in G]), md([s['cg'] for s in G]),
            sum(s['stop'] != 'gradient' for s in G), mr(gr, '%.2f'), to, md([s['rel_misfit'] for s in E], '%.2f'),
            md([s['wall'] for s in E], '%.1f'), md([s['wall'] for s in G], '%.1f')))

print()
print('| cell | horiz. dim | κ on accepted exact steps: median ‖κ‖ (max) | random p: ‖pKp/pH_GNp‖ ; ‖Kp‖/‖H_GN p‖ | λ-RMS of H_GN⁻¹K | **ρ(H_GN⁻¹K) at x\\*** (9 seeds) | GN measured tail rate | ρ at W=120 (3 seeds) | cond H_GN |')
print('|---|---|---|---|---|---|---|---|---|')
seeds9 = tuple(range(1, 10))
for c in CELLS:
    ds = load(60, *c, seeds=seeds9)
    d12 = load(120, *c, seeds=(1, 2, 3))
    F = [d['final'] for d in ds]
    ks = [np.abs(d['runs']['exact_restart']['summary']['kappa_steps']) for d in ds]
    rates = [np.median(d['runs']['gn_restart']['summary']['gn_tail_rate']) for d in ds if d['runs']['gn_restart']['summary']['gn_tail_rate']]
    print('| %s | %d | %s (%s) | %s ; %s | %s | %s | %s | %s | %s |' % (
        NAME[c], F[0]['horiz_dim'], md([float(np.median(k)) for k in ks], '%.2f'), md([float(np.max(k)) for k in ks], '%.2f'),
        md([float(np.median(np.abs(f['rayleigh']))) for f in F], '%.3f'), md([float(np.median(f['norm_ratio'])) for f in F], '%.2f'),
        md([float(np.sqrt(np.mean(np.square(f['lam_all'])))) for f in F], '%.2f'),
        mr([f['rho_gn'] for f in F], '%.2f'), mr(rates, '%.2f'), mr([d['final']['rho_gn'] for d in d12], '%.2f'),
        md([f['gn_cond'] for f in F], '%.1f')))
for W, seeds in ((60, seeds9), (120, (1, 2, 3))):
    for case in ('s0011', 's0000'):
        a = {d['seed']: d['final']['rho_gn'] for d in load(W, case, 'shared', seeds)}
        b = {d['seed']: d['final']['rho_gn'] for d in load(W, case, 'unshared', seeds)}
        diff = [a[k] - b[k] for k in sorted(a)]
        print('W=%d %s paired ρ_shared − ρ_unshared: median %+.2f [%+.2f, %+.2f], shared larger %d/%d' % (
            W, case, np.median(diff), min(diff), max(diff), sum(x > 0 for x in diff), len(diff)))
for case, geom in CELLS:
    ds = load(60, case, geom, seeds9)
    r = [d['runs']['gn_restart']['summary']['outer'] / d['runs']['exact_restart']['summary']['outer'] for d in ds]
    print('GN/exact outer (restart, W=60, 9 seeds, GN capped at 60) %s: median %.1f [%.1f-%.1f]' % (NAME[(case, geom)], np.median(r), min(r), max(r)))
