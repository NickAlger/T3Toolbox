"""Critic pre-check (read-only): simulate v5's g0_reference='zero' DEFAULT on the O2 optimizer sweep's ragged-vs-uniform
pairing (newton_losses compared at rel 1e-7, tests/oracle_sweeps/sweep_optimizers.py:143). Each layer computes its
own zero reference (in its own layer, as the g0 plan requires), passed as g0norm_newton. Compare with today's default."""
import sys; sys.path.insert(0, __import__("os").getcwd()); sys.path.insert(0, "tests/oracle_sweeps")
import numpy as np
import t3toolbox.uniform_tucker_tensor_train as ut3
import t3toolbox.optimizers as topt
import t3toolbox.corewise as cw
import sweep_optimizers as SO
import sweep_models as SM

def zero_ref(geom, kind, sample, y, x0, order, weight):
    problem, init, _ = topt._setup(geom, kind, sample, y, x0, order, weight, None, chunk_size=None)
    g = problem.local_model(cw.corewise_zeros_like(init)).gradient
    return float(problem.geom.inner(g, g)) ** 0.5

worst_today, worst_zero = 0.0, 0.0
for d in (3,):
    for kind in ('apply', 'entries', 'probe', 'apply_derivatives', 'probe_derivatives'):
        order = 2 if kind.endswith('_derivatives') else None
        x, sharing, sample, weight, y = SO.setup(d, False, kind, order, 'none', 11)
        out = {}
        for mode in ('today', 'zero'):
            losses = {}
            for rep in ('ragged', 'uniform'):
                geom = SO.geometry_for(rep, 'manifold', False, sharing)
                x0 = x if rep == 'ragged' else ut3.UniformTuckerTensorTrain.from_t3(x)
                kw = dict(order=order, weight=weight, max_newton=6)
                if mode == 'zero':
                    kw['g0norm_newton'] = zero_ref(geom, kind, sample, y, x0, order, weight)
                _, st = topt.newton_cg(geom, kind, sample, y, x0, **kw)
                losses[rep] = np.array(st['losses'])
            n = min(len(losses['ragged']), len(losses['uniform']))
            out[mode] = SM.rel(losses['ragged'][:n], losses['uniform'][:n])
        worst_today, worst_zero = max(worst_today, out['today']), max(worst_zero, out['zero'])
        print(f"d={d} {kind:18s} ragged-vs-uniform newton_losses rel diff: today {out['today']:.1e} | zero-ref default {out['zero']:.1e}")
print(f'worst: today {worst_today:.1e}, zero-ref {worst_zero:.1e}  (O2 pair threshold 1e-7)')
