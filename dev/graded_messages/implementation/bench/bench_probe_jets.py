"""Old derivative kernels vs the engine at the jet, probe kind."""
import sys, math
import numpy as np
import t3toolbox.backend.sampling_derivatives as sd
import t3toolbox.backend.probing as pr
from t3toolbox.backend.jets import affine_jet
from bench_apply_jets import setup, best, count, CONFIGS

def calls(d, N, r, W, order):
    x, frame, var, ww, pp, _ = setup(d, N, r, W, order)
    jet = affine_jet(ww, pp)
    o = (order,)
    fact = np.array([math.factorial(t) for t in range(order + 1)]).reshape((-1, 1, 1))
    zt = [np.random.randn(order + 1, W, N) for _ in range(d)]
    old = {
        'point_fwd':   lambda: sd.t3_probe_derivatives(ww, pp, x.data, order),
        'tangent_fwd': lambda: sd.tv_probe_derivatives(ww, pp, var, frame, order),
        'transpose':   lambda: sd.tv_probe_derivatives_transpose(zt, ww, pp, frame, order, sum_over_probes=True),
    }
    new = {
        'point_fwd':   lambda: [fact * z for z in pr.t3_probe(x.data, jet, o)],
        'tangent_fwd': lambda: [fact * z for z in pr.tv_probe(frame, var, jet, o)],
        'transpose':   lambda: pr.tv_probe_transpose(frame, jet, [fact * z for z in zt], o),
    }
    return old, new

if __name__ == '__main__':
    mode = sys.argv[1]
    for name in sys.argv[2:]:
        cfg = CONFIGS[name]
        old, new = calls(*cfg)
        print('%s  (d, N, r, W, order) = %s' % (name, cfg))
        for k in old:
            a, b = old[k](), new[k]()
            if k == 'transpose':
                err = max(float(np.max(abs(u - v))) / (float(np.max(abs(u))) + 1e-300) for u, v in zip(a[0] + a[1], b[0] + b[1]))
            else:
                err = max(float(np.max(abs(u - v))) / float(np.max(abs(u))) for u, v in zip(a, b))
            if mode == 'count':
                print('   %-12s rel err %.1e   old %s   new %s' % (k, err, count(old[k]), count(new[k])))
            else:
                to, tn = best(old[k]), best(new[k])
                print('   %-12s rel err %.1e   old %9.2f ms   new %9.2f ms   new/old %.2f' % (k, err, 1e3 * to, 1e3 * tn, tn / to))
        sys.stdout.flush()
