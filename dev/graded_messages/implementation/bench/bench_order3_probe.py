"""The engine with each 3-operand explicit-path einsum given the cheapest FIRST PAIR (by FLOP count), pairwise after."""
import sys, re, functools, numpy as np
from bench_apply_jets import best, CONFIGS
from bench_probe_jets import calls
orig = np.einsum
PAIRS = ([(0, 1), (0, 1)], [(1, 2), (0, 1)], [(0, 2), (0, 1)])

@functools.lru_cache(maxsize=None)
def choose(sub, shapes):
    ops = [np.empty(s) for s in shapes]
    costs = []
    for p in PAIRS:
        text = np.einsum_path(sub, *ops, optimize=['einsum_path'] + p)[1]
        costs.append(float(re.search(r'Optimized FLOP count:\s*([0-9.e+]+)', text).group(1)))
    return ['einsum_path'] + PAIRS[int(np.argmin(costs))]

def cheapest(sub, *ops, **kw):
    if isinstance(kw.get('optimize'), list) and len(ops) == 3:
        kw['optimize'] = choose(sub, tuple(o.shape for o in ops))
    return orig(sub, *ops, **kw)

for name in sys.argv[1:]:
    old, new = calls(*CONFIGS[name])
    print('%s  (d, N, r, W, order) = %s' % (name, CONFIGS[name]))
    for k in ('point_fwd', 'tangent_fwd', 'transpose'):
        a = new[k]()
        np.einsum = cheapest
        try:
            b = new[k]()
        finally:
            np.einsum = orig
        flat = lambda v: np.concatenate([np.ravel(t) for t in (v[0] + v[1] if k == "transpose" else list(v))])
        err = float(np.max(abs(flat(a) - flat(b))) / np.max(abs(flat(a))))
        to, tn = best(old[k]), best(new[k])
        np.einsum = cheapest
        try:
            ts = best(new[k])
        finally:
            np.einsum = orig
        print('   %-12s old %8.1f ms   new %8.1f ms (%.2f)   new, cheapest first pair %8.1f ms (%.2f)  [same values: %.0e]'
              % (k, 1e3 * to, 1e3 * tn, tn / to, 1e3 * ts, ts / to, err))
    sys.stdout.flush()
