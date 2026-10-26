"""Read-only plan check: the block-sum oracle (zero new code) for the derivative kinds vs central FD
of the corewise derivative transposes -- i.e. the test oracle covers all six kinds."""
import sys; sys.path.insert(0, __import__("os").getcwd())
import numpy as np
import t3toolbox.backend.fv_conversions as fvc
import t3toolbox.backend.tv_operations as tv
import t3toolbox.backend.sampling_derivatives as sd
import t3toolbox.backend.entries as en
import t3toolbox.backend.t3_constructors as t3k
rng = np.random.default_rng(13)
def blocksum(gY, d, tr, ttr):
    gYU, gYG = gY
    gU = tuple(gYU[i][..., :tr[i], :] for i in range(d))
    gG = []
    for i in range(d):
        r, n, rn = ttr[i], tr[i], ttr[i + 1]; g = gYG[i]
        if i == 0:   gG.append(g[..., :, :n, rn:] + g[..., :, n:, :rn])
        elif i == d - 1: gG.append(g[..., :r, :n, :] + g[..., r:, n:, :])
        else:        gG.append(g[..., :r, :n, :rn] + g[..., r:, :n, rn:] + g[..., r:, n:, :rn])
    return gU, tuple(gG)
flat = lambda a: np.concatenate([x.ravel() for x in a[0] + a[1]])
add = lambda a, b, s: (tuple(x + s * y for x, y in zip(a[0], b[0])), tuple(x + s * y for x, y in zip(a[1], b[1])))
shape, tr, ttr, order, Wn = (4, 5, 6), (2, 3, 2), (1, 2, 2, 1), 2, 5
d = 3
th = t3k.t3_corewise_randn(shape, tr, ttr)
cf = fvc.t3_corewise_frame(th)
thd = (tuple(rng.standard_normal(u.shape) for u in th[0]), tuple(rng.standard_normal(g.shape) for g in th[1]))
ww = [rng.standard_normal((Wn, N)) for N in shape]; pp = [rng.standard_normal((Wn, N)) for N in shape]
idx = np.stack([rng.integers(0, N, Wn) for N in shape])
c = rng.standard_normal((order + 1, Wn))
zt = [rng.standard_normal((order + 1, Wn, N)) for N in shape]
cases = {
    'apply_derivatives':   lambda cores: sd.t3_apply_corewise_derivatives_transpose(c, ww, pp, cores, order, sum_over_probes=True),
    'entries_derivatives': lambda cores: sd.t3_entries_corewise_derivatives_transpose(c, idx, pp, cores, order, sum_over_probes=True),
    'probe_derivatives':   lambda cores: sd.t3_probe_corewise_derivatives_transpose(zt, ww, pp, cores, order, sum_over_probes=True, chunk_size=None),
    'entries':             lambda cores: en.t3_entries_corewise_transpose(c[0], idx, cores, sum_over_probes=True),
}
h = 1e-5
for name, f in cases.items():
    bs = flat(blocksum(f(tv.tv_to_t3(cf, thd)), d, tr, ttr))
    fd = (flat(f(add(th, thd, h))) - flat(f(add(th, thd, -h)))) / (2 * h)
    print('%-20s block-sum vs FD rel err %.1e' % (name, np.linalg.norm(bs - fd) / np.linalg.norm(fd)))
