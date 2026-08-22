"""newton_cg numpy vs jax+jit (x64) on edge structures, ragged and uniform, all kinds."""
import numpy as np, jax
jax.config.update("jax_enable_x64", True)
import t3toolbox as tb
from t3toolbox import TuckerTensorTrain as T3, MANIFOLD, COREWISE
from t3toolbox.uniform_tucker_tensor_train import UniformTuckerTensorTrain as UT3
from t3toolbox.uniform_manifold import UNIFORM_MANIFOLD as UM
import t3toolbox.optimizers as opt
STRUCTS = {'d1': ((5,), (2,), (1, 1)), 'd2_rank1': ((4, 6), (1, 1), (1, 1, 1)), 'd3_mode1': ((1, 5, 6), (1, 2, 3), (1, 2, 2, 1)),
           'd3_nonmin': ((4, 5, 6), (2, 2, 2), (1, 4, 4, 1)), 'd3_square': ((3, 4, 5), (3, 4, 5), (1, 2, 3, 1))}
for sname, struct in STRUCTS.items():
    np.random.seed(0)
    xt = T3.randn(*struct)
    Wn = 12
    ww = tuple(np.random.randn(Wn, N) for N in struct[0]); pp = tuple(np.random.randn(Wn, N) for N in struct[0])
    idx = np.stack([np.random.randint(0, N, size=Wn) for N in struct[0]])
    x0 = T3.randn(*struct) * 0.1
    kinds = {'probe': (ww, xt.probe(ww), {}), 'apply': (ww, xt.apply(ww), {}), 'entries': (idx, xt.entries(idx), {}),
             'probe_derivatives': ((ww, pp), xt.probe_derivatives(ww, pp, 1), dict(order=1)),
             'apply_derivatives': ((ww, pp), xt.apply_derivatives(ww, pp, 1), dict(order=1)),
             'entries_derivatives': ((idx, pp), xt.entries_derivatives(idx, pp, 1), dict(order=1))}
    for kname, (sample, data, kw) in kinds.items():
        for gname, geom, x0g in [('MANIFOLD', MANIFOLD, x0), ('COREWISE', COREWISE, x0)] + ([('UNIFORM_MANIFOLD', UM, UT3.from_t3(x0))] if sname != 'd1' else []):
            out = {}
            for jit in (False, True):
                try:
                    with tb.safety.unsafe():
                        xf, st = opt.newton_cg(geom, kname, sample, data, x0g, max_newton=3, use_jit=jit, **kw)
                    out[jit] = ('ok', [float(h) for h in np.asarray(st['history']['objective'] if isinstance(st['history'], dict) else [e['objective'] if isinstance(e, dict) else getattr(e, 'objective', e) for e in st['history']])][:4])
                except Exception as e:
                    out[jit] = ('err', '%s: %s' % (type(e).__name__, (str(e).splitlines() or [''])[0][:110]))
            if out[False][0] == 'ok' and out[True][0] == 'ok':
                a, b = np.array(out[False][1]), np.array(out[True][1])
                rel = np.max(np.abs(a - b) / np.maximum(np.abs(a), 1e-300)) if a.shape == b.shape else np.inf
                tag = 'ok' if rel < 1e-6 else 'MISMATCH rel=%.2e np=%s jit=%s' % (rel, a, b)
            else:
                tag = 'np: %s | jit: %s' % (out[False], out[True])
            if tag != 'ok':
                print('%-10s %-20s %-17s %s' % (sname, kname, gname, tag), flush=True)
print('DONE')
