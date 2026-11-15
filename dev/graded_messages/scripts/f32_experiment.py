"""float32 accuracy: derivative-normalized jets (the library) vs Taylor-normalized jets (prototype),
each measured against its own float64 run. Usage: python3 f32_experiment.py <repo>"""
import pathlib, sys, math, types
import numpy as np
repo = sys.argv[1]
sys.path.append(repo)
HERE = pathlib.Path(__file__).resolve().parent
gm = types.ModuleType('gm'); exec(compile((HERE / 'graded_messages_prototype.py').read_text(), 'gm', 'exec'), gm.__dict__)
import t3toolbox.backend.sampling_derivatives as sd

def fact(P, like):
    return np.array([math.factorial(t) for t in range(P)], dtype=like.dtype).reshape((P,) + (1,) * (like.ndim - 1))

def setup(seed, scale, d):
    rng = np.random.default_rng(seed); rn = rng.standard_normal
    Ns, n, r = (6,) * d, (4,) * d, (1,) + (4,) * (d - 1) + (1,)
    W = 32
    B = [rn((n[i], Ns[i])) / math.sqrt(Ns[i]) for i in range(d)]
    G = [rn((r[i], n[i], r[i + 1])) / math.sqrt(n[i]) for i in range(d)]
    ww = [rn((W, N)) for N in Ns]
    pp = [scale * rn((W, N)) for N in Ns]          # |p| / |w| sets how fast the orders grow
    return B, G, ww, pp, Ns, d, W, rng

def run(order, scale, d, seed=0):
    P = order + 1
    B, G, ww, pp, Ns, d, W, rng = setup(seed, scale, d)
    zt = [rng.standard_normal((P, W, N)) for N in Ns]            # residual jets (derivative-normalized)
    c = lambda xs, dt: [x.astype(dt) for x in xs]
    res = {}
    for dt in (np.float64, np.float32):
        Bx, Gx, wx, px, ztx = c(B, dt), c(G, dt), c(ww, dt), c(pp, dt), c(zt, dt)
        # library: derivative-normalized
        lib_fwd = sd.t3_probe_derivatives(wx, px, (Bx, Gx), order)
        lib_T = sd.t3_probe_corewise_derivatives_transpose(ztx, wx, px, (Bx, Gx), order, sum_over_probes=True,
                                                           chunk_size=None)
        # prototype: Taylor-normalized, reversed residual jets
        bnd = {'': gm.jet(P, np.ones((W, 1), dt))}
        leaves = [{'': gm.jet(P, wx[i], px[i]), 'z': (fact(P, ztx[i]) * ztx[i])[::-1]} for i in range(d)]
        out = gm.run(*gm.corewise((Bx, Gx)), leaves, bnd, bnd, P, grad=('z',))
        tay_fwd = [z[''] for z in out['Z']]                       # Taylor coefficients, NOT rescaled
        tay_T = [x['z'][-1].sum(axis=0) for x in out['dB']] + [x['z'][-1].sum(axis=0) for x in out['dG']]
        assert all(a.dtype == dt for a in list(lib_fwd) + tay_fwd + tay_T + list(lib_T[0]) + list(lib_T[1]))
        res[dt] = (lib_fwd, list(lib_T[0]) + list(lib_T[1]), tay_fwd, tay_T)
    rel = lambda a, b: np.linalg.norm(a.astype(np.float64) - b) / np.linalg.norm(b)
    (lf64, lT64, tf64, tT64), (lf32, lT32, tf32, tT32) = res[np.float64], res[np.float32]
    per_order_lib = [max(rel(a[t], b[t]) for a, b in zip(lf32, lf64)) for t in range(P)]
    per_order_tay = [max(rel(a[t], b[t]) for a, b in zip(tf32, tf64)) for t in range(P)]
    mag = [max(float(np.abs(a[t]).max()) for a in lf64) for t in range(P)]
    magT = [max(float(np.abs(a[t]).max()) for a in tf64) for t in range(P)]
    return per_order_lib, per_order_tay, max(rel(a, b) for a, b in zip(lT32, lT64)), \
        max(rel(a, b) for a, b in zip(tT32, tT64)), mag, magT, max(rel(a, b) for a, b in zip(tT64, lT64))

for order, scale, d in ((3, 1.0, 4), (8, 1.0, 12), (12, 1.0, 16), (12, 3.0, 16), (20, 1.0, 24)):
    with np.errstate(all='ignore'):
        pl, pt, Tl, Tt, mag, magT, agree = run(order, scale, d)
    print('d = %d, order %d, |p|/|w| = %g   (float64 transposes agree across normalizations: %.0e)' % (d, order, scale, agree))
    print('   t   max|y^(t)|  max|yhat_t|   f32 err: derivative-norm   Taylor-norm')
    for t in sorted(set([0, 1, 2, order // 2, order - 1, order])):
        print('  %2d   %9.2e   %9.2e                 %9.1e    %9.1e' % (t, mag[t], magT[t], pl[t], pt[t]))
    print('   transpose (gradient) f32 err:              %9.1e    %9.1e' % (Tl, Tt))
