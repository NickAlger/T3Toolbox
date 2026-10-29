"""Critic check (read-only): c14's two_hole at order 0 (claimed but not run), and K=(2,3) / C=(2,) order 1."""
import sys; sys.path.insert(0, __import__("os").getcwd())
sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "..", "checks"))
import numpy as np
import c14_probe_derivs_twohole as c14
import t3toolbox.backend.fv_conversions as fvc
import t3toolbox.backend.tv_operations as tv
import t3toolbox.backend.sampling_derivatives as sd
import t3toolbox.backend.t3_constructors as t3k
rng = np.random.default_rng(5); np.random.seed(5)
for C, order, K in [((), 0, (2,)), ((2,), 0, (2,)), ((2,), 1, (2, 3))]:
    shape, tr, ttr, Wn = (4, 5, 6, 4), (2, 3, 3, 2), (1, 2, 3, 2, 1), 5
    d = 4
    th = t3k.t3_corewise_randn(shape, tr, ttr, stack_shape=C)
    U, G = th
    cf = fvc.t3_corewise_frame(th)
    ww = [rng.standard_normal((Wn, N)) for N in shape]; pp = [rng.standard_normal((Wn, N)) for N in shape]
    zt = [rng.standard_normal((order + 1, Wn) + C + (N,)) for N in shape]
    Ud = tuple(rng.standard_normal(K + u.shape) for u in U); Gd = tuple(rng.standard_normal(K + g.shape) for g in G)
    try:
        dU, dG = c14.two_hole(U, G, Ud, Gd, ww, pp, zt, order, K, C)
    except Exception as e:
        print('C=%s order=%d K=%s: two_hole raised %s: %s' % (C, order, K, type(e).__name__, str(e)[:150])); continue
    errs = []
    for idx in np.ndindex(*K):
        thk = (tuple(a[idx] for a in Ud), tuple(a[idx] for a in Gd))
        ref = c14.blocksum(sd.t3_probe_corewise_derivatives_transpose(zt, ww, pp, tv.tv_to_t3(cf, thk), order,
                                                                      sum_over_probes=True, chunk_size=None), d, tr, ttr)
        errs.append(c14.relerr([a[idx] for a in dU + dG], list(ref[0] + ref[1])))
    print('C=%s order=%d K=%s: max rel err vs block-sum %.1e' % (C, order, K, max(errs)))
