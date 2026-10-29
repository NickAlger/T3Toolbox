"""Critic check (read-only): c14's probe_derivatives two-hole at d = 1 (COREWISE d = 1 runs exact per I-12)."""
import sys; sys.path.insert(0, __import__("os").getcwd())
sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "..", "checks"))
import numpy as np
import c14_probe_derivs_twohole as c14
import t3toolbox.backend.fv_conversions as fvc
import t3toolbox.backend.tv_operations as tv
import t3toolbox.backend.sampling_derivatives as sd
import t3toolbox.backend.t3_constructors as t3k
rng = np.random.default_rng(9); np.random.seed(9)
for ttr in ((1, 1), (2, 3)):
    shape, tr, order, Wn, K, C = (6,), (3,), 2, 5, (2,), ()
    th = t3k.t3_corewise_randn(shape, tr, ttr)
    U, G = th; cf = fvc.t3_corewise_frame(th)
    ww = [rng.standard_normal((Wn, N)) for N in shape]; pp = [rng.standard_normal((Wn, N)) for N in shape]
    zt = [rng.standard_normal((order + 1, Wn) + C + (N,)) for N in shape]
    Ud = tuple(rng.standard_normal(K + u.shape) for u in U); Gd = tuple(rng.standard_normal(K + g.shape) for g in G)
    try:
        dU, dG = c14.two_hole(U, G, Ud, Gd, ww, pp, zt, order, K, C)
    except Exception as e:
        print('d=1 tt=%s: two_hole raised %s: %s' % (ttr, type(e).__name__, str(e)[:160])); continue
    errs = []
    for kk in range(K[0]):
        thk = (tuple(a[kk] for a in Ud), tuple(a[kk] for a in Gd))
        Y = tv.tv_to_t3(cf, thk)
        ref = sd.t3_probe_corewise_derivatives_transpose(zt, ww, pp, Y, order, sum_over_probes=True, chunk_size=None)
        n = tr[0]; r0, r1 = ttr
        refU = ref[0][0][..., :n, :]
        g = ref[1][0]
        refG = g[:, n:, :]          # d = 1 embedding: Y_G = [Gdot | G] along n, Y_U = [U; Udot]
        errs.append(c14.relerr([dU[0][kk], dG[0][kk]], [refU, refG]))
    print('d=1 tt=%s: probe_derivatives two-hole vs block-sum: max %.1e  (|ref U| %.2e, |ref G| %.2e)' % (ttr, max(errs), np.linalg.norm(refU), np.linalg.norm(refG)))
