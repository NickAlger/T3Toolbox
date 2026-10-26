"""Read-only plan check: the plain-apply two-hole from EXISTING kernels (compute_dxi, compute_sigma,
compute_tau, compute_sigma_hat) + three-term assemblies, vs the block-sum oracle; with a K stack."""
import sys; sys.path.insert(0, __import__("os").getcwd())
import numpy as np
import t3toolbox.backend.fv_conversions as fvc
import t3toolbox.backend.tv_operations as tv
import t3toolbox.backend.apply as ap
import t3toolbox.backend.probing as pr
import t3toolbox.backend.contractions as ct
import t3toolbox.backend.t3_constructors as t3k
rng = np.random.default_rng(5)

def blocksum(gY, d, tr, ttr):
    gYU, gYG = gY
    gU = tuple(gYU[i][..., :tr[i], :] for i in range(d))
    gG = []
    for i in range(d):
        r, n, rn = ttr[i], tr[i], ttr[i + 1]
        g = gYG[i]
        if i == 0:   gG.append(g[..., :, :n, rn:] + g[..., :, n:, :rn])
        elif i == d - 1: gG.append(g[..., :r, :n, :] + g[..., r:, n:, :])
        else:        gG.append(g[..., :r, :n, :rn] + g[..., r:, :n, rn:] + g[..., r:, n:, :rn])
    return gU, tuple(gG)

shape, tr, ttr = (5, 6, 6, 5), (2, 3, 3, 2), (1, 2, 3, 2, 1)
d = 4
x = t3k.t3_corewise_randn(shape, tr, ttr)
frame, _ = fvc.t3_orthogonal_representations(x)
U, O, L, R = frame
theta = (tuple(U), tuple(L))
cframe = fvc.t3_corewise_frame(theta)
Wn, Kn = 9, 2
ww = [rng.standard_normal((Wn, N)) for N in shape]
c = rng.standard_normal((Wn,))
th_hat = (tuple(rng.standard_normal((Kn,) + u.shape) for u in U), tuple(rng.standard_normal((Kn,) + g.shape) for g in L))

# --- two-hole from existing kernels ---
xis, mus = ap.tv_precompute_apply_frame_sweep(cframe, ww)   # the lean sweep, same as at the manifold frame
xis_m, mus_m = ap.tv_precompute_apply_frame_sweep(frame, ww)
print('lean sweep identical at manifold vs corewise frame:',
      all(np.allclose(a, b) for a, b in zip(xis, xis_m)), all(np.allclose(a, b) for a, b in zip(mus, mus_m)))
sh = pr.compute_sigma_hat(L, xis, c)                          # residual-seeded adjoint through L (the curvature sweep)
Ud, Gd = th_hat
dxis = pr.compute_dxi(Ud, ww)                                  # K+... : W? check layout
print('dxi shape', dxis[0].shape, 'xi shape', xis[0].shape, 'mu', mus[0].shape, 'sh', sh[0].shape)
mud = pr.compute_sigma(Gd, L, L, xis, dxis, mus)               # mu-dot
shd = pr.compute_tau(Gd, L, L, xis, dxis, sh)                  # sigma-hat-dot
print('mu-dot', mud[0].shape, 'sh-dot', shd[0].shape)
dU, dG = [], []
for k in range(d):
    G = L[k]
    a1 = ct.contract('WKCa,WCi,WCb->KCaib', mud[k], xis[k], sh[k], len_W=1)
    a2 = ct.contract('WCa,WKCi,WCb->KCaib', mus[k], dxis[k], sh[k], len_W=1)
    a3 = ct.contract('WCa,WCi,WKCb->KCaib', mus[k], xis[k], shd[k], len_W=1)
    dG.append(a1 + a2 + a3)
    x1 = ct.contract('WKCa,Caib,WCb->WKCi', mud[k], G, sh[k])
    x2 = ct.contract('WCa,KCaib,WCb->WKCi', mus[k], Gd[k], sh[k], len_C=0)
    x3 = ct.contract('WCa,Caib,WKCb->WKCi', mus[k], G, shd[k])
    dU.append(ct.contract('Wo,WKCa->KCao', ww[k], x1 + x2 + x3))
two_hole = (tuple(dU), tuple(dG))

# --- block-sum oracle, per K element ---
errs = []
for kk in range(Kn):
    th_k = (tuple(a[kk] for a in Ud), tuple(a[kk] for a in Gd))
    ref = blocksum(ap.t3_apply_corewise_transpose(c, ww, tv.tv_to_t3(cframe, th_k), sum_over_probes=True), d, tr, ttr)
    num = sum(np.linalg.norm(a[kk] - b) ** 2 for a, b in zip(two_hole[0] + two_hole[1], ref[0] + ref[1])) ** 0.5
    den = sum(np.linalg.norm(b) ** 2 for b in ref[0] + ref[1]) ** 0.5
    errs.append(num / den)
print('apply two-hole (existing kernels) vs block-sum oracle, per K:', ['%.1e' % e for e in errs])
