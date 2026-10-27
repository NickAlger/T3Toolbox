"""Read-only plan check: (a) probe two-hole built from EXISTING kernels (_sigma_step x2 for the dotted
tau_tilde, compute_deta x2 for the dotted dxi_tilde, existing contract strings for the 9+2-term
assemblies, ONE new string for deta_tilde-dot); (b) T3-T3 two-hole with '...' einsums; both vs the
block-sum oracle, with a K stack and a C stack."""
import sys; sys.path.insert(0, __import__("os").getcwd())
import numpy as np
import t3toolbox.backend.fv_conversions as fvc
import t3toolbox.backend.tt_operations as tto
import t3toolbox.backend.tv_operations as tv; import t3toolbox.backend.t3_linalg as t3_linalg
import t3toolbox.backend.probing as pr
import t3toolbox.backend.contractions as ct
import t3toolbox.backend.t3_constructors as t3k
from t3toolbox.backend.common import get_backend
rng = np.random.default_rng(7)

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

def relerr(a, b):
    fa = np.concatenate([x.ravel() for x in a[0] + a[1]]); fb = np.concatenate([x.ravel() for x in b[0] + b[1]])
    return np.linalg.norm(fa - fb) / np.linalg.norm(fb)

def ungauged_inner_grad(th, z):
    up, down, left, right = fvc.t3_corewise_frame(th)
    ztk, ztt = z
    ztt = tto.tt_squash_tails(ztt)
    _, xmap, _ = get_backend(False, False)
    (o2,) = xmap(t3_linalg._inner_product_rebasis_step, (ztt, ztk, up))
    zl = tto.tt_zipper_left_to_right(o2[:-1], left[:-1])
    zr = tto.tt_zipper_right_to_left(o2[1:], right[1:])
    dG, dB = xmap(t3_linalg._inner_product_variations_step, (zl, zr, ztt, ztk, down, up))
    return (tuple(dB), tuple(dG))

for C in [(), (2,)]:
    shape, tr, ttr = (5, 6, 6, 5), (2, 3, 3, 2), (1, 2, 3, 2, 1)
    d = 4
    Kn, Wn = 3, 6
    th = t3k.t3_corewise_randn(shape, tr, ttr, stack_shape=C)
    U, G = th
    cframe = fvc.t3_corewise_frame(th)
    ww = [rng.standard_normal((Wn, N)) for N in shape]
    zt = [rng.standard_normal((Wn,) + C + (N,)) for N in shape]          # residual (omega^2 r), W+C+(N,)
    Ud = tuple(rng.standard_normal((Kn,) + u.shape) for u in U)
    Gd = tuple(rng.standard_normal((Kn,) + g.shape) for g in G)
    # ---- (a) probe two-hole --------------------------------------------------------------
    xis, mus, nus, etas = pr.tv_precompute_probe_frame_sweep(cframe, ww)          # corewise sweep (nu, eta through G)
    dets = pr.compute_deta_tilde(U, zt)                                          # curvature precompute (residual-dependent)
    tts = pr.compute_tau_tilde(dets, G, xis, mus)
    sts = pr.compute_sigma_tilde(dets, G, xis, nus)
    xid = pr.compute_dxi(Ud, ww)
    mud = pr.compute_sigma(Gd, G, G, xis, xid, mus)
    nud = pr.compute_tau(Gd, G, G, xis, xid, nus)
    etad = pr.compute_deta(Gd, G, G, mus, nus, mud, nud)
    nC = len(C)
    detd = [ct.contract('WCo,KCio->WKCi', z, u, len_C=nC) for z, u in zip(zt, Ud)]
    # dotted tau_tilde: carry recursion = two _sigma_step calls (Leibniz on the two trilinear terms)
    ttd = []; carry = np.zeros(detd[0].shape[:-1] + (G[0].shape[-3],))
    for k in range(d):
        ttd.append(carry)
        carry = pr._sigma_step(carry, G[k], G[k], Gd[k], xis[k], xid[k], tts[k]) \
              + pr._sigma_step(mud[k], G[k], G[k], Gd[k], dets[k], detd[k], mus[k])
    # dotted sigma_tilde: the same on the reversed train
    Gr, Gdr = tto.tt_reverse(G), tto.tt_reverse(Gd)
    std_r = []; carry = np.zeros(detd[0].shape[:-1] + (G[-1].shape[-1],))
    xr, xdr, str_, nur, nudr, detr, detdr = xis[::-1], xid[::-1], sts[::-1], nus[::-1], nud[::-1], dets[::-1], detd[::-1]
    for k in range(d):
        std_r.append(carry)
        carry = pr._sigma_step(carry, Gr[k], Gr[k], Gdr[k], xr[k], xdr[k], str_[k]) \
              + pr._sigma_step(nudr[k], Gr[k], Gr[k], Gdr[k], detr[k], detdr[k], nur[k])
    std = std_r[::-1]
    # dotted dxi_tilde: two compute_deta calls
    dxtd = [a + b for a, b in zip(pr.compute_deta(Gd, G, G, tts, nus, ttd, nud),
                                  pr.compute_deta(Gd, G, G, mus, sts, mud, std))]
    dU = tuple(ct.contract('WCo,WKCa->KCao', z, e, len_W=1) + ct.contract('Wo,WKCa->KCao', w, x)
               for z, e, w, x in zip(zt, etad, ww, dxtd))
    def asm(mu, xi, nu, mud_, xid_, nud_):     # 9-term product rule of mu(x)xi(x)nu over the three legs
        return (ct.contract('WKCi,WCa,WCj->KCiaj', mud_, xi, nu, len_W=1)
                + ct.contract('WCi,WKCa,WCj->KCiaj', mu, xid_, nu, len_W=1)
                + ct.contract('WCi,WCa,WKCj->KCiaj', mu, xi, nud_, len_W=1))
    dG = tuple(asm(mus[k], xis[k], sts[k], mud[k], xid[k], std[k])
               + asm(tts[k], xis[k], nus[k], ttd[k], xid[k], nud[k])
               + asm(mus[k], dets[k], nus[k], mud[k], detd[k], nud[k]) for k in range(d))
    # block-sum oracle per (K, C) element
    errs = []
    for kk in range(Kn):
        thk = (tuple(a[kk] for a in Ud), tuple(a[kk] for a in Gd))
        Y = tv.tv_to_t3(cframe, thk)
        ref = blocksum(pr.t3_probe_corewise_transpose(zt, ww, Y, sum_over_probes=True), d, tr, ttr)
        errs.append(relerr((tuple(a[kk] for a in dU), tuple(a[kk] for a in dG)), ref))
    print('C=%s probe two-hole (existing kernels) vs block-sum, per K:' % (C,), ['%.1e' % e for e in errs])

    # ---- (b) T3-T3 two-hole ('...' machinery: theta_hat K+C, Z and theta C) -----------------
    Z = t3k.t3_corewise_randn(shape, (3, 4, 4, 3), (1, 3, 4, 3, 1), stack_shape=C)   # stands in for tv_to_t3(frame, g)
    ZB, ZG = Z
    e = np.einsum
    BU  = [e('...iz,...xz->...ix', b, u) for b, u in zip(ZB, U)]
    BUd = [e('...iz,...xz->...ix', b, u) for b, u in zip(ZB, Ud)]
    G2  = [e('...aib,...ix->...axb', g, m) for g, m in zip(ZG, BU)]
    G2d = [e('...aib,...ix->...axb', g, m) for g, m in zip(ZG, BUd)]
    ZL = tto.tt_zipper_left_to_right(G2[:-1], G[:-1]); ZR = tto.tt_zipper_right_to_left(G2[1:], G[1:])
    zip3 = lambda Zd, Z0, A, Ad, B, Bd: (e('...ij,...iak,...jal->...kl', Zd, A, B)
                                         + e('...ij,...iak,...jal->...kl', Z0, Ad, B)
                                         + e('...ij,...iak,...jal->...kl', Z0, A, Bd))
    ZLd = [np.zeros((Kn,) + ZL[0].shape)]
    for k in range(d - 1):
        ZLd.append(zip3(ZLd[k], ZL[k], G2[k], G2d[k], G[k], Gd[k]))
    ZRd = [None] * d; ZRd[d - 1] = np.zeros((Kn,) + ZR[-1].shape)
    for k in range(d - 1, 0, -1):     # right-to-left: reversed cores
        rev = lambda A: np.swapaxes(A, -1, -3)
        ZRd[k - 1] = zip3(ZRd[k], ZR[k], rev(G2[k]), rev(G2d[k]), rev(G[k]), rev(Gd[k]))
    env  = [e('...ax,...aib,...by->...xiy', ZL[k], ZG[k], ZR[k]) for k in range(d)]
    envd = [e('...ax,...aib,...by->...xiy', ZLd[k], ZG[k], ZR[k]) + e('...ax,...aib,...by->...xiy', ZL[k], ZG[k], ZRd[k]) for k in range(d)]
    dGt = tuple(e('...xiy,...ij->...xjy', envd[k], BU[k]) + e('...xiy,...ij->...xjy', env[k], BUd[k]) for k in range(d))
    dBt = tuple(e('...ij,...io->...jo', e('...xiy,...xjy->...ij', envd[k], G[k]) + e('...xiy,...xjy->...ij', env[k], Gd[k]), ZB[k]) for k in range(d))
    errs = []
    for kk in range(Kn):
        thk = (tuple(a[kk] for a in Ud), tuple(a[kk] for a in Gd))
        ref = blocksum(ungauged_inner_grad(tv.tv_to_t3(cframe, thk), Z), d, tr, ttr)
        errs.append(relerr((tuple(a[kk] for a in dBt), tuple(a[kk] for a in dGt)), ref))
    print('C=%s T3-T3 two-hole (dotted zippers) vs block-sum, per K:' % (C,), ['%.1e' % x for x in errs])
