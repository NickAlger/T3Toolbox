"""Read-only plan check: the apply_derivatives two-hole (two-index jet: vector order t <= order, core order 1)
composed from EXISTING jet kernels (build_input_jets, compute_sigma_jets at the corewise frame) + ONE new
3-term adjoint-wired scan step (sigma_hat_dot) + the dotted assemblies; vs the block-sum oracle. K and C stacked."""
import sys; sys.path.insert(0, __import__("os").getcwd())
import numpy as np
import t3toolbox.backend.fv_conversions as fvc
import t3toolbox.backend.tv_operations as tv
import t3toolbox.backend.sampling_derivatives as sd
import t3toolbox.backend.probing as pr
import t3toolbox.backend.contractions as ct
import t3toolbox.backend.t3_constructors as t3k
rng = np.random.default_rng(17)
def blocksum(gY, d, tr, ttr):
    gYU, gYG = gY
    gU = tuple(gYU[i][..., :tr[i], :] for i in range(d)); gG = []
    for i in range(d):
        r, n, rn = ttr[i], tr[i], ttr[i + 1]; g = gYG[i]
        if i == 0:   gG.append(g[..., :, :n, rn:] + g[..., :, n:, :rn])
        elif i == d - 1: gG.append(g[..., :r, :n, :] + g[..., r:, n:, :])
        else:        gG.append(g[..., :r, :n, :rn] + g[..., r:, :n, rn:] + g[..., r:, n:, :rn])
    return gU, tuple(gG)
def relerr(a, b):
    fa = np.concatenate([x.ravel() for x in a]); fb = np.concatenate([x.ravel() for x in b])
    return np.linalg.norm(fa - fb) / np.linalg.norm(fb)
for C in [(), (2,)]:
    shape, tr, ttr, order, Wn, Kn = (4, 5, 6, 4), (2, 3, 3, 2), (1, 2, 3, 2, 1), 2, 5, 2
    d = 4
    th = t3k.t3_corewise_randn(shape, tr, ttr, stack_shape=C)
    U, G = th
    cf = fvc.t3_corewise_frame(th)
    ww = [rng.standard_normal((Wn, N)) for N in shape]; pp = [rng.standard_normal((Wn, N)) for N in shape]
    c = rng.standard_normal((order + 1, Wn) + C)
    Ud = tuple(rng.standard_normal((Kn,) + u.shape) for u in U); Gd = tuple(rng.standard_normal((Kn,) + g.shape) for g in G)
    trs = sd.binomial_combine_tensor(order); s2 = min(2, order + 1)
    xi_j, mu_j = sd.tv_precompute_apply_frame_sweep_jets(cf, ww, pp, order)          # reused (== manifold-frame sweep)
    sh_j = sd.compute_sigma_hat_jets(G, xi_j, c, trs)                                  # curvature precompute (Q -> P = G)
    xid_j = sd.build_input_jets(pr.compute_xi(Ud, ww), pr.compute_xi(Ud, pp))          # (2,)+W+K+C
    mud_j = sd.compute_sigma_jets(Gd, G, G, xi_j, xid_j, mu_j, trs)                    # EXISTING kernel = mu-dot jets
    # NEW: sigma_hat-dot jets, reverse sweep, 3 terms (K on carry-dot / G-dot / xi-dot), adjoint wiring
    trs_xi = trs[:, :s2, :]
    shd = [None] * d
    carry = np.zeros((order + 1, Wn, Kn) + C + (G[-1].shape[-1],))
    for k in range(d - 1, -1, -1):
        shd[k] = carry
        Gr, Gdr = np.swapaxes(G[k], -1, -3), np.swapaxes(Gd[k], -1, -3)      # reversed core (the tt_reverse orientation)
        carry = (ct.contract('trs,tWKCa,Caib,rWCi->sWKCb', trs_xi, carry, Gr, xi_j[k][:s2])
                 + ct.contract('trs,tWCa,KCaib,rWCi->sWKCb', trs_xi, sh_j[k], Gdr, xi_j[k][:s2], len_W=1)
                 + ct.contract('trs,tWCa,Caib,rWKCi->sWKCb', trs_xi, sh_j[k], Gr, xid_j[k][:s2]))
    dU, dG = [], []
    for k in range(d):
        dG.append(ct.contract('trs,rWKCa,sWCi,tWCb->KCaib', trs[:, :, :s2], mud_j[k], xi_j[k][:s2], sh_j[k], len_W=1)
                  + ct.contract('trs,rWCa,sWKCi,tWCb->KCaib', trs[:, :, :s2], mu_j[k], xid_j[k][:s2], sh_j[k], len_W=1)
                  + ct.contract('trs,rWCa,sWCi,tWKCb->KCaib', trs[:, :, :s2], mu_j[k], xi_j[k][:s2], shd[k], len_W=1))
        dxh = (ct.contract('trs,rWKCa,Caib,tWCb->sWKCi', trs, mud_j[k], G[k], sh_j[k])
               + ct.contract('trs,rWCa,KCaib,tWCb->sWKCi', trs, mu_j[k], Gd[k], sh_j[k], len_W=1)
               + ct.contract('trs,rWCa,Caib,tWKCb->sWKCi', trs, mu_j[k], G[k], shd[k]))
        wj = np.stack([ww[k], pp[k]])
        dU.append(ct.contract('uWKCa,uWo->KCao', dxh[:s2], wj[:s2]))
    errs = []
    for kk in range(Kn):
        thk = (tuple(a[kk] for a in Ud), tuple(a[kk] for a in Gd))
        ref = blocksum(sd.t3_apply_corewise_derivatives_transpose(c, ww, pp, tv.tv_to_t3(cf, thk), order, sum_over_probes=True), d, tr, ttr)
        errs.append(relerr([a[kk] for a in dU + dG], list(ref[0] + ref[1])))
    print('C=%s apply_derivatives two-hole (order=%d) vs block-sum, per K:' % (C, order), ['%.1e' % e for e in errs])
