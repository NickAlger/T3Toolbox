"""Read-only plan check (v2, the PRIMARY target): the probe_derivatives two-hole (two-index jet:
vector order t <= order, core order <= 1), ragged, K-stacked and C-stacked, vs the block-sum oracle.

Composition: undotted sweep + tildes at the corewise frame come from EXISTING kernels
(tv_precompute_probe_frame_sweep_jets, compute_{deta,tau,sigma}_tilde_jets). The dotted forward
uses EXISTING kernels (build_input_jets, compute_sigma_jets / compute_tau_jets / compute_deta_jets at
Q = O = P = G). New: the dotted adjoint sweep (6 terms = product rule of _adj_sweep_step's prop + src),
the dotted dxi_tilde (6 terms), the dotted assemblies (Tucker 2 + 2, TT 9). Each new term is the trs
wiring of an existing term with K moved onto the dotted operand -- written here as plain contract calls."""
import sys; sys.path.insert(0, __import__("os").getcwd())
import numpy as np
import t3toolbox.backend.fv_conversions as fvc
import t3toolbox.backend.tv_operations as tv
import t3toolbox.backend.tt_operations as tto
import t3toolbox.backend.sampling_derivatives as sd
import t3toolbox.backend.probing as pr
import t3toolbox.backend.t3_constructors as t3k
from t3toolbox.backend.contractions import contract

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

def dotted_adj_sweep(P, Pd, xi, xid, deta, detd, edge, edged, carries, trs, K, C, W):
    """d/ds of _adj_sweep: carry' = prop(carry, P, xi) + src(edge, P, deta); 6 dotted terms."""
    s2 = min(2, trs.shape[0]); trs_xi = trs[:, :s2, :]
    kw = dict(len_W=len(W), len_C=len(C))
    out = []
    cd = np.zeros((trs.shape[0],) + W + K + C + (P[0].shape[-3],))
    for k in range(len(P)):
        out.append(cd)
        cd = (contract('trs,tWKCa,Caib,rWCi->sWKCb', trs_xi, cd,         P[k],  xi[k][:s2],  **kw)
              + contract('trs,tWCa,KCaib,rWCi->sWKCb', trs_xi, carries[k], Pd[k], xi[k][:s2],  **kw)
              + contract('trs,tWCa,Caib,rWKCi->sWKCb', trs_xi, carries[k], P[k],  xid[k][:s2], **kw)
              + contract('trs,rWKCa,Caib,tWCi->sWKCb', trs, edged[k], P[k],  deta[k], **kw)
              + contract('trs,rWCa,KCaib,tWCi->sWKCb', trs, edge[k],  Pd[k], deta[k], **kw)
              + contract('trs,rWCa,Caib,tWKCi->sWKCb', trs, edge[k],  P[k],  detd[k], **kw))
    return out

def two_hole(U, G, Ud, Gd, ww, pp, zt, order, K, C):
    """The probe_derivatives two-hole K_r(theta_hat) at theta = (U, G); theta_hat = (Ud, Gd) carries K."""
    d = len(U); W = tuple(ww[0].shape[:-1])
    cf = fvc.t3_corewise_frame((U, G))
    trs = sd.binomial_combine_tensor(order); s2 = min(2, order + 1)
    kw = dict(len_W=1, len_C=len(C))
    # undotted, corewise frame (existing kernels)
    xi_j, mu_j, nu_j, eta_j = sd.tv_precompute_probe_frame_sweep_jets(cf, ww, pp, order)
    dets = sd.compute_deta_tilde_jets(U, zt)
    tts = sd.compute_tau_tilde_jets(G, xi_j, dets, mu_j, trs)
    sts = sd.compute_sigma_tilde_jets(G, xi_j, dets, nu_j, trs)
    dxts = sd.compute_dxi_tilde_jets(G, mu_j, nu_j, sts, tts, trs)
    # dotted forward (existing kernels at Q = O = P = G)
    xid_j = sd.build_input_jets(pr.compute_xi(Ud, ww), pr.compute_xi(Ud, pp))
    mud_j = sd.compute_sigma_jets(Gd, G, G, xi_j, xid_j, mu_j, trs)
    nud_j = sd.compute_tau_jets(Gd, G, G, xi_j, xid_j, nu_j, trs)
    etad_j = sd.compute_deta_jets(Gd, G, G, mu_j, nu_j, mud_j, nud_j, trs)
    # dotted adjoint-up lift (one new string)
    detd = [contract('tWCo,KCio->tWKCi', z, u, **kw) for z, u in zip(zt, Ud)]
    # dotted adjoint sweeps (new 6-term step), sigma by reversal
    ttd = dotted_adj_sweep(G, Gd, xi_j, xid_j, dets, detd, mu_j, mud_j, tts, trs, K, C, W)
    rv = tto.tt_reverse
    std = dotted_adj_sweep(rv(G), rv(Gd), xi_j[::-1], xid_j[::-1], dets[::-1], detd[::-1],
                           nu_j[::-1], nud_j[::-1], sts[::-1], trs, K, C, W)[::-1]
    dU, dG = [], []
    for k in range(d):
        O, Od = G[k], Gd[k]
        # dotted dxi_tilde: product rule of _dxi_tilde_jets_step's two terms (6 terms)
        dxtd = (contract('trs,tWKCa,Caib,sWCb->rWKCi', trs, ttd[k], O,  nu_j[k], **kw)
                + contract('trs,tWCa,KCaib,sWCb->rWKCi', trs, tts[k], Od, nu_j[k], **kw)
                + contract('trs,tWCa,Caib,sWKCb->rWKCi', trs, tts[k], O,  nud_j[k], **kw)
                + contract('trs,rWKCa,Caib,tWCb->sWKCi', trs, mud_j[k], O,  sts[k], **kw)
                + contract('trs,rWCa,KCaib,tWCb->sWKCi', trs, mu_j[k],  Od, sts[k], **kw)
                + contract('trs,rWCa,Caib,tWKCb->sWKCi', trs, mu_j[k],  O,  std[k], **kw))
        wj = np.stack([ww[k], pp[k]])
        dU.append(contract('tWKCa,tWCo->KCao', etad_j[k], zt[k], **kw)
                  + contract('uWKCa,uWo->KCao', dxtd[:s2], wj[:s2], **kw))
        t_sig = trs[:, :, :s2]; t_tau = trs[:, :s2, :]
        dG.append(contract('trs,rWKCa,sWCi,tWCb->KCaib', t_sig, mud_j[k], xi_j[k][:s2],  sts[k], **kw)
                  + contract('trs,rWCa,sWKCi,tWCb->KCaib', t_sig, mu_j[k],  xid_j[k][:s2], sts[k], **kw)
                  + contract('trs,rWCa,sWCi,tWKCb->KCaib', t_sig, mu_j[k],  xi_j[k][:s2],  std[k], **kw)
                  + contract('trs,tWKCa,rWCi,sWCb->KCaib', t_tau, ttd[k], xi_j[k][:s2],  nu_j[k], **kw)
                  + contract('trs,tWCa,rWKCi,sWCb->KCaib', t_tau, tts[k], xid_j[k][:s2], nu_j[k], **kw)
                  + contract('trs,tWCa,rWCi,sWKCb->KCaib', t_tau, tts[k], xi_j[k][:s2],  nud_j[k], **kw)
                  + contract('trs,rWKCa,tWCi,sWCb->KCaib', trs, mud_j[k], dets[k], nu_j[k], **kw)
                  + contract('trs,rWCa,tWKCi,sWCb->KCaib', trs, mu_j[k],  detd[k], nu_j[k], **kw)
                  + contract('trs,rWCa,tWCi,sWKCb->KCaib', trs, mu_j[k],  dets[k], nud_j[k], **kw))
    return dU, dG


if __name__ == '__main__':
  rng = np.random.default_rng(23); np.random.seed(23)
  for C, order in [((), 2), ((2,), 2), ((), 3), ((), 1)]:
    shape, tr, ttr, Wn, Kn = (4, 5, 6, 4), (2, 3, 3, 2), (1, 2, 3, 2, 1), 5, 2
    d = 4; W, K = (Wn,), (Kn,)
    th = t3k.t3_corewise_randn(shape, tr, ttr, stack_shape=C)
    U, G = th
    cf = fvc.t3_corewise_frame(th)
    ww = [rng.standard_normal((Wn, N)) for N in shape]; pp = [rng.standard_normal((Wn, N)) for N in shape]
    zt = [rng.standard_normal((order + 1, Wn) + C + (N,)) for N in shape]     # residual jets (omega^2 applied), no K
    Ud = tuple(rng.standard_normal(K + u.shape) for u in U); Gd = tuple(rng.standard_normal(K + g.shape) for g in G)
    dU, dG = two_hole(U, G, Ud, Gd, ww, pp, zt, order, K, C)
    errs = []
    for kk in range(Kn):
        thk = (tuple(a[kk] for a in Ud), tuple(a[kk] for a in Gd))
        ref = blocksum(sd.t3_probe_corewise_derivatives_transpose(zt, ww, pp, tv.tv_to_t3(cf, thk), order,
                                                                  sum_over_probes=True, chunk_size=None), d, tr, ttr)
        errs.append(relerr([a[kk] for a in dU + dG], list(ref[0] + ref[1])))
    print('C=%-5s order=%d probe_derivatives two-hole vs block-sum, per K:' % (C, order), ['%.1e' % e for e in errs])
