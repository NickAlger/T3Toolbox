# PREDATES the S3b API (2026-10-27): written against the dense `trs` jet kernels (`trs=` arguments, `sd.binomial_combine_tensor`, the `*_trs` twins), which S3b removed (now `tests/_jet_trs_twins.py`); kept as the record of the plan-time evidence, not rewritten. The shipped kernels are pinned by tests/backend/test_curvature.py.
"""Read-only plan check (v3, m26 + n9): the probe_derivatives two-hole with the v3 RECOMPUTE recipe, fed from the
MANIFOLD model's stored sweep (as LocalModel will):
  - xi_j, mu_j reused from the manifold sweep (identical at both frames, F9);
  - nu_c = T nu  (F16: one (r x r)(W r) GEMM per core, no re-sweep through G);
  - NO corewise eta, NO corewise dxi_tilde (the derivative needs only the dotted ones; m26);
  - undotted tildes at the corewise frame: deta_tilde (U z), tau_tilde (through G = L: equals the manifold one),
    sigma_tilde through G with nu_c.
vs the block-sum oracle, orders 0-3, K in {(2,), (2,3)}, C in {(), (2,)}, d in {1, 3, 4}."""
import sys, os; sys.path.insert(0, __import__("os").getcwd()); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import t3toolbox.backend.fv_conversions as fvc
import t3toolbox.backend.tv_operations as tv
import t3toolbox.backend.tt_operations as tto
import t3toolbox.backend.sampling_derivatives as sd
import t3toolbox.backend.probing as pr
import t3toolbox.backend.t3_constructors as t3k
from t3toolbox.backend.contractions import contract
from c14_probe_derivs_twohole import dotted_adj_sweep, blocksum, relerr

def two_hole_v3(frame, msweep, Ud, Gd, ww, pp, zt, order, K, C, nu_mode='T'):
    U, O, L, R = frame
    G = L                                                  # the base point theta = (U, L)
    d = len(U); W = tuple(ww[0].shape[:-1])
    trs = sd.binomial_combine_tensor(order); s2 = min(2, order + 1)
    kw = dict(len_W=len(W), len_C=len(C))
    xi_j, mu_j, nu_m, _eta_m = msweep                       # the MODEL's stored manifold-frame sweep
    if nu_mode == 'resweep':                                  # v3 default: the geometry-agnostic kind re-sweeps nu through G
        nu_j = sd.compute_nu_jets(G, xi_j, trs)
    elif d > 1:
        T = tto.tt_zipper_right_to_left(L, R)[1:]           # T_i = ZZ[i+1]; T_{d-1} = ones
        nu_j = [np.einsum('...ab,t...b->t...a', T[i], nu_m[i]) for i in range(d)]   # nu_c = T nu  (F16)
    else:
        nu_j = nu_m
    dets = sd.compute_deta_tilde_jets(U, zt)
    tts = sd.compute_tau_tilde_jets(G, xi_j, dets, mu_j, trs)
    sts = sd.compute_sigma_tilde_jets(G, xi_j, dets, nu_j, trs)
    xid_j = sd.build_input_jets(pr.compute_xi(Ud, ww), pr.compute_xi(Ud, pp))
    mud_j = sd.compute_sigma_jets(Gd, G, G, xi_j, xid_j, mu_j, trs)
    nud_j = sd.compute_tau_jets(Gd, G, G, xi_j, xid_j, nu_j, trs)
    etad_j = sd.compute_deta_jets(Gd, G, G, mu_j, nu_j, mud_j, nud_j, trs)
    detd = [contract('tWCo,KCio->tWKCi', z, u, **kw) for z, u in zip(zt, Ud)]
    ttd = dotted_adj_sweep(G, Gd, xi_j, xid_j, dets, detd, mu_j, mud_j, tts, trs, K, C, W)
    rv = tto.tt_reverse
    std = dotted_adj_sweep(rv(G), rv(Gd), xi_j[::-1], xid_j[::-1], dets[::-1], detd[::-1],
                           nu_j[::-1], nud_j[::-1], sts[::-1], trs, K, C, W)[::-1]
    dU, dG = [], []
    for k in range(d):
        Oc, Od = G[k], Gd[k]
        dxtd = (contract('trs,tWKCa,Caib,sWCb->rWKCi', trs, ttd[k], Oc, nu_j[k], **kw)
                + contract('trs,tWCa,KCaib,sWCb->rWKCi', trs, tts[k], Od, nu_j[k], **kw)
                + contract('trs,tWCa,Caib,sWKCb->rWKCi', trs, tts[k], Oc, nud_j[k], **kw)
                + contract('trs,rWKCa,Caib,tWCb->sWKCi', trs, mud_j[k], Oc, sts[k], **kw)
                + contract('trs,rWCa,KCaib,tWCb->sWKCi', trs, mu_j[k], Od, sts[k], **kw)
                + contract('trs,rWCa,Caib,tWKCb->sWKCi', trs, mu_j[k], Oc, std[k], **kw))
        wj = np.stack([ww[k], pp[k]])
        dU.append(contract('tWKCa,tWCo->KCao', etad_j[k], zt[k], **kw) + contract('uWKCa,uWo->KCao', dxtd[:s2], wj[:s2], **kw))
        t_sig = trs[:, :, :s2]; t_tau = trs[:, :s2, :]
        dG.append(contract('trs,rWKCa,sWCi,tWCb->KCaib', t_sig, mud_j[k], xi_j[k][:s2], sts[k], **kw)
                  + contract('trs,rWCa,sWKCi,tWCb->KCaib', t_sig, mu_j[k], xid_j[k][:s2], sts[k], **kw)
                  + contract('trs,rWCa,sWCi,tWKCb->KCaib', t_sig, mu_j[k], xi_j[k][:s2], std[k], **kw)
                  + contract('trs,tWKCa,rWCi,sWCb->KCaib', t_tau, ttd[k], xi_j[k][:s2], nu_j[k], **kw)
                  + contract('trs,tWCa,rWKCi,sWCb->KCaib', t_tau, tts[k], xid_j[k][:s2], nu_j[k], **kw)
                  + contract('trs,tWCa,rWCi,sWKCb->KCaib', t_tau, tts[k], xi_j[k][:s2], nud_j[k], **kw)
                  + contract('trs,rWKCa,tWCi,sWCb->KCaib', trs, mud_j[k], dets[k], nu_j[k], **kw)
                  + contract('trs,rWCa,tWKCi,sWCb->KCaib', trs, mu_j[k], detd[k], nu_j[k], **kw)
                  + contract('trs,rWCa,tWCi,sWKCb->KCaib', trs, mu_j[k], dets[k], nud_j[k], **kw))
    return dU, dG

if __name__ == '__main__':
    np.random.seed(29); rng = np.random.default_rng(29)
    cases = [((4, 5, 6, 4), (2, 3, 3, 2), (1, 2, 3, 2, 1), o, (2,), C) for o in (0, 1, 2, 3) for C in ((), (2,))]
    cases += [((4, 5, 6), (2, 3, 2), (1, 2, 2, 1), 2, (2, 3), (2,)), ((5,), (3,), (1, 1), 2, (2,), ())]
    for shape, tr, ttr, order, K, C in cases:
        d, Wn = len(shape), 5
        x = t3k.t3_corewise_randn(shape, tr, ttr, stack_shape=C)
        frame, _ = fvc.t3_orthogonal_representations(x)
        U, O, L, R = frame
        trr = tuple(u.shape[-2] for u in U); ttrr = (1,) + tuple(l.shape[-1] for l in L)
        ww = [rng.standard_normal((Wn, N)) for N in shape]; pp = [rng.standard_normal((Wn, N)) for N in shape]
        zt = [rng.standard_normal((order + 1, Wn) + C + (N,)) for N in shape]
        Ud = tuple(rng.standard_normal(K + u.shape) for u in U); Gd = tuple(rng.standard_normal(K + l.shape) for l in L)
        msweep = sd.tv_precompute_probe_frame_sweep_jets(frame, ww, pp, order)     # the manifold model's sweep
        cf = fvc.t3_corewise_frame((U, L))
        for nu_mode in ('resweep', 'T'):
          dU, dG = two_hole_v3(frame, msweep, Ud, Gd, ww, pp, zt, order, K, C, nu_mode)
          errs = []
          for idx in np.ndindex(*K):
            thk = (tuple(a[idx] for a in Ud), tuple(a[idx] for a in Gd))
            gY = sd.t3_probe_corewise_derivatives_transpose(zt, ww, pp, tv.tv_to_t3(cf, thk), order,
                                                            sum_over_probes=True, chunk_size=None)
            if d == 1:     # d = 1 embedding (tv_operations.py:275-281): Y_U = [U; Udot], Y_G = [Gdot | G] along n
                n = trr[0]; ref = ((gY[0][0][..., :n, :],), (gY[1][0][..., n:, :],))
            else:
                ref = blocksum(gY, d, trr, ttrr)
            errs.append(relerr([a[idx] for a in dU + dG], list(ref[0] + ref[1])))
          print('d=%d order=%d K=%-6s C=%-5s nu=%-7s v3 recipe vs block-sum: max %.1e' % (d, order, K, C, nu_mode, max(errs)))
