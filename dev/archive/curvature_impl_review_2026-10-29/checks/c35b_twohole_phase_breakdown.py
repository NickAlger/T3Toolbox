# PREDATES the S3b API (2026-10-27): written against the dense `trs` jet kernels (`trs=` arguments, `sd.binomial_combine_tensor`, the `*_trs` twins), which S3b removed (now `tests/_jet_trs_twins.py`); kept as the record of the plan-time evidence, not rewritten. The shipped kernels are pinned by tests/backend/test_curvature.py.
"""Read-only plan check (v5): per-phase wall time of the prototype probe_derivatives two-hole (c22 recipe) vs the
GN matvec's forward and transpose, at a mid and a larger size (numpy CPU)."""
import sys, os, time; sys.path.insert(0, __import__("os").getcwd()); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import t3toolbox.backend.fv_conversions as fvc, t3toolbox.backend.sampling_derivatives as sd
import t3toolbox.backend.t3_constructors as t3k, t3toolbox.backend.tv_operations as tv
import t3toolbox.backend.probing as pr
from c14_probe_derivs_twohole import dotted_adj_sweep
from t3toolbox.backend.contractions import contract
import t3toolbox.backend.tt_operations as tto
np.random.seed(0); rng = np.random.default_rng(0)
def tm(f, n=2):
    f(); t0 = time.perf_counter()
    for _ in range(n):
        out = f()
    return (time.perf_counter() - t0) / n, out
for shape, n, r, order, W in [((20,) * 5, 8, 8, 2, 300), ((24,) * 4, 16, 16, 3, 2000)]:
    d = len(shape); tr = (n,) * d; ttr = (1,) + (r,) * (d - 1) + (1,)
    x = t3k.t3_corewise_randn(shape, tr, ttr); frame, var = fvc.t3_orthogonal_representations(x); U, O, L, R = frame; G = L
    ww = [rng.standard_normal((W, N)) for N in shape]; pp = [rng.standard_normal((W, N)) for N in shape]
    zt = [rng.standard_normal((order + 1, W, N)) for N in shape]
    msw = sd.tv_precompute_probe_frame_sweep_jets(frame, ww, pp, order); trs = sd.binomial_combine_tensor(order); s2 = min(2, order + 1)
    v = tv.tv_orthogonal_gauge_projection(frame, (tuple(rng.standard_normal(a.shape) for a in var[0]), tuple(rng.standard_normal(a.shape) for a in var[1])))
    Ud = tuple(rng.standard_normal((1,) + u.shape) for u in U); Gd = tuple(rng.standard_normal((1,) + l.shape) for l in L)
    kw = dict(len_W=1, len_C=0); K, C, Wt = (1,), (), (W,)
    T = {}
    T['GN forward'], Jv = tm(lambda: sd.tv_probe_jacobian_derivatives_from_sweep(v, ww, pp, frame, msw, order))
    T['GN transpose'], _ = tm(lambda: sd.tv_probe_transpose_derivatives_from_sweep(Jv, ww, pp, frame, msw, order, sum_over_probes=True, chunk_size=None))
    xi_j, mu_j = msw[0], msw[1]
    T['nu re-sweep'], nu_j = tm(lambda: sd.compute_nu_jets(G, xi_j, trs))
    def undotted():
        dets = sd.compute_deta_tilde_jets(U, zt)
        return dets, sd.compute_tau_tilde_jets(G, xi_j, dets, mu_j, trs), sd.compute_sigma_tilde_jets(G, xi_j, dets, nu_j, trs)
    T['undotted tildes'], (dets, tts, sts) = tm(undotted)
    def dfwd():
        xid_j = sd.build_input_jets(pr.compute_xi(Ud, ww), pr.compute_xi(Ud, pp))
        mud_j = sd.compute_sigma_jets(Gd, G, G, xi_j, xid_j, mu_j, trs)
        nud_j = sd.compute_tau_jets(Gd, G, G, xi_j, xid_j, nu_j, trs)
        return xid_j, mud_j, nud_j, sd.compute_deta_jets(Gd, G, G, mu_j, nu_j, mud_j, nud_j, trs)
    T['dotted forward'], (xid_j, mud_j, nud_j, etad_j) = tm(dfwd)
    detd = [contract('tWCo,KCio->tWKCi', z, u, **kw) for z, u in zip(zt, Ud)]
    rv = tto.tt_reverse
    def dadj():
        ttd = dotted_adj_sweep(G, Gd, xi_j, xid_j, dets, detd, mu_j, mud_j, tts, trs, K, C, Wt)
        std = dotted_adj_sweep(rv(G), rv(Gd), xi_j[::-1], xid_j[::-1], dets[::-1], detd[::-1], nu_j[::-1], nud_j[::-1], sts[::-1], trs, K, C, Wt)[::-1]
        return ttd, std
    T['dotted adjoint sweeps'], (ttd, std) = tm(dadj)
    def ddxi():
        return [(contract('trs,tWKCa,Caib,sWCb->rWKCi', trs, ttd[k], G[k], nu_j[k], **kw) + contract('trs,tWCa,KCaib,sWCb->rWKCi', trs, tts[k], Gd[k], nu_j[k], **kw)
                 + contract('trs,tWCa,Caib,sWKCb->rWKCi', trs, tts[k], G[k], nud_j[k], **kw) + contract('trs,rWKCa,Caib,tWCb->sWKCi', trs, mud_j[k], G[k], sts[k], **kw)
                 + contract('trs,rWCa,KCaib,tWCb->sWKCi', trs, mu_j[k], Gd[k], sts[k], **kw) + contract('trs,rWCa,Caib,tWKCb->sWKCi', trs, mu_j[k], G[k], std[k], **kw)) for k in range(d)]
    T['dotted dxi~'], dxtd = tm(ddxi)
    def tucker():
        return [contract('tWKCa,tWCo->KCao', etad_j[k], zt[k], **kw) + contract('uWKCa,uWo->KCao', dxtd[k][:s2], np.stack([ww[k], pp[k]])[:s2], **kw) for k in range(d)]
    T['Tucker assembly (2 terms)'], _ = tm(tucker)
    t_sig = trs[:, :, :s2]; t_tau = trs[:, :s2, :]
    def tt():
        return [(contract('trs,rWKCa,sWCi,tWCb->KCaib', t_sig, mud_j[k], xi_j[k][:s2], sts[k], **kw) + contract('trs,rWCa,sWKCi,tWCb->KCaib', t_sig, mu_j[k], xid_j[k][:s2], sts[k], **kw)
                + contract('trs,rWCa,sWCi,tWKCb->KCaib', t_sig, mu_j[k], xi_j[k][:s2], std[k], **kw) + contract('trs,tWKCa,rWCi,sWCb->KCaib', t_tau, ttd[k], xi_j[k][:s2], nu_j[k], **kw)
                + contract('trs,tWCa,rWKCi,sWCb->KCaib', t_tau, tts[k], xid_j[k][:s2], nu_j[k], **kw) + contract('trs,tWCa,rWCi,sWKCb->KCaib', t_tau, tts[k], xi_j[k][:s2], nud_j[k], **kw)
                + contract('trs,rWKCa,tWCi,sWCb->KCaib', trs, mud_j[k], dets[k], nu_j[k], **kw) + contract('trs,rWCa,tWKCi,sWCb->KCaib', trs, mu_j[k], detd[k], nu_j[k], **kw)
                + contract('trs,rWCa,tWCi,sWKCb->KCaib', trs, mu_j[k], dets[k], nud_j[k], **kw)) for k in range(d)]
    T['TT assembly (9 terms)'], _ = tm(tt)
    gn = T['GN forward'] + T['GN transpose']; two = sum(v for k, v in T.items() if not k.startswith('GN'))
    print(f"d={d} n=r={n} W={W} order={order}: GN {gn:.3f}s (fwd {T['GN forward']:.3f}, T {T['GN transpose']:.3f}); two-hole {two:.3f}s; exact/GN {(gn + two) / gn:.2f}x")
    for k, val in T.items():
        if not k.startswith('GN'):
            print(f"    {k:26s} {val:.3f}s  {100 * val / two:4.0f}%  ({val / T['GN transpose']:.2f}x the GN transpose)")
