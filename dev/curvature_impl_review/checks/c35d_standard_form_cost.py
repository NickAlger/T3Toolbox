# PREDATES the S3b API (2026-10-27): written against the dense `trs` jet kernels (`trs=` arguments, `sd.binomial_combine_tensor`, the `*_trs` twins), which S3b removed (now `tests/_jet_trs_twins.py`); kept as the record of the plan-time evidence, not rewritten. The shipped kernels are pinned by tests/backend/test_curvature.py.
"""Read-only plan check (v5, P5): the probe_derivatives exact/GN cost with every dotted kernel in the HOUSE-STANDARD form
of its GN counterpart: (a) dotted adjoint sweep = the product rule of `_adj_tilde_step` (prop as the two-term reverse
recurrence, src as the order scan) -- three calls of the production step per core (K squeezed so the production step
applies; this is the timing proxy, equal to the dense prototype numerically); (b) dotted dx~i as production
compute_dxi_tilde_jets (dense trs, like GN); (c) the TT assembly product-rule FACTORED (c35c), dense trs like GN.
Checks (a) and the whole two-hole == the dense prototype (c22.two_hole_v3, 'resweep'), then times the standard two-hole against the GN matvec."""
import sys, os, time; sys.path.insert(0, __import__("os").getcwd()); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import t3toolbox.backend.fv_conversions as fvc, t3toolbox.backend.sampling_derivatives as sd
import t3toolbox.backend.t3_constructors as t3k, t3toolbox.backend.tv_operations as tv, t3toolbox.backend.tt_operations as tto
import t3toolbox.backend.probing as pr
from c14_probe_derivs_twohole import dotted_adj_sweep
from c22_probe_derivs_v3_recipe import two_hole_v3
from t3toolbox.backend.contractions import contract
np.random.seed(0); rng = np.random.default_rng(0)
def tm(f, n=2):
    f(); t0 = time.perf_counter()
    for _ in range(n):
        out = f()
    return (time.perf_counter() - t0) / n, out

def dotted_adj_sweep_std(P, Pd, xi, xid, deta, detd, edge, edged, carries, trs):
    """K squeezed (K = ()): carry' = step(cd, P, xi, deta, ed) + step(c, Pd, xi, deta, e) + step(c, P, xid, detd, e)."""
    trs_r = np.moveaxis(trs, 1, 0)
    cd = np.zeros_like(carries[0]); out = []
    for k in range(len(P)):
        out.append(cd)
        cd = (sd._adj_tilde_step(cd, P[k], xi[k], deta[k], edged[k], trs_r, False)
              + sd._adj_tilde_step(carries[k], Pd[k], xi[k], deta[k], edge[k], trs_r, False)
              + sd._adj_tilde_step(carries[k], P[k], xid[k], detd[k], edge[k], trs_r, False))
    return out

for shape, n, r, order, W in [((20,) * 5, 8, 8, 2, 300), ((24,) * 4, 16, 16, 3, 2000)]:
    d = len(shape); tr = (n,) * d; ttr = (1,) + (r,) * (d - 1) + (1,)
    x = t3k.t3_corewise_randn(shape, tr, ttr); frame, var = fvc.t3_orthogonal_representations(x); U, O, L, R = frame; G = L
    ww = [rng.standard_normal((W, N)) for N in shape]; pp = [rng.standard_normal((W, N)) for N in shape]
    zt = [rng.standard_normal((order + 1, W, N)) for N in shape]
    msw = sd.tv_precompute_probe_frame_sweep_jets(frame, ww, pp, order); trs = sd.binomial_combine_tensor(order); s2 = min(2, order + 1)
    v = tv.tv_orthogonal_gauge_projection(frame, (tuple(rng.standard_normal(a.shape) for a in var[0]), tuple(rng.standard_normal(a.shape) for a in var[1])))
    Ud = tuple(rng.standard_normal(u.shape) for u in U); Gd = tuple(rng.standard_normal(l.shape) for l in L)   # K squeezed
    kw0 = dict(len_W=1, len_C=0)
    t_gnf, Jv = tm(lambda: sd.tv_probe_jacobian_derivatives_from_sweep(v, ww, pp, frame, msw, order))
    t_gnt, _ = tm(lambda: sd.tv_probe_transpose_derivatives_from_sweep(Jv, ww, pp, frame, msw, order, sum_over_probes=True, chunk_size=None))
    t_gn = t_gnf + t_gnt
    def two_hole_std():
        xi_j, mu_j = msw[0], msw[1]
        nu_j = sd.compute_nu_jets(G, xi_j, trs)
        dets = sd.compute_deta_tilde_jets(U, zt)
        tts = sd.compute_tau_tilde_jets(G, xi_j, dets, mu_j, trs); sts = sd.compute_sigma_tilde_jets(G, xi_j, dets, nu_j, trs)
        xid_j = sd.build_input_jets(pr.compute_xi(Ud, ww), pr.compute_xi(Ud, pp))
        mud_j = sd.compute_sigma_jets(Gd, G, G, xi_j, xid_j, mu_j, trs)
        nud_j = sd.compute_tau_jets(Gd, G, G, xi_j, xid_j, nu_j, trs)
        etad_j = sd.compute_deta_jets(Gd, G, G, mu_j, nu_j, mud_j, nud_j, trs)
        detd = [contract('tWCo,Cio->tWCi', z, u, **kw0) for z, u in zip(zt, Ud)]
        rv = tto.tt_reverse
        ttd = dotted_adj_sweep_std(G, Gd, xi_j, xid_j, dets, detd, mu_j, mud_j, tts, trs)
        std = dotted_adj_sweep_std(rv(G), rv(Gd), xi_j[::-1], xid_j[::-1], dets[::-1], detd[::-1], nu_j[::-1], nud_j[::-1], sts[::-1], trs)[::-1]
        # dotted dx~i: the product rule of production compute_dxi_tilde_jets (dense trs, as GN)
        dxtd = [contract('trs,tWa,aib,sWb->rWi', trs, ttd[k], G[k], nu_j[k]) + contract('trs,tWa,aib,sWb->rWi', trs, tts[k], Gd[k], nu_j[k])
                + contract('trs,tWa,aib,sWb->rWi', trs, tts[k], G[k], nud_j[k]) + contract('trs,rWa,aib,tWb->sWi', trs, mud_j[k], G[k], sts[k])
                + contract('trs,rWa,aib,tWb->sWi', trs, mu_j[k], Gd[k], sts[k]) + contract('trs,rWa,aib,tWb->sWi', trs, mu_j[k], G[k], std[k])
                for k in range(d)]      # the 6-term product rule of compute_dxi_tilde_jets (dense trs, as GN)
        dU, dG = [], []
        t_sig = trs[:, :, :s2]; t_tau = trs[:, :s2, :]
        for k in range(d):
            wj = np.stack([ww[k], pp[k]])
            dU.append(contract('tWa,tWo->ao', etad_j[k], zt[k]) + contract('uWa,uWo->ao', dxtd[k][:s2], wj[:s2]))
            mu, mud, xi, xid, nu, nud = mu_j[k], mud_j[k], xi_j[k][:s2], xid_j[k][:s2], nu_j[k], nud_j[k]
            P = contract('trs,rWa,sWi->tWai', t_sig, mu, xi); Pd = contract('trs,rWa,sWi->tWai', t_sig, mud, xi) + contract('trs,rWa,sWi->tWai', t_sig, mu, xid)
            Q = contract('trs,tWa,rWi->sWai', t_tau, tts[k], xi); Qd = contract('trs,tWa,rWi->sWai', t_tau, ttd[k], xi) + contract('trs,tWa,rWi->sWai', t_tau, tts[k], xid)
            B = contract('trs,rWa,tWi->sWai', trs, mu, dets[k]); Bd = contract('trs,rWa,tWi->sWai', trs, mud, dets[k]) + contract('trs,rWa,tWi->sWai', trs, mu, detd[k])
            dG.append(contract('tWai,tWb->aib', Pd, sts[k]) + contract('tWai,tWb->aib', P, std[k])
                      + contract('sWai,sWb->aib', Qd, nu) + contract('sWai,sWb->aib', Q, nud)
                      + contract('sWai,sWb->aib', Bd, nu) + contract('sWai,sWb->aib', B, nud))
        return dU, dG, (G, Gd, xi_j, xid_j, dets, detd, mu_j, mud_j, tts)
    t_std, (dU, dG, args) = tm(two_hole_std)
    # (a) equality of the standard dotted adjoint sweep with the dense prototype (K = (1,) there)
    Gk, Gdk, xi_j, xid_j, dets, detd, mu_j, mud_j, tts = args
    a = dotted_adj_sweep_std(*args, trs)
    b = dotted_adj_sweep(Gk, tuple(g[None] for g in Gdk), xi_j, [x[:, :, None] for x in xid_j], dets, [t[:, :, None] for t in detd],
                         mu_j, [m[:, :, None] for m in mud_j], tts, trs, (1,), (), (W,))
    err = max(np.linalg.norm(x - y[:, :, 0]) / max(np.linalg.norm(y), 1e-300) for x, y in zip(a, b))
    pU, pG = two_hole_v3(frame, msw, tuple(u[None] for u in Ud), tuple(g[None] for g in Gd), ww, pp, zt, order, (1,), (), 'resweep')
    err2 = max(np.linalg.norm(a_ - b_[0]) / np.linalg.norm(b_) for a_, b_ in zip(list(dU) + list(dG), list(pU) + list(pG)))
    print(f"d={d} n=r={n} W={W} order={order}: standard-form dotted adjoint == dense prototype: {err:.1e}; whole two-hole {err2:.1e}; "
          f"GN {t_gn:.3f}s, standard two-hole {t_std:.3f}s -> exact/GN {(t_gn + t_std) / t_gn:.2f}x")
