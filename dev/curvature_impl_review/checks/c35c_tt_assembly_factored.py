# PREDATES the S3b API (2026-10-27): written against the dense `trs` jet kernels (`trs=` arguments, `sd.binomial_combine_tensor`, the `*_trs` twins), which S3b removed (now `tests/_jet_trs_twins.py`); kept as the record of the plan-time evidence, not rewritten. The shipped kernels are pinned by tests/backend/test_curvature.py.
"""Read-only plan check (v5): the dotted TT assembly is the two-hole's hot spot (c35b: 31-55%). Product-rule
factoring: for each of the three trilinear groups f(a, b, c), form the pair intermediate P = (a.b) and its dotted twin
Pd = (a_dot.b + a.b_dot) once, then dG = Pd.c + P.c_dot -- 6 W-summed final contractions instead of 9. Equal to the
9-term form? Faster? (numpy CPU, recompute-per-matvec; P is undotted and could also be cached.)"""
import sys, os, time; sys.path.insert(0, __import__("os").getcwd()); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from t3toolbox.backend.contractions import contract
import t3toolbox.backend.sampling_derivatives as sd
rng = np.random.default_rng(0)
def tm(f, n=2):
    f(); t0 = time.perf_counter()
    for _ in range(n):
        out = f()
    return (time.perf_counter() - t0) / n, out
for (n, r, order, W) in [(8, 8, 2, 300), (16, 16, 3, 2000)]:
    o1 = order + 1; s2 = min(2, o1); trs = sd.binomial_combine_tensor(order); kw = dict(len_W=1, len_C=0)
    R = lambda *s: rng.standard_normal(s)
    mu, mud = R(o1, W, r), R(o1, W, 1, r); xi, xid = R(2, W, n), R(2, W, 1, n)
    nu, nud = R(o1, W, r), R(o1, W, 1, r); sts, std = R(o1, W, r), R(o1, W, 1, r)
    tts, ttd = R(o1, W, r), R(o1, W, 1, r); dets, detd = R(o1, W, n), R(o1, W, 1, n)
    t_sig = trs[:, :, :s2]; t_tau = trs[:, :s2, :]
    def nine():
        return (contract('trs,rWKCa,sWCi,tWCb->KCaib', t_sig, mud, xi[:s2], sts, **kw)
                + contract('trs,rWCa,sWKCi,tWCb->KCaib', t_sig, mu, xid[:s2], sts, **kw)
                + contract('trs,rWCa,sWCi,tWKCb->KCaib', t_sig, mu, xi[:s2], std, **kw)
                + contract('trs,tWKCa,rWCi,sWCb->KCaib', t_tau, ttd, xi[:s2], nu, **kw)
                + contract('trs,tWCa,rWKCi,sWCb->KCaib', t_tau, tts, xid[:s2], nu, **kw)
                + contract('trs,tWCa,rWCi,sWKCb->KCaib', t_tau, tts, xi[:s2], nud, **kw)
                + contract('trs,rWKCa,tWCi,sWCb->KCaib', trs, mud, dets, nu, **kw)
                + contract('trs,rWCa,tWKCi,sWCb->KCaib', trs, mu, detd, nu, **kw)
                + contract('trs,rWCa,tWCi,sWKCb->KCaib', trs, mu, dets, nud, **kw))
    def six():
        # sigma group: P_t = sum_{r,s} trs[t,r,s] mu^r xi^s  (over a, i), then with sts^t over (t, W)
        P = contract('trs,rWa,sWi->tWai', t_sig, mu, xi[:s2])
        Pd = contract('trs,rWKa,sWi->tWKai', t_sig, mud, xi[:s2]) + contract('trs,rWa,sWKi->tWKai', t_sig, mu, xid[:s2])
        out = contract('tWKai,tWb->Kaib', Pd, sts) + contract('tWai,tWKb->Kaib', P, std)
        # tau group: Q_s = sum_{t,r} trs[t,r,s] tts^t xi^r (over a, i), then with nu^s
        Q = contract('trs,tWa,rWi->sWai', t_tau, tts, xi[:s2])
        Qd = contract('trs,tWKa,rWi->sWKai', t_tau, ttd, xi[:s2]) + contract('trs,tWa,rWKi->sWKai', t_tau, tts, xid[:s2])
        out = out + contract('sWKai,sWb->Kaib', Qd, nu) + contract('sWai,sWKb->Kaib', Q, nud)
        # deta group: B_s = sum_{t,r} trs[t,r,s] mu^r dets^t, then with nu^s
        B = contract('trs,rWa,tWi->sWai', trs, mu, dets)
        Bd = contract('trs,rWKa,tWi->sWKai', trs, mud, dets) + contract('trs,rWa,tWKi->sWKai', trs, mu, detd)
        return out + contract('sWKai,sWb->Kaib', Bd, nu) + contract('sWai,sWKb->Kaib', B, nud)
    t9, a = tm(nine); t6, b = tm(six)
    print(f'n=r={n} W={W} order={order}: 9-term {t9:.3f}s, factored {t6:.3f}s ({t9 / t6:.2f}x faster); rel diff {np.linalg.norm(a - b) / np.linalg.norm(a):.1e}')
